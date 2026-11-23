// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Compute dispatches of the NVN replay (Op::Dispatch, direct and indirect).
//
// A dispatch runs outside render passes. The compute pipeline comes from the pipeline cache
// (nvn_pipeline_cache.h, GetCompute); its descriptors are built by DrawContext::BindDescriptors
// like a draw's, with two compute specifics:
//   * the driver bank (c0) is laid out per shader from the control section (push constants, the
//     uniform and storage buffer entry tables, the handle tables): FillComputeBank;
//   * the shader's texture handle words live in that bank, so the pipeline cache's texture oracle
//     classifies them with the program's tables (ComputeOracle).

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <set>
#include <vector>
#include <unordered_set>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "common/settings.h"

#include "video_core/upscaler.h"
#include "video_core/shader_log.h"
#include "video_core/nvn/shader/nvn_compute_bank.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/nvn_draw_probe.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_backing_log.h"

namespace Vulkan::Nvn {

namespace {


/// Answers the pipeline cache's "what texture is behind this handle word" for a compute program.
class ComputeOracle final : public TextureOracle, public InputsProvider {
public:
    ComputeOracle(const DrawContext& context_, const VC::ComputeHandleTables& tables_)
        : context{context_}, tables{tables_} {}

    [[nodiscard]] TextureInfo Query(u32 stage, const VC::NvnHandleRecord& record) const override {
        return context.QueryTexture(stage, record, &tables);
    }
    [[nodiscard]] std::string DescribeInputs(const PipelineBase* pipeline) const override {
        return context.DescribeInputs(pipeline, true);
    }
    [[nodiscard]] std::unique_ptr<TextureOracle> Freeze(u32 stage_mask) const override {
        return context.FreezeOracle(stage_mask, &tables);
    }

private:
    const DrawContext& context;
    VC::ComputeHandleTables tables;
};

static_assert(sizeof(BufferBinding) == sizeof(VC::BufferRange));

} // namespace

void DrawContext::FillComputeBank(const VC::ComputeDriverLayout& layout, u8* out) const {
    const StageBindings& bound = stages[STAGE_COMPUTE];
    std::array<VC::BufferRange, 32> uniform;
    std::array<VC::BufferRange, 32> storage;
    for (size_t i = 0; i < uniform.size(); ++i) {
        uniform[i] = {bound.uniform[i].address, bound.uniform[i].size};
        storage[i] = {bound.storage[i].address, bound.storage[i].size};
    }
    const VC::ComputeBankInputs inputs{
        .uniform_buffers = uniform,
        .storage_buffers = storage,
        .push_constants = push_constants[1],
        .textures = bound.combined,
        .images = bound.image,
        .group_counts = compute_groups,
    };
    VC::FillComputeDriverBank(layout, inputs, std::span<u8>{out, DRIVER_BANK_SIZE});
}

void DrawContext::Dispatch(const VC::DispatchPayload& p) {
    ++counters.dispatches;
    const bool indirect = p.indirect != 0;
    if (indirect) {
        ++counters.dispatches_indirect;
    }
    const auto skipped = [&] {
        ++counters.dispatches_skipped;
    };
    if (!indirect && (p.x == 0 || p.y == 0 || p.z == 0)) {
        return; // a dispatch of no groups is a no-op, not an error
    }
    if (!indirect) {
        const std::array<u32, 3> limit = impl.device.GetMaxComputeWorkGroupCount();
        if (p.x > limit[0] || p.y > limit[1] || p.z > limit[2]) {
            skipped();
            host->HostUnsupported("Dispatch",
                                  fmt::format("{}x{}x{} groups exceed the device limit {}x{}x{}",
                                              p.x, p.y, p.z, limit[0], limit[1], limit[2]));
            return;
        }
    }

    // The compute program of the stage-5 binding.
    const StageBindings& bound = stages[STAGE_COMPUTE];
    const auto program_it = impl.programs.find(bound.program_serial);
    const auto hash_it = resources.program_hashes.find(bound.program_serial);
    if (bound.program_serial == 0 || program_it == impl.programs.end() ||
        hash_it == resources.program_hashes.end() || hash_it->second[STAGE_COMPUTE] == 0) {
        skipped();
        host->HostUnsupported("Dispatch", "no compute program is bound");
        return;
    }
    const VC::ProgramDesc& program = *program_it->second;
    std::string error;
    const std::optional<VC::NvnControl> control =
        VC::ParseControl(program.control[STAGE_COMPUTE], error);
    if (!control || !control->has_compute_counts) {
        skipped();
        host->HostUnsupported("Dispatch", fmt::format("compute control section: {}",
                                                      control ? "no driver bank counts" : error));
        return;
    }
    if (VideoCommon::Nvn::TimestampTraceEnabled()) {
        resources.TraceDispatch(control->thread_group[0] == 128);
    }

    const Settings::Upscaler requested = VideoCore::Upscaler::Effective();
    const auto scale = impl.RescaleFactor();
    static Common::BootCounter scale_logged;
    if (scale_logged.First()) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] active scene scale: up={} shift={} ({}x), setting={}, info.active={}",
                 scale.up, scale.shift, scale.UpFactor(),
                 static_cast<u32>(Settings::values.resolution_setup.GetValue()),
                 Settings::values.resolution_info.active);
    }
    const auto backend = SelectedUpscaleBackend(requested);
    const bool replacement_wanted = VideoCore::Upscaler::SelectHlePassPrecedence(
                                        backend, requested, scale.Active(), false)
                                        .replace_game_network;
    const auto log_skip = [](u64 hash, std::string_view reason, const auto& details) {
        static Common::BootCounter boot;
        static std::set<std::pair<u64, std::string_view>> logged;
        if (boot.First()) {
            logged.clear();
        }
        const auto key = std::pair{hash, reason};
        if (!logged.contains(key)) {
            LOG_INFO(Render_Vulkan, "[nvn] NDSR replacement skipped ({}): {}", reason, details());
            logged.emplace(key);
        }
    };
    if (!replacement_wanted) {
        log_skip(0, "policy", [&] {
            return fmt::format("requested={} backend={} scale={}x saved upscaler={} AA={}",
                               static_cast<u32>(requested), static_cast<u32>(backend),
                               scale.UpFactor(),
                               static_cast<u32>(VideoCore::Upscaler::Effective()),
                               static_cast<u32>(Settings::values.anti_aliasing.GetValue()));
        });
    }
    if (replacement_wanted) {
        // The compute chain behind a game's upscaling network: kept only while one may be replaced.
        NoteComputeChain();
    }
    static const bool dlss_trace = std::getenv("EDEN_NX2_DLSS_TRACE") != nullptr;
    if (dlss_trace) {
        static Common::BootCounter boot;
        static std::unordered_set<u64> traced_programs;
        if (boot.First()) {
            traced_programs.clear();
        }
        const u64 hash = hash_it->second[STAGE_COMPUTE];
        if (traced_programs.insert(hash).second) {
            u32 input_id{};
            for (const u64 handle : bound.combined) {
                if (handle != 0) {
                    input_id = VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle));
                    break;
                }
            }
            u32 output_id{};
            for (const u64 handle : bound.image) {
                if (handle != 0) {
                    output_id = VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle));
                    break;
                }
            }
            const auto input = TextureOfId(input_id);
            const auto output = TextureOfId(output_id);
            LOG_INFO(Render_Vulkan,
                     "[dlss] compute trace hash={:016x} groups={}x{}x{} input={}x{} fmt={} output={}x{} fmt={}",
                     hash, compute_groups[0], compute_groups[1], compute_groups[2],
                     input ? input->desc.width : 0, input ? input->desc.height : 0,
                     input ? static_cast<u32>(input->host.format) : 0,
                     output ? output->desc.width : 0, output ? output->desc.height : 0,
                     output ? static_cast<u32>(output->host.format) : 0);
        }
    }
    const bool ndsr_shape = VideoCore::Upscaler::IsNdsrDispatchShape(
        control->thread_group, control->compute.texture_count, control->compute.image_count,
        push_constants[1].size());
    const u64 program_hash = hash_it->second[STAGE_COMPUTE];
    if (replacement_wanted && !ndsr_shape) {
        log_skip(program_hash, "dispatch shape", [&] {
            return fmt::format("hash={:016x} threads={}x{}x{} textures={} images={} push_bytes={}",
                               program_hash, control->thread_group[0], control->thread_group[1],
                               control->thread_group[2], control->compute.texture_count,
                               control->compute.image_count, push_constants[1].size());
        });
    }
    const bool tensor = ndsr_shape && !bound.combined.empty() &&
                        ComputeProgramUsesTensorMath(program_hash, program.code[STAGE_COMPUTE],
                                                     control->inst_bytes);
    if (replacement_wanted && ndsr_shape && bound.combined.empty()) {
        log_skip(program_hash, "no sampled texture", [&] { return fmt::format("hash={:016x}", program_hash); });
    } else if (replacement_wanted && ndsr_shape && !tensor) {
        log_skip(program_hash, "no tensor math", [&] {
            return fmt::format("hash={:016x} instruction_bytes={}", program_hash,
                               control->inst_bytes);
        });
    }
    if (replacement_wanted && tensor) {
        std::array<u32, 8> words{};
        std::memcpy(words.data(), push_constants[1].data(), sizeof(words));
        const u32 input_id = VideoCommon::Nvn::TextureIdOf(static_cast<u32>(bound.combined[0]));
        u32 output_id{};
        const auto input = TextureOfId(input_id);
        std::shared_ptr<Texture> output;
        VideoCore::Upscaler::Rect input_rect{}, output_rect{};
        if (!input) {
            log_skip(program_hash, "texture lookup", [&] {
                return fmt::format("hash={:016x} input missing", program_hash);
            });
        } else {
            bool bounds_valid = false;
            for (const u64 handle : bound.image) {
                if (handle == 0) {
                    continue;
                }
                const u32 candidate_id = VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle));
                auto candidate = TextureOfId(candidate_id);
                if (!candidate) {
                    continue;
                }
                VideoCore::Upscaler::Rect candidate_input{}, candidate_output{};
                if (!VideoCore::Upscaler::DecodeNdsrRects(
                        words, candidate_input, candidate_output, input->desc.width,
                        input->desc.height, candidate->desc.width, candidate->desc.height)) {
                    continue;
                }
                bounds_valid = true;
                input_rect = candidate_input;
                output_rect = candidate_output;
                if (!VideoCore::Upscaler::IsUpscaleNetworkDispatch(
                        control->thread_group, control->compute.texture_count,
                        control->compute.image_count, push_constants[1].size(), true,
                        candidate_input, candidate_output)) {
                    continue;
                }
                output_id = candidate_id;
                output = std::move(candidate);
                break;
            }
            if (!output) {
                log_skip(program_hash, bounds_valid ? "not an upscale" : "rectangle bounds", [&] {
                    return fmt::format(
                        "hash={:016x} words={},{},{},{},{},{},{},{} input={}x{}; no bound output "
                        "slot accepts the network rectangle",
                        program_hash, words[0], words[1], words[2], words[3], words[4], words[5],
                        words[6], words[7], input->desc.width, input->desc.height);
                });
            } else if (UpscaleNetwork(input_rect, output_rect, output_id)) {
                if (VideoCommon::ShaderLog::Enabled()) {
                    // The game's own program is never translated or run: log its bytes and the
                    // inputs it would have read.
                    const ComputeOracle log_oracle{*this, control->compute.HandleTables()};
                    ComputeRequest log_request;
                    log_request.program = {program_hash, &program.control[STAGE_COMPUTE],
                                           &program.code[STAGE_COMPUTE]};
                    log_request.inputs = &log_oracle;
                    resources.Cache().LogComputeFallback(
                        log_request, "upscaling network dispatch replaced by the HLE upscale stage: "
                                     "the game's program is not translated or run");
                }
                skipped();
                return;
            } else {
                log_skip(program_hash, "upscale stage", [&] {
                    return fmt::format("hash={:016x} input={}x{} output={}x{} slot={}",
                                       program_hash, input_rect.width, input_rect.height,
                                       output_rect.width, output_rect.height, output_id);
                });
            }
        }
    }

    const ComputeOracle oracle{*this, control->compute.HandleTables()};
    ComputeRequest request;
    request.program = {hash_it->second[STAGE_COMPUTE], &program.control[STAGE_COMPUTE],
                       &program.code[STAGE_COMPUTE]};
    request.oracle = &oracle;
    request.inputs = &oracle;
    // Only the first lookup of a program is timed (it is the one that translates and compiles).
    const bool first_dispatch = !resources.compute_programs.contains(request.program.hash);
    const auto compile_start =
        first_dispatch ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // A dispatch is never skipped while its pipeline compiles (use_asynchronous_shaders): the
    // lookup waits for a compile in flight, as the emulated path's ComputePipeline::Configure
    // waits for its build. A skipped dispatch leaves the buffers and images it writes stale for
    // every later draw (one-time setup, GPU-driven culling), not one frame of geometry.
    const std::shared_ptr<ComputePipeline> compute = resources.Cache().GetCompute(request);
    if (first_dispatch) {
        const auto compile_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - compile_start)
                                    .count();
        resources.compute_programs.insert(request.program.hash);
        // Once per distinct program: what DK dispatches and what its first lookup cost (a
        // translation and pipeline compile unless the disk cache had it).
        LOG_INFO(Render_Vulkan,
                 "[nvn] compute program {:016x}: group {}x{}x{}, {} push words, {} B shared, first "
                 "dispatch {} ms ({})",
                 request.program.hash, control->thread_group[0], control->thread_group[1],
                 control->thread_group[2], control->compute.push_constant_words,
                 control->shared_memory_bytes, compile_ms, compute ? "ok" : "rejected");
    }
    if (!compute) {
        // The cache counted and logged why (unsupported rows, `shader:`).
        skipped();
        return;
    }
    if (last_compute_kept != compute.get()) {
        host->HostFlight().keepalive.push_back(compute);
        last_compute_kept = compute.get();
    }
    // nvn_draws / nvn_draw_skip (video_core/nvn_draw_probe.h): a skipped direct dispatch is
    // recorded with zero groups below, a skipped indirect one is not recorded.
    bool probe_skip = false;
    const bool probe_active = VideoCommon::NvnDrawProbe::Active();
    const VideoCommon::NvnDrawProbe::DispatchInfo probe_info{
        .hash = program_hash,
        .groups = indirect ? std::array<u32, 3>{} : std::array<u32, 3>{p.x, p.y, p.z},
        .indirect = indirect};
    if (probe_active) [[unlikely]] {
        probe_skip = VideoCommon::NvnDrawProbe::OnDispatch(probe_info);
        if (probe_skip && indirect) {
            skipped();
            return;
        }
    }

    // Indirect group counts are read by the GPU: resolve them first, so a dispatch that is
    // skipped here never marks the storage ranges it would have written.
    BufferRef arguments;
    if (indirect &&
        !ResolveBuffer(p.indirect, 3 * sizeof(u32), false, arguments, ShadowPolicy::Defer)) {
        skipped();
        host->HostFailed("Dispatch", fmt::format("indirect arguments at {:#x} are not in a "
                                                 "memory pool", p.indirect));
        return;
    }
    if (indirect) {
        ResolveIndirectSource(arguments, 3 * sizeof(u32));
    }

    if (Debug::Get().op_log && Debug::FrameSelected(impl.presents_recorded + 1)) {
        std::string line = fmt::format("[nvn] dispatch f{} prog={:016x} {}x{}x{} ubo=[", impl.presents_recorded + 1,
                                       hash_it->second[STAGE_COMPUTE], p.x, p.y, p.z);
        for (u32 slot = 0; slot < bound.uniform.size(); ++slot) {
            if (bound.uniform[slot].address != 0) {
                line += fmt::format("{}:{:#x}/{:#x} ", slot, bound.uniform[slot].address, bound.uniform[slot].size);
            }
        }
        line += "] ssbo=[";
        for (u32 slot = 0; slot < bound.storage.size(); ++slot) {
            if (bound.storage[slot].address != 0) {
                line += fmt::format("{}:{:#x}/{:#x} ", slot, bound.storage[slot].address, bound.storage[slot].size);
            }
        }
        line += "]";
        LOG_INFO(Render_Vulkan, "{}", line);
        const Debug::Config& config = Debug::Get();
        if (config.peek_program != 0 && config.peek_program == hash_it->second[STAGE_COMPUTE]) {
            // EDEN_NX2_NVN_PEEK: the textures the dispatch samples and the guest-memory view of a
            // storage range it reads (what earlier GPU work left there).
            std::string tex = fmt::format("[nvn] peek f{} prog={:016x} tex=[",
                                          impl.presents_recorded + 1, hash_it->second[STAGE_COMPUTE]);
            for (u32 slot = 0; slot < bound.combined.size(); ++slot) {
                const u32 handle = static_cast<u32>(bound.combined[slot]);
                if (handle == 0) {
                    continue;
                }
                const auto& texture = TextureOfId(VideoCommon::Nvn::TextureIdOf(handle));
                const u64 sampler_serial = SamplerSerialOfId(handle >> 20);
                tex += fmt::format("{}:", slot);
                if (texture) {
                    tex += fmt::format("t{:x}/{}x{}/fmt{}/tgt{}/lv{}/view_of{:x}", texture->serial,
                                       texture->desc.width, texture->desc.height,
                                       texture->desc.format, texture->desc.target,
                                       texture->desc.levels, texture->desc.view_of);
                }
                if (const auto it = impl.samplers.find(sampler_serial); it != impl.samplers.end()) {
                    const VC::SamplerDesc& s = it->second;
                    tex += fmt::format("/s{:x}(f{}{} w{}{}{} cmp{}/{} red{} border {} {} {} {})",
                                       sampler_serial, s.min_filter, s.mag_filter, s.wrap_s,
                                       s.wrap_t, s.wrap_r, s.compare_mode, s.compare_func,
                                       s.reduction, s.border[0], s.border[1], s.border[2],
                                       s.border[3]);
                }
                tex += ' ';
            }
            LOG_INFO(Render_Vulkan, "{}]", tex);
            // Slots from 100 on name uniform buffer slots (slot - 100).
            const bool uniform_peek = config.peek_slot >= 100;
            const u32 peek_slot = uniform_peek ? config.peek_slot - 100 : config.peek_slot;
            const u64 address =
                uniform_peek ? (peek_slot < bound.uniform.size() ? bound.uniform[peek_slot].address : 0)
                             : (peek_slot < bound.storage.size() ? bound.storage[peek_slot].address : 0);
            const u8* bytes = address != 0
                                  ? impl.GuestPointer(address + config.peek_offset,
                                                      u64{config.peek_words} * 4)
                                  : nullptr;
            if (address != 0 && config.watch_address == 3) {
                peek_address = address + config.peek_offset;
            }
            if (bytes != nullptr && config.peek_words != 0) {
                std::vector<float> values(config.peek_words);
                std::memcpy(values.data(), bytes, values.size() * 4);
                if (config.chase_words != 0 && u64{config.chase_word} * 4 + 8 <= values.size() * 4) {
                    // EDEN_NX2_NVN_PEEK_CHASE: follow the u64 pointer at that word of the range;
                    // log the guest words there (the GPU view comes from WATCH=peek).
                    u64 pointer = 0;
                    std::memcpy(&pointer, bytes + u64{config.chase_word} * 4, sizeof(pointer));
                    const u64 target = pointer + config.chase_offset;
                    const u64 size = u64{config.chase_words} * 4;
                    std::string hex;
                    if (const u8* chased = pointer != 0 ? impl.GuestPointer(target, size) : nullptr) {
                        for (u64 i = 0; i < size; i += 4) {
                            u32 word;
                            std::memcpy(&word, chased + i, 4);
                            hex += fmt::format(" {:x}", word);
                        }
                    } else {
                        hex = " not readable";
                    }
                    LOG_INFO(Render_Vulkan, "[nvn] chase f{} {:#x}+{:#x} guest:{}",
                             impl.presents_recorded + 1, pointer, config.chase_offset, hex);
                    if (pointer != 0 && config.watch_address == 3) {
                        peek_chase_address = target;
                    }
                }
                if (config.poke_set && config.poke_index < config.peek_words) {
                    // EDEN_NX2_NVN_POKE, wrong on purpose: overwrite one word of the peeked guest
                    // range before the dispatch runs (imported pools only: the GPU reads it).
                    std::memcpy(const_cast<u8*>(bytes) + u64{config.poke_index} * 4,
                                &config.poke_value, 4);
                }
                float lo = values[0];
                float hi = values[0];
                double sum = 0.0;
                u32 zeros = 0;
                for (const float v : values) {
                    lo = (std::min)(lo, v);
                    hi = (std::max)(hi, v);
                    sum += v;
                    zeros += v == 0.0f ? 1 : 0;
                }
                std::string words;
                for (u32 i = 0; i < (std::min)(config.peek_words, 512u); ++i) {
                    words += fmt::format(" {:.6g}", values[i]);
                }
                LOG_INFO(Render_Vulkan,
                         "[nvn] peek f{} slot {} {:#x}+{:#x} x{}: min {:.6g} max {:.6g} mean {:.6g} "
                         "zeros {}; first{}",
                         impl.presents_recorded + 1, config.peek_slot, address, config.peek_offset,
                         config.peek_words, lo, hi, sum / values.size(), zeros, words);
            } else {
                LOG_INFO(Render_Vulkan, "[nvn] peek f{} slot {} {:#x}+{:#x}: not readable",
                         impl.presents_recorded + 1, config.peek_slot, address,
                         config.peek_offset);
            }
        }
    }
    CostSplitWork cost_split{costsplit ? CostSplitBeginDispatch(hash_it->second[STAGE_COMPUTE])
                                       : nullptr,
                             true};
    compute_groups = indirect ? std::array<u32, 3>{} : std::array<u32, 3>{p.x, p.y, p.z};
    if (!BindDescriptors(*compute, VK_PIPELINE_BIND_POINT_COMPUTE)) {
        skipped();
        return;
    }
    // nvn_draw_dump: claimed only now, so every claimed section is written.
    std::string probe_prefix; // nvn_draw_dump: path prefix of the image read-backs after the dispatch
    if (probe_active) [[unlikely]] {
        if (const auto ticket = VideoCommon::NvnDrawProbe::DumpRequested(probe_info)) {
            ProbeDispatchDump(*ticket, *compute, *control, p, indirect ? &arguments : nullptr,
                              program_hash, probe_skip);
            probe_prefix = fmt::format("{}-{}", ticket->path, ticket->index + 1);
        }
    }
    const Debug::Config& backing_config = Debug::Get();
    if (backing_config.log_backing_program == hash_it->second[STAGE_COMPUTE] &&
        log_backing_dispatches < backing_config.log_backing_max) {
        ++log_backing_dispatches;
        for (u32 slot = 0; slot < bound.storage.size(); ++slot) {
            const BufferBinding& storage = bound.storage[slot];
            if (storage.address == 0) {
                continue;
            }
            Pool* pool = nullptr;
            u64 pool_offset = 0;
            const bool mapped = impl.ResolveBuffer(storage.address, storage.size != 0 ? storage.size : 1,
                                                   pool, pool_offset);
            bool shadow = false;
            if (mapped && storage.size != 0) {
                for (const Shadow& live : shadows) {
                    if (live.pool == pool && pool_offset >= live.begin &&
                        pool_offset + storage.size <= live.end) {
                        shadow = true;
                        break;
                    }
                }
            }
            bool raw_table = false;
            bool atomic = false;
            for (const VC::DescriptorBinding& descriptor : compute->descriptors.bindings) {
                if (descriptor.role != VC::DescriptorRole::StorageBuffer ||
                    slot < descriptor.slot || slot - descriptor.slot >= descriptor.count) {
                    continue;
                }
                raw_table |= descriptor.is_raw_global_table &&
                             descriptor.storage_source != VC::StorageSource::None;
                atomic |= descriptor.is_atomic_target;
            }
            const BackingClass backing = ClassifyBacking(
                raw_table, shadow, mapped && pool != nullptr &&
                                       (pool->backing == Pool::Backing::Mirror ||
                                        pool->vram_mirror != nullptr),
                mapped && pool != nullptr && pool->backing == Pool::Backing::Imported,
                mapped && pool != nullptr && pool->backing == Pool::Backing::DeviceLocal);
            LOG_INFO(Render_Vulkan,
                     "[nvn] backing prog={:016x} dispatch={} slot={} guest={:#x} size={:#x} "
                     "class={} atomic={}",
                     hash_it->second[STAGE_COMPUTE], log_backing_dispatches, slot, storage.address,
                     storage.size, BackingClassName(backing), atomic);
        }
    }


    // The three u32 must be in the buffer the GPU sees.
    if (indirect) {
        if (arguments.pool != nullptr && arguments.pool->backing == Pool::Backing::Mirror) {
            host->HostUploadRange(*arguments.pool, arguments.pool_offset, 3 * sizeof(u32));
        }
        host->HostFlushBarrier();
    }
    // EDEN_NX2_PATHDIFF (diagnostic): what this dispatch reads, copied before it.
    std::optional<InFlight::PathDiff> pathdiff;
    if (VideoCommon::PathDiff::Enabled()) [[unlikely]] {
        pathdiff = PathDiffBegin(*compute, false, hash_it->second,
                                 indirect ? &arguments : nullptr, 3 * sizeof(u32));
        if (pathdiff) {
            pathdiff->capture.groups = compute_groups;
            pathdiff->capture.indirect = indirect;
            pathdiff->capture.block = control->thread_group;
        }
    }

    vk::CommandBuffer cmd = host->HostCmd();
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, cmd,
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Dispatch,
                            .zero_size = prof_zero_size,
                            .count = indirect ? 0u
                                              : static_cast<u32>((std::min<u64>)(
                                                    u64{p.x} * p.y * p.z, ~u32{0})),
                            .hash_a = hash_it->second[STAGE_COMPUTE],
                            .width = indirect ? 0u : p.x,
                            .height = indirect ? 0u : p.y,
                            .shape = indirect ? 0u : p.z}
             : GpuProf::Tag{},
        0, cs_mode != CostSplit::Mode::Count ? 1u : 0u};
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *compute->pipeline);
    if (probe_skip) [[unlikely]] {
        // As the cost split's nogroups: the barriers and bookkeeping stay, the work vanishes.
        cmd.Dispatch(0, 0, 0);
    } else if (cs_mode == CostSplit::Mode::NoGroups || cs_mode == CostSplit::Mode::OneGroup)
        [[unlikely]] {
        // Direct or indirect, the mode's groups.
        const std::array<u32, 3> groups = CostSplit::Groups(cs_mode, {p.x, p.y, p.z});
        cmd.Dispatch(groups[0], groups[1], groups[2]);
        CostSplitClamped();
    } else if (indirect) {
        cmd.DispatchIndirect(arguments.buffer, arguments.offset);
    } else {
        cmd.Dispatch(p.x, p.y, p.z);
        counters.dispatch_groups += u64{p.x} * p.y * p.z;
    }
    cost_split.recorded = true;
    // The shader may have written buffers and images: later commands order after them (a later
    // dispatch only behind a game barrier with dispatch ordering by barriers, as on NVN).
    host->HostMarkDispatchWritten();
    if (!probe_prefix.empty()) [[unlikely]] {
        // What the dumped dispatch bound, as the GPU has it after the dispatch.
        ProbeNoteImages(probe_prefix, u64{2048} * 2048);
        ProbeFlushImages();
    }
    if (pathdiff) [[unlikely]] {
        PathDiffEnd(std::move(*pathdiff));
    }
}

} // namespace Vulkan::Nvn
