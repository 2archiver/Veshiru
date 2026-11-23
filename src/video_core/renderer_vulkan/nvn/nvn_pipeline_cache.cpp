// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <boost/container/small_vector.hpp>
#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/frame_stats.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/diag_request.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/program_header.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_disk.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_state.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/buffer_cache/raw_global_table.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_impl.h"
#include "video_core/shader_notify.h"

namespace Vulkan::Nvn {

u64 HashProgramStage(std::span<const u8> control, std::span<const u8> code) {
    // Include the control fields ParseControl uses, including the compute driver-bank layout.
    // Identical instructions with different bank layouts must not share a lowered program.
    constexpr size_t HASHED_CONTROL_BYTES = 0x6c;
    constexpr size_t UNHASHED_BEGIN = 0x48;
    constexpr size_t UNHASHED_END = 0x50;
    constexpr size_t LENGTH_WORD_OFFSET = 0x30;
    std::array<u8, HASHED_CONTROL_BYTES> bytes{};
    const size_t count = std::min(control.size(), HASHED_CONTROL_BYTES);
    if (count != 0) {
        std::memcpy(bytes.data(), control.data(), count);
    }
    if (count > UNHASHED_BEGIN) {
        std::memset(bytes.data() + UNHASHED_BEGIN, 0, std::min(count, UNHASHED_END) - UNHASHED_BEGIN);
    }
    if (count > 0x4c) {
        bytes[0x4c] = control[0x4c]; // register count
    }
    // The control section's own length differs between containers: zero that word.
    if (count > LENGTH_WORD_OFFSET) {
        std::memset(bytes.data() + LENGTH_WORD_OFFSET, 0,
                    std::min<size_t>(sizeof(u32), count - LENGTH_WORD_OFFSET));
    }
    u64 control_hash = Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), count);
    if (count > 0x56 && bytes[0x56] == STAGE_COMPUTE) {
        constexpr std::array<size_t, 7> COMPUTE_FIELDS{0xa9, 0xb1, 0xb2, 0xb3, 0xb6, 0xba, 0xbb};
        std::array<u8, COMPUTE_FIELDS.size() + 1> layout{};
        layout.back() = control.size() >= VC::ComputeControlBytes;
        for (size_t i = 0; i < COMPUTE_FIELDS.size(); ++i) {
            if (COMPUTE_FIELDS[i] < control.size()) {
                layout[i] = control[COMPUTE_FIELDS[i]];
            }
        }
        control_hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(layout.data()),
                                                  layout.size(), control_hash);
    }
    const u64 hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(code.data()),
                                                code.size(), control_hash);
    return hash != 0 ? hash : 1;
}

bool StoresGlobalMemory(const GraphicsRequest& request) noexcept {
    for (const StageProgram& stage : request.stages) {
        if (stage.hash == 0 || stage.code == nullptr) {
            continue;
        }
        Shader::ProgramHeader header;
        if (stage.code->size() < sizeof(header.common0)) {
            continue;
        }
        std::memcpy(&header.common0, stage.code->data(), sizeof(header.common0));
        if (header.common0.does_global_store != 0) {
            return true;
        }
    }
    return false;
}

bool PipelineBase::QueriesMatch(const TextureOracle& oracle) const {
    for (const TextureQuery& query : queries) {
        const TextureInfo now = oracle.Query(query.stage, query.record);
        if (now.type != query.info.type || now.format != query.info.format ||
            now.is_integer != query.info.is_integer) {
            return false;
        }
    }
    return true;
}

namespace {

using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;

/// Components of the input interface the emitter declares for a geometry stage: the generic
/// locations the previous stage writes and the stage loads (or passes through), counted up to the
/// highest one, plus the position built-in.
u32 GeometryInputComponents(const Shader::IR::Program& program,
                            const Shader::RuntimeInfo& runtime_info) {
    const Shader::VaryingState loads{program.info.loads.mask | program.info.passthrough.mask};
    u32 locations = 0;
    for (u32 index = 0; index < Shader::IR::NUM_GENERICS; ++index) {
        if (loads.Generic(index) && runtime_info.previous_stage_stores.Generic(index) &&
            runtime_info.generic_input_types[index] != Shader::AttributeType::Disabled) {
            locations = index + 1;
        }
    }
    return locations * 4 + (loads.AnyComponent(Shader::IR::Attribute::PositionX) ? 4u : 0u);
}

class ByteBuffer {
public:
    template <typename T>
    void Add(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* const first = reinterpret_cast<const u8*>(&value);
        data.insert(data.end(), first, first + sizeof(T));
    }
    [[nodiscard]] u64 Hash() const {
        return Common::CityHash64(reinterpret_cast<const char*>(data.data()), data.size());
    }

private:
    std::vector<u8> data;
};

/// Why the device cannot run a task or mesh program (VK_EXT_mesh_shader support and limits);
/// empty when it can or the program is neither.
std::string MeshLimitExceeded(const Device& device, const Shader::IR::Program& program) {
    const bool task = program.stage == Shader::Stage::Task;
    if (!task && program.stage != Shader::Stage::Mesh) {
        return {};
    }
    if (!device.IsExtMeshShaderSupported()) {
        return "task/mesh programs need VK_EXT_mesh_shader with task and mesh shaders";
    }
    const VkPhysicalDeviceMeshShaderPropertiesEXT& limits = device.MeshShaderProperties();
    const u32 threads = program.workgroup_size[0];
    const u32 payload = program.mesh.payload_words * static_cast<u32>(sizeof(u32));
    if (payload > limits.maxTaskPayloadSize) {
        return fmt::format("task payload of {} bytes exceeds maxTaskPayloadSize {}", payload,
                           limits.maxTaskPayloadSize);
    }
    const u32 max_threads = task ? (std::min)(limits.maxTaskWorkGroupInvocations,
                                              limits.maxTaskWorkGroupSize[0])
                                 : (std::min)(limits.maxMeshWorkGroupInvocations,
                                              limits.maxMeshWorkGroupSize[0]);
    if (threads > max_threads) {
        return fmt::format("{} workgroup of {} threads exceeds the device's {}",
                           task ? "task" : "mesh", threads, max_threads);
    }
    if (!task && (program.mesh.max_vertices > limits.maxMeshOutputVertices ||
                  program.mesh.max_primitives > limits.maxMeshOutputPrimitives)) {
        return fmt::format("mesh outputs {} vertices / {} primitives exceed the device's {} / {}",
                           program.mesh.max_vertices, program.mesh.max_primitives,
                           limits.maxMeshOutputVertices, limits.maxMeshOutputPrimitives);
    }
    return {};
}

/// The translation-relevant content of a RuntimeInfo plus the binding base a stage was emitted at.
/// Task and mesh stages also depend on the pipeline's mesh register words (the task's init word
/// sizes the mesh program's payload input).
u64 HashRuntime(u32 nvn_stage, const Shader::RuntimeInfo& info,
                const Shader::Backend::Bindings& bindings, u64 extra,
                const std::array<u32, 3>& mesh_registers) {
    ByteBuffer buf;
    buf.Add(extra);
    buf.Add(nvn_stage);
    if (nvn_stage == STAGE_TASK || nvn_stage == STAGE_MESH) {
        for (const u32 word : mesh_registers) {
            buf.Add(word);
        }
    }
    buf.Add(info.per_primitive_generics.to_ulong());
    buf.Add(bindings.unified);
    buf.Add(bindings.uniform_buffer);
    buf.Add(bindings.storage_buffer);
    buf.Add(bindings.texture);
    buf.Add(bindings.image);
    buf.Add(bindings.texture_scaling_index);
    buf.Add(bindings.image_scaling_index);
    for (const Shader::AttributeType type : info.generic_input_types) {
        buf.Add(static_cast<u8>(type));
    }
    for (size_t word = 0; word < 512 / 64; ++word) {
        u64 bits = 0;
        for (size_t bit = 0; bit < 64; ++bit) {
            bits |= u64{info.previous_stage_stores.mask[word * 64 + bit]} << bit;
        }
        buf.Add(bits);
    }
    for (const auto& [from, to] : info.previous_stage_legacy_stores_mapping) {
        buf.Add(static_cast<u64>(from));
        buf.Add(static_cast<u64>(to));
    }
    buf.Add(info.convert_depth_mode);
    buf.Add(info.force_early_z);
    buf.Add(static_cast<u32>(info.tess_primitive));
    buf.Add(static_cast<u32>(info.tess_spacing));
    buf.Add(info.tess_clockwise);
    buf.Add(static_cast<u32>(info.input_topology));
    buf.Add(info.fixed_state_point_size.has_value());
    buf.Add(std::bit_cast<u32>(info.fixed_state_point_size.value_or(0.0f)));
    buf.Add(info.alpha_test_func.has_value());
    buf.Add(static_cast<u32>(info.alpha_test_func.value_or(Shader::CompareFunction::Never)));
    buf.Add(std::bit_cast<u32>(info.alpha_test_reference));
    buf.Add(info.y_negate);
    for (const Shader::AttributeType type : info.color_output_types) {
        buf.Add(static_cast<u8>(type));
    }
    buf.Add(info.dual_source_blend);
    return buf.Hash();
}

u64 HashProfile(const Shader::Profile& profile, const Shader::HostTranslateInfo& host) {
    ByteBuffer buf;
    buf.Add(profile.supported_spirv);
    buf.Add(profile.support_descriptor_aliasing);
    buf.Add(profile.support_int8);
    buf.Add(profile.support_int16);
    buf.Add(profile.support_int64);
    buf.Add(profile.support_float_controls);
    buf.Add(profile.support_native_ndc);
    buf.Add(profile.support_scaled_attributes);
    buf.Add(profile.support_multi_viewport);
    buf.Add(profile.support_viewport_index_layer_non_geometry);
    buf.Add(profile.support_geometry_shader_passthrough);
    buf.Add(profile.support_demote_to_helper_invocation);
    buf.Add(profile.support_typeless_image_loads);
    buf.Add(profile.supported_subgroup_stages);
    if (profile.has_broken_spirv_fragment_shuffle) {
        buf.Add(profile.has_broken_spirv_fragment_shuffle); // only when set: other keys unchanged
    }
    buf.Add(profile.max_subgroup_size);
    buf.Add(profile.min_ssbo_alignment);
    buf.Add(profile.max_user_clip_distances);
    buf.Add(profile.support_raw_global_memory);
    // The layout of the raw global table the translated shaders read (profile.h): a binary
    // search over a device table, address at c0 + RAW_GLOBAL_TABLE_ADDRESS_OFFSET.
    buf.Add(Shader::Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET);
    buf.Add(Shader::Profile::RAW_GLOBAL_TABLE_ENTRY_SIZE);
    buf.Add(host.nvn_raw_global_memory);
    buf.Add(host.support_float64);
    buf.Add(host.compute_storage64);
    buf.Add(host.support_float16);
    buf.Add(host.support_int64);
    buf.Add(host.max_per_stage_descriptor_sampled_images);
    buf.Add(host.max_per_stage_resources);
    buf.Add(host.max_descriptor_set_samplers);
    buf.Add(host.max_descriptor_set_uniform_buffers);
    buf.Add(host.max_descriptor_set_storage_buffers);
    buf.Add(host.max_descriptor_set_sampled_images);
    buf.Add(host.max_descriptor_set_storage_images);
    return buf.Hash();
}

} // Anonymous namespace


// ================================================================================================
// diagnostics
// ================================================================================================

bool PipelineCache::Impl::NoteRow(const std::string& reason) {
    if (ShaderLog::Enabled()) {
        LastIssueSlot() = reason;
    }
    std::scoped_lock lock{rows_mutex};
    return ++rows[reason] == 1;
}

void PipelineCache::Impl::NoteIssue(const std::string& reason) {
    if (NoteRow(reason)) {
        LOG_WARNING(Render_Vulkan, "[nvn] {}", reason);
    }
}

void PipelineCache::Impl::Reject(u64 stage_hash, u32 nvn_stage, std::string_view phase,
                                 std::string_view message) {
    const std::string line = FirstLine(message);
    const std::string reason = fmt::format("{}: {}: {}", StageNameOf(nvn_stage), phase, line);
    ++stat_failures;
    NoteRow(reason);
    bool first = false;
    {
        std::scoped_lock lock{reject_mutex};
        first = rejected_stages.emplace(stage_hash, reason).second;
    }
    if (first) {
        ++stat_rejected_shaders;
        LOG_ERROR(Render_Vulkan, "[nvn] shader rejected: {} {:016x}: {}", StageNameOf(nvn_stage),
                  stage_hash, reason);
    }
}

bool PipelineCache::Impl::AnyStageRejected(std::span<const u64> hashes) {
    std::string reason;
    {
        std::scoped_lock lock{reject_mutex};
        if (rejected_stages.empty()) {
            return false;
        }
        for (const u64 hash : hashes) {
            if (hash == 0) {
                continue;
            }
            const auto it = rejected_stages.find(hash);
            if (it != rejected_stages.end()) {
                reason = it->second;
                break;
            }
        }
    }
    if (reason.empty()) {
        return false;
    }
    NoteRow(reason);
    return true;
}

void PipelineCache::Impl::LogCounts() const {
    const u64 loaded = stat_loaded.load();
    const u64 compiled = stat_compiled.load();
    logged_loaded = loaded;
    logged_compiled = compiled;
    LOG_INFO(Render_Vulkan, "[nvn] pipeline cache: {} loaded, {} compiled at run time", loaded,
             compiled);
    if (async_workers) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] pipeline cache async: {} compiles queued, {} failed, {} lookups answered "
                 "not ready",
                 stat_async_queued.load(), stat_async_failed.load(),
                 stat_async_not_ready.load());
    }
    if (const u64 waits = stat_wait_count.load(); waits != 0) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] pipeline cache: {} lookups waited {} ms in total (longest {} ms) for the "
                 "preload or a key in flight",
                 waits, stat_wait_ns.load() / 1'000'000, stat_wait_max_ns.load() / 1'000'000);
    }
    if (compiled != 0) {
        constexpr u64 MS = 1'000'000;
        LOG_INFO(Render_Vulkan,
                 "[nvn] pipeline compile time: {} compiles took {} ms in total (translate {} ms, "
                 "create {} ms, longest {} ms)",
                 compiled, stat_compile_ns.load() / MS, stat_translate_ns.load() / MS,
                 stat_create_ns.load() / MS, stat_compile_max_ns.load() / MS);
    }
    if (async_workers) {
        const u64 first = stat_async_first_ticks.load();
        const u64 last = stat_async_last_ticks.load();
        const u64 span_ms =
            last > first ? std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::duration{static_cast<s64>(last - first)})
                               .count()
                         : 0;
        LOG_INFO(Render_Vulkan,
                 "[nvn] pipeline cache async workers: {} threads, at most {} jobs queued or "
                 "running at once, first job queued to last job done {} ms",
                 async_worker_count, stat_async_peak.load(), span_ms);
    }
    if (options.prefetch_graphics) {
        LOG_INFO(Render_Vulkan, "[nvn] graphics lookahead: {} compiles queued, {} pipelines used",
                 stat_prefetch_queued.load(), stat_prefetch_used.load());
    }
    if (GplEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] pipeline libraries: {} fast links ({} from parts compiled already), {} "
                 "built from earlier translations, {} compiled whole; {} optimized compiles "
                 "swapped in ({} failed, {} over 50 ms, {} ms in total, background); {} program "
                 "stages lowered at creation; {} shader sets of the disk cache warmed",
                 stat_gpl_fast_links.load(), stat_gpl_cached_links.load(), stat_gpl_reused.load(),
                 stat_gpl_whole.load(), stat_gpl_optimized.load(),
                 stat_gpl_optimize_failed.load(), stat_gpl_optimize_slow.load(),
                 stat_gpl_optimize_ns.load() / 1'000'000, stat_prepared.load(),
                 stat_gpl_warmed.load());
    }
}

void PipelineCache::Impl::LogCountsIfChanged() const {
    if (stat_loaded.load() != logged_loaded.load() ||
        stat_compiled.load() != logged_compiled.load()) {
        LogCounts();
    }
}

void PipelineCache::Impl::NoteCompileTime(std::string_view kind, u64 hash,
                                          std::chrono::steady_clock::time_point start,
                                          std::chrono::steady_clock::duration translate,
                                          std::chrono::steady_clock::duration create) {
    using Ms = std::chrono::duration<double, std::milli>;
    constexpr auto SLOW = std::chrono::milliseconds(50);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto ns = [](std::chrono::steady_clock::duration d) {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    const u64 elapsed_ns = ns(elapsed);
    stat_compile_ns += elapsed_ns;
    stat_translate_ns += ns(translate);
    stat_create_ns += ns(create);
    u64 seen = stat_compile_max_ns.load();
    while (elapsed_ns > seen && !stat_compile_max_ns.compare_exchange_weak(seen, elapsed_ns)) {
    }
    const double total_ms = std::chrono::duration_cast<Ms>(elapsed).count();
    const double translate_ms = std::chrono::duration_cast<Ms>(translate).count();
    const double create_ms = std::chrono::duration_cast<Ms>(create).count();
    if (elapsed > SLOW) {
        LOG_WARNING(Render_Vulkan,
                    "[nvn] slow pipeline compile: {:.1f} ms (translate {:.1f}, create {:.1f}) "
                    "{} {:016x}",
                    total_ms, translate_ms, create_ms, kind, hash);
    } else {
        LOG_INFO(Render_Vulkan,
                 "[nvn] pipeline compile: {:.1f} ms (translate {:.1f}, create {:.1f}) {} {:016x}",
                 total_ms, translate_ms, create_ms, kind, hash);
    }
}

// ================================================================================================
// construction
// ================================================================================================

PipelineCache::Impl::Impl(const Device& device_, PipelineCacheOptions options_)
    : device{device_}, options{std::move(options_)}, profile{MakeProfile(device_)},
      host_info{MakeHostInfo(device_, profile)} {
    try_generation_at_init = Common::DiagRequest::TrySwitchGeneration();
    const VkPhysicalDeviceProperties props = device.GetPhysical().GetProperties();
    disk_header.revision = options.revision;
    disk_header.vendor_id = props.vendorID;
    disk_header.device_id = props.deviceID;
    disk_header.driver_version = props.driverVersion;
    disk_header.cache_uuid_hash = Common::CityHash64(
        reinterpret_cast<const char*>(props.pipelineCacheUUID), sizeof(props.pipelineCacheUUID));
    // The vertex input mode is a property of every key and record: a file written in one mode
    // must never be loaded in the other (the header's profile hash is compared as a whole).
    vertex_input_mode = options.allow_dynamic_vertex_input &&
                                device.IsExtVertexInputDynamicStateSupported()
                            ? VertexInputMode::Dynamic
                            : VertexInputMode::Static;
    must_emulate_scaled_formats = device.MustEmulateScaledFormats();
    // The titles whose own tensor programs are lowered, as the emulated GPU picks them
    // (KeplerCompute::EmulatesNativeTensorPrograms); the compact form needs compute pipelines
    // created 32 threads wide.
    tensor_lowering = VideoCommon::NativeTensorLoweringFor(
        Settings::GetCurrentProgramID(),
        device.IsExtSubgroupSizeControlSupported() &&
            device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT));
    if (tensor_lowering != VideoCommon::NativeTensorLowering::None) {
        LOG_INFO(Render_Vulkan, "[nvn] tensor programs: lowered ({} form)",
                 tensor_lowering == VideoCommon::NativeTensorLowering::Compact ? "compact"
                                                                               : "full");
    }
    const u32 mode_word = static_cast<u32>(vertex_input_mode);
    disk_header.profile_hash = Common::CityHash64WithSeed(
        reinterpret_cast<const char*>(&mode_word), sizeof(mode_word),
        HashProfile(profile, host_info));
    const std::string_view epoch = options.translation_epoch.empty()
                                       ? Disk::TranslationEpoch()
                                       : std::string_view{options.translation_epoch};
    TranslationSwitches switches = CurrentTranslationSwitches();
    switches.tensor_lowering = static_cast<u8>(tensor_lowering);
    disk_header.translation_key = Disk::MakeTranslationKey({
        .epoch = epoch,
        .switches_hash = HashTranslationSwitches(switches),
        .vendor_id = disk_header.vendor_id,
        .device_id = disk_header.device_id,
        .driver_version = disk_header.driver_version,
        .cache_uuid_hash = disk_header.cache_uuid_hash,
        .profile_hash = disk_header.profile_hash,
    });

    bool disk = DiskEnabled();
    std::vector<u8> blob;
    if (disk) {
        std::error_code error;
        std::filesystem::create_directories(options.directory, error);
        if (error) {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache directory {} unusable: {}",
                      options.directory.string(), error.message());
            disk = false;
            options.directory.clear();
        }
    }
    if (disk) {
        disk_path = options.directory / Disk::FileNameForKey(disk_header.translation_key);
        LOG_INFO(Render_Vulkan, "[nvn] pipeline cache file {} for {}", disk_path.filename().string(),
                 disk_header.translation_key);
        if (options.driver_cache_file) {
            vk_cache_path = options.directory / VK_CACHE_FILE_NAME;
            blob = ReadVkCacheBlob();
        } else {
            LOG_INFO(Render_Vulkan,
                     "[nvn] driver pipeline cache file off (use_vulkan_driver_pipeline_cache): {} "
                     "is neither read nor written",
                     VK_CACHE_FILE_NAME);
        }
    }
    const auto create_cache = [this](const std::vector<u8>& data) {
        return device.GetLogical().CreatePipelineCache({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data.size(),
            .pInitialData = data.empty() ? nullptr : data.data(),
        });
    };
    try {
        vk_cache = create_cache(blob);
    } catch (const vk::Exception&) {
        vk_cache = create_cache({});
    }
    last_vk_write_ticks = std::chrono::steady_clock::now().time_since_epoch().count();

    if (!disk) {
        writer.SetState(DiskWriter::State::Disabled);
        loader_done = true;
        loader_queued = true;
    } else {
        if (options.preload) {
            const s64 hw = static_cast<s64>(std::thread::hardware_concurrency());
            const size_t threads = static_cast<size_t>(std::max<s64>(1, hw / 2 - 1));
            workers.reset(new Common::ThreadWorker(threads, "NvnPipelineLoader", {},
                                                   Common::ThreadPlacement::Background));
        }
    }
    if (options.async_compile || options.prefetch_graphics) {
        // Run-time compiles block draws that cannot be skipped by the preload, so they run at
        // normal priority on their own threads (the preload workers are a background queue).
        const s64 hw = static_cast<s64>(std::thread::hardware_concurrency());
        size_t threads = static_cast<size_t>(std::clamp<s64>(hw / 4, 2, 6));
        if (const u32 forced = Debug::Get().async_workers; forced != 0) {
            threads = forced; // EDEN_NX2_NVN_ASYNC_WORKERS: worker-count experiments
        }
        async_worker_count = threads;
        LOG_INFO(Render_Vulkan, "[nvn] pipeline compile workers: {} (hardware threads {})", threads,
                 hw);
        async_workers.reset(new Common::ThreadWorker(threads, "NvnPipelineAsync"));
    }
    if (options.graphics_pipeline_library) {
        if (device.IsExtGraphicsPipelineLibrarySupported()) {
            library_cache = std::make_unique<GraphicsPipelineLibraryCache>(device);
            gpl_dynamic_fixed = device.IsExtExtendedDynamicStateSupported() &&
                                device.IsExtExtendedDynamicState2Supported();
            const s64 hw = static_cast<s64>(std::thread::hardware_concurrency());
            const size_t optimize_threads = static_cast<size_t>(std::clamp<s64>(hw / 8, 1, 4));
            // A link compiles at most four parts: one on the calling thread, three here.
            part_workers.reset(new Common::ThreadWorker(3, "NvnPipelineParts"));
            optimize_workers.reset(new Common::ThreadWorker(optimize_threads, "NvnPipelineOptimize",
                                                            {}, Common::ThreadPlacement::Background));
            prepare_workers.reset(new Common::ThreadWorker(1, "NvnProgramPrepare", {},
                                                           Common::ThreadPlacement::Background));
            LOG_INFO(Render_Vulkan,
                     "[nvn] nvn_pipeline_library on: graphics pipelines are fast-linked from "
                     "pipeline libraries when a draw needs them (cull, depth and stencil state {}), "
                     "then compiled whole on {} background threads",
                     gpl_dynamic_fixed ? "dynamic" : "in the parts", optimize_threads);
        } else {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] nvn_pipeline_library is on, but the device has no "
                        "VK_EXT_graphics_pipeline_library with fast linking: pipelines are "
                        "compiled whole");
        }
    }
    // RunLoader can inspect the GPL state as soon as it starts.
    if (disk) {
        loader = std::jthread([this](std::stop_token stop) { RunLoader(stop); });
    }
    // The counts line is printed once the preload finished (RunLoader), whenever the stats
    // thread collects rows and changed counts, and at destruction.
}

PipelineCache::Impl::~Impl() {
    stopping = true;
    if (loader.joinable()) {
        loader.request_stop();
        loader.join();
    }
    workers.reset();
    async_workers.reset();
    // Jobs the stopped workers dropped unrun: balance their MarkShaderBuilding (AsyncQueued).
    if (options.shader_notify != nullptr) {
        std::scoped_lock lock{async_mutex};
        for (; async_outstanding != 0; --async_outstanding) {
            options.shader_notify->MarkShaderComplete();
        }
    }
    // Optimize jobs hold pipelines and shader sets of the maps: joined before those go.
    part_workers.reset();
    optimize_workers.reset();
    prepare_workers.reset();
    if (new_since_vk_write.load() != 0 && DiskEnabled() && !vk_cache_path.empty()) {
        writer.Post([this] { WriteVkCache(); });
    }
    writer.Shutdown();
    LogCounts();
    // Vulkan objects of the maps are destroyed by the member destructors (reverse order).
}

void PipelineCache::Impl::Flush() {
    {
        std::unique_lock lock{loader_mutex};
        loader_cv.wait(lock, [this] { return loader_done || stopping.load(); });
    }
    {
        std::unique_lock lock{async_mutex};
        async_cv.wait(lock, [this] { return async_outstanding == 0 || stopping.load(); });
    }
    {
        std::unique_lock lock{gpl_mutex};
        gpl_cv.wait(lock, [this] { return gpl_outstanding == 0 || stopping.load(); });
    }
    writer.Flush();
}

void PipelineCache::Impl::ClearShaders() {
    Flush();
    auto fresh_cache = device.GetLogical().CreatePipelineCache({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    });
    auto fresh_libraries =
        library_cache ? std::make_unique<GraphicsPipelineLibraryCache>(device) : nullptr;
    const auto before = Stats();
    {
        std::scoped_lock lock{pipeline_mutex};
        graphics.clear();
        compute.clear();
        failed_graphics.clear();
        failed_compute.clear();
    }
    {
        std::scoped_lock lock{translation_mutex};
        translations.clear();
    }
    library_cache = std::move(fresh_libraries);
    {
        std::scoped_lock lock{lowered_mutex};
        lowered.clear();
    }
    {
        std::scoped_lock lock{reject_mutex};
        rejected_stages.clear();
    }
    vk_cache = std::move(fresh_cache);
    const u64 generation = ++shader_generation;
    LOG_INFO(Render_Vulkan,
             "[nvn] shader cache refreshed: generation {} ({} graphics, {} compute, {} lowered); "
             "next use recompiles",
             generation, before.graphics_pipelines, before.compute_pipelines,
             before.lowered_programs);
}

PipelineCacheStats PipelineCache::Impl::Stats() const {
    PipelineCacheStats stats;
    stats.loaded = stat_loaded;
    stats.compiled_at_runtime = stat_compiled;
    stats.translation_failures = stat_failures;
    stats.rejected_shaders = stat_rejected_shaders;
    stats.disk_entries_written = stat_disk_written;
    stats.disk_rejected = stat_disk_rejected;
    stats.variants_retranslated = stat_retranslated;
    stats.async_queued = stat_async_queued;
    stats.async_failed = stat_async_failed;
    stats.async_not_ready = stat_async_not_ready;
    stats.preload_waits = stat_wait_count;
    stats.preload_wait_ns = stat_wait_ns;
    stats.prefetch_queued = stat_prefetch_queued;
    stats.prefetch_used = stat_prefetch_used;
    stats.gpl_fast_links = stat_gpl_fast_links;
    stats.gpl_cached_links = stat_gpl_cached_links;
    stats.gpl_translations_reused = stat_gpl_reused;
    stats.gpl_whole = stat_gpl_whole;
    stats.gpl_optimized = stat_gpl_optimized;
    stats.gpl_shader_sets_warmed = stat_gpl_warmed;
    stats.programs_prepared = stat_prepared;
    {
        std::scoped_lock lock{pipeline_mutex};
        for (const auto& [key, pipelines] : graphics) {
            stats.graphics_pipelines += pipelines.size();
        }
        for (const auto& [key, pipelines] : compute) {
            stats.compute_pipelines += pipelines.size();
        }
    }
    {
        std::scoped_lock lock{lowered_mutex};
        stats.lowered_programs = lowered.size();
    }
    return stats;
}

void PipelineCache::Impl::CollectRows(std::vector<Row>& out) const {
    LogCountsIfChanged();
    std::scoped_lock lock{rows_mutex};
    for (const auto& [reason, count] : rows) {
        out.push_back({reason, count});
    }
}


// ================================================================================================
// translation
// ================================================================================================

LoweredEntry PipelineCache::Impl::GetLowered(const StageProgram& program, u32 nvn_stage) {
    {
        std::scoped_lock lock{lowered_mutex};
        if (const auto it = lowered.find(program.hash); it != lowered.end()) {
            return it->second;
        }
    }
    LoweredEntry entry;
    try {
        entry.program = VC::LowerNvnProgram(
            VC::NvnProgramBinary{
                static_cast<VC::ShaderStage>(nvn_stage),
                std::span<const u8>{program.control->data(), program.control->size()},
                std::span<const u8>{program.code->data(), program.code->size()}},
            // No cooperative matrix lowering in Eglantine: a game's own DLSS network is lowered
            // for the titles NativeTensorLoweringFor names, otherwise rejected and replaced by the
            // replay (nvn_upscale.cpp).
            tensor_lowering);
    } catch (const std::exception& e) {
        auto failed = std::make_shared<VC::LoweredNvnProgram>();
        failed->error = std::string{"exception: "} + e.what();
        entry.program = std::move(failed);
    }
    if (entry.program && entry.program->error.empty()) {
        entry.c1 = std::make_shared<const std::vector<u32>>(entry.program->c1);
        // The GPU profile names programs by stage hash, the shader tools by the native key.
        static const bool log_keys = [] {
            const char* value = std::getenv("EDEN_NX2_NVN_GPU_PROFILE");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        if (log_keys) {
            LOG_INFO(Render_Vulkan, "[shaderkey] nvn stage_hash={:016x} native={:016x} stage={}",
                     program.hash, entry.program->native_key, nvn_stage);
        }
    }
    std::scoped_lock lock{lowered_mutex};
    return lowered.emplace(program.hash, std::move(entry)).first->second;
}

Shader::RuntimeInfo PipelineCache::Impl::MakeRuntimeInfo(
    u32 nvn_stage, const Shader::IR::Program& program, const Shader::IR::Program* previous,
    const TranslationState* translation, bool has_geometry) {
    Shader::RuntimeInfo info;
    if (previous != nullptr) {
        info.previous_stage_stores = previous->info.stores;
        info.previous_stage_legacy_stores_mapping = previous->info.legacy_stores_mapping;
        if (previous->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous != nullptr && previous->stage == Shader::Stage::Mesh) {
        // Fragment inputs the mesh program writes per primitive: the output map of its
        // per-primitive header (NvnEnvironment, from the control section's +0x64/+0x68).
        for (const u32 line : previous->mesh.primitive_lines) {
            const auto attribute = static_cast<Shader::IR::Attribute>(line);
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    if (translation == nullptr) {
        return info; // compute
    }
    // Nothing of the draw's state but `translation` is read here: TranslationState is the key
    // under which the GPL path reuses translated stages (nvn_pipeline_gpl.cpp).
    const TranslationState& ts = *translation;
    const auto topology = static_cast<Shader::InputTopology>(ts.input_topology);
    const float point_size = std::bit_cast<float>(ts.point_size_bits);
    const bool convert_depth = ts.convert_depth_mode != 0;
    switch (nvn_stage) {
    case STAGE_VERTEX:
        if (!has_geometry) {
            if (topology == Shader::InputTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            info.convert_depth_mode = convert_depth;
        }
        for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
            info.generic_input_types[location] =
                static_cast<Shader::AttributeType>(ts.vertex_types[location]);
        }
        break;
    case STAGE_TESS_EVAL:
        NoteIssue("tessellation evaluation: primitive/spacing/winding are not part of the NVN "
                  "pipeline state (recompiler defaults used)");
        break;
    case STAGE_GEOMETRY:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        info.convert_depth_mode = convert_depth;
        break;
    case STAGE_MESH:
        // The last stage before rasterization, like a vertex stage without geometry. A point
        // mesh program (SET_MESH_SHADER_A topology 0) takes the fixed state point size per vertex.
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        info.convert_depth_mode = convert_depth;
        break;
    case STAGE_FRAGMENT:
        if (ts.alpha_func != 0) {
            info.alpha_test_func = static_cast<Shader::CompareFunction>(ts.alpha_func - 1);
        }
        info.alpha_test_reference = std::bit_cast<float>(ts.alpha_ref_bits);
        info.dual_source_blend = ts.dual_source_blend != 0;
        for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
            info.color_output_types[slot] =
                static_cast<Shader::AttributeType>(ts.color_output_types[slot]);
        }
        break;
    default:
        break;
    }
    info.input_topology = topology;
    info.force_early_z = false;
    info.y_negate = false;
    return info;
}

bool PipelineCache::Impl::Translate(const TranslateInput& in, TranslateOutput& out) {
    const auto reject = [&](u64 hash, u32 stage, std::string_view phase, std::string_view why) {
        if (!in.speculative) {
            Reject(hash, stage, phase, why);
        }
    };
    const bool is_compute = in.translation == nullptr;
    struct PoolGuard {
        VC::NvnShaderPools& pools;
        ~PoolGuard() {
            pools.ReleaseContents();
        }
    } guard{ThreadPools()};
    Shader::HostTranslateInfo current_host_info = host_info;
    current_host_info.raw_global_last_hit = VideoCommon::RawGlobalFastEnabled();
    current_host_info.raw_global_store_alias =
        Common::DiagRequest::TrySwitchEnabled("TRY_RO_MIRROR_EXACT") ||
        Common::DiagRequest::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH");
    const Shader::Profile& translation_profile = profile;

    boost::container::small_vector<u32, PIPELINE_ORDER.size()> order;
    if (is_compute) {
        order.push_back(STAGE_COMPUTE);
    } else {
        for (const u32 stage : PIPELINE_ORDER) {
            if (in.programs[stage] != nullptr) {
                order.push_back(stage);
            }
        }
    }

    std::array<LoweredEntry, STAGE_COUNT> lowered_programs;
    std::array<std::unique_ptr<VC::NvnEnvironment>, STAGE_COUNT> environments;
    std::array<Shader::IR::Program, STAGE_COUNT> programs;

    // Task and mesh programs are translated with the register words NVN2's program bind derives
    // from both controls (VC::MeshRegisters); a control that does not parse fails its lowering.
    std::array<u32, 3> mesh_registers{};
    if (!is_compute && in.programs[STAGE_MESH] != nullptr) {
        std::string error;
        const auto mesh = VC::ParseControl(*in.programs[STAGE_MESH]->control, error);
        std::optional<VC::NvnControl> task;
        if (in.programs[STAGE_TASK] != nullptr) {
            task = VC::ParseControl(*in.programs[STAGE_TASK]->control, error);
        }
        if (mesh) {
            mesh_registers = VC::MeshRegisters(task ? &task->mesh : nullptr, mesh->mesh);
        }
    }

    const auto answer = [&in](u32 stage, const VC::NvnHandleRecord& record) -> TextureInfo {
        if (in.oracle != nullptr) {
            return in.oracle->Query(stage, record);
        }
        if (in.environment != nullptr && in.environment->texture_info) {
            return in.environment->texture_info(record);
        }
        return {};
    };

    // 1. lower + translate every stage
    for (const u32 stage : order) {
        const StageProgram& source = *in.programs[stage];
        lowered_programs[stage] = GetLowered(source, stage);
        const auto& lowered_program = lowered_programs[stage].program;
        if (!lowered_program || !lowered_program->error.empty()) {
            reject(source.hash, stage, "lower",
                   lowered_program ? lowered_program->error : std::string{"no result"});
            return false;
        }
        VC::NvnEnvironmentOptions env_options;
        if (in.environment != nullptr) {
            env_options = *in.environment;
        }
        env_options.texture_info = [&answer, stage](const VC::NvnHandleRecord& record) {
            return answer(stage, record);
        };
        env_options.mesh_registers = mesh_registers;
        environments[stage] = std::make_unique<VC::NvnEnvironment>(lowered_program, env_options);
        const char* phase = "translate";
        try {
            programs[stage] = VC::TranslateNvnStage(ThreadPools(), *environments[stage],
                                                    current_host_info);
        } catch (const Shader::NotImplementedException& e) {
            reject(source.hash, stage, phase, std::string{"NotImplemented: "} + e.what());
            return false;
        } catch (const Shader::LogicError& e) {
            reject(source.hash, stage, phase, std::string{"Logic: "} + e.what());
            return false;
        } catch (const Shader::RuntimeError& e) {
            reject(source.hash, stage, phase, std::string{"Runtime: "} + e.what());
            return false;
        } catch (const Shader::InvalidArgument& e) {
            reject(source.hash, stage, phase, std::string{"InvalidArgument: "} + e.what());
            return false;
        } catch (const std::exception& e) {
            reject(source.hash, stage, phase, std::string{"Exception: "} + e.what());
            return false;
        }
        if (ShaderLog::Enabled()) {
            out.cbuf_used[stage] = programs[stage].info.constant_buffer_used_sizes;
            out.cbuf_known |= 1u << stage;
        }
        if (environments[stage]->HandleKeyOverflow()) {
            NoteIssue("texture handles: more distinct handle words than the synthetic key space");
        }
        if (programs[stage].info.requires_layer_emulation) {
            NoteIssue("layer emulation (geometry pass-through) is not supported by the NVN "
                      "replay");
        }
        if (const std::string limit = MeshLimitExceeded(device, programs[stage]); !limit.empty()) {
            reject(source.hash, stage, "limits", limit);
            return false;
        }
    }

    // 2. descriptor layout of the whole pipeline
    std::vector<VC::StageInfo> infos;
    infos.reserve(order.size());
    for (const u32 stage : order) {
        infos.push_back(VC::StageInfo{
            static_cast<VC::ShaderStage>(stage), programs[stage].info,
            is_compute ? &lowered_programs[stage].program->control.compute : nullptr});
    }
    VC::DescriptorLayout layout;
    try {
        layout = VC::BuildDescriptorLayout(infos,
                                           VC::LayoutOptions{profile.support_descriptor_aliasing});
    } catch (const std::exception& e) {
        reject(in.programs[order[0]]->hash, order[0], "layout", e.what());
        return false;
    }

    // 3. runtime info, SPIR-V, per-stage records
    const bool has_geometry = !is_compute && in.programs[STAGE_GEOMETRY] != nullptr &&
                              !programs[STAGE_GEOMETRY].is_geometry_passthrough;
    Shader::Backend::Bindings bindings;
    const Shader::IR::Program* previous = nullptr;
    std::array<u32, STAGE_COUNT> binding_after{};
    std::array<std::shared_ptr<Disk::ShaderRecord>, STAGE_COUNT> records;
    for (const u32 stage : order) {
        const StageProgram& source = *in.programs[stage];
        Shader::IR::Program& program = programs[stage];
        const Shader::RuntimeInfo runtime_info =
            MakeRuntimeInfo(stage, program, previous, in.translation, has_geometry);
        std::vector<u32> spirv;
        u64 runtime_hash{};
        try {
            ConvertLegacyToGeneric(program, runtime_info);
            runtime_hash =
                HashRuntime(stage, runtime_info, bindings, in.extra_key, mesh_registers);
            if (ShaderLog::Enabled()) {
                // What the emitter consumes (kept also when the emit below rejects the stage).
                // A failed dump only loses the text: it must not reach the catches below.
                try {
                    out.ir[stage] = Shader::IR::DumpProgram(program);
                } catch (const std::exception& e) {
                    static std::atomic<bool> warned{};
                    if (!warned.exchange(true, std::memory_order_relaxed)) {
                        LOG_WARNING(Render_Vulkan, "[nvn] shader log: IR dump of {:016x} failed: {}",
                                    source.hash, e.what());
                    }
                }
            }
            program.info.raw_coverage_program = program.info.raw_coverage ? source.hash : 0;
            spirv = EmitSPIRV(translation_profile, runtime_info, program, bindings);
        } catch (const Shader::NotImplementedException& e) {
            reject(source.hash, stage, "emit", std::string{"NotImplemented: "} + e.what());
            return false;
        } catch (const Shader::LogicError& e) {
            reject(source.hash, stage, "emit", std::string{"Logic: "} + e.what());
            return false;
        } catch (const Shader::RuntimeError& e) {
            reject(source.hash, stage, "emit", std::string{"Runtime: "} + e.what());
            return false;
        } catch (const std::exception& e) {
            reject(source.hash, stage, "emit", std::string{"Exception: "} + e.what());
            return false;
        }
        if (stage == STAGE_GEOMETRY) {
            // The input interface (generic locations of the previous stage plus the position
            // built-in) must fit maxGeometryInputComponents; the driver would reject the module.
            const u32 limit = device.GetPhysical().GetProperties().limits.maxGeometryInputComponents;
            const u32 components = GeometryInputComponents(program, runtime_info);
            if (components > limit) {
                reject(source.hash, stage, "limits",
                       fmt::format("geometry stage exceeds maxGeometryInputComponents ({} > {})",
                                   components, limit));
                return false;
            }
        }
        if (const std::string& spirv_dir = Debug::Get().dump_spirv; !spirv_dir.empty()) {
            // Diagnostics: the translated SPIR-V of every stage (compare with spirv-dis).
            std::error_code error;
            std::filesystem::create_directories(spirv_dir, error);
            const std::string path = fmt::format("{}/s{}_{:016x}_{:016x}.spv", spirv_dir, stage,
                                                 source.hash, runtime_hash);
            if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
                std::fwrite(spirv.data(), sizeof(u32), spirv.size(), file);
                std::fclose(file);
            }
        }
        previous = &program;
        binding_after[stage] = bindings.unified;

        auto record = std::make_shared<Disk::ShaderRecord>();
        record->stage = stage;
        record->program_hash = source.hash;
        for (const VC::NvnHandleRecord& handle : environments[stage]->HandleRecords()) {
            record->queries.push_back(TextureQuery{stage, handle, answer(stage, handle)});
        }
        record->id = Disk::MakeContentId(stage, source.hash, runtime_hash,
                                         Disk::HashQueries(record->queries));
        if (program.info.stores[Shader::IR::Attribute::ViewportIndex]) {
            record->flags |= Disk::SHADER_STORES_VIEWPORT_INDEX;
        }
        if (stage == STAGE_VERTEX) {
            for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
                if (program.info.loads.Generic(location)) {
                    record->attributes_read |= 1u << location;
                }
            }
        }
        if (stage == STAGE_FRAGMENT) {
            for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
                if (program.info.stores_frag_color[slot]) {
                    record->outputs_written |= 1u << slot;
                }
            }
        }
        if (is_compute) {
            record->workgroup_size = program.workgroup_size;
            record->shared_memory_size = program.shared_memory_size;
        }
        record->spirv = std::move(spirv);
        records[stage] = std::move(record);
    }
    if (bindings.unified != layout.binding_count) {
        ++stat_failures;
        NoteRow(fmt::format("layout: binding count mismatch (emit {} vs layout {})",
                            bindings.unified, layout.binding_count));
        return false;
    }

    // 4. split the layout per stage
    for (const u32 stage : order) {
        Disk::ShaderRecord& record = *records[stage];
        record.layout.binding_count = binding_after[stage];
        record.layout.uses_rescaling_uniform = programs[stage].info.uses_rescaling_uniform;
        record.layout.uses_render_area = programs[stage].info.uses_render_area;
        record.layout.uses_raw_global_memory = programs[stage].info.uses_raw_global_memory;
        record.layout.stores_raw_global_memory = programs[stage].info.stores_raw_global_memory;
        record.layout.raw_stores_written_through = VC::RawStoresWrittenThrough(programs[stage].info);
        record.layout.uses_bindless_table = std::ranges::any_of(
            programs[stage].info.texture_descriptors,
            [](const Shader::TextureDescriptor& desc) { return desc.bindless; });
        record.layout.requires_guest_warp =
            is_compute && tensor_lowering == VideoCommon::NativeTensorLowering::Compact &&
            programs[stage].info.UsesWarpOps();
        for (const VC::DescriptorBinding& binding : layout.bindings) {
            if (static_cast<u32>(binding.stage) != stage) {
                continue;
            }
            if (!binding.supported) {
                ++record.layout.unsupported_count;
            }
            if (binding.role == VC::DescriptorRole::ProgramConstants) {
                record.has_program_constants = true;
            }
            record.layout.bindings.push_back(binding);
        }
        if (record.has_program_constants) {
            record.program_constants = *lowered_programs[stage].c1;
            out.stages[stage].constants = lowered_programs[stage].c1;
        }
        out.stages[stage].record = std::move(records[stage]);
    }
    if (is_compute) {
        out.compute_layout = lowered_programs[STAGE_COMPUTE].program->control.compute;
    }
    return true;
}


// ================================================================================================
// lookups and compilation
// ================================================================================================

void PipelineCache::Impl::WaitPending(const std::shared_ptr<PendingKey>& pending) {
    std::unique_lock lock{pending->mutex};
    pending->cv.wait(lock, [&] { return pending->outstanding.load() == 0; });
}

bool PipelineCache::Impl::WaitLoaderQueued() {
    if (!loader.joinable()) {
        return false;
    }
    std::unique_lock lock{loader_mutex};
    if (loader_queued) {
        return false;
    }
    loader_cv.wait(lock, [this] { return loader_queued || stopping.load(); });
    return true;
}

void PipelineCache::Impl::NoteWait(std::chrono::steady_clock::time_point start) {
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count());
    const u64 count = ++stat_wait_count;
    stat_wait_ns += ns;
    u64 previous = stat_wait_max_ns.load();
    while (ns > previous && !stat_wait_max_ns.compare_exchange_weak(previous, ns)) {
    }
    if (ns >= 5'000'000 && count <= 8) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] a pipeline lookup waited {} ms for the preload or a key in flight "
                 "(wait #{})",
                 ns / 1'000'000, count);
    }
}

void PipelineCache::Impl::NoteNotReady(std::string_view what, std::span<const u64> hashes) {
    const u64 count = ++stat_async_not_ready;
    if (count > 16) {
        return;
    }
    std::string stages;
    for (const u64 hash : hashes) {
        stages += fmt::format(" {:016x}", hash);
    }
    LOG_INFO(Render_Vulkan, "[nvn] {} skipped awaiting async pipeline (#{}): stages{}", what,
             count, stages);
}

void PipelineCache::Impl::AsyncQueued() {
    if (options.shader_notify != nullptr) {
        options.shader_notify->MarkShaderBuilding();
    }
    const u64 now = static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count());
    u64 expected = 0;
    stat_async_first_ticks.compare_exchange_strong(expected, now);
    std::scoped_lock lock{async_mutex};
    ++async_outstanding;
    if (async_outstanding > stat_async_peak.load()) {
        stat_async_peak = async_outstanding;
    }
}

void PipelineCache::Impl::AsyncDone() {
    if (options.shader_notify != nullptr) {
        options.shader_notify->MarkShaderComplete();
    }
    stat_async_last_ticks = static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::scoped_lock lock{async_mutex};
        --async_outstanding;
    }
    async_cv.notify_all();
}

template <typename Map, typename Key>
void PipelineCache::Impl::ReleasePending(Map& map, const Key& key,
                                         const std::shared_ptr<PendingKey>& pending) {
    {
        std::scoped_lock lock{pipeline_mutex};
        if (--pending->outstanding == 0) {
            map.erase(key);
        }
    }
    {
        std::scoped_lock lock{pending->mutex};
    }
    pending->cv.notify_all();
}

void PipelineCache::Impl::RunAsyncGraphics(AsyncGraphicsJob& job) {
    std::shared_ptr<GraphicsPipeline> pipeline;
    if (!stopping.load()) {
        try {
            pipeline = CompileGraphics(job.request, job.key, job.had_variants, job.speculative);
        } catch (const std::exception& e) {
            ++stat_failures;
            NoteRow(fmt::format("pipeline: async compile failed: {}", FirstLine(e.what())));
        } catch (...) {
            ++stat_failures;
            NoteRow("pipeline: async compile failed: unknown exception");
        }
        if (!pipeline && !job.speculative) {
            ++stat_async_failed;
            std::scoped_lock lock{pipeline_mutex};
            failed_graphics.insert(job.key);
        }
    }
    // The pipeline (or the failed mark) is visible before the key stops being pending.
    ReleasePending(pending_graphics, job.key, job.pending);
    AsyncDone();
}

std::shared_ptr<GraphicsPipeline> PipelineCache::Impl::GetGraphics(const GraphicsRequest& source,
    bool* not_ready, u64& key_hash, const std::function<void(u64)>& look_ahead,
    std::optional<u64> prefetch_skip) {
    if (not_ready != nullptr) {
        *not_ready = false;
    }
    // The state words carry guest garbage in the bits no field covers: key on the canonical form
    // (the same form the disk records hold) and create the pipeline from it. In Dynamic mode that
    // form holds the vertex type classes instead of the vertex layout.
    GraphicsRequest request = source;
    MakeKeyState(request.state, vertex_input_mode, must_emulate_scaled_formats);
    const TextureOracle& oracle = request.oracle != nullptr ? *request.oracle : DEFAULT_ORACLE;
    GraphicsKey key;
    for (size_t i = 0; i < GRAPHICS_STAGE_COUNT; ++i) {
        key.hashes[i] = request.stages[i].hash;
    }
    key.state = request.state;
    key.try_switch_hash = TrySwitchesHash();
    key.targets = request.targets;
    key.Finalize();
    key_hash = key.hash;

    const bool prefetch = prefetch_skip.has_value();
    if (prefetch && (key_hash == *prefetch_skip || !options.prefetch_graphics ||
                     !loader_queued.load(std::memory_order_acquire))) {
        return nullptr;
    }
    if (prefetch) {
        // Bound speculative work so a later frame cannot build up a backlog behind it.
        std::scoped_lock lock{async_mutex};
        if (async_outstanding >= async_worker_count) {
            return nullptr;
        }
    }

    // A draw that may be skipped (`not_ready`) with asynchronous shaders on is compiled on the
    // workers, the GPL path included: translating, compiling the library parts and linking all
    // run there, and the draw waits for the pipeline like the default path's. Any other lookup
    // (a draw that must not be skipped, asynchronous shaders off) builds on this thread.
    const bool gpl = GplEnabled();
    const bool async = prefetch || AsyncEnabled(not_ready);
    // Read before the search: once the preload has registered every cached key, a miss below is
    // not about to be delivered by it, so the key may be compiled instead of waited for.
    const bool preload_registered = !async || loader_queued.load(std::memory_order_acquire);
    std::shared_ptr<PendingKey> pending;
    std::shared_ptr<PendingKey> claim; ///< async: this lookup owns the compile of the key
    bool had_variants = false;
    bool failed = false;
    const auto search = [&]() -> std::shared_ptr<GraphicsPipeline> {
        pending.reset();
        std::scoped_lock lock{pipeline_mutex};
        if (const auto it = graphics.find(key); it != graphics.end()) {
            for (const auto& candidate : it->second) {
                if (candidate->QueriesMatch(oracle)) {
                    if (!prefetch && candidate->prefetched_unused.exchange(false)) {
                        ++stat_prefetch_used;
                    }
                    return candidate;
                }
            }
            had_variants = !it->second.empty();
        }
        if (const auto it = pending_graphics.find(key); it != pending_graphics.end()) {
            pending = it->second;
        } else if (async && preload_registered) {
            if (failed_graphics.contains(key)) {
                failed = true;
            } else {
                // Claimed under the same lock as the search: no other lookup or finished worker
                // can slip in between.
                claim = std::make_shared<PendingKey>();
                claim->speculative = prefetch;
                claim->outstanding = 1;
                pending_graphics.emplace(key, claim);
            }
        }
        if (gpl && failed_graphics.contains(key)) {
            failed = true;
        }
        return nullptr;
    };
    if (auto hit = search()) {
        return hit;
    }
    if (async) {
        if (!preload_registered || pending) {
            if (!prefetch) {
                *not_ready = true;
                NoteNotReady("draw", key.hashes);
            }
            return nullptr;
        }
        if (failed) {
            if (!prefetch) {
                NoteRow("pipeline: draw skipped, the pipeline was rejected earlier");
            }
            return nullptr;
        }
        SCOPE_EXIT {
            if (claim) {
                ReleasePending(pending_graphics, key, claim);
            }
        };
        if (AnyStageRejected(key.hashes)) {
            return nullptr;
        }
        u32 stage_mask = 0;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            if (request.stages[stage].hash != 0) {
                stage_mask |= 1u << stage;
            }
        }
        std::unique_ptr<TextureOracle> frozen = oracle.Freeze(stage_mask);
        if (!frozen) {
            if (prefetch) {
                return nullptr;
            }
            // An oracle that cannot be copied cannot be used by a worker: compile here.
            return CompileGraphics(request, key, had_variants);
        }
        auto job = std::make_shared<AsyncGraphicsJob>();
        job->request.state = request.state;
        job->request.targets = request.targets;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            const StageProgram& program = request.stages[stage];
            if (program.hash == 0) {
                continue;
            }
            StageProgram& owned = job->request.stages[stage];
            owned.hash = program.hash;
            if (program.control != nullptr && program.code != nullptr) {
                job->programs[stage].control = *program.control;
                job->programs[stage].code = *program.code;
                owned.control = &job->programs[stage].control;
                owned.code = &job->programs[stage].code;
            }
        }
        job->oracle = std::move(frozen);
        job->request.oracle = job->oracle.get();
        job->key = key;
        job->had_variants = had_variants;
        job->speculative = prefetch;
        job->pending = claim;
        AsyncQueued();
        try {
            async_workers->QueueWork([this, job] { RunAsyncGraphics(*job); });
        } catch (...) {
            AsyncDone();
            throw;
        }
        claim.reset();
        ++stat_async_queued;
        if (prefetch) {
            ++stat_prefetch_queued;
        }
        if (!prefetch) {
            *not_ready = true;
            NoteNotReady("draw", key.hashes);
        }
        return nullptr;
    }
    // A miss while the loader is still registering the cached pipelines: wait for it, then
    // look again (the key may be about to be delivered).
    if (const auto wait_start = std::chrono::steady_clock::now(); WaitLoaderQueued()) {
        NoteWait(wait_start);
        if (auto hit = search()) {
            return hit;
        }
    }
    bool scanned_ahead = false;
    const auto scan_ahead = [&] {
        if (scanned_ahead || failed || !options.prefetch_graphics || !look_ahead) {
            return;
        }
        {
            std::scoped_lock lock{async_mutex};
            if (async_outstanding >= async_worker_count) {
                return;
            }
        }
        scanned_ahead = true;
        try {
            look_ahead(key_hash);
        } catch (const std::exception& error) {
            NoteIssue(fmt::format("pipeline: graphics lookahead failed: {}", FirstLine(error.what())));
        }
    };
    if (pending) {
        // Keep the other workers busy while replay waits for this speculative build.
        if (pending->speculative) {
            scan_ahead();
        }
        const auto wait_start = std::chrono::steady_clock::now();
        WaitPending(pending);
        NoteWait(wait_start);
        if (auto hit = search()) {
            return hit;
        }
    }
    if (failed) {
        NoteRow("pipeline: draw skipped, the pipeline was rejected earlier");
        return nullptr;
    }
    if (!had_variants) {
        scan_ahead();
    }
    auto result = CompileGraphics(request, key, had_variants);
    if (!result && gpl) {
        // Not built again for every later draw of the key (the asynchronous path does the same).
        std::scoped_lock lock{pipeline_mutex};
        failed_graphics.insert(key);
    }
    return result;
}

void PipelineCache::Impl::LogRuntimeCompile(const GraphicsKey& key, bool had_variants) {
    // Cached pipelines of the same programs and targets: the state words that differ show which
    // field of the key is not stable between launches.
    std::string diff;
    size_t same_programs = 0;
    {
        std::scoped_lock lock{pipeline_mutex};
        for (const auto& [other, candidates] : graphics) {
            if (other.hashes != key.hashes || other.targets != key.targets) {
                continue;
            }
            ++same_programs;
            if (!diff.empty()) {
                continue;
            }
            const auto* a = reinterpret_cast<const u32*>(&key.state);
            const auto* b = reinterpret_cast<const u32*>(&other.state);
            for (size_t i = 0; i < sizeof(GraphicsFixedState) / sizeof(u32); ++i) {
                if (a[i] != b[i]) {
                    diff += fmt::format(" w{}:{:#x}->{:#x}", i, b[i], a[i]);
                }
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] run-time compile: stages {:x} {:x} {:x} {:x} {:x} variants_of_key={} "
             "same-program entries={} differing state words (cached->new):{}",
             key.hashes[0], key.hashes[1], key.hashes[2], key.hashes[3], key.hashes[4], had_variants,
             same_programs, diff.empty() ? " (none: program/target combination is new)" : diff);
}

std::shared_ptr<GraphicsPipeline> PipelineCache::Impl::CompileGraphics(
    const GraphicsRequest& request, const GraphicsKey& key, bool had_variants, bool speculative) {
    const auto start = std::chrono::steady_clock::now();
    CompileFailureNote failure{*this, key.hash};
    if (AnyStageRejected(key.hashes)) {
        return nullptr;
    }
    if (Debug::Get().cache_log) {
        LogRuntimeCompile(key, had_variants);
    }
    // A vertex pipeline has a vertex program, a task/mesh pipeline a mesh program.
    if (request.stages[STAGE_VERTEX].hash == 0 && request.stages[STAGE_MESH].hash == 0) {
        ++stat_failures;
        NoteRow("pipeline: request without a vertex or mesh stage");
        return nullptr;
    }
    const TranslationState translation_state = MakeTranslationState(
        request.state, request.targets, vertex_input_mode, must_emulate_scaled_formats);
    TranslateInput input;
    input.speculative = speculative;
    input.translation = &translation_state;
    input.oracle = request.oracle != nullptr ? request.oracle : &DEFAULT_ORACLE;
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        const StageProgram& program = request.stages[stage];
        if (program.hash == 0) {
            continue;
        }
        if (program.control == nullptr || program.code == nullptr) {
            ++stat_failures;
            NoteRow(fmt::format("pipeline: {} stage without program bytes", StageNameOf(stage)));
            return nullptr;
        }
        input.programs[stage] = &program;
    }
    // GPL path: pipelines whose state differs only in what the translation does not read share
    // their translated stages (and with them the shader library parts). Several workers compile
    // pipelines of one key at once: the first translates, the others wait for it and reuse the set.
    std::shared_ptr<ShaderSet> gpl_set;
    TranslationKey translation_key;
    TranslateOutput fresh;
    bool reused = false;
    const auto translate_start = std::chrono::steady_clock::now();
    if (GplEnabled()) {
        translation_key.hashes = key.hashes;
        translation_key.state = translation_state;
        translation_key.try_switch_hash = key.try_switch_hash;
        translation_key.Finalize();
        gpl_set = FindShaderSet(translation_key, *input.oracle);
        reused = gpl_set != nullptr;
        if (!gpl_set) {
            const TranslationHold hold{*this, translation_key};
            gpl_set = FindShaderSet(translation_key, *input.oracle);
            reused = gpl_set != nullptr; // another worker translated it while this one waited
            if (!gpl_set) {
                if (!Translate(input, fresh)) {
                    failure.ir = std::move(fresh.ir);
                    return nullptr;
                }
                gpl_set = AddShaderSet(translation_key, std::move(fresh));
            }
        }
    } else if (!Translate(input, fresh)) {
        failure.ir = std::move(fresh.ir);
        return nullptr;
    }
    const TranslateOutput& translated = gpl_set ? gpl_set->translated : fresh;
    std::array<const StageArtifact*, GRAPHICS_STAGE_COUNT> stages{};
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        if (translated.stages[stage].record) {
            stages[stage] = &translated.stages[stage];
        }
    }
    const auto create_start = std::chrono::steady_clock::now();
    auto pipeline = AssembleGraphics(request.state, request.targets, stages, gpl_set);
    const auto create_end = std::chrono::steady_clock::now();
    if (!pipeline) {
        failure.ir = gpl_set ? gpl_set->translated.ir : std::move(fresh.ir);
        return nullptr;
    }
    {
        std::scoped_lock lock{pipeline_mutex};
        auto& variants = graphics[key];
        if (ShaderLog::Enabled()) {
            // A shared set keeps its IR text for the next pipeline built from it.
            TranslateOutput shared_copy;
            if (gpl_set) {
                shared_copy = gpl_set->translated;
            }
            LogCompileDone(*pipeline, gpl_set ? shared_copy : fresh,
                           std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count(),
                           static_cast<u32>(variants.size()));
        }
        pipeline->prefetched_unused = speculative;
        variants.push_back(pipeline);
    }
    failure.armed = false;
    ++stat_compiled;
    if (had_variants) {
        ++stat_retranslated;
    }
    if (reused) {
        ++stat_gpl_reused;
    }
    if (pipeline->fast_linked) {
        QueueOptimize(pipeline, gpl_set, request.state, request.targets);
    }
    NoteCompileTime(speculative             ? "graphics prefetch"
                    : !gpl_set               ? "graphics"
                    : pipeline->fast_linked ? "graphics fast-linked"
                                            : "graphics whole (gpl)",
                    key.hash, start, create_start - translate_start, create_end - create_start);
    if (DiskEnabled()) {
        Disk::GraphicsPipelineRecord record;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            if (stages[stage] != nullptr) {
                PostShader(*stages[stage]);
                record.shaders[stage] = stages[stage]->record->id;
            }
        }
        record.state = request.state;
        record.targets = request.targets;
        record.push_constant_size = pipeline->push_constant_size;
        record.viewport_count = pipeline->viewport_count;
        PostGraphicsRecord(record);
        ++new_since_vk_write;
        MaybeWriteVkCache();
    }
    return pipeline;
}

void PipelineCache::Impl::RunAsyncCompute(AsyncComputeJob& job) {
    std::shared_ptr<ComputePipeline> pipeline;
    if (!stopping.load()) {
        try {
            pipeline = CompileCompute(job.request, job.key, job.had_variants);
        } catch (const std::exception& e) {
            ++stat_failures;
            NoteRow(fmt::format("pipeline: async compile failed: {}", FirstLine(e.what())));
        } catch (...) {
            ++stat_failures;
            NoteRow("pipeline: async compile failed: unknown exception");
        }
        if (!pipeline) {
            ++stat_async_failed;
            std::scoped_lock lock{pipeline_mutex};
            failed_compute.insert(job.key);
        }
    }
    ReleasePending(pending_compute, job.key, job.pending);
    AsyncDone();
}

std::shared_ptr<ComputePipeline> PipelineCache::Impl::GetCompute(const ComputeRequest& request,
                                                                 bool* not_ready) {
    if (not_ready != nullptr) {
        *not_ready = false;
    }
    const TextureOracle& oracle = request.oracle != nullptr ? *request.oracle : DEFAULT_ORACLE;
    ComputeKey key;
    key.try_switch_hash = TrySwitchesHash();
    key.hash = request.program.hash;
    if (request.environment.compute_local_memory_size) {
        key.environment_flags |= ENV_LOCAL_MEMORY;
        key.local_memory_size = *request.environment.compute_local_memory_size;
    }
    if (request.environment.compute_shared_memory_size) {
        key.environment_flags |= ENV_SHARED_MEMORY;
        key.shared_memory_size = *request.environment.compute_shared_memory_size;
    }
    const bool async = AsyncEnabled(not_ready);
    const bool preload_registered = !async || loader_queued.load(std::memory_order_acquire);
    std::shared_ptr<PendingKey> pending;
    std::shared_ptr<PendingKey> claim;
    bool had_variants = false;
    bool failed = false;
    const auto search = [&]() -> std::shared_ptr<ComputePipeline> {
        std::scoped_lock lock{pipeline_mutex};
        if (const auto it = compute.find(key); it != compute.end()) {
            for (const auto& candidate : it->second) {
                if (candidate->QueriesMatch(oracle)) {
                    return candidate;
                }
            }
            had_variants = !it->second.empty();
        }
        if (const auto it = pending_compute.find(key); it != pending_compute.end()) {
            pending = it->second;
        } else if (async && preload_registered) {
            if (failed_compute.contains(key)) {
                failed = true;
            } else {
                claim = std::make_shared<PendingKey>();
                claim->outstanding = 1;
                pending_compute.emplace(key, claim);
            }
        }
        return nullptr;
    };
    if (auto hit = search()) {
        return hit;
    }
    if (async) {
        const u64 hashes[1]{key.hash};
        if (!preload_registered || pending) {
            *not_ready = true;
            NoteNotReady("dispatch", hashes);
            return nullptr;
        }
        if (failed) {
            NoteRow("pipeline: dispatch skipped, the pipeline was rejected earlier");
            return nullptr;
        }
        if (AnyStageRejected(hashes)) {
            ReleasePending(pending_compute, key, claim);
            return nullptr;
        }
        std::unique_ptr<TextureOracle> frozen = oracle.Freeze(1u << STAGE_COMPUTE);
        if (!frozen) {
            auto result = CompileCompute(request, key, had_variants);
            ReleasePending(pending_compute, key, claim);
            return result;
        }
        auto job = std::make_shared<AsyncComputeJob>();
        job->request.program.hash = request.program.hash;
        if (request.program.control != nullptr && request.program.code != nullptr) {
            job->program.control = *request.program.control;
            job->program.code = *request.program.code;
            job->request.program.control = &job->program.control;
            job->request.program.code = &job->program.code;
        }
        job->request.environment = request.environment;
        job->request.environment.texture_info = nullptr; // the frozen oracle answers
        job->oracle = std::move(frozen);
        job->request.oracle = job->oracle.get();
        job->key = key;
        job->had_variants = had_variants;
        job->pending = std::move(claim);
        AsyncQueued();
        try {
            async_workers->QueueWork([this, job] { RunAsyncCompute(*job); });
        } catch (...) {
            AsyncDone();
            ReleasePending(pending_compute, key, job->pending);
            throw;
        }
        ++stat_async_queued;
        *not_ready = true;
        NoteNotReady("dispatch", hashes);
        return nullptr;
    }
    if (const auto wait_start = std::chrono::steady_clock::now(); WaitLoaderQueued()) {
        NoteWait(wait_start);
        pending.reset();
        if (auto hit = search()) {
            return hit;
        }
    }
    if (pending) {
        const auto wait_start = std::chrono::steady_clock::now();
        WaitPending(pending);
        NoteWait(wait_start);
        pending.reset();
        if (auto hit = search()) {
            return hit;
        }
    }
    return CompileCompute(request, key, had_variants);
}

std::shared_ptr<ComputePipeline> PipelineCache::Impl::CompileCompute(const ComputeRequest& request,
                                                                     const ComputeKey& key,
                                                                     bool had_variants) {
    const auto start = std::chrono::steady_clock::now();
    CompileFailureNote failure{*this, LogBaseKey(key)};
    const u64 hashes[1]{request.program.hash};
    if (AnyStageRejected(hashes)) {
        return nullptr;
    }
    if (request.program.hash == 0 || request.program.control == nullptr ||
        request.program.code == nullptr) {
        ++stat_failures;
        NoteRow("pipeline: compute request without a program");
        return nullptr;
    }
    TranslateInput input;
    input.programs[STAGE_COMPUTE] = &request.program;
    input.oracle = request.oracle != nullptr ? request.oracle : &DEFAULT_ORACLE;
    input.environment = &request.environment;
    input.extra_key = key.environment_flags | (u64{key.local_memory_size} << 8) |
                      (u64{key.shared_memory_size} << 40);
    TranslateOutput translated;
    const auto translate_start = std::chrono::steady_clock::now();
    if (!Translate(input, translated)) {
        failure.ir = std::move(translated.ir);
        return nullptr;
    }
    const StageArtifact& stage = translated.stages[STAGE_COMPUTE];
    const auto create_start = std::chrono::steady_clock::now();
    auto pipeline = AssembleCompute(stage, translated.compute_layout);
    const auto create_end = std::chrono::steady_clock::now();
    if (!pipeline) {
        failure.ir = std::move(translated.ir);
        return nullptr;
    }
    {
        std::scoped_lock lock{pipeline_mutex};
        auto& variants = compute[key];
        if (ShaderLog::Enabled()) {
            LogCompileDone(*pipeline, translated,
                           std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count(),
                           static_cast<u32>(variants.size()));
        }
        variants.push_back(pipeline);
    }
    failure.armed = false;
    ++stat_compiled;
    if (had_variants) {
        ++stat_retranslated;
    }
    NoteCompileTime("compute", request.program.hash, start, create_start - translate_start,
                    create_end - create_start);
    if (DiskEnabled()) {
        PostShader(stage);
        Disk::ComputePipelineRecord record;
        record.shader = stage.record->id;
        record.push_constant_size = pipeline->push_constant_size;
        record.environment_flags = key.environment_flags;
        record.local_memory_size = key.local_memory_size;
        record.shared_memory_size = key.shared_memory_size;
        record.compute_layout = translated.compute_layout;
        PostComputeRecord(record);
        ++new_since_vk_write;
        MaybeWriteVkCache();
    }
    return pipeline;
}


// ================================================================================================
// public wrappers
// ================================================================================================

PipelineCache::PipelineCache(const Device& device, PipelineCacheOptions options)
    : impl{std::make_unique<Impl>(device, std::move(options))} {}

PipelineCache::~PipelineCache() = default;

void PipelineCache::PrefetchGraphics(const GraphicsRequest& request, u64 current_key) {
    u64 key_hash{};
    (void)impl->GetGraphics(request, nullptr, key_hash, {}, current_key);
}

std::shared_ptr<GraphicsPipeline> PipelineCache::GetGraphics(const GraphicsRequest& request,
    bool* not_ready, const std::function<void(u64)>& look_ahead) {
    const auto begin = Common::FrameStats::Enabled() ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
    u64 key_hash{};
    auto result = impl->GetGraphics(request, not_ready, key_hash, look_ahead);
    if (Common::FrameStats::Enabled()) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
        if (ns > 200'000) { // lookups are a few µs; only compiles and cache loads count
            Common::FrameStats::AddStall(Common::FrameStats::Cause::PipelineCompile, static_cast<u64>(ns));
        }
    }
    if (ShaderLog::Enabled()) {
        impl->LogGraphicsUse(request, result.get(), key_hash, not_ready != nullptr && *not_ready);
    }
    return result;
}

std::shared_ptr<ComputePipeline> PipelineCache::GetCompute(const ComputeRequest& request,
                                                           bool* not_ready) {
    const auto begin = Common::FrameStats::Enabled() ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
    auto result = impl->GetCompute(request, not_ready);
    if (Common::FrameStats::Enabled()) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
        if (ns > 200'000) {
            Common::FrameStats::AddStall(Common::FrameStats::Cause::PipelineCompile, static_cast<u64>(ns));
        }
    }
    if (ShaderLog::Enabled()) {
        impl->LogComputeUse(request, result.get(), not_ready != nullptr && *not_ready);
    }
    return result;
}

void PipelineCache::LogComputeFallback(const ComputeRequest& request, std::string_view reason) {
    impl->LogComputeFallback(request, reason);
}

void PipelineCache::PrepareProgram(const std::array<StageProgram, STAGE_COUNT>& stages,
                                   std::shared_ptr<const void> owner) {
    impl->PrepareProgram(stages, std::move(owner));
}

bool PipelineCache::UsesGraphicsPipelineLibrary() const noexcept {
    return impl->GplEnabled();
}

bool PipelineCache::SyncDiet() const noexcept {
    return impl->options.sync_diet;
}

VkRenderPass PipelineCache::GetRenderPass(const RenderPassKey& key) {
    return impl->GetRenderPass(key);
}

const Shader::Profile& PipelineCache::Profile() const noexcept {
    return impl->profile;
}

VertexInputMode PipelineCache::VertexInput() const noexcept {
    return impl->vertex_input_mode;
}

void PipelineCache::NoteIssue(const std::string& reason) {
    impl->NoteIssue(reason);
}

void PipelineCache::Flush() {
    impl->Flush();
}

void PipelineCache::ClearShaders() {
    impl->ClearShaders();
}

u64 PipelineCache::ShaderGeneration() const noexcept {
    return impl->shader_generation.load(std::memory_order_relaxed);
}

PipelineCacheStats PipelineCache::Stats() const {
    return impl->Stats();
}

void PipelineCache::CollectRows(std::vector<Row>& out) const {
    impl->CollectRows(out);
}

} // namespace Vulkan::Nvn
