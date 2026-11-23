// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_PATHDIFF on the NVN HLE path (video_core/pathdiff.h): what a selected dispatch or draw
// reads, copied in GPU order from the very buffers its descriptors name (BuildDescriptors'
// `scratch_writes`), the bytes behind every mapped pointer in its constant buffers, its texture
// and sampler state, and after it what it may have written. The emulated path writes the same
// records (vk_pathdiff.cpp); tools/pathdiff.py names the first input that differs.

#include <algorithm>
#include <cstring>

#include <fmt/format.h>

#include "common/logging.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/raw_coverage_layout.h"
#include "video_core/nvn/shader/nvn_program_map.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"

namespace Vulkan::Nvn {

namespace PD = VideoCommon::PathDiff;

namespace {

constexpr std::array<const char*, STAGE_COUNT> STAGE_NAMES{"vs",  "fs", "gs",   "tcs",
                                                           "tes", "cs", "task", "mesh"};

/// The constant buffer a uniform binding stands for (c0 driver bank, c1 program constants, c2
/// handle tables, c3+ the game's).
u32 CbufIndexOf(const VC::DescriptorBinding& binding, u32 element) {
    switch (binding.role) {
    case VC::DescriptorRole::DriverBank:
        return 0;
    case VC::DescriptorRole::ProgramConstants:
        return 1;
    case VC::DescriptorRole::HandleTable:
        return 2;
    default:
        return binding.slot + element + 3;
    }
}

std::string StagePrefix(bool graphics, u32 stage) {
    return graphics && stage < STAGE_NAMES.size() ? fmt::format("{}.", STAGE_NAMES[stage])
                                                  : std::string{};
}

} // Anonymous namespace

std::vector<u8> DrawContext::PathDiffUniformBytes(u64 address, u64 size) const {
    std::vector<u8> bytes(size);
    if (const u8* source = impl.GuestPointer(address, size)) {
        std::memcpy(bytes.data(), source, size);
    }
    if (host != nullptr) {
        impl.ApplyPendingInlineWrites(address, bytes.data(), size, host->HostFlight().seq);
    }
    for (const InlineUpdate& update : inline_updates) {
        const u64 begin = (std::max)(update.address, address);
        const u64 end = (std::min)(update.address + update.size, address + size);
        if (begin < end) {
            std::memcpy(bytes.data() + (begin - address), update.data + (begin - update.address),
                        end - begin);
        }
    }
    return bytes;
}

std::optional<InFlight::PathDiff> DrawContext::PathDiffBegin(
    const PipelineBase& pipe, bool graphics, const std::array<u64, STAGE_COUNT>& stage_hashes,
    const BufferRef* indirect, u32 indirect_size) {
    const PD::Spec& spec = PD::Config();
    const u64 present = impl.presents_recorded + 1;
    if (present < spec.present || pathdiff_selector.Done()) {
        return std::nullopt;
    }
    // The selected stage: the first whose program is listed (by stage hash or code key).
    u32 stage = STAGE_COUNT;
    u64 code_key = 0;
    for (u32 s = 0; s < STAGE_COUNT && stage == STAGE_COUNT; ++s) {
        const u64 hash = stage_hashes[s];
        if (hash == 0 || (graphics ? s == STAGE_COMPUTE : s != STAGE_COMPUTE)) {
            continue;
        }
        u64 key = PD::CodeKeyOf(hash);
        if (key == 0) {
            if (const auto it = impl.programs.find(stages[s].program_serial);
                it != impl.programs.end() && !it->second->code[s].empty()) {
                const u32 header = s == STAGE_COMPUTE ? 0 : VC::GraphicsHeaderBytes;
                const std::vector<u8>& code = it->second->code[s];
                if (code.size() > header) {
                    key = PD::CodeKey(std::span<const u8>{code}.subspan(header), header);
                    PD::NoteCodeKey(hash, key);
                }
            }
        }
        if (std::ranges::any_of(spec.hashes,
                                [&](u64 h) { return h == hash || (key != 0 && h == key); })) {
            stage = s;
            code_key = key;
        }
    }
    if (stage == STAGE_COUNT) {
        return std::nullopt;
    }
    const std::optional<u32> ordinal = pathdiff_selector.Claim(present, stage_hashes[stage], code_key);
    if (!ordinal) {
        return std::nullopt;
    }

    InFlight::PathDiff pending;
    PD::Capture& capture = pending.capture;
    capture.path = PD::Path::Hle;
    capture.present = present;
    capture.ordinal = *ordinal;
    capture.kind = graphics ? "draw" : "dispatch";
    capture.stage = STAGE_NAMES[stage];
    capture.hash = stage_hashes[stage];
    capture.code_key = code_key;
    capture.raw_global = pipe.descriptors.uses_raw_global_memory;

    // Copies from GPU buffers, made before (and for what may be written, after) the op.
    struct Source {
        u32 range;
        VkBuffer buffer;
        u64 offset;
        u64 size;
        bool after;
    };
    std::vector<Source> sources;
    const auto add_gpu = [&](PD::Range&& range, VkBuffer buffer, u64 offset, u64 size,
                             bool after) {
        const u32 index = static_cast<u32>(capture.ranges.size());
        range.before.resize(size);
        range.has_after = after;
        if (after) {
            range.after.resize(size);
        }
        capture.ranges.push_back(std::move(range));
        sources.push_back({index, buffer, offset, size, after});
    };
    // CPU views of the constant buffers, to find the pointers in them.
    struct CpuView {
        u32 cbuf;
        std::string prefix;
        std::vector<u8> bytes;
    };
    std::vector<CpuView> cpu_views;

    const auto binding_of = [&](u32 number) -> const VC::DescriptorBinding* {
        for (const VC::DescriptorBinding& binding : pipe.descriptors.bindings) {
            if (binding.binding == number) {
                return &binding;
            }
        }
        return nullptr;
    };
    u32 storage_order = 0;
    for (const VkWriteDescriptorSet& write : scratch_writes) {
        if (write.pBufferInfo == nullptr) {
            continue;
        }
        const VC::DescriptorBinding* binding = binding_of(write.dstBinding);
        if (binding == nullptr) {
            continue;
        }
        const u32 bstage = static_cast<u32>(binding->stage);
        const std::string prefix = StagePrefix(graphics, bstage);
        for (u32 i = 0; i < write.descriptorCount; ++i) {
            const VkDescriptorBufferInfo& info = write.pBufferInfo[i];
            PD::Range range;
            if (binding->kind == VC::DescriptorKind::UniformBuffer) {
                const u32 cbuf = CbufIndexOf(*binding, i);
                range.role = "cbuf";
                range.index = cbuf;
                range.source = prefix + PD::Source(cbuf);
                if (binding->role == VC::DescriptorRole::UniformBuffer && bstage < BIND_STAGES &&
                    binding->slot + i < stages[bstage].uniform.size()) {
                    const BufferBinding& bound = stages[bstage].uniform[binding->slot + i];
                    range.address = bound.address;
                    range.bound_size = bound.size;
                    if (bound.address != 0 && bound.size != 0) {
                        cpu_views.push_back({cbuf, prefix,
                                             PathDiffUniformBytes(bound.address,
                                                                  (std::min<u64>)(bound.size,
                                                                                  info.range))});
                    }
                } else if (binding->role == VC::DescriptorRole::DriverBank) {
                    range.bound_size = DRIVER_BANK_SIZE;
                    if (!graphics) {
                        std::vector<u8> bank(DRIVER_BANK_SIZE);
                        FillComputeBank(pipe.compute_layout, bank.data());
                        cpu_views.push_back({0, prefix, std::move(bank)});
                    }
                    if (capture.raw_global) {
                        capture.path_specific.push_back(
                            {0, Shader::Profile::RAW_GLOBAL_TABLE_OFFSET, 4,
                             "HLE raw global table entry count"});
                        capture.path_specific.push_back(
                            {0, Shader::Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET, 8,
                             "HLE raw global table device address"});
                        capture.path_specific.push_back(
                            {0, Shader::RawCoverage::BANK_RING_OFFSET, 8,
                             "HLE raw coverage ring address"});
                    }
                }
                if (binding->role == VC::DescriptorRole::ProgramConstants ||
                    info.range == VK_WHOLE_SIZE) {
                    // c1 is the program's own constant section: the guest bytes of the program.
                    range.view = "guest";
                    if (const auto it = impl.programs.find(stages[bstage].program_serial);
                        it != impl.programs.end()) {
                        std::string error;
                        const std::vector<u8>& code = it->second->code[bstage];
                        if (const auto control = VC::ParseControl(it->second->control[bstage], error);
                            control && control->c1_offset <= code.size()) {
                            range.before.assign(code.begin() + control->c1_offset,
                                                code.begin() + (std::min<size_t>)(
                                                                   control->code_bytes, code.size()));
                        }
                    }
                    range.bound_size = range.before.size();
                    capture.ranges.push_back(std::move(range));
                    continue;
                }
                add_gpu(std::move(range), info.buffer, info.offset,
                        (std::min<u64>)(info.range, spec.max_range), false);
            } else if (binding->kind == VC::DescriptorKind::StorageBuffer) {
                range.role = "storage";
                range.index = storage_order++;
                range.written = binding->is_written;
                const u32 slot = binding->slot + i;
                BufferBinding resolved{};
                switch (binding->storage_source) {
                case VC::StorageSource::DriverTable:
                    if (bstage < BIND_STAGES && slot < stages[bstage].storage.size()) {
                        resolved = stages[bstage].storage[slot];
                    }
                    break;
                case VC::StorageSource::ComputeStorageEntry:
                    if (slot < stages[STAGE_COMPUTE].storage.size()) {
                        resolved = stages[STAGE_COMPUTE].storage[slot];
                    }
                    break;
                case VC::StorageSource::ComputeUniformEntry:
                    if (slot < stages[STAGE_COMPUTE].uniform.size()) {
                        resolved = stages[STAGE_COMPUTE].uniform[slot];
                    }
                    break;
                case VC::StorageSource::UniformBufferWord: {
                    const u32 offset = binding->storage_offset + i * VC::StorageBufferStride;
                    const auto lo = ReadUniformWord(bstage, binding->slot, offset);
                    const auto hi = ReadUniformWord(bstage, binding->slot, offset + 4);
                    const auto size = ReadUniformWord(bstage, binding->slot, offset + 8);
                    if (lo && hi && size) {
                        resolved = {static_cast<u64>(*lo) | static_cast<u64>(*hi) << 32, *size};
                    }
                    break;
                }
                default:
                    break;
                }
                range.source =
                    binding->storage_source == VC::StorageSource::None
                        ? prefix + "none"
                        : prefix + PD::Source(binding->storage_source ==
                                                      VC::StorageSource::UniformBufferWord
                                                  ? binding->slot + 3
                                                  : binding->storage_bank,
                                              binding->storage_offset + i * VC::StorageBufferStride);
                range.address = resolved.address;
                range.bound_size = resolved.size;
                add_gpu(std::move(range), info.buffer, info.offset,
                        (std::min<u64>)(info.range, spec.max_range), true);
            }
        }
    }
    if (indirect != nullptr && indirect->buffer) {
        PD::Range range;
        range.role = "indirect";
        range.source = "indirect";
        range.bound_size = indirect_size;
        add_gpu(std::move(range), indirect->buffer, indirect->offset, indirect_size, false);
    }

    // The bytes behind every pointer of the constant buffers that is an address of this path.
    u32 pointer_order = 0;
    for (const CpuView& view : cpu_views) {
        if (pointer_order >= spec.max_pointers) {
            break;
        }
        const auto pointers = PD::FindPointers(
            view.bytes,
            [&](u64 value) {
                Pool* pool{};
                u64 offset{};
                return impl.ResolveBuffer(value, 1, pool, offset);
            },
            spec.max_pointers - pointer_order);
        for (const PD::Pointer& pointer : pointers) {
            Pool* pool{};
            u64 offset{};
            if (!impl.ResolveBuffer(pointer.value, 1, pool, offset)) {
                continue;
            }
            const u64 size = (std::min<u64>)(spec.chase_bytes, pool->size - offset) & ~u64{3};
            if (size == 0 ||
                !host->HostResolvePoolRange(pointer.value, size, pool, offset, "PathDiff")) {
                continue;
            }
            host->HostUploadRange(*pool, offset, size);
            PD::Range range;
            range.role = "pointer";
            range.index = pointer_order++;
            range.source = view.prefix + PD::Source(view.cbuf, pointer.offset);
            range.address = pointer.value;
            range.bound_size = size;
            range.written = capture.raw_global && pipe.descriptors.stores_raw_global_memory;
            add_gpu(std::move(range), pool->Handle(), offset, size, true);
        }
    }
    PathDiffTextures(pipe, graphics, capture);

    u64 total = 0;
    for (const Source& source : sources) {
        total += source.size * (source.after ? 2 : 1);
    }
    if (total != 0) {
        pending.buffer = impl.CreateHostBuffer(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               MemoryUsage::Download);
    }
    u64 cursor = 0;
    for (const Source& source : sources) {
        pending.copies.push_back({source.range, false, cursor, source.size, source.buffer,
                                  source.offset});
        cursor += source.size;
        if (source.after) {
            pending.copies.push_back({source.range, true, cursor, source.size, source.buffer,
                                      source.offset});
            cursor += source.size;
        }
    }
    // Every earlier write is visible to the copies; the op's writes wait for them.
    host->HostMarkWritten();
    host->HostFlushBarrier();
    vk::CommandBuffer cmd = host->HostCmd();
    for (const InFlight::PathDiff::Copy& copy : pending.copies) {
        if (!copy.after && copy.size != 0) {
            cmd.CopyBuffer(copy.source, *pending.buffer,
                           VkBufferCopy{copy.source_offset, copy.offset, copy.size});
        }
    }
    host->HostMarkWritten();
    host->HostFlushBarrier();
    return pending;
}

void DrawContext::PathDiffEnd(InFlight::PathDiff&& pending) {
    host->HostMarkWritten();
    host->HostFlushBarrier();
    vk::CommandBuffer cmd = host->HostCmd();
    for (const InFlight::PathDiff::Copy& copy : pending.copies) {
        if (copy.after && copy.size != 0) {
            cmd.CopyBuffer(copy.source, *pending.buffer,
                           VkBufferCopy{copy.source_offset, copy.offset, copy.size});
        }
    }
    host->HostMarkWritten();
    host->HostNeedHostReadBarrier();
    host->HostFlight().pathdiffs.push_back(std::move(pending));
}

void DrawContext::PathDiffTextures(const PipelineBase& pipe, bool graphics, PD::Capture& capture) {
    for (const VC::DescriptorBinding& binding : pipe.descriptors.bindings) {
        if (binding.kind != VC::DescriptorKind::CombinedImageSampler &&
            binding.kind != VC::DescriptorKind::StorageImage) {
            continue;
        }
        const bool is_image = binding.kind == VC::DescriptorKind::StorageImage;
        const u32 stage = static_cast<u32>(binding.stage);
        for (u32 i = 0; i < binding.count; ++i) {
            const VC::HandleRef texture_ref = VC::HandleElement(binding.texture, i);
            const VC::HandleRef sampler_ref = VC::HandleElement(binding.sampler, i);
            const u32 texture_word =
                ReadBindingHandleWord(pipe.descriptors, binding, texture_ref,
                                      is_image ? VC::HandleWordKind::Image
                                               : VC::HandleWordKind::Texture)
                    .value_or(0);
            const u32 sampler_word =
                sampler_ref.source != VC::HandleSource::None
                    ? ReadHandleWord(stage, sampler_ref, VC::HandleWordKind::SeparateSampler)
                          .value_or(0)
                    : 0;
            PD::Texture t;
            t.source = StagePrefix(graphics, stage) +
                       PD::Source(binding.texture.bank, binding.texture.byte_offset);
            t.index = i;
            t.image = is_image;
            t.handle = texture_word;
            const std::shared_ptr<Texture>& texture = TextureOfId(VC::TextureIdOf(texture_word));
            if (texture && texture->storage) {
                const Storage& s = *texture->storage;
                t.resolved = true;
                t.width = s.extent.width;
                t.height = s.extent.height;
                t.depth = s.extent.depth;
                t.levels = s.levels;
                t.layers = s.layers;
                ViewKey key{};
                ViewKeyOf(*texture, s, binding.type, is_image, key);
                t.format = static_cast<u32>(key.format);
                t.view_type = static_cast<u32>(key.type);
                t.base_level = key.base_level;
                t.view_levels = key.levels;
                t.base_layer = key.base_layer;
                t.view_layers = key.layers;
                for (u32 c = 0; c < 4; ++c) {
                    t.swizzle[c] = (key.swizzle >> (4 * c)) & 0xF;
                }
            }
            if (!is_image) {
                const u32 sampler_id =
                    ((sampler_ref.source != VC::HandleSource::None ? (texture_word | sampler_word)
                                                                   : texture_word)) >>
                    20;
                if (const u64 serial = SamplerSerialOfId(sampler_id); serial != 0) {
                    t.sampler = resources.DescribeSampler(serial);
                }
            }
            capture.textures.push_back(std::move(t));
        }
    }
}

} // namespace Vulkan::Nvn
