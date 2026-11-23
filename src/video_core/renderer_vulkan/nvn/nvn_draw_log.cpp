// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Shader logging mode (video_core/shader_log.h): the runtime inputs of a draw or dispatch at the
// first use of its pipeline. Described from the same state BuildDescriptors binds from:
//   * the constant buffer ranges the shader reads: bound address, the bound size, the bytes the
//     shader may read (`Shader::Info::constant_buffer_used_sizes`, at most 64 KiB; the bound size
//     when the pipeline came from the disk cache and the used size is unknown) with the inline
//     updates of the submission applied, exactly what the draw sees;
//   * the texture / sampler handle words of every texture and image binding, the ids in them and
//     the descriptors behind the ids. The HLE replay has no guest TIC/TSC (the NVN library's
//     pool entries are not guest memory here): what the game registered is the NVN texture and
//     sampler description (`desc`), logged raw.
//   * the push constants and the storage buffer ranges the pipeline binds.

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/shader_log.h"

namespace Vulkan::Nvn {

namespace ShaderLog = VideoCommon::ShaderLog;

namespace {

constexpr u64 MAX_CBUF_BYTES = 65536;
constexpr u64 CHUNK = 4096;

/// A JSON number (null for inf/nan: JSON has neither).
std::string F(float value) {
    return std::isfinite(value) ? fmt::format("{}", value) : std::string{"null"};
}

std::string Hex(const u8* bytes, size_t size) {
    static constexpr char DIGITS[] = "0123456789abcdef";

    std::string out;
    out.reserve(size * 2);
    for (size_t i = 0; i < size; ++i) {
        out.push_back(DIGITS[bytes[i] >> 4]);
        out.push_back(DIGITS[bytes[i] & 0xF]);
    }
    return out;
}

std::string_view StageTag(u32 stage) {
    switch (stage) {
    case 0:
        return "vs";
    case 1:
        return "fs";
    case 2:
        return "gs";
    case 3:
        return "tcs";
    case 4:
        return "tes";
    case STAGE_TASK:
        return "task";
    case STAGE_MESH:
        return "mesh";
    default:
        return "cs";
    }
}

std::string_view SourceName(VC::HandleSource source) {
    switch (source) {
    case VC::HandleSource::None:
        return "none";
    case VC::HandleSource::Combined:
        return "combined";
    case VC::HandleSource::SeparateTexture:
        return "separate_texture";
    case VC::HandleSource::SeparateSampler:
        return "separate_sampler";
    case VC::HandleSource::Image:
        return "image";
    case VC::HandleSource::UniformBuffer:
        return "uniform_buffer";
    case VC::HandleSource::ComputeTexture:
        return "compute_texture";
    case VC::HandleSource::ComputeImage:
        return "compute_image";
    case VC::HandleSource::Program:
        return "program";
    default:
        return "unsupported";
    }
}

std::string_view StorageSourceName(VC::StorageSource source) {
    switch (source) {
    case VC::StorageSource::DriverTable:
        return "driver_table";
    case VC::StorageSource::ComputeUniformEntry:
        return "compute_uniform_entry";
    case VC::StorageSource::ComputeStorageEntry:
        return "compute_storage_entry";
    case VC::StorageSource::UniformBufferWord:
        return "uniform_buffer_word";
    default:
        return "none";
    }
}

std::string TextureDescJson(const Texture& texture) {
    const VC::TextureDesc& d = texture.desc;
    const VC::TextureViewDesc& v = d.view;
    return fmt::format(
        "{{\"serial\":{},\"target\":{},\"format\":{},\"flags\":{},\"samples\":{},\"width\":{},"
        "\"height\":{},\"depth\":{},\"levels\":{},\"stride\":{},\"swizzle\":\"{:#x}\","
        "\"set_swizzle\":{},\"stencil\":{},\"pool_serial\":{},\"pool_offset\":{},\"view_of\":{},"
        "\"view\":{{\"set_levels\":{},\"set_layers\":{},\"set_format\":{},\"set_target\":{},"
        "\"set_swizzle\":{},\"set_dsmode\":{},\"dsmode\":{},\"target\":{},\"base_level\":{},"
        "\"nlevels\":{},\"format\":{},\"min_layer\":{},\"nlayers\":{},\"swizzle\":[{},{},{},{}],"
        "\"is_image\":{}}},\"host_vk_format\":{},\"host_pixel_format\":{}}}",
        d.serial, d.target, d.format, d.flags, d.samples, d.width, d.height, d.depth, d.levels,
        d.stride, d.swizzle, d.set_swizzle, d.stencil, d.pool_serial, d.pool_offset, d.view_of,
        v.set_levels, v.set_layers, v.set_format, v.set_target, v.set_swizzle, v.set_dsmode,
        v.dsmode, v.target, v.base_level, v.nlevels, v.format, v.min_layer, v.nlayers,
        v.swizzle[0], v.swizzle[1], v.swizzle[2], v.swizzle[3], v.is_image,
        static_cast<u32>(texture.host.format), static_cast<u32>(texture.host.pixel_format));
}

std::string SamplerDescJson(const VC::SamplerDesc& d) {
    return fmt::format(
        "{{\"serial\":{},\"min_filter\":{},\"mag_filter\":{},\"wrap_s\":{},\"wrap_t\":{},"
        "\"wrap_r\":{},\"compare_mode\":{},\"compare_func\":{},\"reduction\":{},\"lod_min\":{},"
        "\"lod_max\":{},\"lod_bias\":{},\"max_anisotropy\":{},\"border\":[{},{},{},{}]}}",
        d.serial, d.min_filter, d.mag_filter, d.wrap_s, d.wrap_t, d.wrap_r, d.compare_mode,
        d.compare_func, d.reduction, F(d.lod_min), F(d.lod_max), F(d.lod_bias),
        F(d.max_anisotropy), F(d.border[0]), F(d.border[1]), F(d.border[2]), F(d.border[3]));
}

} // namespace

std::string DrawContext::DescribeInputs(const PipelineBase* logged_pipeline, bool compute) const {
    VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire);
    const PipelineLogInfo* const log = logged_pipeline != nullptr ? logged_pipeline->log.get() : nullptr;
    const bool is_compute = logged_pipeline != nullptr ? logged_pipeline->stages[STAGE_COMPUTE].present : compute;

    std::string cbufs;
    std::string textures;
    std::string storage;
    const auto add = [](std::string& list, const std::string& item) {
        if (!list.empty()) {
            list += ',';
        }
        list += item;
    };

    // One constant buffer range: the bytes the draw sees (guest memory with the inline updates of
    // this submission on top).
    const auto add_cbuf = [&](u32 stage, u32 slot, std::string_view via, bool used_known,
                              u32 used) {
        const BufferBinding& binding = stages[stage].uniform[slot];
        const u32 bank = slot + 3;
        std::string item = fmt::format(
            "{{\"stage\":{},\"index\":{},\"slot\":{},\"via\":\"{}\",\"address\":\"{:#x}\","
            "\"size_bound\":{}",
            ShaderLog::Quote(StageTag(stage)), bank, slot, via, binding.address, binding.size);
        u64 want = (std::min)(binding.size, MAX_CBUF_BYTES);
        if (used_known) {
            want = (std::min<u64>)(want, used);
            item += fmt::format(",\"size_used\":{}", used);
        } else {
            item += ",\"size_used\":null";
        }
        if (binding.address == 0 || want == 0 || memory == nullptr) {
            item += fmt::format(",\"size_read\":0,\"complete\":{},\"words\":\"\"}}",
                                binding.address == 0 || want == 0);
            add(cbufs, item);
            return;
        }
        std::vector<u8> bytes(want);
        std::vector<u8> readable((want + CHUNK - 1) / CHUNK, 0);
        // EDEN_NX2_NVN_UBO_SNAPSHOT: the submit-time bytes BuildDescriptors binds.
        const bool from_snapshot = ReadSnapshotBytes(binding.address, bytes.data(), want);
        if (from_snapshot || memory->Read(binding.address, bytes.data(), want)) {
            std::fill(readable.begin(), readable.end(), u8{1});
        } else {
            for (u64 chunk = 0; chunk < readable.size(); ++chunk) {
                const u64 begin = chunk * CHUNK;
                const u64 length = (std::min)(CHUNK, want - begin);
                readable[chunk] =
                    memory->Read(binding.address + begin, bytes.data() + begin, length) ? 1 : 0;
            }
        }
        // Byte spans the inline updates wrote: they complete an unreadable chunk only where they
        // cover all of it.
        std::vector<std::pair<u64, u64>> covered;
        if (host != nullptr && !from_snapshot) {
            impl.ApplyPendingInlineWrites(binding.address, bytes.data(), want,
                                          host->HostFlight().seq, &covered);
        }
        for (const InlineUpdate& update : inline_updates) {
            const u64 lo = (std::max)(update.address, binding.address);
            const u64 hi = (std::min)(update.address + update.size, binding.address + want);
            if (lo < hi) {
                std::memcpy(bytes.data() + (lo - binding.address),
                            update.data + (lo - update.address), hi - lo);
                covered.emplace_back(lo - binding.address, hi - binding.address);
            }
        }
        bool complete = true;
        for (u64 chunk = 0; chunk < readable.size() && complete; ++chunk) {
            if (readable[chunk] != 0) {
                continue;
            }
            const u64 end = (std::min)((chunk + 1) * CHUNK, want);
            for (u64 byte = chunk * CHUNK; byte < end && complete; ++byte) {
                complete = std::any_of(covered.begin(), covered.end(), [byte](const auto& span) {
                    return span.first <= byte && byte < span.second;
                });
            }
        }
        item += fmt::format(",\"size_read\":{},\"complete\":{},\"words\":\"{}\"}}", want, complete,
                            Hex(bytes.data(), bytes.size()));
        add(cbufs, item);
    };

    const auto add_storage = [&](u32 stage, u32 slot, std::string_view source,
                                 const BufferBinding& resolved, bool written) {
        add(storage,
            fmt::format("{{\"stage\":{},\"slot\":{},\"source\":\"{}\",\"address\":\"{:#x}\","
                        "\"size\":{},\"written\":{}}}",
                        ShaderLog::Quote(StageTag(stage)), slot, source, resolved.address,
                        resolved.size, written));
    };

    const auto handle_json = [&](u32 word) {
        return fmt::format("\"{:#x}\"", word);
    };
    const auto describe_texture = [&](u32 stage, std::string_view kind, u32 binding_number,
                                      u32 index, const VC::HandleRef& texture_ref,
                                      const VC::HandleRef& sampler_ref, VC::HandleWordKind word_kind,
                                      Shader::TextureType type, bool depth, bool integer,
                                      bool written,
                                      const VC::DescriptorBinding* binding = nullptr) {
        const u32 texture_word =
            (binding != nullptr && logged_pipeline != nullptr
                 ? ReadBindingHandleWord(logged_pipeline->descriptors, *binding, texture_ref,
                                         word_kind)
                 : ReadHandleWord(stage, texture_ref, word_kind))
                .value_or(0);
        const bool has_sampler = sampler_ref.source != VC::HandleSource::None;
        const u32 sampler_word =
            has_sampler ? ReadHandleWord(stage, sampler_ref, VC::HandleWordKind::SeparateSampler)
                              .value_or(0)
                        : 0;
        const u32 texture_id = VC::TextureIdOf(texture_word);
        const u32 sampler_id =
            has_sampler ? VC::SamplerIdOf(sampler_word) : VC::SamplerIdOf(texture_word);
        std::string item = fmt::format(
            "{{\"stage\":{},\"kind\":\"{}\",\"binding\":{},\"index\":{},\"handle\":{},"
            "\"cbuf\":{},\"offset\":{},\"source\":\"{}\",\"sampler_handle\":{},"
            "\"sampler_cbuf\":{},\"sampler_offset\":{},\"sampler_source\":\"{}\","
            "\"texture_id\":{},\"sampler_id\":{},\"type\":{},\"is_depth\":{},\"is_integer\":{},"
            "\"is_written\":{},\"tic\":null,\"tsc\":null,\"descriptor_source\":\"nvn_desc\"",
            ShaderLog::Quote(StageTag(stage)), kind, binding_number, index,
            handle_json(texture_word), texture_ref.bank, texture_ref.byte_offset,
            SourceName(texture_ref.source), handle_json(sampler_word), sampler_ref.bank,
            sampler_ref.byte_offset, SourceName(sampler_ref.source), texture_id, sampler_id,
            static_cast<u32>(type), depth, integer, written);
        if (texture_ref.source == VC::HandleSource::Program) {
            item += fmt::format(",\"handle_program\":{}", texture_ref.slot);
        }
        const std::shared_ptr<Texture>& texture = TextureOfId(texture_id);
        if (texture) {
            item += ",\"texture\":" + TextureDescJson(*texture);
            item += fmt::format(",\"format\":{}", static_cast<u32>(texture->host.pixel_format));
        } else {
            item += ",\"texture\":null,\"format\":null";
        }
        if (kind != "image" && sampler_id != 0) {
            const u64 serial = SamplerSerialOfId(sampler_id);
            const auto it = serial != 0 ? impl.samplers.find(serial) : impl.samplers.end();
            item += it != impl.samplers.end() ? ",\"sampler\":" + SamplerDescJson(it->second)
                                              : std::string{",\"sampler\":null"};
        } else {
            item += ",\"sampler\":null";
        }
        add(textures, item + "}");
    };

    if (logged_pipeline != nullptr) {
        for (const VC::DescriptorBinding& binding : logged_pipeline->descriptors.bindings) {
            const u32 stage = static_cast<u32>(binding.stage);
            if (stage >= BIND_STAGES) {
                continue;
            }
            switch (binding.kind) {
            case VC::DescriptorKind::UniformBuffer:
                if (binding.role != VC::DescriptorRole::UniformBuffer) {
                    break; // driver bank / handle table / program constants are ours
                }
                for (u32 i = 0; i < binding.count; ++i) {
                    const u32 slot = binding.slot + i;
                    if (slot >= stages[stage].uniform.size()) {
                        continue;
                    }
                    const u32 bank = slot + 3;
                    const bool known = log != nullptr && (log->cbuf_known & (1u << stage)) != 0 &&
                                       bank < LOG_MAX_CBUFS;
                    add_cbuf(stage, slot, "uniform_buffer", known,
                             known ? log->cbuf_used[stage][bank] : 0);
                }
                break;
            case VC::DescriptorKind::StorageBuffer:
                for (u32 i = 0; i < binding.count; ++i) {
                    const u32 slot = binding.slot + i;
                    BufferBinding resolved{};
                    switch (binding.storage_source) {
                    case VC::StorageSource::DriverTable:
                        if (slot < stages[stage].storage.size()) {
                            resolved = stages[stage].storage[slot];
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
                            // The shader reads these bytes as constants through the entry.
                            add_cbuf(STAGE_COMPUTE, slot, "compute_uniform_entry", false, 0);
                        }
                        break;
                    case VC::StorageSource::UniformBufferWord: {
                        const u32 offset = binding.storage_offset + i * VC::StorageBufferStride;
                        const auto lo = ReadUniformWord(stage, binding.slot, offset);
                        const auto hi = ReadUniformWord(stage, binding.slot, offset + 4);
                        const auto size = ReadUniformWord(stage, binding.slot, offset + 8);
                        if (lo && hi && size) {
                            resolved = {static_cast<u64>(*lo) | static_cast<u64>(*hi) << 32, *size};
                        }
                        break;
                    }
                    default:
                        break;
                    }
                    add_storage(stage, slot, StorageSourceName(binding.storage_source), resolved,
                                binding.is_written);
                }
                break;
            default: {
                const bool is_image = binding.kind == VC::DescriptorKind::StorageImage ||
                                      binding.kind == VC::DescriptorKind::StorageTexelBuffer;
                const std::string_view kind =
                    is_image ? "image"
                             : (binding.kind == VC::DescriptorKind::UniformTexelBuffer ? "texel_buffer"
                                                                                       : "combined");
                for (u32 i = 0; i < binding.count; ++i) {
                    const VC::HandleRef texture_ref = VC::HandleElement(binding.texture, i);
                    const VC::HandleRef sampler_ref = VC::HandleElement(binding.sampler, i);
                    describe_texture(stage, kind, binding.binding, i, texture_ref, sampler_ref,
                                     is_image ? VC::HandleWordKind::Image
                                              : VC::HandleWordKind::Texture,
                                     binding.type, binding.is_depth, binding.is_integer,
                                     binding.is_written, &binding);
                }
                break;
            }
            }
        }
    } else if (compute) {
        // A program replaced by a stand-in: no descriptor layout. Everything bound to the compute
        // stage.
        const StageBindings& bound = stages[STAGE_COMPUTE];
        for (u32 slot = 0; slot < bound.uniform.size(); ++slot) {
            if (bound.uniform[slot].address != 0) {
                add_cbuf(STAGE_COMPUTE, slot, "bound", false, 0);
            }
        }
        for (u32 slot = 0; slot < bound.storage.size(); ++slot) {
            if (bound.storage[slot].address != 0) {
                add_storage(STAGE_COMPUTE, slot, "bound", bound.storage[slot], false);
            }
        }
        const auto bound_handle = [&](u32 slot, bool image) {
            const u32 word = static_cast<u32>(image ? bound.image[slot] : bound.combined[slot]);
            if (word == 0) {
                return;
            }
            const VC::HandleRef ref{image ? VC::HandleSource::ComputeImage
                                          : VC::HandleSource::ComputeTexture,
                                    slot, 0, 0};
            describe_texture(STAGE_COMPUTE, image ? "image" : "combined", slot, 0, ref,
                             VC::HandleRef{},
                             image ? VC::HandleWordKind::Image : VC::HandleWordKind::Texture,
                             Shader::TextureType::Color2D, false, false, image);
        };
        for (u32 slot = 0; slot < bound.combined.size(); ++slot) {
            bound_handle(slot, false);
        }
        for (u32 slot = 0; slot < bound.image.size(); ++slot) {
            bound_handle(slot, true);
        }
    }

    const std::span<const u8> push = PushConstants(is_compute);
    return fmt::format("{{\"cbufs\":[{}],\"textures\":[{}],\"storage_buffers\":[{}],"
                       "\"push_constants\":\"{}\"}}",
                       cbufs, textures, storage, Hex(push.data(), push.size()));
}

} // namespace Vulkan::Nvn
