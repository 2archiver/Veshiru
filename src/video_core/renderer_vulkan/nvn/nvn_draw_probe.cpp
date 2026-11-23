// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The replay side of the debug server draw methods `nvn_draws`, `nvn_draw_skip` and
// `nvn_draw_dump` (video_core/nvn_draw_probe.h).

#include <algorithm>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/nvn_draw_probe.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"

namespace Vulkan::Nvn {

namespace {

namespace Probe = VideoCommon::NvnDrawProbe;

constexpr const char* StageNames[Probe::StageCount] = {
    "vertex", "fragment", "geometry", "tess_control", "tess_eval", "compute", "task", "mesh"};

/// Bytes of a game uniform buffer and of a storage buffer a dump shows.
constexpr u64 DumpUniformBytes = 0x1000;
constexpr u64 DumpStorageBytes = 0x100;

void AppendHex(std::string& text, const u8* data, u64 bytes, u64 per_line) {
    for (u64 offset = 0; offset < bytes; offset += per_line) {
        text += fmt::format("  {:06x}:", offset);
        for (u64 j = offset; j < (std::min)(offset + per_line, bytes); ++j) {
            text += fmt::format(" {:02x}", data[j]);
        }
        text += '\n';
    }
}

u32 FloatComponents(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R32_SFLOAT:
        return 1;
    case VK_FORMAT_R32G32_SFLOAT:
        return 2;
    case VK_FORMAT_R32G32B32_SFLOAT:
        return 3;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 4;
    default:
        return 0;
    }
}

const char* TextureTypeName(Shader::TextureType type) {
    constexpr const char* names[] = {"1d",   "1d_array",   "2d",     "2d_array", "3d",
                                     "cube", "cube_array", "buffer", "2d_rect"};
    const auto index = static_cast<size_t>(type);
    return index < std::size(names) ? names[index] : "?";
}

const char* BackingName(Pool::Backing backing) {
    switch (backing) {
    case Pool::Backing::Imported:
        return "imported (guest memory, the GPU reads these bytes)";
    case Pool::Backing::Mirror:
        return "mirror pool (device copy of guest memory, refreshed before use)";
    case Pool::Backing::DeviceLocal:
        return "device-local pool (GPU-only: the guest bytes shown may be stale)";
    case Pool::Backing::Virtual:
        return "virtual pool";
    default:
        return "unbacked pool";
    }
}

} // Anonymous namespace

void DrawContext::AppendBindingDump(std::string& text, const PipelineBase& pipe) {
    const VC::DescriptorLayout& layout = pipe.descriptors;
    text += fmt::format("descriptor bindings: {}\n", layout.bindings.size());

    // One buffer binding: the range the game bound, where the descriptor points (the plan and
    // the patched descriptor infos of BuildDescriptors), then the bytes the GPU reads as far as
    // the CPU can see them.
    const auto buffer = [&](const std::string& label, const BufferBinding& game, bool uniform,
                            bool written, size_t info_index, u64 max_bytes) {
        text += fmt::format("{} guest {:#x} size {:#x}{}", label, game.address, game.size,
                            written ? " written" : "");
        const VkDescriptorBufferInfo* info =
            info_index < scratch_buffers.size() ? &scratch_buffers[info_index] : nullptr;
        if (game.address == 0 || info == nullptr) {
            text += uniform ? ": unbound, reads zero\n" : ": unbound\n";
            return;
        }
        if (info->buffer == resources.DummyBuffer() || info->buffer == resources.SinkBuffer()) {
            text += ": not resolved, bound to the dummy (reads zero) or sink buffer\n";
            return;
        }
        const u64 size = info->range == VK_WHOLE_SIZE ? game.size : info->range;
        const DrawPlan::RangeUse* use = nullptr;
        for (const DrawPlan::RangeUse& range : draw_plan.ranges) {
            if (range.info_index == info_index) {
                use = &range;
                break;
            }
        }
        const size_t snapshot = uniform && usnap.Active() ? usnap.Find(game.address, size)
                                                          : UniformSnapshotView::npos;
        std::vector<u8> bytes((std::min)(size, max_bytes));
        bool readable = false;
        const auto read_guest = [&] {
            if (const u8* data = impl.GuestPointer(game.address, bytes.size())) {
                std::memcpy(bytes.data(), data, bytes.size());
                readable = true;
            }
        };
        std::string where;
        if (uniform && use == nullptr) {
            // BuildDescriptors' private arena copy (inline updates or a misaligned range) or the
            // uniform snapshot: the bytes are rebuilt the same way.
            if (snapshot != UniformSnapshotView::npos) {
                std::memcpy(bytes.data(), usnap.Bytes(snapshot, game.address), bytes.size());
                readable = true;
            } else {
                read_guest();
                if (host != nullptr) {
                    impl.ApplyPendingInlineWrites(game.address, bytes.data(), bytes.size(),
                                                  host->HostFlight().seq);
                }
            }
            const bool inline_updates_pending = InlineUpdatePending(game.address, size);
            const bool copied = inline_updates_pending || snapshot == UniformSnapshotView::npos;
            if (copied) {
                for (const InlineUpdate& update : inline_updates) {
                    const u64 begin = (std::max)(update.address, game.address);
                    const u64 end =
                        (std::min)(update.address + update.size, game.address + bytes.size());
                    if (begin < end) {
                        std::memcpy(bytes.data() + (begin - game.address),
                                    update.data + (begin - update.address), end - begin);
                        readable = true;
                    }
                }
            }
            where = !copied ? "uniform snapshot (the bytes as of the submit call)"
                    : inline_updates_pending ? "private copy (inline uniform updates applied)"
                                             : "private copy (range not 256-byte aligned)";
            if (copied && snapshot != UniformSnapshotView::npos) {
                where += " of the uniform snapshot";
            }
        } else if (use != nullptr && use->redirected) {
            const size_t index = FindShadow(*use->pool, use->offset, use->offset + use->size);
            if (index != shadows.size()) {
                where = shadows[index].mirror ? "VRAM mirror entry (written range)"
                                              : "device shadow copy";
            } else {
                where = "VRAM mirror";
            }
            where += fmt::format(" of pool {:#x} + {:#x}; guest bytes shown, GPU writes not "
                                 "written back yet are not",
                                 game.address - use->offset, use->offset);
            read_guest();
        } else if (use != nullptr) {
            where = fmt::format("{} {:#x} + {:#x}", BackingName(use->pool->backing),
                                game.address - use->offset, use->offset);
            read_guest();
        } else {
            where = "unknown";
            read_guest();
        }
        text += fmt::format(" bound {:#x}: {}\n", size, where);
        if (!readable) {
            text += "  (not CPU-readable)\n";
            return;
        }
        if (bytes.size() < size) {
            text += fmt::format("  first {:#x} bytes:\n", bytes.size());
        }
        Probe::AppendRows(text, bytes);
    };

    size_t info_index = 0; // BuildDescriptors' order: one info per uniform / storage element
    for (const VC::DescriptorBinding& binding : layout.bindings) {
        const u32 stage = static_cast<u32>(binding.stage);
        const char* stage_name = stage < Probe::StageCount ? StageNames[stage] : "?";
        const std::string head = fmt::format("binding {} {}", binding.binding, stage_name);
        switch (binding.kind) {
        case VC::DescriptorKind::UniformBuffer:
            for (u32 i = 0; i < binding.count; ++i, ++info_index) {
                switch (binding.role) {
                case VC::DescriptorRole::DriverBank: {
                    if (binding.stage != VC::ShaderStage::Compute) {
                        text += fmt::format("{} c0 driver bank: built per draw, not dumped\n", head);
                        break;
                    }
                    std::vector<u8> bank(DRIVER_BANK_SIZE);
                    FillComputeBank(pipe.compute_layout, bank.data());
                    if (layout.uses_raw_global_memory) {
                        WriteRawGlobalTable(impl.raw_table.device.get(), 0, bank);
                    }
                    text += fmt::format("{} c0 driver bank (built by the replay{}):\n", head,
                                        layout.uses_raw_global_memory
                                            ? "; raw table header without the alias table"
                                            : "");
                    Probe::AppendRows(text, bank);
                    break;
                }
                case VC::DescriptorRole::ProgramConstants: {
                    const auto& words = pipe.stages[stage].program_constants;
                    text += fmt::format("{} c1 program constants ({} words):\n", head,
                                        words ? words->size() : 0);
                    if (words) {
                        Probe::AppendRows(text, {reinterpret_cast<const u8*>(words->data()),
                                                 words->size() * sizeof(u32)});
                    }
                    break;
                }
                case VC::DescriptorRole::HandleTable:
                    text += fmt::format("{} c2 handle table: the handles of the textures below\n",
                                        head);
                    break;
                default: {
                    const u32 slot = binding.slot + i;
                    const BufferBinding game = slot < stages[stage].uniform.size()
                                                   ? stages[stage].uniform[slot]
                                                   : BufferBinding{};
                    buffer(fmt::format("{} c{} uniform slot {}", head, slot + 3, slot), game,
                           true, false, info_index, DumpUniformBytes);
                    break;
                }
                }
            }
            break;
        case VC::DescriptorKind::StorageBuffer:
            for (u32 i = 0; i < binding.count; ++i, ++info_index) {
                const char* source = "";
                switch (binding.storage_source) {
                case VC::StorageSource::DriverTable:
                    source = "driver table";
                    break;
                case VC::StorageSource::ComputeUniformEntry:
                    source = "compute uniform entry";
                    break;
                case VC::StorageSource::ComputeStorageEntry:
                    source = "compute storage entry";
                    break;
                case VC::StorageSource::UniformBufferWord:
                    source = "uniform buffer word";
                    break;
                default:
                    source = "none";
                    break;
                }
                buffer(fmt::format("{} storage slot {} ({}{}{})", head, binding.slot + i, source,
                                   binding.is_atomic_target ? ", atomics" : "",
                                   binding.is_raw_global_table ? ", raw global table" : ""),
                       ResolveStorageBinding(binding, i), false, binding.is_written, info_index,
                       DumpStorageBytes);
            }
            break;
        case VC::DescriptorKind::CombinedImageSampler:
        case VC::DescriptorKind::StorageImage: {
            const bool is_image = binding.kind == VC::DescriptorKind::StorageImage;
            for (u32 i = 0; i < binding.count; ++i) {
                const VC::HandleRef texture_ref = VC::HandleElement(binding.texture, i);
                const VC::HandleRef sampler_ref = VC::HandleElement(binding.sampler, i);
                const u32 texture_word =
                    ReadBindingHandleWord(layout, binding, texture_ref,
                                          is_image ? VC::HandleWordKind::Image
                                                   : VC::HandleWordKind::Texture)
                        .value_or(0);
                const u32 sampler_word =
                    sampler_ref.source != VC::HandleSource::None
                        ? ReadHandleWord(stage, sampler_ref, VC::HandleWordKind::SeparateSampler)
                              .value_or(0)
                        : 0;
                text += fmt::format("{} {} slot {} {}{} handle {:#x}", head,
                                    is_image ? "image" : "texture", texture_ref.slot,
                                    TextureTypeName(binding.type),
                                    binding.is_written ? " written" : "", texture_word);
                const std::shared_ptr<Texture>& texture = TextureOfId(VC::TextureIdOf(texture_word));
                if (texture) {
                    const VC::TextureDesc& desc = texture->desc;
                    u64 guest = 0;
                    if (const auto it = impl.pools.find(desc.pool_serial); it != impl.pools.end()) {
                        guest = it->second->address + desc.pool_offset;
                    }
                    text += fmt::format(" -> t{:x} {}x{}x{} levels {} format {} target {} guest "
                                        "{:#x} (pool {:x} + {:#x})",
                                        texture->serial, desc.width, desc.height, desc.depth,
                                        desc.levels, desc.format, desc.target, guest,
                                        desc.pool_serial, desc.pool_offset);
                    if (desc.view_of != 0) {
                        text += fmt::format(" view of t{:x}", desc.view_of);
                    }
                    if (desc.view.set_levels || desc.view.set_layers) {
                        text += fmt::format(" view levels {}+{} layers {}+{}",
                                            desc.view.set_levels ? desc.view.base_level : 0,
                                            desc.view.set_levels ? desc.view.nlevels : 0,
                                            desc.view.set_layers ? desc.view.min_layer : 0,
                                            desc.view.set_layers ? desc.view.nlayers : 0);
                    }
                    if (desc.set_swizzle || desc.view.set_swizzle) {
                        text += fmt::format(" swizzle {:#x}", desc.set_swizzle ? desc.swizzle : 0xa9a);
                        if (desc.view.set_swizzle) {
                            text += fmt::format(" view swizzle {} {} {} {}", desc.view.swizzle[0],
                                                desc.view.swizzle[1], desc.view.swizzle[2],
                                                desc.view.swizzle[3]);
                        }
                    }
                    if (desc.view.set_format) {
                        text += fmt::format(" view format {}", desc.view.format);
                    }
                    if (texture->storage) {
                        text += fmt::format(" VkFormat {} storage layers {} levels {}",
                                            static_cast<s32>(texture->storage->host.format),
                                            texture->storage->layers, texture->storage->levels);
                    }
                } else {
                    text += " -> no texture";
                }
                if (!is_image) {
                    const u32 sampler_id =
                        VC::SamplerIdOf(sampler_word != 0 ? sampler_word : texture_word);
                    const u64 serial = SamplerSerialOfId(sampler_id);
                    text += fmt::format("; sampler {} s{:x}", sampler_id, serial);
                    if (const auto it = impl.samplers.find(serial); it != impl.samplers.end()) {
                        const VC::SamplerDesc& s = it->second;
                        text += fmt::format(" filter {}/{} wrap {}/{}/{} compare {}/{} reduction {} "
                                            "lod {}..{} bias {} aniso {} border {} {} {} {}",
                                            s.min_filter, s.mag_filter, s.wrap_s, s.wrap_t,
                                            s.wrap_r, s.compare_mode, s.compare_func, s.reduction,
                                            s.lod_min, s.lod_max, s.lod_bias, s.max_anisotropy,
                                            s.border[0], s.border[1], s.border[2], s.border[3]);
                    }
                }
                text += '\n';
            }
            break;
        }
        case VC::DescriptorKind::UniformTexelBuffer:
            // BuildDescriptors binds a copy of the texture buffer's bytes (TexelBufferView).
            for (u32 i = 0; i < binding.count; ++i) {
                const VC::HandleRef texture_ref = VC::HandleElement(binding.texture, i);
                const u32 texture_word =
                    ReadBindingHandleWord(layout, binding, texture_ref, VC::HandleWordKind::Texture)
                        .value_or(0);
                text += fmt::format("{} texel buffer slot {} handle {:#x}", head, texture_ref.slot,
                                    texture_word);
                const std::shared_ptr<Texture>& texture = TextureOfId(VC::TextureIdOf(texture_word));
                if (texture) {
                    const VC::TextureDesc& desc = texture->desc;
                    text += fmt::format(" -> t{:x} width {} format {} target {} (pool {:x} + "
                                        "{:#x})\n",
                                        texture->serial, desc.width, desc.format, desc.target,
                                        desc.pool_serial, desc.pool_offset);
                } else {
                    text += " -> no texture: zeros bound\n";
                }
            }
            break;
        default:
            text += fmt::format("{} texel buffer slot {}: not supported, not bound\n", head,
                                binding.slot);
            break;
        }
    }
    // Game uniform slots no descriptor reads (folded away or unused): only listed.
    for (u32 stage = 0; stage < BIND_STAGES; ++stage) {
        if (stages[stage].program_serial == 0 || !pipe.stages[stage].present) {
            continue;
        }
        std::string unread;
        for (u32 slot = 0; slot < stages[stage].uniform.size(); ++slot) {
            const BufferBinding& game = stages[stage].uniform[slot];
            const bool read = std::ranges::any_of(layout.bindings, [&](const auto& binding) {
                return static_cast<u32>(binding.stage) == stage &&
                       binding.kind == VC::DescriptorKind::UniformBuffer &&
                       binding.role == VC::DescriptorRole::UniformBuffer && slot >= binding.slot &&
                       slot - binding.slot < binding.count;
            });
            if (game.address != 0 && !read) {
                unread += fmt::format(" c{}={:#x}/{:#x}", slot + 3, game.address, game.size);
            }
        }
        if (!unread.empty()) {
            text += fmt::format("{} uniform buffers bound but read through no descriptor:{}\n",
                                StageNames[stage], unread);
        }
    }
    if (layout.uses_raw_global_memory) {
        const auto& table = impl.raw_table;
        text += fmt::format("raw global memory: stores {} written through {}; table generation {} "
                            "ranges {} pools {}\n",
                            layout.stores_raw_global_memory, layout.raw_stores_written_through,
                            table.generation, table.entries.size(), table.pool_serials.size());
    }
}

bool DrawContext::ProbeDraw(const PipelineBase& pipe, u32 mode, u32 element_count,
                            const IndexRange* index,
                            const std::array<BufferRef, MAX_VERTEX_STREAMS>& vertex,
                            u32 used_streams, bool mesh) {
    Probe::DrawInfo info;
    for (u32 stage = 0; stage < Probe::StageCount; ++stage) {
        // A draw runs no compute stage: the one still bound belongs to the last dispatch.
        info.hashes[stage] = stages[stage].program_serial != 0 && stage != STAGE_COMPUTE
                                 ? ProfStageHash(stage)
                                 : 0;
    }
    info.mode = mode;
    info.count = element_count;
    info.indexed = index != nullptr;
    info.mesh = mesh;
    info.indirect = element_count == 0 && !mesh;
    // The same test PrepareDraw applies when it binds a staged copy.
    const auto staged = [&](u64 address, u64 size, const Pool* pool) {
        if (staged_vertex.empty() || pass_shader_writes || pool == nullptr ||
            pool->backing != Pool::Backing::Imported) {
            return false;
        }
        const auto it = staged_vertex.find(address);
        return it != staged_vertex.end() && it->second.size >= size;
    };
    for (u32 s = 0; s < MAX_VERTEX_STREAMS && info.stream_count < Probe::MaxStreams; ++s) {
        if (!((used_streams >> s) & 1)) {
            continue;
        }
        const VC::BufferRange& range = vertex_buffers[s];
        Probe::Stream& out = info.streams[info.stream_count++];
        out.slot = s;
        out.address = range.address;
        out.size = range.size;
        if (vertex[s].pool != nullptr) {
            out.pool_offset = vertex[s].pool_offset;
            out.pool_address = range.address - vertex[s].pool_offset;
        }
        out.staged = staged(range.address, range.size, vertex[s].pool);
    }
    if (index) {
        info.index_address = index->address;
        info.index_bytes = index->bytes;
        info.index_type = index->type;
        info.index_staged = !staged_vertex.empty() && !pass_shader_writes &&
                            staged_vertex.contains(index->address) &&
                            staged_vertex.find(index->address)->second.size >= index->bytes;
    }
    const bool skip = Probe::OnDraw(info);
    const auto ticket = Probe::DumpRequested(info);
    if (!ticket) {
        return skip;
    }

    std::string text = fmt::format("NVN draw dump {} of {}\nhashes:", ticket->index + 1,
                                   ticket->count);
    for (u32 stage = 0; stage < Probe::StageCount; ++stage) {
        if (info.hashes[stage] != 0) {
            text += fmt::format(" {}={:016x}", StageNames[stage], info.hashes[stage]);
        }
    }
    text += fmt::format("\nmode={} count={} indexed={} indirect={} mesh={} skipped={}\n", mode,
                        element_count, info.indexed, info.indirect, mesh, skip);
    text += "Bytes are the CPU view of guest memory when the draw was recorded (the GPU may read a "
            "staged copy or a VRAM mirror instead, see staged= and each binding's placement; it "
            "runs later).\n";
    for (u32 b = 0; b < vertex_input.binding_count && b < MAX_VERTEX_STREAMS; ++b) {
        const auto& binding = vertex_input.bindings[b];
        text += fmt::format("binding {} stride {} rate {} divisor {}\n", binding.binding,
                            binding.stride, static_cast<u32>(binding.inputRate), binding.divisor);
    }
    for (u32 a = 0; a < vertex_input.attribute_count && a < MAX_VERTEX_ATTRIBUTES; ++a) {
        const auto& attribute = vertex_input.attributes[a];
        text += fmt::format("attribute location {} binding {} VkFormat {} offset {}\n",
                            attribute.location, attribute.binding,
                            static_cast<s32>(attribute.format), attribute.offset);
    }
    u32 max_stride = 0;
    for (u32 i = 0; i < info.stream_count; ++i) {
        const Probe::Stream& stream = info.streams[i];
        u32 stride = 0;
        for (u32 b = 0; b < vertex_input.binding_count && b < MAX_VERTEX_STREAMS; ++b) {
            if (vertex_input.bindings[b].binding == stream.slot) {
                stride = vertex_input.bindings[b].stride;
            }
        }
        max_stride = (std::max)(max_stride, stride);
        const u64 bytes =
            (std::min)(stream.size, stride != 0 ? u64{stride} * 256 : u64{4096});
        text += fmt::format("stream {} guest {:#x} size {:#x} pool {:#x} + {:#x} staged={} "
                            "stride {}: first {} bytes\n",
                            stream.slot, stream.address, stream.size, stream.pool_address,
                            stream.pool_offset, stream.staged, stride, bytes);
        const u8* data = impl.GuestPointer(stream.address, bytes);
        if (data == nullptr) {
            text += "  (not CPU-readable)\n";
            continue;
        }
        AppendHex(text, data, bytes, 32);
        // 32-bit float attributes of this stream, first 32 vertices.
        for (u32 a = 0; a < vertex_input.attribute_count && a < MAX_VERTEX_ATTRIBUTES; ++a) {
            const auto& attribute = vertex_input.attributes[a];
            const u32 components = FloatComponents(attribute.format);
            if (attribute.binding != stream.slot || components == 0 || stride == 0) {
                continue;
            }
            text += fmt::format("  location {} as float{}:\n", attribute.location, components);
            for (u32 v = 0; v < 32; ++v) {
                const u64 at = u64{v} * stride + attribute.offset;
                if (at + u64{components} * 4 > bytes) {
                    break;
                }
                text += fmt::format("    v{}:", v);
                for (u32 c = 0; c < components; ++c) {
                    f32 value;
                    std::memcpy(&value, data + at + c * 4, sizeof(value));
                    text += fmt::format(" {}", value);
                }
                text += '\n';
            }
        }
    }
    if (index) {
        const u32 size = index->type == 0 ? 1 : (index->type == 1 ? 2 : 4);
        text += fmt::format("index guest {:#x} bytes {:#x} type {} staged={}", index->address,
                            index->bytes, index->type, info.index_staged);
        const u8* data = impl.GuestPointer(index->address, index->bytes);
        if (data != nullptr && index->bytes >= size) {
            // The restart marker (all ones in the index type, as the host binds it) is no vertex.
            const bool restart = fixed.primitive_restart != 0;
            const u32 marker = size == 4 ? ~0u : (1u << (size * 8)) - 1;
            u32 lo = ~0u;
            u32 hi = 0;
            bool any = false;
            for (u64 i = 0; i + size <= index->bytes; i += size) {
                u32 value = 0;
                std::memcpy(&value, data + i, size);
                if (restart && value == marker) {
                    continue;
                }
                any = true;
                lo = (std::min)(lo, value);
                hi = (std::max)(hi, value);
            }
            if (any) {
                text += fmt::format(
                    " min {} max {} (bytes needed per stream at max stride {}: {})\n", lo, hi,
                    max_stride, (u64{hi} + 1) * max_stride);
            } else {
                text += " (no vertex indices)\n";
            }
        } else {
            text += " (not CPU-readable)\n";
        }
    }
    AppendBindingDump(text, pipe);
    text += '\n';
    Probe::WriteDump(*ticket, text);
    ProbeNoteImages(fmt::format("{}-{}", ticket->path, ticket->index + 1), u64{2048} * 2048);
    LOG_INFO(Render_Vulkan, "[diag-draw] dump {} of {} of a draw with {:016x}: {}",
             ticket->index + 1, ticket->count, info.hashes[mesh ? 7 : 0], ticket->path);
    return skip;
}

void DrawContext::ProbeDispatchDump(const Probe::DumpTicket& ticket, const PipelineBase& pipe,
                                    const VC::NvnControl& control, const VC::DispatchPayload& p,
                                    const BufferRef* arguments, u64 hash, bool skipped) {
    std::string text = fmt::format("NVN dispatch dump {} of {} (present {})\nhash: compute={:016x}\n",
                                   ticket.index + 1, ticket.count, impl.presents_recorded + 1,
                                   hash);
    if (arguments != nullptr) {
        text += fmt::format("groups: indirect at {:#x}", p.indirect);
        // The CPU view: arguments an earlier GPU command of this submission writes are not there.
        if (const u8* data = impl.GuestPointer(p.indirect, 3 * sizeof(u32))) {
            std::array<u32, 3> groups;
            std::memcpy(groups.data(), data, sizeof(groups));
            text += fmt::format(", guest bytes now {}x{}x{} (the GPU reads them later{})",
                                groups[0], groups[1], groups[2],
                                arguments->pool != nullptr &&
                                        arguments->pool->backing != Pool::Backing::Imported
                                    ? ", from a device copy"
                                    : "");
        } else {
            text += ", not CPU-readable";
        }
        text += '\n';
    } else {
        text += fmt::format("groups: {}x{}x{}\n", p.x, p.y, p.z);
    }
    text += fmt::format("thread group: {}x{}x{}, shared memory {} B, push constant words {}, "
                        "skipped={}\n",
                        control.thread_group[0], control.thread_group[1], control.thread_group[2],
                        control.shared_memory_bytes, control.compute.push_constant_words, skipped);
    text += "Buffers: cN = uniform slot N-3 (c0 driver bank, c1 program constants, c2 handle "
            "table). Each names where the GPU reads it from; bytes are what it reads as the CPU "
            "sees them when the dispatch is recorded (it runs later).\n";
    const size_t push_bytes = (std::min<size_t>)(u64{control.compute.push_constant_words} * 4,
                                                  push_constants[1].size());
    if (push_bytes != 0) {
        text += fmt::format("push constants ({} bytes, in c0 too):\n", push_bytes);
        Probe::AppendRows(text, std::span{push_constants[1]}.first(push_bytes));
    }
    AppendBindingDump(text, pipe);
    text += '\n';
    Probe::WriteDump(ticket, text);
    LOG_INFO(Render_Vulkan, "[diag-draw] dump {} of {} of a dispatch of {:016x}: {}",
             ticket.index + 1, ticket.count, hash, ticket.path);
}

void DrawContext::ProbeNoteImages(const std::string& prefix, u64 max_texels) {
    u32 index = 0;
    for (const DrawPlan::ImageBinding& b : draw_plan.bindings) {
        ++index;
        if (b.texture == nullptr || !b.texture->storage) {
            continue;
        }
        const Storage& s = *b.texture->storage;
        const u64 width = (std::max)(s.extent.width >> b.key.base_level, 1u);
        const u64 height = (std::max)(s.extent.height >> b.key.base_level, 1u);
        if (width * height > max_texels || (s.aspect & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
            continue;
        }
        probe_images.push_back({b.texture->storage, b.key.base_level, b.key.base_layer,
                                fmt::format("{}-b{}-{}{}-t{:x}-l{}-m{}.tga", prefix, index,
                                            b.is_image ? "image" : "texture",
                                            b.scratch ? "-copy" : "", b.texture->serial,
                                            b.key.base_layer, b.key.base_level)});
    }
}

void DrawContext::ProbeFlushImages() {
    for (ProbeImage& image : probe_images) {
        Debug::PendingDump pending;
        pending.path = std::move(image.path);
        pending.format = image.storage->host.format;
        pending.thumb_width = 480;
        host->HostDumpImage(*image.storage, image.level, image.layer, VK_IMAGE_ASPECT_COLOR_BIT,
                            std::move(pending));
    }
    probe_images.clear();
}

} // namespace Vulkan::Nvn
