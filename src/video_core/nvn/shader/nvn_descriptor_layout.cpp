// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include "video_core/nvn/shader/nvn_descriptor_layout.h"

#include <utility>

#include <fmt/format.h>

namespace VideoCommon::Nvn {
namespace {

struct HandleResult {
    HandleRef texture;
    HandleRef sampler;
    std::string issue;
};

u32 TableCapacity(HandleSource source, const ComputeHandleTables* compute) {
    switch (source) {
    case HandleSource::Combined:
        return CombinedHandleSlots;
    case HandleSource::SeparateTexture:
        return SeparateTextureSlots;
    case HandleSource::SeparateSampler:
        return SeparateSamplerSlots;
    case HandleSource::Image:
        return ImageHandleSlots;
    case HandleSource::ComputeTexture:
        return compute != nullptr ? compute->texture_count : 0;
    case HandleSource::ComputeImage:
        return compute != nullptr ? compute->image_count : 0;
    default:
        return ~0U; // uniform buffer words: bounded by the bound buffer, checked at replay
    }
}

HandleRef MakeRef(const HandleClassification& classified, u32 bank, u32 offset) {
    return {classified.source, classified.slot, bank, offset};
}

// Shared by texture, texture buffer and image descriptors.
HandleResult ResolveHandles(bool has_secondary, u32 index, u32 offset, u32 shift_left,
                            u32 secondary_index, u32 secondary_offset, u32 secondary_shift_left,
                            u32 count, u32 size_shift, HandleWant want,
                            const ComputeHandleTables* compute) {
    HandleResult result;
    const auto primary = ClassifyHandleWord(index, offset, want, compute);
    const HandleRef primary_ref = MakeRef(primary, index, offset);
    const auto fail = [&](std::string issue) {
        result.issue = std::move(issue);
        return result;
    };
    if (primary.source == HandleSource::Unsupported) {
        return fail(fmt::format("{} (c{}[{:#x}])", primary.reason, index, offset));
    }
    if (shift_left != 0) {
        return fail(fmt::format("handle word shifted left by {}", shift_left));
    }
    if (has_secondary && (secondary_index != index || secondary_offset != offset)) {
        const auto secondary = ClassifyHandleWord(secondary_index, secondary_offset, want, compute);
        const HandleRef secondary_ref = MakeRef(secondary, secondary_index, secondary_offset);
        if (secondary.source == HandleSource::Unsupported) {
            return fail(fmt::format("{} (c{}[{:#x}])", secondary.reason, secondary_index,
                                    secondary_offset));
        }
        if (secondary_shift_left != 0) {
            return fail(fmt::format("secondary handle word shifted left by {}",
                                    secondary_shift_left));
        }
        const auto is_pair = [&](HandleSource a, HandleSource b) {
            return (primary.source == a && secondary.source == b) ||
                   (primary.source == b && secondary.source == a);
        };
        const bool primary_sampler = primary.source == HandleSource::SeparateSampler;
        if (is_pair(HandleSource::SeparateTexture, HandleSource::SeparateSampler) ||
            is_pair(HandleSource::UniformBuffer, HandleSource::SeparateSampler)) {
            result.texture = primary_sampler ? secondary_ref : primary_ref;
            result.sampler = primary_sampler ? primary_ref : secondary_ref;
        } else if (primary.source == HandleSource::UniformBuffer &&
                   secondary.source == HandleSource::UniformBuffer) {
            return fail("two uniform-buffer handle words: type key ambiguous");
        } else {
            return fail(fmt::format("unsupported handle word pair {} + {}",
                                    HandleSourceName(primary.source),
                                    HandleSourceName(secondary.source)));
        }
    } else {
        switch (primary.source) {
        case HandleSource::SeparateTexture:
            return fail("separate texture handle without a sampler word (constant sampler id?)");
        case HandleSource::SeparateSampler:
            return fail("sampler word without a texture word");
        default:
            break;
        }
        result.texture = primary_ref;
    }
    if (count > 1) {
        if (size_shift != 3) {
            return fail(fmt::format("handle array element size {} is not {}", 1U << size_shift,
                                    HandleStride));
        }
        for (const HandleRef* ref : {&result.texture, &result.sampler}) {
            if (ref->source == HandleSource::None) {
                continue;
            }
            const u32 capacity = TableCapacity(ref->source, compute);
            if (capacity != ~0U && (ref->slot >= capacity || count > capacity - ref->slot)) {
                return fail(fmt::format("handle array {} slots from slot {} overflows its table",
                                        count, ref->slot));
            }
        }
    }
    return result;
}

void Finish(DescriptorLayout& layout, DescriptorBinding&& binding) {
    if (!binding.supported) {
        ++layout.unsupported_count;
    }
    layout.bindings.push_back(std::move(binding));
}

DescriptorBinding Make(u32 binding, DescriptorKind kind, DescriptorRole role, const StageInfo& stage,
                       u32 count) {
    DescriptorBinding result;
    result.binding = binding;
    result.kind = kind;
    result.role = role;
    result.count = count;
    result.stage = stage.stage;
    result.stage_mask = 1U << static_cast<u32>(stage.stage);
    return result;
}

void ApplyHandles(DescriptorBinding& binding, HandleResult&& handles) {
    binding.texture = handles.texture;
    binding.sampler = handles.sampler;
    if (!handles.issue.empty()) {
        binding.supported = false;
        binding.issue = std::move(handles.issue);
    }
}

// A handle the shader reads from a table in memory: a program over uniform and storage words
// evaluated for each draw (Shader::Info::handle_programs), not a word of a handle bank.
void ApplyHandleProgram(DescriptorBinding& binding, const Shader::Info& info, u32 program) {
    if (program >= info.handle_programs.size()) {
        binding.supported = false;
        binding.issue = fmt::format("handle program {} of {}", program, info.handle_programs.size());
        return;
    }
    binding.texture = {HandleSource::Program, program, 0, 0};
    binding.handle_program = info.handle_programs[program];
}

} // namespace

DescriptorLayout BuildDescriptorLayout(std::span<const StageInfo> stages,
                                       const LayoutOptions& options) {
    DescriptorLayout layout;
    bool raw_stores_not_through = false;
    u32 binding = 0;
    for (const StageInfo& stage : stages) {
        const Shader::Info& info = stage.info;
        const ComputeHandleTables compute_tables =
            stage.compute != nullptr ? stage.compute->HandleTables() : ComputeHandleTables{};
        const ComputeHandleTables* compute = stage.compute != nullptr ? &compute_tables : nullptr;
        layout.uses_rescaling_uniform |= info.uses_rescaling_uniform;
        layout.uses_render_area |= info.uses_render_area;
        layout.uses_raw_global_memory |= info.uses_raw_global_memory;
        layout.stores_raw_global_memory |= info.stores_raw_global_memory;
        raw_stores_not_through |= info.stores_raw_global_memory && !RawStoresWrittenThrough(info);

        // 1. Constant buffers: the binding advances by `count` per descriptor; with descriptor
        // aliasing the whole block advances by the descriptor count instead (EmitSPIRV).
        u32 running = binding;
        for (const Shader::ConstantBufferDescriptor& desc : info.constant_buffer_descriptors) {
            DescriptorRole role = DescriptorRole::UniformBuffer;
            u32 slot = 0;
            switch (desc.index) {
            case DriverBank:
                role = DescriptorRole::DriverBank;
                break;
            case ProgramBank:
                role = DescriptorRole::ProgramConstants;
                break;
            case HandleBank:
                role = DescriptorRole::HandleTable;
                break;
            default:
                slot = desc.index - FirstUniformBank;
                break;
            }
            auto entry = Make(running, DescriptorKind::UniformBuffer, role, stage, desc.count);
            entry.slot = slot;
            Finish(layout, std::move(entry));
            running += desc.count;
        }
        binding = options.descriptor_aliasing
                      ? binding + static_cast<u32>(info.constant_buffer_descriptors.size())
                      : running;

        // 2. Storage buffers. Graphics reads its table from the per-stage driver bank table. A
        // compute shader's recompiler descriptors fall in the uniform or the storage entry table of
        // its control-derived driver bank (uniform buffers are reached through global memory too).
        // A descriptor whose address pair lives in a game-bound uniform buffer is read from there.
        u32 storage_index = 0;
        for (const Shader::StorageBufferDescriptor& desc : info.storage_buffers_descriptors) {
            auto entry = Make(binding, DescriptorKind::StorageBuffer, DescriptorRole::StorageBuffer,
                              stage, desc.count);
            entry.storage_index = storage_index++;
            entry.is_written = desc.is_written;
            entry.is_atomic_target = desc.is_atomic_target;
            entry.is_raw_global_table = Shader::IsRawGlobalTable(desc);

            entry.is_read = true;
            entry.storage_bank = desc.cbuf_index;
            entry.storage_offset = desc.cbuf_offset;
            const auto unsupported = [&entry](std::string issue) {
                entry.supported = false;
                entry.issue = std::move(issue);
            };
            if (desc.cbuf_index >= FirstUniformBank) {
                entry.storage_source = StorageSource::UniformBufferWord;
                entry.slot = desc.cbuf_index - FirstUniformBank;
            } else if (desc.cbuf_index != DriverBank) {
                unsupported(fmt::format("storage buffer descriptor in the {} bank",
                                        desc.cbuf_index == ProgramBank ? "program constant"
                                                                       : "handle"));
            } else if (stage.stage == ShaderStage::Compute) {
                const ComputeDriverLayout* layout_info = stage.compute;
                if (layout_info == nullptr) {
                    unsupported("compute storage buffer without a control-derived driver layout");
                } else if (const auto slot = StorageBufferSlot(layout_info->uniform_offset,
                                                              desc.cbuf_offset);
                           slot && *slot + desc.count <= layout_info->uniform_count) {
                    entry.storage_source = StorageSource::ComputeUniformEntry;
                    entry.slot = *slot;
                } else if (const auto storage = StorageBufferSlot(layout_info->storage_offset,
                                                                  desc.cbuf_offset);
                           storage && *storage + desc.count <= layout_info->storage_count) {
                    entry.storage_source = StorageSource::ComputeStorageEntry;
                    entry.slot = *storage;
                } else {
                    unsupported(fmt::format(
                        "compute storage descriptor at {:#x} is outside the uniform ({:#x}+{}) and "
                        "storage ({:#x}+{}) entry tables",
                        desc.cbuf_offset, layout_info->uniform_offset, layout_info->uniform_count,
                        layout_info->storage_offset, layout_info->storage_count));
                }
            } else if (const auto slot = StorageBufferSlot(stage.stage, desc.cbuf_offset);
                       slot && *slot + desc.count <= StorageBufferSlots) {
                entry.storage_source = StorageSource::DriverTable;
                entry.slot = *slot;
            } else {
                unsupported(fmt::format("storage descriptor at {:#x} is not in the {} table",
                                        desc.cbuf_offset, StageName(stage.stage)));
            }
            Finish(layout, std::move(entry));
            binding += desc.count;
        }

        // 3. Texture buffers.
        for (const Shader::TextureBufferDescriptor& desc : info.texture_buffer_descriptors) {
            auto entry = Make(binding, DescriptorKind::UniformTexelBuffer,
                              DescriptorRole::TexelBuffer, stage, desc.count);
            entry.type = Shader::TextureType::Buffer;
            if (desc.handle_program != ~0U) {
                ApplyHandleProgram(entry, info, desc.handle_program);
            } else {
                ApplyHandles(entry, ResolveHandles(desc.has_secondary, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.shift_left,
                                                   desc.secondary_cbuf_index,
                                                   desc.secondary_cbuf_offset,
                                                   desc.secondary_shift_left, desc.count,
                                                   desc.size_shift, HandleWant::Texture, compute));
            }
            Finish(layout, std::move(entry));
            ++binding;
        }

        // 4. Image buffers.
        for (const Shader::ImageBufferDescriptor& desc : info.image_buffer_descriptors) {
            auto entry = Make(binding, DescriptorKind::StorageTexelBuffer,
                              DescriptorRole::TexelBuffer, stage, desc.count);
            entry.type = Shader::TextureType::Buffer;
            entry.image_format = desc.format;
            entry.is_written = desc.is_written;
            entry.is_read = desc.is_read;
            entry.is_integer = desc.is_integer;
            if (desc.handle_program != ~0U) {
                ApplyHandleProgram(entry, info, desc.handle_program);
            } else {
                ApplyHandles(entry, ResolveHandles(false, desc.cbuf_index, desc.cbuf_offset, 0, 0,
                                                   0, 0, desc.count, desc.size_shift,
                                                   HandleWant::Image, compute));
            }
            Finish(layout, std::move(entry));
            ++binding;
        }

        // 5. Textures.
        for (const Shader::TextureDescriptor& desc : info.texture_descriptors) {
            if (desc.bindless) {
                // In the table's set, not numbered with the draw's descriptors (as EmitSPIRV).
                layout.uses_bindless_table = true;
                continue;
            }
            auto entry = Make(binding, DescriptorKind::CombinedImageSampler,
                              DescriptorRole::Texture, stage, desc.count);
            entry.type = desc.type;
            entry.is_depth = desc.is_depth;
            entry.is_multisample = desc.is_multisample;
            entry.is_integer = desc.is_integer;
            if (desc.handle_program != ~0U) {
                ApplyHandleProgram(entry, info, desc.handle_program);
            } else if (const auto stride = [&](const auto& tex) -> u32 {
                           if constexpr (requires { tex.element_stride; }) {
                               return tex.element_stride;
                           }
                           return 0U;
                       }(desc);
                       stride != 0) {
                // A set-bit indexed table (EDEN_NX2_SET_BIT_ARRAYS): its elements are
                // element_stride bytes apart, not consecutive handle slots.
                ApplyHandles(entry, HandleResult{.issue = fmt::format(
                                                     "set-bit handle table (element stride {})",
                                                     stride)});
            } else {
                ApplyHandles(entry, ResolveHandles(desc.has_secondary, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.shift_left,
                                                   desc.secondary_cbuf_index,
                                                   desc.secondary_cbuf_offset,
                                                   desc.secondary_shift_left, desc.count,
                                                   desc.size_shift, HandleWant::Texture, compute));
            }
            Finish(layout, std::move(entry));
            ++binding;
        }

        // 6. Images.
        for (const Shader::ImageDescriptor& desc : info.image_descriptors) {
            auto entry = Make(binding, DescriptorKind::StorageImage, DescriptorRole::Image, stage,
                              desc.count);
            entry.type = desc.type;
            entry.image_format = desc.format;
            entry.is_written = desc.is_written;
            entry.is_read = desc.is_read;
            entry.is_integer = desc.is_integer;
            if (desc.handle_program != ~0U) {
                ApplyHandleProgram(entry, info, desc.handle_program);
            } else {
                ApplyHandles(entry, ResolveHandles(false, desc.cbuf_index, desc.cbuf_offset, 0, 0,
                                                   0, 0, desc.count, desc.size_shift,
                                                   HandleWant::Image, compute));
            }
            Finish(layout, std::move(entry));
            ++binding;
        }
    }
    layout.binding_count = binding;
    layout.raw_stores_written_through = layout.stores_raw_global_memory && !raw_stores_not_through;
    return layout;
}

namespace {

std::string_view RoleName(DescriptorRole role) {
    switch (role) {
    case DescriptorRole::DriverBank:
        return "driver-bank";
    case DescriptorRole::ProgramConstants:
        return "program-constants";
    case DescriptorRole::HandleTable:
        return "handle-table";
    case DescriptorRole::UniformBuffer:
        return "uniform-buffer";
    case DescriptorRole::StorageBuffer:
        return "storage-buffer";
    case DescriptorRole::Texture:
        return "texture";
    case DescriptorRole::Image:
        return "image";
    case DescriptorRole::TexelBuffer:
        return "texel-buffer";
    }
    return "?";
}

std::string Describe(const HandleRef& ref) {
    if (ref.source == HandleSource::None) {
        return "-";
    }
    return fmt::format("{}[{}]@c{}+{:#x}", HandleSourceName(ref.source), ref.slot, ref.bank,
                       ref.byte_offset);
}

} // namespace

std::string DescribeLayout(const DescriptorLayout& layout) {
    std::string text;
    for (const DescriptorBinding& entry : layout.bindings) {
        text += fmt::format("{:>3} {} {} x{}", entry.binding, StageName(entry.stage),
                            RoleName(entry.role), entry.count);
        switch (entry.role) {
        case DescriptorRole::UniformBuffer:
        case DescriptorRole::StorageBuffer:
            text += fmt::format(" slot {}", entry.slot);
            break;
        case DescriptorRole::Texture:
        case DescriptorRole::Image:
        case DescriptorRole::TexelBuffer:
            text += fmt::format(" tex {} samp {}", Describe(entry.texture), Describe(entry.sampler));
            break;
        default:
            break;
        }
        if (!entry.supported) {
            text += " UNSUPPORTED: " + entry.issue;
        }
        text += '\n';
    }
    text += fmt::format("bindings {} unsupported {}{}{}\n", layout.binding_count,
                        layout.unsupported_count, layout.uses_rescaling_uniform ? " rescaling" : "",
                        layout.uses_render_area ? " render-area" : "");
    return text;
}

} // namespace VideoCommon::Nvn
