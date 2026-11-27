// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <span>

#include <boost/container/static_vector.hpp>

#include <fmt/ranges.h>

#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/env_gate.h"
#include "shader_recompiler/backend/nvn_storage_descriptor.h"
#include "shader_recompiler/bindless_table.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/raw_coverage_layout.h"

namespace Shader::Backend::SPIRV {
namespace {
enum class Operation {
    Increment,
    Decrement,
    FPAdd,
    FPMin,
    FPMax,
};

Id ImageType(EmitContext& ctx, const TextureDescriptor& desc) {
    const spv::ImageFormat format{spv::ImageFormat::Unknown};
    const Id type{desc.is_integer ? ctx.U32[1] : ctx.F32[1]};
    const bool depth{desc.is_depth};
    const bool ms{desc.is_multisample};
    switch (desc.type) {
    case TextureType::Color1D:
        return ctx.TypeImage(type, spv::Dim::Dim1D, depth, false, false, 1, format);
    case TextureType::ColorArray1D:
        return ctx.TypeImage(type, spv::Dim::Dim1D, depth, true, false, 1, format);
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return ctx.TypeImage(type, spv::Dim::Dim2D, depth, false, ms, 1, format);
    case TextureType::ColorArray2D:
        return ctx.TypeImage(type, spv::Dim::Dim2D, depth, true, ms, 1, format);
    case TextureType::Color3D:
        return ctx.TypeImage(type, spv::Dim::Dim3D, depth, false, false, 1, format);
    case TextureType::ColorCube:
        return ctx.TypeImage(type, spv::Dim::Cube, depth, false, false, 1, format);
    case TextureType::ColorArrayCube:
        return ctx.TypeImage(type, spv::Dim::Cube, depth, true, false, 1, format);
    case TextureType::Buffer:
        break;
    }
    throw InvalidArgument("Invalid texture type {}", desc.type);
}

spv::ImageFormat GetImageFormat(ImageFormat format) {
    switch (format) {
    case ImageFormat::Typeless:
        return spv::ImageFormat::Unknown;
    case ImageFormat::R8_UINT:
        return spv::ImageFormat::R8ui;
    case ImageFormat::R8_SINT:
        return spv::ImageFormat::R8i;
    case ImageFormat::R16_UINT:
        return spv::ImageFormat::R16ui;
    case ImageFormat::R16_SINT:
        return spv::ImageFormat::R16i;
    case ImageFormat::R32_UINT:
        return spv::ImageFormat::R32ui;
    case ImageFormat::R32G32_UINT:
        return spv::ImageFormat::Rg32ui;
    case ImageFormat::R32G32B32A32_UINT:
        return spv::ImageFormat::Rgba32ui;
    }
    throw InvalidArgument("Invalid image format {}", format);
}

Id ImageType(EmitContext& ctx, const ImageDescriptor& desc, Id sampled_type) {
    const spv::ImageFormat format{GetImageFormat(desc.format)};
    switch (desc.type) {
    case TextureType::Color1D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim1D, false, false, false, 2, format);
    case TextureType::ColorArray1D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim1D, false, true, false, 2, format);
    case TextureType::Color2D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, false, false, 2, format);
    case TextureType::ColorArray2D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, true, false, 2, format);
    case TextureType::Color3D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim3D, false, false, false, 2, format);
    case TextureType::Buffer:
        throw NotImplementedException("Image buffer");
    default:
        break;
    }
    throw InvalidArgument("Invalid texture type {}", desc.type);
}

Id DefineVariable(EmitContext& ctx, Id type, std::optional<spv::BuiltIn> builtin,
                  spv::StorageClass storage_class, std::optional<Id> initializer = std::nullopt) {
    const Id pointer_type{ctx.TypePointer(storage_class, type)};
    const Id id{ctx.AddGlobalVariable(pointer_type, storage_class, initializer)};
    if (builtin) {
        ctx.Decorate(id, spv::Decoration::BuiltIn, *builtin);
    }
    ctx.interfaces.push_back(id);
    return id;
}

u32 NumVertices(InputTopology input_topology) {
    switch (input_topology) {
    case InputTopology::Points:
        return 1;
    case InputTopology::Lines:
        return 2;
    case InputTopology::LinesAdjacency:
        return 4;
    case InputTopology::Triangles:
        return 3;
    case InputTopology::TrianglesAdjacency:
        return 6;
    }
    throw InvalidArgument("Invalid input topology {}", input_topology);
}

Id DefineInput(EmitContext& ctx, Id type, bool per_invocation,
               std::optional<spv::BuiltIn> builtin = std::nullopt) {
    switch (ctx.stage) {
    case Stage::TessellationControl:
    case Stage::TessellationEval:
        if (per_invocation) {
            type = ctx.TypeArray(type, ctx.Const(32u));
        }
        break;
    case Stage::Geometry:
        if (per_invocation) {
            const u32 num_vertices{NumVertices(ctx.runtime_info.input_topology)};
            type = ctx.TypeArray(type, ctx.Const(num_vertices));
        }
        break;
    default:
        break;
    }
    return DefineVariable(ctx, type, builtin, spv::StorageClass::Input);
}

Id DefineOutput(EmitContext& ctx, Id type, std::optional<u32> invocations,
                std::optional<spv::BuiltIn> builtin = std::nullopt,
                std::optional<Id> initializer = std::nullopt) {
    if (invocations && ctx.stage == Stage::TessellationControl) {
        type = ctx.TypeArray(type, ctx.Const(*invocations));
    }
    return DefineVariable(ctx, type, builtin, spv::StorageClass::Output, initializer);
}

void DefineGenericOutput(EmitContext& ctx, size_t index, std::optional<u32> invocations) {
    static constexpr std::string_view swizzle{"xyzw"};
    const size_t base_attr_index{static_cast<size_t>(IR::Attribute::Generic0X) + index * 4};
    u32 element{0};
    while (element < 4) {
        const u32 remainder{4 - element};
        const TransformFeedbackVarying* xfb_varying{};
        const size_t xfb_varying_index{base_attr_index + element};
        if (xfb_varying_index < ctx.runtime_info.xfb_count) {
            xfb_varying = &ctx.runtime_info.xfb_varyings[xfb_varying_index];
            xfb_varying = xfb_varying->components > 0 ? xfb_varying : nullptr;
        }
        const u32 num_components{xfb_varying ? xfb_varying->components : remainder};

        const Id id{DefineOutput(ctx, ctx.F32[num_components], invocations)};
        ctx.Decorate(id, spv::Decoration::Location, static_cast<u32>(index));
        if (element > 0) {
            ctx.Decorate(id, spv::Decoration::Component, element);
        }
        if (xfb_varying) {
            ctx.Decorate(id, spv::Decoration::XfbBuffer, xfb_varying->buffer);
            ctx.Decorate(id, spv::Decoration::XfbStride, xfb_varying->stride);
            ctx.Decorate(id, spv::Decoration::Offset, xfb_varying->offset);
            if (ctx.stage == Stage::Geometry && xfb_varying->stream != 0) {
                ctx.Decorate(id, spv::Decoration::Stream, xfb_varying->stream);
            }
        }
        if (num_components < 4 || element > 0) {
            const std::string_view subswizzle{swizzle.substr(element, num_components)};
            ctx.Name(id, fmt::format("out_attr{}_{}", index, subswizzle));
        } else {
            ctx.Name(id, fmt::format("out_attr{}", index));
        }
        const GenericElementInfo info{
            .id = id,
            .first_element = element,
            .num_components = num_components,
        };
        std::fill_n(ctx.output_generics[index].begin() + element, num_components, info);
        element += num_components;
    }
}

Id GetAttributeType(EmitContext& ctx, AttributeType type) {
    switch (type) {
    case AttributeType::Float:
        return ctx.F32[4];
    case AttributeType::SignedInt:
        return ctx.TypeVector(ctx.TypeInt(32, true), 4);
    case AttributeType::UnsignedInt:
        return ctx.U32[4];
    case AttributeType::SignedScaled:
        return ctx.profile.support_scaled_attributes ? ctx.F32[4]
                                                     : ctx.TypeVector(ctx.TypeInt(32, true), 4);
    case AttributeType::UnsignedScaled:
        return ctx.profile.support_scaled_attributes ? ctx.F32[4] : ctx.U32[4];
    case AttributeType::Disabled:
        break;
    }
    throw InvalidArgument("Invalid attribute type {}", type);
}

InputGenericInfo GetAttributeInfo(EmitContext& ctx, AttributeType type, Id id) {
    switch (type) {
    case AttributeType::Float:
        return InputGenericInfo{id, ctx.input_f32, ctx.F32[1], InputGenericLoadOp::None};
    case AttributeType::UnsignedInt:
        return InputGenericInfo{id, ctx.input_u32, ctx.U32[1], InputGenericLoadOp::Bitcast};
    case AttributeType::SignedInt:
        return InputGenericInfo{id, ctx.input_s32, ctx.TypeInt(32, true),
                                InputGenericLoadOp::Bitcast};
    case AttributeType::SignedScaled:
        return ctx.profile.support_scaled_attributes
                   ? InputGenericInfo{id, ctx.input_f32, ctx.F32[1], InputGenericLoadOp::None}
                   : InputGenericInfo{id, ctx.input_s32, ctx.TypeInt(32, true),
                                      InputGenericLoadOp::SToF};
    case AttributeType::UnsignedScaled:
        return ctx.profile.support_scaled_attributes
                   ? InputGenericInfo{id, ctx.input_f32, ctx.F32[1], InputGenericLoadOp::None}
                   : InputGenericInfo{id, ctx.input_u32, ctx.U32[1], InputGenericLoadOp::UToF};
    case AttributeType::Disabled:
        return InputGenericInfo{};
    }
    throw InvalidArgument("Invalid attribute type {}", type);
}

std::string_view StageName(Stage stage) {
    switch (stage) {
    case Stage::VertexA:
        return "vs_a";
    case Stage::VertexB:
        return "vs";
    case Stage::TessellationControl:
        return "tcs";
    case Stage::TessellationEval:
        return "tes";
    case Stage::Geometry:
        return "gs";
    case Stage::Fragment:
        return "fs";
    case Stage::Compute:
        return "cs";
    case Stage::Task:
        return "ts";
    case Stage::Mesh:
        return "ms";
    }
    throw InvalidArgument("Invalid stage {}", stage);
}

template <typename... Args>
void Name(EmitContext& ctx, Id object, std::string_view format_str, Args&&... args) {
    ctx.Name(object, fmt::format(fmt::runtime(format_str), StageName(ctx.stage),
                                 std::forward<Args>(args)...)
                         .c_str());
}

void DefineConstBuffers(EmitContext& ctx, const Info& info, Id UniformDefinitions::*member_type,
                        u32 binding, Id type, char type_char, u32 element_size) {
    const Id array_type{ctx.TypeArray(type, ctx.Const(65536U / element_size))};
    ctx.Decorate(array_type, spv::Decoration::ArrayStride, element_size);

    const Id struct_type{ctx.TypeStruct(array_type)};
    Name(ctx, struct_type, "{}_cbuf_block_{}{}", ctx.stage, type_char, element_size * CHAR_BIT);
    ctx.Decorate(struct_type, spv::Decoration::Block);
    ctx.MemberName(struct_type, 0, "data");
    ctx.MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);

    const Id struct_pointer_type{ctx.TypePointer(spv::StorageClass::Uniform, struct_type)};
    const Id uniform_type{ctx.TypePointer(spv::StorageClass::Uniform, type)};
    ctx.uniform_types.*member_type = uniform_type;

    for (const ConstantBufferDescriptor& desc : info.constant_buffer_descriptors) {
        const Id id{ctx.AddGlobalVariable(struct_pointer_type, spv::StorageClass::Uniform)};
        ctx.Decorate(id, spv::Decoration::Binding, binding);
        ctx.Decorate(id, spv::Decoration::DescriptorSet, 0U);
        ctx.Name(id, fmt::format("c{}", desc.index));
        for (size_t i = 0; i < desc.count; ++i) {
            ctx.cbufs[desc.index + i].*member_type = id;
        }
        if (ctx.profile.supported_spirv >= 0x00010400) {
            ctx.interfaces.push_back(id);
        }
        binding += desc.count;
    }
}

void DefineSsbos(EmitContext& ctx, StorageTypeDefinition& type_def,
                 Id StorageDefinitions::*member_type, const Info& info, u32 binding, Id type,
                 u32 stride) {
    const Id array_type{ctx.TypeRuntimeArray(type)};
    ctx.Decorate(array_type, spv::Decoration::ArrayStride, stride);

    const Id struct_type{ctx.TypeStruct(array_type)};
    ctx.Decorate(struct_type, spv::Decoration::Block);
    ctx.MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);
    if (info.requires_coherent_global_memory) {
        ctx.MemberDecorate(struct_type, 0, spv::Decoration::Coherent);
    }

    const Id struct_pointer{ctx.TypePointer(spv::StorageClass::StorageBuffer, struct_type)};
    type_def.array = struct_pointer;
    type_def.element = ctx.TypePointer(spv::StorageClass::StorageBuffer, type);

    u32 index{};
    for (const StorageBufferDescriptor& desc : info.storage_buffers_descriptors) {
        const Id id{ctx.AddGlobalVariable(struct_pointer, spv::StorageClass::StorageBuffer)};
        ctx.Decorate(id, spv::Decoration::Binding, binding);
        ctx.Decorate(id, spv::Decoration::DescriptorSet, 0U);
        ctx.Name(id, fmt::format("ssbo{}", index));
        if (ctx.profile.supported_spirv >= 0x00010400) {
            ctx.interfaces.push_back(id);
        }
        for (size_t i = 0; i < desc.count; ++i) {
            ctx.ssbos[index + i].*member_type = id;
        }
        index += desc.count;
        binding += desc.count;
    }
}

Id CasFunction(EmitContext& ctx, Operation operation, Id value_type) {
    const Id func_type{ctx.TypeFunction(value_type, value_type, value_type)};
    const Id func{ctx.OpFunction(value_type, spv::FunctionControlMask::MaskNone, func_type)};
    const Id op_a{ctx.OpFunctionParameter(value_type)};
    const Id op_b{ctx.OpFunctionParameter(value_type)};
    ctx.AddLabel();
    Id result{};
    switch (operation) {
    case Operation::Increment: {
        const Id pred{ctx.OpUGreaterThanEqual(ctx.U1, op_a, op_b)};
        const Id incr{ctx.OpIAdd(value_type, op_a, ctx.Constant(value_type, 1))};
        result = ctx.OpSelect(value_type, pred, ctx.u32_zero_value, incr);
        break;
    }
    case Operation::Decrement: {
        const Id lhs{ctx.OpIEqual(ctx.U1, op_a, ctx.Constant(value_type, 0u))};
        const Id rhs{ctx.OpUGreaterThan(ctx.U1, op_a, op_b)};
        const Id pred{ctx.OpLogicalOr(ctx.U1, lhs, rhs)};
        const Id decr{ctx.OpISub(value_type, op_a, ctx.Constant(value_type, 1))};
        result = ctx.OpSelect(value_type, pred, op_b, decr);
        break;
    }
    case Operation::FPAdd:
        result = ctx.OpFAdd(value_type, op_a, op_b);
        break;
    case Operation::FPMin:
        result = ctx.OpNMin(value_type, op_a, op_b);
        break;
    case Operation::FPMax:
        result = ctx.OpNMax(value_type, op_a, op_b);
        break;
    default:
        break;
    }
    ctx.OpReturnValue(result);
    ctx.OpFunctionEnd();
    return func;
}

// A/B gate: with EGLANTINE_CAS_RETRY_FIX=0 each retry reloads the word instead of reusing the
// compare-exchange result.
bool CasRetryFix() {
    static const bool enabled{Common::EnvGate("EGLANTINE_CAS_RETRY_FIX")};
    return enabled;
}

Id CasLoop(EmitContext& ctx, Operation operation, Id array_pointer, Id element_pointer,
           Id value_type, spv::Scope scope) {
    const bool is_shared{scope == spv::Scope::Workgroup};
    const bool is_struct{!is_shared || ctx.uses_explicit_workgroup_layout};
    const Id cas_func{CasFunction(ctx, operation, value_type)};
    const Id zero{ctx.u32_zero_value};
    const Id scope_id{ctx.Const(static_cast<u32>(scope))};

    const Id loop_header{ctx.OpLabel()};
    const Id continue_block{ctx.OpLabel()};
    const Id merge_block{ctx.OpLabel()};
    const Id func_type{is_shared
                           ? ctx.TypeFunction(value_type, ctx.U32[1], value_type)
                           : ctx.TypeFunction(value_type, ctx.U32[1], value_type, array_pointer)};

    const Id func{ctx.OpFunction(value_type, spv::FunctionControlMask::MaskNone, func_type)};
    const Id index{ctx.OpFunctionParameter(ctx.U32[1])};
    const Id op_b{ctx.OpFunctionParameter(value_type)};
    const Id base{is_shared ? ctx.shared_memory_u32 : ctx.OpFunctionParameter(array_pointer)};
    ctx.AddLabel();
    // Each attempt expects the word the previous compare-exchange returned. A plain reload can
    // keep returning a stale copy (AMD), and the invocations that lost would never match again.
    const Id expected_var{ctx.AddLocalVariable(
        ctx.TypePointer(spv::StorageClass::Function, ctx.U32[1]), spv::StorageClass::Function)};
    const Id word_pointer{is_struct ? ctx.OpAccessChain(element_pointer, base, zero, index)
                                    : ctx.OpAccessChain(element_pointer, base, index)};
    ctx.OpStore(expected_var, ctx.OpLoad(ctx.U32[1], word_pointer));
    ctx.OpBranch(loop_header);
    ctx.AddLabel(loop_header);

    ctx.OpLoopMerge(merge_block, continue_block, spv::LoopControlMask::MaskNone);
    ctx.OpBranch(continue_block);

    ctx.AddLabel(continue_block);
    // The word holds a u32, an f32, an f16x2, or an f32x2 packed as two halves.
    const bool packed_half{value_type.value == ctx.F32[2].value};
    const bool is_u32{value_type.value == ctx.U32[1].value};
    const auto from_word{[&](Id word) {
        if (packed_half) {
            return ctx.OpUnpackHalf2x16(ctx.F32[2], word);
        }
        return is_u32 ? word : ctx.OpBitcast(value_type, word);
    }};
    const Id expected{ctx.OpLoad(ctx.U32[1], expected_var)};
    const Id new_value{ctx.OpFunctionCall(value_type, cas_func, from_word(expected), op_b)};
    const Id new_word{packed_half ? ctx.OpPackHalf2x16(ctx.U32[1], new_value)
                      : is_u32    ? new_value
                                  : ctx.OpBitcast(ctx.U32[1], new_value)};
    const Id atomic_res{ctx.OpAtomicCompareExchange(ctx.U32[1], word_pointer, scope_id, zero, zero,
                                                    new_word, expected)};
    ctx.OpStore(expected_var, atomic_res);
    const Id success{ctx.OpIEqual(ctx.U1, atomic_res, expected)};
    ctx.OpBranchConditional(success, merge_block, loop_header);

    ctx.AddLabel(merge_block);
    ctx.OpReturnValue(from_word(atomic_res));
    ctx.OpFunctionEnd();
    return func;
}

// VTX-HLE raw global memory (DefineRawGlobalMemoryFunctions): the compare-exchange loop and
// the atomics on the word a guest address resolves to.

/// The compare-exchange loop of CasLoop on the word `word_pointer()` yields (called before the
/// loop). Leaves the module in `merge_block` and returns the value the word held before the
/// update. The word is read once; each further attempt expects the word the previous
/// compare-exchange returned, kept in `expected_var` (a Function U32 variable, declared in the
/// entry block of the function): a reload can keep returning a stale copy (AMD), and the
/// invocations that lost would never match again. `load_alignment` is the byte alignment of the
/// plain load (PhysicalStorageBuffer pointers require it), 0 for none.
template <typename WordPointer>
Id CasLoopBody(EmitContext& ctx, Id cas_func, Id op_b, Id value_type, Id expected_var,
               spv::Scope scope, Id loop_header, Id continue_block, Id merge_block,
               u32 load_alignment, WordPointer&& word_pointer) {
    const Id zero{ctx.u32_zero_value};
    const Id scope_id{ctx.Const(static_cast<u32>(scope))};
    const Id pointer{word_pointer()};
    ctx.OpStore(expected_var, load_alignment != 0
                                  ? ctx.OpLoad(ctx.U32[1], pointer, spv::MemoryAccessMask::Aligned,
                                               load_alignment)
                                  : ctx.OpLoad(ctx.U32[1], pointer));
    ctx.OpBranch(loop_header);
    ctx.AddLabel(loop_header);

    ctx.OpLoopMerge(merge_block, continue_block, spv::LoopControlMask::MaskNone);
    ctx.OpBranch(continue_block);

    ctx.AddLabel(continue_block);
    // The word holds a u32, an f32, an f16x2, or an f32x2 packed as two halves.
    const bool packed_half{value_type.value == ctx.F32[2].value};
    const bool is_u32{value_type.value == ctx.U32[1].value};
    const auto from_word{[&](Id word) {
        if (packed_half) {
            return ctx.OpUnpackHalf2x16(ctx.F32[2], word);
        }
        return is_u32 ? word : ctx.OpBitcast(value_type, word);
    }};
    const Id expected{ctx.OpLoad(ctx.U32[1], expected_var)};
    const Id new_value{ctx.OpFunctionCall(value_type, cas_func, from_word(expected), op_b)};
    const Id new_word{packed_half ? ctx.OpPackHalf2x16(ctx.U32[1], new_value)
                      : is_u32    ? new_value
                                  : ctx.OpBitcast(ctx.U32[1], new_value)};
    const Id atomic_res{ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope_id, zero, zero,
                                                    new_word, expected)};
    ctx.OpStore(expected_var, atomic_res);
    const Id success{ctx.OpIEqual(ctx.U1, atomic_res, expected)};
    ctx.OpBranchConditional(success, merge_block, loop_header);

    ctx.AddLabel(merge_block);
    return from_word(atomic_res);
}

/// A Function-storage U32 variable for CasLoopBody; declare it right after the function's first
/// label (OpVariable belongs to the entry block).
Id CasExpectedVariable(EmitContext& ctx) {
    return ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, ctx.U32[1]),
                                spv::StorageClass::Function);
}

/// What a GlobalAtomic* opcode does to the word its address resolves to.
struct GlobalAtomicSpec {
    Id value_type;  // the operand
    Id result_type; // the value of the instruction: the word before the update
    Id word_type;   // the type of the word in memory: U32 or U64
    /// A SPIR-V atomic on the word, or null for an operation that is a compare-exchange loop.
    Id (Sirit::Module::*atomic)(Id, Id, Id, Id, Id){};
    Id cas_func{}; // the arithmetic of the compare-exchange loop
    enum class Result { Plain, F16x2Bits, F32x2Packed } result{Result::Plain};
};

/// The definition of a GlobalAtomic* opcode, or nothing when the host cannot do it: a 64-bit
/// atomic needs shaderBufferInt64Atomics (and, on the storage buffer fallback, a 64-bit view of the
/// buffers); the two 32-bit halves forms (GlobalAtomic*32x2) only exist for hosts without int64,
/// which have no global memory at all. Defines the arithmetic function of a compare-exchange loop,
/// so it has to be called between function definitions.
std::optional<GlobalAtomicSpec> GlobalAtomicSpecOf(EmitContext& ctx, IR::Opcode opcode, bool raw) {
    using Result = GlobalAtomicSpec::Result;
    const auto direct{[&](Id type, Id (Sirit::Module::*atomic)(Id, Id, Id, Id, Id)) {
        return GlobalAtomicSpec{.value_type = type,
                                .result_type = type,
                                .word_type = type,
                                .atomic = atomic};
    }};
    const auto loop{[&](Operation operation, Id value_type, Id result_type, Result result) {
        return GlobalAtomicSpec{.value_type = value_type,
                                .result_type = result_type,
                                .word_type = ctx.U32[1],
                                .cas_func = CasFunction(ctx, operation, value_type),
                                .result = result};
    }};
    const bool atomics64{ctx.profile.support_int64_atomics &&
                         (raw || ctx.profile.support_descriptor_aliasing)};
    using Module = Sirit::Module;
    switch (opcode) {
    case IR::Opcode::GlobalAtomicIAdd32:
        return direct(ctx.U32[1], &Module::OpAtomicIAdd);
    case IR::Opcode::GlobalAtomicSMin32:
        return direct(ctx.U32[1], &Module::OpAtomicSMin);
    case IR::Opcode::GlobalAtomicUMin32:
        return direct(ctx.U32[1], &Module::OpAtomicUMin);
    case IR::Opcode::GlobalAtomicSMax32:
        return direct(ctx.U32[1], &Module::OpAtomicSMax);
    case IR::Opcode::GlobalAtomicUMax32:
        return direct(ctx.U32[1], &Module::OpAtomicUMax);
    case IR::Opcode::GlobalAtomicAnd32:
        return direct(ctx.U32[1], &Module::OpAtomicAnd);
    case IR::Opcode::GlobalAtomicOr32:
        return direct(ctx.U32[1], &Module::OpAtomicOr);
    case IR::Opcode::GlobalAtomicXor32:
        return direct(ctx.U32[1], &Module::OpAtomicXor);
    case IR::Opcode::GlobalAtomicExchange32:
        return direct(ctx.U32[1], &Module::OpAtomicExchange);
    case IR::Opcode::GlobalAtomicInc32:
        return loop(Operation::Increment, ctx.U32[1], ctx.U32[1], Result::Plain);
    case IR::Opcode::GlobalAtomicDec32:
        return loop(Operation::Decrement, ctx.U32[1], ctx.U32[1], Result::Plain);
    case IR::Opcode::GlobalAtomicIAdd64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicIAdd)} : std::nullopt;
    case IR::Opcode::GlobalAtomicSMin64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicSMin)} : std::nullopt;
    case IR::Opcode::GlobalAtomicUMin64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicUMin)} : std::nullopt;
    case IR::Opcode::GlobalAtomicSMax64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicSMax)} : std::nullopt;
    case IR::Opcode::GlobalAtomicUMax64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicUMax)} : std::nullopt;
    case IR::Opcode::GlobalAtomicAnd64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicAnd)} : std::nullopt;
    case IR::Opcode::GlobalAtomicOr64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicOr)} : std::nullopt;
    case IR::Opcode::GlobalAtomicXor64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicXor)} : std::nullopt;
    case IR::Opcode::GlobalAtomicExchange64:
        return atomics64 ? std::optional{direct(ctx.U64, &Module::OpAtomicExchange)} : std::nullopt;
    case IR::Opcode::GlobalAtomicAddF32:
        return loop(Operation::FPAdd, ctx.F32[1], ctx.F32[1], Result::Plain);
    case IR::Opcode::GlobalAtomicAddF16x2:
        return loop(Operation::FPAdd, ctx.F16[2], ctx.U32[1], Result::F16x2Bits);
    case IR::Opcode::GlobalAtomicMinF16x2:
        return loop(Operation::FPMin, ctx.F16[2], ctx.U32[1], Result::F16x2Bits);
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return loop(Operation::FPMax, ctx.F16[2], ctx.U32[1], Result::F16x2Bits);
    case IR::Opcode::GlobalAtomicAddF32x2:
        return loop(Operation::FPAdd, ctx.F32[2], ctx.U32[1], Result::F32x2Packed);
    case IR::Opcode::GlobalAtomicMinF32x2:
        return loop(Operation::FPMin, ctx.F32[2], ctx.U32[1], Result::F32x2Packed);
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return loop(Operation::FPMax, ctx.F32[2], ctx.U32[1], Result::F32x2Packed);
    default:
        return std::nullopt;
    }
}

/// The atomic of `spec` on the word `pointer` points to (a StorageBuffer or PhysicalStorageBuffer
/// pointer, whose plain loads need `load_alignment`). Returns the instruction's value.
/// `expected_var` is the CasExpectedVariable of the enclosing function for a compare-exchange loop
/// (spec.atomic null), unused otherwise.
Id EmitGlobalAtomicOnWord(EmitContext& ctx, const GlobalAtomicSpec& spec, Id pointer, Id value,
                          u32 load_alignment, Id expected_var) {
    constexpr spv::Scope scope{spv::Scope::Device};
    if (spec.atomic) {
        return (ctx.*spec.atomic)(spec.result_type, pointer, ctx.Const(static_cast<u32>(scope)),
                                  ctx.u32_zero_value, value);
    }
    const Id loop_header{ctx.OpLabel()};
    const Id continue_block{ctx.OpLabel()};
    const Id merge_block{ctx.OpLabel()};
    const Id before{CasLoopBody(ctx, spec.cas_func, value, spec.value_type, expected_var, scope,
                                loop_header, continue_block, merge_block, load_alignment,
                                [&] { return pointer; })};
    switch (spec.result) {
    case GlobalAtomicSpec::Result::F16x2Bits:
        return ctx.OpBitcast(ctx.U32[1], before);
    case GlobalAtomicSpec::Result::F32x2Packed:
        return ctx.OpPackHalf2x16(ctx.U32[1], before);
    case GlobalAtomicSpec::Result::Plain:
        break;
    }
    return before;
}

bool LeanGlobalMemory(Stage stage) {
    static const char* const forced = std::getenv("EDEN_NX2_LEAN_SSBO");
    if (forced != nullptr) {
        return forced[0] != '0';
    }
    return stage != Stage::Task && stage != Stage::Mesh;
}

template <typename Desc>
std::string NameOf(Stage stage, const Desc& desc, std::string_view prefix) {
    if (desc.count > 1) {
        return fmt::format("{}_{}{}_{:02x}x{}", StageName(stage), prefix, desc.cbuf_index,
                           desc.cbuf_offset, desc.count);
    } else {
        return fmt::format("{}_{}{}_{:02x}", StageName(stage), prefix, desc.cbuf_index,
                           desc.cbuf_offset);
    }
}

Id DescType(EmitContext& ctx, Id sampled_type, Id pointer_type, u32 count) {
    if (count > 1) {
        const Id array_type{ctx.TypeArray(sampled_type, ctx.Const(count))};
        return ctx.TypePointer(spv::StorageClass::UniformConstant, array_type);
    } else {
        return pointer_type;
    }
}
// EDEN_NX2_LEAN_SSBO=0 turns it off: the global memory helpers are kept as calls and route
// with one switch. Inlined at each of their calls, a program with ~30 candidate buffers took
// NVIDIA's compiler 2.4 s (Ocarina's foliage compute programs; 0.6 s this way, same accesses).
bool LeanGlobalMemory() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_LEAN_SSBO");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}
} // Anonymous namespace

void VectorTypes::Define(Sirit::Module& sirit_ctx, Id base_type, std::string_view name) {
    defs[0] = sirit_ctx.Name(base_type, name);

    std::array<char, 6> def_name;
    for (int i = 1; i < 4; ++i) {
        const std::string_view def_name_view(
            def_name.data(),
            fmt::format_to_n(def_name.data(), def_name.size(), "{}x{}", name, i + 1).size);
        defs[static_cast<size_t>(i)] =
            sirit_ctx.Name(sirit_ctx.TypeVector(base_type, i + 1), def_name_view);
    }
}

EmitContext::EmitContext(const Profile& profile_, const RuntimeInfo& runtime_info_,
                         IR::Program& program, Bindings& bindings)
    : Sirit::Module(profile_.supported_spirv), profile{profile_}, runtime_info{runtime_info_},
      requires_coherent_global_memory{program.info.requires_coherent_global_memory},
      requires_coherent_image_memory{program.info.requires_coherent_image_memory},
      stage{program.stage}, texture_rescaling_index{bindings.texture_scaling_index},
      image_rescaling_index{bindings.image_scaling_index} {
    const bool is_unified{profile.unified_descriptor_binding};
    u32& uniform_binding{is_unified ? bindings.unified : bindings.uniform_buffer};
    u32& storage_binding{is_unified ? bindings.unified : bindings.storage_buffer};
    u32& texture_binding{is_unified ? bindings.unified : bindings.texture};
    u32& image_binding{is_unified ? bindings.unified : bindings.image};
    AddCapability(spv::Capability::Shader);
    DefineCommonTypes(program.info);
    DefineCommonConstants();
    DefineInterfaces(program);
    DefineMesh(program);
    DefineLocalMemory(program);
    DefineSharedMemory(program);
    DefineSharedMemoryFunctions(program);
    DefineConstantBuffers(program.info, uniform_binding);
    DefineConstantBufferIndirectFunctions(program.info);
    DefineStorageBuffers(program.info, storage_binding);
    DefineTextureBuffers(program.info, texture_binding);
    DefineImageBuffers(program.info, image_binding);
    DefineTextures(program.info, texture_binding, bindings.texture_scaling_index);
    DefineImages(program.info, image_binding, bindings.image_scaling_index);
    DefineAttributeMemAccess(program.info);
    DefineWriteStorageCasLoopFunction(program.info);
    DefineGlobalMemoryFunctions(program);
    DefineMeshEmulationOutput();
    DefineRescalingInput(program.info);
    DefineRenderArea(program.info);
}

EmitContext::~EmitContext() = default;

Id EmitContext::Def(const IR::Value& value) {
    if (!value.IsImmediate()) {
        return value.InstRecursive()->Definition<Id>();
    }
    switch (value.Type()) {
    case IR::Type::Void:
        // Void instructions are used for optional arguments (e.g. texture offsets)
        // They are not meant to be used in the SPIR-V module
        return Id{};
    case IR::Type::U1:
        return value.U1() ? true_value : false_value;
    case IR::Type::U32:
        return Const(value.U32());
    case IR::Type::U64:
        return Constant(U64, value.U64());
    case IR::Type::F32:
        return Const(value.F32());
    case IR::Type::F64:
        return Constant(F64[1], value.F64());
    default:
        throw NotImplementedException("Immediate type {}", value.Type());
    }
}

Id EmitContext::BitOffset8(const IR::Value& offset) {
    if (offset.IsImmediate()) {
        return Const((offset.U32() % 4) * 8);
    }
    return OpBitwiseAnd(U32[1], OpShiftLeftLogical(U32[1], Def(offset), Const(3u)), Const(24u));
}

Id EmitContext::BitOffset16(const IR::Value& offset) {
    if (offset.IsImmediate()) {
        return Const(((offset.U32() / 2) % 2) * 16);
    }
    return OpBitwiseAnd(U32[1], OpShiftLeftLogical(U32[1], Def(offset), Const(3u)), Const(16u));
}

void EmitContext::DefineCommonTypes(const Info& info) {
    void_id = TypeVoid();

    U1 = Name(TypeBool(), "u1");

    F32.Define(*this, TypeFloat(32), "f32");
    U32.Define(*this, TypeInt(32, false), "u32");
    S32.Define(*this, TypeInt(32, true), "s32");

    private_u32 = Name(TypePointer(spv::StorageClass::Private, U32[1]), "private_u32");

    input_f32 = Name(TypePointer(spv::StorageClass::Input, F32[1]), "input_f32");
    input_u32 = Name(TypePointer(spv::StorageClass::Input, U32[1]), "input_u32");
    input_s32 = Name(TypePointer(spv::StorageClass::Input, TypeInt(32, true)), "input_s32");

    output_f32 = Name(TypePointer(spv::StorageClass::Output, F32[1]), "output_f32");
    output_u32 = Name(TypePointer(spv::StorageClass::Output, U32[1]), "output_u32");

    if (info.uses_int8 && profile.support_int8) {
        AddCapability(spv::Capability::Int8);
        U8 = Name(TypeInt(8, false), "u8");
        S8 = Name(TypeInt(8, true), "s8");
    }
    if (info.uses_int16 && profile.support_int16) {
        AddCapability(spv::Capability::Int16);
        U16 = Name(TypeInt(16, false), "u16");
        S16 = Name(TypeInt(16, true), "s16");
    }
    if (info.uses_int64 && profile.support_int64) {
        AddCapability(spv::Capability::Int64);
        U64 = Name(TypeInt(64, false), "u64");
    }
    if (info.uses_fp16) {
        AddCapability(spv::Capability::Float16);
        F16.Define(*this, TypeFloat(16), "f16");
    }
    if (info.uses_fp64) {
        AddCapability(spv::Capability::Float64);
        F64.Define(*this, TypeFloat(64), "f64");
    }
}

void EmitContext::DefineCommonConstants() {
    true_value = ConstantTrue(U1);
    false_value = ConstantFalse(U1);
    u32_zero_value = Const(0U);
    f32_zero_value = Const(0.0f);
}

void EmitContext::DefineInterfaces(const IR::Program& program) {
    DefineInputs(program);
    DefineOutputs(program);
}

void EmitContext::DefineLocalMemory(const IR::Program& program) {
    if (program.local_memory_size == 0) {
        return;
    }
    const u32 num_elements{Common::DivCeil(program.local_memory_size, 4U)};
    const Id element_type{TypeStruct(U32[1])};
    const Id type{TypeArray(element_type, Const(num_elements))};
    const Id pointer{TypePointer(spv::StorageClass::Private, type)};
    local_memory = AddGlobalVariable(pointer, spv::StorageClass::Private);
    if (profile.supported_spirv >= 0x00010400) {
        interfaces.push_back(local_memory);
    }
}

void EmitContext::DefineSharedMemory(const IR::Program& program) {
    uses_explicit_workgroup_layout =
        profile.support_explicit_workgroup_layout &&
        (!program.info.uses_int8 || profile.support_workgroup_layout_8bit_access) &&
        (!program.info.uses_int16 || profile.support_workgroup_layout_16bit_access);
    if (program.shared_memory_size == 0) {
        return;
    }
    const auto make{[&](Id element_type, u32 element_size) {
        const u32 num_elements{Common::DivCeil(program.shared_memory_size, element_size)};
        const Id array_type{TypeArray(element_type, Const(num_elements))};
        Decorate(array_type, spv::Decoration::ArrayStride, element_size);

        const Id struct_type{TypeStruct(array_type)};
        MemberDecorate(struct_type, 0U, spv::Decoration::Offset, 0U);
        Decorate(struct_type, spv::Decoration::Block);

        const Id pointer{TypePointer(spv::StorageClass::Workgroup, struct_type)};
        const Id element_pointer{TypePointer(spv::StorageClass::Workgroup, element_type)};
        const Id variable{AddGlobalVariable(pointer, spv::StorageClass::Workgroup)};
        Decorate(variable, spv::Decoration::Aliased);
        interfaces.push_back(variable);

        return std::make_tuple(variable, element_pointer, pointer);
    }};
    if (uses_explicit_workgroup_layout) {
        AddExtension("SPV_KHR_workgroup_memory_explicit_layout");
        AddCapability(spv::Capability::WorkgroupMemoryExplicitLayoutKHR);
        if (program.info.uses_int8 && profile.support_int8) {
            AddCapability(spv::Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR);
            std::tie(shared_memory_u8, shared_u8, std::ignore) = make(U8, 1);
        }
        if (program.info.uses_int16 && profile.support_int16) {
            AddCapability(spv::Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
            std::tie(shared_memory_u16, shared_u16, std::ignore) = make(U16, 2);
        }
        if (program.info.uses_int64 && profile.support_int64) {
            std::tie(shared_memory_u64, shared_u64, std::ignore) = make(U64, 8);
        }
        std::tie(shared_memory_u32, shared_u32, shared_memory_u32_type) = make(U32[1], 4);
        std::tie(shared_memory_u32x2, shared_u32x2, std::ignore) = make(U32[2], 8);
        std::tie(shared_memory_u32x4, shared_u32x4, std::ignore) = make(U32[4], 16);
        return;
    }
    const u32 num_elements{Common::DivCeil(program.shared_memory_size, 4U)};
    const Id type{TypeArray(U32[1], Const(num_elements))};
    shared_memory_u32_type = TypePointer(spv::StorageClass::Workgroup, type);

    shared_u32 = TypePointer(spv::StorageClass::Workgroup, U32[1]);
    shared_memory_u32 = AddGlobalVariable(shared_memory_u32_type, spv::StorageClass::Workgroup);
    interfaces.push_back(shared_memory_u32);

    const Id func_type{TypeFunction(void_id, U32[1], U32[1])};
    const auto make_function{[&](u32 mask, u32 size) {
        const Id loop_header{OpLabel()};
        const Id continue_block{OpLabel()};
        const Id merge_block{OpLabel()};

        const Id func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type)};
        const Id offset{OpFunctionParameter(U32[1])};
        const Id insert_value{OpFunctionParameter(U32[1])};
        AddLabel();
        const Id expected_var{AddLocalVariable(TypePointer(spv::StorageClass::Function, U32[1]),
                                               spv::StorageClass::Function)};
        const Id word_offset{OpShiftRightArithmetic(U32[1], offset, Const(2U))};
        const Id shift_offset{OpShiftLeftLogical(U32[1], offset, Const(3U))};
        const Id bit_offset{OpBitwiseAnd(U32[1], shift_offset, Const(mask))};
        const Id count{Const(size)};
        const Id word_pointer{OpAccessChain(shared_u32, shared_memory_u32, word_offset)};
        OpStore(expected_var, OpLoad(U32[1], word_pointer));
        OpBranch(loop_header);

        AddLabel(loop_header);
        OpLoopMerge(merge_block, continue_block, spv::LoopControlMask::MaskNone);
        OpBranch(continue_block);

        AddLabel(continue_block);
        const Id old_value{OpLoad(U32[1], expected_var)};
        const Id new_value{OpBitFieldInsert(U32[1], old_value, insert_value, bit_offset, count)};
        const Id atomic_res{OpAtomicCompareExchange(U32[1], word_pointer, Const(1U), u32_zero_value,
                                                    u32_zero_value, new_value, old_value)};
        OpStore(expected_var, atomic_res);
        const Id success{OpIEqual(U1, atomic_res, old_value)};
        OpBranchConditional(success, merge_block, loop_header);

        AddLabel(merge_block);
        OpReturn();
        OpFunctionEnd();
        return func;
    }};
    if (program.info.uses_int8) {
        shared_store_u8_func = make_function(24, 8);
    }
    if (program.info.uses_int16) {
        shared_store_u16_func = make_function(16, 16);
    }
}

void EmitContext::DefineSharedMemoryFunctions(const IR::Program& program) {
    if (program.info.uses_shared_increment) {
        increment_cas_shared = CasLoop(*this, Operation::Increment, shared_memory_u32_type,
                                       shared_u32, U32[1], spv::Scope::Workgroup);
    }
    if (program.info.uses_shared_decrement) {
        decrement_cas_shared = CasLoop(*this, Operation::Decrement, shared_memory_u32_type,
                                       shared_u32, U32[1], spv::Scope::Workgroup);
    }
}

void EmitContext::DefineAttributeMemAccess(const Info& info) {
    const auto make_load{[&] {
        const bool is_array{stage == Stage::Geometry};
        const Id end_block{OpLabel()};
        const Id default_label{OpLabel()};

        const Id func_type_load{is_array ? TypeFunction(F32[1], U32[1], U32[1])
                                         : TypeFunction(F32[1], U32[1])};
        const Id func{OpFunction(F32[1], spv::FunctionControlMask::MaskNone, func_type_load)};
        const Id offset{OpFunctionParameter(U32[1])};
        const Id vertex{is_array ? OpFunctionParameter(U32[1]) : Id{}};

        AddLabel();
        const Id base_index{OpShiftRightArithmetic(U32[1], offset, Const(2U))};
        const Id masked_index{OpBitwiseAnd(U32[1], base_index, Const(3U))};
        const Id compare_index{OpShiftRightArithmetic(U32[1], base_index, Const(2U))};
        std::vector<Sirit::Literal> literals;
        std::vector<Id> labels;
        if (info.loads.AnyComponent(IR::Attribute::PositionX)) {
            literals.push_back(static_cast<u32>(IR::Attribute::PositionX) >> 2);
            labels.push_back(OpLabel());
        }
        const u32 base_attribute_value = static_cast<u32>(IR::Attribute::Generic0X) >> 2;
        for (u32 index = 0; index < static_cast<u32>(IR::NUM_GENERICS); ++index) {
            if (!info.loads.Generic(index)) {
                continue;
            }
            literals.push_back(base_attribute_value + index);
            labels.push_back(OpLabel());
        }
        OpSelectionMerge(end_block, spv::SelectionControlMask::MaskNone);
        OpSwitch(compare_index, default_label, literals, labels);
        AddLabel(default_label);
        OpReturnValue(Const(0.0f));
        size_t label_index{0};
        if (info.loads.AnyComponent(IR::Attribute::PositionX)) {
            AddLabel(labels[label_index]);
            const Id pointer{[&]() {
                if (need_input_position_indirect) {
                    if (is_array)
                        return OpAccessChain(input_f32, input_position, vertex, u32_zero_value,
                                             masked_index);
                    else
                        return OpAccessChain(input_f32, input_position, u32_zero_value,
                                             masked_index);
                } else {
                    if (is_array)
                        return OpAccessChain(input_f32, input_position, vertex, masked_index);
                    else
                        return OpAccessChain(input_f32, input_position, masked_index);
                }
            }()};
            const Id result{OpLoad(F32[1], pointer)};
            OpReturnValue(result);
            ++label_index;
        }
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (!info.loads.Generic(index)) {
                continue;
            }
            AddLabel(labels[label_index]);
            const auto& generic{input_generics.at(index)};
            const Id generic_id{generic.id};
            if (!ValidId(generic_id)) {
                OpReturnValue(Const(0.0f));
                ++label_index;
                continue;
            }
            const Id pointer{
                is_array ? OpAccessChain(generic.pointer_type, generic_id, vertex, masked_index)
                         : OpAccessChain(generic.pointer_type, generic_id, masked_index)};
            const Id value{OpLoad(generic.component_type, pointer)};
            const Id result{[this, generic, value]() {
                switch (generic.load_op) {
                case InputGenericLoadOp::Bitcast:
                    return OpBitcast(F32[1], value);
                case InputGenericLoadOp::SToF:
                    return OpConvertSToF(F32[1], value);
                case InputGenericLoadOp::UToF:
                    return OpConvertUToF(F32[1], value);
                default:
                    return value;
                };
            }()};
            OpReturnValue(result);
            ++label_index;
        }
        AddLabel(end_block);
        OpUnreachable();
        OpFunctionEnd();
        return func;
    }};
    const auto make_store{[&] {
        const Id end_block{OpLabel()};
        const Id default_label{OpLabel()};

        const Id func_type_store{TypeFunction(void_id, U32[1], F32[1])};
        const Id func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type_store)};
        const Id offset{OpFunctionParameter(U32[1])};
        const Id store_value{OpFunctionParameter(F32[1])};
        AddLabel();
        const Id base_index{OpShiftRightArithmetic(U32[1], offset, Const(2U))};
        const Id masked_index{OpBitwiseAnd(U32[1], base_index, Const(3U))};
        const Id compare_index{OpShiftRightArithmetic(U32[1], base_index, Const(2U))};
        std::vector<Sirit::Literal> literals;
        std::vector<Id> labels;
        if (info.stores.AnyComponent(IR::Attribute::PositionX)) {
            literals.push_back(static_cast<u32>(IR::Attribute::PositionX) >> 2);
            labels.push_back(OpLabel());
        }
        const u32 base_attribute_value = static_cast<u32>(IR::Attribute::Generic0X) >> 2;
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (!info.stores.Generic(index)) {
                continue;
            }
            literals.push_back(base_attribute_value + static_cast<u32>(index));
            labels.push_back(OpLabel());
        }
        if (info.stores.ClipDistances()) {
            if (profile.max_user_clip_distances >= 4) {
                literals.push_back(static_cast<u32>(IR::Attribute::ClipDistance0) >> 2);
                labels.push_back(OpLabel());
            }
            if (profile.max_user_clip_distances >= 8) {
                literals.push_back(static_cast<u32>(IR::Attribute::ClipDistance4) >> 2);
                labels.push_back(OpLabel());
            }
        }
        OpSelectionMerge(end_block, spv::SelectionControlMask::MaskNone);
        OpSwitch(compare_index, default_label, literals, labels);
        AddLabel(default_label);
        OpReturn();
        size_t label_index{0};
        if (info.stores.AnyComponent(IR::Attribute::PositionX)) {
            AddLabel(labels[label_index]);
            const Id pointer{OpAccessChain(output_f32, output_position, masked_index)};
            OpStore(pointer, store_value);
            OpReturn();
            ++label_index;
        }
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (!info.stores.Generic(index)) {
                continue;
            }
            if (output_generics[index][0].num_components != 4) {
                throw NotImplementedException("Physical stores and transform feedbacks");
            }
            AddLabel(labels[label_index]);
            const Id generic_id{output_generics[index][0].id};
            const Id pointer{OpAccessChain(output_f32, generic_id, masked_index)};
            OpStore(pointer, store_value);
            OpReturn();
            ++label_index;
        }
        if (info.stores.ClipDistances()) {
            if (profile.max_user_clip_distances >= 4) {
                AddLabel(labels[label_index]);
                const Id pointer{OpAccessChain(output_f32, clip_distances, masked_index)};
                OpStore(pointer, store_value);
                OpReturn();
                ++label_index;
            }
            if (profile.max_user_clip_distances >= 8) {
                AddLabel(labels[label_index]);
                const Id fixed_index{OpIAdd(U32[1], masked_index, Const(4U))};
                const Id pointer{OpAccessChain(output_f32, clip_distances, fixed_index)};
                OpStore(pointer, store_value);
                OpReturn();
                ++label_index;
            }
        }
        AddLabel(end_block);
        OpUnreachable();
        OpFunctionEnd();
        return func;
    }};
    if (info.loads_indexed_attributes) {
        indexed_load_func = make_load();
    }
    if (info.stores_indexed_attributes) {
        indexed_store_func = make_store();
    }
}

void EmitContext::DefineWriteStorageCasLoopFunction(const Info& info) {
    // Must match the conditions in EmitWriteStorage{U,S}{8,16}: the CAS loop is also used when
    // storage buffer 8/16-bit access or descriptor aliasing is unavailable (e.g. Qualcomm).
    const bool direct_int8{profile.support_int8 && profile.support_storage_buffer_8bit &&
                           profile.support_descriptor_aliasing};
    const bool direct_int16{profile.support_int16 && profile.support_storage_buffer_16bit &&
                            profile.support_descriptor_aliasing};
    if ((direct_int8 || !info.uses_int8) && (direct_int16 || !info.uses_int16)) {
        return;
    }

    AddCapability(spv::Capability::VariablePointersStorageBuffer);

    const Id ptr_type{TypePointer(spv::StorageClass::StorageBuffer, U32[1])};
    const Id func_type{TypeFunction(void_id, ptr_type, U32[1], U32[1], U32[1])};
    const Id func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type)};
    const Id pointer{OpFunctionParameter(ptr_type)};
    const Id value{OpFunctionParameter(U32[1])};
    const Id bit_offset{OpFunctionParameter(U32[1])};
    const Id bit_count{OpFunctionParameter(U32[1])};

    AddLabel();
    // Each attempt expects the word the previous compare-exchange returned (see CasLoop).
    const Id expected_var{
        AddLocalVariable(TypePointer(spv::StorageClass::Function, U32[1]), spv::StorageClass::Function)};
    OpStore(expected_var, OpLoad(U32[1], pointer));
    const Id scope_device{Const(1u)};
    const Id ordering_relaxed{u32_zero_value};
    const Id body_label{OpLabel()};
    const Id continue_label{OpLabel()};
    const Id endloop_label{OpLabel()};
    const Id beginloop_label{OpLabel()};
    OpBranch(beginloop_label);

    AddLabel(beginloop_label);
    OpLoopMerge(endloop_label, continue_label, spv::LoopControlMask::MaskNone);
    OpBranch(body_label);

    AddLabel(body_label);
    const Id expected_value{OpLoad(U32[1], expected_var)};
    const Id desired_value{OpBitFieldInsert(U32[1], expected_value, value, bit_offset, bit_count)};
    const Id actual_value{OpAtomicCompareExchange(U32[1], pointer, scope_device, ordering_relaxed,
                                                  ordering_relaxed, desired_value, expected_value)};
    OpStore(expected_var, actual_value);
    const Id store_successful{OpIEqual(U1, expected_value, actual_value)};
    OpBranchConditional(store_successful, endloop_label, continue_label);

    AddLabel(endloop_label);
    OpReturn();

    AddLabel(continue_label);
    OpBranch(beginloop_label);

    OpFunctionEnd();

    write_storage_cas_loop_func = func;
}

void EmitContext::DefineRawGlobalMemoryFunctions(const Info& info) {
    raw_global_memory = true;
    // VTX-HLE's raw path uses a table search instead of the candidate-buffer switch. Keep its
    // translation and access helpers as calls too, so each access does not duplicate the search.
    const spv::FunctionControlMask helper_control{LeanGlobalMemory(stage)
                                                      ? spv::FunctionControlMask::DontInline
                                                      : spv::FunctionControlMask::MaskNone};
    const Id zero{u32_zero_value};
    const Id zero64{Constant(U64, u64{0})};
    const Id one{Const(1u)};

    // translate(guest address) -> device address, 0 when no table entry contains the address.
    // The table is an array of {guest base, device base, size} (u64 each), sorted by guest base
    // and never overlapping, so the entry for an address is found by a binary search (the last
    // entry whose base is <= the address); any number of ranges fit. With
    // Info::raw_global_table_ssbo it is read from the last storage buffer of the program (u32
    // words: the entry count at word 0, the entries from RAW_GLOBAL_TABLE_SSBO_ENTRIES_WORD).
    // Otherwise the driver bank (c0) holds a u32 entry count at RAW_GLOBAL_TABLE_OFFSET and the
    // device address of the table at RAW_GLOBAL_TABLE_ADDRESS_OFFSET.
    const bool table_in_ssbo{info.raw_global_table_ssbo};
    const Id translate_type{TypeFunction(U64, U64)};
    const Id translate{OpFunction(U64, helper_control, translate_type)};
    const Id address{OpFunctionParameter(U64)};
    AddLabel();
    const Id index_pointer_type{TypePointer(spv::StorageClass::Function, U32[1])};
    const Id low_var{AddLocalVariable(index_pointer_type, spv::StorageClass::Function, zero)};
    const Id high_var{AddLocalVariable(index_pointer_type, spv::StorageClass::Function, zero)};
    // Info::raw_global_last_hit (EDEN_NX2_TRY_RAWGLOBAL_FAST): the invocation's last candidate
    // entry {guest base, device base, size} in Private variables (size 0: none yet). An address
    // inside it maps without the search. Same result as the search: the entries never overlap,
    // so the entry containing an address is the last one whose base is <= it, and the table does
    // not change while the program runs (reference: VideoCommon::TranslateRawGlobalAddressLastHit).
    std::array<Id, 3> last_hit{};
    if (info.raw_global_last_hit) {
        const Id private_u64{TypePointer(spv::StorageClass::Private, U64)};
        for (Id& variable : last_hit) {
            variable = AddGlobalVariable(private_u64, spv::StorageClass::Private, zero64);
            if (profile.supported_spirv >= 0x00010400) {
                interfaces.push_back(variable);
            }
        }
        const Id relative{OpISub(U64, address, OpLoad(U64, last_hit[0]))};
        const Id hit{OpULessThan(U1, relative, OpLoad(U64, last_hit[2]))};
        const Id hit_label{OpLabel()};
        const Id search_label{OpLabel()};
        OpSelectionMerge(search_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(hit, hit_label, search_label);
        AddLabel(hit_label);
        OpReturnValue(OpIAdd(U64, OpLoad(U64, last_hit[1]), relative));
        AddLabel(search_label);
    }
    Id count{};
    Id table{};
    Id table_ssbo{};
    if (table_in_ssbo) {
        table_ssbo = ssbos[info.raw_global_table_index].U32;
        const Id count_pointer{OpAccessChain(storage_types.U32.element, table_ssbo, zero, zero)};
        count = OpLoad(U32[1], count_pointer);
    } else {
        const Id count_pointer{OpAccessChain(uniform_types.U32, cbufs[0].U32, zero,
                                             Const(Profile::RAW_GLOBAL_TABLE_OFFSET / 4))};
        count = OpLoad(U32[1], count_pointer);
        const Id table_pointer{OpAccessChain(uniform_types.U32x2, cbufs[0].U32x2, zero,
                                             Const(Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET / 8))};
        table = OpBitcast(U64, OpLoad(U32[2], table_pointer));
    }
    OpStore(high_var, count);
    const Id entry_pointer_type{TypePointer(spv::StorageClass::PhysicalStorageBuffer, U64)};
    constexpr u32 ENTRY_ALIGNMENT{8};
    // The u64 at byte `offset` of entry `index`.
    const auto load_entry{[&](Id index, u32 offset) -> Id {
        if (table_in_ssbo) {
            constexpr u32 ENTRY_WORDS{Profile::RAW_GLOBAL_TABLE_ENTRY_SIZE / 4};
            const Id word{OpIAdd(U32[1], OpIMul(U32[1], index, Const(ENTRY_WORDS)),
                                 Const(Profile::RAW_GLOBAL_TABLE_SSBO_ENTRIES_WORD + offset / 4))};
            const Id low_pointer{OpAccessChain(storage_types.U32.element, table_ssbo, zero, word)};
            const Id high_pointer{OpAccessChain(storage_types.U32.element, table_ssbo, zero,
                                                OpIAdd(U32[1], word, one))};
            const Id words{OpCompositeConstruct(U32[2], OpLoad(U32[1], low_pointer),
                                                OpLoad(U32[1], high_pointer))};
            return OpBitcast(U64, words);
        }
        const Id byte_offset{OpIMul(U64, OpUConvert(U64, index),
                                    Constant(U64, u64{Profile::RAW_GLOBAL_TABLE_ENTRY_SIZE}))};
        const Id entry_address{OpIAdd(U64, OpIAdd(U64, table, byte_offset),
                                      Constant(U64, u64{offset}))};
        const Id pointer{OpConvertUToPtr(entry_pointer_type, entry_address)};
        return OpLoad(U64, pointer, spv::MemoryAccessMask::Aligned, ENTRY_ALIGNMENT);
    }};
    const Id header_label{OpLabel()};
    const Id check_label{OpLabel()};
    const Id body_label{OpLabel()};
    const Id continue_label{OpLabel()};
    const Id merge_label{OpLabel()};
    OpBranch(header_label);

    AddLabel(header_label);
    OpLoopMerge(merge_label, continue_label, spv::LoopControlMask::MaskNone);
    OpBranch(check_label);

    AddLabel(check_label);
    const Id low{OpLoad(U32[1], low_var)};
    const Id high{OpLoad(U32[1], high_var)};
    OpBranchConditional(OpULessThan(U1, low, high), body_label, merge_label);

    AddLabel(body_label);
    const Id middle{OpShiftRightLogical(U32[1], OpIAdd(U32[1], low, high), one)};
    const Id middle_base{load_entry(middle, 0)};
    const Id base_not_above{OpULessThanEqual(U1, middle_base, address)};
    OpStore(low_var, OpSelect(U32[1], base_not_above, OpIAdd(U32[1], middle, one), low));
    OpStore(high_var, OpSelect(U32[1], base_not_above, high, middle));
    OpBranch(continue_label);

    AddLabel(continue_label);
    OpBranch(header_label);

    AddLabel(merge_label);
    // `found` entries have a base <= address; the last one is the candidate.
    const Id found{OpLoad(U32[1], low_var)};
    const Id has_candidate{OpINotEqual(U1, found, zero)};
    const Id candidate_label{OpLabel()};
    const Id none_label{OpLabel()};
    OpSelectionMerge(none_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(has_candidate, candidate_label, none_label);

    AddLabel(candidate_label);
    const Id last{OpISub(U32[1], found, one)};
    const Id guest_base{load_entry(last, 0)};
    const Id device_base{load_entry(last, 8)};
    const Id range_size{load_entry(last, 16)};
    if (info.raw_global_last_hit) {
        // Any entry of the table is a valid cache, also one that does not contain the address.
        OpStore(last_hit[0], guest_base);
        OpStore(last_hit[1], device_base);
        OpStore(last_hit[2], range_size);
    }
    const Id relative{OpISub(U64, address, guest_base)};
    const Id in_range{OpULessThan(U1, relative, range_size)};
    OpReturnValue(OpSelect(U64, in_range, OpIAdd(U64, device_base, relative), zero64));

    AddLabel(none_label);
    OpReturnValue(zero64);
    OpFunctionEnd();

    // Info::raw_global_store_alias: alias(guest address) -> the device address of the same byte in
    // an eligible mirror alias target, or 0 when none holds it. The driver bank carries the table
    // (Profile::RAW_GLOBAL_ALIAS_*); entries share the raw table's layout and search. A raw store
    // writes there after the raw-table target, so a read-only mirror stays current or a storage
    // mirror's imported guest backing receives the write too.
    Id alias{};
    if (info.raw_global_store_alias) {
        alias = OpFunction(U64, helper_control, translate_type);
        const Id alias_address{OpFunctionParameter(U64)};
        AddLabel();
        const Id alias_low_var{
            AddLocalVariable(index_pointer_type, spv::StorageClass::Function, zero)};
        const Id alias_high_var{
            AddLocalVariable(index_pointer_type, spv::StorageClass::Function, zero)};
        const auto bank_u64{[&](u32 offset) {
            const Id pointer{
                OpAccessChain(uniform_types.U32x2, cbufs[0].U32x2, zero, Const(offset / 8))};
            return OpBitcast(U64, OpLoad(U32[2], pointer));
        }};
        const Id alias_count{OpLoad(
            U32[1], OpAccessChain(uniform_types.U32, cbufs[0].U32, zero,
                                  Const(Profile::RAW_GLOBAL_ALIAS_COUNT_OFFSET / 4)))};
        const Id alias_table{bank_u64(Profile::RAW_GLOBAL_ALIAS_ADDRESS_OFFSET)};
        const Id bounds_low{bank_u64(Profile::RAW_GLOBAL_ALIAS_LOW_OFFSET)};
        const Id bounds_high{bank_u64(Profile::RAW_GLOBAL_ALIAS_HIGH_OFFSET)};
        // Most stores land outside every mirror: one compare, no search.
        const Id inside{OpULessThan(U1, OpISub(U64, alias_address, bounds_low),
                                    OpISub(U64, bounds_high, bounds_low))};
        const Id outside_label{OpLabel()};
        const Id alias_search_label{OpLabel()};
        OpSelectionMerge(alias_search_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(inside, alias_search_label, outside_label);
        AddLabel(outside_label);
        OpReturnValue(zero64);
        AddLabel(alias_search_label);
        OpStore(alias_high_var, alias_count);
        const auto alias_entry{[&](Id index, u32 offset) -> Id {
            const Id byte_offset{OpIMul(U64, OpUConvert(U64, index),
                                        Constant(U64, u64{Profile::RAW_GLOBAL_TABLE_ENTRY_SIZE}))};
            const Id entry_address{OpIAdd(U64, OpIAdd(U64, alias_table, byte_offset),
                                          Constant(U64, u64{offset}))};
            return OpLoad(U64, OpConvertUToPtr(entry_pointer_type, entry_address),
                          spv::MemoryAccessMask::Aligned, ENTRY_ALIGNMENT);
        }};
        const Id alias_header_label{OpLabel()};
        const Id alias_check_label{OpLabel()};
        const Id alias_body_label{OpLabel()};
        const Id alias_continue_label{OpLabel()};
        const Id alias_merge_label{OpLabel()};
        OpBranch(alias_header_label);

        AddLabel(alias_header_label);
        OpLoopMerge(alias_merge_label, alias_continue_label, spv::LoopControlMask::MaskNone);
        OpBranch(alias_check_label);

        AddLabel(alias_check_label);
        const Id alias_low{OpLoad(U32[1], alias_low_var)};
        const Id alias_high{OpLoad(U32[1], alias_high_var)};
        OpBranchConditional(OpULessThan(U1, alias_low, alias_high), alias_body_label,
                            alias_merge_label);

        AddLabel(alias_body_label);
        const Id alias_middle{
            OpShiftRightLogical(U32[1], OpIAdd(U32[1], alias_low, alias_high), one)};
        const Id alias_not_above{
            OpULessThanEqual(U1, alias_entry(alias_middle, 0), alias_address)};
        OpStore(alias_low_var,
                OpSelect(U32[1], alias_not_above, OpIAdd(U32[1], alias_middle, one), alias_low));
        OpStore(alias_high_var, OpSelect(U32[1], alias_not_above, alias_high, alias_middle));
        OpBranch(alias_continue_label);

        AddLabel(alias_continue_label);
        OpBranch(alias_header_label);

        AddLabel(alias_merge_label);
        const Id alias_found{OpLoad(U32[1], alias_low_var)};
        const Id alias_candidate_label{OpLabel()};
        const Id alias_none_label{OpLabel()};
        // The condition first: OpSelectionMerge must directly precede the branch.
        const Id alias_has_candidate{OpINotEqual(U1, alias_found, zero)};
        OpSelectionMerge(alias_none_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(alias_has_candidate, alias_candidate_label, alias_none_label);

        AddLabel(alias_candidate_label);
        const Id alias_last{OpISub(U32[1], alias_found, one)};
        const Id alias_relative{OpISub(U64, alias_address, alias_entry(alias_last, 0))};
        const Id alias_in_range{OpULessThan(U1, alias_relative, alias_entry(alias_last, 16))};
        OpReturnValue(OpSelect(U64, alias_in_range,
                               OpIAdd(U64, alias_entry(alias_last, 8), alias_relative), zero64));

        AddLabel(alias_none_label);
        OpReturnValue(zero64);
        OpFunctionEnd();
    }
    // Inside a store's mapped branch: write to the alias target when one covers `guest_address`.
    const auto write_through{[&](Id guest_address, const auto& store) {
        if (!Sirit::ValidId(alias)) {
            return;
        }
        const Id mirror{OpFunctionCall(U64, alias, guest_address)};
        const Id mirror_label{OpLabel()};
        const Id mirror_end_label{OpLabel()};
        const Id has_mirror{OpINotEqual(U1, mirror, zero64)};
        OpSelectionMerge(mirror_end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(has_mirror, mirror_label, mirror_end_label);
        AddLabel(mirror_label);
        store(mirror);
        OpBranch(mirror_end_label);
        AddLabel(mirror_end_label);
    }};

    // Raw coverage diagnostic (Info::raw_coverage, raw_coverage_layout.h): report(guest address,
    // op) appends {address, program hash, op | stage << 8, present} to the miss ring whose device
    // address is in the table header (the table storage buffer's words, or c0 next to the table
    // address; nothing when it is 0). Once the ring is full a miss only reads the counter: an
    // atomic per miss on one word (hundreds of millions per window in a pointer-chasing loop)
    // would stall the GPU.
    Id report_miss{};
    if (info.raw_coverage) {
        namespace Cov = RawCoverage;
        static_assert(Cov::BANK_RING_OFFSET == Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET + 8 &&
                      Cov::BANK_RING_OFFSET % 8 == 0);
        report_miss = OpFunction(void_id, helper_control, TypeFunction(void_id, U64, U32[1]));
        const Id miss_address{OpFunctionParameter(U64)};
        const Id miss_op{OpFunctionParameter(U32[1])};
        AddLabel();
        // 1 while this miss may take a record (a null miss only below NULL_RECORDS).
        const Id take_var{AddLocalVariable(index_pointer_type, spv::StorageClass::Function, one)};
        Id ring{};
        if (table_in_ssbo) {
            const auto header_word{[&](u32 word) {
                return OpLoad(U32[1], OpAccessChain(storage_types.U32.element, table_ssbo, zero,
                                                    Const(word)));
            }};
            ring = OpBitcast(U64, OpCompositeConstruct(U32[2], header_word(Cov::TABLE_RING_WORD),
                                                       header_word(Cov::TABLE_RING_WORD + 1)));
        } else {
            const Id ring_pointer{OpAccessChain(uniform_types.U32x2, cbufs[0].U32x2, zero,
                                                Const(Cov::BANK_RING_OFFSET / 8))};
            ring = OpBitcast(U64, OpLoad(U32[2], ring_pointer));
        }
        const Id ring_label{OpLabel()};
        const Id null_label{OpLabel()};
        const Id null_add_label{OpLabel()};
        const Id null_add_end_label{OpLabel()};
        const Id null_end_label{OpLabel()};
        const Id add_label{OpLabel()};
        const Id slot_label{OpLabel()};
        const Id slot_end_label{OpLabel()};
        const Id add_end_label{OpLabel()};
        const Id report_end_label{OpLabel()};
        const Id has_ring{OpINotEqual(U1, ring, zero64)};
        OpSelectionMerge(report_end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(has_ring, ring_label, report_end_label);

        AddLabel(ring_label);
        const Id word_pointer_type{TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[1])};
        const Id counter{OpConvertUToPtr(
            word_pointer_type,
            OpIAdd(U64, ring, Constant(U64, u64{Cov::RING_COUNTER_WORD * sizeof(u32)})))};
        const Id device_scope{Const(static_cast<u32>(spv::Scope::Device))};
        // Null misses: a load of their own counter, an add only while below the cap.
        const Id is_null{
            OpULessThan(U1, miss_address, Constant(U64, u64{Cov::NULL_ADDRESS_LIMIT}))};
        OpSelectionMerge(null_end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(is_null, null_label, null_end_label);

        AddLabel(null_label);
        const Id null_counter{OpConvertUToPtr(
            word_pointer_type,
            OpIAdd(U64, ring, Constant(U64, u64{Cov::RING_NULL_WORD * sizeof(u32)})))};
        OpStore(take_var, zero);
        const Id null_seen{OpAtomicLoad(U32[1], null_counter, device_scope, zero)};
        const Id null_room{OpULessThan(U1, null_seen, Const(Cov::NULL_RECORDS))};
        OpSelectionMerge(null_add_end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(null_room, null_add_label, null_add_end_label);

        AddLabel(null_add_label);
        const Id null_slot{OpAtomicIAdd(U32[1], null_counter, device_scope, zero, one)};
        OpStore(take_var,
                OpSelect(U32[1], OpULessThan(U1, null_slot, Const(Cov::NULL_RECORDS)), one, zero));
        OpBranch(null_add_end_label);

        AddLabel(null_add_end_label);
        OpBranch(null_end_label);

        AddLabel(null_end_label);
        const Id take{OpINotEqual(U1, OpLoad(U32[1], take_var), zero)};
        const Id seen{OpAtomicLoad(U32[1], counter, device_scope, zero)};
        const Id room{OpLogicalAnd(U1, take, OpULessThan(U1, seen, Const(Cov::RING_ENTRIES)))};
        OpSelectionMerge(add_end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(room, add_label, add_end_label);

        AddLabel(add_label);
        const Id slot{OpAtomicIAdd(U32[1], counter, device_scope, zero, one)};
        const Id in_ring{OpULessThan(U1, slot, Const(Cov::RING_ENTRIES))};
        OpSelectionMerge(slot_end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(in_ring, slot_label, slot_end_label);

        AddLabel(slot_label);
        const Id present_pointer{OpConvertUToPtr(
            word_pointer_type,
            OpIAdd(U64, ring, Constant(U64, u64{Cov::RING_PRESENT_WORD * sizeof(u32)})))};
        const Id present{OpLoad(U32[1], present_pointer, spv::MemoryAccessMask::Aligned, 4u)};
        const Id record{OpIAdd(
            U64, ring,
            OpIAdd(U64, Constant(U64, u64{Cov::RING_HEADER_BYTES}),
                   OpIMul(U64, OpUConvert(U64, slot), Constant(U64, u64{Cov::RECORD_BYTES}))))};
        const Id address_words{OpBitcast(U32[2], miss_address)};
        const u64 program_hash{info.raw_coverage_program};
        const Id first_half{OpCompositeConstruct(
            U32[4], OpCompositeExtract(U32[1], address_words, 0u),
            OpCompositeExtract(U32[1], address_words, 1u), Const(static_cast<u32>(program_hash)),
            Const(static_cast<u32>(program_hash >> 32)))};
        const Id kind{OpBitwiseOr(
            U32[1], miss_op, Const(static_cast<u32>(stage) << Cov::RECORD_STAGE_SHIFT))};
        const Id second_half{OpCompositeConstruct(U32[4], kind, present, zero, zero)};
        static_assert(Cov::RECORD_KIND_WORD == 4 && Cov::RECORD_PRESENT_WORD == 5);
        static_assert(Cov::RING_HEADER_BYTES % 16 == 0 && Cov::RECORD_BYTES == 32);
        const Id half_pointer_type{TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[4])};
        OpStore(OpConvertUToPtr(half_pointer_type, record), first_half,
                spv::MemoryAccessMask::Aligned, 16u);
        OpStore(OpConvertUToPtr(half_pointer_type, OpIAdd(U64, record, Constant(U64, u64{16}))),
                second_half, spv::MemoryAccessMask::Aligned, 16u);
        OpBranch(slot_end_label);

        AddLabel(slot_end_label);
        OpBranch(add_end_label);

        AddLabel(add_end_label);
        OpBranch(report_end_label);

        AddLabel(report_end_label);
        OpReturn();
        OpFunctionEnd();
    }
    const auto report{[&](Id addr, RawCoverage::Op op) {
        if (Sirit::ValidId(report_miss)) {
            OpFunctionCall(void_id, report_miss, addr, Const(static_cast<u32>(op)));
        }
    }};

    // Accesses: translate, then go through a PhysicalStorageBuffer64 pointer. Accesses outside
    // every entry are dropped (loads return 0), as with robust buffer access.
    constexpr u32 ALIGNMENT{4};
    const auto define_load{[&](Id type, RawCoverage::Op op) {
        const Id pointer_type{TypePointer(spv::StorageClass::PhysicalStorageBuffer, type)};
        const Id func{OpFunction(type, helper_control, TypeFunction(type, U64))};
        const Id addr{OpFunctionParameter(U64)};
        AddLabel();
        const Id device{OpFunctionCall(U64, translate, addr)};
        const Id then_label{OpLabel()};
        const Id end_label{OpLabel()};
        const Id mapped{OpINotEqual(U1, device, zero64)};
        OpSelectionMerge(end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(mapped, then_label, end_label);
        AddLabel(then_label);
        const Id pointer{OpConvertUToPtr(pointer_type, device)};
        OpReturnValue(OpLoad(type, pointer, spv::MemoryAccessMask::Aligned, ALIGNMENT));
        AddLabel(end_label);
        report(addr, op);
        OpReturnValue(ConstantNull(type));
        OpFunctionEnd();
        return func;
    }};
    const auto define_write{[&](Id type, RawCoverage::Op op) {
        const Id pointer_type{TypePointer(spv::StorageClass::PhysicalStorageBuffer, type)};
        const Id func{OpFunction(void_id, helper_control, TypeFunction(void_id, U64, type))};
        const Id addr{OpFunctionParameter(U64)};
        const Id data{OpFunctionParameter(type)};
        AddLabel();
        const Id device{OpFunctionCall(U64, translate, addr)};
        const Id then_label{OpLabel()};
        const Id end_label{OpLabel()};
        const Id mapped{OpINotEqual(U1, device, zero64)};
        OpSelectionMerge(end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(mapped, then_label, end_label);
        AddLabel(then_label);
        const Id pointer{OpConvertUToPtr(pointer_type, device)};
        OpStore(pointer, data, spv::MemoryAccessMask::Aligned, ALIGNMENT);
        write_through(addr, [&](Id mirror) {
            OpStore(OpConvertUToPtr(pointer_type, mirror), data, spv::MemoryAccessMask::Aligned,
                    ALIGNMENT);
        });
        OpReturn();
        AddLabel(end_label);
        report(addr, op);
        OpReturn();
        OpFunctionEnd();
        return func;
    }};
    load_global_func_u32 = define_load(U32[1], RawCoverage::Op::Load32);
    load_global_func_u32x2 = define_load(U32[2], RawCoverage::Op::Load64);
    load_global_func_u32x4 = define_load(U32[4], RawCoverage::Op::Load128);
    write_global_func_u32 = define_write(U32[1], RawCoverage::Op::Store32);
    write_global_func_u32x2 = define_write(U32[2], RawCoverage::Op::Store64);
    write_global_func_u32x4 = define_write(U32[4], RawCoverage::Op::Store128);

    if (!info.stores_raw_global_memory) {
        return;
    }
    const Id u32_pointer_type{TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[1])};
    for (size_t index = 0; index < NUM_GLOBAL_ATOMICS; ++index) {
        const auto opcode{
            static_cast<IR::Opcode>(static_cast<size_t>(IR::Opcode::GlobalAtomicIAdd32) + index)};
        if ((info.global_atomics & GlobalAtomicBit(opcode)) == 0) {
            continue;
        }
        const std::optional<GlobalAtomicSpec> spec{GlobalAtomicSpecOf(*this, opcode, true)};
        if (!spec) {
            continue;
        }
        const Id func{OpFunction(spec->result_type, helper_control,
                                 TypeFunction(spec->result_type, U64, spec->value_type))};
        const Id addr{OpFunctionParameter(U64)};
        const Id value{OpFunctionParameter(spec->value_type)};
        AddLabel();
        const Id expected_var{spec->atomic ? Id{} : CasExpectedVariable(*this)};
        // The mirror's word takes its own compare-exchange loop (declared in the entry block).
        const Id mirror_expected_var{
            (spec->atomic || !Sirit::ValidId(alias)) ? Id{} : CasExpectedVariable(*this)};
        const Id device{OpFunctionCall(U64, translate, addr)};
        const Id then_label{OpLabel()};
        const Id end_label{OpLabel()};
        const Id mapped{OpINotEqual(U1, device, zero64)};
        OpSelectionMerge(end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(mapped, then_label, end_label);
        AddLabel(then_label);
        const Id pointer{OpConvertUToPtr(
            TypePointer(spv::StorageClass::PhysicalStorageBuffer, spec->word_type), device)};
        const Id before{
            EmitGlobalAtomicOnWord(*this, *spec, pointer, value, ALIGNMENT, expected_var)};
        // Info::raw_global_store_alias: the same operation on the alias target's word, so read-only
        // mirrors stay current and storage mirror writes also reach imported guest memory. The
        // value returned by the instruction remains the result from the raw-table target.
        write_through(addr, [&](Id mirror) {
            const Id mirror_pointer{OpConvertUToPtr(
                TypePointer(spv::StorageClass::PhysicalStorageBuffer, spec->word_type), mirror)};
            EmitGlobalAtomicOnWord(*this, *spec, mirror_pointer, value, ALIGNMENT,
                                   mirror_expected_var);
        });
        OpReturnValue(before);
        AddLabel(end_label);
        report(addr, RawCoverage::Op::Atomic);
        OpReturnValue(ConstantNull(spec->result_type));
        OpFunctionEnd();
        raw_global_atomic_funcs[index] = func;
    }

    // 8 and 16 bit stores replace the field inside the containing aligned word with a
    // compare-exchange loop, so neighbouring invocations cannot lose each other's bytes.
    raw_global_subword_write_func = OpFunction(
        void_id, helper_control, TypeFunction(void_id, U64, U32[1], U32[1], U32[1]));
    const Id word_addr{OpFunctionParameter(U64)};
    const Id field_value{OpFunctionParameter(U32[1])};
    const Id bit_offset{OpFunctionParameter(U32[1])};
    const Id bit_count{OpFunctionParameter(U32[1])};
    AddLabel();
    const Id word_device{OpFunctionCall(U64, translate, word_addr)};
    const Id store_label{OpLabel()};
    const Id store_end_label{OpLabel()};
    // With the coverage diagnostic a miss goes through its own block to report.
    const Id miss_label{Sirit::ValidId(report_miss) ? OpLabel() : store_end_label};
    const Id mapped{OpINotEqual(U1, word_device, zero64)};
    OpSelectionMerge(store_end_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(mapped, store_label, miss_label);
    AddLabel(store_label);
    const Id word_pointer{OpConvertUToPtr(u32_pointer_type, word_device)};
    const Id device_scope{Const(static_cast<u32>(spv::Scope::Device))};
    const Id loop_header{OpLabel()};
    const Id loop_body{OpLabel()};
    const Id loop_continue{OpLabel()};
    const Id loop_merge{OpLabel()};
    OpBranch(loop_header);
    AddLabel(loop_header);
    OpLoopMerge(loop_merge, loop_continue, spv::LoopControlMask::MaskNone);
    OpBranch(loop_body);
    AddLabel(loop_body);
    const Id expected{OpAtomicLoad(U32[1], word_pointer, device_scope, zero)};
    const Id desired{OpBitFieldInsert(U32[1], expected, field_value, bit_offset, bit_count)};
    const Id previous{OpAtomicCompareExchange(U32[1], word_pointer, device_scope, zero, zero,
                                              desired, expected)};
    OpBranchConditional(OpIEqual(U1, previous, expected), loop_merge, loop_continue);
    AddLabel(loop_continue);
    OpBranch(loop_header);
    AddLabel(loop_merge);
    // The mirror's word takes the same field (the other fields stay what the mirror holds, which
    // is what the guest word holds while the page is valid).
    write_through(word_addr, [&](Id mirror) {
        const Id mirror_pointer{OpConvertUToPtr(u32_pointer_type, mirror)};
        const Id mirror_header{OpLabel()};
        const Id mirror_body{OpLabel()};
        const Id mirror_continue{OpLabel()};
        const Id mirror_merge{OpLabel()};
        OpBranch(mirror_header);
        AddLabel(mirror_header);
        OpLoopMerge(mirror_merge, mirror_continue, spv::LoopControlMask::MaskNone);
        OpBranch(mirror_body);
        AddLabel(mirror_body);
        const Id mirror_expected{OpAtomicLoad(U32[1], mirror_pointer, device_scope, zero)};
        const Id mirror_desired{
            OpBitFieldInsert(U32[1], mirror_expected, field_value, bit_offset, bit_count)};
        const Id mirror_previous{OpAtomicCompareExchange(U32[1], mirror_pointer, device_scope,
                                                         zero, zero, mirror_desired,
                                                         mirror_expected)};
        OpBranchConditional(OpIEqual(U1, mirror_previous, mirror_expected), mirror_merge,
                            mirror_continue);
        AddLabel(mirror_continue);
        OpBranch(mirror_header);
        AddLabel(mirror_merge);
    });
    OpBranch(store_end_label);
    if (Sirit::ValidId(report_miss)) {
        AddLabel(miss_label);
        report(word_addr, RawCoverage::Op::StoreSubword);
        OpBranch(store_end_label);
    }
    AddLabel(store_end_label);
    OpReturn();
    OpFunctionEnd();
}

const EmitContext::GlobalMemoryFunctions& EmitContext::GlobalMemoryFunctionsOf(
    const IR::Inst& inst) const {
    const auto route{global_memory_route.find(&inst)};
    return global_memory_functions[route == global_memory_route.end() ? 0 : route->second + 1];
}

/// Defines the functions that route global memory accesses and atomics to the storage buffer
/// holding the address at run time.
void EmitContext::DefineGlobalMemoryFunctions(const IR::Program& program) {
    const Info& info{program.info};
    // When select_global_memory_buffer is enabled (NVIDIA proprietary driver), every access
    // selects among all bound buffers through route 0, avoiding duplicate per-route helper/search
    // functions and driver hangs on per-route OpSelect chains.
    if (!profile.select_global_memory_buffer) {
        global_memory_functions.resize(program.global_memory_routes.size() + 1);
        global_memory_route = program.global_memory_route;
    } else {
        global_memory_functions.resize(1);
    }
    if (!info.uses_global_memory || !profile.support_int64) {
        return;
    }
    if (profile.support_raw_global_memory && info.uses_raw_global_memory) {
        DefineRawGlobalMemoryFunctions(info);
        return;
    }
    // The storage buffer descriptors the functions being defined route among, and those functions
    std::span<const u32> buffers;
    GlobalMemoryFunctions* functions{};
    using DefPtr = Id StorageDefinitions::*;
    const Id zero{u32_zero_value};
    // The buffer a routed access goes through can differ between invocations, so a pointer chosen
    // per invocation and the element pointers derived from it are NonUniform.
    const auto element{[&](Id pointer_type, Id ssbo, Id index) {
        const Id pointer{OpAccessChain(pointer_type, ssbo, zero, index)};
        if (non_uniform_ids.contains(ssbo.value)) {
            Decorate(pointer, spv::Decoration::NonUniform);
            non_uniform_ids.insert(pointer.value);
        }
        return pointer;
    }};
    // Load: no side effects. Write: stores and atomics. PerBuffer: the access needs a uniform
    // buffer pointer (atomics that call a storage buffer CAS loop function).
    enum class Routing { Load, Write, PerBuffer };
    // Routes the access to the candidate buffer that holds the address. NVIDIA inlines the helpers
    // at every access, and a chain of one selection per buffer hangs its Linux driver in the
    // Ocarina programs of #619, with or without a return inside each branch. With
    // select_global_memory_buffer, OpSelect picks the buffer (a variable pointer) and the element
    // index, and one access goes through them: loads take no branch, and stores and atomics take
    // one for an address no buffer holds. Otherwise the access takes one branch per candidate
    // buffer, and every branch merges back so the helper returns once, from its last block.
    // Finds the candidate buffer of `buffers` that holds addr: its descriptor index (~0U when none)
    // and base
    struct Route {
        std::vector<u32> candidates;
        Id route;
        Id route_base;
    };
    const auto search{[&](Id addr) {
        const u64 ssbo_align_mask{~(profile.min_ssbo_alignment - 1U)};
        const auto base_address{[&](size_t index) {
            const auto& ssbo{info.storage_buffers_descriptors[index]};
            const Id words{EmitGetCbufU32x2(*this, IR::Value{ssbo.cbuf_index},
                                           IR::Value{ssbo.cbuf_offset})};
            if (ssbo.indirect_offset == ~0U) {
                return OpBitwiseAnd(U64, OpBitcast(U64, words), Constant(U64, ssbo_align_mask));
            }
            // The pointer is read through the buffer bound for the constant buffer pointer.
            const auto holder{std::ranges::find_if(
                info.storage_buffers_descriptors, [&ssbo](const auto& desc) {
                    return desc.cbuf_index == ssbo.cbuf_index &&
                           desc.cbuf_offset == ssbo.cbuf_offset && desc.indirect_offset == ~0U;
                })};
            if (holder == info.storage_buffers_descriptors.end()) {
                return Constant(U64, ~u64{0});
            }
            const size_t holder_index{static_cast<size_t>(
                std::distance(info.storage_buffers_descriptors.begin(), holder))};
            const Id holder_low{OpCompositeExtract(U32[1], words, 0U)};
            const Id byte{OpIAdd(U32[1],
                                 OpBitwiseAnd(U32[1], holder_low,
                                              Const(static_cast<u32>(profile.min_ssbo_alignment - 1))),
                                 Const(ssbo.indirect_offset))};
            const Id word_index{OpShiftRightLogical(U32[1], byte, Const(2U))};
            const Id storage{ssbos[holder_index].U32};
            const auto load_word{[&](Id word) {
                return OpLoad(U32[1], OpAccessChain(storage_types.U32.element, storage, zero, word));
            }};
            const Id pointer_words{OpCompositeConstruct(
                U32[2], load_word(word_index), load_word(OpIAdd(U32[1], word_index, Const(1U))))};
            return OpBitwiseAnd(U64, OpBitcast(U64, pointer_words), Constant(U64, ssbo_align_mask));
        }};
        std::vector<u32> candidates;
        std::vector<Id> bases;
        Id route{Const(~0U)};
        // The routed buffer's base, so the offset into it is computed once
        Id route_base{Constant(U64, u64{0})};
        if (info.native_global_bounds) {
            // Native pointers carry no NVN size word. The address belongs to the candidate buffer
            // whose base is the closest one at or below it (bounds come from the binding).
            Id best_distance{Constant(U64, ~u64{0})};
            for (const u32 index : buffers) {
                const Id base{base_address(index)};
                const Id distance{OpISub(U64, addr, base)};
                const Id better{OpLogicalAnd(U1, OpUGreaterThanEqual(U1, addr, base),
                                             OpULessThan(U1, distance, best_distance))};
                best_distance = OpSelect(U64, better, distance, best_distance);
                route = OpSelect(U32[1], better, Const(static_cast<u32>(index)), route);
                route_base = OpSelect(U64, better, base, route_base);
                candidates.push_back(static_cast<u32>(index));
                bases.push_back(base);
            }
        } else {
            std::vector<Id> holds;
            for (const u32 index : buffers) {
                if (!UsesNvnStorageDescriptor(info, info.storage_buffers_descriptors[index])) {
                    continue;
                }
                const auto& ssbo{info.storage_buffers_descriptors[index]};
                const Id ssbo_addr{base_address(index)};
                const Id size{EmitGetCbufU32(*this, IR::Value{ssbo.cbuf_index},
                                             IR::Value{ssbo.cbuf_offset + 8})};
                const Id ssbo_end{OpIAdd(U64, ssbo_addr, OpUConvert(U64, size))};
                holds.push_back(OpLogicalAnd(U1, OpUGreaterThanEqual(U1, addr, ssbo_addr),
                                             OpULessThan(U1, addr, ssbo_end)));
                candidates.push_back(static_cast<u32>(index));
                bases.push_back(ssbo_addr);
            }
            // The first buffer that holds the address wins.
            for (size_t i = holds.size(); i-- > 0;) {
                route = OpSelect(U32[1], holds[i], Const(candidates[i]), route);
                route_base = OpSelect(U64, holds[i], bases[i], route_base);
            }
        }
        return Route{std::move(candidates), route, route_base};
    }};
    // With select_global_memory_buffer every access searches all bound buffers. NVIDIA compiles
    // that search slowly at each inlined access (#1848: ~1.7 s per OoT compute pipeline), so the
    // search is one function that is not inlined, and the accesses stay inlined. Task and mesh
    // programs search inline: as calls, the first of two consecutive loads in Ocarina's tree task
    // program returned 0 on NVIDIA, so every tree cell drew its 256-record cap (upstream #907).
    Id search_function{};
    const bool search_call{profile.select_global_memory_buffer && stage != Stage::Task &&
                           stage != Stage::Mesh};
    const auto define_body{[&](DefPtr ssbo_member, Id addr, const StorageTypeDefinition& type_def,
                               u32 shift, Id result_type, Id fallback, Routing routing,
                               auto&& callback) -> Id {
        const Id header_label{AddLabel()};
        const bool select{routing != Routing::PerBuffer && profile.select_global_memory_buffer};
        Route found_route;
        if (select && ValidId(search_function)) {
            for (const u32 index : buffers) {
                if (info.native_global_bounds ||
                    UsesNvnStorageDescriptor(info, info.storage_buffers_descriptors[index])) {
                    found_route.candidates.push_back(index);
                }
            }
            const Id result{OpFunctionCall(U32[3], search_function, addr)};
            found_route.route = OpCompositeExtract(U32[1], result, 0U);
            found_route.route_base =
                OpBitcast(U64, OpVectorShuffle(U32[2], result, result, 1U, 2U));
        } else {
            found_route = search(addr);
        }
        const auto& [candidates, route, route_base] = found_route;
        const bool has_value{result_type.value != void_id.value};
        if (routing != Routing::PerBuffer && profile.select_global_memory_buffer) {
            if (candidates.empty()) {
                return fallback;
            }
            AddCapability(spv::Capability::VariablePointersStorageBuffer);
            Id ssbo_id{ssbos[candidates[0]].*ssbo_member};
            for (size_t i = 0; i < candidates.size(); ++i) {
                const Id routed{OpIEqual(U1, route, Const(candidates[i]))};
                ssbo_id = OpSelect(type_def.array, routed, ssbos[candidates[i]].*ssbo_member,
                                   ssbo_id);
            }
            Decorate(ssbo_id, spv::Decoration::NonUniform);
            non_uniform_ids.insert(ssbo_id.value);
            const Id found{OpINotEqual(U1, route, Const(~0U))};
            // The offset is the same for every candidate: route_base is the routed buffer's base.
            const Id offset{OpUConvert(U32[1], OpISub(U64, addr, route_base))};
            const Id ssbo_index{
                OpSelect(U32[1], found, OpShiftRightLogical(U32[1], offset, Const(shift)), zero)};
            if (routing == Routing::Load) {
                // An address no buffer holds reads element 0 of the first one and returns fallback.
                const Id result{
                    callback(element(type_def.element, ssbo_id, ssbo_index), ssbo_id, ssbo_index)};
                return OpSelect(result_type, found, result, fallback);
            }
            const Id then_label{OpLabel()};
            const Id merge_label{OpLabel()};
            OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(found, then_label, merge_label);
            AddLabel(then_label);
            const Id result{
                callback(element(type_def.element, ssbo_id, ssbo_index), ssbo_id, ssbo_index)};
            OpBranch(merge_label);
            AddLabel(merge_label);
            return has_value ? OpPhi(result_type, result, then_label, fallback, header_label)
                             : Id{};
        }
        Id value{fallback};
        Id predecessor{header_label};
        for (size_t i = 0; i < candidates.size(); ++i) {
            const Id routed{OpIEqual(U1, route, Const(candidates[i]))};
            const Id then_label{OpLabel()};
            const Id merge_label{OpLabel()};
            OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(routed, then_label, merge_label);
            AddLabel(then_label);
            const Id ssbo_id{ssbos[candidates[i]].*ssbo_member};
            const Id ssbo_offset{OpUConvert(U32[1], OpISub(U64, addr, route_base))};
            const Id ssbo_index{OpShiftRightLogical(U32[1], ssbo_offset, Const(shift))};
            const Id result{
                callback(element(type_def.element, ssbo_id, ssbo_index), ssbo_id, ssbo_index)};
            OpBranch(merge_label);
            AddLabel(merge_label);
            if (has_value) {
                value = OpPhi(result_type, result, then_label, value, predecessor);
            }
            predecessor = merge_label;
        }
        return value;
    }};
    // Keep multi-buffer routing helpers out-of-line in classic graphics stages (vertex,
    // tessellation, geometry, fragment) where pipeline_shader_emission disables
    // select_global_memory_buffer and drivers would otherwise inline the full per-buffer branch
    // ladder into every global load/store/atomic site (cuts Switch 2 graphics shader build times
    // from ~4-7 s to sub-second and reduces instruction cache pressure). Compute stages keep
    // inlinable helpers (select_global_memory_buffer already isolates search_function as
    // DontInline, and out-of-line branch helpers miscompiled Ocarina of Time's foliage compute
    // culling on NVIDIA); Task and Mesh stages also keep inline emission to avoid NVIDIA
    // mesh-stage call-frame miscompiles in Hyrule Field; single-buffer routes stay inline.
    // Keep multi-buffer global-memory routing helpers out of line (DontInline) across classic
    // graphics stages (upstream #907: inlining multi-buffer chains cut OoT Market from ~58 to
    // ~45 FPS). Compute, Task, Mesh, emulated mesh-as-compute, and single-buffer routes stay
    // inline to preserve Ocarina of Time foliage culling and Hyrule Field tree accuracy on NVIDIA;
    // select_global_memory_buffer uses search_function on multi-buffer compute shaders instead.
    // EDEN_NX2_LEAN_SSBO=0 restores unconditional inlining.
    static const bool lean_ssbo_helpers{[] {
        const char* const env = std::getenv("EDEN_NX2_LEAN_SSBO");
        return env == nullptr || env[0] != '0';
    }()};
    const auto helper_control{[&]() noexcept {
        if (lean_ssbo_helpers && !profile.select_global_memory_buffer &&
            !runtime_info.mesh_as_compute && stage != Stage::Compute && stage != Stage::Task &&
            stage != Stage::Mesh && buffers.size() > 1) {
            return spv::FunctionControlMask::DontInline;
        }
        return spv::FunctionControlMask::MaskNone;
    }};
    const auto define_load{[&](DefPtr ssbo_member, const StorageTypeDefinition& type_def, Id type,
                               u32 shift) {
        const Id function_type{TypeFunction(type, U64)};
        const Id func_id{OpFunction(type, helper_control(), function_type)};
        const Id addr{OpFunctionParameter(U64)};
        if (!profile.support_descriptor_aliasing && shift > 2) {
            AddLabel();
            std::array<Id, 4> words;
            const u32 count = 1U << (shift - 2);
            for (u32 i = 0; i < count; ++i) {
                const Id word_address{OpIAdd(U64, addr, Constant(U64, i * sizeof(u32)))};
                words[i] = OpFunctionCall(U32[1], functions->load_u32, word_address);
            }
            OpReturnValue(OpCompositeConstruct(type, std::span{words}.first(count)));
            OpFunctionEnd();
            return func_id;
        }
        // Buffers are bound at the device's storage buffer alignment (4 bytes on AMD), so an
        // aligned 8 or 16 byte guest access can start at any word of the binding and the vector
        // view (offset >> shift) would read the vector before it. Those accesses read the words
        // of the buffer found for the address.
        if (shift > 2 && profile.min_ssbo_alignment < (1U << shift)) {
            const u32 count = 1U << (shift - 2);
            OpReturnValue(define_body(
                &StorageDefinitions::U32, addr, storage_types.U32, 2, type, ConstantNull(type),
                Routing::Load, [&](Id, Id ssbo_id, Id word_index) {
                    std::array<Id, 4> words;
                    for (u32 i = 0; i < count; ++i) {
                        const Id index{i == 0 ? word_index : OpIAdd(U32[1], word_index, Const(i))};
                        words[i] =
                            OpLoad(U32[1], element(storage_types.U32.element, ssbo_id, index));
                    }
                    return OpCompositeConstruct(type, std::span{words}.first(count));
                }));
            OpFunctionEnd();
            return func_id;
        }
        OpReturnValue(define_body(ssbo_member, addr, type_def, shift, type, ConstantNull(type),
                                  Routing::Load, [&](Id ssbo_pointer, Id, Id) {
                                      return OpLoad(type, ssbo_pointer);
                                  }));
        OpFunctionEnd();
        return func_id;
    }};
    const auto define_write{[&](DefPtr ssbo_member, const StorageTypeDefinition& type_def, Id type,
                                u32 shift) {
        const Id function_type{TypeFunction(void_id, U64, type)};
        const Id func_id{OpFunction(void_id, helper_control(), function_type)};
        const Id addr{OpFunctionParameter(U64)};
        const Id data{OpFunctionParameter(type)};
        if (!profile.support_descriptor_aliasing && shift > 2) {
            AddLabel();
            const u32 count = 1U << (shift - 2);
            for (u32 i = 0; i < count; ++i) {
                const Id word_address{OpIAdd(U64, addr, Constant(U64, i * sizeof(u32)))};
                const Id word{OpCompositeExtract(U32[1], data, i)};
                OpFunctionCall(void_id, functions->write_u32, word_address, word);
            }
            OpReturn();
            OpFunctionEnd();
            return func_id;
        }
        if (shift > 2 && profile.min_ssbo_alignment < (1U << shift)) {
            const u32 count = 1U << (shift - 2);
            define_body(&StorageDefinitions::U32, addr, storage_types.U32, 2, void_id, Id{},
                        Routing::Write, [&](Id, Id ssbo_id, Id word_index) {
                            for (u32 i = 0; i < count; ++i) {
                                const Id index{i == 0 ? word_index
                                                      : OpIAdd(U32[1], word_index, Const(i))};
                                OpStore(element(storage_types.U32.element, ssbo_id, index),
                                        OpCompositeExtract(U32[1], data, i));
                            }
                            return Id{};
                        });
            OpReturn();
            OpFunctionEnd();
            return func_id;
        }
        define_body(ssbo_member, addr, type_def, shift, void_id, Id{}, Routing::Write,
                    [&](Id ssbo_pointer, Id, Id) {
                        OpStore(ssbo_pointer, data);
                        return Id{};
                    });
        OpReturn();
        OpFunctionEnd();
        return func_id;
    }};
    const auto define{
        [&](DefPtr ssbo_member, const StorageTypeDefinition& type_def, Id type, size_t size) {
            const u32 shift{static_cast<u32>(std::countr_zero(size))};
            const Id load_func{define_load(ssbo_member, type_def, type, shift)};
            const Id write_func{define_write(ssbo_member, type_def, type, shift)};
            return std::make_pair(load_func, write_func);
        }};
    // Atomics the storage buffer pass could not track: the same routing, with the operation on
    // the routed word (INC, DEC and F32 add through their storage buffer CAS loops).
    using enum GlobalAtomic;
    const auto kind_of = [](IR::Opcode opcode) -> std::optional<GlobalAtomic> {
        switch (opcode) {
        case IR::Opcode::GlobalAtomicIAdd32: return IAdd;
        case IR::Opcode::GlobalAtomicSMin32: return SMin;
        case IR::Opcode::GlobalAtomicUMin32: return UMin;
        case IR::Opcode::GlobalAtomicSMax32: return SMax;
        case IR::Opcode::GlobalAtomicUMax32: return UMax;
        case IR::Opcode::GlobalAtomicInc32: return Inc;
        case IR::Opcode::GlobalAtomicDec32: return Dec;
        case IR::Opcode::GlobalAtomicAnd32: return And;
        case IR::Opcode::GlobalAtomicOr32: return Or;
        case IR::Opcode::GlobalAtomicXor32: return Xor;
        case IR::Opcode::GlobalAtomicExchange32: return Exchange;
        case IR::Opcode::GlobalAtomicAddF32: return AddF32;
        default: return std::nullopt;
        }
    };
    // What each route's accesses use: loads and stores of 32, 64 and 128 bits, and atomics
    struct Uses {
        std::array<bool, 3> accesses{};
        std::array<bool, static_cast<size_t>(Count)> atomics{};
        bool compare_exchange{};
    };
    std::vector<Uses> uses(global_memory_functions.size());
    for (const IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            const auto route{global_memory_route.find(&inst)};
            Uses& use{uses[route == global_memory_route.end() ? 0 : route->second + 1]};
            switch (inst.GetOpcode()) {
            case IR::Opcode::LoadGlobal32:
            case IR::Opcode::WriteGlobal32:
                use.accesses[0] = true;
                break;
            case IR::Opcode::LoadGlobal64:
            case IR::Opcode::WriteGlobal64:
                use.accesses[1] = true;
                break;
            case IR::Opcode::LoadGlobal128:
            case IR::Opcode::WriteGlobal128:
                use.accesses[2] = true;
                break;
            case IR::Opcode::GlobalAtomicCompareExchange32:
                use.compare_exchange = true;
                break;
            default:
                if (const auto kind = kind_of(inst.GetOpcode())) {
                    use.atomics[static_cast<size_t>(*kind)] = true;
                }
                break;
            }
        }
    }
    const Id scope{Const(static_cast<u32>(spv::Scope::Device))};
    const auto define_atomic{[&](GlobalAtomic kind) {
        const Id type{kind == AddF32 ? F32[1] : U32[1]};
        const Id func_id{OpFunction(type, helper_control(),
                                    TypeFunction(type, U64, type))};
        const Id addr{OpFunctionParameter(U64)};
        const Id value{OpFunctionParameter(type)};
        const bool calls_cas_loop{kind == Inc || kind == Dec || kind == AddF32};
        OpReturnValue(define_body(
            &StorageDefinitions::U32, addr, storage_types.U32, 2, type, ConstantNull(type),
            calls_cas_loop ? Routing::PerBuffer : Routing::Write, [&](Id pointer, Id ssbo, Id word) {
                const auto atomic = [&](Id (Sirit::Module::*op)(Id, Id, Id, Id, Id)) {
                    return (this->*op)(U32[1], pointer, scope, zero, value);
                };
                switch (kind) {
                case IAdd:
                    return atomic(&Sirit::Module::OpAtomicIAdd);
                case SMin:
                    return atomic(&Sirit::Module::OpAtomicSMin);
                case UMin:
                    return atomic(&Sirit::Module::OpAtomicUMin);
                case SMax:
                    return atomic(&Sirit::Module::OpAtomicSMax);
                case UMax:
                    return atomic(&Sirit::Module::OpAtomicUMax);
                case And:
                    return atomic(&Sirit::Module::OpAtomicAnd);
                case Or:
                    return atomic(&Sirit::Module::OpAtomicOr);
                case Xor:
                    return atomic(&Sirit::Module::OpAtomicXor);
                case Exchange:
                    return atomic(&Sirit::Module::OpAtomicExchange);
                case Inc:
                    return OpFunctionCall(U32[1], increment_cas_ssbo, word, value, ssbo);
                case Dec:
                    return OpFunctionCall(U32[1], decrement_cas_ssbo, word, value, ssbo);
                case AddF32:
                    return OpFunctionCall(F32[1], f32_add_cas, word, value, ssbo);
                case Count:
                    break;
                }
                return ConstantNull(type);
            }));
        OpFunctionEnd();
        return func_id;
    }};
    const auto define_compare_exchange{[&] {
        const Id function_type{TypeFunction(U32[1], U64, U32[1], U32[1])};
        const Id func_id{OpFunction(U32[1], helper_control(), function_type)};
        const Id address{OpFunctionParameter(U64)};
        const Id comparator{OpFunctionParameter(U32[1])};
        const Id value{OpFunctionParameter(U32[1])};
        // No buffer matched: report success so a guest CAS retry loop cannot spin forever.
        OpReturnValue(define_body(&StorageDefinitions::U32, address, storage_types.U32, 2, U32[1],
                                  comparator, Routing::Write, [&](Id pointer, Id, Id) {
                                      return OpAtomicCompareExchange(U32[1], pointer, scope,
                                                                     u32_zero_value, u32_zero_value,
                                                                     value, comparator);
                                  }));
        OpFunctionEnd();
        return func_id;
    }};
    const bool routes_atomics{
        std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& descriptor) {
            return info.native_global_bounds || UsesNvnStorageDescriptor(info, descriptor);
        })};
    std::vector<u32> every_buffer(info.storage_buffers_descriptors.size());
    std::iota(every_buffer.begin(), every_buffer.end(), 0U);
    for (size_t route = 0; route < global_memory_functions.size(); ++route) {
        buffers = route == 0 ? std::span<const u32>{every_buffer}
                             : std::span<const u32>{program.global_memory_routes[route - 1]};
        functions = &global_memory_functions[route];
        const Uses& use{uses[route]};
        // Every access through all buffers gets its functions; the 64 and 128-bit ones read and
        // write words through the 32-bit ones without descriptor aliasing.
        const bool all{route == 0};
        const bool words{!profile.support_descriptor_aliasing &&
                         (use.accesses[1] || use.accesses[2])};
        search_function = Id{};
        if (search_call && buffers.size() > 1) {
            search_function = OpFunction(U32[3], spv::FunctionControlMask::DontInline,
                                         TypeFunction(U32[3], U64));
            const Id addr{OpFunctionParameter(U64)};
            AddLabel();
            const Route found{search(addr)};
            const Id base_words{OpBitcast(U32[2], found.route_base)};
            OpReturnValue(OpCompositeConstruct(U32[3], found.route,
                                               OpCompositeExtract(U32[1], base_words, 0U),
                                               OpCompositeExtract(U32[1], base_words, 1U)));
            OpFunctionEnd();
        }
        if (all || use.accesses[0] || words) {
            std::tie(functions->load_u32, functions->write_u32) =
                define(&StorageDefinitions::U32, storage_types.U32, U32[1], sizeof(u32));
        }
        if (all || use.accesses[1]) {
            std::tie(functions->load_u32x2, functions->write_u32x2) =
                define(&StorageDefinitions::U32x2, storage_types.U32x2, U32[2], sizeof(u32[2]));
        }
        if (all || use.accesses[2]) {
            std::tie(functions->load_u32x4, functions->write_u32x4) =
                define(&StorageDefinitions::U32x4, storage_types.U32x4, U32[4], sizeof(u32[4]));
        }
        if (!routes_atomics) {
            continue;
        }
        for (size_t index = 0; index < use.atomics.size(); ++index) {
            if (use.atomics[index]) {
                functions->atomics[index] = define_atomic(static_cast<GlobalAtomic>(index));
            }
        }
        if (use.compare_exchange) {
            functions->compare_exchange_u32 = define_compare_exchange();
        }
    }
}

void EmitContext::DefineRescalingInput(const Info& info) {
    if (!info.uses_rescaling_uniform) {
        return;
    }
    if (profile.unified_descriptor_binding) {
        DefineRescalingInputPushConstant();
    } else {
        DefineRescalingInputUniformConstant();
    }
}

void EmitContext::DefineRescalingInputPushConstant() {
    boost::container::static_vector<Id, 3> members{};
    u32 member_index{0};

    rescaling_textures_type = TypeArray(U32[1], Const(4u));
    Decorate(rescaling_textures_type, spv::Decoration::ArrayStride, 4u);
    members.push_back(rescaling_textures_type);
    rescaling_textures_member_index = member_index++;

    rescaling_images_type = TypeArray(U32[1], Const(NUM_IMAGE_SCALING_WORDS));
    Decorate(rescaling_images_type, spv::Decoration::ArrayStride, 4u);
    members.push_back(rescaling_images_type);
    rescaling_images_member_index = member_index++;

    if (stage != Stage::Compute) {
        members.push_back(F32[1]);
        rescaling_downfactor_member_index = member_index++;
    }
    const Id push_constant_struct{TypeStruct(std::span(members.data(), members.size()))};
    Decorate(push_constant_struct, spv::Decoration::Block);
    Name(push_constant_struct, "ResolutionInfo");

    MemberDecorate(push_constant_struct, rescaling_textures_member_index, spv::Decoration::Offset,
                   static_cast<u32>(offsetof(RescalingLayout, rescaling_textures)));
    MemberName(push_constant_struct, rescaling_textures_member_index, "rescaling_textures");

    MemberDecorate(push_constant_struct, rescaling_images_member_index, spv::Decoration::Offset,
                   static_cast<u32>(offsetof(RescalingLayout, rescaling_images)));
    MemberName(push_constant_struct, rescaling_images_member_index, "rescaling_images");

    if (stage != Stage::Compute) {
        MemberDecorate(push_constant_struct, rescaling_downfactor_member_index,
                       spv::Decoration::Offset,
                       static_cast<u32>(offsetof(RescalingLayout, down_factor)));
        MemberName(push_constant_struct, rescaling_downfactor_member_index, "down_factor");
    }
    const Id pointer_type{TypePointer(spv::StorageClass::PushConstant, push_constant_struct)};
    rescaling_push_constants = AddGlobalVariable(pointer_type, spv::StorageClass::PushConstant);
    Name(rescaling_push_constants, "rescaling_push_constants");

    if (profile.supported_spirv >= 0x00010400) {
        interfaces.push_back(rescaling_push_constants);
    }
}

void EmitContext::DefineRescalingInputUniformConstant() {
    const Id pointer_type{TypePointer(spv::StorageClass::UniformConstant, F32[4])};
    rescaling_uniform_constant =
        AddGlobalVariable(pointer_type, spv::StorageClass::UniformConstant);
    Decorate(rescaling_uniform_constant, spv::Decoration::Location, 0u);

    if (profile.supported_spirv >= 0x00010400) {
        interfaces.push_back(rescaling_uniform_constant);
    }
}

void EmitContext::DefineRenderArea(const Info& info) {
    if (!info.uses_render_area) {
        return;
    }

    if (profile.unified_descriptor_binding) {
        boost::container::static_vector<Id, 1> members{};
        u32 member_index{0};

        members.push_back(F32[4]);
        render_are_member_index = member_index++;

        const Id push_constant_struct{TypeStruct(std::span(members.data(), members.size()))};
        Decorate(push_constant_struct, spv::Decoration::Block);
        Name(push_constant_struct, "RenderAreaInfo");

        MemberDecorate(push_constant_struct, render_are_member_index, spv::Decoration::Offset, 0);
        MemberName(push_constant_struct, render_are_member_index, "render_area");

        const Id pointer_type{TypePointer(spv::StorageClass::PushConstant, push_constant_struct)};
        render_area_push_constant =
            AddGlobalVariable(pointer_type, spv::StorageClass::PushConstant);
        Name(render_area_push_constant, "render_area_push_constants");

        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(render_area_push_constant);
        }
    }
}

void EmitContext::DefineConstantBuffers(const Info& info, u32& binding) {
    if (info.constant_buffer_descriptors.empty()) {
        return;
    }
    if (!profile.support_descriptor_aliasing) {
        DefineConstBuffers(*this, info, &UniformDefinitions::U32x4, binding, U32[4], 'u',
                           sizeof(u32[4]));
        for (const ConstantBufferDescriptor& desc : info.constant_buffer_descriptors) {
            binding += desc.count;
        }
        return;
    }
    IR::Type types{info.used_constant_buffer_types | info.used_indirect_cbuf_types};
    if (True(types & IR::Type::U8)) {
        if (profile.support_int8 && profile.support_uniform_and_storage_buffer_8bit) {
            DefineConstBuffers(*this, info, &UniformDefinitions::U8, binding, U8, 'u', sizeof(u8));
            DefineConstBuffers(*this, info, &UniformDefinitions::S8, binding, S8, 's', sizeof(s8));
        } else {
            types |= IR::Type::U32;
        }
    }
    if (True(types & IR::Type::U16)) {
        if (profile.support_int16 && profile.support_uniform_and_storage_buffer_16bit) {
            DefineConstBuffers(*this, info, &UniformDefinitions::U16, binding, U16, 'u',
                               sizeof(u16));
            DefineConstBuffers(*this, info, &UniformDefinitions::S16, binding, S16, 's',
                               sizeof(s16));
        } else {
            types |= IR::Type::U32;
        }
    }
    if (True(types & IR::Type::U32)) {
        DefineConstBuffers(*this, info, &UniformDefinitions::U32, binding, U32[1], 'u',
                           sizeof(u32));
    }
    if (True(types & IR::Type::F32)) {
        DefineConstBuffers(*this, info, &UniformDefinitions::F32, binding, F32[1], 'f',
                           sizeof(f32));
    }
    if (True(types & IR::Type::U32x2)) {
        DefineConstBuffers(*this, info, &UniformDefinitions::U32x2, binding, U32[2], 'u',
                           sizeof(u32[2]));
    }
    binding += static_cast<u32>(info.constant_buffer_descriptors.size());
}

void EmitContext::DefineConstantBufferIndirectFunctions(const Info& info) {
    if (!info.uses_cbuf_indirect) {
        return;
    }
    const auto make_accessor{[&](Id buffer_type, Id UniformDefinitions::*member_ptr) {
        const Id func_type{TypeFunction(buffer_type, U32[1], U32[1])};
        const Id func{OpFunction(buffer_type, spv::FunctionControlMask::MaskNone, func_type)};
        const Id binding{OpFunctionParameter(U32[1])};
        const Id offset{OpFunctionParameter(U32[1])};

        AddLabel();

        const Id merge_label{OpLabel()};
        const Id uniform_type{uniform_types.*member_ptr};

        std::array<Id, Info::MAX_INDIRECT_CBUFS> buf_labels;
        std::array<Sirit::Literal, Info::MAX_INDIRECT_CBUFS> buf_literals;
        for (u32 i = 0; i < Info::MAX_INDIRECT_CBUFS; i++) {
            buf_labels[i] = OpLabel();
            buf_literals[i] = Sirit::Literal{i};
        }
        OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
        OpSwitch(binding, buf_labels[0], buf_literals, buf_labels);
        for (u32 i = 0; i < Info::MAX_INDIRECT_CBUFS; i++) {
            AddLabel(buf_labels[i]);
            const Id cbuf{cbufs[i].*member_ptr};
            const Id access_chain{OpAccessChain(uniform_type, cbuf, u32_zero_value, offset)};
            const Id result{OpLoad(buffer_type, access_chain)};
            OpReturnValue(result);
        }
        AddLabel(merge_label);
        OpUnreachable();
        OpFunctionEnd();
        return func;
    }};
    IR::Type types{info.used_indirect_cbuf_types};
    bool supports_aliasing = profile.support_descriptor_aliasing;
    if (supports_aliasing && True(types & IR::Type::U8)) {
        if (profile.support_int8 && profile.support_uniform_and_storage_buffer_8bit) {
            load_const_func_u8 = make_accessor(U8, &UniformDefinitions::U8);
        } else {
            types |= IR::Type::U32;
        }
    }
    if (supports_aliasing && True(types & IR::Type::U16)) {
        if (profile.support_int16 && profile.support_uniform_and_storage_buffer_16bit) {
            load_const_func_u16 = make_accessor(U16, &UniformDefinitions::U16);
        } else {
            types |= IR::Type::U32;
        }
    }
    if (supports_aliasing && True(types & IR::Type::F32)) {
        load_const_func_f32 = make_accessor(F32[1], &UniformDefinitions::F32);
    }
    if (supports_aliasing && True(types & IR::Type::U32)) {
        load_const_func_u32 = make_accessor(U32[1], &UniformDefinitions::U32);
    }
    if (supports_aliasing && True(types & IR::Type::U32x2)) {
        load_const_func_u32x2 = make_accessor(U32[2], &UniformDefinitions::U32x2);
    }
    if (!supports_aliasing || True(types & IR::Type::U32x4)) {
        load_const_func_u32x4 = make_accessor(U32[4], &UniformDefinitions::U32x4);
    }
}

void EmitContext::DefineStorageBuffers(const Info& info, u32& binding) {
    if (info.storage_buffers_descriptors.empty()) {
        return;
    }
    AddExtension("SPV_KHR_storage_buffer_storage_class");

    IR::Type used_types{profile.support_descriptor_aliasing ? info.used_storage_buffer_types
                                                            : IR::Type::U32};
    used_types |= IR::Type::U32;
    if (profile.support_int8 && profile.support_storage_buffer_8bit &&
        True(used_types & IR::Type::U8)) {
        DefineSsbos(*this, storage_types.U8, &StorageDefinitions::U8, info, binding, U8,
                    sizeof(u8));
        DefineSsbos(*this, storage_types.S8, &StorageDefinitions::S8, info, binding, S8,
                    sizeof(u8));
    }
    if (profile.support_int16 && profile.support_storage_buffer_16bit &&
        True(used_types & IR::Type::U16)) {
        DefineSsbos(*this, storage_types.U16, &StorageDefinitions::U16, info, binding, U16,
                    sizeof(u16));
        DefineSsbos(*this, storage_types.S16, &StorageDefinitions::S16, info, binding, S16,
                    sizeof(u16));
    }
    if (True(used_types & IR::Type::U32)) {
        DefineSsbos(*this, storage_types.U32, &StorageDefinitions::U32, info, binding, U32[1],
                    sizeof(u32));
    }
    if (True(used_types & IR::Type::F32)) {
        DefineSsbos(*this, storage_types.F32, &StorageDefinitions::F32, info, binding, F32[1],
                    sizeof(f32));
    }
    if (True(used_types & IR::Type::U64)) {
        DefineSsbos(*this, storage_types.U64, &StorageDefinitions::U64, info, binding, U64,
                    sizeof(u64));
    }
    if (True(used_types & IR::Type::U32x2)) {
        DefineSsbos(*this, storage_types.U32x2, &StorageDefinitions::U32x2, info, binding, U32[2],
                    sizeof(u32[2]));
    }
    if (True(used_types & IR::Type::U32x4)) {
        DefineSsbos(*this, storage_types.U32x4, &StorageDefinitions::U32x4, info, binding, U32[4],
                    sizeof(u32[4]));
    }
    for (const StorageBufferDescriptor& desc : info.storage_buffers_descriptors) {
        binding += desc.count;
    }
    const bool needs_function{
        info.uses_global_increment || info.uses_global_decrement || info.uses_atomic_f32_add ||
        info.uses_atomic_f16x2_add || info.uses_atomic_f16x2_min || info.uses_atomic_f16x2_max ||
        info.uses_atomic_f32x2_add || info.uses_atomic_f32x2_min || info.uses_atomic_f32x2_max};
    if (needs_function) {
        AddCapability(spv::Capability::VariablePointersStorageBuffer);
    }
    if (info.uses_global_increment) {
        increment_cas_ssbo = CasLoop(*this, Operation::Increment, storage_types.U32.array,
                                     storage_types.U32.element, U32[1], spv::Scope::Device);
    }
    if (info.uses_global_decrement) {
        decrement_cas_ssbo = CasLoop(*this, Operation::Decrement, storage_types.U32.array,
                                     storage_types.U32.element, U32[1], spv::Scope::Device);
    }
    if (info.uses_atomic_f32_add) {
        f32_add_cas = CasLoop(*this, Operation::FPAdd, storage_types.U32.array,
                              storage_types.U32.element, F32[1], spv::Scope::Device);
    }
    if (info.uses_atomic_f16x2_add) {
        f16x2_add_cas = CasLoop(*this, Operation::FPAdd, storage_types.U32.array,
                                storage_types.U32.element, F16[2], spv::Scope::Device);
    }
    if (info.uses_atomic_f16x2_min) {
        f16x2_min_cas = CasLoop(*this, Operation::FPMin, storage_types.U32.array,
                                storage_types.U32.element, F16[2], spv::Scope::Device);
    }
    if (info.uses_atomic_f16x2_max) {
        f16x2_max_cas = CasLoop(*this, Operation::FPMax, storage_types.U32.array,
                                storage_types.U32.element, F16[2], spv::Scope::Device);
    }
    if (info.uses_atomic_f32x2_add) {
        f32x2_add_cas = CasLoop(*this, Operation::FPAdd, storage_types.U32.array,
                                storage_types.U32.element, F32[2], spv::Scope::Device);
    }
    if (info.uses_atomic_f32x2_min) {
        f32x2_min_cas = CasLoop(*this, Operation::FPMin, storage_types.U32.array,
                                storage_types.U32.element, F32[2], spv::Scope::Device);
    }
    if (info.uses_atomic_f32x2_max) {
        f32x2_max_cas = CasLoop(*this, Operation::FPMax, storage_types.U32.array,
                                storage_types.U32.element, F32[2], spv::Scope::Device);
    }
}

void EmitContext::DefineTextureBuffers(const Info& info, u32& binding) {
    if (info.texture_buffer_descriptors.empty()) {
        return;
    }
    const spv::ImageFormat format{spv::ImageFormat::Unknown};
    image_buffer_type = TypeImage(F32[1], spv::Dim::Buffer, 0U, false, false, 1, format);

    const Id type{TypePointer(spv::StorageClass::UniformConstant, image_buffer_type)};
    texture_buffers.reserve(info.texture_buffer_descriptors.size());
    for (const TextureBufferDescriptor& desc : info.texture_buffer_descriptors) {
        if (desc.count != 1) {
            throw NotImplementedException("Array of texture buffers");
        }
        const Id id{AddGlobalVariable(type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding);
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        Name(id, NameOf(stage, desc, "texbuf"));
        texture_buffers.push_back({
            .id = id,
            .count = desc.count,
        });
        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(id);
        }
        ++binding;
    }
}

void EmitContext::DefineImageBuffers(const Info& info, u32& binding) {
    image_buffers.reserve(info.image_buffer_descriptors.size());
    for (const ImageBufferDescriptor& desc : info.image_buffer_descriptors) {
        const spv::ImageFormat format{GetImageFormat(desc.format)};
        const Id sampled_type{desc.is_integer ? U32[1] : F32[1]};
        const Id image_type{
            TypeImage(sampled_type, spv::Dim::Buffer, false, false, false, 2, format)};
        const Id pointer_type{TypePointer(spv::StorageClass::UniformConstant, image_type)};
        const Id id{AddGlobalVariable(pointer_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding);
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        Name(id, NameOf(stage, desc, "imgbuf"));
        if (info.requires_coherent_image_memory) {
            Decorate(id, spv::Decoration::Coherent);
        }
        image_buffers.push_back({
            .id = id,
            .image_type = image_type,
            .pointer_type = pointer_type,
            .count = desc.count,
            .is_integer = desc.is_integer,
        });
        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(id);
        }
        ++binding;
    }
}

void EmitContext::DefineTextures(const Info& info, u32& binding, u32& scaling_index) {
    textures.reserve(info.texture_descriptors.size());
    for (const TextureDescriptor& desc : info.texture_descriptors) {
        const Id image_type{ImageType(*this, desc)};
        const Id sampled_type{TypeSampledImage(image_type)};
        if (desc.bindless) {
            // The replay's table of this type, in its own set (bindless_table.h): the descriptors
            // of the draw (set 0) do not number it.
            if (!bindless_samplers.value) {
                bindless_sampler_type = TypeSampler();
                const Id array_type{
                    TypeArray(bindless_sampler_type, Const(BINDLESS_SAMPLER_CAPACITY))};
                bindless_sampler_pointer_type =
                    TypePointer(spv::StorageClass::UniformConstant, bindless_sampler_type);
                bindless_samplers = AddGlobalVariable(
                    TypePointer(spv::StorageClass::UniformConstant, array_type),
                    spv::StorageClass::UniformConstant);
                Decorate(bindless_samplers, spv::Decoration::Binding, BINDLESS_SAMPLER_BINDING);
                Decorate(bindless_samplers, spv::Decoration::DescriptorSet, BINDLESS_SET);
                Name(bindless_samplers, "bindless_samplers");
                if (profile.supported_spirv >= 0x00010400) {
                    interfaces.push_back(bindless_samplers);
                }
            }
            const Id array_type{TypeArray(image_type, Const(BINDLESS_TEXTURE_CAPACITY))};
            const Id id{AddGlobalVariable(TypePointer(spv::StorageClass::UniformConstant,
                                                      array_type),
                                          spv::StorageClass::UniformConstant)};
            Decorate(id, spv::Decoration::Binding,
                     BindlessTextureBinding(desc.type,
                                            BindlessClass(desc.is_integer, desc.is_depth)));
            Decorate(id, spv::Decoration::DescriptorSet, BINDLESS_SET);
            Name(id, NameOf(stage, desc, "bindless_tex"));
            textures.push_back({
                .id = id,
                .sampled_type = sampled_type,
                .pointer_type = TypePointer(spv::StorageClass::UniformConstant, image_type),
                .image_type = image_type,
                .count = BINDLESS_TEXTURE_CAPACITY,
                .is_multisample = desc.is_multisample,
                .is_integer = desc.is_integer,
                .bindless = true,
            });
            if (profile.supported_spirv >= 0x00010400) {
                interfaces.push_back(id);
            }
            ++scaling_index;
            continue;
        }
        const Id pointer_type{TypePointer(spv::StorageClass::UniformConstant, sampled_type)};
        const Id desc_type{DescType(*this, sampled_type, pointer_type, desc.count)};
        const Id id{AddGlobalVariable(desc_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding);
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        Name(id, NameOf(stage, desc, "tex"));
        textures.push_back({
            .id = id,
            .sampled_type = sampled_type,
            .pointer_type = pointer_type,
            .image_type = image_type,
            .count = desc.count,
            .is_multisample = desc.is_multisample,
            .is_integer = desc.is_integer,
        });
        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(id);
        }
        ++binding;
        ++scaling_index;
    }
    if (info.uses_atomic_image_u32) {
        image_u32 = TypePointer(spv::StorageClass::Image, U32[1]);
    }
}

void EmitContext::DefineImages(const Info& info, u32& binding, u32& scaling_index) {
    images.reserve(info.image_descriptors.size());
    for (const ImageDescriptor& desc : info.image_descriptors) {
        const Id sampled_type{desc.is_integer ? U32[1] : F32[1]};
        const Id image_type{ImageType(*this, desc, sampled_type)};
        const Id pointer_type{TypePointer(spv::StorageClass::UniformConstant, image_type)};
        const Id id{AddGlobalVariable(pointer_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding);
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        Name(id, NameOf(stage, desc, "img"));
        if (info.requires_coherent_image_memory) {
            Decorate(id, spv::Decoration::Coherent);
        }
        images.push_back({
            .id = id,
            .image_type = image_type,
            .pointer_type = pointer_type,
            .count = desc.count,
            .is_integer = desc.is_integer,
        });
        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(id);
        }
        ++binding;
        ++scaling_index;
    }
}

void EmitContext::DefineInputs(const IR::Program& program) {
    const Info& info{program.info};
    const VaryingState& loads{info.loads};

    if (info.uses_workgroup_id) {
        workgroup_id = DefineInput(*this, U32[3], false, spv::BuiltIn::WorkgroupId);
    }
    if (info.uses_local_invocation_id) {
        local_invocation_id = DefineInput(*this, U32[3], false, spv::BuiltIn::LocalInvocationId);
    }
    if (info.uses_invocation_id) {
        invocation_id = DefineInput(*this, U32[1], false, spv::BuiltIn::InvocationId);
    }
    if (info.uses_invocation_info &&
        (stage == Shader::Stage::TessellationControl || stage == Shader::Stage::TessellationEval)) {
        patch_vertices_in = DefineInput(*this, U32[1], false, spv::BuiltIn::PatchVertices);
    }
    if (info.uses_sample_id) {
        sample_id = DefineInput(*this, U32[1], false, spv::BuiltIn::SampleId);
        if (stage == Stage::Fragment) {
            Decorate(sample_id, spv::Decoration::Flat);
        }
    }
    if (info.uses_is_helper_invocation) {
        is_helper_invocation = DefineInput(*this, U1, false, spv::BuiltIn::HelperInvocation);
    }
    if (info.uses_subgroup_mask && SupportsSubgroups()) {
        subgroup_mask_eq = DefineInput(*this, U32[4], false, spv::BuiltIn::SubgroupEqMaskKHR);
        subgroup_mask_lt = DefineInput(*this, U32[4], false, spv::BuiltIn::SubgroupLtMaskKHR);
        subgroup_mask_le = DefineInput(*this, U32[4], false, spv::BuiltIn::SubgroupLeMaskKHR);
        subgroup_mask_gt = DefineInput(*this, U32[4], false, spv::BuiltIn::SubgroupGtMaskKHR);
        subgroup_mask_ge = DefineInput(*this, U32[4], false, spv::BuiltIn::SubgroupGeMaskKHR);
        if (stage == Stage::Fragment) {
            Decorate(subgroup_mask_eq, spv::Decoration::Flat);
            Decorate(subgroup_mask_lt, spv::Decoration::Flat);
            Decorate(subgroup_mask_le, spv::Decoration::Flat);
            Decorate(subgroup_mask_gt, spv::Decoration::Flat);
            Decorate(subgroup_mask_ge, spv::Decoration::Flat);
        }
    }
    if ((info.uses_fswzadd || info.uses_subgroup_invocation_id || info.uses_subgroup_shuffles ||
         (profile.warp_size_potentially_larger_than_guest &&
          (info.uses_subgroup_vote || info.uses_subgroup_mask))) &&
        SupportsSubgroups()) {
        AddCapability(spv::Capability::GroupNonUniform);
        subgroup_local_invocation_id =
            DefineInput(*this, U32[1], false, spv::BuiltIn::SubgroupLocalInvocationId);
        if (stage == Stage::Fragment) {
            Decorate(subgroup_local_invocation_id, spv::Decoration::Flat);
        }
    }
    if (info.uses_fswzadd) {
        const Id f32_one{Const(1.0f)};
        const Id f32_minus_one{Const(-1.0f)};
        const Id f32_zero{Const(0.0f)};
        fswzadd_lut_a = ConstantComposite(F32[4], f32_minus_one, f32_one, f32_minus_one, f32_zero);
        fswzadd_lut_b =
            ConstantComposite(F32[4], f32_minus_one, f32_minus_one, f32_one, f32_minus_one);
    }
    if (loads[IR::Attribute::PrimitiveId]) {
        primitive_id = DefineInput(*this, U32[1], false, spv::BuiltIn::PrimitiveId);
        if (stage == Stage::Fragment) {
            // A fragment PrimitiveId input requires Geometry (geometryShader is a required feature).
            AddCapability(spv::Capability::Geometry);
            Decorate(primitive_id, spv::Decoration::Flat);
        }
    }
    if (loads[IR::Attribute::Layer]) {
        AddCapability(spv::Capability::Geometry);
        layer = DefineInput(*this, U32[1], false, spv::BuiltIn::Layer);
        Decorate(layer, spv::Decoration::Flat);
    }
    const bool noperspective_as_smooth{
        stage == Stage::Fragment && runtime_info.scaled_noperspective_inputs &&
        std::ranges::any_of(info.interpolation, [](Interpolation mode) {
            return mode == Interpolation::NoPerspective;
        })};
    const bool quantized_frag_depth{stage == Stage::Fragment && info.stores_frag_depth &&
                                    runtime_info.quantize_frag_depth_d24};
    if (loads.AnyComponent(IR::Attribute::PositionX) ||
        info.passthrough.AnyComponent(IR::Attribute::PositionX) || noperspective_as_smooth ||
        quantized_frag_depth) {
        const bool is_fragment{stage == Stage::Fragment};
        if (!is_fragment && profile.has_broken_spirv_position_input) {
            need_input_position_indirect = true;

            const Id input_position_struct = TypeStruct(F32[4]);
            input_position = DefineInput(*this, input_position_struct, true);

            MemberDecorate(input_position_struct, 0, spv::Decoration::BuiltIn,
                           static_cast<unsigned>(spv::BuiltIn::Position));
            Decorate(input_position_struct, spv::Decoration::Block);
        } else {
            const spv::BuiltIn built_in{is_fragment ? spv::BuiltIn::FragCoord
                                                    : spv::BuiltIn::Position};
            input_position = DefineInput(*this, F32[4], true, built_in);

            if (profile.support_geometry_shader_passthrough) {
                if (info.passthrough.AnyComponent(IR::Attribute::PositionX)) {
                    Decorate(input_position, spv::Decoration::PassthroughNV);
                }
            }
        }
    }
    // BaseInstance, BaseVertex and DrawIndex are only valid with the DrawParameters capability.
    const auto define_draw_parameter{[this](spv::BuiltIn builtin) {
        AddExtension("SPV_KHR_shader_draw_parameters");
        AddCapability(spv::Capability::DrawParameters);
        return DefineInput(*this, U32[1], true, builtin);
    }};
    if (loads[IR::Attribute::InstanceId]) {
        if (profile.support_vertex_instance_id) {
            instance_id = DefineInput(*this, U32[1], true, spv::BuiltIn::InstanceId);
            if (loads[IR::Attribute::BaseInstance]) {
                base_instance = define_draw_parameter(spv::BuiltIn::BaseInstance);
            }
        } else {
            instance_index = DefineInput(*this, U32[1], true, spv::BuiltIn::InstanceIndex);
            base_instance = define_draw_parameter(spv::BuiltIn::BaseInstance);
        }
    } else if (loads[IR::Attribute::BaseInstance]) {
        base_instance = define_draw_parameter(spv::BuiltIn::BaseInstance);
    }
    if (loads[IR::Attribute::VertexId]) {
        if (profile.support_vertex_instance_id) {
            vertex_id = DefineInput(*this, U32[1], true, spv::BuiltIn::VertexId);
            if (loads[IR::Attribute::BaseVertex]) {
                base_vertex = define_draw_parameter(spv::BuiltIn::BaseVertex);
            }
        } else {
            vertex_index = DefineInput(*this, U32[1], true, spv::BuiltIn::VertexIndex);
            base_vertex = define_draw_parameter(spv::BuiltIn::BaseVertex);
        }
    } else if (loads[IR::Attribute::BaseVertex]) {
        base_vertex = define_draw_parameter(spv::BuiltIn::BaseVertex);
    }
    if (loads[IR::Attribute::DrawID]) {
        draw_index = define_draw_parameter(spv::BuiltIn::DrawIndex);
    }
    if (loads[IR::Attribute::FrontFace]) {
        front_face = DefineInput(*this, U1, true, spv::BuiltIn::FrontFacing);
    }
    if (loads[IR::Attribute::PointSpriteS] || loads[IR::Attribute::PointSpriteT]) {
        point_coord = DefineInput(*this, F32[2], true, spv::BuiltIn::PointCoord);
    }
    if (loads[IR::Attribute::TessellationEvaluationPointU] ||
        loads[IR::Attribute::TessellationEvaluationPointV]) {
        tess_coord = DefineInput(*this, F32[3], false, spv::BuiltIn::TessCoord);
    }
    for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
        const AttributeType input_type{runtime_info.generic_input_types[index]};
        if (!runtime_info.previous_stage_stores.Generic(index)) {
            continue;
        }
        if (!loads.Generic(index) && !info.passthrough.Generic(index)) {
            continue;
        }
        if (input_type == AttributeType::Disabled) {
            continue;
        }
        const Id type{GetAttributeType(*this, input_type)};
        const Id id{DefineInput(*this, type, true)};
        Decorate(id, spv::Decoration::Location, static_cast<u32>(index));
        Name(id, fmt::format("in_attr{}", index));
        input_generics[index] = GetAttributeInfo(*this, input_type, id);

        if (info.passthrough.Generic(index) && profile.support_geometry_shader_passthrough) {
            Decorate(id, spv::Decoration::PassthroughNV);
        }
        if (stage != Stage::Fragment) {
            continue;
        }
        const bool is_integer = input_type == AttributeType::SignedInt ||
                                input_type == AttributeType::UnsignedInt;
        if (runtime_info.per_primitive_generics.test(index)) {
            if (runtime_info.mesh_as_compute) {
                // Mesh emulation outputs per-primitive attributes per vertex in a vertex shader.
                Decorate(id, spv::Decoration::Flat);
                continue;
            }
            // Written per primitive by a mesh program: constant over the primitive.
            AddExtension("SPV_EXT_mesh_shader");
            AddCapability(spv::Capability::MeshShadingEXT);
            Decorate(id, spv::Decoration::PerPrimitiveEXT);
            if (is_integer) {
                Decorate(id, spv::Decoration::Flat);
            }
            continue;
        }
        if (is_integer) {
            Decorate(id, spv::Decoration::Flat);
        } else {
            switch (info.interpolation[index]) {
            case Interpolation::Smooth:
                break;
            case Interpolation::NoPerspective:
                // The previous stage may write it times Position.w (noperspective_outputs).
                if (!runtime_info.scaled_noperspective_inputs) {
                    Decorate(id, spv::Decoration::NoPerspective);
                } else {
                    noperspective_as_smooth_inputs[index] = true;
                }
                break;
            case Interpolation::Flat:
                Decorate(id, spv::Decoration::Flat);
                break;
            }
        }
    }
    if (stage == Stage::TessellationEval) {
        for (size_t index = 0; index < info.uses_patches.size(); ++index) {
            if (!info.uses_patches[index]) {
                continue;
            }
            const Id id{DefineInput(*this, F32[4], false)};
            Decorate(id, spv::Decoration::Patch);
            Decorate(id, spv::Decoration::Location, static_cast<u32>(index));
            patches[index] = id;
        }
    }
}

void EmitContext::DefineMesh(const IR::Program& program) {
    if (stage != Stage::Task && stage != Stage::Mesh) {
        return;
    }
    // Mesh emulation: a compute shader with the same workgroup memory, writing a storage buffer.
    const bool as_compute{runtime_info.mesh_as_compute};
    if (!as_compute) {
        AddExtension("SPV_EXT_mesh_shader");
        AddCapability(spv::Capability::MeshShadingEXT);
    }
    mesh_info = program.mesh;
    mesh_threads = program.workgroup_size[0];
    // The task's output region, which its mesh workgroups read as their input region. In the
    // emulation it is plain workgroup memory, copied through the task payload buffer.
    const spv::StorageClass payload_class{as_compute ? spv::StorageClass::Workgroup
                                                     : spv::StorageClass::TaskPayloadWorkgroupEXT};
    const Id payload_type{TypeArray(U32[1], Const((std::max)(mesh_info.payload_words, 1U)))};
    mesh_payload = DefineVariable(*this, payload_type, std::nullopt, payload_class);
    Name(mesh_payload, "task_payload");
    mesh_payload_u32 = TypePointer(payload_class, U32[1]);
    if (stage == Stage::Task) {
        return;
    }
    // Output attribute region: groups of 32 vertices, then groups of 32 primitives, one 32-word
    // line per attribute. The map region keeps one map byte per word.
    const u32 vertex_lines{static_cast<u32>(mesh_info.vertex_lines.size())};
    const u32 primitive_lines{static_cast<u32>(mesh_info.primitive_lines.size())};
    mesh_primitive_base = Common::DivCeil(mesh_info.max_vertices, 32U) * vertex_lines * 32;
    mesh_attribute_words = (std::max)(
        mesh_primitive_base + Common::DivCeil(mesh_info.max_primitives, 32U) * primitive_lines * 32,
        1U);
    mesh_map_bytes = 4 + mesh_info.PrimitiveVertices() * mesh_info.max_primitives;
    mesh_shared_u32 = TypePointer(spv::StorageClass::Workgroup, U32[1]);
    mesh_attributes = DefineVariable(*this, TypeArray(U32[1], Const(mesh_attribute_words)),
                                     std::nullopt, spv::StorageClass::Workgroup);
    Name(mesh_attributes, "isbe_attributes");
    mesh_map = DefineVariable(*this, TypeArray(U32[1], Const(mesh_map_bytes)), std::nullopt,
                              spv::StorageClass::Workgroup);
    Name(mesh_map, "isbe_map");
    mesh_primitive_count =
        DefineVariable(*this, U32[1], std::nullopt, spv::StorageClass::Workgroup);
    Name(mesh_primitive_count, "isbe_primitive_count");

    if (as_compute) {
        return; // the output buffer: DefineMeshEmulationOutput, after the storage buffers
    }

    const Id vertex_count{Const(mesh_info.max_vertices)};
    const Id primitive_count{Const(mesh_info.max_primitives)};
    const u32 primitive_vertices{mesh_info.PrimitiveVertices()};
    mesh_primitive_indices = DefineVariable(
        *this, TypeArray(U32[primitive_vertices], primitive_count),
        primitive_vertices == 1   ? spv::BuiltIn::PrimitivePointIndicesEXT
        : primitive_vertices == 2 ? spv::BuiltIn::PrimitiveLineIndicesEXT
                                  : spv::BuiltIn::PrimitiveTriangleIndicesEXT,
        spv::StorageClass::Output);
    std::array<Id, IR::NUM_GENERICS> generics[2]{};
    Id& point_size{mesh_point_size};
    Id primitive_layer{}, primitive_viewport{}, primitive_id_output{};
    if (primitive_vertices == 1) {
        point_size = DefineVariable(*this, TypeArray(F32[1], vertex_count), spv::BuiltIn::PointSize,
                                    spv::StorageClass::Output);
    }
    const auto per_primitive_builtin = [&](Id& variable, spv::BuiltIn builtin) {
        if (!Sirit::ValidId(variable)) {
            variable = DefineVariable(*this, TypeArray(U32[1], primitive_count), builtin,
                                      spv::StorageClass::Output);
            Decorate(variable, spv::Decoration::PerPrimitiveEXT);
        }
        return MeshLineTarget{variable, ~0U, false};
    };
    const auto target = [&](u32 line, bool per_primitive) -> MeshLineTarget {
        const IR::Attribute attribute{line};
        if (IR::IsGeneric(attribute)) {
            const u32 index{IR::GenericAttributeIndex(attribute)};
            Id& variable{generics[per_primitive][index]};
            if (!Sirit::ValidId(variable)) {
                variable = DefineVariable(
                    *this, TypeArray(F32[4], per_primitive ? primitive_count : vertex_count),
                    std::nullopt, spv::StorageClass::Output);
                Decorate(variable, spv::Decoration::Location, index);
                if (per_primitive) {
                    Decorate(variable, spv::Decoration::PerPrimitiveEXT);
                }
            }
            return {variable, IR::GenericAttributeElement(attribute), true};
        }
        if (per_primitive) {
            switch (attribute) {
            case IR::Attribute::PrimitiveId:
                return per_primitive_builtin(primitive_id_output, spv::BuiltIn::PrimitiveId);
            case IR::Attribute::Layer:
                return per_primitive_builtin(primitive_layer, spv::BuiltIn::Layer);
            case IR::Attribute::ViewportIndex:
                return per_primitive_builtin(primitive_viewport, spv::BuiltIn::ViewportIndex);
            default:
                return {};
            }
        }
        switch (attribute) {
        case IR::Attribute::PositionX:
        case IR::Attribute::PositionY:
        case IR::Attribute::PositionZ:
        case IR::Attribute::PositionW:
            if (!Sirit::ValidId(output_position)) {
                output_position = DefineVariable(*this, TypeArray(F32[4], vertex_count),
                                                 spv::BuiltIn::Position, spv::StorageClass::Output);
            }
            return {output_position,
                    static_cast<u32>(attribute) - static_cast<u32>(IR::Attribute::PositionX), true};
        case IR::Attribute::PointSize:
            if (!Sirit::ValidId(point_size)) {
                point_size = DefineVariable(*this, TypeArray(F32[1], vertex_count),
                                            spv::BuiltIn::PointSize, spv::StorageClass::Output);
            }
            return {point_size, ~0U, true};
        default:
            return {}; // per-primitive values in vertex lines have no vertex output
        }
    };
    for (const u32 line : mesh_info.vertex_lines) {
        mesh_vertex_targets.push_back(target(line, false));
    }
    for (const u32 line : mesh_info.primitive_lines) {
        mesh_primitive_targets.push_back(target(line, true));
    }
}

void EmitContext::DefineMeshEmulationOutput() {
    if ((stage != Stage::Mesh && stage != Stage::Task) || !runtime_info.mesh_as_compute) {
        return;
    }
    // u32 arrays: the output MeshEpilogue writes each workgroup's block to
    // (EmitMeshEmulationVertex) and the task payload, offset and argument buffers (RuntimeInfo).
    // Sirit shares identical types, so reuse the u32 storage buffer type when the program's own
    // storage buffers declared it: decorating it again would be invalid SPIR-V.
    AddExtension("SPV_KHR_storage_buffer_storage_class");
    Id struct_pointer{storage_types.U32.array};
    if (!Sirit::ValidId(struct_pointer)) {
        const Id array_type{TypeRuntimeArray(U32[1])};
        Decorate(array_type, spv::Decoration::ArrayStride, 4U);
        const Id struct_type{TypeStruct(array_type)};
        Decorate(struct_type, spv::Decoration::Block);
        MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);
        struct_pointer = TypePointer(spv::StorageClass::StorageBuffer, struct_type);
    }
    const auto define = [&](u32 binding, std::string_view name) {
        const Id buffer{AddGlobalVariable(struct_pointer, spv::StorageClass::StorageBuffer)};
        Decorate(buffer, spv::Decoration::Binding, binding);
        Decorate(buffer, spv::Decoration::DescriptorSet, 0U);
        Name(buffer, name);
        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(buffer);
        }
        return buffer;
    };
    mesh_output_u32 = TypePointer(spv::StorageClass::StorageBuffer, U32[1]);
    const bool has_task{runtime_info.mesh_task_payload_binding != ~0U};
    if (has_task) {
        mesh_task_payload_buffer =
            define(runtime_info.mesh_task_payload_binding, "mesh_emulation_task_payload");
    }
    if (stage == Stage::Task) {
        return;
    }
    mesh_output_buffer = define(runtime_info.mesh_output_binding, "mesh_emulation_output");
    if (has_task) {
        mesh_task_offsets_buffer =
            define(runtime_info.mesh_task_offsets_binding, "mesh_emulation_task_offsets");
        mesh_task_args_buffer =
            define(runtime_info.mesh_task_args_binding, "mesh_emulation_task_args");
        mesh_group_index = AddGlobalVariable(TypePointer(spv::StorageClass::Private, U32[1]),
                                             spv::StorageClass::Private);
        Name(mesh_group_index, "mesh_group_index");
        if (profile.supported_spirv >= 0x00010400) {
            interfaces.push_back(mesh_group_index);
        }
    }
}

void EmitContext::DefineOutputs(const IR::Program& program) {
    const Info& info{program.info};
    if (stage == Stage::Task || stage == Stage::Mesh) {
        return; // DefineMesh
    }
    const std::optional<u32> invocations{program.invocations};
    if (runtime_info.convert_depth_mode || info.stores.AnyComponent(IR::Attribute::PositionX) ||
        stage == Stage::VertexB) {
        output_position = DefineOutput(*this, F32[4], invocations, spv::BuiltIn::Position);
    }
    if (info.stores[IR::Attribute::PointSize] || runtime_info.fixed_state_point_size) {
        if (stage == Stage::Fragment) {
            throw NotImplementedException("Storing PointSize in fragment stage");
        }
        output_point_size = DefineOutput(*this, F32[1], invocations, spv::BuiltIn::PointSize);
    }
    if (info.stores.ClipDistances()) {
        if (stage == Stage::Fragment) {
            throw NotImplementedException("Storing ClipDistance in fragment stage");
        }
        if (profile.max_user_clip_distances > 0) {
            const u32 used{(std::min)(profile.max_user_clip_distances, 8u)};
            const std::array<Id, 8> zero{f32_zero_value, f32_zero_value, f32_zero_value,
                                         f32_zero_value, f32_zero_value, f32_zero_value,
                                         f32_zero_value, f32_zero_value};
            const Id type{TypeArray(F32[1], Const(used))};
            const Id initializer{ConstantComposite(type, std::span(zero).subspan(0, used))};
            clip_distances =
                DefineOutput(*this, type, invocations, spv::BuiltIn::ClipDistance, initializer);
        }
    }
    if (info.stores[IR::Attribute::Layer] &&
        (profile.support_viewport_index_layer_non_geometry || stage == Stage::Geometry)) {
        if (stage == Stage::Fragment) {
            throw NotImplementedException("Storing Layer in fragment stage");
        }
        layer = DefineOutput(*this, U32[1], invocations, spv::BuiltIn::Layer);
    }
    if (info.stores[IR::Attribute::ViewportIndex] &&
        (profile.support_viewport_index_layer_non_geometry || stage == Stage::Geometry)) {
        if (stage == Stage::Fragment) {
            throw NotImplementedException("Storing ViewportIndex in fragment stage");
        }
        viewport_index = DefineOutput(*this, U32[1], invocations, spv::BuiltIn::ViewportIndex);
    }
    if (info.stores[IR::Attribute::ViewportMask] && profile.support_viewport_mask) {
        viewport_mask = DefineOutput(*this, TypeArray(U32[1], Const(1u)), std::nullopt,
                                     spv::BuiltIn::ViewportMaskNV);
    }
    for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
        if (info.stores.Generic(index)) {
            DefineGenericOutput(*this, index, invocations);
        }
    }
    switch (stage) {
    case Stage::TessellationControl:
        if (info.stores_tess_level_outer) {
            const Id type{TypeArray(F32[1], Const(4U))};
            output_tess_level_outer =
                DefineOutput(*this, type, std::nullopt, spv::BuiltIn::TessLevelOuter);
            Decorate(output_tess_level_outer, spv::Decoration::Patch);
        }
        if (info.stores_tess_level_inner) {
            const Id type{TypeArray(F32[1], Const(2U))};
            output_tess_level_inner =
                DefineOutput(*this, type, std::nullopt, spv::BuiltIn::TessLevelInner);
            Decorate(output_tess_level_inner, spv::Decoration::Patch);
        }
        for (size_t index = 0; index < info.uses_patches.size(); ++index) {
            if (!info.uses_patches[index]) {
                continue;
            }
            const Id id{DefineOutput(*this, F32[4], std::nullopt)};
            Decorate(id, spv::Decoration::Patch);
            Decorate(id, spv::Decoration::Location, static_cast<u32>(index));
            patches[index] = id;
        }
        break;
    case Stage::Fragment:
        for (u32 index = 0; index < 8; ++index) {
            const bool need_dual_source = runtime_info.dual_source_blend && index <= 1;
            if (!need_dual_source && !info.stores_frag_color[index] &&
                !profile.need_declared_frag_colors) {
                continue;
            }
            const Id type{GetAttributeType(*this, runtime_info.color_output_types[index])};
            frag_color[index] = DefineOutput(*this, type, std::nullopt);
            // Correct mapping for dual-source blending
            if (runtime_info.dual_source_blend && index <= 1) {
                Decorate(frag_color[index], spv::Decoration::Location, 0u);
                Decorate(frag_color[index], spv::Decoration::Index, index);
                Name(frag_color[index], index == 0 ? "frag_color0" : "frag_color0_secondary");
            } else {
                Decorate(frag_color[index], spv::Decoration::Location, index);
                Name(frag_color[index], fmt::format("frag_color{}", index));
            }
        }
        if (info.stores_frag_depth) {
            frag_depth = DefineOutput(*this, F32[1], std::nullopt);
            Decorate(frag_depth, spv::Decoration::BuiltIn, spv::BuiltIn::FragDepth);
        }
        if (info.stores_sample_mask) {
            const Id array_type{TypeArray(U32[1], Const(1U))};
            sample_mask = DefineOutput(*this, array_type, std::nullopt);
            Decorate(sample_mask, spv::Decoration::BuiltIn, spv::BuiltIn::SampleMask);
        }
        break;
    default:
        break;
    }
}

} // namespace Shader::Backend::SPIRV
