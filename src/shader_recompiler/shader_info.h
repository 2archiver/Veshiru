// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>
#include <bitset>
#include <map>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/type.h"
#include "shader_recompiler/storage_write_extent.h"
#include "shader_recompiler/varying_state.h"

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Shader {

enum class ReplaceConstant : u32 {
    BaseInstance,
    BaseVertex,
    DrawID,
};

enum class TextureType : u32 {
    Color1D,
    ColorArray1D,
    Color2D,
    ColorArray2D,
    Color3D,
    ColorCube,
    ColorArrayCube,
    Buffer,
    Color2DRect,
};
constexpr u32 NUM_TEXTURE_TYPES = 9;

enum class TexturePixelFormat {
    A8B8G8R8_UNORM,
    A8B8G8R8_SNORM,
    A8B8G8R8_SINT,
    A8B8G8R8_UINT,
    R5G6B5_UNORM,
    B5G6R5_UNORM,
    A1R5G5B5_UNORM,
    A2B10G10R10_UNORM,
    A2B10G10R10_UINT,
    A2R10G10B10_UNORM,
    A1B5G5R5_UNORM,
    A5B5G5R1_UNORM,
    R8_UNORM,
    R8_SNORM,
    R8_SINT,
    R8_UINT,
    R16G16B16A16_FLOAT,
    R16G16B16A16_UNORM,
    R16G16B16A16_SNORM,
    R16G16B16A16_SINT,
    R16G16B16A16_UINT,
    B10G11R11_FLOAT,
    R32G32B32A32_UINT,
    BC1_RGBA_UNORM,
    BC2_UNORM,
    BC3_UNORM,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC7_UNORM,
    BC6H_UFLOAT,
    BC6H_SFLOAT,
    ASTC_2D_4X4_UNORM,
    B8G8R8A8_UNORM,
    R32G32B32A32_FLOAT,
    R32G32B32A32_SINT,
    R32G32_FLOAT,
    R32G32_SINT,
    R32_FLOAT,
    R16_FLOAT,
    R16_UNORM,
    R16_SNORM,
    R16_UINT,
    R16_SINT,
    R16G16_UNORM,
    R16G16_FLOAT,
    R16G16_UINT,
    R16G16_SINT,
    R16G16_SNORM,
    R32G32B32_FLOAT,
    A8B8G8R8_SRGB,
    R8G8_UNORM,
    R8G8_SNORM,
    R8G8_SINT,
    R8G8_UINT,
    R32G32_UINT,
    R16G16B16X16_FLOAT,
    R32_UINT,
    R32_SINT,
    ASTC_2D_8X8_UNORM,
    ASTC_2D_8X5_UNORM,
    ASTC_2D_5X4_UNORM,
    B8G8R8A8_SRGB,
    BC1_RGBA_SRGB,
    BC2_SRGB,
    BC3_SRGB,
    BC7_SRGB,
    A4B4G4R4_UNORM,
    G4R4_UNORM,
    ASTC_2D_4X4_SRGB,
    ASTC_2D_8X8_SRGB,
    ASTC_2D_8X5_SRGB,
    ASTC_2D_5X4_SRGB,
    ASTC_2D_5X5_UNORM,
    ASTC_2D_5X5_SRGB,
    ASTC_2D_10X8_UNORM,
    ASTC_2D_10X8_SRGB,
    ASTC_2D_6X6_UNORM,
    ASTC_2D_6X6_SRGB,
    ASTC_2D_10X6_UNORM,
    ASTC_2D_10X6_SRGB,
    ASTC_2D_10X5_UNORM,
    ASTC_2D_10X5_SRGB,
    ASTC_2D_10X10_UNORM,
    ASTC_2D_10X10_SRGB,
    ASTC_2D_12X10_UNORM,
    ASTC_2D_12X10_SRGB,
    ASTC_2D_12X12_UNORM,
    ASTC_2D_12X12_SRGB,
    ASTC_2D_8X6_UNORM,
    ASTC_2D_8X6_SRGB,
    ASTC_2D_6X5_UNORM,
    ASTC_2D_6X5_SRGB,
    E5B9G9R9_FLOAT,
    D32_FLOAT,
    D16_UNORM,
    X8_D24_UNORM,
    S8_UINT,
    D24_UNORM_S8_UINT,
    S8_UINT_D24_UNORM,
    D32_FLOAT_S8_UINT,
};

enum class ImageFormat : u32 {
    Typeless,
    R8_UINT,
    R8_SINT,
    R16_UINT,
    R16_SINT,
    R32_UINT,
    R32G32_UINT,
    R32G32B32A32_UINT,
};

enum class Interpolation {
    Smooth,
    Flat,
    NoPerspective,
};

struct ConstantBufferDescriptor {
    u32 index;
    u32 count;

    auto operator<=>(const ConstantBufferDescriptor&) const = default;
};

struct StorageBufferDescriptor {
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    bool is_written;
    /// When set, the buffer pointer is read from memory at this offset from the constant buffer
    /// pointer, which is bound as its own descriptor.
    u32 indirect_offset{~0U};
    /// VTX-HLE: a global atomic reaches the buffer.
    bool is_atomic_target{};
    /// How far the program's stores reach past the pointer (unbounded unless proven)
    StorageWriteExtent write_extent{};

    auto operator<=>(const StorageBufferDescriptor&) const = default;
};

/// `cbuf_index` of the storage buffer descriptor that carries the raw global memory address table
/// (HostTranslateInfo::nvn_raw_global_table_ssbo): not a guest storage buffer, bound by the host.
inline constexpr u32 RAW_GLOBAL_TABLE_CBUF{0xffffffffU};

[[nodiscard]] constexpr bool IsRawGlobalTable(const StorageBufferDescriptor& desc) noexcept {
    return desc.cbuf_index == RAW_GLOBAL_TABLE_CBUF;
}

/// The bit of `Info::global_atomics` of a GlobalAtomic* opcode: its position after
/// GlobalAtomicIAdd32 in the opcode list.
[[nodiscard]] constexpr u64 GlobalAtomicBit(IR::Opcode opcode) noexcept {
    return u64{1} << (static_cast<u32>(opcode) - static_cast<u32>(IR::Opcode::GlobalAtomicIAdd32));
}
static_assert(static_cast<u32>(IR::Opcode::GlobalAtomicMaxF32x2) -
                  static_cast<u32>(IR::Opcode::GlobalAtomicIAdd32) < 64,
              "Info::global_atomics has one bit per GlobalAtomic opcode");

struct TextureBufferDescriptor {
    bool has_secondary;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 shift_left;
    u32 secondary_cbuf_index;
    u32 secondary_cbuf_offset;
    u32 secondary_shift_left;
    u32 count;
    u32 size_shift;
    /// Index in Info::handle_programs when the handle is computed at draw time (read from a
    /// storage buffer), ~0 when it is the constant buffer word.
    u32 handle_program{~0U};

    auto operator<=>(const TextureBufferDescriptor&) const = default;
};
using TextureBufferDescriptors = boost::container::small_vector<TextureBufferDescriptor, 6>;

struct ImageBufferDescriptor {
    ImageFormat format;
    bool is_written;
    bool is_read;
    bool is_integer;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    u32 size_shift;
    /// Index in Info::handle_programs when the handle is computed at draw time (read from a
    /// storage buffer), ~0 when it is the constant buffer word.
    u32 handle_program{~0U};

    auto operator<=>(const ImageBufferDescriptor&) const = default;
};
using ImageBufferDescriptors = boost::container::small_vector<ImageBufferDescriptor, 2>;

struct TextureDescriptor {
    TextureType type;
    bool is_depth;
    bool is_multisample;
    bool is_integer;
    bool has_secondary;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 shift_left;
    u32 secondary_cbuf_index;
    u32 secondary_cbuf_offset;
    u32 secondary_shift_left;
    u32 count;
    u32 size_shift;
    /// Index in Info::handle_programs when the handle is computed at draw time (read from a
    /// storage buffer), ~0 when it is the constant buffer word above. With a count above 1 the
    /// program computes the handle of each element (HandleOp::Index): a handle the shader reads
    /// from a table at an index only each invocation knows, the shader indexing the array.
    u32 handle_program{~0U};
    /// The shader reads the handle at run time (bindless_table.h): the descriptor names the
    /// type's table of the replay's bindless set, not a texture of this draw.
    bool bindless{};
    /// Whether the shader indexes the descriptor: an array, whose count the descriptor limits set
    /// (LimitSampledTextureArrays). It stays one when they cut it to a single element.
    bool is_descriptor_array{};
    /// Whether the constant buffer word above reaches the handle only as the address of a load:
    /// a pointer, never a texture handle, which binds no texture (see HandleIsStoragePointer)
    bool handle_is_pointer_word{};

    auto operator<=>(const TextureDescriptor&) const = default;
};
using TextureDescriptors = boost::container::small_vector<TextureDescriptor, 12>;

struct ImageDescriptor {
    TextureType type;
    ImageFormat format;
    bool is_written;
    bool is_read;
    bool is_integer;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    u32 size_shift;
    bool is_byte_addressed{};
    /// Index in Info::handle_programs when the handle is computed at draw time (read from a
    /// storage buffer), ~0 when it is the constant buffer word.
    u32 handle_program{~0U};

    auto operator<=>(const ImageDescriptor&) const = default;
};
using ImageDescriptors = boost::container::small_vector<ImageDescriptor, 4>;

/// Switch 2 (NVN) task and mesh programs: the sizes their mesh registers set and, for mesh
/// programs, the attribute (IR::Attribute index) held by each 128-byte line of a 32-element group
/// in the emulated stage buffer: per-vertex lines from the program's output map, per-primitive
/// lines from the pass-through geometry header bound with it, in attribute order.
struct MeshInfo {
    u32 payload_words{}; ///< Task output region / mesh input region, in 32-bit words
    u32 max_vertices{};
    u32 max_primitives{};
    u32 topology{}; ///< SET_MESH_SHADER_A output topology (0 = points, 1 = lines, 4 = triangles)
    boost::container::static_vector<u32, 136> vertex_lines;
    boost::container::static_vector<u32, 136> primitive_lines;

    /// Vertex indices per primitive, which is also its byte count in the map region.
    [[nodiscard]] u32 PrimitiveVertices() const noexcept {
        return topology == 0 ? 1 : topology == 1 ? 2 : 3;
    }
};

struct Info {
    static constexpr size_t MAX_INDIRECT_CBUFS{14};
    static constexpr size_t MAX_CBUFS{18};
    /// Native (Switch 2) programs take their storage buffers from raw pointers, as many as they
    /// read: Ocarina of Time's foliage culling binds 37 for its lists and per-type instance tables.
    static constexpr size_t MAX_SSBOS{64};
    static constexpr size_t MAX_GRAPHICS_SSBOS{32};

    bool is_task{}; ///< Switch 2 task program (Stage::Task)
    bool is_mesh{}; ///< Switch 2 mesh program (Stage::Mesh)
    bool uses_workgroup_id{};
    bool uses_local_invocation_id{};
    bool uses_invocation_id{};
    bool uses_invocation_info{};
    bool uses_sample_id{};
    bool uses_is_helper_invocation{};
    bool uses_subgroup_invocation_id{};
    bool uses_subgroup_shuffles{};
    bool uses_quad_shuffles{};
    std::array<bool, 30> uses_patches{};

    std::array<Interpolation, 32> interpolation{};
    VaryingState loads;
    VaryingState stores;
    VaryingState passthrough;

    std::map<IR::Attribute, IR::Attribute> legacy_stores_mapping;

    bool loads_indexed_attributes{};

    std::array<bool, 8> stores_frag_color{};
    /// Components (bit 0 = R ... bit 3 = A) of each render target the fragment shader writes
    std::array<u8, 8> frag_color_components{};
    bool stores_sample_mask{};
    bool stores_frag_depth{};

    bool stores_tess_level_outer{};
    bool stores_tess_level_inner{};

    bool stores_indexed_attributes{};

    bool stores_global_memory{};
    bool uses_local_memory{};

    bool uses_fp16{};
    bool uses_fp64{};
    bool uses_fp16_denorms_flush{};
    bool uses_fp16_denorms_preserve{};
    bool uses_fp32_denorms_flush{};
    bool uses_fp32_denorms_preserve{};
    bool uses_int8{};
    bool uses_int16{};
    bool uses_int64{};
    bool uses_image_1d{};
    bool uses_sampled_1d{};
    bool uses_sparse_residency{};
    bool uses_demote_to_helper_invocation{};
    bool uses_subgroup_vote{};
    /// Set when a barrier in conditional control flow was dropped, see ConditionalBarrierPass.
    bool removed_conditional_barrier{};
    bool uses_subgroup_mask{};
    bool uses_fswzadd{};
    bool uses_derivatives{};
    bool uses_typeless_image_reads{};
    bool uses_typeless_image_writes{};
    bool uses_image_buffers{};
    bool uses_shared_increment{};
    bool uses_shared_decrement{};
    bool uses_global_increment{};
    bool uses_global_decrement{};
    bool uses_atomic_f32_add{};
    bool uses_atomic_f16x2_add{};
    bool uses_atomic_f16x2_min{};
    bool uses_atomic_f16x2_max{};
    bool uses_atomic_f32x2_add{};
    bool uses_atomic_f32x2_min{};
    bool uses_atomic_f32x2_max{};
    bool uses_atomic_s32_min{};
    bool uses_atomic_s32_max{};
    bool uses_int64_bit_atomics{};
    bool uses_global_memory{};
    bool requires_coherent_global_memory{};
    bool requires_coherent_image_memory{};
    /// A texture descriptor's handle word is a pointer the program doesn't bind as a storage
    /// buffer: the capture saw that pointer null or unmapped, and a capture with it mapped
    /// translates differently (a handle table instead of the word)
    bool binds_unmapped_pointer_word{};
    /// Set by the SPIR-V backend when HMMA runs on cooperative matrices, which need full subgroups
    bool uses_cooperative_matrix{};
    /// Native global accesses whose pointer is chosen at run time: every storage buffer descriptor
    /// is a candidate and its bound length (not an NVN size word) limits the range.
    bool native_global_bounds{};
    /// VTX-HLE: the program was translated with HostTranslateInfo::nvn_raw_global_memory.
    bool nvn_raw_global_memory{};
    /// VTX-HLE: untracked (raw) global memory instructions survive the storage buffer pass.
    /// `stores_raw_global_memory`: at least one of them writes. Only collected when
    /// `nvn_raw_global_memory` is set.
    bool uses_raw_global_memory{};
    bool stores_raw_global_memory{};
    /// The GlobalAtomic* opcodes (GlobalAtomicBit) in the program: with raw global memory the
    /// backend defines one address-to-memory atomic function for each.
    u64 global_atomics{};
    /// The raw global memory address table is the storage buffer `raw_global_table_index` (the last
    /// descriptor), see HostTranslateInfo::nvn_raw_global_table_ssbo. Otherwise it is read from the
    /// driver bank (c0).
    bool raw_global_table_ssbo{};
    u32 raw_global_table_index{};
    /// Misses of the raw global memory table (either place) are recorded in the coverage ring
    /// (HostTranslateInfo::raw_coverage), tagged with `raw_coverage_program` (the program hash the
    /// pipeline cache sets before emission).
    bool raw_coverage{};
    u64 raw_coverage_program{};
    /// The raw global memory translation reuses the invocation's last table entry
    /// (HostTranslateInfo::raw_global_last_hit).
    bool raw_global_last_hit{};
    /// Raw stores also write through to mirror targets in the driver bank table
    /// (HostTranslateInfo::raw_global_store_alias).
    bool raw_global_store_alias{};
    bool uses_atomic_image_u32{};
    bool uses_shadow_lod{};
    bool uses_rescaling_uniform{};
    bool uses_cbuf_indirect{};
    bool uses_render_area{};

    IR::Type used_constant_buffer_types{};
    IR::Type used_storage_buffer_types{};
    IR::Type used_indirect_cbuf_types{};

    u32 constant_buffer_mask{};
    std::array<u32, MAX_CBUFS> constant_buffer_used_sizes{};
    /// First byte of the driver bank (c0) buffer descriptor table and its entry count (16 for the
    /// fixed per-stage tables). `nvn_buffer_used` has one bit per entry. `nvn_buffer_window` is set
    /// for VTX-HLE's NVN2 compute programs, whose per-shader table is described by
    /// Environment::NvnComputeBufferWindow(); entries are then found by cbuf_offset, not by index.
    static constexpr u32 LEGACY_NVN_BUFFERS{16};
    static constexpr size_t MAX_NVN_BUFFERS{64};
    u32 nvn_buffer_base{};
    u32 nvn_buffer_count{LEGACY_NVN_BUFFERS};
    bool nvn_buffer_window{};
    std::bitset<MAX_NVN_BUFFERS> nvn_buffer_used{};

    bool requires_layer_emulation{};
    IR::Attribute emulated_layer{};

    u32 used_clip_distances{};

    boost::container::static_vector<ConstantBufferDescriptor, MAX_CBUFS>
        constant_buffer_descriptors;
    boost::container::static_vector<StorageBufferDescriptor, MAX_SSBOS> storage_buffers_descriptors;
    TextureBufferDescriptors texture_buffer_descriptors;
    ImageBufferDescriptors image_buffer_descriptors;
    TextureDescriptors texture_descriptors;
    /// Texture handles the shader reads from storage buffers, as programs over constant buffer
    /// and storage buffer words (see EvaluateHandleProgram), evaluated for each draw.
    std::vector<std::vector<u32>> handle_programs;
    ImageDescriptors image_descriptors;

    /// True when the shader only runs correctly if a subgroup spans a whole guest warp.
    [[nodiscard]] bool UsesWarpOps() const noexcept {
        return removed_conditional_barrier || uses_subgroup_vote || uses_subgroup_invocation_id ||
               uses_subgroup_shuffles || uses_subgroup_mask || uses_quad_shuffles ||
               uses_fswzadd || uses_cooperative_matrix;
    }

    bool operator==(const Info&) const = default;
};

template <typename Descriptors>
u32 NumDescriptors(const Descriptors& descriptors) {
    u32 num{};
    for (const auto& desc : descriptors) {
        num += desc.count;
    }
    return num;
}


/// Operations of a handle program, a stack machine over 32-bit words. Operands follow the
/// operation code in the program; values are popped right operand first.
enum class HandleOp : u32 {
    Immediate,     ///< value
    Cbuf,          ///< index, offset: the constant buffer word
    CbufDynamic,   ///< index: the word at the popped byte offset
    Storage,       ///< binding: the storage buffer word at the popped byte offset
    Add, Sub, Mul, And, Or, Xor, Shl, Shr, Sar, Neg, Not,
    UMin, UMax, SMin, SMax,
    SLt, SLe, SGt, SGe, ULt, ULe, UGt, UGe, Eq, Ne,
    LAnd, LOr, LXor, LNot,
    Select, ///< condition, true value, false value
    Tee,    ///< slot: keeps the top value in the slot (shared subexpressions)
    Load,   ///< slot: the value kept there
    Index,  ///< the element of the descriptor array being computed (see TextureDescriptor)
};

/// Runs a handle program; read_cbuf(index, offset) and read_storage(binding, offset) return the
/// words it reads, and element is the descriptor array element computed (HandleOp::Index).
/// Returns nothing for a malformed program.
template <typename ReadCbuf, typename ReadStorage>
std::optional<u32> EvaluateHandleProgram(std::span<const u32> code, ReadCbuf&& read_cbuf,
                                         ReadStorage&& read_storage, u32 element = 0) {
    std::array<u32, 64> stack{};
    std::array<u32, 256> slots{};
    size_t size = 0;
    const auto pop = [&]() -> u32 { return size > 0 ? stack[--size] : 0; };
    const auto push = [&](u32 value) {
        if (size < stack.size()) stack[size++] = value;
    };
    for (size_t pc = 0; pc < code.size();) {
        const HandleOp op = static_cast<HandleOp>(code[pc++]);
        const auto operand = [&]() -> u32 { return pc < code.size() ? code[pc++] : 0; };
        if (op == HandleOp::Immediate) {
            push(operand());
            continue;
        }
        if (op == HandleOp::Cbuf) {
            const u32 index = operand();
            push(read_cbuf(index, operand()));
            continue;
        }
        if (op == HandleOp::CbufDynamic) {
            const u32 index = operand();
            push(read_cbuf(index, pop()));
            continue;
        }
        if (op == HandleOp::Storage) {
            const u32 binding = operand();
            push(read_storage(binding, pop()));
            continue;
        }
        if (op == HandleOp::Neg || op == HandleOp::Not || op == HandleOp::LNot) {
            const u32 a = pop();
            push(op == HandleOp::Neg ? 0U - a : op == HandleOp::Not ? ~a : (a == 0 ? 1U : 0U));
            continue;
        }
        if (op == HandleOp::Tee) {
            const u32 slot = operand();
            if (slot < slots.size() && size > 0) slots[slot] = stack[size - 1];
            continue;
        }
        if (op == HandleOp::Load) {
            const u32 slot = operand();
            push(slot < slots.size() ? slots[slot] : 0U);
            continue;
        }
        if (op == HandleOp::Index) {
            push(element);
            continue;
        }
        if (op == HandleOp::Select) {
            const u32 false_value = pop(), true_value = pop(), condition = pop();
            push(condition != 0 ? true_value : false_value);
            continue;
        }
        const u32 b = pop(), a = pop();
        const s32 sa = static_cast<s32>(a), sb = static_cast<s32>(b);
        switch (op) {
        case HandleOp::Add: push(a + b); break;
        case HandleOp::Sub: push(a - b); break;
        case HandleOp::Mul: push(a * b); break;
        case HandleOp::And: push(a & b); break;
        case HandleOp::Or: push(a | b); break;
        case HandleOp::Xor: push(a ^ b); break;
        case HandleOp::Shl: push(b >= 32 ? 0U : a << b); break;
        case HandleOp::Shr: push(b >= 32 ? 0U : a >> b); break;
        case HandleOp::Sar: push(static_cast<u32>(sa >> (b >= 32 ? 31 : b))); break;
        case HandleOp::UMin: push(a < b ? a : b); break;
        case HandleOp::UMax: push(a > b ? a : b); break;
        case HandleOp::SMin: push(static_cast<u32>(sa < sb ? sa : sb)); break;
        case HandleOp::SMax: push(static_cast<u32>(sa > sb ? sa : sb)); break;
        case HandleOp::SLt: push(sa < sb); break;
        case HandleOp::SLe: push(sa <= sb); break;
        case HandleOp::SGt: push(sa > sb); break;
        case HandleOp::SGe: push(sa >= sb); break;
        case HandleOp::ULt: push(a < b); break;
        case HandleOp::ULe: push(a <= b); break;
        case HandleOp::UGt: push(a > b); break;
        case HandleOp::UGe: push(a >= b); break;
        case HandleOp::Eq: push(a == b); break;
        case HandleOp::Ne: push(a != b); break;
        case HandleOp::LAnd: push(a != 0 && b != 0); break;
        case HandleOp::LOr: push(a != 0 || b != 0); break;
        case HandleOp::LXor: push((a != 0) != (b != 0)); break;
        default: return std::nullopt;
        }
    }
    if (size != 1) return std::nullopt;
    return stack[0];
}

} // namespace Shader
