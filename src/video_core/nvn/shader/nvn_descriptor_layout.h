// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Describes what each descriptor binding of a translated NVN2 pipeline stands for: which NVN
// resource the replay has to bind at SPIR-V binding N. The numbering mirrors EmitSPIRV exactly
// (unified descriptor binding, one counter shared by every stage of a pipeline), so the Vulkan
// side can build its descriptor set layout without looking at SPIR-V.
//
// Known limitation: a constant sampler ID that the shader ORs into a handle is folded to an
// immediate by the recompiler passes and cannot be recovered from Shader::Info. Such a texture
// shows up as a separate-texture word with no sampler word, and is reported as unsupported.

#pragma once

#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace VideoCommon::Nvn {

/// Numerically equal to the matching VkDescriptorType so the Vulkan side can cast.
enum class DescriptorKind : u8 {
    CombinedImageSampler = 1,
    StorageImage = 3,
    UniformTexelBuffer = 4,
    StorageTexelBuffer = 5,
    UniformBuffer = 6,
    StorageBuffer = 7,
};

enum class DescriptorRole : u8 {
    DriverBank,       ///< dynamic reads of c0 (driver table) that survived constant folding
    ProgramConstants, ///< dynamic reads of c1 that survived folding
    HandleTable,      ///< data reads of c2 (not handle reads)
    UniformBuffer,    ///< game-bound uniform buffer, `slot` = bank - 3
    StorageBuffer,    ///< NVN storage buffer, `slot` = driver table slot
    Texture,
    Image,
    TexelBuffer,      ///< texture buffer (uniform texel) or image buffer (storage texel)
};

struct HandleRef {
    HandleSource source{HandleSource::None};
    u32 slot{};        ///< table slot (first slot of an array)
    u32 bank{};
    u32 byte_offset{}; ///< offset of the word in its bank

    auto operator<=>(const HandleRef&) const = default;
};

/// Element `index` of the handle array that starts at `ref`: the word 8 bytes further per element.
/// In a handle table that word is the next slot. In a uniform buffer the slot names the buffer, so
/// it stays and only the offset moves (an array of handles the game wrote into one buffer).
[[nodiscard]] constexpr HandleRef HandleElement(HandleRef ref, u32 index) noexcept {
    ref.byte_offset += index * HandleStride;
    if (ref.source != HandleSource::UniformBuffer) {
        ref.slot += index;
    }
    return ref;
}

/// Where a storage buffer's {address, size} pair comes from.
enum class StorageSource : u8 {
    None,
    DriverTable,         ///< per-stage storage table of the driver bank (graphics)
    ComputeUniformEntry, ///< compute: uniform entry table of the per-shader driver bank
    ComputeStorageEntry, ///< compute: storage entry table of the per-shader driver bank
    UniformBufferWord,   ///< a word pair inside a game-bound uniform buffer
};

struct DescriptorBinding {
    u32 binding{}; ///< SPIR-V Binding (descriptor set 0)
    DescriptorKind kind{DescriptorKind::UniformBuffer};
    DescriptorRole role{DescriptorRole::UniformBuffer};
    u32 count{1}; ///< array size (the binding number advances by 1 for texture/image arrays)
    u32 slot{};   ///< uniform-buffer slot, or the table slot of a storage buffer (see below)
    /// Storage buffers: where the {u64 address, u32 size} pair lives. `slot` indexes that table;
    /// for UniformBufferWord it is the uniform buffer slot and `storage_offset` the byte offset.
    StorageSource storage_source{StorageSource::None};
    u32 storage_bank{};
    u32 storage_offset{}; ///< byte offset of the pair in `storage_bank`
    /// Storage buffers: index in Shader::Info::storage_buffers_descriptors of the stage (what a
    /// handle program's Storage operation names).
    u32 storage_index{};
    ShaderStage stage{};
    u32 stage_mask{}; ///< 1 << ShaderStage
    HandleRef texture;
    HandleRef sampler;
    /// Textures and images whose handle the shader reads from a table in memory: the program
    /// (Shader::Info::handle_programs[texture.slot]) that computes the handle word of a draw;
    /// `texture.source` is HandleSource::Program. The word is the combined texture and sampler
    /// handle, so `sampler` stays empty.
    std::vector<u32> handle_program;
    Shader::TextureType type{Shader::TextureType::Color2D};
    Shader::ImageFormat image_format{Shader::ImageFormat::Typeless};
    bool is_depth{};
    bool is_multisample{};
    bool is_integer{};
    bool is_written{};
    bool is_atomic_target{};
    bool is_raw_global_table{};
    bool is_read{};
    bool supported{true};
    std::string issue; ///< why `supported` is false
};

struct StageInfo {
    ShaderStage stage{};
    const Shader::Info& info;
    /// Compute only: control-derived driver bank layout (null = unknown).
    const ComputeDriverLayout* compute{};
};

struct LayoutOptions {
    /// Profile::support_descriptor_aliasing of the profile the stages are emitted with. With
    /// aliasing, EmitSPIRV advances the binding counter once per constant buffer descriptor
    /// instead of by its `count`.
    bool descriptor_aliasing{};
};

struct DescriptorLayout {
    std::vector<DescriptorBinding> bindings;
    u32 binding_count{}; ///< value of Bindings::unified after the last stage
    bool uses_rescaling_uniform{};
    bool uses_render_area{};
    /// A stage reads or writes untracked global memory through the guest -> device address table
    /// in the driver bank (Shader::Info::uses_raw_global_memory); `stores_raw_global_memory`: it
    /// writes.
    bool uses_raw_global_memory{};
    bool stores_raw_global_memory{};
    /// Every stage that stores raw global memory writes each store through to the mirror alias
    /// table (RawStoresWrittenThrough). False when no stage stores.
    bool raw_stores_written_through{};
    /// A compute program that must run 32 threads per subgroup (the compact tensor helper
    /// shuffles across the warp): its pipeline asks for a required subgroup size of 32.
    bool requires_guest_warp{};
    /// A stage samples through the replay's bindless table (set Shader::BINDLESS_SET): the
    /// pipeline layout carries that set and the draw binds it.
    bool uses_bindless_table{};
    u32 unsupported_count{};

    /// True when the pipeline needs push constants (rescaling or render area).
    [[nodiscard]] bool UsesPushConstants() const noexcept {
        return uses_rescaling_uniform || uses_render_area;
    }
    [[nodiscard]] bool AllSupported() const noexcept {
        return unsupported_count == 0;
    }
};

/// The program's raw stores and atomics all reach the mirror alias table too
/// (Shader::Info::raw_global_store_alias): a store repeats into the alias, an atomic repeats its
/// operation on the aliased word.
[[nodiscard]] inline bool RawStoresWrittenThrough(const Shader::Info& info) noexcept {
    return info.stores_raw_global_memory && info.raw_global_store_alias;
}

/// Builds the layout for `stages` in pipeline order, numbering bindings the way EmitSPIRV does
/// for the same stages emitted in that order with a shared Bindings.
[[nodiscard]] DescriptorLayout BuildDescriptorLayout(std::span<const StageInfo> stages,
                                                     const LayoutOptions& options = {});

/// Text form, for tests and the coverage report.
[[nodiscard]] std::string DescribeLayout(const DescriptorLayout& layout);

} // namespace VideoCommon::Nvn
