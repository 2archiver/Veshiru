// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Untiles block-linear texture bytes (Switch 2 GOB order) into tightly packed rows of blocks on
/// the GPU, buffer to buffer: the emulated path's `block_linear_unswizzle_3d_bcn.comp`
/// (BlockLinearUnswizzle3DPass), for every texel size. The output is byte for byte what
/// `FillStorageUpload` writes for a tiled piece without a decode, so the upload's
/// vkCmdCopyBufferToImage regions are the CPU path's. The replay records the dispatches only:
/// nothing is untiled on the CPU.
class UntileGpu {
public:
    /// One (layer, level) of a storage, as `UploadPiece` describes it.
    struct Piece {
        u64 input_offset{};  ///< bytes from the start of the input buffer to the piece's first GOB
        u64 output_offset{}; ///< where its rows of blocks start in the output buffer
        u32 columns{};       ///< blocks per row
        u32 rows{};          ///< block rows
        u32 slices{1};       ///< depth slices
        u32 bytes_per_block{};
        u32 tile_height{}; ///< log2 GOBs per tile row (the level's block height)
        u32 tile_depth{};  ///< log2 GOBs per tile in depth
    };

    /// The shader's push constants for one piece. A row of `columns * bytes_per_block` bytes is
    /// read as elements of the largest power of two (<= 16 bytes) that divides it, as
    /// `Tegra::Texture::UnswizzleTexture` does: the swizzle keeps the low four bits of a byte
    /// offset, so the bytes land in the same places for any such element size.
    struct Params {
        u32 element_log2{};
        u32 elements_x{}; ///< elements per row
        u32 elements_y{};
        u32 elements_z{};
        u32 slice_size{};  ///< bytes per tile slice in depth (2^tile_depth slices)
        u32 block_size{};  ///< bytes per row of tiles
        u32 x_shift{};     ///< log2 bytes per tile
    };

    /// True when the device can run the shader (push descriptors, 8/16-bit storage buffer access)
    /// and `EDEN_NX2_NVN_NO_UNTILE_GPU` is not set.
    [[nodiscard]] static bool Supported(const Device& device);

    /// Bytes of tiled input the piece reads: whole GOBs, `Tegra::Texture::CalculateSize`.
    [[nodiscard]] static u64 InputSize(const Piece& piece);
    /// Bytes of linear output: columns * rows * slices * bytes_per_block.
    [[nodiscard]] static u64 OutputSize(const Piece& piece);
    /// The shader addresses bytes with 32-bit integers: false when the piece is too big for that
    /// (or empty).
    [[nodiscard]] static bool Fits(const Piece& piece);
    [[nodiscard]] static Params MakeParams(const Piece& piece);

    explicit UntileGpu(const Device& device);
    ~UntileGpu();

    /// Alignment of the input and output offsets (minStorageBufferOffsetAlignment, at least 16).
    [[nodiscard]] u64 OffsetAlignment() const {
        return offset_alignment;
    }
    /// The largest input or output range one piece may bind (maxStorageBufferRange).
    [[nodiscard]] u64 MaxRange() const {
        return max_range;
    }

    /// Records one dispatch per piece into `cmd` (outside a render pass). `input` needs
    /// InputSize() readable bytes at each piece's input offset, `output` OutputSize() writable ones
    /// at its output offset (both STORAGE_BUFFER usage, offsets OffsetAlignment()-aligned, every
    /// piece Fits()). The caller orders earlier writes of the input before the dispatches and the
    /// shader writes before its reads of `output`.
    void Record(vk::CommandBuffer cmd, VkBuffer input, VkBuffer output,
                std::span<const Piece> pieces) const;

private:
    const Device& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    u64 offset_alignment{16};
    u64 max_range{};
};

} // namespace Vulkan::Nvn
