// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Moves the bytes that two textures of different shapes share in a memory pool from one image to
/// the other on the GPU (host_shaders/nvn_alias_transfer.comp). vkCmdCopyImage only copies between
/// images whose bytes are laid out alike; two textures over the same pool bytes with other sizes,
/// texel sizes or block heights see each other's bytes through their own layouts, which is what
/// the shader reproduces: each texel of a piece goes to (or comes from) the byte offset its layout
/// puts it at, inside a window of the pool's bytes.
///
/// All the work is recorded outside render passes into one command buffer; `Dispatch` expects the
/// buffer bound by `Begin` to hold the piece's tight texels and the window at the given offsets.
class AliasTransferGpu {
public:
    /// One level of one layer of a texture, tightly copied into the scratch buffer.
    struct Piece {
        u32 tight_offset{}; ///< byte offset of the piece's texels in the scratch buffer
        s32 piece_base{};   ///< byte offset of the piece's first byte relative to the window start
        u32 cols{};         ///< texels per row
        u32 rows{};
        u32 slices{};
        u32 bytes_per_texel{};
        u32 tile_height{}; ///< log2 GOBs per block, block-linear pieces
        u32 tile_depth{};
        u32 pitch{};       ///< linear pieces: bytes per row; 0 = block-linear
    };

    enum class Mode : u32 {
        Scatter = 0, ///< tight texels -> window
        Gather = 1,  ///< window -> tight texels
    };

    /// Push descriptors and storage buffers with 8-bit access (the shader only moves bytes).
    [[nodiscard]] static bool Supported(const Device& device);

    explicit AliasTransferGpu(const Device& device);
    ~AliasTransferGpu();

    /// Largest scratch buffer the shader can address (32-bit byte offsets).
    [[nodiscard]] u64 MaxScratchBytes() const {
        return max_scratch;
    }

    /// Binds the pipeline and the scratch buffer (`size` bytes from its start).
    void Begin(vk::CommandBuffer cmd, VkBuffer scratch, u64 size) const;

    /// One piece, `window_size` bytes of window at `window_offset` in the scratch buffer.
    void Dispatch(vk::CommandBuffer cmd, Mode mode, u32 window_offset, u32 window_size,
                  const Piece& piece) const;

private:
    const Device& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    u64 max_scratch{};
};

} // namespace Vulkan::Nvn
