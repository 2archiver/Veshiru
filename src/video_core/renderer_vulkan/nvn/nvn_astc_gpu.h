// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Decodes block-linear ASTC texture bytes straight from a storage buffer into the levels of an
/// RGBA8 image, on the GPU (the legacy path's `astc_decoder.comp`, the same shader the default
/// `accelerate_astc = Gpu` runs). The replay thread only records dispatches: nothing is decoded,
/// unswizzled or copied on the CPU, and the input is the guest's own bytes (an imported pool).
class AstcGpuDecoder {
public:
    /// One mip level of a storage (all its layers are one dispatch).
    struct Level {
        u32 level{};
        u32 layers{1};
        u32 columns{};      ///< blocks per row
        u32 rows{};         ///< block rows
        u64 offset{};       ///< bytes from the start of the input range to this level of layer 0
        u32 layer_stride{}; ///< bytes between layers
        u32 block_width{};  ///< texels per block
        u32 block_height{};
        u32 tile_height{};  ///< log2 of the GOBs per tile row (the level's block height)
        u32 tile_depth{};
    };

    /// True when the device can run the decoder (push descriptors, a storable RGBA8 format) and
    /// `EDEN_NX2_NVN_NO_ASTC_GPU` is not set.
    [[nodiscard]] static bool Supported(const Device& device);

    explicit AstcGpuDecoder(const Device& device);
    ~AstcGpuDecoder();

    /// Alignment of the input range offsets (minStorageBufferOffsetAlignment).
    [[nodiscard]] u64 OffsetAlignment() const {
        return offset_alignment;
    }

    /// Format of the storage image views the decoder writes through.
    static constexpr VkFormat VIEW_FORMAT = VK_FORMAT_A8B8G8R8_UNORM_PACK32;

    /// Records the dispatches into `cmd` (outside a render pass). `image` must be in GENERAL
    /// layout with STORAGE usage (extended usage for sRGB images). `input` is bound from
    /// `input_offset + level.offset`, `input_size` bytes in all are readable from `input_offset`.
    /// Returns the image views, which must outlive the command buffer's execution.
    [[nodiscard]] std::vector<vk::ImageView> Record(vk::CommandBuffer cmd, VkBuffer input,
                                                     u64 input_offset, u64 input_size,
                                                     VkImage image,
                                                     std::span<const Level> levels) const;

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
