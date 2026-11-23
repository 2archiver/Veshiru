// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Size of one NVN arrays argument record ({count, instanceCount, first, baseInstance}).
inline constexpr u32 QUAD_ARRAYS_COMMAND_SIZE = 16;
/// Size of one VkDrawIndexedIndirectCommand.
inline constexpr u32 QUAD_INDEXED_COMMAND_SIZE = 20;
/// Quads the fixed index buffer covers; an indirect draw's quads past it are not drawn.
inline constexpr u32 QUAD_INDIRECT_MAX_QUADS = 1u << 16;

/// Indirect draws of QUADS: rewrites their argument records as VkDrawIndexedIndirectCommand on
/// the GPU (host_shaders/nvn_quad_indirect.comp), to be drawn as triangles with `IndexBuffer()`.
/// The count is often written by the GPU itself (particles), so the host cannot generate indices
/// for it. Recorded outside render passes; the caller orders the reads and writes with barriers.
class QuadIndirectGpu {
public:
    /// Push descriptors for the two storage buffers.
    [[nodiscard]] static bool Supported(const Device& device);

    /// `index_buffer`: QUAD_INDIRECT_MAX_QUADS * 6 u32 indices, host-visible, not yet filled.
    QuadIndirectGpu(const Device& device, vk::Buffer index_buffer);
    ~QuadIndirectGpu();

    /// Converts `draws` records `stride` bytes apart starting at `source_offset` of `source`
    /// into `draws` tightly packed 20-byte records at `destination_offset` of `destination`
    /// (aligned to the device's storage buffer offset alignment). `source_offset` and `stride`
    /// are multiples of 4.
    void Convert(vk::CommandBuffer cmd, VkBuffer source, u64 source_offset, u32 stride, u32 draws,
                 VkBuffer destination, u64 destination_offset) const;

    /// VK_INDEX_TYPE_UINT32 indices of the triangles of every quad.
    [[nodiscard]] VkBuffer IndexBuffer() const {
        return *index_buffer;
    }

private:
    const Device& device;
    vk::Buffer index_buffer;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
};

} // namespace Vulkan::Nvn
