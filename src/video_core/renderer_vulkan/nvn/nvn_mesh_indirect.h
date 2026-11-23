// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Size of one NVN2 1D mesh-task argument record ({count, first}, u32 each).
inline constexpr u32 MESH_TASKS_1D_COMMAND_SIZE = 8;
/// Size of one NVN2 3D mesh-task argument record ({x, y, z}), which is also
/// VkDrawMeshTasksIndirectCommandEXT.
inline constexpr u32 MESH_TASKS_3D_COMMAND_SIZE = 12;

/// Rewrites NVN2's 1D mesh-task argument records into VkDrawMeshTasksIndirectCommandEXT records
/// on the GPU (host_shaders/nvn_mesh_indirect.comp): the arguments of an indirect draw are often
/// written by the GPU itself (culling), so the host cannot convert them. Recorded outside render
/// passes; the caller orders the reads and writes with barriers.
class MeshIndirectGpu {
public:
    /// Push descriptors for the two storage buffers.
    [[nodiscard]] static bool Supported(const Device& device);

    explicit MeshIndirectGpu(const Device& device);
    ~MeshIndirectGpu();

    /// Converts `draws` records `stride` bytes apart starting at `source_offset` of `source`
    /// into `draws` tightly packed 12-byte records at `destination_offset` of `destination`
    /// (aligned to the device's storage buffer offset alignment). `source_offset` and `stride`
    /// are multiples of 4.
    void Convert(vk::CommandBuffer cmd, VkBuffer source, u64 source_offset, u32 stride, u32 draws,
                 VkBuffer destination, u64 destination_offset) const;

private:
    const Device& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
};

} // namespace Vulkan::Nvn
