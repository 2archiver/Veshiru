// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Copies the vertex ranges of a render pass from an imported pool into device-local memory with
/// shader loads (host_shaders/nvn_vertex_stage.comp, tasks built by AppendVertexStageTasks in
/// nvn_vertex_stage.h) instead of a transfer command: from host memory, vkCmdCopyBuffer slowed
/// 3-5x while another process shared the GPU, shader reads ~1.4x. Recorded outside render passes;
/// the caller orders the writes before the vertex fetches with a barrier.
class VertexStageGpu {
public:
    /// Push descriptors for the three storage buffers.
    [[nodiscard]] static bool Supported(const Device& device);

    explicit VertexStageGpu(const Device& device);
    ~VertexStageGpu();

    /// Largest source window one dispatch can bind.
    [[nodiscard]] u64 MaxSourceRange() const {
        return max_range;
    }
    /// A source window starts at a multiple of this (the storage buffer offset alignment, at
    /// least 256 so the window keeps the source offsets modulo 16).
    [[nodiscard]] u64 SourceAlignment() const {
        return source_alignment;
    }

    /// Runs `count` tasks of the table at [`tasks_offset`, + `tasks_range`) of `tasks`, from task
    /// `first` on: task units index the window [`source_offset`, + `source_range`) of `source`
    /// and the whole of `destination`.
    void Copy(vk::CommandBuffer cmd, VkBuffer source, u64 source_offset, u64 source_range,
              VkBuffer destination, VkBuffer tasks, u64 tasks_offset, u64 tasks_range, u32 first,
              u32 count) const;

private:
    const Device& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    u64 max_range{};
    u64 source_alignment{};
};

} // namespace Vulkan::Nvn
