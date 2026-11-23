// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#include "video_core/host_shaders/nvn_vertex_stage_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_vertex_stage_gpu.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

/// The push constant block of nvn_vertex_stage.comp.
struct PushConstants {
    u32 first_task;
};
static_assert(sizeof(PushConstants) == 4);

constexpr u32 MAX_GROUPS_X = 65535;

} // Anonymous namespace

bool VertexStageGpu::Supported(const Device& device) {
    return device.IsKhrPushDescriptorSupported() && device.MaxPushDescriptors() >= 3;
}

VertexStageGpu::VertexStageGpu(const Device& device_) : device{device_} {
    max_range = device.GetMaxStorageBufferRange();
    source_alignment = (std::max<u64>)(device.GetStorageBufferAlignment(), 256);
    const vk::Device& logical = device.GetLogical();
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i] = {
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        };
    }
    set_layout = logical.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    pipeline_layout = logical.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    });
    const std::span<const u32> code = NVN_VERTEX_STAGE_COMP_SPV;
    module = logical.CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = static_cast<u32>(code.size_bytes()),
        .pCode = code.data(),
    });
    pipeline = logical.CreateComputePipeline(
        VkComputePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            .layout = *pipeline_layout,
            .basePipelineHandle = {},
            .basePipelineIndex = 0,
        },
        // Created on the replay thread while other threads create theirs: the shared static
        // pipeline cache is externally synchronized, this one small pipeline skips it.
        VK_NULL_HANDLE);
}

VertexStageGpu::~VertexStageGpu() = default;

void VertexStageGpu::Copy(vk::CommandBuffer cmd, VkBuffer source, u64 source_offset,
                          u64 source_range, VkBuffer destination, VkBuffer tasks,
                          u64 tasks_offset, u64 tasks_range, u32 first, u32 count) const {
    const std::array<VkDescriptorBufferInfo, 3> buffers{{
        {.buffer = source, .offset = source_offset, .range = source_range},
        {.buffer = destination, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = tasks, .offset = tasks_offset, .range = tasks_range},
    }};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = i,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffers[i],
            .pTexelBufferView = nullptr,
        };
    }
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, writes);
    while (count != 0) {
        const u32 groups = (std::min)(count, MAX_GROUPS_X);
        const PushConstants constants{.first_task = first};
        cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmd.Dispatch(groups, 1, 1);
        first += groups;
        count -= groups;
    }
}

} // namespace Vulkan::Nvn
