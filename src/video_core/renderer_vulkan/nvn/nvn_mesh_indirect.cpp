// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#include "video_core/host_shaders/nvn_mesh_indirect_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_mesh_indirect.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

/// The push constant block of nvn_mesh_indirect.comp.
struct PushConstants {
    u32 first_word;
    u32 stride_words;
    u32 draws;
};
static_assert(sizeof(PushConstants) == 12);

constexpr u32 WORKGROUP_SIZE = 64;

} // Anonymous namespace

bool MeshIndirectGpu::Supported(const Device& device) {
    return device.IsKhrPushDescriptorSupported() && device.MaxPushDescriptors() >= 2;
}

MeshIndirectGpu::MeshIndirectGpu(const Device& device_) : device{device_} {
    const vk::Device& logical = device.GetLogical();
    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
    }};
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
    const std::span<const u32> code = NVN_MESH_INDIRECT_COMP_SPV;
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

MeshIndirectGpu::~MeshIndirectGpu() = default;

void MeshIndirectGpu::Convert(vk::CommandBuffer cmd, VkBuffer source, u64 source_offset,
                              u32 stride, u32 draws, VkBuffer destination,
                              u64 destination_offset) const {
    // The source descriptor starts at the aligned offset below the first record; the shader
    // indexes the remainder in words.
    const u64 alignment = (std::max<u64>)(device.GetStorageBufferAlignment(), 4);
    const u64 source_base = source_offset / alignment * alignment;
    const u64 source_range =
        source_offset - source_base + static_cast<u64>(draws - 1) * stride +
        MESH_TASKS_1D_COMMAND_SIZE;
    const std::array<VkDescriptorBufferInfo, 2> buffers{{
        {.buffer = source, .offset = source_base, .range = source_range},
        {.buffer = destination,
         .offset = destination_offset,
         .range = static_cast<u64>(draws) * MESH_TASKS_3D_COMMAND_SIZE},
    }};
    const std::array<VkWriteDescriptorSet, 2> writes{{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffers[0],
            .pTexelBufferView = nullptr,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffers[1],
            .pTexelBufferView = nullptr,
        },
    }};
    const PushConstants constants{
        .first_word = static_cast<u32>((source_offset - source_base) / sizeof(u32)),
        .stride_words = stride / static_cast<u32>(sizeof(u32)),
        .draws = draws,
    };
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, writes);
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    cmd.Dispatch((draws + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
}

} // namespace Vulkan::Nvn
