// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <span>

#include "common/common_types.h"
#include "video_core/host_shaders/nvn_alias_transfer_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_alias_gpu.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

/// The push constant block of nvn_alias_transfer.comp.
struct PushConstants {
    u32 mode;
    u32 tight_offset;
    u32 window_offset;
    u32 window_size;
    s32 piece_base;
    u32 cols;
    u32 rows;
    u32 slices;
    u32 bytes_per_texel;
    u32 tile_height;
    u32 tile_depth;
    u32 pitch;
};
static_assert(sizeof(PushConstants) == 48);

constexpr u32 WORKGROUP_SIZE = 256;
constexpr u32 MAX_GROUPS_X = 65535;

} // Anonymous namespace

bool AliasTransferGpu::Supported(const Device& device) {
    return device.IsKhrPushDescriptorSupported() && device.MaxPushDescriptors() >= 1 &&
           device.IsStorageBuffer8BitAccessSupported();
}

AliasTransferGpu::AliasTransferGpu(const Device& device_) : device{device_} {
    // The shader addresses the scratch buffer with 32-bit offsets.
    max_scratch = (std::min<u64>)(device.GetMaxStorageBufferRange(), u64{1} << 30);
    const vk::Device& logical = device.GetLogical();
    const VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    set_layout = logical.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = 1,
        .pBindings = &binding,
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
    const std::span<const u32> code = NVN_ALIAS_TRANSFER_COMP_SPV;
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

AliasTransferGpu::~AliasTransferGpu() = default;

void AliasTransferGpu::Begin(vk::CommandBuffer cmd, VkBuffer scratch, u64 size) const {
    const VkDescriptorBufferInfo buffer_info{
        .buffer = scratch,
        .offset = 0,
        .range = (std::min)(size, max_scratch),
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &buffer_info,
        .pTexelBufferView = nullptr,
    };
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, write);
}

void AliasTransferGpu::Dispatch(vk::CommandBuffer cmd, Mode mode, u32 window_offset,
                                u32 window_size, const Piece& piece) const {
    const PushConstants constants{
        .mode = static_cast<u32>(mode),
        .tight_offset = piece.tight_offset,
        .window_offset = window_offset,
        .window_size = window_size,
        .piece_base = piece.piece_base,
        .cols = piece.cols,
        .rows = piece.rows,
        .slices = piece.slices,
        .bytes_per_texel = piece.bytes_per_texel,
        .tile_height = piece.tile_height,
        .tile_depth = piece.tile_depth,
        .pitch = piece.pitch,
    };
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    const u64 texels = static_cast<u64>(piece.cols) * piece.rows * piece.slices;
    const u64 groups = (texels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    cmd.Dispatch(static_cast<u32>((std::min<u64>)(groups, MAX_GROUPS_X)),
                 static_cast<u32>((groups + MAX_GROUPS_X - 1) / MAX_GROUPS_X), 1);
}

} // namespace Vulkan::Nvn
