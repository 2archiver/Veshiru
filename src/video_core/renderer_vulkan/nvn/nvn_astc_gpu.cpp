// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"
#include "video_core/host_shaders/astc_decoder_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_astc_gpu.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

/// The push constant block of astc_decoder.comp.
struct PushConstants {
    std::array<u32, 2> block_dims;
    u32 layer_stride;
    u32 block_size;
    u32 x_shift;
    u32 block_height;
    u32 block_height_mask;
    u32 native_tiling;
};
static_assert(sizeof(PushConstants) == 32);

constexpr u32 BINDING_INPUT = 0;
constexpr u32 BINDING_OUTPUT = 1;

/// ASTC blocks are 16 bytes; the stride of a row of blocks decides how many GOBs a tile row has.
constexpr u32 ASTC_BLOCK_BYTES = 16;

} // Anonymous namespace

bool AstcGpuDecoder::Supported(const Device& device) {
    static const bool disabled = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_NO_ASTC_GPU");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    if (disabled || !device.IsKhrPushDescriptorSupported() || device.MaxPushDescriptors() < 2) {
        return false;
    }
    return device.IsFormatSupported(VIEW_FORMAT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                    FormatType::Optimal);
}

AstcGpuDecoder::AstcGpuDecoder(const Device& device_) : device{device_} {
    offset_alignment = (std::max)(device.GetStorageBufferAlignment(), u64{16});
    max_range = device.GetMaxStorageBufferRange();
    const vk::Device& logical = device.GetLogical();
    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
        {
            .binding = BINDING_INPUT,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = BINDING_OUTPUT,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
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
    const std::span<const u32> code = ASTC_DECODER_COMP_SPV;
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
        // The decoder is compiled on NvnAstcInit while the replay and present threads create their
        // own pipelines. VkPipelineCache is externally synchronized, so the shared static cache
        // cannot be used here without serializing all of them. This one small pipeline does not
        // justify that lock (or its contention with the game's pipeline preloading).
        VK_NULL_HANDLE);
}

AstcGpuDecoder::~AstcGpuDecoder() = default;

std::vector<vk::ImageView> AstcGpuDecoder::Record(vk::CommandBuffer cmd, VkBuffer input,
                                                  u64 input_offset, u64 input_size, VkImage image,
                                                  std::span<const Level> levels) const {
    static constexpr VkImageViewUsageCreateInfo STORAGE_USAGE{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT,
    };
    std::vector<vk::ImageView> views;
    views.reserve(levels.size());
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    for (const Level& level : levels) {
        views.push_back(device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &STORAGE_USAGE,
            .flags = 0,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = VIEW_FORMAT,
            .components{
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = level.level,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        }));
        const u64 binding_offset = input_offset + level.offset;
        const VkDescriptorBufferInfo buffer_info{
            .buffer = input,
            .offset = binding_offset,
            .range = (std::min)(input_size - level.offset, max_range),
        };
        const VkDescriptorImageInfo image_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = *views.back(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const std::array<VkWriteDescriptorSet, 2> writes{{
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = BINDING_INPUT,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &buffer_info,
                .pTexelBufferView = nullptr,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = BINDING_OUTPUT,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &image_info,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
        }};
        cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, writes);
        // The shader's addressing is block_linear_unswizzle_2d's: a row of blocks is `stride`
        // bytes (rounded up to whole 64-byte GOB columns), a tile is 2^tile_height GOBs tall.
        const u32 stride = level.columns * ASTC_BLOCK_BYTES;
        const u32 gobs_in_x = (stride + Tegra::Texture::GOB_SIZE_X - 1) >> Tegra::Texture::GOB_SIZE_X_SHIFT;
        const u32 x_shift = Tegra::Texture::GOB_SIZE_SHIFT + level.tile_height + level.tile_depth;
        const PushConstants constants{
            .block_dims = {level.block_width, level.block_height},
            .layer_stride = level.layer_stride,
            .block_size = gobs_in_x << x_shift,
            .x_shift = x_shift,
            .block_height = level.tile_height,
            .block_height_mask = (1u << level.tile_height) - 1u,
            .native_tiling = 1,
        };
        cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmd.Dispatch((level.columns + 7) / 8, (level.rows + 7) / 8, level.layers);
    }
    return views;
}

} // namespace Vulkan::Nvn
