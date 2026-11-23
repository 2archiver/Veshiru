// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdint>

#include "video_core/host_shaders/bc7_encoder_comp_spv.h"
#include "video_core/host_shaders/bc_encoder_comp_spv.h"
#include "video_core/host_shaders/nvn_bc_copy_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_bc_encode.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan::Nvn {
namespace {

/// The push constant block of bc_encoder.comp.
struct PushConstants {
    u32 blocks_x;
    u32 blocks_y;
    u32 width;
    u32 height;
    u32 mode;       ///< BcBlocks, | MODE_REFINE
    u32 out_offset; ///< in 8-byte units
};
static_assert(sizeof(PushConstants) == 24);

/// bc_encoder.comp's REFINE bit of `mode`.
constexpr u32 MODE_REFINE = 16;

constexpr u32 BINDING_SOURCE = 0;
constexpr u32 BINDING_OUTPUT = 1;

/// The push constant block of bc7_encoder.comp.
struct Bc7PushConstants {
    u32 blocks_x;
    u32 blocks_y;
    u32 width;
    u32 height;
    u32 mode;
};
static_assert(sizeof(Bc7PushConstants) == 20);

constexpr u32 BC7_BINDING_SOURCE = 0;
constexpr u32 BC7_BINDING_MODES = 1;
constexpr u32 BC7_BINDING_OUTPUT = 2;

/// bc7_encoder.comp's STAGE specialization constant: modes 4-6, one partitioned mode, encode.
constexpr u32 BC7_STAGE_456 = 0;
constexpr u32 BC7_STAGE_PARTITIONS = 1;
constexpr u32 BC7_STAGE_ENCODE = 2;

/// Blocks per workgroup of each stage (64 invocations: 16, 64 and 1 per block).
constexpr std::array<u32, 3> BC7_BLOCKS_PER_GROUP{4, 1, 64};

} // Anonymous namespace

u32 ParseBc7Quality(const char* text) {
    if (text != nullptr && text[0] >= '0' && text[0] <= '0' + static_cast<char>(BC7_QUALITY_MAX) &&
        text[1] == '\0') {
        return static_cast<u32>(text[0] - '0');
    }
    return BC7_QUALITY_DEFAULT;
}

void BcGpuEncoder::RecordCopy(vk::CommandBuffer cmd, VkImageView source, VkImageView destination,
                              VkImageView scratch, std::array<s32, 2> source_origin,
                              std::array<s32, 2> destination_origin,
                              std::array<s32, 2> block_origin, std::array<s32, 2> copy_size,
                              std::array<s32, 2> scratch_size, u32 layers) {
    struct CopyConstants {
        std::array<s32, 2> source, destination, block, size, scratch;
    };
    const auto& logical = device.GetLogical();
    if (!copy_pipeline) {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (u32 i = 0; i < bindings.size(); ++i) {
            bindings[i] = {.binding = i,
                           .descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                    : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                           .descriptorCount = 1,
                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
        }
        copy_set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CopyConstants)};
        copy_pipeline_layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = copy_set_layout.address(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
        copy_module = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(NVN_BC_COPY_COMP_SPV),
            .pCode = NVN_BC_COPY_COMP_SPV,
        });
        copy_pipeline = logical.CreateComputePipeline(
            {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                          .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                          .module = *copy_module,
                          .pName = "main"},
                .layout = *copy_pipeline_layout,
            },
            VK_NULL_HANDLE);
    }
    const std::array<VkDescriptorImageInfo, 3> images{{
        {VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, destination, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, scratch, VK_IMAGE_LAYOUT_GENERAL},
    }};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     .dstBinding = i,
                     .descriptorCount = 1,
                     .descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                              : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                     .pImageInfo = &images[i]};
    }
    const CopyConstants constants{source_origin, destination_origin, block_origin, copy_size,
                                  scratch_size};
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *copy_pipeline);
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *copy_pipeline_layout, 0, writes);
    cmd.PushConstants(*copy_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    cmd.Dispatch((static_cast<u32>(scratch_size[0]) + 7) / 8,
                 (static_cast<u32>(scratch_size[1]) + 7) / 8, layers);
}

BcBlocks BcBlocksOf(AstcStandin standin, bool opaque) {
    switch (standin) {
    case AstcStandin::Bc1:
        return opaque ? BcBlocks::Bc1Opaque : BcBlocks::Bc1;
    case AstcStandin::Bc4:
        return BcBlocks::Bc4;
    case AstcStandin::Bc5:
        return BcBlocks::Bc5;
    default:
        return BcBlocks::Bc3;
    }
}

bool BcGpuEncoder::SupportsBc45(const Device& device) {
    constexpr VkFormatFeatureFlags BC_FEATURES =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return Supported(device) &&
           device.IsFormatSupported(VK_FORMAT_BC4_UNORM_BLOCK, BC_FEATURES, FormatType::Optimal) &&
           device.IsFormatSupported(VK_FORMAT_BC5_UNORM_BLOCK, BC_FEATURES, FormatType::Optimal);
}

bool BcGpuEncoder::SupportsBc7(const Device& device) {
    constexpr VkFormatFeatureFlags BC_FEATURES =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return Supported(device) && device.MaxPushDescriptors() >= 3 &&
           device.IsFormatSupported(VK_FORMAT_BC7_UNORM_BLOCK, BC_FEATURES, FormatType::Optimal) &&
           device.IsFormatSupported(VK_FORMAT_BC7_SRGB_BLOCK, BC_FEATURES, FormatType::Optimal);
}

bool BcGpuEncoder::Supported(const Device& device) {
    if (!device.IsKhrPushDescriptorSupported() || device.MaxPushDescriptors() < 2 ||
        !device.IsOptimalBcnSupported()) {
        return false;
    }
    constexpr VkFormatFeatureFlags BC_FEATURES =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return device.IsFormatSupported(SOURCE_FORMAT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                    FormatType::Optimal) &&
           device.IsFormatSupported(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, BC_FEATURES,
                                    FormatType::Optimal) &&
           device.IsFormatSupported(VK_FORMAT_BC3_UNORM_BLOCK, BC_FEATURES, FormatType::Optimal);
}

BcGpuEncoder::BcGpuEncoder(const Device& device_, bool bc7, std::span<const u32> bc_code)
    : device{device_} {
    const vk::Device& logical = device.GetLogical();
    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
        {
            .binding = BINDING_SOURCE,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = BINDING_OUTPUT,
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
    const std::span<const u32> code =
        bc_code.empty() ? std::span<const u32>{BC_ENCODER_COMP_SPV} : bc_code;
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
        // Compiled on a worker like the ASTC decoder: no shared, externally synchronized cache.
        VK_NULL_HANDLE);
    if (!bc7) {
        return;
    }

    std::array<VkDescriptorSetLayoutBinding, 3> bc7_bindings{};
    for (u32 binding = 0; binding < bc7_bindings.size(); ++binding) {
        bc7_bindings[binding] = {
            .binding = binding,
            .descriptorType = binding == BC7_BINDING_SOURCE ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        };
    }
    bc7_set_layout = logical.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = static_cast<u32>(bc7_bindings.size()),
        .pBindings = bc7_bindings.data(),
    });
    const VkPushConstantRange bc7_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(Bc7PushConstants),
    };
    bc7_pipeline_layout = logical.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = bc7_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &bc7_push_range,
    });
    const std::span<const u32> bc7_code = BC7_ENCODER_COMP_SPV;
    bc7_module = logical.CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = static_cast<u32>(bc7_code.size_bytes()),
        .pCode = bc7_code.data(),
    });
    for (u32 stage = 0; stage < bc7_pipelines.size(); ++stage) {
        const VkSpecializationMapEntry entry{
            .constantID = 0,
            .offset = 0,
            .size = sizeof(u32),
        };
        const VkSpecializationInfo specialization{
            .mapEntryCount = 1,
            .pMapEntries = &entry,
            .dataSize = sizeof(u32),
            .pData = &stage,
        };
        bc7_pipelines[stage] = logical.CreateComputePipeline(
            VkComputePipelineCreateInfo{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = *bc7_module,
                    .pName = "main",
                    .pSpecializationInfo = &specialization,
                },
                .layout = *bc7_pipeline_layout,
                .basePipelineHandle = {},
                .basePipelineIndex = 0,
            },
            VK_NULL_HANDLE);
    }
}

BcGpuEncoder::~BcGpuEncoder() = default;

void BcGpuEncoder::RecordBc7(vk::CommandBuffer cmd, VkImageView source, u32 width, u32 height,
                             u32 layers, u32 quality, VkBuffer modes, u64 modes_size,
                             VkBuffer output, u64 output_size) const {
    const VkDescriptorImageInfo image_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = source,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorBufferInfo modes_info{
        .buffer = modes,
        .offset = 0,
        .range = modes_size,
    };
    const VkDescriptorBufferInfo output_info{
        .buffer = output,
        .offset = 0,
        .range = output_size,
    };
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (u32 binding = 0; binding < writes.size(); ++binding) {
        writes[binding] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = binding == BC7_BINDING_SOURCE ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = binding == BC7_BINDING_SOURCE ? &image_info : nullptr,
            .pBufferInfo = binding == BC7_BINDING_MODES    ? &modes_info
                           : binding == BC7_BINDING_OUTPUT ? &output_info
                                                           : nullptr,
            .pTexelBufferView = nullptr,
        };
    }
    Bc7PushConstants constants{
        .blocks_x = (width + 3) / 4,
        .blocks_y = (height + 3) / 4,
        .width = width,
        .height = height,
        .mode = 0,
    };
    // Each pass reads and rewrites the modes the previous one left.
    const VkMemoryBarrier modes_written{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    const auto dispatch = [&](u32 stage) {
        const u32 per_group = BC7_BLOCKS_PER_GROUP[stage];
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *bc7_pipelines[stage]);
        cmd.PushConstants(*bc7_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmd.Dispatch((constants.blocks_x + per_group - 1) / per_group, constants.blocks_y, layers);
    };
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *bc7_pipelines[BC7_STAGE_456]);
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *bc7_pipeline_layout, 0, writes);
    dispatch(BC7_STAGE_456);
    // DirectXTex's order: 1, 3, 7, then the rarely chosen and slower three-subset modes.
    constexpr std::array<u32, 5> PARTITIONED_MODES{1, 3, 7, 0, 2};
    const u32 partitioned = quality >= 2 ? 5 : (quality >= 1 ? 3 : 0);
    for (u32 i = 0; i < partitioned; ++i) {
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, modes_written);
        constants.mode = PARTITIONED_MODES[i];
        dispatch(BC7_STAGE_PARTITIONS);
    }
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, modes_written);
    constants.mode = 0;
    dispatch(BC7_STAGE_ENCODE);
}

void BcGpuEncoder::Record(vk::CommandBuffer cmd, VkImageView source, u32 width, u32 height,
                          u32 layers, BcBlocks blocks, bool refine, VkBuffer output,
                          u64 output_size) const {
    const VkDescriptorImageInfo image_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = source,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorBufferInfo buffer_info{
        .buffer = output,
        .offset = 0,
        .range = output_size,
    };
    const std::array<VkWriteDescriptorSet, 2> writes{{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = BINDING_SOURCE,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = BINDING_OUTPUT,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_info,
            .pTexelBufferView = nullptr,
        },
    }};
    const PushConstants constants{
        .blocks_x = (width + 3) / 4,
        .blocks_y = (height + 3) / 4,
        .width = width,
        .height = height,
        .mode = static_cast<u32>(blocks) | (refine ? MODE_REFINE : 0u),
        .out_offset = 0,
    };
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, writes);
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    cmd.Dispatch((constants.blocks_x + 7) / 8, (constants.blocks_y + 7) / 8, layers);
}

std::shared_ptr<BcScratch> MakeBcScratch(const Device& device, const MemoryAllocator& allocator,
                                         u32 width, u32 height, u32 layers, bool bc7) {
    auto scratch = std::make_shared<BcScratch>();
    scratch->width = width;
    scratch->height = height;
    scratch->layers = layers;
    scratch->image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = BcGpuEncoder::SOURCE_FORMAT,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    scratch->view = device.GetLogical().CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *scratch->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = BcGpuEncoder::SOURCE_FORMAT,
        .components{
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = layers,
        },
    });
    scratch->blocks = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = BcBlocksSize(width, height, layers, 16),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::DeviceLocal);
    if (bc7) {
        scratch->modes = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = BcBlocksSize(width, height, layers, 8),
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::DeviceLocal);
    }
    return scratch;
}

std::vector<BcChunk> SplitBcLayers(u32 level, u32 first_layer, u32 layers, u32 width, u32 height,
                                   u32 x, u32 y) {
    const u64 layer_bytes = (std::max)(u64{width} * height * 4, u64{1});
    const u32 per_chunk = static_cast<u32>(
        std::clamp<u64>(BC_SCRATCH_BUDGET / layer_bytes, 1, (std::max)(layers, 1u)));
    std::vector<BcChunk> chunks;
    for (u32 layer = 0; layer < layers; layer += per_chunk) {
        chunks.push_back({
            .level = level,
            .first_layer = first_layer + layer,
            .layers = (std::min)(per_chunk, layers - layer),
            .width = width,
            .height = height,
            .x = x,
            .y = y,
        });
    }
    return chunks;
}

} // namespace Vulkan::Nvn
