// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"
#include "video_core/host_shaders/block_linear_unswizzle_3d_bcn_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_untile_gpu.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

using Tegra::Texture::GOB_SIZE_SHIFT;
using Tegra::Texture::GOB_SIZE_X_SHIFT;
using Tegra::Texture::GOB_SIZE_Y_SHIFT;

/// The push constant block of block_linear_unswizzle_3d_bcn.comp (std430 offsets in the comments;
/// `destination` is declared by the shader but not read).
struct PushConstants {
    std::array<u32, 3> blocks_dim; // 0
    u32 bytes_per_block_log2;      // 12
    std::array<u32, 3> origin;     // 16
    u32 slice_size;                // 28
    u32 block_size;                // 32
    u32 x_shift;                   // 36
    u32 block_height;              // 40
    u32 block_height_mask;         // 44
    u32 block_depth;               // 48
    u32 block_depth_mask;          // 52
    u32 native_tiling;             // 56
    u32 unused_60;
    std::array<s32, 3> destination; // 64
    u32 unused_76;
};
static_assert(sizeof(PushConstants) == 80);
static_assert(offsetof(PushConstants, native_tiling) == 56);
static_assert(offsetof(PushConstants, destination) == 64);

constexpr u32 BINDING_INPUT = 0;
constexpr u32 BINDING_OUTPUT = 1;

/// The shader's workgroup (local_size 8 x 8 x 4).
constexpr u32 GROUP_X = 8;
constexpr u32 GROUP_Y = 8;
constexpr u32 GROUP_Z = 4;

u64 AlignUpLog2(u64 value, u32 shift) {
    const u64 mask = (u64{1} << shift) - 1;
    return (value + mask) & ~mask;
}

u64 Align(u64 value, u64 alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // Anonymous namespace

bool UntileGpu::Supported(const Device& device) {
    static const bool disabled = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_NO_UNTILE_GPU");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return !disabled && device.IsKhrPushDescriptorSupported() && device.MaxPushDescriptors() >= 2 &&
           device.IsStorageBuffer8BitAccessSupported() &&
           device.IsStorageBuffer16BitAccessSupported();
}

u64 UntileGpu::InputSize(const Piece& piece) {
    const u64 row_bytes = u64{piece.columns} * piece.bytes_per_block;
    return AlignUpLog2(row_bytes, GOB_SIZE_X_SHIFT) *
           AlignUpLog2(piece.rows, GOB_SIZE_Y_SHIFT + piece.tile_height) *
           AlignUpLog2(piece.slices, piece.tile_depth);
}

u64 UntileGpu::OutputSize(const Piece& piece) {
    return u64{piece.columns} * piece.bytes_per_block * piece.rows * piece.slices;
}

bool UntileGpu::Fits(const Piece& piece) {
    if (piece.columns == 0 || piece.rows == 0 || piece.slices == 0 || piece.bytes_per_block == 0 ||
        piece.bytes_per_block > 16 || piece.tile_height > 5 || piece.tile_depth > 5) {
        return false;
    }
    // Byte offsets into the input are 32-bit, and so are the output indices.
    constexpr u64 LIMIT = u64{1} << 32;
    if (InputSize(piece) >= LIMIT || OutputSize(piece) >= LIMIT) {
        return false;
    }
    // Group counts within the minimum maxComputeWorkGroupCount every device has.
    constexpr u32 MAX_GROUPS = 65535;
    const Params params = MakeParams(piece);
    return (params.elements_x + GROUP_X - 1) / GROUP_X <= MAX_GROUPS &&
           (params.elements_y + GROUP_Y - 1) / GROUP_Y <= MAX_GROUPS &&
           (params.elements_z + GROUP_Z - 1) / GROUP_Z <= MAX_GROUPS;
}

UntileGpu::Params UntileGpu::MakeParams(const Piece& piece) {
    const u32 row_bytes = piece.columns * piece.bytes_per_block;
    const u32 element_log2 = (std::min)(4u, static_cast<u32>(std::countr_zero(row_bytes)));
    const u32 gobs_in_x = (row_bytes + (1u << GOB_SIZE_X_SHIFT) - 1) >> GOB_SIZE_X_SHIFT;
    const u32 x_shift = GOB_SIZE_SHIFT + piece.tile_height + piece.tile_depth;
    const u32 block_size = gobs_in_x << x_shift;
    const u32 tile_rows =
        static_cast<u32>(AlignUpLog2(piece.rows, GOB_SIZE_Y_SHIFT + piece.tile_height) >>
                         (GOB_SIZE_Y_SHIFT + piece.tile_height));
    return Params{
        .element_log2 = element_log2,
        .elements_x = row_bytes >> element_log2,
        .elements_y = piece.rows,
        .elements_z = piece.slices,
        .slice_size = tile_rows * block_size,
        .block_size = block_size,
        .x_shift = x_shift,
    };
}

UntileGpu::UntileGpu(const Device& device_) : device{device_} {
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
    const std::span<const u32> code = BLOCK_LINEAR_UNSWIZZLE_3D_BCN_COMP_SPV;
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
        // Built on a worker while the replay creates its own pipelines: the externally
        // synchronized shared cache is not used (see AstcGpuDecoder).
        VK_NULL_HANDLE);
}

UntileGpu::~UntileGpu() = default;

void UntileGpu::Record(vk::CommandBuffer cmd, VkBuffer input, VkBuffer output,
                       std::span<const Piece> pieces) const {
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    for (const Piece& piece : pieces) {
        const Params params = MakeParams(piece);
        // The u128 / u32 views of the bindings cover whole elements.
        const VkDescriptorBufferInfo input_info{
            .buffer = input,
            .offset = piece.input_offset,
            .range = Align(InputSize(piece), 16),
        };
        const VkDescriptorBufferInfo output_info{
            .buffer = output,
            .offset = piece.output_offset,
            .range = Align(OutputSize(piece), 4),
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
                .pBufferInfo = &input_info,
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
                .pBufferInfo = &output_info,
                .pTexelBufferView = nullptr,
            },
        }};
        cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, writes);
        const PushConstants constants{
            .blocks_dim = {params.elements_x, params.elements_y, params.elements_z},
            .bytes_per_block_log2 = params.element_log2,
            .origin = {0, 0, 0},
            .slice_size = params.slice_size,
            .block_size = params.block_size,
            .x_shift = params.x_shift,
            .block_height = piece.tile_height,
            .block_height_mask = (1u << piece.tile_height) - 1u,
            .block_depth = piece.tile_depth,
            .block_depth_mask = (1u << piece.tile_depth) - 1u,
            .native_tiling = 1,
            .unused_60 = 0,
            .destination = {0, 0, 0},
            .unused_76 = 0,
        };
        cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmd.Dispatch((params.elements_x + GROUP_X - 1) / GROUP_X,
                     (params.elements_y + GROUP_Y - 1) / GROUP_Y,
                     (params.elements_z + GROUP_Z - 1) / GROUP_Z);
    }
}

} // namespace Vulkan::Nvn
