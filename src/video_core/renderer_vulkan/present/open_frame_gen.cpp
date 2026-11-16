// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright (c) 2026 Prithivi Nair, Prajwal R, Madhav S, Rashmi N Ugarakhod (Open-Frame-gen)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "video_core/renderer_vulkan/present/open_frame_gen.h"
#include "video_core/renderer_vulkan/present/open_frame_gen_weights.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"

#include "video_core/host_shaders/open_frame_gen_dw3x3_comp_spv.h"
#include "video_core/host_shaders/open_frame_gen_head_comp_spv.h"
#include "video_core/host_shaders/open_frame_gen_prep_comp_spv.h"
#include "video_core/host_shaders/open_frame_gen_pw1x1_comp_spv.h"
#include "video_core/host_shaders/open_frame_gen_se_comp_spv.h"
#include "video_core/host_shaders/open_frame_gen_warp_comp_spv.h"

namespace Vulkan {

namespace {

enum StaticSetIndex : u32 {
    SetPrep = 0,
    SetE1Dw,
    SetE1Pw,
    SetE2Dw,
    SetE2Pw,
    SetE3Dw,
    SetE3Pw,
    SetBtnDw,
    SetBtnPw,
    SetSe,
    SetD1Dw,
    SetD1Pw,
    SetD2Dw,
    SetD2Pw,
    SetD3Dw,
    SetD3Pw,
    SetHead0,
    SetHead2,
    StaticSetCount,
};

struct PrepPushConstants {
    u32 out_w;
    u32 out_h;
};

struct DwPushConstants {
    u32 in_w;
    u32 in_h;
    u32 out_w;
    u32 out_h;
    u32 stride;
    u32 mode;
    u32 ch_a;
    u32 w_offset;
    u32 b_offset;
};

struct PwPushConstants {
    u32 width;
    u32 height;
    u32 in_ch;
    u32 out_ch;
    u32 w_offset;
    u32 b_offset;
};

struct SePushConstants {
    u32 plane_size;
    u32 fc0_w_off;
    u32 fc0_b_off;
    u32 fc2_w_off;
    u32 fc2_b_off;
};

struct HeadPushConstants {
    u32 width;
    u32 height;
    u32 mode;
    u32 w_offset;
    u32 b_offset;
    f32 flow_norm_x;
    f32 flow_norm_y;
};

struct WarpPushConstants {
    f32 inv_render_w;
    f32 inv_render_h;
    f32 render_w;
    f32 render_h;
    f32 blend_factor;
    f32 sharpness;
};

constexpr u32 DivUp(u32 a, u32 b) noexcept {
    return (a + b - 1u) / b;
}

VkExtent2D ComputeNanoExtent(VkExtent2D extent, f32 flow_scale) {
    const f32 clamped_scale = std::clamp(flow_scale, 0.25f, 1.0f);
    f32 scaled_w = static_cast<f32>(extent.width) * clamped_scale;
    f32 scaled_h = static_cast<f32>(extent.height) * clamped_scale;
    // Cap NanoModel pyramid base at 1280x720 so inference stays sub-millisecond while full-res
    // frame textures are warped at native output resolution.
    constexpr f32 kMaxW = 1280.0f;
    constexpr f32 kMaxH = 720.0f;
    const f32 cap_ratio = std::min({1.0f, kMaxW / std::max(1.0f, scaled_w),
                                    kMaxH / std::max(1.0f, scaled_h)});
    scaled_w *= cap_ratio;
    scaled_h *= cap_ratio;
    const u32 w = std::max(64u, (static_cast<u32>(std::lround(scaled_w)) + 15u) & ~15u);
    const u32 h = std::max(64u, (static_cast<u32>(std::lround(scaled_h)) + 15u) & ~15u);
    return {w, h};
}

VkImageMemoryBarrier MakeImageBarrier(VkImage image, VkAccessFlags src_access,
                                      VkAccessFlags dst_access, VkImageLayout old_layout,
                                      VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}

void ComputeMemoryBarrier(vk::CommandBuffer cmdbuf) {
    static constexpr VkMemoryBarrier kBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, kBarrier);
}

void CopyImageToTwo(vk::CommandBuffer cmdbuf, VkImage source, VkImage dest_a, VkImage dest_b,
                    VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };
    const std::array before{
        MakeImageBarrier(source,
                         VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeImageBarrier(dest_a, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        MakeImageBarrier(dest_b, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest_a,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest_b,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
    const std::array after{
        MakeImageBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeImageBarrier(dest_a, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeImageBarrier(dest_b, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

void CopyImageToImage(vk::CommandBuffer cmdbuf, VkImage source, VkImage destination,
                      VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };
    const std::array before{
        MakeImageBarrier(source,
                         VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeImageBarrier(destination, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
    const std::array after{
        MakeImageBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeImageBarrier(destination, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

vk::Buffer CreateStorageBuffer(MemoryAllocator& allocator, std::size_t float_count,
                               VkBufferUsageFlags extra_usage = 0) {
    return allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = static_cast<VkDeviceSize>(float_count * sizeof(f32)),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra_usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::DeviceLocal);
}

} // namespace

OpenFrameGen::OpenFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                           VkExtent2D extent, VkFormat format, f32 flow_scale)
    : m_allocator{memory_allocator}, m_extent{extent},
      m_net_extent{ComputeNanoExtent(extent, flow_scale)}, m_format{format},
      m_w2{m_net_extent.width / 2u}, m_h2{m_net_extent.height / 2u},
      m_w4{m_net_extent.width / 4u}, m_h4{m_net_extent.height / 4u},
      m_w8{m_net_extent.width / 8u}, m_h8{m_net_extent.height / 8u},
      m_w16{m_net_extent.width / 16u}, m_h16{m_net_extent.height / 16u} {
    CreateResources(device);
    CreateShaders(device);
    CreatePipelines(device);
    WriteStaticDescriptors(device);
    m_initialized = true;
}

OpenFrameGen::~OpenFrameGen() = default;

void OpenFrameGen::CreateResources(const Device& device) {
    m_linear_sampler = CreateBilinearSampler(device);

    m_history_image = CreateWrappedImage(m_allocator, m_extent, m_format);
    m_history_view = CreateWrappedImageView(device, m_history_image, m_format);

    m_current_image = m_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = m_format,
        .extent = {m_extent.width, m_extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    m_current_view = CreateWrappedImageView(device, m_current_image, m_format);

    m_flow_mask_image = CreateWrappedImage(m_allocator, {m_w4, m_h4}, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_flow_mask_view =
        CreateWrappedImageView(device, m_flow_mask_image, VK_FORMAT_R16G16B16A16_SFLOAT);

    const auto weights = OpenFrameGenWeights::UnpackWeights();
    const VkDeviceSize weights_bytes = weights.size() * sizeof(f32);
    m_weights_upload = m_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = weights_bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        },
        MemoryUsage::Upload);
    std::memcpy(m_weights_upload.Mapped().data(), weights.data(), weights_bytes);
    m_weights_upload.Flush();

    m_weights_buffer =
        CreateStorageBuffer(m_allocator, weights.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    const std::size_t p2 = static_cast<std::size_t>(m_w2) * m_h2;
    const std::size_t p4 = static_cast<std::size_t>(m_w4) * m_h4;
    const std::size_t p8 = static_cast<std::size_t>(m_w8) * m_h8;
    const std::size_t p16 = static_cast<std::size_t>(m_w16) * m_h16;

    m_buf_xs = CreateStorageBuffer(m_allocator, 6u * p2);
    m_buf_s1 = CreateStorageBuffer(m_allocator, 48u * p4);
    m_buf_s2 = CreateStorageBuffer(m_allocator, 96u * p8);
    m_buf_s3 = CreateStorageBuffer(m_allocator, 128u * p16);
    m_buf_scratch_a =
        CreateStorageBuffer(m_allocator, std::max({112u * p4, 192u * p8, 224u * p16}));
    m_buf_scratch_b = CreateStorageBuffer(m_allocator, std::max({32u * p4, 64u * p8, 96u * p16}));

    const std::array pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4},
    };
    m_static_desc_pool = device.GetLogical().CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = StaticSetCount,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
}

void OpenFrameGen::CreateShaders(const Device& device) {
    m_prep_shader = BuildShader(device, OPEN_FRAME_GEN_PREP_COMP_SPV);
    m_dw3x3_shader = BuildShader(device, OPEN_FRAME_GEN_DW3X3_COMP_SPV);
    m_pw1x1_shader = BuildShader(device, OPEN_FRAME_GEN_PW1X1_COMP_SPV);
    m_se_shader = BuildShader(device, OPEN_FRAME_GEN_SE_COMP_SPV);
    m_head_shader = BuildShader(device, OPEN_FRAME_GEN_HEAD_COMP_SPV);
    m_warp_shader = BuildShader(device, OPEN_FRAME_GEN_WARP_COMP_SPV);
}

void OpenFrameGen::CreatePipelines(const Device& device) {
    const auto make_pipe = [&](std::span<const VkDescriptorSetLayoutBinding> bindings,
                               u32 push_size, VkShaderModule shader,
                               vk::DescriptorSetLayout& out_dsl, vk::PipelineLayout& out_pl,
                               vk::Pipeline& out_pipe) {
        out_dsl = device.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_size,
        };
        out_pl = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = out_dsl.address(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        });
        out_pipe = device.GetLogical().CreateComputePipeline(VkComputePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage =
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = shader,
                    .pName = "main",
                },
            .layout = *out_pl,
        });
    };

    const std::array prep_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    make_pipe(prep_bindings, sizeof(PrepPushConstants), *m_prep_shader, m_prep_desc_layout,
              m_prep_pipe_layout, m_prep_pipeline);

    const std::array dw_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    make_pipe(dw_bindings, sizeof(DwPushConstants), *m_dw3x3_shader, m_dw_desc_layout,
              m_dw_pipe_layout, m_dw_pipeline);

    const std::array pw_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    make_pipe(pw_bindings, sizeof(PwPushConstants), *m_pw1x1_shader, m_pw_desc_layout,
              m_pw_pipe_layout, m_pw_pipeline);

    const std::array se_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    make_pipe(se_bindings, sizeof(SePushConstants), *m_se_shader, m_se_desc_layout,
              m_se_pipe_layout, m_se_pipeline);

    const std::array head_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    make_pipe(head_bindings, sizeof(HeadPushConstants), *m_head_shader, m_head_desc_layout,
              m_head_pipe_layout, m_head_pipeline);

    const std::array warp_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    make_pipe(warp_bindings, sizeof(WarpPushConstants), *m_warp_shader, m_warp_desc_layout,
              m_warp_pipe_layout, m_warp_pipeline);

    const std::array<VkDescriptorSetLayout, StaticSetCount> layouts{
        *m_prep_desc_layout, // SetPrep
        *m_dw_desc_layout,   // SetE1Dw
        *m_pw_desc_layout,   // SetE1Pw
        *m_dw_desc_layout,   // SetE2Dw
        *m_pw_desc_layout,   // SetE2Pw
        *m_dw_desc_layout,   // SetE3Dw
        *m_pw_desc_layout,   // SetE3Pw
        *m_dw_desc_layout,   // SetBtnDw
        *m_pw_desc_layout,   // SetBtnPw
        *m_se_desc_layout,   // SetSe
        *m_dw_desc_layout,   // SetD1Dw
        *m_pw_desc_layout,   // SetD1Pw
        *m_dw_desc_layout,   // SetD2Dw
        *m_pw_desc_layout,   // SetD2Pw
        *m_dw_desc_layout,   // SetD3Dw
        *m_pw_desc_layout,   // SetD3Pw
        *m_head_desc_layout, // SetHead0
        *m_head_desc_layout, // SetHead2
    };
    m_static_sets = CreateWrappedDescriptorSets(m_static_desc_pool, layouts);
}

void OpenFrameGen::WriteStaticDescriptors(const Device& device) {
    std::vector<VkDescriptorBufferInfo> buf_infos;
    buf_infos.reserve(64);
    std::vector<VkDescriptorImageInfo> img_infos;
    img_infos.reserve(8);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(64);

    const auto bind_buf = [&](VkDescriptorSet set, u32 binding, VkBuffer buffer) {
        buf_infos.push_back(VkDescriptorBufferInfo{
            .buffer = buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        });
        writes.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = binding,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buf_infos.back(),
        });
    };
    const auto bind_dw = [&](StaticSetIndex idx, VkBuffer a, VkBuffer b, VkBuffer out) {
        bind_buf(m_static_sets[idx], 0, a);
        bind_buf(m_static_sets[idx], 1, b);
        bind_buf(m_static_sets[idx], 2, *m_weights_buffer);
        bind_buf(m_static_sets[idx], 3, out);
    };
    const auto bind_pw = [&](StaticSetIndex idx, VkBuffer in, VkBuffer out) {
        bind_buf(m_static_sets[idx], 0, in);
        bind_buf(m_static_sets[idx], 1, *m_weights_buffer);
        bind_buf(m_static_sets[idx], 2, out);
    };

    // SetPrep: prev_frame (0), curr_frame (1), buf_xs (2)
    writes.push_back(CreateWriteDescriptorSet(img_infos, *m_linear_sampler, *m_history_view,
                                              m_static_sets[SetPrep], 0));
    writes.push_back(CreateWriteDescriptorSet(img_infos, *m_linear_sampler, *m_current_view,
                                              m_static_sets[SetPrep], 1));
    bind_buf(m_static_sets[SetPrep], 2, *m_buf_xs);

    // Encoder + Bottleneck + SE
    bind_dw(SetE1Dw, *m_buf_xs, *m_buf_xs, *m_buf_scratch_a);
    bind_pw(SetE1Pw, *m_buf_scratch_a, *m_buf_s1);

    bind_dw(SetE2Dw, *m_buf_s1, *m_buf_s1, *m_buf_scratch_a);
    bind_pw(SetE2Pw, *m_buf_scratch_a, *m_buf_s2);

    bind_dw(SetE3Dw, *m_buf_s2, *m_buf_s2, *m_buf_scratch_a);
    bind_pw(SetE3Pw, *m_buf_scratch_a, *m_buf_s3);

    bind_dw(SetBtnDw, *m_buf_s3, *m_buf_s3, *m_buf_scratch_a);
    bind_pw(SetBtnPw, *m_buf_scratch_a, *m_buf_scratch_b);

    bind_buf(m_static_sets[SetSe], 0, *m_buf_scratch_b);
    bind_buf(m_static_sets[SetSe], 1, *m_weights_buffer);

    // Decoder with skip connections
    bind_dw(SetD1Dw, *m_buf_scratch_b, *m_buf_s3, *m_buf_scratch_a);
    bind_pw(SetD1Pw, *m_buf_scratch_a, *m_buf_scratch_b);

    bind_dw(SetD2Dw, *m_buf_scratch_b, *m_buf_s2, *m_buf_scratch_a);
    bind_pw(SetD2Pw, *m_buf_scratch_a, *m_buf_scratch_b);

    bind_dw(SetD3Dw, *m_buf_scratch_b, *m_buf_s1, *m_buf_scratch_a);
    bind_pw(SetD3Pw, *m_buf_scratch_a, *m_buf_scratch_b);

    // Head
    img_infos.push_back(VkDescriptorImageInfo{
        .imageView = *m_flow_mask_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    });
    const VkDescriptorImageInfo* flow_store_ptr = &img_infos.back();

    bind_buf(m_static_sets[SetHead0], 0, *m_buf_scratch_b);
    bind_buf(m_static_sets[SetHead0], 1, *m_weights_buffer);
    bind_buf(m_static_sets[SetHead0], 2, *m_buf_scratch_a);
    writes.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = m_static_sets[SetHead0],
        .dstBinding = 3,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = flow_store_ptr,
    });

    bind_buf(m_static_sets[SetHead2], 0, *m_buf_scratch_a);
    bind_buf(m_static_sets[SetHead2], 1, *m_weights_buffer);
    bind_buf(m_static_sets[SetHead2], 2, *m_buf_scratch_b);
    writes.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = m_static_sets[SetHead2],
        .dstBinding = 3,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = flow_store_ptr,
    });

    device.GetLogical().UpdateDescriptorSets(writes, {});
}

void OpenFrameGen::Process(const Device&, Scheduler& scheduler, VkImage current_frame, VkImageView,
                           u64 frame_count) {
    if (frame_count == 0) {
        m_has_history = false;
        m_flow_valid = false;
    }

    if (!m_has_history) {
        const bool need_upload = !m_weights_uploaded;
        m_weights_uploaded = true;
        scheduler.Record([this, current_frame, need_upload](vk::CommandBuffer cmdbuf) {
            if (need_upload) {
                const VkBufferCopy copy{
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = OpenFrameGenWeights::kTotalFloatCount * sizeof(f32),
                };
                cmdbuf.CopyBuffer(*m_weights_upload, *m_weights_buffer, copy);
                static constexpr VkMemoryBarrier kWeightBarrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, kWeightBarrier);
            }
            CopyImageToTwo(cmdbuf, current_frame, *m_history_image, *m_current_image, m_extent);
            const auto discard = MakeImageBarrier(*m_flow_mask_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
                                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                                  VK_IMAGE_LAYOUT_GENERAL);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, discard);
        });
        m_has_history = true;
        m_last_frame_count = frame_count;
        return;
    }

    if (frame_count == m_last_frame_count) {
        return;
    }
    m_last_frame_count = frame_count;

    namespace W = OpenFrameGenWeights;

    scheduler.Record([this, current_frame](vk::CommandBuffer cmdbuf) {
        CopyImageToImage(cmdbuf, *m_current_image, *m_history_image, m_extent);
        CopyImageToImage(cmdbuf, current_frame, *m_current_image, m_extent);

        const auto flow_write_barrier =
            MakeImageBarrier(*m_flow_mask_image,
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, flow_write_barrier);

        const auto run_dw = [&](StaticSetIndex set_idx, u32 in_w, u32 in_h, u32 out_w, u32 out_h,
                                u32 ch, u32 stride, u32 mode, u32 ch_a, u32 w_off, u32 b_off) {
            const DwPushConstants pc{in_w, in_h, out_w, out_h, stride, mode, ch_a, w_off, b_off};
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_dw_pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_dw_pipe_layout, 0,
                                      m_static_sets[set_idx], {});
            cmdbuf.PushConstants(*m_dw_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, pc);
            cmdbuf.Dispatch(DivUp(out_w, 16u), DivUp(out_h, 16u), ch);
            ComputeMemoryBarrier(cmdbuf);
        };
        const auto run_pw = [&](StaticSetIndex set_idx, u32 w, u32 h, u32 in_ch, u32 out_ch,
                                u32 w_off, u32 b_off) {
            const PwPushConstants pc{w, h, in_ch, out_ch, w_off, b_off};
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_pw_pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_pw_pipe_layout, 0,
                                      m_static_sets[set_idx], {});
            cmdbuf.PushConstants(*m_pw_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, pc);
            cmdbuf.Dispatch(DivUp(w, 8u), DivUp(h, 8u), DivUp(out_ch, 4u));
            ComputeMemoryBarrier(cmdbuf);
        };

        // 1. Prep: avg_pool2d(x, 2) -> [6, m_h2, m_w2]
        const PrepPushConstants prep_pc{m_w2, m_h2};
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_prep_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_prep_pipe_layout, 0,
                                  m_static_sets[SetPrep], {});
        cmdbuf.PushConstants(*m_prep_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, prep_pc);
        cmdbuf.Dispatch(DivUp(m_w2, 16u), DivUp(m_h2, 16u), 1u);
        ComputeMemoryBarrier(cmdbuf);

        // 2. Encoder (e1 -> e2 -> e3)
        run_dw(SetE1Dw, m_w2, m_h2, m_w4, m_h4, 6u, 2u, 0u, 0u, W::kE1DwW, W::kE1DwB);
        run_pw(SetE1Pw, m_w4, m_h4, 6u, 48u, W::kE1PwW, W::kE1PwB);

        run_dw(SetE2Dw, m_w4, m_h4, m_w8, m_h8, 48u, 2u, 0u, 0u, W::kE2DwW, W::kE2DwB);
        run_pw(SetE2Pw, m_w8, m_h8, 48u, 96u, W::kE2PwW, W::kE2PwB);

        run_dw(SetE3Dw, m_w8, m_h8, m_w16, m_h16, 96u, 2u, 0u, 0u, W::kE3DwW, W::kE3DwB);
        run_pw(SetE3Pw, m_w16, m_h16, 96u, 128u, W::kE3PwW, W::kE3PwB);

        // 3. Bottleneck + SEBlock(96, r=8)
        run_dw(SetBtnDw, m_w16, m_h16, m_w16, m_h16, 128u, 1u, 0u, 0u, W::kBtnDwW, W::kBtnDwB);
        run_pw(SetBtnPw, m_w16, m_h16, 128u, 96u, W::kBtnPwW, W::kBtnPwB);

        const SePushConstants se_pc{m_w16 * m_h16, W::kSeFc0W, W::kSeFc0B, W::kSeFc2W, W::kSeFc2B};
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_se_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_se_pipe_layout, 0,
                                  m_static_sets[SetSe], {});
        cmdbuf.PushConstants(*m_se_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, se_pc);
        cmdbuf.Dispatch(1u, 1u, 1u);
        ComputeMemoryBarrier(cmdbuf);

        // 4. Decoder (d1 -> d2 -> d3)
        run_dw(SetD1Dw, m_w16, m_h16, m_w16, m_h16, 224u, 1u, 1u, 96u, W::kD1DwW, W::kD1DwB);
        run_pw(SetD1Pw, m_w16, m_h16, 224u, 96u, W::kD1PwW, W::kD1PwB);

        run_dw(SetD2Dw, m_w8, m_h8, m_w8, m_h8, 192u, 1u, 2u, 96u, W::kD2DwW, W::kD2DwB);
        run_pw(SetD2Pw, m_w8, m_h8, 192u, 64u, W::kD2PwW, W::kD2PwB);

        run_dw(SetD3Dw, m_w4, m_h4, m_w4, m_h4, 112u, 1u, 2u, 64u, W::kD3DwW, W::kD3DwB);
        run_pw(SetD3Pw, m_w4, m_h4, 112u, 32u, W::kD3PwW, W::kD3PwB);

        // 5. Head (32 -> 32 -> (flow_uv_x, flow_uv_y, sigmoid(mask)))
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_head_pipeline);
        const HeadPushConstants head0_pc{m_w4, m_h4, 0u, W::kHead0W, W::kHead0B, 0.0f, 0.0f};
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_head_pipe_layout, 0,
                                  m_static_sets[SetHead0], {});
        cmdbuf.PushConstants(*m_head_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, head0_pc);
        cmdbuf.Dispatch(DivUp(m_w4, 8u), DivUp(m_h4, 8u), 32u);
        ComputeMemoryBarrier(cmdbuf);

        const f32 flow_norm_x = 4.0f / static_cast<f32>(std::max(1u, m_net_extent.width - 1u));
        const f32 flow_norm_y = 4.0f / static_cast<f32>(std::max(1u, m_net_extent.height - 1u));
        const HeadPushConstants head2_pc{m_w4,        m_h4,        1u,         W::kHead2W,
                                         W::kHead2B, flow_norm_x, flow_norm_y};
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_head_pipe_layout, 0,
                                  m_static_sets[SetHead2], {});
        cmdbuf.PushConstants(*m_head_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, head2_pc);
        cmdbuf.Dispatch(DivUp(m_w4, 8u), DivUp(m_h4, 8u), 1u);

        const auto flow_read_barrier = MakeImageBarrier(
            *m_flow_mask_image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, flow_read_barrier);
    });

    m_flow_valid = true;
}

void OpenFrameGen::Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                            VkImageView destination_storage_view, float blend_factor) {
    if (!m_has_history || !m_flow_valid) {
        return;
    }

    if (!m_warp_pool) {
        m_warp_pool.emplace(device, scheduler);
        m_warp_allocator = m_warp_pool->Allocator(device, scheduler, *m_warp_desc_layout,
                                                  DescriptorBankInfo{.textures = 3, .images = 1});
    }
    const VkDescriptorSet set = m_warp_allocator.Commit();
    std::vector<VkDescriptorImageInfo> img_infos;
    img_infos.reserve(4);
    std::vector<VkWriteDescriptorSet> updates{
        CreateWriteDescriptorSet(img_infos, *m_linear_sampler, *m_history_view, set, 0),
        CreateWriteDescriptorSet(img_infos, *m_linear_sampler, *m_current_view, set, 1),
        CreateWriteDescriptorSet(img_infos, *m_linear_sampler, *m_flow_mask_view, set, 2),
    };
    const VkDescriptorImageInfo dest_info{
        .imageView = destination_storage_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    updates.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = 3,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dest_info,
    });
    device.GetLogical().UpdateDescriptorSets(updates, {});

    const u32 out_w = m_extent.width;
    const u32 out_h = m_extent.height;
    const WarpPushConstants pc{
        .inv_render_w = 1.0f / static_cast<f32>(out_w),
        .inv_render_h = 1.0f / static_cast<f32>(out_h),
        .render_w = static_cast<f32>(out_w),
        .render_h = static_cast<f32>(out_h),
        .blend_factor = blend_factor,
        .sharpness = 0.35f,
    };

    scheduler.Record([this, pc, out_w, out_h, set, destination](vk::CommandBuffer cmdbuf) {
        const auto discard =
            MakeImageBarrier(destination, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, discard);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_warp_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_warp_pipe_layout, 0, set, {});
        cmdbuf.PushConstants(*m_warp_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, pc);
        cmdbuf.Dispatch(DivUp(out_w, 16u), DivUp(out_h, 16u), 1u);
    });
}

} // namespace Vulkan
