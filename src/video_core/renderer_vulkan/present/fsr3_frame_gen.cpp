// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>

#include "video_core/renderer_vulkan/present/fsr3_frame_gen.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"

#include "video_core/host_shaders/fsr3_frame_interpolation_comp_spv.h"
#include "video_core/host_shaders/fsr3_optical_flow_comp_spv.h"

namespace Vulkan {

namespace {

VkExtent2D ComputeFlowExtent(VkExtent2D extent, f32 flow_scale) {
    // Run optical flow at quarter-to-half resolution capped at 640x360 so high-resolution docked
    // outputs (1080p/1440p/4K) do not spend excessive GPU bandwidth searching 1280x720+ grids.
    const f32 scale = std::clamp(flow_scale * 0.5f, 0.25f, 0.5f);
    f32 w = static_cast<f32>(extent.width) * scale;
    f32 h = static_cast<f32>(extent.height) * scale;
    constexpr f32 kMaxW = 640.0f;
    constexpr f32 kMaxH = 360.0f;
    const f32 cap = std::min({1.0f, kMaxW / std::max(1.0f, w), kMaxH / std::max(1.0f, h)});
    w *= cap;
    h *= cap;
    return {
        .width = (std::max)(1u, static_cast<u32>(std::lround(w))),
        .height = (std::max)(1u, static_cast<u32>(std::lround(h))),
    };
}

VkImageMemoryBarrier MakeBarrier(VkImage image, VkAccessFlags src_access,
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

void BarrierComputeWriteToRead(vk::CommandBuffer cmdbuf, VkImage image) {
    const auto barrier = MakeBarrier(image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, barrier);
}

void DiscardForComputeWrite(vk::CommandBuffer cmdbuf, VkImage image) {
    const auto barrier = MakeBarrier(image, 0, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, barrier);
}

void CopyFrameToSlot(vk::CommandBuffer cmdbuf, VkImage source, VkImage destination,
                     VkExtent2D extent, bool dst_initialized) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };

    const auto src_before = MakeBarrier(
        source,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, src_before);

    const auto dst_before = MakeBarrier(
        destination, 0U, VK_ACCESS_TRANSFER_WRITE_BIT,
        dst_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    cmdbuf.PipelineBarrier(dst_initialized ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                           : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, dst_before);

    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

    const auto src_after = MakeBarrier(
        source, 0U, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, src_after);

    const auto dst_after =
        MakeBarrier(destination, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           dst_after);
}

} // anonymous namespace

struct FlowPushConstants {
    float inv_render_w;
    float inv_render_h;
    float render_w;
    float render_h;
    float search_radius;
    float block_size;
};

struct InterpPushConstants {
    float inv_render_w;
    float inv_render_h;
    float render_w;
    float render_h;
    float blend_factor;
    float flow_scale;
};

Fsr3FrameGen::Fsr3FrameGen(const Device& device, MemoryAllocator& memory_allocator,
                           VkExtent2D extent, VkFormat format, f32 flow_scale)
    : m_allocator{memory_allocator}, m_extent{extent},
      m_flow_extent{ComputeFlowExtent(extent, flow_scale)}, m_format{format} {
    CreateResources(device);
    CreateShaders(device);
    CreatePipelines(device);
    m_initialized = true;
}

Fsr3FrameGen::~Fsr3FrameGen() = default;

void Fsr3FrameGen::CreateResources(const Device& device) {
    m_linear_sampler = CreateBilinearSampler(device);

    for (std::size_t i = 0; i < m_frame_images.size(); ++i) {
        m_frame_images[i] = m_allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = m_format,
            .extent = {m_extent.width, m_extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        m_frame_views[i] = CreateWrappedImageView(device, m_frame_images[i], m_format);
        m_frame_initialized[i] = false;
    }

    m_motion_image = CreateWrappedImage(m_allocator, m_flow_extent, VK_FORMAT_R16G16_SFLOAT);
    m_motion_view = CreateWrappedImageView(device, m_motion_image, VK_FORMAT_R16G16_SFLOAT);
    m_motion_initialized = false;

    const std::array pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
    };
    m_desc_pool = device.GetLogical().CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
}

void Fsr3FrameGen::CreateShaders(const Device& device) {
    m_flow_shader = BuildShader(device, FSR3_OPTICAL_FLOW_COMP_SPV);
    m_interp_shader = BuildShader(device, FSR3_FRAME_INTERPOLATION_COMP_SPV);
}

void Fsr3FrameGen::CreatePipelines(const Device& device) {
    const std::array flow_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };

    m_flow_desc_layout = device.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<u32>(flow_bindings.size()),
        .pBindings = flow_bindings.data(),
    });

    const VkPushConstantRange flow_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(FlowPushConstants),
    };

    m_flow_pipe_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_flow_desc_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &flow_push_range,
    });

    m_flow_pipeline = device.GetLogical().CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *m_flow_shader,
                .pName = "main",
            },
        .layout = *m_flow_pipe_layout,
    });

    const std::array interp_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };

    m_interp_desc_layout =
        device.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<u32>(interp_bindings.size()),
            .pBindings = interp_bindings.data(),
        });

    const VkPushConstantRange interp_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(InterpPushConstants),
    };

    m_interp_pipe_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_interp_desc_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &interp_push_range,
    });

    m_interp_pipeline = device.GetLogical().CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *m_interp_shader,
                .pName = "main",
            },
        .layout = *m_interp_pipe_layout,
    });

    const std::array<VkDescriptorSetLayout, 2> flow_layouts{*m_flow_desc_layout,
                                                            *m_flow_desc_layout};
    m_flow_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, flow_layouts);
}

void Fsr3FrameGen::Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                           VkImageView, u64 frame_count, bool capture_history, bool compute_flow) {
    if (frame_count == 0) {
        m_has_history = false;
        m_flow_valid = false;
    }

    // Idle / at-target bypass: zero GPU copies or optical flow dispatches when already at target.
    if (!capture_history && !compute_flow) {
        m_has_history = false;
        m_flow_valid = false;
        return;
    }

    // Duplicate-frame detection
    if (m_has_history && frame_count == m_last_frame_count) {
        return;
    }
    m_last_frame_count = frame_count;

    // Warm-up or fractional-cadence skip frame: only record the frame into the next ping-pong
    // slot without dispatching the optical flow compute shader.
    if (!m_has_history || !compute_flow) {
        const u32 dst_slot = m_has_history ? (m_curr_slot ^ 1u) : 0u;
        const bool dst_init = m_frame_initialized[dst_slot];
        m_frame_initialized[dst_slot] = true;
        m_curr_slot = dst_slot;
        m_has_history = true;
        m_flow_valid = false;

        scheduler.Record([this, current_frame, dst_slot, dst_init](vk::CommandBuffer cmdbuf) {
            CopyFrameToSlot(cmdbuf, current_frame, *m_frame_images[dst_slot], m_extent, dst_init);
        });
        return;
    }

    // Initialize both ping-pong descriptor sets once
    if (!m_flow_descriptors_initialized) {
        std::vector<VkDescriptorImageInfo> image_infos;
        image_infos.reserve(6);
        std::vector<VkWriteDescriptorSet> updates;
        updates.reserve(6);
        const VkDescriptorImageInfo motion_store_info{
            .imageView = *m_motion_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        for (u32 slot = 0; slot < 2; ++slot) {
            const u32 prev_slot = slot ^ 1u;
            updates.push_back(CreateWriteDescriptorSet(
                image_infos, *m_linear_sampler, *m_frame_views[slot], m_flow_desc_sets[slot], 0));
            updates.push_back(CreateWriteDescriptorSet(image_infos, *m_linear_sampler,
                                                       *m_frame_views[prev_slot],
                                                       m_flow_desc_sets[slot], 1));
            updates.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_flow_desc_sets[slot],
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &motion_store_info,
            });
        }
        device.GetLogical().UpdateDescriptorSets(updates, {});
        m_flow_descriptors_initialized = true;
    }

    const u32 dst_slot = m_curr_slot ^ 1u;
    const bool dst_init = m_frame_initialized[dst_slot];
    m_frame_initialized[dst_slot] = true;
    m_curr_slot = dst_slot;

    const bool motion_init = m_motion_initialized;
    m_motion_initialized = true;

    const u32 flow_w = m_flow_extent.width;
    const u32 flow_h = m_flow_extent.height;
    const FlowPushConstants constants{
        .inv_render_w = 1.0f / static_cast<float>(flow_w),
        .inv_render_h = 1.0f / static_cast<float>(flow_h),
        .render_w = static_cast<float>(flow_w),
        .render_h = static_cast<float>(flow_h),
        .search_radius = 2.0f,
        .block_size = 2.0f,
    };

    scheduler.Record([this, constants, current_frame, dst_slot, dst_init, motion_init, flow_w,
                      flow_h](vk::CommandBuffer cmdbuf) {
        CopyFrameToSlot(cmdbuf, current_frame, *m_frame_images[dst_slot], m_extent, dst_init);

        if (!motion_init) {
            DiscardForComputeWrite(cmdbuf, *m_motion_image);
        } else {
            const auto barrier =
                MakeBarrier(*m_motion_image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier);
        }

        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_flow_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_flow_pipe_layout, 0,
                                  m_flow_desc_sets[dst_slot], {});
        cmdbuf.PushConstants(*m_flow_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((flow_w + 15) / 16, (flow_h + 15) / 16, 1);

        BarrierComputeWriteToRead(cmdbuf, *m_motion_image);
    });

    m_flow_valid = true;
}

void Fsr3FrameGen::Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                            VkImageView destination_storage_view, float blend_factor,
                            float flow_scale) {
    if (!m_has_history || !m_flow_valid) {
        return;
    }

    if (!m_interp_pool) {
        m_interp_pool.emplace(device, scheduler);
        m_interp_allocator = m_interp_pool->Allocator(device, scheduler, *m_interp_desc_layout,
                                                      DescriptorBankInfo{.textures = 3, .images = 1});
    }
    const u32 prev_slot = m_curr_slot ^ 1u;
    const VkDescriptorSet set = m_interp_allocator.Commit();
    std::vector<VkDescriptorImageInfo> image_infos;
    image_infos.reserve(4);
    std::vector<VkWriteDescriptorSet> updates{
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_frame_views[m_curr_slot], set,
                                 0),
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_frame_views[prev_slot], set, 1),
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_motion_view, set, 2),
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
    const InterpPushConstants constants{
        .inv_render_w = 1.0f / static_cast<float>(out_w),
        .inv_render_h = 1.0f / static_cast<float>(out_h),
        .render_w = static_cast<float>(out_w),
        .render_h = static_cast<float>(out_h),
        .blend_factor = blend_factor,
        .flow_scale = flow_scale,
    };

    scheduler.Record([this, constants, out_w, out_h, set, destination](vk::CommandBuffer cmdbuf) {
        DiscardForComputeWrite(cmdbuf, destination);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipe_layout, 0, set,
                                  {});
        cmdbuf.PushConstants(*m_interp_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);

        const auto after =
            MakeBarrier(destination, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, after);
    });
}

} // namespace Vulkan
