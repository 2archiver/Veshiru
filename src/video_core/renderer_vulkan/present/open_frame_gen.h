// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright (c) 2026 Prithivi Nair, Prajwal R, Madhav S, Rashmi N Ugarakhod (Open-Frame-gen)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// In-process neural frame generator porting Deadly-pro/Open-Frame-gen's NanoModel
/// (v17, 89.9K parameters, depthwise-separable encoder/bottleneck/SE/decoder + bidirectional warp
/// + local-clamped CAS) directly to Vulkan compute shaders with zero external TensorRT/CUDA/DXGI
/// dependencies.
class OpenFrameGen {
public:
    explicit OpenFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                          VkExtent2D extent, VkFormat format, f32 flow_scale);
    ~OpenFrameGen();

    void Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                 VkImageView current_view, u64 frame_count, bool capture_history,
                 bool compute_flow);

    void Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                  VkImageView destination_storage_view, float blend_factor);

    [[nodiscard]] bool IsReady() const {
        return m_initialized && m_has_history && m_flow_valid;
    }

private:
    void CreateResources(const Device& device);
    void CreateShaders(const Device& device);
    void CreatePipelines(const Device& device);
    void WriteStaticDescriptors(const Device& device);
    void ReleaseWeightsUploadIfFree(Scheduler& scheduler);

    MemoryAllocator& m_allocator;
    VkExtent2D m_extent{};
    VkExtent2D m_net_extent{}; // Base NanoModel coordinate extent (multiple of 16)
    VkFormat m_format{VK_FORMAT_UNDEFINED};

    u32 m_w2{}, m_h2{};
    u32 m_w4{}, m_h4{};
    u32 m_w8{}, m_h8{};
    u32 m_w16{}, m_h16{};

    vk::Sampler m_linear_sampler;

    // Ping-pong pair holding previous and current presented frames (avoids an extra full-res copy
    // per frame). Slot m_curr_slot holds the newest captured frame; slot (m_curr_slot ^ 1) holds
    // the immediately preceding frame.
    std::array<vk::Image, 2> m_frame_images;
    std::array<vk::ImageView, 2> m_frame_views;
    std::array<bool, 2> m_frame_initialized{};
    u32 m_curr_slot{0};

    vk::Image m_flow_mask_image;
    vk::ImageView m_flow_mask_view;
    bool m_flow_mask_initialized{};

    vk::Buffer m_weights_upload;
    u64 m_weights_upload_tick{0};
    vk::Buffer m_weights_buffer;
    vk::Buffer m_buf_xs;
    vk::Buffer m_buf_s1;
    vk::Buffer m_buf_s2;
    vk::Buffer m_buf_s3;
    vk::Buffer m_buf_scratch_a;
    vk::Buffer m_buf_scratch_b;

    vk::ShaderModule m_prep_shader;
    vk::ShaderModule m_dw3x3_shader;
    vk::ShaderModule m_pw1x1_shader;
    vk::ShaderModule m_se_shader;
    vk::ShaderModule m_head_shader;
    vk::ShaderModule m_warp_shader;

    vk::DescriptorSetLayout m_prep_desc_layout;
    vk::PipelineLayout m_prep_pipe_layout;
    vk::Pipeline m_prep_pipeline;

    vk::DescriptorSetLayout m_dw_desc_layout;
    vk::PipelineLayout m_dw_pipe_layout;
    vk::Pipeline m_dw_pipeline;

    vk::DescriptorSetLayout m_pw_desc_layout;
    vk::PipelineLayout m_pw_pipe_layout;
    vk::Pipeline m_pw_pipeline;

    vk::DescriptorSetLayout m_se_desc_layout;
    vk::PipelineLayout m_se_pipe_layout;
    vk::Pipeline m_se_pipeline;

    vk::DescriptorSetLayout m_head_desc_layout;
    vk::PipelineLayout m_head_pipe_layout;
    vk::Pipeline m_head_pipeline;

    vk::DescriptorSetLayout m_warp_desc_layout;
    vk::PipelineLayout m_warp_pipe_layout;
    vk::Pipeline m_warp_pipeline;

    vk::DescriptorPool m_static_desc_pool;
    vk::DescriptorSets m_static_sets;

    std::optional<DescriptorPool> m_warp_pool;
    DescriptorAllocator m_warp_allocator;

    bool m_initialized{};
    bool m_weights_uploaded{};
    bool m_has_history{};
    bool m_flow_valid{};
    u64 m_last_frame_count{};
};

} // namespace Vulkan
