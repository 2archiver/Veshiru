// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Optical Flow Frame Generation pipeline (FSR3-style compute optical flow + bilateral
/// interpolation).
///
/// Optimizations:
///  - Zero GPU work when idle / target FPS is already met
///  - Ping-pong frame history (single copy per captured frame instead of two)
///  - Scaled optical flow resolution capped at 640x360
///  - Shared-memory tile caching + early-out for static pixels in optical flow shader
///  - Duplicate-frame detection & descriptor caching
class Fsr3FrameGen {
public:
    explicit Fsr3FrameGen(const Device& device, MemoryAllocator& memory_allocator,
                          VkExtent2D extent, VkFormat format, f32 flow_scale);
    ~Fsr3FrameGen();

    void Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                 VkImageView current_view, u64 frame_count, bool capture_history,
                 bool compute_flow);

    void Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                  VkImageView destination_storage_view, float blend_factor, float flow_scale);

    [[nodiscard]] bool IsReady() const {
        return m_initialized && m_has_history && m_flow_valid;
    }

private:
    void CreateResources(const Device& device);
    void CreateShaders(const Device& device);
    void CreatePipelines(const Device& device);

    MemoryAllocator& m_allocator;
    VkExtent2D m_extent;
    VkExtent2D m_flow_extent;
    VkFormat m_format;

    vk::Sampler m_linear_sampler;
    vk::ShaderModule m_flow_shader;
    vk::ShaderModule m_interp_shader;

    vk::DescriptorSetLayout m_flow_desc_layout;
    vk::PipelineLayout m_flow_pipe_layout;
    vk::Pipeline m_flow_pipeline;

    vk::DescriptorSetLayout m_interp_desc_layout;
    vk::PipelineLayout m_interp_pipe_layout;
    vk::Pipeline m_interp_pipeline;

    vk::DescriptorPool m_desc_pool;
    vk::DescriptorSets m_flow_desc_sets; // [0]: curr=slot0/prev=slot1, [1]: curr=slot1/prev=slot0
    std::optional<DescriptorPool> m_interp_pool;
    DescriptorAllocator m_interp_allocator;

    // Ping-pong frame history: m_curr_slot holds newest frame, (m_curr_slot ^ 1) holds previous.
    std::array<vk::Image, 2> m_frame_images;
    std::array<vk::ImageView, 2> m_frame_views;
    std::array<bool, 2> m_frame_initialized{};
    u32 m_curr_slot{0};

    vk::Image m_motion_image;
    vk::ImageView m_motion_view;
    bool m_motion_initialized{};

    bool m_initialized{};
    bool m_has_history{};

    // Duplicate-frame detection: skip optical flow when the frame hasn't changed
    u64 m_last_frame_count{};
    bool m_flow_valid{};

    bool m_flow_descriptors_initialized{};
};

} // namespace Vulkan
