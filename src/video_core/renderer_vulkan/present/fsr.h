// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "common/math_util.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

class FSR {
public:
    explicit FSR(const Device& device, MemoryAllocator& memory_allocator, size_t image_count, VkExtent2D extent);
    VkImageView Draw(const Device& device, Scheduler& scheduler, size_t image_index, VkImage source_image,
                     VkImageView source_image_view, VkExtent2D input_image_extent,
                     const Common::Rectangle<f32>& crop_rect);
    /// Records EASU and RCAS into an already-open command buffer (VTX-HLE's replay). Unlike
    /// Draw(), this does not go through the scheduler.
    VkImageView Record(const Device& device, vk::CommandBuffer cmdbuf, size_t image_index,
                       VkImage source_image, VkImageView source_image_view,
                       VkExtent2D input_image_extent, const Common::Rectangle<f32>& crop_rect);
    [[nodiscard]] VkImage ResultImage(size_t image_index) const;
    /// The descriptor sets of the last Record(); the caller keeps them alive until the command
    /// buffer that bound them has completed.
    [[nodiscard]] std::shared_ptr<void> TakeRecordResources() {
        return std::move(m_record_descriptor_sets);
    }

private:
    void CreateImages(const Device& device);
    void CreateRenderPasses(const Device& device);
    void CreateSampler(const Device& device);
    void CreateShaders(const Device& device);
    void CreateDescriptorPool(const Device& device);
    void CreateDescriptorSetLayout(const Device& device);
    void CreateDescriptorSets(const Device& device);
    void CreatePipelineLayouts(const Device& device);
    void CreatePipelines(const Device& device);

    void UploadImages(const Device& device, Scheduler& scheduler);
    void UpdateDescriptorSets(const Device& device, VkImageView image_view, size_t image_index);

    MemoryAllocator& m_memory_allocator;
    const size_t m_image_count;
    const VkExtent2D m_extent;

    enum FsrStage {
        Easu,
        Rcas,
        MaxFsrStage,
    };

    vk::DescriptorPool m_descriptor_pool;
    vk::DescriptorSetLayout m_descriptor_set_layout;
    vk::PipelineLayout m_pipeline_layout;
    vk::ShaderModule m_vert_shader;
    vk::ShaderModule m_easu_shader;
    vk::ShaderModule m_rcas_shader;
    vk::Pipeline m_easu_pipeline;
    vk::Pipeline m_rcas_pipeline;
    vk::RenderPass m_renderpass;
    vk::Sampler m_sampler;

    struct Images {
        vk::DescriptorSets descriptor_sets;
        std::array<vk::Image, MaxFsrStage> images;
        std::array<vk::ImageView, MaxFsrStage> image_views;
        std::array<vk::Framebuffer, MaxFsrStage> framebuffers;
    };
    std::vector<Images> m_dynamic_images;
    bool m_images_ready{};

    struct RecordDescriptorSets {
        vk::DescriptorPool pool;
        vk::DescriptorSets sets;
    };
    std::shared_ptr<RecordDescriptorSets> m_record_descriptor_sets;
    bool m_record_images_ready{};
};

} // namespace Vulkan
