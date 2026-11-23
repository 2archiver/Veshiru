// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/frontend/framebuffer_layout.h"
#include "video_core/framebuffer_config.h"
#include "video_core/host_shaders/vulkan_present_vert_spv.h"
#include "video_core/renderer_vulkan/present/layer.h"
#include "video_core/renderer_vulkan/present/present_push_constants.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/present/window_adapt_pass.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

WindowAdaptPass::WindowAdaptPass(const Device& device, VkFormat frame_format,
                                 vk::Sampler&& sampler_, vk::ShaderModule&& fragment_shader_,
                                 bool push_descriptor)
    : recording_device{device}
    , use_push_descriptors{push_descriptor}
    , sampler(std::move(sampler_))
    , fragment_shader(std::move(fragment_shader_))
{
    CreateDescriptorSetLayout(device, push_descriptor);
    CreatePipelineLayout(device);
    CreateVertexShader(device);
    CreateRenderPass(device, frame_format);
    CreatePipelines(device, push_descriptor);
}

WindowAdaptPass::~WindowAdaptPass() = default;

void WindowAdaptPass::Draw(const Device& device, RasterizerVulkan& rasterizer, Scheduler& scheduler, size_t image_index,
                           std::list<Layer>& layers,
                           std::span<const Tegra::FramebufferConfig> configs,
                           const Layout::FramebufferLayout& layout,
                           VkFramebuffer framebuffer, VkExtent2D extent,
                           const HdrPresent& output) {

    const VkFramebuffer host_framebuffer{framebuffer};
    const VkRenderPass renderpass{*render_pass};
    const VkPipelineLayout graphics_pipeline_layout{*pipeline_layout};
    const VkExtent2D render_area{extent};

    const size_t layer_count = configs.size();
    std::vector<PresentPushConstants> push_constants(layer_count);
    std::vector<VkDescriptorSet> descriptor_sets(layer_count);
    std::vector<VkPipeline> graphics_pipelines(layer_count);

    auto layer_it = layers.begin();
    for (size_t i = 0; i < layer_count; i++) {
        switch (configs[i].blending) {
        case Tegra::BlendMode::Opaque:
        default:
            graphics_pipelines[i] = *opaque_pipeline;
            break;
        case Tegra::BlendMode::Premultiplied:
            graphics_pipelines[i] = *premultiplied_pipeline;
            break;
        case Tegra::BlendMode::Coverage:
            graphics_pipelines[i] = *coverage_pipeline;
            break;
        }

        layer_it->ConfigureDraw(device, &push_constants[i], &descriptor_sets[i], rasterizer, *sampler,
                                image_index, configs[i], layout, output);
        layer_it++;
    }

    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const f32 bg_red = Settings::values.bg_red.GetValue() / 255.0f;
        const f32 bg_green = Settings::values.bg_green.GetValue() / 255.0f;
        const f32 bg_blue = Settings::values.bg_blue.GetValue() / 255.0f;
        const VkClearAttachment clear_attachment{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .colorAttachment = 0,
            .clearValue =
                {
                    .color = {.float32 = {bg_red, bg_green, bg_blue, 1.0f}},
                },
        };
        const VkClearRect clear_rect{
            .rect =
                {
                    .offset = {0, 0},
                    .extent = render_area,
                },
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        BeginRenderPass(cmdbuf, renderpass, host_framebuffer, render_area);
        cmdbuf.ClearAttachments({clear_attachment}, {clear_rect});

        for (size_t i = 0; i < layer_count; i++) {
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipelines[i]);
            cmdbuf.PushConstants(graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                                 push_constants[i]);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0,
                                      descriptor_sets[i], {});
            cmdbuf.Draw(4, 1, 0, 0);
        }

        cmdbuf.EndRenderPass();
    });
}

std::shared_ptr<void> WindowAdaptPass::RecordTexture(vk::CommandBuffer cmd, const Frame& dst,
                                                     VkImageView source,
                                                     const Common::Rectangle<f32>& crop,
                                                     const Common::Rectangle<u32>& rect) const {
    const VkExtent2D extent{dst.width, dst.height};
    BeginRenderPass(cmd, *render_pass, *dst.framebuffer, extent);
    if (rect.GetWidth() != dst.width || rect.GetHeight() != dst.height) {
        const VkClearAttachment clear{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .colorAttachment = 0,
            .clearValue = {.color = {.float32 = {Settings::values.bg_red.GetValue() / 255.0f,
                                                 Settings::values.bg_green.GetValue() / 255.0f,
                                                 Settings::values.bg_blue.GetValue() / 255.0f,
                                                 1.0f}}},
        };
        const VkClearRect area{.rect = {{0, 0}, extent}, .baseArrayLayer = 0, .layerCount = 1};
        cmd.ClearAttachments({clear}, {area});
    }
    const VkDescriptorImageInfo image{
        .sampler = *sampler,
        .imageView = source,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &image,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    const f32 x = static_cast<f32>(rect.left);
    const f32 y = static_cast<f32>(rect.top);
    const f32 right = static_cast<f32>(rect.right);
    const f32 bottom = static_cast<f32>(rect.bottom);
    const PresentPushConstants constants{
        .modelview_matrix =
            MakeOrthographicMatrix(static_cast<f32>(dst.width), static_cast<f32>(dst.height)),
        .vertices = {ScreenRectVertex{x, y, crop.left, crop.top},
                     ScreenRectVertex{right, y, crop.right, crop.top},
                     ScreenRectVertex{x, bottom, crop.left, crop.bottom},
                     ScreenRectVertex{right, bottom, crop.right, crop.bottom}},
    };
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *opaque_pipeline);
    struct DescriptorResources {
        vk::DescriptorPool pool;
        vk::DescriptorSets sets;
    };
    std::shared_ptr<DescriptorResources> descriptors;
    if (use_push_descriptors) {
        cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0, write);
    } else {
        descriptors = std::make_shared<DescriptorResources>();
        descriptors->pool = CreateWrappedDescriptorPool(recording_device, 1, 1);
        descriptors->sets = CreateWrappedDescriptorSets(descriptors->pool, *descriptor_set_layout);
        write.dstSet = descriptors->sets[0];
        recording_device.GetLogical().UpdateDescriptorSets(std::span{&write, 1}, {});
        cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                               descriptors->sets[0], {});
    }
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, constants);
    cmd.Draw(4, 1, 0, 0);
    cmd.EndRenderPass();
    return descriptors;
}

VkDescriptorSetLayout WindowAdaptPass::GetDescriptorSetLayout() {
    return *descriptor_set_layout;
}

VkRenderPass WindowAdaptPass::GetRenderPass() {
    return *render_pass;
}

void WindowAdaptPass::CreateDescriptorSetLayout(const Device& device, bool push_descriptor) {
    if (!push_descriptor) {
        descriptor_set_layout =
            CreateWrappedDescriptorSetLayout(device, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
        return;
    }
    constexpr VkDescriptorSetLayoutBinding source{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };
    descriptor_set_layout = device.GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = 1,
        .pBindings = &source,
    });
}

void WindowAdaptPass::CreatePipelineLayout(const Device& device) {
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PresentPushConstants),
    };

    pipeline_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = descriptor_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
}

void WindowAdaptPass::CreateVertexShader(const Device& device) {
    vertex_shader = BuildShader(device, VULKAN_PRESENT_VERT_SPV);
}

void WindowAdaptPass::CreateRenderPass(const Device& device, VkFormat frame_format) {
    render_pass = CreateWrappedRenderPass(device, frame_format, VK_IMAGE_LAYOUT_UNDEFINED);
}

void WindowAdaptPass::CreatePipelines(const Device& device, bool opaque_only) {
    opaque_pipeline = CreateWrappedPipeline(device, render_pass, pipeline_layout,
                                            std::tie(vertex_shader, fragment_shader));
    if (opaque_only) {
        return;
    }
    premultiplied_pipeline = CreateWrappedPremultipliedBlendingPipeline(
        device, render_pass, pipeline_layout, std::tie(vertex_shader, fragment_shader));
    coverage_pipeline = CreateWrappedCoverageBlendingPipeline(
        device, render_pass, pipeline_layout, std::tie(vertex_shader, fragment_shader));
}

} // namespace Vulkan
