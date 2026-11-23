// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Clears the attachment clear cannot express: color clears with a channel write mask and
// depth/stencil clears with a partial stencil mask. They run as a full-screen triangle with the
// legacy path's host clear shaders in a render pass of their own.

#include <cstring>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/host_shaders/vulkan_color_clear_frag_spv.h"
#include "video_core/host_shaders/vulkan_color_clear_vert_spv.h"
#include "video_core/host_shaders/vulkan_depthstencil_clear_frag_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan::Nvn {

ClearPipelines::ClearPipelines(const Device& device) {
    vert = BuildShader(device, VULKAN_COLOR_CLEAR_VERT_SPV);
    color_frag = BuildShader(device, VULKAN_COLOR_CLEAR_FRAG_SPV);
    stencil_frag = BuildShader(device, VULKAN_DEPTHSTENCIL_CLEAR_FRAG_SPV);
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(f32) * 4,
    };
    layout = device.GetLogical().CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
}

VkPipeline ClearPipelines::Get(const Device& device, const ClearPipelineKey& key) {
    for (auto& [existing, pipeline] : pipelines) {
        if (existing == key) {
            return *pipeline;
        }
    }
    const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = *vert,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = key.stencil ? *stencil_frag : *color_frag,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    }};
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    const VkPipelineViewportStateCreateInfo viewport{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = key.samples,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };
    const VkStencilOpState stencil{
        .failOp = VK_STENCIL_OP_KEEP,
        .passOp = VK_STENCIL_OP_REPLACE,
        .depthFailOp = VK_STENCIL_OP_KEEP,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .compareMask = 0xFF,
        .writeMask = key.stencil_mask,
        .reference = key.stencil_reference,
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = key.stencil ? VK_TRUE : VK_FALSE,
        .front = stencil,
        .back = stencil,
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 0.0f,
    };
    const VkPipelineColorBlendAttachmentState attachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = key.color_mask,
    };
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = key.stencil ? 0u : 1u,
        .pAttachments = key.stencil ? nullptr : &attachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    static constexpr std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    vk::Pipeline pipeline = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<u32>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = *layout,
        .renderPass = key.render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });
    pipelines.emplace_back(key, std::move(pipeline));
    return *pipelines.back().second;
}

namespace {

/// Records one full-screen triangle with `pipeline` into a render pass over `view` only.
bool DrawClearTriangle(DrawHost& host, Context::Impl& impl, VkRenderPass render_pass, VkImageView view,
                       VkExtent2D target_extent, VkRect2D rect, VkPipeline pipeline,
                       VkPipelineLayout layout, const std::array<f32, 4>& push, bool gated) {
    try {
        auto framebuffer = std::make_shared<vk::Framebuffer>(impl.device.GetLogical().CreateFramebuffer({
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = target_extent.width,
            .height = target_extent.height,
            .layers = 1,
        }));
        vk::CommandBuffer cmd = host.HostCmd();
        GpuProf::Flight* const prof = impl.gpu_profiler ? host.HostFlight().gpu_prof.get() : nullptr;
        const GpuProf::Scope prof_scope{
            prof, cmd,
            prof ? GpuProf::Tag{.kind = GpuProf::Kind::Clear,
                                .width = rect.extent.width,
                                .height = rect.extent.height}
                 : GpuProf::Tag{}};
        // A render pass has no external dependency with the sync diet: order it after the pass that
        // just ended (the target was written there).
        host.HostBarrierBeforePass();
        // A barrier request may wait for the next command (EDEN_NX2_TRY_BARRIER_MERGE): asking
        // for the command buffer again records it before the pass begins.
        cmd = host.HostCmd();
        const VkRenderPassBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = render_pass,
            .framebuffer = **framebuffer,
            .renderArea = rect,
            .clearValueCount = 0,
            .pClearValues = nullptr,
        };
        cmd.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
        if (gated) {
            host.HostResumeCondition();
        }
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        const VkViewport viewport{
            .x = static_cast<f32>(rect.offset.x),
            .y = static_cast<f32>(rect.offset.y),
            .width = static_cast<f32>(rect.extent.width),
            .height = static_cast<f32>(rect.extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        cmd.SetViewport(0, std::span{&viewport, 1});
        cmd.SetScissor(0, std::span{&rect, 1});
        cmd.PushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push.data());
        cmd.Draw(3, 1, 0, 0);
        if (gated) {
            host.HostEndRenderCondition();
        }
        cmd.EndRenderPass();
        host.HostFlight().keepalive.push_back(std::move(framebuffer));
        host.HostMarkWritten();
        return true;
    } catch (const vk::Exception& exception) {
        host.HostFailed("Clear", fmt::format("clear draw failed: {}", exception.what()));
        return false;
    }
}

} // namespace

bool DrawContext::MaskedColorClear(const VC::ClearColorPayload& p, VkRect2D rect, bool gated) {
    const TargetBinding& target = color_targets[p.index];
    Texture& texture = *target.texture;
    Storage& storage = *texture.storage;
    HostFormat format = texture.host;
    if (target.ref.view.set_format) {
        if (const auto resolved = ResolveHostFormat(impl.device, target.ref.view.format)) {
            format = *resolved;
        }
    }
    if (p.is_integer || VideoCore::Surface::IsPixelFormatInteger(format.pixel_format)) {
        host->HostUnsupported("ClearColor", "channel-masked clear of an integer target");
        return false;
    }
    AttachmentShape shape;
    if (!ShapeOf(target, false, shape)) {
        return false;
    }
    EndPass(BreakCause::MaskedClear);
    host->HostPauseCounting();
    RenderPassKey key{};
    key.color[0] = static_cast<s32>(format.format);
    key.samples = static_cast<u32>(storage.samples);
    const VkRenderPass render_pass = resources.Cache().GetRenderPass(key);
    if (!render_pass) {
        return false;
    }
    const ClearPipelineKey pipeline_key{
        .render_pass = render_pass,
        .samples = storage.samples,
        .color_mask = p.mask & 0xF,
        .stencil = false,
        .stencil_mask = 0,
        .stencil_reference = 0,
    };
    const VkPipeline clear_pipeline = resources.clears.Get(impl.device, pipeline_key);
    std::array<f32, 4> color;
    std::memcpy(color.data(), p.color, sizeof(color));
    // One render pass per layer of the attachment (a slice of a 3D texture is a layer).
    for (u32 i = 0; i < shape.layer_count; ++i) {
        ViewKey view_key = shape.key;
        view_key.type = VK_IMAGE_VIEW_TYPE_2D;
        view_key.layers = 1;
        view_key.base_layer += i;
        const VkImageView view = resources.GetView(texture, view_key);
        if (!view) {
            return false;
        }
        // The condition region (if any) is closed around every pass, so it is re-opened per layer
        // by DrawClearTriangle only when `gated`.
        if (!DrawClearTriangle(*host, impl, render_pass, view, shape.extent, rect, clear_pipeline,
                               *resources.clears.layout, color, gated)) {
            return false;
        }
    }
    return true;
}

bool DrawContext::MaskedStencilClear(const VC::ClearDepthStencilPayload& p, VkRect2D rect, bool gated) {
    Texture& texture = *depth_target.texture;
    Storage& storage = *texture.storage;
    AttachmentShape shape;
    if (!ShapeOf(depth_target, true, shape)) {
        return false;
    }
    EndPass(BreakCause::MaskedClear);
    host->HostPauseCounting();
    RenderPassKey key{};
    key.depth = static_cast<s32>(storage.host.format);
    key.samples = static_cast<u32>(storage.samples);
    const VkRenderPass render_pass = resources.Cache().GetRenderPass(key);
    if (!render_pass) {
        return false;
    }
    const ClearPipelineKey pipeline_key{
        .render_pass = render_pass,
        .samples = storage.samples,
        .color_mask = 0,
        .stencil = true,
        .stencil_mask = static_cast<u32>(p.stencil_mask) & 0xFF,
        .stencil_reference = static_cast<u32>(p.stencil) & 0xFF,
    };
    const VkPipeline clear_pipeline = resources.clears.Get(impl.device, pipeline_key);
    const std::array<f32, 4> depth{p.depth, 0.0f, 0.0f, 0.0f};
    for (u32 i = 0; i < shape.layer_count; ++i) {
        ViewKey view_key = shape.key;
        view_key.type = VK_IMAGE_VIEW_TYPE_2D;
        view_key.layers = 1;
        view_key.base_layer += i;
        view_key.aspect = storage.aspect;
        const VkImageView view = resources.GetView(texture, view_key);
        if (!view) {
            return false;
        }
        if (!DrawClearTriangle(*host, impl, render_pass, view, shape.extent, rect, clear_pipeline,
                               *resources.clears.layout, depth, gated)) {
            return false;
        }
    }
    return true;
}

} // namespace Vulkan::Nvn
