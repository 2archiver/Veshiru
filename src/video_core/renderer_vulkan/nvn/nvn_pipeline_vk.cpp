// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Vulkan objects of the NVN pipeline cache: render passes, descriptor/pipeline layouts and the
// assembly of graphics and compute VkPipelines from translated stages.

#include <array>
#include <span>

#include <boost/container/small_vector.hpp>
#include <fmt/format.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_bindless.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_impl.h"

namespace Vulkan::Nvn {

using Shader::Backend::SPIRV::RenderAreaLayout;
using Shader::Backend::SPIRV::RescalingLayout;

std::span<const VkSubpassDependency> RenderPassDependencies(bool sync_diet) {
    constexpr VkAccessFlags ALL_ACCESS = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    static constexpr std::array<VkSubpassDependency, 3> DEPENDENCIES{{
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .srcAccessMask = ALL_ACCESS,
            .dstAccessMask = ALL_ACCESS,
            .dependencyFlags = 0,
        },
        {
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            .srcAccessMask = ALL_ACCESS,
            .dstAccessMask = ALL_ACCESS,
            .dependencyFlags = 0,
        },
        {
            // Self-dependency, as the legacy pass: lets the replay put a by-region barrier in
            // the pass (texture/attachment feedback).
            .srcSubpass = 0,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        },
    }};
    // The two external dependencies are full syncs (every stage and access of everything before
    // or after the pass). The replay already puts an explicit barrier of the same scope between a
    // pass and the next op (EndPass marks the writes, the op's FlushPending / PrepareImages emits
    // it), so with the diet the pass has none and Vulkan's implicit ones apply: TOP_OF_PIPE ->
    // ALL_COMMANDS at the start (waits for nothing) and ALL_COMMANDS -> BOTTOM_OF_PIPE at the end
    // (blocks nothing). See DESIGN.md "Pass boundary syncs".
    return sync_diet ? std::span<const VkSubpassDependency>{DEPENDENCIES}.subspan(2)
                     : std::span<const VkSubpassDependency>{DEPENDENCIES};
}

// ================================================================================================
// render passes
// ================================================================================================

VkRenderPass PipelineCache::Impl::GetRenderPass(const RenderPassKey& key) {
    std::scoped_lock lock{render_pass_mutex};
    if (const auto it = render_passes.find(key); it != render_passes.end()) {
        return *it->second;
    }
    const VkSampleCountFlagBits samples =
        static_cast<VkSampleCountFlagBits>(key.samples == 0 ? 1 : key.samples);
    boost::container::small_vector<VkAttachmentDescription, MAX_COLOR_TARGETS + 1> descriptions;
    std::array<VkAttachmentReference, MAX_COLOR_TARGETS> color_references{};
    u32 color_count = 0;
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        const auto format = static_cast<VkFormat>(key.color[slot]);
        if (format == VK_FORMAT_UNDEFINED) {
            color_references[slot] = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
            continue;
        }
        color_references[slot] = {static_cast<u32>(descriptions.size()),
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        descriptions.push_back({
            .flags = 0,
            .format = format,
            .samples = samples,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        });
        color_count = slot + 1;
    }
    const auto depth_format = static_cast<VkFormat>(key.depth);
    VkAttachmentReference depth_reference{};
    const bool has_depth_attachment = depth_format != VK_FORMAT_UNDEFINED;
    if (has_depth_attachment) {
        depth_reference = {static_cast<u32>(descriptions.size()),
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        const bool depth = FormatHasDepth(depth_format);
        const bool stencil = FormatHasStencil(depth_format);
        descriptions.push_back({
            .flags = 0,
            .format = depth_format,
            .samples = samples,
            .loadOp = depth ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = depth ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = stencil ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp =
                stencil ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        });
    }
    const VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = color_count,
        .pColorAttachments = color_count != 0 ? color_references.data() : nullptr,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = has_depth_attachment ? &depth_reference : nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
    const std::span<const VkSubpassDependency> dependencies =
        RenderPassDependencies(options.sync_diet);
    vk::RenderPass pass = device.GetLogical().CreateRenderPass({
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = static_cast<u32>(descriptions.size()),
        .pAttachments = descriptions.empty() ? nullptr : descriptions.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<u32>(dependencies.size()),
        .pDependencies = dependencies.data(),
    });
    const VkRenderPass handle = *pass;
    render_passes.emplace(key, std::move(pass));
    return handle;
}
// ================================================================================================
// Vulkan objects
// ================================================================================================

bool PipelineCache::Impl::CanUsePush(u32 descriptor_count) const {
    if (!device.IsKhrPushDescriptorSupported() || descriptor_count > device.MaxPushDescriptors()) {
        return false;
    }
    return !device.IsExtDescriptorBufferSupported() ||
           device.DescriptorBufferProperties().bufferlessPushDescriptors;
}

PipelineCache::Impl::LayoutObjects PipelineCache::Impl::CreateLayoutObjects(
    const VC::DescriptorLayout& descriptors, bool is_compute, VkShaderStageFlags push_stages) {
    LayoutObjects out;
    boost::container::small_vector<VkDescriptorSetLayoutBinding, 32> bindings;
    u32 total = 0;
    for (const VC::DescriptorBinding& binding : descriptors.bindings) {
        bindings.push_back({
            .binding = binding.binding,
            .descriptorType = static_cast<VkDescriptorType>(binding.kind),
            .descriptorCount = std::max<u32>(binding.count, 1),
            .stageFlags = StageFlagsOf(binding),
            .pImmutableSamplers = nullptr,
        });
        total += std::max<u32>(binding.count, 1);
    }
    out.push_descriptors = !bindings.empty() && CanUsePush(total);
    if (!bindings.empty()) {
        boost::container::small_vector<VkDescriptorBindingFlags, 32> binding_flags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{};
        const void* next = nullptr;
        if (!out.push_descriptors && device.IsDescriptorBindingPartiallyBoundSupported()) {
            binding_flags.assign(bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            binding_flags_ci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .pNext = nullptr,
                .bindingCount = static_cast<u32>(binding_flags.size()),
                .pBindingFlags = binding_flags.data(),
            };
            next = &binding_flags_ci;
        }
        out.set_layout = device.GetLogical().CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = next,
            .flags = out.push_descriptors
                         ? static_cast<VkDescriptorSetLayoutCreateFlags>(
                               VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR)
                         : VkDescriptorSetLayoutCreateFlags{0},
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    }
    if (descriptors.uses_bindless_table) {
        // The table is set BINDLESS_SET: set 0 must exist even when the draw has no descriptors.
        if (bindings.empty()) {
            out.set_layout = device.GetLogical().CreateDescriptorSetLayout({
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .bindingCount = 0,
                .pBindings = nullptr,
            });
        }
        out.bindless_layout = MakeBindlessSetLayout(device);
    }
    VkPushConstantRange range{
        .stageFlags = push_stages,
        .offset = 0,
        .size = 0,
    };
    if (descriptors.UsesPushConstants()) {
        // As DescriptorLayoutBuilder::CreatePipelineLayout: compute has no down_factor word.
        const u32 skip = is_compute ? static_cast<u32>(sizeof(RescalingLayout::down_factor)) : 0u;
        range.size = static_cast<u32>(sizeof(RescalingLayout)) - skip +
                     static_cast<u32>(sizeof(RenderAreaLayout));
        out.push_constant_size = range.size;
    }
    const std::array<VkDescriptorSetLayout, 2> set_handles{
        *out.set_layout, out.bindless_layout ? *out.bindless_layout : VK_NULL_HANDLE};
    const u32 set_count = out.bindless_layout ? 2u : (set_handles[0] != VK_NULL_HANDLE ? 1u : 0u);
    out.layout = device.GetLogical().CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = set_count,
        .pSetLayouts = set_count != 0 ? set_handles.data() : nullptr,
        .pushConstantRangeCount = range.size != 0 ? 1u : 0u,
        .pPushConstantRanges = range.size != 0 ? &range : nullptr,
    });
    return out;
}

VC::DescriptorLayout PipelineCache::Impl::MergeLayouts(
    std::span<const StageArtifact* const> stages) {
    VC::DescriptorLayout merged;
    bool raw_stores_not_through = false;
    for (const StageArtifact* stage : stages) {
        if (stage == nullptr) {
            continue;
        }
        const VC::DescriptorLayout& slice = stage->record->layout;
        merged.bindings.insert(merged.bindings.end(), slice.bindings.begin(), slice.bindings.end());
        merged.uses_rescaling_uniform |= slice.uses_rescaling_uniform;
        merged.uses_render_area |= slice.uses_render_area;
        merged.uses_raw_global_memory |= slice.uses_raw_global_memory;
        merged.stores_raw_global_memory |= slice.stores_raw_global_memory;
        merged.requires_guest_warp |= slice.requires_guest_warp;
        merged.uses_bindless_table |= slice.uses_bindless_table;
        raw_stores_not_through |=
            slice.stores_raw_global_memory && !slice.raw_stores_written_through;
        merged.unsupported_count += slice.unsupported_count;
        merged.binding_count = std::max(merged.binding_count, slice.binding_count);
    }
    merged.raw_stores_written_through = merged.stores_raw_global_memory && !raw_stores_not_through;
    if (!merged.AllSupported()) {
        std::set<std::string> issues;
        for (const StageArtifact* stage : stages) {
            if (stage == nullptr) {
                continue;
            }
            for (const VC::DescriptorBinding& binding : stage->record->layout.bindings) {
                if (!binding.supported) {
                    issues.insert(fmt::format("{} [{} program {:016x}]", binding.issue,
                                              StageNameOf(stage->record->stage),
                                              stage->record->program_hash));
                }
            }
        }
        for (const std::string& issue : issues) {
            NoteIssue("layout: " + FirstLine(issue));
        }
    }
    return merged;
}

void PipelineCache::Impl::FillStages(PipelineBase& pipeline,
                                     std::span<const StageArtifact* const> stages) {
    if (ShaderLog::Enabled()) {
        pipeline.log = std::make_shared<PipelineLogInfo>();
    }
    for (const StageArtifact* stage : stages) {
        if (stage == nullptr) {
            continue;
        }
        const Disk::ShaderRecord& record = *stage->record;
        if (pipeline.log) {
            pipeline.log->records[record.stage] = stage->record;
        }
        PipelineStage& out = pipeline.stages[record.stage];
        out.present = true;
        out.stage = record.stage;
        out.program_constants = stage->constants;
        out.workgroup_size = record.workgroup_size;
        out.shared_memory_size = record.shared_memory_size;
        for (const TextureQuery& q : record.queries) {
            if (q.record.bank != VC::HandleBank || q.stage >= 8) {
                pipeline.queries_handle_bank_only = false;
            }
            if (q.stage < 32) {
                pipeline.query_stage_mask |= 1u << q.stage;
            }
        }
        pipeline.queries.insert(pipeline.queries.end(), record.queries.begin(),
                                record.queries.end());
    }
}

std::shared_ptr<GraphicsPipeline> PipelineCache::Impl::AssembleGraphics(
    const GraphicsFixedState& state, const RenderPassKey& targets,
    const std::array<const StageArtifact*, GRAPHICS_STAGE_COUNT>& stages,
    const std::shared_ptr<ShaderSet>& gpl_set, bool warm_parts) {
    try {
        auto pipeline = std::make_shared<GraphicsPipeline>();
        pipeline->render_pass = GetRenderPass(targets);

        std::array<const StageArtifact*, PIPELINE_ORDER.size()> ordered{};
        for (size_t i = 0; i < PIPELINE_ORDER.size(); ++i) {
            ordered[i] = stages[PIPELINE_ORDER[i]];
        }
        const bool mesh = stages[STAGE_MESH] != nullptr;
        if (mesh && !device.IsExtMeshShaderSupported()) {
            // Never hand the driver task/mesh stages it cannot create (logged once, counted).
            NoteIssue("pipeline: task/mesh programs need VK_EXT_mesh_shader with task and mesh "
                      "shaders; their draws are skipped");
            return nullptr;
        }
        const VkShaderStageFlags push_stages =
            mesh ? VkShaderStageFlags{VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                                      VK_SHADER_STAGE_FRAGMENT_BIT}
                 : VkShaderStageFlags{VK_SHADER_STAGE_ALL_GRAPHICS};
        pipeline->descriptors = MergeLayouts(ordered);
        LayoutObjects layout = CreateLayoutObjects(pipeline->descriptors, false, push_stages);
        if (gpl_set) {
            CreateShaderSetObjects(*gpl_set, pipeline->descriptors, push_stages);
        }

        u32 flags = 0;
        boost::container::small_vector<vk::ShaderModule, GRAPHICS_STAGE_COUNT> modules;
        boost::container::small_vector<VkPipelineShaderStageCreateInfo, GRAPHICS_STAGE_COUNT>
            stage_infos;
        bool has_tessellation = false;
        for (const StageArtifact* stage : ordered) {
            if (stage == nullptr) {
                continue;
            }
            const Disk::ShaderRecord& record = *stage->record;
            flags |= record.flags;
            has_tessellation |=
                record.stage == STAGE_TESS_CONTROL || record.stage == STAGE_TESS_EVAL;
            VkShaderModule module = VK_NULL_HANDLE;
            if (gpl_set) {
                // The set's modules: their handles are part of the library keys.
                module = *gpl_set->modules[record.stage];
            } else {
                modules.push_back(device.GetLogical().CreateShaderModule({
                    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .codeSize = record.spirv.size() * sizeof(u32),
                    .pCode = record.spirv.data(),
                }));
                module = *modules.back();
            }
            stage_infos.push_back({
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VkStageOf(record.stage),
                .module = module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            });
        }
        u32 viewport_count = 1;
        if ((flags & Disk::SHADER_STORES_VIEWPORT_INDEX) != 0 && device.SupportsMultiViewport()) {
            viewport_count = std::min<u32>(MAX_VIEWPORTS, device.GetMaxViewports());
        }
        const StageArtifact* const vertex = stages[STAGE_VERTEX];
        const StageArtifact* const fragment = stages[STAGE_FRAGMENT];
        pipeline->viewport_count = viewport_count;
        pipeline->attributes_read = vertex != nullptr ? vertex->record->attributes_read : 0;
        pipeline->color_outputs_written =
            fragment != nullptr ? fragment->record->outputs_written : 0;
        pipeline->dynamic_vertex_input = !mesh && vertex_input_mode == VertexInputMode::Dynamic;
        pipeline->mesh = mesh;
        pipeline->dynamic_fixed = gpl_set && gpl_dynamic_fixed;
        if (pipeline->dynamic_fixed) {
            pipeline->fixed_values = MakeDynamicFixedState(state);
        }

        std::vector<std::string> issues;
        GraphicsPipelineDescription description;
        DescribeGraphicsPipeline(
            device,
            GraphicsPipelineParams{
                .state = &state,
                .targets = &targets,
                .render_pass = pipeline->render_pass,
                // The shader parts are compiled with the set's layout (an identically defined
                // one: the pipeline's own layout is compatible with it).
                .layout = gpl_set ? *gpl_set->layout.layout : *layout.layout,
                .stages = stage_infos,
                .viewport_count = viewport_count,
                .attributes_read = pipeline->attributes_read,
                .color_outputs_written = pipeline->color_outputs_written,
                .has_tessellation = has_tessellation,
                .mesh = mesh,
                .vertex_input = vertex_input_mode,
                .dynamic_fixed = pipeline->dynamic_fixed,
            },
            description, issues);
        for (const std::string& issue : issues) {
            NoteIssue(issue);
        }
        if (gpl_set && warm_parts) {
            // Preload: compile the parts (on this background thread) and keep them in the
            // library cache; the linked pipeline is not used.
            pipeline->pipeline = library_cache->Link(description.ci, *vk_cache, gpl_set, false);
            return pipeline->pipeline ? pipeline : nullptr;
        }
        if (gpl_set) {
            try {
                pipeline->pipeline = library_cache->Link(description.ci, *vk_cache, gpl_set, true);
                if (pipeline->pipeline) {
                    ++stat_gpl_cached_links;
                } else {
                    pipeline->pipeline = library_cache->Link(description.ci, *vk_cache, gpl_set,
                                                             false, part_workers.get());
                }
            } catch (const vk::Exception& e) {
                // A part or the link failed: compiled whole below.
                NoteIssue(fmt::format("pipeline: library link failed, compiling whole: {}",
                                      FirstLine(e.what())));
                pipeline->pipeline = vk::Pipeline{};
            }
            pipeline->fast_linked = static_cast<bool>(pipeline->pipeline);
            if (pipeline->fast_linked) {
                ++stat_gpl_fast_links;
            } else {
                // Something the parts cannot express (no fragment program, static rasterizer
                // discard): compiled whole, nothing to optimize later.
                ++stat_gpl_whole;
                description.ci.layout = *layout.layout;
            }
        }
        if (!pipeline->pipeline) {
            pipeline->pipeline =
                device.GetLogical().CreateGraphicsPipeline(description.ci, *vk_cache);
        }
        pipeline->layout = std::move(layout.layout);
        pipeline->set_layout = std::move(layout.set_layout);
        pipeline->bindless_layout = std::move(layout.bindless_layout);
        pipeline->push_descriptors = layout.push_descriptors;
        pipeline->push_constant_size = layout.push_constant_size;
        pipeline->push_constant_stages = push_stages;
        FillStages(*pipeline, ordered);
        return pipeline;
    } catch (const vk::Exception& e) {
        ++stat_failures;
        NoteIssue(fmt::format("pipeline: Vulkan error creating a graphics pipeline: {}",
                              FirstLine(e.what())));
    } catch (const std::exception& e) {
        ++stat_failures;
        NoteIssue(fmt::format("pipeline: exception creating a graphics pipeline: {}",
                              FirstLine(e.what())));
    }
    return nullptr;
}

std::shared_ptr<ComputePipeline> PipelineCache::Impl::AssembleCompute(
    const StageArtifact& stage, const VC::ComputeDriverLayout& compute_layout) {
    try {
        auto pipeline = std::make_shared<ComputePipeline>();
        const std::array<const StageArtifact*, 1> single{&stage};
        pipeline->descriptors = MergeLayouts(single);
        LayoutObjects layout =
            CreateLayoutObjects(pipeline->descriptors, true, VK_SHADER_STAGE_COMPUTE_BIT);
        const Disk::ShaderRecord& record = *stage.record;
        vk::ShaderModule shader_module = device.GetLogical().CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = record.spirv.size() * sizeof(u32),
            .pCode = record.spirv.data(),
        });
        // As vk_compute_pipeline.cpp: only programs that need whole warps ask for 32 threads per
        // subgroup (forcing it elsewhere makes Intel compile SIMD32).
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .pNext = nullptr,
            .requiredSubgroupSize = GuestWarpSize,
        };
        const bool guest_warp = record.layout.requires_guest_warp &&
                                device.IsExtSubgroupSizeControlSupported() &&
                                device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
        pipeline->pipeline = device.GetLogical().CreateComputePipeline(
            {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage =
                    {
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .pNext = guest_warp ? &subgroup_size_ci : nullptr,
                        .flags = 0,
                        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                        .module = *shader_module,
                        .pName = "main",
                        .pSpecializationInfo = nullptr,
                    },
                .layout = *layout.layout,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0,
            },
            *vk_cache);
        pipeline->layout = std::move(layout.layout);
        pipeline->set_layout = std::move(layout.set_layout);
        pipeline->bindless_layout = std::move(layout.bindless_layout);
        pipeline->push_descriptors = layout.push_descriptors;
        pipeline->push_constant_size = layout.push_constant_size;
        pipeline->push_constant_stages = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline->compute_layout = compute_layout;
        FillStages(*pipeline, single);
        return pipeline;
    } catch (const vk::Exception& e) {
        ++stat_failures;
        NoteIssue(fmt::format("pipeline: Vulkan error creating a compute pipeline: {}",
                              FirstLine(e.what())));
    } catch (const std::exception& e) {
        ++stat_failures;
        NoteIssue(fmt::format("pipeline: exception creating a compute pipeline: {}",
                              FirstLine(e.what())));
    }
    return nullptr;
}

} // namespace Vulkan::Nvn
