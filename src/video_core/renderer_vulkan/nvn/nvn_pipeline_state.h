// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// NVN enum -> Vulkan / shader recompiler conversions and the VkGraphicsPipelineCreateInfo builder
// of the NVN pipeline cache. Pure functions of the raw NVN state words (state_layouts.h); the
// pipeline cache decides what is cached, this file decides what the words mean.

#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

/// The Shader::Profile / HostTranslateInfo of the device, built exactly as vk_pipeline_cache.cpp
/// builds them for the legacy path.
[[nodiscard]] Shader::Profile MakeProfile(const Device& device);
[[nodiscard]] Shader::HostTranslateInfo MakeHostInfo(const Device& device,
                                                     const Shader::Profile& profile);
/// Whether the NVN HLE lowers HMMA to the cooperative-matrix path (MakeProfile's
/// support_cooperative_matrix): the device's VK_NV_cooperative_matrix admission and
/// EDEN_NX2_TENSOR (`tensor_mode`, unset or "fast") decide, never the upscaler setting. The HLE
/// replaces a game upscale dispatch it recognises with the host upscaler before that program is
/// looked up (DrawContext::Dispatch), so every program that reaches translation is one the game
/// runs itself; rejecting its HMMA for a host upscaler setting skipped e.g. Ocarina of Time's
/// upscaler network and left its 3D scene black under upscaler = DLSS.
[[nodiscard]] bool CooperativeMatrixLoweringEnabled(bool device_cooperative_matrix,
                                                    const char* tensor_mode);

/// The settings and environment switches the shader recompiler reads while it translates (its
/// loop safety setting, the EDEN_NX2_* diagnostic switches); the build's source hash cannot see
/// them. Each field holds the value as the reader interprets it, so two spellings of one effect
/// are one value. Presentation settings (HDR, upscaler, anti-aliasing, scaling filter, frame
/// generation) never change the translated SPIR-V: not listed. The resolution setting does while
/// it scales (the rescaling pass bakes the factors in; at 1X the pass is skipped, see
/// HostTranslateInfo::skip_rescaling_pass): the fields below count only while active.
struct TranslationSwitches {
    bool loop_safety_checks_disabled{};  ///< Settings disable_shader_loop_safety_checks
    bool resolution_scaling_active{};    ///< Settings resolution_info.active (RescalingPass runs)
    u32 resolution_up_scale{1};          ///< baked into the rescaled SPIR-V, only while active
    u32 resolution_down_shift{};
    u32 resolution_up_factor_bits{};     ///< float bits
    u32 resolution_down_factor_bits{};
    bool fp32_denorm_preserve{};         ///< EDEN_NX2_FP32_DENORM_PRESERVE=1
    char nan_outputs{};                  ///< EDEN_NAN_OUTPUTS: 'm', 'z', 'b' or 0
    bool undef_is_zero{true};            ///< EDEN_NX2_UNDEF_ZERO unset or not 0
    bool tensor_mode_set{};              ///< EDEN_NX2_TENSOR present
    std::string tensor_mode;             ///< its value (selects the HMMA lowering)
    u8 tensor_lowering{};                ///< the title's VideoCommon::NativeTensorLowering
    bool raw_coverage{};                 ///< EDEN_NX2_RAW_COVERAGE=1 (the miss ring reports)
    bool lean_global_memory{false};      ///< EDEN_NX2_LEAN_SSBO set and not 0
    bool raw_global_fast{};              ///< EDEN_NX2_TRY_RAWGLOBAL_FAST=1 (last-hit translation)
    bool direct_unused_atomics{};        ///< EDEN_NX2_TRY_DIRECT_UNUSED_ATOMICS=1
    bool native_f32_atomics{};           ///< EDEN_NX2_TRY_NATIVE_F32_ATOMICS=1
    bool atomic_reserve{};               ///< EDEN_NX2_TRY_ATOMIC_RESERVE=1
    bool ro_mirror_exact{};              ///< EDEN_NX2_TRY_RO_MIRROR_EXACT=1 (raw stores reach RO mirrors)
    bool storage_raw_write_through{};   ///< EDEN_NX2_TRY_STORAGE_RAW_WRITE_THROUGH
    bool strong_global_fence{};          ///< EDEN_NX2_STRONG_GLOBAL_FENCE=1 (LDG/STG .STRONG.GPU)
    [[nodiscard]] bool operator==(const TranslationSwitches&) const = default;
};
/// A pipeline-key suffix for currently enabled TRY switches; zero preserves pre-toggle keys.
[[nodiscard]] u64 TrySwitchesHash();

/// Reads the switches of this process now.
[[nodiscard]] TranslationSwitches CurrentTranslationSwitches();
/// 64-bit hash of the switches that matter (every field; the resolution ones only while scaling).
[[nodiscard]] u64 HashTranslationSwitches(const TranslationSwitches& switches);

/// How a vertex attribute of an NVNformat is fed to the shader and to Vulkan.
struct VertexFormatInfo {
    Shader::AttributeType type{Shader::AttributeType::Disabled};
    /// The VkFormat the attribute is declared with; UNDEFINED when the NVN format cannot be a
    /// vertex attribute.
    VkFormat format{VK_FORMAT_UNDEFINED};
    /// Scaled (integer to float) attribute that has no scaled Vulkan counterpart on this device
    /// (32-bit components, or a device that must emulate scaled formats and a profile that does
    /// not): `format` is then the raw integer format.
    bool scaled_without_support{};
};

/// `nvn_format` 0 (unused) gives Disabled / UNDEFINED. Three-component and packed vertex formats
/// included. `format` is UNDEFINED and type Disabled for a non-vertex format.
[[nodiscard]] VertexFormatInfo ResolveVertexFormat(const Device& device, u32 nvn_format);
/// Same, from the only device property the table depends on (`Device::MustEmulateScaledFormats`):
/// lets key building run without a device.
[[nodiscard]] VertexFormatInfo ResolveVertexFormat(bool must_emulate_scaled_formats,
                                                   u32 nvn_format);

/// Rewrites a draw's state into the form that is keyed, compared and stored, for `mode`:
///   Static  - `vertex_types` zero, the layout words kept (unused attributes and streams cleared);
///   Dynamic - `vertex_types[k]` = the Shader::AttributeType of the NVNformat of attribute k
///             (Disabled for unused / non-vertex formats), `attributes`, `stream_stride` and
///             `stream_divisor` zero, so only what the shader translation depends on remains.
/// Then `Disk::CanonicalizeState(state, mode)`. `state` must be a draw request (raw layout words,
/// `vertex_types` ignored). `must_emulate_scaled_formats` as for ResolveVertexFormat.
void MakeKeyState(GraphicsFixedState& state, VertexInputMode mode,
                  bool must_emulate_scaled_formats);

/// The type class the vertex shader is translated for at `location`, read from a key-form state.
[[nodiscard]] Shader::AttributeType VertexAttributeType(const GraphicsFixedState& key_state,
                                                        VertexInputMode mode,
                                                        bool must_emulate_scaled_formats,
                                                        u32 location);

/// Fills `out` from the raw layout words of `layout` (a draw request or a Static key form).
/// Everything that cannot be represented is described in `issues`.
void BuildVertexInput(const Device& device, const GraphicsFixedState& layout,
                      u32 attributes_read, VertexInput& out, std::vector<std::string>& issues);

[[nodiscard]] Shader::InputTopology ToInputTopology(u32 nvn_topology);
/// nullopt: the alpha test is off (ALWAYS or an invalid value).
[[nodiscard]] std::optional<Shader::CompareFunction> ToAlphaTestFunction(u32 nvn_alpha_func);
/// Output type of a color attachment for the fragment shader.
[[nodiscard]] Shader::AttributeType ColorOutputType(VkFormat format);
/// Attachment 0 blends with an SRC1_* function.
[[nodiscard]] bool UsesDualSourceBlend(const GraphicsFixedState& state);

/// Everything of a draw's fixed state that the translation of its graphics stages reads
/// (PipelineCache MakeRuntimeInfo reads nothing else of the state or the targets): draws with
/// the same programs, the same TranslationState and the same texture answers get the same
/// SPIR-V. Plain bytes without padding (hashed, compared with memcmp); zero it before filling.
struct TranslationState {
    u32 input_topology;  ///< Shader::InputTopology of the draw's topology
    u32 point_size_bits; ///< GraphicsFixedState::point_size_bits
    u32 alpha_func;      ///< Shader::CompareFunction + 1 of the alpha test; 0 = no alpha test
    u32 alpha_ref_bits;  ///< GraphicsFixedState::alpha_ref_bits
    u8 convert_depth_mode; ///< 1: depth_mode is NVN's -w..w (GL) range
    u8 dual_source_blend;  ///< UsesDualSourceBlend
    std::array<u8, MAX_COLOR_TARGETS> color_output_types; ///< Shader::AttributeType per target
    std::array<u8, MAX_VERTEX_ATTRIBUTES> vertex_types;   ///< Shader::AttributeType per location
    std::array<u8, 6> pad;
};
static_assert(sizeof(TranslationState) == 64, "TranslationState must have no padding");
static_assert(std::is_trivially_copyable_v<TranslationState>);

/// The TranslationState of a draw whose state is in key form (MakeKeyState) for `mode`.
[[nodiscard]] TranslationState MakeTranslationState(const GraphicsFixedState& key_state,
                                                    const RenderPassKey& targets,
                                                    VertexInputMode mode,
                                                    bool must_emulate_scaled_formats);

/// The cull mode, front face, depth bias enable and depth/stencil values a pipeline of `state` is
/// drawn with, the EDEN_NX2_NVN_* diagnostic overrides (nvn_debug.h) applied. The one source of
/// these values for the static pipeline state and for the dynamic state the replay sets.
[[nodiscard]] DynamicFixedState MakeDynamicFixedState(const GraphicsFixedState& state);

/// Depth/stencil aspect helpers on VkFormat (render pass construction).
[[nodiscard]] bool FormatHasDepth(VkFormat format);
[[nodiscard]] bool FormatHasStencil(VkFormat format);

struct GraphicsPipelineParams {
    const GraphicsFixedState* state{};
    const RenderPassKey* targets{};
    VkRenderPass render_pass{};
    VkPipelineLayout layout{};
    std::span<const VkPipelineShaderStageCreateInfo> stages;
    u32 viewport_count{1};
    u32 attributes_read{};         ///< vertex shader generic input locations
    u32 color_outputs_written{};   ///< fragment shader outputs
    bool has_tessellation{};
    bool mesh{}; ///< task/mesh stages: no vertex input or input assembly state
    VertexInputMode vertex_input{VertexInputMode::Static};
    /// Declare the DynamicFixedState states dynamic (VK_DYNAMIC_STATE_CULL_MODE ...
    /// DEPTH_BIAS_ENABLE) and leave their create-info fields at fixed values, so that pipelines
    /// differing only in them get equal pre-rasterization and fragment shader library keys. Needs
    /// VK_EXT_extended_dynamic_state and VK_EXT_extended_dynamic_state2.
    bool dynamic_fixed{};
};

/// A VkGraphicsPipelineCreateInfo and the state structures it points to (not copyable: `ci`
/// points into the object). `ci.pStages` borrows GraphicsPipelineParams::stages and what it
/// references: the caller keeps those alive until the pipeline creation call returns.
struct GraphicsPipelineDescription {
    GraphicsPipelineDescription() = default;
    GraphicsPipelineDescription(const GraphicsPipelineDescription&) = delete;
    GraphicsPipelineDescription& operator=(const GraphicsPipelineDescription&) = delete;

    boost::container::static_vector<VkVertexInputBindingDescription, MAX_VERTEX_STREAMS>
        vertex_bindings;
    boost::container::static_vector<VkVertexInputBindingDivisorDescriptionEXT, MAX_VERTEX_STREAMS>
        divisors;
    boost::container::static_vector<VkVertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES>
        vertex_attributes;
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    VkPipelineVertexInputDivisorStateCreateInfoEXT divisor{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    VkPipelineTessellationStateCreateInfo tessellation{};
    VkPipelineViewportDepthClipControlCreateInfoEXT depth_clip_control{};
    VkPipelineViewportStateCreateInfo viewport{};
    VkPipelineRasterizationStateCreateInfo rasterization{};
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex{};
    VkSampleMask sample_mask{};
    VkPipelineMultisampleStateCreateInfo multisample{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    boost::container::static_vector<VkPipelineColorBlendAttachmentState, MAX_COLOR_TARGETS>
        blend_attachments;
    VkPipelineColorBlendStateCreateInfo color_blend{};
    boost::container::static_vector<VkDynamicState, 20> dynamic_states;
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    VkGraphicsPipelineCreateInfo ci{};
};

/// Fills `out` with the monolithic create info of the pipeline. Every state that cannot be
/// represented is described in `issues` (distinct reason per call; the caller counts and logs
/// them once).
void DescribeGraphicsPipeline(const Device& device, const GraphicsPipelineParams& params,
                              GraphicsPipelineDescription& out, std::vector<std::string>& issues);

/// Creates the pipeline DescribeGraphicsPipeline describes.
[[nodiscard]] vk::Pipeline CreateGraphicsPipeline(const Device& device, VkPipelineCache cache,
                                                  const GraphicsPipelineParams& params,
                                                  std::vector<std::string>& issues);

} // namespace Vulkan::Nvn
