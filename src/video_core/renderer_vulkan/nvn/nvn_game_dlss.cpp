// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <iterator>
#include <string_view>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_game_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"

namespace Vulkan::Nvn::GameDlss {
namespace {

// DLSS validates the render size against the mode's range: pick the mode for this ratio.
sl::DLSSMode ModeFor(const Request& r) {
    if (r.dlaa) {
        return sl::DLSSMode::eDLAA;
    }
    const f32 ratio = static_cast<f32>(r.color.rect_width) / static_cast<f32>(r.output.rect_width);
    if (ratio >= 0.99f) {
        return sl::DLSSMode::eDLAA;
    }
    if (ratio >= 0.65f) {
        return sl::DLSSMode::eMaxQuality;
    }
    if (ratio >= 0.57f) {
        return sl::DLSSMode::eBalanced;
    }
    if (ratio >= 0.49f) {
        return sl::DLSSMode::eMaxPerformance;
    }
    return sl::DLSSMode::eUltraPerformance;
}

// Same choice as the legacy path (vk_streamline_dlss.cpp): preset K unless EDEN_NX2_DLSS_PRESET.
sl::DLSSPreset Preset() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
    if (value && value[0] >= 'J' && value[0] <= 'M') {
        return static_cast<sl::DLSSPreset>(static_cast<u32>(sl::DLSSPreset::ePresetJ) +
                                           static_cast<u32>(value[0] - 'J'));
    }
    if (value && std::string_view{value} == "default") {
        return sl::DLSSPreset::eDefault;
    }
    return sl::DLSSPreset::ePresetK;
}

sl::float4x4 Identity() {
    sl::float4x4 matrix;
    matrix.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    matrix.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    matrix.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    matrix.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return matrix;
}

sl::Resource Resource(const Image& image) {
    sl::Resource resource{sl::ResourceType::eTex2d, image.image, nullptr, image.view,
                          static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL)};
    resource.width = image.width;
    resource.height = image.height;
    resource.nativeFormat = static_cast<uint32_t>(image.format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = image.usage;
    return resource;
}

sl::Extent Extent(const Image& image) {
    return sl::Extent{image.y, image.x, image.rect_width, image.rect_height};
}

PFun_slDLSSSetOptions* SetOptions() {
    static PFun_slDLSSSetOptions* const function = [] {
        void* found{};
        const sl::Result result = Streamline::GetApi().get_feature_function(
            sl::kFeatureDLSS, "slDLSSSetOptions", found);
        if (result != sl::Result::eOk || !found) {
            LOG_ERROR(Render_Vulkan, "Streamline's DLSS has no slDLSSSetOptions (result {})",
                      static_cast<int>(result));
        }
        return reinterpret_cast<PFun_slDLSSSetOptions*>(found);
    }();
    return function;
}

} // Anonymous namespace

bool Available() {
    return Streamline::IsDlssAvailable();
}

bool Evaluate(VkCommandBuffer cmd, const Request& r) {
    if (!Available()) {
        return false;
    }
    PFun_slDLSSSetOptions* const set_options = SetOptions();
    if (!set_options) {
        return false;
    }
    const Streamline::Api& api = Streamline::GetApi();
    const sl::ViewportHandle viewport{r.viewport};

    sl::DLSSOptions options{};
    options.mode = ModeFor(r);
    options.outputWidth = r.output.rect_width;
    options.outputHeight = r.output.rect_height;
    options.preExposure = r.pre_exposure;
    options.colorBuffersHDR = r.hdr_color ? sl::eTrue : sl::eFalse;
    options.useAutoExposure = r.auto_exposure ? sl::eTrue : sl::eFalse;
    const sl::DLSSPreset preset = Preset();
    options.dlaaPreset = preset;
    options.qualityPreset = preset;
    options.balancedPreset = preset;
    options.performancePreset = preset;
    options.ultraPerformancePreset = preset;
    options.ultraQualityPreset = preset;
    if (const sl::Result result = set_options(viewport, options); result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "[nvn] Streamline rejected the DLSS options (result {})",
                  static_cast<int>(result));
        return false;
    }
    sl::FrameToken* frame{};
    u32 index = r.frame_index;
    if (const sl::Result result = api.get_new_frame_token(frame, &index);
        result != sl::Result::eOk || !frame) {
        LOG_ERROR(Render_Vulkan, "[nvn] Streamline gave no frame token (result {})",
                  static_cast<int>(result));
        return false;
    }

    // DLSS super resolution only reads the jitter, motion and depth conventions; the camera
    // fields are for frame generation.
    sl::Constants constants{};
    constants.cameraViewToClip = Identity();
    constants.clipToCameraView = Identity();
    constants.clipToLensClip = Identity();
    constants.clipToPrevClip = Identity();
    constants.prevClipToClip = Identity();
    constants.jitterOffset = {r.jitter[0], r.jitter[1]};
    constants.mvecScale = {r.motion_scale[0], r.motion_scale[1]};
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.cameraPos = {0.0f, 0.0f, 0.0f};
    constants.cameraUp = {0.0f, 1.0f, 0.0f};
    constants.cameraRight = {1.0f, 0.0f, 0.0f};
    constants.cameraFwd = {0.0f, 0.0f, 1.0f};
    constants.cameraNear = 0.1f;
    constants.cameraFar = 1000.0f;
    constants.cameraFOV = 1.0f;
    constants.cameraAspectRatio =
        static_cast<f32>(r.color.rect_width) / static_cast<f32>(r.color.rect_height);
    constants.depthInverted = r.depth_inverted ? sl::eTrue : sl::eFalse;
    constants.cameraMotionIncluded = sl::eTrue;
    constants.motionVectors3D = sl::eFalse;
    constants.reset = r.reset ? sl::eTrue : sl::eFalse;
    constants.motionVectorsJittered = r.motion_vectors_jittered ? sl::eTrue : sl::eFalse;
    api.set_constants(constants, *frame, viewport);

    sl::Resource color = Resource(r.color);
    sl::Resource depth = Resource(r.depth);
    sl::Resource motion = Resource(r.motion_vectors);
    sl::Resource output = Resource(r.output);
    sl::Resource exposure = Resource(r.exposure);
    sl::Resource bias = Resource(r.bias);
    const sl::Extent color_area = Extent(r.color);
    const sl::Extent depth_area = Extent(r.depth);
    const sl::Extent motion_area = Extent(r.motion_vectors);
    const sl::Extent output_area = Extent(r.output);
    const sl::Extent exposure_area = Extent(r.exposure);
    const sl::Extent bias_area = Extent(r.bias);
    // Local tags: everything stays put until the evaluation is recorded.
    constexpr auto Life = sl::ResourceLifecycle::eValidUntilEvaluate;
    sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor, Life, &color_area};
    sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth, Life, &depth_area};
    sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors, Life, &motion_area};
    sl::ResourceTag output_tag{&output, sl::kBufferTypeScalingOutputColor, Life, &output_area};
    sl::ResourceTag exposure_tag{&exposure, sl::kBufferTypeExposure, Life, &exposure_area};
    sl::ResourceTag bias_tag{&bias, sl::kBufferTypeBiasCurrentColorHint, Life, &bias_area};
    const sl::BaseStructure* inputs[7] = {&viewport, &color_tag, &depth_tag, &motion_tag,
                                          &output_tag};
    u32 count = 5;
    if (r.exposure.image != VK_NULL_HANDLE) {
        inputs[count++] = &exposure_tag;
    }
    if (r.bias.image != VK_NULL_HANDLE) {
        inputs[count++] = &bias_tag;
    }
    const sl::Result result = api.evaluate_feature(sl::kFeatureDLSS, *frame, inputs, count,
                                                   reinterpret_cast<sl::CommandBuffer*>(cmd));
    if (result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "[nvn] NVIDIA DLSS evaluation failed (Streamline result {})",
                  static_cast<int>(result));
        return false;
    }
    return true;
}

void FreeViewport(u32 viewport) {
    if (Available()) {
        Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{viewport});
    }
}

} // namespace Vulkan::Nvn::GameDlss
