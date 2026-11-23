// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#include "common/settings.h"
#include "video_core/renderer_vulkan/nvn/nvn_features.h"

namespace Vulkan::Nvn {

/// The values of the presentation settings the NVN HLE path looks at.
struct PresentSettingsView {
    Settings::ScalingFilter scaling_filter{Settings::ScalingFilter::NearestNeighbor};
    Settings::AntiAliasing anti_aliasing{Settings::AntiAliasing::None};
    bool post_shader_active{};
    bool frame_gen{};
    Settings::FrameGenType frame_gen_type{Settings::FrameGenType::Fsr3};
};

/// Whether the picture has to go through the emulated GPU's presentation passes (BlitScreen:
/// window adapting filter, anti-aliasing, post-processing chain) instead of the replay's plain
/// blit. A nearest or bilinear window filter without anti-aliasing and post-processing is what
/// the blit does (same sampling, same screen rectangle), so those keep the cheaper path.
[[nodiscard]] constexpr bool PresentNeedsPasses(const PresentSettingsView& view) {
    using Settings::ScalingFilter;
    return (view.scaling_filter != ScalingFilter::NearestNeighbor &&
            view.scaling_filter != ScalingFilter::Bilinear) ||
           view.anti_aliasing != Settings::AntiAliasing::None || view.post_shader_active;
}

/// Settings still absent from the replay presentation path. Window filters are applied by
/// RecordWindowFilter; anti-aliasing and the post-processing chain remain separate passes.
[[nodiscard]] inline std::vector<std::string> SettingsNotApplied(const PresentSettingsView& view) {
    std::vector<std::string> result;
    if (view.anti_aliasing != Settings::AntiAliasing::None || view.post_shader_active) {
        result.emplace_back("Anti-aliasing or post-processing "
                            "(not available on VTX-HLE)");
    }
    if (view.frame_gen && !FrameGenTypeSupported(view.frame_gen_type)) {
        result.emplace_back("Frame Generation (not available on VTX-HLE)");
    }
    return result;
}

[[nodiscard]] inline PresentSettingsView CurrentPresentSettings() {
    PresentSettingsView view;
    view.scaling_filter = Settings::values.scaling_filter.GetValue();
    view.anti_aliasing = Settings::values.anti_aliasing.GetValue();
    view.post_shader_active = Settings::values.post_shader_enabled.GetValue() &&
                              !Settings::values.post_shader_chain.GetValue().empty();
    view.frame_gen = Settings::values.frame_gen.GetValue();
    view.frame_gen_type = Settings::values.frame_gen_type.GetValue();
    return view;
}

} // namespace Vulkan::Nvn
