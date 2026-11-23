// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

namespace Vulkan {

/// A viewport coordinate of a rescaled render target; a downscale rounds to whole pixels.
[[nodiscard]] inline float ScaleViewportValue(float value, float scale) {
    float new_value = value * scale;
    if (scale < 1.0f) {
        const bool sign = std::signbit(value);
        new_value = std::round(std::abs(new_value));
        new_value = sign ? -new_value : new_value;
    }
    return new_value;
}

/// A scissor coordinate of a rescaled render target. A non-zero value never becomes zero.
[[nodiscard]] constexpr s32 ScaleScissorValue(s32 value, u32 up_scale, u32 down_shift) {
    if (value == 0) {
        return 0;
    }
    const s32 upset = value * static_cast<s32>(up_scale);
    s32 accumulated = 0;
    if ((up_scale >> down_shift) == 0) {
        accumulated = upset % 2;
    }
    const s32 converted_value = upset >> down_shift;
    return value < 0 ? std::min<s32>(converted_value - accumulated, -1)
                     : std::max<s32>(converted_value + accumulated, 1);
}

/// The 3D engine's surface clip, in the guest's pixels.
struct SurfaceClip {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
};

/// The viewport of a draw with the viewport transform disabled. Its positions are pixels of the
/// surface clip (PositionPass maps them to NDC by the guest's render area), so the viewport is the
/// surface clip, at the render target's scale like every other viewport: unscaled, a rescaled
/// target got only the guest-sized corner of the draw (DK at 2X lit only the top-left quarter).
[[nodiscard]] inline VkViewport UntransformedViewport(SurfaceClip clip, bool lower_left,
                                                      float scale) {
    const float x = ScaleViewportValue(static_cast<float>(clip.x), scale);
    float y = ScaleViewportValue(static_cast<float>(clip.y), scale);
    const float width =
        (std::max)(1.0f, ScaleViewportValue(static_cast<float>((std::max)(1u, clip.width)), scale));
    float height = (std::max)(
        1.0f, ScaleViewportValue(static_cast<float>((std::max)(1u, clip.height)), scale));
    if (lower_left) {
        y += height;
        height = -height;
    }
    return VkViewport{
        .x = x,
        .y = y,
        .width = width,
        .height = height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
}

/// The scissor of a draw with the viewport transform disabled: the surface clip, scaled like the
/// render target (see UntransformedViewport).
[[nodiscard]] constexpr VkRect2D UntransformedScissor(SurfaceClip clip, bool lower_left,
                                                      u32 up_scale, u32 down_shift) {
    s32 x = static_cast<s32>(clip.x);
    s32 y = static_cast<s32>(clip.y);
    const s32 width = static_cast<s32>((std::max)(1u, clip.width));
    const s32 height = static_cast<s32>((std::max)(1u, clip.height));
    if (lower_left) {
        y = static_cast<s32>(clip.height) - (y + height);
    }
    return VkRect2D{
        .offset = {ScaleScissorValue(x, up_scale, down_shift),
                   ScaleScissorValue(y, up_scale, down_shift)},
        .extent = {static_cast<u32>(ScaleScissorValue(width, up_scale, down_shift)),
                   static_cast<u32>(ScaleScissorValue(height, up_scale, down_shift))},
    };
}

} // namespace Vulkan
