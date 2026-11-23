// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The window-space rules of the NVN replay: where an NVN viewport / scissor rectangle lands in the
// Vulkan framebuffer and which winding is a front face. Pure functions (tests: tests/nvn/draw.cpp).
//
// NVN clip space is OpenGL's (+y up) and the first row of a render target is its top, whatever the
// window origin mode. A Vulkan viewport therefore always has a negative height (clip +y lands on
// the first row of the rectangle) and the winding stays the NVN one. The origin only decides where
// the rectangle sits: rows count from the top for UPPER_LEFT and from the bottom for LOWER_LEFT.
// Found on DK (UPPER_LEFT): with a positive height the G-buffer is stored upside down and text
// rendered to a texture is sampled flipped (the world banner).

#pragma once

#include <algorithm>

#include "common/common_types.h"

namespace Vulkan::Nvn {

struct ViewportRect {
    float x;
    float y;
    float width;
    float height;
};

/// `rect` in NVN window space (y counted from the origin). `invert_y` is the experiment knob
/// EDEN_NX2_NVN_FLIP_VIEWPORT (positive height).
[[nodiscard]] constexpr ViewportRect ToVulkanViewport(ViewportRect rect, float surface_height,
                                                      bool lower_left, bool invert_y = false) {
    const float top = lower_left ? surface_height - (rect.y + rect.height) : rect.y;
    if (invert_y) {
        return {rect.x, top, rect.width, rect.height};
    }
    return {rect.x, top + rect.height, rect.width, -rect.height};
}

/// First row of a scissor rectangle of `height` rows starting at `y` in NVN window space.
[[nodiscard]] constexpr s32 ScissorTopRow(s32 y, s32 height, s32 surface_height, bool lower_left) {
    return lower_left ? surface_height - (y + (std::max)(height, 0)) : y;
}

/// Vulkan `frontFace == COUNTER_CLOCKWISE` for the NVN front face (non-zero: counter-clockwise).
/// `invert` is the experiment knob EDEN_NX2_NVN_FACE=1.
[[nodiscard]] constexpr bool VulkanFrontFaceCcw(bool nvn_ccw, bool invert = false) {
    return nvn_ccw != invert;
}

} // namespace Vulkan::Nvn
