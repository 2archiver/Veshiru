// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA Streamline) for the game's own nvnCommandBufferEvaluateDlss on the
// NVN replay. Only built with EGLANTINE_STREAMLINE; see vulkan_common/streamline.h.

#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn::GameDlss {

/// One level of an image in GENERAL layout and the rectangle of it the evaluation uses.
struct Image {
    VkImage image{};
    VkImageView view{};
    VkFormat format{};
    VkImageUsageFlags usage{VK_IMAGE_USAGE_SAMPLED_BIT};
    u32 width{}; ///< of the level
    u32 height{};
    u32 x{};
    u32 y{};
    u32 rect_width{};
    u32 rect_height{};
};

struct Request {
    Image color;
    Image depth;
    Image motion_vectors;
    Image output;
    Image exposure; ///< optional (no image: none)
    Image bias;     ///< optional
    u32 viewport{}; ///< the game network's id
    u32 frame_index{};
    f32 jitter[2]{};
    f32 motion_scale[2]{1.0f, 1.0f}; ///< Streamline's mvecScale
    f32 pre_exposure{1.0f};
    bool dlaa{};
    bool reset{};
    bool depth_inverted{};
    bool hdr_color{};
    bool auto_exposure{};
    bool motion_vectors_jittered{};
};

[[nodiscard]] bool Available();

/// Records one evaluation on `cmd` (outside a render pass). DLSS binds its own pipelines and
/// descriptors there. False when nothing usable was recorded: the caller writes the output itself.
bool Evaluate(VkCommandBuffer cmd, const Request& request);

/// Frees what DLSS keeps for a network that is no longer evaluated.
void FreeViewport(u32 viewport);

} // namespace Vulkan::Nvn::GameDlss
