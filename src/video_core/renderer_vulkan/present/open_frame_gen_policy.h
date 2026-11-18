// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vulkan/vulkan_core.h>

namespace Vulkan {

/// Vulkan handles may be reused after a frame's image/view are destroyed.
struct OpenFrameGenWarpKey {
    uint64_t allocation_id{};
    uint32_t previous_slot{};
    bool operator==(const OpenFrameGenWarpKey&) const = default;
};

/// Open-Frame-Gen uses filtered samples of both history frames and the FP16 flow mask.
/// Its formatless warp writes to the same backing format used by the presentation frame.
constexpr bool CanRunOpenFrameGen(VkFormatFeatureFlags2 output, VkFormatFeatureFlags2 flow,
                                  bool formatless_writes, const VkPhysicalDeviceLimits& limits,
                                  VkExtent2D extent) {
    constexpr VkFormatFeatureFlags2 sample =
        VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    constexpr auto output_features = sample | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                                     VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                                     VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
    constexpr auto flow_features = sample | VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
    const auto groups = [](uint32_t size) { return size / 16 + (size % 16 != 0); };
    return formatless_writes && (output & output_features) == output_features &&
           (flow & flow_features) == flow_features && extent.width != 0 && extent.height != 0 &&
           extent.width <= limits.maxImageDimension2D &&
           extent.height <= limits.maxImageDimension2D &&
           limits.maxComputeWorkGroupCount[2] >= 128 &&
           limits.maxComputeWorkGroupInvocations >= 256 &&
           limits.maxComputeWorkGroupSize[0] >= 16 && limits.maxComputeWorkGroupSize[1] >= 32 &&
           limits.maxComputeWorkGroupSize[2] >= 4 &&
           groups(extent.width) <= limits.maxComputeWorkGroupCount[0] &&
           groups(extent.height) <= limits.maxComputeWorkGroupCount[1];
}

} // namespace Vulkan
