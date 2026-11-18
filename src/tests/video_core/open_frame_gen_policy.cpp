// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/settings_enums.h"
#include "video_core/renderer_vulkan/present/open_frame_gen_policy.h"

TEST_CASE("Open-Frame-Gen requires storage, filtered samples and dispatch capacity",
          "[frame_gen]") {
    constexpr VkFormatFeatureFlags2 flow = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
                                           VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                           VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
    constexpr auto output = flow | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                            VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
    VkPhysicalDeviceLimits limits{};
    limits.maxImageDimension2D = 4096;
    limits.maxComputeWorkGroupCount[0] = limits.maxComputeWorkGroupCount[1] = 65535;
    limits.maxComputeWorkGroupCount[2] = 65535;
    limits.maxComputeWorkGroupSize[0] = 16;
    limits.maxComputeWorkGroupSize[1] = 32;
    limits.maxComputeWorkGroupSize[2] = 4;
    limits.maxComputeWorkGroupInvocations = 256;
    REQUIRE(Vulkan::CanRunOpenFrameGen(output, flow, true, limits, {1920, 1080}));
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, flow, false, limits, {1920, 1080}));
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(flow, flow, true, limits, {1920, 1080}));
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT, true,
                                           limits, {1920, 1080}));
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, flow, true, limits, {1920, 0}));
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, flow, true, limits, {8192, 1080}));
    limits.maxComputeWorkGroupCount[2] = 127;
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, flow, true, limits, {1920, 1080}));
    limits.maxComputeWorkGroupCount[2] = 65535;
    limits.maxComputeWorkGroupCount[0] = 100;
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, flow, true, limits, {1920, 1080}));
    limits.maxComputeWorkGroupInvocations = 128;
    CHECK_FALSE(Vulkan::CanRunOpenFrameGen(output, flow, true, limits, {1280, 720}));
}

TEST_CASE("Open-Frame-Gen retains its persisted type name", "[frame_gen]") {
    using Settings::FrameGenType;
    CHECK(Settings::CanonicalizeEnum(FrameGenType::OpenFrameGen) == "OpenFrameGen");
    CHECK(Settings::ToEnum<FrameGenType>("OpenFrameGen") == FrameGenType::OpenFrameGen);
}

TEST_CASE("Open-Frame-Gen warp descriptors are scoped to a single image allocation",
          "[frame_gen]") {
    using Vulkan::OpenFrameGenWarpKey;
    constexpr OpenFrameGenWarpKey initial{41, 0};
    CHECK((initial == OpenFrameGenWarpKey{41, 0}));
    CHECK((initial != OpenFrameGenWarpKey{41, 1}));
    // A rebuilt image may recycle the same Vulkan handles, but still needs a new descriptor.
    CHECK((initial != OpenFrameGenWarpKey{42, 0}));
}
