// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/settings_enums.h"
#include "video_core/frame_presentation_stats.h"
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

TEST_CASE("Original presents and generated presents stay separate across frame-gen toggles",
          "[frame_gen]") {
    using namespace std::chrono_literals;
    using VideoCore::FramePresentCounts;
    VideoCore::FramePresentSampler sampler;
    const auto start = VideoCore::FramePresentSampler::Clock::time_point{};
    CHECK_FALSE(sampler.Sample(FramePresentCounts{}, start));

    const auto off = sampler.Sample(FramePresentCounts{30, 0}, start + 1s);
    REQUIRE(off);
    CHECK(off->original == 30.0);
    CHECK(off->generated == 0.0);

    const auto on = sampler.Sample(FramePresentCounts{60, 26}, start + 2s);
    REQUIRE(on);
    CHECK(on->original == 30.0);
    CHECK(on->generated == 26.0);

    // In-flight frames may still present after a toggle or a fallback to original frames.
    const auto draining = sampler.Sample(FramePresentCounts{90, 27}, start + 3s);
    REQUIRE(draining);
    CHECK(draining->original == 30.0);
    CHECK(draining->generated == 1.0);
    const auto fallback = sampler.Sample(FramePresentCounts{120, 27}, start + 4s);
    REQUIRE(fallback);
    CHECK(fallback->original == 30.0);
    CHECK(fallback->generated == 0.0);

    // A new renderer or a backend without present counters cannot inherit stale rates.
    CHECK_FALSE(sampler.Sample(FramePresentCounts{1, 0}, start + 5s));
    CHECK_FALSE(sampler.Sample(std::nullopt, start + 6s));
    CHECK_FALSE(sampler.Sample(FramePresentCounts{15, 0}, start + 7s));
    CHECK(VideoCore::FrameCountRate(1, 2, 0.0) == 0.0);
    CHECK(VideoCore::FrameCountRate(2, 1, 1.0) == 0.0);
}
