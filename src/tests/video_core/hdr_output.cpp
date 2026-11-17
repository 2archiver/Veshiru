// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/present/hdr_convert.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

using Settings::HdrSwapchain;
using Vulkan::ChooseHdrSurfaceFormat;

namespace {

constexpr VkSurfaceFormatKHR Sdr{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
constexpr VkSurfaceFormatKHR Hdr10{VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                   VK_COLOR_SPACE_HDR10_ST2084_EXT};
constexpr VkSurfaceFormatKHR Scrgb{VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};

bool Same(const std::optional<VkSurfaceFormatKHR>& a, const VkSurfaceFormatKHR& b) {
    return a && a->format == b.format && a->colorSpace == b.colorSpace;
}

} // Anonymous namespace

TEST_CASE("HDR output takes the preferred pair the surface lists", "[hdr_output]") {
    const std::array both{Sdr, Scrgb, Hdr10};
    REQUIRE(Same(ChooseHdrSurfaceFormat(both, HdrSwapchain::Hdr10), Hdr10));
    REQUIRE(Same(ChooseHdrSurfaceFormat(both, HdrSwapchain::Scrgb), Scrgb));
}

TEST_CASE("HDR output falls back to the other listed pair", "[hdr_output]") {
    const std::array scrgb_only{Sdr, Scrgb};
    REQUIRE(Same(ChooseHdrSurfaceFormat(scrgb_only, HdrSwapchain::Hdr10), Scrgb));
    const std::array hdr10_only{Sdr, Hdr10};
    REQUIRE(Same(ChooseHdrSurfaceFormat(hdr10_only, HdrSwapchain::Scrgb), Hdr10));
}

TEST_CASE("HDR output needs the exact format and colour space pair", "[hdr_output]") {
    // A 10-bit SDR format or an HDR10 colour space on another format is no HDR10 pair.
    const std::array mismatched{
        Sdr,
        VkSurfaceFormatKHR{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
        VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_HDR10_ST2084_EXT},
    };
    REQUIRE_FALSE(ChooseHdrSurfaceFormat(mismatched, HdrSwapchain::Hdr10));
    REQUIRE_FALSE(ChooseHdrSurfaceFormat(std::array{Sdr}, HdrSwapchain::Scrgb));
}

TEST_CASE("Layers are converted only where their encoding differs from the frame's",
          "[hdr_output]") {
    using Vulkan::HdrEncoding;
    using Vulkan::HdrPresent;
    Tegra::FramebufferConfig layer{};
    const HdrPresent sdr{};
    const HdrPresent hdr10{.color_space = VK_COLOR_SPACE_HDR10_ST2084_EXT};
    const HdrPresent scrgb{.color_space = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};

    layer.color_space = Tegra::ColorSpace::RGB709;
    REQUIRE_FALSE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, sdr)));
    const auto sdr_on_hdr10 = Vulkan::MakeHdrConvertParams(layer, hdr10);
    REQUIRE(Vulkan::NeedsHdrConvert(sdr_on_hdr10));
    REQUIRE(sdr_on_hdr10.input == HdrEncoding::Srgb);
    REQUIRE(sdr_on_hdr10.output == HdrEncoding::Pq);

    layer.color_space = Tegra::ColorSpace::RGB2020_PQ;
    REQUIRE_FALSE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, hdr10)));
    REQUIRE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, sdr)));
    REQUIRE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, scrgb)));

    layer.color_space = Tegra::ColorSpace::RGB709_Linear;
    layer.linear_unit_luminance = 0; // unknown: the scRGB 80 nits
    REQUIRE_FALSE(Vulkan::NeedsHdrConvert(Vulkan::MakeHdrConvertParams(layer, scrgb)));
    layer.linear_unit_luminance = 100;
    const auto rescaled = Vulkan::MakeHdrConvertParams(layer, scrgb);
    REQUIRE(Vulkan::NeedsHdrConvert(rescaled));
    REQUIRE(rescaled.nits_per_unit == 100.0f);
}

TEST_CASE("OpenFrameGen scRGB HDR normalization preserves bright highlights without clipping",
          "[hdr_output]") {
    // Mirrors open_frame_gen_prep.comp NormalizeSample() for VK_FORMAT_R16G16B16A16_SFLOAT.
    auto normalize_scrgb = [](float c) -> float {
        const float pos = c > 0.0f ? c : 0.0f;
        return std::sqrt(pos / (1.0f + pos));
    };

    // Negative wide-gamut scRGB values map cleanly to 0 without NaNs.
    CHECK(normalize_scrgb(-0.25f) == 0.0f);
    CHECK(normalize_scrgb(0.0f) == 0.0f);

    // Strictly monotonic across SDR white (1.0 == 80 nits), 1000 nits (12.5), and 4000 nits (50.0),
    // never collapsing highlights > 1.0 to the same flat white ceiling.
    const float v_mid = normalize_scrgb(0.5f);
    const float v_80nits = normalize_scrgb(1.0f);
    const float v_400nits = normalize_scrgb(5.0f);
    const float v_1000nits = normalize_scrgb(12.5f);
    const float v_4000nits = normalize_scrgb(50.0f);
    CHECK(v_mid > 0.0f);
    CHECK(v_80nits > v_mid);
    CHECK(v_400nits > v_80nits);
    CHECK(v_1000nits > v_400nits);
    CHECK(v_4000nits > v_1000nits);
    CHECK(v_4000nits < 1.0f);
}
