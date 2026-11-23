// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the draw path's diagnostics and fallback bookkeeping (nvn_debug.*,
// nvn_draw_resources.h): target dump conversion and the descriptor fallback reasons.

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tests/temp_path.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"

namespace {

using namespace Vulkan::Nvn;

std::vector<u8> ReadFile(const std::string& path) {
    std::ifstream file{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("nvn draw debug: every descriptor fallback reason has a name", "[nvn]") {
    std::set<std::string> names;
    for (u32 i = 0; i < static_cast<u32>(DescReason::Count); ++i) {
        const std::string name = DescReasonName(static_cast<DescReason>(i));
        REQUIRE(name != "unknown");
        REQUIRE(names.insert(name).second);
    }
    REQUIRE(static_cast<u32>(DescReason::Count) <= 16); // DrawCounters::fallbacks rows
}

TEST_CASE("nvn draw debug: a dump converts to a top-down TGA with statistics", "[nvn]") {
    const Tests::TempDir temp{"nvn-draw-debug-test"};
    const auto path = (temp.Path() / "dump.tga").string();
    // 2x2 RGBA8: red and half-transparent green, then blue and white.
    const std::array<u8, 16> pixels{255, 0,   0, 255, 0,   255, 0,   128,
                                    0,   0, 255, 255, 255, 255, 255, 255};
    Debug::PendingDump dump;
    dump.path = path;
    dump.width = 2;
    dump.height = 2;
    dump.format = VK_FORMAT_R8G8B8A8_UNORM;
    dump.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    const std::string summary = Debug::WriteTga(dump, pixels);
    REQUIRE(summary.find("nonzero=100.00%") != std::string::npos);

    const std::vector<u8> file = ReadFile(path);
    std::filesystem::remove(path);
    REQUIRE(file.size() == 18 + 4 * 4);
    REQUIRE(file[2] == 2);   // uncompressed true color
    REQUIRE(file[12] == 2);  // width
    REQUIRE(file[14] == 2);  // height
    REQUIRE(file[16] == 32); // bits per pixel
    REQUIRE((file[17] & 0x20) != 0); // top-down
    // BGRA order, top row first: red pixel first, the blue one starts the second row.
    REQUIRE(file[18] == 0);
    REQUIRE(file[19] == 0);
    REQUIRE(file[20] == 255);
    REQUIRE(file[26] == 255);
    REQUIRE(file[27] == 0);
    REQUIRE(file[28] == 0);
    // 8-bit colour is exact in the TGA: no .raw next to it.
    REQUIRE(!std::filesystem::exists(path + ".raw"));
}

TEST_CASE("nvn draw debug: float and packed-float formats decode", "[nvn]") {
    // A depth dump also writes "<path>.raw": the directory takes both away.
    const Tests::TempDir temp{"nvn-draw-debug-test"};
    const auto path = (temp.Path() / "dump.tga").string();
    // B10G11R11: red = 1.0 (6e5: exponent 15, mantissa 0 -> 0x3C0), green = blue = 0.
    const u32 packed = 0x3C0;
    std::array<u8, 4> pixels{};
    std::memcpy(pixels.data(), &packed, sizeof(packed));
    Debug::PendingDump dump;
    dump.path = path;
    dump.width = 1;
    dump.height = 1;
    dump.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    dump.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    const std::string summary = Debug::WriteTga(dump, pixels);
    REQUIRE(summary.find("mean=(1,0,0,") != std::string::npos);
    const std::vector<u8> file = ReadFile(path);
    std::filesystem::remove(path);
    REQUIRE(file.size() == 22);
    REQUIRE(file[20] == 255); // red channel (BGRA)

    // All three fields at once: red 0.5 (E5M6 exponent 14), green 1.5 (exponent 15, mantissa
    // 32/64), blue 3.0 (E5M5 exponent 16, mantissa 16/32). Red is bits 0-10, green 11-21, blue
    // 22-31 (VK_FORMAT_B10G11R11_UFLOAT_PACK32): a swapped red/blue or a wrong blue mantissa
    // width shows up as a different mean.
    const u32 mixed = 0x380u | (0x3E0u << 11) | (0x210u << 22);
    std::memcpy(pixels.data(), &mixed, sizeof(mixed));
    REQUIRE(Debug::WriteTga(dump, pixels).find("mean=(0.5,1.5,3,") != std::string::npos);
    // Subnormals: red mantissa 1 = 2^-20, blue mantissa 1 = 2^-19 (their own mantissa widths).
    const u32 tiny = 1u | (1u << 22);
    std::memcpy(pixels.data(), &tiny, sizeof(tiny));
    const std::string tiny_summary = Debug::WriteTga(dump, pixels);
    REQUIRE(tiny_summary.find("mean=(9.53674e-07,0,1.90735e-06,") != std::string::npos);
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".raw");

    // Depth: D32 value 0.5 shows as gray and has a range.
    const float depth = 0.5f;
    std::memcpy(pixels.data(), &depth, sizeof(depth));
    dump.format = VK_FORMAT_D32_SFLOAT;
    dump.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    REQUIRE(Debug::PixelBytes(dump.format, dump.aspect) == 4);
    REQUIRE(Debug::WriteTga(dump, pixels).find("r_range=[0.5,0.5]") != std::string::npos);
    std::filesystem::remove(path);

    // R32G32 float (a 2-channel target OoT renders at 216x120): blue 0, 8 bytes a pixel. The TGA
    // clamps 41 to 255, so an unscaled float dump keeps the exact texels in "<path>.raw".
    const std::array<float, 2> rg{0.25f, 41.0f};
    std::array<u8, 8> rg_pixels{};
    std::memcpy(rg_pixels.data(), rg.data(), sizeof(rg));
    dump.format = VK_FORMAT_R32G32_SFLOAT;
    dump.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    REQUIRE(Debug::PixelBytes(dump.format, dump.aspect) == 8);
    REQUIRE(Debug::WriteTga(dump, rg_pixels).find("mean=(0.25,41,0,1)") != std::string::npos);
    const std::vector<u8> raw = ReadFile(path + ".raw");
    REQUIRE(raw == std::vector<u8>(rg_pixels.begin(), rg_pixels.end()));
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".raw");

    // Formats without a conversion are refused, not guessed.
    dump.format = VK_FORMAT_R32G32B32_SFLOAT;
    dump.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    REQUIRE(Debug::PixelBytes(dump.format, dump.aspect) == 0);
    REQUIRE(Debug::WriteTga(dump, pixels) == "unconvertible");
}

TEST_CASE("nvn draw debug: a thumbnail dump is box-downscaled, statistics cover every pixel",
          "[nvn]") {
    const Tests::TempDir temp{"nvn-draw-debug-test"};
    const auto path = (temp.Path() / "dump.tga").string();
    // 6x2 RGBA8 with a width cap of 2: factor ceil(6/2) = 3, so the TGA is 2x1 (the height
    // 2/3 clamps to 1) and holds the averages of the two 3x2 boxes.
    std::array<u8, 6 * 2 * 4> pixels{};
    // Red in 4 of the left box's 6 pixels and 2 of the right box's: averages 170 and 85, which
    // no single sample of either box gives.
    for (const u32 i : {0u, 1u, 2u, 6u, 5u, 11u}) {
        pixels[i * 4 + 0] = 255;
    }
    for (u32 i = 0; i < 12; ++i) {
        pixels[i * 4 + 3] = 255;
    }
    Debug::PendingDump dump;
    dump.path = path;
    dump.width = 6;
    dump.height = 2;
    dump.format = VK_FORMAT_R8G8B8A8_UNORM;
    dump.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    dump.thumb_width = 2;
    const std::string summary = Debug::WriteTga(dump, pixels);
    REQUIRE(summary.find("mean=(0.5,0,0,1)") != std::string::npos); // 6 of 12 pixels
    REQUIRE(summary.find("nonzero=50.00%") != std::string::npos);
    const std::vector<u8> file = ReadFile(path);
    REQUIRE(file.size() == 18 + 2 * 4);
    REQUIRE(file[12] == 2);
    REQUIRE(file[14] == 1);
    REQUIRE(file[20] == 170); // the left box's average red (BGRA)
    REQUIRE(file[24] == 85);  // the right box's

    // Depth thumbnails keep no .raw file.
    const float depth = 0.25f;
    std::array<u8, 6 * 2 * 4> depth_pixels{};
    for (u32 i = 0; i < 12; ++i) {
        std::memcpy(depth_pixels.data() + i * 4, &depth, sizeof(depth));
    }
    dump.format = VK_FORMAT_D32_SFLOAT;
    dump.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    REQUIRE(Debug::WriteTga(dump, depth_pixels).find("mean=(0.25,0.25,0.25") != std::string::npos);
    REQUIRE(!std::filesystem::exists(path + ".raw"));
}
