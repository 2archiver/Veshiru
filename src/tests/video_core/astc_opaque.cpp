// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#include <catch2/catch_test_macros.hpp>

#include "video_core/textures/astc.h"

namespace {

// An LDR void extent block (one RGBA16 colour for every texel) with the given alpha.
std::array<uint8_t, 16> VoidExtent(uint16_t alpha) {
    // Mode 0x1FC, LDR, reserved bits 10-11 set, extent coordinates all ones (no extent).
    std::array<uint8_t, 16> block{0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                  0x00, 0x80, 0x00, 0x40, 0x00, 0xC0};
    block[14] = static_cast<uint8_t>(alpha);
    block[15] = static_cast<uint8_t>(alpha >> 8);
    return block;
}

uint8_t DecodedAlpha(std::span<const uint8_t> block) {
    std::array<uint8_t, 4 * 4 * 4> texels{};
    Tegra::Texture::ASTC::Decompress(block, 4, 4, 1, 4, 4, texels);
    return texels[3];
}

} // Anonymous namespace

TEST_CASE("ASTC opaque scan sends only blocks that decode to full alpha to BC1", "[video_core]") {
    const auto opaque = VoidExtent(0xFFFF);
    const auto masked = VoidExtent(0x8000);
    REQUIRE(DecodedAlpha(opaque) == 255);
    REQUIRE(DecodedAlpha(masked) < 255);

    CHECK(Tegra::Texture::ASTC::IsOpaque(opaque));
    CHECK_FALSE(Tegra::Texture::ASTC::IsOpaque(masked));

    std::array<uint8_t, 32> both{};
    std::copy(opaque.begin(), opaque.end(), both.begin());
    std::copy(masked.begin(), masked.end(), both.begin() + 16);
    CHECK_FALSE(Tegra::Texture::ASTC::IsOpaque(both));
}
