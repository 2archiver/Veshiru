// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <span>

namespace Tegra::Texture::ASTC {

void Decompress(std::span<const uint8_t> data, uint32_t width, uint32_t height, uint32_t depth,
                uint32_t block_width, uint32_t block_height, std::span<uint8_t> output);

/// True if no 16-byte block in `data` decodes to alpha below 1.
bool IsOpaque(std::span<const uint8_t> data);

/// What the blocks of an ASTC texture may decode to (ClassifyChannels). The bits are conservative:
/// a set bit means a stand-in must preserve that channel, not that a decoded texel necessarily
/// uses it. Bit 1: some texel may have R, G and B unequal; bit 0: some texel may have alpha below
/// 1 (set from the endpoint mode alone, e.g. for mode 4 even when both endpoints are opaque).
enum class Channels : uint8_t {
    Luminance = 0,      ///< R = G = B and alpha 1 everywhere: luminance endpoints only
    LuminanceAlpha = 1, ///< R = G = B everywhere: luminance and luminance+alpha endpoints
    Rgb = 2,            ///< alpha 1 everywhere
    Rgba = 3,           ///< anything, also HDR endpoints and blocks that decode to the error colour
};

/// Classifies the 16-byte blocks of `data` (footprint `block_width` x `block_height`) by their
/// colour endpoint modes and void-extent colours; stops at the first block that makes it Rgba.
/// All-zero blocks (block-linear padding) are skipped; `empty` (optional) is set when every
/// block was one (the result is then Luminance).
[[nodiscard]] Channels ClassifyChannels(std::span<const uint8_t> data, uint32_t block_width,
                                        uint32_t block_height, bool* empty = nullptr);

} // namespace Tegra::Texture::ASTC
