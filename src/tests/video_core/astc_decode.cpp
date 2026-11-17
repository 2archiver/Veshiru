// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/textures/astc.h"

namespace {

using Block = std::array<u8, 16>;

// Builds a valid 2D LDR void-extent ASTC block (constant RGBA16 unorm color).
Block MakeVoidExtentLDR(u16 r16, u16 g16, u16 b16, u16 a16) {
    Block blk{};
    // Bits [0..8] = 0x1FC (void-extent), bit 9 = 0 (LDR), bits 10..11 = 1 (reserved),
    // bits 12..63 = all 1s (no void-extent coordinate bounds).
    blk[0] = 0xFC;
    blk[1] = 0xFD;
    for (size_t i = 2; i < 8; ++i) {
        blk[i] = 0xFF;
    }
    blk[8] = static_cast<u8>(r16 & 0xFF);
    blk[9] = static_cast<u8>(r16 >> 8);
    blk[10] = static_cast<u8>(g16 & 0xFF);
    blk[11] = static_cast<u8>(g16 >> 8);
    blk[12] = static_cast<u8>(b16 & 0xFF);
    blk[13] = static_cast<u8>(b16 >> 8);
    blk[14] = static_cast<u8>(a16 & 0xFF);
    blk[15] = static_cast<u8>(a16 >> 8);
    return blk;
}

// Builds a valid 2D HDR void-extent ASTC block (constant RGBA16F IEEE half-float color).
Block MakeVoidExtentHDR(u16 r_half, u16 g_half, u16 b_half, u16 a_half) {
    Block blk = MakeVoidExtentLDR(r_half, g_half, b_half, a_half);
    blk[1] = 0xFF; // Set bit 9 = 1 (HDR void extent), keeping bits 10..11 = 1.
    return blk;
}

// Writes `bits` LSBs of `val` into `blk` starting at bit offset `bit_pos` (little-endian).
void PutBitsLE(Block& blk, u32& bit_pos, u32 val, u32 bits) {
    for (u32 i = 0; i < bits; ++i) {
        if ((val >> i) & 1U) {
            blk[bit_pos >> 3] |= static_cast<u8>(1U << (bit_pos & 7U));
        }
        ++bit_pos;
    }
}

// Writes `bits` LSBs of `val` into the reversed weight stream starting from bit 127 downwards.
void PutWeightBitsFromTop(Block& blk, u32& weight_bit_pos, u32 val, u32 bits) {
    for (u32 i = 0; i < bits; ++i) {
        if ((val >> i) & 1U) {
            const u32 phys_bit = 127U - weight_bit_pos;
            blk[phys_bit >> 3] |= static_cast<u8>(1U << (phys_bit & 7U));
        }
        ++weight_bit_pos;
    }
}

// Constructs a single-partition, single-plane ASTC block with a 4x4 weight grid (16 2-bit weights,
// max weight 3 -> values {0, 21, 43, 64}, packed weight size = 32 bits) and explicit CEM + 8-bit
// endpoint bytes (up to 8 endpoint bytes).
Block MakeSinglePartition4x4Block(u32 cem, std::span<const u8> endpoints,
                                  const std::array<u8, 16>& weights_2bit) {
    Block blk{};
    // Layout 0 (bits 0..1 != 0, bits 2..3 = 00): W = B + 4, H = A + 2,
    // R = ((modeBits & 3) << 1) | ((modeBits >> 4) & 1).
    // Want W=4 (B=0), H=4 (A=2), R=4 (bits0..1 = 0b10, bit4 = 0 -> maxWeights[4 - 2] = 3 ->
    // 2 bits per weight, 16 weights = 32 packed weight bits).
    constexpr u32 mode_bits = (0b10U) | (2U << 5) | (0U << 7); // 0x042
    u32 pos = 0;
    PutBitsLE(blk, pos, mode_bits, 11);
    PutBitsLE(blk, pos, 0U, 2);   // nPartitions = 0 + 1 = 1
    PutBitsLE(blk, pos, cem, 4);  // CEM
    // Available color data bits = 128 - 17 - 32 = 79 bits (>= 8 bytes = 64 bits -> 8-bit unquantized
    // endpoint bytes for <= 8 endpoints!).
    for (u8 v : endpoints) {
        PutBitsLE(blk, pos, v, 8);
    }
    u32 wpos = 0;
    for (u8 w : weights_2bit) {
        PutWeightBitsFromTop(blk, wpos, w & 3U, 2);
    }
    return blk;
}

// Constructs a 2-partition, single-plane ASTC block with a 4x4 weight grid (2-bit weights) and
// uniform CEM across both partitions (baseMode == 0).
Block MakeTwoPartition4x4Block(u32 partition_index, u32 cem, std::span<const u8> endpoints,
                               const std::array<u8, 16>& weights_2bit) {
    Block blk{};
    constexpr u32 mode_bits = (0b10U) | (2U << 5) | (0U << 7); // W=4, H=4, maxWeight=3 (2-bit)
    u32 pos = 0;
    PutBitsLE(blk, pos, mode_bits, 11);
    PutBitsLE(blk, pos, 1U, 2);               // nPartitions = 1 + 1 = 2
    PutBitsLE(blk, pos, partition_index, 10); // 10-bit partition seed
    PutBitsLE(blk, pos, cem << 2, 6);         // baseMode = 0 (shared CEM = cem)
    // Remaining color data bits = 128 - 29 - 32 = 67 bits (holds 8 x 8-bit values cleanly).
    for (u8 v : endpoints) {
        PutBitsLE(blk, pos, v, 8);
    }
    u32 wpos = 0;
    for (u8 w : weights_2bit) {
        PutWeightBitsFromTop(blk, wpos, w & 3U, 2);
    }
    return blk;
}

// Constructs a single-partition DUAL-PLANE ASTC block with a 4x4 weight grid (1-bit weights per
// plane = 32 weights = 32 weight bits -> unquantized weights {0, 64}).
Block MakeDualPlane4x4Block(u32 cem, u32 plane_selector, std::span<const u8> endpoints,
                            const std::array<u8, 16>& w_plane0,
                            const std::array<u8, 16>& w_plane1) {
    Block blk{};
    // Layout 0, W=4 (B=0), H=4 (A=2), R=2 (bits0..1=0b01, bit4=0 -> maxWeight=1, 1 bit/weight),
    // H=0 (bit9=0), D=1 (bit10=1 -> dual plane). Total weight bits = 32.
    constexpr u32 mode_bits = (0b01U) | (2U << 5) | (1U << 10);
    u32 pos = 0;
    PutBitsLE(blk, pos, mode_bits, 11);
    PutBitsLE(blk, pos, 0U, 2);  // nPartitions = 1
    PutBitsLE(blk, pos, cem, 4); // CEM
    // Available color data bits = 128 - 17 - 2 (plane selector) - 32 (weights) = 77 bits.
    for (u8 v : endpoints) {
        PutBitsLE(blk, pos, v, 8);
    }
    // Plane selector sits at bits [128 - 32 - 2 .. 128 - 32 - 1] = [94..95].
    u32 ccs_pos = 128U - 32U - 2U;
    PutBitsLE(blk, ccs_pos, plane_selector & 3U, 2);

    u32 wpos = 0;
    for (size_t i = 0; i < 16; ++i) {
        PutWeightBitsFromTop(blk, wpos, w_plane0[i] & 1U, 1);
        PutWeightBitsFromTop(blk, wpos, w_plane1[i] & 1U, 1);
    }
    return blk;
}

} // namespace

TEST_CASE("ASTC integer LDR interpolation matches reference double formula exactly",
          "[video_core][astc]") {
    for (u32 c = 0; c <= 65535U; ++c) {
        const u32 ref =
            (c == 65535U) ? 255U
                          : static_cast<u32>(255.0 * (static_cast<double>(c) / 65536.0) + 0.5);
        const u32 fast = (c == 65535U) ? 255U : ((c * 255U + 32768U) >> 16);
        REQUIRE(fast == ref);
    }
}

TEST_CASE("ASTC LDR and HDR void-extent decode across representative footprints",
          "[video_core][astc]") {
    const std::array<std::pair<u32, u32>, 6> footprints{{
        {4, 4},
        {5, 5},
        {6, 6},
        {8, 8},
        {10, 10},
        {12, 12},
    }};

    const Block ldr_blk = MakeVoidExtentLDR(0x1234, 0x5678, 0x9ABC, 0xDEF0);
    // HDR FP16: R=1.0 (0x3C00 -> 255), G=0.5 (0x3800 -> 128), B=0.0 (0x0000 -> 0),
    // A=2.0 (0x4000 -> clamped to 255).
    const Block hdr_blk = MakeVoidExtentHDR(0x3C00, 0x3800, 0x0000, 0x4000);

    for (const auto& [bw, bh] : footprints) {
        CAPTURE(bw, bh);
        std::vector<u8> out(static_cast<size_t>(bw) * bh * 4, 0xCC);
        Tegra::Texture::ASTC::Decompress(ldr_blk, bw, bh, 1, bw, bh, out);
        for (size_t px = 0; px < static_cast<size_t>(bw) * bh; ++px) {
            CHECK(out[px * 4 + 0] == 0x12);
            CHECK(out[px * 4 + 1] == 0x56);
            CHECK(out[px * 4 + 2] == 0x9A);
            CHECK(out[px * 4 + 3] == 0xDE);
        }

        std::fill(out.begin(), out.end(), 0xCC);
        Tegra::Texture::ASTC::Decompress(hdr_blk, bw, bh, 1, bw, bh, out);
        for (size_t px = 0; px < static_cast<size_t>(bw) * bh; ++px) {
            CHECK(out[px * 4 + 0] == 255);
            CHECK(out[px * 4 + 1] == 128);
            CHECK(out[px * 4 + 2] == 0);
            CHECK(out[px * 4 + 3] == 255);
        }
    }
}

TEST_CASE("ASTC single-partition LDR direct RGBA and luminance modes", "[video_core][astc]") {
    // CEM 12 (LDR RGBA Direct): endpoints [R0, R1, G0, G1, B0, B1, A0, A1] =
    // (0, 255, 64, 192, 255, 0, 128, 255).
    // 2-bit weights: 0 -> w=0 (endpoint 0), 3 -> w=64 (endpoint 1).
    const std::array<u8, 8> rgba_ep{0, 255, 64, 192, 255, 0, 128, 255};
    std::array<u8, 16> weights{};
    for (size_t i = 0; i < 16; ++i) {
        weights[i] = (i & 1U) ? 3U : 0U;
    }
    const Block blk = MakeSinglePartition4x4Block(12U, rgba_ep, weights);

    std::vector<u8> out(4 * 4 * 4, 0);
    Tegra::Texture::ASTC::Decompress(blk, 4, 4, 1, 4, 4, out);
    for (size_t i = 0; i < 16; ++i) {
        if ((i & 1U) == 0) {
            CHECK(out[i * 4 + 0] == 0);
            CHECK(out[i * 4 + 1] == 64);
            CHECK(out[i * 4 + 2] == 255);
            CHECK(out[i * 4 + 3] == 128);
        } else {
            CHECK(out[i * 4 + 0] == 255);
            CHECK(out[i * 4 + 1] == 192);
            CHECK(out[i * 4 + 2] == 0);
            CHECK(out[i * 4 + 3] == 255);
        }
    }

    // CEM 0 (LDR Luminance Direct): endpoints [L0=32, L1=224], alpha always 255.
    const std::array<u8, 2> lum_ep{32, 224};
    const Block lum_blk = MakeSinglePartition4x4Block(0U, lum_ep, weights);
    Tegra::Texture::ASTC::Decompress(lum_blk, 4, 4, 1, 4, 4, out);
    for (size_t i = 0; i < 16; ++i) {
        const u8 expected = (i & 1U) ? 224 : 32;
        CHECK(out[i * 4 + 0] == expected);
        CHECK(out[i * 4 + 1] == expected);
        CHECK(out[i * 4 + 2] == expected);
        CHECK(out[i * 4 + 3] == 255);
    }
}

TEST_CASE("ASTC two-partition and dual-plane blocks decode distinct endpoints and planes",
          "[video_core][astc]") {
    // 2-partition CEM 0 (LDR Luminance Direct): 4 endpoint bytes [P0_L0=20, P0_L1=20, P1_L0=220,
    // P1_L1=220] so every texel in partition 0 is 20 and every texel in partition 1 is 220.
    const std::array<u8, 4> two_part_ep{20, 20, 220, 220};
    const std::array<u8, 16> zero_weights{};
    const Block two_part = MakeTwoPartition4x4Block(1U, 0U, two_part_ep, zero_weights);

    std::vector<u8> out(4 * 4 * 4, 0);
    Tegra::Texture::ASTC::Decompress(two_part, 4, 4, 1, 4, 4, out);
    size_t count_p0 = 0;
    size_t count_p1 = 0;
    for (size_t i = 0; i < 16; ++i) {
        const u8 r = out[i * 4 + 0];
        CHECK(out[i * 4 + 1] == r);
        CHECK(out[i * 4 + 2] == r);
        CHECK(out[i * 4 + 3] == 255);
        if (r == 20) {
            ++count_p0;
        } else if (r == 220) {
            ++count_p1;
        }
    }
    CHECK(count_p0 > 0);
    CHECK(count_p1 > 0);
    CHECK(count_p0 + count_p1 == 16);

    // Dual-plane CEM 12 (RGBA direct), CCS=3 (alpha uses plane 1; RGB uses plane 0).
    // Endpoints: R(0..200), G(10..210), B(20..220), A(30..230).
    // Plane 0 all 0 (RGB picks endpoint 0), Plane 1 all 1 (A picks endpoint 1 = 230).
    const std::array<u8, 8> dp_ep{0, 200, 10, 210, 20, 220, 30, 230};
    std::array<u8, 16> p0{};
    std::array<u8, 16> p1{};
    p1.fill(1U);
    const Block dual_blk = MakeDualPlane4x4Block(12U, 3U, dp_ep, p0, p1);
    Tegra::Texture::ASTC::Decompress(dual_blk, 4, 4, 1, 4, 4, out);
    for (size_t i = 0; i < 16; ++i) {
        CHECK(out[i * 4 + 0] == 0);
        CHECK(out[i * 4 + 1] == 10);
        CHECK(out[i * 4 + 2] == 20);
        CHECK(out[i * 4 + 3] == 230);
    }
}

TEST_CASE("ASTC non-multiple dimensions, multi-layer 3D, and threaded large decode",
          "[video_core][astc]") {
    // Non-multiple of 6x6 block size: 13x17x3 -> 3 cols x 3 rows x 3 slices = 27 blocks.
    constexpr u32 width = 13;
    constexpr u32 height = 17;
    constexpr u32 depth = 3;
    constexpr u32 bw = 6;
    constexpr u32 bh = 6;
    const u32 cols = (width + bw - 1) / bw;
    const u32 rows = (height + bh - 1) / bh;
    std::vector<u8> input(static_cast<size_t>(depth) * rows * cols * 16);
    for (u32 z = 0; z < depth; ++z) {
        for (u32 r = 0; r < rows; ++r) {
            for (u32 c = 0; c < cols; ++c) {
                const u8 rv = static_cast<u8>(z * 60 + r * 15 + c);
                const Block blk =
                    MakeVoidExtentLDR(static_cast<u16>(rv) << 8, static_cast<u16>(r * 40) << 8,
                                      static_cast<u16>(c * 50) << 8, 0xFF00);
                const size_t idx = (static_cast<size_t>(z) * rows * cols + r * cols + c) * 16;
                std::memcpy(input.data() + idx, blk.data(), 16);
            }
        }
    }

    std::vector<u8> out(static_cast<size_t>(depth) * height * width * 4, 0xAA);
    Tegra::Texture::ASTC::Decompress(input, width, height, depth, bw, bh, out);
    for (u32 z = 0; z < depth; ++z) {
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u32 r = y / bh;
                const u32 c = x / bw;
                const size_t px = (static_cast<size_t>(z) * height * width + y * width + x) * 4;
                CHECK(out[px + 0] == static_cast<u8>(z * 60 + r * 15 + c));
                CHECK(out[px + 1] == static_cast<u8>(r * 40));
                CHECK(out[px + 2] == static_cast<u8>(c * 50));
                CHECK(out[px + 3] == 255);
            }
        }
    }

    // Large 256x256 4x4 texture (4096 blocks > 256 inline threshold) exercises coarse worker-pool
    // row-band scheduling across multiple threads.
    constexpr u32 big_w = 256;
    constexpr u32 big_h = 256;
    constexpr u32 big_cols = big_w / 4;
    constexpr u32 big_rows = big_h / 4;
    std::vector<u8> big_in(static_cast<size_t>(big_cols) * big_rows * 16);
    for (u32 r = 0; r < big_rows; ++r) {
        for (u32 c = 0; c < big_cols; ++c) {
            const Block blk = MakeVoidExtentLDR(static_cast<u16>(c * 3) << 8,
                                                static_cast<u16>(r * 3) << 8, 0x7700, 0xFF00);
            std::memcpy(big_in.data() + (static_cast<size_t>(r) * big_cols + c) * 16, blk.data(),
                        16);
        }
    }
    std::vector<u8> big_out(static_cast<size_t>(big_w) * big_h * 4, 0);
    Tegra::Texture::ASTC::Decompress(big_in, big_w, big_h, 1, 4, 4, big_out);
    for (u32 y = 0; y < big_h; ++y) {
        for (u32 x = 0; x < big_w; ++x) {
            const size_t px = (static_cast<size_t>(y) * big_w + x) * 4;
            REQUIRE(big_out[px + 0] == static_cast<u8>((x / 4) * 3));
            REQUIRE(big_out[px + 1] == static_cast<u8>((y / 4) * 3));
            REQUIRE(big_out[px + 2] == 0x77);
            REQUIRE(big_out[px + 3] == 0xFF);
        }
    }
}

TEST_CASE("ASTC IsOpaque detects opaque and translucent blocks accurately",
          "[video_core][astc]") {
    const Block opaque_ldr = MakeVoidExtentLDR(0x1000, 0x2000, 0x3000, 0xFFFF);
    const Block alpha_ldr = MakeVoidExtentLDR(0x1000, 0x2000, 0x3000, 0x8000);
    const Block opaque_hdr = MakeVoidExtentHDR(0x3C00, 0x3C00, 0x3C00, 0x3C00); // A = 1.0
    const Block alpha_hdr = MakeVoidExtentHDR(0x3C00, 0x3C00, 0x3C00, 0x3800);  // A = 0.5

    CHECK(Tegra::Texture::ASTC::IsOpaque(opaque_ldr));
    CHECK_FALSE(Tegra::Texture::ASTC::IsOpaque(alpha_ldr));
    CHECK(Tegra::Texture::ASTC::IsOpaque(opaque_hdr));
    CHECK_FALSE(Tegra::Texture::ASTC::IsOpaque(alpha_hdr));

    // Opaque luminance block (CEM 0 has no alpha channel -> always opaque).
    const std::array<u8, 2> lum_ep{10, 200};
    const std::array<u8, 16> zero_w{};
    const Block lum_blk = MakeSinglePartition4x4Block(0U, lum_ep, zero_w);
    CHECK(Tegra::Texture::ASTC::IsOpaque(lum_blk));

    // RGBA direct block with non-255 alpha -> not opaque.
    const std::array<u8, 8> rgba_ep{0, 255, 0, 255, 0, 255, 128, 255};
    const Block alpha_blk = MakeSinglePartition4x4Block(12U, rgba_ep, zero_w);
    CHECK_FALSE(Tegra::Texture::ASTC::IsOpaque(alpha_blk));
}
