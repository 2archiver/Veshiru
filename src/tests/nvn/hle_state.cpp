// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/state_layouts.h"

// State-object family. The expected words are literal constants computed independently of the
// accessors: a script packs nxnative's `nvn_obj.h` bit-fields (blend_st, color_st, poly_st,
// ds_st, va_st; least-significant bit first, in declaration order) and prints the word.
namespace {

using NvnHleTest::Rig;
using namespace VideoCommon::Nvn;

constexpr u64 U32_MINUS_1 = 0xFFFF'FFFFull; // -1 as a caller that does not sign-extend passes it

// A slot filled with a recognizable pattern, to see what a body writes beyond its object.
void Fill(Rig& rig, u64 va, u8 value) {
    std::array<u8, 0x40> bytes;
    bytes.fill(value);
    rig.memory.Put(va, bytes);
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Header only
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state layouts: accessors against nxnative bit-field words", "[nvn][state]") {
    // Defaults, as written by the nvn*StateSetDefaults bodies.
    static_assert(BlendState::Defaults() == 0x4'8104'0410ull);
    static_assert(ColorState::Defaults() == 0x8300u);
    static_assert(ChannelMaskState::Defaults() == 0xFFFF'FFFFu);
    static_assert(PolygonState::Defaults() == 0x14u);
    static_assert(DepthStencilState::Defaults() == 0x8'8888'C40Aull);
    static_assert(VertexAttribState::Defaults() == 0u);

    // Field positions: one field set at a time, against the single-bit word.
    static_assert(BlendState::WithTargetIndex(0, 7) == 0x7ull);
    static_assert(BlendState::WithSrcRgbFunc(0, 0x7F) == 0x7Full << 3);
    static_assert(BlendState::WithDstRgbFunc(0, 0x7F) == 0x7Full << 10);
    static_assert(BlendState::WithSrcAlphaFunc(0, 0x7F) == 0x7Full << 17);
    static_assert(BlendState::WithDstAlphaFunc(0, 0x7F) == 0x7Full << 24);
    static_assert(BlendState::WithRgbEquation(0, 7) == 7ull << 31);
    static_assert(BlendState::WithAlphaEquation(0, 7) == 7ull << 34);
    static_assert(BlendState::WithAdvancedMode(0, 0x3F) == 0x3Full << 37);
    static_assert(BlendState::WithAdvancedNotPremultiplied(0, 1) == 1ull << 43);
    static_assert(BlendState::WithAdvancedOverlap(0, 3) == 3ull << 44);
    static_assert(BlendState::WithAdvancedNotNormalizedDst(0, 1) == 1ull << 46);

    static_assert(ColorState::WithBlendEnableMask(0, 0xFF) == 0xFFu);
    static_assert(ColorState::WithLogicOp(0, 0xF) == 0xF00u);
    static_assert(ColorState::WithAlphaFunc(0, 0xF) == 0xF000u);

    static_assert(PolygonState::WithCullFace(0, 3) == 0x3u);
    static_assert(PolygonState::WithFrontFace(0, 1) == 0x4u);
    static_assert(PolygonState::WithPolygonMode(0, 3) == 0x18u);
    static_assert(PolygonState::WithOffsetEnables(0, 7) == 0xE0u);

    static_assert(DepthStencilState::WithDepthTestEnable(0, 1) == 1ull);
    static_assert(DepthStencilState::WithDepthWriteEnable(0, 1) == 2ull);
    static_assert(DepthStencilState::WithDepthFunc(0, 0xF) == 0xFull << 2);
    static_assert(DepthStencilState::WithStencilTestEnable(0, 1) == 1ull << 6);
    static_assert(DepthStencilState::WithFrontStencilFunc(0, 0xF) == 0xFull << 7);
    static_assert(DepthStencilState::WithBackStencilFunc(0, 0xF) == 0xFull << 11);
    static_assert(DepthStencilState::WithFrontFailOp(0, 0xF) == 0xFull << 15);
    static_assert(DepthStencilState::WithFrontDepthFailOp(0, 0xF) == 0xFull << 19);
    static_assert(DepthStencilState::WithFrontDepthPassOp(0, 0xF) == 0xFull << 23);
    static_assert(DepthStencilState::WithBackFailOp(0, 0xF) == 0xFull << 27);
    static_assert(DepthStencilState::WithBackDepthFailOp(0, 0xF) == 0xFull << 31);
    static_assert(DepthStencilState::WithBackDepthPassOp(0, 0xF) == 0xFull << 35);

    static_assert(VertexAttribState::WithFormat(0, 0xFF) == 0xFFu);
    static_assert(VertexAttribState::WithOffset(0, 0x7FFFF) == 0x7FFFFu << 8);
    static_assert(VertexAttribState::WithStreamIndex(0, 0x1F) == 0x1Fu << 27);

    // Round trip through a full word: no field disturbs its neighbours.
    constexpr u64 blend = 0x2911'880E'182Dull;
    static_assert(BlendState::TargetIndex(blend) == 5);
    static_assert(BlendState::SrcRgbFunc(blend) == 5);
    static_assert(BlendState::DstRgbFunc(blend) == 6);
    static_assert(BlendState::SrcAlphaFunc(blend) == 7);
    static_assert(BlendState::DstAlphaFunc(blend) == 8);
    static_assert(BlendState::RgbEquation(blend) == 3);
    static_assert(BlendState::AlphaEquation(blend) == 4);
    static_assert(BlendState::AdvancedMode(blend) == 8);
    static_assert(BlendState::AdvancedNotPremultiplied(blend) == 1);
    static_assert(BlendState::AdvancedOverlap(blend) == 2);
    static_assert(BlendState::AdvancedNotNormalizedDst(blend) == 0);

    constexpr u64 ds = 0x3B'2A19'29D1ull;
    static_assert(DepthStencilState::DepthTestEnable(ds) == 1);
    static_assert(DepthStencilState::DepthWriteEnable(ds) == 0);
    static_assert(DepthStencilState::DepthFunc(ds) == 4);
    static_assert(DepthStencilState::StencilTestEnable(ds) == 1);
    static_assert(DepthStencilState::FrontStencilFunc(ds) == 3);
    static_assert(DepthStencilState::BackStencilFunc(ds) == 5);
    static_assert(DepthStencilState::FrontFailOp(ds) == 2);
    static_assert(DepthStencilState::FrontDepthFailOp(ds) == 3);
    static_assert(DepthStencilState::FrontDepthPassOp(ds) == 4);
    static_assert(DepthStencilState::BackFailOp(ds) == 5);
    static_assert(DepthStencilState::BackDepthFailOp(ds) == 6);
    static_assert(DepthStencilState::BackDepthPassOp(ds) == 7);

    constexpr u32 va = 0x1923'45A3;
    static_assert(VertexAttribState::Format(va) == 0xA3);
    static_assert(VertexAttribState::Offset(va) == 0x12345);
    static_assert(VertexAttribState::StreamIndex(va) == 3);

    // Per target accessors and their out-of-range rules.
    static_assert(ColorState::BlendEnabled(0x81, 0) && ColorState::BlendEnabled(0x81, 7));
    static_assert(!ColorState::BlendEnabled(0x81, 3) && !ColorState::BlendEnabled(0x81, 8) &&
                  !ColorState::BlendEnabled(0x81, -1));
    static_assert(ColorState::WithBlendEnabled(0, 8, true) == 0);
    static_assert(ColorState::WithBlendEnabled(0x81, 7, false) == 0x01);
    static_assert(ChannelMaskState::Channels(0xA000'0005u, 0) == 0x5);
    static_assert(ChannelMaskState::Channels(0xA000'0005u, 7) == 0xA);
    static_assert(ChannelMaskState::Channels(0xA000'0005u, 8) == 0);
    static_assert(ChannelMaskState::WithChannels(0xFFFF'FFFFu, 2, 0) == 0xFFFF'F0FFu);
    static_assert(ChannelMaskState::WithChannels(0xFFFF'FFFFu, 8, 0) == 0xFFFF'FFFFu);
    static_assert(ChannelMaskState::PackChannels(true, false, true, false) == 5);
    static_assert(ChannelMaskState::PackChannels(false, true, false, true) == 0xA);

    // Advanced mode sanitizing and the face rules of the depth-stencil state.
    static_assert(BlendState::SanitizeAdvancedMode(0x32) == 0x32);
    static_assert(BlendState::SanitizeAdvancedMode(0x33) == 0);
    static_assert(DepthStencilState::ReadsBackFace(2));
    static_assert(!DepthStencilState::ReadsBackFace(3) && !DepthStencilState::ReadsBackFace(1) &&
                  !DepthStencilState::ReadsBackFace(0));
    static_assert(DepthStencilState::WithStencilFunc(0, 0, 0xF) == 0);
    static_assert(DepthStencilState::WithStencilFunc(0, 4, 0xF) == 0);
    static_assert(DepthStencilState::WithStencilFunc(0, 1, 0xF) == 0xFull << 7);
    static_assert(DepthStencilState::WithStencilFunc(0, 2, 0xF) == 0xFull << 11);

    // The multisample bytes, read through the header accessors.
    constexpr std::array<u8, 11> ms{1, 4, 1, 1, 8, 2, 1, 3, 1, 1, 1};
    static_assert(MultisampleState::Enable(ms.data()) && MultisampleState::Samples(ms.data()) == 4);
    static_assert(MultisampleState::AlphaToCoverage(ms.data()) &&
                  MultisampleState::AlphaToCoverageDither(ms.data()));
    static_assert(MultisampleState::RasterSamples(ms.data()) == 8 &&
                  MultisampleState::CoverageModulationMode(ms.data()) == 2);
    static_assert(MultisampleState::CoverageToColorEnable(ms.data()) &&
                  MultisampleState::CoverageToColorOutput(ms.data()) == 3);
    static_assert(MultisampleState::SampleLocationsEnable(ms.data()) &&
                  MultisampleState::SampleLocationsGridEnable(ms.data()) &&
                  MultisampleState::CoverageTruncationEnable(ms.data()));

    // Sizes agree with the recorder's per-kind object sizes.
    STATIC_REQUIRE(BlendState::SIZE == STATE_OBJECT_SIZES[0]);
    STATIC_REQUIRE(ColorState::SIZE == STATE_OBJECT_SIZES[1]);
    STATIC_REQUIRE(ChannelMaskState::SIZE == STATE_OBJECT_SIZES[2]);
    STATIC_REQUIRE(MultisampleState::SIZE == STATE_OBJECT_SIZES[3]);
    STATIC_REQUIRE(PolygonState::SIZE == STATE_OBJECT_SIZES[4]);
    STATIC_REQUIRE(DepthStencilState::SIZE == STATE_OBJECT_SIZES[5]);

    // Literal words at run time too, so a failure names the value.
    CHECK(BlendState::Defaults() == 0x4'8104'0410ull);
    CHECK(DepthStencilState::Defaults() == 0x8'8888'C40Aull);
}

// ---------------------------------------------------------------------------------------------
// Blend state
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: blend state defaults and setters pack nxnative blend_st", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0xAA);

    rig.Call("nvnBlendStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u64>(s) == 0x4'8104'0410ull);
    CHECK(rig.memory.Get<u8>(s + 8) == 0xAA); // nothing beyond the 8 bytes

    rig.Call("nvnBlendStateSetBlendTarget", {Rig::I(s), Rig::I(5)});
    rig.Call("nvnBlendStateSetBlendFunc", {Rig::I(s), Rig::I(5), Rig::I(6), Rig::I(7), Rig::I(8)});
    rig.Call("nvnBlendStateSetBlendEquation", {Rig::I(s), Rig::I(3), Rig::I(4)});
    rig.Call("nvnBlendStateSetAdvancedMode", {Rig::I(s), Rig::I(8)});
    rig.Call("nvnBlendStateSetAdvancedOverlap", {Rig::I(s), Rig::I(2)});
    // NVNboolean: only the low byte counts, 0x100 is false, so the source is NOT premultiplied.
    rig.Call("nvnBlendStateSetAdvancedPremultipliedSrc", {Rig::I(s), Rig::I(0x100)});
    CHECK(rig.memory.Get<u64>(s) == 0x2911'880E'182Dull);
    CHECK(rig.memory.Get<u8>(s + 8) == 0xAA);

    CHECK(rig.Call("nvnBlendStateGetBlendTarget", {Rig::I(s)}) == 5);
    const u64 out = Rig::SCRATCH;
    rig.Call("nvnBlendStateGetBlendFunc",
             {Rig::I(s), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8), Rig::I(out + 12)});
    CHECK(rig.memory.Get<u32>(out) == 5);
    CHECK(rig.memory.Get<u32>(out + 4) == 6);
    CHECK(rig.memory.Get<u32>(out + 8) == 7);
    CHECK(rig.memory.Get<u32>(out + 12) == 8);
    rig.Call("nvnBlendStateGetBlendEquation", {Rig::I(s), Rig::I(out + 16), Rig::I(out + 20)});
    CHECK(rig.memory.Get<u32>(out + 16) == 3);
    CHECK(rig.memory.Get<u32>(out + 20) == 4);
    CHECK(rig.Call("nvnBlendStateGetAdvancedMode", {Rig::I(s)}) == 8);
    CHECK(rig.Call("nvnBlendStateGetAdvancedOverlap", {Rig::I(s)}) == 2);
    CHECK(rig.Call("nvnBlendStateGetAdvancedPremultipliedSrc", {Rig::I(s)}) == 0);
    rig.Call("nvnBlendStateSetAdvancedPremultipliedSrc", {Rig::I(s), Rig::I(1)});
    CHECK(rig.Call("nvnBlendStateGetAdvancedPremultipliedSrc", {Rig::I(s)}) == 1);

    // Null output pointers are skipped.
    rig.Call("nvnBlendStateGetBlendFunc", {Rig::I(s), Rig::I(0), Rig::I(0), Rig::I(0), Rig::I(0)});
    rig.Call("nvnBlendStateGetBlendEquation", {Rig::I(s), Rig::I(0), Rig::I(0)});
}

TEST_CASE("NVN state: blend state drops values above the bit-field widths", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    rig.Call("nvnBlendStateSetDefaults", {Rig::I(s)});

    rig.Call("nvnBlendStateSetBlendTarget", {Rig::I(s), Rig::I(0x1B)}); // 3 bits: 3
    rig.Call("nvnBlendStateSetBlendFunc",
             {Rig::I(s), Rig::I(0x85), Rig::I(0x86), Rig::I(0x87), Rig::I(0x88)}); // 7 bits
    rig.Call("nvnBlendStateSetBlendEquation", {Rig::I(s), Rig::I(0xB), Rig::I(0xC)}); // 3 bits
    rig.Call("nvnBlendStateSetAdvancedMode", {Rig::I(s), Rig::I(0x33)}); // above LUMINOSITY: 0
    rig.Call("nvnBlendStateSetAdvancedOverlap", {Rig::I(s), Rig::I(7)});  // 2 bits: 3
    rig.Call("nvnBlendStateSetAdvancedPremultipliedSrc", {Rig::I(s), Rig::I(0x01)});
    CHECK(rig.memory.Get<u64>(s) == 0x3011'880E'182Bull);
    CHECK(rig.Call("nvnBlendStateGetBlendTarget", {Rig::I(s)}) == 3);

    // A mode in range is kept, the last mode too.
    rig.Call("nvnBlendStateSetAdvancedMode", {Rig::I(s), Rig::I(0x32)});
    CHECK(rig.Call("nvnBlendStateGetAdvancedMode", {Rig::I(s)}) == 0x32);
    rig.Call("nvnBlendStateSetAdvancedMode", {Rig::I(s), Rig::I(0x8000'0000)});
    CHECK(rig.Call("nvnBlendStateGetAdvancedMode", {Rig::I(s)}) == 0);
    // A negative int target keeps its low 3 bits.
    rig.Call("nvnBlendStateSetBlendTarget", {Rig::I(s), Rig::I(U32_MINUS_1)});
    CHECK(rig.Call("nvnBlendStateGetBlendTarget", {Rig::I(s)}) == 7);
}

TEST_CASE("NVN state: blend state NVN2 normalized destination", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    rig.Call("nvnBlendStateSetDefaults", {Rig::I(s)});
    CHECK(rig.Call("nvnBlendStateGetAdvancedNormalizedDst", {Rig::I(s)}) == 1);
    rig.Call("nvnBlendStateSetAdvancedNormalizedDst", {Rig::I(s), Rig::I(0)});
    CHECK(rig.Call("nvnBlendStateGetAdvancedNormalizedDst", {Rig::I(s)}) == 0);
    // Stored in bit 46 on top of the nxnative layout; nothing else moves.
    CHECK(rig.memory.Get<u64>(s) == (0x4'8104'0410ull | 1ull << 46));
    rig.Call("nvnBlendStateSetAdvancedNormalizedDst", {Rig::I(s), Rig::I(1)});
    CHECK(rig.memory.Get<u64>(s) == 0x4'8104'0410ull);
    // SetDefaults clears it again.
    rig.Call("nvnBlendStateSetAdvancedNormalizedDst", {Rig::I(s), Rig::I(0)});
    rig.Call("nvnBlendStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u64>(s) == 0x4'8104'0410ull);
}

// ---------------------------------------------------------------------------------------------
// Color and channel mask state
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: color state packs nxnative color_st", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0x55);

    rig.Call("nvnColorStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u32>(s) == 0x8300);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x55);

    rig.Call("nvnColorStateSetBlendEnable", {Rig::I(s), Rig::I(0), Rig::I(1)});
    rig.Call("nvnColorStateSetBlendEnable", {Rig::I(s), Rig::I(3), Rig::I(1)});
    rig.Call("nvnColorStateSetBlendEnable", {Rig::I(s), Rig::I(7), Rig::I(0x101)});
    rig.Call("nvnColorStateSetBlendEnable", {Rig::I(s), Rig::I(3), Rig::I(0x100)}); // off again
    // Out-of-range targets change nothing.
    rig.Call("nvnColorStateSetBlendEnable", {Rig::I(s), Rig::I(8), Rig::I(1)});
    rig.Call("nvnColorStateSetBlendEnable", {Rig::I(s), Rig::I(U32_MINUS_1), Rig::I(1)});
    rig.Call("nvnColorStateSetLogicOp", {Rig::I(s), Rig::I(0x1F)}); // 4 bits: 0xF
    rig.Call("nvnColorStateSetAlphaTest", {Rig::I(s), Rig::I(2)});
    CHECK(rig.memory.Get<u32>(s) == 0x2F81);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x55);

    CHECK(rig.Call("nvnColorStateGetBlendEnable", {Rig::I(s), Rig::I(0)}) == 1);
    CHECK(rig.Call("nvnColorStateGetBlendEnable", {Rig::I(s), Rig::I(3)}) == 0);
    CHECK(rig.Call("nvnColorStateGetBlendEnable", {Rig::I(s), Rig::I(7)}) == 1);
    CHECK(rig.Call("nvnColorStateGetBlendEnable", {Rig::I(s), Rig::I(8)}) == 0);
    CHECK(rig.Call("nvnColorStateGetBlendEnable", {Rig::I(s), Rig::I(U32_MINUS_1)}) == 0);
    CHECK(rig.Call("nvnColorStateGetLogicOp", {Rig::I(s)}) == 0xF);
    CHECK(rig.Call("nvnColorStateGetAlphaTest", {Rig::I(s)}) == 2);

    rig.Call("nvnColorStateSetAlphaTest", {Rig::I(s), Rig::I(0x1A)}); // 4 bits: 0xA
    CHECK(rig.Call("nvnColorStateGetAlphaTest", {Rig::I(s)}) == 0xA);
    CHECK(rig.memory.Get<u32>(s) == 0xAF81);
}

TEST_CASE("NVN state: channel mask state packs one nibble per target", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0x55);

    rig.Call("nvnChannelMaskStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u32>(s) == 0xFFFF'FFFFu);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x55);

    rig.Call("nvnChannelMaskStateSetChannelMask",
             {Rig::I(s), Rig::I(0), Rig::I(1), Rig::I(0), Rig::I(1), Rig::I(0)});
    rig.Call("nvnChannelMaskStateSetChannelMask",
             {Rig::I(s), Rig::I(7), Rig::I(0x100), Rig::I(0x1FF), Rig::I(0), Rig::I(2)});
    rig.Call("nvnChannelMaskStateSetChannelMask",
             {Rig::I(s), Rig::I(2), Rig::I(0), Rig::I(0), Rig::I(0), Rig::I(0)});
    // Target 7: r off (0x100), g on (0x1FF), b off, a on (2): 0b1010 = 0xA. Target 0: r b.
    // Target 2 all off.
    CHECK(rig.memory.Get<u32>(s) == 0xAFFF'F0F5u);

    const u64 out = Rig::SCRATCH;
    rig.memory.Put<u32>(out, 0xFFFF'FFFFu);
    rig.Call("nvnChannelMaskStateGetChannelMask",
             {Rig::I(s), Rig::I(0), Rig::I(out), Rig::I(out + 1), Rig::I(out + 2), Rig::I(out + 3)});
    CHECK(rig.memory.Get<u8>(out) == 1);
    CHECK(rig.memory.Get<u8>(out + 1) == 0);
    CHECK(rig.memory.Get<u8>(out + 2) == 1);
    CHECK(rig.memory.Get<u8>(out + 3) == 0);
    rig.Call("nvnChannelMaskStateGetChannelMask",
             {Rig::I(s), Rig::I(7), Rig::I(out), Rig::I(out + 1), Rig::I(out + 2), Rig::I(out + 3)});
    CHECK(rig.memory.Get<u8>(out) == 0);
    CHECK(rig.memory.Get<u8>(out + 1) == 1);
    CHECK(rig.memory.Get<u8>(out + 2) == 0);
    CHECK(rig.memory.Get<u8>(out + 3) == 1);
    // Out of range reads all-false; a null pointer is skipped.
    rig.Call("nvnChannelMaskStateGetChannelMask",
             {Rig::I(s), Rig::I(8), Rig::I(out), Rig::I(0), Rig::I(out + 2), Rig::I(out + 3)});
    CHECK(rig.memory.Get<u8>(out) == 0);
    CHECK(rig.memory.Get<u8>(out + 1) == 1); // untouched
    CHECK(rig.memory.Get<u8>(out + 2) == 0);
    CHECK(rig.memory.Get<u8>(out + 3) == 0);

    // SDK (sdk+0x5eadc0): the nibble shift is (target * 4) & 31, so target 8 writes target 0 and
    // -1 writes target 7.
    rig.Call("nvnChannelMaskStateSetChannelMask",
             {Rig::I(s), Rig::I(8), Rig::I(1), Rig::I(1), Rig::I(0), Rig::I(0)});
    rig.Call("nvnChannelMaskStateSetChannelMask",
             {Rig::I(s), Rig::I(U32_MINUS_1), Rig::I(0), Rig::I(0), Rig::I(1), Rig::I(0)});
    CHECK(rig.memory.Get<u32>(s) == 0x4FFF'F0F3u);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x55);
}

// ---------------------------------------------------------------------------------------------
// Multisample state
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: multisample state keeps ten u8 fields", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0xEE);

    rig.Call("nvnMultisampleStateSetDefaults", {Rig::I(s)});
    // SDK (sdk+0x5eb280): enable and alpha-to-coverage dither on, everything else 0; byte 10
    // (host-only truncation flag) is cleared as well.
    const std::array<u8, 11> defaults{1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    CHECK(rig.memory.Get<std::array<u8, 11>>(s) == defaults);
    CHECK(rig.memory.Get<u8>(s + 11) == 0xEE);

    rig.Call("nvnMultisampleStateSetSamples", {Rig::I(s), Rig::I(0x104)}); // u8: 4
    rig.Call("nvnMultisampleStateSetAlphaToCoverageEnable", {Rig::I(s), Rig::I(1)});
    rig.Call("nvnMultisampleStateSetAlphaToCoverageDither", {Rig::I(s), Rig::I(0x1FF)});
    rig.Call("nvnMultisampleStateSetRasterSamples", {Rig::I(s), Rig::I(8)});
    rig.Call("nvnMultisampleStateSetCoverageModulationMode", {Rig::I(s), Rig::I(2)});
    rig.Call("nvnMultisampleStateSetCoverageToColorEnable", {Rig::I(s), Rig::I(1)});
    rig.Call("nvnMultisampleStateSetCoverageToColorOutput", {Rig::I(s), Rig::I(3)});
    rig.Call("nvnMultisampleStateSetSampleLocationsEnable", {Rig::I(s), Rig::I(1)});
    rig.Call("nvnMultisampleStateSetSampleLocationsGridEnable", {Rig::I(s), Rig::I(5)});
    rig.Call("nvnMultisampleStateSetMultisampleEnable", {Rig::I(s), Rig::I(0x100)}); // false
    const std::array<u8, 10> expected{0, 4, 1, 1, 8, 2, 1, 3, 1, 1};
    CHECK(rig.memory.Get<std::array<u8, 10>>(s) == expected);
    CHECK(rig.memory.Get<u8>(s + 10) == 0); // the truncation flag is untouched

    CHECK(rig.Call("nvnMultisampleStateGetMultisampleEnable", {Rig::I(s)}) == 0);
    CHECK(rig.Call("nvnMultisampleStateGetSamples", {Rig::I(s)}) == 4);
    CHECK(rig.Call("nvnMultisampleStateGetAlphaToCoverageEnable", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnMultisampleStateGetAlphaToCoverageDither", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnMultisampleStateGetRasterSamples", {Rig::I(s)}) == 8);
    CHECK(rig.Call("nvnMultisampleStateGetCoverageModulationMode", {Rig::I(s)}) == 2);
    CHECK(rig.Call("nvnMultisampleStateGetCoverageToColorEnable", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnMultisampleStateGetCoverageToColorOutput", {Rig::I(s)}) == 3);
    CHECK(rig.Call("nvnMultisampleStateGetSampleLocationsEnable", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnMultisampleStateGetSampleLocationsGridEnable", {Rig::I(s)}) == 1);

    // Values above u8 keep the low byte; booleans are normalized to 0/1.
    rig.Call("nvnMultisampleStateSetRasterSamples", {Rig::I(s), Rig::I(0x2FF)});
    CHECK(rig.memory.Get<u8>(s + 4) == 0xFF);
    CHECK(rig.Call("nvnMultisampleStateGetRasterSamples", {Rig::I(s)}) == 0xFF);
    rig.Call("nvnMultisampleStateSetCoverageToColorOutput", {Rig::I(s), Rig::I(U32_MINUS_1)});
    CHECK(rig.memory.Get<u8>(s + 7) == 0xFF);
    rig.Call("nvnMultisampleStateSetMultisampleEnable", {Rig::I(s), Rig::I(0xFE)});
    CHECK(rig.memory.Get<u8>(s) == 1);

    // Defaults restore everything.
    rig.Call("nvnMultisampleStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<std::array<u8, 11>>(s) == defaults);
}

TEST_CASE("NVN state: multisample state NVN2 additions", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    rig.Call("nvnMultisampleStateSetDefaults", {Rig::I(s)});

    // Coverage truncation lives in the spare byte 10, beyond the bytes the recorder copies.
    CHECK(rig.Call("nvnMultisampleStateGetCoverageTruncationEnable", {Rig::I(s)}) == 0);
    rig.Call("nvnMultisampleStateSetCoverageTruncationEnable", {Rig::I(s), Rig::I(1)});
    CHECK(rig.Call("nvnMultisampleStateGetCoverageTruncationEnable", {Rig::I(s)}) == 1);
    CHECK(rig.memory.Get<u8>(s + 10) == 1);
    const std::array<u8, 10> defaults{1, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    CHECK(rig.memory.Get<std::array<u8, 10>>(s) == defaults);
    rig.Call("nvnMultisampleStateSetDefaults", {Rig::I(s)});
    CHECK(rig.Call("nvnMultisampleStateGetCoverageTruncationEnable", {Rig::I(s)}) == 0);

    // Sample locations are not kept: the setter changes nothing, the grid reads 0 x 0.
    const u64 values = Rig::SCRATCH + 0x100;
    rig.memory.Put<float>(values, 0.25f);
    rig.Call("nvnMultisampleStateSetSampleLocations",
             {Rig::I(s), Rig::I(0), Rig::I(1), Rig::I(values)});
    CHECK(rig.memory.Get<std::array<u8, 10>>(s) == defaults);
    const u64 out = Rig::SCRATCH;
    rig.memory.Put<u64>(out, 0x1111'1111'2222'2222ull);
    rig.Call("nvnMultisampleStateGetSampleLocationsGrid", {Rig::I(s), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<s32>(out) == 0);
    CHECK(rig.memory.Get<s32>(out + 4) == 0);
    // Null output pointers are skipped.
    rig.Call("nvnMultisampleStateGetSampleLocationsGrid", {Rig::I(s), Rig::I(0), Rig::I(0)});
}

// ---------------------------------------------------------------------------------------------
// Polygon state
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: polygon state packs nxnative poly_st", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0x33);

    rig.Call("nvnPolygonStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u32>(s) == 0x14);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x33);
    CHECK(rig.Call("nvnPolygonStateGetCullFace", {Rig::I(s)}) == 0);
    CHECK(rig.Call("nvnPolygonStateGetFrontFace", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnPolygonStateGetPolygonMode", {Rig::I(s)}) == 2);
    CHECK(rig.Call("nvnPolygonStateGetPolygonOffsetEnables", {Rig::I(s)}) == 0);

    rig.Call("nvnPolygonStateSetCullFace", {Rig::I(s), Rig::I(3)});
    rig.Call("nvnPolygonStateSetFrontFace", {Rig::I(s), Rig::I(0)});
    rig.Call("nvnPolygonStateSetPolygonMode", {Rig::I(s), Rig::I(1)});
    rig.Call("nvnPolygonStateSetPolygonOffsetEnables", {Rig::I(s), Rig::I(5)});
    CHECK(rig.memory.Get<u32>(s) == 0xAB);
    CHECK(rig.Call("nvnPolygonStateGetCullFace", {Rig::I(s)}) == 3);
    CHECK(rig.Call("nvnPolygonStateGetFrontFace", {Rig::I(s)}) == 0);
    CHECK(rig.Call("nvnPolygonStateGetPolygonMode", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnPolygonStateGetPolygonOffsetEnables", {Rig::I(s)}) == 5);

    // Out of range: cull 2 bits, front 1 bit, mode 2 bits, offsets 3 bits.
    rig.Call("nvnPolygonStateSetCullFace", {Rig::I(s), Rig::I(7)});
    rig.Call("nvnPolygonStateSetFrontFace", {Rig::I(s), Rig::I(3)});
    rig.Call("nvnPolygonStateSetPolygonMode", {Rig::I(s), Rig::I(6)});
    rig.Call("nvnPolygonStateSetPolygonOffsetEnables", {Rig::I(s), Rig::I(0xF)});
    CHECK(rig.memory.Get<u32>(s) == 0xF7);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x33);
    CHECK(rig.Call("nvnPolygonStateGetPolygonMode", {Rig::I(s)}) == 2);
    CHECK(rig.Call("nvnPolygonStateGetPolygonOffsetEnables", {Rig::I(s)}) == 7);
}

// ---------------------------------------------------------------------------------------------
// Depth-stencil state
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: depth-stencil state packs nxnative ds_st", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0x77);

    rig.Call("nvnDepthStencilStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u64>(s) == 0x8'8888'C40Aull);
    CHECK(rig.memory.Get<u8>(s + 8) == 0x77);
    CHECK(rig.Call("nvnDepthStencilStateGetDepthTestEnable", {Rig::I(s)}) == 0);
    CHECK(rig.Call("nvnDepthStencilStateGetDepthWriteEnable", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnDepthStencilStateGetDepthFunc", {Rig::I(s)}) == 2);
    CHECK(rig.Call("nvnDepthStencilStateGetStencilTestEnable", {Rig::I(s)}) == 0);

    rig.Call("nvnDepthStencilStateSetDepthTestEnable", {Rig::I(s), Rig::I(1)});
    rig.Call("nvnDepthStencilStateSetDepthWriteEnable", {Rig::I(s), Rig::I(0x100)}); // false
    rig.Call("nvnDepthStencilStateSetDepthFunc", {Rig::I(s), Rig::I(4)});
    rig.Call("nvnDepthStencilStateSetStencilTestEnable", {Rig::I(s), Rig::I(0x11)});
    rig.Call("nvnDepthStencilStateSetStencilFunc", {Rig::I(s), Rig::I(1), Rig::I(3)}); // front
    rig.Call("nvnDepthStencilStateSetStencilFunc", {Rig::I(s), Rig::I(2), Rig::I(5)}); // back
    // Both faces, then the back face again.
    rig.Call("nvnDepthStencilStateSetStencilOp",
             {Rig::I(s), Rig::I(3), Rig::I(2), Rig::I(3), Rig::I(4)});
    rig.Call("nvnDepthStencilStateSetStencilOp",
             {Rig::I(s), Rig::I(2), Rig::I(5), Rig::I(6), Rig::I(7)});
    CHECK(rig.memory.Get<u64>(s) == 0x3B'2A19'29D1ull);
    CHECK(rig.memory.Get<u8>(s + 8) == 0x77);

    CHECK(rig.Call("nvnDepthStencilStateGetDepthTestEnable", {Rig::I(s)}) == 1);
    CHECK(rig.Call("nvnDepthStencilStateGetDepthWriteEnable", {Rig::I(s)}) == 0);
    CHECK(rig.Call("nvnDepthStencilStateGetDepthFunc", {Rig::I(s)}) == 4);
    CHECK(rig.Call("nvnDepthStencilStateGetStencilTestEnable", {Rig::I(s)}) == 1);

    // Getters read the back face only for exactly NVN_FACE_BACK.
    CHECK(rig.Call("nvnDepthStencilStateGetStencilFunc", {Rig::I(s), Rig::I(1)}) == 3);
    CHECK(rig.Call("nvnDepthStencilStateGetStencilFunc", {Rig::I(s), Rig::I(2)}) == 5);
    CHECK(rig.Call("nvnDepthStencilStateGetStencilFunc", {Rig::I(s), Rig::I(3)}) == 3);
    CHECK(rig.Call("nvnDepthStencilStateGetStencilFunc", {Rig::I(s), Rig::I(0)}) == 3);
    const u64 out = Rig::SCRATCH;
    rig.Call("nvnDepthStencilStateGetStencilOp",
             {Rig::I(s), Rig::I(1), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8)});
    CHECK(rig.memory.Get<u32>(out) == 2);
    CHECK(rig.memory.Get<u32>(out + 4) == 3);
    CHECK(rig.memory.Get<u32>(out + 8) == 4);
    rig.Call("nvnDepthStencilStateGetStencilOp",
             {Rig::I(s), Rig::I(2), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8)});
    CHECK(rig.memory.Get<u32>(out) == 5);
    CHECK(rig.memory.Get<u32>(out + 4) == 6);
    CHECK(rig.memory.Get<u32>(out + 8) == 7);
    // A null pointer is skipped.
    rig.memory.Put<u32>(out + 4, 0xDEAD);
    rig.Call("nvnDepthStencilStateGetStencilOp",
             {Rig::I(s), Rig::I(1), Rig::I(out), Rig::I(0), Rig::I(out + 8)});
    CHECK(rig.memory.Get<u32>(out + 4) == 0xDEAD);

    // Out of range: a 4-bit depth func keeps 0xF; faces without bit 0/1 change nothing.
    rig.Call("nvnDepthStencilStateSetDepthFunc", {Rig::I(s), Rig::I(0x1F)});
    CHECK(rig.memory.Get<u64>(s) == 0x3B'2A19'29FDull);
    rig.Call("nvnDepthStencilStateSetStencilFunc", {Rig::I(s), Rig::I(0), Rig::I(0xF)});
    rig.Call("nvnDepthStencilStateSetStencilFunc", {Rig::I(s), Rig::I(4), Rig::I(0xF)});
    rig.Call("nvnDepthStencilStateSetStencilOp",
             {Rig::I(s), Rig::I(0), Rig::I(0xF), Rig::I(0xF), Rig::I(0xF)});
    CHECK(rig.memory.Get<u64>(s) == 0x3B'2A19'29FDull);
    // Op values are 4 bits too: 0x11 and 0x21 keep 1.
    rig.Call("nvnDepthStencilStateSetStencilOp",
             {Rig::I(s), Rig::I(3), Rig::I(0x11), Rig::I(0x21), Rig::I(0x31)});
    CHECK(rig.Call("nvnDepthStencilStateGetStencilFunc", {Rig::I(s), Rig::I(2)}) == 5);
    rig.Call("nvnDepthStencilStateGetStencilOp",
             {Rig::I(s), Rig::I(2), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8)});
    CHECK(rig.memory.Get<u32>(out) == 1);
    CHECK(rig.memory.Get<u32>(out + 4) == 1);
    CHECK(rig.memory.Get<u32>(out + 8) == 1);
}

// ---------------------------------------------------------------------------------------------
// Vertex attribute and stream state
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: vertex attribute state packs nxnative va_st", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0x99);

    rig.Call("nvnVertexAttribStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u32>(s) == 0);
    CHECK(rig.memory.Get<u8>(s + 4) == 0x99);

    // The format is an 8-bit field (0x1A3 keeps 0xA3), the stream 5 bits (0x23 keeps 3).
    rig.Call("nvnVertexAttribStateSetFormat", {Rig::I(s), Rig::I(0x1A3), Rig::I(0x12345)});
    rig.Call("nvnVertexAttribStateSetStreamIndex", {Rig::I(s), Rig::I(0x23)});
    CHECK(rig.memory.Get<u32>(s) == 0x1923'45A3);
    CHECK(rig.Call("nvnVertexAttribStateGetStreamIndex", {Rig::I(s)}) == 3);

    const u64 out = Rig::SCRATCH;
    rig.memory.Put<u64>(out, ~0ull);
    rig.memory.Put<u64>(out + 8, ~0ull);
    rig.Call("nvnVertexAttribStateGetFormat", {Rig::I(s), Rig::I(out), Rig::I(out + 8)});
    CHECK(rig.memory.Get<u32>(out) == 0xA3);
    CHECK(rig.memory.Get<u32>(out + 4) == 0xFFFF'FFFFu); // only 4 bytes of the format
    CHECK(rig.memory.Get<s64>(out + 8) == 0x12345);       // the offset is ptrdiff_t
    // Null pointers are skipped.
    rig.Call("nvnVertexAttribStateGetFormat", {Rig::I(s), Rig::I(0), Rig::I(0)});

    // SetFormat keeps the stream; SetStreamIndex keeps format and offset.
    rig.Call("nvnVertexAttribStateSetFormat", {Rig::I(s), Rig::I(0x10), Rig::I(0x20)});
    CHECK(rig.memory.Get<u32>(s) == (0x3u << 27 | 0x20u << 8 | 0x10u));
    rig.Call("nvnVertexAttribStateSetStreamIndex", {Rig::I(s), Rig::I(0x1F)});
    CHECK(rig.memory.Get<u32>(s) == (0x1Fu << 27 | 0x20u << 8 | 0x10u));

    // Offsets that do not fit the 19 bits are truncated (nxnative logs it and does the same).
    rig.Call("nvnVertexAttribStateSetFormat", {Rig::I(s), Rig::I(0x1A3), Rig::I(0x80004)});
    CHECK(rig.memory.Get<u32>(s) == 0xF800'04A3u);
    rig.Call("nvnVertexAttribStateSetFormat",
             {Rig::I(s), Rig::I(0x1A3), Rig::I(static_cast<u64>(-4))});
    CHECK(rig.memory.Get<u32>(s) == 0xFFFF'FCA3u);
    rig.Call("nvnVertexAttribStateGetFormat", {Rig::I(s), Rig::I(out), Rig::I(out + 8)});
    CHECK(rig.memory.Get<s64>(out + 8) == 0x7FFFC);
}

TEST_CASE("NVN state: vertex stream state is two u32", "[nvn][state]") {
    Rig rig;
    const u64 s = Rig::Obj(1);
    Fill(rig, s, 0x44);

    rig.Call("nvnVertexStreamStateSetDefaults", {Rig::I(s)});
    CHECK(rig.memory.Get<u64>(s) == 0);
    CHECK(rig.memory.Get<u8>(s + 8) == 0x44);

    rig.Call("nvnVertexStreamStateSetStride", {Rig::I(s), Rig::I(0x30)});
    rig.Call("nvnVertexStreamStateSetDivisor", {Rig::I(s), Rig::I(2)});
    CHECK(rig.memory.Get<u32>(s) == 0x30);
    CHECK(rig.memory.Get<u32>(s + 4) == 2);
    CHECK(rig.Call("nvnVertexStreamStateGetStride", {Rig::I(s)}) == 0x30);
    CHECK(rig.Call("nvnVertexStreamStateGetDivisor", {Rig::I(s)}) == 2);

    // The stride is a ptrdiff_t stored as u32 (0x1'0000'0010 keeps 0x10) and reads back
    // zero-extended; a divisor of -1 reads back as the 32-bit pattern (a negative int).
    rig.Call("nvnVertexStreamStateSetStride", {Rig::I(s), Rig::I(0x1'0000'0010ull)});
    CHECK(rig.memory.Get<u32>(s) == 0x10);
    rig.Call("nvnVertexStreamStateSetStride", {Rig::I(s), Rig::I(static_cast<u64>(-1))});
    CHECK(rig.Call("nvnVertexStreamStateGetStride", {Rig::I(s)}) == 0xFFFF'FFFFull);
    rig.Call("nvnVertexStreamStateSetDivisor", {Rig::I(s), Rig::I(U32_MINUS_1)});
    CHECK(rig.memory.Get<u32>(s + 4) == 0xFFFF'FFFFu);
    CHECK((rig.Call("nvnVertexStreamStateGetDivisor", {Rig::I(s)}) & 0xFFFF'FFFFull) ==
          0xFFFF'FFFFull);
    CHECK(rig.memory.Get<u8>(s + 8) == 0x44);
}

// ---------------------------------------------------------------------------------------------
// SDK defaults through the getters (3.2.0 and 1.0.0 SetDefaults bodies are identical)
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN state: SetDefaults answer the SDK's getter values", "[nvn][state]") {
    Rig rig;
    const u64 out = Rig::SCRATCH;

    // sdk+0x5eabd0: 0x01020102'110c0000.
    const u64 blend = Rig::Obj(1);
    Fill(rig, blend, 0xAA);
    rig.Call("nvnBlendStateSetDefaults", {Rig::I(blend)});
    CHECK(rig.Call("nvnBlendStateGetBlendTarget", {Rig::I(blend)}) == 0);
    rig.Call("nvnBlendStateGetBlendFunc",
             {Rig::I(blend), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8), Rig::I(out + 12)});
    CHECK(rig.memory.Get<std::array<u32, 4>>(out) == std::array<u32, 4>{2, 1, 2, 1}); // ONE ZERO
    rig.Call("nvnBlendStateGetBlendEquation", {Rig::I(blend), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<u32, 2>>(out) == std::array<u32, 2>{1, 1}); // ADD ADD
    CHECK(rig.Call("nvnBlendStateGetAdvancedMode", {Rig::I(blend)}) == 0);
    CHECK(rig.Call("nvnBlendStateGetAdvancedOverlap", {Rig::I(blend)}) == 0); // UNCORRELATED
    CHECK((rig.Call("nvnBlendStateGetAdvancedPremultipliedSrc", {Rig::I(blend)}) & 0xFF) == 1);
    CHECK((rig.Call("nvnBlendStateGetAdvancedNormalizedDst", {Rig::I(blend)}) & 0xFF) == 1);

    // sdk+0x5eae40: 0x08030000.
    const u64 color = Rig::Obj(2);
    Fill(rig, color, 0xAA);
    rig.Call("nvnColorStateSetDefaults", {Rig::I(color)});
    for (u64 target = 0; target < 8; ++target) {
        CHECK((rig.Call("nvnColorStateGetBlendEnable", {Rig::I(color), Rig::I(target)}) & 0xFF) ==
              0);
    }
    CHECK(rig.Call("nvnColorStateGetLogicOp", {Rig::I(color)}) == 3);   // COPY
    CHECK(rig.Call("nvnColorStateGetAlphaTest", {Rig::I(color)}) == 8); // ALWAYS

    // sdk+0x5eadb0: 0xffffffff.
    const u64 mask = Rig::Obj(3);
    rig.Call("nvnChannelMaskStateSetDefaults", {Rig::I(mask)});
    for (u64 target = 0; target < 8; ++target) {
        rig.memory.Put<u32>(out, 0);
        rig.Call("nvnChannelMaskStateGetChannelMask", {Rig::I(mask), Rig::I(target), Rig::I(out),
                                                       Rig::I(out + 1), Rig::I(out + 2),
                                                       Rig::I(out + 3)});
        CHECK(rig.memory.Get<u32>(out) == 0x0101'0101u);
    }

    // sdk+0x5eb280: word 5 (enable, dither), sample locations 0x88 (not kept by the host).
    const u64 ms = Rig::Obj(4);
    Fill(rig, ms, 0xAA);
    rig.Call("nvnMultisampleStateSetDefaults", {Rig::I(ms)});
    CHECK((rig.Call("nvnMultisampleStateGetMultisampleEnable", {Rig::I(ms)}) & 0xFF) == 1);
    CHECK(rig.Call("nvnMultisampleStateGetSamples", {Rig::I(ms)}) == 0);
    CHECK((rig.Call("nvnMultisampleStateGetAlphaToCoverageEnable", {Rig::I(ms)}) & 0xFF) == 0);
    CHECK((rig.Call("nvnMultisampleStateGetAlphaToCoverageDither", {Rig::I(ms)}) & 0xFF) == 1);
    CHECK(rig.Call("nvnMultisampleStateGetRasterSamples", {Rig::I(ms)}) == 0);
    CHECK(rig.Call("nvnMultisampleStateGetCoverageModulationMode", {Rig::I(ms)}) == 0); // NONE
    CHECK((rig.Call("nvnMultisampleStateGetCoverageToColorEnable", {Rig::I(ms)}) & 0xFF) == 0);
    CHECK(rig.Call("nvnMultisampleStateGetCoverageToColorOutput", {Rig::I(ms)}) == 0);
    CHECK((rig.Call("nvnMultisampleStateGetSampleLocationsEnable", {Rig::I(ms)}) & 0xFF) == 0);
    CHECK((rig.Call("nvnMultisampleStateGetSampleLocationsGridEnable", {Rig::I(ms)}) & 0xFF) ==
          0);
    CHECK((rig.Call("nvnMultisampleStateGetCoverageTruncationEnable", {Rig::I(ms)}) & 0xFF) == 0);

    // sdk+0x5eb1b0: 0x14.
    const u64 poly = Rig::Obj(5);
    rig.Call("nvnPolygonStateSetDefaults", {Rig::I(poly)});
    CHECK(rig.Call("nvnPolygonStateGetCullFace", {Rig::I(poly)}) == 0);   // NONE
    CHECK(rig.Call("nvnPolygonStateGetFrontFace", {Rig::I(poly)}) == 1);  // CCW
    CHECK(rig.Call("nvnPolygonStateGetPolygonMode", {Rig::I(poly)}) == 2); // FILL
    CHECK(rig.Call("nvnPolygonStateGetPolygonOffsetEnables", {Rig::I(poly)}) == 0);

    // sdk+0x5eafd0: 0x11181118'00000022.
    const u64 ds = Rig::Obj(6);
    rig.Call("nvnDepthStencilStateSetDefaults", {Rig::I(ds)});
    CHECK((rig.Call("nvnDepthStencilStateGetDepthTestEnable", {Rig::I(ds)}) & 0xFF) == 0);
    CHECK((rig.Call("nvnDepthStencilStateGetDepthWriteEnable", {Rig::I(ds)}) & 0xFF) == 1);
    CHECK(rig.Call("nvnDepthStencilStateGetDepthFunc", {Rig::I(ds)}) == 2); // LESS
    CHECK((rig.Call("nvnDepthStencilStateGetStencilTestEnable", {Rig::I(ds)}) & 0xFF) == 0);
    for (u64 face : {1u, 2u}) {
        CHECK(rig.Call("nvnDepthStencilStateGetStencilFunc", {Rig::I(ds), Rig::I(face)}) == 8);
        rig.Call("nvnDepthStencilStateGetStencilOp",
                 {Rig::I(ds), Rig::I(face), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8)});
        CHECK(rig.memory.Get<std::array<u32, 3>>(out) == std::array<u32, 3>{1, 1, 1}); // KEEP
    }

    // sdk+0x5eaf00 / sdk+0x5eaf80: all zero.
    const u64 attrib = Rig::Obj(7);
    Fill(rig, attrib, 0xAA);
    rig.Call("nvnVertexAttribStateSetDefaults", {Rig::I(attrib)});
    rig.memory.Put<u64>(out, ~0ull);
    rig.memory.Put<u64>(out + 8, ~0ull);
    rig.Call("nvnVertexAttribStateGetFormat", {Rig::I(attrib), Rig::I(out), Rig::I(out + 8)});
    CHECK(rig.memory.Get<u32>(out) == 0);
    CHECK(rig.memory.Get<s64>(out + 8) == 0);
    CHECK(rig.Call("nvnVertexAttribStateGetStreamIndex", {Rig::I(attrib)}) == 0);
    const u64 stream = Rig::Obj(8);
    Fill(rig, stream, 0xAA);
    rig.Call("nvnVertexStreamStateSetDefaults", {Rig::I(stream)});
    CHECK(rig.Call("nvnVertexStreamStateGetStride", {Rig::I(stream)}) == 0);
    CHECK(rig.Call("nvnVertexStreamStateGetDivisor", {Rig::I(stream)}) == 0);
}

// ---------------------------------------------------------------------------------------------
// Sampler builder defaults (functions_sampler.cpp)
// ---------------------------------------------------------------------------------------------

TEST_CASE("NVN sampler: builder defaults answer the SDK's getter values", "[nvn][state]") {
    Rig rig;
    const u64 builder = Rig::Obj(1);
    const u64 sampler = Rig::Obj(2);
    const u64 out = Rig::SCRATCH;
    Fill(rig, builder, 0xCC);
    rig.Call("nvnSamplerBuilderSetDevice", {Rig::I(builder), Rig::I(0x1234'5678ull)});
    rig.Call("nvnSamplerBuilderSetDefaults", {Rig::I(builder)});

    // SDK sdk+0x5ea460 (1.0.0 sdk+0x5c6220): state word 0x08110111, LOD clamp [0, 1000], bias 0,
    // anisotropy 1, border 0; the device stays.
    CHECK(rig.Call("nvnSamplerBuilderGetDevice", {Rig::I(builder)}) == 0x1234'5678ull);
    rig.Call("nvnSamplerBuilderGetMinMagFilter", {Rig::I(builder), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<s32, 2>>(out) == std::array<s32, 2>{1, 1}); // LINEAR LINEAR
    rig.Call("nvnSamplerBuilderGetWrapMode",
             {Rig::I(builder), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8)});
    CHECK(rig.memory.Get<std::array<s32, 3>>(out) == std::array<s32, 3>{1, 1, 1}); // REPEAT
    rig.Call("nvnSamplerBuilderGetLodClamp", {Rig::I(builder), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<f32, 2>>(out) == std::array<f32, 2>{0.0f, 1000.0f});
    CHECK(rig.CallF("nvnSamplerBuilderGetLodBias", {Rig::I(builder)}) == 0.0f);
    rig.Call("nvnSamplerBuilderGetCompare", {Rig::I(builder), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<s32, 2>>(out) == std::array<s32, 2>{0, 2}); // NONE LESS
    rig.Call("nvnSamplerBuilderGetBorderColor", {Rig::I(builder), Rig::I(out)});
    CHECK(rig.memory.Get<std::array<f32, 4>>(out) == std::array<f32, 4>{});
    CHECK(rig.CallF("nvnSamplerBuilderGetMaxAnisotropy", {Rig::I(builder)}) == 1.0f);
    CHECK(rig.Call("nvnSamplerBuilderGetReductionFilter", {Rig::I(builder)}) == 0); // AVERAGE
    CHECK(rig.CallF("nvnSamplerBuilderGetLodSnap", {Rig::I(builder)}) == 0.0f);

    // SDK sdk+0x5ea740: Initialize copies the builder and returns true; the object getters
    // (sdk+0x5ea780..0x5ea8c0) answer the same values.
    REQUIRE((rig.Call("nvnSamplerInitialize", {Rig::I(sampler), Rig::I(builder)}) & 0xFF) == 1);
    REQUIRE(rig.sink.samplers.size() == 1);
    const auto& desc = rig.sink.samplers.back();
    CHECK(desc.min_filter == 1);
    CHECK(desc.mag_filter == 1);
    CHECK(desc.lod_min == 0.0f);
    CHECK(desc.lod_max == 1000.0f);
    rig.Call("nvnSamplerGetMinMagFilter", {Rig::I(sampler), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<s32, 2>>(out) == std::array<s32, 2>{1, 1});
    rig.Call("nvnSamplerGetWrapMode",
             {Rig::I(sampler), Rig::I(out), Rig::I(out + 4), Rig::I(out + 8)});
    CHECK(rig.memory.Get<std::array<s32, 3>>(out) == std::array<s32, 3>{1, 1, 1});
    rig.Call("nvnSamplerGetLodClamp", {Rig::I(sampler), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<f32, 2>>(out) == std::array<f32, 2>{0.0f, 1000.0f});
    CHECK(rig.CallF("nvnSamplerGetLodBias", {Rig::I(sampler)}) == 0.0f);
    rig.Call("nvnSamplerGetCompare", {Rig::I(sampler), Rig::I(out), Rig::I(out + 4)});
    CHECK(rig.memory.Get<std::array<s32, 2>>(out) == std::array<s32, 2>{0, 2});
    rig.memory.Put(out, std::array<f32, 4>{9.0f, 9.0f, 9.0f, 9.0f});
    rig.Call("nvnSamplerGetBorderColor", {Rig::I(sampler), Rig::I(out)});
    CHECK(rig.memory.Get<std::array<f32, 4>>(out) == std::array<f32, 4>{});
    CHECK(rig.CallF("nvnSamplerGetMaxAnisotropy", {Rig::I(sampler)}) == 1.0f);
    CHECK(rig.Call("nvnSamplerGetReductionFilter", {Rig::I(sampler)}) == 0);
}
