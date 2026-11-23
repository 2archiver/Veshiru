// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include "common/common_types.h"

namespace VideoCommon::Nvn {

// Packed in-place layouts of the NVN state objects (blend, color, channel mask, multisample,
// polygon, depth-stencil, vertex attribute, vertex stream). They are nxnative's `nvn_obj.h`
// `blend_st`, `color_st`, `chmask_st`, `ms_st`, `poly_st`, `ds_st`, `va_st` and `vs_st`: the
// game's NVN state object memory holds these bytes, the HLE bodies (functions_state.cpp) write
// them and the recorder copies them verbatim into BindState / BindVertexAttribs / BindVertexStreams
// records, which the Vulkan replay decodes with the accessors below. Nothing else may interpret
// the bytes.
//
// The bit-fields of nxnative are allocated least-significant-bit first in declaration order and
// never straddle their declared storage unit; the accessors spell the shifts out (no C++
// bit-fields: their layout is implementation-defined). A field value that is wider than its
// field is truncated to the field width, exactly as the C bit-field assignment does.
//
// Words are little-endian in guest memory (u64 for blend and depth-stencil, u32 for the others).

namespace StateBits {

[[nodiscard]] constexpr u64 Mask(u32 width) {
    return (u64{1} << width) - 1;
}
[[nodiscard]] constexpr u64 Extract(u64 word, u32 shift, u32 width) {
    return (word >> shift) & Mask(width);
}
[[nodiscard]] constexpr u64 Insert(u64 word, u32 shift, u32 width, u64 value) {
    return (word & ~(Mask(width) << shift)) | ((value & Mask(width)) << shift);
}

} // namespace StateBits

// `Name(word)` reads a field; `WithName(word, value)` returns the word with the field replaced
// (the value truncated to the field width). `Name_SHIFT` / `Name_WIDTH` give its position.
#define NVN_STATE_FIELD(Name, Word, Shift, Width)                                                  \
    static constexpr u32 Name##_SHIFT = Shift;                                                     \
    static constexpr u32 Name##_WIDTH = Width;                                                     \
    [[nodiscard]] static constexpr u32 Name(Word word) {                                           \
        return static_cast<u32>(StateBits::Extract(word, Shift, Width));                           \
    }                                                                                              \
    [[nodiscard]] static constexpr Word With##Name(Word word, u64 value) {                         \
        return static_cast<Word>(StateBits::Insert(word, Shift, Width, value));                    \
    }

// NVN enum values the defaults and the sanitizing refer to (the NVN header's numbering).
inline constexpr u32 STATE_FACE_NONE = 0;
inline constexpr u32 STATE_FACE_FRONT = 1;
inline constexpr u32 STATE_FACE_BACK = 2;
inline constexpr u32 STATE_BLEND_FUNC_ZERO = 1;
inline constexpr u32 STATE_BLEND_FUNC_ONE = 2;
inline constexpr u32 STATE_BLEND_EQUATION_ADD = 1;
inline constexpr u32 STATE_LOGIC_OP_COPY = 3;
inline constexpr u32 STATE_ALPHA_FUNC_ALWAYS = 8;
inline constexpr u32 STATE_FRONT_FACE_CCW = 1;
inline constexpr u32 STATE_POLYGON_MODE_FILL = 2;
inline constexpr u32 STATE_DEPTH_FUNC_LESS = 2;
inline constexpr u32 STATE_STENCIL_FUNC_ALWAYS = 8;
inline constexpr u32 STATE_STENCIL_OP_KEEP = 1;
/// Highest NVNblendAdvancedMode (HSL_LUMINOSITY); larger modes are stored as 0 (none).
inline constexpr u32 STATE_BLEND_ADVANCED_MODE_MAX = 0x32;

/// NVNblendState: 8 bytes, one u64 (nxnative blend_st).
///   target:3 src_rgb:7 dst_rgb:7 src_a:7 dst_a:7 eq_rgb:3 eq_a:3 adv:6 adv_npm:1 adv_ovl:2
///   plus bit 46 (this port): adv_nnd, "advanced blend destination is NOT normalized" (NVN2 adds
///   SetAdvancedNormalizedDst, which nxnative lacks; inverted like adv_npm so that the all-zero
///   default means "normalized"). Bits 47..63 are zero.
/// Funcs and equations are raw NVNblendFunc / NVNblendEquation values; the 7-bit and 3-bit
/// fields drop higher bits.
struct BlendState {
    using Word = u64;
    static constexpr u32 SIZE = 8;

    NVN_STATE_FIELD(TargetIndex, u64, 0, 3)
    NVN_STATE_FIELD(SrcRgbFunc, u64, 3, 7)
    NVN_STATE_FIELD(DstRgbFunc, u64, 10, 7)
    NVN_STATE_FIELD(SrcAlphaFunc, u64, 17, 7)
    NVN_STATE_FIELD(DstAlphaFunc, u64, 24, 7)
    NVN_STATE_FIELD(RgbEquation, u64, 31, 3)
    NVN_STATE_FIELD(AlphaEquation, u64, 34, 3)
    NVN_STATE_FIELD(AdvancedMode, u64, 37, 6)
    NVN_STATE_FIELD(AdvancedNotPremultiplied, u64, 43, 1)
    NVN_STATE_FIELD(AdvancedOverlap, u64, 44, 2)
    NVN_STATE_FIELD(AdvancedNotNormalizedDst, u64, 46, 1)

    /// Mode stored by nvnBlendStateSetAdvancedMode (modes above HSL_LUMINOSITY become 0).
    [[nodiscard]] static constexpr u32 SanitizeAdvancedMode(u32 mode) {
        return mode <= STATE_BLEND_ADVANCED_MODE_MAX ? mode : 0;
    }

    /// nvnBlendStateSetDefaults: target 0, funcs ONE/ZERO/ONE/ZERO, equations ADD/ADD.
    [[nodiscard]] static constexpr u64 Defaults() {
        u64 word = 0;
        word = WithSrcRgbFunc(word, STATE_BLEND_FUNC_ONE);
        word = WithDstRgbFunc(word, STATE_BLEND_FUNC_ZERO);
        word = WithSrcAlphaFunc(word, STATE_BLEND_FUNC_ONE);
        word = WithDstAlphaFunc(word, STATE_BLEND_FUNC_ZERO);
        word = WithRgbEquation(word, STATE_BLEND_EQUATION_ADD);
        word = WithAlphaEquation(word, STATE_BLEND_EQUATION_ADD);
        return word;
    }
};

/// NVNcolorState: 4 bytes, one u32 (nxnative color_st): blend_en:8 (bit i = target i blends)
/// logic_op:4 alpha_func:4; bits 16..31 are zero.
struct ColorState {
    using Word = u32;
    static constexpr u32 SIZE = 4;
    static constexpr u32 TARGETS = 8;

    NVN_STATE_FIELD(BlendEnableMask, u32, 0, 8)
    NVN_STATE_FIELD(LogicOp, u32, 8, 4)
    NVN_STATE_FIELD(AlphaFunc, u32, 12, 4)

    [[nodiscard]] static constexpr bool BlendEnabled(u32 word, s32 target) {
        return target >= 0 && target < static_cast<s32>(TARGETS) &&
               ((BlendEnableMask(word) >> target) & 1) != 0;
    }
    /// Out-of-range targets leave the word unchanged (nxnative returns early).
    [[nodiscard]] static constexpr u32 WithBlendEnabled(u32 word, s32 target, bool enable) {
        if (target < 0 || target >= static_cast<s32>(TARGETS)) {
            return word;
        }
        const u32 mask = BlendEnableMask(word);
        const u32 bit = u32{1} << target;
        return WithBlendEnableMask(word, enable ? (mask | bit) : (mask & ~bit));
    }

    /// nvnColorStateSetDefaults: no blending, logic op COPY, alpha test ALWAYS.
    [[nodiscard]] static constexpr u32 Defaults() {
        u32 word = 0;
        word = WithLogicOp(word, STATE_LOGIC_OP_COPY);
        word = WithAlphaFunc(word, STATE_ALPHA_FUNC_ALWAYS);
        return word;
    }
};

/// NVNchannelMaskState: 4 bytes, one u32 (nxnative chmask_st): 4 bits per target, target 0 in the
/// low nibble; per nibble r = 1, g = 2, b = 4, a = 8, bit set = channel is written.
struct ChannelMaskState {
    using Word = u32;
    static constexpr u32 SIZE = 4;
    static constexpr u32 TARGETS = 8;

    /// Nibble of `target` (0 for an out-of-range target).
    [[nodiscard]] static constexpr u32 Channels(u32 word, s32 target) {
        if (target < 0 || target >= static_cast<s32>(TARGETS)) {
            return 0;
        }
        return (word >> (4 * target)) & 0xF;
    }
    /// Out-of-range targets leave the word unchanged.
    [[nodiscard]] static constexpr u32 WithChannels(u32 word, s32 target, u32 nibble) {
        if (target < 0 || target >= static_cast<s32>(TARGETS)) {
            return word;
        }
        return (word & ~(u32{0xF} << (4 * target))) | ((nibble & 0xF) << (4 * target));
    }
    [[nodiscard]] static constexpr u32 PackChannels(bool r, bool g, bool b, bool a) {
        return (r ? 1u : 0u) | (g ? 2u : 0u) | (b ? 4u : 0u) | (a ? 8u : 0u);
    }

    /// nvnChannelMaskStateSetDefaults: every channel of every target written.
    [[nodiscard]] static constexpr u32 Defaults() {
        return 0xFFFF'FFFFu;
    }
};

/// NVNmultisampleState: 10 bytes of u8 fields (nxnative ms_st), in this order. Every boolean is
/// stored as 0 or 1; the others keep the low 8 bits of the argument. NVNmultisampleState is 0x18
/// bytes, so this port keeps the NVN2 coverage-truncation flag in the spare byte 10 (not part of
/// the 10 bytes the recorder copies).
struct MultisampleState {
    static constexpr u32 SIZE = 10;
    static constexpr u32 OFFSET_ENABLE = 0;
    static constexpr u32 OFFSET_SAMPLES = 1;
    static constexpr u32 OFFSET_ALPHA_TO_COVERAGE = 2;
    static constexpr u32 OFFSET_ALPHA_TO_COVERAGE_DITHER = 3;
    static constexpr u32 OFFSET_RASTER_SAMPLES = 4;
    static constexpr u32 OFFSET_COVERAGE_MODULATION_MODE = 5;
    static constexpr u32 OFFSET_COVERAGE_TO_COLOR_ENABLE = 6;
    static constexpr u32 OFFSET_COVERAGE_TO_COLOR_OUTPUT = 7;
    static constexpr u32 OFFSET_SAMPLE_LOCATIONS_ENABLE = 8;
    static constexpr u32 OFFSET_SAMPLE_LOCATIONS_GRID_ENABLE = 9;
    /// Beyond SIZE (host-only): nvnMultisampleStateSetCoverageTruncationEnable.
    static constexpr u32 OFFSET_COVERAGE_TRUNCATION_ENABLE = 10;
    /// Bytes nvnMultisampleStateSetDefaults writes (SIZE plus the truncation flag).
    static constexpr u32 DEFAULTS_SIZE = 11;

    [[nodiscard]] static constexpr bool Enable(const u8* bytes) {
        return bytes[OFFSET_ENABLE] != 0;
    }
    [[nodiscard]] static constexpr u32 Samples(const u8* bytes) {
        return bytes[OFFSET_SAMPLES];
    }
    [[nodiscard]] static constexpr bool AlphaToCoverage(const u8* bytes) {
        return bytes[OFFSET_ALPHA_TO_COVERAGE] != 0;
    }
    [[nodiscard]] static constexpr bool AlphaToCoverageDither(const u8* bytes) {
        return bytes[OFFSET_ALPHA_TO_COVERAGE_DITHER] != 0;
    }
    [[nodiscard]] static constexpr u32 RasterSamples(const u8* bytes) {
        return bytes[OFFSET_RASTER_SAMPLES];
    }
    [[nodiscard]] static constexpr u32 CoverageModulationMode(const u8* bytes) {
        return bytes[OFFSET_COVERAGE_MODULATION_MODE];
    }
    [[nodiscard]] static constexpr bool CoverageToColorEnable(const u8* bytes) {
        return bytes[OFFSET_COVERAGE_TO_COLOR_ENABLE] != 0;
    }
    [[nodiscard]] static constexpr u32 CoverageToColorOutput(const u8* bytes) {
        return bytes[OFFSET_COVERAGE_TO_COLOR_OUTPUT];
    }
    [[nodiscard]] static constexpr bool SampleLocationsEnable(const u8* bytes) {
        return bytes[OFFSET_SAMPLE_LOCATIONS_ENABLE] != 0;
    }
    [[nodiscard]] static constexpr bool SampleLocationsGridEnable(const u8* bytes) {
        return bytes[OFFSET_SAMPLE_LOCATIONS_GRID_ENABLE] != 0;
    }
    /// Only valid when the 0x18-byte guest object was read (see OFFSET_COVERAGE_TRUNCATION_ENABLE).
    [[nodiscard]] static constexpr bool CoverageTruncationEnable(const u8* bytes) {
        return bytes[OFFSET_COVERAGE_TRUNCATION_ENABLE] != 0;
    }
};

/// NVNpolygonState: 4 bytes, one u32 (nxnative poly_st): cull:2 front:1 mode:2 offset_en:3.
/// Raw NVNface, NVNfrontFace, NVNpolygonMode and NVNpolygonOffsetEnable bits.
struct PolygonState {
    using Word = u32;
    static constexpr u32 SIZE = 4;

    NVN_STATE_FIELD(CullFace, u32, 0, 2)
    NVN_STATE_FIELD(FrontFace, u32, 2, 1)
    NVN_STATE_FIELD(PolygonMode, u32, 3, 2)
    NVN_STATE_FIELD(OffsetEnables, u32, 5, 3)

    /// nvnPolygonStateSetDefaults: no culling, CCW front, FILL, no offsets.
    [[nodiscard]] static constexpr u32 Defaults() {
        u32 word = 0;
        word = WithCullFace(word, STATE_FACE_NONE);
        word = WithFrontFace(word, STATE_FRONT_FACE_CCW);
        word = WithPolygonMode(word, STATE_POLYGON_MODE_FILL);
        return word;
    }
};

/// NVNdepthStencilState: 8 bytes, one u64 (nxnative ds_st):
///   dt:1 dw:1 df:4 st:1 sf_f:4 sf_b:4 fail_f:4 zfail_f:4 zpass_f:4 fail_b:4 zfail_b:4 zpass_b:4
/// (bits 39..63 zero). Funcs and ops are raw NVNdepthFunc / NVNstencilFunc / NVNstencilOp values.
struct DepthStencilState {
    using Word = u64;
    static constexpr u32 SIZE = 8;

    NVN_STATE_FIELD(DepthTestEnable, u64, 0, 1)
    NVN_STATE_FIELD(DepthWriteEnable, u64, 1, 1)
    NVN_STATE_FIELD(DepthFunc, u64, 2, 4)
    NVN_STATE_FIELD(StencilTestEnable, u64, 6, 1)
    NVN_STATE_FIELD(FrontStencilFunc, u64, 7, 4)
    NVN_STATE_FIELD(BackStencilFunc, u64, 11, 4)
    NVN_STATE_FIELD(FrontFailOp, u64, 15, 4)
    NVN_STATE_FIELD(FrontDepthFailOp, u64, 19, 4)
    NVN_STATE_FIELD(FrontDepthPassOp, u64, 23, 4)
    NVN_STATE_FIELD(BackFailOp, u64, 27, 4)
    NVN_STATE_FIELD(BackDepthFailOp, u64, 31, 4)
    NVN_STATE_FIELD(BackDepthPassOp, u64, 35, 4)

    /// nvnDepthStencilStateGet* read the back face only for `face == NVN_FACE_BACK` exactly
    /// (FRONT, FRONT_AND_BACK and NONE read the front face), the setters apply to every face bit.
    [[nodiscard]] static constexpr bool ReadsBackFace(u32 face) {
        return face == STATE_FACE_BACK;
    }
    [[nodiscard]] static constexpr u32 StencilFunc(u64 word, bool back) {
        return back ? BackStencilFunc(word) : FrontStencilFunc(word);
    }
    [[nodiscard]] static constexpr u32 FailOp(u64 word, bool back) {
        return back ? BackFailOp(word) : FrontFailOp(word);
    }
    [[nodiscard]] static constexpr u32 DepthFailOp(u64 word, bool back) {
        return back ? BackDepthFailOp(word) : FrontDepthFailOp(word);
    }
    [[nodiscard]] static constexpr u32 DepthPassOp(u64 word, bool back) {
        return back ? BackDepthPassOp(word) : FrontDepthPassOp(word);
    }
    /// nvnDepthStencilStateSetStencilFunc: applies to the front face if `face` has bit 0, to the
    /// back face if it has bit 1.
    [[nodiscard]] static constexpr u64 WithStencilFunc(u64 word, u32 face, u32 func) {
        if ((face & STATE_FACE_FRONT) != 0) {
            word = WithFrontStencilFunc(word, func);
        }
        if ((face & STATE_FACE_BACK) != 0) {
            word = WithBackStencilFunc(word, func);
        }
        return word;
    }
    /// nvnDepthStencilStateSetStencilOp, same face rule.
    [[nodiscard]] static constexpr u64 WithStencilOps(u64 word, u32 face, u32 fail, u32 depth_fail,
                                                      u32 depth_pass) {
        if ((face & STATE_FACE_FRONT) != 0) {
            word = WithFrontFailOp(word, fail);
            word = WithFrontDepthFailOp(word, depth_fail);
            word = WithFrontDepthPassOp(word, depth_pass);
        }
        if ((face & STATE_FACE_BACK) != 0) {
            word = WithBackFailOp(word, fail);
            word = WithBackDepthFailOp(word, depth_fail);
            word = WithBackDepthPassOp(word, depth_pass);
        }
        return word;
    }

    /// nvnDepthStencilStateSetDefaults: depth test off, depth write on, LESS, stencil off, ALWAYS
    /// and KEEP on both faces.
    [[nodiscard]] static constexpr u64 Defaults() {
        u64 word = 0;
        word = WithDepthWriteEnable(word, 1);
        word = WithDepthFunc(word, STATE_DEPTH_FUNC_LESS);
        word = WithStencilFunc(word, STATE_FACE_FRONT | STATE_FACE_BACK, STATE_STENCIL_FUNC_ALWAYS);
        word = WithStencilOps(word, STATE_FACE_FRONT | STATE_FACE_BACK, STATE_STENCIL_OP_KEEP,
                              STATE_STENCIL_OP_KEEP, STATE_STENCIL_OP_KEEP);
        return word;
    }
};

/// NVNvertexAttribState: 4 bytes, one u32 (nxnative va_st): fmt:8 offset:19 stream:5. The
/// recorder's BindVertexAttribs records hold an array of these words.
struct VertexAttribState {
    using Word = u32;
    static constexpr u32 SIZE = 4;
    static constexpr u32 MAX_OFFSET = (1u << 19) - 1;

    NVN_STATE_FIELD(Format, u32, 0, 8)
    NVN_STATE_FIELD(Offset, u32, 8, 19)
    NVN_STATE_FIELD(StreamIndex, u32, 27, 5)

    /// nvnVertexAttribStateSetDefaults: all zero.
    [[nodiscard]] static constexpr u32 Defaults() {
        return 0;
    }
};

/// NVNvertexStreamState: 8 bytes, two u32 (nxnative vs_st): stride at +0, divisor at +4, i.e.
/// exactly record.h's `VertexStreamState`, which the recorder's BindVertexStreams records hold
/// an array of (so no second struct is declared here).
inline constexpr u32 VERTEX_STREAM_STATE_SIZE = 8;
inline constexpr u32 VERTEX_STREAM_STATE_STRIDE_OFFSET = 0;
inline constexpr u32 VERTEX_STREAM_STATE_DIVISOR_OFFSET = 4;

#undef NVN_STATE_FIELD

} // namespace VideoCommon::Nvn
