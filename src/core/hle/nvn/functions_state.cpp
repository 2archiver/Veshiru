// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "video_core/nvn/state_layouts.h"

// State-object family: NVNblendState, NVNcolorState, NVNchannelMaskState, NVNmultisampleState,
// NVNpolygonState, NVNdepthStencilState, NVNvertexAttribState and NVNvertexStreamState. The
// objects live entirely in the game's memory in nxnative's packed layouts (`blend_st` ... `vs_st`,
// spelled out in video_core/nvn/state_layouts.h), which the recorder copies verbatim when the
// game binds them. Every body is one load and one store of the object's word (or one byte store
// for the multisample state). Ported from nxnative hle_nvn.c (state section and "Getters and the
// rest of the multisample state"); values too wide for their bit-field are truncated like the C
// bit-field assignment does.
namespace Core::NvnHle::Hle {
namespace {

using namespace VideoCommon::Nvn;

void LogOnce(Common::BootCounter& logged, const char* what) {
    if (logged.First()) {
        LOG_WARNING(Core, "[nvn] {}", what);
    }
}

/// Read-modify-write of the guest word at `va`.
template <typename Word, typename Fn>
void Modify(Call& c, u64 va, Fn&& fn) {
    c.Store<Word>(va, fn(c.Load<Word>(va)));
}

// ---------------------------------------------------------------------------------------------
// NVNblendState
// ---------------------------------------------------------------------------------------------

void nvnBlendStateSetDefaults(Call& c) {
    c.Store<u64>(c.args.NextPointer(), BlendState::Defaults());
}

void nvnBlendStateSetBlendTarget(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 target = c.args.NextU32();
    Modify<u64>(c, state, [&](u64 w) { return BlendState::WithTargetIndex(w, target); });
}

void nvnBlendStateSetBlendFunc(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 src_rgb = c.args.NextU32();
    const u32 dst_rgb = c.args.NextU32();
    const u32 src_alpha = c.args.NextU32();
    const u32 dst_alpha = c.args.NextU32();
    Modify<u64>(c, state, [&](u64 w) {
        w = BlendState::WithSrcRgbFunc(w, src_rgb);
        w = BlendState::WithDstRgbFunc(w, dst_rgb);
        w = BlendState::WithSrcAlphaFunc(w, src_alpha);
        return BlendState::WithDstAlphaFunc(w, dst_alpha);
    });
}

void nvnBlendStateSetBlendEquation(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 rgb = c.args.NextU32();
    const u32 alpha = c.args.NextU32();
    Modify<u64>(c, state, [&](u64 w) {
        return BlendState::WithAlphaEquation(BlendState::WithRgbEquation(w, rgb), alpha);
    });
}

// Advanced blend modes (NV_blend_equation_advanced): kept in the state; while one is set the
// hardware ignores the blend functions. Mode 0 (none) is the default; modes above
// HSL_LUMINOSITY are stored as 0.
void nvnBlendStateSetAdvancedMode(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 mode = BlendState::SanitizeAdvancedMode(c.args.NextU32());
    Modify<u64>(c, state, [&](u64 w) { return BlendState::WithAdvancedMode(w, mode); });
}

void nvnBlendStateSetAdvancedPremultipliedSrc(Call& c) {
    const u64 state = c.args.NextPointer();
    const bool premultiplied = c.args.NextBool();
    Modify<u64>(c, state, [&](u64 w) {
        return BlendState::WithAdvancedNotPremultiplied(w, premultiplied ? 0 : 1);
    });
}

void nvnBlendStateSetAdvancedOverlap(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 overlap = c.args.NextU32();
    Modify<u64>(c, state, [&](u64 w) { return BlendState::WithAdvancedOverlap(w, overlap); });
}

// NVN2 (nxnative lacks it): stored inverted in bit 46, so the all-zero default reads "normalized",
// as the SDK default does (3.2.0 sdk+0x5eabd0 sets both advanced premultiplied-src and
// normalized-dst bits, 0x000c0000 of the first word).
void nvnBlendStateSetAdvancedNormalizedDst(Call& c) {
    const u64 state = c.args.NextPointer();
    const bool normalized = c.args.NextBool();
    Modify<u64>(c, state, [&](u64 w) {
        return BlendState::WithAdvancedNotNormalizedDst(w, normalized ? 0 : 1);
    });
}

void nvnBlendStateGetBlendTarget(Call& c) {
    c.ret.SetU32(BlendState::TargetIndex(c.Load<u64>(c.args.NextPointer())));
}

void nvnBlendStateGetBlendFunc(Call& c) {
    const u64 state = c.args.NextPointer();
    const u64 src_rgb = c.args.NextPointer();
    const u64 dst_rgb = c.args.NextPointer();
    const u64 src_alpha = c.args.NextPointer();
    const u64 dst_alpha = c.args.NextPointer();
    const u64 w = c.Load<u64>(state);
    if (src_rgb != 0) {
        c.Store<u32>(src_rgb, BlendState::SrcRgbFunc(w));
    }
    if (dst_rgb != 0) {
        c.Store<u32>(dst_rgb, BlendState::DstRgbFunc(w));
    }
    if (src_alpha != 0) {
        c.Store<u32>(src_alpha, BlendState::SrcAlphaFunc(w));
    }
    if (dst_alpha != 0) {
        c.Store<u32>(dst_alpha, BlendState::DstAlphaFunc(w));
    }
}

void nvnBlendStateGetBlendEquation(Call& c) {
    const u64 state = c.args.NextPointer();
    const u64 rgb = c.args.NextPointer();
    const u64 alpha = c.args.NextPointer();
    const u64 w = c.Load<u64>(state);
    if (rgb != 0) {
        c.Store<u32>(rgb, BlendState::RgbEquation(w));
    }
    if (alpha != 0) {
        c.Store<u32>(alpha, BlendState::AlphaEquation(w));
    }
}

void nvnBlendStateGetAdvancedMode(Call& c) {
    c.ret.SetU32(BlendState::AdvancedMode(c.Load<u64>(c.args.NextPointer())));
}

void nvnBlendStateGetAdvancedOverlap(Call& c) {
    c.ret.SetU32(BlendState::AdvancedOverlap(c.Load<u64>(c.args.NextPointer())));
}

void nvnBlendStateGetAdvancedPremultipliedSrc(Call& c) {
    c.ret.SetBool(BlendState::AdvancedNotPremultiplied(c.Load<u64>(c.args.NextPointer())) == 0);
}

void nvnBlendStateGetAdvancedNormalizedDst(Call& c) {
    c.ret.SetBool(BlendState::AdvancedNotNormalizedDst(c.Load<u64>(c.args.NextPointer())) == 0);
}

// ---------------------------------------------------------------------------------------------
// NVNcolorState
// ---------------------------------------------------------------------------------------------

void nvnColorStateSetDefaults(Call& c) {
    c.Store<u32>(c.args.NextPointer(), ColorState::Defaults());
}

/// SDK (3.2.0 sdk+0x5eae50, 1.0.0 sdk+0x5c6c20): sets/clears bit `target & 31` of the 8-bit enable
/// byte; a target of 8..23 would OR the enable into its logic-op / alpha-test bits (a layout this
/// host word does not share), so out-of-range targets (API misuse) change nothing here.
void nvnColorStateSetBlendEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    const s32 target = c.args.NextS32();
    const bool enable = c.args.NextBool();
    if (target < 0 || target >= static_cast<s32>(ColorState::TARGETS)) {
        return;
    }
    Modify<u32>(c, state, [&](u32 w) { return ColorState::WithBlendEnabled(w, target, enable); });
}

void nvnColorStateSetLogicOp(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 op = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return ColorState::WithLogicOp(w, op); });
}

void nvnColorStateSetAlphaTest(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 func = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return ColorState::WithAlphaFunc(w, func); });
}

void nvnColorStateGetBlendEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    const s32 target = c.args.NextS32();
    c.ret.SetBool(ColorState::BlendEnabled(c.Load<u32>(state), target));
}

void nvnColorStateGetLogicOp(Call& c) {
    c.ret.SetU32(ColorState::LogicOp(c.Load<u32>(c.args.NextPointer())));
}

void nvnColorStateGetAlphaTest(Call& c) {
    c.ret.SetU32(ColorState::AlphaFunc(c.Load<u32>(c.args.NextPointer())));
}

// ---------------------------------------------------------------------------------------------
// NVNchannelMaskState
// ---------------------------------------------------------------------------------------------

void nvnChannelMaskStateSetDefaults(Call& c) {
    c.Store<u32>(c.args.NextPointer(), ChannelMaskState::Defaults());
}

/// SDK (3.2.0 sdk+0x5eadc0, 1.0.0 sdk+0x5c6b90): the nibble shift is `(target * 4) & 31`, so the
/// target index is taken modulo 8 (no range check).
void nvnChannelMaskStateSetChannelMask(Call& c) {
    const u64 state = c.args.NextPointer();
    const s32 target =
        static_cast<s32>(static_cast<u32>(c.args.NextS32()) % ChannelMaskState::TARGETS);
    const bool r = c.args.NextBool();
    const bool g = c.args.NextBool();
    const bool b = c.args.NextBool();
    const bool a = c.args.NextBool();
    const u32 nibble = ChannelMaskState::PackChannels(r, g, b, a);
    Modify<u32>(c, state,
                [&](u32 w) { return ChannelMaskState::WithChannels(w, target, nibble); });
}

void nvnChannelMaskStateGetChannelMask(Call& c) {
    const u64 state = c.args.NextPointer();
    const s32 target = c.args.NextS32();
    const u64 r = c.args.NextPointer();
    const u64 g = c.args.NextPointer();
    const u64 b = c.args.NextPointer();
    const u64 a = c.args.NextPointer();
    const u32 nibble = ChannelMaskState::Channels(c.Load<u32>(state), target);
    if (r != 0) {
        c.Store<u8>(r, (nibble & 1) != 0);
    }
    if (g != 0) {
        c.Store<u8>(g, (nibble & 2) != 0);
    }
    if (b != 0) {
        c.Store<u8>(b, (nibble & 4) != 0);
    }
    if (a != 0) {
        c.Store<u8>(a, (nibble & 8) != 0);
    }
}

// ---------------------------------------------------------------------------------------------
// NVNmultisampleState (10 u8 fields; byte 10 holds the NVN2 coverage truncation flag)
// ---------------------------------------------------------------------------------------------

void StoreMultisampleByte(Call& c, u64 state, u32 offset, u8 value) {
    c.Store<u8>(state + offset, value);
}

u8 LoadMultisampleByte(Call& c, u64 state, u32 offset) {
    return c.Load<u8>(state + offset);
}

/// SDK (3.2.0 and 1.0.0): state word 5 = multisample enable | alpha-to-coverage dither, everything
/// else off; programmable sample locations all (0.5, 0.5), which the host does not keep (see
/// nvnMultisampleStateSetSampleLocations).
void nvnMultisampleStateSetDefaults(Call& c) {
    const u64 state = c.args.NextPointer();
    u8 bytes[MultisampleState::DEFAULTS_SIZE]{};
    bytes[MultisampleState::OFFSET_ENABLE] = 1;
    bytes[MultisampleState::OFFSET_ALPHA_TO_COVERAGE_DITHER] = 1;
    c.mem.Write(state, bytes, sizeof(bytes));
}

void nvnMultisampleStateSetMultisampleEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_ENABLE, c.args.NextBool());
}

void nvnMultisampleStateSetSamples(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_SAMPLES,
                         static_cast<u8>(c.args.NextU32()));
}

void nvnMultisampleStateSetAlphaToCoverageEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_ALPHA_TO_COVERAGE, c.args.NextBool());
}

void nvnMultisampleStateSetAlphaToCoverageDither(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_ALPHA_TO_COVERAGE_DITHER,
                         c.args.NextBool());
}

void nvnMultisampleStateSetRasterSamples(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_RASTER_SAMPLES,
                         static_cast<u8>(c.args.NextU32()));
}

void nvnMultisampleStateSetCoverageModulationMode(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_MODULATION_MODE,
                         static_cast<u8>(c.args.NextU32()));
}

void nvnMultisampleStateSetCoverageToColorEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_TO_COLOR_ENABLE,
                         c.args.NextBool());
}

void nvnMultisampleStateSetCoverageToColorOutput(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_TO_COLOR_OUTPUT,
                         static_cast<u8>(c.args.NextU32()));
}

void nvnMultisampleStateSetSampleLocationsEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_SAMPLE_LOCATIONS_ENABLE,
                         c.args.NextBool());
}

void nvnMultisampleStateSetSampleLocationsGridEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_SAMPLE_LOCATIONS_GRID_ENABLE,
                         c.args.NextBool());
}

// NVN2 (nxnative lacks it). Prototype not in nvn_signatures.inc: (state, NVNboolean) by analogy
// with its siblings. [UNVERIFIED]
void nvnMultisampleStateSetCoverageTruncationEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    StoreMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_TRUNCATION_ENABLE,
                         c.args.NextBool());
}

// nxnative drops the programmable sample positions (the replay does not use them): a documented
// no-op, arguments (state, int start, int count, const float* values).
void nvnMultisampleStateSetSampleLocations(Call& c) {
    static Common::BootCounter logged;
    static_cast<void>(c.args.NextPointer());
    LogOnce(logged, "nvnMultisampleStateSetSampleLocations: programmable sample locations are "
                    "ignored");
}

void nvnMultisampleStateGetMultisampleEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(LoadMultisampleByte(c, state, MultisampleState::OFFSET_ENABLE) != 0);
}

void nvnMultisampleStateGetSamples(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetU32(LoadMultisampleByte(c, state, MultisampleState::OFFSET_SAMPLES));
}

void nvnMultisampleStateGetAlphaToCoverageEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(LoadMultisampleByte(c, state, MultisampleState::OFFSET_ALPHA_TO_COVERAGE) != 0);
}

void nvnMultisampleStateGetAlphaToCoverageDither(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(LoadMultisampleByte(c, state, MultisampleState::OFFSET_ALPHA_TO_COVERAGE_DITHER) !=
                  0);
}

void nvnMultisampleStateGetRasterSamples(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetU32(LoadMultisampleByte(c, state, MultisampleState::OFFSET_RASTER_SAMPLES));
}

void nvnMultisampleStateGetCoverageModulationMode(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetU32(LoadMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_MODULATION_MODE));
}

void nvnMultisampleStateGetCoverageToColorEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(
        LoadMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_TO_COLOR_ENABLE) != 0);
}

void nvnMultisampleStateGetCoverageToColorOutput(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetU32(LoadMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_TO_COLOR_OUTPUT));
}

void nvnMultisampleStateGetSampleLocationsEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(LoadMultisampleByte(c, state, MultisampleState::OFFSET_SAMPLE_LOCATIONS_ENABLE) !=
                  0);
}

void nvnMultisampleStateGetSampleLocationsGridEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(
        LoadMultisampleByte(c, state, MultisampleState::OFFSET_SAMPLE_LOCATIONS_GRID_ENABLE) != 0);
}

void nvnMultisampleStateGetCoverageTruncationEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetBool(
        LoadMultisampleByte(c, state, MultisampleState::OFFSET_COVERAGE_TRUNCATION_ENABLE) != 0);
}

// (state, int* width, int* height). nxnative has no body; the host keeps no programmable grid, so
// it reports an empty one (0 x 0). [UNVERIFIED] against the NVN2 driver.
void nvnMultisampleStateGetSampleLocationsGrid(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const u64 width = c.args.NextPointer();
    const u64 height = c.args.NextPointer();
    if (width != 0) {
        c.Store<s32>(width, 0);
    }
    if (height != 0) {
        c.Store<s32>(height, 0);
    }
}

// NVN2, no prototype anywhere: the argument list is unknown, so nothing is written (a guess could
// scribble over the caller's memory). The host keeps no sample locations.
void nvnMultisampleStateGetSampleLocations(Call& c) {
    static Common::BootCounter logged;
    LogOnce(logged, "nvnMultisampleStateGetSampleLocations: no prototype, nothing returned");
    static_cast<void>(c);
}

// ---------------------------------------------------------------------------------------------
// NVNpolygonState
// ---------------------------------------------------------------------------------------------

void nvnPolygonStateSetDefaults(Call& c) {
    c.Store<u32>(c.args.NextPointer(), PolygonState::Defaults());
}

void nvnPolygonStateSetCullFace(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 face = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return PolygonState::WithCullFace(w, face); });
}

void nvnPolygonStateSetFrontFace(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 face = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return PolygonState::WithFrontFace(w, face); });
}

void nvnPolygonStateSetPolygonMode(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return PolygonState::WithPolygonMode(w, mode); });
}

void nvnPolygonStateSetPolygonOffsetEnables(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 enables = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return PolygonState::WithOffsetEnables(w, enables); });
}

void nvnPolygonStateGetCullFace(Call& c) {
    c.ret.SetU32(PolygonState::CullFace(c.Load<u32>(c.args.NextPointer())));
}

void nvnPolygonStateGetFrontFace(Call& c) {
    c.ret.SetU32(PolygonState::FrontFace(c.Load<u32>(c.args.NextPointer())));
}

void nvnPolygonStateGetPolygonMode(Call& c) {
    c.ret.SetU32(PolygonState::PolygonMode(c.Load<u32>(c.args.NextPointer())));
}

void nvnPolygonStateGetPolygonOffsetEnables(Call& c) {
    c.ret.SetU32(PolygonState::OffsetEnables(c.Load<u32>(c.args.NextPointer())));
}

// ---------------------------------------------------------------------------------------------
// NVNdepthStencilState
// ---------------------------------------------------------------------------------------------

void nvnDepthStencilStateSetDefaults(Call& c) {
    c.Store<u64>(c.args.NextPointer(), DepthStencilState::Defaults());
}

void nvnDepthStencilStateSetDepthTestEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    const bool enable = c.args.NextBool();
    Modify<u64>(c, state, [&](u64 w) { return DepthStencilState::WithDepthTestEnable(w, enable); });
}

void nvnDepthStencilStateSetDepthWriteEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    const bool enable = c.args.NextBool();
    Modify<u64>(c, state,
                [&](u64 w) { return DepthStencilState::WithDepthWriteEnable(w, enable); });
}

void nvnDepthStencilStateSetDepthFunc(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 func = c.args.NextU32();
    Modify<u64>(c, state, [&](u64 w) { return DepthStencilState::WithDepthFunc(w, func); });
}

void nvnDepthStencilStateSetStencilTestEnable(Call& c) {
    const u64 state = c.args.NextPointer();
    const bool enable = c.args.NextBool();
    Modify<u64>(c, state,
                [&](u64 w) { return DepthStencilState::WithStencilTestEnable(w, enable); });
}

void nvnDepthStencilStateSetStencilFunc(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 face = c.args.NextU32();
    const u32 func = c.args.NextU32();
    Modify<u64>(c, state,
                [&](u64 w) { return DepthStencilState::WithStencilFunc(w, face, func); });
}

void nvnDepthStencilStateSetStencilOp(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 face = c.args.NextU32();
    const u32 fail = c.args.NextU32();
    const u32 depth_fail = c.args.NextU32();
    const u32 depth_pass = c.args.NextU32();
    Modify<u64>(c, state, [&](u64 w) {
        return DepthStencilState::WithStencilOps(w, face, fail, depth_fail, depth_pass);
    });
}

void nvnDepthStencilStateGetDepthTestEnable(Call& c) {
    c.ret.SetBool(DepthStencilState::DepthTestEnable(c.Load<u64>(c.args.NextPointer())) != 0);
}

void nvnDepthStencilStateGetDepthWriteEnable(Call& c) {
    c.ret.SetBool(DepthStencilState::DepthWriteEnable(c.Load<u64>(c.args.NextPointer())) != 0);
}

void nvnDepthStencilStateGetDepthFunc(Call& c) {
    c.ret.SetU32(DepthStencilState::DepthFunc(c.Load<u64>(c.args.NextPointer())));
}

void nvnDepthStencilStateGetStencilTestEnable(Call& c) {
    c.ret.SetBool(DepthStencilState::StencilTestEnable(c.Load<u64>(c.args.NextPointer())) != 0);
}

void nvnDepthStencilStateGetStencilFunc(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 face = c.args.NextU32();
    c.ret.SetU32(DepthStencilState::StencilFunc(c.Load<u64>(state),
                                                DepthStencilState::ReadsBackFace(face)));
}

void nvnDepthStencilStateGetStencilOp(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 face = c.args.NextU32();
    const u64 fail = c.args.NextPointer();
    const u64 depth_fail = c.args.NextPointer();
    const u64 depth_pass = c.args.NextPointer();
    const u64 w = c.Load<u64>(state);
    const bool back = DepthStencilState::ReadsBackFace(face);
    if (fail != 0) {
        c.Store<u32>(fail, DepthStencilState::FailOp(w, back));
    }
    if (depth_fail != 0) {
        c.Store<u32>(depth_fail, DepthStencilState::DepthFailOp(w, back));
    }
    if (depth_pass != 0) {
        c.Store<u32>(depth_pass, DepthStencilState::DepthPassOp(w, back));
    }
}

// ---------------------------------------------------------------------------------------------
// NVNvertexAttribState / NVNvertexStreamState
// ---------------------------------------------------------------------------------------------

void nvnVertexAttribStateSetDefaults(Call& c) {
    c.Store<u32>(c.args.NextPointer(), VertexAttribState::Defaults());
}

/// SDK (3.2.0 sdk+0x5eaf10, 1.0.0 sdk+0x5c6ce0): format in 10 bits, relative offset in 14 bits
/// (larger offsets spill into the format bits), plus an "attribute set" bit. This word keeps 8
/// format bits (every vertex format fits) and 19 offset bits, so every offset the SDK keeps is
/// kept unchanged; the replay treats format 0 (NONE) as disabled.
void nvnVertexAttribStateSetFormat(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 format = c.args.NextU32();
    const s64 offset = c.args.NextS64();
    if (offset < 0 || offset > static_cast<s64>(VertexAttribState::MAX_OFFSET)) {
        static Common::BootCounter logged;
        LogOnce(logged, "vertex attribute offset exceeds the packed 19-bit range, truncated");
    }
    Modify<u32>(c, state, [&](u32 w) {
        return VertexAttribState::WithOffset(VertexAttribState::WithFormat(w, format),
                                             static_cast<u64>(offset));
    });
}

void nvnVertexAttribStateSetStreamIndex(Call& c) {
    const u64 state = c.args.NextPointer();
    const u32 stream = c.args.NextU32();
    Modify<u32>(c, state, [&](u32 w) { return VertexAttribState::WithStreamIndex(w, stream); });
}

void nvnVertexAttribStateGetFormat(Call& c) {
    const u64 state = c.args.NextPointer();
    const u64 format = c.args.NextPointer();
    const u64 offset = c.args.NextPointer();
    const u32 w = c.Load<u32>(state);
    if (format != 0) {
        c.Store<u32>(format, VertexAttribState::Format(w));
    }
    if (offset != 0) {
        c.Store<s64>(offset, static_cast<s64>(VertexAttribState::Offset(w)));
    }
}

void nvnVertexAttribStateGetStreamIndex(Call& c) {
    c.ret.SetU32(VertexAttribState::StreamIndex(c.Load<u32>(c.args.NextPointer())));
}

void nvnVertexStreamStateSetDefaults(Call& c) {
    c.Store<u64>(c.args.NextPointer(), 0);
}

void nvnVertexStreamStateSetStride(Call& c) {
    const u64 state = c.args.NextPointer();
    c.Store<u32>(state + VERTEX_STREAM_STATE_STRIDE_OFFSET, static_cast<u32>(c.args.NextU64()));
}

void nvnVertexStreamStateSetDivisor(Call& c) {
    const u64 state = c.args.NextPointer();
    c.Store<u32>(state + VERTEX_STREAM_STATE_DIVISOR_OFFSET, c.args.NextU32());
}

// Monster Hunter Rise reads its stream states back while it builds a draw (nxnative comment).
void nvnVertexStreamStateGetStride(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetU64(c.Load<u32>(state + VERTEX_STREAM_STATE_STRIDE_OFFSET));
}

void nvnVertexStreamStateGetDivisor(Call& c) {
    const u64 state = c.args.NextPointer();
    c.ret.SetU32(c.Load<u32>(state + VERTEX_STREAM_STATE_DIVISOR_OFFSET));
}

} // namespace

void RegisterStateFunctions(Registry& registry) {
    NVN_HLE(registry, nvnBlendStateSetDefaults);
    NVN_HLE(registry, nvnBlendStateSetBlendTarget);
    NVN_HLE(registry, nvnBlendStateSetBlendFunc);
    NVN_HLE(registry, nvnBlendStateSetBlendEquation);
    NVN_HLE(registry, nvnBlendStateSetAdvancedMode);
    NVN_HLE(registry, nvnBlendStateSetAdvancedOverlap);
    NVN_HLE(registry, nvnBlendStateSetAdvancedPremultipliedSrc);
    NVN_HLE(registry, nvnBlendStateSetAdvancedNormalizedDst);
    NVN_HLE(registry, nvnBlendStateGetBlendTarget);
    NVN_HLE(registry, nvnBlendStateGetBlendFunc);
    NVN_HLE(registry, nvnBlendStateGetBlendEquation);
    NVN_HLE(registry, nvnBlendStateGetAdvancedMode);
    NVN_HLE(registry, nvnBlendStateGetAdvancedOverlap);
    NVN_HLE(registry, nvnBlendStateGetAdvancedPremultipliedSrc);
    NVN_HLE(registry, nvnBlendStateGetAdvancedNormalizedDst);

    NVN_HLE(registry, nvnColorStateSetDefaults);
    NVN_HLE(registry, nvnColorStateSetBlendEnable);
    NVN_HLE(registry, nvnColorStateSetLogicOp);
    NVN_HLE(registry, nvnColorStateSetAlphaTest);
    NVN_HLE(registry, nvnColorStateGetBlendEnable);
    NVN_HLE(registry, nvnColorStateGetLogicOp);
    NVN_HLE(registry, nvnColorStateGetAlphaTest);

    NVN_HLE(registry, nvnChannelMaskStateSetDefaults);
    NVN_HLE(registry, nvnChannelMaskStateSetChannelMask);
    NVN_HLE(registry, nvnChannelMaskStateGetChannelMask);

    NVN_HLE(registry, nvnMultisampleStateSetDefaults);
    NVN_HLE(registry, nvnMultisampleStateSetMultisampleEnable);
    NVN_HLE(registry, nvnMultisampleStateSetSamples);
    NVN_HLE(registry, nvnMultisampleStateSetAlphaToCoverageEnable);
    NVN_HLE(registry, nvnMultisampleStateSetAlphaToCoverageDither);
    NVN_HLE(registry, nvnMultisampleStateSetRasterSamples);
    NVN_HLE(registry, nvnMultisampleStateSetCoverageModulationMode);
    NVN_HLE(registry, nvnMultisampleStateSetCoverageToColorEnable);
    NVN_HLE(registry, nvnMultisampleStateSetCoverageToColorOutput);
    NVN_HLE(registry, nvnMultisampleStateSetSampleLocationsEnable);
    NVN_HLE(registry, nvnMultisampleStateSetSampleLocationsGridEnable);
    NVN_HLE(registry, nvnMultisampleStateSetSampleLocations);
    NVN_HLE(registry, nvnMultisampleStateSetCoverageTruncationEnable);
    NVN_HLE(registry, nvnMultisampleStateGetMultisampleEnable);
    NVN_HLE(registry, nvnMultisampleStateGetSamples);
    NVN_HLE(registry, nvnMultisampleStateGetAlphaToCoverageEnable);
    NVN_HLE(registry, nvnMultisampleStateGetAlphaToCoverageDither);
    NVN_HLE(registry, nvnMultisampleStateGetRasterSamples);
    NVN_HLE(registry, nvnMultisampleStateGetCoverageModulationMode);
    NVN_HLE(registry, nvnMultisampleStateGetCoverageToColorEnable);
    NVN_HLE(registry, nvnMultisampleStateGetCoverageToColorOutput);
    NVN_HLE(registry, nvnMultisampleStateGetSampleLocationsEnable);
    NVN_HLE(registry, nvnMultisampleStateGetSampleLocationsGridEnable);
    NVN_HLE(registry, nvnMultisampleStateGetSampleLocationsGrid);
    NVN_HLE(registry, nvnMultisampleStateGetSampleLocations);
    NVN_HLE(registry, nvnMultisampleStateGetCoverageTruncationEnable);

    NVN_HLE(registry, nvnPolygonStateSetDefaults);
    NVN_HLE(registry, nvnPolygonStateSetCullFace);
    NVN_HLE(registry, nvnPolygonStateSetFrontFace);
    NVN_HLE(registry, nvnPolygonStateSetPolygonMode);
    NVN_HLE(registry, nvnPolygonStateSetPolygonOffsetEnables);
    NVN_HLE(registry, nvnPolygonStateGetCullFace);
    NVN_HLE(registry, nvnPolygonStateGetFrontFace);
    NVN_HLE(registry, nvnPolygonStateGetPolygonMode);
    NVN_HLE(registry, nvnPolygonStateGetPolygonOffsetEnables);

    NVN_HLE(registry, nvnDepthStencilStateSetDefaults);
    NVN_HLE(registry, nvnDepthStencilStateSetDepthTestEnable);
    NVN_HLE(registry, nvnDepthStencilStateSetDepthWriteEnable);
    NVN_HLE(registry, nvnDepthStencilStateSetDepthFunc);
    NVN_HLE(registry, nvnDepthStencilStateSetStencilTestEnable);
    NVN_HLE(registry, nvnDepthStencilStateSetStencilFunc);
    NVN_HLE(registry, nvnDepthStencilStateSetStencilOp);
    NVN_HLE(registry, nvnDepthStencilStateGetDepthTestEnable);
    NVN_HLE(registry, nvnDepthStencilStateGetDepthWriteEnable);
    NVN_HLE(registry, nvnDepthStencilStateGetDepthFunc);
    NVN_HLE(registry, nvnDepthStencilStateGetStencilTestEnable);
    NVN_HLE(registry, nvnDepthStencilStateGetStencilFunc);
    NVN_HLE(registry, nvnDepthStencilStateGetStencilOp);

    NVN_HLE(registry, nvnVertexAttribStateSetDefaults);
    NVN_HLE(registry, nvnVertexAttribStateSetFormat);
    NVN_HLE(registry, nvnVertexAttribStateSetStreamIndex);
    NVN_HLE(registry, nvnVertexAttribStateGetFormat);
    NVN_HLE(registry, nvnVertexAttribStateGetStreamIndex);

    NVN_HLE(registry, nvnVertexStreamStateSetDefaults);
    NVN_HLE(registry, nvnVertexStreamStateSetStride);
    NVN_HLE(registry, nvnVertexStreamStateSetDivisor);
    NVN_HLE(registry, nvnVertexStreamStateGetStride);
    NVN_HLE(registry, nvnVertexStreamStateGetDivisor);
}

} // namespace Core::NvnHle::Hle
