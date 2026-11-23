// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <cstddef>
#include <cstring>
#include <optional>

#include "core/hle/nvn/hle_call.h"
#include "video_core/nvn/sink.h"

// Sampler family: NVNsamplerBuilder (kept in place, nxnative `samp_builder_t`) and NVNsampler
// (host state in the Recorder, tagged at +0).
namespace Core::NvnHle::Hle {
namespace {

using ::VideoCommon::Nvn::SamplerDesc;

/// nxnative `samp_par`: the sampler parameters, shared by the builder and the sampler object. All
/// enum values are below 256; `reduction` takes the padding byte.
struct GuestSamplerParams {
    u8 min_filter;
    u8 mag_filter;
    u8 wrap_s;
    u8 wrap_t;
    u8 wrap_r;
    u8 compare_mode;
    u8 compare_func;
    u8 reduction;
    f32 lod_min;
    f32 lod_max;
    f32 lod_bias;
    f32 max_anisotropy;
    f32 border[4];
};
static_assert(sizeof(GuestSamplerParams) == 40);

/// `samp_builder_t` (0x30 bytes; the SDK's builder is larger).
struct GuestSamplerBuilder {
    u64 device;
    GuestSamplerParams params;
};
static_assert(sizeof(GuestSamplerBuilder) == 0x30);

constexpr u64 SbDevice = offsetof(GuestSamplerBuilder, device);
constexpr u64 SbParams = offsetof(GuestSamplerBuilder, params);
constexpr u64 SbFilters = SbParams + offsetof(GuestSamplerParams, min_filter); // min, mag
constexpr u64 SbWrap = SbParams + offsetof(GuestSamplerParams, wrap_s);        // s, t, r
constexpr u64 SbCompare = SbParams + offsetof(GuestSamplerParams, compare_mode); // mode, func
constexpr u64 SbReduction = SbParams + offsetof(GuestSamplerParams, reduction);
constexpr u64 SbLodClamp = SbParams + offsetof(GuestSamplerParams, lod_min); // min, max
constexpr u64 SbLodBias = SbParams + offsetof(GuestSamplerParams, lod_bias);
constexpr u64 SbAnisotropy = SbParams + offsetof(GuestSamplerParams, max_anisotropy);
constexpr u64 SbBorder = SbParams + offsetof(GuestSamplerParams, border);

// NVN enum values the defaults use (nvn.h).
constexpr u8 MIN_FILTER_LINEAR = 1;
constexpr u8 MAG_FILTER_LINEAR = 1;
constexpr u8 WRAP_MODE_REPEAT = 1;
constexpr u8 COMPARE_FUNC_LESS = 2;

SamplerDesc ToDesc(const GuestSamplerParams& p) {
    SamplerDesc desc{};
    desc.min_filter = p.min_filter;
    desc.mag_filter = p.mag_filter;
    desc.wrap_s = p.wrap_s;
    desc.wrap_t = p.wrap_t;
    desc.wrap_r = p.wrap_r;
    desc.compare_mode = p.compare_mode;
    desc.compare_func = p.compare_func;
    desc.reduction = p.reduction;
    desc.lod_min = p.lod_min;
    desc.lod_max = p.lod_max;
    desc.lod_bias = p.lod_bias;
    desc.max_anisotropy = p.max_anisotropy;
    std::memcpy(desc.border, p.border, sizeof(desc.border));
    return desc;
}

// ---- Builder: setters -----------------------------------------------------------------------------

/// SDK (3.2.0 sdk+0x5ea460, 1.0.0 sdk+0x5c6220, identical; read back through the SDK builder
/// getters sdk+0x5ea5e0..0x5ea730): min and mag LINEAR (no mipmapping), wrap REPEAT on s/t/r,
/// compare mode NONE with func LESS, reduction AVERAGE, LOD clamp [0, 1000], LOD bias 0, max
/// anisotropy 1, border color 0, LOD snap 0; the device is kept.
void nvnSamplerBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    GuestSamplerBuilder defaults;
    std::memset(&defaults, 0, sizeof(defaults));
    defaults.device = c.Load<u64>(builder + SbDevice);
    GuestSamplerParams& p = defaults.params;
    p.min_filter = MIN_FILTER_LINEAR;
    p.mag_filter = MAG_FILTER_LINEAR;
    p.wrap_s = p.wrap_t = p.wrap_r = WRAP_MODE_REPEAT;
    p.compare_func = COMPARE_FUNC_LESS;
    p.lod_min = 0.0f;
    p.lod_max = 1000.0f;
    p.max_anisotropy = 1.0f;
    c.Store(builder, defaults);
}

void nvnSamplerBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + SbDevice, c.args.NextPointer());
}

void nvnSamplerBuilderSetMinMagFilter(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 min = c.args.NextS32();
    const s32 mag = c.args.NextS32();
    c.Store(builder + SbFilters, std::array<u8, 2>{static_cast<u8>(min), static_cast<u8>(mag)});
}

void nvnSamplerBuilderSetWrapMode(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 s = c.args.NextS32();
    const s32 t = c.args.NextS32();
    const s32 r = c.args.NextS32();
    c.Store(builder + SbWrap,
            std::array<u8, 3>{static_cast<u8>(s), static_cast<u8>(t), static_cast<u8>(r)});
}

void nvnSamplerBuilderSetLodClamp(Call& c) {
    const u64 builder = c.args.NextPointer();
    const f32 lo = c.args.NextF32();
    const f32 hi = c.args.NextF32();
    c.Store(builder + SbLodClamp, std::array<f32, 2>{lo, hi});
}

void nvnSamplerBuilderSetLodBias(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<f32>(builder + SbLodBias, c.args.NextF32());
}

void nvnSamplerBuilderSetCompare(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 mode = c.args.NextS32();
    const s32 func = c.args.NextS32();
    c.Store(builder + SbCompare, std::array<u8, 2>{static_cast<u8>(mode), static_cast<u8>(func)});
}

void nvnSamplerBuilderSetBorderColor(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 color = c.args.NextPointer();
    std::array<f32, 4> values{};
    if (c.mem.Read(color, values.data(), sizeof(values))) {
        c.Store(builder + SbBorder, values);
    }
}

/// Integer border colors are stored as the float with the same numeric value, not bit-cast: the
/// replay selects its fixed border colors numerically (nxnative), and bit-casting integer 1 would
/// select transparent black instead of opaque black / white.
void nvnSamplerBuilderSetBorderColori(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 color = c.args.NextPointer();
    std::array<s32, 4> source{};
    if (c.mem.Read(color, source.data(), sizeof(source))) {
        c.Store(builder + SbBorder,
                std::array<f32, 4>{static_cast<f32>(source[0]), static_cast<f32>(source[1]),
                                   static_cast<f32>(source[2]), static_cast<f32>(source[3])});
    }
}

void nvnSamplerBuilderSetBorderColorui(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 color = c.args.NextPointer();
    std::array<u32, 4> source{};
    if (c.mem.Read(color, source.data(), sizeof(source))) {
        c.Store(builder + SbBorder,
                std::array<f32, 4>{static_cast<f32>(source[0]), static_cast<f32>(source[1]),
                                   static_cast<f32>(source[2]), static_cast<f32>(source[3])});
    }
}

void nvnSamplerBuilderSetMaxAnisotropy(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<f32>(builder + SbAnisotropy, c.args.NextF32());
}

void nvnSamplerBuilderSetReductionFilter(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u8>(builder + SbReduction, static_cast<u8>(c.args.NextS32()));
}

/// No Metal / Vulkan equivalent: nxnative ignores it. The builder has no room for it either.
void nvnSamplerBuilderSetLodSnap(Call& c) {
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextF32());
}

// ---- Builder: getters -----------------------------------------------------------------------------

void nvnSamplerBuilderGetDevice(Call& c) {
    // No prototype in the nxnative header: the device pointer set with SetDevice.
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + SbDevice));
}

void nvnSamplerBuilderGetMinMagFilter(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 min = c.args.NextPointer();
    const u64 mag = c.args.NextPointer();
    const auto filters = c.Load<std::array<u8, 2>>(builder + SbFilters);
    if (min) {
        c.Store<s32>(min, filters[0]);
    }
    if (mag) {
        c.Store<s32>(mag, filters[1]);
    }
}

void nvnSamplerBuilderGetWrapMode(Call& c) {
    const u64 builder = c.args.NextPointer();
    const auto wrap = c.Load<std::array<u8, 3>>(builder + SbWrap);
    for (const u8 mode : wrap) {
        if (const u64 out = c.args.NextPointer()) {
            c.Store<s32>(out, mode);
        }
    }
}

/// Bayonetta 3 reads the clamp back while building its samplers.
void nvnSamplerBuilderGetLodClamp(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 lo = c.args.NextPointer();
    const u64 hi = c.args.NextPointer();
    const auto clamp = c.Load<std::array<f32, 2>>(builder + SbLodClamp);
    if (lo) {
        c.Store<f32>(lo, clamp[0]);
    }
    if (hi) {
        c.Store<f32>(hi, clamp[1]);
    }
}

void nvnSamplerBuilderGetLodBias(Call& c) {
    c.ret.SetF32(c.Load<f32>(c.args.NextPointer() + SbLodBias));
}

void nvnSamplerBuilderGetCompare(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 mode = c.args.NextPointer();
    const u64 func = c.args.NextPointer();
    const auto compare = c.Load<std::array<u8, 2>>(builder + SbCompare);
    if (mode) {
        c.Store<s32>(mode, compare[0]);
    }
    if (func) {
        c.Store<s32>(func, compare[1]);
    }
}

void nvnSamplerBuilderGetBorderColor(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 out = c.args.NextPointer();
    c.Store(out, c.Load<std::array<f32, 4>>(builder + SbBorder));
}

void nvnSamplerBuilderGetBorderColori(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 out = c.args.NextPointer();
    const auto border = c.Load<std::array<f32, 4>>(builder + SbBorder);
    c.Store(out, std::array<s32, 4>{static_cast<s32>(border[0]), static_cast<s32>(border[1]),
                                    static_cast<s32>(border[2]), static_cast<s32>(border[3])});
}

void nvnSamplerBuilderGetBorderColorui(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 out = c.args.NextPointer();
    const auto border = c.Load<std::array<f32, 4>>(builder + SbBorder);
    c.Store(out, std::array<u32, 4>{static_cast<u32>(border[0]), static_cast<u32>(border[1]),
                                    static_cast<u32>(border[2]), static_cast<u32>(border[3])});
}

void nvnSamplerBuilderGetMaxAnisotropy(Call& c) {
    c.ret.SetF32(c.Load<f32>(c.args.NextPointer() + SbAnisotropy));
}

void nvnSamplerBuilderGetReductionFilter(Call& c) {
    c.ret.SetU32(c.Load<u8>(c.args.NextPointer() + SbReduction));
}

/// The snap is not kept (see SetLodSnap): 0, the NVN default.
void nvnSamplerBuilderGetLodSnap(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetF32(0.0f);
}

// ---- Sampler objects ------------------------------------------------------------------------------

void nvnSamplerInitialize(Call& c) {
    const u64 sampler = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const auto params = c.Load<GuestSamplerParams>(builder + SbParams);
    c.ret.SetBool(c.rec.SamplerInitialize(sampler, ToDesc(params)));
}

void nvnSamplerFinalize(Call& c) {
    c.rec.SamplerFinalize(c.args.NextPointer());
}

void nvnSamplerSetDebugLabel(Call& c) {
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextPointer());
}

// The getters read the Recorder's copy; a sampler that was never initialized, or was finalized,
// answers zeros.

void nvnSamplerGetMinMagFilter(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const u64 min = c.args.NextPointer();
    const u64 mag = c.args.NextPointer();
    if (min) {
        c.Store<s32>(min, desc ? desc->min_filter : 0);
    }
    if (mag) {
        c.Store<s32>(mag, desc ? desc->mag_filter : 0);
    }
}

void nvnSamplerGetWrapMode(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const std::array<u8, 3> wrap{desc ? desc->wrap_s : u8{0}, desc ? desc->wrap_t : u8{0},
                                 desc ? desc->wrap_r : u8{0}};
    for (const u8 mode : wrap) {
        if (const u64 out = c.args.NextPointer()) {
            c.Store<s32>(out, mode);
        }
    }
}

void nvnSamplerGetLodClamp(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const u64 lo = c.args.NextPointer();
    const u64 hi = c.args.NextPointer();
    if (lo) {
        c.Store<f32>(lo, desc ? desc->lod_min : 0.0f);
    }
    if (hi) {
        c.Store<f32>(hi, desc ? desc->lod_max : 0.0f);
    }
}

void nvnSamplerGetLodBias(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    c.ret.SetF32(desc ? desc->lod_bias : 0.0f);
}

void nvnSamplerGetCompare(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const u64 mode = c.args.NextPointer();
    const u64 func = c.args.NextPointer();
    if (mode) {
        c.Store<s32>(mode, desc ? desc->compare_mode : 0);
    }
    if (func) {
        c.Store<s32>(func, desc ? desc->compare_func : 0);
    }
}

void nvnSamplerGetBorderColor(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const u64 out = c.args.NextPointer();
    std::array<f32, 4> border{};
    if (desc) {
        std::memcpy(border.data(), desc->border, sizeof(border));
    }
    c.Store(out, border);
}

void nvnSamplerGetBorderColori(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const u64 out = c.args.NextPointer();
    std::array<s32, 4> border{};
    if (desc) {
        for (size_t i = 0; i < border.size(); ++i) {
            border[i] = static_cast<s32>(desc->border[i]);
        }
    }
    c.Store(out, border);
}

void nvnSamplerGetBorderColorui(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    const u64 out = c.args.NextPointer();
    std::array<u32, 4> border{};
    if (desc) {
        for (size_t i = 0; i < border.size(); ++i) {
            border[i] = static_cast<u32>(desc->border[i]);
        }
    }
    c.Store(out, border);
}

void nvnSamplerGetMaxAnisotropy(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    c.ret.SetF32(desc ? desc->max_anisotropy : 0.0f);
}

void nvnSamplerGetReductionFilter(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    c.ret.SetU32(desc ? desc->reduction : 0);
}

/// The unique id of the host object (not in nxnative; the NVN debug id only has to be unique).
void nvnSamplerGetDebugID(Call& c) {
    const auto desc = c.rec.GetSampler(c.args.NextPointer());
    c.ret.SetU64(desc ? desc->serial : 0);
}

/// Whether two samplers have the same state (games use it to deduplicate sampler pool entries).
void nvnSamplerCompare(Call& c) {
    const auto a = c.rec.GetSampler(c.args.NextPointer());
    const auto b = c.rec.GetSampler(c.args.NextPointer());
    if (!a || !b) {
        c.ret.SetBool(false);
        return;
    }
    c.ret.SetBool(a->min_filter == b->min_filter && a->mag_filter == b->mag_filter &&
                  a->wrap_s == b->wrap_s && a->wrap_t == b->wrap_t && a->wrap_r == b->wrap_r &&
                  a->compare_mode == b->compare_mode && a->compare_func == b->compare_func &&
                  a->reduction == b->reduction && a->lod_min == b->lod_min &&
                  a->lod_max == b->lod_max && a->lod_bias == b->lod_bias &&
                  a->max_anisotropy == b->max_anisotropy &&
                  std::memcmp(a->border, b->border, sizeof(a->border)) == 0);
}

} // namespace

void RegisterSamplerFunctions(Registry& registry) {
    NVN_HLE(registry, nvnSamplerBuilderSetDevice);
    NVN_HLE(registry, nvnSamplerBuilderSetDefaults);
    NVN_HLE(registry, nvnSamplerBuilderSetMinMagFilter);
    NVN_HLE(registry, nvnSamplerBuilderSetWrapMode);
    NVN_HLE(registry, nvnSamplerBuilderSetLodClamp);
    NVN_HLE(registry, nvnSamplerBuilderSetLodBias);
    NVN_HLE(registry, nvnSamplerBuilderSetCompare);
    NVN_HLE(registry, nvnSamplerBuilderSetBorderColor);
    NVN_HLE(registry, nvnSamplerBuilderSetBorderColori);
    NVN_HLE(registry, nvnSamplerBuilderSetBorderColorui);
    NVN_HLE(registry, nvnSamplerBuilderSetMaxAnisotropy);
    NVN_HLE(registry, nvnSamplerBuilderSetReductionFilter);
    NVN_HLE(registry, nvnSamplerBuilderSetLodSnap);
    NVN_HLE(registry, nvnSamplerBuilderGetDevice);
    NVN_HLE(registry, nvnSamplerBuilderGetMinMagFilter);
    NVN_HLE(registry, nvnSamplerBuilderGetWrapMode);
    NVN_HLE(registry, nvnSamplerBuilderGetLodClamp);
    NVN_HLE(registry, nvnSamplerBuilderGetLodBias);
    NVN_HLE(registry, nvnSamplerBuilderGetCompare);
    NVN_HLE(registry, nvnSamplerBuilderGetBorderColor);
    NVN_HLE(registry, nvnSamplerBuilderGetBorderColori);
    NVN_HLE(registry, nvnSamplerBuilderGetBorderColorui);
    NVN_HLE(registry, nvnSamplerBuilderGetMaxAnisotropy);
    NVN_HLE(registry, nvnSamplerBuilderGetReductionFilter);
    NVN_HLE(registry, nvnSamplerBuilderGetLodSnap);
    NVN_HLE(registry, nvnSamplerInitialize);
    NVN_HLE(registry, nvnSamplerFinalize);
    NVN_HLE(registry, nvnSamplerSetDebugLabel);
    NVN_HLE(registry, nvnSamplerGetMinMagFilter);
    NVN_HLE(registry, nvnSamplerGetWrapMode);
    NVN_HLE(registry, nvnSamplerGetLodClamp);
    NVN_HLE(registry, nvnSamplerGetLodBias);
    NVN_HLE(registry, nvnSamplerGetCompare);
    NVN_HLE(registry, nvnSamplerGetBorderColor);
    NVN_HLE(registry, nvnSamplerGetBorderColori);
    NVN_HLE(registry, nvnSamplerGetBorderColorui);
    NVN_HLE(registry, nvnSamplerGetMaxAnisotropy);
    NVN_HLE(registry, nvnSamplerGetReductionFilter);
    NVN_HLE(registry, nvnSamplerCompare);
    NVN_HLE(registry, nvnSamplerGetDebugID);
}

} // namespace Core::NvnHle::Hle
