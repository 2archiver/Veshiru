// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <atomic>
#include <limits>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"

// Command buffer state and binds: everything a later draw consumes (render targets, clears, viewport
// state, programs, state objects, vertex/uniform/storage buffers, texture handles, pools, stencil
// and the small rasterizer switches). Each body decodes its arguments in prototype order and hands
// them to the Recorder, which copies the data into the command buffer's list. Behaviour ported from
// nxnative hle_nvn.c; where nxnative records into the game's command memory, the Recorder keeps the
// host-side list instead.
//
// The bodies are the hottest in the HLE (hundreds of calls a frame each): no allocation, no locks,
// no logging except once per unexpected input.
namespace Core::NvnHle::Hle {
namespace {

using VideoCommon::Nvn::DepthRange;
using VideoCommon::Nvn::Op;
using VideoCommon::Nvn::ScissorRect;
using VideoCommon::Nvn::SetDepthRangesPayload;
using VideoCommon::Nvn::SetScissorsPayload;
using VideoCommon::Nvn::SetViewportsPayload;
using VideoCommon::Nvn::SetViewportSwizzlesPayload;
using VideoCommon::Nvn::StateKind;
using VideoCommon::Nvn::ViewportRect;

#define NVN_LOG_ONCE(level, ...)                                                                   \
    do {                                                                                           \
        static Common::BootCounter nvn_once;                                                       \
        if (nvn_once.First()) {                                                                    \
            level(Core, __VA_ARGS__);                                                              \
        }                                                                                          \
    } while (false)

constexpr s32 MaxRenderTargets = 8;
constexpr s32 MaxViewports = 16;

// ---- Render targets and clears ------------------------------------------------------------------

void nvnCommandBufferSetRenderTargets(Call& c) {
    const u64 cmd = c.args.NextPointer();
    s32 count = c.args.NextS32();
    const u64 colors = c.args.NextPointer();
    const u64 views = c.args.NextPointer();
    const u64 depth = c.args.NextPointer();
    const u64 depth_view = c.args.NextPointer();
    if (count < 0 || count > MaxRenderTargets) {
        // nxnative clamps (the host keeps eight color targets) and logs.
        NVN_LOG_ONCE(LOG_WARNING, "[nvn] SetRenderTargets count {} clamped to 0..8", count);
        count = count < 0 ? 0 : MaxRenderTargets;
    }
    c.rec.SetRenderTargets(cmd, count, colors, views, depth, depth_view);
}

/// ClearColor (floats), ClearColori (ints), ClearColorui (uints). The Recorder keeps the four
/// values as raw bit patterns plus an `is_integer` flag; nxnative converted the integers to
/// floats at record time instead.
template <bool Integer>
void ClearColorImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 index = c.args.NextS32();
    const u64 values = c.args.NextPointer();
    const u32 mask = c.args.NextU32();
    c.rec.ClearColor(cmd, index, values, mask, Integer);
}

void nvnCommandBufferClearDepthStencil(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const float depth = c.args.NextF32();
    const bool depth_mask = c.args.NextBool();
    const s32 stencil = c.args.NextS32();
    const s32 stencil_mask = c.args.NextS32();
    c.rec.ClearDepthStencil(cmd, depth, depth_mask ? 1 : 0, stencil,
                            static_cast<u32>(stencil_mask));
}

// ---- Viewports, scissors, depth ranges ------------------------------------------------------------
// The single-viewport forms have no pointer to give the Recorder's array methods: they emit the
// same record (prefix + one element) through EmitRaw, from a stack buffer.

void nvnCommandBufferSetViewport(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 x = c.args.NextS32();
    const s32 y = c.args.NextS32();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    struct {
        SetViewportsPayload prefix;
        ViewportRect rect;
    } record{{0, 1},
             {static_cast<f32>(x), static_cast<f32>(y), static_cast<f32>(width),
              static_cast<f32>(height)}};
    c.rec.EmitRaw(cmd, static_cast<u16>(Op::SetViewports), 0, &record, sizeof(record));
}

void nvnCommandBufferSetViewports(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 values = c.args.NextPointer();
    c.rec.SetViewports(cmd, first, count, values);
}

void nvnCommandBufferSetScissor(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 x = c.args.NextS32();
    const s32 y = c.args.NextS32();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    struct {
        SetScissorsPayload prefix;
        ScissorRect rect;
    } record{{0, 1}, {x, y, width, height}};
    c.rec.EmitRaw(cmd, static_cast<u16>(Op::SetScissors), 0, &record, sizeof(record));
}

void nvnCommandBufferSetScissors(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 values = c.args.NextPointer();
    c.rec.SetScissors(cmd, first, count, values);
}

void nvnCommandBufferSetDepthRange(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const float near_plane = c.args.NextF32();
    const float far_plane = c.args.NextF32();
    struct {
        SetDepthRangesPayload prefix;
        DepthRange range;
    } record{{0, 1}, {near_plane, far_plane}};
    c.rec.EmitRaw(cmd, static_cast<u16>(Op::SetDepthRanges), 0, &record, sizeof(record));
}

void nvnCommandBufferSetDepthRanges(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 values = c.args.NextPointer();
    c.rec.SetDepthRanges(cmd, first, count, values);
}

/// The game passes `count` viewports * 4 NVNviewportSwizzle (32-bit enums: x, y, z, w each). The
/// record holds one byte per component, so the enums are narrowed here (Recorder::SetViewportSwizzles
/// would read the array as bytes).
void nvnCommandBufferSetViewportSwizzles(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 values = c.args.NextPointer();
    if (first < 0 || count < 0 || count > MaxViewports - first) {
        NVN_LOG_ONCE(LOG_WARNING, "[nvn] SetViewportSwizzles first={} count={} out of range", first,
                     count);
        return;
    }
    struct {
        SetViewportSwizzlesPayload prefix;
        std::array<u8, 4 * MaxViewports> swizzles;
    } record{};
    record.prefix = {static_cast<u32>(first), static_cast<u32>(count)};
    std::array<u32, 4 * MaxViewports> enums{};
    const size_t components = 4 * static_cast<size_t>(count);
    if (components != 0 && !c.mem.Read(values, enums.data(), components * sizeof(u32))) {
        NVN_LOG_ONCE(LOG_WARNING, "[nvn] SetViewportSwizzles: unreadable array at {:#x}", values);
        return;
    }
    for (size_t i = 0; i < components; ++i) {
        record.swizzles[i] = static_cast<u8>(enums[i] & 7);
    }
    c.rec.EmitRaw(cmd, static_cast<u16>(Op::SetViewportSwizzles), 0, &record,
                  static_cast<u32>(sizeof(record.prefix) + components));
}

// ---- Programs and state objects ---------------------------------------------------------------------

void nvnCommandBufferBindProgram(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 program = c.args.NextPointer();
    const s32 stages = c.args.NextS32();
    c.rec.BindProgram(cmd, program, static_cast<u32>(stages));
}

/// BindBlendState and friends: the Recorder copies the packed in-place state object.
template <StateKind Kind>
void BindStateImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 state = c.args.NextPointer();
    c.rec.BindState(cmd, Kind, state);
}

void nvnCommandBufferBindVertexAttribState(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 states = c.args.NextPointer();
    c.rec.BindVertexAttribState(cmd, count, states);
}

void nvnCommandBufferBindVertexStreamState(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 states = c.args.NextPointer();
    c.rec.BindVertexStreamState(cmd, count, states);
}

// ---- Buffers ----------------------------------------------------------------------------------------

void nvnCommandBufferBindVertexBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 index = c.args.NextS32();
    const u64 address = c.args.NextU64();
    const u64 size = c.args.NextU64();
    c.rec.BindVertexBuffer(cmd, index, address, size);
}

void nvnCommandBufferBindVertexBuffers(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 ranges = c.args.NextPointer();
    c.rec.BindVertexBuffers(cmd, first, count, ranges);
}

/// SDK (3.2.0 sdk+0x5fe960): the hardware takes the size in 256-byte units, rounded up; size 0
/// unbinds.
void nvnCommandBufferBindUniformBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 stage = c.args.NextU32();
    const s32 index = c.args.NextS32();
    const u64 address = c.args.NextU64();
    const u64 size = c.args.NextU64();
    c.rec.BindUniformBuffer(cmd, stage, index, address, (size + 0xFF) & ~u64{0xFF});
}

void nvnCommandBufferBindStorageBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 stage = c.args.NextU32();
    const s32 index = c.args.NextS32();
    const u64 address = c.args.NextU64();
    const u64 size = c.args.NextU64();
    c.rec.BindStorageBuffer(cmd, stage, index, address, size);
}

/// BindUniformBuffers / BindUniformBuffersDeferred (the NVN2 Deferred variants have no prototype in
/// the tables; they take the same arguments). Deferred: the ranges are read again at submit.
template <bool Deferred>
void BindUniformBuffersImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 stage = c.args.NextU32();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 ranges = c.args.NextPointer();
    c.rec.BindUniformBuffers(cmd, stage, first, count, ranges, Deferred);
}

template <bool Deferred>
void BindStorageBuffersImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 stage = c.args.NextU32();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 ranges = c.args.NextPointer();
    c.rec.BindStorageBuffers(cmd, stage, first, count, ranges, Deferred);
}

// ---- Texture, image and sampler handles ---------------------------------------------------------------

/// BindTexture / BindImage / BindSeparateTexture / BindSeparateSampler: one 64-bit handle (the
/// by-value NVNseparate*Handle structs hold one u64 and travel in one x register).
template <Op Kind>
void BindHandleImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 stage = c.args.NextU32();
    const s32 index = c.args.NextS32();
    const u64 handle = c.args.NextU64();
    c.rec.BindHandle(cmd, Kind, stage, index, handle);
}

/// BindTextures / BindImages / BindSeparateTextures / BindSeparateSamplers and their Deferred
/// variants (NVN2, no prototype in the tables: same arguments; the GPU reads the handles when the
/// list executes, so the Recorder re-reads them at submit).
template <Op Kind, bool Deferred>
void BindHandlesImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 stage = c.args.NextU32();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 handles = c.args.NextPointer();
    c.rec.BindHandles(cmd, Kind, stage, first, count, handles, Deferred);
}

void nvnCommandBufferSetTexturePool(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    c.rec.SetTexturePool(cmd, pool);
}

void nvnCommandBufferSetSamplerPool(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    c.rec.SetSamplerPool(cmd, pool);
}

// ---- Stencil and rasterizer switches ----------------------------------------------------------------------

/// SetStencilValueMask (which 0), SetStencilMask (1, the write mask), SetStencilRef (2).
template <u32 Which>
void SetStencilImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 face = c.args.NextS32();
    const s32 value = c.args.NextS32();
    c.rec.SetStencil(cmd, face, Which, value);
}

void nvnCommandBufferSetPolygonOffsetClamp(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const float factor = c.args.NextF32();
    const float units = c.args.NextF32();
    const float clamp = c.args.NextF32();
    c.rec.SetPolygonOffsetClamp(cmd, factor, units, clamp);
}

void nvnCommandBufferSetBlendColor(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 color = c.args.NextPointer();
    c.rec.SetBlendColor(cmd, color);
}

void nvnCommandBufferSetDepthClamp(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetDepthClamp(cmd, c.args.NextBool());
}

void nvnCommandBufferSetAlphaRef(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetAlphaRef(cmd, c.args.NextF32());
}

void nvnCommandBufferSetPrimitiveRestart(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const bool enable = c.args.NextBool();
    const s32 index = c.args.NextS32();
    c.rec.SetPrimitiveRestart(cmd, enable, static_cast<u32>(index));
}

void nvnCommandBufferSetRasterizerDiscard(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetRasterizerDiscard(cmd, c.args.NextBool());
}

void nvnCommandBufferSetSampleMask(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetSampleMask(cmd, c.args.NextU32());
}

/// (cmd, NVNprovokingVertexMode): one SET_PROVOKING_VERTEX method holding the mode (OoT
/// sdk+0x686300); the SDK's initial GPU state selects the last vertex.
void nvnCommandBufferSetProvokingVertexMode(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetProvokingVertexMode(cmd, c.args.NextU32());
}

void nvnCommandBufferSetPointSize(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetPointSize(cmd, c.args.NextF32());
}

void nvnCommandBufferSetLineWidth(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetLineWidth(cmd, c.args.NextF32());
}

void nvnCommandBufferSetRenderEnable(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetRenderEnable(cmd, c.args.NextBool());
}

/// The driver (3.2.0 sdk+0x61c160, 1.0.0 sdk+0x5faa00) programs only RENDER_IF_EQUAL (0) and
/// RENDER_IF_NOT_EQUAL (1); any other mode returns without touching the render enable state.
void nvnCommandBufferSetRenderEnableConditional(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u64 address = c.args.NextU64();
    if (mode > 1) {
        return;
    }
    c.rec.SetRenderEnableConditional(cmd, mode, address);
}

// ---- Accepted and ignored, as nxnative ---------------------------------------------------------------------

/// nxnative lists SetDepthBounds among its no-ops: the host records no depth bounds test. The
/// driver (3.2.0 sdk+0x6071d0) enables the test with [min, max] or disables it. A test over [0, 1]
/// or wider passes every stored depth, so only a narrower enabled range would change pixels.
void nvnCommandBufferSetDepthBounds(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const bool enable = c.args.NextBool();
    const float min_depth = c.args.NextF32();
    const float max_depth = c.args.NextF32();
    if (enable && (min_depth > 0.0f || max_depth < 1.0f)) {
        NVN_LOG_ONCE(LOG_WARNING,
                     "[nvn] nvnCommandBufferSetDepthBounds(true, {}, {}) from {:#x}: ignored by "
                     "the host",
                     min_depth, max_depth, c.ctx.lr);
    }
}

// The switches below are ignored by the host. Each logs its value with the game's call site
// whenever it changes (at most NVN_LOG_CHANGES_LIMIT times per boot), so a run shows whether DK
// ever leaves the value the host behaves as. The first call of a boot always logs (nvn_logged is
// per boot, nvn_last is not). Steady-state cost: two relaxed loads and a compare.
constexpr u32 NVN_LOG_CHANGES_LIMIT = 8;
#define NVN_LOG_CHANGES(value, ...)                                                                \
    do {                                                                                           \
        static std::atomic<s64> nvn_last{std::numeric_limits<s64>::min()};                         \
        static Common::BootCounter nvn_logged;                                                     \
        const s64 nvn_value = static_cast<s64>(value);                                             \
        if (nvn_logged.Count() == 0 || nvn_last.load(std::memory_order_relaxed) != nvn_value) {    \
            nvn_last.store(nvn_value, std::memory_order_relaxed);                                  \
            if (nvn_logged.Next() < NVN_LOG_CHANGES_LIMIT) {                                       \
                LOG_INFO(Core, __VA_ARGS__);                                                       \
            }                                                                                      \
        }                                                                                          \
    } while (false)

/// The driver (3.2.0 sdk+0x608930) sets method 0x452 (conservative raster enable). The host
/// rasterizes normally; only an enable would change pixels.
void nvnCommandBufferSetConservativeRasterEnable(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const bool enable = c.args.NextBool();
    NVN_LOG_CHANGES(enable,
                    "[nvn] nvnCommandBufferSetConservativeRasterEnable({}) from {:#x}: ignored by "
                    "the host",
                    enable, c.ctx.lr);
}

// Function ids without a prototype in the tables (nxnative NVN_NOPROTO): (cmd, int mode).
/// The driver (3.2.0 sdk+0x605300) writes the mode to method 0xe26; the host has one geometry
/// pipeline. DK's first call passes 0.
void nvnCommandBufferSetGeometryProcessingMode(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 mode = c.args.NextS32();
    NVN_LOG_CHANGES(mode,
                    "[nvn] nvnCommandBufferSetGeometryProcessingMode({}) from {:#x}: no effect on "
                    "the host",
                    mode, c.ctx.lr);
}

/// The driver (3.2.0 sdk+0x60b440) stores the mode in a side-stream entry (tag 0x27). Overlap only
/// permits dispatches to run concurrently: the host's submission order is a valid schedule.
void nvnCommandBufferSetComputeOverlapMode(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 mode = c.args.NextS32();
    NVN_LOG_CHANGES(mode,
                    "[nvn] nvnCommandBufferSetComputeOverlapMode({}) from {:#x}: the host runs "
                    "compute in submission order",
                    mode, c.ctx.lr);
}

// Variable rate shading. The host shades every pixel (the full rate), the most detailed result
// any shading rate image or palette can ask for, so ignoring both only costs host GPU time.
/// (cmd, const NVNtexture* image, const NVNtextureView* view); a NULL image unbinds (the driver,
/// 3.2.0 sdk+0x69b9c0, writes method 0x4aa with zeroes then).
void nvnCommandBufferBindShadingRateImage(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const u64 image = c.args.NextPointer();
    NVN_LOG_CHANGES(image != 0,
                    "[nvn] nvnCommandBufferBindShadingRateImage({:#x}) from {:#x}: the host shades "
                    "at the full rate",
                    image, c.ctx.lr);
}

/// (cmd, int viewport, int count, const NVNshadingRate* rates); the driver (3.2.0 sdk+0x69b420)
/// packs 16 4-bit rates, entries past `count` (or all of them for NULL rates) at rate 5.
void nvnCommandBufferShadingRateImagePalette(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 viewport = c.args.NextS32();
    const s32 count = c.args.NextS32();
    NVN_LOG_ONCE(LOG_INFO,
                 "[nvn] nvnCommandBufferShadingRateImagePalette(viewport {}, {} rates) from "
                 "{:#x}: the host shades at the full rate",
                 viewport, count, c.ctx.lr);
}

#undef NVN_LOG_CHANGES

#undef NVN_LOG_ONCE

} // namespace

void RegisterCommandBufferBindFunctions(Registry& registry) {
    NVN_HLE(registry, nvnCommandBufferSetRenderTargets);
    registry.Add<&ClearColorImpl<false>>("nvnCommandBufferClearColor");
    registry.Add<&ClearColorImpl<true>>("nvnCommandBufferClearColori");
    registry.Add<&ClearColorImpl<true>>("nvnCommandBufferClearColorui");
    NVN_HLE(registry, nvnCommandBufferClearDepthStencil);

    NVN_HLE(registry, nvnCommandBufferSetViewport);
    NVN_HLE(registry, nvnCommandBufferSetViewports);
    NVN_HLE(registry, nvnCommandBufferSetScissor);
    NVN_HLE(registry, nvnCommandBufferSetScissors);
    NVN_HLE(registry, nvnCommandBufferSetDepthRange);
    NVN_HLE(registry, nvnCommandBufferSetDepthRanges);
    NVN_HLE(registry, nvnCommandBufferSetViewportSwizzles);

    NVN_HLE(registry, nvnCommandBufferBindProgram);
    registry.Add<&BindStateImpl<StateKind::Blend>>("nvnCommandBufferBindBlendState");
    registry.Add<&BindStateImpl<StateKind::Color>>("nvnCommandBufferBindColorState");
    registry.Add<&BindStateImpl<StateKind::ChannelMask>>("nvnCommandBufferBindChannelMaskState");
    registry.Add<&BindStateImpl<StateKind::Multisample>>("nvnCommandBufferBindMultisampleState");
    registry.Add<&BindStateImpl<StateKind::Polygon>>("nvnCommandBufferBindPolygonState");
    registry.Add<&BindStateImpl<StateKind::DepthStencil>>("nvnCommandBufferBindDepthStencilState");
    NVN_HLE(registry, nvnCommandBufferBindVertexAttribState);
    NVN_HLE(registry, nvnCommandBufferBindVertexStreamState);

    NVN_HLE(registry, nvnCommandBufferBindVertexBuffer);
    NVN_HLE(registry, nvnCommandBufferBindVertexBuffers);
    NVN_HLE(registry, nvnCommandBufferBindUniformBuffer);
    NVN_HLE(registry, nvnCommandBufferBindStorageBuffer);
    registry.Add<&BindUniformBuffersImpl<false>>("nvnCommandBufferBindUniformBuffers");
    registry.Add<&BindUniformBuffersImpl<true>>("nvnCommandBufferBindUniformBuffersDeferred");
    registry.Add<&BindStorageBuffersImpl<false>>("nvnCommandBufferBindStorageBuffers");
    registry.Add<&BindStorageBuffersImpl<true>>("nvnCommandBufferBindStorageBuffersDeferred");

    registry.Add<&BindHandleImpl<Op::BindTextures>>("nvnCommandBufferBindTexture");
    registry.Add<&BindHandleImpl<Op::BindImages>>("nvnCommandBufferBindImage");
    registry.Add<&BindHandleImpl<Op::BindSeparateTextures>>("nvnCommandBufferBindSeparateTexture");
    registry.Add<&BindHandleImpl<Op::BindSeparateSamplers>>("nvnCommandBufferBindSeparateSampler");
    registry.Add<&BindHandlesImpl<Op::BindTextures, false>>("nvnCommandBufferBindTextures");
    registry.Add<&BindHandlesImpl<Op::BindTextures, true>>("nvnCommandBufferBindTexturesDeferred");
    registry.Add<&BindHandlesImpl<Op::BindImages, false>>("nvnCommandBufferBindImages");
    registry.Add<&BindHandlesImpl<Op::BindImages, true>>("nvnCommandBufferBindImagesDeferred");
    registry.Add<&BindHandlesImpl<Op::BindSeparateTextures, false>>(
        "nvnCommandBufferBindSeparateTextures");
    registry.Add<&BindHandlesImpl<Op::BindSeparateTextures, true>>(
        "nvnCommandBufferBindSeparateTexturesDeferred");
    registry.Add<&BindHandlesImpl<Op::BindSeparateSamplers, false>>(
        "nvnCommandBufferBindSeparateSamplers");
    registry.Add<&BindHandlesImpl<Op::BindSeparateSamplers, true>>(
        "nvnCommandBufferBindSeparateSamplersDeferred");
    NVN_HLE(registry, nvnCommandBufferSetTexturePool);
    NVN_HLE(registry, nvnCommandBufferSetSamplerPool);

    registry.Add<&SetStencilImpl<0>>("nvnCommandBufferSetStencilValueMask");
    registry.Add<&SetStencilImpl<1>>("nvnCommandBufferSetStencilMask");
    registry.Add<&SetStencilImpl<2>>("nvnCommandBufferSetStencilRef");
    NVN_HLE(registry, nvnCommandBufferSetPolygonOffsetClamp);
    NVN_HLE(registry, nvnCommandBufferSetBlendColor);
    NVN_HLE(registry, nvnCommandBufferSetDepthClamp);
    NVN_HLE(registry, nvnCommandBufferSetAlphaRef);
    NVN_HLE(registry, nvnCommandBufferSetPrimitiveRestart);
    NVN_HLE(registry, nvnCommandBufferSetRasterizerDiscard);
    NVN_HLE(registry, nvnCommandBufferSetSampleMask);
    NVN_HLE(registry, nvnCommandBufferSetProvokingVertexMode);
    NVN_HLE(registry, nvnCommandBufferSetPointSize);
    NVN_HLE(registry, nvnCommandBufferSetLineWidth);
    NVN_HLE(registry, nvnCommandBufferSetRenderEnable);
    NVN_HLE(registry, nvnCommandBufferSetRenderEnableConditional);

    NVN_HLE(registry, nvnCommandBufferSetDepthBounds);
    NVN_HLE(registry, nvnCommandBufferSetConservativeRasterEnable);
    NVN_HLE(registry, nvnCommandBufferSetGeometryProcessingMode);
    NVN_HLE(registry, nvnCommandBufferSetComputeOverlapMode);
    NVN_HLE(registry, nvnCommandBufferBindShadingRateImage);
    NVN_HLE(registry, nvnCommandBufferShadingRateImagePalette);
}

} // namespace Core::NvnHle::Hle
