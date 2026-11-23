// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <atomic>
#include <chrono>
#include <optional>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"
#include "video_core/nvn/timestamp_mode.h"

// Device family: NVNdevice, its builder, handles and the device-info table. Behaviour ported from
// nxnative hle_nvn.c ("Device" section), with the NVN2 constants it measured on Donkey Kong
// Bananza (sdk 20.5.19).
namespace Core::NvnHle::Hle {
namespace {

/// NVNdeviceBuilder, SDK layout (3.2.0 sdk+0x5d4ef0 / 1.0.0 sdk+0x5b0c30): SetDefaults clears
/// 0x18 bytes, SetFlags stores the flags at +4 (sdk+0x5d4f00 / +0x5b0c40).
constexpr u64 BuilderFlagsOffset = 4;
constexpr u64 BuilderBytes = 0x18;

/// NVN_DEVICE_INFO item 0x9A reads "the device reserved its GPU address range" (device+0x9f0, an
/// NvRm address-space allocation made at device creation unless the builder flags are negative,
/// 3.2.0 sdk+0x5e2a40).
constexpr u32 ItemAddressRangeReserved = 0x9A;
constexpr u32 DeviceFlagNoAddressReservation = 0x8000'0000;

/// nvnDeviceGetInteger as the Donkey Kong Bananza 3.2.0 SDK answers it on Switch 2 (sdk+0x5d5240 ->
/// sdk+0x5d9960). Index = NVNdeviceInfo item; items above 0xB0 answer 0.
///
/// The SDK starts from its device-independent table (sdk+0x5c0a20, items 2..0xB0; items it does not
/// list answer 0) and overrides: item 0 -> 231, 1 -> 30; 0x2A/0x6E/0x7E/0x8F/0x93/0xA9 -> 1;
/// 0x2E/0x2F/0x9C/0x9E -> 201; 0x30/0x9F -> 17; 0x9D -> 14; 0x31/0x5A -> 0. The remaining
/// overrides read device fields that NvRmGpuDeviceGetInfo filled from the chip descriptor of chip
/// 0x17E "GA10F" (sdk+0xdcc0f0: 48 warps per SM, warp size 32; its topology sdk+0xb988ec: 1 GPC,
/// 6 TPCs, 2 SMs per TPC) and from the GPU characteristics (SM version 0x808, the one
/// nvhost_ctrl_gpu reports): 0x17 SMs = 12, 0x18 SMs * warps = 576, 0x37 SMs per TPC * warps = 96,
/// 0x62 warp size = 32, 0x7D = 8 (SM 8.8), 0x87 = 1, 0x99 = 2, 0x2B L2 = 1 MiB, 0x3A page size
/// 64 KiB, 0x2D/0x38/0x3D..0x41 constants (sdk+0xae58b0/+0xae6300). Every one of these equals the
/// SDK's own device-independent table entry for the item, which is what a real Switch 2 reports.
/// (Our emulated nvgpu still reports the Tegra X1 L2 size 256 KiB and TPC mask 3; the SDK table,
/// not those, is the hardware.)
///
/// The 1.0.0 SDK (sdk+0x5b0f90 -> +0x5b57c0, table sdk+0x59bf50) answers differently for items
/// 1 (16), 0x30/0x9D/0x9F (11), 0x3E (134217728), 0x7F/0x82 (65535), 0x98 (131072) and
/// 0xAB/0xAC/0xAE (0): OlderSdkDeviceInfo, chosen by the size of the game sdk's GetProcAddress
/// table (nvnDeviceGetInteger).
constexpr std::array<s32, 0xB1> DeviceInfo{
    /* 0x00 */ 231, 30, 14, 65536, 256, 8, 32, 4,
    /* 0x08 */ 16, 32, 16, 4, 4, 4, 32, 32,
    /* 0x10 */ 32, 256, 256, 4, 8, 512, 256, 12,
    /* 0x18 */ 576, 4096, 16384, 16, 1024, 1024, 64, 1024,
    /* 0x20 */ 2147483647, 65535, 65535, 8, 1048576, 4096, 16, 4096,
    /* 0x28 */ 1, 1, 1, 1048576, 16, 2048, 201, 201,
    /* 0x30 */ 17, 0, 1, 8, 8, 4, 32, 96,
    /* 0x38 */ 32, 128, 65536, 1, 4, 32768, 268435456, 16384,
    /* 0x40 */ 32768, 32768, 1, 1, 1, 1, 5, 1,
    /* 0x48 */ 1, 1, 1, 1, 1, 16, 2048, 32,
    /* 0x50 */ 4096, 65536, 65536, 256, 16384, 262144, 4096, 1,
    /* 0x58 */ 4, 2, 0, 4096, 16384, 4096, 128, 32,
    /* 0x60 */ 65535, 1, 32, 256, 16, 65535, 65535, 256,
    /* 0x68 */ 1, 16, 16, 16, 1, 1, 1, 256,
    /* 0x70 */ 0, 16, 8, 16, 32, 31, 536870912, 16777216,
    /* 0x78 */ 16777216, 1073741824, 32, 1, 64, 8, 1, 2147483647,
    /* 0x80 */ 32, 32, 4194303, 512, 256, 32, 16383, 1,
    /* 0x88 */ 8, 0, 0, 4, 256, 16, 0, 1,
    /* 0x90 */ 31, 0, 64, 1, 58368, 24576, 0, 256,
    /* 0x98 */ 262144, 2, 1, 0, 201, 14, 201, 17,
    /* 0xa0 */ 163840, 8, 4, 256, 256, 4, 65536, 201,
    /* 0xa8 */ 5, 1, 0, 63, 1, 0, 1, 0,
    /* 0xb0 */ 4096,
};

void nvnDeviceBuilderSetDefaults(Call& c) {
    c.Zero(c.args.NextPointer(), BuilderBytes);
}

void nvnDeviceBuilderSetFlags(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 flags = c.args.NextS32();
    c.Store<u32>(builder + BuilderFlagsOffset, static_cast<u32>(flags));
}

void nvnDeviceBuilderGetFlags(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.ret.SetU32(c.Load<u32>(builder + BuilderFlagsOffset));
}

void nvnDeviceInitialize(Call& c) {
    const u64 device = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    if (!c.rec.DeviceInitialize(device)) {
        c.ret.SetBool(false);
        return;
    }
    c.runtime.device.flags = c.Load<u32>(builder + BuilderFlagsOffset);
    // NVN's defaults (a game that wants Vulkan/D3D conventions switches them right after).
    c.runtime.device.window_origin_mode = 0;
    c.runtime.device.depth_mode = 0;
    LOG_INFO(Core, "[nvn] DeviceInitialize {:#x} flags={:#x}", device,
             c.runtime.device.flags.load());
    c.ret.SetBool(true);
}

void nvnDeviceFinalize(Call& c) {
    c.rec.DeviceFinalize(c.args.NextPointer());
}

void nvnDeviceSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

/// The answers of the 1.0.0 SDK where they differ from 3.2.0's (see DeviceInfo).
s32 OlderSdkDeviceInfo(u32 item, s32 value) {
    switch (item) {
    case 0x01:
        return 16;
    case 0x30:
    case 0x9D:
    case 0x9F:
        return 11;
    case 0x3E:
        return 134217728;
    case 0x7F:
    case 0x82:
        return 65535;
    case 0x98:
        return 131072;
    case 0xAB:
    case 0xAC:
    case 0xAE:
        return 0;
    default:
        return value;
    }
}

void nvnDeviceGetInteger(Call& c) {
    static_cast<void>(c.args.NextPointer()); // device
    const u32 item = c.args.NextU32();
    const u64 out = c.args.NextPointer();
    // The loaded game's SDK decides what its driver reports (API version, GPU code, limits): its
    // constants win. Items it reads from the device object at run time, and an SDK whose switch
    // was not recognised, fall back to DeviceInfo (the Switch 2 table of DK's 3.2.0 SDK) with the
    // 1.0.0-generation differences applied by OlderSdkDeviceInfo.
    if (const auto sdk = c.runtime.device.SdkInfo()) {
        if (const std::optional<s32> answer = sdk->Get(item)) {
            c.Store<s32>(out, *answer);
            return;
        }
    }
    s32 value = item < DeviceInfo.size() ? DeviceInfo[item] : 0;
    if (item == ItemAddressRangeReserved &&
        (c.runtime.device.flags.load(std::memory_order_relaxed) & DeviceFlagNoAddressReservation) != 0) {
        value = 0;
    }
    // The SDK generation comes from the game's own sdk (DeviceState::OlderSdk).
    const bool older_sdk = c.runtime.device.OlderSdk();
    if (older_sdk) {
        value = OlderSdkDeviceInfo(item, value);
    }
    // The set of items a game asks for is part of the NVN coverage census: log each once.
    static std::array<Common::BootCounter, 256> seen;
    if (item < seen.size() && seen[item].First()) {
        LOG_INFO(Core, "[nvn] GetInteger item {:#x} -> {} ({} sdk answers)", item, value,
                 older_sdk ? "1.0.0-generation" : "3.2.0-generation");
    }
    c.Store<s32>(out, value);
}

/// Image, texel-fetch and combined handles: the SDK widens the int ids with sign extension
/// (3.2.0 sdk+0x5d5230 / +0x5d52b0 / +0x5d52c0: `(long)id`, `(long)texture | (long)sampler << 20`).
u64 SignExtend(s32 id) {
    return static_cast<u64>(static_cast<s64>(id));
}

void nvnDeviceGetImageHandle(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU64(SignExtend(c.args.NextS32()));
}

/// Separate texture / sampler handles are descriptor indices (the sampler index goes in bit 20 and
/// above, consistent with the combined handle layout). NVNseparateTextureHandle is a struct of one
/// u64, returned in x0.
void nvnDeviceGetSeparateTextureHandle(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU64(static_cast<u64>(c.args.NextU32()));
}

void nvnDeviceGetSeparateSamplerHandle(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU64(static_cast<u64>(c.args.NextU32()) << 20);
}

/// Combined handle = texture id | sampler id << 20, both sign-extended as the SDK does. The SDKs
/// whose function starts with a texture-id test (Ocarina of Time's, sdk+0x651c60) answer 0 for
/// texture id 0; DeviceState::texture_zero_null_handle says which one the game has.
void nvnDeviceGetTextureHandle(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 texture_id = c.args.NextS32();
    const s32 sampler_id = c.args.NextS32();
    if (texture_id == 0 &&
        c.runtime.device.texture_zero_null_handle.load(std::memory_order_relaxed)) {
        c.ret.SetU64(0);
        return;
    }
    c.ret.SetU64(SignExtend(texture_id) | SignExtend(sampler_id) << 20);
}

/// A texel fetch reads the texture descriptor and no sampler: the handle is the texture id.
void nvnDeviceGetTexelFetchHandle(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU64(SignExtend(c.args.NextS32()));
}

void nvnDeviceInstallDebugCallback(Call& c) {
    static_cast<void>(c);
}

void nvnDeviceGenerateDebugDomainId(Call& c) {
    c.ret.SetU32(c.runtime.next_debug_domain_id.fetch_add(1));
}

void nvnDeviceSetWindowOriginMode(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 mode = c.args.NextS32();
    c.runtime.device.window_origin_mode = mode;
    c.rec.SetDeviceModes(static_cast<u32>(mode), static_cast<u32>(c.runtime.device.depth_mode.load()));
    LOG_INFO(Core, "[nvn] window origin {}", mode != 0 ? "upper left" : "lower left");
}

void nvnDeviceSetDepthMode(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 mode = c.args.NextS32();
    c.runtime.device.depth_mode = mode;
    c.rec.SetDeviceModes(static_cast<u32>(c.runtime.device.window_origin_mode.load()), static_cast<u32>(mode));
    LOG_INFO(Core, "[nvn] depth mode {}", mode != 0 ? "near=0" : "near=-w");
}

void nvnDeviceGetWindowOriginMode(Call& c) {
    c.ret.SetU32(static_cast<u32>(c.runtime.device.window_origin_mode.load()));
}

void nvnDeviceGetDepthMode(Call& c) {
    c.ret.SetU32(static_cast<u32>(c.runtime.device.depth_mode.load()));
}

void nvnDeviceGetFlags(Call& c) {
    c.ret.SetU32(c.runtime.device.flags);
}

/// Fast clear values are hints for the Tegra hardware only.
void nvnDeviceRegisterFastClear(Call& c) {
    c.ret.SetBool(true);
}

void nvnDeviceSetIntermediateShaderCache(Call& c) {
    static_cast<void>(c);
}

void nvnDeviceFinalizeCommandHandle(Call& c) {
    static_cast<void>(c);
}

void nvnDeviceApplyDeferredFinalizes(Call& c) {
    static_cast<void>(c);
}

void nvnDeviceIsExternalDebuggerAttached(Call& c) {
    c.ret.SetBool(false);
}

void nvnSetExternalDebuggerEnabledNVX(Call& c) {
    static_cast<void>(c);
}

/// EDEN_NX2_NVN_TSTRACE: every call (the game reads about five per frame).
bool TraceCall(Common::BootCounter& counter) {
    if (!VideoCommon::Nvn::TimestampTraceEnabled()) {
        return false;
    }
    static_cast<void>(counter.Next());
    return true;
}

/// Reads one report's time. The field holds the time in the unit the title's GPU writes
/// (timestamp_mode.h): Tegra X1 ticks, which its library converts, or Switch 2 nanoseconds, which
/// its library returns unchanged (3.2.0 sdk+0x5d52d0: field * mul / div with the GA10F chip's
/// 1 / 1).
void nvnDeviceGetTimestampInNanoseconds(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const u64 counter_data = c.args.NextPointer();
    // NVNcounterData { u64 counter; u64 timestamp; }
    const u64 field = c.Load<u64>(counter_data + 8);
    // Diagnostic (EDEN_NX2_NVN_DKFRAME=<low 12 bits of the return address in hex>): the state of
    // the game's frame-rate controller when it reads the end-of-frame timestamp, taken from the
    // callee-saved `this` register of the calling function. Measurement only; no behaviour.
    static const u64 dk_lr_low = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_DKFRAME");
        return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 16)) : u64{0};
    }();
    if (dk_lr_low != 0 && (c.ctx.lr & 0xfff) == dk_lr_low) {
        const u64 fw = c.ctx.r[19];
        const u64 ctl = c.Load<u64>(fw + 0x218);
        static std::atomic<u64> last_acquire_ns{};
        static std::atomic<u64> last_sync_ns{};
        const u64 acquire_ns = c.runtime.acquire_wait_ns.load(std::memory_order_relaxed);
        const u64 sync_ns = c.runtime.sync_wait_ns.load(std::memory_order_relaxed);
        const u64 acquire_wait_us = (acquire_ns - last_acquire_ns.exchange(acquire_ns)) / 1000;
        const u64 sync_wait_us = (sync_ns - last_sync_ns.exchange(sync_ns)) / 1000;
        LOG_INFO(Core,
                 "[nvn] dkframe acquire_wait_us={} sync_wait_us={} fw={:#x} present={} guest_ns={} report_ns={} frame_start_ticks={} "
                 "cost_ticks={} fps_e0={} interval_88={} flags_1ca={}/{} ctl: target_50={} "
                 "ceil_54={} cool_4c={} cnt_48={} cfg: margin_14={} hyst_18={} cool_1c={} min_20={} "
                 "max_24={} step_28={} legacy: min_0={} max_4={} margin_8={} hyst_c={} cool_10={} "
                 "cur_40={} ceil_44={} cnt_48={} cooldown_4c={}",
                 acquire_wait_us, sync_wait_us, fw,
                 Core::g_nx2_present_count.load(std::memory_order_relaxed),
                 VideoCommon::Nvn::GuestClockNanoseconds(),
                 VideoCommon::Nvn::ReportFieldToNanoseconds(Settings::CurrentProgramIsSwitch2(), field),
                 c.Load<u64>(fw + 0xf0), c.Load<u64>(fw + 0x128), c.Load<s32>(fw + 0xe0),
                 c.Load<s32>(fw + 0x88), c.Load<u8>(fw + 0x1ca), c.Load<u8>(fw + 0x1cb),
                 ctl != 0 ? c.Load<s32>(ctl + 0x50) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x54) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x4c) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x48) : -1,
                 ctl != 0 ? c.Load<f32>(ctl + 0x14) : -1.0f, ctl != 0 ? c.Load<s32>(ctl + 0x18) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x1c) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x20) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x24) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x28) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x00) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x04) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x08) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x0c) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x10) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x40) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x44) : -1, ctl != 0 ? c.Load<s32>(ctl + 0x48) : -1,
                 ctl != 0 ? c.Load<s32>(ctl + 0x4c) : -1);
    }
    const u64 result =
        VideoCommon::Nvn::ReportFieldToNanoseconds(Settings::CurrentProgramIsSwitch2(), field);
    static Common::BootCounter calls;
    static std::atomic<u64> last{};
    if (TraceCall(calls)) {
        const u64 previous = last.exchange(result, std::memory_order_relaxed);
        LOG_INFO(Core,
                 "[nvn] tstrace GetTimestamp present={} lr={:#x} report={:#x} counter={} field={} "
                 "ns={} (delta to the previous call {}, report is {} us behind the guest clock)",
                 Core::g_nx2_present_count.load(std::memory_order_relaxed), c.ctx.lr, counter_data,
                 c.Load<u64>(counter_data), field, result, static_cast<s64>(result - previous),
                 (static_cast<s64>(VideoCommon::Nvn::GuestClockNanoseconds()) -
                  static_cast<s64>(result)) /
                     1000);
    }
    c.ret.SetU64(result);
}

/// The guest's global clock (`CoreTiming::GetGlobalTimeNs`, the clock every guest-visible time
/// source runs on; nxnative uses the guest clock here too): the epoch of the report times.
void nvnDeviceGetCurrentTimestampInNanoseconds(Call& c) {
    const u64 result = VideoCommon::Nvn::GuestClockNanoseconds();
    static Common::BootCounter calls;
    static std::atomic<u64> last{};
    if (TraceCall(calls)) {
        const u64 previous = last.exchange(result, std::memory_order_relaxed);
        LOG_INFO(Core,
                 "[nvn] tstrace GetCurrentTimestamp present={} lr={:#x} ns={} (delta to the previous "
                 "call {})",
                 Core::g_nx2_present_count.load(std::memory_order_relaxed), c.ctx.lr, result,
                 static_cast<s64>(result - previous));
    }
    c.ret.SetU64(result);
}

} // namespace

void RegisterDeviceFunctions(Registry& registry) {
    NVN_HLE(registry, nvnDeviceBuilderSetDefaults);
    NVN_HLE(registry, nvnDeviceBuilderSetFlags);
    NVN_HLE(registry, nvnDeviceBuilderGetFlags);
    NVN_HLE(registry, nvnDeviceInitialize);
    NVN_HLE(registry, nvnDeviceFinalize);
    NVN_HLE(registry, nvnDeviceSetDebugLabel);
    NVN_HLE(registry, nvnDeviceGetInteger);
    NVN_HLE(registry, nvnDeviceGetCurrentTimestampInNanoseconds);
    NVN_HLE(registry, nvnDeviceSetIntermediateShaderCache);
    NVN_HLE(registry, nvnDeviceGetTextureHandle);
    NVN_HLE(registry, nvnDeviceGetTexelFetchHandle);
    NVN_HLE(registry, nvnDeviceGetImageHandle);
    NVN_HLE(registry, nvnDeviceInstallDebugCallback);
    NVN_HLE(registry, nvnDeviceGenerateDebugDomainId);
    NVN_HLE(registry, nvnDeviceSetWindowOriginMode);
    NVN_HLE(registry, nvnDeviceSetDepthMode);
    registry.Add<&nvnDeviceRegisterFastClear>("nvnDeviceRegisterFastClearColor");
    registry.Add<&nvnDeviceRegisterFastClear>("nvnDeviceRegisterFastClearColori");
    registry.Add<&nvnDeviceRegisterFastClear>("nvnDeviceRegisterFastClearColorui");
    registry.Add<&nvnDeviceRegisterFastClear>("nvnDeviceRegisterFastClearDepth");
    NVN_HLE(registry, nvnDeviceGetWindowOriginMode);
    NVN_HLE(registry, nvnDeviceGetDepthMode);
    NVN_HLE(registry, nvnDeviceGetTimestampInNanoseconds);
    NVN_HLE(registry, nvnDeviceApplyDeferredFinalizes);
    NVN_HLE(registry, nvnDeviceFinalizeCommandHandle);
    NVN_HLE(registry, nvnDeviceGetSeparateTextureHandle);
    NVN_HLE(registry, nvnDeviceGetSeparateSamplerHandle);
    NVN_HLE(registry, nvnDeviceIsExternalDebuggerAttached);
    NVN_HLE(registry, nvnDeviceGetFlags);
    NVN_HLE(registry, nvnSetExternalDebuggerEnabledNVX);
}

} // namespace Core::NvnHle::Hle
