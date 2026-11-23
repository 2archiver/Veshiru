// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <span>

#include "common/boot_counter.h"
#include "common/diag_request.h"
#include "common/frame_stats.h"
#include "common/game_presets.h"
#include "common/hdr.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"
#include "core/hle/nvn/present_pacing.h"
#include "core/perf_stats.h"

// Window family: NVNwindow, its builder (kept in place using the NVN2 SDK layout), acquire and
// present. The window's host textures are the NVN textures the game registered; a slot is free
// again once its present has completed on the GPU.
namespace Core::NvnHle::Hle {
namespace {

// NVN2 window builder in place. Offsets and defaults are from Donkey Kong Bananza's sdk NSO
// (nvnWindowBuilder* at 0x5cbcd0..0x5cbea0); the older nxnative win_builder_t has a different ABI.
constexpr u64 WbDevice = 0x08;
constexpr u64 WbTextureCount = 0x10;
constexpr u64 WbActiveTextureCount = 0x14;
constexpr u64 WbNative = 0x18;
constexpr u64 WbInterval = 0x20; // stored as interval - 1
constexpr u64 WbColorSpace = 0x24;
constexpr u64 WbTextures = 0x28; // NVNtexture* const* (not an inline array)
constexpr u64 WbHdrDisplayMetadata = 0x30;
constexpr u64 WbHdrContentMetadata = 0x58;
constexpr u64 WbSize = 0x60;
constexpr u64 WindowHdrDisplayMetadata = 0x38;
constexpr u64 WindowHdrContentMetadata = 0x60;
constexpr s32 MaxWindowTextures = 4;

// [INFERENCE] NVNcolorSpace is not in the public NVN1 headers. These values match the SDK's
// HdrColorEncoding accepted by MakeBufferPreference: 0 = SDR, 1 = HDR10/PQ, 0x40 = scRGB.
Common::Hdr::Encoding DecodeColorSpace(u32 color_space) {
    using Common::Hdr::Encoding;
    switch (color_space) {
    case 0:
        return Encoding::Srgb;
    case 1:
        return Encoding::Pq2020;
    case 0x40:
        return Encoding::ScRgbLinear;
    default:
        LOG_WARNING(Core, "[hdr] nvn window unknown colour space {}, using sRGB", color_space);
        return Encoding::Srgb;
    }
}

const char* EncodingName(Common::Hdr::Encoding encoding) {
    using Common::Hdr::Encoding;
    switch (encoding) {
    case Encoding::Srgb:
        return "sRGB";
    case Encoding::Pq2020:
        return "PQ2020";
    case Encoding::ScRgbLinear:
        return "scRGB-linear";
    }
    return "unknown";
}

/// HDR display metadata in guest memory, SDK order: red, green, blue, white point (x, y each),
/// then minimum and maximum luminance. nvnWindowBuilderSetDefaults (3.2.0 sdk+0x5efbe0) stores
/// 0.03 then 1000 there, and the setter (sdk+0x5efc60) stores its 9th and 10th float in that
/// order.
Common::Hdr::ContentMetadata LoadMetadata(Call& c, u64 display) {
    Common::Hdr::ContentMetadata metadata;
    const std::array fields = {
        &metadata.display_primary_red[0],   &metadata.display_primary_red[1],
        &metadata.display_primary_green[0], &metadata.display_primary_green[1],
        &metadata.display_primary_blue[0],  &metadata.display_primary_blue[1],
        &metadata.white_point[0],           &metadata.white_point[1],
        &metadata.min_luminance,            &metadata.max_luminance,
    };
    for (size_t i = 0; i < fields.size(); ++i) {
        *fields[i] = c.Load<f32>(display + 4 * i);
    }
    metadata.max_content_light_level = c.Load<f32>(display + 0x28);
    metadata.max_frame_average_light_level = c.Load<f32>(display + 0x2c);
    return metadata;
}

void StoreDisplayMetadata(Call& c, u64 display, const Common::Hdr::ContentMetadata& metadata) {
    const std::array values = {
        metadata.display_primary_red[0],   metadata.display_primary_red[1],
        metadata.display_primary_green[0], metadata.display_primary_green[1],
        metadata.display_primary_blue[0],  metadata.display_primary_blue[1],
        metadata.white_point[0],           metadata.white_point[1],
        metadata.min_luminance,            metadata.max_luminance,
    };
    for (size_t i = 0; i < values.size(); ++i) {
        c.Store<f32>(display + 4 * i, values[i]);
    }
}

void StoreContentMetadata(Call& c, u64 content, const Common::Hdr::ContentMetadata& metadata) {
    c.Store<f32>(content, metadata.max_content_light_level);
    c.Store<f32>(content + 4, metadata.max_frame_average_light_level);
}

void PublishWindowRequest(Call& c, u64 window, u32 color_space, u64 display) {
    Common::Hdr::GuestRequest request;
    request.encoding = DecodeColorSpace(color_space);
    request.has_metadata = true;
    request.metadata = LoadMetadata(c, display);
    const auto previous = Common::Hdr::GetGuestRequest();
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        Common::Hdr::PublishGuestRequest(request);
        c.runtime.guest_request_producer = window;
    }
    if (!previous.has_metadata || previous.encoding != request.encoding) {
        LOG_INFO(Core, "[hdr] nvn window colour space {} -> {}", color_space,
                 EncodingName(request.encoding));
    }
    if (!previous.has_metadata || previous.metadata != request.metadata) {
        LOG_INFO(Core,
                 "[hdr] nvn window HDR metadata peak {} nits min {} nits MaxCLL {} nits MaxFALL {} nits",
                 request.metadata.max_luminance, request.metadata.min_luminance,
                 request.metadata.max_content_light_level,
                 request.metadata.max_frame_average_light_level);
    }
}

void SetDefaultHdrMetadata(Call& c, u64 builder) {
    // nvnWindowBuilderSetDefaults' constants: Rec.709 primaries, D65, 1000/0.03 nit display,
    // and 200/60 nit content light levels.
    const Common::Hdr::ContentMetadata metadata{
        .display_primary_red = {0.64f, 0.33f},
        .display_primary_green = {0.30f, 0.60f},
        .display_primary_blue = {0.15f, 0.06f},
        .white_point = {0.3127f, 0.3290f},
        .max_luminance = 1000.0f,
        .min_luminance = 0.03f,
        .max_content_light_level = 200.0f,
        .max_frame_average_light_level = 60.0f,
    };
    StoreDisplayMetadata(c, builder + WbHdrDisplayMetadata, metadata);
    StoreContentMetadata(c, builder + WbHdrContentMetadata, metadata);
}

constexpr u32 AcquireSuccess = 0;
constexpr u32 AcquireError = 1;

void nvnWindowBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 device = c.Load<u64>(builder + WbDevice);
    c.Zero(builder + WbTextureCount, WbSize - WbTextureCount);
    c.Store<u64>(builder + WbDevice, device);
    SetDefaultHdrMetadata(c, builder);
}

void nvnWindowBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + WbDevice, c.args.NextPointer());
}

void nvnWindowBuilderSetNativeWindow(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + WbNative, c.args.NextPointer());
}

/// The SDK stores the count as given (3.2.0 sdk+0x5efc20); WindowInitialize clamps it.
void nvnWindowBuilderSetTextures(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 textures = c.args.NextPointer();
    c.Store<s32>(builder + WbTextureCount, count);
    c.Store<u64>(builder + WbTextures, textures);
}

void nvnWindowBuilderSetPresentInterval(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + WbInterval, c.args.NextS32() - 1);
}

void nvnWindowBuilderSetNumActiveTextures(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + WbActiveTextureCount, c.args.NextS32());
}

void nvnWindowBuilderGetDevice(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + WbDevice));
}

void nvnWindowBuilderGetNumTextures(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + WbTextureCount));
}

void nvnWindowBuilderGetTexture(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 index = c.args.NextS32();
    const u64 textures = c.Load<u64>(builder + WbTextures);
    c.ret.SetU64(index >= 0 && index < c.Load<s32>(builder + WbTextureCount)
                     ? c.Load<u64>(textures + 8 * index)
                     : 0);
}

void nvnWindowBuilderGetNativeWindow(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + WbNative));
}

void nvnWindowBuilderGetPresentInterval(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + WbInterval) + 1);
}

void nvnWindowBuilderGetNumActiveTextures(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + WbActiveTextureCount));
}

void nvnWindowBuilderSetColorSpace(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u32>(builder + WbColorSpace, c.args.NextU32());
}

void nvnWindowBuilderGetColorSpace(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + WbColorSpace));
}

void SetDisplayMetadataArguments(Call& c, u64 display) {
    for (u64 offset = 0; offset < 10 * sizeof(f32); offset += sizeof(f32)) {
        c.Store<f32>(display + offset, c.args.NextF32());
    }
}

void GetDisplayMetadata(Call& c, u64 display) {
    const auto metadata = LoadMetadata(c, display);
    const std::array values = {
        metadata.display_primary_red[0],   metadata.display_primary_red[1],
        metadata.display_primary_green[0], metadata.display_primary_green[1],
        metadata.display_primary_blue[0],  metadata.display_primary_blue[1],
        metadata.white_point[0],           metadata.white_point[1],
        metadata.min_luminance,            metadata.max_luminance,
    };
    for (const f32 value : values) {
        c.Store<f32>(c.args.NextPointer(), value);
    }
}

void nvnWindowBuilderSetHdrContentMetadata(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<f32>(builder + WbHdrContentMetadata, c.args.NextF32());
    c.Store<f32>(builder + WbHdrContentMetadata + 4, c.args.NextF32());
}

void nvnWindowBuilderSetHdrDisplayMetadata(Call& c) {
    SetDisplayMetadataArguments(c, c.args.NextPointer() + WbHdrDisplayMetadata);
}

void nvnWindowBuilderGetHdrContentMetadata(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<f32>(c.args.NextPointer(), c.Load<f32>(builder + WbHdrContentMetadata));
    c.Store<f32>(c.args.NextPointer(), c.Load<f32>(builder + WbHdrContentMetadata + 4));
}

void nvnWindowBuilderGetHdrDisplayMetadata(Call& c) {
    GetDisplayMetadata(c, c.args.NextPointer() + WbHdrDisplayMetadata);
}

void nvnWindowSetHdrContentMetadata(Call& c) {
    const u64 window = c.args.NextPointer();
    c.Store<f32>(window + WindowHdrContentMetadata, c.args.NextF32());
    c.Store<f32>(window + WindowHdrContentMetadata + 4, c.args.NextF32());
    u32 color_space = 0;
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
            color_space = it->second.color_space;
        }
    }
    PublishWindowRequest(c, window, color_space, window + WindowHdrDisplayMetadata);
}

void nvnWindowSetHdrDisplayMetadata(Call& c) {
    const u64 window = c.args.NextPointer();
    SetDisplayMetadataArguments(c, window + WindowHdrDisplayMetadata);
    u32 color_space = 0;
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
            color_space = it->second.color_space;
        }
    }
    PublishWindowRequest(c, window, color_space, window + WindowHdrDisplayMetadata);
}

void nvnWindowGetHdrContentMetadata(Call& c) {
    const u64 window = c.args.NextPointer();
    c.Store<f32>(c.args.NextPointer(), c.Load<f32>(window + WindowHdrContentMetadata));
    c.Store<f32>(c.args.NextPointer(), c.Load<f32>(window + WindowHdrContentMetadata + 4));
}

void nvnWindowGetHdrDisplayMetadata(Call& c) {
    GetDisplayMetadata(c, c.args.NextPointer() + WindowHdrDisplayMetadata);
}

void nvnWindowInitialize(Call& c) {
    const u64 window = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    // The SDK takes min(count, 4) as unsigned (3.2.0 sdk+0x5efdc0): a negative count means 4.
    const s32 count = c.Load<s32>(builder + WbTextureCount);
    const u32 used = std::min(static_cast<u32>(count), static_cast<u32>(MaxWindowTextures));
    const s32 interval = c.Load<s32>(builder + WbInterval) + 1;
    const u64 texture_array = c.Load<u64>(builder + WbTextures);
    std::array<u64, MaxWindowTextures> textures{};
    for (u32 i = 0; i < used; ++i) {
        textures[i] = c.Load<u64>(texture_array + 8 * i);
    }
    if (!c.rec.WindowInitialize(window, std::span<const u64>{textures.data(), used},
                                static_cast<u32>(interval))) {
        LOG_ERROR(Core, "[nvn] WindowInitialize failed ({} textures)", count);
        c.ret.SetBool(false);
        return;
    }
    const auto metadata = LoadMetadata(c, builder + WbHdrDisplayMetadata);
    StoreDisplayMetadata(c, window + WindowHdrDisplayMetadata, metadata);
    StoreContentMetadata(c, window + WindowHdrContentMetadata, metadata);
    Runtime::WindowState state;
    state.count = used;
    state.native_window = c.Load<u64>(builder + WbNative);
    state.color_space = c.Load<u32>(builder + WbColorSpace);
    state.present_interval = interval;
    if (used != 0) {
        if (const auto texture = c.rec.GetTexture(textures[0])) {
            state.first_width = texture->width;
            state.first_height = texture->height;
        }
    }
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        c.runtime.windows[window] = state;
    }
    PublishWindowRequest(c, window, state.color_space, window + WindowHdrDisplayMetadata);
    LOG_INFO(Core, "[nvn] WindowInitialize swap chain {} textures, interval {}, first {}x{}", used,
             interval, state.first_width, state.first_height);
    c.ret.SetBool(true);
}

void nvnWindowFinalize(Call& c) {
    const u64 window = c.args.NextPointer();
    c.rec.WindowFinalize(window);
    std::scoped_lock lock{c.runtime.window_mutex};
    c.runtime.windows.erase(window);
    if (c.runtime.guest_request_producer == window) {
        Common::Hdr::PublishGuestRequest({});
        c.runtime.guest_request_producer = 0;
    }
}

void nvnWindowSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

void nvnWindowGetNativeWindow(Call& c) {
    const u64 window = c.args.NextPointer();
    std::scoped_lock lock{c.runtime.window_mutex};
    const auto it = c.runtime.windows.find(window);
    c.ret.SetU64(it != c.runtime.windows.end() ? it->second.native_window : 0);
}

void nvnWindowGetPresentInterval(Call& c) {
    const u64 window = c.args.NextPointer();
    std::scoped_lock lock{c.runtime.window_mutex};
    const auto it = c.runtime.windows.find(window);
    c.ret.SetU32(static_cast<u32>(it != c.runtime.windows.end() ? it->second.present_interval : 1));
}

void nvnWindowSetPresentInterval(Call& c) {
    const u64 window = c.args.NextPointer();
    const s32 interval = c.args.NextS32();
    c.rec.WindowSetPresentInterval(window, static_cast<u32>(interval));
    std::scoped_lock lock{c.runtime.window_mutex};
    if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
        if (it->second.present_interval != interval) {
            LOG_DEBUG(Core, "[nvn] the game sets present interval {} -> {}",
                      it->second.present_interval, interval);
        }
        it->second.present_interval = interval;
    }
}

void nvnWindowSetCrop(Call& c) {
    const u64 window = c.args.NextPointer();
    const s32 x = c.args.NextS32();
    const s32 y = c.args.NextS32();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    c.rec.WindowSetCrop(window, VideoCommon::Nvn::PresentCrop{x, y, width, height});
    std::scoped_lock lock{c.runtime.window_mutex};
    if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
        it->second.crop[0] = x;
        it->second.crop[1] = y;
        it->second.crop[2] = width;
        it->second.crop[3] = height;
    }
}

/// Before any SetCrop the crop is the whole first texture, as NVN starts it.
void nvnWindowGetCrop(Call& c) {
    const u64 window = c.args.NextPointer();
    const u64 out = c.args.NextPointer();
    s32 rect[4]{};
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
            if (it->second.crop[2] > 0 && it->second.crop[3] > 0) {
                std::copy(std::begin(it->second.crop), std::end(it->second.crop), rect);
            } else {
                rect[2] = static_cast<s32>(it->second.first_width);
                rect[3] = static_cast<s32>(it->second.first_height);
            }
        }
    }
    c.Store(out, rect);
}

void nvnWindowGetNumTextures(Call& c) {
    const u64 window = c.args.NextPointer();
    std::scoped_lock lock{c.runtime.window_mutex};
    const auto it = c.runtime.windows.find(window);
    c.ret.SetU32(it != c.runtime.windows.end() ? it->second.count : 0);
}

void nvnWindowSetNumActiveTextures(Call& c) {
    static_cast<void>(c);
}

constexpr u64 DisplayPeriodNs = 16'666'667; // 60 Hz; a present interval counts these
constexpr u64 OnDisplay = ~u64{0};

u64 HostNanoseconds() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Hands out the free slot whose last present is oldest. A slot is free when the game does not
/// hold it (acquired and not yet presented), its last present completed on the GPU and its
/// texture left the display (the next frame replaced it). Sets the slot's acquired bit. When no
/// slot is free yet only because of the display, `*wait_ns` receives the time until the first one.
int TryAcquireSlot(Call& c, u64 window, u64* wait_ns) {
    const u64 completed = c.rec.CompletedSeq();
    const u64 now = HostNanoseconds();
    std::scoped_lock lock{c.runtime.window_mutex};
    const auto it = c.runtime.windows.find(window);
    if (it == c.runtime.windows.end()) {
        return -2;
    }
    Runtime::WindowState& state = it->second;
    int best = -1;
    u64 best_seq = 0;
    for (u32 slot = 0; slot < state.count; ++slot) {
        if ((state.acquired_mask >> slot) & 1) {
            continue;
        }
        const u64 seq = c.rec.WindowSlotSeq(window, slot);
        if (seq > completed) {
            continue;
        }
        const u64 avail = state.avail_ns[slot];
        if (avail > now) {
            if (avail != OnDisplay && wait_ns != nullptr && (*wait_ns == ~u64{0} || avail - now < *wait_ns)) {
                *wait_ns = avail - now;
            }
            continue;
        }
        if (best < 0 || seq < best_seq) {
            best = static_cast<int>(slot);
            best_seq = seq;
        }
    }
    if (best >= 0) {
        state.acquired_mask |= 1u << best;
    }
    return best;
}

bool AcquireTexture(Call& c, u64 window, u64 index_out) {
    int slot = -1;
    while (true) {
        u64 wait_ns = ~u64{0};
        slot = TryAcquireSlot(c, window, &wait_ns);
        if (slot != -1) {
            break;
        }
        // Sleep until a completion frees a slot, or until the display releases one. An indefinite
        // wait also ends when a display deadline appears (nothing notifies at that time), so the
        // loop rearms with that deadline.
        u64 next_wait_ns = ~u64{0};
        bool ready;
        {
            ready = c.WaitUntil(
                [&] {
                    next_wait_ns = ~u64{0};
                    slot = TryAcquireSlot(c, window, &next_wait_ns);
                    return slot != -1 || (wait_ns == ~u64{0} && next_wait_ns != ~u64{0});
                },
                wait_ns == ~u64{0} ? std::chrono::nanoseconds{-1}
                                   : std::chrono::nanoseconds{static_cast<s64>(wait_ns)});
        }
        if (ready) {
            if (slot == -1) {
                continue; // a display deadline appeared: wait for it
            }
            break;
        }
        if (wait_ns == ~u64{0}) {
            return false; // stopped
        }
    }
    if (slot < 0) {
        return false;
    }
    if (!c.Store<s32>(index_out, slot)) {
        // The game never learns the index: hand the slot back instead of leaking it.
        std::scoped_lock lock{c.runtime.window_mutex};
        if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
            it->second.acquired_mask &= ~(1u << slot);
        }
        return false;
    }
    return true;
}

/// SDK (3.2.0 sdk+0x5f0320, 1.0.0 sdk+0x5cc420): the acquired slot's 0x20-byte sync is copied
/// into the game's sync, a signalled sync or the fence of the texture's release by the display
/// (WindowInitialize sets every slot sync signalled, 3.2.0 sdk+0x5efdc0). The HLE hands a slot
/// out only once its texture is free, so the game's sync becomes signalled without a GPU fence:
/// nvnSyncWait answers ALREADY_SIGNALED and a queue wait on it waits for nothing.
void nvnWindowAcquireTexture(Call& c) {
    const u64 window = c.args.NextPointer();
    const u64 sync = c.args.NextPointer();
    const u64 index_out = c.args.NextPointer();
    const bool acquired = AcquireTexture(c, window, index_out);
    if (acquired && sync != 0) {
        c.rec.SyncSetSignaled(sync);
    }
    c.ret.SetU32(acquired ? AcquireSuccess : AcquireError);
}

/// The older API: the queue waits for the texture on the GPU; the host waits on the CPU.
void nvnQueueAcquireTexture(Call& c) {
    static_cast<void>(c.args.NextPointer()); // queue
    const u64 window = c.args.NextPointer();
    const u64 index_out = c.args.NextPointer();
    c.ret.SetU32(AcquireTexture(c, window, index_out) ? 0 : 1);
}

/// The legacy nx2:vi2 present service logs the time the guest spends per present every 600
/// presents; the same cadence here, for the fields this path has (it has no GPU fence wait or
/// composite on the guest thread).
void LogPresentStats(std::chrono::steady_clock::time_point entered,
                     std::chrono::steady_clock::time_point paced) {
    struct Stats {
        long long frame_us{}, max_frame_us{}, hitches{}, pace_us{};
        u32 presents{};
        std::chrono::steady_clock::time_point last_present{};
    };
    static std::mutex mutex;
    static Stats stats;
    const auto to_us = [](auto d) -> long long {
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
    };
    std::scoped_lock lock{mutex};
    static Common::BootCounter boot;
    if (boot.First()) {
        stats = {}; // the first present of a boot: nothing of the previous game's frames counts
    }
    stats.pace_us += to_us(paced - entered);
    if (stats.last_present != std::chrono::steady_clock::time_point{}) {
        const long long frame = to_us(entered - stats.last_present);
        stats.frame_us += frame;
        stats.max_frame_us = std::max(stats.max_frame_us, frame);
        stats.hitches += frame > 100'000 ? 1 : 0;
    }
    stats.last_present = entered;
    if (++stats.presents == 600) {
        LOG_INFO(Service,
                 "nvn last 600 presents: frame {} us (max {}, {} over 100 ms), speed limiter {} us",
                 stats.frame_us / 599, stats.max_frame_us, stats.hitches, stats.pace_us / 600);
        stats = {.last_present = entered};
    }
}

/// The pacing inputs of the settings now (and of the guest's video playback).
PresentPaceInputs PaceInputsFor(Call& c, s32 present_interval) {
    const bool video = c.system != nullptr && c.system->GetNVDECActive();
    const Kernel::KProcess* const process = c.system ? c.system->ApplicationProcess() : nullptr;
    const bool mkw_adaptive = process &&
                              process->GetProgramId() == Settings::MarioKartWorldProgramId &&
                              process->IsMkwDynamicFpsPatchApplied() &&
                              !Settings::values.mkw_30fps.GetValue();
    return {
        .present_interval = AdaptivePresentInterval(present_interval, mkw_adaptive, video),
        .frame_rate_cap = Settings::FrameRateCap(),
        .use_speed_limit = Settings::values.use_speed_limit.GetValue(),
        .speed_limit_percent = Settings::SpeedLimit(),
        .video_framerate = Settings::values.use_video_framerate.GetValue() && video,
    };
}

void PacePresent(Call& c, const PresentPaceInputs& pace) {
    const PacePlan plan = PlanPresentPacing(pace);
    if (plan.mode == PaceMode::SpeedLimiter) {
        c.system->SpeedLimiter().DoSpeedLimiting(c.system->CoreTiming().GetGlobalTimeUs());
        return;
    }
    if (plan.mode == PaceMode::None) {
        return;
    }
    const auto interval = plan.interval;
    thread_local std::chrono::steady_clock::time_point next_present{};
    const auto now = std::chrono::steady_clock::now();
    if (now > next_present + interval) {
        // Running late: show at once and don't bank the lost time.
        next_present = now;
    } else if (now < next_present) {
        auto& kernel = c.system->Kernel();
        const auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(next_present - now);
        static_cast<void>(Kernel::GetCurrentThread(kernel).Sleep(
            kernel, kernel.HardwareTimer().GetTick() + wait.count() + 2));
    }
    next_present += interval;
}

/// EDEN_NX2_MAX_FRAMES_IN_FLIGHT (present_pacing.h PresentDepthLimit): the present returns once
/// the present `n` frames earlier has completed on the GPU. Unset defaults to 1.
/// `self` waits for this present itself (n = 0); 0 or off opts out.
void WaitPresentDepth(Call& c, u64 seq) {
    u64 target = 0;
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        if (!c.runtime.present_depth_read) {
            c.runtime.present_depth_read = true;
            const char* value = std::getenv("EDEN_NX2_MAX_FRAMES_IN_FLIGHT");
            if (value == nullptr) {
                // Compatibility with the candidate switch; the new name wins when both are set.
                value = std::getenv("EDEN_NX2_TRY_MAX_FRAMES_IN_FLIGHT");
            }
            const Kernel::KProcess* const process =
                c.system ? c.system->ApplicationProcess() : nullptr;
            const bool is_mkw =
                process != nullptr && process->GetProgramId() == Settings::MarioKartWorldProgramId;
            if (value == nullptr && is_mkw) {
                c.runtime.present_depth.emplace(2);
                LOG_INFO(Core,
                         "[nvn] MAX_FRAMES_IN_FLIGHT: a present waits until at most 2 "
                         "earlier present(s) are unfinished on the GPU "
                         "(source: Mario Kart World default)");
            } else if (const auto n = ParseMaxFramesInFlight(value)) {
                c.runtime.present_depth.emplace(*n);
                LOG_INFO(Core, "[nvn] MAX_FRAMES_IN_FLIGHT: a present waits until at most {} "
                               "earlier present(s) are unfinished on the GPU (source: {})",
                         *n, value == nullptr ? "default" : "environment");
            } else if (value != nullptr) {
                LOG_INFO(Core, "[nvn] MAX_FRAMES_IN_FLIGHT: extra GPU wait disabled by "
                               "environment ('{}')",
                         value);
            }
        }
        if (!c.runtime.present_depth) {
            return;
        }
        target = c.runtime.present_depth->OnPresent(seq);
    }
    if (target == 0 || c.rec.CompletedSeq() >= target) {
        return;
    }
    // Bounded: a present the GPU can only finish after a guest signal that follows this return
    // must not deadlock the game; say so once and let it run ahead.
    if (!c.WaitUntil([&] { return c.rec.CompletedSeq() >= target; }, std::chrono::seconds{1}) &&
        !c.runtime.HasFailed()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LOG_WARNING(Core, "[nvn] MAX_FRAMES_IN_FLIGHT: seq {} not done after 1 s (completed "
                              "{}); the present returns without it",
                        target, c.rec.CompletedSeq());
        }
    }
}

void nvnQueuePresentTexture(Call& c) {
    const u64 queue = c.args.NextPointer();
    const u64 window = c.args.NextPointer();
    const s32 slot = c.args.NextS32();
    const u64 present_seq =
        slot < 0 ? 0 : c.rec.QueuePresentTexture(queue, window, static_cast<u32>(slot));
    if (present_seq == 0) {
        LOG_ERROR(Core, "[nvn] QueuePresentTexture failed (window {:#x}, slot {})", window, slot);
        if (slot >= 0 && slot < 4) {
            // The game gives the texture up either way: do not leak its slot.
            std::scoped_lock lock{c.runtime.window_mutex};
            if (const auto it = c.runtime.windows.find(window); it != c.runtime.windows.end()) {
                it->second.acquired_mask &= ~(1u << slot);
            }
        }
        return;
    }
    PresentPaceInputs pace;
    {
        std::scoped_lock lock{c.runtime.window_mutex};
        const auto it = c.runtime.windows.find(window);
        pace = PaceInputsFor(c, it != c.runtime.windows.end() ? it->second.present_interval : 1);
        if (it != c.runtime.windows.end()) {
            Runtime::WindowState& state = it->second;
            state.acquired_mask &= ~(1u << slot);
            ++state.presents;
            // The frame is shown at the next display slot after the previous one (interval
            // vsyncs later), or at once when the game is late. Showing it frees the previous
            // frame's texture.
            const u64 now = HostNanoseconds();
            const u64 period = ScaledDisplayPeriodNs(
                DisplayPeriodNs * static_cast<u64>(std::max(pace.present_interval, 0)), pace);
            const u64 display = state.last_display_ns == 0 ? now : std::max(state.last_display_ns + period, now);
            if (state.displayed_slot >= 0 && state.displayed_slot != slot &&
                state.displayed_slot < 4) {
                state.avail_ns[state.displayed_slot] = display;
            }
            state.avail_ns[slot] = OnDisplay;
            state.displayed_slot = slot;
            state.last_display_ns = display;
        }
    }
    const u64 presents = c.runtime.presents.fetch_add(1, std::memory_order_relaxed) + 1;
    if (presents <= 4 || presents % 600 == 0) {
        LOG_INFO(Core, "[nvn] present #{} (window {:#x}, slot {})", presents, window, slot);
    }
    // The frame counter and pacing normally run in the nx2:vi2 present service, which this path
    // replaces: frame-locked key scripts, the census `calls_per_frame` column and the speed limiter
    // depend on them.
    Core::g_nx2_present_count.fetch_add(1, std::memory_order_relaxed);
    WaitPresentDepth(c, present_seq);
    if (c.system != nullptr) {
        const auto entered = std::chrono::steady_clock::now();
        PacePresent(c, pace);
        const auto paced = std::chrono::steady_clock::now();
        Common::FrameStats::AddStall(
            Common::FrameStats::Cause::GuestPace,
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(paced - entered).count()));
        Common::FrameStats::OnPresent(entered);
        Common::DiagRequest::Poll(entered);
        Core::PerfStats& perf = c.system->GetPerfStats();
        perf.EndSystemFrame();
        // The window-title FPS that tools (eden-mcp perf, ab_runner) read counts game frames, which
        // the renderer normally reports as it composites a frame; this path draws itself, so the
        // guest present is the frame.
        perf.EndGameFrame();
        perf.BeginSystemFrame();
        LogPresentStats(entered, paced);
    }
}

} // namespace

void RegisterWindowFunctions(Registry& registry) {
    NVN_HLE(registry, nvnWindowBuilderSetDefaults);
    NVN_HLE(registry, nvnWindowBuilderSetDevice);
    NVN_HLE(registry, nvnWindowBuilderSetNativeWindow);
    NVN_HLE(registry, nvnWindowBuilderSetTextures);
    NVN_HLE(registry, nvnWindowBuilderSetPresentInterval);
    NVN_HLE(registry, nvnWindowBuilderSetNumActiveTextures);
    NVN_HLE(registry, nvnWindowBuilderGetDevice);
    NVN_HLE(registry, nvnWindowBuilderGetNumTextures);
    NVN_HLE(registry, nvnWindowBuilderGetTexture);
    NVN_HLE(registry, nvnWindowBuilderGetNativeWindow);
    NVN_HLE(registry, nvnWindowBuilderGetPresentInterval);
    NVN_HLE(registry, nvnWindowBuilderGetNumActiveTextures);
    NVN_HLE(registry, nvnWindowBuilderGetColorSpace);
    NVN_HLE(registry, nvnWindowBuilderGetHdrContentMetadata);
    NVN_HLE(registry, nvnWindowBuilderGetHdrDisplayMetadata);
    NVN_HLE(registry, nvnWindowBuilderSetColorSpace);
    NVN_HLE(registry, nvnWindowBuilderSetHdrContentMetadata);
    NVN_HLE(registry, nvnWindowBuilderSetHdrDisplayMetadata);
    NVN_HLE(registry, nvnWindowGetHdrContentMetadata);
    NVN_HLE(registry, nvnWindowGetHdrDisplayMetadata);
    NVN_HLE(registry, nvnWindowSetHdrContentMetadata);
    NVN_HLE(registry, nvnWindowSetHdrDisplayMetadata);
    NVN_HLE(registry, nvnWindowInitialize);
    NVN_HLE(registry, nvnWindowFinalize);
    NVN_HLE(registry, nvnWindowSetDebugLabel);
    NVN_HLE(registry, nvnWindowAcquireTexture);
    NVN_HLE(registry, nvnWindowGetNativeWindow);
    NVN_HLE(registry, nvnWindowGetPresentInterval);
    NVN_HLE(registry, nvnWindowSetPresentInterval);
    NVN_HLE(registry, nvnWindowSetCrop);
    NVN_HLE(registry, nvnWindowGetCrop);
    NVN_HLE(registry, nvnWindowSetNumActiveTextures);
    registry.Add<&nvnWindowGetNumTextures>("nvnWindowGetNumActiveTextures");
    NVN_HLE(registry, nvnWindowGetNumTextures);
    NVN_HLE(registry, nvnQueueAcquireTexture);
    NVN_HLE(registry, nvnQueuePresentTexture);
}

} // namespace Core::NvnHle::Hle
