// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace Core::NvnHle {

/// The patched MKW main loop measures the full frame. Let a late race frame use the next
/// 60 Hz slot without rounding it up to a second vblank. Video keeps the requested interval.
[[nodiscard]] constexpr s32 AdaptivePresentInterval(s32 requested, bool mkw_adaptive,
                                                    bool video) noexcept {
    return mkw_adaptive && !video && requested == 2 ? 1 : requested;
}

/// What the pacing of one NVN present depends on.
struct PresentPaceInputs {
    /// The swap interval the display uses (60 Hz vsyncs per frame; the window's, or 1 while the
    /// game's VRR capability is on).
    s32 present_interval{1};
    /// Settings::FrameRateCap(): 0 for the "Auto" Frame Rate Cap, else the capped rate in Hz.
    u32 frame_rate_cap{};
    bool use_speed_limit{true};
    u32 speed_limit_percent{100};
    /// The "Sync to framerate of video playback" setting is on and the guest is decoding a video
    /// (System::GetNVDECActive): as in the VI conductor (vi/conductor.cpp GetNextTicks), the
    /// frame period is the intended presentation rate whatever the speed limit says.
    bool video_framerate{};
};

enum class PaceMode {
    /// No frame rate cap: the emulator's speed limiter paces the present.
    SpeedLimiter,
    /// No pacing here (speed limit off, or a limit of 0): the display slots alone pace the game.
    None,
    /// Hold the present until `PacePlan::interval` after the previous one.
    Interval,
};

struct PacePlan {
    PaceMode mode{PaceMode::None};
    std::chrono::nanoseconds interval{};
};

/// The present pacing of the NVN HLE: the same rules as the nx2:vi2 present
/// (Nx2Gsv::PaceToVblank) and the Vulkan swapchain's frame pacing. The interval is the swap
/// interval's period at the speed limit, never shorter than the Frame Rate Cap's period (a cap
/// only slows the game down). While a video plays with "Sync to framerate of video playback" the
/// speed limit counts as 100% (and applies even when it is off), so a video keeps its intended
/// rate under a turbo or slow-motion limit.
[[nodiscard]] constexpr PacePlan PlanPresentPacing(const PresentPaceInputs& in) noexcept {
    const bool limited = in.use_speed_limit && in.speed_limit_percent != 0;
    if (!in.video_framerate) {
        if (in.frame_rate_cap == 0) {
            return {PaceMode::SpeedLimiter, {}};
        }
        if (!limited) {
            return {PaceMode::None, {}};
        }
    }
    const s64 percent = in.video_framerate ? 100 : in.speed_limit_percent;
    const s64 swap = std::clamp<s64>(in.present_interval, 1, 4);
    s64 ns = 100'000'000'000LL * swap / 60 / percent;
    if (in.frame_rate_cap != 0) {
        ns = std::max<s64>(ns, 100'000'000'000LL / in.frame_rate_cap / percent);
    }
    return {PaceMode::Interval, std::chrono::nanoseconds{ns}};
}

/// The display's slot period (60 Hz per swap interval, `period_ns` for the present's interval)
/// at the speed limit: a texture leaves the display, and the game may acquire it again, one
/// period later. This is where the Speed Limit % setting acts in multi-core mode (the emulated
/// GPU scales the VI vsync event the same way, Conductor::GetNextTicks, and Nx2Gsv::PaceToVblank
/// its vblank interval): 50% doubles the period, 200% halves it. No speed limit, or a limit of
/// 0, leaves the period alone; a video with "Sync to framerate of video playback" runs at 100%.
[[nodiscard]] constexpr u64 ScaledDisplayPeriodNs(u64 period_ns,
                                                  const PresentPaceInputs& in) noexcept {
    if (in.video_framerate || !in.use_speed_limit || in.speed_limit_percent == 0) {
        return period_ns;
    }
    return period_ns * 100 / in.speed_limit_percent;
}

/// EDEN_NX2_MAX_FRAMES_IN_FLIGHT=<n> bounds how far the game runs ahead of the host GPU.
/// A present returns once the present `n` frames before it has completed (1 is the previous
/// present, 2 is the one before that). The default is 1.
/// `self` (n = 0) returns a present only once this present itself completed: no frame stays in
/// flight (the CPU/GPU overlap is gone; every GPU result the game reads next frame is complete).
/// Setting 0 or off removes this extra limit; then only the window's texture count bounds the
/// queue (a slot is handed out once its last present completed), so a game that reads GPU results
/// back without a fence (OoT's exposure sums) sees them up to (textures - 1) frames late.
class PresentDepthLimit {
public:
    static constexpr u32 MaxDepth = 7;

    explicit constexpr PresentDepthLimit(u32 max_unfinished) noexcept
        : limit{std::min(max_unfinished, MaxDepth)} {}

    [[nodiscard]] constexpr u32 Limit() const noexcept {
        return limit;
    }

    /// Records the submission sequence of a present (in present order). Returns the sequence that
    /// must have completed before this present returns, or 0 when there is none to wait for yet.
    [[nodiscard]] constexpr u64 OnPresent(u64 seq) noexcept {
        ring[count % ring.size()] = seq;
        ++count;
        if (count <= limit) {
            return 0;
        }
        return ring[(count - 1 - limit) % ring.size()];
    }

private:
    u32 limit;
    u64 count{};
    std::array<u64, MaxDepth + 1> ring{};
};

/// Resolves EDEN_NX2_MAX_FRAMES_IN_FLIGHT: unset = 1; `self` = 0 (wait for this present to
/// complete on the GPU); 0, off, empty or invalid = unlimited.
[[nodiscard]] inline std::optional<u32> ParseMaxFramesInFlight(const char* value) noexcept {
    if (value == nullptr) {
        return 1;
    }
    if (std::string_view{value} == "self") {
        return 0;
    }
    if (std::string_view{value} == "off") {
        return std::nullopt;
    }
    if (*value < '0' || *value > '9') {
        return std::nullopt;
    }
    char* end = nullptr;
    const unsigned long n = std::strtoul(value, &end, 10);
    if (*end != '\0' || n == 0) {
        return std::nullopt;
    }
    return static_cast<u32>(std::min<unsigned long>(n, PresentDepthLimit::MaxDepth));
}

} // namespace Core::NvnHle
