// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// HD rumble on the eccentric-mass motors of an Xbox-style pad: the mapping from the guest's two
// frequency bands to motor strengths, and the scheduler that decides when the operating system is
// actually called. Nothing in here touches SDL, the settings or a clock, so both halves are unit
// tested without hardware (src/tests/input_common/xbox_hd_rumble.cpp).
//
// The mapping follows the "Direct" Xbox mode of the Eglantine hdrumble-mvp branch
// (https://github.com/Octelly/Eglantine, samaBR85 and hermes): the low band drives the heavy
// motor, the high band the light one, part of the high band is added to the heavy motor, and a
// sudden rise gets a short burst. The perceptual curve, the frequency weight, the change
// threshold, the minimum interval and the refresh are Veshiru's.

#pragma once

#include <optional>

#include "common/common_types.h"

namespace InputCommon {

/// Motor strengths as the operating system takes them, 0..65535.
/// left_motor is the heavy low-frequency grip motor, right_motor the light high-frequency one.
struct XboxMotors {
    u16 left_motor{};
    u16 right_motor{};
    u16 left_trigger{};
    u16 right_trigger{};

    bool operator==(const XboxMotors&) const = default;

    bool IsZero() const {
        return left_motor == 0 && right_motor == 0 && left_trigger == 0 && right_trigger == 0;
    }
};

/// The two grip motors of one mapped vibration.
struct XboxGripMotors {
    u16 left_motor{};
    u16 right_motor{};

    bool operator==(const XboxGripMotors&) const = default;
};

/// User tunables of the mapping, as plain factors (1.0 = 100 %).
struct XboxRumbleParams {
    /// Multiplies the guest's amplitude before the curves.
    f32 gain{1.0f};
    /// 0..1: lifts weak vibrations more than strong ones.
    f32 low_boost{0.0f};
    /// Multiplies the motor strengths after the curves. 0 silences the pad.
    f32 strength{1.0f};
    /// 0..1: share of the high band that is also played on the heavy motor.
    f32 detail_mix{0.5f};
    /// 0..1: share of the light motor's strength that is played on the trigger motors.
    f32 trigger_mix{0.0f};
};

/// One half of the emulated controller (left or right Joy-Con side) as the guest sent it.
struct HdRumbleSide {
    f32 low_amplitude{};
    f32 low_frequency{160.0f};
    f32 high_amplitude{};
    f32 high_frequency{320.0f};
};

/// Gain, then a power curve that lifts weak vibrations more than strong ones (exponent 1.0 at
/// low_boost 0 down to 0.4 at low_boost 1). Below a small knee the curve turns quadratic so the
/// near-silent values games send between hits do not become a constant buzz. Returns 0..1; a NaN
/// or negative input is silence.
f32 ShapeHdRumbleAmplitude(f32 amplitude, f32 gain, f32 low_boost);

/// Maps one HD rumble sample to the two grip motors. The low band goes to the heavy (left) motor
/// and the high band to the light (right) one.
///
/// An eccentric-mass motor has one knob: its drive level sets speed and force together. It cannot
/// play a frequency, so the band frequencies only weaken a band slightly (at most 15 %) as it
/// rises above the nominal 160 Hz / 320 Hz. The amplitude goes through a perceptual curve because
/// these motors barely turn at low drive levels. The result is monotonic in both amplitudes and
/// in strength, and is always within 0..65535.
XboxGripMotors MapHdRumbleToXboxMotors(f32 low_amplitude, f32 low_frequency, f32 high_amplitude,
                                       f32 high_frequency, const XboxRumbleParams& params);

/// Both halves of the emulated controller on one pad: each grip motor follows the stronger half,
/// and each trigger takes trigger_mix of its own half's light motor (left half on LT, right on RT).
XboxMotors MixXboxSides(const HdRumbleSide& left, const HdRumbleSide& right,
                        const XboxRumbleParams& params);

/// What the caller has to send to the pad.
struct XboxRumbleSend {
    XboxMotors motors{};
    /// True when the values are unchanged and only the bounded duration is being restarted.
    bool refresh{};
};

/// Decides when the pad is written to. A guest sends a vibration every few milliseconds; a pad
/// is written to when the target changed by more than a threshold (and not more often than a
/// minimum interval), when a stop arrives (always at once), or when the refresh interval ran out.
/// Every write carries DurationMs, so if the stop is lost, or nothing calls Tick any more, the
/// motors stop on their own.
class XboxRumbleScheduler {
public:
    struct Config {
        /// A playing vibration is sent again after this long.
        s64 refresh_interval_us{400'000};
        /// Changes closer together than this wait for Tick. Stops and hit accents do not wait.
        s64 min_interval_us{8'000};
        /// A change of every motor below this is not worth a write until the next refresh.
        u16 change_threshold{656};
        /// 0..1: how far a sudden rise overshoots towards full strength. 0 disables the burst.
        f32 hit_accent{0.0f};
        /// A grip motor rising by at least this much since the last write is a hit.
        u16 hit_rise{9830};
        s64 hit_duration_us{25'000};
    };

    /// How long the pad plays a write without another one.
    static constexpr u32 DurationMs = 1000;

    /// A new target from the guest. Returns what must be written now, if anything.
    std::optional<XboxRumbleSend> Offer(s64 now_us, const XboxMotors& new_target,
                                        const Config& config);

    /// A write whose timing the caller's source already decided (RenHDrum's own mix paces
    /// itself): always written, with no threshold, interval or accent. Tick still refreshes it.
    XboxRumbleSend Force(s64 now_us, const XboxMotors& new_target);

    /// Periodic: the end of a hit accent, a change that had to wait, or a refresh.
    std::optional<XboxRumbleSend> Tick(s64 now_us, const Config& config);

    /// Emulation stopped or paused, the pad or the setting went away: forgets the target and
    /// always returns zeros to write.
    XboxRumbleSend Stop(s64 now_us);

    /// The last write was refused by the pad: try the target again on the next Tick.
    void MarkFailed();

    const XboxMotors& LastSent() const {
        return sent;
    }

private:
    XboxRumbleSend Send(s64 now_us, const XboxMotors& motors, bool refresh);

    XboxMotors target{};
    XboxMotors sent{};
    s64 last_send_us{};
    s64 hit_until_us{};
    bool has_sent{};
    bool pending{};
};

} // namespace InputCommon
