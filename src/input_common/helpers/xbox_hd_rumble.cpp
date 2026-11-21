// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "input_common/helpers/xbox_hd_rumble.h"

namespace InputCommon {

namespace {

constexpr f32 MotorMax = 65535.0f;

/// 0 for NaN and anything not above zero, otherwise the value (which may be infinite).
f32 NonNegative(f32 value) {
    return value > 0.0f ? value : 0.0f;
}

f32 Clamp01(f32 value) {
    return (std::min)(NonNegative(value), 1.0f);
}

u16 ToMotor(f32 value) {
    return static_cast<u16>(std::lround(Clamp01(value) * MotorMax));
}

/// value^exponent on 0..1, quadratic below a small knee so near-silence stays near-silent.
f32 KneedPower(f32 value, f32 exponent) {
    constexpr f32 knee = 0.02f;
    if (value <= 0.0f) {
        return 0.0f;
    }
    if (value < knee) {
        const f32 ratio = value / knee;
        return std::pow(knee, exponent) * ratio * ratio;
    }
    return std::pow(value, exponent);
}

/// An eccentric-mass motor barely turns at low drive levels, and what is felt grows more slowly
/// than the drive level above that, so the low end is lifted. An assumption about this class of
/// motor, not a measurement of any pad.
f32 PerceptualCurve(f32 value) {
    constexpr f32 exponent = 0.6f;
    return KneedPower(value, exponent);
}

/// 1 at or below the nominal frequency, falling to 0.85 one octave above it and no further. A
/// frequency that is not a number, not finite or not positive is treated as nominal.
f32 FrequencyWeight(f32 frequency, f32 nominal) {
    constexpr f32 floor_weight = 0.85f;
    if (!std::isfinite(frequency) || frequency <= nominal) {
        return 1.0f;
    }
    const f32 octaves = (std::min)(std::log2(frequency / nominal), 1.0f);
    return 1.0f - (1.0f - floor_weight) * octaves;
}

bool Differs(const XboxMotors& a, const XboxMotors& b, u16 threshold) {
    const auto apart = [threshold](u16 x, u16 y) {
        return std::abs(static_cast<int>(x) - static_cast<int>(y)) > static_cast<int>(threshold);
    };
    return apart(a.left_motor, b.left_motor) || apart(a.right_motor, b.right_motor) ||
           apart(a.left_trigger, b.left_trigger) || apart(a.right_trigger, b.right_trigger);
}

u16 Overshoot(u16 value, f32 accent) {
    return ToMotor((static_cast<f32>(value) + (MotorMax - static_cast<f32>(value)) * accent) /
                   MotorMax);
}

} // Anonymous namespace

f32 ShapeHdRumbleAmplitude(f32 amplitude, f32 gain, f32 low_boost) {
    const f32 value = (std::min)(NonNegative(amplitude) * NonNegative(gain), 1.0f);
    // 0 * infinity is not a number; it is silence here
    if (!(value > 0.0f)) {
        return 0.0f;
    }
    const f32 boost = Clamp01(low_boost);
    if (boost <= 0.0f) {
        return value;
    }
    return KneedPower(value, 1.0f - 0.6f * boost);
}

XboxGripMotors MapHdRumbleToXboxMotors(f32 low_amplitude, f32 low_frequency, f32 high_amplitude,
                                       f32 high_frequency, const XboxRumbleParams& params) {
    constexpr f32 nominal_low_frequency = 160.0f;
    constexpr f32 nominal_high_frequency = 320.0f;

    const f32 low =
        PerceptualCurve(ShapeHdRumbleAmplitude(low_amplitude, params.gain, params.low_boost)) *
        FrequencyWeight(low_frequency, nominal_low_frequency);
    const f32 high =
        PerceptualCurve(ShapeHdRumbleAmplitude(high_amplitude, params.gain, params.low_boost)) *
        FrequencyWeight(high_frequency, nominal_high_frequency);

    // The heavy motor also carries part of the fine vibration, where it is felt
    const f32 heavy = (std::max)(low, high * Clamp01(params.detail_mix));
    const f32 strength = NonNegative(params.strength);
    // Clamp before multiplying so an infinite strength cannot meet a zero
    const auto scaled = [strength](f32 value) {
        return value > 0.0f && strength > 0.0f ? (std::min)(value * strength, 1.0f) : 0.0f;
    };
    return {
        .left_motor = ToMotor(scaled(heavy)),
        .right_motor = ToMotor(scaled(high)),
    };
}

XboxMotors MixXboxSides(const HdRumbleSide& left, const HdRumbleSide& right,
                        const XboxRumbleParams& params) {
    const XboxGripMotors left_half = MapHdRumbleToXboxMotors(
        left.low_amplitude, left.low_frequency, left.high_amplitude, left.high_frequency, params);
    const XboxGripMotors right_half =
        MapHdRumbleToXboxMotors(right.low_amplitude, right.low_frequency, right.high_amplitude,
                                right.high_frequency, params);
    const f32 trigger_mix = Clamp01(params.trigger_mix);
    const auto on_trigger = [trigger_mix](u16 value) {
        return ToMotor(static_cast<f32>(value) * trigger_mix / MotorMax);
    };
    return {
        .left_motor = (std::max)(left_half.left_motor, right_half.left_motor),
        .right_motor = (std::max)(left_half.right_motor, right_half.right_motor),
        .left_trigger = on_trigger(left_half.right_motor),
        .right_trigger = on_trigger(right_half.right_motor),
    };
}

XboxRumbleSend XboxRumbleScheduler::Send(s64 now_us, const XboxMotors& motors, bool refresh) {
    sent = motors;
    last_send_us = now_us;
    has_sent = true;
    pending = false;
    return {.motors = motors, .refresh = refresh};
}

std::optional<XboxRumbleSend> XboxRumbleScheduler::Offer(s64 now_us, const XboxMotors& new_target,
                                                         const Config& config) {
    target = new_target;

    // A stop is never held back or merged with anything
    if (target.IsZero()) {
        hit_until_us = 0;
        pending = false;
        if (sent.IsZero()) {
            return std::nullopt;
        }
        return Send(now_us, target, false);
    }

    if (hit_until_us != 0) {
        if (now_us < hit_until_us) {
            // Tick settles on the latest target when the burst ends
            return std::nullopt;
        }
        hit_until_us = 0;
    }

    // A sudden rise gets a short burst so the motor spins up in time for quick taps
    const f32 accent = Clamp01(config.hit_accent);
    if (accent > 0.0f) {
        const auto is_hit = [&config](u16 now, u16 before) {
            return static_cast<int>(now) - static_cast<int>(before) >=
                   static_cast<int>(config.hit_rise);
        };
        const bool left_hit = is_hit(target.left_motor, sent.left_motor);
        const bool right_hit = is_hit(target.right_motor, sent.right_motor);
        if (left_hit || right_hit) {
            XboxMotors burst = target;
            if (left_hit) {
                burst.left_motor = Overshoot(target.left_motor, accent);
            }
            if (right_hit) {
                burst.right_motor = Overshoot(target.right_motor, accent);
            }
            hit_until_us = now_us + config.hit_duration_us;
            return Send(now_us, burst, false);
        }
    }

    const bool starting = sent.IsZero();
    if (!starting && !Differs(target, sent, config.change_threshold)) {
        // Too small to be worth a write; the next refresh carries it
        return std::nullopt;
    }
    if (!starting && has_sent && now_us - last_send_us < config.min_interval_us) {
        pending = true;
        return std::nullopt;
    }
    return Send(now_us, target, false);
}

XboxRumbleSend XboxRumbleScheduler::Force(s64 now_us, const XboxMotors& new_target) {
    target = new_target;
    hit_until_us = 0;
    return Send(now_us, target, false);
}

std::optional<XboxRumbleSend> XboxRumbleScheduler::Tick(s64 now_us, const Config& config) {
    if (hit_until_us != 0) {
        if (now_us < hit_until_us) {
            return std::nullopt;
        }
        hit_until_us = 0;
        if (target != sent) {
            return Send(now_us, target, false);
        }
    }
    if (pending && now_us - last_send_us >= config.min_interval_us) {
        return Send(now_us, target, false);
    }
    if (!sent.IsZero() && now_us - last_send_us >= config.refresh_interval_us) {
        return Send(now_us, target, target == sent);
    }
    return std::nullopt;
}

XboxRumbleSend XboxRumbleScheduler::Stop(s64 now_us) {
    target = {};
    hit_until_us = 0;
    return Send(now_us, target, false);
}

void XboxRumbleScheduler::MarkFailed() {
    pending = true;
}

} // namespace InputCommon
