// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>

#include "common/common_types.h"

/// Translation of host mouse motion into the relative movement a Joy-Con 2 reports in mouse mode.
/// Kept separate from the Qt and HID code so the scaling and clamping can be unit tested.
namespace InputCommon::MouseMode {

struct Config {
    /// 1.0 is the host mouse at 1:1. Values below slow the emulated sensor down.
    float sensitivity{1.0f};
    bool invert_x{};
    bool invert_y{};
    /// Ignore motion smaller than this many host pixels per event (jitter from some sensors).
    float deadzone_pixels{};
};

struct Delta {
    float x{};
    float y{};

    friend constexpr bool operator==(const Delta&, const Delta&) = default;
};

[[nodiscard]] inline float ClampSensitivity(float percent) {
    return std::clamp(percent, 10.0f, 400.0f) / 100.0f;
}

/// Applies sensitivity, inversion and the deadzone to a host mouse delta.
[[nodiscard]] inline Delta Scale(int dx, int dy, const Config& config) {
    Delta out{};
    const float x = static_cast<float>(dx);
    const float y = static_cast<float>(dy);
    if (std::abs(x) <= config.deadzone_pixels && std::abs(y) <= config.deadzone_pixels) {
        return out;
    }
    out.x = x * config.sensitivity * (config.invert_x ? -1.0f : 1.0f);
    out.y = y * config.sensitivity * (config.invert_y ? -1.0f : 1.0f);
    return out;
}

/// Accumulates relative motion between HID polls, with a soft speed clamp so a single fast flick
/// cannot exceed what a real optical sensor would report in one report.
class Accumulator {
public:
    static constexpr float MaxPerReport = 2047.0f;

    void Add(const Delta& delta) {
        pending_.x += delta.x;
        pending_.y += delta.y;
        pending_.x = std::clamp(pending_.x, -MaxPerReport, MaxPerReport);
        pending_.y = std::clamp(pending_.y, -MaxPerReport, MaxPerReport);
    }

    [[nodiscard]] Delta Pending() const {
        return pending_;
    }

    /// Returns the motion accumulated since the last call and starts a new report.
    Delta Consume() {
        const Delta out = pending_;
        pending_ = {};
        return out;
    }

    void Reset() {
        pending_ = {};
    }

private:
    Delta pending_{};
};

/// A right-stick projection for games that do not read the mouse device: the latest motion pushes
/// the stick and it decays back to centre while the mouse is still, like mouse panning.
class StickProjection {
public:
    void Push(const Delta& delta, float scale = 0.02f) {
        x_ = std::clamp(x_ + delta.x * scale, -1.5f, 1.5f);
        y_ = std::clamp(y_ + delta.y * scale, -1.5f, 1.5f);
    }

    /// dt_ms is the time since the previous update; decay is in [0, 1] per second.
    void Decay(float dt_ms, float decay_per_second = 12.0f) {
        const float factor = std::exp(-std::max(0.0f, decay_per_second) * dt_ms / 1000.0f);
        x_ *= factor;
        y_ *= factor;
        if (std::abs(x_) < 0.001f) {
            x_ = 0.0f;
        }
        if (std::abs(y_) < 0.001f) {
            y_ = 0.0f;
        }
    }

    /// Stick axes in [-1, 1], y positive up.
    [[nodiscard]] Delta Value() const {
        return {std::clamp(x_, -1.0f, 1.0f), -std::clamp(y_, -1.0f, 1.0f)};
    }

    [[nodiscard]] bool Active() const {
        return x_ != 0.0f || y_ != 0.0f;
    }

    void Reset() {
        x_ = 0.0f;
        y_ = 0.0f;
    }

private:
    float x_{};
    float y_{};
};

} // namespace InputCommon::MouseMode
