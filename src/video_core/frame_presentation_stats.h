// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <optional>

#include "common/common_types.h"

namespace VideoCore {

/// Successful host swapchain presents. These are not GPU scanout or guest frame counters.
struct FramePresentCounts {
    u64 original{};
    u64 generated{};
};

struct FramePresentRates {
    double original{};
    double generated{};
};

constexpr double FrameCountRate(u64 before, u64 after, double seconds) {
    return seconds > 0.0 && after >= before ? static_cast<double>(after - before) / seconds : 0.0;
}

constexpr FramePresentRates PresentRatesBetween(FramePresentCounts before, FramePresentCounts after,
                                                double seconds) {
    return {FrameCountRate(before.original, after.original, seconds),
            FrameCountRate(before.generated, after.generated, seconds)};
}

class FramePresentSampler {
public:
    using Clock = std::chrono::steady_clock;

    [[nodiscard]] std::optional<FramePresentRates> Sample(std::optional<FramePresentCounts> counts,
                                                          Clock::time_point now = Clock::now()) {
        const auto previous = last;
        last = counts;
        const auto seconds = std::chrono::duration<double>(now - last_time).count();
        last_time = now;
        if (!counts || !previous || seconds <= 0.0 || counts->original < previous->original ||
            counts->generated < previous->generated) {
            return std::nullopt;
        }
        return PresentRatesBetween(*previous, *counts, seconds);
    }

private:
    std::optional<FramePresentCounts> last;
    Clock::time_point last_time{};
};

} // namespace VideoCore
