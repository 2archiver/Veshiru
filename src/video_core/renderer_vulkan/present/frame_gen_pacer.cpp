// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>

#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/present/frame_gen_pacer.h"

namespace Vulkan {

namespace {

using Clock = std::chrono::steady_clock;

constexpr f32 INTERVAL_SMOOTHING = 0.25f;
constexpr f32 MINIMUM_BASE_RATE = 10.0f;
constexpr f32 BURST_CADENCE_RATIO = 3.0f;
constexpr f32 BURST_TARGET_RATIO = 2.0f;
constexpr f32 PROBE_THROUGHPUT_TOLERANCE = 0.95f;
constexpr f32 PROBE_BASE_RETENTION = 0.80f;
constexpr f32 STEADY_BASE_RETENTION = 0.70f;
constexpr f32 TARGET_SATISFIED_RATIO = 0.95f;
constexpr f32 TARGET_START_RATIO = 0.92f;
constexpr f32 CREDIT_EPSILON = 1.0e-4f;
constexpr u32 MAX_PROBE_FAILURES = 4;

constexpr auto STABILIZATION_DURATION = std::chrono::seconds(1);
constexpr auto PROBE_DURATION = std::chrono::seconds(1);
constexpr auto DEFICIT_DURATION = std::chrono::seconds(1);
constexpr auto PROBE_STEP_DELAY = std::chrono::milliseconds(250);

[[nodiscard]] Clock::duration ProbeBackoff(u32 failures) {
    switch (failures) {
    case 1:
        return std::chrono::seconds(5);
    case 2:
        return std::chrono::seconds(15);
    case 3:
        return std::chrono::seconds(30);
    default:
        return std::chrono::seconds(60);
    }
}

} // Anonymous namespace

// With a target rate, the generations of a frame fill its interval at that rate. The display
// takes no more than the target (FIFO at 60/120 Hz), so once a slow stretch has raised the count,
// the presents of each frame take the whole interval, the renderer waits for a free frame, and
// the interval it measures never shortens again: Ocarina of Time stayed at 42 fps with 1.9
// generated frames each (real + generated = 120 a second, the GPU thread waiting 6-11 ms of
// every frame for a frame to draw into) where it runs 60 with one. The time spent waiting for the
// present thread (excluding GPU execution fences) is subtracted only while we are actively
// generating frames, down to the interval the game or target is paced to.
f32 FrameGenPacer::PlanningInterval(f32 interval_seconds, f32 blocked_seconds, f32 target_rate,
                                    f32 guest_interval_seconds, bool had_generations) {
    if (!had_generations || blocked_seconds <= 0.0f) {
        return interval_seconds;
    }
    const f32 unblocked = interval_seconds - std::min(blocked_seconds, interval_seconds);
    if (guest_interval_seconds > 0.0f) {
        return std::max(unblocked, std::min(guest_interval_seconds, interval_seconds));
    }
    if (target_rate > 0.0f) {
        // Without a reported guest frame interval (e.g. Switch 1 titles or menus before guest
        // pacing settles), subtract FIFO present-queue waiting down to the target-rate interval so
        // generated frames cannot pin a 60 FPS menu at 30 real FPS on a 60 Hz FIFO display.
        return std::max(unblocked, std::min(1.0f / target_rate, interval_seconds));
    }
    return interval_seconds;
}

void FrameGenPacer::LogPlans(f32 interval_seconds, f32 blocked_seconds, size_t generations) {
    if (!Common::DkbFrameTrace::Enabled()) {
        return;
    }
    ++plan_log.frames;
    plan_log.generations += static_cast<u32>(generations);
    plan_log.interval += interval_seconds;
    plan_log.blocked += blocked_seconds;
    if (plan_log.frames < 300) {
        return;
    }
    const f32 frames = static_cast<f32>(plan_log.frames);
    LOG_INFO(Render_Vulkan,
             "Frame gen plan (300 frames): interval {:.2f} ms, of it waiting for a present frame "
             "{:.2f} ms, planned against {:.2f} ms, {:.2f} generations a frame, limit {}",
             plan_log.interval / frames * 1000.0f, plan_log.blocked / frames * 1000.0f,
             planning_interval * 1000.0f, static_cast<f32>(plan_log.generations) / frames, limit);
    plan_log = {};
}

FrameGenPlan FrameGenPacer::Plan(size_t capacity, Clock::duration blocked) {
    const f32 target_rate = static_cast<f32>(Settings::values.frame_gen_target_rate.GetValue());
    const f32 guest_interval =
        static_cast<f32>(Tegra::g_nx2_guest_frame_interval_ns.load(std::memory_order_relaxed)) *
        1.0e-9f;
    return PlanAt(Clock::now(), capacity, blocked, target_rate, guest_interval,
                  Settings::FrameGenMaxGenerations(), Settings::FrameGenGenerations());
}

FrameGenPlan FrameGenPacer::PlanAt(Clock::time_point now, size_t capacity, Clock::duration blocked,
                                   f32 target_rate, f32 guest_interval_seconds,
                                   size_t max_generations, size_t fixed_generations) {
    const size_t ceiling = std::min(capacity, max_generations);
    if (ceiling == 0) {
        Reset();
        return {};
    }

    if (!last_frame) {
        last_frame = now;
        return {};
    }

    const Clock::duration interval = now - *last_frame;
    const f32 interval_seconds = std::chrono::duration<f32>(interval).count();
    last_frame = now;

    if (interval_seconds <= 0.0f) {
        Stabilize(now);
        return {};
    }

    if (smoothed_interval > 0.0f) {
        f32 burst_threshold = BURST_CADENCE_RATIO / smoothed_interval;
        if (target_rate > 0.0f) {
            burst_threshold = std::max(burst_threshold, target_rate * BURST_TARGET_RATIO);
        }
        if (1.0f / interval_seconds > burst_threshold) {
            DeferEvaluations(interval);
            output_credit = 0.0f;
            return {};
        }
    }

    if (interval_seconds > 1.0f / MINIMUM_BASE_RATE) {
        // If frame time spiked past 100 ms while probing a higher generation limit, count it as a
        // probe failure with exponential backoff rather than re-probing every 2 seconds forever.
        if (probe_until) {
            probe_failures = std::min(probe_failures + 1, MAX_PROBE_FAILURES);
            next_probe = now + ProbeBackoff(probe_failures);
        }
        Stabilize(now);
        return {};
    }

    smoothed_interval = smoothed_interval > 0.0f
                            ? smoothed_interval +
                                  (interval_seconds - smoothed_interval) * INTERVAL_SMOOTHING
                            : interval_seconds;
    const f32 blocked_seconds = std::chrono::duration<f32>(blocked).count();
    const f32 planned = PlanningInterval(interval_seconds, blocked_seconds, target_rate,
                                         guest_interval_seconds, limit > 0);
    planning_interval = planning_interval > 0.0f
                            ? planning_interval + (planned - planning_interval) * INTERVAL_SMOOTHING
                            : planned;

    if (stable_until) {
        if (now < *stable_until) {
            LogPlans(interval_seconds, blocked_seconds, 0);
            return {};
        }
        stable_until.reset();
    }

    if (target_rate == 0.0f) {
        limit = std::min(fixed_generations, ceiling);
        output_credit = 0.0f;
        return {.generations = limit, .warm = limit > 0};
    }

    const f32 base_rate = 1.0f / planning_interval;
    const f32 real_rate = 1.0f / smoothed_interval;
    UpdateLimit(now, base_rate, real_rate, target_rate, ceiling);

    const size_t allowed = std::min(limit, ceiling);
    const f32 desired_outputs = planning_interval * target_rate;
    if (allowed == 0 || base_rate >= target_rate * TARGET_SATISFIED_RATIO ||
        desired_outputs <= 1.0f) {
        output_credit = 0.0f;
        LogPlans(interval_seconds, blocked_seconds, 0);
        return {};
    }

    output_credit += desired_outputs;
    const size_t outputs =
        std::max<size_t>(1, static_cast<size_t>(std::floor(output_credit + CREDIT_EPSILON)));
    const size_t generations = std::min(outputs - 1, allowed);

    output_credit -= static_cast<f32>(generations + 1);
    if (output_credit < 0.0f) {
        output_credit = 0.0f;
    } else if (generations == allowed && output_credit >= 1.0f) {
        output_credit = std::fmod(output_credit, 1.0f);
    }

    LogPlans(interval_seconds, blocked_seconds, generations);
    return {.generations = generations, .warm = true};
}

void FrameGenPacer::UpdateLimit(Clock::time_point now, f32 base_rate, f32 real_rate,
                                f32 target_rate, size_t ceiling) {
    limit = std::min(limit, ceiling);

    if (probe_until) {
        if (now < *probe_until) {
            return;
        }
        probe_until.reset();
        output_credit = 0.0f;

        const f32 previous_output =
            std::min(target_rate, probe_base_rate * static_cast<f32>(probe_previous_limit + 1));
        const f32 current_output =
            std::min(target_rate, base_rate * static_cast<f32>(limit + 1));

        const bool throughput_regressed =
            current_output < previous_output * PROBE_THROUGHPUT_TOLERANCE;
        const bool base_degraded = real_rate < probe_real_rate * PROBE_BASE_RETENTION;

        if (throughput_regressed || base_degraded) {
            limit = probe_previous_limit;
            probe_failures = std::min(probe_failures + 1, MAX_PROBE_FAILURES);
            next_probe = now + ProbeBackoff(probe_failures);
            deficit_since.reset();
            return;
        }

        probe_failures = 0;
        active_real_rate = real_rate;
        next_probe = now + PROBE_STEP_DELAY;
        surplus_since.reset();
        slowdown_since.reset();
    }

    // Native rate already meets the goal (e.g. main menu at ~60 FPS with Goal 60): drop to zero
    // generations immediately so frame-gen pipelines go completely idle with zero GPU cost.
    if (base_rate >= target_rate * TARGET_SATISFIED_RATIO) {
        limit = 0;
        output_credit = 0.0f;
        active_real_rate = 0.0f;
        deficit_since.reset();
        surplus_since.reset();
        slowdown_since.reset();
        return;
    }

    // Continuous GPU-overload / heavy-scene guard: if real emulation speed drops sharply below the
    // accepted probe rate while generating, step down to re-baseline without penalizing
    // probe_failures; a subsequent probe will apply backoff only if frame generation itself causes
    // the slowdown.
    if (limit > 0 && active_real_rate > 0.0f &&
        real_rate < active_real_rate * STEADY_BASE_RETENTION) {
        deficit_since.reset();
        surplus_since.reset();
        if (!slowdown_since) {
            slowdown_since = now;
        } else if (now - *slowdown_since >= PROBE_DURATION) {
            --limit;
            output_credit = 0.0f;
            active_real_rate = limit > 0 ? real_rate : 0.0f;
            slowdown_since.reset();
        }
        return;
    }
    slowdown_since.reset();

    // If a lower generation cap already satisfies the target rate, step down instead of staying
    // latched at a high multiplier from an earlier heavier scene.
    if (limit > 1 && base_rate * static_cast<f32>(limit) >= target_rate * TARGET_SATISFIED_RATIO) {
        deficit_since.reset();
        if (!surplus_since) {
            surplus_since = now;
        } else if (now - *surplus_since >= DEFICIT_DURATION) {
            --limit;
            output_credit = 0.0f;
            surplus_since.reset();
        }
        return;
    }
    surplus_since.reset();

    // Hysteresis when starting from idle (limit == 0): ignore minor VSync jitter near the target
    // rate (e.g. 56-60 FPS main menu with a 60 FPS target) so menus stay at full native speed.
    const f32 start_ratio = limit == 0 ? TARGET_START_RATIO : TARGET_SATISFIED_RATIO;
    if (base_rate * static_cast<f32>(limit + 1) >= target_rate * start_ratio || limit >= ceiling) {
        deficit_since.reset();
        return;
    }

    if (!deficit_since) {
        deficit_since = now;
        return;
    }
    if (now - *deficit_since < DEFICIT_DURATION) {
        return;
    }
    if (next_probe && now < *next_probe) {
        return;
    }

    probe_previous_limit = limit;
    probe_base_rate = base_rate;
    probe_real_rate = real_rate;
    ++limit;
    probe_until = now + PROBE_DURATION;
    deficit_since.reset();
    surplus_since.reset();
    slowdown_since.reset();
    output_credit = 0.0f;
}

void FrameGenPacer::DeferEvaluations(Clock::duration amount) {
    const auto defer = [amount](std::optional<Clock::time_point>& deadline) {
        if (deadline) {
            *deadline += amount;
        }
    };
    defer(stable_until);
    defer(probe_until);
    defer(next_probe);
    deficit_since.reset();
    surplus_since.reset();
    slowdown_since.reset();
}

void FrameGenPacer::Stabilize(Clock::time_point now) {
    stable_until = now + STABILIZATION_DURATION;
    probe_until.reset();
    deficit_since.reset();
    surplus_since.reset();
    slowdown_since.reset();
    smoothed_interval = 0.0f;
    planning_interval = 0.0f;
    output_credit = 0.0f;
    active_real_rate = 0.0f;
    limit = 0;
}

void FrameGenPacer::Reset() {
    last_frame.reset();
    stable_until.reset();
    probe_until.reset();
    next_probe.reset();
    deficit_since.reset();
    surplus_since.reset();
    slowdown_since.reset();
    smoothed_interval = 0.0f;
    planning_interval = 0.0f;
    output_credit = 0.0f;
    probe_base_rate = 0.0f;
    probe_real_rate = 0.0f;
    active_real_rate = 0.0f;
    probe_previous_limit = 0;
    limit = 0;
    probe_failures = 0;
}

} // namespace Vulkan
