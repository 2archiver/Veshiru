// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef HAS_LSFG

#include <chrono>
#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/frame_pacer.h"
#include "video_core/renderer_vulkan/present/frame_gen_pacer.h"

namespace {

using Clock = std::chrono::steady_clock;

Clock::duration FromSeconds(double seconds) {
    return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

} // Anonymous namespace

TEST_CASE("FrameGenPacer stays completely idle on 60 FPS menu with Goal 60", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};
    // Realistic 60 Hz VSync jitter between 56.2 and 60.0 FPS.
    constexpr double kIntervalsMs[6] = {16.67, 17.50, 16.67, 17.80, 16.67, 17.10};

    size_t total_generations = 0;
    size_t total_warm = 0;
    for (int i = 0; i < 300; ++i) {
        now += FromSeconds(kIntervalsMs[i % 6] * 1e-3);
        const auto plan = pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
        total_generations += plan.generations;
        total_warm += plan.warm ? 1 : 0;
    }

    CHECK(total_generations == 0);
    CHECK(total_warm == 0);
    CHECK(pacer.CurrentLimit() == 0);
}

TEST_CASE("FrameGenPacer sustains 30->60 FPS without post-warmup idle cycles", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    size_t post_warmup_frames = 0;
    size_t post_warmup_generations = 0;
    size_t post_warmup_idle = 0;
    for (int i = 0; i < 240; ++i) {
        const double dt = pacer.CurrentLimit() > 0 ? (1.0 / 29.5) : (1.0 / 30.0);
        now += FromSeconds(dt);
        const auto plan = pacer.PlanAt(now, 4, {}, 60.0f, 1.0f / 30.0f, 3, 1);
        if (i >= 60) {
            ++post_warmup_frames;
            post_warmup_generations += plan.generations;
            if (plan.generations == 0) {
                ++post_warmup_idle;
            }
        }
    }

    CHECK(pacer.CurrentLimit() == 1);
    CHECK(post_warmup_generations >= post_warmup_frames - 2);
    CHECK(post_warmup_idle <= 2);
}

TEST_CASE("FrameGenPacer sustains 20->60 FPS cleanly", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    size_t post_warmup_frames = 0;
    size_t post_warmup_generations = 0;
    for (int i = 0; i < 260; ++i) {
        const double dt = pacer.CurrentLimit() > 0 ? (1.0 / 19.2) : (1.0 / 20.0);
        now += FromSeconds(dt);
        const auto plan = pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
        if (i >= 100) {
            ++post_warmup_frames;
            post_warmup_generations += plan.generations;
        }
    }

    CHECK(pacer.CurrentLimit() >= 2);
    CHECK(pacer.ProbeFailures() == 0);
    CHECK(post_warmup_generations >= post_warmup_frames * 2 - 4);
}

TEST_CASE("FrameGenPacer preserves state across isolated 105 ms hitch at 18 FPS", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    // Ramp up to steady generation at 18 FPS targeting 60 FPS (limit >= 2).
    for (int i = 0; i < 140; ++i) {
        now += FromSeconds(1.0 / 18.0);
        (void)pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
    }
    const size_t limit_before = pacer.CurrentLimit();
    REQUIRE(limit_before >= 2);

    // Inject a single 105 ms shader-compile / asset-stream hitch.
    now += FromSeconds(0.105);
    const auto hitch_plan = pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
    CHECK(hitch_plan.generations == 0);
    CHECK(hitch_plan.warm);
    CHECK(hitch_plan.hitch_skipped);
    CHECK(pacer.CurrentLimit() == limit_before);
    CHECK(pacer.ProbeFailures() == 0);

    // Immediately on the very next normal 18 FPS frame, generation must resume without resetting.
    now += FromSeconds(1.0 / 18.0);
    const auto resumed_plan = pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
    CHECK(resumed_plan.generations >= 2);
    CHECK(resumed_plan.warm);
    CHECK_FALSE(resumed_plan.hitch_skipped);
    CHECK(pacer.CurrentLimit() == limit_before);
}

TEST_CASE("FrameGenPacer tolerates 18 FPS with +-20 percent frame-time variance", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};
    // +-20% frame-time jitter around 18 FPS (55.5 ms -> 44.4 ms .. 66.7 ms).
    constexpr double kMultipliers[8] = {0.82, 1.18, 0.90, 1.15, 0.85, 1.20, 0.95, 1.05};

    size_t post_warmup_frames = 0;
    size_t post_warmup_generations = 0;
    for (int i = 0; i < 300; ++i) {
        const double base_dt = 1.0 / 18.0;
        now += FromSeconds(base_dt * kMultipliers[i % 8]);
        const auto plan = pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
        if (i >= 120) {
            ++post_warmup_frames;
            post_warmup_generations += plan.generations;
        }
    }

    CHECK(pacer.ProbeFailures() == 0);
    CHECK(pacer.CurrentLimit() >= 2);
    CHECK(post_warmup_generations >= post_warmup_frames * 2);
}

TEST_CASE("FrameGenPacer survives isolated hitch during probe without poisoning backoff",
          "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    bool injected_hitch = false;
    for (int i = 0; i < 160; ++i) {
        double dt = 1.0 / 24.0;
        if (!injected_hitch && pacer.CurrentLimit() == 1) {
            // Single isolated 130 ms hitch right as probe begins at limit == 1.
            dt = 0.130;
            injected_hitch = true;
        }
        now += FromSeconds(dt);
        const auto plan = pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
        if (dt == 0.130) {
            CHECK(plan.generations == 0);
            CHECK(plan.warm);
            CHECK(plan.hitch_skipped);
        }
    }

    CHECK(injected_hitch);
    CHECK(pacer.ProbeFailures() == 0);
    CHECK(pacer.CurrentLimit() >= 1);
}

TEST_CASE("FrameGenPacer recovers quickly after >=400 ms loading screen and clears backoff",
          "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    // First trigger a probe failure so backoff is armed.
    for (int i = 0; i < 120; ++i) {
        const bool active = pacer.CurrentLimit() > 0;
        now += FromSeconds(active ? (1.0 / 12.0) : (1.0 / 30.0));
        (void)pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
    }
    REQUIRE(pacer.ProbeFailures() >= 1);
    REQUIRE(pacer.CurrentLimit() == 0);

    // Loading screen stall (>= 400 ms) should invalidate optical-flow warm state AND clear backoff.
    now += FromSeconds(0.600);
    const auto load_plan = pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
    CHECK(load_plan.generations == 0);
    CHECK_FALSE(load_plan.warm);
    CHECK_FALSE(load_plan.hitch_skipped);
    CHECK(pacer.ProbeFailures() == 0);

    // Gameplay resumes cleanly at 30 FPS: should recover to limit == 1 within ~1.5 seconds (45
    // frames) rather than waiting through a stale 15-60 s backoff.
    size_t frames_to_recover = 0;
    for (int i = 0; i < 60; ++i) {
        now += FromSeconds(1.0 / 30.0);
        const auto plan = pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
        if (plan.generations > 0 && frames_to_recover == 0) {
            frames_to_recover = static_cast<size_t>(i + 1);
        }
    }

    CHECK(frames_to_recover > 0);
    CHECK(frames_to_recover <= 45);
    CHECK(pacer.CurrentLimit() == 1);
}

TEST_CASE("FrameGenPacer rejects probes that collapse base FPS to chase 60 FPS", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    size_t frames_at_limit0 = 0;
    size_t frames_at_limit1 = 0;
    for (int i = 0; i < 600; ++i) {
        const bool active = pacer.CurrentLimit() > 0;
        now += FromSeconds(active ? (1.0 / 18.0) : (1.0 / 30.0));
        (void)pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
        if (active) {
            ++frames_at_limit1;
        } else {
            ++frames_at_limit0;
        }
    }

    CHECK(pacer.ProbeFailures() >= 1);
    CHECK(pacer.CurrentLimit() == 0);
    CHECK(frames_at_limit0 > frames_at_limit1 * 4);
}

TEST_CASE("FrameGenPacer backs off on persistent >100 ms probe overload instead of looping",
          "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    size_t probe_hitches = 0;
    for (int i = 0; i < 900; ++i) {
        const bool active = pacer.CurrentLimit() > 0;
        if (active) {
            ++probe_hitches;
        }
        now += FromSeconds(active ? 0.125 : (1.0 / 30.0));
        (void)pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
    }

    CHECK(pacer.ProbeFailures() >= 2);
    CHECK(probe_hitches <= 10);
    CHECK(pacer.CurrentLimit() == 0);
}

TEST_CASE("FrameGenPacer tracks per-level probe failure without dropping a healthy lower level",
          "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    // At 20 FPS base: limit=1 runs fine (~19.5 FPS), but limit=2 overloads the GPU (drops to 10
    // FPS). Pacer must stay stably generating at limit=1 while backing off only level 2.
    size_t frames_at_limit1 = 0;
    size_t frames_at_limit0_after_ramp = 0;
    for (int i = 0; i < 400; ++i) {
        const size_t lim = pacer.CurrentLimit();
        const double fps = lim >= 2 ? 10.0 : (lim == 1 ? 19.5 : 20.0);
        now += FromSeconds(1.0 / fps);
        (void)pacer.PlanAt(now, 6, {}, 60.0f, 0.0f, 3, 1);
        if (i >= 60) {
            if (pacer.CurrentLimit() == 1) {
                ++frames_at_limit1;
            } else if (pacer.CurrentLimit() == 0) {
                ++frames_at_limit0_after_ramp;
            }
        }
    }

    CHECK(pacer.CurrentLimit() == 1);
    CHECK(frames_at_limit0_after_ramp == 0);
    CHECK(frames_at_limit1 > 300);
    CHECK(pacer.LevelFailures(2) >= 1);
    CHECK(pacer.LevelFailures(1) == 0);
}

TEST_CASE("FrameGenPacer unpins 60 FPS menu from 60 Hz FIFO blocking", "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    // Ramp to limit == 1 at 30 FPS first.
    for (int i = 0; i < 90; ++i) {
        now += FromSeconds(1.0 / 30.0);
        (void)pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
    }
    REQUIRE(pacer.CurrentLimit() == 1);

    // Menu returns to 60 FPS natively, with 60 Hz FIFO blocking present queue for 25 ms/frame.
    for (int i = 0; i < 60; ++i) {
        const bool active = pacer.CurrentLimit() > 0;
        now += FromSeconds(active ? (1.0 / 30.0) : (1.0 / 60.0));
        const auto blocked = active ? FromSeconds(0.025) : Clock::duration{};
        (void)pacer.PlanAt(now, 4, blocked, 60.0f, 0.0f, 3, 1);
    }

    CHECK(pacer.CurrentLimit() == 0);
}

TEST_CASE("FrameGenPacer keeps GPU-bound 30 FPS stable and handles scene drops cleanly",
          "[video_core]") {
    Vulkan::FrameGenPacer pacer;
    auto now = Clock::time_point{};

    // 3s at 45 FPS -> limit == 1.
    for (int i = 0; i < 135; ++i) {
        now += FromSeconds(1.0 / 45.0);
        (void)pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
    }
    REQUIRE(pacer.CurrentLimit() == 1);

    // Legitimate scene transition to 30 FPS should re-baseline cleanly without backoff penalties.
    for (int i = 0; i < 120; ++i) {
        now += FromSeconds(1.0 / 30.0);
        (void)pacer.PlanAt(now, 4, {}, 60.0f, 0.0f, 3, 1);
    }

    CHECK(pacer.ProbeFailures() == 0);
    CHECK(pacer.CurrentLimit() == 1);
}

TEST_CASE("GenerationsWithoutWaiting preserves real game frame and trims generated frames",
          "[video_core]") {
    // 0 or 1 free frames: must keep the free slot for the next real game frame (0 generations).
    CHECK(Vulkan::GenerationsWithoutWaiting(1, 0) == 0);
    CHECK(Vulkan::GenerationsWithoutWaiting(1, 1) == 0);
    CHECK(Vulkan::GenerationsWithoutWaiting(3, 1) == 0);

    // 2 free frames: 1 generation + 1 spare for next real frame.
    CHECK(Vulkan::GenerationsWithoutWaiting(1, 2) == 1);
    CHECK(Vulkan::GenerationsWithoutWaiting(3, 2) == 1);

    // 4 free frames: up to 3 generations without stalling the next real frame.
    CHECK(Vulkan::GenerationsWithoutWaiting(3, 4) == 3);
    CHECK(Vulkan::GenerationsWithoutWaiting(2, 4) == 2);
}

TEST_CASE("FramePacer counts game frames instead of generated presents across refresh rates",
          "[video_core]") {
    using namespace std::chrono;

    // 60 FPS game limiter with 2x FG (game_frame_presents == 2) must pace at 120 presents/s
    // (8.333 ms per present), NEVER 60 presents/s (30 game FPS).
    {
        Vulkan::FramePacer pacer;
        const auto t0 = Clock::time_point{milliseconds(1000)};
        auto now = t0;

        for (int p = 1; p <= 12; ++p) {
            const auto target = pacer.Next(now, 60.0, 2);
            REQUIRE(target.has_value());
            const double elapsed_ms = duration<double, std::milli>(*target - t0).count();
            const double expected_ms = (static_cast<double>(p) * 0.5) * (1000.0 / 60.0);
            CHECK(std::abs(elapsed_ms - expected_ms) < 0.05);
            now = *target;
        }
    }

    // 30 FPS game limiter with 3x FG (game_frame_presents == 3) at 90 Hz output:
    // 3 presents = 1 game frame = 33.333 ms (11.111 ms per present).
    {
        Vulkan::FramePacer pacer;
        const auto t0 = Clock::time_point{milliseconds(5000)};
        auto now = t0;

        for (int p = 1; p <= 9; ++p) {
            const auto target = pacer.Next(now, 30.0, 3);
            REQUIRE(target.has_value());
            const double elapsed_ms = duration<double, std::milli>(*target - t0).count();
            const double expected_ms = (static_cast<double>(p) / 3.0) * (1000.0 / 30.0);
            CHECK(std::abs(elapsed_ms - expected_ms) < 0.05);
            now = *target;
        }
    }

    // Late present returns nullopt immediately without sleeping.
    {
        Vulkan::FramePacer pacer;
        auto now = Clock::time_point{milliseconds(1000)};
        REQUIRE(pacer.Next(now, 60.0, 2).has_value());
        now += milliseconds(50);
        CHECK_FALSE(pacer.Next(now, 60.0, 2).has_value());
    }
}

#endif // HAS_LSFG
