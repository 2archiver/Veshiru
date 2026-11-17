// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef HAS_LSFG

#include <chrono>

#include <catch2/catch_test_macros.hpp>

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
        if (i >= 90) {
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

TEST_CASE("FrameGenPacer backs off on >100 ms probe hitches instead of looping", "[video_core]") {
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
    CHECK(probe_hitches <= 4);
    CHECK(pacer.CurrentLimit() == 0);
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

#endif // HAS_LSFG
