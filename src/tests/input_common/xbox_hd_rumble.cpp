// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <initializer_list>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "input_common/helpers/xbox_hd_rumble.h"

namespace {

using InputCommon::HdRumbleSide;
using InputCommon::MapHdRumbleToXboxMotors;
using InputCommon::MixXboxSides;
using InputCommon::XboxMotors;
using InputCommon::XboxRumbleParams;
using InputCommon::XboxRumbleScheduler;

constexpr f32 LowHz = 160.0f;
constexpr f32 HighHz = 320.0f;

constexpr XboxRumbleScheduler::Config QuietConfig{
    .refresh_interval_us = 400'000,
    .min_interval_us = 8'000,
    .change_threshold = 656,
    .hit_accent = 0.0f,
};

constexpr XboxMotors Half{.left_motor = 30000, .right_motor = 20000};

} // Anonymous namespace

TEST_CASE("HD rumble to Xbox motors: silence stays silent", "[input_common][xbox_hd_rumble]") {
    const auto motors = MapHdRumbleToXboxMotors(0.0f, LowHz, 0.0f, HighHz, {});
    REQUIRE(motors.left_motor == 0);
    REQUIRE(motors.right_motor == 0);

    const auto mixed = MixXboxSides({}, {}, {.trigger_mix = 1.0f});
    REQUIRE(mixed.IsZero());
}

TEST_CASE("HD rumble to Xbox motors: each band drives its own motor",
          "[input_common][xbox_hd_rumble]") {
    // No detail mix: the bands stay apart
    const XboxRumbleParams apart{.detail_mix = 0.0f};
    const auto thud = MapHdRumbleToXboxMotors(1.0f, LowHz, 0.0f, HighHz, apart);
    REQUIRE(thud.left_motor == 65535);
    REQUIRE(thud.right_motor == 0);

    const auto buzz = MapHdRumbleToXboxMotors(0.0f, LowHz, 1.0f, HighHz, apart);
    REQUIRE(buzz.left_motor == 0);
    REQUIRE(buzz.right_motor == 65535);

    // The default mix puts half of the high band on the heavy motor as well
    const auto detailed = MapHdRumbleToXboxMotors(0.0f, LowHz, 1.0f, HighHz, {});
    REQUIRE(detailed.right_motor == 65535);
    REQUIRE(detailed.left_motor > 30000);
    REQUIRE(detailed.left_motor < 35000);
}

TEST_CASE("HD rumble to Xbox motors: the output is clamped", "[input_common][xbox_hd_rumble]") {
    const f32 infinity = std::numeric_limits<f32>::infinity();
    const std::array<XboxRumbleParams, 3> loud{{
        {.gain = 10.0f, .strength = 10.0f},
        {.gain = infinity, .strength = infinity},
        {.gain = 1.0f, .low_boost = 5.0f, .strength = 1000.0f, .detail_mix = 9.0f},
    }};
    for (const auto& params : loud) {
        const auto motors = MapHdRumbleToXboxMotors(10.0f, LowHz, 10.0f, HighHz, params);
        REQUIRE(motors.left_motor == 65535);
        REQUIRE(motors.right_motor == 65535);
    }
    const auto endless = MapHdRumbleToXboxMotors(infinity, LowHz, infinity, HighHz, {});
    REQUIRE(endless.left_motor == 65535);
    REQUIRE(endless.right_motor == 65535);
}

TEST_CASE("HD rumble to Xbox motors: monotonic in amplitude and strength",
          "[input_common][xbox_hd_rumble]") {
    const std::array<XboxRumbleParams, 3> all_params{{
        {},
        {.gain = 2.5f, .low_boost = 0.7f, .strength = 0.6f, .detail_mix = 1.0f},
        {.gain = 0.25f, .low_boost = 1.0f, .strength = 3.0f, .detail_mix = 0.0f},
    }};
    for (const auto& params : all_params) {
        XboxMotors previous_low{};
        XboxMotors previous_high{};
        for (int step = 0; step <= 2000; ++step) {
            const f32 amplitude = static_cast<f32>(step) / 2000.0f;
            const auto by_low = MapHdRumbleToXboxMotors(amplitude, LowHz, 0.3f, HighHz, params);
            REQUIRE(by_low.left_motor >= previous_low.left_motor);
            REQUIRE(by_low.right_motor >= previous_low.right_motor);
            previous_low = {.left_motor = by_low.left_motor, .right_motor = by_low.right_motor};

            const auto by_high = MapHdRumbleToXboxMotors(0.3f, LowHz, amplitude, HighHz, params);
            REQUIRE(by_high.left_motor >= previous_high.left_motor);
            REQUIRE(by_high.right_motor >= previous_high.right_motor);
            previous_high = {.left_motor = by_high.left_motor,
                             .right_motor = by_high.right_motor};
        }
    }

    XboxMotors previous{};
    for (int step = 0; step <= 300; ++step) {
        const XboxRumbleParams params{.strength = static_cast<f32>(step) / 100.0f};
        const auto motors = MapHdRumbleToXboxMotors(0.4f, LowHz, 0.2f, HighHz, params);
        REQUIRE(motors.left_motor >= previous.left_motor);
        REQUIRE(motors.right_motor >= previous.right_motor);
        previous = {.left_motor = motors.left_motor, .right_motor = motors.right_motor};
    }
}

TEST_CASE("HD rumble to Xbox motors: strength 0 is silence", "[input_common][xbox_hd_rumble]") {
    const XboxRumbleParams off{.gain = 10.0f, .strength = 0.0f, .trigger_mix = 1.0f};
    const auto motors = MapHdRumbleToXboxMotors(1.0f, LowHz, 1.0f, HighHz, off);
    REQUIRE(motors.left_motor == 0);
    REQUIRE(motors.right_motor == 0);

    const HdRumbleSide full{1.0f, LowHz, 1.0f, HighHz};
    REQUIRE(MixXboxSides(full, full, off).IsZero());

    const XboxRumbleParams negative{.strength = -3.0f};
    const auto below = MapHdRumbleToXboxMotors(1.0f, LowHz, 1.0f, HighHz, negative);
    REQUIRE(below.left_motor == 0);
    REQUIRE(below.right_motor == 0);
}

TEST_CASE("HD rumble to Xbox motors: bad frequencies and amplitudes are safe",
          "[input_common][xbox_hd_rumble]") {
    const f32 nan = std::numeric_limits<f32>::quiet_NaN();
    const f32 infinity = std::numeric_limits<f32>::infinity();
    const auto nominal = MapHdRumbleToXboxMotors(0.5f, LowHz, 0.5f, HighHz, {});

    // A frequency that says nothing is treated as the nominal one
    for (const f32 frequency : {nan, infinity, -infinity, 0.0f, -40.0f}) {
        const auto motors = MapHdRumbleToXboxMotors(0.5f, frequency, 0.5f, frequency, {});
        REQUIRE(motors == nominal);
    }

    // A frequency far above the band only weakens it a little
    const auto high = MapHdRumbleToXboxMotors(0.5f, 1.0e9f, 0.5f, 1.0e9f, {});
    REQUIRE(high.left_motor <= nominal.left_motor);
    REQUIRE(high.right_motor <= nominal.right_motor);
    REQUIRE(high.left_motor >= nominal.left_motor * 0.84f);
    REQUIRE(high.right_motor >= nominal.right_motor * 0.84f);

    // An amplitude or a setting that is not a number is silence, not a crash or a full blast
    const auto no_amplitude = MapHdRumbleToXboxMotors(nan, LowHz, nan, HighHz, {});
    REQUIRE(no_amplitude.left_motor == 0);
    REQUIRE(no_amplitude.right_motor == 0);
    const auto negative = MapHdRumbleToXboxMotors(-1.0f, LowHz, -infinity, HighHz, {});
    REQUIRE(negative.left_motor == 0);
    REQUIRE(negative.right_motor == 0);
    const XboxRumbleParams broken{.gain = nan, .low_boost = nan, .strength = nan,
                                  .detail_mix = nan, .trigger_mix = nan};
    const HdRumbleSide full{1.0f, LowHz, 1.0f, HighHz};
    REQUIRE(MixXboxSides(full, full, broken).IsZero());
}

TEST_CASE("HD rumble to Xbox motors: both halves share the pad",
          "[input_common][xbox_hd_rumble]") {
    const XboxRumbleParams params{.detail_mix = 0.0f, .trigger_mix = 0.5f};
    const HdRumbleSide left{1.0f, LowHz, 0.0f, HighHz};
    const HdRumbleSide right{0.0f, LowHz, 1.0f, HighHz};

    const auto motors = MixXboxSides(left, right, params);
    REQUIRE(motors.left_motor == 65535);
    REQUIRE(motors.right_motor == 65535);
    // Only the right half has a high band, so only the right trigger moves
    REQUIRE(motors.left_trigger == 0);
    REQUIRE(motors.right_trigger > 32000);
    REQUIRE(motors.right_trigger < 33500);

    // Triggers stay off unless they are mixed in
    const auto no_triggers = MixXboxSides(right, right, {});
    REQUIRE(no_triggers.left_trigger == 0);
    REQUIRE(no_triggers.right_trigger == 0);
}

TEST_CASE("Xbox rumble scheduler: identical input is one write per refresh interval",
          "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler scheduler;
    int writes = 0;
    int refreshes = 0;
    // A guest repeating one value every 5 ms for two seconds, with a 1 ms tick
    for (s64 now_us = 0; now_us < 2'000'000; now_us += 1'000) {
        if (now_us % 5'000 == 0) {
            if (const auto send = scheduler.Offer(now_us, Half, QuietConfig)) {
                REQUIRE(send->motors == Half);
                REQUIRE_FALSE(send->refresh);
                ++writes;
            }
        }
        if (const auto send = scheduler.Tick(now_us, QuietConfig)) {
            REQUIRE(send->motors == Half);
            REQUIRE(send->refresh);
            ++refreshes;
        }
    }
    REQUIRE(writes == 1);
    // 2 s of a 400 ms refresh: at 400, 800, 1200 and 1600 ms
    REQUIRE(refreshes == 4);
}

TEST_CASE("Xbox rumble scheduler: a change is written at once", "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler scheduler;
    REQUIRE(scheduler.Offer(0, Half, QuietConfig).has_value());

    const XboxMotors stronger{.left_motor = 50000, .right_motor = 20000};
    const auto send = scheduler.Offer(20'000, stronger, QuietConfig);
    REQUIRE(send.has_value());
    REQUIRE(send->motors == stronger);
    REQUIRE_FALSE(send->refresh);
    REQUIRE(scheduler.LastSent() == stronger);
}

TEST_CASE("Xbox rumble scheduler: small and rapid changes are coalesced",
          "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler scheduler;
    REQUIRE(scheduler.Offer(0, Half, QuietConfig).has_value());

    // Below the threshold: nothing until the refresh, which then carries the latest value
    const XboxMotors nudged{.left_motor = 30100, .right_motor = 20000};
    REQUIRE_FALSE(scheduler.Offer(50'000, nudged, QuietConfig).has_value());
    REQUIRE_FALSE(scheduler.Tick(100'000, QuietConfig).has_value());
    const auto refreshed = scheduler.Tick(400'000, QuietConfig);
    REQUIRE(refreshed.has_value());
    REQUIRE(refreshed->motors == nudged);

    // Above the threshold but inside the minimum interval: held for Tick, latest value wins
    const XboxMotors first{.left_motor = 40000, .right_motor = 20000};
    const XboxMotors second{.left_motor = 45000, .right_motor = 20000};
    REQUIRE_FALSE(scheduler.Offer(402'000, first, QuietConfig).has_value());
    REQUIRE_FALSE(scheduler.Offer(404'000, second, QuietConfig).has_value());
    REQUIRE_FALSE(scheduler.Tick(405'000, QuietConfig).has_value());
    const auto held = scheduler.Tick(408'000, QuietConfig);
    REQUIRE(held.has_value());
    REQUIRE(held->motors == second);
    REQUIRE_FALSE(scheduler.Tick(409'000, QuietConfig).has_value());
}

TEST_CASE("Xbox rumble scheduler: a stop is a zero write", "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler scheduler;
    REQUIRE(scheduler.Offer(0, Half, QuietConfig).has_value());

    // From the guest, even inside the minimum interval
    const auto stopped = scheduler.Offer(1'000, {}, QuietConfig);
    REQUIRE(stopped.has_value());
    REQUIRE(stopped->motors.IsZero());
    // Already stopped: no second write, and nothing to refresh
    REQUIRE_FALSE(scheduler.Offer(2'000, {}, QuietConfig).has_value());
    REQUIRE_FALSE(scheduler.Tick(5'000'000, QuietConfig).has_value());

    // From the emulator (pause, stop, disconnect, setting off): always zeros
    REQUIRE(scheduler.Offer(6'000'000, Half, QuietConfig).has_value());
    REQUIRE(scheduler.Stop(6'001'000).motors.IsZero());
    REQUIRE(scheduler.Stop(6'002'000).motors.IsZero());
    REQUIRE_FALSE(scheduler.Tick(9'000'000, QuietConfig).has_value());

    // And the next vibration starts at once
    REQUIRE(scheduler.Offer(9'001'000, Half, QuietConfig).has_value());
}

TEST_CASE("Xbox rumble scheduler: a refused write is tried again",
          "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler scheduler;
    REQUIRE(scheduler.Offer(0, Half, QuietConfig).has_value());
    scheduler.MarkFailed();
    REQUIRE_FALSE(scheduler.Tick(1'000, QuietConfig).has_value());
    const auto retry = scheduler.Tick(8'000, QuietConfig);
    REQUIRE(retry.has_value());
    REQUIRE(retry->motors == Half);

    // A refused stop too
    REQUIRE(scheduler.Offer(20'000, {}, QuietConfig).has_value());
    scheduler.MarkFailed();
    const auto stop_again = scheduler.Tick(30'000, QuietConfig);
    REQUIRE(stop_again.has_value());
    REQUIRE(stop_again->motors.IsZero());
}

TEST_CASE("Xbox rumble scheduler: a hit overshoots, then settles",
          "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler::Config config = QuietConfig;
    config.hit_accent = 0.5f;

    XboxRumbleScheduler scheduler;
    const auto burst = scheduler.Offer(0, Half, config);
    REQUIRE(burst.has_value());
    REQUIRE(burst->motors.left_motor > Half.left_motor);
    REQUIRE(burst->motors.right_motor > Half.right_motor);

    // During the burst nothing else is written, and the latest target is kept
    const XboxMotors softer{.left_motor = 25000, .right_motor = 15000};
    REQUIRE_FALSE(scheduler.Offer(10'000, softer, config).has_value());
    REQUIRE_FALSE(scheduler.Tick(20'000, config).has_value());
    const auto settled = scheduler.Tick(25'000, config);
    REQUIRE(settled.has_value());
    REQUIRE(settled->motors == softer);

    // A stop ends a burst at once
    REQUIRE(scheduler.Offer(100'000, {}, config).has_value());
    REQUIRE(scheduler.Offer(200'000, Half, config).has_value());
    const auto stopped = scheduler.Offer(201'000, {}, config);
    REQUIRE(stopped.has_value());
    REQUIRE(stopped->motors.IsZero());
    REQUIRE_FALSE(scheduler.Tick(300'000, config).has_value());
}

TEST_CASE("Xbox rumble scheduler: a forced write is never held back",
          "[input_common][xbox_hd_rumble]") {
    XboxRumbleScheduler scheduler;
    const auto first = scheduler.Force(0, Half);
    REQUIRE(first.motors == Half);
    REQUIRE_FALSE(first.refresh);

    // The same value again, inside the minimum interval: still written, since the source that
    // paces itself asked for it
    const auto again = scheduler.Force(1'000, Half);
    REQUIRE(again.motors == Half);
    REQUIRE_FALSE(again.refresh);
    REQUIRE(scheduler.LastSent() == Half);

    // It is refreshed like any other playing vibration, and a stop still zeroes it
    REQUIRE_FALSE(scheduler.Tick(100'000, QuietConfig).has_value());
    const auto refresh = scheduler.Tick(1'000 + QuietConfig.refresh_interval_us, QuietConfig);
    REQUIRE(refresh.has_value());
    REQUIRE(refresh->refresh);
    REQUIRE(refresh->motors == Half);
    REQUIRE(scheduler.Stop(900'000).motors.IsZero());
    REQUIRE_FALSE(scheduler.Tick(5'000'000, QuietConfig).has_value());
}
