// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The pacing of an NVN HLE present: Frame Rate Cap, speed limit and "Sync to framerate of video
// playback" (core/hle/nvn/present_pacing.h).

#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/nvn/present_pacing.h"
#include "video_core/renderer_vulkan/nvn/nvn_submit_policy.h"

namespace {

using Core::NvnHle::PaceMode;
using Core::NvnHle::PlanPresentPacing;
using Core::NvnHle::PresentPaceInputs;
using std::chrono::nanoseconds;

constexpr s64 FRAME_60 = 16'666'666;

} // namespace

TEST_CASE("NVN present pacing: Auto cap leaves the pacing to the speed limiter", "[nvn][present]") {
    const auto plan = PlanPresentPacing({.present_interval = 1, .frame_rate_cap = 0});
    REQUIRE(plan.mode == PaceMode::SpeedLimiter);
}

TEST_CASE("NVN present pacing: a fixed cap holds the present at the cap's period",
          "[nvn][present]") {
    SECTION("30 fps cap on a 60 Hz game: 33.3 ms") {
        const auto plan = PlanPresentPacing({.present_interval = 1, .frame_rate_cap = 30});
        REQUIRE(plan.mode == PaceMode::Interval);
        REQUIRE(plan.interval == nanoseconds{33'333'333});
    }
    SECTION("a cap above the display rate never speeds the game up") {
        const auto plan = PlanPresentPacing({.present_interval = 1, .frame_rate_cap = 120});
        REQUIRE(plan.mode == PaceMode::Interval);
        REQUIRE(plan.interval == nanoseconds{FRAME_60});
    }
    SECTION("the swap interval's period wins over a faster cap") {
        const auto plan = PlanPresentPacing({.present_interval = 2, .frame_rate_cap = 60});
        REQUIRE(plan.interval == nanoseconds{33'333'333});
    }
    SECTION("the swap interval is clamped to 1..4") {
        REQUIRE(PlanPresentPacing({.present_interval = 0, .frame_rate_cap = 60}).interval ==
                nanoseconds{FRAME_60});
        REQUIRE(PlanPresentPacing({.present_interval = 9, .frame_rate_cap = 60}).interval ==
                PlanPresentPacing({.present_interval = 4, .frame_rate_cap = 60}).interval);
    }
    SECTION("the speed limit scales the period") {
        const auto plan = PlanPresentPacing(
            {.present_interval = 1, .frame_rate_cap = 30, .speed_limit_percent = 200});
        REQUIRE(plan.interval == nanoseconds{16'666'666});
    }
    SECTION("no speed limit, or a limit of 0: only the display paces") {
        REQUIRE(PlanPresentPacing({.frame_rate_cap = 30, .use_speed_limit = false}).mode ==
                PaceMode::None);
        REQUIRE(PlanPresentPacing({.frame_rate_cap = 30, .speed_limit_percent = 0}).mode ==
                PaceMode::None);
    }
}

TEST_CASE("NVN present pacing: video playback runs at its intended rate", "[nvn][present]") {
    SECTION("under a turbo limit the video keeps 100%") {
        const auto video = PlanPresentPacing({.present_interval = 2,
                                              .frame_rate_cap = 0,
                                              .use_speed_limit = true,
                                              .speed_limit_percent = 300,
                                              .video_framerate = true});
        REQUIRE(video.mode == PaceMode::Interval);
        REQUIRE(video.interval == nanoseconds{33'333'333});
        const auto other = PlanPresentPacing({.present_interval = 2,
                                              .frame_rate_cap = 0,
                                              .use_speed_limit = true,
                                              .speed_limit_percent = 300,
                                              .video_framerate = false});
        REQUIRE(other.mode == PaceMode::SpeedLimiter);
    }
    SECTION("under a slow-motion limit the video keeps 100%") {
        const auto video = PlanPresentPacing({.present_interval = 1,
                                              .frame_rate_cap = 0,
                                              .use_speed_limit = true,
                                              .speed_limit_percent = 50,
                                              .video_framerate = true});
        REQUIRE(video.interval == nanoseconds{FRAME_60});
    }
    SECTION("it applies with the speed limit off, as the vsync event does") {
        const auto video = PlanPresentPacing({.present_interval = 1,
                                              .frame_rate_cap = 0,
                                              .use_speed_limit = false,
                                              .video_framerate = true});
        REQUIRE(video.mode == PaceMode::Interval);
        REQUIRE(video.interval == nanoseconds{FRAME_60});
    }
    SECTION("the Frame Rate Cap still applies during a video") {
        const auto video = PlanPresentPacing({.present_interval = 1,
                                              .frame_rate_cap = 30,
                                              .speed_limit_percent = 300,
                                              .video_framerate = true});
        REQUIRE(video.interval == nanoseconds{33'333'333});
    }
}

TEST_CASE("NVN present pacing: the speed limit scales the display's slot period", "[nvn][present]") {
    using Core::NvnHle::ScaledDisplayPeriodNs;
    constexpr u64 PERIOD = 16'666'667;
    SECTION("100% leaves the period alone (defaults unchanged)") {
        REQUIRE(ScaledDisplayPeriodNs(PERIOD, {}) == PERIOD);
    }
    SECTION("a slow-motion limit lengthens it, a turbo limit shortens it") {
        REQUIRE(ScaledDisplayPeriodNs(PERIOD, {.speed_limit_percent = 50}) == PERIOD * 2);
        REQUIRE(ScaledDisplayPeriodNs(PERIOD, {.speed_limit_percent = 200}) == PERIOD / 2);
        REQUIRE(ScaledDisplayPeriodNs(2 * PERIOD, {.speed_limit_percent = 150}) ==
                2 * PERIOD * 100 / 150);
    }
    SECTION("no speed limit, or a limit of 0: the period is unchanged") {
        REQUIRE(ScaledDisplayPeriodNs(PERIOD, {.use_speed_limit = false,
                                               .speed_limit_percent = 50}) == PERIOD);
        REQUIRE(ScaledDisplayPeriodNs(PERIOD, {.speed_limit_percent = 0}) == PERIOD);
    }
    SECTION("a period of 0 (no present interval) stays 0") {
        REQUIRE(ScaledDisplayPeriodNs(0, {.speed_limit_percent = 50}) == 0);
    }
    SECTION("a video with Sync to framerate of video playback runs at 100%") {
        REQUIRE(ScaledDisplayPeriodNs(PERIOD, {.speed_limit_percent = 300,
                                               .video_framerate = true}) == PERIOD);
    }
}

TEST_CASE("NVN present depth limit: a present waits for the one n frames before it",
          "[nvn][present]") {
    using Core::NvnHle::PresentDepthLimit;
    SECTION("n = 1: the previous present") {
        PresentDepthLimit limit{1};
        REQUIRE(limit.OnPresent(10) == 0);
        REQUIRE(limit.OnPresent(25) == 10);
        REQUIRE(limit.OnPresent(31) == 25);
    }
    SECTION("n = 2: two frames may stay queued") {
        PresentDepthLimit limit{2};
        REQUIRE(limit.OnPresent(10) == 0);
        REQUIRE(limit.OnPresent(20) == 0);
        REQUIRE(limit.OnPresent(30) == 10);
        REQUIRE(limit.OnPresent(40) == 20);
    }
    SECTION("n = 0: this present itself") {
        PresentDepthLimit limit{0};
        REQUIRE(limit.OnPresent(7) == 7);
        REQUIRE(limit.OnPresent(9) == 9);
    }
    SECTION("the depth is clamped to the ring, and the ring wraps") {
        PresentDepthLimit limit{100};
        REQUIRE(limit.Limit() == PresentDepthLimit::MaxDepth);
        for (u64 i = 1; i <= PresentDepthLimit::MaxDepth; ++i) {
            REQUIRE(limit.OnPresent(i * 10) == 0);
        }
        for (u64 i = PresentDepthLimit::MaxDepth + 1; i <= 40; ++i) {
            REQUIRE(limit.OnPresent(i * 10) == (i - PresentDepthLimit::MaxDepth) * 10);
        }
    }
}

TEST_CASE("MKW adaptive present only shortens the two-vblank gameplay interval",
          "[nvn][present][mkw]") {
    using Core::NvnHle::AdaptivePresentInterval;
    REQUIRE(AdaptivePresentInterval(2, true, false) == 1);
    REQUIRE(AdaptivePresentInterval(2, false, false) == 2);
    REQUIRE(AdaptivePresentInterval(2, true, true) == 2);
    REQUIRE(AdaptivePresentInterval(0, true, false) == 0);
    REQUIRE(AdaptivePresentInterval(1, true, false) == 1);
    REQUIRE(AdaptivePresentInterval(3, true, false) == 3);
}

TEST_CASE("MKW early submit keeps bounded overlap through long passes and shadows", "[nvn][mkw]") {
    using Vulkan::Nvn::MkwEarlySubmitDue;
    CHECK_FALSE(MkwEarlySubmitDue(0, 299, 300, 4000, false, false));
    CHECK(MkwEarlySubmitDue(0, 300, 300, 4000, false, false));
    CHECK_FALSE(MkwEarlySubmitDue(1, 1499, 1500, 4000, false, false));
    CHECK(MkwEarlySubmitDue(1, 1500, 1500, 4000, false, false));
    CHECK_FALSE(MkwEarlySubmitDue(0, 3999, 300, 4000, true, false));
    CHECK(MkwEarlySubmitDue(0, 4000, 300, 4000, true, false));
    CHECK_FALSE(MkwEarlySubmitDue(0, 3999, 300, 4000, false, true));
    CHECK(MkwEarlySubmitDue(0, 4000, 300, 4000, false, true));
    CHECK_FALSE(MkwEarlySubmitDue(0, 5000, 300, 0, true, true));
    CHECK_FALSE(MkwEarlySubmitDue(2, 100000, 300, 4000, false, false));
    CHECK_FALSE(MkwEarlySubmitDue(2, 100000, 300, 4000, true, true));
}

TEST_CASE("NVN present depth limit: EDEN_NX2_MAX_FRAMES_IN_FLIGHT parsing", "[nvn][present]") {
    using Core::NvnHle::ParseMaxFramesInFlight;
    REQUIRE(ParseMaxFramesInFlight(nullptr) == 1u);
    REQUIRE_FALSE(ParseMaxFramesInFlight("0").has_value());
    REQUIRE_FALSE(ParseMaxFramesInFlight("off").has_value());
    REQUIRE_FALSE(ParseMaxFramesInFlight("").has_value());
    REQUIRE_FALSE(ParseMaxFramesInFlight("-1").has_value());
    REQUIRE_FALSE(ParseMaxFramesInFlight("2x").has_value());
    REQUIRE(ParseMaxFramesInFlight("1") == 1u);
    REQUIRE(ParseMaxFramesInFlight("2") == 2u);
    REQUIRE(ParseMaxFramesInFlight("7") == 7u);
    REQUIRE(ParseMaxFramesInFlight("99") == Core::NvnHle::PresentDepthLimit::MaxDepth);
    // `self`: the present waits for itself (depth 0), unlike "0" which is no limit at all.
    REQUIRE(ParseMaxFramesInFlight("self") == 0u);
    REQUIRE_FALSE(ParseMaxFramesInFlight("Self").has_value());
}
