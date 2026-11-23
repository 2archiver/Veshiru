// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// MirrorLineGuard (nvn_line_guard.h): which pages of a pool the line-mode write-back merge may
// store whole, and the wait that keeps a guest access from landing inside a merge in flight.

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_line_guard.h"

namespace {

using Vulkan::Nvn::LineGuardStats;
using Vulkan::Nvn::MirrorLineGuard;
using Run = MirrorLineGuard::Run;
using Pages = std::vector<std::pair<u64, u64>>;
using namespace std::chrono_literals;

constexpr u64 PAGE = MirrorLineGuard::PAGE;

std::shared_ptr<MirrorLineGuard> MakeGuard(u64 pages, std::shared_ptr<LineGuardStats>& stats,
                                           std::thread::id replay = {},
                                           std::chrono::milliseconds limit = 5000ms) {
    stats = std::make_shared<LineGuardStats>();
    return std::make_shared<MirrorLineGuard>(pages * PAGE, stats, replay, limit);
}

} // Anonymous namespace

TEST_CASE("NVN line guard: a fresh range is one line run, armed once", "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const auto guard = MakeGuard(16, stats);
    // Word-aligned bounds inside pages 1..4 keep their exact bytes; every page they touch is armed.
    auto first = guard->Record(PAGE + 8, 4 * PAGE + 100);
    CHECK(first.runs == std::vector<Run>{{PAGE + 8, 4 * PAGE + 100, true}});
    CHECK(first.arm == Pages{{1, 4}});
    CHECK(first.token != nullptr);
    CHECK_FALSE(first.disarm_all);
    // The traps stay: the same range again arms nothing, a larger one only the new pages.
    auto second = guard->Record(PAGE + 8, 4 * PAGE + 100);
    CHECK(second.arm.empty());
    auto third = guard->Record(0, 8 * PAGE);
    CHECK(third.arm == Pages{{0, 0}, {5, 7}});
    CHECK(third.runs == std::vector<Run>{{0, 8 * PAGE, true}});
    CHECK(stats->armed_pages.load() == 4 + 4);
    CHECK(stats->line_bytes.load() == 2 * (3 * PAGE + 92) + 8 * PAGE);
}

TEST_CASE("NVN line guard: a touched page is merged exactly from then on", "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const auto guard = MakeGuard(16, stats);
    guard->Record(0, 16 * PAGE);
    CHECK(guard->Touched(5 * PAGE + 64, 5 * PAGE + 128)); // no merge in flight: no wait
    CHECK(guard->HotPage(5));
    CHECK_FALSE(guard->HotPage(4));
    auto plan = guard->Record(2 * PAGE, 9 * PAGE);
    CHECK(plan.runs == std::vector<Run>{{2 * PAGE, 5 * PAGE, true},
                                        {5 * PAGE, 6 * PAGE, false},
                                        {6 * PAGE, 9 * PAGE, true}});
    CHECK(plan.arm.empty()); // the hot page is not armed again
    CHECK(plan.token != nullptr);
    // A range inside the hot page is all exact and needs no token.
    auto hot_only = guard->Record(5 * PAGE + 16, 5 * PAGE + 256);
    CHECK(hot_only.runs == std::vector<Run>{{5 * PAGE + 16, 5 * PAGE + 256, false}});
    CHECK(hot_only.token == nullptr);
    CHECK(stats->hot_pages.load() == 1);
    // Line mode off for the call: one exact run, no token, no arming.
    auto off = guard->Record(0, 4 * PAGE, false);
    CHECK(off.runs == std::vector<Run>{{0, 4 * PAGE, false}});
    CHECK(off.token == nullptr);
    CHECK(off.arm.empty());
}

TEST_CASE("NVN line guard: a touch waits for the line merges in flight on its pages",
          "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const auto guard = MakeGuard(16, stats);
    auto plan = guard->Record(0, 4 * PAGE);
    std::shared_ptr<void> token = std::move(plan.token); // the GPU is merging

    // A page no merge in flight covers does not wait.
    const auto begin = std::chrono::steady_clock::now();
    CHECK(guard->Touched(10 * PAGE, 10 * PAGE + 4));
    CHECK(std::chrono::steady_clock::now() - begin < 2000ms);
    CHECK(stats->waits.load() == 0);

    // A covered page waits until the merge completes (the token dies), and is hot already while
    // it waits: a merge recorded now is exact there.
    std::atomic<bool> returned{false};
    std::atomic<bool> finished_in_time{false};
    std::thread guest{[&] {
        finished_in_time = guard->Touched(2 * PAGE + 4, 2 * PAGE + 8);
        returned = true;
    }};
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!guard->HotPage(2) && !returned.load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    if (!guard->HotPage(2)) {
        token.reset();
        guest.join();
        FAIL("page 2 never became hot");
    }
    std::this_thread::sleep_for(50ms);
    CHECK_FALSE(returned.load());
    auto during = guard->Record(0, 4 * PAGE);
    CHECK(during.runs == std::vector<Run>{{0, 2 * PAGE, true},
                                          {2 * PAGE, 3 * PAGE, false},
                                          {3 * PAGE, 4 * PAGE, true}});
    CHECK_FALSE(returned.load()); // the new merge does not cover page 2 in line mode
    token.reset();
    guest.join();
    CHECK(returned.load());
    CHECK(finished_in_time.load());
    CHECK(stats->waits.load() == 1);
    CHECK(stats->timeouts.load() == 0);
}

TEST_CASE("NVN line guard: the replay thread's touch never waits", "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const auto guard = MakeGuard(4, stats, std::this_thread::get_id());
    auto plan = guard->Record(0, 4 * PAGE);
    const auto begin = std::chrono::steady_clock::now();
    CHECK(guard->Touched(0, 4));
    CHECK(std::chrono::steady_clock::now() - begin < 2000ms);
    CHECK(guard->HotPage(0));
}

TEST_CASE("NVN line guard: a wait that times out makes the pool exact for good", "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const auto guard = MakeGuard(8, stats, {}, 40ms);
    auto plan = guard->Record(0, 8 * PAGE);
    CHECK_FALSE(guard->Touched(PAGE, PAGE + 4)); // the token never dies
    CHECK(stats->timeouts.load() == 1);
    CHECK(guard->ExactOnly());
    plan.token.reset();
    auto next = guard->Record(0, 8 * PAGE);
    CHECK(next.disarm_all); // once
    CHECK(next.runs == std::vector<Run>{{0, 8 * PAGE, false}});
    CHECK(next.token == nullptr);
    CHECK(next.arm.empty());
    CHECK_FALSE(guard->Record(0, 8 * PAGE).disarm_all);
    CHECK(stats->exact_pools.load() == 1);
}

TEST_CASE("NVN line guard: too many touched pages make the pool exact for good",
          "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const u64 pages = MirrorLineGuard::MAX_HOT_PAGES + 8;
    const auto guard = MakeGuard(pages, stats);
    guard->Record(0, pages * PAGE);
    for (u64 page = 0; page < MirrorLineGuard::MAX_HOT_PAGES; ++page) {
        CHECK(guard->Touched(page * PAGE, page * PAGE + 4));
    }
    CHECK_FALSE(guard->ExactOnly());
    // One touch of several pages at once counts every page.
    CHECK(guard->Touched(MirrorLineGuard::MAX_HOT_PAGES * PAGE,
                         (MirrorLineGuard::MAX_HOT_PAGES + 2) * PAGE));
    CHECK(guard->ExactOnly());
    auto plan = guard->Record(0, pages * PAGE);
    CHECK(plan.disarm_all);
    CHECK(plan.runs == std::vector<Run>{{0, pages * PAGE, false}});
    CHECK(plan.token == nullptr);
}

TEST_CASE("NVN line guard: whole untouched words of pages are one run", "[nvn][line_guard]") {
    std::shared_ptr<LineGuardStats> stats;
    const auto guard = MakeGuard(300, stats);
    guard->Record(0, 300 * PAGE);
    auto plan = guard->Record(0, 300 * PAGE);
    CHECK(plan.arm.empty());
    CHECK(plan.runs == std::vector<Run>{{0, 300 * PAGE, true}});
    plan.token.reset(); // the GPU is done
    CHECK(guard->Touched(130 * PAGE, 131 * PAGE));
    plan = guard->Record(10 * PAGE, 290 * PAGE);
    CHECK(plan.runs == std::vector<Run>{{10 * PAGE, 130 * PAGE, true},
                                        {130 * PAGE, 131 * PAGE, false},
                                        {131 * PAGE, 290 * PAGE, true}});
}
