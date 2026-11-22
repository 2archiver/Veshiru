// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include "core/device_memory.h"

using namespace std::chrono_literals;

TEST_CASE("Zero-page trim: never while bulk mapping is active", "[core][device_memory]") {
    // A pass holds VAD locks over the whole guest working set; a load's map calls stalled 10-14 ms
    // behind one, so mapping wins whatever else is true.
    CHECK_FALSE(Core::ShouldTrimZeroPages(true, false, 600s, true, 120s));
    CHECK_FALSE(Core::ShouldTrimZeroPages(true, true, 600s, true, 120s));
}

TEST_CASE("Zero-page trim: a quiet guest is trimmed", "[core][device_memory]") {
    // Paused, loading, or sitting in a menu: the guest is not presenting, so a pass costs nothing.
    CHECK(Core::ShouldTrimZeroPages(false, false, 0s, false, 120s));
    CHECK(Core::ShouldTrimZeroPages(false, false, 0s, true, 20s));
}

TEST_CASE("Zero-page trim: a presenting guest is not frozen for a saving",
          "[core][device_memory]") {
    // Ocarina of Time's logs: 1.7 and 2.1 s passes in the middle of gameplay. Without host
    // memory pressure the pass waits indefinitely for a quiet moment.
    CHECK_FALSE(Core::ShouldTrimZeroPages(false, true, 0s, false, 120s));
    CHECK_FALSE(Core::ShouldTrimZeroPages(false, true, 60s, false, 120s));
    CHECK_FALSE(Core::ShouldTrimZeroPages(false, true, 119s, false, 120s));
    CHECK_FALSE(Core::ShouldTrimZeroPages(false, true, 600s, false, 120s));
    // Under pressure it still runs, but once per escape interval instead of every 5-30 s: two
    // minutes on a 32 GiB host, twenty seconds on a 16 GiB one.
    CHECK_FALSE(Core::ShouldTrimZeroPages(false, true, 119s, true, 120s));
    CHECK(Core::ShouldTrimZeroPages(false, true, 120s, true, 120s));
    CHECK_FALSE(Core::ShouldTrimZeroPages(false, true, 19s, true, 20s));
    CHECK(Core::ShouldTrimZeroPages(false, true, 20s, true, 20s));
}
