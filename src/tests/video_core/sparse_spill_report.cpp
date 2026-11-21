// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/sparse_spill_report.h"

namespace {

using Vulkan::SparseSpillReport;

constexpr u64 Tile = 64 * 1024;
constexpr s64 Interval = SparseSpillReport::IntervalMs;

constexpr SparseSpillReport::Counters Counts(u64 tiles, u64 bytes, u64 failed) {
    return {.tiles = tiles, .bytes = bytes, .failed = failed};
}

} // namespace

TEST_CASE("Sparse spill report: the first spill is reported at once",
          "[video_core][sparse_spill]") {
    SparseSpillReport report;
    const auto first = report.Record(5'000, Tile, true);
    REQUIRE(first.has_value());
    REQUIRE(first->interval == Counts(1, Tile, 0));
    REQUIRE(first->total == first->interval);
    REQUIRE(report.Pending() == Counts(0, 0, 0));
}

TEST_CASE("Sparse spill report: at most one report per interval",
          "[video_core][sparse_spill]") {
    SparseSpillReport report;
    REQUIRE(report.Record(0, Tile, true).has_value());

    // A burst inside the interval is counted, not reported
    for (s64 i = 1; i <= 200; ++i) {
        REQUIRE_FALSE(report.Record(i * 10, Tile, true).has_value());
    }
    REQUIRE_FALSE(report.Record(Interval - 1, Tile, false).has_value());
    REQUIRE(report.Pending() ==
            Counts(200, 200 * Tile, 1));

    // The next spill after the interval carries everything since the last report
    const auto second = report.Record(Interval, 2 * Tile, true);
    REQUIRE(second.has_value());
    REQUIRE(second->interval ==
            Counts(201, 202 * Tile, 1));
    REQUIRE(second->total ==
            Counts(202, 203 * Tile, 1));
    REQUIRE(report.Pending() == Counts(0, 0, 0));

    // The interval restarts from the report, not from the first spill
    REQUIRE_FALSE(report.Record(2 * Interval - 1, Tile, true).has_value());
    REQUIRE(report.Record(2 * Interval, Tile, true).has_value());
}

TEST_CASE("Sparse spill report: a tile no heap took is a failure, not a spilled tile",
          "[video_core][sparse_spill]") {
    SparseSpillReport report;
    const auto first = report.Record(0, Tile, false);
    REQUIRE(first.has_value());
    REQUIRE(first->interval == Counts(0, 0, 1));
    REQUIRE(report.Total() == Counts(0, 0, 1));
}

TEST_CASE("Sparse spill report: a quiet period does not produce a report by itself",
          "[video_core][sparse_spill]") {
    SparseSpillReport report;
    REQUIRE(report.Record(0, Tile, true).has_value());
    REQUIRE_FALSE(report.Record(100, Tile, true).has_value());
    // Nothing is reported until another spill happens, however long that takes; the counters
    // wait in Pending() and the totals in Total()
    REQUIRE(report.Pending() == Counts(1, Tile, 0));
    const auto later = report.Record(100 * Interval, Tile, true);
    REQUIRE(later.has_value());
    REQUIRE(later->interval ==
            Counts(2, 2 * Tile, 0));
    REQUIRE(later->total == Counts(3, 3 * Tile, 0));
}
