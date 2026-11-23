// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Pool mirror page bookkeeping (nvn_pool_mirror.h) and the vertex coverage line (nvn_vtxcov.h).
// The GPU paths are in compute_vulkan.cpp ("NVN pool mirror", "NVN vertex coverage").

#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_pool_mirror.h"
#include "video_core/renderer_vulkan/nvn/nvn_vtxcov.h"

namespace {

using Vulkan::Nvn::MirrorPages;
using Runs = std::vector<std::pair<u64, u64>>;
constexpr u64 PAGE = MirrorPages::PAGE;

} // namespace

TEST_CASE("NVN pool mirror pages: the first use takes every page it touches, once", "[nvn]") {
    MirrorPages pages{16 * PAGE};
    REQUIRE_FALSE(pages.AllValid(0, 1));
    Runs runs;
    // An unaligned range touches pages 1-3.
    pages.TakeInvalid(PAGE + 8, 2 * PAGE, runs);
    REQUIRE(runs == Runs{{PAGE, 4 * PAGE}});
    REQUIRE(pages.AllValid(PAGE, 3 * PAGE));
    REQUIRE(pages.AllValid(PAGE + 100, 10));
    REQUIRE_FALSE(pages.AllValid(0, 2 * PAGE));
    runs.clear();
    pages.TakeInvalid(PAGE, 3 * PAGE, runs);
    REQUIRE(runs.empty());
    REQUIRE(pages.ValidPages() == 3);
}

TEST_CASE("NVN pool mirror pages: invalidated pages come back as separate runs", "[nvn]") {
    MirrorPages pages{200 * PAGE};
    Runs runs;
    pages.TakeInvalid(0, 200 * PAGE, runs);
    REQUIRE(runs == Runs{{0, 200 * PAGE}});
    // Single pages across the first 64-page word, a stretch over a word boundary.
    pages.Invalidate(3 * PAGE, 1);
    pages.Invalidate(5 * PAGE + 10, PAGE);
    pages.Invalidate(60 * PAGE, 10 * PAGE);
    REQUIRE(pages.ValidPages() == 200 - 1 - 2 - 10);
    REQUIRE_FALSE(pages.AllValid(0, 4 * PAGE));
    REQUIRE(pages.AllValid(7 * PAGE, 53 * PAGE));
    runs.clear();
    pages.TakeInvalid(0, 200 * PAGE, runs);
    REQUIRE(runs == Runs{{3 * PAGE, 4 * PAGE}, {5 * PAGE, 7 * PAGE}, {60 * PAGE, 70 * PAGE}});
    REQUIRE(pages.AllValid(0, 200 * PAGE));
}

TEST_CASE("NVN pool mirror pages: the pool end clamps ranges and runs", "[nvn]") {
    // 2.5 pages: the last run stops at the pool size.
    MirrorPages pages{2 * PAGE + PAGE / 2};
    Runs runs;
    pages.TakeInvalid(PAGE, 100 * PAGE, runs);
    REQUIRE(runs == Runs{{PAGE, 2 * PAGE + PAGE / 2}});
    REQUIRE(pages.AllValid(2 * PAGE, PAGE));
    // Empty and outside ranges take and test nothing.
    runs.clear();
    pages.TakeInvalid(3 * PAGE, PAGE, runs);
    pages.TakeInvalid(0, 0, runs);
    REQUIRE(runs.empty());
    REQUIRE_FALSE(pages.AllValid(3 * PAGE, 1));
    REQUIRE_FALSE(pages.AllValid(PAGE, 0));
    pages.Invalidate(5 * PAGE, PAGE); // outside: ignored
    REQUIRE(pages.ValidPages() == 2);
}

TEST_CASE("NVN pool mirror pages: InvalidateAll forgets every page", "[nvn]") {
    MirrorPages pages{130 * PAGE};
    Runs runs;
    pages.TakeInvalid(0, 130 * PAGE, runs);
    REQUIRE(pages.ValidPages() == 130);
    pages.InvalidateAll();
    REQUIRE(pages.ValidPages() == 0);
    runs.clear();
    pages.TakeInvalid(64 * PAGE, 2 * PAGE, runs);
    REQUIRE(runs == Runs{{64 * PAGE, 66 * PAGE}});
}

TEST_CASE("NVN pool mirror pages: Set marks pages and TakeInvalidWithin takes only those",
          "[nvn]") {
    MirrorPages pages{256 * PAGE};
    MirrorPages wanted{256 * PAGE};
    std::vector<std::pair<u64, u64>> runs;

    // Nothing wanted: nothing taken, whatever is invalid.
    pages.TakeInvalidWithin(wanted, runs);
    CHECK(runs.empty());
    CHECK(pages.ValidPages() == 0);

    // The working set is a few ranges of a big pool (one crosses a 64-page word): the first copy
    // is those pages, not the pool.
    wanted.Set(5 * PAGE + 17, 100);         // page 5
    wanted.Set(62 * PAGE, 4 * PAGE);        // pages 62..65
    wanted.Set(200 * PAGE, 2 * PAGE);       // pages 200..201
    wanted.Set(255 * PAGE + 4000, 5000);    // page 255, clamped at the pool end
    pages.TakeInvalidWithin(wanted, runs);
    const std::vector<std::pair<u64, u64>> first{
        {5 * PAGE, 6 * PAGE}, {62 * PAGE, 66 * PAGE}, {200 * PAGE, 202 * PAGE},
        {255 * PAGE, 256 * PAGE}};
    CHECK(runs == first);
    CHECK(pages.ValidPages() == 1 + 4 + 2 + 1);
    CHECK(pages.AllValid(62 * PAGE, 4 * PAGE));
    CHECK_FALSE(pages.AllValid(0, 100 * PAGE));

    // Current: a second take finds nothing. A page of the set that goes stale comes back alone;
    // stale pages outside the set stay stale.
    runs.clear();
    pages.TakeInvalidWithin(wanted, runs);
    CHECK(runs.empty());
    pages.Invalidate(63 * PAGE, 1);
    pages.Invalidate(100 * PAGE, 1); // not wanted
    pages.TakeInvalidWithin(wanted, runs);
    CHECK(runs == std::vector<std::pair<u64, u64>>{{63 * PAGE, 64 * PAGE}});
    CHECK_FALSE(pages.AllValid(100 * PAGE, 1));

    // Invalidating everything brings back exactly the working set, merged across words.
    pages.InvalidateAll();
    runs.clear();
    pages.TakeInvalidWithin(wanted, runs);
    CHECK(runs == first);
}

TEST_CASE("NVN pool mirror pages: a pool that ends inside a word keeps its last run short",
          "[nvn]") {
    MirrorPages pages{70 * PAGE - 100};
    MirrorPages wanted{70 * PAGE - 100};
    wanted.Set(60 * PAGE, 10 * PAGE);
    std::vector<std::pair<u64, u64>> runs;
    pages.TakeInvalidWithin(wanted, runs);
    REQUIRE(runs.size() == 1);
    CHECK(runs[0].first == 60 * PAGE);
    CHECK(runs[0].second == 70 * PAGE - 100);
}

TEST_CASE("NVN vertex coverage: the vtxcov line is per present, vertex/index, reasons that happened",
          "[nvn]") {
    using Vulkan::Nvn::VtxCov;
    const auto at = [](VtxCov reason) { return static_cast<size_t>(reason); };
    Vulkan::Nvn::VtxCovCounts counts;
    counts.binds[0][at(VtxCov::Staged)] = 10;
    counts.bytes[0][at(VtxCov::Staged)] = 10 * 1024;
    counts.binds[1][at(VtxCov::Mirror)] = 2;
    counts.bytes[1][at(VtxCov::Mirror)] = 2 * 1024;
    counts.binds[0][at(VtxCov::NotScannedTargets)] = 4;
    counts.bytes[0][at(VtxCov::NotScannedTargets)] = 4 * 4096;
    counts.binds[1][at(VtxCov::IndexIndirect)] = 6;
    counts.bytes[1][at(VtxCov::IndexIndirect)] = 6 * 1024;
    CHECK(Vulkan::Nvn::FormatVtxCov(counts, 2, 8192, 4096,
                                    Vulkan::Nvn::VTXCOV_INPASS | Vulkan::Nvn::VTXCOV_BOUND) ==
          "presents=2 per present, vertex/index: served=5.0/1.0{staged=5.0/0.0,mirror=0.0/1.0} "
          "not_served=2.0/3.0{not_scanned_rt=2.0/0.0,index_indirect=0.0/3.0} bytes "
          "served_KiB=5/1 not_served_KiB=8/3{not_scanned_rt=8/0,index_indirect=0/3} "
          "copy_in_KiB=4 stage_copy_KiB=2 switches=inpass,bound");
    CHECK(Vulkan::Nvn::VtxCovSwitchNames(0) == "none");
    CHECK(Vulkan::Nvn::VtxCovSwitchNames(Vulkan::Nvn::VTXCOV_ALL) ==
          "inpass,shaderwrites,bound,bigpool");
    CHECK(Vulkan::Nvn::VtxCovServed(VtxCov::Device));
    CHECK_FALSE(Vulkan::Nvn::VtxCovServed(VtxCov::StageOff));
}
