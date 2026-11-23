// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// When a render-condition predicate computed ahead (nvn_condition_prefetch.h) may still serve a
// later SetRenderEnableConditional, and which writes drop it. Pure logic, no device.

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_condition_prefetch.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"

namespace {

using Vulkan::Nvn::ConditionPrefetch;
using Vulkan::Nvn::Pool;

ConditionPrefetch::Entry Make(u64 address, u32 mode, const Pool& pool, u64 offset,
                              u64 position = 0) {
    return {.address = address,
            .mode = mode,
            .slot = 0,
            .position = position,
            .pool_a = &pool,
            .offset_a = offset,
            .pool_b = &pool,
            .offset_b = offset + 16};
}

} // namespace

TEST_CASE("NVN condition prefetch: a predicate serves the same address, mode and pair only",
          "[nvn]") {
    Pool pool;
    Pool other;
    ConditionPrefetch prefetch;
    prefetch.Add(Make(0x1000, 1, pool, 0x100));
    CHECK(prefetch.Find(0x1000, 1, &pool, 0x100, &pool, 0x110) != nullptr);
    CHECK(prefetch.Find(0x1000, 0, &pool, 0x100, &pool, 0x110) == nullptr); // other mode
    CHECK(prefetch.Find(0x1010, 1, &pool, 0x110, &pool, 0x120) == nullptr); // other address
    // The address now resolves elsewhere (the pool was replaced): not the bytes compared.
    CHECK(prefetch.Find(0x1000, 1, &other, 0x100, &other, 0x110) == nullptr);
    CHECK(prefetch.Contains(0x1000, 1));
    CHECK_FALSE(prefetch.Contains(0x1000, 0));
}

TEST_CASE("NVN condition prefetch: a write to either counter drops the predicate", "[nvn]") {
    Pool pool;
    Pool other;
    ConditionPrefetch prefetch;
    const auto fill = [&] {
        prefetch.Clear();
        prefetch.Add(Make(0x1000, 0, pool, 0x100));
        prefetch.Add(Make(0x2000, 0, pool, 0x200));
    };
    fill();
    // Another pool, the bytes between the counters, the end of a report (its time field), the byte
    // before the pair: nothing the comparison reads.
    prefetch.Invalidate(other, 0x100, 0x200);
    prefetch.Invalidate(pool, 0x108, 8);
    prefetch.Invalidate(pool, 0x118, 8);
    prefetch.Invalidate(pool, 0xFF, 1);
    CHECK(prefetch.Size() == 2);
    // The first counter, one byte of it.
    prefetch.Invalidate(pool, 0x107, 1);
    CHECK_FALSE(prefetch.Contains(0x1000, 0));
    CHECK(prefetch.Contains(0x2000, 0));
    fill();
    // The second counter (a report on address + 16).
    prefetch.Invalidate(pool, 0x210, 16);
    CHECK(prefetch.Contains(0x1000, 0));
    CHECK_FALSE(prefetch.Contains(0x2000, 0));
    fill();
    // A long write over both pairs (checked predicate by predicate, not granule by granule).
    prefetch.Invalidate(pool, 0, 0x10000);
    CHECK(prefetch.Empty());
    fill();
    // A size of 0 counts as one byte.
    prefetch.Invalidate(pool, 0x210, 0);
    CHECK_FALSE(prefetch.Contains(0x2000, 0));
    // Unaligned counters span two granules.
    prefetch.Clear();
    prefetch.Add(Make(0x3000, 1, pool, 0x304));
    prefetch.Invalidate(pool, 0x308, 4);
    CHECK(prefetch.Empty());
}

TEST_CASE("NVN condition prefetch: a replaced predicate keeps the index right", "[nvn]") {
    Pool pool;
    ConditionPrefetch prefetch;
    prefetch.Add(Make(0x1000, 0, pool, 0x100));
    // The same address and mode computed again, now from other bytes (pool remapped).
    prefetch.Add(Make(0x1000, 0, pool, 0x400));
    CHECK(prefetch.Size() == 1);
    // The old bytes do not drop it any more, the new ones do.
    prefetch.Invalidate(pool, 0x100, 8);
    CHECK(prefetch.Contains(0x1000, 0));
    prefetch.Invalidate(pool, 0x410, 8);
    CHECK(prefetch.Empty());
}

TEST_CASE("NVN condition prefetch: predicates whose slot the ring handed out again are dropped",
          "[nvn]") {
    Pool pool;
    ConditionPrefetch prefetch;
    constexpr u64 RING = 2048;
    prefetch.Add(Make(0x1000, 0, pool, 0x100, 10));
    prefetch.Add(Make(0x2000, 0, pool, 0x200, 11));
    // Slot positions 10 and 11 come back at 10 + RING and 11 + RING.
    prefetch.DropRecycled(10 + RING, RING);
    CHECK(prefetch.Size() == 2);
    prefetch.DropRecycled(11 + RING, RING);
    CHECK_FALSE(prefetch.Contains(0x1000, 0));
    CHECK(prefetch.Contains(0x2000, 0));
    prefetch.DropRecycled(12 + RING, RING);
    CHECK(prefetch.Empty());
}
