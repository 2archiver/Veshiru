// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "common/vram_policy.h"

using namespace Common::Literals;
namespace VP = Common::VramPolicy;

TEST_CASE("VRAM reserve follows the card, not the recent failures", "[common][vram]") {
    // The GTX 1650's driver budget on this laptop: ~3.0 GiB after the driver keeps its own share.
    CHECK(VP::AutoReserve(3096_MiB) == 387_MiB);
    CHECK(VP::AutoReserve(4_GiB) == 512_MiB);
    // An eighth of 8 GiB is 1 GiB, which the cap allows; 16 GiB is capped to 1 GiB, not 2.
    CHECK(VP::AutoReserve(8_GiB) == 1_GiB);
    CHECK(VP::AutoReserve(16_GiB) == VP::MaxReserve);
    CHECK(VP::AutoReserve(24_GiB) == VP::MaxReserve);
}

TEST_CASE("VRAM reserve never exceeds an eighth and is zero for no budget", "[common][vram]") {
    for (const u64 budget : {256_MiB, 1_GiB, 2_GiB, 3_GiB, 6_GiB, 12_GiB, 48_GiB}) {
        CHECK(VP::AutoReserve(budget) <= budget / 8);
        CHECK(VP::AutoReserve(budget) <= VP::MaxReserve);
    }
    CHECK(VP::AutoReserve(0) == 0);
    // Below 8 MiB an eighth rounds to nothing; the cache still works, it just has no margin.
    CHECK(VP::AutoReserve(4_MiB) == 0);
}

TEST_CASE("Critical memory leaves the reserve free and stays above half", "[common][vram]") {
    CHECK(VP::CriticalMemory(3096_MiB, VP::AutoReserve(3096_MiB)) == 2709_MiB);
    CHECK(VP::CriticalMemory(8_GiB, VP::AutoReserve(8_GiB)) == 7_GiB);
    // A reserve larger than the budget can't take the threshold to zero or below.
    CHECK(VP::CriticalMemory(512_MiB, 4_GiB) == 256_MiB);
    CHECK(VP::CriticalMemory(1_GiB, 1_GiB) == 512_MiB);
    CHECK(VP::CriticalMemory(0, 0) == 0);
}

TEST_CASE("The 4 GB laptop keeps well over half its budget for textures", "[common][vram]") {
    // The report that started this: the log said "keeping 2048 MiB more free" on a card whose
    // driver budget is about 3 GiB, so the cache held 2 GiB of it back and re-decoded textures
    // every frame. Under the single policy the cache may fill 2709 of 3096 MiB.
    const u64 budget = 3096_MiB;
    const u64 critical = VP::CriticalMemory(budget, VP::AutoReserve(budget));
    CHECK(critical > 2_GiB);
    CHECK(budget - critical < 512_MiB);
}
