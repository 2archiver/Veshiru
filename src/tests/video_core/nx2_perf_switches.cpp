// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_perf.h"

namespace {

using namespace VideoCommon::Nx2Perf;

constexpr size_t Index(Opt opt) {
    return static_cast<size_t>(opt);
}

} // namespace

TEST_CASE("Nx2Perf pending_wake is on by default and off under SAFE", "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::PendingWake)] == "pending_wake");
    REQUIRE(IsOff(DefaultOffMask(), Index(Opt::PendingWake)));
    REQUIRE(!IsOff(ParseOffMask(false, "", ""), Index(Opt::PendingWake)));
    REQUIRE(IsOff(ParseOffMask(true, "", ""), Index(Opt::PendingWake)));
    REQUIRE(!IsOff(ParseOffMask(true, "pending_wake", ""), Index(Opt::PendingWake)));
    REQUIRE(ParseOffMask(false, "", "pending_wake") == Bit(Opt::PendingWake));
}
