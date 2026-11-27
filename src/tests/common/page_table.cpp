// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "common/sparse_large_vector.h"

namespace {

using Entry = Common::PageTable::PageEntryData;

const std::size_t EntriesPerHostPage = Common::HostPageSize / sizeof(Entry);

} // namespace

TEST_CASE("Page table block ids do not wrap", "[common]") {
    REQUIRE(Entry::BlockIdForMapping(0) == 0);
    REQUIRE(Entry::BlockIdForMapping(65534) == 65534);
    REQUIRE(Entry::BlockIdForMapping(65535) == Entry::INVALID_BLOCK_ID);
    REQUIRE(Entry::BlockIdForMapping(65536) == Entry::INVALID_BLOCK_ID);
    REQUIRE(Entry::BlockIdForMapping((std::numeric_limits<u64>::max)()) ==
            Entry::INVALID_BLOCK_ID);
}
