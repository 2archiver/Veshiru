// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/fs/path_util.h"

using namespace Common::FS;

TEST_CASE("Veshiru keeps its data apart from Eden's", "[fs]") {
    const auto& data = GetEdenPath(EdenPath::EdenDir);
    if (data.filename() == "user") {
        SKIP("portable user directory in use");
    }
    CHECK(data.filename() == "veshiru");
    CHECK(GetLegacyPath(LegacyEdenDir).filename() == "eden");
    CHECK(GetLegacyPath(LegacyEdenDir).parent_path() == data.parent_path());
    CHECK(GetLegacyPath(EglantineDir).filename() == "eglantine");
    CHECK(GetLegacyPath(EglantineDir).parent_path() == data.parent_path());
    CHECK(GetLegacyPath(LichiiDir).filename() == "lichii");
    CHECK(GetLegacyPath(LichiiDir).parent_path() == data.parent_path());
    CHECK(GetLegacyPath(NemyuDir).filename() == "nemyu");
    CHECK(GetLegacyPath(NemyuDir).parent_path() == data.parent_path());
}
