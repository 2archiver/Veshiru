// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "qt_common/game_list/parse_cache.h"

namespace {

GameList::ParseCache::EntryData MakeData(const std::string& path, u64 program_id) {
    GameList::ParseCache::EntryData data;
    data.path = path;
    data.name = "Test Game";
    data.icon = {1, 2, 3, 4};
    data.file_type_string = "Nintendo Switch Program";
    data.program_id = program_id;
    data.size = 0x1000;
    data.patch_versions = "Update (1.0.1)";
    data.game_version = "1.0.1";
    data.loose_dump = false;
    return data;
}

} // Anonymous namespace

TEST_CASE("The parse cache reuses a row only while its file is unchanged",
          "[game_list][parse_cache]") {
    GameList::ParseCache::Cache cache;

    // Nothing was parsed yet: a lookup must miss rather than invent a row.
    CHECK(cache.Find("games/Mario.nsp", 0x1000, 1000) == nullptr);

    cache.Store(MakeData("games/Mario.nsp", 0x100'0000'0000'0001), 0x1000, 1000);

    // Same path, size and mtime: the row is reused as-is.
    const auto* cached = cache.Find("games/Mario.nsp", 0x1000, 1000);
    REQUIRE(cached != nullptr);
    CHECK(cached->program_id == 0x100'0000'0000'0001);
    CHECK(cached->name == "Test Game");
    CHECK(cached->patch_versions == "Update (1.0.1)");
    CHECK(cached->game_version == "1.0.1");
    CHECK(cached->icon == std::vector<u8>{1, 2, 3, 4});

    // A different file, a modified file and a touched file all miss: the worker must
    // re-parse them instead of reusing the stale row.
    CHECK(cache.Find("games/Other.nsp", 0x1000, 1000) == nullptr);
    CHECK(cache.Find("games/Mario.nsp", 0x2000, 1000) == nullptr);
    CHECK(cache.Find("games/Mario.nsp", 0x1000, 2000) == nullptr);

    // A re-parsed file replaces the cached row under its new fingerprint.
    auto updated = MakeData("games/Mario.nsp", 0x100'0000'0000'0001);
    updated.game_version = "1.0.2";
    cache.Store(std::move(updated), 0x2000, 2000);
    CHECK(cache.Find("games/Mario.nsp", 0x1000, 1000) == nullptr);
    const auto* reparsed = cache.Find("games/Mario.nsp", 0x2000, 2000);
    REQUIRE(reparsed != nullptr);
    CHECK(reparsed->game_version == "1.0.2");

    // Two fingerprints of the same path never coexist: the cache stays bounded by the
    // number of discovered files, not by the number of scans.
    CHECK(cache.Size() == 1);
}

TEST_CASE("The parse cache is cleared when keys or firmware change", "[game_list][parse_cache]") {
    GameList::ParseCache::Cache cache;
    cache.Store(MakeData("games/Mario.nsp", 1), 0x1000, 1000);
    cache.Store(MakeData("games/Zelda.nsp", 2), 0x2000, 2000);
    REQUIRE(cache.Find("games/Mario.nsp", 0x1000, 1000) != nullptr);
    REQUIRE(cache.Find("games/Zelda.nsp", 0x2000, 2000) != nullptr);

    cache.Clear();

    // Installing keys or firmware invalidates every entry: a dump can parse differently
    // afterwards, so nothing may be reused.
    CHECK(cache.Find("games/Mario.nsp", 0x1000, 1000) == nullptr);
    CHECK(cache.Find("games/Zelda.nsp", 0x2000, 2000) == nullptr);
    CHECK(cache.Size() == 0);

    // The cache accepts new entries again afterwards.
    cache.Store(MakeData("games/Mario.nsp", 1), 0x1000, 1000);
    CHECK(cache.Find("games/Mario.nsp", 0x1000, 1000) != nullptr);
}

TEST_CASE("Refresh reasons report themselves in the summary log line", "[game_list]") {
    CHECK(ToString(GameListRefreshReason::Startup) == "startup");
    CHECK(ToString(GameListRefreshReason::Settings) == "settings");
    CHECK(ToString(GameListRefreshReason::Watcher) == "watcher");
}
