// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "qt_common/game_list/refresh_coalescer.h"

TEST_CASE("Watcher events of both kinds in one window run both refreshes", "[game_list]") {
    GameListRefresh::Coalescer coalescer;
    CHECK_FALSE(coalescer.Any());

    // A game-directory event and an external-content event land in the same 30 ms window.
    coalescer.OnGameDir();
    coalescer.OnExternal();
    coalescer.OnGameDir(); // a second burst of the same kind is still one refresh
    CHECK(coalescer.Any());

    const auto first = coalescer.Take();
    CHECK(first.game_dir);
    CHECK(first.external);
    CHECK_FALSE(coalescer.Any());

    // The window consumed them: nothing runs a second time.
    const auto second = coalescer.Take();
    CHECK_FALSE(second.game_dir);
    CHECK_FALSE(second.external);
}

TEST_CASE("A game-directory event alone is not swallowed by an external flag", "[game_list]") {
    GameListRefresh::Coalescer coalescer;

    coalescer.OnGameDir();
    const auto requested = coalescer.Take();
    CHECK(requested.game_dir);
    CHECK_FALSE(requested.external);

    // The dropped case PR #22 had: external set first, game dir second, one flag.
    coalescer.OnExternal();
    coalescer.OnGameDir();
    const auto both = coalescer.Take();
    CHECK(both.external);
    CHECK(both.game_dir);

    // External-only windows still work.
    coalescer.OnExternal();
    const auto external_only = coalescer.Take();
    CHECK(external_only.external);
    CHECK_FALSE(external_only.game_dir);
}
