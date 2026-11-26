// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "qt_common/game_list/satsuma_compat.h"

TEST_CASE("Satsuma's compatibility list finds the games it knows", "[game_list][compat]") {
    // Switch 2 games from the list: the status and its rank come through unchanged.
    const auto playable = QtCommon::Game::FindSatsumaCompat(0x010015100B514000);
    REQUIRE(playable.has_value());
    CHECK(playable->rank == QtCommon::Game::CompatPlayable.rank);
    CHECK(std::string_view{playable->text} == "Playable");

    const auto in_game_plus = QtCommon::Game::FindSatsumaCompat(0x0400C3F00006E000);
    REQUIRE(in_game_plus.has_value());
    CHECK(in_game_plus->rank == QtCommon::Game::CompatInGamePlus.rank);
    CHECK(std::string_view{in_game_plus->text} == "In Game +");
}

TEST_CASE("Satsuma's compatibility list stays empty for unknown programs", "[game_list][compat]") {
    CHECK_FALSE(QtCommon::Game::FindSatsumaCompat(0).has_value());
    CHECK_FALSE(QtCommon::Game::FindSatsumaCompat(0x0100FFFFFFFFFF00).has_value());
}

TEST_CASE("Compatibility ranks sort from best to worst", "[game_list][compat]") {
    CHECK(QtCommon::Game::CompatPlayable.rank < QtCommon::Game::CompatInGamePlus.rank);
    CHECK(QtCommon::Game::CompatInGamePlus.rank < QtCommon::Game::CompatInGame.rank);
    CHECK(QtCommon::Game::CompatInGame.rank < QtCommon::Game::CompatMenus.rank);
    CHECK(QtCommon::Game::CompatMenus.rank < QtCommon::Game::CompatBoots.rank);
}
