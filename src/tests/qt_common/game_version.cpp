// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "qt_common/game_list/game_version.h"

using GameListVersion::Inputs;

TEST_CASE("The enabled update decides the version", "[game_list]") {
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "Update (1.8.0)\nUpdate (1.7.2)",
              .control_metadata_version = "1.0.0",
              .control_metadata_exists = true,
          }) == "1.8.0");
}

TEST_CASE("A disabled update does not decide the version", "[game_list]") {
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "[D] Update (1.8.0)",
              .control_metadata_version = "1.0.0",
              .control_metadata_exists = true,
          }) == "1.0.0");
}

TEST_CASE("Without an update the control metadata version is used", "[game_list]") {
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "",
              .control_metadata_version = "3.2.0",
              .control_metadata_exists = true,
          }) == "3.2.0");
    // A packed update carried by the container shows as its own line.
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "Update (PACKED)",
              .control_metadata_version = "1.1.1",
              .control_metadata_exists = true,
          }) == "PACKED");
}

TEST_CASE("The loader's display version is the next fallback", "[game_list]") {
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "",
              .control_metadata_version = "",
              .control_metadata_exists = true,
              .display_version = "2.0.0",
          }) == "2.0.0");
}

TEST_CASE("A base NACP without a version reports 1.0.0; no NACP reports nothing", "[game_list]") {
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "",
              .control_metadata_version = "",
              .control_metadata_exists = true,
              .display_version = "",
          }) == "1.0.0");
    // Nothing known: the caller hides the badge rather than show "Version ".
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "",
              .control_metadata_version = "",
              .control_metadata_exists = false,
              .display_version = "",
          }) == "");
    // A loose dump with a display version still names one.
    CHECK(GameListVersion::Resolve(Inputs{
              .patch_versions = "",
              .control_metadata_version = "",
              .control_metadata_exists = false,
              .display_version = "0.9.1",
          }) == "0.9.1");
}
