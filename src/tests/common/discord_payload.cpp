// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "qt_common/discord/discord_payload.h"

TEST_CASE("Discord presence payload includes only approved game fields") {
    const DiscordRPC::ActivityData activity{
        .browsing = false,
        .game_title = "Example Adventure",
        .game_version = "1.2.3",
        .started_at = 1'800'000'000,
    };
    const std::string payload =
        DiscordRPC::BuildSetActivityPayload(activity, 42, "nonce-1");

    CHECK(payload.find("\"details\":\"Playing Example Adventure\"") != std::string::npos);
    CHECK(payload.find("\"state\":\"Veshiru 2.3.0 · Version 1.2.3\"") !=
          std::string::npos);
    CHECK(payload.find("\"timestamps\":{\"start\":1800000000}") != std::string::npos);
    CHECK_FALSE(payload.find("assets") != std::string::npos);
    CHECK_FALSE(payload.find("party") != std::string::npos);
    CHECK_FALSE(payload.find("secrets") != std::string::npos);
    CHECK(DiscordRPC::BuildPresencePayload(activity).started_at == 1'800'000'000);
}

TEST_CASE("Discord browsing payload does not disclose game details") {
    const DiscordRPC::ActivityData activity{.browsing = true};
    const std::string payload =
        DiscordRPC::BuildSetActivityPayload(activity, 42, "nonce-2");

    CHECK(payload.find("\"details\":\"Browsing library\"") != std::string::npos);
    CHECK(payload.find("\"state\":\"Veshiru 2.3.0\"") != std::string::npos);
    CHECK(payload.find("timestamps") == std::string::npos);
    CHECK(DiscordRPC::BuildPresencePayload(activity).started_at == std::nullopt);
}
