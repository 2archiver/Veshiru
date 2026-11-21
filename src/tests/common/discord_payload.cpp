// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "common/scm_rev.h"
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
    CHECK(payload.find("\"state\":\"" + DiscordRPC::PresenceName() + " - Version 1.2.3\"") !=
          std::string::npos);
    CHECK(payload.find("\"timestamps\":{\"start\":1800000000}") != std::string::npos);
    CHECK(payload.find("\"pid\":42") != std::string::npos);
    CHECK(payload.find("\"nonce\":\"nonce-1\"") != std::string::npos);
    CHECK(payload.find("assets") == std::string::npos);
    CHECK(payload.find("party") == std::string::npos);
    CHECK(payload.find("secrets") == std::string::npos);
    CHECK(DiscordRPC::BuildPresencePayload(activity).started_at == 1'800'000'000);
}

TEST_CASE("Discord browsing payload does not disclose game details") {
    const DiscordRPC::ActivityData activity{.browsing = true};
    const std::string payload =
        DiscordRPC::BuildSetActivityPayload(activity, 42, "nonce-2");

    CHECK(payload.find("\"details\":\"Browsing library\"") != std::string::npos);
    CHECK(payload.find("\"state\":\"" + DiscordRPC::PresenceName() + "\"") != std::string::npos);
    CHECK(payload.find("timestamps") == std::string::npos);
    CHECK(DiscordRPC::BuildPresencePayload(activity).started_at == std::nullopt);
}

TEST_CASE("Discord presence names the build version the About dialog shows") {
    // The same generated constant, so the presence can never disagree with the build.
    CHECK(DiscordRPC::PresenceName().find(Common::g_build_version) != std::string::npos);
    CHECK(DiscordRPC::PresenceName().starts_with("Veshiru "));
    const DiscordRPC::ActivityData activity{
        .browsing = false,
        .game_title = "Example Adventure",
        .game_version = "1.2.3",
        .started_at = 0,
    };
    const std::string state = DiscordRPC::BuildPresencePayload(activity).state;
    CHECK(state.find(Common::g_build_version) != std::string::npos);
}

TEST_CASE("Discord handshake and clear payloads are well formed") {
    const std::string handshake = DiscordRPC::BuildHandshakePayload("123456789012345678");
    CHECK(handshake == "{\"v\":1,\"client_id\":\"123456789012345678\"}");

    const std::string clear = DiscordRPC::BuildClearActivityPayload(7, "n-3");
    CHECK(clear == "{\"cmd\":\"SET_ACTIVITY\",\"args\":{\"pid\":7,\"activity\":null},"
                   "\"nonce\":\"n-3\"}");
}

TEST_CASE("Discord frame header is a little-endian opcode and length") {
    const std::string frame = DiscordRPC::BuildFrame(1, "abc");
    REQUIRE(frame.size() == 11);
    // opcode 1
    CHECK(frame[0] == '\x01');
    CHECK(frame[1] == '\x00');
    CHECK(frame[2] == '\x00');
    CHECK(frame[3] == '\x00');
    // length 3
    CHECK(frame[4] == '\x03');
    CHECK(frame[5] == '\x00');
    CHECK(frame[6] == '\x00');
    CHECK(frame[7] == '\x00');
    CHECK(frame.substr(8) == "abc");

    const std::string big = DiscordRPC::BuildFrame(2, std::string(0x1234, 'x'));
    CHECK(static_cast<unsigned char>(big[4]) == 0x34);
    CHECK(static_cast<unsigned char>(big[5]) == 0x12);
    CHECK(big.size() == 8 + 0x1234);

    const std::string empty = DiscordRPC::BuildFrame(0, {});
    CHECK(empty.size() == 8);
}

TEST_CASE("Discord payload escapes quotes, backslashes and control characters") {
    const DiscordRPC::ActivityData activity{
        .browsing = false,
        .game_title = "Game \"Quoted\" \\ Backslash \t Tab \n Newline",
        .game_version = "1.0.0",
        .started_at = 0,
    };
    const std::string payload = DiscordRPC::BuildSetActivityPayload(activity, 1, "n");

    // The title appears only in escaped form.
    CHECK(payload.find("Game \\\"Quoted\\\" \\\\ Backslash \\t Tab \\n Newline") !=
          std::string::npos);
    // No raw control characters and no unescaped quote inside the title.
    CHECK(payload.find('\t') == std::string::npos);
    CHECK(payload.find('\n') == std::string::npos);
    // A handshake with a quote in the id stays a single line of valid JSON.
    CHECK(DiscordRPC::BuildHandshakePayload("a\"b").find("a\\\"b") != std::string::npos);
}

TEST_CASE("Discord payload passes unicode titles through untouched") {
    const DiscordRPC::ActivityData activity{
        .browsing = false,
        .game_title = "スプラトゥーン ¥ € 🎮",
        .game_version = "9.9.9",
        .started_at = 0,
    };
    const std::string payload = DiscordRPC::BuildSetActivityPayload(activity, 1, "n");
    CHECK(payload.find("スプラトゥーン ¥ € 🎮") != std::string::npos);
}

TEST_CASE("Discord payload carries no paths or file names of its own") {
    // The builder receives only the resolved title and version; it has no path to leak. What it
    // is given, it escapes: a hostile title cannot smuggle a second JSON field in.
    const DiscordRPC::ActivityData activity{
        .browsing = false,
        .game_title = "C:\\Users\\Someone\\game.nup\",\"secrets\":\"leak",
        .game_version = "1.0.0",
        .started_at = 0,
    };
    const std::string payload = DiscordRPC::BuildSetActivityPayload(activity, 1, "n");
    CHECK(payload.find("\"secrets\":\"leak") == std::string::npos);
    CHECK(payload.find("\\\\Users\\\\") != std::string::npos); // escaped, still just the title
    CHECK(payload.find("/home/") == std::string::npos);
    CHECK(payload.find(".nup\")") == std::string::npos); // no unescaped file suffix either
}
