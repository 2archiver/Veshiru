// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace DiscordRPC {

struct ActivityData {
    bool browsing{};
    std::string game_title;
    std::string game_version;
    std::int64_t started_at{};

    bool operator==(const ActivityData&) const = default;
};

struct PresencePayload {
    std::string details;
    std::string state;
    std::optional<std::int64_t> started_at;
};

/// "Veshiru <build version>", from the same generated constant the About dialog and the
/// updater use, so the presence can never name a version the build does not have.
std::string PresenceName();

PresencePayload BuildPresencePayload(const ActivityData& activity);
std::string BuildHandshakePayload(std::string_view application_id);
std::string BuildSetActivityPayload(const ActivityData& activity, std::uint32_t process_id,
                                    std::string_view nonce);
std::string BuildClearActivityPayload(std::uint32_t process_id, std::string_view nonce);

/// A Discord IPC frame: the little-endian opcode and payload length followed by the payload.
std::string BuildFrame(std::uint32_t opcode, std::string_view payload);

} // namespace DiscordRPC
