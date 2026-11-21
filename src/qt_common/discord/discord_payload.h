// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace DiscordRPC {

inline constexpr char VeshiruPresenceName[] = "Veshiru 2.3.0";

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

PresencePayload BuildPresencePayload(const ActivityData& activity);
std::string BuildHandshakePayload(std::string_view application_id);
std::string BuildSetActivityPayload(const ActivityData& activity, std::uint32_t process_id,
                                    std::string_view nonce);
std::string BuildClearActivityPayload(std::uint32_t process_id, std::string_view nonce);

} // namespace DiscordRPC
