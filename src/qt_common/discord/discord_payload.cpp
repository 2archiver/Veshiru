// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qt_common/discord/discord_payload.h"

#include <string>

namespace DiscordRPC {
namespace {

std::string EscapeJson(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size() + 2);
    for (const unsigned char c : value) {
        switch (c) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                escaped += "\\u00";
                escaped += hex[c >> 4];
                escaped += hex[c & 0x0f];
            } else {
                escaped += static_cast<char>(c);
            }
            break;
        }
    }
    return escaped;
}

} // namespace

PresencePayload BuildPresencePayload(const ActivityData& activity) {
    if (activity.browsing) {
        return {.details = "Browsing library", .state = VeshiruPresenceName};
    }
    return {
        .details = "Playing " + activity.game_title,
        .state = std::string{VeshiruPresenceName} + " · Version " + activity.game_version,
        .started_at = activity.started_at,
    };
}

std::string BuildHandshakePayload(std::string_view application_id) {
    return "{\"v\":1,\"client_id\":\"" + EscapeJson(application_id) + "\"}";
}

std::string BuildSetActivityPayload(const ActivityData& activity, std::uint32_t process_id,
                                    std::string_view nonce) {
    const PresencePayload presence = BuildPresencePayload(activity);
    std::string payload = "{\"cmd\":\"SET_ACTIVITY\",\"args\":{\"pid\":" +
                          std::to_string(process_id) + ",\"activity\":{\"type\":0,\"details\":\"" +
                          EscapeJson(presence.details) + "\",\"state\":\"" +
                          EscapeJson(presence.state) + "\"";
    if (presence.started_at) {
        payload += ",\"timestamps\":{\"start\":" +
                   std::to_string(*presence.started_at) + "}";
    }
    payload += "}},\"nonce\":\"" + EscapeJson(nonce) + "\"}";
    return payload;
}

std::string BuildClearActivityPayload(std::uint32_t process_id, std::string_view nonce) {
    return "{\"cmd\":\"SET_ACTIVITY\",\"args\":{\"pid\":" +
           std::to_string(process_id) +
           ",\"activity\":null},\"nonce\":\"" + EscapeJson(nonce) + "\"}";
}

} // namespace DiscordRPC
