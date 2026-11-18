// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <charconv>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "veshiru/util/artwork_http.h"

namespace Veshiru::ArtworkHttp {
namespace {

constexpr std::size_t MaxJsonBytes = 128 * 1024;
constexpr std::size_t MaxImageBytes = 2 * 1024 * 1024;

bool ValidHeader(const httplib::Response& response, Kind kind, std::size_t limit) {
    const auto encoding = response.get_header_value("Content-Encoding");
    if (!encoding.empty() && encoding != "identity") {
        return false;
    }
    const auto length_header = response.get_header_value("Content-Length");
    if (!length_header.empty()) {
        std::size_t length = 0;
        const auto* end = length_header.data() + length_header.size();
        const auto [ptr, ec] = std::from_chars(length_header.data(), end, length);
        if (ec != std::errc{} || ptr != end || length == 0 || length > limit) {
            return false;
        }
    }
    const auto type = response.get_header_value("Content-Type");
    const auto media = std::string_view{type}.substr(0, type.find(';'));
    return kind == Kind::Json ? media == "application/json" : media == "image/png";
}

} // namespace

bool SearchGate::TryBegin(Clock::time_point now) {
    std::scoped_lock lock{mutex};
    return now >= retry_after && !std::exchange(running, true);
}

void SearchGate::End(Clock::time_point now, bool rate_limited) {
    std::scoped_lock lock{mutex};
    retry_after = now + std::chrono::seconds{rate_limited ? 60 : 5};
    running = false;
}

Result Get(httplib::Client& client, const std::string& path, const httplib::Headers& headers,
           Kind kind, Clock::time_point deadline, const std::atomic_bool& cancelled) {
    using namespace std::chrono;
    const auto remaining = ceil<milliseconds>(deadline - Clock::now());
    if (cancelled || remaining.count() <= 0) {
        return {cancelled ? Status::Cancelled : Status::Timeout, {}};
    }
    const auto limit = kind == Kind::Json ? MaxJsonBytes : MaxImageBytes;
    client.set_follow_location(false);
    client.set_decompress(false);
    client.set_payload_max_length(limit);
    client.set_max_timeout(remaining);
    client.set_connection_timeout(std::min(remaining, milliseconds{3000}));
    client.set_read_timeout(std::min(remaining, milliseconds{3000}));
    client.set_write_timeout(std::min(remaining, milliseconds{3000}));
    client.set_path_encode(false);

    // Cancel blocked socket reads (including before headers), not just the next body chunk.
    std::condition_variable_any changed;
    std::mutex mutex;
    const std::jthread watchdog([&](std::stop_token token) {
        std::unique_lock lock(mutex);
        while (!token.stop_requested()) {
            if (cancelled || Clock::now() >= deadline) {
                client.stop();
                return;
            }
            changed.wait_until(lock, token, std::min(deadline, Clock::now() + milliseconds{50}),
                               [] { return false; });
        }
    });

    Status status = Status::Failed;
    std::string body;
    const auto response = client.Get(
        path, headers,
        [&](const httplib::Response& header) {
            switch (header.status) {
            case 200:
                status = ValidHeader(header, kind, limit) ? Status::Ok : Status::Invalid;
                return status == Status::Ok;
            case 401:
            case 403:
                status = Status::Unauthorized;
                break;
            case 404:
                status = Status::NotFound;
                break;
            case 429:
                status = Status::RateLimited;
                break;
            default:
                break;
            }
            return false; // Never follow redirects or receive an error body.
        },
        [&](const char* data, std::size_t size) {
            if (size > limit - body.size()) {
                status = Status::Invalid;
                return false;
            }
            if (cancelled || Clock::now() >= deadline) {
                return false;
            }
            body.append(data, size);
            return true;
        });
    if (cancelled || Clock::now() >= deadline) {
        return {cancelled ? Status::Cancelled : Status::Timeout, {}};
    }
    if (status == Status::Ok) {
        if (!response || body.empty()) {
            return {Status::Invalid, {}};
        }
        return {Status::Ok, std::move(body)};
    }
    return {status, {}};
}

std::optional<std::string> AssetPath(std::string_view url) {
    constexpr std::string_view cdn = "https://cdn2.steamgriddb.com/";
    if (url.size() > 512 || !url.starts_with(cdn)) {
        return std::nullopt;
    }
    const auto path = url.substr(cdn.size() - 1);
    const auto file = path.substr(1);
    if (file.empty() || file.find("..") != file.npos || !file.ends_with(".png")) {
        return std::nullopt;
    }
    const bool clean = std::ranges::all_of(file, [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '-' || c == '_' || c == '.' || c == '/';
    });
    return clean && !file.starts_with('/') ? std::optional{std::string{path}} : std::nullopt;
}

nlohmann::json ParseData(std::string_view body) {
    struct TooDeep {};
    if (body.size() <= MaxJsonBytes) {
        try {
            auto json = nlohmann::json::parse(
                body,
                [](int depth, nlohmann::json::parse_event_t, nlohmann::json&) {
                    if (depth > 12) {
                        throw TooDeep{};
                    }
                    return true;
                },
                false);
            if (json.is_object() && json.value("success", nlohmann::json{}) == true &&
                json.contains("data") && json["data"].is_array() && json["data"].size() <= 128) {
                return std::move(json["data"]);
            }
        } catch (const TooDeep&) {
        }
    }
    return nlohmann::json::value_t::discarded;
}

} // namespace Veshiru::ArtworkHttp
