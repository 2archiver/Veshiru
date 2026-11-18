// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json_fwd.hpp>
#include "common/httplib.h"

namespace Veshiru::ArtworkHttp {

using Clock = std::chrono::steady_clock;
enum class Kind { Json, Icon };
enum class Status { Ok, NotFound, Unauthorized, RateLimited, Cancelled, Timeout, Invalid, Failed };
struct Result {
    Status status;
    std::string body;
};

class SearchGate {
public:
    bool TryBegin(Clock::time_point now);
    void End(Clock::time_point now, bool rate_limited);

private:
    std::mutex mutex;
    bool running{};
    Clock::time_point retry_after{Clock::time_point::min()};
};

/// Bounded transport for one GET. Production callers choose the HTTPS host before passing a client.
Result Get(httplib::Client& client, const std::string& path, const httplib::Headers& headers,
           Kind kind, Clock::time_point deadline, const std::atomic_bool& cancelled);

/// Accept only direct PNG asset URLs at SteamGridDB's dedicated CDN, never API credentials.
std::optional<std::string> AssetPath(std::string_view url);

/// Returns a validated SGDB data array or a discarded JSON value.
nlohmann::json ParseData(std::string_view body);

} // namespace Veshiru::ArtworkHttp
