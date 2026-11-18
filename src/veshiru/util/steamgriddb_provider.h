// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "veshiru/util/artwork_http.h"

namespace Veshiru::SteamGridDB {

struct Icons {
    ArtworkHttp::Status status;
    std::vector<std::string> pngs;
};

Icons FindIcons(std::string_view title, std::string_view key, const std::atomic_bool& cancelled);

/// Same provider protocol, with injectable clients and deadline for loopback tests.
Icons FindIcons(httplib::Client& api, httplib::Client& cdn, const httplib::Headers& headers,
                std::string_view title, std::string_view key,
                ArtworkHttp::Clock::time_point deadline, const std::atomic_bool& cancelled);

} // namespace Veshiru::SteamGridDB
