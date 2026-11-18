// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <limits>
#include <utility>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/artwork_cache.h"
#include "common/scm_rev.h"
#include "veshiru/util/steamgriddb_provider.h"
#ifdef VESHIRU_BUNDLED_OPENSSL
#include <openssl/cert.h>
#endif

namespace Veshiru::SteamGridDB {
namespace {

namespace Http = ArtworkHttp;
using Http::Status;
constexpr std::size_t MaxChoices = 6;
constexpr int MinCandidateScore = 300;

std::string EncodeQuery(std::string_view title) {
    constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(title.size() * 3);
    for (const unsigned char c : title) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

bool IsValidKey(std::string_view key) {
    return !key.empty() && key.size() <= 128 && std::ranges::all_of(key, [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '-';
    });
}

u32 BestMatch(const nlohmann::json& matches, std::string_view query) {
    int best_score = MinCandidateScore - 1;
    u32 best_id = 0;
    for (const auto& entry : matches) {
        if (!entry.is_object() || !entry.contains("id") || !entry["id"].is_number_unsigned() ||
            !entry.contains("name") || !entry["name"].is_string()) {
            continue;
        }
        const auto& name = entry["name"].get_ref<const std::string&>();
        const auto id = entry["id"].get<u64>();
        if (id == 0 || id > std::numeric_limits<u32>::max() ||
            !Common::ArtworkCache::IsSafeArtworkTitle(name)) {
            continue;
        }
        const auto score = Common::ArtworkCache::ScoreArtworkCandidate(name, query);
        if (score > best_score) {
            best_score = score;
            best_id = static_cast<u32>(id);
        }
    }
    return best_id;
}

void ConfigureTls(httplib::Client& client) {
    client.enable_server_certificate_verification(true);
    client.enable_server_hostname_verification(true);
#ifdef VESHIRU_BUNDLED_OPENSSL
    client.load_ca_cert_store(kCert, sizeof(kCert));
#endif
}

} // namespace

Icons FindIcons(std::string_view title, std::string_view key, const std::atomic_bool& cancelled) {
    const auto deadline = Http::Clock::now() + std::chrono::seconds{15};
    httplib::Client api{"https://www.steamgriddb.com"};
    httplib::Client cdn{"https://cdn2.steamgriddb.com"};
    ConfigureTls(api);
    ConfigureTls(cdn);
    const httplib::Headers headers{
        {"User-Agent",
         fmt::format("Veshiru/{} (https://github.com/2archiver/Veshiru)", Common::g_build_version)},
        {"Accept-Encoding", "identity"},
    };
    return FindIcons(api, cdn, headers, title, key, deadline, cancelled);
}

Icons FindIcons(httplib::Client& api, httplib::Client& cdn, const httplib::Headers& headers,
                std::string_view title, std::string_view key, Http::Clock::time_point deadline,
                const std::atomic_bool& cancelled) {
    if (!IsValidKey(key)) {
        return {Status::Unauthorized, {}};
    }
    if (!Common::ArtworkCache::IsSafeArtworkTitle(title)) {
        return {Status::Invalid, {}};
    }
    auto api_headers = headers;
    api_headers.emplace("Authorization", fmt::format("Bearer {}", key));
    const auto search = Http::Get(api, "/api/v2/search/autocomplete/" + EncodeQuery(title),
                                  api_headers, Http::Kind::Json, deadline, cancelled);
    if (search.status != Status::Ok) {
        return {search.status, {}};
    }
    const auto matches = Http::ParseData(search.body);
    if (matches.is_discarded()) {
        return {Status::Invalid, {}};
    }
    const auto id = BestMatch(matches, title);
    if (id == 0) {
        return {Status::NotFound, {}};
    }
    const auto path =
        fmt::format("/api/v2/icons/game/{}?dimensions=512,256,128&mimes=image%2Fpng&types=static"
                    "&nsfw=false&humor=false&epilepsy=false&limit={}",
                    id, MaxChoices);
    const auto list = Http::Get(api, path, api_headers, Http::Kind::Json, deadline, cancelled);
    if (list.status != Status::Ok) {
        return {list.status, {}};
    }
    const auto metadata = Http::ParseData(list.body);
    if (metadata.is_discarded()) {
        return {Status::Invalid, {}};
    }
    // The CDN receives no Authorization header, cookie, title, or local title ID.
    const httplib::Headers cdn_headers{{"Accept-Encoding", "identity"}};
    Icons result{Status::NotFound, {}};
    std::vector<std::string> attempted;
    for (const auto& entry : metadata) {
        if (!entry.is_object() || !entry.contains("url") || !entry["url"].is_string()) {
            continue;
        }
        auto asset = Http::AssetPath(entry["url"].get_ref<const std::string&>());
        if (!asset || std::ranges::find(attempted, *asset) != attempted.end()) {
            continue;
        }
        auto icon = Http::Get(cdn, *asset, cdn_headers, Http::Kind::Icon, deadline, cancelled);
        attempted.push_back(std::move(*asset));
        if (icon.status == Status::Cancelled || icon.status == Status::RateLimited) {
            return {icon.status, {}};
        }
        if (icon.status == Status::Ok) {
            result.pngs.push_back(std::move(icon.body));
            result.status = Status::Ok;
        } else if (result.pngs.empty()) {
            result.status = icon.status;
        }
        if (icon.status == Status::Timeout || attempted.size() >= MaxChoices) {
            break;
        }
    }
    return result;
}

} // namespace Veshiru::SteamGridDB
