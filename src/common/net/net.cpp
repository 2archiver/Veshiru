// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <boost/algorithm/string/replace.hpp>

#include <fmt/format.h>
#include "common/scm_rev.h"
#include "net.h"

#include "common/logging.h"

#include "common/httplib.h"

#ifdef VESHIRU_BUNDLED_OPENSSL
#include <openssl/cert.h>
#endif

namespace Common::Net {

LinuxPackage CurrentLinuxPackage() {
#ifdef __linux__
    const char* appimage = std::getenv("APPIMAGE");
    return appimage && *appimage ? LinuxPackage::AppImage : LinuxPackage::Archive;
#else
    return LinuxPackage::Any;
#endif
}

std::vector<Asset> Release::GetPlatformAssets() const {
    return GetPlatformAssets(Common::g_build_release_platform, CurrentLinuxPackage());
}

std::vector<Asset> Release::GetPlatformAssets(std::string_view platform,
                                              LinuxPackage package) const {
    const bool pgo = platform.ends_with("-pgo");
    if (pgo) {
        platform.remove_suffix(4);
    }

    std::vector<std::string_view> extensions;
    if (platform == "linux-x86_64" || platform == "linux-aarch64") {
        if (package != LinuxPackage::Archive) {
            extensions.push_back(".AppImage");
        }
        if (package != LinuxPackage::AppImage) {
            extensions.push_back(".tar.gz");
        }
    } else if (platform == "windows-x86_64" || (platform == "macos-arm64" && !pgo)) {
        extensions = {".zip"};
    } else {
        return {};
    }

    const std::string host_url = "https://github.com";
    const auto path = fmt::format("/{}/releases/download/{}/", Common::g_build_auto_update_repo, tag);
    for (const std::string_view variant : pgo ? std::vector<std::string_view>{"-pgo", ""}
                                              : std::vector<std::string_view>{""}) {
        for (const auto extension : extensions) {
            const auto filename = fmt::format("veshiru-{}{}{}", platform, variant, extension);
            const auto url = host_url + path + filename;
            if (std::ranges::find(assets, url) == assets.end()) {
                continue;
            }
            Asset asset{.name = variant.empty() ? "Standard" : "PGO",
                        .url = host_url,
                        .path = path + filename,
                        .filename = filename};
            if (const auto info = asset_info.find(url); info != asset_info.end()) {
                asset.sha256 = info->second.sha256;
                asset.size = info->second.size;
            }
            return {std::move(asset)};
        }
    }
    return {};
}

static inline u64 ParseIsoTimestamp(const std::string& iso) {
    if (iso.empty())
        return 0;

    std::string buf = iso;
    if (buf.back() == 'Z')
        buf.pop_back();

    std::tm tm{};
    std::istringstream ss(buf);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail())
        return 0;

#ifdef _WIN32
    return static_cast<u64>(_mkgmtime(&tm));
#else
    return static_cast<u64>(timegm(&tm));
#endif
}

std::optional<Release> Release::FromJson(const nlohmann::json& json, const std::string& host,
                                         const std::string& repo) {
    Release rel;
    if (!json.is_object())
        return std::nullopt;
    if (json.value("draft", false))
        return std::nullopt;

    rel.tag = json.value("tag_name", std::string{});
    if (rel.tag.empty())
        return std::nullopt;

    rel.title = json.contains("name") && json["name"].is_string()
                    ? json["name"].get<std::string>() : rel.tag;
    rel.id = json.value("id", std::hash<std::string>{}(rel.title));

    rel.published = ParseIsoTimestamp(json.value("published_at", std::string{}));
    rel.prerelease = json.value("prerelease", false);

    auto body = json.contains("body") && json["body"].is_string()
                    ? json["body"].get<std::string>() : rel.title;
    boost::replace_all(body, "\\r", "");
    boost::replace_all(body, "\\n", "\n");
    rel.body = body;

    rel.host = host;

    const auto release_base =
        fmt::format("{}/{}/releases", Common::g_build_auto_update_website, repo);
    const auto fallback_html = fmt::format("{}/tag/{}", release_base, rel.tag);
    rel.html_url = json.value("html_url", fallback_html);

    rel.base_download_url = fmt::format("/{}/releases/download/{}", repo, rel.tag);
    const auto assets = json.value("assets", nlohmann::json::array());
    if (!assets.is_array()) {
        return std::nullopt;
    }
    for (const auto& asset : assets) {
        const auto url = asset.value("browser_download_url", std::string{});
        rel.assets.emplace_back(url);

        AssetInfo info;
        if (asset.contains("size") && asset["size"].is_number_unsigned()) {
            info.size = asset["size"].get<u64>();
        }
        constexpr std::string_view digest_prefix = "sha256:";
        if (asset.contains("digest") && asset["digest"].is_string()) {
            const auto digest = asset["digest"].get<std::string>();
            const std::string_view hex =
                digest.starts_with(digest_prefix)
                    ? std::string_view{digest}.substr(digest_prefix.size())
                    : std::string_view{};
            if (hex.size() == 64 &&
                std::ranges::all_of(hex, [](unsigned char c) { return std::isxdigit(c); })) {
                info.sha256 = std::string{hex};
                std::ranges::transform(info.sha256, info.sha256.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
            }
        }
        rel.asset_info.emplace(url, std::move(info));
    }

    return rel;
}

std::optional<Release> Release::FromJson(const std::string_view& json, const std::string& host,
                                         const std::string& repo) {
    try {
        return FromJson(nlohmann::json::parse(json), host, repo);
    } catch (std::exception& e) {
        LOG_WARNING(Common, "Failed to parse JSON: {}", e.what());
    }

    return {};
}

std::vector<Release> Release::ListFromJson(const nlohmann::json& json, const std::string& host,
                                           const std::string& repo) {
    if (!json.is_array())
        return {};

    std::vector<Release> releases;
    for (const auto& obj : json) {
        auto rel = Release::FromJson(obj, host, repo);
        if (rel)
            releases.emplace_back(rel.value());
    }
    return releases;
}

std::vector<Release> Release::ListFromJson(const std::string_view& json, const std::string& host,
                                           const std::string& repo) {
    try {
        return ListFromJson(nlohmann::json::parse(json), host, repo);
    } catch (std::exception& e) {
        LOG_WARNING(Common, "Failed to parse JSON: {}", e.what());
    }

    return {};
}

RequestResult Fetch(const std::string& url, const std::string& path) {
    try {
        constexpr std::size_t timeout_seconds = 15;

        std::unique_ptr<httplib::Client> client = std::make_unique<httplib::Client>(url);
        client->set_connection_timeout(timeout_seconds);
        client->set_read_timeout(timeout_seconds);
        client->set_write_timeout(timeout_seconds);

#ifdef VESHIRU_BUNDLED_OPENSSL
        client->load_ca_cert_store(kCert, sizeof(kCert));
#endif

        if (client == nullptr) {
            LOG_ERROR(Common, "Invalid URL {}{}", url, path);
            return {};
        }

        httplib::Request request{
            .method = "GET",
            .path = path,
            .headers = {{"Accept", "application/vnd.github+json"}, {"User-Agent", "Veshiru"}},
        };

        client->set_follow_location(true);
        httplib::Result result = client->send(request);

        if (!result) {
            LOG_ERROR(Common, "GET to {}{} returned null", url, path);
            return {};
        }

        const auto& response = result.value();
        if (response.status >= 400) {
            LOG_ERROR(Common, "GET to {}{} returned error status code: {}", url, path,
                      response.status);
            return {.status = response.status};
        }
        if (!response.has_header("content-type")) {
            LOG_ERROR(Common, "GET to {}{} returned no content", url, path);
            return {.status = response.status};
        }

        return {.body = response.body, .status = response.status};
    } catch (std::exception& e) {
        LOG_ERROR(Common, "GET to {}{} failed during update check: {}", url, path, e.what());
        return {};
    }
}

std::optional<std::string> MakeRequest(const std::string& url, const std::string& path) {
    return Fetch(url, path).body;
}

ReleaseList GetReleases() {
    const auto url = fmt::format("https://{}", Common::g_build_auto_update_stable_api);
    const auto response = Fetch(url, Common::g_build_auto_update_api_path);

    if (!response.body) {
        LOG_WARNING(Common, "Failed to get Veshiru releases");
        return {.status = response.status};
    }

    return {.releases = Release::ListFromJson(std::string_view{*response.body}, url,
                                              Common::g_build_auto_update_stable_repo),
            .ok = true,
            .status = response.status};
}

std::optional<std::string> GetReleasesBody() {
    const auto releases_path = Common::g_build_auto_update_api_path;
    const auto url = fmt::format("https://{}", Common::g_build_auto_update_stable_api);

    return MakeRequest(url, releases_path);
}

void ApplySha256SumsText(Release& release, std::string_view sums_text) {
    std::unordered_map<std::string, std::string> by_filename;
    std::size_t pos = 0;
    while (pos < sums_text.size()) {
        const std::size_t end = sums_text.find('\n', pos);
        std::string_view line =
            end == std::string_view::npos ? sums_text.substr(pos) : sums_text.substr(pos, end - pos);
        pos = end == std::string_view::npos ? sums_text.size() : end + 1;
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.size() < 66) {
            continue;
        }
        const std::string_view hex = line.substr(0, 64);
        if (!std::ranges::all_of(hex, [](unsigned char c) { return std::isxdigit(c); })) {
            continue;
        }
        std::size_t name_start = 64;
        while (name_start < line.size() &&
               (line[name_start] == ' ' || line[name_start] == '\t' || line[name_start] == '*')) {
            ++name_start;
        }
        if (name_start >= line.size()) {
            continue;
        }
        std::string_view name = line.substr(name_start);
        if (const std::size_t slash = name.find_last_of("/\\"); slash != std::string_view::npos) {
            name = name.substr(slash + 1);
        }
        if (name.empty()) {
            continue;
        }
        std::string lower_hex{hex};
        std::ranges::transform(lower_hex, lower_hex.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        by_filename.emplace(std::string{name}, std::move(lower_hex));
    }

    if (by_filename.empty()) {
        return;
    }

    for (const auto& url : release.assets) {
        const std::size_t slash = url.rfind('/');
        if (slash == std::string::npos || slash + 1 >= url.size()) {
            continue;
        }
        const std::string filename = url.substr(slash + 1);
        auto& info = release.asset_info[url];
        if (!info.sha256.empty()) {
            continue;
        }
        if (const auto it = by_filename.find(filename); it != by_filename.end()) {
            info.sha256 = it->second;
        }
    }
}

void PopulateMissingAssetDigests(Release& release) {
    bool any_missing = false;
    std::string sums_url;
    for (const auto& url : release.assets) {
        const std::size_t slash = url.rfind('/');
        const std::string_view filename =
            slash == std::string::npos ? std::string_view{url}
                                       : std::string_view{url}.substr(slash + 1);
        if (filename == "SHA256SUMS.txt") {
            sums_url = url;
            continue;
        }
        const auto it = release.asset_info.find(url);
        if (it == release.asset_info.end() || it->second.sha256.empty()) {
            any_missing = true;
        }
    }
    if (!any_missing || sums_url.empty()) {
        return;
    }

    const std::size_t scheme_end = sums_url.find("://");
    if (scheme_end == std::string::npos) {
        return;
    }
    const std::size_t path_start = sums_url.find('/', scheme_end + 3);
    if (path_start == std::string::npos) {
        return;
    }
    const std::string host = sums_url.substr(0, path_start);
    const std::string path = sums_url.substr(path_start);
    if (const auto body = MakeRequest(host, path)) {
        ApplySha256SumsText(release, *body);
    }
}

} // namespace Common::Net
