// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <charconv>
#include "common/logging.h"
#include "common/scm_rev.h"
#include "frontend_common/update_checker.h"

namespace UpdateChecker {

namespace {

std::optional<u64> ParseBuildNumber(std::string_view tag) {
    constexpr std::string_view prefix = "build-";
    if (!tag.starts_with(prefix)) {
        return std::nullopt;
    }
    tag.remove_prefix(prefix.size());
    u64 number{};
    const auto [end, error] = std::from_chars(tag.data(), tag.data() + tag.size(), number);
    if (error != std::errc{} || end != tag.data() + tag.size() || number == 0) {
        return std::nullopt;
    }
    return number;
}

// A stable, manually tagged release (vX.Y.Z) is a separate line from the build-N pre-releases:
// a build number and a semantic version cannot be ordered against each other, so a stable tag is
// only ever offered to an installed stable release.
std::optional<std::array<u64, 3>> ParseVersionTag(std::string_view tag) {
    constexpr std::string_view prefix = "v";
    if (!tag.starts_with(prefix)) {
        return std::nullopt;
    }
    tag.remove_prefix(prefix.size());
    std::array<u64, 3> parts{};
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const auto dot = tag.find('.');
        const auto field = tag.substr(0, dot);
        if (field.empty()) {
            return std::nullopt;
        }
        const auto [end, error] =
            std::from_chars(field.data(), field.data() + field.size(), parts[i]);
        if (error != std::errc{} || end != field.data() + field.size()) {
            return std::nullopt;
        }
        if (i + 1 < parts.size()) {
            if (dot == std::string_view::npos) {
                return std::nullopt;
            }
            tag.remove_prefix(dot + 1);
        } else if (dot != std::string_view::npos) {
            // A suffix after the patch field (a release candidate) is not a stable release.
            return std::nullopt;
        }
    }
    return parts;
}

bool IsNewerStable(std::span<const u64, 3> candidate, std::span<const u64, 3> installed) {
    return std::lexicographical_compare(installed.begin(), installed.end(), candidate.begin(),
                                        candidate.end());
}

} // namespace

Failure ClassifyFailure(int status) {
    if (status == 403 || status == 429) {
        return Failure::RateLimited;
    }
    return status >= 400 ? Failure::Http : Failure::Network;
}

std::optional<Common::Net::Release> SelectUpdate(std::span<const Common::Net::Release> releases,
                                                 std::string_view build_tag,
                                                 std::string_view platform,
                                                 Common::Net::LinuxPackage package) {
    if (const auto installed = ParseBuildNumber(build_tag)) {
        const Common::Net::Release* selected = nullptr;
        auto newest = *installed;
        for (const auto& release : releases) {
            const auto number = ParseBuildNumber(release.tag);
            if (number && *number > newest &&
                !release.GetPlatformAssets(platform, package).empty()) {
                newest = *number;
                selected = &release;
            }
        }
        if (selected) {
            return *selected;
        }
        // Bridge legacy build-N installations onto non-prerelease stable vX.Y.Z releases once no
        // newer build-N pre-release exists.
        std::array<u64, 3> newest_stable{};
        for (const auto& release : releases) {
            if (release.prerelease) {
                continue;
            }
            const auto version = ParseVersionTag(release.tag);
            if (version && (!selected || IsNewerStable(*version, newest_stable)) &&
                !release.GetPlatformAssets(platform, package).empty()) {
                newest_stable = *version;
                selected = &release;
            }
        }
        return selected ? std::make_optional(*selected) : std::nullopt;
    }

    if (const auto installed = ParseVersionTag(build_tag)) {
        const Common::Net::Release* selected = nullptr;
        std::array<u64, 3> newest = *installed;
        for (const auto& release : releases) {
            const auto version = ParseVersionTag(release.tag);
            if (version && IsNewerStable(*version, newest) &&
                !release.GetPlatformAssets(platform, package).empty()) {
                newest = *version;
                selected = &release;
            }
        }
        return selected ? std::make_optional(*selected) : std::nullopt;
    }

std::vector<std::string> RankUpdates(std::span<const Common::Net::Tag> tags,
                                     std::string_view current_tag, std::string_view current_commit,
                                     Channel channel) {
    const auto current_channel = ChannelOfTag(current_tag);
    if (!current_channel) {
        return {};
    }
    const bool to_daily = channel == Channel::Daily;
    const bool switching = to_daily != (*current_channel == Channel::Daily);

    std::vector<std::string> ranked;
    if (to_daily) {
        const auto current_date = switching ? std::nullopt : ParseDailyTag(current_tag);
        std::vector<std::pair<u32, std::string>> found;
        for (const auto& tag : tags) {
            const auto date = ParseDailyTag(tag.name);
            if (!date) {
                continue;
            }
            if (current_date) {
                // An unknown commit on either side is not taken for a rebuild.
                const bool rebuilt =
                    !tag.commit.empty() && !current_commit.empty() && tag.commit != current_commit;
                if (*date < *current_date || (*date == *current_date && !rebuilt)) {
                    continue;
                }
            }
            found.emplace_back(*date, tag.name);
        }
        std::ranges::stable_sort(found, std::greater{}, &std::pair<u32, std::string>::first);
        for (auto& [date, name] : found) {
            ranked.push_back(std::move(name));
        }
    } else {
        const auto current_version = switching ? std::nullopt : ParseTag(current_tag);
        std::vector<std::pair<Version, std::string>> found;
        for (const auto& tag : tags) {
            const auto version = ParseTag(tag.name);
            if (!version || (current_version && *version <= *current_version)) {
                continue;
            }
            if (channel == Channel::Stable && version->rc != 0) {
                continue;
            }
            found.emplace_back(*version, tag.name);
        }
        std::ranges::stable_sort(found, std::greater{}, &std::pair<Version, std::string>::first);
        for (auto& [version, name] : found) {
            ranked.push_back(std::move(name));
        }
    }
    return ranked;
}

namespace {

bool HasAsset(const Common::Net::Release& release, std::string_view platform,
              Common::Net::LinuxPackage package) {
    return platform.empty() || !release.GetPlatformAssets(platform, package).empty();
}

} // namespace

std::optional<Common::Net::Release> SelectUpdate(std::span<const Common::Net::Release> releases,
                                                 std::string_view current_tag,
                                                 std::string_view current_commit, Channel channel,
                                                 std::string_view platform,
                                                 Common::Net::LinuxPackage package) {
    std::vector<Common::Net::Tag> tags;
    for (const auto& release : releases) {
        tags.push_back({.name = release.tag, .commit = release.commit});
    }
    for (const auto& name : RankUpdates(tags, current_tag, current_commit, channel)) {
        const auto release = std::ranges::find(releases, name, &Common::Net::Release::tag);
        if (HasAsset(*release, platform, package)) {
            return *release;
        }
    }
    return std::nullopt;
}

CheckResult Check() {
    const bool tracked_build = ParseBuildNumber(Common::g_build_release_tag).has_value() ||
                               ParseVersionTag(Common::g_build_release_tag).has_value();
    if (!tracked_build || std::string_view{Common::g_build_release_platform}.empty()) {
        return {};
    }

    const auto list = Common::Net::GetReleases();
    if (!list.ok) {
        const auto failure = ClassifyFailure(list.status);
        LOG_WARNING(Frontend, "Update check failed with status {}", list.status);
        return {.failure = failure, .status = list.status};
    }

    auto update = SelectUpdate(list.releases, Common::g_build_release_tag,
                               Common::g_build_release_platform,
                               Common::Net::CurrentLinuxPackage());
    if (update) {
        Common::Net::PopulateMissingAssetDigests(*update);
        LOG_INFO(Frontend, "Received update {}", update->title);
    }
    return {.update = std::move(update)};
}

std::optional<Common::Net::Release> GetUpdate() {
    return Check().update;
}

} // namespace UpdateChecker
