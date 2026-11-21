// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qt_common/game_list/game_version.h"

#include <optional>

#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/loader/loader.h"

namespace GameListVersion {
namespace {

/// The version of the enabled update in a FormatPatchNameVersions line list: a line that
/// starts with "Update (" and ends with ")". A disabled patch is prefixed "[D] " and does not
/// decide the version the player runs.
std::optional<std::string> EnabledUpdateVersion(std::string_view lines) {
    std::size_t begin = 0;
    while (begin < lines.size()) {
        const std::size_t end = lines.find('\n', begin);
        const std::string_view line =
            lines.substr(begin, end == std::string_view::npos ? std::string_view::npos
                                                             : end - begin);
        if (line.starts_with("Update (") && line.ends_with(")") && line.size() > 9) {
            return std::string{line.substr(8, line.size() - 9)};
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return std::nullopt;
}

} // namespace

std::string Resolve(const Inputs& inputs) {
    if (const auto update = EnabledUpdateVersion(inputs.patch_versions)) {
        return *update;
    }
    if (!inputs.control_metadata_version.empty()) {
        return inputs.control_metadata_version;
    }
    if (!inputs.display_version.empty()) {
        return inputs.display_version;
    }
    // A base control NACP exists but names no version: Switch games without one report 1.0.0.
    // With no control data at all (a loose dump, an unreadable container) nothing is known.
    if (inputs.control_metadata_exists) {
        return "1.0.0";
    }
    return {};
}

std::string Resolve(std::string_view patch_versions, const FileSys::PatchManager& patch_manager,
                    Loader::AppLoader& loader) {
    const auto metadata = patch_manager.GetControlMetadata();
    std::string display_version;
    if (loader.ReadDisplayVersion(display_version) != Loader::ResultStatus::Success) {
        display_version.clear();
    }
    return Resolve(Inputs{
        .patch_versions = patch_versions,
        .control_metadata_version =
            metadata.first != nullptr ? metadata.first->GetVersionString() : "",
        .control_metadata_exists = metadata.first != nullptr,
        .display_version = std::move(display_version),
    });
}

} // namespace GameListVersion
