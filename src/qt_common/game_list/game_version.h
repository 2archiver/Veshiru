// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <string_view>

namespace FileSys {
class PatchManager;
}

namespace Loader {
class AppLoader;
}

namespace GameListVersion {

/// The inputs the version can come from, so the resolution is testable without a content
/// provider. Fill them from a PatchManager and an AppLoader with the gather overload.
struct Inputs {
    /// A FormatPatchNameVersions line list: the enabled update appears as "Update (x.y.z)".
    std::string_view patch_versions;
    /// PatchManager::GetControlMetadata()->GetVersionString(); the base's own control data.
    std::string control_metadata_version;
    /// Whether GetControlMetadata returned a control NACP at all.
    bool control_metadata_exists{};
    /// Loader::ReadDisplayVersion.
    std::string display_version;
};

/// The version the player runs: the enabled update's if one is installed, else the base's
/// control metadata version, else the loader's display version, and "1.0.0" only when a base
/// control NACP exists. Empty when nothing is known: hide the badge rather than guess.
std::string Resolve(const Inputs& inputs);

std::string Resolve(std::string_view patch_versions, const FileSys::PatchManager& patch_manager,
                    Loader::AppLoader& loader);

} // namespace GameListVersion
