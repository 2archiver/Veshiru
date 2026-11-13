// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/loader/nup.h"

#include <algorithm>
#include <cstring>
#include <string_view>

#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/deconstructed_rom_directory.h"

namespace Loader {

namespace {

using FileSys::NupContainer;

// NACP stores 16 uncompressed language entries; map BCP-47 tags onto them.
struct LanguageSlot {
    std::string_view tag;
    std::size_t index;
};

constexpr std::array<LanguageSlot, 18> LanguageSlots{{
    {"en-us", 0},   {"en-gb", 1},  {"ja", 2},     {"fr", 3},     {"de", 4},
    {"es-419", 5},  {"es", 6},     {"it", 7},     {"nl", 8},     {"fr-ca", 9},
    {"pt", 10},     {"ru", 11},    {"ko", 12},    {"zh-hant", 13}, {"zh-hans", 14},
    {"pt-br", 15},  {"pl", 16},    {"th", 17},
}};

void CopyNameToEntry(FileSys::LanguageEntry& entry, std::string_view name) {
    const std::size_t length = std::min(name.size(), entry.application_name.size() - 1);
    std::memcpy(entry.application_name.data(), name.data(), length);
    entry.application_name[length] = '\0';
}

void CopyDeveloperToEntry(FileSys::LanguageEntry& entry, std::string_view developer) {
    const std::size_t length = std::min(developer.size(), entry.developer_name.size() - 1);
    std::memcpy(entry.developer_name.data(), developer.data(), length);
    entry.developer_name[length] = '\0';
}

std::unique_ptr<FileSys::NACP> BuildNacp(const NupContainer::PackageInfo& base,
                                         const std::string& version) {
    const std::string& effective_title =
        !base.title_name.empty() ? base.title_name : base.source_dump;
    if (effective_title.empty() && base.localized_titles.empty() && base.publisher.empty() &&
        version.empty()) {
        return nullptr;
    }
    auto nacp = std::make_unique<FileSys::NACP>();
    nacp->raw.supported_languages = static_cast<FileSys::SupportedLanguage>(0x3FFFF);

    for (const auto& [lang, name] : base.localized_titles) {
        for (const auto& slot : LanguageSlots) {
            // The uncompressed RawNACP only stores 16 entries; Polish/Thai exist
            // only in the compressed format, so they fall back to the default.
            if (slot.index < 16 && slot.tag == lang) {
                CopyNameToEntry(nacp->raw.language_entries.language_entries[slot.index], name);
            }
        }
    }
    // Slot 0 is en-us; an exact en-us localized title wins over the default title (SPEC 5.1).
    auto& english = nacp->raw.language_entries.language_entries[0];
    if (!effective_title.empty() && english.application_name[0] == '\0') {
        CopyNameToEntry(english, effective_title);
    }
    if (!base.publisher.empty()) {
        for (auto& entry : nacp->raw.language_entries.language_entries) {
            CopyDeveloperToEntry(entry, base.publisher);
        }
    }
    if (!version.empty()) {
        const std::size_t length = std::min(version.size(), nacp->raw.version_string.size() - 1);
        std::memcpy(nacp->raw.version_string.data(), version.data(), length);
        nacp->raw.version_string[length] = '\0';
    }
    nacp->raw.save_data_owner_id = base.title_id;

    nacp->language_entries.resize(16);
    std::memcpy(nacp->language_entries.data(), nacp->raw.language_entries.language_entries.data(),
                sizeof(nacp->raw.language_entries.language_entries));
    return nacp;
}

std::string DecodeVersion(u32 version) {
    if (version == 0) {
        return {};
    }
    return fmt::format("{}.{}.{}", version >> 16, (version >> 8) & 0xFF, version & 0xFF);
}

bool IsAddonDisabled(u64 title_id, std::string_view name) {
    const auto it = Settings::values.disabled_addons.find(title_id);
    return it != Settings::values.disabled_addons.end() &&
           std::find(it->second.begin(), it->second.end(), name) != it->second.end();
}

bool IsUpdateDisabled(u64 title_id, u32 version) {
    return IsAddonDisabled(title_id, "Update") ||
           IsAddonDisabled(title_id, fmt::format("Update@{}", version));
}

// An update only borrows a base with the same application title ID. Peek package records first;
// do not decompress a directory full of unrelated containers while opening or refreshing a game.
std::unique_ptr<NupContainer> FindCompanionBase(const FileSys::VirtualFile& update_file,
                                                u64 update_title_id,
                                                std::optional<std::size_t>& out_index) {
    out_index.reset();
    if (update_file == nullptr) {
        return nullptr;
    }
    const auto directory = update_file->GetContainingDirectory();
    if (directory == nullptr) {
        return nullptr;
    }
    auto candidates = directory->GetFiles();
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a->GetName() < b->GetName();
    });
    const auto update_base_id = FileSys::GetBaseTitleID(update_title_id);
    for (const auto& candidate : candidates) {
        if (candidate == nullptr || candidate->GetName() == update_file->GetName() ||
            Common::ToLower(candidate->GetExtension()) != "nup") {
            continue;
        }
        const auto peeked = NupContainer::PeekBaseTitleId(candidate);
        if (!peeked || FileSys::GetBaseTitleID(*peeked) != update_base_id) {
            continue;
        }
        auto container = std::make_unique<NupContainer>(candidate);
        if (container->GetStatus() != ResultStatus::Success) {
            continue;
        }
        const auto index = container->FindBasePackage();
        if (!index || FileSys::GetBaseTitleID(container->GetPackages()[*index].title_id) !=
                          update_base_id || container->GetExeFSDir(*index) == nullptr) {
            continue;
        }
        out_index = index;
        return container;
    }
    return nullptr;
}

} // namespace

AppLoader_NUP::AppLoader_NUP(FileSys::VirtualFile file_)
    : AppLoader(std::move(file_)), nup(std::make_unique<NupContainer>(file)) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return;
    }

    const auto base_index = nup->FindBasePackage();
    auto update_index = nup->FindBestUpdatePackage();
    if (base_index && update_index && nup->GetExeFSDir(*base_index) != nullptr &&
        IsUpdateDisabled(nup->GetPackages()[*base_index].title_id,
                         nup->GetPackages()[*update_index].version)) {
        LOG_INFO(Loader, "NUP update is disabled in the game's add-ons; booting the base");
        update_index.reset();
    }
    selected_update_index = update_index;

    // The program identity comes from the base. A complete standalone update can boot alone;
    // a patch-only update needs its matching base, which may be a sibling .nup.
    std::optional<std::size_t> program_index;
    std::optional<std::size_t> exefs_index;
    bool program_from_base_nup = false;
    if (base_index && nup->GetExeFSDir(*base_index) != nullptr) {
        program_index = base_index;
        exefs_index = update_index && nup->GetExeFSDir(*update_index) != nullptr
                          ? update_index
                          : base_index;
    } else if (update_index && nup->GetExeFSDir(*update_index) != nullptr) {
        const auto& update = nup->GetPackages()[*update_index];
        base_nup = FindCompanionBase(file, update.title_id, base_nup_index);
        if (base_nup && base_nup_index) {
            LOG_INFO(Loader, "NUP update uses companion base {:016X}",
                     base_nup->GetPackages()[*base_nup_index].title_id);
            program_index = base_nup_index;
            program_from_base_nup = true;
            exefs_index = update_index;
        } else {
            LOG_WARNING(Loader,
                        "NUP update {:016X} has no matching base in this folder; trying its own "
                        "program tree. A patch-only update needs the base game's .nup beside it.",
                        update.title_id);
            program_index = update_index;
            exefs_index = update_index;
        }
    }
    if (!program_index || !exefs_index) {
        LOG_ERROR(Loader, "NUP container has no package with a usable ExeFS");
        return;
    }

    const auto& program = program_from_base_nup ? base_nup->GetPackages()[*base_nup_index]
                                                : nup->GetPackages()[*program_index];
    base_title_id = program.title_id;

    NupContainer::PackageInfo meta;
    meta.title_id = program.title_id;
    meta.title_name = program.title_name;
    meta.localized_titles = program.localized_titles;
    meta.source_dump = program.source_dump;
    meta.publisher = program.publisher;
    meta.icon = program.icon;
    meta.localized_icons = program.localized_icons;
    if (base_index && *base_index != *program_index) {
        const auto& base = nup->GetPackages()[*base_index];
        if (!base.title_name.empty()) {
            meta.title_name = base.title_name;
        }
        if (!base.localized_titles.empty()) {
            meta.localized_titles = base.localized_titles;
        }
        if (!base.source_dump.empty()) {
            meta.source_dump = base.source_dump;
        }
        if (!base.publisher.empty()) {
            meta.publisher = base.publisher;
        }
        if (!base.icon.empty() || !base.localized_icons.empty()) {
            meta.icon = base.icon;
            meta.localized_icons = base.localized_icons;
        }
    }
    display_title = meta.title_name;
    for (const auto& [lang, name] : meta.localized_titles) {
        if (lang == "en-us" && !name.empty()) {
            display_title = name;
        }
    }
    has_program = true;

    const auto exefs = nup->GetExeFSDir(*exefs_index);
    if (exefs == nullptr || exefs->GetFile("main") == nullptr ||
        exefs->GetFile("main.npdm") == nullptr) {
        LOG_ERROR(Loader, "NUP package has no usable ExeFS");
        return;
    }
    secondary_loader = std::make_unique<AppLoader_DeconstructedRomDirectory>(exefs, true, false);

    if (update_index) {
        const auto& update = nup->GetPackages()[*update_index];
        display_version = update.display_version.empty() ? DecodeVersion(update.version)
                                                         : update.display_version;
    } else {
        display_version = program.display_version.empty() ? DecodeVersion(program.version)
                                                          : program.display_version;
    }
    nacp_file = BuildNacp(meta, display_version);
    if (!meta.icon.empty()) {
        icon_file = std::make_shared<FileSys::VectorVfsFile>(meta.icon, "icon.dat");
    } else {
        for (const auto& [lang, icon] : meta.localized_icons) {
            if (lang == "en-us") {
                icon_file = std::make_shared<FileSys::VectorVfsFile>(icon, "icon_" + lang + ".dat");
            }
        }
        if (icon_file == nullptr && !meta.localized_icons.empty()) {
            icon_file = std::make_shared<FileSys::VectorVfsFile>(
                meta.localized_icons.front().second, "icon.dat");
        }
    }

    const auto addon_count = std::count_if(nup->GetPackages().begin(), nup->GetPackages().end(),
                                           [](const auto& package) {
                                               return package.kind ==
                                                      NupContainer::PackageKind::Addon;
                                           });
    if (addon_count > 0) {
        LOG_INFO(Loader, "NUP container carries {} add-on content package(s)", addon_count);
    }
}

AppLoader_NUP::~AppLoader_NUP() = default;

FileType AppLoader_NUP::IdentifyType(const FileSys::VirtualFile& nup_file) {
    return FileSys::NupContainer::IsNupFile(nup_file) ? FileType::NUP : FileType::Error;
}

AppLoader::LoadResult AppLoader_NUP::Load(Kernel::KProcess& process, Core::System& system) {
    if (is_loaded) {
        return {ResultStatus::ErrorAlreadyLoaded, {}};
    }
    if (nup->GetStatus() != ResultStatus::Success) {
        return {nup->GetStatus(), {}};
    }
    if (!has_program || secondary_loader == nullptr) {
        return {ResultStatus::ErrorNUPNoBasePackage, {}};
    }

    const auto result = secondary_loader->Load(process, system);
    if (result.first != ResultStatus::Success) {
        return result;
    }

    system.GetFileSystemController().RegisterProcess(
        process.GetProcessId(), base_title_id,
        std::make_shared<FileSys::RomFSFactory>(*this, system.GetContentProvider(),
                                                system.GetFileSystemController()));

    is_loaded = true;
    return result;
}

ResultStatus AppLoader_NUP::VerifyIntegrity(
    std::function<bool(size_t, size_t)> progress_callback) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return nup->GetStatus();
    }
    return nup->VerifyIntegrity(std::move(progress_callback));
}

std::vector<FileSys::Patch> AppLoader_NUP::GetCarriedPatches() const {
    std::vector<FileSys::Patch> out;
    if (!has_program) {
        return out;
    }
    // An update-only file is the bootable program, not an optional layer over a local base.
    const auto base_index = nup->FindBasePackage();
    const auto update_index = nup->FindBestUpdatePackage();
    if (base_index && update_index) {
        const auto& update = nup->GetPackages()[*update_index];
        out.push_back({
            .enabled = selected_update_index.has_value(),
            .name = "Update",
            .version = update.display_version.empty() ? DecodeVersion(update.version)
                                                      : update.display_version,
            .type = FileSys::PatchType::Update,
            .program_id = base_title_id,
            .title_id = update.title_id,
            .source = FileSys::PatchSource::Packed,
            .numeric_version = update.version,
        });
    }
    std::vector<u64> addons;
    for (const auto& package : nup->GetPackages()) {
        if (package.kind == NupContainer::PackageKind::Addon &&
            FileSys::GetBaseTitleID(package.title_id) == FileSys::GetBaseTitleID(base_title_id) &&
            std::find(addons.begin(), addons.end(), package.title_id) == addons.end()) {
            addons.push_back(package.title_id);
        }
    }
    if (!addons.empty()) {
        std::sort(addons.begin(), addons.end());
        std::string ids;
        for (const u64 title_id : addons) {
            ids += fmt::format("{}{}", ids.empty() ? "" : ", ", FileSys::GetAOCID(title_id));
        }
        out.push_back({
            .enabled = !IsAddonDisabled(base_title_id, "DLC"),
            .name = "DLC",
            .version = std::move(ids),
            .type = FileSys::PatchType::DLC,
            .program_id = base_title_id,
            .title_id = base_title_id,
            .source = FileSys::PatchSource::Packed,
        });
    }
    return out;
}

ResultStatus AppLoader_NUP::ReadCarriedPatches(std::vector<FileSys::Patch>& out_patches) const {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    auto carried = GetCarriedPatches();
    out_patches.insert(out_patches.end(), std::make_move_iterator(carried.begin()),
                       std::make_move_iterator(carried.end()));
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadAddOnTitleIds(std::vector<u64>& out_title_ids) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return nup->GetStatus();
    }
    for (const auto& package : nup->GetPackages()) {
        if (package.kind == NupContainer::PackageKind::Addon &&
            std::find(out_title_ids.begin(), out_title_ids.end(), package.title_id) ==
                out_title_ids.end()) {
            out_title_ids.push_back(package.title_id);
        }
    }
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadAddOnRomFS(u64 title_id, FileSys::VirtualFile& out_file) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return nup->GetStatus();
    }
    if (FileSys::GetBaseTitleID(title_id) != FileSys::GetBaseTitleID(base_title_id) ||
        IsAddonDisabled(base_title_id, "DLC")) {
        return ResultStatus::ErrorNoRomFS;
    }
    std::scoped_lock lock{addon_romfs_mutex};
    if (const auto it = addon_romfs.find(title_id); it != addon_romfs.end()) {
        out_file = it->second;
        return ResultStatus::Success;
    }
    std::optional<std::size_t> index;
    const auto& packages = nup->GetPackages();
    for (std::size_t i = 0; i < packages.size(); ++i) {
        if (packages[i].kind == NupContainer::PackageKind::Addon &&
            packages[i].title_id == title_id &&
            (!index || packages[i].version > packages[*index].version)) {
            index = i;
        }
    }
    const auto directory = index ? nup->GetRomFSDir(*index) : FileSys::VirtualDir{};
    if (directory == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    auto romfs = FileSys::CreateRomFS(directory);
    if (romfs == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    addon_romfs.emplace(title_id, romfs);
    out_file = std::move(romfs);
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadRomFS(FileSys::VirtualFile& out_file) {
    // NupContainer lazily builds its directory caches; serialize those builds with add-on opens.
    std::scoped_lock lock{addon_romfs_mutex};
    FileSys::VirtualDir base_romfs;
    if (base_nup && base_nup_index) {
        base_romfs = base_nup->GetRomFSDir(*base_nup_index);
    } else if (const auto base_index = nup->FindBasePackage()) {
        base_romfs = nup->GetRomFSDir(*base_index);
    }
    const auto update_romfs = selected_update_index ? nup->GetRomFSDir(*selected_update_index)
                                                    : FileSys::VirtualDir{};
    // A standalone update carries the complete merged tree; a base+update pair
    // overlays the update onto the base (SPEC §4.1.4).
    const auto merged = FileSys::NupContainer::MergeShadowDirs(base_romfs, update_romfs);
    if (merged == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    out_file = FileSys::CreateRomFS(merged);
    if (out_file == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    return ResultStatus::Success;
}

bool AppLoader_NUP::IsRomFSUpdatable() const {
    // The update is already merged into the exposed RomFS.
    return false;
}

ResultStatus AppLoader_NUP::ReadProgramId(u64& out_program_id) {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    out_program_id = base_title_id;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadProgramIds(std::vector<u64>& out_program_ids) {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    out_program_ids.push_back(base_title_id);
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadIcon(std::vector<u8>& buffer) {
    if (icon_file == nullptr) {
        return ResultStatus::ErrorNoIcon;
    }
    buffer = icon_file->ReadAllBytes();
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadTitle(std::string& title) {
    if (!display_title.empty()) {
        title = display_title;
        return ResultStatus::Success;
    }
    if (nacp_file == nullptr) {
        return ResultStatus::ErrorNoControl;
    }
    title = nacp_file->GetApplicationName();
    if (title.empty()) {
        return ResultStatus::ErrorNoControl;
    }
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadControlData(FileSys::NACP& control) {
    if (nacp_file == nullptr) {
        return ResultStatus::ErrorNoControl;
    }
    control = *nacp_file;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadNSOModules(Modules& modules) {
    if (secondary_loader == nullptr) {
        return ResultStatus::ErrorNotInitialized;
    }
    return secondary_loader->ReadNSOModules(modules);
}

} // namespace Loader
