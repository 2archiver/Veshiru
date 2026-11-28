// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/loader/nup.h"

#include <algorithm>
#include <cstring>
#include <string_view>

#include "common/artwork_cache.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/decrypted_nca.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
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
    const bool has_raw_nacp = base.nacp.size() == sizeof(FileSys::RawNACP);
    if (!has_raw_nacp && effective_title.empty() && base.localized_titles.empty() &&
        base.publisher.empty() && version.empty()) {
        return nullptr;
    }
    auto nacp = std::make_unique<FileSys::NACP>();
    if (has_raw_nacp) {
        std::memcpy(&nacp->raw, base.nacp.data(), sizeof(FileSys::RawNACP));
    } else {
        nacp->raw.supported_languages = static_cast<FileSys::SupportedLanguage>(0x3FFFF);
        nacp->raw.save_data_owner_id = base.title_id;
    }

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
    if (!effective_title.empty() && (!has_raw_nacp || english.application_name[0] == '\0')) {
        CopyNameToEntry(english, effective_title);
    }
    if (!base.publisher.empty() &&
        (!has_raw_nacp || english.developer_name[0] == '\0')) {
        for (auto& entry : nacp->raw.language_entries.language_entries) {
            CopyDeveloperToEntry(entry, base.publisher);
        }
    }
    if (!version.empty()) {
        const std::size_t length = std::min(version.size(), nacp->raw.version_string.size() - 1);
        std::memcpy(nacp->raw.version_string.data(), version.data(), length);
        nacp->raw.version_string[length] = '\0';
    }
    if (nacp->raw.save_data_owner_id == 0) {
        nacp->raw.save_data_owner_id = base.title_id;
    }

    nacp->language_entries.resize(16);
    std::memcpy(nacp->language_entries.data(), nacp->raw.language_entries.language_entries.data(),
                sizeof(nacp->raw.language_entries.language_entries));
    return nacp;
}

bool HasUsableExeFS(NupContainer& container, std::size_t package_index) {
    const auto exefs = container.GetExeFSDir(package_index);
    return exefs != nullptr && exefs->GetFile("main") != nullptr &&
           exefs->GetFile("main.npdm") != nullptr;
}

constexpr bool IsSwitch2Program(u64 title_id) noexcept {
    return (title_id & 0xFFF0'0000'0000'0000ULL) == 0x0400'0000'0000'0000ULL;
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
    const bool base_has_exefs = base_index && HasUsableExeFS(*nup, *base_index);
    if (base_index && update_index && base_has_exefs &&
        IsUpdateDisabled(nup->GetPackages()[*base_index].title_id,
                         nup->GetPackages()[*update_index].version)) {
        LOG_INFO(Loader, "NUP update is disabled in the game's add-ons; booting the base");
        update_index.reset();
    }
    selected_update_index = update_index;

    const auto addon_count = std::count_if(nup->GetPackages().begin(), nup->GetPackages().end(),
                                           [](const auto& package) {
                                               return package.kind ==
                                                      NupContainer::PackageKind::Addon;
                                           });

    // The program identity comes from the base (or native Switch 2 Edition upgrade over a Switch 1
    // base). A standalone or native-upgrade update can supply ExeFS for an executable-less base; a
    // patch-only update needs its matching base, which may be a sibling .nup.
    std::optional<std::size_t> program_index;
    std::optional<std::size_t> exefs_index;
    bool program_from_base_nup = false;
    bool exefs_from_base_nup = false;
    if (base_index && base_has_exefs) {
        program_index = base_index;
        exefs_index = base_index;
        if (update_index && HasUsableExeFS(*nup, *update_index)) {
            const u64 base_tid = nup->GetPackages()[*base_index].title_id;
            const u64 update_tid = nup->GetPackages()[*update_index].title_id;
            const bool base_is_nx2 = IsSwitch2Program(base_tid);
            const bool update_is_nx2 = IsSwitch2Program(update_tid);
            if (base_is_nx2 && !update_is_nx2) {
                // Switch 2 Edition base + legacy Switch 1 update: preserve Switch 2 native ExeFS
                // while still merging update RomFS below.
                exefs_index = base_index;
            } else if (!base_is_nx2 && update_is_nx2) {
                // Switch 1 base + Switch 2 Edition update: promote to native Switch 2 program.
                program_index = update_index;
                exefs_index = update_index;
            } else {
                exefs_index = update_index;
            }
        }
    } else if (update_index && HasUsableExeFS(*nup, *update_index)) {
        const auto& update = nup->GetPackages()[*update_index];
        if (base_index) {
            // Executable-less base inside the same container paired with an update carrying ExeFS.
            const bool base_is_nx2 = IsSwitch2Program(nup->GetPackages()[*base_index].title_id);
            const bool update_is_nx2 = IsSwitch2Program(update.title_id);
            program_index = (!base_is_nx2 && update_is_nx2) ? update_index : base_index;
            exefs_index = update_index;
        } else {
            base_nup = FindCompanionBase(file, update.title_id, base_nup_index);
            if (base_nup && base_nup_index) {
                const u64 comp_tid = base_nup->GetPackages()[*base_nup_index].title_id;
                LOG_INFO(Loader, "NUP update uses companion base {:016X}", comp_tid);
                const bool base_is_nx2 = IsSwitch2Program(comp_tid);
                const bool update_is_nx2 = IsSwitch2Program(update.title_id);
                if (!base_is_nx2 && update_is_nx2) {
                    program_index = update_index;
                    exefs_index = update_index;
                } else if (base_is_nx2 && !update_is_nx2 &&
                           HasUsableExeFS(*base_nup, *base_nup_index)) {
                    program_index = base_nup_index;
                    program_from_base_nup = true;
                    exefs_index = base_nup_index;
                    exefs_from_base_nup = true;
                } else {
                    program_index = base_nup_index;
                    program_from_base_nup = true;
                    exefs_index = update_index;
                }
            } else {
                LOG_WARNING(
                    Loader,
                    "NUP update {:016X} has no matching base in this folder; trying its own "
                    "program tree. A patch-only update needs the base game's .nup beside it.",
                    update.title_id);
                program_index = update_index;
                exefs_index = update_index;
            }
        }
    }
    if (!program_index || !exefs_index) {
        if (addon_count > 0) {
            LOG_DEBUG(Loader, "NUP container carries {} add-on content package(s) (no program)",
                      addon_count);
        } else {
            LOG_ERROR(Loader, "NUP container has no package with a usable ExeFS");
        }
        return;
    }

    const auto& program = program_from_base_nup ? base_nup->GetPackages()[*base_nup_index]
                                                : nup->GetPackages()[*program_index];
    base_title_id = program.title_id;

    NupContainer::PackageInfo meta;
    meta.title_id = program.title_id;
    meta.nacp = program.nacp;
    meta.title_name = program.title_name;
    meta.localized_titles = program.localized_titles;
    meta.source_dump = program.source_dump;
    meta.publisher = program.publisher;
    meta.icon = program.icon;
    meta.localized_icons = program.localized_icons;
    if (update_index) {
        const auto& update = nup->GetPackages()[*update_index];
        if (!update.nacp.empty()) {
            meta.nacp = update.nacp;
        }
        if (meta.title_name.empty() && !update.title_name.empty()) {
            meta.title_name = update.title_name;
        }
        if (meta.localized_titles.empty() && !update.localized_titles.empty()) {
            meta.localized_titles = update.localized_titles;
        }
        if (meta.publisher.empty() && !update.publisher.empty()) {
            meta.publisher = update.publisher;
        }
        if (meta.icon.empty() && meta.localized_icons.empty() &&
            (!update.icon.empty() || !update.localized_icons.empty())) {
            meta.icon = update.icon;
            meta.localized_icons = update.localized_icons;
        }
    }
    if (base_index && (*base_index != *program_index || program_from_base_nup)) {
        const auto& base = nup->GetPackages()[*base_index];
        if (meta.nacp.empty() && !base.nacp.empty()) {
            meta.nacp = base.nacp;
        }
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

    const auto exefs = exefs_from_base_nup ? base_nup->GetExeFSDir(*exefs_index)
                                           : nup->GetExeFSDir(*exefs_index);
    if (exefs == nullptr || exefs->GetFile("main") == nullptr ||
        exefs->GetFile("main.npdm") == nullptr) {
        LOG_ERROR(Loader, "NUP package has no usable ExeFS");
        has_program = false;
        return;
    }
    secondary_loader = std::make_unique<AppLoader_DeconstructedRomDirectory>(exefs, true, false);

    if (update_index) {
        const auto& update = nup->GetPackages()[*update_index];
        effective_version = update.version;
        display_version = update.display_version.empty() ? DecodeVersion(update.version)
                                                         : update.display_version;
    } else {
        effective_version = program.version;
        display_version = program.display_version.empty() ? DecodeVersion(program.version)
                                                          : program.display_version;
    }
    display_publisher = meta.publisher;
    nacp_file = BuildNacp(meta, display_version);
    if (nacp_file && nacp_file->raw.dlc_base_title_id == 0) {
        std::vector<u64> carried;
        for (const auto& package : nup->GetPackages()) {
            if (package.kind == NupContainer::PackageKind::Addon) {
                carried.push_back(package.title_id);
            }
        }
        nacp_file->raw.dlc_base_title_id = FileSys::ResolveAOCBaseTitleID(base_title_id, 0, carried);
    }
    if (display_title.empty() && nacp_file != nullptr) {
        display_title = nacp_file->GetApplicationName();
    }
    if (display_publisher.empty() && nacp_file != nullptr) {
        display_publisher = nacp_file->GetDeveloperName();
    }
    if (display_version.empty() && nacp_file != nullptr) {
        display_version = nacp_file->GetVersionString();
    }
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

    if (addon_count > 0) {
        LOG_INFO(Loader, "NUP container carries {} add-on content package(s)", addon_count);
    }
}

u64 AppLoader_NUP::AddonApplicationId() const {
    if (nacp_file != nullptr && nacp_file->GetDLCBaseTitleId() != 0) {
        return FileSys::GetBaseTitleID(nacp_file->GetDLCBaseTitleId());
    }
    return FileSys::GetBaseTitleID(base_title_id);
}

bool AppLoader_NUP::MatchesAddonTitle(u64 addon_title_id) const {
    const u64 addon_base = FileSys::GetBaseTitleID(addon_title_id);
    if (base_title_id == 0) {
        return false;
    }
    return addon_base == FileSys::GetBaseTitleID(base_title_id) ||
           addon_base == AddonApplicationId();
}

void AppLoader_NUP::EnsureCompanionAddons() const {
    if (companion_addons_searched || !has_program || file == nullptr) {
        return;
    }
    companion_addons_searched = true;
    const auto directory = file->GetContainingDirectory();
    if (directory == nullptr) {
        return;
    }
    const u64 app_base = FileSys::GetBaseTitleID(base_title_id);
    const u64 aoc_base = AddonApplicationId();
    auto candidates = directory->GetFiles();
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a->GetName() < b->GetName();
    });
    for (const auto& candidate : candidates) {
        if (candidate == nullptr || candidate->GetName() == file->GetName() ||
            Common::ToLower(candidate->GetExtension()) != "nup") {
            continue;
        }
        const auto peeked = NupContainer::PeekAddonTitleIds(candidate);
        const bool has_match = std::ranges::any_of(peeked, [app_base, aoc_base](u64 tid) {
            const u64 tid_base = FileSys::GetBaseTitleID(tid);
            return tid_base == app_base || tid_base == aoc_base;
        });
        if (!has_match) {
            continue;
        }
        auto container = std::make_unique<NupContainer>(candidate);
        if (container->GetStatus() == ResultStatus::Success) {
            companion_addon_nups.push_back(std::move(container));
        }
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

    auto romfs_factory = std::make_shared<FileSys::RomFSFactory>(
        *this, system.GetContentProvider(), system.GetFileSystemController());
    if (!IsAddonDisabled(base_title_id, "DLC")) {
        std::vector<u64> addon_ids;
        if (ReadAddOnTitleIds(addon_ids) == ResultStatus::Success) {
            for (const u64 addon_tid : addon_ids) {
                if (!MatchesAddonTitle(addon_tid)) {
                    continue;
                }
                FileSys::VirtualFile addon_romfs;
                if (ReadAddOnRomFS(addon_tid, addon_romfs) == ResultStatus::Success &&
                    addon_romfs != nullptr) {
                    romfs_factory->SetLooseAOC(addon_tid, std::move(addon_romfs));
                }
            }
        }
    }
    system.GetFileSystemController().RegisterProcess(process.GetProcessId(), base_title_id,
                                                     std::move(romfs_factory));

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

bool AppLoader_NUP::RegisterContent(FileSys::ManualContentProvider& provider, bool only_content,
                                    std::optional<u64> base_program_id) const {
    if (nup == nullptr || nup->GetStatus() != ResultStatus::Success) {
        return false;
    }

    bool added = false;
    const auto& packages = nup->GetPackages();
    std::map<u64, std::size_t> newest_addons;
    const auto addon_base = AddonApplicationId();
    for (std::size_t i = 0; i < packages.size(); ++i) {
        const auto& package = packages[i];
        const u64 pkg_base = FileSys::GetBaseTitleID(package.title_id);
        if (package.kind != NupContainer::PackageKind::Addon ||
            (base_program_id && pkg_base != FileSys::GetBaseTitleID(*base_program_id) &&
             pkg_base != addon_base)) {
            continue;
        }
        const auto [it, inserted] = newest_addons.try_emplace(package.title_id, i);
        if (!inserted && package.version > packages[it->second].version) {
            it->second = i;
        }
    }
    for (const auto& [title_id, index] : newest_addons) {
        const auto romfs_dir = nup->GetRomFSDir(index);
        const auto romfs = romfs_dir != nullptr ? FileSys::CreateRomFS(romfs_dir) : nullptr;
        if (romfs == nullptr) {
            LOG_WARNING(Loader, "NUP add-on {:016X} has no RomFS", title_id);
            continue;
        }
        if (auto nca = FileSys::BuildAddOnContentNca(title_id, romfs); nca != nullptr) {
            provider.AddEntry(FileSys::TitleType::AOC, FileSys::ContentRecordType::Data, title_id,
                              std::move(nca));
            added = true;
        }
    }

    // Skip application registration for DLC-only containers, standalone patch files whose base
    // .nup is already exposed independently, or unbootable incomplete patch-only dumps.
    if (!only_content && !base_program_id.has_value() && has_program &&
        secondary_loader != nullptr && base_title_id != 0 && file != nullptr && !IsUpdateOnly()) {
        std::vector<u8> icon_bytes;
        if (icon_file != nullptr && icon_file->GetSize() > 0) {
            icon_bytes = icon_file->ReadAllBytes();
        }
        if (icon_bytes.empty()) {
            icon_bytes = Common::ArtworkCache::ReadCachedArtwork(base_title_id);
        }

        std::string fallback_title = display_title;
        if (fallback_title.empty()) {
            fallback_title = Common::ArtworkCache::ReadSidecarTitle(base_title_id);
        }
        if (fallback_title.empty()) {
            std::string stem = file->GetName();
            if (const auto dot = stem.rfind('.'); dot != std::string::npos && dot > 0) {
                stem.resize(dot);
            }
            fallback_title = std::move(stem);
        }

        const FileSys::RawNACP raw_nacp = nacp_file != nullptr ? nacp_file->raw : FileSys::RawNACP{};
        provider.RegisterApplicationEntry(base_title_id, file, raw_nacp, icon_bytes, fallback_title,
                                          display_publisher, display_version, effective_version,
                                          file->GetFullPath(), file->GetSize());
        added = true;
    }

    // If this is a standalone update-only .nup paired with a companion base .nup in the same
    // directory, upgrade the base application's launch path/version/control metadata when no base
    // update was bundled inside the base container itself.
    if (!only_content && !base_program_id.has_value() && has_program &&
        secondary_loader != nullptr && base_title_id != 0 && file != nullptr && IsUpdateOnly() &&
        base_nup != nullptr) {
        const auto existing = provider.GetApplicationInfo(base_title_id);
        if (!existing.has_value() || effective_version >= existing->version) {
            std::vector<u8> icon_bytes;
            if (icon_file != nullptr && icon_file->GetSize() > 0) {
                icon_bytes = icon_file->ReadAllBytes();
            }
            if (icon_bytes.empty()) {
                icon_bytes = Common::ArtworkCache::ReadCachedArtwork(base_title_id);
            }
            std::string fallback_title = display_title;
            if (fallback_title.empty()) {
                fallback_title = Common::ArtworkCache::ReadSidecarTitle(base_title_id);
            }
            if (fallback_title.empty()) {
                std::string stem = file->GetName();
                if (const auto dot = stem.rfind('.'); dot != std::string::npos && dot > 0) {
                    stem.resize(dot);
                }
                fallback_title = std::move(stem);
            }
            const FileSys::RawNACP raw_nacp =
                nacp_file != nullptr ? nacp_file->raw : FileSys::RawNACP{};
            const u64 total_size =
                file->GetSize() + (existing.has_value() ? existing->occupied_size : 0);
            provider.RegisterApplicationEntry(base_title_id, file, raw_nacp, icon_bytes,
                                              fallback_title, display_publisher, display_version,
                                              effective_version, file->GetFullPath(), total_size);
            added = true;
        }
    }

    return added;
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
    const auto collect_matching_addons = [&](const NupContainer& container) {
        for (const auto& package : container.GetPackages()) {
            if (package.kind == NupContainer::PackageKind::Addon &&
                MatchesAddonTitle(package.title_id) &&
                std::find(addons.begin(), addons.end(), package.title_id) == addons.end()) {
                addons.push_back(package.title_id);
            }
        }
    };
    collect_matching_addons(*nup);
    if (base_nup) {
        collect_matching_addons(*base_nup);
    }
    {
        std::scoped_lock lock{addon_romfs_mutex};
        EnsureCompanionAddons();
        for (const auto& companion : companion_addon_nups) {
            collect_matching_addons(*companion);
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
    const auto append_all = [&](const NupContainer& container, bool filter_matching) {
        for (const auto& package : container.GetPackages()) {
            if (package.kind == NupContainer::PackageKind::Addon &&
                (!filter_matching || MatchesAddonTitle(package.title_id)) &&
                std::find(out_title_ids.begin(), out_title_ids.end(), package.title_id) ==
                    out_title_ids.end()) {
                out_title_ids.push_back(package.title_id);
            }
        }
    };
    append_all(*nup, false);
    if (base_nup) {
        append_all(*base_nup, true);
    }
    {
        std::scoped_lock lock{addon_romfs_mutex};
        EnsureCompanionAddons();
        for (const auto& companion : companion_addon_nups) {
            append_all(*companion, true);
        }
    }
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadAddOnRomFS(u64 title_id, FileSys::VirtualFile& out_file) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return nup->GetStatus();
    }
    if (!MatchesAddonTitle(title_id) || IsAddonDisabled(base_title_id, "DLC")) {
        return ResultStatus::ErrorNoRomFS;
    }
    std::scoped_lock lock{addon_romfs_mutex};
    if (const auto it = addon_romfs.find(title_id); it != addon_romfs.end()) {
        out_file = it->second;
        return ResultStatus::Success;
    }
    EnsureCompanionAddons();

    NupContainer* best_container = nullptr;
    std::optional<std::size_t> best_index;
    u32 best_version = 0;
    const auto check_container = [&](NupContainer& container) {
        const auto& packages = container.GetPackages();
        for (std::size_t i = 0; i < packages.size(); ++i) {
            if (packages[i].kind == NupContainer::PackageKind::Addon &&
                packages[i].title_id == title_id &&
                (!best_index || packages[i].version > best_version)) {
                best_container = &container;
                best_index = i;
                best_version = packages[i].version;
            }
        }
    };
    check_container(*nup);
    if (base_nup) {
        check_container(*base_nup);
    }
    for (const auto& companion : companion_addon_nups) {
        check_container(*companion);
    }

    const auto directory =
        (best_container && best_index) ? best_container->GetRomFSDir(*best_index) : FileSys::VirtualDir{};
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
    // Prefer user-refreshed or sidecar cached HD artwork when present, falling back to the
    // container's embedded icon without ever mutating the .nup archive on disk.
    if (base_title_id != 0) {
        auto cached = Common::ArtworkCache::ReadCachedArtwork(base_title_id);
        if (!cached.empty()) {
            buffer = std::move(cached);
            return ResultStatus::Success;
        }
    }
    if (icon_file != nullptr) {
        auto embedded = icon_file->ReadAllBytes();
        if (!embedded.empty()) {
            buffer = std::move(embedded);
            return ResultStatus::Success;
        }
    }
    return ResultStatus::ErrorNoIcon;
}

ResultStatus AppLoader_NUP::ReadTitle(std::string& title) {
    if (!display_title.empty()) {
        title = display_title;
        return ResultStatus::Success;
    }
    if (nacp_file != nullptr) {
        title = nacp_file->GetApplicationName();
        if (!title.empty()) {
            return ResultStatus::Success;
        }
    }
    if (base_title_id != 0) {
        if (auto sidecar_title = Common::ArtworkCache::ReadSidecarTitle(base_title_id);
            !sidecar_title.empty()) {
            title = std::move(sidecar_title);
            return ResultStatus::Success;
        }
    }
    return ResultStatus::ErrorNoControl;
}

ResultStatus AppLoader_NUP::ReadDisplayVersion(std::string& version) {
    if (!display_version.empty()) {
        version = display_version;
        return ResultStatus::Success;
    }
    if (nacp_file != nullptr) {
        version = nacp_file->GetVersionString();
        if (!version.empty()) {
            return ResultStatus::Success;
        }
    }
    return AppLoader::ReadDisplayVersion(version);
}

ResultStatus AppLoader_NUP::ReadControlData(FileSys::NACP& control) const {
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
