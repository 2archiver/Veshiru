// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/common_funcs.h"
#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/ns/content_management_interface.h"
#include "core/hle/service/ns/home_menu_alias.h"
#include "core/hle/service/ns/ns_types.h"

namespace Service::NS {

namespace {

u64 SumEntriesForTitle(const FileSys::ContentProvider& provider, FileSys::TitleType type,
                       u64 match_title_id, bool match_base_id) {
    u64 total = 0;
    const auto list = provider.ListEntriesFilter(type, std::nullopt, std::nullopt);
    for (const auto& entry : list) {
        const bool match = match_base_id
                               ? (FileSys::GetBaseTitleID(entry.title_id) == match_title_id &&
                                  entry.title_id != match_title_id &&
                                  entry.title_id != FileSys::GetUpdateTitleID(match_title_id))
                               : (entry.title_id == match_title_id);
        if (!match) {
            continue;
        }
        if (const auto raw = provider.GetEntryRaw(entry.title_id, entry.type); raw != nullptr) {
            total += raw->GetSize();
        }
    }
    return total;
}

} // namespace

IContentManagementInterface::IContentManagementInterface(Core::System& system_)
    : ServiceFramework{system_, "IContentManagementInterface"} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {11, D<&IContentManagementInterface::CalculateApplicationOccupiedSize>, "CalculateApplicationOccupiedSize"},
        {43, D<&IContentManagementInterface::CheckSdCardMountStatus>, "CheckSdCardMountStatus"},
        {47, D<&IContentManagementInterface::GetTotalSpaceSize>, "GetTotalSpaceSize"},
        {48, D<&IContentManagementInterface::GetFreeSpaceSize>, "GetFreeSpaceSize"},
        {71, D<&IContentManagementInterface::GetUnknown71>, "Unknown71"},
        {600, D<&IContentManagementInterface::CountApplicationContentMeta>, "CountApplicationContentMeta"},
        {601, nullptr, "ListApplicationContentMetaStatus"},
        {605, nullptr, "ListApplicationContentMetaStatusWithRightsCheck"},
        {607, D<&IContentManagementInterface::IsAnyApplicationRunning>, "IsAnyApplicationRunning"},
        {608, nullptr, "Unknown608"}, //21.0.0+
    };
    // clang-format on

    RegisterHandlers(functions);
}

IContentManagementInterface::~IContentManagementInterface() = default;

Result IContentManagementInterface::CalculateApplicationOccupiedSize(
    Out<ApplicationOccupiedSize> out_size, u64 application_id) {
    const u64 real_id = ResolveCallerApplicationId(system, application_id);
    LOG_DEBUG(Service_NS, "called, application_id={:016X} (resolved={:016X})", application_id,
              real_id);

    ApplicationOccupiedSize occupied{};
    const auto& fsc = system.GetFileSystemController();
    const auto& union_cp = system.GetContentProviderUnion();
    const u64 update_id = FileSys::GetUpdateTitleID(real_id);

    ApplicationOccupiedSizeEntity nand_entity{
        .storage_id = FileSys::StorageId::NandUser,
        .app_size = 0,
        .patch_size = 0,
        .aoc_size = 0,
    };
    if (const auto* nand = fsc.GetUserNANDContents(); nand != nullptr) {
        nand_entity.app_size =
            SumEntriesForTitle(*nand, FileSys::TitleType::Application, real_id, false);
        nand_entity.patch_size =
            SumEntriesForTitle(*nand, FileSys::TitleType::Update, update_id, false);
        nand_entity.aoc_size = SumEntriesForTitle(*nand, FileSys::TitleType::AOC, real_id, true);
    }

    ApplicationOccupiedSizeEntity sd_entity{
        .storage_id = FileSys::StorageId::SdCard,
        .app_size = 0,
        .patch_size = 0,
        .aoc_size = 0,
    };
    if (const auto* sdmc = fsc.GetSDMCContents(); sdmc != nullptr) {
        sd_entity.app_size =
            SumEntriesForTitle(*sdmc, FileSys::TitleType::Application, real_id, false);
        sd_entity.patch_size =
            SumEntriesForTitle(*sdmc, FileSys::TitleType::Update, update_id, false);
        sd_entity.aoc_size = SumEntriesForTitle(*sdmc, FileSys::TitleType::AOC, real_id, true);
    }
    if (const auto* manual = union_cp.GetManualProvider(); manual != nullptr) {
        if (sd_entity.app_size == 0 && nand_entity.app_size == 0) {
            if (const auto info = manual->GetApplicationInfo(real_id);
                info.has_value() && info->occupied_size > 0) {
                sd_entity.app_size = info->occupied_size;
            } else {
                sd_entity.app_size =
                    SumEntriesForTitle(*manual, FileSys::TitleType::Application, real_id, false);
            }
        }
        sd_entity.patch_size +=
            SumEntriesForTitle(*manual, FileSys::TitleType::Update, update_id, false);
        sd_entity.aoc_size += SumEntriesForTitle(*manual, FileSys::TitleType::AOC, real_id, true);
    }

    occupied.entities[0] = sd_entity;
    occupied.entities[1] = nand_entity;
    occupied.entities[2] = {.storage_id = FileSys::StorageId::None};
    occupied.entities[3] = {.storage_id = FileSys::StorageId::None};
    *out_size = occupied;
    R_SUCCEED();
}

Result IContentManagementInterface::CheckSdCardMountStatus() {
    LOG_WARNING(Service_NS, "(STUBBED) called");
    R_SUCCEED();
}

Result IContentManagementInterface::GetTotalSpaceSize(Out<s64> out_total_space_size,
                                                      FileSys::StorageId storage_id) {
    LOG_INFO(Service_NS, "(STUBBED) called, storage_id={}", storage_id);
    *out_total_space_size = system.GetFileSystemController().GetTotalSpaceSize(storage_id);
    R_SUCCEED();
}

Result IContentManagementInterface::GetFreeSpaceSize(Out<s64> out_free_space_size,
                                                     FileSys::StorageId storage_id) {
    LOG_INFO(Service_NS, "(STUBBED) called, storage_id={}", storage_id);
    *out_free_space_size = system.GetFileSystemController().GetFreeSpaceSize(storage_id);
    R_SUCCEED();
}

Result IContentManagementInterface::GetUnknown71(Out<u64> out_value_a, Out<u64> out_value_b,
                                                 u8 flag) {
    LOG_INFO(Service_NS, "(STUBBED) called, flag={:02X}", flag);
    *out_value_a = 0;
    *out_value_b = 0;
    R_SUCCEED();
}

Result IContentManagementInterface::CountApplicationContentMeta(Out<s32> out_count,
                                                                u64 application_id) {
    const u64 real_id = ResolveCallerApplicationId(system, application_id);
    const auto& cp = system.GetContentProvider();
    s32 count = 0;
    if (cp.HasEntry(real_id, FileSys::ContentRecordType::Program)) {
        ++count;
    }
    if (cp.HasEntry(FileSys::GetUpdateTitleID(real_id), FileSys::ContentRecordType::Program)) {
        ++count;
    }
    const auto aocs = cp.ListEntriesFilter(FileSys::TitleType::AOC,
                                           FileSys::ContentRecordType::Data, std::nullopt);
    for (const auto& entry : aocs) {
        if (FileSys::GetBaseTitleID(entry.title_id) == real_id) {
            ++count;
        }
    }
    *out_count = count;
    R_SUCCEED();
}

Result IContentManagementInterface::IsAnyApplicationRunning(Out<bool> out_is_running) {
    *out_is_running = !IsSystemAppletProcess(system) && system.GetApplicationProcessProgramID() != 0;
    R_SUCCEED();
}

} // namespace Service::NS
