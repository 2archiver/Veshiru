// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/fs/fs.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/registered_cache.h"
#include "core/hle/service/am/am_types.h"
#include "core/hle/service/am/applet.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/am/process_creation.h"
#include "core/hle/service/am/service/application_accessor.h"
#include "core/hle/service/am/service/application_creator.h"
#include "core/hle/service/am/window_system.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/ns/home_menu_alias.h"
#include "core/loader/loader.h"
#include "core/launch_timestamp_cache.h"
#include "core/hle/api_version.h"

namespace Service::AM {

namespace {

std::string ResolveFrontendLaunchPath(const FileSys::ContentProviderUnion& storage,
                                      u64 real_program_id, const FileSys::VirtualFile& nca_raw) {
    if (const auto* manual = storage.GetManualProvider(); manual != nullptr) {
        if (const auto info = manual->GetApplicationInfo(real_program_id);
            info.has_value() && !info->launch_path.empty() &&
            Common::FS::Exists(info->launch_path)) {
            return info->launch_path;
        }
    }
    if (nca_raw != nullptr) {
        if (const std::string path = nca_raw->GetFullPath();
            !path.empty() && Common::FS::Exists(path)) {
            return path;
        }
    }
    return {};
}

Result CreateGuestApplication(SharedPointer<IApplicationAccessor>* out_application_accessor,
                              Core::System& system, WindowSystem& window_system,
                              u64 requested_program_id) {
    auto& storage = system.GetContentProviderUnion();
    const u64 real_program_id =
        Service::NS::FromHomeMenuApplicationId(requested_program_id, storage);
    const FileSys::VirtualFile nca_raw =
        storage.GetEntryRaw(real_program_id, FileSys::ContentRecordType::Program);

    R_UNLESS(nca_raw != nullptr, ResultUnknown);

    // When a frontend launch callback is registered, delegate the full boot to the frontend once
    // qlaunch calls IApplicationAccessor::Start(). This cleanly tears down qlaunch first and boots
    // Switch 1 & Switch 2 .nup/.nx/.nsp/.xci/.nro games with per-title settings, full Switch 2
    // extended DRAM, shader cache preloading, carried updates/DLC, and the selected user profile.
    if (system.HasLaunchApplicationCallback()) {
        if (std::string launch_path = ResolveFrontendLaunchPath(storage, real_program_id, nca_raw);
            !launch_path.empty()) {
            LOG_INFO(Service_AM,
                     "Preparing frontend handoff for {:016X} (requested {:016X}) -> {}",
                     real_program_id, requested_program_id, launch_path);
            *out_application_accessor = std::make_shared<IApplicationAccessor>(
                system, window_system, real_program_id, requested_program_id,
                std::move(launch_path));
            R_SUCCEED();
        }
    }

    std::vector<u8> control;
    std::unique_ptr<Loader::AppLoader> loader;
    Loader::ResultStatus result;
    auto process =
        CreateApplicationProcess(control, loader, result, system, nca_raw, real_program_id, 0);
    R_UNLESS(process != nullptr, ResultUnknown);

    const auto applet = std::make_shared<Applet>(system, std::move(process), true);
    applet->program_id = real_program_id;
    applet->applet_id = AppletId::Application;
    applet->type = AppletType::Application;
    applet->library_applet_mode = LibraryAppletMode::AllForeground;

    window_system.TrackApplet(applet, true);

    *out_application_accessor = std::make_shared<IApplicationAccessor>(system, applet, window_system);
    R_SUCCEED();
}

} // namespace

IApplicationCreator::IApplicationCreator(Core::System& system_, WindowSystem& window_system)
    : ServiceFramework{system_, "IApplicationCreator"}, m_window_system{window_system} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IApplicationCreator::CreateApplication>, "CreateApplication"},
        {1, nullptr, "PopLaunchRequestedApplication"},
        {10, D<&IApplicationCreator::CreateSystemApplication>, "CreateSystemApplication"},
        {100, nullptr, "PopFloatingApplicationForDevelopment"},
    };
    // clang-format on

    RegisterHandlers(functions);
}

IApplicationCreator::~IApplicationCreator() = default;

Result IApplicationCreator::CreateApplication(
    Out<SharedPointer<IApplicationAccessor>> out_application_accessor, u64 application_id) {
    const u64 real_id =
        Service::NS::FromHomeMenuApplicationId(application_id, system.GetContentProvider());
    LOG_INFO(Service_NS, "called, application_id={:016X} (resolved={:016X})", application_id,
             real_id);
    Core::LaunchTimestampCache::SaveLaunchTimestamp(real_id);
    R_RETURN(
        CreateGuestApplication(out_application_accessor, system, m_window_system, application_id));
}

Result IApplicationCreator::CreateSystemApplication(
    Out<SharedPointer<IApplicationAccessor>> out_application_accessor, u64 application_id) {

    FileSys::VirtualFile nca_raw{};

    auto& storage = system.GetContentProviderUnion();
    nca_raw = storage.GetEntryRaw(application_id, FileSys::ContentRecordType::Program);

    R_UNLESS(nca_raw != nullptr, ResultUnknown);

    std::vector<u8> control;
    std::unique_ptr<Loader::AppLoader> loader;
    auto process = CreateProcess(system, application_id, 1, HLE::ApiVersion::HOS_VERSION_MAJOR);
    R_UNLESS(process != nullptr, ResultUnknown);

    const auto applet = std::make_shared<Applet>(system, std::move(process), true);
    applet->program_id = application_id;
    applet->applet_id = AppletId::Starter;
    applet->type = AppletType::LibraryApplet;
    applet->library_applet_mode = LibraryAppletMode::AllForeground;

    m_window_system.TrackApplet(applet, true);

    *out_application_accessor = std::make_shared<IApplicationAccessor>(system, applet, m_window_system);
    Core::LaunchTimestampCache::SaveLaunchTimestamp(application_id);
    R_SUCCEED();
}

} // namespace Service::AM
