// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/decrypted_nca.h"
#include "core/file_sys/patch_manager.h"
#include "core/hle/result.h"
#include "core/hle/service/am/am_types.h"
#include "core/hle/service/am/applet.h"
#include "core/hle/service/am/applet_data_broker.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/am/service/application_accessor.h"
#include "core/hle/service/am/service/library_applet_accessor.h"
#include "core/hle/service/am/service/storage.h"
#include "core/hle/service/am/window_system.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/glue/glue_manager.h"

namespace Service::AM {

IApplicationAccessor::IApplicationAccessor(Core::System& system_, std::shared_ptr<Applet> applet,
                                           WindowSystem& window_system)
    : ServiceFramework{system_, "IApplicationAccessor"}, m_window_system{window_system},
      m_applet{std::move(applet)}, m_context{system_, "IApplicationAccessor"},
      m_deferred_state_event{m_context},
      m_real_program_id{m_applet ? m_applet->program_id : 0},
      m_exposed_program_id{m_real_program_id} {
    RegisterCommandHandlers();
}

IApplicationAccessor::IApplicationAccessor(Core::System& system_, WindowSystem& window_system,
                                           u64 real_program_id, u64 exposed_program_id,
                                           std::string deferred_launch_path)
    : ServiceFramework{system_, "IApplicationAccessor"}, m_window_system{window_system},
      m_applet{nullptr}, m_context{system_, "IApplicationAccessor"},
      m_deferred_state_event{m_context}, m_real_program_id{real_program_id},
      m_exposed_program_id{exposed_program_id},
      m_deferred_launch_path{std::move(deferred_launch_path)} {
    RegisterCommandHandlers();
}

void IApplicationAccessor::RegisterCommandHandlers() {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IApplicationAccessor::GetAppletStateChangedEvent>, "GetAppletStateChangedEvent"},
        {1, D<&IApplicationAccessor::IsCompleted>, "IsCompleted"},
        {10, D<&IApplicationAccessor::Start>, "Start"},
        {20, D<&IApplicationAccessor::RequestExit>, "RequestExit"},
        {25, D<&IApplicationAccessor::Terminate>, "Terminate"},
        {30, D<&IApplicationAccessor::GetResult>, "GetResult"},
        {101, D<&IApplicationAccessor::RequestForApplicationToGetForeground>, "RequestForApplicationToGetForeground"},
        {110, D<&IApplicationAccessor::TerminateAllLibraryApplets>, "TerminateAllLibraryApplets"},
        {111, D<&IApplicationAccessor::AreAnyLibraryAppletsLeft>, "AreAnyLibraryAppletsLeft"},
        {112, D<&IApplicationAccessor::GetCurrentLibraryApplet>, "GetCurrentLibraryApplet"},
        {120, D<&IApplicationAccessor::GetApplicationId>, "GetApplicationId"},
        {121, D<&IApplicationAccessor::PushLaunchParameter>, "PushLaunchParameter"},
        {122, D<&IApplicationAccessor::GetApplicationControlProperty>, "GetApplicationControlProperty"},
        {123, nullptr, "GetApplicationLaunchProperty"},
        {124, nullptr, "GetApplicationLaunchRequestInfo"},
        {130, D<&IApplicationAccessor::SetUsers>, "SetUsers"},
        {131, D<&IApplicationAccessor::CheckRightsEnvironmentAvailable>, "CheckRightsEnvironmentAvailable"},
        {132, D<&IApplicationAccessor::GetNsRightsEnvironmentHandle>, "GetNsRightsEnvironmentHandle"},
        {140, nullptr, "GetDesirableUids"},
        {150, D<&IApplicationAccessor::ReportApplicationExitTimeout>, "ReportApplicationExitTimeout"},
        {160, D<&IApplicationAccessor::SetApplicationAttribute>, "SetApplicationAttribute"},
        {170, D<&IApplicationAccessor::HasSaveDataAccessPermission>, "HasSaveDataAccessPermission"},
        {180, nullptr, "PushToFriendInvitationStorageChannel"},
        {190, nullptr, "PushToNotificationStorageChannel"},
        {200, nullptr, "RequestApplicationSoftReset"},
        {201, nullptr, "RestartApplicationTimer"},
    };
    // clang-format on

    RegisterHandlers(functions);
}

IApplicationAccessor::~IApplicationAccessor() = default;

Result IApplicationAccessor::IsCompleted(Out<bool> out_is_completed) {
    if (!m_applet) {
        *out_is_completed = false;
        R_SUCCEED();
    }
    std::scoped_lock lk{m_applet->lock};
    *out_is_completed = m_applet->is_completed;
    R_SUCCEED();
}

Result IApplicationAccessor::Start() {
    LOG_INFO(Service_AM, "called, program_id={:016X}", m_real_program_id);
    if (!m_applet) {
        if (!m_launch_dispatched && !m_deferred_launch_path.empty()) {
            m_launch_dispatched = true;
            std::vector<u8> user_bytes;
            if (m_selected_user.has_value() && m_selected_user->IsValid()) {
                user_bytes.resize(sizeof(Common::UUID));
                std::memcpy(user_bytes.data(), &(*m_selected_user), sizeof(Common::UUID));
            }
            system.RequestLaunchApplication(m_real_program_id, m_deferred_launch_path,
                                            std::move(user_bytes));
        }
        R_SUCCEED();
    }
    m_applet->process->Run();
    R_SUCCEED();
}

Result IApplicationAccessor::RequestExit() {
    LOG_INFO(Service_AM, "called");
    if (!m_applet) {
        R_SUCCEED();
    }

    std::scoped_lock lk{m_applet->lock};
    if (m_applet->exit_locked) {
        m_applet->lifecycle_manager.RequestExit(system.Kernel());
        m_applet->UpdateSuspensionStateLocked(true);
    } else {
        m_applet->process->Terminate();
    }

    R_SUCCEED();
}

Result IApplicationAccessor::Terminate() {
    LOG_INFO(Service_AM, "called");
    if (!m_applet) {
        R_SUCCEED();
    }
    m_applet->process->Terminate();
    R_SUCCEED();
}

Result IApplicationAccessor::GetResult() {
    LOG_INFO(Service_AM, "called");
    if (!m_applet) {
        R_SUCCEED();
    }
    std::scoped_lock lk{m_applet->lock};
    R_RETURN(m_applet->terminate_result);
}

Result IApplicationAccessor::GetAppletStateChangedEvent(
    OutCopyHandle<Kernel::KReadableEvent> out_event) {
    LOG_INFO(Service_AM, "called");
    *out_event =
        m_applet ? m_applet->state_changed_event.GetHandle() : m_deferred_state_event.GetHandle();
    R_SUCCEED();
}

Result IApplicationAccessor::TerminateAllLibraryApplets() {
    R_SUCCEED();
}

Result IApplicationAccessor::AreAnyLibraryAppletsLeft(Out<bool> out_any_left) {
    *out_any_left = false;
    R_SUCCEED();
}

Result IApplicationAccessor::GetApplicationId(Out<u64> out_application_id) {
    *out_application_id = m_exposed_program_id;
    R_SUCCEED();
}

Result IApplicationAccessor::PushLaunchParameter(LaunchParameterKind kind,
                                                 SharedPointer<IStorage> storage) {
    LOG_INFO(Service_AM, "called, kind={}", kind);
    if (!storage) {
        R_SUCCEED();
    }

    const auto data = storage->GetData();
    if (kind == LaunchParameterKind::AccountPreselectedUser) {
        if (data.size() >= sizeof(LaunchParameterAccountPreselectedUser)) {
            LaunchParameterAccountPreselectedUser lp{};
            std::memcpy(&lp, data.data(), sizeof(lp));
            if (lp.magic == LaunchParameterAccountPreselectedUserMagic &&
                lp.is_account_selected != 0 && lp.current_user.IsValid()) {
                m_selected_user = lp.current_user;
            }
        }
        if (m_applet) {
            m_applet->preselected_user_launch_parameter.push_back(data);
        }
        R_SUCCEED();
    }
    if (kind == LaunchParameterKind::UserChannel) {
        if (m_applet) {
            m_applet->user_channel_launch_parameter.push_back(data);
        } else {
            system.GetUserChannel().push_back(data);
        }
        R_SUCCEED();
    }
    LOG_WARNING(Service_AM, "Ignoring unsupported launch parameter kind={}", kind);
    R_SUCCEED();
}

Result IApplicationAccessor::GetApplicationControlProperty(
    OutBuffer<BufferAttr_HipcMapAlias> out_control_property) {
    LOG_INFO(Service_AM, "called, program_id={:016X}", m_real_program_id);

    std::vector<u8> nacp;
    if (m_applet && R_SUCCEEDED(system.GetARPManager().GetControlProperty(&nacp, m_real_program_id)) &&
        !nacp.empty()) {
        const size_t copy_len = (std::min)(out_control_property.size(), nacp.size());
        std::memcpy(out_control_property.data(), nacp.data(), copy_len);
        if (copy_len < out_control_property.size()) {
            std::memset(out_control_property.data() + copy_len, 0,
                        out_control_property.size() - copy_len);
        }
        R_SUCCEED();
    }

    const auto control =
        FileSys::PatchManager::GetMetadataFromBaseOrUpdate(system, m_real_program_id);
    FileSys::RawNACP raw{};
    if (control.first != nullptr) {
        const auto bytes = control.first->GetRawBytes();
        std::memcpy(&raw, bytes.data(), (std::min)(bytes.size(), sizeof(raw)));
    }
    FileSys::NormalizeNacpForHomeMenu(raw, m_exposed_program_id);
    if (m_exposed_program_id != m_real_program_id) {
        raw.save_data_owner_id = m_exposed_program_id;
        raw.presence_group_id = m_exposed_program_id;
        raw.apparent_platform = FileSys::ApparentPlatform::NX;
    }
    const size_t copy_len = (std::min)(out_control_property.size(), sizeof(raw));
    std::memcpy(out_control_property.data(), &raw, copy_len);
    if (copy_len < out_control_property.size()) {
        std::memset(out_control_property.data() + copy_len, 0,
                    out_control_property.size() - copy_len);
    }
    R_SUCCEED();
}

Result IApplicationAccessor::SetUsers(bool enable,
                                      InArray<Common::UUID, BufferAttr_HipcMapAlias> user_ids) {
    LOG_INFO(Service_AM, "called, enable={} user_id_count={}", enable, user_ids.size());
    if (enable && !user_ids.empty() && user_ids[0].IsValid() && !m_selected_user.has_value()) {
        m_selected_user = user_ids[0];
    }
    R_SUCCEED();
}

Result IApplicationAccessor::GetCurrentLibraryApplet(
    Out<SharedPointer<ILibraryAppletAccessor>> out_accessor) {
    LOG_INFO(Service_AM, "(STUBBED) called");
    *out_accessor = nullptr;
    R_SUCCEED();
}

Result IApplicationAccessor::RequestForApplicationToGetForeground() {
    LOG_INFO(Service_AM, "called");
    if (m_applet) {
        m_window_system.RequestApplicationToGetForeground();
    }
    R_SUCCEED();
}

Result IApplicationAccessor::CheckRightsEnvironmentAvailable(Out<bool> out_is_available) {
    LOG_DEBUG(Service_AM, "called");
    *out_is_available = true;
    R_SUCCEED();
}

Result IApplicationAccessor::GetNsRightsEnvironmentHandle(Out<u64> out_handle) {
    LOG_DEBUG(Service_AM, "called");
    *out_handle = 0xdeadbeef;
    R_SUCCEED();
}

Result IApplicationAccessor::ReportApplicationExitTimeout() {
    LOG_WARNING(Service_AM, "called");
    R_SUCCEED();
}

Result IApplicationAccessor::SetApplicationAttribute(u32 attribute) {
    LOG_DEBUG(Service_AM, "called, attribute={:#x}", attribute);
    R_SUCCEED();
}

Result IApplicationAccessor::HasSaveDataAccessPermission(Out<bool> out_has_permission) {
    *out_has_permission = true;
    R_SUCCEED();
}

} // namespace Service::AM
