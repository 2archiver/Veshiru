// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>

#include "common/uuid.h"
#include "core/hle/service/am/am_types.h"
#include "core/hle/service/cmif_types.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/os/event.h"
#include "core/hle/service/service.h"

namespace Service::AM {

struct Applet;
class ILibraryAppletAccessor;
class IStorage;
class WindowSystem;

class IApplicationAccessor final : public ServiceFramework<IApplicationAccessor> {
public:
    explicit IApplicationAccessor(Core::System& system_, std::shared_ptr<Applet> applet,
                                  WindowSystem& window_system);
    explicit IApplicationAccessor(Core::System& system_, WindowSystem& window_system,
                                  u64 real_program_id, u64 exposed_program_id,
                                  std::string deferred_launch_path);
    ~IApplicationAccessor() override;

private:
    void RegisterCommandHandlers();

    Result IsCompleted(Out<bool> out_is_completed);
    Result Start();
    Result RequestExit();
    Result Terminate();
    Result GetResult();
    Result GetAppletStateChangedEvent(OutCopyHandle<Kernel::KReadableEvent> out_event);
    Result TerminateAllLibraryApplets();
    Result AreAnyLibraryAppletsLeft(Out<bool> out_any_left);
    Result GetApplicationId(Out<u64> out_application_id);
    Result PushLaunchParameter(LaunchParameterKind kind, SharedPointer<IStorage> storage);
    Result GetApplicationControlProperty(OutBuffer<BufferAttr_HipcMapAlias> out_control_property);
    Result SetUsers(bool enable, InArray<Common::UUID, BufferAttr_HipcMapAlias> user_ids);
    Result GetCurrentLibraryApplet(Out<SharedPointer<ILibraryAppletAccessor>> out_accessor);
    Result RequestForApplicationToGetForeground();
    Result CheckRightsEnvironmentAvailable(Out<bool> out_is_available);
    Result GetNsRightsEnvironmentHandle(Out<u64> out_handle);
    Result ReportApplicationExitTimeout();
    Result SetApplicationAttribute(u32 attribute);
    Result HasSaveDataAccessPermission(Out<bool> out_has_permission);

    WindowSystem& m_window_system;
    const std::shared_ptr<Applet> m_applet;
    KernelHelpers::ServiceContext m_context;
    Event m_deferred_state_event;
    u64 m_real_program_id{};
    u64 m_exposed_program_id{};
    std::string m_deferred_launch_path;
    std::optional<Common::UUID> m_selected_user;
    bool m_launch_dispatched{};
};

} // namespace Service::AM
