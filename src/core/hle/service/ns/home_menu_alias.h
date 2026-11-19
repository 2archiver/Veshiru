// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "core/core.h"
#include "core/file_sys/registered_cache.h"

namespace Service::NS {

/// Switch 1 HOME Menu (qlaunch) expects Switch 1 0x01xx... application IDs and rejects or hides
/// Switch 2 (Ounce) 0x0400... Program IDs on product_model=1. Map 0x0400... <-> 0x01F0... (an
/// unallocated Switch 1 prefix) deterministically so Switch 2 .nup/.nx titles appear, sort, cache
/// icons, and launch cleanly while never altering any running game's real Program ID or save path.
inline constexpr u64 HomeMenuPrefixMask = 0xFFF0'0000'0000'0000ULL;
inline constexpr u64 Switch2ProgramPrefix = 0x0400'0000'0000'0000ULL;
inline constexpr u64 Switch2HomeMenuAliasPrefix = 0x01F0'0000'0000'0000ULL;

[[nodiscard]] constexpr bool IsSwitch2ApplicationId(u64 application_id) noexcept {
    return (application_id & HomeMenuPrefixMask) == Switch2ProgramPrefix;
}

[[nodiscard]] constexpr bool IsHomeMenuAliasId(u64 application_id) noexcept {
    return (application_id & HomeMenuPrefixMask) == Switch2HomeMenuAliasPrefix;
}

[[nodiscard]] constexpr u64 ToHomeMenuApplicationId(u64 real_application_id) noexcept {
    if (IsSwitch2ApplicationId(real_application_id)) {
        return Switch2HomeMenuAliasPrefix | (real_application_id & ~HomeMenuPrefixMask);
    }
    return real_application_id;
}

[[nodiscard]] inline u64 FromHomeMenuApplicationId(u64 application_id,
                                                   const FileSys::ContentProvider& provider) {
    if (!IsHomeMenuAliasId(application_id)) {
        return application_id;
    }
    // Preserve any real 0x01F0... title if one is ever installed.
    if (provider.HasEntry(application_id, FileSys::ContentRecordType::Program) ||
        provider.HasEntry(application_id, FileSys::ContentRecordType::Control)) {
        return application_id;
    }
    const u64 candidate = Switch2ProgramPrefix | (application_id & ~HomeMenuPrefixMask);
    if (provider.HasEntry(candidate, FileSys::ContentRecordType::Program) ||
        provider.HasEntry(candidate, FileSys::ContentRecordType::Control)) {
        return candidate;
    }
    return application_id;
}

/// Returns true when the active foreground process is a Switch system/library applet (0x0100000000001xxx:
/// qlaunch, Starter, Settings, MyPage, ProfileSelect, etc.) rather than a user game.
[[nodiscard]] inline bool IsSystemAppletProcess(const Core::System& system) {
    const u64 pid = system.GetApplicationProcessProgramID();
    return pid >= 0x0100'0000'0000'1000ULL && pid <= 0x0100'0000'0000'1FFFULL;
}

[[nodiscard]] inline u64 ResolveCallerApplicationId(const Core::System& system,
                                                    u64 application_id) {
    if (!IsSystemAppletProcess(system)) {
        return application_id;
    }
    return FromHomeMenuApplicationId(application_id, system.GetContentProvider());
}

} // namespace Service::NS
