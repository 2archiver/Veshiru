// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>

namespace Vulkan::Nvn {

enum class BackingClass {
    ImportedHostPool,
    PoolMirror,
    ShadowCopy,
    DeviceLocalPool,
    RawGlobalTable,
    Other,
};

/// Classifies the effective backing of a compute storage binding from replay state. More-specific
/// overlays take precedence over the pool's base allocation.
[[nodiscard]] constexpr BackingClass ClassifyBacking(bool raw_global_table, bool shadow_copy,
                                                     bool pool_mirror, bool imported_host_pool,
                                                     bool device_local_pool) noexcept {
    if (raw_global_table) {
        return BackingClass::RawGlobalTable;
    }
    if (shadow_copy) {
        return BackingClass::ShadowCopy;
    }
    if (pool_mirror) {
        return BackingClass::PoolMirror;
    }
    if (imported_host_pool) {
        return BackingClass::ImportedHostPool;
    }
    if (device_local_pool) {
        return BackingClass::DeviceLocalPool;
    }
    return BackingClass::Other;
}

[[nodiscard]] constexpr std::string_view BackingClassName(BackingClass backing) noexcept {
    switch (backing) {
    case BackingClass::ImportedHostPool:
        return "imported host pool";
    case BackingClass::PoolMirror:
        return "pool mirror";
    case BackingClass::ShadowCopy:
        return "shadow copy";
    case BackingClass::DeviceLocalPool:
        return "device-local pool";
    case BackingClass::RawGlobalTable:
        return "raw-global table";
    case BackingClass::Other:
        return "other";
    }
    return "other";
}

} // namespace Vulkan::Nvn
