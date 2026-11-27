// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_try_switch.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace Common::Nx2TrySwitch {
namespace {

struct Registry {
    std::mutex mutex;
    std::map<std::string, bool, std::less<>> overrides;
    u64 generation{};
};

Registry& GetRegistry() {
    static Registry registry;
    return registry;
}

std::optional<bool> EnvironmentValue(std::string_view name) {
    std::string env_name{"EDEN_NX2_"};
    env_name.append(name);
    const char* const value = std::getenv(env_name.c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return value[0] == '1';
}

bool DefaultEnabled(std::string_view name) {
    if (const auto value = EnvironmentValue(name)) {
        return *value;
    }
    // Raw stores write through to the read-only and storage VTX-HLE pool mirrors. Raw-writing draws
    // then skip the whole-pool snapshot (OoT market: 345 -> 62 MiB copied each way per present).
    return name == "TRY_NATIVE_F32_ATOMICS" || name == "TRY_ATOMIC_RESERVE" ||
           name == "TRY_RO_MIRROR_EXACT" || name == "TRY_STORAGE_RAW_WRITE_THROUGH";
}

bool EnabledLocked(const Registry& registry, std::string_view name, bool fallback) {
    if (const auto it = registry.overrides.find(name); it != registry.overrides.end()) {
        return it->second;
    }
    if (const auto value = EnvironmentValue(name)) {
        return *value;
    }
    return fallback;
}

} // namespace

bool Enabled(std::string_view name) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    return EnabledLocked(registry, name, DefaultEnabled(name));
}

bool Enabled(std::string_view name, bool recommended_value) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    return EnabledLocked(registry, name, recommended_value);
}

bool Set(std::string_view name, bool enabled) {
    if (!name.starts_with("TRY_") || name.size() <= 4 ||
        !std::all_of(name.begin(), name.end(), [](unsigned char c) {
            return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        })) {
        return false;
    }
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    const bool changed = EnabledLocked(registry, name, DefaultEnabled(name)) != enabled;
    registry.overrides[std::string{name}] = enabled;
    if (changed) {
        ++registry.generation;
    }
    return true;
}

u64 Generation() {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    return registry.generation;
}

bool DiffersFromDefault(std::string_view name) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    const auto it = registry.overrides.find(name);
    return it != registry.overrides.end() && it->second != DefaultEnabled(name);
}

} // namespace Common::Nx2TrySwitch
