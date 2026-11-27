// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>

#include "common/common_types.h"

namespace Common::Nx2TrySwitch {

/// Whether an NX2 experiment switch is enabled by a runtime override, environment, or default.
[[nodiscard]] bool Enabled(std::string_view name);
/// Whether it is enabled, using `recommended_value` when no override or environment value exists.
[[nodiscard]] bool Enabled(std::string_view name, bool recommended_value);
/// Runtime override used by tests and in-process experiment controls.
[[nodiscard]] bool Set(std::string_view name, bool enabled);
/// Generation increments whenever a runtime override changes its effective value.
[[nodiscard]] u64 Generation();
/// Whether a runtime override differs from the environment/default value.
[[nodiscard]] bool DiffersFromDefault(std::string_view name);

} // namespace Common::Nx2TrySwitch
