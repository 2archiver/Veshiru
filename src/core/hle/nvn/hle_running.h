// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Kept apart from bridge.h, which Qt translation units cannot include (its `slots` member meets
// Qt's `slots` macro).

namespace Core::NvnHle {

/// Whether the running application's NVN calls are taken by the HLE (its SDK is hooked and the
/// HLE was not given up for want of a replay context). Never creates the bridge, so the frontend
/// may ask at any time; false between programs and for Switch 1 programs.
[[nodiscard]] bool HleRunning();

} // namespace Core::NvnHle
