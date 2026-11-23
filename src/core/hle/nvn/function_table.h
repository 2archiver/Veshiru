// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string_view>

namespace Core::NvnHle {

/// NVN function names indexed by function id (the line index of nxnative's names.txt), taken from
/// video_core/nvn/nvn_functions.inc.
[[nodiscard]] std::span<const std::string_view> FunctionNames();

} // namespace Core::NvnHle
