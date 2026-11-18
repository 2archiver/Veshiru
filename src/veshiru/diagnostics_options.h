// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "common/logging.h"

namespace Veshiru {

enum class ConsoleMode { Inherit, Enable, Disable };

struct DiagnosticsOptions {
    ConsoleMode console{ConsoleMode::Inherit};
    std::optional<Common::Log::Level> level;
    std::string_view log_path;
    bool help{};
    bool version{};
    std::string_view error;
    std::vector<int> remaining;
};

// Arguments are UTF-8 and include argv[0]. The returned path aliases the argument storage.
[[nodiscard]] DiagnosticsOptions ParseDiagnosticsOptions(std::span<const std::string_view> args);

} // namespace Veshiru
