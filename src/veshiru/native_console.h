// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>

#include "veshiru/diagnostics_options.h"

namespace Veshiru {

class NativeConsole {
public:
    explicit NativeConsole(ConsoleMode mode);
    ~NativeConsole();

    [[nodiscard]] static bool ShutdownRequested();
    [[nodiscard]] static bool HasPrivateConsole();
    /// Do not rely on Qt's platform plugin to report a fatal error loading that plugin.
    static void ShowEarlyError(std::string_view utf8_message);

private:
    bool handler_registered{};
};

} // namespace Veshiru
