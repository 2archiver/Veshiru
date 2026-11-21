// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common::Windows {

/// On an unhandled exception, abort() or a CRT invalid-parameter failure, logs one Critical "Host
/// crash" line (code, faulting address and stack as module+offset, thread) and flushes it to the
/// log. It deliberately writes no crash dump: diagnostics are the log and the sanitised Export
/// diagnostics zip. Call after Common::Log::Initialize.
void InstallCrashHandler();

} // namespace Common::Windows
