// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common {

/// Makes std::terminate log the exception that caused it (type and what()) and flush the log
/// before aborting, so an uncaught exception on any thread leaves a reason in the log file.
void InstallTerminateHandler();

} // namespace Common
