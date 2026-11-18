// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "veshiru/native_console.h"

#ifdef _WIN32
#include <atomic>
#include <windows.h>

#include "common/string_util.h"
#endif

namespace Veshiru {

#ifdef _WIN32
namespace {
std::atomic_bool shutdown_requested{};
bool private_console{};

BOOL WINAPI HandleConsoleControl(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) {
        return FALSE;
    }
    shutdown_requested.store(true, std::memory_order_relaxed);
    return TRUE;
}

bool HasRedirectedOutput(DWORD stream) {
    const auto handle = GetStdHandle(stream);
    if (!handle || handle == INVALID_HANDLE_VALUE) {
        return false;
    }
    const auto type = GetFileType(handle);
    return type == FILE_TYPE_DISK || type == FILE_TYPE_PIPE;
}
} // namespace
#endif

NativeConsole::NativeConsole(ConsoleMode mode) {
#ifdef _WIN32
    DWORD process{};
    const auto attached = GetConsoleProcessList(&process, 1);
    private_console = attached == 1 && !HasRedirectedOutput(STD_OUTPUT_HANDLE) &&
                      !HasRedirectedOutput(STD_ERROR_HANDLE);
    if (mode == ConsoleMode::Disable) {
        if (attached > 0) {
            FreeConsole();
        }
        return;
    }
    if (mode == ConsoleMode::Enable && attached == 0 && !HasRedirectedOutput(STD_OUTPUT_HANDLE) &&
        !HasRedirectedOutput(STD_ERROR_HANDLE)) {
        if (!AttachConsole(ATTACH_PARENT_PROCESS) && AllocConsole()) {
            private_console = true;
        }
    }
    // Register after attaching: AttachConsole/AllocConsole reset the handler table.
    SetConsoleCtrlHandler(nullptr, FALSE);
    handler_registered = SetConsoleCtrlHandler(HandleConsoleControl, TRUE) != FALSE;
#else
    (void)mode;
#endif
}

NativeConsole::~NativeConsole() {
#ifdef _WIN32
    if (handler_registered) {
        SetConsoleCtrlHandler(HandleConsoleControl, FALSE);
    }
#endif
}

bool NativeConsole::ShutdownRequested() {
#ifdef _WIN32
    return shutdown_requested.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

bool NativeConsole::HasPrivateConsole() {
#ifdef _WIN32
    return private_console;
#else
    return false;
#endif
}

void NativeConsole::ShowEarlyError(std::string_view utf8_message) {
#ifdef _WIN32
    if (HasPrivateConsole()) {
        MessageBoxW(nullptr, Common::UTF8ToUTF16W(utf8_message).c_str(), L"Veshiru could not start",
                    MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
    }
#else
    (void)utf8_message;
#endif
}

} // namespace Veshiru
