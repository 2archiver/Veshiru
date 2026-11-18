// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/vulkan_common/vulkan_wrapper.h"

#include <chrono>
#include <exception>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

#include "common/logging.h"
#include "common/scope_exit.h"
#include "veshiru/native_console.h"
#include "veshiru/startup_checks.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"

bool ProbeVulkan() {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    try {
        Vulkan::vk::InstanceDispatch dld;
        const auto library = Vulkan::OpenLibrary();
        const auto instance = Vulkan::CreateInstance(*library, dld, VK_API_VERSION_1_1);
        return !instance.EnumeratePhysicalDevices().empty();
    } catch (const std::exception&) {
        return false;
    }
}

bool CheckVulkanInChild([[maybe_unused]] const char* executable) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + seconds{15};
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) {
        LOG_ERROR(Frontend, "Could not locate the executable for the Vulkan startup check");
        return false;
    }
    path.resize(length);
    const std::string_view argument{VulkanProbeArgument};
    std::wstring command = L"\"" + path + L"\" ";
    command.append(argument.begin(), argument.end());
    STARTUPINFOW info{};
    info.cb = sizeof(info);
    PROCESS_INFORMATION child{};
    // This probe must neither open another console nor inherit handles to the parent's log.
    if (!CreateProcessW(path.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                        nullptr, nullptr, &info, &child)) {
        LOG_ERROR(Frontend, "Could not start the Vulkan startup check (Win32 error {})",
                  GetLastError());
        return false;
    }
    SCOPE_EXIT {
        CloseHandle(child.hThread);
        CloseHandle(child.hProcess);
    };
    DWORD waited;
    do {
        waited = WaitForSingleObject(child.hProcess, 50);
    } while (waited == WAIT_TIMEOUT && steady_clock::now() < deadline &&
             !Veshiru::NativeConsole::ShutdownRequested());
    if (waited != WAIT_OBJECT_0) {
        TerminateProcess(child.hProcess, 1);
        if (!Veshiru::NativeConsole::ShutdownRequested()) {
            LOG_ERROR(Frontend, "Vulkan startup check did not complete; check the graphics driver");
        }
        return false;
    }
    DWORD exit_code{};
    const bool passed = GetExitCodeProcess(child.hProcess, &exit_code) && exit_code == 0;
#else
    pid_t child{};
    std::string arg = VulkanProbeArgument;
    char* argv[] = {const_cast<char*>(executable), arg.data(), nullptr};
    if (const auto error = posix_spawnp(&child, executable, nullptr, nullptr, argv, environ)) {
        LOG_ERROR(Frontend, "Could not start the Vulkan startup check (error {})", error);
        return false;
    }
    int status{};
    pid_t waited{};
    do {
        waited = waitpid(child, &status, WNOHANG);
        if (waited == child || (waited == -1 && errno != EINTR)) {
            break;
        }
        if (steady_clock::now() >= deadline) {
            kill(child, SIGKILL);
            while (waitpid(child, &status, 0) == -1 && errno == EINTR) {
            }
            LOG_ERROR(Frontend, "Vulkan startup check timed out; check the graphics driver");
            return false;
        }
        std::this_thread::sleep_for(milliseconds{15});
    } while (true);
    const bool passed = waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
    if (!passed) {
        LOG_ERROR(Frontend,
                  "Vulkan startup check failed: check that a Vulkan-capable GPU and current "
                  "graphics driver are installed; trying a fallback renderer");
    }
    return passed;
}
