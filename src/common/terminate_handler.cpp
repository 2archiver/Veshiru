// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <typeinfo>
#include <utility>

#if !defined(_MSC_VER) && __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define COMMON_HAS_CXXABI 1
#endif

#include "common/logging.h"
#include "common/terminate_handler.h"

namespace Common {
namespace {

/// The thrown type's name. Without RTTI (-fno-rtti, /GR-) only the C++ ABI can tell it, from the
/// exception's own type info; MSVC has no equivalent.
std::string CurrentExceptionType() {
#ifdef COMMON_HAS_CXXABI
    if (const std::type_info* type = abi::__cxa_current_exception_type()) {
        int status{};
        if (char* name = abi::__cxa_demangle(type->name(), nullptr, nullptr, &status)) {
            std::string result{name};
            std::free(name);
            return result;
        }
        return type->name();
    }
#endif
    return "of unknown type";
}

[[noreturn]] void OnTerminate() {
    // A throw while logging comes back here: abort. Another thread terminating meanwhile waits
    // for the first one's abort, so its message isn't cut short.
    static std::atomic_flag entered{};
    thread_local bool reentered{};
    if (std::exchange(reentered, true)) {
        std::abort();
    }
    if (entered.test_and_set()) {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds{1});
        }
    }
    if (const std::exception_ptr exception = std::current_exception()) {
        try {
            std::rethrow_exception(exception);
        } catch (const std::exception& e) {
            LOG_CRITICAL(Debug, "Uncaught exception {}: {}", CurrentExceptionType(), e.what());
        } catch (...) {
            LOG_CRITICAL(Debug, "Uncaught exception {}", CurrentExceptionType());
        }
    } else {
        LOG_CRITICAL(Debug, "std::terminate called without an active exception");
    }
    Log::Stop();
    std::abort();
}

} // Anonymous namespace

void InstallTerminateHandler() {
    std::set_terminate(OnTerminate);
}

} // namespace Common
