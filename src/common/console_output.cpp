// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

#include "common/console_output.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__OPENORBIS__)
#include <cstdio>
#else
#include <cerrno>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace Common::Log {

#ifdef _WIN32
namespace {

constexpr WORD White = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
constexpr WORD ForegroundMask = White | FOREGROUND_INTENSITY;

WORD Foreground(ConsoleColor color) {
    switch (color) {
    case ConsoleColor::Warning:
        return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
    case ConsoleColor::Error:
        return FOREGROUND_RED | FOREGROUND_INTENSITY;
    default:
        return White;
    }
}

} // namespace
#endif

void WriteStandardStream(std::string_view text, bool to_stderr,
                         [[maybe_unused]] ConsoleColor color) noexcept {
#ifdef _WIN32
    const HANDLE handle = GetStdHandle(to_stderr ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
    if (!handle || handle == INVALID_HANDLE_VALUE || text.empty()) {
        return;
    }
    DWORD mode{};
    if (GetConsoleMode(handle, &mode)) {
        try {
            // Do not change the parent shell's code page.
            const auto size = static_cast<int>(text.size());
            const int count = MultiByteToWideChar(CP_UTF8, 0, text.data(), size, nullptr, 0);
            if (count > 0) {
                std::wstring wide(static_cast<std::size_t>(count), L'\0');
                MultiByteToWideChar(CP_UTF8, 0, text.data(), size, wide.data(), count);
                // Console attributes are shared by stdout and stderr. Keep the change and restore
                // in one critical section, without locking file or pipe writes.
                static std::mutex console_mutex;
                std::scoped_lock lock{console_mutex};
                CONSOLE_SCREEN_BUFFER_INFO info{};
                const bool restore = color != ConsoleColor::Default &&
                                     GetConsoleScreenBufferInfo(handle, &info) &&
                                     SetConsoleTextAttribute(handle, static_cast<WORD>(
                                         (info.wAttributes & ~ForegroundMask) | Foreground(color)));
                DWORD written{};
                WriteConsoleW(handle, wide.data(), static_cast<DWORD>(wide.size()), &written,
                              nullptr);
                if (restore) {
                    SetConsoleTextAttribute(handle, info.wAttributes);
                }
            }
        } catch (...) {
            // A diagnostic should not cause termination on allocation failure.
        }
        return;
    }
    while (!text.empty()) {
        DWORD written{};
        if (!WriteFile(handle, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) ||
            written == 0) {
            break;
        }
        text.remove_prefix(written);
    }
#elif defined(__OPENORBIS__)
    auto* stream = to_stderr ? stderr : stdout;
    std::fwrite(text.data(), 1, text.size(), stream);
    std::fflush(stream);
#else
    const int fd = to_stderr ? STDERR_FILENO : STDOUT_FILENO;
    // Protect only this write, without altering the frontend's SIGPIPE handler or the parent
    // shell's open-file flags. Consume our own SIGPIPE before restoring the mask.
    sigset_t signals, old_mask, pending;
    sigemptyset(&signals);
    sigaddset(&signals, SIGPIPE);
    if (pthread_sigmask(SIG_BLOCK, &signals, &old_mask) != 0) {
        return;
    }
    sigpending(&pending);
    while (!text.empty()) {
        const auto written = write(fd, text.data(), text.size());
        if (written > 0) {
            text.remove_prefix(static_cast<std::size_t>(written));
        } else if (written < 0 && errno != EINTR) {
            if (errno == EPIPE && !sigismember(&pending, SIGPIPE)) {
                // When SIGPIPE is ignored it is not queued. Checking first keeps sigwait safe
                // on platforms that don't provide sigtimedwait (macOS).
                sigset_t now;
                sigpending(&now);
                if (sigismember(&now, SIGPIPE)) {
                    int signal{};
                    sigwait(&signals, &signal);
                }
            }
            break;
        } else if (written == 0) {
            break;
        }
    }
    pthread_sigmask(SIG_SETMASK, &old_mask, nullptr);
#endif
}

struct StandardStreamSink::State {
    struct Line {
        std::string text;
        bool to_stderr;
        ConsoleColor color;
    };

    void Run() {
        std::unique_lock lock{mutex};
#ifdef _WIN32
        thread_id = GetCurrentThreadId();
#endif
        for (;;) {
            changed.wait(lock, [this] { return closing || !pending.empty(); });
            if (pending.empty()) {
                done = true;
                changed.notify_all();
                return;
            }
            auto line = std::move(pending.front());
            pending.pop_front();
            writing = true;
            lock.unlock();
            WriteStandardStream(line.text, line.to_stderr, line.color);
            lock.lock();
            writing = false;
            if (pending.empty()) {
                changed.notify_all();
            }
        }
    }

    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Line> pending;
    bool writing{};
    bool closing{};
    bool done{};
#ifdef _WIN32
    DWORD thread_id{};
#endif
};

StandardStreamSink::StandardStreamSink()
    : state{std::make_shared<State>()}, writer{[owned = state] { owned->Run(); }} {}

StandardStreamSink::~StandardStreamSink() {
    Close();
}

void StandardStreamSink::Enqueue(std::string text, bool to_stderr, bool important,
                                 ConsoleColor color) {
    constexpr std::size_t limit = 512;
    constexpr std::size_t reserve = 32;
    std::scoped_lock lock{state->mutex};
    if (!state->closing && state->pending.size() < (important ? limit : limit - reserve)) {
        state->pending.push_back(State::Line{std::move(text), to_stderr, color});
        state->changed.notify_one();
    }
}

void StandardStreamSink::Flush(std::chrono::milliseconds timeout) {
    std::unique_lock lock{state->mutex};
    state->changed.wait_for(lock, timeout,
                            [this] { return !state->writing && state->pending.empty(); });
}

void StandardStreamSink::Close() {
    using namespace std::chrono_literals;
    if (!writer.joinable()) {
        return;
    }
    std::unique_lock lock{state->mutex};
    state->closing = true;
    state->changed.notify_all();
    const auto done = [this] { return state->done; };
    if (!state->changed.wait_for(lock, 500ms, done)) {
        state->pending.clear();
#ifdef _WIN32
        // Redirects can be inherited synchronous handles, so OVERLAPPED cannot be assumed.
        if (const HANDLE thread = OpenThread(THREAD_TERMINATE, FALSE, state->thread_id)) {
            CancelSynchronousIo(thread);
            CloseHandle(thread);
        }
#endif
        if (!state->changed.wait_for(lock, 50ms, done)) {
            // An uncooperative reader/host must not keep the process alive. The in-flight write
            // owns its text and state and touches neither the logger nor any frontend objects.
            writer.detach();
            return;
        }
    }
    lock.unlock();
    writer.join();
}

} // namespace Common::Log
