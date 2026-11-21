// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qt_common/discord/discord_impl.h"
#include "qt_common/discord/discord_payload.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <fmt/format.h>

#include "common/logging.h"
#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/loader/loader.h"
#include "qt_common/game_list/game_version.h"

namespace DiscordRPC {
namespace {

using Clock = std::chrono::steady_clock;

/// At least this long between SET_ACTIVITY frames on one connection: Discord rate-limits
/// presence updates, and the elapsed-time timestamp keeps counting client-side without us.
constexpr auto SEND_INTERVAL = std::chrono::seconds{15};
/// While connected and idle, how often the pipe is checked for replies to drain.
constexpr auto DRAIN_INTERVAL = std::chrono::milliseconds{1000};
/// How long an empty worker sleeps when there is nothing to connect to.
constexpr auto DORMANT_INTERVAL = std::chrono::hours{1};
/// Reconnect attempts after a failure or a closed pipe: 1, 2, 4, 8, 16, 32, 60, 60, ... s.
constexpr std::array<std::chrono::milliseconds, 7> RETRY_BACKOFFS{
    std::chrono::milliseconds{1000},  std::chrono::milliseconds{2000},
    std::chrono::milliseconds{4000},  std::chrono::milliseconds{8000},
    std::chrono::milliseconds{16000}, std::chrono::milliseconds{32000},
    std::chrono::milliseconds{60000},
};

std::int64_t UnixSecondsNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool IsApplicationId(std::string_view value) {
    return !value.empty() && value.size() <= 32 &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return c >= '0' && c <= '9';
           });
}

#ifdef _WIN32
std::uint32_t CurrentProcessId() {
    return static_cast<std::uint32_t>(GetCurrentProcessId());
}

constexpr DWORD IO_TIMEOUT_MS = 400;          ///< A stuck Discord must not stall the worker long
constexpr DWORD SHUTDOWN_IO_TIMEOUT_MS = 200; ///< Shutdown stays bounded well under a second

std::uint32_t ReadU32(const char* input) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(input[0])) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(input[1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(input[2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(input[3])) << 24);
}

bool Transfer(HANDLE pipe, void* buffer, DWORD size, bool reading, DWORD timeout_ms) {
    auto* bytes = static_cast<char*>(buffer);
    DWORD offset = 0;
    while (offset < size) {
        OVERLAPPED operation{};
        operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!operation.hEvent) {
            return false;
        }

        DWORD transferred = 0;
        const BOOL complete = reading
                                  ? ReadFile(pipe, bytes + offset, size - offset, &transferred,
                                             &operation)
                                  : WriteFile(pipe, bytes + offset, size - offset, &transferred,
                                              &operation);
        bool succeeded = complete != FALSE;
        if (!succeeded && GetLastError() == ERROR_IO_PENDING) {
            const DWORD wait = WaitForSingleObject(operation.hEvent, timeout_ms);
            if (wait == WAIT_OBJECT_0) {
                succeeded = GetOverlappedResult(pipe, &operation, &transferred, FALSE) != FALSE;
            } else {
                CancelIoEx(pipe, &operation);
                WaitForSingleObject(operation.hEvent, INFINITE);
                GetOverlappedResult(pipe, &operation, &transferred, FALSE);
            }
        }
        CloseHandle(operation.hEvent);
        if (!succeeded || transferred == 0) {
            return false;
        }
        offset += transferred;
    }
    return true;
}

bool WriteFrame(HANDLE pipe, std::uint32_t opcode, std::string_view payload, DWORD timeout_ms) {
    const std::string frame = BuildFrame(opcode, payload);
    return Transfer(pipe, const_cast<char*>(frame.data()), static_cast<DWORD>(frame.size()), false,
                    timeout_ms);
}

bool BytesWaiting(HANDLE pipe, DWORD& available) {
    DWORD readable = 0;
    if (!PeekNamedPipe(pipe, nullptr, 0, nullptr, &readable, nullptr)) {
        return false;
    }
    available = readable;
    return true;
}

/// Reads one frame if a complete one (header plus payload) is already waiting on the pipe.
/// Returns false only when the pipe is broken; a frame still on its way is not an error.
bool ReadWaitingFrame(HANDLE pipe, DWORD timeout_ms) {
    std::array<char, 8> header{};
    // The peek said the header is there, so this cannot block for long; the timeout only
    // guards a Discord that changed its mind mid-write.
    if (!Transfer(pipe, header.data(), static_cast<DWORD>(header.size()), true, timeout_ms)) {
        return false;
    }
    const std::uint32_t size = ReadU32(header.data() + 4);
    if (size > 1024 * 1024) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    std::string payload(size, '\0');
    return Transfer(pipe, payload.data(), size, true, timeout_ms);
}

/// Reads and discards every complete frame waiting on the pipe, so Discord's replies never
/// fill it and stall the connection.
bool DrainPipe(HANDLE pipe, DWORD timeout_ms) {
    for (int frames = 0; frames < 32; ++frames) {
        DWORD available = 0;
        if (!BytesWaiting(pipe, available)) {
            return false;
        }
        if (available < 8) {
            return true;
        }
        if (!ReadWaitingFrame(pipe, timeout_ms)) {
            return false;
        }
    }
    return true;
}

/// Reads one frame even if it has not fully arrived yet (bounded by the timeout).
bool ReadFrame(HANDLE pipe, DWORD timeout_ms) {
    std::array<char, 8> header{};
    if (!Transfer(pipe, header.data(), static_cast<DWORD>(header.size()), true, timeout_ms)) {
        return false;
    }
    const std::uint32_t size = ReadU32(header.data() + 4);
    if (size > 1024 * 1024) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    std::string payload(size, '\0');
    return Transfer(pipe, payload.data(), size, true, timeout_ms);
}

/// Opens one of discord-ipc-0..9 and completes the handshake. Returns the connected pipe.
HANDLE ConnectAndHandshake(std::string_view application_id) {
    for (int pipe_index = 0; pipe_index < 10; ++pipe_index) {
        const std::wstring pipe_name =
            L"\\\\.\\pipe\\discord-ipc-" + std::to_wstring(pipe_index);
        if (!WaitNamedPipeW(pipe_name.c_str(), 20)) {
            continue;
        }
        HANDLE pipe = CreateFileW(pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                  OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        if (pipe == INVALID_HANDLE_VALUE) {
            continue;
        }

        if (!WriteFrame(pipe, 0, BuildHandshakePayload(application_id), IO_TIMEOUT_MS)) {
            CloseHandle(pipe);
            continue;
        }
        // The handshake reply is read with the same bounded transfer: a Discord that accepts
        // the connection but never answers cannot stall the worker either.
        if (!ReadFrame(pipe, IO_TIMEOUT_MS)) {
            CloseHandle(pipe);
            continue;
        }
        return pipe;
    }
    return INVALID_HANDLE_VALUE;
}
#endif // _WIN32

/// The RPC connection's owner. The UI and emulation threads only ever push the latest activity
/// into a mutex-guarded slot; opening the pipe, handshaking, sending and draining all happen on
/// this thread, so a stuck or absent Discord can never block them.
///
/// A null activity means "no presence wanted" (the DiscordImpl was destroyed: the feature was
/// toggled off, or the app is shutting down): the worker clears the activity, says goodbye and
/// closes the pipe, then sleeps until a real activity arrives.
class RpcWorker final {
public:
    RpcWorker() : thread{[this] { Run(); }} {}

    ~RpcWorker() {
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        condition.notify_one();
        if (thread.joinable()) {
            thread.join();
        }
    }

    void SetApplicationId(std::string value) {
        {
            std::scoped_lock lock{mutex};
            if (application_id == value) {
                return;
            }
            application_id = std::move(value);
            ++activity_sequence;
        }
        condition.notify_one();
    }

    void SetActivity(std::optional<ActivityData> value) {
        {
            std::scoped_lock lock{mutex};
            if (activity == value) {
                return;
            }
            activity = std::move(value);
            ++activity_sequence;
        }
        condition.notify_one();
    }

private:
    void Run() {
#ifdef _WIN32
        std::uint64_t handled_sequence = 0;
        std::uint64_t nonce = 0;
        std::optional<ActivityData> sent; ///< What Discord last saw on this connection
        bool ever_sent = false;
        Clock::time_point last_send{};
        Clock::time_point next_reconnect = Clock::now();
        size_t backoff = 0;
        bool connected = false;
        std::string connected_id;
        std::optional<bool> reported; ///< Last availability state logged, if any

        const auto note_unavailable = [&] {
            if (!reported.has_value() || *reported) {
                LOG_INFO(Frontend, "discord: unavailable");
            }
            reported = false;
        };
        const auto note_connected = [&] {
            if (!reported.has_value() || !*reported) {
                LOG_INFO(Frontend, "discord: connected");
            }
            reported = true;
        };

        std::unique_lock lock{mutex};
        while (true) {
            const auto now = Clock::now();
            const auto next_wake = [&] {
                Clock::time_point wake = now + DORMANT_INTERVAL;
                if (!connected) {
                    if (activity.has_value() && IsApplicationId(application_id)) {
                        wake = std::min(wake, next_reconnect);
                    }
                } else if (activity_sequence != handled_sequence && ever_sent) {
                    // A change arrived inside the send limit: send the latest at its boundary.
                    wake = std::min(wake, last_send + SEND_INTERVAL);
                } else {
                    wake = std::min(wake, now + DRAIN_INTERVAL);
                }
                return wake;
            }();
            condition.wait_until(lock, next_wake);

            if (stopping) {
                break;
            }
            const std::uint64_t sequence = activity_sequence;
            const std::string app_id = application_id;
            const std::optional<ActivityData> current = activity;
            lock.unlock();

            if (!IsApplicationId(app_id) || !current.has_value()) {
                if (connected) {
                    // Nothing to show (feature off) or nothing to connect with: say goodbye.
                    Goodbye();
                    connected = false;
                    connected_id.clear();
                    handled_sequence = sequence;
                    ever_sent = false;
                    sent.reset();
                    last_send = {};
                }
                lock.lock();
                continue;
            }

            if (connected && app_id != connected_id) {
                // The application id changed: the handshake belongs to the old one.
                Goodbye();
                connected = false;
                connected_id.clear();
                next_reconnect = Clock::now();
                ever_sent = false;
                sent.reset();
                last_send = {};
            }

            if (!connected) {
                if (Clock::now() < next_reconnect) {
                    lock.lock();
                    continue;
                }
                HANDLE candidate = ConnectAndHandshake(app_id);
                if (candidate != INVALID_HANDLE_VALUE) {
                    pipe = candidate;
                    connected = true;
                    connected_id = app_id;
                    backoff = 0;
                    ever_sent = false;
                    sent.reset();
                    last_send = {};
                    note_connected();
                } else {
                    next_reconnect = Clock::now() + RETRY_BACKOFFS[backoff];
                    backoff = std::min(backoff + 1, RETRY_BACKOFFS.size() - 1);
                    note_unavailable();
                }
                lock.lock();
                continue;
            }

            // Connected: keep the pipe empty and check it is still alive.
            if (!DrainPipe(pipe, IO_TIMEOUT_MS)) {
                DropConnection();
                connected = false;
                connected_id.clear();
                next_reconnect = Clock::now() + RETRY_BACKOFFS[backoff];
                backoff = std::min(backoff + 1, RETRY_BACKOFFS.size() - 1);
                note_unavailable();
                lock.lock();
                continue;
            }

            const bool changed = sequence != handled_sequence || !ever_sent || sent != current;
            const bool send_due = !ever_sent || Clock::now() - last_send >= SEND_INTERVAL;
            if (changed && send_due) {
                const std::string nonce_text = fmt::format("{}", ++nonce);
                const std::string payload =
                    BuildSetActivityPayload(*current, CurrentProcessId(), nonce_text);
                if (WriteFrame(pipe, 1, payload, IO_TIMEOUT_MS) && DrainPipe(pipe, IO_TIMEOUT_MS)) {
                    sent = current;
                    handled_sequence = sequence;
                    ever_sent = true;
                    last_send = Clock::now();
                } else {
                    DropConnection();
                    connected = false;
                    connected_id.clear();
                    next_reconnect = Clock::now() + RETRY_BACKOFFS[backoff];
                    backoff = std::min(backoff + 1, RETRY_BACKOFFS.size() - 1);
                    note_unavailable();
                }
            }
            lock.lock();
        }
        lock.unlock();

        // Shutdown: tell Discord to drop the activity, say goodbye, and close, all bounded so a
        // stuck client cannot hold the join past a fraction of a second.
        if (connected) {
            Goodbye(SHUTDOWN_IO_TIMEOUT_MS);
        }
#else
        // No Discord IPC on this platform: the worker exists only so the interface is the same.
        std::unique_lock lock{mutex};
        while (!stopping) {
            condition.wait(lock, [this] { return stopping; });
        }
#endif
    }

#ifdef _WIN32
    /// Sends a clear-activity and an op 2 close, then closes the handle.
    void Goodbye(DWORD timeout_ms = IO_TIMEOUT_MS) {
        const std::string nonce_text = fmt::format("{}", ++goodbye_nonce);
        WriteFrame(pipe, 1, BuildClearActivityPayload(CurrentProcessId(), nonce_text), timeout_ms);
        WriteFrame(pipe, 2, {}, timeout_ms);
        DropConnection();
    }

    void DropConnection() {
        if (pipe != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe);
            pipe = INVALID_HANDLE_VALUE;
        }
    }

    HANDLE pipe = INVALID_HANDLE_VALUE;
    std::uint64_t goodbye_nonce{};
#endif

    std::mutex mutex;
    std::condition_variable condition;
    std::string application_id;
    std::optional<ActivityData> activity;
    std::uint64_t activity_sequence{};
    bool stopping{};
    std::thread thread;
};

RpcWorker& Worker() {
    static RpcWorker worker;
    return worker;
}

std::string CleanTitle(std::string title) {
    if (title.empty()) {
        return "Unknown game";
    }
    constexpr std::size_t max_bytes = 120;
    if (title.size() <= max_bytes) {
        return title;
    }
    std::size_t end = max_bytes;
    while (end > 0 && (static_cast<unsigned char>(title[end]) & 0xc0) == 0x80) {
        --end;
    }
    title.resize(end);
    return title;
}

} // namespace

DiscordImpl::DiscordImpl(Core::System& system_, std::string application_id)
    : system{system_}, application_id{std::move(application_id)} {
    if (this->application_id.empty()) {
        LOG_WARNING(Frontend,
                    "Discord Rich Presence is enabled but no Discord Application ID is configured; "
                    "RPC is disabled until one is supplied in Web settings.");
    } else if (!IsApplicationId(this->application_id)) {
        LOG_WARNING(Frontend,
                    "Discord Application ID must contain only decimal digits; Rich Presence is "
                    "disabled.");
    }
    Worker().SetApplicationId(this->application_id);
}

DiscordImpl::~DiscordImpl() {
    Worker().SetActivity(std::nullopt);
}

void DiscordImpl::Pause() {
    SetBrowsing();
}

void DiscordImpl::SetBrowsing() {
    current_game_key.clear();
    game_started_at = 0;
    Worker().SetActivity(ActivityData{.browsing = true});
}

void DiscordImpl::Update() {
    if (!system.IsPoweredOn()) {
        SetBrowsing();
        return;
    }

    std::string title;
    system.GetAppLoader().ReadTitle(title);
    title = CleanTitle(std::move(title));

    const u64 program_id = system.GetApplicationProcessProgramID();
    const FileSys::PatchManager patch_manager{program_id, system.GetFileSystemController(),
                                              system.GetContentProvider()};
    // The same resolver the game-list worker uses for the version badge, so the presence and
    // the library can never disagree. A running game's control metadata already reflects a
    // registered update; empty only when nothing at all is known.
    std::string version = GameListVersion::Resolve({}, patch_manager, system.GetAppLoader());
    if (version.empty()) {
        version = "1.0.0";
    }

    const std::string game_key = fmt::format("{:016X}:{}:{}", program_id, title, version);
    if (current_game_key != game_key || game_started_at == 0) {
        current_game_key = game_key;
        game_started_at = UnixSecondsNow();
    }
    Worker().SetActivity(ActivityData{
        .browsing = false,
        .game_title = std::move(title),
        .game_version = std::move(version),
        .started_at = game_started_at,
    });
}

} // namespace DiscordRPC
