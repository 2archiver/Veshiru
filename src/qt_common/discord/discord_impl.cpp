// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qt_common/discord/discord_impl.h"

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
#else
#include <unistd.h>
#endif

#include <fmt/format.h>

#include "common/logging.h"
#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/loader/loader.h"

namespace DiscordRPC {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto UPDATE_INTERVAL = std::chrono::seconds{15};
constexpr auto RETRY_INTERVAL = std::chrono::seconds{5};
#ifdef _WIN32
constexpr DWORD IO_TIMEOUT_MS = 400;
#endif

std::int64_t UnixSecondsNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::uint32_t CurrentProcessId() {
#ifdef _WIN32
    return static_cast<std::uint32_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint32_t>(getpid());
#endif
}

bool IsApplicationId(std::string_view value) {
    return !value.empty() && value.size() <= 32 &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return c >= '0' && c <= '9';
           });
}

#ifdef _WIN32
void AppendU32(std::string& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

std::uint32_t ReadU32(const char* input) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(input[0])) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(input[1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(input[2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(input[3])) << 24);
}

bool Transfer(HANDLE pipe, void* buffer, DWORD size, bool reading) {
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
            const DWORD wait = WaitForSingleObject(operation.hEvent, IO_TIMEOUT_MS);
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

bool WriteFrame(HANDLE pipe, std::uint32_t opcode, std::string_view payload) {
    if (payload.size() > (std::numeric_limits<std::uint32_t>::max)()) {
        return false;
    }
    std::string frame;
    frame.reserve(8 + payload.size());
    AppendU32(frame, opcode);
    AppendU32(frame, static_cast<std::uint32_t>(payload.size()));
    frame.append(payload);
    return Transfer(pipe, frame.data(), static_cast<DWORD>(frame.size()), false);
}

std::optional<std::pair<std::uint32_t, std::string>> ReadFrame(HANDLE pipe) {
    std::array<char, 8> header{};
    if (!Transfer(pipe, header.data(), static_cast<DWORD>(header.size()), true)) {
        return std::nullopt;
    }
    const std::uint32_t opcode = ReadU32(header.data());
    const std::uint32_t size = ReadU32(header.data() + 4);
    if (size > 1024 * 1024) {
        return std::nullopt;
    }
    std::string payload(size, '\0');
    if (size != 0 && !Transfer(pipe, payload.data(), size, true)) {
        return std::nullopt;
    }
    return std::pair{opcode, std::move(payload)};
}

bool SendToPipe(std::string_view application_id, const ActivityData* activity,
                std::uint64_t nonce) {
    for (int pipe_index = 0; pipe_index < 10; ++pipe_index) {
        const std::wstring pipe_name =
            L"\\\\?\\pipe\\discord-ipc-" + std::to_wstring(pipe_index);
        if (!WaitNamedPipeW(pipe_name.c_str(), 20)) {
            continue;
        }
        HANDLE pipe = CreateFileW(pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                  OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        if (pipe == INVALID_HANDLE_VALUE) {
            continue;
        }

        const std::string handshake = BuildHandshakePayload(application_id);
        bool ready = WriteFrame(pipe, 0, handshake);
        const auto response = ready ? ReadFrame(pipe) : std::nullopt;
        if (!response || response->first != 1) {
            CloseHandle(pipe);
            continue;
        }
        ready = response->second.find("\"evt\":\"READY\"") != std::string::npos;
        if (ready) {
            const std::string nonce_text = fmt::format("{}", nonce);
            const std::string command = activity
                                            ? BuildSetActivityPayload(*activity, CurrentProcessId(),
                                                                      nonce_text)
                                            : BuildClearActivityPayload(CurrentProcessId(),
                                                                        nonce_text);
            ready = WriteFrame(pipe, 1, command);
        }
        CloseHandle(pipe);
        if (ready) {
            return true;
        }
    }
    return false;
}
#else
bool SendToPipe(std::string_view, const ActivityData*, std::uint64_t) {
    return false;
}
#endif

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
        std::scoped_lock lock{mutex};
        if (application_id == value) {
            return;
        }
        application_id = std::move(value);
        ++sequence;
        condition.notify_one();
    }

    void SetActivity(std::optional<ActivityData> value) {
        std::scoped_lock lock{mutex};
        if (activity == value) {
            return;
        }
        activity = std::move(value);
        ++sequence;
        condition.notify_one();
    }

private:
    void Run() {
        std::uint64_t handled_sequence = 0;
        std::uint64_t nonce = 0;
        auto next_send = Clock::time_point::max();
        std::unique_lock lock{mutex};
        while (!stopping) {
            condition.wait_until(lock, next_send, [this, handled_sequence] {
                return stopping || sequence != handled_sequence;
            });
            if (stopping) {
                return;
            }

            const auto now = Clock::now();
            const bool changed = sequence != handled_sequence;
            if (!changed && now < next_send) {
                continue;
            }
            handled_sequence = sequence;
            const std::string app_id = application_id;
            const std::optional<ActivityData> current = activity;
            lock.unlock();

            bool sent = true;
            if (IsApplicationId(app_id)) {
                sent = SendToPipe(app_id, current ? &*current : nullptr, ++nonce);
            }
            next_send = Clock::now() + (sent ? UPDATE_INTERVAL : RETRY_INTERVAL);
            lock.lock();
        }
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::string application_id;
    std::optional<ActivityData> activity;
    std::uint64_t sequence{};
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
    const auto metadata = patch_manager.GetControlMetadata();
    std::string version = metadata.first != nullptr ? metadata.first->GetVersionString() : "";
    if (version.empty()) {
        system.GetAppLoader().ReadDisplayVersion(version);
    }
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
