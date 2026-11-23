// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/diag_request.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <mutex>
#include <string_view>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <system_error>

#include <fmt/format.h>

#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/frame_stats.h"
#include "common/logging.h"

namespace Common::DiagRequest {

constexpr s64 PollIntervalNs = 250'000'000;
constexpr u32 DefaultCount = 64;
constexpr u32 MaxCount = 100'000;
constexpr const char* RequestFileName = "nx2-request.txt";

/// The output folder shared by the commands of one request file.
struct Batch {
    std::filesystem::path root;
    std::optional<std::filesystem::path> directory;
};

namespace {

struct Registry {
    std::mutex mutex;
    std::map<std::string, Handler> handlers;
    std::map<std::string, std::string> unavailable;
    std::map<std::string, bool, std::less<>> try_switches;
    std::filesystem::path user_directory_override;
    u64 try_generation{};
};

Registry& GetRegistry() {
    static Registry registry;
    return registry;
}
bool TrySwitchDefaultEnabled(std::string_view name) {
    std::string env_name{"EDEN_NX2_"};
    env_name.append(name);
    const char* value = std::getenv(env_name.c_str());
    if (value != nullptr) {
        return value[0] == '1';
    }
    // TRY_RO_MIRROR_EXACT / TRY_STORAGE_RAW_WRITE_THROUGH: raw stores write through to the read-only
    // and storage VTX-HLE pool mirrors, so raw-writing draws skip the whole-pool snapshot (OoT
    // market: 345 -> 62 MiB copied each way per present, +2.5 fps). On by default; =0 opts out.
    return name == "TRY_NATIVE_F32_ATOMICS" || name == "TRY_ATOMIC_RESERVE" ||
           name == "TRY_RO_MIRROR_EXACT" || name == "TRY_STORAGE_RAW_WRITE_THROUGH";
}

bool TrySwitchEnabledLocked(const Registry& registry, std::string_view name) {
    if (const auto it = registry.try_switches.find(name); it != registry.try_switches.end()) {
        return it->second;
    }
    return TrySwitchDefaultEnabled(name);
}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REQUESTS");
        return value == nullptr || std::string_view{value} != "0";
    }();
    return enabled;
}

std::string Lower(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string TimestampName() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &local);
    return buffer;
}

void InstallBuiltins() {
    Register("textures", [](const Request& request) {
        const u32 count = request.NumberArg(0, DefaultCount);
        TextureBudget().Arm(count, request.Directory());
        LOG_INFO(Core, "[diag] textures: capturing the textures of the next {} draws into {}", count,
                 request.Directory().string());
    });
    Register("frame", [](const Request& request) {
        const u32 count = request.NumberArg(0, 1);
        FrameBudget().Arm(count, request.Directory());
        LOG_INFO(Core, "[diag] frame: capturing the next {} presented frames into {}", count,
                 request.Directory().string());
    });
    Register("stats", [](const Request& request) {
        if (!FrameStats::Enabled()) {
            LOG_WARNING(Core, "[diag] stats: frame statistics are off (start with EDEN_NX2_FRAME_STATS=1)");
            return;
        }
        FrameStats::DumpNow(request.Directory());
    });
    Register("try", [](const Request& request) {
        if (request.args.size() != 2 || Lower(request.args[0]) != "set") {
            LOG_WARNING(Core, "[diag] try: expected SET TRY_NAME=0|1");
            return;
        }
        const std::string_view assignment{request.args[1]};
        const size_t equals = assignment.find('=');
        if (equals == std::string_view::npos || (assignment.substr(equals + 1) != "0" &&
                                                  assignment.substr(equals + 1) != "1") ||
            !SetTrySwitch(assignment.substr(0, equals), assignment[equals + 1] == '1')) {
            LOG_WARNING(Core, "[diag] try: expected SET TRY_NAME=0|1");
            return;
        }
        LOG_INFO(Core, "[nx2] try {}={} applied at present {}", assignment.substr(0, equals),
                 assignment[equals + 1], FrameStats::PresentCount());
    });
}

} // namespace

namespace Detail {
std::atomic<s64> g_next_poll_ns{0};
} // namespace Detail

const std::filesystem::path& Request::Directory() const {
    Batch& b = *batch;
    if (!b.directory) {
        std::filesystem::path path = b.root / "diag" / TimestampName();
        std::error_code error;
        // Two request files inside one second get distinct folders.
        for (u32 n = 2; std::filesystem::exists(path, error) && n < 1000; ++n) {
            path = b.root / "diag" / fmt::format("{}-{}", TimestampName(), n);
        }
        std::filesystem::create_directories(path, error);
        if (error) {
            LOG_ERROR(Core, "[diag] cannot create {}: {}", path.string(), error.message());
        }
        LOG_INFO(Core, "[diag] output folder: {}", path.string());
        b.directory = std::move(path);
    }
    return *b.directory;
}

u32 Request::NumberArg(size_t index, u32 fallback) const {
    if (index >= args.size()) {
        return fallback;
    }
    const std::string& token = args[index];
    if (token.empty() || token[0] == '-' || token[0] == '+') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(token.c_str(), &end, 0);
    if (end == token.c_str() || *end != '\0') {
        return fallback;
    }
    return static_cast<u32>(std::min<unsigned long long>(value, MaxCount));
}

void Register(std::string command, Handler handler) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    registry.unavailable.erase(command);
    registry.handlers[std::move(command)] = std::move(handler);
}

void RegisterUnavailable(std::string command, std::string reason) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    if (!registry.handlers.contains(command)) {
        registry.unavailable[std::move(command)] = std::move(reason);
    }
}

bool TrySwitchEnabled(std::string_view name) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    return TrySwitchEnabledLocked(registry, name);
}

bool TrySwitchEnabled(std::string_view name, bool recommended_value) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    if (const auto it = registry.try_switches.find(name); it != registry.try_switches.end()) {
        return it->second;
    }
    std::string env_name{"EDEN_NX2_"};
    env_name.append(name);
    if (const char* value = std::getenv(env_name.c_str()); value != nullptr) {
        return value[0] == '1';
    }
    return recommended_value;
}

bool SetTrySwitch(std::string_view name, bool enabled) {
    if (!name.starts_with("TRY_") || name.size() <= 4 ||
        !std::all_of(name.begin(), name.end(), [](unsigned char c) {
            return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        })) {
        return false;
    }
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    const bool changed = TrySwitchEnabledLocked(registry, name) != enabled;
    registry.try_switches[std::string{name}] = enabled;
    if (changed) {
        ++registry.try_generation;
    }
    return true;
}

u64 TrySwitchGeneration() {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    return registry.try_generation;
}
bool TrySwitchDiffersFromDefault(std::string_view name) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    const auto it = registry.try_switches.find(name);
    return it != registry.try_switches.end() && it->second != TrySwitchDefaultEnabled(name);
}

void SetUserDirectoryForTesting(std::filesystem::path directory) {
    Registry& registry = GetRegistry();
    std::scoped_lock lock{registry.mutex};
    registry.user_directory_override = std::move(directory);
    Detail::g_next_poll_ns.store(0, std::memory_order_relaxed);
}

void Detail::PollSlow(s64 now_ns) {
    // Another thread may be polling at the same time: only one proceeds per interval.
    s64 expected = g_next_poll_ns.load(std::memory_order_relaxed);
    if (now_ns < expected ||
        !g_next_poll_ns.compare_exchange_strong(expected, now_ns + PollIntervalNs,
                                                std::memory_order_relaxed)) {
        return;
    }
    if (!Enabled()) {
        g_next_poll_ns.store(std::numeric_limits<s64>::max(), std::memory_order_relaxed);
        return;
    }
    static std::once_flag builtins;
    std::call_once(builtins, InstallBuiltins);

    Registry& registry = GetRegistry();
    std::filesystem::path root;
    {
        std::scoped_lock lock{registry.mutex};
        root = registry.user_directory_override;
    }
    if (root.empty()) {
        root = FS::GetEdenPath(FS::EdenPath::EdenDir);
    }
    const std::filesystem::path file = root / RequestFileName;
    std::error_code error;
    if (!std::filesystem::exists(file, error)) {
        return;
    }
    // Take the file first: a tool replacing it while we read must not lose its commands.
    const std::filesystem::path taken = root / (std::string{RequestFileName} + ".taken");
    std::filesystem::remove(taken, error);
    std::filesystem::rename(file, taken, error);
    if (error) {
        return; // the writer still holds it; try again at the next poll
    }
    std::string text;
    bool read_ok = false;
    {
        std::ifstream in{taken, std::ios::binary};
        if (in) {
            std::stringstream buffer;
            buffer << in.rdbuf();
            read_ok = !in.bad();
            text = buffer.str();
        }
    }
    if (!read_ok) {
        LOG_ERROR(Core, "[diag] cannot read {}", taken.string());
        // Put the request back so the next poll retries it, unless a newer one replaced it.
        if (!std::filesystem::exists(file, error)) {
            std::filesystem::rename(taken, file, error);
        }
        return;
    }
    std::filesystem::remove(taken, error);
    if (text.size() >= 3 && text.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        text.erase(0, 3);
    }

    Batch batch{.root = root, .directory = std::nullopt};
    std::istringstream lines{text};
    std::string line;
    while (std::getline(lines, line)) {
        while (!line.empty() && (line.back() == '\r' || std::isspace(static_cast<unsigned char>(line.back())))) {
            line.pop_back();
        }
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream words{line.substr(first)};
        Request request;
        words >> request.command;
        request.command = Lower(request.command);
        std::string word;
        while (words >> word) {
            request.args.push_back(std::move(word));
        }
        request.batch = &batch;
        LOG_INFO(Core, "[diag] request: {}", line.substr(first));
        Handler handler;
        std::string reason;
        {
            std::scoped_lock lock{registry.mutex};
            if (const auto it = registry.handlers.find(request.command); it != registry.handlers.end()) {
                handler = it->second;
            } else if (const auto un = registry.unavailable.find(request.command);
                       un != registry.unavailable.end()) {
                reason = un->second;
            }
        }
        if (handler) {
            handler(request);
        } else if (!reason.empty()) {
            LOG_WARNING(Core, "[diag] {}: unavailable: {}", request.command, reason);
        } else {
            LOG_WARNING(Core, "[diag] unknown command: {}", request.command);
        }
    }
}

void Budget::Arm(u32 count, std::filesystem::path dir) {
    std::scoped_lock lock{mutex};
    directory = std::move(dir);
    claimed = 0;
    epoch.fetch_add(1, std::memory_order_relaxed);
    remaining.store(count, std::memory_order_release);
}

bool Budget::Claim(std::filesystem::path& out_directory, u32* ordinal) {
    if (remaining.load(std::memory_order_acquire) == 0) {
        return false;
    }
    std::scoped_lock lock{mutex};
    const u32 left = remaining.load(std::memory_order_relaxed);
    if (left == 0) {
        return false;
    }
    remaining.store(left - 1, std::memory_order_release);
    out_directory = directory;
    if (ordinal != nullptr) {
        *ordinal = claimed;
    }
    ++claimed;
    return true;
}

std::filesystem::path Budget::Directory() const {
    std::scoped_lock lock{mutex};
    return directory;
}

Budget& TextureBudget() {
    static Budget budget;
    return budget;
}

Budget& FrameBudget() {
    static Budget budget;
    return budget;
}

} // namespace Common::DiagRequest
