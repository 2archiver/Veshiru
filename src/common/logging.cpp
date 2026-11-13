// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <functional>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <regex>
#include <stop_token>
#include <thread>
#include <unordered_map>

#if defined(__ANDROID__)
#include <android/log.h>
#endif
#ifdef _WIN32
#include <windows.h> // For OutputDebugStringW
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <boost/algorithm/string/replace.hpp>
#include <fmt/ranges.h>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/literals.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"

#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "common/bounded_threadsafe_queue.h"

namespace Common::Log {

/// @brief A log entry. Log entries are store in a structured format to permit more varied output
/// formatting on different frontends, as well as facilitating filtering and aggregation.
struct Entry {
    char const* message = nullptr;
    size_t message_len = 0;
    std::chrono::microseconds timestamp;
    Class log_class{};
    Level log_level{};
    const char* filename = nullptr;
    const char* function = nullptr;
    uint32_t line_num = 0;
};

namespace {

/// @brief Returns the name of the passed log class as a C-string. Subclasses are separated by periods
/// instead of underscores as in the enumeration.
/// @note GetClassName is a macro defined by Windows.h, grrr...
const char* GetLogClassNameImpl(Class log_class) {
    switch (log_class) {
#define CLS(x) case Class::x: return #x;
#define SUB(x, y) case Class::x##_##y: return #x "." #y;
#include "common/log_classes.inc"
#undef CLS
#undef SUB
    default: return "?";
    }
}

/// @brief Returns the name of the passed log level as a C-string.
const char* GetLevelNameImpl(Level log_level) {
    switch (log_level) {
#define LVL(x) case Level::x: return #x;
    LVL(Trace)
    LVL(Debug)
    LVL(Info)
    LVL(Warning)
    LVL(Error)
    LVL(Critical)
#undef LVL
    default: return "?";
    }
}

// Some IDEs prefer <file>:<line> instead, so let's just do that :)
std::string FormatLogMessage(const Entry& entry) noexcept {
    if (!entry.filename) return "";
    auto const time_seconds = uint32_t(entry.timestamp.count() / 1000000);
    auto const time_fractional = uint32_t(entry.timestamp.count() % 1000000);
    auto const class_name = GetLogClassNameImpl(entry.log_class);
    auto const level_name = GetLevelNameImpl(entry.log_level);
    return fmt::format("[{:4d}.{:06d}] {} <{}> {}:{}:{}: {}\n", time_seconds, time_fractional, class_name, level_name, entry.filename, entry.line_num, entry.function, entry.message);
}

template <typename It>
Level GetLevelByName(const It begin, const It end) {
    for (u32 i = 0; i < u32(Level::Count); ++i) {
        const char* level_name = GetLevelNameImpl(Level(i));
        if (Common::ComparePartialString(begin, end, level_name))
            return Level(i);
    }
    return Level::Count;
}

template <typename It>
Class GetClassByName(const It begin, const It end) {
    for (u32 i = 0; i < u32(Class::Count); ++i) {
        const char* level_name = GetLogClassNameImpl(Class(i));
        if (Common::ComparePartialString(begin, end, level_name))
            return Class(i);
    }
    return Class::Count;
}

template <typename Iterator>
bool ParseFilterRule(Filter& instance, Iterator begin, Iterator end) {
    auto level_separator = std::find(begin, end, ':');
    if (level_separator == end) {
        LOG_ERROR(Log, "Invalid log filter. Must specify a log level after `:`: {}", std::string(begin, end));
        return false;
    }
    const Level level = GetLevelByName(level_separator + 1, end);
    if (level == Level::Count) {
        LOG_ERROR(Log, "Unknown log level in filter: {}", std::string(begin, end));
        return false;
    }
    if (Common::ComparePartialString(begin, level_separator, "*")) {
        instance.class_levels.fill(level);
        return true;
    }
    const Class log_class = GetClassByName(begin, level_separator);
    if (log_class == Class::Count) {
        LOG_ERROR(Log, "Unknown log class in filter: {}", std::string(begin, end));
        return false;
    }
    instance.SetClassLevel(log_class, level);
    return true;
}

/// @brief Trims up to and including the last of ../, ..\, src/, src\ in a string
/// do not be fooled this isn't generating new strings on .rodata :)
constexpr const char* TrimSourcePath(std::string_view source) noexcept {
    const auto rfind = [source](const std::string_view match) {
        return source.rfind(match) == source.npos ? 0 : (source.rfind(match) + match.size());
    };
    auto idx = (std::max)({rfind("src/"), rfind("src\\"), rfind("../"), rfind("..\\")});
    return source.data() + idx;
}

Subsystem GetSubsystemImpl(Class log_class) noexcept {
    switch (log_class) {
    case Class::Core:
    case Class::Core_ARM:
    case Class::Kernel:
    case Class::Kernel_SVC:
    case Class::Debug_Emulated:
    case Class::Debug_Breakpoint:
    case Class::Debug_GDBStub:
        return Subsystem::CPU;
    case Class::HW_GPU:
    case Class::Render:
    case Class::Render_Software:
    case Class::Render_OpenGL:
    case Class::Render_Vulkan:
    case Class::Shader:
    case Class::Shader_SPIRV:
    case Class::Shader_GLASM:
    case Class::Shader_GLSL:
    case Class::Debug_GPU:
    case Class::Service_NVDRV:
    case Class::Service_Nvnflinger:
    case Class::Service_VI:
        return Subsystem::GPU;
    case Class::Service:
    case Class::Service_ACC:
    case Class::Service_AM:
    case Class::Service_AOC:
    case Class::Service_APM:
    case Class::Service_ARP:
    case Class::Service_BCAT:
    case Class::Service_BPC:
    case Class::Service_BGTC:
    case Class::Service_BTDRV:
    case Class::Service_BTM:
    case Class::Service_Capture:
    case Class::Service_ERPT:
    case Class::Service_ETicket:
    case Class::Service_EUPLD:
    case Class::Service_Fatal:
    case Class::Service_FGM:
    case Class::Service_Friend:
    case Class::Service_GRC:
    case Class::Service_HID:
    case Class::Service_IRS:
    case Class::Service_JIT:
    case Class::Service_LBL:
    case Class::Service_LDR:
    case Class::Service_LM:
    case Class::Service_Migration:
    case Class::Service_Mii:
    case Class::Service_MM:
    case Class::Service_MNPP:
    case Class::Service_NCM:
    case Class::Service_NFC:
    case Class::Service_NFP:
    case Class::Service_NGC:
    case Class::Service_NIM:
    case Class::Service_NOTIF:
    case Class::Service_NPNS:
    case Class::Service_NS:
    case Class::Service_OLSC:
    case Class::Service_PCIE:
    case Class::Service_PCTL:
    case Class::Service_PCV:
    case Class::Service_PM:
    case Class::Service_PREPO:
    case Class::Service_PSC:
    case Class::Service_PTM:
    case Class::Service_SET:
    case Class::Service_SM:
    case Class::Service_SPL:
    case Class::Service_TCAP:
    case Class::Service_Time:
    case Class::Service_USB:
    case Class::IPC:
        return Subsystem::HLE;
    case Class::Common_Filesystem:
    case Class::Service_FS:
    case Class::Loader:
        return Subsystem::FS;
    case Class::Audio:
    case Class::Audio_DSP:
    case Class::Audio_Sink:
    case Class::Service_Audio:
        return Subsystem::Audio;
    case Class::Network:
    case Class::Service_LDN:
    case Class::Service_NIFM:
    case Class::Service_SSL:
    case Class::Service_WLAN:
    case Class::WebService:
        return Subsystem::Networking;
    case Class::Core_Timing:
    case Class::Performance:
    case Class::HW_Memory:
    case Class::Common_Memory:
        return Subsystem::Performance;
    case Class::Compatibility:
        return Subsystem::Compatibility;
    default:
        return Subsystem::Other;
    }
}

std::string SanitizeLogMessageImpl(std::string_view message) {
    std::string text{message};
#ifndef __ANDROID__
    if (Settings::values.censor_username.GetValue()) {
        static const std::string username = []() -> std::string {
            for (const auto* var : {"LOGNAME", "USERNAME", "USER"}) {
                if (const auto* value = std::getenv(var); value != nullptr && *value != '\0') {
                    return std::string{value};
                }
            }
            return {};
        }();
        if (!username.empty()) {
            boost::replace_all(text, username, "<user>");
        }
    }
#endif
    static const std::string home = []() -> std::string {
#ifdef _WIN32
        if (const auto* value = std::getenv("USERPROFILE"); value != nullptr) {
            return std::string{value};
        }
        return {};
#else
        return FS::PathToUTF8String(FS::GetHomeDirectory());
#endif
    }();
    static const std::string appdata = []() -> std::string {
#ifdef _WIN32
        return FS::PathToUTF8String(FS::GetAppDataRoamingDirectory());
#else
        return {};
#endif
    }();
    if (!home.empty()) {
        boost::replace_all(text, home, "<home>");
    }
    if (!appdata.empty()) {
        boost::replace_all(text, appdata, "<appdata>");
    }

    // Absolute paths can name personal files. Keep relative source paths and title IDs, but
    // replace host file locations with a stable placeholder.
    static const std::regex windows_path{R"(\b[A-Za-z]:[\\/][^\s"'<>]*)"};
    static const std::regex unc_path{R"(\\\\[^\s"'<>]*)"};
    static const std::regex posix_path{
        R"((^|[\s"'(=])(/(?:home|Users|tmp|var|mnt|media|run|srv|opt|root)/[^\s"'<>]*))"};
    static const std::regex credential{
        R"((authorization\s*:\s*bearer\s+|token\s*[:=]\s*|api[_-]?key\s*[:=]\s*|password\s*[:=]\s*|secret\s*[:=]\s*)([^\s,;]+))",
        std::regex::icase};
    static const std::regex url_userinfo{R"((https?://)[^\s/@:]+:[^\s/@]+@)"};
    text = std::regex_replace(text, windows_path, "<path>");
    text = std::regex_replace(text, unc_path, "<path>");
    text = std::regex_replace(text, posix_path, "$1<path>");
    text = std::regex_replace(text, credential, "$1<redacted>");
    text = std::regex_replace(text, url_userinfo, "$1<redacted>@");
    return text;
}

/// @brief Interface for logging backends.
struct Backend {
    virtual ~Backend() noexcept = default;
    virtual void Write(const Entry& entry) noexcept = 0;
    virtual void Flush() noexcept = 0;
};

/// @brief Instead of using fmt::format() just use the system's formatting capabilities directly
struct DirectFormatArgs {
    const char *class_name;
    const char *level_name;
    uint32_t time_seconds;
    uint32_t time_fractional;
};
[[nodiscard]] inline DirectFormatArgs GetDirectFormatArgs(Entry const& entry) noexcept {
    return {
        .class_name = GetLogClassNameImpl(entry.log_class),
        .level_name = GetLevelNameImpl(entry.log_level),
        .time_seconds = uint32_t(entry.timestamp.count() / 1000000),
        .time_fractional = uint32_t(entry.timestamp.count() % 1000000),
    };
}

/// @brief Backend that writes to stdout and with color
struct ColorConsoleBackend final : public Backend {
#ifdef _WIN32
    explicit ColorConsoleBackend() noexcept {
        console_handle = GetStdHandle(STD_ERROR_HANDLE);
        GetConsoleScreenBufferInfo(console_handle, &original_info);
    }
    ~ColorConsoleBackend() noexcept override {
        SetConsoleTextAttribute(console_handle, original_info.wAttributes);
    }
    void Write(const Entry& entry) noexcept override {
        if (enabled && console_handle != INVALID_HANDLE_VALUE) {
            WORD color = WORD([&entry]() {
                switch (entry.log_level) {
                case Level::Debug: return FOREGROUND_GREEN | FOREGROUND_BLUE; // Cyan
                case Level::Info: return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; // Bright gray
                case Level::Warning: return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
                case Level::Error: return FOREGROUND_RED | FOREGROUND_INTENSITY;
                case Level::Critical: return FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
                default: break;
                }
                return FOREGROUND_INTENSITY; // Grey
            }());
            SetConsoleTextAttribute(console_handle, color);
            auto const df = GetDirectFormatArgs(entry);
            // One write per line: MinGW's fprintf emits each character separately, and stdout is
            // unbuffered here, so every character became a WriteFile on the thread that logs.
            fmt::memory_buffer line;
            fmt::format_to(fmt::appender(line), "[{:4d}.{:06d}] {} <{}> {}:{}:{}: ", df.time_seconds, df.time_fractional, df.class_name, df.level_name, entry.filename, entry.line_num, entry.function);
            line.append(entry.message, entry.message + entry.message_len);
            line.push_back('\n');
            std::fwrite(line.data(), 1, line.size(), stdout);
        }
    }
    void Flush() noexcept override {}
    CONSOLE_SCREEN_BUFFER_INFO original_info = {};
    HANDLE console_handle = INVALID_HANDLE_VALUE;
    std::atomic_bool enabled = false;
#else // ^^^ Windows vvv POSIX
    explicit ColorConsoleBackend() noexcept {}
    ~ColorConsoleBackend() noexcept override {}
    void Write(const Entry& entry) noexcept override {
        if (enabled) {
            auto const color_str = [&entry]() -> const char* {
                switch (entry.log_level) {
                case Level::Debug: return "[0;36m"; // Cyan
                case Level::Info: return "[0;37m"; // Bright gray
                case Level::Warning: return "[1;33m"; // Bright yellow
                case Level::Error: return "[1;31m"; // Bright red
                case Level::Critical: return "[1;35m"; // Bright magenta
                default: return "[1;30m"; // Grey
                }
            }();
            auto const df = GetDirectFormatArgs(entry);
            // more restrictive, because take for example this simple prelude:
            // [  50.872256] Config <Info> common/settings.cpp:142:LogSettings:
            char buffer[256];
            auto result = fmt::format_to_n(buffer, sizeof(buffer) - 1, "\x1b{}[{:4d}.{:06d}] {} <{}> {}:{}:{}: ", color_str, df.time_seconds, df.time_fractional, df.class_name, df.level_name, entry.filename, entry.line_num, entry.function, entry.message);
            std::fwrite(buffer, 1, (std::min)(sizeof(buffer) - 1, result.size), stdout);
            std::fwrite(entry.message, 1, entry.message_len, stdout);
            std::fwrite("\x1b[0m\n", 1, sizeof("\x1b[0m\n"), stdout);
        }
    }
    void Flush() noexcept override {
        std::fflush(stdout);
    }
    std::atomic_bool enabled = false;
#endif
};

#ifndef __OPENORBIS__
/// @brief Backend that writes to a file passed into the constructor
struct FileBackend final : public Backend {
    explicit FileBackend(const std::filesystem::path filename) noexcept {
        // Move a log over its ".old.txt" copy. The rename replaces the copy atomically, and fails
        // if the log is still open elsewhere (e.g. in a Veshiru process that hasn't exited yet).
        // A log that can't even be checked counts as a failure, so it is never truncated.
        const auto rotate = [](const std::filesystem::path& path) {
            std::error_code ec;
            if (!std::filesystem::exists(path, ec))
                return !ec;
            auto old_path = path;
            old_path += ".old.txt";
            std::filesystem::rename(path, old_path, ec);
            return !ec;
        };
        // If the log can't be moved, write this session to a second log (eden_log_2.txt) so that
        // the previous session is kept and this one still gets the full size limit. If that one
        // can't be moved either, append to it.
        auto path = filename;
        auto mode = FS::FileAccessMode::Write;
        if (!rotate(path)) {
            path = filename.parent_path() / filename.stem();
            path += "_2";
            path += filename.extension();
            if (!rotate(path))
                mode = FS::FileAccessMode::Append;
        }
        file.emplace(path, mode, FS::FileType::TextFile);
        // Count what is already there so the size limit still applies when appending.
        if (mode == FS::FileAccessMode::Append)
            bytes_written = file->GetSize();
    }
    ~FileBackend() noexcept override = default;

    void Write(const Entry& entry) noexcept override {
        if (!enabled)
            return;

        auto message = FormatLogMessage(entry);
#ifndef __ANDROID__
        if (Settings::values.censor_username.GetValue()) {
            // This must be a static otherwise it would get checked on EVERY
            // instance of logging an entry...
            static std::string username = []() -> std::string {
                // in order of precedence
                // LOGNAME usually works on UNIX, USERNAME on Windows
                // Some UNIX systems suck and don't use LOGNAME so we also need USER :(
                for (auto const var : { "LOGNAME", "USERNAME", "USER", })
                    if (auto const s = ::getenv(var); s != nullptr)
                        return std::string{s};
                return std::string{};
            }();
            if (!username.empty())
                boost::replace_all(message, username, "user");
        }
#endif
        bytes_written += file->WriteSpan(std::span<const char>{message.begin(), message.end()});

        // Option to log each line rather than 4k buffers
        if (Settings::values.log_flush_line.GetValue())
            file->Flush();

        using namespace Common::Literals;
        // Prevent logs from exceeding a set maximum size in the event that log entries are spammed.
        const auto write_limit = Settings::values.extended_logging.GetValue() ? 1_GiB : 100_MiB;
        const bool write_limit_exceeded = bytes_written > write_limit;
        if (entry.log_level >= Level::Error || write_limit_exceeded) {
            // Stop writing after the write limit is exceeded.
            // Don't close the file so we can print a stacktrace if necessary
            if (write_limit_exceeded)
                enabled = false;
            file->Flush();
        }
    }

    void Flush() noexcept override {
        file->Flush();
    }
private:
    std::optional<FS::IOFile> file;
    std::size_t bytes_written = 0;
    bool enabled = true;
};
#endif
/// @brief Backend that keeps the last lines in memory, off unless the debug server turns it on
struct RingBackend final : public Backend {
    void Write(const Entry& entry) noexcept override {
        if (capacity.load(std::memory_order_relaxed) == 0) {
            return;
        }
        try {
            std::string text = FormatLogMessage(entry);
            if (!text.empty() && text.back() == '\n') {
                text.pop_back();
            }
            std::scoped_lock lock{mutex};
            lines.push_back({next_sequence++, std::move(text)});
            while (lines.size() > capacity.load(std::memory_order_relaxed)) {
                lines.pop_front();
            }
        } catch (const std::exception&) {
            // Out of memory: drop the line rather than the process.
        }
    }
    void Flush() noexcept override {}

    std::atomic<std::size_t> capacity{};
    std::mutex mutex;
    std::deque<RingLine> lines;
    u64 next_sequence = 1;
};
/// @brief Asynchronous, bounded backend for the Qt live console. Logging threads only copy the
/// already-formatted message into a bounded MPSC queue; sanitising, rate limiting and buffering
/// happen on a worker that is started only while the console is open.
struct LiveConsoleBackend final : public Backend {
    struct Pending {
        std::string message;
        const char* filename{};
        const char* function{};
        u32 line_num{};
        u64 timestamp_us{};
        Class log_class{};
        Level log_level{};
    };

    void Write(const Entry& entry) noexcept override {
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }
        try {
            Pending pending{
                .message = std::string(entry.message, entry.message_len),
                .filename = entry.filename,
                .function = entry.function,
                .line_num = entry.line_num,
                .timestamp_us = static_cast<u64>(entry.timestamp.count()),
                .log_class = entry.log_class,
                .log_level = entry.log_level,
            };
            if (!queue.TryEmplace(std::move(pending))) {
                dropped_queue.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) {
            dropped_queue.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void Flush() noexcept override {}

    void Enable() {
        bool expected = false;
        if (!enabled.compare_exchange_strong(expected, true)) {
            return;
        }
        if (!worker.joinable()) {
            worker = std::jthread([this](std::stop_token stop_token) { Loop(stop_token); });
        }
    }

    void Disable() {
        enabled.store(false, std::memory_order_relaxed);
        if (worker.joinable()) {
            worker.request_stop();
            worker.join();
        }
    }

    void Loop(std::stop_token stop_token) noexcept {
        Pending pending;
        while (!stop_token.stop_requested()) {
            if (!queue.TryPop(pending)) {
                queue.PopWait(pending, stop_token);
                if (stop_token.stop_requested()) {
                    break;
                }
            }
            Process(pending);
            while (queue.TryPop(pending)) {
                Process(pending);
            }
        }
        // Drain what was already queued when the console closed so the bounded buffer and the
        // dropped counters stay consistent.
        while (queue.TryPop(pending)) {
            Process(pending);
        }
    }

    void Process(const Pending& pending) noexcept {
        try {
            constexpr u64 window_us = 1'000'000;
            constexpr u32 max_per_window = 32;
            const std::size_t key = std::hash<std::string>{}(pending.message) ^
                                    (static_cast<std::size_t>(pending.log_class) << 8) ^
                                    (static_cast<std::size_t>(pending.log_level) << 16);
            auto& state = rate_limits[key];
            if (pending.timestamp_us - state.window_start_us >= window_us) {
                state.window_start_us = pending.timestamp_us;
                state.count = 0;
            }
            if (++state.count > max_per_window) {
                rate_limited.fetch_add(1, std::memory_order_relaxed);
                if (rate_limits.size() > 4096) {
                    rate_limits.clear();
                }
                return;
            }
            if (rate_limits.size() > 4096) {
                rate_limits.clear();
            }

            ConsoleEntry entry{
                .sequence = next_sequence++,
                .timestamp_us = pending.timestamp_us,
                .log_class = pending.log_class,
                .log_level = pending.log_level,
                .subsystem = GetSubsystemImpl(pending.log_class),
                .message = SanitizeLogMessageImpl(pending.message),
                .source = fmt::format("{}:{}:{}", pending.filename ? pending.filename : "?",
                                      pending.line_num,
                                      pending.function ? pending.function : "?"),
            };
            std::scoped_lock lock{mutex};
            entries.push_back(std::move(entry));
            const std::size_t limit = capacity.load(std::memory_order_relaxed);
            while (entries.size() > limit) {
                entries.pop_front();
                dropped_buffer.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) {
            dropped_buffer.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::vector<ConsoleEntry> Read(u64 after, std::size_t max_entries) {
        std::vector<ConsoleEntry> result;
        if (max_entries == 0) {
            return result;
        }
        std::scoped_lock lock{mutex};
        result.reserve((std::min)(max_entries, entries.size()));
        for (const auto& entry : entries) {
            if (entry.sequence > after) {
                result.push_back(entry);
                if (result.size() >= max_entries) {
                    break;
                }
            }
        }
        return result;
    }

    void Clear() {
        std::scoped_lock lock{mutex};
        entries.clear();
    }

    ConsoleStats Stats() {
        ConsoleStats stats{
            .dropped_queue = dropped_queue.load(std::memory_order_relaxed),
            .dropped_buffer = dropped_buffer.load(std::memory_order_relaxed),
            .rate_limited = rate_limited.load(std::memory_order_relaxed),
        };
        std::scoped_lock lock{mutex};
        stats.buffered = entries.size();
        return stats;
    }

    struct RateState {
        u64 window_start_us{};
        u32 count{};
    };

    std::atomic_bool enabled{false};
    std::atomic<std::size_t> capacity{10000};
    Common::MPSCQueue<Pending, 4096> queue;
    std::jthread worker;
    std::mutex mutex;
    std::deque<ConsoleEntry> entries;
    std::unordered_map<std::size_t, RateState> rate_limits;
    u64 next_sequence{1};
    std::atomic<u64> dropped_queue{0};
    std::atomic<u64> dropped_buffer{0};
    std::atomic<u64> rate_limited{0};
};

#ifdef _WIN32
/// @brief Backend that writes to Visual Studio's output window
struct DebuggerBackend final : public Backend {
    explicit DebuggerBackend() noexcept = default;
    ~DebuggerBackend() noexcept override = default;
    void Write(const Entry& entry) noexcept override {
        ::OutputDebugStringW(UTF8ToUTF16W(FormatLogMessage(entry)).c_str());
    }
    void Flush() noexcept override {}
};
#endif
#ifdef __ANDROID__
/// @brief Backend that writes to the Android logcat
struct LogcatBackend : public Backend {
    explicit LogcatBackend() noexcept = default;
    ~LogcatBackend() noexcept override = default;
    void Write(const Entry& entry) noexcept override {
        android_LogPriority android_log_priority = [&]() {
            switch (entry.log_level) {
            case Level::Debug: return ANDROID_LOG_DEBUG;
            case Level::Info: return ANDROID_LOG_INFO;
            case Level::Warning: return ANDROID_LOG_WARN;
            case Level::Error: return ANDROID_LOG_ERROR;
            case Level::Critical: return ANDROID_LOG_FATAL;
            case Level::Count:
            case Level::Trace: return ANDROID_LOG_VERBOSE;
            }
        }();
        auto const df = GetDirectFormatArgs(entry);
        __android_log_print(android_log_priority, "VeshiruNative", "%s %s:%u:%s: %s", df.class_name, entry.filename, entry.line_num, entry.function, entry.message);
    }
    void Flush() noexcept override {}
};
#endif

/// @brief Static state as a singleton.
struct Impl {
    // Well, I mean it's the default constructor!
    explicit Impl() noexcept : filter(Level::Trace) {}

    template<typename F>
    void ForEachBackend(F&& lambda) noexcept {
        lambda(static_cast<Backend&>(color_console_backend));
        lambda(static_cast<Backend&>(ring_backend));
        lambda(static_cast<Backend&>(live_console_backend));
#ifndef __OPENORBIS__
        if (file_backend)
            lambda(static_cast<Backend&>(*file_backend));
#endif
#ifdef _WIN32
        lambda(static_cast<Backend&>(debugger_backend));
#endif
#ifdef __ANDROID__
        lambda(static_cast<Backend&>(lc_backend));
#endif
    }

    Filter filter;
    ColorConsoleBackend color_console_backend{};
    RingBackend ring_backend{};
    LiveConsoleBackend live_console_backend{};
#ifndef __OPENORBIS__
    std::optional<FileBackend> file_backend;
#endif
#ifdef _WIN32
    DebuggerBackend debugger_backend{};
#endif
#ifdef __ANDROID__
    LogcatBackend lc_backend{};
#endif
    std::chrono::steady_clock::time_point time_origin{std::chrono::steady_clock::now()};
};
} // Anonymous namespace

void Filter::ParseFilterString(std::string_view filter_view) {
    auto clause_begin = filter_view.cbegin();
    while (clause_begin < filter_view.cend()) {
        auto clause_end = std::find(clause_begin, filter_view.cend(), ' ');
        // If clause isn't empty
        if (clause_end != clause_begin) {
            ParseFilterRule(*this, clause_begin, clause_end);
        }
        if (clause_end != filter_view.cend()) {
            // Skip over the whitespace
            ++clause_end;
        }
        clause_begin = clause_end;
    }
}

// Constructor shall NOT depend upon Settings() or whatever
// it's ran at global static ctor() time... so BE CAREFUL MFER!
static std::optional<Common::Log::Impl> logging_instance{};

void Initialize() {
    if (logging_instance) {
        LOG_WARNING(Log, "Reinitializing logging backend");
    } else {
        logging_instance.emplace();
        logging_instance->filter.ParseFilterString(Settings::values.log_filter.GetValue());
#ifndef __OPENORBIS__
        using namespace Common::FS;
        const auto& log_dir = GetEdenPath(EdenPath::LogDir);
        void(CreateDir(log_dir));
        logging_instance->file_backend.emplace(log_dir / LOG_FILE);
#endif
    }
}

void Start() {
}

void Stop() {
    if (logging_instance) {
        logging_instance->ForEachBackend([](Backend& backend) { backend.Flush(); });
        logging_instance->live_console_backend.Disable();
    }
}

void FmtCrashMessageImpl(const char* filename, unsigned int line_num, const char* function, fmt::string_view format, const fmt::format_args& args) noexcept {
#ifndef __OPENORBIS__
    if (!logging_instance || !logging_instance->file_backend)
        return;
    char buffer[2048];
    const auto result = fmt::vformat_to_n(buffer, sizeof(buffer) - 1, format, args);
    const std::size_t length = (std::min)(result.size, sizeof(buffer) - 1);
    buffer[length] = '\0';
    const Entry e{
        .message = buffer,
        .message_len = length,
        .timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - logging_instance->time_origin),
        .log_class = Class::Common,
        .log_level = Level::Critical,
        .filename = TrimSourcePath(filename),
        .function = function,
        .line_num = line_num,
    };
    logging_instance->file_backend->Write(e);
    logging_instance->file_backend->Flush();
#endif
}

void SetGlobalFilter(const Filter& filter) {
    if (logging_instance)
        logging_instance->filter = filter;
}

void SetColorConsoleBackendEnabled(bool enabled) {
    if (logging_instance)
        logging_instance->color_console_backend.enabled = enabled;
}

void SetRingCapacity(std::size_t capacity) {
    if (!logging_instance)
        return;
    auto& ring = logging_instance->ring_backend;
    std::scoped_lock lock{ring.mutex};
    ring.capacity.store(capacity, std::memory_order_relaxed);
    while (ring.lines.size() > capacity) {
        ring.lines.pop_front();
    }
}

std::vector<RingLine> ReadRing(u64 after) {
    std::vector<RingLine> result;
    if (!logging_instance)
        return result;
    auto& ring = logging_instance->ring_backend;
    std::scoped_lock lock{ring.mutex};
    for (const RingLine& line : ring.lines) {
        if (line.sequence > after) {
            result.push_back(line);
        }
    }
    return result;
}

bool IsLevelEnabled(Class log_class, Level log_level) {
    return logging_instance && logging_instance->filter.CheckMessage(log_class, log_level);
}

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename, unsigned int line_num, const char* function, fmt::string_view format, const fmt::format_args& args) {
    if (logging_instance && logging_instance->filter.CheckMessage(log_class, log_level)) {
        auto const flush = ::Settings::values.log_flush_line.GetValue();
        char buffer[BUFSIZ];
        auto result = fmt::vformat_to_n(buffer, sizeof(buffer) - 1, format, args);
        Entry e{
            .message = nullptr,
            .message_len = 0,
            .timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - logging_instance->time_origin),
            .log_class = log_class,
            .log_level = log_level,
            .filename = TrimSourcePath(filename),
            .function = function,
            .line_num = line_num,
        };
        if (result.size <= sizeof(buffer) - 1) {
            buffer[(std::min)(result.size, sizeof(buffer) - 1)] = '\0';
            e.message = buffer;
            e.message_len = (std::min)(result.size, sizeof(buffer) - 1);
            logging_instance->ForEachBackend([=](Backend& backend) {
                backend.Write(e);
                if (flush) backend.Flush();
            });
        } else {
            std::string s = fmt::vformat(format, args);
            e.message = s.c_str();
            e.message_len = s.size();
            logging_instance->ForEachBackend([=](Backend& backend) {
                backend.Write(e);
                if (flush) backend.Flush();
            });
        }
    }
}
std::string_view GetSubsystemName(Subsystem subsystem) {
    switch (subsystem) {
    case Subsystem::CPU: return "CPU";
    case Subsystem::GPU: return "GPU";
    case Subsystem::HLE: return "HLE";
    case Subsystem::FS: return "FS";
    case Subsystem::Audio: return "Audio";
    case Subsystem::Networking: return "Networking";
    case Subsystem::Performance: return "Performance";
    case Subsystem::Compatibility: return "Compatibility";
    case Subsystem::Other: return "Other";
    default: return "?";
    }
}

Subsystem GetSubsystem(Class log_class) {
    return GetSubsystemImpl(log_class);
}

std::string_view GetLogClassName(Class log_class) {
    return GetLogClassNameImpl(log_class);
}

std::string_view GetLogLevelName(Level log_level) {
    return GetLevelNameImpl(log_level);
}

void SetLiveConsoleEnabled(bool enabled) {
    if (!logging_instance) {
        return;
    }
    if (enabled) {
        logging_instance->live_console_backend.Enable();
    } else {
        logging_instance->live_console_backend.Disable();
    }
}

bool IsLiveConsoleEnabled() {
    return logging_instance && logging_instance->live_console_backend.enabled.load(std::memory_order_relaxed);
}

void SetLiveConsoleCapacity(std::size_t capacity) {
    if (!logging_instance) {
        return;
    }
    auto& backend = logging_instance->live_console_backend;
    backend.capacity.store(capacity, std::memory_order_relaxed);
    std::scoped_lock lock{backend.mutex};
    while (backend.entries.size() > capacity) {
        backend.entries.pop_front();
        backend.dropped_buffer.fetch_add(1, std::memory_order_relaxed);
    }
}

std::vector<ConsoleEntry> ReadLiveConsole(u64 after, std::size_t max_entries) {
    if (!logging_instance) {
        return {};
    }
    return logging_instance->live_console_backend.Read(after, max_entries);
}

void ClearLiveConsole() {
    if (logging_instance) {
        logging_instance->live_console_backend.Clear();
    }
}

ConsoleStats GetLiveConsoleStats() {
    if (!logging_instance) {
        return {};
    }
    return logging_instance->live_console_backend.Stats();
}

std::string SanitizeLogMessage(std::string_view message) {
    return SanitizeLogMessageImpl(message);
}

} // namespace Common::Log
