// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <regex>
#include <thread>
#include <utility>

#include <boost/algorithm/string/replace.hpp>

#if defined(__ANDROID__)
#include <android/log.h>
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#elif !defined(__OPENORBIS__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "common/console_output.h"
#include "common/fs/fs.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "common/terminate_handler.h"
#include "common/thread.h"

namespace Common::Log {
namespace {

using Clock = std::chrono::steady_clock;
using Microseconds = std::chrono::microseconds;

struct Entry {
    Microseconds timestamp{};
    Class log_class{};
    Level log_level{};
    const char* filename{};
    const char* function{};
    unsigned int line_num{};
};

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
    const auto idx = (std::max)({rfind("src/"), rfind("src\\"), rfind("../"), rfind("..\\")});
    const bool absolute = source.starts_with('/') || source.starts_with('\\') ||
                          (source.size() > 1 && source[1] == ':');
    return source.data() + (idx || !absolute ? idx : source.find_last_of("/\\") + 1);
}

std::string FormatFileLine(const Entry& entry, std::string_view message) {
    using namespace std::chrono;
    const auto seconds = duration_cast<std::chrono::seconds>(entry.timestamp);
    const auto fraction = entry.timestamp - seconds;
    return fmt::format("[{:4d}.{:06d}] {} <{}> {}:{}:{}: {}\n", seconds.count(), fraction.count(),
                       GetLogClassNameImpl(entry.log_class), GetLevelNameImpl(entry.log_level),
                       entry.filename, entry.line_num, entry.function, message);
}

std::string FormatConsoleLine(const Entry& entry, std::string_view message) {
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(entry.timestamp);
    const std::string level = fmt::format("[{}]", GetLevelNameImpl(entry.log_level));
    return fmt::format("[{}.{:03d}s] {:<10} [{}] {}\n", millis.count() / 1000,
                       millis.count() % 1000, level, GetLogClassNameImpl(entry.log_class), message);
}

std::string WithoutTerminalEscapes(std::string_view message) {
    std::string text;
    text.reserve(message.size());
    while (!message.empty()) {
        const auto escape = message.find('\x1b');
        text += message.substr(0, escape);
        if (escape == message.npos) {
            break;
        }
        message.remove_prefix(escape + 1);
        if (message.starts_with('[')) {
            const auto end =
                message.substr(1).find_first_not_of("0123456789:;<=>? !\"#$%&'()*+,-./");
            if (end != message.npos && message[end + 1] >= '@' && message[end + 1] <= '~') {
                message.remove_prefix(end + 2);
            }
        }
    }
    return text;
}

std::string SanitizeLogMessageImpl(std::string_view message, bool censor_username) {
    std::string text = WithoutTerminalEscapes(message);
    // Match paths with spaces too. When a path is unquoted, prefer removing the rest of its line
    // over guessing where a private filename ends. Keep guest paths such as /dev/nvhost intact.
    if (text.find_first_of("/\\") != std::string::npos) {
        static const std::regex host_path{
            R"((?:\b[A-Za-z]:[\\/]|\\\\)[^\r\n"'<>]*|(^|[\s"'(=])/(?:home|Users|tmp|var|mnt|media|run|srv|opt|root|private)/[^\r\n"'<>]*)"};
        static const std::regex url_userinfo{R"((https?://)[^\s/@]+@)"};
        text = std::regex_replace(text, host_path, "$1<path>");
        if (text.find('@') != std::string::npos) {
            text = std::regex_replace(text, url_userinfo, "$1<redacted>@");
        }
    }
    if (text.find_first_of(":=") != std::string::npos) {
        static const std::regex credential{
            R"REGEX((\b(?:authorization|cookie|set-cookie)\s*:\s*)[^\r\n]*|(\b(?:(?:access|refresh|nex|eden)[_-]?)?token["']?\s*[:=]\s*|(?:\b|_)(?:api[_-]?key|password|secret)["']?\s*[:=]\s*)(?:"[^"\r\n]*"?|'[^'\r\n]*'?|[^\s,;"']+))REGEX",
            std::regex::icase};
        text = std::regex_replace(text, credential, "$1$2<redacted>");
    }
#ifndef __ANDROID__
    if (censor_username) {
        static const std::string username = [] {
            for (const auto* var : {"LOGNAME", "USERNAME", "USER"}) {
                if (const auto* value = std::getenv(var); value && *value) {
                    return std::string{value};
                }
            }
            return std::string{};
        }();
        if (username.size() >= 2) {
            boost::ireplace_all(text, username, "<user>");
        }
    }
#else
    (void)censor_username;
#endif
    return text;
}

// Keep exactly one record per line and never forward terminal control sequences. Invalid UTF-8
// from a driver, a guest string or truncation is replaced without dropping adjacent ASCII.
std::string MakePrintable(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if (lead < 0x80) {
            switch (lead) {
            case '\r':
                out += "\\r";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += lead >= 0x20 && lead < 0x7F ? text[i] : '?';
                break;
            }
            ++i;
            continue;
        }
        unsigned int count = lead < 0xC2   ? 0
                             : lead < 0xE0 ? 2
                             : lead < 0xF0 ? 3
                             : lead < 0xF5 ? 4
                                           : 0;
        if (count > text.size() - i) {
            count = 0;
        }
        const auto byte = [&](unsigned int offset) {
            return static_cast<unsigned char>(text[i + offset]);
        };
        for (unsigned int j = 1; j < count; ++j) {
            if ((byte(j) & 0xC0) != 0x80) {
                count = 0;
            }
        }
        if ((count == 3 &&
             ((lead == 0xE0 && byte(1) < 0xA0) || (lead == 0xED && byte(1) >= 0xA0))) ||
            (count == 4 &&
             ((lead == 0xF0 && byte(1) < 0x90) || (lead == 0xF4 && byte(1) >= 0x90)))) {
            count = 0;
        }
        if (count == 0) {
            out += "\xEF\xBF\xBD";
            ++i;
        } else {
            if (lead == 0xC2 && byte(1) < 0xA0) {
                out += '?'; // C1 control characters
            } else {
                out.append(text.substr(i, count));
            }
            i += count;
        }
    }
    return out;
}

#ifndef __OPENORBIS__
class FileBackend {
public:
    explicit FileBackend(std::filesystem::path path, bool rotate) : active_path(std::move(path)) {
        bool append = !rotate;
        if (rotate && !Rotate(active_path)) {
            auto fallback = active_path.parent_path() / active_path.stem();
            fallback += "_2";
            fallback += active_path.extension();
            active_path = std::move(fallback);
            append = !Rotate(active_path);
        }
        file = Open(active_path, append);
        opened = enabled = file != nullptr;
        // The session file has a size cap. The explicit mirror is user-managed and appends.
        if (enabled && append && rotate) {
            std::error_code ec;
            const auto size = std::filesystem::file_size(active_path, ec);
            bytes_written = ec ? 0 : size;
        }
        capped = rotate;
    }

    ~FileBackend() {
        Close();
    }

    void Close() {
        std::scoped_lock lock{mutex};
        if (auto* closing = std::exchange(file, nullptr)) {
            std::fclose(closing);
        }
        enabled = false;
    }

    [[nodiscard]] bool WasOpened() const {
        return opened;
    }
    [[nodiscard]] const std::filesystem::path& GetPath() const {
        return active_path;
    }

    void Write(std::string_view line, bool flush, bool extended) {
        using namespace Common::Literals;
        std::scoped_lock lock{mutex};
        if (!enabled) {
            return;
        }
        const auto written = std::fwrite(line.data(), 1, line.size(), file);
        bytes_written += written;
        const auto limit = extended ? 1_GiB : 100_MiB;
        if (written != line.size() || (capped && bytes_written >= limit)) {
            enabled = false;
        }
        if ((flush || !enabled) && std::fflush(file) != 0) {
            enabled = false;
        }
    }

    void Flush() {
        std::scoped_lock lock{mutex};
        if (enabled && std::fflush(file) != 0) {
            enabled = false;
        }
    }

    /// Do not wait for a lock the crashing thread (or an I/O worker) might hold.
    void TryWriteCrash(std::string_view line) {
        std::unique_lock lock{mutex, std::try_to_lock};
        if (lock && file) {
            std::fwrite(line.data(), 1, line.size(), file);
            std::fflush(file);
        }
    }

private:
    static std::FILE* Open(const std::filesystem::path& path, bool append) {
#ifdef _WIN32
        // Only open local drive paths (not the device or UNC namespaces, which also contain pipes).
        const auto name = path.lexically_normal().native();
        if (name.size() < 3 || name[1] != L':' || name[2] != L'\\' ||
            name.find(L':', 2) != name.npos) {
            return nullptr;
        }
        const auto access = append ? FILE_APPEND_DATA : GENERIC_WRITE;
        HANDLE handle = CreateFileW(name.c_str(), access, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            return nullptr;
        }
        if (GetFileType(handle) != FILE_TYPE_DISK || (!append && !SetEndOfFile(handle))) {
            CloseHandle(handle);
            return nullptr;
        }
        const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_BINARY | _O_WRONLY);
        if (fd == -1) {
            CloseHandle(handle);
            return nullptr;
        }
        if (auto* stream = _fdopen(fd, "wb")) {
            return stream;
        }
        _close(fd);
#else
        const int fd =
            open(path.c_str(),
                 O_WRONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK | (append ? O_APPEND : 0), 0600);
        if (fd < 0) {
            return nullptr;
        }
        struct stat status{};
        if (fstat(fd, &status) == 0 && S_ISREG(status.st_mode) &&
            (append || ftruncate(fd, 0) == 0)) {
            if (auto* stream = fdopen(fd, append ? "ab" : "wb")) {
                return stream;
            }
        }
        close(fd);
#endif
        return nullptr;
    }

    static bool Rotate(const std::filesystem::path& path) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            return !ec;
        }
        // Never rename a directory passed by mistake.
        if (!std::filesystem::is_regular_file(path, ec)) {
            return false;
        }
        auto old_path = path;
        old_path += ".old.txt";
        std::filesystem::rename(path, old_path, ec);
        return !ec;
    }

    std::filesystem::path active_path;
    std::mutex mutex;
    std::FILE* file{};
    std::uintmax_t bytes_written{};
    bool capped{};
    bool opened{};
    bool enabled{};
};
#endif

struct RingBackend {
    void Write(std::string_view line) {
        std::scoped_lock lock{mutex};
        if (const auto limit = capacity.load(std::memory_order_relaxed); limit != 0) {
            lines.push_back({next_sequence++, std::string{line}});
            while (lines.size() > limit) {
                lines.pop_front();
            }
        }
    }

    std::atomic<std::size_t> capacity{};
    std::mutex mutex;
    std::deque<RingLine> lines;
    u64 next_sequence = 1;
};

struct Pending {
    Entry entry;
    std::string message;
};

struct Impl {
    explicit Impl(const Options& options)
        : fixed_level{options.fixed_level}, asynchronous{options.asynchronous},
          console_enabled{options.console} {
        for (auto& level : filter) {
            level.store(fixed_level.value_or(Level::Info), std::memory_order_relaxed);
        }
    }

    ~Impl() {
        Shutdown();
    }

    [[nodiscard]] Microseconds Elapsed() const {
        return std::chrono::duration_cast<Microseconds>(Clock::now() - time_origin);
    }

    void StartWorker() {
        if (console_enabled.load(std::memory_order_relaxed)) {
            stream_sink.emplace();
        }
        worker = std::thread{[this] { Run(); }};
    }

    void Submit(Entry entry, std::string message) {
        if (!asynchronous) {
            entry.timestamp = Elapsed();
            Emit(entry, message);
            return;
        }

        // Leave room for warnings and errors if verbose messages saturate the queue.
        constexpr std::size_t capacity = 4096;
        constexpr std::size_t reserve = 128;
        {
            std::scoped_lock lock{queue_mutex};
            if (closing) {
                return;
            }
            const bool important = entry.log_level >= Level::Warning;
            if (pending.size() >= (important ? capacity : capacity - reserve)) {
                ++dropped[important ? 1 : 0];
                return;
            }
            entry.timestamp = Elapsed();
            pending.push_back({entry, std::move(message)});
        }
        queue_changed.notify_one();
    }

    void Run() {
        Common::SetCurrentThreadName("LogWriter");
        std::deque<Pending> batch;
        auto next_flush = Clock::now() + std::chrono::seconds{1};
        for (;;) {
            std::unique_lock lock{queue_mutex};
            queue_changed.wait_until(lock, next_flush, [this] {
                return closing || !pending.empty() || flush_requested != flush_done;
            });
            batch.swap(pending);
            const auto missed = std::exchange(dropped, std::array<u64, 2>{});
            const auto drop_time = Elapsed();
            const auto ticket = flush_requested;
            const bool flush = ticket != flush_done;
            const bool stop = closing;
            lock.unlock();

            for (const auto& item : batch) {
                Emit(item.entry, item.message);
            }
            batch.clear();
            if (missed[0] || missed[1]) {
                try {
                    Emit(
                        {drop_time, Class::Log, Level::Warning, "common/logging.cpp", "Run", 0},
                        fmt::format("Log queue full: dropped {} below Warning, {} Warning or above",
                                    missed[0], missed[1]));
                } catch (const std::bad_alloc&) {
                }
            }
            if (flush || stop || Clock::now() >= next_flush) {
                FlushFiles();
                next_flush = Clock::now() + std::chrono::seconds{1};
            }
            if (flush) {
                lock.lock();
                flush_done = ticket;
                lock.unlock();
                flushed.notify_all();
            }
            if (stop) {
                return;
            }
        }
    }

    void Flush() {
        using namespace std::chrono_literals;
        if (!asynchronous) {
            FlushFiles();
            return;
        }
        std::unique_lock lock{queue_mutex};
        if (closing || !worker.joinable() || std::this_thread::get_id() == worker.get_id()) {
            return;
        }
        const auto ticket = ++flush_requested;
        queue_changed.notify_one();
        flushed.wait_for(lock, 1s, [this, ticket] { return flush_done >= ticket; });
    }

    void WriteConsole(const Entry& entry, std::string_view message) {
        if (!console_enabled.load(std::memory_order_relaxed)) {
            return;
        }
        auto line = FormatConsoleLine(entry, message);
        const bool error = entry.log_level >= Level::Error;
        const bool important = entry.log_level >= Level::Warning;
        const auto color = error ? ConsoleColor::Error
                                 : (important ? ConsoleColor::Warning : ConsoleColor::Normal);
        if (stream_sink) {
            stream_sink->Enqueue(std::move(line), error, important, color);
        } else if (!asynchronous) {
            WriteStandardStream(line, error, color);
        }
    }

    void Shutdown() {
        accepting.store(false, std::memory_order_relaxed);
        if (asynchronous) {
            {
                std::scoped_lock lock{queue_mutex};
                if (closing) {
                    return;
                }
                closing = true;
            }
            queue_changed.notify_one();
            if (worker.joinable()) {
                worker.join();
            }
            if (stream_sink) {
                stream_sink->Close();
            }
        } else {
            FlushFiles();
        }
#ifndef __OPENORBIS__
        if (file_backend) {
            file_backend->Close();
        }
        if (extra_backend) {
            extra_backend->Close();
        }
#endif
    }

    void FlushFiles() {
#ifndef __OPENORBIS__
        if (file_backend) {
            file_backend->Flush();
        }
        if (extra_backend) {
            extra_backend->Flush();
        }
#endif
    }

    void Emit(const Entry& entry, std::string_view message) noexcept {
        try {
            const auto printable = MakePrintable(
                SanitizeLogMessageImpl(message, censor_username.load(std::memory_order_relaxed)));
            const auto line = FormatFileLine(entry, printable);
#ifndef __OPENORBIS__
            const bool flush =
                entry.log_level >= Level::Warning || flush_line.load(std::memory_order_relaxed);
            const bool extend = extended.load(std::memory_order_relaxed);
            if (file_backend) {
                file_backend->Write(line, flush, extend);
            }
            if (extra_backend) {
                extra_backend->Write(line, flush, extend);
            }
#endif
            if (ring.capacity.load(std::memory_order_relaxed) != 0) {
                ring.Write(std::string_view{line}.substr(0, line.size() - 1));
            }
#ifdef _WIN32
            if (IsDebuggerPresent()) {
                ::OutputDebugStringW(UTF8ToUTF16W(line).c_str());
            }
#endif
#ifdef __ANDROID__
            constexpr std::array priorities{ANDROID_LOG_VERBOSE, ANDROID_LOG_DEBUG,
                                            ANDROID_LOG_INFO,    ANDROID_LOG_WARN,
                                            ANDROID_LOG_ERROR,   ANDROID_LOG_FATAL};
            __android_log_write(priorities[std::size_t(entry.log_level)], "VeshiruNative",
                                line.c_str());
#endif
            WriteConsole(entry, printable);
        } catch (const std::exception&) {
            // In particular, a failed allocation in a diagnostic sink is not fatal.
        }
    }

    const std::optional<Level> fixed_level;
    const bool asynchronous;
    const Clock::time_point time_origin{Clock::now()};
    std::atomic_bool accepting{true};
    std::atomic_bool console_enabled;
    std::atomic_bool flush_line{};
    std::atomic_bool extended{};
    std::atomic_bool censor_username{true};
    std::array<std::atomic<Level>, std::size_t(Class::Count)> filter;
    RingBackend ring;
#ifndef __OPENORBIS__
    std::optional<FileBackend> file_backend;
    std::optional<FileBackend> extra_backend;
#endif
    std::optional<StandardStreamSink> stream_sink;
    std::mutex queue_mutex;
    std::condition_variable queue_changed;
    std::condition_variable flushed;
    std::deque<Pending> pending;
    std::array<u64, 2> dropped{};
    u64 flush_requested{};
    u64 flush_done{};
    bool closing{};
    std::thread worker;
};

std::optional<Impl> logging_instance;

} // namespace

void Filter::ParseFilterString(std::string_view filter_view) {
    auto clause_begin = filter_view.cbegin();
    while (clause_begin < filter_view.cend()) {
        auto clause_end = std::find(clause_begin, filter_view.cend(), ' ');
        if (clause_end != clause_begin) {
            ParseFilterRule(*this, clause_begin, clause_end);
        }
        if (clause_end != filter_view.cend()) {
            ++clause_end;
        }
        clause_begin = clause_end;
    }
}

bool Initialize(const Options& options) {
    if (logging_instance) {
        LOG_WARNING(Log, "Reinitializing logging backend");
        return options.extra_file.empty();
    }

    auto& impl = logging_instance.emplace(options);
    UpdateFileLoggingOptions();
    bool extra_opened = options.extra_file.empty();
#ifndef __OPENORBIS__
    using namespace Common::FS;
    const auto& log_dir = GetEdenPath(EdenPath::LogDir);
    void(CreateDirs(log_dir));
    impl.file_backend.emplace(log_dir / LOG_FILE, true);
    if (!impl.file_backend->WasOpened()) {
        LOG_WARNING(Log,
                    "Could not open the session log; check that the log directory is writable");
    }
    if (!options.extra_file.empty()) {
        std::error_code ec;
        // Don't write twice if the requested mirror points to the active session log.
        if (impl.file_backend->WasOpened() &&
            std::filesystem::equivalent(options.extra_file, impl.file_backend->GetPath(), ec)) {
            extra_opened = true;
        } else {
            impl.extra_backend.emplace(options.extra_file, false);
            extra_opened = impl.extra_backend->WasOpened();
        }
    }
#endif
    InstallTerminateHandler();
    if (impl.asynchronous) {
        impl.StartWorker();
    }
    if (!options.fixed_level) {
        Filter filter;
        filter.ParseFilterString(Settings::values.log_filter.GetValue());
        SetGlobalFilter(filter);
    }
    return extra_opened;
}

void Start() {}

void Stop() {
    Flush();
}

void Flush() {
    if (logging_instance) {
        logging_instance->Flush();
    }
}

void FlushConsole() {
    if (logging_instance && logging_instance->stream_sink) {
        logging_instance->stream_sink->Flush(std::chrono::milliseconds{250});
    }
}

void Shutdown() {
    if (logging_instance) {
        logging_instance->Shutdown();
    }
}

void UpdateFileLoggingOptions() {
    if (logging_instance) {
        logging_instance->flush_line.store(Settings::values.log_flush_line.GetValue(),
                                           std::memory_order_relaxed);
        logging_instance->extended.store(Settings::values.extended_logging.GetValue(),
                                         std::memory_order_relaxed);
        logging_instance->censor_username.store(Settings::values.censor_username.GetValue(),
                                                std::memory_order_relaxed);
    }
}

std::filesystem::path GetActiveLogFilePath() {
#ifndef __OPENORBIS__
    if (logging_instance && logging_instance->accepting.load(std::memory_order_relaxed) &&
        logging_instance->file_backend && logging_instance->file_backend->WasOpened()) {
        return logging_instance->file_backend->GetPath();
    }
#endif
    return {};
}

void FmtCrashMessageImpl(const char* filename, unsigned int line_num, const char* function,
                         fmt::string_view format, const fmt::format_args& args) noexcept {
    if (!logging_instance) {
        return;
    }
    try {
        const Entry entry{logging_instance->Elapsed(), Class::Common, Level::Critical,
                          TrimSourcePath(filename),    function,      line_num};
        const auto message = MakePrintable(SanitizeLogMessageImpl(
            fmt::vformat(format, args),
            logging_instance->censor_username.load(std::memory_order_relaxed)));
#ifndef __OPENORBIS__
        const auto line = FormatFileLine(entry, message);
        if (logging_instance->file_backend) {
            logging_instance->file_backend->TryWriteCrash(line);
        }
        if (logging_instance->extra_backend) {
            logging_instance->extra_backend->TryWriteCrash(line);
        }
#endif
        // Crash handlers must not take the live console queue's lock. Qt reports fatal
        // platform errors via its separate pre-GUI diagnostic path.

    } catch (const std::exception&) {
    }
}

void SetGlobalFilter(const Filter& filter) {
    if (!logging_instance || logging_instance->fixed_level) {
        return;
    }
    for (std::size_t index = 0; index < filter.class_levels.size(); ++index) {
        logging_instance->filter[index].store(filter.class_levels[index],
                                              std::memory_order_relaxed);
    }
}

void SetColorConsoleBackendEnabled(bool enabled) {
    if (logging_instance) {
        logging_instance->console_enabled.store(enabled, std::memory_order_relaxed);
    }
}

void SetRingCapacity(std::size_t capacity) {
    if (!logging_instance) {
        return;
    }
    auto& ring = logging_instance->ring;
    std::scoped_lock lock{ring.mutex};
    ring.capacity.store(capacity, std::memory_order_relaxed);
    while (ring.lines.size() > capacity) {
        ring.lines.pop_front();
    }
}

std::vector<RingLine> ReadRing(u64 after) {
    std::vector<RingLine> result;
    if (!logging_instance) {
        return result;
    }
    auto& ring = logging_instance->ring;
    std::scoped_lock lock{ring.mutex};
    for (const RingLine& line : ring.lines) {
        if (line.sequence > after) {
            result.push_back(line);
        }
    }
    return result;
}

bool IsLevelEnabled(Class log_class, Level log_level) {
    return logging_instance && logging_instance->accepting.load(std::memory_order_relaxed) &&
           log_level >=
               logging_instance->filter[std::size_t(log_class)].load(std::memory_order_relaxed);
}

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename,
                       unsigned int line_num, const char* function, fmt::string_view format,
                       const fmt::format_args& args) {
    if (IsLevelEnabled(log_class, log_level)) {
        constexpr std::string_view suffix = " ... [truncated]";
        char buffer[16 * 1024];
        const auto limit = sizeof(buffer) - suffix.size();
        const auto result = fmt::vformat_to_n(buffer, limit, format, args);
        std::string message(buffer, std::min(result.size, limit));
        if (result.size > limit) {
            message += suffix;
        }
        logging_instance->Submit(
            {{}, log_class, log_level, TrimSourcePath(filename), function, line_num},
            std::move(message));
    }
}

std::optional<Level> ParseLevel(std::string_view name) {
    const auto lower = [](unsigned char ch) { return std::tolower(ch); };
    for (u32 i = 0; i < u32(Level::Count); ++i) {
        const auto level = static_cast<Level>(i);
        if (std::ranges::equal(name, std::string_view{GetLevelNameImpl(level)}, {}, lower, lower)) {
            return level;
        }
    }
    return std::nullopt;
}

std::string SanitizeLogMessage(std::string_view message) {
    return MakePrintable(SanitizeLogMessageImpl(message, true));
}

} // namespace Common::Log
