// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/shader_log.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <fmt/format.h>

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

namespace VideoCommon::ShaderLog {

namespace detail {
std::atomic<bool> g_enabled{false};
} // namespace detail

namespace {

constexpr u64 DefaultCapMegabytes = 1024;
/// shaders.jsonl is bounded on its own (lines are a few hundred bytes to a few KB).
constexpr u64 MaxJsonlBytes = 256ULL << 20;
constexpr u32 SessionJsonEveryPipelines = 64;
constexpr auto DeferWait = std::chrono::milliseconds(30);
constexpr auto SessionRefresh = std::chrono::seconds(2);

const char* PathName(Path path) {
    return path == Path::Hle ? "hle" : "emulated";
}

const char* OutcomeName(Outcome outcome) {
    switch (outcome) {
    case Outcome::Translated:
        return "translated";
    case Outcome::DiskCache:
        return "disk_cache";
    case Outcome::Rejected:
        return "rejected";
    case Outcome::Fallback:
        return "fallback";
    }
    return "translated";
}

struct ProgramJob {
    u64 hash{};
    std::string stage;
    u64 guest_address{};
    std::vector<u8> bytes;
    std::vector<u8> control;
    bool maxwell{};
    bool rejected{};
    std::string reason;
    u32 failed_instruction{~0U};
};

struct PipelineJob {
    Path path{};
    u64 key{};
    std::vector<std::pair<std::string, u64>> stages;
    Outcome outcome{};
    std::string reason;
    double compile_ms{};
    std::string state_json;
    std::vector<File> files;
    double time_s{};
    u64 frame{};
    u32 async_skips{};
    bool compute{};
};

using Job = std::variant<ProgramJob, PipelineJob>;

struct ProgramInfo {
    u64 size{};
    u64 guest_address{};
    bool has_bin{};
    bool has_ctl{};
    bool maxwell{};
    bool rejected{};
    std::string reason;
    u32 failed_instruction{~0U};
};

class Session {
public:
    Session() = default;

    void Begin() {
        started = std::chrono::steady_clock::now();
        started_wall = std::time(nullptr);
        cap_bytes = DefaultCapMegabytes << 20;
        if (const char* text = std::getenv("EDEN_NX2_SHADER_LOG_MAX_MB")) {
            const u64 megabytes = std::strtoull(text, nullptr, 10);
            if (megabytes != 0) {
                cap_bytes = megabytes << 20;
            }
        }
        writer = std::thread{[this] { Run(); }};
    }

    void SetTitle(u64 title, std::string_view build) {
        std::scoped_lock lock{title_mutex};
        title_id = title;
        build_id = std::string{build};
        title_known.store(true, std::memory_order_relaxed);
    }

    bool TitleKnown() const noexcept {
        return title_known.load(std::memory_order_relaxed);
    }

    bool ClaimProgram(u64 hash) {
        std::scoped_lock lock{claim_mutex};
        return programs_claimed.insert(hash).second;
    }

    bool ClaimPipeline(u64 key) {
        std::scoped_lock lock{claim_mutex};
        return pipelines_claimed.insert(key).second;
    }

    void NoteAsyncSkip(u64 key) {
        std::scoped_lock lock{claim_mutex};
        ++async_skips[key];
    }

    double Seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    u32 TakeAsyncSkips(u64 key) {
        std::scoped_lock lock{claim_mutex};
        const auto it = async_skips.find(key);
        return it == async_skips.end() ? 0 : it->second;
    }

    void Push(Job&& job) {
        {
            std::scoped_lock lock{queue_mutex};
            if (stopping) {
                return;
            }
            queue.push_back(std::move(job));
        }
        queue_cv.notify_one();
    }

    void Stop() {
        {
            std::scoped_lock lock{queue_mutex};
            if (stopping) {
                return;
            }
            stopping = true;
        }
        queue_cv.notify_all();
        if (writer.joinable()) {
            writer.join();
        }
        // The session object outlives the game (see g_session): free what only it needed.
        {
            std::scoped_lock lock{claim_mutex};
            programs_claimed = {};
            pipelines_claimed = {};
            async_skips = {};
        }
        written_files = {};
        program_info = {};
    }

private:
    // -- writer thread ---------------------------------------------------------------------

    void Run() {
        std::deque<PipelineJob> deferred;
        for (;;) {
            Job job;
            bool final_defer = false;
            bool idle_timeout = false;
            {
                std::unique_lock lock{queue_mutex};
                if (queue.empty() && !deferred.empty() && !stopping) {
                    // Pipelines whose program record has not arrived yet: give it a moment.
                    queue_cv.wait_for(lock, DeferWait,
                                      [this] { return !queue.empty() || stopping; });
                    final_defer = queue.empty();
                } else if (queue.empty() && deferred.empty() && !stopping) {
                    FlushStreams();
                    // Wake now and then to refresh session.json: the run may be killed, not stopped.
                    idle_timeout = !queue_cv.wait_for(lock, SessionRefresh,
                                                      [this] { return !queue.empty() || stopping; });
                }
                if (queue.empty()) {
                    // Read under queue_mutex; a Stop during the drain is seen next iteration.
                    const bool stop_now = stopping;
                    if (stop_now || final_defer) {
                        // Nothing more is coming (or the wait ran out): write what is deferred.
                        lock.unlock();
                        while (!deferred.empty()) {
                            WritePipeline(deferred.front());
                            deferred.pop_front();
                        }
                        if (stop_now) {
                            break;
                        }
                        continue;
                    }
                    if (idle_timeout) {
                        lock.unlock();
                        WriteSessionJson(false);
                    }
                    continue;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            if (auto* program = std::get_if<ProgramJob>(&job)) {
                WriteProgram(*program);
                // A deferred pipeline may have been waiting for exactly this one.
                for (size_t i = 0; i < deferred.size();) {
                    if (!MissingProgram(deferred[i])) {
                        WritePipeline(deferred[i]);
                        deferred.erase(deferred.begin() + static_cast<std::ptrdiff_t>(i));
                    } else {
                        ++i;
                    }
                }
            } else {
                PipelineJob& pipeline = std::get<PipelineJob>(job);
                if (MissingProgram(pipeline)) {
                    deferred.push_back(std::move(pipeline));
                } else {
                    WritePipeline(pipeline);
                }
            }
        }
        WriteSessionJson(true);
        // The session is over: release the file (the session object outlives the game, and an open
        // handle keeps the session folder from being moved or removed until the next session).
        // jsonl is opened by EnsureDirectory (possibly on another thread) before dir_ready is set.
        if (dir_ready.load(std::memory_order_acquire)) {
            jsonl.close();
        }
    }

    bool MissingProgram(const PipelineJob& pipeline) const {
        for (const auto& [name, hash] : pipeline.stages) {
            if (!program_info.contains(hash)) {
                return true;
            }
        }
        return false;
    }

    void FlushStreams() {
        // Until dir_ready, EnsureDirectory may be opening jsonl on another thread.
        if (dir_ready.load(std::memory_order_acquire) && jsonl.is_open()) {
            jsonl.flush();
        }
    }

    bool EnsureDirectory() {
        if (dir_ready.load(std::memory_order_acquire)) {
            return !directory_failed;
        }
        std::scoped_lock dir_lock{dir_mutex};
        if (dir_ready.load(std::memory_order_relaxed)) {
            return !directory_failed;
        }
        u64 title;
        std::string build;
        {
            std::scoped_lock lock{title_mutex};
            title = title_id;
            build = build_id;
        }
        std::filesystem::path root;
        if (const char* override_dir = std::getenv("EDEN_NX2_SHADER_LOG_DIR");
            override_dir != nullptr && *override_dir != '\0') {
            root = override_dir;
        } else {
            root = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "shaders";
        }
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &started_wall);
#else
        localtime_r(&started_wall, &tm);
#endif
        char stamp[32]{};
        std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
        session_title = title;
        session_build = build.empty() ? std::string{"unknown"} : build;
        const std::string base_name = fmt::format("{:016x}-{}-{}", title, session_build, stamp);
        std::error_code error;
        std::filesystem::create_directories(root, error);
        // A game restarted within the same second gets "-2", "-3"... instead of sharing a session.
        for (u32 attempt = 1; !error; ++attempt) {
            directory = root / (attempt == 1 ? base_name : fmt::format("{}-{}", base_name, attempt));
            if (std::filesystem::create_directory(directory, error) || attempt == 100) {
                break;
            }
        }
        if (error || !std::filesystem::is_directory(directory)) {
            directory_failed = true;
            LOG_ERROR(Render_Vulkan, "[shader-log] cannot create {}: {}", directory.string(),
                      error.message());
            dir_ready.store(true, std::memory_order_release);
            return false;
        }
        jsonl.open(directory / "shaders.jsonl", std::ios::binary | std::ios::trunc);
        if (!jsonl) {
            LOG_ERROR(Render_Vulkan, "[shader-log] cannot open {}: no shaders.jsonl lines",
                      (directory / "shaders.jsonl").string());
        }
        LOG_INFO(Render_Vulkan, "[shader-log] writing {}", directory.string());
        // session.json is written by the writer thread only (its counters are writer-only).
        dir_ready.store(true, std::memory_order_release);
        return true;
    }

    bool WriteFile(const std::string& name, const void* data, size_t size) {
        if (written_files.contains(name)) {
            return true;
        }
        if (cap_reached || files_bytes + size > cap_bytes) {
            if (!cap_reached) {
                cap_reached = true;
                LOG_WARNING(Render_Vulkan,
                            "[shader-log] size cap of {} MiB reached: no more files are written "
                            "(shaders.jsonl lines continue, marked files_skipped)",
                            cap_bytes >> 20);
            }
            return false;
        }
        std::ofstream out(directory / name, std::ios::binary | std::ios::trunc);
        if (out) {
            out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
            out.close();
        }
        if (!out) {
            if (!write_failed) {
                write_failed = true;
                LOG_ERROR(Render_Vulkan, "[shader-log] cannot write {} (further failures are "
                          "only marked files_skipped)", (directory / name).string());
            }
            return false;
        }
        written_files.insert(name);
        files_bytes += size;
        return true;
    }

    void WriteProgram(const ProgramJob& job) {
        if (!EnsureDirectory()) {
            return;
        }
        ProgramInfo info;
        info.size = job.bytes.size();
        info.guest_address = job.guest_address;
        info.maxwell = job.maxwell;
        info.rejected = job.rejected;
        info.reason = job.reason;
        info.failed_instruction = job.failed_instruction;
        if (!job.bytes.empty()) {
            info.has_bin =
                WriteFile(fmt::format("{:016x}.bin", job.hash), job.bytes.data(), job.bytes.size());
        }
        if (!job.control.empty()) {
            info.has_ctl = WriteFile(fmt::format("{:016x}.ctl", job.hash), job.control.data(),
                                     job.control.size());
        }
        program_info[job.hash] = std::move(info);
        ++program_count;
    }

    void WritePipeline(PipelineJob& job) {
        if (!EnsureDirectory()) {
            return;
        }
        std::string files_json;
        bool skipped = false;
        for (const File& file : job.files) {
            if (written_files.contains(file.name)) {
                continue;
            }
            if (WriteFile(file.name, file.data.data(), file.data.size())) {
                files_json += fmt::format("{}{}", files_json.empty() ? "" : ",", Quote(file.name));
            } else {
                skipped = true;
            }
        }
        std::string stages_json;
        for (const auto& [name, hash] : job.stages) {
            ProgramInfo info;
            if (const auto it = program_info.find(hash); it != program_info.end()) {
                info = it->second;
            } else {
                info.has_bin = false;
            }
            if (!info.has_bin && program_info.contains(hash) && info.size != 0) {
                skipped = true;
            }
            stages_json += fmt::format(
                "{}{{\"stage\":{},\"hash\":\"{:016x}\",\"size\":{},\"guest_address\":{},"
                "\"bin\":{},\"ctl\":{},\"isa\":\"{}\",\"program_rejected\":{},\"program_reason\":{},"
                "\"failed_instruction\":{}}}",
                stages_json.empty() ? "" : ",", Quote(name), hash,
                info.has_bin ? std::to_string(info.size) : std::string{"null"},
                info.guest_address != 0 ? fmt::format("\"{:#x}\"", info.guest_address)
                                        : std::string{"null"},
                info.has_bin ? Quote(fmt::format("{:016x}.bin", hash)) : std::string{"null"},
                info.has_ctl ? Quote(fmt::format("{:016x}.ctl", hash)) : std::string{"null"},
                info.maxwell ? "maxwell" : "ampere", info.rejected ? "true" : "false", Quote(info.reason),
                info.failed_instruction == ~0U ? std::string{"null"}
                                               : std::to_string(info.failed_instruction));
        }
        std::string title_text;
        std::string build_text;
        title_text = fmt::format("{:016x}", session_title);
        build_text = session_build;
        const std::string line = fmt::format(
            "{{\"seq\":{},\"path\":\"{}\",\"kind\":\"{}\",\"title_id\":\"{}\",\"build_id\":{},"
            "\"pipeline_key\":\"{:016x}\",\"stages\":[{}],\"outcome\":\"{}\",\"reason\":{},"
            "\"compile_ms\":{:.3f},\"time_s\":{:.3f},\"frame\":{},\"async_skips\":{},"
            "\"state\":{},\"files\":[{}],\"files_skipped\":{}}}\n",
            ++pipeline_count, PathName(job.path), job.compute ? "compute" : "graphics", title_text,
            Quote(build_text), job.key, stages_json, OutcomeName(job.outcome), Quote(job.reason),
            job.compile_ms, job.time_s, job.frame, job.async_skips,
            job.state_json.empty() ? std::string{"{}"} : job.state_json, files_json,
            skipped ? "true" : "false");
        if (jsonl_bytes + line.size() <= MaxJsonlBytes) {
            jsonl.write(line.data(), static_cast<std::streamsize>(line.size()));
            jsonl_bytes += line.size();
        } else if (!jsonl_full) {
            jsonl_full = true;
            LOG_WARNING(Render_Vulkan, "[shader-log] shaders.jsonl reached {} MiB: no more lines",
                        MaxJsonlBytes >> 20);
        }
        if (pipeline_count % SessionJsonEveryPipelines == 0) {
            WriteSessionJson(false);
        }
    }

    void WriteSessionJson(bool final) {
        if (!dir_ready.load(std::memory_order_acquire) || directory_failed) {
            return;
        }
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &started_wall);
#else
        localtime_r(&started_wall, &tm);
#endif
        char stamp[40]{};
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", &tm);
        const std::string text = fmt::format(
            "{{\"title_id\":\"{:016x}\",\"build_id\":{},\"started\":\"{}\",\"finished\":{},"
            "\"duration_s\":{:.3f},\"frames\":{},\"pipelines\":{},\"programs\":{},"
            "\"files_bytes\":{},\"files_cap_bytes\":{},\"cap_reached\":{}}}\n",
            session_title, Quote(session_build), stamp, final ? "true" : "false", Seconds(),
            Frame(), pipeline_count, program_count, files_bytes, cap_bytes,
            cap_reached ? "true" : "false");
        std::ofstream out(directory / "session.json", std::ios::binary | std::ios::trunc);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out && !write_failed) {
            write_failed = true;
            LOG_ERROR(Render_Vulkan, "[shader-log] cannot write {}",
                      (directory / "session.json").string());
        }
        if (final) {
            LOG_INFO(Render_Vulkan,
                     "[shader-log] session done: {} pipelines, {} programs, {} KiB written in {}",
                     pipeline_count, program_count, files_bytes >> 10, directory.string());
        }
    }

public:
    /// Creates the session directory on first use (any thread); empty when it cannot be made.
    std::filesystem::path Directory() {
        return EnsureDirectory() ? directory : std::filesystem::path{};
    }

    std::chrono::steady_clock::time_point started;

private:
    std::time_t started_wall{};
    u64 cap_bytes{};

    std::mutex title_mutex;
    u64 title_id{};
    std::string build_id;
    std::atomic<bool> title_known{false};

    std::mutex claim_mutex;
    std::unordered_set<u64> programs_claimed;
    std::unordered_set<u64> pipelines_claimed;
    std::unordered_map<u64, u32> async_skips;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<Job> queue;
    bool stopping{};
    std::thread writer;

    // Set once under dir_mutex (EnsureDirectory); read-only afterwards.
    std::mutex dir_mutex;
    std::atomic<bool> dir_ready{false};
    std::filesystem::path directory;
    bool directory_failed{};
    // Writer thread only (session_title / session_build: set once in EnsureDirectory, published
    // by dir_ready).
    u64 session_title{};
    std::string session_build;
    std::ofstream jsonl;
    u64 jsonl_bytes{};
    bool jsonl_full{};
    bool write_failed{};
    std::unordered_set<std::string> written_files;
    std::unordered_map<u64, ProgramInfo> program_info;
    u64 files_bytes{};
    bool cap_reached{};
    u64 pipeline_count{};
    u64 program_count{};
};

std::atomic<u64> g_frame{0};
// The running session; null between games. Ended sessions are leaked on purpose (their large
// containers are freed by Stop): a hotkey or diag request on another thread may still hold one.
std::atomic<Session*> g_session{nullptr};
// Serialises Start/Stop (game start and shutdown).
std::mutex g_lifecycle_mutex;

Session* Current() noexcept {
    return g_session.load(std::memory_order_acquire);
}

bool EnvEnabled(bool& set) {
    const char* text = std::getenv("EDEN_NX2_SHADER_LOG");
    set = text != nullptr && *text != '\0';
    return set && std::atoi(text) != 0;
}

} // namespace

void Start() {
    std::scoped_lock lock{g_lifecycle_mutex};
    if (Current() != nullptr) {
        return;
    }
    bool env_set = false;
    const bool env_on = EnvEnabled(env_set);
    // EDEN_NX2_SHADER_LOG only: VTX-HLE starts the session (Nvn::Context).
    if (!env_set || !env_on) {
        return;
    }
    g_frame.store(0, std::memory_order_relaxed);
    auto* session = new Session;
    session->Begin();
    g_session.store(session, std::memory_order_release);
    detail::g_enabled.store(true, std::memory_order_release);
}

void SetTitle(u64 title_id, std::string_view build_id) {
    if (Session* session = Current()) {
        session->SetTitle(title_id, build_id);
    }
}

bool TitleKnown() noexcept {
    const Session* session = Current();
    return session != nullptr && session->TitleKnown();
}

void Stop() {
    std::scoped_lock lock{g_lifecycle_mutex};
    Session* const session = g_session.exchange(nullptr, std::memory_order_acq_rel);
    if (session == nullptr) {
        return;
    }
    detail::g_enabled.store(false, std::memory_order_release);
    session->Stop();
}

std::filesystem::path SessionDirectory() {
    Session* session = Current();
    return session != nullptr ? session->Directory() : std::filesystem::path{};
}

void NoteFrame() noexcept {
    g_frame.fetch_add(1, std::memory_order_relaxed);
}

u64 Frame() noexcept {
    return g_frame.load(std::memory_order_relaxed);
}

bool ClaimProgram(u64 hash) {
    Session* session = Current();
    return session != nullptr && session->ClaimProgram(hash);
}

void SubmitProgram(const Program& program) {
    Session* session = Current();
    if (session == nullptr) {
        return;
    }
    ProgramJob job;
    job.hash = program.hash;
    job.stage = std::string{program.stage};
    job.guest_address = program.guest_address;
    job.bytes.assign(program.bytes.begin(), program.bytes.end());
    job.control.assign(program.control.begin(), program.control.end());
    job.maxwell = program.maxwell;
    job.rejected = program.rejected;
    job.reason = std::string{program.reason};
    job.failed_instruction = program.failed_instruction;
    session->Push(std::move(job));
}

bool ClaimPipeline(u64 pipeline_key) {
    Session* session = Current();
    return session != nullptr && session->ClaimPipeline(pipeline_key);
}

void NoteAsyncSkip(u64 pipeline_key) {
    if (Session* session = Current()) {
        session->NoteAsyncSkip(pipeline_key);
    }
}

void SubmitPipeline(Pipeline&& pipeline) {
    Session* session = Current();
    if (session == nullptr) {
        return;
    }
    PipelineJob job;
    job.path = pipeline.path;
    job.key = pipeline.key;
    job.stages.reserve(pipeline.stages.size());
    for (const Stage& stage : pipeline.stages) {
        job.stages.emplace_back(std::string{stage.stage}, stage.hash);
    }
    job.compute = pipeline.stages.size() == 1 && pipeline.stages[0].stage == "cs";
    job.outcome = pipeline.outcome;
    job.reason = std::move(pipeline.reason);
    job.compile_ms = pipeline.compile_ms;
    job.state_json = std::move(pipeline.state_json);
    job.files = std::move(pipeline.files);
    job.time_s = session->Seconds();
    job.frame = Frame();
    job.async_skips = session->TakeAsyncSkips(pipeline.key);
    session->Push(std::move(job));
}

std::string Quote(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                out += fmt::format("\\u{:04x}", c);
            } else {
                out.push_back(raw);
            }
        }
    }
    out.push_back('"');
    return out;
}

} // namespace VideoCommon::ShaderLog
