// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

/// Shader logging mode ("Log shaders for analysis", `EDEN_NX2_SHADER_LOG=1`; off by default).
///
/// Every program the game really uses in a session lands in a directory agents can read after the
/// run: %APPDATA%\fukko\log\shaders\<titleid>-<build-id>-<timestamp>\ (see docs/NX2_DIAGNOSTICS.md
/// and tools/shader_log_report.py). The log is the source of truth, not our interpretation of it:
///
///   shaders.jsonl     one line per first use of each pipeline key (all stages in `stages`), see
///                     the writer for the fields
///   <hash>.bin        the native program exactly as the GPU reads it: SPH header + every code word,
///                     unmodified; written once per stage hash that is new to the session
///   <hash>.ctl        NVN HLE only: the program's control section (with the code it is the
///                     offline coverage key)
///   <hash>.ir.txt     our IR after translation (first pipeline that translated the hash)
///   <hash>.spv        our SPIR-V (same)
///   p<key>.inputs.json runtime inputs that decide what the words do: bound constant buffer
///                     ranges the shader reads, texture/sampler handles and descriptors
///   session.json     title, build, counters (rewritten at the end)
///
/// Hooks are cheap when off: `Enabled()` is one relaxed load of a flag. When on, the call sites
/// build their records on the thread that already did the work (a compile worker, the GPU
/// thread) and hand them to `Submit*`; one background thread formats and writes. Bounded: the
/// total size of the per-hash and per-pipeline files (`.bin`, `.ctl`, `.ir.txt`, `.spv`,
/// `.inputs.json`) is capped (`EDEN_NX2_SHADER_LOG_MAX_MB`, default 1024); once reached a log line
/// says so and no further such files are written. `shaders.jsonl` and `session.json` do not count
/// toward that cap: `shaders.jsonl` lines keep coming after it (marked `files_skipped`) up to its
/// own bound of 256 MiB.
namespace VideoCommon::ShaderLog {

enum class Path { Emulated, Hle };

/// Outcome of a first use. `Fallback` = a stand-in was used instead of the translated program
/// (e.g. the upscale stand-in); `reason` says which.
enum class Outcome { Translated, DiskCache, Rejected, Fallback };

namespace detail {
extern std::atomic<bool> g_enabled;
} // namespace detail

/// True when the log is on for this session. Everything below is a no-op when false.
[[nodiscard]] inline bool Enabled() noexcept {
    return detail::g_enabled.load(std::memory_order_relaxed);
}

/// Called at every game start (CreateGPU): decides from `EDEN_NX2_SHADER_LOG` whether this game
/// session is logged; when it is, starts a new session and its writer thread. The session directory is created on the first
/// write. A call while a session runs is ignored. `title_id` / `build_id` may still be unknown
/// here: see SetTitle.
void Start();

/// Title id and main-module build id (uppercase hex, whatever the loader has; "" unknown). They
/// name the session directory and session.json, so call before the first Submit; calls after the
/// directory was created do not rename it.
void SetTitle(u64 title_id, std::string_view build_id);

/// True once SetTitle ran for the current session (false when no session runs).
[[nodiscard]] bool TitleKnown() noexcept;

/// Ends the session (game stop): flushes the queue, writes session.json and joins the writer; the
/// next Start begins a new session. Idempotent.
void Stop();

/// One presented frame (the frame number of a record is the count at first use). Cheap.
void NoteFrame() noexcept;
[[nodiscard]] u64 Frame() noexcept;

// -- per stage hash ----------------------------------------------------------------------------

/// `hash` is the emulator's stage identity (HLE: HashProgramStage; emulated: the unique hash of
/// the stage). True for the first caller of a hash in this session (claims it): only then build
/// and submit the program record, so repeats cost one lookup, not a copy.
[[nodiscard]] bool ClaimProgram(u64 hash);

struct Program {
    u64 hash{};
    std::string_view stage;       ///< "vs", "tcs", "tes", "gs", "fs", "cs"
    u64 guest_address{};          ///< GPU virtual address of the program (0: NVN HLE, none)
    /// The native program as the GPU reads it: SPH header followed by every code word (graphics);
    /// code words only for compute. Unmodified guest bytes.
    std::span<const u8> bytes;
    std::span<const u8> control;  ///< HLE: the NVN control section; empty otherwise
    bool maxwell{};               ///< legacy Maxwell program (64-bit words, SPH 0x50); else Ampere
    bool rejected{};              ///< the lowering rejected the program
    std::string_view reason;
    u32 failed_instruction{~0U};  ///< instruction index where the lowering stopped (~0 = none)
};
/// Copies what it needs (the caller's buffers are free on return).
void SubmitProgram(const Program& program);

// -- per pipeline key --------------------------------------------------------------------------

/// True for the first caller of this pipeline key in this session (claims it).
[[nodiscard]] bool ClaimPipeline(u64 pipeline_key);

/// A draw/dispatch was skipped because the pipeline was still compiling (async shaders).
void NoteAsyncSkip(u64 pipeline_key);

struct File {
    std::string name;             ///< bare file name inside the session directory
    std::vector<u8> data;
};

struct Stage {
    std::string_view stage;       ///< "vs", "tcs", "tes", "gs", "fs", "cs"
    u64 hash{};
};

struct Pipeline {
    Path path{};
    u64 key{};                    ///< pipeline key hash (what ClaimPipeline took)
    std::vector<Stage> stages;
    Outcome outcome{Outcome::Translated};
    std::string reason;           ///< rejected / fallback only
    double compile_ms{};          ///< translate + pipeline creation; 0 for DiskCache
    /// JSON object text (`{...}`) of the pipeline state the shaders were translated with and the
    /// pipeline was created from (topology, render target formats, blend, vertex layout class...).
    std::string state_json;
    /// Optional extra files for this use (`<hash>.ir.txt`, `<hash>.spv`, `p<key>.inputs.json`).
    /// A file whose name exists is not written again.
    std::vector<File> files;
};
/// Frame number, time and async-skip count are taken at this call.
void SubmitPipeline(Pipeline&& pipeline);

/// The session directory (created on first use, thread safe); empty when the log is off or the
/// directory cannot be created. Used by the command capture (gpu_capture.h).
[[nodiscard]] std::filesystem::path SessionDirectory();

/// JSON string literal of `text` (with the quotes).
[[nodiscard]] std::string Quote(std::string_view text);

} // namespace VideoCommon::ShaderLog
