// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

/// Live draw and dispatch census, draw and dispatch skipping by program hash and a draw or
/// dispatch dump for the NVN replay (VTX-HLE), driven by the debug server methods `nvn_draws`,
/// `nvn_draw_skip` and `nvn_draw_dump` (core/debugger/nx2_debug_tools.cpp). No Vulkan types: the
/// replay fills it (DrawContext::ProbeDraw, DrawContext::ProbeDispatchDump), the debug server
/// thread reads it. Unused cost: one relaxed atomic load per draw and dispatch.
namespace VideoCommon::NvnDrawProbe {

/// NVNshaderStage numbering: 0 vertex, 1 fragment, 2 geometry, 3 tess control, 4 tess eval,
/// 5 compute (never set), 6 task, 7 mesh.
inline constexpr size_t StageCount = 8;
inline constexpr size_t MaxStreams = 4; ///< vertex streams described per draw
inline constexpr u32 MaxDumpCount = 8;  ///< draws or dispatches one dump file can hold

struct Stream {
    u32 slot{};
    u64 address{};     ///< guest address of the bound range
    u64 size{};
    u64 pool_offset{}; ///< offset of the range in its pool (0 when unresolved)
    u64 pool_address{};
    bool staged{};     ///< the draw binds the pass's staged device-local copy
};

/// One draw as the replay is about to record it.
struct DrawInfo {
    std::array<u64, StageCount> hashes{};
    u32 mode{};          ///< NVNdrawPrimitive (0xFFFFFFFF = mesh tasks)
    u32 count{};         ///< vertices or indices of a direct draw (0 for indirect / mesh)
    bool indexed{};
    bool indirect{};
    bool mesh{};
    u32 stream_count{};
    std::array<Stream, MaxStreams> streams{};
    u64 index_address{};
    u64 index_bytes{};
    u32 index_type{};
    bool index_staged{};
};

/// One distinct pipeline (stage hashes + topology + draw kind) of a present.
struct Row {
    DrawInfo first;      ///< the first draw of this pipeline in the present
    u64 draws{};
    u64 elements{};      ///< sum of `count`
    u64 skipped{};
};

/// One compute dispatch as the replay is about to record it.
struct DispatchInfo {
    u64 hash{};                  ///< compute program hash, as the GPU profiler prints it
    std::array<u32, 3> groups{}; ///< workgroups of a direct dispatch (0 for indirect)
    bool indirect{};
};

/// The dispatches of one compute program in a present.
struct DispatchRow {
    u64 hash{};
    u64 count{};
    u64 groups{};   ///< sum of x*y*z of the direct ones
    u64 indirect{};
    u64 skipped{};
};

struct Census {
    u64 present{};       ///< replay present number the census ends at (0: none yet)
    u64 draws{};
    u64 skipped{};
    std::vector<Row> rows;
    u64 dispatches{};
    u64 skipped_dispatches{};
    std::vector<DispatchRow> dispatch_rows; ///< most dispatches first
};

/// Draws: Skip drops the draws having a listed stage hash, Only drops every other draw.
/// Dispatches: in both modes the ones whose compute hash is listed are dropped.
enum class SkipMode { Skip, Only };

namespace Detail {
extern std::atomic<bool> g_active;
}

/// Anything armed (census, skip set, dump)? Checked per draw and dispatch.
[[nodiscard]] inline bool Active() {
    return Detail::g_active.load(std::memory_order_relaxed);
}

// ---- replay thread ----
/// Records the draw; returns true when it must be skipped.
[[nodiscard]] bool OnDraw(const DrawInfo& info);
/// Records the dispatch; returns true when its work must be dropped (the replay records a direct
/// one with zero workgroups and does not record an indirect one).
[[nodiscard]] bool OnDispatch(const DispatchInfo& info);
/// A present was recorded: the census of the frame so far becomes the last one.
void OnPresent();
/// One section of the armed dump, claimed by a matching draw or dispatch.
struct DumpTicket {
    std::string path;
    u32 index{}; ///< 0-based: section `index` of `count` in the file
    u32 count{};
    u64 generation{}; ///< the ArmDump call the section belongs to
};
/// A dump is armed for a draw having the hash in any stage: claims the next section.
[[nodiscard]] std::optional<DumpTicket> DumpRequested(const DrawInfo& info);
/// The same for a dispatch whose compute hash matches.
[[nodiscard]] std::optional<DumpTicket> DumpRequested(const DispatchInfo& info);
/// Writes a claimed section (the first one creates the file, the others append to it). The dump
/// is done when its last section is written or a write failed.
void WriteDump(const DumpTicket& ticket, std::string_view text);
/// The dump files' byte rows: per 16 bytes the offset, the bytes in hex, four f32 and four u32.
/// Runs of rows of zeros are collapsed into one line.
void AppendRows(std::string& text, std::span<const u8> bytes);

// ---- debug server ----
/// Keeps the census running for the next `presents` presents. False when it was not running: the
/// first full frame from now on becomes the next census (the partial one up to the next present
/// is dropped).
bool ArmCensus(u32 presents);
[[nodiscard]] Census LastCensus();
/// Disarms everything and forgets the census, skip set and dump (debug server session end).
void Reset();
/// Replaces the set. Returns the current set and mode.
void SetSkip(std::vector<u64> hashes, SkipMode mode);
void GetSkip(std::vector<u64>& hashes, SkipMode& mode);
/// Arms a dump of the next `count` (1 to MaxDumpCount) draws having `hash` in any stage, or
/// dispatches of the compute program `hash`, into one file in `directory`.
void ArmDump(u64 hash, std::string directory, u32 count = 1);
/// State of the last armed dump.
struct DumpState {
    bool armed{};
    bool done{}; ///< every section written, or `error`
    u64 hash{};
    u32 count{};
    u32 written{};
    std::string kind; ///< "draw" or "dispatch" once the first section is claimed
    std::string path;
    std::string error;
};
[[nodiscard]] DumpState GetDump();

} // namespace VideoCommon::NvnDrawProbe
