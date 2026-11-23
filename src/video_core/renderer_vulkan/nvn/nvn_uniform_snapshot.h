// SPDX-FileCopyrightText: Copyright (c) 2026 jhaizhou (d2r_gfx)
// SPDX-FileCopyrightText: Copyright (c) 2026 MacNativeGames contributors
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Uniform snapshot at submit (EDEN_NX2_NVN_UBO_SNAPSHOT), the replay side; the copies are taken
// in the submit call (video_core/nvn/uniform_snapshot.h). A uniform binding that lies in a range
// of its submission's snapshot is bound to an arena copy of the range's bytes (one upload per
// range and submission), and ReadUniformWord (handle words, storage pointers) reads the same
// bytes, so the CPU and the GPU agree.
//
// Drop rule (MacNativeGames usnap_drop / usnap_take, mapped onto the replay's GPU write notes):
// guest memory, not the copy, holds what the GPU wrote. Every GPU write the replay records into a
// pool (Recorder::NotePoolWrite: copies, fills, reports, written storage ranges, write-backs;
// HostNoteShadowedWrite: stores into a storage shadow or mirror entry) is noted here with its
// submission's seq. A range of a snapshot taken when submission `completed_seq` was the newest
// completed one is dropped (its bindings read memory again) when a write of a later submission
// overlaps it: one recorded earlier in the same submission, or one of an earlier submission that
// had not completed in the submit call, or one of another queue's submission replayed meanwhile.
// Inline uniform updates (UpdateUniformBuffer) carry their bytes: one of the submission itself or
// of an earlier submission is layered into the copy instead (in replay order), so the bytes stay
// "memory as the GPU would see it"; one of a later submission drops the range. Raw global stores
// name no range: in the strict mode a raw-writing draw drops every range a raw pointer could reach
// (the raw table's guest ranges) and makes them GPU history; with EDEN_NX2_NVN_UBO_SNAPSHOT=2 only
// the capture's CPU-written test keeps them out. Every noted write except an inline update also
// goes into the GPU history (video_core/nvn/uniform_snapshot.h): later captures never copy those
// pages. The write-back of a mirror entry is not noted again (its stores were noted at the draw,
// or are raw stores). A WaitEvent passed after the CPU changed
// that event's word later than the submit call drops the whole copy for the rest of the
// submission: the game may have written uniforms before it signalled.

#pragma once

#include <chrono>
#include <cstddef>
#include <map>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/sink.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

namespace VC = VideoCommon::Nvn;

struct RawGlobalRange;

/// One submission's snapshot as a DrawContext replays it.
class UniformSnapshotView {
public:
    static constexpr size_t npos = ~size_t{0};

    struct Upload {
        VkBuffer buffer{};
        u64 offset{}; ///< of the range's first byte (same phase modulo 256 as its address)
    };

    /// `snapshot` null: none (every binding reads memory). `seq`: the submission's.
    void Reset(VC::UniformSnapshot* snapshot, u64 seq = 0);
    /// UniformSnapshotTracker's CPU event clock when the submission arrived.
    u64 cpu_clock{};
    [[nodiscard]] bool Active() const noexcept {
        return snapshot != nullptr;
    }
    [[nodiscard]] u64 Seq() const noexcept {
        return seq;
    }
    [[nodiscard]] u64 CompletedSeq() const noexcept {
        return snapshot != nullptr ? snapshot->completed_seq : 0;
    }
    /// The range holding all of [address, address + size), npos when there is none or it was
    /// dropped.
    [[nodiscard]] size_t Find(u64 address, u64 size) const;
    [[nodiscard]] const VC::UniformSnapshot::Range& RangeAt(size_t index) const {
        return snapshot->ranges[index];
    }
    [[nodiscard]] const u8* Bytes(size_t index, u64 address) const {
        const VC::UniformSnapshot::Range& range = snapshot->ranges[index];
        return snapshot->bytes.data() + range.offset + (address - range.begin);
    }
    [[nodiscard]] Upload& UploadAt(size_t index) {
        return uploads[index];
    }
    /// A GPU write of [begin, end): the ranges it overlaps read memory from now on. Returns how
    /// many were dropped.
    u32 Drop(u64 begin, u64 end);
    /// Bytes an inline update writes: they go into the live ranges they overlap (whose next
    /// binding uploads them again; bindings made before keep their upload). Returns how many
    /// ranges took bytes.
    u32 Layer(u64 address, const u8* data, u64 size);
    /// Every range reads memory from now on. Returns how many were live.
    u32 DropAll();
    /// Drops the ranges that overlap one of `sorted` ([begin, end), sorted and disjoint).
    u32 DropOverlapping(const std::vector<std::pair<u64, u64>>& sorted);

private:
    /// The first range whose end is above `address`.
    [[nodiscard]] size_t First(u64 address) const;

    VC::UniformSnapshot* snapshot{};
    u64 seq{};
    std::vector<u8> dropped;
    std::vector<Upload> uploads;
};

/// The GPU writes and the snapshots being replayed. Replay thread only.
class UniformSnapshotTracker {
public:
    /// A submission carrying a snapshot entered the streams (ProcessSubmit order).
    void Arrive(u64 seq, u64 completed_seq);
    /// It left them (replayed, failed or dropped).
    void Leave(u64 seq, u64 completed_seq);
    /// Its replay starts: applies the writes of submissions later than its capture's completed
    /// one (layers earlier submissions' inline updates, drops the rest), then follows later
    /// writes. A snapshot taken before the tracker saw its first one (writes may be missing) is
    /// not used.
    void Activate(UniformSnapshotView& view);
    void Deactivate(UniformSnapshotView& view);
    /// A GPU write of [begin, end) by submission `seq`.
    void NoteWrite(u64 begin, u64 end, u64 seq);
    /// An inline update of submission `seq` lands in memory with these bytes. The GPU write noted
    /// for exactly these bytes until EndInlineWrite is this one (NoteWrite ignores it).
    void NoteInlineWrite(u64 address, const u8* data, u64 size, u64 seq);
    void EndInlineWrite() noexcept {
        inline_write = {};
    }
    /// The guest CPU changed the word of the event at `address` (ordered with the submissions).
    void NoteCpuEvent(u64 address);
    /// The view's submission passed a wait on the event at `address`: drops its whole copy when
    /// the CPU changed that event after the submission was made.
    void OnEventWait(UniformSnapshotView& view, u64 address);
    /// A draw of submission `seq` stores through raw global pointers; `generation` and `ranges`
    /// are the raw table it used (guest ranges). Strict mode only.
    void NoteRawWrite(u64 generation, std::span<const RawGlobalRange> ranges, u64 seq);

    /// Counters for the log (replay thread).
    u64 drops{};
    u64 layers{};
    u64 refused{};
    u64 event_drops{};
    u64 raw_drops{};

private:
    struct Write {
        u64 begin;
        u64 end;
        u64 seq;
        std::vector<u8> data; ///< an inline update's bytes; empty for any other write
    };
    struct InlineWrite {
        u64 begin{};
        u64 end{};
        u64 seq{};
    };
    static constexpr size_t PRUNE_AT = 4096;

    /// One write against one view.
    void Apply(UniformSnapshotView& view, u64 begin, u64 end, u64 seq, const u8* data);
    void Append(u64 begin, u64 end, u64 seq, const u8* data);
    void Prune();
    void LogPerSecond();

    std::vector<Write> writes;
    size_t prune_at{PRUNE_AT};
    InlineWrite inline_write;
    /// completed_seq of every snapshot-carrying submission still in the streams.
    std::multiset<u64> pending;
    /// The largest completed_seq of a snapshot that arrived (later ones carry one as large).
    u64 latest_completed{};
    /// The first submission with a snapshot that arrived: writes of earlier ones may be missing.
    u64 enabled_from{};
    std::vector<UniformSnapshotView*> active;
    /// CPU event changes: the clock, the last change per event address, the clock at each
    /// pending submission's arrival.
    u64 cpu_clock{};
    std::map<u64, u64> cpu_events;
    std::map<u64, u64> arrival_clock;
    /// The last GPU write put into the history (repeats are skipped).
    std::pair<u64, u64> last_history{};
    /// Strict mode: the guest ranges of the raw table of the last raw-writing draw (sorted,
    /// merged), its generation, and the newest submission with such a draw.
    std::vector<std::pair<u64, u64>> raw_ranges;
    u64 raw_generation{~u64{0}};
    u64 raw_seq{};
    u64 activated{};
    std::chrono::steady_clock::time_point log_time{};
};

} // namespace Vulkan::Nvn
