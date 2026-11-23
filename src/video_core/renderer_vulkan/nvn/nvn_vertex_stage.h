// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Vertex ranges of a render pass copied into device-local memory before the pass begins
// (DrawContext::StageVertexRanges in nvn_draw.cpp; tests: src/tests/nvn/pass_state.cpp; DESIGN.md
// "Vertex staging").
//
// Guest geometry lives in imported host memory, so the GPU fetches every vertex across PCIe. With
// thousands of small draws (DK's shadow passes: ~3,800 draws of ~530 indices and a ~4 KiB vertex
// range each per frame) that fetch latency is most of the pass. When a pass begins, the replay
// looks ahead in the recorded commands for the vertex buffers the pass binds, copies those ranges
// into device-local memory (neighbouring ranges of a pool as one copy region) and lets the draws
// read the copies.
//
// The copies are exact: they run in stream order right before the pass, after the barrier that
// orders everything recorded earlier, and read the very bytes the draws would fetch. Within the pass
// only the pass's own draws can write memory; once one of them writes through shaders the pass stops
// using the copies (NVN orders such writes before later vertex fetches only through a Barrier, which
// ends the pass).
//
// Later passes of the same command buffer reuse a copy instead of copying the range again (DK draws
// the same meshes into several shadow cascades) as long as nothing recorded since can have written
// its bytes: the copies are dropped at every pass end that is not a pure render target / image
// change, after a pass that wrote through shaders or has counter reports to write, at every op
// recorded outside a pass (dispatches, copies, clears, reports, waits) and at every command buffer
// start; a pool write the replay records (DrawContext::OnPoolWrite) drops the copies it overlaps.
// Guest CPU writes a later pass must see are ordered before it through a wait the replay passes
// before recording that pass; a command buffer is submitted only after it is recorded (an early
// submission starts a new one), so the copies, which execute after the submission, read those
// writes as well.
//
// Index ranges (EDEN_NX2_NVN_INDEX_STAGE) are staged the same way and in the same gather: the
// direct indexed draws of the pass name their index bytes (address, count x index size) and the
// bind of the index buffer reads the copy. They are bytes like the vertex ranges, so one table by
// guest address serves both.

#pragma once

#include <algorithm>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/record.h"

namespace Vulkan::Nvn {

struct VertexStageLimits {
    u32 min_draws;       ///< fewer draws ahead in the pass: not worth a copy and a barrier
    u64 max_range_bytes; ///< larger vertex buffer bindings stay where they are
    u64 max_total_bytes; ///< no range is added past this many bytes per pass
    u64 max_gap_bytes;   ///< ranges of a pool this close share one copy region (gap included)
    u64 max_span_bytes;  ///< no copy region grows past this size by merging
};

/// What the replay stages for a pass. Copies from imported host memory cost per byte and per
/// region; fitted from DK city runs on an RTX 5090 (F:/eg/results/vertex-stage2): ~0.15 ms per MiB
/// and ~0.7 us per region on an idle GPU, ~0.4 ms per MiB and ~2.4 us per region while another
/// emulator shares it. A gap is worth copying while it costs less than a region, ~5-6 KiB either
/// way, so ranges up to 4 KiB apart share one region.
inline constexpr VertexStageLimits VERTEX_STAGE_LIMITS{
    .min_draws = 32,
    .max_range_bytes = u64{1} << 20,
    .max_total_bytes = u64{64} << 20,
    .max_gap_bytes = u64{4} << 10,
    .max_span_bytes = u64{2} << 20,
};

/// EDEN_NX2_TRY_VTX_MIRROR: the stale pages a pass copies into pool mirrors for its vertex and
/// index ranges stop at this many bytes (the range that crosses it still goes in whole): a pool's
/// first use cannot stall a pass for a whole pool, and the pages left over come with the later
/// passes that need them. At the fitted ~0.15 ms per MiB (above) it is ~0.6 ms.
inline constexpr u64 VTX_MIRROR_COPY_BYTES = u64{4} << 20;

struct VertexStageScan {
    /// Distinct start addresses in first-use order; a start bound with several sizes keeps the
    /// largest.
    std::vector<VideoCommon::Nvn::BufferRange> ranges;
    u32 draws{}; ///< Draw / DrawIndexed records before the scan stopped
    u64 bytes{}; ///< sum of the ranges' sizes
    bool capped{}; ///< a range (or a larger size of one) was left out for max_total_bytes
    /// Where the scan stopped: the first record it did not read (the op that ends the window, the
    /// list's End) or the end of the records; null for an empty list.
    const u64* end{};
    std::unordered_map<u64, size_t> index; ///< address -> position in `ranges` (scratch)

    void Clear() {
        ranges.clear();
        index.clear();
        draws = 0;
        bytes = 0;
        capped = false;
        end = nullptr;
    }
};

/// The scan predicate of EDEN_NX2_TRY_STAGE_SPAN: `keeps(op)` (ops that can be recorded inside a
/// render pass) as before, but a SetRenderTargets does not end the window. Behind the first one
/// the window also ends at a Barrier (which ends a pass that wrote through shaders, dropping the
/// copies), a render condition (its predicate is computed outside a pass) and a CallList (the
/// records of the called list are not scanned): ranges past them would be copied for nothing.
template <typename Keeps>
struct SpanWindow {
    Keeps keeps;
    bool crossed{};

    [[nodiscard]] bool operator()(VideoCommon::Nvn::Op op) {
        namespace VC = VideoCommon::Nvn;
        if (op == VC::Op::SetRenderTargets) {
            crossed = true;
            return true;
        }
        if (crossed && (op == VC::Op::Barrier || op == VC::Op::SetRenderCondition ||
                        op == VC::Op::CallList)) {
            return false;
        }
        return keeps(op);
    }
};
template <typename Keeps>
SpanWindow(Keeps) -> SpanWindow<Keeps>;

/// Which index ranges ScanPassVertexRanges adds.
struct IndexStageFilter {
    bool enabled{};       ///< the index ranges of direct DrawIndexed records are staged too
    bool uint8_indices{}; ///< the device binds 8-bit indices (VK_EXT_index_type_uint8)
};

/// Bytes of one index of NVNindexType `type` (0 = 8-bit, 1 = 16-bit, 2 = 32-bit); 0 otherwise.
[[nodiscard]] constexpr u32 VertexStageIndexSize(u32 type) noexcept {
    return type <= 2 ? 1u << type : 0u;
}

/// Whether a direct indexed draw binds its guest index bytes as they are: line loops (2), quads (7)
/// and quad strips (8), and 8-bit indices on a device without them, are converted into generated
/// indices (DrawContext::DrawIndexed) and never bind the guest range.
[[nodiscard]] constexpr bool VertexStageBindsIndices(const VideoCommon::Nvn::DrawIndexedPayload& p,
                                                     bool uint8_indices) noexcept {
    return p.indirect == 0 && p.count_address == 0 && p.count != 0 && p.index_address != 0 &&
           p.mode != 2 && p.mode != 7 && p.mode != 8 && VertexStageIndexSize(p.index_type) != 0 &&
           (p.index_type != 0 || uint8_indices);
}

/// Collects the vertex buffer ranges a pass reads: `bound` (bound when the pass begins) and every
/// BindVertexBuffers record of `records` (the records after the one being replayed) up to the list's
/// End, the first record that cannot stay in a render pass (`keeps_pass(op)` false) or the first
/// malformed one; with `indices.enabled`, also the index range of every DrawIndexed record that
/// binds its guest indices (VertexStageBindsIndices). Counts the draw records on the way. Ranges of
/// size 0 or above max_range_bytes are left out, and so is a range that would take the total past
/// max_total_bytes.
template <typename KeepsPass>
void ScanPassVertexRanges(std::span<const u64> records,
                          std::span<const VideoCommon::Nvn::BufferRange> bound,
                          const VertexStageLimits& limits, const IndexStageFilter& indices,
                          KeepsPass&& keeps_pass, VertexStageScan& out) {
    namespace VC = VideoCommon::Nvn;
    out.Clear();
    const auto add = [&](const VC::BufferRange& range) {
        if (range.address == 0 || range.size == 0 || range.size > limits.max_range_bytes) {
            return;
        }
        const auto [it, inserted] = out.index.try_emplace(range.address, out.ranges.size());
        if (inserted) {
            if (out.bytes + range.size > limits.max_total_bytes) {
                out.index.erase(it);
                out.capped = true;
                return;
            }
            out.ranges.push_back(range);
            out.bytes += range.size;
            return;
        }
        VC::BufferRange& known = out.ranges[it->second];
        if (range.size > known.size) {
            if (out.bytes + (range.size - known.size) <= limits.max_total_bytes) {
                out.bytes += range.size - known.size;
                known.size = range.size;
            } else {
                out.capped = true;
            }
        }
    };
    for (const VC::BufferRange& range : bound) {
        add(range);
    }
    VC::RecordReader reader{records.data(), records.size()};
    out.end = records.data();
    while (true) {
        const u64* const before = reader.Cursor();
        const VC::RecordHeader* header = reader.Next();
        if (header == nullptr) {
            out.end = before;
            return;
        }
        const auto op = VC::RecordOp(header);
        if (!keeps_pass(op)) {
            out.end = before;
            return;
        }
        if (op == VC::Op::Draw) {
            ++out.draws;
        } else if (op == VC::Op::DrawIndexed) {
            ++out.draws;
            if (indices.enabled &&
                VC::RecordPayloadSize(header) >= sizeof(VC::DrawIndexedPayload)) {
                const auto* draw = VC::RecordPayload<VC::DrawIndexedPayload>(header);
                if (VertexStageBindsIndices(*draw, indices.uint8_indices)) {
                    add({draw->index_address,
                         u64{draw->count} * VertexStageIndexSize(draw->index_type)});
                }
            }
        } else if (op == VC::Op::BindVertexBuffers &&
                   VC::RecordPayloadSize(header) >= sizeof(VC::BindVertexBuffersPayload)) {
            const auto* payload = VC::RecordPayload<VC::BindVertexBuffersPayload>(header);
            const size_t available =
                (VC::RecordPayloadSize(header) - sizeof(VC::BindVertexBuffersPayload)) /
                sizeof(VC::BufferRange);
            const size_t count = (std::min<size_t>)(payload->count, available);
            const auto* ranges = VC::TrailingArray<VC::BufferRange>(payload);
            for (size_t i = 0; i < count; ++i) {
                add(ranges[i]);
            }
        }
    }
}

/// A staged range resolved to its pool.
struct VertexStageSource {
    u64 pool;    ///< identifies the pool buffer (copy regions never cross pools)
    u64 offset;  ///< in the pool
    u64 size;
    u64 address; ///< the guest address the draws bind
    u32 span{};  ///< out: the copy region holding it (index into the spans)
};

/// One copy region: bytes [begin, end) of a pool.
struct VertexStageSpan {
    u64 pool;
    u64 begin;
    u64 end;
};

/// Sorts `sources` by pool and offset and merges them into copy regions: a source joins the open
/// region of its pool when it overlaps it or starts at most max_gap_bytes past its end, the region
/// stays within max_span_bytes (a single larger source keeps its own region), and the gap bytes
/// copied for the pass stay within the bytes of the sources themselves (so a pass never copies
/// more than twice what its draws read). Sets every source's `span`; returns the bytes the
/// regions cover.
inline u64 CoalesceVertexStageSources(std::span<VertexStageSource> sources,
                                      const VertexStageLimits& limits,
                                      std::vector<VertexStageSpan>& spans) {
    spans.clear();
    std::ranges::sort(sources, [](const VertexStageSource& a, const VertexStageSource& b) {
        return a.pool != b.pool ? a.pool < b.pool : a.offset < b.offset;
    });
    u64 source_bytes = 0;
    for (const VertexStageSource& source : sources) {
        source_bytes += source.size;
    }
    u64 gap_bytes = 0;
    u64 covered = 0;
    for (VertexStageSource& source : sources) {
        const u64 end = source.offset + source.size;
        if (!spans.empty()) {
            VertexStageSpan& open = spans.back();
            const u64 gap = source.offset > open.end ? source.offset - open.end : 0;
            const u64 merged_end = (std::max)(open.end, end);
            if (open.pool == source.pool && gap <= limits.max_gap_bytes &&
                merged_end - open.begin <= limits.max_span_bytes &&
                gap_bytes + gap <= source_bytes) {
                gap_bytes += gap;
                covered += merged_end - open.end;
                open.end = merged_end;
                source.span = static_cast<u32>(spans.size() - 1);
                continue;
            }
        }
        source.span = static_cast<u32>(spans.size());
        spans.push_back({source.pool, source.offset, end});
        covered += source.size;
    }
    return covered;
}

/// One workgroup of nvn_vertex_stage.comp (VertexStageGpu): `count` 16-byte units from unit
/// `source` of the bound source window to unit `destination` of the bound destination buffer.
struct VertexStageTask {
    u32 source;
    u32 destination;
    u32 count;
    u32 unused;
};
static_assert(sizeof(VertexStageTask) == 16);

/// 16-byte units one workgroup moves at most (64 KiB): big regions spread over many workgroups.
inline constexpr u32 VERTEX_STAGE_TASK_UNITS = 4096;

/// Appends the tasks that copy `size` bytes from byte `source_offset` of the source window to byte
/// `destination_offset` of the destination buffer; the two offsets are equal modulo 16. The copy
/// widens to whole 16-byte units at both ends: the extra bytes are the source's own neighbours,
/// and they land in the destination's alignment padding (AllocShadow starts every allocation at a
/// multiple of 256 plus the source offset modulo 256, and chunk sizes are multiples of 256).
inline void AppendVertexStageTasks(u64 source_offset, u64 destination_offset, u64 size,
                                   std::vector<VertexStageTask>& tasks) {
    const u64 head = source_offset % 16;
    u64 source_unit = source_offset / 16;
    u64 destination_unit = (destination_offset - head) / 16;
    u64 units = (head + size + 15) / 16;
    while (units != 0) {
        const u64 count = (std::min<u64>)(units, VERTEX_STAGE_TASK_UNITS);
        tasks.push_back({static_cast<u32>(source_unit), static_cast<u32>(destination_unit),
                         static_cast<u32>(count), 0});
        source_unit += count;
        destination_unit += count;
        units -= count;
    }
}

} // namespace Vulkan::Nvn
