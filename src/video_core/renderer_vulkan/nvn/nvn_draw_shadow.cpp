// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Storage shadows: device-local copies that stand in for written storage ranges of imported pools.
//
// An imported pool is the guest's own memory (host memory mapped into the GPU). Shader stores and
// atomics into it cross PCIe: DK's shadow-map pass does one OpAtomicUMax per fragment into a small
// storage buffer, 15-19 ns each (4.5 ms per frame at 720x405; DESIGN.md "GPU profiler"). A written
// storage range of at most `shadow_max_bytes` (1 MiB) is therefore bound through a device-local
// shadow:
//
//   * created when a draw or dispatch binds the written range and the pass is not open (a pass
//     cannot copy): copy host -> shadow (the shadow starts as an exact copy of the guest bytes);
//   * while it is live the shadow is the truth for [begin, end) and the pool's host memory is stale.
//     Later draws and dispatches that bind a range inside it (storage or uniform) are served from
//     it too (no copies at all for repeated use); every other use of those bytes (another range
//     that only partly overlaps, vertex/index/indirect reads, copies, clears, reports, inline
//     updates, raw global memory) first copies the shadow back and ends it;
//   * copied back (shadow -> host) when it ends: on such a conflict, and at the latest before the
//     host barrier of the command buffer (`Recorder::EndSegment` calls FlushShadows). A completion
//     callback therefore never fires before the guest bytes hold what the GPU wrote, and another
//     queue that waits for a fence of the segment reads the bytes through host memory as before.
//
// Kept in host memory (as before): ranges bigger than the limit, size-0 "rest of the pool" ranges,
// draws with raw global memory (their shaders reach every pool by address), ranges that another use
// of the same draw overlaps (any use that is not a descriptor binding inside the range), ranges
// that first appear in an open pass (a pass break would cost more than it gains; counted), and
// everything when `EDEN_NX2_NVN_SHADOW_STORAGE=0`.
//
// Shadows never outlive a command buffer: the recorder flushes them before it ends one, so a
// submission boundary, a cross-queue wait or an early partial submit sees host memory up to date.
//
// Pool mirrors (EDEN_NX2_NVN_POOL_MIRROR, DESIGN.md "Pool mirrors") replace the shadows' copies
// with a VRAM copy of the whole pool that outlives command buffers (Pool::vram_mirror). The same
// entries and conflict rules apply, with three differences: an entry's buffer is the mirror, so
// creating one copies in only the pages that are stale in the mirror (never copied, or written by
// the guest CPU or the replay since); read-only storage ranges are served from the mirror too
// (copied in the same way, no entry); and an entry ends with a word merge (snapshot of the mirror
// taken before the GPU wrote, then only the words that differ go back to the guest bytes), which
// keeps the mirror valid for the next command buffer and keeps the guest CPU's writes to the other
// words of the range. After the idea of switch2-recomp (nvn_gpu_pool.cpp) and totk-pc-recomp
// (vk_memory.c: VRAM mirror of buffer-bound pools, write-back merge).

#include <algorithm>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_write_back.h"

namespace Vulkan::Nvn {
namespace {

/// Live storage shadows (copies in per-command-buffer chunks); entries of pool mirrors may use all
/// of DrawContext::MAX_LIVE_SHADOWS.
constexpr size_t MAX_LIVE_COPY_SHADOWS = 32;
constexpr u64 SHADOW_ALIGNMENT = 256;
constexpr u32 MAX_LOGGED_SHADOWS = 96;
/// EDEN_NX2_NVN_WB_MONITOR arms access traps on partial-pool entries up to this size (the raw
/// dispatches' whole-pool entries stay unarmed: protecting them every segment would dominate the
/// measurement).
constexpr u64 MONITOR_MAX_ENTRY_BYTES = u64{32} << 20;

[[nodiscard]] constexpr bool Overlaps(u64 begin, u64 end, u64 other_begin, u64 other_end) {
    return begin < other_end && other_begin < end;
}

} // namespace

DrawContext::ShadowDecision DrawContext::ClassifyShadows(const DrawPlan& plan) {
    ShadowDecision decision;
    decision.any =
        plan.shadow_candidates || (plan.raw_global && impl.try_storage_vram);
    if (shadows.empty()) {
        return decision;
    }
    if (plan.raw_global) {
        // The shaders reach every pool by address: nothing may stay in a shadow. A mirror entry
        // lives in the pool's mirror, which the raw table resolves the pool's addresses to
        // (EDEN_NX2_TRY_RAW_KEEP_ENTRIES): the raw loads and stores meet the entry's bytes, and its
        // word merge sees the raw stores as GPU changes too, so it can stay.
        const bool keep_mirror_entries = impl.raw_keep_entries && impl.try_storage_vram;
        for (size_t i = 0; i < shadows.size(); ++i) {
            if (!keep_mirror_entries || !shadows[i].mirror) {
                decision.flush_mask.set(i);
                decision.flush_needed = true;
            }
        }
        decision.any = true;
        return decision;
    }
    for (const DrawPlan::RangeUse& use : plan.ranges) {
        if (use.pool->backing != Pool::Backing::Imported) {
            continue;
        }
        const u64 use_end = use.offset + use.size;
        for (size_t i = 0; i < shadows.size(); ++i) {
            const Shadow& shadow = shadows[i];
            if (shadow.pool != use.pool || !Overlaps(use.offset, use_end, shadow.begin, shadow.end)) {
                continue;
            }
            decision.any = true;
            // A descriptor binding inside the shadow is served from it; anything else would read or
            // write stale host bytes.
            const bool inside = use.info_index != DrawPlan::RangeUse::NO_INFO &&
                                use.offset >= shadow.begin && use_end <= shadow.end;
            if (!inside) {
                const bool mirror_ro =
                    shadow.mirror && use.shadowable && !use.write &&
                    use.info_index != DrawPlan::RangeUse::NO_INFO &&
                    std::ranges::none_of(plan.ranges, [&](const DrawPlan::RangeUse& other) {
                        return other.write && other.pool == use.pool &&
                               Overlaps(use.offset, use_end, other.offset,
                                        other.offset + other.size);
                    });
                if (mirror_ro) {
                    if (pass_open && !impl.MirrorCurrent(*use.pool, use.offset, use.size)) {
                        decision.need_outside_pass = true;
                    }
                    continue;
                }
                decision.flush_mask.set(i);
                decision.flush_needed = true;
            }
        }
    }
    return decision;
}

void DrawContext::ApplyShadows(DrawPlan& plan, bool allow_create) {
    // In an open pass PrepareOutsidePass already confirmed that no live shadow conflicts.
    // Outside a pass, image and buffer preparation may have changed the live shadows.
    if (allow_create) {
        const ShadowDecision decision = ClassifyShadows(plan);
        if (decision.flush_needed) {
            const auto cause = plan.raw_global ? &DrawCounters::shadow_flush_raw
                                               : &DrawCounters::shadow_flush_conflict;
            const ShadowCopyBatch batch{*this};
            for (size_t i = shadows.size(); i-- > 0;) {
                if (decision.flush_mask.test(i)) {
                    FlushShadow(i, cause);
                }
            }
        }
    }
    if (impl.pool_mirror_mode != 0) {
        ApplyMirrors(plan, allow_create);
        return;
    }
    // 2. Written storage ranges become shadows.
    if (plan.shadow_candidates) {
        const ShadowCopyBatch batch{*this};
        for (DrawPlan::RangeUse& use : plan.ranges) {
            if (!use.shadowable) {
                continue;
            }
            const u64 use_end = use.offset + use.size;
            if (FindShadow(*use.pool, use.offset, use_end) != shadows.size()) {
                continue; // inside a live shadow: served from it below
            }
            if (!allow_create) {
                ++counters.shadow_skipped_open_pass;
                continue;
            }
            if (!ShadowIsAloneIn(plan, use)) {
                ++counters.shadow_skipped_alias;
                continue;
            }
            if (shadows.size() >= MAX_LIVE_COPY_SHADOWS || !CreateShadow(plan, use)) {
                ++counters.shadow_skipped_no_room;
            }
        }
    }
    // 3. Descriptor bindings inside a live shadow are served from it.
    if (shadows.empty()) {
        return;
    }
    for (DrawPlan::RangeUse& use : plan.ranges) {
        if (use.info_index == DrawPlan::RangeUse::NO_INFO ||
            use.pool->backing != Pool::Backing::Imported) {
            continue;
        }
        const size_t index = FindShadow(*use.pool, use.offset, use.offset + use.size);
        if (index == shadows.size()) {
            continue;
        }
        const Shadow& shadow = shadows[index];
        scratch_buffers[use.info_index] = {shadow.buffer, shadow.offset + (use.offset - shadow.begin),
                                           use.size};
        use.redirected = true;
        ++counters.shadow_redirects;
    }
}

void DrawContext::ApplyMirrors(DrawPlan& plan, bool allow_create) {
    const auto serve = [&](DrawPlan::RangeUse& use, VkBuffer buffer) {
        scratch_buffers[use.info_index] = {buffer, use.offset, use.size};
        use.redirected = true;
        impl.stats.vram_bindings.fetch_add(1, std::memory_order_relaxed);
    };
    // A mirror entry's buffer holds the pool at its own offsets, a copy shadow's slot does not.
    const auto serve_entry = [&](DrawPlan::RangeUse& use, const Shadow& entry) {
        scratch_buffers[use.info_index] = {entry.buffer, entry.offset + (use.offset - entry.begin),
                                           use.size};
        use.redirected = true;
        impl.stats.vram_bindings.fetch_add(1, std::memory_order_relaxed);
    };
    // A raw pointer can address any byte of a mirrored pool. Bring every stale page in before
    // dispatch, and snapshot the complete pool when the shader may write through that pointer.
    if (plan.raw_global) {
        for (Pool* const pool : impl.vram_mirrored) {
            PoolMirror* const mirror = impl.VramMirror(*pool, false);
            // Read-only mirrors are not in the raw table: raw pointers reach their guest bytes.
            if (mirror == nullptr || !mirror->storage) {
                continue;
            }
            mirror_runs.clear();
            impl.MirrorTakeStale(*pool, 0, pool->size, mirror_runs);
            for (const auto& [begin, end] : mirror_runs) {
                mirror_copies.push_back({pool, begin, end});
            }
            if (!plan.raw_global_writes || FindShadow(*pool, 0, pool->size) != shadows.size()) {
                continue;
            }
            // These stores also reach the imported guest backing, so a whole-pool snapshot and
            // merge would only repeat bytes already visible to the guest.
            if (plan.raw_stores_written_through && mirror->written_through) {
                continue;
            }
            if (shadows.size() >= MAX_LIVE_SHADOWS) {
                // Kept entries (EDEN_NX2_TRY_RAW_KEEP_ENTRIES) can fill the table: a partial
                // entry goes back (a whole-pool entry of this draw or an earlier one covers its
                // range) so that no pool of a raw-writing draw is left without a snapshot.
                const auto partial = std::ranges::find_if(shadows, [](const Shadow& entry) {
                    return entry.mirror && !entry.snapshot_pending &&
                           !(entry.begin == 0 && entry.end >= entry.pool->size);
                });
                if (partial == shadows.end()) {
                    ++counters.shadow_skipped_no_room;
                    continue;
                }
                FlushShadow(static_cast<size_t>(partial - shadows.begin()),
                            &DrawCounters::shadow_flush_conflict);
            }
            VkBuffer snapshot{};
            u64 snapshot_offset = 0;
            if (!AllocShadow(pool->size, 0, snapshot, snapshot_offset)) {
                ++counters.shadow_skipped_no_room;
                continue;
            }
            shadows.push_back({pool, 0, pool->size, *mirror->buffer, 0, true, snapshot,
                               snapshot_offset, true});
            ++counters.shadow_creates;
            counters.shadow_bytes_in += pool->size;
            impl.stats.vram_write_ranges.fetch_add(1, std::memory_order_relaxed);
            impl.stats.vram_write_bytes.fetch_add(pool->size, std::memory_order_relaxed);
        }
    }
    // A written range served from guest memory makes every other range of the plan it overlaps
    // stay there too: one draw must not see the same bytes in two places.
    const auto host_write_overlaps = [&](const DrawPlan::RangeUse& use) {
        for (const DrawPlan::RangeUse& other : plan.ranges) {
            if (&other != &use && other.write && other.pool == use.pool && !other.redirected &&
                Overlaps(use.offset, use.offset + use.size, other.offset,
                         other.offset + other.size)) {
                return true;
            }
        }
        return false;
    };
    // Pass 1: written ranges (entries), so that pass 2 knows which writes stay in guest memory.
    for (DrawPlan::RangeUse& use : plan.ranges) {
        if (!use.shadowable || !use.write) {
            continue;
        }
        const u64 use_end = use.offset + use.size;
        if (const size_t index = FindShadow(*use.pool, use.offset, use_end); index != shadows.size()) {
            serve_entry(use, shadows[index]);
            continue;
        }
        if (!allow_create) {
            // The snapshot needs a copy: the write goes to guest memory (Recorder::NotePoolWrite
            // makes the mirror copy those pages in again at their next use).
            ++counters.shadow_skipped_open_pass;
            impl.stats.vram_bindings_host.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!ShadowIsAloneIn(plan, use)) {
            ++counters.shadow_skipped_alias;
            continue;
        }
        PoolMirror* const mirror = impl.VramMirror(*use.pool, true);
        if (mirror == nullptr) {
            // No mirror for this pool (EDEN_NX2_TRY_REFUSED_SHADOW): a small range goes through a
            // copy shadow as with mirrors off. Raw-global plans end copy shadows, so not for them.
            const Debug::Config& debug = Debug::Get();
            const bool copy_shadow =
                impl.refused_shadow && !plan.raw_global && debug.shadow_storage &&
                use.size <= debug.refused_shadow_max_bytes && shadows.size() < MAX_LIVE_SHADOWS &&
                std::ranges::count_if(shadows, [](const Shadow& entry) { return !entry.mirror; }) <
                    static_cast<std::ptrdiff_t>(MAX_LIVE_COPY_SHADOWS);
            bool created = false;
            if (copy_shadow) {
                const ShadowCopyBatch batch{*this};
                created = CreateShadow(plan, use);
            }
            if (created) {
                serve_entry(use, shadows.back());
                continue;
            }
            ++counters.shadow_skipped_no_room;
            if (debug.shadow_log && host_write_logged.size() < MAX_LOGGED_SHADOWS &&
                host_write_logged.emplace(use.pool->address, use.offset, use.size).second) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] written storage range stays in host memory (pool has no mirror): "
                         "pool {:#x} ({:#x} bytes, flags {:#x}) range {:#x}+{:#x} for {} "
                         "vs={:016x} fs={:016x} cs={:016x}",
                         use.pool->address, use.pool->size, use.pool->flags, use.offset, use.size,
                         plan.graphics ? "a draw" : "a dispatch", ProfStageHash(0),
                         ProfStageHash(4), ProfStageHash(STAGE_COMPUTE));
            }
            continue;
        }
        if (shadows.size() >= MAX_LIVE_SHADOWS) {
            // The oldest entry goes back first (merged; its mirror pages stay valid), unless this
            // plan made it or already binds it (its writes would land after the merge).
            const Shadow& front = shadows.front();
            const bool served =
                std::ranges::any_of(plan.ranges, [&](const DrawPlan::RangeUse& other) {
                    return other.redirected && other.pool == front.pool &&
                           Overlaps(other.offset, other.offset + other.size, front.begin,
                                    front.end);
                });
            if (front.snapshot_pending || served) {
                ++counters.shadow_skipped_no_room;
                continue;
            }
            FlushShadow(0, &DrawCounters::shadow_flush_conflict);
        }
        VkBuffer snapshot{};
        u64 snapshot_offset = 0;
        if (!AllocShadow(use.size, use.offset, snapshot, snapshot_offset)) {
            ++counters.shadow_skipped_no_room;
            continue;
        }
        mirror_runs.clear();
        impl.MirrorTakeStale(*use.pool, use.offset, use.size, mirror_runs);
        for (const auto& [begin, end] : mirror_runs) {
            mirror_copies.push_back({use.pool, begin, end});
        }
        shadows.push_back({use.pool, use.offset, use_end, *mirror->buffer, use.offset, true,
                           snapshot, snapshot_offset, true});
        ++counters.shadow_creates;
        counters.shadow_bytes_in += use.size;
        impl.stats.vram_write_ranges.fetch_add(1, std::memory_order_relaxed);
        impl.stats.vram_write_bytes.fetch_add(use.size, std::memory_order_relaxed);
        if (Debug::Get().shadow_log && shadow_logged.size() < MAX_LOGGED_SHADOWS &&
            shadow_logged.emplace(use.pool->address, use.offset, use.size).second) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] pool mirror entry: pool {:#x} ({:#x} bytes, flags {:#x}) range "
                     "{:#x}+{:#x} for {} vs={:016x} fs={:016x} cs={:016x}",
                     use.pool->address, use.pool->size, use.pool->flags, use.offset, use.size,
                     plan.graphics ? "a draw" : "a dispatch", ProfStageHash(0), ProfStageHash(4),
                     ProfStageHash(STAGE_COMPUTE));
        }
        serve(use, *mirror->buffer);
    }
    // Pass 2: read-only ranges.
    for (DrawPlan::RangeUse& use : plan.ranges) {
        if (!use.shadowable || use.write) {
            continue;
        }
        const u64 use_end = use.offset + use.size;
        if (const size_t index = FindShadow(*use.pool, use.offset, use_end); index != shadows.size()) {
            serve_entry(use, shadows[index]);
            continue;
        }
        if (host_write_overlaps(use)) {
            continue;
        }
        PoolMirror* const mirror =
            impl.VramMirror(*use.pool, allow_create && (impl.pool_mirror_mode >= 2 || use.creates_mirror),
                            !use.creates_mirror);
        if (mirror == nullptr) {
            continue;
        }
        // A read-only mirror does not hold what a raw store of this draw writes.
        if (!mirror->storage && plan.raw_global_writes) {
            impl.stats.vram_bindings_host.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (use.creates_mirror) {
            // The pages uniform bindings use: PrefetchMirrors keeps them current from now on.
            impl.MirrorMarkPrefetch(*use.pool, use.offset, use.size);
        }
        if (!allow_create) {
            if (impl.MirrorCurrent(*use.pool, use.offset, use.size)) {
                serve(use, *mirror->buffer);
            } else {
                impl.stats.vram_bindings_host.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        mirror_runs.clear();
        impl.MirrorTakeStale(*use.pool, use.offset, use.size, mirror_runs);
        for (const auto& [begin, end] : mirror_runs) {
            mirror_copies.push_back({use.pool, begin, end});
        }
        serve(use, *mirror->buffer);
    }
    RecordMirrorCopies();
    const auto pending = [](const Shadow& entry) { return entry.snapshot_pending; };
    if (std::ranges::none_of(shadows, pending)) {
        return;
    }
    // The snapshots: the mirror's bytes (copied in above) before the GPU writes them.
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy, .hash_a = GpuProf::SUB_SHADOW}
             : GpuProf::Tag{}};
    thread_local std::vector<BufferAccess> accesses;
    accesses.clear();
    for (const Shadow& entry : shadows) {
        if (entry.snapshot_pending) {
            accesses.push_back({entry.buffer, entry.begin, entry.end, false});
            accesses.push_back({entry.snapshot, entry.snapshot_offset,
                                entry.snapshot_offset + (entry.end - entry.begin), true});
        }
    }
    host->HostBeginBufferOp(accesses);
    for (Shadow& entry : shadows) {
        if (!entry.snapshot_pending) {
            continue;
        }
        entry.snapshot_pending = false;
        const VkBufferCopy region{.srcOffset = entry.begin,
                                  .dstOffset = entry.snapshot_offset,
                                  .size = entry.end - entry.begin};
        host->HostCmd().CopyBuffer(entry.buffer, entry.snapshot, region);
    }
    host->HostEndBufferOp();
}

void DrawContext::RecordMirrorCopies(bool before_vertex_input) {
    if (mirror_copies.empty()) {
        return;
    }
    // A page copied in must not overwrite what a live entry's GPU writes left in the mirror (an
    // entry need not cover whole pages): its bytes are left out, and the page stays stale so that
    // a use after the entry's merge copies it whole (with the guest CPU's writes, if any).
    for (const Shadow& entry : shadows) {
        if (!entry.mirror || entry.snapshot_pending) {
            continue;
        }
        for (size_t i = 0; i < mirror_copies.size(); ++i) {
            MirrorCopy& copy = mirror_copies[i];
            if (copy.pool != entry.pool || !Overlaps(copy.begin, copy.end, entry.begin, entry.end)) {
                continue;
            }
            impl.MirrorInvalidate(*copy.pool, copy.begin, copy.end - copy.begin);
            const MirrorCopy tail{copy.pool, entry.end, copy.end};
            copy.end = (std::max)(copy.begin, entry.begin);
            if (tail.end > tail.begin) {
                mirror_copies.push_back(tail);
            }
        }
    }
    std::erase_if(mirror_copies, [](const MirrorCopy& copy) { return copy.end <= copy.begin; });
    if (mirror_copies.empty()) {
        return;
    }
    std::ranges::sort(mirror_copies, [](const MirrorCopy& a, const MirrorCopy& b) {
        return a.pool != b.pool ? a.pool < b.pool : a.begin < b.begin;
    });
    u64 bytes = 0;
    for (const MirrorCopy& copy : mirror_copies) {
        bytes += copy.end - copy.begin;
    }
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Upload,
                            .count = static_cast<u32>((std::min<u64>)(bytes, ~u32{0})),
                            .hash_a = GpuProf::SUB_MIRROR}
             : GpuProf::Tag{}};
    // Earlier commands may still read these mirror pages (or write the guest bytes).
    thread_local std::vector<BufferAccess> accesses;
    accesses.clear();
    for (const MirrorCopy& copy : mirror_copies) {
        accesses.push_back({copy.pool->Handle(), copy.begin, copy.end, false});
        accesses.push_back({*copy.pool->vram_mirror->buffer, copy.begin, copy.end, true});
    }
    host->HostBeginBufferOp(accesses);
    for (size_t i = 0; i < mirror_copies.size();) {
        Pool* const pool = mirror_copies[i].pool;
        mirror_regions.clear();
        for (; i < mirror_copies.size() && mirror_copies[i].pool == pool; ++i) {
            const MirrorCopy& copy = mirror_copies[i];
            if (!mirror_regions.empty() &&
                mirror_regions.back().srcOffset + mirror_regions.back().size >= copy.begin) {
                const u64 end = (std::max)(mirror_regions.back().srcOffset +
                                               mirror_regions.back().size,
                                           copy.end);
                mirror_regions.back().size = end - mirror_regions.back().srcOffset;
                continue;
            }
            mirror_regions.push_back(
                {.srcOffset = copy.begin, .dstOffset = copy.begin, .size = copy.end - copy.begin});
        }
        host->HostCmd().CopyBuffer(pool->Handle(), *pool->vram_mirror->buffer, mirror_regions);
    }
    if (before_vertex_input) {
        // The pass that begins next fetches vertices and indices from these pages (the pending
        // barrier is not emitted before a render pass begins).
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
        };
        host->HostCmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, barrier);
    }
    host->HostEndBufferOp();
    mirror_copies.clear();
}

void DrawContext::PrefetchMirrors() {
    // Uniform ranges, and the vertex/index ranges EDEN_NX2_TRY_VTXCOV_INPASS/_SHADERWRITES found
    // stale at their bind, mark the pages they want (PoolMirror::wanted).
    if (!impl.ubo_mirror && (impl.vtxcov & (VTXCOV_INPASS | VTXCOV_SHADER_WRITES)) == 0) {
        return;
    }
    for (const auto& entry : impl.vram_prefetch_mirrors) {
        if (entry.write_epoch != nullptr &&
            entry.write_epoch->load(std::memory_order_acquire) == entry.mirror->epoch_clean &&
            !entry.mirror->wanted_dirty) {
            continue;
        }
        mirror_runs.clear();
        impl.MirrorTakeStaleWanted(*entry.pool, mirror_runs);
        for (const auto& [begin, end] : mirror_runs) {
            mirror_copies.push_back({entry.pool, begin, end});
        }
    }
    // Before the barrier flush of the pass begin: its barrier orders the copies before the draws'
    // uniform reads.
    RecordMirrorCopies();
}

size_t DrawContext::FindShadow(const Pool& pool, u64 begin, u64 end) const {
    for (size_t i = 0; i < shadows.size(); ++i) {
        if (shadows[i].pool == &pool && begin >= shadows[i].begin && end <= shadows[i].end) {
            return i;
        }
    }
    return shadows.size();
}

bool DrawContext::ShadowIsAloneIn(const DrawPlan& plan, const DrawPlan::RangeUse& candidate) const {
    const u64 end = candidate.offset + candidate.size;
    for (const DrawPlan::RangeUse& other : plan.ranges) {
        if (&other == &candidate || other.pool != candidate.pool ||
            !Overlaps(candidate.offset, end, other.offset, other.offset + other.size)) {
            continue;
        }
        // Another descriptor binding inside the range is served from the same shadow.
        const bool inside = other.info_index != DrawPlan::RangeUse::NO_INFO &&
                            other.offset >= candidate.offset && other.offset + other.size <= end;
        if (!inside) {
            return false;
        }
    }
    return true;
}

bool DrawContext::AllocShadow(u64 size, u64 begin, VkBuffer& buffer, u64& offset) {
    InFlight& flight = host->HostFlight();
    // The shadow offset keeps the pool offset's alignment modulo 256: a binding inside it has the
    // alignment its host binding had (uniform and storage offset limits).
    const u64 skew = begin % SHADOW_ALIGNMENT;
    const auto start_in = [&](const ShadowChunk& chunk) {
        return (chunk.used + SHADOW_ALIGNMENT - 1) / SHADOW_ALIGNMENT * SHADOW_ALIGNMENT + skew;
    };
    if (flight.shadow.empty() || start_in(*flight.shadow.back()) + size > flight.shadow.back()->size) {
        std::shared_ptr<ShadowChunk> chunk = resources.AcquireShadow(size + 2 * SHADOW_ALIGNMENT);
        if (!chunk) {
            return false;
        }
        flight.shadow.push_back(std::move(chunk));
    }
    ShadowChunk& chunk = *flight.shadow.back();
    offset = start_in(chunk);
    chunk.used = offset + size;
    buffer = *chunk.buffer;
    return true;
}

bool DrawContext::CreateShadow(DrawPlan& plan, const DrawPlan::RangeUse& use) {
    VkBuffer buffer{};
    u64 offset = 0;
    if (!AllocShadow(use.size, use.offset, buffer, offset)) {
        return false;
    }
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                            .count = static_cast<u32>(use.size),
                            .hash_a = GpuProf::SUB_SHADOW}
             : GpuProf::Tag{}};
    // Earlier commands may have written these bytes.
    const std::array<BufferAccess, 2> accesses{{
        {use.pool->Handle(), use.offset, use.offset + use.size, false},
        {buffer, offset, offset + use.size, true},
    }};
    host->HostBeginBufferOp(accesses);
    const VkBufferCopy region{.srcOffset = use.offset, .dstOffset = offset, .size = use.size};
    host->HostCmd().CopyBuffer(use.pool->Handle(), buffer, region);
    EndShadowCopy();
    shadows.push_back({use.pool, use.offset, use.offset + use.size, buffer, offset});
    ++counters.shadow_creates;
    counters.shadow_bytes_in += use.size;
    if (Debug::Get().shadow_log && shadow_logged.size() < MAX_LOGGED_SHADOWS &&
        shadow_logged.emplace(use.pool->address, use.offset, use.size).second) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage shadow: pool {:#x} ({:#x} bytes, flags {:#x}) range {:#x}+{:#x} "
                 "for {} vs={:016x} fs={:016x} cs={:016x}",
                 use.pool->address, use.pool->size, use.pool->flags, use.offset, use.size,
                 plan.graphics ? "a draw" : "a dispatch", ProfStageHash(0), ProfStageHash(4),
                 ProfStageHash(STAGE_COMPUTE));
    }
    return true;
}

void DrawContext::OnCpuWrite(Pool& pool, u64 offset, u64 size) {
    if (host == nullptr || size == 0 || offset >= pool.size) {
        return;
    }
    const u64 end = offset + (std::min)(size, pool.size - offset);
    bool overlaps = false;
    for (const Shadow& shadow : shadows) {
        overlaps |= shadow.pool == &pool && shadow.begin < end && offset < shadow.end;
    }
    if (!overlaps) {
        return;
    }

    EndPass();
    host->HostFlushBarrier();
    for (const Shadow& shadow : shadows) {
        if (shadow.pool != &pool || shadow.begin >= end || offset >= shadow.end) {
            continue;
        }
        const u64 begin = (std::max)(shadow.begin, offset);
        const u64 finish = (std::min)(shadow.end, end);
        // vkCmdCopyBuffer requires 4-byte aligned offsets and size. Storage-shadow ranges are
        // word-aligned, so include the complete words touched by the CPU write.
        const u64 copy_begin = begin & ~u64{3};
        const u64 copy_end = (finish + 3) & ~u64{3};
        if (copy_begin < shadow.begin || copy_end > shadow.end) {
            continue;
        }
        const u64 delta = copy_begin - shadow.begin;
        const u64 bytes = copy_end - copy_begin;
        const VkBufferCopy result{.srcOffset = copy_begin,
                                   .dstOffset = shadow.offset + delta,
                                   .size = bytes};
        host->HostCmd().CopyBuffer(pool.Handle(), shadow.buffer, result);
        if (shadow.mirror) {
            const VkBufferCopy snapshot{.srcOffset = copy_begin,
                                        .dstOffset = shadow.snapshot_offset + delta,
                                        .size = bytes};
            host->HostCmd().CopyBuffer(pool.Handle(), shadow.snapshot, snapshot);
        }
    }
    MarkShadowCopy();
}

void DrawContext::FlushShadow(size_t index, std::atomic<u64> DrawCounters::*cause) {
    const Shadow shadow = shadows[index];
    shadows.erase(shadows.begin() + static_cast<std::ptrdiff_t>(index));
    if (shadow.mirror) {
        WriteBackMirrorEntries({&shadow, 1}, cause);
        return;
    }
    const u64 size = shadow.end - shadow.begin;
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                            .count = static_cast<u32>(size),
                            .hash_a = GpuProf::SUB_SHADOW}
             : GpuProf::Tag{}};
    // The shader writes of the passes and dispatches that used the shadow.
    const std::array<BufferAccess, 2> accesses{{
        {shadow.buffer, shadow.offset, shadow.offset + size, false},
        {shadow.pool->Handle(), shadow.begin, shadow.begin + size, true},
    }};
    host->HostBeginBufferOp(accesses);
    const VkBufferCopy region{.srcOffset = shadow.offset, .dstOffset = shadow.begin, .size = size};
    host->HostCmd().CopyBuffer(shadow.buffer, shadow.pool->Handle(), region);
    // The guest sees the bytes through the host barrier at the end of the command buffer.
    host->HostNotePoolWrite(*shadow.pool, shadow.begin, size);
    EndShadowCopy();
    (counters.*cause)++;
    counters.shadow_bytes_out += size;
}

void DrawContext::MarkShadowCopy() {
    if (shadow_batch_depth != 0) {
        shadow_batch_copied = true;
    } else {
        host->HostMarkWritten();
    }
}

void DrawContext::EndShadowCopy() {
    if ((host->HostBarrierMerge() & BARRIER_MERGE_BUFFERS) != 0) {
        host->HostEndBufferOp();
    } else {
        MarkShadowCopy();
    }
}

void DrawContext::WriteBackMirrorEntries(std::span<const Shadow> entries,
                                         std::atomic<u64> DrawCounters::*cause) {
    if (entries.empty()) {
        return;
    }
    u64 bytes = 0;
    for (const Shadow& entry : entries) {
        bytes += entry.end - entry.begin;
    }
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                            .count = static_cast<u32>((std::min<u64>)(bytes, ~u32{0})),
                            .hash_a = GpuProf::SUB_SHADOW}
             : GpuProf::Tag{}};
    // The shader writes of the passes and dispatches that used the entries. Entries cover disjoint
    // pool ranges, so the write-back of one needs no barrier before the write-back of another that
    // the loop of a conflict flush records after it (EDEN_NX2_TRY_BARRIER_MERGE): the merge reads
    // the entry's mirror and snapshot bytes and rewrites the pool's bytes of its range; line mode
    // stores whole units around a changed word, so every range is widened by the unit reach.
    thread_local std::vector<BufferAccess> accesses;
    accesses.clear();
    constexpr u64 REACH = 256;
    for (const Shadow& entry : entries) {
        const u64 length = entry.end - entry.begin;
        const auto widen = [&](VkBuffer buffer, u64 begin, u64 end, bool write) {
            accesses.push_back({buffer, begin > REACH ? begin - REACH : 0, end + REACH, write});
        };
        widen(entry.buffer, entry.begin, entry.end, false);
        widen(entry.snapshot, entry.snapshot_offset, entry.snapshot_offset + length, false);
        widen(entry.pool->Handle(), entry.begin, entry.end, true);
    }
    host->HostBeginBufferOp(accesses);
    WriteBackMerge* const merge = impl.WriteBackMerger();
    if (merge != nullptr) {
        thread_local std::vector<MergeRange> ranges;
        ranges.clear();
        VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire);
        for (const Shadow& entry : entries) {
            PoolMirror& mirror = *entry.pool->vram_mirror;
            const u32 stat_slot = mirror.monitor ? mirror.monitor->slot : u32{0xFFFFFFFFu};
            const auto add_range = [&](u64 begin, u64 end, bool lines) {
                ranges.push_back({.result = entry.buffer,
                                  .result_off = begin,
                                  .snapshot = entry.snapshot,
                                  .snapshot_off = entry.snapshot_offset + (begin - entry.begin),
                                  .guest = entry.pool->Handle(),
                                  .guest_off = begin,
                                  .size = end - begin,
                                  .stat_slot = stat_slot,
                                  .lines = lines});
            };
            if (mirror.line_guard == nullptr || memory == nullptr) {
                add_range(entry.begin, entry.end, false);
                continue;
            }
            // Line runs only on pages with an access trap and no touch since (nvn_line_guard.h).
            MirrorLineGuard::Recording plan = mirror.line_guard->Record(entry.begin, entry.end);
            if (plan.disarm_all) {
                memory->DisarmAccessTrap(mirror.track_id, entry.pool->address, entry.pool->size);
            }
            for (const auto& [first, last] : plan.arm) {
                memory->ArmAccessTrap(mirror.track_id, entry.pool->address + first * MirrorPages::PAGE,
                                      (last - first + 1) * MirrorPages::PAGE);
                if (mirror.monitor) {
                    mirror.monitor->armed_pages.fetch_add(last - first + 1,
                                                          std::memory_order_relaxed);
                    mirror.monitor->armed_runs.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (plan.token) {
                // The GPU may store into these pages until the submission completes.
                host->HostFlight().keepalive.push_back(std::move(plan.token));
            }
            for (const MirrorLineGuard::Run& run : plan.runs) {
                add_range(run.begin, run.end, run.lines);
            }
        }
        merge->Record(host->HostCmd(), ranges);
    } else {
        // No merge shader: the whole ranges go back (the storage shadows' behaviour).
        for (const Shadow& entry : entries) {
            const VkBufferCopy region{
                .srcOffset = entry.begin, .dstOffset = entry.begin, .size = entry.end - entry.begin};
            host->HostCmd().CopyBuffer(entry.buffer, entry.pool->Handle(), region);
        }
        impl.stats.vram_write_backs_plain.fetch_add(entries.size(), std::memory_order_relaxed);
    }
    for (const Shadow& entry : entries) {
        // The guest bytes now hold what the mirror holds: its pages stay valid.
        host->HostNoteMirrorWriteBack(*entry.pool, entry.begin, entry.end - entry.begin);
        (counters.*cause)++;
        counters.shadow_bytes_out += entry.end - entry.begin;
        if (const std::shared_ptr<MirrorMonitor>& monitor = entry.pool->vram_mirror->monitor;
            monitor) {
            const u64 length = entry.end - entry.begin;
            monitor->entries.fetch_add(1, std::memory_order_relaxed);
            monitor->entry_bytes.fetch_add(length, std::memory_order_relaxed);
            const bool whole_pool = entry.begin == 0 && entry.end >= entry.pool->size;
            if (whole_pool) {
                monitor->whole_entries.fetch_add(1, std::memory_order_relaxed);
            } else if (length <= MONITOR_MAX_ENTRY_BYTES && !entry.pool->vram_mirror->line_guard) {
                if (VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire)) {
                    memory->ArmAccessTrap(entry.pool->vram_mirror->track_id,
                                          entry.pool->address + entry.begin, length);
                    monitor->armed_pages.fetch_add(
                        (length + MirrorPages::PAGE - 1) / MirrorPages::PAGE,
                        std::memory_order_relaxed);
                    monitor->armed_runs.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    host->HostEndBufferOp();
}

void DrawContext::FlushShadows() {
    if (shadows.empty()) {
        return;
    }
    if (pass_open) {
        EndPass(BreakCause::ShadowAlias);
    }
    // Mirror entries go back together (one barrier, one merge recording).
    thread_local std::vector<Shadow> entries;
    entries.clear();
    for (const Shadow& shadow : shadows) {
        if (shadow.mirror) {
            entries.push_back(shadow);
        }
    }
    std::erase_if(shadows, [](const Shadow& shadow) { return shadow.mirror; });
    WriteBackMirrorEntries(entries, &DrawCounters::shadow_flush_segment);
    const ShadowCopyBatch batch{*this};
    while (!shadows.empty()) {
        FlushShadow(0, &DrawCounters::shadow_flush_segment);
    }
}

void DrawContext::FlushShadowsOverlapping(const Pool& pool, u64 offset, u64 size) {
    if (pool.backing != Pool::Backing::Imported) {
        return;
    }
    const u64 end = offset + size;
    const ShadowCopyBatch batch{*this};
    for (size_t i = shadows.size(); i-- > 0;) {
        if (shadows[i].pool != &pool || !Overlaps(offset, end, shadows[i].begin, shadows[i].end)) {
            continue;
        }
        if (pass_open) {
            EndPass(BreakCause::ShadowAlias);
        }
        FlushShadow(i, &DrawCounters::shadow_flush_conflict);
    }
}

bool DrawContext::HasOverlappingShadow(const Pool& pool, u64 begin, u64 end) const noexcept {
    for (const Shadow& shadow : shadows) {
        if (shadow.pool == &pool && Overlaps(begin, end, shadow.begin, shadow.end)) {
            return true;
        }
    }
    return false;
}

void DrawContext::NoteExtraRead(const BufferRef& ref, u64 size) {
    if (ref.pool != nullptr && ref.pool->backing == Pool::Backing::Imported &&
        (Debug::Get().shadow_storage || impl.pool_mirror_mode != 0)) {
        extra_uses.push_back({ref.pool, ref.pool_offset, size, false});
    }
}

void DrawContext::ResolveIndirectSource(BufferRef& ref, u64 size) {
    if (ref.pool == nullptr || ref.pool->backing != Pool::Backing::Imported) {
        return;
    }
    if (ref.pool->vram_mirror) {
        // GPU-driven arguments (a culling dispatch writes them) are read where they were written:
        // inside a live entry, or wherever the mirror holds the current bytes and no entry of
        // this command buffer overlaps the range.
        const u64 end = ref.pool_offset + size;
        bool serve = FindShadow(*ref.pool, ref.pool_offset, end) != shadows.size();
        if (!serve && impl.MirrorCurrent(*ref.pool, ref.pool_offset, size)) {
            serve = std::ranges::none_of(shadows, [&](const Shadow& entry) {
                return !entry.mirror && entry.pool == ref.pool &&
                       Overlaps(ref.pool_offset, end, entry.begin, entry.end);
            });
        }
        if (serve) {
            ref.buffer = *ref.pool->vram_mirror->buffer;
            ref.offset = ref.pool_offset;
            impl.stats.vram_bindings.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    // Read through the guest memory: the live shadows over it go back first.
    if (!shadows.empty()) {
        FlushShadowsOverlapping(*ref.pool, ref.pool_offset, (std::max<u64>)(size, 1));
    }
    NoteExtraRead(ref, size);
}

} // namespace Vulkan::Nvn
