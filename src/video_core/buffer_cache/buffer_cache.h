// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/buffer_cache/uniform_cache_history.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/game_presets.h"
#include "common/range_sets.inc"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/multi_range_refresh.h"
#include "video_core/buffer_cache/nx2_bytes_equal.h"
#include "video_core/buffer_cache/storage_buffer_size.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/page_filter.h"
#include "video_core/texture_cache/memory_limits.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

/// EDEN_NX2_BINDING_STATS=1: count how native compute storage bindings are bounded and log a line
/// every 10 s. Always counting cost a clock read and three relaxed atomics on every native compute
/// binding, and a title binds them tens of thousands of times a second (Yoshi's demo in the 3.2
/// report: 371,040 in one 10 s window), so the counters wait for the switch.
inline bool NativeBindingStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BINDING_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Counts one native compute storage binding: whether its size word reached past the 8 MiB cap,
/// and whether the binding stopped at a mapping gap instead of its size.
inline void ReportNativeBindingStats(u32 size, u32 wanted) {
    static std::atomic<u64> computed{}, past_cap{}, at_gap{}, logged{};
    static std::atomic<u32> largest{};
    static std::atomic<s64> next_log{};
    const u64 total = computed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (size > 8_MiB) {
        past_cap.fetch_add(1, std::memory_order_relaxed);
        u32 seen = largest.load(std::memory_order_relaxed);
        while (size > seen && !largest.compare_exchange_weak(seen, size, std::memory_order_relaxed)) {
        }
    }
    if (size < wanted) {
        at_gap.fetch_add(1, std::memory_order_relaxed);
    }
    const s64 now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    const u64 long_bindings = past_cap.load(std::memory_order_relaxed);
    const u64 gap_bindings = at_gap.load(std::memory_order_relaxed);
    s64 due = next_log.load(std::memory_order_relaxed);
    if (now >= due && logged.load(std::memory_order_relaxed) != long_bindings + gap_bindings &&
        next_log.compare_exchange_strong(due, now + 10)) {
        logged.store(long_bindings + gap_bindings, std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "Native compute storage bindings: {} reach past 8 MiB through their size word "
                 "(largest {:#x} bytes), {} stop at a mapping gap, of {} bound",
                 long_bindings, largest.load(std::memory_order_relaxed), gap_bindings, total);
    }
}

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef VESHIRU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const u64 device_local_u64 = runtime.GetDeviceLocalMemory();
    const auto native_gc = TextureCacheMemory::ResolveNativeBufferGcThresholds(device_local_u64);
    native_gc_expected_bytes = native_gc.expected_bytes;
    native_gc_critical_bytes = native_gc.critical_bytes;
    native_gc_pressure_floor_bytes = native_gc.pressure_floor_bytes;
    native_gc_tight_vram = native_gc.tight_vram;

    const s64 device_local_memory = static_cast<s64>(device_local_u64);
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                   DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                   DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() {
    VideoCore::g_read_downloads.fetch_sub(static_cast<u32>(read_downloads.size()),
                                          std::memory_order_relaxed);
}

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    const StutterTrace::SiteScope site{StutterTrace::Site::BufferCollect};
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            if (!aggressive_gc) {
                return false;
            }
            // Its download would land nowhere (a Switch 2 sparse range without memory) and the
            // buffer made again would read zeros: it stays.
            if (!device_memory.IsBacked(buffer.CpuAddr(), buffer.SizeBytes())) {
                if (create_trace.enabled) {
                    ++create_trace.kept_unbacked;
                }
                return false;
            }
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        if (create_trace.enabled) {
            ++create_trace.collected;
            create_trace.collected_bytes += slot_buffers[buffer_id].SizeBytes();
        }
        DeleteBuffer(buffer_id);
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    VideoCommon::g_hot_page_epoch.fetch_add(1, std::memory_order_relaxed);
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    frame_created_buffers.clear();
    static const bool mem_trace = std::getenv("EDEN_NX2_MEM_TRACE") != nullptr;
    if (static std::chrono::steady_clock::time_point last{}; mem_trace) {
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{2}) {
            last = now;
            LOG_INFO(HW_GPU, "Memory trace: buffer cache {} MiB", buffer_bytes >> 20);
        }
    }
    if (CreateTrace& trace = create_trace; trace.enabled) {
        if (const auto now = std::chrono::steady_clock::now(); now - trace.last >= std::chrono::seconds{10}) {
            trace.last = now;
            LOG_INFO(HW_GPU,
                     "Buffer trace: {} MiB in buffers; in 10 s {} created ({} MiB), {} joined into "
                     "them ({} MiB copied), {} stream leaps, {} collected ({} MiB), {} GPU-written "
                     "kept for unbacked device memory",
                     buffer_bytes >> 20, trace.creates, trace.created_bytes >> 20, trace.joins,
                     trace.joined_bytes >> 20, trace.leaps, trace.collected,
                     trace.collected_bytes >> 20, trace.kept_unbacked);
            trace.creates = trace.created_bytes = trace.joins = trace.joined_bytes = trace.leaps =
                trace.collected = trace.collected_bytes = trace.kept_unbacked = 0;
        }
    }
    if (const GeometryStats& stats = geometry_stats; frame_tick % 600 == 0 && stats.syncs != 0) {
        if (Nx2Profile::Enabled() || Nx2Perf::Verify()) {
            LOG_INFO(HW_GPU,
                     "Geometry upload scans in 600 frames: {} of {} skipped, {} verify mismatches",
                     stats.syncs_skipped, stats.syncs, stats.verify_mismatches);
        }
        geometry_stats = {};
    }
    if (frame_tick % 600 == 0) {
        UniformKeep& keep = uniform_keep;
        if (Nx2Profile::Enabled() || ubo_keep_verify) {
            LOG_INFO(HW_GPU,
                     "Uniform keeps in 600 frames: {} kept, {} streamed, {} verified, {} verify "
                     "mismatches",
                     keep.kept, keep.streamed_binds, keep.verified, keep.mismatches);
        }
        keep.kept = keep.streamed_binds = keep.verified = keep.mismatches = 0;
    }
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    if (static u64 frames{}; ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU, "Buffer cache: {} buffers {} MiB; largest {}", count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                           ", "));
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    AdvanceUniformCacheHistory(channel_state->uniform_cache_hits);
    AdvanceUniformCacheHistory(channel_state->uniform_cache_shots);

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget).
        // Under device pressure on 4 GB cards, run only non-aggressive collection to shed stale
        // clean buffers without stalling the GPU to download live GPU-written working sets; only
        // escalate to aggressive eviction when buffer memory itself reaches its critical cap.
        const bool device_pressure =
            native_gc_tight_vram && total_used_memory >= critical_memory &&
            buffer_bytes >= native_gc_pressure_floor_bytes;
        if (buffer_bytes >= native_gc_expected_bytes || device_pressure) {
            RunGarbageCollector(buffer_bytes >= native_gc_critical_bytes);
        }
    } else if (total_used_memory >= minimum_memory) {
        RunGarbageCollector(total_used_memory >= critical_memory);
    }
    ++frame_tick;
    if (frame_tick % 60 == 0 && Nx2Perf::On(Nx2Perf::Opt::PreflushAging)) {
        Nx2AgePreflushBlocks();
    }
    delayed_destruction_ring.Tick();

    // Pages draws wrote: their early downloads, now that the frame's work is recorded.
    RecordEarlyDownloads();
    early_bytes_this_frame = 0;
    // Read downloads no thread waited for (the other ranges the guest read before, early ones).
    if constexpr (!IS_OPENGL) {
        if (!read_downloads.empty()) {
            (void)runtime.IsFreeNow(read_downloads.front().tick);
        }
    }
    WriteCompletedReadDownloads();
    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data,
                                 bool engine_write) {
    Nx2NoteListHostWrite(device_addr, size);
    NoteUniformWrite(device_addr, size);
    // EDEN_NX2_READ_CHECK: a newer value where an early download landed, as ClearDownload
    // forgets one over GPU-modified bytes: without early downloads the write had taken the bytes
    // out of the GPU-modified ones, and the read's synchronous flush leaves them. Landed bytes
    // are no longer GPU-modified, so the write reaches no ClearDownload: written back to the
    // landed value after an upload of another, they read as GPU bytes the download missed.
    if (!early_landed.empty()) {
        ForgetEarlyLanded(device_addr, size);
    }
    // The written range lies within its pages: when they hold no GPU-written data, both lookups
    // below find nothing, so one lookup answers them.
    if (keep_page_gpu_data) {
        const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
        if (!gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
            if (!Nx2Perf::Verify()) {
                if (engine_write) {
                    // Untracking the pages and tracking them again at the upload changed their
                    // host protection twice (two VirtualProtect calls each, ~4% of the GPU
                    // thread at Ocarina of Time's water) for writes no guest CPU access saw.
                    memory_tracker.MarkRegionAsWrittenTracked(device_addr, size);
                } else {
                    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
                }
                MarkVertexBuffersStale(device_addr, size);
                return;
            }
            static std::atomic<u32> logged{};
            if (IsRegionGpuModified(device_addr, size) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: GPU-written data in {:#x}+{:#x} outside its pages",
                          device_addr, size);
            }
        }
    }
    if (IsRegionGpuModified(device_addr, size)) {
        ClearReplacedDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    // Only the scheduler's owner may download (it records and submits): unmaps and other writes
    // reach this from CPU and service threads too, and submitting there raced the GPU thread's
    // own submissions (crashes in Scheduler::SubmitExecution). Those keep the old behaviour;
    // guest CPU writes come through TryCacheInvalidation, which sends such pages to the GPU thread.
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data &&
        VideoCommon::GpuSplit::OwnsScheduler() &&
        gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        // Switch 2 titles: upload the CPU's bytes into the buffer instead, as CachedWriteMemory
        // does, so the page stays GPU-owned and nothing waits for the host GPU.
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_INLINE_PAGE_WRITES");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        const bool inline_writes =
            forced >= 0 ? forced == 1 : Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
        if (inline_writes && IsRegionRegistered(device_addr, size)) {
            tmp_buffer.resize_destructive(size);
            device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);
            InlineMemoryImplementation(device_addr, size, tmp_buffer);
            return;
        }
        DownloadMemory(page_begin, page_end - page_begin);
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    last_synced_count = 0;
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    NoteUniformWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    // Preflushable data is downloaded at fences, but only where it is marked (QueueFenceDownload):
    // the rest of the range still needs this flush.
    if (memory_tracker.IsRegionPreflushable(device_addr, size) &&
        !HasGpuWritesNotDownloadedAtFence(device_addr_start_aligned,
                                          device_addr_end_aligned - device_addr_start_aligned)) {
        // Fence downloads keep it, once the first read's download is in guest memory.
        area->in_flight = std::ranges::any_of(read_downloads, [&](const ReadDownload& download) {
            return download.write_ranges.Intersects(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        });
        area->preemtive = !area->in_flight;
        // The core keeps this area: a later GPU write to its unmarked pages must arm them again.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
        return area;
    };
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    NoteReadBack(device_addr_start_aligned, device_addr_end_aligned, !area->preemtive);
    // EDEN_NX2_READ_CHECK: a read an early download serves takes a flush, which compares
    // what it finds with the GPU's bytes now.
    const bool check_early =
        ReadChecks() && EarlyDownloadServes(device_addr_start_aligned,
                                            device_addr_end_aligned - device_addr_start_aligned);
    if (!area->preemtive && !check_early && !read_downloads.empty() &&
        ReadDownloadsCover(device_addr_start_aligned,
                           device_addr_end_aligned - device_addr_start_aligned)) {
        // All of it on its way, for an earlier read or right behind its GPU write: no new copies.
        // The core took the trap, as below.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
        if (!WriteFinishedReadDownloads(device_addr_start_aligned,
                                        device_addr_end_aligned - device_addr_start_aligned)) {
            area->in_flight = true;
            return area;
        }
        // The host GPU had run them: in guest memory now, nothing is left to wait for.
        area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                               device_addr_end_aligned - device_addr_start_aligned);
        if (area->preemtive) {
            return area;
        }
    }
    if (check_early) {
        area->preemtive = false;
    }
    // A preflushable mark persists and downloads later GPU writes at every fence. Native titles
    // instead keep read traps and flush on demand.
    const bool preflush = !Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
    if (!area->preemtive) {
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Marked(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned, preflush);
        }
        if (preflush) {
            memory_tracker.MarkRegionAsPreflushable(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        } else {
            if constexpr (USE_MEMORY_MAPS) {
                cpu_read_ranges.Add(area->start_address, area->end_address - area->start_address);
                download_cpu_read_ranges = true;
            }
            // The core consumed this trap before GetFlushArea. Forget it so the next GPU write
            // really rearms the core instead of being skipped by the deduplication cache.
            read_trapped_ranges.Subtract(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        }
    } else {
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    if constexpr (USE_MEMORY_MAPS) {
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
            cpu_read_ranges.ForEach([&](DAddr lower, DAddr upper) {
                gpu_modified_ranges.ForEachInRange(lower, upper - lower,
                                                   [&](DAddr start, DAddr end) {
                                                       ranges.emplace_back(start, end - start);
                                                   });
            });
            DownloadRangesMemory(ranges);
            return;
        }
    }
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        const DAddr start = (std::max)(device_addr, buffer.CpuAddr());
        const DAddr end = (std::min)(device_addr + size, buffer.CpuAddr() + buffer.SizeBytes());
        if (start < end) {
            DownloadBufferMemory(buffer, start, end - start);
        }
    });
}

template <class P>
void BufferCache<P>::DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges) {
    constexpr bool has_mapped = requires(Buffer& buffer) { buffer.Mapped(); };
    if constexpr (!USE_MEMORY_MAPS) {
        for (const auto& [device_addr, size] : ranges) {
            DownloadMemory(device_addr, size);
        }
    } else {
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Download, 16> staged;
        boost::container::small_vector<Download, 16> mapped;
        u64 total_size_bytes = 0;
        for (const auto& [device_addr, size] : ranges) {
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(device_addr, buffer_addr);
                const DAddr end = (std::min)(device_addr + size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                bool is_mapped = false;
                if constexpr (has_mapped) {
                    is_mapped = !buffer.Mapped().empty();
                }
                memory_tracker.ForEachDownloadRangeAndClear(
                    begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            device_addr_out, range_size, [&](DAddr start, DAddr range_end) {
                                const BufferCopy copy{
                                    .src_offset = start - buffer_addr,
                                    .dst_offset = total_size_bytes,
                                    .size = range_end - start,
                                };
                                if (is_mapped) {
                                    mapped.push_back({buffer_id, copy});
                                    return;
                                }
                                staged.push_back({buffer_id, copy});
                                total_size_bytes += Common::AlignUp(range_end - start, 64);
                            });
                        ClearReplacedDownload(device_addr_out, range_size);
                        gpu_modified_ranges.Subtract(device_addr_out, range_size);
                    });
            });
        }
        if (staged.empty() && mapped.empty()) {
            return;
        }
        // Once the host GPU ran the copies: the CPU writes collected meanwhile keep their bytes
        // too.
        const auto land = [&](const Async_Buffer* staging) {
            DAddr landing_begin = ~DAddr{0};
            DAddr landing_end = 0;
            const auto cover = [&](const Download& download) {
                const DAddr address =
                    slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
                landing_begin = (std::min)(landing_begin, address);
                landing_end = (std::max)(landing_end, address + download.copy.size);
            };
            std::ranges::for_each(staged, cover);
            std::ranges::for_each(mapped, cover);
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            if (staging != nullptr) {
                for (const Download& download : staged) {
                    WriteDownloadedMemory(
                        cpu_written,
                        slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset,
                        staging->mapped_span.data() + (download.copy.dst_offset - staging->offset),
                        download.copy.size);
                }
            }
            if constexpr (has_mapped) {
                for (const Download& download : mapped) {
                    Buffer& buffer = slot_buffers[download.buffer_id];
                    WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + download.copy.src_offset,
                                          buffer.Mapped().data() + download.copy.src_offset,
                                          download.copy.size);
                }
            }
        };
        if (staged.empty()) {
            if constexpr (has_mapped) {
                runtime.FinishForHostRead();
            }
            land(nullptr);
        } else {
            auto staging = runtime.DownloadStagingBuffer(total_size_bytes);
            for (Download& download : staged) {
                download.copy.dst_offset += staging.offset;
                Buffer& buffer = slot_buffers[download.buffer_id];
                buffer.MarkUsage(download.copy.src_offset, download.copy.size);
                runtime.CopyBuffer(staging.buffer, buffer, std::span{&download.copy, 1}, true);
            }
            if constexpr (has_mapped) {
                runtime.FinishForHostRead();
            } else {
                runtime.Finish();
            }
            land(&staging);
        }
    }
}

template <class P>
void BufferCache<P>::DownloadMemoryForRead(DAddr device_addr, u64 size) {
    if constexpr (IS_OPENGL || !USE_MEMORY_MAPS) {
        DownloadMemory(device_addr, size);
    } else {
        WriteCompletedReadDownloads();
        // Early downloads still to record are of older writes: they go first.
        RecordEarlyDownloads();
        // The ranges DownloadMemory takes: prefetch nearby previously-read GPU-written pages first,
        // bounded to MaxSpeculativeReadBytes so one small read does not block on huge ring sweeps.
        boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            u64 speculative_bytes = 0;
            const DAddr own_end = device_addr + size;
            const auto add_piece = [&](DAddr start, DAddr end) {
                if (start >= end) {
                    return;
                }
                if (speculative_bytes >= MaxSpeculativeReadBytes) {
                    download_cpu_read_ranges = true;
                    return;
                }
                const u64 remaining = MaxSpeculativeReadBytes - speculative_bytes;
                const u64 length = end - start;
                if (length > remaining) {
                    ranges.emplace_back(start, remaining);
                    speculative_bytes = MaxSpeculativeReadBytes;
                    download_cpu_read_ranges = true;
                    return;
                }
                ranges.emplace_back(start, length);
                speculative_bytes += length;
            };
            const auto add_speculative = [&](DAddr start, DAddr end) {
                if (start < device_addr) {
                    add_piece(start, (std::min)(end, device_addr));
                }
                if (end > own_end) {
                    add_piece((std::max)(start, own_end), end);
                }
            };
            constexpr u64 BackWindow = MaxSpeculativeReadBytes / 4;
            constexpr u64 AheadWindow = MaxSpeculativeReadBytes - BackWindow;
            const DAddr window_begin = device_addr > BackWindow ? device_addr - BackWindow : 0;
            const DAddr window_end =
                own_end <= ~DAddr{0} - AheadWindow ? own_end + AheadWindow : ~DAddr{0};
            const auto visit_read_range = [&](DAddr lower, DAddr upper) {
                if (download_cpu_read_ranges) {
                    return;
                }
                gpu_modified_ranges.ForEachInRange(lower, upper - lower, add_speculative);
            };
            cpu_read_ranges.ForEachInRange(window_begin, window_end - window_begin,
                                           visit_read_range);
            if (window_end < ~DAddr{0} && !download_cpu_read_ranges) {
                cpu_read_ranges.ForEachInRange(window_end, ~DAddr{0} - window_end,
                                               visit_read_range);
            }
            if (window_begin > 0 && !download_cpu_read_ranges) {
                cpu_read_ranges.ForEachInRange(0, window_begin, visit_read_range);
            }
        }
        const std::array own{std::pair<DAddr, u64>{device_addr, size}};
        RecordReadDownload(ranges, own, false);
    }
}

template <class P>
void BufferCache<P>::RecordReadDownload(std::span<const std::pair<DAddr, u64>> ranges,
                                        std::span<const std::pair<DAddr, u64>> own, bool early) {
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        struct Copy {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Copy, 16> copies;
        ReadDownload download;
        download.early = early;
        u64 total_size_bytes = 0;
        const auto add_copy = [&](BufferId buffer_id, DAddr buffer_addr, DAddr begin, DAddr end) {
            copies.push_back({buffer_id, BufferCopy{
                                             .src_offset = begin - buffer_addr,
                                             .dst_offset = total_size_bytes,
                                             .size = end - begin,
                                         }});
            total_size_bytes += Common::AlignUp(end - begin, 64);
            download.write_ranges.Add(begin, end - begin);
            download.clean_ranges.Add(begin, end - begin);
            read_download_ranges.Add(begin, end - begin);
        };
        // EDEN_NX2_READ_CHECK: what a synchronous flush (DownloadRangesMemory) copies for
        // the read itself.
        std::unique_ptr<ReadCheck> check;
        boost::container::small_vector<Copy, 16> check_copies;
        u64 check_size = 0;
        const u64 check_every = read_check.every != 0 ? read_check.every : ReadCheckEvery();
        if (check_every != 0 && read_check.next++ % check_every == 0) {
            check = std::make_unique<ReadCheck>();
            check->read_begin = ~DAddr{0};
            for (const auto& [own_addr, own_size] : own) {
                check->read_begin = (std::min)(check->read_begin, own_addr);
                check->read_end = (std::max)(check->read_end, own_addr + own_size);
            }
            check->early = early;
            for (const auto& [own_addr, own_size] : own) {
                ForEachBufferInRange(own_addr, own_size, [&](BufferId buffer_id, Buffer& buffer) {
                    const DAddr buffer_addr = buffer.CpuAddr();
                    const DAddr begin = (std::max)(own_addr, buffer_addr);
                    const DAddr end =
                        (std::min)(own_addr + own_size, buffer_addr + buffer.SizeBytes());
                    if (begin >= end) {
                        return;
                    }
                    memory_tracker.ForEachDownloadRange(
                        begin, end - begin, false, [&](u64 run_addr, u64 run_size) {
                            gpu_modified_ranges.ForEachInRange(
                                run_addr, run_size, [&](DAddr start, DAddr range_end) {
                                    check_copies.push_back(
                                        {buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = check_size,
                                                        .size = range_end - start,
                                                    }});
                                    check_size += Common::AlignUp(range_end - start, 64);
                                });
                        });
                });
            }
        }
        boost::container::small_vector<std::pair<DAddr, DAddr>, 8> gaps;
        for (const auto& [range_addr, range_size] : ranges) {
            ForEachBufferInRange(range_addr, range_size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(range_addr, buffer_addr);
                const DAddr end =
                    (std::min)(range_addr + range_size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                // The bytes DownloadRangesMemory takes, but they stay GPU-modified until they
                // are in guest memory (WriteReadDownloads).
                memory_tracker.ForEachDownloadRange(
                    begin, end - begin, false, [&](u64 run_addr, u64 run_size) {
                        gpu_modified_ranges.ForEachInRange(
                            run_addr, run_size, [&](DAddr start, DAddr range_end) {
                                // Without the bytes a read download already brings (an earlier
                                // one, or this one: the ranges can overlap).
                                gaps.clear();
                                DAddr next = start;
                                read_download_ranges.ForEachInRange(
                                    start, range_end - start, [&](DAddr lower, DAddr upper) {
                                        if (lower > next) {
                                            gaps.emplace_back(next, lower);
                                        }
                                        next = (std::max)(next, upper);
                                    });
                                if (next < range_end) {
                                    gaps.emplace_back(next, range_end);
                                }
                                for (const auto& [gap_begin, gap_end] : gaps) {
                                    add_copy(buffer_id, buffer_addr, gap_begin, gap_end);
                                }
                            });
                        // The downloads taken at fences are older: they must not land over these.
                        ClearFenceDownloads(run_addr, run_size);
                    });
            });
        }
        if (check) {
            // Each byte a synchronous flush copies is in this download or an earlier one (kept).
            for (const Copy& entry : check_copies) {
                const DAddr start = slot_buffers[entry.buffer_id].CpuAddr() + entry.copy.src_offset;
                Common::RangeSet<DAddr> rest;
                rest.Add(start, entry.copy.size);
                const auto take_out = [&](const Common::RangeSet<DAddr>& taken, bool earlier_one) {
                    taken.ForEachInRange(start, entry.copy.size, [&](DAddr lower, DAddr upper) {
                        if (earlier_one) {
                            rest.ForEachInRange(lower, upper - lower, [&](DAddr from, DAddr to) {
                                check->kept.Add(from, to - from);
                            });
                        }
                        rest.Subtract(lower, upper - lower);
                    });
                };
                take_out(download.write_ranges, false);
                for (const ReadDownload& earlier : read_downloads) {
                    take_out(earlier.write_ranges, true);
                }
                rest.ForEach([&](DAddr lower, DAddr upper) {
                    if (read_check.logged++ < 16) {
                        LOG_ERROR(HW_GPU,
                                  "Read download check: {:#x}+{:#x} of the {} [{:#x}, {:#x}) is "
                                  "GPU-modified but no download brings it",
                                  lower, upper - lower, early ? "early download" : "read",
                                  own.front().first, own.back().first + own.back().second);
                    }
                    read_check.uncovered += upper - lower;
                });
            }
        }
        // EDEN_NX2_READ_CHECK: what early downloads brought to the read's pages before it
        // came. No longer GPU-modified, they are compared as they landed.
        struct LandedCopy {
            u64 offset; ///< In the check's staging, past its own offset
            std::vector<u8> as_landed;
        };
        boost::container::small_vector<LandedCopy, 4> landed_copies;
        if (check && !early && !early_landed.empty()) {
            const auto add_landed = [&](DAddr lower, DAddr upper, const u8* as_landed) {
                ForEachBufferInRange(lower, upper - lower, [&](BufferId buffer_id, Buffer& buffer) {
                    const DAddr buffer_addr = buffer.CpuAddr();
                    const DAddr begin = (std::max)(lower, buffer_addr);
                    const DAddr end = (std::min)(upper, buffer_addr + buffer.SizeBytes());
                    if (begin >= end) {
                        return;
                    }
                    const u8* const bytes = as_landed + (begin - lower);
                    landed_copies.push_back(
                        {check_size, std::vector<u8>(bytes, bytes + (end - begin))});
                    check_copies.push_back({buffer_id, BufferCopy{
                                                           .src_offset = begin - buffer_addr,
                                                           .dst_offset = check_size,
                                                           .size = end - begin,
                                                       }});
                    check_size += Common::AlignUp(end - begin, 64);
                    check->kept.Add(begin, end - begin);
                    check->landed.Add(begin, end - begin);
                });
            };
            for (const auto& [own_addr, own_size] : own) {
                TakeEarlyLanded(own_addr, own_size, add_landed);
            }
        }
        // EDEN_NX2_READ_CHECK: a check whose read's bytes other downloads bring is a download of its own.
        const bool check_only = !early && copies.empty() && check && !check_copies.empty();
        if (copies.empty() && !check_only) {
            return;
        }
        if (!copies.empty()) {
            auto staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
            boost::container::small_vector<BufferCopy, 16> batch;
            for (size_t i = 0; i < copies.size();) {
                const BufferId batch_buffer_id = copies[i].buffer_id;
                Buffer& buffer = slot_buffers[batch_buffer_id];
                batch.clear();
                while (i < copies.size() && copies[i].buffer_id == batch_buffer_id) {
                    BufferCopy& copy = copies[i].copy;
                    copy.dst_offset += staging.offset;
                    buffer.MarkUsage(copy.src_offset, copy.size);
                    if (!batch.empty() &&
                        batch.back().src_offset + batch.back().size == copy.src_offset &&
                        batch.back().dst_offset + batch.back().size == copy.dst_offset) {
                        batch.back().size += copy.size;
                        download.copies.back().size += copy.size;
                    } else {
                        batch.push_back(copy);
                        download.copies.push_back(BufferCopy{
                            .src_offset = buffer.CpuAddr() + copy.src_offset,
                            .dst_offset = copy.dst_offset,
                            .size = copy.size,
                        });
                    }
                    ++i;
                }
                // Mapped buffers too: the GPU thread's later work may write them before the copy
                // is read.
                runtime.CopyBuffer(staging.buffer, buffer, batch, true);
            }
            download.staging = staging;
        }
        if (check && !check_copies.empty()) {
            // The same bytes again, from the same point of the GPU's work: what a synchronous
            // flush would have written.
            check->staging = runtime.DownloadStagingBuffer(check_size, true);
            for (Copy& entry : check_copies) {
                Buffer& buffer = slot_buffers[entry.buffer_id];
                entry.copy.dst_offset += check->staging.offset;
                buffer.MarkUsage(entry.copy.src_offset, entry.copy.size);
                runtime.CopyBuffer(check->staging.buffer, buffer, std::span{&entry.copy, 1}, true);
                check->copies.push_back(BufferCopy{
                    .src_offset = buffer.CpuAddr() + entry.copy.src_offset,
                    .dst_offset = entry.copy.dst_offset,
                    .size = entry.copy.size,
                });
                if (check_only) {
                    check->own.Add(buffer.CpuAddr() + entry.copy.src_offset, entry.copy.size);
                }
            }
            if (!check->kept.Empty()) {
                check->landed_bytes.resize(check_size);
            }
            for (const LandedCopy& landed : landed_copies) {
                std::ranges::copy(landed.as_landed, check->landed_bytes.data() + landed.offset);
            }
            ++(early ? read_check.early_recorded
                     : (check_only ? read_check.alone_recorded : read_check.reads_recorded));
            download.check = std::move(check);
        }
        // A read waits for its copies; an early download's go with the next submission.
        download.tick = early ? runtime.QueueForHostRead() : runtime.SubmitForHostRead();
        if (early) {
            ++pending_early_downloads;
            early_bytes_this_frame += total_size_bytes;
        }
        read_downloads.push_back(std::move(download));
        VideoCore::g_read_downloads.fetch_add(1, std::memory_order_release);
    }
}

template <class P>
std::optional<u64> BufferCache<P>::FinishReadDownloads(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        std::unique_lock lock{mutex};
        // The newest one over the range: those before it are written first.
        std::optional<u64> tick;
        for (const ReadDownload& download : read_downloads) {
            // EDEN_NX2_READ_CHECK: a check alone is waited for too, as its read's download.
            if (download.write_ranges.Intersects(device_addr, size) ||
                (download.check && download.check->own.Intersects(device_addr, size))) {
                tick = download.tick;
            }
        }
        if (!tick) {
            return std::nullopt;
        }
        u64 waited = 0;
        if (!runtime.IsFree(*tick)) {
            lock.unlock();
            const u64 wait_start = VideoCore::AsyncReadback::Now();
            runtime.WaitForHostRead(*tick);
            waited = (std::max)(VideoCore::AsyncReadback::Now() - wait_start, u64{1});
            VideoCore::AsyncReadback::CountHostWait(waited);
            lock.lock();
        }
        WriteReadDownloads(*tick, VideoCore::AsyncReadback::t_reading
                                      ? VideoCore::AsyncReadback::Landing::Reader
                                      : VideoCore::AsyncReadback::Landing::GpuSideWrite);
        return waited;
    }
    return std::nullopt;
}

template <class P>
void BufferCache<P>::OutdateReadDownloads(DAddr device_addr, u64 size) {
    // EDEN_NX2_READ_CHECK: the GPU's copy gets newer bytes where an early download landed,
    // which no read compares. A query result copy (SyncValues) is no GPU write the cache records:
    // a synchronous flush doesn't download it, and guest memory gets the result once the query
    // cache writes it there. MarkWrittenBuffer's bytes are GPU-modified from now on, which
    // TakeEarlyLanded leaves out, and every way out of that forgets them too.
    if (!early_landed.empty()) {
        ForgetEarlyLanded(device_addr, size);
    }
    // The bytes stay GPU-modified when the downloads land, so a later read downloads them.
    if (!read_downloads.empty()) {
        for (ReadDownload& download : read_downloads) {
            download.clean_ranges.Subtract(device_addr, size);
        }
        read_download_ranges.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::WriteFinishedReadDownloads(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        std::optional<u64> tick;
        for (const ReadDownload& download : read_downloads) {
            if (download.write_ranges.Intersects(device_addr, size)) {
                tick = download.tick;
            }
        }
        if (!tick || !runtime.IsFreeNow(*tick)) {
            return false;
        }
        WriteReadDownloads(*tick);
        return true;
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::NoteWorkRecorded(bool dispatch) {
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        if (dispatch) {
            // Outside any render pass: the copies go right behind the dispatch.
            RecordEarlyDownloads();
        }
        // Early downloads the host GPU ran go to guest memory, where their reads then find them.
        // Asking the host GPU is a call: after each dispatch, and every few draws.
        if (pending_early_downloads != 0 && (dispatch || ++early_landing_checks % 32 == 0) &&
            runtime.IsFreeNow(read_downloads.front().tick)) {
            WriteCompletedReadDownloads();
        }
    }
}

template <class P>
void BufferCache<P>::NoteReadBack(DAddr begin, DAddr end, bool gpu_modified) {
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        if (!gpu_modified && read_back_pages.empty()) {
            return;
        }
        for (DAddr page = begin; page < end; page += DEVICE_PAGESIZE) {
            auto it = std::ranges::lower_bound(read_back_pages, page, {}, &ReadBackPage::page);
            if (it == read_back_pages.end() || it->page != page) {
                // Learned from reads that find GPU writes.
                if (!gpu_modified) {
                    continue;
                }
                if (read_back_pages.size() >= MaxReadBackPages) {
                    // The page whose writes were read longest ago makes room.
                    read_back_pages.erase(
                        std::ranges::min_element(read_back_pages, {}, &ReadBackPage::last_read));
                    it = std::ranges::lower_bound(read_back_pages, page, {}, &ReadBackPage::page);
                }
                it = read_back_pages.insert(it, ReadBackPage{.page = page, .written = true});
                read_back_first.store(read_back_pages.front().page, std::memory_order_release);
                read_back_last.store(read_back_pages.back().page, std::memory_order_release);
            }
            if (!it->written) {
                // Nothing new since its last read.
                continue;
            }
            it->written = false;
            it->unread_writes = 0;
            it->reads = (std::min)(static_cast<u16>(it->reads + 1), EarlyAfterReads);
            it->last_read = ++read_back_reads;
            const bool proven_ring = it != read_back_pages.begin() &&
                                     it->reads >= EarlyAfterReads &&
                                     std::prev(it)->page + DEVICE_PAGESIZE == page &&
                                     std::prev(it)->reads >= EarlyAfterReads;
            if (proven_ring && page + 2 * DEVICE_PAGESIZE >= end) {
                // Once two adjacent pages are established as recurring GPU-to-CPU readbacks,
                // pre-arm up to two unseen frontier pages ahead of the cursor so Switch 2 cyclic
                // compute/query readback rings schedule early downloads on first write instead of
                // stalling the guest CPU on every new slot.
                const u64 seed_read = it->last_read;
                for (u32 step = 1; step <= 2; ++step) {
                    const DAddr next_page = page + step * DEVICE_PAGESIZE;
                    if (next_page < page || read_back_pages.size() >= MaxReadBackPages) {
                        break;
                    }
                    auto next_it =
                        std::ranges::lower_bound(read_back_pages, next_page, {}, &ReadBackPage::page);
                    if (next_it == read_back_pages.end() || next_it->page != next_page) {
                        read_back_pages.insert(
                            next_it, ReadBackPage{
                                         .page = next_page,
                                         .last_read = seed_read,
                                         .reads = EarlyAfterReads,
                                     });
                    }
                }
                read_back_first.store(read_back_pages.front().page, std::memory_order_release);
                read_back_last.store(read_back_pages.back().page, std::memory_order_release);
            }
        }
    }
}

template <class P>
bool BufferCache<P>::NoteReportPending(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        const DAddr page = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        if (std::ranges::binary_search(read_back_pages, page, {}, &ReadBackPage::page)) {
            pending_reports.push_back({device_addr, size, false});
            return true;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::FinishReport(DAddr device_addr, u64 size) {
    const auto it = std::ranges::find_if(pending_reports, [&](const PendingReport& report) {
        return report.addr == device_addr && report.size == size;
    });
    if (it != pending_reports.end()) {
        pending_reports.erase(it);
    }
}

template <class P>
void BufferCache<P>::NoteReadBackWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        for (PendingReport& report : pending_reports) {
            if (report.addr < device_addr + size && device_addr < report.addr + report.size) {
                report.rewritten = true;
            }
        }
        if (read_back_pages.empty()) {
            return;
        }
        const DAddr end = device_addr + size;
        for (auto it = std::ranges::lower_bound(read_back_pages,
                                                Common::AlignDown(device_addr, DEVICE_PAGESIZE), {},
                                                &ReadBackPage::page);
             it != read_back_pages.end() && it->page < end; ++it) {
            it->written = true;
            if (it->unread_writes < MaxUnreadWrites) {
                ++it->unread_writes;
            } else {
                // The guest no longer reads it after its writes.
                it->reads = 0;
            }
            if (it->reads >= EarlyAfterReads && !it->queued) {
                it->queued = true;
                early_pages.push_back(it->page);
            }
        }
    }
}

template <class P>
void BufferCache<P>::RecordEarlyDownloads() {
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        if (early_pages.empty()) {
            return;
        }
        // Sort and coalesce adjacent 4 KiB pages into single contiguous download spans so multi-MiB
        // compute ring writes become a handful of copies instead of thousands.
        std::ranges::sort(early_pages);
        boost::container::small_vector<std::pair<DAddr, u64>, 16> pages;
        auto page_it = read_back_pages.begin();
        for (const DAddr page : early_pages) {
            page_it = std::ranges::lower_bound(page_it, read_back_pages.end(), page, {},
                                               &ReadBackPage::page);
            if (page_it != read_back_pages.end() && page_it->page == page) {
                page_it->queued = false;
            }
            if (!pages.empty()) {
                auto& [run_addr, run_size] = pages.back();
                const DAddr run_end = run_addr + run_size;
                if (page < run_end) {
                    continue;
                }
                if (page == run_end) {
                    run_size += DEVICE_PAGESIZE;
                    continue;
                }
            }
            pages.emplace_back(page, DEVICE_PAGESIZE);
        }
        early_pages.clear();
        // Bounded: past it, the reads of these pages flush on demand.
        if (pending_early_downloads >= MaxPendingEarlyDownloads ||
            early_bytes_this_frame >= EarlyBytesPerFrame) {
            return;
        }
        RecordReadDownload(pages, pages, true);
    }
}

template <class P>
bool BufferCache<P>::EarlyDownloadServes(DAddr device_addr, u64 size) const {
    if (std::ranges::any_of(read_downloads, [&](const ReadDownload& download) {
            return download.early && download.write_ranges.Intersects(device_addr, size);
        })) {
        return true;
    }
    for (DAddr page = Common::AlignDown(device_addr, DEVICE_PAGESIZE); page < device_addr + size;
         page += DEVICE_PAGESIZE) {
        const auto it = early_landed.find(page);
        if (it != early_landed.end() && it->second.bytes.Intersects(device_addr, size)) {
            return true;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::NoteEarlyLanded(const ReadDownload& download,
                                     const Common::RangeSet<DAddr>& cpu_written) {
    // Pages no read came for since keep theirs until a download lands there. Bounded: past it,
    // their reads are not compared.
    if (early_landed.size() >= 4 * MaxReadBackPages) {
        early_landed.clear();
    }
    download.clean_ranges.ForEach([&](DAddr start, DAddr end) {
        Common::RangeSet<DAddr> landed;
        landed.Add(start, end - start);
        cpu_written.ForEachInRange(start, end - start, [&](DAddr lower, DAddr upper) {
            landed.Subtract(lower, upper - lower);
        });
        landed.ForEach([&](DAddr lower, DAddr upper) {
            for (DAddr page = Common::AlignDown(lower, DEVICE_PAGESIZE); page < upper;
                 page += DEVICE_PAGESIZE) {
                const DAddr from = (std::max)(lower, page);
                const DAddr to = (std::min)(upper, page + DEVICE_PAGESIZE);
                EarlyLanded& entry = early_landed[page];
                entry.as_landed.resize(DEVICE_PAGESIZE);
                device_memory.ReadBlockUnsafe(from, entry.as_landed.data() + (from - page),
                                              to - from);
                entry.bytes.Add(from, to - from);
            }
        });
    });
}

template <class P>
template <typename Func>
void BufferCache<P>::TakeEarlyLanded(DAddr device_addr, u64 size, Func&& func) {
    for (DAddr page = Common::AlignDown(device_addr, DEVICE_PAGESIZE); page < device_addr + size;
         page += DEVICE_PAGESIZE) {
        const auto it = early_landed.find(page);
        if (it == early_landed.end()) {
            continue;
        }
        Common::RangeSet<DAddr> taken;
        it->second.bytes.ForEachInRange(
            device_addr, size, [&](DAddr lower, DAddr upper) { taken.Add(lower, upper - lower); });
        // A newer GPU write's bytes are the read's own again.
        gpu_modified_ranges.ForEachInRange(page, DEVICE_PAGESIZE, [&](DAddr lower, DAddr upper) {
            taken.Subtract(lower, upper - lower);
        });
        taken.ForEach([&](DAddr lower, DAddr upper) {
            func(lower, upper, it->second.as_landed.data() + (lower - page));
            it->second.bytes.Subtract(lower, upper - lower);
        });
        if (it->second.bytes.Empty()) {
            early_landed.erase(it);
        }
    }
}

template <class P>
void BufferCache<P>::ForgetRewrittenEarlyLanded(DAddr device_addr, u64 size) {
    std::array<u8, DEVICE_PAGESIZE> guest;
    for (DAddr page = Common::AlignDown(device_addr, DEVICE_PAGESIZE); page < device_addr + size;
         page += DEVICE_PAGESIZE) {
        const auto it = early_landed.find(page);
        if (it == early_landed.end()) {
            continue;
        }
        EarlyLanded& entry = it->second;
        Common::RangeSet<DAddr> rewritten;
        entry.bytes.ForEachInRange(device_addr, size, [&](DAddr lower, DAddr upper) {
            const u64 length = upper - lower;
            device_memory.ReadBlockUnsafe(lower, guest.data(), length);
            const u8* const landed = entry.as_landed.data() + (lower - page);
            for (u64 i = 0; i < length;) {
                if (guest[i] == landed[i]) {
                    ++i;
                    continue;
                }
                u64 end = i + 1;
                while (end < length && guest[end] != landed[end]) {
                    ++end;
                }
                rewritten.Add(lower + i, end - i);
                i = end;
            }
        });
        rewritten.ForEach(
            [&](DAddr lower, DAddr upper) { entry.bytes.Subtract(lower, upper - lower); });
        if (entry.bytes.Empty()) {
            early_landed.erase(it);
        }
    }
}

template <class P>
void BufferCache<P>::ForgetEarlyLanded(DAddr device_addr, u64 size) {
    for (DAddr page = Common::AlignDown(device_addr, DEVICE_PAGESIZE); page < device_addr + size;
         page += DEVICE_PAGESIZE) {
        const auto it = early_landed.find(page);
        if (it == early_landed.end()) {
            continue;
        }
        it->second.bytes.Subtract(device_addr, size);
        if (it->second.bytes.Empty()) {
            early_landed.erase(it);
        }
    }
}

template <class P>
void BufferCache<P>::WriteReadDownloads(u64 tick, VideoCore::AsyncReadback::Landing landed_by) {
    if constexpr (!IS_OPENGL) {
        while (!read_downloads.empty() && read_downloads.front().tick <= tick) {
            ReadDownload& download = read_downloads.front();
            // Guest memory this writes, and what its read check reads. The guest CPU writes
            // collected from now on store their bytes once this download is in guest memory.
            DAddr landing_begin = ~DAddr{0};
            DAddr landing_end = 0;
            const auto cover = [&](const BufferCopy& copy) {
                landing_begin = (std::min)(landing_begin, static_cast<DAddr>(copy.src_offset));
                landing_end =
                    (std::max)(landing_end, static_cast<DAddr>(copy.src_offset + copy.size));
            };
            std::ranges::for_each(download.copies, cover);
            if (download.check) {
                std::ranges::for_each(download.check->copies, cover);
            }
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            // CPU writes are collected before they reach guest memory, and stay pending until
            // applied, which takes them out of write_ranges (ClaimCpuWrites): either way they
            // keep their bytes.
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            const u8* const base = download.staging.mapped_span.data();
            for (const BufferCopy& copy : download.copies) {
                const DAddr copy_addr = static_cast<DAddr>(copy.src_offset);
                const u8* const data = base + (copy.dst_offset - download.staging.offset);
                download.write_ranges.ForEachInRange(
                    copy_addr, copy.size, [&](DAddr start, DAddr end) {
                        WriteDownloadedMemory(cpu_written, start, data + (start - copy_addr),
                                              end - start);
                    });
            }
            // EDEN_NX2_READ_CHECK: what early downloads brought, for the reads they serve.
            if (download.early && ReadChecks()) {
                NoteEarlyLanded(download, cpu_written);
            }
            // EDEN_NX2_READ_CHECK: what later reads' checks find of an earlier download where they kept its bytes.
            for (size_t later = 1; later < read_downloads.size(); ++later) {
                if (read_downloads[later].check) {
                    NoteLandedBytes(*read_downloads[later].check, download.write_ranges,
                                    cpu_written);
                }
            }
            if (download.check) {
                CheckReadDownload(*download.check, cpu_written, landed_by);
                async_buffers_death_ring.push_back(download.check->staging);
            }
            // A query result older than these bytes lands over them later: they stay
            // GPU-modified, for a later read to download again.
            for (const PendingReport& report : pending_reports) {
                if (report.rewritten) {
                    download.clean_ranges.Subtract(report.addr, report.size);
                    read_download_ranges.Subtract(report.addr, report.size);
                }
            }
            download.clean_ranges.ForEach([&](DAddr start, DAddr end) {
                gpu_modified_ranges.Subtract(start, end - start);
                read_download_ranges.Subtract(start, end - start);
            });
            // As a synchronous download leaves them: pages without GPU-modified bytes left are
            // clean in the tracker too.
            download.clean_ranges.ForEach([&](DAddr start, DAddr end) {
                const DAddr last = Common::AlignUp(end, DEVICE_PAGESIZE);
                for (DAddr page = Common::AlignDown(start, DEVICE_PAGESIZE); page < last;
                     page += DEVICE_PAGESIZE) {
                    if (!gpu_modified_ranges.Intersects(page, DEVICE_PAGESIZE)) {
                        memory_tracker.UnmarkRegionAsGpuModified(page, DEVICE_PAGESIZE);
                    }
                }
            });
            // Freed by the GPU thread (TickFrame). A read check alone has none.
            if (!download.copies.empty()) {
                async_buffers_death_ring.push_back(download.staging);
            }
            if (download.early) {
                --pending_early_downloads;
            }
            read_downloads.pop_front();
            VideoCore::g_read_downloads.fetch_sub(1, std::memory_order_release);
        }
    }
}

template <class P>
void BufferCache<P>::WriteCompletedReadDownloads() {
    if constexpr (!IS_OPENGL) {
        while (!read_downloads.empty() && runtime.IsFree(read_downloads.front().tick)) {
            WriteReadDownloads(read_downloads.front().tick);
        }
    }
}

template <class P>
bool BufferCache<P>::ReadDownloadsCover(DAddr device_addr, u64 size) const {
    bool covered = true;
    gpu_modified_ranges.ForEachInRange(device_addr, size, [&](DAddr start, DAddr end) {
        DAddr next = start;
        read_download_ranges.ForEachInRange(start, end - start, [&](DAddr lower, DAddr upper) {
            covered = covered && lower <= next;
            next = (std::max)(next, upper);
        });
        covered = covered && next >= end;
    });
    return covered;
}

template <class P>
void BufferCache<P>::CheckReadDownload(const ReadCheck& check,
                                       const Common::RangeSet<DAddr>& cpu_written,
                                       VideoCore::AsyncReadback::Landing landing) {
    // Guest memory now holds this download and the earlier ones: compare it with the synchronous
    // copy, except where a newer value replaced it since (CPU writes, ClearDownload). Writes
    // collected since the download was written count as newer too. Where an earlier download
    // brought the bytes (kept), the synchronous flush of its own read left them, and guest memory
    // keeps what the GPU's side writes there after that, as without these downloads: those bytes
    // are compared as they landed (NoteLandedBytes).
    const Common::RangeSet<DAddr> cpu_written_since = PendingCpuWriteRanges();
    const u8* const base = check.staging.mapped_span.data();
    enum class Source : u8 {
        Own,    ///< This download brings it
        Kept,   ///< An earlier one was to bring it, and hasn't
        Landed, ///< An earlier one brought it
    };
    enum class Result : u8 {
        Same,
        Rewritten, ///< Newer bytes since an earlier download landed it
        Differs,   ///< This download's bytes differ
        Outdated,  ///< An earlier download's bytes differ: the GPU wrote them since
        NotLanded, ///< No earlier download wrote it
    };
    const auto report = [this, &check, landing](Result result, DAddr address, u64 size, u8 found,
                                                u8 wanted) {
        if (result == Result::Rewritten) {
            read_check.rewritten += size;
            if (read_check.logged_rewrites++ < 16) {
                LOG_INFO(HW_GPU,
                         "Read download check: guest memory {:#x}+{:#x} got newer bytes after an "
                         "earlier read's download brought it, as it does after synchronous "
                         "flushes: not compared (written by {})",
                         address, size, VideoCore::ReadCheckWrites::Find(address, size));
            }
            return;
        }
        read_check.mismatched += size;
        read_check.mismatched_by_landing[static_cast<size_t>(landing)] += size;
        if (read_check.logged++ >= 16) {
            return;
        }
        const char* const why =
            result == Result::Outdated
                ? ": an earlier read's download brought it, and the GPU wrote it since without "
                  "taking it out of that download"
                : (result == Result::NotLanded ? ": no earlier read's download wrote it" : "");
        LOG_ERROR(HW_GPU,
                  "Read download check: guest memory {:#x}+{:#x} differs from a synchronous "
                  "flush ({:#04x} for {:#04x} at its start){}; {} download of [{:#x}, {:#x}), "
                  "written to guest memory by {}",
                  address, size, found, wanted, why,
                  check.early ? "early" : (check.own.Empty() ? "read" : "check-only"),
                  check.read_begin, check.read_end,
                  VideoCore::AsyncReadback::landing_names[static_cast<size_t>(landing)]);
    };
    std::vector<u8> guest;
    std::vector<Source> sources;
    for (const BufferCopy& copy : check.copies) {
        const DAddr copy_addr = static_cast<DAddr>(copy.src_offset);
        const u64 copy_offset = copy.dst_offset - check.staging.offset;
        const u8* const expected = base + copy_offset;
        const u8* const landed =
            check.landed_bytes.empty() ? nullptr : check.landed_bytes.data() + copy_offset;
        Common::RangeSet<DAddr> compared;
        compared.Add(copy_addr, copy.size);
        const auto take_out = [&](const Common::RangeSet<DAddr>& taken) {
            taken.ForEachInRange(copy_addr, copy.size, [&](DAddr lower, DAddr upper) {
                compared.Subtract(lower, upper - lower);
            });
        };
        take_out(check.changed);
        take_out(cpu_written);
        take_out(cpu_written_since);
        compared.ForEach([&](DAddr start, DAddr end) {
            const u64 length = end - start;
            guest.resize(length);
            device_memory.ReadBlockUnsafe(start, guest.data(), length);
            read_check.compared += length;
            const u64 offset = start - copy_addr;
            sources.assign(length, Source::Own);
            const auto mark = [&](const Common::RangeSet<DAddr>& ranges, Source source) {
                ranges.ForEachInRange(start, length, [&](DAddr lower, DAddr upper) {
                    std::fill(sources.begin() + static_cast<std::ptrdiff_t>(lower - start),
                              sources.begin() + static_cast<std::ptrdiff_t>(upper - start), source);
                });
            };
            mark(check.kept, Source::Kept);
            mark(check.landed, Source::Landed);
            const auto result_at = [&](u64 i) {
                const u8 wanted = expected[offset + i];
                switch (sources[i]) {
                case Source::Own:
                    return guest[i] == wanted ? Result::Same : Result::Differs;
                case Source::Kept:
                    return guest[i] == wanted ? Result::Same : Result::NotLanded;
                case Source::Landed:
                    if (guest[i] != landed[offset + i]) {
                        return Result::Rewritten;
                    }
                    return guest[i] == wanted ? Result::Same : Result::Outdated;
                }
                return Result::Same;
            };
            for (u64 i = 0; i < length;) {
                const Result result = result_at(i);
                if (result == Result::Same) {
                    ++i;
                    continue;
                }
                u64 run_end = i + 1;
                while (run_end < length && result_at(run_end) == result) {
                    ++run_end;
                }
                report(result, start + i, run_end - i, guest[i], expected[offset + i]);
                i = run_end;
            }
        });
    }
    ++read_check.checked;
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    if (now >= read_check.last_log + 10) {
        read_check.last_log = now;
        LOG_INFO(HW_GPU,
                 "Read download check (EDEN_NX2_READ_CHECK): {} downloads compared "
                 "with a synchronous flush, {} KiB; {} bytes differ (landed by the reading "
                 "thread {}, by a GPU-side write {}, after the host GPU ran them {}), {} "
                 "GPU-modified bytes no download brought (both must stay 0); {} bytes got newer "
                 "values after an earlier read's download brought them (not compared)",
                 read_check.checked, read_check.compared / 1024, read_check.mismatched,
                 read_check.mismatched_by_landing[0], read_check.mismatched_by_landing[1],
                 read_check.mismatched_by_landing[2], read_check.uncovered, read_check.rewritten);
    }
}

template <class P>
void BufferCache<P>::NoteLandedBytes(ReadCheck& check, const Common::RangeSet<DAddr>& written,
                                     const Common::RangeSet<DAddr>& cpu_written) {
    if (check.kept.Empty()) {
        return;
    }
    for (const BufferCopy& copy : check.copies) {
        const DAddr copy_addr = static_cast<DAddr>(copy.src_offset);
        u8* const landed = check.landed_bytes.data() + (copy.dst_offset - check.staging.offset);
        written.ForEachInRange(copy_addr, copy.size, [&](DAddr lower, DAddr upper) {
            check.kept.ForEachInRange(lower, upper - lower, [&](DAddr start, DAddr end) {
                device_memory.ReadBlockUnsafe(start, landed + (start - copy_addr), end - start);
                check.landed.Add(start, end - start);
                // A guest write collected and not yet applied kept its bytes from the download:
                // newer than both copies, as a synchronous flush leaves it. Applied before this
                // check lands, it no longer counts as collected, and would read as landed.
                cpu_written.ForEachInRange(start, end - start, [&](DAddr from, DAddr to) {
                    check.changed.Add(from, to - from);
                });
            });
        });
    }
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    if (const std::optional<u64> waited = FinishReadDownloads(device_addr, size)) {
        VideoCore::AsyncReadback::CountClearLanding(*waited);
    }
    ClearReplacedDownload(device_addr, size);
}

template <class P>
void BufferCache<P>::ClearReplacedDownload(DAddr device_addr, u64 size) {
    ClearFenceDownloads(device_addr, size);
    // EDEN_NX2_READ_CHECK: a newer value there, as without early downloads.
    if (!early_landed.empty()) {
        ForgetEarlyLanded(device_addr, size);
    }
    if (!read_downloads.empty()) {
        for (ReadDownload& download : read_downloads) {
            download.write_ranges.Subtract(device_addr, size);
            download.clean_ranges.Subtract(device_addr, size);
            if (download.check) {
                download.check->changed.Add(device_addr, size);
            }
        }
        read_download_ranges.Subtract(device_addr, size);
    }
}

template <class P>
void BufferCache<P>::ClearFenceDownloads(DAddr device_addr, u64 size) {
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(
    GPUVAddr src_address, GPUVAddr dest_address, u64 amount,
    boost::container::small_vector<std::pair<DAddr, size_t>, 4>* gpu_written_dest) {
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);
    NoteUniformWrite(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            Tegra::Nx2MarkGpuWritten(new_base_address, size);
        }
        QueueFenceDownload(new_base_address, size);
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        NoteReadBackWrite(pair.first, pair.second);
    }
    // The copy rewrites all of the destination, not only the bytes that came from GPU-written
    // source data: a command list page's copy taken before it no longer holds the page.
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Nx2NoteListWrite(buffer_b, *cpu_dest_address, amount);
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (gpu_written_dest) {
        *gpu_written_dest = tmp_intervals;
    }
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync =
            Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, amount);
    if (!dest_dirty) {
        return false;
    }

    const size_t size = amount * sizeof(u32);
    ClearDownload(*cpu_dst_address, size);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);
    NoteUniformWrite(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    // As DMACopy: the CPU's bytes on these pages reach the buffer first. A command list page is
    // copied from the buffer whole after the clear (Nx2CopyListPages).
    SynchronizeBuffer(dest_buffer, *cpu_dst_address, static_cast<u32>(size));
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Nx2NoteListWrite(buffer, *cpu_dst_address, size);
    }
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        buffer.MarkUsage(buffer.Offset(device_addr), size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
u64 BufferCache<P>::KeptGraphicsBindingsHash(bool vertex, bool storage, bool texture) const {
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&](u64 value) { hash = (hash ^ value) * 0x100000001b3ULL; };
    const auto mix_binding = [&](const Binding& binding) {
        mix(binding.device_addr);
        mix(binding.size);
    };
    if (vertex) {
        for (const Binding& binding : channel_state->vertex_buffers) {
            mix_binding(binding);
        }
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (storage) {
            mix(channel_state->enabled_storage_buffers[stage]);
            mix(channel_state->written_storage_buffers[stage]);
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                mix_binding(channel_state->storage_buffers[stage][index]);
            });
        }
        if (texture) {
            mix(channel_state->enabled_texture_buffers[stage]);
            ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
                const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
                mix_binding(binding);
                mix(static_cast<u64>(binding.format));
            });
        }
    }
    return hash;
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    ++native_indirect_uniform_draw_ticket;
    last_synced_count = 0;
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    if (stage == 0) {
        ++native_indirect_uniform_draw_ticket;
        last_synced_count = 0;
    }
    if (channel_state->enabled_uniform_buffer_masks[stage] != 0) {
        BindHostGraphicsUniformBuffers(stage);
    }
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        BindHostGraphicsStorageBuffers(stage);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        BindHostGraphicsTextureBuffers(stage);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    Nx2TraceComputeBufferInputs();
    ++native_indirect_uniform_draw_ticket;
    last_synced_count = 0;
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers();
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers();
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::Nx2TraceBufferInput(DAddr addr, u64 size,
                                        BufferDependencyTrace::Input input) {
    if (!nx2_buffer_deps || addr == 0 || size == 0) {
        return;
    }
    const bool cpu_dirty = memory_tracker.IsRegionCpuModified(addr, size);
    u64 dirty_bytes = 0;
    gpu_modified_ranges.ForEachInRange(addr, size, [&](DAddr begin, DAddr end) {
        dirty_bytes += end - begin;
        nx2_buffer_deps->ReadDirty(begin, end - begin, input, frame_tick, cpu_dirty);
    });
    nx2_buffer_deps->ReadClean(size - dirty_bytes, input, cpu_dirty);
}

template <class P>
void BufferCache<P>::BeginGraphicsUniforms(const Nx2FastDraw::Delta* delta) {
    UniformKeep& keep = uniform_keep;
    const u64 sync_serial = Nx2FastDraw::SyncSerial();
    const u64 host_serial = gpu_memory != nullptr ? gpu_memory->HostPointerSerial() : 0;
    const bool known = delta != nullptr && !delta->no_uniform_keeps && !keep.memory_writes_lost &&
                       keep.draw_deletions == buffer_deletions &&
                       keep.sync_serial == sync_serial && keep.host_serial == host_serial &&
                       keep.gpu_memory == gpu_memory;
    keep.draw_deletions = buffer_deletions;
    keep.gpu_memory = gpu_memory;
    keep.sync_serial = sync_serial;
    keep.host_serial = host_serial;
    if (!known) {
        keep.clean = {};
    }
    for (size_t stage = 0; stage < NUM_STAGES && known; ++stage) {
        u32 clean = keep.clean[stage] & ~delta->binds[stage];
        for (u32 mask = clean; mask != 0; mask &= mask - 1) {
            const u32 index = static_cast<u32>(std::countr_zero(mask));
            const auto [begin, size] = keep.ranges[stage][index];
            const auto overlaps = [begin, end = begin + size](DAddr write_begin, DAddr write_end) {
                return write_begin < end && begin < write_end;
            };
            bool written = std::ranges::any_of(keep.memory_writes, [&](const auto& write) {
                return overlaps(write.first, write.second);
            });
            // The device memory inline constant data landed on (writes to unmapped addresses
            // land nowhere).
            for (u32 write = 0; write < delta->num_device_writes && !written; ++write) {
                written = overlaps(delta->device_begin[write], delta->device_end[write]);
            }
            if (written) {
                clean &= ~(1u << index);
            }
        }
        keep.clean[stage] = clean;
    }
    keep.memory_writes.clear();
    keep.memory_writes_lost = false;
}

template <class P>
bool BufferCache<P>::UniformRangeKept(size_t stage, u32 index, DAddr device_addr,
                                      u32 size) const noexcept {
    const UniformKeep& keep = uniform_keep;
    return ((keep.clean[stage] >> index) & 1) != 0 &&
           keep.ranges[stage][index] == std::pair<DAddr, u32>{device_addr, size};
}

template <class P>
bool BufferCache<P>::ConstantDataReachesBuffers(DAddr device_addr, u64 size) const {
    if (memory_tracker.IsRegionEverSynchronized(device_addr, size) ||
        Tegra::Nx2MayBeGpuWritten(device_addr, size)) {
        return true;
    }
    // Guest memory newer than a command list copy of the page (Nx2NoteListHostWrite).
    if (Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
        for (DAddr page = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
             page < device_addr + size; page += DEVICE_PAGESIZE) {
            if (Tegra::Nx2IsCommandListPage(page)) {
                return true;
            }
        }
    }
    // Repacked vertex buffers read guest memory again only when marked stale.
    if constexpr (requires { runtime.AnyVertexRepacked(); }) {
        if (runtime.AnyVertexRepacked()) {
            return true;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::Nx2TraceComputeBufferInputs() {
    if (!nx2_buffer_deps) {
        return;
    }
    using Input = BufferDependencyTrace::Input;
    ForEachBoundBuffer(true, [&](u32, bool storage, u32, bool, const Binding& binding, u32 size,
                                const Buffer*, u32) {
        const Input input = storage ? Input::Storage : Input::Uniform;
        if (storage && binding.segment_count >= 2 &&
            binding.segment_first + binding.segment_count <= compute_segments.size()) {
            for (u32 i = 0; i < binding.segment_count; ++i) {
                const auto& segment = compute_segments[binding.segment_first + i];
                Nx2TraceBufferInput(segment.device_addr, segment.size, input);
            }
        } else {
            Nx2TraceBufferInput(binding.device_addr, size, input);
        }
    });
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const auto& binding = channel_state->compute_texture_buffers[index];
        Nx2TraceBufferInput(binding.device_addr, binding.size, Input::Texture);
    });
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written,
                                               u32 indirect_offset) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        LOG_ERROR(HW_GPU, "Invalid graphics storage buffer binding: stage {}, index {}",
                  stage, ssbo_index);
        return false;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU, "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= u64{1} << ssbo_index;
    channel_state->written_storage_buffers[stage] |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
    GPUVAddr holder{};
    if (indirect_offset != ~0U) {
        // The pointer is an entry of a table the constant buffer points to.
        holder = gpu_memory->Read<u64>(ssbo_addr);
        if (holder == 0) {
            channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;
            return false;
        }
        ssbo_addr = holder + indirect_offset;
    }
    // NVN graphics descriptors carry a size; bare native pointers and tables do not.
    const bool native_pointer =
        (maxwell3d->GetNativeMacroEngine().native_turing &&
         !IsNvnGraphicsStorageDescriptor(stage, cbuf_index, cbuf_offset, indirect_offset)) ||
        indirect_offset != ~0U;
    channel_state->storage_buffers[stage][ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer,
                             indirect_offset != ~0U);
    if (indirect_offset != ~0U) {
        TraceTablePointer(false, stage, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          channel_state->storage_buffers[stage][ssbo_index]);
    }
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 indirect_offset,
                                              const Shader::StorageWriteExtent& write_extent) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU, "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    GPUVAddr holder{};
    const bool table_pointer = indirect_offset != ~0U;
    if (table_pointer) {
        holder = gpu_memory->Read<u64>(ssbo_addr);
        ssbo_addr = holder + indirect_offset;
    }
    // Mario Kart World and Nintendo Switch Sports pass NVN {address, size} descriptors in compute
    // constant buffers; treating them as bare pointers inflates small SSBOs to 8 MiB and truncates
    // written SSBOs larger than 8 MiB (rendering models black).
    const u64 program_id = Settings::GetCurrentProgramID();
    const bool pointer_only = program_id != Settings::MarioKartWorldProgramId &&
                              program_id != Settings::SwitchSportsProgramId;
    const bool native_pointer = kepler_compute->IsNativeMode() && pointer_only;
    channel_state->native_compute_storage_buffers =
        (channel_state->native_compute_storage_buffers & ~(u64{1} << ssbo_index)) |
        (u64{native_pointer ? 1U : 0U} << ssbo_index);
    const Binding binding =
        table_pointer && holder == 0
            ? NULL_BINDING
            : StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer,
                                   table_pointer, true);
    channel_state->compute_storage_buffers[ssbo_index] = binding;
    if (table_pointer && holder != 0) {
        TraceTablePointer(true, 0, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          binding);
    }

    // How much of the binding the dispatch marks GPU-written. A native pointer has no size, so the
    // binding is up to 8 MiB past it, and marking all of it defers the CPU's later writes to those
    // pages until the next fence and has fences copy the over-marked range back.
    u64 mark = binding.size;
    if (native_pointer && is_written && binding.size != 0) {
        switch (Settings::values.native_compute_write_mark.GetValue()) {
        case Settings::NativeComputeWriteMark::Auto: {
            const auto& qmd = launch_desc;
            const std::array<u32, 3> block{qmd.block_dim_x.Value(), qmd.block_dim_y.Value(),
                                           qmd.block_dim_z.Value()};
            std::optional<std::array<u32, 3>> grid;
            if (!kepler_compute->GetIndirectComputeAddress()) {
                grid = std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                          qmd.grid_dim_z.Value()};
            }
            if (const auto extent = Shader::ResolveStorageWriteExtent(write_extent, block, grid)) {
                // The extent counts from the pointer, the binding from the pointer aligned down.
                const GPUVAddr pointer = gpu_memory->Read<u64>(ssbo_addr);
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                mark = std::min<u64>(binding.size, pad + *extent);
            }
            break;
        }
        case Settings::NativeComputeWriteMark::Full:
            break;
        }
    }
    channel_state->compute_storage_write_marks[ssbo_index] = static_cast<u32>(mark);
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    Nx2Downloads downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    // One copy command per buffer, with nearby ranges merged: see Nx2BatchDownloads.
    Nx2Downloads batched;
    u64 batched_size_bytes = 0;
    if (Nx2Perf::On(Nx2Perf::Opt::DownloadBatch)) {
        batched_size_bytes = Nx2BatchDownloads(downloads, batched);
        if (Nx2Perf::Verify() && !Nx2VerifyDownloadBatch(downloads, batched)) {
            batched.clear();
        }
    }
    const bool use_batched = !batched.empty();
    auto download_staging =
        runtime.DownloadStagingBuffer(use_batched ? batched_size_bytes : total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr orig_device_addr = static_cast<DAddr>(buffer.CpuAddr()) + copy.src_offset;
        if (Nx2Profile::Enabled()) {
            Nx2CountDownloadAges(orig_device_addr, copy.size);
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12;
                 page <= (orig_device_addr + copy.size - 1) >> 12; ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Copied(orig_device_addr, copy.size);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        if (use_batched) {
            continue;
        }
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        second_copy.src_offset = static_cast<size_t>(orig_device_addr);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    if (use_batched) {
        boost::container::small_vector<BufferCopy, 16> copies;
        for (size_t i = 0; i < batched.size();) {
            const BufferId buffer_id = batched[i].second;
            Buffer& buffer = slot_buffers[buffer_id];
            copies.clear();
            for (; i < batched.size() && batched[i].second == buffer_id; ++i) {
                BufferCopy copy = batched[i].first;
                copy.dst_offset += download_staging.offset;
                copies.push_back(copy);
                BufferCopy second_copy{copy};
                second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
                normalized_copies.push_back(second_copy);
            }
            runtime.CopyBuffer(download_staging.buffer, buffer,
                               std::span<const BufferCopy>(copies.data(), copies.size()), false);
        }
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
u64 BufferCache<P>::Nx2BatchDownloads(const Nx2Downloads& downloads, Nx2Downloads& batched) {
    // Ocarina downloaded ~13 ranges per fence, each its own recorded copy command. Merged ranges
    // copy their gap (at most 4 KiB) to staging too, but PopAsyncBuffers writes back only
    // pending_download_ranges, which keeps the exact ranges.
    constexpr u64 max_gap = 4_KiB;
    batched.assign(downloads.begin(), downloads.end());
    std::sort(batched.begin(), batched.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second.index != rhs.second.index) {
            return lhs.second.index < rhs.second.index;
        }
        return lhs.first.src_offset < rhs.first.src_offset;
    });
    size_t last = 0;
    for (size_t i = 1; i < batched.size(); ++i) {
        BufferCopy& merged = batched[last].first;
        const BufferCopy& next = batched[i].first;
        const u64 merged_end = merged.src_offset + merged.size;
        if (batched[i].second == batched[last].second && next.src_offset <= merged_end + max_gap) {
            merged.size = (std::max)(merged_end, next.src_offset + next.size) - merged.src_offset;
            continue;
        }
        batched[++last] = batched[i];
    }
    batched.erase(batched.begin() + last + 1, batched.end());
    u64 total_size_bytes = 0;
    for (auto& [copy, buffer_id] : batched) {
        copy.dst_offset = total_size_bytes;
        total_size_bytes += Common::AlignUp(copy.size, 64);
    }
    return total_size_bytes;
}

template <class P>
bool BufferCache<P>::Nx2VerifyDownloadBatch(const Nx2Downloads& downloads,
                                            const Nx2Downloads& batched) {
    static std::atomic<u64> mismatches{};
    bool ok = true;
    const auto report = [&](std::string_view what, BufferId buffer_id, u64 offset, u64 size) {
        ok = false;
        const u64 count = mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 64) {
            LOG_ERROR(HW_GPU,
                      "Nx2Perf download_batch: {} (buffer {}, offset {:#x}, size {:#x}; {} copies, "
                      "{} batched; mismatch {})",
                      what, buffer_id.index, offset, size, downloads.size(), batched.size(),
                      count);
        }
    };
    // Every unbatched copy lies inside a batched region of its buffer.
    for (const auto& [copy, buffer_id] : downloads) {
        const bool covered = std::ranges::any_of(batched, [&](const auto& region) {
            return region.second == buffer_id && region.first.src_offset <= copy.src_offset &&
                   copy.src_offset + copy.size <= region.first.src_offset + region.first.size;
        });
        if (!covered) {
            report("copy not covered", buffer_id, copy.src_offset, copy.size);
        }
    }
    // PopAsyncBuffers writes back the exact ranges inside each batched region: the same bytes as
    // the unbatched copies, each once, and nothing from the gaps.
    Common::RangeSet<DAddr> exact;
    u64 exact_bytes = 0;
    for (const auto& [copy, buffer_id] : downloads) {
        exact.Add(slot_buffers[buffer_id].CpuAddr() + copy.src_offset, copy.size);
        exact_bytes += copy.size;
    }
    u64 written_bytes = 0;
    u64 staging_end = 0;
    for (const auto& [region, buffer_id] : batched) {
        const Buffer& buffer = slot_buffers[buffer_id];
        if (region.src_offset + region.size > buffer.SizeBytes() ||
            region.dst_offset < staging_end) {
            report("region outside its buffer or overlapping in staging", buffer_id,
                   region.src_offset, region.size);
        }
        staging_end = region.dst_offset + region.size;
        exact.ForEachInRange(buffer.CpuAddr() + region.src_offset, region.size,
                             [&](DAddr start, DAddr end) { written_bytes += end - start; });
    }
    if (written_bytes != exact_bytes) {
        report("bytes written back differ", BufferId{}, written_bytes, exact_bytes);
    }
    return ok;
}

template <class P>
void BufferCache<P>::Nx2CountDownloadAges(DAddr device_addr, u64 size) {
    using Nx2Profile::Counter;
    constexpr u64 block_size = u64{1} << NX2_PREFLUSH_BLOCK_BITS;
    const DAddr end = device_addr + size;
    for (u64 block = device_addr >> NX2_PREFLUSH_BLOCK_BITS;
         block <= (end - 1) >> NX2_PREFLUSH_BLOCK_BITS; ++block) {
        const DAddr begin = (std::max)(device_addr, block * block_size);
        const DAddr block_end = (std::min)(end, (block + 1) * block_size);
        Counter counter = Counter::DownloadAgeNone;
        if (const auto it = nx2_preflush_blocks.find(block); it != nx2_preflush_blocks.end()) {
            const u64 age = frame_tick - it->second.last_read_frame;
            counter = age < 60     ? Counter::DownloadAge60
                      : age < 120  ? Counter::DownloadAge120
                      : age < 600  ? Counter::DownloadAge600
                      : age < 3600 ? Counter::DownloadAge3600
                                   : Counter::DownloadAgeOlder;
        }
        Nx2Profile::Add(counter, block_end - begin);
    }
}

template <class P>
void BufferCache<P>::Nx2StampPreflushBlocks(DAddr begin, DAddr end, bool marked) {
    const bool aging = Nx2Perf::On(Nx2Perf::Opt::PreflushAging);
    for (u64 block = begin >> NX2_PREFLUSH_BLOCK_BITS;
         block <= (end - 1) >> NX2_PREFLUSH_BLOCK_BITS; ++block) {
        if (!marked) {
            // A read of a marked range may span blocks without marks: stamp only marked ones.
            if (const auto it = nx2_preflush_blocks.find(block);
                it != nx2_preflush_blocks.end() && !it->second.dropped) {
                it->second.last_read_frame = frame_tick;
            }
            continue;
        }
        const auto [it, inserted] =
            nx2_preflush_blocks.try_emplace(block, Nx2PreflushBlock{.last_read_frame = frame_tick});
        if (inserted || !aging) {
            continue;
        }
        Nx2PreflushBlock& entry = it->second;
        entry.last_read_frame = frame_tick;
        if (!entry.dropped) {
            continue;
        }
        // Dropped too early: the read flushed (stalled). Keep the mark twice as long next time.
        entry.dropped = false;
        entry.limit_frames = (std::min)(entry.limit_frames * 2, NX2_PREFLUSH_MAX_LIMIT);
        entry.limit_frame = frame_tick;
        ++nx2_preflush_remarks;
        if (Nx2Perf::Verify()) {
            static u64 reports{};
            if (++reports <= 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf preflush_aging: read of dropped block {:#x} stalled (read "
                          "{:#x}-{:#x}, new limit {} frames; {} stalls in {} drops)",
                          block << NX2_PREFLUSH_BLOCK_BITS, begin, end, entry.limit_frames,
                          nx2_preflush_remarks, nx2_preflush_drops);
            }
        }
    }
}

template <class P>
void BufferCache<P>::Nx2AgePreflushBlocks() {
    // Invariant: GPU data not yet in guest memory (gpu_modified_ranges) is either in the download
    // pipeline with its mark (marked when QueueGpuWrite saw the write) or read-trapped (unmarked
    // at the write: QueueGpuWrite armed it; dropped here before PopAsyncBuffers wrote it back:
    // armed below). A read of a dropped block then takes GetFlushArea's flush path, which waits
    // for the GPU's data, and marks the block again.
    constexpr u64 block_size = u64{1} << NX2_PREFLUSH_BLOCK_BITS;
    size_t marked = 0;
    for (auto it = nx2_preflush_blocks.begin(); it != nx2_preflush_blocks.end();) {
        Nx2PreflushBlock& entry = it->second;
        if (entry.dropped) {
            // The limit halves per 3600 frames unread; back at the minimum the block is forgotten.
            if (frame_tick - entry.limit_frame >= NX2_PREFLUSH_MAX_LIMIT) {
                if (entry.limit_frames <= NX2_PREFLUSH_MIN_LIMIT) {
                    it = nx2_preflush_blocks.erase(it);
                    continue;
                }
                entry.limit_frames = (std::max)(entry.limit_frames / 2, NX2_PREFLUSH_MIN_LIMIT);
                entry.limit_frame = frame_tick;
            }
            ++it;
            continue;
        }
        if (frame_tick - entry.last_read_frame <= entry.limit_frames) {
            ++marked;
            ++it;
            continue;
        }
        const DAddr begin = static_cast<DAddr>(it->first << NX2_PREFLUSH_BLOCK_BITS);
        memory_tracker.UnmarkRegionAsPreflushable(begin, block_size);
        gpu_modified_ranges.ForEachInRange(begin, block_size, [&](DAddr start, DAddr end) {
            ArmReadTraps(start, end - start);
        });
        entry.dropped = true;
        entry.limit_frame = frame_tick;
        ++nx2_preflush_drops;
        ++it;
    }
    if (frame_tick % 3600 == 0) {
        LOG_INFO(HW_GPU,
                 "Nx2Perf preflush_aging: {} blocks marked, {} dropped; {} drops and {} stalled "
                 "reads of dropped blocks since start",
                 marked, nx2_preflush_blocks.size() - marked, nx2_preflush_drops,
                 nx2_preflush_remarks);
    }
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
template <typename Ranges>
void BufferCache<P>::ClaimCpuWrites(const Ranges& ranges) {
    for (const auto& [device_addr, size] : ranges) {
        if (IsRegionGpuModified(device_addr, size)) {
            ClearReplacedDownload(device_addr, size);
            gpu_modified_ranges.Subtract(device_addr, size);
        }
    }
}

template <class P>
Common::RangeSet<DAddr> BufferCache<P>::PendingCpuWriteRanges() {
    Common::RangeSet<DAddr> ranges;
    if (pending_cpu_writes) {
        pending_cpu_writes(
            [&ranges](DAddr device_addr, size_t size) { ranges.Add(device_addr, size); });
    }
    return ranges;
}

template <class P>
void BufferCache<P>::WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written,
                                           DAddr device_addr, const u8* data, u64 size) {
    // EDEN_NX2_READ_CHECK: newer GPU bytes than an early download brought before.
    if (!early_landed.empty()) {
        ForgetEarlyLanded(device_addr, size);
    }
    if (!cpu_written.Intersects(device_addr, size)) {
        device_memory.WriteBlockUnsafe(device_addr, data, size);
        VideoCommon::g_buffer_write_pages.Mark(device_addr, size);
        return;
    }
    Common::RangeSet<DAddr> parts;
    parts.Add(device_addr, size);
    cpu_written.ForEachInRange(device_addr, size, [&parts](DAddr start, DAddr end) {
        parts.Subtract(start, end - start);
    });
    parts.ForEach([&](DAddr start, DAddr end) {
        device_memory.WriteBlockUnsafe(start, data + (start - device_addr), end - start);
        VideoCommon::g_buffer_write_pages.Mark(start, end - start);
    });
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    // The host GPU is past this fence: the read downloads taken before its copies are complete,
    // and go to guest memory first. Those taken after took their bytes out of its copies.
    WriteCompletedReadDownloads();
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const auto& copy : downloads) {
        landing_begin = (std::min)(landing_begin, static_cast<DAddr>(copy.src_offset));
        landing_end = (std::max)(landing_end, static_cast<DAddr>(copy.src_offset + copy.size));
    }
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    // This runs on the fencing thread while the CPU keeps writing: CPU writes collected since the
    // copy was taken are newer than it, and the GPU thread hasn't dropped them from `ranges` yet.
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const auto& copy : downloads) {
        const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
        const u64 dst_offset = copy.dst_offset - base_offset;
        const u8* read_mapped_memory = base + dst_offset;
        // Only what no write replaced since the copy was taken: the rest is older than guest
        // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
        ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
            WriteDownloadedMemory(cpu_written, start, &read_mapped_memory[start - device_addr],
                                  end - start);
            gpu_modified_ranges.Subtract(start, end - start);
            written_bytes += end - start;
        });
    }
    if (Nx2Profile::Enabled()) {
        Nx2Profile::g_fencing_written_bytes.fetch_add(written_bytes, std::memory_order_relaxed);
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    return gpu_modified_ranges.Intersects(addr, size);
}

template <class P>
bool BufferCache<P>::ComputeWritesHoldGpuData() {
    bool holds = true;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers &
                          channel_state->written_compute_storage_buffers,
                      [&](u32 index) {
                          const Binding& binding = channel_state->compute_storage_buffers[index];
                          // As much as BindHostComputeStorageBuffers marks written
                          const bool native =
                              ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
                          const u32 size =
                              native ? (std::min)(binding.size,
                                                  channel_state->compute_storage_write_marks[index])
                                     : binding.size;
                          holds &= IsRegionFullyGpuModified(binding.device_addr, size);
                      });
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers &
                          channel_state->written_compute_texture_buffers,
                      [&](u32 index) {
                          const Binding& binding = channel_state->compute_texture_buffers[index];
                          holds &= IsRegionFullyGpuModified(binding.device_addr, binding.size);
                      });
    return holds;
}

template <class P>
bool BufferCache<P>::IsRegionFullyGpuModified(DAddr addr, size_t size) {
    size_t covered = 0;
    gpu_modified_ranges.ForEachInRange(addr, size,
                                       [&](DAddr lower, DAddr upper) { covered += upper - lower; });
    return covered == size;
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if constexpr (requires { runtime.BindHostUint8IndexBuffer(std::span<const u8>{}); }) {
        if (draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
            draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
            draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
            runtime.WidensUint8IndicesOnHost()) {
            std::span<const u8> indices;
            const auto& index_buffer = draw_state.index_buffer;
            const u32 guest_size = (std::min)(
                static_cast<u32>(index_buffer.EndAddress() - index_buffer.StartAddress()),
                index_buffer.first + index_buffer.count);
            if (!draw_state.inline_index_draw_indexes.empty()) {
                indices = std::span{draw_state.inline_index_draw_indexes.data(), size};
            } else if (guest_size != 0 &&
                       !IsRegionGpuModified(channel_state->index_buffer.device_addr, guest_size)) {
                guest_uint8_indices.resize(guest_size);
                device_memory.ReadBlockUnsafe(channel_state->index_buffer.device_addr,
                                              guest_uint8_indices.data(), guest_size);
                indices = guest_uint8_indices;
            }
            if (!indices.empty()) {
                runtime.BindHostUint8IndexBuffer(indices);
                return;
            }
        }
    }
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeGeometryBuffer(NUM_VERTEX_BUFFERS, channel_state->index_buffer.buffer_id,
                                  channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{
                {BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(),
                        draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first *
                                            u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format,
                                draw_state.index_buffer.first, draw_state.index_buffer.count,
                                buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = maxwell3d->dirty.flags;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers =
        Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings;
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings.buffers.clear();
            bindings.offsets.clear();
            bindings.sizes.clear();
            bindings.strides.clear();
            bindings.min_index = NUM_VERTEX_BUFFERS;
            bindings.max_index = 0;
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBufferOnce(vertex_touches[index], buffer, binding.buffer_id);
            SynchronizeGeometryBuffer(index, binding.buffer_id, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
        // The draw reads these words on the GPU: a later upload to them in this batch must not
        // be reordered ahead of it (DK Bananza resets its terrain draw records with inline
        // writes right after the draws that read them; hoisted, every record read zero).
        buffer.MarkUsage(buffer.Offset(binding.device_addr), binding.size);
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index,
                                                   bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    // Banks 24-31 (the root tables; NVN2 keeps push constants in table 0) are read from bank 17,
    // which such games do not bind: stream the tables' current words, one after another, as that
    // buffer.
    if (index == 17 && maxwell3d->root_tables.loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = maxwell3d->root_tables.tables;
        static_assert(sizeof(words) == 8 * 256);
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index,
                                          std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    // EDEN_NX2_BIND_STATS: bind cost by size class, and whether the bytes equal this slot's
    // previous bind (what extending the stream reuse to larger buffers could exploit).
    bool uniform_same = false;
    if (VideoCommon::Nx2DrawStats::HashContents() && size != 0) [[unlikely]] {
        static thread_local std::vector<u8> scratch;
        scratch.resize((std::min)(size, u32{0x10000}));
        device_memory.ReadBlockUnsafe(device_addr, scratch.data(), scratch.size());
        const u64 key = (u64{static_cast<u32>(stage)} << 40) ^ (u64{index} << 32) ^ size;
        uniform_same = VideoCommon::Nx2DrawStats::SameAsLast(
            key, VideoCommon::Nx2DrawStats::HashBytes(scratch.data(), scratch.size()));
    }
    VideoCommon::Nx2DrawStats::Timer uniform_probe{
        size <= 4096 ? VideoCommon::Nx2DrawStats::Site::UniformSmall
        : size < 0x10000 ? VideoCommon::Nx2DrawStats::Site::UniformMedium
                         : VideoCommon::Nx2DrawStats::Site::UniformBank};
    uniform_probe.hit = uniform_same;
    Buffer& buffer = slot_buffers[binding.buffer_id];
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    // Native programs keep tables next to their code in a 64 KiB bank read with indirect offsets
    // (Splatoon Raiders: the dialogue box corners in c1). Writes to that memory can go unnoticed by
    // the memory tracker, leaving stale host contents, so upload such a bank again once per frame.
    const bool native_bank = has_host_buffer &&
                             (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000 &&
                             size != 0 && gpu_memory != nullptr && gpu_memory->UsesNativeTiling();
    bool verify_kept = false;
    if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
        // Unnoticed writes would reach a kept bank too: those stream as before.
        if (!native_bank && UniformRangeKept(stage, index, device_addr, size)) {
            if (ubo_keep_verify) {
                // Only the ranges RebindKeptUniform would bind again: the others are streamed.
                if (VideoCommon::KeptUniformMatches(runtime, stage, index, device_addr, size)) {
                    ++uniform_keep.verified;
                    verify_kept = true;
                    if (!VideoCommon::KeptUniformHolds<Runtime>(device_memory, stage, index,
                                                                 device_addr, size)) {
                        if (uniform_keep.mismatches++ < 16) {
                            LOG_ERROR(HW_GPU,
                                      "Uniform keep verify: stage {} slot {} at {:#x}+{:#x} "
                                      "changed since the previous draw",
                                      stage, index, device_addr, size);
                        }
                    }
                }
            } else if (VideoCommon::RebindKeptUniform(runtime, stage, index, device_addr, size)) {
                channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                ++uniform_keep.kept;
                return;
            }
        }
        uniform_keep.clean[stage] &= ~(1u << index);
    }
    // A kept range binds a stream allocation, not the buffer: the buffer is touched once it is
    // read.
    TouchBufferOnce(uniform_touches[stage][index], buffer, binding.buffer_id);
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    // Native programs keep tables next to their code in uniform banks read with indirect offsets
    // (Splatoon Raiders: the dialogue box corners in c1). Writes to that memory can go unnoticed by
    // the memory tracker, leaving stale host contents, so upload such a bank again once per frame
    // by default (with 4 KiB page-level shadow diffing so unchanged pages stay clean).
    // EDEN_NX2_BANK_REUPLOAD: 0 off, 1 once per frame (the default), 2 every bind of a 64 KiB
    // bank, 3 every bind of any native bank.
    static const char bank_mode = [] {
        const char* v = std::getenv("EDEN_NX2_BANK_REUPLOAD");
        return v != nullptr ? v[0] : '1';
    }();
    // EDEN_NX2_HOST_IMPORT: a native 64 KiB bank nothing on the GPU writes is read
    // by the host GPU from the guest's own pages, with no copy and no re-upload. Debug knob 10
    // inverts it while the game runs.
    if constexpr (requires { runtime.TryBindImportedUniformBuffer(static_cast<const u8*>(nullptr), u32{}); }) {
        // On by default for the NVIDIA proprietary driver on Linux; =1 / =0 force it on / off.
        static const bool import_start = [this] {
            const char* const value = std::getenv("EDEN_NX2_HOST_IMPORT");
            if (value != nullptr) {
                return value[0] == '1';
            }
            if constexpr (requires { runtime.HostImportByDefault(); }) {
                return runtime.HostImportByDefault();
            }
            return false;
        }();
        if (has_host_buffer && size != 0 && gpu_memory != nullptr &&
            gpu_memory->UsesNativeTiling() &&
            (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000 &&
            VideoCommon::Nx2Knob::Live(import_start, VideoCommon::Nx2Knob::HostImport) &&
            !memory_tracker.IsRegionGpuModified(device_addr, size)) {
            static std::atomic<u64> imported{0}, copied{0}, no_span{0}, no_import{0};
            const u8* const host_ptr = device_memory.GetSpan(device_addr, size);
            if (host_ptr != nullptr && runtime.TryBindImportedUniformBuffer(host_ptr, size)) {
                channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
                const u64 count = imported.fetch_add(1, std::memory_order_relaxed) + 1;
                if ((count % 500000) == 0) {
                    LOG_INFO(HW_GPU,
                             "nx2: {} banks bound from guest memory, {} copied instead ({} not "
                             "physically contiguous, {} unaligned or not importable)",
                             count, copied.load(std::memory_order_relaxed),
                             no_span.load(std::memory_order_relaxed),
                             no_import.load(std::memory_order_relaxed));
                }
                return;
            }
            copied.fetch_add(1, std::memory_order_relaxed);
            (host_ptr == nullptr ? no_span : no_import).fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (bank_mode != '0' && has_host_buffer && size != 0 && gpu_memory != nullptr &&
        gpu_memory->UsesNativeTiling() &&
        (bank_mode == '3' || (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000)) {
        Nx2Profile::Scope bank_scope{Nx2Profile::Zone::BindUniformBank};
        // The bank size is only an upper limit: stop at the end of the mapped memory.
        // Only mark 4 KiB pages whose guest bytes actually changed since the last upload so
        // unchanged banks do not toggle VirtualProtect or break the Vulkan render pass on every draw.
        NativeUniformBankShadow* shadow = nullptr;
        if (device_addr == native_indirect_uniform_last_bank &&
            native_indirect_uniform_last_ptr != nullptr) [[likely]] {
            shadow = native_indirect_uniform_last_ptr;
        } else {
            const size_t cache_idx = ((device_addr >> 12) ^ (device_addr >> 20)) & 7;
            auto& cached = native_indirect_uniform_cache[cache_idx];
            if (cached.first == device_addr && cached.second != nullptr) [[likely]] {
                shadow = cached.second;
            } else {
                shadow = &native_indirect_uniform_shadow[device_addr];
                cached = {device_addr, shadow};
            }
            native_indirect_uniform_last_bank = device_addr;
            native_indirect_uniform_last_ptr = shadow;
        }
        // Nx2Perf bank_once_per_frame: compare the guest bytes on the frame's first bind only.
        // Verify (which always takes the per-draw path) reports changes that path would have
        // uploaded one frame late.
        const bool once_per_frame = bank_mode != '2' && bank_mode != '3' &&
                                    Nx2Perf::On(Nx2Perf::Opt::BankOncePerFrame);
        const bool skip_this_frame = once_per_frame && !Nx2Perf::Verify();
        const bool seen_this_frame = shadow->last_frame_tick == frame_tick;
        if (skip_this_frame && seen_this_frame) {
            if (Nx2Profile::Enabled() &&
                shadow->last_draw_ticket != native_indirect_uniform_draw_ticket &&
                shadow->last_skipped_draw_ticket != native_indirect_uniform_draw_ticket &&
                !memory_tracker.IsRegionGpuModified(device_addr, size)) {
                shadow->last_skipped_draw_ticket = native_indirect_uniform_draw_ticket;
                Nx2Profile::Add(Nx2Profile::Counter::BankRechecksSkippedByFrame, 1);
            }
        } else if (!memory_tracker.IsRegionGpuModified(device_addr, size)) {
            const bool check_late = once_per_frame && seen_this_frame;
            shadow->last_frame_tick = frame_tick;
            if (shadow->last_draw_ticket != native_indirect_uniform_draw_ticket) {
                shadow->last_draw_ticket = native_indirect_uniform_draw_ticket;
                const u32 check_size = (std::min<u32>)(size, 0x10000U);
                const u8* const base_ptr = device_memory.template GetPointer<u8>(device_addr);
                const bool contiguous =
                    base_ptr != nullptr &&
                    (check_size <= Core::DEVICE_PAGESIZE ||
                     base_ptr + (check_size - 1) ==
                         device_memory.template GetPointer<u8>(device_addr + check_size - 1));
                u32 offset_in_bank = 0;
                u32 page_idx = 0;
                while (offset_in_bank < check_size) {
                    const DAddr cur_addr = device_addr + offset_in_bank;
                    const u64 page_offset = cur_addr & Core::DEVICE_PAGEMASK;
                    const u32 chunk_size = static_cast<u32>((std::min<u64>)(
                        Core::DEVICE_PAGESIZE - page_offset, check_size - offset_in_bank));
                    const u8* const ptr = contiguous
                                              ? (base_ptr + offset_in_bank)
                                              : device_memory.template GetPointer<u8>(cur_addr);
                    if (ptr == nullptr) {
                        break;
                    }
                    const u32 page_bit = 1U << (page_idx & 31);
                    const bool page_initialized = (shadow->initialized_mask & page_bit) != 0;
                    const bool already_cpu_modified =
                        memory_tracker.IsRegionCpuModified(cur_addr, chunk_size);
                    u8* const shadow_dst = shadow->bytes.data() + offset_in_bank;
                    if (!page_initialized || already_cpu_modified ||
                        std::memcmp(shadow_dst, ptr, chunk_size) != 0) {
                        if (check_late && page_initialized && !already_cpu_modified)
                            [[unlikely]] {
                            static std::atomic<u64> late_changes{};
                            const u64 late =
                                late_changes.fetch_add(1, std::memory_order_relaxed) + 1;
                            Nx2Profile::Add(Nx2Profile::Counter::BankLateChanges, 1);
                            if (late <= 64) {
                                LOG_ERROR(HW_GPU,
                                          "Nx2Perf bank_once_per_frame: bank {:#x} page {} "
                                          "changed after the frame's first bind (frame {})",
                                          device_addr, page_idx, frame_tick);
                            } else if (late % 1024 == 0) {
                                LOG_ERROR(HW_GPU,
                                          "Nx2Perf bank_once_per_frame: {} bank pages changed "
                                          "after the frame's first bind so far",
                                          late);
                            }
                        }
                        std::memcpy(shadow_dst, ptr, chunk_size);
                        shadow->initialized_mask |= page_bit;
                        if (!already_cpu_modified) {
                            memory_tracker.MarkRegionAsCpuModified(cur_addr, chunk_size);
                            last_synced_count = 0;
                        }
                    }
                    offset_in_bank += chunk_size;
                    ++page_idx;
                }
            }
        }
    }
    const bool use_fast_buffer =
        needs_alignment_stream ||
        (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
         !memory_tracker.IsRegionGpuModified(device_addr, size));
    if (use_fast_buffer) {
        Nx2Profile::Scope stream_scope{Nx2Profile::Zone::BindUniformStream};
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off.
            if (VideoCommon::UniformStreamReuseEnabled()) {
                VideoCommon::Nx2Hazard::CheckRead(VideoCommon::Nx2Hazard::Site::UniformReuse,
                                                  device_addr, size);
                streamed = VideoCommon::BindSmallUniformReusing(
                    runtime, device_memory, stage, index, binding_index, device_addr, size);
                if (streamed) {
                    uniform_keep.clean[stage] |= 1u << index;
                    uniform_keep.ranges[stage][index] = {device_addr, size};
                    ++uniform_keep.streamed_binds;
                }
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            VideoCommon::Nx2Hazard::CheckRead(VideoCommon::Nx2Hazard::Site::UniformFast, device_addr,
                                              size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        }
        return;
    }
    if (native_bank && has_host_buffer && !memory_tracker.IsRegionGpuModified(device_addr, size)) {
        CheckNativeBankPagesTracked(buffer, binding.buffer_id, device_addr, size);
    }
    // Classic cached path
    if (verify_kept && uniform_keep.mismatches++ < 16) {
        LOG_ERROR(HW_GPU,
                  "Uniform keep verify: stage {} slot {} at {:#x}+{:#x} is no longer streamed",
                  stage, index, device_addr, size);
    }
    if (SynchronizeBuffer(buffer, device_addr, size)) {
        ++channel_state->uniform_cache_hits[0];
    } else {
        const u64 cpu_epoch = memory_tracker.CpuEpoch();
        TouchBuffer(buffer, binding.buffer_id);
        // Classic cached path
        if (SynchronizeBuffer(buffer, device_addr, size)) {
            ++channel_state->uniform_cache_hits[0];
        }
        if (memo != nullptr) {
            BindMemoNote(*memo, binding.buffer_id, device_addr, size, cpu_epoch);
        }
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    if (!memo_hit || !memo->usage_marked) {
        buffer.MarkUsage(offset, size);
        if (memo != nullptr) {
            BindMemoNoteUsage(*memo);
        }
    }
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
void BufferCache<P>::CheckNativeBankPagesTracked(Buffer& buffer, BufferId buffer_id,
                                                 DAddr device_addr, u32 size) {
    u64 mapped_size = 0;
    while (mapped_size < size &&
           device_memory.template GetPointer<u8>(device_addr + mapped_size) != nullptr) {
        mapped_size +=
            Core::DEVICE_PAGESIZE - ((device_addr + mapped_size) & Core::DEVICE_PAGEMASK);
    }
    const u64 active_size = (std::min<u64>)(mapped_size, size);
    if (active_size == 0) {
        return;
    }
    if (native_bank_pages.size() >= MaxNativeBankPageSnapshots) {
        native_bank_pages.clear();
    }
    const DAddr first_page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    const DAddr end = device_addr + active_size;
    const DAddr aligned_end = Common::AlignUp(end, Core::DEVICE_PAGESIZE);
    const bool any_cpu_dirty =
        memory_tracker.IsRegionCpuModified(first_page, aligned_end - first_page);
    NativeBankUploads uploads;
    for (DAddr page = first_page; page < end; page += Core::DEVICE_PAGESIZE) {
        const u8* const guest = device_memory.template GetPointer<u8>(page);
        if (guest == nullptr) {
            continue;
        }
        const u16 slice_begin = static_cast<u16>((std::max)(page, device_addr) - page);
        const u16 slice_end =
            static_cast<u16>((std::min<DAddr>)(page + Core::DEVICE_PAGESIZE, end) - page);
        const u32 slice_size = static_cast<u32>(slice_end - slice_begin);
        const bool collected = VideoCommon::g_collected_write_pages.Take(page);
        const bool written = VideoCommon::g_buffer_write_pages.Take(page);
        auto [it, inserted] = native_bank_pages.try_emplace(page);
        NativeBankPage& snapshot = it->second;
        if (snapshot.buffer_id != buffer_id) {
            snapshot.buffer_id = buffer_id;
            snapshot.covered_begin = static_cast<u16>(Core::DEVICE_PAGESIZE);
            snapshot.covered_end = 0;
        }
        if (inserted || slice_begin < snapshot.covered_begin || slice_end > snapshot.covered_end) {
            std::memcpy(snapshot.words.data() + slice_begin, guest + slice_begin, slice_size);
            snapshot.covered_begin = (std::min)(snapshot.covered_begin, slice_begin);
            snapshot.covered_end = (std::max)(snapshot.covered_end, slice_end);
            snapshot.checked_frame = frame_tick;
            snapshot.recheck = collected || written;
            QueueNativeBankUpload(buffer, page + slice_begin, snapshot.words.data() + slice_begin,
                                  slice_size, uploads);
            continue;
        }
        const bool recheck = std::exchange(snapshot.recheck, collected || written);
        const bool hot = snapshot.changed_frame != ~u64{0} &&
                         frame_tick - snapshot.changed_frame < NativeBankHotFrames &&
                         snapshot.checked_frame != frame_tick;
        const bool should_compare =
            Nx2Perf::Verify() || collected || written || recheck ||
            (any_cpu_dirty && memory_tracker.IsRegionCpuModified(page, Core::DEVICE_PAGESIZE)) ||
            hot || snapshot.checked_frame != frame_tick;
        if (!should_compare) {
            continue;
        }
        snapshot.checked_frame = frame_tick;
        if (VideoCommon::Nx2BytesEqual(snapshot.words.data() + slice_begin, guest + slice_begin,
                                       slice_size)) {
            continue;
        }
        snapshot.changed_frame = frame_tick;
        std::memcpy(snapshot.words.data() + slice_begin, guest + slice_begin, slice_size);
        QueueNativeBankUpload(buffer, page + slice_begin, snapshot.words.data() + slice_begin,
                              slice_size, uploads);
    }
    if (uploads.count != 0) {
        FlushNativeBankUploads(buffer, uploads);
    }
}

template <class P>
void BufferCache<P>::QueueNativeBankUpload([[maybe_unused]] Buffer& buffer, DAddr slice_addr,
                                           [[maybe_unused]] const u8* words, u32 slice_size,
                                           [[maybe_unused]] NativeBankUploads& uploads) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        if (uploads.count < uploads.slices.size() && slice_addr >= buffer.CpuAddr() &&
            slice_addr + slice_size <= buffer.CpuAddr() + buffer.SizeBytes() &&
            !gpu_modified_ranges.Intersects(slice_addr, slice_size)) {
            uploads.slices[uploads.count++] = NativeBankUpload{
                .addr = slice_addr,
                .words = words,
                .size = slice_size,
            };
            uploads.total_bytes += slice_size;
            return;
        }
    }
    memory_tracker.MarkRegionAsCpuModified(Common::AlignDown(slice_addr, Core::DEVICE_PAGESIZE),
                                           Core::DEVICE_PAGESIZE);
}

template <class P>
void BufferCache<P>::FlushNativeBankUploads([[maybe_unused]] Buffer& buffer,
                                            [[maybe_unused]] const NativeBankUploads& uploads) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(uploads.total_bytes);
        std::array<BufferCopy, NativeBankUploadCapacity> copies{};
        size_t num_copies = 0;
        u64 staging_offset = 0;
        for (size_t i = 0; i < uploads.count; ++i) {
            const NativeBankUpload& slice = uploads.slices[i];
            std::memcpy(upload_staging.mapped_span.data() + staging_offset, slice.words,
                        slice.size);
            const u64 dst_offset = buffer.Offset(slice.addr);
            if (num_copies != 0 &&
                copies[num_copies - 1].dst_offset + copies[num_copies - 1].size == dst_offset) {
                copies[num_copies - 1].size += slice.size;
            } else {
                copies[num_copies++] = BufferCopy{
                    .src_offset = upload_staging.offset + staging_offset,
                    .dst_offset = dst_offset,
                    .size = slice.size,
                };
            }
            staging_offset += slice.size;
        }
        const std::span<BufferCopy> copy_span(copies.data(), num_copies);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copy_span);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copy_span, true, can_reorder);
    }
}

template <class P>
void BufferCache<P>::RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        VideoCommon::RefreshMultiRangeSegments(
            std::span{pool}, [this](BufferId id) { return slot_buffers.contains(id); },
            [this](DAddr address, u32 size) {
                return FindBuffer(address, size, runtime.PrefersSparseSources());
            });
    }
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.single_range || binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        // Ranges found in one segment, until a map changes: the query is a hash lookup, for
        // every storage binding of every draw.
        const u64 map_serial = gpu_memory->MapSerial();
        SingleRangeMemo& memo = single_ranges[(binding.gpu_addr >> 8) % single_ranges.size()];
        const bool memo_hit = memo.memory == gpu_memory && memo.map_serial == map_serial &&
                              memo.gpu_addr == binding.gpu_addr && binding.size <= memo.size;
        if (memo_hit && !Nx2Perf::Verify()) {
            return;
        }
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            // No segments also means a gap: a shorter range at the address may still be split.
            if (found && found->size() == 1) {
                memo = {gpu_memory, map_serial, binding.gpu_addr, binding.size};
            }
            return;
        }
        if (memo_hit) {
            LOG_ERROR(HW_GPU, "Nx2Perf storage range: {:#x}+{:#x} remembered in one segment, has {}",
                      binding.gpu_addr, binding.size, found->size());
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id = FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool,
                                           u64 write_mark_limit) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first > pool.size() ||
            binding.segment_count > pool.size() - binding.segment_first) {
            return false;
        }
        const auto segments = pool.subspan(binding.segment_first, binding.segment_count);
        if (!MultiRangeSegmentsLive(segments,
                                    [this](BufferId id) { return slot_buffers.contains(id); })) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            SynchronizeBuffer(buffer, segment.device_addr, segment.size);
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written && write_mark_limit != 0) {
                // write_mark_limit: the native write mark (BindComputeStorageBuffer), spread over
                // the segments in order.
                const u32 mark = static_cast<u32>(std::min<u64>(segment.size, write_mark_limit));
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, mark);
                if (write_mark_limit != ~u64{0}) {
                    write_mark_limit -= mark;
                }
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return BindFreshMultiRangeStorage(runtime, key, is_written);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    nx2_graphics_stage = static_cast<int>(stage + 1);
    SCOPE_EXIT {
        nx2_graphics_stage = -1;
    };
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        Nx2Profile::Scope storage_scope{Nx2Profile::Zone::BindStorage};
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        BindMemo* const memo = BindMemoEnabled() && binding.buffer_id != NULL_BUFFER_ID
                                   ? &storage_bind_memo[stage][index]
                                   : nullptr;
        const bool memo_hit =
            memo != nullptr && !is_written &&
            BindMemoHit(*memo, buffer, binding.buffer_id, binding.device_addr, binding.size,
                        buffer.Offset(binding.device_addr), "storage", stage, index) &&
            memo->usage_marked;
        const u64 cpu_epoch = memory_tracker.CpuEpoch();
        if (!memo_hit) {
            TouchBuffer(buffer, binding.buffer_id);
        }
        const u32 size = binding.size;
        if (!memo_hit) {
            SynchronizeBuffer(buffer, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        if (!memo_hit) {
            buffer.MarkUsage(offset, size);
            if (memo != nullptr && is_written) {
                memo->valid = false;
            } else if (memo != nullptr) {
                BindMemoNote(*memo, binding.buffer_id, binding.device_addr, size, cpu_epoch);
                BindMemoNoteUsage(*memo);
            }
        }

        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    nx2_graphics_stage = static_cast<int>(stage + 1);
    SCOPE_EXIT {
        nx2_graphics_stage = -1;
    };
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !memory_tracker.IsRegionGpuModified(binding.device_addr, size));
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                bool streamed = false;
                if constexpr (VideoCommon::StreamUniformReusable<Runtime>) {
                    if (VideoCommon::UniformStreamReuseEnabled()) {
                        streamed = VideoCommon::BindSmallComputeUniformReusing(
                            runtime, device_memory, index, binding_index, binding.device_addr,
                            size);
                    }
                }
                if (!streamed) {
                    const std::span<u8> span =
                        runtime.BindMappedUniformBuffer(0, binding_index, size);
                    VideoCommon::Nx2Hazard::CheckRead(VideoCommon::Nx2Hazard::Site::UniformCompute,
                                                      binding.device_addr, size);
                    device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                }
                return;
            }
        }

        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        // A native program's binding is min(layout, 8 MiB) because its pointer has no size, and
        // marking all of it GPU-written defers the CPU's later writes to those pages. Mark only
        // what BindComputeStorageBuffer worked out the dispatch can write; the binding keeps its
        // size so the shader can still read all of it.
        const bool native =
            ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
        const u64 write_mark_limit =
            native ? u64{channel_state->compute_storage_write_marks[index]} : ~u64{0};
        if (BindMultiRangeStorage(binding, is_written, compute_segments, write_mark_limit)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        const u32 mark =
            is_written ? static_cast<u32>(std::min<u64>(size, write_mark_limit)) : 0;
        if (mark != 0) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, true);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            if (channel_state->enabled_uniform_buffer_masks[stage] != 0) {
                UpdateUniformBuffers(stage);
            }
            if (channel_state->enabled_storage_buffers[stage] != 0) {
                UpdateStorageBuffers(stage);
            }
            if (channel_state->enabled_texture_buffers[stage] != 0) {
                UpdateTextureBuffers(stage);
            }
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
        RefreshMultiRangeSegments(graphics_segments);
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
        RefreshMultiRangeSegments(compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    const VideoCommon::Nx2DrawStats::FindCaller find_caller{VideoCommon::Nx2DrawStats::Site::FindBufferIndex};
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            DeleteBuffer(inline_buffer_id);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) *
                          u32(index_buffer_ref.FormatSizeInBytes());
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    auto& current = channel_state->index_buffer;
    const u64 map_serial = gpu_memory->MapSerial();
    const std::optional<DAddr> device_addr =
        (current.gpu_addr == gpu_addr_begin && current.map_serial == map_serial &&
         current.device_addr != 0)
            ? std::optional<DAddr>{current.device_addr}
            : gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    if (!device_addr) {
        current = NULL_BINDING;
        return;
    }
    current = Binding{
        .device_addr = *device_addr,
        .gpu_addr = gpu_addr_begin,
        .map_serial = map_serial,
        .size = size,
        .buffer_id = FindBuffer(*device_addr, size, false),
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    const VideoCommon::Nx2DrawStats::FindCaller find_caller{VideoCommon::Nx2DrawStats::Site::FindBufferVertex};
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    const auto flags_words = std::bit_cast<std::array<u64, 4>>(flags);
    static_assert(Dirty::VertexBuffer31 < 64, "vertex buffer flags must fit in the first word");
    u32 dirty_vb_mask = static_cast<u32>(flags_words[0] >> Dirty::VertexBuffer0);
    if (const u32 unsupported = dirty_vb_mask & ~VERTEX_BUFFER_MASK; unsupported != 0) {
        LOG_WARNING(HW_GPU, "Ignoring unsupported vertex streams, mask={:#x}", unsupported);
        for (u32 index = NUM_VERTEX_BUFFERS; index < 32; ++index) {
            flags[Dirty::VertexBuffer0 + index] = false;
        }
        dirty_vb_mask &= VERTEX_BUFFER_MASK;
    }
    while (dirty_vb_mask != 0) {
        const u32 index = static_cast<u32>(std::countr_zero(dirty_vb_mask));
        dirty_vb_mask &= dirty_vb_mask - 1;
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    if (array.enable == 0) {
        // Nothing binds a disabled stream (BindHostVertexBuffers takes the enabled ones), so its
        // flag stayed set and every later change of another stream resolved it again.
        maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index] = false;
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    u32 size = address_size; // TODO: Analyze stride and number of vertices
    if (array.enable == 0 || size == 0) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    auto& current = channel_state->vertex_buffers[index];
    const u64 map_serial = gpu_memory->MapSerial();
    if (!channel_state->has_deleted_buffers && current.buffer_id &&
        current.buffer_id != NULL_BUFFER_ID && current.gpu_addr == gpu_addr_begin &&
        current.size == size && current.map_serial == map_serial && current.device_addr != 0 &&
        !Settings::IsGPUFenceBehaviorAccurate()) [[likely]] {
        return;
    }
    const std::optional<DAddr> device_addr =
        (current.gpu_addr == gpu_addr_begin && current.map_serial == map_serial &&
         current.device_addr != 0)
            ? std::optional<DAddr>{current.device_addr}
            : gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    if (!device_addr) {
        current = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    const Binding binding{
        .device_addr = *device_addr,
        .gpu_addr = gpu_addr_begin,
        .map_serial = map_serial,
        .size = size,
        .buffer_id = buffer_id,
    };
    current = binding;
    UpdateVertexBufferSlot(index, binding);
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    const u32 to_resolve = channel_state->enabled_uniform_buffer_masks[stage] &
                           channel_state->unresolved_uniform_buffers[stage];
    if (to_resolve == 0) [[likely]] {
        return;
    }
    channel_state->unresolved_uniform_buffers[stage] &= ~to_resolve;
    ForEachEnabledBit(to_resolve, [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    const VideoCommon::Nx2DrawStats::FindCaller find_caller{VideoCommon::Nx2DrawStats::Site::FindBufferStorage};
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->storage_buffers[stage][index];
        if (!binding.buffer_id) {
            binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        }
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, graphics_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    const VideoCommon::Nx2DrawStats::FindCaller find_caller{VideoCommon::Nx2DrawStats::Site::FindBufferTexture};
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    const u64 map_serial = gpu_memory->MapSerial();
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        const auto& launch_desc = kepler_compute->launch_description;
        if (((launch_desc.const_buffer_enable_mask >> index) & 1) != 0) {
            const auto& cbuf = launch_desc.const_buffer_config[index];
            const GPUVAddr gpu_addr = cbuf.Address();
            if (!channel_state->has_deleted_buffers && binding.buffer_id &&
                binding.buffer_id != NULL_BUFFER_ID && binding.gpu_addr == gpu_addr &&
                binding.size == cbuf.size && binding.map_serial == map_serial &&
                binding.device_addr != 0) [[likely]] {
                return;
            }
            const std::optional<DAddr> device_addr =
                (binding.gpu_addr == gpu_addr && binding.map_serial == map_serial &&
                 binding.device_addr != 0)
                    ? std::optional<DAddr>{binding.device_addr}
                    : gpu_memory->GpuToCpuAddress(gpu_addr);
            if (device_addr) {
                if (!channel_state->has_deleted_buffers && binding.buffer_id &&
                    binding.buffer_id != NULL_BUFFER_ID &&
                    binding.device_addr == *device_addr && binding.size == cbuf.size) {
                    binding.gpu_addr = gpu_addr;
                    binding.map_serial = map_serial;
                    return;
                }
                binding.device_addr = *device_addr;
                binding.gpu_addr = gpu_addr;
                binding.map_serial = map_serial;
                binding.size = cbuf.size;
                binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
                return;
            }
        }
        binding = NULL_BINDING;
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        if (!binding.buffer_id) {
            binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        }
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size,
                                       bool needs_sync) {
    if (nx2_buffer_deps) {
        using Producer = BufferDependencyTrace::Producer;
        const u64 shader = nx2_buffer_deps->ActiveShader();
        const Producer kind = shader ? Producer::Compute
                             : nx2_graphics_stage >= 0 ? Producer::Graphics : Producer::Other;
        nx2_buffer_deps->Write(device_addr, size, {kind, shader, frame_tick});
    }
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    OutdateReadDownloads(device_addr, size);
    NoteReadBackWrite(device_addr, size);
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    NoteUniformWrite(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    last_synced_count = 0;
    gpu_modified_ranges.Add(device_addr, size);
    const bool ampere = Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
    if (ampere) {
        // Before QueueFenceDownload arms read traps: guest reads skip the buffer cache's lock on
        // pages without the mark (RasterizerVulkan::GetFlushArea), and the arm is what ends a
        // core's read area.
        Tegra::Nx2MarkGpuWritten(device_addr, size);
    }
    QueueFenceDownload(device_addr, size);
    if (ampere) {
        Nx2NoteListWrite(buffer_id, device_addr, size);
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    if (nx2_list_copy_demand) {
                        entry.demand.NoteWrite(nx2_list_copy_shader);
                    }
                    if (nx2_writer_trace) {
                        nx2_writer_trace->NoteGpuWrite(page_addr, entry.writes,
                                                       nx2_collect_list_copies);
                    }
                    ++entry.writes;
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU, "Command list page {:#x} is written by draws too: its "
                                         "lists wait for the work before the last host sync",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    if (nx2_list_copy_requests.empty() ||
                        nx2_list_copy_requests.back().second != page_addr) {
                        nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                    }
                }
                if (nx2_graphics_stage >= 0 && maxwell3d != nullptr) {
                    Tegra::Nx2MarkGraphicsWriterShader(
                        maxwell3d->regs.pipelines[nx2_graphics_stage].reserved[1]);
                }
                if (Tegra::Nx2CmdPageDiag::g_enabled) {
                    Tegra::Nx2CmdPageDiag::NoteWrite(page_addr, device_addr, size,
                                                     !nx2_collect_list_copies);
                }
            }
            ++page;
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const DAddr page = nx2_list_copy_requests[i].second;
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            if (nx2_list_copy_demand) {
                std::scoped_lock lock{nx2_list_mutex};
                const auto it = nx2_list_pages.find(page);
                if (it != nx2_list_pages.end() && !it->second.demand.ShouldCopy()) {
                    if (nx2_list_copy_log) {
                        ++nx2_list_copy_stats.demand_skipped;
                    }
                    continue;
                }
            }
            const u64 page_index = page >> CACHING_PAGEBITS;
            if (page_index >= page_table.size()) {
                continue;
            }
            const BufferId live_id = page_table[page_index];
            if (!live_id) {
                continue;
            }
            Buffer& buffer = slot_buffers[live_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            // A copy recorded earlier in the open tick is read only once the tick is done, after
            // this one: copy into its buffer again instead of allocating a fresh staging buffer.
            std::optional<Nx2Staging> open_copy;
            {
                std::scoped_lock lock{nx2_list_mutex};
                const auto it = nx2_list_pages.find(page);
                if (it != nx2_list_pages.end() && it->second.copy &&
                    it->second.copy_tick == tick) {
                    open_copy = it->second.copy;
                }
            }
            const Nx2Staging staging =
                open_copy ? *open_copy : runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy && !open_copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            if (nx2_writer_trace) {
                nx2_writer_trace->RegisterCopy(reinterpret_cast<uintptr_t>(staging.mapped_span.data()),
                                               page, entry.writes, static_cast<u32>(begin - page),
                                               static_cast<u32>(end - page));
            }
            entry.copy = staging;
            entry.copy_read = false;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            if (open_copy && begin - page <= entry.copy_end && end - page >= entry.copy_begin) {
                entry.copy_begin = (std::min)(entry.copy_begin, static_cast<u32>(begin - page));
                entry.copy_end = (std::max)(entry.copy_end, static_cast<u32>(end - page));
            } else {
                entry.copy_begin = static_cast<u32>(begin - page);
                entry.copy_end = static_cast<u32>(end - page);
            }
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            if (nx2_writer_trace) {
                std::scoped_lock lock{nx2_list_mutex};
                if (nx2_writer_trace->CompleteCopy(
                        reinterpret_cast<uintptr_t>(retired.second.mapped_span.data()),
                        retired.second.mapped_span)) {
                    for (const auto& [shader, s] : nx2_writer_trace->ShaderStats()) {
                        LOG_INFO(HW_GPU,
                                 "nx2 list writer bytes: shader {:016x}: {} completed page copies, "
                                 "{} baselines, {} unchanged, {} changed ({} adjacent declared "
                                 "compute epochs, {} with recent sampled input; {} mixed), "
                                 "{} incomparable, {} out of order; {} front-read copies, "
                                 "{} copies with changed read bytes, {} adjacent, {} recent; {} trace drops",
                                 shader, s.completed, s.baselines, s.unchanged, s.changed,
                                 s.adjacent_changed, s.recent_changed, s.mixed_changed,
                                 s.incomparable, s.out_of_order, s.read_copies, s.read_changed,
                                 s.adjacent_read_changed, s.recent_read_changed,
                                 nx2_writer_trace->Dropped());
                    }
                }
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync,
                                     u64* pending_tick) -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        u64 tick{};
        u64 writes{};
        u64 copy_time{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                if (nx2_list_copy_log) {
                    ++nx2_list_copy_stats.misses;
                }
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            if (entry.draw_written) {
                // EDEN_NX2_CMDPAGE_DRAIN=1: drain and flush for every list on such pages.
                static const bool always_drain = std::getenv("EDEN_NX2_CMDPAGE_DRAIN") != nullptr;
                if (always_drain) {
                    return Nx2ListRead::Unknown;
                }
                if (!past_sync) {
                    return Nx2ListRead::NeedsSync;
                }
                // Past the sync the back noted every write the list can see: a copy made after
                // the last of them holds the page, without a flush and Scheduler::Finish.
            }
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced.
            if (entry.synced_writes == entry.writes) {
                return Nx2ListRead::Synced;
            }
            if (nx2_list_copy_demand && entry.demand.Request() && nx2_list_copy_log) {
                ++nx2_list_copy_stats.demand_promoted;
            }
            if (!current(entry)) {
                if (nx2_list_copy_log) {
                    ++nx2_list_copy_stats.misses;
                }
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                if (nx2_list_copy_log) {
                    entry.copy_read = true;
                    ++nx2_list_copy_stats.copied;
                }
                if (nx2_writer_trace) {
                    nx2_writer_trace->NoteFrontRead(
                        reinterpret_cast<uintptr_t>(entry.copy->mapped_span.data()), offset, out.size());
                }
                std::memcpy(out.data(), entry.copy->mapped_span.data() + offset, out.size());
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
            if (pending_tick != nullptr) {
                *pending_tick = tick;
                return Nx2ListRead::Pending;
            }
            copy_time = Nx2Profile::Enabled() ? Nx2Profile::Now() : 0;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes). The
        // back can submit its open tick itself; front readers must ask its owner to do so.
        {
            Nx2Profile::Scope profile_scope{Nx2Profile::Zone::ListCopyWait};
            if (GpuSplit::OnBackThread()) {
                runtime.Wait(tick);
            } else {
                runtime.WaitSubmittedTick(tick);
            }
        }
        Nx2Profile::AddListCopyDone(copy_time);
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            if (nx2_list_copy_log) {
                ++nx2_list_copy_stats.misses;
            }
            return Nx2ListRead::Unknown;
        }
        if (nx2_list_copy_log) {
            it->second.copy_read = true;
            ++nx2_list_copy_stats.copied;
        }
        if (nx2_writer_trace) {
            nx2_writer_trace->NoteFrontRead(
                reinterpret_cast<uintptr_t>(it->second.copy->mapped_span.data()), offset, out.size());
        }
        std::memcpy(out.data(), it->second.copy->mapped_span.data() + offset, out.size());
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
    }
}

template <class P>
void BufferCache<P>::Nx2FlushListPages(std::span<const DAddr> pages) {
    constexpr u64 page_size = u64{1} << Tegra::Nx2WrittenPageBits;
    boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges;
    for (const DAddr page : pages) {
        ranges.emplace_back(page, page_size);
    }
    boost::container::small_vector<std::pair<DAddr, u64>, 16> drawn;
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, entry] : nx2_list_pages) {
            if (entry.draw_written && entry.synced_writes != entry.writes) {
                ranges.emplace_back(page, page_size);
                drawn.emplace_back(page, entry.writes);
            }
        }
    }
    DownloadRangesMemory(ranges);
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, writes] : drawn) {
            const auto it = nx2_list_pages.find(page);
            if (it != nx2_list_pages.end() && it->second.writes == writes) {
                it->second.synced_writes = writes;
            }
        }
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        constexpr u64 page_size = u64{1} << page_bits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const DAddr page_addr = page << page_bits;
            const auto it = nx2_list_pages.find(page_addr);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            if (nx2_list_copy_demand) {
                entry.demand.NoteWrite(0);
            }
            const bool copy_valid = entry.copy && entry.copy_writes == entry.writes;
            if (nx2_writer_trace) {
                nx2_writer_trace->NoteHostWrite(page_addr);
            }
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
            if (!data.empty() && copy_valid && runtime.IsFree(entry.copy_tick)) {
                const u64 range_start = (std::max<u64>)(device_addr, page_addr);
                const u64 range_end = (std::min<u64>)(device_addr + size, page_addr + page_size);
                if (range_start < range_end) {
                    const u64 offset_in_page = range_start - page_addr;
                    const u64 offset_in_src = range_start - device_addr;
                    const u64 copy_len = range_end - range_start;
                    if (offset_in_page >= entry.copy_begin &&
                        offset_in_page + copy_len <= entry.copy_end &&
                        offset_in_src + copy_len <= data.size()) {
                        std::memcpy(entry.copy->mapped_span.data() + offset_in_page,
                                    data.data() + offset_in_src, copy_len);
                        entry.copy_writes = entry.writes;
                    }
                }
            }
            // A write the copy didn't take (a CPU write trap carries no data) leaves the copy
            // stale: Nx2ReadListCopy then answers Unknown and the list is read the safe way. Don't
            // queue a new copy here: Nx2CopyListPages copies the host buffer, which doesn't hold
            // these CPU-written words until the next synchronization, so the copy would be marked
            // current with the GPU's older data (Mario Kart World parsed float data as commands:
            // "method 0x1 count 8063", then garbage render targets and an abort).
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible,
                                    std::source_location caller) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    // A recent result for the page, found without touching the buffer (draws look up ~20 ranges
    // each). Only a deletion takes a buffer off the page table: a bigger one joining it deletes it.
    const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
    FindMemo& memo = find_memos[page % find_memos.size()];
    const bool memo_hit = !should_sync && memo.page == page &&
                          memo.deletions == buffer_deletions && device_addr >= memo.begin &&
                          device_addr + size <= memo.end &&
                          (memo.sparse_compatible || !sparse_compatible);
    if (memo_hit && !Nx2Perf::Verify()) {
        return memo.buffer_id;
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool sparse = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                sparse = buffer.IsSparseCompatible();
            }
            if (!sparse_compatible || sparse) {
                if (memo_hit && memo.buffer_id != buffer_id) {
                    LOG_ERROR(HW_GPU, "Nx2Perf find buffer: {:#x}+{:#x} remembered in {}, found in {}",
                              device_addr, size, memo.buffer_id.index, buffer_id.index);
                }
                memo = {page, buffer_deletions, buffer.CpuAddr(),
                        buffer.CpuAddr() + buffer.SizeBytes(), buffer_id, sparse};
                return buffer_id;
            }
        }
    }
    if (memo_hit) {
        LOG_ERROR(HW_GPU, "Nx2Perf find buffer: {:#x}+{:#x} remembered in {}, not in the page table",
                  device_addr, size, memo.buffer_id.index);
    }
    return CreateBuffer(device_addr, size, sparse_compatible, caller);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    // Buffers this large that share at most one caching page with the wanted range are cut rather
    // than joined, preventing end-to-end heap blocks from snow-balling past 128 MiB -> 2 GiB.
    // Buffers created earlier in the same frame are still joined so two ranges of one draw never
    // evict each other in a loop.
    static constexpr u64 CHAIN_LIMIT = 128_MiB;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    boost::container::small_vector<BufferId, 2> cut_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    const DAddr wanted_begin = begin;
    const DAddr wanted_end = end;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .cut = std::move(cut_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        const DAddr overlap_device_addr = overlap.CpuAddr();
        if (overlap.SizeBytes() >= CHAIN_LIMIT &&
            std::ranges::find(frame_created_buffers, overlap_id) == frame_created_buffers.end()) {
            const DAddr shared_begin = (std::max)(overlap_device_addr, wanted_begin);
            const DAddr shared_end =
                (std::min)(overlap_device_addr + overlap.SizeBytes(), wanted_end);
            if (shared_end <= shared_begin + CACHING_PAGESIZE) {
                overlap.Pick();
                cut_ids.push_back(overlap_id);
                continue;
            }
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .cut = std::move(cut_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
bool BufferCache<P>::JoinCopyIsDead(DAddr device_addr, u64 size) {
    if (!memory_tracker.IsRegionWhollyCpuModified(device_addr, size) ||
        gpu_modified_ranges.Intersects(device_addr, size) ||
        uncommitted_gpu_modified_ranges.Intersects(device_addr, size)) {
        return false;
    }
    const auto intersects = [&](const Common::RangeSet<DAddr>& ranges) {
        return ranges.Intersects(device_addr, size);
    };
    return std::ranges::none_of(committed_gpu_modified_ranges, intersects) &&
           std::ranges::none_of(pending_download_ranges, intersects);
}

template <class P>
bool BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    const bool copy = !JoinCopyIsDead(overlap.CpuAddr(), overlap.SizeBytes());
    if (copy) {
        runtime.CopyBuffer(new_buffer, overlap, copies, true);
    }
#ifdef VESHIRU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
    return copy;
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size, bool sparse_compatible,
                                      std::source_location caller) {
    const StutterTrace::SiteScope site{StutterTrace::Site::BufferCreate};
    const DAddr asked_addr = device_addr;
    const u32 asked_size = wanted_size;
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    for (const BufferId cut_id : overlap.cut) {
        DownloadBufferMemory(slot_buffers[cut_id]);
        DeleteBuffer(cut_id);
    }
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    if (CreateTrace& trace = create_trace; trace.enabled) {
        u64 joined{};
        for (const BufferId overlap_id : overlap.ids) {
            joined += slot_buffers[overlap_id].SizeBytes();
        }
        ++trace.creates;
        trace.created_bytes += size;
        trace.joins += overlap.ids.size();
        trace.joined_bytes += joined;
        trace.leaps += overlap.has_stream_leap ? 1 : 0;
        if (size >= 64_MiB && trace.large_logged < 256) {
            ++trace.large_logged;
            LOG_INFO(HW_GPU,
                     "Buffer trace: created {:#x}+{:#x} ({} MiB) for {:#x}+{:#x} at line {}, joining "
                     "{} buffers ({} MiB){}",
                     overlap.begin, size, size >> 20, asked_addr, asked_size, caller.line(),
                     overlap.ids.size(), joined >> 20, overlap.has_stream_leap ? ", stream leap" : "");
        }
    }
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    frame_created_buffers.push_back(new_buffer_id);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    bool copied = false;
    bool skipped = false;
    for (const BufferId overlap_id : overlap.ids) {
        const bool joined_copy = JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
        copied |= joined_copy;
        skipped |= !joined_copy;
    }
    if (skipped && !copied) {
        if constexpr (requires { runtime.PostCopyBarrier(); }) {
            runtime.PostCopyBarrier();
        }
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    if (insert) {
        buffer_bytes += Common::AlignUp(size, 1024);
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= Common::AlignUp(size, 1024);
        total_used_memory -= Common::AlignUp(size, 1024);
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBufferOnce(TouchMemo& memo, Buffer& buffer, BufferId buffer_id) noexcept {
    // The collector goes by the frame of a buffer's last touch: one a frame is enough.
    if (memo.frame != frame_tick || memo.buffer_id != buffer_id ||
        memo.deletions != buffer_deletions) {
        memo = {frame_tick, buffer_id, buffer_deletions};
        TouchBuffer(buffer, buffer_id);
    }
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
    // An upload over a clean word of the tracker (word_manager.h) finds and changes nothing:
    // walk only the dirty ones. Verify walks every word and checks the clean words.
    const bool skip_clean = !Nx2Perf::Verify();
    if (Nx2Perf::Verify()) {
        static std::atomic<u32> logged{};
        if (!memory_tracker.CleanWordsHold(device_addr, size) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf draw-binds: clean tracker words wrong near {:#x}+{:#x}",
                      device_addr, size);
        }
    }
    const auto for_each_upload_range = [&](auto&& func) {
        if (skip_clean) {
            memory_tracker.ForEachDirtyUploadRange(device_addr, size, func);
        } else {
            memory_tracker.ForEachUploadRange(device_addr, size, func);
        }
    };
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers =
        Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end)
                return;
            const u64 range_size = end - start;
            ForgetUploadedEarlyLanded(start, range_size);  // EDEN_NX2_READ_CHECK
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        note_synced_range();
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            ForgetUploadedEarlyLanded(device_addr_out, range_size);  // EDEN_NX2_READ_CHECK
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        note_synced_range();
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
void BufferCache<P>::SynchronizeGeometryBuffer(u32 slot, BufferId buffer_id, DAddr device_addr,
                                               u32 size) {
    Buffer& buffer = slot_buffers[buffer_id];
    // The deletion count keeps a freed and reused buffer id from passing for the scanned one.
    GeometrySync& sync = geometry_syncs[slot];
    ++geometry_stats.syncs;
    if (memory_tracker.IsSynchronized(slot) && sync.buffer_id == buffer_id &&
        sync.device_addr == device_addr && sync.size == size &&
        sync.deletions == buffer_deletions) {
        ++geometry_stats.syncs_skipped;
        if (Nx2Perf::Verify() && !SynchronizeBuffer(buffer, device_addr, size)) {
            ++geometry_stats.verify_mismatches;
        }
        return;
    }
    SynchronizeBuffer(buffer, device_addr, size);
    sync = GeometrySync{
        .buffer_id = buffer_id,
        .device_addr = device_addr,
        .size = size,
        .deletions = buffer_deletions,
    };
    memory_tracker.WatchSynchronized(slot, device_addr, size);
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            VideoCommon::Nx2Hazard::CheckRead(VideoCommon::Nx2Hazard::Site::ImmediateUpload,
                                              device_addr, copy.size);
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            VideoCommon::Nx2Hazard::CheckRead(VideoCommon::Nx2Hazard::Site::MappedUpload,
                                              device_addr, copy.size);
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer) {
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        return false;
    }

    NoteUniformWrite(dest_address, copy_size);
    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);
    last_synced_count = 0;

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearReplacedDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    const StutterTrace::SiteScope site{StutterTrace::Site::BufferDownload};
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const Download& download : downloads) {
        const DAddr address = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        landing_begin = (std::min)(landing_begin, address);
        landing_end = (std::max)(landing_end, address + download.copy.size);
    }
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearReplacedDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const StutterTrace::SiteScope site{StutterTrace::Site::BufferDownload};
    // Guest memory is written once the host GPU ran the copies: the CPU writes collected meanwhile
    // keep their bytes too.
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const BufferCopy& copy : copies) {
        landing_begin = (std::min)(landing_begin, buffer.CpuAddr() + copy.src_offset);
        landing_end = (std::max)(landing_end, buffer.CpuAddr() + copy.src_offset + copy.size);
    }

    if constexpr (requires { buffer.Mapped(); }) {
        if (const auto mapped = buffer.Mapped(); !mapped.empty()) {
            runtime.FinishForHostRead();
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            for (const BufferCopy& copy : copies) {
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + copy.src_offset,
                                      mapped.data() + copy.src_offset, copy.size);
            }
            return;
        }
    }
    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        const VideoCore::DownloadLanding landing{landing_begin, landing_end};
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, copy_mapped_memory, copy.size);
        }
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            const VideoCore::DownloadLanding landing{copy_device_addr,
                                                     copy_device_addr + copy.size};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            WriteDownloadedMemory(cpu_written, copy_device_addr, immediate_buffer.data(), copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    const StutterTrace::SiteScope site{StutterTrace::Site::BufferDelete};
    ++buffer_deletions;
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    cpu_read_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](auto& bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };
    // Every channel keeps its own bindings, and a resolved uniform buffer id survives across its
    // draws. A channel left out would address whatever buffer reuses the slot next at the old
    // offset (Ocarina of Time draws from several channels: a frame of garbage constants).
    const auto scrub = [&](BufferCacheChannelInfo& channel) {
        auto& flags = channel.maxwell3d.dirty.flags;
        if (channel.index_buffer.buffer_id == buffer_id) {
            channel.index_buffer.buffer_id = BufferId{};
            flags[Dirty::IndexBuffer] = true;
        }
        for (u32 index = 0; index < channel.vertex_buffers.size(); index++) {
            auto& binding = channel.vertex_buffers[index];
            if (binding.buffer_id != buffer_id) {
                continue;
            }
            binding.buffer_id = BufferId{};
            if (&channel == channel_state) {
                UpdateVertexBufferSlot(index, binding);
            }
            flags[Dirty::VertexBuffers] = true;
            flags[Dirty::VertexBuffer0 + index] = true;
        }
        std::ranges::for_each(channel.uniform_buffers, replace);
        std::ranges::for_each(channel.storage_buffers, replace);
        std::ranges::for_each(channel.texture_buffers, replace);
        replace(channel.transform_feedback_buffers);
        scalar_replace(channel.count_buffer_binding);
        scalar_replace(channel.indirect_buffer_binding);
        replace(channel.compute_uniform_buffers);
        replace(channel.compute_storage_buffers);
        replace(channel.compute_texture_buffers);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel.dirty_uniform_buffers.fill(~u32{0});
            channel.uniform_buffer_binding_sizes.fill({});
        }
        channel.has_deleted_buffers = true;
    };
    if (channel_state != nullptr) {
        scrub(*channel_state);
    }
    for (const size_t channel_id : active_channel_ids) {
        if (&channel_storage[channel_id] != channel_state) {
            scrub(channel_storage[channel_id]);
        }
    }

    if (!native_bank_pages.empty()) {
        const Buffer& deleted = slot_buffers[buffer_id];
        const DAddr first_page = Common::AlignDown(deleted.CpuAddr(), Core::DEVICE_PAGESIZE);
        const DAddr end_page =
            Common::AlignUp(deleted.CpuAddr() + deleted.SizeBytes(), Core::DEVICE_PAGESIZE);
        const u64 num_pages = (end_page - first_page) >> Core::DEVICE_PAGEBITS;
        if (num_pages <= native_bank_pages.size()) {
            for (DAddr page = first_page; page < end_page; page += Core::DEVICE_PAGESIZE) {
                native_bank_pages.erase(page);
            }
        } else {
            for (auto it = native_bank_pages.begin(); it != native_bank_pages.end();) {
                if (it->first >= first_page && it->first < end_page) {
                    it = native_bank_pages.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    // Adapted from the pre-refresh branch: merged buffers invalidate segment references too.
    InvalidateMultiRangeSegments(std::span{graphics_segments}, buffer_id);
    InvalidateMultiRangeSegments(std::span{compute_segments}, buffer_id);

    // The other channels keep their own bindings, and the resolved buffer ids survive across
    // draws there. Left alone, a stale id would address whatever buffer reuses this slot next
    // (Ocarina of Time draws from several channels: descriptors ended up ~100 MiB past 128 KiB
    // buffers and the GPU faulted).
    for (const size_t other_id : active_channel_ids) {
        BufferCacheChannelInfo& other = channel_storage[other_id];
        if (&other == channel_state) {
            continue;
        }
        bool other_dirty_index = false;
        if (other.index_buffer.buffer_id == buffer_id) {
            other.index_buffer.buffer_id = BufferId{};
            other_dirty_index = true;
        }
        auto& other_flags = other.maxwell3d.dirty.flags;
        for (u32 index = 0; index < other.vertex_buffers.size(); index++) {
            if (other.vertex_buffers[index].buffer_id == buffer_id) {
                other.vertex_buffers[index].buffer_id = BufferId{};
                other_flags[Dirty::VertexBuffers] = true;
                other_flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
        if (other_dirty_index) {
            other_flags[Dirty::IndexBuffer] = true;
        }
        std::ranges::for_each(other.uniform_buffers, replace);
        other.unresolved_uniform_buffers.fill(~u32{0});
        std::ranges::for_each(other.storage_buffers, replace);
        replace(other.transform_feedback_buffers);
        replace(other.compute_uniform_buffers);
        replace(other.compute_storage_buffers);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            other.dirty_uniform_buffers.fill(~u32{0});
            other.uniform_buffer_binding_sizes.fill({});
        }
        other.has_deleted_buffers = true;
    }

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        last_synced_count = 0;
    }

    Unregister(buffer_id);
    if (buffer_id == inline_buffer_id) {
        inline_buffer_id = NULL_BUFFER_ID;
    }

#ifdef VESHIRU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);
}

template <class P>
void BufferCache<P>::TraceTablePointer(bool compute, size_t stage, size_t ssbo_index,
                                       u32 cbuf_index, u32 cbuf_offset, u32 indirect_offset,
                                       GPUVAddr holder, const Binding& binding) {
    // A table entry is read here from guest memory, and again by the shader from the holder's
    // host buffer. Logs each change of a slot's entry or size, and whether the two copies of the
    // entry can differ: GPU-written (host newer) or CPU-written and not yet uploaded (guest newer).
    static const bool enabled = std::getenv("EDEN_NX2_TABLE_POINTER_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    const GPUVAddr entry_addr = holder + indirect_offset;
    const GPUVAddr entry = gpu_memory->Read<u64>(entry_addr);
    const u64 key = (u64{compute} << 63) | (u64{stage & 7} << 60) | (u64{ssbo_index & 63} << 54) |
                    (u64{cbuf_index & 31} << 49) | (u64{cbuf_offset & 0x7ffff} << 30) |
                    (indirect_offset & 0x3fffffff);
    const std::pair<GPUVAddr, u32> state{entry, binding.size};
    const auto [it, inserted] = traced_table_pointers.try_emplace(key, state);
    if (!inserted && it->second == state) {
        return;
    }
    it->second = state;
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    if (++lines > max_lines) {
        if (lines == max_lines + 1) {
            LOG_INFO(HW_GPU, "Table pointer trace stopped after {} lines", max_lines);
        }
        return;
    }
    const std::optional<DAddr> entry_device = gpu_memory->GpuToCpuAddress(entry_addr);
    const bool gpu_modified =
        entry_device && memory_tracker.IsRegionGpuModified(*entry_device, sizeof(u64));
    const bool cpu_modified =
        entry_device && memory_tracker.IsRegionCpuModified(*entry_device, sizeof(u64));
    LOG_INFO(HW_GPU,
             "Table pointer frame={} {} stage={} ssbo={} c{}[{:#x}]+{:#x}: holder={:#x} "
             "entry={:#x} bound={:#x} size={:#x} entry_gpu_modified={} entry_cpu_modified={}",
             frame_tick, compute ? "compute" : "graphics", stage, ssbo_index, cbuf_index,
             cbuf_offset, indirect_offset, holder, entry, binding.device_addr, binding.size,
             gpu_modified, cpu_modified);
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer, bool table_pointer,
                                             bool compute) const {
    const GPUVAddr gpu_addr = gpu_memory->Read<u64>(ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    const bool honor_written_size =
        native_pointer && compute && is_written &&
        Settings::GetCurrentProgramID() == Settings::KirbyAirRidersProgramId;
    const bool reads_size_word =
        !(table_pointer && !is_written) &&
        (!native_pointer || (compute && (!is_written || honor_written_size)));
    const u64 next_qword = reads_size_word ? gpu_memory->Read<u64>(ssbo_addr + 8) : 0;
    const u32 alignment = runtime.GetStorageBufferAlignment();
    struct CachedBinding {
        size_t owner;
        GPUVAddr gpu_addr;
        u64 next_qword;
        u64 serial;
        u32 alignment;
        u32 flags;
        Binding binding;
    };
    constexpr size_t ways = 4;
    thread_local std::array<std::array<CachedBinding, ways>, 512> binding_cache{};
    const u32 flags =
        (is_written ? 1U : 0U) | (native_pointer ? 2U : 0U) | (table_pointer ? 4U : 0U) |
        (compute ? 8U : 0U) | (honor_written_size ? 16U : 0U);
    const size_t owner = gpu_memory->GetID() + 1;
    const auto generation = gpu_memory->StableMappingGeneration();
    const u64 serial = generation.value_or(gpu_memory->MapSerial());
    auto& set = binding_cache[((gpu_addr ^ next_qword ^ flags) * 0x9E3779B97F4A7C15ULL) >> 55];
    for (size_t way = 0; generation && way < ways; ++way) {
        const CachedBinding& cached = set[way];
        if (cached.owner == owner && cached.gpu_addr == gpu_addr &&
            cached.next_qword == next_qword && cached.serial == serial &&
            cached.alignment == alignment && cached.flags == flags && !Nx2Perf::Verify()) {
            const Binding hit = cached.binding;
            std::rotate(set.begin(), set.begin() + way, set.begin() + way + 1);
            return hit;
        }
    }
    const Binding binding = [&]() -> Binding {
        u32 size = [&]() {
            const u32 memory_layout_size = static_cast<u32>((std::min)(
                u64{gpu_memory->GetMemoryLayoutSize(gpu_addr)}, u64{(std::numeric_limits<u32>::max)()}));
            const u32 packed_size = static_cast<u32>(next_qword);
            const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                            packed_size != 0 && packed_size <= memory_layout_size;
            const u64 reach = (std::min)(runtime.GetMaxStorageBufferRange(), u64{1_GiB});
            const u32 limit = static_cast<u32>(reach - (std::min)(reach, u64{64_KiB}));
            // A pointer read from a table in memory is the base of an object pool, read anywhere in
            // it: bound to the end of its mapping, as far as a storage buffer may reach (at most
            // 1 GiB). Ocarina of Time's grass instances lie 12-18 MiB past theirs (an 8 MiB cap
            // culled every blade), and its light probe records reach about 130 MiB past theirs: a
            // 64 MiB cap left the far probes outside the binding, and the fog took one overbright
            // constant from them. Written pools keep a cap: a write marks the whole binding
            // GPU-modified.
            if (table_pointer && !is_written) {
                return (std::min)(memory_layout_size, limit);
            }
            // A native compute program's pointer is bare: the word after it is another parameter
            // (a dimension, stride or texture handle), not an NVN buffer size. Taking it as one
            // bound Fortune's Weave's DLSS input features to 266 bytes and dropped the rest.
            if (native_pointer) {
                const u32 bare = (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
                // Mario Kart World's lighting passes read past 8 MiB from theirs, inside the size
                // word after the pointer (Eglantine #609 reads it): capped at 8 MiB those reads
                // came back zero and lit land and characters too dark. Kirby Air Riders writes into
                // a 64 MiB compute particle pool sized by the same word.
                if (compute && (!is_written || honor_written_size) && next_qword_is_size &&
                    packed_size > bare) {
                    return (std::min)(packed_size, limit);
                }
                return bare;
            }
            if (next_qword_is_size) {
                return packed_size;
            }
            return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
        }();
        if ((native_pointer || table_pointer) && size != 0) {
            // Stop where the mapping or its device range does: the binding is one device range, and
            // a written one isn't split into separate ranges without sparse sources.
            const u32 wanted = size;
            size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, size));
            if (native_pointer && compute && NativeBindingStatsEnabled()) {
                ReportNativeBindingStats(size, wanted);
            }
        }

        // Alignment only applies to the offset of the buffer
        const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
        const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

        const std::optional<DAddr> aligned_device_addr =
            gpu_memory->GpuToCpuAddress(aligned_gpu_addr);
        if (!aligned_device_addr || size == 0) {
            LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
            return NULL_BINDING;
        }
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
                   cbuf_index);
        // The end address used for size calculation does not need to be aligned
        const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

        u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
        if (is_written) {
            binding_size = aligned_size;
        }
        return Binding{
            .device_addr = *aligned_device_addr,
            .gpu_addr = aligned_gpu_addr,
            .size = binding_size,
            .buffer_id = BufferId{},
        };
    }();
    if (generation) {
        std::rotate(set.begin(), set.end() - 1, set.end());
        set[0] = {owner, gpu_addr, next_qword, serial, alignment, flags, binding};
    }
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
