// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// The privacy-safe performance summary: one Info line per window that greps cleanly, so a
// before/after A/B takes minutes and no log upload is needed.
//
//   perf: window=10.0s native_fps=14.2 ft_p50=68.1 p95=94.0 p99=140.2 low1=112.0 low01=410.5
//         gt33=141 gt100=9 syncwait=87/312.4ms pipe_builds=3 genpresents=0
//
// Definitions (they must not drift; mirror them in docs/2.3-performance-review.md):
// - Frame time is the native present-to-present interval. Generated frames are excluded from
//   the frame times and counted separately in genpresents.
// - The 1% / 0.1% lows are the mean of the slowest 1% / 0.1% of frame times in the window;
//   with fewer than 1000 frames the 0.1% low is "n/a".
// - Percentiles are nearest-rank.
// - syncwait counts guest CPU reads that waited for a synchronous GPU flush (GPU::OnCPURead)
//   and the total time they waited.
//
// The window is set from the Debug > Performance Summary menu (off / 10 s / 30 s), persisted
// with the UI settings, and VESHIRU_PERF_SUMMARY=<seconds> overrides it at boot.
// Ctrl+F10 dumps one summary on demand. When off, OnPresent returns on a single branch.

namespace VideoCore::PerfSummary {

inline u32 ParseWindowEnv() {
    const char* value = std::getenv("VESHIRU_PERF_SUMMARY");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    const long seconds = std::strtol(value, nullptr, 10);
    if (seconds <= 0 || seconds > 3600) {
        return 0;
    }
    return static_cast<u32>(seconds);
}

/// Seconds per summary window; 0 disables the periodic report.
inline std::atomic<u32> g_window_seconds{ParseWindowEnv()};

/// Set by the hotkey / menu: report once at the next present, whatever the window is.
inline std::atomic<bool> g_dump_pending{};

/// Guest CPU reads that waited for a synchronous GPU flush: count and total wait.
inline std::atomic<u64> g_sync_waits{};
inline std::atomic<u64> g_sync_wait_ns{};

/// Pipelines whose build finished in the window (any kind, any thread).
inline std::atomic<u64> g_pipeline_builds{};

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, and the environment override wins over the UI setting only when set.
inline void ReadSwitch() {
    const char* value = std::getenv("VESHIRU_PERF_SUMMARY");
    if (value != nullptr && *value != '\0') {
        g_window_seconds.store(ParseWindowEnv(), std::memory_order_relaxed);
    }
}

inline void SetWindow(u32 seconds) {
    g_window_seconds.store(seconds, std::memory_order_relaxed);
}

inline u32 Window() {
    return g_window_seconds.load(std::memory_order_relaxed);
}

inline void RequestDump() {
    g_dump_pending.store(true, std::memory_order_relaxed);
}

inline void CountSyncWait(u64 ns) {
    if (g_window_seconds.load(std::memory_order_relaxed) == 0 &&
        !g_dump_pending.load(std::memory_order_relaxed)) {
        return;
    }
    g_sync_waits.fetch_add(1, std::memory_order_relaxed);
    g_sync_wait_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountPipelineBuild() {
    if (g_window_seconds.load(std::memory_order_relaxed) == 0 &&
        !g_dump_pending.load(std::memory_order_relaxed)) {
        return;
    }
    g_pipeline_builds.fetch_add(1, std::memory_order_relaxed);
}

/// One line at boot naming every performance flag this session runs with, so a log always
/// says which build and configuration it came from.
inline void LogFlags() {
    const auto on_off = [](const char* name) {
        return std::getenv(name) != nullptr ? "on" : "off";
    };
    LOG_INFO(HW_GPU,
             "perf flags: gpu_profile={} stutter_trace={} dkb_frame_trace={} read_check={} "
             "perf_verify={} dma_read_verify={} perf_summary={}s",
             on_off("EDEN_NX2_GPU_PROFILE"), on_off("EDEN_NX2_STUTTER_TRACE"),
             on_off("EDEN_DKB_FRAME_TRACE"), on_off("EDEN_NX2_READ_CHECK"),
             on_off("EDEN_NX2_PERF_VERIFY"), on_off("EDEN_NX2_DMA_READ_VERIFY"),
             g_window_seconds.load(std::memory_order_relaxed));
}

/// The native frame times of the current window. Written and read only on the GPU thread
/// (OnPresent), with no allocation per frame.
struct FrameRing {
    static constexpr std::size_t Capacity = 8192;
    std::array<float, Capacity> ms{};
    std::size_t count{};
    std::size_t next{};

    void Push(float value) {
        ms[next] = value;
        next = (next + 1) % Capacity;
        count = std::min(count + 1, Capacity);
    }

    void Clear() {
        count = 0;
        next = 0;
    }
};

inline FrameRing g_frames;
inline u32 g_generated_presents{};       ///< Generated frames queued in the window (GPU thread)
inline std::chrono::steady_clock::time_point g_window_start{};
inline std::chrono::steady_clock::time_point g_last_present{};
inline bool g_window_started{};

/// Called by the renderer once per native present, on the GPU thread, after the present is
/// queued. `generated` is the number of generated frames queued within this native frame.
inline void OnPresent(u32 generated) {
    const u32 window = g_window_seconds.load(std::memory_order_relaxed);
    const bool dump_requested = g_dump_pending.load(std::memory_order_relaxed);
    if (window == 0 && !dump_requested) {
        g_last_present = std::chrono::steady_clock::now();
        g_window_started = false;
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!g_window_started) {
        g_window_started = true;
        g_window_start = now;
        g_last_present = now;
        g_frames.Clear();
        g_generated_presents = 0;
        g_sync_waits.store(0, std::memory_order_relaxed);
        g_sync_wait_ns.store(0, std::memory_order_relaxed);
        g_pipeline_builds.store(0, std::memory_order_relaxed);
        return;
    }
    g_frames.Push(
        std::chrono::duration<float, std::chrono::milliseconds::period>(now - g_last_present)
            .count());
    g_last_present = now;
    g_generated_presents += generated;

    const auto window_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now -
                                                                                g_window_start)
                               .count();
    if (!dump_requested && window_ns < static_cast<s64>(window) * 1'000'000'000LL) {
        return;
    }
    g_dump_pending.store(false, std::memory_order_relaxed);

    const auto& ring = g_frames;
    const std::size_t frames = ring.count;
    if (frames == 0) {
        LOG_INFO(HW_GPU, "perf: window={:.1f}s no native frames recorded yet",
                 static_cast<double>(window_ns) / 1e9);
        g_window_start = now;
        return;
    }

    std::vector<float> sorted(ring.ms.begin(), ring.ms.begin() + frames);
    std::sort(sorted.begin(), sorted.end());
    const auto nearest_rank = [&](double percentile) {
        const std::size_t rank = static_cast<std::size_t>(
            std::ceil(percentile * static_cast<double>(frames)));
        return sorted[(std::max<std::size_t>)(rank, 1) - 1];
    };
    // The mean of the slowest fraction of frames, nearest-rank: the "1% low".
    const auto low = [&](double fraction) {
        const std::size_t ranked =
            static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(frames)));
        const std::size_t slowest = (std::max<std::size_t>)(ranked, 1);
        double total = 0.0;
        for (std::size_t i = frames - slowest; i < frames; ++i) {
            total += sorted[i];
        }
        return total / static_cast<double>(slowest);
    };

    const double elapsed_s = static_cast<double>(window_ns) / 1e9;
    const u64 sync_waits = g_sync_waits.exchange(0, std::memory_order_relaxed);
    const u64 sync_wait_ns = g_sync_wait_ns.exchange(0, std::memory_order_relaxed);
    const u64 pipeline_builds = g_pipeline_builds.exchange(0, std::memory_order_relaxed);
    const u32 generated_total = std::exchange(g_generated_presents, 0);

    std::size_t gt33 = 0;
    std::size_t gt100 = 0;
    for (std::size_t i = 0; i < frames; ++i) {
        gt33 += sorted[i] > 33.0 ? 1 : 0;
        gt100 += sorted[i] > 100.0 ? 1 : 0;
    }

    if (frames >= 1000) {
        LOG_INFO(HW_GPU,
                 "perf: window={:.1f}s native_fps={:.1f} ft_p50={:.1f} p95={:.1f} p99={:.1f} "
                 "low1={:.1f} low01={:.1f} gt33={} gt100={} syncwait={}/{:.1f}ms "
                 "pipe_builds={} genpresents={}",
                 elapsed_s, static_cast<double>(frames) / elapsed_s, nearest_rank(0.50),
                 nearest_rank(0.95), nearest_rank(0.99), low(0.01), low(0.001), gt33, gt100,
                 sync_waits, static_cast<double>(sync_wait_ns) / 1e6, pipeline_builds,
                 generated_total);
    } else {
        LOG_INFO(HW_GPU,
                 "perf: window={:.1f}s native_fps={:.1f} ft_p50={:.1f} p95={:.1f} p99={:.1f} "
                 "low1={:.1f} low01=n/a gt33={} gt100={} syncwait={}/{:.1f}ms "
                 "pipe_builds={} genpresents={}",
                 elapsed_s, static_cast<double>(frames) / elapsed_s, nearest_rank(0.50),
                 nearest_rank(0.95), nearest_rank(0.99), low(0.01), gt33, gt100, sync_waits,
                 static_cast<double>(sync_wait_ns) / 1e6, pipeline_builds, generated_total);
    }

    g_frames.Clear();
    g_window_start = now;
}

} // namespace VideoCore::PerfSummary
