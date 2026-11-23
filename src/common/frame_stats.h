// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>

#include "common/common_types.h"

/// Frame-time statistics for the Switch 2 paths (legacy GPU emulation and NVN HLE).
///
/// Off unless `EDEN_NX2_FRAME_STATS` is set (anything but `0`); then every guest present calls
/// `OnPresent` and the subsystems add the time they spent stalling the frame with `AddStall` or
/// `Scope`. A value other than `1` is also the path of a CSV with one line per present
/// (`present,t_ms,dt_us,<cause us...>`). Every 600 presents one log line gives the present-to-present
/// percentiles, the counts over 16.7/33/50/100 ms and, for the frames over 33 ms, which causes
/// were active and for how long.
namespace Common::FrameStats {

enum class Cause : u32 {
    GuestAcquireWait, ///< HLE: guest blocked in AcquireTexture
    GuestSyncWait,    ///< HLE: guest blocked in SyncWait / QueueFinish / EventWait
    GuestPace,        ///< speed limiter / vblank pacing on the present thread
    LegacyFence,      ///< legacy: present thread waiting for the GPU thread's fence
    LegacyComposite,  ///< legacy: composite on the present thread
    PipelineCompile,  ///< replay: pipeline translation/creation on the critical path
    PoolBacking,      ///< replay: host import / device-local backing
    RawTableRebuild,  ///< raw global address table: replay rebuild; legacy GPU build of the guest-memory
                      ///< ranges, including the wait for their imports (RawGuestMemory::Ranges)
    TextureUpload,    ///< replay: texture upload, unswizzle, ASTC decode
    CoherencyCheck,   ///< replay: guest-write detection (signatures, hashes)
    HeldBack,         ///< replay: submission parked on a missing fence
    ReplayExecute,    ///< replay: ExecuteSubmission (recording) busy time
    QueueSubmit,      ///< replay: vkQueueSubmit
    PresentBlit,      ///< replay: present recording, acquire, submit
    PresentIdle,      ///< replay: waiting for the next frame's work
    PresentAcquire,   ///< replay: AcquireFrame (waiting for a free present frame); inside present_blit
    PresentHandOver,  ///< replay: PresentFrame (swapchain acquire, blit submit, vkQueuePresent when
                      ///< not on the present thread); inside present_blit
    Count,
};

constexpr std::array<const char*, static_cast<size_t>(Cause::Count)> CauseNames{
    "guest_acquire", "guest_syncwait", "guest_pace",     "legacy_fence", "legacy_composite",
    "pipeline",      "pool_backing",   "raw_table",      "tex_upload",   "coherency",
    "held_back",     "replay_exec",    "queue_submit",   "present_blit", "present_idle",
    "present_acquire", "present_handover",
};

namespace Detail {
extern const bool g_enabled;
extern std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> g_stall_ns;
} // namespace Detail

[[nodiscard]] inline bool Enabled() {
    return Detail::g_enabled;
}

inline void AddStall(Cause cause, u64 ns) {
    if (Detail::g_enabled) {
        Detail::g_stall_ns[static_cast<size_t>(cause)].fetch_add(ns, std::memory_order_relaxed);
    }
}

/// Adds the lifetime of the object to `cause`.
class Scope {
public:
    explicit Scope(Cause cause_) : cause{cause_}, active{Detail::g_enabled} {
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~Scope() {
        if (active) {
            AddStall(cause, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - start)
                                                 .count()));
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Cause cause;
    bool active;
    std::chrono::steady_clock::time_point start{};
};

/// Called by the guest present on either path (the frame boundary). No-op when disabled.
void OnPresent(std::chrono::steady_clock::time_point now);
/// Guest presents observed by this instance, for request/event logs.
[[nodiscard]] u64 PresentCount();

/// The `stats` diagnostic request: logs the current window (without resetting it), the stall causes
/// since start and a frame-time histogram, and writes them to `<directory>/frame-stats.txt`. No-op
/// when the statistics are off.
void DumpNow(const std::filesystem::path& directory);

} // namespace Common::FrameStats
