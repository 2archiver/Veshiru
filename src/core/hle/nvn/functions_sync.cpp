// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <atomic>
#include <chrono>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/frame_stats.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// Sync family: NVNsync. `nvnQueueFenceSync` and `nvnQueueWaitSync` live in functions_queue.cpp (the
// Recorder fences the sync with the sequence number of the submission).
namespace Core::NvnHle::Hle {
namespace {

// NVNsyncWaitResult.
constexpr u32 SyncAlreadySignaled = 0;
constexpr u32 SyncConditionSatisfied = 1;
constexpr u32 SyncTimeoutExpired = 2;
constexpr u32 SyncFailed = 3;

/// NVN's "wait forever" is ~0ULL; anything beyond a year is treated as infinite as well (and keeps
/// the deadline arithmetic of the waits away from overflow).
constexpr u64 InfiniteTimeoutNs = u64{365} * 24 * 3600 * 1'000'000'000ULL;

void nvnSyncInitialize(Call& c) {
    const u64 sync = c.args.NextPointer();
    static_cast<void>(c.args.NextPointer()); // device
    c.ret.SetBool(c.rec.SyncInitialize(sync));
}

void nvnSyncFinalize(Call& c) {
    c.rec.SyncFinalize(c.args.NextPointer());
}

void nvnSyncSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

/// BLOCKING. SDK (3.2.0 sdk+0x5ec180, 1.0.0 sdk+0x5c8300), in this order: a sync that was only
/// initialized (nothing fenced or signalled it) FAILS; a signalled sync (a completed fence, or the
/// sync nvnWindowAcquireTexture hands back, Recorder::SyncSetSignaled) is ALREADY_SIGNALED,
/// whatever the timeout; otherwise the wait runs to a deadline of timeout / 1000 microseconds
/// (~0 waits forever), so a timeout below 1000 ns polls once and expires at once, and a fence
/// that completes before the deadline is CONDITION_SATISFIED.
///
/// A fenced sync is signaled once the GPU completed the submission that carried the fence. A
/// waiting guest thread sleeps in the kernel until the Recorder's completion callback wakes it
/// (Runtime::WakeWaiters), so the physical core keeps running other guest threads meanwhile.
void nvnSyncWait(Call& c) {
    const u64 sync = c.args.NextPointer();
    const u64 timeout = c.args.NextU64();
    const auto state = c.rec.GetSyncState(sync);
    if (!state) {
        c.ret.SetU32(SyncFailed);
        return;
    }
    if (state->signaled) {
        c.ret.SetU32(SyncAlreadySignaled);
        return;
    }
    if (!state->fenced) {
        c.ret.SetU32(SyncFailed);
        return;
    }
    const u64 timeout_us = timeout / 1000;
    if (timeout != ~u64{0} && timeout_us == 0) {
        c.ret.SetU32(SyncTimeoutExpired);
        return;
    }
    const std::chrono::nanoseconds wait =
        timeout >= InfiniteTimeoutNs ? std::chrono::nanoseconds{-1}
                                     : std::chrono::microseconds{static_cast<s64>(timeout_us)};
    VideoCommon::Nvn::Recorder& rec = c.rec;
    const u64 fence = state->fence_seq;
    const bool profile = Runtime::GpuProfileEnabled();
    const bool stats = Common::FrameStats::Enabled();
    const auto begin = std::chrono::steady_clock::now();
    const u64 last = (profile || stats) ? rec.LastSeq() : 0;
    const u64 completed = (profile || stats) ? rec.CompletedSeq() : 0;
    bool signaled;
    const bool timeline = Runtime::TimelineEnabled();
    if (timeline) {
        Runtime::TimelineNote("guest_wait_begin", fence,
                              fmt::format("completed={}", rec.CompletedSeq()));
    }
    {
        signaled = c.WaitUntil([&rec, fence] { return rec.CompletedSeq() >= fence; }, wait);
    }
    if (timeline) {
        Runtime::TimelineNote("guest_wait_end", fence);
    }
    if (stats) {
        // Which fence the game waits for and how deep the queue is (Phase 4 diagnosis of the serial
        // CPU/GPU frame): fence == last means the game waits for everything it submitted.
        static Common::BootCounter logged;
        const auto waited_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
        if (waited_us > 2000 && logged.Next() < 400) {
            LOG_INFO(Core,
                     "[framestats] nvnSyncWait fence {} (last submitted {}, completed {}) waited {} us, "
                     "present {}",
                     fence, last, completed, waited_us,
                     Core::g_nx2_present_count.load(std::memory_order_relaxed));
        }
    }
    if (profile) {
        c.runtime.NoteSyncWait(fence, last, completed, std::chrono::steady_clock::now() - begin);
    }
    c.ret.SetU32(signaled ? SyncConditionSatisfied : SyncTimeoutExpired);
}

/// GL interop does not exist on the console: the sync stays untouched and the calls fail as they
/// do without a GL context (nxnative does not implement them either). No prototype in nvn.h.
void nvnSyncGlInteropFails(Call& c) {
    c.ret.SetU64(0);
}

} // namespace

void RegisterSyncFunctions(Registry& registry) {
    NVN_HLE(registry, nvnSyncInitialize);
    NVN_HLE(registry, nvnSyncFinalize);
    NVN_HLE(registry, nvnSyncSetDebugLabel);
    NVN_HLE(registry, nvnSyncWait);
    registry.Add<&nvnSyncGlInteropFails>("nvnSyncInitializeFromFencedGLSync");
    registry.Add<&nvnSyncGlInteropFails>("nvnSyncCreateGLSync");
}

} // namespace Core::NvnHle::Hle
