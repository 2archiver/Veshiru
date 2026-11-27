// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <thread>
#if defined(ARCHITECTURE_x86_64)
#include <immintrin.h>
#endif

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "common/assert.h"
#include "common/dkb_cpu_trace.h"
#include "common/game_presets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

namespace {

/// Switch 2: start a command list no sooner than Nx2GpuLatency::Latency() after the game
/// submitted it, or once a game thread waits on the GPU (nx2_gpu_latency.h).
template <typename StopToken>
void HoldForGpuLatency(const SubmitListCommand& list, const StopToken& stop_token) {
    const auto latency = Nx2GpuLatency::Latency();
    if (latency.count() <= 0 || !Nx2GpuLatency::Steady() ||
        !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    const auto until = list.submit_time + latency;
    for (auto now = std::chrono::steady_clock::now(); now < until;
         now = std::chrono::steady_clock::now()) {
        if (Nx2GpuLatency::GuestWaits() != list.guest_waits || stop_token.stop_requested()) {
            return;
        }
        // Windows sleeps a millisecond or more at a time: sleep only while that much is left.
        if (until - now > std::chrono::microseconds{1500}) {
            std::this_thread::sleep_for(std::chrono::microseconds{500});
        } else {
            std::this_thread::yield();
        }
    }
}

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

// pending_wake: the GPU thread's pending sleeps, logged per second under EDEN_NX2_PERF_VERIFY or
// EDEN_NX2_GPU_PROFILE (GPU thread only).
struct PendingWakeStats {
    u64 waits{}, by_push{}, by_latch{}, timeouts{}, stale{}, skipped{};
    std::array<u32, 24> latency{}; ///< Latch to resume, bucket k: under 2^k us
    u32 latency_samples{};
    s64 window_start_ns{};
};

void NotePendingLatchResume(PendingWakeStats& stats, s64 latch_ns) {
    const s64 us = (Tegra::Nx2PendingWakeState::NowNs() - latch_ns) / 1000;
    const size_t bucket = std::min<size_t>(
        us <= 0 ? 0 : static_cast<size_t>(std::bit_width(static_cast<u64>(us))),
        stats.latency.size() - 1);
    ++stats.latency[bucket];
    ++stats.latency_samples;
}

void ReportPendingWake(PendingWakeStats& stats) {
    const s64 now = Tegra::Nx2PendingWakeState::NowNs();
    if (stats.window_start_ns == 0) {
        stats.window_start_ns = now;
        return;
    }
    if (now - stats.window_start_ns < 1'000'000'000) {
        return;
    }
    u64 p50_us = 0;
    for (u32 k = 0, seen = 0; k < stats.latency.size() && stats.latency_samples != 0; ++k) {
        seen += stats.latency[k];
        if (seen * 2 >= stats.latency_samples) {
            p50_us = u64{1} << k;
            break;
        }
    }
    LOG_INFO(HW_GPU,
             "Nx2Perf pending_wake: {} sleeps/s ended by push {}, latch {}, timeout {}, stale {} "
             "({} skipped before the wait); latch to resume p50 under {} us ({} samples); "
             "{} resumed unmet so far",
             stats.waits, stats.by_push, stats.by_latch, stats.timeouts, stats.stale, stats.skipped,
             p50_us, stats.latency_samples,
             Tegra::Nx2PendingWakeState::verify_unmet.load(std::memory_order_relaxed));
    stats = {};
    stats.window_start_ns = now;
}

/// pending_wake's form of the pending 1 ms sleep: ends early on a command queued or a latch
/// since the poll that saw `push_seen` and `latch_seen`.
bool PendingWakeSleep(PendingWakeStats& stats, u64 push_seen, u64 latch_seen) {
    namespace Wake = Tegra::Nx2PendingWakeState;
    namespace Profile = VideoCommon::Nx2Profile;
    Wake::wake_event.Reset();
    Wake::wake_reasons.store(0, std::memory_order_relaxed);
    Wake::front_sleeping.store(true, std::memory_order_seq_cst);
    u32 reason = 0;
    if (Wake::push_seq.load(std::memory_order_seq_cst) != push_seen) {
        reason |= Wake::ByPush;
    }
    if (Wake::latch_seq.load(std::memory_order_seq_cst) != latch_seen) {
        reason |= Wake::ByLatch;
    }
    const bool skipped = reason != 0;
    bool woke = skipped;
    if (!skipped) {
        woke = Wake::wake_event.WaitFor(std::chrono::milliseconds{1});
        reason = woke ? Wake::wake_reasons.load(std::memory_order_relaxed) : 0;
    }
    Wake::front_sleeping.store(false, std::memory_order_seq_cst);
    // Verify checks the resumes the next polls make.
    Wake::verify_resume = reason != 0 && VideoCommon::Nx2Perf::Verify();

    const bool by_push = (reason & Wake::ByPush) != 0;
    const bool by_latch = !by_push && (reason & Wake::ByLatch) != 0;
    if (by_latch && !skipped) {
        const s64 latch_ns = Wake::last_latch_ns.exchange(0, std::memory_order_relaxed);
        if (latch_ns != 0) {
            NotePendingLatchResume(stats, latch_ns);
        }
    }
    ++stats.waits;
    stats.by_push += by_push ? 1 : 0;
    stats.by_latch += by_latch ? 1 : 0;
    stats.timeouts += woke ? 0 : 1;
    stats.stale += woke && reason == 0 ? 1 : 0;
    stats.skipped += skipped ? 1 : 0;
    Profile::Add(Profile::Counter::PendingWaits, 1);
    Profile::Add(Profile::Counter::PendingWakePush, by_push ? 1 : 0);
    Profile::Add(Profile::Counter::PendingWakeLatch, by_latch ? 1 : 0);
    Profile::Add(Profile::Counter::PendingWakeTimeout, woke ? 0 : 1);
    if (VideoCommon::Nx2Perf::Verify() || Profile::Enabled()) {
        ReportPendingWake(stats);
    }
    return woke;
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuCommand};
        const VideoCommon::StutterTrace::ThreadScope stutter_thread{
            VideoCommon::StutterTrace::Thread::Front};
        Common::SetCurrentThreadName("GPU");
        VideoCommon::GpuSplit::MarkGpuThread();
#ifdef __linux__
        // The pending-channel waits below sleep 500 us; the default timer slack adds up to 50 more.
        prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        if (Settings::values.pin_gpu_threads.GetValue()) {
            Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        }
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the waits below).
        auto last_progress = std::chrono::steady_clock::now();
        // For this long after that the thread waits in hardware, then it blocks in the OS. In
        // hardware it looks at the words the channels wait on every WordCheckPeriod and polls the
        // channels in full every PendingPollPeriod.
        constexpr std::chrono::nanoseconds PendingPoll = std::chrono::microseconds{500};
        constexpr std::chrono::nanoseconds PendingPollPeriod = std::chrono::microseconds{20};
        constexpr std::chrono::nanoseconds WordCheckPeriod = std::chrono::microseconds{2};
        // How the last wait ended, counted when a channel moves after it.
        enum class WaitEnd { None, Woken, PollTimedOut, SleepTimedOut };
        WaitEnd last_wait = WaitEnd::None;
        auto& wake = Tegra::Nx2PendingWake::Get();
        auto& waiters = Tegra::Nx2SemaphoreWaiters::Get();
        auto& acquire_stats = VideoCommon::g_acquire_stats;
        PendingWakeStats pending_wake_stats;
        namespace Profile = VideoCommon::Nx2Profile;
        auto& split = VideoCommon::GpuSplit::Get();
        namespace BackIdleTrace = VideoCommon::BackIdleTrace;
        using FrontState = BackIdleTrace::FrontState;
        while (!stop_token.stop_requested()) {
            Profile::Report();
            split.MaybeStart(system);
            if (VideoCommon::g_acquire_counting) {
                split.ReportAcquire();
            }
            if (scheduler.HasPendingChannels()) {
                const u64 push_seen = Tegra::Nx2PendingWakeState::push_seq.load(
                    std::memory_order_seq_cst);
                const u64 latch_seen = Tegra::Nx2PendingWakeState::latch_seq.load(
                    std::memory_order_seq_cst);
                // Suspended channels wait on semaphores and list copies: a new command, a GPU
                // release or a finished copy wakes the wait below at once (Nx2PendingWake), so
                // arm it before looking.
                wake.Arm();
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                const bool popped = state.queue.TryPop(next);
                if (popped) {
                    BackIdleTrace::SetFrontState(FrontState::Parse);
                }
                if (Tegra::Nx2PendingWakeState::verify_resume) {
                    Tegra::Nx2PendingWakeState::verify_resume = false;
                    if (!resumed && !popped) {
                        Tegra::Nx2PendingWakeState::verify_unmet.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
                if (resumed || popped) {
                    if (last_wait == WaitEnd::PollTimedOut) {
                        VideoCommon::Count(acquire_stats.front_found_polling);
                    } else if (last_wait == WaitEnd::SleepTimedOut) {
                        VideoCommon::Count(acquire_stats.front_found_late);
                    }
                    last_wait = WaitEnd::None;
                    last_progress = std::chrono::steady_clock::now();
                }
                if (!popped) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    if (BackIdleTrace::Enabled()) {
                        BackIdleTrace::SetFrontState(scheduler.WaitsForListCopy()
                                                         ? FrontState::PendingCopy
                                                         : FrontState::PendingSemaphore);
                    }
                    // Mario Kart World's channels wait for semaphores its CPU releases, stores
                    // that send no wake, and the CPU usually releases soon after: for half a
                    // millisecond wait in hardware, watching the semaphore words, then block in
                    // the OS for 500 us at a time. A full poll every 20 us catches what sends no
                    // wake and changes no word, such as a release a shader wrote to the caches.
                    // Ocarina of Time waits here ~330 times a second and its present waits for the
                    // work behind them: with 1 ms sleeps the present waited ~17 ms a frame in
                    // Kokiri village, with 50 us ~1 ms.
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    split.SettleIdleBack();
                    const auto now = std::chrono::steady_clock::now();
                    bool woken = false;
                    if (now - last_progress < PendingPoll) {
                        VideoCommon::Count(acquire_stats.front_polls);
                        last_wait = WaitEnd::PollTimedOut;
                        const auto poll_end =
                            std::min(now + PendingPollPeriod, last_progress + PendingPoll);
                        auto until = now;
                        do {
                            until = std::min(std::chrono::steady_clock::now() + WordCheckPeriod,
                                             poll_end);
                            if (wake.Poll(until)) {
                                woken = true;
                                break;
                            }
                            if (waiters.WordStored()) {
                                VideoCommon::Count(acquire_stats.front_stored);
                                last_wait = WaitEnd::Woken;
                                break;
                            }
                        } while (until < poll_end);
                    } else if (Settings::GetCurrentProgramID() ==
                               Settings::MarioKartWorldProgramId) {
                        // Mario Kart World releases pending channel semaphores via plain guest
                        // CPU stores that do not signal Nx2PendingWake: yielding avoids Windows
                        // timer-quantum oversleeps while still handing the core back to the OS.
                        VideoCommon::Count(acquire_stats.front_sleeps);
                        last_wait = WaitEnd::SleepTimedOut;
                        std::this_thread::yield();
                        woken = wake.Poll(std::chrono::steady_clock::now());
                    } else {
                        VideoCommon::Count(acquire_stats.front_sleeps);
                        last_wait = WaitEnd::SleepTimedOut;
                        if (Tegra::Nx2PendingWakeState::On() &&
                            !scheduler.WaitsForListCopy()) {
                            woken = PendingWakeSleep(pending_wake_stats, push_seen, latch_seen);
                        } else {
                            woken = wake.Sleep(std::chrono::microseconds{500});
                        }
                    }
                    if (woken) {
                        VideoCommon::Count(acquire_stats.front_woken);
                        last_wait = WaitEnd::Woken;
                    }
                    continue;
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                BackIdleTrace::SetFrontState(FrontState::QueueEmpty);
                state.queue.PopWait(next, stop_token);
                BackIdleTrace::SetFrontState(FrontState::Parse);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (split.Active()) {
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (!std::holds_alternative<SubmitListCommand>(next.data)) {
                    // Maps and flushes may change what the snapshot copier reads.
                    Tegra::DmaPusher::WaitForSnapshotCopies();
                }
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    // The guest may have written pages the caches don't watch since the lists
                    // before it: uniform keeps end here, in order on the back.
                    split.Defer([] { VideoCommon::Nx2FastDraw::NoteSync(); },
                                VideoCommon::GpuSplit::FastDrawNeutral);
                    Profile::Scope scope{Profile::Zone::Push};
                    if (VideoCommon::Nx2Hazard::Enabled()) {
                        split.Defer([epoch = submit_list->epoch] {
                            VideoCommon::Nx2Hazard::SetListEpoch(epoch);
                        });
                    }
                    const u64 timeline_seq = submit_list->timeline_seq;
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                    if (timeline_seq != 0) {
                        VideoCommon::Nx2Timeline::OnFront(timeline_seq);
                        split.Defer([timeline_seq] { VideoCommon::Nx2Timeline::OnBack(timeline_seq); });
                    }
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    // TickWork ends fast draw reuse itself when it runs a request.
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); },
                                VideoCommon::GpuSplit::FastDrawNeutral);
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        // GPU memory maps unmap caches (UnmapMemory downloads): the back's state.
                        split.DeferFunction(std::move(run->func));
                        split.Drain(VideoCommon::DrainSite::Map);
                    } else {
                        split.DeferFunction(std::move(run->func));
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = flush->addr, size = flush->size] {
                        rasterizer_ptr->FlushRegion(addr, size);
                    });
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    split.Defer(
                        [rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                            rasterizer_ptr->OnCacheInvalidation(addr, size);
                        },
                        VideoCommon::GpuSplit::Effect::None);
                    // It may drop programs that draws queued after it run: early MME reads
                    // past those draws check it like a noted write (MacroReadWriterSet).
                    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet)) {
                        split.NoteDeviceRange(invalidate->addr, invalidate->size);
                    }
                } else {
                    ASSERT(false);
                }
                split.Defer([state_ptr = &state, fence = next.fence, block = next.block] {
                    SignalCommand(*state_ptr, fence, block);
                }, VideoCommon::GpuSplit::FastDrawNeutral);
                split.Publish();
                last_progress = std::chrono::steady_clock::now();
                continue;
            }
            if (!std::holds_alternative<SubmitListCommand>(next.data)) {
                Tegra::DmaPusher::WaitForSnapshotCopies();
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                VideoCommon::Nx2FastDraw::NoteSync();
                Profile::Scope scope{Profile::Zone::Push};
                VideoCommon::Nx2Hazard::SetListEpoch(submit_list->epoch);
                const u64 timeline_seq = submit_list->timeline_seq;
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                VideoCommon::Nx2Timeline::OnFront(timeline_seq);
                VideoCommon::Nx2Timeline::OnBack(timeline_seq);
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
            last_progress = std::chrono::steady_clock::now();
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads) {
    PushCommand(RunCommand{std::move(func), before_later_reads}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);
    if (Tegra::Nx2PendingWakeState::On()) {
        Tegra::Nx2PendingWakeState::Wake(Tegra::Nx2PendingWakeState::push_seq,
                                         Tegra::Nx2PendingWakeState::ByPush);
    }
    Tegra::Nx2PendingWake::Get().Notify();

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    // Queued snapshot copies read the channels' memory managers, which services free next.
    Tegra::DmaPusher::WaitForSnapshotCopies();
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
