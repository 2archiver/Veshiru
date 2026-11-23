// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <source_location>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/present_pacing.h"
#include "video_core/nvn/recorder.h"
#include "video_core/nvn/sink_gate.h"

namespace Kernel {
class KernelCore;
class KThread;
} // namespace Kernel

namespace Core::NvnHle::Hle {

/// One row of unsupported-<title>.tsv.
struct UnsupportedRow {
    std::string where;  ///< recorder, replay or shader
    std::string reason;
    u64 count{};
};

/// Process-wide HLE state: the Recorder (created at the first HLE call, when the renderer
/// exists), the device state, per-window bookkeeping and the guest-thread wait list.
class Runtime {
public:
    /// Waits on the host in unit tests (no kernel).
    using WaitHook =
        std::function<bool(const std::function<bool()>& ready, std::chrono::nanoseconds timeout)>;

    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    static Runtime& Instance();

    /// EDEN_NX2_NVN_GPU_PROFILE is set (read once): the guest-wait report is on.
    [[nodiscard]] static bool GpuProfileEnabled();
    /// A blocking SyncWait finished (profile only): it waited for the submission `fence`, `last`
    /// was the newest submission and `completed` the newest completed one when it began.
    void NoteSyncWait(u64 fence, u64 last, u64 completed, std::chrono::nanoseconds waited);
    /// EDEN_NX2_NVN_TIMELINE is set (read once): guest-side hand-offs are logged like the replay's.
    [[nodiscard]] static bool TimelineEnabled();
    /// `[nvn] tl <us> <event> <seq> [detail]` with the replay timeline's time base (steady-clock
    /// microseconds since its epoch), so `rundiag.py timeline` reads both.
    static void TimelineNote(const char* event, u64 seq, std::string_view detail = {});

    /// Production entry from the bridge: creates the Recorder on first use and runs `handler`.
    /// `observer`: the registers are left alone when the HLE cannot run (the original function
    /// runs next).
    void Dispatch(Core::System& system, Kernel::Svc::ThreadContext& ctx, Handler handler,
                  bool observer = false);

    /// Unit tests: runs `handler` against an explicit memory and recorder.
    void Run(Kernel::Svc::ThreadContext& ctx, VideoCommon::Nvn::GuestMemory& memory,
             VideoCommon::Nvn::Recorder& recorder, Handler handler);

    /// Creates the Recorder over `target` (through a SinkGate) and `guest_memory`, and starts the
    /// status heartbeat. `kernel_` is null when there is no guest kernel (unit tests). False when a
    /// Recorder exists already (it has to be Released first). Production: the first HLE call.
    bool Start(VideoCommon::Nvn::Sink& target,
               std::unique_ptr<VideoCommon::Nvn::GuestMemory> guest_memory,
               Kernel::KernelCore* kernel_);

    /// Stops the HLE: wakes every waiting guest thread, writes nothing more and disconnects the
    /// Recorder from the sink (SinkGate::Detach), so the renderer can be destroyed while a guest
    /// thread is still inside an NVN call. The Recorder and the guest memory stay alive for those
    /// threads until Release.
    void Shutdown();
    /// Destroys the Recorder and the guest memory. Shutdown first; call it once no guest thread can
    /// enter or still be in an NVN call (the CPU threads are joined): the shutdown of the emulated
    /// process does, after the renderer is gone. Without it the next boot's ResetForNewProcess
    /// destroys them.
    void Release();
    /// Makes the runtime usable again after Shutdown (a second boot in the same process): releases
    /// what is left of the previous boot and clears every piece of per-boot state.
    void ResetForNewProcess();
    [[nodiscard]] bool IsStopped() const {
        return stopped.load(std::memory_order_acquire);
    }
    /// The host GPU reported a failure (device lost): true after OnSinkFailure ran.
    [[nodiscard]] bool HasFailed() const {
        return failed.load(std::memory_order_acquire);
    }

    /// Wakes the guest threads in Call::WaitUntil so they re-check their condition.
    void WakeWaiters();
    bool WaitUntil(Core::System* system, const std::function<bool()>& ready,
                   std::chrono::nanoseconds timeout, const std::source_location& where);
    void SetWaitHook(WaitHook hook) {
        wait_hook = std::move(hook);
    }

    DeviceState device;

    // ---- Window bookkeeping (functions_window.cpp) ---------------------------------------------
    struct WindowState {
        u32 acquired_mask{};
        u32 count{};
        u64 presents{};
        u64 native_window{};
        u32 color_space{};
        s32 present_interval{1};
        s32 crop[4]{};
        u32 first_width{};
        u32 first_height{};
        /// Display pacing (host steady-clock nanoseconds): when the last presented frame is shown,
        /// and per slot when its texture becomes free (0 = at once, ~0 = still on display).
        u64 last_display_ns{};
        s32 displayed_slot{-1};
        u64 avail_ns[4]{};
    };
    std::mutex window_mutex;
    std::map<u64, WindowState> windows;
    u64 guest_request_producer{};
    /// EDEN_NX2_MAX_FRAMES_IN_FLIGHT (functions_window.cpp; guarded by window_mutex).
    std::optional<PresentDepthLimit> present_depth;
    bool present_depth_read{};
    /// The next id nvnDeviceGenerateDebugDomainId hands out (per boot: a new process starts at 1).
    std::atomic<u32> next_debug_domain_id{1};

    // ---- Counters --------------------------------------------------------------------------------
    std::atomic<u64> presents{};
    std::atomic<u64> blocking_waits{};
    std::atomic<u64> blocking_wait_ns{};
    /// The same wait time split by call: window acquire waits and every other blocking wait
    /// (nvnSyncWait, queue finish...). Read by the EDEN_NX2_NVN_DKFRAME diagnostic.
    std::atomic<u64> acquire_wait_ns{};
    std::atomic<u64> sync_wait_ns{};
    /// EDEN_NX2_NVN_GPU_PROFILE only: blocking waits and nanoseconds by the NVN function that
    /// waited (the heartbeat prints the change since its last line).
    std::mutex wait_by_mutex;
    std::map<std::string, std::pair<u64, u64>> wait_by_function;
    /// Profile only, SyncWaits that blocked: count, sum of (newest submission - waited fence) and
    /// sum of (waited fence - newest completed) when they began.
    std::atomic<u64> sync_wait_n{};
    std::atomic<u64> sync_after_fence_sum{};
    std::atomic<u64> sync_behind_sum{};
    std::atomic<u64> wait_timeouts{};
    /// nv::InitializeGraphics was called (nv::InitCheck answers from it).
    std::atomic<bool> nv_initialized{};
    std::atomic<u64> nv_pool_size{};

    /// The rows of unsupported-<title>.tsv from the recorder, the replay and the shaders.
    [[nodiscard]] std::vector<UnsupportedRow> UnsupportedRows();

    [[nodiscard]] VideoCommon::Nvn::Recorder* GetRecorder() {
        return recorder.get();
    }

    /// A host-side service wrote [address, address + size) of guest memory the GPU may read (the
    /// video decoder writing a frame into a decoder image): the replay's copies of that memory
    /// follow (Sink::NotifyHostWrite). Callable from any thread; nothing before the first HLE call
    /// or after Shutdown.
    void NotifyHostWrite(u64 address, u64 size) {
        if (!created.load(std::memory_order_acquire) || stopped.load(std::memory_order_acquire)) {
            return;
        }
        if (VideoCommon::Nvn::Recorder* const target = recorder.get()) {
            target->NotifyHostWrite(address, size);
        }
    }

private:
    class ProcessMemory;
    bool EnsureRecorder(Core::System& system);
    /// The sink reported that the host GPU failed (device lost): log it and stop the emulation.
    void OnSinkFailure(Core::System& system, std::string reason);

    std::mutex create_mutex;
    std::atomic<bool> created{};
    std::atomic<bool> stopped{false};
    std::atomic<bool> failed{false};
    // Declaration order is destruction order backwards: the Recorder goes first (it calls the gate),
    // then the gate, then the memory both point into.
    std::unique_ptr<VideoCommon::Nvn::GuestMemory> memory;
    std::unique_ptr<VideoCommon::Nvn::SinkGate> gate;
    std::unique_ptr<VideoCommon::Nvn::Recorder> recorder;
    Kernel::KernelCore* kernel{};
    WaitHook wait_hook;
    std::jthread heartbeat;
    void HeartbeatLoop(std::stop_token token);
    /// EDEN_NX2_NVN_SYNTH_LIST_MIB as given to the Recorder (0: no cap), for the memory line.
    std::atomic<u64> synth_list_cap_mib{0};

    /// Guest threads asleep in WaitUntil. Guarded by the kernel scheduler lock.
    std::vector<Kernel::KThread*> waiters;
    std::atomic<u32> waiter_count{};
};

// Tables of the function families that are keyed by Recorder serials (which restart with every
// Recorder). Cleared when the Recorder is released or a new boot starts.
void ClearPoolTable();        // functions_pool.cpp
void ClearTextureBindings();  // functions_texture.cpp

} // namespace Core::NvnHle::Hle
