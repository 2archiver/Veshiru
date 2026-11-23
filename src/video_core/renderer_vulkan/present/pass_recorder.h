// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// Where a presentation pass (window adapting filter, anti-aliasing, post-processing chain, FSR /
/// SGSR / FSR3 scaling, frame generation chains) records its commands.
///
/// Two modes with the same interface:
///  - Scheduler: the legacy renderer. Commands are deferred to the scheduler's worker thread,
///    ticks are the scheduler's, Finish() submits and waits for the GPU.
///  - Replay: the NVN HLE replay. It owns its command buffer, so a pass records straight into
///    the one the replay has open for the present (Bind / Unbind bracket every present). The tick
///    of the open command buffer is the sequence number of its flight; Wait(tick) blocks until
///    that flight has completed. Finish() cannot submit the open command buffer, it only waits for
///    the flights submitted before it: work recorded in the open command buffer before Finish()
///    runs before everything recorded after it, which is all the passes need from it.
///
/// Used by one thread at a time: the renderer's GPU thread in Scheduler mode, the replay thread in
/// Replay mode.
class PassRecorder {
public:
    struct ReplayHooks {
        /// Blocks until the flight with this sequence number (a submitted one) has completed on
        /// the GPU.
        std::function<void(u64 seq)> wait_completed;
        /// Blocks until every flight submitted before the one numbered `seq` has completed.
        std::function<void(u64 seq)> wait_previous;
    };

    explicit PassRecorder(Scheduler& scheduler_) noexcept : scheduler{&scheduler_} {}
    explicit PassRecorder(ReplayHooks hooks_) : hooks{std::move(hooks_)} {}

    PassRecorder(const PassRecorder&) = delete;
    PassRecorder& operator=(const PassRecorder&) = delete;

    [[nodiscard]] bool IsReplay() const noexcept {
        return scheduler == nullptr;
    }

    /// Replay mode: records into `cmdbuf_` (the command buffer of flight `seq_`) until Unbind().
    /// Objects given to KeepAlive() live until that flight has completed.
    void Bind(vk::CommandBuffer cmdbuf_, u64 seq_,
              std::vector<std::shared_ptr<void>>* keepalive_) noexcept {
        cmdbuf = cmdbuf_;
        seq = seq_;
        keepalive = keepalive_;
        bound = true;
    }

    /// Nothing is open any more: waits on earlier flights stop (the resources being released at
    /// this point are idle).
    void Unbind() noexcept {
        bound = false;
        seq = 0;
        keepalive = nullptr;
    }

    [[nodiscard]] bool IsBound() const noexcept {
        return bound;
    }

    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void Record(T&& command) {
        if (scheduler != nullptr) {
            scheduler->Record(std::forward<T>(command));
        } else {
            command(cmdbuf);
        }
    }

    /// Requests a context outside of a render pass. The replay has no render pass open while a
    /// present is recorded.
    void RequestOutsideRenderPassOperationContext() {
        if (scheduler != nullptr) {
            scheduler->RequestOutsideRenderPassOperationContext();
        }
    }

    /// Scheduler: submits and waits. Replay: waits for the flights submitted before the open one.
    void Finish() {
        if (scheduler != nullptr) {
            scheduler->Finish();
        } else if (bound && hooks.wait_previous) {
            hooks.wait_previous(seq);
        }
    }

    /// The tick the work recorded now belongs to.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return scheduler != nullptr ? scheduler->CurrentTick() : seq;
    }

    /// Waits until the work of `tick` (a CurrentTick() of earlier) has completed. In Replay mode
    /// the tick of the open flight (or none) is already ordered before whatever is recorded next.
    /// Any other tick is a flight the replay has submitted (the passes only record into open
    /// flights, and every flight is submitted before the next one is recorded), so waiting for
    /// it never waits for the replay thread itself. Flight sequence numbers do not follow the
    /// order the replay records them in (held-back submissions run later): the hooks wait for
    /// that one flight, not for every flight numbered below it.
    void Wait(u64 tick) {
        if (scheduler != nullptr) {
            scheduler->Wait(tick);
        } else if (tick != 0 && (!bound || tick != seq) && hooks.wait_completed) {
            hooks.wait_completed(tick);
        }
    }

    /// Keeps `object` alive until the GPU is done with what is recorded now. Scheduler mode:
    /// nothing to do, the callers Finish() before dropping their objects.
    void KeepAlive(std::shared_ptr<void> object) {
        if (scheduler == nullptr && keepalive != nullptr) {
            keepalive->push_back(std::move(object));
        }
    }

private:
    Scheduler* scheduler{};
    ReplayHooks hooks{};
    vk::CommandBuffer cmdbuf{};
    u64 seq{};
    std::vector<std::shared_ptr<void>>* keepalive{};
    bool bound{};
};

} // namespace Vulkan
