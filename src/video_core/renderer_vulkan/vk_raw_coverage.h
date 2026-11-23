// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <span>
#include <string_view>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

/// The miss rings of the raw global memory coverage diagnostic (VideoCommon::RawCoverageEnabled,
/// Shader::RawCoverage layout): two host-visible buffers the shaders reach by device address. The
/// active one collects a window of RAW_COVERAGE_WINDOW_SECONDS; then the rings swap and the old
/// one is handed out once the GPU work of its last use has completed (no wait), then cleared.
class RawCoverageRings {
public:
    struct Window {
        std::span<const u8> bytes;
        u64 dispatches{};
        u64 seconds{};
    };

    /// The queue the shaders that write the rings run on.
    class Queue {
    public:
        virtual ~Queue() = default;
        /// The tick the work being recorded now is complete at.
        [[nodiscard]] virtual u64 CurrentTick() = 0;
        /// Makes the ring writes of the work recorded so far visible to host reads once
        /// CurrentTick() is complete.
        virtual void HostReadBarrier() = 0;
        [[nodiscard]] virtual bool IsDone(u64 tick) = 0;
    };

    /// `path` names the GPU path in the start line ("" for the legacy path).
    explicit RawCoverageRings(const Device& device, MemoryAllocator& allocator,
                              std::string_view path = {});
    ~RawCoverageRings();

    RawCoverageRings(const RawCoverageRings&) = delete;
    RawCoverageRings& operator=(const RawCoverageRings&) = delete;

    /// False when the rings could not be made (no buffer device address).
    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }

    /// A table about to be used by a draw or dispatch recorded on `queue`: the device address of
    /// the active ring (it stamps the present number the shaders copy into their records).
    [[nodiscard]] u64 Use(Queue& queue);

    /// Swaps the rings when the window is over; returns the old ring once the GPU is done with it.
    /// Call Recycle after reading it.
    [[nodiscard]] const Window* Poll(std::chrono::steady_clock::time_point now, Queue& queue);
    void Recycle();

    /// At shutdown, with the GPU idle: the misses counted in the active (unfinished) window.
    [[nodiscard]] u64 TailMisses() const;

private:
    struct Ring {
        vk::Buffer buffer;
        u64 address{};
        u64 dispatches{};
        u64 last_tick{}; ///< the latest tick of a draw or dispatch that used the ring
        u32 present{~0u};
    };

    void Clear(Ring& ring);

    std::array<Ring, 2> rings;
    bool valid{};
    size_t active{};
    bool pending{};
    u64 pending_tick{};
    Window pending_window{};
    std::chrono::steady_clock::time_point window_start;
};

/// The legacy path's queue: one shader-write -> host-read barrier at the swap.
class SchedulerRawCoverageQueue final : public RawCoverageRings::Queue {
public:
    explicit SchedulerRawCoverageQueue(Scheduler& scheduler_) : scheduler{scheduler_} {}

    [[nodiscard]] u64 CurrentTick() override;
    void HostReadBarrier() override;
    [[nodiscard]] bool IsDone(u64 tick) override;

private:
    Scheduler& scheduler;
};

} // namespace Vulkan
