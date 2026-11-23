// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "core/core.h"
#include "shader_recompiler/raw_coverage_layout.h"
#include "video_core/buffer_cache/raw_coverage.h"
#include "video_core/renderer_vulkan/vk_raw_coverage.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

namespace Cov = Shader::RawCoverage;

RawCoverageRings::RawCoverageRings(const Device& device, MemoryAllocator& allocator,
                                   std::string_view path)
    : window_start{std::chrono::steady_clock::now()} {
    const std::string path_field{path.empty() ? std::string{} : fmt::format(" path={}", path)};
    if (!device.IsBufferDeviceAddressSupported()) {
        LOG_WARNING(HW_GPU, "[raw] coverage off: no buffer device address{}", path_field);
        return;
    }
    const VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = VideoCommon::RAW_COVERAGE_RING_BYTES,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    for (Ring& ring : rings) {
        ring.buffer = allocator.CreateBuffer(create_info, MemoryUsage::Download);
        // The GPU adds to the counter while the host stamps the present number next to it: a
        // flush of non-coherent memory would write the whole range back over the GPU's records.
        if (!ring.buffer.IsHostCoherent()) {
            LOG_WARNING(HW_GPU, "[raw] coverage off: no host-coherent readback memory{}",
                        path_field);
            return;
        }
        ring.address = device.GetLogical().GetBufferDeviceAddress(*ring.buffer);
        std::memset(ring.buffer.Mapped().data(), 0, VideoCommon::RAW_COVERAGE_RING_BYTES);
    }
    valid = true;
    LOG_INFO(HW_GPU, "[raw] coverage on: 2 rings of {} misses, window {}s{}", Cov::RING_ENTRIES,
             VideoCommon::RAW_COVERAGE_WINDOW_SECONDS, path_field);
}

RawCoverageRings::~RawCoverageRings() = default;

u64 RawCoverageRings::Use(Queue& queue) {
    Ring& ring{rings[active]};
    ++ring.dispatches;
    ring.last_tick = (std::max)(ring.last_tick, queue.CurrentTick());
    const u32 present{static_cast<u32>(Core::g_nx2_present_count.load(std::memory_order_relaxed))};
    if (present != ring.present) {
        ring.present = present;
        std::memcpy(ring.buffer.Mapped().data() + Cov::RING_PRESENT_WORD * sizeof(u32), &present,
                    sizeof(present));
    }
    return ring.address;
}

const RawCoverageRings::Window* RawCoverageRings::Poll(std::chrono::steady_clock::time_point now,
                                                       Queue& queue) {
    if (pending) {
        return queue.IsDone(pending_tick) ? &pending_window : nullptr;
    }
    const auto elapsed{std::chrono::duration_cast<std::chrono::seconds>(now - window_start)};
    if (elapsed.count() < static_cast<s64>(VideoCommon::RAW_COVERAGE_WINDOW_SECONDS)) {
        return nullptr;
    }
    // The shaders' writes to the old ring become visible to the host once this completes (and
    // every earlier use of the ring has).
    queue.HostReadBarrier();
    Ring& old{rings[active]};
    pending = true;
    pending_tick = (std::max)(old.last_tick, queue.CurrentTick());
    pending_window = Window{
        .bytes = old.buffer.Mapped(),
        .dispatches = old.dispatches,
        .seconds = static_cast<u64>(elapsed.count()),
    };
    active ^= 1;
    window_start = now;
    return nullptr;
}

void RawCoverageRings::Recycle() {
    if (!pending) {
        return;
    }
    Clear(rings[active ^ 1]);
    pending = false;
}

u64 RawCoverageRings::TailMisses() const {
    const auto counter{[](const Ring& ring) {
        u32 value{};
        std::memcpy(&value, ring.buffer.Mapped().data() + Cov::RING_COUNTER_WORD * sizeof(u32),
                    sizeof(value));
        return u64{value};
    }};
    return counter(rings[active]) + (pending ? counter(rings[active ^ 1]) : 0);
}

void RawCoverageRings::Clear(Ring& ring) {
    std::memset(ring.buffer.Mapped().data(), 0, Cov::RING_HEADER_BYTES);
    ring.dispatches = 0;
    ring.last_tick = 0;
    ring.present = ~0u;
}

u64 SchedulerRawCoverageQueue::CurrentTick() {
    return scheduler.CurrentTick();
}

void SchedulerRawCoverageQueue::HostReadBarrier() {
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               barrier);
    });
}

bool SchedulerRawCoverageQueue::IsDone(u64 tick) {
    return scheduler.IsFree(tick);
}

} // namespace Vulkan
