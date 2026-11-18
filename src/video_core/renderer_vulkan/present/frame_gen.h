// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <optional>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/present/frame_gen_pacer.h"
#include "video_core/renderer_vulkan/present/lsfg_chain.h"
#include "video_core/renderer_vulkan/present/lsfg_shaders.h"
#include "video_core/renderer_vulkan/present/fsr3_frame_gen.h"
#include "video_core/renderer_vulkan/present/open_frame_gen.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

class Device;
class Scheduler;
class PresentManager;
struct Frame;

class FrameGen {
public:
    explicit FrameGen(MemoryAllocator& memory_allocator, Scheduler& scheduler);
    ~FrameGen();

    void Process(const Device& device, PresentManager& present, Frame* frame,
                 VkExtent2D guest_extent);

    [[nodiscard]] size_t PlannedGenerations() const;
    void CompleteFrame(size_t queued_generations, const PresentManager& present);
    void ResourceFailed(VkFormat format);

    /// The measured time between real frames in seconds, or 0 while it is unknown.
    [[nodiscard]] f32 FrameInterval() const {
        return pacer.SmoothedInterval();
    }

    void GenerateInto(const Device& device, Frame* destination, size_t generation);

    [[nodiscard]] VideoCore::FrameGenStatus CurrentStatus() const {
        return {.state = state.load(std::memory_order_relaxed),
                .generated_frames = generated_frames.load(std::memory_order_relaxed)};
    }

private:
    struct Config {
        u32 width;
        u32 height;
        VkFormat format;
        f32 scale;
        Settings::FrameGenType type;
        bool operator==(const Config&) const = default;
    };
    void ResetHistory();
    void Rebuild(const Device& device, const Config& config);
    void PlanGenerations(PresentManager& present);
    /// Updates the status bar state and logs each change with its reason.
    void SetState(VideoCore::FrameGenState new_state, VkFormat format);
    /// Logs every 10 s how many frames were generated, when that changes.
    void LogActivity(size_t queued_generations, VideoCore::FramePresentCounts presents);
    void DumpDebugImages(u64 count);

    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    std::optional<LsfgShaders> shaders;
    std::optional<LsfgChain> chain;
    std::optional<Fsr3FrameGen> fsr3_chain;
    std::optional<OpenFrameGen> open_fg_chain;
    FrameGenPacer pacer;
    FrameGenPlan plan{};
    VkExtent2D peak_guest_extent{};
    std::optional<Config> current_config;
    std::optional<Config> blocked_config;
    VideoCore::FrameGenState blocked_state{};
    u64 frame_count{};
    u64 last_count{};
    size_t last_generations{};
    u32 warm_streak{};
    bool generated{};
    bool dumped{};
    std::atomic<VideoCore::FrameGenState> state{VideoCore::FrameGenState::Pending};
    std::atomic<u64> generated_frames{};
    struct {
        std::chrono::steady_clock::time_point since{};
        VideoCore::FramePresentCounts presents;
        u64 frames{};
        u64 generated{};
        u64 hitch_skips{};
        u64 display_trims{};
        f32 logged_per_frame{-1.0f};
    } activity;
};

} // namespace Vulkan
