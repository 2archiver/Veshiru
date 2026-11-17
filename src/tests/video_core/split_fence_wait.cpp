// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "tests/core/system_test_context.h"
#include "video_core/fence_manager.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {

struct Cache {
    std::mutex mutex;
    bool HasUncommittedFlushes() const { return true; }
    bool HasAsyncFlushWork() const { return true; }
    bool ShouldWaitAsyncFlushes() const { return true; }
    void AccumulateFlushes() {}
    void CommitAsyncFlushes() {}
    void PopAsyncFlushes() {}
};

struct Fence : VideoCommon::FenceBase {
    Fence(bool stubbed, u64 tick_) : FenceBase{stubbed}, tick{tick_} {}
    u64 tick;
};

struct Traits {
    using FenceType = std::shared_ptr<Fence>;
    using TextureCacheType = Cache;
    using BufferCacheType = Cache;
    using QueryCacheType = Cache;
    static constexpr bool HAS_ASYNC_CHECK = true;
};

struct Submission {
    std::mutex mutex;
    std::condition_variable cv;
    u64 submitted{};
    std::atomic<bool> timed_out{};
    bool wrong_owner{};
    const std::thread::id owner = std::this_thread::get_id();
};

class FenceManager final : public VideoCommon::FenceManager<Traits> {
public:
    FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_, Cache& texture,
                 Cache& buffer, Cache& query, Submission& submission_)
        : VideoCommon::FenceManager<Traits>{rasterizer_, gpu_, texture, buffer, query},
          submission{submission_} {}

    u64 queued{};
    u32 submissions{};

protected:
    std::shared_ptr<Fence> CreateFence(bool stubbed) override {
        return std::make_shared<Fence>(stubbed, queued + 1);
    }

    void QueueFence(std::shared_ptr<Fence>& fence) override {
        queued = fence->tick;
    }

    bool IsFenceSignaled(std::shared_ptr<Fence>& fence) const override {
        std::scoped_lock lock{submission.mutex};
        return submission.submitted >= fence->tick;
    }

    void WaitFence(std::shared_ptr<Fence>& fence) override {
        std::unique_lock lock{submission.mutex};
        if (!submission.cv.wait_for(lock, std::chrono::seconds{2},
                                    [&] { return submission.submitted >= fence->tick; })) {
            submission.timed_out.store(true);
        }
    }

    void SubmitDeferredWork() override {
        std::scoped_lock lock{submission.mutex};
        submission.wrong_owner |= std::this_thread::get_id() != submission.owner;
        submission.submitted = queued;
        ++submissions;
        submission.cv.notify_all();
    }

private:
    Submission& submission;
};

} // namespace

TEST_CASE("Delayed fence callbacks submit before the owner waits", "[video_core][split_fence_wait]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    SCOPE_EXIT { Settings::values.gpu_fence_behavior.SetValue(previous); };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Accurate);
    std::vector<u32> callbacks;
    {
        FenceManager manager{rasterizer, gpu, texture, buffer, query, submission};
        for (u32 i = 0; i < 128; ++i) {
            manager.SyncOperation([&, i] { callbacks.push_back(i * 2); });
            manager.SignalFence([&, i] { callbacks.push_back(i * 2 + 1); });
            REQUIRE(manager.submissions == i + 1);
        }
        manager.WaitPendingFences(true);
        REQUIRE(manager.submissions == 130);
        REQUIRE_FALSE(submission.timed_out.load());
        REQUIRE_FALSE(submission.wrong_owner);
    }
    REQUIRE(callbacks.size() == 256);
    for (u32 i = 0; i < callbacks.size(); ++i) {
        REQUIRE(callbacks[i] == i);
    }
}

TEST_CASE("Immediate fence waits still submit outstanding work", "[video_core][split_fence_wait]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Cache texture, buffer, query;
    Submission submission;
    const auto previous = Settings::values.gpu_fence_behavior.GetValue();
    const auto previous_accuracy = Settings::values.current_gpu_accuracy;
    SCOPE_EXIT {
        Settings::values.gpu_fence_behavior.SetValue(previous);
        Settings::values.current_gpu_accuracy = previous_accuracy;
    };
    Settings::values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Default);
    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;
    std::atomic<bool> callback{};
    {
        FenceManager manager{rasterizer, gpu, texture, buffer, query, submission};
        manager.SyncOperation([&] { callback.store(true); });
        manager.SignalFence([] {});
        REQUIRE(manager.submissions == 0);
        manager.WaitPendingFences(false);
        REQUIRE(manager.submissions == 0);
        manager.WaitPendingFences(true);
        REQUIRE(manager.submissions == 1);
        Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::High;
        manager.WaitPendingFences(true);
        REQUIRE(manager.submissions == 3);
        REQUIRE_FALSE(submission.timed_out.load());
        REQUIRE_FALSE(submission.wrong_owner);
    }
    REQUIRE(callback.load());
}

TEST_CASE("GpuSplit Stop resets synchronization state cleanly across restarts",
          "[video_core][split_fence_wait][gpu_split]") {
    ScopedSystemTestContext context;
    Core::System system;
    auto& split = VideoCommon::GpuSplit::Get();

    const bool prev_ampere = Tegra::g_nx2_ampere.exchange(true, std::memory_order_relaxed);
    const auto prev_gpu_acc = Settings::values.current_gpu_accuracy;
    const auto prev_dma_acc = Settings::values.dma_accuracy.GetValue();
    const auto prev_backend = Settings::values.renderer_backend.GetValue();
    SCOPE_EXIT {
        split.SetIdleHook(nullptr, nullptr);
        split.Stop();
        Tegra::g_nx2_ampere.store(prev_ampere, std::memory_order_relaxed);
        Settings::values.current_gpu_accuracy = prev_gpu_acc;
        Settings::values.dma_accuracy.SetValue(prev_dma_acc);
        Settings::values.renderer_backend.SetValue(prev_backend);
    };

    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Normal;
    Settings::values.dma_accuracy.SetValue(Settings::DmaAccuracy::Default);
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);

    struct HookState {
        std::atomic<u32> calls{};
        std::atomic<bool> on_back{true};
    };

    for (u32 session = 0; session < 2; ++session) {
        HookState hook_state;
        split.SetIdleHook(
            [](void* arg) {
                auto* s = static_cast<HookState*>(arg);
                s->calls.fetch_add(1, std::memory_order_relaxed);
                if (!VideoCommon::GpuSplit::OnBackThread() ||
                    !VideoCommon::GpuSplit::OwnsScheduler()) {
                    s->on_back.store(false, std::memory_order_relaxed);
                }
            },
            &hook_state);

        split.MaybeStart(system);
        if (!split.Active()) {
            SKIP("GpuSplit disabled via EDEN_NX2_SPLIT=0");
        }
        REQUIRE(split.Active());

        std::atomic<u32> executed{};
        std::atomic<bool> all_on_back{true};
        for (u32 i = 0; i < 64; ++i) {
            struct Job {
                std::atomic<u32>* count;
                std::atomic<bool>* on_back;
                void operator()() const {
                    if (!VideoCommon::GpuSplit::OnBackThread() ||
                        !VideoCommon::GpuSplit::OwnsScheduler()) {
                        on_back->store(false, std::memory_order_relaxed);
                    }
                    count->fetch_add(1, std::memory_order_relaxed);
                }
            };
            split.Defer(Job{&executed, &all_on_back});
            if ((i & 7U) == 7U) {
                split.RequestIdleHook();
            }
        }
        split.NoteSemaphoreWrite(0x1000, 42U, true);
        split.Drain(VideoCommon::DrainSite::Other);

        REQUIRE(executed.load() == 64U);
        REQUIRE(all_on_back.load());
        REQUIRE(hook_state.calls.load() >= 1U);
        REQUIRE(hook_state.on_back.load());
        REQUIRE_FALSE(split.QueuedSemaphoreWriteAt(0x1000).has_value());

        split.SetIdleHook(nullptr, nullptr);
        split.Stop();
        REQUIRE_FALSE(split.Active());
        REQUIRE_FALSE(split.BackRunning());
    }
}
