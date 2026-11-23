// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Lifecycle of the NVN HLE across boots in one process (a GUI that starts and stops games): the
// Recorder is destroyed after the renderer (its sink) died, guest threads can still be inside an
// NVN call when the renderer goes away, and the second boot starts from a clean state.

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/boot_counter.h"
#include "core/hle/nvn/hle_runtime.h"
#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/recorder.h"
#include "video_core/nvn/sink.h"
#include "video_core/nvn/sink_gate.h"

namespace {

namespace VC = VideoCommon::Nvn;
using namespace std::chrono_literals;
using Core::NvnHle::Hle::Runtime;

/// What happened to a sink after the "renderer" died. Outlives the sink.
struct Life {
    std::atomic<bool> dead{false};
    std::atomic<u64> submits{0};
    std::atomic<u64> calls_after_death{0};
    std::atomic<u64> callbacks_after_death{0};
};

/// A sink shaped like the Vulkan context: its completion callback runs on a thread of its own and
/// SetCompletionCallback({}) waits for a running one. `Kill` stands for the renderer's death:
/// from then on any call into the sink (or any callback) is a use after free in production and is
/// counted here.
class LifeSink final : public VC::Sink {
public:
    explicit LifeSink(std::shared_ptr<Life> life_) : life{std::move(life_)} {}

    void Kill() {
        life->dead.store(true);
    }

    void CreatePool(const VC::PoolDesc&) override {
        Touch();
    }
    void CreateTexture(const VC::TextureDesc&) override {
        Touch();
    }
    void CreateSampler(const VC::SamplerDesc&) override {
        Touch();
    }
    void CreateProgram(const VC::ProgramDesc&) override {
        Touch();
    }
    void Destroy(u64) override {
        Touch();
    }
    void Submit(VC::Submission&&) override {
        Touch();
        life->submits.fetch_add(1);
    }
    void Present(const VC::PresentRequest&) override {
        Touch();
    }
    bool Wait(u64, std::chrono::nanoseconds) override {
        Touch();
        return true;
    }
    void SetCompletionCallback(std::function<void(u64)> callback_) override {
        Touch();
        std::scoped_lock lock{callback_mutex}; // waits for a running callback
        callback = std::move(callback_);
    }
    void NotifyEventCpu(u64, VC::EventCpuChange, u32) override {
        Touch();
    }
    void SetFailureCallback(std::function<void(std::string)> callback_) override {
        Touch();
        std::scoped_lock lock{failure_mutex};
        failure = std::move(callback_);
    }
    void SetGuestMemory(VC::GuestMemory* memory_) override {
        Touch();
        guest = memory_;
    }

    /// The completion thread: reports `seq` and everything before it.
    void Complete(u64 seq) {
        std::scoped_lock lock{callback_mutex};
        if (!callback) {
            return;
        }
        if (life->dead.load()) {
            life->callbacks_after_death.fetch_add(1);
        }
        for (; completed < seq;) {
            callback(++completed);
        }
    }
    void Fail(const std::string& reason) {
        std::scoped_lock lock{failure_mutex};
        if (failure) {
            failure(reason);
        }
    }
    [[nodiscard]] bool HasCallback() {
        std::scoped_lock lock{callback_mutex};
        return static_cast<bool>(callback);
    }
    [[nodiscard]] bool HasFailureCallback() {
        std::scoped_lock lock{failure_mutex};
        return static_cast<bool>(failure);
    }
    [[nodiscard]] VC::GuestMemory* Guest() const {
        return guest;
    }

private:
    void Touch() {
        if (life->dead.load()) {
            life->calls_after_death.fetch_add(1);
        }
    }

    std::shared_ptr<Life> life;
    std::mutex callback_mutex;
    std::function<void(u64)> callback;
    std::mutex failure_mutex;
    std::function<void(std::string)> failure;
    VC::GuestMemory* guest{};
    u64 completed{0};
};

/// Guest memory that reports its destruction.
class TrackedMemory final : public NvnHleTest::FakeMemory {
public:
    explicit TrackedMemory(std::shared_ptr<std::atomic<bool>> destroyed_)
        : destroyed{std::move(destroyed_)} {}
    ~TrackedMemory() override {
        destroyed->store(true);
    }

private:
    std::shared_ptr<std::atomic<bool>> destroyed;
};

constexpr u64 BASE = NvnHleTest::Rig::BASE;
constexpr u64 POOL = NvnHleTest::Rig::Obj(0);
constexpr u64 QUEUE = NvnHleTest::Rig::Obj(1);
constexpr u64 SYNC = NvnHleTest::Rig::Obj(2);

/// One boot: starts the runtime on a fresh sink, lets a "guest thread" call the Recorder and a
/// "completion thread" report completions, kills the renderer after Runtime::Shutdown (as the
/// system does) while the guest thread is still calling, then releases.
struct Boot {
    std::shared_ptr<Life> life = std::make_shared<Life>();
    std::shared_ptr<std::atomic<bool>> memory_destroyed = std::make_shared<std::atomic<bool>>(false);
    std::unique_ptr<LifeSink> sink = std::make_unique<LifeSink>(life);
};

} // namespace

TEST_CASE("nvn lifecycle: BootCounter counts per boot", "[nvn][lifecycle]") {
    static Common::BootCounter counter;
    REQUIRE(counter.First());
    REQUIRE_FALSE(counter.First());
    REQUIRE(counter.Next() == 2);
    Common::BootCounter::ResetAll();
    REQUIRE(counter.Count() == 0);
    REQUIRE(counter.First());
}

TEST_CASE("nvn lifecycle: a detached gate stops reaching the sink", "[nvn][lifecycle]") {
    auto life = std::make_shared<Life>();
    LifeSink sink{life};
    NvnHleTest::FakeMemory memory;
    memory.Map(BASE, NvnHleTest::Rig::MEMORY_SIZE);
    VC::SinkGate gate{sink};
    VC::Recorder recorder{memory, gate};
    REQUIRE(sink.HasCallback());
    REQUIRE(sink.Guest() == &memory);
    gate.SetFailureCallback([](std::string) {});
    REQUIRE(sink.HasFailureCallback());

    REQUIRE(recorder.MemoryPoolInitialize(POOL, NvnHleTest::Rig::STORAGE, 0x1000, 0));
    REQUIRE(recorder.QueueInitialize(QUEUE));
    REQUIRE(recorder.SyncInitialize(SYNC));
    REQUIRE(recorder.QueueFenceSync(QUEUE, SYNC, 0, 0) == 1);
    REQUIRE(life->submits.load() == 1);

    gate.Detach();
    REQUIRE_FALSE(gate.Attached());
    REQUIRE_FALSE(sink.HasCallback());
    REQUIRE_FALSE(sink.HasFailureCallback());
    REQUIRE(sink.Guest() == nullptr);
    sink.Kill();

    // Everything the game does afterwards ends at the gate; the Recorder can even be destroyed.
    REQUIRE(recorder.QueueFenceSync(QUEUE, SYNC, 0, 0) == 2);
    recorder.FlushDestroys();
    CHECK(life->submits.load() == 1);
}

TEST_CASE("nvn lifecycle: Detach waits for a running completion callback", "[nvn][lifecycle]") {
    auto life = std::make_shared<Life>();
    LifeSink sink{life};
    NvnHleTest::FakeMemory memory;
    memory.Map(BASE, NvnHleTest::Rig::MEMORY_SIZE);
    VC::SinkGate gate{sink};
    VC::Recorder recorder{memory, gate};
    REQUIRE(recorder.MemoryPoolInitialize(POOL, NvnHleTest::Rig::STORAGE, 0x1000, 0));
    REQUIRE(recorder.QueueInitialize(QUEUE));
    REQUIRE(recorder.SyncInitialize(SYNC));
    REQUIRE(recorder.QueueFenceSync(QUEUE, SYNC, 0, 0) == 1);

    // The completion thread sits inside the Recorder's callback (held by the observer).
    std::atomic<bool> inside{false};
    std::atomic<bool> release{false};
    recorder.SetCompletionObserver([&](u64) {
        inside = true;
        while (!release) {
            std::this_thread::sleep_for(1ms);
        }
    });
    std::thread completion{[&] { sink.Complete(1); }};
    const auto inside_deadline = std::chrono::steady_clock::now() + 5s;
    while (!inside && std::chrono::steady_clock::now() < inside_deadline) {
        std::this_thread::sleep_for(1ms);
    }
    if (!inside) {
        // The observer never ran: let a late one return and stop the completion thread first.
        release = true;
        completion.join();
        recorder.SetCompletionObserver({});
        FAIL("the completion observer never started");
    }
    std::atomic<bool> detached{false};
    std::thread detacher{[&] {
        gate.Detach();
        detached = true;
    }};
    std::this_thread::sleep_for(50ms);
    CHECK_FALSE(detached.load()); // the callback is still running: the sink is not free to go
    release = true;
    completion.join();
    detacher.join();
    CHECK(detached.load());
    recorder.SetCompletionObserver({});
}

TEST_CASE("nvn lifecycle: two boots in one process, the renderer dying under a guest thread",
          "[nvn][lifecycle]") {
    Runtime runtime;
    std::vector<std::shared_ptr<Life>> lives;
    for (int boot_index = 0; boot_index < 2; ++boot_index) {
        CAPTURE(boot_index);
        Boot boot;
        lives.push_back(boot.life);
        auto memory = std::make_unique<TrackedMemory>(boot.memory_destroyed);
        memory->Map(BASE, NvnHleTest::Rig::MEMORY_SIZE);
        REQUIRE(runtime.Start(*boot.sink, std::move(memory), nullptr));
        REQUIRE_FALSE(runtime.Start(*boot.sink, std::make_unique<NvnHleTest::FakeMemory>(), nullptr));

        VC::Recorder* recorder = runtime.GetRecorder();
        REQUIRE(recorder != nullptr);
        // A second boot starts from nothing: a fresh Recorder numbers its submissions from 1 and
        // the debug domain ids start again.
        REQUIRE(recorder->Stats().submissions == 0);
        REQUIRE(runtime.next_debug_domain_id.fetch_add(1) == 1);
        REQUIRE(recorder->MemoryPoolInitialize(POOL, NvnHleTest::Rig::STORAGE, 0x1000, 0));
        REQUIRE(recorder->QueueInitialize(QUEUE));
        REQUIRE(recorder->SyncInitialize(SYNC));
        REQUIRE(recorder->QueueFenceSync(QUEUE, SYNC, 0, 0) == 1);
        boot.sink->Complete(1);
        REQUIRE(recorder->CompletedSeq() == 1);
        {
            std::scoped_lock lock{runtime.window_mutex};
            runtime.windows[0x1234] = {};
        }
        runtime.guest_request_producer = 0x1234;

        std::atomic<bool> finish{false};
        std::atomic<u64> guest_calls{0};
        // The guest thread keeps calling the Recorder past the renderer's death.
        std::thread guest{[&] {
            while (!finish) {
                recorder->QueueFenceSync(QUEUE, SYNC, 0, 0);
                ++guest_calls;
                std::this_thread::yield();
            }
        }};
        // The completion thread reports whatever the guest submitted.
        std::thread completion{[&] {
            while (!finish) {
                boot.sink->Complete(recorder->LastSeq());
                std::this_thread::yield();
            }
        }};
        std::this_thread::sleep_for(30ms);

        // System shutdown: the runtime stops first, the renderer dies, and only after the CPU
        // threads are gone is the Recorder released.
        runtime.Shutdown();
        const bool stopped = runtime.IsStopped();
        CHECK_FALSE(boot.sink->HasCallback());
        boot.sink->Kill();
        // Two more finished calls: at least one started after the sink died.
        const u64 calls_at_death = guest_calls.load();
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (guest_calls.load() < calls_at_death + 2 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(1ms);
        }
        finish = true;
        guest.join();
        completion.join();
        REQUIRE(stopped);
        CHECK(guest_calls.load() >= calls_at_death + 2);
        CHECK(boot.life->calls_after_death.load() == 0);
        CHECK(boot.life->callbacks_after_death.load() == 0);
        CHECK_FALSE(boot.memory_destroyed->load()); // still alive for a thread inside a call
        CHECK(runtime.GetRecorder() != nullptr);

        runtime.Release();
        CHECK(boot.memory_destroyed->load());
        CHECK(runtime.GetRecorder() == nullptr);
        CHECK(boot.life->calls_after_death.load() == 0);

        runtime.ResetForNewProcess();
        CHECK_FALSE(runtime.IsStopped());
        CHECK(runtime.next_debug_domain_id.load() == 1);
        CHECK(runtime.guest_request_producer == 0);
        {
            std::scoped_lock lock{runtime.window_mutex};
            CHECK(runtime.windows.empty());
        }
    }
    // The two boots used two different sinks, and neither saw a call after its death.
    REQUIRE(lives.size() == 2);
    CHECK(lives[0]->submits.load() > 0);
    CHECK(lives[1]->submits.load() > 0);
}

TEST_CASE("nvn lifecycle: a boot never released is cleaned up by the next one", "[nvn][lifecycle]") {
    Runtime runtime;
    Boot first;
    auto memory = std::make_unique<TrackedMemory>(first.memory_destroyed);
    memory->Map(BASE, NvnHleTest::Rig::MEMORY_SIZE);
    REQUIRE(runtime.Start(*first.sink, std::move(memory), nullptr));
    runtime.Shutdown();
    first.sink->Kill();
    CHECK_FALSE(first.memory_destroyed->load());

    runtime.ResetForNewProcess();
    CHECK(first.memory_destroyed->load());
    CHECK(first.life->calls_after_death.load() == 0);

    Boot second;
    auto second_memory = std::make_unique<TrackedMemory>(second.memory_destroyed);
    second_memory->Map(BASE, NvnHleTest::Rig::MEMORY_SIZE);
    REQUIRE(runtime.Start(*second.sink, std::move(second_memory), nullptr));
    CHECK(runtime.GetRecorder() != nullptr);
    runtime.Release();
    CHECK(second.memory_destroyed->load());
}

TEST_CASE("nvn lifecycle: the failure callback reaches the owner until detached",
          "[nvn][lifecycle]") {
    auto life = std::make_shared<Life>();
    LifeSink sink{life};
    VC::SinkGate gate{sink};
    std::vector<std::string> reasons;
    gate.SetFailureCallback([&](std::string reason) { reasons.push_back(std::move(reason)); });
    sink.Fail("device lost");
    REQUIRE(reasons == std::vector<std::string>{"device lost"});
    gate.Detach();
    sink.Fail("again");
    CHECK(reasons.size() == 1);
}
