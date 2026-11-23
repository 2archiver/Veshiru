// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "core/core.h"
#include "core/hle/nvn/bridge.h"
#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/guest_memory.h"

using namespace NvnHleTest;
using VideoCommon::Nvn::GuestTag;
using VideoCommon::Nvn::GUEST_TAG_MAGIC;
using VideoCommon::Nvn::ObjectKind;
using VideoCommon::Nvn::TextureDesc;

namespace {

constexpr u64 POOL = Rig::Obj(0);
constexpr u64 QUEUE = Rig::Obj(1);
constexpr u64 DEVICE = Rig::Obj(2);
constexpr u64 BUILDER = Rig::Obj(4);
constexpr u64 WINDOW = Rig::Obj(6);
constexpr u64 SYNC = Rig::Obj(8);

class ScopedEnv {
public:
    ScopedEnv(const char* name_, const char* value) : name{name_} {
        if (const char* old = std::getenv(name); old != nullptr) {
            previous = old;
        }
        Set(value);
    }

    ~ScopedEnv() {
        Set(previous ? previous->c_str() : nullptr);
    }

private:
    void Set(const char* value) const {
#ifdef _WIN32
        _putenv_s(name, value != nullptr ? value : "");
#else
        if (value != nullptr) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    const char* name;
    std::optional<std::string> previous;
};

TextureDesc WindowTextureDesc() {
    TextureDesc desc{};
    desc.target = 1;
    desc.format = 0x25;
    desc.width = 1280;
    desc.height = 720;
    desc.depth = 1;
    desc.levels = 1;
    return desc;
}

/// Creates a window of `count` textures through the HLE bodies.
void MakeWindow(Rig& rig, u32 count) {
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
    REQUIRE(rig.Call("nvnQueueInitialize", {Rig::I(QUEUE), Rig::I(BUILDER)}) == 1);
    std::vector<u64> textures;
    for (u32 i = 0; i < count; ++i) {
        const u64 texture = Rig::Obj(20 + i);
        REQUIRE(rig.recorder.TextureInitialize(texture, WindowTextureDesc(), POOL, 0x10000ull * i));
        textures.push_back(texture);
        rig.memory.Put<u64>(Rig::SCRATCH + 8 * i, texture);
    }
    rig.Call("nvnWindowBuilderSetDefaults", {Rig::I(BUILDER)});
    rig.Call("nvnWindowBuilderSetTextures",
             {Rig::I(BUILDER), Rig::I(count), Rig::I(Rig::SCRATCH)});
    REQUIRE(rig.Call("nvnWindowInitialize", {Rig::I(WINDOW), Rig::I(BUILDER)}) == 1);
}

} // namespace

TEST_CASE("nvn hle device: info table, handles and modes", "[nvn]") {
    Rig rig;
    const u64 out = Rig::SCRATCH;
    const auto get = [&](u32 item) {
        rig.memory.Put<s32>(out, 77);
        rig.Call("nvnDeviceGetInteger", {Rig::I(DEVICE), Rig::I(item), Rig::I(out)});
        return rig.memory.Get<s32>(out);
    };
    // The 3.2.0 SDK (sdk+0x5d9960 over its table sdk+0x5c0a20): API 231.30, GPU code 201.17.
    CHECK(get(0) == 231);
    CHECK(get(1) == 30);
    CHECK(get(0x2E) == 201);
    CHECK(get(0x30) == 17);
    CHECK(get(0x9D) == 14);
    CHECK(get(0x06) == 32); // vertex buffer bindings
    CHECK(get(0x0E) == 32); // vertex attributes
    // Items the SDK reads from the device: the Switch 2 GPU (12 SMs of 48 warps, 2 per TPC).
    CHECK(get(0x17) == 12);
    CHECK(get(0x18) == 576);
    CHECK(get(0x37) == 96);
    CHECK(get(0x2B) == 1048576);
    CHECK(get(0x3D) == 32768);
    CHECK(get(0x3E) == 268435456);
    CHECK(get(0x3F) == 16384);
    CHECK(get(0x62) == 32);
    CHECK(get(0x7D) == 8);
    CHECK(get(0x87) == 1);
    CHECK(get(0x99) == 2);
    CHECK(get(0x9A) == 1);
    CHECK(get(0xAB) == 63);
    CHECK(get(0xB0) == 4096);
    CHECK(get(0xB1) == 0);
    CHECK(get(0x1000) == 0);
    // A device created with negative builder flags reserved no address range.
    rig.runtime.device.flags = 0x8000'0000u;
    CHECK(get(0x9A) == 0);
    rig.runtime.device.flags = 0;
    // A game built with the 1.0.0-generation sdk (698 GetProcAddress names) gets its answers.
    rig.runtime.device.sdk_proc_names = 698;
    CHECK(get(1) == 16);
    CHECK(get(0x30) == 11);
    CHECK(get(0x9D) == 11);
    CHECK(get(0x3E) == 134217728);
    CHECK(get(0x7F) == 65535);
    CHECK(get(0x98) == 131072);
    CHECK(get(0xAB) == 0);
    CHECK(get(0x17) == 12);
    rig.runtime.device.sdk_proc_names = 726;
    CHECK(get(1) == 30);

    CHECK(rig.Call("nvnDeviceGetTextureHandle", {Rig::I(DEVICE), Rig::I(3), Rig::I(5)}) ==
          (3ull | 5ull << 20));
    CHECK(rig.Call("nvnDeviceGetTexelFetchHandle", {Rig::I(DEVICE), Rig::I(9)}) == 9);
    CHECK(rig.Call("nvnDeviceGetSeparateSamplerHandle", {Rig::I(DEVICE), Rig::I(7)}) ==
          (7ull << 20));
    CHECK(rig.Call("nvnDeviceGetSeparateTextureHandle", {Rig::I(DEVICE), Rig::I(7)}) == 7);
    // Only the int's 32 bits count (the caller's upper register half is garbage); the SDK widens
    // them with sign extension.
    CHECK(rig.Call("nvnDeviceGetImageHandle", {Rig::I(DEVICE), Rig::I(0xDEAD'0000'0000'0004ull)}) ==
          4);
    CHECK(rig.Call("nvnDeviceGetImageHandle", {Rig::I(DEVICE), Rig::I(0xFFFF'FFFFull)}) == ~0ull);
    CHECK(rig.Call("nvnDeviceGetTexelFetchHandle", {Rig::I(DEVICE), Rig::I(0xFFFF'FFFEull)}) ==
          ~1ull);
    CHECK(rig.Call("nvnDeviceGetTextureHandle", {Rig::I(DEVICE), Rig::I(3), Rig::I(0xFFFF'FFFFull)}) ==
          (3ull | ~0ull << 20));
    // Donkey Kong Bananza's SDKs: texture id 0 is an id like any other.
    CHECK(rig.Call("nvnDeviceGetTextureHandle", {Rig::I(DEVICE), Rig::I(0), Rig::I(5)}) ==
          (5ull << 20));
    // Ocarina of Time's SDK (sdk+0x651c60) answers the null handle for texture id 0 whatever the
    // sampler; other ids keep the combined form.
    rig.runtime.device.texture_zero_null_handle = true;
    CHECK(rig.Call("nvnDeviceGetTextureHandle", {Rig::I(DEVICE), Rig::I(0), Rig::I(5)}) == 0);
    CHECK(rig.Call("nvnDeviceGetTextureHandle",
                   {Rig::I(DEVICE), Rig::I(0xFFFF'FFFF'0000'0000ull), Rig::I(5)}) == 0);
    CHECK(rig.Call("nvnDeviceGetTextureHandle", {Rig::I(DEVICE), Rig::I(3), Rig::I(5)}) ==
          (3ull | 5ull << 20));
    rig.runtime.device.texture_zero_null_handle = false;

    CHECK(rig.Call("nvnDeviceGetWindowOriginMode", {Rig::I(DEVICE)}) == 0);
    rig.Call("nvnDeviceSetWindowOriginMode", {Rig::I(DEVICE), Rig::I(1)});
    rig.Call("nvnDeviceSetDepthMode", {Rig::I(DEVICE), Rig::I(1)});
    CHECK(rig.Call("nvnDeviceGetWindowOriginMode", {Rig::I(DEVICE)}) == 1);
    CHECK(rig.Call("nvnDeviceGetDepthMode", {Rig::I(DEVICE)}) == 1);
}

TEST_CASE("nvn hle device: initialize tags the object and stores the builder flags", "[nvn]") {
    Rig rig;
    rig.memory.Put(BUILDER, std::array<u64, 4>{~0ull, ~0ull, ~0ull, ~0ull});
    rig.Call("nvnDeviceBuilderSetDefaults", {Rig::I(BUILDER)});
    // SDK layout: SetDefaults clears 0x18 bytes, the flags live at +4.
    CHECK(rig.memory.Get<std::array<u64, 4>>(BUILDER) == std::array<u64, 4>{0, 0, 0, ~0ull});
    rig.Call("nvnDeviceBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0x1234)});
    CHECK(rig.memory.Get<u32>(BUILDER + 4) == 0x1234);
    CHECK(rig.Call("nvnDeviceBuilderGetFlags", {Rig::I(BUILDER)}) == 0x1234);
    CHECK(rig.Call("nvnDeviceInitialize", {Rig::I(DEVICE), Rig::I(BUILDER)}) == 1);
    const GuestTag tag = rig.memory.Get<GuestTag>(DEVICE);
    CHECK(tag.magic == GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(ObjectKind::Device));
    CHECK(rig.Call("nvnDeviceGetFlags", {Rig::I(DEVICE)}) == 0x1234);
}

TEST_CASE("nvn hle queue: builder layout and the SDK's queue memory size", "[nvn]") {
    Rig rig;
    rig.Call("nvnQueueBuilderSetDefaults", {Rig::I(BUILDER)});
    // Device defaults: 64 KiB command, 256 KiB compute, 16 KiB control, 4 KiB flush threshold,
    // NVN_QUEUE_PRIORITY_NORMAL.
    CHECK(rig.Call("nvnQueueBuilderGetCommandMemorySize", {Rig::I(BUILDER)}) == 65536);
    CHECK(rig.Call("nvnQueueBuilderGetComputeMemorySize", {Rig::I(BUILDER)}) == 262144);
    CHECK(rig.Call("nvnQueueBuilderGetControlMemorySize", {Rig::I(BUILDER)}) == 16384);
    CHECK(rig.Call("nvnQueueBuilderGetCommandFlushThreshold", {Rig::I(BUILDER)}) == 4096);
    CHECK(rig.Call("nvnQueueBuilderGetQueuePriority", {Rig::I(BUILDER)}) == 100);
    // sdk+0x5e6fa0: 64 KiB command + 16 KiB control + the internal block (sdk+0x5e7cf4: scratch
    // 0x3900 + 0x1800 + a page, zcull page, 60 compute slots of 0x1100) rounded to 4 KiB.
    CHECK(rig.Call("nvnQueueBuilderGetQueueMemorySize", {Rig::I(BUILDER)}) == 376832);
    // Flag 0x10: no internal block.
    rig.Call("nvnQueueBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0x10)});
    CHECK(rig.Call("nvnQueueBuilderGetQueueMemorySize", {Rig::I(BUILDER)}) == 0x14000);
    // NO_ZCULL drops the zcull context and its page alignment: 0x6100 + 0x3fc00 -> 0x46000.
    rig.Call("nvnQueueBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0x2)});
    CHECK(rig.Call("nvnQueueBuilderGetQueueMemorySize", {Rig::I(BUILDER)}) == 0x14000 + 0x46000);
    rig.Call("nvnQueueBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0)});
    // Set sizes count as given; the compute memory only adds slots of 0x1100 bytes.
    rig.Call("nvnQueueBuilderSetCommandMemorySize", {Rig::I(BUILDER), Rig::I(1000)});
    rig.Call("nvnQueueBuilderSetComputeMemorySize", {Rig::I(BUILDER), Rig::I(1000)});
    rig.Call("nvnQueueBuilderSetControlMemorySize", {Rig::I(BUILDER), Rig::I(1000)});
    CHECK(rig.Call("nvnQueueBuilderGetQueueMemorySize", {Rig::I(BUILDER)}) == 2000 + 0x8000);
    // The builder's device survives SetDefaults, which forgets the set sizes.
    rig.Call("nvnQueueBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
    rig.Call("nvnQueueBuilderSetDefaults", {Rig::I(BUILDER)});
    CHECK(rig.Call("nvnQueueBuilderGetDevice", {Rig::I(BUILDER)}) == DEVICE);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x40) == 0);
    CHECK(rig.Call("nvnQueueBuilderGetQueueMemorySize", {Rig::I(BUILDER)}) == 376832);
}

TEST_CASE("nvn hle queue: initialize refuses the builders the SDK refuses", "[nvn]") {
    Rig rig;
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
    const auto init = [&](u64 queue) {
        return rig.Call("nvnQueueInitialize", {Rig::I(queue), Rig::I(BUILDER)});
    };
    const auto fresh = [&] { rig.Call("nvnQueueBuilderSetDefaults", {Rig::I(BUILDER)}); };
    const auto set = [&](const char* name, u64 value) {
        rig.Call(name, {Rig::I(BUILDER), Rig::I(value)});
    };
    constexpr u64 MEMORY = 0x7000'0000;
    // Command memory: a multiple of 4 KiB, at least 64 KiB.
    fresh();
    set("nvnQueueBuilderSetCommandMemorySize", 0x10800);
    CHECK(init(QUEUE) == 0);
    set("nvnQueueBuilderSetCommandMemorySize", 0xF000);
    CHECK(init(QUEUE) == 0);
    // Control memory: a power of two of at least 4 KiB.
    fresh();
    set("nvnQueueBuilderSetControlMemorySize", 0x3000);
    CHECK(init(QUEUE) == 0);
    set("nvnQueueBuilderSetControlMemorySize", 0);
    CHECK(init(QUEUE) == 0);
    // Compute memory: a multiple of 256, 0 or at least 16 KiB.
    fresh();
    set("nvnQueueBuilderSetComputeMemorySize", 0x2000);
    CHECK(init(QUEUE) == 0);
    set("nvnQueueBuilderSetComputeMemorySize", 0x4080);
    CHECK(init(QUEUE) == 0);
    // Flush threshold: between 4 KiB and the command memory size.
    fresh();
    set("nvnQueueBuilderSetCommandFlushThreshold", 0x800);
    CHECK(init(QUEUE) == 0);
    set("nvnQueueBuilderSetCommandFlushThreshold", 0x20000);
    CHECK(init(QUEUE) == 0);
    // Queue memory: 4 KiB aligned and at least GetQueueMemorySize.
    fresh();
    const u64 size = rig.Call("nvnQueueBuilderGetQueueMemorySize", {Rig::I(BUILDER)});
    rig.Call("nvnQueueBuilderSetQueueMemory", {Rig::I(BUILDER), Rig::I(MEMORY), Rig::I(size - 0x1000)});
    CHECK(init(QUEUE) == 0);
    rig.Call("nvnQueueBuilderSetQueueMemory", {Rig::I(BUILDER), Rig::I(MEMORY + 0x10), Rig::I(size)});
    CHECK(init(QUEUE) == 0);
    rig.Call("nvnQueueBuilderSetQueueMemory", {Rig::I(BUILDER), Rig::I(MEMORY), Rig::I(size)});
    CHECK(init(QUEUE) == 1);
    // Unset sizes are the SDK defaults and pass.
    fresh();
    set("nvnQueueBuilderSetCommandFlushThreshold", 0x2000);
    set("nvnQueueBuilderSetComputeMemorySize", 0);
    CHECK(init(Rig::Obj(3)) == 1);
}

TEST_CASE("nvn hle queue: finish sleeps until the GPU completed the submission", "[nvn]") {
    WaitProbe probe;
    Rig rig;
    probe.Install(rig);
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
    REQUIRE(rig.Call("nvnQueueInitialize", {Rig::I(QUEUE), Rig::I(BUILDER)}) == 1);
    REQUIRE(rig.recorder.SyncInitialize(SYNC));
    // Nothing submitted yet: nothing to wait for.
    rig.Call("nvnQueueFinish", {Rig::I(QUEUE)});
    rig.Call("nvnQueueFenceSync", {Rig::I(QUEUE), Rig::I(SYNC), Rig::I(0), Rig::I(0)});
    REQUIRE(rig.sink.submissions.size() == 1);
    const u64 seq = rig.sink.submissions[0].seq;
    std::atomic<bool> finished{false};
    std::thread waiter{[&] {
        rig.Call("nvnQueueFinish", {Rig::I(QUEUE)});
        finished = true;
    }};
    // The worker reaches the completion wait while the submission is unfinished.
    CHECK(probe.AwaitWaiting());
    CHECK_FALSE(finished.load());
    rig.sink.Complete(seq);
    waiter.join();
    CHECK(finished.load());
    CHECK(rig.recorder.CompletedSeq() >= seq);
}

TEST_CASE("nvn hle pool: an invalidate waits for the GPU's writes of the submissions before it",
          "[nvn]") {
    using VideoCommon::Nvn::GpuWriteState;
    Rig rig;
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
    REQUIRE(rig.Call("nvnQueueInitialize", {Rig::I(QUEUE), Rig::I(BUILDER)}) == 1);
    REQUIRE(rig.recorder.SyncInitialize(SYNC));
    rig.Call("nvnQueueFenceSync", {Rig::I(QUEUE), Rig::I(SYNC), Rig::I(0), Rig::I(0)});
    REQUIRE(rig.sink.submissions.size() == 1);
    const u64 seq = rig.sink.submissions[0].seq;
    const auto invalidate = [&] {
        rig.Call("nvnMemoryPoolInvalidateMappedRange",
                 {Rig::I(POOL), Rig::I(0x100), Rig::I(0x40)});
    };

    SECTION("tracking off: no wait, the sink is not polled") {
        std::atomic<int> polls{0};
        rig.sink.poll = [&](u64, u64, u64, u64) {
            ++polls;
            return GpuWriteState{};
        };
        invalidate();
        CHECK(polls.load() == 0);
    }
    SECTION("a write of a submission made before it holds the call until the GPU completed it") {
        rig.sink.write_marker = 5;
        std::atomic<int> polls{0};
        std::atomic<u64> max_seq{~u64{0}};
        std::atomic<u64> asked_address{0};
        std::atomic<u64> asked_size{0};
        std::atomic<u64> asked_marker{0};
        rig.sink.poll = [&](u64 marker, u64 address, u64 size, u64 limit_seq) {
            asked_marker = marker;
            asked_address = address;
            asked_size = size;
            max_seq = limit_seq;
            GpuWriteState state;
            state.wait_limit_ms = 5000;
            // The replay reaches the marker only after a few polls: the writer is not known before.
            state.replay_reached = ++polls > 3;
            state.writer_seq = state.replay_reached ? seq : 0;
            return state;
        };
        std::atomic<bool> finished{false};
        std::thread waiter{[&] {
            invalidate();
            finished = true;
        }};
        // The 4th poll makes the writer known; from then on the worker waits for its completion.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
        while (polls.load() < 4 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
        CHECK(polls.load() >= 4);
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        CHECK_FALSE(finished.load());
        CHECK(rig.recorder.CompletedSeq() < seq);
        rig.sink.Complete(seq);
        waiter.join();
        CHECK(finished.load());
        CHECK(rig.recorder.CompletedSeq() >= seq);
        CHECK(asked_marker.load() == 5);
        CHECK(asked_address.load() == Rig::STORAGE + 0x100);
        CHECK(asked_size.load() == 0x40);
        // Only the submissions made before the call count: later writes are not waited for.
        CHECK(max_seq.load() == seq);
    }
    SECTION("no unfinished writer: the call returns at once") {
        rig.sink.write_marker = 5;
        rig.sink.poll = [&](u64, u64, u64, u64) {
            GpuWriteState state;
            state.wait_limit_ms = 5000;
            return state;
        };
        const auto begin = std::chrono::steady_clock::now();
        invalidate();
        CHECK(std::chrono::steady_clock::now() - begin < std::chrono::milliseconds{500});
    }
    SECTION("a submission that never completes is waited for only up to the limit") {
        rig.sink.write_marker = 5;
        rig.sink.poll = [&](u64, u64, u64, u64) {
            GpuWriteState state;
            state.wait_limit_ms = 60;
            state.writer_seq = seq;
            return state;
        };
        const auto begin = std::chrono::steady_clock::now();
        invalidate();
        CHECK(std::chrono::steady_clock::now() - begin >= std::chrono::milliseconds{60});
        CHECK(rig.recorder.CompletedSeq() < seq);
    }
}

TEST_CASE("nvn hle queue: wait sync ignores a sync nobody fenced", "[nvn]") {
    Rig rig;
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
    REQUIRE(rig.Call("nvnQueueInitialize", {Rig::I(QUEUE), Rig::I(BUILDER)}) == 1);
    REQUIRE(rig.recorder.SyncInitialize(SYNC));
    // The acquire sync of a window is never fenced: a GPU-side wait on it would park the replay.
    CHECK(rig.Call("nvnQueueWaitSync", {Rig::I(QUEUE), Rig::I(SYNC)}) == 1);
    CHECK(rig.sink.submissions.empty());
    rig.Call("nvnQueueFenceSync", {Rig::I(QUEUE), Rig::I(SYNC), Rig::I(0), Rig::I(0)});
    const size_t before = rig.sink.submissions.size();
    CHECK(rig.Call("nvnQueueWaitSync", {Rig::I(QUEUE), Rig::I(SYNC)}) == 1);
    CHECK(rig.sink.submissions.size() == before + 1);
}

TEST_CASE("nvn hle window: acquire hands out free slots, present frees them on completion",
          "[nvn]") {
    WaitProbe probe;
    Rig rig;
    probe.Install(rig);
    MakeWindow(rig, 3);
    const u64 index = Rig::SCRATCH + 0x100;
    CHECK(rig.Call("nvnWindowGetNumTextures", {Rig::I(WINDOW)}) == 3);
    // Crop defaults to the first texture.
    rig.Call("nvnWindowGetCrop", {Rig::I(WINDOW), Rig::I(index)});
    CHECK(rig.memory.Get<s32>(index + 8) == 1280);
    CHECK(rig.memory.Get<s32>(index + 12) == 720);

    std::vector<s32> got;
    for (int i = 0; i < 3; ++i) {
        REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index)}) == 0);
        got.push_back(rig.memory.Get<s32>(index));
    }
    CHECK(got == std::vector<s32>{0, 1, 2});

    // All three are held by the game: the next acquire blocks until one is presented, completed on
    // the GPU and replaced on the display by the next frame.
    std::atomic<int> acquired{-1};
    std::thread acquirer{[&] {
        rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index + 0x10)});
        acquired = rig.memory.Peek<s32>(index + 0x10);
    }};
    CHECK(probe.AwaitWaiting());
    CHECK(acquired.load() == -1);
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(0)});
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(1)});
    if (rig.sink.presents.size() != 2) {
        // Nothing would free a slot: release the acquirer before failing.
        probe.cancel = true;
        acquirer.join();
        FAIL("expected 2 presents, got " << rig.sink.presents.size());
    }
    CHECK(rig.sink.presents[0].window_slot == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    CHECK(acquired.load() == -1); // presented and shown, but the presents have not completed on the GPU
    rig.sink.Complete(rig.sink.presents[0].seq);
    acquirer.join();
    CHECK(acquired.load() == 0); // slot 1 is still on display
}

TEST_CASE("nvn hle window: the default frame queue limit waits for the previous present",
          "[nvn]") {
    ScopedEnv legacy{"EDEN_NX2_TRY_MAX_FRAMES_IN_FLIGHT", nullptr};
    ScopedEnv configured{"EDEN_NX2_MAX_FRAMES_IN_FLIGHT", nullptr};
    WaitProbe probe;
    Rig rig;
    probe.Install(rig);
    MakeWindow(rig, 3);
    const u64 index = Rig::SCRATCH + 0x100;
    for (int i = 0; i < 2; ++i) {
        REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index)}) == 0);
    }
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(0)});
    REQUIRE(rig.sink.presents.size() == 1);
    const u64 first_seq = rig.sink.presents[0].seq;

    std::atomic<bool> returned{false};
    std::thread presenter{[&] {
        rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(1)});
        returned = true;
    }};
    // The presenter published its request and waits for the depth limit.
    const bool waiting = probe.AwaitWaiting();
    CHECK(waiting);
    CHECK_FALSE(returned.load());
    if (waiting) {
        CHECK(rig.sink.presents.size() == 2);
    }
    rig.sink.Complete(first_seq);
    presenter.join();
    CHECK(returned.load());
}

TEST_CASE("nvn hle window: EDEN_NX2_MAX_FRAMES_IN_FLIGHT=2 allows two presents then waits for "
          "the oldest",
          "[nvn]") {
    ScopedEnv legacy{"EDEN_NX2_TRY_MAX_FRAMES_IN_FLIGHT", nullptr};
    ScopedEnv configured{"EDEN_NX2_MAX_FRAMES_IN_FLIGHT", "2"};
    WaitProbe probe;
    Rig rig;
    probe.Install(rig);
    MakeWindow(rig, 3);
    const u64 index = Rig::SCRATCH + 0x100;
    for (int slot = 0; slot < 3; ++slot) {
        REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index)}) == 0);
    }
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(0)});
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(1)});
    REQUIRE(rig.sink.presents.size() == 2);
    const u64 first_seq = rig.sink.presents[0].seq;

    std::atomic<bool> returned{false};
    std::thread presenter{[&] {
        rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(2)});
        returned = true;
    }};
    // The presenter published its request and waits for the depth limit.
    const bool waiting = probe.AwaitWaiting();
    CHECK(waiting);
    CHECK_FALSE(returned.load());
    if (waiting) {
        CHECK(rig.sink.presents.size() == 3);
    }
    rig.sink.Complete(first_seq);
    presenter.join();
    CHECK(returned.load());
}

TEST_CASE("nvn hle window: EDEN_NX2_MAX_FRAMES_IN_FLIGHT=0 or off disables the extra GPU wait",
          "[nvn]") {
    for (const char* value : {"0", "off"}) {
        INFO(value);
        ScopedEnv legacy{"EDEN_NX2_TRY_MAX_FRAMES_IN_FLIGHT", nullptr};
        ScopedEnv configured{"EDEN_NX2_MAX_FRAMES_IN_FLIGHT", value};
        Rig rig;
        MakeWindow(rig, 3);
        const u64 index = Rig::SCRATCH + 0x100;
        for (int slot = 0; slot < 3; ++slot) {
            REQUIRE(rig.Call("nvnWindowAcquireTexture",
                             {Rig::I(WINDOW), Rig::I(0), Rig::I(index)}) == 0);
            rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(slot)});
        }
        CHECK(rig.sink.presents.size() == 3); // none completed, none waited
    }
}

TEST_CASE("nvn hle window: the crop reaches the presents after it", "[nvn]") {
    Rig rig;
    MakeWindow(rig, 2);
    const u64 index = Rig::SCRATCH + 0x100;
    REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index)}) == 0);
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(0)});
    rig.Call("nvnWindowSetCrop",
             {Rig::I(WINDOW), Rig::I(0), Rig::I(0), Rig::I(1024), Rig::I(576)});
    REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index)}) == 0);
    rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(1)});
    REQUIRE(rig.sink.presents.size() == 2);
    CHECK(rig.sink.presents[0].crop.width == 0); // unset: the whole texture
    CHECK(rig.sink.presents[1].crop.width == 1024);
    CHECK(rig.sink.presents[1].crop.height == 576);
}

TEST_CASE("nvn hle window: oldest presented slot is reused first, present interval round trip",
          "[nvn]") {
    Rig rig;
    MakeWindow(rig, 2);
    const u64 index = Rig::SCRATCH + 0x100;
    CHECK(rig.Call("nvnWindowGetPresentInterval", {Rig::I(WINDOW)}) == 1);
    rig.Call("nvnWindowSetPresentInterval", {Rig::I(WINDOW), Rig::I(2)});
    CHECK(rig.Call("nvnWindowGetPresentInterval", {Rig::I(WINDOW)}) == 2);
    for (int frame = 0; frame < 4; ++frame) {
        rig.Call("nvnWindowAcquireTexture", {Rig::I(WINDOW), Rig::I(0), Rig::I(index)});
        const s32 slot = rig.memory.Get<s32>(index);
        CHECK(slot == frame % 2);
        rig.Call("nvnQueuePresentTexture", {Rig::I(QUEUE), Rig::I(WINDOW), Rig::I(slot)});
        rig.sink.Complete(rig.sink.presents.back().seq);
    }
    CHECK(rig.sink.presents.back().present_interval == 2);
    CHECK(rig.runtime.presents.load() == 4);
}

TEST_CASE("nvn hle nv: observers record the pool and leave the registers alone", "[nvn]") {
    Rig rig;
    CHECK_FALSE(rig.runtime.nv_initialized.load());
    const auto ctx = rig.Invoke("_ZN2nv18InitializeGraphicsEPvm",
                                {Rig::I(0x2000'0000), Rig::I(0x1000'0000)});
    CHECK(rig.runtime.nv_initialized.load());
    CHECK(rig.runtime.nv_pool_size.load() == 0x1000'0000);
    // The original function runs next with its arguments intact.
    CHECK(ctx.r[0] == 0x2000'0000);
    CHECK(ctx.r[1] == 0x1000'0000);
    rig.Call("_ZN2nv16FinalizeGraphicsEv");
    CHECK_FALSE(rig.runtime.nv_initialized.load());
}

TEST_CASE("nvn hle window: builder defaults in SDK order and the texture count clamp", "[nvn]") {
    Rig rig;
    rig.Call("nvnWindowBuilderSetDefaults", {Rig::I(BUILDER)});
    // sdk+0x5efbe0: Rec.709 primaries, D65, then minimum 0.03 and maximum 1000 nits, content
    // light levels 200 / 60.
    CHECK(rig.memory.Get<std::array<f32, 12>>(BUILDER + 0x30) ==
          std::array<f32, 12>{0.64f, 0.33f, 0.30f, 0.60f, 0.15f, 0.06f, 0.3127f, 0.3290f, 0.03f,
                              1000.0f, 200.0f, 60.0f});
    // The count is stored as given (sdk+0x5efc20)...
    rig.Call("nvnWindowBuilderSetTextures", {Rig::I(BUILDER), Rig::I(0xFFFF'FFFFull), Rig::I(Rig::SCRATCH)});
    CHECK(rig.Call("nvnWindowBuilderGetNumTextures", {Rig::I(BUILDER)}) == 0xFFFF'FFFFull);
    // ...and WindowInitialize takes at most four (sdk+0x5efdc0).
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
    for (u32 i = 0; i < 4; ++i) {
        const u64 texture = Rig::Obj(20 + i);
        REQUIRE(rig.recorder.TextureInitialize(texture, WindowTextureDesc(), POOL, 0x10000ull * i));
        rig.memory.Put<u64>(Rig::SCRATCH + 8 * i, texture);
    }
    rig.Call("nvnWindowBuilderSetTextures", {Rig::I(BUILDER), Rig::I(6), Rig::I(Rig::SCRATCH)});
    REQUIRE(rig.Call("nvnWindowInitialize", {Rig::I(WINDOW), Rig::I(BUILDER)}) == 1);
    CHECK(rig.Call("nvnWindowGetNumTextures", {Rig::I(WINDOW)}) == 4);
}

TEST_CASE("nvn hle program: SetShaders refuses entries the SDK refuses, builder keeps the device",
          "[nvn]") {
    Rig rig;
    constexpr u64 PROGRAM = Rig::Obj(9);
    REQUIRE(rig.Call("nvnProgramInitialize", {Rig::I(PROGRAM), Rig::I(DEVICE)}) == 1);
    const u64 data = Rig::SCRATCH + 0x1000;
    // NVNshaderData {data, control}: no code, then a misaligned control section.
    rig.memory.Put(data, std::array<u64, 2>{0, Rig::SCRATCH});
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(1), Rig::I(data)}) == 0);
    rig.memory.Put(data, std::array<u64, 2>{Rig::STORAGE, Rig::SCRATCH + 4});
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(1), Rig::I(data)}) == 0);
    rig.memory.Put(data, std::array<u64, 3>{Rig::STORAGE, 0, 0});
    CHECK(rig.Call("nvnProgramSetShadersExt", {Rig::I(PROGRAM), Rig::I(1), Rig::I(data)}) == 0);
    // A refused set leaves an initialized (emptied) program; no shaders is a valid set.
    CHECK(rig.memory.Get<GuestTag>(PROGRAM).kind == static_cast<u32>(ObjectKind::Program));
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(0), Rig::I(data)}) == 1);

    rig.Call("nvnProgramBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
    rig.Call("nvnProgramBuilderSetShaders", {Rig::I(BUILDER), Rig::I(2), Rig::I(data)});
    rig.Call("nvnProgramBuilderSetDefaults", {Rig::I(BUILDER)});
    CHECK(rig.Call("nvnProgramBuilderGetDevice", {Rig::I(BUILDER)}) == DEVICE);
    CHECK(rig.Call("nvnProgramBuilderGetShaderCount", {Rig::I(BUILDER)}) == 0);
}

TEST_CASE("nvn hle bridge: the sdk's GetProcAddress names come from its own table", "[nvn]") {
    std::vector<u8> image(0x1000, 0);
    const auto put_string = [&](size_t offset, std::string_view text) {
        std::memcpy(image.data() + offset, text.data(), text.size());
        image[offset + text.size()] = 0;
    };
    const auto put_u32 = [&](size_t offset, u32 value) {
        std::memcpy(image.data() + offset, &value, sizeof(value));
    };
    // The blob follows non-NUL bytes, so its first name is not NUL-delimited on the left.
    std::memset(image.data() + 0xF0, 'Q', 0x10);
    put_string(0x100, "Alpha");
    put_string(0x106, "DeviceGetProcAddress");
    put_string(0x11B, "Zeta");
    // The u32 offsets from the blob's start, in name order.
    put_u32(0x200, 0);
    put_u32(0x204, 0x06);
    put_u32(0x208, 0x1B);
    // Extension names; a non-identifier-looking one and a table name are not extensions.
    put_string(0x300, "nvnQueueSetTimeoutNVX");
    put_string(0x320, "nvn_no_vsync_capability");
    put_string(0x340, "nvnZeta");
    const Core::NvnHle::SdkProcNames names = Core::NvnHle::FindSdkProcNames(image);
    CHECK(names.table ==
          std::vector<std::string>{"nvnAlpha", "nvnDeviceGetProcAddress", "nvnZeta"});
    CHECK(names.extras == std::vector<std::string>{"nvnQueueSetTimeoutNVX"});
    // No table: nothing.
    CHECK(Core::NvnHle::FindSdkProcNames(std::vector<u8>(0x100, 0)).table.empty());
}

TEST_CASE("nvn hle bridge: the sdk's name table is found when its blob starts after a NUL or an "
          "unrelated name",
          "[nvn]") {
    const auto build = [](bool unrelated_prefix) {
        std::vector<u8> image(0x1000, 0);
        const auto put_string = [&](size_t offset, std::string_view text) {
            std::memcpy(image.data() + offset, text.data(), text.size());
        };
        const auto put_u32 = [&](size_t offset, u32 value) {
            std::memcpy(image.data() + offset, &value, sizeof(value));
        };
        std::memset(image.data() + 0xF0, 'Q', 0x0F); // NUL at 0xFF: the blob is NUL-delimited
        if (unrelated_prefix) {
            put_string(0x100, "Aardvark"); // sorts below the table's first name
            put_string(0x109, "Alpha");
            put_string(0x10F, "DeviceGetProcAddress");
            put_string(0x124, "Zeta");
            put_u32(0x200, 0); // table offsets are relative to "Alpha"
            put_u32(0x204, 0x06);
            put_u32(0x208, 0x1B);
        } else {
            put_string(0x100, "Alpha");
            put_string(0x106, "DeviceGetProcAddress");
            put_string(0x11B, "Zeta");
            put_u32(0x200, 0);
            put_u32(0x204, 0x06);
            put_u32(0x208, 0x1B);
        }
        return image;
    };
    const std::vector<std::string> expected{"nvnAlpha", "nvnDeviceGetProcAddress", "nvnZeta"};
    CHECK(Core::NvnHle::FindSdkProcNames(build(false)).table == expected);
    CHECK(Core::NvnHle::FindSdkProcNames(build(true)).table == expected);
}
