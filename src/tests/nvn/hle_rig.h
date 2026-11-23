// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/kernel/svc_types.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/recorder.h"

/// Test rig for the NVN HLE function bodies: a fake guest memory, a capturing sink, a Recorder and
/// the registry of every family. `Invoke("nvnFoo", {Rig::I(va), Rig::I(3), Rig::F(0.5f)})` builds
/// the AAPCS64 register state (integers/pointers in x0.., floats in v0..) and runs the body, as the
/// bridge would. Stack arguments are not supported (none of the NVN functions needs more than
/// eight integer registers).
namespace NvnHleTest {

class FakeMemory : public VideoCommon::Nvn::GuestMemory {
public:
    /// Maps `size` zeroed bytes at `base`.
    void Map(u64 base, u64 size) {
        regions.push_back({base, std::vector<u8>(size, 0)});
    }

    bool Read(u64 va, void* dst, size_t size) override {
        const u8* source = Find(va, size);
        if (source == nullptr) {
            return false;
        }
        std::memcpy(dst, source, size);
        return true;
    }
    bool Write(u64 va, const void* src, size_t size) override {
        u8* target = Find(va, size);
        if (target == nullptr) {
            return false;
        }
        std::memcpy(target, src, size);
        return true;
    }
    u8* Pointer(u64 va) override {
        return Find(va, 1);
    }

    template <typename T>
    void Put(u64 va, const T& value) {
        REQUIRE(Write(va, &value, sizeof(T)));
    }
    template <typename T>
    T Get(u64 va) {
        T value{};
        REQUIRE(Read(va, &value, sizeof(T)));
        return value;
    }
    /// Get without Catch assertions (usable from helper threads).
    template <typename T>
    T Peek(u64 va) {
        T value{};
        Read(va, &value, sizeof(T));
        return value;
    }
    void PutString(u64 va, const std::string& text) {
        REQUIRE(Write(va, text.c_str(), text.size() + 1));
    }

private:
    struct Region {
        u64 base;
        std::vector<u8> bytes;
    };
    u8* Find(u64 va, size_t size) {
        for (Region& region : regions) {
            if (va >= region.base && va - region.base <= region.bytes.size() &&
                size <= region.bytes.size() - (va - region.base)) {
                return region.bytes.data() + (va - region.base);
            }
        }
        return nullptr;
    }
    std::vector<Region> regions;
};

/// One Sink::NotifyEventCpu call.
struct EventCpuNote {
    u64 address;
    VideoCommon::Nvn::EventCpuChange change;
    u32 value;
};

class CaptureSink final : public VideoCommon::Nvn::Sink {
public:
    void CreatePool(const VideoCommon::Nvn::PoolDesc& desc) override {
        pools.push_back(desc);
    }
    void CreateTexture(const VideoCommon::Nvn::TextureDesc& desc) override {
        textures.push_back(desc);
    }
    void CreateSampler(const VideoCommon::Nvn::SamplerDesc& desc) override {
        samplers.push_back(desc);
    }
    void CreateProgram(const VideoCommon::Nvn::ProgramDesc& desc) override {
        programs.push_back(desc);
    }
    void Destroy(u64 serial) override {
        destroyed.push_back(serial);
    }
    void Submit(VideoCommon::Nvn::Submission&& submission) override {
        submissions.push_back(std::move(submission));
    }
    void Present(const VideoCommon::Nvn::PresentRequest& request) override {
        presents.push_back(request);
    }
    bool Wait(u64, std::chrono::nanoseconds) override {
        return true;
    }
    void SetCompletionCallback(std::function<void(u64)> callback_) override {
        callback = std::move(callback_);
    }
    void SetGuestMemory(VideoCommon::Nvn::GuestMemory* memory_) override {
        memory = memory_;
    }
    void MapVirtual(u64 pool_serial,
                    std::vector<VideoCommon::Nvn::VirtualMapping>&& mappings) override {
        virtual_maps.emplace_back(pool_serial, std::move(mappings));
    }
    void NotifyCpuWrite(u64 address, u64 size) override {
        cpu_writes.emplace_back(address, size);
    }
    void NotifyHostWrite(u64 address, u64 size) override {
        host_writes.emplace_back(address, size);
    }
    void NotifyTextureWritten(u64 texture_serial) override {
        texture_writes.push_back(texture_serial);
    }
    bool FlushTextureToGuest(u64 texture_serial) override {
        flushed.push_back(texture_serial);
        return true;
    }
    void NotifyEventCpu(u64 address, VideoCommon::Nvn::EventCpuChange change,
                        u32 value) override {
        event_cpu.push_back({address, change, value});
    }
    /// Sink::GpuWriteMarker / PollGpuWrites: tracking is off while `write_marker` is 0.
    u64 GpuWriteMarker() override {
        return write_marker;
    }
    VideoCommon::Nvn::GpuWriteState PollGpuWrites(u64 marker, u64 address, u64 size,
                                                  u64 max_seq) override {
        return poll ? poll(marker, address, size, max_seq) : VideoCommon::Nvn::GpuWriteState{};
    }
    u64 write_marker = 0;
    std::function<VideoCommon::Nvn::GpuWriteState(u64, u64, u64, u64)> poll;

    /// The GPU finished everything up to `seq`: callbacks arrive once per seq, in order.
    void Complete(u64 seq) {
        for (u64 next = completed + 1; next <= seq; ++next) {
            callback(next);
        }
        completed = std::max(completed, seq);
    }

    VideoCommon::Nvn::GuestMemory* memory = nullptr;
    std::function<void(u64)> callback;
    u64 completed = 0;
    std::vector<VideoCommon::Nvn::PoolDesc> pools;
    std::vector<VideoCommon::Nvn::TextureDesc> textures;
    std::vector<VideoCommon::Nvn::SamplerDesc> samplers;
    std::vector<VideoCommon::Nvn::ProgramDesc> programs;
    std::vector<u64> destroyed;
    std::vector<VideoCommon::Nvn::Submission> submissions;
    std::vector<VideoCommon::Nvn::PresentRequest> presents;
    std::vector<std::pair<u64, std::vector<VideoCommon::Nvn::VirtualMapping>>> virtual_maps;
    std::vector<std::pair<u64, u64>> cpu_writes;
    std::vector<std::pair<u64, u64>> host_writes;
    std::vector<u64> texture_writes;
    std::vector<u64> flushed;
    std::vector<EventCpuNote> event_cpu;
};

/// One call argument: an integer/pointer or a float/double.
struct Arg {
    enum class Kind { Int, F32, F64 } kind;
    u64 bits;
};

class Rig {
public:
    static constexpr u64 BASE = 0x1000'0000;
    static constexpr u64 MEMORY_SIZE = 0x400000;
    /// Pool storage the tests may use (page aligned), 1 MiB.
    static constexpr u64 STORAGE = BASE + 0x200000;
    static constexpr u64 STORAGE_SIZE = 0x100000;
    /// Scratch for argument data passed by pointer.
    static constexpr u64 SCRATCH = BASE + 0x100000;

    Rig() {
        memory.Map(BASE, MEMORY_SIZE);
        // Waits poll on the host: the rig has no kernel.
        runtime.SetWaitHook([](const std::function<bool()>& ready, std::chrono::nanoseconds timeout) {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (!ready()) {
                if (timeout.count() >= 0 && std::chrono::steady_clock::now() >= deadline) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds{50});
            }
            return true;
        });
        Core::NvnHle::Hle::RegisterAll(registry);
        REQUIRE(registry.Rejected() == 0);
    }

    /// Address of the n-th 0x100-byte guest object slot (low memory, zeroed).
    static constexpr u64 Obj(u32 n) {
        return BASE + 0x100ULL * n;
    }

    static Arg I(u64 value) {
        return {Arg::Kind::Int, value};
    }
    static Arg F(float value) {
        return {Arg::Kind::F32, std::bit_cast<u32>(value)};
    }
    static Arg D(double value) {
        return {Arg::Kind::F64, std::bit_cast<u64>(value)};
    }

    /// Runs the registered body of `name`. Returns the thread context after the call.
    Kernel::Svc::ThreadContext Invoke(const std::string& name, std::initializer_list<Arg> args = {}) {
        // No Catch assertions here: tests call this from helper threads.
        const auto it = registry.Handlers().find(name);
        if (it == registry.Handlers().end()) {
            throw std::runtime_error("no NVN body registered for " + name);
        }
        Kernel::Svc::ThreadContext ctx{};
        size_t x = 0;
        size_t v = 0;
        for (const Arg& arg : args) {
            if (arg.kind == Arg::Kind::Int) {
                if (x >= 8) {
                    throw std::runtime_error("more than eight integer arguments");
                }
                ctx.r[x++] = arg.bits;
            } else {
                if (v >= 8) {
                    throw std::runtime_error("more than eight float arguments");
                }
                ctx.v[v++] = {arg.bits, 0};
            }
        }
        ctx.sp = SCRATCH + 0x8000;
        runtime.Run(ctx, memory, recorder, it->second);
        return ctx;
    }
    /// Invoke and return x0.
    u64 Call(const std::string& name, std::initializer_list<Arg> args = {}) {
        return Invoke(name, args).r[0];
    }
    /// Invoke and return the float in v0.
    float CallF(const std::string& name, std::initializer_list<Arg> args = {}) {
        return std::bit_cast<float>(static_cast<u32>(Invoke(name, args).v[0][0]));
    }

    FakeMemory memory;
    CaptureSink sink;
    VideoCommon::Nvn::Recorder recorder{memory, sink};
    Core::NvnHle::Hle::Runtime runtime;
    Core::NvnHle::Hle::Registry registry{nullptr};
};

/// The rig's polling wait hook, plus what the tests need to synchronise with a guest call on a
/// helper thread: how many waits found their condition false, and a way to release them (a
/// released wait returns false, as a stopped emulator does). Declare it before the Rig.
struct WaitProbe {
    std::atomic<u32> waiting{0};
    std::atomic<bool> cancel{false};
    /// Whether a wait asked for no timeout.
    std::atomic<bool> saw_infinite{false};
    /// When not negative, a wait with no timeout gives up after this long (set before Install).
    std::chrono::nanoseconds infinite_limit{-1};

    void Install(Rig& rig) {
        rig.runtime.SetWaitHook(
            [this](const std::function<bool()>& ready, std::chrono::nanoseconds timeout) {
                if (timeout.count() < 0) {
                    saw_infinite = true;
                    timeout = infinite_limit;
                }
                const auto deadline = std::chrono::steady_clock::now() + timeout;
                bool counted = false;
                while (!ready()) {
                    if (!counted) {
                        counted = true;
                        waiting.fetch_add(1);
                    }
                    if (cancel.load() || (timeout.count() >= 0 &&
                                          std::chrono::steady_clock::now() >= deadline)) {
                        return false;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds{50});
                }
                return true;
            });
    }

    /// Whether at least `count` waits found their condition false within 5 s.
    bool AwaitWaiting(u32 count = 1) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (waiting.load() < count) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
        return true;
    }
};

} // namespace NvnHleTest
