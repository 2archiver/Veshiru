// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/recorder.h"

using namespace VideoCommon::Nvn;

namespace {

constexpr u64 BASE = 0x1000'0000;
constexpr u64 MEMORY_SIZE = 0x300000;
constexpr u64 STORAGE = BASE + 0x100000; // pool storage (and so the pool's GPU address)
constexpr u64 STORAGE_SIZE = 0x100000;
constexpr u64 SCRATCH = BASE + 0x80000; // argument data the "game" passes by pointer

constexpr u64 Obj(u32 n) {
    return BASE + 0x100ULL * n;
}

class FakeMemory final : public GuestMemory {
public:
    FakeMemory() : bytes(MEMORY_SIZE, 0) {}

    bool Read(u64 va, void* dst, size_t size) override {
        if (va < BASE || va + size > BASE + bytes.size()) {
            return false;
        }
        std::memcpy(dst, &bytes[va - BASE], size);
        return true;
    }
    bool Write(u64 va, const void* src, size_t size) override {
        if (va < BASE || va + size > BASE + bytes.size()) {
            return false;
        }
        std::memcpy(&bytes[va - BASE], src, size);
        return true;
    }
    u8* Pointer(u64 va) override {
        return va < BASE || va >= BASE + bytes.size() ? nullptr : &bytes[va - BASE];
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

    std::vector<u8> bytes;
};

/// One Sink::NotifyEventCpu call.
struct EventCpuNote {
    u64 address;
    EventCpuChange change;
    u32 value;
};

class CaptureSink final : public Sink {
public:
    void CreatePool(const PoolDesc& desc) override {
        pools.push_back(desc);
    }
    void CreateTexture(const TextureDesc& desc) override {
        textures.push_back(desc);
    }
    void CreateSampler(const SamplerDesc& desc) override {
        samplers.push_back(desc);
    }
    void CreateProgram(const ProgramDesc& desc) override {
        programs.push_back(desc);
    }
    void Destroy(u64 serial) override {
        destroyed.push_back(serial);
    }
    void Submit(Submission&& submission) override {
        submissions.push_back(std::move(submission));
    }
    void Present(const PresentRequest& request) override {
        presents.push_back(request);
    }
    bool Wait(u64, std::chrono::nanoseconds) override {
        return true;
    }
    void SetCompletionCallback(std::function<void(u64)> callback_) override {
        callback = std::move(callback_);
    }
    void SetGuestMemory(GuestMemory* memory_) override {
        memory = memory_;
    }
    void NotifyEventCpu(u64 address, EventCpuChange change, u32 value) override {
        event_cpu.push_back({address, change, value});
    }
    void NoteTextureHolders(TextureHolderReport&& report) override {
        holders = std::move(report);
    }

    /// The GPU finished everything up to `seq`: callbacks arrive once per seq, in order.
    void Complete(u64 seq) {
        for (u64 next = completed + 1; next <= seq; ++next) {
            callback(next);
        }
        completed = std::max(completed, seq);
    }

    GuestMemory* memory = nullptr;
    std::function<void(u64)> callback;
    u64 completed = 0;
    std::vector<PoolDesc> pools;
    std::vector<TextureDesc> textures;
    std::vector<SamplerDesc> samplers;
    std::vector<ProgramDesc> programs;
    std::vector<u64> destroyed;
    std::vector<Submission> submissions;
    std::vector<PresentRequest> presents;
    std::vector<EventCpuNote> event_cpu;
    TextureHolderReport holders;
};

std::vector<Op> Ops(const std::vector<u64>& list) {
    std::vector<Op> ops;
    RecordReader reader{list.data(), list.size()};
    while (const RecordHeader* header = reader.Next()) {
        ops.push_back(RecordOp(header));
    }
    return ops;
}

const RecordHeader* FindRecord(const std::vector<u64>& list, Op op, size_t nth = 0) {
    RecordReader reader{list.data(), list.size()};
    while (const RecordHeader* header = reader.Next()) {
        if (RecordOp(header) == op && nth-- == 0) {
            return header;
        }
    }
    return nullptr;
}

/// The payload of the nth `op` record, which the list must hold.
template <typename T>
const T* RequiredPayload(const std::vector<u64>& list, Op op, size_t nth = 0) {
    const RecordHeader* header = FindRecord(list, op, nth);
    REQUIRE(header != nullptr);
    return RecordPayload<T>(header);
}

TextureDesc MakeTextureDesc(u32 width = 64, u32 height = 64) {
    TextureDesc desc{};
    desc.target = 1;
    desc.format = 0x25;
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.levels = 1;
    return desc;
}

struct Fixture {
    static constexpr u64 POOL = Obj(0);
    static constexpr u64 QUEUE = Obj(1);
    static constexpr u64 CMD = Obj(2);

    Fixture() {
        REQUIRE(recorder.MemoryPoolInitialize(POOL, STORAGE, STORAGE_SIZE, 0));
        REQUIRE(recorder.QueueInitialize(QUEUE));
        REQUIRE(recorder.CommandBufferInitialize(CMD));
    }

    /// Records a list into the command memory at `offset` of the pool and returns its handle.
    u64 Record(u64 offset, const std::function<void()>& commands, u64 size = 0x1000) {
        recorder.AddCommandMemory(CMD, POOL, static_cast<s64>(offset), size);
        recorder.BeginRecording(CMD);
        commands();
        return recorder.EndRecording(CMD);
    }

    FakeMemory memory;
    CaptureSink sink;
    Recorder recorder{memory, sink};
};

} // namespace

TEST_CASE("MKW command-list batching preserves large repeated submissions", "[nvn][mkw]") {
    std::vector<u64> expected;
    for (bool batch : {false, true}) {
        Fixture f;
        f.recorder.SetCommandListBatching(batch);
        const u64 handle = f.Record(0, [&] {
            for (u32 i = 0; i < 2048; ++i) {
                f.recorder.DrawArrays(Fixture::CMD, 4, i, 3 + i % 20, 1, 0);
            }
        }, 0x40000);
        REQUIRE(handle != 0);
        const auto start = std::chrono::steady_clock::now();
        for (u32 i = 0; i < 200; ++i) {
            const u64 seq = f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle});
            REQUIRE(seq != 0);
            REQUIRE(f.sink.submissions.size() == 1);
            REQUIRE(f.sink.submissions[0].lists.size() == 1);
            const std::vector<u64>& words = f.sink.submissions[0].lists[0];
            if (expected.empty()) {
                // The reference list holds every draw, in order.
                u32 draws = 0;
                RecordReader reader{words.data(), words.size()};
                while (const RecordHeader* header = reader.Next()) {
                    if (RecordOp(header) != Op::Draw) {
                        continue;
                    }
                    const auto* draw = RecordPayload<DrawPayload>(header);
                    REQUIRE(draw->first == static_cast<s32>(draws));
                    REQUIRE(draw->count == 3 + draws % 20);
                    ++draws;
                }
                REQUIRE(draws == 2048);
                expected = words;
            }
            REQUIRE(words == expected);
            f.sink.Complete(seq);
            f.sink.submissions.clear();
        }
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - start).count() / 200;
        std::printf("MKW 2048-record submit: batching=%u %.2f us (includes verification)\n", batch, us);
    }
}

TEST_CASE("nvn recorder: tagged objects, serials and re-initialization", "[nvn]") {
    Fixture f;
    REQUIRE(f.sink.memory == &f.memory);
    REQUIRE(f.sink.pools.size() == 1);
    CHECK(f.sink.pools[0].guest_address == STORAGE);
    CHECK(f.sink.pools[0].size == STORAGE_SIZE);

    const u64 texture_va = Obj(10);
    REQUIRE(f.recorder.TextureInitialize(texture_va, MakeTextureDesc(), Fixture::POOL, 0x4000));
    REQUIRE(f.sink.textures.size() == 1);
    const TextureDesc first = f.sink.textures[0];
    CHECK(first.width == 64);
    CHECK(first.pool_serial == f.sink.pools[0].serial);
    CHECK(first.pool_offset == 0x4000);
    CHECK(first.view_of == 0);

    const GuestTag tag = f.memory.Get<GuestTag>(texture_va);
    CHECK(tag.magic == GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(ObjectKind::Texture));
    CHECK(tag.serial == first.serial);
    CHECK(first.serial != 0);

    // Games copy objects by value: the tag travels and resolves to the same host object.
    const u64 copy_va = Obj(11);
    f.memory.Put(copy_va, tag);
    const std::optional<TextureDesc> copy = f.recorder.GetTexture(copy_va);
    REQUIRE(copy.has_value());
    CHECK(copy->serial == first.serial);

    // Initializing the same address again gives a new serial; the old object is untouched.
    REQUIRE(f.recorder.TextureInitialize(texture_va, MakeTextureDesc(128, 32), Fixture::POOL, 0));
    REQUIRE(f.sink.textures.size() == 2);
    CHECK(f.sink.textures[1].serial != first.serial);
    CHECK(f.memory.Get<GuestTag>(texture_va).serial == f.sink.textures[1].serial);
    CHECK(f.recorder.GetTexture(copy_va)->serial == first.serial);
    CHECK(f.recorder.Stats().reinitialized_objects == 1);

    // Buffers: tag plus the guest-written offset/size at +0x18/+0x20, read at every use.
    const u64 buffer_va = Obj(12);
    REQUIRE(f.recorder.BufferInitialize(buffer_va, Fixture::POOL, 0x100, 0x40));
    CHECK(f.recorder.GetBuffer(buffer_va)->address == STORAGE + 0x100);
    f.memory.Put<u64>(buffer_va + 0x18, 0x200);
    f.memory.Put<u64>(buffer_va + 0x20, 0x80);
    const BufferInfo info = *f.recorder.GetBuffer(buffer_va);
    CHECK(info.address == STORAGE + 0x200);
    CHECK(info.size == 0x80);
}

TEST_CASE("nvn recorder: descriptor pool bases and snapshots", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    REQUIRE(f.recorder.TextureInitialize(Obj(11), MakeTextureDesc(), Fixture::POOL, 0x10000));
    REQUIRE(f.recorder.TexturePoolInitialize(Obj(20), Fixture::POOL, 0x20000, 100));
    REQUIRE(f.recorder.TexturePoolInitialize(Obj(21), Fixture::POOL, 0x30000, 200));
    CHECK(f.recorder.GetTexturePool(Obj(20))->base == 0);
    CHECK(f.recorder.GetTexturePool(Obj(21))->base == 256);

    f.recorder.RegisterTexture(Obj(20), 5, Obj(10), 0);
    f.recorder.RegisterTexture(Obj(21), 3, Obj(11), 0);

    SamplerDesc sampler{};
    sampler.min_filter = 2;
    REQUIRE(f.recorder.SamplerPoolInitialize(Obj(30), Fixture::POOL, 0x40000, 16));
    REQUIRE(f.recorder.SamplerInitialize(Obj(31), sampler));
    f.recorder.RegisterSampler(Obj(30), 2, Obj(31));

    const u64 handle = f.Record(0, [] {});
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    REQUIRE(f.sink.submissions.size() == 1);
    const Submission& submission = f.sink.submissions[0];
    REQUIRE(submission.texture_pool_snapshot.size() == 256 + 3 + 1);
    CHECK(submission.texture_pool_snapshot[5] == f.sink.textures[0].serial);
    CHECK(submission.texture_pool_snapshot[259] == f.sink.textures[1].serial);
    CHECK(submission.texture_pool_snapshot[4] == 0);
    REQUIRE(submission.sampler_pool_snapshot.size() == 3);
    CHECK(submission.sampler_pool_snapshot[2] == f.sink.samplers[0].serial);

    // A later registration does not change the snapshot that was already submitted.
    f.recorder.RegisterTexture(Obj(20), 5, Obj(11), 0);
    CHECK(f.sink.submissions[0].texture_pool_snapshot[5] == f.sink.textures[0].serial);

    // A view registration creates a view object that refers to the texture.
    std::array<u8, 16> view{};
    view[0] = GUEST_VIEW_SET_LEVELS;
    view[2] = 1; // base_level
    view[3] = 2; // nlevels
    f.memory.Put(SCRATCH, view);
    f.recorder.RegisterTexture(Obj(20), 6, Obj(10), SCRATCH);
    REQUIRE(f.sink.textures.size() == 3);
    CHECK(f.sink.textures[2].view_of == f.sink.textures[0].serial);
    CHECK(f.sink.textures[2].view.set_levels == 1);
    CHECK(f.sink.textures[2].view.base_level == 1);
    CHECK(f.sink.textures[2].serial != f.sink.textures[0].serial);
    // The same view again reuses the object.
    f.recorder.RegisterTexture(Obj(20), 7, Obj(10), SCRATCH);
    CHECK(f.sink.textures.size() == 3);

    // Submissions with no registration in between share one table; a registration makes a new one.
    const u64 second = f.Record(0, [] {});
    const u64 third = f.Record(0, [] {});
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{second}) != 0);
    // Re-registering the same objects must retain the immutable table, including a view.
    f.recorder.RegisterTexture(Obj(20), 5, Obj(11), 0);
    f.recorder.RegisterTexture(Obj(20), 6, Obj(10), SCRATCH);
    f.recorder.RegisterSampler(Obj(30), 2, Obj(31));
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{third}) != 0);
    REQUIRE(f.sink.submissions.size() == 3);
    CHECK(f.sink.submissions[1].texture_pool_snapshot.Values().data() ==
          f.sink.submissions[2].texture_pool_snapshot.Values().data());
    CHECK(f.sink.submissions[1].sampler_pool_snapshot.Values().data() ==
          f.sink.submissions[2].sampler_pool_snapshot.Values().data());
    CHECK(f.sink.submissions[0].texture_pool_snapshot.Values().data() !=
          f.sink.submissions[2].texture_pool_snapshot.Values().data());

    // Reinitializing a registered texture replaces its pool serials in future snapshots, including
    // view registrations, but does not rewrite snapshots already submitted.
    f.recorder.RegisterTexture(Obj(20), 5, Obj(10), 0);
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(128, 32), Fixture::POOL, 0));
    REQUIRE(f.sink.textures.size() == 5);
    const u64 new_texture = f.sink.textures[3].serial;
    const u64 new_view = f.sink.textures[4].serial;
    CHECK(new_texture != f.sink.textures[0].serial);
    CHECK(new_view != f.sink.textures[2].serial);
    CHECK(f.sink.textures[4].view_of == new_texture);
    CHECK(f.sink.textures[4].view.set_levels == 1);
    const u64 refreshed = f.Record(0, [] {});
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{refreshed}) != 0);
    REQUIRE(f.sink.submissions.size() == 4);
    CHECK(f.sink.submissions[3].texture_pool_snapshot[5] == new_texture);
    CHECK(f.sink.submissions[3].texture_pool_snapshot[6] == new_view);
    CHECK(f.sink.submissions[3].texture_pool_snapshot[7] == new_view);
    CHECK(f.sink.submissions[0].texture_pool_snapshot[5] == f.sink.textures[0].serial);
    CHECK(f.sink.submissions[0].texture_pool_snapshot[6] == 0);

    // Finalizing a pool frees its range for the next one.
    f.recorder.TexturePoolFinalize(Obj(20));
    REQUIRE(f.recorder.TexturePoolInitialize(Obj(22), Fixture::POOL, 0x50000, 50));
    CHECK(f.recorder.GetTexturePool(Obj(22))->base == 0);
}

TEST_CASE("nvn recorder: full recording sequence", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));

    // A program with one NVN2 fragment stage (control section at SCRATCH, code behind it).
    std::array<u8, 0x100> control{};
    const u32 magic = NVN2_CONTROL_MAGIC;
    const u32 version = NVN2_CONTROL_VERSION;
    const u32 code_size = 16;
    const u16 stage = 1;
    std::memcpy(control.data(), &magic, 4);
    std::memcpy(control.data() + 4, &version, 4);
    std::memcpy(control.data() + 0x44, &code_size, 4);
    std::memcpy(control.data() + 0x56, &stage, 2);
    f.memory.Put(SCRATCH, control);
    const std::array<u8, 16> code{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    f.memory.Put(STORAGE + 0x8000, code);
    REQUIRE(f.recorder.ProgramInitialize(Obj(11)));
    const std::array<ShaderStageInput, 1> stages{ShaderStageInput{STORAGE + 0x8000, SCRATCH, -1}};
    REQUIRE(f.recorder.ProgramSetShaders(Obj(11), stages));
    REQUIRE(f.sink.programs.size() == 1);
    CHECK(f.sink.programs[0].stage_mask == 1U << 1);
    CHECK(f.sink.programs[0].code[1] == std::vector<u8>(code.begin(), code.end()));
    CHECK(f.sink.programs[0].control[1].size() == 0x100);

    f.memory.Put<u64>(SCRATCH + 0x1000, Obj(10)); // pointer array for SetRenderTargets
    const std::array<float, 4> color{0.25f, 0.5f, 0.75f, 1.0f};
    f.memory.Put(SCRATCH + 0x1100, color);

    const u64 command_va = STORAGE + 0x1000;
    f.recorder.AddCommandMemory(Fixture::CMD, Fixture::POOL, 0x1000, 0x1000);
    f.recorder.BeginRecording(Fixture::CMD);
    CHECK(f.recorder.IsRecording(Fixture::CMD));
    std::vector<u64> used{f.recorder.GetCommandMemoryUsed(Fixture::CMD)};
    const auto step = [&] {
        used.push_back(f.recorder.GetCommandMemoryUsed(Fixture::CMD));
        CHECK(used.back() > used[used.size() - 2]);
    };
    f.recorder.SetRenderTargets(Fixture::CMD, 1, SCRATCH + 0x1000, 0, 0, 0);
    step();
    f.recorder.ClearColor(Fixture::CMD, 0, SCRATCH + 0x1100, 0xF, false);
    step();
    f.recorder.BindProgram(Fixture::CMD, Obj(11), 0x2);
    step();
    f.recorder.DrawArrays(Fixture::CMD, 4, 0, 3, 1, 0);
    step();
    const u64 handle = f.recorder.EndRecording(Fixture::CMD);
    CHECK_FALSE(f.recorder.IsRecording(Fixture::CMD));
    CHECK(handle == command_va);

    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const Submission& submission = f.sink.submissions.at(0);
    REQUIRE(submission.lists.size() == 1);
    const std::vector<u64>& list = submission.lists[0];
    CHECK(Ops(list) == std::vector<Op>{Op::SetRenderTargets, Op::ClearColor, Op::BindProgram,
                                       Op::Draw});
    // The list ends with an End record.
    CHECK(reinterpret_cast<const RecordHeader*>(&list.back())->op == static_cast<u16>(Op::End));

    const auto* targets = RequiredPayload<SetRenderTargetsPayload>(list, Op::SetRenderTargets);
    CHECK(targets->color_count == 1);
    CHECK(targets->has_depth == 0);
    CHECK(TrailingArray<RenderTargetRef>(targets)[0].texture.serial == f.sink.textures[0].serial);
    const auto* clear = RequiredPayload<ClearColorPayload>(list, Op::ClearColor);
    CHECK(clear->mask == 0xF);
    float clear_color[4];
    std::memcpy(clear_color, clear->color, sizeof(clear_color));
    CHECK(clear_color[1] == 0.5f);
    const auto* bind = RequiredPayload<BindProgramPayload>(list, Op::BindProgram);
    CHECK(bind->program_serial == f.sink.programs[0].serial);
    CHECK(bind->stages == 2);
    const auto* draw = RequiredPayload<DrawPayload>(list, Op::Draw);
    CHECK(draw->mode == 4);
    CHECK(draw->count == 3);
    CHECK(draw->max_draws == 1);
    CHECK(submission.retained_serials.size() == 2);

    // Fence a sync behind the queue's work: signaled only once the GPU finished it.
    REQUIRE(f.recorder.SyncInitialize(Obj(12)));
    CHECK_FALSE(f.recorder.GetSyncState(Obj(12))->fenced);
    const u64 fence_seq = f.recorder.QueueFenceSync(Fixture::QUEUE, Obj(12), 0, 0);
    CHECK(fence_seq == 2);
    CHECK(f.recorder.GetSyncState(Obj(12))->fenced);
    CHECK_FALSE(f.recorder.GetSyncState(Obj(12))->signaled);
    CHECK(f.recorder.SyncWait(Obj(12), std::chrono::nanoseconds{0}) ==
          SyncWaitResult::TimeoutExpired);
    f.sink.Complete(1);
    CHECK_FALSE(f.recorder.GetSyncState(Obj(12))->signaled);
    f.sink.Complete(2);
    CHECK(f.recorder.GetSyncState(Obj(12))->signaled);
    CHECK(f.recorder.SyncWait(Obj(12), std::chrono::nanoseconds{0}) ==
          SyncWaitResult::AlreadySignaled);
    CHECK(f.recorder.QueueFinish(Fixture::QUEUE, std::chrono::nanoseconds{0}));
    const Submission& fence_submission = f.sink.submissions.at(1);
    const RecordHeader* fence = FindRecord(fence_submission.lists[0], Op::FenceSync);
    REQUIRE(fence != nullptr);
    CHECK(RecordPayload<FenceSyncPayload>(fence)->sync_serial ==
          f.memory.Get<GuestTag>(Obj(12)).serial);
}

TEST_CASE("nvn recorder: data is copied at record time", "[nvn]") {
    Fixture f;
    f.recorder.SetCommandListBatching(GENERATE(false, true));
    f.recorder.AddCommandMemory(Fixture::CMD, Fixture::POOL, 0, 0x1000);
    f.recorder.BeginRecording(Fixture::CMD);

    std::array<u8, 16> uniform_data;
    for (u8 i = 0; i < 16; ++i) {
        uniform_data[i] = static_cast<u8>(i + 1);
    }
    f.memory.Put(SCRATCH, uniform_data);
    f.recorder.UpdateUniformBuffer(Fixture::CMD, STORAGE + 0x2000, 0x10, 16, SCRATCH);

    const std::array<u8, 8> blend{1, 2, 3, 4, 5, 6, 7, 8};
    f.memory.Put(SCRATCH + 0x100, blend);
    f.recorder.BindState(Fixture::CMD, StateKind::Blend, SCRATCH + 0x100);

    const std::array<u32, 2> attribs{0xAAAA0001, 0xBBBB0002};
    f.memory.Put(SCRATCH + 0x200, attribs);
    f.recorder.BindVertexAttribState(Fixture::CMD, 2, SCRATCH + 0x200);

    const std::array<float, 4> viewport{1.f, 2.f, 3.f, 4.f};
    f.memory.Put(SCRATCH + 0x300, viewport);
    f.recorder.SetViewports(Fixture::CMD, 0, 1, SCRATCH + 0x300);

    const std::array<u64, 2> handles_in{0x00010002, 0};
    f.memory.Put(SCRATCH + 0x400, handles_in);
    f.recorder.BindHandles(Fixture::CMD, Op::BindTextures, 1, 0, 2, SCRATCH + 0x400, false);

    const std::array<u8, 4> push{9, 8, 7, 6};
    f.memory.Put(SCRATCH + 0x500, push);
    f.recorder.PushConstants(Fixture::CMD, false, 0, 4, 4, SCRATCH + 0x500);
    const u64 handle = f.recorder.EndRecording(Fixture::CMD);

    // The game reuses all of its argument memory.
    const std::array<u8, 256> garbage{};
    f.memory.Put(SCRATCH, garbage);
    f.memory.Put(SCRATCH + 0x100, garbage);
    f.memory.Put(SCRATCH + 0x200, garbage);
    f.memory.Put(SCRATCH + 0x300, garbage);
    f.memory.Put(SCRATCH + 0x400, garbage);
    f.memory.Put(SCRATCH + 0x500, garbage);

    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const std::vector<u64>& list = f.sink.submissions.at(0).lists.at(0);

    const auto* update = RecordPayload<UpdateUniformBufferPayload>(
        FindRecord(list, Op::UpdateUniformBuffer));
    CHECK(update->address == STORAGE + 0x2010);
    CHECK(update->size == 16);
    CHECK(std::memcmp(update + 1, uniform_data.data(), 16) == 0);

    const auto* state = RecordPayload<BindStatePayload>(FindRecord(list, Op::BindState));
    CHECK(state->kind == static_cast<u32>(StateKind::Blend));
    CHECK(state->size == 8);
    CHECK(std::memcmp(state + 1, blend.data(), 8) == 0);

    const auto* attrib = RecordPayload<BindVertexAttribsPayload>(
        FindRecord(list, Op::BindVertexAttribs));
    CHECK(attrib->count == 2);
    CHECK(TrailingArray<u32>(attrib)[0] == 0xAAAA0001);
    CHECK(TrailingArray<u32>(attrib)[1] == 0xBBBB0002);

    const auto* viewports = RecordPayload<SetViewportsPayload>(FindRecord(list, Op::SetViewports));
    CHECK(TrailingArray<ViewportRect>(viewports)[0].height == 4.f);

    const auto* binds = RecordPayload<BindHandlesPayload>(FindRecord(list, Op::BindTextures));
    CHECK(binds->count == 2);
    CHECK(TrailingArray<u64>(binds)[0] == 0x00010002);
    CHECK(TrailingArray<u64>(binds)[1] == 0);

    const auto* constants = RecordPayload<PushConstantsPayload>(FindRecord(list, Op::PushConstants));
    CHECK(constants->offset == 4);
    CHECK(std::memcmp(constants + 1, push.data(), 4) == 0);
}

TEST_CASE("nvn recorder: queue order and nested call expansion", "[nvn]") {
    Fixture f;
    f.recorder.SetCommandListBatching(GENERATE(false, true));
    REQUIRE(f.recorder.QueueInitialize(Obj(3)));
    const u64 queue_a = Fixture::QUEUE;
    const u64 queue_b = Obj(3);

    const auto draw = [&](u32 count) { f.recorder.DrawArrays(Fixture::CMD, 4, 0, count, 1, 0); };
    const u64 h1 = f.Record(0x0000, [&] { draw(1); });
    const u64 h3 = f.Record(0x1000, [&] { draw(3); });
    const u64 h4 = f.Record(0x2000, [&] { draw(4); });
    const u64 h2 = f.Record(0x3000, [&] {
        draw(20);
        const std::array<u64, 1> callee{h1};
        f.recorder.CallCommands(Fixture::CMD, callee);
        draw(21);
    });
    CHECK(h1 == STORAGE + 0x0000);
    CHECK(h2 == STORAGE + 0x3000);

    CHECK(f.recorder.QueueSubmitCommands(queue_a, std::array{h1, h2}) == 1);
    CHECK(f.recorder.QueueSubmitCommands(queue_b, std::array{h3}) == 2);
    CHECK(f.recorder.QueueSubmitCommands(queue_a, std::array{h4}) == 3);

    REQUIRE(f.sink.submissions.size() == 3);
    for (u64 i = 0; i < 3; ++i) {
        CHECK(f.sink.submissions[i].seq == i + 1);
    }
    CHECK(f.sink.submissions[0].queue == f.sink.submissions[2].queue);
    CHECK(f.sink.submissions[0].queue != f.sink.submissions[1].queue);
    REQUIRE(f.sink.submissions[0].lists.size() == 2);

    // h2 = draw(20), call h1 (expanded inline: draw(1)), draw(21); no CallList survives.
    const std::vector<u64>& expanded = f.sink.submissions[0].lists[1];
    CHECK(Ops(expanded) == std::vector<Op>{Op::Draw, Op::Draw, Op::Draw});
    CHECK(RecordPayload<DrawPayload>(FindRecord(expanded, Op::Draw, 0))->count == 20);
    CHECK(RecordPayload<DrawPayload>(FindRecord(expanded, Op::Draw, 1))->count == 1);
    CHECK(RecordPayload<DrawPayload>(FindRecord(expanded, Op::Draw, 2))->count == 21);
    CHECK(FindRecord(expanded, Op::CallList) == nullptr);

    // A call chain deeper than the cap is cut and counted.
    u64 previous = h1;
    for (u32 i = 0; i < 20; ++i) {
        const u64 callee = previous;
        previous = f.Record(0x4000 + 0x1000ULL * i, [&] {
            const std::array<u64, 1> handles{callee};
            f.recorder.CallCommands(Fixture::CMD, handles);
        });
    }
    CHECK(f.recorder.QueueSubmitCommands(queue_a, std::array{previous}) == 4);
    CHECK(f.recorder.Stats().call_depth_exceeded > 0);

    // A handle that was never recorded is skipped and counted.
    CHECK(f.recorder.QueueSubmitCommands(queue_a, std::array<u64, 1>{0x12345678}) ==
          f.recorder.QueueLastSeq(queue_a));
    CHECK(f.recorder.Stats().unknown_call_handle > 0);
}

TEST_CASE("nvn recorder: re-recording the same memory keeps old submissions intact", "[nvn]") {
    Fixture f;
    const u64 first = f.Record(0, [&] { f.recorder.DrawArrays(Fixture::CMD, 4, 0, 1, 1, 0); });
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{first}) == 1);

    const u64 second = f.Record(0, [&] {
        f.recorder.DrawArrays(Fixture::CMD, 4, 0, 2, 1, 0);
        f.recorder.DrawArrays(Fixture::CMD, 4, 0, 3, 1, 0);
    });
    CHECK(second == first);
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{second}) == 2);

    CHECK(Ops(f.sink.submissions[0].lists[0]) == std::vector<Op>{Op::Draw});
    CHECK(Ops(f.sink.submissions[1].lists[0]) == std::vector<Op>{Op::Draw, Op::Draw});

    // A recording that starts inside an older list's bytes kills the older list.
    const u64 inside = f.Record(0x10, [&] { f.recorder.DrawArrays(Fixture::CMD, 4, 0, 5, 1, 0); });
    CHECK(inside == STORAGE + 0x10);
    const u64 submitted = f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{first});
    CHECK(submitted == f.recorder.QueueLastSeq(Fixture::QUEUE)); // `first` is gone: nothing new
    CHECK(f.sink.submissions.size() == 2);
}

TEST_CASE("nvn recorder: finalized objects live on while records reference them", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    const u64 serial = f.sink.textures.at(0).serial;

    f.memory.Put<u64>(SCRATCH, Obj(10));
    const u64 handle = f.Record(0, [&] {
        f.recorder.SetRenderTargets(Fixture::CMD, 1, SCRATCH, 0, 0, 0);
    });
    // Finalized after recording, before submit: the record still resolves to the texture.
    f.recorder.TextureFinalize(Obj(10));
    CHECK_FALSE(f.recorder.GetTexture(Obj(10)).has_value()); // the tag is gone
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const Submission& submission = f.sink.submissions.at(0);
    const auto* targets = RecordPayload<SetRenderTargetsPayload>(
        FindRecord(submission.lists[0], Op::SetRenderTargets));
    CHECK(TrailingArray<RenderTargetRef>(targets)[0].texture.serial == serial);
    CHECK(std::find(submission.retained_serials.begin(), submission.retained_serials.end(),
                    serial) != submission.retained_serials.end());

    // Not destroyed while the list that references it exists, even after the GPU finished.
    f.sink.Complete(1);
    f.recorder.FlushDestroys();
    CHECK(f.sink.destroyed.empty());

    // Overwriting the list (re-recording the same memory) releases the reference.
    f.Record(0, [] {});
    f.recorder.FlushDestroys();
    REQUIRE(f.sink.destroyed.size() == 1);
    CHECK(f.sink.destroyed[0] == serial);
}

// The SDK's AddCommandMemory (OoT sdk+0x64c0d0) does not check the block against the pool size.
// Ocarina of Time's 1 MiB command ring hands out 0x200-byte blocks past the end near its wrap
// (offset 0xffe30): the commands that fit are recorded in the pool, nothing is refused.
TEST_CASE("nvn recorder: command memory past the pool end is taken as given", "[nvn]") {
    Fixture f;
    const u64 before = f.recorder.Stats().bad_argument;
    const u64 segments = f.recorder.Stats().synthetic_segments;
    const u64 offset = STORAGE_SIZE - 0x1d0;
    const u64 handle = f.Record(offset, [&] { f.recorder.ClearDepthStencil(Fixture::CMD, 1.0f, 1, 0, 0); },
                                0x200);
    CHECK(handle == STORAGE + offset);
    CHECK(f.recorder.Stats().bad_argument == before);
    CHECK(f.recorder.Stats().synthetic_segments == segments);
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    CHECK(Ops(f.sink.submissions.at(0).lists.at(0)).size() == 1);
}

TEST_CASE("nvn recorder: command memory callbacks are never invoked", "[nvn]") {
    Fixture f;
    f.recorder.SetMemoryCallback(Fixture::CMD, 0xDEADBEEF);
    f.recorder.SetMemoryCallbackData(Fixture::CMD, 0x1234);
    // A 64-byte segment holds two records; recording continues in a synthetic segment.
    f.recorder.AddCommandMemory(Fixture::CMD, Fixture::POOL, 0x1000, 64);
    CHECK(f.recorder.GetCommandMemoryFree(Fixture::CMD) == 64);
    f.recorder.BeginRecording(Fixture::CMD);
    u64 last_used = f.recorder.GetCommandMemoryUsed(Fixture::CMD);
    for (u32 i = 0; i < 20; ++i) {
        f.recorder.ClearDepthStencil(Fixture::CMD, 1.0f, 1, 0, 0);
        const u64 used = f.recorder.GetCommandMemoryUsed(Fixture::CMD);
        CHECK(used > last_used);
        last_used = used;
    }
    const u64 handle = f.recorder.EndRecording(Fixture::CMD);
    CHECK(handle == STORAGE + 0x1000);
    const RecorderStats stats = f.recorder.Stats();
    CHECK(stats.callbacks_skipped >= 1);
    CHECK(stats.synthetic_segments >= 1);
    CHECK(f.recorder.GetCommandMemoryFree(Fixture::CMD) > 0); // what is left of the synthetic one

    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const std::vector<u64>& list = f.sink.submissions.at(0).lists.at(0);
    CHECK(Ops(list).size() == 20);

    // No command memory at all: also synthetic.
    f.recorder.CommandBufferInitialize(Obj(5));
    f.recorder.BeginRecording(Obj(5));
    f.recorder.ClearDepthStencil(Obj(5), 1.0f, 1, 0, 0);
    const u64 synthetic_handle = f.recorder.EndRecording(Obj(5));
    CHECK(synthetic_handle >= (1ULL << 62));
    CHECK(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{synthetic_handle}) == 2);
}

namespace {

/// Records a list on a command buffer without command memory: it lands at a synthetic address.
u64 RecordSynthetic(Fixture& f, u64 cmd_va, const std::function<void()>& commands) {
    f.recorder.BeginRecording(cmd_va);
    commands();
    return f.recorder.EndRecording(cmd_va);
}

bool Destroyed(const Fixture& f, u64 serial) {
    return std::find(f.sink.destroyed.begin(), f.sink.destroyed.end(), serial) !=
           f.sink.destroyed.end();
}

} // namespace

TEST_CASE("nvn recorder: synthetic lists are kept without a cap and dropped beyond it", "[nvn]") {
    Fixture f;
    constexpr u64 SYNTH_CMD = Obj(5);
    REQUIRE(f.recorder.CommandBufferInitialize(SYNTH_CMD));
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0x8000));
    const u64 serial = f.sink.textures.at(0).serial;
    f.memory.Put<u64>(SCRATCH, Obj(10));
    const u64 handle = RecordSynthetic(f, SYNTH_CMD, [&] {
        f.recorder.SetRenderTargets(SYNTH_CMD, 1, SCRATCH, 0, 0, 0);
    });
    REQUIRE(handle >= (1ULL << 62));
    RecorderStats stats = f.recorder.Stats();
    CHECK(stats.synthetic_lists == 1);
    CHECK(stats.synthetic_list_bytes > 0);

    // The game finalizes the texture: only the synthetic list keeps it.
    f.recorder.TextureFinalize(Obj(10));
    f.sink.Complete(f.recorder.LastSeq());
    f.recorder.FlushDestroys();
    CHECK_FALSE(Destroyed(f, serial));
    TextureHolderCounts counts = f.recorder.PublishTextureHolders();
    CHECK(counts.synthetic_lists == 1);
    CHECK(counts.guest_lists == 0);
    REQUIRE(f.sink.holders.serials.size() == 1);
    CHECK(f.sink.holders.serials[0] == std::pair{serial, TextureHolder::SyntheticList});

    // A cap that lists used in the last presents are exempt from drops nothing yet.
    f.recorder.SetSyntheticListCap(1, DEFAULT_SYNTHETIC_LIST_IDLE_PRESENTS);
    CHECK(f.recorder.Stats().synthetic_lists == 1);
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) != 0);

    // Idle long enough (no idle requirement here): dropped, its reference released.
    f.recorder.SetSyntheticListCap(1, 0);
    stats = f.recorder.Stats();
    CHECK(stats.synthetic_lists == 0);
    CHECK(stats.synthetic_list_bytes == 0);
    CHECK(stats.synthetic_cap_evictions == 1);
    f.sink.Complete(f.recorder.LastSeq());
    f.recorder.FlushDestroys();
    CHECK(Destroyed(f, serial));
    counts = f.recorder.PublishTextureHolders();
    CHECK(counts.synthetic_lists == 0);
    CHECK(f.sink.holders.serials.empty());
}

TEST_CASE("nvn recorder: the synthetic list cap keeps lists another list calls", "[nvn]") {
    Fixture f;
    constexpr u64 SYNTH_CMD = Obj(5);
    REQUIRE(f.recorder.CommandBufferInitialize(SYNTH_CMD));
    const u64 child = RecordSynthetic(f, SYNTH_CMD, [&] {
        f.recorder.ClearDepthStencil(SYNTH_CMD, 1.0f, 1, 0, 0);
    });
    REQUIRE(child >= (1ULL << 62));
    const u64 parent = f.Record(0, [&] { f.recorder.CallCommands(Fixture::CMD, std::array{child}); });

    f.recorder.SetSyntheticListCap(1, 0);
    CHECK(f.recorder.Stats().synthetic_lists == 1);
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{parent}) != 0);
    CHECK(Ops(f.sink.submissions.back().lists.at(0)) == std::vector<Op>{Op::ClearDepthStencil});

    // The game records over its list: nothing calls the child any more, the next insert drops it.
    f.Record(0, [&] { f.recorder.ClearDepthStencil(Fixture::CMD, 1.0f, 1, 0, 0); });
    CHECK(f.recorder.Stats().synthetic_lists == 0);
    CHECK(f.recorder.Stats().synthetic_cap_evictions == 1);
}

TEST_CASE("nvn recorder: the holder report names re-initialised and list-held textures", "[nvn]") {
    Fixture f;
    // Initialized twice without a finalize: the first object is never destroyed.
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0x8000));
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0x8000));
    const u64 orphan = f.sink.textures.at(0).serial;
    // Finalized while a list in the game's command memory references it.
    REQUIRE(f.recorder.TextureInitialize(Obj(11), MakeTextureDesc(), Fixture::POOL, 0x10000));
    const u64 listed = f.sink.textures.at(2).serial;
    f.memory.Put<u64>(SCRATCH, Obj(11));
    f.Record(0, [&] { f.recorder.SetRenderTargets(Fixture::CMD, 1, SCRATCH, 0, 0, 0); });
    f.recorder.TextureFinalize(Obj(11));

    const TextureHolderCounts counts = f.recorder.PublishTextureHolders();
    CHECK(counts.reinitialized == 1);
    CHECK(counts.guest_lists == 1);
    CHECK(counts.synthetic_lists == 0);
    std::vector<std::pair<u64, TextureHolder>> expected{{orphan, TextureHolder::Reinitialized},
                                                        {listed, TextureHolder::GuestList}};
    std::sort(expected.begin(), expected.end());
    CHECK(f.sink.holders.serials == expected);

    // Finalized through a copy of the old object: an ordinary retirement, no longer re-initialised.
    f.memory.Put(Obj(12), GuestTag{GUEST_TAG_MAGIC, static_cast<u32>(ObjectKind::Texture), orphan});
    f.recorder.TextureFinalize(Obj(12));
    f.sink.Complete(f.recorder.LastSeq());
    f.recorder.FlushDestroys();
    CHECK(Destroyed(f, orphan));
    CHECK(f.recorder.PublishTextureHolders().reinitialized == 0);
}

TEST_CASE("nvn recorder: deferred handles are read at submit", "[nvn]") {
    Fixture f;
    f.recorder.SetCommandListBatching(GENERATE(false, true));
    REQUIRE(f.recorder.TexturePoolInitialize(Obj(20), Fixture::POOL, 0x20000, 64));
    REQUIRE(f.recorder.TexturePoolInitialize(Obj(21), Fixture::POOL, 0x30000, 64));
    const std::array<u64, 2> before{0x00100002, 7};
    f.memory.Put(SCRATCH, before);

    const u64 handle = f.Record(0, [&] {
        f.recorder.SetTexturePool(Fixture::CMD, Obj(21)); // base 256
        f.recorder.BindHandles(Fixture::CMD, Op::BindTextures, 0, 0, 2, SCRATCH, true);
        f.recorder.BindHandles(Fixture::CMD, Op::BindImages, 0, 0, 2, SCRATCH, false);
    });

    const std::array<u64, 2> after{0x00300004, 9};
    f.memory.Put(SCRATCH, after);
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const std::vector<u64>& list = f.sink.submissions.at(0).lists.at(0);

    // Deferred: the values at submit time, ids moved to the pool's base (texture 4 + 256,
    // sampler 3 + 0, as no sampler pool was bound).
    const RecordHeader* deferred = FindRecord(list, Op::BindTextures);
    REQUIRE(deferred != nullptr);
    CHECK((deferred->flags & RECORD_FLAG_DEFERRED) == 0);
    const auto* binds = RecordPayload<BindHandlesPayload>(deferred);
    CHECK(TrailingArray<u64>(binds)[0] == (4 + 256) + (u64{3} << 20));
    CHECK(TrailingArray<u64>(binds)[1] == 9 + 256);
    // Immediate: the values at record time.
    const auto* images = RecordPayload<BindHandlesPayload>(FindRecord(list, Op::BindImages));
    CHECK(TrailingArray<u64>(images)[0] == 2 + 256 + (u64{0x1} << 20));
    CHECK(TrailingArray<u64>(images)[1] == 7 + 256);
}

TEST_CASE("nvn recorder: completion applies event and report writes in order", "[nvn]") {
    Fixture f;
    f.recorder.SetCommandListBatching(GENERATE(false, true));
    constexpr u64 event_offset = 0x20000;
    constexpr u64 report_address = STORAGE + 0x21000;
    REQUIRE(f.recorder.EventInitialize(Obj(10), Fixture::POOL, event_offset));
    f.memory.Put<u32>(STORAGE + event_offset, 1);
    f.memory.Put<u32>(report_address, 0);
    constexpr u64 counter_address = STORAGE + 0x22000;
    f.memory.Put<u64>(counter_address, 0xDEAD);

    const u64 handle = f.Record(0, [&] {
        f.recorder.ReportValue(Fixture::CMD, 0xCAFE, report_address);
        f.recorder.SignalEvent(Fixture::CMD, Obj(10), 0, 0, 0, 7);  // write 7
        f.recorder.ReportCounter(Fixture::CMD, 1, counter_address);
        f.recorder.SignalEvent(Fixture::CMD, Obj(10), 1, 0, 0, 5);  // add 5 -> 12 only in order
    });
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const std::vector<u64>& list = f.sink.submissions.at(0).lists.at(0);
    CHECK(Ops(list) == std::vector<Op>{Op::ReportValue, Op::SignalEvent, Op::ReportCounter,
                                       Op::SignalEvent});

    // Nothing is visible to the guest before the GPU completed the submission.
    CHECK(f.memory.Get<u32>(STORAGE + event_offset) == 1);
    CHECK(f.memory.Get<u32>(report_address) == 0);
    CHECK(f.recorder.EventGetValue(Obj(10)).value() == 1U);
    CHECK(f.recorder.CompletedSeq() == 0);
    CHECK_FALSE(f.recorder.WaitForSeq(1, std::chrono::nanoseconds{0}));

    f.sink.Complete(1);
    // Report values and counters are GPU data: the sink writes them on the GPU timeline, the
    // recorder applies nothing (a second write could put an older value back).
    CHECK(f.memory.Get<u32>(report_address) == 0);
    CHECK(f.memory.Get<u64>(counter_address) == 0xDEAD);
    CHECK(f.recorder.EventGetValue(Obj(10)).value() == 12U);
    CHECK(f.recorder.CompletedSeq() == 1);
    CHECK(f.recorder.WaitForSeq(1, std::chrono::nanoseconds{0}));
    CHECK(f.recorder.Stats().completion_actions == 2);

    // Immediate CPU-side signals.
    f.recorder.EventSignal(Obj(10), 0, 3);
    CHECK(f.recorder.EventGetValue(Obj(10)).value() == 3U);
    f.recorder.EventSignal(Obj(10), 1, 4);
    CHECK(f.recorder.EventGetValue(Obj(10)).value() == 7U);

    // The sink hears about the event and its CPU writes in call order, with the raw values.
    constexpr u64 event_address = STORAGE + event_offset;
    REQUIRE(f.sink.event_cpu.size() == 3);
    CHECK(f.sink.event_cpu[0].address == event_address);
    CHECK(f.sink.event_cpu[0].change == EventCpuChange::Forget);
    CHECK(f.sink.event_cpu[1].address == event_address);
    CHECK(f.sink.event_cpu[1].change == EventCpuChange::Write);
    CHECK(f.sink.event_cpu[1].value == 3);
    CHECK(f.sink.event_cpu[2].address == event_address);
    CHECK(f.sink.event_cpu[2].change == EventCpuChange::Add);
    CHECK(f.sink.event_cpu[2].value == 4);
}

TEST_CASE("nvn recorder: presents are ordered with submits", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    REQUIRE(f.recorder.TextureInitialize(Obj(11), MakeTextureDesc(), Fixture::POOL, 0x10000));
    REQUIRE(f.recorder.WindowInitialize(Obj(12), std::array<u64, 2>{Obj(10), Obj(11)}, 2));

    const u64 handle = f.Record(0, [&] { f.recorder.DrawArrays(Fixture::CMD, 4, 0, 3, 1, 0); });
    CHECK(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    CHECK(f.recorder.QueuePresentTexture(Fixture::QUEUE, Obj(12), 1) == 2);
    CHECK(f.recorder.QueuePresentTexture(Fixture::QUEUE, Obj(12), 5) == 0); // no such slot
    REQUIRE(f.sink.presents.size() == 1);
    CHECK(f.sink.presents[0].seq == 2);
    CHECK(f.sink.presents[0].window_slot == 1);
    CHECK(f.sink.presents[0].present_interval == 2);
    CHECK(f.sink.presents[0].texture_serial == f.sink.textures[1].serial);
    CHECK(f.recorder.WindowSlotSeq(Obj(12), 1) == 2);
    CHECK(f.recorder.WindowSlotSeq(Obj(12), 0) == 0);
    CHECK(f.recorder.QueueLastSeq(Fixture::QUEUE) == 2);

    // Completion callbacks cover presents too.
    f.sink.Complete(2);
    CHECK(f.recorder.QueueFinish(Fixture::QUEUE, std::chrono::nanoseconds{0}));
}

TEST_CASE("nvn recorder: unknown and stale objects are counted, not fatal", "[nvn]") {
    Fixture f;
    const RecorderStats before = f.recorder.Stats();

    // Nothing tagged there.
    f.recorder.TextureFinalize(Obj(50));
    CHECK_FALSE(f.recorder.GetTexture(Obj(50)).has_value());
    CHECK_FALSE(f.recorder.QueueSubmitCommands(Obj(50), std::array<u64, 1>{1}));
    f.recorder.BeginRecording(Obj(50));
    f.recorder.DrawArrays(Obj(50), 4, 0, 3, 1, 0);
    CHECK(f.recorder.EndRecording(Obj(50)) == 0);
    // Unmapped guest address.
    CHECK_FALSE(f.recorder.GetTexture(0x10).has_value());
    CHECK_FALSE(f.recorder.GetTexture(0).has_value());
    // Commands outside a recording.
    f.recorder.DrawArrays(Fixture::CMD, 4, 0, 3, 1, 0);
    // A texture object whose tag was overwritten by the game.
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    f.memory.Put<u64>(Obj(10), 0x1111);
    CHECK_FALSE(f.recorder.GetTexture(Obj(10)).has_value());
    // A tag with a serial that does not exist (finalized and gone).
    GuestTag stale{GUEST_TAG_MAGIC, static_cast<u32>(ObjectKind::Texture), 0x0700'0000'7FFF'0001};
    f.memory.Put(Obj(11), stale);
    CHECK_FALSE(f.recorder.GetTexture(Obj(11)).has_value());
    // Out of range register / bad pool.
    REQUIRE(f.recorder.TexturePoolInitialize(Obj(20), Fixture::POOL, 0, 8));
    f.recorder.RegisterTexture(Obj(20), 9, Obj(10), 0);
    CHECK_FALSE(f.recorder.TexturePoolInitialize(Obj(21), Obj(50), 0, 8));

    const RecorderStats after = f.recorder.Stats();
    CHECK(after.tag_mismatch > before.tag_mismatch);
    CHECK(after.stale_serial > before.stale_serial);
    CHECK(after.bad_guest_read > before.bad_guest_read);
    CHECK(after.bad_argument > before.bad_argument);
    CHECK(after.records_dropped > before.records_dropped);
    CHECK(f.sink.submissions.empty());
}

namespace {

/// Gives the program at `program_va` one NVN2 fragment stage whose code is `fill` bytes.
bool SetTestProgram(Fixture& f, u64 program_va, u8 fill) {
    std::array<u8, 0x100> control{};
    const u32 magic = NVN2_CONTROL_MAGIC;
    const u32 version = NVN2_CONTROL_VERSION;
    const u32 code_size = 16;
    const u16 stage = 1;
    std::memcpy(control.data(), &magic, 4);
    std::memcpy(control.data() + 4, &version, 4);
    std::memcpy(control.data() + 0x44, &code_size, 4);
    std::memcpy(control.data() + 0x56, &stage, 2);
    f.memory.Put(SCRATCH + 0x2000, control);
    std::array<u8, 16> code;
    code.fill(fill);
    f.memory.Put(STORAGE + 0x8000, code);
    const std::array<ShaderStageInput, 1> stages{
        ShaderStageInput{STORAGE + 0x8000, SCRATCH + 0x2000, -1}};
    return f.recorder.ProgramSetShaders(program_va, stages);
}

} // namespace

TEST_CASE("nvn recorder: ClearTexture keeps its channel mask through submit", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    const std::array<s32, 6> region{0, 0, 0, 64, 64, 1};
    f.memory.Put(SCRATCH, region);
    const std::array<u32, 4> color{1, 2, 3, 4};
    f.memory.Put(SCRATCH + 0x100, color);
    const u64 handle = f.Record(0, [&] {
        f.recorder.ClearTexture(Fixture::CMD, Obj(10), 0, SCRATCH, SCRATCH + 0x100, 0x1, true);
        f.recorder.ClearTexture(Fixture::CMD, Obj(10), 0, SCRATCH, SCRATCH + 0x100, 0xF, false);
        f.recorder.ClearTexture(Fixture::CMD, Obj(10), 0, SCRATCH, SCRATCH + 0x100, 0x6, false);
    });
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const std::vector<u64>& list = f.sink.submissions.at(0).lists.at(0);
    REQUIRE(FindRecord(list, Op::ClearTexture, 2) != nullptr);
    CHECK(FindRecord(list, Op::ClearTexture, 0)->flags == 0x1);
    CHECK(FindRecord(list, Op::ClearTexture, 1)->flags == 0xF);
    CHECK(FindRecord(list, Op::ClearTexture, 2)->flags == 0x6);
    CHECK(RecordPayload<ClearTexturePayload>(FindRecord(list, Op::ClearTexture, 0))->is_integer ==
          1);
}

TEST_CASE("nvn recorder: objects used by an open recording are not destroyed", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    REQUIRE(f.recorder.ProgramInitialize(Obj(11)));
    REQUIRE(SetTestProgram(f, Obj(11), 1));
    const u64 texture_serial = f.sink.textures.at(0).serial;
    const u64 program_serial = f.sink.programs.at(0).serial;
    f.memory.Put<u64>(SCRATCH + 0x1000, Obj(10));

    f.recorder.AddCommandMemory(Fixture::CMD, Fixture::POOL, 0, 0x1000);
    f.recorder.BeginRecording(Fixture::CMD);
    f.recorder.SetRenderTargets(Fixture::CMD, 1, SCRATCH + 0x1000, 0, 0, 0);
    f.recorder.BindProgram(Fixture::CMD, Obj(11), 0x2);
    // The game finalizes the texture and re-sets the program's shaders before EndRecording.
    f.recorder.TextureFinalize(Obj(10));
    REQUIRE(SetTestProgram(f, Obj(11), 2));
    f.recorder.FlushDestroys();
    CHECK(f.sink.destroyed.empty());
    const u64 handle = f.recorder.EndRecording(Fixture::CMD);
    f.recorder.FlushDestroys();
    CHECK(f.sink.destroyed.empty());

    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    const Submission& submission = f.sink.submissions.at(0);
    CHECK(std::find(submission.retained_serials.begin(), submission.retained_serials.end(),
                    texture_serial) != submission.retained_serials.end());
    CHECK(std::find(submission.retained_serials.begin(), submission.retained_serials.end(),
                    program_serial) != submission.retained_serials.end());
    f.sink.Complete(1);
    f.recorder.FlushDestroys();
    CHECK(f.sink.destroyed.empty()); // the recorded list still references both

    f.Record(0, [] {}); // overwriting the list releases them
    f.recorder.FlushDestroys();
    REQUIRE(f.sink.destroyed.size() == 2);
    CHECK(std::find(f.sink.destroyed.begin(), f.sink.destroyed.end(), texture_serial) !=
          f.sink.destroyed.end());
    CHECK(std::find(f.sink.destroyed.begin(), f.sink.destroyed.end(), program_serial) !=
          f.sink.destroyed.end());
}

TEST_CASE("nvn recorder: an abandoned recording releases its references", "[nvn]") {
    Fixture f;
    REQUIRE(f.recorder.TextureInitialize(Obj(10), MakeTextureDesc(), Fixture::POOL, 0));
    const u64 texture_serial = f.sink.textures.at(0).serial;
    f.memory.Put<u64>(SCRATCH + 0x1000, Obj(10));
    f.recorder.AddCommandMemory(Fixture::CMD, Fixture::POOL, 0, 0x1000);
    f.recorder.BeginRecording(Fixture::CMD);
    f.recorder.SetRenderTargets(Fixture::CMD, 1, SCRATCH + 0x1000, 0, 0, 0);
    f.recorder.TextureFinalize(Obj(10));
    f.recorder.FlushDestroys();
    CHECK(f.sink.destroyed.empty());
    f.recorder.BeginRecording(Fixture::CMD); // the first recording never ended
    f.recorder.FlushDestroys();
    REQUIRE(f.sink.destroyed.size() == 1);
    CHECK(f.sink.destroyed[0] == texture_serial);
}

TEST_CASE("nvn recorder: infinite wait timeout and threads", "[nvn]") {
    Fixture f;
    const u64 handle = f.Record(0, [&] { f.recorder.DrawArrays(Fixture::CMD, 4, 0, 3, 1, 0); });
    REQUIRE(f.recorder.QueueSubmitCommands(Fixture::QUEUE, std::array{handle}) == 1);
    std::thread gpu{[&] {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        f.sink.Complete(1);
    }};
    // ~0ULL cast to a signed duration is -1 ns: it must mean "forever", not "already expired".
    CHECK(f.recorder.WaitForSeq(1, std::chrono::nanoseconds{-1}));
    gpu.join();
    CHECK(f.recorder.CompletedSeq() == 1);
}
