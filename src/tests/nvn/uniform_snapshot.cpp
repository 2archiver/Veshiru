// SPDX-FileCopyrightText: Copyright (c) 2026 jhaizhou (d2r_gfx)
// SPDX-FileCopyrightText: Copyright (c) 2026 MacNativeGames contributors
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/nvn/uniform_snapshot.h"

// Uniform snapshot at submit (video_core/nvn/uniform_snapshot.h): the submission carries the
// uniform bytes as they were in the submit call, whatever the game writes afterwards.
namespace {

using namespace VideoCommon::Nvn;
using NvnHleTest::Arg;
using NvnHleTest::Rig;

Arg I(u64 value) {
    return Rig::I(value);
}

constexpr u64 STAGE_VERTEX = 0;
constexpr u64 CPU_NO_ACCESS = 0x1;
constexpr u64 UBO = Rig::STORAGE + 0x20000;
constexpr u64 FAR_UBO = Rig::STORAGE + 0x30000;
constexpr u64 GPU_ONLY = Rig::STORAGE + 0x80000;

struct SnapshotRig {
    static constexpr u64 POOL = Rig::Obj(0);
    static constexpr u64 QUEUE = Rig::Obj(1);
    static constexpr u64 CMD = Rig::Obj(2);
    static constexpr u64 GPU_POOL = Rig::Obj(3);

    /// `set_mode` false leaves the Recorder's default mode.
    explicit SnapshotRig(bool enabled, bool set_mode = true) {
        if (set_mode) {
            rig.recorder.SetUniformSnapshot(enabled ? UniformSnapshotMode::Strict
                                                    : UniformSnapshotMode::Off);
        }
        REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, 0x80000, 0));
        REQUIRE(rig.recorder.MemoryPoolInitialize(GPU_POOL, GPU_ONLY, 0x80000, CPU_NO_ACCESS));
        REQUIRE(rig.recorder.QueueInitialize(QUEUE));
        REQUIRE(rig.recorder.CommandBufferInitialize(CMD));
        rig.recorder.AddCommandMemory(CMD, POOL, 0, 0x10000);
        rig.recorder.BeginRecording(CMD);
    }

    void Bind(u64 slot, u64 address, u64 size) {
        rig.Call("nvnCommandBufferBindUniformBuffer",
                 {I(CMD), I(STAGE_VERTEX), I(slot), I(address), I(size)});
    }

    const Submission& Submit() {
        const u64 handle = rig.recorder.EndRecording(CMD);
        REQUIRE(rig.recorder.QueueSubmitCommands(QUEUE, std::array{handle}) != 0);
        return rig.sink.submissions.back();
    }

    void Fill(u64 address, size_t size, u8 seed) {
        std::vector<u8> bytes(size);
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<u8>(seed + i * 7);
        }
        REQUIRE(rig.memory.Write(address, bytes.data(), bytes.size()));
    }

    Rig rig;
};

} // namespace

TEST_CASE("nvn uniform snapshot: the bytes of the submit call, merged per pool", "[nvn]") {
    SnapshotRig f{true};
    f.Fill(UBO, 0x400, 1);
    f.Fill(FAR_UBO, 0x100, 2);
    f.Bind(0, UBO, 0x100);
    f.Bind(1, UBO + 0x100, 0x80);  // rounded up to 0x100: touches the first, merged
    f.Bind(2, UBO + 0x300, 0x100); // 0x100 apart: merged
    f.Bind(3, FAR_UBO, 0x100);     // far: a range of its own
    f.Bind(4, GPU_ONLY, 0x100);    // CPU_NO_ACCESS pool: never copied
    f.Bind(5, UBO + 0x40, 0x40);   // inside the first range: nothing new

    const Submission& submission = f.Submit();
    // The game rewrites its blocks for the next frame right after submitting.
    f.Fill(UBO, 0x400, 100);
    f.Fill(FAR_UBO, 0x100, 200);

    REQUIRE(submission.uniform_snapshot != nullptr);
    const UniformSnapshot& snapshot = *submission.uniform_snapshot;
    REQUIRE(snapshot.ranges.size() == 2);
    CHECK(snapshot.ranges[0].begin == UBO);
    CHECK(snapshot.ranges[0].end == UBO + 0x400);
    CHECK(snapshot.ranges[1].begin == FAR_UBO);
    CHECK(snapshot.ranges[1].end == FAR_UBO + 0x100);
    REQUIRE(snapshot.bytes.size() == 0x500);
    REQUIRE(snapshot.ranges[0].offset <= snapshot.bytes.size() - 0x400);
    REQUIRE(snapshot.ranges[1].offset <= snapshot.bytes.size() - 0x100);
    for (size_t i = 0; i < 0x400; ++i) {
        CHECK(snapshot.bytes[snapshot.ranges[0].offset + i] == static_cast<u8>(1 + i * 7));
    }
    for (size_t i = 0; i < 0x100; ++i) {
        CHECK(snapshot.bytes[snapshot.ranges[1].offset + i] == static_cast<u8>(2 + i * 7));
    }
}

TEST_CASE("nvn uniform snapshot: off by default, and nothing to copy", "[nvn]") {
    SECTION("switch off") {
        SnapshotRig f{false, false}; // the default, not set
        f.Bind(0, UBO, 0x100);
        CHECK(f.Submit().uniform_snapshot == nullptr);
    }
    SECTION("only bindings outside CPU-accessible pools") {
        SnapshotRig f{true};
        f.Bind(0, GPU_ONLY, 0x100);
        f.Bind(1, 0, 0x100);
        CHECK(f.Submit().uniform_snapshot == nullptr);
    }
}

namespace {

// A guest memory with page write tracking (the core page tracker's contract): writes to armed
// pages make them dirty until TakeWritten.
class TrackingMemory final : public GuestMemory {
public:
    static constexpr u64 BASE = 0x8000'0000;
    TrackingMemory() : bytes(0x100000) {}

    bool Read(u64 va, void* dst, size_t size) override {
        if (va < BASE || va + size > BASE + bytes.size()) {
            return false;
        }
        std::memcpy(dst, bytes.data() + (va - BASE), size);
        return true;
    }
    bool Write(u64 va, const void* src, size_t size) override {
        if (va < BASE || va + size > BASE + bytes.size()) {
            return false;
        }
        std::memcpy(bytes.data() + (va - BASE), src, size);
        for (u64 page = va >> 12; page <= (va + size - 1) >> 12; ++page) {
            if (armed.contains(page)) {
                dirty.insert(page);
            }
        }
        return true;
    }
    u8* Pointer(u64 va) override {
        return va >= BASE && va < BASE + bytes.size() ? bytes.data() + (va - BASE) : nullptr;
    }
    u64 Track(u64, u64) override {
        return 1;
    }
    void Arm(u64, u64 va, u64 size) override {
        for (u64 page = va >> 12; page < (va + size) >> 12; ++page) {
            armed.insert(page);
            dirty.erase(page);
        }
    }
    void TakeWritten(u64, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs) override {
        for (u64 page = va >> 12; page < (va + size) >> 12; ++page) {
            if (dirty.erase(page) != 0) {
                runs.emplace_back(page << 12, (page + 1) << 12);
            }
        }
    }

    void Fill(u64 va, size_t size, u8 value) {
        const std::vector<u8> data(size, value);
        REQUIRE(Write(va, data.data(), size));
    }

private:
    std::vector<u8> bytes;
    std::set<u64> armed;
    std::set<u64> dirty;
};

/// One list binding uniform ranges {address, size} to the vertex stage.
std::vector<std::vector<u64>> BindList(std::initializer_list<BufferRange> ranges) {
    const u32 count = static_cast<u32>(ranges.size());
    const u32 size = static_cast<u32>(sizeof(RecordHeader) + sizeof(BindBuffersPayload) +
                                      sizeof(BufferRange) * count);
    std::vector<u64> words(size / 8 + 1, 0);
    const RecordHeader header{static_cast<u16>(Op::BindUniformBuffers), 0, size};
    std::memcpy(words.data(), &header, sizeof(header));
    const BindBuffersPayload payload{0, 0, count, 0, 0};
    std::memcpy(words.data() + 1, &payload, sizeof(payload));
    std::memcpy(words.data() + 1 + sizeof(payload) / 8, ranges.begin(),
                sizeof(BufferRange) * count);
    words.back() = u64{8} << 32; // End
    return {words};
}

} // namespace

TEST_CASE("nvn uniform snapshot: only pages the CPU wrote lately and the GPU never", "[nvn]") {
    constexpr u64 B = TrackingMemory::BASE;
    TrackingMemory memory;
    UniformSnapshotPools pools{{B, UniformSnapshotPool{B + 0x100000}}};
    UniformSnapshotCpuPages cpu;
    UniformSnapshotShared shared;
    shared.Reset(UniformSnapshotMode::Strict);
    UniformSnapshotScratch scratch;
    UniformSnapshotTotals totals;
    const auto lists = BindList({{B + 0x1000, 0x100}, {B + 0x5000, 0x100}, {B + 0x9000, 0x100}});
    const auto capture = [&] {
        return CaptureUniformSnapshot(lists, pools, cpu, shared, memory, 0, scratch, totals);
    };

    // First sight: the pages get armed, nothing is known to be the CPU's yet.
    CHECK(capture() == nullptr);
    CHECK(totals.not_cpu == 3);

    // The CPU writes the first two blocks; the GPU wrote the second one's page at some point.
    memory.Fill(B + 0x1000, 0x100, 0x11);
    memory.Fill(B + 0x5000, 0x100, 0x22);
    shared.NoteGpuWrite(B + 0x5080, B + 0x5090);
    totals = {};
    auto snapshot = capture();
    REQUIRE(snapshot != nullptr);
    REQUIRE(snapshot->ranges.size() == 1);
    CHECK(snapshot->ranges[0].begin == B + 0x1000);
    CHECK(snapshot->bytes[0] == 0x11);
    CHECK(totals.not_cpu == 1);     // B + 0x9000: never written by the CPU
    CHECK(totals.gpu_written == 1); // B + 0x5000

    SECTION("a write stays recent for this frame and the next") {
        ++cpu.presents;
        CHECK(capture() != nullptr);
        ++cpu.presents;
        CHECK(capture() == nullptr);
        memory.Fill(B + 0x1000, 0x10, 0x33);
        snapshot = capture();
        REQUIRE(snapshot != nullptr);
        CHECK(snapshot->bytes[0] == 0x33);
    }
    SECTION("the history of a finalized pool goes with it") {
        shared.Forget(B, B + 0x100000);
        CHECK(shared.Runs() == 0);
    }
}
