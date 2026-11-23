// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <cstring>
#include <functional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/record.h"

// Command buffer family (functions_cmdbuf.cpp): argument decoding of every non-bind command, the
// lifecycle and memory getters, and what reaches the sink through nvnQueueSubmitCommands.
namespace {

using namespace VideoCommon::Nvn;
using NvnHleTest::Rig;

constexpr u64 POOL = Rig::Obj(0);
constexpr u64 QUEUE = Rig::Obj(1);
constexpr u64 CMD = Rig::Obj(2);
constexpr u64 TEX_A = Rig::Obj(10);
constexpr u64 TEX_B = Rig::Obj(11);
constexpr u64 EVENT = Rig::Obj(12);
constexpr u64 SYNC = Rig::Obj(13);
constexpr u64 HANDLE_ARRAY = Rig::SCRATCH + 0x7000;
constexpr u64 EVENT_OFFSET = 0x20000;

std::vector<Op> Ops(const std::vector<u64>& list) {
    std::vector<Op> ops;
    RecordReader reader{list.data(), list.size()};
    while (const RecordHeader* header = reader.Next()) {
        ops.push_back(RecordOp(header));
    }
    return ops;
}

const RecordHeader* Find(const std::vector<u64>& list, Op op, size_t nth = 0) {
    RecordReader reader{list.data(), list.size()};
    while (const RecordHeader* header = reader.Next()) {
        if (RecordOp(header) == op && nth-- == 0) {
            return header;
        }
    }
    return nullptr;
}

template <typename T>
const T* Payload(const std::vector<u64>& list, Op op, size_t nth = 0) {
    const RecordHeader* header = Find(list, op, nth);
    REQUIRE(header != nullptr);
    return RecordPayload<T>(header);
}

TextureDesc MakeTexture(u32 width, u32 height) {
    TextureDesc desc{};
    desc.target = 1;
    desc.format = 0x25;
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.levels = 1;
    return desc;
}

/// A rig with a pool, a queue and an initialized command buffer with 8 KiB of command memory.
struct Fx : Rig {
    Fx() {
        REQUIRE(recorder.MemoryPoolInitialize(POOL, STORAGE, STORAGE_SIZE, 0));
        REQUIRE(recorder.QueueInitialize(QUEUE));
        REQUIRE((Call("nvnCommandBufferInitialize", {I(CMD), I(0)}) & 1) == 1);
    }

    void Begin(u64 pool_offset = 0, u64 size = 0x2000) {
        Call("nvnCommandBufferAddCommandMemory", {I(CMD), I(POOL), I(pool_offset), I(size)});
        Call("nvnCommandBufferBeginRecording", {I(CMD)});
    }
    u64 End() {
        return Call("nvnCommandBufferEndRecording", {I(CMD)});
    }
    /// Records `commands` into command memory at `pool_offset` and returns the handle.
    u64 Record(u64 pool_offset, const std::function<void()>& commands) {
        Begin(pool_offset);
        commands();
        return End();
    }
    /// Submits one handle through nvnQueueSubmitCommands; returns the first submitted list.
    std::vector<u64> Submit(u64 handle) {
        memory.Put<u64>(HANDLE_ARRAY, handle);
        const size_t before = sink.submissions.size();
        Call("nvnQueueSubmitCommands", {I(QUEUE), I(1), I(HANDLE_ARRAY)});
        REQUIRE(sink.submissions.size() == before + 1);
        REQUIRE(sink.submissions.back().lists.size() == 1);
        return sink.submissions.back().lists[0];
    }
    /// Records `commands` and returns the submitted list.
    std::vector<u64> Run(const std::function<void()>& commands) {
        return Submit(Record(0, commands));
    }
    void InitTextures() {
        REQUIRE(recorder.TextureInitialize(TEX_A, MakeTexture(64, 32), POOL, 0x10000));
        REQUIRE(recorder.TextureInitialize(TEX_B, MakeTexture(16, 8), POOL, 0x40000));
    }
    u64 Serial(u64 object) {
        return memory.Get<GuestTag>(object).serial;
    }
};

} // namespace

TEST_CASE("nvn hle cmdbuf: lifecycle, memory getters and the handle QueueSubmitCommands accepts",
          "[nvn]") {
    Fx f;
    const GuestTag tag = f.memory.Get<GuestTag>(CMD);
    CHECK(tag.magic == GUEST_TAG_MAGIC);
    CHECK(f.Call("nvnCommandBufferIsRecording", {Rig::I(CMD)}) == 0);

    f.Call("nvnCommandBufferAddCommandMemory",
           {Rig::I(CMD), Rig::I(POOL), Rig::I(0x1000), Rig::I(0x2000)});
    CHECK(f.Call("nvnCommandBufferGetCommandMemoryUsed", {Rig::I(CMD)}) == 0);
    CHECK(f.Call("nvnCommandBufferGetCommandMemoryFree", {Rig::I(CMD)}) == 0x2000);
    CHECK(f.Call("nvnCommandBufferGetCommandMemorySize", {Rig::I(CMD)}) == 0x2000);

    // Control memory: the host records no control commands, the whole block stays free.
    f.Call("nvnCommandBufferAddControlMemory", {Rig::I(CMD), Rig::I(0x12340), Rig::I(0x800)});
    CHECK(f.Call("nvnCommandBufferGetControlMemorySize", {Rig::I(CMD)}) == 0x800);
    CHECK(f.Call("nvnCommandBufferGetControlMemoryFree", {Rig::I(CMD)}) == 0x800);
    CHECK(f.Call("nvnCommandBufferGetControlMemoryUsed", {Rig::I(CMD)}) == 0);

    f.Call("nvnCommandBufferBeginRecording", {Rig::I(CMD)});
    CHECK(f.Call("nvnCommandBufferIsRecording", {Rig::I(CMD)}) == 1);
    u64 last_used = f.Call("nvnCommandBufferGetCommandMemoryUsed", {Rig::I(CMD)});
    for (u32 bits : {0x1U, 0x2U, 0x4U}) {
        f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(bits)});
        const u64 used = f.Call("nvnCommandBufferGetCommandMemoryUsed", {Rig::I(CMD)});
        const u64 free = f.Call("nvnCommandBufferGetCommandMemoryFree", {Rig::I(CMD)});
        CHECK(used > last_used);
        CHECK(free < 0x2000);
        CHECK(f.Call("nvnCommandBufferGetCommandMemorySize", {Rig::I(CMD)}) == used + free);
        last_used = used;
    }
    const u64 handle = f.End();
    CHECK(handle == Rig::STORAGE + 0x1000);
    CHECK(f.Call("nvnCommandBufferIsRecording", {Rig::I(CMD)}) == 0);

    // The handle EndRecording returned is what nvnQueueSubmitCommands accepts.
    const std::vector<u64> list = f.Submit(handle);
    CHECK(Ops(list) == std::vector<Op>{Op::Barrier, Op::Barrier, Op::Barrier});
    CHECK(Payload<BarrierPayload>(list, Op::Barrier, 0)->bits == 0x1);
    CHECK(Payload<BarrierPayload>(list, Op::Barrier, 2)->bits == 0x4);
    CHECK(f.recorder.Stats().unknown_call_handle == 0);

    // Finalize: the object is gone for the recorder.
    f.Call("nvnCommandBufferFinalize", {Rig::I(CMD)});
    CHECK(f.Call("nvnCommandBufferIsRecording", {Rig::I(CMD)}) == 0);
    CHECK(f.Call("nvnCommandBufferGetCommandMemoryUsed", {Rig::I(CMD)}) == 0);
}

TEST_CASE("nvn hle cmdbuf: memory callback storage", "[nvn]") {
    Fx f;
    CHECK(f.Call("nvnCommandBufferGetMemoryCallback", {Rig::I(CMD)}) == 0);
    f.Call("nvnCommandBufferSetMemoryCallback", {Rig::I(CMD), Rig::I(0x7100'2000'1234)});
    f.Call("nvnCommandBufferSetMemoryCallbackData", {Rig::I(CMD), Rig::I(0x7100'3000'5678)});
    f.Call("nvnCommandBufferSetCommandMemoryCallbackEnabled", {Rig::I(CMD), Rig::I(1)});
    CHECK(f.recorder.GetMemoryCallback(CMD) == 0x7100'2000'1234);
    CHECK(f.recorder.GetMemoryCallbackData(CMD) == 0x7100'3000'5678);
    CHECK(f.Call("nvnCommandBufferGetMemoryCallback", {Rig::I(CMD)}) == 0x7100'2000'1234);
    CHECK(f.Call("nvnCommandBufferGetMemoryCallbackData", {Rig::I(CMD)}) == 0x7100'3000'5678);

    // The callback is never invoked: a small segment overflows into a synthetic one.
    f.Begin(0x1000, 64);
    for (u32 i = 0; i < 10; ++i) {
        f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(1)});
    }
    const u64 handle = f.End();
    CHECK(handle == Rig::STORAGE + 0x1000);
    CHECK(f.recorder.Stats().callbacks_skipped >= 1);
    CHECK(Ops(f.Submit(handle)).size() == 10);
}

TEST_CASE("nvn hle cmdbuf: draw argument decoding", "[nvn]") {
    Fx f;
    const auto list = f.Run([&] {
        // (cmd, mode, first, count)
        f.Call("nvnCommandBufferDrawArrays", {Rig::I(CMD), Rig::I(5), Rig::I(7), Rig::I(9)});
        // (cmd, mode, first, count, base instance, instance count); a 32-bit negative `first`
        // arrives with garbage in the register's upper half.
        f.Call("nvnCommandBufferDrawArraysInstanced",
               {Rig::I(CMD), Rig::I(4), Rig::I(0xDEADBEEF'FFFFFFFB), Rig::I(30), Rig::I(3),
                Rig::I(11)});
        // (cmd, mode, index type, count, indices)
        f.Call("nvnCommandBufferDrawElements", {Rig::I(CMD), Rig::I(4), Rig::I(2), Rig::I(36),
                                                Rig::I(Rig::STORAGE + 0x100)});
        // (..., base vertex)
        f.Call("nvnCommandBufferDrawElementsBaseVertex",
               {Rig::I(CMD), Rig::I(3), Rig::I(1), Rig::I(12), Rig::I(Rig::STORAGE + 0x200),
                Rig::I(0xFFFFFFFB)});
        // (cmd, mode, index type, count, indices, base vertex, base instance, instance count)
        f.Call("nvnCommandBufferDrawElementsInstanced",
               {Rig::I(CMD), Rig::I(6), Rig::I(2), Rig::I(300), Rig::I(Rig::STORAGE + 0x300),
                Rig::I(0x1'0000'0010), Rig::I(5), Rig::I(17)});
    });
    REQUIRE(Ops(list) == std::vector<Op>{Op::Draw, Op::Draw, Op::DrawIndexed, Op::DrawIndexed,
                                         Op::DrawIndexed});

    const auto* plain = Payload<DrawPayload>(list, Op::Draw, 0);
    CHECK(plain->mode == 5);
    CHECK(plain->first == 7);
    CHECK(plain->count == 9);
    CHECK(plain->instance_count == 1);
    CHECK(plain->base_instance == 0);
    CHECK(plain->indirect == 0);

    const auto* instanced = Payload<DrawPayload>(list, Op::Draw, 1);
    CHECK(instanced->mode == 4);
    CHECK(instanced->first == -5);
    CHECK(instanced->count == 30);
    CHECK(instanced->base_instance == 3);
    CHECK(instanced->instance_count == 11);

    const auto* elements = Payload<DrawIndexedPayload>(list, Op::DrawIndexed, 0);
    CHECK(elements->mode == 4);
    CHECK(elements->index_type == 2);
    CHECK(elements->count == 36);
    CHECK(elements->index_address == Rig::STORAGE + 0x100);
    CHECK(elements->base_vertex == 0);
    CHECK(elements->base_instance == 0);
    CHECK(elements->instance_count == 1);

    const auto* base_vertex = Payload<DrawIndexedPayload>(list, Op::DrawIndexed, 1);
    CHECK(base_vertex->mode == 3);
    CHECK(base_vertex->index_type == 1);
    CHECK(base_vertex->count == 12);
    CHECK(base_vertex->index_address == Rig::STORAGE + 0x200);
    CHECK(base_vertex->base_vertex == -5);
    CHECK(base_vertex->base_instance == 0);
    CHECK(base_vertex->instance_count == 1);

    const auto* both = Payload<DrawIndexedPayload>(list, Op::DrawIndexed, 2);
    CHECK(both->mode == 6);
    CHECK(both->index_type == 2);
    CHECK(both->count == 300);
    CHECK(both->index_address == Rig::STORAGE + 0x300);
    CHECK(both->base_vertex == 16); // only the low 32 bits are the int argument
    CHECK(both->base_instance == 5);
    CHECK(both->instance_count == 17);
}

TEST_CASE("nvn hle cmdbuf: indirect draws, dispatch, draw texture and transform feedback",
          "[nvn]") {
    Fx f;
    const std::array<float, 4> dst{1.f, 2.f, 3.f, 4.f};
    const std::array<float, 4> src{0.f, 0.25f, 0.5f, 1.f};
    f.memory.Put(Rig::SCRATCH, dst);
    f.memory.Put(Rig::SCRATCH + 0x20, src);
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferDrawArraysIndirect",
               {Rig::I(CMD), Rig::I(5), Rig::I(Rig::STORAGE + 0x1000)});
        f.Call("nvnCommandBufferDrawElementsIndirect",
               {Rig::I(CMD), Rig::I(4), Rig::I(1), Rig::I(Rig::STORAGE + 0x2000),
                Rig::I(Rig::STORAGE + 0x3000)});
        // (cmd, mode, indirect, count address, max draws, stride)
        f.Call("nvnCommandBufferMultiDrawArraysIndirectCount",
               {Rig::I(CMD), Rig::I(3), Rig::I(Rig::STORAGE + 0x4000),
                Rig::I(Rig::STORAGE + 0x5000), Rig::I(64), Rig::I(32)});
        // (cmd, mode, index type, indices, indirect, count address, max, stride); a negative
        // max count is clamped to 0.
        f.Call("nvnCommandBufferMultiDrawElementsIndirectCount",
               {Rig::I(CMD), Rig::I(4), Rig::I(2), Rig::I(Rig::STORAGE + 0x6000),
                Rig::I(Rig::STORAGE + 0x7000), Rig::I(Rig::STORAGE + 0x8000),
                Rig::I(0xFFFFFFFF), Rig::I(20)});
        f.Call("nvnCommandBufferDrawTexture", {Rig::I(CMD), Rig::I(0x00030002), Rig::I(Rig::SCRATCH),
                                               Rig::I(Rig::SCRATCH + 0x20)});
        f.Call("nvnCommandBufferDispatchCompute", {Rig::I(CMD), Rig::I(8), Rig::I(4), Rig::I(2)});
        f.Call("nvnCommandBufferDispatchComputeIndirect",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x9000)});
        f.Call("nvnCommandBufferDrawTransformFeedback",
               {Rig::I(CMD), Rig::I(5), Rig::I(Rig::STORAGE + 0xA000)});
    });
    REQUIRE(Ops(list) == std::vector<Op>{Op::Draw, Op::DrawIndexed, Op::Draw, Op::DrawIndexed,
                                         Op::DrawTexture, Op::Dispatch, Op::Dispatch,
                                         Op::DrawTransformFeedback});

    const auto* indirect = Payload<DrawPayload>(list, Op::Draw, 0);
    CHECK(indirect->mode == 5);
    CHECK(indirect->indirect == Rig::STORAGE + 0x1000);
    CHECK(indirect->instance_count == 1);

    const auto* indexed = Payload<DrawIndexedPayload>(list, Op::DrawIndexed, 0);
    CHECK(indexed->mode == 4);
    CHECK(indexed->index_type == 1);
    CHECK(indexed->index_address == Rig::STORAGE + 0x2000);
    CHECK(indexed->indirect == Rig::STORAGE + 0x3000);

    const auto* multi = Payload<DrawPayload>(list, Op::Draw, 1);
    CHECK(multi->mode == 3);
    CHECK(multi->indirect == Rig::STORAGE + 0x4000);
    CHECK(multi->count_address == Rig::STORAGE + 0x5000);
    CHECK(multi->max_draws == 64);
    CHECK(multi->stride == 32);

    const auto* multi_indexed = Payload<DrawIndexedPayload>(list, Op::DrawIndexed, 1);
    CHECK(multi_indexed->index_type == 2);
    CHECK(multi_indexed->index_address == Rig::STORAGE + 0x6000);
    CHECK(multi_indexed->indirect == Rig::STORAGE + 0x7000);
    CHECK(multi_indexed->count_address == Rig::STORAGE + 0x8000);
    CHECK(multi_indexed->max_draws == 0);
    CHECK(multi_indexed->stride == 20);

    const auto* texture = Payload<DrawTexturePayload>(list, Op::DrawTexture);
    CHECK(texture->handle == 0x00030002);
    CHECK(texture->dst[2] == 3.f);
    CHECK(texture->src[1] == 0.25f);

    const auto* dispatch = Payload<DispatchPayload>(list, Op::Dispatch, 0);
    CHECK(dispatch->x == 8);
    CHECK(dispatch->y == 4);
    CHECK(dispatch->z == 2);
    CHECK(dispatch->indirect == 0);
    const auto* dispatch_indirect = Payload<DispatchPayload>(list, Op::Dispatch, 1);
    CHECK(dispatch_indirect->indirect == Rig::STORAGE + 0x9000);

    const auto* feedback = Payload<DrawTransformFeedbackPayload>(list, Op::DrawTransformFeedback);
    CHECK(feedback->mode == 5);
    CHECK(feedback->address == Rig::STORAGE + 0xA000);
}

// The SDK's task/mesh draw entries (3.2.0 sdk+0x681470..0x681cb0, multi helper sdk+0x680e20): one
// Op::DrawMeshTasks each. Int arguments are the low 32 bits of their registers; the multi entries
// record nothing for a draw count <= 0 and the argument size (8 1D, 12 3D) as a stride <= 0.
TEST_CASE("nvn hle cmdbuf: task/mesh draws", "[nvn]") {
    Fx f;
    constexpr u64 high = 0xDEAD'0000'0000'0000ULL;
    const auto list = f.Run([&] {
        // (cmd, first, count)
        f.Call("nvnCommandBufferDrawMeshTasks", {Rig::I(CMD), Rig::I(high | 5), Rig::I(high | 3)});
        // (cmd, x, y, z)
        f.Call("nvnCommandBufferDrawMeshTasks3D",
               {Rig::I(CMD), Rig::I(2), Rig::I(high | 3), Rig::I(4)});
        // (cmd, indirect); OoT leaves junk in the later registers
        f.Call("nvnCommandBufferDrawMeshTasksIndirect",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x1010), Rig::I(0x400000), Rig::I(7)});
        f.Call("nvnCommandBufferDrawMeshTasks3DIndirect",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x2000)});
        // (cmd, indirect, draw count, stride)
        f.Call("nvnCommandBufferMultiDrawMeshTasksIndirect",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x3000), Rig::I(4), Rig::I(0)});
        // (cmd, indirect, count address, max draw count, stride)
        f.Call("nvnCommandBufferMultiDrawMeshTasksIndirectCount",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x4000), Rig::I(Rig::STORAGE + 0x5000),
                Rig::I(16), Rig::I(high | 24)});
        f.Call("nvnCommandBufferMultiDrawMeshTasks3DIndirect",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x6000), Rig::I(2), Rig::I(0xFFFFFFFF)});
        f.Call("nvnCommandBufferMultiDrawMeshTasks3DIndirectCount",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x7000), Rig::I(Rig::STORAGE + 0x8000),
                Rig::I(6), Rig::I(32)});
        // Draw counts <= 0: the SDK emits nothing.
        f.Call("nvnCommandBufferMultiDrawMeshTasksIndirect",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x3000), Rig::I(0), Rig::I(8)});
        f.Call("nvnCommandBufferMultiDrawMeshTasks3DIndirectCount",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x7000), Rig::I(Rig::STORAGE + 0x8000),
                Rig::I(0xFFFFFFFF), Rig::I(32)});
        // Variable rate shading: accepted, nothing recorded (the host shades at the full rate).
        f.Call("nvnCommandBufferBindShadingRateImage",
               {Rig::I(CMD), Rig::I(Rig::Obj(9)), Rig::I(Rig::Obj(10))});
        f.Call("nvnCommandBufferShadingRateImagePalette",
               {Rig::I(CMD), Rig::I(0), Rig::I(4), Rig::I(Rig::SCRATCH)});
    });
    REQUIRE(Ops(list) == std::vector<Op>(8, Op::DrawMeshTasks));

    const auto check = [&](size_t nth, u32 dims, u32 first, u32 x, u32 y, u32 z, u32 max_draws,
                           u64 indirect, u64 count_address, u32 stride) {
        const auto* p = Payload<DrawMeshTasksPayload>(list, Op::DrawMeshTasks, nth);
        REQUIRE(p != nullptr);
        CHECK(p->dims == dims);
        CHECK(p->first == first);
        CHECK(p->x == x);
        CHECK(p->y == y);
        CHECK(p->z == z);
        CHECK(p->max_draws == max_draws);
        CHECK(p->indirect == indirect);
        CHECK(p->count_address == count_address);
        CHECK(p->stride == stride);
        CHECK(p->pad == 0);
    };
    check(0, 1, 5, 3, 1, 1, 1, 0, 0, 0);
    check(1, 3, 0, 2, 3, 4, 1, 0, 0, 0);
    check(2, 1, 0, 0, 0, 0, 1, Rig::STORAGE + 0x1010, 0, 0);
    check(3, 3, 0, 0, 0, 0, 1, Rig::STORAGE + 0x2000, 0, 0);
    check(4, 1, 0, 0, 0, 0, 4, Rig::STORAGE + 0x3000, 0, 8);
    check(5, 1, 0, 0, 0, 0, 16, Rig::STORAGE + 0x4000, Rig::STORAGE + 0x5000, 24);
    check(6, 3, 0, 0, 0, 0, 2, Rig::STORAGE + 0x6000, 0, 12);
    check(7, 3, 0, 0, 0, 0, 6, Rig::STORAGE + 0x7000, Rig::STORAGE + 0x8000, 32);
}

TEST_CASE("nvn hle cmdbuf: transform feedback binds and state", "[nvn]") {
    Fx f;
    const std::array<u64, 4> ranges{Rig::STORAGE + 0x100, 0x40, Rig::STORAGE + 0x200, 0x80};
    f.memory.Put(Rig::SCRATCH, ranges);
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferBindTransformFeedbackBuffer",
               {Rig::I(CMD), Rig::I(3), Rig::I(Rig::STORAGE + 0x900), Rig::I(0x1000)});
        f.Call("nvnCommandBufferBindTransformFeedbackBuffers",
               {Rig::I(CMD), Rig::I(1), Rig::I(2), Rig::I(Rig::SCRATCH)});
        f.Call("nvnCommandBufferBeginTransformFeedback", {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x10)});
        f.Call("nvnCommandBufferPauseTransformFeedback", {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x20)});
        f.Call("nvnCommandBufferResumeTransformFeedback", {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x30)});
        f.Call("nvnCommandBufferEndTransformFeedback", {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x40)});
    });
    REQUIRE(Ops(list) ==
            std::vector<Op>{Op::BindTransformFeedbackBuffer, Op::BindTransformFeedbackBuffer,
                            Op::BindTransformFeedbackBuffer, Op::TransformFeedbackState,
                            Op::TransformFeedbackState, Op::TransformFeedbackState,
                            Op::TransformFeedbackState});
    const auto* single = Payload<BindTransformFeedbackBufferPayload>(
        list, Op::BindTransformFeedbackBuffer, 0);
    CHECK(single->index == 3);
    CHECK(single->address == Rig::STORAGE + 0x900);
    CHECK(single->size == 0x1000);
    const auto* first = Payload<BindTransformFeedbackBufferPayload>(
        list, Op::BindTransformFeedbackBuffer, 1);
    CHECK(first->index == 1);
    CHECK(first->address == Rig::STORAGE + 0x100);
    CHECK(first->size == 0x40);
    const auto* second = Payload<BindTransformFeedbackBufferPayload>(
        list, Op::BindTransformFeedbackBuffer, 2);
    CHECK(second->index == 2);
    CHECK(second->address == Rig::STORAGE + 0x200);
    CHECK(second->size == 0x80);

    // begin 0, pause 2, resume 3, end 1, in call order.
    const u32 expected_what[4] = {0, 2, 3, 1};
    for (size_t i = 0; i < 4; ++i) {
        const auto* state = Payload<TransformFeedbackStatePayload>(list, Op::TransformFeedbackState, i);
        CHECK(state->what == expected_what[i]);
        CHECK(state->address == Rig::STORAGE + 0x10 * (i + 1));
    }
}

TEST_CASE("nvn hle cmdbuf: barrier, counters, reports and events", "[nvn]") {
    Fx f;
    REQUIRE(f.recorder.EventInitialize(EVENT, POOL, EVENT_OFFSET));
    f.memory.Put<u32>(Rig::STORAGE + EVENT_OFFSET, 1);
    const u64 report = Rig::STORAGE + 0x30000;
    f.memory.Put<u32>(report, 0);

    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(0xDEAD0005)});
        f.Call("nvnCommandBufferResetCounter", {Rig::I(CMD), Rig::I(4)});
        f.Call("nvnCommandBufferReportCounter", {Rig::I(CMD), Rig::I(1), Rig::I(Rig::STORAGE + 0x31000)});
        f.Call("nvnCommandBufferReportValue", {Rig::I(CMD), Rig::I(0xCAFE), Rig::I(report)});
        // (cmd, event, mode, location, flags, value): write 7, then add 5.
        f.Call("nvnCommandBufferSignalEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(0), Rig::I(3),
                                               Rig::I(0x20), Rig::I(7)});
        f.Call("nvnCommandBufferSignalEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(1), Rig::I(2),
                                               Rig::I(0), Rig::I(5)});
        // The wait is a record, after the signals before it.
        f.Call("nvnCommandBufferWaitEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(0), Rig::I(12)});
    });
    CHECK(Ops(list) == std::vector<Op>{Op::Barrier, Op::ResetCounter, Op::ReportCounter,
                                       Op::ReportValue, Op::SignalEvent, Op::SignalEvent,
                                       Op::WaitEvent});
    CHECK(Payload<BarrierPayload>(list, Op::Barrier)->bits == 0xDEAD0005); // bits are a u32
    CHECK(Payload<ResetCounterPayload>(list, Op::ResetCounter)->counter_type == 4);
    const auto* counter = Payload<ReportCounterPayload>(list, Op::ReportCounter);
    CHECK(counter->counter_type == 1);
    CHECK(counter->address == Rig::STORAGE + 0x31000);
    const auto* value = Payload<ReportValuePayload>(list, Op::ReportValue);
    CHECK(value->value == 0xCAFE);
    CHECK(value->address == report);
    const auto* first = Payload<SignalEventPayload>(list, Op::SignalEvent, 0);
    CHECK(first->event_serial == f.Serial(EVENT));
    CHECK(first->mode == 0);
    CHECK(first->value == 7);
    const auto* second = Payload<SignalEventPayload>(list, Op::SignalEvent, 1);
    CHECK(second->mode == 1);
    CHECK(second->value == 5);
    CHECK(first->event_address == Rig::STORAGE + EVENT_OFFSET);
    const auto* wait = Payload<WaitEventPayload>(list, Op::WaitEvent);
    CHECK(wait->event_address == Rig::STORAGE + EVENT_OFFSET);
    CHECK(wait->mode == 0);
    CHECK(wait->value == 12);

    // Event writes land when the GPU completes the submission; the report value is the sink's
    // (a GPU write), the recorder applies nothing.
    CHECK(f.memory.Get<u32>(report) == 0);
    f.sink.Complete(1);
    CHECK(f.memory.Get<u32>(report) == 0);
    CHECK(f.recorder.EventGetValue(EVENT).value() == 12U);
}

TEST_CASE("nvn hle cmdbuf: WaitEvent records the wait in order with the event's address", "[nvn]") {
    Fx f;
    REQUIRE(f.recorder.EventInitialize(EVENT, POOL, EVENT_OFFSET));
    f.memory.Put<u32>(Rig::STORAGE + EVENT_OFFSET, 0);
    constexpr u64 address = Rig::STORAGE + EVENT_OFFSET;
    const auto list = f.Run([&] {
        // (cmd, event, mode WRITE, location BOTTOM, flags, value)
        f.Call("nvnCommandBufferSignalEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(0), Rig::I(2),
                                               Rig::I(0), Rig::I(5)});
        // (cmd, event, mode EQUAL, value)
        f.Call("nvnCommandBufferWaitEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(0), Rig::I(5)});
        f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(1)});
        // GEQUAL_WRAP
        f.Call("nvnCommandBufferWaitEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(1), Rig::I(6)});
        f.Call("nvnCommandBufferSignalEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(1), Rig::I(2),
                                               Rig::I(0), Rig::I(1)});
    });
    // The order of the waits and signals within the queue is the recording order.
    CHECK(Ops(list) == std::vector<Op>{Op::SignalEvent, Op::WaitEvent, Op::Barrier, Op::WaitEvent,
                                       Op::SignalEvent});
    const auto* signal = Payload<SignalEventPayload>(list, Op::SignalEvent, 0);
    CHECK(signal->event_serial == f.Serial(EVENT));
    CHECK(signal->event_address == address);
    CHECK(signal->mode == 0);
    CHECK(signal->value == 5);
    const auto* add = Payload<SignalEventPayload>(list, Op::SignalEvent, 1);
    CHECK(add->event_address == address);
    CHECK(add->mode == 1);
    CHECK(add->value == 1);
    const auto* equal = Payload<WaitEventPayload>(list, Op::WaitEvent, 0);
    CHECK(equal->event_address == address);
    CHECK(equal->mode == 0);
    CHECK(equal->value == 5);
    const auto* gequal = Payload<WaitEventPayload>(list, Op::WaitEvent, 1);
    CHECK(gequal->event_address == address);
    CHECK(gequal->mode == 1);
    CHECK(gequal->value == 6);

    // Recording a wait does not touch the guest word, and signals still land at completion.
    CHECK(f.recorder.EventGetValue(EVENT).value() == 0U);
    f.sink.Complete(1);
    CHECK(f.recorder.EventGetValue(EVENT).value() == 6U);
}

TEST_CASE("nvn hle cmdbuf: WaitEvent on an unknown event records nothing", "[nvn]") {
    Fx f;
    REQUIRE(f.recorder.EventInitialize(EVENT, POOL, EVENT_OFFSET));
    constexpr u64 NOT_AN_EVENT = Rig::Obj(14); // zeroed memory: no guest tag
    const RecorderStats before = f.recorder.Stats();
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferWaitEvent",
               {Rig::I(CMD), Rig::I(NOT_AN_EVENT), Rig::I(0), Rig::I(5)});
        f.Call("nvnCommandBufferWaitEvent", {Rig::I(CMD), Rig::I(EVENT), Rig::I(0), Rig::I(9)});
    });
    // Only the wait on the real event is recorded; the other is dropped and counted.
    CHECK(Ops(list) == std::vector<Op>{Op::WaitEvent});
    CHECK(Payload<WaitEventPayload>(list, Op::WaitEvent)->value == 9);
    const RecorderStats after = f.recorder.Stats();
    CHECK(after.records_dropped == before.records_dropped + 1);
    CHECK(after.tag_mismatch == before.tag_mismatch + 1);
}

TEST_CASE("nvn hle cmdbuf: fence and wait sync", "[nvn]") {
    Fx f;
    REQUIRE(f.recorder.SyncInitialize(SYNC));
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferFenceSync", {Rig::I(CMD), Rig::I(SYNC), Rig::I(0), Rig::I(0x2)});
        f.Call("nvnCommandBufferWaitSync", {Rig::I(CMD), Rig::I(SYNC)});
    });
    CHECK(Ops(list) == std::vector<Op>{Op::FenceSync, Op::WaitSync});
    const auto* fence = Payload<FenceSyncPayload>(list, Op::FenceSync);
    CHECK(fence->sync_serial == f.Serial(SYNC));
    CHECK(fence->flags == 0x2);
    CHECK(Payload<WaitSyncPayload>(list, Op::WaitSync)->sync_serial == f.Serial(SYNC));
    CHECK(f.recorder.GetSyncState(SYNC)->fenced);
}

TEST_CASE("nvn hle cmdbuf: waiting on a sync signalled without a fence records nothing", "[nvn]") {
    Fx f;
    // The window's acquire sync: signalled, no fence ever replays for it.
    REQUIRE(f.recorder.SyncSetSignaled(SYNC));
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferWaitSync", {Rig::I(CMD), Rig::I(SYNC)});
    });
    CHECK(Ops(list).empty());
}

TEST_CASE("nvn hle cmdbuf: CallCommands and CopyCommands arrays", "[nvn]") {
    Fx f;
    const u64 h1 = f.Record(0x0000, [&] {
        f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(1)});
    });
    const u64 h2 = f.Record(0x1000, [&] {
        f.Call("nvnCommandBufferResetCounter", {Rig::I(CMD), Rig::I(2)});
    });
    const u64 h3 = f.Record(0x2000, [&] {
        f.Call("nvnCommandBufferReportValue", {Rig::I(CMD), Rig::I(3), Rig::I(Rig::STORAGE + 0x400)});
    });
    f.memory.Put(Rig::SCRATCH, std::array<u64, 3>{h1, h2, h3});
    const u64 outer = f.Record(0x4000, [&] {
        f.Call("nvnCommandBufferCallCommands", {Rig::I(CMD), Rig::I(3), Rig::I(Rig::SCRATCH)});
        f.Call("nvnCommandBufferCopyCommands", {Rig::I(CMD), Rig::I(2), Rig::I(Rig::SCRATCH + 8)});
        f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(8)});
    });
    // The submit expands the calls inline, in array order.
    const std::vector<u64> list = f.Submit(outer);
    CHECK(Ops(list) == std::vector<Op>{Op::Barrier, Op::ResetCounter, Op::ReportValue,
                                       Op::ResetCounter, Op::ReportValue, Op::Barrier});
    CHECK(Find(list, Op::CallList) == nullptr);
    CHECK(Payload<BarrierPayload>(list, Op::Barrier, 0)->bits == 1);
    CHECK(Payload<BarrierPayload>(list, Op::Barrier, 1)->bits == 8);
    CHECK(Payload<ResetCounterPayload>(list, Op::ResetCounter, 1)->counter_type == 2);
}

TEST_CASE("nvn hle cmdbuf: ClearTexture masks, views and integer variants", "[nvn]") {
    Fx f;
    f.InitTextures();
    f.memory.Put(Rig::SCRATCH, std::array<s32, 6>{1, 2, 0, 30, 20, 1});
    f.memory.Put(Rig::SCRATCH + 0x40, std::array<float, 4>{1.f, 0.5f, 0.f, 0.25f});
    f.memory.Put(Rig::SCRATCH + 0x60, std::array<s32, 4>{-1, 2, -3, 4});
    GuestTextureView view{};
    view.set_bits = GUEST_VIEW_SET_LEVELS | GUEST_VIEW_SET_LAYERS;
    view.base_level = 3;
    view.nlevels = 1;
    view.min_layer = 2;
    view.nlayers = 4;
    f.memory.Put(Rig::SCRATCH + 0x80, view);

    const auto call = [&](const char* name, u64 texture, u64 view_va, u64 values, u32 mask) {
        f.Call(name, {Rig::I(CMD), Rig::I(texture), Rig::I(view_va), Rig::I(Rig::SCRATCH),
                      Rig::I(values), Rig::I(mask)});
    };
    const auto list = f.Run([&] {
        call("nvnCommandBufferClearTexture", TEX_A, 0, Rig::SCRATCH + 0x40, 0xF);
        call("nvnCommandBufferClearTexture", TEX_A, 0, Rig::SCRATCH + 0x40, 0x1);
        call("nvnCommandBufferClearTexture", TEX_B, 0, Rig::SCRATCH + 0x40, 0x6);
        call("nvnCommandBufferClearTexture", TEX_B, 0, Rig::SCRATCH + 0x40, 0x0);
        call("nvnCommandBufferClearTexturei", TEX_A, Rig::SCRATCH + 0x80, Rig::SCRATCH + 0x60, 0xA);
        call("nvnCommandBufferClearTextureui", TEX_A, 0, Rig::SCRATCH + 0x60, 0x8);
    });
    REQUIRE(Ops(list) == std::vector<Op>(6, Op::ClearTexture));
    const u32 masks[6] = {0xF, 0x1, 0x6, 0x0, 0xA, 0x8};
    for (size_t i = 0; i < 6; ++i) {
        CHECK(Find(list, Op::ClearTexture, i)->flags == masks[i]);
    }
    const auto* a = Payload<ClearTexturePayload>(list, Op::ClearTexture, 0);
    CHECK(a->texture.serial == f.Serial(TEX_A));
    CHECK(a->region.x == 1);
    CHECK(a->region.y == 2);
    CHECK(a->region.width == 30);
    CHECK(a->region.height == 20);
    CHECK(a->is_integer == 0);
    float green;
    std::memcpy(&green, &a->color[1], 4);
    CHECK(green == 0.5f);
    CHECK(Payload<ClearTexturePayload>(list, Op::ClearTexture, 2)->texture.serial ==
          f.Serial(TEX_B));

    const auto* with_view = Payload<ClearTexturePayload>(list, Op::ClearTexture, 4);
    CHECK(with_view->is_integer == 1);
    CHECK(with_view->texture.level == 3);
    CHECK(with_view->texture.layer == 2);
    CHECK(with_view->layer_count == 4);
    CHECK(static_cast<s32>(with_view->color[0]) == -1);
    CHECK(static_cast<s32>(with_view->color[2]) == -3);
    CHECK(Payload<ClearTexturePayload>(list, Op::ClearTexture, 5)->is_integer == 1);
}

TEST_CASE("nvn hle cmdbuf: ClearBuffer and CopyBufferToBuffer", "[nvn]") {
    Fx f;
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferClearBuffer", {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x100),
                                                Rig::I(0x1'0000'0040), Rig::I(0xFFEEDDCC)});
        f.Call("nvnCommandBufferCopyBufferToBuffer",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x200), Rig::I(Rig::STORAGE + 0x300),
                Rig::I(0x500), Rig::I(0x1)});
    });
    REQUIRE(Ops(list) == std::vector<Op>{Op::ClearBuffer, Op::CopyBufferToBuffer});
    const auto* clear = Payload<ClearBufferPayload>(list, Op::ClearBuffer);
    CHECK(clear->address == Rig::STORAGE + 0x100);
    CHECK(clear->size == 0x1'0000'0040); // size_t
    CHECK(clear->value == 0xFFEEDDCC);
    const auto* copy = Payload<CopyBufferToBufferPayload>(list, Op::CopyBufferToBuffer);
    CHECK(copy->src == Rig::STORAGE + 0x200);
    CHECK(copy->dst == Rig::STORAGE + 0x300);
    CHECK(copy->size == 0x500);
    CHECK(copy->flags == 1);
}

TEST_CASE("nvn hle cmdbuf: copies with regions read from guest memory and copy strides", "[nvn]") {
    Fx f;
    f.InitTextures();
    f.memory.Put(Rig::SCRATCH, std::array<s32, 6>{4, 8, 0, 16, 12, 1});
    f.memory.Put(Rig::SCRATCH + 0x40, std::array<s32, 6>{1, 2, 3, 6, 5, 4});

    CHECK(f.Call("nvnCommandBufferGetCopyRowStride", {Rig::I(CMD)}) == 0);
    f.Call("nvnCommandBufferSetCopyRowStride", {Rig::I(CMD), Rig::I(256)});
    f.Call("nvnCommandBufferSetCopyImageStride", {Rig::I(CMD), Rig::I(0x4000)});
    CHECK(f.Call("nvnCommandBufferGetCopyRowStride", {Rig::I(CMD)}) == 256);
    CHECK(f.Call("nvnCommandBufferGetCopyImageStride", {Rig::I(CMD)}) == 0x4000);

    const auto list = f.Run([&] {
        // (cmd, src address, texture, view, region, flags)
        f.Call("nvnCommandBufferCopyBufferToTexture",
               {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x500), Rig::I(TEX_A), Rig::I(0),
                Rig::I(Rig::SCRATCH), Rig::I(0x2)});
        // (cmd, texture, view, region, dst address, flags)
        f.Call("nvnCommandBufferCopyTextureToBuffer",
               {Rig::I(CMD), Rig::I(TEX_B), Rig::I(0), Rig::I(Rig::SCRATCH + 0x40),
                Rig::I(Rig::STORAGE + 0x600), Rig::I(0x4)});
        // (cmd, src, src view, src region, dst, dst view, dst region, flags)
        f.Call("nvnCommandBufferCopyTextureToTexture",
               {Rig::I(CMD), Rig::I(TEX_A), Rig::I(0), Rig::I(Rig::SCRATCH), Rig::I(TEX_B),
                Rig::I(0), Rig::I(Rig::SCRATCH + 0x40), Rig::I(0x1)});
    });
    REQUIRE(Ops(list) == std::vector<Op>{Op::CopyBufferToTexture, Op::CopyTextureToBuffer,
                                         Op::CopyTextureToTexture});

    const auto* to_texture = Payload<CopyBufferTexturePayload>(list, Op::CopyBufferToTexture);
    CHECK(to_texture->buffer == Rig::STORAGE + 0x500);
    CHECK(to_texture->texture.serial == f.Serial(TEX_A));
    CHECK(to_texture->region.x == 4);
    CHECK(to_texture->region.width == 16);
    CHECK(to_texture->row_stride == 256);
    CHECK(to_texture->image_stride == 0x4000);
    CHECK(to_texture->flags == 2);

    const auto* to_buffer = Payload<CopyBufferTexturePayload>(list, Op::CopyTextureToBuffer);
    CHECK(to_buffer->buffer == Rig::STORAGE + 0x600);
    CHECK(to_buffer->texture.serial == f.Serial(TEX_B));
    CHECK(to_buffer->region.z == 3);
    CHECK(to_buffer->region.depth == 4);
    CHECK(to_buffer->flags == 4);

    const auto* t2t = Payload<CopyTextureToTexturePayload>(list, Op::CopyTextureToTexture);
    CHECK(t2t->src.serial == f.Serial(TEX_A));
    CHECK(t2t->dst.serial == f.Serial(TEX_B));
    CHECK(t2t->src_region.y == 8);
    CHECK(t2t->src_region.height == 12);
    CHECK(t2t->dst_region.x == 1);
    CHECK(t2t->dst_region.height == 5);
    CHECK(t2t->flags == 1);
}

TEST_CASE("nvn hle cmdbuf: Downsample resolves whole textures", "[nvn]") {
    Fx f;
    f.InitTextures();
    GuestTextureView src_view{};
    src_view.set_bits = GUEST_VIEW_SET_LEVELS;
    src_view.base_level = 2;
    src_view.nlevels = 1;
    f.memory.Put(Rig::SCRATCH, src_view);
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferDownsample", {Rig::I(CMD), Rig::I(TEX_A), Rig::I(TEX_B)});
        f.Call("nvnCommandBufferDownsampleTextureView",
               {Rig::I(CMD), Rig::I(TEX_A), Rig::I(Rig::SCRATCH), Rig::I(TEX_B), Rig::I(0)});
        f.Call("nvnCommandBufferDownsampleTextureView",
               {Rig::I(CMD), Rig::I(TEX_A), Rig::I(0), Rig::I(TEX_B), Rig::I(0)});
    });
    REQUIRE(Ops(list) == std::vector<Op>(3, Op::CopyTextureToTexture));
    for (size_t i = 0; i < 3; ++i) {
        const auto* resolve = Payload<CopyTextureToTexturePayload>(list, Op::CopyTextureToTexture, i);
        CHECK(resolve->flags == 0x40000000);
        CHECK(resolve->src.serial == f.Serial(TEX_A));
        CHECK(resolve->dst.serial == f.Serial(TEX_B));
        CHECK(resolve->src_region.width == 64);
        CHECK(resolve->src_region.height == 32);
        CHECK(resolve->src_region.depth == 1);
    }
    // With views, each texture's own extent is the region.
    const auto* with_view = Payload<CopyTextureToTexturePayload>(list, Op::CopyTextureToTexture, 1);
    CHECK(with_view->src.level == 2);
    CHECK(with_view->dst_region.width == 16);
    CHECK(with_view->dst_region.height == 8);
}

TEST_CASE("nvn hle cmdbuf: push constants follow the SDK's size and offset rules", "[nvn]") {
    Fx f;
    std::array<u8, 256> bytes;
    for (u32 i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<u8>(i + 1);
    }
    f.memory.Put(Rig::SCRATCH, bytes);
    const u64 bad_argument = f.recorder.Stats().bad_argument;
    const auto list = f.Run([&] {
        // (cmd, block, byte offset, size, data), 32-bit: the upper register halves are ignored.
        // Graphics: the offset is a byte (0x110 & 0xFF), whole words only (10 -> 8).
        f.Call("nvnCommandBufferPushGraphicsConstants",
               {Rig::I(CMD), Rig::I(0), Rig::I(0x1'0000'0110), Rig::I(10), Rig::I(Rig::SCRATCH)});
        // Compute: whole-word offset (6 -> 4), a size above 255 pushes 256 bytes.
        f.Call("nvnCommandBufferPushComputeConstants",
               {Rig::I(CMD), Rig::I(0x77), Rig::I(6), Rig::I(0x300), Rig::I(Rig::SCRATCH)});
        // A size below 4 pushes nothing (no failure either).
        f.Call("nvnCommandBufferPushGraphicsConstants",
               {Rig::I(CMD), Rig::I(0), Rig::I(0), Rig::I(3), Rig::I(Rig::SCRATCH)});
    });
    REQUIRE(Ops(list) == std::vector<Op>{Op::PushConstants, Op::PushConstants});
    const auto* graphics = Payload<PushConstantsPayload>(list, Op::PushConstants, 0);
    CHECK(graphics->compute == 0);
    CHECK(graphics->offset == 0x10);
    CHECK(graphics->size == 8);
    CHECK(graphics->word == 0);
    CHECK(std::memcmp(graphics + 1, bytes.data(), 8) == 0);
    const auto* compute = Payload<PushConstantsPayload>(list, Op::PushConstants, 1);
    CHECK(compute->compute == 1);
    CHECK(compute->offset == 4);
    CHECK(compute->size == 256);
    CHECK(compute->word == 0x77);
    CHECK(std::memcmp(compute + 1, bytes.data(), 256) == 0);
    CHECK(f.recorder.Stats().bad_argument == bad_argument);
}

TEST_CASE("nvn hle cmdbuf: UpdateUniformBuffer copies its data at record time", "[nvn]") {
    Fx f;
    std::array<u8, 24> data;
    for (u8 i = 0; i < 24; ++i) {
        data[i] = static_cast<u8>(0x40 + i);
    }
    f.memory.Put(Rig::SCRATCH, data);
    f.Begin();
    // (cmd, buffer address, buffer size, offset, size, data): offset and size are 32-bit and only
    // whole words load (26 -> 24 bytes).
    f.Call("nvnCommandBufferUpdateUniformBuffer",
           {Rig::I(CMD), Rig::I(Rig::STORAGE + 0x2000), Rig::I(0x1000), Rig::I(0x1'0000'0010),
            Rig::I(26), Rig::I(Rig::SCRATCH)});
    f.memory.Put(Rig::SCRATCH, std::array<u8, 24>{}); // the game reuses its memory
    const u64 handle = f.End();

    const std::vector<u64> list = f.Submit(handle);
    const auto* update = Payload<UpdateUniformBufferPayload>(list, Op::UpdateUniformBuffer);
    CHECK(update->address == Rig::STORAGE + 0x2010);
    CHECK(update->size == 24);
    CHECK(std::memcmp(update + 1, data.data(), 24) == 0);
}

TEST_CASE("nvn hle cmdbuf: hints record nothing and commands outside a recording are dropped",
          "[nvn]") {
    Fx f;
    const auto list = f.Run([&] {
        f.Call("nvnCommandBufferPushDebugGroupStatic", {Rig::I(CMD), Rig::I(1), Rig::I(Rig::SCRATCH)});
        f.Call("nvnCommandBufferInsertDebugMarker", {Rig::I(CMD), Rig::I(Rig::SCRATCH)});
        f.Call("nvnCommandBufferPopDebugGroup", {Rig::I(CMD)});
        f.Call("nvnCommandBufferSetStencilCullCriteria", {Rig::I(CMD), Rig::I(5), Rig::I(0xFF)});
        f.Call("nvnCommandBufferSetShaderScratchMemory",
               {Rig::I(CMD), Rig::I(POOL), Rig::I(0), Rig::I(0x1000)});
        f.Call("nvnCommandBufferSetRecordingMode", {Rig::I(CMD), Rig::I(2)});
        f.Call("nvnCommandBufferDiscardColor", {Rig::I(CMD), Rig::I(0)});
        f.Call("nvnCommandBufferSetTiledCacheAction", {Rig::I(CMD), Rig::I(1)});
        f.Call("nvnCommandBufferSaveZCullData", {Rig::I(CMD), Rig::I(Rig::STORAGE), Rig::I(64)});
    });
    CHECK(Ops(list).empty());

    // Not recording: nothing is recorded, nothing throws.
    const u64 recorded = f.recorder.Stats().records_recorded;
    f.Call("nvnCommandBufferDrawArrays", {Rig::I(CMD), Rig::I(4), Rig::I(0), Rig::I(3)});
    f.Call("nvnCommandBufferBarrier", {Rig::I(CMD), Rig::I(1)});
    CHECK(f.recorder.Stats().records_recorded == recorded);
    CHECK(f.Call("nvnCommandBufferIsRecording", {Rig::I(CMD)}) == 0);
}
