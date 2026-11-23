// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <atomic>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "video_core/nvn/record.h"

// Command buffer family, everything except the render-state binds (functions_cmdbuf_bind.cpp):
// lifecycle and memory, transfers and clears, synchronization and reports, draws and dispatches,
// push constants, transform feedback, and the hints that have no host effect. Every body decodes
// its arguments in prototype order (nvn_protos.h; the NVN2 additions from the nxnative bodies) and
// hands them to the Recorder; nothing here allocates, locks or logs on the hot path.
namespace Core::NvnHle::Hle {
namespace {

namespace VC = VideoCommon::Nvn;

/// Logs `text` once per boot (the handler's own flag).
#define NVN_LOG_ONCE(flag, ...)                                                                  \
    do {                                                                                         \
        static Common::BootCounter flag;                                                         \
        if (flag.First()) {                                                                      \
            LOG_INFO(Core, __VA_ARGS__);                                                         \
        }                                                                                        \
    } while (false)

// ---- Lifecycle and memory ---------------------------------------------------------------------

void nvnCommandBufferInitialize(Call& c) {
    const u64 cmd = c.args.NextPointer();
    static_cast<void>(c.args.NextPointer()); // device
    c.ret.SetBool(c.rec.CommandBufferInitialize(cmd));
}

void nvnCommandBufferFinalize(Call& c) {
    c.rec.CommandBufferFinalize(c.args.NextPointer());
}

void nvnCommandBufferSetMemoryCallback(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetMemoryCallback(cmd, c.args.NextPointer());
}

void nvnCommandBufferSetMemoryCallbackData(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetMemoryCallbackData(cmd, c.args.NextPointer());
}

void nvnCommandBufferGetMemoryCallback(Call& c) {
    c.ret.SetU64(c.rec.GetMemoryCallback(c.args.NextPointer()));
}

void nvnCommandBufferGetMemoryCallbackData(Call& c) {
    c.ret.SetU64(c.rec.GetMemoryCallbackData(c.args.NextPointer()));
}

void nvnCommandBufferAddCommandMemory(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const u64 size = c.args.NextU64();
    c.rec.AddCommandMemory(cmd, pool, offset, size);
}

void nvnCommandBufferAddControlMemory(Call& c) {
    const u64 cmd = c.args.NextPointer();
    static_cast<void>(c.args.NextPointer()); // memory: the host records no control commands
    c.rec.AddControlMemory(cmd, c.args.NextU64());
}

void nvnCommandBufferGetCommandMemorySize(Call& c) {
    c.ret.SetU64(c.rec.GetCommandMemorySize(c.args.NextPointer()));
}

void nvnCommandBufferGetCommandMemoryUsed(Call& c) {
    c.ret.SetU64(c.rec.GetCommandMemoryUsed(c.args.NextPointer()));
}

void nvnCommandBufferGetCommandMemoryFree(Call& c) {
    c.ret.SetU64(c.rec.GetCommandMemoryFree(c.args.NextPointer()));
}

void nvnCommandBufferGetControlMemorySize(Call& c) {
    c.ret.SetU64(c.rec.GetControlMemorySize(c.args.NextPointer()));
}

/// SDK (3.2.0 sdk+0x5c4510, 1.0.0 sdk+0x59f2a0): control cursor - control base, i.e. the control
/// entries the recording wrote (16 bytes at BeginRecording, 8 at EndRecording, 0x18/0x20 per
/// pushbuffer chunk, 16 per SignalEvent/dispatch/scratch-memory entry...), a function of the
/// SDK's pushbuffer chunking the host does not have. The host writes no control memory: 0, which
/// is the closest answer that keeps DK's decisions. DK (1.0.0 main 0x30319c and its inlined
/// copies) uses it only to advance its control-memory ring (the next AddControlMemory address,
/// ignored by the host; a ring that never fills takes the same branches as one that does not
/// overflow on hardware) and in `command used != 0 || control used != 0`, where command used is
/// never 0 after EndRecording on the host.
void nvnCommandBufferGetControlMemoryUsed(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU64(0);
}

void nvnCommandBufferGetControlMemoryFree(Call& c) {
    c.ret.SetU64(c.rec.GetControlMemoryFree(c.args.NextPointer()));
}

void nvnCommandBufferBeginRecording(Call& c) {
    c.rec.BeginRecording(c.args.NextPointer());
}

void nvnCommandBufferEndRecording(Call& c) {
    c.ret.SetU64(c.rec.EndRecording(c.args.NextPointer()));
}

void nvnCommandBufferIsRecording(Call& c) {
    c.ret.SetBool(c.rec.IsRecording(c.args.NextPointer()));
}

/// CallCommands and CopyCommands: the replay layer treats the two alike (nxnative).
void nvnCommandBufferCallCommands(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    c.rec.CallCommands(cmd, count, c.args.NextPointer());
}

// ---- Transfers and clears -----------------------------------------------------------------------

/// nvnCommandBufferClearTexture (float values).
void nvnCommandBufferClearTexture(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 values = c.args.NextPointer();
    c.rec.ClearTexture(cmd, texture, view, region, values, c.args.NextU32(), false);
}

/// nvnCommandBufferClearTexturei / ClearTextureui: the values are copied as raw 32-bit words.
void nvnCommandBufferClearTextureInteger(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 values = c.args.NextPointer();
    c.rec.ClearTexture(cmd, texture, view, region, values, c.args.NextU32(), true);
}

void nvnCommandBufferClearBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 address = c.args.NextU64();
    const u64 size = c.args.NextU64();
    c.rec.ClearBuffer(cmd, address, size, c.args.NextU32());
}

void nvnCommandBufferCopyBufferToBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 src = c.args.NextU64();
    const u64 dst = c.args.NextU64();
    const u64 size = c.args.NextU64();
    c.rec.CopyBufferToBuffer(cmd, src, dst, size, c.args.NextU32());
}

void nvnCommandBufferCopyBufferToTexture(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 src = c.args.NextU64();
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    c.rec.CopyBufferToTexture(cmd, src, texture, view, region, c.args.NextU32());
}

void nvnCommandBufferCopyTextureToBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 dst = c.args.NextU64();
    c.rec.CopyTextureToBuffer(cmd, texture, view, region, dst, c.args.NextU32());
}

void nvnCommandBufferCopyTextureToTexture(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 src = c.args.NextPointer();
    const u64 src_view = c.args.NextPointer();
    const u64 src_region = c.args.NextPointer();
    const u64 dst = c.args.NextPointer();
    const u64 dst_view = c.args.NextPointer();
    const u64 dst_region = c.args.NextPointer();
    c.rec.CopyTextureToTexture(cmd, src, src_view, src_region, dst, dst_view, dst_region,
                               c.args.NextU32());
}

/// Multisample resolve of whole textures; the tiled variants resolve the same way on a GPU that
/// has no tiled cache.
void nvnCommandBufferDownsample(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 src = c.args.NextPointer();
    c.rec.Downsample(cmd, src, c.args.NextPointer());
}

constexpr u32 CopyFlagResolve = 0x40000000; // nxnative DM_COPY_RESOLVE

/// With views, the resolve is a texture copy with the resolve flag. The Recorder reads the two
/// whole-texture regions from guest memory, so they are placed on the caller's stack below sp,
/// which is dead while the call runs (nxnative: the regions are the textures' own extents).
void nvnCommandBufferDownsampleTextureView(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 src = c.args.NextPointer();
    const u64 src_view = c.args.NextPointer();
    const u64 dst = c.args.NextPointer();
    const u64 dst_view = c.args.NextPointer();
    if (src_view == 0 && dst_view == 0) {
        c.rec.Downsample(cmd, src, dst);
        return;
    }
    const auto src_desc = c.rec.GetTexture(src);
    const auto dst_desc = c.rec.GetTexture(dst);
    if (!src_desc || !dst_desc) {
        return;
    }
    const VC::CopyRegion regions[2] = {
        {0, 0, 0, static_cast<s32>(src_desc->width), static_cast<s32>(src_desc->height), 1},
        {0, 0, 0, static_cast<s32>(dst_desc->width), static_cast<s32>(dst_desc->height), 1},
    };
    const u64 scratch = (c.ctx.sp - sizeof(regions)) & ~u64{15};
    if (!c.mem.Write(scratch, regions, sizeof(regions))) {
        return;
    }
    c.rec.CopyTextureToTexture(cmd, src, src_view, scratch, dst, dst_view,
                               scratch + sizeof(VC::CopyRegion), CopyFlagResolve);
}

void nvnCommandBufferSetCopyRowStride(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetCopyRowStride(cmd, c.args.NextS64());
}

void nvnCommandBufferSetCopyImageStride(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.SetCopyImageStride(cmd, c.args.NextS64());
}

void nvnCommandBufferGetCopyRowStride(Call& c) {
    c.ret.SetU64(static_cast<u64>(c.rec.GetCopyRowStride(c.args.NextPointer())));
}

void nvnCommandBufferGetCopyImageStride(Call& c) {
    c.ret.SetU64(static_cast<u64>(c.rec.GetCopyImageStride(c.args.NextPointer())));
}

/// (cmd, buffer address, buffer size, offset, size, data); the data is copied at record time.
/// SDK (3.2.0 sdk+0x608ee0, 1.0.0 sdk+0x5e7280): offset and size are 32-bit and the load is
/// size / 4 whole words, so a size below 4 loads nothing and a 1-3 byte remainder is dropped.
void nvnCommandBufferUpdateUniformBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 address = c.args.NextU64();
    static_cast<void>(c.args.NextU64()); // buffer size: the SDK selects the buffer with it
    const u32 offset = static_cast<u32>(c.args.NextU64());
    const u32 size = static_cast<u32>(c.args.NextU64()) & ~3U;
    c.rec.UpdateUniformBuffer(cmd, address, offset, size, c.args.NextPointer());
}

// ---- Synchronization and reports ----------------------------------------------------------------

void nvnCommandBufferFenceSync(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 sync = c.args.NextPointer();
    const u32 condition = c.args.NextU32();
    c.rec.FenceSync(cmd, sync, condition, c.args.NextU32());
}

void nvnCommandBufferWaitSync(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.WaitSync(cmd, c.args.NextPointer());
}

/// (cmd, event, NVNeventSignalMode, NVNeventSignalLocation, flags, value)
void nvnCommandBufferSignalEvent(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 event = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 location = c.args.NextU32();
    const u32 flags = c.args.NextU32();
    c.rec.SignalEvent(cmd, event, mode, location, flags, c.args.NextU32());
}

/// (cmd, event, NVNeventWaitMode, value). SDK (3.2.0 sdk+0x5df7e0, 1.0.0 sdk+0x5bb6d0): a GPU
/// semaphore acquire on the event's memory, so a queue may wait for a signal another queue submits
/// later. The Recorder records it; the replay stalls the queue's stream there until a signal
/// satisfies it.
void nvnCommandBufferWaitEvent(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 event = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    c.rec.WaitEvent(cmd, event, mode, c.args.NextU32());
}

void nvnCommandBufferReportValue(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 value = c.args.NextU32();
    c.rec.ReportValue(cmd, value, c.args.NextU64());
}

void nvnCommandBufferReportCounter(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 type = c.args.NextU32();
    c.rec.ReportCounter(cmd, type, c.args.NextU64());
}

void nvnCommandBufferResetCounter(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.ResetCounter(cmd, c.args.NextU32());
}

void nvnCommandBufferBarrier(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.Barrier(cmd, c.args.NextU32());
}

// ---- Draws and dispatches -----------------------------------------------------------------------

void nvnCommandBufferDrawArrays(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const s32 first = c.args.NextS32();
    const u32 count = c.args.NextU32();
    c.rec.DrawArrays(cmd, mode, first, count, 1, 0);
}

/// (cmd, mode, first, count, base instance, instance count)
void nvnCommandBufferDrawArraysInstanced(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const s32 first = c.args.NextS32();
    const u32 count = c.args.NextU32();
    const u32 base_instance = c.args.NextU32();
    const u32 instances = c.args.NextU32();
    c.rec.DrawArrays(cmd, mode, first, count, instances, base_instance);
}

void nvnCommandBufferDrawArraysIndirect(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    c.rec.DrawArraysIndirect(cmd, mode, c.args.NextU64());
}

void nvnCommandBufferDrawElements(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 index_type = c.args.NextU32();
    const u32 count = c.args.NextU32();
    c.rec.DrawElements(cmd, mode, index_type, count, c.args.NextU64(), 0, 0, 1);
}

void nvnCommandBufferDrawElementsBaseVertex(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 index_type = c.args.NextU32();
    const u32 count = c.args.NextU32();
    const u64 indices = c.args.NextU64();
    c.rec.DrawElements(cmd, mode, index_type, count, indices, c.args.NextS32(), 0, 1);
}

/// (cmd, mode, index type, count, index address, base vertex, base instance, instance count)
void nvnCommandBufferDrawElementsInstanced(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 index_type = c.args.NextU32();
    const u32 count = c.args.NextU32();
    const u64 indices = c.args.NextU64();
    const s32 base_vertex = c.args.NextS32();
    const u32 base_instance = c.args.NextU32();
    const u32 instances = c.args.NextU32();
    c.rec.DrawElements(cmd, mode, index_type, count, indices, base_vertex, base_instance,
                       instances);
}

void nvnCommandBufferDrawElementsIndirect(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 index_type = c.args.NextU32();
    const u64 indices = c.args.NextU64();
    c.rec.DrawElementsIndirect(cmd, mode, index_type, indices, c.args.NextU64());
}

/// (cmd, mode, indirect, count address, max draws, stride)
void nvnCommandBufferMultiDrawArraysIndirectCount(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u64 indirect = c.args.NextU64();
    const u64 count_address = c.args.NextU64();
    const s32 max_draws = c.args.NextS32();
    const s64 stride = c.args.NextS64();
    c.rec.MultiDrawArraysIndirectCount(cmd, mode, indirect, count_address,
                                       max_draws < 0 ? 0 : static_cast<u32>(max_draws), stride);
}

/// (cmd, mode, index type, index address, indirect, count address, max draws, stride)
void nvnCommandBufferMultiDrawElementsIndirectCount(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 index_type = c.args.NextU32();
    const u64 indices = c.args.NextU64();
    const u64 indirect = c.args.NextU64();
    const u64 count_address = c.args.NextU64();
    const s32 max_draws = c.args.NextS32();
    const s64 stride = c.args.NextS64();
    c.rec.MultiDrawElementsIndirectCount(cmd, mode, index_type, indices, indirect, count_address,
                                         max_draws < 0 ? 0 : static_cast<u32>(max_draws),
                                         stride);
}

void nvnCommandBufferDrawTexture(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 handle = c.args.NextU64();
    const u64 dst = c.args.NextPointer();
    c.rec.DrawTexture(cmd, handle, dst, c.args.NextPointer());
}

/// The thunk returns to the game, so lr is the call site (recorded for diagnostics).
void nvnCommandBufferDispatchCompute(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 x = c.args.NextU32();
    const u32 y = c.args.NextU32();
    const u32 z = c.args.NextU32();
    c.rec.DispatchCompute(cmd, x, y, z, c.ctx.lr);
}

void nvnCommandBufferDispatchComputeIndirect(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.DispatchComputeIndirect(cmd, c.args.NextU64(), c.ctx.lr);
}

// ---- Task/mesh draws (NVN2; no prototypes in the tables) -----------------------------------------
// The driver (3.2.0 sdk+0x681470..0x681cb0) hands each entry to an MME macro: the direct entries'
// arguments go in the pushbuffer, the indirect ones have the GPU fetch {count, first} (1D, 8 bytes)
// or {x, y, z} (3D, 12 bytes) at the address. The multi entries (helper sdk+0x680e20) emit nothing
// for a draw count <= 0 and use the argument size as the stride when the stride is <= 0; the
// *Count ones have the GPU fetch one u32 draw count.
constexpr u32 MESH_TASKS_1D_ARGS_BYTES = 8;
constexpr u32 MESH_TASKS_3D_ARGS_BYTES = 12;

/// (cmd, int first, int count)
void nvnCommandBufferDrawMeshTasks(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 first = c.args.NextU32();
    const u32 count = c.args.NextU32();
    c.rec.DrawMeshTasks(cmd, 1, first, count, 1, 1);
}

/// (cmd, int x, int y, int z)
void nvnCommandBufferDrawMeshTasks3D(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 x = c.args.NextU32();
    const u32 y = c.args.NextU32();
    const u32 z = c.args.NextU32();
    c.rec.DrawMeshTasks(cmd, 3, 0, x, y, z);
}

/// (cmd, NVNbufferAddress indirect)
void nvnCommandBufferDrawMeshTasksIndirect(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.DrawMeshTasksIndirect(cmd, 1, c.args.NextU64(), 0, 1, 0);
}

/// (cmd, NVNbufferAddress indirect)
void nvnCommandBufferDrawMeshTasks3DIndirect(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.DrawMeshTasksIndirect(cmd, 3, c.args.NextU64(), 0, 1, 0);
}

template <u32 Dims, bool HasCount>
void MultiDrawMeshTasksImpl(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 indirect = c.args.NextU64();
    const u64 count_address = HasCount ? c.args.NextU64() : 0;
    const s32 draws = c.args.NextS32();
    const s32 stride = c.args.NextS32();
    if (draws <= 0) {
        return;
    }
    constexpr u32 args_bytes = Dims == 1 ? MESH_TASKS_1D_ARGS_BYTES : MESH_TASKS_3D_ARGS_BYTES;
    c.rec.DrawMeshTasksIndirect(cmd, Dims, indirect, count_address, static_cast<u32>(draws),
                                stride > 0 ? static_cast<u32>(stride) : args_bytes);
}

// ---- Push constants -------------------------------------------------------------------------------

/// NVN2: (cmd, block, byte offset, size, data), all 32-bit. SDK graphics 3.2.0 sdk+0x5f6fe0 /
/// 1.0.0 sdk+0x5d34b0, compute sdk+0x5f74a0 / sdk+0x5d3a30: a size above 255 pushes 256 bytes, a
/// size below 4 pushes nothing, only whole words are pushed; the graphics offset is a byte
/// (offset & 0xFF), the compute offset whole words. The Recorder copies the data and keeps the
/// block word (DK passes 0).
template <bool Compute>
void PushConstants(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 block = c.args.NextU32();
    const u32 offset = c.args.NextU32();
    u32 size = c.args.NextU32();
    const u64 data = c.args.NextPointer();
    if (size > 0xFF) {
        size = 0x100;
    }
    if (size < 4) {
        return;
    }
    c.rec.PushConstants(cmd, Compute, block, Compute ? offset & ~3U : offset & 0xFFU, size & ~3U,
                        data);
}

void nvnCommandBufferPushGraphicsConstants(Call& c) {
    PushConstants<false>(c);
}

void nvnCommandBufferPushComputeConstants(Call& c) {
    PushConstants<true>(c);
}

// ---- Transform feedback ---------------------------------------------------------------------------

void nvnCommandBufferBindTransformFeedbackBuffer(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 index = c.args.NextS32();
    const u64 address = c.args.NextU64();
    c.rec.BindTransformFeedbackBuffer(cmd, index, address, c.args.NextU64());
}

void nvnCommandBufferBindTransformFeedbackBuffers(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    const u64 ranges = c.args.NextPointer();
    // NVN has 4 transform feedback buffers; a range outside them is a bad argument: bind nothing.
    constexpr s64 TransformFeedbackSlots = 4;
    if (first < 0 || count < 0 || s64{first} + s64{count} > TransformFeedbackSlots) {
        return;
    }
    for (s32 i = 0; i < count; ++i) {
        const auto range = c.Load<VC::BufferRange>(ranges + static_cast<u64>(i) * sizeof(VC::BufferRange));
        c.rec.BindTransformFeedbackBuffer(cmd, first + i, range.address, range.size);
    }
}

/// what: 0 begin, 1 end, 2 pause, 3 resume (record.h TransformFeedbackStatePayload).
template <u32 What>
void TransformFeedbackState(Call& c) {
    const u64 cmd = c.args.NextPointer();
    c.rec.TransformFeedbackState(cmd, What, c.args.NextU64());
}

void nvnCommandBufferDrawTransformFeedback(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    c.rec.DrawTransformFeedback(cmd, mode, c.args.NextU64());
}

// ---- Hints with no host effect --------------------------------------------------------------------

/// Debug labels, groups, markers and checkpoints; ZCull data, tiled cache and color compression
/// hints, shader scratch memory, discard hints, workload priorities, the stencil cull criteria
/// and the command-memory callback enables: nothing to record (nxnative "g_nops" and its empty
/// bodies; discarded targets stay valid, which is correct). The SDK bodies DK reaches only
/// program hardware: SetShaderScratchMemory (3.2.0 sdk+0x5c5fe0, 1.0.0 sdk+0x5a0de0) a control
/// entry with the scratch GPU address (the host gives shaders their own scratch),
/// SetStencilCullCriteria (sdk+0x60aee0 / sdk+0x5e9270) the ZCull stencil test, SetTiledCache*
/// (sdk+0x608250/0x608320, sdk+0x5e65c0/0x5e6690; the tile-size clamp is 2^15 in 3.2.0, 2^14 in
/// 1.0.0) the binner, Set{Command,Control}MemoryCallbackEnabled (sdk+0x5c41d0/0x5c43c0,
/// sdk+0x59ef60) a flag read only when memory runs out, and the host never invokes the callback.
void NoOp(Call& c) {
    static_cast<void>(c);
}

void nvnCommandBufferSetRecordingMode(Call& c) {
    static_cast<void>(c.args.NextPointer());
    const s32 mode = c.args.NextS32();
    NVN_LOG_ONCE(logged, "[nvn] nvnCommandBufferSetRecordingMode {}: the host records every command buffer alike", mode);
}

void nvnCommandBufferSetConservativeRasterDilate(Call& c) {
    NVN_LOG_ONCE(logged, "[nvn] SetConservativeRasterDilate: ignored by the host");
    static_cast<void>(c);
}

void nvnCommandBufferSetConservativeRasterMode(Call& c) {
    NVN_LOG_ONCE(logged, "[nvn] SetConservativeRasterMode: ignored by the host");
    static_cast<void>(c);
}

void nvnCommandBufferSetSubpixelPrecisionBias(Call& c) {
    NVN_LOG_ONCE(logged, "[nvn] SetSubpixelPrecisionBias: ignored by the host");
    static_cast<void>(c);
}

} // namespace

void RegisterCommandBufferFunctions(Registry& registry) {
    NVN_HLE(registry, nvnCommandBufferInitialize);
    NVN_HLE(registry, nvnCommandBufferFinalize);
    NVN_HLE(registry, nvnCommandBufferSetMemoryCallback);
    NVN_HLE(registry, nvnCommandBufferSetMemoryCallbackData);
    NVN_HLE(registry, nvnCommandBufferGetMemoryCallback);
    NVN_HLE(registry, nvnCommandBufferGetMemoryCallbackData);
    NVN_HLE(registry, nvnCommandBufferAddCommandMemory);
    NVN_HLE(registry, nvnCommandBufferAddControlMemory);
    NVN_HLE(registry, nvnCommandBufferGetCommandMemorySize);
    NVN_HLE(registry, nvnCommandBufferGetCommandMemoryUsed);
    NVN_HLE(registry, nvnCommandBufferGetCommandMemoryFree);
    NVN_HLE(registry, nvnCommandBufferGetControlMemorySize);
    NVN_HLE(registry, nvnCommandBufferGetControlMemoryUsed);
    NVN_HLE(registry, nvnCommandBufferGetControlMemoryFree);
    NVN_HLE(registry, nvnCommandBufferBeginRecording);
    NVN_HLE(registry, nvnCommandBufferEndRecording);
    NVN_HLE(registry, nvnCommandBufferIsRecording);
    NVN_HLE(registry, nvnCommandBufferCallCommands);
    registry.Add<&nvnCommandBufferCallCommands>("nvnCommandBufferCopyCommands");

    NVN_HLE(registry, nvnCommandBufferClearTexture);
    registry.Add<&nvnCommandBufferClearTextureInteger>("nvnCommandBufferClearTexturei");
    registry.Add<&nvnCommandBufferClearTextureInteger>("nvnCommandBufferClearTextureui");
    NVN_HLE(registry, nvnCommandBufferClearBuffer);
    NVN_HLE(registry, nvnCommandBufferCopyBufferToBuffer);
    NVN_HLE(registry, nvnCommandBufferCopyBufferToTexture);
    NVN_HLE(registry, nvnCommandBufferCopyTextureToBuffer);
    NVN_HLE(registry, nvnCommandBufferCopyTextureToTexture);
    NVN_HLE(registry, nvnCommandBufferDownsample);
    registry.Add<&nvnCommandBufferDownsample>("nvnCommandBufferTiledDownsample");
    NVN_HLE(registry, nvnCommandBufferDownsampleTextureView);
    registry.Add<&nvnCommandBufferDownsampleTextureView>(
        "nvnCommandBufferTiledDownsampleTextureView");
    NVN_HLE(registry, nvnCommandBufferSetCopyRowStride);
    NVN_HLE(registry, nvnCommandBufferSetCopyImageStride);
    NVN_HLE(registry, nvnCommandBufferGetCopyRowStride);
    NVN_HLE(registry, nvnCommandBufferGetCopyImageStride);
    NVN_HLE(registry, nvnCommandBufferUpdateUniformBuffer);

    NVN_HLE(registry, nvnCommandBufferFenceSync);
    NVN_HLE(registry, nvnCommandBufferWaitSync);
    NVN_HLE(registry, nvnCommandBufferSignalEvent);
    NVN_HLE(registry, nvnCommandBufferWaitEvent);
    NVN_HLE(registry, nvnCommandBufferReportValue);
    NVN_HLE(registry, nvnCommandBufferReportCounter);
    NVN_HLE(registry, nvnCommandBufferResetCounter);
    NVN_HLE(registry, nvnCommandBufferBarrier);

    NVN_HLE(registry, nvnCommandBufferDrawArrays);
    NVN_HLE(registry, nvnCommandBufferDrawArraysInstanced);
    NVN_HLE(registry, nvnCommandBufferDrawArraysIndirect);
    NVN_HLE(registry, nvnCommandBufferDrawElements);
    NVN_HLE(registry, nvnCommandBufferDrawElementsBaseVertex);
    NVN_HLE(registry, nvnCommandBufferDrawElementsInstanced);
    NVN_HLE(registry, nvnCommandBufferDrawElementsIndirect);
    NVN_HLE(registry, nvnCommandBufferMultiDrawArraysIndirectCount);
    NVN_HLE(registry, nvnCommandBufferMultiDrawElementsIndirectCount);
    NVN_HLE(registry, nvnCommandBufferDrawTexture);
    NVN_HLE(registry, nvnCommandBufferDispatchCompute);
    NVN_HLE(registry, nvnCommandBufferDispatchComputeIndirect);
    NVN_HLE(registry, nvnCommandBufferDrawMeshTasks);
    NVN_HLE(registry, nvnCommandBufferDrawMeshTasks3D);
    NVN_HLE(registry, nvnCommandBufferDrawMeshTasksIndirect);
    NVN_HLE(registry, nvnCommandBufferDrawMeshTasks3DIndirect);
    registry.Add<&MultiDrawMeshTasksImpl<1, false>>("nvnCommandBufferMultiDrawMeshTasksIndirect");
    registry.Add<&MultiDrawMeshTasksImpl<1, true>>(
        "nvnCommandBufferMultiDrawMeshTasksIndirectCount");
    registry.Add<&MultiDrawMeshTasksImpl<3, false>>(
        "nvnCommandBufferMultiDrawMeshTasks3DIndirect");
    registry.Add<&MultiDrawMeshTasksImpl<3, true>>(
        "nvnCommandBufferMultiDrawMeshTasks3DIndirectCount");

    NVN_HLE(registry, nvnCommandBufferPushGraphicsConstants);
    NVN_HLE(registry, nvnCommandBufferPushComputeConstants);

    NVN_HLE(registry, nvnCommandBufferBindTransformFeedbackBuffer);
    NVN_HLE(registry, nvnCommandBufferBindTransformFeedbackBuffers);
    registry.Add<&TransformFeedbackState<0>>("nvnCommandBufferBeginTransformFeedback");
    registry.Add<&TransformFeedbackState<1>>("nvnCommandBufferEndTransformFeedback");
    registry.Add<&TransformFeedbackState<2>>("nvnCommandBufferPauseTransformFeedback");
    registry.Add<&TransformFeedbackState<3>>("nvnCommandBufferResumeTransformFeedback");
    NVN_HLE(registry, nvnCommandBufferDrawTransformFeedback);

    NVN_HLE(registry, nvnCommandBufferSetRecordingMode);
    NVN_HLE(registry, nvnCommandBufferSetConservativeRasterDilate);
    NVN_HLE(registry, nvnCommandBufferSetConservativeRasterMode);
    NVN_HLE(registry, nvnCommandBufferSetSubpixelPrecisionBias);
    static constexpr std::string_view NoOps[] = {
        "nvnCommandBufferSetDebugLabel",
        "nvnCommandBufferSetCommandMemoryCallbackEnabled",
        "nvnCommandBufferSetControlMemoryCallbackEnabled",
        "nvnCommandBufferSetPatchSize",
        "nvnCommandBufferSetTiledCacheAction",
        "nvnCommandBufferSetTiledCacheTileSize",
        "nvnCommandBufferSetShaderScratchMemory",
        "nvnCommandBufferSaveZCullData",
        "nvnCommandBufferRestoreZCullData",
        "nvnCommandBufferSetStencilCullCriteria",
        "nvnCommandBufferDiscardColor",
        "nvnCommandBufferDiscardDepthStencil",
        "nvnCommandBufferSetColorReductionEnable",
        "nvnCommandBufferSetColorReductionThresholds",
        "nvnCommandBufferSetComputeWorkloadPriority",
        "nvnCommandBufferSetGraphicsWorkloadPriority",
        "nvnCommandBufferPushDebugGroup",
        "nvnCommandBufferPushDebugGroupStatic",
        "nvnCommandBufferPushDebugGroupDynamic",
        "nvnCommandBufferPushDebugGroupCompute",
        "nvnCommandBufferPushDebugGroupGraphics",
        "nvnCommandBufferPopDebugGroup",
        "nvnCommandBufferPopDebugGroupId",
        "nvnCommandBufferPopDebugGroupCompute",
        "nvnCommandBufferPopDebugGroupGraphics",
        "nvnCommandBufferInsertDebugMarker",
        "nvnCommandBufferInsertDebugMarkerStatic",
        "nvnCommandBufferInsertDebugMarkerDynamic",
        "nvnCommandBufferInsertDebugMarkerCompute",
        "nvnCommandBufferInsertDebugMarkerGraphics",
        "nvnCommandBufferInsertCheckpointStatic",
        "nvnCommandBufferInsertCheckpointDynamic",
    };
    for (const std::string_view name : NoOps) {
        registry.Add<&NoOp>(name);
    }
}

} // namespace Core::NvnHle::Hle
