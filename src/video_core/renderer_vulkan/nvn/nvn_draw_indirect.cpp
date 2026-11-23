// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Indirect and multi-draw-indirect(-count) draws of the NVN replay (NvnIndirect): the arguments are
// read by the GPU, so the host never knows how many primitives a draw has. Everything the draw
// path does for a direct draw (pipeline, dynamic state, descriptors, vertex and index buffers,
// render conditions) is reused through `DrawContext::PrepareDraw`. The indirect mesh-task draws
// (vkCmdDrawMeshTasksIndirect[Count]EXT) are here too.

#include <cstring>
#include <algorithm>
#include <string>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_quad_indirect.h"

namespace Vulkan::Nvn {
namespace {

/// NVNdrawArraysIndirectData: { count, instanceCount, first, baseInstance } (VkDrawIndirectCommand).
constexpr u32 ARRAYS_COMMAND_SIZE = 16;
/// NVNdrawElementsIndirectData: { count, instanceCount, firstIndex, baseVertex, baseInstance }
/// (VkDrawIndexedIndirectCommand).
constexpr u32 ELEMENTS_COMMAND_SIZE = 20;
/// The draw count is a 32-bit word.
constexpr u32 COUNT_SIZE = 4;

/// QUADS, QUAD_STRIP and LINE_LOOP have no Vulkan topology: a direct draw converts them with
/// generated indices, which need the count that an indirect draw's GPU-written arguments hold.
/// Non-indexed QUADS are drawn all the same: the GPU rewrites their arguments (QuadIndirectGpu).
bool NeedsGeneratedIndices(u32 mode) {
    return mode == 7 || mode == 8 || mode == 2;
}

struct IndirectCall {
    std::string_view op;
    u32 mode;
    u32 command_size;
    u32 max_draws;
    u32 stride;
    u64 indirect;
    u64 count_address;
    /// QUADS whose arguments the GPU rewrites as indexed triangle draws (QuadIndirectGpu).
    bool quads_converted{};
};

/// Resolves the argument and count buffers and checks what the device can do. False = the draw is
/// dropped (counted, logged). Nothing is recorded.
bool ResolveArguments(DrawContext& context, DrawHost& host, const IndirectCall& call,
                      BufferRef& args, BufferRef& count, u64& args_size) {
    Context::Impl& impl = context.GetImpl();
    const auto drop = [&](const std::string& reason) {
        impl.stats.indirect_unsupported.fetch_add(1, std::memory_order_relaxed);
        host.HostUnsupported(call.op, reason);
        return false;
    };
    if (NeedsGeneratedIndices(call.mode) && !call.quads_converted) {
        return drop(fmt::format("indirect draw of primitive {}: the host converts it with indices "
                                "that depend on the GPU-written count",
                                call.mode));
    }
    const bool counted = call.count_address != 0;
    if (counted && !impl.device.IsKhrDrawIndirectCountSupported()) {
        return drop("a draw count in memory needs VK_KHR_draw_indirect_count");
    }
    if (call.max_draws > 1 && !impl.device.IsMultiDrawIndirectSupported()) {
        return drop("several indirect draws in one call need the multiDrawIndirect feature");
    }
    if (call.max_draws > 1 &&
        (call.stride % 4 != 0 || call.stride < call.command_size)) {
        return drop(fmt::format("indirect draw stride {} is not a multiple of 4 that holds a "
                                "{}-byte command",
                                call.stride, call.command_size));
    }
    args_size = static_cast<u64>(call.max_draws - 1) * call.stride + call.command_size;
    using Policy = DrawContext::ShadowPolicy;
    if (!context.ResolveBuffer(call.indirect, args_size, false, args, Policy::Defer) ||
        !args.buffer) {
        host.HostFailed(call.op, fmt::format("indirect arguments {:#x} (+{:#x}) are not in a pool",
                                             call.indirect, args_size));
        return false;
    }
    if (counted && (!context.ResolveBuffer(call.count_address, COUNT_SIZE, false, count,
                                           Policy::Defer) ||
                    !count.buffer)) {
        host.HostFailed(call.op, fmt::format("draw count {:#x} is not in a pool", call.count_address));
        return false;
    }
    context.ResolveIndirectSource(args, args_size);
    if (counted) {
        context.ResolveIndirectSource(count, COUNT_SIZE);
    }
    return true;
}

} // Anonymous namespace

/// Everything an indirect draw needs recorded OUTSIDE a render pass, before conditional rendering
/// and the draw's own pass: a refresh of mirrored argument/count/index ranges, and the end of a
/// pass whose draws wrote a storage buffer (the culling that builds the draw records is a shader
/// writing one; inside the same pass that write is not visible to the indirect read). Without such
/// writes the barrier in front of the pass already orders the arguments.
void DrawContext::OrderIndirectArguments(const BufferRef& args, u64 args_size,
                                         const BufferRef& count, bool has_count,
                                         BufferRef* index) {
    const auto refresh = [&](const BufferRef& ref, u64 size) {
        if (ref.pool != nullptr && ref.pool->backing == Pool::Backing::Mirror &&
            host->HostMirrorNeedsUpload(*ref.pool, ref.pool_offset, size)) {
            EndPass(BreakCause::IndirectUpload);
            host->HostUploadRange(*ref.pool, ref.pool_offset, size);
        }
    };
    refresh(args, args_size);
    if (has_count) {
        refresh(count, COUNT_SIZE);
    }
    if (index != nullptr) {
        refresh(*index, index->range);
    }
    if (pass_open && pass_buffer_writes) {
        EndPass(BreakCause::PassBufferWrites);
        host->HostMarkWritten();
    }
    // Arguments, counts or indices a counter report of the open pass writes (when the pass ends).
    const auto reported = [&](const BufferRef& ref, u64 size) {
        return ref.pool != nullptr && host->HostReportPending(*ref.pool, ref.pool_offset, size);
    };
    if (pass_open && host->HostReportsPending() &&
        (reported(args, args_size) || (has_count && reported(count, COUNT_SIZE)) ||
         (index != nullptr && reported(*index, index->range)))) {
        EndPass(BreakCause::ReportRead);
        host->HostMarkWritten();
    }
}

void DrawContext::DrawIndirect(const VC::DrawPayload& p) {
    const bool counted = p.count_address != 0;
    QuadIndirectGpu* const quads = p.mode == 7 ? impl.QuadIndirect() : nullptr;
    const IndirectCall call{
        .op = "DrawIndirect",
        .mode = p.mode,
        .command_size = ARRAYS_COMMAND_SIZE,
        .max_draws = counted ? p.max_draws : 1,
        .stride = p.stride != 0 ? p.stride : ARRAYS_COMMAND_SIZE,
        .indirect = p.indirect,
        .count_address = p.count_address,
        .quads_converted = quads != nullptr,
    };
    if (call.max_draws == 0) {
        return;
    }
    BufferRef args;
    BufferRef count;
    u64 args_size = 0;
    if (!ResolveArguments(*this, *host, call, args, count, args_size)) {
        return;
    }
    if (Debug::Tracing()) {
        dbg_call = {};
        dbg_call.indirect = p.indirect;
        if (const u8* bytes = impl.GuestPointer(p.indirect, 16)) {
            std::memcpy(dbg_call.args, bytes, 16);
            dbg_call.args_read = true;
        }
        dbg_call.count = dbg_call.args[0];
        dbg_call.instances = dbg_call.args[1];
        dbg_call.first = dbg_call.args[2];
    }
    // The pass changes happen before conditional rendering begins: a conditional region cannot
    // straddle the end of a render pass.
    OrderIndirectArguments(args, args_size, count, counted, nullptr);
    if (quads != nullptr) {
        // {count, instances, first, base} become VkDrawIndexedIndirectCommand records over the
        // fixed quad index buffer on the GPU, outside the pass and before conditional rendering
        // begins (nvn_quad_indirect.comp).
        if (args.offset % sizeof(u32) != 0 || call.stride % sizeof(u32) != 0) {
            impl.stats.indirect_unsupported.fetch_add(1, std::memory_order_relaxed);
            host->HostUnsupported(call.op, "quad arguments not aligned to 4 bytes");
            return;
        }
        const u64 alignment = (std::max<u64>)(impl.device.GetStorageBufferAlignment(), 4);
        const u64 base = args.offset / alignment * alignment;
        const u64 range = args.offset - base +
                          (call.max_draws > 0 ? static_cast<u64>(call.max_draws - 1) * call.stride
                                              : 0) +
                          QUAD_ARRAYS_COMMAND_SIZE;
        if (range > impl.device.GetMaxStorageBufferRange()) {
            impl.stats.indirect_unsupported.fetch_add(1, std::memory_order_relaxed);
            host->HostUnsupported(call.op, "quad arguments exceed maxStorageBufferRange");
            return;
        }
        EndPass(BreakCause::MeshIndirectConvert);
        u8* unused = nullptr;
        const BufferRef converted =
            AllocArena(static_cast<u64>(call.max_draws) * QUAD_INDEXED_COMMAND_SIZE,
                       impl.device.GetStorageBufferAlignment(), &unused);
        if (!converted.buffer) {
            host->HostFailed(call.op, "no arena space for the converted arguments");
            return;
        }
        host->HostMarkWritten();
        host->HostFlushBarrier();
        quads->Convert(host->HostCmd(), args.buffer, args.offset, call.stride, call.max_draws,
                       converted.buffer, converted.offset);
        // The pass the draw opens starts with the barrier that makes the records visible.
        host->HostMarkWritten();
        if (!host->HostBeginRenderCondition()) {
            return;
        }
        PreparedDraw prepared;
        HostIndex host_index;
        if (PrepareDraw(p.mode, 0, nullptr, prepared, host_index)) {
            vk::CommandBuffer cmd = host->HostCmd();
            cmd.BindIndexBuffer(quads->IndexBuffer(), 0, VK_INDEX_TYPE_UINT32);
            bound_state.ForgetIndexBuffer();
            if (counted) {
                cmd.DrawIndexedIndirectCount(converted.buffer, converted.offset, count.buffer,
                                             count.offset, call.max_draws,
                                             QUAD_INDEXED_COMMAND_SIZE);
                impl.stats.indirect_counted_draws.fetch_add(1, std::memory_order_relaxed);
            } else {
                cmd.DrawIndexedIndirect(converted.buffer, converted.offset, 1,
                                        QUAD_INDEXED_COMMAND_SIZE);
            }
            impl.stats.indirect_draws.fetch_add(1, std::memory_order_relaxed);
        }
        host->HostEndRenderCondition();
        return;
    }
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    PreparedDraw prepared;
    HostIndex host_index;
    if (PrepareDraw(p.mode, 0, nullptr, prepared, host_index)) {
        vk::CommandBuffer cmd = host->HostCmd();
        if (counted) {
            cmd.DrawIndirectCount(args.buffer, args.offset, count.buffer, count.offset,
                                  call.max_draws, call.stride);
            impl.stats.indirect_counted_draws.fetch_add(1, std::memory_order_relaxed);
        } else {
            cmd.DrawIndirect(args.buffer, args.offset, 1, call.stride);
        }
        impl.stats.indirect_draws.fetch_add(1, std::memory_order_relaxed);
    }
    host->HostEndRenderCondition();
}

void DrawContext::DrawIndexedIndirect(const VC::DrawIndexedPayload& p) {
    const bool counted = p.count_address != 0;
    const IndirectCall call{
        .op = "DrawIndexedIndirect",
        .mode = p.mode,
        .command_size = ELEMENTS_COMMAND_SIZE,
        .max_draws = counted ? p.max_draws : 1,
        .stride = p.stride != 0 ? p.stride : ELEMENTS_COMMAND_SIZE,
        .indirect = p.indirect,
        .count_address = p.count_address,
    };
    if (call.max_draws == 0) {
        return;
    }
    BufferRef args;
    BufferRef count;
    u64 args_size = 0;
    if (!ResolveArguments(*this, *host, call, args, count, args_size)) {
        return;
    }
    if (p.index_type == 0 && !impl.device.IsExtIndexTypeUint8Supported()) {
        impl.stats.indirect_unsupported.fetch_add(1, std::memory_order_relaxed);
        host->HostUnsupported(call.op, "8-bit indices need VK_EXT_index_type_uint8 (an indirect "
                                       "draw cannot widen them on the CPU)");
        return;
    }
    // The index range an indirect draw can reach is not known: everything from the base address
    // to the end of the game's buffer object (p.count), or of its pool when no object holds it.
    // Bounding it matters for a mirror pool, whose whole range is hashed and uploaded before use.
    BufferRef index;
    if (!ResolveBuffer(p.index_address, 1, false, index) || index.pool == nullptr) {
        host->HostFailed(call.op, fmt::format("index buffer {:#x} is not in a pool",
                                              p.index_address));
        return;
    }
    index.range = index.pool->size - index.pool_offset;
    if (p.count != 0) {
        index.range = (std::min<u64>)(index.range, p.count);
    }
    if (Debug::Tracing()) {
        dbg_call = {};
        dbg_call.indexed = true;
        dbg_call.index_address = p.index_address;
        dbg_call.indirect = p.indirect;
        dbg_call.count_address = p.count_address;
        dbg_call.max_draws = p.max_draws;
        dbg_call.stride = p.stride;
        dbg_call.args_backing = args.pool ? static_cast<s32>(args.pool->backing) : -1;
        dbg_call.args_flags = args.pool ? args.pool->flags : 0;
        if (const u8* bytes = impl.GuestPointer(p.indirect, sizeof(dbg_call.args))) {
            std::memcpy(dbg_call.args, bytes, sizeof(dbg_call.args));
            dbg_call.args_read = true;
        }
        dbg_call.count = dbg_call.args[0];
        dbg_call.instances = dbg_call.args[1];
        dbg_call.first = static_cast<s32>(dbg_call.args[3]);
        if (dbg_call.args[0] != 0 && watch_indirect_hint == 0) {
            watch_indirect_hint = p.indirect; // a chunk that drew last frame: the first one
        }
    }
    OrderIndirectArguments(args, args_size, count, counted, &index);
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    // Mirrored index ranges were refreshed above, once per submission.
    IndexRange range{p.index_address, index.range, p.index_type, true};
    PreparedDraw prepared;
    HostIndex host_index;
    if (PrepareDraw(p.mode, 0, &range, prepared, host_index)) {
        vk::CommandBuffer cmd = host->HostCmd();
        if (counted) {
            cmd.DrawIndexedIndirectCount(args.buffer, args.offset, count.buffer, count.offset,
                                         call.max_draws, call.stride);
            impl.stats.indirect_counted_draws.fetch_add(1, std::memory_order_relaxed);
        } else {
            cmd.DrawIndexedIndirect(args.buffer, args.offset, 1, call.stride);
        }
        impl.stats.indirect_draws.fetch_add(1, std::memory_order_relaxed);
    }
    host->HostEndRenderCondition();
}

void DrawContext::DrawMeshTasksIndirect(const VC::DrawMeshTasksPayload& p) {
    const bool counted = p.count_address != 0;
    if (p.dims == 3) {
        // The 3D macros launch x*y*z workgroups and leave {x, y, z} in the driver bank for the
        // task program; GPU-written dims cannot reach the bank this replay builds on the CPU.
        ++counters.draws;
        ++counters.draws_skipped;
        impl.stats.indirect_unsupported.fetch_add(1, std::memory_order_relaxed);
        host->HostUnsupported("DrawMeshTasks3DIndirect",
                              "3D indirect mesh-task draws: the task program reads the GPU-written "
                              "workgroup counts from the driver bank");
        return;
    }
    const IndirectCall call{
        .op = "DrawMeshTasksIndirect",
        .mode = MESH_DRAW_MODE,
        .command_size = MESH_TASKS_1D_COMMAND_SIZE,
        .max_draws = p.max_draws,
        .stride = p.stride != 0 ? p.stride : MESH_TASKS_1D_COMMAND_SIZE,
        .indirect = p.indirect,
        .count_address = p.count_address,
    };
    if (call.max_draws == 0) {
        return;
    }
    BufferRef args;
    BufferRef count;
    u64 args_size = 0;
    if (!ResolveArguments(*this, *host, call, args, count, args_size)) {
        return;
    }
    MeshIndirectGpu* const converter = impl.MeshIndirect();
    if (converter == nullptr || args.offset % sizeof(u32) != 0 || call.stride % sizeof(u32) != 0) {
        ++counters.draws;
        ++counters.draws_skipped;
        impl.stats.indirect_unsupported.fetch_add(1, std::memory_order_relaxed);
        host->HostUnsupported(call.op, converter == nullptr
                                           ? "the argument conversion shader is unavailable"
                                           : "arguments not aligned to 4 bytes");
        return;
    }
    if (Debug::Tracing()) {
        // The log line and EDEN_NX2_NVN_WATCH=ind: the first record's {count, first} as guest
        // memory holds it now (what the GPU wrote there the last time the record was produced).
        dbg_call = {};
        dbg_call.indirect = p.indirect;
        dbg_call.count_address = p.count_address;
        dbg_call.max_draws = call.max_draws;
        dbg_call.stride = call.stride;
        dbg_call.args_backing = args.pool ? static_cast<s32>(args.pool->backing) : -1;
        dbg_call.args_flags = args.pool ? args.pool->flags : 0;
        if (const u8* bytes = impl.GuestPointer(p.indirect, MESH_TASKS_1D_COMMAND_SIZE)) {
            std::memcpy(dbg_call.args, bytes, MESH_TASKS_1D_COMMAND_SIZE);
            dbg_call.args_read = true;
        }
        dbg_call.count = dbg_call.args[0];
        dbg_call.instances = 1;
        dbg_call.first = dbg_call.args[1];
        if (watch_indirect_hint == 0) {
            watch_indirect_hint = p.indirect;
        }
    }
    OrderIndirectArguments(args, args_size, count, counted, nullptr);
    // NVN's {count, first} records become VkDrawMeshTasksIndirectCommandEXT {count, 1, 1} on the
    // GPU, outside the pass and before conditional rendering begins (nvn_mesh_indirect.comp).
    EndPass(BreakCause::MeshIndirectConvert);
    u8* unused = nullptr;
    const BufferRef converted =
        AllocArena(static_cast<u64>(call.max_draws) * MESH_TASKS_3D_COMMAND_SIZE,
                   impl.device.GetStorageBufferAlignment(), &unused);
    if (!converted.buffer) {
        host->HostFailed(call.op, "no arena space for the converted arguments");
        return;
    }
    host->HostMarkWritten();
    host->HostFlushBarrier();
    if (Debug::Tracing()) {
        DebugReadbackIndirect(p.indirect, args, MESH_TASKS_1D_COMMAND_SIZE,
                              counted ? &count : nullptr);
    }
    converter->Convert(host->HostCmd(), args.buffer, args.offset, call.stride, call.max_draws,
                       converted.buffer, converted.offset);
    // The pass the draw opens starts with the barrier that makes the records visible.
    host->HostMarkWritten();
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    PreparedDraw prepared;
    HostIndex host_index;
    if (PrepareDraw(MESH_DRAW_MODE, 0, nullptr, prepared, host_index)) {
        vk::CommandBuffer cmd = host->HostCmd();
        if (counted) {
            cmd.DrawMeshTasksIndirectCountEXT(converted.buffer, converted.offset, count.buffer,
                                              count.offset, call.max_draws,
                                              MESH_TASKS_3D_COMMAND_SIZE);
            impl.stats.indirect_counted_draws.fetch_add(1, std::memory_order_relaxed);
        } else {
            cmd.DrawMeshTasksIndirectEXT(converted.buffer, converted.offset, call.max_draws,
                                         MESH_TASKS_3D_COMMAND_SIZE);
        }
        impl.stats.indirect_draws.fetch_add(1, std::memory_order_relaxed);
        impl.stats.mesh_task_draws.fetch_add(1, std::memory_order_relaxed);
    }
    host->HostEndRenderCondition();
}

} // namespace Vulkan::Nvn
