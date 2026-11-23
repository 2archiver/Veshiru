// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <atomic>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// Buffer family: NVNbufferBuilder (kept in place, nxnative `buf_builder_t`) and NVNbuffer. The guest
// object holds the Recorder's tag at +0x00 and the game-writable offset (+0x18) and size (+0x20);
// the Recorder reads them at every use, so every getter here goes through `Recorder::GetBuffer`.
namespace Core::NvnHle::Hle {

// Defined in functions_pool.cpp: guest VA and flags of a memory pool by serial.
bool FindMemoryPoolBySerial(u64 serial, u64& pool_va, u32& flags);

namespace {

// NVNbufferBuilder in place (nxnative buf_builder_t, 0x20 of the builder's 0x40 bytes).
constexpr u64 BbDevice = 0x00;
constexpr u64 BbPool = 0x08;
constexpr u64 BbOffset = 0x10;
constexpr u64 BbSize = 0x18;
constexpr u64 BbBytes = 0x20;

constexpr u32 PoolCpuNoAccess = 0x1;
constexpr u32 PoolPhysical = 0x100;
constexpr u32 PoolVirtual = 0x200;

/// NVNbuffer bytes the SDK clears at Initialize.
constexpr u64 BufferObjectBytes = 0x30;

/// Keeps the device (nxnative): SetDefaults does not reset it.
void nvnBufferBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 device = c.Load<u64>(builder + BbDevice);
    c.Zero(builder, BbBytes);
    c.Store<u64>(builder + BbDevice, device);
}

void nvnBufferBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + BbDevice, c.args.NextPointer());
}

void nvnBufferBuilderSetStorage(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const u64 size = c.args.NextU64();
    c.Store<u64>(builder + BbPool, pool);
    c.Store<s64>(builder + BbOffset, offset);
    c.Store<u64>(builder + BbSize, size);
}

/// No prototype in nvn.h (the header declares the builder getters wrongly, by value): the pointer
/// the builder keeps, as nxnative's NOPROTO bodies do.
void nvnBufferBuilderGetDevice(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + BbDevice));
}

void nvnBufferBuilderGetMemoryPool(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + BbPool));
}

void nvnBufferBuilderGetMemoryOffset(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + BbOffset));
}

void nvnBufferBuilderGetSize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + BbSize));
}

/// The driver does not check a buffer's range against its pool, and the replay never goes by the
/// buffer: commands carry an address and a length. A buffer that reaches past its pool (nxnative:
/// Breath of the Wild's screen-readback buffer) is not an error here, only the access that leaves
/// the pool is. SDK (3.2.0 and 1.0.0): the object is cleared first, and a buffer in a PHYSICAL pool
/// (flags & 0x100, storage only reachable through a virtual pool) fails, leaving it cleared.
void nvnBufferInitialize(Call& c) {
    const u64 buffer = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const u64 pool = c.Load<u64>(builder + BbPool);
    const u64 offset = c.Load<u64>(builder + BbOffset);
    const u64 size = c.Load<u64>(builder + BbSize);
    const auto pool_info = c.rec.GetMemoryPool(pool);
    if (pool_info && (pool_info->flags & PoolPhysical) != 0) {
        c.Zero(buffer, BufferObjectBytes);
        c.ret.SetBool(false);
        return;
    }
    if (!pool_info) {
        // Cleared first, as the SDK does before any check.
        c.Zero(buffer, BufferObjectBytes);
    }
    if (!pool_info || !c.rec.BufferInitialize(buffer, pool, offset, size)) {
        static Common::BootCounter logged;
        if (logged.First()) {
            LOG_WARNING(Core,
                        "[nvn] BufferInitialize failed (pool {:#x} offset {:#x} size {:#x}): the "
                        "pool is not an initialized memory pool",
                        pool, offset, size);
        }
        c.ret.SetBool(false);
        return;
    }
    c.ret.SetBool(true);
}

void nvnBufferFinalize(Call& c) {
    c.rec.BufferFinalize(c.args.NextPointer());
}

void nvnBufferSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

/// 1.2 million calls in a Donkey Kong Bananza run: two lock-free lookups and one 16-byte guest
/// read in the Recorder (pool address + the offset the game may have rewritten).
void nvnBufferGetAddress(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    c.ret.SetU64(info ? info->address : 0);
}

/// NULL for a buffer in a pool without CPU access, as in NVN (see nvnMemoryPoolMap).
void nvnBufferMap(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    if (!info) {
        c.ret.SetU64(0);
        return;
    }
    u64 pool_va = 0;
    u32 flags = 0;
    if (FindMemoryPoolBySerial(info->pool_serial, pool_va, flags) &&
        (flags & (PoolCpuNoAccess | PoolVirtual)) != 0) {
        c.ret.SetU64(0);
        return;
    }
    c.ret.SetU64(info->address);
}

/// The CPU side is guest memory itself. A flush tells the replay which bytes the CPU wrote (it
/// looks at the textures stored there at their next use); an invalidate is the pool's
/// (InvalidateMappedRange: it waits for the GPU's writes with EDEN_NX2_TRY_INVALIDATE_WRITEBACK).
void nvnBufferFlushMappedRange(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    const s64 offset = c.args.NextS64();
    const u64 size = c.args.NextU64();
    if (info && offset >= 0 && static_cast<u64>(offset) < info->size) {
        NoteMappedRangeWritten(c, info->address + static_cast<u64>(offset),
                               (std::min)(size, info->size - static_cast<u64>(offset)));
    }
}

void nvnBufferInvalidateMappedRange(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    const s64 offset = c.args.NextS64();
    const u64 size = c.args.NextU64();
    if (info && offset >= 0 && static_cast<u64>(offset) < info->size) {
        InvalidateMappedRange(c, info->address + static_cast<u64>(offset),
                              (std::min)(size, info->size - static_cast<u64>(offset)));
    }
}

void nvnBufferGetMemoryPool(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    u64 pool_va = 0;
    u32 flags = 0;
    if (info && FindMemoryPoolBySerial(info->pool_serial, pool_va, flags)) {
        c.ret.SetU64(pool_va);
        return;
    }
    c.ret.SetU64(0);
}

void nvnBufferGetMemoryOffset(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    c.ret.SetU64(info ? info->offset : 0);
}

void nvnBufferGetSize(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    c.ret.SetU64(info ? info->size : 0);
}

/// A stable id for debug tools: the host serial.
void nvnBufferGetDebugID(Call& c) {
    const auto info = c.rec.GetBuffer(c.args.NextPointer());
    c.ret.SetU64(info ? info->serial : 0);
}

} // namespace

void RegisterBufferFunctions(Registry& registry) {
    NVN_HLE(registry, nvnBufferBuilderSetDefaults);
    NVN_HLE(registry, nvnBufferBuilderSetDevice);
    NVN_HLE(registry, nvnBufferBuilderSetStorage);
    NVN_HLE(registry, nvnBufferBuilderGetDevice);
    NVN_HLE(registry, nvnBufferBuilderGetMemoryPool);
    NVN_HLE(registry, nvnBufferBuilderGetMemoryOffset);
    NVN_HLE(registry, nvnBufferBuilderGetSize);
    NVN_HLE(registry, nvnBufferInitialize);
    NVN_HLE(registry, nvnBufferFinalize);
    NVN_HLE(registry, nvnBufferSetDebugLabel);
    NVN_HLE(registry, nvnBufferGetAddress);
    NVN_HLE(registry, nvnBufferMap);
    NVN_HLE(registry, nvnBufferGetMemoryPool);
    NVN_HLE(registry, nvnBufferGetMemoryOffset);
    NVN_HLE(registry, nvnBufferGetSize);
    NVN_HLE(registry, nvnBufferGetDebugID);
    NVN_HLE(registry, nvnBufferFlushMappedRange);
    NVN_HLE(registry, nvnBufferInvalidateMappedRange);
}

} // namespace Core::NvnHle::Hle
