// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <bit>
#include <chrono>

#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// Queue family: NVNqueueBuilder (kept in place, nxnative `queue_builder_t`), NVNqueue, submit and
// fence. Present and acquire live in functions_window.cpp.
namespace Core::NvnHle::Hle {
namespace {

// NVNqueueBuilder in place (nxnative queue_builder_t, 0x40 bytes).
constexpr u64 QbDevice = 0x00;
constexpr u64 QbFlags = 0x08;
constexpr u64 QbCommandSize = 0x10;
constexpr u64 QbComputeSize = 0x18;
constexpr u64 QbControlSize = 0x20;
constexpr u64 QbFlushThreshold = 0x28;
constexpr u64 QbMemory = 0x30;
constexpr u64 QbMemorySize = 0x38;
// NVNqueueBuilder fields nxnative does not have: priority lives in the 4 bytes of padding after
// `flags`. The SDK builder is at least 0x48 bytes (its SetDefaults writes +0x44, 3.2.0
// sdk+0x5e6ce0), so the "size set" bytes of the SDK builder (+0x24..+0x28 there) go after the
// nxnative 0x40: a size the game never set is the SDK default and is not validated.
constexpr u64 QbPriority = 0x0C;
constexpr u64 QbCommandSet = 0x40;
constexpr u64 QbComputeSet = 0x41;
constexpr u64 QbControlSet = 0x42;
constexpr u64 QbFlushThresholdSet = 0x43;

/// NVN_QUEUE_PRIORITY_NORMAL, the SDK builder default (+0x44 = 100).
constexpr s32 DefaultQueuePriority = 100;
/// The SDK's sizes when the game did not set one (sdk+0x5e6fa0, sdk+0x5e7cf4).
constexpr s64 DefaultCommandSize = 0x10000;
constexpr s64 DefaultControlSize = 0x4000;
constexpr s64 DefaultComputeSize = 0x40000;

/// NVNqueueBuilderFlags bits the size computation reads.
constexpr u32 QueueFlagNoZcull = 0x2;
constexpr u32 QueueFlagExtraMemory = 0x8; // [INFERENCE] name unknown: adds 0x39500 bytes
constexpr u32 QueueFlagNoInternalMemory = 0x10;
constexpr u32 QueueFlagsNoScratch = 0x15;
/// Device flags (NVNdeviceBuilder) that enlarge the queue's internal blocks.
constexpr u32 DeviceFlagsLargeQueue = 0x300;

/// GPU facts the size reads from the device (see functions_device.cpp DeviceInfo): SMs * warps
/// per SM (item 0x18), and the zcull context size and alignment NvRm stores at device creation
/// (3.2.0 sdk+0x5ab3d4: NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE, alignment 0x1000). The context size is
/// what our nvgpu answers (nvhost_ctrl_gpu.cpp ZCullGetCtxSize: 1 byte, one page once aligned).
constexpr u32 GpuWarpSlots = 576;
constexpr u32 ZcullContextSize = 1;
constexpr u32 ZcullAlignment = 0x1000;

/// The builder's sizes as the SDK reads them: 32-bit, sign-extended, the default when unset.
s64 SizeOrDefault(Call& c, u64 builder, u64 set_offset, u64 value_offset, s64 fallback) {
    if (c.Load<u8>(builder + set_offset) == 0) {
        return fallback;
    }
    return static_cast<s64>(static_cast<s32>(c.Load<u64>(builder + value_offset)));
}

constexpr u64 AlignUp(u64 value, u64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// The queue's driver-internal block (3.2.0 sdk+0x5e7cf4, 1.0.0 sdk+0x5c3b64, same sizes): scratch,
/// a fixed area, optional extra area, a page unless flag 0x10, the zcull context, per-compute-
/// slot areas (the compute memory size / 0x1100 slots) and ray-tracing memory (device flag 0x200
/// only; DK's device flags are 0).
u64 InternalQueueMemorySize(Call& c, u64 builder) {
    const u32 flags = c.Load<u32>(builder + QbFlags);
    const bool large = (c.runtime.device.flags.load(std::memory_order_relaxed) & DeviceFlagsLargeQueue) != 0;
    const u64 scratch = (flags & QueueFlagsNoScratch) == 0 ? (GpuWarpSlots * 4 + 0x30ffU) & ~0xffU : 0;
    u64 total = ((large ? 0x3800U : 0xe00U) + 0xaffU) & 0x7e00U;
    total += scratch;
    total += (flags & QueueFlagExtraMemory) != 0 ? 0x39500 : 0;
    total += (flags & QueueFlagNoInternalMemory) == 0 ? 0x1000 : 0;
    if (ZcullContextSize != 0 && (flags & QueueFlagNoZcull) == 0) {
        total = AlignUp(total, ZcullAlignment) + AlignUp(ZcullContextSize, ZcullAlignment);
    }
    const u64 compute = static_cast<u64>(
        SizeOrDefault(c, builder, QbComputeSet, QbComputeSize, DefaultComputeSize));
    const u64 slots = compute / (large ? 0x1700 : 0x1100);
    total = AlignUp(total, 0x100) + slots * 0x700 + (slots << (large ? 11 : 9)) + slots * 0x800;
    // [INFERENCE] device flag 0x200 creates the ray-tracing state (sdk+0x5f31b0); the HLE does not
    // keep a ray-tracing memory size (DK never sets the flag), so none is added.
    return AlignUp(total, 0x100);
}

/// nvnQueueBuilderGetQueueMemorySize (3.2.0 sdk+0x5e6fa0, 1.0.0 sdk+0x5c2e30): command + control
/// + the internal block rounded to 4 KiB (none with flag 0x10). The compute memory is inside the
/// internal block, as compute slots.
u64 QueueMemorySize(Call& c, u64 builder) {
    const s64 command = SizeOrDefault(c, builder, QbCommandSet, QbCommandSize, DefaultCommandSize);
    const s64 control = SizeOrDefault(c, builder, QbControlSet, QbControlSize, DefaultControlSize);
    const u64 internal = (c.Load<u32>(builder + QbFlags) & QueueFlagNoInternalMemory) == 0
                             ? AlignUp(InternalQueueMemorySize(c, builder), 0x1000)
                             : 0;
    return static_cast<u64>(command + control) + internal;
}

/// The builder checks of nvnQueueInitialize (3.2.0 sdk+0x5e6d80, 1.0.0 the same): every size the
/// game set must respect the device's granularity and minimum (items 0x50/0x51, 0x5D/0x5B and a
/// power of two, 0x53/0x54), the queue memory must be 4 KiB aligned and large enough, and a set
/// flush threshold must lie in [item 0x56, command size].
bool ValidQueueBuilder(Call& c, u64 builder) {
    const auto multiple = [](s64 value, u64 granularity) {
        return static_cast<u64>(value) % granularity == 0;
    };
    const s64 command = SizeOrDefault(c, builder, QbCommandSet, QbCommandSize, DefaultCommandSize);
    if (c.Load<u8>(builder + QbCommandSet) != 0 &&
        (!multiple(command, 0x1000) || static_cast<u32>(command) < 0x10000)) {
        return false;
    }
    if (c.Load<u8>(builder + QbControlSet) != 0) {
        const s64 control = SizeOrDefault(c, builder, QbControlSet, QbControlSize, 0);
        if (control == 0 || std::popcount(static_cast<u64>(control)) > 1 ||
            !multiple(control, 0x1000) || static_cast<u32>(control) < 0x1000) {
            return false;
        }
    }
    if (c.Load<u8>(builder + QbComputeSet) != 0) {
        const s64 compute = SizeOrDefault(c, builder, QbComputeSet, QbComputeSize, 0);
        if (!multiple(compute, 0x100) || (compute != 0 && static_cast<u32>(compute) < 0x4000)) {
            return false;
        }
    }
    const s32 memory_size = static_cast<s32>(c.Load<u64>(builder + QbMemorySize));
    if (memory_size != 0 &&
        ((memory_size & 0xfff) != 0 || (c.Load<u64>(builder + QbMemory) & 0xfff) != 0 ||
         static_cast<u64>(static_cast<s64>(memory_size)) < QueueMemorySize(c, builder))) {
        return false;
    }
    if (c.Load<u8>(builder + QbFlushThresholdSet) != 0) {
        const s64 threshold = SizeOrDefault(c, builder, QbFlushThresholdSet, QbFlushThreshold, 0);
        if (static_cast<u32>(threshold) < 0x1000 ||
            static_cast<u64>(threshold) > static_cast<u64>(command)) {
            return false;
        }
    }
    return true;
}

void nvnQueueBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 device = c.Load<u64>(builder + QbDevice);
    c.Zero(builder, 0x40);
    c.Zero(builder + QbCommandSet, 4);
    c.Store<u64>(builder + QbDevice, device);
    c.Store<s32>(builder + QbPriority, DefaultQueuePriority);
    // What the getters answer for sizes the game did not set.
    c.Store<u64>(builder + QbCommandSize, DefaultCommandSize);
    c.Store<u64>(builder + QbComputeSize, DefaultComputeSize);
    c.Store<u64>(builder + QbControlSize, DefaultControlSize);
    c.Store<u64>(builder + QbFlushThreshold, 4096);
}

void nvnQueueBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + QbDevice, c.args.NextPointer());
}

void nvnQueueBuilderSetFlags(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u32>(builder + QbFlags, static_cast<u32>(c.args.NextS32()));
}

void SetSize(Call& c, u64 value_offset, u64 set_offset) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + value_offset, c.args.NextU64());
    c.Store<u8>(builder + set_offset, 1);
}

void nvnQueueBuilderSetCommandMemorySize(Call& c) {
    SetSize(c, QbCommandSize, QbCommandSet);
}

void nvnQueueBuilderSetComputeMemorySize(Call& c) {
    SetSize(c, QbComputeSize, QbComputeSet);
}

void nvnQueueBuilderSetControlMemorySize(Call& c) {
    SetSize(c, QbControlSize, QbControlSet);
}

void nvnQueueBuilderSetCommandFlushThreshold(Call& c) {
    SetSize(c, QbFlushThreshold, QbFlushThresholdSet);
}

void nvnQueueBuilderGetQueueMemorySize(Call& c) {
    c.ret.SetU64(QueueMemorySize(c, c.args.NextPointer()));
}

void nvnQueueBuilderSetQueueMemory(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + QbMemory, c.args.NextPointer());
    c.Store<u64>(builder + QbMemorySize, c.args.NextU64());
}

void nvnQueueBuilderSetQueuePriority(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + QbPriority, c.args.NextS32());
}

void nvnQueueBuilderGetQueuePriority(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + QbPriority));
}

void nvnQueueBuilderGetDevice(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbDevice));
}

void nvnQueueBuilderGetFlags(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + QbFlags));
}

void nvnQueueBuilderGetCommandMemorySize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbCommandSize));
}

void nvnQueueBuilderGetComputeMemorySize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbComputeSize));
}

void nvnQueueBuilderGetControlMemorySize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbControlSize));
}

void nvnQueueBuilderGetCommandFlushThreshold(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbFlushThreshold));
}

void nvnQueueBuilderGetMemorySize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbMemorySize));
}

void nvnQueueBuilderGetMemory(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + QbMemory));
}

/// Newer interfaces without a prototype. nvnQueueBuilderSetPrimaryQueue (3.2.0 sdk+0x5e7100) only
/// stores the queue for nvnQueueInitialize, which makes a queue with flag 0x4 share the primary
/// queue's GPU context and fails when the primary lacks flag 0x8: the host has one Vulkan queue
/// behind the replay thread, so the pairing has nothing to set up.
void nvnQueueBuilderNoOp(Call& c) {
    static_cast<void>(c);
}

void nvnQueueBuilderGetZero(Call& c) {
    c.ret.SetU64(0);
}

void nvnQueueInitialize(Call& c) {
    const u64 queue = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    if (!ValidQueueBuilder(c, builder)) {
        LOG_WARNING(Core, "[nvn] QueueInitialize {:#x}: the builder's sizes or memory are invalid",
                    queue);
        c.ret.SetBool(false);
        return;
    }
    c.ret.SetBool(c.rec.QueueInitialize(queue));
}

void nvnQueueFinalize(Call& c) {
    c.rec.QueueFinalize(c.args.NextPointer());
}

void nvnQueueSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

/// 0 = NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR. The error info struct is left untouched (its size
/// is not documented).
void nvnQueueGetError(Call& c) {
    c.ret.SetU32(0);
}

void nvnQueueSubmitCommands(Call& c) {
    const u64 queue = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 handles = c.args.NextPointer();
    c.rec.QueueSubmitCommands(queue, count, handles);
}

/// Submissions go to the replay immediately, so a flush has nothing left to push.
void nvnQueueFlush(Call& c) {
    static_cast<void>(c);
}

/// Blocks the guest thread until every submission on the queue completed and its guest writes
/// landed. The thread sleeps in the kernel; the core runs other guest threads meanwhile.
void nvnQueueFinish(Call& c) {
    const u64 queue = c.args.NextPointer();
    const u64 seq = c.rec.QueueLastSeq(queue);
    if (seq == 0) {
        return;
    }
    VideoCommon::Nvn::Recorder& rec = c.rec;
    c.WaitUntil([&rec, seq] { return rec.CompletedSeq() >= seq; }, std::chrono::nanoseconds{-1});
}

void nvnQueueFenceSync(Call& c) {
    const u64 queue = c.args.NextPointer();
    const u64 sync = c.args.NextPointer();
    const u32 condition = c.args.NextU32();
    const u32 flags = c.args.NextU32();
    c.rec.QueueFenceSync(queue, sync, condition, flags);
}

/// A sync that nobody fenced (the acquire syncs the window hands out: the texture is free by the
/// time AcquireTexture returns) has nothing to wait for. A GPU-side wait on it would park the
/// replay until its timeout.
void nvnQueueWaitSync(Call& c) {
    const u64 queue = c.args.NextPointer();
    const u64 sync = c.args.NextPointer();
    const auto state = c.rec.GetSyncState(sync);
    if (state && state->fenced) {
        c.rec.QueueWaitSync(queue, sync);
    }
    c.ret.SetBool(true);
}

void nvnQueueGetTotalMemoryUsed(Call& c) {
    c.ret.SetU64(0);
}

} // namespace

void RegisterQueueFunctions(Registry& registry) {
    NVN_HLE(registry, nvnQueueBuilderSetDefaults);
    NVN_HLE(registry, nvnQueueBuilderSetDevice);
    NVN_HLE(registry, nvnQueueBuilderSetFlags);
    NVN_HLE(registry, nvnQueueBuilderSetCommandMemorySize);
    NVN_HLE(registry, nvnQueueBuilderSetComputeMemorySize);
    NVN_HLE(registry, nvnQueueBuilderSetControlMemorySize);
    NVN_HLE(registry, nvnQueueBuilderSetCommandFlushThreshold);
    NVN_HLE(registry, nvnQueueBuilderGetQueueMemorySize);
    NVN_HLE(registry, nvnQueueBuilderSetQueueMemory);
    NVN_HLE(registry, nvnQueueBuilderSetQueuePriority);
    NVN_HLE(registry, nvnQueueBuilderGetQueuePriority);
    NVN_HLE(registry, nvnQueueBuilderGetDevice);
    NVN_HLE(registry, nvnQueueBuilderGetFlags);
    NVN_HLE(registry, nvnQueueBuilderGetCommandMemorySize);
    NVN_HLE(registry, nvnQueueBuilderGetComputeMemorySize);
    NVN_HLE(registry, nvnQueueBuilderGetControlMemorySize);
    NVN_HLE(registry, nvnQueueBuilderGetCommandFlushThreshold);
    NVN_HLE(registry, nvnQueueBuilderGetMemorySize);
    NVN_HLE(registry, nvnQueueBuilderGetMemory);
    registry.Add<&nvnQueueBuilderNoOp>("nvnQueueBuilderSetPrimaryQueue");
    registry.Add<&nvnQueueBuilderNoOp>("nvnQueueBuilderSetRayTracingMemorySize");
    registry.Add<&nvnQueueBuilderGetZero>("nvnQueueBuilderGetPrimaryQueue");
    registry.Add<&nvnQueueBuilderGetZero>("nvnQueueBuilderGetRayTracingMemorySize");
    NVN_HLE(registry, nvnQueueInitialize);
    NVN_HLE(registry, nvnQueueFinalize);
    NVN_HLE(registry, nvnQueueSetDebugLabel);
    NVN_HLE(registry, nvnQueueGetError);
    NVN_HLE(registry, nvnQueueSubmitCommands);
    NVN_HLE(registry, nvnQueueFlush);
    NVN_HLE(registry, nvnQueueFinish);
    NVN_HLE(registry, nvnQueueFenceSync);
    NVN_HLE(registry, nvnQueueWaitSync);
    registry.Add<&nvnQueueGetTotalMemoryUsed>("nvnQueueGetTotalCommandMemoryUsed");
    registry.Add<&nvnQueueGetTotalMemoryUsed>("nvnQueueGetTotalControlMemoryUsed");
    registry.Add<&nvnQueueGetTotalMemoryUsed>("nvnQueueGetTotalComputeMemoryUsed");
    registry.Add<&nvnQueueGetTotalMemoryUsed>("nvnQueueGetTotalRayTracingMemoryUsed");
    registry.Add<&nvnQueueBuilderNoOp>("nvnQueueResetMemoryUsageCounts");
}

} // namespace Core::NvnHle::Hle
