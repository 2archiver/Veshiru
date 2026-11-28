// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/debugger/mapped_range_hook.h"
#include "core/hle/nvn/config.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// Pool family: NVNmemoryPool and its builder, NVNtexturePool, NVNsamplerPool. Behaviour ported from
// nxnative hle_nvn.c ("Memory pools" and "Descriptor pools"). The guest objects only carry the
// Recorder's tag; pool address, size and flags live in the Recorder.
namespace Core::NvnHle::Hle {

// ---- Pool table shared with functions_buffer.cpp and functions_event.cpp -----------------------------
//
// The Recorder identifies a pool by serial and answers its queries by guest VA of the NVNmemoryPool
// object. A buffer, an event or a descriptor pool only knows the serial of its pool, yet NVN returns
// the pool's guest pointer (`BufferGetMemoryPool`, ...) and `BufferMap` needs the pool flags. This
// table maps serial -> {guest VA, address, flags}. It is written at MemoryPoolInitialize/Finalize
// only; readers take the mutex briefly (the hot path of the game, MemoryPoolGetFlags, bypasses it).
// [INFERENCE] Obsolete as soon as the Recorder's BufferInfo / DescriptorPoolInfo carry the pool's VA
// and flags.
namespace {

struct PoolEntry {
    u64 va;
    u64 address;
    u64 size;
    u32 flags;
};

class PoolTable {
public:
    void Insert(u64 serial, const PoolEntry& entry) {
        std::lock_guard lock{mutex};
        // The object at this VA was initialized again (or belongs to an earlier recorder in unit
        // tests): its old entry is stale.
        for (auto it = entries.begin(); it != entries.end();) {
            it = it->second.va == entry.va ? entries.erase(it) : std::next(it);
        }
        entries[serial] = entry;
    }
    void EraseByVa(u64 va) {
        std::lock_guard lock{mutex};
        for (auto it = entries.begin(); it != entries.end();) {
            it = it->second.va == va ? entries.erase(it) : std::next(it);
        }
    }
    void Clear() {
        std::lock_guard lock{mutex};
        entries.clear();
    }
    bool FindBySerial(u64 serial, PoolEntry& out) {
        std::lock_guard lock{mutex};
        const auto it = entries.find(serial);
        if (it == entries.end()) {
            return false;
        }
        out = it->second;
        return true;
    }
    /// Calls `visit(serial, entry)` for every entry until it returns true.
    template <typename Visitor>
    void ForEach(Visitor&& visit) {
        std::lock_guard lock{mutex};
        for (const auto& [serial, entry] : entries) {
            if (visit(serial, entry)) {
                return;
            }
        }
    }

private:
    std::mutex mutex;
    std::unordered_map<u64, PoolEntry> entries;
};

PoolTable& Table() {
    // Leaked on purpose: a late guest call must not meet a destroyed table at process exit.
    static PoolTable* const table = new PoolTable;
    return *table;
}

} // namespace

bool FindMemoryPoolBySerial(u64 serial, u64& pool_va, u32& flags);
bool FindMemoryPoolByAddress(VideoCommon::Nvn::Recorder& rec, u64 address, u64& pool_va,
                             u64& offset);

/// Guest VA and flags of the memory pool with `serial`. False when the pool is not tracked.
bool FindMemoryPoolBySerial(u64 serial, u64& pool_va, u32& flags) {
    PoolEntry entry{};
    if (!Table().FindBySerial(serial, entry)) {
        return false;
    }
    pool_va = entry.va;
    flags = entry.flags;
    return true;
}

/// The memory pool whose storage contains the GPU address `address`: guest VA of the pool object
/// and the offset of `address` in it. `rec` validates every candidate (the table may hold stale
/// entries of an earlier recorder in unit tests).
bool FindMemoryPoolByAddress(VideoCommon::Nvn::Recorder& rec, u64 address, u64& pool_va,
                             u64& offset) {
    bool found = false;
    Table().ForEach([&](u64 serial, const PoolEntry& entry) {
        if (entry.address == 0 || address < entry.address || address >= entry.address + entry.size) {
            return false;
        }
        const auto info = rec.GetMemoryPool(entry.va);
        if (!info || info->serial != serial) {
            return false;
        }
        pool_va = entry.va;
        offset = address - entry.address;
        found = true;
        return true;
    });
    return found;
}

/// Forgets every pool: serials restart with the next Recorder.
void ClearPoolTable() {
    Table().Clear();
}

namespace {

// NVNmemoryPoolBuilder in place (nxnative pool_builder_t, 0x20 of the builder's 0x40 bytes).
constexpr u64 PbDevice = 0x00;
constexpr u64 PbMemory = 0x08;
constexpr u64 PbSize = 0x10;
constexpr u64 PbFlags = 0x18;

// NVNsamplerBuilder in place (nxnative samp_builder_t): {NVNdevice* dev; samp_par p;}, samp_par =
// {u8 minf, magf, ws, wt, wr, cmp_mode, cmp_func, reduction; f32 lod_min, lod_max, lod_bias, aniso,
// border[4]}. The sampler family (functions_sampler.cpp) owns the setters.
constexpr u64 SbParams = 0x08;
constexpr size_t SampParBytes = 40;

// NVNmappingRequest (0x28 bytes): {pool, physical offset, virtual offset, size, storage class}.
constexpr u64 MapRequestStride = 0x28;
/// Requests per kernel call in the SDK's MapVirtual (3.2.0 sdk+0x5f1620).
constexpr s32 MapRequestBatch = 512;

// NVNmemoryPoolFlags (nvn.h).
constexpr u32 PoolCpuNoAccess = 0x1;
constexpr u32 PoolCpuUncached = 0x2;
constexpr u32 PoolCpuCached = 0x4;
constexpr u32 PoolGpuNoAccess = 0x8;
constexpr u32 PoolGpuUncached = 0x10;
constexpr u32 PoolGpuCached = 0x20;
constexpr u32 PoolPhysical = 0x100;
constexpr u32 PoolVirtual = 0x200;
constexpr u32 PoolBuilderDefaultFlags = 0x21; // CPU_NO_ACCESS | GPU_CACHED

/// nvnMemoryPoolGetFlags masks the stored flags with 0x1fffff (3.2.0 sdk+0x5e1200); the 1.0.0
/// SDK (sdk+0x5bd330) with 0x3fffff.
constexpr u32 PoolGetFlagsMask = 0x1fffff;
constexpr u32 PoolGetFlagsMaskOlderSdk = 0x3fffff;

/// Storage of a non-virtual pool: address aligned and size a multiple of 4 KiB (3.2.0
/// sdk+0x5f0cb8, 1.0.0 sdk+0x5ccdc0).
constexpr u64 PoolStorageAlignment = 0x1000;
/// The device's pool page (device +0xd8, NVN_DEVICE_INFO_MEMORY_POOL_PAGE_SIZE, 65536 in
/// functions_device.cpp): size of PHYSICAL and VIRTUAL pools and every MapVirtual offset/size.
constexpr u64 PoolPageSize = 0x10000;

/// Device flag bit 15 (device +0x8, the builder's flags): CPU-uncached pools become CPU-cached.
constexpr u32 DeviceFlagUpgradeCpuUncached = 0x8000;

/// The flags MemoryPoolInitialize stores (3.2.0 sdk+0x5e0ea0, 1.0.0 sdk+0x5bcef0): one CPU mode
/// (UNCACHED over CACHED over NO_ACCESS, UNCACHED when none is set, CACHED instead of UNCACHED on
/// a device with flag bit 15), one GPU mode (CACHED over UNCACHED over NO_ACCESS, CACHED when none
/// is set); a PHYSICAL pool has no GPU access and a VIRTUAL pool no CPU access. Bits 6 and up are
/// kept as given.
u32 StoredPoolFlags(u32 flags, u32 device_flags) {
    u32 cpu;
    bool uncached;
    if ((flags & PoolCpuUncached) != 0) {
        cpu = PoolCpuUncached;
        uncached = true;
    } else if ((flags & PoolCpuCached) != 0) {
        cpu = PoolCpuCached;
        uncached = false;
    } else if ((flags & PoolCpuNoAccess) != 0) {
        cpu = PoolCpuNoAccess;
        uncached = false;
    } else {
        cpu = PoolCpuUncached;
        uncached = true;
    }
    if (uncached && (device_flags & DeviceFlagUpgradeCpuUncached) != 0) {
        cpu = PoolCpuCached;
    }
    u32 gpu;
    if ((flags & PoolGpuCached) != 0) {
        gpu = PoolGpuCached;
    } else if ((flags & PoolGpuUncached) != 0) {
        gpu = PoolGpuUncached;
    } else if ((flags & PoolGpuNoAccess) != 0) {
        gpu = PoolGpuNoAccess;
    } else {
        gpu = PoolGpuCached;
    }
    if ((flags & PoolPhysical) != 0) {
        gpu = PoolGpuNoAccess;
    }
    if ((flags & PoolVirtual) != 0) {
        cpu = PoolCpuNoAccess;
    }
    return cpu | (flags & ~u32{0x3f}) | gpu;
}

/// SDK: storage, size and flags are reset, flags to CPU_NO_ACCESS | GPU_CACHED (0x21); the device
/// is left as it was.
void nvnMemoryPoolBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + PbMemory, 0);
    c.Store<u64>(builder + PbSize, 0);
    c.Store<u64>(builder + PbFlags, PoolBuilderDefaultFlags);
}

void nvnMemoryPoolBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + PbDevice, c.args.NextPointer());
}

void nvnMemoryPoolBuilderSetStorage(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 memory = c.args.NextPointer();
    const u64 size = c.args.NextU64();
    c.Store<u64>(builder + PbMemory, memory);
    c.Store<u64>(builder + PbSize, size);
}

void nvnMemoryPoolBuilderSetFlags(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u32>(builder + PbFlags, static_cast<u32>(c.args.NextS32()));
}

void nvnMemoryPoolBuilderGetDevice(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + PbDevice));
}

void nvnMemoryPoolBuilderGetMemory(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + PbMemory));
}

void nvnMemoryPoolBuilderGetSize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + PbSize));
}

void nvnMemoryPoolBuilderGetFlags(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + PbFlags));
}

/// A pool's GPU address is the guest VA of its storage. A virtual pool (flags & 0x200) has no
/// storage until MapVirtual: the Recorder gives it an address range of its own (see MapVirtual).
///
/// SDK (3.2.0 sdk+0x5e0ea0 -> +0x5f0c30 / +0x5f1440, 1.0.0 sdk+0x5bcef0 -> +0x5ccd38 / +0x5cd560)
/// fails a non-virtual pool whose storage is null or not 4 KiB aligned, whose size is 0 or not a
/// multiple of 4 KiB, a PHYSICAL pool whose size is not a multiple of the pool page; a VIRTUAL
/// pool fails with storage, with size 0 or not a multiple of the pool page. The flags it stores
/// (and GetFlags returns) are normalized, see StoredPoolFlags.
void nvnMemoryPoolInitialize(Call& c) {
    const u64 pool = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const u64 storage = c.Load<u64>(builder + PbMemory);
    const u64 size = c.Load<u64>(builder + PbSize);
    const u32 flags =
        StoredPoolFlags(c.Load<u32>(builder + PbFlags), c.runtime.device.flags.load());
    bool valid;
    if ((flags & PoolVirtual) != 0) {
        valid = storage == 0 && size != 0 && size % PoolPageSize == 0;
    } else {
        valid = storage != 0 && storage % PoolStorageAlignment == 0 && size != 0 &&
                size % PoolStorageAlignment == 0 &&
                ((flags & PoolPhysical) == 0 || size % PoolPageSize == 0);
    }
    if (!valid) {
        static Common::BootCounter logged;
        if (logged.First()) {
            LOG_WARNING(Core, "[nvn] MemoryPoolInitialize refused: storage {:#x}, size {:#x}, "
                              "flags {:#x}",
                        storage, size, flags);
        }
        c.ret.SetBool(false);
        return;
    }
    if (!c.rec.MemoryPoolInitialize(pool, storage, size, flags)) {
        c.ret.SetBool(false);
        return;
    }
    if (const auto info = c.rec.GetMemoryPool(pool)) {
        Table().Insert(info->serial, PoolEntry{pool, info->address, info->size, info->flags});
    }
    c.ret.SetBool(true);
}

void nvnMemoryPoolFinalize(Call& c) {
    const u64 pool = c.args.NextPointer();
    Table().EraseByVa(pool);
    c.rec.MemoryPoolFinalize(pool);
}

/// Debug labels are for NVN's debugging tools only.
void nvnPoolSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

} // namespace

namespace {

/// EDEN_NX2_DIAG_INVALIDATE=1 (measurement only): one log line per Flush/InvalidateMappedRange,
/// with the first raw words the CPU sees at the range start and where the GPU stands.
bool DiagMappedRanges() {
    static const bool on = SwitchOn("EDEN_NX2_DIAG_INVALIDATE");
    return on;
}

/// EDEN_NX2_DIAG_INVALIDATE_WORDS=<n>: how many words those lines show (default 16, at most 64).
size_t DiagMappedRangeWords() {
    static const size_t count = [] {
        const char* text = std::getenv("EDEN_NX2_DIAG_INVALIDATE_WORDS");
        const unsigned long n = text != nullptr ? std::strtoul(text, nullptr, 10) : 0;
        return n == 0 ? size_t{16} : static_cast<size_t>((std::min)(n, 64UL));
    }();
    return count;
}

void LogMappedRange(Call& c, const char* what, u64 address, u64 size, const std::string& note) {
    if (size == 0) {
        return;
    }
    static std::atomic<u64> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) >= 200000) {
        return;
    }
    std::array<u32, 64> words{};
    const u64 bytes = (std::min<u64>)(size, DiagMappedRangeWords() * sizeof(u32));
    c.mem.Read(address, words.data(), bytes);
    std::string hex;
    for (u64 i = 0; i < bytes / 4; ++i) {
        hex += fmt::format(" {:08x}", words[i]);
    }
    LOG_INFO(Core,
             "[nvn] mapped-range {} f{} addr={:#x} size={:#x} last_seq={} completed={} words:{}{}",
             what, Core::g_nx2_present_count.load(std::memory_order_relaxed), address, size,
             c.rec.LastSeq(), c.rec.CompletedSeq(), hex, note);
}

} // namespace

void NoteMappedRangeWritten(Call& c, u64 address, u64 size) {
    c.rec.NotifyCpuWrite(address, size);
    if (DiagMappedRanges()) {
        LogMappedRange(c, "flush", address, size, {});
    }
}

/// The CPU is about to read GPU results from [address, address + size) (an invalidate: on the
/// console it drops stale CPU cache lines; the GPU's writes are already in memory as far as the
/// GPU has run). The replay lags the guest, so with EDEN_NX2_TRY_INVALIDATE_WRITEBACK the call
/// waits, bounded, until the submissions made before it that write the range completed on the
/// host GPU: what the GPU would have written by then on the console (DESIGN.md "Invalidate").
void InvalidateMappedRange(Call& c, u64 address, u64 size) {
    std::string note;
    const u64 last_seq = c.rec.LastSeq();
    const u64 marker = c.rec.GpuWriteMarker();
    if (marker != 0 && size != 0) {
        const auto begin = std::chrono::steady_clock::now();
        bool reached = false;
        u64 writer = 0;
        u32 limit_ms = 0;
        const auto ready = [&] {
            if (!reached) {
                const VideoCommon::Nvn::GpuWriteState state =
                    c.rec.PollGpuWrites(marker, address, size, last_seq);
                limit_ms = state.wait_limit_ms;
                if (!state.replay_reached) {
                    return false;
                }
                reached = true;
                writer = state.writer_seq;
            }
            return writer == 0 || c.rec.CompletedSeq() >= writer;
        };
        bool done = ready();
        if (!done) {
            // Slices: the replay reaching the marker wakes nobody, the GPU's completions do.
            while (!done) {
                const auto waited = std::chrono::steady_clock::now() - begin;
                if (waited >= std::chrono::milliseconds{limit_ms} || c.runtime.HasFailed()) {
                    break;
                }
                done = c.WaitUntil(ready, std::chrono::milliseconds{1});
            }
        }
        const auto waited_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
        static std::atomic<u64> waits{0};
        static std::atomic<u64> timeouts{0};
        static std::atomic<u64> waited_total_us{0};
        if (writer != 0) {
            waits.fetch_add(1, std::memory_order_relaxed);
            waited_total_us.fetch_add(static_cast<u64>(waited_us), std::memory_order_relaxed);
        }
        if (!done) {
            if (timeouts.fetch_add(1, std::memory_order_relaxed) == 0) {
                LOG_WARNING(Core,
                            "[nvn] INVALIDATE_WRITEBACK: seq {} (last {}) not complete after {} ms "
                            "(completed {}, replay reached: {}); the read goes on without it",
                            writer, last_seq, limit_ms, c.rec.CompletedSeq(), reached);
            }
        }
        if (const u64 n = waits.load(std::memory_order_relaxed); n != 0 && n % 600 == 0 && writer != 0) {
            LOG_INFO(Core,
                     "[nvn] INVALIDATE_WRITEBACK: {} waits, {} timeouts, {} ms waited in all",
                     n, timeouts.load(std::memory_order_relaxed),
                     waited_total_us.load(std::memory_order_relaxed) / 1000);
        }
        note = fmt::format(" writer_seq={} waited_us={} done={}", writer, waited_us, done);
    }
    if (DiagMappedRanges()) {
        LogMappedRange(c, "invalidate", address, size, note);
    }
    if (const auto hook = Core::Trace::g_invalidate_hook.load(std::memory_order_acquire)) {
        hook(address, size);
    }
}

namespace {

/// The CPU side of a pool is guest memory itself: the replay reads it from there (zero-copy import
/// or the mirror upload). A flush tells the replay which bytes the CPU wrote, so textures stored
/// there are looked at again on their next use. The GPU's writes reach guest memory when the
/// GPU runs the write-back (imported pools: the end of the command buffer) or when the submission
/// completes (mirror pools); an invalidate waits for them with EDEN_NX2_TRY_INVALIDATE_WRITEBACK.
void nvnMemoryPoolFlushMappedRange(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const u64 size = c.args.NextU64();
    if (const auto info = c.rec.QueryMemoryPool(pool);
        info && offset >= 0 && static_cast<u64>(offset) < info->size) {
        NoteMappedRangeWritten(c, info->address + static_cast<u64>(offset),
                               (std::min)(size, info->size - static_cast<u64>(offset)));
    }
}

void nvnMemoryPoolInvalidateMappedRange(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const u64 size = c.args.NextU64();
    if (const auto info = c.rec.QueryMemoryPool(pool);
        info && offset >= 0 && static_cast<u64>(offset) < info->size) {
        InvalidateMappedRange(c, info->address + static_cast<u64>(offset),
                              (std::min)(size, info->size - static_cast<u64>(offset)));
    }
}

/// SDK (3.2.0 sdk+0x5e1140, 1.0.0 sdk+0x5bd260): the storage pointer, NULL when the stored flags
/// have CPU_NO_ACCESS (always for a virtual pool). EA FC's renderer takes a buffer's GPU address
/// as GetAddress + (data pointer - Map), with a null data pointer for such a buffer.
void nvnMemoryPoolMap(Call& c) {
    const auto info = c.rec.QueryMemoryPool(c.args.NextPointer());
    c.ret.SetU64(info && (info->flags & (PoolCpuNoAccess | PoolVirtual)) == 0 ? info->address : 0);
}

/// SDK (3.2.0 sdk+0x5e1180, 1.0.0 sdk+0x5bd2a0): the pool's GPU address, which a PHYSICAL pool
/// does not have (it is only reachable through virtual pools; Initialize leaves it 0, 3.2.0
/// sdk+0x5f1140). The Recorder keeps the physical storage address for MapVirtual.
void nvnMemoryPoolGetBufferAddress(Call& c) {
    const auto info = c.rec.QueryMemoryPool(c.args.NextPointer());
    c.ret.SetU64(info && (info->flags & PoolPhysical) == 0 ? info->address : 0);
}

void nvnMemoryPoolGetSize(Call& c) {
    const auto info = c.rec.QueryMemoryPool(c.args.NextPointer());
    c.ret.SetU64(info ? info->size : 0);
}

/// 854 calls per frame: one tag read and a lock-free slab lookup, no allocation.
void nvnMemoryPoolGetFlags(Call& c) {
    const auto info = c.rec.QueryMemoryPool(c.args.NextPointer());
    const u32 mask = c.runtime.device.OlderSdk() ? PoolGetFlagsMaskOlderSdk : PoolGetFlagsMask;
    c.ret.SetU32(info ? info->flags & mask : 0);
}

/// nvnMemoryPoolMapVirtual(pool, count, requests): maps ranges of physical pools into a virtual
/// pool; the replay keeps the mapping table and resolves the virtual pool's GPU addresses through
/// it. SDK (3.2.0 sdk+0x5e1190 -> +0x5f1620, 1.0.0 sdk+0x5bd2b0 -> +0x5cd730): fails on a pool
/// that is not virtual, succeeds with nothing to do for count <= 0, fails a request whose physical
/// offset, virtual offset or size is not a multiple of the pool page; the kernel refuses ranges
/// outside the pools (checked here). The SDK hands the requests to the kernel in batches of up
/// to 512 and stops at the first failing batch, leaving the earlier batches mapped: so does the
/// HLE (a request array the guest cannot read fails its batch).
void nvnMemoryPoolMapVirtual(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 requests = c.args.NextPointer();
    const auto info = c.rec.GetMemoryPool(pool);
    if (!info || (info->flags & PoolVirtual) == 0) {
        c.ret.SetBool(false);
        return;
    }
    struct GuestRequest {
        u64 physical_pool;
        s64 physical_offset;
        s64 virtual_offset;
        u64 size;
        u64 storage_class;
    };
    static_assert(sizeof(GuestRequest) == MapRequestStride);
    std::array<GuestRequest, MapRequestBatch> batch;
    std::vector<VideoCommon::Nvn::Recorder::VirtualMapRequest> mapped;
    mapped.reserve((std::min)(static_cast<size_t>((std::max)(count, 0)), batch.size()));
    for (s64 first = 0; first < count; first += MapRequestBatch) {
        const s32 n = static_cast<s32>((std::min<s64>)(count - first, MapRequestBatch));
        if (!c.mem.Read(requests + static_cast<u64>(first) * MapRequestStride, batch.data(),
                        static_cast<size_t>(n) * MapRequestStride)) {
            c.ret.SetBool(false);
            return;
        }
        mapped.clear();
        for (s32 i = 0; i < n; ++i) {
            const GuestRequest& r = batch[static_cast<size_t>(i)];
            if (((static_cast<u64>(r.physical_offset) | static_cast<u64>(r.virtual_offset) |
                  r.size) &
                 (PoolPageSize - 1)) != 0) {
                c.ret.SetBool(false);
                return;
            }
            if (r.physical_pool == 0) {
                // Unmapping: only the virtual range has to be valid.
                if (r.virtual_offset < 0 || r.size > info->size ||
                    static_cast<u64>(r.virtual_offset) > info->size - r.size) {
                    c.ret.SetBool(false);
                    return;
                }
                mapped.push_back({0, 0, static_cast<u64>(r.virtual_offset), r.size});
                continue;
            }
            const auto physical_info = c.rec.GetMemoryPool(r.physical_pool);
            if (!physical_info || r.physical_offset < 0 || r.virtual_offset < 0 ||
                r.size > info->size || static_cast<u64>(r.virtual_offset) > info->size - r.size ||
                r.size > physical_info->size ||
                static_cast<u64>(r.physical_offset) > physical_info->size - r.size) {
                static Common::BootCounter logged;
                if (logged.First()) {
                    LOG_WARNING(Core,
                                "[nvn] MapVirtual: bad request (virtual {:#x}+{:#x} of {:#x}, "
                                "physical pool {:#x} offset {:#x})",
                                r.virtual_offset, r.size, info->size, r.physical_pool,
                                r.physical_offset);
                }
                c.ret.SetBool(false);
                return;
            }
            mapped.push_back({r.physical_pool, static_cast<u64>(r.physical_offset),
                              static_cast<u64>(r.virtual_offset), r.size});
        }
        if (!c.rec.MemoryPoolMapVirtual(pool, mapped)) {
            c.ret.SetBool(false);
            return;
        }
    }
    c.ret.SetBool(true);
}

// ---- Texture and sampler pools ---------------------------------------------------------------------------

/// The SDK (3.2.0 sdk+0x5eebe0, 1.0.0 sdk+0x5cad10) writes one reserved descriptor, id 0: a null
/// texture (block-linear 1D header with address 0). The replay answers an id nothing was
/// registered at with no texture (DrawContext::TextureOfId), the same null binding.
void nvnTexturePoolInitialize(Call& c) {
    const u64 texture_pool = c.args.NextPointer();
    const u64 memory_pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const s32 count = c.args.NextS32();
    c.ret.SetBool(c.rec.TexturePoolInitialize(texture_pool, memory_pool, offset, count));
}

void nvnTexturePoolFinalize(Call& c) {
    c.rec.TexturePoolFinalize(c.args.NextPointer());
}

/// 46,000 calls in a Donkey Kong Bananza run: the Recorder resolves the texture and the optional
/// view (16 bytes of GuestTextureView) and stores the host serial in its descriptor table.
void nvnTexturePoolRegisterTexture(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s32 id = c.args.NextS32();
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    c.rec.RegisterTexture(pool, id, texture, view);
}

/// Image descriptors share the texture pool's id space (nxnative: Hyrule Warriors registers the
/// images its compute shaders write).
void nvnTexturePoolRegisterImage(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s32 id = c.args.NextS32();
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    c.rec.RegisterImage(pool, id, texture, view);
}

u64 MemoryPoolVaOfSerial(u64 pool_serial) {
    u64 va = 0;
    u32 flags = 0;
    return FindMemoryPoolBySerial(pool_serial, va, flags) ? va : 0;
}

void nvnTexturePoolGetMemoryPool(Call& c) {
    const auto info = c.rec.GetTexturePool(c.args.NextPointer());
    c.ret.SetU64(info ? MemoryPoolVaOfSerial(info->pool_serial) : 0);
}

void nvnTexturePoolGetMemoryOffset(Call& c) {
    const auto info = c.rec.GetTexturePool(c.args.NextPointer());
    c.ret.SetU64(info ? info->pool_offset : 0);
}

void nvnTexturePoolGetSize(Call& c) {
    const auto info = c.rec.GetTexturePool(c.args.NextPointer());
    c.ret.SetU32(info ? info->count : 0);
}

/// The SDK (3.2.0 sdk+0x5ea8d0, 1.0.0 sdk+0x5c6690) writes three reserved sampler descriptors
/// into the pool, ids 0..2, from builders that differ from SamplerBuilderSetDefaults: wrap
/// CLAMP_TO_EDGE on all axes, LOD clamp [0, 1000], bias 0, border 0, compare off (func LESS),
/// reduction AVERAGE; id 0 linear/linear anisotropy 1, id 1 linear/linear anisotropy 2, id 2
/// nearest/nearest anisotropy 2. The replay resolves sampler ids through the Recorder's table,
/// so the same samplers are registered there (ids the pool holds; the SDK writes all three).
VideoCommon::Nvn::SamplerDesc ReservedSampler(u8 filter, f32 max_anisotropy) {
    constexpr u8 WrapClampToEdge = 7;
    constexpr u8 CompareFuncLess = 2;
    VideoCommon::Nvn::SamplerDesc desc{};
    desc.min_filter = filter;
    desc.mag_filter = filter;
    desc.wrap_s = desc.wrap_t = desc.wrap_r = WrapClampToEdge;
    desc.compare_func = CompareFuncLess;
    desc.lod_max = 1000.0f;
    desc.max_anisotropy = max_anisotropy;
    return desc;
}

void nvnSamplerPoolInitialize(Call& c) {
    const u64 sampler_pool = c.args.NextPointer();
    const u64 memory_pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const s32 count = c.args.NextS32();
    if (!c.rec.SamplerPoolInitialize(sampler_pool, memory_pool, offset, count)) {
        c.ret.SetBool(false);
        return;
    }
    constexpr u8 FilterNearest = 0; // NVN_MIN_FILTER_NEAREST / NVN_MAG_FILTER_NEAREST
    constexpr u8 FilterLinear = 1;  // NVN_MIN_FILTER_LINEAR / NVN_MAG_FILTER_LINEAR
    const VideoCommon::Nvn::SamplerDesc reserved[] = {
        ReservedSampler(FilterLinear, 1.0f),
        ReservedSampler(FilterLinear, 2.0f),
        ReservedSampler(FilterNearest, 2.0f),
    };
    for (s32 id = 0; id < static_cast<s32>(std::size(reserved)) && id < count; ++id) {
        c.rec.RegisterSamplerParams(sampler_pool, id, reserved[id]);
    }
    c.ret.SetBool(true);
}

void nvnSamplerPoolFinalize(Call& c) {
    c.rec.SamplerPoolFinalize(c.args.NextPointer());
}

void nvnSamplerPoolRegisterSampler(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s32 id = c.args.NextS32();
    const u64 sampler = c.args.NextPointer();
    c.rec.RegisterSampler(pool, id, sampler);
}

/// Registers straight from a builder, without a sampler object (nxnative: FINAL FANTASY TACTICS
/// registers its samplers this way). The builder's parameters are decoded in place.
void nvnSamplerPoolRegisterSamplerBuilder(Call& c) {
    const u64 pool = c.args.NextPointer();
    const s32 id = c.args.NextS32();
    const u64 builder = c.args.NextPointer();
    u8 raw[SampParBytes];
    if (!c.mem.Read(builder + SbParams, raw, sizeof(raw))) {
        return;
    }
    VideoCommon::Nvn::SamplerDesc desc{};
    desc.min_filter = raw[0];
    desc.mag_filter = raw[1];
    desc.wrap_s = raw[2];
    desc.wrap_t = raw[3];
    desc.wrap_r = raw[4];
    desc.compare_mode = raw[5];
    desc.compare_func = raw[6];
    desc.reduction = raw[7];
    std::memcpy(&desc.lod_min, raw + 8, 4);
    std::memcpy(&desc.lod_max, raw + 12, 4);
    std::memcpy(&desc.lod_bias, raw + 16, 4);
    std::memcpy(&desc.max_anisotropy, raw + 20, 4);
    std::memcpy(desc.border, raw + 24, 16);
    c.rec.RegisterSamplerParams(pool, id, desc);
}

void nvnSamplerPoolGetMemoryPool(Call& c) {
    const auto info = c.rec.GetSamplerPool(c.args.NextPointer());
    c.ret.SetU64(info ? MemoryPoolVaOfSerial(info->pool_serial) : 0);
}

void nvnSamplerPoolGetMemoryOffset(Call& c) {
    const auto info = c.rec.GetSamplerPool(c.args.NextPointer());
    c.ret.SetU64(info ? info->pool_offset : 0);
}

void nvnSamplerPoolGetSize(Call& c) {
    const auto info = c.rec.GetSamplerPool(c.args.NextPointer());
    c.ret.SetU32(info ? info->count : 0);
}

} // namespace

void RegisterPoolFunctions(Registry& registry) {
    NVN_HLE(registry, nvnMemoryPoolBuilderSetDefaults);
    NVN_HLE(registry, nvnMemoryPoolBuilderSetDevice);
    NVN_HLE(registry, nvnMemoryPoolBuilderSetStorage);
    NVN_HLE(registry, nvnMemoryPoolBuilderSetFlags);
    NVN_HLE(registry, nvnMemoryPoolBuilderGetDevice);
    NVN_HLE(registry, nvnMemoryPoolBuilderGetMemory);
    NVN_HLE(registry, nvnMemoryPoolBuilderGetSize);
    NVN_HLE(registry, nvnMemoryPoolBuilderGetFlags);
    NVN_HLE(registry, nvnMemoryPoolInitialize);
    NVN_HLE(registry, nvnMemoryPoolFinalize);
    NVN_HLE(registry, nvnMemoryPoolMap);
    NVN_HLE(registry, nvnMemoryPoolGetBufferAddress);
    NVN_HLE(registry, nvnMemoryPoolGetSize);
    NVN_HLE(registry, nvnMemoryPoolGetFlags);
    NVN_HLE(registry, nvnMemoryPoolMapVirtual);
    NVN_HLE(registry, nvnMemoryPoolFlushMappedRange);
    NVN_HLE(registry, nvnMemoryPoolInvalidateMappedRange);
    registry.Add<&nvnPoolSetDebugLabel>("nvnMemoryPoolSetDebugLabel");

    NVN_HLE(registry, nvnTexturePoolInitialize);
    NVN_HLE(registry, nvnTexturePoolFinalize);
    NVN_HLE(registry, nvnTexturePoolRegisterTexture);
    NVN_HLE(registry, nvnTexturePoolRegisterImage);
    NVN_HLE(registry, nvnTexturePoolGetMemoryPool);
    NVN_HLE(registry, nvnTexturePoolGetMemoryOffset);
    NVN_HLE(registry, nvnTexturePoolGetSize);
    registry.Add<&nvnPoolSetDebugLabel>("nvnTexturePoolSetDebugLabel");

    NVN_HLE(registry, nvnSamplerPoolInitialize);
    NVN_HLE(registry, nvnSamplerPoolFinalize);
    NVN_HLE(registry, nvnSamplerPoolRegisterSampler);
    NVN_HLE(registry, nvnSamplerPoolRegisterSamplerBuilder);
    NVN_HLE(registry, nvnSamplerPoolGetMemoryPool);
    NVN_HLE(registry, nvnSamplerPoolGetMemoryOffset);
    NVN_HLE(registry, nvnSamplerPoolGetSize);
    registry.Add<&nvnPoolSetDebugLabel>("nvnSamplerPoolSetDebugLabel");
}

} // namespace Core::NvnHle::Hle
