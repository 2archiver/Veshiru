// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <mutex>
#include <unordered_map>

#include <fmt/format.h>

#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// Event family: NVNeventBuilder (kept in place, nxnative `ev_builder_t`) and NVNevent. An event is a
// 32-bit value in a memory pool (the builder's storage); the Recorder keeps its address, the game's
// CPU reads and writes it directly and the GPU signals it through `nvnCommandBufferSignalEvent`.
// There is no nvnEventWait in names.txt (NVN waits on events through command buffers and by polling
// the value), so no body blocks here.
namespace Core::NvnHle::Hle {
namespace {

// NVNeventBuilder in place (nxnative ev_builder_t {const pool_t* pool; int64_t off;}, 0x10 of the
// builder's 0x20 bytes).
constexpr u64 EbPool = 0x00;
constexpr u64 EbOffset = 0x08;
constexpr u64 EbBytes = 0x10;

constexpr u32 SignalModeAdd = 1;

/// The storage an event was initialized with, by serial: the Recorder keeps the event's address but
/// not the pool object it came from, and NVN returns that pointer (`nvnEventGetMemoryPool`).
/// Written at initialize and finalize only. [INFERENCE] Obsolete once the Recorder exposes it.
struct EventStorage {
    u64 pool;
    s64 offset;
};

class StorageTable {
public:
    void Set(u64 serial, const EventStorage& storage) {
        std::lock_guard lock{mutex};
        entries[serial] = storage;
    }
    void Erase(u64 serial) {
        std::lock_guard lock{mutex};
        entries.erase(serial);
    }
    bool Find(u64 serial, EventStorage& out) {
        std::lock_guard lock{mutex};
        const auto it = entries.find(serial);
        if (it == entries.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

private:
    std::mutex mutex;
    std::unordered_map<u64, EventStorage> entries;
};

StorageTable& Storage() {
    // Leaked on purpose: a late guest call must not meet a destroyed table at process exit.
    static StorageTable* const table = new StorageTable;
    return *table;
}

/// Serial of the tagged object at `va` (0 when untagged).
u64 TagSerial(Call& c, u64 va) {
    const auto tag = c.Load<VideoCommon::Nvn::GuestTag>(va);
    return tag.magic == VideoCommon::Nvn::GUEST_TAG_MAGIC &&
                   tag.kind == static_cast<u32>(VideoCommon::Nvn::ObjectKind::Event)
               ? tag.serial
               : 0;
}

void nvnEventBuilderSetDefaults(Call& c) {
    c.Zero(c.args.NextPointer(), EbBytes);
}

void nvnEventBuilderSetStorage(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    c.Store<u64>(builder + EbPool, pool);
    c.Store<s64>(builder + EbOffset, offset);
}

/// No prototype in nvn.h; the pointer and offset the builder keeps (nxnative NOPROTO bodies).
void nvnEventBuilderGetMemoryPool(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + EbPool));
}

void nvnEventBuilderGetMemoryOffset(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + EbOffset));
}

void nvnEventInitialize(Call& c) {
    const u64 event = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const u64 pool = c.Load<u64>(builder + EbPool);
    const s64 offset = c.Load<s64>(builder + EbOffset);
    if (!c.rec.EventInitialize(event, pool, offset)) {
        c.ret.SetBool(false);
        return;
    }
    if (const u64 serial = TagSerial(c, event)) {
        Storage().Set(serial, EventStorage{pool, offset});
    }
    c.ret.SetBool(true);
}

void nvnEventFinalize(Call& c) {
    const u64 event = c.args.NextPointer();
    if (const u64 serial = TagSerial(c, event)) {
        Storage().Erase(serial);
    }
    c.rec.EventFinalize(event);
}

void nvnEventGetValue(Call& c) {
    const u64 event = c.args.NextPointer();
    const auto value = c.rec.EventGetValue(event);
    if (Runtime::TimelineEnabled()) {
        Runtime::TimelineNote("guest_eventget", 0,
                              fmt::format("event={:#x} value={}", event, value.value_or(0)));
    }
    c.ret.SetU32(value ? *value : 0);
}

/// CPU-side signal: mode 0 (WRITE) stores `value`, mode 1 (ADD) adds it. The value changes in
/// guest memory, where the game polls it; a guest thread sleeping on a condition that reads it is
/// woken to re-check.
void nvnEventSignal(Call& c) {
    const u64 event = c.args.NextPointer();
    const u32 mode = c.args.NextU32();
    const u32 value = c.args.NextU32();
    if (Runtime::TimelineEnabled()) {
        // The enq_eventcpu line that follows names the event's memory address.
        Runtime::TimelineNote("guest_eventsignal", 0,
                              fmt::format("event={:#x} mode={} value={}", event, mode, value));
    }
    c.rec.EventSignal(event, mode == SignalModeAdd ? 1 : 0, value);
    c.runtime.WakeWaiters();
}

/// No prototype in nvn.h; returns the storage the event was initialized with.
void nvnEventGetMemoryPool(Call& c) {
    EventStorage storage{};
    const u64 serial = TagSerial(c, c.args.NextPointer());
    c.ret.SetU64(serial != 0 && Storage().Find(serial, storage) ? storage.pool : 0);
}

void nvnEventGetMemoryOffset(Call& c) {
    EventStorage storage{};
    const u64 serial = TagSerial(c, c.args.NextPointer());
    c.ret.SetU64(serial != 0 && Storage().Find(serial, storage)
                     ? static_cast<u64>(storage.offset)
                     : 0);
}

} // namespace

void RegisterEventFunctions(Registry& registry) {
    NVN_HLE(registry, nvnEventBuilderSetDefaults);
    NVN_HLE(registry, nvnEventBuilderSetStorage);
    NVN_HLE(registry, nvnEventBuilderGetMemoryPool);
    NVN_HLE(registry, nvnEventBuilderGetMemoryOffset);
    NVN_HLE(registry, nvnEventInitialize);
    NVN_HLE(registry, nvnEventFinalize);
    NVN_HLE(registry, nvnEventGetValue);
    NVN_HLE(registry, nvnEventSignal);
    NVN_HLE(registry, nvnEventGetMemoryPool);
    NVN_HLE(registry, nvnEventGetMemoryOffset);
}

} // namespace Core::NvnHle::Hle
