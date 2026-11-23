// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>
#include <functional>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::Nvn {

/// Access to guest virtual memory for the NVN HLE. Implemented by the bridge on top of the guest
/// process memory; tests provide a fake.
class GuestMemory {
public:
    virtual ~GuestMemory() = default;

    /// Copies `size` bytes from guest `va` into `dst`. Returns false when any byte is unmapped.
    virtual bool Read(u64 va, void* dst, size_t size) = 0;
    /// Copies `size` bytes from `src` to guest `va`. Returns false when any byte is unmapped.
    virtual bool Write(u64 va, const void* src, size_t size) = 0;
    /// Host pointer for guest `va`, nullptr when unmapped. The pointer is only valid for the
    /// contiguous mapping that contains `va`.
    ///
    /// Lifetime: the replay imports a CPU-accessible memory pool through this pointer once, at
    /// pool creation, and the GPU keeps reading and writing that host range until the pool is
    /// destroyed (after every submission that used it completed). The guest must keep the pool's
    /// storage mapped for that long; NVN requires it too (the storage belongs to the pool until
    /// MemoryPoolFinalize). If the guest unmaps or remaps the storage earlier, the GPU still
    /// accesses the host pages the pointer referred to at creation: it neither sees the new
    /// mapping nor faults, as long as the host backing is not released. Nothing re-validates the
    /// pointer later.
    virtual u8* Pointer(u64 va) = 0;

    /// Guest CPU write tracking of guest pages (4 KiB), for device-local copies of guest memory
    /// (pool mirrors). Optional: the default implementation tracks nothing (Track returns 0 and
    /// the replay keeps such ranges in guest memory).
    ///
    /// Track registers [va, va + size) (page-rounded) with no page tracked yet and returns an id
    /// (0: unavailable). Arm starts tracking pages (clean from now on: the caller holds their
    /// contents); TakeWritten appends the sorted, merged page-aligned runs [begin, end) of armed
    /// pages in [va, va + size) the guest CPU wrote since they were armed or last taken, and makes
    /// them clean; AnyWritten tells whether there is one, without changing anything. Writes the
    /// HLE makes through Write/Pointer are not seen (the write-back of GPU results; an HLE function
    /// that writes data into guest memory announces it with Recorder::NotifyHostWrite). Callable
    /// from any thread.
    virtual u64 Track(u64 va, u64 size) {
        static_cast<void>(va);
        static_cast<void>(size);
        return 0;
    }
    virtual void Untrack(u64 id) {
        static_cast<void>(id);
    }
    virtual void Arm(u64 id, u64 va, u64 size) {
        static_cast<void>(id);
        static_cast<void>(va);
        static_cast<void>(size);
    }
    virtual void TakeWritten(u64 id, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs) {
        static_cast<void>(id);
        static_cast<void>(va);
        static_cast<void>(size);
        static_cast<void>(runs);
    }
    [[nodiscard]] virtual bool AnyWritten(u64 id, u64 va, u64 size) {
        static_cast<void>(id);
        static_cast<void>(va);
        static_cast<void>(size);
        return false;
    }

    /// A lock-free counter of the pages of a Track registration that turned dirty: it grows with
    /// every guest write to an armed page. While it equals a value read before a TakeWritten over
    /// the whole registration, nothing was written since (AnyWritten would be false everywhere).
    /// Null: unknown, the caller asks TakeWritten / AnyWritten every time.
    [[nodiscard]] virtual std::shared_ptr<const std::atomic<u64>> WriteEpoch(u64 id) {
        static_cast<void>(id);
        return nullptr;
    }

    /// Whether SetAccessCallback/ArmAccessTrap/DisarmAccessTrap do anything (the guarded write-back
    /// merge needs them).
    [[nodiscard]] virtual bool SupportsAccessTraps() const {
        return false;
    }
    /// Access traps (any guest read or write) on pages of a Track registration: after
    /// ArmAccessTrap the guest's next access to each page calls the owner's callback with the
    /// armed page run it touched, before the access lands, on the accessing thread (it may block
    /// and must be idempotent); the page is disarmed afterwards. Optional, like Track.
    virtual void SetAccessCallback(u64 id, std::function<void(u64 va, u64 size)> callback) {
        static_cast<void>(id);
        static_cast<void>(callback);
    }
    virtual void ArmAccessTrap(u64 id, u64 va, u64 size) {
        static_cast<void>(id);
        static_cast<void>(va);
        static_cast<void>(size);
    }
    virtual void DisarmAccessTrap(u64 id, u64 va, u64 size) {
        static_cast<void>(id);
        static_cast<void>(va);
        static_cast<void>(size);
    }
};

/// Kinds of guest NVN objects that carry a GuestTag.
enum class ObjectKind : u32 {
    Invalid = 0,
    Device = 1,
    Queue = 2,
    CommandBuffer = 3,
    MemoryPool = 4,
    TexturePool = 5,
    SamplerPool = 6,
    Texture = 7,
    Sampler = 8,
    Buffer = 9,
    Program = 10,
    Sync = 11,
    Event = 12,
    Window = 13,
};

inline constexpr u32 GUEST_TAG_MAGIC = 0x564E4445;

/// 16-byte tag at offset 0 of every non-builder guest NVN object the HLE initializes. Games copy
/// objects by value; the tag travels with the copy and resolves to the same host object.
struct GuestTag {
    u32 magic;
    u32 kind; // ObjectKind
    u64 serial;
};
static_assert(sizeof(GuestTag) == 16);

} // namespace VideoCommon::Nvn
