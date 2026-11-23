// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "core/memory.h"

namespace Core::Memory {

/// CPU write and access tracking of guest ranges a host mirror holds (VTX-HLE's device-local pool
/// mirrors and lazy GPU-to-guest write-back), per 4 KiB page of one process' memory. It reuses
/// the rasterizer's trapping: a tracked page gets the RasterizerCachedMemory page type (and under
/// fastmem a write-protected host page), exactly like a page the buffer or texture cache tracks,
/// so the guest's first write to it reaches Memory's slow path, which marks it dirty and releases
/// it again; later writes run at full speed until CollectDirty arms the page again. Pages the
/// device memory manager also caches stay cached for it.
///
/// Writes that count: JIT stores (fastmem or page table), Memory::Write*, WriteExclusive*,
/// WriteBlock, WriteBlockUnsafe, ZeroBlock, CopyBlock, data cache clean/flush, native-code write
/// faults, and a raw-pointer writer's Memory::NotifyHostWrite (the kernel's IPC copies). Writes
/// that do not: through raw host pointers that do not tell (GetPointer, GetPointerSilent, the
/// device memory backing) and WriteBlockUntracked; the HLE writes back GPU results that way.
///
/// Every member is thread-safe; addresses are guest virtual addresses, rounded out to pages.
class PageTracker {
public:
    using AccessCallback = std::function<void(u64 va, u64 size)>;

    /// Owned by Memory: one per process memory (Memory::PageTracker()).
    explicit PageTracker(Memory::Impl& impl_) : impl{impl_} {}
    PageTracker(const PageTracker&) = delete;
    PageTracker& operator=(const PageTracker&) = delete;

    /// Registers [va, va + size). With `armed` every page starts clean and write-armed (the
    /// caller holds its contents); without, no page is tracked until Arm. Returns the owner id,
    /// 0 when the range is empty or outside the address space, or the tracker is off
    /// (Memory::EnablePageTracker).
    u64 Register(u64 va, u64 size, bool armed = true);
    /// Stops tracking the owner's pages and drops its access traps (without calling back).
    void Unregister(u64 id);
    /// The owner holds the contents of these pages from now on: clean and write-armed.
    void Arm(u64 id, u64 va, u64 size);
    /// Appends the page-aligned [begin, end) VA runs written inside [va, va + size) since they
    /// were last armed or collected, sorted and merged, and arms them again. Pages outside the
    /// sub-range keep their state.
    void CollectDirty(u64 id, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs);
    /// Whether CollectDirty would report anything for the sub-range (clears nothing).
    [[nodiscard]] bool AnyDirty(u64 id, u64 va, u64 size);
    /// PageTrackerState::WriteEpoch: a lock-free counter of the pages that turned dirty.
    [[nodiscard]] std::shared_ptr<const std::atomic<u64>> WriteEpoch(u64 id);

    /// Called on the guest thread whose read or write hits an access-armed page, before that
    /// access lands, with the armed page run of this owner it touched. No tracker lock is held:
    /// it may block, call back into the tracker (e.g. DisarmAccessTrap on a larger range it
    /// flushed) and run on several threads at once, so it must be idempotent.
    void SetAccessCallback(u64 id, AccessCallback callback);
    /// The next guest access to each page calls the owner's callback, after which the page is
    /// disarmed (unless it was armed again meanwhile). Write tracking is unaffected.
    void ArmAccessTrap(u64 id, u64 va, u64 size);
    void DisarmAccessTrap(u64 id, u64 va, u64 size);

private:
    Memory::Impl& impl;
};

} // namespace Core::Memory
