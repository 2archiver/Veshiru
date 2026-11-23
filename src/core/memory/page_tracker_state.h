// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "core/memory/read_traps.h"

namespace Core::Memory {

/// What the page tracker and the device memory manager want from one guest page. A page whose
/// claim is cached has the RasterizerCachedMemory type (and, under fastmem, a write-protected or
/// fully protected host page), so guest accesses reach Memory's slow path.
struct PageClaim {
    bool device{}; ///< the device memory manager caches it (RasterizerMarkRegionCached)
    bool write{};  ///< an owner holds the page's contents and waits for its next write
    bool access{}; ///< an owner waits for its next read or write

    [[nodiscard]] bool Cached() const {
        return device || write || access;
    }
    bool operator==(const PageClaim&) const = default;
};

/// Where the tracker's decisions land: Memory applies page types and protections.
class PageClaimSink {
public:
    virtual ~PageClaimSink() = default;
    /// The pages [page_begin, page_end) follow `claim` from now on.
    virtual void ApplyClaim(u64 page_begin, u64 page_end, PageClaim claim) = 0;
    /// Whether the device memory manager caches this page now. Asked only for pages no owner
    /// covers, whose page type only the device memory manager sets.
    [[nodiscard]] virtual bool DeviceCached(u64 page) const = 0;
};

/// Per-page CPU write and access tracking of guest ranges that a host mirror holds (VTX-HLE's
/// device-local pool mirrors), on top of the rasterizer's page-type trapping. Every owner
/// registers a range; per page it records whether it holds the contents (armed), whether the
/// guest wrote since (dirty) and whether it waits for the next access (access trap, for lazy
/// GPU-to-guest write-back). The device memory manager's own cached state of each covered page
/// is kept beside it, so neither side releases a page the other still needs.
///
/// Not thread-safe: Memory serializes every member except Empty() and HasAccessTraps().
class PageTrackerState {
public:
    using AccessCallback = std::function<void(u64 va, u64 size)>;

    /// An armed range the guest accessed: the callback runs without the lock, then FinishAccess
    /// disarms what was armed up to `generation`.
    struct AccessHit {
        std::shared_ptr<const AccessCallback> callback;
        u64 owner{};
        u64 page_begin{};
        u64 page_end{};
        u64 generation{};
    };

    explicit PageTrackerState(PageClaimSink& sink_) : sink{sink_} {}
    ~PageTrackerState();

    PageTrackerState(const PageTrackerState&) = delete;
    PageTrackerState& operator=(const PageTrackerState&) = delete;

    /// No owner is registered (read without the lock).
    [[nodiscard]] bool Empty() const {
        return owner_count.load(std::memory_order_acquire) == 0;
    }
    /// Some page is armed for its next access (read without the lock).
    [[nodiscard]] bool HasAccessTraps() const {
        return access_pages.load(std::memory_order_acquire) != 0;
    }

    /// Registers the pages [page_begin, page_end). With `armed`, every page starts clean and
    /// write-armed; without, no page is tracked until Arm. Returns the owner id (never 0).
    u64 Register(u64 page_begin, u64 page_end, bool armed);
    void Unregister(u64 id);
    /// The owner holds the contents of these pages from now on: clean and write-armed.
    void Arm(u64 id, u64 page_begin, u64 page_end);
    /// Appends the merged [begin, end) page runs written since their last collect inside
    /// [page_begin, page_end) and makes them clean (armed) again.
    void CollectDirty(u64 id, u64 page_begin, u64 page_end, std::vector<std::pair<u64, u64>>& runs);
    [[nodiscard]] bool AnyDirty(u64 id, u64 page_begin, u64 page_end) const;
    /// The owner's write epoch: a counter that grows each time a page of the owner turns dirty (a
    /// guest write to an armed page, a remap of one). Read it without any lock; while it equals a
    /// value read before a full CollectDirty of the owner, no page of the owner is dirty. Null for
    /// an unknown id.
    [[nodiscard]] std::shared_ptr<const std::atomic<u64>> WriteEpoch(u64 id) const;

    void SetAccessCallback(u64 id, AccessCallback callback);
    void ArmAccess(u64 id, u64 page_begin, u64 page_end);
    void DisarmAccess(u64 id, u64 page_begin, u64 page_end);
    /// Appends the armed access traps in [page_begin, page_end). Returns whether the device path
    /// still has to see the access: a page there is untracked or device cached.
    bool CollectAccessHits(u64 page_begin, u64 page_end, std::vector<AccessHit>& hits) const;
    /// Disarms the hits' pages that were not armed again after the hit was collected.
    void FinishAccess(std::span<const AccessHit> hits);
    [[nodiscard]] bool AccessArmed(u64 page) const;

    /// A guest write to [page_begin, page_end) landed: armed pages turn dirty, and pages nobody
    /// waits on any more are released. Returns whether the device path still has to see it.
    bool OnWrite(u64 page_begin, u64 page_end);
    /// Whether the device path has to see an access to [page_begin, page_end).
    [[nodiscard]] bool NeedsDevicePath(u64 page_begin, u64 page_end) const;

    /// The device memory manager caches or releases [page_begin, page_end); untracked pages
    /// follow it directly, tracked ones keep their owners' claims.
    void SetDeviceCached(u64 page_begin, u64 page_end, bool cached);
    /// The guest mapped or unmapped [page_begin, page_end): tracked contents are gone (dirty) and
    /// the device's marks with them. Mapped pages get their claims back.
    void OnRemap(u64 page_begin, u64 page_end, bool mapped);

private:
    struct Owner {
        u64 id{};
        u64 page_begin{};
        u64 page_end{};
        std::vector<u64> armed;  ///< the owner holds the page's contents (bit per page)
        std::vector<u64> dirty;  ///< written since the owner took the contents
        std::vector<u64> device; ///< the device memory manager caches the page
        size_t dirty_count{};
        /// Counts every page that turned dirty (WriteEpoch).
        std::shared_ptr<std::atomic<u64>> epoch{std::make_shared<std::atomic<u64>>(0)};
        ReadTraps access;
        std::shared_ptr<const AccessCallback> callback;
    };

    /// Emits page runs of equal claims to the sink.
    class RunEmitter {
    public:
        explicit RunEmitter(PageClaimSink& sink_) : sink{sink_} {}
        ~RunEmitter() {
            Flush();
        }
        void Add(u64 page, PageClaim claim);
        void AddRun(u64 run_begin, u64 run_end, PageClaim claim);
        void Flush();

    private:
        PageClaimSink& sink;
        u64 begin{};
        u64 end{};
        PageClaim claim{};
    };

    Owner* Find(u64 id);
    const Owner* Find(u64 id) const;
    PageClaim ClaimOf(u64 page, bool& tracked) const;
    void RecomputeMaxPages();
    /// Fills `spans` with the sorted, merged parts of [page_begin, page_end) some owner covers.
    void TrackedSpans(u64 page_begin, u64 page_end);

    template <typename Func>
    void ForEachOwner(u64 page_begin, u64 page_end, Func&& func) const;

    PageClaimSink& sink;
    std::vector<std::unique_ptr<Owner>> owners; ///< sorted by page_begin
    u64 max_pages{};                            ///< longest owner range, bounds overlap lookups
    u64 next_id{1};
    std::vector<std::pair<u64, u64>> spans; ///< TrackedSpans' result, reused
    std::atomic<size_t> owner_count{};
    std::atomic<size_t> access_pages{};
};

} // namespace Core::Memory
