// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Vulkan::Nvn {

/// Counters of every MirrorLineGuard of a context (the replay heartbeat, DescribeState).
struct LineGuardStats {
    std::atomic<u64> line_bytes{};   ///< bytes of merge ranges recorded in line mode
    std::atomic<u64> exact_bytes{};  ///< ... recorded as the exact word merge (hot pages, exact pools)
    std::atomic<u64> armed_pages{};  ///< access traps armed
    std::atomic<u64> hot_pages{};    ///< pages the guest touched (they use the exact merge from then on)
    std::atomic<u64> waits{};        ///< touches that had to wait for a merge in flight
    std::atomic<u64> wait_us{};      ///< ... for this long in total
    std::atomic<u64> timeouts{};     ///< ... and that gave up waiting
    std::atomic<u64> exact_pools{};  ///< pools that went exact for good (too many hot pages, a timeout)
};

/// Guards the line-mode write-back merge of one pool mirror (nvn_wb_merge.comp, DESIGN.md "Merge
/// line mode"). A line merge stores whole units of 8 or 32 guest words, the words the GPU did not
/// change as the guest holds them an instant before. A guest CPU store to one of those words that
/// lands inside that instant would be lost, so a line merge is only allowed on pages the guest has
/// not touched since the guard armed an access trap on them (PageTracker::ArmAccessTrap: the first
/// guest read or write of a page calls Touched before it lands):
///
///  - Record (replay thread, when it records a merge of a pool range) splits the range into runs.
///    Runs of untouched pages are `lines` runs: their pages get armed (once; `arm`) and the range
///    counts as in flight until the returned token dies (InFlight::keepalive: the GPU is done).
///    Touched ("hot") pages form runs of the exact word merge, which stores only the words the GPU
///    changed and so never loses a CPU store to another word.
///  - Touched (the guest thread that hit the trap, before its access lands) marks the pages hot,
///    then waits until no line merge covering them is in flight: from then on every merge of those
///    pages is exact (Record sees them hot, both under the same mutex), and the ones recorded
///    before have finished storing. The wait is bounded.
///  - After MAX_HOT_PAGES hot pages or a wait that timed out the pool is exact for good (Record
///    returns `disarm_all` once: the caller drops the traps).
///
/// Everything is thread-safe.
class MirrorLineGuard : public std::enable_shared_from_this<MirrorLineGuard> {
public:
    static constexpr u64 PAGE = 4096;
    /// A pool whose guest touches more pages than this while armed is merged exactly from then on.
    static constexpr u32 MAX_HOT_PAGES = 16;
    static constexpr std::chrono::milliseconds DEFAULT_WAIT_LIMIT{250};

    struct Run {
        u64 begin{};
        u64 end{};
        bool lines{};
        friend bool operator==(const Run&, const Run&) = default;
    };

    /// Held in InFlight::keepalive while the GPU may still be merging a line run.
    class Token {
    public:
        Token(std::shared_ptr<MirrorLineGuard> guard_, std::vector<std::pair<u64, u64>> pages_);
        ~Token();
        Token(const Token&) = delete;
        Token& operator=(const Token&) = delete;

    private:
        friend class MirrorLineGuard;
        std::shared_ptr<MirrorLineGuard> guard;
        std::vector<std::pair<u64, u64>> pages; ///< [first, last] page runs
    };

    struct Recording {
        std::vector<Run> runs;
        /// Page runs [first, last] (pool offsets / PAGE) to arm an access trap on now.
        std::vector<std::pair<u64, u64>> arm;
        /// Null when no run is a line run.
        std::shared_ptr<Token> token;
        /// The pool just went exact for good: drop every trap of it.
        bool disarm_all{false};
    };

    /// `replay_thread`: the thread that records merges; a touch on it never waits (it would wait
    /// for work it has not submitted yet).
    MirrorLineGuard(u64 pool_size, std::shared_ptr<LineGuardStats> stats_,
                    std::thread::id replay_thread_ = {},
                    std::chrono::milliseconds wait_limit_ = DEFAULT_WAIT_LIMIT);

    /// Plans the merge of pool bytes [begin, end). `lines` false: everything is an exact run (line
    /// mode is off for this call).
    Recording Record(u64 begin, u64 end, bool lines = true);

    /// The guest touched the pool bytes [begin, end) (the access callback). Returns false when it
    /// gave up waiting for a merge in flight.
    bool Touched(u64 begin, u64 end);

    [[nodiscard]] bool ExactOnly() const;
    [[nodiscard]] u32 HotPages() const;
    [[nodiscard]] bool HotPage(u64 page) const;

private:
    [[nodiscard]] bool Bit(const std::vector<u64>& bits, u64 page) const {
        return ((bits[page / 64] >> (page % 64)) & 1) != 0;
    }
    void SetBit(std::vector<u64>& bits, u64 page) {
        bits[page / 64] |= u64{1} << (page % 64);
    }
    void ClearBit(std::vector<u64>& bits, u64 page) {
        bits[page / 64] &= ~(u64{1} << (page % 64));
    }
    /// A line merge in flight covers a page of [first, last].
    [[nodiscard]] bool InFlight(u64 first, u64 last) const;
    void Release(const Token& token);

    const u64 pages;
    const std::shared_ptr<LineGuardStats> stats;
    const std::thread::id replay_thread;
    const std::chrono::milliseconds wait_limit;
    mutable std::mutex mutex;
    std::condition_variable idle;
    std::vector<u64> hot;   ///< one bit per page
    std::vector<u64> armed; ///< one bit per page: an access trap sits on it
    u32 hot_pages{0};
    bool exact_only{false};
    bool disarm_pending{false};
    std::vector<const Token*> live;
};

} // namespace Vulkan::Nvn
