// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Render-condition predicates computed ahead of their SetRenderEnableConditional (the recorder's
// PrefetchConditions, nvn_exec.cpp; tests: src/tests/nvn/condition_prefetch.cpp).
//
// A predicate computed at an earlier point of the command buffer is the one a later condition
// would compute at its own point exactly when nothing wrote either counter of its report pair in
// between and its scratch slot still holds it. The recorder reports every write that may reach
// pool bytes (`Invalidate`, or `Clear` when it cannot say which bytes), and the slot ring position
// (`DropRecycled`); a condition finds its predicate only if it compares the same pair (`Find`).

#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>

#include "common/common_types.h"

namespace Vulkan::Nvn {

struct Pool;

class ConditionPrefetch {
public:
    /// Bytes of one counter a condition compares (the first u64 of an NVNcounterData).
    static constexpr u64 COUNTER_BYTES = 8;
    /// Writes up to this many 8-byte granules are checked granule by granule; longer ones look at
    /// every predicate of the pool.
    static constexpr u64 GRANULE_CHECK_LIMIT = 64;

    struct Entry {
        u64 address; ///< guest address of the report pair (the condition's operand)
        u32 mode;    ///< NVN conditional render mode (0 equal, 1 not equal)
        u32 slot;    ///< byte offset of the predicate's scratch slot
        u64 position; ///< CounterEngine::SlotsAllocated() position of that slot
        const Pool* pool_a;
        u64 offset_a;
        const Pool* pool_b;
        u64 offset_b;
    };

    /// Adds the predicate of (address, mode), replacing an older one.
    void Add(const Entry& entry) {
        const u64 key = Key(entry.address, entry.mode);
        if (const auto it = entries.find(key); it != entries.end()) {
            Unindex(it->second);
            it->second = entry;
        } else {
            entries.emplace(key, entry);
        }
        Index(entry, +1);
    }

    /// The predicate of (address, mode) if it compares this very pair, else nullptr.
    [[nodiscard]] const Entry* Find(u64 address, u32 mode, const Pool* pool_a, u64 offset_a,
                                    const Pool* pool_b, u64 offset_b) const {
        const auto it = entries.find(Key(address, mode));
        if (it == entries.end()) {
            return nullptr;
        }
        const Entry& e = it->second;
        if (e.pool_a != pool_a || e.offset_a != offset_a || e.pool_b != pool_b ||
            e.offset_b != offset_b) {
            return nullptr;
        }
        return &e;
    }

    [[nodiscard]] bool Contains(u64 address, u32 mode) const {
        return entries.contains(Key(address, mode));
    }

    /// Bytes [offset, offset + size) of `pool` may have changed (size 0 counts as 1): drops every
    /// predicate whose pair reads one of them.
    void Invalidate(const Pool& pool, u64 offset, u64 size) {
        if (entries.empty()) {
            return;
        }
        const auto pool_it = per_pool.find(&pool);
        if (pool_it == per_pool.end()) {
            return;
        }
        const u64 end = offset + (size == 0 ? 1 : size);
        const u64 first = offset / COUNTER_BYTES;
        const u64 last = (end - 1) / COUNTER_BYTES;
        if (last - first < GRANULE_CHECK_LIMIT) {
            bool hit = false;
            for (u64 g = first; g <= last && !hit; ++g) {
                hit = granules.contains(GranuleKey{&pool, g});
            }
            if (!hit) {
                return;
            }
        }
        const auto touches = [&](const Pool* p, u64 at) {
            return p == &pool && at < end && offset < at + COUNTER_BYTES;
        };
        EraseIf([&](const Entry& e) {
            return touches(e.pool_a, e.offset_a) || touches(e.pool_b, e.offset_b);
        });
    }

    /// The ring handed out `allocated` slots so far, of `ring_slots`: drops the predicates whose
    /// slot it has come back to.
    void DropRecycled(u64 allocated, u64 ring_slots) {
        EraseIf([&](const Entry& e) { return allocated > e.position + ring_slots; });
    }

    void Clear() {
        entries.clear();
        per_pool.clear();
        granules.clear();
    }

    [[nodiscard]] bool Empty() const noexcept {
        return entries.empty();
    }
    [[nodiscard]] size_t Size() const noexcept {
        return entries.size();
    }

private:
    struct GranuleKey {
        const Pool* pool;
        u64 granule;
        bool operator==(const GranuleKey&) const = default;
    };
    struct GranuleHash {
        size_t operator()(const GranuleKey& k) const noexcept {
            return std::hash<const void*>{}(k.pool) ^
                   (std::hash<u64>{}(k.granule) * 0x9E3779B97F4A7C15ull);
        }
    };

    [[nodiscard]] static u64 Key(u64 address, u32 mode) noexcept {
        return (address << 1) | (mode & 1);
    }

    void IndexCounter(const Pool* pool, u64 offset, int delta) {
        for (u64 g = offset / COUNTER_BYTES; g <= (offset + COUNTER_BYTES - 1) / COUNTER_BYTES;
             ++g) {
            const GranuleKey key{pool, g};
            if (delta > 0) {
                ++granules[key];
            } else if (const auto it = granules.find(key);
                       it != granules.end() && --it->second == 0) {
                granules.erase(it);
            }
        }
        if (delta > 0) {
            ++per_pool[pool];
        } else if (const auto it = per_pool.find(pool); it != per_pool.end() && --it->second == 0) {
            per_pool.erase(it);
        }
    }
    void Index(const Entry& e, int delta) {
        IndexCounter(e.pool_a, e.offset_a, delta);
        IndexCounter(e.pool_b, e.offset_b, delta);
    }
    void Unindex(const Entry& e) {
        Index(e, -1);
    }
    template <typename Predicate>
    void EraseIf(Predicate&& predicate) {
        for (auto it = entries.begin(); it != entries.end();) {
            if (predicate(it->second)) {
                Unindex(it->second);
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::unordered_map<u64, Entry> entries;
    /// Counters indexed per pool and per 8-byte granule (both counters of every predicate): a
    /// write to a pool without predicates, or a short write that no counter overlaps, costs a
    /// lookup or a few.
    std::unordered_map<const Pool*, u32> per_pool;
    std::unordered_map<GranuleKey, u32, GranuleHash> granules;
};

} // namespace Vulkan::Nvn
