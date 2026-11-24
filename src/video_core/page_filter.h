// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>

#include "common/common_types.h"

namespace VideoCommon {

/// The 4 KiB device pages a cache has ever registered objects on. Bits are never cleared, so a
/// clear bit proves the cache holds nothing there and an invalidation of those pages does nothing.
/// Ranges past the 36-bit space aren't tracked: Any reports them as possibly holding objects
/// (such as stand-in addresses of images without CPU backing).
class PageFilter {
public:
    PageFilter() : words{std::make_unique<std::atomic<u64>[]>(NumWords)} {}

    /// Records that the cache holds objects on [address, address + size).
    void Mark(u64 address, u64 size) {
        if (size == 0) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumWords * 64) {
            return;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_relaxed);
            }
            page += count;
        }
    }

    /// Whether the cache may hold objects on [address, address + size).
    [[nodiscard]] bool Any(u64 address, u64 size) const {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumWords * 64) {
            return true;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if (words[page / 64].load(std::memory_order_relaxed) & mask) {
                return true;
            }
            page += count;
        }
        return false;
    }

    /// Atomically tests and clears the single 4 KiB page containing `address`.
    /// Ranges past the 36-bit space return true so callers conservatively inspect them.
    [[nodiscard]] bool Take(u64 address) {
        const u64 page = address >> PageBits;
        if (page >= NumWords * 64) {
            return true;
        }
        const u64 bit = u64{1} << (page % 64);
        auto& word = words[page / 64];
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            return false;
        }
        return (word.fetch_and(~bit, std::memory_order_relaxed) & bit) != 0;
    }

    /// Clears every tracked page bit (called when guest memory is reset for a new session).
    void Clear() {
        for (u64 i = 0; i < NumWords; ++i) {
            words[i].store(0, std::memory_order_relaxed);
        }
    }

private:
    static constexpr u64 PageBits = 12;
    static constexpr u64 NumWords = (u64{1} << (36 - PageBits)) / 64; // 2 MiB of bits
    std::unique_ptr<std::atomic<u64>[]> words;
};

/// 4 KiB device pages for which guest CPU writes were collected since the page was last taken.
inline PageFilter g_collected_write_pages{};

/// 4 KiB device pages written by GPU downloads or engine memory writes since last taken.
inline PageFilter g_buffer_write_pages{};

} // namespace VideoCommon
