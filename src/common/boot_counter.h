// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace Common {

/// A counter that belongs to one emulated boot, for "log the first N" gates and sampled trace
/// lines kept in function-local statics. A static would count over the whole process: in a GUI that
/// boots game after game, the second game would log nothing a warning of the first one already used
/// up. `ResetAll()` (called when a new boot starts) sets every counter back to zero.
///
///   static Common::BootCounter logged;
///   if (logged.First()) { LOG_WARNING(...); }
class BootCounter {
public:
    BootCounter() noexcept {
        next = Head().load(std::memory_order_relaxed);
        while (!Head().compare_exchange_weak(next, this, std::memory_order_release,
                                             std::memory_order_relaxed)) {
        }
    }
    BootCounter(const BootCounter&) = delete;
    BootCounter& operator=(const BootCounter&) = delete;

    /// Returns how many times Next/First ran before this call, then counts it.
    u64 Next() noexcept {
        return count.fetch_add(1, std::memory_order_relaxed);
    }
    /// True exactly once per boot.
    bool First() noexcept {
        return Next() == 0;
    }
    [[nodiscard]] u64 Count() const noexcept {
        return count.load(std::memory_order_relaxed);
    }

    /// Starts the counting of a new boot. Nodes live in function-local statics, i.e. until the
    /// process ends, so the list never dangles.
    static void ResetAll() noexcept {
        for (BootCounter* node = Head().load(std::memory_order_acquire); node != nullptr;
             node = node->next) {
            node->count.store(0, std::memory_order_relaxed);
        }
    }

private:
    static std::atomic<BootCounter*>& Head() noexcept {
        static std::atomic<BootCounter*> head{nullptr};
        return head;
    }

    std::atomic<u64> count{0};
    BootCounter* next{nullptr};
};

} // namespace Common
