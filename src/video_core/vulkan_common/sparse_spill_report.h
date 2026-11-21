// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"

namespace Vulkan {

/// Counts the sparse texture tiles that could not be placed in device-local memory and were
/// tried on another heap instead (MemoryAllocator::AllocateSparseMemory), and decides when to say
/// so. A tile outside video memory is sampled over PCIe; whether, and by how much, that costs
/// frame time in a given scene has not been measured, so this only makes the event visible. It
/// changes nothing about where a tile goes.
///
/// A spill is reported at most once per interval: the first one at once, later ones together at
/// the first spill after the interval has passed. A report therefore never comes from a timer;
/// what happened after the last report is in Pending().
///
/// Not thread safe; the allocator holds a mutex around it.
class SparseSpillReport {
public:
    static constexpr s64 IntervalMs = 10'000;

    struct Counters {
        /// Tiles placed outside device-local memory.
        u64 tiles{};
        u64 bytes{};
        /// Tiles no other heap took either: the allocation failed, as it did before 2.1.0.
        u64 failed{};

        bool operator==(const Counters&) const = default;
    };

    struct Report {
        /// Since the previous report.
        Counters interval;
        /// Since the allocator was created.
        Counters total;

        bool operator==(const Report&) const = default;
    };

    /// One tile that device-local memory had no room for. `placed` is whether another heap took
    /// it. Returns the report to log now, if one is due.
    constexpr std::optional<Report> Record(s64 now_ms, u64 bytes, bool placed) {
        if (placed) {
            ++pending.tiles;
            pending.bytes += bytes;
            ++total.tiles;
            total.bytes += bytes;
        } else {
            ++pending.failed;
            ++total.failed;
        }
        if (has_reported && now_ms - last_report_ms < IntervalMs) {
            return std::nullopt;
        }
        has_reported = true;
        last_report_ms = now_ms;
        const Report report{.interval = pending, .total = total};
        pending = {};
        return report;
    }

    /// What was recorded since the last report (for a final line when the allocator goes away).
    constexpr const Counters& Pending() const {
        return pending;
    }

    constexpr const Counters& Total() const {
        return total;
    }

private:
    Counters pending{};
    Counters total{};
    s64 last_report_ms{};
    bool has_reported{};
};

} // namespace Vulkan
