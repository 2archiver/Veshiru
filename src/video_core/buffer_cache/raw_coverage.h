// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "shader_recompiler/raw_coverage_layout.h"
#include "video_core/buffer_cache/raw_global_table.h"

namespace VideoCommon {

/// Raw global memory coverage diagnostic (EDEN_NX2_RAW_COVERAGE=1; measurement only, the picture
/// is unchanged). The programs of both GPU paths report every raw global access their address
/// table does not hold (raw_coverage_layout.h: the legacy path's table storage buffer, the NVN
/// HLE path's driver bank); the host reads the miss ring of each 10 s window back and classifies
/// the misses (ClassifyRawMiss):
/// - null:       address below 4 KiB (a null pointer plus a field offset; at most
///               Shader::RawCoverage::NULL_RECORDS of them are recorded and counted per window);
/// - unmapped:   no GPU mapping (legacy: the GPU page table; NVN HLE: no memory pool holds it; on
///               hardware an MMU fault: the pointer itself is wrong, e.g. data the guest wrote
///               that we did not bring over);
/// - past-end:   mapped, within 64 KiB after the end of a table range (that range is too short);
/// - mapped-gap: mapped, but no table range holds it (our table misses a page: our bug; on NVN
///               HLE the kind names the pool and why the table leaves it out);
/// - late:       the table holds it by the time the window is read (it came in after the access,
///               e.g. a guest memory import that was still queued).
[[nodiscard]] inline bool RawCoverageEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_RAW_COVERAGE");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

static_assert(Shader::RawCoverage::TABLE_RING_WORD + 2 <= RAW_GLOBAL_TABLE_HEADER_BYTES / 4,
              "the ring address lives in the table header");

inline constexpr size_t RAW_COVERAGE_RING_BYTES{Shader::RawCoverage::RING_HEADER_BYTES +
                                                size_t{Shader::RawCoverage::RING_ENTRIES} *
                                                    Shader::RawCoverage::RECORD_BYTES};
inline constexpr u64 RAW_COVERAGE_WINDOW_SECONDS{10};
inline constexpr u64 RAW_COVERAGE_PAGE_SIZE{4096};
inline constexpr u64 RAW_COVERAGE_NULL_LIMIT{Shader::RawCoverage::NULL_ADDRESS_LIMIT};
inline constexpr u64 RAW_COVERAGE_PAST_END_REACH{64 * 1024};

/// The ring device address in the table header (words TABLE_RING_WORD..+1).
[[nodiscard]] inline u64 RawTableRingAddress(std::span<const u8> table) {
    u64 address{};
    if (table.size() >= RAW_GLOBAL_TABLE_HEADER_BYTES) {
        std::memcpy(&address, table.data() + Shader::RawCoverage::TABLE_RING_WORD * 4,
                    sizeof(address));
    }
    return address;
}

inline void SetRawTableRingAddress(std::span<u8> table, u64 address) {
    if (table.size() >= RAW_GLOBAL_TABLE_HEADER_BYTES) {
        std::memcpy(table.data() + Shader::RawCoverage::TABLE_RING_WORD * 4, &address,
                    sizeof(address));
    }
}

/// The entries of a table built by BuildRawGlobalTable.
[[nodiscard]] inline std::vector<RawGlobalRange> RawTableRanges(std::span<const u8> table) {
    if (table.size() < RAW_GLOBAL_TABLE_HEADER_BYTES) {
        return {};
    }
    u32 count{};
    std::memcpy(&count, table.data(), sizeof(count));
    const size_t stored{(table.size() - RAW_GLOBAL_TABLE_HEADER_BYTES) / sizeof(RawGlobalTableEntry)};
    std::vector<RawGlobalRange> ranges((std::min)(size_t{count}, stored));
    for (size_t index = 0; index < ranges.size(); ++index) {
        RawGlobalTableEntry entry;
        std::memcpy(&entry,
                    table.data() + RAW_GLOBAL_TABLE_HEADER_BYTES + index * sizeof(entry),
                    sizeof(entry));
        ranges[index] = {entry.guest, entry.device, entry.size};
    }
    return ranges;
}

struct RawCoverageMiss {
    u64 guest;
    u64 program;
    u32 op;
    u32 stage;
    u32 present;
};

struct RawCoverageRing {
    std::vector<RawCoverageMiss> misses;
    /// Misses the shaders counted. Once the ring is `full` they stop counting (a miss then only
    /// reads the counter), so the count is a lower bound.
    u64 total{};
    u64 dropped{}; ///< counted, but past the end of the ring (not recorded)
    bool full{};
};

/// Decodes a ring the GPU is done with (raw_coverage_layout.h).
inline void ParseRawCoverageRing(std::span<const u8> bytes, RawCoverageRing& out) {
    namespace Cov = Shader::RawCoverage;
    out.misses.clear();
    out.total = 0;
    out.dropped = 0;
    out.full = false;
    if (bytes.size() < Cov::RING_HEADER_BYTES) {
        return;
    }
    u32 counter{};
    std::memcpy(&counter, bytes.data() + Cov::RING_COUNTER_WORD * 4, sizeof(counter));
    const size_t capacity{(std::min<size_t>)(
        Cov::RING_ENTRIES, (bytes.size() - Cov::RING_HEADER_BYTES) / Cov::RECORD_BYTES)};
    const size_t recorded{(std::min)(size_t{counter}, capacity)};
    out.total = counter;
    out.dropped = counter - recorded;
    out.full = counter >= Cov::RING_ENTRIES;
    out.misses.reserve(recorded);
    for (size_t slot = 0; slot < recorded; ++slot) {
        std::array<u32, Cov::RECORD_BYTES / 4> words;
        std::memcpy(words.data(), bytes.data() + Cov::RING_HEADER_BYTES + slot * Cov::RECORD_BYTES,
                    sizeof(words));
        out.misses.push_back(RawCoverageMiss{
            .guest = words[0] | (u64{words[1]} << 32),
            .program = words[2] | (u64{words[3]} << 32),
            .op = words[Cov::RECORD_KIND_WORD] & ((1u << Cov::RECORD_STAGE_SHIFT) - 1),
            .stage = words[Cov::RECORD_KIND_WORD] >> Cov::RECORD_STAGE_SHIFT,
            .present = words[Cov::RECORD_PRESENT_WORD],
        });
    }
}

enum class RawMissClass : u8 { MappedGap, PastEnd, Unmapped, Null, Late };
inline constexpr size_t NUM_RAW_MISS_CLASSES{5};

struct RawMissVerdict {
    RawMissClass kind{};
    /// PastEnd: the range the address lies after, and how far past its end. MappedGap with
    /// `pool`: the pool that holds the address.
    u64 range_guest{};
    u64 range_size{};
    u64 past{};
    /// MappedGap (NVN HLE): why the table leaves the pool out (a static string), null otherwise.
    const char* pool{};
};

/// Classifies a missed guest address against the current table `ranges` (flattened) and the GPU
/// page table (`is_mapped(address)`: a mapping holds the byte).
template <typename IsMapped>
[[nodiscard]] RawMissVerdict ClassifyRawMiss(u64 guest, std::span<const RawGlobalRange> ranges,
                                             IsMapped&& is_mapped) {
    if (guest < RAW_COVERAGE_NULL_LIMIT) {
        return {.kind = RawMissClass::Null};
    }
    if (TranslateRawGlobalAddress(ranges, guest) != 0) {
        return {.kind = RawMissClass::Late};
    }
    if (!is_mapped(guest)) {
        return {.kind = RawMissClass::Unmapped};
    }
    const auto next = std::upper_bound(ranges.begin(), ranges.end(), guest,
                                       [](u64 value, const RawGlobalRange& range) {
                                           return value < range.guest;
                                       });
    if (next != ranges.begin()) {
        const RawGlobalRange& range{*std::prev(next)};
        const u64 end{range.guest + range.size};
        if (guest >= end && guest - end < RAW_COVERAGE_PAST_END_REACH) {
            return {.kind = RawMissClass::PastEnd,
                    .range_guest = range.guest,
                    .range_size = range.size,
                    .past = guest - end};
        }
    }
    return {.kind = RawMissClass::MappedGap};
}

[[nodiscard]] inline std::string_view RawMissClassName(RawMissClass kind) {
    switch (kind) {
    case RawMissClass::MappedGap:
        return "mapped-gap";
    case RawMissClass::PastEnd:
        return "past-end";
    case RawMissClass::Unmapped:
        return "unmapped";
    case RawMissClass::Null:
        return "null";
    case RawMissClass::Late:
        return "late";
    }
    return "?";
}

[[nodiscard]] inline std::string RawMissKindText(const RawMissVerdict& verdict) {
    if (verdict.kind == RawMissClass::PastEnd) {
        return fmt::format("past-end(+{:#x} of {:#x}+{:#x})", verdict.past, verdict.range_guest,
                           verdict.range_size);
    }
    if (verdict.kind == RawMissClass::MappedGap && verdict.pool != nullptr) {
        return fmt::format("mapped-gap({} pool {:#x}+{:#x})", verdict.pool, verdict.range_guest,
                           verdict.range_size);
    }
    return std::string{RawMissClassName(verdict.kind)};
}

[[nodiscard]] inline std::string RawCoverageOpName(u32 op) {
    using Shader::RawCoverage::Op;
    switch (static_cast<Op>(op)) {
    case Op::Load32:
        return "ld32";
    case Op::Load64:
        return "ld64";
    case Op::Load128:
        return "ld128";
    case Op::Store32:
        return "st32";
    case Op::Store64:
        return "st64";
    case Op::Store128:
        return "st128";
    case Op::StoreSubword:
        return "stsub";
    case Op::Atomic:
        return "atom";
    }
    return fmt::format("op{}", op);
}

/// Shader::Stage values.
[[nodiscard]] inline std::string RawCoverageStageName(u32 stage) {
    static constexpr std::array names{"vs", "tcs", "tes", "gs", "fs", "cs", "vsa", "task", "mesh"};
    return stage < names.size() ? std::string{names[stage]} : fmt::format("stage{}", stage);
}

/// Totals and log lines of the diagnostic: one summary line per window, one line the first time a
/// (program, page) misses, the run totals so far after every window and the pages that missed
/// most every TOP_EVERY_WINDOWS windows and at exit. Every line of a report made for a named GPU
/// path ends with ` path=<name>` (the legacy path's lines have none).
class RawCoverageReport {
public:
    static constexpr size_t MAX_NEW_LINES_PER_WINDOW{64};
    static constexpr size_t MAX_NEW_LINES_PER_RUN{1024};
    static constexpr size_t TOP_ROWS{20};
    static constexpr u64 TOP_EVERY_WINDOWS{6};

    RawCoverageReport() = default;
    explicit RawCoverageReport(std::string_view path)
        : path_field{path.empty() ? std::string{} : fmt::format(" path={}", path)} {}

    /// Adds a window's misses; `classify(address)` gives the RawMissVerdict of an address (the
    /// lowest missed address of each page is classified). Returns the lines to log.
    template <typename Classify>
    std::vector<std::string> AddWindow(const RawCoverageRing& ring, u64 dispatches,
                                       u64 window_seconds, Classify&& classify) {
        struct PageMisses {
            u64 lowest{~u64{0}};
            u64 count{};
            u32 first_present{~0u};
            u32 op{};
            u32 stage{};
        };
        std::map<std::pair<u64, u64>, PageMisses> pages; // (program, page)
        for (const RawCoverageMiss& miss : ring.misses) {
            PageMisses& page{pages[{miss.program, miss.guest & ~(RAW_COVERAGE_PAGE_SIZE - 1)}]};
            if (miss.guest < page.lowest) {
                page.lowest = miss.guest;
                page.op = miss.op;
                page.stage = miss.stage;
            }
            page.first_present = (std::min)(page.first_present, miss.present);
            ++page.count;
        }
        std::array<u64, NUM_RAW_MISS_CLASSES> classes{};
        std::vector<std::string> new_lines;
        size_t new_pages{};
        for (const auto& [key, page] : pages) {
            const RawMissVerdict verdict{classify(page.lowest)};
            classes[static_cast<size_t>(verdict.kind)] += page.count;
            auto [it, inserted] = run_pages.try_emplace(key);
            RunPage& run{it->second};
            run.count += page.count;
            if (!inserted) {
                continue;
            }
            run.kind = RawMissKindText(verdict);
            run.op = page.op;
            run.stage = page.stage;
            run.first_present = page.first_present;
            ++new_pages;
            if (new_lines.size() < MAX_NEW_LINES_PER_WINDOW &&
                logged_new_lines < MAX_NEW_LINES_PER_RUN) {
                ++logged_new_lines;
                new_lines.push_back(MissLine(key, run, page.count));
            }
        }
        ++windows;
        run_dispatches += dispatches;
        run_misses += ring.total;
        run_dropped += ring.dropped;
        full_windows += ring.full ? 1 : 0;
        for (size_t index = 0; index < classes.size(); ++index) {
            run_classes[index] += classes[index];
        }
        std::vector<std::string> lines;
        lines.push_back(fmt::format(
            "[raw] coverage window={}s dispatches={} misses={} dropped={} full={} {} pages={} new={}{}",
            window_seconds, dispatches, ring.total, ring.dropped, ring.full ? 1 : 0,
            ClassText(classes), pages.size(), new_pages, path_field));
        lines.insert(lines.end(), std::make_move_iterator(new_lines.begin()),
                     std::make_move_iterator(new_lines.end()));
        // Runs usually end by a kill (no exit summary): the totals so far after every window,
        // the top pages every TOP_EVERY_WINDOWS windows.
        std::vector<std::string> summary{Summary(0, windows % TOP_EVERY_WINDOWS == 0)};
        lines.insert(lines.end(), std::make_move_iterator(summary.begin()),
                     std::make_move_iterator(summary.end()));
        return lines;
    }

    /// The run totals so far and, with `top`, the pages that missed most. `tail` counts the misses
    /// of the last, unfinished window (not classified; at exit).
    [[nodiscard]] std::vector<std::string> Summary(u64 tail, bool top_rows) const {
        std::vector<std::string> lines;
        lines.push_back(fmt::format("[raw] coverage run windows={} dispatches={} misses={} "
                                    "dropped={} full_windows={} {} pages={} tail={}{}",
                                    windows, run_dispatches, run_misses, run_dropped,
                                    full_windows, ClassText(run_classes), run_pages.size(), tail,
                                    path_field));
        if (!top_rows) {
            return lines;
        }
        std::vector<std::pair<std::pair<u64, u64>, const RunPage*>> top;
        top.reserve(run_pages.size());
        for (const auto& [key, page] : run_pages) {
            top.emplace_back(key, &page);
        }
        const size_t rows{(std::min)(top.size(), TOP_ROWS)};
        std::partial_sort(top.begin(), top.begin() + rows, top.end(),
                          [](const auto& lhs, const auto& rhs) {
                              return lhs.second->count != rhs.second->count
                                         ? lhs.second->count > rhs.second->count
                                         : lhs.first < rhs.first;
                          });
        for (size_t rank = 0; rank < rows; ++rank) {
            lines.push_back(fmt::format("[raw] coverage top rank={} {}{}", rank + 1,
                                        MissFields(top[rank].first, *top[rank].second,
                                                   top[rank].second->count),
                                        path_field));
        }
        return lines;
    }

private:
    struct RunPage {
        u64 count{};
        std::string kind;
        u32 op{};
        u32 stage{};
        u32 first_present{};
    };

    [[nodiscard]] static std::string ClassText(const std::array<u64, NUM_RAW_MISS_CLASSES>& classes) {
        return fmt::format("mapped_gap={} past_end={} unmapped={} null={} late={}",
                           classes[static_cast<size_t>(RawMissClass::MappedGap)],
                           classes[static_cast<size_t>(RawMissClass::PastEnd)],
                           classes[static_cast<size_t>(RawMissClass::Unmapped)],
                           classes[static_cast<size_t>(RawMissClass::Null)],
                           classes[static_cast<size_t>(RawMissClass::Late)]);
    }

    [[nodiscard]] static std::string MissFields(const std::pair<u64, u64>& key, const RunPage& page,
                                                u64 count) {
        return fmt::format("prog={:016x} stage={} page={:#x}/4K kind={} op={} count={} "
                           "first_present={}",
                           key.first, RawCoverageStageName(page.stage), key.second, page.kind,
                           RawCoverageOpName(page.op), count, page.first_present);
    }

    [[nodiscard]] std::string MissLine(const std::pair<u64, u64>& key, const RunPage& page,
                                       u64 count) const {
        return "[raw] coverage miss " + MissFields(key, page, count) + path_field;
    }

    /// " path=<name>" for a named GPU path, empty for the legacy path.
    std::string path_field;

    u64 windows{};
    u64 run_dispatches{};
    u64 run_misses{};
    u64 run_dropped{};
    u64 full_windows{};
    std::array<u64, NUM_RAW_MISS_CLASSES> run_classes{};
    std::map<std::pair<u64, u64>, RunPage> run_pages;
    size_t logged_new_lines{};
};

} // namespace VideoCommon
