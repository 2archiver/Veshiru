// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/diag_request.h"

namespace VideoCommon {

/// Raw global memory (global accesses at an address the guest loaded from memory) on the legacy GPU
/// path. The shader gets a table of guest GPU address ranges and the device addresses
/// (VK_KHR_buffer_device_address) that hold them: the buffer cache buffers that were bound, and
/// under them the guest memory of every other mapped page (imported host memory,
/// Vulkan::RawGuestMemory). It is an extra storage buffer: u32 words, the entry count at word 0
/// and `RawGlobalTableEntry`s from word 4 (Shader::Profile::RAW_GLOBAL_TABLE_SSBO_ENTRIES_WORD).
struct RawGlobalRange {
    u64 guest;  ///< First byte, in the guest GPU (or any other linear) address space
    u64 device; ///< Address of the same byte on the other side
    u64 size;
};

struct RawGlobalTableEntry {
    u64 guest;
    u64 device;
    u64 size;
};
static_assert(sizeof(RawGlobalTableEntry) == 24);

inline constexpr size_t RAW_GLOBAL_TABLE_HEADER_BYTES{16};
inline constexpr size_t RAW_GLOBAL_TABLE_MAX_ENTRIES{1u << 16};

/// Sorts `ranges` by guest address and removes overlaps: where two ranges overlap the one with the
/// lower start (the longer one at the same start) keeps the overlapped bytes. Ranges that touch in
/// both address spaces are joined. The result is sorted and never overlaps, which the shader's
/// binary search relies on.
inline void FlattenRawGlobalRanges(std::vector<RawGlobalRange>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const RawGlobalRange& lhs, const RawGlobalRange& rhs) {
        return lhs.guest != rhs.guest ? lhs.guest < rhs.guest : lhs.size > rhs.size;
    });
    size_t out{};
    for (size_t index = 0; index < ranges.size(); ++index) {
        RawGlobalRange range{ranges[index]};
        if (range.size == 0) {
            continue;
        }
        if (out != 0) {
            RawGlobalRange& last{ranges[out - 1]};
            const u64 last_end{last.guest + last.size};
            if (range.guest < last_end) {
                const u64 covered{last_end - range.guest};
                if (covered >= range.size) {
                    continue;
                }
                range.guest += covered;
                range.device += covered;
                range.size -= covered;
            }
            if (range.guest == last_end && range.device == last.device + last.size) {
                last.size += range.size;
                continue;
            }
        }
        ranges[out++] = range;
    }
    ranges.resize(out);
}

/// The table storage buffer content for flattened `ranges` (see FlattenRawGlobalRanges). At most
/// RAW_GLOBAL_TABLE_MAX_ENTRIES entries are written. Returns the number of entries.
inline size_t BuildRawGlobalTable(std::span<const RawGlobalRange> ranges, std::vector<u8>& bytes) {
    const size_t count{(std::min)(ranges.size(), RAW_GLOBAL_TABLE_MAX_ENTRIES)};
    bytes.assign(RAW_GLOBAL_TABLE_HEADER_BYTES + count * sizeof(RawGlobalTableEntry), 0);
    const u32 count32{static_cast<u32>(count)};
    std::memcpy(bytes.data(), &count32, sizeof(count32));
    for (size_t index = 0; index < count; ++index) {
        const RawGlobalTableEntry entry{
            .guest = ranges[index].guest,
            .device = ranges[index].device,
            .size = ranges[index].size,
        };
        std::memcpy(bytes.data() + RAW_GLOBAL_TABLE_HEADER_BYTES + index * sizeof(entry), &entry,
                    sizeof(entry));
    }
    return count;
}

/// The binary search of the shader: the number of entries whose guest base is <= `guest` (the
/// candidate is the one before that index; 0: none).
inline u32 RawGlobalSearch(std::span<const RawGlobalRange> ranges, u64 guest) {
    u32 low{};
    u32 high{static_cast<u32>(ranges.size())};
    while (low < high) {
        const u32 middle{(low + high) >> 1};
        if (ranges[middle].guest <= guest) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

/// What the shader's binary search does with the table: the device address of `guest`, or 0 when
/// no entry contains it. The reference for the SPIR-V in DefineRawGlobalMemoryFunctions.
inline u64 TranslateRawGlobalAddress(std::span<const RawGlobalRange> ranges, u64 guest) {
    const u32 found{RawGlobalSearch(ranges, guest)};
    if (found == 0) {
        return 0;
    }
    const RawGlobalRange& range{ranges[found - 1]};
    const u64 relative{guest - range.guest};
    return relative < range.size ? range.device + relative : 0;
}

/// The table entry an invocation translated through last (size 0: none yet).
struct RawGlobalLastHit {
    u64 guest{};
    u64 device{};
    u64 size{};
};

/// TranslateRawGlobalAddress with the shader's last-hit cache (Shader::Info::raw_global_last_hit,
/// EDEN_NX2_TRY_RAWGLOBAL_FAST=1): an address inside `last` maps without the search; otherwise
/// the search's candidate entry becomes `last`. Equal to TranslateRawGlobalAddress for every
/// address as long as `last` is an entry of `ranges` (sorted, never overlapping) or empty.
inline u64 TranslateRawGlobalAddressLastHit(std::span<const RawGlobalRange> ranges, u64 guest,
                                            RawGlobalLastHit& last) {
    const u64 cached{guest - last.guest};
    if (cached < last.size) {
        return last.device + cached;
    }
    const u32 found{RawGlobalSearch(ranges, guest)};
    if (found == 0) {
        return 0;
    }
    const RawGlobalRange& range{ranges[found - 1]};
    last = {range.guest, range.device, range.size};
    const u64 relative{guest - range.guest};
    return relative < range.size ? range.device + relative : 0;
}

/// EDEN_NX2_TRY_RAWGLOBAL_FAST=1: shaders translate raw global addresses with the last-hit
/// cache (HostTranslateInfo::raw_global_last_hit). Checked for every translation.
[[nodiscard]] inline bool RawGlobalFastEnabled() {
    return Common::DiagRequest::TrySwitchEnabled("TRY_RAWGLOBAL_FAST");
}

/// Adds `fallback` ranges under `ranges` (both flattened, see FlattenRawGlobalRanges): bytes only a
/// fallback range holds join the table, bytes both hold keep the mapping of `ranges`. The result
/// is flattened. The legacy path's fallback is the guest memory itself (imported host memory) under
/// the buffer cache buffers that were bound: what the GPU's own caches hold wins.
inline void AddRawGlobalFallbackRanges(std::vector<RawGlobalRange>& ranges,
                                       std::span<const RawGlobalRange> fallback) {
    if (fallback.empty()) {
        return;
    }
    std::vector<RawGlobalRange> uncovered;
    size_t primary{};
    for (const RawGlobalRange& range : fallback) {
        u64 begin{range.guest};
        const u64 end{range.guest + range.size};
        while (primary < ranges.size() && ranges[primary].guest + ranges[primary].size <= begin) {
            ++primary;
        }
        for (size_t index = primary; index < ranges.size() && begin < end; ++index) {
            const RawGlobalRange& covered{ranges[index]};
            if (covered.guest >= end) {
                break;
            }
            if (covered.guest > begin) {
                uncovered.push_back({begin, range.device + (begin - range.guest), covered.guest - begin});
            }
            begin = (std::max)(begin, covered.guest + covered.size);
        }
        if (begin < end) {
            uncovered.push_back({begin, range.device + (begin - range.guest), end - begin});
        }
    }
    ranges.insert(ranges.end(), uncovered.begin(), uncovered.end());
    FlattenRawGlobalRanges(ranges);
}

/// Guest pages whose bytes lie at consecutive host addresses.
struct RawHostRun {
    u64 guest;
    u8* host;
    u64 size;
};

/// Splits [guest, guest + size) (whole pages of `page_size`) into the runs of pages that follow
/// each other on the host. `page_pointer(page)` is the host pointer of a guest page, nullptr for a
/// page without one (left out).
template <typename PagePointer>
void SplitRawHostRuns(u64 guest, u64 size, u64 page_size, PagePointer&& page_pointer,
                      std::vector<RawHostRun>& out) {
    RawHostRun run{};
    for (u64 page = guest; page < guest + size; page += page_size) {
        u8* const host{page_pointer(page)};
        if (host != nullptr && run.size != 0 && run.guest + run.size == page &&
            run.host + run.size == host) {
            run.size += page_size;
            continue;
        }
        if (run.size != 0) {
            out.push_back(run);
        }
        run = host != nullptr ? RawHostRun{page, host, page_size} : RawHostRun{};
    }
    if (run.size != 0) {
        out.push_back(run);
    }
}

/// Host memory imported as device memory: [host, host + size) is at `device`.
struct RawHostChunk {
    const u8* host;
    u64 size;
    u64 device;
};

/// A host range no chunk holds yet.
struct RawHostSpan {
    u8* host;
    u64 size;
};

/// The table ranges of `runs` through `chunks` (sorted by host address, never overlapping),
/// flattened. The host bytes of the runs that no chunk holds go to `uncovered`, sorted, with
/// spans that touch or overlap joined (they are all guest-mapped memory: one import can hold them).
[[nodiscard]] inline std::vector<RawGlobalRange> RawChunkedRanges(
    std::span<const RawHostRun> runs, std::span<const RawHostChunk> chunks,
    std::vector<RawHostSpan>& uncovered) {
    std::vector<RawGlobalRange> ranges;
    std::vector<RawHostSpan> missing;
    for (const RawHostRun& run : runs) {
        u64 offset{};
        while (offset < run.size) {
            u8* const host{run.host + offset};
            // The last chunk starting at or before `host`.
            const auto next = std::upper_bound(chunks.begin(), chunks.end(), host,
                                               [](const u8* value, const RawHostChunk& chunk) {
                                                   return value < chunk.host;
                                               });
            u64 length{run.size - offset};
            if (next != chunks.begin() && host < std::prev(next)->host + std::prev(next)->size) {
                const RawHostChunk& chunk{*std::prev(next)};
                const u64 chunk_offset{static_cast<u64>(host - chunk.host)};
                length = (std::min)(length, chunk.size - chunk_offset);
                ranges.push_back({run.guest + offset, chunk.device + chunk_offset, length});
            } else {
                if (next != chunks.end()) {
                    length = (std::min)(length, static_cast<u64>(reinterpret_cast<std::uintptr_t>(next->host) -
                                                               reinterpret_cast<std::uintptr_t>(host)));
                }
                missing.push_back({host, length});
            }
            offset += length;
        }
    }
    FlattenRawGlobalRanges(ranges);
    std::sort(missing.begin(), missing.end(),
              [](const RawHostSpan& lhs, const RawHostSpan& rhs) { return lhs.host < rhs.host; });
    uncovered.clear();
    for (const RawHostSpan& span : missing) {
        if (!uncovered.empty() && span.host <= uncovered.back().host + uncovered.back().size) {
            RawHostSpan& last{uncovered.back()};
            last.size = (std::max)(last.size, static_cast<u64>(span.host + span.size - last.host));
            continue;
        }
        uncovered.push_back(span);
    }
    return ranges;
}

/// The imports that give `uncovered` (sorted, see RawChunkedRanges) to the table, few and bounded:
/// each span grows by the committed host memory on its sides (`used_around(begin, end, low, high)`
/// with offsets from `region`, see Common::HostMemory::UsedAround) up to the neighbouring chunks
/// and its `view_size` view of `region` (Common::HostMemory::BackingViewSize). A scene's mapped
/// pages lie scattered over the guest heap: one import per view holds the hundreds of spans in it,
/// where one per span cost a thousand imports; an import never leaves its view (its driver time
/// grows with the size). Extents that touch in one view are joined. Sorted, never overlapping
/// each other or `chunks`.
template <typename UsedAround>
[[nodiscard]] std::vector<RawHostSpan> RawImportExtents(std::span<const RawHostSpan> uncovered,
                                                        std::span<const RawHostChunk> chunks,
                                                        u8* region, u64 region_size,
                                                        u64 view_size, UsedAround&& used_around) {
    std::vector<RawHostSpan> extents;
    for (const RawHostSpan& span : uncovered) {
        u64 begin{static_cast<u64>(span.host - region)};
        const u64 span_end{begin + span.size};
        while (begin < span_end) {
            const u64 view_begin{begin / view_size * view_size};
            const u64 end{(std::min)(span_end, view_begin + view_size)};
            u64 low{view_begin};
            u64 high{(std::min)(view_begin + view_size, region_size)};
            const auto next = std::upper_bound(chunks.begin(), chunks.end(), region + begin,
                                               [](const u8* value, const RawHostChunk& chunk) {
                                                   return value < chunk.host;
                                               });
            if (next != chunks.begin()) {
                const RawHostChunk& previous{*std::prev(next)};
                low = (std::max)(low, static_cast<u64>(previous.host + previous.size - region));
            }
            if (next != chunks.end()) {
                high = (std::min)(high, static_cast<u64>(next->host - region));
            }
            if (!extents.empty()) {
                const RawHostSpan& last{extents.back()};
                low = (std::max)(low, static_cast<u64>(last.host + last.size - region));
            }
            if (begin < low) {
                // Already held by the previous extent.
                begin = (std::min)(low, end);
                continue;
            }
            const auto [grown_begin, grown_end] = used_around(begin, end, low, (std::max)(high, end));
            if (!extents.empty()) {
                RawHostSpan& last{extents.back()};
                const u64 last_begin{static_cast<u64>(last.host - region)};
                if (last_begin + last.size == grown_begin && last_begin >= view_begin) {
                    last.size += grown_end - grown_begin;
                    begin = grown_end;
                    continue;
                }
            }
            extents.push_back({region + grown_begin, grown_end - grown_begin});
            begin = grown_end;
        }
    }
    return extents;
}

} // namespace VideoCommon
