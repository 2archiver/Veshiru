// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"

namespace FileSys {

constexpr u64 AOC_TITLE_ID_MASK = 0x7FF;
constexpr u64 AOC_TITLE_ID_OFFSET = 0x1000;
constexpr u64 BASE_TITLE_ID_MASK = 0xFFFFFFFFFFFFE000;

/**
 * Gets the base title ID from a given title ID.
 *
 * @param title_id The title ID.
 * @returns The base title ID.
 */
[[nodiscard]] constexpr u64 GetBaseTitleID(u64 title_id) {
    return title_id & BASE_TITLE_ID_MASK;
}

/**
 * Gets the base title ID with a program index offset from a given title ID.
 *
 * @param title_id The title ID.
 * @param program_index The program index.
 * @returns The base title ID with a program index offset.
 */
[[nodiscard]] constexpr u64 GetBaseTitleIDWithProgramIndex(u64 title_id, u64 program_index) {
    return GetBaseTitleID(title_id) + program_index;
}

/**
 * Gets the AOC (Add-On Content) base title ID from a given title ID.
 *
 * @param title_id The title ID.
 * @returns The AOC base title ID.
 */
[[nodiscard]] constexpr u64 GetAOCBaseTitleID(u64 title_id) {
    return GetBaseTitleID(title_id) + AOC_TITLE_ID_OFFSET;
}

/**
 * Gets the AOC (Add-On Content) ID from a given AOC title ID.
 *
 * @param aoc_title_id The AOC title ID.
 * @returns The AOC ID.
 */
[[nodiscard]] constexpr u64 GetAOCID(u64 aoc_title_id) {
    return aoc_title_id & AOC_TITLE_ID_MASK;
}

// Legacy NUP metadata may omit the DLC base. Only infer it from add-ons of this program;
// if the package IDs disagree, use the standard namespace rather than depending on their order.
[[nodiscard]] constexpr u64 ResolveAOCBaseTitleID(u64 program_id, u64 metadata_base,
                                                  std::span<const u64> carried) {
    if (metadata_base != 0) {
        return metadata_base;
    }
    u64 inferred{};
    for (const u64 id : carried) {
        if (GetBaseTitleID(id) != GetBaseTitleID(program_id)) {
            continue;
        }
        const u64 base = id & ~u64{0xFFF};
        if (inferred != 0 && inferred != base) {
            return GetAOCBaseTitleID(program_id);
        }
        inferred = base;
    }
    return inferred != 0 ? inferred : GetAOCBaseTitleID(program_id);
}

/// Whether `addon_index` (what aoc:u's ListAddOnContent hands the guest: an add-on's title id
/// masked with AOC_TITLE_ID_MASK) names one of `add_on_title_ids` that belongs to `program_id`.
/// Add-ons of other programs are ignored, as ListAddOnContent ignores them. A negative index
/// names nothing.
[[nodiscard]] constexpr bool IsListedAddOnContentIndex(u64 program_id, s32 addon_index,
                                                       std::span<const u64> add_on_title_ids) {
    if (addon_index < 0) {
        return false;
    }
    const u64 index = static_cast<u64>(addon_index);
    for (const u64 id : add_on_title_ids) {
        if (GetBaseTitleID(id) == GetBaseTitleID(program_id) && GetAOCID(id) == index) {
            return true;
        }
    }
    return false;
}

} // namespace FileSys
