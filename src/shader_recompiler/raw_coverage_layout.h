// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

/// Raw global memory coverage diagnostic (EDEN_NX2_RAW_COVERAGE=1, HostTranslateInfo::raw_coverage):
/// a raw global access whose guest address no entry of the address table holds (the load reads 0,
/// the store or atomic is dropped) appends a record to a miss ring the host reads back. Layout
/// shared by the SPIR-V backend (EmitContext::DefineRawGlobalMemoryFunctions) and the host
/// (VideoCommon::ParseRawCoverageRing).
namespace Shader::RawCoverage {

/// u32 words 2-3 of the table storage buffer header (16 bytes, entries from
/// Profile::RAW_GLOBAL_TABLE_SSBO_ENTRIES_WORD): the device address of the miss ring, 0 for none.
inline constexpr u32 TABLE_RING_WORD{2};

/// The table in the driver bank (NVN HLE: c0, Profile::RAW_GLOBAL_TABLE_OFFSET): the device
/// address of the miss ring is the u64 at this byte offset of c0, right after the table's
/// address (Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET); 0 for none.
inline constexpr u32 BANK_RING_OFFSET{0xF10};

/// Ring header: u32 word 0 counts the misses (atomic add, may exceed RING_ENTRIES), word 1 is the
/// present number the host last wrote, word 2 counts the null misses that asked for a record, the
/// rest is zero. Records follow.
inline constexpr u32 RING_HEADER_BYTES{32};
inline constexpr u32 RING_COUNTER_WORD{0};
inline constexpr u32 RING_PRESENT_WORD{1};
inline constexpr u32 RING_NULL_WORD{2};
inline constexpr u32 RING_ENTRIES{4096};

/// A miss below this address is a null pointer plus a field offset.
inline constexpr u32 NULL_ADDRESS_LIMIT{4096};
/// Null misses take at most this many records per window (then they are not counted): one
/// program reading through a null pointer in every invocation would otherwise fill the ring in a
/// frame and hide every other miss.
inline constexpr u32 NULL_RECORDS{256};

/// Record (u32 words): 0-1 guest address, 2-3 program hash, 4 Op | Stage << 8, 5 present number,
/// 6-7 zero.
inline constexpr u32 RECORD_BYTES{32};
inline constexpr u32 RECORD_KIND_WORD{4};
inline constexpr u32 RECORD_PRESENT_WORD{5};
inline constexpr u32 RECORD_STAGE_SHIFT{8};

/// The access that missed.
enum class Op : u32 {
    Load32 = 1,
    Load64 = 2,
    Load128 = 3,
    Store32 = 4,
    Store64 = 5,
    Store128 = 6,
    StoreSubword = 7, ///< 8/16-bit store (the aligned word that holds it)
    Atomic = 8,
};

} // namespace Shader::RawCoverage
