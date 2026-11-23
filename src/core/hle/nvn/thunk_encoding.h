// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>

#include "common/common_types.h"

namespace Core::NvnHle::Thunk {

// Thunk region appended to the `sdk` module image:
//   [0x0000, 0x8000) 2048 slots of 16 bytes (RX), written once at load
//   [0x8000, 0xC000) 2048 x 8-byte target pointers (RW)
inline constexpr u32 SlotCount = 2048;
inline constexpr u32 SlotBytes = 16;
inline constexpr u32 CodeBytes = SlotCount * SlotBytes;
inline constexpr u32 TableOffset = CodeBytes;
inline constexpr u32 TableBytes = SlotCount * 8;
inline constexpr u32 RegionBytes = TableOffset + TableBytes;
inline constexpr u32 ReturnSlot = SlotCount - 1;
/// SVC immediate of slot i is SvcBase + i.
inline constexpr u32 SvcBase = 0x8000;

inline constexpr u32 BrX16 = 0xD61F0200;
inline constexpr u32 Brk0 = 0xD4200000;

[[nodiscard]] constexpr u32 EncodeSvc(u32 imm16) {
    return 0xD4000001u | ((imm16 & 0xFFFFu) << 5);
}

/// `ldr x16, <pc + byte_offset>` (64-bit LDR literal).
[[nodiscard]] constexpr u32 EncodeLdrX16Literal(s32 byte_offset) {
    return 0x58000000u | ((static_cast<u32>(byte_offset >> 2) & 0x7FFFFu) << 5) | 16u;
}

[[nodiscard]] constexpr u32 SlotOffset(u32 slot) {
    return slot * SlotBytes;
}

[[nodiscard]] constexpr u32 TableEntryOffset(u32 slot) {
    return TableOffset + slot * 8;
}

/// The four instruction words of slot `slot`: svc, ldr x16 (table entry), br x16, brk #0.
[[nodiscard]] constexpr std::array<u32, 4> SlotWords(u32 slot) {
    // The ldr sits at slot offset + 4.
    const s32 ldr_pc = static_cast<s32>(SlotOffset(slot) + 4);
    const s32 distance = static_cast<s32>(TableEntryOffset(slot)) - ldr_pc;
    return {EncodeSvc(SvcBase + slot), EncodeLdrX16Literal(distance), BrX16, Brk0};
}

/// Decodes `ldr x16, <literal>` located at `pc`; returns the literal's address, or false.
[[nodiscard]] bool DecodeLdrX16Literal(u32 word, u64 pc, u64& literal_address);

/// Fills the code part of the region (CodeBytes) with the slot code. The table part is untouched.
void WriteSlotCode(std::span<u8> region);

} // namespace Core::NvnHle::Thunk
