// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "core/hle/nvn/thunk_encoding.h"

namespace Core::NvnHle::Thunk {

bool DecodeLdrX16Literal(u32 word, u64 pc, u64& literal_address) {
    if ((word & 0xFF000000u) != 0x58000000u || (word & 0x1Fu) != 16u) {
        return false;
    }
    // Sign-extend imm19.
    s32 imm19 = static_cast<s32>((word >> 5) & 0x7FFFFu);
    if (imm19 & 0x40000) {
        imm19 -= 0x80000;
    }
    literal_address = pc + static_cast<u64>(static_cast<s64>(imm19) * 4);
    return true;
}

void WriteSlotCode(std::span<u8> region) {
    for (u32 slot = 0; slot < SlotCount; ++slot) {
        const auto words = SlotWords(slot);
        std::memcpy(region.data() + SlotOffset(slot), words.data(), sizeof(words));
    }
}

} // namespace Core::NvnHle::Thunk
