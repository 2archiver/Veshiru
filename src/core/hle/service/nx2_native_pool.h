// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/alignment.h"
#include "common/common_types.h"

// The Switch 2 graphics server's native image layout (nx2_gsv.h), shared with the NVN HLE.

namespace Service {

// Block-linear layout of an InterModuleUserImagePreference (16 words): word 2 holds
// (width - 1) | (height - 1) << 16, word 4 the block height (log2 of GOB rows), word 5 the row
// pitch in 64-byte GOBs. Formula from changes.patch (native_image_layout.h); preferences outside
// it fall back to 4 bytes per pixel and 128-row blocks.
struct Nx2ImageLayout {
    u32 width;
    u32 height;
    u64 pitch;
    u32 block_height_log2;
    u64 size;
    u64 alignment;
};

inline Nx2ImageLayout Nx2CalculateImageLayout(const u32* preference) {
    Nx2ImageLayout layout{};
    layout.width = (preference[2] & 0xFFFF) + 1;
    layout.height = (preference[2] >> 16) + 1;
    const u64 pitch_gobs = preference[5];
    const u32 block_height_log2 = preference[4];
    if (block_height_log2 <= 8 && pitch_gobs >= (u64{layout.width} * 4 + 63) / 64 &&
        pitch_gobs <= 4096) {
        layout.pitch = pitch_gobs * 64;
        layout.block_height_log2 = block_height_log2;
    } else {
        layout.pitch = Common::AlignUp(u64{layout.width} * 4, 64);
        layout.block_height_log2 = 7;
    }
    const u64 block_rows = u64{1} << layout.block_height_log2;
    layout.size = layout.pitch * Common::AlignUp(u64{layout.height}, block_rows);
    layout.alignment = std::max<u64>(64 * block_rows, 0x1000);
    return layout;
}

} // namespace Service
