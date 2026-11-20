// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"
#include "common/literals.h"

/// How much video memory Veshiru keeps free of the driver's budget. Shared by the settings
/// (VRAM Reserve, the reported figure) and the texture cache's collector thresholds so both use
/// the same number.
///
/// One policy replaces the margins Veshiru 1.x stacked: a base share, a reaction to every
/// allocation that missed the budget, and a reserve for the allocator's block space. They reacted
/// to each other. An allocation that missed the budget raised the margin by 256-512 MiB, the
/// raised margin made the next allocations miss, and on a 4 GB card the margin reached a 2 GiB
/// cap: the cache then kept half the card free and re-decoded textures every frame (the log's
/// "Out of device memory: keeping 2048 MiB more free" once per few hundred milliseconds). The
/// reserve now follows the card's budget alone, so it settles instead of ratcheting up.
namespace Common::VramPolicy {

using namespace Common::Literals;

/// The most kept free, whatever the card.
inline constexpr u64 MaxReserve = 1_GiB;

/// An eighth of the budget, at most MaxReserve: about 480 MiB on the 4 GB GTX 1650 and about
/// 960 MiB on an 8 GB card. Covers driver allocations the cache does not own, the unused part of
/// the allocator's blocks (which the driver counts as used), and a frame's worth of new images.
[[nodiscard]] constexpr u64 AutoReserve(u64 budget) {
    return (std::min)(budget / 8, MaxReserve);
}

/// The budget the cache fills up to: the driver budget less reserve, never below half the budget
/// so a shrunken budget (another program took VRAM) can't drive the cache to zero.
[[nodiscard]] constexpr u64 CriticalMemory(u64 budget, u64 reserve) {
    return budget - (std::min)(reserve, budget / 2);
}

} // namespace Common::VramPolicy
