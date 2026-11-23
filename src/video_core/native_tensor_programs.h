// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCommon {

/// How a game's native matrix multiplies (HMMA, its own DLSS-style network) are lowered.
enum class NativeTensorLowering : u8 {
    None,    ///< rejected (the program is skipped or replaced)
    Full,    ///< inline expansion
    Compact, ///< one call of a shared helper per HMMA; needs compute shaders 32 threads wide
};

inline constexpr u64 DonkeyKongBananzaProgramId = 0x040074A01BF12000ULL;
inline constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;

/// The titles whose tensor programs are lowered at all, chosen per title, not per dispatch.
/// Donkey Kong Bananza's programs write nothing when rejected (the screen is purple from its mode
/// select on). Ocarina of Time draws its final image through an 11-program network after the
/// tonemap. Other titles keep them rejected: emulating title 4's network takes the host driver
/// minutes to compile.
[[nodiscard]] constexpr bool LowersNativeTensorPrograms(u64 program_id) noexcept {
    return program_id == DonkeyKongBananzaProgramId || program_id == OcarinaOfTimeProgramId;
}

/// Of those, the titles lowered in the compact form where compute shaders can be created 32
/// threads wide (the helper shuffles across 32 lanes).
[[nodiscard]] constexpr bool UsesCompactNativeTensorPrograms(u64 program_id) noexcept {
    return program_id == OcarinaOfTimeProgramId;
}

/// The lowering for `program_id`'s compute programs when the host can (`guest_warp_compute`) or
/// cannot create compute shaders with a required subgroup size of 32.
[[nodiscard]] constexpr NativeTensorLowering NativeTensorLoweringFor(
    u64 program_id, bool guest_warp_compute) noexcept {
    if (!LowersNativeTensorPrograms(program_id)) {
        return NativeTensorLowering::None;
    }
    return UsesCompactNativeTensorPrograms(program_id) && guest_warp_compute
               ? NativeTensorLowering::Compact
               : NativeTensorLowering::Full;
}

} // namespace VideoCommon
