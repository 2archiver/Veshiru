// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"
#include "common/literals.h"

namespace VideoCommon::TextureCacheMemory {

using namespace Common::Literals;

/// The lowest limit the texture cache accepts, however small a value is set.
inline constexpr u64 MinLimit = 1_GiB;

/// What Auto (a limit of 0) uses: three quarters of the device memory.
[[nodiscard]] constexpr u64 RecommendedLimit(u64 device_memory) {
    return device_memory / 4 * 3;
}

/// The limit in bytes for a setting value in MiB. Without a known device memory (0) there is
/// nothing to follow or clamp to.
[[nodiscard]] constexpr u64 ResolveLimit(u32 setting_mib, u64 device_memory) {
    if (device_memory == 0) {
        return setting_mib != 0 ? u64{setting_mib} << 20 : ~u64{0};
    }
    if (setting_mib == 0) {
        return RecommendedLimit(device_memory);
    }
    return std::clamp(u64{setting_mib} << 20, (std::min)(MinLimit, device_memory), device_memory);
}

/// True when an explicit limit sits under what Auto would use.
[[nodiscard]] constexpr bool IsBelowRecommended(u32 setting_mib, u64 device_memory) {
    return setting_mib != 0 && device_memory != 0 &&
           (u64{setting_mib} << 20) < RecommendedLimit(device_memory);
}

/// Lowest dynamic driver budget (VK_EXT_memory_budget) the texture cache tracks when updating
/// native-tiling thresholds. Lowering from 2 GiB to 1 GiB ensures 4 GB GPUs (e.g. GTX 1650 with
/// ~3.04 GiB base budget) keep reacting if background desktop apps temporarily push available
/// VRAM below 2 GiB instead of staying frozen at a stale threshold.
inline constexpr u64 MinTrackedDeviceBudget = 1_GiB;

/// Switch 2 native buffer-cache GC thresholds scaled by VRAM tier.
struct NativeBufferGcThresholds {
    u64 expected_bytes;      ///< Start non-aggressive buffer GC
    u64 critical_bytes;      ///< Escalate to aggressive buffer GC
    u64 pressure_floor_bytes; ///< Minimum buffer footprint to shed under device-wide pressure
    bool tight_vram;         ///< True on <= 4.5 GiB cards that must cooperate with TextureCache
};

[[nodiscard]] constexpr NativeBufferGcThresholds ResolveNativeBufferGcThresholds(
    u64 device_local_memory) {
    // Keep the 512 MiB collection start across VRAM tiers: Switch 2 scenes keep ~300-550 MiB of
    // live GPU-written buffers (e.g. Mario Kart World at 285-330 MiB), and firing aggressive
    // eviction below that forces synchronous host-readback stalls and constant re-creation every
    // frame. On 4 GB GPUs (e.g. GTX 1650 at ~3935 MiB), shed only clean (non-GPU-modified) stale
    // buffers above 256 MiB when device-wide memory is critical, and reserve aggressive eviction
    // for 768 MiB+ buffer footprints.
    if (device_local_memory > 0 && device_local_memory <= (u64{9} << 29)) { // <= 4.5 GiB
        return {
            .expected_bytes = 512_MiB,
            .critical_bytes = 768_MiB,
            .pressure_floor_bytes = 256_MiB,
            .tight_vram = true,
        };
    }
    // 6+ GB GPUs (or unknown device-local memory): 512 MiB non-aggressive, 1 GiB aggressive.
    return {
        .expected_bytes = 512_MiB,
        .critical_bytes = 1_GiB,
        .pressure_floor_bytes = 512_MiB,
        .tight_vram = false,
    };
}

/// Decoded RGBA8 byte ceiling for trimmed oversized GPU-decoded ASTC texture arrays.
[[nodiscard]] constexpr u64 ResolveTrimmedAstcArrayLimit(u64 device_local_memory) {
    if (device_local_memory > 0 && device_local_memory < 5_GiB) {
        return 128_MiB;
    }
    if (device_local_memory < 7_GiB) {
        return 256_MiB;
    }
    return 512_MiB;
}

} // namespace VideoCommon::TextureCacheMemory
