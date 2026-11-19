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

/// Computes the effective GPU memory ceiling for an integrated / unified-memory GPU, eliminating
/// fixed 512/2048 MiB BIOS UMA carveout caps and 4/6 GiB unified-RAM ceilings while keeping
/// sufficient headroom for Switch 2 guest DRAM and the host OS.
/// - `host_ram_bytes`: OS-visible physical RAM (Common::GetMemInfo().TotalPhysicalMemory), or 0.
/// - `local_heap_bytes`: Size of the largest VK_MEMORY_HEAP_DEVICE_LOCAL_BIT heap (BIOS UMA
///   carveout on Windows WDDM APUs, or the unified heap on Linux RADV / Intel ANV / MoltenVK).
/// - `shared_heap_bytes`: Sum of non-DEVICE_LOCAL heap sizes (shared system GPU segment on
///   multi-heap iGPU drivers; 0 on single-heap unified drivers).
/// - `usable_pool_bytes`: Live usable bytes across the primary local + shared heaps (budget minus
///   initial usage when VK_EXT_memory_budget is present, or 0 to fall back to heap sizes).
/// - `aggressive`: True for Aggressive VRAM mode or when sizing the maximum selectable UI ceiling.
[[nodiscard]] constexpr u64 ResolveIntegratedGpuMemory(u64 host_ram_bytes, u64 local_heap_bytes,
                                                       u64 shared_heap_bytes, u64 usable_pool_bytes,
                                                       bool aggressive) {
    const u64 combined_heap = local_heap_bytes <= ~u64{0} - shared_heap_bytes
                                  ? local_heap_bytes + shared_heap_bytes
                                  : ~u64{0};
    const u64 pool = usable_pool_bytes != 0 ? usable_pool_bytes : combined_heap;
    if (pool == 0) {
        return 0;
    }
    if (host_ram_bytes == 0) {
        const u64 fallback = aggressive ? (pool / 4 * 3) : (pool / 2);
        return (std::min)(pool, (std::max)(fallback, (std::min)(pool, u64{4_GiB})));
    }
    // On multi-heap iGPU drivers (e.g. Windows WDDM APUs with a 512 MiB or 2048 MiB BIOS UMA
    // carveout plus a shared system segment), the UMA carveout was already subtracted from
    // OS-visible physical RAM; add it back to recover total unified machine memory. On single-heap
    // unified drivers (Linux RADV, Apple Silicon MoltenVK), local_heap overlaps host_ram directly.
    const bool separate_uma_carveout =
        shared_heap_bytes != 0 && local_heap_bytes <= (host_ram_bytes <= ~u64{0} / 4
                                                           ? host_ram_bytes * 4
                                                           : ~u64{0});
    const u64 total_unified = separate_uma_carveout && host_ram_bytes <= ~u64{0} - local_heap_bytes
                                  ? host_ram_bytes + local_heap_bytes
                                  : host_ram_bytes;
    u64 share = aggressive ? (total_unified / 8 * 5) : (total_unified / 8 * 3);
    if (separate_uma_carveout) {
        // Dedicated BIOS UMA carveouts (e.g. 8 GiB handhelds or 64-96 GiB unified APUs) are
        // inaccessible to the host OS, so always let the GPU utilize its entire carveout.
        share = (std::max)(share, local_heap_bytes);
    }
    return (std::min)(pool, share);
}

/// Effective live memory budget cap for an integrated GPU, allowing an explicit user-configured
/// texture_cache_memory_limit (in MiB) to expand above the conservative default up to the full
/// unified ceiling (`max_unified_bytes`) plus its reserve margin, bounded by the live driver
/// budget (`driver_budget_bytes`).
[[nodiscard]] constexpr u64 ClampIntegratedGpuBudget(u64 driver_budget_bytes,
                                                     u64 default_budget_bytes,
                                                     u64 max_unified_bytes,
                                                     u32 explicit_limit_mib,
                                                     u64 reserve_bytes) {
    u64 cap = default_budget_bytes;
    if (explicit_limit_mib != 0 && max_unified_bytes > cap) {
        const u64 explicit_bytes = u64{explicit_limit_mib} << 20;
        const u64 needed = explicit_bytes <= ~u64{0} - reserve_bytes
                               ? explicit_bytes + reserve_bytes
                               : ~u64{0};
        cap = (std::max)(cap, (std::min)(max_unified_bytes, needed));
    }
    return cap != 0 ? (std::min)(driver_budget_bytes, cap) : driver_budget_bytes;
}

} // namespace VideoCommon::TextureCacheMemory
