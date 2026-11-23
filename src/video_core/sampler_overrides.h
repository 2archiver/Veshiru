// SPDX-FileCopyrightText: Copyright 2026 Duck912
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

/// Sampler diagnostics shared by the legacy texture cache and the NVN HLE replay. All of them are
/// read from the environment once, when the first sampler is created, and cost nothing per draw.
///
///  - `EDEN_NX2_FORCE_FILTER=linear|point|nearest`: every guest sampler gets that filter (mag, min and
///    mip). `linear` also drops the guest's LOD clamp (min 0, max none) so minified art uses its
///    mips; `point` / `nearest` forces nearest filtering and anisotropy 1.
///  - `EDEN_NX2_FORCE_ANISO=<n>`: max anisotropy n (1..16) on every sampler whose min and mag
///    filters end up linear.
///  - `EDEN_NX2_SAMPLER_DIAGNOSTICS`: the first 64 created samplers are logged.
///
/// The filter override idea and the LOD-clamp drop come from Duck912's sampler diagnostics.
namespace VideoCore::SamplerOverrides {

enum class ForcedFilter : u32 { None, Linear, Point };

struct Config {
    ForcedFilter filter{};
    u32 anisotropy{}; ///< 0: not forced
    bool diagnostics{};
};

[[nodiscard]] inline const Config& Get() {
    static const Config config = [] {
        Config result;
        if (const char* filter = std::getenv("EDEN_NX2_FORCE_FILTER"); filter != nullptr) {
            result.filter = (std::strcmp(filter, "point") == 0 || std::strcmp(filter, "nearest") == 0)
                                ? ForcedFilter::Point
                                : ForcedFilter::Linear;
        }
        if (const char* aniso = std::getenv("EDEN_NX2_FORCE_ANISO"); aniso != nullptr) {
            // Invalid or non-positive values leave the override off.
            char* end = nullptr;
            const long value = std::strtol(aniso, &end, 10);
            if (end != aniso && *end == '\0' && value > 0) {
                result.anisotropy = static_cast<u32>(std::min(value, 16L));
            }
        }
        result.diagnostics = std::getenv("EDEN_NX2_SAMPLER_DIAGNOSTICS") != nullptr;
        return result;
    }();
    return config;
}

/// True for the first 64 calls once `EDEN_NX2_SAMPLER_DIAGNOSTICS` is set.
[[nodiscard]] inline bool ShouldLog() {
    if (!Get().diagnostics) {
        return false;
    }
    static std::atomic<u32> logged{};
    return logged.fetch_add(1, std::memory_order_relaxed) < 64;
}

/// Applies the filter and anisotropy overrides to a sampler's filters; returns whether it changed
/// anything it was configured to change (the sampler counts as forced).
inline bool ApplyFilters(VkFilter& mag, VkFilter& min, VkSamplerMipmapMode& mip, f32& anisotropy) {
    const Config& config = Get();
    bool forced = false;
    if (config.filter != ForcedFilter::None) {
        const bool point = config.filter == ForcedFilter::Point;
        mag = point ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        min = mag;
        mip = point ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        // Vulkan requires linear min/mag filters whenever anisotropy is enabled.
        if (point) {
            anisotropy = 1.0f;
        }
        forced = true;
    }
    if (config.anisotropy != 0 && mag == VK_FILTER_LINEAR && min == VK_FILTER_LINEAR) {
        anisotropy = static_cast<f32>(config.anisotropy);
        forced = true;
    }
    return forced;
}

/// `EDEN_NX2_FORCE_FILTER=linear`: the guest's LOD clamp is dropped for every forced sampler (tests
/// whether it is what leaves minified art on mip 0); point mode keeps it. Returns whether the range
/// was replaced.
inline bool ApplyLod(f32& min_lod, f32& max_lod) {
    if (Get().filter != ForcedFilter::Linear) {
        return false;
    }
    min_lod = 0.0f;
    max_lod = VK_LOD_CLAMP_NONE;
    return true;
}

} // namespace VideoCore::SamplerOverrides
