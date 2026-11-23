// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>

#include "common/common_types.h"
#include "common/settings.h"
#include "video_core/nvn/sink.h"
#include "video_core/textures/texture.h"
#include "video_core/vulkan_common/vulkan.h"

// The Anisotropic Filtering setting on the VTX-HLE replay's samplers. The rule is the emulated
// sampler's (Tegra::Texture::TSCEntry::MaxAnisotropy, shared through BoostedAnisotropy), applied to
// the NVN sampler's own values:
//
//  - Default adds nothing: the sampler keeps the guest's max anisotropy untouched.
//  - A sampler that asked for anisotropy itself (guest 2x or more) gets the setting's doublings on
//    top (guest 4x + X16's four steps = 16x after the clamp).
//  - A sampler that asked for none only gets the setting when it filters mipmapped (min >= 2), uses
//    the whole LOD range, has a linear min filter and plain weighted-average reduction, and is not
//    a depth-compare sampler. Nearest/point, clamped-LOD, shadow and min/max reduction samplers
//    stay at 1x.
//  - Per view, as the emulated Sampler::VariantKey::reduce_anisotropy: a view the emulated path's
//    ImageViewBase::SupportsAnisotropy refuses (a single mip, a non-2D view, one of its formats)
//    is sampled with the guest's anisotropy, not the boosted one.
//  - The result is clamped to [1, maxSamplerAnisotropy]; the samplerAnisotropy feature is
//    mandatory for the device (FOR_EACH_VK_MANDATORY_FEATURE).
namespace Vulkan::Nvn {

/// How many log2 steps the Anisotropic Filtering setting adds to a sampler's anisotropy, as
/// TSCEntry::MaxAnisotropy applies it: Default adds none, 2x..16x add 1..4, Automatic follows the
/// resolution scale (a 2x scale adds one step, a 3x scale two, ...).
[[nodiscard]] constexpr s32 AnisotropyBoostSteps(Settings::AnisotropyMode mode,
                                                 u32 resolution_scale) noexcept {
    switch (mode) {
    case Settings::AnisotropyMode::Default:
    case Settings::AnisotropyMode::X2:
    case Settings::AnisotropyMode::X4:
    case Settings::AnisotropyMode::X8:
    case Settings::AnisotropyMode::X16:
        return s32(mode) - 1;
    case Settings::AnisotropyMode::Automatic:
        return resolution_scale > 1U ? static_cast<s32>(resolution_scale - 1U) : 0;
    }
    return 0;
}

/// The same for the current settings (`Settings::values.max_anisotropy` and the resolution scale).
[[nodiscard]] inline s32 CurrentAnisotropyBoostSteps() noexcept {
    const u32 resolution_scale = Settings::values.resolution_info.up_scale >>
                                 Settings::values.resolution_info.down_shift;
    return AnisotropyBoostSteps(Settings::values.max_anisotropy.GetValue(), resolution_scale);
}

/// Whether a sampler that asks for no anisotropy of its own may get some from the setting: it
/// filters mipmapped, uses the whole LOD range, is bilinear on min (and a plain weighted average,
/// not min/max reduction) and is not a depth-compare sampler.
[[nodiscard]] constexpr bool SamplerAcceptsAddedAnisotropy(bool mipmapped, bool full_lod_range,
                                                           bool min_linear,
                                                           bool weighted_average,
                                                           bool depth_compare) noexcept {
    return mipmapped && full_lod_range && min_linear && weighted_average && !depth_compare;
}

/// The anisotropy a sampler is built with: the guest's value (1 when it asked for none) with
/// `steps` doublings added. A sampler that asked for none (guest below 2x) only gets the added
/// steps when `accepts_added` is true. With no steps the guest value is returned untouched. Not
/// clamped to the device limit.
[[nodiscard]] inline f32 BoostedAnisotropy(f32 guest, bool accepts_added, s32 steps) noexcept {
    if (steps <= 0 || (guest < 2.0f && !accepts_added)) {
        return guest;
    }
    // A NaN or huge guest value must not reach the u32 conversion.
    const f32 normalized = std::isfinite(guest) ? std::clamp(guest, 1.0f, 65536.0f) : 1.0f;
    const u32 base = std::bit_floor(static_cast<u32>(normalized));
    return std::ldexp(static_cast<f32>(base), steps);
}

/// The max anisotropy NVN sampler `d` is built with when the setting adds `steps` doublings; the
/// guest's own value (unclamped) for steps <= 0.
[[nodiscard]] inline f32 SamplerAnisotropy(const VideoCommon::Nvn::SamplerDesc& d, s32 steps) {
    const bool accepts_added = SamplerAcceptsAddedAnisotropy(
        d.min_filter >= 2,   // NVN minFilter NEAREST_MIPMAP_NEAREST and up
        d.lod_min < 1.0f / 256.0f && d.lod_max >= 1.0f, // TSC min LOD clamp 0, max clamp >= 256
        (d.min_filter & 1) != 0, d.reduction == 0, d.compare_mode == 1);
    return BoostedAnisotropy(d.max_anisotropy, accepts_added, steps);
}

/// Whether the emulated path (ImageViewBase::SupportsAnisotropy) lets a view of `format`/`type`
/// with `levels` mip levels be sampled with the setting's anisotropy.
[[nodiscard]] inline bool ViewSupportsAnisotropy(VkFormat format, VkImageViewType type,
                                                 u32 levels) {
    if (levels <= 1 ||
        (type != VK_IMAGE_VIEW_TYPE_2D && type != VK_IMAGE_VIEW_TYPE_2D_ARRAY)) {
        return false;
    }
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R4G4_UNORM_PACK8:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return false;
    default:
        return true;
    }
}

/// The anisotropy fields of the VkSamplerCreateInfo of NVN sampler `d` built with `steps`
/// doublings (SamplerStepsKey's result). Steps 0 is the guest's own sampler: its value, clamped
/// to 16 only, exactly as before the setting existed. With steps the result is also clamped to the
/// device's `max_limit`.
struct SamplerAnisotropyFields {
    VkBool32 enable;
    f32 max_anisotropy;
};

[[nodiscard]] inline SamplerAnisotropyFields SamplerAnisotropyOf(
    const VideoCommon::Nvn::SamplerDesc& d, s32 steps, f32 max_limit) {
    const f32 value = steps > 0 ? SamplerAnisotropy(d, steps) : d.max_anisotropy;
    return {static_cast<VkBool32>(value > 1.0f),
            std::clamp(value, 1.0f, steps > 0 ? max_limit : 16.0f)};
}

/// The doublings a VkSampler of NVN sampler `d` is keyed and built with when the setting adds
/// `steps` and the device allows `max_limit`: `steps` when they change the sampler's anisotropy,
/// else 0, the guest's own sampler. One VkSampler then serves every setting a sampler ignores
/// (nearest/point, shadow, clamped-LOD samplers), and two settings that change it never share one.
[[nodiscard]] inline s32 SamplerStepsKey(const VideoCommon::Nvn::SamplerDesc& d, s32 steps,
                                         f32 max_limit) {
    if (steps > 0 && std::clamp(SamplerAnisotropy(d, steps), 1.0f, max_limit) !=
                         std::clamp(d.max_anisotropy, 1.0f, max_limit)) {
        return steps;
    }
    return 0;
}

} // namespace Vulkan::Nvn
