// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <string>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/surface.h"
#include "video_core/textures/astc.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// The `VideoCore::Surface::PixelFormat` an NVNformat value maps to, or nullopt when the NVN format
/// has no counterpart (three-component formats, *_I2F integer-to-float reads, ...).
[[nodiscard]] std::optional<VideoCore::Surface::PixelFormat> NvnToPixelFormat(s32 nvn_format);

[[nodiscard]] constexpr bool IsAstc8x8(VideoCore::Surface::PixelFormat format) {
    using VideoCore::Surface::PixelFormat;
    return format == PixelFormat::ASTC_2D_8X8_UNORM || format == PixelFormat::ASTC_2D_8X8_SRGB;
}

struct HostFormat {
    VkFormat format{VK_FORMAT_UNDEFINED};
    VideoCore::Surface::PixelFormat pixel_format{};
    bool attachable{};
    bool storage{};
    /// The device cannot hold the format natively (ASTC on a desktop GPU, BCn on mobile ...) and
    /// `format` is an uncompressed stand-in. The replay decodes the guest's blocks on the CPU when
    /// it uploads the texture (nvn_upload.cpp); nothing else can read or write the texture's guest
    /// bytes through the image (a texture-to-buffer copy would have to re-encode).
    bool needs_transcode{};
    /// An ASTC stand-in held as BC1/BC3/BC7 (`format`) instead of RGBA8 (AstcStandin): the replay
    /// decodes the blocks to RGBA8 in a scratch image and re-encodes them (nvn_bc_encode.h). Set
    /// per storage by the texture creation (ToBcStandin), never by ResolveHostFormat.
    bool bc_standin{};
    bool depth{};
    bool stencil{};
    /// Depth/stencil whose host texel layout is not the guest's (D24S8, D32S8): copies between
    /// two images and clears are fine, guest bytes cannot be uploaded as they are.
    bool guest_layout_differs{};
    /// The format of the storage-image views of a texture whose own format has no storage support
    /// (VK_FORMAT_UNDEFINED otherwise, or when the device cannot store that format either). NVN
    /// images of an sRGB texture read and write the texel bits as they are, with no sRGB
    /// conversion (the same-size UNORM format); Vulkan expresses that as a UNORM view of an image
    /// created with STORAGE usage and VK_IMAGE_CREATE_EXTENDED_USAGE_BIT.
    VkFormat storage_view{VK_FORMAT_UNDEFINED};
    /// The format cannot be stored but an NVN texture view of another format of the same texel
    /// size can bind its storage as an image (OoT binds an R32UI view of an RGB9E5 texture and
    /// packs the shared exponent itself). Such textures are created with STORAGE usage and
    /// VK_IMAGE_CREATE_EXTENDED_USAGE_BIT, as the views need it; set when the unsigned integer
    /// format of the texel size (`SameSizeUintFormat`) can be stored by the device.
    bool storage_alias{};
};

/// The unsigned integer colour format with `bytes` per texel (R8/R16/R32/R32G32/R32G32B32A32
/// UINT), VK_FORMAT_UNDEFINED for any other size. Every uncompressed colour format of that size
/// is in its Vulkan compatibility class.
[[nodiscard]] VkFormat SameSizeUintFormat(u32 bytes);

/// The UNORM format with the bits of an 8-bit-per-channel sRGB colour format, VK_FORMAT_UNDEFINED
/// for every other format.
[[nodiscard]] VkFormat SrgbStorageViewFormat(VkFormat format);

/// NVNformat -> PixelFormat -> `MaxwellToVK::SurfaceFormat`. Returns nullopt when unsupported.
[[nodiscard]] std::optional<HostFormat> ResolveHostFormat(const Device& device, s32 nvn_format);

/// What an ASTC stand-in holds on a device without ASTC: the decoded RGBA8 texels, or those
/// re-encoded to BC1 / BC3 / BC7 (an eighth / a quarter / a quarter of the RGBA8 memory). BC4
/// (an eighth: the luminance) and BC5 (a quarter: luminance and alpha) are only chosen per texture
/// by the smart selection (SmartBc).
enum class AstcStandin : u8 { Rgba8, Bc1, Bc3, Bc7, Bc4, Bc5 };
inline constexpr size_t ASTC_STANDIN_COUNT = 6;

/// `EDEN_NX2_NVN_ASTC_STANDIN` (`rgba8`, `bc1`, `bc3`, `bc7`; null or anything else: not set)
/// over the astc_recompression setting (which has no BC7).
[[nodiscard]] AstcStandin ChooseAstcStandin(const char* environment,
                                            Settings::AstcRecompression setting);

[[nodiscard]] const char* AstcStandinName(AstcStandin standin);

/// Smart BC stand-ins: with the BC3 stand-in kind (the default), each ASTC texture is held in the
/// BC format its channels need (Tegra::Texture::ASTC::ClassifyChannels): luminance as BC4,
/// luminance+alpha as BC5 (L, A), opaque colour as BC1 and the rest as BC3. Refining the BC1/BC3
/// colour endpoints is opt-in: off unless EDEN_NX2_NVN_BC_REFINE is set (to anything but 0).
/// Each part has a switch of its own, to bisect a visual bug.
struct SmartBc {
    bool enabled{};  ///< per-texture formats (EDEN_NX2_NVN_SMART_BC, default on with BC3)
    bool bc1{};      ///< opaque textures as BC1 (EDEN_NX2_NVN_SMART_BC1)
    bool bc45{};     ///< luminance(+alpha) textures as BC4/BC5 (EDEN_NX2_NVN_SMART_BC45)
    bool refine{};   ///< refined BC1/BC3 colour endpoints (EDEN_NX2_NVN_BC_REFINE)
};

/// The smart selection for stand-in kind `standin` from the four switches above (null: not set;
/// `0` turns one off). Only the BC3 kind enables it; `0` in `smart` disables every part.
/// `setting` is Settings::values.nvn_smart_bc (the GUI row); each env switch overrides: "0" off,
/// "1" on, null: the setting (the first one) or the part's own default (the others).
[[nodiscard]] SmartBc ParseSmartBc(AstcStandin standin, bool setting, const char* smart,
                                   const char* bc1, const char* bc45, const char* refine);

/// Startup log form of `smart`: "off" or the parts that are on.
[[nodiscard]] std::string SmartBcName(const SmartBc& smart);

/// The stand-in of an ASTC texture whose blocks decode to `channels`: Bc4 / Bc5 / Bc1 / Bc3 as
/// `smart` allows. sRGB textures have no BC4/BC5 form: luminance goes to BC1 or BC3 there.
[[nodiscard]] AstcStandin SmartStandinFor(Tegra::Texture::ASTC::Channels channels, bool srgb,
                                          const SmartBc& smart);

/// Whether a stand-in of kind `standin` holds every texel of a texture in class `channels`
/// (a re-upload into it of another class must recreate it, Recorder::RefitStandin).
[[nodiscard]] bool StandinHolds(AstcStandin standin, Tegra::Texture::ASTC::Channels channels);

/// `host` (an RGBA8 ASTC stand-in) held as `standin`'s BC format, with the sRGB-ness of the ASTC
/// format; `host` unchanged when it is no ASTC stand-in or `standin` is Rgba8. An sRGB format asked
/// for as BC4 / BC5 becomes BC1 / BC3.
[[nodiscard]] HostFormat ToBcStandin(const HostFormat& host, AstcStandin standin);

/// What a BC stand-in holds (by its format), Rgba8 for any other host format.
[[nodiscard]] AstcStandin BcStandinOf(const HostFormat& host);

/// Bytes per 4x4 block of a BC stand-in (8 for BC1 and BC4, 16 for BC3, BC5 and BC7), 0 for any
/// other format.
[[nodiscard]] u32 BcStandinBlockBytes(const HostFormat& host);

/// Whether `format` can view a BC stand-in of format `standin` (the same BC format, sRGB or not).
[[nodiscard]] bool ViewsBcStandin(VkFormat format, VkFormat standin);

/// The format a texture object of host format `texture` (its own sRGB-ness) views a BC stand-in
/// storage of host format `storage` with: the storage's BC format, sRGB as the texture's ASTC
/// format; VK_FORMAT_UNDEFINED when that does not exist (an sRGB view of a BC4/BC5 stand-in).
[[nodiscard]] VkFormat StandinViewFormat(const HostFormat& texture, const HostFormat& storage);

/// The view swizzle that reads a stand-in of kind `standin` as the game reads the RGBA8 stand-in
/// through `guest` (a swizzle of R, G, B, A, ZERO and ONE, no IDENTITY): the ASTC decode of a
/// BC4 / BC5 texture is (L, L, L, 1) / (L, L, L, A), held as R / (R = L, G = A), so each
/// component of `guest` is composed with (R, R, R, ONE) / (R, R, R, G). Other kinds keep `guest`.
[[nodiscard]] std::array<VkComponentSwizzle, 4> ComposeStandinSwizzle(
    const std::array<VkComponentSwizzle, 4>& guest, AstcStandin standin);

} // namespace Vulkan::Nvn
