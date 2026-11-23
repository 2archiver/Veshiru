// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// The NVNformat numbering and its grouping follow nxnative dm/gfx/tex/nvn_fmt_mtl.c and
// dm/gfx/nvn/nvn_tex.c.

#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

using VideoCore::Surface::PixelFormat;

constexpr size_t FORMAT_COUNT = 0x9D;

constexpr std::array<std::optional<PixelFormat>, FORMAT_COUNT> MakeTable() {
    std::array<std::optional<PixelFormat>, FORMAT_COUNT> t{};
    t[0x01] = PixelFormat::R8_UNORM;
    t[0x02] = PixelFormat::R8_SNORM;
    t[0x03] = PixelFormat::R8_UINT;
    t[0x04] = PixelFormat::R8_SINT;
    t[0x05] = PixelFormat::R16_FLOAT;
    t[0x06] = PixelFormat::R16_UNORM;
    t[0x07] = PixelFormat::R16_SNORM;
    t[0x08] = PixelFormat::R16_UINT;
    t[0x09] = PixelFormat::R16_SINT;
    t[0x0A] = PixelFormat::R32_FLOAT;
    t[0x0B] = PixelFormat::R32_UINT;
    t[0x0C] = PixelFormat::R32_SINT;
    t[0x0D] = PixelFormat::R8G8_UNORM;
    t[0x0E] = PixelFormat::R8G8_SNORM;
    t[0x0F] = PixelFormat::R8G8_UINT;
    t[0x10] = PixelFormat::R8G8_SINT;
    t[0x11] = PixelFormat::R16G16_FLOAT;
    t[0x12] = PixelFormat::R16G16_UNORM;
    t[0x13] = PixelFormat::R16G16_SNORM;
    t[0x14] = PixelFormat::R16G16_UINT;
    t[0x15] = PixelFormat::R16G16_SINT;
    t[0x16] = PixelFormat::R32G32_FLOAT;
    t[0x17] = PixelFormat::R32G32_UINT;
    t[0x18] = PixelFormat::R32G32_SINT;
    // 0x19-0x21 RGB8 / RGB16: no host counterpart.
    t[0x22] = PixelFormat::R32G32B32_FLOAT;
    // 0x23, 0x24 RGB32 integer: no host counterpart.
    t[0x25] = PixelFormat::A8B8G8R8_UNORM;
    t[0x26] = PixelFormat::A8B8G8R8_SNORM;
    t[0x27] = PixelFormat::A8B8G8R8_UINT;
    t[0x28] = PixelFormat::A8B8G8R8_SINT;
    t[0x29] = PixelFormat::R16G16B16A16_FLOAT;
    t[0x2A] = PixelFormat::R16G16B16A16_UNORM;
    t[0x2B] = PixelFormat::R16G16B16A16_SNORM;
    t[0x2C] = PixelFormat::R16G16B16A16_UINT;
    t[0x2D] = PixelFormat::R16G16B16A16_SINT;
    t[0x2E] = PixelFormat::R32G32B32A32_FLOAT;
    t[0x2F] = PixelFormat::R32G32B32A32_UINT;
    t[0x30] = PixelFormat::R32G32B32A32_SINT;
    t[0x31] = PixelFormat::S8_UINT;
    t[0x32] = PixelFormat::D16_UNORM;
    t[0x33] = PixelFormat::X8_D24_UNORM;
    t[0x34] = PixelFormat::D32_FLOAT;
    t[0x35] = PixelFormat::D24_UNORM_S8_UINT;
    t[0x36] = PixelFormat::D32_FLOAT_S8_UINT;
    t[0x37] = PixelFormat::A8B8G8R8_SRGB; // RGBX8_SRGB
    t[0x38] = PixelFormat::A8B8G8R8_SRGB;
    t[0x39] = PixelFormat::A4B4G4R4_UNORM;     // [UNVERIFIED] bit order
    t[0x3C] = PixelFormat::R5G6B5_UNORM;       // [UNVERIFIED] bit order
    t[0x3D] = PixelFormat::A2B10G10R10_UNORM;
    t[0x3E] = PixelFormat::A2B10G10R10_UINT;
    t[0x3F] = PixelFormat::B10G11R11_FLOAT;
    t[0x40] = PixelFormat::E5B9G9R9_FLOAT;
    t[0x41] = PixelFormat::BC1_RGBA_UNORM; // RGB_DXT1
    t[0x42] = PixelFormat::BC1_RGBA_UNORM;
    t[0x43] = PixelFormat::BC2_UNORM;
    t[0x44] = PixelFormat::BC3_UNORM;
    t[0x45] = PixelFormat::BC1_RGBA_SRGB; // RGB_DXT1_SRGB
    t[0x46] = PixelFormat::BC1_RGBA_SRGB;
    t[0x47] = PixelFormat::BC2_SRGB;
    t[0x48] = PixelFormat::BC3_SRGB;
    t[0x49] = PixelFormat::BC4_UNORM;
    t[0x4A] = PixelFormat::BC4_SNORM;
    t[0x4B] = PixelFormat::BC5_UNORM;
    t[0x4C] = PixelFormat::BC5_SNORM;
    t[0x4D] = PixelFormat::BC7_UNORM;
    t[0x4E] = PixelFormat::BC7_SRGB;
    t[0x4F] = PixelFormat::BC6H_SFLOAT;
    t[0x50] = PixelFormat::BC6H_UFLOAT;
    // 0x51-0x68 *_UI2F / *_I2F: integer storage read as float, no host counterpart.
    // 0x69-0x6C RGB10A2 signed / I2F: no host counterpart.
    t[0x6D] = PixelFormat::A8B8G8R8_UNORM; // RGBX8*
    t[0x6E] = PixelFormat::A8B8G8R8_SNORM;
    t[0x6F] = PixelFormat::A8B8G8R8_UINT;
    t[0x70] = PixelFormat::A8B8G8R8_SINT;
    t[0x71] = PixelFormat::R16G16B16X16_FLOAT;
    t[0x72] = PixelFormat::R16G16B16A16_UNORM;
    t[0x73] = PixelFormat::R16G16B16A16_SNORM;
    t[0x74] = PixelFormat::R16G16B16A16_UINT;
    t[0x75] = PixelFormat::R16G16B16A16_SINT;
    t[0x76] = PixelFormat::R32G32B32A32_FLOAT;
    t[0x77] = PixelFormat::R32G32B32A32_UINT;
    t[0x78] = PixelFormat::R32G32B32A32_SINT;
    t[0x95] = PixelFormat::B5G6R5_UNORM; // BGR565, [UNVERIFIED] bit order
    t[0x99] = PixelFormat::B8G8R8A8_UNORM;
    t[0x9A] = PixelFormat::B8G8R8A8_UNORM;
    t[0x9B] = PixelFormat::B8G8R8A8_SRGB;
    t[0x9C] = PixelFormat::B8G8R8A8_SRGB;
    return t;
}

constexpr auto TABLE = MakeTable();

constexpr std::array<PixelFormat, 14> ASTC_UNORM{
    PixelFormat::ASTC_2D_4X4_UNORM,  PixelFormat::ASTC_2D_5X4_UNORM,
    PixelFormat::ASTC_2D_5X5_UNORM,  PixelFormat::ASTC_2D_6X5_UNORM,
    PixelFormat::ASTC_2D_6X6_UNORM,  PixelFormat::ASTC_2D_8X5_UNORM,
    PixelFormat::ASTC_2D_8X6_UNORM,  PixelFormat::ASTC_2D_8X8_UNORM,
    PixelFormat::ASTC_2D_10X5_UNORM, PixelFormat::ASTC_2D_10X6_UNORM,
    PixelFormat::ASTC_2D_10X8_UNORM, PixelFormat::ASTC_2D_10X10_UNORM,
    PixelFormat::ASTC_2D_12X10_UNORM, PixelFormat::ASTC_2D_12X12_UNORM,
};
constexpr std::array<PixelFormat, 14> ASTC_SRGB{
    PixelFormat::ASTC_2D_4X4_SRGB,  PixelFormat::ASTC_2D_5X4_SRGB,
    PixelFormat::ASTC_2D_5X5_SRGB,  PixelFormat::ASTC_2D_6X5_SRGB,
    PixelFormat::ASTC_2D_6X6_SRGB,  PixelFormat::ASTC_2D_8X5_SRGB,
    PixelFormat::ASTC_2D_8X6_SRGB,  PixelFormat::ASTC_2D_8X8_SRGB,
    PixelFormat::ASTC_2D_10X5_SRGB, PixelFormat::ASTC_2D_10X6_SRGB,
    PixelFormat::ASTC_2D_10X8_SRGB, PixelFormat::ASTC_2D_10X10_SRGB,
    PixelFormat::ASTC_2D_12X10_SRGB, PixelFormat::ASTC_2D_12X12_SRGB,
};



} // Anonymous namespace

std::optional<PixelFormat> NvnToPixelFormat(s32 f) {
    if (f >= 0x79 && f <= 0x94) {
        const size_t k = static_cast<size_t>((f - 0x79) % 14);
        return f < 0x87 ? ASTC_UNORM[k] : ASTC_SRGB[k];
    }
    if (f <= 0 || static_cast<size_t>(f) >= TABLE.size()) {
        return std::nullopt;
    }
    return TABLE[static_cast<size_t>(f)];
}

std::optional<HostFormat> ResolveHostFormat(const Device& device, s32 nvn_format) {
    const std::optional<PixelFormat> pixel_format = NvnToPixelFormat(nvn_format);
    if (!pixel_format) {
        return std::nullopt;
    }
    const auto info = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, *pixel_format);
    if (info.format == VK_FORMAT_UNDEFINED) {
        return std::nullopt;
    }
    HostFormat result{};
    result.format = info.format;
    result.pixel_format = *pixel_format;
    result.attachable = info.attachable;
    result.storage = info.storage;
    // MaxwellToVK::SurfaceFormat swaps in an uncompressed (or differently compressed) format when
    // the device cannot sample the guest's compressed format; those textures cannot take the
    // guest bytes as they are.
    result.needs_transcode =
        (VideoCore::Surface::IsPixelFormatASTC(*pixel_format) && !device.IsOptimalAstcSupported()) ||
        (VideoCore::Surface::IsPixelFormatBCn(*pixel_format) && !device.IsOptimalBcnSupported()) ||
        (VideoCore::Surface::IsPixelFormatETC2(*pixel_format) && !device.IsOptimalEtc2Supported());
    switch (*pixel_format) {
    case PixelFormat::D16_UNORM:
    case PixelFormat::D32_FLOAT:
    case PixelFormat::X8_D24_UNORM:
        result.depth = true;
        break;
    case PixelFormat::S8_UINT:
        result.stencil = true;
        break;
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
    case PixelFormat::D32_FLOAT_S8_UINT:
        result.depth = true;
        result.stencil = true;
        break;
    default:
        break;
    }
    if (result.depth || result.stencil) {
        // Depth/stencil texel layouts differ between hosts; only the formats Vulkan stores the
        // same way can take the guest bytes. Copies and clears between images do not care.
        const bool same_layout =
            (info.format == VK_FORMAT_D16_UNORM && *pixel_format == PixelFormat::D16_UNORM) ||
            (info.format == VK_FORMAT_D32_SFLOAT && *pixel_format == PixelFormat::D32_FLOAT) ||
            (info.format == VK_FORMAT_S8_UINT && *pixel_format == PixelFormat::S8_UINT);
        result.needs_transcode = false;
        result.guest_layout_differs = !same_layout;
    }
    if (!result.storage && !result.needs_transcode && !result.depth && !result.stencil) {
        const VkFormat view = SrgbStorageViewFormat(info.format);
        if (view != VK_FORMAT_UNDEFINED &&
            device.IsFormatSupported(view, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, FormatType::Optimal)) {
            result.storage_view = view;
        }
        using namespace VideoCore::Surface;
        if (DefaultBlockWidth(*pixel_format) == 1 && DefaultBlockHeight(*pixel_format) == 1) {
            const VkFormat alias = SameSizeUintFormat(BytesPerBlock(*pixel_format));
            result.storage_alias =
                alias != VK_FORMAT_UNDEFINED &&
                device.IsFormatSupported(alias, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                         FormatType::Optimal);
        }
    }
    return result;
}

AstcStandin ChooseAstcStandin(const char* environment, Settings::AstcRecompression setting) {
    if (environment != nullptr) {
        const std::string_view value{environment};
        if (value == "rgba8") {
            return AstcStandin::Rgba8;
        }
        if (value == "bc1") {
            return AstcStandin::Bc1;
        }
        if (value == "bc3") {
            return AstcStandin::Bc3;
        }
        if (value == "bc7") {
            return AstcStandin::Bc7;
        }
    }
    switch (setting) {
    case Settings::AstcRecompression::Bc1:
        return AstcStandin::Bc1;
    case Settings::AstcRecompression::Bc3:
        return AstcStandin::Bc3;
    default:
        return AstcStandin::Rgba8;
    }
}

const char* AstcStandinName(AstcStandin standin) {
    switch (standin) {
    case AstcStandin::Bc1:
        return "BC1";
    case AstcStandin::Bc3:
        return "BC3";
    case AstcStandin::Bc7:
        return "BC7";
    case AstcStandin::Bc4:
        return "BC4";
    case AstcStandin::Bc5:
        return "BC5";
    default:
        return "RGBA8";
    }
}

SmartBc ParseSmartBc(AstcStandin standin, bool setting, const char* smart, const char* bc1,
                     const char* bc45, const char* refine) {
    const auto on = [](const char* value) {
        return value == nullptr || std::string_view{value} != "0";
    };
    // The refined endpoints (+1.9 dB for 1.8x the encode time) are opt-in: BC_REFINE=1.
    const auto on_if_named = [](const char* value) {
        return value != nullptr && std::string_view{value} != "0";
    };
    SmartBc result{};
    result.enabled = standin == AstcStandin::Bc3 &&
                     (smart == nullptr ? setting : std::string_view{smart} != "0");
    result.bc1 = result.enabled && on(bc1);
    result.bc45 = result.enabled && on(bc45);
    result.refine = result.enabled && on_if_named(refine);
    return result;
}

std::string SmartBcName(const SmartBc& smart) {
    if (!smart.enabled) {
        return "off";
    }
    std::string parts;
    for (const auto& [part, name] : {std::pair{smart.bc1, "BC1"}, std::pair{smart.bc45, "BC4/BC5"},
                                     std::pair{smart.refine, "refined endpoints"}}) {
        if (part) {
            parts += parts.empty() ? name : std::string{", "} + name;
        }
    }
    return parts.empty() ? std::string{"on (BC3 only)"} : "on (" + parts + ")";
}

AstcStandin SmartStandinFor(Tegra::Texture::ASTC::Channels channels, bool srgb,
                            const SmartBc& smart) {
    using Tegra::Texture::ASTC::Channels;
    if (!smart.enabled) {
        return AstcStandin::Bc3;
    }
    switch (channels) {
    case Channels::Luminance:
        return smart.bc45 && !srgb ? AstcStandin::Bc4
               : smart.bc1         ? AstcStandin::Bc1
                                   : AstcStandin::Bc3;
    case Channels::LuminanceAlpha:
        return smart.bc45 && !srgb ? AstcStandin::Bc5 : AstcStandin::Bc3;
    case Channels::Rgb:
        return smart.bc1 ? AstcStandin::Bc1 : AstcStandin::Bc3;
    default:
        return AstcStandin::Bc3;
    }
}

bool StandinHolds(AstcStandin standin, Tegra::Texture::ASTC::Channels channels) {
    using Tegra::Texture::ASTC::Channels;
    switch (standin) {
    case AstcStandin::Bc1:
        return channels == Channels::Luminance || channels == Channels::Rgb;
    case AstcStandin::Bc4:
        return channels == Channels::Luminance;
    case AstcStandin::Bc5:
        return channels == Channels::Luminance || channels == Channels::LuminanceAlpha;
    default:
        return true;
    }
}

HostFormat ToBcStandin(const HostFormat& host, AstcStandin standin) {
    if (!host.needs_transcode || standin == AstcStandin::Rgba8 ||
        !VideoCore::Surface::IsPixelFormatASTC(host.pixel_format)) {
        return host;
    }
    const bool srgb = VideoCore::Surface::IsPixelFormatSRGB(host.pixel_format);
    HostFormat result = host;
    switch (standin) {
    case AstcStandin::Bc1:
        result.format = srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        break;
    case AstcStandin::Bc7:
        result.format = srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
        break;
    case AstcStandin::Bc4:
        result.format = srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC4_UNORM_BLOCK;
        break;
    case AstcStandin::Bc5:
        result.format = srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK;
        break;
    default:
        result.format = srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
        break;
    }
    result.bc_standin = true;
    result.attachable = false;
    result.storage = false;
    result.storage_view = VK_FORMAT_UNDEFINED;
    result.storage_alias = false;
    return result;
}

AstcStandin BcStandinOf(const HostFormat& host) {
    if (!host.bc_standin) {
        return AstcStandin::Rgba8;
    }
    switch (host.format) {
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        return AstcStandin::Bc1;
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
        return AstcStandin::Bc3;
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return AstcStandin::Bc7;
    case VK_FORMAT_BC4_UNORM_BLOCK:
        return AstcStandin::Bc4;
    case VK_FORMAT_BC5_UNORM_BLOCK:
        return AstcStandin::Bc5;
    default:
        return AstcStandin::Rgba8;
    }
}

u32 BcStandinBlockBytes(const HostFormat& host) {
    switch (BcStandinOf(host)) {
    case AstcStandin::Bc1:
    case AstcStandin::Bc4:
        return 8;
    case AstcStandin::Bc3:
    case AstcStandin::Bc5:
    case AstcStandin::Bc7:
        return 16;
    default:
        return 0;
    }
}

bool ViewsBcStandin(VkFormat format, VkFormat standin) {
    const auto bc1 = [](VkFormat f) {
        return f == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || f == VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    };
    const auto bc3 = [](VkFormat f) {
        return f == VK_FORMAT_BC3_UNORM_BLOCK || f == VK_FORMAT_BC3_SRGB_BLOCK;
    };
    const auto bc7 = [](VkFormat f) {
        return f == VK_FORMAT_BC7_UNORM_BLOCK || f == VK_FORMAT_BC7_SRGB_BLOCK;
    };
    const auto same = [&](VkFormat f) {
        return f == standin &&
               (f == VK_FORMAT_BC4_UNORM_BLOCK || f == VK_FORMAT_BC5_UNORM_BLOCK);
    };
    return (bc1(format) && bc1(standin)) || (bc3(format) && bc3(standin)) ||
           (bc7(format) && bc7(standin)) || same(format);
}

VkFormat StandinViewFormat(const HostFormat& texture, const HostFormat& storage) {
    const AstcStandin kind = BcStandinOf(storage);
    const bool srgb = VideoCore::Surface::IsPixelFormatSRGB(texture.pixel_format);
    switch (kind) {
    case AstcStandin::Bc1:
        return srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case AstcStandin::Bc3:
        return srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
    case AstcStandin::Bc7:
        return srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    case AstcStandin::Bc4:
    case AstcStandin::Bc5:
        return srgb ? VK_FORMAT_UNDEFINED : storage.format;
    default:
        return storage.format;
    }
}

std::array<VkComponentSwizzle, 4> ComposeStandinSwizzle(
    const std::array<VkComponentSwizzle, 4>& guest, AstcStandin standin) {
    if (standin != AstcStandin::Bc4 && standin != AstcStandin::Bc5) {
        return guest;
    }
    // What each RGBA8 channel of the decoded (L, L, L, A) is in the BC image.
    const VkComponentSwizzle alpha =
        standin == AstcStandin::Bc4 ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_G;
    std::array<VkComponentSwizzle, 4> result{};
    for (size_t c = 0; c < 4; ++c) {
        switch (guest[c]) {
        case VK_COMPONENT_SWIZZLE_R:
        case VK_COMPONENT_SWIZZLE_G:
        case VK_COMPONENT_SWIZZLE_B:
            result[c] = VK_COMPONENT_SWIZZLE_R;
            break;
        case VK_COMPONENT_SWIZZLE_A:
            result[c] = alpha;
            break;
        case VK_COMPONENT_SWIZZLE_IDENTITY:
            result[c] = c == 3 ? alpha : VK_COMPONENT_SWIZZLE_R;
            break;
        default:
            result[c] = guest[c]; // ZERO, ONE
            break;
        }
    }
    return result;
}

VkFormat SameSizeUintFormat(u32 bytes) {
    switch (bytes) {
    case 1:
        return VK_FORMAT_R8_UINT;
    case 2:
        return VK_FORMAT_R16_UINT;
    case 4:
        return VK_FORMAT_R32_UINT;
    case 8:
        return VK_FORMAT_R32G32_UINT;
    case 16:
        return VK_FORMAT_R32G32B32A32_UINT;
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

VkFormat SrgbStorageViewFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_UNORM;
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

} // namespace Vulkan::Nvn
