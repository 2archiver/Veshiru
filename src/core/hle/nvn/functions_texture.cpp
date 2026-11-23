// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <unordered_map>
#include <vector>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/sink.h"
#include "core/hle/service/nx2_native_pool.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/textures/decoders.h"

// Texture family: NVNtextureBuilder (kept in place, nxnative `tex_builder_t`), NVNtexture (host
// state in the Recorder, tagged at +0), NVNtextureView (kept in place, `GuestTextureView`).
namespace Core::NvnHle::Hle {
namespace {

namespace Layout = ::Vulkan::Nvn;
using ::VideoCommon::Nvn::GUEST_TAG_MAGIC;
using ::VideoCommon::Nvn::GuestTag;
using ::VideoCommon::Nvn::GuestTextureView;
using ::VideoCommon::Nvn::ObjectKind;
using ::VideoCommon::Nvn::TextureDesc;

// ---- NVNtextureBuilder in place (nxnative nvn_obj.h tex_builder_t, 0x60 bytes) ----------------

/// nxnative `nvn_texdesc` (nvn_tex.h), the part of a texture that determines its storage.
struct GuestTextureDesc {
    s32 target;
    s32 format;
    s32 flags;
    s32 samples;
    u32 width;
    u32 height;
    u32 depth;
    u32 levels;
    u32 stride;
    s32 pkg_block_h; ///< packaged layout byte 0 + 1, 0 = automatic
};
static_assert(sizeof(GuestTextureDesc) == 40);

/// `tex_builder_t`. `gl_name` takes the 4 padding bytes at the end (nxnative has no GL interop).
struct GuestTextureBuilder {
    u64 device;
    GuestTextureDesc desc;
    u64 pool;   ///< guest VA of the NVNmemoryPool object (0 = none)
    s64 offset; ///< byte offset of the storage in the pool
    u8 has_layout;
    u8 layout[8];
    u64 pkg_data; ///< guest VA of the packaged texture data
    u8 bits;      ///< set_swizzle:1, stencil:1
    u16 swizzle;  ///< r | g << 3 | b << 6 | a << 9
    u32 gl_name;
};
static_assert(offsetof(GuestTextureBuilder, desc) == 0x08);
static_assert(offsetof(GuestTextureBuilder, pool) == 0x30);
static_assert(offsetof(GuestTextureBuilder, offset) == 0x38);
static_assert(offsetof(GuestTextureBuilder, has_layout) == 0x40);
static_assert(offsetof(GuestTextureBuilder, layout) == 0x41);
static_assert(offsetof(GuestTextureBuilder, pkg_data) == 0x50);
static_assert(offsetof(GuestTextureBuilder, bits) == 0x58);
static_assert(offsetof(GuestTextureBuilder, swizzle) == 0x5A);
static_assert(offsetof(GuestTextureBuilder, gl_name) == 0x5C);
static_assert(sizeof(GuestTextureBuilder) == 0x60);

constexpr u64 TbDevice = offsetof(GuestTextureBuilder, device);
constexpr u64 TbDesc = offsetof(GuestTextureBuilder, desc);
constexpr u64 TbTarget = TbDesc + offsetof(GuestTextureDesc, target);
constexpr u64 TbFormat = TbDesc + offsetof(GuestTextureDesc, format);
constexpr u64 TbFlags = TbDesc + offsetof(GuestTextureDesc, flags);
constexpr u64 TbSamples = TbDesc + offsetof(GuestTextureDesc, samples);
constexpr u64 TbWidth = TbDesc + offsetof(GuestTextureDesc, width);
constexpr u64 TbHeight = TbDesc + offsetof(GuestTextureDesc, height);
constexpr u64 TbDepth = TbDesc + offsetof(GuestTextureDesc, depth);
constexpr u64 TbLevels = TbDesc + offsetof(GuestTextureDesc, levels);
constexpr u64 TbStride = TbDesc + offsetof(GuestTextureDesc, stride);
constexpr u64 TbPkgBlockH = TbDesc + offsetof(GuestTextureDesc, pkg_block_h);
constexpr u64 TbPool = offsetof(GuestTextureBuilder, pool);
constexpr u64 TbOffset = offsetof(GuestTextureBuilder, offset);
constexpr u64 TbHasLayout = offsetof(GuestTextureBuilder, has_layout);
constexpr u64 TbLayout = offsetof(GuestTextureBuilder, layout);
constexpr u64 TbPkgData = offsetof(GuestTextureBuilder, pkg_data);
constexpr u64 TbBits = offsetof(GuestTextureBuilder, bits);
constexpr u64 TbSwizzle = offsetof(GuestTextureBuilder, swizzle);
constexpr u64 TbGlName = offsetof(GuestTextureBuilder, gl_name);

constexpr u8 BitSetSwizzle = 1 << 0;
constexpr u8 BitStencil = 1 << 1;

constexpr s32 TargetBuffer = 10;
constexpr s32 TARGET_2D = 1;
constexpr s32 SWIZZLE_R = 2;
constexpr s32 SWIZZLE_G = 3;
constexpr s32 SWIZZLE_B = 4;
constexpr s32 SWIZZLE_A = 5;
constexpr u16 IDENTITY_SWIZZLE = SWIZZLE_R | SWIZZLE_G << 3 | SWIZZLE_B << 6 | SWIZZLE_A << 9;
constexpr s32 DEPTH_STENCIL_MODE_STENCIL = 1;
constexpr s32 FLAGS_LINEAR_ANY = 0x50; // LINEAR | LINEAR_RENDER_TARGET

u16 PackSwizzle(s32 r, s32 g, s32 b, s32 a) {
    return static_cast<u16>((r & 7) | (g & 7) << 3 | (b & 7) << 6 | (a & 7) << 9);
}

/// `NVNtextureSparseTileLayout`: {numTiledLevels, tileWidth, tileHeight, tileDepth, reserved[16]}.
struct GuestSparseTileLayout {
    s32 num_tiled_levels;
    s32 tile_width;
    s32 tile_height;
    s32 tile_depth;
    u8 reserved[16];
};
static_assert(sizeof(GuestSparseTileLayout) == 32);

/// `NVNcopyRegion`.
struct GuestCopyRegion {
    s32 x;
    s32 y;
    s32 z;
    s32 width;
    s32 height;
    s32 depth;
};
static_assert(sizeof(GuestCopyRegion) == 24);

/// Both `GuestTextureDesc` and the sink's `TextureDesc` carry these fields under the same names.
template <typename Desc>
Layout::TextureLayoutDesc ToLayout(const Desc& d) {
    return {d.target, d.format, d.flags, d.samples, d.width, d.height, d.depth, d.levels, d.stride,
            d.pkg_block_h};
}

// ---- Sparse tile layout (nxnative nvn_tex_sparse_tiles; nvn_texture_layout.h has no port) --------

u32 Mip(u32 size, u32 level) {
    if (level >= 32) {
        return 1;
    }
    size >>= level;
    return size ? size : 1;
}

u32 Blk(u32 size, u32 block) {
    return (size + block - 1) / block;
}

int SparseTileHeight(u32 bytes) {
    return bytes == 1 ? 5 : (bytes == 2 || bytes == 4) ? 4 : (bytes == 8 || bytes == 16) ? 3 : -1;
}

GuestSparseTileLayout ComputeSparseTileLayout(const Layout::TextureLayoutDesc& d) {
    GuestSparseTileLayout out{};
    const Layout::BlockInfo t = Layout::FormatBlockInfo(d.format);
    const bool sparse = (d.flags & Layout::TexFlag::Sparse) && !(d.flags & FLAGS_LINEAR_ANY) &&
                        d.target != TargetBuffer;
    if (!sparse || !t.bytes) {
        return out;
    }
    // Dimensions per layer as in nxnative `shape`: 1D and 1D arrays 1, 3D 3, the rest 2.
    const u32 dims = (d.target == 0 || d.target == 3) ? 1 : d.target == 2 ? 3 : 2;
    const u32 w = d.width;
    const u32 h = dims == 1 ? 1 : d.height;
    const u32 z = d.depth ? d.depth : 1;
    u32 sx = 1;
    u32 sy = 1;
    if (d.samples == 2) {
        sx = 2;
    } else if (d.samples == 4) {
        sx = sy = 2;
    } else if (d.samples == 8) {
        sx = 4;
        sy = 2;
    }
    const Layout::TextureLayout layout = Layout::ComputeTextureLayout(d);
    const u32 sh = (dims == 2 && !d.pkg_block_h && SparseTileHeight(t.bytes) >= 0)
                       ? static_cast<u32>(SparseTileHeight(t.bytes))
                       : layout.tile_h;
    u32 gob_columns = 128u >> (sh + layout.tile_d); // GOB columns of one 64 KiB tile
    if (!gob_columns) {
        gob_columns = 1;
    }
    out.tile_width = static_cast<s32>(gob_columns * 64 / t.bytes * t.width / sx);
    out.tile_height = static_cast<s32>(dims >= 2 ? (8u << sh) * t.height / sy : 1);
    out.tile_depth = static_cast<s32>(dims >= 3 ? 1u << layout.tile_d : 1);
    const u32 levels = d.levels ? d.levels : 1;
    u32 n = 0;
    for (; n < levels; ++n) { // a level at least one tile wide, high and deep
        const u32 lwb = Blk(Mip(w * sx, n), t.width) * t.bytes;
        const u32 lh = dims >= 2 ? Blk(Mip(h * sy, n), t.height) : 1;
        const u32 ld = dims >= 3 ? Mip(z, n) : 1;
        if (lwb < gob_columns * 64 || lh < (8u << sh) || ld < (1u << layout.tile_d)) {
            break;
        }
    }
    out.num_tiled_levels = static_cast<s32>(n);
    return out;
}

// ---- SDK storage rules ----------------------------------------------------------------------------
//
// nvnTextureBuilderGetStorageSize / GetStorageAlignment / GetStorageClass / GetZCullStorageSize as
// the SDK answers them (3.2.0 sdk+0x5ed1d0 / +0x5ece10 / +0x5ed260 / +0x5ee8e0; 1.0.0 sdk+0x5c92e0
// / +0x5c8f10 / +0x5c9360 / +0x5caa30), for the device the SDK opens on our emulated GPU:
//  * Layout: sdk+0x5ed6f0 fills the layout library's parameters (sdk+0x5c0b8c), picks the block
//    shape (sdk+0x5c0c28: VIDEO_DECODE, sparse tiles sdk+0x5c0dc0, automatic sdk+0x6324a0,
//    MINIMAL_LAYOUT) or takes the packaged one, and sizes the level chain (sdk+0x5c0ff0 ->
//    sdk+0x632540, layered targets sdk+0x632200), times the depth field for every target but 3D.
//    Ported line for line and checked against the SDK code itself run on 5568 random builders.
//  * Kind (the PTE kind a texture's storage needs): sdk+0x5e2f90 asks NvRmGpuDeviceChooseKind
//    (sdk+0x59f8d0), which on the GA10F the emulator reports (chip 0x17e, nvhost_ctrl_gpu.cpp) runs
//    the arch at vtable sdk+0xdcd388 (choose sdk+0x5a60d8). Color formats get GENERIC (6), or
//    GENERIC_COMPRESSIBLE (8) with the COMPRESSIBLE flag unless the format has blocks wider than one
//    texel; depth / stencil formats get their Z kinds (SdkDepthKind). The device's default kind is
//    6 (device +0x1d68, sdk+0x5d93c0; 1.0.0 +0x1da0), compression is allowed (+0x1d6c from the
//    driver setting +0x1c48 = 1, sdk+0x5d8910). [INFERENCE] The arch's "disable PLC" open attribute
//    (attribute byte 7, sdk+0x5a7168) is 0 as in the default attributes (sdk+0xb990d0).
//  * A storage whose kind is not the default is padded to, and aligned on, the device's
//    compression page alignment (device +0xd4 = NvRmGpuDeviceInfo +0x17c, sdk+0x5aafdc: 0x1000 for
//    GA10F when the GPU reports a compression page size, nvhost_ctrl_gpu.cpp 0x20000).

constexpr s32 FLAG_DISPLAY = 0x1;
constexpr s32 FLAG_VIDEO_DECODE = 0x2;
constexpr s32 FLAG_COMPRESSIBLE = 0x8;
constexpr s32 FLAG_ADAPTIVE_ZCULL = 0x80;
constexpr s32 FLAG_ZCULL_SUPPORT_STENCIL = 0x400;
constexpr s32 TargetCubemap = 8;
constexpr s32 Target1DArray = 3;
constexpr s32 Target3D = 2;

constexpr u32 KindGeneric = 6;
constexpr u32 CompressionPageAlignment = 0x1000;
/// Device flag bit 16 (device +0xa bit 0): block-linear storage classes get bit 9 as well.
constexpr u32 DeviceFlagStorageClassHigh = 0x10000;

/// NVNgpuZcullInfo as our nvhost_ctrl_gpu ZCULL_GET_INFO answers it (nvhost_ctrl_gpu.cpp
/// ZCullGetInfo); the SDK copies it to device +0xe4 (sdk+0x5ab294, sdk+0x5e2f84).
constexpr u32 ZcullWidthAlign = 0x20;
constexpr u32 ZcullHeightAlign = 0x20;
constexpr u32 ZcullPixelSquaresByAliquots = 0x400;
constexpr u32 ZcullAliquotTotal = 0x800;
constexpr u32 ZcullRegionByteMultiplier = 0x20;
constexpr u32 ZcullRegionHeaderSize = 0x20;
constexpr u32 ZcullSubregionHeaderSize = 0xc0;
constexpr u32 ZcullSubregionWidthAlign = 0x20;
constexpr u32 ZcullSubregionHeightAlign = 0x40;

/// What the SDK rules read from a builder (or a texture's description).
struct SdkTexture {
    s32 target;
    s32 format;
    s32 flags;
    s32 samples;
    u32 width;
    u32 height;
    u32 depth;
    u32 levels;
    u32 stride;
    u16 packaged; ///< SDK encoding (builder +0x54): bit 0 set, block height, depth, tile width
    u32 gl_name;
};

/// SetPackagedTextureLayout keeps the first two bytes as the SDK does (3.2.0 sdk+0x5ed400): byte 0
/// low nibble the block height, high nibble the block depth, byte 1 low nibble the tile width.
u16 SdkPackaged(bool has_layout, const u8* layout) {
    if (!has_layout) {
        return 0;
    }
    return static_cast<u16>(1 | (layout[0] & 0xF) << 4 | (layout[0] & 0xF0) << 4 |
                            (layout[1] & 0xF) << 12);
}

SdkTexture SdkTextureOf(const GuestTextureBuilder& b) {
    return {b.desc.target, b.desc.format, b.desc.flags,  b.desc.samples,
            b.desc.width,  b.desc.height, b.desc.depth,  b.desc.levels,
            b.desc.stride, SdkPackaged(b.has_layout != 0, b.layout), b.gl_name};
}

SdkTexture SdkTextureOf(const TextureDesc& d) {
    return {d.target, d.format, d.flags,  d.samples, d.width,
            d.height, d.depth,  d.levels, d.stride,  SdkPackaged(d.has_layout != 0, d.layout),
            0};
}

constexpr u32 Shl(u32 value, u32 shift) {
    return value << (shift & 31);
}

constexpr u32 Shr(u32 value, u32 shift) {
    return value >> (shift & 31);
}

/// The SDK's format table entry (sdk+0xe03968): formats 0x01-0x18, 0x22-0x50 and 0x6d-0xa2 exist.
std::optional<Layout::BlockInfo> SdkFormatBlock(s32 format) {
    if (format < 1 || format > 0xA2 || (format > 0x18 && format < 0x22) ||
        (format > 0x50 && format < 0x6D)) {
        return std::nullopt;
    }
    switch (format) {
    case 0x9D: // the BC formats NVN2 added after ASTC: 4x4 blocks of 8 and 16 bytes
    case 0x9F:
        return Layout::BlockInfo{8, 4, 4};
    case 0x9E:
    case 0xA0:
        return Layout::BlockInfo{16, 4, 4};
    case 0xA1:
    case 0xA2:
        return Layout::BlockInfo{8, 1, 1};
    default:
        return Layout::FormatBlockInfo(format);
    }
}

struct DepthStencilBits {
    u32 depth;
    u32 stencil;
};

/// Entry +0xe / +0xf of the format table: STENCIL8 .. DEPTH32F_STENCIL8.
DepthStencilBits SdkDepthStencil(s32 format) {
    switch (format) {
    case 0x31:
        return {0, 8};
    case 0x32:
        return {16, 0};
    case 0x33:
        return {24, 0};
    case 0x34:
        return {32, 0};
    case 0x35:
        return {24, 8};
    case 0x36:
        return {32, 8};
    default:
        return {0, 0};
    }
}

/// GA10F Z kinds (sdk+0x5a7028 plain, sdk+0x5a70c0 compressible) of the depth / stencil formats.
u32 SdkDepthKind(s32 format, bool compressible) {
    switch (format) {
    case 0x31:
        return compressible ? 10 : 2;
    case 0x32:
        return compressible ? 11 : 1;
    case 0x33:
    case 0x35:
        return compressible ? 12 : 3;
    case 0x34:
        return compressible ? 9 : KindGeneric;
    default: // 0x36
        return compressible ? 13 : 4;
    }
}

/// Storage kind (3.2.0 sdk+0x5ed260 / sdk+0x5e2f90). Linear and buffer textures take the device's
/// default; a format without a table entry gets 0.
u32 SdkKind(const SdkTexture& t) {
    if (t.target == TargetBuffer || (t.flags & FLAGS_LINEAR_ANY) != 0) {
        return KindGeneric;
    }
    const auto block = SdkFormatBlock(t.format);
    if (!block) {
        return 0;
    }
    const bool compressible = (t.flags & FLAG_COMPRESSIBLE) != 0;
    const DepthStencilBits bits = SdkDepthStencil(t.format);
    if (bits.depth == 0 && bits.stencil == 0) {
        return compressible && block->width <= 1 ? 8 : KindGeneric;
    }
    return SdkDepthKind(t.format, compressible);
}

/// Block shape, log2 in GOBs: `w` is the tile width the GOB columns round to.
struct SdkBlockShape {
    u32 w;
    u32 h;
    u32 d;
};

/// The layout library's parameters (sdk+0x5c0b8c).
struct SdkLayoutParams {
    u32 flags;
    s32 target;
    u32 samples;
    u32 levels;
    u32 width;
    u32 height;
    u32 depth;
    u32 bytes;
    u32 block_w;
    u32 block_h;
    u32 dims;
    bool layered;
};

/// Index of the sample-count tables: (samples - 2) / 2, out of range for odd counts.
constexpr u32 SampleIndex(u32 samples) {
    return ((samples - 2) >> 1) | (samples << 31);
}

/// Horizontal sample factor (sdk+0xaed1f0, same as sdk+0xae6910) by SampleIndex.
constexpr std::array<u32, 8> SAMPLE_SCALE_X{2, 2, 1, 4, 1, 1, 1, 4};

u32 SampleScaleX(u32 samples) {
    const u32 index = SampleIndex(samples);
    return index < SAMPLE_SCALE_X.size() ? SAMPLE_SCALE_X[index] : 1;
}

u32 SampleScaleY(u32 samples) {
    return samples == 4 || samples == 8 ? 2 : samples == 16 ? 4 : 1;
}

/// Width and height in samples (sdk+0x5c0ff0 / sdk+0x5c0c28).
std::pair<u32, u32> SampleExtent(const SdkLayoutParams& p) {
    if (p.samples < 2) {
        return {p.width, p.height};
    }
    return {p.width * SampleScaleX(p.samples), p.height * SampleScaleY(p.samples)};
}

/// Bytes of a level chain (sdk+0x632540, no border, `round` = flag 0x800, `tile` = the tile width in
/// GOBs a level at least one tile large rounds its GOB columns to).
u64 SdkChainSize(const SdkBlockShape& blk, u32 width, u32 height, u32 depth, u32 dims, u32 bytes,
                 u32 block_w, u32 block_h, u32 levels, u32 round, u32 tile) {
    u64 total = 0;
    const u32 gob_blocks = bytes != 0 ? 64 / bytes : 0;
    const u32 min_size = round + 1;
    const u32 rows_limit = Shl(8, blk.h - 1);
    const u32 slices_limit = Shl(1, blk.d - 1);
    for (u32 level = 0; level < levels; ++level) {
        const u32 lw = std::max(min_size, Shr(width - round, level) + round);
        const u32 lh = std::max(min_size, Shr(height - round, level) + round);
        const u32 ld = dims < 3 ? 1 : std::max(min_size, Shr(depth - round, level) + round);
        const u32 rows = block_h != 0 ? (block_h + lh - 1) / block_h : 0;
        const u32 cols = block_w != 0 ? (block_w + lw - 1) / block_w : 0;
        const u32 row_bytes = cols * bytes;
        u32 align = tile;
        if (Shr(ld, blk.d) == 0 || rows < Shl(8, blk.h) || cols < gob_blocks * tile) {
            align = 1;
        }
        u32 tw = blk.w;
        if (tw != 0) {
            for (u32 limit = Shl(0x40, tw - 1); row_bytes <= limit; limit >>= 1) {
                if (--tw == 0) {
                    break;
                }
            }
        }
        u32 th = blk.h;
        if (th != 0 && rows <= rows_limit) {
            for (u32 limit = rows_limit;;) {
                if (--th == 0) {
                    break;
                }
                limit >>= 1;
                if (rows > limit) {
                    break;
                }
            }
        }
        u32 td = blk.d;
        if (td != 0 && ld <= slices_limit) {
            for (u32 limit = slices_limit;;) {
                if (--td == 0) {
                    break;
                }
                limit >>= 1;
                if (ld > limit) {
                    break;
                }
            }
        }
        const u32 gobs_wide = Shr(Shl(1, tw) + ((row_bytes + 0x3F) >> 6) - 1, tw);
        const u32 columns = align < 2 ? gobs_wide : (align + gobs_wide - 1) & (0u - align);
        const u32 gob_rows = Shr(Shl(1, th) + ((rows + 7) >> 3) - 1, th);
        const u32 slices = Shr(ld + Shl(1, td) - 1, td);
        total += (u64{gob_rows} * columns * slices) << ((tw + th + td + 9) & 63);
    }
    return total;
}

/// A layer of an array target: the chain rounded up to the block of level 0 (sdk+0x632200).
u64 SdkLayerChainSize(const SdkBlockShape& blk, u32 width, u32 height, u32 depth, u32 dims,
                      u32 bytes, u32 block_w, u32 block_h, u32 levels, u32 round, u32 tile) {
    u64 size =
        SdkChainSize(blk, width, height, depth, dims, bytes, block_w, block_h, levels, round, tile);
    if (tile >= 2) {
        const u64 unit = Shl(Shl(tile, 9), blk.d + blk.h);
        return (size + unit - 1) & (u64{0} - unit);
    }
    u32 x = width;
    u32 y = dims < 2 ? 0 : height;
    if (block_w > 1) {
        x = (block_w + x - 1) / block_w;
        y = block_h != 0 ? (block_h + y - 1) / block_h : 0;
    }
    u32 w = blk.w;
    if (w != 0) {
        for (u32 limit = Shl(0x40, w - 1); x * bytes <= limit; limit >>= 1) {
            if (--w == 0) {
                break;
            }
        }
    }
    u32 h = blk.h;
    if (h != 0) {
        for (u32 limit = Shl(8, h - 1); y <= limit; limit >>= 1) {
            if (--h == 0) {
                break;
            }
        }
    }
    u32 d = blk.d;
    if (d != 0) {
        for (u32 limit = Shl(1, d - 1); depth <= limit; limit >>= 1) {
            if (--d == 0) {
                break;
            }
        }
    }
    const u32 shift = w + h + d;
    const u64 unit = Shl(0x200, shift);
    const u64 whole = (size >> ((shift + 9) & 63)) * unit;
    return whole != size ? unit + whole : size;
}

/// Bytes of one layer with the block shape `blk` (sdk+0x5c0ff0).
u64 SdkLayerSize(const SdkLayoutParams& p, const SdkBlockShape& blk) {
    const auto [w, h] = SampleExtent(p);
    const u32 d = p.target == Target3D ? p.depth : 1;
    const u32 round = (p.flags >> 11) & 1;
    const SdkBlockShape shape{0, blk.h, blk.d};
    const u32 tile = Shl(1, blk.w);
    return p.layered ? SdkLayerChainSize(shape, w, h, d, p.dims, p.bytes, p.block_w, p.block_h,
                                         p.levels, round, tile)
                     : SdkChainSize(shape, w, h, d, p.dims, p.bytes, p.block_w, p.block_h,
                                    p.levels, round, tile);
}

/// A sparse texture's tile shape (sdk+0x5c0dc0), when at least its first level holds a tile.
std::optional<SdkBlockShape> SdkSparseShape(const SdkLayoutParams& p) {
    if (p.bytes == 12 || (p.flags & 0x12) != 0 ||
        (p.target >= 0 && p.target < 11 && ((0x409u >> p.target) & 1) != 0) ||
        (p.flags & Layout::TexFlag::Sparse) == 0) {
        return std::nullopt;
    }
    SdkBlockShape blk;
    if (p.target == Target3D) {
        blk = p.bytes == 16 || p.bytes == 8 ? SdkBlockShape{2, 1, 4}
              : p.bytes == 4                ? SdkBlockShape{1, 2, 4}
                                            : SdkBlockShape{0, 2, 5};
    } else {
        const u32 index = SampleIndex(p.bytes);
        blk = index < 8 ? SdkBlockShape{static_cast<u32>((0x0402020204020303ULL >> (8 * index)) & 0xFF),
                                        static_cast<u32>((0x0305050503050404ULL >> (8 * index)) & 0xFF),
                                        0}
                        : SdkBlockShape{2, 5, 0};
    }
    u32 tile_w = Shl(p.bytes != 0 ? 64 / p.bytes : 0, blk.w);
    u32 tile_h = Shl(8, blk.h);
    const u32 tile_d = Shl(1, blk.d);
    if (p.samples >= 2) {
        tile_w = (tile_w & 0xFFFF) / SampleScaleX(p.samples);
        tile_h /= SampleScaleY(p.samples);
    }
    if (p.block_w > 1) {
        tile_w *= p.block_w;
        tile_h *= p.block_h;
    }
    const u32 round = (p.flags >> 11) & 1;
    u32 tiled = p.levels;
    u32 w = p.width;
    u32 h = p.height;
    u32 d = p.depth;
    for (u32 level = 0; level < p.levels; ++level) {
        if (static_cast<s32>(w) < static_cast<s32>(tile_w) ||
            static_cast<s32>(h) < static_cast<s32>(tile_h) ||
            static_cast<s32>(d) < static_cast<s32>(tile_d)) {
            tiled = level;
            break;
        }
        w = round + ((w - round) >> 1);
        h = round + ((h - round) >> 1);
        if (p.target == Target3D) {
            d = round + ((d - round) >> 1);
        }
    }
    if (tiled == 0) {
        return std::nullopt;
    }
    return blk;
}

/// The automatic block shape (sdk+0x6324a0): about two thirds of the extent in GOBs.
SdkBlockShape SdkAutoShape(u32 dims, u32 rows, u32 slices) {
    if (dims == 3) {
        const u32 z = slices + (slices >> 1);
        return {0, 0, z >= 16 ? 4u : z >= 8 ? 3u : z >= 4 ? 2u : z > 1 ? 1u : 0u};
    }
    const u32 y = rows + (rows >> 1);
    return {0, y >= 0x80 ? 4u : y >= 0x40 ? 3u : y >= 0x20 ? 2u : y >= 0x10 ? 1u : 0u, 0};
}

/// sdk+0x5c0c28.
SdkBlockShape SdkSelectShape(const SdkLayoutParams& p) {
    if ((p.flags & FLAG_VIDEO_DECODE) != 0) {
        return {0, 1, 0};
    }
    if (const auto sparse = SdkSparseShape(p)) {
        return *sparse;
    }
    auto [w, h] = SampleExtent(p);
    if (p.block_w > 1) {
        w = (w + p.block_w - 1) / p.block_w;
        h = (h + p.block_h - 1) / p.block_h;
    }
    const SdkBlockShape blk = SdkAutoShape(p.dims, h, p.target == Target3D ? p.depth : 1);
    if ((p.flags & 0x2200) == 0) {
        return blk;
    }
    // MINIMAL_LAYOUT keeps the automatic shape only when 1-GOB blocks take as many bytes.
    constexpr SdkBlockShape flat{0, 0, 0};
    if ((p.flags & Layout::TexFlag::MinimalLayout) != 0 &&
        SdkLayerSize(p, flat) == SdkLayerSize(p, blk)) {
        return {0, blk.h, blk.d};
    }
    return flat;
}

SdkLayoutParams SdkParamsOf(const SdkTexture& t, const Layout::BlockInfo& block) {
    u32 height = t.height & 0xFFFF;
    u32 depth = t.depth & 0xFFFF;
    if (t.target == TargetCubemap) {
        depth = 6;
    } else if (t.target == Target1DArray) {
        depth = height;
        height = 1;
    }
    SdkLayoutParams p{static_cast<u32>(t.flags), t.target, static_cast<u32>(t.samples),
                      t.target >= 5 && t.target <= 7 ? 1u : t.levels, t.width, height, depth,
                      block.bytes, block.width, block.height, 2, false};
    switch (t.target) {
    case 0:
        p.dims = 1;
        break;
    case Target3D:
        p.dims = 3;
        break;
    case Target1DArray:
        p.dims = 1;
        p.layered = true;
        break;
    case 4:
    case 9:
    case TargetCubemap:
        p.layered = true;
        break;
    default:
        break;
    }
    return p;
}

SdkBlockShape SdkShapeOf(const SdkTexture& t, const SdkLayoutParams& p) {
    if ((t.packaged & 0xF) != 0) {
        return {static_cast<u32>(t.packaged >> 12), static_cast<u32>((t.packaged >> 4) & 0xF),
                static_cast<u32>((t.packaged >> 8) & 0xF)};
    }
    return SdkSelectShape(p);
}

/// The layout's storage size (sdk+0x5ed6f0), padded for a kind other than the default.
u64 SdkLayoutSize(const SdkTexture& t, const Layout::BlockInfo& block) {
    constexpr u64 Mask48 = (u64{1} << 48) - 1;
    if (t.target == TargetBuffer) {
        return u32{block.bytes * t.width};
    }
    if ((t.flags & FLAGS_LINEAR_ANY) != 0) {
        const u32 rows = t.target == Target1DArray ? 1 : (t.height & 0xFFFF);
        return u32{t.stride * rows};
    }
    const SdkLayoutParams p = SdkParamsOf(t, block);
    u64 size = SdkLayerSize(p, SdkShapeOf(t, p));
    if (t.target != Target3D) {
        size *= p.depth;
    }
    size &= Mask48;
    if (SdkKind(t) != KindGeneric) {
        size = ((size + CompressionPageAlignment - 1) & ~u64{CompressionPageAlignment - 1}) & Mask48;
    }
    return size;
}

/// A DISPLAY texture's size and alignment (3.2.0 sdk+0x5e22b0): the SDK asks the graphics server
/// (nn::wsi::nv::CalculateTextureRequirement -> nn::gsv::shim::CalculateRequirements, sdk+0x412070,
/// gsv 1510) for a block-linear image of the texture's size, block height and pitch, and raises
/// the alignment to 4 KiB. Answered with our gsv service's formula (Nx2CalculateImageLayout).
/// 1.0.0 has no such path (its DISPLAY textures take the layout, 1.0.0 sdk+0x5c92e0).
struct DisplayRequirement {
    u64 size;
    u64 alignment;
};

DisplayRequirement SdkDisplayRequirement(const SdkTexture& t) {
    const Layout::BlockInfo block = SdkFormatBlock(t.format).value_or(Layout::BlockInfo{0, 0, 0});
    const SdkLayoutParams p = SdkParamsOf(t, block);
    const SdkBlockShape blk = SdkShapeOf(t, p);
    // The pitch in GOBs, rounded to the tile width; a tile width (sparse or packaged) needs the
    // tile column count of sdk+0x632890, which no DISPLAY texture has: 1 GOB here.
    const u32 columns = block.width != 0 ? (t.width + block.width - 1) / block.width : 0;
    const u32 pitch_gobs = (block.bytes * columns + 63) / 64;
    u32 format_code = 0; // sdk+0x5e22b0's pixel format switch
    switch (t.format) {
    case 0x25:
    case 0x38:
        format_code = 1;
        break;
    case 0x37:
    case 0x6D:
        format_code = 2;
        break;
    case 0x9A:
    case 0x9C:
        format_code = 3;
        break;
    case 0x3D:
        format_code = 4;
        break;
    case 0x29:
        format_code = 5;
        break;
    case 0x71:
        format_code = 6;
        break;
    default:
        break;
    }
    std::array<u32, 16> preference{};
    preference[1] = 2; // block-linear (MakeInterModuleBlockUserImagePreference, sdk+0x4121c0)
    preference[2] = ((t.width - 1) & 0xFFFF) | ((t.height - 1) & 0xFFFF) << 16;
    preference[4] = blk.h + 3; // log2 of the block's rows
    preference[5] = pitch_gobs;
    preference[6] = format_code;
    const ::Service::Nx2ImageLayout layout = ::Service::Nx2CalculateImageLayout(preference.data());
    const u64 alignment = std::max<u64>(layout.alignment, 0x1000);
    return {(layout.size + alignment - 1) & ~(alignment - 1), alignment};
}

/// nvnTextureBuilderGetStorageSize (3.2.0 sdk+0x5ed1d0): 0 for a GL texture name or a format
/// without a table entry. The 1.0.0 SDK has no DISPLAY path (`older_sdk`, DeviceState::OlderSdk).
u64 SdkStorageSize(const SdkTexture& t, bool older_sdk) {
    if (t.gl_name != 0) {
        return 0;
    }
    if ((t.flags & FLAG_DISPLAY) != 0 && !older_sdk) {
        return SdkDisplayRequirement(t).size;
    }
    const auto block = SdkFormatBlock(t.format);
    return block ? SdkLayoutSize(t, *block) : 0;
}

/// nvnTextureBuilderGetStorageAlignment (3.2.0 sdk+0x5ece10). 1.0.0 (sdk+0x5c8f10, `older_sdk`)
/// differs for texture buffers of 1-byte formats (0x200 there, 1 here) and has no DISPLAY path.
u64 SdkStorageAlignment(const SdkTexture& t, bool older_sdk) {
    u64 alignment = 0x200;
    if ((t.flags & Layout::TexFlag::LinearRenderTarget) != 0) {
        alignment = 0x80;
    } else if ((t.flags & Layout::TexFlag::Linear) != 0) {
        alignment = 0x20;
    } else if (t.target == TargetBuffer) {
        const u32 bytes = t.format >= 0 && t.format <= 0xA2
                              ? SdkFormatBlock(t.format).value_or(Layout::FormatBlockInfo(t.format)).bytes
                              : 0;
        alignment = bytes == 12 ? 4 : bytes == 1 && older_sdk ? 0x200 : bytes;
    }
    if ((t.flags & FLAG_DISPLAY) != 0 && !older_sdk) {
        alignment = std::max(alignment, SdkDisplayRequirement(t).alignment);
    }
    return SdkKind(t) != KindGeneric ? CompressionPageAlignment : alignment;
}

/// nvnTextureBuilderGetStorageClass (3.2.0 sdk+0x5ed260): 0x100 for the default kind, else
/// kind | 0x100; kind | 0x300 on a device with flag bit 16 for block-linear storage.
u32 SdkStorageClass(const SdkTexture& t, u32 device_flags) {
    const u32 kind = SdkKind(t) & 0xFF;
    const bool plain = t.target == TargetBuffer || (t.flags & FLAGS_LINEAR_ANY) != 0;
    if ((device_flags & DeviceFlagStorageClassHigh) != 0 && !plain) {
        return kind | 0x300;
    }
    return kind == KindGeneric ? 0x100 : kind | 0x100;
}

/// Scaled aliquot count of a zcull mode letter (sdk+0x631b40).
u32 ZcullScaled(u32 count, u8 mode) {
    switch (mode) {
    case 0:
    case 7:
        return (count + 1) >> 1;
    case 5:
    case 6:
    case 9:
    case 10:
        return count << 1;
    case 8:
    case 11:
        return count << 2;
    case 12:
        return count << 3;
    case 15:
        return 0;
    default:
        return count;
    }
}

/// Aliquots the SDK's mode mix for `half` half-aliquots per subregion uses (sdk+0x631660 ->
/// sdk+0x631b40). Below 16 the mix is `half` subregions of the last (cheapest) letter; past the
/// end of the SDK's mix tables (48 entries for 3 letters, 112 for 4) all 16 take the first letter.
/// In between the SDK picks a mix of the 16 subregions from a table whose letters' weights (8, 4,
/// 2, 1 half-aliquots for scale 4, 2, 1, 1/2) add up to `half`: for an even subregion count every
/// such mix uses count x half / 2 aliquots. The table is not reproduced (SDK data): for an odd
/// count the cheapest letter's share rounds up, so the HLE takes it at its minimum (half & 1) and
/// can answer up to a few aliquots less than the SDK (open item).
u32 ZcullMixAliquots(u32 count, std::span<const u8> modes, u32 half) {
    if (half < 16) {
        return half * ZcullScaled(count, modes.back());
    }
    if (half - 16 >= (modes.size() == 3 ? 0x30u : 0x70u)) {
        return ZcullScaled(count, modes.front()) << 4;
    }
    const u32 cheapest = half == 16 ? 16 : (half & 1);
    return count * (half - cheapest) / 2 + cheapest * ((count + 1) >> 1);
}

/// nvnTextureBuilderGetZCullStorageSize (3.2.0 sdk+0x5ee8e0 -> sdk+0x5edaf0): region header +
/// region byte multiplier x aliquots + subregion header; 0 for a format without depth (or, with
/// ZCULL_SUPPORT_STENCIL, stencil). Without ADAPTIVE_ZCULL the aliquots cover the aligned extent
/// (sdk+0x632160) per layer; with it the device's adaptive setting (device +0x1c50 = 1, its default,
/// sdk+0x5d8910) sizes subregions (sdk+0x631ee0 -> sdk+0x631660) out of the aliquot total.
u32 SdkZcullStorageSize(const SdkTexture& t) {
    const DepthStencilBits bits = SdkDepthStencil(t.format);
    const bool depth_only = (t.flags & FLAG_ZCULL_SUPPORT_STENCIL) == 0 || bits.stencil == 0;
    if (depth_only && bits.depth == 0) {
        return 0;
    }
    const u32 layers = t.depth;
    u32 aliquots;
    if ((t.flags & FLAG_ADAPTIVE_ZCULL) == 0) {
        const u32 w = (t.width + ZcullWidthAlign - 1) / ZcullWidthAlign * ZcullWidthAlign;
        const u32 h = (t.height + ZcullHeightAlign - 1) / ZcullHeightAlign * ZcullHeightAlign;
        aliquots = (w * h + ZcullPixelSquaresByAliquots - 1) / ZcullPixelSquaresByAliquots *
                   std::max<u32>(layers, 1);
    } else {
        const u32 w = (t.width + ZcullSubregionWidthAlign - 1) / ZcullSubregionWidthAlign *
                      ZcullSubregionWidthAlign;
        const u32 h = (t.height + ZcullSubregionHeightAlign - 1) / ZcullSubregionHeightAlign *
                      ZcullSubregionHeightAlign;
        const u32 squares = (w * h + ZcullPixelSquaresByAliquots - 1) / ZcullPixelSquaresByAliquots;
        const u32 subregions = ((squares + 15) >> 4) * layers;
        // Mode letters (sdk+0xbdba90.. 0xbdba96): depth only or with stencil, 8 samples or not.
        static constexpr std::array<u8, 4> depth_modes{8, 5, 2, 0};
        static constexpr std::array<u8, 4> depth_modes_8x{8, 6, 3, 0};
        static constexpr std::array<u8, 3> stencil_modes{5, 2, 0};
        static constexpr std::array<u8, 3> stencil_modes_8x{6, 3, 0};
        const bool samples8 = t.samples == 8;
        const std::span<const u8> modes =
            depth_only ? std::span<const u8>{samples8 ? depth_modes_8x : depth_modes}
                       : std::span<const u8>{samples8 ? stencil_modes_8x : stencil_modes};
        const u32 cap = ZcullAliquotTotal; // device +0x1c60 = ~0 (no cap of its own)
        u32 per_subregion;
        if ((subregions & 1) == 0) {
            per_subregion = subregions != 0 ? cap * 2 / subregions : 0;
        } else {
            per_subregion = std::max(cap * 2 / (subregions + 1), (cap * 2 - 32) / subregions);
        }
        u32 half = per_subregion + 1;
        u32 used = ZcullMixAliquots(subregions, modes, half);
        if (cap < used) {
            half = per_subregion;
            used = ZcullMixAliquots(subregions, modes, half);
        }
        aliquots = std::min(used, cap) * layers;
        // The table-free mix is inexact only here (ZcullMixAliquots): tell a diagnostic run whether
        // the game reaches it, once per extent and flags, at most 8 lines.
        const u32 table_end = 16 + (modes.size() == 3 ? 0x30u : 0x70u);
        if ((subregions & 1) != 0 && half > 16 && half < table_end) {
            static std::mutex logged_mutex;
            static std::array<std::array<u32, 3>, 8> logged{};
            static size_t logged_count = 0;
            const std::array<u32, 3> key{t.width, t.height, static_cast<u32>(t.flags)};
            std::lock_guard lock{logged_mutex};
            if (logged_count < logged.size() &&
                std::find(logged.begin(), logged.begin() + logged_count, key) ==
                    logged.begin() + logged_count) {
                logged[logged_count++] = key;
                LOG_INFO(Core,
                         "[nvn] ZCullStorageSize adaptive odd subregions: {}x{} layers {} format "
                         "{:#x} flags {:#x} samples {}, {} subregions -> {} aliquots (approximate)",
                         t.width, t.height, layers, t.format, t.flags, t.samples, subregions,
                         std::min(used, cap));
            }
        }
    }
    return ZcullRegionHeaderSize + ZcullRegionByteMultiplier * aliquots + ZcullSubregionHeaderSize;
}

// ---- Host side tables ---------------------------------------------------------------------------

/// Where a texture's storage lives. The Recorder keeps the pool by serial; `NVNtexture` getters
/// answer with guest addresses, and the guest object may not hold anything but its tag.
struct StorageBinding {
    u64 pool_va;
    u64 address; ///< GPU address = guest VA of the first byte of the storage
};
std::mutex bindings_mutex;
std::unordered_map<u64, StorageBinding> bindings; // by texture serial

std::optional<StorageBinding> FindBinding(u64 serial) {
    std::lock_guard lock{bindings_mutex};
    const auto it = bindings.find(serial);
    if (it == bindings.end()) {
        return std::nullopt;
    }
    return it->second;
}

/// The tag of a live texture object at `va`, if it carries one.
std::optional<GuestTag> ReadTextureTag(Call& c, u64 va) {
    const GuestTag tag = c.Load<GuestTag>(va);
    if (tag.magic != GUEST_TAG_MAGIC || tag.kind != static_cast<u32>(ObjectKind::Texture)) {
        return std::nullopt;
    }
    return tag;
}

/// Returns the bytes copied (less than `size` when a read or write failed).
u64 CopyPackagedData(Call& c, u64 source, u64 destination, u64 size) {
    std::array<u8, 16384> buffer;
    u64 done = 0;
    while (done < size) {
        const size_t chunk = static_cast<size_t>(std::min<u64>(size - done, buffer.size()));
        if (!c.mem.Read(source + done, buffer.data(), chunk) ||
            !c.mem.Write(destination + done, buffer.data(), chunk)) {
            static Common::BootCounter logged;
            if (logged.First()) {
                LOG_WARNING(Core, "[nvn] packaged texture data {:#x} -> {:#x}: copy failed at {:#x}",
                            source, destination, done);
            }
            break;
        }
        done += chunk;
    }
    // The driver's own copy into the storage: the replay's copies of those bytes (a pool mirror)
    // follow, as they do for a store of the guest CPU.
    c.rec.NotifyHostWrite(destination, done);
    return done;
}

// ---- Texture builder: setters -------------------------------------------------------------------

/// SDK (3.2.0 sdk+0x5ecb60, 1.0.0 sdk+0x5c8c60): target 2D, width, height and depth 0, one level,
/// format, samples, flags, stride, storage and packaged layout 0, identity swizzle; the device is
/// kept.
void nvnTextureBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    GuestTextureBuilder defaults;
    std::memset(&defaults, 0, sizeof(defaults));
    defaults.device = c.Load<u64>(builder + TbDevice);
    defaults.desc.target = TARGET_2D;
    defaults.desc.levels = 1;
    c.Store(builder, defaults);
}

void nvnTextureBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + TbDevice, c.args.NextPointer());
}

void nvnTextureBuilderSetFlags(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbFlags, c.args.NextS32());
}

void nvnTextureBuilderSetTarget(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbTarget, c.args.NextS32());
}

void nvnTextureBuilderSetWidth(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbWidth, c.args.NextS32());
}

void nvnTextureBuilderSetHeight(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbHeight, c.args.NextS32());
}

void nvnTextureBuilderSetDepth(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbDepth, c.args.NextS32());
}

void nvnTextureBuilderSetSize1D(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 width = c.args.NextS32();
    const std::array<s32, 3> size{width, 1, 1}; // width, height, depth are adjacent
    static_assert(TbHeight == TbWidth + 4 && TbDepth == TbWidth + 8);
    c.Store(builder + TbWidth, size);
}

void nvnTextureBuilderSetSize2D(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    const std::array<s32, 3> size{width, height, 1};
    c.Store(builder + TbWidth, size);
}

void nvnTextureBuilderSetSize3D(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    const s32 depth = c.args.NextS32();
    const std::array<s32, 3> size{width, height, depth};
    c.Store(builder + TbWidth, size);
}

void nvnTextureBuilderSetLevels(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbLevels, c.args.NextS32());
}

void nvnTextureBuilderSetFormat(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbFormat, c.args.NextS32());
}

void nvnTextureBuilderSetSamples(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + TbSamples, c.args.NextS32());
}

void nvnTextureBuilderSetStride(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u32>(builder + TbStride, static_cast<u32>(c.args.NextS64()));
}

void nvnTextureBuilderSetStorage(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    const s64 offset = c.args.NextS64();
    const std::array<u64, 2> storage{pool, static_cast<u64>(offset)}; // pool, offset are adjacent
    static_assert(TbOffset == TbPool + 8);
    c.Store(builder + TbPool, storage);
}

void nvnTextureBuilderSetPackagedTextureLayout(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 layout = c.args.NextPointer();
    std::array<u8, 8> bytes{};
    const bool present = layout != 0 && c.mem.Read(layout, bytes.data(), bytes.size());
    c.Store<u8>(builder + TbHasLayout, present ? 1 : 0);
    if (present) {
        c.Store(builder + TbLayout, bytes);
    }
    c.Store<s32>(builder + TbPkgBlockH, present ? bytes[0] + 1 : 0);
}

void nvnTextureBuilderSetPackagedTextureData(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + TbPkgData, c.args.NextPointer());
}

void nvnTextureBuilderSetSwizzle(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 r = c.args.NextS32();
    const s32 g = c.args.NextS32();
    const s32 b = c.args.NextS32();
    const s32 a = c.args.NextS32();
    const u8 bits = c.Load<u8>(builder + TbBits);
    c.Store<u8>(builder + TbBits, bits | BitSetSwizzle);
    c.Store<u16>(builder + TbSwizzle, PackSwizzle(r, g, b, a));
}

/// STENCIL: the texture reads the stencil of its depth-stencil storage, as a stencil-mode view
/// does (Unreal Engine 4.27 builds its stencil shader resources this way).
void nvnTextureBuilderSetDepthStencilMode(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 mode = c.args.NextS32();
    const u8 bits = c.Load<u8>(builder + TbBits);
    c.Store<u8>(builder + TbBits,
                mode == DEPTH_STENCIL_MODE_STENCIL ? (bits | BitStencil) : (bits & ~BitStencil));
}

void nvnTextureBuilderSetGLTextureName(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u32>(builder + TbGlName, c.args.NextU32());
}

// ---- Texture builder: getters -------------------------------------------------------------------

void nvnTextureBuilderGetDevice(Call& c) {
    // No prototype in the nxnative header: the device pointer set with SetDevice.
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + TbDevice));
}

void nvnTextureBuilderGetFlags(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbFlags));
}

void nvnTextureBuilderGetTarget(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbTarget));
}

void nvnTextureBuilderGetWidth(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbWidth));
}

void nvnTextureBuilderGetHeight(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbHeight));
}

void nvnTextureBuilderGetDepth(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbDepth));
}

void nvnTextureBuilderGetLevels(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbLevels));
}

void nvnTextureBuilderGetFormat(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbFormat));
}

void nvnTextureBuilderGetSamples(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbSamples));
}

void nvnTextureBuilderGetStride(Call& c) {
    c.ret.SetU64(c.Load<u32>(c.args.NextPointer() + TbStride));
}

void nvnTextureBuilderGetMemoryPool(Call& c) {
    // The pool object set with SetStorage (EA SPORTS FC reads it back before initializing).
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + TbPool));
}

void nvnTextureBuilderGetMemoryOffset(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + TbOffset));
}

void nvnTextureBuilderGetPackagedTextureData(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + TbPkgData));
}

void nvnTextureBuilderGetPackagedTextureLayout(Call& c) {
    // No prototype in the nxnative header. [INFERENCE] returns a pointer to the 8 layout bytes
    // (NVNpackagedTextureLayout*), null when none was set.
    const u64 builder = c.args.NextPointer();
    c.ret.SetU64(c.Load<u8>(builder + TbHasLayout) ? builder + TbLayout : 0);
}

void nvnTextureBuilderGetSwizzle(Call& c) {
    const u64 builder = c.args.NextPointer();
    const std::array<u64, 4> out{c.args.NextPointer(), c.args.NextPointer(), c.args.NextPointer(),
                                 c.args.NextPointer()};
    const u16 swizzle =
        (c.Load<u8>(builder + TbBits) & BitSetSwizzle) ? c.Load<u16>(builder + TbSwizzle)
                                                       : IDENTITY_SWIZZLE;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i]) {
            c.Store<s32>(out[i], (swizzle >> (3 * i)) & 7);
        }
    }
}

void nvnTextureBuilderGetDepthStencilMode(Call& c) {
    const u8 bits = c.Load<u8>(c.args.NextPointer() + TbBits);
    c.ret.SetU32((bits & BitStencil) ? DEPTH_STENCIL_MODE_STENCIL : 0);
}

void nvnTextureBuilderGetGLTextureName(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + TbGlName));
}

/// The SDK rules (SdkStorageSize): ~12 calls per frame in DK, one guest read, no allocation.
void nvnTextureBuilderGetStorageSize(Call& c) {
    const GuestTextureBuilder b = c.Load<GuestTextureBuilder>(c.args.NextPointer());
    c.ret.SetU64(SdkStorageSize(SdkTextureOf(b), c.runtime.device.OlderSdk()));
}

void nvnTextureBuilderGetStorageAlignment(Call& c) {
    const GuestTextureBuilder b = c.Load<GuestTextureBuilder>(c.args.NextPointer());
    c.ret.SetU64(SdkStorageAlignment(SdkTextureOf(b), c.runtime.device.OlderSdk()));
}

void nvnTextureBuilderGetZCullStorageSize(Call& c) {
    const GuestTextureBuilder b = c.Load<GuestTextureBuilder>(c.args.NextPointer());
    c.ret.SetU64(SdkZcullStorageSize(SdkTextureOf(b)));
}

void nvnTextureBuilderGetSparseTileLayout(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 out = c.args.NextPointer();
    const GuestTextureDesc desc = c.Load<GuestTextureDesc>(builder + TbDesc);
    c.Store(out, ComputeSparseTileLayout(ToLayout(desc)));
}

/// The kind's storage class (SdkStorageClass). DK forwards it into its MemoryPoolMapVirtual
/// requests, which the HLE maps without looking at the class (functions_pool.cpp).
void nvnTextureBuilderGetStorageClass(Call& c) {
    const GuestTextureBuilder b = c.Load<GuestTextureBuilder>(c.args.NextPointer());
    c.ret.SetU32(SdkStorageClass(SdkTextureOf(b), c.runtime.device.flags.load()));
}

/// No prototype; nxnative returns 1 for the builder and the texture alike (the driver's raw
/// class: meaning unverified; callers compare it with 1).
void nvnTextureBuilderGetRawStorageClass(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU32(1);
}

// ---- Texture views (in place, GuestTextureView) --------------------------------------------------

void nvnTextureViewSetDefaults(Call& c) {
    c.Store(c.args.NextPointer(), GuestTextureView{});
}

void nvnTextureViewSetLevels(Call& c) {
    const u64 va = c.args.NextPointer();
    const s32 base = c.args.NextS32();
    const s32 count = c.args.NextS32();
    GuestTextureView view = c.Load<GuestTextureView>(va);
    view.set_bits |= ::VideoCommon::Nvn::GUEST_VIEW_SET_LEVELS;
    view.base_level = static_cast<u8>(base);
    view.nlevels = static_cast<u8>(count);
    c.Store(va, view);
}

void nvnTextureViewSetLayers(Call& c) {
    const u64 va = c.args.NextPointer();
    const s32 first = c.args.NextS32();
    const s32 count = c.args.NextS32();
    GuestTextureView view = c.Load<GuestTextureView>(va);
    view.set_bits |= ::VideoCommon::Nvn::GUEST_VIEW_SET_LAYERS;
    view.min_layer = static_cast<u16>(first);
    view.nlayers = static_cast<u16>(count);
    c.Store(va, view);
}

void nvnTextureViewSetFormat(Call& c) {
    const u64 va = c.args.NextPointer();
    const s32 format = c.args.NextS32();
    GuestTextureView view = c.Load<GuestTextureView>(va);
    view.set_bits |= ::VideoCommon::Nvn::GUEST_VIEW_SET_FORMAT;
    view.format = static_cast<u16>(format);
    c.Store(va, view);
}

void nvnTextureViewSetTarget(Call& c) {
    const u64 va = c.args.NextPointer();
    const s32 target = c.args.NextS32();
    GuestTextureView view = c.Load<GuestTextureView>(va);
    view.set_bits |= ::VideoCommon::Nvn::GUEST_VIEW_SET_TARGET;
    view.target = static_cast<u8>(target);
    c.Store(va, view);
}

void nvnTextureViewSetSwizzle(Call& c) {
    const u64 va = c.args.NextPointer();
    const s32 r = c.args.NextS32();
    const s32 g = c.args.NextS32();
    const s32 b = c.args.NextS32();
    const s32 a = c.args.NextS32();
    GuestTextureView view = c.Load<GuestTextureView>(va);
    view.set_bits |= ::VideoCommon::Nvn::GUEST_VIEW_SET_SWIZZLE;
    view.swizzle[0] = static_cast<u8>(r);
    view.swizzle[1] = static_cast<u8>(g);
    view.swizzle[2] = static_cast<u8>(b);
    view.swizzle[3] = static_cast<u8>(a);
    c.Store(va, view);
}

/// The 2-bit mode sits in bits 6..7 of the first byte, next to the set flags.
void nvnTextureViewSetDepthStencilMode(Call& c) {
    const u64 va = c.args.NextPointer();
    const s32 mode = c.args.NextS32();
    GuestTextureView view = c.Load<GuestTextureView>(va);
    view.set_bits = static_cast<u8>((view.set_bits & 0x1F) |
                                    ::VideoCommon::Nvn::GUEST_VIEW_SET_DSMODE | (mode & 3) << 6);
    c.Store(va, view);
}

void nvnTextureViewGetLevels(Call& c) {
    const GuestTextureView view = c.Load<GuestTextureView>(c.args.NextPointer());
    const u64 base = c.args.NextPointer();
    const u64 count = c.args.NextPointer();
    if (base) {
        c.Store<s32>(base, view.base_level);
    }
    if (count) {
        c.Store<s32>(count, view.nlevels);
    }
    c.ret.SetBool((view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_LEVELS) != 0);
}

void nvnTextureViewGetLayers(Call& c) {
    const GuestTextureView view = c.Load<GuestTextureView>(c.args.NextPointer());
    const u64 first = c.args.NextPointer();
    const u64 count = c.args.NextPointer();
    if (first) {
        c.Store<s32>(first, view.min_layer);
    }
    if (count) {
        c.Store<s32>(count, view.nlayers);
    }
    c.ret.SetBool((view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_LAYERS) != 0);
}

void nvnTextureViewGetFormat(Call& c) {
    const GuestTextureView view = c.Load<GuestTextureView>(c.args.NextPointer());
    const u64 format = c.args.NextPointer();
    if (format) {
        c.Store<s32>(format, view.format);
    }
    c.ret.SetBool((view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_FORMAT) != 0);
}

void nvnTextureViewGetTarget(Call& c) {
    const GuestTextureView view = c.Load<GuestTextureView>(c.args.NextPointer());
    const u64 target = c.args.NextPointer();
    if (target) {
        c.Store<s32>(target, view.target);
    }
    c.ret.SetBool((view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_TARGET) != 0);
}

void nvnTextureViewGetSwizzle(Call& c) {
    const GuestTextureView view = c.Load<GuestTextureView>(c.args.NextPointer());
    const bool set = (view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_SWIZZLE) != 0;
    for (size_t i = 0; i < 4; ++i) {
        const u64 out = c.args.NextPointer();
        if (out) {
            c.Store<s32>(out, set ? view.swizzle[i] : SWIZZLE_R + static_cast<s32>(i));
        }
    }
    c.ret.SetBool(set);
}

void nvnTextureViewGetDepthStencilMode(Call& c) {
    const GuestTextureView view = c.Load<GuestTextureView>(c.args.NextPointer());
    const u64 mode = c.args.NextPointer();
    if (mode) {
        c.Store<s32>(mode, view.set_bits >> 6);
    }
    c.ret.SetBool((view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_DSMODE) != 0);
}

/// Two views are equal when they set the same parameters to the same values (parameters that are
/// not set inherit from the texture and do not count). nxnative has no body for this.
void nvnTextureViewCompare(Call& c) {
    namespace L = ::VideoCommon::Nvn;
    const GuestTextureView a = c.Load<GuestTextureView>(c.args.NextPointer());
    const GuestTextureView b = c.Load<GuestTextureView>(c.args.NextPointer());
    constexpr u8 all_set = 0x3F;
    bool equal = (a.set_bits & all_set) == (b.set_bits & all_set);
    if (equal && (a.set_bits & L::GUEST_VIEW_SET_LEVELS)) {
        equal = a.base_level == b.base_level && a.nlevels == b.nlevels;
    }
    if (equal && (a.set_bits & L::GUEST_VIEW_SET_LAYERS)) {
        equal = a.min_layer == b.min_layer && a.nlayers == b.nlayers;
    }
    if (equal && (a.set_bits & L::GUEST_VIEW_SET_FORMAT)) {
        equal = a.format == b.format;
    }
    if (equal && (a.set_bits & L::GUEST_VIEW_SET_TARGET)) {
        equal = a.target == b.target;
    }
    if (equal && (a.set_bits & L::GUEST_VIEW_SET_SWIZZLE)) {
        equal = std::memcmp(a.swizzle, b.swizzle, sizeof(a.swizzle)) == 0;
    }
    if (equal && (a.set_bits & L::GUEST_VIEW_SET_DSMODE)) {
        equal = (a.set_bits >> 6) == (b.set_bits >> 6);
    }
    c.ret.SetBool(equal);
}

// ---- Texture objects ------------------------------------------------------------------------------

/// A failed nvnTextureInitialize: the SDK clears the whole 0x80-byte object before it checks
/// anything (3.2.0 sdk+0x5ee120; the last two bytes become 0xffff), so a texture object that was
/// live is no longer one. The recorder retires its old host object, the side table its entry.
void FailTextureInitialize(Call& c, u64 texture) {
    constexpr size_t TextureObjectBytes = 0x80;
    constexpr u64 TextureObjectMarker = 0x78;
    const std::optional<GuestTag> previous = ReadTextureTag(c, texture);
    if (previous) {
        c.rec.TextureFinalize(texture);
        std::lock_guard lock{bindings_mutex};
        bindings.erase(previous->serial);
    }
    c.Zero(texture, TextureObjectBytes);
    c.Store<u16>(texture + TextureObjectMarker, 0xFFFF);
    c.ret.SetBool(false);
}

void nvnTextureInitialize(Call& c) {
    const u64 texture = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const GuestTextureBuilder b = c.Load<GuestTextureBuilder>(builder);
    u64 packaged_bytes = 0;

    const auto pool = c.rec.GetMemoryPool(b.pool);
    if (!pool) {
        static Common::BootCounter logged;
        if (logged.First()) {
            LOG_WARNING(Core, "[nvn] TextureInitialize: no memory pool at {:#x} (SetStorage missing "
                              "or the pool was finalized)",
                        b.pool);
        }
        FailTextureInitialize(c, texture);
        return;
    }
    const u64 offset = static_cast<u64>(b.offset);
    const u64 storage = pool->address + offset;

    // The SDK's refusals (3.2.0 sdk+0x5ee120, 1.0.0 sdk+0x5ca240): a PHYSICAL pool, a format without
    // a table entry, COMPRESSIBLE in a pool neither COMPRESSIBLE nor VIRTUAL, and block-linear
    // storage of a kind other than the default outside a VIRTUAL pool unless the pool is
    // COMPRESSIBLE and the storage starts on the compression page (sdk+0x5e1278, which maps the
    // range with that kind; the size is padded to the page already).
    {
        constexpr u32 PoolCompressible = 0x80;
        constexpr u32 PoolPhysical = 0x100;
        constexpr u32 PoolVirtual = 0x200;
        const SdkTexture sdk = SdkTextureOf(b);
        const char* refusal = nullptr;
        if ((pool->flags & PoolPhysical) != 0) {
            refusal = "the pool is PHYSICAL";
        } else if (!SdkFormatBlock(sdk.format)) {
            refusal = "the format has no layout";
        } else if ((sdk.flags & FLAG_COMPRESSIBLE) != 0 &&
                   (pool->flags & (PoolCompressible | PoolVirtual)) == 0) {
            refusal = "COMPRESSIBLE texture in a pool that is neither COMPRESSIBLE nor VIRTUAL";
        } else if ((sdk.flags & FLAGS_LINEAR_ANY) == 0 && sdk.target != TargetBuffer &&
                   (pool->flags & PoolVirtual) == 0 && SdkKind(sdk) != KindGeneric &&
                   ((pool->flags & PoolCompressible) == 0 ||
                    (offset & (CompressionPageAlignment - 1)) != 0)) {
            refusal = "a compressible or depth kind needs a COMPRESSIBLE pool and a 4 KiB offset";
        }
        if (refusal != nullptr) {
            static Common::BootCounter logged;
            if (logged.Next() < 8) {
                LOG_WARNING(Core,
                            "[nvn] TextureInitialize refused as the SDK does: {} (format {:#x} "
                            "flags {:#x} pool flags {:#x} offset {:#x})",
                            refusal, sdk.format, sdk.flags, pool->flags, offset);
            }
            FailTextureInitialize(c, texture);
            return;
        }
    }
    const Layout::TextureLayoutDesc layout_desc = ToLayout(b.desc);

    // Packaged texture data in CPU memory: NVN copies it into the texture storage here (the CPU
    // cannot write a compressible pool; the driver copies). 13 Sentinels creates all its textures
    // this way. The storage is the texture's content from then on, so the desc does not carry the
    // CPU pointer on (the buffer behind it is the game's to free).
    if (b.pkg_data != 0 && offset < pool->size) {
        // Never more than the storage the game sized with nvnTextureBuilderGetStorageSize (the
        // host layout pads sparse layers further) nor than the pool holds.
        const u64 size = std::min({Layout::ComputeTextureLayout(layout_desc).size,
                                   SdkStorageSize(SdkTextureOf(b), c.runtime.device.OlderSdk()),
                                   pool->size - offset});
        packaged_bytes = CopyPackagedData(c, b.pkg_data, storage, size);
    }

    TextureDesc desc{};
    desc.target = b.desc.target;
    desc.format = b.desc.format;
    desc.flags = b.desc.flags;
    desc.samples = b.desc.samples;
    desc.width = b.desc.width;
    desc.height = b.desc.height;
    desc.depth = b.desc.depth;
    desc.levels = b.desc.levels;
    desc.stride = b.desc.stride;
    desc.pkg_block_h = b.desc.pkg_block_h;
    desc.has_layout = b.has_layout;
    std::memcpy(desc.layout, b.layout, sizeof(desc.layout));
    desc.set_swizzle = (b.bits & BitSetSwizzle) ? 1 : 0;
    desc.stencil = (b.bits & BitStencil) ? 1 : 0;
    desc.swizzle = b.swizzle;
    desc.pkg_data = 0;

    // SMBW re-initializes its transient render targets on the same object every frame and never
    // finalizes them: the Recorder retires the old host object, the side table drops its entry.
    const std::optional<GuestTag> previous = ReadTextureTag(c, texture);
    if (!c.rec.TextureInitialize(texture, desc, b.pool, offset)) {
        FailTextureInitialize(c, texture);
        return;
    }
    const GuestTag tag = c.Load<GuestTag>(texture);
    {
        std::lock_guard lock{bindings_mutex};
        if (previous) {
            bindings.erase(previous->serial);
        }
        bindings.insert_or_assign(tag.serial, StorageBinding{b.pool, storage});
    }
    if (packaged_bytes != 0) {
        // After CreateTexture in submit order: the replay looks at the storage again.
        c.rec.NotifyTextureWritten(texture);
    }
    c.ret.SetBool(true);
}

void nvnTextureFinalize(Call& c) {
    const u64 texture = c.args.NextPointer();
    const std::optional<GuestTag> tag = ReadTextureTag(c, texture);
    c.rec.TextureFinalize(texture);
    if (tag) {
        std::lock_guard lock{bindings_mutex};
        bindings.erase(tag->serial);
    }
}

void nvnTextureSetDebugLabel(Call& c) {
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextPointer());
}

// The getters read the Recorder's copy of the description (no allocation). A texture that was
// never initialized, or was finalized, answers 0.

/// The SDK keeps the flags as 16 bits and sign-extends them (3.2.0 sdk+0x5eea60).
void nvnTextureGetFlags(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? static_cast<u32>(static_cast<s32>(static_cast<s16>(desc->flags))) : 0);
}

void nvnTextureGetTarget(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? static_cast<u32>(desc->target) : 0);
}

void nvnTextureGetWidth(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? desc->width : 0);
}

/// The SDK keeps the height as 16 bits (3.2.0 sdk+0x5eeac0).
void nvnTextureGetHeight(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? desc->height & 0xFFFF : 0);
}

void nvnTextureGetDepth(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? desc->depth : 0);
}

void nvnTextureGetLevels(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? desc->levels : 0);
}

void nvnTextureGetFormat(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? static_cast<u32>(desc->format) : 0);
}

void nvnTextureGetSamples(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? static_cast<u32>(desc->samples) : 0);
}

/// The row stride of a linear texture as built (0 for block-linear ones).
void nvnTextureGetStride(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU64(desc ? desc->stride : 0);
}

void nvnTextureGetSwizzle(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    const u16 swizzle = desc && desc->set_swizzle ? desc->swizzle : IDENTITY_SWIZZLE;
    for (size_t i = 0; i < 4; ++i) {
        const u64 out = c.args.NextPointer();
        if (out) {
            c.Store<s32>(out, (swizzle >> (3 * i)) & 7);
        }
    }
}

void nvnTextureGetDepthStencilMode(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc && desc->stencil ? DEPTH_STENCIL_MODE_STENCIL : 0);
}

/// The builder's rules applied to the texture's description. [INFERENCE: the texture-side
/// getters were not decompiled; DK does not call them.]
void nvnTextureGetStorageSize(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? static_cast<u32>(SdkStorageSize(SdkTextureOf(*desc),
                                                        c.runtime.device.OlderSdk()))
                      : 0);
}

void nvnTextureGetZCullStorageSize(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU64(desc ? SdkZcullStorageSize(SdkTextureOf(*desc)) : 0);
}

void nvnTextureGetSparseTileLayout(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    const u64 out = c.args.NextPointer();
    c.Store(out, desc ? ComputeSparseTileLayout(ToLayout(*desc)) : GuestSparseTileLayout{});
}

void nvnTextureGetStorageClass(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU32(desc ? SdkStorageClass(SdkTextureOf(*desc), c.runtime.device.flags.load()) : 0);
}

void nvnTextureGetRawStorageClass(Call& c) {
    static_cast<void>(c.args.NextPointer());
    c.ret.SetU32(1);
}

void nvnTextureGetMemoryPool(Call& c) {
    // Engage (Unity) maps "the pool the texture lives in" at startup; the Recorder only knows the
    // pool by serial, so the guest address comes from the side table.
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    const auto binding = desc ? FindBinding(desc->serial) : std::nullopt;
    c.ret.SetU64(binding ? binding->pool_va : 0);
}

void nvnTextureGetMemoryOffset(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU64(desc ? desc->pool_offset : 0);
}

/// Pool address + offset, the address space of nvnMemoryPoolGetBufferAddress (DRAGON QUEST XI S).
void nvnTextureGetTextureAddress(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    const auto binding = desc ? FindBinding(desc->serial) : std::nullopt;
    c.ret.SetU64(binding ? binding->address : 0);
}

/// The unique id of the host object (not in nxnative; the NVN debug id only has to be unique).
void nvnTextureGetDebugID(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    c.ret.SetU64(desc ? desc->serial : 0);
}

/// Two textures compare equal when they describe the same storage the same way. The channel
/// mapping counts: BotW and Odyssey rebuild captures on the same storage with a new mapping and
/// replace the registered descriptor only if the two compare different.
void nvnTextureCompare(Call& c) {
    const auto a = c.rec.GetTexture(c.args.NextPointer());
    const auto b = c.rec.GetTexture(c.args.NextPointer());
    if (!a || !b) {
        c.ret.SetBool(false);
        return;
    }
    const auto swizzle = [](const TextureDesc& d) {
        return d.set_swizzle ? d.swizzle : IDENTITY_SWIZZLE;
    };
    c.ret.SetBool(a->target == b->target && a->format == b->format && a->flags == b->flags &&
                  a->samples == b->samples && a->width == b->width && a->height == b->height &&
                  a->depth == b->depth && a->levels == b->levels && a->stride == b->stride &&
                  a->pkg_block_h == b->pkg_block_h && a->pool_serial == b->pool_serial &&
                  a->pool_offset == b->pool_offset && a->stencil == b->stencil &&
                  swizzle(*a) == swizzle(*b));
}

/// Byte offset of a view's first level and layer inside the texture storage.
u64 ViewOffset(Call& c, const Layout::TextureLayoutDesc& desc, u64 view_va) {
    if (view_va == 0) {
        return 0;
    }
    const GuestTextureView view = c.Load<GuestTextureView>(view_va);
    u64 offset = 0;
    if (view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_LEVELS) {
        offset = Layout::TextureLevelOffset(desc, view.base_level);
    }
    if (view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_LAYERS) {
        offset += u64{view.min_layer} * Layout::ComputeTextureLayout(desc).layer_size;
    }
    return offset;
}

void nvnTextureGetViewOffset(Call& c) {
    const auto desc = c.rec.GetTexture(c.args.NextPointer());
    const u64 view = c.args.NextPointer();
    c.ret.SetU64(desc ? ViewOffset(c, ToLayout(*desc), view) : 0);
}

/// No prototype and no nxnative body: the builder form of GetViewOffset, same arguments with the
/// builder first. [INFERENCE]
void nvnTextureBuilderGetViewOffset(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const GuestTextureDesc desc = c.Load<GuestTextureDesc>(builder + TbDesc);
    c.ret.SetU64(ViewOffset(c, ToLayout(desc), view));
}

// ---- CPU texel access ------------------------------------------------------------------------------
//
// nxnative (WriteTexels / ReadTexels, hle_nvn.c 569-587 and 2626-2680) de-swizzles and swizzles
// block-linear storage in guest memory and tells the replay layer the texture was written.
//  * Linear textures (flags LINEAR / LINEAR_RENDER_TARGET, one level): rows are copied between the
//    caller's buffer and the storage in guest memory.
//  * Block-linear textures: the region is swizzled into (or unswizzled from) the level's bytes in
//    guest memory with the Switch 2 GOB order (`native_tiling`), 2D and 2D-array textures, one
//    slice per call. 3D regions and storage in virtual pools have no guest bytes to touch and are
//    reported once.
//  * A write tells the replay (NotifyTextureWritten) so its image is uploaded again. A read first
//    asks the replay to flush the texture to guest memory (FlushTextureToGuest): when the GPU
//    wrote the texture, its guest bytes are stale until then.
// FlushTexels / InvalidateTexels do nothing (unified memory; nxnative the same) but are counted.

Common::BootCounter write_texels_calls;
Common::BootCounter read_texels_calls;
Common::BootCounter flush_texels_calls;
Common::BootCounter invalidate_texels_calls;

void CountCall(Common::BootCounter& counter, const char* name) {
    const u64 n = counter.Next() + 1;
    if (std::has_single_bit(n)) {
        LOG_INFO(Core, "[nvn] {}: {} call(s)", name, n);
    }
}

bool CopyGuest(Call& c, u64 from, u64 to, size_t size) {
    std::array<u8, 4096> buffer;
    for (size_t done = 0; done < size;) {
        const size_t chunk = std::min(size - done, buffer.size());
        if (!c.mem.Read(from + done, buffer.data(), chunk) ||
            !c.mem.Write(to + done, buffer.data(), chunk)) {
            return false;
        }
        done += chunk;
    }
    return true;
}

/// The guest ranges one texel access wrote through the HLE's own copies (the storage of a write,
/// the caller's buffer of a read), told to the replay when the call ends: its copies of those bytes
/// (a pool mirror) follow, as they do for a store of the guest CPU. One notification per range, not
/// per row.
class HostWrites {
public:
    explicit HostWrites(Call& call_) : call{call_} {}
    HostWrites(const HostWrites&) = delete;
    HostWrites& operator=(const HostWrites&) = delete;
    ~HostWrites() {
        Flush(storage);
        Flush(caller);
    }
    void Storage(u64 va, u64 size) {
        Add(storage, va, size);
    }
    void Caller(u64 va, u64 size) {
        Add(caller, va, size);
    }

private:
    struct Span {
        u64 begin{~u64{0}};
        u64 end{0};
    };
    static void Add(Span& span, u64 va, u64 size) {
        span.begin = (std::min)(span.begin, va);
        span.end = (std::max)(span.end, va + size);
    }
    void Flush(const Span& span) {
        if (span.end > span.begin) {
            call.rec.NotifyHostWrite(span.begin, span.end - span.begin);
        }
    }

    Call& call;
    Span storage;
    Span caller;
};

void TexelAccess(Call& c, bool write, u64 texture, u64 view_va, u64 region_va, u64 data, s64 row,
                 s64 image) {
    static_cast<void>(image); // one slice per call: no image stride
    const auto desc = c.rec.GetTexture(texture);
    if (!desc) {
        return;
    }
    HostWrites writes{c};
    const char* const what = write ? "TextureWriteTexels" : "TextureReadTexels";
    const auto bad = [&](const char* reason) {
        static Common::BootCounter logged;
        if (logged.Next() < 8) {
            LOG_WARNING(Core, "[nvn] {}: {} (format {:#x} target {} {}x{}x{}, flags {:#x})", what,
                        reason, desc->format, desc->target, desc->width, desc->height, desc->depth,
                        desc->flags);
        }
    };
    const Layout::TextureLayoutDesc layout_desc = ToLayout(*desc);
    const bool linear = Layout::IsLinearStorage(layout_desc);
    if (desc->target == TargetBuffer) {
        bad("texture buffers have no texels to access");
        return;
    }
    const Layout::BlockInfo block = Layout::FormatBlockInfo(desc->format);
    const auto binding = FindBinding(desc->serial);
    if (!block.bytes || !binding || region_va == 0 || data == 0) {
        bad("unknown format, storage or arguments");
        return;
    }
    const GuestCopyRegion region = c.Load<GuestCopyRegion>(region_va);
    u32 level = 0;
    u32 layer = 0;
    if (view_va != 0) {
        const GuestTextureView view = c.Load<GuestTextureView>(view_va);
        if (view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_LEVELS) {
            level = view.base_level;
        }
        if (view.set_bits & ::VideoCommon::Nvn::GUEST_VIEW_SET_LAYERS) {
            layer = view.min_layer;
        }
    }
    if (level >= std::max<u32>(desc->levels, 1)) {
        bad("level out of range");
        return;
    }
    const u32 level_width = Mip(desc->width, level);
    const u32 level_height = Mip(desc->height, level);
    const bool arrayed = desc->target == 3 || desc->target == 4 || desc->target == 6 ||
                         desc->target == 8 || desc->target == 9;
    if (linear && level != 0) {
        bad("level other than 0 of a linear texture");
        return;
    }
    if (region.x < 0 || region.y < 0 || region.width <= 0 || region.height <= 0 ||
        static_cast<u32>(region.x) + static_cast<u32>(region.width) > level_width ||
        static_cast<u32>(region.y) + static_cast<u32>(region.height) > level_height) {
        bad("region outside the texture");
        return;
    }
    const u32 slice_count = static_cast<u32>(std::max(region.depth, 1));
    const bool volume = desc->target == 2;
    if (!volume && !arrayed && (region.z != 0 || slice_count > 1)) {
        bad("a region with depth on a texture that has one slice");
        return;
    }
    if (volume && (region.z < 0 || static_cast<u32>(region.z) + slice_count > Mip(desc->depth, level))) {
        bad("region outside the texture");
        return;
    }
    if (arrayed) {
        layer += static_cast<u32>(std::max(region.z, 0));
    }
    if (linear && slice_count > 1) {
        bad("several slices of a linear texture");
        return;
    }
    // A write replaces part of what the guest bytes hold: when the GPU wrote the texture they are
    // stale, so a partial write would erase the rest of the GPU's texels.
    if (!c.rec.FlushTextureToGuest(texture)) {
        bad("the texture could not be flushed to guest memory");
        return;
    }
    const Layout::TextureLayout layout = Layout::ComputeTextureLayout(layout_desc);
    const u64 row_bytes = u64{Blk(static_cast<u32>(region.width), block.width)} * block.bytes;
    const u64 rows = Blk(static_cast<u32>(region.height), block.height);
    const u64 pitch = row > 0 ? std::max(static_cast<u64>(row), row_bytes) : row_bytes;
    const u64 first_row = static_cast<u32>(region.y) / block.height;
    const u64 first_column_blocks = static_cast<u32>(region.x) / block.width;
    if (linear) {
        const u64 stride =
            desc->stride ? desc->stride
                         : (u64{block.bytes} * Blk(desc->width, block.width) + layout.align - 1) &
                               ~u64{layout.align - 1};
        for (u64 y = 0; y < rows; ++y) {
            const u64 storage = binding->address + (first_row + y) * stride +
                                first_column_blocks * block.bytes;
            const u64 caller = data + y * pitch;
            if (write) {
                writes.Storage(storage, row_bytes);
            } else {
                writes.Caller(caller, row_bytes);
            }
            if (!(write ? CopyGuest(c, caller, storage, static_cast<size_t>(row_bytes))
                        : CopyGuest(c, storage, caller, static_cast<size_t>(row_bytes)))) {
                bad("guest memory access failed");
                return;
            }
        }
        if (write) {
            c.rec.NotifyTextureWritten(texture);
        }
        return;
    }

    // Block-linear: the bytes of the level in guest memory.
    if (level >= std::max<u32>(desc->levels, 1) || layer + (volume ? 0 : slice_count) > std::max<u32>(layout.layers, 1)) {
        bad("level or layer out of range");
        return;
    }
    const u32 cols = Blk(level_width, block.width);
    const u32 level_rows = Blk(level_height, block.height);
    const u32 region_cols = Blk(static_cast<u32>(region.width), block.width);
    u32 tile_h = 0;
    u32 tile_d = 0;
    Layout::TextureLevelTiles(layout_desc, level, tile_h, tile_d);
    const u64 image_pitch = image > 0 ? static_cast<u64>(image) : pitch * rows;
    const u32 level_count = std::max<u32>(desc->levels, 1);
    // [begin, end) of the level inside the storage for layer `which`.
    const auto level_bytes = [&](u32 which, u64& begin, u64& end) {
        begin = u64{which} * layout.layer_size + Layout::TextureLevelOffset(layout_desc, level);
        end = level + 1 < level_count
                  ? u64{which} * layout.layer_size +
                        Layout::TextureLevelOffset(layout_desc, level + 1)
                  : u64{which + 1} * layout.layer_size;
        return end > begin;
    };

    if (volume) {
        // A region of a 3D texture: unswizzle the whole level, patch the slices, swizzle it back.
        const u32 depth = Mip(desc->depth, level);
        u64 begin = 0;
        u64 end = 0;
        if (!level_bytes(0, begin, end)) {
            bad("empty level");
            return;
        }
        const u64 needed = Tegra::Texture::CalculateSize(true, block.bytes, cols, level_rows, depth,
                                                         tile_h, tile_d);
        std::vector<u8> bytes(static_cast<size_t>(std::max<u64>(end - begin, needed)), 0);
        if (!c.mem.Read(binding->address + begin, bytes.data(), static_cast<size_t>(end - begin))) {
            bad("storage is not guest memory (virtual pool?)");
            return;
        }
        const u64 level_row_bytes = u64{cols} * block.bytes;
        std::vector<u8> volume_bytes(static_cast<size_t>(level_row_bytes * level_rows * depth));
        Tegra::Texture::UnswizzleTexture(volume_bytes, bytes, block.bytes, cols, level_rows, depth,
                                         tile_h, tile_d, 0, true);
        for (u32 slice = 0; slice < slice_count; ++slice) {
            for (u64 y = 0; y < rows; ++y) {
                u8* const texel_row =
                    volume_bytes.data() +
                    ((u64{static_cast<u32>(region.z)} + slice) * level_rows + first_row + y) *
                        level_row_bytes +
                    first_column_blocks * block.bytes;
                const u64 caller = data + slice * image_pitch + y * pitch;
                if (!write) {
                    writes.Caller(caller, row_bytes);
                }
                if (write ? !c.mem.Read(caller, texel_row, static_cast<size_t>(row_bytes))
                          : !c.mem.Write(caller, texel_row, static_cast<size_t>(row_bytes))) {
                    bad("guest memory access failed");
                    return;
                }
            }
        }
        if (write) {
            Tegra::Texture::SwizzleTexture(bytes, volume_bytes, block.bytes, cols, level_rows,
                                           depth, tile_h, tile_d, 0, true);
            writes.Storage(binding->address + begin, end - begin);
            if (!c.mem.Write(binding->address + begin, bytes.data(),
                             static_cast<size_t>(end - begin))) {
                bad("guest memory access failed");
                return;
            }
            c.rec.NotifyTextureWritten(texture);
        }
        return;
    }

    for (u32 slice = 0; slice < slice_count; ++slice) {
        u64 begin = 0;
        u64 end = 0;
        if (!level_bytes(layer + slice, begin, end)) {
            bad("empty level");
            return;
        }
        std::vector<u8> bytes(static_cast<size_t>(end - begin));
        const u64 guest_level = binding->address + begin;
        if (!c.mem.Read(guest_level, bytes.data(), bytes.size())) {
            bad("storage is not guest memory (virtual pool?)");
            return;
        }
        const u64 caller = data + slice * image_pitch;
        // Packed host scratch (row_bytes per row, bounded by the level): the guest row stride
        // only places the rows in guest memory.
        std::vector<u8> linear_bytes(static_cast<size_t>(row_bytes * rows));
        if (write) {
            for (u64 y = 0; y < rows; ++y) {
                if (!c.mem.Read(caller + y * pitch, linear_bytes.data() + y * row_bytes,
                                static_cast<size_t>(row_bytes))) {
                    bad("guest memory access failed");
                    return;
                }
            }
            Tegra::Texture::SwizzleSubrect(bytes, linear_bytes, block.bytes, cols, level_rows, 1,
                                           static_cast<u32>(first_column_blocks),
                                           static_cast<u32>(first_row), region_cols,
                                           static_cast<u32>(rows), tile_h, tile_d,
                                           static_cast<u32>(row_bytes), true);
            writes.Storage(guest_level, bytes.size());
            if (!c.mem.Write(guest_level, bytes.data(), bytes.size())) {
                bad("guest memory access failed");
                return;
            }
            continue;
        }
        Tegra::Texture::UnswizzleSubrect(linear_bytes, bytes, block.bytes, cols, level_rows, 1,
                                         static_cast<u32>(first_column_blocks),
                                         static_cast<u32>(first_row), region_cols,
                                         static_cast<u32>(rows), tile_h, tile_d,
                                         static_cast<u32>(row_bytes), true);
        for (u64 y = 0; y < rows; ++y) {
            writes.Caller(caller + y * pitch, row_bytes);
            if (!c.mem.Write(caller + y * pitch, linear_bytes.data() + y * row_bytes,
                             static_cast<size_t>(row_bytes))) {
                bad("guest memory access failed");
                return;
            }
        }
    }
    if (write) {
        c.rec.NotifyTextureWritten(texture);
    }
}

void nvnTextureWriteTexels(Call& c) {
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 src = c.args.NextPointer();
    CountCall(write_texels_calls, "TextureWriteTexels");
    TexelAccess(c, true, texture, view, region, src, 0, 0);
}

void nvnTextureWriteTexelsStrided(Call& c) {
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 src = c.args.NextPointer();
    const s64 row = c.args.NextS64();
    const s64 image = c.args.NextS64();
    CountCall(write_texels_calls, "TextureWriteTexels");
    TexelAccess(c, true, texture, view, region, src, row, image);
}

void nvnTextureReadTexels(Call& c) {
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 dst = c.args.NextPointer();
    CountCall(read_texels_calls, "TextureReadTexels");
    TexelAccess(c, false, texture, view, region, dst, 0, 0);
}

void nvnTextureReadTexelsStrided(Call& c) {
    const u64 texture = c.args.NextPointer();
    const u64 view = c.args.NextPointer();
    const u64 region = c.args.NextPointer();
    const u64 dst = c.args.NextPointer();
    const s64 row = c.args.NextS64();
    const s64 image = c.args.NextS64();
    CountCall(read_texels_calls, "TextureReadTexels");
    TexelAccess(c, false, texture, view, region, dst, row, image);
}

void nvnTextureFlushTexels(Call& c) {
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextPointer());
    CountCall(flush_texels_calls, "TextureFlushTexels");
}

void nvnTextureInvalidateTexels(Call& c) {
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextPointer());
    static_cast<void>(c.args.NextPointer());
    CountCall(invalidate_texels_calls, "TextureInvalidateTexels");
}

} // namespace

/// Forgets every texture's storage binding: serials restart with the next Recorder.
void ClearTextureBindings() {
    std::lock_guard lock{bindings_mutex};
    bindings.clear();
}

void RegisterTextureFunctions(Registry& registry) {
    // Builder.
    NVN_HLE(registry, nvnTextureBuilderSetDefaults);
    NVN_HLE(registry, nvnTextureBuilderSetDevice);
    NVN_HLE(registry, nvnTextureBuilderSetFlags);
    NVN_HLE(registry, nvnTextureBuilderSetTarget);
    NVN_HLE(registry, nvnTextureBuilderSetWidth);
    NVN_HLE(registry, nvnTextureBuilderSetHeight);
    NVN_HLE(registry, nvnTextureBuilderSetDepth);
    NVN_HLE(registry, nvnTextureBuilderSetSize1D);
    NVN_HLE(registry, nvnTextureBuilderSetSize2D);
    NVN_HLE(registry, nvnTextureBuilderSetSize3D);
    NVN_HLE(registry, nvnTextureBuilderSetLevels);
    NVN_HLE(registry, nvnTextureBuilderSetFormat);
    NVN_HLE(registry, nvnTextureBuilderSetSamples);
    NVN_HLE(registry, nvnTextureBuilderSetSwizzle);
    NVN_HLE(registry, nvnTextureBuilderSetDepthStencilMode);
    NVN_HLE(registry, nvnTextureBuilderSetStorage);
    NVN_HLE(registry, nvnTextureBuilderSetPackagedTextureData);
    NVN_HLE(registry, nvnTextureBuilderSetPackagedTextureLayout);
    NVN_HLE(registry, nvnTextureBuilderSetStride);
    NVN_HLE(registry, nvnTextureBuilderSetGLTextureName);
    NVN_HLE(registry, nvnTextureBuilderGetDevice);
    NVN_HLE(registry, nvnTextureBuilderGetFlags);
    NVN_HLE(registry, nvnTextureBuilderGetTarget);
    NVN_HLE(registry, nvnTextureBuilderGetWidth);
    NVN_HLE(registry, nvnTextureBuilderGetHeight);
    NVN_HLE(registry, nvnTextureBuilderGetDepth);
    NVN_HLE(registry, nvnTextureBuilderGetLevels);
    NVN_HLE(registry, nvnTextureBuilderGetFormat);
    NVN_HLE(registry, nvnTextureBuilderGetSamples);
    NVN_HLE(registry, nvnTextureBuilderGetSwizzle);
    NVN_HLE(registry, nvnTextureBuilderGetDepthStencilMode);
    NVN_HLE(registry, nvnTextureBuilderGetPackagedTextureData);
    NVN_HLE(registry, nvnTextureBuilderGetPackagedTextureLayout);
    NVN_HLE(registry, nvnTextureBuilderGetStride);
    NVN_HLE(registry, nvnTextureBuilderGetGLTextureName);
    NVN_HLE(registry, nvnTextureBuilderGetMemoryPool);
    NVN_HLE(registry, nvnTextureBuilderGetMemoryOffset);
    NVN_HLE(registry, nvnTextureBuilderGetStorageSize);
    NVN_HLE(registry, nvnTextureBuilderGetStorageAlignment);
    NVN_HLE(registry, nvnTextureBuilderGetStorageClass);
    NVN_HLE(registry, nvnTextureBuilderGetRawStorageClass);
    NVN_HLE(registry, nvnTextureBuilderGetZCullStorageSize);
    NVN_HLE(registry, nvnTextureBuilderGetSparseTileLayout);
    NVN_HLE(registry, nvnTextureBuilderGetViewOffset);

    // Views.
    NVN_HLE(registry, nvnTextureViewSetDefaults);
    NVN_HLE(registry, nvnTextureViewSetLevels);
    NVN_HLE(registry, nvnTextureViewSetLayers);
    NVN_HLE(registry, nvnTextureViewSetFormat);
    NVN_HLE(registry, nvnTextureViewSetSwizzle);
    NVN_HLE(registry, nvnTextureViewSetDepthStencilMode);
    NVN_HLE(registry, nvnTextureViewSetTarget);
    NVN_HLE(registry, nvnTextureViewGetLevels);
    NVN_HLE(registry, nvnTextureViewGetLayers);
    NVN_HLE(registry, nvnTextureViewGetFormat);
    NVN_HLE(registry, nvnTextureViewGetSwizzle);
    NVN_HLE(registry, nvnTextureViewGetDepthStencilMode);
    NVN_HLE(registry, nvnTextureViewGetTarget);
    NVN_HLE(registry, nvnTextureViewCompare);

    // Textures.
    NVN_HLE(registry, nvnTextureInitialize);
    NVN_HLE(registry, nvnTextureFinalize);
    NVN_HLE(registry, nvnTextureSetDebugLabel);
    NVN_HLE(registry, nvnTextureGetZCullStorageSize);
    NVN_HLE(registry, nvnTextureGetStorageClass);
    NVN_HLE(registry, nvnTextureGetRawStorageClass);
    NVN_HLE(registry, nvnTextureGetViewOffset);
    NVN_HLE(registry, nvnTextureGetFlags);
    NVN_HLE(registry, nvnTextureGetTarget);
    NVN_HLE(registry, nvnTextureGetWidth);
    NVN_HLE(registry, nvnTextureGetHeight);
    NVN_HLE(registry, nvnTextureGetDepth);
    NVN_HLE(registry, nvnTextureGetLevels);
    NVN_HLE(registry, nvnTextureGetFormat);
    NVN_HLE(registry, nvnTextureGetSamples);
    NVN_HLE(registry, nvnTextureGetSwizzle);
    NVN_HLE(registry, nvnTextureGetDepthStencilMode);
    NVN_HLE(registry, nvnTextureGetStride);
    NVN_HLE(registry, nvnTextureGetTextureAddress);
    NVN_HLE(registry, nvnTextureGetSparseTileLayout);
    NVN_HLE(registry, nvnTextureGetMemoryPool);
    NVN_HLE(registry, nvnTextureGetMemoryOffset);
    NVN_HLE(registry, nvnTextureGetStorageSize);
    NVN_HLE(registry, nvnTextureGetDebugID);
    NVN_HLE(registry, nvnTextureCompare);
    NVN_HLE(registry, nvnTextureWriteTexels);
    NVN_HLE(registry, nvnTextureWriteTexelsStrided);
    NVN_HLE(registry, nvnTextureReadTexels);
    NVN_HLE(registry, nvnTextureReadTexelsStrided);
    NVN_HLE(registry, nvnTextureFlushTexels);
    NVN_HLE(registry, nvnTextureInvalidateTexels);
}

} // namespace Core::NvnHle::Hle
