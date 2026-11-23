// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright (c) 2026 MacNativeGames contributors
// SPDX-FileCopyrightText: Copyright (C) 2018-2020 fincs (deko3d)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT AND Zlib

// Storage size / alignment / per-level offsets of NVN textures (Tegra X1 block-linear layout),
// ported from nxnative dm/gfx/nvn/nvn_tex.c. That file credits deko3d source/dk_image.cpp (zlib
// license, Copyright (C) 2018-2020 fincs) for the algorithm, the packaged-layout derivation, and
// the block depth nibble / pixel rows found by digitalhivehub against the original NVN driver
// (nxnative PR #11), taken from MacNativeGames 3c482a7.

#include <array>

#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"

namespace Vulkan::Nvn {
namespace {

constexpr u64 SPARSE_TILE = 0x10000;

struct FormatEntry {
    s32 first;
    s32 last;
    u8 bytes;
};

// Uncompressed formats: NVNformat range -> bytes per pixel.
constexpr std::array<FormatEntry, 37> UNCOMPRESSED_FORMATS{{
    {0x01, 0x04, 1},  {0x05, 0x09, 2},  {0x0A, 0x0C, 4},  {0x0D, 0x10, 2},  {0x11, 0x15, 4},
    {0x16, 0x18, 8},  {0x19, 0x1C, 3},  {0x1D, 0x21, 6},  {0x22, 0x24, 12}, {0x25, 0x28, 4},
    {0x29, 0x2D, 8},  {0x2E, 0x30, 16}, {0x31, 0x31, 1},  {0x32, 0x32, 2},  {0x33, 0x35, 4},
    {0x36, 0x36, 8},  {0x37, 0x38, 4},  {0x39, 0x3C, 2},  {0x3D, 0x40, 4},  {0x51, 0x52, 1},
    {0x53, 0x54, 2},  {0x55, 0x56, 4},  {0x57, 0x58, 2},  {0x59, 0x5A, 4},  {0x5B, 0x5C, 8},
    {0x5D, 0x5E, 3},  {0x5F, 0x60, 6},  {0x61, 0x62, 12}, {0x63, 0x64, 4},  {0x65, 0x66, 8},
    {0x67, 0x68, 16}, {0x69, 0x6C, 4},  {0x6D, 0x70, 4},  {0x71, 0x75, 8},  {0x76, 0x78, 16},
    {0x95, 0x98, 2},  {0x99, 0x9C, 4},
}};

constexpr std::array<std::array<u8, 2>, 14> ASTC_BLOCKS{{
    {4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6}, {8, 8}, {10, 5}, {10, 6}, {10, 8},
    {10, 10}, {12, 10}, {12, 12},
}};

u32 Mip(u32 size, u32 level) {
    size >>= level;
    return size ? size : 1;
}

u32 Blk(u32 size, u32 block) {
    return (size + block - 1) / block;
}

u32 PickTile(u32 gobs) {
    return gobs >= 16 ? 4 : gobs >= 8 ? 3 : gobs >= 4 ? 2 : gobs >= 2 ? 1 : 0;
}

u32 AdjTile(u32 shift, u32 unit, u32 dim) {
    if (!shift) {
        return 0;
    }
    u32 x = unit << (shift - 1);
    if (x >= dim) {
        while (--shift) {
            x >>= 1;
            if (x < dim) {
                break;
            }
        }
    }
    return shift;
}

struct Shape {
    u32 dims_per_layer; // 1, 2 or 3
    bool layered;
    u32 w, h, z;
};

Shape ShapeOf(const TextureLayoutDesc& d) {
    Shape s{.dims_per_layer = 2, .layered = false, .w = d.width, .h = d.height, .z = d.depth};
    switch (static_cast<TexTarget>(d.target)) {
    case TexTarget::Dim1D:
        s.dims_per_layer = 1;
        s.h = 1;
        s.z = 1;
        break;
    case TexTarget::Dim1DArray:
        s.dims_per_layer = 1;
        s.z = d.height; // the layer count is in height
        s.h = 1;
        s.layered = true;
        break;
    case TexTarget::Dim3D:
        s.dims_per_layer = 3;
        break;
    case TexTarget::Dim2DArray:
    case TexTarget::Dim2DMultisampleArray:
        s.layered = true;
        break;
    case TexTarget::Cubemap:
        s.z = 6;
        s.layered = true;
        break;
    case TexTarget::CubemapArray:
        s.layered = true; // the depth counts layer-faces
        break;
    default:
        s.z = 1;
        break;
    }
    if (s.z == 0) {
        s.z = 1;
    }
    return s;
}

void SamplesXY(s32 samples, u32& sx, u32& sy) {
    sx = sy = 1;
    if (samples == 2) {
        sx = 2;
    } else if (samples == 4) {
        sx = sy = 2;
    } else if (samples == 8) {
        sx = 4;
        sy = 2;
    }
}

bool IsSparse(const TextureLayoutDesc& d) {
    return (d.flags & TexFlag::Sparse) && !(d.flags & 0x50) &&
           d.target != static_cast<s32>(TexTarget::Buffer);
}

// Block height (log2 GOBs) of a sparse texture's 64 KiB tile for a block size, -1 for others.
int SparseTileHeight(u32 bytes) {
    return bytes == 1 ? 5 : (bytes == 2 || bytes == 4) ? 4 : (bytes == 8 || bytes == 16) ? 3 : -1;
}

u32 TileShapeHeight(const TextureLayoutDesc& d, u32 dims, const BlockInfo& t, u32 tile_h) {
    return dims == 2 && !d.pkg_block_h && SparseTileHeight(t.bytes) >= 0
               ? static_cast<u32>(SparseTileHeight(t.bytes))
               : tile_h;
}

u64 LevelWalk(const TextureLayoutDesc& d, u32 level, const Shape& s, u32 tile_h, u32 tile_d,
              const BlockInfo& t, u32* gobs) {
    u32 sx, sy;
    SamplesXY(d.samples, sx, sy);
    const u32 sh = TileShapeHeight(d, s.dims_per_layer, t, tile_h);
    u32 tw = 128u >> (sh + tile_d); // GOB columns of a 64 KiB tile
    if (!tw) {
        tw = 1;
    }
    bool tiled = IsSparse(d);
    u64 off = 0;
    for (u32 i = 0;; ++i) {
        const u32 lw = Blk(Mip(s.w * sx, i), t.width);
        const u32 lh = s.dims_per_layer >= 2 ? Blk(Mip(s.h * sy, i), t.height) : 1;
        const u32 ld = s.dims_per_layer >= 3 ? Mip(s.z, i) : 1;
        const u32 lwb = lw * t.bytes;
        const u32 th = AdjTile(tile_h, 8, lh);
        const u32 td = AdjTile(tile_d, 1, ld);
        u32 wg = (lwb + 63) / 64;
        const u32 hg = (lh + 7) / 8;
        tiled = tiled && lwb >= tw * 64 && lh >= (8u << sh) && ld >= (1u << tile_d);
        if (tiled) {
            wg = (wg + tw - 1) / tw * tw;
        }
        if (i == level) {
            if (gobs) {
                *gobs = wg;
            }
            return off;
        }
        const u32 ht = (hg + (1u << th) - 1) >> th;
        const u32 dt = (ld + (1u << td) - 1) >> td;
        off += (static_cast<u64>(wg) * ht * dt) << (9 + th + td);
    }
}

u64 LayerBytes(const TextureLayoutDesc& d, const Shape& s, u32 tile_h, u32 tile_d,
               const BlockInfo& t) {
    u64 ls = LevelWalk(d, d.levels ? d.levels : 1, s, tile_h, tile_d, t, nullptr);
    if (IsSparse(d)) {
        ls = (ls + SPARSE_TILE - 1) & ~(SPARSE_TILE - 1);
    }
    if (s.layered) {
        const u32 hh = Blk(s.h, t.height);
        u32 th = tile_h;
        u32 td = tile_d;
        while (th && hh <= (8u << (th - 1))) {
            th--;
        }
        while (td && s.z <= (1u << (td - 1))) {
            td--;
        }
        const u32 sh = th + td + 9;
        const u64 g = ls >> sh;
        if (ls != (g << sh)) {
            ls = (g + 1) << sh;
        }
    }
    return ls;
}

} // Anonymous namespace

BlockInfo FormatBlockInfo(s32 f) {
    if (f >= 0x79 && f <= 0x94) {
        const auto& block = ASTC_BLOCKS[static_cast<size_t>((f - 0x79) % 14)];
        return {16, block[0], block[1]};
    }
    if (f >= 0x41 && f <= 0x50) { // DXT / RGTC / BPTC: 4x4 blocks
        const bool b8 = f <= 0x42 || f == 0x45 || f == 0x46 || f == 0x49 || f == 0x4A;
        return {b8 ? 8u : 16u, 4, 4};
    }
    for (const FormatEntry& entry : UNCOMPRESSED_FORMATS) {
        if (f >= entry.first && f <= entry.last) {
            return {entry.bytes, 1, 1};
        }
    }
    return {0, 1, 1};
}

bool IsLinearStorage(const TextureLayoutDesc& d) {
    return (d.flags & 0x50) != 0;
}

TextureLayout ComputeTextureLayout(const TextureLayoutDesc& d) {
    const BlockInfo t = FormatBlockInfo(d.format);
    const Shape s = ShapeOf(d);
    u32 sx, sy;
    SamplesXY(d.samples, sx, sy);
    TextureLayout o{};
    o.bytes_per_block = t.bytes;
    o.layers = s.layered ? s.z : 1;
    if (!t.bytes) {
        o.align = 512;
        return o;
    }
    if (d.target == static_cast<s32>(TexTarget::Buffer)) {
        o.layer_size = o.size = static_cast<u64>(t.bytes) * d.width;
        o.align = t.bytes == 12 ? 4 : t.bytes;
        return o;
    }
    if (d.flags & 0x50) { // LINEAR or LINEAR_RENDER_TARGET
        o.align = (d.flags & TexFlag::LinearRenderTarget) ? 128 : 32;
        const u32 stride =
            d.stride ? d.stride : (t.bytes * Blk(s.w, t.width) + o.align - 1) & ~(o.align - 1);
        o.layer_size = o.size = static_cast<u64>(stride) * s.h;
        return o;
    }
    u32 tile_h = 0;
    u32 tile_d = 0;
    if (d.pkg_block_h) { // packaged byte 0: low nibble block height, high nibble block depth
        const u32 pl = static_cast<u32>(d.pkg_block_h - 1) & 0xff;
        if (s.dims_per_layer >= 2) {
            tile_h = pl & 0xf;
        }
        if (s.dims_per_layer == 3) {
            tile_d = pl >> 4;
        }
    } else if (s.dims_per_layer == 2 && IsSparse(d) && SparseTileHeight(t.bytes) >= 0 &&
               Blk(s.w * sx, t.width) * t.bytes >=
                   (128u >> SparseTileHeight(t.bytes)) * 64 &&
               Blk(s.h * sy, t.height) >= (8u << SparseTileHeight(t.bytes))) {
        tile_h = static_cast<u32>(SparseTileHeight(t.bytes));
    } else if (s.dims_per_layer == 2) {
        const u32 ah = Blk(s.h * sy, t.height);
        tile_h = PickTile(ah * 3 / 16);
    } else if (s.dims_per_layer == 3) {
        tile_d = PickTile(s.z + s.z / 2);
    }
    u64 ls = LayerBytes(d, s, tile_h, tile_d, t);
    // MINIMAL_LAYOUT keeps the automatic block height and depth only when the whole layer is no
    // larger than with 1-GOB blocks, else it takes 1-GOB blocks. A packaged layout and a sparse
    // texture's tiles keep their own.
    if ((d.flags & TexFlag::MinimalLayout) && !d.pkg_block_h && !IsSparse(d) &&
        (tile_h || tile_d)) {
        const u64 one = LayerBytes(d, s, 0, 0, t);
        if (one != ls) {
            tile_h = tile_d = 0;
            ls = one;
        }
    }
    o.tile_h = tile_h;
    o.tile_d = tile_d;
    o.layer_size = ls;
    o.size = s.layered ? ls * s.z : ls;
    o.align = IsSparse(d) ? static_cast<u32>(SPARSE_TILE) : 512;
    return o;
}

u64 TextureLevelOffset(const TextureLayoutDesc& d, u32 level) {
    const BlockInfo t = FormatBlockInfo(d.format);
    if (!t.bytes || (d.flags & 0x50) || d.target == static_cast<s32>(TexTarget::Buffer)) {
        return 0;
    }
    const Shape s = ShapeOf(d);
    const TextureLayout layout = ComputeTextureLayout(d);
    return LevelWalk(d, level, s, layout.tile_h, layout.tile_d, t, nullptr);
}

u32 TextureLevelGobs(const TextureLayoutDesc& d, u32 level) {
    const BlockInfo t = FormatBlockInfo(d.format);
    if (!t.bytes || (d.flags & 0x50) || d.target == static_cast<s32>(TexTarget::Buffer)) {
        return 0;
    }
    const Shape s = ShapeOf(d);
    const TextureLayout layout = ComputeTextureLayout(d);
    u32 gobs = 0;
    LevelWalk(d, level, s, layout.tile_h, layout.tile_d, t, &gobs);
    return gobs;
}

void TextureLevelTiles(const TextureLayoutDesc& d, u32 level, u32& tile_h, u32& tile_d) {
    const BlockInfo t = FormatBlockInfo(d.format);
    const Shape s = ShapeOf(d);
    u32 sx, sy;
    SamplesXY(d.samples, sx, sy);
    const TextureLayout layout = ComputeTextureLayout(d);
    const u32 lh = s.dims_per_layer >= 2 ? Blk(Mip(s.h * sy, level), t.height) : 1;
    const u32 ld = s.dims_per_layer >= 3 ? Mip(s.z, level) : 1;
    tile_h = AdjTile(layout.tile_h, 8, lh);
    tile_d = AdjTile(layout.tile_d, 1, ld);
}

bool IsBlockView(s32 texture_format, s32 view_format) {
    const BlockInfo texture = FormatBlockInfo(texture_format);
    const BlockInfo view = FormatBlockInfo(view_format);
    return texture.bytes != 0 && texture.bytes == view.bytes &&
           (texture.width > 1 || texture.height > 1) && view.width == 1 && view.height == 1;
}

std::optional<BlockViewAlias> BlockViewAliasOf(const TextureLayoutDesc& base, s32 view_format,
                                               u32 level, u32 layer) {
    if (!IsBlockView(base.format, view_format) || base.samples > 1 || IsLinearStorage(base) ||
        base.target == static_cast<s32>(TexTarget::Buffer)) {
        return std::nullopt;
    }
    const Shape s = ShapeOf(base);
    // A 3D level interleaves its slices in GOB blocks of 2^tile_d slices: no slice is a 2D image
    // of its own, so the alias of a 3D level is a 3D texture of all its slices (layer 0).
    const bool volume = s.dims_per_layer == 3;
    if ((s.dims_per_layer != 2 && !volume) || level >= (base.levels ? base.levels : 1) ||
        layer >= (s.layered ? s.z : 1)) {
        return std::nullopt;
    }
    const BlockInfo t = FormatBlockInfo(base.format);
    u32 tile_h = 0;
    u32 tile_d = 0;
    TextureLevelTiles(base, level, tile_h, tile_d);
    BlockViewAlias alias;
    alias.desc = TextureLayoutDesc{
        .target = static_cast<s32>(volume ? TexTarget::Dim3D : TexTarget::Dim2D),
        .format = view_format,
        .flags = 0,
        .samples = base.samples,
        .width = Blk(Mip(s.w, level), t.width),
        .height = Blk(Mip(s.h, level), t.height),
        .depth = volume ? Mip(s.z, level) : 1,
        .levels = 1,
        .stride = 0,
        // The level's own block height and depth: a single-level texture would pick its own
        // otherwise.
        .pkg_block_h = static_cast<s32>(tile_h | tile_d << 4) + 1,
    };
    alias.offset = static_cast<u64>(layer) * ComputeTextureLayout(base).layer_size +
                   TextureLevelOffset(base, level);
    u32 alias_h = 0;
    u32 alias_d = 0;
    TextureLevelTiles(alias.desc, 0, alias_h, alias_d);
    if (alias_h != tile_h || alias_d != tile_d ||
        TextureLevelGobs(alias.desc, 0) != TextureLevelGobs(base, level)) {
        return std::nullopt; // e.g. a sparse texture pads its tiled levels to whole tiles
    }
    return alias;
}

std::optional<SparseShape> SparseTileShape(const TextureLayoutDesc& d, u32 levels) {
    const BlockInfo t = FormatBlockInfo(d.format);
    const Shape s = ShapeOf(d);
    if (!IsSparse(d) || !t.bytes || d.samples > 1 || d.pkg_block_h || s.dims_per_layer != 2 ||
        SparseTileHeight(t.bytes) < 0) {
        return std::nullopt;
    }
    const u32 sh = static_cast<u32>(SparseTileHeight(t.bytes));
    const TextureLayout layout = ComputeTextureLayout(d);
    if (layout.tile_h != sh || layout.tile_d != 0) {
        return std::nullopt; // the texture is too small for tiles: ordinary block-linear
    }
    SparseShape shape{
        .tile_cols = ((128u >> sh) * 64) / t.bytes,
        .tile_rows = 8u << sh,
        .tile_h = sh,
        .tiled_levels = 0,
    };
    // The levels LevelWalk pads to whole tiles: the leading ones that are at least a tile wide
    // and tall.
    for (u32 level = 0; level < levels; ++level) {
        const u32 cols = Blk(Mip(s.w, level), t.width);
        const u32 rows = Blk(Mip(s.h, level), t.height);
        if (cols < shape.tile_cols || rows < shape.tile_rows) {
            break;
        }
        ++shape.tiled_levels;
    }
    if (shape.tiled_levels == 0) {
        return std::nullopt;
    }
    return shape;
}

} // namespace Vulkan::Nvn
