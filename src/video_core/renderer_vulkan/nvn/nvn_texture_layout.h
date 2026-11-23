// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright (C) 2018-2020 fincs (deko3d)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT AND Zlib

#pragma once

#include <optional>

#include "common/common_types.h"

namespace Vulkan::Nvn {

/// NVN texture targets (NVNtextureTarget).
enum class TexTarget : s32 {
    Dim1D = 0,
    Dim2D = 1,
    Dim3D = 2,
    Dim1DArray = 3,
    Dim2DArray = 4,
    Dim2DMultisample = 5,
    Dim2DMultisampleArray = 6,
    Rectangle = 7,
    Cubemap = 8,
    CubemapArray = 9,
    Buffer = 10,
};

/// NVN texture flag bits (NVNtextureFlags) the layout depends on.
namespace TexFlag {
constexpr s32 Sparse = 0x20;
constexpr s32 Linear = 0x10;
constexpr s32 LinearRenderTarget = 0x40;
constexpr s32 MinimalLayout = 0x200;
} // namespace TexFlag

/// The fields of an NVN texture that determine where its bytes are (nxnative nvn_texdesc).
struct TextureLayoutDesc {
    s32 target{};
    s32 format{};
    s32 flags{};
    s32 samples{};
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 levels{};
    u32 stride{};
    /// Packaged layout byte 0 plus one (low nibble log2 block height, high nibble log2 block
    /// depth), 0 for the automatic algorithm.
    s32 pkg_block_h{};
};

struct TextureLayout {
    u64 size{};       ///< Total storage bytes.
    u64 layer_size{}; ///< Bytes per layer.
    u32 align{};      ///< Storage alignment.
    u32 layers{};
    u32 bytes_per_block{};
    u32 tile_h{}; ///< log2 of the block height in GOBs.
    u32 tile_d{}; ///< log2 of the block depth in GOBs.
};

struct BlockInfo {
    u32 bytes{}; ///< 0 for formats without a layout.
    u32 width{1};
    u32 height{1};
};

/// Bytes per block and block extent of an NVNformat value.
BlockInfo FormatBlockInfo(s32 format);

/// Storage size, alignment and block shape of a texture (Tegra X1 block-linear rules, as found by
/// nxnative for NVN).
TextureLayout ComputeTextureLayout(const TextureLayoutDesc& desc);

/// Offset of `level` from the start of its layer.
u64 TextureLevelOffset(const TextureLayoutDesc& desc, u32 level);

/// GOB columns level `level` is laid out with (its width in GOBs; padded to whole 64 KiB tiles in a
/// sparse texture's tiled levels).
u32 TextureLevelGobs(const TextureLayoutDesc& desc, u32 level);

/// Block height and depth (log2 of the GOB count) actually used by `level`.
void TextureLevelTiles(const TextureLayoutDesc& desc, u32 level, u32& tile_h, u32& tile_d);

/// True for textures whose storage is linear rows instead of block-linear GOBs.
bool IsLinearStorage(const TextureLayoutDesc& desc);

/// True when `view_format` views a compressed `texture_format` block by block: an uncompressed
/// format whose texel is one block of the texture (BC4 as R16G16B16A16_UINT, BC7 as
/// R32G32B32A32_UINT). Rendering through such a view writes whole blocks (GPU block encoders).
bool IsBlockView(s32 texture_format, s32 view_format);

/// One level and layer of a compressed texture as a texture of its own in a block view format:
/// `desc` is a single-level texture of the level's size in blocks, laid out exactly like that
/// level (same GOB columns, block height and block depth), `offset` bytes into the texture's
/// storage. A 2D level's alias is 2D; a 3D level's is 3D with all its slices (layer 0: block-linear
/// 3D levels interleave their slices, so no single slice has a 2D layout of its own).
struct BlockViewAlias {
    TextureLayoutDesc desc;
    u64 offset{};
};

/// The alias of `level` / `layer` of `base` in `view_format`, nullopt when `view_format` is not a
/// block view of it (IsBlockView) or no single-level texture has the level's layout (1D textures,
/// multisampled or linear storage, a sparse texture's tile-padded levels).
std::optional<BlockViewAlias> BlockViewAliasOf(const TextureLayoutDesc& base, s32 view_format,
                                               u32 level, u32 layer);

/// How a sparse texture's tiled levels are cut into 64 KiB tiles (nxnative nvn_tex_sparse_tiles):
/// a tile is one block row of GOB columns (`tile_cols` x `tile_rows` blocks, block height
/// 2^`tile_h` GOBs), a level's tiles follow each other row by row from the level's offset, and
/// the leading levels that are at least a tile wide and tall are padded to whole tiles; the
/// rest (the mip tail) is ordinary block-linear.
struct SparseShape {
    u32 tile_cols{};
    u32 tile_rows{};
    u32 tile_h{};
    u32 tiled_levels{};
};

/// The tile shape of a 2D sparse texture with `levels` mip levels, nullopt when it does not use
/// tiles (not sparse, multisampled, packaged layout, a block size without a tile shape, smaller
/// than one tile).
std::optional<SparseShape> SparseTileShape(const TextureLayoutDesc& desc, u32 levels);

} // namespace Vulkan::Nvn
