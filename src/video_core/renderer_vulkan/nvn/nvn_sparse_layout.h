// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Where the 64 KiB tiles of a sparse texture are, and how one tile's guest bytes become texels.
// Pure CPU code over a Storage's shape (the tests run it without a Vulkan device).
//
// A sparse NVN texture (NVN_TEXTURE_FLAGS_SPARSE) in a virtual memory pool holds only the tiles
// the game mapped with nvnMemoryPoolMapVirtual: a tile is 64 KiB (the page size of the pool),
// one block row of GOB columns of a level (nvn_texture_layout.h `SparseShape`), and the layer's
// tiles follow each other from its first byte: level 0 row by row, then level 1, and so on for
// the leading levels that are at least a tile wide and tall. What follows (the small levels, the
// mip tail) is ordinary block-linear data up to the end of the layer. A tile nobody mapped reads
// as zero. The layout is nxnative's (dm/gfx/tex nvn_deswizzle_tile, the tile rules of
// docs/reference/nvn-objects.md "Large sparse textures as Metal placement sparse textures").

#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"

namespace Vulkan::Nvn {

inline constexpr u64 SPARSE_TILE_BYTES = 0x10000;

/// One tiled level of a layer.
struct SparseLevel {
    u32 width{};      ///< texels
    u32 height{};
    u32 tiles_x{};    ///< tiles across and down
    u32 tiles_y{};
    u32 first_tile{}; ///< index of its first tile within a layer (tiles are contiguous)
    u64 offset{};     ///< guest bytes from the start of the layer
};

/// A tile of a storage.
struct SparseTileRef {
    u32 layer{};
    u32 level{};
    u32 x{}; ///< tile column and row within the level
    u32 y{};
};

/// A texel rectangle of one level.
struct SparseRect {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
};

struct SparseLayout {
    SparseShape shape;
    BlockInfo block;
    u32 layers{};
    u32 levels{};
    u64 layer_size{};
    /// Guest bytes from the start of a layer to its first non-tiled level (the mip tail).
    u64 tail_offset{};
    u32 tiles_per_layer{};
    /// Texel extent of a tile (`shape.tile_cols` x `shape.tile_rows` blocks).
    u32 tile_width{};
    u32 tile_height{};
    std::vector<SparseLevel> level; ///< the tiled levels

    [[nodiscard]] u32 TileCount() const noexcept {
        return layers * tiles_per_layer;
    }
    [[nodiscard]] bool HasTail() const noexcept {
        return tail_offset < layer_size && level.size() < levels;
    }
    [[nodiscard]] u64 TailSize() const noexcept {
        return layer_size - tail_offset;
    }
    /// Tile `id` (layer * tiles_per_layer + tile within the layer).
    [[nodiscard]] SparseTileRef Tile(u32 id) const;
    /// Guest bytes from the start of the storage to the start of tile `id`.
    [[nodiscard]] u64 TileOffset(u32 id) const;
    /// The texels tile `ref` covers, cut to the level's extent.
    [[nodiscard]] SparseRect TileRect(const SparseTileRef& ref) const;
    /// Visits every tile (`tile(id)`) and every layer's mip tail (`tail(layer)`) whose guest bytes
    /// intersect [begin, end) of the storage.
    template <typename TileFn, typename TailFn>
    void ForEachUnit(u64 begin, u64 end, TileFn&& tile, TailFn&& tail) const {
        end = (std::min)(end, static_cast<u64>(layers) * layer_size);
        if (begin >= end) {
            return;
        }
        for (u64 layer = begin / layer_size; layer * layer_size < end; ++layer) {
            const u64 base = layer * layer_size;
            const u64 from = (std::max)(begin, base) - base;
            const u64 to = (std::min)(end, base + layer_size) - base;
            if (from < tail_offset) {
                const u64 last = (std::min)(to, tail_offset) - 1;
                for (u64 index = from / SPARSE_TILE_BYTES; index <= last / SPARSE_TILE_BYTES;
                     ++index) {
                    tile(static_cast<u32>(layer * tiles_per_layer + index));
                }
            }
            if (to > tail_offset && HasTail()) {
                tail(static_cast<u32>(layer));
            }
        }
    }
};

/// The layout of `s`, nullopt when its tiles cannot be handled one by one (not a sparse 2D
/// texture, multisampled, a layout without the tile shape, a block size without one).
[[nodiscard]] std::optional<SparseLayout> BuildSparseLayout(const Storage& s);

/// Bytes of the decoded texels of one tile (`DecodeSparseTile` output): `tile_width` x
/// `tile_height` texels in the host's texel size, rows packed.
[[nodiscard]] u64 SparseTileStagingBytes(const Storage& s, const SparseLayout& layout);

/// Turns the 64 KiB guest bytes of a tile into texels (decoded to the stand-in format when the
/// host cannot hold the compressed one) in `out` (`SparseTileStagingBytes`). The tile is an
/// independent block-linear picture of `tile_cols` x `tile_rows` blocks.
void DecodeSparseTile(const Storage& s, const SparseLayout& layout,
                      std::span<const u8> guest, std::span<u8> out);

/// The upload of one layer's mip tail: the pieces (levels `level.size()` and up, guest offsets
/// relative to the start of the tail) of layer 0; `FillStorageUpload` reads the tail's bytes.
/// Copy the regions to another layer by changing `imageSubresource.baseArrayLayer`.
[[nodiscard]] UploadPlan PlanSparseTail(const Storage& s, const SparseLayout& layout);

} // namespace Vulkan::Nvn
