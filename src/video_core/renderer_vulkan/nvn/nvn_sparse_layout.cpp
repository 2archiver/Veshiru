// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>

#include "video_core/renderer_vulkan/nvn/nvn_sparse_layout.h"
#include "video_core/textures/decoders.h"

namespace Vulkan::Nvn {
namespace {

u32 CeilDiv(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

} // Anonymous namespace

std::optional<SparseLayout> BuildSparseLayout(const Storage& s) {
    // Colour and single-aspect depth (D16, D32F: the guest and host texel layouts are the same);
    // depth+stencil formats are not.
    if (s.type != VK_IMAGE_TYPE_2D || s.samples != VK_SAMPLE_COUNT_1_BIT || s.layers == 0 ||
        s.host.stencil || s.host.guest_layout_differs) {
        return std::nullopt;
    }
    const std::optional<SparseShape> shape = SparseTileShape(s.desc, s.levels);
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!shape || !block.bytes) {
        return std::nullopt;
    }
    SparseLayout layout;
    layout.shape = *shape;
    layout.block = block;
    layout.layers = s.layers;
    layout.levels = s.levels;
    layout.layer_size = s.layout.layer_size;
    layout.tile_width = shape->tile_cols * block.width;
    layout.tile_height = shape->tile_rows * block.height;
    if (layout.layer_size % SPARSE_TILE_BYTES != 0 ||
        s.layout.size != layout.layer_size * (s.layout.layers ? s.layout.layers : 1) ||
        (s.layout.layers ? s.layout.layers : 1) != s.layers) {
        return std::nullopt;
    }
    u64 offset = 0;
    u32 tiles = 0;
    for (u32 level = 0; level < shape->tiled_levels; ++level) {
        SparseLevel entry;
        entry.width = Mip(s.extent.width, level);
        entry.height = Mip(s.extent.height, level);
        entry.tiles_x = CeilDiv(CeilDiv(entry.width, block.width), shape->tile_cols);
        entry.tiles_y = CeilDiv(CeilDiv(entry.height, block.height), shape->tile_rows);
        entry.first_tile = tiles;
        entry.offset = offset;
        if (TextureLevelOffset(s.desc, level) != offset) {
            return std::nullopt; // the layout walk disagrees with the tile walk
        }
        tiles += entry.tiles_x * entry.tiles_y;
        offset += static_cast<u64>(entry.tiles_x) * entry.tiles_y * SPARSE_TILE_BYTES;
        layout.level.push_back(entry);
    }
    layout.tiles_per_layer = tiles;
    layout.tail_offset = offset;
    if (offset > layout.layer_size ||
        (shape->tiled_levels < s.levels && TextureLevelOffset(s.desc, shape->tiled_levels) != offset)) {
        return std::nullopt;
    }
    return layout;
}

SparseTileRef SparseLayout::Tile(u32 id) const {
    SparseTileRef ref;
    ref.layer = id / tiles_per_layer;
    const u32 index = id % tiles_per_layer;
    const auto it = std::upper_bound(level.begin(), level.end(), index,
                                     [](u32 value, const SparseLevel& entry) {
                                         return value < entry.first_tile;
                                     });
    const auto found = std::prev(it);
    ref.level = static_cast<u32>(found - level.begin());
    ref.x = (index - found->first_tile) % found->tiles_x;
    ref.y = (index - found->first_tile) / found->tiles_x;
    return ref;
}

u64 SparseLayout::TileOffset(u32 id) const {
    return static_cast<u64>(id / tiles_per_layer) * layer_size +
           static_cast<u64>(id % tiles_per_layer) * SPARSE_TILE_BYTES;
}

SparseRect SparseLayout::TileRect(const SparseTileRef& ref) const {
    const SparseLevel& entry = level[ref.level];
    SparseRect rect;
    rect.x = ref.x * tile_width;
    rect.y = ref.y * tile_height;
    rect.width = (std::min)(tile_width, entry.width - rect.x);
    rect.height = (std::min)(tile_height, entry.height - rect.y);
    return rect;
}

u64 SparseTileStagingBytes(const Storage& s, const SparseLayout& layout) {
    if (!s.host.needs_transcode) {
        return static_cast<u64>(layout.shape.tile_cols) * layout.shape.tile_rows *
               layout.block.bytes;
    }
    return static_cast<u64>(layout.tile_width) * layout.tile_height * DecodedTexelBytes(s.host);
}

void DecodeSparseTile(const Storage& s, const SparseLayout& layout, std::span<const u8> guest,
                      std::span<u8> out) {
    const u32 bytes = layout.block.bytes;
    const u64 block_bytes = static_cast<u64>(layout.shape.tile_cols) * layout.shape.tile_rows * bytes;
    std::vector<u8> blocks;
    std::span<u8> destination = out;
    if (s.host.needs_transcode) {
        blocks.resize(static_cast<size_t>(block_bytes));
        destination = blocks;
    }
    Tegra::Texture::UnswizzleTexture(destination.first(static_cast<size_t>(block_bytes)), guest,
                                     bytes, layout.shape.tile_cols, layout.shape.tile_rows, 1,
                                     layout.shape.tile_h, 0, 0, true);
    if (s.host.needs_transcode) {
        DecodeBlocks(s.host, layout.block, blocks, layout.tile_width, layout.tile_height, 1, out);
    }
}

UploadPlan PlanSparseTail(const Storage& s, const SparseLayout& layout) {
    UploadPlan full = PlanStorageUpload(s, layout.layer_size);
    UploadPlan plan;
    plan.decode = full.decode;
    plan.linear = full.linear;
    plan.aspect = full.aspect;
    plan.unsupported = full.unsupported;
    u64 total = 0;
    for (UploadPiece piece : full.pieces) {
        if (piece.layer != 0 || piece.level < layout.level.size()) {
            continue;
        }
        piece.guest_offset -= layout.tail_offset;
        total = (total + 63) / 64 * 64;
        piece.staging_offset = total;
        total += piece.staging_size;
        plan.pieces.push_back(piece);
    }
    plan.staging_size = total;
    return plan;
}

} // namespace Vulkan::Nvn
