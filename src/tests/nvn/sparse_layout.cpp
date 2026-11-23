// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// The tile layout of sparse NVN textures and the tile state of their backing: CPU only.

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse_layout.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

namespace {

namespace Nvn = Vulkan::Nvn;
using VideoCore::Surface::PixelFormat;

constexpr s32 NVN_FORMAT_R8 = 0x01;
constexpr s32 NVN_FORMAT_RGBA8 = 0x25;
constexpr s32 NVN_FORMAT_BC1 = 0x42;
constexpr s32 NVN_FORMAT_ASTC_8X8 = 0x80;
constexpr s32 TARGET_2D = 1;
constexpr s32 TARGET_2D_ARRAY = 4;
constexpr s32 FLAG_SPARSE = 0x20;

/// The storage CreateTextureNow builds for a sparse texture (without a VkImage).
Nvn::Storage MakeSparse(s32 format, u32 width, u32 height, u32 levels, u32 layers = 1,
                        bool transcode = false, PixelFormat pixel_format = PixelFormat::R8_UNORM) {
    Nvn::Storage s;
    s.desc = {.target = layers > 1 ? TARGET_2D_ARRAY : TARGET_2D, .format = format,
              .flags = FLAG_SPARSE, .samples = 0, .width = width, .height = height,
              .depth = layers, .levels = levels, .stride = 0, .pkg_block_h = 0};
    s.layout = Nvn::ComputeTextureLayout(s.desc);
    s.host.pixel_format = pixel_format;
    s.host.needs_transcode = transcode;
    s.type = VK_IMAGE_TYPE_2D;
    s.extent = {width, height, 1};
    s.levels = levels;
    s.layers = layers;
    return s;
}

std::vector<u8> Noise(size_t size, u32 seed) {
    std::mt19937 random{seed};
    std::vector<u8> bytes(size);
    for (u8& byte : bytes) {
        byte = static_cast<u8>(random());
    }
    return bytes;
}

std::vector<u32> TilesIn(const Nvn::SparseLayout& layout, u64 begin, u64 end,
                         std::vector<u32>* tails = nullptr) {
    std::vector<u32> tiles;
    layout.ForEachUnit(
        begin, end, [&](u32 id) { tiles.push_back(id); },
        [&](u32 layer) {
            if (tails) {
                tails->push_back(layer);
            }
        });
    return tiles;
}

} // Anonymous namespace

TEST_CASE("nvn sparse: row batching preserves every block mapping", "[nvn][sparse]") {
    constexpr u32 GRANULE = 256;
    std::vector<VkSparseImageMemoryBind> binds;
    // Allocation boundaries, gaps, unbinds, layers, mips, and clipped edge blocks.
    for (u32 layer = 0; layer < 68; ++layer) {
        for (u32 mip = 0; mip < 3; ++mip) {
            for (u32 y = 0; y < 2; ++y) {
                for (u32 x = 0; x < 32; ++x) {
                    const u64 page = ((layer * 3 + mip) * 2 + y) * 32 + x;
                    binds.push_back({
                        .subresource{VK_IMAGE_ASPECT_COLOR_BIT, mip, layer},
                        .offset{static_cast<s32>(x * GRANULE), static_cast<s32>(y * GRANULE), 0},
                        .extent{x == 31 ? 37u : GRANULE, y == 1 ? 53u : GRANULE, 1},
                        .memory = x % 7 == 0 ? VK_NULL_HANDLE
                                             : std::bit_cast<VkDeviceMemory>(u64{1} + page / 64),
                        .memoryOffset = (page % 64 + 2) * Nvn::SPARSE_TILE_BYTES,
                    });
                }
            }
        }
    }
    // Compare the effective per-block bindings, independent of the number/order of records.
    const auto expand = [&](const std::vector<VkSparseImageMemoryBind>& records) {
        using Position = std::tuple<u32, u32, u32, s32, s32, s32>;
        using Mapping = std::tuple<VkDeviceMemory, u64, u32, u32, u32>;
        std::map<Position, Mapping> blocks;
        for (const auto& bind : records) {
            // The map below has no depth: every record covers one slice.
            REQUIRE(bind.extent.depth == 1);
            for (u32 x = 0; x < bind.extent.width; x += GRANULE) {
                const Position pos{bind.subresource.aspectMask,
                                   bind.subresource.arrayLayer,
                                   bind.subresource.mipLevel,
                                   bind.offset.z,
                                   bind.offset.y,
                                   bind.offset.x + static_cast<s32>(x)};
                const u64 offset = bind.memory == VK_NULL_HANDLE
                                       ? 0
                                       : bind.memoryOffset + (x / GRANULE) * Nvn::SPARSE_TILE_BYTES;
                const Mapping mapping{bind.memory, offset, bind.flags,
                                      (std::min)(GRANULE, bind.extent.width - x),
                                      bind.extent.height};
                REQUIRE(blocks.emplace(pos, mapping).second);
            }
        }
        return blocks;
    };
    const auto expected = expand(binds);
    const size_t original_count = binds.size();
    std::mt19937 random{41};
    std::shuffle(binds.begin(), binds.end(), random);
    Nvn::CompactSparseImageBinds(binds, GRANULE, GRANULE);
    CHECK(binds.size() < original_count / 2);
    CHECK(expand(binds) == expected);
    const size_t compact_count = binds.size();
    Nvn::CompactSparseImageBinds(binds, GRANULE, GRANULE);
    CHECK(binds.size() == compact_count);
    CHECK(expand(binds) == expected);
}

TEST_CASE("nvn sparse: row batching keeps incompatible binds separate", "[nvn][sparse]") {
    const VkSparseImageMemoryBind first{
        .subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0},
        .offset{0, 0, 0},
        .extent{256, 256, 1},
        .memory = std::bit_cast<VkDeviceMemory>(u64{1}),
        .memoryOffset = 0,
    };
    VkSparseImageMemoryBind second = first;
    second.offset.x = 256;
    second.memoryOffset = Nvn::SPARSE_TILE_BYTES;
    SECTION("discontiguous physical pages") {
        second.memoryOffset *= 2;
    }
    SECTION("different memory objects") {
        second.memory = std::bit_cast<VkDeviceMemory>(u64{2});
    }
    SECTION("bound followed by unbound") {
        second.memory = VK_NULL_HANDLE;
    }
    SECTION("different binding flags") {
        second.flags = VK_SPARSE_MEMORY_BIND_METADATA_BIT;
    }
    SECTION("different aspect") {
        second.subresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    SECTION("different layer") {
        second.subresource.arrayLayer = 1;
    }
    SECTION("different mip") {
        second.subresource.mipLevel = 1;
    }
    SECTION("different row") {
        second.offset.y = 256;
    }
    SECTION("different depth slice") {
        second.offset.z = 1;
    }
    SECTION("gap between blocks") {
        second.offset.x = 512;
    }
    SECTION("different row height") {
        second.extent.height = 128;
    }
    SECTION("multiple host rows") {
        second.extent.height = 512;
    }
    SECTION("multiple depth slices") {
        second.extent.depth = 2;
    }
    std::vector<VkSparseImageMemoryBind> binds{first, second};
    Nvn::CompactSparseImageBinds(binds, 256, 256);
    CHECK(binds.size() == 2);
}

TEST_CASE("nvn sparse: a tile is 64 KiB of one block row of GOB columns", "[nvn][sparse]") {
    struct Case {
        s32 format;
        u32 block_bytes;
        u32 tile_cols;
        u32 tile_rows;
    };
    // Block size -> tile shape in blocks (the standard sparse block shapes of the hardware).
    const std::array<Case, 4> cases{{
        {NVN_FORMAT_R8, 1, 256, 256},
        {NVN_FORMAT_RGBA8, 4, 128, 128},
        {NVN_FORMAT_BC1, 8, 128, 64},
        {NVN_FORMAT_ASTC_8X8, 16, 64, 64},
    }};
    for (const Case& c : cases) {
        Nvn::Storage s = MakeSparse(c.format, 4096, 4096, 1);
        const auto shape = Nvn::SparseTileShape(s.desc, 1);
        REQUIRE(shape);
        CHECK(shape->tile_cols == c.tile_cols);
        CHECK(shape->tile_rows == c.tile_rows);
        CHECK(u64{shape->tile_cols} * shape->tile_rows * c.block_bytes == Nvn::SPARSE_TILE_BYTES);
    }
    // Not tiled: smaller than one tile, not sparse, multisampled, packaged layout.
    Nvn::Storage small = MakeSparse(NVN_FORMAT_R8, 128, 128, 1);
    CHECK_FALSE(Nvn::SparseTileShape(small.desc, 1));
    Nvn::Storage plain = MakeSparse(NVN_FORMAT_R8, 1024, 1024, 1);
    plain.desc.flags = 0;
    CHECK_FALSE(Nvn::SparseTileShape(plain.desc, 1));
    Nvn::Storage packaged = MakeSparse(NVN_FORMAT_R8, 1024, 1024, 1);
    packaged.desc.pkg_block_h = 4;
    CHECK_FALSE(Nvn::SparseTileShape(packaged.desc, 1));
}

TEST_CASE("nvn sparse: single-aspect depth is tiled, depth+stencil is not", "[nvn][sparse]") {
    constexpr s32 NVN_FORMAT_DEPTH16 = 0x32;
    // The shadow-map arrays of Mario Kart World: 512x512 depth16 layers.
    Nvn::Storage depth = MakeSparse(NVN_FORMAT_DEPTH16, 512, 512, 4, 101);
    depth.host.depth = true;
    const auto layout = Nvn::BuildSparseLayout(depth);
    REQUIRE(layout);
    CHECK(layout->layers == 101);
    CHECK(layout->block.bytes == 2);
    CHECK(u64{layout->shape.tile_cols} * layout->shape.tile_rows * layout->block.bytes ==
          Nvn::SPARSE_TILE_BYTES);
    Nvn::Storage stencil = MakeSparse(NVN_FORMAT_DEPTH16, 512, 512, 4, 101);
    stencil.host.depth = true;
    stencil.host.stencil = true;
    CHECK_FALSE(Nvn::BuildSparseLayout(stencil));
    Nvn::Storage differs = MakeSparse(NVN_FORMAT_DEPTH16, 512, 512, 4, 101);
    differs.host.depth = true;
    differs.host.guest_layout_differs = true;
    CHECK_FALSE(Nvn::BuildSparseLayout(differs));
}

TEST_CASE("nvn sparse: the tiled levels come first, the mip tail follows", "[nvn][sparse]") {
    // 1024x512 R8: level 0 is 4x2 tiles, level 1 (512x256) is 2x1, the rest is the tail.
    Nvn::Storage s = MakeSparse(NVN_FORMAT_R8, 1024, 512, 5);
    const auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    REQUIRE(layout->level.size() == 2);
    CHECK(layout->level[0].tiles_x == 4);
    CHECK(layout->level[0].tiles_y == 2);
    CHECK(layout->level[1].tiles_x == 2);
    CHECK(layout->level[1].tiles_y == 1);
    CHECK(layout->level[1].first_tile == 8);
    CHECK(layout->tiles_per_layer == 10);
    CHECK(layout->tail_offset == 10 * Nvn::SPARSE_TILE_BYTES);
    CHECK(layout->tail_offset == Nvn::TextureLevelOffset(s.desc, 2));
    CHECK(layout->layer_size == s.layout.layer_size);
    REQUIRE(layout->HasTail());
    CHECK(layout->TailSize() == layout->layer_size - layout->tail_offset);

    const Nvn::SparseTileRef first = layout->Tile(0);
    CHECK((first.level == 0 && first.x == 0 && first.y == 0));
    const Nvn::SparseTileRef row = layout->Tile(5);
    CHECK((row.level == 0 && row.x == 1 && row.y == 1));
    const Nvn::SparseTileRef next = layout->Tile(9);
    CHECK((next.level == 1 && next.x == 1 && next.y == 0));
    CHECK(layout->TileOffset(9) == 9 * Nvn::SPARSE_TILE_BYTES);

    // The tail's pieces are the levels the tiles do not hold, offsets counted from the tail.
    const Nvn::UploadPlan tail = Nvn::PlanSparseTail(s, *layout);
    REQUIRE(tail.unsupported == nullptr);
    REQUIRE(tail.pieces.size() == 3);
    for (u32 i = 0; i < tail.pieces.size(); ++i) {
        CHECK(tail.pieces[i].level == 2 + i);
        CHECK(tail.pieces[i].layer == 0);
        CHECK(tail.pieces[i].guest_offset ==
              Nvn::TextureLevelOffset(s.desc, 2 + i) - layout->tail_offset);
    }

    // Only tiled levels: no tail.
    Nvn::Storage tiled_only = MakeSparse(NVN_FORMAT_R8, 1024, 512, 2);
    const auto no_tail = Nvn::BuildSparseLayout(tiled_only);
    REQUIRE(no_tail);
    CHECK_FALSE(no_tail->HasTail());
}

TEST_CASE("nvn sparse: bytes of the storage map to their tiles and tails", "[nvn][sparse]") {
    // Three layers of 1024x512 R8 with a tail each.
    Nvn::Storage s = MakeSparse(NVN_FORMAT_R8, 1024, 512, 4, 3);
    const auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    REQUIRE(layout->layers == 3);
    const u64 layer = layout->layer_size;
    CHECK(layout->TileCount() == 3 * layout->tiles_per_layer);
    CHECK(layout->TileOffset(layout->tiles_per_layer) == layer);

    // One byte of one tile.
    CHECK(TilesIn(*layout, 70000, 70001) == std::vector<u32>{1});
    // A range over a tile boundary takes both tiles.
    CHECK(TilesIn(*layout, 0x10000 - 1, 0x10000 + 1) == std::vector<u32>{0, 1});
    // The last tile of layer 0 and the first one of layer 1.
    std::vector<u32> tails;
    const std::vector<u32> across =
        TilesIn(*layout, layout->tail_offset - 1, layer + 1, &tails);
    CHECK(across == std::vector<u32>{layout->tiles_per_layer - 1, layout->tiles_per_layer});
    CHECK(tails == std::vector<u32>{0});
    // A range inside a tail names only the tail.
    tails.clear();
    CHECK(TilesIn(*layout, 2 * layer + layout->tail_offset + 5, 2 * layer + layout->tail_offset + 9,
                  &tails)
              .empty());
    CHECK(tails == std::vector<u32>{2});
    // Past the storage nothing is left.
    CHECK(TilesIn(*layout, 3 * layer, 4 * layer).empty());
    // The whole storage: every tile and every tail.
    tails.clear();
    CHECK(TilesIn(*layout, 0, 3 * layer, &tails).size() == layout->TileCount());
    CHECK(tails == std::vector<u32>{0, 1, 2});
}

TEST_CASE("MKW BC6H sparse layout retains lightmap layers beyond 4 GiB", "[nvn][mkw][sparse]") {
    Nvn::Storage s = MakeSparse(0x50, 8192, 8192, 12, 68, false, PixelFormat::BC6H_UFLOAT);
    const auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    CHECK(s.layout.size == u64{68} * layout->layer_size);
    REQUIRE(s.layout.size > (u64{4} << 30));
    const u32 last_layer_tile = 67 * layout->tiles_per_layer;
    const u64 last_layer_offset = layout->TileOffset(last_layer_tile);
    REQUIRE(last_layer_offset > (u64{4} << 30));
    CHECK(layout->Tile(last_layer_tile).layer == 67);
    CHECK(TilesIn(*layout, last_layer_offset, last_layer_offset + Nvn::SPARSE_TILE_BYTES) ==
          std::vector<u32>{last_layer_tile});
    const auto backing = Nvn::SparseBacking::FromLayout(*layout);
    backing->Remap(last_layer_offset, last_layer_offset + Nvn::SPARSE_TILE_BYTES,
                   [](u64, u64) { return true; });
    CHECK(backing->MappedTiles() == 1);
    const auto tail = Nvn::PlanSparseTail(s, *layout);
    REQUIRE(tail.unsupported == nullptr);
    REQUIRE_FALSE(tail.pieces.empty());
    for (const auto& piece : tail.pieces) {
        CHECK(piece.guest_offset + layout->tail_offset ==
              Nvn::TextureLevelOffset(s.desc, piece.level));
        CHECK(piece.guest_offset + layout->tail_offset + last_layer_offset < s.layout.size);
    }
}

TEST_CASE("nvn sparse: edge tiles are cut to the level", "[nvn][sparse]") {
    Nvn::Storage s = MakeSparse(NVN_FORMAT_R8, 1100, 300, 1);
    const auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    REQUIRE(layout->level.size() == 1);
    CHECK(layout->level[0].tiles_x == 5);
    CHECK(layout->level[0].tiles_y == 2);
    const Nvn::SparseRect inner = layout->TileRect({0, 0, 3, 0});
    CHECK((inner.x == 768 && inner.y == 0 && inner.width == 256 && inner.height == 256));
    const Nvn::SparseRect corner = layout->TileRect({0, 0, 4, 1});
    CHECK((corner.x == 1024 && corner.y == 256 && corner.width == 76 && corner.height == 44));
}

TEST_CASE("nvn sparse: a tile decodes like the same texels of a whole block-linear level",
          "[nvn][sparse]") {
    // 1024x512 R8, one level: the whole level swizzled in one go is the tiles one after another.
    Nvn::Storage s = MakeSparse(NVN_FORMAT_R8, 1024, 512, 1);
    const auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    const std::vector<u8> texels = Noise(1024 * 512, 7);
    std::vector<u8> guest(static_cast<size_t>(s.layout.size), 0);
    u32 tile_h = 0;
    u32 tile_d = 0;
    Nvn::TextureLevelTiles(s.desc, 0, tile_h, tile_d);
    Tegra::Texture::SwizzleTexture(guest, texels, 1, 1024, 512, 1, tile_h, tile_d, 0, true);

    const u64 staging = Nvn::SparseTileStagingBytes(s, *layout);
    REQUIRE(staging == 256 * 256);
    for (u32 id = 0; id < layout->TileCount(); ++id) {
        std::vector<u8> out(static_cast<size_t>(staging));
        Nvn::DecodeSparseTile(
            s, *layout,
            std::span<const u8>{guest.data() + layout->TileOffset(id), Nvn::SPARSE_TILE_BYTES}, out);
        const Nvn::SparseTileRef ref = layout->Tile(id);
        const Nvn::SparseRect rect = layout->TileRect(ref);
        for (u32 y = 0; y < rect.height; ++y) {
            REQUIRE(std::memcmp(out.data() + y * layout->tile_width,
                                texels.data() + (rect.y + y) * 1024 + rect.x, rect.width) == 0);
        }
    }
}

TEST_CASE("nvn sparse: transcoded formats stage decoded texels", "[nvn][sparse]") {
    // ASTC 8x8 on a device without ASTC: a 64 KiB tile is 512x512 texels, held as RGBA8.
    Nvn::Storage s = MakeSparse(NVN_FORMAT_ASTC_8X8, 4096, 4096, 1, 1, true,
                                PixelFormat::ASTC_2D_8X8_UNORM);
    const auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    CHECK(layout->tile_width == 512);
    CHECK(layout->tile_height == 512);
    CHECK(Nvn::SparseTileStagingBytes(s, *layout) == 512 * 512 * 4);
    // Native BC1 stays blocks.
    Nvn::Storage native = MakeSparse(NVN_FORMAT_BC1, 4096, 4096, 1, 1, false,
                                     PixelFormat::BC1_RGBA_UNORM);
    const auto native_layout = Nvn::BuildSparseLayout(native);
    REQUIRE(native_layout);
    CHECK(native_layout->tile_width == 512);
    CHECK(native_layout->tile_height == 256);
    CHECK(Nvn::SparseTileStagingBytes(native, *native_layout) == Nvn::SPARSE_TILE_BYTES);
}

TEST_CASE("nvn sparse: a remap marks the tiles of its range, not the storage", "[nvn][sparse]") {
    Nvn::Storage s = MakeSparse(NVN_FORMAT_R8, 1024, 512, 4);
    auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    const auto backing = Nvn::SparseBacking::FromLayout(*layout);
    CHECK_FALSE(backing->HasWork());
    CHECK(backing->MappedTiles() == 0);

    // The pool maps tiles 2 and 3 only: a remap of tiles 1..4 marks those two as mapped.
    const auto mapped = [](u64 offset, u64 size) {
        return offset < 4 * Nvn::SPARSE_TILE_BYTES && offset + size > 2 * Nvn::SPARSE_TILE_BYTES;
    };
    backing->Remap(1 * Nvn::SPARSE_TILE_BYTES, 5 * Nvn::SPARSE_TILE_BYTES,
                   [&](u64 offset, u64 size) { return mapped(offset, size); });
    CHECK(backing->MappedTiles() == 2);
    CHECK(backing->HasWork());

    // Tiles that were not asked about stay as they are.
    backing->Remap(0, Nvn::SPARSE_TILE_BYTES, [](u64, u64) { return true; });
    CHECK(backing->MappedTiles() == 3);
    // Unmapping a tile that never got memory only drops its mark.
    backing->Remap(0, Nvn::SPARSE_TILE_BYTES, [](u64, u64) { return false; });
    CHECK(backing->MappedTiles() == 2);

    // The tail has its own mark, off the tile count.
    backing->Remap(layout->tail_offset, layout->layer_size, [](u64, u64) { return true; });
    CHECK(backing->MappedTiles() == 2);

    // Nothing is verified before a tile has memory.
    CHECK(backing->NextVerifyTile() == ~0u);
}

TEST_CASE("nvn sparse: CPU writes mark only the mapped tiles they touch", "[nvn][sparse]") {
    Nvn::Storage s = MakeSparse(NVN_FORMAT_R8, 1024, 512, 1);
    auto layout = Nvn::BuildSparseLayout(s);
    REQUIRE(layout);
    const auto backing = Nvn::SparseBacking::FromLayout(*layout);
    // Tile 1 mapped, then its work taken away (as a Prepare would).
    backing->Remap(Nvn::SPARSE_TILE_BYTES, 2 * Nvn::SPARSE_TILE_BYTES,
                   [](u64, u64) { return true; });
    CHECK(backing->HasWork());
    // A change over unmapped tiles marks nothing that was not marked already.
    backing->ContentChanged(5 * Nvn::SPARSE_TILE_BYTES, 6 * Nvn::SPARSE_TILE_BYTES);
    CHECK(backing->MappedTiles() == 1);
}
