// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/util.h"

namespace {
constexpr u64 KiB = 1024;
constexpr u64 MiB = 1024 * KiB;

struct SparseRegionScope {
    GPUVAddr address;
    ~SparseRegionScope() {
        Tegra::Nx2SparseRegionDestroyed(address);
    }
};
} // namespace

TEST_CASE("Sparse HDR atlas tracks mapped tiles beyond 4 GiB", "[nx2_memory]") {
    using namespace VideoCommon;
    constexpr GPUVAddr address = 0x9200000000ULL;
    constexpr s32 layer = 67;
    constexpr u32 stride = 0x5560000;
    constexpr u64 layer_offset = u64{layer} * stride;
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::BC6H_UFLOAT;
    info.size = {8192, 8192, 1};
    info.resources = {12, 68};
    info.block = {0, 4, 0};
    info.layer_stride = stride;
    info.native_tiling = true;

    constexpr u64 size = 0x16ad80000ULL;
    REQUIRE(CalculateGuestSizeInBytes(info) == size);
    const ImageBase image(info, address, 0);
    const auto base = image.TryFindBase(address + layer_offset);
    REQUIRE(base);
    CHECK(base->layer == layer);
    CHECK_FALSE(image.TryFindBase(address + size));
    CHECK_FALSE(image.TryFindBase(address + size + 4));

    SparseRegionScope scope{address};
    Tegra::Nx2SparseRegionCreated(address, size, 4 * MiB);
    Tegra::Nx2SparseTilesMapped(address, layer_offset, 64 * KiB, true);
    const auto ranges = Tegra::Nx2SparseDataRanges(address, size);
    const auto rects = SparseTileRects(info, ranges, 256, 256, 5);
    REQUIRE_FALSE(rects.empty());
    for (const LevelRect& rect : rects) {
        CHECK(rect.layer == layer);
        CHECK(rect.level == 0);
    }
    CHECK(UnswizzledRectsBytes(info, rects) > 0);
    CHECK(UnswizzledRectsBytes(info, rects) < MiB);
}

TEST_CASE("Sparse BC6H upload batches multiple GOB rows into one copy", "[nx2_memory]") {
    using namespace VideoCommon;
    Core::System system;
    Core::DeviceMemory device_memory;
    Tegra::MaxwellDeviceMemoryManager smmu(device_memory);
    Tegra::MemoryManager memory(system, smmu, 32);
    memory.EnableNativeTiling();

    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::BC6H_UFLOAT;
    info.size = {128, 128, 1};
    info.resources = {1, 2};
    info.layer_stride = CalculateLayerStride(info);
    info.native_tiling = true;

    const LevelRect rect{.level = 0, .x = 16, .y = 16, .width = 32, .height = 80, .layer = 1};
    const std::span rects{&rect, 1};
    std::vector<u8> staging(UnswizzledRectsBytes(info, rects), 0xff);
    const auto copies = UnswizzleImageRects(memory, 0x100000, info, rects, staging);
    REQUIRE(copies.size() == 1);
    const auto& copy = copies.front();
    CHECK(copy.buffer_offset == 512);
    CHECK(copy.buffer_size == 2560);
    CHECK(copy.buffer_row_length == 32);
    CHECK(copy.buffer_image_height == 80);
    CHECK(copy.image_subresource.base_layer == 1);
    CHECK(copy.image_offset == Offset3D{16, 16, 0});
    CHECK(copy.image_extent == Extent3D{32, 80, 1});
    CHECK(std::ranges::all_of(staging, [](u8 value) { return value == 0; }));
}

TEST_CASE("Sparse atlas ranges split at layer boundaries", "[nx2_memory]") {
    using namespace VideoCommon;
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::BC6H_UFLOAT;
    info.size = {128, 128, 1};
    info.resources = {3, 2};
    info.layer_stride = CalculateLayerStride(info);
    const std::array ranges{std::pair<u64, u64>{info.layer_stride - 512, 1024}};
    const auto rects = SparseTileRects(info, ranges, 64, 64, 1);
    REQUIRE_FALSE(rects.empty());
    CHECK(std::ranges::any_of(rects, [](const auto& rect) { return rect.layer == 0 && rect.level == 2; }));
    CHECK(std::ranges::any_of(rects, [](const auto& rect) { return rect.layer == 1 && rect.level == 0; }));
    CHECK(std::ranges::all_of(rects, [](const auto& rect) { return rect.layer < 2; }));
}
