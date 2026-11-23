// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/textures/decoders.h"

namespace {

using NvnHleTest::Rig;

constexpr u64 POOL = Rig::Obj(1);
constexpr u64 TEXTURE = Rig::Obj(2);
constexpr u64 TEXTURE2 = Rig::Obj(3);
constexpr u64 BUILDER = Rig::Obj(4);
constexpr u64 BUILDER2 = Rig::Obj(5);
constexpr u64 VIEW = Rig::Obj(6);
constexpr u64 VIEW2 = Rig::Obj(7);
constexpr u64 SAMPLER = Rig::Obj(8);
constexpr u64 SAMPLER2 = Rig::Obj(9);
constexpr u64 SAMPLER_BUILDER = Rig::Obj(10);
constexpr u64 SAMPLER_BUILDER2 = Rig::Obj(11);
constexpr u64 OUT = Rig::SCRATCH + 0x1000;   // out parameters
constexpr u64 INPUT = Rig::SCRATCH + 0x2000; // by-pointer inputs
constexpr u64 REGION = Rig::SCRATCH + 0x3000;
constexpr u64 DATA = Rig::SCRATCH + 0x4000;
constexpr u64 DATA2 = Rig::SCRATCH + 0x5000;

constexpr s32 NVN_TARGET_2D = 1;
constexpr s32 NVN_TARGET_3D = 2;
constexpr s32 NVN_TARGET_2D_ARRAY = 4;
constexpr s32 NVN_FORMAT_R8 = 0x01;
constexpr s32 NVN_FORMAT_RGBA8 = 0x25;
constexpr s32 NVN_FORMAT_BPTC_UNORM = 0x4D;
constexpr s32 NVN_FORMAT_ASTC_4x4 = 0x79;
constexpr s32 NVN_FORMAT_ASTC_8x8 = 0x80;
constexpr s32 FLAG_LINEAR = 0x10;
constexpr s32 FLAG_SPARSE = 0x20;

NvnHleTest::Arg I(u64 value) {
    return Rig::I(value);
}

void Fill(Rig& rig, u64 va, size_t size, u8 value) {
    const std::vector<u8> bytes(size, value);
    REQUIRE(rig.memory.Write(va, bytes.data(), size));
}

std::vector<u8> Bytes(Rig& rig, u64 va, size_t size) {
    std::vector<u8> bytes(size);
    REQUIRE(rig.memory.Read(va, bytes.data(), size));
    return bytes;
}

template <typename T>
void Put(std::vector<u8>& bytes, size_t offset, T value) {
    REQUIRE(offset + sizeof(T) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

template <typename T>
T Get(Rig& rig, u64 va) {
    return rig.memory.Get<T>(va);
}

struct TexSpec {
    s32 target = NVN_TARGET_2D;
    s32 format = NVN_FORMAT_RGBA8;
    s32 flags = 0;
    s32 samples = 0;
    s32 width = 1;
    s32 height = 1;
    s32 depth = 1;
    s32 levels = 1;
    s64 stride = 0;
};

void Build(Rig& rig, u64 builder, const TexSpec& spec) {
    rig.Call("nvnTextureBuilderSetDefaults", {I(builder)});
    rig.Call("nvnTextureBuilderSetTarget", {I(builder), I(static_cast<u32>(spec.target))});
    rig.Call("nvnTextureBuilderSetFormat", {I(builder), I(static_cast<u32>(spec.format))});
    rig.Call("nvnTextureBuilderSetFlags", {I(builder), I(static_cast<u32>(spec.flags))});
    rig.Call("nvnTextureBuilderSetSamples", {I(builder), I(static_cast<u32>(spec.samples))});
    rig.Call("nvnTextureBuilderSetWidth", {I(builder), I(static_cast<u32>(spec.width))});
    rig.Call("nvnTextureBuilderSetHeight", {I(builder), I(static_cast<u32>(spec.height))});
    rig.Call("nvnTextureBuilderSetDepth", {I(builder), I(static_cast<u32>(spec.depth))});
    rig.Call("nvnTextureBuilderSetLevels", {I(builder), I(static_cast<u32>(spec.levels))});
    rig.Call("nvnTextureBuilderSetStride", {I(builder), I(static_cast<u64>(spec.stride))});
}

Vulkan::Nvn::TextureLayoutDesc LayoutOf(const TexSpec& spec) {
    return {.target = spec.target,
            .format = spec.format,
            .flags = spec.flags,
            .samples = spec.samples,
            .width = static_cast<u32>(spec.width),
            .height = static_cast<u32>(spec.height),
            .depth = static_cast<u32>(spec.depth),
            .levels = static_cast<u32>(spec.levels),
            .stride = static_cast<u32>(spec.stride),
            .pkg_block_h = 0};
}

void MakePool(Rig& rig) {
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
}

/// Builds a texture from `spec` in the pool at `offset` and initializes it at `texture`.
u64 MakeTexture(Rig& rig, u64 texture, u64 builder, const TexSpec& spec, u64 offset) {
    Build(rig, builder, spec);
    rig.Call("nvnTextureBuilderSetStorage", {I(builder), I(POOL), I(offset)});
    return rig.Call("nvnTextureInitialize", {I(texture), I(builder)}) & 0xFF;
}

} // namespace

TEST_CASE("nvn hle texture: builder defaults write the nxnative layout verbatim", "[nvn]") {
    Rig rig;
    Fill(rig, BUILDER, 0x80, 0xCC);
    rig.Call("nvnTextureBuilderSetDevice", {I(BUILDER), I(0x1122334455667788ULL)});
    rig.Call("nvnTextureBuilderSetDefaults", {I(BUILDER)});

    // tex_builder_t is 0x60 bytes: SetDefaults clears all of it except the device and, as the SDK
    // (3.2.0 sdk+0x5ecb60), sets target 2D with width, height and depth 0 and one level. Bytes
    // behind it belong to the (larger) SDK object.
    std::vector<u8> expected(0x80, 0xCC);
    std::fill(expected.begin(), expected.begin() + 0x60, 0);
    Put<u64>(expected, 0x00, 0x1122334455667788ULL);
    Put<s32>(expected, 0x08, 1);
    Put<u32>(expected, 0x24, 1);
    REQUIRE(Bytes(rig, BUILDER, 0x80) == expected);
}

TEST_CASE("nvn hle texture: every builder setter lands at its nxnative offset", "[nvn]") {
    Rig rig;
    rig.Call("nvnTextureBuilderSetDefaults", {I(BUILDER)});
    rig.Call("nvnTextureBuilderSetDevice", {I(BUILDER), I(0xD0D0D0D0ULL)});
    rig.Call("nvnTextureBuilderSetTarget", {I(BUILDER), I(NVN_TARGET_2D_ARRAY)});
    rig.Call("nvnTextureBuilderSetFormat", {I(BUILDER), I(NVN_FORMAT_RGBA8)});
    rig.Call("nvnTextureBuilderSetFlags", {I(BUILDER), I(0x1234)});
    rig.Call("nvnTextureBuilderSetSamples", {I(BUILDER), I(4)});
    rig.Call("nvnTextureBuilderSetWidth", {I(BUILDER), I(640)});
    rig.Call("nvnTextureBuilderSetHeight", {I(BUILDER), I(480)});
    rig.Call("nvnTextureBuilderSetDepth", {I(BUILDER), I(7)});
    rig.Call("nvnTextureBuilderSetLevels", {I(BUILDER), I(9)});
    // ptrdiff_t: only the low 32 bits are kept (nxnative stores (uint32_t)s).
    rig.Call("nvnTextureBuilderSetStride", {I(BUILDER), I(0x1'0000'0180ULL)});
    rig.Call("nvnTextureBuilderSetStorage", {I(BUILDER), I(POOL), I(0x2000)});
    rig.memory.Put(INPUT, std::array<u8, 8>{0x35, 1, 2, 3, 4, 5, 6, 7});
    rig.Call("nvnTextureBuilderSetPackagedTextureLayout", {I(BUILDER), I(INPUT)});
    rig.Call("nvnTextureBuilderSetPackagedTextureData", {I(BUILDER), I(0xABCDEF00)});
    // swizzle (ZERO, ONE, A, R) = 0 | 1 << 3 | 5 << 6 | 2 << 9
    rig.Call("nvnTextureBuilderSetSwizzle", {I(BUILDER), I(0), I(1), I(5), I(2)});
    rig.Call("nvnTextureBuilderSetDepthStencilMode", {I(BUILDER), I(1)});
    rig.Call("nvnTextureBuilderSetGLTextureName", {I(BUILDER), I(77)});

    std::vector<u8> expected(0x60, 0);
    Put<u64>(expected, 0x00, 0xD0D0D0D0ULL);
    Put<s32>(expected, 0x08, NVN_TARGET_2D_ARRAY);
    Put<s32>(expected, 0x0C, NVN_FORMAT_RGBA8);
    Put<s32>(expected, 0x10, 0x1234);
    Put<s32>(expected, 0x14, 4);
    Put<u32>(expected, 0x18, 640);
    Put<u32>(expected, 0x1C, 480);
    Put<u32>(expected, 0x20, 7);
    Put<u32>(expected, 0x24, 9);
    Put<u32>(expected, 0x28, 0x180);
    Put<s32>(expected, 0x2C, 0x36); // pkg_block_h = layout byte 0 + 1
    Put<u64>(expected, 0x30, POOL);
    Put<u64>(expected, 0x38, 0x2000);
    Put<u8>(expected, 0x40, 1); // has_layout
    const std::array<u8, 8> layout{0x35, 1, 2, 3, 4, 5, 6, 7};
    std::memcpy(expected.data() + 0x41, layout.data(), 8);
    Put<u64>(expected, 0x50, 0xABCDEF00);
    Put<u8>(expected, 0x58, 3); // set_swz | stencil << 1
    Put<u16>(expected, 0x5A, 0x548);
    Put<u32>(expected, 0x5C, 77);
    REQUIRE(Bytes(rig, BUILDER, 0x60) == expected);

    SECTION("getters read the same bytes back") {
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetFlags", {I(BUILDER)})) == 0x1234);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetTarget", {I(BUILDER)})) ==
              NVN_TARGET_2D_ARRAY);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetWidth", {I(BUILDER)})) == 640);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetHeight", {I(BUILDER)})) == 480);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetDepth", {I(BUILDER)})) == 7);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetLevels", {I(BUILDER)})) == 9);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetFormat", {I(BUILDER)})) ==
              NVN_FORMAT_RGBA8);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetSamples", {I(BUILDER)})) == 4);
        CHECK(rig.Call("nvnTextureBuilderGetStride", {I(BUILDER)}) == 0x180);
        CHECK(rig.Call("nvnTextureBuilderGetMemoryPool", {I(BUILDER)}) == POOL);
        CHECK(rig.Call("nvnTextureBuilderGetMemoryOffset", {I(BUILDER)}) == 0x2000);
        CHECK(rig.Call("nvnTextureBuilderGetPackagedTextureData", {I(BUILDER)}) == 0xABCDEF00);
        CHECK(rig.Call("nvnTextureBuilderGetPackagedTextureLayout", {I(BUILDER)}) == BUILDER + 0x41);
        CHECK(rig.Call("nvnTextureBuilderGetDevice", {I(BUILDER)}) == 0xD0D0D0D0ULL);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetDepthStencilMode", {I(BUILDER)})) == 1);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetGLTextureName", {I(BUILDER)})) == 77);
        rig.Call("nvnTextureBuilderGetSwizzle", {I(BUILDER), I(OUT), I(OUT + 4), I(OUT + 8), I(OUT + 12)});
        CHECK(Get<std::array<s32, 4>>(rig, OUT) == std::array<s32, 4>{0, 1, 5, 2});
    }

    SECTION("SetDepthStencilMode(depth) clears only the stencil bit") {
        rig.Call("nvnTextureBuilderSetDepthStencilMode", {I(BUILDER), I(0)});
        CHECK(Get<u8>(rig, BUILDER + 0x58) == 1);
        CHECK(static_cast<u32>(rig.Call("nvnTextureBuilderGetDepthStencilMode", {I(BUILDER)})) == 0);
    }

    SECTION("a null packaged layout clears the flag and the block height") {
        rig.Call("nvnTextureBuilderSetPackagedTextureLayout", {I(BUILDER), I(0)});
        CHECK(Get<u8>(rig, BUILDER + 0x40) == 0);
        CHECK(Get<s32>(rig, BUILDER + 0x2C) == 0);
        CHECK(rig.Call("nvnTextureBuilderGetPackagedTextureLayout", {I(BUILDER)}) == 0);
    }

    SECTION("the swizzle is the identity until SetSwizzle is called") {
        rig.Call("nvnTextureBuilderSetDefaults", {I(BUILDER2)});
        rig.Call("nvnTextureBuilderGetSwizzle",
                 {I(BUILDER2), I(OUT), I(OUT + 4), I(OUT + 8), I(OUT + 12)});
        CHECK(Get<std::array<s32, 4>>(rig, OUT) == std::array<s32, 4>{2, 3, 4, 5});
    }
}

TEST_CASE("nvn hle texture: Size1D, Size2D and Size3D set the three dimensions", "[nvn]") {
    Rig rig;
    rig.Call("nvnTextureBuilderSetDefaults", {I(BUILDER)});
    rig.Call("nvnTextureBuilderSetSize3D", {I(BUILDER), I(32), I(16), I(8)});
    CHECK(Get<std::array<u32, 3>>(rig, BUILDER + 0x18) == std::array<u32, 3>{32, 16, 8});
    rig.Call("nvnTextureBuilderSetSize2D", {I(BUILDER), I(100), I(50)});
    CHECK(Get<std::array<u32, 3>>(rig, BUILDER + 0x18) == std::array<u32, 3>{100, 50, 1});
    rig.Call("nvnTextureBuilderSetSize1D", {I(BUILDER), I(256)});
    CHECK(Get<std::array<u32, 3>>(rig, BUILDER + 0x18) == std::array<u32, 3>{256, 1, 1});
}

TEST_CASE("nvn hle texture: TextureInitialize produces the expected TextureDesc", "[nvn]") {
    Rig rig;
    MakePool(rig);
    const auto pool = rig.recorder.GetMemoryPool(POOL);
    REQUIRE(pool);

    Build(rig, BUILDER, {.target = NVN_TARGET_2D_ARRAY,
                         .format = NVN_FORMAT_RGBA8,
                         .flags = 0x4,
                         .samples = 1,
                         .width = 64,
                         .height = 32,
                         .depth = 3,
                         .levels = 5,
                         .stride = 0});
    rig.Call("nvnTextureBuilderSetSwizzle", {I(BUILDER), I(0), I(1), I(5), I(2)});
    rig.Call("nvnTextureBuilderSetDepthStencilMode", {I(BUILDER), I(1)});
    rig.memory.Put(INPUT, std::array<u8, 8>{0x24, 9, 8, 7, 6, 5, 4, 3});
    rig.Call("nvnTextureBuilderSetPackagedTextureLayout", {I(BUILDER), I(INPUT)});
    rig.Call("nvnTextureBuilderSetStorage", {I(BUILDER), I(POOL), I(0x4000)});

    REQUIRE((rig.Call("nvnTextureInitialize", {I(TEXTURE), I(BUILDER)}) & 0xFF) == 1);
    REQUIRE(rig.sink.textures.size() == 1);
    const auto& desc = rig.sink.textures.back();
    CHECK(desc.target == NVN_TARGET_2D_ARRAY);
    CHECK(desc.format == NVN_FORMAT_RGBA8);
    CHECK(desc.flags == 0x4);
    CHECK(desc.samples == 1);
    CHECK(desc.width == 64);
    CHECK(desc.height == 32);
    CHECK(desc.depth == 3);
    CHECK(desc.levels == 5);
    CHECK(desc.stride == 0);
    CHECK(desc.pkg_block_h == 0x25);
    CHECK(desc.has_layout == 1);
    CHECK(std::array<u8, 8>{desc.layout[0], desc.layout[1], desc.layout[2], desc.layout[3],
                            desc.layout[4], desc.layout[5], desc.layout[6], desc.layout[7]} ==
          std::array<u8, 8>{0x24, 9, 8, 7, 6, 5, 4, 3});
    CHECK(desc.set_swizzle == 1);
    CHECK(desc.stencil == 1);
    CHECK(desc.swizzle == 0x548);
    CHECK(desc.pkg_data == 0);
    CHECK(desc.serial != 0);
    CHECK(desc.pool_serial == pool->serial);
    CHECK(desc.pool_offset == 0x4000);
    CHECK(desc.view_of == 0);

    // The object carries the tag and nothing else the HLE wrote.
    const auto tag = Get<VideoCommon::Nvn::GuestTag>(rig, TEXTURE);
    CHECK(tag.magic == VideoCommon::Nvn::GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(VideoCommon::Nvn::ObjectKind::Texture));
    CHECK(tag.serial == desc.serial);

    SECTION("getters read the Recorder's copy") {
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetFlags", {I(TEXTURE)})) == 0x4);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetTarget", {I(TEXTURE)})) == NVN_TARGET_2D_ARRAY);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetWidth", {I(TEXTURE)})) == 64);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetHeight", {I(TEXTURE)})) == 32);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetDepth", {I(TEXTURE)})) == 3);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetLevels", {I(TEXTURE)})) == 5);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetFormat", {I(TEXTURE)})) == NVN_FORMAT_RGBA8);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetSamples", {I(TEXTURE)})) == 1);
        CHECK(rig.Call("nvnTextureGetStride", {I(TEXTURE)}) == 0);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetDepthStencilMode", {I(TEXTURE)})) == 1);
        CHECK(rig.Call("nvnTextureGetMemoryPool", {I(TEXTURE)}) == POOL);
        CHECK(rig.Call("nvnTextureGetMemoryOffset", {I(TEXTURE)}) == 0x4000);
        CHECK(rig.Call("nvnTextureGetTextureAddress", {I(TEXTURE)}) == pool->address + 0x4000);
        CHECK(rig.Call("nvnTextureGetDebugID", {I(TEXTURE)}) == desc.serial);
        rig.Call("nvnTextureGetSwizzle", {I(TEXTURE), I(OUT), I(OUT + 4), I(OUT + 8), I(OUT + 12)});
        CHECK(Get<std::array<s32, 4>>(rig, OUT) == std::array<s32, 4>{0, 1, 5, 2});
        // The SDK sizes the packaged layout from its first two bytes: block height 4, depth 2 and
        // a tile width of 2^9 GOBs (byte 1), which rounds every layer to a 3 x 0x100000 tile
        // (value of the SDK code, 3.2.0 sdk+0x5ed1d0).
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetStorageSize", {I(TEXTURE)})) == 0x3000000);
    }

    SECTION("Compare looks at the description, the storage and the channel mapping") {
        CHECK((rig.Call("nvnTextureCompare", {I(TEXTURE), I(TEXTURE)}) & 0xFF) == 1);
        // Same storage and description, another channel mapping: different.
        rig.Call("nvnTextureBuilderSetSwizzle", {I(BUILDER), I(2), I(3), I(4), I(5)});
        REQUIRE((rig.Call("nvnTextureInitialize", {I(TEXTURE2), I(BUILDER)}) & 0xFF) == 1);
        CHECK((rig.Call("nvnTextureCompare", {I(TEXTURE), I(TEXTURE2)}) & 0xFF) == 0);
        // The mapping set explicitly to the identity equals no mapping at all.
        std::vector<u8> plain = Bytes(rig, BUILDER, 0x60);
        plain[0x58] &= static_cast<u8>(~1u); // clear set_swizzle only; keep the stencil mode
        Put<u16>(plain, 0x5A, 0);
        REQUIRE(rig.memory.Write(BUILDER2, plain.data(), plain.size()));
        REQUIRE((rig.Call("nvnTextureInitialize", {I(Rig::Obj(12)), I(BUILDER2)}) & 0xFF) == 1);
        CHECK((rig.Call("nvnTextureCompare", {I(TEXTURE2), I(Rig::Obj(12))}) & 0xFF) == 1);
    }

    SECTION("Finalize clears the tag: getters answer 0") {
        rig.Call("nvnTextureFinalize", {I(TEXTURE)});
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetFlags", {I(TEXTURE)})) == 0);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetWidth", {I(TEXTURE)})) == 0);
        CHECK(rig.Call("nvnTextureGetMemoryPool", {I(TEXTURE)}) == 0);
        CHECK(rig.Call("nvnTextureGetTextureAddress", {I(TEXTURE)}) == 0);
    }

    SECTION("re-initializing a live object gives it a new serial and storage") {
        const u64 first_serial = desc.serial;
        rig.Call("nvnTextureBuilderSetStorage", {I(BUILDER), I(POOL), I(0x8000)});
        rig.Call("nvnTextureBuilderSetWidth", {I(BUILDER), I(16)});
        REQUIRE((rig.Call("nvnTextureInitialize", {I(TEXTURE), I(BUILDER)}) & 0xFF) == 1);
        REQUIRE(rig.sink.textures.size() == 2);
        CHECK(rig.sink.textures.back().serial != first_serial);
        CHECK(static_cast<u32>(rig.Call("nvnTextureGetWidth", {I(TEXTURE)})) == 16);
        CHECK(rig.Call("nvnTextureGetTextureAddress", {I(TEXTURE)}) == pool->address + 0x8000);
    }
}

TEST_CASE("nvn hle texture: TextureInitialize fails without a pool", "[nvn]") {
    Rig rig;
    Build(rig, BUILDER, {.width = 8, .height = 8});
    // No SetStorage: pool 0.
    CHECK((rig.Call("nvnTextureInitialize", {I(TEXTURE), I(BUILDER)}) & 0xFF) == 0);
    CHECK(rig.sink.textures.empty());
    // A pool that was never initialized.
    rig.Call("nvnTextureBuilderSetStorage", {I(BUILDER), I(POOL), I(0)});
    CHECK((rig.Call("nvnTextureInitialize", {I(TEXTURE), I(BUILDER)}) & 0xFF) == 0);
    CHECK(rig.sink.textures.empty());
}

TEST_CASE("nvn hle texture: packaged texture data is copied into the storage", "[nvn]") {
    Rig rig;
    MakePool(rig);
    const TexSpec spec{.format = NVN_FORMAT_R8, .flags = FLAG_LINEAR, .width = 16, .height = 4,
                       .stride = 32};
    const auto size = Vulkan::Nvn::ComputeTextureLayout(LayoutOf(spec)).size;
    REQUIRE(size == 128);
    std::vector<u8> pattern(size);
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<u8>(i * 7 + 1);
    }
    REQUIRE(rig.memory.Write(DATA, pattern.data(), pattern.size()));
    Build(rig, BUILDER, spec);
    rig.Call("nvnTextureBuilderSetStorage", {I(BUILDER), I(POOL), I(0x100)});
    rig.Call("nvnTextureBuilderSetPackagedTextureData", {I(BUILDER), I(DATA)});
    REQUIRE((rig.Call("nvnTextureInitialize", {I(TEXTURE), I(BUILDER)}) & 0xFF) == 1);
    CHECK(Bytes(rig, Rig::STORAGE + 0x100, size) == pattern);
    CHECK(rig.sink.textures.back().pkg_data == 0);
    // Not past the texture.
    CHECK(Get<u8>(rig, Rig::STORAGE + 0x100 + size) == 0);
    // The driver's copy into the storage is a write of guest memory the replay is told about.
    REQUIRE(rig.sink.host_writes.size() == 1);
    CHECK(rig.sink.host_writes[0] == std::pair<u64, u64>{Rig::STORAGE + 0x100, size});
}

// Expected values are what the SDK code itself (3.2.0 sdk+0x5ed1d0 / +0x5ece10 / +0x5ed260 /
// +0x5ee8e0) answers for the same builder on the emulated GA10F device.
TEST_CASE("nvn hle texture: builder storage size and alignment are the SDK's", "[nvn]") {
    Rig rig;

    const auto size_of = [&](const TexSpec& spec) {
        Build(rig, BUILDER, spec);
        return rig.Call("nvnTextureBuilderGetStorageSize", {I(BUILDER)});
    };
    const auto align_of = [&](const TexSpec& spec) {
        Build(rig, BUILDER, spec);
        return rig.Call("nvnTextureBuilderGetStorageAlignment", {I(BUILDER)});
    };

    SECTION("block-linear chains, arrays, 3D, multisample, minimal layout") {
        CHECK(size_of({.width = 256, .height = 256}) == 0x40000);
        CHECK(size_of({.format = NVN_FORMAT_BPTC_UNORM, .width = 256, .height = 256}) == 65536);
        CHECK(size_of({.format = NVN_FORMAT_ASTC_4x4, .width = 64, .height = 64}) == 4096);
        CHECK(size_of({.target = NVN_TARGET_2D_ARRAY, .width = 64, .height = 64, .depth = 4}) ==
              65536);
        CHECK(size_of({.width = 1920, .height = 1080, .levels = 11}) == 0xBCD000);
        CHECK(size_of({.format = NVN_FORMAT_ASTC_8x8, .width = 300, .height = 200, .levels = 6}) ==
              0x7200);
        CHECK(size_of({.target = NVN_TARGET_2D_ARRAY, .format = NVN_FORMAT_ASTC_4x4, .width = 128,
                       .height = 128, .depth = 12, .levels = 8}) == 0x48000);
        CHECK(size_of({.target = NVN_TARGET_3D, .width = 64, .height = 64, .depth = 33,
                       .levels = 7}) == 0xD3000);
        CHECK(size_of({.target = 5, .samples = 4, .width = 1280, .height = 720}) == 0xF00000);
        CHECK(size_of({.flags = 0x200, .width = 320, .height = 320}) == 0x64000);
        CHECK(size_of({.target = 8, .width = 256, .height = 256, .levels = 9}) == 0x204000);
        CHECK(size_of({.target = 3, .width = 256, .height = 16}) == 0x20000);
        // Every target but 3D multiplies the layer by the depth field, 2D too.
        CHECK(size_of({.width = 64, .height = 64, .depth = 4}) == 0x10000);
        // A 0x0 texture still takes one GOB.
        CHECK(size_of({.width = 0, .height = 0}) == 0x200);
        CHECK(align_of({.width = 256, .height = 256}) == 0x200);
    }

    SECTION("sparse textures take the sparse tile shape but no 64 KiB padding or alignment") {
        CHECK(size_of({.format = NVN_FORMAT_BPTC_UNORM, .flags = FLAG_SPARSE, .width = 4096,
                       .height = 2048, .levels = 13}) == 0xAAB600);
        CHECK(size_of({.flags = FLAG_SPARSE, .width = 1024, .height = 1024, .levels = 11}) ==
              0x555C00);
        CHECK(align_of({.flags = FLAG_SPARSE, .width = 1024, .height = 1024}) == 0x200);
    }

    SECTION("linear textures are stride x height, buffers bytes x width") {
        CHECK(size_of({.flags = FLAG_LINEAR, .width = 100, .height = 10}) == 0);
        CHECK(size_of({.flags = FLAG_LINEAR, .width = 100, .height = 10, .stride = 512}) == 5120);
        CHECK(align_of({.flags = FLAG_LINEAR, .width = 100, .height = 10}) == 0x20);
        CHECK(align_of({.flags = 0x40, .width = 100, .height = 10, .stride = 512}) == 0x80);
        CHECK(size_of({.target = 10, .format = NVN_FORMAT_R8, .width = 100}) == 100);
        CHECK(align_of({.target = 10, .format = NVN_FORMAT_R8, .width = 100}) == 1);
        CHECK(align_of({.target = 10, .format = 0x22, .width = 100}) == 4); // 12-byte texels
        // The 1.0.0-generation sdk aligns 1-byte texture buffers on 0x200.
        rig.runtime.device.sdk_proc_names = 698;
        CHECK(align_of({.target = 10, .format = NVN_FORMAT_R8, .width = 100}) == 0x200);
        rig.runtime.device.sdk_proc_names = 0;
    }

    SECTION("a compressible or depth kind pads to and aligns on 4 KiB") {
        CHECK(size_of({.format = NVN_FORMAT_R8, .flags = 0x8, .width = 1, .height = 1}) == 0x1000);
        CHECK(align_of({.format = NVN_FORMAT_R8, .flags = 0x8, .width = 1, .height = 1}) == 0x1000);
        CHECK(align_of({.format = 0x35, .width = 1920, .height = 1080}) == 0x1000);
        // DEPTH32F keeps the generic kind unless it is compressible.
        CHECK(align_of({.format = 0x34, .width = 1920, .height = 1080}) == 0x200);
        CHECK(align_of({.format = 0x34, .flags = 0x8, .width = 1920, .height = 1080}) == 0x1000);
    }

    SECTION("formats: NVN2's BC additions have a layout, the removed RGB formats do not") {
        CHECK(size_of({.format = 0x9D, .width = 16, .height = 16}) == 0x200);
        CHECK(size_of({.format = 0x19, .width = 16, .height = 16}) == 0);
        CHECK(size_of({.format = 0x7FF, .width = 16, .height = 16}) == 0);
    }

    SECTION("a packaged layout's byte 1 is the tile width") {
        Build(rig, BUILDER, {.width = 100, .height = 256});
        rig.memory.Put(INPUT, std::array<u8, 8>{0x04, 0x00, 0, 0, 0, 0, 0, 0});
        rig.Call("nvnTextureBuilderSetPackagedTextureLayout", {I(BUILDER), I(INPUT)});
        CHECK(rig.Call("nvnTextureBuilderGetStorageSize", {I(BUILDER)}) == 0x1C000);
        rig.memory.Put(INPUT, std::array<u8, 8>{0x04, 0x02, 0, 0, 0, 0, 0, 0});
        rig.Call("nvnTextureBuilderSetPackagedTextureLayout", {I(BUILDER), I(INPUT)});
        CHECK(rig.Call("nvnTextureBuilderGetStorageSize", {I(BUILDER)}) == 0x20000);
    }

    SECTION("a GL texture name has no storage") {
        Build(rig, BUILDER, {.width = 256, .height = 256});
        rig.Call("nvnTextureBuilderSetGLTextureName", {I(BUILDER), I(5)});
        CHECK(rig.Call("nvnTextureBuilderGetStorageSize", {I(BUILDER)}) == 0);
    }
}

TEST_CASE("nvn hle texture: builder storage class is the SDK kind's", "[nvn]") {
    Rig rig;
    const auto class_of = [&](const TexSpec& spec) {
        Build(rig, BUILDER, spec);
        return static_cast<u32>(rig.Call("nvnTextureBuilderGetStorageClass", {I(BUILDER)}));
    };
    CHECK(class_of({.width = 256, .height = 256}) == 0x100);
    CHECK(class_of({.flags = 0x8, .width = 256, .height = 256}) == 0x108);
    // Block-compressed formats stay generic even when compressible.
    CHECK(class_of({.format = NVN_FORMAT_BPTC_UNORM, .flags = 0x8, .width = 256, .height = 256}) ==
          0x100);
    const std::array<std::array<u32, 3>, 6> depth{{{0x31, 0x102, 0x10A},
                                                    {0x32, 0x101, 0x10B},
                                                    {0x33, 0x103, 0x10C},
                                                    {0x34, 0x100, 0x109},
                                                    {0x35, 0x103, 0x10C},
                                                    {0x36, 0x104, 0x10D}}};
    for (const auto& [format, plain, compressible] : depth) {
        CHECK(class_of({.format = static_cast<s32>(format), .width = 64, .height = 64}) == plain);
        CHECK(class_of({.format = static_cast<s32>(format), .flags = 0x8, .width = 64,
                        .height = 64}) == compressible);
    }
    // Linear and buffer storage keep the default kind.
    CHECK(class_of({.format = 0x35, .flags = FLAG_LINEAR, .width = 64, .height = 64,
                    .stride = 256}) == 0x100);
}

TEST_CASE("nvn hle texture: ZCull storage size is the SDK's", "[nvn]") {
    Rig rig;
    const auto zcull_of = [&](const TexSpec& spec) {
        Build(rig, BUILDER, spec);
        return rig.Call("nvnTextureBuilderGetZCullStorageSize", {I(BUILDER)});
    };
    // No depth: no zcull.
    CHECK(zcull_of({.width = 1920, .height = 1080}) == 0);
    CHECK(zcull_of({.format = 0x31, .width = 1920, .height = 1080}) == 0);
    // 0x20 + 0x20 x ceil(1920 x 1088 / 0x400) + 0xc0.
    CHECK(zcull_of({.format = 0x33, .width = 1920, .height = 1080}) == 0xFFE0);
    CHECK(zcull_of({.format = 0x35, .flags = 0x400, .width = 1920, .height = 1080}) == 0xFFE0);
    // ADAPTIVE_ZCULL: subregions out of the 0x800 aliquots.
    CHECK(zcull_of({.format = 0x33, .flags = 0x80, .width = 2560, .height = 1440}) == 0xF540);
    CHECK(zcull_of({.target = NVN_TARGET_2D_ARRAY, .format = 0x35, .flags = 0x480, .width = 1280,
                    .height = 720, .depth = 2}) == 0x1FEE0);
}

TEST_CASE("nvn hle texture: TextureInitialize refuses what the SDK refuses", "[nvn]") {
    Rig rig;
    constexpr u64 COMPRESSIBLE_POOL = Rig::Obj(13);
    REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, 0x80000, 0x24));
    REQUIRE(rig.recorder.MemoryPoolInitialize(COMPRESSIBLE_POOL, Rig::STORAGE + 0x80000, 0x80000,
                                              0xA4));
    const auto init = [&](const TexSpec& spec, u64 pool, u64 offset) {
        Build(rig, BUILDER, spec);
        rig.Call("nvnTextureBuilderSetStorage", {I(BUILDER), I(pool), I(offset)});
        return rig.Call("nvnTextureInitialize", {I(TEXTURE), I(BUILDER)}) & 0xFF;
    };
    CHECK(init({.width = 64, .height = 64}, POOL, 0x200) == 1);
    CHECK(init({.format = 0x19, .width = 64, .height = 64}, POOL, 0) == 0);
    // The SDK clears the object before it checks: the refused re-initialization leaves no texture.
    CHECK(static_cast<u32>(rig.Call("nvnTextureGetWidth", {I(TEXTURE)})) == 0);
    CHECK(init({.flags = 0x8, .width = 64, .height = 64}, POOL, 0) == 0);
    CHECK(init({.flags = 0x8, .width = 64, .height = 64}, COMPRESSIBLE_POOL, 0) == 1);
    // Depth kinds need a compressible pool and a 4 KiB offset; DEPTH32F is generic.
    CHECK(init({.format = 0x35, .width = 64, .height = 64}, POOL, 0) == 0);
    CHECK(init({.format = 0x35, .width = 64, .height = 64}, COMPRESSIBLE_POOL, 0x200) == 0);
    CHECK(init({.format = 0x35, .width = 64, .height = 64}, COMPRESSIBLE_POOL, 0x1000) == 1);
    CHECK(init({.format = 0x34, .width = 64, .height = 64}, POOL, 0x200) == 1);
}

TEST_CASE("nvn hle texture: GetFlags sign-extends the SDK's 16 flag bits", "[nvn]") {
    Rig rig;
    MakePool(rig);
    REQUIRE(MakeTexture(rig, TEXTURE, BUILDER, {.flags = 0x18004, .width = 8, .height = 8}, 0) ==
            1);
    CHECK(static_cast<u32>(rig.Call("nvnTextureGetFlags", {I(TEXTURE)})) == 0xFFFF8004);
}

TEST_CASE("nvn hle texture: sparse tile layout", "[nvn]") {
    Rig rig;
    const auto layout_at = [&](u64 va) {
        return Get<std::array<s32, 4>>(rig, va);
    };

    SECTION("RGBA8 1024x1024 with 11 levels: 128x128 texel tiles, 4 tiled levels") {
        Build(rig, BUILDER, {.flags = FLAG_SPARSE, .width = 1024, .height = 1024, .levels = 11});
        Fill(rig, OUT, 32, 0xFF);
        rig.Call("nvnTextureBuilderGetSparseTileLayout", {I(BUILDER), I(OUT)});
        CHECK(layout_at(OUT) == std::array<s32, 4>{4, 128, 128, 1});
        // The 16 reserved bytes are zeroed.
        CHECK(Bytes(rig, OUT + 16, 16) == std::vector<u8>(16, 0));
    }

    SECTION("a texture that is not sparse has no tiles") {
        Build(rig, BUILDER, {.width = 1024, .height = 1024, .levels = 11});
        Fill(rig, OUT, 32, 0xFF);
        rig.Call("nvnTextureBuilderGetSparseTileLayout", {I(BUILDER), I(OUT)});
        CHECK(layout_at(OUT) == std::array<s32, 4>{0, 0, 0, 0});
    }

    SECTION("an initialized texture answers the same") {
        MakePool(rig);
        REQUIRE(MakeTexture(rig, TEXTURE, BUILDER,
                            {.flags = FLAG_SPARSE, .width = 1024, .height = 1024, .levels = 11},
                            0) == 1);
        rig.Call("nvnTextureGetSparseTileLayout", {I(TEXTURE), I(OUT)});
        CHECK(layout_at(OUT) == std::array<s32, 4>{4, 128, 128, 1});
    }
}

TEST_CASE("nvn hle texture: view setters write the nxnative view_t bytes", "[nvn]") {
    Rig rig;
    Fill(rig, VIEW, 16, 0xFF);
    rig.Call("nvnTextureViewSetDefaults", {I(VIEW)});
    CHECK(Bytes(rig, VIEW, 16) == std::vector<u8>(16, 0));

    rig.Call("nvnTextureViewSetLevels", {I(VIEW), I(1), I(3)});
    rig.Call("nvnTextureViewSetLayers", {I(VIEW), I(2), I(5)});
    rig.Call("nvnTextureViewSetFormat", {I(VIEW), I(NVN_FORMAT_RGBA8)});
    rig.Call("nvnTextureViewSetTarget", {I(VIEW), I(NVN_TARGET_2D_ARRAY)});
    rig.Call("nvnTextureViewSetSwizzle", {I(VIEW), I(2), I(3), I(4), I(5)});
    {
        std::vector<u8> expected(16, 0);
        Put<u8>(expected, 0, 0x1F);       // levels, layers, format, target, swizzle
        Put<u8>(expected, 1, 4);          // target
        Put<u8>(expected, 2, 1);          // base_level
        Put<u8>(expected, 3, 3);          // nlevels
        Put<u16>(expected, 4, 0x25);      // format
        Put<u16>(expected, 6, 2);         // min_layer
        Put<u16>(expected, 8, 5);         // nlayers
        Put<u8>(expected, 10, 2);
        Put<u8>(expected, 11, 3);
        Put<u8>(expected, 12, 4);
        Put<u8>(expected, 13, 5);
        CHECK(Bytes(rig, VIEW, 16) == expected);
    }
    rig.Call("nvnTextureViewSetDepthStencilMode", {I(VIEW), I(1)});
    // set_dsmode (bit 5) and dsmode = 1 in bits 6..7.
    CHECK(Get<u8>(rig, VIEW) == 0x7F);
    rig.Call("nvnTextureViewSetDepthStencilMode", {I(VIEW), I(0)});
    CHECK(Get<u8>(rig, VIEW) == 0x3F);

    SECTION("getters return the set flag and the values") {
        CHECK((rig.Call("nvnTextureViewGetLevels", {I(VIEW), I(OUT), I(OUT + 4)}) & 0xFF) == 1);
        CHECK(Get<std::array<s32, 2>>(rig, OUT) == std::array<s32, 2>{1, 3});
        CHECK((rig.Call("nvnTextureViewGetLayers", {I(VIEW), I(OUT), I(OUT + 4)}) & 0xFF) == 1);
        CHECK(Get<std::array<s32, 2>>(rig, OUT) == std::array<s32, 2>{2, 5});
        CHECK((rig.Call("nvnTextureViewGetFormat", {I(VIEW), I(OUT)}) & 0xFF) == 1);
        CHECK(Get<s32>(rig, OUT) == NVN_FORMAT_RGBA8);
        CHECK((rig.Call("nvnTextureViewGetTarget", {I(VIEW), I(OUT)}) & 0xFF) == 1);
        CHECK(Get<s32>(rig, OUT) == NVN_TARGET_2D_ARRAY);
        CHECK((rig.Call("nvnTextureViewGetDepthStencilMode", {I(VIEW), I(OUT)}) & 0xFF) == 1);
        CHECK(Get<s32>(rig, OUT) == 0);
        CHECK((rig.Call("nvnTextureViewGetSwizzle",
                        {I(VIEW), I(OUT), I(OUT + 4), I(OUT + 8), I(OUT + 12)}) &
               0xFF) == 1);
        CHECK(Get<std::array<s32, 4>>(rig, OUT) == std::array<s32, 4>{2, 3, 4, 5});
    }

    SECTION("an untouched view reports nothing set and the identity swizzle") {
        rig.Call("nvnTextureViewSetDefaults", {I(VIEW2)});
        CHECK((rig.Call("nvnTextureViewGetLevels", {I(VIEW2), I(OUT), I(OUT + 4)}) & 0xFF) == 0);
        CHECK((rig.Call("nvnTextureViewGetLayers", {I(VIEW2), I(OUT), I(OUT + 4)}) & 0xFF) == 0);
        CHECK((rig.Call("nvnTextureViewGetFormat", {I(VIEW2), I(OUT)}) & 0xFF) == 0);
        CHECK((rig.Call("nvnTextureViewGetTarget", {I(VIEW2), I(OUT)}) & 0xFF) == 0);
        CHECK((rig.Call("nvnTextureViewGetDepthStencilMode", {I(VIEW2), I(OUT)}) & 0xFF) == 0);
        CHECK((rig.Call("nvnTextureViewGetSwizzle",
                        {I(VIEW2), I(OUT), I(OUT + 4), I(OUT + 8), I(OUT + 12)}) &
               0xFF) == 0);
        CHECK(Get<std::array<s32, 4>>(rig, OUT) == std::array<s32, 4>{2, 3, 4, 5});
    }

    SECTION("Compare ignores values of parameters that are not set") {
        // Two default views, one holding a base level whose flag is not set: equal.
        constexpr u64 VIEW3 = Rig::Obj(20);
        rig.Call("nvnTextureViewSetDefaults", {I(VIEW2)});
        rig.Call("nvnTextureViewSetDefaults", {I(VIEW3)});
        auto unset = rig.memory.Get<VideoCommon::Nvn::GuestTextureView>(VIEW3);
        unset.base_level = 4;
        rig.memory.Put(VIEW3, unset);
        CHECK((rig.Call("nvnTextureViewCompare", {I(VIEW2), I(VIEW3)}) & 0xFF) == 1);

        rig.Call("nvnTextureViewSetDefaults", {I(VIEW2)});
        CHECK((rig.Call("nvnTextureViewCompare", {I(VIEW), I(VIEW2)}) & 0xFF) == 0);
        rig.Call("nvnTextureViewSetLevels", {I(VIEW2), I(1), I(3)});
        rig.Call("nvnTextureViewSetLayers", {I(VIEW2), I(2), I(5)});
        rig.Call("nvnTextureViewSetFormat", {I(VIEW2), I(NVN_FORMAT_RGBA8)});
        rig.Call("nvnTextureViewSetTarget", {I(VIEW2), I(NVN_TARGET_2D_ARRAY)});
        rig.Call("nvnTextureViewSetSwizzle", {I(VIEW2), I(2), I(3), I(4), I(5)});
        rig.Call("nvnTextureViewSetDepthStencilMode", {I(VIEW2), I(0)});
        CHECK((rig.Call("nvnTextureViewCompare", {I(VIEW), I(VIEW2)}) & 0xFF) == 1);
        rig.Call("nvnTextureViewSetLayers", {I(VIEW2), I(2), I(6)});
        CHECK((rig.Call("nvnTextureViewCompare", {I(VIEW), I(VIEW2)}) & 0xFF) == 0);
    }
}

TEST_CASE("nvn hle texture: GetViewOffset adds the level offset and whole layers", "[nvn]") {
    Rig rig;
    MakePool(rig);
    const TexSpec spec{.target = NVN_TARGET_2D_ARRAY, .width = 64, .height = 64, .depth = 4,
                       .levels = 3};
    REQUIRE(MakeTexture(rig, TEXTURE, BUILDER, spec, 0) == 1);
    const auto layout = Vulkan::Nvn::ComputeTextureLayout(LayoutOf(spec));
    const u64 level1 = Vulkan::Nvn::TextureLevelOffset(LayoutOf(spec), 1);
    REQUIRE(level1 != 0);
    REQUIRE(layout.layer_size != 0);

    CHECK(rig.Call("nvnTextureGetViewOffset", {I(TEXTURE), I(0)}) == 0);
    rig.Call("nvnTextureViewSetDefaults", {I(VIEW)});
    CHECK(rig.Call("nvnTextureGetViewOffset", {I(TEXTURE), I(VIEW)}) == 0);
    rig.Call("nvnTextureViewSetLevels", {I(VIEW), I(1), I(1)});
    CHECK(rig.Call("nvnTextureGetViewOffset", {I(TEXTURE), I(VIEW)}) == level1);
    rig.Call("nvnTextureViewSetLayers", {I(VIEW), I(2), I(1)});
    CHECK(rig.Call("nvnTextureGetViewOffset", {I(TEXTURE), I(VIEW)}) ==
          level1 + 2 * layout.layer_size);
    // The builder form answers the same for the same description.
    CHECK(rig.Call("nvnTextureBuilderGetViewOffset", {I(BUILDER), I(VIEW)}) ==
          level1 + 2 * layout.layer_size);
}

TEST_CASE("nvn hle texture: WriteTexels and ReadTexels on a linear texture", "[nvn]") {
    Rig rig;
    MakePool(rig);
    // 8x4 RGBA8, linear: stride (32 + 31) & ~31 = 32.
    REQUIRE(MakeTexture(rig, TEXTURE, BUILDER,
                        {.flags = FLAG_LINEAR, .width = 8, .height = 4}, 0x1000) == 1);
    const u64 storage = Rig::STORAGE + 0x1000;
    std::vector<u8> source(24);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<u8>(i + 1);
    }
    REQUIRE(rig.memory.Write(DATA, source.data(), source.size()));
    rig.memory.Put(REGION, std::array<s32, 6>{2, 1, 0, 3, 2, 1}); // x y z w h d

    SECTION("packed source rows land at the right storage rows") {
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        std::vector<u8> expected(128, 0);
        std::memcpy(expected.data() + 32 + 8, source.data(), 12);
        std::memcpy(expected.data() + 64 + 8, source.data() + 12, 12);
        CHECK(Bytes(rig, storage, 128) == expected);

        // Read back the same region into another buffer, packed.
        rig.Call("nvnTextureReadTexels", {I(TEXTURE), I(0), I(REGION), I(DATA2)});
        CHECK(Bytes(rig, DATA2, 24) == source);
    }

    SECTION("strided variants use the caller's row pitch") {
        std::vector<u8> padded(2 * 16, 0xEE);
        std::memcpy(padded.data(), source.data(), 12);
        std::memcpy(padded.data() + 16, source.data() + 12, 12);
        REQUIRE(rig.memory.Write(DATA, padded.data(), padded.size()));
        rig.Call("nvnTextureWriteTexelsStrided",
                 {I(TEXTURE), I(0), I(REGION), I(DATA), I(16), I(0)});
        std::vector<u8> expected(128, 0);
        std::memcpy(expected.data() + 32 + 8, source.data(), 12);
        std::memcpy(expected.data() + 64 + 8, source.data() + 12, 12);
        CHECK(Bytes(rig, storage, 128) == expected);

        Fill(rig, DATA2, 32, 0);
        rig.Call("nvnTextureReadTexelsStrided", {I(TEXTURE), I(0), I(REGION), I(DATA2), I(16), I(0)});
        std::vector<u8> read_back(32, 0);
        std::memcpy(read_back.data(), source.data(), 12);
        std::memcpy(read_back.data() + 16, source.data() + 12, 12);
        CHECK(Bytes(rig, DATA2, 32) == read_back);
    }

    SECTION("a region outside the texture writes nothing") {
        rig.memory.Put(REGION, std::array<s32, 6>{6, 0, 0, 3, 2, 1});
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        CHECK(Bytes(rig, storage, 128) == std::vector<u8>(128, 0));
    }

    SECTION("flush and invalidate are accepted") {
        rig.Call("nvnTextureFlushTexels", {I(TEXTURE), I(0), I(REGION)});
        rig.Call("nvnTextureInvalidateTexels", {I(TEXTURE), I(0), I(REGION)});
        CHECK(Bytes(rig, storage, 128) == std::vector<u8>(128, 0));
    }
}

TEST_CASE("nvn hle texture: texel access tells the replay which guest bytes it wrote", "[nvn]") {
    // The HLE copies texels through its own accessor, which the guest page tracker never sees:
    // a pool mirror only learns of them through the notification.
    Rig rig;
    MakePool(rig);
    std::vector<u8> source(24);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<u8>(i + 1);
    }
    REQUIRE(rig.memory.Write(DATA, source.data(), source.size()));

    SECTION("a linear write announces the storage rows, once") {
        // 8x4 RGBA8, linear: stride 32.
        REQUIRE(MakeTexture(rig, TEXTURE, BUILDER,
                            {.flags = FLAG_LINEAR, .width = 8, .height = 4}, 0x1000) == 1);
        const u64 storage = Rig::STORAGE + 0x1000;
        rig.memory.Put(REGION, std::array<s32, 6>{2, 1, 0, 3, 2, 1}); // x y z w h d
        rig.sink.host_writes.clear();
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        // Row 1 from x = 2 to row 2 up to x = 5: [32 + 8, 64 + 8 + 12).
        REQUIRE(rig.sink.host_writes.size() == 1);
        CHECK(rig.sink.host_writes[0] == std::pair<u64, u64>{storage + 40, 44});
    }

    SECTION("a linear read announces the caller's rows, once") {
        REQUIRE(MakeTexture(rig, TEXTURE, BUILDER,
                            {.flags = FLAG_LINEAR, .width = 8, .height = 4}, 0x1000) == 1);
        rig.memory.Put(REGION, std::array<s32, 6>{2, 1, 0, 3, 2, 1});
        rig.sink.host_writes.clear();
        rig.Call("nvnTextureReadTexels", {I(TEXTURE), I(0), I(REGION), I(DATA2)});
        // Two packed rows of 12 bytes.
        REQUIRE(rig.sink.host_writes.size() == 1);
        CHECK(rig.sink.host_writes[0] == std::pair<u64, u64>{DATA2, 24});
    }

    SECTION("a block-linear write announces the level's storage, a read the caller's rows") {
        constexpr u32 SIZE = 32;
        const TexSpec spec{.width = SIZE, .height = SIZE, .levels = 1};
        REQUIRE(MakeTexture(rig, TEXTURE, BUILDER, spec, 0x2000) == 1);
        const u64 storage = Rig::STORAGE + 0x2000;
        std::vector<u8> texels(SIZE * SIZE * 4, 0x5A);
        REQUIRE(rig.memory.Write(DATA, texels.data(), texels.size()));
        rig.memory.Put(REGION, std::array<s32, 6>{0, 0, 0, SIZE, SIZE, 1});
        rig.sink.host_writes.clear();
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        REQUIRE(rig.sink.host_writes.size() == 1);
        CHECK(rig.sink.host_writes[0].first == storage);
        CHECK(rig.sink.host_writes[0].second >= texels.size());

        rig.sink.host_writes.clear();
        rig.Call("nvnTextureReadTexels", {I(TEXTURE), I(0), I(REGION), I(DATA2)});
        REQUIRE(rig.sink.host_writes.size() == 1);
        CHECK(rig.sink.host_writes[0] == std::pair<u64, u64>{DATA2, texels.size()});
    }

    SECTION("a region outside the texture writes nothing and announces nothing") {
        REQUIRE(MakeTexture(rig, TEXTURE, BUILDER,
                            {.flags = FLAG_LINEAR, .width = 8, .height = 4}, 0x1000) == 1);
        rig.memory.Put(REGION, std::array<s32, 6>{6, 0, 0, 3, 2, 1});
        rig.sink.host_writes.clear();
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        CHECK(rig.sink.host_writes.empty());
    }
}

TEST_CASE("nvn hle texture: WriteTexels and ReadTexels on a block-linear texture", "[nvn]") {
    Rig rig;
    MakePool(rig);
    constexpr u32 SIZE = 32;
    const TexSpec spec{.width = SIZE, .height = SIZE, .levels = 1};
    REQUIRE(MakeTexture(rig, TEXTURE, BUILDER, spec, 0x2000) == 1);
    const u64 storage = Rig::STORAGE + 0x2000;
    const auto layout_desc = LayoutOf(spec);
    const auto layout = Vulkan::Nvn::ComputeTextureLayout(layout_desc);
    u32 tile_h = 0;
    u32 tile_d = 0;
    Vulkan::Nvn::TextureLevelTiles(layout_desc, 0, tile_h, tile_d);

    std::vector<u8> texels(SIZE * SIZE * 4);
    for (size_t i = 0; i < texels.size(); ++i) {
        texels[i] = static_cast<u8>(i * 7 + (i >> 5));
    }
    REQUIRE(rig.memory.Write(DATA, texels.data(), texels.size() - 0) == true);
    rig.memory.Put(REGION, std::array<s32, 6>{0, 0, 0, SIZE, SIZE, 1});

    SECTION("a whole-level write is the Switch 2 GOB order of the texels") {
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        std::vector<u8> expected(layout.size, 0);
        Tegra::Texture::SwizzleTexture(expected, texels, 4, SIZE, SIZE, 1, tile_h, tile_d, 0, true);
        CHECK(Bytes(rig, storage, expected.size()) == expected);
        // The replay is told which texture changed.
        REQUIRE(rig.sink.texture_writes.size() == 1);
        CHECK(rig.sink.texture_writes[0] == rig.sink.textures[0].serial);

        // The write flushed the GPU's texels into the storage first (a partial write must not
        // erase them); the read flushes again.
        REQUIRE(rig.sink.flushed.size() == 1);
        Fill(rig, DATA2, texels.size(), 0);
        rig.Call("nvnTextureReadTexels", {I(TEXTURE), I(0), I(REGION), I(DATA2)});
        CHECK(Bytes(rig, DATA2, texels.size()) == texels);
        REQUIRE(rig.sink.flushed.size() == 2);
        CHECK(rig.sink.flushed[1] == rig.sink.textures[0].serial);
    }

    SECTION("a sub-rectangle leaves the other texels alone") {
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});
        std::vector<u8> patch(8 * 4 * 4, 0xEE);
        REQUIRE(rig.memory.Write(DATA2, patch.data(), patch.size()));
        rig.memory.Put(REGION, std::array<s32, 6>{4, 6, 0, 8, 4, 1});
        rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA2)});

        std::vector<u8> linear = texels;
        for (u32 y = 6; y < 10; ++y) {
            for (u32 x = 4; x < 12; ++x) {
                std::memset(linear.data() + (y * SIZE + x) * 4, 0xEE, 4);
            }
        }
        std::vector<u8> expected(layout.size, 0);
        Tegra::Texture::SwizzleTexture(expected, linear, 4, SIZE, SIZE, 1, tile_h, tile_d, 0, true);
        CHECK(Bytes(rig, storage, expected.size()) == expected);

        // And reads the same sub-rectangle back with a caller pitch.
        Fill(rig, DATA, 8 * 4 * 64, 0);
        rig.Call("nvnTextureReadTexelsStrided",
                 {I(TEXTURE), I(0), I(REGION), I(DATA), I(64), I(0)});
        for (u32 row = 0; row < 4; ++row) {
            CHECK(Bytes(rig, DATA + row * 64, 32) == std::vector<u8>(32, 0xEE));
        }
    }
}

TEST_CASE("nvn hle texture: WriteTexels on a 3D texture patches whole slices", "[nvn]") {
    Rig rig;
    MakePool(rig);
    constexpr u32 SIZE = 16;
    const TexSpec spec{.target = NVN_TARGET_3D, .width = SIZE, .height = SIZE, .depth = SIZE};
    REQUIRE(MakeTexture(rig, TEXTURE, BUILDER, spec, 0) == 1);
    const auto layout_desc = LayoutOf(spec);
    const auto layout = Vulkan::Nvn::ComputeTextureLayout(layout_desc);
    u32 tile_h = 0;
    u32 tile_d = 0;
    Vulkan::Nvn::TextureLevelTiles(layout_desc, 0, tile_h, tile_d);

    // Two slices (z = 3 and 4) of the top-left 8x8.
    std::vector<u8> data(2 * 8 * 8 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<u8>(i * 13 + 5);
    }
    REQUIRE(rig.memory.Write(DATA, data.data(), data.size()));
    rig.memory.Put(REGION, std::array<s32, 6>{0, 0, 3, 8, 8, 2});
    rig.Call("nvnTextureWriteTexels", {I(TEXTURE), I(0), I(REGION), I(DATA)});

    std::vector<u8> linear(SIZE * SIZE * SIZE * 4, 0);
    for (u32 slice = 0; slice < 2; ++slice) {
        for (u32 y = 0; y < 8; ++y) {
            std::memcpy(linear.data() + (((3 + slice) * SIZE + y) * SIZE) * 4,
                        data.data() + (slice * 64 + y * 8) * 4, 8 * 4);
        }
    }
    std::vector<u8> expected(layout.size, 0);
    Tegra::Texture::SwizzleTexture(expected, linear, 4, SIZE, SIZE, SIZE, tile_h, tile_d, 0, true);
    CHECK(Bytes(rig, Rig::STORAGE, expected.size()) == expected);

    Fill(rig, DATA2, data.size(), 0);
    rig.Call("nvnTextureReadTexels", {I(TEXTURE), I(0), I(REGION), I(DATA2)});
    CHECK(Bytes(rig, DATA2, data.size()) == data);
}

TEST_CASE("nvn hle sampler: builder defaults write the nxnative layout verbatim", "[nvn]") {
    Rig rig;
    Fill(rig, SAMPLER_BUILDER, 0x60, 0xCC);
    rig.Call("nvnSamplerBuilderSetDevice", {I(SAMPLER_BUILDER), I(0xA1B2C3D4E5F60718ULL)});
    rig.Call("nvnSamplerBuilderSetDefaults", {I(SAMPLER_BUILDER)});

    std::vector<u8> expected(0x60, 0xCC);
    std::fill(expected.begin(), expected.begin() + 0x30, 0);
    Put<u64>(expected, 0x00, 0xA1B2C3D4E5F60718ULL);
    Put<u8>(expected, 0x08, 1);  // min: LINEAR (SDK sdk+0x5ea460)
    Put<u8>(expected, 0x09, 1);  // mag: LINEAR
    Put<u8>(expected, 0x0A, 1);  // wrap s/t/r: REPEAT
    Put<u8>(expected, 0x0B, 1);
    Put<u8>(expected, 0x0C, 1);
    Put<u8>(expected, 0x0D, 0);  // compare mode NONE
    Put<u8>(expected, 0x0E, 2);  // compare func LESS
    Put<u8>(expected, 0x0F, 0);  // reduction AVERAGE
    Put<f32>(expected, 0x10, 0.0f);
    Put<f32>(expected, 0x14, 1000.0f);
    Put<f32>(expected, 0x18, 0.0f);
    Put<f32>(expected, 0x1C, 1.0f);
    REQUIRE(Bytes(rig, SAMPLER_BUILDER, 0x60) == expected);
}

TEST_CASE("nvn hle sampler: setters mix floats and integers and land at their offsets", "[nvn]") {
    Rig rig;
    rig.Call("nvnSamplerBuilderSetDefaults", {I(SAMPLER_BUILDER)});
    rig.Call("nvnSamplerBuilderSetMinMagFilter", {I(SAMPLER_BUILDER), I(5), I(0)});
    rig.Call("nvnSamplerBuilderSetWrapMode", {I(SAMPLER_BUILDER), I(5), I(6), I(7)});
    rig.Call("nvnSamplerBuilderSetLodClamp", {I(SAMPLER_BUILDER), Rig::F(-2.5f), Rig::F(7.0f)});
    rig.Call("nvnSamplerBuilderSetLodBias", {I(SAMPLER_BUILDER), Rig::F(0.75f)});
    rig.Call("nvnSamplerBuilderSetCompare", {I(SAMPLER_BUILDER), I(1), I(3)});
    rig.memory.Put(INPUT, std::array<f32, 4>{0.1f, 0.2f, 0.3f, 0.4f});
    rig.Call("nvnSamplerBuilderSetBorderColor", {I(SAMPLER_BUILDER), I(INPUT)});
    rig.Call("nvnSamplerBuilderSetMaxAnisotropy", {I(SAMPLER_BUILDER), Rig::F(8.0f)});
    rig.Call("nvnSamplerBuilderSetReductionFilter", {I(SAMPLER_BUILDER), I(2)});

    std::vector<u8> expected(0x30, 0);
    Put<u8>(expected, 0x08, 5);
    Put<u8>(expected, 0x09, 0);
    Put<u8>(expected, 0x0A, 5);
    Put<u8>(expected, 0x0B, 6);
    Put<u8>(expected, 0x0C, 7);
    Put<u8>(expected, 0x0D, 1);
    Put<u8>(expected, 0x0E, 3);
    Put<u8>(expected, 0x0F, 2);
    Put<f32>(expected, 0x10, -2.5f);
    Put<f32>(expected, 0x14, 7.0f);
    Put<f32>(expected, 0x18, 0.75f);
    Put<f32>(expected, 0x1C, 8.0f);
    Put<f32>(expected, 0x20, 0.1f);
    Put<f32>(expected, 0x24, 0.2f);
    Put<f32>(expected, 0x28, 0.3f);
    Put<f32>(expected, 0x2C, 0.4f);
    REQUIRE(Bytes(rig, SAMPLER_BUILDER, 0x30) == expected);

    SECTION("builder getters") {
        rig.Call("nvnSamplerBuilderGetMinMagFilter", {I(SAMPLER_BUILDER), I(OUT), I(OUT + 4)});
        CHECK(Get<std::array<s32, 2>>(rig, OUT) == std::array<s32, 2>{5, 0});
        rig.Call("nvnSamplerBuilderGetWrapMode",
                 {I(SAMPLER_BUILDER), I(OUT), I(OUT + 4), I(OUT + 8)});
        CHECK(Get<std::array<s32, 3>>(rig, OUT) == std::array<s32, 3>{5, 6, 7});
        rig.Call("nvnSamplerBuilderGetLodClamp", {I(SAMPLER_BUILDER), I(OUT), I(OUT + 4)});
        CHECK(Get<std::array<f32, 2>>(rig, OUT) == std::array<f32, 2>{-2.5f, 7.0f});
        CHECK(rig.CallF("nvnSamplerBuilderGetLodBias", {I(SAMPLER_BUILDER)}) == 0.75f);
        rig.Call("nvnSamplerBuilderGetCompare", {I(SAMPLER_BUILDER), I(OUT), I(OUT + 4)});
        CHECK(Get<std::array<s32, 2>>(rig, OUT) == std::array<s32, 2>{1, 3});
        rig.Call("nvnSamplerBuilderGetBorderColor", {I(SAMPLER_BUILDER), I(OUT)});
        CHECK(Get<std::array<f32, 4>>(rig, OUT) == std::array<f32, 4>{0.1f, 0.2f, 0.3f, 0.4f});
        CHECK(rig.CallF("nvnSamplerBuilderGetMaxAnisotropy", {I(SAMPLER_BUILDER)}) == 8.0f);
        CHECK(static_cast<u32>(rig.Call("nvnSamplerBuilderGetReductionFilter", {I(SAMPLER_BUILDER)})) == 2);
    }

    SECTION("integer border colors are converted by value, not bit-cast") {
        rig.memory.Put(INPUT, std::array<s32, 4>{1, -2, 3, 0});
        rig.Call("nvnSamplerBuilderSetBorderColori", {I(SAMPLER_BUILDER), I(INPUT)});
        CHECK(Get<std::array<f32, 4>>(rig, SAMPLER_BUILDER + 0x20) ==
              std::array<f32, 4>{1.0f, -2.0f, 3.0f, 0.0f});
        rig.Call("nvnSamplerBuilderGetBorderColori", {I(SAMPLER_BUILDER), I(OUT)});
        CHECK(Get<std::array<s32, 4>>(rig, OUT) == std::array<s32, 4>{1, -2, 3, 0});

        rig.memory.Put(INPUT, std::array<u32, 4>{5, 6, 7, 1});
        rig.Call("nvnSamplerBuilderSetBorderColorui", {I(SAMPLER_BUILDER), I(INPUT)});
        CHECK(Get<std::array<f32, 4>>(rig, SAMPLER_BUILDER + 0x20) ==
              std::array<f32, 4>{5.0f, 6.0f, 7.0f, 1.0f});
        rig.Call("nvnSamplerBuilderGetBorderColorui", {I(SAMPLER_BUILDER), I(OUT)});
        CHECK(Get<std::array<u32, 4>>(rig, OUT) == std::array<u32, 4>{5, 6, 7, 1});
    }
}

TEST_CASE("nvn hle sampler: SamplerInitialize produces the expected SamplerDesc", "[nvn]") {
    Rig rig;
    rig.Call("nvnSamplerBuilderSetDefaults", {I(SAMPLER_BUILDER)});
    rig.Call("nvnSamplerBuilderSetMinMagFilter", {I(SAMPLER_BUILDER), I(3), I(1)});
    rig.Call("nvnSamplerBuilderSetWrapMode", {I(SAMPLER_BUILDER), I(7), I(6), I(5)});
    rig.Call("nvnSamplerBuilderSetLodClamp", {I(SAMPLER_BUILDER), Rig::F(-1.5f), Rig::F(12.0f)});
    rig.Call("nvnSamplerBuilderSetLodBias", {I(SAMPLER_BUILDER), Rig::F(-0.25f)});
    rig.Call("nvnSamplerBuilderSetCompare", {I(SAMPLER_BUILDER), I(1), I(4)});
    rig.memory.Put(INPUT, std::array<f32, 4>{1.0f, 0.5f, 0.25f, 0.125f});
    rig.Call("nvnSamplerBuilderSetBorderColor", {I(SAMPLER_BUILDER), I(INPUT)});
    rig.Call("nvnSamplerBuilderSetMaxAnisotropy", {I(SAMPLER_BUILDER), Rig::F(16.0f)});
    rig.Call("nvnSamplerBuilderSetReductionFilter", {I(SAMPLER_BUILDER), I(1)});

    REQUIRE((rig.Call("nvnSamplerInitialize", {I(SAMPLER), I(SAMPLER_BUILDER)}) & 0xFF) == 1);
    REQUIRE(rig.sink.samplers.size() == 1);
    const auto& desc = rig.sink.samplers.back();
    CHECK(desc.min_filter == 3);
    CHECK(desc.mag_filter == 1);
    CHECK(desc.wrap_s == 7);
    CHECK(desc.wrap_t == 6);
    CHECK(desc.wrap_r == 5);
    CHECK(desc.compare_mode == 1);
    CHECK(desc.compare_func == 4);
    CHECK(desc.reduction == 1);
    CHECK(desc.lod_min == -1.5f);
    CHECK(desc.lod_max == 12.0f);
    CHECK(desc.lod_bias == -0.25f);
    CHECK(desc.max_anisotropy == 16.0f);
    CHECK(desc.border[0] == 1.0f);
    CHECK(desc.border[1] == 0.5f);
    CHECK(desc.border[2] == 0.25f);
    CHECK(desc.border[3] == 0.125f);
    CHECK(desc.serial != 0);

    const auto tag = Get<VideoCommon::Nvn::GuestTag>(rig, SAMPLER);
    CHECK(tag.magic == VideoCommon::Nvn::GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(VideoCommon::Nvn::ObjectKind::Sampler));
    CHECK(tag.serial == desc.serial);

    SECTION("object getters") {
        rig.Call("nvnSamplerGetMinMagFilter", {I(SAMPLER), I(OUT), I(OUT + 4)});
        CHECK(Get<std::array<s32, 2>>(rig, OUT) == std::array<s32, 2>{3, 1});
        rig.Call("nvnSamplerGetWrapMode", {I(SAMPLER), I(OUT), I(OUT + 4), I(OUT + 8)});
        CHECK(Get<std::array<s32, 3>>(rig, OUT) == std::array<s32, 3>{7, 6, 5});
        rig.Call("nvnSamplerGetLodClamp", {I(SAMPLER), I(OUT), I(OUT + 4)});
        CHECK(Get<std::array<f32, 2>>(rig, OUT) == std::array<f32, 2>{-1.5f, 12.0f});
        CHECK(rig.CallF("nvnSamplerGetLodBias", {I(SAMPLER)}) == -0.25f);
        rig.Call("nvnSamplerGetCompare", {I(SAMPLER), I(OUT), I(OUT + 4)});
        CHECK(Get<std::array<s32, 2>>(rig, OUT) == std::array<s32, 2>{1, 4});
        rig.Call("nvnSamplerGetBorderColor", {I(SAMPLER), I(OUT)});
        CHECK(Get<std::array<f32, 4>>(rig, OUT) == std::array<f32, 4>{1.0f, 0.5f, 0.25f, 0.125f});
        CHECK(rig.CallF("nvnSamplerGetMaxAnisotropy", {I(SAMPLER)}) == 16.0f);
        CHECK(static_cast<u32>(rig.Call("nvnSamplerGetReductionFilter", {I(SAMPLER)})) == 1);
        CHECK(rig.Call("nvnSamplerGetDebugID", {I(SAMPLER)}) == desc.serial);
    }

    SECTION("Compare is true for equal state and false after any difference") {
        REQUIRE((rig.Call("nvnSamplerInitialize", {I(SAMPLER2), I(SAMPLER_BUILDER)}) & 0xFF) == 1);
        CHECK((rig.Call("nvnSamplerCompare", {I(SAMPLER), I(SAMPLER2)}) & 0xFF) == 1);
        rig.Call("nvnSamplerBuilderSetLodBias", {I(SAMPLER_BUILDER), Rig::F(0.5f)});
        REQUIRE((rig.Call("nvnSamplerInitialize", {I(SAMPLER2), I(SAMPLER_BUILDER)}) & 0xFF) == 1);
        CHECK((rig.Call("nvnSamplerCompare", {I(SAMPLER), I(SAMPLER2)}) & 0xFF) == 0);
    }

    SECTION("Finalize clears the tag: getters answer zeros") {
        rig.Call("nvnSamplerFinalize", {I(SAMPLER)});
        CHECK(rig.CallF("nvnSamplerGetMaxAnisotropy", {I(SAMPLER)}) == 0.0f);
        CHECK((rig.Call("nvnSamplerCompare", {I(SAMPLER), I(SAMPLER)}) & 0xFF) == 0);
    }
}
