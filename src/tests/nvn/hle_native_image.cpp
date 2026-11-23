// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_native_image_directory.h"
#include "tests/nvn/hle_rig.h"

using namespace NvnHleTest;
using Service::Nx2NativeImageDirectory;
using Service::Nx2NativeImageInfo;

namespace {

constexpr const char* Initialize =
    "_Z24nvnNativeImageInitializeP14NVNnativeImagePK21NVNnativeImageBuilder";
constexpr const char* Finalize = "_Z22nvnNativeImageFinalizeP14NVNnativeImage";
constexpr const char* TextureFrom =
    "_Z35nvnTextureInitializeFromNativeImageP10NVNtextureP14NVNnativeImage";

constexpr u64 DEVICE = Rig::Obj(2);
constexpr u64 BUILDER = Rig::Obj(10);
constexpr u64 IMAGE = Rig::Obj(40); // 0x1a0 bytes: slots 40 and 41
constexpr u64 TEXTURE = Rig::Obj(60);
constexpr u32 HANDLE = 7;
constexpr u32 FORMAT_RGBA8 = 0x25;

/// What nx2:gsv answers for HANDLE: a 256x128 image of 0x20000 bytes at the start of the rig's
/// pool storage plus `offset` (the descriptor's offset word).
Nx2NativeImageInfo Info(u32 kind, u32 offset) {
    Nx2NativeImageInfo info;
    info.address = Rig::STORAGE;
    info.size = 0x20000;
    info.descriptor[0] = 1;
    info.descriptor[2] = 255u | (127u << 16);
    info.descriptor[8] = 1;
    info.descriptor[9] = kind;
    info.descriptor[10] = offset;
    info.descriptor[11] = 0x20000;
    return info;
}

/// Makes the directory answer for HANDLE for the lifetime of the object.
class ScopedImage {
public:
    explicit ScopedImage(Nx2NativeImageInfo info_) : info{info_} {
        Nx2NativeImageDirectory::Get().Add(this, [this](u32 handle) -> std::optional<Nx2NativeImageInfo> {
            return handle == HANDLE ? std::optional{info} : std::nullopt;
        });
    }
    ~ScopedImage() {
        Nx2NativeImageDirectory::Get().Remove(this);
    }
    ScopedImage(const ScopedImage&) = delete;
    ScopedImage& operator=(const ScopedImage&) = delete;

private:
    Nx2NativeImageInfo info;
};

void WriteBuilder(Rig& rig, u32 handle, u32 format) {
    rig.memory.Put<u64>(BUILDER, DEVICE);
    rig.memory.Put<u32>(BUILDER + 8, handle);
    rig.memory.Put<u32>(BUILDER + 0xC, 1);
    rig.memory.Put<u32>(BUILDER + 0x10, format);
}

} // namespace

TEST_CASE("nvn hle native image: the pool is the image's pages and the texture is built over it",
          "[nvn][nativeimage]") {
    Rig rig;
    const ScopedImage image_source{Info(1, 0x100)}; // the image starts 0x100 bytes into its first page
    WriteBuilder(rig, HANDLE, FORMAT_RGBA8);

    REQUIRE(rig.Call(Initialize, {Rig::I(IMAGE), Rig::I(BUILDER)}) == 1);
    const auto pool = rig.recorder.GetMemoryPool(IMAGE);
    REQUIRE(pool.has_value());
    CHECK(pool->address == Rig::STORAGE);
    CHECK(pool->size == 0x21000); // 0x100 + 0x20000 rounded up to pages
    CHECK(rig.memory.Get<u64>(IMAGE + 0x100) == DEVICE);
    CHECK(rig.memory.Get<u32>(IMAGE + 0x108) == FORMAT_RGBA8);
    CHECK(rig.memory.Get<u64>(IMAGE + 0x110) == 0x100); // storage offset in the pool
    CHECK(rig.memory.Get<u32>(IMAGE + 0x118) == 1);     // descriptor copy

    REQUIRE(rig.Call(TextureFrom, {Rig::I(TEXTURE), Rig::I(IMAGE)}) == 1);
    const auto texture = rig.recorder.GetTexture(TEXTURE);
    REQUIRE(texture.has_value());
    CHECK(texture->target == 1);
    CHECK(texture->format == static_cast<s32>(FORMAT_RGBA8));
    CHECK(texture->width == 256);
    CHECK(texture->height == 128);
    CHECK(texture->levels == 1);
    CHECK(texture->pool_offset == 0x100);
    CHECK(texture->flags == 5);
    // Layout kind 1: stored order reads as RGBA.
    CHECK(texture->set_swizzle == 1);
    CHECK(texture->swizzle == (2 | (3 << 3) | (4 << 6) | (5 << 9)));

    rig.Call(Finalize, {Rig::I(IMAGE)});
    CHECK(!rig.recorder.GetMemoryPool(IMAGE).has_value());
    CHECK(rig.memory.Get<u64>(IMAGE + 0x100) == 0);
}

TEST_CASE("nvn hle native image: refusals", "[nvn][nativeimage]") {
    Rig rig;
    WriteBuilder(rig, HANDLE, FORMAT_RGBA8);

    // No session knows the handle.
    CHECK(rig.Call(Initialize, {Rig::I(IMAGE), Rig::I(BUILDER)}) == 0);
    CHECK(!rig.recorder.GetMemoryPool(IMAGE).has_value());

    {
        // A descriptor without a size is not a user image.
        Nx2NativeImageInfo empty = Info(1, 0);
        empty.descriptor[11] = 0;
        const ScopedImage image_source{empty};
        CHECK(rig.Call(Initialize, {Rig::I(IMAGE), Rig::I(BUILDER)}) == 0);
    }

    {
        // Layout kind 1 takes RGBA formats only; kind 2 takes 0x99-0x9c and swaps red and blue.
        const ScopedImage image_source{Info(1, 0)};
        WriteBuilder(rig, HANDLE, 0x99);
        REQUIRE(rig.Call(Initialize, {Rig::I(IMAGE), Rig::I(BUILDER)}) == 1);
        CHECK(rig.Call(TextureFrom, {Rig::I(TEXTURE), Rig::I(IMAGE)}) == 0);
        CHECK(!rig.recorder.GetTexture(TEXTURE).has_value());
        rig.Call(Finalize, {Rig::I(IMAGE)});
    }
    {
        const ScopedImage image_source{Info(2, 0)};
        WriteBuilder(rig, HANDLE, 0x99);
        REQUIRE(rig.Call(Initialize, {Rig::I(IMAGE), Rig::I(BUILDER)}) == 1);
        REQUIRE(rig.Call(TextureFrom, {Rig::I(TEXTURE), Rig::I(IMAGE)}) == 1);
        const auto texture = rig.recorder.GetTexture(TEXTURE);
        REQUIRE(texture.has_value());
        CHECK(texture->swizzle == (4 | (3 << 3) | (2 << 6) | (5 << 9)));
    }
}
