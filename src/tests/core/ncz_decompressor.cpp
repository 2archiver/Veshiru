// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <zstd.h>

#include "core/crypto/ncz_decompressor.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace {

void PutLe(std::vector<u8>& buf, std::size_t off, u64 val, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
        buf[off + i] = static_cast<u8>((val >> (i * 8)) & 0xFF);
    }
}

std::vector<u8> CompressZstd(const u8* src, std::size_t len) {
    std::vector<u8> out(ZSTD_compressBound(len));
    const std::size_t written = ZSTD_compress(out.data(), out.size(), src, len, 1);
    REQUIRE(!ZSTD_isError(written));
    out.resize(written);
    return out;
}

} // namespace

TEST_CASE("NCZBLOCK header validates block count against decompressed size", "[ncz]") {
    constexpr std::size_t HeaderSize = 0x4000;
    constexpr std::size_t SectionStart = HeaderSize;
    constexpr std::size_t BlockStart = SectionStart + 16 + 0x40;
    constexpr std::size_t BlockSize = 1u << 14; // 16384
    constexpr std::size_t TotalPlainSize = 20000;

    std::vector<u8> plain(TotalPlainSize, 0x42);
    const auto blk0 = CompressZstd(plain.data(), BlockSize);
    const auto blk1 = CompressZstd(plain.data() + BlockSize, TotalPlainSize - BlockSize);

    std::vector<u8> bytes(BlockStart + 0x18 + 8 + blk0.size() + blk1.size(), 0);
    std::memcpy(bytes.data() + SectionStart, "NCZSECTN", 8);
    PutLe(bytes, SectionStart + 8, 1, 8);
    PutLe(bytes, SectionStart + 16, HeaderSize, 8);
    PutLe(bytes, SectionStart + 24, TotalPlainSize, 8);
    PutLe(bytes, SectionStart + 32, 1, 8); // plaintext crypto_type

    std::memcpy(bytes.data() + BlockStart, "NCZBLOCK", 8);
    bytes[BlockStart + 8] = 2;  // version
    bytes[BlockStart + 9] = 1;  // type
    bytes[BlockStart + 11] = 14; // exponent (16 KiB blocks -> 2 blocks for 20000 bytes)
    PutLe(bytes, BlockStart + 12, 2, 4);
    PutLe(bytes, BlockStart + 16, TotalPlainSize, 8);
    PutLe(bytes, BlockStart + 24, blk0.size(), 4);
    PutLe(bytes, BlockStart + 28, blk1.size(), 4);
    std::memcpy(bytes.data() + BlockStart + 32, blk0.data(), blk0.size());
    std::memcpy(bytes.data() + BlockStart + 32 + blk0.size(), blk1.data(), blk1.size());

    SECTION("valid NCZBLOCK header is accepted") {
        CHECK(Core::Crypto::IsSupportedNCZ(
            std::make_shared<FileSys::VectorVfsFile>(bytes)));
    }
    SECTION("block count that does not match the decompressed size is rejected") {
        bytes[BlockStart + 11] = 32; // 4 GiB blocks, so 20000 bytes must only have 1 block, not 2
        CHECK_FALSE(Core::Crypto::IsSupportedNCZ(
            std::make_shared<FileSys::VectorVfsFile>(std::move(bytes))));
    }
    SECTION("decompressed size needing more blocks than the table holds is rejected") {
        PutLe(bytes, BlockStart + 16, 40000, 8);
        CHECK_FALSE(Core::Crypto::IsSupportedNCZ(
            std::make_shared<FileSys::VectorVfsFile>(std::move(bytes))));
    }
}
