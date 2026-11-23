// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/nvn/sdk_impl_table.h"

// The SDK's GetProcAddress implementation table (FindSdkImplTables / SdkImplOffset) and the
// texture handle rule read from its nvnDeviceGetTextureHandle. The images are synthetic: the
// instruction words are the ones the games' SDKs compile these functions to (checked with a
// disassembler on Ocarina of Time's and Donkey Kong Bananza 1.0.0 / 3.2.0's images, which are not
// part of the repository).

using namespace Core::NvnHle;

namespace {

constexpr u32 Ret = 0xD65F03C0;
constexpr u32 LdrX0 = 0xF9400000;      // ldr x0, [x0]
constexpr u32 StrX1 = 0xF9000001;      // str x1, [x0]
constexpr u32 LdrW0Flags = 0xB9402000; // ldr w0, [x0, #0x20]
constexpr u32 StrW1Flags = 0xB9002001; // str w1, [x0, #0x20]
constexpr u32 SxtwX8W1 = 0x93407C28;
constexpr u32 SxtwX9W2 = 0x93407C49;
constexpr u32 CmpW1Zero = 0x7100003F;
constexpr u32 OrrX8X8X9Lsl20 = 0xAA095108;
constexpr u32 OrrX0X8X9Lsl20 = 0xAA095100;
constexpr u32 CselX0XzrX8Eq = 0x9A8803E0;
constexpr u32 CselX0X8XzrNe = 0x9A9F1100;
constexpr u32 SxtwX0W1 = 0x93407C20;

constexpr u32 IMAGE_SIZE = 0x2000;
constexpr u32 CODE = 0x100;
constexpr u32 TABLE = 0x1000;

struct Image {
    std::vector<u8> bytes = std::vector<u8>(IMAGE_SIZE);
    std::vector<std::string> names{
        "nvnDeviceGetImageHandle",        "nvnDeviceGetTextureHandle",
        "nvnDlssNetworkBuilderGetDevice", "nvnDlssNetworkBuilderGetFlags",
        "nvnDlssNetworkBuilderSetDevice", "nvnDlssNetworkBuilderSetFlags",
    };
    u32 pc = CODE;

    void Word(u32 offset, u32 value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }
    u32 Function(std::initializer_list<u32> words) {
        const u32 start = pc;
        for (const u32 w : words) {
            Word(pc, w);
            pc += 4;
        }
        pc += 0x10;
        return start;
    }
    void Entry(u32 table, std::string_view name, u64 target) {
        const auto it = std::ranges::find(names, name);
        REQUIRE(it != names.end());
        const u64 offset = table + 8 * static_cast<u64>(it - names.begin());
        std::memcpy(bytes.data() + offset, &target, sizeof(target));
    }
};

/// An SDK whose texture handle function is `texture_handle`.
Image MakeImage(std::initializer_list<u32> texture_handle) {
    Image image;
    REQUIRE(std::ranges::is_sorted(image.names));
    const u32 get_device = image.Function({LdrX0, Ret});
    const u32 set_device = image.Function({StrX1, Ret});
    const u32 get_flags = image.Function({LdrW0Flags, Ret});
    const u32 set_flags = image.Function({StrW1Flags, Ret});
    const u32 image_handle = image.Function({SxtwX0W1, Ret});
    const u32 handle = image.Function(texture_handle);
    image.Entry(TABLE, "nvnDlssNetworkBuilderGetDevice", get_device);
    image.Entry(TABLE, "nvnDlssNetworkBuilderSetDevice", set_device);
    image.Entry(TABLE, "nvnDlssNetworkBuilderGetFlags", get_flags);
    image.Entry(TABLE, "nvnDlssNetworkBuilderSetFlags", set_flags);
    image.Entry(TABLE, "nvnDeviceGetImageHandle", image_handle);
    image.Entry(TABLE, "nvnDeviceGetTextureHandle", handle);
    REQUIRE(image.pc < TABLE);
    return image;
}

/// Ocarina of Time's SDK (sdk+0x651c60): texture id 0 gives the null handle.
Image NullForZeroSdk() {
    return MakeImage({SxtwX8W1, SxtwX9W2, CmpW1Zero, OrrX8X8X9Lsl20, CselX0XzrX8Eq, Ret});
}

/// Donkey Kong Bananza 3.2.0's SDK (sdk+0x5d52c0): no texture id test.
Image PlainSdk() {
    return MakeImage({SxtwX8W1, SxtwX9W2, OrrX0X8X9Lsl20, Ret});
}

} // namespace

TEST_CASE("nvn sdk implementation table: found by its DLSS accessors, entries by name",
          "[nvn][sdkimpl]") {
    const Image image = NullForZeroSdk();
    const std::vector<u64> tables = FindSdkImplTables(image.bytes, image.names);
    REQUIRE(tables == std::vector<u64>{TABLE});
    const auto handle =
        SdkImplOffset(image.bytes, image.names, tables, "nvnDeviceGetTextureHandle");
    REQUIRE(handle.has_value());
    CHECK(SdkImageWord(image.bytes, *handle) == SxtwX8W1);
    CHECK(!SdkImplOffset(image.bytes, image.names, tables, "nvnQueueSubmitCommands"));
    CHECK(!SdkImplOffset(image.bytes, image.names, {}, "nvnDeviceGetTextureHandle"));

    // Unsorted names are not the SDK's binary-searched table; a missing accessor is no table.
    std::vector<std::string> unsorted = image.names;
    std::swap(unsorted[0], unsorted[1]);
    CHECK(FindSdkImplTables(image.bytes, unsorted).empty());
    Image broken = NullForZeroSdk();
    broken.Entry(TABLE, "nvnDlssNetworkBuilderGetFlags", CODE + 4);
    CHECK(FindSdkImplTables(broken.bytes, broken.names).empty());
}

TEST_CASE("nvn sdk implementation table: copies agree, tables that disagree give no entry",
          "[nvn][sdkimpl]") {
    constexpr u32 SECOND_TABLE = TABLE + 0x400;
    Image image = PlainSdk();
    const size_t table_bytes = 8 * image.names.size();
    std::copy_n(image.bytes.begin() + TABLE, table_bytes, image.bytes.begin() + SECOND_TABLE);
    std::vector<u64> tables = FindSdkImplTables(image.bytes, image.names);
    REQUIRE(tables == std::vector<u64>{TABLE, SECOND_TABLE});
    CHECK(SdkImplOffset(image.bytes, image.names, tables, "nvnDeviceGetImageHandle").has_value());
    image.Entry(SECOND_TABLE, "nvnDeviceGetImageHandle", CODE);
    tables = FindSdkImplTables(image.bytes, image.names);
    CHECK(!SdkImplOffset(image.bytes, image.names, tables, "nvnDeviceGetImageHandle"));
    CHECK(SdkImplOffset(image.bytes, image.names, tables, "nvnDeviceGetTextureHandle").has_value());
}

TEST_CASE("nvn sdk texture handle rule: only an SDK that tests the texture id nulls id 0",
          "[nvn][sdkimpl]") {
    const auto rule = [](const Image& image) {
        const std::vector<u64> tables = FindSdkImplTables(image.bytes, image.names);
        const auto handle =
            SdkImplOffset(image.bytes, image.names, tables, "nvnDeviceGetTextureHandle");
        REQUIRE(handle.has_value());
        return SdkTextureHandleNullForTextureZero(image.bytes, *handle);
    };
    CHECK(rule(NullForZeroSdk()));
    CHECK_FALSE(rule(PlainSdk()));
    // The same test written with the operands the other way round.
    CHECK(rule(MakeImage({SxtwX8W1, SxtwX9W2, CmpW1Zero, OrrX8X8X9Lsl20, CselX0X8XzrNe, Ret})));
    // A compare without the null select, or a select of another register's compare: no rule.
    CHECK_FALSE(rule(MakeImage({SxtwX8W1, SxtwX9W2, CmpW1Zero, OrrX0X8X9Lsl20, Ret})));
    CHECK_FALSE(rule(MakeImage({SxtwX8W1, SxtwX9W2, OrrX8X8X9Lsl20, CselX0XzrX8Eq, Ret})));
    // No `ret` within the function's few instructions: not this shape.
    CHECK_FALSE(rule(MakeImage({SxtwX8W1, SxtwX9W2, CmpW1Zero, OrrX8X8X9Lsl20, CselX0XzrX8Eq,
                                SxtwX8W1, SxtwX8W1, SxtwX8W1, Ret})));
}
