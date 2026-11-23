// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/nvn/sdk_dlss_layout.h"
#include "core/hle/nvn/sdk_impl_table.h"

// The DLSS object sizes read from the game's own SDK (ReadSdkDlssSizes) and the per-SDK network
// layouts (KnownDlssNetworkLayouts). The images are synthetic: the SDK's GetProcAddress
// implementation table and the three functions of the shapes the NVN2 SDKs compile them to,
// assembled here from the instruction forms (the real images are not part of the repository).

using namespace Core::NvnHle;

namespace {

constexpr u32 Ret = 0xD65F03C0;
constexpr u32 Pacibsp = 0xD503237F;
constexpr u32 MoviV0Zero = 0x6F00E400;      // movi v0.2d, #0
constexpr u32 FmovV0One = 0x0F03F600;       // fmov v0.2s, #1.0
constexpr u32 XZR = 31;

constexpr u32 MovZ32(u32 rd, u32 imm16, u32 halfword = 0) {
    return 0x52800000 | (halfword << 21) | (imm16 << 5) | rd;
}
constexpr u32 MovN64(u32 rd, u32 imm16) {
    return 0x92800000 | (imm16 << 5) | rd;
}
constexpr u32 MovReg32(u32 rd, u32 rm) {
    return 0x2A0003E0 | (rm << 16) | rd;
}
constexpr u32 MovReg64(u32 rd, u32 rm) {
    return 0xAA0003E0 | (rm << 16) | rd;
}
constexpr u32 SubSp(u32 imm) {
    return 0xD1000000 | (imm << 10) | (31 << 5) | 31;
}
constexpr u32 Bl(u32 pc, u32 target) {
    return 0x94000000 | (((target - pc) / 4) & 0x3FFFFFF);
}
constexpr u32 B(u32 pc, u32 target) {
    return 0x14000000 | (((target - pc) / 4) & 0x3FFFFFF);
}
constexpr u32 LdrX(u32 rt, u32 rn, u32 offset) {
    return 0xF9400000 | ((offset / 8) << 10) | (rn << 5) | rt;
}
constexpr u32 LdrW(u32 rt, u32 rn, u32 offset) {
    return 0xB9400000 | ((offset / 4) << 10) | (rn << 5) | rt;
}
constexpr u32 StrX(u32 rt, u32 rn, u32 offset) {
    return 0xF9000000 | ((offset / 8) << 10) | (rn << 5) | rt;
}
constexpr u32 StrW(u32 rt, u32 rn, u32 offset) {
    return 0xB9000000 | ((offset / 4) << 10) | (rn << 5) | rt;
}
constexpr u32 StrD(u32 rt, u32 rn, u32 offset) {
    return 0xFD000000 | ((offset / 8) << 10) | (rn << 5) | rt;
}
constexpr u32 StrQ(u32 rt, u32 rn, u32 offset) {
    return 0x3D800000 | ((offset / 16) << 10) | (rn << 5) | rt;
}
constexpr u32 SturQ(u32 rt, u32 rn, s32 offset) {
    return 0x3C800000 | ((static_cast<u32>(offset) & 0x1FF) << 12) | (rn << 5) | rt;
}
constexpr u32 StpX(u32 rt, u32 rt2, u32 rn, s32 offset) {
    return 0xA9000000 | ((static_cast<u32>(offset / 8) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) |
           rt;
}
constexpr u32 StpQ(u32 rt, u32 rt2, u32 rn, s32 offset) {
    return 0xAD000000 | ((static_cast<u32>(offset / 16) & 0x7F) << 15) | (rt2 << 10) |
           (rn << 5) | rt;
}

constexpr u32 IMAGE_SIZE = 0x4000;
constexpr u32 CODE = 0x100;
constexpr u32 TABLE = 0x3000;
constexpr u32 DECOY_TABLE = 0x2800;

/// One SDK generation's object sizes, as its functions are shaped.
struct Generation {
    u32 builder;
    u32 network;
    u32 parameters;
};
constexpr Generation NEWEST{0xC0, 0x258, 0xC0}; // API 231.39/231.40
constexpr Generation SDK_20_5{0xB8, 0x220, 0xC0}; // API 231.30
constexpr Generation OLDEST{0xC0, 0x218, 0xC8};   // API 231.16

std::vector<std::string> Names() {
    std::vector<std::string> names{
        "nvnBufferGetAddress",
        "nvnCommandBufferEvaluateDlss",
        "nvnDlssNetworkBuilderGetDevice",
        "nvnDlssNetworkBuilderGetFlags",
        "nvnDlssNetworkBuilderSetDefaults",
        "nvnDlssNetworkBuilderSetDevice",
        "nvnDlssNetworkBuilderSetFlags",
        "nvnDlssNetworkFinalize",
        "nvnDlssNetworkInitialize",
        "nvnDlssParametersSetDefaults",
        "nvnQueueSubmitCommands",
    };
    std::ranges::sort(names);
    return names;
}

struct Image {
    std::vector<u8> bytes = std::vector<u8>(IMAGE_SIZE);
    std::vector<std::string> names = Names();
    u32 pc = CODE;

    void Word(u32 offset, u32 value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }
    /// Appends a function; returns its address.
    u32 Function(std::initializer_list<u32> words) {
        const u32 start = pc;
        for (const u32 w : words) {
            Word(pc, w);
            pc += 4;
        }
        pc += 0x10; // padding between functions
        return start;
    }
    void Entry(u32 table, std::string_view name, u64 target) {
        const auto it = std::ranges::find(names, name);
        REQUIRE(it != names.end());
        const u64 offset = table + 8 * static_cast<u64>(it - names.begin());
        std::memcpy(bytes.data() + offset, &target, sizeof(target));
    }
};

/// nvnDlssNetworkBuilderSetDefaults: everything from +8 cleared, one evaluation entry; the 0xC0
/// builders also clear their last word.
u32 BuilderSetDefaults(Image& image, u32 size) {
    if (size == 0xC0) {
        return image.Function({MoviV0Zero, MovZ32(8, 1), StrX(XZR, 0, 0xB8), SturQ(0, 0, 0x28),
                               SturQ(0, 0, 0x08), SturQ(0, 0, 0x18), SturQ(0, 0, 0x38),
                               SturQ(0, 0, 0x48), SturQ(0, 0, 0x58), SturQ(0, 0, 0x68),
                               SturQ(0, 0, 0x78), SturQ(0, 0, 0x88), SturQ(0, 0, 0x98),
                               SturQ(0, 0, 0xA8), StrW(8, 0, 0x34), Ret});
    }
    return image.Function({MoviV0Zero, MovZ32(8, 1), SturQ(0, 0, 0x28), StrW(8, 0, 0x34),
                           SturQ(0, 0, 0x08), SturQ(0, 0, 0x18), SturQ(0, 0, 0x38),
                           SturQ(0, 0, 0x48), SturQ(0, 0, 0x58), SturQ(0, 0, 0x68),
                           SturQ(0, 0, 0x78), SturQ(0, 0, 0x88), SturQ(0, 0, 0x98),
                           SturQ(0, 0, 0xA8), Ret});
}

/// nvnDlssParametersSetDefaults: zero, scale (1, 1), pre-exposure 1, overlays ~0; the 0xC8
/// parameters clear their last word with the second half of an STP.
u32 ParametersSetDefaults(Image& image, u32 size) {
    const u32 tail = size == 0xC8 ? StpX(8, XZR, 0, 0xB8) : StpX(8, 8, 0, 0xB0);
    const u32 overlays = size == 0xC8 ? StrX(8, 0, 0xB0) : FmovV0One;
    return image.Function({MoviV0Zero, MovZ32(8, 0x3F80, 1), StpQ(0, 0, 0, 0x10),
                           StpQ(0, 0, 0, 0x30), StrQ(0, 0, 0), StpQ(0, 0, 0, 0x50),
                           StrW(8, 0, 0x60), MovN64(8, 0), tail, StpQ(0, 0, 0, 0x70),
                           StpQ(0, 0, 0, 0x90), overlays, FmovV0One, StrD(0, 0, 0x58), Ret});
}

/// nvnDlssNetworkInitialize: the prologue, then memset(network, 0, size).
u32 Initialize(Image& image, u32 size, u32 memset) {
    const u32 start = image.pc;
    return image.Function({Pacibsp, SubSp(0x90), MovReg64(20, 1), MovZ32(2, size),
                           MovReg32(1, XZR), MovReg64(19, 0), Bl(start + 24, memset),
                           LdrX(8, 20, 0), StrX(8, 19, 8), Ret});
}

Image MakeImage(const Generation& g) {
    Image image;
    const u32 memset = image.Function({Ret});
    const u32 other = image.Function({Ret});
    const u32 get_device = image.Function({LdrX(0, 0, 0), Ret});
    const u32 set_device = image.Function({StrX(1, 0, 0), Ret});
    const u32 get_flags = image.Function({LdrW(0, 0, 0x20), Ret});
    const u32 set_flags = image.Function({StrW(1, 0, 0x20), Ret});
    const u32 builder = BuilderSetDefaults(image, g.builder);
    const u32 initialize = Initialize(image, g.network, memset);
    const u32 parameters = ParametersSetDefaults(image, g.parameters);
    for (const std::string& name : image.names) {
        image.Entry(TABLE, name, other);
    }
    image.Entry(TABLE, "nvnDlssNetworkBuilderGetDevice", get_device);
    image.Entry(TABLE, "nvnDlssNetworkBuilderSetDevice", set_device);
    image.Entry(TABLE, "nvnDlssNetworkBuilderGetFlags", get_flags);
    image.Entry(TABLE, "nvnDlssNetworkBuilderSetFlags", set_flags);
    image.Entry(TABLE, "nvnDlssNetworkBuilderSetDefaults", builder);
    image.Entry(TABLE, "nvnDlssNetworkInitialize", initialize);
    image.Entry(TABLE, "nvnDlssParametersSetDefaults", parameters);
    // A table of thunks (the SDK's exported nvn* symbols jump through a runtime table): none of
    // its entries is an accessor, so it is not the implementation table.
    const u32 thunk = image.Function({B(image.pc, get_device)});
    for (const std::string& name : image.names) {
        image.Entry(DECOY_TABLE, name, thunk);
    }
    REQUIRE(image.pc < DECOY_TABLE);
    return image;
}

/// ReadSdkDlssSizes over the implementation tables FindSdkImplTables finds, as the bridge reads.
std::optional<SdkDlssSizes> Read(const std::vector<u8>& bytes,
                                 const std::vector<std::string>& names) {
    return ReadSdkDlssSizes(bytes, names, FindSdkImplTables(bytes, names));
}

} // namespace

TEST_CASE("nvn sdk dlss layout: the sizes of each SDK generation come from its own code",
          "[nvn][dlss]") {
    for (const Generation& g : {NEWEST, SDK_20_5, OLDEST}) {
        INFO("network " << g.network);
        const Image image = MakeImage(g);
        const auto sizes = Read(image.bytes, image.names);
        REQUIRE(sizes.has_value());
        CHECK(sizes->builder == g.builder);
        CHECK(sizes->network == g.network);
        CHECK(sizes->parameters == g.parameters);
        CHECK(sizes->network_initialize > CODE);
        const DlssLayout layout = DlssLayoutFor(*sizes);
        CHECK(layout.builder_size == g.builder);
        CHECK(layout.parameters_size == g.parameters);
        REQUIRE(layout.network != nullptr);
        CHECK(layout.network->size == g.network);
    }
}

TEST_CASE("nvn sdk dlss layout: anything but the expected shapes leaves the sizes unknown",
          "[nvn][dlss]") {
    // Not an NVN2 SDK with DLSS: no names, or no DLSS names.
    {
        const Image image = MakeImage(NEWEST);
        CHECK(!Read(image.bytes, {}).has_value());
        std::vector<std::string> names = image.names;
        std::erase(names, "nvnDlssNetworkInitialize");
        CHECK(!Read(image.bytes, names).has_value());
    }
    // No implementation table: the accessors are not where the table says.
    {
        Image image = MakeImage(NEWEST);
        image.Entry(TABLE, "nvnDlssNetworkBuilderGetFlags", CODE);
        CHECK(!Read(image.bytes, image.names).has_value());
    }
    // SetDefaults calls a function (another shape): not read.
    {
        Image image = MakeImage(NEWEST);
        const u32 builder = image.Function({MoviV0Zero, Bl(image.pc + 4, CODE), Ret});
        image.Entry(TABLE, "nvnDlssNetworkBuilderSetDefaults", builder);
        CHECK(!Read(image.bytes, image.names).has_value());
    }
    // Initialize's memset size comes from a register, or x0 is replaced before the call.
    {
        Image image = MakeImage(NEWEST);
        const u32 start = image.pc;
        const u32 initialize = image.Function(
            {Pacibsp, MovReg64(2, 9), MovReg32(1, XZR), Bl(start + 12, CODE), Ret});
        image.Entry(TABLE, "nvnDlssNetworkInitialize", initialize);
        CHECK(!Read(image.bytes, image.names).has_value());
    }
    {
        Image image = MakeImage(NEWEST);
        const u32 start = image.pc;
        const u32 initialize = image.Function(
            {Pacibsp, MovZ32(2, 0x258), MovReg32(1, XZR), MovReg64(0, 20), Bl(start + 16, CODE),
             Ret});
        image.Entry(TABLE, "nvnDlssNetworkInitialize", initialize);
        CHECK(!Read(image.bytes, image.names).has_value());
    }
    // A second implementation table: the same answer is kept, a different one is no answer.
    {
        constexpr u32 SECOND_TABLE = TABLE + 0x400;
        Image image = MakeImage(NEWEST);
        const size_t table_bytes = 8 * image.names.size();
        std::copy_n(image.bytes.begin() + TABLE, table_bytes, image.bytes.begin() + SECOND_TABLE);
        const auto same = Read(image.bytes, image.names);
        REQUIRE(same.has_value());
        CHECK(same->network == 0x258);
        image.Entry(SECOND_TABLE, "nvnDlssNetworkInitialize", Initialize(image, 0x220, CODE));
        REQUIRE(image.pc < DECOY_TABLE);
        CHECK(!Read(image.bytes, image.names).has_value());
    }
}

TEST_CASE("nvn sdk dlss layout: the network layouts of the three SDK generations",
          "[nvn][dlss]") {
    const auto layouts = KnownDlssNetworkLayouts();
    REQUIRE(layouts.size() == 3);
    // Newest first, and the default without an SDK.
    CHECK(layouts[0].size == 0x258);
    CHECK(NewestDlssLayout().network == &layouts[0]);
    CHECK(NewestDlssLayout().builder_size == 0xC0);
    CHECK(NewestDlssLayout().parameters_size == 0xC0);

    // Ocarina of Time / Star Fox (sdk+0x659b40 / +0x661340).
    const DlssNetworkLayout* newest = FindDlssNetworkLayout(0x258);
    REQUIRE(newest != nullptr);
    CHECK(newest->storage == 0xC8);
    CHECK(newest->storage_size == 0xD0);
    CHECK(newest->max_evaluation_entries == 0x200);
    CHECK(newest->method == 0x20C);
    CHECK(newest->quality == 0x214);
    CHECK(newest->preset == 0x218);
    CHECK(newest->output == 0x224);
    CHECK(newest->current_output == 0x240);
    CHECK(newest->initial_output == 0x248);
    CHECK(newest->flags == 0x250);

    // API 231.30 (SDK 20.5): 0x38 bytes shorter; storage 0x18 bytes earlier.
    const DlssNetworkLayout* sdk20 = FindDlssNetworkLayout(0x220);
    REQUIRE(sdk20 != nullptr);
    CHECK(sdk20->storage_device == 0xA8);
    CHECK(sdk20->storage == 0xB0);
    CHECK(sdk20->storage_size == 0xB8);
    CHECK(sdk20->max_evaluation_entries == 0x1C8);
    CHECK(sdk20->method == 0x1D4);
    CHECK(sdk20->ngx_flags == 0x1D8);
    CHECK(sdk20->quality == 0x1DC);
    CHECK(sdk20->preset == 0x1E0);
    CHECK(sdk20->optimal == 0x1E4);
    CHECK(sdk20->output == 0x1EC);
    CHECK(sdk20->output_subrects == 0x1F4);
    CHECK(sdk20->max_render == 0x1F8);
    CHECK(sdk20->min_render == 0x200);
    CHECK(sdk20->current_output == 0x208);
    CHECK(sdk20->initial_output == 0x210);
    CHECK(sdk20->flags == 0x218);

    // API 231.16: no initial output (no Resize); the flags follow the current output.
    const DlssNetworkLayout* oldest = FindDlssNetworkLayout(0x218);
    REQUIRE(oldest != nullptr);
    CHECK(oldest->current_output == 0x208);
    CHECK(oldest->initial_output == 0);
    CHECK(oldest->flags == 0x210);

    // Every field of every layout lies inside the network.
    for (const DlssNetworkLayout& l : layouts) {
        INFO(l.sdk);
        CHECK(l.size <= MaxDlssNetworkSize);
        for (const u32 end : {l.storage_size + 8, l.flags + 4, l.current_output + 8,
                              l.min_render + 8, l.output_subrects + 4}) {
            CHECK(end <= l.size);
        }
        if (l.initial_output != 0) {
            CHECK(l.initial_output + 8 <= l.size);
        }
    }

    // A size no known SDK has: no layout.
    CHECK(FindDlssNetworkLayout(0x230) == nullptr);
    CHECK(DlssLayoutFor(SdkDlssSizes{.builder = 0xC0, .network = 0x230, .parameters = 0xC0})
              .network == nullptr);
}
