// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/nvn/sdk_dlss_layout.h"
#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/record.h"

// NVNdlssNetworkBuilder / NVNdlssNetwork / NVNdlssParameters HLE and the EvaluateDlss record, as
// the games' own nnSdk does them (OoT 0400486024DEC000 sdk+0x6599d0..+0x65b010, Star Fox the
// same): the three in-place layouts, SetDefaults, the SDK's GetStorageSize formula (reference
// values taken from both SDKs' own code), the network lifecycle, and the rules EvaluateDlss applies
// before it calls the backend.
using namespace NvnHleTest;

namespace {

namespace VC = VideoCommon::Nvn;

constexpr u64 BUILDER = Rig::Obj(1);
constexpr u64 NETWORK = Rig::SCRATCH + 0x1000;  // the game gives the network 0x800 bytes
constexpr u64 NETWORK2 = Rig::SCRATCH + 0x4000; // a second network
constexpr u64 NETWORK_BYTES = 0x800;
constexpr u64 OUT = Rig::SCRATCH + 0x100;      // seven result slots, 8 bytes apart
constexpr u64 TEX_BUILDER = Rig::SCRATCH + 0x3000; // the tests' own texture builder
constexpr u64 DEVICE = 0xDE'0000;
constexpr u64 POOL = 0x7001'0000;
constexpr u64 IDS = Rig::SCRATCH + 0x400;
constexpr u64 IDS_OUT = Rig::SCRATCH + 0x600;
/// The network's working memory: the SDK's sizes are tens of MiB.
constexpr u64 DLSS_STORAGE = 0x2000'0000;
constexpr u64 DLSS_STORAGE_SIZE = 0x400'0000;

// Builder offsets (OoT sdk+0x65ab90..).
constexpr u64 OffDevice = 0x00;
constexpr u64 OffPool = 0x08;
constexpr u64 OffStorage = 0x10;
constexpr u64 OffStorageSize = 0x18;
constexpr u64 OffFlags = 0x20;
constexpr u64 OffMethod = 0x24;
constexpr u64 OffPreset = 0x28;
constexpr u64 OffWidth = 0x2c;
constexpr u64 OffHeight = 0x30;
constexpr u64 OffEntries = 0x34;
constexpr u64 OffIds = 0x38;
constexpr u64 OffReserved = 0xb8;
constexpr u64 BuilderEnd = 0xc0;
constexpr u64 IdBytes = 0x80;

// Network offsets (OoT sdk+0x659b40).
constexpr u64 NOffDevice = 0x08;
constexpr u64 NOffPool = 0x10;
constexpr u64 NOffIds = 0x18;
constexpr u64 NOffHandle = 0xa0;
constexpr u64 NOffDevice2 = 0xc0;
constexpr u64 NOffStorage = 0xc8;
constexpr u64 NOffStorageSize = 0xd0;
constexpr u64 NOffEntries = 0x200;
constexpr u64 NOffMethod = 0x20c;
constexpr u64 NOffNgxFlags = 0x210;
constexpr u64 NOffQuality = 0x214;
constexpr u64 NOffPreset = 0x218;
constexpr u64 NOffOptimal = 0x21c;
constexpr u64 NOffOutput = 0x224;
constexpr u64 NOffSubrects = 0x22c;
constexpr u64 NOffMax = 0x230;
constexpr u64 NOffMin = 0x238;
constexpr u64 NOffCurrent = 0x240;
constexpr u64 NOffInitial = 0x248;
constexpr u64 NOffFlags = 0x250;
constexpr u64 NetworkEnd = 0x258;

constexpr u32 MAX_PERF = 0;
constexpr u32 BALANCED = 1;
constexpr u32 MAX_QUALITY = 2;
constexpr u32 ULTRA_PERFORMANCE = 3;
constexpr u32 ULTRA_QUALITY = 4;
constexpr u32 DLAA = 5;

// Builder flags; Star Fox sets 0x4a (HDR colour, low-res MVs, inverted depth, auto exposure).
constexpr u32 FLAGS_STAR_FOX = 0x4a;
constexpr u32 FLAGS_HDR_MANUAL_EXPOSURE = 0x0a;

constexpr u64 FormatRG16F = 0x11;
constexpr u64 FormatRGBA16F = 0x29;

/// A rig with the network's working memory mapped.
struct DlssRig : Rig {
    DlssRig() {
        memory.Map(DLSS_STORAGE, DLSS_STORAGE_SIZE);
    }
};

void Defaults(Rig& rig, u64 builder = BUILDER) {
    rig.Call("nvnDlssNetworkBuilderSetDefaults", {Rig::I(builder)});
}

void Describe(Rig& rig, s32 width, s32 height, u32 method) {
    rig.Call("nvnDlssNetworkBuilderSetOutputResolution", {Rig::I(BUILDER), Rig::I(static_cast<u32>(width)),
                                                          Rig::I(static_cast<u32>(height))});
    rig.Call("nvnDlssNetworkBuilderSetMethod", {Rig::I(BUILDER), Rig::I(method)});
}

struct Optimal {
    u64 result;
    s32 optimal_w, optimal_h, max_w, max_h, min_w, min_h;
    float seventh;
};

constexpr s32 UNTOUCHED = -12345;

/// GetOptimalSettings with all six outputs, and a seventh pointer in x7 that must not be written
/// (the SDK forwards six).
Optimal GetOptimal(Rig& rig) {
    for (u64 i = 0; i < 6; ++i) {
        rig.memory.Put<s32>(OUT + 8 * i, UNTOUCHED);
    }
    rig.memory.Put<float>(OUT + 48, 1.0f);
    Optimal o{};
    o.result = rig.Call("nvnDlssNetworkBuilderGetOptimalSettings",
                        {Rig::I(BUILDER), Rig::I(OUT), Rig::I(OUT + 8), Rig::I(OUT + 16),
                         Rig::I(OUT + 24), Rig::I(OUT + 32), Rig::I(OUT + 40), Rig::I(OUT + 48)});
    o.optimal_w = rig.memory.Get<s32>(OUT);
    o.optimal_h = rig.memory.Get<s32>(OUT + 8);
    o.max_w = rig.memory.Get<s32>(OUT + 16);
    o.max_h = rig.memory.Get<s32>(OUT + 24);
    o.min_w = rig.memory.Get<s32>(OUT + 32);
    o.min_h = rig.memory.Get<s32>(OUT + 40);
    o.seventh = rig.memory.Get<float>(OUT + 48);
    return o;
}

u64 StorageSize(Rig& rig) {
    return rig.Call("nvnDlssNetworkBuilderGetStorageSize", {Rig::I(BUILDER)});
}

u64 PageUp(u64 bytes) {
    return (bytes + 0xfff) & ~u64{0xfff};
}

/// nvnTextureBuilderGetStorageSize of a default 2D texture builder of width x height and `format`
/// (that function's answers are pinned by its own tests), rounded up to pages.
u64 TexturePages(Rig& rig, s32 width, s32 height, u64 format) {
    rig.Call("nvnTextureBuilderSetDefaults", {Rig::I(TEX_BUILDER)});
    rig.Call("nvnTextureBuilderSetDevice", {Rig::I(TEX_BUILDER), Rig::I(DEVICE)});
    rig.Call("nvnTextureBuilderSetSize2D", {Rig::I(TEX_BUILDER), Rig::I(static_cast<u32>(width)),
                                            Rig::I(static_cast<u32>(height))});
    rig.Call("nvnTextureBuilderSetFormat", {Rig::I(TEX_BUILDER), Rig::I(format)});
    return PageUp(rig.Call("nvnTextureBuilderGetStorageSize", {Rig::I(TEX_BUILDER)}));
}

using IdArray = std::array<u8, IdBytes>;

IdArray IdPattern(u8 seed) {
    IdArray ids{};
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<u8>(seed + i * 7);
    }
    return ids;
}

/// The builder the game makes before Initialize: 1280x720 MaxQuality, 4 entries, texture IDs,
/// storage of exactly the size the builder asks for.
void BuildComplete(Rig& rig, u32 flags = FLAGS_STAR_FOX) {
    Defaults(rig);
    Describe(rig, 1280, 720, MAX_QUALITY);
    rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
    rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries", {Rig::I(BUILDER), Rig::I(4)});
    rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(flags)});
    rig.Call("nvnDlssNetworkBuilderSetPreset", {Rig::I(BUILDER), Rig::I(5)});
    rig.memory.Put(IDS, IdPattern(3));
    rig.Call("nvnDlssNetworkBuilderSetTextureIDs", {Rig::I(BUILDER), Rig::I(POOL), Rig::I(IDS)});
    const u64 size = StorageSize(rig);
    REQUIRE(size <= DLSS_STORAGE_SIZE);
    rig.Call("nvnDlssNetworkBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(DLSS_STORAGE), Rig::I(size)});
}

u64 Initialize(Rig& rig, u64 network = NETWORK) {
    return rig.Call("nvnDlssNetworkInitialize", {Rig::I(network), Rig::I(BUILDER)}) & 0xFF;
}

u64 Handle(Rig& rig, u64 network = NETWORK) {
    return rig.memory.Get<u64>(network + NOffHandle);
}

void FillNetwork(Rig& rig, u64 network, u8 value) {
    std::array<u8, NETWORK_BYTES> bytes;
    bytes.fill(value);
    rig.memory.Put(network, bytes);
}

} // namespace

// ---- NVNdlssNetworkBuilder ---------------------------------------------------------------------------

TEST_CASE("nvn hle dlss: builder setters write the SDK's in-place layout", "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);
    rig.memory.Put(IDS, IdPattern(0x40));
    rig.memory.Put<u64>(BUILDER + BuilderEnd, 0x1122334455667788ull);
    rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
    rig.Call("nvnDlssNetworkBuilderSetOutputResolution",
             {Rig::I(BUILDER), Rig::I(3840), Rig::I(2160)});
    rig.Call("nvnDlssNetworkBuilderSetMethod", {Rig::I(BUILDER), Rig::I(ULTRA_QUALITY)});
    rig.Call("nvnDlssNetworkBuilderSetPreset", {Rig::I(BUILDER), Rig::I(9)});
    rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0xA5A5'0001)});
    rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries", {Rig::I(BUILDER), Rig::I(6)});
    rig.Call("nvnDlssNetworkBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(Rig::STORAGE), Rig::I(0x2'0000'1000ull)});
    // (builder, pool, ids): no count argument; the 128 bytes are copied.
    rig.Call("nvnDlssNetworkBuilderSetTextureIDs", {Rig::I(BUILDER), Rig::I(POOL), Rig::I(IDS)});

    CHECK(rig.memory.Get<u64>(BUILDER + OffDevice) == DEVICE);
    CHECK(rig.memory.Get<u64>(BUILDER + OffPool) == POOL);
    CHECK(rig.memory.Get<u64>(BUILDER + OffStorage) == Rig::STORAGE);
    CHECK(rig.memory.Get<u64>(BUILDER + OffStorageSize) == 0x2'0000'1000ull);
    CHECK(rig.memory.Get<u32>(BUILDER + OffFlags) == 0xA5A5'0001);
    CHECK(rig.memory.Get<u32>(BUILDER + OffMethod) == ULTRA_QUALITY);
    CHECK(rig.memory.Get<u32>(BUILDER + OffPreset) == 9);
    CHECK(rig.memory.Get<s32>(BUILDER + OffWidth) == 3840);
    CHECK(rig.memory.Get<s32>(BUILDER + OffHeight) == 2160);
    CHECK(rig.memory.Get<s32>(BUILDER + OffEntries) == 6);
    CHECK(rig.memory.Get<IdArray>(BUILDER + OffIds) == IdPattern(0x40));
    CHECK(rig.memory.Get<u64>(BUILDER + OffReserved) == 0);
    // A copy, not a pointer: the game may reuse its array.
    rig.memory.Put(IDS, IdPattern(0x90));
    CHECK(rig.memory.Get<IdArray>(BUILDER + OffIds) == IdPattern(0x40));
    CHECK(rig.memory.Get<u64>(BUILDER + BuilderEnd) == 0x1122334455667788ull);
}

TEST_CASE("nvn hle dlss: builder getters read the layout", "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);
    rig.memory.Put<u64>(BUILDER + OffDevice, DEVICE);
    rig.memory.Put<u64>(BUILDER + OffPool, POOL);
    rig.memory.Put<u64>(BUILDER + OffStorage, Rig::STORAGE);
    rig.memory.Put<u64>(BUILDER + OffStorageSize, 0x1'2345'6000ull);
    rig.memory.Put<u32>(BUILDER + OffFlags, 0x80000003u);
    rig.memory.Put<u32>(BUILDER + OffMethod, DLAA);
    rig.memory.Put<u32>(BUILDER + OffPreset, 8);
    rig.memory.Put<s32>(BUILDER + OffWidth, 2560);
    rig.memory.Put<s32>(BUILDER + OffHeight, 1440);
    rig.memory.Put<s32>(BUILDER + OffEntries, 12);
    rig.memory.Put(BUILDER + OffIds, IdPattern(0x21));

    CHECK(rig.Call("nvnDlssNetworkBuilderGetDevice", {Rig::I(BUILDER)}) == DEVICE);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetTexturePool", {Rig::I(BUILDER)}) == POOL);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetMemory", {Rig::I(BUILDER)}) == Rig::STORAGE);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetMemorySize", {Rig::I(BUILDER)}) == 0x1'2345'6000ull);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetFlags", {Rig::I(BUILDER)}) == 0x80000003u);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetMethod", {Rig::I(BUILDER)}) == DLAA);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetPreset", {Rig::I(BUILDER)}) == 8);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetMaxEvaluationEntries", {Rig::I(BUILDER)}) == 12);

    // (builder, void* out): the 128 bytes are copied out, nothing past them.
    std::array<u8, IdBytes + 8> out;
    out.fill(0xEE);
    rig.memory.Put(IDS_OUT, out);
    rig.Call("nvnDlssNetworkBuilderGetTextureIDs", {Rig::I(BUILDER), Rig::I(IDS_OUT)});
    CHECK(rig.memory.Get<IdArray>(IDS_OUT) == IdPattern(0x21));
    CHECK(rig.memory.Get<u64>(IDS_OUT + IdBytes) == 0xEEEE'EEEE'EEEE'EEEEull);

    // (builder, int* w, int* h): both stored.
    rig.memory.Put<s32>(OUT, UNTOUCHED);
    rig.memory.Put<s32>(OUT + 8, UNTOUCHED);
    rig.Call("nvnDlssNetworkBuilderGetOutputResolution",
             {Rig::I(BUILDER), Rig::I(OUT), Rig::I(OUT + 8)});
    CHECK(rig.memory.Get<s32>(OUT) == 2560);
    CHECK(rig.memory.Get<s32>(OUT + 8) == 1440);
}

TEST_CASE("nvn hle dlss: builder SetDefaults keeps the device and sets one entry", "[nvn][dlss]") {
    Rig rig;
    std::array<u8, BuilderEnd> garbage;
    garbage.fill(0xFF);
    rig.memory.Put(BUILDER, garbage);
    rig.memory.Put<u64>(BUILDER + OffDevice, DEVICE);
    rig.memory.Put<u64>(BUILDER + BuilderEnd, 0x1122334455667788ull);
    Defaults(rig);
    CHECK(rig.memory.Get<u64>(BUILDER + OffDevice) == DEVICE);
    CHECK(rig.memory.Get<s32>(BUILDER + OffEntries) == 1);
    for (u64 offset = OffPool; offset < BuilderEnd; offset += 4) {
        if (offset == OffEntries) {
            continue;
        }
        INFO(offset);
        CHECK(rig.memory.Get<u32>(BUILDER + offset) == 0);
    }
    CHECK(rig.memory.Get<u64>(BUILDER + BuilderEnd) == 0x1122334455667788ull);
    CHECK(rig.Call("nvnDlssNetworkBuilderGetMaxEvaluationEntries", {Rig::I(BUILDER)}) == 1);
}

TEST_CASE("nvn hle dlss: optimal settings per method write six outputs, never a seventh",
          "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);

    struct Row {
        u32 method;
        s32 optimal_w, optimal_h;
    };
    // 3840x2160 output: 1/2, 0.58 (2227.2 and 1252.8 round), 2/3, 1/3, 0.77 (2956.8 and 1663.2
    // round) and the full output for DLAA (the host's numbers: the SDK asks the system library).
    constexpr std::array<Row, 6> rows{{
        {MAX_PERF, 1920, 1080},
        {BALANCED, 2227, 1253},
        {MAX_QUALITY, 2560, 1440},
        {ULTRA_PERFORMANCE, 1280, 720},
        {ULTRA_QUALITY, 2957, 1663},
        {DLAA, 3840, 2160},
    }};
    for (const Row& row : rows) {
        INFO("method " << row.method);
        Describe(rig, 3840, 2160, row.method);
        const Optimal o = GetOptimal(rig);
        CHECK(o.result == 1);
        CHECK(o.optimal_w == row.optimal_w);
        CHECK(o.optimal_h == row.optimal_h);
        CHECK(o.max_w == 3840);
        CHECK(o.max_h == 2160);
        CHECK(o.min_w == std::min(1280, row.optimal_w));
        CHECK(o.min_h == std::min(720, row.optimal_h));
        // x7 is not an output of the SDK's function: whatever it points at stays.
        CHECK(o.seventh == 1.0f);
    }
}

TEST_CASE("nvn hle dlss: optimal settings round half up and stay within 1..output", "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);

    Describe(rig, 3, 5, MAX_PERF); // 1.5 and 2.5 round up
    Optimal o = GetOptimal(rig);
    CHECK(o.result == 1);
    CHECK(o.optimal_w == 2);
    CHECK(o.optimal_h == 3);
    CHECK(o.max_w == 3);
    CHECK(o.max_h == 5);
    CHECK(o.min_w == 1);
    CHECK(o.min_h == 2);

    Describe(rig, 10, 10, ULTRA_PERFORMANCE); // 3.33 -> 3; the floor ceil(10/3) = 4 is capped to 3
    o = GetOptimal(rig);
    CHECK(o.optimal_w == 3);
    CHECK(o.min_w == 3);
    CHECK(o.max_w == 10);

    Describe(rig, 1, 1, ULTRA_PERFORMANCE); // never below one pixel
    o = GetOptimal(rig);
    CHECK(o.result == 1);
    CHECK(o.optimal_w == 1);
    CHECK(o.optimal_h == 1);
    CHECK(o.min_w == 1);
    CHECK(o.min_h == 1);

    // Large sizes use 64-bit intermediate products.
    Describe(rig, 0x7FFF'FFFF, 0x7FFF'FFFF, BALANCED);
    o = GetOptimal(rig);
    CHECK(o.result == 1);
    CHECK(o.optimal_w == 1245540515); // 2147483647 * 29 / 50 = 1245540515.26
    CHECK(o.max_w == 0x7FFF'FFFF);
}

TEST_CASE("nvn hle dlss: optimal settings skip NULL outputs", "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);
    Describe(rig, 1920, 1080, MAX_PERF);

    for (u64 i = 0; i < 6; ++i) {
        rig.memory.Put<s32>(OUT + 8 * i, UNTOUCHED);
    }
    CHECK(rig.Call("nvnDlssNetworkBuilderGetOptimalSettings",
                   {Rig::I(BUILDER), Rig::I(OUT), Rig::I(OUT + 8), Rig::I(0), Rig::I(0), Rig::I(0),
                    Rig::I(0)}) == 1);
    CHECK(rig.memory.Get<s32>(OUT) == 960);
    CHECK(rig.memory.Get<s32>(OUT + 8) == 540);
    for (u64 i = 2; i < 6; ++i) {
        CHECK(rig.memory.Get<s32>(OUT + 8 * i) == UNTOUCHED);
    }

    for (u64 i = 0; i < 6; ++i) {
        rig.memory.Put<s32>(OUT + 8 * i, UNTOUCHED);
    }
    CHECK(rig.Call("nvnDlssNetworkBuilderGetOptimalSettings",
                   {Rig::I(BUILDER), Rig::I(0), Rig::I(0), Rig::I(OUT + 16), Rig::I(OUT + 24),
                    Rig::I(OUT + 32), Rig::I(OUT + 40)}) == 1);
    CHECK(rig.memory.Get<s32>(OUT) == UNTOUCHED);
    CHECK(rig.memory.Get<s32>(OUT + 8) == UNTOUCHED);
    CHECK(rig.memory.Get<s32>(OUT + 16) == 1920);
    CHECK(rig.memory.Get<s32>(OUT + 24) == 1080);
    CHECK(rig.memory.Get<s32>(OUT + 32) == 640);
    CHECK(rig.memory.Get<s32>(OUT + 40) == 360);

    CHECK(rig.Call("nvnDlssNetworkBuilderGetOptimalSettings",
                   {Rig::I(BUILDER), Rig::I(0), Rig::I(0), Rig::I(0), Rig::I(0), Rig::I(0),
                    Rig::I(0)}) == 1);
}

TEST_CASE("nvn hle dlss: optimal settings refuse an unknown method or a bad output size",
          "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);

    const auto refused = [&](const char* what) {
        INFO(what);
        const Optimal o = GetOptimal(rig);
        CHECK(o.result == 0);
        CHECK(o.optimal_w == UNTOUCHED);
        CHECK(o.optimal_h == UNTOUCHED);
        CHECK(o.max_w == UNTOUCHED);
        CHECK(o.max_h == UNTOUCHED);
        CHECK(o.min_w == UNTOUCHED);
        CHECK(o.min_h == UNTOUCHED);
        CHECK(o.seventh == 1.0f);
    };

    for (const u32 method : {6u, 8u, 0x100u, 0xFFFF'FFFFu}) {
        Describe(rig, 1920, 1080, method);
        refused("unknown method");
    }
    Describe(rig, 0, 1080, MAX_QUALITY);
    refused("zero width");
    Describe(rig, 1920, 0, MAX_QUALITY);
    refused("zero height");
    Describe(rig, -1920, 1080, MAX_QUALITY);
    refused("negative width");
    Describe(rig, 1920, -1, MAX_QUALITY);
    refused("negative height");
}

TEST_CASE("nvn hle dlss: storage size is the SDK's (reference values from its own code)",
          "[nvn][dlss]") {
    // (entries, method, preset, W, H, flags, device flags) -> the SDK's answer with both working
    // texture sizes forced to 0. Presets 8/9 ask for no texture: exact. Otherwise the RG16F
    // texture counts twice and the RGBA16F one once.
    struct Row {
        s32 entries;
        u32 method;
        u32 preset;
        s32 width, height;
        u32 flags;
        u32 device_flags;
        u64 reference;
    };
    constexpr std::array<Row, 8> rows{{
        {1, 2, 0, 1920, 1080, 0x4a, 0, 0x34c4000},
        {3, 0, 5, 1920, 1080, 0x4a, 0, 0x34e4000},
        {1, 1, 4, 1280, 720, 0x2000004a, 0, 0x2869000},
        {1, 2, 0, 3840, 2160, 0x4a, 0xc00, 0x986f000},
        {2, 0, 8, 1920, 1080, 0x4a, 0, 0x3208000},
        {1, 1, 9, 1280, 720, 0x20000000, 0xc00, 0x1968000},
        {1, 3, 8, 2560, 1440, 0, 0x400, 0x4ec3000},
        {1, 5, 0, 1920, 1080, 0x1000004a, 0x400, 0x34c4000},
    }};
    Rig rig;
    for (const Row& row : rows) {
        INFO("entries " << row.entries << " method " << row.method << " preset " << row.preset
                        << " " << row.width << "x" << row.height << " flags " << row.flags
                        << " device " << row.device_flags);
        rig.runtime.device.flags = row.device_flags;
        Defaults(rig);
        rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
        Describe(rig, row.width, row.height, row.method);
        rig.Call("nvnDlssNetworkBuilderSetPreset", {Rig::I(BUILDER), Rig::I(row.preset)});
        rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(row.flags)});
        rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries",
                 {Rig::I(BUILDER), Rig::I(static_cast<u32>(row.entries))});
        const bool preset_8_9 = (row.preset & ~1u) == 8;
        u64 expected = row.reference;
        if (!preset_8_9) {
            const u64 rg16f = TexturePages(rig, row.width, row.height, FormatRG16F);
            const u64 rgba16f = TexturePages(rig, row.width, row.height, FormatRGBA16F);
            REQUIRE(rg16f != 0);
            REQUIRE(rgba16f != 0);
            expected += 2 * rg16f + rgba16f;
        }
        CHECK(StorageSize(rig) == expected);
        // The texture builder runs on the guest stack, never in the game's builder.
        CHECK(rig.memory.Get<u32>(BUILDER + OffPreset) == row.preset);
        CHECK(rig.memory.Get<u64>(BUILDER + OffDevice) == DEVICE);
    }
}

TEST_CASE("nvn hle dlss: storage size follows entries, flag bit 29 and the device flags",
          "[nvn][dlss]") {
    Rig rig;
    Defaults(rig);
    rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
    Describe(rig, 1920, 1080, MAX_QUALITY);
    const u64 one = StorageSize(rig);
    CHECK(one % 0x1000 == 0);
    // 64 KiB per evaluation entry.
    rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries", {Rig::I(BUILDER), Rig::I(5)});
    CHECK(StorageSize(rig) == one + 4 * 0x10000);
    rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries", {Rig::I(BUILDER), Rig::I(1)});
    // Bit 28 and the bits that reach the backend do not change the size; bit 29 enlarges it.
    rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0x1000'007F)});
    CHECK(StorageSize(rig) == one);
    rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0x2000'0000)});
    CHECK(StorageSize(rig) > one);
    rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0)});
    // Only both device flag bits 10 and 11 select the larger set.
    rig.runtime.device.flags = 0x400;
    CHECK(StorageSize(rig) == one);
    rig.runtime.device.flags = 0x800;
    CHECK(StorageSize(rig) == one);
    rig.runtime.device.flags = 0xc00;
    CHECK(StorageSize(rig) > one);
}

// ---- NVNdlssNetwork ---------------------------------------------------------------------------------

TEST_CASE("nvn hle dlss: initialize clears the network and fills the SDK's fields", "[nvn][dlss]") {
    DlssRig rig;
    BuildComplete(rig, 0x1000'004a);
    const u64 size = StorageSize(rig);
    FillNetwork(rig, NETWORK, 0xCC);
    // The game's storage is cleared, and nothing past it.
    rig.memory.Put<u64>(DLSS_STORAGE, 0xABCD'ABCD'ABCD'ABCDull);
    rig.memory.Put<u64>(DLSS_STORAGE + size / 2, 0xABCD'ABCD'ABCD'ABCDull);
    rig.memory.Put<u64>(DLSS_STORAGE + size - 8, 0xABCD'ABCD'ABCD'ABCDull);
    rig.memory.Put<u64>(DLSS_STORAGE + size, 0x5555'5555'5555'5555ull);

    REQUIRE(Initialize(rig) == 1);
    CHECK(rig.memory.Get<u64>(DLSS_STORAGE) == 0);
    CHECK(rig.memory.Get<u64>(DLSS_STORAGE + size / 2) == 0);
    CHECK(rig.memory.Get<u64>(DLSS_STORAGE + size - 8) == 0);
    CHECK(rig.memory.Get<u64>(DLSS_STORAGE + size) == 0x5555'5555'5555'5555ull);

    CHECK(rig.memory.Get<u64>(NETWORK + 0x00) == 0);
    CHECK(rig.memory.Get<u64>(NETWORK + NOffDevice) == DEVICE);
    CHECK(rig.memory.Get<u64>(NETWORK + NOffPool) == POOL);
    CHECK(rig.memory.Get<IdArray>(NETWORK + NOffIds) == IdPattern(3));
    CHECK(Handle(rig) != 0);
    CHECK(rig.memory.Get<u64>(NETWORK + NOffDevice2) == DEVICE);
    CHECK(rig.memory.Get<u64>(NETWORK + NOffStorage) == DLSS_STORAGE);
    CHECK(rig.memory.Get<u64>(NETWORK + NOffStorageSize) == size);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffEntries) == 4);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffMethod) == MAX_QUALITY);
    // NGX create flags: bits 0x4e as given, IsHDR = !(bit 0); bit 28 is not an NGX flag here.
    CHECK(rig.memory.Get<u32>(NETWORK + NOffNgxFlags) == 0x4b);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffQuality) == MAX_QUALITY);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffPreset) == 5);
    // Render sizes in GetOptimalSettings' order: optimal, max, min (1280x720 MaxQuality).
    CHECK(rig.memory.Get<s32>(NETWORK + NOffOptimal) == 853);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffOptimal + 4) == 480);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffOutput) == 1280);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffOutput + 4) == 720);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffSubrects) == 1);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffMax) == 1280);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffMax + 4) == 720);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffMin) == 427);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffMin + 4) == 240);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffCurrent) == 1280);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffCurrent + 4) == 720);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffInitial) == 1280);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffInitial + 4) == 720);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffFlags) == 0x1000'004au);
    // Fields Initialize does not set are cleared; the rest of the game's 0x800 bytes is its own.
    for (const u64 offset : {u64{0x98}, u64{0xa8}, u64{0xb0}, u64{0xb8}, u64{0xd8}, u64{0x1f8},
                             u64{0x204}, u64{0x254}}) {
        INFO(offset);
        CHECK(rig.memory.Get<u32>(NETWORK + offset) == 0);
    }
    CHECK(rig.memory.Get<u64>(NETWORK + NetworkEnd) == 0xCCCC'CCCC'CCCC'CCCCull);
    CHECK(rig.memory.Get<u64>(NETWORK + NETWORK_BYTES - 8) == 0xCCCC'CCCC'CCCC'CCCCull);
}

TEST_CASE("nvn hle dlss: initialize forwards the flags to NGX as the SDK does", "[nvn][dlss]") {
    DlssRig rig;
    struct Row {
        u32 flags;
        u32 ngx;
        u32 subrects;
    };
    // Bit 0 set = LDR (NGX IsHDR clear); 0x10, 0x20, 0x80 and bit 29 never reach NGX.
    constexpr std::array<Row, 6> rows{{
        {0x0, 0x1, 0},
        {0x1, 0x0, 0},
        {0x4a, 0x4b, 0},
        {0xff, 0x4e, 0},
        {0x30, 0x1, 0},
        {0x3000'0001, 0x0, 1},
    }};
    for (const Row& row : rows) {
        INFO(row.flags);
        BuildComplete(rig, row.flags);
        REQUIRE(Initialize(rig) == 1);
        CHECK(rig.memory.Get<u32>(NETWORK + NOffNgxFlags) == row.ngx);
        CHECK(rig.memory.Get<u32>(NETWORK + NOffSubrects) == row.subrects);
        CHECK(rig.memory.Get<u32>(NETWORK + NOffFlags) == row.flags);
    }
}

TEST_CASE("nvn hle dlss: every initialize gets a new nonzero network handle", "[nvn][dlss]") {
    DlssRig rig;
    BuildComplete(rig);
    std::vector<u64> handles;
    REQUIRE(Initialize(rig, NETWORK) == 1);
    handles.push_back(Handle(rig, NETWORK));
    REQUIRE(Initialize(rig, NETWORK2) == 1);
    handles.push_back(Handle(rig, NETWORK2));
    // Re-initializing the same network, also after Finalize, is a new feature.
    REQUIRE(Initialize(rig, NETWORK) == 1);
    handles.push_back(Handle(rig, NETWORK));
    rig.Call("nvnDlssNetworkFinalize", {Rig::I(NETWORK)});
    REQUIRE(Initialize(rig, NETWORK) == 1);
    handles.push_back(Handle(rig, NETWORK));
    for (const u64 handle : handles) {
        CHECK(handle != 0);
        CHECK(handle <= 0xFFFF'FFFFull);
    }
    std::vector<u64> sorted = handles;
    std::sort(sorted.begin(), sorted.end());
    CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
}

TEST_CASE("nvn hle dlss: initialize accepts storage larger than needed and every NGX method",
          "[nvn][dlss]") {
    DlssRig rig;
    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(DLSS_STORAGE), Rig::I(StorageSize(rig) + 0x5000)});
    CHECK(Initialize(rig) == 1);

    for (const u32 method : {MAX_PERF, BALANCED, MAX_QUALITY, ULTRA_PERFORMANCE, ULTRA_QUALITY, DLAA}) {
        INFO(method);
        rig.Call("nvnDlssNetworkBuilderSetMethod", {Rig::I(BUILDER), Rig::I(method)});
        rig.Call("nvnDlssNetworkBuilderSetStorage",
                 {Rig::I(BUILDER), Rig::I(DLSS_STORAGE), Rig::I(StorageSize(rig))});
        CHECK(Initialize(rig) == 1);
        CHECK(rig.memory.Get<u32>(NETWORK + NOffMethod) == method);
        CHECK(rig.memory.Get<u32>(NETWORK + NOffQuality) == method);
    }
}

TEST_CASE("nvn hle dlss: a refused initialize leaves the network zeroed", "[nvn][dlss]") {
    DlssRig rig;

    const auto refused = [&](const char* what) {
        INFO(what);
        FillNetwork(rig, NETWORK, 0xCC);
        CHECK(Initialize(rig) == 0);
        // The SDK clears the network before it looks at the builder.
        for (u64 offset = 0; offset < NetworkEnd; offset += 8) {
            INFO(offset);
            CHECK(rig.memory.Get<u64>(NETWORK + offset) == 0);
        }
        CHECK(Handle(rig) == 0);
        CHECK(rig.memory.Get<u64>(NETWORK + NetworkEnd) == 0xCCCC'CCCC'CCCC'CCCCull);
    };

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0)});
    refused("no device");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetStorage", {Rig::I(BUILDER), Rig::I(0), Rig::I(StorageSize(rig))});
    refused("no storage");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(DLSS_STORAGE), Rig::I(StorageSize(rig) - 0x1000)});
    refused("storage one page short");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetStorage", {Rig::I(BUILDER), Rig::I(DLSS_STORAGE), Rig::I(0)});
    refused("zero storage size");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(0x5000'0000), Rig::I(StorageSize(rig))});
    refused("storage the guest does not map");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetOutputResolution", {Rig::I(BUILDER), Rig::I(0), Rig::I(720)});
    refused("zero width");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetOutputResolution",
             {Rig::I(BUILDER), Rig::I(1280), Rig::I(static_cast<u32>(-1))});
    refused("negative height");

    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetMethod", {Rig::I(BUILDER), Rig::I(6)});
    refused("unknown method");

    // The storage must cover what the entries need: raise the count after the storage was set.
    BuildComplete(rig);
    rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries", {Rig::I(BUILDER), Rig::I(64)});
    refused("storage sized for fewer entries");

    Defaults(rig);
    refused("empty builder");
}

TEST_CASE("nvn hle dlss: finalize clears only the network handle", "[nvn][dlss]") {
    DlssRig rig;
    BuildComplete(rig);
    REQUIRE(Initialize(rig) == 1);
    std::array<u8, NETWORK_BYTES> before{};
    REQUIRE(rig.memory.Read(NETWORK, before.data(), before.size()));
    rig.Call("nvnDlssNetworkFinalize", {Rig::I(NETWORK)});
    std::array<u8, NETWORK_BYTES> after{};
    REQUIRE(rig.memory.Read(NETWORK, after.data(), after.size()));
    CHECK(Handle(rig) == 0);
    for (u64 i = 0; i < NETWORK_BYTES; ++i) {
        if (i >= NOffHandle && i < NOffHandle + 8) {
            continue;
        }
        INFO("byte " << i);
        CHECK(after[i] == before[i]);
    }
    // A finalized network can be built again.
    CHECK(Initialize(rig) == 1);
    CHECK(Handle(rig) != 0);
}

TEST_CASE("nvn hle dlss: resize stays within the size the network was initialized with",
          "[nvn][dlss]") {
    DlssRig rig;
    BuildComplete(rig);
    const auto resize = [&](s32 width, s32 height) {
        return rig.Call("nvnDlssNetworkResize", {Rig::I(NETWORK), Rig::I(static_cast<u32>(width)),
                                                 Rig::I(static_cast<u32>(height))}) &
               0xFF;
    };
    const auto pair = [&](u64 offset) {
        return std::array<s32, 2>{rig.memory.Get<s32>(NETWORK + offset),
                                  rig.memory.Get<s32>(NETWORK + offset + 4)};
    };
    using Size = std::array<s32, 2>;

    // A network that was never initialized cannot be resized.
    CHECK(resize(640, 360) == 0);
    CHECK(pair(NOffCurrent) == Size{0, 0});

    REQUIRE(Initialize(rig) == 1);
    const u64 handle = Handle(rig);
    CHECK(resize(960, 540) == 1);
    CHECK(pair(NOffCurrent) == Size{960, 540});
    CHECK(pair(NOffOutput) == Size{960, 540});
    // The render sizes follow for the network's method (MaxQuality).
    CHECK(pair(NOffOptimal) == Size{640, 360});
    CHECK(pair(NOffMax) == Size{960, 540});
    CHECK(pair(NOffMin) == Size{320, 180});
    // The bound and everything else stay.
    CHECK(pair(NOffInitial) == Size{1280, 720});
    CHECK(Handle(rig) == handle);
    CHECK(rig.memory.Get<u64>(NETWORK + NOffDevice) == DEVICE);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffMethod) == MAX_QUALITY);

    // Larger than the Initialize size in either direction: refused, the size stays.
    CHECK(resize(1281, 720) == 0);
    CHECK(resize(1280, 721) == 0);
    CHECK(pair(NOffCurrent) == Size{960, 540});
    // Back up to the Initialize size is fine.
    CHECK(resize(1280, 720) == 1);
    CHECK(pair(NOffCurrent) == Size{1280, 720});
    CHECK(pair(NOffOptimal) == Size{853, 480});

    // (host) Not positive: refused.
    CHECK(resize(0, 360) == 0);
    CHECK(resize(640, -4) == 0);
    CHECK(pair(NOffCurrent) == Size{1280, 720});

    rig.Call("nvnDlssNetworkFinalize", {Rig::I(NETWORK)});
    CHECK(resize(640, 360) == 0);
    CHECK(pair(NOffCurrent) == Size{1280, 720});
}

TEST_CASE("nvn hle dlss: the sequence the game runs", "[nvn][dlss]") {
    DlssRig rig;
    // SetDefaults, resolution, method, preset, optimal settings, device, texture IDs, flags,
    // entries, storage size, storage, initialize.
    Defaults(rig);
    rig.Call("nvnDlssNetworkBuilderSetOutputResolution", {Rig::I(BUILDER), Rig::I(1280), Rig::I(720)});
    rig.Call("nvnDlssNetworkBuilderSetMethod", {Rig::I(BUILDER), Rig::I(MAX_PERF)});
    rig.Call("nvnDlssNetworkBuilderSetPreset", {Rig::I(BUILDER), Rig::I(4)});
    const Optimal o = GetOptimal(rig);
    REQUIRE(o.result == 1);
    CHECK(o.optimal_w == 640);
    CHECK(o.optimal_h == 360);
    rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(DEVICE)});
    rig.Call("nvnDlssNetworkBuilderSetTextureIDs", {Rig::I(BUILDER), Rig::I(POOL), Rig::I(IDS)});
    rig.Call("nvnDlssNetworkBuilderSetFlags", {Rig::I(BUILDER), Rig::I(FLAGS_STAR_FOX)});
    rig.Call("nvnDlssNetworkBuilderSetMaxEvaluationEntries", {Rig::I(BUILDER), Rig::I(2)});
    const u64 size = StorageSize(rig);
    REQUIRE(size != 0);
    REQUIRE(size % 0x1000 == 0);
    REQUIRE(size <= DLSS_STORAGE_SIZE);
    rig.Call("nvnDlssNetworkBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(DLSS_STORAGE), Rig::I(size)});
    CHECK(Initialize(rig) == 1);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffCurrent) == 1280);
    CHECK(rig.memory.Get<u32>(NETWORK + NOffPreset) == 4);
    CHECK(rig.memory.Get<s32>(NETWORK + NOffOptimal) == 640);
}

// ---- NVNdlssParameters and nvnCommandBufferEvaluateDlss ----------------------------------------------

namespace {

constexpr u64 PARAMS = Rig::SCRATCH + 0x2000;
constexpr u64 MEM_POOL = Rig::Obj(30);
constexpr u64 QUEUE = Rig::Obj(31);
constexpr u64 CMD = Rig::Obj(32);
constexpr u64 TEX_INPUT = Rig::Obj(40);
constexpr u64 TEX_OUTPUT = Rig::Obj(41);
constexpr u64 TEX_MOTION = Rig::Obj(42);
constexpr u64 TEX_DEPTH = Rig::Obj(43);
constexpr u64 TEX_BIAS = Rig::Obj(44);
constexpr u64 TEX_EXPOSURE = Rig::Obj(45);
constexpr u64 NOT_A_TEXTURE = Rig::Obj(60); // zeroed: no NVNtexture tag
constexpr u64 VIEW = Rig::Obj(50);
constexpr u64 HANDLE_ARRAY = Rig::SCRATCH + 0x7000;

// NVNdlssParameters offsets (OoT sdk+0x65ad60..).
constexpr u64 POffInput = 0x00;
constexpr u64 POffOutput = 0x10;
constexpr u64 POffMotion = 0x20;
constexpr u64 POffDepth = 0x30;
constexpr u64 POffBias = 0x40;
constexpr u64 POffJitter = 0x50;
constexpr u64 POffScale = 0x58;
constexpr u64 POffPreExposure = 0x60;
constexpr u64 POffExposure = 0x68;
constexpr u64 POffReset = 0x78;
constexpr u64 POffIndex = 0x7c;
constexpr u64 POffSize = 0x80;
constexpr u64 POffInputBase = 0x88;
constexpr u64 POffDepthBase = 0x90;
constexpr u64 POffMotionBase = 0x98;
constexpr u64 POffBiasBase = 0xa0;
constexpr u64 POffOutputBase = 0xa8;
constexpr u64 POffOverlayDebugLayer = 0xb0;
constexpr u64 POffOverlayFullScreen = 0xb4;
constexpr u64 POffOverlayShowNans = 0xb8;
constexpr u64 POffOverlayJitterDebug = 0xbc;
constexpr u64 PEnd = 0xc0;

// Textures of the evaluation: 64x32, the input 48x24.
constexpr u32 TEX_W = 64;
constexpr u32 TEX_H = 32;
constexpr u32 INPUT_W = 48;
constexpr u32 INPUT_H = 24;

using ParamBytes = std::array<u8, PEnd>;

ParamBytes ReadParams(Rig& rig) {
    ParamBytes bytes{};
    REQUIRE(rig.memory.Read(PARAMS, bytes.data(), bytes.size()));
    return bytes;
}

void FillParams(Rig& rig, u8 value) {
    std::array<u8, 0x100> bytes;
    bytes.fill(value);
    rig.memory.Put(PARAMS, bytes);
}

VC::TextureDesc MakeTexture(u32 width, u32 height) {
    VC::TextureDesc desc{};
    desc.target = 1;
    desc.format = 0x25;
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.levels = 1;
    return desc;
}

std::vector<VC::Op> Ops(const std::vector<u64>& list) {
    std::vector<VC::Op> ops;
    VC::RecordReader reader{list.data(), list.size()};
    while (const VC::RecordHeader* header = reader.Next()) {
        ops.push_back(VC::RecordOp(header));
    }
    return ops;
}

const VC::EvaluateDlssPayload* Evaluation(const std::vector<u64>& list, size_t nth = 0) {
    VC::RecordReader reader{list.data(), list.size()};
    while (const VC::RecordHeader* header = reader.Next()) {
        if (VC::RecordOp(header) == VC::Op::EvaluateDlss && nth-- == 0) {
            // A record too short for the payload is not an evaluation.
            if (VC::RecordPayloadSize(header) < sizeof(VC::EvaluateDlssPayload)) {
                return nullptr;
            }
            return VC::RecordPayload<VC::EvaluateDlssPayload>(header);
        }
    }
    return nullptr;
}

/// A rig with a pool, a queue, a command buffer and the six textures of an evaluation.
struct Fx : DlssRig {
    Fx() {
        REQUIRE(recorder.MemoryPoolInitialize(MEM_POOL, STORAGE, STORAGE_SIZE, 0));
        REQUIRE(recorder.QueueInitialize(QUEUE));
        REQUIRE((Call("nvnCommandBufferInitialize", {I(CMD), I(0)}) & 1) == 1);
        u64 offset = 0x10000;
        for (const u64 texture :
             {TEX_INPUT, TEX_OUTPUT, TEX_MOTION, TEX_DEPTH, TEX_BIAS, TEX_EXPOSURE}) {
            REQUIRE(recorder.TextureInitialize(texture, MakeTexture(TEX_W, TEX_H), MEM_POOL, offset));
            offset += 0x10000;
        }
    }

    u64 Serial(u64 object) {
        return memory.Get<VC::GuestTag>(object).serial;
    }

    /// Records `commands` into the command buffer and returns the submitted list.
    std::vector<u64> Run(const std::function<void()>& commands) {
        Call("nvnCommandBufferAddCommandMemory", {I(CMD), I(MEM_POOL), I(0), I(0x2000)});
        Call("nvnCommandBufferBeginRecording", {I(CMD)});
        commands();
        const u64 handle = Call("nvnCommandBufferEndRecording", {I(CMD)});
        memory.Put<u64>(HANDLE_ARRAY, handle);
        const size_t before = sink.submissions.size();
        Call("nvnQueueSubmitCommands", {I(QUEUE), I(1), I(HANDLE_ARRAY)});
        REQUIRE(sink.submissions.size() == before + 1);
        REQUIRE(sink.submissions.back().lists.size() == 1);
        return sink.submissions.back().lists[0];
    }

    /// One evaluation recorded alone; REQUIREs it was recorded.
    VC::EvaluateDlssPayload EvaluateOne() {
        const auto list = Run([&] { Evaluate(); });
        REQUIRE(Ops(list) == std::vector<VC::Op>{VC::Op::EvaluateDlss});
        return *Evaluation(list);
    }

    void Evaluate(u64 network = NETWORK, u64 params = PARAMS) {
        Call("nvnCommandBufferEvaluateDlss", {I(CMD), I(network), I(params)});
    }

    /// The game's network: 1280x720 MaxQuality, preset 5, `flags`, initialized.
    void InitializeNetwork(u32 flags = FLAGS_STAR_FOX, u64 network = NETWORK) {
        BuildComplete(*this, flags);
        REQUIRE(Initialize(*this, network) == 1);
    }

    /// What the game's wrapper does through the NVN functions (48x24 of the input).
    void SetParametersThroughNvn() {
        Call("nvnDlssParametersSetDefaults", {I(PARAMS)});
        Call("nvnDlssParametersSetDepth", {I(PARAMS), I(TEX_DEPTH), I(0)});
        Call("nvnDlssParametersSetOutputColor", {I(PARAMS), I(TEX_OUTPUT), I(0)});
        Call("nvnDlssParametersSetInputColor", {I(PARAMS), I(TEX_INPUT), I(0)});
        Call("nvnDlssParametersSetMotionVectors", {I(PARAMS), I(TEX_MOTION), I(0)});
        Call("nvnDlssParametersSetBiasCurrentColorMask", {I(PARAMS), I(TEX_BIAS), I(0)});
        Call("nvnDlssParametersSetDepthSubrectangleBase", {I(PARAMS), I(0), I(0)});
        Call("nvnDlssParametersSetMotionVectorSubrectangleBase", {I(PARAMS), I(0), I(0)});
        Call("nvnDlssParametersSetOutputSubrectangleBase", {I(PARAMS), I(0), I(0)});
        Call("nvnDlssParametersSetBiasCurrentColorSubrectangleBase", {I(PARAMS), I(0), I(0)});
        Call("nvnDlssParametersSetJitterOffset", {I(PARAMS), F(0.25f), F(-0.125f)});
        Call("nvnDlssParametersSetInputSubrectangleSize", {I(PARAMS), I(INPUT_W), I(INPUT_H)});
        Call("nvnDlssParametersSetInputColorSubrectangleBase", {I(PARAMS), I(0), I(0)});
        Call("nvnDlssParametersSetHistoryReset", {I(PARAMS), I(1)});
        Call("nvnDlssParametersSetMotionVectorScale", {I(PARAMS), F(-24.0f), F(12.0f)});
        Call("nvnDlssParametersSetEvaluationIndex", {I(PARAMS), I(1)});
    }

    /// The same parameters through direct stores (after the same SetDefaults).
    void SetParametersDirectly() {
        Call("nvnDlssParametersSetDefaults", {I(PARAMS)});
        const auto texture = [&](u64 offset, u64 object) {
            memory.Put<u64>(PARAMS + offset, object);
            memory.Put<u64>(PARAMS + offset + 8, 0);
        };
        texture(POffInput, TEX_INPUT);
        texture(POffOutput, TEX_OUTPUT);
        texture(POffMotion, TEX_MOTION);
        texture(POffDepth, TEX_DEPTH);
        texture(POffBias, TEX_BIAS);
        for (const u64 offset : {POffInputBase, POffDepthBase, POffMotionBase, POffBiasBase,
                                 POffOutputBase}) {
            memory.Put<u64>(PARAMS + offset, 0);
        }
        memory.Put<float>(PARAMS + POffJitter, 0.25f);
        memory.Put<float>(PARAMS + POffJitter + 4, -0.125f);
        memory.Put<s32>(PARAMS + POffSize, INPUT_W);
        memory.Put<s32>(PARAMS + POffSize + 4, INPUT_H);
        memory.Put<u8>(PARAMS + POffReset, 1);
        memory.Put<float>(PARAMS + POffScale, -24.0f);
        memory.Put<float>(PARAMS + POffScale + 4, 12.0f);
        memory.Put<s32>(PARAMS + POffIndex, 1);
    }

    /// Calls the setter `name`, which must change bytes of [begin, end) of the parameters only.
    void ChangesOnly(const char* name, std::initializer_list<Arg> args, u64 begin, u64 end) {
        INFO(name);
        FillParams(*this, 0x5A);
        const ParamBytes before = ReadParams(*this);
        Call(name, args);
        const ParamBytes after = ReadParams(*this);
        bool changed = false;
        for (u64 i = 0; i < PEnd; ++i) {
            if (i >= begin && i < end) {
                changed = changed || after[i] != before[i];
            } else {
                INFO("byte " << i);
                CHECK(after[i] == before[i]);
            }
        }
        CHECK(changed);
        CHECK(memory.Get<u8>(PARAMS + PEnd) == 0x5A);
    }
};

} // namespace

TEST_CASE("nvn hle dlss parameters: SetDefaults sets the SDK's 0xc0 bytes", "[nvn][dlss]") {
    Rig rig;
    FillParams(rig, 0xCC);
    rig.Call("nvnDlssParametersSetDefaults", {Rig::I(PARAMS)});
    const ParamBytes bytes = ReadParams(rig);
    for (u64 i = 0; i < POffOverlayDebugLayer; ++i) {
        const bool scale = i >= POffScale && i < POffScale + 8;
        const bool pre_exposure = i >= POffPreExposure && i < POffPreExposure + 4;
        INFO("byte " << i);
        if (!scale && !pre_exposure) {
            CHECK(bytes[i] == 0);
        }
    }
    // Every overlay switch is -1.
    for (u64 i = POffOverlayDebugLayer; i < PEnd; ++i) {
        INFO("byte " << i);
        CHECK(bytes[i] == 0xFF);
    }
    CHECK(rig.memory.Get<u8>(PARAMS + PEnd) == 0xCC);
    CHECK(rig.memory.Get<u8>(PARAMS + 0xFF) == 0xCC);

    CHECK(rig.memory.Get<float>(PARAMS + POffScale) == 1.0f);
    CHECK(rig.memory.Get<float>(PARAMS + POffScale + 4) == 1.0f);
    CHECK(rig.memory.Get<float>(PARAMS + POffPreExposure) == 1.0f);
    CHECK(rig.CallF("nvnDlssParametersGetPreExposure", {Rig::I(PARAMS)}) == 1.0f);
    CHECK(rig.Call("nvnDlssParametersGetHistoryReset", {Rig::I(PARAMS)}) == 0);
    CHECK(rig.Call("nvnDlssParametersGetEvaluationIndex", {Rig::I(PARAMS)}) == 0);
}

TEST_CASE("nvn hle dlss parameters: every setter writes its own field", "[nvn][dlss]") {
    Fx f;
    // Texture setters: (parameters, texture, view); the view is kept as given.
    f.ChangesOnly("nvnDlssParametersSetInputColor",
                  {Rig::I(PARAMS), Rig::I(TEX_INPUT), Rig::I(VIEW)}, POffInput, POffInput + 16);
    CHECK(f.memory.Get<u64>(PARAMS + POffInput) == TEX_INPUT);
    CHECK(f.memory.Get<u64>(PARAMS + POffInput + 8) == VIEW);
    f.ChangesOnly("nvnDlssParametersSetOutputColor",
                  {Rig::I(PARAMS), Rig::I(TEX_OUTPUT), Rig::I(0)}, POffOutput, POffOutput + 16);
    CHECK(f.memory.Get<u64>(PARAMS + POffOutput) == TEX_OUTPUT);
    CHECK(f.memory.Get<u64>(PARAMS + POffOutput + 8) == 0);
    f.ChangesOnly("nvnDlssParametersSetMotionVectors",
                  {Rig::I(PARAMS), Rig::I(TEX_MOTION), Rig::I(VIEW)}, POffMotion, POffMotion + 16);
    CHECK(f.memory.Get<u64>(PARAMS + POffMotion) == TEX_MOTION);
    CHECK(f.memory.Get<u64>(PARAMS + POffMotion + 8) == VIEW);
    f.ChangesOnly("nvnDlssParametersSetDepth", {Rig::I(PARAMS), Rig::I(TEX_DEPTH), Rig::I(VIEW)},
                  POffDepth, POffDepth + 16);
    CHECK(f.memory.Get<u64>(PARAMS + POffDepth) == TEX_DEPTH);
    CHECK(f.memory.Get<u64>(PARAMS + POffDepth + 8) == VIEW);
    f.ChangesOnly("nvnDlssParametersSetBiasCurrentColorMask",
                  {Rig::I(PARAMS), Rig::I(TEX_BIAS), Rig::I(VIEW)}, POffBias, POffBias + 16);
    CHECK(f.memory.Get<u64>(PARAMS + POffBias) == TEX_BIAS);
    CHECK(f.memory.Get<u64>(PARAMS + POffBias + 8) == VIEW);
    f.ChangesOnly("nvnDlssParametersSetExposureTexture",
                  {Rig::I(PARAMS), Rig::I(TEX_EXPOSURE), Rig::I(VIEW)}, POffExposure,
                  POffExposure + 16);
    CHECK(f.memory.Get<u64>(PARAMS + POffExposure) == TEX_EXPOSURE);
    CHECK(f.memory.Get<u64>(PARAMS + POffExposure + 8) == VIEW);

    // Integer pairs.
    f.ChangesOnly("nvnDlssParametersSetInputSubrectangleSize",
                  {Rig::I(PARAMS), Rig::I(1280), Rig::I(720)}, POffSize, POffSize + 8);
    CHECK(f.memory.Get<s32>(PARAMS + POffSize) == 1280);
    CHECK(f.memory.Get<s32>(PARAMS + POffSize + 4) == 720);
    // A negative int arrives with garbage in the register's upper half.
    f.ChangesOnly("nvnDlssParametersSetInputColorSubrectangleBase",
                  {Rig::I(PARAMS), Rig::I(0xDEADBEEF'FFFFFFF0), Rig::I(7)}, POffInputBase,
                  POffInputBase + 8);
    CHECK(f.memory.Get<s32>(PARAMS + POffInputBase) == -16);
    CHECK(f.memory.Get<s32>(PARAMS + POffInputBase + 4) == 7);
    f.ChangesOnly("nvnDlssParametersSetDepthSubrectangleBase",
                  {Rig::I(PARAMS), Rig::I(1), Rig::I(2)}, POffDepthBase, POffDepthBase + 8);
    CHECK(f.memory.Get<s32>(PARAMS + POffDepthBase + 4) == 2);
    f.ChangesOnly("nvnDlssParametersSetMotionVectorSubrectangleBase",
                  {Rig::I(PARAMS), Rig::I(3), Rig::I(4)}, POffMotionBase, POffMotionBase + 8);
    CHECK(f.memory.Get<s32>(PARAMS + POffMotionBase) == 3);
    f.ChangesOnly("nvnDlssParametersSetBiasCurrentColorSubrectangleBase",
                  {Rig::I(PARAMS), Rig::I(5), Rig::I(6)}, POffBiasBase, POffBiasBase + 8);
    CHECK(f.memory.Get<s32>(PARAMS + POffBiasBase + 4) == 6);
    f.ChangesOnly("nvnDlssParametersSetOutputSubrectangleBase",
                  {Rig::I(PARAMS), Rig::I(7), Rig::I(8)}, POffOutputBase, POffOutputBase + 8);
    CHECK(f.memory.Get<s32>(PARAMS + POffOutputBase) == 7);

    // Floats arrive in v0, v1.
    f.ChangesOnly("nvnDlssParametersSetJitterOffset",
                  {Rig::I(PARAMS), Rig::F(0.25f), Rig::F(-0.125f)}, POffJitter, POffJitter + 8);
    CHECK(f.memory.Get<float>(PARAMS + POffJitter) == 0.25f);
    CHECK(f.memory.Get<float>(PARAMS + POffJitter + 4) == -0.125f);
    f.ChangesOnly("nvnDlssParametersSetMotionVectorScale",
                  {Rig::I(PARAMS), Rig::F(-640.0f), Rig::F(360.0f)}, POffScale, POffScale + 8);
    CHECK(f.memory.Get<float>(PARAMS + POffScale) == -640.0f);
    CHECK(f.memory.Get<float>(PARAMS + POffScale + 4) == 360.0f);
    f.ChangesOnly("nvnDlssParametersSetPreExposure", {Rig::I(PARAMS), Rig::F(4.5f)},
                  POffPreExposure, POffPreExposure + 4);
    CHECK(f.memory.Get<float>(PARAMS + POffPreExposure) == 4.5f);
    f.ChangesOnly("nvnDlssParametersSetEvaluationIndex", {Rig::I(PARAMS), Rig::I(3)}, POffIndex,
                  POffIndex + 4);
    CHECK(f.memory.Get<s32>(PARAMS + POffIndex) == 3);

    // The reset is one byte: its neighbours stay.
    f.ChangesOnly("nvnDlssParametersSetHistoryReset", {Rig::I(PARAMS), Rig::I(1)}, POffReset,
                  POffReset + 1);
    CHECK(f.memory.Get<u8>(PARAMS + POffReset) == 1);

    // Each overlay switch is its own u32, the whole 32-bit argument.
    f.ChangesOnly("nvnDlssParametersSetOverlayDebugLayer",
                  {Rig::I(PARAMS), Rig::I(0xDEAD'BEEF'0000'0001ull)}, POffOverlayDebugLayer,
                  POffOverlayDebugLayer + 4);
    CHECK(f.memory.Get<u32>(PARAMS + POffOverlayDebugLayer) == 1);
    f.ChangesOnly("nvnDlssParametersSetOverlayFullScreen", {Rig::I(PARAMS), Rig::I(0x1234'5600)},
                  POffOverlayFullScreen, POffOverlayFullScreen + 4);
    CHECK(f.memory.Get<u32>(PARAMS + POffOverlayFullScreen) == 0x1234'5600u);
    f.ChangesOnly("nvnDlssParametersSetOverlayShowNans", {Rig::I(PARAMS), Rig::I(0)},
                  POffOverlayShowNans, POffOverlayShowNans + 4);
    CHECK(f.memory.Get<u32>(PARAMS + POffOverlayShowNans) == 0);
    f.ChangesOnly("nvnDlssParametersSetOverlayJitterDebug", {Rig::I(PARAMS), Rig::I(2)},
                  POffOverlayJitterDebug, POffOverlayJitterDebug + 4);
    CHECK(f.memory.Get<u32>(PARAMS + POffOverlayJitterDebug) == 2);
}

TEST_CASE("nvn hle dlss parameters: the history reset is an NVNboolean (its low byte)",
          "[nvn][dlss]") {
    Rig rig;
    rig.Call("nvnDlssParametersSetDefaults", {Rig::I(PARAMS)});
    rig.Call("nvnDlssParametersSetHistoryReset", {Rig::I(PARAMS), Rig::I(0xFFFFFF02)});
    CHECK(rig.memory.Get<u8>(PARAMS + POffReset) == 1);
    CHECK(rig.Call("nvnDlssParametersGetHistoryReset", {Rig::I(PARAMS)}) == 1);
    rig.Call("nvnDlssParametersSetHistoryReset", {Rig::I(PARAMS), Rig::I(0x100)});
    CHECK(rig.memory.Get<u8>(PARAMS + POffReset) == 0);
    CHECK(rig.Call("nvnDlssParametersGetHistoryReset", {Rig::I(PARAMS)}) == 0);
}

TEST_CASE("nvn hle dlss parameters: getters return what was stored, also by direct stores",
          "[nvn][dlss]") {
    Fx f;
    f.SetParametersDirectly();
    f.memory.Put<s32>(PARAMS + POffInputBase, 11);
    f.memory.Put<s32>(PARAMS + POffInputBase + 4, -12);
    f.memory.Put<s32>(PARAMS + POffDepthBase, 21);
    f.memory.Put<s32>(PARAMS + POffDepthBase + 4, 22);
    f.memory.Put<s32>(PARAMS + POffMotionBase, 31);
    f.memory.Put<s32>(PARAMS + POffMotionBase + 4, 32);
    f.memory.Put<s32>(PARAMS + POffBiasBase, 41);
    f.memory.Put<s32>(PARAMS + POffBiasBase + 4, 42);
    f.memory.Put<s32>(PARAMS + POffOutputBase, 51);
    f.memory.Put<s32>(PARAMS + POffOutputBase + 4, 52);
    f.memory.Put<float>(PARAMS + POffPreExposure, 3.5f);

    const auto pair = [&](const char* name, u64 first = OUT, u64 second = OUT + 8) {
        INFO(name);
        f.memory.Put<s32>(OUT, UNTOUCHED);
        f.memory.Put<s32>(OUT + 8, UNTOUCHED);
        f.Call(name, {Rig::I(PARAMS), Rig::I(first), Rig::I(second)});
    };
    pair("nvnDlssParametersGetInputSubrectangleSize");
    CHECK(f.memory.Get<s32>(OUT) == static_cast<s32>(INPUT_W));
    CHECK(f.memory.Get<s32>(OUT + 8) == static_cast<s32>(INPUT_H));
    pair("nvnDlssParametersGetInputColorSubrectangleBase");
    CHECK(f.memory.Get<s32>(OUT) == 11);
    CHECK(f.memory.Get<s32>(OUT + 8) == -12);
    pair("nvnDlssParametersGetDepthSubrectangleBase");
    CHECK(f.memory.Get<s32>(OUT) == 21);
    CHECK(f.memory.Get<s32>(OUT + 8) == 22);
    pair("nvnDlssParametersGetMotionVectorSubrectangleBase");
    CHECK(f.memory.Get<s32>(OUT) == 31);
    CHECK(f.memory.Get<s32>(OUT + 8) == 32);
    pair("nvnDlssParametersGetBiasCurrentColorSubrectangleBase");
    CHECK(f.memory.Get<s32>(OUT) == 41);
    CHECK(f.memory.Get<s32>(OUT + 8) == 42);
    pair("nvnDlssParametersGetOutputSubrectangleBase");
    CHECK(f.memory.Get<s32>(OUT) == 51);
    CHECK(f.memory.Get<s32>(OUT + 8) == 52);
    pair("nvnDlssParametersGetJitterOffset");
    CHECK(f.memory.Get<float>(OUT) == 0.25f);
    CHECK(f.memory.Get<float>(OUT + 8) == -0.125f);
    pair("nvnDlssParametersGetMotionVectorScale");
    CHECK(f.memory.Get<float>(OUT) == -24.0f);
    CHECK(f.memory.Get<float>(OUT + 8) == 12.0f);
    CHECK(f.Call("nvnDlssParametersGetHistoryReset", {Rig::I(PARAMS)}) == 1);
    CHECK(f.Call("nvnDlssParametersGetEvaluationIndex", {Rig::I(PARAMS)}) == 1);
    CHECK(f.CallF("nvnDlssParametersGetPreExposure", {Rig::I(PARAMS)}) == 3.5f);

    // Either output pointer may be NULL.
    pair("nvnDlssParametersGetInputSubrectangleSize", 0, OUT + 8);
    CHECK(f.memory.Get<s32>(OUT) == UNTOUCHED);
    CHECK(f.memory.Get<s32>(OUT + 8) == static_cast<s32>(INPUT_H));
    pair("nvnDlssParametersGetMotionVectorScale", OUT, 0);
    CHECK(f.memory.Get<float>(OUT) == -24.0f);
    CHECK(f.memory.Get<s32>(OUT + 8) == UNTOUCHED);
    f.Call("nvnDlssParametersGetJitterOffset", {Rig::I(PARAMS), Rig::I(0), Rig::I(0)});
}

TEST_CASE("nvn hle dlss evaluate: records the evaluation the game's parameters describe",
          "[nvn][dlss]") {
    Fx f;
    // HDR colour, manual exposure and output sub-rectangles: everything the game set counts.
    f.InitializeNetwork(FLAGS_HDR_MANUAL_EXPOSURE | VC::DLSS_FLAG_OUTPUT_SUBRECTS);
    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetPreExposure", {Rig::I(PARAMS), Rig::F(2.0f)});
    f.Call("nvnDlssParametersSetExposureTexture", {Rig::I(PARAMS), Rig::I(TEX_EXPOSURE), Rig::I(0)});
    f.Call("nvnDlssParametersSetInputColorSubrectangleBase", {Rig::I(PARAMS), Rig::I(16), Rig::I(8)});
    f.Call("nvnDlssParametersSetDepthSubrectangleBase", {Rig::I(PARAMS), Rig::I(1), Rig::I(2)});
    f.Call("nvnDlssParametersSetMotionVectorSubrectangleBase", {Rig::I(PARAMS), Rig::I(3), Rig::I(4)});
    f.Call("nvnDlssParametersSetBiasCurrentColorSubrectangleBase",
           {Rig::I(PARAMS), Rig::I(5), Rig::I(6)});
    f.Call("nvnDlssParametersSetOutputSubrectangleBase", {Rig::I(PARAMS), Rig::I(7), Rig::I(9)});

    const VC::EvaluateDlssPayload e = f.EvaluateOne();
    CHECK(e.input.texture.serial == f.Serial(TEX_INPUT));
    CHECK(e.output.texture.serial == f.Serial(TEX_OUTPUT));
    CHECK(e.motion_vectors.texture.serial == f.Serial(TEX_MOTION));
    CHECK(e.depth.texture.serial == f.Serial(TEX_DEPTH));
    CHECK(e.bias.texture.serial == f.Serial(TEX_BIAS));
    CHECK(e.exposure.texture.serial == f.Serial(TEX_EXPOSURE));
    CHECK(e.input.texture.serial != e.output.texture.serial);
    // The game gave no view: the whole texture, level 0, layer 0.
    CHECK(e.input.texture.level == 0);
    CHECK(e.input.texture.layer == 0);
    CHECK(e.input.view.set_levels == 0);

    CHECK(e.input_width == static_cast<s32>(INPUT_W));
    CHECK(e.input_height == static_cast<s32>(INPUT_H));
    // The output size is the network's current one, not a parameter.
    CHECK(e.output_width == 1280);
    CHECK(e.output_height == 720);
    CHECK(e.input_base.x == 16);
    CHECK(e.input_base.y == 8);
    CHECK(e.depth_base.x == 1);
    CHECK(e.depth_base.y == 2);
    CHECK(e.motion_vector_base.x == 3);
    CHECK(e.motion_vector_base.y == 4);
    CHECK(e.bias_base.x == 5);
    CHECK(e.bias_base.y == 6);
    CHECK(e.output_base.x == 7);
    CHECK(e.output_base.y == 9);
    CHECK(e.jitter[0] == 0.25f);
    CHECK(e.jitter[1] == -0.125f);
    CHECK(e.motion_scale[0] == -24.0f);
    CHECK(e.motion_scale[1] == 12.0f);
    CHECK(e.pre_exposure == 2.0f);
    CHECK(e.history_reset == 1);
    CHECK(e.flags == (FLAGS_HDR_MANUAL_EXPOSURE | VC::DLSS_FLAG_OUTPUT_SUBRECTS));
    CHECK(e.method == MAX_QUALITY);
    CHECK(e.preset == 5);
    CHECK(e.network == Handle(f));
    CHECK(e.network != 0);

    // The record is a snapshot: the game reuses both objects for the next evaluation.
    const auto two = f.Run([&] {
        f.Evaluate();
        f.Call("nvnDlssParametersSetInputSubrectangleSize", {Rig::I(PARAMS), Rig::I(32), Rig::I(16)});
        f.Call("nvnDlssParametersSetJitterOffset", {Rig::I(PARAMS), Rig::F(-0.5f), Rig::F(0.5f)});
        f.Call("nvnDlssParametersSetHistoryReset", {Rig::I(PARAMS), Rig::I(0)});
        REQUIRE(f.Call("nvnDlssNetworkResize", {Rig::I(NETWORK), Rig::I(960), Rig::I(540)}) == 1);
        f.Evaluate();
    });
    REQUIRE(Ops(two) == std::vector<VC::Op>(2, VC::Op::EvaluateDlss));
    const VC::EvaluateDlssPayload* first = Evaluation(two, 0);
    const VC::EvaluateDlssPayload* second = Evaluation(two, 1);
    CHECK(first->input_width == static_cast<s32>(INPUT_W));
    CHECK(first->output_width == 1280);
    CHECK(first->jitter[0] == 0.25f);
    CHECK(first->history_reset == 1);
    CHECK(second->input_width == 32);
    CHECK(second->input_height == 16);
    CHECK(second->output_width == 960);
    CHECK(second->output_height == 540);
    CHECK(second->jitter[0] == -0.5f);
    CHECK(second->jitter[1] == 0.5f);
    CHECK(second->history_reset == 0);
    CHECK(second->network == first->network);
}

TEST_CASE("nvn hle dlss evaluate: exposure and pre-exposure only for HDR without auto exposure",
          "[nvn][dlss]") {
    struct Row {
        u32 flags;
        bool exposure;
    };
    // (flags & 0x41) == 0: HDR colour (bit 0 clear) and no automatic exposure (bit 6 clear).
    constexpr std::array<Row, 5> rows{{
        {0x0a, true},
        {0x00, true},
        {FLAGS_STAR_FOX, false}, // auto exposure
        {0x01, false},           // LDR colour
        {0x41, false},
    }};
    for (const Row& row : rows) {
        INFO(row.flags);
        Fx f;
        f.InitializeNetwork(row.flags);
        f.SetParametersThroughNvn();
        f.Call("nvnDlssParametersSetPreExposure", {Rig::I(PARAMS), Rig::F(2.0f)});
        f.Call("nvnDlssParametersSetExposureTexture",
               {Rig::I(PARAMS), Rig::I(TEX_EXPOSURE), Rig::I(0)});
        const VC::EvaluateDlssPayload e = f.EvaluateOne();
        CHECK(e.flags == row.flags);
        if (row.exposure) {
            CHECK(e.exposure.texture.serial == f.Serial(TEX_EXPOSURE));
            CHECK(e.pre_exposure == 2.0f);
        } else {
            CHECK(e.exposure.texture.serial == 0);
            CHECK(e.pre_exposure == 1.0f);
        }
    }
}

TEST_CASE("nvn hle dlss evaluate: the output origin counts only with flag bit 28", "[nvn][dlss]") {
    Fx f;
    f.InitializeNetwork(FLAGS_STAR_FOX);
    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetOutputSubrectangleBase", {Rig::I(PARAMS), Rig::I(7), Rig::I(9)});
    f.Call("nvnDlssParametersSetInputColorSubrectangleBase", {Rig::I(PARAMS), Rig::I(4), Rig::I(2)});
    VC::EvaluateDlssPayload e = f.EvaluateOne();
    CHECK(e.output_base.x == 0);
    CHECK(e.output_base.y == 0);
    // The input side's origins are not gated.
    CHECK(e.input_base.x == 4);
    CHECK(e.input_base.y == 2);

    f.InitializeNetwork(FLAGS_STAR_FOX | VC::DLSS_FLAG_OUTPUT_SUBRECTS);
    e = f.EvaluateOne();
    CHECK(e.output_base.x == 7);
    CHECK(e.output_base.y == 9);
}

TEST_CASE("nvn hle dlss evaluate: an input size of 0x0 is the input texture's size", "[nvn][dlss]") {
    Fx f;
    f.InitializeNetwork();
    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetInputSubrectangleSize", {Rig::I(PARAMS), Rig::I(0), Rig::I(0)});
    VC::EvaluateDlssPayload e = f.EvaluateOne();
    CHECK(e.input_width == static_cast<s32>(TEX_W));
    CHECK(e.input_height == static_cast<s32>(TEX_H));

    // SetDefaults leaves 0x0 too.
    f.Call("nvnDlssParametersSetDefaults", {Rig::I(PARAMS)});
    f.Call("nvnDlssParametersSetInputColor", {Rig::I(PARAMS), Rig::I(TEX_INPUT), Rig::I(0)});
    f.Call("nvnDlssParametersSetOutputColor", {Rig::I(PARAMS), Rig::I(TEX_OUTPUT), Rig::I(0)});
    e = f.EvaluateOne();
    CHECK(e.input_width == static_cast<s32>(TEX_W));
    CHECK(e.input_height == static_cast<s32>(TEX_H));

    // One zero, or a negative size, is the game's: recorded as given (the replay judges it).
    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetInputSubrectangleSize", {Rig::I(PARAMS), Rig::I(0), Rig::I(24)});
    e = f.EvaluateOne();
    CHECK(e.input_width == 0);
    CHECK(e.input_height == 24);
    f.Call("nvnDlssParametersSetInputSubrectangleSize",
           {Rig::I(PARAMS), Rig::I(48), Rig::I(static_cast<u32>(-1))});
    e = f.EvaluateOne();
    CHECK(e.input_width == 48);
    CHECK(e.input_height == -1);
}

TEST_CASE("nvn hle dlss evaluate: each network is recorded with its own id", "[nvn][dlss]") {
    Fx f;
    f.InitializeNetwork(FLAGS_STAR_FOX, NETWORK);
    f.InitializeNetwork(FLAGS_STAR_FOX, NETWORK2);
    f.SetParametersThroughNvn();
    const auto list = f.Run([&] {
        f.Evaluate(NETWORK);
        f.Evaluate(NETWORK2);
    });
    REQUIRE(Ops(list) == std::vector<VC::Op>(2, VC::Op::EvaluateDlss));
    CHECK(Evaluation(list, 0)->network == Handle(f, NETWORK));
    CHECK(Evaluation(list, 1)->network == Handle(f, NETWORK2));
    CHECK(Evaluation(list, 0)->network != Evaluation(list, 1)->network);

    // Re-initializing is a new network for the backend.
    const u64 old = Handle(f, NETWORK);
    f.InitializeNetwork(FLAGS_STAR_FOX, NETWORK);
    const VC::EvaluateDlssPayload e = f.EvaluateOne();
    CHECK(e.network == Handle(f, NETWORK));
    CHECK(e.network != old);
}

TEST_CASE("nvn hle dlss evaluate: the setter path and direct stores record the same",
          "[nvn][dlss]") {
    Fx f;
    f.InitializeNetwork();
    f.SetParametersThroughNvn();
    const auto through_nvn = f.Run([&] { f.Evaluate(); });
    f.SetParametersDirectly();
    const auto direct = f.Run([&] { f.Evaluate(); });
    const VC::EvaluateDlssPayload* a = Evaluation(through_nvn);
    const VC::EvaluateDlssPayload* b = Evaluation(direct);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CHECK(std::memcmp(a, b, sizeof(VC::EvaluateDlssPayload)) == 0);
    CHECK(a->input_width == static_cast<s32>(INPUT_W));
    CHECK(a->history_reset == 1);
}

TEST_CASE("nvn hle dlss evaluate: a texture view selects the level and format", "[nvn][dlss]") {
    Fx f;
    f.InitializeNetwork();
    f.SetParametersThroughNvn();
    f.Call("nvnTextureViewSetDefaults", {Rig::I(VIEW)});
    f.Call("nvnTextureViewSetLevels", {Rig::I(VIEW), Rig::I(2), Rig::I(1)});
    f.Call("nvnTextureViewSetFormat", {Rig::I(VIEW), Rig::I(0x25)});
    f.Call("nvnDlssParametersSetInputColor", {Rig::I(PARAMS), Rig::I(TEX_INPUT), Rig::I(VIEW)});
    const VC::EvaluateDlssPayload e = f.EvaluateOne();
    CHECK(e.input.texture.serial == f.Serial(TEX_INPUT));
    CHECK(e.input.texture.level == 2);
    CHECK(e.input.view.set_levels == 1);
    CHECK(e.input.view.base_level == 2);
    CHECK(e.input.view.set_format == 1);
    CHECK(e.input.view.format == 0x25);
    CHECK(e.output.texture.level == 0);
    CHECK(e.output.view.set_levels == 0);
}

TEST_CASE("nvn hle dlss evaluate: the images that only help DLSS may be missing", "[nvn][dlss]") {
    Fx f;
    f.InitializeNetwork(FLAGS_HDR_MANUAL_EXPOSURE);
    f.SetParametersThroughNvn();
    // No bias mask and no exposure texture; a depth that is not a texture is logged by the lookup
    // and left out.
    f.Call("nvnDlssParametersSetBiasCurrentColorMask", {Rig::I(PARAMS), Rig::I(0), Rig::I(0)});
    f.Call("nvnDlssParametersSetDepth", {Rig::I(PARAMS), Rig::I(NOT_A_TEXTURE), Rig::I(0)});
    const u64 mismatches = f.recorder.Stats().tag_mismatch;
    const VC::EvaluateDlssPayload e = f.EvaluateOne();
    CHECK(e.input.texture.serial == f.Serial(TEX_INPUT));
    CHECK(e.output.texture.serial == f.Serial(TEX_OUTPUT));
    CHECK(e.motion_vectors.texture.serial == f.Serial(TEX_MOTION));
    CHECK(e.bias.texture.serial == 0);
    CHECK(e.exposure.texture.serial == 0);
    CHECK(e.depth.texture.serial == 0);
    CHECK(f.recorder.Stats().tag_mismatch == mismatches + 1);
}

TEST_CASE("nvn hle dlss evaluate: only a missing feature or a missing input/output is dropped",
          "[nvn][dlss]") {
    Fx f;
    const auto nothing = [&](const char* what) {
        INFO(what);
        const auto list = f.Run([&] { f.Evaluate(); });
        CHECK(Ops(list).empty());
    };

    f.SetParametersThroughNvn();
    nothing("network never initialized");

    f.InitializeNetwork();
    f.Call("nvnDlssNetworkFinalize", {Rig::I(NETWORK)});
    nothing("finalized network");

    f.InitializeNetwork();
    f.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0)});
    REQUIRE(Initialize(f) == 0);
    nothing("refused network");

    f.InitializeNetwork();
    f.EvaluateOne();

    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetInputColor", {Rig::I(PARAMS), Rig::I(0), Rig::I(0)});
    nothing("no input texture");

    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetOutputColor", {Rig::I(PARAMS), Rig::I(0), Rig::I(0)});
    nothing("no output texture");

    // Parameters or a network at an address the guest does not map, or NULL.
    f.SetParametersThroughNvn();
    {
        const auto list = f.Run([&] {
            f.Evaluate(NETWORK, 0);
            f.Evaluate(NETWORK, 0x5000'0000);
            f.Evaluate(0, PARAMS);
            f.Evaluate(0x5000'0000, PARAMS);
        });
        CHECK(Ops(list).empty());
    }

    // Textures the recorder does not know: the whole record is dropped and counted.
    f.SetParametersThroughNvn();
    f.Call("nvnDlssParametersSetInputColor", {Rig::I(PARAMS), Rig::I(NOT_A_TEXTURE), Rig::I(0)});
    {
        const u64 dropped = f.recorder.Stats().records_dropped;
        const auto list = f.Run([&] { f.Evaluate(); });
        CHECK(Ops(list).empty());
        CHECK(f.recorder.Stats().records_dropped == dropped + 1);
    }

    // Outside BeginRecording/EndRecording.
    f.SetParametersThroughNvn();
    {
        const u64 dropped = f.recorder.Stats().records_dropped;
        f.Evaluate();
        CHECK(f.recorder.Stats().records_dropped == dropped + 1);
    }
}

// ---- The layouts of the other SDK generations ---------------------------------------------------------

namespace {

/// One SDK generation's object sizes, as its own code gives them (sdk_dlss_layout.cpp).
struct SdkCase {
    const char* name;
    u32 builder;
    u32 network;
    u32 parameters;
};
constexpr std::array<SdkCase, 3> SDKS{{
    {"API 231.39/231.40 (OoT, Star Fox)", 0xc0, 0x258, 0xc0},
    {"API 231.30 (SDK 20.5)", 0xb8, 0x220, 0xc0},
    {"API 231.16", 0xc0, 0x218, 0xc8},
}};

Core::NvnHle::DlssLayout LayoutOf(const SdkCase& sdk) {
    return Core::NvnHle::DlssLayoutFor(
        {.builder = sdk.builder, .network = sdk.network, .parameters = sdk.parameters});
}

constexpr u64 SENTINEL = 0x5A5A'5A5A'5A5A'5A5Aull;

/// Bytes [from, to) of `va` all equal `value`.
bool AllBytes(Rig& rig, u64 va, u64 from, u64 to, u8 value) {
    for (u64 i = from; i < to; ++i) {
        if (rig.memory.Get<u8>(va + i) != value) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("nvn hle dlss sdk layouts: every body writes the SDK's fields and nothing past its "
          "objects",
          "[nvn][dlss]") {
    for (const SdkCase& sdk : SDKS) {
        INFO(sdk.name);
        DlssRig rig;
        const Core::NvnHle::DlssLayout layout = LayoutOf(sdk);
        REQUIRE(layout.network != nullptr);
        const Core::NvnHle::DlssNetworkLayout& l = *layout.network;
        rig.runtime.device.SetDlssLayout(layout);

        // Builder: SetDefaults clears up to the SDK's size, the word after it is the game's.
        std::array<u8, 0x100> builder_bytes;
        builder_bytes.fill(0xFF);
        rig.memory.Put(BUILDER, builder_bytes);
        rig.memory.Put<u64>(BUILDER + sdk.builder, SENTINEL);
        BuildComplete(rig);
        CHECK(rig.memory.Get<u64>(BUILDER + sdk.builder) == SENTINEL);
        if (sdk.builder == 0xc0) {
            CHECK(rig.memory.Get<u64>(BUILDER + OffReserved) == 0);
        }
        CHECK(rig.Call("nvnDlssNetworkBuilderGetMaxEvaluationEntries", {Rig::I(BUILDER)}) == 4);
        const u64 storage_size = StorageSize(rig);

        // Initialize: the SDK's fields, the rest of the game's 0x800 bytes untouched.
        FillNetwork(rig, NETWORK, 0xCC);
        REQUIRE(Initialize(rig) == 1);
        CHECK(AllBytes(rig, NETWORK, sdk.network, NETWORK_BYTES, 0xCC));
        const auto get32 = [&](u32 offset) { return rig.memory.Get<s32>(NETWORK + offset); };
        const auto get64 = [&](u32 offset) { return rig.memory.Get<u64>(NETWORK + offset); };
        CHECK(get64(0) == 0);
        CHECK(get64(Core::NvnHle::DlssNetworkDevice) == DEVICE);
        CHECK(get64(Core::NvnHle::DlssNetworkTexturePool) == POOL);
        CHECK(rig.memory.Get<IdArray>(NETWORK + Core::NvnHle::DlssNetworkTextureIds) ==
              IdPattern(3));
        CHECK(Handle(rig) != 0);
        CHECK(get64(l.storage_device) == DEVICE);
        CHECK(get64(l.storage) == DLSS_STORAGE);
        CHECK(get64(l.storage_size) == storage_size);
        CHECK(get32(l.max_evaluation_entries) == 4);
        CHECK(get32(l.method) == static_cast<s32>(MAX_QUALITY));
        CHECK(get32(l.ngx_flags) == 0x4b);
        CHECK(get32(l.quality) == static_cast<s32>(MAX_QUALITY));
        CHECK(get32(l.preset) == 5);
        CHECK(get32(l.optimal) == 853);
        CHECK(get32(l.optimal + 4) == 480);
        CHECK(get32(l.output) == 1280);
        CHECK(get32(l.output + 4) == 720);
        CHECK(get32(l.output_subrects) == 0);
        CHECK(get32(l.max_render) == 1280);
        CHECK(get32(l.min_render + 4) == 240);
        CHECK(get32(l.current_output) == 1280);
        CHECK(get32(l.current_output + 4) == 720);
        if (l.initial_output != 0) {
            CHECK(get32(l.initial_output) == 1280);
            CHECK(get32(l.initial_output + 4) == 720);
        }
        CHECK(get32(l.flags) == static_cast<s32>(FLAGS_STAR_FOX));

        // Resize: the SDKs that have one; the oldest has none and nothing changes.
        const u64 resized = rig.Call("nvnDlssNetworkResize",
                                     {Rig::I(NETWORK), Rig::I(960), Rig::I(540)}) &
                            0xFF;
        if (l.initial_output != 0) {
            CHECK(resized == 1);
            CHECK(get32(l.current_output) == 960);
            CHECK(get32(l.output + 4) == 540);
            CHECK(get32(l.optimal) == 640);
            CHECK(get32(l.initial_output) == 1280);
            CHECK(rig.Call("nvnDlssNetworkResize",
                           {Rig::I(NETWORK), Rig::I(1281), Rig::I(720)}) == 0);
        } else {
            CHECK(resized == 0);
            CHECK(get32(l.current_output) == 1280);
        }
        CHECK(AllBytes(rig, NETWORK, sdk.network, NETWORK_BYTES, 0xCC));

        // Finalize clears the handle only; a refused Initialize clears the SDK's bytes only.
        rig.Call("nvnDlssNetworkFinalize", {Rig::I(NETWORK)});
        CHECK(Handle(rig) == 0);
        CHECK(get64(l.storage) == DLSS_STORAGE);
        rig.Call("nvnDlssNetworkBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0)});
        FillNetwork(rig, NETWORK, 0xCC);
        CHECK(Initialize(rig) == 0);
        CHECK(AllBytes(rig, NETWORK, 0, sdk.network, 0x00));
        CHECK(AllBytes(rig, NETWORK, sdk.network, NETWORK_BYTES, 0xCC));

        // Parameters: the SDK's bytes, the 0xC8 ones with a zero last word.
        FillParams(rig, 0xCC);
        rig.Call("nvnDlssParametersSetDefaults", {Rig::I(PARAMS)});
        CHECK(rig.memory.Get<u32>(PARAMS + POffOverlayJitterDebug) == ~0u);
        CHECK(rig.memory.Get<float>(PARAMS + POffScale) == 1.0f);
        CHECK(AllBytes(rig, PARAMS, PEnd, sdk.parameters, 0x00));
        CHECK(AllBytes(rig, PARAMS, sdk.parameters, 0x100, 0xCC));
    }
}

TEST_CASE("nvn hle dlss sdk layouts: evaluate reads the network in the SDK's layout",
          "[nvn][dlss]") {
    for (const SdkCase& sdk : SDKS) {
        INFO(sdk.name);
        Fx f;
        const Core::NvnHle::DlssLayout layout = LayoutOf(sdk);
        f.runtime.device.SetDlssLayout(layout);
        f.InitializeNetwork(FLAGS_HDR_MANUAL_EXPOSURE | VC::DLSS_FLAG_OUTPUT_SUBRECTS);
        f.SetParametersThroughNvn();
        f.Call("nvnDlssParametersSetOutputSubrectangleBase", {Rig::I(PARAMS), Rig::I(7), Rig::I(9)});
        const VC::EvaluateDlssPayload e = f.EvaluateOne();
        CHECK(e.output_width == 1280);
        CHECK(e.output_height == 720);
        CHECK(e.flags == (FLAGS_HDR_MANUAL_EXPOSURE | VC::DLSS_FLAG_OUTPUT_SUBRECTS));
        CHECK(e.output_base.x == 7);
        CHECK(e.method == MAX_QUALITY);
        CHECK(e.preset == 5);
        CHECK(e.network == Handle(f));
        CHECK(e.input_width == static_cast<s32>(INPUT_W));
        if (layout.network->initial_output != 0) {
            REQUIRE(f.Call("nvnDlssNetworkResize", {Rig::I(NETWORK), Rig::I(960), Rig::I(540)}) ==
                    1);
            const VC::EvaluateDlssPayload resized = f.EvaluateOne();
            CHECK(resized.output_width == 960);
            CHECK(resized.output_height == 540);
        }
    }
}

TEST_CASE("nvn hle dlss sdk layouts: an SDK network of no known layout is never written",
          "[nvn][dlss]") {
    Fx f;
    f.runtime.device.SetDlssLayout(
        Core::NvnHle::DlssLayoutFor({.builder = 0xc0, .network = 0x230, .parameters = 0xc0}));
    BuildComplete(f);
    FillNetwork(f, NETWORK, 0xCC);
    CHECK(Initialize(f) == 0);
    CHECK(AllBytes(f, NETWORK, 0, NETWORK_BYTES, 0xCC));
    CHECK(f.Call("nvnDlssNetworkResize", {Rig::I(NETWORK), Rig::I(960), Rig::I(540)}) == 0);
    CHECK(AllBytes(f, NETWORK, 0, NETWORK_BYTES, 0xCC));
    f.SetParametersThroughNvn();
    const auto list = f.Run([&] { f.Evaluate(); });
    CHECK(Ops(list).empty());
}
