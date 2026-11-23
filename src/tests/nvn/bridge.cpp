// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_funcs.h"
#include "core/core.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/nvn/aapcs64.h"
#include "core/hle/nvn/bridge.h"
#include "core/hle/nvn/config.h"
#include "core/hle/nvn/dynsym.h"
#include "core/hle/nvn/thunk_encoding.h"
#include "core/loader/nso.h"
#include "tests/temp_path.h"

namespace {

using Core::NvnHle::Bridge;
using Core::NvnHle::Config;
using Core::NvnHle::SlotKind;
namespace Aapcs64 = Core::NvnHle::Aapcs64;
namespace Thunk = Core::NvnHle::Thunk;
using Context = Kernel::Svc::ThreadContext;

// ---------------------------------------------------------------------------------------------
// AAPCS64 helpers

constexpr u64 StackBase = 0x7000;

struct FakeStack {
    std::vector<u8> bytes = std::vector<u8>(0x400);

    template <typename T>
    void Put(size_t offset, T value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }
    Aapcs64::MemoryReader Reader() {
        return [this](u64 address, void* destination, size_t size) {
            if (address < StackBase || address - StackBase + size > bytes.size()) {
                return false;
            }
            std::memcpy(destination, bytes.data() + (address - StackBase), size);
            return true;
        };
    }
    Aapcs64::MemoryWriter Writer() {
        return [this](u64 address, const void* source, size_t size) {
            if (address < StackBase || address - StackBase + size > bytes.size()) {
                return false;
            }
            std::memcpy(bytes.data() + (address - StackBase), source, size);
            return true;
        };
    }
};

Context MakeContext() {
    Context ctx{};
    ctx.sp = StackBase;
    for (size_t i = 0; i < 8; ++i) {
        ctx.r[i] = 100 + i;
        ctx.v[i] = {std::bit_cast<u64>(1.5 + static_cast<double>(i)), 0};
    }
    return ctx;
}

TEST_CASE("NVN bridge: AAPCS64 integers spill to the stack after x7", "[nvn][bridge]") {
    FakeStack stack;
    stack.Put<u64>(0, 1000);
    stack.Put<u64>(8, 1001);
    stack.Put<u64>(16, 1002);
    auto ctx = MakeContext();
    Aapcs64::ArgCursor args{ctx, stack.Reader()};
    for (u64 i = 0; i < 8; ++i) {
        REQUIRE(args.NextU64() == 100 + i);
    }
    REQUIRE(args.NextU64() == 1000);
    REQUIRE(args.NextU64() == 1001);
    REQUIRE(args.NextU64() == 1002);
    REQUIRE(args.StackBytesUsed() == 24);
}

TEST_CASE("NVN bridge: AAPCS64 doubles spill to the stack after v7", "[nvn][bridge]") {
    FakeStack stack;
    stack.Put<double>(0, 100.25);
    stack.Put<double>(8, 101.25);
    auto ctx = MakeContext();
    Aapcs64::ArgCursor args{ctx, stack.Reader()};
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(args.NextF64() == 1.5 + static_cast<double>(i));
    }
    REQUIRE(args.NextF64() == 100.25);
    REQUIRE(args.NextF64() == 101.25);
}

TEST_CASE("NVN bridge: AAPCS64 int and float register classes are independent",
          "[nvn][bridge]") {
    FakeStack stack;
    stack.Put<u64>(0, 7777);
    auto ctx = MakeContext();
    Aapcs64::ArgCursor args{ctx, stack.Reader()};
    // 10 ints interleaved with 2 floats: floats still use v0/v1, ints spill after x7.
    for (u64 i = 0; i < 8; ++i) {
        REQUIRE(args.NextU64() == 100 + i);
    }
    REQUIRE(args.NextF64() == 1.5);
    REQUIRE(args.NextU64() == 7777);
    REQUIRE(args.NextF64() == 2.5);
    REQUIRE(args.UsedFloatRegisters() == 2);
}

TEST_CASE("NVN bridge: AAPCS64 masks sub-32-bit integers", "[nvn][bridge]") {
    FakeStack stack;
    auto ctx = MakeContext();
    ctx.r[0] = 0xDEADBEEF'00000007ULL;
    ctx.r[1] = 0xFFFFFFFF'FFFF1234ULL;
    ctx.r[2] = 0xAAAAAAAA'AAAAAA81ULL;
    ctx.r[3] = 0xCCCCCCCC'CCCCCC00ULL; // NVNboolean false with dirty upper bits
    ctx.r[4] = 0x12345678'FFFFFFFEULL;
    Aapcs64::ArgCursor args{ctx, stack.Reader()};
    REQUIRE(args.NextU32() == 7u);
    REQUIRE(args.NextU16() == 0x1234u);
    REQUIRE(args.NextS8() == static_cast<s8>(0x81));
    REQUIRE_FALSE(args.NextBool());
    REQUIRE(args.NextS32() == -2);
}

TEST_CASE("NVN bridge: AAPCS64 16-byte arguments use even register pairs", "[nvn][bridge]") {
    FakeStack stack;
    auto ctx = MakeContext();
    Aapcs64::ArgCursor args{ctx, stack.Reader()};
    REQUIRE(args.NextU64() == 100); // x0
    const u128 wide = args.NextI128(); // skips x1, takes x2:x3
    REQUIRE(wide[0] == 102);
    REQUIRE(wide[1] == 103);
    REQUIRE(args.NextU64() == 104); // x4
    // 16-aligned composite after an odd count lands on an even pair again.
    std::array<u64, 2> composite{};
    REQUIRE(args.NextComposite(composite.data(), 16, 16)); // ngrn 5 -> 6, x6:x7
    REQUIRE(composite[0] == 106);
    REQUIRE(composite[1] == 107);
}

TEST_CASE("NVN bridge: AAPCS64 16-byte stack arguments are 16-aligned", "[nvn][bridge]") {
    FakeStack stack;
    stack.Put<u64>(0, 1);          // 8-byte stack argument
    stack.Put<u64>(16, 0x1111);    // 16-aligned slot
    stack.Put<u64>(24, 0x2222);
    auto ctx = MakeContext();
    Aapcs64::ArgCursor args{ctx, stack.Reader()};
    for (int i = 0; i < 8; ++i) {
        (void)args.NextU64();
    }
    REQUIRE(args.NextU64() == 1); // stack offset 0
    const u128 wide = args.NextI128(); // offset 8 rounds to 16
    REQUIRE(wide[0] == 0x1111);
    REQUIRE(wide[1] == 0x2222);
}

TEST_CASE("NVN bridge: AAPCS64 composites, HFAs and indirect arguments", "[nvn][bridge]") {
    FakeStack stack;
    auto ctx = MakeContext();
    SECTION("small composite takes consecutive x registers") {
        Aapcs64::ArgCursor args{ctx, stack.Reader()};
        (void)args.NextU64();
        struct Pair {
            u64 a;
            u32 b;
        } pair{};
        REQUIRE(args.NextComposite(&pair, 12));
        REQUIRE(pair.a == 101);
        REQUIRE(pair.b == 102);
        REQUIRE(args.UsedIntegerRegisters() == 3);
    }
    SECTION("composite that does not fit goes to the stack and exhausts x registers") {
        stack.Put<u64>(0, 0xAAAA);
        stack.Put<u64>(8, 0xBBBB);
        Aapcs64::ArgCursor args{ctx, stack.Reader()};
        for (int i = 0; i < 7; ++i) {
            (void)args.NextU64();
        }
        std::array<u64, 2> composite{};
        REQUIRE(args.NextComposite(composite.data(), 16));
        REQUIRE(composite[0] == 0xAAAA);
        REQUIRE(composite[1] == 0xBBBB);
        REQUIRE(args.UsedIntegerRegisters() == 8);
        // x7 stays unused: the next integer argument also goes to the stack.
        stack.Put<u64>(16, 0xCCCC);
        REQUIRE(args.NextU64() == 0xCCCC);
    }
    SECTION("composite larger than 16 bytes is passed by pointer") {
        std::array<u32, 6> original{1, 2, 3, 4, 5, 6};
        stack.Put(0x100, original);
        ctx.r[1] = StackBase + 0x100;
        Aapcs64::ArgCursor args{ctx, stack.Reader()};
        (void)args.NextU64();
        std::array<u32, 6> copy{};
        REQUIRE(args.NextComposite(copy.data(), sizeof(copy)));
        REQUIRE(copy == original);
        REQUIRE(args.UsedIntegerRegisters() == 2);
    }
    SECTION("HFA of four floats takes consecutive v registers") {
        for (size_t i = 0; i < 8; ++i) {
            ctx.v[i] = {std::bit_cast<u32>(static_cast<float>(i) + 0.5f), 0};
        }
        Aapcs64::ArgCursor args{ctx, stack.Reader()};
        (void)args.NextF32(); // v0
        std::array<float, 4> hfa{};
        REQUIRE(args.NextHfa(hfa.data(), sizeof(float), 4));
        REQUIRE(hfa == std::array<float, 4>{1.5f, 2.5f, 3.5f, 4.5f});
        REQUIRE(args.UsedFloatRegisters() == 5);
        // Another HFA of 4 does not fit in v5-v7: it goes to the stack, v registers exhausted.
        stack.Put<std::array<float, 4>>(0, {9.f, 8.f, 7.f, 6.f});
        std::array<float, 4> spilled{};
        REQUIRE(args.NextHfa(spilled.data(), sizeof(float), 4));
        REQUIRE(spilled == std::array<float, 4>{9.f, 8.f, 7.f, 6.f});
        REQUIRE(args.UsedFloatRegisters() == 8);
    }
    SECTION("128-bit vector takes a whole v register") {
        ctx.v[2] = {0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL};
        Aapcs64::ArgCursor args{ctx, stack.Reader()};
        (void)args.NextF32();
        (void)args.NextF32();
        const u128 vector = args.NextVector128();
        REQUIRE(vector[0] == 0x1122334455667788ULL);
        REQUIRE(vector[1] == 0x99AABBCCDDEEFF00ULL);
    }
}

TEST_CASE("NVN bridge: AAPCS64 return values", "[nvn][bridge]") {
    FakeStack stack;
    Context ctx{};
    Aapcs64::ReturnWriter writer{ctx, stack.Writer()};
    SECTION("integers are zero-extended") {
        ctx.r[0] = ~0ULL;
        writer.SetU32(0x80000001u);
        REQUIRE(ctx.r[0] == 0x80000001ULL);
        writer.SetBool(true);
        REQUIRE(ctx.r[0] == 1);
        writer.SetU64(0x123456789ABCDEF0ULL);
        REQUIRE(ctx.r[0] == 0x123456789ABCDEF0ULL);
    }
    SECTION("16-byte integer returns x0:x1") {
        writer.SetI128({5, 6});
        REQUIRE(ctx.r[0] == 5);
        REQUIRE(ctx.r[1] == 6);
    }
    SECTION("floats go to the low bits of v0 with zeroed upper bits") {
        ctx.v[0] = {~0ULL, ~0ULL};
        writer.SetF32(2.5f);
        REQUIRE(ctx.v[0][0] == std::bit_cast<u32>(2.5f));
        REQUIRE(ctx.v[0][1] == 0);
        ctx.v[0] = {~0ULL, ~0ULL};
        writer.SetF64(-3.25);
        REQUIRE(ctx.v[0][0] == std::bit_cast<u64>(-3.25));
        REQUIRE(ctx.v[0][1] == 0);
    }
    SECTION("128-bit vector returns in v0") {
        writer.SetVector128({7, 8});
        REQUIRE(ctx.v[0][0] == 7);
        REQUIRE(ctx.v[0][1] == 8);
    }
    SECTION("HFA returns in v0..v3") {
        const std::array<float, 3> hfa{1.f, 2.f, 3.f};
        writer.SetHfa(hfa.data(), sizeof(float), 3);
        REQUIRE(ctx.v[0][0] == std::bit_cast<u32>(1.f));
        REQUIRE(ctx.v[1][0] == std::bit_cast<u32>(2.f));
        REQUIRE(ctx.v[2][0] == std::bit_cast<u32>(3.f));
    }
    SECTION("small composite returns in x0:x1") {
        struct Small {
            u64 a;
            u32 b;
        } value{0x1111, 0x2222};
        REQUIRE(writer.SetComposite(&value, 12));
        REQUIRE(ctx.r[0] == 0x1111);
        REQUIRE(ctx.r[1] == 0x2222);
    }
    SECTION("large composite returns through the x8 buffer") {
        std::array<u32, 8> value{1, 2, 3, 4, 5, 6, 7, 8};
        ctx.r[8] = StackBase + 0x80;
        REQUIRE(writer.SetComposite(value.data(), sizeof(value)));
        REQUIRE(ctx.r[0] == StackBase + 0x80);
        REQUIRE(std::memcmp(stack.bytes.data() + 0x80, value.data(), sizeof(value)) == 0);
    }
}

// ---------------------------------------------------------------------------------------------
// Thunk encoding

TEST_CASE("NVN bridge: thunk slot words", "[nvn][bridge]") {
    static_assert(Thunk::CodeBytes == 0x8000);
    static_assert(Thunk::RegionBytes == 0xC000);
    const auto first = Thunk::SlotWords(0);
    REQUIRE(first[0] == 0xD4100001u); // svc #0x8000
    REQUIRE(first[1] == 0x5803FFF0u); // ldr x16, +0x7ffc
    REQUIRE(first[2] == 0xD61F0200u); // br x16
    REQUIRE(first[3] == 0xD4200000u); // brk #0

    const auto last = Thunk::SlotWords(Thunk::ReturnSlot);
    REQUIRE(last[0] == (0xD4000001u | ((0x8000u + 2047u) << 5)));

    // Every ldr must land on that slot's own table entry.
    for (u32 slot = 0; slot < Thunk::SlotCount; ++slot) {
        const auto words = Thunk::SlotWords(slot);
        REQUIRE(words[0] == Thunk::EncodeSvc(Thunk::SvcBase + slot));
        u64 literal = 0;
        REQUIRE(Thunk::DecodeLdrX16Literal(words[1], Thunk::SlotOffset(slot) + 4, literal));
        REQUIRE(literal == Thunk::TableEntryOffset(slot));
    }
    u64 literal = 0;
    REQUIRE_FALSE(Thunk::DecodeLdrX16Literal(Thunk::BrX16, 0, literal));
}

TEST_CASE("NVN bridge: thunk region code is written into the image", "[nvn][bridge]") {
    std::vector<u8> region(Thunk::RegionBytes, 0xEE);
    Thunk::WriteSlotCode(region);
    for (u32 slot : {0u, 1u, 1000u, 2047u}) {
        std::array<u32, 4> words{};
        std::memcpy(words.data(), region.data() + Thunk::SlotOffset(slot), sizeof(words));
        REQUIRE(words == Thunk::SlotWords(slot));
    }
    // The table part is left alone.
    REQUIRE(region[Thunk::TableOffset] == 0xEE);
}

// ---------------------------------------------------------------------------------------------
// Synthetic sdk image

struct SyntheticSymbol {
    std::string name;
    u64 value;
};

constexpr u32 SyntheticRegionOffset = 0x2000;

/// A minimal NSO-like image: MOD0 at 0x100, dynamic at 0x140, dynsym at 0x400, then the thunk
/// region at SyntheticRegionOffset.
std::vector<u8> MakeSdkImage(const std::vector<SyntheticSymbol>& symbols) {
    std::vector<u8> image(SyntheticRegionOffset + Thunk::RegionBytes, 0);
    const auto put = [&](size_t offset, const auto& value) {
        std::memcpy(image.data() + offset, &value, sizeof(value));
    };
    put(4, u32{0x100});
    put(0x100, u32{0x30444F4D});
    put(0x104, u32{0x40}); // dynamic offset relative to MOD0
    constexpr u64 symtab = 0x400;
    const u64 strtab = symtab + 24 * (symbols.size() + 1);
    size_t dyn = 0x140;
    const auto put_dyn = [&](s64 tag, u64 value) {
        put(dyn, tag);
        put(dyn + 8, value);
        dyn += 16;
    };
    put_dyn(6, symtab);
    put_dyn(5, strtab);
    put_dyn(11, 24);
    put_dyn(0, 0);

    size_t name_offset = 1;
    for (size_t i = 0; i < symbols.size(); ++i) {
        const size_t entry = symtab + 24 * (i + 1);
        put(entry, static_cast<u32>(name_offset));
        image[entry + 4] = 0x12;  // global func
        put(entry + 6, u16{1});   // defined
        put(entry + 8, symbols[i].value);
        put(entry + 16, u64{0x20});
        std::memcpy(image.data() + strtab + name_offset, symbols[i].name.c_str(),
                    symbols[i].name.size() + 1);
        name_offset += symbols[i].name.size() + 1;
    }
    return image;
}

u64 ReadSymbolValue(const std::vector<u8>& image, size_t index) {
    u64 value{};
    std::memcpy(&value, image.data() + 0x400 + 24 * (index + 1) + 8, sizeof(value));
    return value;
}

bool WantsFooPrefix(std::string_view name) {
    return name.starts_with("foo");
}

TEST_CASE("NVN bridge: dynsym probe, find and rewrite", "[nvn][bridge]") {
    auto image = MakeSdkImage({{"foo_a", 0x1000},
                               {"bar", 0x1100},
                               {"foo_undefined", 0},
                               {"foo_b", 0x1200}});
    Core::NvnHle::DynsymLayout layout;
    REQUIRE(Core::NvnHle::ProbeDynsym(image, layout));
    REQUIRE(layout.symtab == 0x400);
    REQUIRE(layout.entry_size == 24);

    const auto hits = Core::NvnHle::FindDynsymHits(image, &WantsFooPrefix);
    REQUIRE(hits.size() == 2); // foo_undefined has value 0
    std::map<std::string, u64> by_name;
    for (const auto& hit : hits) {
        by_name[hit.name] = hit.value;
    }
    REQUIRE(by_name.at("foo_a") == 0x1000);
    REQUIRE(by_name.at("foo_b") == 0x1200);

    const std::map<std::string, u64, std::less<>> rewrites{
        {"foo_a", 0x2010}, {"foo_undefined", 0x2020}, {"missing", 0x2030}};
    REQUIRE(Core::NvnHle::RewriteSymbolValues(image, rewrites) == 1);
    REQUIRE(ReadSymbolValue(image, 0) == 0x2010);
    REQUIRE(ReadSymbolValue(image, 1) == 0x1100);
    REQUIRE(ReadSymbolValue(image, 2) == 0);
    REQUIRE(ReadSymbolValue(image, 3) == 0x1200);
}

TEST_CASE("NVN bridge: dynsym rejects malformed images", "[nvn][bridge]") {
    std::vector<u8> image(0x200, 0);
    Core::NvnHle::DynsymLayout layout;
    REQUIRE_FALSE(Core::NvnHle::ProbeDynsym(image, layout));
    REQUIRE(Core::NvnHle::FindDynsymHits(image, &WantsFooPrefix).empty());
    REQUIRE(Core::NvnHle::RewriteSymbolValues(
                image, std::map<std::string, u64, std::less<>>{{"x", 1}}) == 0);

    // MOD0 offset past the end of the image.
    const u32 bad = 0xFFFFFF00;
    std::memcpy(image.data() + 4, &bad, sizeof(bad));
    REQUIRE_FALSE(Core::NvnHle::ProbeDynsym(image, layout));
    REQUIRE_FALSE(Core::NvnHle::ProbeDynsym(std::span<const u8>{}, layout));
}

// ---------------------------------------------------------------------------------------------
// Bridge dispatch on a fake guest

class FakePort final : public Core::NvnHle::GuestPort {
public:
    void GetContext(Context& out) const override {
        out = ctx;
    }
    void SetContext(const Context& in) override {
        ctx = in;
        ++set_context_calls;
    }
    bool ReadBlock(u64 address, void* destination, size_t size) override {
        auto* out = static_cast<u8*>(destination);
        for (size_t i = 0; i < size; ++i) {
            const auto it = memory.find(address + i);
            out[i] = it == memory.end() ? 0 : it->second;
        }
        return true;
    }
    bool WriteBlock(u64 address, const void* source, size_t size) override {
        const auto* in = static_cast<const u8*>(source);
        for (size_t i = 0; i < size; ++i) {
            memory[address + i] = in[i];
        }
        return true;
    }
    std::string ReadCString(u64 address, size_t max_length) override {
        std::string text;
        for (size_t i = 0; i < max_length; ++i) {
            const auto it = memory.find(address + i);
            if (it == memory.end() || it->second == 0) {
                break;
            }
            text.push_back(static_cast<char>(it->second));
        }
        return text;
    }

    void PutString(u64 address, const std::string& text) {
        WriteBlock(address, text.c_str(), text.size() + 1);
    }
    u64 ReadU64(u64 address) {
        u64 value{};
        ReadBlock(address, &value, sizeof(value));
        return value;
    }

    Context ctx{};
    std::unordered_map<u64, u8> memory;
    int set_context_calls{};
};

constexpr u64 LoadBase = 0x80000000;
constexpr u64 RegionBase = LoadBase + SyntheticRegionOffset;
constexpr u64 TitleId = 0x040074A01BF12000ULL;
constexpr u64 NamePointer = 0x900000;

std::filesystem::path TemporaryDirectory(const char* tag) {
    return Tests::UniqueTempPath(std::string{"nvn-bridge-test-"} + tag);
}

u32 FindSlot(const Bridge& bridge, std::string_view name) {
    for (u32 slot = 0; slot < bridge.AllocatedSlots(); ++slot) {
        if (bridge.SlotName(slot) == name) {
            return slot;
        }
    }
    FAIL("slot not found: " << name);
    return 0;
}

/// Puts the fake guest right after the svc of `slot`, as the JIT does when it halts.
void EnterSlot(FakePort& port, u32 slot) {
    port.ctx.pc = RegionBase + Thunk::SlotOffset(slot) + 4;
}

constexpr std::array<std::string_view, 4> FakeNames{
    "nvnDeviceGetProcAddress", "nvnFoo", "nvnBar", "nvnBootstrapLoader"};

void IncrementBody(Core::System&, Context& ctx) {
    ctx.r[0] += 1000;
}

bool BackendAvailable(Core::System*) {
    return true;
}
bool BackendMissing(Core::System*) {
    return false;
}

struct BridgeFixture {
    explicit BridgeFixture(Config config_, const char* tag,
                           std::vector<SyntheticSymbol> symbols = DefaultSymbols())
        : directory{TemporaryDirectory(tag)}, config{std::move(config_)} {
        config.output_dir = directory.string();
        config.report_interval_seconds = 3600;
        bridge = std::make_unique<Bridge>(config, FakeNames);
        image = MakeSdkImage(symbols);
        hooks = bridge->PrepareSdk({image, SyntheticRegionOffset, LoadBase, "0123456789abcdef"});
        bridge->Activate(&owner, RegionBase, TitleId);
        port.ctx.sp = 0x7FF000;
        port.ctx.tpidr = 0x55;
        port.ctx.lr = 0x4000;
    }
    ~BridgeFixture() {
        bridge.reset();
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }
    static std::vector<SyntheticSymbol> DefaultSymbols() {
        return {{"nvnBootstrapLoader", 0x1000},
                {"nvnDeviceGetProcAddress", 0x1100},
                {"_ZN2nv18InitializeGraphicsEPvm", 0x1200},
                {"_ZN2nv9InitCheckEv", 0x1300},
                {"unrelated", 0x1400}};
    }
    bool Call(u32 slot) {
        EnterSlot(port, slot);
        return bridge->Dispatch(&system, port, &owner, Thunk::SvcBase + slot);
    }

    std::filesystem::path directory;
    Config config;
    Core::System system;
    std::unique_ptr<Bridge> bridge;
    std::vector<u8> image;
    size_t hooks{};
    int owner{};
    FakePort port;
};

TEST_CASE("NVN bridge: PrepareSdk hooks the sdk exports", "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "prepare"};

    REQUIRE(fx.hooks == 4);
    // Slot code and table entries were written into the image.
    std::array<u32, 4> words{};
    std::memcpy(words.data(), fx.image.data() + SyntheticRegionOffset, sizeof(words));
    REQUIRE(words == Thunk::SlotWords(0));

    const u32 bootstrap = FindSlot(*fx.bridge, "nvnBootstrapLoader");
    u64 target{};
    std::memcpy(&target, fx.image.data() + SyntheticRegionOffset + Thunk::TableEntryOffset(bootstrap),
                sizeof(target));
    REQUIRE(target == LoadBase + 0x1000);
    REQUIRE(fx.bridge->GetSlotKind(bootstrap) == SlotKind::WrapProc);
    REQUIRE(fx.bridge->GetSlotKind(FindSlot(*fx.bridge, "_ZN2nv18InitializeGraphicsEPvm")) ==
            SlotKind::CountLog);
    REQUIRE(fx.bridge->GetSlotKind(FindSlot(*fx.bridge, "_ZN2nv9InitCheckEv")) == SlotKind::Count);

    // The dynsym entries now point at their thunk slots; unrelated ones stay.
    REQUIRE(ReadSymbolValue(fx.image, 0) == SyntheticRegionOffset + 16 * FindSlot(*fx.bridge, "nvnBootstrapLoader"));
    REQUIRE(ReadSymbolValue(fx.image, 4) == 0x1400);
    REQUIRE(fx.bridge->GetSlotKind(Thunk::ReturnSlot) == SlotKind::Return);
}

TEST_CASE("NVN bridge: counting thunk is a pure pass-through", "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "count"};
    const u32 slot = FindSlot(*fx.bridge, "_ZN2nv9InitCheckEv");

    fx.port.ctx.r[0] = 11;
    fx.port.ctx.r[1] = 22;
    const int set_calls_before = fx.port.set_context_calls;
    REQUIRE(fx.Call(slot));
    REQUIRE(fx.Call(slot));
    REQUIRE(fx.Call(slot));
    REQUIRE(fx.bridge->SlotCalls(slot) == 3);
    // pc stays at the ldr; registers are untouched and no context was written.
    REQUIRE(fx.port.ctx.pc == RegionBase + Thunk::SlotOffset(slot) + 4);
    REQUIRE(fx.port.ctx.r[0] == 11);
    REQUIRE(fx.port.set_context_calls == set_calls_before);
}

TEST_CASE("NVN bridge: dispatch rejects foreign svcs and pcs", "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "reject"};
    const u32 slot = FindSlot(*fx.bridge, "_ZN2nv9InitCheckEv");

    // Wrong svc range.
    REQUIRE_FALSE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, 0x7F));
    REQUIRE_FALSE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, Thunk::SvcBase + 2048));
    // pc not right after the slot's svc (e.g. a guest svc #0x8000 elsewhere).
    fx.port.ctx.pc = 0x1234;
    REQUIRE_FALSE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, Thunk::SvcBase + slot));
    // Different owner process.
    int other{};
    EnterSlot(fx.port, slot);
    REQUIRE_FALSE(fx.bridge->Dispatch(&fx.system, fx.port, &other, Thunk::SvcBase + slot));
    // Slot index mismatch between svc immediate and pc.
    EnterSlot(fx.port, slot);
    REQUIRE_FALSE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, Thunk::SvcBase + slot + 1));
    // Unused slot.
    REQUIRE_FALSE(fx.Call(1500));
    REQUIRE(fx.bridge->SlotCalls(slot) == 0);
}

/// Runs GetProcAddress("name") through the wrapper and its return trampoline. `result` is what the
/// original function returned. Returns the x0 the guest sees.
u64 CallGetProcAddress(BridgeFixture& fx, const std::string& name, u64 result, u64 sp = 0x7FF000,
                       u64 caller_lr = 0x4000) {
    const u32 slot = FindSlot(*fx.bridge, "nvnDeviceGetProcAddress");
    fx.port.PutString(NamePointer, name);
    fx.port.ctx.r[0] = 0xDE;             // device
    fx.port.ctx.r[1] = NamePointer;      // name
    fx.port.ctx.sp = sp;
    fx.port.ctx.lr = caller_lr;
    REQUIRE(fx.Call(slot));
    // The pre-hook swapped lr for the return trampoline and left pc at the ldr.
    REQUIRE(fx.port.ctx.lr == RegionBase + Thunk::SlotOffset(Thunk::ReturnSlot));
    REQUIRE(fx.port.ctx.pc == RegionBase + Thunk::SlotOffset(slot) + 4);
    // The original function returns into the trampoline.
    fx.port.ctx.r[0] = result;
    fx.port.ctx.r[1] = 0x777; // garbage
    EnterSlot(fx.port, Thunk::ReturnSlot);
    REQUIRE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner,
                                Thunk::SvcBase + Thunk::ReturnSlot));
    REQUIRE(fx.port.ctx.pc == caller_lr);
    REQUIRE(fx.port.ctx.lr == caller_lr);
    return fx.port.ctx.r[0];
}

TEST_CASE("NVN bridge: GetProcAddress post-hook replaces procs with counting thunks",
          "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "procaddr"};

    const u64 proc_a = 0x80123450;
    const u64 thunk_a = CallGetProcAddress(fx, "nvnQueueSubmitCommands", proc_a);
    REQUIRE(thunk_a != proc_a);
    REQUIRE(thunk_a >= RegionBase);
    REQUIRE(thunk_a < RegionBase + Thunk::CodeBytes);
    REQUIRE((thunk_a - RegionBase) % Thunk::SlotBytes == 0);
    const u32 slot_a = static_cast<u32>((thunk_a - RegionBase) / Thunk::SlotBytes);
    REQUIRE(fx.bridge->SlotName(slot_a) == "nvnQueueSubmitCommands");
    REQUIRE(fx.bridge->GetSlotKind(slot_a) == SlotKind::Count);
    // The table entry holds the real target.
    REQUIRE(fx.port.ReadU64(RegionBase + Thunk::TableEntryOffset(slot_a)) == proc_a);

    // Same (target, name) -> same thunk; a different name gets its own slot.
    REQUIRE(CallGetProcAddress(fx, "nvnQueueSubmitCommands", proc_a) == thunk_a);
    const u64 thunk_b = CallGetProcAddress(fx, "nvnQueuePresentTexture", 0x80123460);
    REQUIRE(thunk_b != thunk_a);

    // Null results are left alone.
    REQUIRE(CallGetProcAddress(fx, "nvnDoesNotExist", 0) == 0);

    // The thunk counts and passes through.
    REQUIRE(fx.Call(slot_a));
    REQUIRE(fx.Call(slot_a));
    REQUIRE(fx.bridge->SlotCalls(slot_a) == 2);
    REQUIRE(fx.port.ctx.pc == RegionBase + Thunk::SlotOffset(slot_a) + 4);

    // The proc address wrapper itself is wrapped when handed out.
    const u64 wrapper = CallGetProcAddress(fx, "nvnDeviceGetProcAddress", 0x80200000);
    const u32 wrapper_slot = static_cast<u32>((wrapper - RegionBase) / Thunk::SlotBytes);
    REQUIRE(fx.bridge->GetSlotKind(wrapper_slot) == SlotKind::WrapProc);
    REQUIRE(fx.port.ReadU64(RegionBase + Thunk::TableEntryOffset(wrapper_slot)) == 0x80200000);
}

TEST_CASE("NVN bridge: nested GetProcAddress frames restore their own lr", "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "nested"};
    const u32 slot = FindSlot(*fx.bridge, "nvnDeviceGetProcAddress");

    // Outer call.
    fx.port.PutString(NamePointer, "nvnOuter");
    fx.port.ctx.r[1] = NamePointer;
    fx.port.ctx.sp = 0x7FE000;
    fx.port.ctx.lr = 0x1111;
    REQUIRE(fx.Call(slot));
    const Context outer_after_enter = fx.port.ctx;

    // Inner call on the same thread with a deeper stack, before the outer one returns.
    fx.port.PutString(NamePointer + 0x100, "nvnInner");
    fx.port.ctx.r[1] = NamePointer + 0x100;
    fx.port.ctx.sp = 0x7FD000;
    fx.port.ctx.lr = 0x2222;
    REQUIRE(fx.Call(slot));

    // Inner returns first.
    fx.port.ctx.r[0] = 0x90000010;
    EnterSlot(fx.port, Thunk::ReturnSlot);
    REQUIRE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, Thunk::SvcBase + Thunk::ReturnSlot));
    REQUIRE(fx.port.ctx.pc == 0x2222);
    REQUIRE(fx.bridge->SlotName(static_cast<u32>((fx.port.ctx.r[0] - RegionBase) / 16)) ==
            "nvnInner");

    // Then the outer one.
    fx.port.ctx = outer_after_enter;
    fx.port.ctx.r[0] = 0x90000020;
    EnterSlot(fx.port, Thunk::ReturnSlot);
    REQUIRE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, Thunk::SvcBase + Thunk::ReturnSlot));
    REQUIRE(fx.port.ctx.pc == 0x1111);
    REQUIRE(fx.bridge->SlotName(static_cast<u32>((fx.port.ctx.r[0] - RegionBase) / 16)) ==
            "nvnOuter");
}

TEST_CASE("NVN bridge: return trampoline without a frame faults the guest, not the host",
          "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "noframe"};
    EnterSlot(fx.port, Thunk::ReturnSlot);
    // Handled (no fall-through into Svc::Call); pc stays on the slot's ldr, whose table entry is
    // 0, so the guest faults at address 0.
    REQUIRE(fx.bridge->Dispatch(&fx.system, fx.port, &fx.owner, Thunk::SvcBase + Thunk::ReturnSlot));
    REQUIRE(fx.port.ctx.pc == RegionBase + Thunk::SlotOffset(Thunk::ReturnSlot) + 4);
    REQUIRE(fx.port.ReadU64(RegionBase + Thunk::TableEntryOffset(Thunk::ReturnSlot)) == 0);
}

TEST_CASE("NVN bridge: census rows and TSV formatting", "[nvn][bridge]") {
    Config config;
    config.census = true;
    BridgeFixture fx{config, "tsv"};
    const u32 init_check = FindSlot(*fx.bridge, "_ZN2nv9InitCheckEv");
    for (int i = 0; i < 5; ++i) {
        REQUIRE(fx.Call(init_check));
    }
    const u64 thunk = CallGetProcAddress(fx, "nvnQueueSubmitCommands", 0x80123450);
    const u32 submit = static_cast<u32>((thunk - RegionBase) / 16);
    REQUIRE(fx.Call(submit));
    REQUIRE(fx.Call(submit));

    const auto rows = fx.bridge->CensusRows();
    REQUIRE(rows.size() >= 3);
    REQUIRE(rows[0].name == "_ZN2nv9InitCheckEv");
    REQUIRE(rows[0].calls == 5);
    REQUIRE(rows[1].calls <= rows[0].calls);
    bool found_submit = false;
    for (const auto& row : rows) {
        if (row.name == "nvnQueueSubmitCommands") {
            found_submit = true;
            REQUIRE(row.calls == 2);
        }
    }
    REQUIRE(found_submit);

    const std::vector<Bridge::Row> sample{{"a", 10}, {"b", 0}};
    REQUIRE(Bridge::FormatTsv(sample, 4) ==
            "name\tcalls\tcalls_per_frame\na\t10\t2.5000\nb\t0\t0.0000\n");
    REQUIRE(Bridge::FormatTsv(sample, 0) ==
            "name\tcalls\tcalls_per_frame\na\t10\t0.0000\nb\t0\t0.0000\n");

    REQUIRE(fx.bridge->WriteReports(10));
    std::ifstream in{fx.directory / "census-040074A01BF12000.tsv"};
    REQUIRE(in.good());
    std::stringstream content;
    content << in.rdbuf();
    REQUIRE(content.str().starts_with("name\tcalls\tcalls_per_frame\n_ZN2nv9InitCheckEv\t5\t0.5000\n"));
}

TEST_CASE("NVN bridge: HLE bootstrap hands out HLE thunks", "[nvn][bridge]") {
    Config config;
    config.hle = true;
    BridgeFixture fx{config, "hle"};
    fx.bridge->SetBackendProbe(&BackendAvailable);
    fx.bridge->RegisterHle(1, &IncrementBody); // nvnFoo implemented, nvnBar not

    const u32 bootstrap = FindSlot(*fx.bridge, "nvnBootstrapLoader");
    REQUIRE(fx.bridge->GetSlotKind(bootstrap) == SlotKind::HleBootstrap);

    const auto bootstrap_call = [&](const std::string& name) {
        fx.port.PutString(NamePointer, name);
        fx.port.ctx.r[0] = NamePointer;
        fx.port.ctx.lr = 0x5000;
        REQUIRE(fx.Call(bootstrap));
        // HLE return: pc = x30, the thunk's ldr/br tail never runs.
        REQUIRE(fx.port.ctx.pc == 0x5000);
        return fx.port.ctx.r[0];
    };

    const u64 foo = bootstrap_call("nvnFoo");
    REQUIRE(foo != 0);
    REQUIRE(bootstrap_call("nvnFoo") == foo);
    REQUIRE(bootstrap_call("nvnUnknownName") == 0);
    REQUIRE(fx.bridge->IsHleActive());

    const u32 foo_slot = static_cast<u32>((foo - RegionBase) / 16);
    REQUIRE(fx.bridge->GetSlotKind(foo_slot) == SlotKind::Hle);

    // Registered body runs and returns to the caller.
    fx.port.ctx.r[0] = 5;
    fx.port.ctx.lr = 0x6000;
    REQUIRE(fx.Call(foo_slot));
    REQUIRE(fx.port.ctx.r[0] == 1005);
    REQUIRE(fx.port.ctx.pc == 0x6000);

    // Unregistered function: x0 = 0, returns to the caller, counted once in the unimplemented set.
    const u64 bar = bootstrap_call("nvnBar");
    const u32 bar_slot = static_cast<u32>((bar - RegionBase) / 16);
    for (int i = 0; i < 3; ++i) {
        fx.port.ctx.r[0] = 99;
        fx.port.ctx.lr = 0x7000;
        REQUIRE(fx.Call(bar_slot));
        REQUIRE(fx.port.ctx.r[0] == 0);
        REQUIRE(fx.port.ctx.pc == 0x7000);
    }
    const auto unimplemented = fx.bridge->UnimplementedRows();
    REQUIRE(unimplemented.size() == 1);
    REQUIRE(unimplemented[0].name == "nvnBar");
    REQUIRE(unimplemented[0].calls == 3);

    REQUIRE(fx.bridge->WriteReports(1));
    REQUIRE(std::filesystem::exists(fx.directory / "unimplemented-040074A01BF12000.tsv"));
}

TEST_CASE("NVN bridge: unimplemented stub zeroes every result register, unsupported TSV",
          "[nvn][bridge]") {
    Config config;
    config.hle = true;
    BridgeFixture fx{config, "hle-unsupported"};
    fx.bridge->SetBackendProbe(&BackendAvailable);
    fx.port.PutString(NamePointer, "nvnBar");
    fx.port.ctx.r[0] = NamePointer;
    fx.port.ctx.lr = 0x5000;
    REQUIRE(fx.Call(FindSlot(*fx.bridge, "nvnBootstrapLoader")));
    const u64 bar = fx.port.ctx.r[0];
    const u32 bar_slot = static_cast<u32>((bar - RegionBase) / 16);
    fx.port.ctx.r[0] = 99;
    fx.port.ctx.r[1] = 98;
    fx.port.ctx.v[0] = {0x3F800000, 0xFFFF};
    fx.port.ctx.lr = 0x7000;
    REQUIRE(fx.Call(bar_slot));
    CHECK(fx.port.ctx.r[0] == 0);
    CHECK(fx.port.ctx.r[1] == 0);
    CHECK(fx.port.ctx.v[0][0] == 0);
    CHECK(fx.port.ctx.v[0][1] == 0);

    // Without a provider there is no unsupported file; with one, rows sort by count.
    REQUIRE(fx.bridge->WriteReports(10));
    CHECK_FALSE(std::filesystem::exists(fx.directory / "unsupported-040074A01BF12000.tsv"));
    fx.bridge->SetUnsupportedProvider([] {
        return std::vector<Bridge::UnsupportedRow>{{"recorder", "tag_mismatch", 2},
                                                   {"replay", "alias_unsupported", 50}};
    });
    REQUIRE(fx.bridge->WriteReports(10));
    std::ifstream in{fx.directory / "unsupported-040074A01BF12000.tsv"};
    std::string text{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
    CHECK(text == "where\treason\tcount\tper_frame\n"
                  "replay\talias_unsupported\t50\t5.0000\n"
                  "recorder\ttag_mismatch\t2\t0.2000\n");
}

namespace {
int observed_calls = 0;
void ObserverBody(Core::System&, Context& ctx) {
    ++observed_calls;
    ctx.r[5] = 0; // an observer's register changes must not matter: nothing is written back
}
} // namespace

TEST_CASE("NVN bridge: nv observers see the call and the original function still runs",
          "[nvn][bridge]") {
    Config config;
    config.hle = true;
    BridgeFixture fx{config, "hle-observer", {}};
    fx.bridge->RegisterNvObserver("_ZN2nv18InitializeGraphicsEPvm", &ObserverBody);
    fx.image = MakeSdkImage(BridgeFixture::DefaultSymbols());
    fx.bridge->PrepareSdk({fx.image, SyntheticRegionOffset, LoadBase, "0123456789abcdef"});
    fx.bridge->Activate(&fx.owner, RegionBase, TitleId);
    fx.bridge->SetBackendProbe(&BackendAvailable);
    const u32 slot = FindSlot(*fx.bridge, "_ZN2nv18InitializeGraphicsEPvm");
    observed_calls = 0;
    fx.port.ctx.r[0] = 11;
    fx.port.ctx.r[1] = 22;
    fx.port.ctx.lr = 0x5000;
    REQUIRE(fx.Call(slot));
    CHECK(observed_calls == 1);
    // Not an HLE return: pc stays right after the slot's svc so ldr/br run the original.
    CHECK(fx.port.ctx.pc == RegionBase + Thunk::SlotOffset(slot) + 4);
    CHECK(fx.port.ctx.r[0] == 11);
    CHECK(fx.port.ctx.r[1] == 22);
}

TEST_CASE("NVN bridge: HLE GetProcAddress resolves names from x1", "[nvn][bridge]") {
    Config config;
    config.hle = true;
    BridgeFixture fx{config, "hle-getproc"};
    fx.bridge->SetBackendProbe(&BackendAvailable);

    const u32 get_proc = FindSlot(*fx.bridge, "nvnDeviceGetProcAddress");
    REQUIRE(fx.bridge->GetSlotKind(get_proc) == SlotKind::HleProcAddress);
    fx.port.PutString(NamePointer, "nvnFoo");
    fx.port.ctx.r[0] = 0xDE;
    fx.port.ctx.r[1] = NamePointer;
    fx.port.ctx.lr = 0x5000;
    REQUIRE(fx.Call(get_proc));
    REQUIRE(fx.port.ctx.r[0] >= RegionBase);
    REQUIRE(fx.port.ctx.pc == 0x5000);

    // Asking for the proc address function itself yields an HLE proc-address thunk.
    fx.port.PutString(NamePointer, "nvnDeviceGetProcAddress");
    fx.port.ctx.r[1] = NamePointer;
    REQUIRE(fx.Call(get_proc));
    const u32 own = static_cast<u32>((fx.port.ctx.r[0] - RegionBase) / 16);
    REQUIRE(fx.bridge->GetSlotKind(own) == SlotKind::HleProcAddress);
}

TEST_CASE("NVN bridge: HLE without a Vulkan backend installs nothing", "[nvn][bridge]") {
    Config config;
    config.hle = true;
    BridgeFixture fx{config, "hle-nobackend"};
    fx.bridge->SetBackendProbe(&BackendMissing);
    const u32 bootstrap = FindSlot(*fx.bridge, "nvnBootstrapLoader");

    fx.port.PutString(NamePointer, "nvnFoo");
    fx.port.ctx.r[0] = NamePointer;
    fx.port.ctx.lr = 0x5000;
    REQUIRE(fx.Call(bootstrap));
    // Pass-through: registers and pc untouched, no HLE thunk handed out.
    REQUIRE(fx.port.ctx.pc == RegionBase + Thunk::SlotOffset(bootstrap) + 4);
    REQUIRE(fx.port.ctx.r[0] == NamePointer);
    REQUIRE_FALSE(fx.bridge->IsHleActive());
    REQUIRE(fx.bridge->UnimplementedRows().empty());
}

// ---------------------------------------------------------------------------------------------
// Loader layout

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value == '\0') {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

/// Saves the NVN switches on construction and restores them (including "unset") on destruction,
/// so a failing test cannot leak its environment and a user-set switch survives the run.
class SwitchGuard {
public:
    SwitchGuard() {
        for (size_t i = 0; i < names.size(); ++i) {
            if (const char* value = std::getenv(names[i])) {
                saved[i] = value;
            }
        }
    }
    ~SwitchGuard() {
        for (size_t i = 0; i < names.size(); ++i) {
            SetEnvironment(names[i], saved[i] ? saved[i]->c_str() : "");
        }
    }
    SwitchGuard(const SwitchGuard&) = delete;
    SwitchGuard& operator=(const SwitchGuard&) = delete;

private:
    static constexpr std::array<const char*, 2> names{"EDEN_NX2_NVN_CENSUS", "EDEN_NX2_NVN_HLE"};
    std::array<std::optional<std::string>, 2> saved;
};

TEST_CASE("NVN bridge: sdk module layout includes the thunk region only when enabled",
          "[nvn][bridge]") {
    const SwitchGuard switch_guard;
    Core::System system;
    Kernel::KProcess process{system.Kernel()};
    Loader::NSOHeader header{};
    header.magic = Common::MakeMagic('N', 'S', 'O', '0');
    const std::vector<u8> text(16, 0x42);
    header.segments[0] = {sizeof(header), 0, 16, {}};
    header.segments[1] = {static_cast<u32>(sizeof(header) + text.size()), 0x1000, 16, {}};
    header.segments[2] = {static_cast<u32>(sizeof(header) + text.size() + 16), 0x2000, 16, {}};
    // Eglantine's loader reads every segment by its stored size, as real NSOs give it.
    header.segments_compressed_size = {16, 16, 16};
    std::vector<u8> bytes(sizeof(header) + text.size() + 32);
    std::memcpy(bytes.data() + sizeof(header), text.data(), text.size());
    std::memcpy(bytes.data(), &header, sizeof(header));

    const auto layout = [&](const char* module_name) {
        FileSys::VectorVfsFile file{bytes, module_name};
        return Loader::AppLoader_NSO::LoadModule(process, system, file, 0x8000000, false, false);
    };

    // Same image as core/nso.cpp's valid case: 0x8003000 without the region.
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "");
    SetEnvironment("EDEN_NX2_NVN_HLE", "");
    REQUIRE(layout("sdk") == 0x8003000ULL);

    SetEnvironment("EDEN_NX2_NVN_CENSUS", "1");
    REQUIRE(layout("sdk") == 0x8003000ULL + Thunk::RegionBytes);
    REQUIRE(layout("main") == 0x8003000ULL); // other modules are never extended
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "0");
    REQUIRE(layout("sdk") == 0x8003000ULL);

    SetEnvironment("EDEN_NX2_NVN_HLE", "1");
    REQUIRE(layout("sdk") == 0x8003000ULL + Thunk::RegionBytes);
    SetEnvironment("EDEN_NX2_NVN_HLE", "");
    REQUIRE(layout("sdk") == 0x8003000ULL);
}

TEST_CASE("NVN bridge: switch parsing", "[nvn][bridge]") {
    const SwitchGuard switch_guard;
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "");
    SetEnvironment("EDEN_NX2_NVN_HLE", "");
    REQUIRE_FALSE(Core::NvnHle::ConfigFromEnvironment().AnyEnabled());
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "0");
    REQUIRE_FALSE(Core::NvnHle::ConfigFromEnvironment().census);
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "1");
    REQUIRE(Core::NvnHle::ConfigFromEnvironment().census);
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "yes");
    REQUIRE(Core::NvnHle::ConfigFromEnvironment().census);
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "");
    SetEnvironment("EDEN_NX2_NVN_HLE", "1");
    const auto config = Core::NvnHle::ConfigFromEnvironment();
    REQUIRE(config.hle);
    REQUIRE_FALSE(config.census);
    SetEnvironment("EDEN_NX2_NVN_HLE", "");
}

TEST_CASE("NVN HLE choice: environment over setting, Switch 2 programs only", "[nvn][bridge]") {
    using Settings::GraphicsPath;
    using Settings::NvnHleReason;
    // The setting decides when the variable is unset; a Switch 1 program stays on the emulated GPU.
    REQUIRE(Settings::ChooseNvnHle(nullptr, GraphicsPath::Hle, true).hle);
    REQUIRE(Settings::ChooseNvnHle(nullptr, GraphicsPath::Hle, true).reason ==
            NvnHleReason::SettingHle);
    REQUIRE_FALSE(Settings::ChooseNvnHle(nullptr, GraphicsPath::Emulated, true).hle);
    REQUIRE(Settings::ChooseNvnHle(nullptr, GraphicsPath::Emulated, true).reason ==
            NvnHleReason::SettingEmulated);
    REQUIRE_FALSE(Settings::ChooseNvnHle(nullptr, GraphicsPath::Hle, false).hle);
    REQUIRE(Settings::ChooseNvnHle(nullptr, GraphicsPath::Hle, false).reason ==
            NvnHleReason::NotSwitch2);
    // EDEN_NX2_NVN_HLE overrides both ways, for any program.
    REQUIRE(Settings::ChooseNvnHle("1", GraphicsPath::Emulated, false).hle);
    REQUIRE(Settings::ChooseNvnHle("1", GraphicsPath::Emulated, false).reason ==
            NvnHleReason::EnvironmentOn);
    REQUIRE(Settings::ChooseNvnHle("yes", GraphicsPath::Emulated, true).hle);
    REQUIRE_FALSE(Settings::ChooseNvnHle("0", GraphicsPath::Hle, true).hle);
    REQUIRE(Settings::ChooseNvnHle("0", GraphicsPath::Hle, true).reason ==
            NvnHleReason::EnvironmentOff);
}

TEST_CASE("NVN HLE choice: the bridge configuration follows the setting", "[nvn][bridge]") {
    const SwitchGuard switch_guard;
    SetEnvironment("EDEN_NX2_NVN_CENSUS", "");
    SetEnvironment("EDEN_NX2_NVN_HLE", "");
    const bool was_switch2 = Settings::CurrentProgramIsSwitch2();
    const auto setting = Settings::values.graphics_path.GetValue();
    // The emulated GPU is the default graphics path.
    REQUIRE(Settings::values.graphics_path.GetDefault() == Settings::GraphicsPath::Emulated);
    Settings::values.graphics_path.SetValue(Settings::GraphicsPath::Hle);

    Settings::SetCurrentProgramIsSwitch2(true);
    REQUIRE(Core::NvnHle::ConfigFromEnvironment().hle);
    REQUIRE(Core::NvnHle::ConfigFromEnvironment().AnyEnabled());
    Settings::SetCurrentProgramIsSwitch2(false);
    REQUIRE_FALSE(Core::NvnHle::ConfigFromEnvironment().hle);

    Settings::SetCurrentProgramIsSwitch2(true);
    Settings::values.graphics_path.SetValue(Settings::GraphicsPath::Emulated);
    REQUIRE_FALSE(Core::NvnHle::ConfigFromEnvironment().hle);
    SetEnvironment("EDEN_NX2_NVN_HLE", "1");
    REQUIRE(Core::NvnHle::ConfigFromEnvironment().hle);
    SetEnvironment("EDEN_NX2_NVN_HLE", "");

    Settings::values.graphics_path.SetValue(setting);
    Settings::SetCurrentProgramIsSwitch2(was_switch2);
}

TEST_CASE("NVN bridge: output directory and call counting switches", "[nvn][bridge]") {
    const SwitchGuard switch_guard;
    const std::optional<std::string> saved = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_OUTPUT_DIR");
        return value != nullptr ? std::optional<std::string>{value} : std::nullopt;
    }();
    SetEnvironment("EDEN_NX2_NVN_OUTPUT_DIR", "some/where");
    REQUIRE(Core::NvnHle::OutputDirectory() == std::filesystem::path{"some/where"});
    SetEnvironment("EDEN_NX2_NVN_OUTPUT_DIR", saved ? saved->c_str() : "");

    Config config;
    REQUIRE_FALSE(config.CountCalls());
    config.hle = true;
    REQUIRE_FALSE(config.CountCalls());
    config.census = true;
    REQUIRE(config.CountCalls());
    config.census = false;
    config.thread_dump = true;
    REQUIRE(config.CountCalls());
}

} // namespace
