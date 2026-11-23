// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// NVN raw global memory. A global access whose address is the result of a memory load (a table of
// pointers the shader reads, DK's terrain draw records) can never be tied to a c0 storage buffer:
// with `HostTranslateInfo::nvn_raw_global_memory` it stays a global access and the SPIR-V backend
// (`Profile::support_raw_global_memory`) emits it through PhysicalStorageBuffer64 pointers and the
// guest -> device address table in the driver bank.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/raw_coverage_layout.h"
#include "shader_recompiler/runtime_info.h"
#include "tests/temp_path.h"
#include "video_core/nvn/shader/nvn_program_map.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"

namespace {

using namespace VideoCommon::Nvn;

enum Size : u64 { B32 = 4, B64 = 5 };

/// LDC.size Rd, c[0][offset]
constexpr u64 Ldc(u32 rd, u32 offset, Size size) {
    return 0xef90000000000000ULL | rd | (u64{0xff} << 8) | (u64{offset} << 20) | (u64{size} << 48);
}
/// LDG.E.size Rd, [Ra + offset]
constexpr u64 LdgE(u32 rd, u32 ra, u32 offset, Size size) {
    return 0xeed0000000000000ULL | rd | (u64{ra} << 8) | (u64{offset} << 20) | (u64{1} << 45) |
           (u64{size} << 48);
}
/// STG.E.size [Ra + offset], Rdata
constexpr u64 StgE(u32 rdata, u32 ra, u32 offset, Size size) {
    return 0xeed8000000000000ULL | rdata | (u64{ra} << 8) | (u64{offset} << 20) | (u64{1} << 45) |
           (u64{size} << 48);
}
constexpr u64 Exit = 0xe30000000007000fULL;

class VertexEnvironment final : public Shader::Environment {
public:
    explicit VertexEnvironment(std::span<const u64> program) {
        stage = Shader::Stage::VertexB;
        start_address = 0;
        is_proprietary_driver = true;
        code.push_back(0); // scheduling word of the first group
        for (const u64 instruction : program) {
            if ((code.size() * sizeof(u64)) % 32 == 0) {
                code.push_back(0);
            }
            code.push_back(instruction);
        }
        code.push_back(0);
    }
    u64 ReadInstruction(u32 address) override {
        return address / 8 < code.size() ? code[address / 8] : 0;
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return 2;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::vector<u64> code;
};

struct Translated {
    NvnShaderPools pools;
    Shader::IR::Program program;

    [[nodiscard]] size_t Count(Shader::IR::Opcode opcode) const {
        size_t count = 0;
        for (const Shader::IR::Block* block : program.post_order_blocks) {
            for (const Shader::IR::Inst& inst : block->Instructions()) {
                count += inst.GetOpcode() == opcode ? 1 : 0;
            }
        }
        return count;
    }
};

std::unique_ptr<Translated> Translate(std::span<const u64> program, bool raw_global_memory) {
    VertexEnvironment env{program};
    auto result = std::make_unique<Translated>();
    Shader::Maxwell::Flow::CFG cfg(env, result->pools.flow_block, env.StartAddress(), false);
    Shader::HostTranslateInfo host_info = MakeReferenceHostInfo();
    host_info.nvn_raw_global_memory = raw_global_memory;
    result->program = Shader::Maxwell::TranslateProgram(result->pools.inst, result->pools.block,
                                                        env, cfg, host_info);
    return result;
}

std::vector<u32> Emit(Translated& translated, bool profile_raw_global_memory) {
    Shader::Profile profile = MakeReferenceProfile();
    profile.support_descriptor_aliasing = true;
    profile.support_raw_global_memory = profile_raw_global_memory;
    Shader::RuntimeInfo runtime_info;
    runtime_info.previous_stage_stores.mask.set();
    Shader::Backend::Bindings bindings;
    std::vector<u32> words =
        Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, translated.program, bindings);
    REQUIRE(words.size() > 5);
    return words;
}

/// Runs spirv-val (EDEN_NVN_SPIRV_VAL, else the SDK the coverage tool uses) on the module. True when
/// the tool is not installed: the check only runs where it can.
bool ValidSpirv(std::span<const u32> words) {
    const char* configured = std::getenv("EDEN_NVN_SPIRV_VAL");
    const bool explicit_tool = configured != nullptr && *configured != '\0';
    const std::string tool =
        explicit_tool ? configured : "C:/eg/VulkanSDK/1.4.341.1/Bin/spirv-val.exe";
    if (tool == "none") {
        return true;
    }
    if (!std::filesystem::exists(tool)) {
        if (explicit_tool) {
            // A validator asked for by name must run.
            UNSCOPED_INFO("EDEN_NVN_SPIRV_VAL names a missing file: " << tool);
            return false;
        }
        return true;
    }
    const std::filesystem::path path = Tests::UniqueTempPath("nvn_raw_global_test", ".spv");
    {
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        REQUIRE(file != nullptr);
        std::fwrite(words.data(), sizeof(u32), words.size(), file);
        std::fclose(file);
    }
    const std::filesystem::path log = path.string() + ".log";
    std::string command = "\"" + tool + "\" --target-env vulkan1.3 --uniform-buffer-standard-layout \"" +
                          path.string() + "\" > \"" + log.string() + "\" 2>&1";
#ifdef _WIN32
    // cmd.exe /c strips the outer quotes of a line that starts with a quoted path.
    command = "\"" + command + "\"";
#endif
    const int code = std::system(command.c_str());
    if (code != 0) {
        std::FILE* file = std::fopen(log.string().c_str(), "rb");
        std::string message(512, '\0');
        if (file != nullptr) {
            message.resize(std::fread(message.data(), 1, message.size(), file));
            std::fclose(file);
        }
        UNSCOPED_INFO("spirv-val: " << message);
    } else {
        std::filesystem::remove(path);
    }
    std::filesystem::remove(log);
    return code == 0;
}

struct ModuleFacts {
    bool physical_storage_capability{};
    u32 addressing_model{~0u};
    u32 convert_u_to_ptr{};
    u32 physical_pointer_types{};
    bool malformed{};
};

ModuleFacts Scan(std::span<const u32> words) {
    constexpr u32 OpMemoryModel = 14;
    constexpr u32 OpCapability = 17;
    constexpr u32 OpTypePointer = 32;
    constexpr u32 OpConvertUToPtr = 120;
    constexpr u32 CapabilityPhysicalStorageBufferAddresses = 5347;
    constexpr u32 StorageClassPhysicalStorageBuffer = 5349;
    ModuleFacts facts;
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16;
        const u32 opcode = words[i] & 0xffff;
        if (count == 0 || count > words.size() - i) {
            facts.malformed = true; // truncated: its operands are not there
            break;
        }
        if (opcode == OpCapability && count >= 2 &&
            words[i + 1] == CapabilityPhysicalStorageBufferAddresses) {
            facts.physical_storage_capability = true;
        } else if (opcode == OpMemoryModel && count >= 2) {
            facts.addressing_model = words[i + 1];
        } else if (opcode == OpConvertUToPtr) {
            ++facts.convert_u_to_ptr;
        } else if (opcode == OpTypePointer && count >= 3 &&
                   words[i + 2] == StorageClassPhysicalStorageBuffer) {
            ++facts.physical_pointer_types;
        }
        i += count;
    }
    return facts;
}

/// The pointer table at c0[0x110] holds a pointer; the shader loads it and stores through it.
constexpr std::array<u64, 4> POINTER_THROUGH_TABLE{
    Ldc(4, 0x110, B64),        // R4:R5 = c0 storage buffer 0 (address, size)
    LdgE(6, 4, 0, B64),        // R6:R7 = *(u64*)R4:R5: an address read from memory
    StgE(8, 6, 0, B32),        // *(u32*)R6:R7 = R8
    Exit,
};

/// Only the c0 storage buffer is accessed.
constexpr std::array<u64, 4> TRACKED_ONLY{
    Ldc(4, 0x110, B64),
    LdgE(2, 4, 0, B32),
    StgE(2, 4, 4, B32),
    Exit,
};

} // namespace

TEST_CASE("NVN raw global memory: an address read from memory is not a storage buffer", "[nvn][shaders]") {
    using Shader::IR::Opcode;
    SECTION("off: Eglantine's pass keeps the store through the loaded pointer global") {
        const auto translated = Translate(POINTER_THROUGH_TABLE, false);
        CHECK(translated->Count(Opcode::LoadStorage64) == 1);
        CHECK(translated->Count(Opcode::WriteStorage32) == 0);
        CHECK(translated->Count(Opcode::WriteGlobal32) == 1);
        CHECK_FALSE(translated->program.info.uses_raw_global_memory);
        CHECK_FALSE(translated->program.info.stores_raw_global_memory);
    }
    SECTION("on: the load keeps its storage buffer, the store stays raw global memory") {
        const auto translated = Translate(POINTER_THROUGH_TABLE, true);
        CHECK(translated->Count(Opcode::LoadStorage64) == 1);
        CHECK(translated->Count(Opcode::WriteStorage32) == 0);
        CHECK(translated->Count(Opcode::WriteGlobal32) == 1);
        const Shader::Info& info = translated->program.info;
        CHECK(info.nvn_raw_global_memory);
        CHECK(info.uses_raw_global_memory);
        CHECK(info.stores_raw_global_memory);
        // The table is read from the driver bank.
        CHECK((info.constant_buffer_mask & 1u) != 0);
        CHECK(info.constant_buffer_used_sizes[0] >= 0x1000);
    }
    SECTION("on: accesses with a known c0 storage buffer base are still tracked") {
        const auto translated = Translate(TRACKED_ONLY, true);
        CHECK(translated->Count(Opcode::LoadStorage32) == 1);
        CHECK(translated->Count(Opcode::WriteStorage32) == 1);
        CHECK(translated->Count(Opcode::LoadGlobal32) == 0);
        CHECK(translated->Count(Opcode::WriteGlobal32) == 0);
        CHECK_FALSE(translated->program.info.uses_raw_global_memory);
    }
}

TEST_CASE("NVN raw global memory: emission through PhysicalStorageBuffer64", "[nvn][shaders]") {
    SECTION("profile support: capability, addressing model, pointer conversions") {
        auto translated = Translate(POINTER_THROUGH_TABLE, true);
        const std::vector<u32> words = Emit(*translated, true);
        CHECK(ValidSpirv(words));
        const ModuleFacts facts = Scan(words);
        CHECK_FALSE(facts.malformed);
        CHECK(facts.physical_storage_capability);
        CHECK(facts.addressing_model == 5348); // PhysicalStorageBuffer64
        CHECK(facts.convert_u_to_ptr >= 1);
        CHECK(facts.physical_pointer_types >= 1);
    }
    SECTION("no profile support: the c0 storage buffer table fallback, logical addressing") {
        auto translated = Translate(POINTER_THROUGH_TABLE, true);
        const ModuleFacts facts = Scan(Emit(*translated, false));
        CHECK_FALSE(facts.malformed);
        CHECK_FALSE(facts.physical_storage_capability);
        CHECK(facts.addressing_model == 0); // Logical
        CHECK(facts.convert_u_to_ptr == 0);
    }
    SECTION("no raw access: the module is the same with the switches on and off") {
        auto on = Translate(TRACKED_ONLY, true);
        auto off = Translate(TRACKED_ONLY, false);
        const std::vector<u32> with_flags = Emit(*on, true);
        const std::vector<u32> without_flags = Emit(*off, false);
        CHECK(with_flags == without_flags);
        CHECK_FALSE(Scan(with_flags).physical_storage_capability);
    }
}

TEST_CASE("NVN raw global memory: address range table", "[nvn]") {
    using Vulkan::Nvn::RawGlobalRange;
    using Vulkan::Nvn::AppendRawGlobalRange;


    SECTION("ranges contiguous in both address spaces are merged") {
        std::vector<RawGlobalRange> ranges;
        AppendRawGlobalRange(ranges, 0x1000, 0x7000'0000, 0x1000);
        AppendRawGlobalRange(ranges, 0x2000, 0x7000'1000, 0x2000); // continues both
        AppendRawGlobalRange(ranges, 0x4000, 0x9000'0000, 0x1000); // guest continues, device does not
        AppendRawGlobalRange(ranges, 0x9000, 0x9000'1000, 0x1000); // device continues, guest does not
        AppendRawGlobalRange(ranges, 0xA000, 0x9000'2000, 0);      // empty
        REQUIRE(ranges.size() == 3);
        CHECK(ranges[0].guest == 0x1000);
        CHECK(ranges[0].device == 0x7000'0000);
        CHECK(ranges[0].size == 0x3000);
        CHECK(ranges[1].guest == 0x4000);
        CHECK(ranges[2].guest == 0x9000);
    }
    SECTION("a higher priority source cuts a lower one; the result is sorted and disjoint") {
        using Vulkan::Nvn::FlattenRawGlobalRanges;
        using Vulkan::Nvn::RawGlobalSource;
        // A chunk [0x10000, 0x20000) with two pools of their own inside it, and a pool outside
        // listed first.
        const std::vector<RawGlobalSource> sources{
            {0x40000, 0xA000'0000, 0x1000, 1},   // outside every other range
            {0x10000, 0x5000'0000, 0x10000, 0},  // the chunk
            {0x14000, 0x6000'0000, 0x2000, 1},   // a pool inside the chunk
            {0x1F000, 0x7000'0000, 0x1000, 1},   // a pool at the chunk's end
        };
        const std::vector<RawGlobalRange> table = FlattenRawGlobalRanges(sources);
        REQUIRE(table.size() == 5);
        CHECK(table[0].guest == 0x10000);
        CHECK(table[0].device == 0x5000'0000);
        CHECK(table[0].size == 0x4000);
        CHECK(table[1].guest == 0x14000);
        CHECK(table[1].device == 0x6000'0000);
        CHECK(table[1].size == 0x2000);
        CHECK(table[2].guest == 0x16000);
        CHECK(table[2].device == 0x5000'6000); // the chunk continues at its own offset
        CHECK(table[2].size == 0x9000);
        CHECK(table[3].guest == 0x1F000);
        CHECK(table[3].device == 0x7000'0000);
        CHECK(table[3].size == 0x1000);
        CHECK(table[4].guest == 0x40000);
        CHECK(table[4].device == 0xA000'0000);

        for (size_t i = 1; i < table.size(); ++i) {
            CHECK(table[i - 1].guest + table[i - 1].size <= table[i].guest);
        }
    }
    SECTION("VRAM mirror alias replaces the imported-pool raw address") {
        using Vulkan::Nvn::FlattenRawGlobalRanges;
        using Vulkan::Nvn::RawGlobalSource;
        const std::vector<RawGlobalSource> sources{
            {0x10000, 0x8000'0000, 0x4000, 1}, // imported host pool
            {0x10000, 0xA000'0000, 0x4000, 2}, // addressable VRAM mirror
        };
        const std::vector<RawGlobalRange> table = FlattenRawGlobalRanges(sources);
        REQUIRE(table.size() == 1);
        CHECK(table[0].guest == 0x10000);
        CHECK(table[0].device == 0xA000'0000);
        CHECK(table[0].size == 0x4000);
    }
    SECTION("sources of equal priority: the earlier one wins; empty sources are ignored") {
        using Vulkan::Nvn::FlattenRawGlobalRanges;
        using Vulkan::Nvn::RawGlobalSource;
        const std::vector<RawGlobalSource> sources{{0x1000, 0x8000, 0x2000, 1},
                                                   {0x2000, 0xC000, 0x2000, 1},
                                                   {0x9000, 0xB000, 0, 1}};
        const std::vector<RawGlobalRange> table = FlattenRawGlobalRanges(sources);
        REQUIRE(table.size() == 2);
        CHECK(table[0].guest == 0x1000);
        CHECK(table[0].device == 0x8000);
        CHECK(table[0].size == 0x2000);
        CHECK(table[1].guest == 0x3000);
        CHECK(table[1].device == 0xC000 + 0x1000);
        CHECK(table[1].size == 0x1000);
    }
    SECTION("hundreds of small pools inside a chunk keep every one reachable") {
        using Vulkan::Nvn::FlattenRawGlobalRanges;
        using Vulkan::Nvn::RawGlobalSource;
        std::vector<RawGlobalSource> sources{{0x1000'0000, 0x5000'0000, 0x1000'0000, 0}};
        for (u64 i = 0; i < 300; ++i) {
            sources.push_back({0x1000'0000 + i * 0x10000, 0x9000'0000 + i * 0x20000, 0x4000, 1});
        }
        const std::vector<RawGlobalRange> table = FlattenRawGlobalRanges(sources);
        CHECK(table.size() == 600); // 300 pools + the chunk parts between and after them
        for (u64 i = 0; i < 300; ++i) {
            const auto it = std::ranges::find(table, 0x1000'0000 + i * 0x10000, &RawGlobalRange::guest);
            REQUIRE(it != table.end());
            CHECK(it->device == 0x9000'0000 + i * 0x20000);
            CHECK(it->size == 0x4000);
        }
    }
    SECTION("the driver bank header: count at 0xf00, table address at 0xf08, coverage ring at 0xf10") {
        Vulkan::Nvn::RawTableBuffer table;
        table.count = 7;
        table.address = 0x1234'5678'9ABC'DEF0;
        std::array<u8, 0x1000> bank{};
        Vulkan::Nvn::WriteRawGlobalTable(&table, 0, bank);
        u32 count = 0;
        std::memcpy(&count, bank.data() + 0xF00, sizeof(count));
        CHECK(count == 7);
        u64 address = 0;
        std::memcpy(&address, bank.data() + 0xF08, sizeof(address));
        CHECK(address == 0x1234'5678'9ABC'DEF0);
        static_assert(Shader::Profile::RAW_GLOBAL_TABLE_OFFSET == 0xF00);
        static_assert(Shader::Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET == 0xF08);
        static_assert(Shader::RawCoverage::BANK_RING_OFFSET == 0xF10);
        // Without the coverage diagnostic the rest of the bank is untouched (zero): the header
        // leaves the NVN words alone, among them the mesh stage's storage table (0x770..0x870),
        // which the old header at 0x800 overlapped.
        const auto zero_except_header = [&] {
            return std::all_of(bank.begin(), bank.begin() + 0xF00, [](u8 b) { return b == 0; }) &&
                   std::all_of(bank.begin() + 0xF10, bank.end(), [](u8 b) { return b == 0; });
        };
        CHECK(zero_except_header());
        // The NVN words below the header keep what they held, not only zeros.
        {
            std::array<u8, 0x1000> filled;
            filled.fill(0x5A);
            Vulkan::Nvn::WriteRawGlobalTable(&table, 0, filled);
            CHECK(std::all_of(filled.begin(), filled.begin() + 0xF00,
                              [](u8 b) { return b == 0x5A; }));
            u32 filled_count = 0;
            std::memcpy(&filled_count, filled.data() + 0xF00, sizeof(filled_count));
            CHECK(filled_count == 7);
        }
        // With it, the ring's device address follows the table's.
        Vulkan::Nvn::WriteRawGlobalTable(&table, 0xFEDC'BA98'7654'3210, bank);
        u64 ring = 0;
        std::memcpy(&ring, bank.data() + 0xF10, sizeof(ring));
        CHECK(ring == 0xFEDC'BA98'7654'3210);
        CHECK(std::all_of(bank.begin() + 0xF18, bank.end(), [](u8 b) { return b == 0; }));
        Vulkan::Nvn::WriteRawGlobalTable(nullptr, 0, bank);
        std::memcpy(&count, bank.data() + 0xF00, sizeof(count));
        std::memcpy(&address, bank.data() + 0xF08, sizeof(address));
        std::memcpy(&ring, bank.data() + 0xF10, sizeof(ring));
        CHECK(count == 0);
        CHECK(address == 0);
        CHECK(ring == 0);
        CHECK(zero_except_header());
    }
    SECTION("the read-only mirror table follows the ring: count 0xf18, address 0xf20, bounds "
            "0xf28/0xf30") {
        static_assert(Shader::Profile::RAW_GLOBAL_ALIAS_COUNT_OFFSET == 0xF18);
        static_assert(Shader::Profile::RAW_GLOBAL_ALIAS_ADDRESS_OFFSET == 0xF20);
        static_assert(Shader::Profile::RAW_GLOBAL_ALIAS_LOW_OFFSET == 0xF28);
        static_assert(Shader::Profile::RAW_GLOBAL_ALIAS_HIGH_OFFSET == 0xF30);
        Vulkan::Nvn::RawTableBuffer table;
        table.count = 7;
        table.address = 0x1234'5678'9ABC'DEF0;
        Vulkan::Nvn::RawTableBuffer mirrors;
        mirrors.count = 3;
        mirrors.address = 0x0BAD'F00D'0000'1000;
        std::array<u8, 0x1000> bank{};
        std::memset(bank.data() + 0xF18, 0xCD, 0x20); // stale words of an earlier bank
        Vulkan::Nvn::WriteRawGlobalTable(&table, 0, bank,
                                         {&mirrors, 0x5000'0000, 0x5003'0000});
        const auto u64_at = [&](u32 offset) {
            u64 value = 0;
            std::memcpy(&value, bank.data() + offset, sizeof(value));
            return value;
        };
        CHECK(u64_at(0xF18) == 3); // count, upper word zero
        CHECK(u64_at(0xF20) == 0x0BAD'F00D'0000'1000);
        CHECK(u64_at(0xF28) == 0x5000'0000);
        CHECK(u64_at(0xF30) == 0x5003'0000);
        CHECK(std::all_of(bank.begin() + 0xF38, bank.end(), [](u8 b) { return b == 0; }));
        // Without a table (or an empty one) every word is zero: no address is inside the bounds.
        Vulkan::Nvn::RawTableBuffer empty;
        Vulkan::Nvn::WriteRawGlobalTable(&table, 0, bank, {&empty, 0x5000'0000, 0x5003'0000});
        CHECK(std::all_of(bank.begin() + 0xF18, bank.end(), [](u8 b) { return b == 0; }));
        std::memset(bank.data() + 0xF18, 0xCD, 0x20);
        Vulkan::Nvn::WriteRawGlobalTable(&table, 0, bank);
        CHECK(std::all_of(bank.begin() + 0xF18, bank.end(), [](u8 b) { return b == 0; }));
    }
}

TEST_CASE("NVN raw coverage: a missed address is classified against the pools",
          "[nvn][raw_global][raw_coverage]") {
    using Vulkan::Nvn::ClassifyHleRawMiss;
    using Vulkan::Nvn::Pool;
    using VideoCommon::RawMissClass;
    std::map<u64, std::shared_ptr<Pool>> by_address;
    std::unordered_map<u64, std::shared_ptr<Pool>> by_serial;
    const auto add = [&](u64 serial, u64 address, u64 size, Pool::Backing backing) {
        auto pool = std::make_shared<Pool>();
        pool->serial = serial;
        pool->address = address;
        pool->size = size;
        pool->backing = backing;
        by_address[address] = pool;
        by_serial[serial] = pool;
        return pool;
    };
    // Pools far apart (more than the past-end reach): each miss is classified by its own pool.
    add(1, 0x10000, 0x8000, Pool::Backing::Imported);
    add(2, 0x1000000, 0x4000, Pool::Backing::Mirror);
    add(3, 0x2000000, 0x4000, Pool::Backing::Imported)->pending = true;
    add(4, 0x3000000, 0x2000, Pool::Backing::DeviceLocal);
    const auto virtual_pool = add(5, 0x4000000, 0x40000, Pool::Backing::Virtual);
    virtual_pool->virtual_runs[0x0] = {.size = 0x1000, .physical_serial = 2, .physical_offset = 0};
    virtual_pool->virtual_runs[0x2000] = {.size = 0x1000, .physical_serial = 4,
                                          .physical_offset = 0};
    virtual_pool->virtual_runs[0x20000] = {.size = 0x1000, .physical_serial = 99,
                                           .physical_offset = 0};
    // The table holds the first half of pool 1 and the virtual run onto pool 4.
    const std::vector<VideoCommon::RawGlobalRange> table{{0x10000, 0x7000'0000, 0x4000},
                                                         {0x4002000, 0x7100'0000, 0x1000}};
    const auto classify = [&](u64 guest) {
        return ClassifyHleRawMiss(guest, table, by_address, by_serial);
    };
    const auto text = [&](u64 guest) { return VideoCommon::RawMissKindText(classify(guest)); };

    CHECK(classify(0x10).kind == RawMissClass::Null);
    CHECK(classify(0x10010).kind == RawMissClass::Late); // the table holds it by now
    CHECK(classify(0x4002010).kind == RawMissClass::Late);
    CHECK(text(0x14010) == "past-end(+0x10 of 0x10000+0x4000)"); // in pool 1, after its range
    CHECK(classify(0x900000).kind == RawMissClass::Unmapped); // no pool
    // Pools the table leaves out, by why.
    CHECK(text(0x1000010) == "mapped-gap(mirror pool 0x1000000+0x4000)");
    CHECK(text(0x2000010) == "mapped-gap(pending pool 0x2000000+0x4000)");
    CHECK(text(0x3000010) == "mapped-gap(device-local pool 0x3000000+0x2000)");
    // A virtual pool: mapped only by its runs, named by the physical pool behind the run.
    CHECK(text(0x4000010) == "mapped-gap(virtual->mirror pool 0x4000000+0x40000)");
    CHECK(classify(0x4001010).kind == RawMissClass::Unmapped); // a hole between runs
    CHECK(text(0x4020010) == "mapped-gap(virtual->destroyed pool 0x4000000+0x40000)");
}
