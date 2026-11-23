// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// NVN2 compute: the recompiler's driver bank window, the c0 builder and (with a Vulkan device)
// dispatch execution.

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/object_pool.h"
#include "video_core/nvn/shader/nvn_compute_bank.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

using namespace VideoCommon::Nvn;

// ---- Hand-assembled Maxwell programs (the form the recompiler consumes) -------------------------

constexpr u64 MovCbuf(u32 rd, u32 bank, u32 byte_offset) {
    return 0x4c98000000000000ULL | (u64{0xf} << 39) | (u64{bank} << 34) |
           (u64{byte_offset / 4} << 20) | rd;
}
constexpr u64 Mov32I(u32 rd, u32 imm) {
    return 0x0100000000000000ULL | (u64{imm} << 20) | (u64{0xf} << 12) | rd;
}
/// IADD3 rd, ra, c[bank][offset], rc
constexpr u64 Iadd3Cbuf(u32 rd, u32 ra, u32 bank, u32 byte_offset, u32 rc) {
    return 0x4cc0000000000000ULL | (u64{rc} << 39) | (u64{bank} << 34) |
           (u64{byte_offset / 4} << 20) | (u64{ra} << 8) | rd;
}
/// LDG.E.32 rd, [ra + offset]
constexpr u64 Ldg32(u32 rd, u32 ra, u32 offset) {
    return 0xeed4000000000000ULL | (u64{1} << 45) | (u64{offset} << 20) | (u64{ra} << 8) | rd;
}
/// STG.E.32 [ra + offset], rs
constexpr u64 Stg32(u32 ra, u32 offset, u32 rs) {
    return 0xeedc000000000000ULL | (u64{1} << 45) | (u64{offset} << 20) | (u64{ra} << 8) | rs;
}
constexpr u64 Exit = 0xe30000000007000fULL;

/// [0x50-byte SPH][instructions with a scheduling word before every third] as the NVN lowering
/// packs a compute program.
std::vector<u64> Pack(std::span<const u64> program) {
    std::vector<u64> code(SphBytes / sizeof(u64), 0);
    for (const u64 instruction : program) {
        if ((code.size() * sizeof(u64)) % 32 == 0) {
            code.push_back(0);
        }
        code.push_back(instruction);
    }
    code.push_back(0);
    return code;
}

class ComputeEnvironment final : public Shader::Environment {
public:
    ComputeEnvironment(std::span<const u64> program, Shader::NvnBufferWindow window_ = {})
        : code{Pack(program)}, window{window_} {
        stage = Shader::Stage::Compute;
        start_address = SphBytes;
        is_proprietary_driver = true;
    }

    u64 ReadInstruction(u32 address) override {
        REQUIRE(address / 8 < code.size());
        return code[address / 8];
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
        return {32, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    Shader::NvnBufferWindow NvnComputeBufferWindow() const override {
        return window;
    }
    void Dump(u64, u64) override {}

private:
    std::vector<u64> code;
    Shader::NvnBufferWindow window;
};

struct Translated {
    NvnShaderPools pools;
    Shader::IR::Program program;
};

std::unique_ptr<Translated> Translate(ComputeEnvironment& env) {
    auto result = std::make_unique<Translated>();
    Shader::Maxwell::Flow::CFG cfg(env, result->pools.flow_block, env.StartAddress(), false);
    result->program = Shader::Maxwell::TranslateProgram(result->pools.inst, result->pools.block,
                                                        env, cfg, MakeReferenceHostInfo());
    return result;
}

std::vector<u32> Emit(Translated& translated) {
    static const bool logging = [] {
        // A failed recompiler assertion aborts: keep its message.
        const std::filesystem::path log_dir =
            std::filesystem::temp_directory_path() / "eden-nvn-compute-test-log";
        std::filesystem::create_directories(log_dir);
        Common::FS::SetEdenPath(Common::FS::EdenPath::LogDir, log_dir);
        Common::Log::Initialize();
        Common::Log::SetGlobalFilter(Common::Log::Filter{Common::Log::Level::Error});
        Common::Log::SetColorConsoleBackendEnabled(true);
        Common::Log::Start();
        return true;
    }();
    (void)logging;
    // The fallback global memory functions read the driver bank through aliased uniform
    // definitions, which only exist with descriptor aliasing (every desktop profile has it).
    Shader::Profile profile = MakeReferenceProfile();
    profile.support_descriptor_aliasing = true;
    return Shader::Backend::SPIRV::EmitSPIRV(profile, translated.program);
}

bool HasStorageDescriptor(const Shader::Info& info, u32 bank, u32 offset) {
    return std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& desc) {
        return desc.cbuf_index == bank && desc.cbuf_offset == offset;
    });
}

/// Global memory accessed through an address the recompiler cannot trace to a c0 word: only the
/// driver bank window decides which c0 entries the fallback functions cover.
/// R4/R5 = a constant address; R6 = c0[entry_offset] (kept alive by the store).
std::array<u64, 5> UntrackedProgram(u32 entry_offset) {
    return {Mov32I(4, 0x1000), Mov32I(5, 0), MovCbuf(6, 0, entry_offset), Stg32(4, 0, 6), Exit};
}

} // namespace

TEST_CASE("NVN compute window replaces the fixed 0x310 base", "[nvn][compute]") {
    // Uniform entries 0x10 and 0x20, storage entry 0x30.
    const Shader::NvnBufferWindow window{0x10, 3, true};

    SECTION("an entry read inside the window is a used buffer") {
        const auto program = UntrackedProgram(0x20);
        ComputeEnvironment env{program, window};
        auto translated = Translate(env);
        const Shader::Info& info = translated->program.info;
        CHECK(info.nvn_buffer_window);
        CHECK(info.nvn_buffer_base == 0x10);
        CHECK(info.nvn_buffer_count == 3);
        CHECK(info.nvn_buffer_used.count() == 1);
        CHECK(info.nvn_buffer_used[1]);
        CHECK(info.uses_global_memory);
        CHECK(HasStorageDescriptor(info, 0, 0x20));
        const auto spirv = Emit(*translated);
        REQUIRE(spirv.size() > 5);
        CHECK(spirv[0] == 0x07230203);
    }
    SECTION("the legacy base does not see the same read") {
        const auto program = UntrackedProgram(0x20);
        ComputeEnvironment env{program};
        auto translated = Translate(env);
        const Shader::Info& info = translated->program.info;
        CHECK_FALSE(info.nvn_buffer_window);
        CHECK(info.nvn_buffer_base == 0x310);
        CHECK(info.nvn_buffer_count == 16);
        CHECK(info.nvn_buffer_used.none());
        CHECK_FALSE(HasStorageDescriptor(info, 0, 0x20));
    }
    SECTION("a read past the window is not an entry") {
        const auto program = UntrackedProgram(0x40);
        ComputeEnvironment env{program, window};
        auto translated = Translate(env);
        CHECK(translated->program.info.nvn_buffer_used.none());
    }
    SECTION("a program without buffer entries has an empty active window") {
        const auto program = UntrackedProgram(0x320);
        ComputeEnvironment env{program, Shader::NvnBufferWindow{0, 0, true}};
        auto translated = Translate(env);
        const Shader::Info& info = translated->program.info;
        CHECK(info.nvn_buffer_window);
        CHECK(info.nvn_buffer_used.none()); // 0x320 is no entry of an empty window
    }
}
TEST_CASE("NVN driver window preserves colliding storage slots across banks",
          "[nvn][compute][storage]") {
    // c0[0x20] is a driver entry; c3[0x20] is a different unbiased descriptor with a colliding offset.
    const std::array program{Mov32I(4, 0x1000), Mov32I(5, 0),
                             MovCbuf(6, 0, 0x20), Stg32(4, 0, 6),
                             MovCbuf(4, 3, 0x20), MovCbuf(5, 3, 0x24),
                             Ldg32(0, 4, 0), Exit};
    ComputeEnvironment env{program, Shader::NvnBufferWindow{0x10, 3, true}};
    auto translated = Translate(env);
    const Shader::Info& info = translated->program.info;
    CHECK(HasStorageDescriptor(info, 3, 0x20));
    CHECK(HasStorageDescriptor(info, 0, 0x20));
}


TEST_CASE("NVN compute tracks storage buffers inside the window", "[nvn][compute]") {
    // R4 = c0[0x20] + c1[0x8] (address low), R5 = c0[0x24]; load, then store to the same buffer.
    const std::array program{MovCbuf(2, 1, 0x8),
                             Iadd3Cbuf(4, 2, 0, 0x20, 255),
                             MovCbuf(5, 0, 0x24),
                             Ldg32(0, 4, 0),
                             Stg32(4, 4, 0),
                             Exit};
    ComputeEnvironment env{program, Shader::NvnBufferWindow{0x10, 3, true}};
    auto translated = Translate(env);
    const Shader::Info& info = translated->program.info;
    REQUIRE(info.storage_buffers_descriptors.size() >= 1);
    // The descriptor is the driver bank entry, never the program constant bank.
    CHECK(HasStorageDescriptor(info, 0, 0x20));
    CHECK_FALSE(std::ranges::any_of(info.storage_buffers_descriptors,
                                    [](const auto& desc) { return desc.cbuf_index != 0; }));
    const auto spirv = Emit(*translated);
    REQUIRE(spirv.size() > 5);
}

TEST_CASE("NVN compute window with the legacy values sees the same buffers", "[nvn][compute]") {
    // Entry 1 of the fixed compute table (0x320). Analysis (Info) agrees between the legacy and
    // the window path; only the emitted fallback differs, because the legacy emitter pairs bit k of
    // the used set with descriptor k (compute_legacy.cpp pins that output), the window emitter
    // pairs them by c0 offset.
    const auto program = UntrackedProgram(0x320);
    ComputeEnvironment legacy_env{program};
    ComputeEnvironment window_env{program, Shader::NvnBufferWindow{0x310, 16, true}};
    auto legacy = Translate(legacy_env);
    auto windowed = Translate(window_env);
    const Shader::Info& a = legacy->program.info;
    const Shader::Info& b = windowed->program.info;
    CHECK_FALSE(a.nvn_buffer_window);
    CHECK(b.nvn_buffer_window);
    CHECK(a.nvn_buffer_base == 0x310);
    CHECK(a.nvn_buffer_count == 16);
    CHECK(a.nvn_buffer_used[1]);
    CHECK(a.nvn_buffer_used.count() == 1);
    CHECK(HasStorageDescriptor(a, 0, 0x320));
    CHECK(b.nvn_buffer_base == a.nvn_buffer_base);
    CHECK(b.nvn_buffer_count == a.nvn_buffer_count);
    CHECK(b.nvn_buffer_used == a.nvn_buffer_used);
    CHECK(b.storage_buffers_descriptors == a.storage_buffers_descriptors);
    CHECK(Emit(*legacy).size() > 5);
    CHECK(Emit(*windowed).size() > 5);
}

TEST_CASE("NVN compute driver bank follows the control-derived layout", "[nvn][compute]") {
    // 3 push words at 0x10, then from the 0x10 boundary 2 uniform entries, 2 storage entries,
    // 1 texture, 1 image (what ParseControl derives from those counts).
    ComputeDriverLayout layout;
    layout.push_constant_words = 3;
    layout.uniform_offset = 0x20;
    layout.uniform_count = 2;
    layout.storage_offset = 0x40;
    layout.storage_count = 2;
    layout.texture_offset = 0x60;
    layout.texture_count = 1;
    layout.image_offset = 0x68;
    layout.image_count = 1;
    layout.end_offset = 0x70;
    REQUIRE(ComputeBankSize(layout) == 0x70);

    const std::array<BufferRange, 2> uniform{{{0x1'0000'1000, 0x40}, {0}}};
    const std::array<BufferRange, 3> storage{{{0x2'0000'0000, 0x100}, {0x3000, 0x8}, {0x4000, 0x4}}};
    const std::array<u8, 16> push{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const std::array<u64, 1> textures{0xffff00300007};
    const std::array<u64, 2> images{0x11, 0x22};
    const auto bank = BuildComputeDriverBank(
        layout, ComputeBankInputs{uniform, storage, push, textures, images, {7, 8, 9}});
    REQUIRE(bank.size() == 0x70);

    const auto u32_at = [&](u32 offset) {
        u32 value;
        std::memcpy(&value, bank.data() + offset, sizeof(value));
        return value;
    };
    const auto u64_at = [&](u32 offset) {
        u64 value;
        std::memcpy(&value, bank.data() + offset, sizeof(value));
        return value;
    };
    // Header (DK's legacy cbuf 0): the bank size and the group counts of the dispatch.
    CHECK(u32_at(0x00) == 0x70);
    CHECK(u32_at(0x04) == 7);
    CHECK(u32_at(0x08) == 8);
    CHECK(u32_at(0x0c) == 9);
    // Push constants: only the 3 words the layout declares, the rest of the block is dropped.
    CHECK(u32_at(0x10) == 0x04030201);
    CHECK(u32_at(0x14) == 0x08070605);
    CHECK(u32_at(0x18) == 0x0c0b0a09);
    CHECK(u32_at(0x1c) == 0); // alignment gap before the entry tables
    CHECK(u64_at(0x20) == 0x1'0000'1000);
    CHECK(u32_at(0x28) == 0x40);
    CHECK(u64_at(0x30) == 0); // unbound uniform slot 1
    // Storage slot 2 exists in the inputs but not in the layout: not written.
    CHECK(u64_at(0x40) == 0x2'0000'0000);
    CHECK(u32_at(0x48) == 0x100);
    CHECK(u64_at(0x50) == 0x3000);
    CHECK(u32_at(0x58) == 0x8);
    CHECK(u32_at(0x60) == 0x00300007);
    CHECK(u32_at(0x68) == 0x11); // one image slot only
}
