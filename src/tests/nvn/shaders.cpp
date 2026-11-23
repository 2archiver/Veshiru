// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Game-free tests of the NVN2 shader mapping (video_core/nvn/shader): control section parsing,
// handle classification, lowering/packing parity with the in-game path, translation through the
// NVN environment, descriptor numbering and SPIR-V emission.

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/nvn/shader/nvn_descriptor_layout.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"
#include "video_core/nvn/shader/nvn_handle_program.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

using Shader::Ampere::Instruction;
using namespace VideoCommon::Nvn;

constexpr u32 RoundUp(u32 value, u32 alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void Put32(std::vector<u8>& bytes, size_t offset, u32 value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

u32 SphType(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return 1;
    case ShaderStage::TessControl:
        return 2;
    case ShaderStage::TessEval:
        return 3;
    case ShaderStage::Geometry:
        return 4;
    default:
        return 5;
    }
}

struct Binary {
    std::vector<u8> control;
    std::vector<u8> code;

    [[nodiscard]] NvnProgramBinary View(ShaderStage stage) const {
        return {stage, control, code};
    }
};

/// A synthetic NVN2 control section of `size` bytes.
std::vector<u8> MakeControl(u32 stage_byte, u32 inst_bytes, u32 c1_offset, u32 code_bytes,
                            size_t size = 0x100) {
    std::vector<u8> control(size, 0);
    Put32(control, 0, ControlMagic);
    Put32(control, 4, ControlVersion);
    Put32(control, 0x38, inst_bytes);
    Put32(control, 0x40, c1_offset);
    Put32(control, 0x44, code_bytes);
    control[0x4c] = 12;
    control[0x56] = static_cast<u8>(stage_byte);
    Put32(control, 0x58, 8);
    Put32(control, 0x5c, 4);
    Put32(control, 0x60, 2);
    return control;
}

Binary MakeBinary(ShaderStage stage, std::span<const Instruction> program,
                  std::span<const u32> c1 = {}) {
    const bool compute = stage == ShaderStage::Compute;
    const u32 header = compute ? 0 : GraphicsHeaderBytes;
    const u32 inst_bytes = header + static_cast<u32>(program.size() * sizeof(Instruction));
    const u32 c1_offset = RoundUp(inst_bytes, 0x100);
    const u32 code_bytes = c1_offset + static_cast<u32>(c1.size() * sizeof(u32));
    Binary binary;
    binary.control = MakeControl(static_cast<u32>(stage), inst_bytes, c1_offset, code_bytes);
    binary.code.assign(code_bytes, 0);
    if (!compute) {
        Put32(binary.code, 0, SphType(stage) << 10);
    }
    std::memcpy(binary.code.data() + header, program.data(), program.size_bytes());
    if (!c1.empty()) {
        std::memcpy(binary.code.data() + c1_offset, c1.data(), c1.size_bytes());
    }
    return binary;
}

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

// The captured program of ampere_bindless_tex.cpp: two bindless TEX with separately merged
// handles, c2[0x168] / c2[0x170] each ORed with c2[0x568].
constexpr std::array<Instruction, 13> TwoTextureProgram{{
    {{0x00067ab9, 0x00815a00, 0x00000800, 0x000fe400}},
    {{0x00047ab9, 0x00805a00, 0x00000800, 0x000fe400}},
    {{0x00077ab9, 0x00805c00, 0x00000800, 0x000fe400}},
    {{0x06047292, 0x00000004, 0x0f8efc3f, 0x000fe400}},
    {{0x06067292, 0x00000007, 0x0f8efc3f, 0x000fc400}},
    {{0x3f057899, 0x00000014, 0x08011604, 0x000fe400}},
    {{0x3f077899, 0x00000014, 0x08011606, 0x000fe200}},
    {{0x00007308, 0x00000000, 0x00001000, 0x001e6400}},
    {{0x00047220, 0x00000001, 0x0041c000, 0x042fe200}},
    {{0x00057220, 0x00000005, 0x0041c000, 0x004fca00}},
    {{0x04087f60, 0x30000605, 0x081e03ff, 0x100f4200}},
    {{0x04067f60, 0x30000405, 0x081e01ff, 0x000f4200}},
    Exit,
}};

} // namespace

TEST_CASE("NVN control section parsing", "[nvn][shaders]") {
    std::string error;
    SECTION("fields of a well-formed graphics control") {
        const auto control = MakeControl(1, 0x180, 0x200, 0x240);
        const auto parsed = ParseControl(control, error);
        REQUIRE(parsed.has_value());
        CHECK(parsed->inst_bytes == 0x180);
        CHECK(parsed->c1_offset == 0x200);
        CHECK(parsed->code_bytes == 0x240);
        CHECK(parsed->register_count == 12);
        CHECK(parsed->stage_byte == 1);
        CHECK(parsed->thread_group == std::array<u32, 3>{8, 4, 2});
        CHECK_FALSE(parsed->has_compute_counts);
    }
    SECTION("compute counts derive the driver layout") {
        auto control = MakeControl(5, 0x100, 0x100, 0x100);
        control[0xa9] = 3;  // push constant words
        control[0xb1] = 2;  // uniform entries
        control[0xb2] = 4;  // storage entries
        control[0xb3] = 1;  // texture handles
        control[0xb6] = 2;  // image handles
        Put32(control, 0x64, 0x2a80); // shared memory bytes
        // The entry tables start at the 0x10 boundary after the 3 push words (0x1c -> 0x20).
        const u32 end = 0x20 + 2 * 0x10 + 4 * 0x10 + 8 + 2 * 8;
        control[0xba] = static_cast<u8>(end & 0xFF);
        control[0xbb] = static_cast<u8>(end >> 8);
        const auto parsed = ParseControl(control, error);
        REQUIRE(parsed.has_value());
        REQUIRE(parsed->has_compute_counts);
        const auto& layout = parsed->compute;
        CHECK(layout.uniform_offset == 0x20);
        CHECK(layout.storage_offset == 0x40);
        CHECK(layout.texture_offset == 0x80);
        CHECK(layout.image_offset == 0x88);
        CHECK(layout.end_offset == end);
        CHECK(layout.total_matches);
        const auto tables = layout.HandleTables();
        CHECK(tables.texture_offset == 0x80);
        CHECK(tables.image_count == 2);
        CHECK(parsed->shared_memory_bytes == 0x2a80);
    }
    SECTION("odd push-constant blocks: the declared totals of Ocarina of Time's shaders") {
        struct Case {
            u8 push_words, uniforms, storages, textures, images;
            u32 declared_total, texture_offset;
        };
        // Counts and declared totals (control +0xa9/+0xb1/+0xb2/+0xb3/+0xb6/+0xba) of the two
        // OoT compute shaders with an odd push-constant block.
        for (const Case& c : {Case{15, 0, 0, 3, 1, 0x70, 0x50},
                              Case{29, 11, 9, 7, 2, 0x218, 0x1d0}}) {
            auto control = MakeControl(5, 0x100, 0x100, 0x100);
            control[0xa9] = c.push_words;
            control[0xb1] = c.uniforms;
            control[0xb2] = c.storages;
            control[0xb3] = c.textures;
            control[0xb6] = c.images;
            control[0xba] = static_cast<u8>(c.declared_total & 0xFF);
            control[0xbb] = static_cast<u8>(c.declared_total >> 8);
            const auto parsed = ParseControl(control, error);
            REQUIRE(parsed.has_value());
            const auto& layout = parsed->compute;
            CHECK(layout.uniform_offset % 0x10 == 0);
            CHECK(layout.texture_offset == c.texture_offset);
            CHECK(layout.end_offset == c.declared_total);
            CHECK(layout.total_matches);
        }
    }
    SECTION("handles of the 15-word shader classify at the aligned table offsets") {
        auto control = MakeControl(5, 0x100, 0x100, 0x100);
        control[0xa9] = 15;
        control[0xb3] = 3;
        control[0xb6] = 1;
        const auto parsed = ParseControl(control, error);
        REQUIRE(parsed.has_value());
        const auto tables = parsed->compute.HandleTables();
        const auto texture1 = ClassifyHandleWord(DriverBank, 0x58, HandleWant::Texture, &tables);
        CHECK(texture1.source == HandleSource::ComputeTexture);
        CHECK(texture1.slot == 1);
        const auto texture2 = ClassifyHandleWord(DriverBank, 0x60, HandleWant::Texture, &tables);
        CHECK(texture2.source == HandleSource::ComputeTexture);
        CHECK(texture2.slot == 2);
        const auto image0 = ClassifyHandleWord(DriverBank, 0x68, HandleWant::Image, &tables);
        CHECK(image0.source == HandleSource::ComputeImage);
        CHECK(image0.slot == 0);
    }
    SECTION("errors are reported, not thrown") {
        CHECK_FALSE(ParseControl(std::vector<u8>(0x20, 0), error));
        CHECK(error.find("control section is 32 bytes") != std::string::npos);

        auto control = MakeControl(1, 0x180, 0x200, 0x240);
        Put32(control, 0, 0x1234);
        CHECK_FALSE(ParseControl(control, error));
        CHECK(error.find("bad control magic") != std::string::npos);

        control = MakeControl(1, 0x180, 0x200, 0x240);
        Put32(control, 4, 0xc8);
        CHECK_FALSE(ParseControl(control, error));
        CHECK(error.find("unsupported control version") != std::string::npos);

        control = MakeControl(9, 0x180, 0x200, 0x240);
        CHECK_FALSE(ParseControl(control, error));
        CHECK(error.find("bad stage byte") != std::string::npos);

        control = MakeControl(1, 0x180, 0x300, 0x240);
        CHECK_FALSE(ParseControl(control, error));
        CHECK(error.find("c1 offset") != std::string::npos);

        control = MakeControl(1, 0x280, 0x200, 0x240);
        CHECK_FALSE(ParseControl(control, error));
        CHECK(error.find("instruction bytes") != std::string::npos);

        control = MakeControl(5, 0x100, 0x100, 0x100, 0xa0);
        CHECK_FALSE(ParseControl(control, error));
        CHECK(error.find("compute control section") != std::string::npos);
    }
}

TEST_CASE("NVN compute local memory comes from the control section", "[nvn][shaders]") {
    // A control with 0x2000 bytes of local memory per warp at +0x50 belongs to a compute shader
    // whose threads have 0x100 bytes each (Kirby Air Riders: its STLs store at 0x0..0xf0 in 16
    // byte steps). The environment used to report 0, which dropped every STL and left the LDLs
    // without a local memory variable.
    const std::array<Instruction, 1> program{Exit};
    auto binary = MakeBinary(ShaderStage::Compute, program);
    Put32(binary.control, 0x50, 0x2000);
    std::string error;
    const auto parsed = ParseControl(binary.control, error);
    REQUIRE(parsed.has_value());
    CHECK(parsed->local_memory_per_warp_bytes == 0x2000);
    CHECK(parsed->LocalMemoryBytes() == 0x100);

    const auto lowered = LowerNvnProgram(binary.View(ShaderStage::Compute));
    REQUIRE(lowered->error.empty());
    CHECK(NvnEnvironment{lowered}.LocalMemorySize() == 0x100);
    // The caller can still override the control's value.
    NvnEnvironmentOptions options;
    options.compute_local_memory_size = 0x40;
    CHECK(NvnEnvironment{lowered, options}.LocalMemorySize() == 0x40);

    // A shader without local memory has none, and the declared size reaches the IR program.
    auto none = MakeBinary(ShaderStage::Compute, program);
    const auto lowered_none = LowerNvnProgram(none.View(ShaderStage::Compute));
    REQUIRE(lowered_none->error.empty());
    CHECK(NvnEnvironment{lowered_none}.LocalMemorySize() == 0);
    NvnShaderPools pools;
    NvnEnvironment env{lowered};
    const auto translated = TranslateNvnStage(pools, env, MakeReferenceHostInfo());
    CHECK(translated.local_memory_size == 0x100);
}

TEST_CASE("NVN handle word classification", "[nvn][shaders]") {
    struct Case {
        u32 bank;
        u32 offset;
        HandleWant want;
        HandleSource source;
        u32 slot;
    };
    using S = HandleSource;
    constexpr auto T = HandleWant::Texture;
    constexpr auto I = HandleWant::Image;
    const Case cases[] = {
        {2, 0x20, T, S::Combined, 0},
        {2, 0x118, T, S::Combined, 31},
        {2, 0x120, T, S::Combined, 32},     // a texture descriptor there is combined slot 32+n
        {2, 0x120, I, S::Image, 0},         // an image descriptor there is image slot n
        {2, 0x158, I, S::Image, 7},
        {2, 0x160, T, S::Combined, 40},
        {2, 0x160, I, S::Unsupported, 0},   // only 8 image slots
        {2, 0x118, I, S::Unsupported, 0},
        {2, 0x167, T, S::Unsupported, 0},   // misaligned
        {2, 0x164, T, S::Unsupported, 0},
        {2, 0x168, T, S::SeparateTexture, 0},
        {2, 0x168, I, S::Unsupported, 0},
        {2, 0x560, T, S::SeparateTexture, 127},
        {2, 0x568, T, S::SeparateSampler, 0},
        {2, 0x960, T, S::SeparateSampler, 127},
        {2, 0x968, T, S::Unsupported, 0},   // past the handle bank
        {2, 0x18, T, S::Unsupported, 0},    // before the tables
        {0, 0x20, T, S::Unsupported, 0},    // driver bank
        {1, 0x20, T, S::Unsupported, 0},    // program bank
        {3, 0x40, T, S::UniformBuffer, 0},
        {7, 0x0, I, S::UniformBuffer, 4},
    };
    for (const Case& c : cases) {
        INFO("bank " << c.bank << " offset " << std::hex << c.offset);
        const auto result = ClassifyHandleWord(c.bank, c.offset, c.want);
        CHECK(result.source == c.source);
        if (c.source != S::Unsupported) {
            CHECK(result.slot == c.slot);
            CHECK(result.reason.empty());
        } else {
            CHECK_FALSE(result.reason.empty());
        }
    }
    // Compute reads its handle tables from the per-shader driver bank.
    const ComputeHandleTables tables{0x7c, 2, 0x8c, 3};
    CHECK(ClassifyHandleWord(0, 0x84, T, &tables).source == S::ComputeTexture);
    CHECK(ClassifyHandleWord(0, 0x84, T, &tables).slot == 1);
    CHECK(ClassifyHandleWord(0, 0x80, T, &tables).source == S::Unsupported); // misaligned in table
    CHECK(ClassifyHandleWord(0, 0x94, I, &tables).source == S::ComputeImage);
    CHECK(ClassifyHandleWord(0, 0x94, I, &tables).slot == 1);
    // A handle is a pool index whichever table holds it: texture ops on an image-table word and
    // image ops on a texture-table word resolve to the table that holds the word.
    CHECK(ClassifyHandleWord(0, 0x94, T, &tables).source == S::ComputeImage);
    CHECK(ClassifyHandleWord(0, 0x94, T, &tables).slot == 1);
    CHECK(ClassifyHandleWord(0, 0x84, I, &tables).source == S::ComputeTexture);
    // Array elements: the next slot of a handle table, the next word of the same uniform buffer.
    CHECK(HandleElement({S::Combined, 2, HandleBank, 0x30}, 3) ==
          HandleRef{S::Combined, 5, HandleBank, 0x48});
    CHECK(HandleElement({S::UniformBuffer, 6, 9, 0x300}, 2) ==
          HandleRef{S::UniformBuffer, 6, 9, 0x310});
    CHECK(ClassifyHandleWord(0, 0x98, I, &tables).source == S::Unsupported); // misaligned
    CHECK(ClassifyHandleWord(0, 0xa4, I, &tables).source == S::Unsupported); // past the table
    // Handle decoding.
    CHECK(TextureIdOf(0x123abcde) == 0xabcde);
    CHECK(SamplerIdOf(0x123abcde) == 0x123);
    CHECK(MakeHandle(0xabcde, 0x123) == 0x123abcde);
    CHECK(SeparateSamplerWord(0x123) == 0x12300000);
}

TEST_CASE("NVN handle words are rebased into the global id space", "[nvn][shaders]") {
    constexpr u32 combined = 0x0050'0007; // sampler 5, texture 7
    // A combined word moves both ids; the pool bases are those of the bound pools.
    CHECK(RebaseHandleWord(combined, HandleWordKind::Texture, 100, 20) == 0x0190'006b);
    // Id 0 means "none" and stays 0.
    CHECK(RebaseHandleWord(0x0050'0000, HandleWordKind::Texture, 100, 20) == 0x0190'0000);
    CHECK(RebaseHandleWord(0x0000'0007, HandleWordKind::Texture, 100, 20) == 0x0000'006b);
    CHECK(RebaseHandleWord(0, HandleWordKind::Texture, 100, 20) == 0);
    // A separate sampler word only has a sampler id; a separate texture word only a texture id.
    CHECK(RebaseHandleWord(0x0050'0000, HandleWordKind::SeparateSampler, 100, 20) == 0x0190'0000);
    CHECK(RebaseHandleWord(0x0000'0007, HandleWordKind::SeparateSampler, 100, 20) == 7);
    // An image word keeps bits 20.. as they are.
    CHECK(RebaseHandleWord(combined, HandleWordKind::Image, 100, 20) == 0x0050'006b);
    // Base 0 changes nothing.
    CHECK(RebaseHandleWord(combined, HandleWordKind::Texture, 0, 0) == combined);
}

TEST_CASE("NVN storage buffer slot math", "[nvn][shaders]") {
    struct Base {
        ShaderStage stage;
        u32 base;
    };
    const Base bases[] = {{ShaderStage::Vertex, 0x110},      {ShaderStage::TessControl, 0x210},
                          {ShaderStage::TessEval, 0x310},    {ShaderStage::Geometry, 0x410},
                          {ShaderStage::Fragment, 0x510},    {ShaderStage::Task, 0x670},
                          {ShaderStage::Mesh, 0x770},        {ShaderStage::Compute, 0x310}};
    for (const Base& entry : bases) {
        INFO(StageName(entry.stage));
        CHECK(StorageBufferBase(entry.stage) == entry.base);
        CHECK(StorageBufferSlot(entry.stage, entry.base) == 0u);
        CHECK(StorageBufferSlot(entry.stage, entry.base + 0x10) == 1u);
        CHECK(StorageBufferSlot(entry.stage, entry.base + 15 * 0x10) == 15u);
        CHECK_FALSE(StorageBufferSlot(entry.stage, entry.base + 16 * 0x10).has_value());
        CHECK_FALSE(StorageBufferSlot(entry.stage, entry.base - 0x10).has_value());
        CHECK_FALSE(StorageBufferSlot(entry.stage, entry.base + 8).has_value());
    }
    CHECK(StorageBufferSlot(0x40, 0x70) == 3u);
}

TEST_CASE("NVN lowering packs exactly like LowerNativeCode", "[nvn][shaders]") {
    SECTION("EXIT-only program") {
        const std::array<Instruction, 1> program{Exit};
        const auto binary = MakeBinary(ShaderStage::Vertex, program);
        const auto lowered = LowerNvnProgram(binary.View(ShaderStage::Vertex));
        REQUIRE(lowered->error.empty());
        CHECK(lowered->capture_complete);
        CHECK(lowered->texture_bound == ~0U);
        REQUIRE(lowered->code.size() >= 0x50 / 8 + 1);
        const auto reference = Shader::Ampere::LowerToMaxwell(program);
        REQUIRE(static_cast<bool>(reference));
        // The first instruction follows the 0x50-byte SPH.
        CHECK(lowered->code[0x50 / 8] == reference.instructions[0]);
        CHECK(lowered->source_size == GraphicsHeaderBytes + reference.source_size);
        CHECK(Shader::Ampere::HasNativeAddressMap(lowered->code));
        CHECK(Shader::Ampere::NativeProgramAddress(lowered->code, 0) == 0x50u);
        CHECK_FALSE(Shader::Ampere::NativeProgramAddress(lowered->code, 16).has_value());
        // [sph][instructions][0][address map][count][magic]
        const size_t expected = 0x50 / 8 + reference.instructions.size() + 1 +
                                reference.source_instruction_offsets.size() + 2;
        CHECK(lowered->code.size() == expected);
    }
    SECTION("scheduling words and the address map of a longer program") {
        const auto binary = MakeBinary(ShaderStage::Fragment, TwoTextureProgram);
        const auto lowered = LowerNvnProgram(binary.View(ShaderStage::Fragment));
        REQUIRE(lowered->error.empty());
        const auto reference = Shader::Ampere::LowerToMaxwell(TwoTextureProgram);
        REQUIRE(static_cast<bool>(reference));
        for (size_t i = 0; i < reference.instructions.size(); ++i) {
            const size_t address = 0x50 + i * 8 + ((i + 1) / 3) * 8;
            REQUIRE(address / 8 < lowered->code.size());
            CHECK(lowered->code[address / 8] == reference.instructions[i]);
            CHECK(address % 32 != 0); // never lands on a scheduling word
        }
        for (size_t native = 0; native < reference.source_instruction_offsets.size(); ++native) {
            const u32 index = reference.source_instruction_offsets[native];
            const u32 address = 0x50 + index * 8 + ((index + 1) / 3) * 8;
            CHECK(Shader::Ampere::NativeProgramAddress(lowered->code,
                                                       static_cast<u32>(native * 16)) == address);
        }
        // Every 32-byte group starts with a zero scheduling word, up to the last instruction.
        const size_t count = reference.instructions.size();
        const size_t last_address = 0x50 + (count - 1) * 8 + (count / 3) * 8;
        for (size_t address = 0x60; address <= last_address; address += 32) {
            CHECK(lowered->code[address / 8] == 0);
        }
    }
    SECTION("c1 is the code area past the instructions, rounded to 0x100") {
        const std::array<Instruction, 1> program{Exit};
        const std::array<u32, 4> c1{0x11111111, 0x22222222, 0x33333333, 0x44444444};
        const auto binary = MakeBinary(ShaderStage::Fragment, program, c1);
        const auto lowered = LowerNvnProgram(binary.View(ShaderStage::Fragment));
        REQUIRE(lowered->error.empty());
        CHECK(lowered->c1 == std::vector<u32>(c1.begin(), c1.end()));
    }
    SECTION("errors carry a category and never throw") {
        const std::array<Instruction, 1> program{Exit};
        auto binary = MakeBinary(ShaderStage::Fragment, program);
        // A stage that does not match the control's stage byte.
        CHECK(LowerNvnProgram(binary.View(ShaderStage::Compute))->error.find("control:") == 0);
        // Code area size not matching the control.
        binary.code.push_back(0);
        CHECK(LowerNvnProgram(binary.View(ShaderStage::Fragment))->error.find("control:") == 0);
        binary.code.pop_back();
        // Truncated control.
        binary.control.resize(0x30);
        CHECK(LowerNvnProgram(binary.View(ShaderStage::Fragment))->error.find("control:") == 0);
        // A branch leaving the captured program.
        const std::array<Instruction, 2> bad{{
            {{0x947U | (7U << 12), 0x1000, 0x03800000, 0}},
            Exit,
        }};
        const auto escaped = MakeBinary(ShaderStage::Fragment, bad);
        const auto result = LowerNvnProgram(escaped.View(ShaderStage::Fragment));
        CHECK_FALSE(result->error.empty());
    }
    SECTION("a program the driver uploaded ends at its size footer") {
        // A subroutine reached by CALL.REL.NOINC after the EXIT: no EXIT/RET rule completes it.
        // The last eight instructions are the footer: MOV RZ, 0x12345678; MOV RZ, inst_bytes.
        const Instruction nop{{0x00007918, 0, 0, 0x000fc000}};
        const std::array<Instruction, 8> program{{
            {{0x00027802, 0x20, 0x00000f00, 0x000fe200}},       // MOV R2, 0x20
            {{0x00007944, 0x20, 0x03c00000, 0x000fea00}},       // CALL.REL.NOINC 0x40
            Exit,
            {{0x00007947, 0xfffffff0, 0x0383ffff, 0x000fc000}}, // BRA 0x30 (itself)
            {{0x00007802, 0x3f800000, 0x00000f00, 0x000fe200}}, // MOV R0, 0x3f800000
            {{0x02007950, 0xffffffa0, 0x03c3ffff, 0x000fea00}}, // RET.REL.NODEC R2, 0x0
            nop,
            nop,
        }};
        std::vector<Instruction> uploaded(program.begin(), program.end());
        const u32 inst_bytes = GraphicsHeaderBytes + 16 * static_cast<u32>(sizeof(Instruction));
        uploaded.push_back({{0x00ff7802, 0x12345678, 0x00000f00, 0x000fc000}});
        uploaded.push_back({{0x00ff7802, inst_bytes, 0x00000f00, 0x000fc000}});
        uploaded.resize(16, nop);
        const auto binary = MakeBinary(ShaderStage::Fragment, uploaded);
        const auto lowered = LowerNvnProgram(binary.View(ShaderStage::Fragment));
        REQUIRE(lowered->error.empty());
        CHECK(lowered->capture_complete);
        const auto reference = Shader::Ampere::LowerToMaxwell(program);
        REQUIRE(static_cast<bool>(reference));
        CHECK(lowered->instruction_count == reference.instructions.size());
        CHECK(lowered->source_size == GraphicsHeaderBytes + reference.source_size);
    }
}

namespace {

struct Translated {
    std::shared_ptr<const LoweredNvnProgram> lowered;
    std::unique_ptr<NvnEnvironment> env;
    NvnShaderPools pools;
    Shader::IR::Program program;
};

std::unique_ptr<Translated> Translate(ShaderStage stage, std::span<const Instruction> code) {
    auto result = std::make_unique<Translated>();
    const auto binary = MakeBinary(stage, code);
    result->lowered = LowerNvnProgram(binary.View(stage));
    REQUIRE(result->lowered->error.empty());
    result->env = std::make_unique<NvnEnvironment>(result->lowered);
    result->program =
        TranslateNvnStage(result->pools, *result->env, MakeReferenceHostInfo());
    return result;
}

} // namespace

TEST_CASE("NVN environment hands out synthetic handles", "[nvn][shaders]") {
    const std::array<u32, 8> c1{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<Instruction, 1> program{Exit};
    const auto binary = MakeBinary(ShaderStage::Fragment, program, c1);
    auto lowered = LowerNvnProgram(binary.View(ShaderStage::Fragment));
    REQUIRE(lowered->error.empty());
    NvnEnvironment env{lowered, {[](const NvnHandleRecord& record) {
                           NvnTextureInfo info;
                           info.type = record.offset == 0x170 ? Shader::TextureType::ColorCube
                                                              : Shader::TextureType::Color2D;
                           return info;
                       }}};
    CHECK(env.StartAddress() == 0);
    CHECK(env.IsProprietaryDriver());
    CHECK(env.TextureBoundBuffer() == HandleBank);
    CHECK(env.ReadViewportTransformState() == 1);
    CHECK(env.ReadCbufValue(ProgramBank, 8) == 3);
    CHECK(env.ReadCbufValue(ProgramBank, 0x1000) == 0);
    CHECK(env.ReadCbufValue(DriverBank, 0x110) == 0);
    const u32 texture_a = env.ReadCbufValue(HandleBank, 0x168);
    const u32 texture_b = env.ReadCbufValue(HandleBank, 0x170);
    const u32 sampler = env.ReadCbufValue(HandleBank, 0x568);
    CHECK(texture_a != texture_b);
    CHECK(texture_a < (1U << 20));
    CHECK(sampler == SeparateSamplerWord(SamplerIdOf(sampler)));
    CHECK(env.ReadCbufValue(HandleBank, 0x168) == texture_a); // stable
    CHECK(env.ReadCbufValue(HandleBank, 0x968) == 0);
    CHECK(env.ReadTextureType(texture_a | sampler) == Shader::TextureType::Color2D);
    CHECK(env.ReadTextureType(sampler | texture_b) == Shader::TextureType::ColorCube);
    CHECK(env.HandleRecords().size() == 3);
    CHECK_FALSE(env.HandleKeyOverflow());
    CHECK_THROWS_AS(env.ReadInstruction(0x1000), Shader::RuntimeError);
    CHECK_THROWS_AS(env.TranslateProgramAddress(0x20), Shader::RuntimeError);
    CHECK(env.TranslateProgramAddress(0) == 0x50);
}

TEST_CASE("NVN translation yields the expected descriptors and bindings", "[nvn][shaders]") {
    const auto translated = Translate(ShaderStage::Fragment, TwoTextureProgram);
    const auto& info = translated->program.info;
    REQUIRE(info.texture_descriptors.size() == 2);
    for (u32 i = 0; i < 2; ++i) {
        const auto& descriptor = info.texture_descriptors[i];
        CHECK(descriptor.type == Shader::TextureType::Color2D);
        CHECK(descriptor.has_secondary);
        CHECK(descriptor.cbuf_index == HandleBank);
        CHECK(descriptor.cbuf_offset == (i == 0 ? 0x168U : 0x170U));
        CHECK(descriptor.secondary_cbuf_index == HandleBank);
        CHECK(descriptor.secondary_cbuf_offset == 0x568);
        CHECK(descriptor.count == 1);
    }

    const StageInfo stages[] = {{ShaderStage::Fragment, info}};
    const auto layout = BuildDescriptorLayout(stages);
    INFO(DescribeLayout(layout));
    CHECK(layout.AllSupported());
    std::vector<const DescriptorBinding*> textures;
    for (const auto& binding : layout.bindings) {
        if (binding.role == DescriptorRole::Texture) {
            textures.push_back(&binding);
        }
    }
    REQUIRE(textures.size() == 2);
    for (u32 i = 0; i < 2; ++i) {
        CHECK(textures[i]->kind == DescriptorKind::CombinedImageSampler);
        CHECK(textures[i]->texture.source == HandleSource::SeparateTexture);
        CHECK(textures[i]->texture.slot == i);
        CHECK(textures[i]->sampler.source == HandleSource::SeparateSampler);
        CHECK(textures[i]->sampler.slot == 0);
        CHECK(textures[i]->stage_mask == (1U << static_cast<u32>(ShaderStage::Fragment)));
    }
    CHECK(textures[1]->binding == textures[0]->binding + 1);
    CHECK(layout.binding_count ==
          Shader::NumDescriptors(info.constant_buffer_descriptors) +
              Shader::NumDescriptors(info.storage_buffers_descriptors) + 2);
}

TEST_CASE("NVN vertex programs read c0[0x644] as the draw's base instance", "[nvn][shaders]") {
    // MOV R6, c[0x0][0x644] (OoT's text vertex program f1f183d55f7989b9, register changed);
    // IMAD.MOV.U32 R2/R3, RZ, RZ, c[0x0][0x110/0x114]; STG.E.STRONG.SM [R2], R6 (nvdisasm).
    constexpr Instruction MovBaseInstance{{0x00067a02, 0x00019100, 0x00000f00, 0x000fe200}};
    constexpr Instruction MovAddressLow{{0xff027624, 0x00004400, 0x078e00ff, 0x000fe200}};
    constexpr Instruction MovAddressHigh{{0xff037624, 0x00004500, 0x078e00ff, 0x000fe200}};
    constexpr Instruction Store{{0x02007386, 0x00000006, 0x0010a900, 0x000fe200}};
    const std::array<Instruction, 5> program{MovAddressLow, MovAddressHigh, MovBaseInstance, Store,
                                             Exit};
    // The draw macros write the base instance to the driver bank, Vulkan hands the vertex stage
    // the same value as BaseInstance.
    const auto vertex = Translate(ShaderStage::Vertex, program);
    CHECK(vertex->program.info.loads[Shader::IR::Attribute::BaseInstance]);
    // Other stages keep the plain constant buffer word.
    const auto fragment = Translate(ShaderStage::Fragment, program);
    CHECK_FALSE(fragment->program.info.loads[Shader::IR::Attribute::BaseInstance]);
}

TEST_CASE("NVN layout reports unsupported handle words", "[nvn][shaders]") {
    Shader::Info info;
    // A separate texture with no sampler word: the constant sampler ID was folded away.
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, false,
                                        HandleBank, 0x168, 0, 0, 0, 0, 1, 0});
    // A combined handle, an array of them, and a bad array stride.
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, false,
                                        HandleBank, 0x20, 0, 0, 0, 0, 1, 0});
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, false,
                                        HandleBank, 0x20, 0, 0, 0, 0, 4, 3});
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, false,
                                        HandleBank, 0x20, 0, 0, 0, 0, 4, 2});
    // Images: image table, and a combined-range word.
    info.image_descriptors.push_back({Shader::TextureType::Color2D, Shader::ImageFormat::Typeless,
                                      true, false, false, HandleBank, 0x120, 1, 0});
    info.image_descriptors.push_back({Shader::TextureType::Color2D, Shader::ImageFormat::Typeless,
                                      true, false, false, HandleBank, 0x40, 1, 0});
    // Storage buffers: in-table, then outside the fragment table.
    info.storage_buffers_descriptors.push_back({DriverBank, 0x520, 1, true});
    info.storage_buffers_descriptors.push_back({DriverBank, 0x110, 1, false});
    const StageInfo stages[] = {{ShaderStage::Fragment, info}};
    const auto layout = BuildDescriptorLayout(stages);
    INFO(DescribeLayout(layout));
    REQUIRE(layout.bindings.size() == 8);
    // Order: storage buffers, then textures, then images.
    CHECK(layout.bindings[0].role == DescriptorRole::StorageBuffer);
    CHECK(layout.bindings[0].supported);
    CHECK(layout.bindings[0].slot == 1);
    CHECK_FALSE(layout.bindings[1].supported);
    CHECK_FALSE(layout.bindings[2].supported); // separate texture without sampler
    CHECK(layout.bindings[3].supported);       // combined
    CHECK(layout.bindings[3].texture.source == HandleSource::Combined);
    CHECK(layout.bindings[4].supported);       // 4 slots, 8-byte stride
    CHECK_FALSE(layout.bindings[5].supported); // 4-byte stride
    CHECK(layout.bindings[6].supported);
    CHECK(layout.bindings[6].texture.source == HandleSource::Image);
    CHECK_FALSE(layout.bindings[7].supported);
    CHECK(layout.unsupported_count == 4);
    for (u32 i = 0; i < layout.bindings.size(); ++i) {
        CHECK(layout.bindings[i].binding == i); // arrays take one binding
    }
    CHECK(layout.binding_count == 8);
}

TEST_CASE("NVN layout resolves textures whose handle is a program", "[nvn][shaders]") {
    Shader::Info info;
    info.handle_programs.push_back({static_cast<u32>(Shader::HandleOp::Immediate), 8U});
    info.storage_buffers_descriptors.push_back({DriverBank, 0x510, 1, false});
    info.storage_buffers_descriptors.push_back({DriverBank, 0x540, 1, false});
    Shader::TextureDescriptor program{Shader::TextureType::ColorArray2D, false, false, false,
                                      false, DriverBank, 0, 0, 0, 0, 0, 1, 0};
    program.handle_program = 0;
    info.texture_descriptors.push_back(program);
    program.handle_program = 5; // no such program
    info.texture_descriptors.push_back(program);
    const StageInfo stages[] = {{ShaderStage::Fragment, info}};
    const auto layout = BuildDescriptorLayout(stages);
    INFO(DescribeLayout(layout));
    REQUIRE(layout.bindings.size() == 4);
    CHECK(layout.bindings[0].storage_index == 0);
    CHECK(layout.bindings[1].storage_index == 1);
    CHECK(layout.bindings[2].supported);
    CHECK(layout.bindings[2].texture.source == HandleSource::Program);
    CHECK(layout.bindings[2].texture.slot == 0);
    CHECK(layout.bindings[2].handle_program == info.handle_programs[0]);
    CHECK(layout.bindings[2].sampler.source == HandleSource::None);
    CHECK_FALSE(layout.bindings[3].supported);
    CHECK(layout.unsupported_count == 1);
}

TEST_CASE("NVN handle programs read the handle table from guest memory", "[nvn][shaders]") {
    using Op = Shader::HandleOp;
    // handle = storage1[((c4[0x88] & 0xfffff) + c5[0x2ec]) << 5 + 0x10]
    const std::vector<u32> code{
        static_cast<u32>(Op::Cbuf), 4, 0x88, static_cast<u32>(Op::Immediate), 0xfffff,
        static_cast<u32>(Op::And),  static_cast<u32>(Op::Cbuf), 5, 0x2ec, static_cast<u32>(Op::Add),
        static_cast<u32>(Op::Immediate), 5, static_cast<u32>(Op::Shl),
        static_cast<u32>(Op::Immediate), 0x10, static_cast<u32>(Op::Add),
        static_cast<u32>(Op::Storage), 1};
    constexpr u64 table = 0x20008; // not aligned: the program's offsets start at 0x20000
    const auto uniform = [](u32 bank, u32 offset) -> u32 {
        if (bank == 4 && offset == 0x88) {
            return 0x8000'0003; // flag bits above 20 are masked off
        }
        if (bank == 5 && offset == 0x2ec) {
            return 2;
        }
        return 0;
    };
    const auto memory = [](u64 address) -> std::optional<u32> {
        return address == 0x20000 + 5 * 32 + 0x10 ? std::optional<u32>{0x0050'1234}
                                                  : std::optional<u32>{0};
    };
    const auto storage = [&](u32 index) -> HandleProgramBuffer {
        return index == 1 ? HandleProgramBuffer{table, 0x1000} : HandleProgramBuffer{};
    };
    CHECK(EvaluateHandleProgram(code, 16, uniform, storage, memory) == 0x0050'1234u);
    // The buffer's own robustness: a word past its size reads 0. The buffer starts at 0x20008, so
    // 0xa8 bytes end at offset 0xb0 from the aligned base and the word at 0xb0 is outside it.
    const auto small = [&](u32 index) -> HandleProgramBuffer {
        return index == 1 ? HandleProgramBuffer{table, 0xa8} : HandleProgramBuffer{};
    };
    CHECK(EvaluateHandleProgram(code, 16, uniform, small, memory) == 0u);
    // An unbound buffer reads 0.
    const auto none = [](u32) { return HandleProgramBuffer{}; };
    CHECK(EvaluateHandleProgram(code, 16, uniform, none, memory) == 0u);
}

TEST_CASE("NVN SPIR-V emission matches the layout's binding numbering", "[nvn][shaders]") {
    const Shader::Profile profile = MakeReferenceProfile();
    Shader::RuntimeInfo runtime_info;
    runtime_info.previous_stage_stores.mask.set();

    SECTION("fragment with two textures") {
        auto translated = Translate(ShaderStage::Fragment, TwoTextureProgram);
        const StageInfo stages[] = {{ShaderStage::Fragment, translated->program.info}};
        const auto layout = BuildDescriptorLayout(stages);
        Shader::Backend::Bindings bindings;
        const std::vector<u32> code =
            Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, translated->program, bindings);
        REQUIRE(code.size() > 5);
        CHECK(code[0] == 0x07230203);
        CHECK(bindings.unified == layout.binding_count);
        CHECK(bindings.unified >= 2);
    }
    SECTION("EXIT-only vertex") {
        const std::array<Instruction, 1> program{Exit};
        auto translated = Translate(ShaderStage::Vertex, program);
        const StageInfo stages[] = {{ShaderStage::Vertex, translated->program.info}};
        const auto layout = BuildDescriptorLayout(stages);
        Shader::Backend::Bindings bindings;
        const std::vector<u32> code =
            Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, translated->program, bindings);
        REQUIRE(code.size() > 5);
        CHECK(code[0] == 0x07230203);
        CHECK(bindings.unified == layout.binding_count);
    }
    SECTION("bindings continue across stages") {
        auto vertex = Translate(ShaderStage::Vertex, std::array<Instruction, 1>{Exit});
        auto fragment = Translate(ShaderStage::Fragment, TwoTextureProgram);
        const StageInfo stages[] = {{ShaderStage::Vertex, vertex->program.info},
                                    {ShaderStage::Fragment, fragment->program.info}};
        const auto layout = BuildDescriptorLayout(stages);
        Shader::Backend::Bindings bindings;
        (void)Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, vertex->program, bindings);
        runtime_info.previous_stage_stores = vertex->program.info.stores;
        (void)Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, fragment->program, bindings);
        CHECK(bindings.unified == layout.binding_count);
    }
}

TEST_CASE("NVN compute layout classifies storage descriptors by driver table", "[nvn][shaders]") {
    // 2 uniform entries at 0x10, 3 storage entries at 0x30, 1 texture at 0x60, 1 image at 0x68.
    ComputeDriverLayout compute;
    compute.uniform_offset = 0x10;
    compute.uniform_count = 2;
    compute.storage_offset = 0x30;
    compute.storage_count = 3;
    compute.texture_offset = 0x60;
    compute.texture_count = 1;
    compute.image_offset = 0x68;
    compute.image_count = 1;
    Shader::Info info;
    info.storage_buffers_descriptors.push_back({DriverBank, 0x20, 1, false});  // uniform entry 1
    info.storage_buffers_descriptors.push_back({DriverBank, 0x50, 1, true});   // storage entry 2
    info.storage_buffers_descriptors.push_back({DriverBank, 0x60, 1, false});  // outside both
    info.storage_buffers_descriptors.push_back({DriverBank, 0x38, 1, false});  // misaligned
    info.storage_buffers_descriptors.push_back({4, 0x18, 1, false});           // word pair in c4
    info.storage_buffers_descriptors.push_back({HandleBank, 0x18, 1, false});  // bad bank
    // A compute texture and image read from their tables of the driver bank.
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, false,
                                        DriverBank, 0x60, 0, 0, 0, 0, 1, 0});
    info.image_descriptors.push_back({Shader::TextureType::Color2D, Shader::ImageFormat::Typeless,
                                      true, false, false, DriverBank, 0x68, 1, 0});
    const StageInfo stages[] = {{ShaderStage::Compute, info, &compute}};
    const auto layout = BuildDescriptorLayout(stages);
    INFO(DescribeLayout(layout));
    REQUIRE(layout.bindings.size() == 8);
    CHECK(layout.bindings[0].supported);
    CHECK(layout.bindings[0].storage_source == StorageSource::ComputeUniformEntry);
    CHECK(layout.bindings[0].slot == 1);
    CHECK(layout.bindings[1].storage_source == StorageSource::ComputeStorageEntry);
    CHECK(layout.bindings[1].slot == 2);
    CHECK(layout.bindings[1].is_written);
    CHECK_FALSE(layout.bindings[2].supported);
    CHECK_FALSE(layout.bindings[3].supported);
    CHECK(layout.bindings[4].supported);
    CHECK(layout.bindings[4].storage_source == StorageSource::UniformBufferWord);
    CHECK(layout.bindings[4].slot == 1); // c4 is uniform buffer slot 1
    CHECK_FALSE(layout.bindings[5].supported);
    CHECK(layout.bindings[6].supported);
    CHECK(layout.bindings[6].texture.source == HandleSource::ComputeTexture);
    CHECK(layout.bindings[7].supported);
    CHECK(layout.bindings[7].texture.source == HandleSource::ComputeImage);
    CHECK(layout.unsupported_count == 3);

    // Without the control-derived layout a compute storage buffer cannot be placed.
    const StageInfo bare[] = {{ShaderStage::Compute, info}};
    CHECK_FALSE(BuildDescriptorLayout(bare).AllSupported());
}

TEST_CASE("NVN layout rejects ambiguous uniform-buffer handle pairs", "[nvn][shaders]") {
    Shader::Info info;
    // Two words of uniform buffers (c3 and c4): both dense keys would OR into a wrong record.
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, true, 3,
                                        0x10, 0, 4, 0x20, 0, 1, 0});
    // One uniform-buffer word plus a separate sampler word is fine.
    info.texture_descriptors.push_back({Shader::TextureType::Color2D, false, false, false, true, 3,
                                        0x10, 0, HandleBank, SeparateSamplerBase, 0, 1, 0});
    const StageInfo stages[] = {{ShaderStage::Fragment, info}};
    const auto layout = BuildDescriptorLayout(stages);
    INFO(DescribeLayout(layout));
    REQUIRE(layout.bindings.size() == 2);
    CHECK_FALSE(layout.bindings[0].supported);
    CHECK(layout.bindings[0].issue.find("type key ambiguous") != std::string::npos);
    CHECK(layout.bindings[1].supported);
}

TEST_CASE("NVN environment reserves the last handle key", "[nvn][shaders]") {
    const std::array<Instruction, 1> program{Exit};
    const auto binary = MakeBinary(ShaderStage::Fragment, program);
    auto lowered = LowerNvnProgram(binary.View(ShaderStage::Fragment));
    REQUIRE(lowered->error.empty());
    NvnEnvironment env{lowered, {[](const NvnHandleRecord&) {
                           NvnTextureInfo info;
                           info.type = Shader::TextureType::ColorCube;
                           return info;
                       }}};
    // Uniform-buffer words are unbounded: distinct (bank, offset) pairs fill the key space.
    u32 last = 0;
    for (u32 i = 0; i < 0x1000; ++i) {
        last = env.ReadCbufValue(3, i * 4);
    }
    CHECK(env.HandleKeyOverflow());
    CHECK(env.HandleRecords().size() == 0xFFE);
    CHECK(last == 0xFFF);
    // The reserved key names no record: it gets the default type, never a real one.
    CHECK(env.ReadTextureType(0xFFF) == Shader::TextureType::Color2D);
    CHECK(env.ReadTextureType(0xFFE) == Shader::TextureType::ColorCube);
}

// Task (stage 6) and mesh (stage 7) controls of Ocarina of Time's programs, and the register words
// NVN2's program bind (nnSdk+0x694880) derives from them; the expected words are the values the
// native path received for those programs.
TEST_CASE("NVN task and mesh controls give the SDK's mesh register words", "[nvn][shaders]") {
    std::string error;
    auto mesh = MakeControl(7, 0x180, 0x200, 0x200);
    // +0x58 threads 0x20, +0x5f triangles, +0x60 max vertices 0x80, +0x62 max primitives 0x40.
    Put32(mesh, 0x58, 0x00000020);
    Put32(mesh, 0x5c, 0x03000000);
    Put32(mesh, 0x60, 0x00400080);
    auto task = MakeControl(6, 0x180, 0x200, 0x200);
    // +0x58 threads 0x20, +0x5a 0xc80 bytes passed to the mesh workgroups.
    Put32(task, 0x58, 0x0c800020);
    Put32(task, 0x5c, 0);
    Put32(task, 0x60, 0);
    const auto parsed_mesh = ParseControl(mesh, error);
    const auto parsed_task = ParseControl(task, error);
    REQUIRE(parsed_mesh.has_value());
    REQUIRE(parsed_task.has_value());
    CHECK(parsed_mesh->stage_byte == static_cast<u32>(ShaderStage::Mesh));
    CHECK(parsed_task->stage_byte == static_cast<u32>(ShaderStage::Task));
    CHECK(parsed_mesh->mesh.max_vertices == 0x80);
    CHECK(parsed_mesh->mesh.max_primitives == 0x40);
    CHECK(parsed_task->mesh.output_bytes == 0xc80);
    CHECK(MeshShaderA(parsed_mesh->mesh) == 0x80804);
    CHECK(MeshShaderB(parsed_mesh->mesh) == 0x2000000);
    CHECK(MeshInitShader(parsed_task->mesh) == 0x6419020);
    CHECK(MeshRegisters(&parsed_task->mesh, parsed_mesh->mesh) ==
          std::array<u32, 3>{0x80804, 0x2000000, 0x6419020});
    // Without a task program the SDK writes no init word.
    CHECK(MeshRegisters(nullptr, parsed_mesh->mesh)[2] == 0);
    // Topology byte: 1 points -> 0, 2 lines -> 1, anything else triangles -> 4.
    MeshControl points = parsed_mesh->mesh;
    points.topology = 1;
    CHECK((MeshShaderA(points) & 0xf) == 0);
    points.topology = 2;
    CHECK((MeshShaderA(points) & 0xf) == 1);
    // A task output size rounds up to 128-byte lines; local bytes add lines to the task's own.
    MeshControl lines = parsed_task->mesh;
    lines.output_bytes = 0;
    lines.local_bytes = 0x81;
    CHECK(MeshInitShader(lines) == (1u << 22 | 0x20 | 3u << 12));
    // Stage-driver tables: task and mesh programs have their own (the SDK's BindStorageBuffer
    // stage table: entries 0x67 and 0x77), not the vertex and tess-eval ones.
    CHECK(StorageBufferBase(ShaderStage::Task) == 0x670);
    CHECK(StorageBufferBase(ShaderStage::Mesh) == 0x770);
    CHECK(ToRecompilerStage(ShaderStage::Task) == Shader::Stage::Task);
    CHECK(ToRecompilerStage(ShaderStage::Mesh) == Shader::Stage::Mesh);
}

// The per-primitive header of a mesh program (the SDK's control getters 0x1b0/0x1b8: u32 +0x64
// offset from the code, u16 +0x68 size; the bind binds the geometry pipeline there when the size
// is non-zero). OoT's captured bind: mesh code 0x1148c25e00, header 0x1148c26600 (+0x800).
TEST_CASE("NVN mesh controls name the per-primitive header", "[nvn][shaders]") {
    std::string error;
    auto control = MakeControl(7, 0x180, 0x200, 0x300);
    Put32(control, 0x64, 0x200);
    Put32(control, 0x68, 0x50);
    const auto parsed = ParseControl(control, error);
    REQUIRE(parsed.has_value());
    CHECK(parsed->mesh.primitive_header_offset == 0x200);
    CHECK(parsed->mesh.primitive_header_bytes == 0x50);

    std::vector<u8> code(0x300);
    for (u32 i = 0; i < SphBytes; ++i) {
        code[0x200 + i] = static_cast<u8>(0xA0 + i);
    }
    const auto lowered = LowerNvnProgram(NvnProgramBinary{ShaderStage::Mesh, control, code});
    REQUIRE(lowered->primitive_header.has_value());
    CHECK((*lowered->primitive_header)[0] == 0xA3A2A1A0u);
    CHECK((*lowered->primitive_header)[19] == 0xEFEEEDECu);

    // No size: no header (the SDK disables the geometry pipeline).
    Put32(control, 0x68, 0);
    const auto none = LowerNvnProgram(NvnProgramBinary{ShaderStage::Mesh, control, code});
    CHECK_FALSE(none->primitive_header.has_value());
    // A header outside the code area or shorter than an SPH is refused, visibly.
    Put32(control, 0x68, 0x50);
    Put32(control, 0x64, 0x2c0);
    const auto outside = LowerNvnProgram(NvnProgramBinary{ShaderStage::Mesh, control, code});
    CHECK(outside->error.find("per-primitive header") != std::string::npos);
    Put32(control, 0x64, 0x200);
    Put32(control, 0x68, 0x40);
    const auto short_header = LowerNvnProgram(NvnProgramBinary{ShaderStage::Mesh, control, code});
    CHECK(short_header->error.find("per-primitive header") != std::string::npos);
}

TEST_CASE("NVN environment: native pointer search over the driver bank's buffer entries",
          "[nvn][shaders]") {
    const std::array<Instruction, 1> program{Exit};
    // A compute control with 2 uniform entries (c0+0x10) and 3 storage entries (c0+0x30).
    auto compute = MakeBinary(ShaderStage::Compute, program);
    compute.control[0xb1] = 2;
    compute.control[0xb2] = 3;
    const auto lowered = LowerNvnProgram(compute.View(ShaderStage::Compute));
    REQUIRE(lowered->error.empty());
    NvnEnvironment env{lowered};
    CHECK(env.IsNativeProgram());
    for (const u32 offset : {0x10U, 0x20U, 0x30U, 0x40U, 0x50U}) {
        CHECK(env.IsCbufStorageBufferPointer(DriverBank, offset));
    }
    CHECK_FALSE(env.IsCbufStorageBufferPointer(DriverBank, 0x0));  // bank size, group counts
    CHECK_FALSE(env.IsCbufStorageBufferPointer(DriverBank, 0x18)); // size word of an entry
    CHECK_FALSE(env.IsCbufStorageBufferPointer(DriverBank, 0x60)); // past the storage entries
    CHECK_FALSE(env.IsCbufStorageBufferPointer(ProgramBank, 0x10));
    CHECK_FALSE(env.IsCbufStorageBufferPointer(HandleBank, 0x10));
    CHECK_FALSE(env.IsCbufStorageBufferPointer(FirstUniformBank, 0x10));

    SECTION("graphics stages: their own storage table") {
        const auto vertex = MakeBinary(ShaderStage::Vertex, program);
        const auto lowered_vertex = LowerNvnProgram(vertex.View(ShaderStage::Vertex));
        REQUIRE(lowered_vertex->error.empty());
        NvnEnvironment vertex_env{lowered_vertex};
        CHECK(vertex_env.IsNativeProgram());
        CHECK(vertex_env.IsCbufStorageBufferPointer(DriverBank, 0x110));
        CHECK(vertex_env.IsCbufStorageBufferPointer(DriverBank, 0x200));
        CHECK_FALSE(vertex_env.IsCbufStorageBufferPointer(DriverBank, 0x118));
        CHECK_FALSE(vertex_env.IsCbufStorageBufferPointer(DriverBank, 0x210));
        CHECK_FALSE(vertex_env.IsCbufStorageBufferPointer(DriverBank, 0x510));
    }

    SECTION("task and mesh stages: their own storage tables, not the vertex/tess-eval ones") {
        const auto task = MakeBinary(ShaderStage::Task, program);
        const auto lowered_task = LowerNvnProgram(task.View(ShaderStage::Task));
        REQUIRE(lowered_task->error.empty());
        NvnEnvironment task_env{lowered_task};
        CHECK(task_env.IsCbufStorageBufferPointer(DriverBank, 0x670));
        CHECK(task_env.IsCbufStorageBufferPointer(DriverBank, 0x680));
        CHECK(task_env.IsCbufStorageBufferPointer(DriverBank, 0x760));
        CHECK_FALSE(task_env.IsCbufStorageBufferPointer(DriverBank, 0x110));
        CHECK_FALSE(task_env.IsCbufStorageBufferPointer(DriverBank, 0x770));
        const auto mesh = MakeBinary(ShaderStage::Mesh, program);
        const auto lowered_mesh = LowerNvnProgram(mesh.View(ShaderStage::Mesh));
        REQUIRE(lowered_mesh->error.empty());
        NvnEnvironment mesh_env{lowered_mesh};
        CHECK(mesh_env.IsCbufStorageBufferPointer(DriverBank, 0x770));
        CHECK(mesh_env.IsCbufStorageBufferPointer(DriverBank, 0x860));
        CHECK_FALSE(mesh_env.IsCbufStorageBufferPointer(DriverBank, 0x310));
        CHECK_FALSE(mesh_env.IsCbufStorageBufferPointer(DriverBank, 0x870));
    }

    SECTION("a load through a phi of a storage entry and an indexed uniform table stays raw") {
        // Ocarina of Time's foliage cull (6a1cfc8a, SASS 0x4750-0x47c0): `@P4 LDC.64 R10,
        // c[0x4][R2 * 16 + 0x80]; @P4 LDG.E.64 R8, [R10]`. Where P4 is false R10 keeps an older
        // value built from c0+0x50, so the address is a phi of both. Rebasing the load on storage
        // entry 0 read that buffer at the list pointer's offset: zeros, so the main view's per-item
        // words were 0 and none of its foliage was drawn on the HLE path.
        using namespace Shader;
        ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        IR::IREmitter ir{block};
        IR::Program ir_program;
        ir_program.info.nvn_buffer_window = true;
        ir_program.info.nvn_buffer_base = 0x10;
        ir_program.info.nvn_buffer_count = 5;
        const IR::U32 slot{ir.IAdd(
            IR::U32{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(4))}, ir.Imm32(0x80))};
        const IR::U32 table_low{ir.GetCbuf(ir.Imm32(FirstUniformBank + 1), slot)};
        const IR::U32 table_high{
            ir.GetCbuf(ir.Imm32(FirstUniformBank + 1), ir.IAdd(slot, ir.Imm32(4)))};
        const IR::U1 pred{ir.GetPred(IR::Pred::P4)};
        const IR::U32 low{
            ir.Select(pred, table_low, ir.GetCbuf(ir.Imm32(DriverBank), ir.Imm32(0x50)))};
        const IR::U32 high{
            ir.Select(pred, table_high, ir.GetCbuf(ir.Imm32(DriverBank), ir.Imm32(0x54)))};
        (void)ir.LoadGlobal64(ir.PackUint2x32(ir.CompositeConstruct(low, high)));
        HostTranslateInfo host{};
        host.min_ssbo_alignment = 16;
        host.nvn_raw_global_memory = true;
        ir_program.post_order_blocks.push_back(&block);
        Optimization::GlobalMemoryToStorageBufferPass(env, ir_program, host);
        const auto has = [&](IR::Opcode opcode) {
            return std::ranges::any_of(block.Instructions(), [=](const IR::Inst& inst) {
                return inst.GetOpcode() == opcode;
            });
        };
        CHECK(has(IR::Opcode::LoadGlobal64));
        CHECK_FALSE(has(IR::Opcode::LoadStorage64));
    }
}

// Ocarina of Time's foliage task program (afd1769404cc86e6) loads the u64 at offset 0 of the
// buffer in its storage slot 0 (`c0+0x670`) and its mesh program (7affb504671dce36) reads through
// the buffer in its slot 0 (`c0+0x770`). With the task/mesh tables placed at 0x110/0x310, the HLE
// bank held nothing at those offsets: both read through a null pointer every frame and the
// main-view foliage (canopies) was not drawn.
TEST_CASE("NVN task and mesh programs bind the buffers of their own driver bank tables",
          "[nvn][shaders]") {
    using namespace Shader;
    const std::array<Instruction, 1> program{Exit};
    struct Case {
        ShaderStage stage;
        Stage recompiler_stage;
        u32 entry;
    };
    for (const Case& c : {Case{ShaderStage::Task, Stage::Task, 0x670},
                          Case{ShaderStage::Mesh, Stage::Mesh, 0x770}}) {
        INFO(StageName(c.stage));
        const auto binary = MakeBinary(c.stage, program);
        const auto lowered = LowerNvnProgram(binary.View(c.stage));
        REQUIRE(lowered->error.empty());
        NvnEnvironment env{lowered};
        ObjectPool<IR::Inst> pool;
        IR::Block block{pool};
        IR::IREmitter ir{block};
        IR::Program ir_program;
        ir_program.stage = c.recompiler_stage;
        const IR::U32 low{ir.GetCbuf(ir.Imm32(DriverBank), ir.Imm32(c.entry))};
        const IR::U32 high{ir.GetCbuf(ir.Imm32(DriverBank), ir.Imm32(c.entry + 4))};
        (void)ir.LoadGlobal64(ir.PackUint2x32(ir.CompositeConstruct(low, high)));
        ir_program.post_order_blocks.push_back(&block);
        HostTranslateInfo host{};
        host.min_ssbo_alignment = 16;
        host.nvn_raw_global_memory = true;
        Optimization::GlobalMemoryToStorageBufferPass(env, ir_program, host);
        Optimization::CollectShaderInfoPass(env, ir_program);
        const Info& info{ir_program.info};
        CHECK(info.nvn_buffer_base == c.entry);
        CHECK(info.nvn_buffer_used[0]);
        REQUIRE(info.storage_buffers_descriptors.size() == 1);
        CHECK(info.storage_buffers_descriptors[0].cbuf_index == DriverBank);
        CHECK(info.storage_buffers_descriptors[0].cbuf_offset == c.entry);
        CHECK(std::ranges::any_of(block.Instructions(), [](const IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::LoadStorage64;
        }));
        // The descriptor resolves to slot 0 of the stage's driver table, the slot the game's
        // nvnCommandBufferBindStorageBuffer(stage, 0, ...) fills.
        const StageInfo stages[] = {{c.stage, info}};
        const auto layout = BuildDescriptorLayout(stages);
        INFO(DescribeLayout(layout));
        // The program reads c0 itself, so the layout also has the driver bank's uniform binding.
        const auto storage = std::ranges::find(layout.bindings, DescriptorRole::StorageBuffer,
                                               &DescriptorBinding::role);
        REQUIRE(storage != layout.bindings.end());
        CHECK(storage->supported);
        CHECK(storage->storage_source == StorageSource::DriverTable);
        CHECK(storage->slot == 0);
        CHECK(layout.unsupported_count == 0);
    }
}
