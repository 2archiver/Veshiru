// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The vertex position must come out of two different shaders bit for bit alike (a depth pre-pass
// followed by EQUAL depth tests): `Profile::position_invariant` decorates the position output
// Invariant, and the math that feeds it is NoContraction (guest FMUL/FADD/FFMA already are;
// `Profile::position_precise` adds the operations the recompiler makes up). Both the NVN and the
// legacy GPU profile set them; mesh programs (OoT's foliage pre-pass and forward draws) included.

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

using namespace VideoCommon::Nvn;

constexpr u64 AldAttribute(u32 rd, u32 byte_offset) {
    return 0xefd8000000000000ULL | (u64{0xff} << 39) | (u64{byte_offset} << 20) | (u64{0xff} << 8) |
           rd;
}
constexpr u64 AstAttribute(u32 rs, u32 byte_offset) {
    return 0xeff0000000000000ULL | (u64{0xff} << 39) | (u64{byte_offset} << 20) | (u64{0xff} << 8) |
           rs;
}
constexpr u64 FmulReg(u32 rd, u32 ra, u32 rb) {
    return 0x5c68000000000000ULL | (u64{rb} << 20) | (u64{ra} << 8) | rd;
}
constexpr u64 Exit = 0xe30000000007000fULL;

constexpr u32 AttributeGeneric0 = 0x80;
constexpr u32 AttributePositionX = 0x70;

class VertexEnvironment final : public Shader::Environment {
public:
    explicit VertexEnvironment(std::span<const u64> program,
                               Shader::Stage stage_ = Shader::Stage::VertexB,
                               u32 mesh_topology = 4) {
        stage = stage_;
        start_address = 0;
        is_proprietary_driver = true;
        if (stage == Shader::Stage::Mesh) {
            // SET_MESH_SHADER_A: topology (4 = triangles), 128 vertices, 64 primitives;
            // SET_MESH_SHADER_B: 32 threads; SET_MESH_INIT_SHADER: two 128-byte payload lines. The
            // SPH's system output map names position XYZW as the per-vertex outputs.
            mesh_registers = {0x80800 | mesh_topology, 0x2000000, 0x802020};
            sph.vtg.omap_systemb.raw = 0xf0;
        }
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

std::vector<u32> EmitWords(const Shader::Profile& profile, std::span<const u64> program,
                           Shader::Stage stage = Shader::Stage::VertexB, u32 mesh_topology = 4,
                           Shader::RuntimeInfo runtime_info = {}) {
    VertexEnvironment env{program, stage, mesh_topology};
    NvnShaderPools pools;
    Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, env.StartAddress(), false);
    Shader::IR::Program ir = Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg,
                                                               MakeReferenceHostInfo());
    runtime_info.previous_stage_stores.mask.set();
    Shader::Backend::Bindings bindings;
    std::vector<u32> words = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, bindings);
    REQUIRE(words.size() > 5);
    return words;
}

/// What a mesh module declares about its output primitives: the OpExecutionMode output mode and
/// the BuiltIn decorations of its index array and PointSize.
struct MeshOutput {
    std::vector<u32> modes;
    std::vector<u32> builtins;
    bool malformed{};
    [[nodiscard]] bool HasMode(u32 mode) const {
        return std::ranges::find(modes, mode) != modes.end();
    }
    [[nodiscard]] bool HasBuiltIn(u32 builtin) const {
        return std::ranges::find(builtins, builtin) != builtins.end();
    }
};

MeshOutput ScanMeshOutput(std::span<const u32> words) {
    constexpr u32 OpExecutionMode = 16;
    constexpr u32 OpDecorate = 71;
    constexpr u32 DecorationBuiltIn = 11;
    MeshOutput result;
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16;
        const u32 opcode = words[i] & 0xffff;
        if (count == 0 || count > words.size() - i) {
            result.malformed = true; // truncated: its operands are not there
            break;
        }
        if (opcode == OpExecutionMode && count >= 3) {
            result.modes.push_back(words[i + 2]);
        } else if (opcode == OpDecorate && count >= 4 && words[i + 2] == DecorationBuiltIn) {
            result.builtins.push_back(words[i + 3]);
        }
        i += count;
    }
    return result;
}

} // namespace

TEST_CASE("Shaders: mesh programs output points, lines or triangles", "[nvn][shaders][mesh]") {
    // SET_MESH_SHADER_A bits 3:0 = output topology: 0 points, 1 lines, 4 triangles. OoT's garden
    // hedges use point mesh programs (0x00040200); they were rejected ("Mesh output topology 0"),
    // so every one of their draws was dropped.
    constexpr u32 ModeOutputPoints = 27;
    constexpr u32 ModeOutputLinesEXT = 5269;
    constexpr u32 ModeOutputTrianglesEXT = 5298;
    constexpr u32 BuiltInPointSize = 1;
    constexpr u32 BuiltInPrimitivePointIndicesEXT = 5294;
    constexpr u32 BuiltInPrimitiveLineIndicesEXT = 5295;
    constexpr u32 BuiltInPrimitiveTriangleIndicesEXT = 5296;
    const std::array<u64, 1> program{Exit};
    const Shader::Profile profile = MakeReferenceProfile();

    Shader::RuntimeInfo fixed_size;
    fixed_size.fixed_state_point_size = 4.0f;
    const MeshOutput points =
        ScanMeshOutput(EmitWords(profile, program, Shader::Stage::Mesh, 0, fixed_size));
    CHECK_FALSE(points.malformed);
    CHECK(points.HasMode(ModeOutputPoints));
    CHECK(points.HasBuiltIn(BuiltInPrimitivePointIndicesEXT));
    // Vulkan needs a size per point vertex: the fixed state one is written.
    CHECK(points.HasBuiltIn(BuiltInPointSize));
    CHECK_FALSE(points.HasBuiltIn(BuiltInPrimitiveTriangleIndicesEXT));

    const MeshOutput lines = ScanMeshOutput(EmitWords(profile, program, Shader::Stage::Mesh, 1));
    CHECK_FALSE(lines.malformed);
    CHECK(lines.HasMode(ModeOutputLinesEXT));
    CHECK(lines.HasBuiltIn(BuiltInPrimitiveLineIndicesEXT));
    CHECK_FALSE(lines.HasBuiltIn(BuiltInPointSize));

    const MeshOutput triangles = ScanMeshOutput(EmitWords(profile, program, Shader::Stage::Mesh));
    CHECK_FALSE(triangles.malformed);
    CHECK(triangles.HasMode(ModeOutputTrianglesEXT));
    CHECK(triangles.HasBuiltIn(BuiltInPrimitiveTriangleIndicesEXT));
    CHECK_FALSE(triangles.HasMode(ModeOutputPoints));

    // Other topology codes stay a visible rejection.
    CHECK_THROWS(EmitWords(profile, program, Shader::Stage::Mesh, 2));
}
