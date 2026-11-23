// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The recompiler's compute global-memory handling of the legacy path must not change when NVN
// compute learns its own driver bank window. The expected hashes below were produced by this same
// test on the recompiler *before* the window change (nvn/main fca94e1cb8); it only uses APIs that
// exist both before and after.

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

using namespace VideoCommon::Nvn;

constexpr u64 Exit = 0xe30000000007000fULL;

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

/// A compute environment of the legacy kind: knows nothing about NVN driver bank windows.
class LegacyComputeEnvironment final : public Shader::Environment {
public:
    explicit LegacyComputeEnvironment(std::span<const u64> program) : code{Pack(program)} {
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
    void Dump(u64, u64) override {}

private:
    std::vector<u64> code;
};

} // namespace

TEST_CASE("NVN legacy compute treats a dynamic driver bank read as every buffer", "[nvn][compute]") {
    // A constant buffer 0 read whose offset is not an immediate may hit any storage buffer entry
    // of the fixed 16-entry table: all 16 bits, never more (the table is wider in the set now).
    const std::array program{Exit};
    LegacyComputeEnvironment env{program};
    Shader::ObjectPool<Shader::IR::Inst> pool;
    Shader::IR::Block block{pool};
    Shader::IR::Program ir;
    ir.stage = Shader::Stage::Compute;
    ir.blocks = {&block};
    ir.post_order_blocks = {&block};
    Shader::IR::IREmitter emitter{block};
    emitter.GetCbuf(emitter.Imm32(0), emitter.GetReg(Shader::IR::Reg::R1));
    Shader::Optimization::CollectShaderInfoPass(env, ir);
    CHECK(ir.info.nvn_buffer_base == 0x310);
    CHECK(ir.info.nvn_buffer_count == 16);
    CHECK_FALSE(ir.info.nvn_buffer_window);
    CHECK(ir.info.nvn_buffer_used.to_ullong() == 0xFFFFu);
}
