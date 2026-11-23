// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Maps an NVN2 shader program (control section + code area, as the game hands them to
// nvnProgramSetShaders) onto the shader recompiler: lowering to the Maxwell stream, a
// Shader::Environment that serves the NVN constant-buffer ABI, and translation to IR.
//
// The capture/pack logic mirrors GenericEnvironment::LowerNativeCode (video_core/
// shader_environment.cpp) exactly, so a program lowers to the same bytes here and in the legacy
// GPU path. That code is shared with the in-game path and deliberately not touched.
//
// Control section layout (nxnative shader_split.py / dmsh_ampere.py; fields marked [DATA] were
// checked against the 28,154 unique DK shaders by the coverage tool, the rest are [UNVERIFIED]):
//   +0x00 magic 0x98761234   +0x04 version 0xc9   [DATA]
//   +0x30 own length (0xd8 in BNSH, 0xd0 in effects, 0 in BAHS and in memory)
//   +0x38 instruction bytes (graphics: includes the 0x80-byte SPH)   [DATA]
//   +0x40 inst bytes rounded up to 0x100 = start of c1 inside the code area   [DATA]
//   +0x44 code area length          +0x4c register count
//   +0x56 stage byte = NVNshaderStage (0 VS, 1 FS, 2 GS, 3 TCS, 4 TES, 5 CS, 6 task, 7 mesh)   [DATA]
//   +0x58 compute thread group, 3 x u32   [DATA]
//   +0x58..+0x63 task and mesh programs: see MeshControl   [SDK]
//   +0x64 compute shared memory bytes   [DATA, inferred]
//   +0xa9/+0xb1/+0xb2/+0xb3/+0xb6, u16 +0xba: compute driver-bank counts and total
//   (see ComputeDriverLayout)   [DATA]

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/native_tensor_programs.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"

namespace VideoCommon::Nvn {

constexpr u32 ControlMagic = 0x98761234;
constexpr u32 ControlVersion = 0xc9;
constexpr u32 SphBytes = 0x50;       ///< bytes of the SPH the environment keeps
constexpr u32 GraphicsHeaderBytes = 0x80; ///< graphics code starts after this many bytes
constexpr u32 MinimumControlBytes = 0x68; ///< up to and including the compute shared size
constexpr u32 ComputeControlBytes = 0xbc; ///< up to and including the compute counts

/// The uniform/storage entry tables of a compute driver bank start at this alignment (see below).
constexpr u32 ComputeEntryTableAlign = 0x10;

/// Per-shader compute driver bank (c0) layout, derived from the control section's counts.
/// Measured on the 205 compute shaders of the DK dump (see the coverage report) and the 29
/// distinct layouts of Ocarina of Time's pipeline caches; the declared total (u16 at +0xba)
/// equals the derived end for all of them:
///   c0+0x10: push constants (`push_constant_words` words, byte +0xa9; zero in every DK shader)
///   then, from the next 0x10 boundary, `uniform_count` entries (byte +0xb1) and `storage_count`
///   entries (byte +0xb2), 0x10 each,
///   then `texture_count` handles (byte +0xb3) and `image_count` handles (byte +0xb6), 8 each.
/// The boundary only shows with an odd push-constant block: OoT's 15-word shader (3 textures,
/// 1 image) declares 0x70, not 0x6c, and reads its handles at c0+0x58/0x60 (textures 1, 2) and
/// 0x68 (image 0); its 29-word one (11 uniform, 9 storage, 7 textures, 2 images) declares 0x218,
/// not 0x20c (an 8-byte boundary would give 0x210).
/// Uniform and storage entries are {u64 address, u32 size, u32 pad}. The shader reaches
/// uniform entries too through global-memory access, so the recompiler's storage buffer
/// descriptors of a compute shader fall in either table.
struct ComputeDriverLayout {
    u32 push_constant_offset{0x10};
    u32 push_constant_words{};
    u32 uniform_offset{};
    u32 uniform_count{};
    u32 storage_offset{};
    u32 storage_count{};
    u32 texture_offset{};
    u32 texture_count{};
    u32 image_offset{};
    u32 image_count{};
    u32 end_offset{};     ///< derived end of the tables
    u32 declared_total{}; ///< u16 at control +0xba
    bool total_matches{}; ///< `end_offset == declared_total`

    [[nodiscard]] ComputeHandleTables HandleTables() const noexcept {
        return {texture_offset, texture_count, image_offset, image_count};
    }
};

/// The setup of a task (stage 6) or mesh (stage 7) program, from its control section. NVN2's
/// program bind (nnSdk+0x694880 of Ocarina of Time's SDK) turns these fields into
/// SET_MESH_SHADER_A / SET_MESH_SHADER_B (mesh program) and SET_MESH_INIT_SHADER (task program);
/// the recompiler translates task and mesh programs from those register words. The buffer sizes
/// are bytes the SDK rounds up to 128-byte stage-buffer lines.
struct MeshControl {
    u16 threads{};        ///< u16 +0x58: threads per workgroup (6 bits used)
    u16 output_bytes{};   ///< u16 +0x5a: task: bytes passed to the mesh workgroups
    u16 local_bytes{};    ///< u16 +0x5c: further stage-buffer bytes of the program itself
    u8 topology{};        ///< byte +0x5f: mesh output primitive, 1 points, 2 lines, else triangles
    u16 max_vertices{};   ///< u16 +0x60
    u16 max_primitives{}; ///< u16 +0x62
    /// Mesh programs: the per-primitive (pass-through geometry) program header. The SDK's control
    /// reader (getters 0x1b0 / 0x1b8 of the NVN2 control class) gives u32 +0x64 as its offset
    /// from the program's code and u16 +0x68 as its size; the bind (nnSdk+0x694880) binds the
    /// geometry pipeline at that address when the size is non-zero (MME macro 0xe08, slot 9) and
    /// disables it otherwise (macro 0xe3a with 0x49).
    u32 primitive_header_offset{};
    u16 primitive_header_bytes{};
};

/// SET_MESH_SHADER_A of a mesh program: topology 3:0 (0 points, 1 lines, 4 triangles), max
/// vertices 12:4, max primitives 22:13.
[[nodiscard]] u32 MeshShaderA(const MeshControl& mesh) noexcept;
/// SET_MESH_SHADER_B of a mesh program: stage-buffer lines 16:8, threads 25:20.
[[nodiscard]] u32 MeshShaderB(const MeshControl& mesh) noexcept;
/// SET_MESH_INIT_SHADER of a task program: threads 5:0, local lines 20:12, lines passed to the
/// mesh stage 30:22.
[[nodiscard]] u32 MeshInitShader(const MeshControl& task) noexcept;
/// The register words {SET_MESH_SHADER_A, SET_MESH_SHADER_B, SET_MESH_INIT_SHADER} a task (null
/// when the program has none) and mesh pair is translated with. The SDK writes the init word only
/// for a task program; without one it stays 0 here (no task payload).
[[nodiscard]] std::array<u32, 3> MeshRegisters(const MeshControl* task,
                                               const MeshControl& mesh) noexcept;

struct NvnControl {
    u32 magic{};
    u32 version{};
    u32 own_length{};
    u32 inst_bytes{};
    u32 c1_offset{};
    u32 code_bytes{};
    u32 register_count{};
    u32 stage_byte{};
    /// u32 at +0x50: local memory (the LDL/STL space) for one warp, in bytes. Measured on the 69
    /// DK graphics shaders that have local memory: it equals 32 times the per-thread size of
    /// their SPH (shader_local_memory_low_size) in every one, and is 0 when there is none. Kirby
    /// Air Riders has a compute shader with 0x2000 here whose largest STL offset + size is 0x100
    /// (the 205 compute shaders of the DK dump all have 0). [INFERRED from data, not verified
    /// against NVN2 documentation.]
    u32 local_memory_per_warp_bytes{};
    std::array<u32, 3> thread_group{};
    /// u32 at +0x64. Measured on the 205 compute shaders of the DK dump: it is non-zero exactly
    /// for the 31 that use LDS/STS, and in all of them at least the largest static LDS/STS
    /// offset (+ access size). Taken as the compute shared memory size in bytes. [INFERRED from
    /// data, not verified against NVN2 documentation.]
    u32 shared_memory_bytes{};
    /// Local memory of one compute thread in bytes: the warp's local memory over its 32 threads.
    [[nodiscard]] u32 LocalMemoryBytes() const noexcept {
        return local_memory_per_warp_bytes / 32;
    }
    bool has_compute_counts{};
    std::array<u8, 0x14> compute_raw{}; ///< control bytes 0xa8..0xbb
    ComputeDriverLayout compute;
    /// Task and mesh programs only (`stage_byte` 6 / 7).
    MeshControl mesh;
};

/// Parses and bounds-checks a control section. Returns nullopt and fills `error` on failure.
[[nodiscard]] std::optional<NvnControl> ParseControl(std::span<const u8> control,
                                                     std::string& error);

struct NvnProgramBinary {
    ShaderStage stage{};
    std::span<const u8> control;
    std::span<const u8> code;
};

/// Graphics stage of a binary according to the SPH at the start of its code area.
[[nodiscard]] std::optional<ShaderStage> StageFromSph(std::span<const u8> code);

/// Result of lowering. Pure data: thread-safe to share and cache.
struct LoweredNvnProgram {
    ShaderStage stage{};
    NvnControl control;
    /// [0x50-byte SPH][instructions, a zero scheduling word whenever (size*8)%32==0][0]
    /// [native address map], exactly as GenericEnvironment::LowerNativeCode packs it.
    std::vector<u64> code;
    u32 texture_bound{~0U};
    /// Constant bank of the bound surface (SULD/SUST/SUATOM) descriptors when the lowering named
    /// one (Ampere encodes it per instruction); ~0: the surfaces use the texture bank.
    u32 surface_bound{~0U};
    u32 source_size{};
    std::array<u32, SphBytes / sizeof(u32)> sph{};
    std::vector<u32> c1; ///< words of the program's constant segment
    bool capture_complete{};
    u32 instruction_count{};
    /// `Shader::Ampere::NativeProgramKey` of the captured instructions (names it in tools).
    u64 native_key{};
    std::string error;
    u32 error_offset{};
    std::array<u32, 4> error_words{};
    /// Instruction index (16-byte native instructions after the header) where the capture or the
    /// lowering stopped; ~0 when the rejection is not about one instruction (control section...).
    u32 failed_instruction{~0U};
    /// Mesh programs whose control section names a per-primitive header (MeshControl): that
    /// header's SPH, read from the code area. Empty otherwise.
    std::optional<std::array<u32, SphBytes / sizeof(u32)>> primitive_header;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty();
    }
};

/// Captures, lowers and packs a program. Compute programs lower their matrix multiplies (HMMA) as
/// `tensor` says; EDEN_NX2_TENSOR overrides it.
[[nodiscard]] std::shared_ptr<const LoweredNvnProgram> LowerNvnProgram(
    const NvnProgramBinary& binary, NativeTensorLowering tensor = NativeTensorLowering::None);

/// A constant-buffer word whose value the environment hands out as a synthetic handle.
struct NvnHandleRecord {
    u32 bank{};
    u32 offset{};
};

struct NvnTextureInfo {
    Shader::TextureType type{Shader::TextureType::Color2D};
    Shader::TexturePixelFormat format{Shader::TexturePixelFormat::A8B8G8R8_UNORM};
    bool is_integer{};
};
using NvnTextureInfoFn = std::function<NvnTextureInfo(const NvnHandleRecord&)>;

struct NvnEnvironmentOptions {
    /// Real texture types from the bound texture (the Vulkan side); default is Color2D RGBA8.
    NvnTextureInfoFn texture_info;
    /// Compute local/shared memory sizes. Unset: both come from the control section
    /// (NvnControl::LocalMemoryBytes, NvnControl::shared_memory_bytes).
    std::optional<u32> compute_local_memory_size;
    std::optional<u32> compute_shared_memory_size;
    /// Task and mesh programs: {SET_MESH_SHADER_A, SET_MESH_SHADER_B, SET_MESH_INIT_SHADER} of
    /// the pipeline (MeshRegisters over its task and mesh controls).
    std::array<u32, 3> mesh_registers{};
};

class NvnEnvironment final : public Shader::Environment {
public:
    explicit NvnEnvironment(std::shared_ptr<const LoweredNvnProgram> lowered,
                            NvnEnvironmentOptions options = {});
    ~NvnEnvironment() override;

    [[nodiscard]] u64 ReadInstruction(u32 address) override;
    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;
    /// Lowered Switch 2 code: the storage buffer pass looks at every pointer an address can
    /// come from (a predicated assignment merges two in a phi) instead of the first one found.
    [[nodiscard]] bool IsNativeProgram() const override;
    [[nodiscard]] bool IsNvnHleProgram() const override {
        return true;
    }
    /// Only the buffer entries of the driver bank hold addresses this path binds as storage
    /// buffers: the compute program's uniform and storage entry tables, or the stage's table.
    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 cbuf_index, u32 cbuf_offset) override;
    [[nodiscard]] u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;
    [[nodiscard]] Shader::TextureType ReadTextureType(u32 raw_handle) override;
    [[nodiscard]] Shader::TexturePixelFormat ReadTexturePixelFormat(u32 raw_handle) override;
    [[nodiscard]] bool IsTexturePixelFormatInteger(u32 raw_handle) override;
    [[nodiscard]] u32 ReadViewportTransformState() override;
    [[nodiscard]] u32 TextureBoundBuffer() const override;
    [[nodiscard]] u32 SurfaceBoundBuffer() const;
    [[nodiscard]] u32 LocalMemorySize() const override;
    [[nodiscard]] u32 SharedMemorySize() const override;
    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const override;
    [[nodiscard]] bool HasHLEMacroState() const override;
    [[nodiscard]] std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                              u32 offset) override;
    /// Compute programs: the uniform and storage entry tables of the control-derived driver bank.
    [[nodiscard]] Shader::NvnBufferWindow NvnComputeBufferWindow() const override;
    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    [[nodiscard]] const LoweredNvnProgram& Program() const noexcept {
        return *lowered;
    }
    /// Handle words handed out so far (key - 1 indexes this).
    [[nodiscard]] std::span<const NvnHandleRecord> HandleRecords() const noexcept {
        return records;
    }
    /// True if more distinct handle words were read than the synthetic key space holds.
    [[nodiscard]] bool HandleKeyOverflow() const noexcept {
        return key_overflow;
    }

private:
    [[nodiscard]] const NvnHandleRecord* RecordOf(u32 raw_handle) const noexcept;
    [[nodiscard]] NvnTextureInfo InfoOf(u32 raw_handle) const;

    std::shared_ptr<const LoweredNvnProgram> lowered;
    NvnEnvironmentOptions options;
    std::vector<NvnHandleRecord> records;
    std::unordered_map<u64, u32> record_index;
    u32 texture_bound{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};
    bool key_overflow{};
};

/// Pools translation needs (mirrors Vulkan::ShaderPools without the Vulkan header).
struct NvnShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

/// CFG + TranslateProgram, with the entry offsets the in-game path uses (0x50 for both graphics
/// and compute). Throws Shader::Exception types on translation failure. For geometry, the
/// caller still fills `previous_stage_stores` of the RuntimeInfo it emits with.
[[nodiscard]] Shader::IR::Program TranslateNvnStage(NvnShaderPools& pools, NvnEnvironment& env,
                                                    const Shader::HostTranslateInfo& host_info);

/// A Profile of an RTX-class device (SPIR-V 1.6, int8/16/64, float controls, subgroup ops,
/// demote, derivative control). For tests and the offline tool; the Vulkan slice passes its own.
[[nodiscard]] Shader::Profile MakeReferenceProfile();
[[nodiscard]] Shader::HostTranslateInfo MakeReferenceHostInfo();

} // namespace VideoCommon::Nvn
