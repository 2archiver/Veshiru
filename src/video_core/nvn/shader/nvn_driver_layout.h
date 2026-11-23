// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// NVN2 constant-buffer ABI that shaders read: which bank holds what, and where the handle and
// storage-buffer tables sit inside it. The layout is ported from nxnative
// (docs/agents/04-graphics.md section 3 and hle_nvn.c); every offset below is `[UNVERIFIED]`
// against NVN2 hardware behaviour except where the coverage tool measured it.

#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/common_types.h"
#include "shader_recompiler/stage.h"

namespace VideoCommon::Nvn {

/// Constant buffer banks, as the guest shader addresses them.
constexpr u32 DriverBank = 0;       ///< c0: driver table (storage-buffer descriptors, per stage)
constexpr u32 ProgramBank = 1;      ///< c1: the program's own constant segment (never bound)
constexpr u32 HandleBank = 2;       ///< c2: texture/sampler/image handle tables
constexpr u32 FirstUniformBank = 3; ///< c3..: uniform buffers bound by the game (slot = bank - 3)

/// NVN2 shaders read the push-constant block as constant bank 24 (root table 0 of the hardware,
/// 256 bytes; Maxwell3D::RootTables on the emulated path). The Ampere lowering turns bank 24 into
/// bank 17, the last one the recompiler has, which games do not bind (uniform slot 14).
constexpr u32 PushConstantBank = 17;
constexpr u32 PushConstantBankSize = 0x100;

/// Handle tables inside the handle bank. Every entry is 8 bytes; the handle is the first word.
constexpr u32 HandleStride = 8;
constexpr u32 CombinedHandleBase = 0x20; ///< 41 combined texture+sampler handles (slots 0..40)
constexpr u32 CombinedHandleEnd = 0x168;
constexpr u32 CombinedHandleSlots = (CombinedHandleEnd - CombinedHandleBase) / HandleStride;
constexpr u32 ImageHandleBase = 0x120;   ///< 8 image handles; overlaps combined slots 32..40
constexpr u32 ImageHandleEnd = 0x160;
constexpr u32 ImageHandleSlots = (ImageHandleEnd - ImageHandleBase) / HandleStride;
constexpr u32 SeparateTextureBase = 0x168; ///< 128 separate texture handles
constexpr u32 SeparateTextureSlots = 128;
constexpr u32 SeparateSamplerBase = 0x568; ///< 128 separate sampler handles (id << 20)
constexpr u32 SeparateSamplerSlots = 128;
constexpr u32 HandleBankEnd = 0x968;
static_assert(SeparateTextureBase + SeparateTextureSlots * HandleStride == SeparateSamplerBase);
static_assert(SeparateSamplerBase + SeparateSamplerSlots * HandleStride == HandleBankEnd);

/// Storage-buffer descriptors in the driver bank: 16 per stage, 0x10 bytes each
/// ({u64 address, u32 size, u32 pad}).
constexpr u32 StorageBufferStride = 0x10;
constexpr u32 StorageBufferSlots = 16;

/// Words the NVN2 draw macros write into the driver bank per draw (MME code of Ocarina of Time's
/// NVN, captured on the native path): the index of the draw within a multi-draw (as the vertex
/// multi-draw macros do), and the {x, y, z} workgroup counts of a 3D mesh-task draw, which the
/// hardware launches as x*y*z workgroups in one dimension (the task program linearizes with them).
constexpr u32 DriverDrawIdOffset = 0x648;
constexpr u32 DriverMeshDimsOffset = 0x8e0;
/// The base instance of an instanced draw, where vertex programs read gl_BaseInstance. The SDK's
/// nvnCommandBufferDrawElementsInstanced / DrawArraysInstanced call another macro when the base
/// instance (or, for elements, the base vertex) is not 0; replayed from OoT's macro RAM, the
/// elements one writes {base vertex, base instance} to c0+0x640/0x644, the arrays one the base
/// instance to c0+0x644, both draw and then write the words back to 0. The draws through the other
/// macros leave them 0, so the word is the draw's base instance for every draw. The base vertex
/// word (c0+0x640) is not provided: it is 0 for non-indexed draws, which Vulkan's BaseVertex is
/// not, and none of OoT's programs read it.
constexpr u32 DriverBaseInstanceOffset = 0x644;

/// Shader stages in NVN's NVNshaderStage order (control section byte +0x56). NVN2 adds the task
/// (6) and mesh (7) stages, each with its own storage table in the driver bank (0x670 / 0x770, see
/// StorageBufferBase).
enum class ShaderStage : u8 { Vertex, Fragment, Geometry, TessControl, TessEval, Compute, Task, Mesh };
constexpr u32 ShaderStageCount = 8;

[[nodiscard]] constexpr Shader::Stage ToRecompilerStage(ShaderStage stage) noexcept {
    switch (stage) {
    case ShaderStage::Vertex:
        return Shader::Stage::VertexB;
    case ShaderStage::Fragment:
        return Shader::Stage::Fragment;
    case ShaderStage::Geometry:
        return Shader::Stage::Geometry;
    case ShaderStage::TessControl:
        return Shader::Stage::TessellationControl;
    case ShaderStage::TessEval:
        return Shader::Stage::TessellationEval;
    case ShaderStage::Compute:
        return Shader::Stage::Compute;
    case ShaderStage::Task:
        return Shader::Stage::Task;
    case ShaderStage::Mesh:
        return Shader::Stage::Mesh;
    }
    return Shader::Stage::VertexB;
}

[[nodiscard]] constexpr std::string_view StageName(ShaderStage stage) noexcept {
    switch (stage) {
    case ShaderStage::Vertex:
        return "vs";
    case ShaderStage::Fragment:
        return "fs";
    case ShaderStage::Geometry:
        return "gs";
    case ShaderStage::TessControl:
        return "tcs";
    case ShaderStage::TessEval:
        return "tes";
    case ShaderStage::Compute:
        return "cs";
    case ShaderStage::Task:
        return "task";
    case ShaderStage::Mesh:
        return "mesh";
    }
    return "?";
}

/// Graphics stage from SPH word 0 bits 10..13 (1 VS, 2 HS, 3 DS, 4 GS, 5 PS).
[[nodiscard]] constexpr std::optional<ShaderStage> StageFromSphType(u32 shader_type) noexcept {
    switch (shader_type) {
    case 1:
        return ShaderStage::Vertex;
    case 2:
        return ShaderStage::TessControl;
    case 3:
        return ShaderStage::TessEval;
    case 4:
        return ShaderStage::Geometry;
    case 5:
        return ShaderStage::Fragment;
    default:
        return std::nullopt;
    }
}

/// Offset of the first storage-buffer descriptor of a graphics stage in the driver bank.
/// nvnCommandBufferBindStorageBuffer (Ocarina of Time's SDK, 0x67ce70) writes the {address, size}
/// of a graphics stage's slot into the driver bank at (first[stage] + slot) * 0x10, from a per-stage
/// table (sdk+0xbb27d0, in 16-byte entries): vertex 0x11, fragment 0x51, geometry 0x41, tess
/// control 0x21, tess eval 0x31, task 0x67, mesh 0x77. OoT's task and mesh programs read their
/// slots there (task c0+0x670/0x680/0x690, mesh c0+0x770). The recompiler's CollectShaderInfoPass
/// uses the same bases. Compute does not use a fixed base: see ComputeDriverLayout in
/// nvn_program_map.h.
[[nodiscard]] constexpr u32 StorageBufferBase(ShaderStage stage) noexcept {
    switch (stage) {
    case ShaderStage::Vertex:
        return 0x110;
    case ShaderStage::TessControl:
        return 0x210;
    case ShaderStage::TessEval:
        return 0x310;
    case ShaderStage::Geometry:
        return 0x410;
    case ShaderStage::Fragment:
        return 0x510;
    case ShaderStage::Task:
        return 0x670;
    case ShaderStage::Mesh:
        return 0x770;
    case ShaderStage::Compute:
        return 0x310; // what the recompiler assumes; the real base is per shader
    }
    return 0;
}

/// Slot of a storage-buffer descriptor at `offset` of the driver bank, if it is one.
[[nodiscard]] constexpr std::optional<u32> StorageBufferSlot(u32 base, u32 offset) noexcept {
    if (offset < base || (offset - base) % StorageBufferStride != 0) {
        return std::nullopt;
    }
    const u32 slot = (offset - base) / StorageBufferStride;
    if (slot >= StorageBufferSlots) {
        return std::nullopt;
    }
    return slot;
}

[[nodiscard]] constexpr std::optional<u32> StorageBufferSlot(ShaderStage stage,
                                                             u32 offset) noexcept {
    return StorageBufferSlot(StorageBufferBase(stage), offset);
}

/// A 32-bit NVN texture handle: texture id in bits 0..19, sampler id in bits 20..31.
[[nodiscard]] constexpr u32 TextureIdOf(u32 handle) noexcept {
    return handle & 0xFFFFF;
}
[[nodiscard]] constexpr u32 SamplerIdOf(u32 handle) noexcept {
    return handle >> 20;
}
[[nodiscard]] constexpr u32 MakeHandle(u32 texture_id, u32 sampler_id) noexcept {
    return (texture_id & 0xFFFFF) | (sampler_id << 20);
}
/// Which handle table a word belongs to (it decides which of its ids are pool-relative).
enum class HandleWordKind : u8 {
    Texture,         ///< combined or separate texture word: texture id and sampler id
    SeparateSampler, ///< separate sampler word: only the sampler id (bits 20..)
    Image,           ///< image word: only the texture id
};

/// Moves a handle word from the game's pool-relative ids into the global descriptor id space
/// (each texture / sampler pool owns a range of it; `texture_base` / `sampler_base` are the bases
/// of the pools the command buffer bound). Ids of 0 stay 0. The hardware does this implicitly: the
/// bound pool is the table the ids index. The recorder applies it to the handles of the bind
/// commands; a handle read from memory (a uniform buffer word the game filled with
/// nvnDeviceGetTextureHandle results) is rebased by the replay with the pools bound when it is read.
[[nodiscard]] constexpr u64 RebaseHandleWord(u32 word, HandleWordKind kind, u32 texture_base,
                                             u32 sampler_base) noexcept {
    u64 texture_id = word & 0xFFFFF;
    u64 sampler_id = word >> 20;
    if (kind == HandleWordKind::SeparateSampler) {
        return sampler_id ? (sampler_id + sampler_base) << 20 : word;
    }
    if (texture_id) {
        texture_id += texture_base;
    }
    if (sampler_id && kind == HandleWordKind::Texture) {
        sampler_id += sampler_base;
    }
    return texture_id | sampler_id << 20;
}

/// What a separate sampler table entry holds for sampler `id`.
[[nodiscard]] constexpr u32 SeparateSamplerWord(u32 sampler_id) noexcept {
    return sampler_id << 20;
}

/// Where a handle word the shader reads comes from.
enum class HandleSource : u8 {
    None,             ///< no word (e.g. the sampler of a combined handle)
    Combined,         ///< combined texture+sampler handle table of the handle bank
    SeparateTexture,  ///< separate texture handle table
    SeparateSampler,  ///< separate sampler handle table
    Image,            ///< image handle table
    UniformBuffer,    ///< a word of a bound uniform buffer (slot = bank - 3)
    ComputeTexture,   ///< compute: texture handle table inside the per-shader driver bank
    ComputeImage,     ///< compute: image handle table inside the per-shader driver bank
    Program,          ///< computed per draw from words of uniform and storage buffers (slot = index
                      ///< in Shader::Info::handle_programs): a handle table the shader reads
    Unsupported,
};

[[nodiscard]] constexpr std::string_view HandleSourceName(HandleSource source) noexcept {
    switch (source) {
    case HandleSource::None:
        return "none";
    case HandleSource::Combined:
        return "combined";
    case HandleSource::SeparateTexture:
        return "separate-texture";
    case HandleSource::SeparateSampler:
        return "separate-sampler";
    case HandleSource::Image:
        return "image";
    case HandleSource::UniformBuffer:
        return "uniform-buffer";
    case HandleSource::ComputeTexture:
        return "compute-texture";
    case HandleSource::ComputeImage:
        return "compute-image";
    case HandleSource::Program:
        return "program";
    case HandleSource::Unsupported:
        return "unsupported";
    }
    return "?";
}

/// What kind of descriptor asks for the handle: decides the meaning of the ambiguous
/// 0x120..0x167 range of the handle bank.
enum class HandleWant : u8 { Texture, Image };

/// Per-shader compute driver bank layout (derived from the control section, see
/// nvn_program_map.h). Tables are searched when classifying bank-0 words of a compute shader.
struct ComputeHandleTables {
    u32 texture_offset{};
    u32 texture_count{};
    u32 image_offset{};
    u32 image_count{};
};

struct HandleClassification {
    HandleSource source{HandleSource::Unsupported};
    u32 slot{};
    std::string reason; ///< why the word is Unsupported; empty otherwise
};

/// Classifies one handle word, addressed by (bank, byte offset). Everything is decided from the
/// descriptor kind `want`: the 0x120..0x167 range of the handle bank is combined slot 32+n for a
/// texture and image slot n for an image.
[[nodiscard]] inline HandleClassification ClassifyHandleWord(
    u32 bank, u32 offset, HandleWant want, const ComputeHandleTables* compute = nullptr) {
    const auto unsupported = [](std::string reason) {
        return HandleClassification{HandleSource::Unsupported, 0, std::move(reason)};
    };
    if (bank >= FirstUniformBank) {
        return {HandleSource::UniformBuffer, bank - FirstUniformBank, {}};
    }
    if (bank == DriverBank) {
        if (compute == nullptr) {
            return unsupported("handle read from the driver bank");
        }
        const auto in_table = [offset](u32 base, u32 count) {
            return offset >= base && offset < base + count * HandleStride;
        };
        const bool in_texture = in_table(compute->texture_offset, compute->texture_count);
        const bool in_image = in_table(compute->image_offset, compute->image_count);
        // A handle word is an index into the pool whichever table holds it: a shader may sample
        // (texture op) a handle of its image table, or run image ops on one of its texture table.
        const bool texture_table = in_texture;
        if (in_texture || in_image) {
            const u32 base = texture_table ? compute->texture_offset : compute->image_offset;
            if ((offset - base) % HandleStride != 0) {
                return unsupported("misaligned compute handle offset");
            }
            return {texture_table ? HandleSource::ComputeTexture : HandleSource::ComputeImage,
                    (offset - base) / HandleStride,
                    {}};
        }
        return unsupported(fmt::format(
            "compute handle outside the texture/image tables (textures {}x at {:#x}, images {}x at "
            "{:#x})",
            compute->texture_count, compute->texture_offset, compute->image_count,
            compute->image_offset));
    }
    if (bank == ProgramBank) {
        return unsupported("handle read from the program constant bank");
    }
    // bank == HandleBank
    if (offset % HandleStride != 0) {
        return unsupported("misaligned handle offset");
    }
    if (offset < CombinedHandleBase || offset >= HandleBankEnd) {
        return unsupported("handle offset outside the handle tables");
    }
    if (offset < ImageHandleBase) {
        if (want == HandleWant::Image) {
            return unsupported("image handle in the combined texture range");
        }
        return {HandleSource::Combined, (offset - CombinedHandleBase) / HandleStride, {}};
    }
    if (offset < CombinedHandleEnd) {
        if (want == HandleWant::Texture) {
            return {HandleSource::Combined, (offset - CombinedHandleBase) / HandleStride, {}};
        }
        if (offset >= ImageHandleEnd) {
            return unsupported("image handle slot out of range");
        }
        return {HandleSource::Image, (offset - ImageHandleBase) / HandleStride, {}};
    }
    if (want == HandleWant::Image) {
        return unsupported("image handle in the separate texture/sampler range");
    }
    if (offset < SeparateSamplerBase) {
        return {HandleSource::SeparateTexture, (offset - SeparateTextureBase) / HandleStride, {}};
    }
    return {HandleSource::SeparateSampler, (offset - SeparateSamplerBase) / HandleStride, {}};
}

} // namespace VideoCommon::Nvn
