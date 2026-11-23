// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// On-disk format of the NVN pipeline cache (`nvn_pipelines.bin`) and the file helpers around it.
// No Vulkan objects here: the layer is testable without a device.
//
//   file    := header record*
//   header  := "NVNPIPEL" u32 version u32 revision_len revision u32 key_len translation_key
//              u32 vendor u32 device u32 driver_version u64 cache_uuid_hash u64 profile_hash
//              u32 crc32(all before)
//              (translation_key names every input of the translated output, see
//              `MakeTranslationKey`; revision is the build that created the file, for diagnostics
//              only; profile_hash covers the shader profile and the vertex input mode)
//   record  := u32 tag u32 type u32 payload_size u32 crc32(type, size, payload) payload
//
// Integers are little-endian. Every length is bounds-checked against the bytes that remain; a
// record whose frame, size or checksum is wrong ends parsing and the records before it are kept.
// Records are only ever appended, one `write` of a complete record.

#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"

namespace Vulkan::Nvn::Disk {

/// 4: GraphicsFixedState gained `vertex_types` and, with VK_EXT_vertex_input_dynamic_state, no
/// longer holds the vertex layout; the vertex input mode is part of the header's profile hash.
/// 5: SR_WSCALEFACTOR is the guardband scale (RuntimeInfo::geometry_guardband_*) instead of a stub
/// constant, which changes the translated SPIR-V of programs that read it.
/// 6: Ampere SUST.P partial masks, SHF.HI, TLD.SCR.LZ.2D and HFMA2.RELU lowerings and the handle
/// program descriptors of image and texture-buffer accesses changed translated programs (layer 10).
/// 7: Ampere ISA lowerings (TLD4 per-tap and array gathers, TLD.MS arrays, ULDC offset folding, SUST
/// acceptance, unsigned I2F rounding) and typed-image `is_integer` descriptors changed translated
/// SPIR-V and binding flags; a gather lowered with a wrong offset type may have been cached.
/// 8: fragment shaders keep their own reads of constant bank 24 below 0x30 (BlitConstantsPass, which
/// changes their SPIR-V and bindings) and F2I with a negated or absolute constant-buffer operand
/// applies the modifier.
/// 9: the ISA oracle fixes (FFMA 0x623 operand negations, LEA/IADD3/IMAD negate and carry operands,
/// LDG descriptor addressing, TEX .F16 and LOD modes, TLD4 vector layout) changed lowered programs.
/// 10: the global memory fallback functions select storage buffers by their NVN slot, not by the
/// position in the descriptor list (SPIR-V of programs with raw NVN buffer access changed).
/// 11: the shader parity batch (bindless TEX/TXD/TLD4 uniform handles, register rename past R245,
/// BlitConstantsPass) changed the SPIR-V of previously translating programs under unchanged keys.
/// 12: the shader speed batch (float sign-chain folds, IMul/XMAD folds, unguarded scratch words,
/// F2I.TRUNC without a separate truncation, tensor staging, MUFU.RCP AllowRecip, subgroup atomic
/// combine, denorm mode classification) changed the SPIR-V of translating programs.
/// 13: the unused-result atomic combine became a bounded per-address waterfall (format 12 caches
/// built with the single-shot AllEqual combine hold different SPIR-V under the same keys).
/// 14: the program hash covers the compute control fields at +0x50..+0x6b (per-warp local memory,
/// thread group, shared memory), so every stored key changed.
/// 15: the header carries the translation key (build-time hash of the translation sources,
/// translation-relevant settings and environment switches, device and driver) instead of the git
/// revision as the compatibility rule; files of format 14 and older are rejected.
/// 16: task/mesh pipelines: graphics records hold 8 stage slots (NVNshaderStage 0..7, task 6 and
/// mesh 7), shader records may be stage 6/7, and task/mesh content ids hash the mesh registers.
/// 17: descriptor bindings carry the handle program of a texture whose handle the shader reads from
/// a table in memory (HandleSource::Program) and the storage index of a storage buffer.
inline constexpr u32 FORMAT_VERSION = VideoCommon::NVN_PIPELINE_CACHE_VERSION;
inline constexpr u32 RECORD_TAG = 0x4352564Eu; // "NVRC"
inline constexpr u32 MAX_RECORD_BYTES = 256u << 20;
inline constexpr size_t FRAME_BYTES = 16;

enum class RecordType : u32 {
    Shader = 1,
    GraphicsPipeline = 2,
    ComputePipeline = 3,
};

/// 128-bit content id of a translated shader: stage, program hash, hash of the runtime info /
/// binding base it was emitted with, hash of the texture queries it depended on. Never zero for
/// a real shader (zero = "absent stage").
struct ContentId {
    u64 lo{};
    u64 hi{};
    [[nodiscard]] bool operator==(const ContentId&) const = default;
    [[nodiscard]] bool IsZero() const noexcept {
        return lo == 0 && hi == 0;
    }
};
struct ContentIdHash {
    [[nodiscard]] size_t operator()(const ContentId& id) const noexcept {
        return static_cast<size_t>(id.lo ^ (id.hi * 0x9E3779B97F4A7C15ull));
    }
};

[[nodiscard]] ContentId MakeContentId(u32 stage, u64 program_hash, u64 runtime_hash,
                                      u64 queries_hash);
/// Hash of a query list (order matters: it is the translation's handle order).
[[nodiscard]] u64 HashQueries(std::span<const TextureQuery> queries);

/// Rewrites `state` to the form that is hashed, compared and stored: every bit pipeline creation
/// does not read is cleared and every boolean is 0/1. The NVN state words come from guest state
/// objects whose bits outside the fields (polygon bits 8..31, depth-stencil bits 39..63, blend
/// target index and high bits, multisample bytes the pipeline ignores, ...) hold whatever the
/// guest left in memory; raw they make every draw a new key and no key survives a launch. Apply
/// before computing a GraphicsKey and before encoding a GraphicsPipelineRecord. Idempotent.
/// `mode` selects the vertex layout form (see `MakeKeyState`): Static keeps the attribute words,
/// strides and divisors and zeroes `vertex_types`; Dynamic zeroes the words and keeps
/// `vertex_types` (clamped to valid AttributeType values).
void CanonicalizeState(GraphicsFixedState& state, VertexInputMode mode);

struct Header {
    /// Build that created the file. Diagnostics only: never compared.
    std::string revision;
    /// Everything the stored SPIR-V depends on (`MakeTranslationKey`); a file whose key differs
    /// from the running build's is discarded.
    std::string translation_key;
    u32 vendor_id{};
    u32 device_id{};
    u32 driver_version{};
    u64 cache_uuid_hash{};
    u64 profile_hash{};
    [[nodiscard]] bool operator==(const Header&) const = default;
};

inline constexpr u32 SHADER_STORES_VIEWPORT_INDEX = 1u << 0;

/// One translated stage: everything a pipeline needs of it without translating again.
struct ShaderRecord {
    ContentId id;
    u32 stage{}; ///< NVNshaderStage
    u64 program_hash{};
    u32 flags{};
    u32 attributes_read{};  ///< vertex stage: generic input locations read
    u32 outputs_written{};  ///< fragment stage: color outputs written
    std::array<u32, 3> workgroup_size{};
    u32 shared_memory_size{};
    bool has_program_constants{};
    std::vector<u32> program_constants;
    std::vector<u32> spirv;
    /// The bindings of this stage only; `binding_count` is the value after this stage was
    /// emitted, `unsupported_count` counts this stage's unsupported bindings.
    VC::DescriptorLayout layout;
    std::vector<TextureQuery> queries;
};

struct GraphicsPipelineRecord {
    std::array<ContentId, GRAPHICS_STAGE_COUNT> shaders{}; ///< by NVNshaderStage, zero = absent
    GraphicsFixedState state{};
    RenderPassKey targets;
    u32 push_constant_size{};
    u32 viewport_count{1};
};

struct ComputePipelineRecord {
    ContentId shader;
    u32 push_constant_size{};
    /// ComputeRequest::environment overrides (present flags in bit 0 / bit 1).
    u32 environment_flags{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    VC::ComputeDriverLayout compute_layout;
};

struct Contents {
    std::vector<ShaderRecord> shaders;
    std::vector<GraphicsPipelineRecord> graphics;
    std::vector<ComputePipelineRecord> compute;
};

enum class LoadStatus {
    Missing,  ///< no file (or empty): start fresh, nothing rejected
    Rejected, ///< header damaged or from another version/revision/device: discard everything
    Loaded,   ///< header accepted; `Contents` holds every valid record before the first bad one
};

struct LoadResult {
    LoadStatus status{LoadStatus::Missing};
    std::string reason;       ///< Rejected: why; Loaded with a bad tail: what ended the parse
    size_t valid_bytes{};     ///< header + valid records
    size_t input_bytes{};
    size_t records{};
    [[nodiscard]] bool HasBadTail() const noexcept {
        return status == LoadStatus::Loaded && valid_bytes != input_bytes;
    }
};

[[nodiscard]] u32 Crc32(std::span<const u8> data, u32 crc = 0);

[[nodiscard]] std::vector<u8> EncodeHeader(const Header& header);
/// Complete framed records, ready for one append.
[[nodiscard]] std::vector<u8> EncodeShader(const ShaderRecord& record);
[[nodiscard]] std::vector<u8> EncodeGraphicsPipeline(const GraphicsPipelineRecord& record);
[[nodiscard]] std::vector<u8> EncodeComputePipeline(const ComputePipelineRecord& record);

/// Parses a whole file image. Never reads out of bounds; malformed input is reported in the
/// result, but allocation failures (std::bad_alloc) propagate.
[[nodiscard]] LoadResult Parse(std::span<const u8> bytes, const Header& expected,
                               Contents& contents);

[[nodiscard]] std::vector<u8> SerializeDescriptorLayout(const VC::DescriptorLayout& layout);
/// Consumes exactly the span; false (and `out` unspecified) on any inconsistency.
[[nodiscard]] bool DeserializeDescriptorLayout(std::span<const u8> bytes,
                                               VC::DescriptorLayout& out);

/// Reads and parses `path`. A missing file is `Missing`.
[[nodiscard]] LoadResult LoadFile(const std::filesystem::path& path, const Header& expected,
                                  Contents& contents);

/// Makes `path` ready for appending after `load`: a rejected file is replaced by a fresh one
/// holding only the header (written beside it and renamed over), a file with a bad tail is cut to
/// its valid prefix, a missing file is created. False when the file cannot be prepared.
[[nodiscard]] bool PrepareFile(const std::filesystem::path& path, const Header& header,
                               const LoadResult& load);

/// Append-only writer; each Append is one `write` of a complete record followed by a flush.
class Appender {
public:
    [[nodiscard]] bool Open(const std::filesystem::path& path);
    [[nodiscard]] bool Append(std::span<const u8> record);
    [[nodiscard]] bool IsOpen() const noexcept {
        return file.is_open();
    }

private:
    std::ofstream file;
};

/// Writes `data` beside `path` and renames it over `path`.
[[nodiscard]] bool WriteFileAtomic(const std::filesystem::path& path, std::span<const u8> data);

/// Identity of the translation sources this binary was built from: SHA-256 (hex) over every file
/// that decides translated SPIR-V (shader_recompiler, the NVN shader lowering, the pipeline
/// translation and key code, the sirit pin), computed at every build by
/// CMakeModules/GenerateTranslationEpoch.cmake. A build that changes none of them (a comment in
/// another module, a new git revision) has the same epoch and reuses the cache file.
[[nodiscard]] std::string_view TranslationEpoch();

/// Every input of the cached output that is not part of a record key.
struct TranslationInputs {
    std::string_view epoch;   ///< `TranslationEpoch()`
    u64 switches_hash{};      ///< `HashTranslationSwitches` (settings and environment switches)
    u32 vendor_id{};
    u32 device_id{};
    u32 driver_version{};
    u64 cache_uuid_hash{};
    u64 profile_hash{};       ///< shader profile, host translate info, vertex input mode
};

/// The compatibility rule of a cache file: the file was written for this key or it is discarded.
/// Readable text (it is stored in the header) that includes `FORMAT_VERSION`, so a manual bump
/// of the format still invalidates everything; the epoch covers the translation sources, so a
/// forgotten bump cannot serve stale SPIR-V.
[[nodiscard]] std::string MakeTranslationKey(const TranslationInputs& inputs);

/// Name of the cache file of `key`: `nvn_pipelines.k<16 hex digits>.bin`. Every key keeps its own
/// file, so builds that translate differently, other device or driver, or another set of
/// switches that share one profile directory (A/B runs, a release next to a development tree)
/// do not throw each other's cache away, and builds that translate alike share one.
[[nodiscard]] std::string FileNameForKey(std::string_view key);

/// Removes the caches of other keys in `directory` once there are more than `keep` files (this
/// key's `keep_path` included), oldest write time first. Includes files of the earlier
/// per-revision naming (`nvn_pipelines.<12 characters>.bin`): they are never read again (format
/// 14 and older are rejected) and age out here. `keep_path` gets a fresh write time so the files
/// in use stay. Failures are ignored: a stale file only costs disk space.
void PruneCacheFiles(const std::filesystem::path& directory,
                     const std::filesystem::path& keep_path, size_t keep);

} // namespace Vulkan::Nvn::Disk
