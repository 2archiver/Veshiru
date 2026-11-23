// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Test-only extractor of NVN2 (Switch 2) shader programs from the containers of a game dump:
// zstd -> SARC -> BNSH / BAHS / NDSR / sead primitive drawer / NSO. Ported from the container logic
// of nxnative's dm/gfx/tools/shader_split.py and checked against the Donkey Kong Bananza dump.
//
// Nothing in here throws or aborts on malformed input: every problem becomes a Failure whose
// reason starts with "container:". The code is deliberately kept out of video_core (no zstd/LZ4
// dependency there, no file parsing in the library).

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::Nvn::Containers {

/// NVN2 control section constants (nxnative dmsh_ampere.py).
inline constexpr u32 ControlMagic = 0x98761234;
inline constexpr u32 ControlVersion = 0xc9;
/// Control bytes covered by the dedupe key (the +0x30 own-length word is zeroed).
inline constexpr size_t ControlKeyBytes = 0x48;
/// Size of a control section as stored in a BNSH (+0x30 holds this value there).
inline constexpr u32 ControlSize = 0xd8;

struct ExtractedShader {
    std::string source; // container path label, e.g. "romfs/Shader/X.bfsha.zs>X.bnsh#var3.vs"
    std::string kind;   // "bnsh" | "bahs" | "ndsr" | "sead" | "nso"
    u32 stage_byte = 0; // control +0x56 (NVNshaderStage: 0 VS, 1 FS, 2 GS, 3 TCS, 4 TES, 5 CS)
    std::vector<u8> control; // control section (copied, as large as found in the container)
    std::vector<u8> code;    // code area, size == u32 at control+0x44
    std::string key;         // dedupe key, see ShaderKey
};

struct Failure {
    std::string source;
    std::string reason; // starts with "container:"
};

struct ScanOutput {
    std::vector<ExtractedShader> shaders; // already deduped by key (first occurrence wins)
    std::vector<Failure> failures;
    size_t duplicates = 0;
    /// Keys of `shaders`, used for the O(1) dedupe. ScanBlob maintains it; callers need not touch it
    /// unless they push to `shaders` themselves.
    std::unordered_set<std::string> seen_keys;
};

/// Recursive scan of one blob (zstd -> SARC -> BNSH/BAHS/NDSR/sead(by name_hint)/NSO).
/// `label` doubles as the sead name hint: a label containing "primitive_drawer_nvn2_shader" is
/// parsed as sead's primitive drawer. Never throws, never aborts.
void ScanBlob(std::span<const u8> blob, const std::string& label, ScanOutput& out);

/// Reads the whole file and calls ScanBlob. Returns false and records a failure if unreadable.
bool ScanFile(const std::filesystem::path& file, const std::string& label, ScanOutput& out);

/// Walks root recursively (sorted, so results are deterministic) and scans every regular file that
/// could hold shaders: *.zs (except known non-shader payloads such as *.bntx.zs), *.bfsha, *.bnsh,
/// *.sarc, *.pck, *.bin, *.bahs and the file named `main` under an `exefs` directory. Labels are
/// the paths relative to root (generic_string). Scanning a file and finding nothing is not a
/// failure.
void ScanRoot(const std::filesystem::path& root, ScanOutput& out);

/// Dedupe key: hex of the first 16 bytes of SHA-256 over (control[0..0x48) with the u32 at +0x30
/// zeroed) || code. Returns an empty string when control is shorter than 0x48 bytes.
std::string ShaderKey(std::span<const u8> control, std::span<const u8> code);

/// Lowercase hex SHA-256 of `data` (exposed so tests can pin the implementation).
std::string Sha256Hex(std::span<const u8> data);

// ---------------------------------------------------------------------------------------------
// Builders for synthetic containers (tests only).
// ---------------------------------------------------------------------------------------------

/// A valid 0xd8-byte NVN2 control section: magic at +0, version 0xc9 at +4, own length 0xd8 at
/// +0x30, instruction bytes at +0x38, round_up(inst_bytes, 0x100) at +0x40, code area length at
/// +0x44, register count 16 at +0x4c, stage byte at +0x56 and, for compute (5), a 32x1x1 thread
/// group at +0x58.
std::vector<u8> MakeControl(u32 stage_byte, u32 code_bytes_total, u32 inst_bytes);

/// `size` bytes of deterministic non-trivial code, different for every `seed`.
std::vector<u8> MakeCode(size_t size, u32 seed);

/// One stage of a BNSH variation: `slot` is the descriptor index at program+8 (0 VS, 1 HS, 2 DS,
/// 3 GS, 4 PS, 5 CS in BNSH order).
struct BnshStage {
    u32 slot = 0;
    std::vector<u8> control;
    std::vector<u8> code;
};
using BnshVariation = std::vector<BnshStage>;

/// A BNSH with the given variations (variation with no stages gets a zero program pointer), laid
/// out like the real files: header with `grsc` at +0x60, count at +0x7c, table pointer at +0x80,
/// 0x40-byte variations (program pointer at +0x10), programs (stage pointers at +8), stage
/// descriptors {+8 control pointer, +0x10 code pointer, +0x18 code size, +0x1c control size}.
/// Pointers are relative to the BNSH start.
std::vector<u8> MakeBnsh(std::span<const BnshVariation> variations);

/// A (little-endian unless `big_endian`) SARC with the given named members.
std::vector<u8> MakeSarc(std::span<const std::pair<std::string, std::vector<u8>>> members,
                         bool big_endian = false);

/// One BAHS entry: control section plus the code area that goes into the code region.
struct BahsEntry {
    std::vector<u8> control;
    std::vector<u8> code;
};
/// A BAHS file: header with the code region size at +0x18 and start at +0x1c, a 16-byte entry per
/// shader {code offset in region, 0, control size, signed offset from the entry to the control}
/// followed by the control sections, then the code region (each code area 0x100-aligned).
std::vector<u8> MakeBahs(std::span<const BahsEntry> entries);

/// An NDSR pack: 'NDSR', 16-byte records from +0x10 {control offset, control size << 4 | type,
/// code offset, code size / 16 << 8 | type}, a zero record and the data.
std::vector<u8> MakeNdsr(std::span<const BahsEntry> shaders);

/// sead's primitive drawer: four u32 offsets (VS control, PS control, VS code, PS code) followed
/// by the data.
std::vector<u8> MakeSeadPrimitiveDrawer(const BahsEntry& vertex, const BahsEntry& pixel);

/// zstd-compress into one frame. With `record_content_size == false` the frame header carries no
/// content size (like many files in real dumps), which forces the streaming decode path.
std::vector<u8> ZstdCompress(std::span<const u8> data, bool record_content_size = true);

} // namespace VideoCommon::Nvn::Containers
