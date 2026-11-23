// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

/// Path diff (`EDEN_NX2_PATHDIFF`, diagnostic, off by default): what one program sees at its
/// dispatch or draw, captured the same way on the emulated GPU path (the game's own NVN driver,
/// the reference) and on the NVN HLE path, so `tools/pathdiff.py` (C:/eg/tools/pathdiff.py) can
/// name the first input that differs.
///
///   EDEN_NX2_PATHDIFF=<hash>[,<hash>...]:<present>[:<count>]
///
/// `<hash>` is a program's stage hash on either path or its code key (CodeKey: the same program
/// has different stage hashes on the two paths, the code key is equal). From present `<present>`
/// on (1-based, counted by each path), the first `<count>` (default 8) dispatches/draws of those
/// programs are captured. Per capture, one JSON (ToJson) and one blob with the bytes:
///   * every constant buffer the shader binds, as the GPU sees it at the dispatch;
///   * every storage buffer the shader binds (with the constant buffer word its address and size
///     come from), before and, when written, after the dispatch;
///   * for each u64 in a constant buffer that is a mapped address on that path, the bytes behind
///     it (`EDEN_NX2_PATHDIFF_CHASE` bytes, default 4096), before and after: raw pointer loads;
///   * texture/sampler descriptors (format, size, levels, swizzle, filter, wrap, border, compare);
///   * group counts (indirect: the arguments as the GPU reads them) or draw counts.
/// Files go to `EDEN_NX2_PATHDIFF_DIR` (default <log dir>/pathdiff)/<path>-<time>/.
/// Off: Enabled() is one relaxed load and no path records anything.
namespace VideoCommon::PathDiff {

enum class Path : u8 { Emulated, Hle };
[[nodiscard]] std::string_view PathName(Path path) noexcept;

struct Spec {
    std::vector<u64> hashes; ///< stage hashes (either path) or code keys
    u64 present{};           ///< first present captured (1-based)
    u32 count{8};            ///< captures per path
    u32 chase_bytes{4096};   ///< bytes read behind each pointer
    u32 max_pointers{64};    ///< pointers chased per capture
    u64 max_range{u64{16} << 20}; ///< bytes kept of one buffer range
};

/// `<hash>[,<hash>...]:<present>[:<count>]`; hashes in hex with or without 0x.
[[nodiscard]] std::optional<Spec> ParseSpec(std::string_view text, std::string& error);

namespace detail {
extern std::atomic<bool> g_enabled;
} // namespace detail

/// True when EDEN_NX2_PATHDIFF holds a valid spec (read once).
[[nodiscard]] inline bool Enabled() noexcept {
    return detail::g_enabled.load(std::memory_order_relaxed);
}
/// The spec (valid while Enabled()). The first call reads the environment.
[[nodiscard]] const Spec& Config();
/// Tests: replaces the spec (nullopt: off).
void SetConfigForTest(std::optional<Spec> spec);

/// The code key of a native Ampere program: the program is captured instruction by instruction
/// with the rules both paths use (Shader::Ampere::ProgramCaptureState; `code` starts at the
/// first instruction, `header_bytes` is what the footer size counts before it), then hashed with
/// Shader::Ampere::NativeProgramKey. Equal to LoweredNativeCode::native_key of the emulated path
/// for the same program; trailing padding or data after the end does not change it.
[[nodiscard]] u64 CodeKey(std::span<const u8> code, u32 header_bytes);

/// Stage hash -> code key, filled where a path learns both (thread safe).
void NoteCodeKey(u64 stage_hash, u64 code_key);
[[nodiscard]] u64 CodeKeyOf(u64 stage_hash); ///< 0: unknown

/// Which draws/dispatches of one path are captured. One per path, used by its recording thread.
class Selector {
public:
    /// The ordinal of this use of the program within `present` (counted for every use of a
    /// selected program from the first selected present on) when it is to be captured; nullopt
    /// otherwise. `present` is 1-based.
    [[nodiscard]] std::optional<u32> Claim(u64 present, u64 stage_hash, u64 code_key);
    [[nodiscard]] bool Done() const noexcept;

private:
    u64 current_present{};
    std::unordered_map<u64, u32> ordinals;
    u32 taken{};
};

/// One buffer range of a capture.
struct Range {
    std::string role;   ///< "cbuf", "storage", "pointer", "indirect"
    u32 index{};        ///< cbuf index; storage: descriptor order; pointer: chase order
    std::string source; ///< path-independent identity: "c3", "c0+0x90" (storage: where its
                        ///< address/size pair lives; pointer: where the pointer lives)
    u64 address{};      ///< GPU address on this path (differs between paths)
    u64 bound_size{};   ///< size the binding names
    bool written{};
    std::string view{"gpu"}; ///< "gpu": copied in GPU order; "guest": CPU memory at record time
    std::vector<u8> before;
    std::vector<u8> after;
    bool has_after{};
};

struct Sampler {
    u32 mag_filter{};
    u32 min_filter{};
    u32 mipmap_mode{};
    std::array<u32, 3> address{};
    float lod_bias{};
    float min_lod{};
    float max_lod{};
    bool anisotropy{};
    float max_anisotropy{};
    bool compare{};
    u32 compare_op{};
    u32 border_color{};            ///< VkBorderColor
    std::array<float, 4> border{}; ///< the custom border colour (RGBA)
    u32 reduction{};               ///< VkSamplerReductionMode
};

struct Texture {
    std::string source; ///< "c<bank>+0x<offset>" of the handle word
    u32 index{};        ///< array element
    bool image{};       ///< storage image (else sampled)
    bool resolved{};    ///< a texture was found behind the handle
    u32 handle{};       ///< the handle word (path specific ids)
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 levels{};
    u32 layers{};
    u32 format{};       ///< VkFormat of the view
    u32 view_type{};    ///< VkImageViewType
    u32 base_level{};
    u32 view_levels{};
    u32 base_layer{};
    u32 view_layers{};
    std::array<u32, 4> swizzle{}; ///< VkComponentSwizzle r,g,b,a (IDENTITY written as R..A)
    std::optional<Sampler> sampler;
};

/// Bytes of a constant buffer that are the path's own and not the game's (excluded from the diff).
struct PathSpecific {
    u32 cbuf{};
    u32 offset{};
    u32 size{};
    std::string why;
};

struct Draw {
    bool indexed{};
    bool indirect{};
    u32 count{}; ///< vertices or indices
    u32 instances{};
    u32 first{};
    s32 base_vertex{};
    u32 base_instance{};
    u32 topology{};
};

struct Capture {
    Path path{};
    u64 present{};
    u32 ordinal{};
    std::string kind{"dispatch"}; ///< "dispatch" or "draw"
    std::string stage{"cs"};
    u64 hash{};
    u64 code_key{};
    std::array<u32, 3> groups{}; ///< direct dispatch group counts
    bool indirect{};
    std::array<u32, 3> block{};
    std::optional<Draw> draw;
    bool raw_global{};
    std::vector<Range> ranges;
    std::vector<Texture> textures;
    std::vector<PathSpecific> path_specific;
    std::vector<std::string> notes;
};

/// A pointer found in a constant buffer.
struct Pointer {
    u32 offset{}; ///< byte offset of the u64 in the buffer
    u64 value{};
};
/// The 8-byte aligned non-zero u64 of `bytes` that `resolves` accepts, in offset order, at most
/// `limit`.
[[nodiscard]] std::vector<Pointer> FindPointers(std::span<const u8> bytes,
                                                const std::function<bool(u64)>& resolves,
                                                u32 limit);

/// "c<cbuf>" / "c<cbuf>+0x<offset>".
[[nodiscard]] std::string Source(u32 cbuf);
[[nodiscard]] std::string Source(u32 cbuf, u32 offset);

/// The capture as JSON; `blob_name` is the blob's file name. The blob is every range's `before`
/// then `after` in range order (offsets in the JSON).
[[nodiscard]] std::string ToJson(const Capture& capture, std::string_view blob_name);
[[nodiscard]] std::vector<u8> Blob(const Capture& capture);

/// Writes `<seq>-<code key>-p<present>-o<ordinal>.json` and `.bin` into the path's directory
/// (created on the first write) and logs one line. Thread safe. Returns the JSON path, or an empty
/// path (with the error logged) when either file could not be written.
std::filesystem::path Write(const Capture& capture);

/// The directory of a path's captures (empty until the first Write).
[[nodiscard]] std::filesystem::path Directory(Path path);

} // namespace VideoCommon::PathDiff
