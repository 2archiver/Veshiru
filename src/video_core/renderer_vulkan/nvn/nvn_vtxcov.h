// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Vertex and index coverage (DESIGN.md "Vertex and index coverage"): why each vertex stream and
// index buffer a draw binds is, or is not, read from device-local memory
// (EDEN_NX2_NVN_VTXCOV_DIAG), and the candidate switches that serve more of them from pool mirrors
// (EDEN_NX2_TRY_VTXCOV_*). Tests: src/tests/nvn/pool_mirror.cpp, compute_vulkan.cpp.

#pragma once

#include <array>
#include <string>
#include <string_view>

#include "common/common_types.h"

namespace Vulkan::Nvn {

/// Where a bound vertex stream or index buffer was read from, or why it stayed in host memory.
enum class VtxCov : u8 {
    // Device-local.
    Staged,     ///< a vertex staging copy (shadow chunk) of the pass
    Mirror,     ///< the pool's VRAM mirror, through the pass's staging scan (TRY_VTX_MIRROR)
    MirrorBind, ///< the pool's VRAM mirror, current at the bind (TRY_VTXCOV_INPASS/_SHADERWRITES)
    Device,     ///< the pool is device-local itself (Backing::Mirror, CPU_NO_ACCESS)
    // Imported host memory.
    StageOff,          ///< EDEN_NX2_NVN_VERTEX_STAGE=0
    ShaderWrites,      ///< a draw of the pass wrote through shaders: staged copies are not used
    TooLarge,          ///< over VertexStageLimits::max_range_bytes: the scan leaves it out
    TotalCap,          ///< the scan reached max_total_bytes before it
    NotScannedTargets, ///< bound past the SetRenderTargets that ended the scan (the pass went on)
    NotScannedCall,    ///< bound past a CallList that ended the scan (records of the called list)
    NotScannedEnd,     ///< bound past the end of the list the scan read (the pass went on)
    NotScanned,        ///< not in the pass's scan for another reason
    IndexIndirect,     ///< the index buffer of an indirect draw (its count is on the GPU)
    Pending,           ///< the pool was still being backed at the scan
    Shadowed,          ///< under a live storage shadow at the scan
    FewDraws,          ///< a pass of fewer than min_draws draws, no pool mirrors (TRY_VTX_MIRROR off)
    NoMirrorSize,      ///< pool over TRY_STORAGE_VRAM_MAX_MIB (read-only mirrors keep that limit)
    NoMirrorBudget,    ///< read-only mirrors hold half of POOL_MIRROR_MAX_MIB already
    NoMirrorAlign,     ///< pool address not page aligned
    NoMirrorRefused,   ///< the pool was refused a mirror for good (budget, tracker, allocation)
    NoDeviceMemory,    ///< no shadow chunk for its staging copy
    CopyBound,         ///< the pass's mirror copy-ins reached their bound before it
    MirrorStale,       ///< bind-time serving: the mirror's pages are not current (prefetched next)
    RawWrites,         ///< bind-time serving: this draw stores through raw pointers
    Dropped,           ///< staged by the scan, the copy was dropped since (a pool write)
    SizeShort,         ///< staged with fewer bytes than this bind
    Other,
    Count,
};

inline constexpr size_t VTXCOV_REASONS = static_cast<size_t>(VtxCov::Count);

inline constexpr std::array<std::string_view, VTXCOV_REASONS> VTXCOV_NAMES{
    "staged",          "mirror",         "mirror_bind",      "device",
    "stage_off",       "shader_writes",  "too_large",        "total_cap",
    "not_scanned_rt",  "not_scanned_call", "not_scanned_end", "not_scanned",
    "index_indirect",  "pending",        "shadowed",         "few_draws",
    "no_mirror_size",  "no_mirror_budget", "no_mirror_align", "no_mirror_refused",
    "no_device_memory", "copy_bound",    "mirror_stale",     "raw_writes",
    "dropped",         "size_short",     "other",
};
static_assert(VTXCOV_NAMES.back() == "other");

[[nodiscard]] constexpr bool VtxCovServed(VtxCov reason) noexcept {
    return reason <= VtxCov::Device;
}

/// EDEN_NX2_TRY_VTXCOV_<NAME>=1 (candidates, off by default); EDEN_NX2_TRY_VTXCOV=1 sets them all.
enum VtxCovSwitch : u32 {
    /// A vertex/index range that no staging copy serves is served at the bind from its pool's
    /// mirror when the mirror is current there; otherwise its pages join the pool's prefetch set
    /// and are copied in at the next pass begin. Ranges first bound inside an open pass, indirect
    /// index buffers and ranges past the copy bound are covered from the next frame on.
    VTXCOV_INPASS = 1u << 0,
    /// The same bind-time serving after a draw of the pass wrote through shaders.
    VTXCOV_SHADER_WRITES = 1u << 1,
    /// The stale pages a pass copies into mirrors for its scan stop at VTXCOV_COPY_BYTES.
    VTXCOV_BOUND = 1u << 2,
    /// Read-only mirrors for pools over TRY_STORAGE_VRAM_MAX_MIB; those never become storage ones.
    VTXCOV_BIGPOOL = 1u << 3,
    VTXCOV_ALL = VTXCOV_INPASS | VTXCOV_SHADER_WRITES | VTXCOV_BOUND | VTXCOV_BIGPOOL,
};

/// VTXCOV_BOUND: the per-pass bound of the mirror copy-ins (VTX_MIRROR_COPY_BYTES otherwise).
inline constexpr u64 VTXCOV_COPY_BYTES = u64{32} << 20;

/// VTXCOV_INPASS / _SHADER_WRITES serve ranges up to this size: an indirect draw's index range
/// runs to the end of its pool (its count is on the GPU), which would put whole pools in the
/// prefetch set.
inline constexpr u64 VTXCOV_BIND_MAX_BYTES = u64{16} << 20;

/// "inpass,bound" for a switch mask; "none" for 0.
[[nodiscard]] std::string VtxCovSwitchNames(u32 switches);

/// Binds and bound bytes by kind (0 vertex stream, 1 index buffer) and reason.
struct VtxCovCounts {
    std::array<std::array<u64, VTXCOV_REASONS>, 2> binds{};
    std::array<std::array<u64, VTXCOV_REASONS>, 2> bytes{};
};

/// The body of the `[nvn] vtxcov:` line: `counts` over `presents` presents, per present, as
/// vertex/index pairs; reasons with no bind are left out. `copy_in_bytes`: mirror copy-ins,
/// `staged_bytes`: vertex staging copies over the same presents.
[[nodiscard]] std::string FormatVtxCov(const VtxCovCounts& counts, u64 presents, u64 copy_in_bytes,
                                       u64 staged_bytes, u32 switches);

} // namespace Vulkan::Nvn
