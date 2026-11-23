// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The per-shader driver bank (c0) a compute shader reads on NVN2, built from the bindings of a
// dispatch. The layout comes from the program's control section (ComputeDriverLayout, measured on
// the DK and OoT shaders): c0+0x00 holds the declared bank size in bytes and c0+0x04..0x0c the
// group counts of the dispatch (measured on the legacy path's cbuf 0 for DK dispatches), then push
// constants from c0+0x10, then, from the next 0x10 boundary, the uniform-buffer entries, the
// storage-buffer entries and the texture and image handle tables. Buffer entries are {u64 address,
// u32 size, u32 pad}; the shader reads them with plain constant-buffer loads and uses the addresses
// for global memory accesses, which the recompiler maps to storage buffers (see
// Shader::NvnBufferWindow).
//
// [UNVERIFIED] against NVN2 hardware: entry k of the uniform table is the buffer bound with
// BindUniformBuffer(compute, binding k), entry k of the storage table the one bound with
// BindStorageBuffer(compute, binding k); the push-constant block starts at c0+0x10 with the
// game's offsets unchanged.

#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace VideoCommon::Nvn {

struct ComputeBankInputs {
    /// Buffers by binding slot. A slot beyond the span, or with size 0, is unbound (zero entry).
    std::span<const BufferRange> uniform_buffers;
    std::span<const BufferRange> storage_buffers;
    /// The push-constant block (offset 0 = c0+push_constant_offset). Bytes beyond the layout's
    /// push_constant_words * 4 are not part of the bank.
    std::span<const u8> push_constants;
    /// Handles by slot (texture id | sampler id << 20; images: texture id). Only the low 32 bits
    /// of each 8-byte table entry are written.
    std::span<const u64> textures;
    std::span<const u64> images;
    /// Group counts of a direct dispatch; zero for an indirect one (the GPU reads those, the CPU
    /// does not know them).
    std::array<u32, 3> group_counts{};
};

/// Bytes of the bank: the derived end of the tables (at least the push-constant start).
[[nodiscard]] u32 ComputeBankSize(const ComputeDriverLayout& layout) noexcept;

/// Fills `out` (zeroed first) with the bank. Tables that do not fit in `out` are cut off; `out`
/// should be at least ComputeBankSize(layout) bytes.
void FillComputeDriverBank(const ComputeDriverLayout& layout, const ComputeBankInputs& inputs,
                           std::span<u8> out);

/// The bank as a vector of ComputeBankSize(layout) bytes.
[[nodiscard]] std::vector<u8> BuildComputeDriverBank(const ComputeDriverLayout& layout,
                                                     const ComputeBankInputs& inputs);

} // namespace VideoCommon::Nvn
