// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The texture handle a shader reads from a table in memory (a handle program of
// Shader::Info::handle_programs), evaluated for one draw of the VTX-HLE replay.
//
// The emulated path evaluates the same programs per draw (video_core/shader_handle_program.h,
// with the Maxwell3D constant buffer addresses); here the words come from the uniform buffers the
// game bound, the storage buffers it bound through the driver table, and guest memory.

#pragma once

#include <algorithm>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"

namespace VideoCommon::Nvn {

/// {address, size} of a bound storage buffer.
struct HandleProgramBuffer {
    u64 address{};
    u64 size{};
};

/// The handle word the program computes (before the pool rebase of `RebaseHandleWord`), or
/// nothing for a malformed program.
///
/// read_uniform(bank, offset): the word of the constant buffer c`bank` (0 when it cannot be read).
/// storage(index): the storage buffer `index` of Shader::Info::storage_buffers_descriptors of the
/// stage (an unbound buffer: address 0).
/// read_memory(address): the guest word at `address` (nothing when unreadable).
///
/// A storage offset of a program is relative to the buffer address aligned down to
/// `storage_alignment` (the host storage buffer offset alignment the shader was translated with:
/// StorageOffset of the global memory to storage buffer pass), and the shader's own load is
/// robust: a word that reaches past the buffer's size reads 0.
template <typename ReadUniform, typename Storage, typename ReadMemory>
[[nodiscard]] std::optional<u32> EvaluateHandleProgram(std::span<const u32> code,
                                                       u64 storage_alignment,
                                                       ReadUniform&& read_uniform, Storage&& storage,
                                                       ReadMemory&& read_memory) {
    const u64 alignment = (std::max)(storage_alignment, u64{1});
    return Shader::EvaluateHandleProgram(
        code,
        [&](u32 bank, u32 offset) -> u32 { return read_uniform(bank, offset); },
        [&](u32 index, u32 offset) -> u32 {
            const HandleProgramBuffer buffer = storage(index);
            if (buffer.address == 0) {
                return 0;
            }
            const u64 base = buffer.address & ~(alignment - 1);
            const u64 end = buffer.address - base + buffer.size;
            if (u64{offset} + sizeof(u32) > end) {
                return 0;
            }
            return read_memory(base + offset).value_or(0);
        });
}

} // namespace VideoCommon::Nvn
