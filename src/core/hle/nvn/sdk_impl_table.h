// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Core::NvnHle {

/// Little-endian words of a module image (module-relative offsets). The caller bounds-checks.
[[nodiscard]] u32 SdkImageWord(std::span<const u8> image, u64 offset);
[[nodiscard]] u64 SdkImageU64(std::span<const u8> image, u64 offset);

/// The NVN2 SDK's GetProcAddress implementation table: the u64 array of implementation offsets
/// (module-relative) that its sorted proc-name table (`proc_names`, FindSdkProcNames, with the
/// "nvn" prefix) indexes. The SDK copies it into its runtime dispatch table at device creation
/// (OoT sdk+0xeb6528, DK 3.2.0 sdk+0xe056e0, DK 1.0.0 sdk+0xd95168).
///
/// Found by shape: the tables whose nvnDlssNetworkBuilder Get/SetDevice and Get/SetFlags entries
/// are the SDK's one-instruction accessors (`ldr x0,[x0]` / `str x1,[x0]` / `ldr w0,[x0,#0x20]` /
/// `str w1,[x0,#0x20]`, each then `ret`). Returns the offsets of every such table (empty when the
/// names are not sorted or lack those four functions).
[[nodiscard]] std::vector<u64> FindSdkImplTables(std::span<const u8> image,
                                                 std::span<const std::string> proc_names);

/// Module offset of the implementation of `name` when every table of `tables` holds the same
/// in-image function for it; nullopt otherwise (unknown name, tables that disagree, no table).
[[nodiscard]] std::optional<u64> SdkImplOffset(std::span<const u8> image,
                                               std::span<const std::string> proc_names,
                                               std::span<const u64> tables, std::string_view name);

/// Whether the SDK's nvnDeviceGetTextureHandle at `address` answers the null handle for texture
/// id 0. Ocarina of Time's SDK (API 231.40, sdk+0x651c60) is
/// `sxtw x8,w1; sxtw x9,w2; cmp w1,#0; orr x8,x8,x9,lsl #20; csel x0,xzr,x8,eq; ret`; Donkey Kong
/// Bananza's (231.16 and 231.30) has no compare: `(long)texture | (long)sampler << 20` whatever the
/// texture id. True when the function (up to its `ret`, at most 8 instructions) compares w1 with 0
/// and selects xzr into x0 on EQ.
[[nodiscard]] bool SdkTextureHandleNullForTextureZero(std::span<const u8> image, u64 address);

} // namespace Core::NvnHle
