// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string>

#include "common/common_types.h"

namespace Core::NvnHle {

/// Where an NVNdlssNetwork keeps its fields, as one NVN2 SDK generation's nvnDlssNetworkInitialize
/// lays them out (the network is plain data in the game's memory; the SDK clears `size` bytes of it
/// first). The fields every generation keeps at the same place are the Network* constants below.
struct DlssNetworkLayout {
    const char* sdk;            ///< the SDKs the layout was read from (log)
    u32 size;                   ///< bytes Initialize clears before anything else
    u32 storage_device;         ///< u64: the device again
    u32 storage;                ///< u64
    u32 storage_size;           ///< u64
    u32 max_evaluation_entries; ///< s32
    u32 method;                 ///< u32: what Resize recomputes the render sizes for
    u32 ngx_flags;              ///< u32: the backend's create flags
    u32 quality;                ///< u32: the method again, the backend's quality argument
    u32 preset;                 ///< u32
    u32 optimal;                ///< s32[2]
    u32 output;                 ///< s32[2]
    u32 output_subrects;        ///< u32: DLSS_FLAG_OUTPUT_SUBRECTS as 0/1
    u32 max_render;             ///< s32[2]
    u32 min_render;             ///< s32[2]
    u32 current_output;         ///< s32[2]: what EvaluateDlss upscales to
    u32 initial_output;         ///< s32[2]: Resize's upper bound; 0 = none (no Resize in that SDK)
    u32 flags;                  ///< u32: the builder's flags word
};

/// Offsets every known NVNdlssNetwork layout shares.
inline constexpr u32 DlssNetworkDevice = 0x08;      ///< u64
inline constexpr u32 DlssNetworkTexturePool = 0x10; ///< u64
inline constexpr u32 DlssNetworkTextureIds = 0x18;  ///< 0x80 bytes, the builder's
inline constexpr u32 DlssNetworkHandle = 0xa0;      ///< u64: the backend feature, nonzero = made
inline constexpr u32 DlssTextureIdBytes = 0x80;
/// The largest known network (the bytes a host copy of one needs).
inline constexpr u32 MaxDlssNetworkSize = 0x258;

/// The known layouts, newest SDK first.
[[nodiscard]] std::span<const DlssNetworkLayout> KnownDlssNetworkLayouts();
/// The known layout of a network of `size` bytes; null when no known SDK has one of that size.
[[nodiscard]] const DlssNetworkLayout* FindDlssNetworkLayout(u32 size);

/// The DLSS object layout the HLE bodies follow: the sizes the SDK's SetDefaults clear and the
/// network layout its Initialize writes.
struct DlssLayout {
    u32 builder_size;                 ///< NVNdlssNetworkBuilder bytes (SetDefaults keeps +0)
    u32 parameters_size;              ///< NVNdlssParameters bytes
    const DlssNetworkLayout* network; ///< null: the SDK's network is of no known layout
};
/// The newest known SDK's layout (Ocarina of Time's and Star Fox's: 0xc0 / 0x258 / 0xc0), which
/// the HLE follows when the game's SDK was not read.
[[nodiscard]] DlssLayout NewestDlssLayout();

/// The object sizes the game's own SDK gives the DLSS objects, read from its machine code: the
/// bytes nvnDlssNetworkBuilderSetDefaults and nvnDlssParametersSetDefaults store through their
/// object pointer (one past the highest byte), and the memset size nvnDlssNetworkInitialize clears
/// the network with before anything else.
struct SdkDlssSizes {
    u32 builder{};
    u32 network{};
    u32 parameters{};
    u64 builder_set_defaults{}; ///< module offsets of the functions read (log)
    u64 network_initialize{};
    u64 parameters_set_defaults{};
};

/// Reads SdkDlssSizes from a module image (module-relative addresses = offsets). The functions are
/// the entries of the SDK's GetProcAddress implementation table (`impl_tables`, FindSdkImplTables
/// over `proc_names`, the names in table order with the "nvn" prefix). nullopt when a function is
/// not in the table (or the tables disagree on it) or is not of the expected shape.
[[nodiscard]] std::optional<SdkDlssSizes> ReadSdkDlssSizes(std::span<const u8> image,
                                                           std::span<const std::string> proc_names,
                                                           std::span<const u64> impl_tables);

/// The layout of the SDK whose sizes are `sizes` (its network layout looked up by size).
[[nodiscard]] DlssLayout DlssLayoutFor(const SdkDlssSizes& sizes);

} // namespace Core::NvnHle
