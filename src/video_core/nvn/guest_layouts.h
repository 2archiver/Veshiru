// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include "common/common_types.h"

namespace VideoCommon::Nvn {

// In-place guest layouts the recorder reads (written by the HLE function bodies). They follow
// nxnative's `nvn_obj.h` verbatim. Bit-fields are least-significant-bit first (MSVC and clang
// agree on that).

/// NVNtextureView in nxnative `view_t` layout (16 bytes):
///   byte 0: set_levels:1, set_layers:1, set_format:1, set_target:1, set_swizzle:1, set_dsmode:1,
///           dsmode:2
///   byte 1 target, byte 2 base_level, byte 3 nlevels, u16 format, u16 min_layer, u16 nlayers,
///   u8 swizzle[4], u16 reserved.
struct GuestTextureView {
    u8 set_bits;
    u8 target;
    u8 base_level;
    u8 nlevels;
    u16 format;
    u16 min_layer;
    u16 nlayers;
    u8 swizzle[4];
    u16 reserved;
};
static_assert(sizeof(GuestTextureView) == 16);

inline constexpr u8 GUEST_VIEW_SET_LEVELS = 1 << 0;
inline constexpr u8 GUEST_VIEW_SET_LAYERS = 1 << 1;
inline constexpr u8 GUEST_VIEW_SET_FORMAT = 1 << 2;
inline constexpr u8 GUEST_VIEW_SET_TARGET = 1 << 3;
inline constexpr u8 GUEST_VIEW_SET_SWIZZLE = 1 << 4;
inline constexpr u8 GUEST_VIEW_SET_DSMODE = 1 << 5;

/// NVNbuffer: the tag is at +0x00; the game's ring allocators store the buffer's offset in its
/// memory pool and its size directly, so the host reads them at every use.
inline constexpr u64 GUEST_BUFFER_OFFSET_OFFSET = 0x18;
inline constexpr u64 GUEST_BUFFER_SIZE_OFFSET = 0x20;

/// NVNshaderData: {NVNbufferAddress data; const void* control;}.
struct GuestShaderData {
    u64 data;
    u64 control;
};
static_assert(sizeof(GuestShaderData) == 16);

/// NVN2 shader control section (version 0xc9): magic at +0, version at +4, stage at +0x56 (u16,
/// [UNVERIFIED] for graphics stages), code area length at +0x44.
inline constexpr u32 NVN2_CONTROL_MAGIC = 0x98761234;
inline constexpr u32 NVN2_CONTROL_VERSION = 0xC9;

} // namespace VideoCommon::Nvn
