// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

// The bindless texture table of VTX-HLE: what a shader needs when it samples a texture whose
// handle it reads at run time (a handle stored in a buffer the game fills, indexed per pixel or
// per light), where no per-draw handle program can name the texture. The replay keeps one
// descriptor set (set BINDLESS_SET) holding every texture the game registered in its texture
// pool, one array per shader image type, and every sampler of the sampler pool. A handle is the
// game's: texture id in bits 0..19 and sampler id from bit 20, rebased into the replay's global
// id space (RebaseHandleWord) with the pool bases the host puts in the driver bank (c0).

namespace Shader {

/// Descriptor set of the tables (set 0 holds the draw's own descriptors).
constexpr u32 BINDLESS_SET = 1;

/// Entries of every texture array / of the sampler array. Ids past the capacity read the last
/// entry, which stays a stand-in. The sampler id is 12 bits wide.
constexpr u32 BINDLESS_TEXTURE_CAPACITY = 1u << 15;
constexpr u32 BINDLESS_SAMPLER_CAPACITY = 1u << 12;

/// Binding of the sampler array.
constexpr u32 BINDLESS_SAMPLER_BINDING = 0;

/// Bindings of the texture arrays: one per shader image type and sampled class (float, integer,
/// depth: a depth-compare sampler needs a depth-typed image).
constexpr u32 BINDLESS_TYPE_COUNT = 8; ///< Color1D .. ColorArrayCube, Color2DRect
constexpr u32 BINDLESS_CLASS_COUNT = 3;
constexpr u32 BINDLESS_BINDING_COUNT = 1 + BINDLESS_CLASS_COUNT * BINDLESS_TYPE_COUNT;

[[nodiscard]] constexpr bool BindlessSupportsType(TextureType type) noexcept {
    return static_cast<u32>(type) < BINDLESS_TYPE_COUNT;
}

/// Class 0 float, 1 integer, 2 depth.
[[nodiscard]] constexpr u32 BindlessClass(bool is_integer, bool is_depth) noexcept {
    return is_depth ? 2u : (is_integer ? 1u : 0u);
}

[[nodiscard]] constexpr u32 BindlessTextureBinding(TextureType type, u32 klass) noexcept {
    return 1 + static_cast<u32>(type) + klass * BINDLESS_TYPE_COUNT;
}

/// Where the host puts the bases of the bound pools in the driver bank (c0), u32 each. Past
/// every offset the guest's shaders and the stage tables use.
constexpr u32 BINDLESS_DRIVER_TEXTURE_BASE = 0xf00;
constexpr u32 BINDLESS_DRIVER_SAMPLER_BASE = 0xf04;

} // namespace Shader
