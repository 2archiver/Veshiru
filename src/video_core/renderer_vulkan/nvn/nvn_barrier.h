// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// What an nvnCommandBufferBarrier orders (tests: src/tests/nvn/pass_state.cpp).
//
// The NVN driver in OoT's SDK records these 3D-class methods per bit (names from NVIDIA
// open-gpu-doc clc797.h / clc56f.h; nvn_enums.inc NVNbarrierBits):
//  - ORDER_PRIMITIVES (0x1): WAIT_FOR_IDLE. Every later command waits for every earlier one.
//  - ORDER_INDIRECT_DATA (0x8, and the private bit 0x400 on the same path): NO_OPERATION and the
//    end of the GPFIFO segment, so the front end fetches the later commands (and the indirect
//    arguments they push) only after the earlier ones. [INFERENCE: the segment end waits for the
//    engine.] Ordered like ORDER_PRIMITIVES.
//  - ORDER_FRAGMENTS (0x2): PIXEL_SHADER_BARRIER (BLOCK_BEFORE_PS_AND_ZTEST); ORDER_FRAGMENTS_TILED
//    (0x4): the tiled cache barrier. Later pixel work waits for earlier pixel work; vertex,
//    primitive and compute work does not wait.
//  - INVALIDATE_TEXTURE (0x10): INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI; INVALIDATE_SHADER (0x20):
//    INVALIDATE_SHADER_CACHES_NO_WFI (instruction, global data, constant); INVALIDATE_TEXTURE_
//    DESCRIPTOR (0x40) and INVALIDATE_ZCULL (0x80): their caches. None of these waits.
//
// In Vulkan terms only 0x1 / 0x8 / 0x400 are an execution dependency from ALL_COMMANDS to
// ALL_COMMANDS, and no narrower stage mask is correct for them. The others order pixel work
// only: the replay ends every render pass behind a pending full barrier and orders every
// non-dispatch consumer after earlier dispatches, so outside a pass they need no command of
// their own; inside a pass they are handled by the draw path (nvn_draw.cpp, Op::Barrier).

#pragma once

#include "common/common_types.h"

namespace Vulkan::Nvn {

inline constexpr u32 NVN_BARRIER_ORDER_PRIMITIVES = 0x1;
inline constexpr u32 NVN_BARRIER_ORDER_FRAGMENTS = 0x2;
inline constexpr u32 NVN_BARRIER_ORDER_FRAGMENTS_TILED = 0x4;
inline constexpr u32 NVN_BARRIER_ORDER_INDIRECT_DATA = 0x8;
/// Not in the public enum; the driver handles it together with ORDER_INDIRECT_DATA.
inline constexpr u32 NVN_BARRIER_PRIVATE_INDIRECT = 0x400;

/// The barrier makes every later command wait for every earlier one (see above).
[[nodiscard]] constexpr bool NvnBarrierOrdersAllWork(u32 bits) {
    return (bits & (NVN_BARRIER_ORDER_PRIMITIVES | NVN_BARRIER_ORDER_INDIRECT_DATA |
                    NVN_BARRIER_PRIVATE_INDIRECT)) != 0;
}

/// The barrier makes some later work wait for earlier work: all of it, or pixel work.
[[nodiscard]] constexpr bool NvnBarrierOrders(u32 bits) {
    return NvnBarrierOrdersAllWork(bits) ||
           (bits & (NVN_BARRIER_ORDER_FRAGMENTS | NVN_BARRIER_ORDER_FRAGMENTS_TILED)) != 0;
}

} // namespace Vulkan::Nvn
