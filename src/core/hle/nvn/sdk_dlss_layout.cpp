// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <string_view>

#include "core/hle/nvn/sdk_dlss_layout.h"
#include "core/hle/nvn/sdk_impl_table.h"

namespace Core::NvnHle {
namespace {

// The NVNdlssNetwork layouts of the NVN2 SDKs, as each one's nvnDlssNetworkInitialize writes them
// (and its Resize, Finalize and nvnCommandBufferEvaluateDlss read them):
//   0x258: Ocarina of Time's (API 231.40, sdk+0x659b40) and Star Fox's (231.39, sdk+0x661340);
//   0x220: Donkey Kong Bananza 3.2.0's (231.30, sdk+0x5dcd90). switch2-recomp (drakolordx7,
//          runtime/src/gfx/nvn/api_dlss.cpp) documents the same layout for Pokopia's SDK 20.5.25;
//   0x218: Donkey Kong Bananza 1.0.0's (231.16, sdk+0x5b8bd0): no initial output size (its SDK
//          has no nvnDlssNetworkResize), the flags word right after the current output.
// The newer layout's extra 0x38 bytes are backend state (a 0x18-byte block before the storage
// fields, 0x20 bytes before the evaluation entries) the HLE does not keep.
constexpr std::array<DlssNetworkLayout, 3> Layouts{{
    {
        .sdk = "API 231.39/231.40 (Ocarina of Time, Star Fox)",
        .size = 0x258,
        .storage_device = 0xc0,
        .storage = 0xc8,
        .storage_size = 0xd0,
        .max_evaluation_entries = 0x200,
        .method = 0x20c,
        .ngx_flags = 0x210,
        .quality = 0x214,
        .preset = 0x218,
        .optimal = 0x21c,
        .output = 0x224,
        .output_subrects = 0x22c,
        .max_render = 0x230,
        .min_render = 0x238,
        .current_output = 0x240,
        .initial_output = 0x248,
        .flags = 0x250,
    },
    {
        .sdk = "API 231.30 (SDK 20.5)",
        .size = 0x220,
        .storage_device = 0xa8,
        .storage = 0xb0,
        .storage_size = 0xb8,
        .max_evaluation_entries = 0x1c8,
        .method = 0x1d4,
        .ngx_flags = 0x1d8,
        .quality = 0x1dc,
        .preset = 0x1e0,
        .optimal = 0x1e4,
        .output = 0x1ec,
        .output_subrects = 0x1f4,
        .max_render = 0x1f8,
        .min_render = 0x200,
        .current_output = 0x208,
        .initial_output = 0x210,
        .flags = 0x218,
    },
    {
        .sdk = "API 231.16",
        .size = 0x218,
        .storage_device = 0xa8,
        .storage = 0xb0,
        .storage_size = 0xb8,
        .max_evaluation_entries = 0x1c8,
        .method = 0x1d4,
        .ngx_flags = 0x1d8,
        .quality = 0x1dc,
        .preset = 0x1e0,
        .optimal = 0x1e4,
        .output = 0x1ec,
        .output_subrects = 0x1f4,
        .max_render = 0x1f8,
        .min_render = 0x200,
        .current_output = 0x208,
        .initial_output = 0,
        .flags = 0x210,
    },
}};

/// Every field lies inside its network and past the shared fields (handle, texture IDs).
constexpr bool FieldsInside(const DlssNetworkLayout& l) {
    if (l.size > MaxDlssNetworkSize || DlssNetworkTextureIds + DlssTextureIdBytes > l.size) {
        return false;
    }
    const auto inside = [&](u32 offset, u32 bytes) {
        return offset >= DlssNetworkHandle + 8 && offset + bytes <= l.size;
    };
    return inside(l.storage_device, 8) && inside(l.storage, 8) && inside(l.storage_size, 8) &&
           inside(l.max_evaluation_entries, 4) && inside(l.method, 4) && inside(l.ngx_flags, 4) &&
           inside(l.quality, 4) && inside(l.preset, 4) && inside(l.optimal, 8) &&
           inside(l.output, 8) && inside(l.output_subrects, 4) && inside(l.max_render, 8) &&
           inside(l.min_render, 8) && inside(l.current_output, 8) &&
           (l.initial_output == 0 || inside(l.initial_output, 8)) && inside(l.flags, 4);
}
static_assert(std::ranges::all_of(Layouts, FieldsInside));

constexpr u32 Ret = 0xd65f03c0;

s64 SignExtend(u32 value, u32 bits) {
    const u32 shift = 32 - bits;
    return static_cast<s32>(value << shift) >> shift;
}

struct Store {
    s64 offset;
    u32 bytes;
};

/// A store through x0 at an immediate offset (STR unsigned offset, STUR, STP signed offset, of
/// general or SIMD&FP registers), or nullopt for any other instruction.
std::optional<Store> StoreThroughX0(u32 w) {
    if (((w >> 5) & 31) != 0) {
        return std::nullopt;
    }
    const bool simd = ((w >> 26) & 1) != 0;
    const u32 size = w >> 30;
    // Bytes of one register of a single-register store (opc 0 B/H/S/D/W/X, SIMD opc 2 size 0 Q).
    const auto single_bytes = [&]() -> u32 {
        const u32 opc = (w >> 22) & 3;
        if (opc == 0) {
            return 1u << size;
        }
        return simd && opc == 2 && size == 0 ? 16u : 0u;
    };
    if ((w & 0x3b000000) == 0x39000000) { // STR (immediate, unsigned offset)
        const u32 bytes = single_bytes();
        if (bytes == 0) {
            return std::nullopt;
        }
        return Store{static_cast<s64>((w >> 10) & 0xfff) * bytes, bytes};
    }
    if ((w & 0x3b200c00) == 0x38000000) { // STUR
        const u32 bytes = single_bytes();
        if (bytes == 0) {
            return std::nullopt;
        }
        return Store{SignExtend((w >> 12) & 0x1ff, 9), bytes};
    }
    if ((w & 0x3bc00000) == 0x29000000) { // STP (signed offset)
        u32 bytes = 0;
        if (simd) {
            bytes = size == 0 ? 4u : size == 1 ? 8u : size == 2 ? 16u : 0u;
        } else {
            bytes = size == 0 ? 4u : size == 2 ? 8u : 0u;
        }
        if (bytes == 0) {
            return std::nullopt;
        }
        return Store{SignExtend((w >> 15) & 0x7f, 7) * bytes, 2 * bytes};
    }
    return std::nullopt;
}

bool IsMoveWide(u32 w) { // MOVZ / MOVN / MOVK
    return (w & 0x1f800000) == 0x12800000;
}

bool IsSimdModifiedImmediate(u32 w) { // MOVI / MVNI / FMOV (vector, immediate)
    return (w & 0x9ff80400) == 0x0f000400;
}

/// One past the highest byte a SetDefaults stores through its object pointer: a leaf of
/// immediate moves and stores through x0 that ends in `ret` (the shape every NVN2 SDK compiles the
/// DLSS SetDefaults to). nullopt for anything else.
std::optional<u32> SetDefaultsExtent(std::span<const u8> image, u64 address) {
    constexpr u32 MaxInstructions = 32;
    if (address % 4 != 0) {
        return std::nullopt;
    }
    s64 end = 0;
    for (u32 i = 0; i < MaxInstructions && address + 4 * (i + 1) <= image.size(); ++i) {
        const u32 w = SdkImageWord(image, address + 4 * i);
        if (w == Ret) {
            return end > 0 ? std::optional<u32>{static_cast<u32>(end)} : std::nullopt;
        }
        if (const std::optional<Store> store = StoreThroughX0(w)) {
            if (store->offset < 0) {
                return std::nullopt;
            }
            end = std::max(end, store->offset + store->bytes);
            continue;
        }
        if (IsSimdModifiedImmediate(w) || (IsMoveWide(w) && (w & 31) != 0)) {
            continue;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

/// The size nvnDlssNetworkInitialize clears its network with: the `mov w2, #size` and `mov w1,
/// wzr` before its first call (memset(network, 0, size)), with x0 untouched until then.
std::optional<u32> InitializeClearSize(std::span<const u8> image, u64 address) {
    constexpr u32 MaxInstructions = 24;
    constexpr u32 MovW1Wzr = 0x2a1f03e1;
    if (address % 4 != 0) {
        return std::nullopt;
    }
    std::optional<u32> size;
    bool w1_zero = false;
    for (u32 i = 0; i < MaxInstructions && address + 4 * (i + 1) <= image.size(); ++i) {
        const u32 w = SdkImageWord(image, address + 4 * i);
        if ((w & 0xfc000000) == 0x94000000) { // BL
            return w1_zero ? size : std::nullopt;
        }
        const u32 rd = w & 31;
        if ((w & 0xffe0001f) == 0x52800002) { // MOVZ w2, #imm16
            size = (w >> 5) & 0xffff;
            continue;
        }
        if (w == MovW1Wzr) {
            w1_zero = true;
            continue;
        }
        // x0 (the network) and the two arguments must reach the call as set.
        const bool writes_low = IsMoveWide(w) || (w & 0x1f000000) == 0x11000000 || // ADD/SUB imm
                                (w & 0x1f000000) == 0x0a000000 ||                  // logical reg
                                (w & 0x1f800000) == 0x12000000;                    // logical imm
        if (writes_low && (rd == 0 || rd == 1 || rd == 2)) {
            return std::nullopt;
        }
        if ((w & 0xfc000000) == 0x14000000 || (w & 0xfe1ffc1f) == 0xd61f0000 /* BR/BLR/RET */) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

} // namespace

std::span<const DlssNetworkLayout> KnownDlssNetworkLayouts() {
    return Layouts;
}

const DlssNetworkLayout* FindDlssNetworkLayout(u32 size) {
    const auto it = std::ranges::find(Layouts, size, &DlssNetworkLayout::size);
    return it == Layouts.end() ? nullptr : &*it;
}

DlssLayout NewestDlssLayout() {
    return DlssLayout{.builder_size = 0xc0, .parameters_size = 0xc0, .network = &Layouts[0]};
}

DlssLayout DlssLayoutFor(const SdkDlssSizes& sizes) {
    return DlssLayout{.builder_size = sizes.builder,
                      .parameters_size = sizes.parameters,
                      .network = FindDlssNetworkLayout(sizes.network)};
}

std::optional<SdkDlssSizes> ReadSdkDlssSizes(std::span<const u8> image,
                                             std::span<const std::string> proc_names,
                                             std::span<const u64> impl_tables) {
    const auto entry = [&](std::string_view name) {
        return SdkImplOffset(image, proc_names, impl_tables, name);
    };
    const std::optional<u64> builder_defaults = entry("nvnDlssNetworkBuilderSetDefaults");
    const std::optional<u64> initialize = entry("nvnDlssNetworkInitialize");
    const std::optional<u64> parameters_defaults = entry("nvnDlssParametersSetDefaults");
    if (!builder_defaults || !initialize || !parameters_defaults) {
        return std::nullopt;
    }
    const std::optional<u32> builder = SetDefaultsExtent(image, *builder_defaults);
    const std::optional<u32> network = InitializeClearSize(image, *initialize);
    const std::optional<u32> parameters = SetDefaultsExtent(image, *parameters_defaults);
    if (!builder || !network || !parameters) {
        return std::nullopt;
    }
    return SdkDlssSizes{.builder = *builder,
                        .network = *network,
                        .parameters = *parameters,
                        .builder_set_defaults = *builder_defaults,
                        .network_initialize = *initialize,
                        .parameters_set_defaults = *parameters_defaults};
}

} // namespace Core::NvnHle
