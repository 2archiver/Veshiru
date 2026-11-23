// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <optional>
#include <type_traits>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"
#include "video_core/nvn/record.h"

// DLSS family of NVN2: NVNdlssNetworkBuilder, NVNdlssNetwork and NVNdlssParameters, all kept in
// place in guest memory, and nvnCommandBufferEvaluateDlss. Every layout, argument list and rule
// here is the one of the games' own nnSdk (Ocarina of Time's 0400486024DEC000 update, API 231.40,
// and Star Fox's, API 231.39; the same semantics in both): builder accessors OoT sdk+0x65ab90..
// +0x65ad54, GetOptimalSettings +0x659ae0, GetStorageSize +0x659fc0, Initialize +0x659b40, Finalize
// +0x659f20, Resize +0x6599d0, parameter accessors +0x65ad60..+0x65b010, EvaluateDlss +0x65a4e0.
// Older SDKs lay the objects out differently (Donkey Kong Bananza's 231.30 and 231.16 SDKs: a
// 0xB8-byte builder, 0x220- and 0x218-byte networks with other field offsets, 0xC8-byte
// parameters): the sizes come from the game's own SDK code (DeviceState::GetDlssLayout,
// sdk_dlss_layout.h) and no body writes past the object the game's SDK defines. The builder and
// parameter fields sit at the same offsets in every SDK.
// The DLSS backend itself (the optimal render sizes, the feature and its evaluation) is not in the
// game: the SDK loads a system library named "dlss" at run time and calls its handshake interface
// (OoT sdk+0x6593b0). So the render sizes GetOptimalSettings answers are the host's (a known
// limitation, see there), the backend feature is a handle the replay maps to a Streamline viewport,
// and the evaluation is the replay's (an Op::EvaluateDlss record).
namespace Core::NvnHle::Hle {
namespace {

namespace VC = VideoCommon::Nvn;

/// NVNdlssNetworkBuilder in place, 0xC0 bytes (OoT sdk+0x65ab90..: SetDefaults and the accessors);
/// the 231.30 SDK's has no `reserved` field (0xB8 bytes).
struct Builder {
    u64 device;                 // +0x00, kept by SetDefaults
    u64 texture_pool;           // +0x08
    u64 storage;                // +0x10
    u64 storage_size;           // +0x18
    u32 flags;                  // +0x20 (DLSS_FLAG_*)
    u32 method;                 // +0x24 (NGX perf quality)
    u32 preset;                 // +0x28 (NGX render preset hint)
    s32 width;                  // +0x2c output resolution
    s32 height;                 // +0x30
    s32 max_evaluation_entries; // +0x34
    std::array<u8, 0x80> texture_ids; // +0x38, copied whole (no count argument exists)
    u64 reserved;               // +0xb8, cleared by SetDefaults; no function sets it
};
static_assert(std::is_standard_layout_v<Builder> && std::is_trivially_copyable_v<Builder>);
static_assert(offsetof(Builder, device) == 0x00);
static_assert(offsetof(Builder, texture_pool) == 0x08);
static_assert(offsetof(Builder, storage) == 0x10);
static_assert(offsetof(Builder, storage_size) == 0x18);
static_assert(offsetof(Builder, flags) == 0x20);
static_assert(offsetof(Builder, method) == 0x24);
static_assert(offsetof(Builder, preset) == 0x28);
static_assert(offsetof(Builder, width) == 0x2c);
static_assert(offsetof(Builder, height) == 0x30);
static_assert(offsetof(Builder, max_evaluation_entries) == 0x34);
static_assert(offsetof(Builder, texture_ids) == 0x38);
static_assert(offsetof(Builder, reserved) == 0xb8);
static_assert(sizeof(Builder) == 0xc0);

/// The game's builder, as many bytes as its SDK's builder has (the rest zero): a smaller builder
/// may end right where the game's mapping ends.
Builder LoadBuilder(Call& c, u64 va, const DlssLayout& layout) {
    Builder b{};
    c.mem.Read(va, &b, std::min<size_t>(layout.builder_size, sizeof(Builder)));
    return b;
}

/// One NVNdlssNetwork's bytes in the layout of the game's SDK (DlssNetworkLayout: the game gives
/// 0x800 bytes; the SDK's network is 0x218..0x258 of them). The ranges no field covers are written
/// by the backend or not at all; Initialize leaves them zero.
class Network {
public:
    explicit Network(const DlssNetworkLayout& layout_) : layout{layout_} {}

    /// Reads the network's bytes; false (all zero) when they are not guest memory.
    bool Load(Call& c, u64 va) {
        if (c.mem.Read(va, bytes.data(), layout.size)) {
            return true;
        }
        bytes.fill(0);
        return false;
    }
    bool Store(Call& c, u64 va) const {
        return c.mem.Write(va, bytes.data(), layout.size);
    }

    template <typename T>
    [[nodiscard]] T Get(u32 offset) const {
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    }
    template <typename T>
    void Put(u32 offset, const T& value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }
    void PutPair(u32 offset, s32 x, s32 y) {
        Put(offset, x);
        Put(offset + 4, y);
    }

    const DlssNetworkLayout& layout;

private:
    std::array<u8, MaxDlssNetworkSize> bytes{};
};

/// One NVNtexture and its optional NVNtextureView, as the parameters keep them.
struct TextureEntry {
    u64 texture;
    u64 view;
};

/// NVNdlssParameters in place, 0xC0 bytes (OoT sdk+0x65ad60..: SetDefaults and the accessors).
struct Parameters {
    TextureEntry input_color;     // +0x00
    TextureEntry output_color;    // +0x10
    TextureEntry motion_vectors;  // +0x20
    TextureEntry depth;           // +0x30
    TextureEntry bias_mask;       // +0x40
    float jitter[2];              // +0x50
    float motion_scale[2];        // +0x58
    float pre_exposure;           // +0x60
    u32 reserved_64;
    TextureEntry exposure;        // +0x68
    u8 history_reset;             // +0x78 (NVNboolean, one byte)
    u8 reserved_79[3];
    s32 evaluation_index;         // +0x7c: only the SDK's devtools trace reads it
    s32 input_size[2];            // +0x80
    s32 input_base[2];            // +0x88
    s32 depth_base[2];            // +0x90
    s32 motion_vector_base[2];    // +0x98
    s32 bias_base[2];             // +0xa0
    s32 output_base[2];           // +0xa8
    u32 overlay_debug_layer;      // +0xb0: the four debug overlays, each a u32 NVNboolean,
    u32 overlay_full_screen;      // +0xb4  ~0 by default; stored and never applied (the host
    u32 overlay_show_nans;        // +0xb8  draws no DLSS overlay, and no getter exists)
    u32 overlay_jitter_debug;     // +0xbc
};
static_assert(std::is_standard_layout_v<Parameters> && std::is_trivially_copyable_v<Parameters>);
static_assert(offsetof(Parameters, input_color) == 0x00);
static_assert(offsetof(Parameters, output_color) == 0x10);
static_assert(offsetof(Parameters, motion_vectors) == 0x20);
static_assert(offsetof(Parameters, depth) == 0x30);
static_assert(offsetof(Parameters, bias_mask) == 0x40);
static_assert(offsetof(Parameters, jitter) == 0x50);
static_assert(offsetof(Parameters, motion_scale) == 0x58);
static_assert(offsetof(Parameters, pre_exposure) == 0x60);
static_assert(offsetof(Parameters, exposure) == 0x68);
static_assert(offsetof(Parameters, history_reset) == 0x78);
static_assert(offsetof(Parameters, evaluation_index) == 0x7c);
static_assert(offsetof(Parameters, input_size) == 0x80);
static_assert(offsetof(Parameters, input_base) == 0x88);
static_assert(offsetof(Parameters, depth_base) == 0x90);
static_assert(offsetof(Parameters, motion_vector_base) == 0x98);
static_assert(offsetof(Parameters, bias_base) == 0xa0);
static_assert(offsetof(Parameters, output_base) == 0xa8);
static_assert(offsetof(Parameters, overlay_debug_layer) == 0xb0);
static_assert(offsetof(Parameters, overlay_full_screen) == 0xb4);
static_assert(offsetof(Parameters, overlay_show_nans) == 0xb8);
static_assert(offsetof(Parameters, overlay_jitter_debug) == 0xbc);
static_assert(sizeof(Parameters) == 0xc0);

/// NGX_PerfQuality_Value, which is what the game feeds `SetMethod` and the SDK passes to the
/// backend as its quality argument.
constexpr u32 MethodMaxPerf = 0;
constexpr u32 MethodBalanced = 1;
constexpr u32 MethodMaxQuality = 2;
constexpr u32 MethodUltraPerformance = 3;
constexpr u32 MethodUltraQuality = 4;
constexpr u32 MethodDlaa = 5;

/// NVNformat of the two working textures GetStorageSize sizes (RG16F, RGBA16F).
constexpr u64 FormatRG16F = 0x11;
constexpr u64 FormatRGBA16F = 0x29;

/// Device builder flags that select the SDK's larger set of sub-allocations in GetStorageSize.
constexpr u32 DeviceFlagsLargeDlss = 0xc00;

/// Scratch builders live below the caller's stack pointer: nothing else uses those bytes while the
/// body runs (the SDK builds its texture builder on its stack the same way).
constexpr u64 BuilderScratch = 0x200;

/// Feature handles: what the backend's create returns in the SDK. Process-wide, never 0, unique per
/// Initialize (the replay keys its Streamline viewport on it).
std::atomic<u32> next_network_handle{1};

u32 NewNetworkHandle() {
    u32 handle;
    do {
        handle = next_network_handle.fetch_add(1, std::memory_order_relaxed);
    } while (handle == 0);
    return handle;
}

/// The backend's create flags (OoT sdk+0x659e40): bits 0x4e as given and NGX's IsHDR, which is
/// the inverse of the builder's bit 0.
constexpr u32 NgxCreateFlags(u32 flags) {
    return (flags & VC::DLSS_FLAGS_TO_NGX) | ((flags & VC::DLSS_FLAG_LDR) != 0 ? 0u : 1u);
}

/// The render sizes GetOptimalSettings reports (and Initialize/Resize keep in the network).
struct RenderSizes {
    s32 optimal[2];
    s32 max[2];
    s32 min[2];
};

/// Known limitation: the SDK asks the system "dlss" library for these numbers, which is not in
/// the game, so they are the host's. Scale ratio of the input resolution to the output one, by
/// method: MaxPerf 1/2, Balanced 0.58, MaxQuality 2/3, UltraPerformance 1/3 and DLAA 1 are NGX's
/// documented factors; UltraQuality is 0.77 (the host's reading of NGX's ultra quality factor).
/// Balanced is 29/50 and UltraQuality 77/100 exactly, so the rounding does not depend on float
/// representation. The maximum is the full output, the minimum a third of it (DLSS's lowest
/// scale, UltraPerformance), never above the optimal size. An unknown method or an output size
/// that is not positive has no sizes (refused).
std::optional<RenderSizes> RenderSizesFor(u32 method, s32 width, s32 height) {
    struct Ratio {
        u64 numerator;
        u64 denominator;
    };
    Ratio ratio;
    switch (method) {
    case MethodMaxPerf:
        ratio = {1, 2};
        break;
    case MethodBalanced:
        ratio = {29, 50};
        break;
    case MethodMaxQuality:
        ratio = {2, 3};
        break;
    case MethodUltraPerformance:
        ratio = {1, 3};
        break;
    case MethodUltraQuality:
        ratio = {77, 100};
        break;
    case MethodDlaa:
        ratio = {1, 1};
        break;
    default:
        return std::nullopt;
    }
    if (width <= 0 || height <= 0) {
        return std::nullopt;
    }
    const auto scaled = [&](s32 output) {
        const u64 size = static_cast<u64>(output);
        const u64 rounded = (size * ratio.numerator + ratio.denominator / 2) / ratio.denominator;
        return static_cast<s32>(std::clamp<u64>(rounded, 1, size));
    };
    const auto floor_size = [](s32 output, s32 optimal) {
        return std::min(optimal, std::max<s32>(1, static_cast<s32>((s64{output} + 2) / 3)));
    };
    RenderSizes sizes{};
    sizes.optimal[0] = scaled(width);
    sizes.optimal[1] = scaled(height);
    sizes.max[0] = width;
    sizes.max[1] = height;
    sizes.min[0] = floor_size(width, sizes.optimal[0]);
    sizes.min[1] = floor_size(height, sizes.optimal[1]);
    return sizes;
}

/// Writes a network's render sizes for its output size (Initialize, Resize).
void PutRenderSizes(Network& n, const RenderSizes& sizes) {
    n.PutPair(n.layout.optimal, sizes.optimal[0], sizes.optimal[1]);
    n.PutPair(n.layout.max_render, sizes.max[0], sizes.max[1]);
    n.PutPair(n.layout.min_render, sizes.min[0], sizes.min[1]);
}

constexpr u64 Page(u64 bytes) {
    return bytes & ~u64{0xfff};
}

/// nvnTextureBuilderGetStorageSize of a default 2D texture builder of the builder's output size and
/// `format`, asked through our own texture builder as the SDK asks the driver's.
u64 WorkingTextureSize(Call& c, const Builder& b, u64 format) {
    const u64 builder = c.ctx.sp - BuilderScratch;
    c.CallNvn("nvnTextureBuilderSetDefaults", {builder});
    c.CallNvn("nvnTextureBuilderSetDevice", {builder, b.device});
    c.CallNvn("nvnTextureBuilderSetSize2D",
              {builder, static_cast<u32>(b.width), static_cast<u32>(b.height)});
    c.CallNvn("nvnTextureBuilderSetFormat", {builder, format});
    return c.CallNvn("nvnTextureBuilderGetStorageSize", {builder});
}

/// nvnDlssNetworkBuilderGetStorageSize as the SDK computes it (OoT sdk+0x659fc0, Star Fox
/// sdk+0x661790; checked against both SDKs' code for 2560 input combinations each): the sum of its
/// page-aligned sub-allocations, which depend on the output size, the entries, Balanced (method 1),
/// presets 8/9, the device's flags 0xc00 (device +8, its builder flags), flag bit 29 and, except
/// for presets 8/9, the storage of an RG16F and an RGBA16F texture of the output size. All
/// arithmetic wraps like the SDK's 64-bit registers; the sizes and entries are sign-extended.
u64 StorageSizeFor(Call& c, const Builder& b) {
    const u64 width = static_cast<u64>(static_cast<s64>(b.width));
    const u64 height = static_cast<u64>(static_cast<s64>(b.height));
    const u64 entries = static_cast<u64>(static_cast<s64>(b.max_evaluation_entries)) << 16;
    const bool balanced = b.method == MethodBalanced;
    const bool preset_8_9 = (b.preset & ~1u) == 8;
    const bool large =
        (c.runtime.device.flags.load(std::memory_order_relaxed) & DeviceFlagsLargeDlss) ==
        DeviceFlagsLargeDlss;
    const bool extra = (b.flags & VC::DLSS_FLAG_EXTRA_STORAGE) != 0;
    const u64 area = (width + 64) * (height + 64);

    u64 rg16f = 0;
    u64 rgba16f = 0;
    if (!preset_8_9) {
        rg16f = Page(WorkingTextureSize(c, b, FormatRG16F) + 0xfff);
        rgba16f = Page(WorkingTextureSize(c, b, FormatRGBA16F) + 0xfff);
    }
    // The two rodata tables of fixed sub-allocations (OoT sdk+0xbb3200.., Star Fox sdk+0xbaa840..):
    // 0x34000+0x54000+0x5000+0x12000+0x84000+0x47000+0x1b000+0x37000 for the large device and
    // 0x23000+0x13000+0x2000+0x9000+0x43000+0x43000+0x12000+0x16000 otherwise.
    constexpr u64 TableLarge = 0x1bc000;
    constexpr u64 TableSmall = 0xef000;
    const u64 history = 2 * Page(2 * area + 0x357ff);

    u64 x;
    if (preset_8_9) {
        x = (balanced ? entries + 0x2000 : history + entries + 0x4000) + 0xf7000;
    } else if (balanced) {
        x = rg16f + entries + 0x3000 + 0xf7000;
    } else {
        x = rg16f + entries + 0x3000 + (large ? TableLarge : TableSmall) +
            (large ? 0x8000 : 0x6000) + 0x2000;
    }
    if (extra) {
        x += Page(area + 0x63abf) + Page(area + 0xfff) + 0x2000;
    }
    if (balanced) {
        x += 0x27000;
    } else {
        x += (preset_8_9 || large) ? 0x27000 : 0x2000;
        if (!preset_8_9) {
            x += 0x25000;
        }
    }
    if (extra) {
        x += Page(area + 0xfff) + 0x1000;
    }

    u64 y = 0xdc000 + x;
    if (balanced) {
        y += 0x2000;
    } else if (preset_8_9) {
        y += Page(area + 0xfff) + 0x3000;
    } else {
        y += large ? 0x4b000 + 0x91000 : 0x2000;
    }
    if (extra) {
        y += Page(2 * area + 0xc657f) + 0x1000;
    }
    if (!preset_8_9) {
        y += rgba16f + 0x1000;
    }

    const u64 z = Page(area + 0x2c82d);
    u64 f = y + z + 0x1000;
    if (preset_8_9 && !balanced) {
        f += history + 0x2000;
    }
    f += Page(7 * area + 0x7b6fff);

    u64 total;
    if (preset_8_9) {
        total = f + z + 0xc000;
    } else {
        total = rg16f + z + f + (!balanced && large ? 0x7000 : 0x5000) + 0x8000 +
                Page(7 * area + 0x7f53ff) + 0x1000;
    }
    return Page(total + 0xfff);
}

#define DLSS_FIELD(member) (offsetof(Builder, member))

template <typename T>
void SetField(Call& c, u64 object, size_t offset, T value) {
    c.Store<T>(object + offset, value);
}

/// Per-call-site warning, the first few per boot (the game evaluates every frame).
#define DLSS_WARN(...)                                                                             \
    do {                                                                                           \
        static Common::BootCounter dlss_warned;                                                    \
        if (dlss_warned.Next() < 4) {                                                              \
            LOG_WARNING(Core, __VA_ARGS__);                                                        \
        }                                                                                          \
    } while (false)

// ---- NVNdlssNetworkBuilder -------------------------------------------------------------------------

/// Clears everything but the device (+0x00), up to the end of the SDK's builder, and sets one
/// evaluation entry.
void nvnDlssNetworkBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u32 size = c.runtime.device.GetDlssLayout().builder_size;
    if (size > DLSS_FIELD(texture_pool)) {
        c.Zero(builder + DLSS_FIELD(texture_pool), size - DLSS_FIELD(texture_pool));
    }
    SetField<s32>(c, builder, DLSS_FIELD(max_evaluation_entries), 1);
}

void nvnDlssNetworkBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    SetField(c, builder, DLSS_FIELD(device), c.args.NextPointer());
}

void nvnDlssNetworkBuilderSetFlags(Call& c) {
    const u64 builder = c.args.NextPointer();
    SetField(c, builder, DLSS_FIELD(flags), c.args.NextU32());
}

void nvnDlssNetworkBuilderSetMaxEvaluationEntries(Call& c) {
    const u64 builder = c.args.NextPointer();
    SetField(c, builder, DLSS_FIELD(max_evaluation_entries), c.args.NextS32());
}

void nvnDlssNetworkBuilderSetMethod(Call& c) {
    const u64 builder = c.args.NextPointer();
    SetField(c, builder, DLSS_FIELD(method), c.args.NextU32());
}

void nvnDlssNetworkBuilderSetOutputResolution(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    SetField(c, builder, DLSS_FIELD(width), width);
    SetField(c, builder, DLSS_FIELD(height), height);
}

void nvnDlssNetworkBuilderSetPreset(Call& c) {
    const u64 builder = c.args.NextPointer();
    SetField(c, builder, DLSS_FIELD(preset), c.args.NextU32());
}

void nvnDlssNetworkBuilderSetStorage(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 storage = c.args.NextPointer();
    const u64 size = c.args.NextU64();
    SetField(c, builder, DLSS_FIELD(storage), storage);
    SetField(c, builder, DLSS_FIELD(storage_size), size);
}

/// (builder, texture pool, const void* ids): the pool and the 128 bytes of texture IDs the
/// network may use, copied into the builder (the SDK takes no count).
void nvnDlssNetworkBuilderSetTextureIDs(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 pool = c.args.NextPointer();
    const u64 ids = c.args.NextPointer();
    SetField(c, builder, DLSS_FIELD(texture_pool), pool);
    c.Store(builder + DLSS_FIELD(texture_ids), c.Load<decltype(Builder::texture_ids)>(ids));
}

void nvnDlssNetworkBuilderGetDevice(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + DLSS_FIELD(device)));
}

void nvnDlssNetworkBuilderGetFlags(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + DLSS_FIELD(flags)));
}

void nvnDlssNetworkBuilderGetMaxEvaluationEntries(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + DLSS_FIELD(max_evaluation_entries)));
}

/// GetMemory / GetMemorySize: the storage SetStorage was given.
void nvnDlssNetworkBuilderGetMemory(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + DLSS_FIELD(storage)));
}

void nvnDlssNetworkBuilderGetMemorySize(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + DLSS_FIELD(storage_size)));
}

void nvnDlssNetworkBuilderGetMethod(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + DLSS_FIELD(method)));
}

void nvnDlssNetworkBuilderGetPreset(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + DLSS_FIELD(preset)));
}

/// (const builder, int* width, int* height): both stored, as the SDK does (it has no NULL check;
/// a NULL pointer is a write the guest memory refuses here).
void nvnDlssNetworkBuilderGetOutputResolution(Call& c) {
    const Builder b = LoadBuilder(c, c.args.NextPointer(), c.runtime.device.GetDlssLayout());
    const u64 width = c.args.NextPointer();
    const u64 height = c.args.NextPointer();
    c.Store(width, b.width);
    c.Store(height, b.height);
}

void nvnDlssNetworkBuilderGetTexturePool(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + DLSS_FIELD(texture_pool)));
}

/// (const builder, void* ids): copies the 128 bytes of texture IDs out; no return value.
void nvnDlssNetworkBuilderGetTextureIDs(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 ids = c.args.NextPointer();
    c.Store(ids, c.Load<decltype(Builder::texture_ids)>(builder + DLSS_FIELD(texture_ids)));
}

void nvnDlssNetworkBuilderGetStorageSize(Call& c) {
    const Builder b = LoadBuilder(c, c.args.NextPointer(), c.runtime.device.GetDlssLayout());
    c.ret.SetU64(StorageSizeFor(c, b));
}

/// (const builder, int* optimal_w, int* optimal_h, int* max_w, int* max_h, int* min_w, int* min_h)
/// -> NVNboolean: the six pointers the SDK forwards to the backend (OoT sdk+0x659ae0). The sizes
/// are the host's (RenderSizesFor: a known limitation); every pointer may be NULL. Refused, with
/// nothing written, for an unknown method or an output size that is not positive.
void nvnDlssNetworkBuilderGetOptimalSettings(Call& c) {
    const Builder b = LoadBuilder(c, c.args.NextPointer(), c.runtime.device.GetDlssLayout());
    const u64 optimal_w = c.args.NextPointer();
    const u64 optimal_h = c.args.NextPointer();
    const u64 max_w = c.args.NextPointer();
    const u64 max_h = c.args.NextPointer();
    const u64 min_w = c.args.NextPointer();
    const u64 min_h = c.args.NextPointer();

    const std::optional<RenderSizes> sizes = RenderSizesFor(b.method, b.width, b.height);
    if (!sizes) {
        c.ret.SetBool(false);
        return;
    }
    const auto put = [&](u64 va, s32 value) {
        if (va != 0) {
            c.Store(va, value);
        }
    };
    put(optimal_w, sizes->optimal[0]);
    put(optimal_h, sizes->optimal[1]);
    put(max_w, sizes->max[0]);
    put(max_h, sizes->max[1]);
    put(min_w, sizes->min[0]);
    put(min_h, sizes->min[1]);
    c.ret.SetBool(true);
}

// ---- NVNdlssNetwork --------------------------------------------------------------------------------

/// (network, const builder) -> NVNboolean (OoT sdk+0x659b40). The SDK clears the network (all of
/// the bytes of its layout, DlssNetworkLayout::size) before anything else, so a refused network is
/// left zeroed (handle 0). Refused (host: the backend's create would fail) for no device, no
/// storage, an output size that is not positive, an unknown method (GetOptimalSettings refuses) or
/// storage smaller than GetStorageSize (the backend allocates from it). On success the game's
/// storage is cleared as the SDK does (sdk+0x659d10) and the network gets a new feature handle.
/// An SDK whose network is of no known layout gets nothing written and every network refused.
void nvnDlssNetworkInitialize(Call& c) {
    const u64 network = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const DlssLayout layout = c.runtime.device.GetDlssLayout();
    if (layout.network == nullptr) {
        DLSS_WARN("[nvn] nvnDlssNetworkInitialize: the game's SDK has a DLSS network of no known "
                  "layout; network {:#x} is refused",
                  network);
        c.ret.SetBool(false);
        return;
    }
    const DlssNetworkLayout& l = *layout.network;
    c.Zero(network, l.size);
    const Builder b = LoadBuilder(c, builder, layout);
    const std::optional<RenderSizes> sizes = RenderSizesFor(b.method, b.width, b.height);
    if (b.device == 0 || b.storage == 0 || !sizes || b.storage_size < StorageSizeFor(c, b)) {
        c.ret.SetBool(false);
        return;
    }
    if (!c.Zero(b.storage, b.storage_size)) {
        // (host) Storage the guest does not map is no memory the backend could work in.
        LOG_WARNING(Core,
                    "[nvn] nvnDlssNetworkInitialize: storage {:#x}+{:#x} is not guest memory; the "
                    "network is refused",
                    b.storage, b.storage_size);
        c.ret.SetBool(false);
        return;
    }
    Network n{l};
    n.Put(DlssNetworkDevice, b.device);
    n.Put(DlssNetworkTexturePool, b.texture_pool);
    n.Put(DlssNetworkTextureIds, b.texture_ids);
    n.Put(l.storage_device, b.device);
    n.Put(l.storage, b.storage);
    n.Put(l.storage_size, b.storage_size);
    n.Put(l.max_evaluation_entries, b.max_evaluation_entries);
    n.Put(l.method, b.method);
    n.Put(l.ngx_flags, NgxCreateFlags(b.flags));
    n.Put(l.quality, b.method);
    n.Put(l.preset, b.preset);
    PutRenderSizes(n, *sizes);
    n.PutPair(l.output, b.width, b.height);
    n.Put<u32>(l.output_subrects, (b.flags & VC::DLSS_FLAG_OUTPUT_SUBRECTS) != 0 ? 1 : 0);
    n.PutPair(l.current_output, b.width, b.height);
    if (l.initial_output != 0) {
        n.PutPair(l.initial_output, b.width, b.height);
    }
    n.Put(l.flags, b.flags);
    n.Put<u64>(DlssNetworkHandle, NewNetworkHandle());
    c.ret.SetBool(n.Store(c, network));
}

/// (network): the SDK destroys the backend feature and clears the handle; nothing else of the
/// network changes (OoT sdk+0x659f20).
void nvnDlssNetworkFinalize(Call& c) {
    SetField<u64>(c, c.args.NextPointer(), DlssNetworkHandle, 0);
}

/// (network, int width, int height) -> NVNboolean (OoT sdk+0x6599d0): never larger than the output
/// size the network was initialized with. Also refused (host) for a network without a feature
/// and a size that is not positive. Sets the current and the reported output size and recomputes
/// the render sizes for the network's method. An SDK without a Resize (its network keeps no
/// initial size, API 231.16) gives the game no way to call it: refused.
void nvnDlssNetworkResize(Call& c) {
    const u64 network = c.args.NextPointer();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    const DlssLayout layout = c.runtime.device.GetDlssLayout();
    if (layout.network == nullptr || layout.network->initial_output == 0) {
        DLSS_WARN("[nvn] nvnDlssNetworkResize: the game's SDK has no Resize for its DLSS network "
                  "layout; network {:#x} is not resized",
                  network);
        c.ret.SetBool(false);
        return;
    }
    const DlssNetworkLayout& l = *layout.network;
    Network n{l};
    n.Load(c, network);
    if (width > n.Get<s32>(l.initial_output) || height > n.Get<s32>(l.initial_output + 4)) {
        c.ret.SetBool(false);
        return;
    }
    const std::optional<RenderSizes> sizes = RenderSizesFor(n.Get<u32>(l.method), width, height);
    if (n.Get<u64>(DlssNetworkHandle) == 0 || !sizes) {
        c.ret.SetBool(false);
        return;
    }
    n.PutPair(l.current_output, width, height);
    n.PutPair(l.output, width, height);
    PutRenderSizes(n, *sizes);
    c.ret.SetBool(n.Store(c, network));
}

// ---- NVNdlssParameters ------------------------------------------------------------------------------

#define DLSS_PARAM(member) (offsetof(Parameters, member))

/// (params, texture, view): the image and the optional NVNtextureView selecting its level, layer
/// and format; the replay resolves both when the evaluation is recorded.
void SetTexture(Call& c, size_t offset) {
    const u64 params = c.args.NextPointer();
    TextureEntry entry{};
    entry.texture = c.args.NextPointer();
    entry.view = c.args.NextPointer();
    c.Store(params + offset, entry);
}

/// (params, int x, int y): the origin of one of the rectangles in its image.
void SetBase(Call& c, size_t offset) {
    const u64 params = c.args.NextPointer();
    const s32 x = c.args.NextS32();
    const s32 y = c.args.NextS32();
    const s32 base[2] = {x, y};
    c.Store(params + offset, base);
}

/// (const params, T* first, T* second): a pair of stored values (rectangle origins, the input size,
/// jitter, motion vector scale); either pointer may be NULL.
template <typename T>
void GetPair(Call& c, size_t offset) {
    const u64 params = c.args.NextPointer();
    const u64 first = c.args.NextPointer();
    const u64 second = c.args.NextPointer();
    if (first != 0) {
        c.Store(first, c.Load<T>(params + offset));
    }
    if (second != 0) {
        c.Store(second, c.Load<T>(params + offset + sizeof(T)));
    }
}

/// Every byte of the SDK's parameters: zero, motion vector scale (1, 1), pre-exposure 1 and every
/// overlay ~0 (0xC0 bytes; the API 231.16 SDK's are 0xC8, the last 8 zero).
void nvnDlssParametersSetDefaults(Call& c) {
    const u64 params = c.args.NextPointer();
    const u32 size = c.runtime.device.GetDlssLayout().parameters_size;
    Parameters defaults{};
    defaults.motion_scale[0] = 1.0f;
    defaults.motion_scale[1] = 1.0f;
    defaults.pre_exposure = 1.0f;
    defaults.overlay_debug_layer = ~0u;
    defaults.overlay_full_screen = ~0u;
    defaults.overlay_show_nans = ~0u;
    defaults.overlay_jitter_debug = ~0u;
    c.mem.Write(params, &defaults, std::min<size_t>(size, sizeof(Parameters)));
    if (size > sizeof(Parameters)) {
        c.Zero(params + sizeof(Parameters), size - sizeof(Parameters));
    }
}

void nvnDlssParametersSetInputColor(Call& c) {
    SetTexture(c, DLSS_PARAM(input_color));
}

void nvnDlssParametersSetOutputColor(Call& c) {
    SetTexture(c, DLSS_PARAM(output_color));
}

void nvnDlssParametersSetMotionVectors(Call& c) {
    SetTexture(c, DLSS_PARAM(motion_vectors));
}

void nvnDlssParametersSetDepth(Call& c) {
    SetTexture(c, DLSS_PARAM(depth));
}

void nvnDlssParametersSetBiasCurrentColorMask(Call& c) {
    SetTexture(c, DLSS_PARAM(bias_mask));
}

void nvnDlssParametersSetExposureTexture(Call& c) {
    SetTexture(c, DLSS_PARAM(exposure));
}

void nvnDlssParametersSetInputSubrectangleSize(Call& c) {
    const u64 params = c.args.NextPointer();
    const s32 width = c.args.NextS32();
    const s32 height = c.args.NextS32();
    const s32 size[2] = {width, height};
    c.Store(params + DLSS_PARAM(input_size), size);
}

void nvnDlssParametersSetInputColorSubrectangleBase(Call& c) {
    SetBase(c, DLSS_PARAM(input_base));
}

void nvnDlssParametersSetDepthSubrectangleBase(Call& c) {
    SetBase(c, DLSS_PARAM(depth_base));
}

void nvnDlssParametersSetMotionVectorSubrectangleBase(Call& c) {
    SetBase(c, DLSS_PARAM(motion_vector_base));
}

void nvnDlssParametersSetOutputSubrectangleBase(Call& c) {
    SetBase(c, DLSS_PARAM(output_base));
}

void nvnDlssParametersSetBiasCurrentColorSubrectangleBase(Call& c) {
    SetBase(c, DLSS_PARAM(bias_base));
}

void nvnDlssParametersSetMotionVectorScale(Call& c) {
    const u64 params = c.args.NextPointer();
    const float x = c.args.NextF32();
    const float y = c.args.NextF32();
    const float scale[2] = {x, y};
    c.Store(params + DLSS_PARAM(motion_scale), scale);
}

void nvnDlssParametersSetJitterOffset(Call& c) {
    const u64 params = c.args.NextPointer();
    const float x = c.args.NextF32();
    const float y = c.args.NextF32();
    const float jitter[2] = {x, y};
    c.Store(params + DLSS_PARAM(jitter), jitter);
}

void nvnDlssParametersSetPreExposure(Call& c) {
    const u64 params = c.args.NextPointer();
    SetField(c, params, DLSS_PARAM(pre_exposure), c.args.NextF32());
}

void nvnDlssParametersSetHistoryReset(Call& c) {
    const u64 params = c.args.NextPointer();
    SetField(c, params, DLSS_PARAM(history_reset), static_cast<u8>(c.args.NextBool() ? 1 : 0));
}

void nvnDlssParametersSetEvaluationIndex(Call& c) {
    const u64 params = c.args.NextPointer();
    SetField(c, params, DLSS_PARAM(evaluation_index), c.args.NextS32());
}

/// (params, NVNboolean): the SDK stores the whole 32-bit w1 into the overlay's own field.
void SetOverlay(Call& c, size_t offset) {
    const u64 params = c.args.NextPointer();
    SetField(c, params, offset, c.args.NextU32());
}

void nvnDlssParametersSetOverlayDebugLayer(Call& c) {
    SetOverlay(c, DLSS_PARAM(overlay_debug_layer));
}

void nvnDlssParametersSetOverlayFullScreen(Call& c) {
    SetOverlay(c, DLSS_PARAM(overlay_full_screen));
}

void nvnDlssParametersSetOverlayJitterDebug(Call& c) {
    SetOverlay(c, DLSS_PARAM(overlay_jitter_debug));
}

void nvnDlssParametersSetOverlayShowNans(Call& c) {
    SetOverlay(c, DLSS_PARAM(overlay_show_nans));
}

void nvnDlssParametersGetInputSubrectangleSize(Call& c) {
    GetPair<s32>(c, DLSS_PARAM(input_size));
}

void nvnDlssParametersGetInputColorSubrectangleBase(Call& c) {
    GetPair<s32>(c, DLSS_PARAM(input_base));
}

void nvnDlssParametersGetDepthSubrectangleBase(Call& c) {
    GetPair<s32>(c, DLSS_PARAM(depth_base));
}

void nvnDlssParametersGetMotionVectorSubrectangleBase(Call& c) {
    GetPair<s32>(c, DLSS_PARAM(motion_vector_base));
}

void nvnDlssParametersGetOutputSubrectangleBase(Call& c) {
    GetPair<s32>(c, DLSS_PARAM(output_base));
}

void nvnDlssParametersGetBiasCurrentColorSubrectangleBase(Call& c) {
    GetPair<s32>(c, DLSS_PARAM(bias_base));
}

void nvnDlssParametersGetMotionVectorScale(Call& c) {
    GetPair<float>(c, DLSS_PARAM(motion_scale));
}

void nvnDlssParametersGetJitterOffset(Call& c) {
    GetPair<float>(c, DLSS_PARAM(jitter));
}

/// (const params) -> float.
void nvnDlssParametersGetPreExposure(Call& c) {
    c.ret.SetF32(c.Load<float>(c.args.NextPointer() + DLSS_PARAM(pre_exposure)));
}

/// (const params) -> NVNboolean.
void nvnDlssParametersGetHistoryReset(Call& c) {
    c.ret.SetBool(c.Load<u8>(c.args.NextPointer() + DLSS_PARAM(history_reset)) != 0);
}

/// (const params) -> int.
void nvnDlssParametersGetEvaluationIndex(Call& c) {
    c.ret.SetU32(static_cast<u32>(c.Load<s32>(c.args.NextPointer() + DLSS_PARAM(evaluation_index))));
}

// ---- Evaluation ---------------------------------------------------------------------------------------

/// (command buffer, const network, const parameters): the game's upscale, recorded as one command
/// that reads both objects now (the game reuses them for the next frame), with the SDK's rules
/// (OoT sdk+0x65a4e0, EvaluateDlssPayload in record.h): the input size is the input texture's when
/// the game left it 0x0 (sdk+0x65a9a0), the output origin counts only with the output-subrects
/// flag, the exposure texture and pre-exposure only for HDR colour without automatic exposure.
/// The SDK validates nothing; the host drops (with a warning) only what cannot be evaluated at
/// all: a network without a feature (never initialized, refused or finalized) and parameters
/// without an input or an output texture. Rectangles are the replay's to judge.
void nvnCommandBufferEvaluateDlss(Call& c) {
    const u64 cmd = c.args.NextPointer();
    const u64 network = c.args.NextPointer();
    const u64 params = c.args.NextPointer();
    const DlssLayout layout = c.runtime.device.GetDlssLayout();
    if (layout.network == nullptr) {
        DLSS_WARN("[nvn] nvnCommandBufferEvaluateDlss: the game's SDK has a DLSS network of no "
                  "known layout; the evaluation of network {:#x} is dropped",
                  network);
        return;
    }
    const DlssNetworkLayout& l = *layout.network;
    Network n{l};
    n.Load(c, network);
    Parameters p{};
    c.mem.Read(params, &p, std::min<size_t>(layout.parameters_size, sizeof(Parameters)));
    const u64 handle = n.Get<u64>(DlssNetworkHandle);
    const u32 flags = n.Get<u32>(l.flags);
    if (handle == 0) {
        DLSS_WARN("[nvn] nvnCommandBufferEvaluateDlss: network {:#x} has no DLSS feature (not "
                  "initialized); the evaluation is dropped",
                  network);
        return;
    }
    if (p.input_color.texture == 0 || p.output_color.texture == 0) {
        DLSS_WARN("[nvn] nvnCommandBufferEvaluateDlss: parameters {:#x} give no input/output "
                  "texture (input {:#x} output {:#x}); the evaluation is dropped",
                  params, p.input_color.texture, p.output_color.texture);
        return;
    }
    const auto rect = [](const s32 (&base)[2]) { return VC::DlssRect{base[0], base[1]}; };
    VC::Recorder::DlssEvaluateArgs args{};
    args.input_texture = p.input_color.texture;
    args.input_view = p.input_color.view;
    args.output_texture = p.output_color.texture;
    args.output_view = p.output_color.view;
    args.motion_vectors_texture = p.motion_vectors.texture;
    args.motion_vectors_view = p.motion_vectors.view;
    args.depth_texture = p.depth.texture;
    args.depth_view = p.depth.view;
    args.bias_texture = p.bias_mask.texture;
    args.bias_view = p.bias_mask.view;
    const bool exposure = (flags & (VC::DLSS_FLAG_LDR | VC::DLSS_FLAG_AUTO_EXPOSURE)) == 0;
    if (exposure) {
        args.exposure_texture = p.exposure.texture;
        args.exposure_view = p.exposure.view;
    }
    // The SDK passes 0 ("none") without exposure; 1.0 is the same for Streamline.
    args.pre_exposure = exposure ? p.pre_exposure : 1.0f;
    args.input_base = rect(p.input_base);
    args.depth_base = rect(p.depth_base);
    args.motion_vector_base = rect(p.motion_vector_base);
    args.bias_base = rect(p.bias_base);
    if ((flags & VC::DLSS_FLAG_OUTPUT_SUBRECTS) != 0) {
        args.output_base = rect(p.output_base);
    }
    args.input_width = p.input_size[0];
    args.input_height = p.input_size[1];
    if (p.input_size[0] == 0 && p.input_size[1] == 0) {
        // The input texture's level 0 size (the SDK reads the texture's +0x20/+0x24, the height's
        // low 16 bits).
        if (const std::optional<VC::TextureDesc> input = c.rec.GetTexture(p.input_color.texture)) {
            args.input_width = static_cast<s32>(input->width);
            args.input_height = static_cast<s32>(input->height & 0xFFFF);
        }
    }
    args.output_width = n.Get<s32>(l.current_output);
    args.output_height = n.Get<s32>(l.current_output + 4);
    std::memcpy(args.motion_scale, p.motion_scale, sizeof(args.motion_scale));
    std::memcpy(args.jitter, p.jitter, sizeof(args.jitter));
    args.history_reset = p.history_reset != 0;
    args.flags = flags;
    args.method = n.Get<u32>(l.quality);
    args.preset = n.Get<u32>(l.preset);
    args.network = static_cast<u32>(handle);
    c.rec.EvaluateDlss(cmd, args);
}

#undef DLSS_PARAM
#undef DLSS_WARN
#undef DLSS_FIELD

} // namespace

void RegisterDlssFunctions(Registry& registry) {
    NVN_HLE(registry, nvnDlssNetworkBuilderSetDefaults);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetDevice);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetFlags);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetMaxEvaluationEntries);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetMethod);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetOutputResolution);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetPreset);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetStorage);
    NVN_HLE(registry, nvnDlssNetworkBuilderSetTextureIDs);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetDevice);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetFlags);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetMaxEvaluationEntries);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetMemory);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetMemorySize);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetMethod);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetOptimalSettings);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetOutputResolution);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetPreset);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetStorageSize);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetTextureIDs);
    NVN_HLE(registry, nvnDlssNetworkBuilderGetTexturePool);
    NVN_HLE(registry, nvnDlssNetworkInitialize);
    NVN_HLE(registry, nvnDlssNetworkFinalize);
    NVN_HLE(registry, nvnDlssNetworkResize);
    NVN_HLE(registry, nvnCommandBufferEvaluateDlss);
    NVN_HLE(registry, nvnDlssParametersSetDefaults);
    NVN_HLE(registry, nvnDlssParametersSetInputColor);
    NVN_HLE(registry, nvnDlssParametersSetOutputColor);
    NVN_HLE(registry, nvnDlssParametersSetMotionVectors);
    NVN_HLE(registry, nvnDlssParametersSetDepth);
    NVN_HLE(registry, nvnDlssParametersSetBiasCurrentColorMask);
    NVN_HLE(registry, nvnDlssParametersSetExposureTexture);
    NVN_HLE(registry, nvnDlssParametersSetInputSubrectangleSize);
    NVN_HLE(registry, nvnDlssParametersSetInputColorSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersSetDepthSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersSetMotionVectorSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersSetOutputSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersSetBiasCurrentColorSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersSetMotionVectorScale);
    NVN_HLE(registry, nvnDlssParametersSetJitterOffset);
    NVN_HLE(registry, nvnDlssParametersSetPreExposure);
    NVN_HLE(registry, nvnDlssParametersSetHistoryReset);
    NVN_HLE(registry, nvnDlssParametersSetEvaluationIndex);
    NVN_HLE(registry, nvnDlssParametersSetOverlayDebugLayer);
    NVN_HLE(registry, nvnDlssParametersSetOverlayFullScreen);
    NVN_HLE(registry, nvnDlssParametersSetOverlayJitterDebug);
    NVN_HLE(registry, nvnDlssParametersSetOverlayShowNans);
    NVN_HLE(registry, nvnDlssParametersGetInputSubrectangleSize);
    NVN_HLE(registry, nvnDlssParametersGetInputColorSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersGetDepthSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersGetMotionVectorSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersGetOutputSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersGetBiasCurrentColorSubrectangleBase);
    NVN_HLE(registry, nvnDlssParametersGetMotionVectorScale);
    NVN_HLE(registry, nvnDlssParametersGetJitterOffset);
    NVN_HLE(registry, nvnDlssParametersGetPreExposure);
    NVN_HLE(registry, nvnDlssParametersGetHistoryReset);
    NVN_HLE(registry, nvnDlssParametersGetEvaluationIndex);
}

} // namespace Core::NvnHle::Hle
