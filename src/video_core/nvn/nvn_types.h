// SPDX-FileCopyrightText: Copyright (c) 2020 h1k421
// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT
//
// The NVN type definitions, enumerations and the procedure-name table are derived from the MIT-licensed
// nvn.h by h1k421 (reconstructed from Darksiders II DWARF information) as carried in
// https://github.com/seedds/nxnative (dm/gfx/nvn/include/nvn.h, names.txt, nvn_protos.h).
// See LICENSES/MIT.txt.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

#include "common/common_types.h"

// Guest-visible NVN types. Nothing here is behaviour: it is the layout contract the HLE reads and writes.
//
// Objects the game allocates are opaque blobs of the SDK's size. For objects the recorder manages (device,
// queue, memory pool, buffer, texture, sampler, command buffer, ...) the HLE never reinterprets them as host
// records; it stores a 16-byte tag at offset 0 (see the NVN2 HLE contract) and keeps host state elsewhere.
// Builders, the state objects (NVNblendState, NVNcolorState, NVNchannelMaskState, NVNmultisampleState,
// NVNpolygonState, NVNdepthStencilState, NVNvertexAttribState, NVNvertexStreamState) and texture views
// carry no tag: they are used in place in their packed layouts.
//
// Layout status of every type is in `LayoutTable` below. "Verified" means a documented layout that shipping
// games read or write directly and that nxnative's tests exercise; it does NOT mean it was measured on
// Donkey Kong Bananza (NVN2, SDK 20.5.19). Nothing has been measured there yet [UNVERIFIED for NVN2].

namespace VideoCommon::Nvn {

// ------------------------------------------------------------------------------------------------
// Scalars and handles
// ------------------------------------------------------------------------------------------------

using NVNboolean = u8;
using NVNtextureHandle = u64;
using NVNimageHandle = u64;
using NVNcommandHandle = u64;
using NVNbufferAddress = u64;
using NVNtextureAddress = u64;
using NVNdebugDomainId = s32;
using NVNstorageClass = s32;

// Device info items: nvn.h stops at 0x61. The game's own NVN2 `DeviceGetInteger` switch handles 0 to 0xA9
// (Donkey Kong Bananza sdk+0x5d9960); items 0x62-0xA9 are unnamed here [UNVERIFIED for NVN2].
inline constexpr s32 kDeviceInfoLastNvn1 = 0x61;
inline constexpr s32 kDeviceInfoLastNvn2 = 0xA9;

// ------------------------------------------------------------------------------------------------
// Enumerations (generated from nvn.h by tools/gen_nvn_functions.py)
// ------------------------------------------------------------------------------------------------

#include "video_core/nvn/nvn_enums.inc"

// ------------------------------------------------------------------------------------------------
// Plain structs: the layout is the public ABI. Guest pointers are u64 (the guest is AArch64).
// ------------------------------------------------------------------------------------------------

struct NVNrectangle {
    s32 x;
    s32 y;
    s32 width;
    s32 height;
};

struct NVNbufferRange {
    NVNbufferAddress address;
    u64 size;
};

struct NVNcopyRegion {
    s32 xoffset;
    s32 yoffset;
    s32 zoffset;
    s32 width;
    s32 height;
    s32 depth;
};

struct NVNshaderData {
    NVNbufferAddress data;
    u64 control; // guest pointer to the control section
};

struct NVNcounterData {
    u64 counter;
    u64 timestamp;
};

struct NVNseparateTextureHandle {
    u64 value;
};

struct NVNseparateSamplerHandle {
    u64 value;
};

// nvnCommandBufferDrawTexture: a rectangle in the render target (pixels) or in the texture (texels).
struct NVNdrawTextureRegion {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct NVNtextureSparseTileLayout {
    s32 numTiledLevels; // levels laid out in whole 64 KB tiles; the rest are the mip tail
    s32 tileWidth;
    s32 tileHeight;
    s32 tileDepth;
    std::array<u8, 16> reserved;
};

// Byte 0: log2 block height in the low nibble, log2 block depth in the high nibble.
struct NVNpackagedTextureLayout {
    std::array<u8, 8> layout;
};

// nvnMemoryPoolMapVirtual request. nvn.h declares it `char dummy[1]`; the real record is 0x28 bytes
// (nxnative hle_nvn.c `map_req`, asserted there and exercised by Monster Hunter Rise and SMBW).
struct NVNmappingRequest {
    u64 physicalPool; // guest pointer to the physical NVNmemoryPool
    s64 physicalOffset;
    s64 virtualOffset;
    u64 size;
    NVNstorageClass storageClass;
    u32 pad;
};

// nvnQueueGetError output. nvn.h declares it `char dummy[1]` as a placeholder; the real layout is not known.
// Intentionally an incomplete type so nobody takes sizeof of the placeholder or writes through it blindly.
struct NVNqueueErrorInfo;

// ------------------------------------------------------------------------------------------------
// Opaque objects
// ------------------------------------------------------------------------------------------------

enum class LayoutStatus : u32 {
    // Documented layout that games use directly; nxnative asserts it. Not measured on NVN2.
    Verified,
    // Size / offsets carried over from the NVN 1 headers (nvn.h). [UNVERIFIED for NVN2].
    Nvn1CarryOver,
    // Newer SDKs allocate a smaller object than nvn.h says: host records must fit `min_known_size`.
    ContradictedByNewerSdk,
    // The layout is not known (NVNqueueErrorInfo).
    Unknown,
};

// X(type, size, min_known_size, status, note)
//   size: the size nvn.h reserves. min_known_size: the smallest allocation any SDK is known to use.
#define NVN_OPAQUE_OBJECTS(X)                                                                          \
    X(NVNdevice, 0x3000, 0x3000, Nvn1CarryOver, "")                                                    \
    X(NVNdeviceBuilder, 0x40, 0x40, Nvn1CarryOver, "")                                                 \
    X(NVNqueue, 0x2000, 0x2000, Nvn1CarryOver, "")                                                     \
    X(NVNqueueBuilder, 0x40, 0x40, Nvn1CarryOver, "")                                                  \
    X(NVNwindow, 384, 384, Nvn1CarryOver, "")                                                          \
    X(NVNwindowBuilder, 0x40, 0x40, Nvn1CarryOver, "")                                                 \
    X(NVNtexture, 0xC0, 0x80, ContradictedByNewerSdk,                                                  \
      "0x80 on SDK 16 / NVN 55 (SMBW); a 0x88 host record corrupted the heap")                         \
    X(NVNtextureView, 0x28, 0x10, ContradictedByNewerSdk,                                              \
      "16 bytes on SDK 16 / NVN 55: games keep it in a 16-byte stack slot")                            \
    X(NVNtextureBuilder, 0x80, 0x80, Nvn1CarryOver, "")                                                \
    X(NVNprogram, 0xC0, 0x70, ContradictedByNewerSdk,                                                  \
      "smaller on SMBW; nxnative keeps its record at 0x70")                                            \
    X(NVNcommandBuffer, 0xA0, 0xA0, Nvn1CarryOver,                                                     \
      "guest inlined code writes the byte at +0x10 (SDK 16); see NVNcommandBufferGuestFields")         \
    X(NVNbuffer, 0x30, 0x30, Nvn1CarryOver,                                                            \
      "guest code writes offset at +0x18 and size at +0x20 (SDK 12.3.6); see NVNbufferGuestFields")    \
    X(NVNbufferBuilder, 0x40, 0x40, Nvn1CarryOver, "")                                                 \
    X(NVNsync, 0x40, 0x40, Nvn1CarryOver, "")                                                          \
    X(NVNmemoryPool, 0x100, 0x100, Nvn1CarryOver, "")                                                  \
    X(NVNmemoryPoolBuilder, 0x40, 0x40, Nvn1CarryOver, "")                                             \
    X(NVNtexturePool, 0x20, 0x20, Nvn1CarryOver, "")                                                   \
    X(NVNsamplerPool, 0x20, 0x20, Nvn1CarryOver, "")                                                   \
    X(NVNsampler, 0x30, 0x30, Nvn1CarryOver, "")                                                       \
    X(NVNsamplerBuilder, 0x60, 0x60, Nvn1CarryOver, "")                                                \
    X(NVNevent, 0x40, 0x40, Nvn1CarryOver, "")                                                         \
    X(NVNeventBuilder, 0x20, 0x20, Nvn1CarryOver, "")                                                  \
    X(NVNblendState, 0x8, 0x8, Nvn1CarryOver, "")                                                      \
    X(NVNcolorState, 0x4, 0x4, Nvn1CarryOver, "")                                                      \
    X(NVNchannelMaskState, 0x4, 0x4, Nvn1CarryOver, "")                                                \
    X(NVNpolygonState, 0x4, 0x4, Nvn1CarryOver, "")                                                    \
    X(NVNdepthStencilState, 0x8, 0x8, Nvn1CarryOver, "")                                               \
    X(NVNvertexAttribState, 0x4, 0x4, Nvn1CarryOver, "")                                               \
    X(NVNvertexStreamState, 0x8, 0x8, Nvn1CarryOver, "")                                               \
    X(NVNmultisampleState, 0x18, 0x18, Nvn1CarryOver, "")

// Opaque objects are byte blobs. Alignment is 8 where the size allows it (the 4-byte state objects keep 4)
// so `sizeof` equals the SDK size.
#define NVN_DECLARE_OPAQUE(type, size, min_size, status, note)                                        \
    struct type {                                                                                      \
        alignas((size) >= 8 ? 8 : (size)) std::array<u8, (size)> reserved;                             \
    };                                                                                                 \
    inline constexpr u32 SizeOf##type = (size);
NVN_OPAQUE_OBJECTS(NVN_DECLARE_OPAQUE)
#undef NVN_DECLARE_OPAQUE

// Guest-written fields. The HLE must READ these at every use and never write them.
struct NVNbufferGuestFields {
    std::array<u8, 0x18> head;
    u64 offset; // +0x18: offset of the buffer in its pool, written by guest ring allocators
    u64 size;   // +0x20: size of the buffer, written by guest ring allocators
    std::array<u8, 0x8> tail;
};
inline constexpr u32 kBufferOffsetFieldOffset = 0x18;
inline constexpr u32 kBufferSizeFieldOffset = 0x20;

struct NVNcommandBufferGuestFields {
    std::array<u8, 0x10> head;
    u8 guest_byte; // +0x10: written by inlined SDK code after AddCommandMemory; belongs to the guest
    std::array<u8, 0xA0 - 0x11> tail;
};
inline constexpr u32 kCommandBufferGuestByteOffset = 0x10;

// ------------------------------------------------------------------------------------------------
// Layout status table
// ------------------------------------------------------------------------------------------------

struct LayoutEntry {
    std::string_view name;
    u32 size;
    u32 min_known_size;
    LayoutStatus status;
    std::string_view note;
};

inline constexpr auto LayoutTable = std::to_array<LayoutEntry>({
#define NVN_LAYOUT_OPAQUE(type, size, min_size, status, note)                                         \
    LayoutEntry{#type, size, min_size, LayoutStatus::status, note},
    NVN_OPAQUE_OBJECTS(NVN_LAYOUT_OPAQUE)
#undef NVN_LAYOUT_OPAQUE
    LayoutEntry{"NVNrectangle", sizeof(NVNrectangle), sizeof(NVNrectangle), LayoutStatus::Verified, ""},
    LayoutEntry{"NVNbufferRange", sizeof(NVNbufferRange), sizeof(NVNbufferRange), LayoutStatus::Verified, ""},
    LayoutEntry{"NVNcopyRegion", sizeof(NVNcopyRegion), sizeof(NVNcopyRegion), LayoutStatus::Verified, ""},
    LayoutEntry{"NVNshaderData", sizeof(NVNshaderData), sizeof(NVNshaderData), LayoutStatus::Verified,
                "control is a guest pointer"},
    LayoutEntry{"NVNcounterData", sizeof(NVNcounterData), sizeof(NVNcounterData), LayoutStatus::Verified, ""},
    LayoutEntry{"NVNseparateTextureHandle", sizeof(NVNseparateTextureHandle), sizeof(NVNseparateTextureHandle),
                LayoutStatus::Verified, ""},
    LayoutEntry{"NVNseparateSamplerHandle", sizeof(NVNseparateSamplerHandle), sizeof(NVNseparateSamplerHandle),
                LayoutStatus::Verified, ""},
    LayoutEntry{"NVNdrawTextureRegion", sizeof(NVNdrawTextureRegion), sizeof(NVNdrawTextureRegion),
                LayoutStatus::Verified, ""},
    LayoutEntry{"NVNtextureSparseTileLayout", sizeof(NVNtextureSparseTileLayout),
                sizeof(NVNtextureSparseTileLayout), LayoutStatus::Verified, ""},
    LayoutEntry{"NVNpackagedTextureLayout", sizeof(NVNpackagedTextureLayout), sizeof(NVNpackagedTextureLayout),
                LayoutStatus::Verified, ""},
    LayoutEntry{"NVNmappingRequest", sizeof(NVNmappingRequest), sizeof(NVNmappingRequest),
                LayoutStatus::Verified, "nvn.h has `char dummy[1]`; real size 0x28 per nxnative"},
    LayoutEntry{"NVNqueueErrorInfo", 0, 0, LayoutStatus::Unknown, "nvn.h has `char dummy[1]`; layout unknown"},
});

[[nodiscard]] constexpr size_t CountLayoutStatus(LayoutStatus status) {
    size_t count = 0;
    for (const LayoutEntry& entry : LayoutTable) {
        count += entry.status == status ? 1 : 0;
    }
    return count;
}

[[nodiscard]] constexpr const LayoutEntry* FindLayout(std::string_view name) {
    for (const LayoutEntry& entry : LayoutTable) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

// ------------------------------------------------------------------------------------------------
// Procedure table. ID = 0-based line index in nxnative dm/gfx/nvn/names.txt.
// `nvnBootstrapLoader` is not in the table: it is the one symbol the game imports, and it resolves these.
// ------------------------------------------------------------------------------------------------

enum class FunctionId : u32 {
#define NVN_FUNCTION(id, name, has_signature) name = id,
#include "video_core/nvn/nvn_functions.inc"
#undef NVN_FUNCTION
};

struct FunctionInfo {
    u32 id;
    std::string_view name;
    bool has_signature;
};

inline constexpr auto FunctionTable = std::to_array<FunctionInfo>({
#define NVN_FUNCTION(id, name, has_signature) FunctionInfo{id, #name, has_signature != 0},
#include "video_core/nvn/nvn_functions.inc"
#undef NVN_FUNCTION
});
inline constexpr size_t NumFunctions = FunctionTable.size();

// C spelling of a function's prototype with the nxnative header bugs corrected (see gen_nvn_functions.py).
// Only functions whose prototype exists in nxnative have one (`has_signature`); the rest (about 257, the
// newer interfaces: ray tracing, DLSS, ProgramBuilder, HDR metadata, push constants, deferred binds, ...)
// are known by name only.
struct SignatureInfo {
    u32 id;
    std::string_view name;
    std::string_view return_type;
    std::string_view parameters;
};

inline constexpr auto SignatureTable = std::to_array<SignatureInfo>({
#define NVN_SIGNATURE(id, name, ret, params) SignatureInfo{id, #name, ret, params},
#include "video_core/nvn/nvn_signatures.inc"
#undef NVN_SIGNATURE
});

// Linear lookup: fine for setup paths; callers on hot paths keep their own hash map.
[[nodiscard]] constexpr std::optional<u32> FindFunctionId(std::string_view name) {
    for (const FunctionInfo& info : FunctionTable) {
        if (info.name == name) {
            return info.id;
        }
    }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view FunctionName(u32 id) {
    return id < FunctionTable.size() ? FunctionTable[id].name : std::string_view{};
}

// Ids are ascending in SignatureTable.
[[nodiscard]] constexpr const SignatureInfo* FindSignature(u32 id) {
    size_t lo = 0;
    size_t hi = SignatureTable.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (SignatureTable[mid].id < id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < SignatureTable.size() && SignatureTable[lo].id == id ? &SignatureTable[lo] : nullptr;
}

// ------------------------------------------------------------------------------------------------
// Static checks
// ------------------------------------------------------------------------------------------------

#define NVN_CHECK_OPAQUE(type, size, min_size, status, note)                                          \
    static_assert(sizeof(type) == (size), #type " size differs from nvn.h");                           \
    static_assert(sizeof(type) == SizeOf##type);                                                       \
    static_assert((min_size) <= (size), #type " min_known_size exceeds size");
NVN_OPAQUE_OBJECTS(NVN_CHECK_OPAQUE)
#undef NVN_CHECK_OPAQUE

static_assert(sizeof(NVNrectangle) == 16);
static_assert(sizeof(NVNbufferRange) == 16);
static_assert(sizeof(NVNcopyRegion) == 24);
static_assert(sizeof(NVNshaderData) == 16);
static_assert(sizeof(NVNcounterData) == 16);
static_assert(sizeof(NVNseparateTextureHandle) == 8);
static_assert(sizeof(NVNseparateSamplerHandle) == 8);
static_assert(sizeof(NVNdrawTextureRegion) == 16);
static_assert(sizeof(NVNtextureSparseTileLayout) == 32);
static_assert(sizeof(NVNpackagedTextureLayout) == 8);
static_assert(sizeof(NVNmappingRequest) == 0x28);
static_assert(offsetof(NVNmappingRequest, size) == 0x18);
static_assert(offsetof(NVNmappingRequest, storageClass) == 0x20);

static_assert(sizeof(NVNbufferGuestFields) == SizeOfNVNbuffer);
static_assert(offsetof(NVNbufferGuestFields, offset) == kBufferOffsetFieldOffset);
static_assert(offsetof(NVNbufferGuestFields, size) == kBufferSizeFieldOffset);
static_assert(sizeof(NVNcommandBufferGuestFields) == SizeOfNVNcommandBuffer);
static_assert(offsetof(NVNcommandBufferGuestFields, guest_byte) == kCommandBufferGuestByteOffset);

// Every tagged (non-builder) object must hold the 16-byte HLE tag at offset 0; the smallest known allocation
// that gets one is NVNtexture at 0x80. The 0x10-byte NVNtextureView is a value type and is never tagged.
static_assert(SizeOfNVNtexture >= 16 && SizeOfNVNsync >= 16 && SizeOfNVNsamplerPool >= 16);

static_assert(NumFunctions == 734, "names.txt has 734 names");
static_assert(static_cast<u32>(FunctionId::nvnDeviceBuilderSetDefaults) == 0);

} // namespace VideoCommon::Nvn
