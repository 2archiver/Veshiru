// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"
#include "video_core/nvn/sink.h"

namespace VideoCommon::Nvn {

/// Version of the record format. Bump whenever a payload layout changes or an op is added (a decoder
/// that does not know the op cannot skip or read it): 3 changed WaitEvent (queue sync), 4 added
/// EvaluateDlss, 5 added DrawMeshTasks, 6 replaced EvaluateDlss' evaluation index with the network
/// id, 7 added SetProvokingVertex.
inline constexpr u32 RECORD_FORMAT_VERSION = 7;

/// A recorded command list is a std::vector<u64>. Every record starts with a RecordHeader and is
/// followed by its payload; `size` counts the header and is a multiple of 8, so a record always
/// occupies a whole number of u64 words and payload structs are 8-byte aligned.
struct RecordHeader {
    u16 op;
    u16 flags;
    u32 size;
};
static_assert(sizeof(RecordHeader) == 8);

/// Ops 0-31 are the shared contract. Draw-path ops start at 32 (owned by the recorder slice).
enum class Op : u16 {
    End = 0,
    CallList = 1,
    ClearTexture = 2,
    ClearBuffer = 3,
    CopyBufferToBuffer = 4,
    CopyBufferToTexture = 5,
    CopyTextureToBuffer = 6,
    CopyTextureToTexture = 7,
    FenceSync = 8,
    SignalEvent = 9,
    WaitSync = 10,
    WaitEvent = 11,

    // ---- Draw-path ops (recorder slice). Payload structs are below the contract structs. ----
    SetRenderTargets = 32,
    ClearColor = 33,
    ClearDepthStencil = 34,
    SetViewports = 35,
    SetScissors = 36,
    SetDepthRanges = 37,
    BindProgram = 38,
    BindState = 39,
    BindVertexAttribs = 40,
    BindVertexStreams = 41,
    BindVertexBuffers = 42,
    BindUniformBuffers = 43,
    BindStorageBuffers = 44,
    BindTextures = 45,         ///< combined texture+sampler handles
    BindSeparateTextures = 46, ///< separate texture handles (slots 0..n of the separate-texture table)
    BindSeparateSamplers = 47,
    BindImages = 48,
    SetStencil = 49,
    SetPolygonOffset = 50,
    Draw = 51,
    DrawIndexed = 52,
    Dispatch = 53,
    Barrier = 54,
    SetTexturePool = 55,
    SetSamplerPool = 56,
    SetBlendColor = 57,
    SetDepthClamp = 58,
    SetAlphaRef = 59,
    PushConstants = 60,
    UpdateUniformBuffer = 61,
    ReportValue = 62,
    ReportCounter = 63,
    ResetCounter = 64,
    SetRenderCondition = 65,
    DrawTexture = 66,
    SetPrimitiveRestart = 67,
    SetViewportSwizzles = 68,
    BindTransformFeedbackBuffer = 69,
    TransformFeedbackState = 70,
    SetRasterizerDiscard = 71,
    SetSampleMask = 72,
    SetPointSize = 73,
    SetLineWidth = 74,
    DrawTransformFeedback = 75,
    EvaluateDlss = 76,
    DrawMeshTasks = 77,
    /// nvnCommandBufferSetProvokingVertexMode: U32Payload, the SET_PROVOKING_VERTEX (0x1684) value
    /// the SDK emits (0 first vertex, 1 last vertex).
    SetProvokingVertex = 78,
};

struct CopyRegion {
    s32 x, y, z, width, height, depth;
}; // NVNcopyRegion layout

struct TextureRef {
    u64 serial;
    u32 level;
    u32 layer;
};

/// ClearTexture: RecordHeader::flags holds the channel write mask (bit0 R, bit1 G, bit2 B, bit3 A).
struct ClearTexturePayload {
    TextureRef texture;
    CopyRegion region;
    u32 layer_count;
    u32 is_integer;
    u32 color[4];
};

struct ClearBufferPayload {
    u64 address;
    u64 size;
    u32 value;
    u32 pad;
};

struct CopyBufferToBufferPayload {
    u64 src;
    u64 dst;
    u64 size;
    u32 flags;
    u32 pad;
};

struct CopyBufferTexturePayload {
    u64 buffer;
    TextureRef texture;
    CopyRegion region;
    u32 row_stride;
    u32 image_stride;
    u32 flags;
    u32 pad;
};

struct CopyTextureToTexturePayload {
    TextureRef src;
    CopyRegion src_region;
    TextureRef dst;
    CopyRegion dst_region;
    u32 flags;
    u32 pad;
};

struct FenceSyncPayload {
    u64 sync_serial;
    u32 condition;
    u32 flags;
};

struct SignalEventPayload {
    u64 event_serial;
    u64 event_address; ///< the event's memory (guest VA): events are identified by their memory
    u32 mode;          ///< NVNeventSignalMode: 0 WRITE, 1 ADD
    u32 value;
};

/// nvnCommandBufferWaitEvent: the queue's stream stalls until the event's word satisfies the wait.
struct WaitEventPayload {
    u64 event_address;
    u32 mode;  ///< NVNeventWaitMode: 0 EQUAL, 1 GEQUAL_WRAP
    u32 value;
};

struct WaitSyncPayload {
    u64 sync_serial;
};

struct CallListPayload {
    u64 handle;
};

static_assert(sizeof(CopyRegion) == 24);
static_assert(sizeof(TextureRef) == 16);
static_assert(sizeof(ClearTexturePayload) == 64);
static_assert(sizeof(ClearBufferPayload) == 24);
static_assert(sizeof(CopyBufferToBufferPayload) == 32);
static_assert(sizeof(CopyBufferTexturePayload) == 64);
static_assert(sizeof(CopyTextureToTexturePayload) == 88);
static_assert(sizeof(FenceSyncPayload) == 16);
static_assert(sizeof(SignalEventPayload) == 24);
static_assert(sizeof(WaitEventPayload) == 16);
static_assert(sizeof(WaitSyncPayload) == 8);
static_assert(sizeof(CallListPayload) == 8);

// ---------------------------------------------------------------------------------------------
// Draw-path payloads (ops >= 32). Variable-length payloads are a fixed prefix struct followed by
// an array of elements (see TrailingArray); the record's `size` covers prefix + array, padded to
// 8 bytes. Every value is a raw NVN value (enum values, bit-packed state objects in nxnative's
// layouts) unless noted; buffer addresses are GPU addresses (= guest VA of the pool storage plus
// the offset), texture ids/handles are indexes into the submission's pool snapshots.
// ---------------------------------------------------------------------------------------------

/// Record flag: the payload's handle/range array is re-read from guest memory at submit. Only the
/// ops BindUniformBuffers..BindImages (see IsDeferrableOp) use bit 0 of RecordHeader::flags this
/// way; for every other op `flags` is op-specific (ClearTexture: the NVN channel write mask,
/// bit0 R .. bit3 A) and is passed through untouched.
inline constexpr u16 RECORD_FLAG_DEFERRED = 1;

enum class StateKind : u32 { Blend = 0, Color = 1, ChannelMask = 2, Multisample = 3, Polygon = 4, DepthStencil = 5 };
/// Bytes of each state object's packed in-place layout (nxnative blend_st, color_st, chmask_st,
/// ms_st, poly_st, ds_st), indexed by StateKind.
inline constexpr u32 STATE_OBJECT_SIZES[6] = {8, 4, 4, 10, 4, 8};

struct BufferRange {
    u64 address;
    u64 size;
};
struct ViewportRect {
    f32 x, y, width, height;
};
struct ScissorRect {
    s32 x, y, width, height;
};
struct DepthRange {
    f32 near_plane, far_plane;
};
struct VertexStreamState {
    u32 stride;
    u32 divisor;
};

/// A render target with its (possibly default) view. texture.level/layer = view base level/layer.
struct RenderTargetRef {
    TextureRef texture;
    TextureViewDesc view;
};
/// Prefix; followed by RenderTargetRef[color_count + has_depth] (the depth target last). A color
/// target with serial 0 is an empty slot.
struct SetRenderTargetsPayload {
    u32 color_count;
    u32 has_depth;
};
struct ClearColorPayload {
    s32 index;
    u32 mask; // channel write mask, bit0 R .. bit3 A
    u32 is_integer;
    u32 pad;
    u32 color[4]; // float bit patterns, or int/uint values when is_integer
};
struct ClearDepthStencilPayload {
    f32 depth;
    u32 depth_mask; // 0/1
    s32 stencil;
    u32 stencil_mask;
};
/// Prefix; followed by ViewportRect[count].
struct SetViewportsPayload {
    u32 first;
    u32 count;
};
/// Prefix; followed by ScissorRect[count].
struct SetScissorsPayload {
    u32 first;
    u32 count;
};
/// Prefix; followed by DepthRange[count].
struct SetDepthRangesPayload {
    u32 first;
    u32 count;
};
struct BindProgramPayload {
    u64 program_serial;
    u32 stages; // NVNshaderStageBits mask
    u32 pad;
};
/// Prefix; followed by `size` bytes (STATE_OBJECT_SIZES[kind]) of the packed state object.
struct BindStatePayload {
    u32 kind; // StateKind
    u32 size;
};
/// Prefix; followed by u32[count]: nxnative va_st bit-fields {fmt:8, offset:19, stream:5}.
struct BindVertexAttribsPayload {
    u32 first;
    u32 count;
};
/// Prefix; followed by VertexStreamState[count].
struct BindVertexStreamsPayload {
    u32 first;
    u32 count;
};
/// Prefix; followed by BufferRange[count].
struct BindVertexBuffersPayload {
    u32 first;
    u32 count;
};
/// BindUniformBuffers / BindStorageBuffers. Prefix; followed by BufferRange[count]. With
/// RECORD_FLAG_DEFERRED the ranges are re-read from guest VA `src` (count * 16 bytes) at submit.
struct BindBuffersPayload {
    u32 stage; // NVNshaderStage
    u32 first;
    u32 count;
    u32 pad;
    u64 src;
};
/// BindTextures / BindSeparateTextures / BindSeparateSamplers / BindImages. Prefix; followed by
/// u64 handles[count], already rebased by the command buffer's pool bases:
///   BindTextures: texture id | sampler id << 20; BindSeparateTextures: texture id;
///   BindSeparateSamplers: sampler id << 20; BindImages: texture id.
/// Ids of 0 stay 0. With RECORD_FLAG_DEFERRED the handles are re-read from guest VA `src`
/// (count * 8 bytes) at submit and rebased with tex_base/samp_base.
struct BindHandlesPayload {
    u32 stage;
    u32 first; // slot of the first handle
    u32 count;
    u32 pad;
    u64 src;
    u32 tex_base;
    u32 samp_base;
};
struct SetStencilPayload {
    s32 face; // NVNface
    u32 which; // 0 value mask, 1 write mask, 2 reference
    s32 value;
    u32 pad;
};
struct SetPolygonOffsetPayload {
    f32 factor, units, clamp;
    u32 pad;
};
/// Plain, indirect and multi-draw-indirect-count draws (arrays). `indirect != 0`: the arguments
/// are read by the GPU at that address. `count_address != 0`: multi-draw with the u32 draw count at
/// count_address, at most `max_draws` draws of `stride` bytes each; otherwise `max_draws` is 1.
struct DrawPayload {
    u32 mode; // NVNdrawPrimitive
    s32 first;
    u32 count;
    u32 instance_count; // 0 counts as 1
    u32 base_instance;
    u32 max_draws;
    u64 indirect;
    u64 count_address;
    u32 stride;
    u32 pad;
};
struct DrawIndexedPayload {
    u32 mode;
    u32 index_type; // NVNindexType
    /// Indices of a direct draw. Indirect draw: the bytes from index_address to the end of the
    /// buffer object that holds it (saturated; 0 = unknown), which bound the indices it can reach.
    u32 count;
    s32 base_vertex;
    u32 base_instance;
    u32 instance_count;
    u64 index_address;
    u64 indirect;
    u64 count_address;
    u32 max_draws;
    u32 stride;
};
struct DispatchPayload {
    u32 x, y, z;
    u32 pad;
    u64 indirect; // != 0: group counts read by the GPU at this address
    u64 caller;   // return address of the game call site (diagnostics), 0 if unknown
};
struct BarrierPayload {
    u32 bits;
    u32 pad;
};
/// SetTexturePool / SetSamplerPool.
struct SetPoolPayload {
    u64 pool_serial;
    u32 base; // base of this pool in the global id space
    u32 pad;
};
struct SetBlendColorPayload {
    f32 color[4];
};
struct EnablePayload {
    u32 enable; // SetDepthClamp, SetRasterizerDiscard
    u32 pad;
};
struct SetAlphaRefPayload {
    f32 reference;
    u32 pad;
};
/// NVN2 push constants. Prefix; followed by `size` bytes of data.
struct PushConstantsPayload {
    u32 compute; // 1 = compute block, 0 = graphics block
    u32 offset;
    u32 size;
    u32 pad;
    u64 word; // the word the game passes before the offset (meaning unknown)
};
/// Prefix; followed by `size` bytes copied at record time (the game may reuse its data).
struct UpdateUniformBufferPayload {
    u64 address;
    u64 size;
};
struct ReportValuePayload {
    u64 address;
    u32 value;
    u32 pad;
};
struct ReportCounterPayload {
    u64 address;
    u32 counter_type; // NVNcounterType
    u32 pad;
};
struct ResetCounterPayload {
    u32 counter_type;
    u32 pad;
};
struct SetRenderConditionPayload {
    u32 kind; // 0 = conditional on the counters at `address`, 1 = SetRenderEnable(true), 2 = (false)
    u32 mode; // NVNconditionalRenderMode when kind == 0
    u64 address;
};
struct DrawTexturePayload {
    u64 handle; // texture id | sampler id << 20, rebased
    f32 dst[4]; // x0, y0, x1, y1
    f32 src[4];
};
struct SetPrimitiveRestartPayload {
    u32 enable;
    u32 index;
};
/// Prefix; followed by u8[4] (NVNviewportSwizzle of x, y, z, w) per viewport.
struct SetViewportSwizzlesPayload {
    u32 first;
    u32 count;
};
struct BindTransformFeedbackBufferPayload {
    u32 index;
    u32 pad;
    u64 address;
    u64 size;
};
struct TransformFeedbackStatePayload {
    u32 what; // 0 begin, 1 end, 2 pause, 3 resume
    u32 pad;
    u64 address; // the game's control buffer
};
struct DrawTransformFeedbackPayload {
    u32 mode;
    u32 pad;
    u64 address;
};
/// Every nvnCommandBufferDrawMeshTasks* entry (task/mesh pipelines). The driver (3.2.0
/// sdk+0x681470..0x681cb0) hands each to an MME macro: direct 1D (first, count), direct 3D (x, y, z),
/// or argument words the GPU fetches at `indirect`: per draw {count, first} (8 bytes) for the 1D
/// entries, {x, y, z} (12 bytes) for the 3D ones. The multi entries fetch `max_draws` records
/// `stride` bytes apart; the *IndirectCount ones also one u32 draw count at `count_address`
/// (the draws made are the lesser of the two). On the GPU (the macros, replayed from OoT's macro
/// code) every draw launches the task range [first, first + count): a 3D draw launches x * y * z
/// tasks from 0 with x, y, z written to constant bank offsets 0x8e0/0x8e4/0x8e8, and each draw of
/// a multi entry writes its draw index to offset 0x648 first ([UNVERIFIED] the bank is the driver
/// bank: its address comes from the macros' data RAM).
struct DrawMeshTasksPayload {
    u32 dims;          // 1: DrawMeshTasks, DrawMeshTasksIndirect, MultiDrawMeshTasksIndirect[Count]; 3: *3D*
    u32 first;         // direct 1D: the first task; otherwise 0
    u32 x, y, z;       // direct: task workgroup counts (1D: x = count, y = z = 1); 0 when indirect
    u32 max_draws;     // multi: the draw count (non-Count) or the maximum (Count); otherwise 1
    u64 indirect;      // != 0: GPU address of the arguments
    u64 count_address; // != 0: *IndirectCount entries' u32 draw count
    u32 stride;        // multi: bytes between argument records (the SDK default when <= 0); else 0
    u32 pad;
};
struct U32Payload {
    u32 value; // SetSampleMask, SetProvokingVertex
    u32 pad;
};
struct F32Payload {
    f32 value; // SetPointSize, SetLineWidth
    u32 pad;
};

/// A rectangle's origin in texels of its image.
struct DlssRect {
    s32 x, y;
};

/// NVNdlssNetworkBuilder flag bits as the SDK forwards them (Initialize, OoT sdk+0x659e40): the
/// backend's create flags are (flags & DLSS_FLAGS_TO_NGX) | (flags & DLSS_FLAG_LDR ? 0 : 1), so bit
/// 0 is NGX's IsHDR inverted. Bits 0x10, 0x20 (sharpening), 0x80 and the rest never reach the
/// backend.
inline constexpr u32 DLSS_FLAG_LDR = 0x1; // clear = HDR colour (NGX IsHDR inverted)
inline constexpr u32 DLSS_FLAG_MV_LOW_RES = 0x2;
inline constexpr u32 DLSS_FLAG_MV_JITTERED = 0x4;
inline constexpr u32 DLSS_FLAG_DEPTH_INVERTED = 0x8;
inline constexpr u32 DLSS_FLAG_AUTO_EXPOSURE = 0x40;
/// NGX EnableOutputSubrects: EvaluateDlss passes the output origin only with this bit.
inline constexpr u32 DLSS_FLAG_OUTPUT_SUBRECTS = 1u << 28;
/// Only enlarges GetStorageSize.
inline constexpr u32 DLSS_FLAG_EXTRA_STORAGE = 1u << 29;
inline constexpr u32 DLSS_FLAGS_TO_NGX = 0x4e;

/// nvnCommandBufferEvaluateDlss: the network's settings and the game's NVNdlssParameters, read at
/// record time (the game reuses both objects between evaluations), with the rules the SDK applies
/// before it calls the backend (OoT sdk+0x65a4e0):
/// - `exposure` is set only when (flags & (DLSS_FLAG_LDR | DLSS_FLAG_AUTO_EXPOSURE)) == 0 (HDR
///   colour without automatic exposure); `pre_exposure` is the game's then, else 1.0f (the SDK
///   passes 0, "none"; 1.0 is Streamline's "none").
/// - `output_base` is the game's only with DLSS_FLAG_OUTPUT_SUBRECTS, else {0, 0}.
/// - `input_width/height` is the game's input size, or the input texture's level 0 size when the
///   game left it 0x0 (OoT sdk+0x65a9a0); not validated here (the replay judges the rectangle).
/// - `output_width/height` is the network's current output size (Initialize or Resize).
/// - `flags` is the raw builder flags word (DLSS_FLAG_*), `method` NGX's perf quality value (0
///   MaxPerf, 1 Balanced, 2 MaxQuality, 3 UltraPerformance, 4 UltraQuality, 5 DLAA), `preset` NGX's
///   render preset hint (0 default, 1 = A ... 13 = M).
/// - `network` identifies the network's backend feature: nonzero, unique per Initialize.
/// The images are the input colour, the output colour, the motion vectors, the depth, the
/// bias-current-colour mask and the exposure texture; every rectangle of the input side has the
/// one size (input_width, input_height). `motion_scale` multiplies the game's motion vectors to give
/// input pixels (NGX's MV scale). The SDK's evaluation index never reaches the backend (only its
/// devtools trace) and is not recorded.
struct EvaluateDlssPayload {
    RenderTargetRef input;
    RenderTargetRef output;
    RenderTargetRef motion_vectors;
    RenderTargetRef depth;
    RenderTargetRef bias;
    RenderTargetRef exposure;
    DlssRect input_base;
    DlssRect depth_base;
    DlssRect motion_vector_base;
    DlssRect bias_base;
    DlssRect output_base;
    s32 input_width;
    s32 input_height;
    s32 output_width;
    s32 output_height;
    f32 motion_scale[2];
    f32 jitter[2];
    f32 pre_exposure;
    u32 history_reset;
    u32 flags;
    u32 method;
    u32 preset;
    u32 network;
};

static_assert(sizeof(RenderTargetRef) == 40);
static_assert(sizeof(ClearColorPayload) == 32);
static_assert(sizeof(BindBuffersPayload) == 24);
static_assert(sizeof(BindHandlesPayload) == 32);
static_assert(sizeof(DrawPayload) == 48);
static_assert(sizeof(DrawIndexedPayload) == 56);
static_assert(sizeof(DispatchPayload) == 32);
static_assert(sizeof(PushConstantsPayload) == 24);
static_assert(sizeof(DrawTexturePayload) == 40);
static_assert(sizeof(EvaluateDlssPayload) == 336);
static_assert(sizeof(DrawMeshTasksPayload) == 48);

/// Payload of a record (the bytes after the header).
template <typename T>
[[nodiscard]] inline const T* RecordPayload(const RecordHeader* header) {
    return reinterpret_cast<const T*>(header + 1);
}

/// The array following a prefix struct.
template <typename Element, typename Prefix>
[[nodiscard]] inline const Element* TrailingArray(const Prefix* prefix) {
    return reinterpret_cast<const Element*>(prefix + 1);
}

/// Iterates the records of a list (a std::vector<u64>). Stops at End or on a malformed size.
class RecordReader {
public:
    explicit RecordReader(const u64* words, size_t count) : cursor{words}, end{words + count} {}

    /// Returns the next record or nullptr at End / end of list / malformed record.
    [[nodiscard]] const RecordHeader* Next() {
        if (cursor >= end) {
            return nullptr;
        }
        const auto* header = reinterpret_cast<const RecordHeader*>(cursor);
        if (header->op == static_cast<u16>(Op::End) || header->size < sizeof(RecordHeader) ||
            (header->size & 7) != 0 || cursor + header->size / 8 > end) {
            return nullptr;
        }
        cursor += header->size / 8;
        return header;
    }

    /// The next word to read (the record Next() returns next, or the list's End).
    [[nodiscard]] const u64* Cursor() const {
        return cursor;
    }

private:
    const u64* cursor;
    const u64* end;
};

[[nodiscard]] inline Op RecordOp(const RecordHeader* header) {
    return static_cast<Op>(header->op);
}

/// Ops whose payload arrays can be deferred (RECORD_FLAG_DEFERRED): BindUniformBuffers,
/// BindStorageBuffers, BindTextures, BindSeparateTextures, BindSeparateSamplers, BindImages.
[[nodiscard]] inline bool IsDeferrableOp(Op op) {
    return op >= Op::BindUniformBuffers && op <= Op::BindImages;
}

/// Number of payload bytes of a record.
[[nodiscard]] inline u32 RecordPayloadSize(const RecordHeader* header) {
    return header->size - static_cast<u32>(sizeof(RecordHeader));
}

} // namespace VideoCommon::Nvn
