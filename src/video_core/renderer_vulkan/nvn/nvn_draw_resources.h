// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Context-wide resources of the draw path: the pipeline cache, samplers, image views, the
// per-submission buffer arena, dummy resources for unbound descriptors, program hashes.
// Replay thread only, except the arena free list (the completion thread returns chunks).

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <chrono>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

#include "video_core/renderer_vulkan/nvn/nvn_pass_state.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"

namespace Vulkan::Nvn {

/// A persistently mapped host buffer that submissions carve slices from.
struct ArenaChunk {
    vk::Buffer buffer;
    u8* pointer{};
    u64 size{};
    u64 used{};
};

/// A device-local buffer that submissions carve the shadows of written storage ranges from
/// (nvn_draw_shadow.cpp). Returned to the free list when the submission completed.
struct ShadowChunk {
    vk::Buffer buffer;
    u64 size{};
    u64 used{};
};

/// What an image view is created for.
struct ViewKey {
    VkFormat format{};
    VkImageViewType type{};
    VkImageAspectFlags aspect{};
    u32 base_level{};
    u32 levels{};
    u32 base_layer{};
    u32 layers{};
    u32 swizzle{}; ///< 4 x VkComponentSwizzle (r,g,b,a), 4 bits each
    bool operator==(const ViewKey&) const = default;
};

/// The views of one Texture object. Holds the storage so the image outlives the views.
struct TextureViews {
    std::shared_ptr<Storage> storage;
    std::vector<std::pair<ViewKey, vk::ImageView>> views;
};

struct DummyImage {
    vk::Image image;
    vk::ImageView view;
};

struct ClearPipelineKey {
    VkRenderPass render_pass{};
    VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
    u32 color_mask{};
    bool stencil{};
    u32 stencil_mask{};
    u32 stencil_reference{};
    bool operator==(const ClearPipelineKey&) const = default;
};

/// The legacy host clear shaders as pipelines for channel-masked and partial-stencil clears.
struct ClearPipelines {
    explicit ClearPipelines(const Device& device);
    VkPipeline Get(const Device& device, const ClearPipelineKey& key);

    vk::ShaderModule vert;
    vk::ShaderModule color_frag;
    vk::ShaderModule stencil_frag;
    vk::PipelineLayout layout;
    std::vector<std::pair<ClearPipelineKey, vk::Pipeline>> pipelines;
};

/// Why a draw bound a dummy resource instead of what the game bound.
enum class DescReason : u32 {
    TextureNoHandle,     ///< the handle word names no registered texture (id 0 or empty pool slot)
    TextureNoStorage,    ///< the texture has no image (creation failed)
    TextureIncompatible, ///< the texture's type cannot be bound as the shader's type
    TextureNoView,       ///< the image view could not be made
    TexelBuffer,         ///< texel buffer descriptors
    UniformUnbound,      ///< no range bound to the uniform buffer slot (address 0)
    UniformUnresolved,   ///< the bound range is in no pool
    StorageUnbound,      ///< no range bound to the storage buffer slot (address 0)
    StorageUnresolved,
    StorageZeroSize,     ///< a storage buffer bound with an address and size 0 (bound as the rest of its pool)
    VertexUnresolved,    ///< the vertex buffer range of a used stream is in no pool
    UniformZeroSize,     ///< a uniform buffer bound with an address and size 0
    Count,
};

/// False for the bindings the hardware itself leaves empty or unsized, which are not faults of
/// ours: an unbound or size-0 uniform buffer reads zero (constant-buffer reads are bounds
/// checked, nxnative reads its all-zero bank), a storage buffer bound with size 0 is the rest of
/// its pool (loads and stores use the address), and a storage slot nothing was bound to is only
/// reached by shader code the game never runs with it (a store through it would fault on the
/// GPU; `DrawContext` watches the sink it binds instead and warns when something is written).
[[nodiscard]] constexpr bool DescReasonIsFallback(DescReason reason) {
    switch (reason) {
    case DescReason::UniformUnbound:
    case DescReason::UniformZeroSize:
    case DescReason::StorageUnbound:
    case DescReason::StorageZeroSize:
        return false;
    default:
        return true;
    }
}

[[nodiscard]] constexpr const char* DescReasonName(DescReason reason) {
    switch (reason) {
    case DescReason::TextureNoHandle:
        return "texture handle names no texture";
    case DescReason::TextureNoStorage:
        return "texture has no image";
    case DescReason::TextureIncompatible:
        return "texture type incompatible with the shader";
    case DescReason::TextureNoView:
        return "image view not created";
    case DescReason::TexelBuffer:
        return "texel buffer descriptor";
    case DescReason::UniformUnbound:
        return "uniform buffer slot unbound";
    case DescReason::UniformUnresolved:
        return "uniform buffer range in no pool";
    case DescReason::StorageUnbound:
        return "storage buffer slot unbound";
    case DescReason::StorageUnresolved:
        return "storage buffer range in no pool";
    case DescReason::StorageZeroSize:
        return "storage buffer bound with an address and size 0 (bound as the rest of its pool)";
    case DescReason::UniformZeroSize:
        return "uniform buffer bound with an address and size 0 (reads zero)";
    case DescReason::VertexUnresolved:
        return "vertex buffer range in no pool";
    default:
        return "unknown";
    }
}

/// Why a render pass ended while it was open (`DrawContext::EndPass`).
enum class BreakCause : u32 {
    SetRenderTargets, ///< the game bound other targets
    Barrier,          ///< a game barrier op after shader writes of the pass
    Dispatch,         ///< a compute dispatch (outside a pass)
    ReportCounter,    ///< a (non-timestamp) counter report
    ReportValue,      ///< a report value write
    ResetCounter,     ///< a counter reset
    RenderCondition,  ///< a render condition (its predicate is computed outside a pass)
    OtherOp,          ///< a copy, clear or other op recorded outside a pass
    FeedbackCopy,     ///< a sampled render target is copied before the draw
    ContentsUpload,   ///< a sampled storage loads its first contents
    LayoutChange,     ///< a sampled image is not in the layout the draw needs
    MirrorUpload,     ///< a mirror pool range is copied in before a draw
    IndirectUpload,   ///< indirect arguments / counts of a mirror pool are copied in
    PassBufferWrites, ///< an indirect draw reads a range an earlier draw of the pass wrote
    MaskedClear,      ///< a channel-masked clear draws its own pass
    ShadowAlias,      ///< a storage range shadowed in device memory is used another way (host copy stale)
    ReportRead,       ///< a draw uses memory a counter report of the pass writes (in-pass queries)
    MeshIndirectConvert, ///< 1D mesh-task arguments are rewritten as VkDrawMeshTasksIndirectCommandEXT
    InlineUpdate,     ///< pending inline uniform updates flushed to GPU pool memory
    SegmentEnd,       ///< the command buffer ends
    Other,
    Count,
};

[[nodiscard]] constexpr const char* BreakCauseName(BreakCause cause) {
    switch (cause) {
    case BreakCause::SetRenderTargets:
        return "SetRenderTargets";
    case BreakCause::Barrier:
        return "Barrier";
    case BreakCause::Dispatch:
        return "Dispatch";
    case BreakCause::ReportCounter:
        return "ReportCounter";
    case BreakCause::ReportValue:
        return "ReportValue";
    case BreakCause::ResetCounter:
        return "ResetCounter";
    case BreakCause::RenderCondition:
        return "render condition";
    case BreakCause::OtherOp:
        return "copy/clear/other op";
    case BreakCause::FeedbackCopy:
        return "feedback copy";
    case BreakCause::ContentsUpload:
        return "sampled storage contents upload";
    case BreakCause::LayoutChange:
        return "sampled image layout transition";
    case BreakCause::MirrorUpload:
        return "mirror upload";
    case BreakCause::IndirectUpload:
        return "indirect mirror upload";
    case BreakCause::PassBufferWrites:
        return "indirect after pass buffer writes";
    case BreakCause::MaskedClear:
        return "masked clear";
    case BreakCause::SegmentEnd:
        return "segment end";
    case BreakCause::ShadowAlias:
        return "storage shadow alias";
    case BreakCause::ReportRead:
        return "draw reads a counter report of its pass";
    case BreakCause::MeshIndirectConvert:
        return "mesh indirect argument conversion";
    case BreakCause::InlineUpdate:
        return "inline uniform update flush";
    default:
        return "other";
    }
}

/// Vertex staging (nvn_vertex_stage.h): whether the device-local copies of vertex ranges stay
/// current past the end of a pass, for later passes of the same command buffer. The pass must not
/// have written through shaders nor have counter reports to write at its end, and the break must
/// write images or nothing, never pool memory: a target change, a feedback copy (render target to
/// scratch image), an image layout change, a masked clear (draws into the targets) or the first
/// contents of a sampled storage (pool to image). Every other cause, and every op recorded outside
/// a pass, drops the copies; pool writes recorded meanwhile drop the copies they overlap
/// (DrawContext::OnPoolWrite). `mesh_convert_survives` (EDEN_NX2_TRY_STAGE_SPAN): the argument
/// conversion of a mesh indirect draw reads pool memory and writes only its own arena, so that
/// break leaves the copies current too.
[[nodiscard]] constexpr bool StagedVertexSurvives(BreakCause cause, bool pass_shader_writes,
                                                  bool reports_pending,
                                                  bool mesh_convert_survives = false) {
    if (pass_shader_writes || reports_pending) {
        return false;
    }
    switch (cause) {
    case BreakCause::MeshIndirectConvert:
        return mesh_convert_survives;
    case BreakCause::SetRenderTargets:
    case BreakCause::FeedbackCopy:
    case BreakCause::LayoutChange:
    case BreakCause::MaskedClear:
    case BreakCause::ContentsUpload:
        return true;
    default:
        return false;
    }
}

struct DrawCounters {
    std::atomic<u64> draws{};
    std::atomic<u64> draws_skipped{};
    /// Draws skipped because their pipeline compiles on a worker (use_asynchronous_shaders).
    std::atomic<u64> draws_skipped_async{};
    std::atomic<u64> pass_begins{};
    /// Sampled bindings of a colour image in GENERAL, sampled there with no layout transition.
    std::atomic<u64> sampled_in_general{};
    /// Recorded state a draw skipped because the open pass had it already (nvn_pass_state.h).
    /// Feedback copies by what the sampled view covers: a subresource a target of the draw
    /// attaches (a real loop) or only other levels/layers of the same image.
    std::atomic<u64> feedback_overlapping{};
    std::atomic<u64> feedback_disjoint{};
    /// Feedback copies not recorded because the scratch already holds the source since the last
    /// target change, barrier or other op (nothing the hardware orders can have changed it).
    std::atomic<u64> feedback_skipped{};
    std::atomic<u64> elided_push_constants{};
    std::atomic<u64> elided_descriptors{};
    std::atomic<u64> elided_vertex_binds{};
    std::atomic<u64> elided_index_binds{};
    std::atomic<u64> driver_bank_uploads{};
    std::atomic<u64> handle_bank_uploads{};
    std::atomic<u64> pipeline_state_reuses{};
    /// Vertex staging (nvn_vertex_stage.h): passes that staged, ranges staged, copy regions and
    /// bytes copied (gaps between merged ranges included), ranges and bytes a pass found staged by
    /// an earlier pass of the command buffer, vertex buffer and index buffer binds served from a
    /// staged copy.
    std::atomic<u64> vertex_stage_passes{};
    std::atomic<u64> vertex_stage_ranges{};
    std::atomic<u64> vertex_stage_regions{};
    std::atomic<u64> vertex_stage_bytes{};
    std::atomic<u64> vertex_stage_reused_ranges{};
    std::atomic<u64> vertex_stage_reused_bytes{};
    std::atomic<u64> vertex_stage_binds{};
    std::atomic<u64> index_stage_binds{};
    /// Device-local shadow chunks created (a device memory allocation each) and destroyed because
    /// the free list was full or they were too large to keep.
    std::atomic<u64> shadow_chunks_created{};
    std::atomic<u64> shadow_chunks_destroyed{};
    std::atomic<u64> pass_breaks{};
    std::atomic<u64> descriptors_unsupported{};
    /// Passes that ended while open, by cause (index = BreakCause).
    std::array<std::atomic<u64>, static_cast<size_t>(BreakCause::Count)> break_causes{};
    std::atomic<u64> inline_updates{};
    /// Inline updates a later one of the same pass flush rewrote whole: no copy recorded.
    std::atomic<u64> inline_updates_superseded{};
    /// Transfer barriers between inline update copies to overlapping bytes.
    std::atomic<u64> inline_update_barriers{};
    /// Flushes of inline updates that recorded no copy (all superseded or unresolved): the barrier
    /// in front of the copies and the one after them were left out (EDEN_NX2_TRY_BARRIER_MERGE).
    std::atomic<u64> inline_flushes_without_copy{};
    /// EDEN_NX2_NVN_UBO_SNAPSHOT: uniform bindings served from a submit-time copy, the bytes
    /// uploaded for them, and inline updates flushed into a submission that has a copy
    /// (nvn_uniform_snapshot.h).
    std::atomic<u64> uniform_snapshot_binds{};
    std::atomic<u64> uniform_snapshot_upload_bytes{};
    std::atomic<u64> uniform_snapshot_layers{};
    /// Draws that sampled one of their own render targets: the target was copied first.
    std::atomic<u64> feedback_copies{};
    /// Such draws that could not be handled (multisampled target, no scratch image).
    std::atomic<u64> feedback_unsupported{};
    std::atomic<u64> placeholder_copies{}; ///< 1x1x1 2D placeholders copied to a 3D/cube scratch
    std::atomic<u64> feedback_depth{}; ///< feedback copies of a depth/stencil target
    std::atomic<u64> feedback_color{}; ///< feedback copies of a color target
    std::atomic<u64> barriers_kept{}; ///< game barriers that left the pass open (no shader writes)
    /// Game barriers outside a pass that order no compute work (no ORDER_PRIMITIVES /
    /// ORDER_INDIRECT_DATA) and so emitted nothing (EDEN_NX2_TRY_DISPATCH_ORDER).
    std::atomic<u64> barriers_without_wait{};
    /// Storages moved to their sampled layout when a pass began (instead of a break at the draw).
    std::atomic<u64> flip_transitions{};
    /// Flip targets learned while the list was full (the entry used longest ago made room).
    std::atomic<u64> flip_targets_replaced{};
    /// Feedback copies of a target already copied since the last SetRenderTargets.
    std::atomic<u64> feedback_repeat{};
    std::atomic<u64> render_targets_ops{};
    std::atomic<u64> render_targets_same{}; ///< SetRenderTargets that bound what was bound
    /// Where a draw bound a dummy instead of what the game bound, by reason (index = DescReason)
    /// and bind point ([0] graphics, [1] compute).
    std::array<std::array<std::atomic<u64>, 16>, 2> fallbacks{};
    std::atomic<u64> dispatches{};
    std::atomic<u64> dispatches_indirect{};
    std::atomic<u64> dispatches_skipped{};
    /// Presents that had at least one async-skipped draw in the frame before.
    std::atomic<u64> frames_with_async_skips{};
    std::atomic<u64> dispatch_groups{};
    /// Storage shadows (nvn_draw_shadow.cpp): written storage ranges of imported pools bound through
    /// a device-local copy. Shadows created / bindings redirected to a live shadow / shadows copied
    /// back (by cause) / bytes copied in and out.
    std::atomic<u64> shadow_creates{};
    std::atomic<u64> shadow_redirects{};
    std::atomic<u64> shadow_flush_segment{};  ///< copied back at the end of the command buffer
    std::atomic<u64> shadow_flush_conflict{}; ///< copied back because another use touched the range
    std::atomic<u64> shadow_flush_raw{};      ///< copied back before a draw with raw global memory
    std::atomic<u64> shadow_bytes_in{};
    std::atomic<u64> shadow_bytes_out{};
    /// Written storage ranges that stayed in host memory: another use overlaps / pass already open /
    /// no device memory.
    std::atomic<u64> shadow_skipped_alias{};
    std::atomic<u64> shadow_skipped_open_pass{};
    std::atomic<u64> shadow_skipped_no_room{};
};

struct DrawResources {
    explicit DrawResources(Context::Impl& impl);
    ~DrawResources();

    Context::Impl& impl;
    /// Created by OpenCache (title known: disk cache on) or on first use (no disk cache).
    std::unique_ptr<PipelineCache> cache_instance;
    [[nodiscard]] PipelineCache& Cache();
    void OpenCache(u64 title_id);
    /// Thread-safe (the stats thread calls it): rows of the cache if it exists.
    void CollectCacheRows(std::vector<PipelineCache::Row>& rows);
    std::mutex cache_mutex;
    bool shader_refresh{};
    std::chrono::steady_clock::time_point last_shader_refresh{};
    /// Called at present boundaries; existing flights keep their pipelines until GPU completion.
    void MaybeRefreshShaders();
    ClearPipelines clears;
    DrawCounters counters;
    /// Called once per present (replay thread): logs the draws skipped awaiting an asynchronous
    /// pipeline since the previous present and counts the frames that had any.
    void NoteAsyncSkips(u64 present);
    u64 noted_draw_skips{};
    u64 async_skip_lines{};

    /// EDEN_NX2_NVN_TSTRACE (replay thread only): the render pass sizes and compute dispatches of
    /// the present being recorded, logged per present by `TracePresent`.
    struct FrameTrace {
        std::map<std::pair<u32, u32>, u32> passes; ///< framebuffer extent -> render pass begins
        u32 dispatches{};
        u32 dispatches_128{}; ///< dispatches of programs with 128 threads in x (the tensor upscaler)
        std::pair<u32, u32> last_main{};
        std::chrono::steady_clock::time_point last_present{};
    } frame_trace;
    void TracePass(u32 width, u32 height);
    void TraceDispatch(bool tensor_like);
    void TracePresent(u64 present);

    // --- programs ---
    /// Content hash per stage of every program (0 = stage absent), by program serial.
    std::unordered_map<u64, std::array<u64, STAGE_COUNT>> program_hashes;
    /// Records the stage hashes of a program the game created; on the GPL path the cache also
    /// lowers its stages now (PipelineCache::PrepareProgram), ahead of its first draw.
    void OnProgramCreated(const std::shared_ptr<const VC::ProgramDesc>& program);
    /// Content hashes of the compute programs dispatched so far (logged once each).
    std::unordered_set<u64> compute_programs;

    // --- samplers ---
    /// The VkSamplers built for each sampler object `serial`: the guest's own (steps 0) and, when
    /// the Anisotropic Filtering setting changes its anisotropy, one per setting value in use
    /// (steps = the doublings it adds; see nvn_sampler_aniso.h).
    struct SamplerVariant {
        s32 steps;
        vk::Sampler sampler;
    };
    std::unordered_map<u64, std::vector<SamplerVariant>> samplers;
    vk::Sampler fallback_sampler;
    /// The doublings the Anisotropic Filtering setting adds to samplers now (RefreshSamplerSetting).
    s32 anisotropy_steps{0};
    /// The device's maxSamplerAnisotropy (the samplerAnisotropy feature is mandatory).
    f32 max_sampler_anisotropy{16.0f};
    /// Reads the setting once per submission; a change retires the descriptor-slot samplers
    /// (samplers_epoch) so the next draw asks for the new variant.
    void RefreshSamplerSetting();
    /// VkSampler of the sampler object `serial` (a fallback sampler for unknown ones). `boosted`:
    /// the sampler for a view the setting's anisotropy applies to (ViewSupportsAnisotropy); false
    /// gives the guest's own sampler.
    VkSampler GetSampler(u64 serial, bool boosted);
    /// EDEN_NX2_PATHDIFF: the host sampler state of a sampler descriptor (nullopt: unknown).
    [[nodiscard]] std::optional<VideoCommon::PathDiff::Sampler> DescribeSampler(u64 serial) const;
    void ReleaseSampler(u64 serial, u64 after_seq);

    // --- views ---
    std::unordered_map<u64, TextureViews> texture_views;
    u64 views_epoch{1}; ///< Bumped when a texture_views node is erased (Texture::view_slot dies).
    u64 samplers_epoch{1}; ///< Bumped when a sampler is released (DrawContext::sampler_slots).
    /// Image view of a texture object for `key`, created on first use. Null on failure.
    VkImageView GetView(const Texture& texture, const ViewKey& key);
    /// Detaches the views of a destroyed texture so they can be released after its last use.
    std::shared_ptr<TextureViews> TakeViews(u64 serial);
    /// Detaches the views of every texture over `storage`, whose image is being replaced, so they
    /// can be released after its last use (the textures make new ones on the new image).
    std::vector<vk::ImageView> TakeStorageViews(const Storage& storage);

    // --- feedback copies (a texture sampled while it is a render target of the draw) ---
    struct Scratch {
        std::shared_ptr<Storage> storage;
        std::shared_ptr<Texture> texture; ///< view source (serial from a private range)
        u32 slot{};
        /// The last feedback copy into this scratch (DrawContext::PruneFreshFeedback): the source,
        /// the context and its `render_target_epoch` at the time, and the levels/layers copied.
        FeedbackSnapshot<Storage> copied;
    };
    std::vector<Scratch> scratch_images;
    u64 next_scratch_serial{0x8000'0000'0000'0000ull};
    /// A scratch image of the shape of `source`; `slot` tells apart the scratches of one draw.
    /// Created lazily, kept for the context's lifetime. Null when it cannot be created.
    Scratch* GetScratch(const Storage& source, u32 slot);
    /// The scratch whose image is `storage` (null when it is not one).
    Scratch* FindScratch(const Storage& storage);
    /// The type a shader samples a 2D texture as (GetRetypedScratch).
    enum class RetypeTarget { Volume, Cube, Row };
    /// A scratch image of `source`'s format and the type a shader samples it as: a one-slice 3D
    /// image (Volume), a cube-compatible 2D image of six faces (Cube) or a 1D image of its width
    /// (Row). `slot` tells apart the scratches of one draw. Created lazily and kept. Null on
    /// failure.
    Scratch* GetRetypedScratch(const Storage& source, u32 slot, RetypeTarget target);

    // --- arena ---
    std::mutex arena_mutex;
    std::vector<std::shared_ptr<ArenaChunk>> free_arenas;
    std::shared_ptr<ArenaChunk> AcquireArena(u64 min_size);
    void ReleaseArenas(std::vector<std::shared_ptr<ArenaChunk>>& chunks);

    // --- storage shadows ---
    std::mutex shadow_mutex;
    std::vector<std::shared_ptr<ShadowChunk>> free_shadows;
    /// A chunk of at least `min_size` bytes (null when device memory ran out).
    std::shared_ptr<ShadowChunk> AcquireShadow(u64 min_size);
    void ReleaseShadows(std::vector<std::shared_ptr<ShadowChunk>>& chunks);


    // --- dummies for unbound descriptors ---
    vk::Buffer dummy_buffer; ///< 64 KiB of zeros
    std::array<DummyImage, 7> dummy_images; ///< by VkImageViewType 1D,2D,3D,CUBE,1D_ARRAY,2D_ARRAY,CUBE_ARRAY
    /// A black 1x1 view of the image view type (SHADER_READ_ONLY_OPTIMAL).
    VkImageView DummyView(VkImageViewType type);
    VkBuffer DummyBuffer();

    // --- the sink of storage buffers nothing is bound to ---
    /// A shader that stores through a storage slot the game never bound would fault on the GPU, so
    /// the shaders DK runs only reference such slots in code they do not execute. The binding is
    /// the zeros of the dummy buffer for reads; a written binding gets its own slice of this sink
    /// per (program, stage, slot) instead, so nothing a shader stores can reach the dummy that
    /// unbound uniform buffers read zeros from, and `CheckSink` can report a store.
    vk::Buffer sink_buffer;
    /// Byte offset of the sink slice of (program hash, stage, slot); the first slice is shared by
    /// the bindings that do not get their own once the sink is full.
    [[nodiscard]] u64 SinkSlice(u64 program_hash, u32 stage, u32 slot);
    [[nodiscard]] VkBuffer SinkBuffer() const {
        return *sink_buffer;
    }
    /// Every SINK_CHECK_PRESENTS presents: logs the first nonzero word of each slice (once each).
    void CheckSink(u64 present);
    static constexpr u64 SINK_SLICE_SIZE = 4096;
    static constexpr u32 SINK_SLICES = 256;
    static constexpr u64 SINK_CHECK_PRESENTS = 120;
    struct SinkKey {
        u64 program_hash{};
        u32 stage{};
        u32 slot{};
        bool reported{false};
    };
    std::vector<SinkKey> sink_keys; ///< slice i + 1 belongs to sink_keys[i]
    std::map<std::tuple<u64, u32, u32>, u32> sink_index; ///< key -> index into sink_keys
    bool sink_overflow_reported{false};

    // --- per-pipeline program constant buffers (c1) ---
    std::unordered_map<const PipelineBase*, std::array<vk::Buffer, STAGE_COUNT>> constant_buffers;
    VkBuffer ProgramConstants(const PipelineBase& pipeline, u32 stage);

};

} // namespace Vulkan::Nvn
