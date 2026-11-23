// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// The graphics draw path of the NVN replay: tracked NVN state (render targets, state objects,
// viewports, vertex input, bindings), render pass management, clears, and the preparation of a
// draw (pipeline lookup, dynamic state, descriptors, vertex/index buffers).
//
// `DrawContext` is the per-queue state; nvn_exec.cpp's Recorder (one per submission) drives it
// through `DrawContext::Handle` and implements `DrawHost`, the services the draw path needs from
// the command recording.
//
// Other draw-like families (indirect draws, compute dispatches) reuse `PrepareDraw` /
// `BindDescriptors` and the bindings accessors; they must call `EndPass()` before recording
// anything that is not allowed inside a render pass.

#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <map>
#include <optional>
#include <memory>
#include <set>
#include <span>
#include <string_view>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "shader_recompiler/bindless_table.h"
#include "video_core/renderer_vulkan/nvn/nvn_barrier_track.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/nvn/record.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_plan.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_diag_textures.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_pass_state.h"
#include "video_core/renderer_vulkan/nvn/nvn_vertex_stage.h"
#include "video_core/upscaler.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"

namespace VideoCommon::NvnDrawProbe {
struct DumpTicket;
}

namespace Vulkan::Nvn {

inline constexpr u32 BIND_STAGES = 8;
inline constexpr u32 PUSH_CONSTANT_BLOCK_SIZE = 256;
inline constexpr u32 DRIVER_BANK_SIZE = 0x1000;
/// The `mode` PrepareDraw / ResolvePipeline receive for a mesh-task draw (DrawMeshTasks op): the
/// pipeline is built from the task, mesh and fragment programs and has no input assembly.
inline constexpr u32 MESH_DRAW_MODE = ~0u;

struct BufferBinding {
    u64 address{};
    u64 size{};
};

/// What the game has bound for one shader stage (0..5 are NVNshaderStage; 6/7 are stored, nothing
/// reads them).
struct StageBindings {
    std::array<BufferBinding, 32> uniform{};
    std::array<BufferBinding, 32> storage{};
    std::array<u64, 64> combined{};          ///< BindTextures: texture id | sampler id << 20, by slot
    std::array<u64, 128> separate_texture{}; ///< texture id
    std::array<u64, 128> separate_sampler{}; ///< sampler id << 20
    std::array<u64, 32> image{};             ///< texture id
    u64 program_serial{};                    ///< program bound to this stage (0 = none)
};

/// A render target as tracked (resolved at SetRenderTargets).
struct TargetBinding {
    /// What the draw path attaches: the game's target, or its block-view alias in the alias's own
    /// format (level 0; layer 0, or the same slices of a 3D level; Context::Impl::TargetAlias).
    VC::RenderTargetRef ref{};
    VC::RenderTargetRef game_ref{}; ///< the target as the game set it (re-bind detection)
    std::shared_ptr<Texture> texture; ///< null: empty slot
    /// The texture was replaced by its block-view alias (Context::Impl::TargetAlias).
    bool alias{false};
    /// Resolution scaling: the pass draws at the scaled size and `texture` is the scaled twin of
    /// the game's texture, which this names (null: the pass draws at the guest size).
    std::shared_ptr<Texture> native;
};

/// Services the draw path needs from the recorder that owns the command buffer.
class DrawHost {
public:
    virtual ~DrawHost() = default;
    virtual Context::Impl& HostImpl() = 0;
    virtual InFlight& HostFlight() = 0;
    virtual vk::CommandBuffer HostCmd() = 0;
    /// Emits the pending memory barrier (outside a render pass only).
    virtual void HostFlushBarrier() = 0;
    /// Before a `vkCmdBeginRenderPass` the draw path records without going through `BeginPass`
    /// (the masked clears): the render passes have no external dependency with the sync diet
    /// (`RenderPassDependencies`), so the pending barrier is emitted here; without the diet the
    /// pass's own dependency orders it and this does nothing.
    virtual void HostBarrierBeforePass() = 0;
    /// GPU writes happened: the next op outside a pass starts with a memory barrier.
    virtual void HostMarkWritten() = 0;
    /// EDEN_NX2_TRY_BARRIER_MERGE: which parts are on (BARRIER_MERGE_* bits, 0 = off). With
    /// BARRIER_MERGE_BUFFERS the replay's own commands whose buffer ranges it knows (the guest ->
    /// mirror and mirror -> snapshot copies, the shadow copies, the write-back merges) skip their
    /// barrier when no byte they touch conflicts with the buffer commands recorded since the last
    /// barrier and nothing else (a draw, a dispatch, an image copy) wrote since
    /// (nvn_barrier_track.h). Off: HostBeginBufferOp is HostFlushBarrier and HostEndBufferOp is
    /// HostMarkWritten, the commands are the ones the replay always recorded.
    [[nodiscard]] virtual u32 HostBarrierMerge() const {
        return 0;
    }
    /// Before such a command: the barrier it needs (if any), then its accesses are noted.
    virtual void HostBeginBufferOp(std::span<const BufferAccess> accesses) {
        static_cast<void>(accesses);
        HostFlushBarrier();
    }
    /// After it: its writes are pending for whatever comes next.
    virtual void HostEndBufferOp() {
        HostMarkWritten();
    }
    /// A game dispatch may have written buffers and images. Every later command of the replay, draw,
    /// pass and copy orders after it as after HostMarkWritten; a later dispatch does through
    /// HostFlushBarrierBeforeDispatch.
    virtual void HostMarkDispatchWritten() = 0;
    /// The pending memory barrier before a dispatch. With dispatch ordering by barriers
    /// (ContextOptions::dispatch_order_by_barriers) the shader writes of earlier dispatches that no
    /// game barrier and no replay command came after stay unordered, as NVN leaves them (a dispatch
    /// waits for earlier ones only behind nvnCommandBufferBarrier).
    virtual void HostFlushBarrierBeforeDispatch() = 0;
    /// The draw writes host-visible memory the host reads after the flight completes (the raw
    /// coverage diagnostic's miss ring): the segment ends with a host barrier.
    virtual void HostNeedHostReadBarrier() = 0;
    /// Transitions the whole image (outside a render pass only).
    virtual void HostTransition(Storage& storage, VkImageLayout layout) = 0;
    /// The same for several images in one barrier call; records nothing when all are in layout
    /// (the caller flushes the pending barrier itself).
    virtual void HostTransitionMany(std::span<const std::pair<Storage*, VkImageLayout>> images,
                                    bool track_writes = true) = 0;
    virtual void HostNoteImageWrite(Storage& storage) = 0;
    virtual void HostNoteTouched(const std::shared_ptr<Storage>& storage) = 0;
    /// Creates the image and loads the guest contents on first use (outside a render pass only).
    /// False: the image could not be created (the op must be dropped).
    virtual bool HostEnsureContents(Storage& storage, bool will_overwrite) = 0;
    virtual bool HostEnsureAttachmentContents(Storage& storage) = 0;
    /// True when HostEnsureContents(storage, false) would record commands (first upload, CPU
    /// change, overlapping writes); creates the image. Safe inside a render pass.
    virtual bool HostNeedsCommands(Storage& storage, bool will_overwrite) = 0;
    /// HostNeedsCommands(storage, false) for a guest storage whose scaled twin is the newer image:
    /// that alone is not a need (the caller reads the twin; Resolution scaling).
    virtual bool HostNeedsNativeUpdate(Storage& storage) = 0;
    /// Before a pass draws into the twin of `native` (outside a pass): the twin's image exists and
    /// holds the guest-sized image's GPU-written pixels. False: the twin cannot be made.
    virtual bool HostPrepareScaledTarget(Storage& native) = 0;
    /// Before a pass draws into the guest-sized image of `storage`: when it has a twin that is
    /// the newer image, the guest-sized one is made from it (outside a pass).
    virtual void HostMakeNativeCurrent(Storage& storage) = 0;
    virtual bool HostResolvePoolRange(u64 address, u64 size, Pool*& pool, u64& offset,
                                      std::string_view op) = 0;
    /// Mirror pools: copies the guest pages of the range that changed since the mirror last saw
    /// them (nothing when none did) into the mirror, before the GPU reads them (outside a pass).
    virtual void HostUploadRange(Pool& pool, u64 offset, u64 size) = 0;
    /// Mirror pools: true when HostUploadRange(pool, offset, size) would record a copy (a guest
    /// page of the range changed). Safe inside a render pass: records nothing.
    virtual bool HostMirrorNeedsUpload(Pool& pool, u64 offset, u64 size) = 0;
    virtual void HostNotePoolWrite(Pool& pool, u64 offset, u64 size) = 0;
    /// A pool mirror entry was written back into the pool's guest bytes (they equal the mirror
    /// there now): as HostNotePoolWrite, but the mirror's pages stay valid.
    virtual void HostNoteMirrorWriteBack(Pool& pool, u64 offset, u64 size) = 0;
    /// A draw or dispatch stores through raw global memory (it may write any pool).
    /// `written_through`: every raw store/atomic in the draw also writes through the alias table.
    virtual void HostNoteRawGlobalWrite(bool written_through = false) = 0;
    /// A draw or dispatch stores into a storage range served from a live shadow (the pool bytes
    /// take the stores when the shadow is copied back).
    virtual void HostNoteShadowedWrite(const Pool& pool, u64 offset, u64 size) = 0;
    virtual void HostUnsupported(std::string_view op, const std::string& reason) = 0;
    virtual void HostFailed(std::string_view op, const std::string& reason) = 0;
    /// Render conditions (NvnIndirect): false = the op is disabled by SetRenderEnable(false).
    virtual bool HostBeginRenderCondition() {
        return true;
    }
    virtual void HostEndRenderCondition() {}
    /// Draw log: the op being recorded is gated by a GPU-computed render-condition predicate (the
    /// GPU may discard it; a CPU-decided condition never records a disabled op).
    [[nodiscard]] virtual bool HostConditional() const {
        return false;
    }
    /// Ends an open conditional-rendering region before the draw path changes render passes
    /// (true if one was open) and starts it again afterwards.
    virtual bool HostSuspendCondition() {
        return false;
    }
    virtual void HostResumeCondition() {}
    /// SAMPLES_PASSED with in-pass queries (CounterEngine::InPassQueries): a draw is about to be
    /// recorded inside the open pass (a query is begun when the draw counts and none is open).
    virtual void HostBeforeDraw() {}
    /// The open pass is about to end: a query begun inside it ends here.
    virtual void HostPassEnding() {}
    /// The pass ended: the counter reports of the pass are written now.
    virtual void HostPassEnded() {}
    /// A masked clear is about to draw through a pass of its own: its samples are not counted (a
    /// clear does not count; a query spanning passes ends here).
    virtual void HostPauseCounting() {}
    /// A counter report of the open pass that is not written yet lands in
    /// [offset, offset + size) of `pool`: a draw of this pass that reads or writes the range
    /// must start a new pass.
    [[nodiscard]] virtual bool HostReportPending(const Pool& pool, u64 offset, u64 size) {
        static_cast<void>(pool);
        static_cast<void>(offset);
        static_cast<void>(size);
        return false;
    }
    /// Some counter report of the open pass is not written yet.
    [[nodiscard]] virtual bool HostReportsPending() {
        return false;
    }
    /// The records after the one being replayed, to the end of its command list (lookahead only;
    /// empty when unknown).
    [[nodiscard]] virtual std::span<const u64> HostRecordsAhead() {
        return {};
    }
    /// Following command lists in the current submission, sharing its descriptor snapshots.
    [[nodiscard]] virtual std::span<const std::vector<u64>> HostFollowingLists() {
        return {};
    }
    /// Diagnostics (EDEN_NX2_NVN_DUMP_TARGETS): records a copy of one subresource of `storage`
    /// into a host buffer that is written as a TGA when the submission completes (outside a pass).
    virtual void HostDumpImage(Storage& storage, u32 level, u32 layer, VkImageAspectFlags aspect,
                               Debug::PendingDump dump) {
        static_cast<void>(storage);
        static_cast<void>(level);
        static_cast<void>(layer);
        static_cast<void>(aspect);
        static_cast<void>(dump);
    }
    /// Diagnostics (`textures` request): records a copy of mip 0, layer 0 of `storage` into a host
    /// buffer that is converted and written as a PNG when the submission completes (outside a pass).
    virtual void HostDumpTexture(Storage& storage, const DiagTextures::Job& job) {
        static_cast<void>(storage);
        static_cast<void>(job);
    }
};

/// Where a buffer range ended up.
struct BufferRef {
    VkBuffer buffer{};
    u64 offset{};
    u64 range{};
    Pool* pool{}; ///< null for arena / dummy
    u64 pool_offset{};
};

/// The result of PrepareDraw. The caller records the draw command.
struct PreparedDraw {
    GraphicsPipeline* pipeline{};
    u32 host_topology{}; ///< NVNdrawPrimitive the host draws (after quad/loop conversion)
    bool quad_list{};    ///< QUADS: draw triangles from generated indices
    bool quad_strip{};   ///< QUAD_STRIP
    bool line_loop{};    ///< LINE_LOOP
};

struct IndexRange {
    u64 address{};
    u64 bytes{};
    u32 type{}; ///< NVNindexType
    /// The caller already refreshed the range (mirror pools): the draw does not upload it again.
    bool uploaded{};
};

struct HostIndex {
    VkBuffer buffer{};
    u64 offset{};
    VkIndexType type{VK_INDEX_TYPE_UINT32};
};

class DrawContext {
public:
    DrawContext(Context::Impl& impl, u32 queue);
    ~DrawContext();

    /// Binds the submission (descriptor pool snapshots) and host for the duration of its recording.
    void BeginSubmission(DrawHost& host, const VC::Submission& submission);
    void EndSubmission();

    /// The recorder starts / ends a command buffer.
    void OnSegmentBegin(bool keep_shadows = false);
    void OnSegmentEnd();
    /// The recorder recorded a GPU write of [offset, offset + size) of `pool`: staged vertex
    /// copies of those bytes (nvn_vertex_stage.h) are not current any more.
    void OnPoolWrite(const Pool& pool, u64 offset, u64 size);
    /// Refreshes the overlapping bytes of live storage shadows after a guest CPU write.
    void OnCpuWrite(Pool& pool, u64 offset, u64 size);
    /// The recorder passed a WaitEvent on the event at `event_address` (the uniform snapshot is
    /// dropped when the CPU signalled it after the submit, nvn_uniform_snapshot.h).
    void OnEventWaitPassed(u64 event_address);

    /// Handles a draw-path op. Returns false when the op is not this context's.
    bool Handle(const VC::RecordHeader* header);

    /// Ends the open render pass, if any, and writes pending inline uniform updates to memory.
    /// Idempotent. The single place a pass ends.
    void EndPass(BreakCause cause = BreakCause::Other);
    /// A draw of the open pass wrote a buffer range (storage buffer): an indirect read of that
    /// range needs the pass to end first.
    [[nodiscard]] bool PassWroteBuffers() const noexcept {
        return pass_buffer_writes;
    }
    [[nodiscard]] bool PassOpen() const noexcept {
        return pass_open;
    }
    /// The frame being recorded is over (a diag request bound to it just retired): logs the summary
    /// of its draws now instead of at the next draw.
    void DebugEndFrame();
    /// True when `op` can be recorded while a render pass is open (draw-path ops).
    [[nodiscard]] static bool OpKeepsPass(VC::Op op) noexcept;

    /// Binds pipeline, dynamic state, descriptors, vertex buffers (and the index buffer when
    /// `index` is given) for a draw and leaves the render pass open. False = the draw must be
    /// skipped (the reason is counted/logged). `element_count`: the vertices or indices of a direct
    /// draw, 0 for indirect and mesh draws (it decides whether the draw may skip a pipeline that is
    /// still compiling, MaySkipUnbuiltDraw).
    bool PrepareDraw(u32 mode, u32 element_count, const IndexRange* index, PreparedDraw& out,
                     HostIndex& host_index);

    /// Pushes (or allocates and binds) the descriptors of `pipeline` for the bindings of its
    /// stages. Compute dispatches use it too (`bind_point` COMPUTE). Must be called outside a
    /// render pass or, for graphics, inside the pass the draw runs in.
    bool BindDescriptors(PipelineBase& pipeline, VkPipelineBindPoint bind_point);

    [[nodiscard]] const StageBindings& Bindings(u32 stage) const {
        return stages[stage];
    }
    /// Graphics (false) or compute (true) push constant block.
    [[nodiscard]] std::span<const u8> PushConstants(bool compute) const {
        return push_constants[compute ? 1 : 0];
    }
    [[nodiscard]] const DrawCounters& Counters() const {
        return counters;
    }

    /// What the replay binds behind a shader handle word right now (the pipeline cache oracle).
    /// `compute` (compute stage only) classifies handle words of the driver bank by the program's
    /// control-derived tables.
    [[nodiscard]] TextureInfo QueryTexture(u32 stage, const VC::NvnHandleRecord& record,
                                           const VC::ComputeHandleTables* compute = nullptr) const;

    /// Shader log (nvn_draw_log.cpp): JSON of the constant buffer ranges, texture/sampler handles
    /// and descriptors, push constants and storage ranges `pipeline` reads with the current
    /// bindings. `pipeline` null (a program replaced by a stand-in): everything bound to the
    /// stage (`compute`: the compute stage). Only called while the shader log is on.
    [[nodiscard]] std::string DescribeInputs(const PipelineBase* pipeline, bool compute) const;

    /// A copy of everything QueryTexture reads for the stages of `stage_mask` (the stage bindings,
    /// the contents of their bound uniform buffers, what the texture ids that can come out of
    /// them are): it answers like QueryTexture does now, from any thread, after the replay state
    /// moved on. Used by the pipeline cache for asynchronous compiles.
    [[nodiscard]] std::unique_ptr<TextureOracle> FreezeOracle(
        u32 stage_mask, const VC::ComputeHandleTables* compute = nullptr) const;

    /// Host buffer space for this submission (host visible; uniform/storage/index/vertex/transfer
    /// source). Valid until the submission completes.
    [[nodiscard]] BufferRef AllocArena(u64 size, u64 alignment, u8** pointer);

    /// What a pool range the draw path resolves means for the storage shadows (nvn_draw_shadow.cpp):
    /// Flush: the live shadows that overlap the range are copied back first (the pass ends when
    /// it is open), the range is then read or written through host memory; Defer: the caller
    /// registers the range in the draw plan and PrepareOutsidePass decides; Ignore: debugging only.
    enum class ShadowPolicy { Flush, Defer, Ignore };
    /// Resolves a GPU address range to a host buffer (null buffer + `false` when unmapped).
    bool ResolveBuffer(u64 address, u64 size, bool write, BufferRef& out,
                       ShadowPolicy policy = ShadowPolicy::Flush);

    /// Copies every live storage shadow back to its pool and forgets it (ends the pass when it is
    /// open). The recorder calls it before a command buffer ends.
    void FlushShadows();
    /// The same for the shadows overlapping a pool range that an op outside the draw path reads or
    /// writes through host memory.
    void FlushShadowsOverlapping(const Pool& pool, u64 offset, u64 size);
    [[nodiscard]] bool HasShadows() const noexcept {
        return !shadows.empty();
    }
    [[nodiscard]] bool HasInlineUpdates() const noexcept {
        return !inline_updates.empty();
    }
    /// UpdateUniformBuffer writes recorded but not copied yet (they land when the pass ends): true
    /// when one overlaps guest bytes [address, address + size).
    [[nodiscard]] bool InlineUpdatePending(u64 address, u64 size) const noexcept {
        return std::ranges::any_of(inline_updates, [&](const InlineUpdate& update) {
            return update.address < address + size && address < update.address + update.size;
        });
    }
    /// A range the next draw or dispatch reads outside its descriptors (indirect arguments and
    /// counts, indices): no shadow is created over it for that call.
    void NoteExtraRead(const BufferRef& ref, u64 size);
    /// Indirect arguments and draw counts (resolved with ShadowPolicy::Defer): read from the pool
    /// mirror when it holds the bytes (`ref` is pointed at it), else from guest memory after the
    /// shadows over the range went back (NoteExtraRead).
    void ResolveIndirectSource(BufferRef& ref, u64 size);
    [[nodiscard]] bool HasOverlappingShadow(const Pool& pool, u64 begin, u64 end) const noexcept;

    /// Reads a u32 of a uniform buffer bound to `stage`/`slot` as the draw would see it (inline
    /// updates applied; from the submission's uniform snapshot when it holds the word).
    [[nodiscard]] std::optional<u32> ReadUniformWord(u32 stage, u32 slot, u32 byte_offset) const;

    [[nodiscard]] Context::Impl& GetImpl() {
        return impl;
    }
    /// EDEN_NX2_NVN_WATCH=gs: first storage buffer of the geometry stage of the last draw with one.
    [[nodiscard]] u64 WatchHint() const noexcept {
        return watch_hint;
    }
    /// EDEN_NX2_NVN_WATCH=ind: arguments of the first traced indexed indirect draw whose guest
    /// record drew, or of the first traced mesh-task indirect draw (whatever its count).
    [[nodiscard]] u64 WatchIndirectHint() const noexcept {
        return watch_indirect_hint;
    }
    /// EDEN_NX2_NVN_WATCH=peek: the storage range the last peeked dispatch read (0 once taken).
    [[nodiscard]] u64 TakePeekAddress() noexcept {
        return std::exchange(peek_address, 0);
    }
    u64 peek_address{0};
    u64 peek_chase_address{0};

private:
    struct InlineUpdate {
        u64 address{};
        u64 size{};
        u8* data{};
        VkBuffer buffer{};
        u64 buffer_offset{};
    };

    // ---- ops (nvn_draw.cpp) ----
    void SetRenderTargets(const VC::RecordHeader* header);
    void ClearColor(const VC::ClearColorPayload& p);
    void ClearDepthStencil(const VC::ClearDepthStencilPayload& p);
    void SetStencil(const VC::SetStencilPayload& p);
    void BindState(const VC::RecordHeader* header);
    void BindVertexAttribs(const VC::RecordHeader* header);
    void BindVertexStreams(const VC::RecordHeader* header);
    void BindVertexBuffers(const VC::RecordHeader* header);
    void BindBuffers(const VC::RecordHeader* header, bool uniform);
    void BindHandles(const VC::RecordHeader* header, VC::Op op);
    void PushConstantsOp(const VC::RecordHeader* header);
    void UpdateUniformBuffer(const VC::RecordHeader* header);
    void Draw(const VC::DrawPayload& p);
    void OrderIndirectArguments(const BufferRef& args, u64 args_size, const BufferRef& count,
                                bool has_count, BufferRef* index);
    void DrawIndirect(const VC::DrawPayload& p);
    void DrawIndexedIndirect(const VC::DrawIndexedPayload& p);
    bool GenerateIndices(const PreparedDraw& prepared, const IndexRange* source, u32 first,
                         u32 count, s32 base_vertex, HostIndex& out, u32& out_count);
    void DrawIndexed(const VC::DrawIndexedPayload& p);
    /// Every nvnCommandBufferDrawMeshTasks* entry (nvn_draw_mesh.cpp): vkCmdDrawMeshTasks*EXT.
    void DrawMeshTasks(const VC::DrawMeshTasksPayload& p);
    void DrawMeshTasksIndirect(const VC::DrawMeshTasksPayload& p);
    void SetProgram(const VC::BindProgramPayload& p);

    // ---- storage shadows and pool mirrors (nvn_draw_shadow.cpp) ----
    /// A written storage range of an imported pool that lives in device memory while it is
    /// bound: `buffer` holds the bytes of pool [begin, end) at `offset`. The pool's host memory is
    /// stale until the shadow is copied back (FlushShadow); nothing may read or write those bytes
    /// through the host buffer in between (every use is classified: ClassifyShadows).
    ///
    /// With pool mirrors (`mirror`) the buffer is the pool's VRAM mirror (offset == begin), which
    /// outlives the entry, and `snapshot` holds the mirror's bytes from before the first GPU write
    /// of the entry: FlushShadow writes back only the 32-bit words that differ from it, so guest
    /// CPU writes to other words of the range made meanwhile survive.
    struct Shadow {
        Pool* pool;
        u64 begin;
        u64 end;
        VkBuffer buffer;
        u64 offset;
        bool mirror{false};
        VkBuffer snapshot{};
        u64 snapshot_offset{};
        /// Created by the plan being prepared: its snapshot copy is not recorded yet.
        bool snapshot_pending{false};
    };
    /// What PrepareOutsidePass does about the shadows for the current plan.
    static constexpr size_t MAX_LIVE_SHADOWS = 128;
    struct ShadowDecision {
        std::bitset<MAX_LIVE_SHADOWS> flush_mask{}; ///< live shadows (by index) a use conflicts with
        bool flush_needed{false};      ///< flush_mask is not empty
        bool need_outside_pass{false}; ///< a read-only mirror binding needs stale non-shadow pages copied in outside a pass
        bool any{false};               ///< the plan touches shadows or can create one
    };
    /// Looks at the ranges the plan uses against the live shadows; records nothing.
    [[nodiscard]] ShadowDecision ClassifyShadows(const DrawPlan& plan);
    /// Records the copies (outside a pass only): copies back the shadows the plan conflicts with
    /// (classified again: the pass break may have copied some back already), creates the shadows of
    /// the plan's candidates when `allow_create`, and patches the descriptors of every binding
    /// inside a live shadow.
    void ApplyShadows(DrawPlan& plan, bool allow_create);
    /// The live shadow that holds [begin, end) of the pool (shadows.size() when there is none).
    [[nodiscard]] size_t FindShadow(const Pool& pool, u64 begin, u64 end) const;
    /// No use of the plan overlaps the candidate except descriptor bindings inside its range.
    [[nodiscard]] bool ShadowIsAloneIn(const DrawPlan& plan,
                                       const DrawPlan::RangeUse& candidate) const;
    bool AllocShadow(u64 size, u64 begin, VkBuffer& buffer, u64& offset);
    bool CreateShadow(DrawPlan& plan, const DrawPlan::RangeUse& use);
    /// Copies the shadow back to its pool and forgets it (outside a pass only).
    void FlushShadow(size_t index, std::atomic<u64> DrawCounters::*cause);
    /// The copies of one loop over shadows (creations of one plan, write-backs of one flush) need
    /// one barrier before them, not one each: live shadows cover disjoint pool ranges and each has
    /// its own slot, so no copy of the loop reads or writes bytes another one writes (the copies
    /// read the pool or a slot and write a slot or their own pool range). Inside a batch a copy
    /// leaves the HostMarkWritten to the batch's end; the next copy's HostFlushBarrier still emits
    /// the barrier for work recorded before it (the dispatches that wrote a shadow, a pass end).
    /// Every batch end marks its copies written, a nested one too: a copy after it waits for them.
    struct ShadowCopyBatch {
        explicit ShadowCopyBatch(DrawContext& context_) : context{context_} {
            ++context.shadow_batch_depth;
        }
        ~ShadowCopyBatch() {
            --context.shadow_batch_depth;
            if (context.shadow_batch_copied) {
                context.shadow_batch_copied = false;
                context.host->HostMarkWritten();
            }
        }
        ShadowCopyBatch(const ShadowCopyBatch&) = delete;
        ShadowCopyBatch& operator=(const ShadowCopyBatch&) = delete;
        DrawContext& context;
    };
    /// After a shadow copy: HostMarkWritten now, or at the end of the open batch.
    void MarkShadowCopy();
    /// After a shadow copy whose accesses went through HostBeginBufferOp: the tracker holds its
    /// writes with EDEN_NX2_TRY_BARRIER_MERGE, else MarkShadowCopy.
    void EndShadowCopy();
    u32 shadow_batch_depth{0};
    bool shadow_batch_copied{false};
    /// Pool mirrors (ApplyShadows with EDEN_NX2_NVN_POOL_MIRROR): serves the plan's storage ranges
    /// from their pools' VRAM mirrors. Outside a pass the stale pages are copied in and a written
    /// range gets an entry (snapshot); inside one only ranges the mirror holds current are served
    /// (reads), or ranges of a live entry.
    void ApplyMirrors(DrawPlan& plan, bool allow_create);
    /// Records the guest -> mirror copies of `mirror_copies` (one barrier) and clears it;
    /// `before_vertex_input`: also orders them before the vertex/index fetches of the next pass.
    void RecordMirrorCopies(bool before_vertex_input = false);
    /// EDEN_NX2_TRY_UBO_MIRROR: before a pass begins, copies into the mirrors made for uniform
    /// bindings (PoolMirror::prefetch) every stale page uniform bindings used before (`wanted`):
    /// the working set bounds the copy, so a pool's first pass copies nothing it never bound.
    void PrefetchMirrors();
    /// Writes the GPU's changes of mirror entries back into their pools (one barrier, word merge
    /// when available) and forgets the entries. Outside a pass only.
    void WriteBackMirrorEntries(std::span<const Shadow> entries, std::atomic<u64> DrawCounters::*cause);
    std::vector<Shadow> shadows;
    /// Ranges the next draw or dispatch reads outside its descriptors (NoteExtraRead).
    std::vector<DrawPlan::RangeUse> extra_uses;
    std::set<std::tuple<u64, u64, u64>> shadow_logged; ///< EDEN_NX2_NVN_SHADOW_LOG only
    std::set<std::tuple<u64, u64, u64>> host_write_logged; ///< EDEN_NX2_NVN_SHADOW_LOG only
    /// Guest -> mirror page runs ApplyMirrors collected: pool, [begin, end).
    struct MirrorCopy {
        Pool* pool;
        u64 begin;
        u64 end;
    };
    std::vector<MirrorCopy> mirror_copies;
    std::vector<std::pair<u64, u64>> mirror_runs;   ///< scratch of MirrorTakeStale
    std::vector<VkBufferCopy> mirror_regions;       ///< scratch of the copy recordings

    // ---- compute (nvn_compute.cpp) ----
    void Dispatch(const VC::DispatchPayload& p);

    // ---- upscaling (nvn_upscale.cpp) ----
    /// One image of an upscale: the texture and the level and layer (and format, when the game's
    /// view reinterprets it) it is read or written at, and the origin of its rectangle.
    struct UpscaleImage {
        Texture* texture{};
        u32 level{};
        u32 layer{};
        VkFormat format{VK_FORMAT_UNDEFINED}; ///< undefined: the texture's own format
        u32 x{};
        u32 y{};
    };
    /// What one upscale needs: DK's NDSR replacement (colour only, `host_inputs`) and the game's
    /// own nvnCommandBufferEvaluateDlss (every input the game made) both run through it.
    struct UpscaleJob {
        UpscaleImage input;
        UpscaleImage output;
        VideoCore::Upscaler::Rect input_rect;  ///< in `input`'s level
        VideoCore::Upscaler::Rect output_rect; ///< in `output`'s level
        /// The game's depth, motion vectors, bias mask and exposure. Their rectangles are
        /// input_rect's size at (x, y), except the motion vectors' (motion_width x
        /// motion_height); the bias and exposure images are optional.
        UpscaleImage depth;
        UpscaleImage motion_vectors;
        UpscaleImage bias;
        UpscaleImage exposure;
        /// The host makes the motion vectors (optical flow of consecutive inputs), the exposure and,
        /// unless `depth` is set (the scene's depth, same rectangle as the input), a far-plane
        /// depth itself, because the game has no such inputs (NDSR binds colour only). Otherwise
        /// the evaluation needs the game's depth and motion vectors.
        bool host_inputs{};
        float jitter[2]{}; ///< pixels
        /// Streamline's mvecScale: the game's motion vectors times it are normalized.
        float motion_scale[2]{1.0f, 1.0f};
        float pre_exposure{1.0f};
        bool reset{};
        bool depth_inverted{};
        bool hdr_color{};
        bool auto_exposure{};
        bool motion_vectors_jittered{};
        /// Size of the motion vector rectangle (full-resolution motion vectors: the output
        /// size); 0 = input_rect's size.
        u32 motion_width{};
        u32 motion_height{};
        /// The job is the game's own DLSS evaluation: the setting's "game's upscaler" (Auto, Game)
        /// then means DLSS, the family of network the game asked for, not FSR1.
        bool game_dlss{};
        /// The job replaces a game's own upscaling network at the Resolution setting above 1X with
        /// the Auto upscaler: Auto then means DLSS (FSR1 when it cannot run), as for game_dlss.
        bool prefer_dlss{};
        /// The game network's id (its Streamline viewport, never 0), NGX method and preset hint.
        u32 viewport{};
        u32 game_method{};
        u32 game_preset{};
    };
    /// Replaces a game's upscaling network dispatch (NDSR shape, tensor math, an output larger than
    /// the input: IsUpscaleNetworkDispatch) with the selected upscale stage. The caller has
    /// already recognised the dispatch; this helper resolves the current compute handle tables,
    /// finds the scene colour behind the network's input (a packed prepass output, or the input
    /// itself), takes its current twin and the scene depth when the Resolution setting is active,
    /// and records the upscale in the active command buffer. False (nothing recorded, the game's
    /// own program runs) when the colour cannot be found or used.
    /// `output_id` is the storage image whose dimensions accepted output_rect; a network may bind
    /// auxiliary images before its real output.
    bool UpscaleNetwork(const VideoCore::Upscaler::Rect& input_rect,
                        const VideoCore::Upscaler::Rect& output_rect, u32 output_id);
    /// The backend the setting selects for a game's upscale stage (SelectBackend with the
    /// device's tensor-core support, as the compute dispatch decides it).
    [[nodiscard]] VideoCore::Upscaler::Backend SelectedUpscaleBackend(
        Settings::Upscaler requested) const;
    /// Whether the game's upscaling network would be replaced now (ReplacesGameNetwork with the
    /// current setting and Resolution factor). Cheap; the replay keeps the data a replacement
    /// needs (compute chain, pass depths) only while it is true.
    [[nodiscard]] bool NetworkReplacementWanted() const;
    /// Remembers the textures the bound compute dispatch samples and writes (compute_chain).
    void NoteComputeChain();
    /// Whether the bound compute program holds tensor-core math (cached by program hash).
    [[nodiscard]] bool ComputeProgramUsesTensorMath(u64 hash, const std::vector<u8>& code,
                                                    u32 instruction_bytes);
    /// The current scaled twin of `texture` for an upscale (level 0, current, no guest update
    /// needed), created on demand; false when there is none, and then `why` (when given) names
    /// the reason as a static string.
    [[nodiscard]] bool UpscaleTwinOf(Texture& texture, u32 level, std::shared_ptr<Texture>& out,
                                     const char** why = nullptr);
    /// Op::EvaluateDlss: the game's own DLSS evaluation (nvnCommandBufferEvaluateDlss), run by the
    /// upscaler the user selected. Outside a render pass.
    void EvaluateDlssOp(const VC::EvaluateDlssPayload& p);
    /// Writes `job.output_rect` of the output with the upscaler the setting selects: DLSS when it
    /// can evaluate, otherwise FSR1 or bilinear (a failed DLSS evaluation included, and bilinear
    /// when FSR1 cannot run), so the output is written whenever this returns true. `stage` names
    /// what wrote it. The images' contents are already ensured and in place; false when nothing
    /// could be recorded.
    bool RunUpscale(const UpscaleJob& job, const char*& stage);
    /// DLSS super resolution / DLAA of the job. False when nothing was recorded for the output
    /// (the caller then writes it with another stage). `input_rect_override` is DLAA's aligned
    /// output-size crop; all other inputs and the output stay at the job's rectangles.
    bool EvaluateDlss(const UpscaleJob& job, bool dlaa,
                      const VideoCore::Upscaler::Rect* input_rect_override = nullptr);
    /// The non-DLSS stage of RunUpscale: FSR1 or bilinear.
    bool UpscaleWithoutDlss(const UpscaleJob& job, bool use_fsr);
    /// The compute driver bank (c0) of the current compute bindings, DRIVER_BANK_SIZE bytes.
    void FillComputeBank(const VC::ComputeDriverLayout& layout, u8* out) const;

    // ---- passes (nvn_draw.cpp) ----
    bool BeginPass();
    /// Where a render target attaches: the view, its extent, layer count.
    struct AttachmentShape {
        ViewKey key;
        VkExtent2D extent;
        u32 layer_count;
        VkFormat format;
    };
    bool ShapeOf(const TargetBinding& target, bool is_depth, AttachmentShape& out);
    /// True when `storage` is bound as a render target of the draw path right now.
    [[nodiscard]] bool IsBoundTarget(const Storage* storage) const;
    /// True when a bound render target of `storage` attaches a level and layer that the sampled
    /// view `key` also covers (a real feedback loop); false when they are disjoint subresources.
    [[nodiscard]] bool TargetOverlapsView(const Storage& storage, const ViewKey& key) const;
    /// Records the copies of render targets that the draw samples (outside the pass).
    void RecordFeedbackCopies(DrawPlan& plan);
    /// Drops the feedback copies of the plan whose scratch already holds the source (see the
    /// DESIGN.md paragraph "Feedback copies per target epoch").
    void PruneFreshFeedback(DrawPlan& plan);
    void FlushInlineUpdates();

    // ---- uniform snapshot (nvn_uniform_snapshot.cpp) ----
    /// The descriptor of a uniform binding served from range `index` of the snapshot (uploaded
    /// to the arena at its first binding in the submission); nullopt when it cannot be.
    [[nodiscard]] std::optional<VkDescriptorBufferInfo> SnapshotUniform(size_t index, u64 address,
                                                                        u64 size);
    /// Copies [address, address + size) from the snapshot when one live range holds it all.
    [[nodiscard]] bool ReadSnapshotBytes(u64 address, void* out, u64 size) const;
    UniformSnapshotView usnap;

    [[nodiscard]] VkRect2D ClearRect() const;
    bool MaskedColorClear(const VC::ClearColorPayload& p, VkRect2D rect, bool gated);
    bool MaskedStencilClear(const VC::ClearDepthStencilPayload& p, VkRect2D rect, bool gated);
    [[nodiscard]] RenderPassKey MakePassKey() const;
    [[nodiscard]] std::optional<RenderPassKey> LookaheadPassKey(
        const VC::RecordHeader* header) const;
    [[nodiscard]] bool TargetsUsable() const;
    /// True when the game bound no colour and no depth target (an attachment-less pass).
    [[nodiscard]] bool NamesNoTarget() const;
    /// Slot by slot why TargetsUsable() is false (log text).
    [[nodiscard]] std::string DescribeUnusableTargets() const;

    // ---- pipeline and state (nvn_draw.cpp) ----
    /// The pipeline of the next draw. Null with `awaiting_async` set: its compile runs on a worker
    /// (PipelineCacheOptions::async_compile) and the draw may be skipped (MaySkipUnbuiltDraw);
    /// such a draw that may not be skipped waits for the compile. Null without it: it cannot be
    /// built.
    struct PipelineLookahead {
        u64 skip_key;
        RenderPassKey targets;
    };
    std::shared_ptr<GraphicsPipeline> ResolvePipeline(u32 mode, u32 element_count,
                                                      bool& awaiting_async,
                                                      const PipelineLookahead* lookahead = nullptr);
    void PrefetchUpcomingGraphics(u32 current_mode, u64 current_key);
    void ApplyDynamicState(const GraphicsPipeline& pipeline);

    // ---- descriptors (nvn_draw_descriptors.cpp) ----
    bool BuildDescriptors(PipelineBase& pipeline, DrawPlan& plan, bool graphics);
    bool PrepareOutsidePass(DrawPlan& plan);
    // ---- bindless texture table (nvn_bindless.cpp) ----
    /// The replay's table of every texture and sampler the game registered, for shaders that
    /// read a handle at run time (shader_recompiler/bindless_table.h).
    struct BindlessState {
        bool failed{false};
        vk::DescriptorSetLayout layout;
        vk::DescriptorPool pool;
        VkDescriptorSet set{VK_NULL_HANDLE};
        /// The stand-in of each texture array and what each entry holds now.
        struct WrittenEntry {
            VkImageView view{VK_NULL_HANDLE};
            VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
            bool operator==(const WrittenEntry&) const = default;
        };
        std::array<VkImageView, Shader::BINDLESS_BINDING_COUNT> dummy{};
        std::array<std::vector<WrittenEntry>, Shader::BINDLESS_BINDING_COUNT> written;
        std::vector<VkSampler> written_samplers;
        /// The storages of the entries that hold an image (their layouts are checked per draw).
        std::vector<std::shared_ptr<Storage>> storages;
        u64 storages_gen{0};
        /// Fast-path state for BindlessPrepare: while inside the same open render pass with no
        /// intervening GPU writes, pass breaks, or table updates, all table storages are already
        /// in their read layout (or bound as targets).
        u64 prepared_seq{~u64{0}};
        u64 prepared_write_clock{~u64{0}};
        u64 prepared_pass_breaks{~u64{0}};
        u64 prepared_pass_begins{~u64{0}};
        u64 prepared_storages_gen{~u64{0}};
        /// Last descriptor set bind recorded by BindlessBind, to elide redundant binds on
        /// consecutive bindless draws in the same open render pass with the same pipeline layout.
        VkCommandBuffer bound_cmd{VK_NULL_HANDLE};
        VkPipelineLayout bound_layout{VK_NULL_HANDLE};
        VkPipelineBindPoint bound_point{VK_PIPELINE_BIND_POINT_GRAPHICS};
        u64 bound_seq{~u64{0}};
        u64 bound_pass_begins{~u64{0}};
        u64 bound_draw{0};
        /// What entry `id` was last written from. While the registration, the texture, its
        /// storage and that storage's image are the same, the entry holds what it should and the
        /// sync leaves it (the pool has tens of thousands of registrations; walking each one's
        /// views at every submission was ~10% of the replay thread in Ocarina of Time).
        struct Entry {
            u64 serial{};
            /// The texture's node of Impl::textures (null: none); dies when the serial is erased.
            const std::shared_ptr<Texture>* texture{};
            const Storage* storage{};
            u32 image_gen{};
            VkFormat format{};
            VkImageLayout layout{};
            bool ready{}; ///< the storage had an image holding the game's data
            bool any{};   ///< the entry holds a view of it
        };
        std::vector<Entry> entries;
        u64 entries_epoch{}; ///< Impl::texture_epoch the node pointers were checked at
        u64 synced_seq{~u64{0}};
        const void* synced_textures{};
        const void* synced_samplers{};
        u32 synced_count{};
        u32 synced_sampler_count{};
        u32 sync_logs{};
        /// Textures registered without an image yet are created (and their data loaded) as draws
        /// need the table, within a budget per submission; a change makes the entries refresh
        /// before the set is bound.
        bool dirty{false};
        u64 prime_seq{~u64{0}};
        u64 prime_bytes{};
        u32 prime_count{};
        u32 prime_cursor{1};
        /// Submission whose walk of the pool found nothing left to create (BindlessPrime).
        u64 prime_exhausted_seq{~u64{0}};
        u32 primed_total{};
    } bindless;
    /// Creates the table on first use; false when the device cannot hold it.
    bool BindlessReady();
    /// Brings the table up to the submission's pool registrations (once per submission).
    void BindlessSync(bool force = false);
    /// Adds to `plan` the registered textures that have no image yet, up to the submission's
    /// budget (creating one needs a pass break, and its memory counts).
    void BindlessPrime(DrawPlan& plan);
    /// Syncs the table and adds to `plan` the layout changes its images need.
    bool BindlessPrepare(DrawPlan& plan);
    /// Binds the table's set for the pipeline's layout.
    void BindlessBind(PipelineBase& pipeline, VkPipelineBindPoint bind_point);
    /// Phase B of the descriptors: creates the image views the plan resolved and patches the
    /// image infos (the images exist and are in layout by now).
    void FinishImages(DrawPlan& plan);
    bool PushPlan(PipelineBase& pipeline, DrawPlan& plan, VkPipelineBindPoint bind_point);
    VkDescriptorSet AllocateSet(VkDescriptorSetLayout layout);
    /// The handle word of `ref` in global descriptor ids: words of the bind commands were
    /// rebased by the recorder, words read from a uniform buffer are rebased here with the
    /// texture / sampler pools bound when the draw runs (`kind`: which ids the word holds).
    [[nodiscard]] std::optional<u32> ReadHandleWord(u32 stage, const VC::HandleRef& ref,
                                                    VC::HandleWordKind kind =
                                                        VC::HandleWordKind::Texture) const;
    /// The handle word of a texture or image binding: `ReadHandleWord` of `ref`, or, when the
    /// shader reads the handle from a table in memory (HandleSource::Program), the evaluation of
    /// the binding's handle program for this draw.
    [[nodiscard]] std::optional<u32> ReadBindingHandleWord(const VC::DescriptorLayout& layout,
                                                           const VC::DescriptorBinding& binding,
                                                           const VC::HandleRef& ref,
                                                           VC::HandleWordKind kind) const;
    /// The {address, size} the game bound for element `element` of a storage buffer binding
    /// (address 0: nothing bound or the source word is unreadable).
    [[nodiscard]] BufferBinding ResolveStorageBinding(const VC::DescriptorBinding& binding,
                                                      u32 element) const;
    /// The texture behind a descriptor id of the submission's snapshot (null reference: none).
    /// The reference stays valid until a texture is destroyed.
    [[nodiscard]] const std::shared_ptr<Texture>& TextureOfId(u32 id) const;
    struct TextureSlot {
        u32 id{~0u};
        u64 serial{};
        std::shared_ptr<Texture> texture;
    };
    mutable std::array<TextureSlot, 4096> texture_slots{};
    u64 texture_slots_cleaned_epoch{0};
    [[nodiscard]] u64 SamplerSerialOfId(u32 id) const;
    void NoteFallback(bool graphics, DescReason reason, const std::string& detail = {});
    /// Describes (once per program, slot and source) a storage buffer binding the game left empty.
    void LogUnboundStorage(bool graphics, const PipelineBase& pipe,
                           const VC::DescriptorBinding& binding, u32 slot,
                           const BufferBinding& resolved);
    std::set<std::tuple<u64, u32, u32, u32>> unbound_logged;
    u64 watch_hint{0};
    u64 watch_indirect_hint{0};
    u32 log_backing_dispatches{};
    std::string dbg_notes; ///< fallbacks of the draw being prepared (tracing only)

    Context::Impl& impl;
    DrawResources& resources;
    u32 queue;
    DrawHost* host{};
    const VC::Submission* submission{};

    // ---- tracked NVN state (persists across submissions of the queue) ----
    std::array<TargetBinding, MAX_COLOR_TARGETS> color_targets{};
    TargetBinding depth_target{};
    u32 color_target_count{};
    std::array<VC::ViewportRect, MAX_VIEWPORTS> viewports{};
    std::array<VC::ScissorRect, MAX_VIEWPORTS> scissors{};
    std::array<VC::DepthRange, MAX_VIEWPORTS> depth_ranges{};
    u32 viewports_set{};  ///< bit k: viewport k was set
    u32 scissors_set{};   ///< bit k: scissor k was set
    std::array<StageBindings, BIND_STAGES> stages{};
    /// Per-slot resolved buffer cache for `stages[stage].uniform` / `storage` (parallel to
    /// `stages` so PrefetchUpcomingGraphics does not copy it). Validated by address, binding size,
    /// pool/mapping/alias generations, and pool backing state before use.
    struct CachedBuffer {
        BufferRef ref;
        u64 pool_gen{~u64{0}};
        u64 mapping_gen{~u64{0}};
        u64 alias_gen{~u64{0}};
        u64 address{};
        u64 binding_size{};
    };
    std::array<std::array<CachedBuffer, 32>, BIND_STAGES> uniform_cache{};
    std::array<std::array<CachedBuffer, 32>, BIND_STAGES> storage_cache{};
    /// Base of the texture / sampler pool bound last in the global id space (SetTexturePool /
    /// SetSamplerPool): the numbering of the ids a game stores in memory.
    u32 texture_pool_base{};
    u32 sampler_pool_base{};
    std::array<std::vector<u32>, 2> pool_bases_logged; ///< texture, sampler
    std::set<u64> incompatible_logged; ///< (texture serial, shader type) described once
    std::set<u64> placeholder_logged;  ///< (texture serial, shader type) a typed placeholder was bound for
    GraphicsFixedState fixed{};
    std::array<VC::BufferRange, MAX_VERTEX_STREAMS> vertex_buffers{};
    struct StencilFace {
        u32 value_mask{0xFF};
        u32 write_mask{0xFF};
        u32 reference{0};
    };
    std::array<StencilFace, 2> stencil{};
    f32 blend_color[4]{};
    f32 polygon_offset[3]{};
    f32 line_width{1.0f};
    std::array<std::array<u8, PUSH_CONSTANT_BLOCK_SIZE>, 2> push_constants{};
    std::vector<InlineUpdate> inline_updates;
    /// Group counts of the compute dispatch being bound (c0+4..0xc of its driver bank).
    std::array<u32, 3> compute_groups{};

    // ---- per command buffer / pass state ----
    // The flight owns this framebuffer and its attachments until execution completes.
    struct {
        VkRenderPass pass{};
        VkFramebuffer handle{};
        VkExtent2D extent{};
        u32 layers{};
        std::array<VkImageView, MAX_COLOR_TARGETS + 1> views{};
    } last_framebuffer;
    bool pass_open{false};
    void NoteFlipTarget(const std::shared_ptr<Storage>& storage, VkImageLayout sampled_layout);
    /// Storages that flip between attachment and sampled use, with the layout they are sampled in:
    /// a pass that begins moves them to it before its first draw, so a draw that samples one
    /// (typically a post-processing buffer written by the previous pass) does not end the pass for
    /// the transition. Learned from the layout breaks PrepareOutsidePass sees (FlipTargetList).
    FlipTargetList<Storage> flip_targets;
    std::vector<std::pair<Storage*, VkImageLayout>> transition_scratch;
    /// Counts the SetRenderTargets ops that changed the bound targets (feedback copy reuse).
    FeedbackEpoch render_target_epoch;
    /// A draw of the open pass wrote through a shader (storage buffer, storage image, raw global
    /// memory): a game barrier that orders draws must end the pass.
    bool pass_shader_writes{false};
    bool pass_buffer_writes{false};
    /// A game barrier that orders draws (NVN barrier bits 0xF) came after a counter report of the
    /// open pass: later draws that read the report's bytes must see it (PrepareOutsidePass).
    bool pass_reports_ordered{false};
    bool targets_dirty{true};
    bool pipeline_dirty{true};
    bool vertex_dirty{true};
    /// Vertex layout of the draw being prepared: BuildVertexInput of `fixed` for the pipeline's
    /// `attributes_read`. Rebuilt only when the layout words (BindVertexAttribs/BindVertexStreams)
    /// or the pipeline's attributes_read changed. Gives the streams whose buffers are bound; with
    /// a dynamic-vertex-input pipeline it is also what vkCmdSetVertexInputEXT is called with.
    VertexInput vertex_input;
    u32 vertex_input_read{};
    bool vertex_layout_dirty{true};
    /// `vertex_input` is the vertex input state of the command buffer (dynamic pipelines only):
    /// cleared when the layout changes, at the start of a command buffer and whenever a pipeline
    /// is bound with no known predecessor (a pipeline bound outside this tracking, e.g. the
    /// clear pipeline, replaces the dynamic state with its own static one).
    bool vertex_input_set{false};
    std::vector<std::string> vertex_issues; ///< scratch of BuildVertexInput
    u32 dynamic_dirty{~0u};
    u32 last_mode{~0u};
    u32 last_viewport_count{0};
    VkPipeline bound_pipeline{VK_NULL_HANDLE};
    /// GPL pipelines (GraphicsPipeline::dynamic_fixed): the cull, depth and stencil values set in
    /// the command buffer. Unknown after a pipeline with them static was bound, or one with no
    /// known predecessor (the same rule as `vertex_input_set`).
    bool fixed_state_set{false};
    DynamicFixedState fixed_state;
    /// What the open pass has recorded already (pipeline layout, push constants, descriptors,
    /// vertex and index buffers): a draw that would record the same again skips it. Reset at every
    /// pass begin/end and command buffer start (nvn_pass_state.h).
    PassBoundState bound_state;
    /// Vertex ranges of imported pools copied into device-local memory (nvn_vertex_stage.h), by
    /// guest address. Filled at pass begin; kept for later passes of the command buffer while
    /// nothing could have written their bytes (StagedSurvives, OnPoolWrite), dropped otherwise.
    /// Draws use them only while no draw of the pass has written through shaders.
    struct StagedRange {
        u64 size;
        VkBuffer buffer;
        u64 offset;
        const Pool* pool;
        u64 pool_offset;
        bool mirror{}; ///< `buffer` is the pool's VRAM mirror (TRY_VTX_MIRROR)
    };
    std::unordered_map<u64, StagedRange> staged_vertex;
    /// EDEN_NX2_NVN_VTXCOV_DIAG: what the last staging scan did with each range it found (by
    /// address; a served reason: it went into staged_vertex), and why a range it did not find is
    /// not staged (the op that ended the scan, or the byte cap).
    struct StageOutcome {
        VtxCov reason;
        u64 size;
    };
    std::unordered_map<u64, StageOutcome> stage_outcomes;
    VtxCov stage_unscanned{VtxCov::NotScanned};
    /// EDEN_NX2_TRY_STAGE_SPAN: the records [begin, end) of the current list the last staging scan
    /// covered, while its copies are current: a pass beginning inside does not scan again.
    const u64* stage_covered_begin{};
    const u64* stage_covered_end{};
    VertexStageScan stage_scan;
    std::vector<VertexStageSource> stage_sources;
    std::vector<VertexStageSpan> stage_spans;
    /// Where each of stage_spans was copied (chunk buffer, offset), by span index.
    struct StagedSpan {
        VkBuffer buffer;
        u64 offset;
    };
    std::vector<StagedSpan> stage_span_dst;
    struct StageCopy {
        const Pool* source;
        VkBuffer destination;
        VkBufferCopy region;
    };
    std::vector<StageCopy> stage_copies;
    /// A run of stage_copies with the same source and destination: one dispatch of
    /// VertexStageGpu over `task_count` tasks from `first_task` of stage_tasks, or (`shader`
    /// false) one vkCmdCopyBuffer.
    struct StageRun {
        size_t begin;
        size_t end;
        bool shader;
        u64 window_base;
        u64 window_range;
        u32 first_task;
        u32 task_count;
    };
    std::vector<StageRun> stage_runs;
    std::vector<VertexStageTask> stage_tasks;
    std::vector<VkBufferCopy> stage_regions;
    /// Records the copies of the ranges the pass about to begin reads (outside a pass, after the
    /// pending barrier was flushed).
    void StageVertexRanges();
    /// EDEN_NX2_NVN_VTXCOV_DIAG: notes the scan just made of `ahead` (stage_outcomes cleared,
    /// stage_unscanned from the op that ended it).
    void VtxCovNoteScan(std::span<const u64> ahead);
    /// A vertex stream or index buffer of an imported pool that no staged copy serves: served
    /// from its pool's mirror when EDEN_NX2_TRY_VTXCOV_INPASS/_SHADERWRITES allow it and the
    /// mirror is current (`ref` rebound), else why it stays in host memory. `found`: what the
    /// staged copy lookup found (SizeShort, else Other).
    VtxCov VtxCovServeOrMiss(BufferRef& ref, u64 address, u64 size, bool indirect_index,
                             const DrawPlan& plan, VtxCov found);
    std::vector<PassBoundState::WriteSig> scratch_write_sigs;
    u32 feedback_logged{0};
    std::vector<PassBoundState::ImageSig> scratch_image_sigs;
    std::shared_ptr<GraphicsPipeline> pipeline;
    /// BuildDescriptors: the view key a texture object gets for a binding kind (`ViewKeyOf`).
    struct ViewSlot {
        const Texture* texture{};
        u64 serial{};
        const Storage* storage{};
        VkFormat storage_format{};
        u32 kind{~0u};
        bool compatible{};
        ViewKey key{};
    };
    std::array<ViewSlot, 1024> view_slots;
    /// The view `texture` (on `s`) is bound with for a shader of `type`; false when the storage
    /// cannot be bound as that type. `key` is always filled.
    bool ViewKeyOf(const Texture& texture, const Storage& s, Shader::TextureType type, bool is_image,
                   ViewKey& key);
    /// Samplers by descriptor id, valid while `DrawResources::samplers_epoch` is unchanged:
    /// `plain` is the guest's own sampler, `boosted` the one with the Anisotropic Filtering
    /// setting applied (for views ViewSupportsAnisotropy accepts); null until first asked for.
    struct SamplerSlot {
        u32 id{~0u};
        u64 serial{};
        u64 epoch{};
        VkSampler plain{};
        VkSampler boosted{};
    };
    std::array<SamplerSlot, 256> sampler_slots{};
    [[nodiscard]] VkSampler SamplerOf(u32 id, u64 serial, bool boosted);
    /// ResolvePipeline's memo of recent requests (program serials, the draw's raw fixed state and
    /// the pass formats): a draw whose state words are the ones of an earlier draw finds its
    /// pipeline without canonicalizing and hashing the key again.
    struct PipelineMemo {
        std::array<u64, GRAPHICS_STAGE_COUNT> programs{};
        GraphicsFixedState state{};
        RenderPassKey targets{};
        std::shared_ptr<GraphicsPipeline> pipeline;
        std::array<u64, BIND_STAGES> query_handle_gen{};
        u64 query_sub_seq{~u64{0}};
        u64 query_tex_epoch{~u64{0}};
    };
    std::vector<PipelineMemo> pipeline_memo;
    struct QueryCache {
        u64 sub_seq{~u64{0}};
        u64 tex_epoch{~u64{0}};
        std::array<u64, BIND_STAGES> handle_gen{};
    } query_cache;
    [[nodiscard]] bool QueriesValidFor(const GraphicsPipeline& candidate,
                                       const std::array<u64, BIND_STAGES>& cached_gen,
                                       u64 cached_sub_seq, u64 cached_tex_epoch) const noexcept;
    u64 shader_generation{};
    /// The driver (c0) and handle (c2) banks of the graphics stages are rebuilt only when what
    /// they are made of changed (storage buffer table, push constants, handle tables, raw global
    /// table): the arena copy of the last build is reused by every draw of the submission until
    /// then. The generations come from `bank_clock`, bumped by the bind ops.
    u64 bank_clock{0};
    std::array<u64, BIND_STAGES> storage_gen{};
    std::array<u64, BIND_STAGES> handle_gen{};
    std::array<u64, 2> push_gen{};
    /// Workgroup counts of the last direct 3D mesh-task draw, left in the driver bank like the
    /// draw macro does (VC::DriverMeshDimsOffset); `mesh_dims_gen` changes with them.
    std::array<u32, 3> mesh_dims{};
    u64 mesh_dims_gen{};
    struct BankCache {
        u64 seq{~u64{0}}; ///< the submission whose arena `ref` points into
        u64 a{};
        u64 b{};
        u64 c{};
        const void* d{};
        u64 e{};
        u64 f{}; ///< the raw coverage ring written into the bank
        u64 g{}; ///< the read-only mirror table written into the bank (generation + 1, 0: none)
        u64 h{}; ///< the bases of the bound texture and sampler pools (bindless_table.h)
        BufferRef ref;
    };
    std::array<BankCache, BIND_STAGES> driver_banks;
    std::array<BankCache, BIND_STAGES> handle_banks;
    /// The push-constant block as the bank shaders read it from (VC::PushConstantBank), graphics
    /// and compute.
    std::array<BankCache, 2> push_banks;
    /// The open pass's framebuffer in the game's pixels (viewports, scissors and the render area
    /// are the game's numbers).
    VkExtent2D framebuffer_extent{};
    /// ... and in the pixels of the images the pass draws into (`framebuffer_extent` scaled when
    /// the pass is scaled: the extent of its framebuffer and render area, the clamp of every
    /// scaled rectangle).
    VkExtent2D host_extent{};
    u32 framebuffer_layers{1};
    /// Resolution scaling: the bound targets draw at the scaled size (DecideTargetScale, at
    /// SetRenderTargets): their textures are the scaled twins. `target_scale` is the factor then.
    bool targets_scaled{false};
    Rescale::Factor target_scale{};
    /// The factor of the pixels the pass's coordinates are in: inactive for a guest-sized pass.
    [[nodiscard]] Rescale::Factor PassFactor() const noexcept {
        return targets_scaled ? target_scale : Rescale::Factor{};
    }
    void DecideTargetScale();
    /// The targets draw at the guest size again (their twins could not be made).
    void RevertTargetScale();
    /// A rectangle of the game's pixels in the pixels the open pass draws at, clamped to them.
    [[nodiscard]] VkRect2D HostRect(VkRect2D guest) const;
    /// The twin texture a draw (or dispatch) samples instead of `texture`, or null: the guest
    /// texture is read (see DrawContext::ScaledTextureFor in nvn_draw_descriptors.cpp).
    [[nodiscard]] std::shared_ptr<Texture> ScaledTextureFor(Texture& texture, bool graphics,
                                                            bool is_image, bool written);
    std::shared_ptr<vk::DescriptorPool> set_pool;
    u32 set_pool_used{};
    /// Per-submission cache of allocated descriptor sets for pipelines without push descriptors,
    /// keyed by set layout + exact descriptor writes/buffers/images (valid for `seq`).
    struct DescSetSlot {
        u64 seq{~u64{0}};
        u64 hash{};
        VkDescriptorSetLayout set_layout{VK_NULL_HANDLE};
        VkDescriptorSet set{VK_NULL_HANDLE};
        std::vector<PassBoundState::WriteSig> writes;
        std::vector<VkDescriptorBufferInfo> buffers;
        std::vector<PassBoundState::ImageSig> images;
    };
    std::vector<DescSetSlot> desc_set_cache;
    const GraphicsPipeline* last_kept{nullptr};
    const ComputePipeline* last_compute_kept{nullptr};
    /// The submission and raw global table generation whose pools were last noted as used.
    u64 raw_noted_seq{~u64{0}};
    u64 raw_noted_generation{~u64{0}};
    /// ... and the read-only mirror table generation whose pools were last noted as used.
    u64 alias_noted_seq{~u64{0}};
    u64 alias_noted_generation{~u64{0}};
    DrawCounters& counters{resources.counters};
    // Scratch reused by every draw (no per-draw allocation in steady state).
    std::vector<VkWriteDescriptorSet> scratch_writes;
    std::vector<VkDescriptorBufferInfo> scratch_buffers;
    std::vector<VkDescriptorImageInfo> scratch_images;
    std::vector<VkBufferView> scratch_views;
    /// Texture buffers bound in the current submission: a view over an arena copy of the guest
    /// bytes each, made once per submission (TexelBufferView).
    struct TexelView {
        u64 address;
        u64 size;
        VkFormat format;
        VkBufferView view;
    };
    std::vector<TexelView> texel_views;
    u64 texel_views_seq{~u64{0}};
    /// The view a texel buffer descriptor binds for `texture` (a texture buffer); null or
    /// unreadable: a view of zeros. VK_NULL_HANDLE when no arena space is left.
    [[nodiscard]] VkBufferView TexelBufferView(const Texture* texture);
    DrawPlan draw_plan; ///< the plan of the draw being prepared (capacity kept between draws)
    std::array<u32, 2> fallback_logged{}; ///< DescReason bits logged once, per bind point

    /// nvn_draw_dump: read-backs of the textures a dumped draw or dispatch bound, recorded once no
    /// pass is open (a draw's at the end of its pass, a dispatch's right after it). Each writes a
    /// TGA next to the dump and a "[nvn] dump" statistics line.
    struct ProbeImage {
        std::shared_ptr<Storage> storage;
        u32 level;
        u32 layer;
        std::string path;
    };
    std::vector<ProbeImage> probe_images;
    /// Notes the bindings of `draw_plan` with at most `max_texels` texels (path `prefix`-b<n>-...).
    void ProbeNoteImages(const std::string& prefix, u64 max_texels);
    void ProbeFlushImages();

    // ---- diagnostics (nvn_draw_debug.cpp; used only when Debug::Tracing()) ----
    struct DebugTargetStat {
        u64 draws{};
        u64 indexed{};
        u64 passes{};
        u64 vertices{};
    };
    struct DebugState {
        u64 frame{~u64{0}};
        bool selected{false};
        /// The effective diagnostics of the selected frame: the environment configuration or, for
        /// a frame a diag request is bound to, what the request asks for.
        u32 draw_log{};
        u32 dump_pass{};
        u32 dump_draw{};
        std::string dump_dir;
        std::filesystem::path report_pass; ///< request only: nvn-pass-<n>.txt
        std::filesystem::path report_draw; ///< request only: nvn-draw-<n>.txt
        u32 logged{};
        u32 draw_index{};
        u32 pass_index{};
        u32 pass_draws{};
        u32 dumped{};
        std::map<std::string, DebugTargetStat> targets;
        std::map<std::string, u64> break_reasons;
    } dbg;
    void DebugSync();
    void DebugFlushFrame();
    [[nodiscard]] std::string DebugTargetKey() const;
    void DebugOnPassBegin();
    void DebugOnPassEnd();
    void DebugOnDraw(u32 mode);
    /// Draw log of an indirect draw: the `IND=` words of its line are guest memory at record time,
    /// before the GPU ran the culling that writes them. This copies the first record (`bytes`) and
    /// the draw count word (when `count` is set) as the GPU holds them when the draw reads them,
    /// for a `[nvn] watch draw-args f<frame> #<draw> ind=<address>: <hex words>` line at
    /// completion (`draw-count` for the count). Outside a render pass, after the barrier that
    /// orders the writes; `#<draw>` is the number the next logged draw gets.
    void DebugReadbackIndirect(u64 address, const BufferRef& args, u32 bytes,
                               const BufferRef* count);
    /// EDEN_NX2_NVN_ONLY_DRAWS: false for a draw of the selected frame that is not listed.
    [[nodiscard]] bool DebugDrawAllowed() const;
    /// Census / skip / dump of the draw for the debug server draw methods
    /// (video_core/nvn_draw_probe.h, nvn_draw_probe.cpp). True: the draw is skipped.
    [[nodiscard]] bool ProbeDraw(const PipelineBase& pipe, u32 mode, u32 element_count,
                                 const IndexRange* index,
                                 const std::array<BufferRef, MAX_VERTEX_STREAMS>& vertex,
                                 u32 used_streams, bool mesh);
    /// nvn_draw_dump of a dispatch whose descriptors are bound (BindDescriptors done):
    /// `arguments` is the resolved indirect argument range (null: direct).
    void ProbeDispatchDump(const VideoCommon::NvnDrawProbe::DumpTicket& ticket,
                           const PipelineBase& pipe, const VC::NvnControl& control,
                           const VC::DispatchPayload& p, const BufferRef* arguments, u64 hash,
                           bool skipped);
    /// The dump section both share: every descriptor binding of `pipe` with what backs it (where
    /// the GPU reads a buffer from and the bytes it reads, as far as the CPU can see them).
    void AppendBindingDump(std::string& text, const PipelineBase& pipe);
    /// `textures` diag request (nvn_draw_debug.cpp): the textures of the draw being prepared are
    /// noted (`DiagNoteTextures`, after BuildDescriptors) and read back when the pass has ended
    /// (`DiagFlushTextures`, from EndPass).
    struct DiagPending {
        std::shared_ptr<Storage> storage; ///< the image the draw samples (a feedback copy, or the texture)
        DiagTextures::Job job;
    };
    std::vector<DiagPending> diag_textures;
    /// BuildDescriptors records stage / slot / sampler in the plan's bindings (set by PrepareDraw
    /// while a texture capture is active).
    bool diag_bindings{false};
    /// While PrepareDraw runs: the entries of `diag_textures` from this index on belong to the draw
    /// being prepared and are not read back yet (~0: none).
    size_t diag_hold{~size_t{0}};
    void DiagNoteTextures();
    void DiagFlushTextures();

    // ---- path diff (nvn_pathdiff.cpp; EDEN_NX2_PATHDIFF, video_core/pathdiff.h) ----
    VideoCommon::PathDiff::Selector pathdiff_selector;
    /// With the descriptors of `pipe` bound (BindDescriptors done, outside a render pass): when
    /// this dispatch/draw is to be captured, builds its capture (ranges, textures), records the
    /// copies of what it reads and returns it; nullopt otherwise. `stage_hashes` by
    /// NVNshaderStage; `indirect`: the arguments the GPU reads (`indirect_size` bytes).
    [[nodiscard]] std::optional<InFlight::PathDiff> PathDiffBegin(
        const PipelineBase& pipe, bool graphics, const std::array<u64, STAGE_COUNT>& stage_hashes,
        const BufferRef* indirect, u32 indirect_size);
    /// After the dispatch/draw (outside a render pass): records the copies of what it may have
    /// written and hands the capture to the flight.
    void PathDiffEnd(InFlight::PathDiff&& pending);
    /// The CPU view of a uniform buffer range: memory with the inline updates applied.
    [[nodiscard]] std::vector<u8> PathDiffUniformBytes(u64 address, u64 size) const;
    void PathDiffTextures(const PipelineBase& pipe, bool graphics,
                          VideoCommon::PathDiff::Capture& capture);
    struct DebugCall {
        u32 count{};
        u32 instances{};
        s64 first{};
        bool indexed{};
        u64 index_address{};
        u64 indirect{}; ///< address of the indirect arguments (0: direct draw)
        u32 args[5]{};  ///< the first indirect command as read from guest memory (tracing only)
        bool args_read{};
        u64 count_address{};
        u32 max_draws{};
        u32 stride{};
        s32 args_backing{-1};
        u32 args_flags{};
    } dbg_call; ///< set by the draw ops for the log

    // ---- GPU profiler (nvn_draw.cpp; idle unless EDEN_NX2_NVN_GPU_PROFILE is set) ----
    /// The submission's timestamps; null when the profiler is off.
    [[nodiscard]] GpuProf::Flight* ProfFlight() const {
        return impl.gpu_profiler ? host->HostFlight().gpu_prof.get() : nullptr;
    }
    /// The open render pass as the profiler sees it.
    struct ProfPass {
        u32 begin{GpuProf::NONE}; ///< timestamp before the pass began
        u32 last{GpuProf::NONE};  ///< latest timestamp inside it (per-draw mode)
        u32 draws{0};
        u32 split{0};             ///< draws of the cost-split class (EDEN_NX2_NVN_COSTSPLIT)
        u64 units{0};             ///< vertices x instances of the pass's direct draws
        GpuProf::Tag tag;         ///< programs of the first draw, target shape
        bool draw_open{false};    ///< a draw has been recorded since `last`
        GpuProf::Tag draw_tag;
        u64 draw_units{0};
    } prof_pass;
    /// The latest BuildDescriptors bound a storage buffer entry with size 0 (the rest of its pool).
    bool prof_zero_size{false};
    /// Vertices x instances of the direct draw being prepared (0 for indirect draws).
    u64 prof_units{0};
    void ProfBeginPass(const RenderPassKey& key, VkExtent2D extent, u32 layers, u32 samples);
    void ProfOnDraw();
    /// Per-draw mode: closes the interval of the draw recorded last (a timestamp after it).
    void ProfCloseDraw(GpuProf::Flight& prof);
    void ProfAfterEndPass(GpuProf::Flight& prof);
    [[nodiscard]] u64 ProfStageHash(u32 stage) const;
    /// GPU command capture (video_core/gpu_capture.h): one DRAW_BINDINGS record per Draw /
    /// DrawIndexed / Dispatch / DrawTexture / DrawTransformFeedback op, whether or not the draw
    /// runs, so the k-th record belongs to the k-th such op of the NVN stream.

    // ---- cost split (nvn_costsplit.cpp; EDEN_NX2_NVN_COSTSPLIT, measurement only) ----
    /// Null when off: the one value the draw path tests.
    CostSplit::Recorder* const costsplit;
    /// Mode of the class draw / dispatch being prepared; Count (the hooks do nothing) outside
    /// PrepareDraw / Dispatch and for work outside the class.
    CostSplit::Mode cs_mode{CostSplit::Mode::Count};
    /// The mode PrepareDraw leaves for its caller's draw command (Base outside the class).
    CostSplit::Mode cs_draw_mode{CostSplit::Mode::Base};
    /// The work is in a slot past its warm-up: it is counted.
    bool cs_measured{false};
    std::chrono::steady_clock::time_point cs_start;
    /// NoBind: class draws in the open pass, this draw's index among them, the layout the first
    /// one pushed, the vertex streams and index buffer bound in the pass.
    u32 cs_pass_draws{0};
    u32 cs_draw_index{0};
    VkPipelineLayout cs_pushed_layout{VK_NULL_HANDLE};
    u32 cs_bound_streams{0};
    bool cs_index_bound{false};
    /// Vram: the ring the copies go to (made at the first copy; host-visible, device-local when
    /// the device has such memory) and its fill.
    vk::Buffer cs_ring;
    u64 cs_ring_used{0};
    bool cs_ring_failed{false};
    /// Ends the cost split of a PrepareDraw / Dispatch on every return.
    struct CostSplitWork {
        DrawContext* context;
        bool dispatch;
        bool recorded{false};
        ~CostSplitWork() {
            if (context != nullptr) {
                context->CostSplitEnd(dispatch, recorded);
            }
        }
    };
    /// Starts a draw (render targets and programs set) / a dispatch of `program`: the class test,
    /// the slot's mode and the replay time.
    DrawContext* CostSplitBeginDraw();
    DrawContext* CostSplitBeginDispatch(u64 program);
    void CostSplitEnd(bool dispatch, bool recorded);
    /// A pass began: NoBind starts over.
    void CostSplitOnPass();
    /// Where a bound range of the class lives (`arena`: a private copy of it).
    void CostSplitNote(CostSplit::Data data, const BufferRef& ref, u64 size, bool arena = false);
    /// Vram (and its per-kind variants, CostSplit::CopiesToRing): a read-only range of an imported pool is copied into the ring and `ref` points at
    /// the copy; false (ref unchanged) for anything else.
    bool CostSplitToRing(CostSplit::Data data, BufferRef& ref, u64 address, u64 size);
    /// A count mode changed a direct draw / dispatch.
    void CostSplitClamped();
    [[nodiscard]] u32 CostSplitInstances(u32 instances) {
        if (cs_draw_mode == CostSplit::Mode::NoInst) [[unlikely]] {
            CostSplitClamped();
        }
        return CostSplit::Instances(cs_draw_mode, instances);
    }
    [[nodiscard]] u32 CostSplitCount(u32 count) {
        if (cs_draw_mode == CostSplit::Mode::Tri3) [[unlikely]] {
            CostSplitClamped();
        }
        return CostSplit::VertexCount(cs_draw_mode, count);
    }
};

} // namespace Vulkan::Nvn
