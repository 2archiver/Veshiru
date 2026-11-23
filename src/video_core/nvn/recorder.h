// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/nvn/uniform_snapshot.h"

namespace VideoCommon::Nvn {

/// The NVN entry point the calling thread is executing (set by the HLE bridge around a body), so
/// the Recorder's diagnostics can name the game call that produced them. Null outside a call.
/// Thread-local; a diagnostic only, never read for behaviour. Set around each HLE body; a body
/// that blocks the guest thread can see another guest thread's call name here while it sleeps.
inline thread_local const char* g_current_nvn_call = nullptr;

/// Counters of things that went wrong or were skipped. All are cumulative.
struct RecorderStats {
    u64 tag_mismatch;            ///< object lookups whose guest tag magic/kind did not match
    u64 stale_serial;            ///< tag valid but the host object is gone
    u64 bad_guest_read;          ///< GuestMemory::Read failures (the call recorded nothing)
    u64 bad_guest_write;         ///< GuestMemory::Write failures
    u64 bad_argument;            ///< out of range counts/ids, commands outside a recording, ...
    u64 callbacks_skipped;       ///< command memory ran out and the game's callback was not invoked
    u64 synthetic_segments;      ///< synthetic command memory segments handed out
    u64 unknown_call_handle;     ///< CallList / submitted handles with no recorded list
    u64 call_depth_exceeded;     ///< CallList nesting deeper than 16 at submit
    u64 deferred_read_failed;    ///< deferred handle/range arrays that could not be read at submit
    u64 program_code_unavailable; ///< programs created with a stage whose code area was unreadable
    u64 stage_fallback;          ///< programs whose stages were assigned by position
    u64 reinitialized_objects;   ///< Initialize over a live tagged object (new serial)
    u64 submissions;
    u64 presents;
    u64 lists_submitted;
    u64 records_recorded;
    u64 records_dropped;
    u64 completion_actions;      ///< event/report writes applied at completion
    u64 retired_objects;         ///< objects passed to Sink::Destroy
    u64 synthetic_evictions;     ///< synthetic lists dropped to stay under the cap (least recently used)
    u64 finalized_pool_lists;    ///< lists dropped because their memory pool was finalized
    u64 synthetic_lists;         ///< recorded lists at synthetic command memory held now
    u64 synthetic_list_bytes;    ///< ... and the host memory they take (words, references, actions)
    u64 synthetic_cap_evictions; ///< synthetic lists dropped by the byte cap (SetSyntheticListCap)
};

/// Textures (not view objects) by who keeps them alive, at a PublishTextureHolders call; the
/// game's own live textures are not counted.
struct TextureHolderCounts {
    u64 retiring{};       ///< finalized, waiting for submissions or an open recording
    u64 guest_lists{};    ///< finalized, a list in the game's command memory references it
    u64 synthetic_lists{}; ///< finalized, only lists at synthetic command memory reference it
    u64 reinitialized{};  ///< initialized again without a finalize: never destroyed
};

/// Synthetic lists unused for fewer presents than this are never dropped by the byte cap.
inline constexpr u64 DEFAULT_SYNTHETIC_LIST_IDLE_PRESENTS = 600;

struct PoolInfo {
    u64 serial;
    u64 address; ///< GPU address = guest VA of the storage
    u64 size;
    u32 flags;
};

struct DescriptorPoolInfo {
    u64 serial;
    u64 pool_serial;
    u64 pool_offset;
    u32 count;
    u32 base; ///< first id of this pool in the global (snapshot) id space
};

struct BufferInfo {
    u64 serial;
    u64 pool_serial;
    u64 address; ///< pool address + offset read from the guest object
    u64 offset;
    u64 size;
};

/// `fenced`: a queue or command buffer fenced the sync (`fence_seq` is that submission).
/// `signaled`: the fence completed, or the sync was signalled without a fence (SyncSetSignaled).
/// A sync that is neither fenced nor signalled was never given anything to wait for.
struct SyncState {
    bool fenced;
    bool signaled;
    u64 fence_seq;
};

enum class SyncWaitResult { AlreadySignaled, ConditionSatisfied, TimeoutExpired, Failed };

/// One shader stage given to ProgramSetShaders (NVNshaderData plus an optional explicit stage).
struct ShaderStageInput {
    u64 data_address;
    u64 control_address;
    s32 stage = -1; ///< NVNshaderStage, or -1 to derive it from the control section / position
};

/// Host-side object model and command recording for the NVN HLE.
///
/// Every `*_va` argument is the guest virtual address of an NVN object. Non-builder objects carry
/// a GuestTag at offset 0 written by the *Initialize calls; lookups resolve it to a host object by
/// serial. Pointer-like arguments (`values_va`, `data_va`, ...) are guest addresses read through
/// GuestMemory at call time, so data whose lifetime ends with the call is copied into the record.
///
/// Threading: command buffer methods are single-threaded per command buffer (as in NVN). Queue,
/// pool and object methods may be called from any thread. Sink calls other than Wait are
/// serialized by one mutex; the completion callback never takes it.
class Recorder {
public:
    Recorder(GuestMemory& memory, Sink& sink);
    ~Recorder();
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    [[nodiscard]] RecorderStats Stats() const;
    /// Forwards the device's window origin / depth mode to the sink (Sink::SetDeviceModes).
    void SetDeviceModes(u32 window_origin, u32 depth_mode);
    /// Copies consecutive plain records together at submit; set before the first submission.
    void SetCommandListBatching(bool enabled);
    /// EDEN_NX2_NVN_UBO_SNAPSHOT: every later submission carries the CPU-written uniform bytes it
    /// binds as of its submit call (Submission::uniform_snapshot, uniform_snapshot.h). Off by
    /// default; set before the first submission.
    void SetUniformSnapshot(UniformSnapshotMode mode);
    [[nodiscard]] bool UniformSnapshotEnabled() const;
    /// EDEN_NX2_NVN_SYNTH_LIST_MIB: byte cap of the lists recorded at synthetic command memory (0,
    /// the default: none). Beyond it the least recently used ones that no submission or call used
    /// for `idle_presents` presents and that no other list calls are dropped, with their
    /// references. Not safe by construction: the game holds the handle of such a list and could
    /// submit it again later (DESIGN.md "Synthetic command lists").
    void SetSyntheticListCap(u64 bytes, u64 idle_presents = DEFAULT_SYNTHETIC_LIST_IDLE_PRESENTS);
    /// Diagnostics: tells the sink which textures the game no longer names are still alive and
    /// why (Sink::NoteTextureHolders) and returns the counts. Any thread; walks every list.
    TextureHolderCounts PublishTextureHolders();

    // ---- Device / queue -------------------------------------------------------------------
    bool DeviceInitialize(u64 device_va);
    void DeviceFinalize(u64 device_va);

    bool QueueInitialize(u64 queue_va);
    void QueueFinalize(u64 queue_va);
    /// Recorder queue index used in Submission::queue, nullopt for an unknown queue.
    [[nodiscard]] std::optional<u32> QueueIndex(u64 queue_va);
    /// Submits the lists behind `handles` in order. Returns the sequence number, 0 on failure.
    u64 QueueSubmitCommands(u64 queue_va, std::span<const u64> handles);
    /// Same, reading `count` handles from guest memory at `handles_va`.
    u64 QueueSubmitCommands(u64 queue_va, s32 count, u64 handles_va);
    /// Fences `sync_va` after all work submitted on the queue (submits a FenceSync record).
    u64 QueueFenceSync(u64 queue_va, u64 sync_va, u32 condition, u32 flags);
    /// Makes later work on the queue wait for `sync_va` (submits a WaitSync record).
    u64 QueueWaitSync(u64 queue_va, u64 sync_va);
    /// Sequence number of the last submission/present on the queue (0 = none).
    [[nodiscard]] u64 QueueLastSeq(u64 queue_va);
    /// Blocks until everything submitted on the queue completed and its guest writes landed.
    bool QueueFinish(u64 queue_va, std::chrono::nanoseconds timeout);
    /// Presents window texture `slot` on the queue. Returns the sequence number, 0 on failure.
    u64 QueuePresentTexture(u64 queue_va, u64 window_va, u32 slot);

    // ---- Completion -----------------------------------------------------------------------
    [[nodiscard]] u64 LastSeq() const;      ///< last assigned sequence number
    [[nodiscard]] u64 CompletedSeq() const; ///< all seq <= this completed and their writes landed
    /// Blocks until CompletedSeq() >= seq. Prefer this to Sink::Wait for guest-visible waits: it
    /// also guarantees the completion actions (event/report writes) were applied.
    bool WaitForSeq(u64 seq, std::chrono::nanoseconds timeout);
    /// Called on the completion thread after every completed seq (CompletedSeq() already updated,
    /// guest writes applied). Replaces the previous observer; pass {} to remove it. The observer
    /// must not call back into the Recorder's blocking methods.
    void SetCompletionObserver(std::function<void(u64 seq)> observer);
    /// Calls Sink::Destroy for objects that became unreachable. Done automatically on every
    /// submit/present/create; exposed for tests and shutdown.
    void FlushDestroys();

    // ---- Memory pools ---------------------------------------------------------------------
    bool MemoryPoolInitialize(u64 pool_va, u64 storage_va, u64 size, u32 flags);
    void MemoryPoolFinalize(u64 pool_va);
    [[nodiscard]] std::optional<PoolInfo> GetMemoryPool(u64 pool_va);
    /// GetMemoryPool for the NVN getters (GetFlags, GetSize, Map, GetBufferAddress, the mapped
    /// range calls). They only read fields of the guest's pool object and never check that it
    /// was initialised, so a zeroed object (pool never initialised, or finalized) answers
    /// 0/empty on hardware without complaint: that miss is silent here too. Any other tag that
    /// does not match still warns.
    [[nodiscard]] std::optional<PoolInfo> QueryMemoryPool(u64 pool_va);
    /// One nvnMemoryPoolMapVirtual request: `physical_pool_va` is the guest address of the
    /// physical NVNmemoryPool, 0 unmaps the virtual range.
    struct VirtualMapRequest {
        u64 physical_pool_va;
        u64 physical_offset;
        u64 virtual_offset;
        u64 size;
    };
    /// Maps ranges of physical pools into the virtual pool `pool_va`. False when the pool is not a
    /// virtual pool, a physical pool is unknown, or a range leaves its pool.
    bool MemoryPoolMapVirtual(u64 pool_va, std::span<const VirtualMapRequest> requests);
    /// The CPU wrote [address, address + size) of guest memory (a MemoryPool/Buffer
    /// FlushMappedRange). Forwarded to the sink in submit order.
    void NotifyCpuWrite(u64 address, u64 size);
    /// A host-side service wrote [address, address + size) of guest memory (Sink::NotifyHostWrite).
    /// Callable from any thread; forwarded to the sink in submit order.
    void NotifyHostWrite(u64 address, u64 size);
    /// EDEN_NX2_TRY_INVALIDATE_WRITEBACK (Sink::GpuWriteMarker / PollGpuWrites): 0 when the sink
    /// does not track the GPU's writes. Read LastSeq(), then take the marker (every submission up
    /// to that LastSeq is enqueued before it), poll until `replay_reached`, and wait for
    /// CompletedSeq() >= `writer_seq` (the GPU's unfinished writes of those submissions).
    [[nodiscard]] u64 GpuWriteMarker();
    [[nodiscard]] GpuWriteState PollGpuWrites(u64 marker, u64 address, u64 size, u64 max_seq);
    /// The CPU wrote texture `texture_va`'s storage directly (WriteTexels).
    void NotifyTextureWritten(u64 texture_va);
    /// Blocks until guest memory holds the contents of texture `texture_va` (see
    /// Sink::FlushTextureToGuest). True when the bytes are current.
    bool FlushTextureToGuest(u64 texture_va);

    // ---- Texture / sampler pools and registrations -------------------------------------------
    bool TexturePoolInitialize(u64 pool_va_of_tp, u64 memory_pool_va, s64 offset, s32 count);
    void TexturePoolFinalize(u64 texture_pool_va);
    [[nodiscard]] std::optional<DescriptorPoolInfo> GetTexturePool(u64 texture_pool_va);
    /// view_va may be 0. RegisterImage declares storage-image use.
    void RegisterTexture(u64 texture_pool_va, s32 id, u64 texture_va, u64 view_va);
    void RegisterImage(u64 texture_pool_va, s32 id, u64 texture_va, u64 view_va);

    bool SamplerPoolInitialize(u64 sampler_pool_va, u64 memory_pool_va, s64 offset, s32 count);
    void SamplerPoolFinalize(u64 sampler_pool_va);
    [[nodiscard]] std::optional<DescriptorPoolInfo> GetSamplerPool(u64 sampler_pool_va);
    void RegisterSampler(u64 sampler_pool_va, s32 id, u64 sampler_va);
    /// Registers parameters directly (RegisterSamplerBuilder); the serial in `desc` is ignored.
    void RegisterSamplerParams(u64 sampler_pool_va, s32 id, const SamplerDesc& desc);

    // ---- Textures, samplers, buffers --------------------------------------------------------------
    /// Creates the host texture. desc.serial/pool_serial/pool_offset/view_of/view are filled in
    /// here. Re-initializing a live address creates a new serial.
    bool TextureInitialize(u64 texture_va, const TextureDesc& desc, u64 memory_pool_va,
                           u64 pool_offset);
    void TextureFinalize(u64 texture_va);
    [[nodiscard]] std::optional<TextureDesc> GetTexture(u64 texture_va);

    bool SamplerInitialize(u64 sampler_va, const SamplerDesc& desc);
    void SamplerFinalize(u64 sampler_va);
    [[nodiscard]] std::optional<SamplerDesc> GetSampler(u64 sampler_va);

    /// Writes the tag, the offset (+0x18) and the size (+0x20) into the guest object.
    bool BufferInitialize(u64 buffer_va, u64 memory_pool_va, u64 offset, u64 size);
    void BufferFinalize(u64 buffer_va);
    /// Reads offset/size from the guest object (the game may have rewritten them).
    [[nodiscard]] std::optional<BufferInfo> GetBuffer(u64 buffer_va);

    // ---- Programs ---------------------------------------------------------------------------------
    bool ProgramInitialize(u64 program_va);
    void ProgramFinalize(u64 program_va);
    /// Copies the control sections and code areas. A new serial per call (programs are immutable).
    bool ProgramSetShaders(u64 program_va, std::span<const ShaderStageInput> stages);
    /// Reads `count` NVNshaderData entries at `shader_data_va`.
    bool ProgramSetShaders(u64 program_va, s32 count, u64 shader_data_va);

    // ---- Sync / events ----------------------------------------------------------------------------
    bool SyncInitialize(u64 sync_va);
    void SyncFinalize(u64 sync_va);
    /// Makes `sync_va` a signalled sync without a fence (the sync nvnWindowAcquireTexture hands
    /// back), tagging it first when it is not a live sync. A later fence replaces the state.
    bool SyncSetSignaled(u64 sync_va);
    [[nodiscard]] std::optional<SyncState> GetSyncState(u64 sync_va);
    /// NVN semantics: Failed for a sync that is neither fenced nor signalled.
    SyncWaitResult SyncWait(u64 sync_va, std::chrono::nanoseconds timeout);

    bool EventInitialize(u64 event_va, u64 memory_pool_va, s64 offset);
    void EventFinalize(u64 event_va);
    [[nodiscard]] std::optional<u32> EventGetValue(u64 event_va);
    /// Immediate CPU-side signal: mode 0 writes `value`, mode 1 adds it.
    void EventSignal(u64 event_va, u32 mode, u32 value);

    // ---- Window -----------------------------------------------------------------------------------
    /// `texture_vas` are the (initialized) window textures, at most 4.
    bool WindowInitialize(u64 window_va, std::span<const u64> texture_vas, u32 present_interval);
    void WindowFinalize(u64 window_va);
    void WindowSetPresentInterval(u64 window_va, u32 interval);
    /// nvnWindowSetCrop: applies to the presents after it.
    void WindowSetCrop(u64 window_va, PresentCrop crop);
    /// Sequence number of the last present of `slot` (0 = never presented).
    [[nodiscard]] u64 WindowSlotSeq(u64 window_va, u32 slot);

    // ---- Command buffers: lifetime and memory ---------------------------------------------------
    bool CommandBufferInitialize(u64 cmd_va);
    void CommandBufferFinalize(u64 cmd_va);
    /// The callback is stored but never invoked (see callbacks_skipped).
    void SetMemoryCallback(u64 cmd_va, u64 callback_va);
    void SetMemoryCallbackData(u64 cmd_va, u64 data_va);
    /// The stored callback / data pointers (0 when unset or the command buffer is unknown).
    [[nodiscard]] u64 GetMemoryCallback(u64 cmd_va);
    [[nodiscard]] u64 GetMemoryCallbackData(u64 cmd_va);
    /// `pool_va` 0 or n < 64 hands out a synthetic segment.
    void AddCommandMemory(u64 cmd_va, u64 memory_pool_va, s64 offset, u64 size);
    void AddControlMemory(u64 cmd_va, u64 size);
    [[nodiscard]] u64 GetCommandMemoryUsed(u64 cmd_va);
    [[nodiscard]] u64 GetCommandMemoryFree(u64 cmd_va);
    [[nodiscard]] u64 GetCommandMemorySize(u64 cmd_va);
    [[nodiscard]] u64 GetControlMemoryFree(u64 cmd_va);
    [[nodiscard]] u64 GetControlMemorySize(u64 cmd_va);
    void BeginRecording(u64 cmd_va);
    /// Returns the command handle (virtual command memory address of the first record).
    u64 EndRecording(u64 cmd_va);
    [[nodiscard]] bool IsRecording(u64 cmd_va);
    /// CallCommands / CopyCommands: record a call of each list handle.
    void CallCommands(u64 cmd_va, std::span<const u64> handles);
    void CallCommands(u64 cmd_va, s32 count, u64 handles_va);

    // ---- Command buffers: transfer and synchronization -------------------------------------------
    void ClearTexture(u64 cmd_va, u64 texture_va, u64 view_va, u64 region_va, u64 value_va,
                      u32 mask, bool is_integer);
    void ClearBuffer(u64 cmd_va, u64 address, u64 size, u32 value);
    void CopyBufferToBuffer(u64 cmd_va, u64 src, u64 dst, u64 size, u32 flags);
    void CopyBufferToTexture(u64 cmd_va, u64 src_address, u64 texture_va, u64 view_va,
                             u64 region_va, u32 flags);
    void CopyTextureToBuffer(u64 cmd_va, u64 texture_va, u64 view_va, u64 region_va,
                             u64 dst_address, u32 flags);
    void CopyTextureToTexture(u64 cmd_va, u64 src_va, u64 src_view_va, u64 src_region_va,
                              u64 dst_va, u64 dst_view_va, u64 dst_region_va, u32 flags);
    /// Multisample resolve of the whole textures (a CopyTextureToTexture with flag 0x40000000).
    void Downsample(u64 cmd_va, u64 src_va, u64 dst_va);
    void SetCopyRowStride(u64 cmd_va, s64 stride);
    void SetCopyImageStride(u64 cmd_va, s64 stride);
    [[nodiscard]] s64 GetCopyRowStride(u64 cmd_va);
    [[nodiscard]] s64 GetCopyImageStride(u64 cmd_va);
    void UpdateUniformBuffer(u64 cmd_va, u64 buffer_address, u64 offset, u64 size, u64 data_va);
    void FenceSync(u64 cmd_va, u64 sync_va, u32 condition, u32 flags);
    void WaitSync(u64 cmd_va, u64 sync_va);
    void SignalEvent(u64 cmd_va, u64 event_va, u32 mode, u32 location, u32 flags, u32 value);
    /// Records the wait: the replay stalls the queue's stream there until the event's word
    /// satisfies it. mode: 0 EQUAL, 1 GEQUAL_WRAP.
    void WaitEvent(u64 cmd_va, u64 event_va, u32 mode, u32 value);
    void ReportValue(u64 cmd_va, u32 value, u64 address);
    void ReportCounter(u64 cmd_va, u32 counter_type, u64 address);
    void ResetCounter(u64 cmd_va, u32 counter_type);
    void Barrier(u64 cmd_va, u32 bits);

    // ---- Command buffers: render state ---------------------------------------------------------------
    /// `textures_va` / `views_va`: arrays of `count` guest pointers (views_va may be 0).
    void SetRenderTargets(u64 cmd_va, s32 count, u64 textures_va, u64 views_va, u64 depth_va,
                          u64 depth_view_va);
    /// `values_va`: 4 floats (or 4 ints/uints when is_integer).
    void ClearColor(u64 cmd_va, s32 index, u64 values_va, u32 mask, bool is_integer);
    void ClearDepthStencil(u64 cmd_va, f32 depth, u32 depth_mask, s32 stencil, u32 stencil_mask);
    void SetViewports(u64 cmd_va, s32 first, s32 count, u64 values_va); ///< 4 floats each
    void SetScissors(u64 cmd_va, s32 first, s32 count, u64 values_va);  ///< 4 ints each
    void SetDepthRanges(u64 cmd_va, s32 first, s32 count, u64 values_va); ///< 2 floats each
    void SetViewportSwizzles(u64 cmd_va, s32 first, s32 count, u64 values_va); ///< 4 bytes each
    void BindProgram(u64 cmd_va, u64 program_va, u32 stages);
    void BindState(u64 cmd_va, StateKind kind, u64 state_va);
    void BindVertexAttribState(u64 cmd_va, s32 count, u64 states_va);
    void BindVertexStreamState(u64 cmd_va, s32 count, u64 states_va);
    void BindVertexBuffers(u64 cmd_va, s32 first, s32 count, u64 ranges_va);
    void BindVertexBuffer(u64 cmd_va, s32 index, u64 address, u64 size);
    void BindUniformBuffers(u64 cmd_va, u32 stage, s32 first, s32 count, u64 ranges_va,
                            bool deferred);
    void BindUniformBuffer(u64 cmd_va, u32 stage, s32 index, u64 address, u64 size);
    void BindStorageBuffers(u64 cmd_va, u32 stage, s32 first, s32 count, u64 ranges_va,
                            bool deferred);
    void BindStorageBuffer(u64 cmd_va, u32 stage, s32 index, u64 address, u64 size);
    /// `kind` selects the op: BindTextures, BindSeparateTextures, BindSeparateSamplers,
    /// BindImages. Handles are read from `handles_va` (count * 8 bytes), at submit if deferred.
    void BindHandles(u64 cmd_va, Op kind, u32 stage, s32 first, s32 count, u64 handles_va,
                     bool deferred);
    void BindHandle(u64 cmd_va, Op kind, u32 stage, s32 index, u64 handle);
    void SetTexturePool(u64 cmd_va, u64 texture_pool_va);
    void SetSamplerPool(u64 cmd_va, u64 sampler_pool_va);
    void SetStencil(u64 cmd_va, s32 face, u32 which, s32 value);
    void SetPolygonOffsetClamp(u64 cmd_va, f32 factor, f32 units, f32 clamp);
    void SetBlendColor(u64 cmd_va, u64 values_va); ///< 4 floats
    void SetDepthClamp(u64 cmd_va, bool enable);
    void SetAlphaRef(u64 cmd_va, f32 reference);
    void SetPrimitiveRestart(u64 cmd_va, bool enable, u32 index);
    void SetRasterizerDiscard(u64 cmd_va, bool enable);
    void SetSampleMask(u64 cmd_va, u32 mask);
    /// nvnCommandBufferSetProvokingVertexMode: the SDK writes the mode into SET_PROVOKING_VERTEX,
    /// whose bit 0 selects the vertex (0 first, 1 last); recorded as that bit.
    void SetProvokingVertexMode(u64 cmd_va, u32 mode);
    void SetPointSize(u64 cmd_va, f32 size);
    void SetLineWidth(u64 cmd_va, f32 width);
    void SetRenderEnable(u64 cmd_va, bool enable);
    void SetRenderEnableConditional(u64 cmd_va, u32 mode, u64 address);
    void PushConstants(u64 cmd_va, bool compute, u64 word, u64 offset, u64 size, u64 data_va);
    void BindTransformFeedbackBuffer(u64 cmd_va, s32 index, u64 address, u64 size);
    void TransformFeedbackState(u64 cmd_va, u32 what, u64 address);
    void DrawTransformFeedback(u64 cmd_va, u32 mode, u64 address);

    /// What nvnCommandBufferEvaluateDlss decoded from the game's NVNdlssNetwork and
    /// NVNdlssParameters objects, with the SDK's rules already applied (EvaluateDlssPayload,
    /// record.h). Texture and view fields are guest addresses (0: none).
    struct DlssEvaluateArgs {
        u64 input_texture, input_view;
        u64 output_texture, output_view;
        u64 motion_vectors_texture, motion_vectors_view;
        u64 depth_texture, depth_view;
        u64 bias_texture, bias_view;
        u64 exposure_texture, exposure_view;
        DlssRect input_base, depth_base, motion_vector_base, bias_base, output_base;
        s32 input_width, input_height;
        s32 output_width, output_height;
        f32 motion_scale[2];
        f32 jitter[2];
        f32 pre_exposure;
        bool history_reset;
        u32 flags, method, preset;
        u32 network;
    };
    /// Records the upscale of the input into the output. Dropped when the input or the output is
    /// not a live NVNtexture of this device.
    void EvaluateDlss(u64 cmd_va, const DlssEvaluateArgs& args);

    // ---- Command buffers: draws and dispatches ------------------------------------------------------
    void DrawArrays(u64 cmd_va, u32 mode, s32 first, u32 count, u32 instance_count,
                    u32 base_instance);
    void DrawArraysIndirect(u64 cmd_va, u32 mode, u64 indirect);
    void MultiDrawArraysIndirectCount(u64 cmd_va, u32 mode, u64 indirect, u64 count_address,
                                      u32 max_draws, s64 stride);
    void DrawElements(u64 cmd_va, u32 mode, u32 index_type, u32 count, u64 index_address,
                      s32 base_vertex, u32 base_instance, u32 instance_count);
    void DrawElementsIndirect(u64 cmd_va, u32 mode, u32 index_type, u64 index_address,
                              u64 indirect);
    void MultiDrawElementsIndirectCount(u64 cmd_va, u32 mode, u32 index_type, u64 index_address,
                                        u64 indirect, u64 count_address, u32 max_draws,
                                        s64 stride);
    void DrawTexture(u64 cmd_va, u64 handle, u64 dst_region_va, u64 src_region_va);
    void DispatchCompute(u64 cmd_va, u32 x, u32 y, u32 z, u64 caller);
    void DispatchComputeIndirect(u64 cmd_va, u64 indirect, u64 caller);
    /// Task/mesh draws (Op::DrawMeshTasks, DrawMeshTasksPayload). `dims` 1 or 3. Direct: `first`
    /// (1D only) and the workgroup counts. Indirect: the argument address; `count_address` 0 and
    /// `max_draws` 1 for the single-draw entries; `stride` the bytes between argument records.
    void DrawMeshTasks(u64 cmd_va, u32 dims, u32 first, u32 x, u32 y, u32 z);
    void DrawMeshTasksIndirect(u64 cmd_va, u32 dims, u64 indirect, u64 count_address,
                               u32 max_draws, u32 stride);

    /// Escape hatch for ops added later: appends a record with the given payload bytes.
    void EmitRaw(u64 cmd_va, u16 op, u16 flags, const void* payload, u32 payload_bytes);

    /// Implementation state (recorder_impl.h). Public only so the implementation files can name it.
    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCommon::Nvn
