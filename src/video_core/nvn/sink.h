// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/guest_memory.h"

namespace VideoCommon::Nvn {

/// A memory pool. The GPU address of the pool storage equals the guest VA of that storage.
struct PoolDesc {
    u64 serial;
    u64 guest_address;
    u64 size;
    u32 flags; // NVNmemoryPoolFlags as given by the game (virtual pools have 0x200 set)
    u32 pad;
};

/// View parameters (nxnative `view_t`, all raw NVN enum values). `set_*` tells which fields the
/// game set; unset fields inherit from the texture.
struct TextureViewDesc {
    u8 set_levels;
    u8 set_layers;
    u8 set_format;
    u8 set_target;
    u8 set_swizzle;
    u8 set_dsmode;
    u8 dsmode;
    u8 target;
    u8 base_level;
    u8 nlevels;
    u16 format;
    u16 min_layer;
    u16 nlayers;
    u8 swizzle[4];
    u8 is_image; // registered with RegisterImage: needs storage-image usage
    u8 pad[3];
};

/// A texture or a view object. Fields mirror nxnative `tex_builder_t` (`nvn_texdesc d` plus
/// layout/swizzle/stencil/packaged data), all raw NVN values. For a plain texture `view_of == 0`.
/// When a texture pool registers a texture with a non-default view (or as an image), the recorder
/// creates a second TextureDesc with `view_of` = the texture's serial, a new `serial`, a copy of
/// the texture fields and `view` filled in; descriptor pool snapshots then refer to that serial.
struct TextureDesc {
    s32 target;   // NVNtextureTarget
    s32 format;   // NVNformat
    s32 flags;    // NVNtextureFlags
    s32 samples;  // 0 or 1 = single sampled
    u32 width;
    u32 height;
    u32 depth;
    u32 levels;
    u32 stride;       // row stride of a linear texture, 0 for block-linear
    s32 pkg_block_h;  // packaged texture: log2 block height | log2 block depth << 4, +1; 0 = automatic
    u8 has_layout;    // packaged texture layout supplied
    u8 layout[8];
    u8 set_swizzle;
    u8 stencil;       // TextureBuilderSetDepthStencilMode STENCIL
    u16 swizzle;      // r | g << 3 | b << 6 | a << 9 (0 zero 1 one 2 R 3 G 4 B 5 A)
    u64 pkg_data;     // guest VA of the packaged texture data, 0 = none
    u64 serial;
    u64 pool_serial;  // memory pool holding the storage
    u64 pool_offset;  // byte offset of the storage in that pool
    u64 view_of;      // 0 = a texture; otherwise the serial of the texture this is a view of
    TextureViewDesc view;
};

/// A sampler (nxnative `samp_builder_t` parameters: raw NVN enum values, all below 256).
struct SamplerDesc {
    u8 min_filter;
    u8 mag_filter;
    u8 wrap_s;
    u8 wrap_t;
    u8 wrap_r;
    u8 compare_mode;
    u8 compare_func;
    u8 reduction;
    f32 lod_min;
    f32 lod_max;
    f32 lod_bias;
    f32 max_anisotropy;
    f32 border[4];
    u64 serial;
};

/// NVN program stages (NVNshaderStage; NVN2 adds 6 task and 7 mesh).
inline constexpr size_t PROGRAM_STAGES = 8;

/// Shader program. Index = NVNshaderStage (0 vertex, 1 fragment, 2 geometry, 3 tess control, 4 tess
/// evaluation, 5 compute, 6 task, 7 mesh). `control` is a copy of the guest control section, `code`
/// a copy of the code area (its length comes from the control section).
struct ProgramDesc {
    u64 serial;
    std::vector<u8> control[PROGRAM_STAGES];
    std::vector<u8> code[PROGRAM_STAGES];
    u32 stage_mask;
};

/// An immutable table of serials by descriptor id, shared between the recorder (which hands out
/// the same table again until a registration changes it) and the submissions that captured it.
class IdSnapshot {
public:
    IdSnapshot() = default;
    explicit IdSnapshot(std::shared_ptr<const std::vector<u64>> table_) : table{std::move(table_)} {}
    [[nodiscard]] size_t size() const noexcept {
        return table ? table->size() : 0;
    }
    [[nodiscard]] u64 operator[](size_t id) const {
        return (*table)[id];
    }
    [[nodiscard]] const std::vector<u64>& Values() const {
        static const std::vector<u64> empty;
        return table ? *table : empty;
    }

private:
    std::shared_ptr<const std::vector<u64>> table;
};

/// The uniform buffer bytes a submission binds, as guest memory held them in the submit call
/// (EDEN_NX2_NVN_UBO_SNAPSHOT, video_core/nvn/uniform_snapshot.h). The replay runs behind the
/// game, which rewrites per-object uniform blocks right after it submits: the replay binds and
/// reads these copies instead of the live pool memory.
struct UniformSnapshot {
    struct Range {
        u64 begin;  ///< GPU address (= guest VA) of the first byte
        u64 end;
        u64 offset; ///< of the range's bytes in `bytes`
    };
    /// Sorted by `begin` and disjoint; each lies in one CPU-accessible physical pool and holds
    /// every uniform binding of the submission that starts inside it.
    std::vector<Range> ranges;
    std::vector<u8> bytes;
    /// The newest submission that had completed before the bytes were read: the writes of a later
    /// one may be missing from them.
    u64 completed_seq{};
};

struct Submission {
    u64 seq;   // global, monotonic, shared with PresentRequest
    u32 queue; // recorder queue index
    /// One entry per submitted command handle; CallList records are expanded inline and deferred
    /// handle arrays are resolved. Every list ends with an End record.
    std::vector<std::vector<u64>> lists;
    /// Serials of the objects the lists reference (textures, view objects, programs, syncs,
    /// events, samplers), deduplicated.
    std::vector<u64> retained_serials;
    /// Descriptor registrations at submit time: element i is the serial registered at global
    /// texture (sampler) ID i, 0 = empty. Trimmed after the last non-empty entry.
    IdSnapshot texture_pool_snapshot;
    IdSnapshot sampler_pool_snapshot;
    /// Null when the snapshot is off or the lists bind no uniform buffer it covers. Not const:
    /// the replay layers the submission's inline uniform updates into the bytes in command order.
    std::shared_ptr<UniformSnapshot> uniform_snapshot;
};

/// nvnWindowSetCrop: the part of the presented texture that is shown. Zero width or height (the
/// window's default) is the whole texture.
struct PresentCrop {
    s32 x{};
    s32 y{};
    s32 width{};
    s32 height{};
};

struct PresentRequest {
    u64 seq;
    u32 queue;
    u64 texture_serial;
    u32 window_slot;
    u32 present_interval;
    PresentCrop crop{};
};

/// What a CPU write to an event's memory did (Sink::NotifyEventCpu).
enum class EventCpuChange : u32 {
    Forget = 0, ///< unknown: the replay re-reads the guest word at the event's next use (EventInitialize)
    Write = 1,  ///< nvnEventSignal WRITE: the word is `value`
    Add = 2,    ///< nvnEventSignal ADD: `value` was added
};

/// One MapVirtual request: bytes [virtual_offset, virtual_offset + size) of a virtual pool are
/// backed by bytes [physical_offset, physical_offset + size) of the physical pool
/// `physical_pool_serial`. A serial of 0 unmaps the range.
struct VirtualMapping {
    u64 virtual_offset;
    u64 physical_pool_serial;
    u64 physical_offset;
    u64 size;
};

/// One counter of something the replay could not do (unsupported-<title>.tsv).
struct SinkStat {
    std::string reason;
    u64 count;
};

/// Where the GPU's writes to a guest range stand (Sink::PollGpuWrites).
struct GpuWriteState {
    /// The replay has processed every command enqueued before the marker (so every write of the
    /// submissions made before it is noted).
    bool replay_reached{true};
    /// The newest submission <= max_seq that has not completed yet and writes the range; 0 = none.
    u64 writer_seq{};
    /// The longest a guest thread may wait for it, in milliseconds (the sink's setting).
    u32 wait_limit_ms{};
};

/// Why a texture object the sink still has is alive (Sink::NoteTextureHolders, diagnostics only).
/// In order of strength: a storage several textures reference is held by the strongest of them.
enum class TextureHolder : u8 {
    Reinitialized = 0, ///< the game initialized its object again without a finalize: never destroyed
    SyntheticList = 1, ///< finalized; recorded lists at synthetic command memory reference it
    GuestList = 2,     ///< finalized; a list in the game's command memory references it
    Retiring = 3,      ///< finalized; waits for its submissions (or an open recording)
    Game = 4,          ///< the game's own live texture (not listed in a report)
};
inline constexpr size_t TEXTURE_HOLDER_KINDS = 5;

/// Every texture and view object serial whose holder is not TextureHolder::Game, sorted.
struct TextureHolderReport {
    std::vector<std::pair<u64, TextureHolder>> serials;
};

/// Implemented by the Vulkan backend. Every call except Wait and SetCompletionCallback is made
/// from one thread at a time, in global order.
class Sink {
public:
    virtual ~Sink() = default;

    virtual void CreatePool(const PoolDesc&) = 0;
    virtual void CreateTexture(const TextureDesc&) = 0;
    virtual void CreateSampler(const SamplerDesc&) = 0;
    virtual void CreateProgram(const ProgramDesc&) = 0;
    /// Takes effect after every submission that retained `serial` completes.
    virtual void Destroy(u64 serial) = 0;
    /// Called in global submit order.
    virtual void Submit(Submission&&) = 0;
    /// Ordered with Submit.
    virtual void Present(const PresentRequest&) = 0;
    /// GPU completion of `seq`.
    virtual bool Wait(u64 seq, std::chrono::nanoseconds timeout) = 0;
    /// The callback is called in seq order, once per Submit/Present seq, after the GPU finished it.
    virtual void SetCompletionCallback(std::function<void(u64 seq)>) = 0;

    /// Called once, from a sink thread, when the host GPU can no longer run the game's work (the
    /// Vulkan device was lost). The reason is for the log. From then on the sink completes every
    /// submission without executing it, so no guest wait hangs; the owner is expected to stop the
    /// emulation. The callback must not block on the sink. SetFailureCallback({}) waits for a
    /// running callback, like SetCompletionCallback.
    virtual void SetFailureCallback(std::function<void(std::string reason)> callback) {
        static_cast<void>(callback);
    }

    /// Called once by the owner before the first CreatePool so the sink can reach guest memory
    /// (pool host pointers, texture contents, readback writes). Optional for test sinks.
    virtual void SetGuestMemory(GuestMemory* memory) {
        static_cast<void>(memory);
    }

    /// The device's window origin (NVNwindowOriginMode: 0 lower left, 1 upper left) and depth mode
    /// (NVNdepthMode: 0 near is -w, 1 near is 0). Set by the game before it draws; the replay reads
    /// them when it builds pipelines and viewports.
    virtual void SetDeviceModes(u32 window_origin, u32 depth_mode) {
        static_cast<void>(window_origin);
        static_cast<void>(depth_mode);
    }

    /// The title's program id (before the first CreatePool): the replay opens its per-title
    /// pipeline cache.
    virtual void SetTitleId(u64 title_id) {
        static_cast<void>(title_id);
    }

    /// Appends the counters of unsupported cases (every reason, zero counts included).
    virtual void CollectStats(std::vector<SinkStat>& out) const {
        static_cast<void>(out);
    }

    /// A virtual pool (PoolDesc flags & 0x200, whose `guest_address` is a synthetic GPU address
    /// the recorder assigned) got new mappings. Ordered with Submit.
    virtual void MapVirtual(u64 pool_serial, std::vector<VirtualMapping>&& mappings) {
        static_cast<void>(pool_serial);
        static_cast<void>(mappings);
    }

    /// The guest CPU wrote [guest_address, guest_address + size) (MemoryPool/Buffer
    /// FlushMappedRange). Only a hint: the replay detects CPU changes itself, this makes it look
    /// at the affected textures at the next use. Ordered with Submit.
    virtual void NotifyCpuWrite(u64 guest_address, u64 size) {
        static_cast<void>(guest_address);
        static_cast<void>(size);
    }

    /// A host-side agent (a system service: the video decoder writing a frame into a decoder
    /// image) wrote [guest_address, guest_address + size) of guest memory, as a DMA engine writes
    /// pool memory on the console: the GPU's next read of the range sees the new bytes, whatever
    /// the pool's CPU access, and the bytes replace what the GPU last wrote to a texture that lies
    /// inside the range. Ordered with Submit.
    virtual void NotifyHostWrite(u64 guest_address, u64 size) {
        static_cast<void>(guest_address);
        static_cast<void>(size);
    }

    /// The guest CPU changed the word of the event at `address` (nvnEventSignal, or a new event
    /// object created over that memory): the replay's model of the event follows it. Ordered with
    /// Submit. Not optional: a sink that replays event waits must see CPU signals.
    virtual void NotifyEventCpu(u64 address, EventCpuChange change, u32 value) = 0;

    /// The guest CPU wrote the storage of texture `serial` itself (WriteTexels): the next use
    /// uploads it again even if the GPU last wrote it. Ordered with Submit.
    virtual void NotifyTextureWritten(u64 texture_serial) {
        static_cast<void>(texture_serial);
    }

    /// Blocks until guest memory holds the texture's current contents (the GPU wrote it: the
    /// replay reads it back and swizzles it into the storage), then returns true. Returns true at
    /// once when the guest bytes are already current. False when the readback failed or the
    /// texture is unknown. Used by ReadTexels; never called from the replay or completion thread.
    virtual bool FlushTextureToGuest(u64 texture_serial) {
        static_cast<void>(texture_serial);
        return true;
    }

    /// EDEN_NX2_TRY_INVALIDATE_WRITEBACK (nvnMemoryPoolInvalidateMappedRange): a guest thread that
    /// is about to read GPU results asks where the GPU's writes to the range stand. GpuWriteMarker
    /// returns 0 when the sink does not track them; else a marker of the commands enqueued so far.
    /// PollGpuWrites then says whether the replay processed them (every write of the submissions
    /// up to `max_seq` is noted) and the newest of those submissions still unfinished that writes
    /// [guest_address, guest_address + size). Both are lock free reads, callable from any thread.
    [[nodiscard]] virtual u64 GpuWriteMarker() {
        return 0;
    }
    [[nodiscard]] virtual GpuWriteState PollGpuWrites(u64 marker, u64 guest_address, u64 size,
                                                      u64 max_seq) {
        static_cast<void>(marker);
        static_cast<void>(guest_address);
        static_cast<void>(size);
        static_cast<void>(max_seq);
        return {};
    }

    /// One line describing what the replay is doing right now (diagnostics: stuck work).
    [[nodiscard]] virtual std::string DescribeState() const {
        return {};
    }

    /// The memory the replay holds, for the `[nvn] memory:` status line. Any thread.
    [[nodiscard]] virtual std::string DescribeMemory() const {
        return {};
    }

    /// The recorder's view of who keeps textures alive (diagnostics: the storage census and the
    /// memory line split the textures' storages by it). Ordered with Submit.
    virtual void NoteTextureHolders(TextureHolderReport&& report) {
        static_cast<void>(report);
    }
};

} // namespace VideoCommon::Nvn
