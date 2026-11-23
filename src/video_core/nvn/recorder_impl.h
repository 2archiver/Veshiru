// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Internal to the recorder (recorder.cpp, recorder_commands.cpp). Not part of the public API.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/recorder.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"
#include "video_core/nvn/uniform_snapshot.h"

namespace VideoCommon::Nvn {

/// Synthetic command memory lives far above any guest address.
inline constexpr u64 SYNTHETIC_BASE = 0x7E00'0000'0000'0000ULL;
/// GPU addresses given to virtual memory pools (they have no guest storage), above every guest
/// address and below the synthetic command memory.
inline constexpr u64 VIRTUAL_POOL_BASE = 0x7D00'0000'0000'0000ULL;
inline constexpr u64 VIRTUAL_POOL_ALIGNMENT = 2 * 1024 * 1024;
inline constexpr u64 SYNTHETIC_SEGMENT_SIZE = 256 * 1024;
/// Room kept after every record but the End (nxnative OP_RESERVE): one JUMP.
inline constexpr u32 COMMAND_RESERVE = 16;
inline constexpr u32 MAX_CALL_DEPTH = 16;
inline constexpr size_t MAX_SYNTHETIC_LISTS = 65536;
inline constexpr u64 MAX_UNIFORM_UPDATE = 1 << 20;
inline constexpr u32 MAX_PUSH_CONSTANTS = 256;

#define NVN_WARN_LIMITED(counter, ...)                                                            \
    do {                                                                                          \
        if (const u64 nvn_n = ++(counter); nvn_n <= 8 || (nvn_n & 0xFFF) == 0) {                  \
            LOG_WARNING(Render_Vulkan, __VA_ARGS__);                                              \
        }                                                                                         \
    } while (false)

/// The NVN entry point the calling thread executes, for log lines ("?" outside the HLE bridge).
[[nodiscard]] inline const char* CurrentNvnCall() {
    return g_current_nvn_call != nullptr ? g_current_nvn_call : "?";
}

/// A rejected argument: counted in `stats.bad_argument`, and the first few per call site logged with
/// the reason, the Recorder function (`where`) and the NVN call the game made. `stats` is the
/// Recorder's stats object.
#define NVN_BAD_ARGUMENT_IN(stats, where, ...)                                                    \
    do {                                                                                          \
        ++(stats).bad_argument;                                                                   \
        static Common::BootCounter nvn_site_count;                                                \
        if (const u64 nvn_n = nvn_site_count.Next() + 1; nvn_n <= 4 || (nvn_n & 0xFFF) == 0) {    \
            LOG_WARNING(Render_Vulkan, "[nvn] bad argument in {} (NVN call {}): {}", where,       \
                        CurrentNvnCall(), fmt::format(__VA_ARGS__));                              \
        }                                                                                         \
    } while (false)
#define NVN_BAD_ARGUMENT(stats, ...) NVN_BAD_ARGUMENT_IN(stats, __func__, __VA_ARGS__)

[[nodiscard]] constexpr u64 Align8(u64 value) {
    return (value + 7) & ~u64{7};
}

[[nodiscard]] constexpr u64 MakeSerial(ObjectKind kind, u32 generation, u32 index) {
    return (static_cast<u64>(kind) << 56) | (static_cast<u64>(generation & 0xFFFFFF) << 32) | index;
}

[[nodiscard]] constexpr ObjectKind SerialKind(u64 serial) {
    return static_cast<ObjectKind>(serial >> 56);
}

// ---- Slab storage indexed by serial ------------------------------------------------------------

struct Object {
    u64 serial{};
};

/// Chunked slab: object addresses are stable, Get is lock-free. serial = kind << 56 |
/// generation << 32 | slot index; a freed slot bumps its generation so stale serials stop
/// resolving.
template <typename T>
class Slab {
public:
    explicit Slab(ObjectKind kind_) : kind{kind_} {}

    ~Slab() {
        for (T* object : graveyard) {
            delete object;
        }
        for (auto& chunk_ptr : chunks) {
            Chunk* chunk = chunk_ptr.load();
            if (!chunk) {
                continue;
            }
            for (auto& slot : chunk->slots) {
                delete slot.object.load();
            }
            delete chunk;
        }
    }

    T* Allocate() {
        std::lock_guard lock{mutex};
        u32 index;
        if (!free_slots.empty()) {
            index = free_slots.back();
            free_slots.pop_back();
        } else {
            index = next_index++;
        }
        const u32 chunk_index = index / CHUNK_SLOTS;
        if (chunk_index >= MAX_CHUNKS) {
            --next_index;
            return nullptr;
        }
        Chunk* chunk = chunks[chunk_index].load();
        if (!chunk) {
            chunk = new Chunk;
            chunks[chunk_index].store(chunk, std::memory_order_release);
        }
        Slot& slot = chunk->slots[index % CHUNK_SLOTS];
        T* object = new T;
        object->serial = MakeSerial(kind, slot.generation.load(), index);
        slot.object.store(object, std::memory_order_release);
        return object;
    }

    [[nodiscard]] T* Get(u64 serial) const {
        if (SerialKind(serial) != kind) {
            return nullptr;
        }
        const u32 index = static_cast<u32>(serial);
        const u32 chunk_index = index / CHUNK_SLOTS;
        if (chunk_index >= MAX_CHUNKS) {
            return nullptr;
        }
        const Chunk* chunk = chunks[chunk_index].load(std::memory_order_acquire);
        if (!chunk) {
            return nullptr;
        }
        const Slot& slot = chunk->slots[index % CHUNK_SLOTS];
        if (slot.generation.load(std::memory_order_acquire) != ((serial >> 32) & 0xFFFFFF)) {
            return nullptr;
        }
        T* object = slot.object.load(std::memory_order_acquire);
        // A slot freed and reused between the generation check and the load holds another object.
        return object && object->serial == serial ? object : nullptr;
    }

    void Free(u64 serial) {
        std::lock_guard lock{mutex};
        const u32 index = static_cast<u32>(serial);
        const u32 chunk_index = index / CHUNK_SLOTS;
        if (SerialKind(serial) != kind || chunk_index >= MAX_CHUNKS) {
            return;
        }
        Chunk* chunk = chunks[chunk_index].load();
        if (!chunk) {
            return;
        }
        Slot& slot = chunk->slots[index % CHUNK_SLOTS];
        // Re-check under the lock: a concurrent Free of the same serial must not queue the slot twice.
        if (slot.generation.load() != ((serial >> 32) & 0xFFFFFF)) {
            return;
        }
        T* object = slot.object.exchange(nullptr);
        if (!object) {
            return;
        }
        u32 generation = (slot.generation.load() + 1) & 0xFFFFFF;
        if (generation == 0) {
            generation = 1;
        }
        slot.generation.store(generation, std::memory_order_release);
        // Lock-free Get callers may still hold the pointer: delete it only after many more frees.
        graveyard.push_back(object);
        if (graveyard.size() > GRAVEYARD_LIMIT) {
            const size_t half = graveyard.size() / 2;
            for (size_t i = 0; i < half; ++i) {
                delete graveyard[i];
            }
            graveyard.erase(graveyard.begin(), graveyard.begin() + half);
        }
        free_slots.push_back(index);
    }

private:
    static constexpr u32 CHUNK_SLOTS = 1024;
    static constexpr u32 MAX_CHUNKS = 4096;

    struct Slot {
        std::atomic<u32> generation{1};
        std::atomic<T*> object{nullptr};
    };
    struct Chunk {
        Slot slots[CHUNK_SLOTS];
    };

    ObjectKind kind;
    std::mutex mutex;
    std::vector<u32> free_slots;
    std::vector<T*> graveyard;
    u32 next_index = 0;
    static constexpr size_t GRAVEYARD_LIMIT = 4096;
    std::array<std::atomic<Chunk*>, MAX_CHUNKS> chunks{};
};

// ---- Host objects ---------------------------------------------------------------------------------

/// State of objects the sink knows (pools, textures, samplers, programs): they are destroyed in the
/// sink only after the game finalized them, no recorded list references them and every submission
/// made before the finalize completed.
struct SinkObject : Object {
    std::atomic<u32> list_refs{0};
    bool finalized = false;
    u64 destroy_after = 0;
};

struct DeviceObject : Object {};

struct QueueObject : Object {
    u32 index = 0;
    std::atomic<u64> last_seq{0};
};

struct MemoryPoolObject : SinkObject {
    u64 address = 0;
    u64 size = 0;
    u32 flags = 0;
};

struct DescriptorPoolObject : Object {
    u64 pool_serial = 0;
    u64 offset = 0;
    u32 count = 0;
    u32 base = 0;
};

struct TextureObject : SinkObject {
    TextureDesc desc{};
    /// View objects created for pool registrations (destroyed with the texture).
    std::vector<std::pair<TextureViewDesc, u64>> views;
};

struct SamplerObject : SinkObject {
    SamplerDesc desc{};
};

struct BufferObject : Object {
    u64 pool_serial = 0;
    u64 gpu_begin = 0;
    u64 gpu_end = 0;
};

struct ProgramObject : SinkObject {
    ProgramDesc desc{};
    std::atomic<bool> created{false};
    u64 code_address[PROGRAM_STAGES]{};
    u32 code_size[PROGRAM_STAGES]{};
    u32 pending_mask = 0; ///< stages whose code area was not readable at SetShaders
};

/// SyncObject::fence_seq of a sync signalled without a fence (no submission ever gets this seq).
constexpr u64 SYNC_SIGNALED_WITHOUT_FENCE = ~u64{0};

struct SyncObject : Object {
    std::atomic<u64> fence_seq{0};
};

struct EventObject : Object {
    u64 address = 0;
};

struct WindowObject : Object {
    u32 count = 0;
    u64 textures[4]{};
    u32 interval = 1;
    PresentCrop crop{};
    std::atomic<u64> slot_seq[4]{};
};

/// Something to do when a submission completed.
struct Action {
    enum class Kind : u32 { SignalEvent, FenceSync };
    u32 word = 0; ///< word index of the record in its list
    Kind kind = Kind::SignalEvent;
    u64 a = 0;    ///< event address / report address / sync serial
    u32 c = 0;    ///< signal mode / counter type
    u32 d = 0;    ///< signal value / report value
};

/// An ended recording: immutable.
struct List {
    std::vector<u64> words;
    std::vector<u64> refs; ///< sorted, unique serials referenced by the records
    std::vector<Action> actions;
    std::vector<u64> callees; ///< sorted, unique synthetic handles its CallList records call
    std::vector<size_t> expansion_offsets; ///< calls and deferred records in word order
    size_t record_end{}; ///< word index of the end marker
};

/// Host memory a list takes.
[[nodiscard]] inline u64 ListBytes(const List& list) {
    return sizeof(List) + list.words.size() * sizeof(u64) + list.refs.size() * sizeof(u64) +
           list.actions.size() * sizeof(Action) + list.callees.size() * sizeof(u64) +
           list.expansion_offsets.size() * sizeof(size_t);
}

struct ListEntry {
    u64 end = 0; ///< end of the first command memory piece
    u64 stamp = 0; ///< synthetic lists: position in Impl::synthetic_lru (last submit/insert)
    u64 bytes = 0; ///< ListBytes
    u64 last_used_present = 0; ///< presents at the insert or the last submit/call
    u32 callers = 0; ///< synthetic lists: lists whose CallList records call this one
    std::shared_ptr<const List> list;
};

struct CommandBufferObject : Object {
    u64 callback = 0;
    u64 callback_data = 0;
    u64 cur = 0; ///< virtual write cursor in the current segment
    u64 end = 0;
    u64 piece_start = 0;
    u64 used = 0;
    u64 begin = 0;
    u8 recording = 0; ///< 0 none, 1 recording, 2 recording without a record yet
    std::vector<std::pair<u64, u64>> pieces;
    std::vector<u64> words;
    std::vector<u64> refs; ///< references taken on this open recording (may hold duplicates)
    std::array<u64, 256> ref_cache{};
    std::vector<Action> actions;
    u32 tex_base = 0;
    u32 samp_base = 0;
    s64 copy_row_stride = 0;
    s64 copy_image_stride = 0;
    u64 control_size = 0;
};

/// Global id space of one kind of descriptor pools (texture or sampler).
class DescriptorTable {
public:
    u32 Allocate(u32 count);
    /// Releases the range and clears its entries.
    void Release(u32 base);
    void Set(u32 base, u32 id, u64 serial);
    /// Replaces every pool registration of an object serial after its address is reinitialized.
    void ReplaceSerial(u64 old_serial, u64 new_serial);
    /// Entries up to the last non-empty one.
    [[nodiscard]] IdSnapshot Snapshot();

private:
    struct Range {
        u32 base;
        u32 span;
        bool used;
    };
    std::mutex mutex;
    std::vector<u64> entries;
    std::vector<Range> ranges;
    std::shared_ptr<const std::vector<u64>> snapshot;
    bool dirty = true;
};

/// Converts a handle as the game gives it into the rebased form records carry: texture id in bits
/// 0-19, sampler id from bit 20, each moved into the command buffer's pool numbering. Ids of 0
/// stay 0. Only the low 32 bits of a handle reach the GPU (as in the driver).
[[nodiscard]] inline u64 RebaseHandle(Op op, u64 raw, u32 texture_base, u32 sampler_base) {
    HandleWordKind kind = HandleWordKind::Image;
    if (op == Op::BindSeparateSamplers) {
        kind = HandleWordKind::SeparateSampler;
    } else if (op == Op::BindTextures || op == Op::BindSeparateTextures) {
        kind = HandleWordKind::Texture;
    }
    return RebaseHandleWord(static_cast<u32>(raw), kind, texture_base, sampler_base);
}

struct AtomicStats {
    std::atomic<u64> tag_mismatch{};
    std::atomic<u64> stale_serial{};
    std::atomic<u64> bad_guest_read{};
    std::atomic<u64> bad_guest_write{};
    std::atomic<u64> bad_argument{};
    std::atomic<u64> callbacks_skipped{};
    std::atomic<u64> synthetic_segments{};
    std::atomic<u64> unknown_call_handle{};
    std::atomic<u64> call_depth_exceeded{};
    std::atomic<u64> deferred_read_failed{};
    std::atomic<u64> program_code_unavailable{};
    std::atomic<u64> stage_fallback{};
    std::atomic<u64> reinitialized_objects{};
    std::atomic<u64> submissions{};
    std::atomic<u64> presents{};
    std::atomic<u64> lists_submitted{};
    std::atomic<u64> records_recorded{};
    std::atomic<u64> records_dropped{};
    std::atomic<u64> completion_actions{};
    std::atomic<u64> retired_objects{};
    std::atomic<u64> synthetic_evictions{};
    std::atomic<u64> finalized_pool_lists{};
    std::atomic<u64> synthetic_lists{};
    std::atomic<u64> synthetic_list_bytes{};
    std::atomic<u64> synthetic_cap_evictions{};
};

struct Recorder::Impl {
    Impl(GuestMemory& memory_, Sink& sink_);

    GuestMemory& memory;
    Sink& sink;
    AtomicStats stats;

    Slab<DeviceObject> devices{ObjectKind::Device};
    Slab<QueueObject> queues{ObjectKind::Queue};
    Slab<CommandBufferObject> command_buffers{ObjectKind::CommandBuffer};
    Slab<MemoryPoolObject> memory_pools{ObjectKind::MemoryPool};
    Slab<DescriptorPoolObject> texture_pools{ObjectKind::TexturePool};
    Slab<DescriptorPoolObject> sampler_pools{ObjectKind::SamplerPool};
    Slab<TextureObject> textures{ObjectKind::Texture};
    Slab<SamplerObject> samplers{ObjectKind::Sampler};
    Slab<BufferObject> buffers{ObjectKind::Buffer};
    /// The guest range of every live buffer object, for BufferEndOf.
    struct BufferSpan {
        u64 begin;
        u64 end;
        u64 serial;
    };
    static constexpr u64 BUFFER_SPAN_CHUNK_SHIFT = 20; // 1 MiB buckets
    static constexpr u64 BUFFER_SPAN_LARGE_LIMIT = u64{16} << 20;
    std::shared_mutex buffer_spans_mutex;
    std::unordered_map<u64, std::vector<BufferSpan>> buffer_span_chunks;
    std::vector<BufferSpan> buffer_spans_large;
    /// Bytes from `address` to the end of the largest live buffer object that holds it, saturated
    /// to 32 bits; 0 when none does.
    [[nodiscard]] u32 BufferBytesFrom(u64 address);
    Slab<ProgramObject> programs{ObjectKind::Program};
    Slab<SyncObject> syncs{ObjectKind::Sync};
    Slab<EventObject> events{ObjectKind::Event};
    Slab<WindowObject> windows{ObjectKind::Window};

    DescriptorTable texture_table;
    DescriptorTable sampler_table;

    // Serializes seq assignment and every Sink call except Wait/SetCompletionCallback.
    std::mutex sink_mutex;
    std::atomic<u64> next_seq{0};
    std::atomic<u32> next_queue_index{0};
    std::atomic<u64> synthetic_cursor{SYNTHETIC_BASE};
    std::atomic<u64> virtual_pool_cursor{VIRTUAL_POOL_BASE};

    // Completion
    struct Pending {
        u64 seq;
        std::vector<Action> actions;
    };
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    /// Serializes the read-modify-write of event ADD signals (CPU thread vs completion thread).
    std::mutex event_mutex;
    std::mutex observer_mutex;
    std::function<void(u64)> completion_observer;
    std::deque<Pending> pending;
    std::atomic<u64> completed_seq{0};

    // Uniform snapshot at submit (uniform_snapshot.h). The pools, pages, scratch and counters are
    // under sink_mutex.
    std::atomic<UniformSnapshotMode> uniform_snapshot{UniformSnapshotMode::Off};
    std::atomic<bool> batch_command_copies{false};
    UniformSnapshotPools snapshot_pools;
    UniformSnapshotCpuPages snapshot_cpu;
    UniformSnapshotScratch snapshot_scratch;
    UniformSnapshotTotals snapshot_totals;
    u64 snapshot_submissions = 0;
    u64 snapshot_ns = 0;
    size_t last_expanded_words = 4096;
    u64 expand_submit_count = 0;
    std::chrono::steady_clock::time_point snapshot_log_time{};
    /// Requires sink_mutex: the submission's uniform snapshot, when the switch is on.
    void CaptureUniformsLocked(Submission& submission);

    // Object retirement
    std::mutex retire_mutex;
    std::vector<u64> retire_list;
    /// Textures whose guest object was initialized again while they were alive (under
    /// retire_mutex). Nothing finalizes them, so they are never destroyed; the ones finalized
    /// through a copy of the object, or gone, are pruned.
    std::vector<u64> reinitialized_textures;
    size_t reinitialized_prune_at = 4096;
    void PruneReinitializedLocked();
    std::mutex destroy_mutex;
    std::vector<u64> destroy_queue;

    // Handle -> recorded list
    std::mutex lists_mutex;
    std::map<u64, ListEntry> lists;
    std::unordered_map<u64, ListEntry*> lists_by_handle;
    /// Synthetic lists by last use (stamp -> handle): the least recently submitted goes first when
    /// the cap is reached.
    std::map<u64, u64> synthetic_lru;
    u64 lru_clock = 0;
    u64 synthetic_count = 0;
    u64 synthetic_bytes = 0;
    /// SetSyntheticListCap (0: no byte cap).
    u64 synthetic_cap_bytes = 0;
    u64 synthetic_idle_presents = DEFAULT_SYNTHETIC_LIST_IDLE_PRESENTS;
    /// Requires lists_mutex: removes the entry (its LRU slot, byte count and the caller counts of
    /// its callees) and moves its list to `removed`, whose references the caller releases.
    std::map<u64, ListEntry>::iterator EraseListLocked(
        std::map<u64, ListEntry>::iterator it, std::vector<std::shared_ptr<const List>>& removed);
    /// Requires lists_mutex: counts `list` as a caller of its callees (or no longer).
    void NoteCalleesLocked(const List& list, bool add);
    /// Requires lists_mutex: drops idle, uncalled synthetic lists, least recently used first,
    /// until they are within synthetic_cap_bytes.
    void EvictSyntheticOverCapLocked(std::vector<std::shared_ptr<const List>>& removed);

    // Anonymous samplers created for RegisterSamplerParams, deduplicated by parameters.
    std::mutex anonymous_sampler_mutex;
    std::unordered_map<u64, std::vector<u64>> anonymous_samplers;

    // ---- guest tags -----------------------------------------------------------------------
    bool ReadTag(u64 va, GuestTag& tag);
    bool WriteTag(u64 va, ObjectKind kind, u64 serial);
    void ClearTag(u64 va);
    /// Resolves the tagged guest object at `va`. `untagged_is_empty`: a never-initialised (all
    /// zero tag) object is an ordinary miss for a getter, not a tag mismatch (see
    /// Recorder::QueryMemoryPool).
    template <typename T>
    T* Lookup(Slab<T>& slab, ObjectKind kind, u64 va, const char* function,
              bool untagged_is_empty = false);
    /// Allocates a host object for `va` (releasing a live object at that address first) and writes
    /// the tag.
    template <typename T>
    T* Create(Slab<T>& slab, ObjectKind kind, u64 va, bool count_reinit = true);
    /// Finalize: invalidates the tag and frees (or retires) the host object.
    void Release(ObjectKind kind, u64 va, u64 serial);
    void FreeSimple(u64 serial);

    // ---- retirement ---------------------------------------------------------------------------
    [[nodiscard]] SinkObject* StateOf(u64 serial);
    void MarkFinalized(u64 serial);
    void ReleaseRefs(const std::vector<u64>& refs);
    void RunRetire();
    void FlushDestroysLocked();

    // ---- sink-facing creation (take sink_mutex) ----------------------------------------------
    /// Requires sink_mutex. Creates the program in the sink once all stage code is readable (or
    /// unconditionally with `force`). Returns whether it exists in the sink.
    bool EnsureProgramLocked(ProgramObject& program, bool force);
    u64 ViewSerialFor(TextureObject& texture, const TextureViewDesc& view);
    u64 AnonymousSamplerSerial(const SamplerDesc& desc);

    // ---- submission -----------------------------------------------------------------------------
    [[nodiscard]] const List* FindListLocked(u64 handle, u64 presents);
    bool ExpandLocked(u64 handle, std::vector<u64>& out, std::vector<Action>& actions,
                      std::vector<u64>& refs, u32 depth, u64 presents);
    void ResolveDeferred(u64* record);
    u64 FinishSubmitLocked(QueueObject& queue, Submission&& submission,
                           std::vector<Action>&& actions, std::vector<u64>&& refs);
    u64 SubmitInternal(QueueObject& queue, std::vector<u64>&& list_words,
                       std::vector<Action>&& actions, std::vector<u64>&& refs);
    void OnComplete(u64 seq);
    void ApplyAction(const Action& action);

    // ---- command buffers ---------------------------------------------------------------------
    /// Looks up the command buffer and requires it to be recording.
    CommandBufferObject* Recording(u64 cmd_va, const char* function);
    void NewSegment(CommandBufferObject& cb, u64 start, u64 end_va);
    void NewSyntheticSegment(CommandBufferObject& cb, u64 needed);
    /// Reserves a record; returns a pointer to its header word (valid until the next call).
    u64* AllocRecord(CommandBufferObject& cb, Op op, u16 flags, u32 payload_bytes, u32 keep);
    void InsertList(u64 handle, const std::vector<std::pair<u64, u64>>& ranges,
                    std::shared_ptr<const List> list);

    template <typename P>
    P* Emit(CommandBufferObject& cb, Op op, u16 flags = 0, u32 tail_bytes = 0) {
        static_assert(sizeof(P) % 8 == 0);
        u64* header = AllocRecord(cb, op, flags, static_cast<u32>(sizeof(P)) + tail_bytes,
                                  COMMAND_RESERVE);
        return reinterpret_cast<P*>(header + 1);
    }

    /// Records that the open recording references `serial` and takes a list reference on it right
    /// away, so the object cannot be retired while the recording is open. Duplicates (cache misses)
    /// are possible; EndRecording releases the extra references.
    void AddRef(CommandBufferObject& cb, u64 serial) {
        u64& cached = cb.ref_cache[(serial * 0x9E3779B97F4A7C15ULL) >> 56];
        if (cached == serial) {
            return;
        }
        cached = serial;
        cb.refs.push_back(serial);
        if (SinkObject* state = StateOf(serial)) {
            state->list_refs.fetch_add(1);
        }
    }
    /// Drops the references of a recording that never ended and clears its state.
    void AbandonRecording(CommandBufferObject& cb);

    /// Reads a texture view from the guest (view_va == 0: default view).
    bool ReadView(u64 view_va, TextureViewDesc& view);
    /// Resolves a texture object + view into a RenderTargetRef and retains the texture.
    bool ResolveTarget(CommandBufferObject& cb, u64 texture_va, u64 view_va, RenderTargetRef& out,
                       const char* function, TextureObject** texture = nullptr);

    bool GuestRead(u64 va, void* dst, size_t size);
    bool GuestWrite(u64 va, const void* src, size_t size);
};

template <typename T>
T* Recorder::Impl::Lookup(Slab<T>& slab, ObjectKind kind, u64 va, const char* function,
                          bool untagged_is_empty) {
    if (va == 0) {
        NVN_BAD_ARGUMENT_IN(stats, function, "null object pointer (expected kind {})",
                            static_cast<u32>(kind));
        return nullptr;
    }
    GuestTag tag{};
    if (!ReadTag(va, tag)) {
        return nullptr;
    }
    if (untagged_is_empty && tag.magic == 0) {
        return nullptr;
    }
    if (tag.magic != GUEST_TAG_MAGIC || tag.kind != static_cast<u32>(kind)) {
        NVN_WARN_LIMITED(stats.tag_mismatch,
                         "[nvn] tag mismatch in {} (NVN call {}): expected kind {} at {:#x}, found "
                         "magic {:#x} kind {}",
                         function, CurrentNvnCall(), static_cast<u32>(kind), va, tag.magic,
                         tag.kind);
        return nullptr;
    }
    T* object = slab.Get(tag.serial);
    if (!object) {
        NVN_WARN_LIMITED(stats.stale_serial,
                         "[nvn] stale object in {}: kind {} at {:#x} serial {:#x}", function,
                         static_cast<u32>(kind), va, tag.serial);
    }
    return object;
}

template <typename T>
T* Recorder::Impl::Create(Slab<T>& slab, ObjectKind kind, u64 va, bool count_reinit) {
    if (va == 0) {
        NVN_BAD_ARGUMENT_IN(stats, "Create", "null object pointer (kind {})",
                            static_cast<u32>(kind));
        return nullptr;
    }
    GuestTag old{};
    if (count_reinit && memory.Read(va, &old, sizeof(old)) && old.magic == GUEST_TAG_MAGIC &&
        old.kind == static_cast<u32>(kind) && slab.Get(old.serial)) {
        // The old object stays alive for the records that retained it (and for copies of it).
        ++stats.reinitialized_objects;
        if constexpr (std::is_same_v<T, TextureObject>) {
            std::lock_guard lock{retire_mutex};
            reinitialized_textures.push_back(old.serial);
            if (reinitialized_textures.size() >= reinitialized_prune_at) {
                PruneReinitializedLocked();
            }
        }
    }
    T* object = slab.Allocate();
    if (!object) {
        NVN_BAD_ARGUMENT_IN(stats, "Create", "object table full (kind {})",
                            static_cast<u32>(kind));
        return nullptr;
    }
    if (!WriteTag(va, kind, object->serial)) {
        slab.Free(object->serial);
        return nullptr;
    }
    return object;
}

} // namespace VideoCommon::Nvn
