// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cstring>

#include "video_core/nvn/recorder_impl.h"

namespace VideoCommon::Nvn {

// ---- DescriptorTable ---------------------------------------------------------------------------

u32 DescriptorTable::Allocate(u32 count) {
    const u32 span = std::max<u32>(256, (count + 255) & ~255U);
    std::lock_guard lock{mutex};
    for (Range& range : ranges) {
        if (!range.used && range.span >= span) {
            range.used = true;
            return range.base;
        }
    }
    const u32 base = ranges.empty() ? 0 : ranges.back().base + ranges.back().span;
    ranges.push_back(Range{base, span, true});
    return base;
}

void DescriptorTable::Release(u32 base) {
    std::lock_guard lock{mutex};
    for (Range& range : ranges) {
        if (range.base == base && range.used) {
            range.used = false;
            const size_t first = std::min<size_t>(base, entries.size());
            const size_t last = std::min<size_t>(static_cast<size_t>(base) + range.span, entries.size());
            std::fill(entries.begin() + first, entries.begin() + last, u64{0});
            dirty = true;
            return;
        }
    }
}

void DescriptorTable::Set(u32 base, u32 id, u64 serial) {
    std::lock_guard lock{mutex};
    const size_t index = static_cast<size_t>(base) + id;
    if (index >= entries.size()) {
        entries.resize(index + 1, 0);
    }
    if (entries[index] == serial) {
        return;
    }
    entries[index] = serial;
    dirty = true;
}

void DescriptorTable::ReplaceSerial(u64 old_serial, u64 new_serial) {
    std::lock_guard lock{mutex};
    for (u64& serial : entries) {
        if (serial == old_serial) {
            serial = new_serial;
            dirty = true;
        }
    }
}

IdSnapshot DescriptorTable::Snapshot() {
    std::lock_guard lock{mutex};
    if (dirty || !snapshot) {
        size_t size = entries.size();
        while (size > 0 && entries[size - 1] == 0) {
            --size;
        }
        snapshot = std::make_shared<const std::vector<u64>>(entries.begin(), entries.begin() + size);
        dirty = false;
    }
    return IdSnapshot{snapshot};
}

// ---- Impl -----------------------------------------------------------------------------------------

Recorder::Impl::Impl(GuestMemory& memory_, Sink& sink_) : memory{memory_}, sink{sink_} {}

bool Recorder::Impl::GuestRead(u64 va, void* dst, size_t size) {
    // Fast path: a read inside one 4 KiB page goes straight through the host pointer.
    if (size <= 0x1000 && (va & 0xFFF) + size <= 0x1000) {
        if (const u8* host = memory.Pointer(va)) {
            std::memcpy(dst, host, size);
            return true;
        }
    }
    if (!memory.Read(va, dst, size)) {
        ++stats.bad_guest_read;
        return false;
    }
    return true;
}

bool Recorder::Impl::GuestWrite(u64 va, const void* src, size_t size) {
    if (!memory.Write(va, src, size)) {
        ++stats.bad_guest_write;
        return false;
    }
    return true;
}

bool Recorder::Impl::ReadTag(u64 va, GuestTag& tag) {
    return GuestRead(va, &tag, sizeof(tag));
}

bool Recorder::Impl::WriteTag(u64 va, ObjectKind kind, u64 serial) {
    const GuestTag tag{GUEST_TAG_MAGIC, static_cast<u32>(kind), serial};
    return GuestWrite(va, &tag, sizeof(tag));
}

void Recorder::Impl::ClearTag(u64 va) {
    const GuestTag tag{};
    GuestWrite(va, &tag, sizeof(tag));
}

void Recorder::Impl::FreeSimple(u64 serial) {
    switch (SerialKind(serial)) {
    case ObjectKind::Device:
        devices.Free(serial);
        break;
    case ObjectKind::Queue:
        queues.Free(serial);
        break;
    case ObjectKind::CommandBuffer:
        command_buffers.Free(serial);
        break;
    case ObjectKind::MemoryPool:
        memory_pools.Free(serial);
        break;
    case ObjectKind::TexturePool:
        texture_pools.Free(serial);
        break;
    case ObjectKind::SamplerPool:
        sampler_pools.Free(serial);
        break;
    case ObjectKind::Texture:
        textures.Free(serial);
        break;
    case ObjectKind::Sampler:
        samplers.Free(serial);
        break;
    case ObjectKind::Buffer:
        buffers.Free(serial);
        break;
    case ObjectKind::Program:
        programs.Free(serial);
        break;
    case ObjectKind::Sync:
        syncs.Free(serial);
        break;
    case ObjectKind::Event:
        events.Free(serial);
        break;
    case ObjectKind::Window:
        windows.Free(serial);
        break;
    case ObjectKind::Invalid:
        break;
    }
}

void Recorder::Impl::Release(ObjectKind kind, u64 va, u64 serial) {
    ClearTag(va);
    switch (kind) {
    case ObjectKind::MemoryPool:
    case ObjectKind::Texture:
    case ObjectKind::Sampler:
    case ObjectKind::Program:
        MarkFinalized(serial);
        break;
    default:
        FreeSimple(serial);
        break;
    }
}

SinkObject* Recorder::Impl::StateOf(u64 serial) {
    switch (SerialKind(serial)) {
    case ObjectKind::MemoryPool:
        return memory_pools.Get(serial);
    case ObjectKind::Texture:
        return textures.Get(serial);
    case ObjectKind::Sampler:
        return samplers.Get(serial);
    case ObjectKind::Program:
        return programs.Get(serial);
    default:
        return nullptr;
    }
}

void Recorder::Impl::MarkFinalized(u64 serial) {
    SinkObject* state = StateOf(serial);
    if (!state) {
        return;
    }
    {
        std::lock_guard lock{retire_mutex};
        if (state->finalized) {
            return;
        }
        state->finalized = true;
        state->destroy_after = next_seq.load();
        retire_list.push_back(serial);
    }
    RunRetire();
}

void Recorder::Impl::AbandonRecording(CommandBufferObject& cb) {
    if (cb.recording == 0 && cb.refs.empty()) {
        return;
    }
    ReleaseRefs(cb.refs);
    cb.words.clear();
    cb.refs.clear();
    cb.actions.clear();
    cb.pieces.clear();
    cb.ref_cache.fill(0);
    cb.recording = 0;
    RunRetire();
}

void Recorder::Impl::ReleaseRefs(const std::vector<u64>& refs) {
    for (const u64 serial : refs) {
        if (SinkObject* state = StateOf(serial)) {
            state->list_refs.fetch_sub(1);
        }
    }
}

void Recorder::Impl::RunRetire() {
    std::vector<u64> ready;
    {
        std::lock_guard lock{retire_mutex};
        const u64 completed = completed_seq.load();
        for (size_t i = 0; i < retire_list.size();) {
            const u64 serial = retire_list[i];
            const SinkObject* state = StateOf(serial);
            if (!state) {
                retire_list[i] = retire_list.back();
                retire_list.pop_back();
                continue;
            }
            if (completed >= state->destroy_after && state->list_refs.load() == 0) {
                ready.push_back(serial);
                retire_list[i] = retire_list.back();
                retire_list.pop_back();
                continue;
            }
            ++i;
        }
    }
    if (ready.empty()) {
        return;
    }
    std::vector<u64> destroy;
    for (const u64 serial : ready) {
        switch (SerialKind(serial)) {
        case ObjectKind::Texture: {
            if (TextureObject* texture = textures.Get(serial)) {
                for (const auto& view : texture->views) {
                    destroy.push_back(view.second);
                    textures.Free(view.second);
                }
            }
            destroy.push_back(serial);
            break;
        }
        case ObjectKind::Program: {
            const ProgramObject* program = programs.Get(serial);
            if (program && program->created.load()) {
                destroy.push_back(serial);
            }
            break;
        }
        default:
            destroy.push_back(serial);
            break;
        }
        FreeSimple(serial);
    }
    std::lock_guard lock{destroy_mutex};
    destroy_queue.insert(destroy_queue.end(), destroy.begin(), destroy.end());
}

void Recorder::Impl::FlushDestroysLocked() {
    std::vector<u64> serials;
    {
        std::lock_guard lock{destroy_mutex};
        serials.swap(destroy_queue);
    }
    for (const u64 serial : serials) {
        sink.Destroy(serial);
        ++stats.retired_objects;
    }
}

// ---- Programs ----------------------------------------------------------------------------------------

bool Recorder::Impl::EnsureProgramLocked(ProgramObject& program, bool force) {
    if (program.created.load()) {
        return true;
    }
    for (u32 stage = 0; stage < PROGRAM_STAGES; ++stage) {
        if (!(program.pending_mask & (1U << stage))) {
            continue;
        }
        std::vector<u8> code(program.code_size[stage]);
        if (memory.Read(program.code_address[stage], code.data(), code.size())) {
            program.desc.code[stage] = std::move(code);
            program.pending_mask &= ~(1U << stage);
        }
    }
    if (program.pending_mask != 0) {
        if (!force) {
            return false;
        }
        NVN_WARN_LIMITED(stats.program_code_unavailable,
                         "[nvn] program {:#x} created with unreadable code (pending mask {:#x})",
                         program.serial, program.pending_mask);
        program.pending_mask = 0;
    }
    FlushDestroysLocked();
    sink.CreateProgram(program.desc);
    program.created.store(true);
    return true;
}

// ---- Textures ----------------------------------------------------------------------------------------

u64 Recorder::Impl::ViewSerialFor(TextureObject& texture, const TextureViewDesc& view) {
    std::lock_guard lock{sink_mutex};
    for (const auto& entry : texture.views) {
        if (std::memcmp(&entry.first, &view, sizeof(view)) == 0) {
            return entry.second;
        }
    }
    TextureObject* object = textures.Allocate();
    if (!object) {
        NVN_BAD_ARGUMENT(stats, "texture table full");
        return texture.serial;
    }
    object->desc = texture.desc;
    object->desc.serial = object->serial;
    object->desc.view_of = texture.serial;
    object->desc.view = view;
    FlushDestroysLocked();
    sink.CreateTexture(object->desc);
    texture.views.emplace_back(view, object->serial);
    return object->serial;
}

u64 Recorder::Impl::AnonymousSamplerSerial(const SamplerDesc& desc) {
    // FNV-1a over the parameters (everything but the serial).
    u64 hash = 0xCBF29CE484222325ULL;
    const auto mix = [&hash](const void* data, size_t size) {
        const auto* bytes = static_cast<const u8*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash = (hash ^ bytes[i]) * 0x100000001B3ULL;
        }
    };
    const u8 small[8] = {desc.min_filter, desc.mag_filter,   desc.wrap_s,   desc.wrap_t,
                         desc.wrap_r,     desc.compare_mode, desc.compare_func, desc.reduction};
    mix(small, sizeof(small));
    mix(&desc.lod_min, 4);
    mix(&desc.lod_max, 4);
    mix(&desc.lod_bias, 4);
    mix(&desc.max_anisotropy, 4);
    mix(desc.border, sizeof(desc.border));

    const auto same = [&desc](const SamplerDesc& other) {
        return std::memcmp(&desc.min_filter, &other.min_filter, 8) == 0 &&
               desc.lod_min == other.lod_min && desc.lod_max == other.lod_max &&
               desc.lod_bias == other.lod_bias && desc.max_anisotropy == other.max_anisotropy &&
               std::memcmp(desc.border, other.border, sizeof(desc.border)) == 0;
    };
    std::lock_guard anonymous_lock{anonymous_sampler_mutex};
    auto& bucket = anonymous_samplers[hash];
    for (const u64 serial : bucket) {
        if (const SamplerObject* existing = samplers.Get(serial); existing && same(existing->desc)) {
            return serial;
        }
    }
    SamplerObject* object = samplers.Allocate();
    if (!object) {
        NVN_BAD_ARGUMENT(stats, "sampler table full");
        return 0;
    }
    object->desc = desc;
    object->desc.serial = object->serial;
    {
        std::lock_guard lock{sink_mutex};
        FlushDestroysLocked();
        sink.CreateSampler(object->desc);
    }
    bucket.push_back(object->serial);
    return object->serial;
}

// ---- Handle -> list map -----------------------------------------------------------------------------

const List* Recorder::Impl::FindListLocked(u64 handle, u64 presents) {
    const auto it = lists_by_handle.find(handle);
    if (it == lists_by_handle.end()) {
        return nullptr;
    }
    ListEntry& entry = *it->second;
    entry.last_used_present = presents;
    if (handle >= SYNTHETIC_BASE &&
        (synthetic_cap_bytes != 0 || synthetic_lru.size() >= MAX_SYNTHETIC_LISTS / 2)) {
        synthetic_lru.erase(entry.stamp);
        entry.stamp = ++lru_clock;
        synthetic_lru.emplace(entry.stamp, handle);
    }
    return entry.list.get();
}

void Recorder::Impl::NoteCalleesLocked(const List& list, bool add) {
    for (const u64 callee : list.callees) {
        const auto it = lists.find(callee);
        if (it == lists.end()) {
            continue;
        }
        if (add) {
            ++it->second.callers;
        } else if (it->second.callers != 0) {
            --it->second.callers;
        }
    }
}

std::map<u64, ListEntry>::iterator Recorder::Impl::EraseListLocked(
    std::map<u64, ListEntry>::iterator it, std::vector<std::shared_ptr<const List>>& removed) {
    if (it->first >= SYNTHETIC_BASE) {
        synthetic_lru.erase(it->second.stamp);
        synthetic_count -= (std::min<u64>)(synthetic_count, 1);
        synthetic_bytes -= (std::min)(synthetic_bytes, it->second.bytes);
        stats.synthetic_lists.store(synthetic_count, std::memory_order_relaxed);
        stats.synthetic_list_bytes.store(synthetic_bytes, std::memory_order_relaxed);
    }
    if (it->second.list) {
        NoteCalleesLocked(*it->second.list, false);
    }
    removed.push_back(std::move(it->second.list));
    lists_by_handle.erase(it->first);
    return lists.erase(it);
}

void Recorder::Impl::EvictSyntheticOverCapLocked(std::vector<std::shared_ptr<const List>>& removed) {
    if (synthetic_cap_bytes == 0) {
        return;
    }
    // Lists other lists call stay (the caller may be submitted again); after this many of them
    // the pass gives up until the next insert.
    constexpr size_t MAX_SKIPPED = 1024;
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    size_t skipped = 0;
    auto it = synthetic_lru.begin();
    while (it != synthetic_lru.end() && synthetic_bytes > synthetic_cap_bytes) {
        const auto victim = lists.find(it->second);
        if (victim == lists.end()) {
            it = synthetic_lru.erase(it);
            continue;
        }
        const ListEntry& entry = victim->second;
        if (presents - (std::min)(presents, entry.last_used_present) < synthetic_idle_presents) {
            break; // the rest of the order was used later
        }
        if (entry.callers != 0) {
            if (++skipped > MAX_SKIPPED) {
                break;
            }
            ++it;
            continue;
        }
        const auto next = std::next(it);
        EraseListLocked(victim, removed);
        ++stats.synthetic_cap_evictions;
        it = next;
    }
}

void Recorder::Impl::InsertList(u64 handle, const std::vector<std::pair<u64, u64>>& ranges,
                                std::shared_ptr<const List> list) {
    std::vector<std::shared_ptr<const List>> removed;
    {
        std::lock_guard lock{lists_mutex};
        // A recording overwrites the guest bytes of the command memory it was written into, so
        // every older list that overlaps any of those bytes is dead.
        for (const auto& [low, high] : ranges) {
            auto it = lists.lower_bound(low);
            if (it != lists.begin()) {
                const auto previous = std::prev(it);
                if (previous->second.end > low) {
                    EraseListLocked(previous, removed);
                }
            }
            while (it != lists.end() && it->first < high) {
                it = EraseListLocked(it, removed);
            }
        }
        if (const auto existing = lists.find(handle); existing != lists.end()) {
            EraseListLocked(existing, removed);
        }
        const List& inserted = *list;
        auto [inserted_it, inserted_ok] = lists.try_emplace(handle);
        ListEntry& entry = inserted_it->second;
        lists_by_handle[handle] = &entry;
        entry.end = ranges.front().second;
        entry.bytes = ListBytes(inserted);
        entry.last_used_present = stats.presents.load(std::memory_order_relaxed);
        entry.list = std::move(list);
        NoteCalleesLocked(inserted, true);
        if (handle >= SYNTHETIC_BASE) {
            entry.stamp = ++lru_clock;
            synthetic_lru.emplace(entry.stamp, handle);
            ++synthetic_count;
            synthetic_bytes += entry.bytes;
            stats.synthetic_lists.store(synthetic_count, std::memory_order_relaxed);
            stats.synthetic_list_bytes.store(synthetic_bytes, std::memory_order_relaxed);
            while (synthetic_lru.size() > MAX_SYNTHETIC_LISTS) {
                // Evict the list that was submitted (or recorded) least recently, not the oldest.
                const auto oldest = synthetic_lru.begin();
                const auto victim = lists.find(oldest->second);
                if (victim != lists.end()) {
                    EraseListLocked(victim, removed);
                    ++stats.synthetic_evictions;
                } else {
                    synthetic_lru.erase(oldest);
                }
            }
        }
        // After any insert: a list recorded over may have been the last caller of one.
        EvictSyntheticOverCapLocked(removed);
    }
    if (removed.empty()) {
        return;
    }
    for (const auto& list_ptr : removed) {
        if (list_ptr) {
            ReleaseRefs(list_ptr->refs);
        }
    }
    RunRetire();
}

// ---- Submission -----------------------------------------------------------------------------------------

namespace {

/// A read inside one 4 KiB page goes straight through the host pointer (the deferred handle and
/// range arrays of every bind record are read this way, thousands per frame).
bool ReadGuestFast(GuestMemory& memory, u64 va, void* dst, size_t size) {
    if (size <= 0x1000 && (va & 0xFFF) + size <= 0x1000) {
        if (const u8* host = memory.Pointer(va)) {
            std::memcpy(dst, host, size);
            return true;
        }
    }
    return memory.Read(va, dst, size);
}

} // namespace

void Recorder::Impl::ResolveDeferred(u64* record) {
    auto* header = reinterpret_cast<RecordHeader*>(record);
    const Op op = static_cast<Op>(header->op);
    switch (op) {
    case Op::BindUniformBuffers:
    case Op::BindStorageBuffers: {
        auto* payload = const_cast<BindBuffersPayload*>(RecordPayload<BindBuffersPayload>(header));
        auto* ranges = const_cast<BufferRange*>(TrailingArray<BufferRange>(payload));
        if (!ReadGuestFast(memory, payload->src, ranges, sizeof(BufferRange) * payload->count)) {
            ++stats.deferred_read_failed;
        } else if (op == Op::BindUniformBuffers) {
            // The sizes the GPU sees: 256-byte units, rounded up (BindUniformBuffers at record).
            for (u32 i = 0; i < payload->count; ++i) {
                ranges[i].size = (ranges[i].size + 0xFF) & ~u64{0xFF};
            }
        }
        break;
    }
    case Op::BindTextures:
    case Op::BindSeparateTextures:
    case Op::BindSeparateSamplers:
    case Op::BindImages: {
        auto* payload = const_cast<BindHandlesPayload*>(RecordPayload<BindHandlesPayload>(header));
        auto* handles = const_cast<u64*>(TrailingArray<u64>(payload));
        if (!ReadGuestFast(memory, payload->src, handles, sizeof(u64) * payload->count)) {
            ++stats.deferred_read_failed;
            break;
        }
        for (u32 i = 0; i < payload->count; ++i) {
            handles[i] = RebaseHandle(op, handles[i], payload->tex_base, payload->samp_base);
        }
        break;
    }
    default:
        break;
    }
    if (IsDeferrableOp(op)) {
        header->flags &= static_cast<u16>(~RECORD_FLAG_DEFERRED);
    }
}

bool Recorder::Impl::ExpandLocked(u64 handle, std::vector<u64>& out, std::vector<Action>& actions,
                                  std::vector<u64>& refs, u32 depth, u64 presents) {
    const List* const list = FindListLocked(handle, presents);
    if (list == nullptr) {
        return false;
    }
    refs.insert(refs.end(), list->refs.begin(), list->refs.end());
    const std::vector<u64>& words = list->words;
    if (depth == 0) {
        out.reserve(out.size() + words.size()); // about as long as the list itself
    }
    size_t action_index = 0;
    const auto copy = [&](size_t begin, size_t end) {
        if (end > begin) {
            out.insert(out.end(), words.data() + begin, words.data() + end);
        }
    };
    const auto expand_record = [&](size_t offset) {
        const auto* header = reinterpret_cast<const RecordHeader*>(&words[offset]);
        const size_t next = offset + header->size / 8;
        while (action_index < list->actions.size() && list->actions[action_index].word <= offset) {
            actions.push_back(list->actions[action_index++]);
        }
        const Op op = RecordOp(header);
        if (op == Op::CallList) {
            const u64 callee = RecordPayload<CallListPayload>(header)->handle;
            if (depth + 1 > MAX_CALL_DEPTH) {
                ++stats.call_depth_exceeded;
            } else if (!ExpandLocked(callee, out, actions, refs, depth + 1, presents)) {
                NVN_WARN_LIMITED(stats.unknown_call_handle,
                                 "[nvn] CallList of unknown command handle {:#x}", callee);
            }
        } else {
            const size_t start = out.size();
            copy(offset, next);
            if ((header->flags & RECORD_FLAG_DEFERRED) && IsDeferrableOp(op)) {
                ResolveDeferred(&out[start]);
            }
        }
        return next;
    };
    if (batch_command_copies.load(std::memory_order_relaxed)) {
        size_t begin = 0;
        for (size_t offset : list->expansion_offsets) {
            copy(begin, offset);
            begin = expand_record(offset);
        }
        copy(begin, list->record_end);
    } else {
        for (size_t offset = 0; offset < list->record_end;) {
            offset = expand_record(offset);
        }
    }
    actions.insert(actions.end(), list->actions.begin() + action_index, list->actions.end());
    return true;
}

u64 Recorder::Impl::FinishSubmitLocked(QueueObject& queue, Submission&& submission,
                                       std::vector<Action>&& actions, std::vector<u64>&& refs) {
    std::sort(refs.begin(), refs.end());
    refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
    for (const u64 serial : refs) {
        if (SerialKind(serial) == ObjectKind::Program) {
            if (ProgramObject* program = programs.Get(serial)) {
                EnsureProgramLocked(*program, true);
            }
        }
    }
    submission.retained_serials = std::move(refs);
    submission.texture_pool_snapshot = texture_table.Snapshot();
    submission.sampler_pool_snapshot = sampler_table.Snapshot();
    CaptureUniformsLocked(submission);
    const u64 seq = ++next_seq;
    submission.seq = seq;
    submission.queue = queue.index;

    Pending completion{seq, {}};
    for (const Action& action : actions) {
        if (action.kind == Action::Kind::FenceSync) {
            if (SyncObject* sync = syncs.Get(action.a)) {
                sync->fence_seq.store(seq);
            }
        } else if (action.kind == Action::Kind::SignalEvent) {
            completion.actions.push_back(action);
        }
    }
    {
        std::lock_guard lock{completion_mutex};
        pending.push_back(std::move(completion));
    }
    queue.last_seq.store(seq);
    ++stats.submissions;
    stats.lists_submitted += submission.lists.size();
    sink.Submit(std::move(submission));
    return seq;
}

void Recorder::Impl::CaptureUniformsLocked(Submission& submission) {
    if (uniform_snapshot.load(std::memory_order_relaxed) == UniformSnapshotMode::Off) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    // Read before the bytes: a submission completing meanwhile only makes the replay drop more.
    const u64 completed = completed_seq.load(std::memory_order_acquire);
    submission.uniform_snapshot = CaptureUniformSnapshot(
        submission.lists, snapshot_pools, snapshot_cpu, SharedUniformSnapshot(), memory,
        completed, snapshot_scratch, snapshot_totals);
    const auto now = std::chrono::steady_clock::now();
    snapshot_ns += static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
    ++snapshot_submissions;
    if (now - snapshot_log_time >= std::chrono::seconds{1}) {
        if (snapshot_log_time != std::chrono::steady_clock::time_point{}) {
            LOG_DEBUG(Render_Vulkan,
                      "[nvn] uniform snapshot: {} submissions, {} bindings ({} left live: not "
                      "CPU-written lately, {} left live: GPU-written pages), {} ranges, {} KiB "
                      "copied, {:.3f} ms on the submit threads in the last second ({} GPU-written "
                      "runs known)",
                      snapshot_submissions, snapshot_totals.bindings, snapshot_totals.not_cpu,
                      snapshot_totals.gpu_written, snapshot_totals.ranges,
                      snapshot_totals.bytes / 1024, static_cast<double>(snapshot_ns) / 1e6,
                      SharedUniformSnapshot().Runs());
        }
        snapshot_log_time = now;
        snapshot_totals = {};
        snapshot_submissions = 0;
        snapshot_ns = 0;
    }
}

u64 Recorder::Impl::SubmitInternal(QueueObject& queue, std::vector<u64>&& list_words,
                                   std::vector<Action>&& actions, std::vector<u64>&& refs) {
    std::lock_guard lock{sink_mutex};
    FlushDestroysLocked();
    Submission submission{};
    submission.lists.push_back(std::move(list_words));
    return FinishSubmitLocked(queue, std::move(submission), std::move(actions), std::move(refs));
}

/// The guest-visible event word is written here, at completion in submission order; the replay
/// models the word itself from the waits and signals in the record stream.
void Recorder::Impl::ApplyAction(const Action& action) {
    ++stats.completion_actions;
    switch (action.kind) {
    case Action::Kind::SignalEvent: {
        u32 value = action.d;
        std::lock_guard event_lock{event_mutex};
        if (action.c == 1) { // NVN_EVENT_SIGNAL_MODE_ADD
            u32 current = 0;
            if (GuestRead(action.a, &current, sizeof(current))) {
                value += current;
            }
        }
        GuestWrite(action.a, &value, sizeof(value));
        break;
    }
    // ReportValue and ReportCounter are written by the sink, on the GPU timeline.
    case Action::Kind::FenceSync:
        break;
    }
}

void Recorder::Impl::OnComplete(u64 seq) {
    {
        std::lock_guard lock{completion_mutex};
        while (!pending.empty() && pending.front().seq <= seq) {
            for (const Action& action : pending.front().actions) {
                ApplyAction(action);
            }
            pending.pop_front();
        }
        if (seq > completed_seq.load()) {
            completed_seq.store(seq);
        }
    }
    completion_cv.notify_all();
    RunRetire();
    std::function<void(u64)> observer;
    {
        std::lock_guard lock{observer_mutex};
        observer = completion_observer;
    }
    if (observer) {
        observer(seq);
    }
}

// ---- Recorder -----------------------------------------------------------------------------------------

Recorder::Recorder(GuestMemory& memory, Sink& sink) : impl{std::make_unique<Impl>(memory, sink)} {
    SharedUniformSnapshot().Reset(UniformSnapshotMode::Off);
    sink.SetGuestMemory(&memory);
    Impl* state = impl.get();
    sink.SetCompletionCallback([state](u64 seq) { state->OnComplete(seq); });
}

void Recorder::SetCompletionObserver(std::function<void(u64)> observer) {
    std::lock_guard lock{impl->observer_mutex};
    impl->completion_observer = std::move(observer);
}

Recorder::~Recorder() {
    impl->sink.SetCompletionCallback({});
    FlushDestroys();
}

void Recorder::SetDeviceModes(u32 window_origin, u32 depth_mode) {
    impl->sink.SetDeviceModes(window_origin, depth_mode);
}

void Recorder::SetCommandListBatching(bool enabled) {
    impl->batch_command_copies.store(enabled, std::memory_order_relaxed);
}

void Recorder::SetUniformSnapshot(UniformSnapshotMode mode) {
    impl->uniform_snapshot.store(mode, std::memory_order_relaxed);
    SharedUniformSnapshot().Reset(mode);
}

bool Recorder::UniformSnapshotEnabled() const {
    return impl->uniform_snapshot.load(std::memory_order_relaxed) != UniformSnapshotMode::Off;
}

RecorderStats Recorder::Stats() const {
    const AtomicStats& s = impl->stats;
    return RecorderStats{
        s.tag_mismatch.load(),       s.stale_serial.load(),         s.bad_guest_read.load(),
        s.bad_guest_write.load(),    s.bad_argument.load(),         s.callbacks_skipped.load(),
        s.synthetic_segments.load(), s.unknown_call_handle.load(),  s.call_depth_exceeded.load(),
        s.deferred_read_failed.load(), s.program_code_unavailable.load(), s.stage_fallback.load(),
        s.reinitialized_objects.load(), s.submissions.load(),       s.presents.load(),
        s.lists_submitted.load(),    s.records_recorded.load(),     s.records_dropped.load(),
        s.completion_actions.load(), s.retired_objects.load(), s.synthetic_evictions.load(),
        s.finalized_pool_lists.load(), s.synthetic_lists.load(), s.synthetic_list_bytes.load(),
        s.synthetic_cap_evictions.load(),
    };
}

void Recorder::SetSyntheticListCap(u64 bytes, u64 idle_presents) {
    std::vector<std::shared_ptr<const List>> removed;
    {
        std::lock_guard lock{impl->lists_mutex};
        impl->synthetic_cap_bytes = bytes;
        impl->synthetic_idle_presents = idle_presents;
        impl->EvictSyntheticOverCapLocked(removed);
    }
    for (const auto& list : removed) {
        if (list) {
            impl->ReleaseRefs(list->refs);
        }
    }
    impl->RunRetire();
}

void Recorder::Impl::PruneReinitializedLocked() {
    // Gone (retired), or finalized through a copy of the object: the retirement covers those.
    std::erase_if(reinitialized_textures, [this](u64 serial) {
        const TextureObject* texture = textures.Get(serial);
        return texture == nullptr || texture->finalized;
    });
    std::sort(reinitialized_textures.begin(), reinitialized_textures.end());
    reinitialized_textures.erase(
        std::unique(reinitialized_textures.begin(), reinitialized_textures.end()),
        reinitialized_textures.end());
    reinitialized_prune_at = (std::max<size_t>)(4096, reinitialized_textures.size() * 2);
}

TextureHolderCounts Recorder::PublishTextureHolders() {
    Impl& s = *impl;
    // Lists are immutable: take them under the lock and read them outside it.
    std::vector<std::pair<bool, std::shared_ptr<const List>>> lists;
    {
        std::lock_guard lock{s.lists_mutex};
        lists.reserve(s.lists.size());
        for (const auto& [handle, entry] : s.lists) {
            if (entry.list) {
                lists.emplace_back(handle >= SYNTHETIC_BASE, entry.list);
            }
        }
    }
    // Textures the game finalized that are not destroyed yet, and the re-initialised ones.
    std::unordered_map<u64, TextureHolder> holders;
    {
        std::lock_guard lock{s.retire_mutex};
        for (const u64 serial : s.retire_list) {
            if (SerialKind(serial) != ObjectKind::Texture) {
                continue;
            }
            const TextureObject* texture = s.textures.Get(serial);
            if (texture != nullptr && texture->desc.view_of == 0) {
                holders.emplace(serial, TextureHolder::Retiring);
            }
        }
        s.PruneReinitializedLocked();
        for (const u64 serial : s.reinitialized_textures) {
            holders.emplace(serial, TextureHolder::Reinitialized);
        }
    }
    // A finalized texture a list references lives as long as that list: a list in the game's
    // command memory dies when the game records over it, a synthetic one only by eviction.
    std::unordered_map<u64, u8> referenced; // serial -> 1 guest list, 2 synthetic list
    for (const auto& [synthetic, list] : lists) {
        for (const u64 ref : list->refs) {
            const auto it = holders.find(ref);
            if (it != holders.end() && it->second == TextureHolder::Retiring) {
                u8& bits = referenced[ref];
                bits = static_cast<u8>(bits | (synthetic ? 2 : 1));
            }
        }
    }
    for (const auto& [serial, bits] : referenced) {
        holders[serial] = (bits & 1) != 0 ? TextureHolder::GuestList : TextureHolder::SyntheticList;
    }
    TextureHolderCounts counts;
    for (const auto& [serial, holder] : holders) {
        switch (holder) {
        case TextureHolder::Retiring:
            ++counts.retiring;
            break;
        case TextureHolder::GuestList:
            ++counts.guest_lists;
            break;
        case TextureHolder::SyntheticList:
            ++counts.synthetic_lists;
            break;
        case TextureHolder::Reinitialized:
            ++counts.reinitialized;
            break;
        case TextureHolder::Game:
            break;
        }
    }
    std::lock_guard lock{s.sink_mutex};
    // The view objects go with their texture (sink_mutex guards TextureObject::views).
    TextureHolderReport report;
    report.serials.reserve(holders.size());
    for (const auto& [serial, holder] : holders) {
        report.serials.emplace_back(serial, holder);
        if (const TextureObject* texture = s.textures.Get(serial)) {
            for (const auto& view : texture->views) {
                report.serials.emplace_back(view.second, holder);
            }
        }
    }
    std::sort(report.serials.begin(), report.serials.end());
    s.FlushDestroysLocked();
    s.sink.NoteTextureHolders(std::move(report));
    return counts;
}

// ---- Device / queue --------------------------------------------------------------------------------------

bool Recorder::DeviceInitialize(u64 device_va) {
    return impl->Create(impl->devices, ObjectKind::Device, device_va) != nullptr;
}

void Recorder::DeviceFinalize(u64 device_va) {
    if (const DeviceObject* device =
            impl->Lookup(impl->devices, ObjectKind::Device, device_va, __func__)) {
        impl->Release(ObjectKind::Device, device_va, device->serial);
    }
}

bool Recorder::QueueInitialize(u64 queue_va) {
    QueueObject* queue = impl->Create(impl->queues, ObjectKind::Queue, queue_va);
    if (!queue) {
        return false;
    }
    queue->index = impl->next_queue_index++;
    return true;
}

void Recorder::QueueFinalize(u64 queue_va) {
    if (const QueueObject* queue =
            impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__)) {
        impl->Release(ObjectKind::Queue, queue_va, queue->serial);
    }
}

std::optional<u32> Recorder::QueueIndex(u64 queue_va) {
    if (const QueueObject* queue =
            impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__)) {
        return queue->index;
    }
    return std::nullopt;
}

u64 Recorder::QueueSubmitCommands(u64 queue_va, std::span<const u64> handles) {
    QueueObject* queue = impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__);
    if (!queue) {
        return 0;
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->FlushDestroysLocked();
    Submission submission{};
    std::vector<Action> actions;
    std::vector<u64> refs;
    actions.reserve(64);
    refs.reserve(256);
    {
        std::lock_guard lists_lock{impl->lists_mutex};
        const u64 presents = impl->stats.presents.load(std::memory_order_relaxed);
        for (const u64 handle : handles) {
            std::vector<u64> words;
            words.reserve(impl->last_expanded_words);
            if (!impl->ExpandLocked(handle, words, actions, refs, 0, presents)) {
                NVN_WARN_LIMITED(impl->stats.unknown_call_handle,
                                 "[nvn] QueueSubmitCommands of unknown command handle {:#x}",
                                 handle);
                continue;
            }
            if (++impl->expand_submit_count % 64 == 0) {
                impl->last_expanded_words =
                    (std::max<size_t>)(4096, impl->last_expanded_words * 15 / 16);
            }
            impl->last_expanded_words = (std::min<size_t>)(
                (std::max)(words.size() + 1, impl->last_expanded_words), size_t{4} << 20);
            words.push_back(0); // End record: op 0, flags 0, size 8
            words.back() = u64{8} << 32;
            submission.lists.push_back(std::move(words));
        }
    }
    if (submission.lists.empty()) {
        return queue->last_seq.load();
    }
    return impl->FinishSubmitLocked(*queue, std::move(submission), std::move(actions),
                                    std::move(refs));
}

u64 Recorder::QueueSubmitCommands(u64 queue_va, s32 count, u64 handles_va) {
    if (count < 0 || count > 4096) {
        NVN_BAD_ARGUMENT(impl->stats, "{} submitted lists (limit 4096)", count);
        return 0;
    }
    std::vector<u64> handles(static_cast<size_t>(count));
    if (count > 0 && !impl->GuestRead(handles_va, handles.data(), handles.size() * sizeof(u64))) {
        return 0;
    }
    return QueueSubmitCommands(queue_va, handles);
}

u64 Recorder::QueueFenceSync(u64 queue_va, u64 sync_va, u32 condition, u32 flags) {
    QueueObject* queue = impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__);
    const SyncObject* sync = impl->Lookup(impl->syncs, ObjectKind::Sync, sync_va, __func__);
    if (!queue || !sync) {
        return 0;
    }
    std::vector<u64> words(3 + 1, 0);
    auto* header = reinterpret_cast<RecordHeader*>(words.data());
    *header = RecordHeader{static_cast<u16>(Op::FenceSync), 0, 8 + sizeof(FenceSyncPayload)};
    *reinterpret_cast<FenceSyncPayload*>(words.data() + 1) =
        FenceSyncPayload{sync->serial, condition, flags};
    words[3] = u64{8} << 32; // End
    Action action;
    action.kind = Action::Kind::FenceSync;
    action.a = sync->serial;
    return impl->SubmitInternal(*queue, std::move(words), {action}, {sync->serial});
}

u64 Recorder::QueueWaitSync(u64 queue_va, u64 sync_va) {
    QueueObject* queue = impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__);
    const SyncObject* sync = impl->Lookup(impl->syncs, ObjectKind::Sync, sync_va, __func__);
    if (!queue || !sync) {
        return 0;
    }
    std::vector<u64> words(2 + 1, 0);
    auto* header = reinterpret_cast<RecordHeader*>(words.data());
    *header = RecordHeader{static_cast<u16>(Op::WaitSync), 0, 8 + sizeof(WaitSyncPayload)};
    *reinterpret_cast<WaitSyncPayload*>(words.data() + 1) = WaitSyncPayload{sync->serial};
    words[2] = u64{8} << 32; // End
    return impl->SubmitInternal(*queue, std::move(words), {}, {sync->serial});
}

u64 Recorder::QueueLastSeq(u64 queue_va) {
    if (const QueueObject* queue =
            impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__)) {
        return queue->last_seq.load();
    }
    return 0;
}

bool Recorder::QueueFinish(u64 queue_va, std::chrono::nanoseconds timeout) {
    const QueueObject* queue = impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__);
    return queue && WaitForSeq(queue->last_seq.load(), timeout);
}

u64 Recorder::QueuePresentTexture(u64 queue_va, u64 window_va, u32 slot) {
    QueueObject* queue = impl->Lookup(impl->queues, ObjectKind::Queue, queue_va, __func__);
    WindowObject* window = impl->Lookup(impl->windows, ObjectKind::Window, window_va, __func__);
    if (!queue || !window || slot >= window->count) {
        NVN_BAD_ARGUMENT(impl->stats,
                         "queue {:#x} ({}), window {:#x} ({}), slot {} of {}", queue_va,
                         queue != nullptr ? "found" : "unknown", window_va,
                         window != nullptr ? "found" : "unknown", slot,
                         window != nullptr ? window->count : 0u);
        return 0;
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->FlushDestroysLocked();
    const u64 seq = ++impl->next_seq;
    {
        std::lock_guard completion_lock{impl->completion_mutex};
        impl->pending.push_back(Impl::Pending{seq, {}});
    }
    queue->last_seq.store(seq);
    window->slot_seq[slot].store(seq);
    ++impl->stats.presents;
    // The uniform snapshot's "written lately" is this frame or the last.
    ++impl->snapshot_cpu.presents;
    impl->sink.Present(PresentRequest{seq, queue->index, window->textures[slot], slot,
                                      window->interval, window->crop});
    return seq;
}

// ---- Completion -----------------------------------------------------------------------------------------

u64 Recorder::LastSeq() const {
    return impl->next_seq.load();
}

u64 Recorder::CompletedSeq() const {
    return impl->completed_seq.load();
}

bool Recorder::WaitForSeq(u64 seq, std::chrono::nanoseconds timeout) {
    std::unique_lock lock{impl->completion_mutex};
    const auto done = [&] { return impl->completed_seq.load() >= seq; };
    if (done()) {
        return true;
    }
    // NVN's "wait forever" is ~0ULL, which wraps to -1 ns as a signed duration: treat any negative
    // timeout as infinite.
    if (timeout.count() < 0 || timeout >= std::chrono::hours{24 * 365}) {
        impl->completion_cv.wait(lock, done);
        return true;
    }
    return impl->completion_cv.wait_for(lock, timeout, done);
}

void Recorder::FlushDestroys() {
    impl->RunRetire();
    std::lock_guard lock{impl->sink_mutex};
    impl->FlushDestroysLocked();
}

// ---- Memory pools ----------------------------------------------------------------------------------------

bool Recorder::MemoryPoolInitialize(u64 pool_va, u64 storage_va, u64 size, u32 flags) {
    MemoryPoolObject* pool = impl->Create(impl->memory_pools, ObjectKind::MemoryPool, pool_va);
    if (!pool) {
        return false;
    }
    constexpr u32 VirtualPoolFlag = 0x200;
    if ((flags & VirtualPoolFlag) != 0 && storage_va == 0) {
        // A virtual pool has no guest storage: give it a GPU address range of its own, so buffer
        // addresses (pool address + offset) resolve to it and to nothing else.
        const u64 span = (size + VIRTUAL_POOL_ALIGNMENT - 1) / VIRTUAL_POOL_ALIGNMENT *
                         VIRTUAL_POOL_ALIGNMENT;
        storage_va = impl->virtual_pool_cursor.fetch_add(span ? span : VIRTUAL_POOL_ALIGNMENT);
    }
    pool->address = storage_va;
    pool->size = size;
    pool->flags = flags;
    std::lock_guard lock{impl->sink_mutex};
    constexpr u32 CpuNoAccessFlag = 0x1;
    if ((flags & (VirtualPoolFlag | CpuNoAccessFlag)) == 0 && storage_va != 0 && size != 0) {
        impl->snapshot_pools[storage_va] = UniformSnapshotPool{storage_va + size};
    }
    impl->FlushDestroysLocked();
    impl->sink.CreatePool(PoolDesc{pool->serial, storage_va, size, flags, 0});
    return true;
}

bool Recorder::MemoryPoolMapVirtual(u64 pool_va, std::span<const VirtualMapRequest> requests) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, pool_va, __func__);
    if (!pool || (pool->flags & 0x200) == 0) {
        NVN_BAD_ARGUMENT(impl->stats, "pool {:#x} is {}", pool_va,
                         pool != nullptr ? "not a virtual pool" : "not a pool");
        return false;
    }
    std::vector<VirtualMapping> mappings;
    mappings.reserve(requests.size());
    for (const VirtualMapRequest& request : requests) {
        u64 physical_serial = 0;
        if (request.physical_pool_va != 0) {
            const MemoryPoolObject* physical = impl->Lookup(
                impl->memory_pools, ObjectKind::MemoryPool, request.physical_pool_va, __func__);
            if (!physical || request.physical_offset > physical->size ||
                request.size > physical->size - request.physical_offset) {
                NVN_BAD_ARGUMENT(impl->stats,
                                 "physical pool {:#x} offset {:#x} size {:#x} ({})",
                                 request.physical_pool_va, request.physical_offset, request.size,
                                 physical != nullptr ? "outside the pool" : "not a pool");
                return false;
            }
            physical_serial = physical->serial;
        }
        if (request.virtual_offset > pool->size || request.size > pool->size - request.virtual_offset) {
            NVN_BAD_ARGUMENT(impl->stats, "virtual range {:#x}+{:#x} outside the {:#x}-byte pool",
                             request.virtual_offset, request.size, pool->size);
            return false;
        }
        mappings.push_back({request.virtual_offset, physical_serial, request.physical_offset,
                            request.size});
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->FlushDestroysLocked();
    impl->sink.MapVirtual(pool->serial, std::move(mappings));
    return true;
}

void Recorder::NotifyCpuWrite(u64 address, u64 size) {
    if (size == 0) {
        return;
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->sink.NotifyCpuWrite(address, size);
}

void Recorder::NotifyHostWrite(u64 address, u64 size) {
    if (size == 0) {
        return;
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->sink.NotifyHostWrite(address, size);
}

u64 Recorder::GpuWriteMarker() {
    // Under the sink mutex, as the commands it counts: every submit this guest thread made is in.
    std::lock_guard lock{impl->sink_mutex};
    return impl->sink.GpuWriteMarker();
}

GpuWriteState Recorder::PollGpuWrites(u64 marker, u64 address, u64 size, u64 max_seq) {
    return impl->sink.PollGpuWrites(marker, address, size, max_seq);
}

void Recorder::NotifyTextureWritten(u64 texture_va) {
    const TextureObject* texture =
        impl->Lookup(impl->textures, ObjectKind::Texture, texture_va, __func__);
    if (!texture) {
        return;
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->sink.NotifyTextureWritten(texture->serial);
}

bool Recorder::FlushTextureToGuest(u64 texture_va) {
    const TextureObject* texture =
        impl->Lookup(impl->textures, ObjectKind::Texture, texture_va, __func__);
    if (!texture) {
        return false;
    }
    const u64 serial = texture->serial;
    // Not under sink_mutex: the call blocks until the replay thread has read the texture back.
    return impl->sink.FlushTextureToGuest(serial);
}

void Recorder::MemoryPoolFinalize(u64 pool_va) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, pool_va, __func__);
    if (!pool) {
        return;
    }
    const u64 low = pool->address;
    const u64 high = pool->address + pool->size;
    const u64 serial = pool->serial;
    {
        std::lock_guard lock{impl->sink_mutex};
        if (const auto it = impl->snapshot_pools.find(low);
            it != impl->snapshot_pools.end() && it->second.end == high) {
            if (it->second.track_id != 0) {
                impl->memory.Untrack(it->second.track_id);
            }
            impl->snapshot_pools.erase(it);
            std::erase_if(impl->snapshot_cpu.written, [low, high](const auto& page) {
                return (page.first << 12) >= low && (page.first << 12) < high;
            });
            SharedUniformSnapshot().Forget(low, high);
        }
    }
    // Lists recorded into the pool's memory die with it: drop them and their references, or the
    // textures and programs they use are never destroyed (VRAM would leak across level loads).
    std::vector<std::shared_ptr<const List>> removed;
    {
        std::lock_guard lock{impl->lists_mutex};
        auto it = impl->lists.lower_bound(low);
        while (it != impl->lists.end() && it->first < high) {
            it = impl->EraseListLocked(it, removed);
        }
    }
    impl->stats.finalized_pool_lists += removed.size();
    for (const auto& list : removed) {
        if (list) {
            impl->ReleaseRefs(list->refs);
        }
    }
    impl->Release(ObjectKind::MemoryPool, pool_va, serial);
    impl->RunRetire();
}

std::optional<PoolInfo> Recorder::GetMemoryPool(u64 pool_va) {
    if (const MemoryPoolObject* pool =
            impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, pool_va, __func__)) {
        return PoolInfo{pool->serial, pool->address, pool->size, pool->flags};
    }
    return std::nullopt;
}

std::optional<PoolInfo> Recorder::QueryMemoryPool(u64 pool_va) {
    if (const MemoryPoolObject* pool = impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool,
                                                    pool_va, __func__, true)) {
        return PoolInfo{pool->serial, pool->address, pool->size, pool->flags};
    }
    return std::nullopt;
}

// ---- Descriptor pools -------------------------------------------------------------------------------------

bool Recorder::TexturePoolInitialize(u64 pool_va_of_tp, u64 memory_pool_va, s64 offset, s32 count) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, memory_pool_va, __func__);
    if (!pool || count <= 0 || offset < 0) {
        NVN_BAD_ARGUMENT(impl->stats, "texture pool {:#x}: memory pool {:#x} ({}), count {}, offset {}",
                         pool_va_of_tp, memory_pool_va, pool != nullptr ? "found" : "not a pool",
                         count, offset);
        return false;
    }
    DescriptorPoolObject* object =
        impl->Create(impl->texture_pools, ObjectKind::TexturePool, pool_va_of_tp);
    if (!object) {
        return false;
    }
    object->pool_serial = pool->serial;
    object->offset = static_cast<u64>(offset);
    object->count = static_cast<u32>(count);
    object->base = impl->texture_table.Allocate(object->count);
    return true;
}

void Recorder::TexturePoolFinalize(u64 texture_pool_va) {
    if (const DescriptorPoolObject* object = impl->Lookup(
            impl->texture_pools, ObjectKind::TexturePool, texture_pool_va, __func__)) {
        impl->texture_table.Release(object->base);
        impl->Release(ObjectKind::TexturePool, texture_pool_va, object->serial);
    }
}

std::optional<DescriptorPoolInfo> Recorder::GetTexturePool(u64 texture_pool_va) {
    if (const DescriptorPoolObject* object = impl->Lookup(
            impl->texture_pools, ObjectKind::TexturePool, texture_pool_va, __func__)) {
        return DescriptorPoolInfo{object->serial, object->pool_serial, object->offset,
                                  object->count, object->base};
    }
    return std::nullopt;
}

namespace {
bool PlainView(const TextureViewDesc& view) {
    return !view.set_levels && !view.set_layers && !view.set_format && !view.set_target &&
           !view.set_swizzle && !view.set_dsmode && !view.is_image;
}
} // namespace

bool Recorder::Impl::ReadView(u64 view_va, TextureViewDesc& view) {
    std::memset(&view, 0, sizeof(view));
    if (view_va == 0) {
        return true;
    }
    GuestTextureView guest{};
    if (!GuestRead(view_va, &guest, sizeof(guest))) {
        return false;
    }
    view.set_levels = (guest.set_bits & GUEST_VIEW_SET_LEVELS) ? 1 : 0;
    view.set_layers = (guest.set_bits & GUEST_VIEW_SET_LAYERS) ? 1 : 0;
    view.set_format = (guest.set_bits & GUEST_VIEW_SET_FORMAT) ? 1 : 0;
    view.set_target = (guest.set_bits & GUEST_VIEW_SET_TARGET) ? 1 : 0;
    view.set_swizzle = (guest.set_bits & GUEST_VIEW_SET_SWIZZLE) ? 1 : 0;
    view.set_dsmode = (guest.set_bits & GUEST_VIEW_SET_DSMODE) ? 1 : 0;
    view.dsmode = static_cast<u8>(guest.set_bits >> 6);
    view.target = guest.target;
    view.base_level = guest.base_level;
    view.nlevels = guest.nlevels;
    view.format = guest.format;
    view.min_layer = guest.min_layer;
    view.nlayers = guest.nlayers;
    std::memcpy(view.swizzle, guest.swizzle, sizeof(view.swizzle));
    return true;
}

namespace {
void RegisterTextureImpl(Recorder::Impl& impl, u64 texture_pool_va, s32 id, u64 texture_va,
                         u64 view_va, bool image, const char* function) {
    const DescriptorPoolObject* pool =
        impl.Lookup(impl.texture_pools, ObjectKind::TexturePool, texture_pool_va, function);
    if (!pool) {
        return;
    }
    if (id < 0 || static_cast<u32>(id) >= pool->count) {
        NVN_BAD_ARGUMENT_IN(impl.stats, function, "texture pool {:#x}: id {} of {} descriptors",
                            texture_pool_va, id, pool->count);
        return;
    }
    u64 serial = 0;
    if (texture_va != 0) {
        TextureObject* texture = impl.Lookup(impl.textures, ObjectKind::Texture, texture_va, function);
        if (!texture) {
            return;
        }
        TextureViewDesc view;
        if (!impl.ReadView(view_va, view)) {
            return;
        }
        view.is_image = image ? 1 : 0;
        serial = PlainView(view) ? texture->serial : impl.ViewSerialFor(*texture, view);
    }
    impl.texture_table.Set(pool->base, static_cast<u32>(id), serial);
}
} // namespace

void Recorder::RegisterTexture(u64 texture_pool_va, s32 id, u64 texture_va, u64 view_va) {
    RegisterTextureImpl(*impl, texture_pool_va, id, texture_va, view_va, false, __func__);
}

void Recorder::RegisterImage(u64 texture_pool_va, s32 id, u64 texture_va, u64 view_va) {
    RegisterTextureImpl(*impl, texture_pool_va, id, texture_va, view_va, true, __func__);
}

bool Recorder::SamplerPoolInitialize(u64 sampler_pool_va, u64 memory_pool_va, s64 offset,
                                     s32 count) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, memory_pool_va, __func__);
    if (!pool || count <= 0 || offset < 0) {
        NVN_BAD_ARGUMENT(impl->stats, "sampler pool {:#x}: memory pool {:#x} ({}), count {}, offset {}",
                         sampler_pool_va, memory_pool_va, pool != nullptr ? "found" : "not a pool",
                         count, offset);
        return false;
    }
    DescriptorPoolObject* object =
        impl->Create(impl->sampler_pools, ObjectKind::SamplerPool, sampler_pool_va);
    if (!object) {
        return false;
    }
    object->pool_serial = pool->serial;
    object->offset = static_cast<u64>(offset);
    object->count = static_cast<u32>(count);
    object->base = impl->sampler_table.Allocate(object->count);
    return true;
}

void Recorder::SamplerPoolFinalize(u64 sampler_pool_va) {
    if (const DescriptorPoolObject* object = impl->Lookup(
            impl->sampler_pools, ObjectKind::SamplerPool, sampler_pool_va, __func__)) {
        impl->sampler_table.Release(object->base);
        impl->Release(ObjectKind::SamplerPool, sampler_pool_va, object->serial);
    }
}

std::optional<DescriptorPoolInfo> Recorder::GetSamplerPool(u64 sampler_pool_va) {
    if (const DescriptorPoolObject* object = impl->Lookup(
            impl->sampler_pools, ObjectKind::SamplerPool, sampler_pool_va, __func__)) {
        return DescriptorPoolInfo{object->serial, object->pool_serial, object->offset,
                                  object->count, object->base};
    }
    return std::nullopt;
}

void Recorder::RegisterSampler(u64 sampler_pool_va, s32 id, u64 sampler_va) {
    const DescriptorPoolObject* pool = impl->Lookup(impl->sampler_pools, ObjectKind::SamplerPool,
                                                    sampler_pool_va, __func__);
    if (!pool) {
        return;
    }
    if (id < 0 || static_cast<u32>(id) >= pool->count) {
        NVN_BAD_ARGUMENT(impl->stats, "sampler pool {:#x}: id {} of {} descriptors", sampler_pool_va,
                         id, pool->count);
        return;
    }
    u64 serial = 0;
    if (sampler_va != 0) {
        const SamplerObject* sampler =
            impl->Lookup(impl->samplers, ObjectKind::Sampler, sampler_va, __func__);
        if (!sampler) {
            return;
        }
        serial = sampler->serial;
    }
    impl->sampler_table.Set(pool->base, static_cast<u32>(id), serial);
}

void Recorder::RegisterSamplerParams(u64 sampler_pool_va, s32 id, const SamplerDesc& desc) {
    const DescriptorPoolObject* pool = impl->Lookup(impl->sampler_pools, ObjectKind::SamplerPool,
                                                    sampler_pool_va, __func__);
    if (!pool) {
        return;
    }
    if (id < 0 || static_cast<u32>(id) >= pool->count) {
        NVN_BAD_ARGUMENT(impl->stats, "sampler pool {:#x}: id {} of {} descriptors", sampler_pool_va,
                         id, pool->count);
        return;
    }
    impl->sampler_table.Set(pool->base, static_cast<u32>(id), impl->AnonymousSamplerSerial(desc));
}

// ---- Textures, samplers, buffers ---------------------------------------------------------------------

bool Recorder::TextureInitialize(u64 texture_va, const TextureDesc& desc, u64 memory_pool_va,
                                 u64 pool_offset) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, memory_pool_va, __func__);
    if (!pool) {
        return false;
    }
    const TextureObject* previous = nullptr;
    if (GuestTag old; impl->ReadTag(texture_va, old) && old.magic == GUEST_TAG_MAGIC &&
        old.kind == static_cast<u32>(ObjectKind::Texture)) {
        previous = impl->textures.Get(old.serial);
    }
    const u64 previous_serial = previous ? previous->serial : 0;
    const auto previous_views = previous ? previous->views : decltype(previous->views){};
    TextureObject* texture = impl->Create(impl->textures, ObjectKind::Texture, texture_va);
    if (!texture) {
        return false;
    }
    texture->desc = desc;
    texture->desc.serial = texture->serial;
    texture->desc.pool_serial = pool->serial;
    texture->desc.pool_offset = pool_offset;
    texture->desc.view_of = 0;
    std::memset(&texture->desc.view, 0, sizeof(texture->desc.view));
    {
        std::lock_guard lock{impl->sink_mutex};
        impl->FlushDestroysLocked();
        impl->sink.CreateTexture(texture->desc);
    }
    if (previous) {
        // A texture pool entry names the object at its guest address. Reinitializing that address
        // gives it a new serial, so future submissions must resolve the same registrations to the
        // new object (and its views), while old submissions keep their immutable snapshots.
        impl->texture_table.ReplaceSerial(previous_serial, texture->serial);
        for (const auto& [view, old_view_serial] : previous_views) {
            impl->texture_table.ReplaceSerial(old_view_serial,
                                              impl->ViewSerialFor(*texture, view));
        }
    }
    return true;
}

void Recorder::TextureFinalize(u64 texture_va) {
    if (const TextureObject* texture =
            impl->Lookup(impl->textures, ObjectKind::Texture, texture_va, __func__)) {
        impl->Release(ObjectKind::Texture, texture_va, texture->serial);
    }
}

std::optional<TextureDesc> Recorder::GetTexture(u64 texture_va) {
    if (const TextureObject* texture =
            impl->Lookup(impl->textures, ObjectKind::Texture, texture_va, __func__)) {
        return texture->desc;
    }
    return std::nullopt;
}

bool Recorder::SamplerInitialize(u64 sampler_va, const SamplerDesc& desc) {
    SamplerObject* sampler = impl->Create(impl->samplers, ObjectKind::Sampler, sampler_va);
    if (!sampler) {
        return false;
    }
    sampler->desc = desc;
    sampler->desc.serial = sampler->serial;
    std::lock_guard lock{impl->sink_mutex};
    impl->FlushDestroysLocked();
    impl->sink.CreateSampler(sampler->desc);
    return true;
}

void Recorder::SamplerFinalize(u64 sampler_va) {
    if (const SamplerObject* sampler =
            impl->Lookup(impl->samplers, ObjectKind::Sampler, sampler_va, __func__)) {
        impl->Release(ObjectKind::Sampler, sampler_va, sampler->serial);
    }
}

std::optional<SamplerDesc> Recorder::GetSampler(u64 sampler_va) {
    if (const SamplerObject* sampler =
            impl->Lookup(impl->samplers, ObjectKind::Sampler, sampler_va, __func__)) {
        return sampler->desc;
    }
    return std::nullopt;
}

bool Recorder::BufferInitialize(u64 buffer_va, u64 memory_pool_va, u64 offset, u64 size) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, memory_pool_va, __func__);
    if (!pool) {
        return false;
    }
    BufferObject* buffer = impl->Create(impl->buffers, ObjectKind::Buffer, buffer_va);
    if (!buffer) {
        return false;
    }
    buffer->pool_serial = pool->serial;
    const u64 fields[2] = {offset, size};
    if (!impl->GuestWrite(buffer_va + GUEST_BUFFER_OFFSET_OFFSET, fields, sizeof(fields))) {
        impl->Release(ObjectKind::Buffer, buffer_va, buffer->serial);
        return false;
    }
    const u64 begin = pool->address + offset;
    const u64 end = begin + size;
    buffer->gpu_begin = begin;
    buffer->gpu_end = end;
    if (size != 0) {
        const std::unique_lock lock{impl->buffer_spans_mutex};
        if (size > Impl::BUFFER_SPAN_LARGE_LIMIT) {
            impl->buffer_spans_large.push_back({begin, end, buffer->serial});
        } else {
            const u64 first_chunk = begin >> Impl::BUFFER_SPAN_CHUNK_SHIFT;
            const u64 last_chunk = (end - 1) >> Impl::BUFFER_SPAN_CHUNK_SHIFT;
            for (u64 chunk = first_chunk; chunk <= last_chunk; ++chunk) {
                impl->buffer_span_chunks[chunk].push_back({begin, end, buffer->serial});
            }
        }
    }
    return true;
}

void Recorder::BufferFinalize(u64 buffer_va) {
    if (const BufferObject* buffer =
            impl->Lookup(impl->buffers, ObjectKind::Buffer, buffer_va, __func__)) {
        const u64 begin = buffer->gpu_begin;
        const u64 end = buffer->gpu_end;
        const u64 serial = buffer->serial;
        if (end > begin) {
            const auto erase_serial = [serial](std::vector<Impl::BufferSpan>& spans) {
                for (size_t i = spans.size(); i-- > 0;) {
                    if (spans[i].serial == serial) {
                        spans[i] = spans.back();
                        spans.pop_back();
                    }
                }
            };
            const std::unique_lock lock{impl->buffer_spans_mutex};
            if (end - begin > Impl::BUFFER_SPAN_LARGE_LIMIT) {
                erase_serial(impl->buffer_spans_large);
            } else {
                const u64 first_chunk = begin >> Impl::BUFFER_SPAN_CHUNK_SHIFT;
                const u64 last_chunk = (end - 1) >> Impl::BUFFER_SPAN_CHUNK_SHIFT;
                for (u64 chunk = first_chunk; chunk <= last_chunk; ++chunk) {
                    const auto it = impl->buffer_span_chunks.find(chunk);
                    if (it == impl->buffer_span_chunks.end()) {
                        continue;
                    }
                    erase_serial(it->second);
                    if (it->second.empty()) {
                        impl->buffer_span_chunks.erase(it);
                    }
                }
            }
        }
        impl->Release(ObjectKind::Buffer, buffer_va, serial);
    }
}

u32 Recorder::Impl::BufferBytesFrom(u64 address) {
    const std::shared_lock lock{buffer_spans_mutex};
    u64 end = 0;
    if (const auto it = buffer_span_chunks.find(address >> BUFFER_SPAN_CHUNK_SHIFT);
        it != buffer_span_chunks.end()) {
        for (const BufferSpan& span : it->second) {
            if (span.begin <= address && address < span.end) {
                end = (std::max)(end, span.end);
            }
        }
    }
    for (const BufferSpan& span : buffer_spans_large) {
        if (span.begin <= address && address < span.end) {
            end = (std::max)(end, span.end);
        }
    }
    return static_cast<u32>((std::min<u64>)(end != 0 ? end - address : 0, 0xFFFF'FFFFu));
}

std::optional<BufferInfo> Recorder::GetBuffer(u64 buffer_va) {
    const BufferObject* buffer =
        impl->Lookup(impl->buffers, ObjectKind::Buffer, buffer_va, __func__);
    if (!buffer) {
        return std::nullopt;
    }
    u64 fields[2]{};
    if (!impl->GuestRead(buffer_va + GUEST_BUFFER_OFFSET_OFFSET, fields, sizeof(fields))) {
        return std::nullopt;
    }
    const MemoryPoolObject* pool = impl->memory_pools.Get(buffer->pool_serial);
    if (!pool) {
        ++impl->stats.stale_serial;
        return std::nullopt;
    }
    return BufferInfo{buffer->serial, buffer->pool_serial, pool->address + fields[0], fields[0],
                      fields[1]};
}

// ---- Programs ---------------------------------------------------------------------------------------------

namespace {

struct ControlSection {
    std::vector<u8> bytes;
    u32 code_size = 0;
    s32 stage = -1;
};

bool ReadControlSection(Recorder::Impl& impl, u64 va, ControlSection& out) {
    u8 header[0x58]{};
    if (!impl.GuestRead(va, header, sizeof(header))) {
        return false;
    }
    u32 magic = 0;
    u32 version = 0;
    std::memcpy(&magic, header, 4);
    std::memcpy(&version, header + 4, 4);
    if (magic == NVN2_CONTROL_MAGIC && version == NVN2_CONTROL_VERSION) {
        std::memcpy(&out.code_size, header + 0x44, 4);
        u16 stage = 0;
        std::memcpy(&stage, header + 0x56, 2);
        out.stage = stage < PROGRAM_STAGES ? stage : -1;
        // [UNVERIFIED] full length of an NVN2 control section: 0x100 covers the compute c0 fields
        // (+0xa9..+0xba); fall back to the 0x58-byte header.
        out.bytes.resize(0x100);
        if (!impl.memory.Read(va, out.bytes.data(), out.bytes.size())) {
            out.bytes.assign(header, header + sizeof(header));
        }
        return true;
    }
    u32 own = 0;
    std::memcpy(&own, header + 0x20, 4);
    const u32 length = own >= 0x760 && own < 0x880 ? own : 0x880;
    out.bytes.resize(length);
    if (!impl.GuestRead(va, out.bytes.data(), out.bytes.size())) {
        return false;
    }
    std::memcpy(&out.code_size, out.bytes.data() + 0x704, 4);
    return true;
}

} // namespace

bool Recorder::ProgramInitialize(u64 program_va) {
    return impl->Create(impl->programs, ObjectKind::Program, program_va) != nullptr;
}

void Recorder::ProgramFinalize(u64 program_va) {
    if (const ProgramObject* program =
            impl->Lookup(impl->programs, ObjectKind::Program, program_va, __func__)) {
        impl->Release(ObjectKind::Program, program_va, program->serial);
    }
}

bool Recorder::ProgramSetShaders(u64 program_va, std::span<const ShaderStageInput> stages) {
    if (stages.size() > PROGRAM_STAGES) {
        NVN_BAD_ARGUMENT(impl->stats, "{} shader stages (limit {})", stages.size(), PROGRAM_STAGES);
        return false;
    }
    ProgramObject* old = impl->Lookup(impl->programs, ObjectKind::Program, program_va, __func__);
    if (!old) {
        return false;
    }
    std::array<ControlSection, PROGRAM_STAGES> controls;
    std::array<s32, PROGRAM_STAGES> assigned{};
    bool use_hints = true;
    u32 seen = 0;
    for (size_t i = 0; i < stages.size(); ++i) {
        if (!ReadControlSection(*impl, stages[i].control_address, controls[i])) {
            return false;
        }
        s32 stage = stages[i].stage >= 0 ? stages[i].stage : controls[i].stage;
        if (stage < 0 || stage >= static_cast<s32>(PROGRAM_STAGES) || (seen & (1U << stage))) {
            use_hints = false;
        } else {
            seen |= 1U << stage;
        }
        assigned[i] = stage;
    }
    if (!use_hints) {
        ++impl->stats.stage_fallback;
        for (size_t i = 0; i < stages.size(); ++i) {
            assigned[i] = static_cast<s32>(i);
        }
    }
    ProgramObject* program =
        impl->Create(impl->programs, ObjectKind::Program, program_va, /*count_reinit=*/false);
    if (!program) {
        return false;
    }
    program->desc.serial = program->serial;
    for (size_t i = 0; i < stages.size(); ++i) {
        const u32 stage = static_cast<u32>(assigned[i]);
        program->desc.stage_mask |= 1U << stage;
        program->desc.control[stage] = std::move(controls[i].bytes);
        const u32 code_size = controls[i].code_size;
        if (stages[i].data_address != 0 && code_size != 0 && code_size < (1U << 24)) {
            program->code_address[stage] = stages[i].data_address;
            program->code_size[stage] = code_size;
            program->pending_mask |= 1U << stage;
        }
    }
    impl->MarkFinalized(old->serial);
    std::lock_guard lock{impl->sink_mutex};
    impl->EnsureProgramLocked(*program, false);
    return true;
}

bool Recorder::ProgramSetShaders(u64 program_va, s32 count, u64 shader_data_va) {
    if (count < 0 || count > static_cast<s32>(PROGRAM_STAGES)) {
        NVN_BAD_ARGUMENT(impl->stats, "{} shaders (limit {})", count, PROGRAM_STAGES);
        return false;
    }
    std::array<GuestShaderData, PROGRAM_STAGES> data{};
    if (count > 0 && !impl->GuestRead(shader_data_va, data.data(), sizeof(GuestShaderData) * count)) {
        return false;
    }
    std::array<ShaderStageInput, PROGRAM_STAGES> inputs{};
    for (s32 i = 0; i < count; ++i) {
        inputs[i] = ShaderStageInput{data[i].data, data[i].control, -1};
    }
    return ProgramSetShaders(program_va, std::span<const ShaderStageInput>{inputs.data(),
                                                                          static_cast<size_t>(count)});
}

// ---- Sync / events -------------------------------------------------------------------------------------

bool Recorder::SyncInitialize(u64 sync_va) {
    return impl->Create(impl->syncs, ObjectKind::Sync, sync_va) != nullptr;
}

void Recorder::SyncFinalize(u64 sync_va) {
    if (const SyncObject* sync = impl->Lookup(impl->syncs, ObjectKind::Sync, sync_va, __func__)) {
        impl->Release(ObjectKind::Sync, sync_va, sync->serial);
    }
}

bool Recorder::SyncSetSignaled(u64 sync_va) {
    // The SDK overwrites the whole sync object: an object that is not a live sync becomes one.
    // A live sync keeps its host object (recreating it every frame would pile up objects).
    SyncObject* sync = nullptr;
    GuestTag tag{};
    if (sync_va != 0 && impl->memory.Read(sync_va, &tag, sizeof(tag)) &&
        tag.magic == GUEST_TAG_MAGIC && tag.kind == static_cast<u32>(ObjectKind::Sync)) {
        sync = impl->syncs.Get(tag.serial);
    }
    if (!sync) {
        sync = impl->Create(impl->syncs, ObjectKind::Sync, sync_va);
    }
    if (!sync) {
        return false;
    }
    sync->fence_seq.store(SYNC_SIGNALED_WITHOUT_FENCE);
    return true;
}

std::optional<SyncState> Recorder::GetSyncState(u64 sync_va) {
    const SyncObject* sync = impl->Lookup(impl->syncs, ObjectKind::Sync, sync_va, __func__);
    if (!sync) {
        return std::nullopt;
    }
    const u64 fence = sync->fence_seq.load();
    if (fence == SYNC_SIGNALED_WITHOUT_FENCE) {
        return SyncState{false, true, 0};
    }
    return SyncState{fence != 0, fence != 0 && impl->completed_seq.load() >= fence, fence};
}

SyncWaitResult Recorder::SyncWait(u64 sync_va, std::chrono::nanoseconds timeout) {
    const std::optional<SyncState> state = GetSyncState(sync_va);
    if (!state) {
        return SyncWaitResult::Failed;
    }
    if (state->signaled) {
        return SyncWaitResult::AlreadySignaled;
    }
    if (!state->fenced) {
        return SyncWaitResult::Failed;
    }
    return WaitForSeq(state->fence_seq, timeout) ? SyncWaitResult::ConditionSatisfied
                                                 : SyncWaitResult::TimeoutExpired;
}

bool Recorder::EventInitialize(u64 event_va, u64 memory_pool_va, s64 offset) {
    const MemoryPoolObject* pool =
        impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, memory_pool_va, __func__);
    if (!pool || offset < 0) {
        NVN_BAD_ARGUMENT(impl->stats, "event {:#x}: memory pool {:#x} ({}), offset {}", event_va,
                         memory_pool_va, pool != nullptr ? "found" : "not a pool", offset);
        return false;
    }
    EventObject* event = impl->Create(impl->events, ObjectKind::Event, event_va);
    if (!event) {
        return false;
    }
    event->address = pool->address + static_cast<u64>(offset);
    {
        // A new event over this memory starts from whatever the guest word holds.
        std::lock_guard lock{impl->sink_mutex};
        impl->sink.NotifyEventCpu(event->address, EventCpuChange::Forget, 0);
    }
    return true;
}

void Recorder::EventFinalize(u64 event_va) {
    if (const EventObject* event = impl->Lookup(impl->events, ObjectKind::Event, event_va, __func__)) {
        impl->Release(ObjectKind::Event, event_va, event->serial);
    }
}

std::optional<u32> Recorder::EventGetValue(u64 event_va) {
    const EventObject* event = impl->Lookup(impl->events, ObjectKind::Event, event_va, __func__);
    u32 value = 0;
    if (!event || !impl->GuestRead(event->address, &value, sizeof(value))) {
        return std::nullopt;
    }
    return value;
}

void Recorder::EventSignal(u64 event_va, u32 mode, u32 value) {
    const EventObject* event = impl->Lookup(impl->events, ObjectKind::Event, event_va, __func__);
    if (!event) {
        return;
    }
    {
        std::lock_guard event_lock{impl->event_mutex};
        u32 written = value;
        if (mode == 1) {
            u32 current = 0;
            if (impl->GuestRead(event->address, &current, sizeof(current))) {
                written += current;
            }
        }
        impl->GuestWrite(event->address, &written, sizeof(written));
    }
    std::lock_guard lock{impl->sink_mutex};
    impl->sink.NotifyEventCpu(event->address,
                              mode == 1 ? EventCpuChange::Add : EventCpuChange::Write, value);
}

// ---- Window ------------------------------------------------------------------------------------------

bool Recorder::WindowInitialize(u64 window_va, std::span<const u64> texture_vas,
                                u32 present_interval) {
    if (texture_vas.size() > 4) {
        NVN_BAD_ARGUMENT(impl->stats, "{} window textures (limit 4)", texture_vas.size());
        return false;
    }
    std::array<u64, 4> serials{};
    for (size_t i = 0; i < texture_vas.size(); ++i) {
        const TextureObject* texture =
            impl->Lookup(impl->textures, ObjectKind::Texture, texture_vas[i], __func__);
        if (!texture) {
            return false;
        }
        serials[i] = texture->serial;
    }
    WindowObject* window = impl->Create(impl->windows, ObjectKind::Window, window_va);
    if (!window) {
        return false;
    }
    window->count = static_cast<u32>(texture_vas.size());
    std::copy(serials.begin(), serials.end(), window->textures);
    window->interval = present_interval;
    return true;
}

void Recorder::WindowFinalize(u64 window_va) {
    if (const WindowObject* window =
            impl->Lookup(impl->windows, ObjectKind::Window, window_va, __func__)) {
        impl->Release(ObjectKind::Window, window_va, window->serial);
    }
}

void Recorder::WindowSetPresentInterval(u64 window_va, u32 interval) {
    if (WindowObject* window = impl->Lookup(impl->windows, ObjectKind::Window, window_va, __func__)) {
        window->interval = interval;
    }
}

void Recorder::WindowSetCrop(u64 window_va, PresentCrop crop) {
    if (WindowObject* window = impl->Lookup(impl->windows, ObjectKind::Window, window_va, __func__)) {
        std::lock_guard lock{impl->sink_mutex};
        window->crop = crop;
    }
}

u64 Recorder::WindowSlotSeq(u64 window_va, u32 slot) {
    const WindowObject* window =
        impl->Lookup(impl->windows, ObjectKind::Window, window_va, __func__);
    return window && slot < window->count ? window->slot_seq[slot].load() : 0;
}

} // namespace VideoCommon::Nvn
