// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <cstring>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/core.h"
#include "video_core/nvn/recorder_impl.h"
#include "video_core/nvn/timestamp_mode.h"

namespace VideoCommon::Nvn {

namespace {

constexpr u32 COPY_FLAG_RESOLVE = 0x40000000; // nxnative DM_COPY_RESOLVE

std::vector<u8>& Scratch() {
    thread_local std::vector<u8> scratch;
    return scratch;
}

} // namespace

// ---- Command memory and records ------------------------------------------------------------------

CommandBufferObject* Recorder::Impl::Recording(u64 cmd_va, const char* function) {
    CommandBufferObject* cb = Lookup(command_buffers, ObjectKind::CommandBuffer, cmd_va, function);
    if (!cb) {
        ++stats.records_dropped;
        return nullptr;
    }
    if (cb->recording == 0) {
        NVN_WARN_LIMITED(stats.bad_argument, "[nvn] {} outside BeginRecording/EndRecording",
                         function);
        ++stats.records_dropped;
        return nullptr;
    }
    return cb;
}

void Recorder::Impl::NewSegment(CommandBufferObject& cb, u64 start, u64 end_va) {
    const u64 aligned = Align8(start);
    if (cb.recording == 1 && cb.cur) {
        // The recording continues elsewhere: close the piece in the old segment. The real driver
        // writes a 16-byte jump there, which GetCommandMemoryUsed counts.
        cb.pieces.emplace_back(cb.piece_start, cb.cur);
        cb.used += 16;
    }
    cb.cur = aligned;
    cb.end = end_va;
    cb.piece_start = aligned;
}

void Recorder::Impl::NewSyntheticSegment(CommandBufferObject& cb, u64 needed) {
    const u64 size = std::max<u64>(SYNTHETIC_SEGMENT_SIZE, (needed + 16 + 0xFFF) & ~u64{0xFFF});
    const u64 start = synthetic_cursor.fetch_add(size);
    ++stats.synthetic_segments;
    NewSegment(cb, start, start + size);
}

u64* Recorder::Impl::AllocRecord(CommandBufferObject& cb, Op op, u16 flags, u32 payload_bytes,
                                 u32 keep) {
    const u32 size = static_cast<u32>(8 + Align8(payload_bytes));
    if (!cb.cur || cb.cur + size + keep > cb.end) {
        // Out of command memory: the game's callback is never invoked, recording continues in a
        // synthetic segment.
        ++stats.callbacks_skipped;
        NewSyntheticSegment(cb, size + keep);
    }
    if (cb.recording == 2) {
        cb.begin = cb.cur;
        cb.piece_start = cb.cur;
        cb.recording = 1;
    }
    cb.cur += size;
    cb.used += size;
    const size_t index = cb.words.size();
    cb.words.resize(index + size / 8);
    *reinterpret_cast<RecordHeader*>(&cb.words[index]) =
        RecordHeader{static_cast<u16>(op), flags, size};
    ++stats.records_recorded;
    return &cb.words[index];
}

bool Recorder::Impl::ResolveTarget(CommandBufferObject& cb, u64 texture_va, u64 view_va,
                                   RenderTargetRef& out, const char* function,
                                   TextureObject** texture_out) {
    std::memset(&out, 0, sizeof(out));
    if (texture_out) {
        *texture_out = nullptr;
    }
    if (texture_va == 0) {
        return true;
    }
    TextureObject* texture = Lookup(textures, ObjectKind::Texture, texture_va, function);
    if (!texture) {
        return false;
    }
    if (!ReadView(view_va, out.view)) {
        return false;
    }
    out.texture.serial = texture->serial;
    out.texture.level = out.view.set_levels ? out.view.base_level : 0;
    out.texture.layer = out.view.set_layers ? out.view.min_layer : 0;
    AddRef(cb, texture->serial);
    if (texture_out) {
        *texture_out = texture;
    }
    return true;
}

#define RECORDING(cb)                                                                             \
    CommandBufferObject* cb = impl->Recording(cmd_va, __func__);                                  \
    if (!cb) {                                                                                    \
        return;                                                                                   \
    }

#define DROP()                                                                                    \
    do {                                                                                          \
        static Common::BootCounter nvn_logged;                                                    \
        if (nvn_logged.Next() < 4) {                                                              \
            LOG_WARNING(Render_Vulkan, "[nvn] {} dropped a record", __func__);                    \
        }                                                                                         \
        ++impl->stats.records_dropped;                                                            \
        return;                                                                                   \
    } while (false)

#define BAD_ARGUMENT()                                                                            \
    do {                                                                                          \
        static Common::BootCounter nvn_logged;                                                    \
        if (nvn_logged.Next() < 4) {                                                              \
            LOG_WARNING(Render_Vulkan, "[nvn] {} dropped a record: bad argument (NVN call {})",   \
                        __func__, CurrentNvnCall());                                              \
        }                                                                                         \
        ++impl->stats.bad_argument;                                                               \
        ++impl->stats.records_dropped;                                                            \
        return;                                                                                   \
    } while (false)

#define BAD_ARGUMENT_DETAIL(...)                                                                  \
    do {                                                                                          \
        static Common::BootCounter nvn_detail_logged;                                             \
        if (nvn_detail_logged.Next() < 6) {                                                       \
            LOG_WARNING(Render_Vulkan, __VA_ARGS__);                                              \
        }                                                                                         \
    } while (false)

// ---- Lifetime and memory ------------------------------------------------------------------------------

bool Recorder::CommandBufferInitialize(u64 cmd_va) {
    return impl->Create(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va) != nullptr;
}

void Recorder::CommandBufferFinalize(u64 cmd_va) {
    if (CommandBufferObject* cb =
            impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__)) {
        impl->AbandonRecording(*cb);
        impl->Release(ObjectKind::CommandBuffer, cmd_va, cb->serial);
    }
}

void Recorder::SetMemoryCallback(u64 cmd_va, u64 callback_va) {
    if (CommandBufferObject* cb =
            impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__)) {
        cb->callback = callback_va;
    }
}

u64 Recorder::GetMemoryCallback(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb ? cb->callback : 0;
}

u64 Recorder::GetMemoryCallbackData(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb ? cb->callback_data : 0;
}

void Recorder::SetMemoryCallbackData(u64 cmd_va, u64 data_va) {
    if (CommandBufferObject* cb =
            impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__)) {
        cb->callback_data = data_va;
    }
}

void Recorder::AddCommandMemory(u64 cmd_va, u64 memory_pool_va, s64 offset, u64 size) {
    CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    if (!cb) {
        return;
    }
    const MemoryPoolObject* pool =
        memory_pool_va
            ? impl->Lookup(impl->memory_pools, ObjectKind::MemoryPool, memory_pool_va, __func__)
            : nullptr;
    if (!pool || size < 64) {
        // An empty pool or a tiny block: the game cannot write commands there. Keep recording in
        // a synthetic segment (GetCommandMemoryUsed keeps counting).
        impl->NewSyntheticSegment(*cb, size);
        return;
    }
    // The SDK (OoT's sdk+0x64c0d0, nvnCommandBufferAddCommandMemory) takes the block as given:
    // cursor = pool address + offset, end = cursor + size, with no check against the pool size.
    // Ocarina of Time's command-memory ring hands out 0x200-byte blocks whose end passes its
    // 1 MiB pool near the wrap (offset 0xffe30, size 0x200) and records only what still fits.
    const u64 start = pool->address + static_cast<u64>(offset);
    impl->NewSegment(*cb, start, start + size);
    cb->used = 0; // counted from the memory the game added last, across recordings
}

void Recorder::AddControlMemory(u64 cmd_va, u64 size) {
    if (CommandBufferObject* cb =
            impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__)) {
        cb->control_size = size; // no control commands are recorded: the whole block stays free
    }
}

u64 Recorder::GetCommandMemoryUsed(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb ? cb->used : 0;
}

u64 Recorder::GetCommandMemoryFree(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb && cb->end > cb->cur ? cb->end - cb->cur : 0;
}

u64 Recorder::GetCommandMemorySize(u64 cmd_va) {
    return GetCommandMemoryUsed(cmd_va) + GetCommandMemoryFree(cmd_va);
}

u64 Recorder::GetControlMemoryFree(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb ? cb->control_size : 0;
}

u64 Recorder::GetControlMemorySize(u64 cmd_va) {
    return GetControlMemoryFree(cmd_va);
}

void Recorder::BeginRecording(u64 cmd_va) {
    CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    if (!cb) {
        return;
    }
    if (cb->recording != 0) {
        NVN_WARN_LIMITED(impl->stats.bad_argument, "[nvn] BeginRecording re-entered");
    }
    impl->AbandonRecording(*cb); // drops the references of a recording that never ended
    cb->recording = 2; // the first record is the handle: there is no BEGIN record
}

u64 Recorder::EndRecording(u64 cmd_va) {
    CommandBufferObject* cb = impl->Recording(cmd_va, __func__);
    if (!cb) {
        return 0;
    }
    impl->AllocRecord(*cb, Op::End, 0, 0, 0); // nothing follows: the End may take the last bytes
    cb->pieces.emplace_back(cb->piece_start, cb->cur);

    auto list = std::make_shared<List>();
    list->words = std::move(cb->words);
    list->refs = std::move(cb->refs);
    list->actions = std::move(cb->actions);
    std::sort(list->refs.begin(), list->refs.end());
    // AddRef took one reference per entry; the list keeps exactly one per distinct serial.
    std::vector<u64> extra;
    for (size_t i = 1; i < list->refs.size(); ++i) {
        if (list->refs[i] == list->refs[i - 1]) {
            extra.push_back(list->refs[i]);
        }
    }
    impl->ReleaseRefs(extra);
    list->refs.erase(std::unique(list->refs.begin(), list->refs.end()), list->refs.end());
    // Index the records that need work at submit and retain synthetic callees.
    const u64* const words = list->words.data();
    RecordReader reader{words, list->words.size()};
    size_t offset = 0;
    while (const RecordHeader* header = reader.Next()) {
        const Op op = RecordOp(header);
        if (op == Op::CallList) {
            const u64 callee = RecordPayload<CallListPayload>(header)->handle;
            if (callee >= SYNTHETIC_BASE) {
                list->callees.push_back(callee);
            }
            list->expansion_offsets.push_back(offset);
        } else if ((header->flags & RECORD_FLAG_DEFERRED) && IsDeferrableOp(op)) {
            list->expansion_offsets.push_back(offset);
        }
        offset = static_cast<size_t>(reader.Cursor() - words);
    }
    list->record_end = offset;
    std::sort(list->callees.begin(), list->callees.end());
    list->callees.erase(std::unique(list->callees.begin(), list->callees.end()),
                        list->callees.end());

    const u64 handle = cb->begin;
    std::vector<std::pair<u64, u64>> ranges = std::move(cb->pieces);
    cb->words.clear();
    cb->refs.clear();
    cb->actions.clear();
    cb->pieces.clear();
    cb->ref_cache.fill(0);
    cb->recording = 0;
    impl->InsertList(handle, ranges, std::move(list));
    return handle;
}

bool Recorder::IsRecording(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb && cb->recording != 0;
}

void Recorder::CallCommands(u64 cmd_va, std::span<const u64> handles) {
    RECORDING(cb)
    for (const u64 handle : handles) {
        impl->Emit<CallListPayload>(*cb, Op::CallList)->handle = handle;
    }
}

void Recorder::CallCommands(u64 cmd_va, s32 count, u64 handles_va) {
    if (count < 0 || count > 4096) {
        NVN_BAD_ARGUMENT(impl->stats, "{} called command lists (limit 4096)", count);
        return;
    }
    std::vector<u64> handles(static_cast<size_t>(count));
    if (count > 0 && !impl->GuestRead(handles_va, handles.data(), handles.size() * sizeof(u64))) {
        return;
    }
    CallCommands(cmd_va, handles);
}

void Recorder::EmitRaw(u64 cmd_va, u16 op, u16 flags, const void* payload, u32 payload_bytes) {
    RECORDING(cb)
    u64* header = impl->AllocRecord(*cb, static_cast<Op>(op), flags, payload_bytes, COMMAND_RESERVE);
    if (payload_bytes) {
        std::memcpy(header + 1, payload, payload_bytes);
    }
}

// ---- Transfer and synchronization -----------------------------------------------------------------

void Recorder::ClearTexture(u64 cmd_va, u64 texture_va, u64 view_va, u64 region_va, u64 value_va,
                            u32 mask, bool is_integer) {
    RECORDING(cb)
    RenderTargetRef target;
    CopyRegion region{};
    u32 color[4]{};
    if (!impl->ResolveTarget(*cb, texture_va, view_va, target, __func__) || !target.texture.serial ||
        !impl->GuestRead(region_va, &region, sizeof(region)) ||
        !impl->GuestRead(value_va, color, sizeof(color))) {
        DROP();
    }
    auto* payload = impl->Emit<ClearTexturePayload>(*cb, Op::ClearTexture, static_cast<u16>(mask));
    payload->texture = target.texture;
    payload->region = region;
    payload->layer_count = target.view.set_layers ? target.view.nlayers : 0;
    payload->is_integer = is_integer ? 1 : 0;
    std::memcpy(payload->color, color, sizeof(color));
}

void Recorder::ClearBuffer(u64 cmd_va, u64 address, u64 size, u32 value) {
    RECORDING(cb)
    auto* payload = impl->Emit<ClearBufferPayload>(*cb, Op::ClearBuffer);
    payload->address = address;
    payload->size = size;
    payload->value = value;
}

void Recorder::CopyBufferToBuffer(u64 cmd_va, u64 src, u64 dst, u64 size, u32 flags) {
    RECORDING(cb)
    auto* payload = impl->Emit<CopyBufferToBufferPayload>(*cb, Op::CopyBufferToBuffer);
    payload->src = src;
    payload->dst = dst;
    payload->size = size;
    payload->flags = flags;
}

void Recorder::CopyBufferToTexture(u64 cmd_va, u64 src_address, u64 texture_va, u64 view_va,
                                   u64 region_va, u32 flags) {
    RECORDING(cb)
    RenderTargetRef target;
    CopyRegion region{};
    if (!impl->ResolveTarget(*cb, texture_va, view_va, target, __func__) || !target.texture.serial ||
        !impl->GuestRead(region_va, &region, sizeof(region))) {
        DROP();
    }
    auto* payload = impl->Emit<CopyBufferTexturePayload>(*cb, Op::CopyBufferToTexture);
    payload->buffer = src_address;
    payload->texture = target.texture;
    payload->region = region;
    payload->row_stride = static_cast<u32>(cb->copy_row_stride);
    payload->image_stride = static_cast<u32>(cb->copy_image_stride);
    payload->flags = flags;
}

void Recorder::CopyTextureToBuffer(u64 cmd_va, u64 texture_va, u64 view_va, u64 region_va,
                                   u64 dst_address, u32 flags) {
    RECORDING(cb)
    RenderTargetRef target;
    CopyRegion region{};
    if (!impl->ResolveTarget(*cb, texture_va, view_va, target, __func__) || !target.texture.serial ||
        !impl->GuestRead(region_va, &region, sizeof(region))) {
        DROP();
    }
    auto* payload = impl->Emit<CopyBufferTexturePayload>(*cb, Op::CopyTextureToBuffer);
    payload->buffer = dst_address;
    payload->texture = target.texture;
    payload->region = region;
    payload->row_stride = static_cast<u32>(cb->copy_row_stride);
    payload->image_stride = static_cast<u32>(cb->copy_image_stride);
    payload->flags = flags;
}

void Recorder::CopyTextureToTexture(u64 cmd_va, u64 src_va, u64 src_view_va, u64 src_region_va,
                                    u64 dst_va, u64 dst_view_va, u64 dst_region_va, u32 flags) {
    RECORDING(cb)
    RenderTargetRef src;
    RenderTargetRef dst;
    CopyRegion src_region{};
    CopyRegion dst_region{};
    if (!impl->ResolveTarget(*cb, src_va, src_view_va, src, __func__) || !src.texture.serial ||
        !impl->ResolveTarget(*cb, dst_va, dst_view_va, dst, __func__) || !dst.texture.serial ||
        !impl->GuestRead(src_region_va, &src_region, sizeof(src_region)) ||
        !impl->GuestRead(dst_region_va, &dst_region, sizeof(dst_region))) {
        DROP();
    }
    auto* payload = impl->Emit<CopyTextureToTexturePayload>(*cb, Op::CopyTextureToTexture);
    payload->src = src.texture;
    payload->src_region = src_region;
    payload->dst = dst.texture;
    payload->dst_region = dst_region;
    payload->flags = flags;
}

void Recorder::Downsample(u64 cmd_va, u64 src_va, u64 dst_va) {
    RECORDING(cb)
    RenderTargetRef src;
    RenderTargetRef dst;
    TextureObject* src_texture = nullptr;
    if (!impl->ResolveTarget(*cb, src_va, 0, src, __func__, &src_texture) || !src_texture ||
        !impl->ResolveTarget(*cb, dst_va, 0, dst, __func__) || !dst.texture.serial) {
        DROP();
    }
    CopyRegion region{};
    region.width = static_cast<s32>(src_texture->desc.width);
    region.height = static_cast<s32>(src_texture->desc.height);
    region.depth = static_cast<s32>(std::max<u32>(1, src_texture->desc.depth));
    auto* payload = impl->Emit<CopyTextureToTexturePayload>(*cb, Op::CopyTextureToTexture);
    payload->src = src.texture;
    payload->src_region = region;
    payload->dst = dst.texture;
    payload->dst_region = region;
    payload->flags = COPY_FLAG_RESOLVE;
}

void Recorder::SetCopyRowStride(u64 cmd_va, s64 stride) {
    if (CommandBufferObject* cb =
            impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__)) {
        cb->copy_row_stride = stride;
    }
}

void Recorder::SetCopyImageStride(u64 cmd_va, s64 stride) {
    if (CommandBufferObject* cb =
            impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__)) {
        cb->copy_image_stride = stride;
    }
}

s64 Recorder::GetCopyRowStride(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb ? cb->copy_row_stride : 0;
}

s64 Recorder::GetCopyImageStride(u64 cmd_va) {
    const CommandBufferObject* cb =
        impl->Lookup(impl->command_buffers, ObjectKind::CommandBuffer, cmd_va, __func__);
    return cb ? cb->copy_image_stride : 0;
}

void Recorder::UpdateUniformBuffer(u64 cmd_va, u64 buffer_address, u64 offset, u64 size,
                                   u64 data_va) {
    RECORDING(cb)
    if (size > MAX_UNIFORM_UPDATE) {
        BAD_ARGUMENT();
    }
    std::vector<u8>& data = Scratch();
    data.resize(size);
    if (size && !impl->GuestRead(data_va, data.data(), size)) {
        DROP();
    }
    auto* payload = impl->Emit<UpdateUniformBufferPayload>(*cb, Op::UpdateUniformBuffer, 0,
                                                           static_cast<u32>(size));
    payload->address = buffer_address + offset;
    payload->size = size;
    if (size) {
        std::memcpy(payload + 1, data.data(), size);
    }
}

void Recorder::FenceSync(u64 cmd_va, u64 sync_va, u32 condition, u32 flags) {
    RECORDING(cb)
    const SyncObject* sync = impl->Lookup(impl->syncs, ObjectKind::Sync, sync_va, __func__);
    if (!sync) {
        DROP();
    }
    u64* header = impl->AllocRecord(*cb, Op::FenceSync, 0, sizeof(FenceSyncPayload), COMMAND_RESERVE);
    *reinterpret_cast<FenceSyncPayload*>(header + 1) =
        FenceSyncPayload{sync->serial, condition, flags};
    Action action;
    action.word = static_cast<u32>(header - cb->words.data());
    action.kind = Action::Kind::FenceSync;
    action.a = sync->serial;
    cb->actions.push_back(action);
    impl->AddRef(*cb, sync->serial);
}

void Recorder::WaitSync(u64 cmd_va, u64 sync_va) {
    RECORDING(cb)
    const SyncObject* sync = impl->Lookup(impl->syncs, ObjectKind::Sync, sync_va, __func__);
    if (!sync) {
        DROP();
    }
    if (sync->fence_seq.load() == SYNC_SIGNALED_WITHOUT_FENCE) {
        // Signalled without a fence (the acquire sync): the wait is already satisfied, and the
        // replay knows no fence for it to wait on.
        return;
    }
    impl->Emit<WaitSyncPayload>(*cb, Op::WaitSync)->sync_serial = sync->serial;
    impl->AddRef(*cb, sync->serial);
}

void Recorder::SignalEvent(u64 cmd_va, u64 event_va, u32 mode, u32 location, u32 flags,
                           u32 value) {
    static_cast<void>(location);
    static_cast<void>(flags);
    RECORDING(cb)
    const EventObject* event = impl->Lookup(impl->events, ObjectKind::Event, event_va, __func__);
    if (!event) {
        DROP();
    }
    u64* header =
        impl->AllocRecord(*cb, Op::SignalEvent, 0, sizeof(SignalEventPayload), COMMAND_RESERVE);
    *reinterpret_cast<SignalEventPayload*>(header + 1) =
        SignalEventPayload{event->serial, event->address, mode, value};
    Action action;
    action.word = static_cast<u32>(header - cb->words.data());
    action.kind = Action::Kind::SignalEvent;
    action.a = event->address;
    action.c = mode;
    action.d = value;
    cb->actions.push_back(action);
    impl->AddRef(*cb, event->serial);
}

/// A record: the replay stalls the queue's stream at it until the event's word satisfies the wait
/// (a signal another queue submits later can be the one), so the order of waits and signals within
/// a queue is the record order. The event is named by its memory address, like the replay's model.
void Recorder::WaitEvent(u64 cmd_va, u64 event_va, u32 mode, u32 value) {
    RECORDING(cb)
    const EventObject* event = impl->Lookup(impl->events, ObjectKind::Event, event_va, __func__);
    if (!event) {
        DROP();
    }
    u64* header =
        impl->AllocRecord(*cb, Op::WaitEvent, 0, sizeof(WaitEventPayload), COMMAND_RESERVE);
    *reinterpret_cast<WaitEventPayload*>(header + 1) =
        WaitEventPayload{event->address, mode, value};
}

/// Like ReportCounter, the sink writes the value on the GPU timeline (a second write at completion
/// could put an older value back over a newer one).
void Recorder::ReportValue(u64 cmd_va, u32 value, u64 address) {
    RECORDING(cb)
    if (TimestampTraceEnabled()) {
        static Common::BootCounter traced;
        if (const u64 n = traced.Next(); n < 1000 || n % 64 == 0) {
            LOG_INFO(Core, "[nvn] tstrace ReportValue present={} value={:#x} address={:#x}",
                     Core::g_nx2_present_count.load(std::memory_order_relaxed), value, address);
        }
    }
    u64* header =
        impl->AllocRecord(*cb, Op::ReportValue, 0, sizeof(ReportValuePayload), COMMAND_RESERVE);
    *reinterpret_cast<ReportValuePayload*>(header + 1) = ReportValuePayload{address, value, 0};
}

/// The sink writes the report ({counter, timestamp}, 16 bytes) on the GPU timeline, so a render
/// condition or shader that reads it sees it ordered after the work before it; the recorder has
/// nothing to apply at completion.
void Recorder::ReportCounter(u64 cmd_va, u32 counter_type, u64 address) {
    RECORDING(cb)
    if (TimestampTraceEnabled()) {
        static Common::BootCounter traced;
        if (const u64 n = traced.Next(); n < 3000 || n % 64 == 0) {
            LOG_INFO(Core, "[nvn] tstrace ReportCounter present={} type={:#x} address={:#x} cmd={:#x}",
                     Core::g_nx2_present_count.load(std::memory_order_relaxed), counter_type, address,
                     cmd_va);
        }
    }
    u64* header =
        impl->AllocRecord(*cb, Op::ReportCounter, 0, sizeof(ReportCounterPayload), COMMAND_RESERVE);
    *reinterpret_cast<ReportCounterPayload*>(header + 1) =
        ReportCounterPayload{address, counter_type, 0};
}

void Recorder::ResetCounter(u64 cmd_va, u32 counter_type) {
    RECORDING(cb)
    if (TimestampTraceEnabled()) {
        static Common::BootCounter traced;
        if (const u64 n = traced.Next(); n < 3000 || n % 64 == 0) {
            LOG_INFO(Core, "[nvn] tstrace ResetCounter present={} type={:#x} cmd={:#x}",
                     Core::g_nx2_present_count.load(std::memory_order_relaxed), counter_type, cmd_va);
        }
    }
    impl->Emit<ResetCounterPayload>(*cb, Op::ResetCounter)->counter_type = counter_type;
}

void Recorder::Barrier(u64 cmd_va, u32 bits) {
    RECORDING(cb)
    impl->Emit<BarrierPayload>(*cb, Op::Barrier)->bits = bits;
}

// ---- Render state --------------------------------------------------------------------------------------

void Recorder::SetRenderTargets(u64 cmd_va, s32 count, u64 textures_va, u64 views_va, u64 depth_va,
                                u64 depth_view_va) {
    RECORDING(cb)
    if (count < 0 || count > 8) {
        BAD_ARGUMENT();
    }
    std::array<u64, 8> color_vas{};
    std::array<u64, 8> view_vas{};
    if (count > 0 && textures_va &&
        !impl->GuestRead(textures_va, color_vas.data(), sizeof(u64) * count)) {
        DROP();
    }
    if (count > 0 && views_va &&
        !impl->GuestRead(views_va, view_vas.data(), sizeof(u64) * count)) {
        DROP();
    }
    std::array<RenderTargetRef, 9> targets{};
    for (s32 i = 0; i < count; ++i) {
        // A target that does not resolve (stale or untagged) becomes an empty slot; the lookup
        // already counted it.
        impl->ResolveTarget(*cb, color_vas[i], view_vas[i], targets[i], __func__);
    }
    const u32 has_depth = depth_va != 0 ? 1 : 0;
    if (has_depth) {
        impl->ResolveTarget(*cb, depth_va, depth_view_va, targets[count], __func__);
    }
    const u32 total = static_cast<u32>(count) + has_depth;
    auto* payload = impl->Emit<SetRenderTargetsPayload>(*cb, Op::SetRenderTargets, 0,
                                                        total * sizeof(RenderTargetRef));
    payload->color_count = static_cast<u32>(count);
    payload->has_depth = has_depth;
    std::memcpy(const_cast<RenderTargetRef*>(TrailingArray<RenderTargetRef>(payload)),
                targets.data(), total * sizeof(RenderTargetRef));
}

void Recorder::ClearColor(u64 cmd_va, s32 index, u64 values_va, u32 mask, bool is_integer) {
    RECORDING(cb)
    u32 color[4]{};
    if (!impl->GuestRead(values_va, color, sizeof(color))) {
        DROP();
    }
    auto* payload = impl->Emit<ClearColorPayload>(*cb, Op::ClearColor);
    payload->index = index;
    payload->mask = mask;
    payload->is_integer = is_integer ? 1 : 0;
    std::memcpy(payload->color, color, sizeof(color));
}

void Recorder::ClearDepthStencil(u64 cmd_va, f32 depth, u32 depth_mask, s32 stencil,
                                 u32 stencil_mask) {
    RECORDING(cb)
    auto* payload = impl->Emit<ClearDepthStencilPayload>(*cb, Op::ClearDepthStencil);
    payload->depth = depth;
    payload->depth_mask = depth_mask ? 1 : 0;
    payload->stencil = stencil;
    payload->stencil_mask = stencil_mask;
}

namespace {

/// Emits a {first, count} prefix record followed by `element_size * count` bytes read from
/// `values_va`. Nothing is recorded when the arguments are out of range or the read fails.
template <typename Prefix>
void EmitArray(Recorder::Impl& impl, CommandBufferObject& cb, Op op, s32 first, s32 count,
               u64 values_va, size_t element_size, s32 limit) {
    if (first < 0 || count < 0 || first > limit || count > limit - first) {
        NVN_BAD_ARGUMENT(impl.stats, "range first {} count {} (limit {})", first, count, limit);
        ++impl.stats.records_dropped;
        return;
    }
    std::vector<u8>& data = Scratch();
    data.resize(element_size * count);
    if (!data.empty() && !impl.GuestRead(values_va, data.data(), data.size())) {
        ++impl.stats.records_dropped;
        return;
    }
    auto* payload = impl.Emit<Prefix>(cb, op, 0, static_cast<u32>(data.size()));
    payload->first = static_cast<u32>(first);
    payload->count = static_cast<u32>(count);
    if (!data.empty()) {
        std::memcpy(const_cast<u8*>(reinterpret_cast<const u8*>(payload + 1)), data.data(),
                    data.size());
    }
}

} // namespace

void Recorder::SetViewports(u64 cmd_va, s32 first, s32 count, u64 values_va) {
    RECORDING(cb)
    EmitArray<SetViewportsPayload>(*impl, *cb, Op::SetViewports, first, count, values_va,
                                   sizeof(ViewportRect), 16);
}

void Recorder::SetScissors(u64 cmd_va, s32 first, s32 count, u64 values_va) {
    RECORDING(cb)
    EmitArray<SetScissorsPayload>(*impl, *cb, Op::SetScissors, first, count, values_va,
                                  sizeof(ScissorRect), 16);
}

void Recorder::SetDepthRanges(u64 cmd_va, s32 first, s32 count, u64 values_va) {
    RECORDING(cb)
    EmitArray<SetDepthRangesPayload>(*impl, *cb, Op::SetDepthRanges, first, count, values_va,
                                     sizeof(DepthRange), 16);
}

void Recorder::SetViewportSwizzles(u64 cmd_va, s32 first, s32 count, u64 values_va) {
    RECORDING(cb)
    EmitArray<SetViewportSwizzlesPayload>(*impl, *cb, Op::SetViewportSwizzles, first, count,
                                          values_va, 4, 16);
}

/// nvnCommandBufferBindProgram (3.2.0 sdk+0x5fe0e0, 1.0.0 sdk+0x5daa80):
/// - `stages` 0: nothing happens (even for a NULL program).
/// - NULL program: the stages of `stages` are unbound (recorded with program serial 0).
/// - A program whose compute stage is in `stages`: compute is bound; the mask's other stages
///   that the program lacks are unbound (the replay finds no code for them on this program).
/// - Otherwise (graphics): the driver copies the program's whole precompiled pushbuffer, so every
///   graphics stage the program has is bound even when `stages` leaves it out; the mask's stages
///   the program lacks are unbound as above.
void Recorder::BindProgram(u64 cmd_va, u64 program_va, u32 stages) {
    RECORDING(cb)
    if (stages == 0) {
        return;
    }
    if (program_va == 0) {
        auto* payload = impl->Emit<BindProgramPayload>(*cb, Op::BindProgram);
        payload->program_serial = 0;
        payload->stages = stages;
        return;
    }
    ProgramObject* program =
        impl->Lookup(impl->programs, ObjectKind::Program, program_va, __func__);
    if (!program) {
        DROP();
    }
    constexpr u32 compute_bit = 1U << 5;
    const u32 program_stages = program->desc.stage_mask; // set once by ProgramSetShaders
    if ((stages & program_stages & compute_bit) == 0) {
        const u32 widened = stages | (program_stages & ~compute_bit);
        if (widened != stages) {
            static Common::BootCounter logged;
            if (logged.First()) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] BindProgram mask {:#x} narrower than the program's stages {:#x}: "
                         "binding {:#x} as the driver does",
                         stages, program_stages, widened);
            }
        }
        stages = widened;
    }
    {
        std::lock_guard lock{impl->sink_mutex};
        impl->EnsureProgramLocked(*program, false);
    }
    auto* payload = impl->Emit<BindProgramPayload>(*cb, Op::BindProgram);
    payload->program_serial = program->serial;
    payload->stages = stages;
    impl->AddRef(*cb, program->serial);
}

void Recorder::BindState(u64 cmd_va, StateKind kind, u64 state_va) {
    RECORDING(cb)
    const u32 index = static_cast<u32>(kind);
    if (index >= 6) {
        BAD_ARGUMENT();
    }
    const u32 size = STATE_OBJECT_SIZES[index];
    u8 bytes[16]{};
    if (!impl->GuestRead(state_va, bytes, size)) {
        DROP();
    }
    auto* payload = impl->Emit<BindStatePayload>(*cb, Op::BindState, 0, size);
    payload->kind = index;
    payload->size = size;
    std::memcpy(const_cast<u8*>(reinterpret_cast<const u8*>(payload + 1)), bytes, size);
}

void Recorder::BindVertexAttribState(u64 cmd_va, s32 count, u64 states_va) {
    RECORDING(cb)
    EmitArray<BindVertexAttribsPayload>(*impl, *cb, Op::BindVertexAttribs, 0, count, states_va, 4,
                                        16);
}

/// The driver (3.2.0 sdk+0x619d80, 1.0.0 sdk+0x5f84c0) programs each stream's stride as
/// `stride & 0xfff`: the record carries the stride the GPU sees.
void Recorder::BindVertexStreamState(u64 cmd_va, s32 count, u64 states_va) {
    RECORDING(cb)
    constexpr s32 max_streams = 16;
    if (count < 0 || count > max_streams) {
        NVN_BAD_ARGUMENT(impl->stats, "range first 0 count {} (limit {})", count, max_streams);
        ++impl->stats.records_dropped;
        return;
    }
    std::array<VertexStreamState, max_streams> states{};
    const u32 bytes = static_cast<u32>(sizeof(VertexStreamState)) * static_cast<u32>(count);
    if (bytes != 0 && !impl->GuestRead(states_va, states.data(), bytes)) {
        ++impl->stats.records_dropped;
        return;
    }
    for (s32 i = 0; i < count; ++i) {
        states[i].stride &= 0xFFF;
    }
    auto* payload = impl->Emit<BindVertexStreamsPayload>(*cb, Op::BindVertexStreams, 0, bytes);
    payload->first = 0;
    payload->count = static_cast<u32>(count);
    if (bytes != 0) {
        std::memcpy(const_cast<VertexStreamState*>(TrailingArray<VertexStreamState>(payload)),
                    states.data(), bytes);
    }
}

void Recorder::BindVertexBuffers(u64 cmd_va, s32 first, s32 count, u64 ranges_va) {
    RECORDING(cb)
    EmitArray<BindVertexBuffersPayload>(*impl, *cb, Op::BindVertexBuffers, first, count,
                                        ranges_va, sizeof(BufferRange), 16);
}

void Recorder::BindVertexBuffer(u64 cmd_va, s32 index, u64 address, u64 size) {
    RECORDING(cb)
    if (index < 0 || index >= 16) {
        BAD_ARGUMENT();
    }
    auto* payload = impl->Emit<BindVertexBuffersPayload>(*cb, Op::BindVertexBuffers, 0,
                                                         sizeof(BufferRange));
    payload->first = static_cast<u32>(index);
    payload->count = 1;
    *const_cast<BufferRange*>(TrailingArray<BufferRange>(payload)) = BufferRange{address, size};
}

namespace {

void BindBuffers(Recorder::Impl& impl, CommandBufferObject& cb, Op op, u32 stage, s32 first,
                 s32 count, u64 ranges_va, bool deferred) {
    if (stage >= 8 || first < 0 || count < 0 || first > 32 || count > 32 - first) {
        NVN_BAD_ARGUMENT(impl.stats, "stage {} first {} count {} (limit 32)", stage, first, count);
        ++impl.stats.records_dropped;
        return;
    }
    std::array<BufferRange, 32> ranges{};
    if (count > 0 && !impl.memory.Read(ranges_va, ranges.data(), sizeof(BufferRange) * count) &&
        !deferred) {
        ++impl.stats.bad_guest_read;
        ++impl.stats.records_dropped;
        return;
    }
    if (op == Op::BindUniformBuffers) {
        // BindUniformBuffers (3.2.0 sdk+0x5feda0, 1.0.0 sdk+0x5db930) programs each size in
        // 256-byte units, rounded up (0 stays 0), like nvnCommandBufferBindUniformBuffer.
        for (s32 i = 0; i < count; ++i) {
            ranges[i].size = (ranges[i].size + 0xFF) & ~u64{0xFF};
        }
    }
    auto* payload = impl.Emit<BindBuffersPayload>(cb, op, deferred ? RECORD_FLAG_DEFERRED : 0,
                                                  static_cast<u32>(sizeof(BufferRange)) * count);
    payload->stage = stage;
    payload->first = static_cast<u32>(first);
    payload->count = static_cast<u32>(count);
    payload->src = deferred ? ranges_va : 0;
    std::memcpy(const_cast<BufferRange*>(TrailingArray<BufferRange>(payload)), ranges.data(),
                sizeof(BufferRange) * count);
}

} // namespace

void Recorder::BindUniformBuffers(u64 cmd_va, u32 stage, s32 first, s32 count, u64 ranges_va,
                                  bool deferred) {
    RECORDING(cb)
    BindBuffers(*impl, *cb, Op::BindUniformBuffers, stage, first, count, ranges_va, deferred);
}

void Recorder::BindStorageBuffers(u64 cmd_va, u32 stage, s32 first, s32 count, u64 ranges_va,
                                  bool deferred) {
    RECORDING(cb)
    BindBuffers(*impl, *cb, Op::BindStorageBuffers, stage, first, count, ranges_va, deferred);
}

void Recorder::BindUniformBuffer(u64 cmd_va, u32 stage, s32 index, u64 address, u64 size) {
    RECORDING(cb)
    if (stage >= 8 || index < 0) {
        BAD_ARGUMENT_DETAIL("[nvn] BindUniformBuffer: stage {} index {}", stage, index);
        BAD_ARGUMENT();
    }
    auto* payload =
        impl->Emit<BindBuffersPayload>(*cb, Op::BindUniformBuffers, 0, sizeof(BufferRange));
    payload->stage = stage;
    payload->first = static_cast<u32>(index);
    payload->count = 1;
    *const_cast<BufferRange*>(TrailingArray<BufferRange>(payload)) = BufferRange{address, size};
}

void Recorder::BindStorageBuffer(u64 cmd_va, u32 stage, s32 index, u64 address, u64 size) {
    RECORDING(cb)
    if (stage >= 8 || index < 0) {
        BAD_ARGUMENT_DETAIL("[nvn] BindStorageBuffer: stage {} index {}", stage, index);
        BAD_ARGUMENT();
    }
    auto* payload =
        impl->Emit<BindBuffersPayload>(*cb, Op::BindStorageBuffers, 0, sizeof(BufferRange));
    payload->stage = stage;
    payload->first = static_cast<u32>(index);
    payload->count = 1;
    *const_cast<BufferRange*>(TrailingArray<BufferRange>(payload)) = BufferRange{address, size};
}

void Recorder::BindHandles(u64 cmd_va, Op kind, u32 stage, s32 first, s32 count, u64 handles_va,
                           bool deferred) {
    RECORDING(cb)
    if ((kind != Op::BindTextures && kind != Op::BindSeparateTextures &&
         kind != Op::BindSeparateSamplers && kind != Op::BindImages) ||
        stage >= 8 || first < 0 || count < 0 || count > 32) {
        BAD_ARGUMENT();
    }
    std::array<u64, 32> handles{};
    if (count > 0 && !impl->memory.Read(handles_va, handles.data(), sizeof(u64) * count)) {
        if (!deferred) {
            ++impl->stats.bad_guest_read;
            DROP();
        }
        handles.fill(0);
    }
    auto* payload = impl->Emit<BindHandlesPayload>(*cb, kind, deferred ? RECORD_FLAG_DEFERRED : 0,
                                                   static_cast<u32>(sizeof(u64)) * count);
    payload->stage = stage;
    payload->first = static_cast<u32>(first);
    payload->count = static_cast<u32>(count);
    payload->src = deferred ? handles_va : 0;
    payload->tex_base = cb->tex_base;
    payload->samp_base = cb->samp_base;
    u64* out = const_cast<u64*>(TrailingArray<u64>(payload));
    for (s32 i = 0; i < count; ++i) {
        out[i] = RebaseHandle(kind, handles[i], cb->tex_base, cb->samp_base);
    }
}

void Recorder::BindHandle(u64 cmd_va, Op kind, u32 stage, s32 index, u64 handle) {
    RECORDING(cb)
    if ((kind != Op::BindTextures && kind != Op::BindSeparateTextures &&
         kind != Op::BindSeparateSamplers && kind != Op::BindImages) ||
        stage >= 8 || index < 0) {
        BAD_ARGUMENT_DETAIL("[nvn] BindHandle: op {} stage {} index {}", static_cast<u32>(kind), stage, index);
        BAD_ARGUMENT();
    }
    auto* payload = impl->Emit<BindHandlesPayload>(*cb, kind, 0, sizeof(u64));
    payload->stage = stage;
    payload->first = static_cast<u32>(index);
    payload->count = 1;
    payload->tex_base = cb->tex_base;
    payload->samp_base = cb->samp_base;
    *const_cast<u64*>(TrailingArray<u64>(payload)) =
        RebaseHandle(kind, handle, cb->tex_base, cb->samp_base);
}

void Recorder::SetTexturePool(u64 cmd_va, u64 texture_pool_va) {
    RECORDING(cb)
    const DescriptorPoolObject* pool =
        impl->Lookup(impl->texture_pools, ObjectKind::TexturePool, texture_pool_va, __func__);
    if (!pool) {
        DROP();
    }
    auto* payload = impl->Emit<SetPoolPayload>(*cb, Op::SetTexturePool);
    payload->pool_serial = pool->serial;
    payload->base = pool->base;
    cb->tex_base = pool->base;
}

void Recorder::SetSamplerPool(u64 cmd_va, u64 sampler_pool_va) {
    RECORDING(cb)
    const DescriptorPoolObject* pool =
        impl->Lookup(impl->sampler_pools, ObjectKind::SamplerPool, sampler_pool_va, __func__);
    if (!pool) {
        DROP();
    }
    auto* payload = impl->Emit<SetPoolPayload>(*cb, Op::SetSamplerPool);
    payload->pool_serial = pool->serial;
    payload->base = pool->base;
    cb->samp_base = pool->base;
}

void Recorder::SetStencil(u64 cmd_va, s32 face, u32 which, s32 value) {
    RECORDING(cb)
    auto* payload = impl->Emit<SetStencilPayload>(*cb, Op::SetStencil);
    payload->face = face;
    payload->which = which;
    payload->value = value;
}

void Recorder::SetPolygonOffsetClamp(u64 cmd_va, f32 factor, f32 units, f32 clamp) {
    RECORDING(cb)
    auto* payload = impl->Emit<SetPolygonOffsetPayload>(*cb, Op::SetPolygonOffset);
    payload->factor = factor;
    payload->units = units;
    payload->clamp = clamp;
}

void Recorder::SetBlendColor(u64 cmd_va, u64 values_va) {
    RECORDING(cb)
    f32 color[4]{};
    if (!impl->GuestRead(values_va, color, sizeof(color))) {
        DROP();
    }
    std::memcpy(impl->Emit<SetBlendColorPayload>(*cb, Op::SetBlendColor)->color, color,
                sizeof(color));
}

void Recorder::SetDepthClamp(u64 cmd_va, bool enable) {
    RECORDING(cb)
    impl->Emit<EnablePayload>(*cb, Op::SetDepthClamp)->enable = enable ? 1 : 0;
}

void Recorder::SetAlphaRef(u64 cmd_va, f32 reference) {
    RECORDING(cb)
    impl->Emit<SetAlphaRefPayload>(*cb, Op::SetAlphaRef)->reference = reference;
}

void Recorder::SetPrimitiveRestart(u64 cmd_va, bool enable, u32 index) {
    RECORDING(cb)
    auto* payload = impl->Emit<SetPrimitiveRestartPayload>(*cb, Op::SetPrimitiveRestart);
    payload->enable = enable ? 1 : 0;
    payload->index = index;
}

void Recorder::SetRasterizerDiscard(u64 cmd_va, bool enable) {
    RECORDING(cb)
    impl->Emit<EnablePayload>(*cb, Op::SetRasterizerDiscard)->enable = enable ? 1 : 0;
}

void Recorder::SetSampleMask(u64 cmd_va, u32 mask) {
    RECORDING(cb)
    impl->Emit<U32Payload>(*cb, Op::SetSampleMask)->value = mask;
}

void Recorder::SetProvokingVertexMode(u64 cmd_va, u32 mode) {
    RECORDING(cb)
    impl->Emit<U32Payload>(*cb, Op::SetProvokingVertex)->value = mode & 1;
}

void Recorder::SetPointSize(u64 cmd_va, f32 size) {
    RECORDING(cb)
    impl->Emit<F32Payload>(*cb, Op::SetPointSize)->value = size;
}

void Recorder::SetLineWidth(u64 cmd_va, f32 width) {
    RECORDING(cb)
    impl->Emit<F32Payload>(*cb, Op::SetLineWidth)->value = width;
}

void Recorder::SetRenderEnable(u64 cmd_va, bool enable) {
    RECORDING(cb)
    impl->Emit<SetRenderConditionPayload>(*cb, Op::SetRenderCondition)->kind = enable ? 1 : 2;
}

void Recorder::SetRenderEnableConditional(u64 cmd_va, u32 mode, u64 address) {
    RECORDING(cb)
    auto* payload = impl->Emit<SetRenderConditionPayload>(*cb, Op::SetRenderCondition);
    payload->kind = 0;
    payload->mode = mode;
    payload->address = address;
}

void Recorder::PushConstants(u64 cmd_va, bool compute, u64 word, u64 offset, u64 size,
                             u64 data_va) {
    RECORDING(cb)
    if (size == 0 || size > 4096 || offset > 4096 || offset + size > 4096) {
        BAD_ARGUMENT();
    }
    std::vector<u8>& data = Scratch();
    data.resize(size);
    if (!impl->GuestRead(data_va, data.data(), size)) {
        DROP();
    }
    auto* payload =
        impl->Emit<PushConstantsPayload>(*cb, Op::PushConstants, 0, static_cast<u32>(size));
    payload->compute = compute ? 1 : 0;
    payload->offset = static_cast<u32>(offset);
    payload->size = static_cast<u32>(size);
    payload->word = word;
    std::memcpy(payload + 1, data.data(), size);
}

void Recorder::BindTransformFeedbackBuffer(u64 cmd_va, s32 index, u64 address, u64 size) {
    RECORDING(cb)
    auto* payload = impl->Emit<BindTransformFeedbackBufferPayload>(
        *cb, Op::BindTransformFeedbackBuffer);
    payload->index = static_cast<u32>(index);
    payload->address = address;
    payload->size = size;
}

void Recorder::TransformFeedbackState(u64 cmd_va, u32 what, u64 address) {
    RECORDING(cb)
    auto* payload = impl->Emit<TransformFeedbackStatePayload>(*cb, Op::TransformFeedbackState);
    payload->what = what;
    payload->address = address;
}

void Recorder::DrawTransformFeedback(u64 cmd_va, u32 mode, u64 address) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawTransformFeedbackPayload>(*cb, Op::DrawTransformFeedback);
    payload->mode = mode;
    payload->address = address;
}

void Recorder::EvaluateDlss(u64 cmd_va, const DlssEvaluateArgs& args) {
    RECORDING(cb)
    RenderTargetRef input;
    RenderTargetRef output;
    if (!impl->ResolveTarget(*cb, args.input_texture, args.input_view, input, __func__) ||
        !input.texture.serial ||
        !impl->ResolveTarget(*cb, args.output_texture, args.output_view, output, __func__) ||
        !output.texture.serial) {
        DROP();
    }
    // The other images only help DLSS: one the game named that is not a texture (the lookup logged
    // it and cleared the reference) is an image the evaluation runs without.
    RenderTargetRef motion_vectors;
    RenderTargetRef depth;
    RenderTargetRef bias;
    RenderTargetRef exposure;
    static_cast<void>(impl->ResolveTarget(*cb, args.motion_vectors_texture,
                                          args.motion_vectors_view, motion_vectors, __func__));
    static_cast<void>(
        impl->ResolveTarget(*cb, args.depth_texture, args.depth_view, depth, __func__));
    static_cast<void>(impl->ResolveTarget(*cb, args.bias_texture, args.bias_view, bias, __func__));
    static_cast<void>(
        impl->ResolveTarget(*cb, args.exposure_texture, args.exposure_view, exposure, __func__));
    auto* payload = impl->Emit<EvaluateDlssPayload>(*cb, Op::EvaluateDlss);
    payload->input = input;
    payload->output = output;
    payload->motion_vectors = motion_vectors;
    payload->depth = depth;
    payload->bias = bias;
    payload->exposure = exposure;
    payload->input_base = args.input_base;
    payload->depth_base = args.depth_base;
    payload->motion_vector_base = args.motion_vector_base;
    payload->bias_base = args.bias_base;
    payload->output_base = args.output_base;
    payload->input_width = args.input_width;
    payload->input_height = args.input_height;
    payload->output_width = args.output_width;
    payload->output_height = args.output_height;
    std::memcpy(payload->motion_scale, args.motion_scale, sizeof(payload->motion_scale));
    std::memcpy(payload->jitter, args.jitter, sizeof(payload->jitter));
    payload->pre_exposure = args.pre_exposure;
    payload->history_reset = args.history_reset ? 1 : 0;
    payload->flags = args.flags;
    payload->method = args.method;
    payload->preset = args.preset;
    payload->network = args.network;
}

// ---- Draws and dispatches -------------------------------------------------------------------------------

void Recorder::DrawArrays(u64 cmd_va, u32 mode, s32 first, u32 count, u32 instance_count,
                          u32 base_instance) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawPayload>(*cb, Op::Draw);
    payload->mode = mode;
    payload->first = first;
    payload->count = count;
    payload->instance_count = instance_count;
    payload->base_instance = base_instance;
    payload->max_draws = 1;
}

void Recorder::DrawArraysIndirect(u64 cmd_va, u32 mode, u64 indirect) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawPayload>(*cb, Op::Draw);
    payload->mode = mode;
    payload->instance_count = 1;
    payload->max_draws = 1;
    payload->indirect = indirect;
}

void Recorder::MultiDrawArraysIndirectCount(u64 cmd_va, u32 mode, u64 indirect,
                                            u64 count_address, u32 max_draws, s64 stride) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawPayload>(*cb, Op::Draw);
    payload->mode = mode;
    payload->instance_count = 1;
    payload->indirect = indirect;
    payload->count_address = count_address;
    payload->max_draws = max_draws;
    payload->stride = static_cast<u32>(stride);
}

void Recorder::DrawElements(u64 cmd_va, u32 mode, u32 index_type, u32 count, u64 index_address,
                            s32 base_vertex, u32 base_instance, u32 instance_count) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawIndexedPayload>(*cb, Op::DrawIndexed);
    payload->mode = mode;
    payload->index_type = index_type;
    payload->count = count;
    payload->base_vertex = base_vertex;
    payload->base_instance = base_instance;
    payload->instance_count = instance_count;
    payload->index_address = index_address;
    payload->max_draws = 1;
}

void Recorder::DrawElementsIndirect(u64 cmd_va, u32 mode, u32 index_type, u64 index_address,
                                    u64 indirect) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawIndexedPayload>(*cb, Op::DrawIndexed);
    payload->mode = mode;
    payload->index_type = index_type;
    payload->instance_count = 1;
    payload->count = impl->BufferBytesFrom(index_address);
    payload->index_address = index_address;
    payload->indirect = indirect;
    payload->max_draws = 1;
}

void Recorder::MultiDrawElementsIndirectCount(u64 cmd_va, u32 mode, u32 index_type,
                                              u64 index_address, u64 indirect,
                                              u64 count_address, u32 max_draws, s64 stride) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawIndexedPayload>(*cb, Op::DrawIndexed);
    payload->mode = mode;
    payload->index_type = index_type;
    payload->instance_count = 1;
    payload->count = impl->BufferBytesFrom(index_address);
    payload->index_address = index_address;
    payload->indirect = indirect;
    payload->count_address = count_address;
    payload->max_draws = max_draws;
    payload->stride = static_cast<u32>(stride);
}

void Recorder::DrawTexture(u64 cmd_va, u64 handle, u64 dst_region_va, u64 src_region_va) {
    RECORDING(cb)
    f32 dst[4]{};
    f32 src[4]{};
    if (!impl->GuestRead(dst_region_va, dst, sizeof(dst)) ||
        !impl->GuestRead(src_region_va, src, sizeof(src))) {
        DROP();
    }
    auto* payload = impl->Emit<DrawTexturePayload>(*cb, Op::DrawTexture);
    payload->handle = RebaseHandle(Op::BindTextures, handle, cb->tex_base, cb->samp_base);
    std::memcpy(payload->dst, dst, sizeof(dst));
    std::memcpy(payload->src, src, sizeof(src));
}

void Recorder::DispatchCompute(u64 cmd_va, u32 x, u32 y, u32 z, u64 caller) {
    RECORDING(cb)
    auto* payload = impl->Emit<DispatchPayload>(*cb, Op::Dispatch);
    payload->x = x;
    payload->y = y;
    payload->z = z;
    payload->caller = caller;
}

void Recorder::DispatchComputeIndirect(u64 cmd_va, u64 indirect, u64 caller) {
    RECORDING(cb)
    auto* payload = impl->Emit<DispatchPayload>(*cb, Op::Dispatch);
    payload->indirect = indirect;
    payload->caller = caller;
}

void Recorder::DrawMeshTasks(u64 cmd_va, u32 dims, u32 first, u32 x, u32 y, u32 z) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawMeshTasksPayload>(*cb, Op::DrawMeshTasks);
    payload->dims = dims;
    payload->first = first;
    payload->x = x;
    payload->y = y;
    payload->z = z;
    payload->max_draws = 1;
}

void Recorder::DrawMeshTasksIndirect(u64 cmd_va, u32 dims, u64 indirect, u64 count_address,
                                     u32 max_draws, u32 stride) {
    RECORDING(cb)
    auto* payload = impl->Emit<DrawMeshTasksPayload>(*cb, Op::DrawMeshTasks);
    payload->dims = dims;
    payload->max_draws = max_draws;
    payload->indirect = indirect;
    payload->count_address = count_address;
    payload->stride = stride;
}

} // namespace VideoCommon::Nvn
