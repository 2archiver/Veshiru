// SPDX-FileCopyrightText: Copyright (c) 2026 jhaizhou (d2r_gfx)
// SPDX-FileCopyrightText: Copyright (c) 2026 MacNativeGames contributors
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <cstring>

#include "common/logging.h"
#include "video_core/nvn/uniform_snapshot.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_uniform_snapshot.h"

namespace Vulkan::Nvn {

// ---- UniformSnapshotView ----------------------------------------------------------------------

void UniformSnapshotView::Reset(VC::UniformSnapshot* snapshot_, u64 seq_) {
    snapshot = snapshot_ != nullptr && !snapshot_->ranges.empty() ? snapshot_ : nullptr;
    seq = snapshot != nullptr ? seq_ : 0;
    const size_t count = snapshot != nullptr ? snapshot->ranges.size() : 0;
    dropped.assign(count, 0);
    uploads.assign(count, Upload{});
}

size_t UniformSnapshotView::First(u64 address) const {
    const auto& ranges = snapshot->ranges;
    const auto it = std::upper_bound(
        ranges.begin(), ranges.end(), address,
        [](u64 value, const VC::UniformSnapshot::Range& range) { return value < range.end; });
    return static_cast<size_t>(it - ranges.begin());
}

size_t UniformSnapshotView::Find(u64 address, u64 size) const {
    if (snapshot == nullptr || size == 0) {
        return npos;
    }
    const size_t index = First(address);
    if (index >= snapshot->ranges.size() || dropped[index] != 0) {
        return npos;
    }
    const VC::UniformSnapshot::Range& range = snapshot->ranges[index];
    return range.begin <= address && address + size <= range.end ? index : npos;
}

u32 UniformSnapshotView::Drop(u64 begin, u64 end) {
    if (snapshot == nullptr || end <= begin) {
        return 0;
    }
    u32 count = 0;
    for (size_t i = First(begin); i < snapshot->ranges.size() && snapshot->ranges[i].begin < end;
         ++i) {
        if (dropped[i] == 0) {
            dropped[i] = 1;
            ++count;
        }
    }
    return count;
}

u32 UniformSnapshotView::Layer(u64 address, const u8* data, u64 size) {
    if (snapshot == nullptr || size == 0) {
        return 0;
    }
    const u64 end = address + size;
    u32 count = 0;
    for (size_t i = First(address); i < snapshot->ranges.size() && snapshot->ranges[i].begin < end;
         ++i) {
        if (dropped[i] != 0) {
            continue;
        }
        const VC::UniformSnapshot::Range& range = snapshot->ranges[i];
        const u64 lo = (std::max)(address, range.begin);
        const u64 hi = (std::min)(end, range.end);
        std::memcpy(snapshot->bytes.data() + range.offset + (lo - range.begin),
                    data + (lo - address), hi - lo);
        uploads[i] = Upload{};
        ++count;
    }
    return count;
}

u32 UniformSnapshotView::DropAll() {
    if (snapshot == nullptr) {
        return 0;
    }
    const u32 live = static_cast<u32>(std::count(dropped.begin(), dropped.end(), u8{0}));
    std::fill(dropped.begin(), dropped.end(), u8{1});
    return live;
}

u32 UniformSnapshotView::DropOverlapping(const std::vector<std::pair<u64, u64>>& sorted) {
    if (snapshot == nullptr) {
        return 0;
    }
    u32 count = 0;
    size_t j = 0;
    for (size_t i = 0; i < snapshot->ranges.size(); ++i) {
        const VC::UniformSnapshot::Range& range = snapshot->ranges[i];
        while (j < sorted.size() && sorted[j].second <= range.begin) {
            ++j;
        }
        if (j < sorted.size() && sorted[j].first < range.end && dropped[i] == 0) {
            dropped[i] = 1;
            ++count;
        }
    }
    return count;
}

// ---- UniformSnapshotTracker -------------------------------------------------------------------

void UniformSnapshotTracker::Arrive(u64 seq, u64 completed_seq) {
    if (enabled_from == 0) {
        enabled_from = seq;
        LOG_INFO(Render_Vulkan, "[nvn] uniform snapshot: the replay binds submit-time uniform "
                                "copies from submission {} on",
                 seq);
    }
    pending.insert(completed_seq);
    latest_completed = (std::max)(latest_completed, completed_seq);
    arrival_clock[seq] = cpu_clock;
}

void UniformSnapshotTracker::Leave(u64 seq, u64 completed_seq) {
    if (const auto it = pending.find(completed_seq); it != pending.end()) {
        pending.erase(it);
    }
    arrival_clock.erase(seq);
}

void UniformSnapshotTracker::NoteCpuEvent(u64 address) {
    if (enabled_from != 0) {
        cpu_events[address] = ++cpu_clock;
    }
}

void UniformSnapshotTracker::OnEventWait(UniformSnapshotView& view, u64 address) {
    if (!view.Active()) {
        return;
    }
    const auto it = cpu_events.find(address);
    if (it != cpu_events.end() && it->second > view.cpu_clock) {
        event_drops += view.DropAll();
    }
}

void UniformSnapshotTracker::Apply(UniformSnapshotView& view, u64 begin, u64 end, u64 seq,
                                   const u8* data) {
    if (seq <= view.CompletedSeq()) {
        return; // in memory before the capture
    }
    if (data != nullptr && seq <= view.Seq()) {
        // An inline update of this submission or of an earlier one: its bytes are known and come
        // before this submission's reads on the GPU.
        layers += view.Layer(begin, data, end - begin);
    } else {
        drops += view.Drop(begin, end);
    }
}

void UniformSnapshotTracker::Activate(UniformSnapshotView& view) {
    Deactivate(view);
    if (!view.Active()) {
        return;
    }
    if (enabled_from == 0 || view.CompletedSeq() + 1 < enabled_from) {
        // Writes of submissions before the first snapshot were not noted.
        view.Reset(nullptr);
        ++refused;
        return;
    }
    const auto clock = arrival_clock.find(view.Seq());
    view.cpu_clock = clock != arrival_clock.end() ? clock->second : cpu_clock;
    if (raw_seq > view.CompletedSeq()) {
        raw_drops += view.DropOverlapping(raw_ranges);
    }
    for (const Write& write : writes) {
        Apply(view, write.begin, write.end, write.seq,
              write.data.empty() ? nullptr : write.data.data());
    }
    active.push_back(&view);
    ++activated;
    LogPerSecond();
}

void UniformSnapshotTracker::Deactivate(UniformSnapshotView& view) {
    std::erase(active, &view);
}

void UniformSnapshotTracker::Append(u64 begin, u64 end, u64 seq, const u8* data) {
    if (data == nullptr && !writes.empty()) {
        Write& last = writes.back();
        if (last.seq == seq && last.data.empty() && begin <= last.end && last.begin <= end) {
            last.begin = (std::min)(last.begin, begin);
            last.end = (std::max)(last.end, end);
            return;
        }
    }
    Write& write = writes.emplace_back(Write{begin, end, seq, {}});
    if (data != nullptr) {
        write.data.assign(data, data + (end - begin));
    }
    if (writes.size() >= prune_at) {
        Prune();
    }
}

void UniformSnapshotTracker::NoteWrite(u64 begin, u64 end, u64 seq) {
    if (enabled_from == 0 || end <= begin) {
        return;
    }
    if (begin == inline_write.begin && end == inline_write.end && seq == inline_write.seq) {
        return; // NoteInlineWrite took it
    }
    Append(begin, end, seq, nullptr);
    for (UniformSnapshotView* view : active) {
        Apply(*view, begin, end, seq, nullptr);
    }
    if (last_history != std::pair{begin, end}) {
        last_history = {begin, end};
        VC::SharedUniformSnapshot().NoteGpuWrite(begin, end);
    }
}

void UniformSnapshotTracker::NoteRawWrite(u64 generation, std::span<const RawGlobalRange> ranges,
                                          u64 seq) {
    if (enabled_from == 0 ||
        VC::SharedUniformSnapshot().Mode() != VC::UniformSnapshotMode::Strict) {
        return;
    }
    if (generation != raw_generation) {
        raw_generation = generation;
        raw_ranges.clear();
        for (const RawGlobalRange& range : ranges) {
            if (range.size == 0) {
                continue;
            }
            const u64 end = range.guest + range.size;
            if (!raw_ranges.empty() && range.guest <= raw_ranges.back().second) {
                raw_ranges.back().second = (std::max)(raw_ranges.back().second, end);
            } else {
                raw_ranges.emplace_back(range.guest, end);
            }
        }
        for (const auto& [begin, end] : raw_ranges) {
            VC::SharedUniformSnapshot().NoteGpuWrite(begin, end);
        }
    } else if (seq == raw_seq) {
        return; // this submission's views dropped them already
    }
    raw_seq = seq;
    for (UniformSnapshotView* view : active) {
        if (seq > view->CompletedSeq()) {
            raw_drops += view->DropOverlapping(raw_ranges);
        }
    }
}

void UniformSnapshotTracker::NoteInlineWrite(u64 address, const u8* data, u64 size, u64 seq) {
    if (enabled_from == 0 || size == 0) {
        return;
    }
    inline_write = {address, address + size, seq};
    Append(address, address + size, seq, data);
    for (UniformSnapshotView* view : active) {
        Apply(*view, address, address + size, seq, data);
    }
}

void UniformSnapshotTracker::Prune() {
    // A write matters to a snapshot only if its submission is later than the one completed at the
    // capture; every snapshot still to come carries at least latest_completed.
    const u64 horizon = pending.empty() ? latest_completed : (std::min)(*pending.begin(),
                                                                         latest_completed);
    std::erase_if(writes, [horizon](const Write& write) { return write.seq <= horizon; });
    prune_at = (std::max)(PRUNE_AT, writes.size() * 2);
}

void UniformSnapshotTracker::LogPerSecond() {
    const auto now = std::chrono::steady_clock::now();
    if (now - log_time < std::chrono::seconds{1}) {
        return;
    }
    if (log_time != std::chrono::steady_clock::time_point{}) {
        LOG_DEBUG(Render_Vulkan,
                  "[nvn] uniform snapshot replay: {} submissions, {} ranges dropped by GPU writes, "
                  "{} by CPU-signalled event waits, {} by raw stores, {} inline updates layered, "
                  "{} snapshots refused, {} writes tracked in the last second",
                  activated, drops, event_drops, raw_drops, layers, refused, writes.size());
    }
    log_time = now;
    activated = 0;
    drops = 0;
    layers = 0;
    refused = 0;
    event_drops = 0;
    raw_drops = 0;
}

// ---- DrawContext --------------------------------------------------------------------------------

std::optional<VkDescriptorBufferInfo> DrawContext::SnapshotUniform(size_t index, u64 address,
                                                                   u64 size) {
    const VC::UniformSnapshot::Range& range = usnap.RangeAt(index);
    UniformSnapshotView::Upload& upload = usnap.UploadAt(index);
    if (!upload.buffer) {
        // The whole range at once: the other bindings into it share the upload.
        const u64 length = range.end - range.begin;
        const u64 phase = range.begin & 0xFF;
        u8* pointer = nullptr;
        const BufferRef copy = AllocArena(length + phase, 256, &pointer);
        if (!copy.buffer) {
            return std::nullopt;
        }
        std::memcpy(pointer + phase, usnap.Bytes(index, range.begin), length);
        upload = {copy.buffer, copy.offset + phase};
        counters.uniform_snapshot_upload_bytes += length;
    }
    const u64 offset = upload.offset + (address - range.begin);
    if (offset % 256 != 0) {
        return std::nullopt;
    }
    ++counters.uniform_snapshot_binds;
    return VkDescriptorBufferInfo{upload.buffer, offset, size};
}

void DrawContext::OnEventWaitPassed(u64 event_address) {
    impl.uniform_snapshots.OnEventWait(usnap, event_address);
    if (host != nullptr) {
        impl.InvalidatePendingInlineBytes(host->HostFlight().seq);
    }
}

bool DrawContext::ReadSnapshotBytes(u64 address, void* out, u64 size) const {
    const size_t index = usnap.Find(address, size);
    if (index == UniformSnapshotView::npos) {
        return false;
    }
    std::memcpy(out, usnap.Bytes(index, address), size);
    return true;
}

} // namespace Vulkan::Nvn
