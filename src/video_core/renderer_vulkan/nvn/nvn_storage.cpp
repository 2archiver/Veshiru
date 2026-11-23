// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Texture storages (the VkImages behind NVN textures), virtual memory pools and the coherency of
// both with guest memory: the storage cache, lazy image creation, MapVirtual, CPU write
// notifications and the ReadTexels flush. See DESIGN.md "Memory and coherency".

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"

#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/textures/decoders.h"

namespace Vulkan::Nvn {
namespace {

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

} // Anonymous namespace

u64 Context::Impl::PollTick() {
    // Idle ticks of one ordinary frame are not kept when the next present comes, so at a steady
    // frame rate the clock moves one tick per present. An ordinary frame is up to twice the
    // recent frame length, and 64 ms at least. With the 64 ms alone every frame below ~15 fps
    // kept its idle ticks: the clock ran by wall time (~7 ticks per present at 9 fps) and each
    // present paid the texture checks of ~7 (Ocarina of Time, Hyrule Field).
    constexpr u64 IDLE_FRAME_TICKS = 4;
    const auto now = std::chrono::steady_clock::now();
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    if (poll_last_change == std::chrono::steady_clock::time_point{}) {
        poll_presents_seen = presents;
        poll_last_change = now;
        poll_tick = (std::max)(poll_tick, presents);
        poll_tick_at_change = poll_tick;
    } else if (presents != poll_presents_seen) {
        // A stretch without presents (a loading screen, a cold start's compile stalls) moved the
        // clock ahead of the present count. Counting from the present count again froze it until
        // the presents caught up, and a check scheduled during the stretch waited that long
        // (Mario Kart World: glyphs written then stayed blank for ~900 presents). The next
        // stretch counts from the clock as it stands.
        const u64 idle_ticks = poll_tick - poll_tick_at_change;
        const u64 advance = presents > poll_presents_seen ? presents - poll_presents_seen : 0;
        const u64 ordinary = (std::max)(IDLE_FRAME_TICKS, poll_frame_ticks_x8 / 4);
        // The recent frame length (a stall counts as an ordinary frame at most).
        const u64 frame_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - poll_last_change).count());
        const u64 frame_x8 = (std::min)(frame_ms / 2 / (std::max)(advance, u64{1}), ordinary * 8);
        poll_frame_ticks_x8 = (poll_frame_ticks_x8 * 7 + frame_x8 + 4) / 8;
        poll_tick_at_change += advance + (idle_ticks > ordinary ? idle_ticks : 0);
        poll_tick = (std::max)(poll_tick, poll_tick_at_change);
        poll_presents_seen = presents;
        poll_last_change = now;
    }
    const u64 idle =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(now - poll_last_change)
                             .count()) /
        16;
    poll_tick = (std::max)(poll_tick, poll_tick_at_change + idle);
    return poll_tick;
}

bool Context::Impl::TakePoolCpuWrites(Pool& pool) {
    if (!mirror_hash_tracked) {
        return false;
    }
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr) {
        return false;
    }
    if (!pool.split_runs.empty()) {
        // Raw stores reach a split pool's guest bytes past the tracker: once one completed, every
        // page is hashed again before it is trusted (unless write-through alias kept mirrors current).
        // Load split_raw_epoch (acquire) before split_raw_epoch_unaliased (acquire) to pair with
        // the completion thread's reverse release store order.
        const u64 raw_epoch = split_raw_epoch.load(std::memory_order_acquire);
        pool.split_raw_epoch_unaliased_seen =
            split_raw_epoch_unaliased.load(std::memory_order_acquire);
        if (raw_epoch != pool.split_raw_epoch_seen) {
            pool.split_raw_epoch_seen = raw_epoch;
            if (!mario_kart_world) {
                for (Pool::MirrorPage& page : pool.mirror_pages) {
                    page.trusted = false;
                }
            }
            // Textures of the pool must see raw writes even in Mario Kart World; only the mirror
            // page exception is Mario Kart World specific. The store names no page: the textures
            // lose their full baseline (DetectGuestChange), the pages are not marked written.
            // Marking every page made each check of each texture of every split pool rehash the
            // whole texture, at every raw store (Ocarina of Time: ~130 MB hashed per present).
            pool.raw_write_gen = ++pool.cpu_write_gen;
        }
    }
    std::vector<std::pair<u64, u64>> writebacks;
    {
        std::scoped_lock lock{pool.dirty_mutex};
        writebacks.swap(pool.landed_writebacks);
    }
    if (!writebacks.empty()) {
        ++pool.cpu_write_gen;
        for (const auto& [offset, end] : writebacks) {
            for (u64 page = offset / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < end; ++page) {
                const size_t index = static_cast<size_t>(page);
                if (index < pool.page_write_gen.size()) {
                    pool.page_write_gen[index] = pool.cpu_write_gen;
                }
            }
        }
    }
    if (pool.mirror_track == 0) {
        if (pool.mirror_track_tried) {
            return false;
        }
        pool.mirror_track_tried = true;
        pool.mirror_track = memory->Track(pool.address, pool.size);
        if (pool.mirror_track == 0) {
            return false;
        }
        // Watched from now on: a page hashed after this shows any later CPU write. No page is
        // trusted yet.
        memory->Arm(pool.mirror_track, pool.address, pool.size);
        pool.mirror_write_epoch = memory->WriteEpoch(pool.mirror_track);
        // mirror_epoch_clean stays as it was: the next call collects writes made right after Arm.
        pool.page_write_gen.assign(
            static_cast<size_t>((pool.size + Pool::MIRROR_PAGE - 1) / Pool::MIRROR_PAGE), 0);
        return true;
    }
    u64 epoch = 0;
    if (pool.mirror_write_epoch) {
        epoch = pool.mirror_write_epoch->load(std::memory_order_acquire);
        if (epoch == pool.mirror_epoch_clean) {
            return true;
        }
    }
    // The whole pool: a taken write lives on as an untrusted page until that page is hashed, so a
    // range checked later does not lose it.
    thread_local std::vector<std::pair<u64, u64>> written;
    written.clear();
    memory->TakeWritten(pool.mirror_track, pool.address, pool.size, written);
    if (!written.empty()) {
        ++pool.cpu_write_gen;
    }
    for (const auto& [begin, end] : written) {
        const u64 from = (std::max)(begin, pool.address) - pool.address;
        const u64 to = (std::min)(end - pool.address, pool.size);
        for (u64 page = from / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < to; ++page) {
            const size_t index = static_cast<size_t>(page);
            if (index < pool.mirror_pages.size()) {
                pool.mirror_pages[index].trusted = false;
                if (pool.raw_written_through &&
                    pool.mirror_pages[index].raw_epoch != pool.split_raw_epoch_seen) {
                    pool.mirror_pages[index].valid = false;
                }
            }
            if (index < pool.page_write_gen.size()) {
                pool.page_write_gen[index] = pool.cpu_write_gen;
            }
        }
    }
    pool.mirror_epoch_clean = epoch; // read before the take: a write in between shows as a change
    return true;
}

bool Context::Impl::PoolWrittenSince(const Pool& pool, u64 offset, u64 size, u32 gen) {
    if (size == 0) {
        return false;
    }
    const u64 first = offset / Pool::MIRROR_PAGE;
    const u64 last = (offset + size - 1) / Pool::MIRROR_PAGE;
    if (last >= pool.page_write_gen.size()) {
        return true;
    }
    for (u64 page = first; page <= last; ++page) {
        if (pool.page_write_gen[static_cast<size_t>(page)] > gen) {
            return true;
        }
    }
    return false;
}

bool Context::Impl::SettleLinearWriteback(Storage& s) {
    if (s.linear_writeback_seq == 0) {
        return true;
    }
    if (completed_seq.load(std::memory_order_acquire) < s.linear_writeback_seq) {
        return false;
    }
    s.linear_writeback_seq = 0;
    s.watch_valid = false;
    if (s.origin == StorageOrigin::Gpu) {
        s.gpu_guest_sig = 0;
        s.gpu_guest_sig_valid = false;
        s.gpu_takeover_armed = false;
        const auto pool_it = pools.find(s.pool_serial);
        if (pool_it != pools.end() && pool_it->second) {
            Pool& pool = *pool_it->second;
            if ((pool.flags & PoolFlag::CpuNoAccess) == 0 && !pool.IsVirtual() &&
                s.pool_offset < pool.size) {
                const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
                if (size != 0) {
                    std::vector<u8> scratch;
                    if (const u8* const data = GuestBytes(pool, s.pool_offset, size, scratch)) {
                        s.gpu_guest_sig = SampleGuestBytes(data, size, mario_kart_world);
                        s.gpu_guest_sig_valid = true;
                    }
                    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
                    if (pool.takeover_track == 0 && !pool.takeover_track_tried) {
                        pool.takeover_track_tried = true;
                        if (memory != nullptr) {
                            pool.takeover_track = memory->Track(pool.address, pool.size);
                        }
                    }
                    if (pool.takeover_track != 0 && memory != nullptr) {
                        memory->Arm(pool.takeover_track, pool.address + s.pool_offset, size);
                        s.gpu_takeover_armed = true;
                    }
                }
            }
        }
    }
    return true;
}

void Context::Impl::DetectGuestChange(Storage& s, u64 seq) {
    // A storage the GPU wrote is watched too (a decoder writing a video plane the game cleared
    // first): its baseline is the guest signature at the first check after the GPU write, so a
    // GPU-only change never looks like a CPU one.
    if (s.linear_writeback_seq != 0 && !SettleLinearWriteback(s)) {
        return;
    }
    if (s.sparse || s.force_upload || s.samples != VK_SAMPLE_COUNT_1_BIT ||
        s.origin == StorageOrigin::None || s.checked_seq == seq) {
        return;
    }
    s.checked_seq = seq;
    const u64 tick = PollTick();
    if (!s.force_check && tick < s.next_check_tick) {
        return;
    }
    const bool forced = s.force_check;
    s.force_check = false;
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end() || s.pool_offset >= pool_it->second->size) {
        return;
    }
    Pool& pool = *pool_it->second;
    const bool watching = s.origin == StorageOrigin::Gpu;
    if (watching && ((pool.flags & PoolFlag::CpuNoAccess) != 0 || pool.IsVirtual())) {
        return;
    }
    const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    if (size == 0) {
        return;
    }
    constexpr u64 SMALL_STORAGE = u64{16} << 20;
    const auto schedule_unchanged = [&](bool frequent) {
        ++s.unchanged_checks;
        u64 backoff = CheckBackoff((std::min)(s.unchanged_checks, 13u)); // 13+: the 30 cap
        if (mario_kart_world && !frequent && !s.dynamic && s.unchanged_checks >= 13) {
            backoff = 120;
        }
        if (frequent && size <= SMALL_STORAGE) {
            backoff = (std::min)(backoff, u64{3}); // a new glyph shows within a few frames
        }
        s.next_check_tick = tick + 1 + (s.dynamic ? (std::min)(backoff, u64{1}) : backoff);
    };
    // The page tracker saw none of the storage's pages written since a full hash matched: the
    // bytes are the same, no hash needed. Forced checks (a flush, a host write) still hash.
    Pool* check_pool = &pool;
    u64 check_offset = s.pool_offset;
    bool single_physical = !pool.IsVirtual();
    if (!single_physical) {
        single_physical =
            ResolveVirtualRange(pool, s.pool_offset, size, check_pool, check_offset) &&
            check_pool != nullptr;
    }
    const bool tracked = tex_hash_tracked && !watching && single_physical &&
                         (check_pool->flags & PoolFlag::CpuNoAccess) == 0 &&
                         TakePoolCpuWrites(*check_pool);
    if (tracked && s.cpu_trusted_pool != check_pool->serial) {
        s.cpu_trusted_pool = check_pool->serial;
        s.cpu_trusted_gen = 0;
        s.cpu_full_gen = 0;
        s.page_hashes.clear();
    }
    const u32 tracked_gen = check_pool->cpu_write_gen;
    const bool trusted = tracked && !forced && s.cpu_trusted_gen != 0;
    // A raw store completed since the full compare: it may have written any byte, so the check
    // below is the one of an untracked texture (signature, full compare when due). Pages the
    // tracker saw written are still compared first.
    const bool raw_pending = trusted && check_pool->raw_write_gen > s.cpu_full_gen;
    const bool written =
        trusted && PoolWrittenSince(*check_pool, check_offset, size, s.cpu_trusted_gen);
    if (trusted && !written && !raw_pending) {
        stats.cpu_checks_trusted.fetch_add(1, std::memory_order_relaxed);
        schedule_unchanged(s.dynamic);
        return;
    }
    // The storage's part of pool page `page`, hashed (data = the storage's first byte).
    const u64 first_page = check_offset / Pool::MIRROR_PAGE;
    const u64 last_page = (check_offset + size - 1) / Pool::MIRROR_PAGE;
    const auto page_hash = [&](const u8* data, u64 page) {
        const u64 begin = (std::max)(page * Pool::MIRROR_PAGE, check_offset);
        const u64 end = (std::min)((page + 1) * Pool::MIRROR_PAGE, check_offset + size);
        return Common::CityHash64(reinterpret_cast<const char*>(data + (begin - check_offset)),
                                  static_cast<size_t>(end - begin));
    };
    if (written) {
        s.wants_page_hashes = true;
    }
    // Some pages were written since the bytes were trusted: compare those pages only. Their old
    // hashes describe the contents (an upload drops them), so a page that differs means the
    // contents are stale, and equal pages mean the bytes are the trusted ones again.
    if (written && s.page_hashes.size() == last_page - first_page + 1 &&
        last_page < check_pool->page_write_gen.size()) {
        std::vector<u8> page_scratch;
        const auto page_begin = std::chrono::steady_clock::now();
        if (const u8* bytes = GuestBytes(pool, s.pool_offset, size, page_scratch)) {
            bool differs = false;
            u64 rehashed = 0;
            for (u64 page = first_page; page <= last_page; ++page) {
                if (check_pool->page_write_gen[static_cast<size_t>(page)] <= s.cpu_trusted_gen) {
                    continue;
                }
                ++rehashed;
                if (page_hash(bytes, page) != s.page_hashes[static_cast<size_t>(page - first_page)]) {
                    differs = true;
                    break;
                }
            }
            stats.hash_ns.fetch_add(static_cast<u64>(
                                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - page_begin)
                                            .count()),
                                    std::memory_order_relaxed);
            stats.hash_bytes.fetch_add(rehashed * Pool::MIRROR_PAGE, std::memory_order_relaxed);
            stats.cpu_pages_rehashed.fetch_add(rehashed, std::memory_order_relaxed);
            if (differs) {
                stats.cpu_changes.fetch_add(1, std::memory_order_relaxed);
                s.cpu_trusted_gen = 0;
                s.page_hashes.clear();
                s.dynamic = true;
                s.force_upload = true;
                return;
            }
            stats.cpu_checks_pages.fetch_add(1, std::memory_order_relaxed);
            s.cpu_trusted_gen = tracked_gen;
            if (!raw_pending) {
                schedule_unchanged(true);
                return;
            }
        }
    }
    // A texture the CPU keeps writing parts of (a glyph atlas) shows new content only in a few
    // lines the signature may not sample: small uncompressed ones and ones that changed before
    // are compared in full at every check. Block-compressed data is rewritten whole (a streamed
    // tile), so its signature plus the scheduled full compare are enough. Mario Kart World's
    // glyph atlas is 1024x8192 R8 (8 MiB): above 4 MiB it only had the scheduled full compare,
    // whose per-present budget (FULL_HASH_BYTES_PER_PRESENT) the many new textures of a cold
    // start used up first, so glyphs written then stayed blank.
    const bool compressed = s.desc.format != 0 && FormatBlockInfo(s.desc.format).width > 1;
    const bool every_check =
        forced || s.dynamic || ((!trusted || written) && size <= SMALL_STORAGE && !compressed);
    bool full = every_check || tick >= s.next_full_tick;
    if (full && !every_check) {
        // The scheduled full compares of the rest share a byte budget per present (a 2 GB texture
        // set has ~20 MB due each present otherwise): one that does not fit stays due and is
        // looked at again at its next check.
        if (full_hash_tick != tick) {
            full_hash_tick = tick;
            full_hash_bytes = 0;
        }
        if (full_hash_bytes >= FULL_HASH_BYTES_PER_PRESENT) {
            full = false;
        } else {
            full_hash_bytes += size;
        }
    }
    // A virtual pool's storage is gathered from the physical pools: not copied whole for a
    // signature alone.
    const bool signature_only = watching || !full;
    const bool gathered_lines = pool.IsVirtual() && signature_only;
    std::vector<u8> scratch;
    const auto guest_begin = std::chrono::steady_clock::now();
    const u8* data = gathered_lines ? nullptr : GuestBytes(pool, s.pool_offset, size, scratch);
    if (!data && !gathered_lines) {
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    const u64 guest_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(begin - guest_begin).count());
    const auto signature = [&]() -> u64 {
        if (data) {
            return SampleGuestBytes(data, size, mario_kart_world);
        }
        return SampleGuestBytesBy(size, [&](u64 offset, u64 count, u8* destination) {
            std::memset(destination, 0, static_cast<size_t>(count));
            GatherVirtual(pool, s.pool_offset + offset, count, destination);
        }, mario_kart_world);
    };
    bool changed = false;
    u64 hashed = (std::min)(size, u64{16384});
    if (watching) {
        const u64 sample = signature();
        changed = s.watch_valid && sample != s.sample_hash;
        s.sample_hash = sample;
        s.watch_valid = true;
    } else {
        changed = signature() != s.sample_hash;
        if (!changed && full) {
            changed = HashGuestBytes(data, size) != s.guest_hash;
            hashed += size;
            s.next_full_tick = tick + FULL_VERIFY_PRESENTS;
            // Taken before the bytes were read: a write since shows with a later generation.
            s.cpu_trusted_pool = (!changed && tracked) ? check_pool->serial : 0;
            s.cpu_trusted_gen = (!changed && tracked) ? tracked_gen : 0;
            s.cpu_full_gen = s.cpu_trusted_gen;
            // The bytes are the baseline's again: page hashes taken of it still describe them.
            const bool pages_current = s.cpu_trusted_gen != 0 &&
                                       s.page_hashes.size() == last_page - first_page + 1;
            if (!pages_current) {
                s.page_hashes.clear();
            }
            if (s.cpu_trusted_gen != 0 && tex_page_hash && s.wants_page_hashes && !pages_current) {
                s.page_hashes.resize(static_cast<size_t>(last_page - first_page + 1));
                for (u64 page = first_page; page <= last_page; ++page) {
                    s.page_hashes[static_cast<size_t>(page - first_page)] = page_hash(data, page);
                }
                hashed += size;
                TakePoolCpuWrites(*check_pool);
                if (PoolWrittenSince(*check_pool, check_offset, size, tracked_gen)) {
                    // CPU wrote memory between full compare and page hashing: drop the baseline.
                    s.cpu_trusted_gen = 0;
                    s.page_hashes.clear();
                }
            }
        }
        if (changed) {
            s.cpu_trusted_gen = 0;
            s.page_hashes.clear();
        }
    }
    const u64 check_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - begin)
                                              .count());
    stats.hash_ns.fetch_add(check_ns, std::memory_order_relaxed);
    stats.hash_bytes.fetch_add(hashed, std::memory_order_relaxed);
    if (s.desc.format != 0 && FormatBlockInfo(s.desc.format).width > 1) {
        stats.hash_bytes_compressed.fetch_add(hashed, std::memory_order_relaxed);
    }
    if (s.dynamic) {
        stats.hash_bytes_dynamic.fetch_add(hashed, std::memory_order_relaxed);
    }
    stats.cpu_checks.fetch_add(1, std::memory_order_relaxed);
    if (Debug::Get().upload_log && check_ns + guest_ns > 1'000'000) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] coherency check: present {} {}x{} fmt {:#x} size {:.2f} MB origin {} pool "
                 "{} full {} forced {} changed {}: hash {:.2f} ms, guest read {:.2f} ms",
                 stats.presents.load(std::memory_order_relaxed), s.extent.width, s.extent.height,
                 s.desc.format, size / 1048576.0, static_cast<int>(s.origin),
                 pool.IsVirtual() ? "virtual" : (pool.host ? "host" : "nohost"), full, forced,
                 changed, check_ns / 1e6, guest_ns / 1e6);
    }
    if (!changed) {
        schedule_unchanged(s.dynamic || written || !trusted);
        return;
    }
    stats.cpu_changes.fetch_add(1, std::memory_order_relaxed);
    s.dynamic = true;
    s.force_upload = true; // makes the upload treat it as a later one
}

bool Context::Impl::StorageNeedsCommands(Storage& s, bool will_overwrite, u64 seq,
                                         bool scaled_ok) {
    if (!will_overwrite && !s.sparse && s.image && s.contents_valid && !s.force_upload &&
        s.checked_seq == seq && s.scale_owner == nullptr &&
        (scaled_ok || !Rescale::NativeIsStale(s.scale_state)) && s.linear_writeback_seq == 0 &&
        s.pool_index != nullptr && s.pool_index_epoch == storage_index_epoch &&
        static_cast<const PoolIndex*>(s.pool_index)->write_gen == s.pool_gen_seen) {
        return false;
    }
    if (s.sparse && s.image && !will_overwrite && s.checked_seq == seq && !s.sparse->HasWork()) {
        return false;
    }
    if (!EnsureImage(s)) {
        return false;
    }
    if (s.scale_owner != nullptr) {
        return false; // a twin has no guest bytes: its owner decides what is current
    }
    if (will_overwrite) {
        return false;
    }
    if (!scaled_ok && Rescale::NativeIsStale(s.scale_state)) {
        return true; // the guest-sized image has to be made from the twin
    }
    if (s.sparse) {
        if (s.checked_seq != seq) {
            VerifySparse(s);
            s.checked_seq = seq;
        }
        return s.sparse->HasWork();
    }
    if (!s.contents_valid || s.force_upload) {
        return true;
    }
    DetectGuestChange(s, seq);
    if (s.force_upload) {
        return true;
    }
    const PoolIndex* index_ptr = static_cast<const PoolIndex*>(s.pool_index);
    if (index_ptr == nullptr || s.pool_index_epoch != storage_index_epoch) {
        const auto index_it = storage_index.find(s.pool_serial);
        if (index_it == storage_index.end()) {
            return false;
        }
        index_ptr = &index_it->second;
        s.pool_index = index_ptr;
        s.pool_index_epoch = storage_index_epoch;
    }
    const PoolIndex& index = *index_ptr;
    if (s.pool_gen_seen == index.write_gen) {
        return false;
    }
    const u64 begin = s.pool_offset;
    const u64 end = begin + s.layout.size;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        const Storage& other = *it->second;
        if (&other != &s && (other.image || Rescale::NativeIsStale(other.scale_state)) &&
            other.gpu_write_gen > s.sync_gen &&
            it->first + other.layout.size > begin) {
            return true;
        }
    }
    // Nothing newer overlaps: what Recorder::ResolveOverlaps records when it finds no source, so
    // the next check skips the scan until another GPU write to the pool.
    s.sync_gen = write_clock;
    s.pool_gen_seen = index.write_gen;
    return false;
}

Tegra::Texture::ASTC::Channels Context::Impl::ClassifyStandin(const Storage& s, const u8* data,
                                                              u64 size, bool& empty) {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    const auto begin = std::chrono::steady_clock::now();
    const Tegra::Texture::ASTC::Channels channels = Tegra::Texture::ASTC::ClassifyChannels(
        std::span<const u8>{data, static_cast<size_t>(size)}, block.width, block.height, &empty);
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - begin)
                                        .count());
    const size_t bucket = empty ? 4 : static_cast<size_t>(channels);
    for (SmartBcCounts* counts : {&smart_total, &smart_window}) {
        counts->classes[bucket].fetch_add(1, std::memory_order_relaxed);
        counts->scan_bytes.fetch_add(size, std::memory_order_relaxed);
        counts->scan_ns.fetch_add(ns, std::memory_order_relaxed);
    }
    return channels;
}

void Context::Impl::ChooseSmartStandin(Storage& s) {
    s.standin_empty = false;
    s.standin_fresh = false;
    s.standin_class = Tegra::Texture::ASTC::Channels::Rgba;
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end()) {
        return;
    }
    const Pool& pool = *pool_it->second;
    if ((!pool.pending.load(std::memory_order_acquire) && pool.backing == Pool::Backing::None) ||
        s.pool_offset >= pool.size) {
        return;
    }
    const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    std::vector<u8> scratch;
    const u8* data = GuestBytes(pool, s.pool_offset, size, scratch);
    if (data == nullptr) {
        return;
    }
    bool empty = false;
    const Tegra::Texture::ASTC::Channels channels = ClassifyStandin(s, data, size, empty);
    s.standin_fresh = true;
    if (empty) {
        s.standin_empty = true; // not loaded yet: BC3 until a whole upload brings blocks
        return;
    }
    s.standin_class = channels;
    SetStandin(s, SmartStandinFor(channels,
                                  VideoCore::Surface::IsPixelFormatSRGB(s.host.pixel_format) ||
                                      s.standin_srgb_view,
                                  smart_bc));
}

void Context::Impl::SetStandin(Storage& s, AstcStandin standin) {
    s.host = ToBcStandin(s.host, standin);
    const u32 block_bytes = BcStandinBlockBytes(s.host);
    u64 bytes = 0;
    for (u32 level = 0; level < s.levels; ++level) {
        bytes += BcBlocksSize((std::max)(s.extent.width >> level, 1u),
                              (std::max)(s.extent.height >> level, 1u), 1, block_bytes);
    }
    s.vram_bytes = bytes * s.layers;
}

vk::Image Context::Impl::TakeRecycledImage(const VkImageCreateInfo& info) {
    if (image_recycle_count == 0) {
        return {};
    }
    const ImageRecycleKey key{
        .flags = info.flags,
        .type = info.imageType,
        .format = info.format,
        .width = info.extent.width,
        .height = info.extent.height,
        .depth = info.extent.depth,
        .levels = info.mipLevels,
        .layers = info.arrayLayers,
        .samples = info.samples,
        .usage = info.usage,
    };
    const auto it = image_recycle_pool.find(key);
    if (it == image_recycle_pool.end() || it->second.empty()) {
        return {};
    }
    RecycledImage entry = std::move(it->second.back());
    it->second.pop_back();
    if (it->second.empty()) {
        image_recycle_pool.erase(it);
    }
    image_recycle_bytes -= (std::min)(image_recycle_bytes, entry.vram_bytes);
    --image_recycle_count;
    stats.storages_recycled.fetch_add(1, std::memory_order_relaxed);
    return std::move(entry.image);
}

void Context::Impl::RecycleImage(Storage& s) {
    if (!s.image || s.sparse || s.vram_bytes == 0 || s.vram_bytes > IMAGE_RECYCLE_MAX_BYTES / 2) {
        return;
    }
    const ImageRecycleKey key{
        .flags = s.create_flags,
        .type = s.type,
        .format = s.host.format,
        .width = s.extent.width,
        .height = s.extent.height,
        .depth = s.extent.depth,
        .levels = s.levels,
        .layers = s.layers,
        .samples = s.samples,
        .usage = s.usage,
    };
    const u64 bytes = s.vram_bytes;
    image_recycle_pool[key].push_back(RecycledImage{
        .key = key,
        .image = std::move(s.image),
        .vram_bytes = bytes,
        .lru_tick = ++image_recycle_tick,
    });
    s.image = vk::Image{};
    image_recycle_bytes += bytes;
    ++image_recycle_count;
    while (image_recycle_bytes > IMAGE_RECYCLE_MAX_BYTES ||
           image_recycle_count > IMAGE_RECYCLE_MAX_COUNT) {
        auto oldest_bucket = image_recycle_pool.end();
        u64 oldest_tick = ~u64{0};
        for (auto it = image_recycle_pool.begin(); it != image_recycle_pool.end(); ++it) {
            if (!it->second.empty() && it->second.front().lru_tick < oldest_tick) {
                oldest_tick = it->second.front().lru_tick;
                oldest_bucket = it;
            }
        }
        if (oldest_bucket == image_recycle_pool.end()) {
            break;
        }
        const u64 evicted_bytes = oldest_bucket->second.front().vram_bytes;
        if (oldest_bucket->second.size() == 1) {
            image_recycle_pool.erase(oldest_bucket);
        } else {
            oldest_bucket->second.erase(oldest_bucket->second.begin());
        }
        image_recycle_bytes -= (std::min)(image_recycle_bytes, evicted_bytes);
        --image_recycle_count;
    }
}

void Context::Impl::ClearRecycledImages() {
    image_recycle_pool.clear();
    image_recycle_bytes = 0;
    image_recycle_count = 0;
}

bool Context::Impl::EnsureImage(Storage& s) {
    if (s.image) {
        return true;
    }
    if (s.image_failed) {
        return false;
    }
    // The first image of a smart stand-in (Recorder::RefitStandin chooses for later ones).
    if (s.smart_standin && !s.standin_pinned && s.origin == StorageOrigin::None) {
        ChooseSmartStandin(s);
    }
    try {
        const VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = s.create_flags,
            .imageType = s.type,
            .format = s.host.format,
            .extent = s.extent,
            .mipLevels = s.levels,
            .arrayLayers = s.layers,
            .samples = s.samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = s.usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if (s.sparse) {
            // Tile-backed: a sparse residency image with no memory; the tiles the pool maps get
            // theirs (SparseBacking::Prepare).
            const char* reason = "the device cannot create a sparse residency image";
            // Shared tile pages (SparseShareEnabled) need SPARSE_ALIASED images.
            s.image = allocator->CreateUnbackedImage(image_info, SparseShareEnabled());
            ++s.image_gen;
            if (s.image && s.sparse->Attach(*this, s, reason)) {
                s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                stats.storages_created.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            LOG_WARNING(Render_Vulkan,
                        "[nvn] sparse texture {}x{}x{} ({} levels, format {:#x}) is not tile "
                        "backed: {}; it is an ordinary image",
                        s.extent.width, s.extent.height, s.layers, s.levels, s.desc.format, reason);
            s.image = vk::Image{};
            s.sparse.reset();
            // The whole-storage paths start from scratch.
            s.contents_valid = false;
            s.origin = StorageOrigin::None;
        }
        s.image = TakeRecycledImage(image_info);
        if (!s.image) {
            s.image = allocator->CreateImage(image_info, false, AllocationKind::StorageImage);
        }
        ++s.image_gen;
    } catch (const vk::Exception& exception) {
        s.image_failed = true;
        stats.textures_failed.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR(Render_Vulkan, "[nvn] texture storage ({}x{}x{}, {} levels, format {:#x}) image failed: {}",
                  s.extent.width, s.extent.height, s.extent.depth, s.levels, s.desc.format,
                  exception.what());
        return false;
    }
    s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    s.last_used_present = stats.presents.load(std::memory_order_relaxed);
    stats.storages_created.fetch_add(1, std::memory_order_relaxed);
    if (s.cached) {
        storage_cache_bytes += s.vram_bytes;
        TrimStorageCache();
        stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
    }
    return true;
}

const u8* Context::Impl::GuestBytes(const Pool& pool, u64 offset, u64 size,
                                    std::vector<u8>& scratch) const {
    if (size == 0 || offset > pool.size || size > pool.size - offset) {
        return nullptr;
    }
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    if (!pool.IsVirtual()) {
        if (pool.host) {
            return pool.host + offset;
        }
        if (const u8* pointer = GuestPointer(pool.address + offset, size)) {
            return pointer;
        }
        scratch.resize(static_cast<size_t>(size));
        if (memory && memory->Read(pool.address + offset, scratch.data(), size)) {
            return scratch.data();
        }
        return nullptr;
    }
    // A virtual pool: gather the mapped parts from the physical pools, zero elsewhere.
    scratch.assign(static_cast<size_t>(size), 0);
    GatherVirtual(pool, offset, size, scratch.data());
    return scratch.data();
}

void Context::Impl::GatherVirtual(const Pool& pool, u64 offset, u64 size, u8* out) const {
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    auto it = pool.virtual_runs.upper_bound(offset);
    if (it != pool.virtual_runs.begin()) {
        --it;
    }
    for (; it != pool.virtual_runs.end() && it->first < offset + size; ++it) {
        const u64 run_begin = it->first;
        const Pool::VirtualRun& run = it->second;
        const u64 begin = (std::max)(run_begin, offset);
        const u64 end = (std::min)(run_begin + run.size, offset + size);
        if (begin >= end) {
            continue;
        }
        const auto physical = pools.find(run.physical_serial);
        if (physical == pools.end()) {
            continue;
        }
        const Pool& source = *physical->second;
        const u64 source_offset = run.physical_offset + (begin - run_begin);
        if (source_offset > source.size || end - begin > source.size - source_offset) {
            continue;
        }
        u8* const target = out + (begin - offset);
        if (const u8* pointer = source.host ? source.host + source_offset
                                            : GuestPointer(source.address + source_offset,
                                                           end - begin)) {
            std::memcpy(target, pointer, static_cast<size_t>(end - begin));
        } else if (memory) {
            memory->Read(source.address + source_offset, target, end - begin);
        }
    }
}

bool Context::Impl::ResolveVirtualRange(const Pool& virtual_pool, u64 offset, u64 size,
                                        Pool*& physical, u64& physical_offset) const {
    const auto& runs = virtual_pool.virtual_runs;
    auto it = runs.upper_bound(offset);
    if (it == runs.begin()) {
        return false;
    }
    --it;
    u64 cursor = offset;
    const u64 end = offset + (std::max)(size, u64{1});
    Pool* first = nullptr;
    u64 first_offset = 0;
    for (; it != runs.end() && cursor < end; ++it) {
        const u64 run_begin = it->first;
        const Pool::VirtualRun& run = it->second;
        if (cursor < run_begin || cursor >= run_begin + run.size) {
            return false; // a hole
        }
        const auto found = pools.find(run.physical_serial);
        if (found == pools.end()) {
            return false;
        }
        Pool* const pool = found->second.get();
        const u64 at = run.physical_offset + (cursor - run_begin);
        if (!first) {
            first = pool;
            first_offset = at;
        } else if (pool != first || at != first_offset + (cursor - offset)) {
            return false; // scattered over the physical pools: one buffer cannot cover it
        }
        cursor = run_begin + run.size;
    }
    if (cursor < end || !first) {
        return false;
    }
    physical = first;
    physical_offset = first_offset;
    return true;
}

std::string Context::Impl::DescribeUnresolved(u64 address, u64 size) const {
    auto it = pools_by_address.upper_bound(address);
    if (it != pools_by_address.begin()) {
        --it;
        const Pool& pool = *it->second;
        if (address >= pool.address && address - pool.address <= pool.size) {
            if (pool.IsVirtual()) {
                return fmt::format("range {:#x}+{:#x} of virtual pool {:#x} is unmapped or is "
                                   "backed by scattered physical ranges",
                                   address - pool.address, size, pool.address);
            }
            return fmt::format("range {:#x}+{:#x} leaves pool {:#x} ({:#x} bytes)",
                               address - pool.address, size, pool.address, pool.size);
        }
    }
    return fmt::format("address {:#x} (+{:#x}) is not inside a memory pool", address, size);
}

namespace {

/// A mapping of the virtual pool over a range, in the form the old and the new state of a remap
/// are compared in.
struct RunView {
    u64 start;
    u64 size;
    u64 serial;
    u64 physical;
    bool operator==(const RunView&) const = default;
};

/// The runs of [begin, end), cut to it, neighbours that continue each other merged.
std::vector<RunView> SnapshotRuns(
    const std::map<u64, Pool::VirtualRun>& runs, u64 begin, u64 end,
    const std::unordered_map<u64, std::shared_ptr<Pool>>* physical_pools) {
    std::vector<RunView> out;
    auto it = runs.upper_bound(begin);
    if (it != runs.begin()) {
        --it;
    }
    for (; it != runs.end() && it->first < end; ++it) {
        const u64 run_end = it->first + it->second.size;
        const u64 from = (std::max)(it->first, begin);
        const u64 to = (std::min)(run_end, end);
        if (from >= to) {
            continue;
        }
        RunView view{from, to - from, it->second.physical_serial,
                     it->second.physical_offset + (from - it->first)};
        if (physical_pools != nullptr) {
            const auto physical = physical_pools->find(view.serial);
            if (physical != physical_pools->end() && physical->second->address != 0 &&
                (physical->second->flags & PoolFlag::Virtual) == 0) {
                view.physical += physical->second->address;
                view.serial = 0; // canonical guest backing, independent of the pool object
            }
        }
        if (!out.empty() && out.back().start + out.back().size == view.start &&
            out.back().serial == view.serial &&
            out.back().physical + out.back().size == view.physical) {
            out.back().size += view.size;
        } else {
            out.push_back(view);
        }
    }
    return out;
}

} // Anonymous namespace

bool Context::Impl::VirtualRangeMapped(const Pool& virtual_pool, u64 offset, u64 size) {
    const auto& runs = virtual_pool.virtual_runs;
    auto it = runs.upper_bound(offset);
    if (it != runs.begin()) {
        const auto previous = std::prev(it);
        if (previous->first + previous->second.size > offset) {
            return true;
        }
    }
    return it != runs.end() && it->first < offset + size;
}

void Context::Impl::MapVirtualNow(u64 pool_serial, const std::vector<VC::VirtualMapping>& mappings) {
    const auto found = pools.find(pool_serial);
    if (found == pools.end() || !found->second->IsVirtual()) {
        LOG_ERROR(Render_Vulkan, "[nvn] MapVirtual on {:#x}: not a virtual pool", pool_serial);
        return;
    }
    Pool& pool = *found->second;
    if (!mappings.empty()) {
        ++mkw_mapping_generation;
    }
    PoolIndex* const index = [&]() -> PoolIndex* {
        const auto it = storage_index.find(pool_serial);
        return it == storage_index.end() ? nullptr : &it->second;
    }();
    for (const VC::VirtualMapping& mapping : mappings) {
        if (mapping.size == 0 || mapping.virtual_offset > pool.size ||
            mapping.size > pool.size - mapping.virtual_offset) {
            continue;
        }
        const u64 begin = mapping.virtual_offset;
        const u64 end = begin + mapping.size;
        mkw_sparse_users_dirty = true;
        auto& runs = pool.virtual_runs;
        // As on the emulated GPU path, mapping the same backing through another pool
        // object must not replace a lightmap tile with an upload of older guest bytes.
        const auto* physical_pools = mario_kart_world ? &pools : nullptr;
        const std::vector<RunView> before = SnapshotRuns(runs, begin, end, physical_pools);
        // Cut the existing runs that overlap [begin, end).
        auto it = runs.lower_bound(begin);
        if (it != runs.begin()) {
            auto previous = std::prev(it);
            const u64 previous_end = previous->first + previous->second.size;
            if (previous_end > begin) {
                Pool::VirtualRun tail = previous->second;
                const u64 tail_begin = end;
                previous->second.size = begin - previous->first;
                if (previous_end > end) {
                    tail.physical_offset += tail_begin - previous->first;
                    tail.size = previous_end - tail_begin;
                    runs.emplace(tail_begin, tail);
                }
            }
        }
        it = runs.lower_bound(begin);
        while (it != runs.end() && it->first < end) {
            const u64 run_end = it->first + it->second.size;
            if (run_end > end) {
                Pool::VirtualRun tail = it->second;
                tail.physical_offset += end - it->first;
                tail.size = run_end - end;
                runs.emplace(end, tail);
            }
            it = runs.erase(it);
        }
        if (mapping.physical_pool_serial != 0) {
            runs.emplace(begin, Pool::VirtualRun{mapping.size, mapping.physical_pool_serial,
                                                 mapping.physical_offset});
        }
        // A remap that leaves the bytes where they were (the game maps what is already mapped)
        // changes nothing for the storages over it.
        if (!index || SnapshotRuns(runs, begin, end, physical_pools) == before) {
            continue;
        }
        // Storages placed over the remapped bytes read different bytes now.
        const u64 scan_from = begin > index->max_size ? begin - index->max_size : 0;
        for (auto s = index->spans.lower_bound(scan_from);
             s != index->spans.end() && s->first < end; ++s) {
            Storage& storage = *s->second;
            if (s->first + storage.layout.size <= begin) {
                continue;
            }
            if (storage.sparse) {
                // Tile by tile: only the tiles of the range are bound, unbound or uploaded again.
                const u64 from = (std::max)(begin, s->first) - s->first;
                const u64 to = (std::min)(end, s->first + storage.layout.size) - s->first;
                storage.sparse->Remap(from, to, [&](u64 offset, u64 size) {
                    return VirtualRangeMapped(pool, s->first + offset, size);
                });
                continue;
            }
            storage.contents_valid = false;
            storage.origin = StorageOrigin::None;
            storage.gpu_write_gen = 0; // no longer a source for overlapping textures
            storage.linear_writeback_seq = 0;
            storage.force_upload = false;
            storage.cpu_trusted_gen = 0;
            storage.cpu_full_gen = 0;
            storage.page_hashes.clear();
        }
    }
    stats.virtual_maps.fetch_add(1, std::memory_order_relaxed);
    if (stats.virtual_maps.load(std::memory_order_relaxed) <= 4) {
        u64 mapped = 0;
        for (const auto& [offset, run] : pool.virtual_runs) {
            mapped += run.size;
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] MapVirtual: virtual pool {:#x} ({:#x} bytes): {} request(s), now {} run(s), "
                 "{:#x} bytes mapped",
                 pool.address, pool.size, mappings.size(), pool.virtual_runs.size(), mapped);
    }
}

void Context::Impl::VerifySparse(Storage& s) {
    SparseBacking& sparse = *s.sparse;
    if (s.origin == StorageOrigin::Gpu) {
        return; // the GPU wrote it: the guest bytes are not what it holds
    }
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end()) {
        return;
    }
    const Pool& virtual_pool = *pool_it->second;
    // At most this many tiles per present in all, and per call: a 64 KiB gather and a hash each.
    constexpr u32 TILES_PER_TICK = 64;
    constexpr u32 TILES_PER_CALL = 16;
    constexpr u32 TRACKED_SCAN_PER_CALL = 64;
    const u64 tick = PollTick();
    if (sparse_verify_tick != tick) {
        sparse_verify_tick = tick;
        sparse_verify_count = 0;
    }
    thread_local std::vector<u8> bytes;
    bytes.resize(static_cast<size_t>(SPARSE_TILE_BYTES));
    u32 hashed = 0;
    for (u32 n = 0; n < TRACKED_SCAN_PER_CALL && hashed < TILES_PER_CALL &&
                    sparse_verify_count < TILES_PER_TICK;
         ++n) {
        const u32 id = sparse.NextVerifyTile();
        if (id == ~0u) {
            break;
        }
        const u64 tile_offset = s.pool_offset + sparse.Layout().TileOffset(id);
        Pool* phys = nullptr;
        u64 phys_offset = 0;
        u32 before_gen = 0;
        if (tex_hash_tracked &&
            ResolveVirtualRange(virtual_pool, tile_offset, SPARSE_TILE_BYTES, phys, phys_offset) &&
            phys != nullptr && (phys->flags & PoolFlag::CpuNoAccess) == 0 &&
            TakePoolCpuWrites(*phys)) {
            before_gen = phys->cpu_write_gen;
            const u32 verified_gen = sparse.TileVerifiedGen(id);
            const bool same_pool = sparse.TileVerifiedPool(id) == phys->serial;
            if (same_pool && verified_gen != 0 &&
                (before_gen == verified_gen ||
                 (phys->raw_write_gen <= verified_gen &&
                  !PoolWrittenSince(*phys, phys_offset, SPARSE_TILE_BYTES, verified_gen)))) {
                sparse.SetTileVerifiedGen(id, phys->serial, before_gen);
                continue;
            }
        }
        ++hashed;
        ++sparse_verify_count;
        std::fill(bytes.begin(), bytes.end(), u8{0});
        GatherVirtual(virtual_pool, tile_offset, SPARSE_TILE_BYTES, bytes.data());
        if (Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size()) !=
            sparse.TileHash(id)) {
            sparse.MarkTileDirty(id);
        } else if (before_gen != 0 && phys != nullptr) {
            sparse.SetTileVerifiedGen(id, phys->serial, before_gen);
        }
    }
}

void Context::Impl::FlushedRuns(Pool& pool, u64 begin, u64 size,
                                std::vector<std::pair<u64, u64>>& runs) {
    constexpr u64 PAGE = MirrorPages::PAGE;
    PoolMirror& mirror = *pool.vram_mirror;
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    const u64 end = (std::min)(begin + size, pool.size);
    if (memory == nullptr) {
        runs.emplace_back(begin, end);
        return;
    }
    thread_local std::vector<std::pair<u64, u64>> written;
    written.clear();
    // The guest CPU's stores to pages the mirror holds are all in the tracker.
    memory->TakeWritten(mirror.track_id, pool.address + begin, end - begin, written);
    size_t next = 0;
    u64 kept = 0;
    u64 taken = 0;
    const auto push = [&](u64 from, u64 to) {
        from = (std::max)(from, begin);
        to = (std::min)(to, end);
        if (from >= to) {
            return;
        }
        if (!runs.empty() && runs.back().second == from) {
            runs.back().second = to;
        } else {
            runs.emplace_back(from, to);
        }
    };
    for (u64 page = begin / PAGE; page * PAGE < end; ++page) {
        const u64 offset = page * PAGE;
        while (next < written.size() && written[next].second <= pool.address + offset) {
            ++next;
        }
        const bool was_written = next < written.size() && written[next].first < pool.address + offset + PAGE;
        if (was_written || !mirror.pages.AllValid(offset, PAGE)) {
            push(offset, offset + PAGE);
            ++taken;
        } else {
            ++kept;
        }
    }
    stats.vram_flush_pages_kept.fetch_add(kept, std::memory_order_relaxed);
    stats.vram_flush_pages_taken.fetch_add(taken, std::memory_order_relaxed);
}

bool Context::Impl::SparseShareEnabled() {
    if (!sparse_share_enabled) {
        const char* value = std::getenv("EDEN_NX2_NVN_SPARSE_SHARE");
        const char forced = value == nullptr ? '\0' : value[0];
        const bool wanted = forced == '1' || (forced != '0' && mario_kart_world);
        const bool supported = device.IsSparseResidencyAliasedSupported();
        sparse_share_enabled = wanted && supported;
        LOG_INFO(Render_Vulkan, "[nvn] sparse tiles of the same guest bytes share pages: {}",
                 *sparse_share_enabled ? "on"
                 : !wanted             ? "off"
                                       : "off (the device cannot alias sparse memory)");
    }
    return *sparse_share_enabled;
}

std::span<const u64> Context::Impl::MkwSparseVirtualUsers(u64 physical_serial) {
    if (mkw_sparse_users_dirty) {
        mkw_sparse_physical_users.clear();
        for (const u64 serial : sparse_virtual_pools) {
            const auto found = pools.find(serial);
            if (found == pools.end()) {
                continue;
            }
            for (const auto& [offset, run] : found->second->virtual_runs) {
                auto& users = mkw_sparse_physical_users[run.physical_serial];
                // sparse_virtual_pools is ordered: append each virtual pool only once.
                if (users.empty() || users.back() != serial) {
                    users.push_back(serial);
                }
            }
        }
        mkw_sparse_users_dirty = false;
    }
    const auto found = mkw_sparse_physical_users.find(physical_serial);
    return found != mkw_sparse_physical_users.end() ? std::span<const u64>{found->second}
                                                   : std::span<const u64>{};
}

void Context::Impl::NotifyCpuWriteNow(u64 address, u64 size, bool host_write) {
    auto pool_it = pools_by_address.upper_bound(address);
    if (pool_it == pools_by_address.begin()) {
        return;
    }
    --pool_it;
    Pool& pool = *pool_it->second;
    if (address < pool.address || address - pool.address >= pool.size) {
        return;
    }
    const u64 begin = address - pool.address;
    // What the mirror has to look at: the whole range, or with `flush_tracked` the pages of a
    // guest flush that the guest wrote or the mirror does not hold.
    thread_local std::vector<std::pair<u64, u64>> runs;
    runs.clear();
    if (flush_tracked && !host_write && pool.vram_mirror) {
        FlushedRuns(pool, begin, size, runs);
    } else {
        runs.emplace_back(begin, begin + size);
    }
    // Live storage shadows are normally authoritative until their segment ends. Refresh only the
    // CPU-written overlap before either a whole-shadow copy-back or a mirror word-merge.
    InvalidatePendingInlineBytesOverlapping(address, size);
    for (const auto& [run_begin, run_end] : runs) {
        for (auto& [owner_queue, draw] : draw_contexts) {
            static_cast<void>(owner_queue);
            draw->OnCpuWrite(pool, run_begin, run_end - run_begin);
        }
    }
    const u64 end = begin + size;
    if (!pool.pending.load(std::memory_order_acquire) && !pool.mirror_pages.empty()) {
        // A mirror's pages in the range are compared with the guest's again at their next use,
        // even in a submission that already found them unchanged. Trusted pages stay trusted for a
        // guest flush with `flush_tracked` (the guest CPU's stores are in the page tracker), not
        // for a host service's write through a host pointer.
        const bool untrust = host_write || !flush_tracked;
        const u64 last = (std::min)(end, pool.size);
        for (u64 page = begin / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < last; ++page) {
            Pool::MirrorPage& mirror_page = pool.mirror_pages[static_cast<size_t>(page)];
            mirror_page.checked = 0;
            if (untrust) {
                mirror_page.trusted = false;
            }
        }
    }
    // A pool mirror copies the range in again at its next use (the guest CPU's writes are seen by
    // the page tracker too; a host service's writes through a host pointer only through this).
    if (pool.vram_mirror && pool.vram_mirror->monitor) {
        // EDEN_NX2_NVN_WB_MONITOR: how much of what the game declares written the page tracker
        // saw written (the rest is only a flush the guest made without a JIT store, or a host
        // service's write).
        MirrorMonitor& monitor = *pool.vram_mirror->monitor;
        VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
        const u64 last = (std::min)(end, pool.size);
        monitor.notify_calls.fetch_add(1, std::memory_order_relaxed);
        for (u64 page = begin / MirrorPages::PAGE; page * MirrorPages::PAGE < last; ++page) {
            const u64 offset = page * MirrorPages::PAGE;
            monitor.notify_pages.fetch_add(1, std::memory_order_relaxed);
            if (!pool.vram_mirror->pages.AllValid(offset, MirrorPages::PAGE)) {
                monitor.notify_invalid.fetch_add(1, std::memory_order_relaxed);
            } else if (memory != nullptr &&
                       memory->AnyWritten(pool.vram_mirror->track_id, pool.address + offset,
                                          MirrorPages::PAGE)) {
                monitor.notify_dirty.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (runs.empty() && flush_tracked && !host_write && pool.vram_mirror &&
        pool.mirror_pages.empty()) {
        return;
    }
    for (const auto& [run_begin, run_end] : runs) {
        MirrorInvalidate(pool, run_begin, run_end - run_begin);
    }
    if (!sparse_virtual_pools.empty() && !pool.IsVirtual()) {
        // The sparse textures that map these bytes through a virtual pool read them again.
        const auto notify_virtual = [&](u64 serial) {
            stats.sparse_write_pool_checks.fetch_add(1, std::memory_order_relaxed);
            const auto virtual_it = pools.find(serial);
            const auto index_it = storage_index.find(serial);
            if (virtual_it == pools.end() || index_it == storage_index.end()) {
                return;
            }
            for (const auto& [run_begin, run] : virtual_it->second->virtual_runs) {
                if (run.physical_serial != pool.serial || run.physical_offset >= end ||
                    run.physical_offset + run.size <= begin) {
                    continue;
                }
                const u64 from = (std::max)(run.physical_offset, begin) - run.physical_offset;
                const u64 to = (std::min)(run.physical_offset + run.size, end) - run.physical_offset;
                const u64 virtual_from = run_begin + from;
                const u64 virtual_to = run_begin + to;
                const PoolIndex& virtual_index = index_it->second;
                const u64 scan_from = virtual_from > virtual_index.max_size
                                          ? virtual_from - virtual_index.max_size
                                          : 0;
                for (auto it = virtual_index.spans.lower_bound(scan_from);
                     it != virtual_index.spans.end() && it->first < virtual_to; ++it) {
                    Storage& storage = *it->second;
                    if (!storage.sparse || it->first + storage.layout.size <= virtual_from) {
                        continue;
                    }
                    storage.sparse->ContentChanged(
                        (std::max)(virtual_from, it->first) - it->first,
                        (std::min)(virtual_to, it->first + storage.layout.size) - it->first);
                }
            }
        };
        if (mario_kart_world) {
            for (const u64 serial : MkwSparseVirtualUsers(pool.serial)) {
                notify_virtual(serial);
            }
        } else {
            for (const u64 serial : sparse_virtual_pools) {
                notify_virtual(serial);
            }
        }
    }
    const auto index_it = storage_index.find(pool.serial);
    if (index_it == storage_index.end()) {
        return;
    }
    const PoolIndex& index = index_it->second;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        Storage& storage = *it->second;
        if (it->first + storage.layout.size <= begin) {
            continue;
        }
        if (storage.origin == StorageOrigin::Cpu) {
            // The bytes may have changed: look at them at the next use, whatever the back-off says.
            // A guest flush with `flush_tracked` has its stores in the page tracker, so the check
            // may still go by it (as a mirror's trusted pages do); anything else hashes in full.
            if (flush_tracked && !host_write && tex_hash_tracked && tex_flush_trusted) {
                storage.next_check_tick = 0;
            } else {
                storage.force_check = true;
            }
        } else if (storage.origin == StorageOrigin::Gpu) {
            // A guest flush after a linear GPU write-back may have overwritten the written-back
            // bytes before DetectGuestChange ever sampled them; force an upload if the write-back
            // had already completed, otherwise drop the backoff so the next use checks at once.
            if (storage.linear_writeback_seq != 0 &&
                storage.linear_writeback_seq <= completed_seq.load(std::memory_order_acquire)) {
                storage.linear_writeback_seq = 0;
                storage.watch_valid = false;
                storage.force_upload = true;
            } else {
                storage.force_check = true;
            }
        }
    }
}

void Context::Impl::NotifyHostWriteNow(u64 address, u64 size) {
    // What a CPU write gets: mirror pages compared again, sparse tiles and CPU-sourced storages
    // looked at again.
    NotifyCpuWriteNow(address, size, true);
    auto pool_it = pools_by_address.upper_bound(address);
    if (pool_it == pools_by_address.begin()) {
        return;
    }
    --pool_it;
    Pool& pool = *pool_it->second;
    if (address < pool.address || address - pool.address >= pool.size || pool.IsVirtual()) {
        return;
    }
    // Buffer fills need 4-byte aligned ranges; the extra bytes are the guest's own.
    const u64 begin = (address - pool.address) & ~u64{3};
    const u64 end = (std::min)((address - pool.address + size + 3) & ~u64{3}, pool.size);
    // The bytes the host wrote, for the texture containment below.
    const u64 exact_begin = address - pool.address;
    const u64 exact_end = (std::min)(exact_begin + size, pool.size);
    // A GPU-only pool's buffer is the GPU's copy of the memory (an imported one is the memory
    // itself, a mirror compares its pages at their next use): it takes the new bytes now. A pool
    // still waiting for its first upload reads them then.
    if (pool.backing == Pool::Backing::DeviceLocal && !pool.upload_pending &&
        !pool.pending.load(std::memory_order_acquire)) {
        const PoolUpload range{&pool, begin, end - begin};
        UploadPoolRanges({&range, 1}, true);
        stats.host_write_uploads.fetch_add(1, std::memory_order_relaxed);
    }
    // A texture inside the range holds the written bytes now, whatever the GPU wrote to it before
    // (a decoder frame over the plane the game cleared); one only partly inside keeps its image
    // (the rest of its bytes may be GPU contents the guest memory does not hold).
    const auto index_it = storage_index.find(pool.serial);
    if (index_it == storage_index.end()) {
        return;
    }
    const PoolIndex& index = index_it->second;
    for (auto it = index.spans.lower_bound(exact_begin);
         it != index.spans.end() && it->first < exact_end; ++it) {
        Storage& storage = *it->second;
        if (storage.sparse || it->first + storage.layout.size > exact_end) {
            continue;
        }
        storage.force_upload = true;
        storage.force_check = true;
    }
}

void Context::Impl::NotifyTextureWrittenNow(u64 texture_serial) {
    const auto it = textures.find(texture_serial);
    if (it == textures.end() || !it->second->storage) {
        return;
    }
    Storage& storage = *it->second->storage;
    if (storage.sparse) {
        storage.sparse->MarkAllDirty(); // the game wrote texels: the tiles are uploaded again
        return;
    }
    storage.force_upload = true;
    storage.force_check = true;
}

void Context::Impl::EvictPoolStorages(u64 pool_serial) {
    const auto index_it = storage_index.find(pool_serial);
    if (index_it == storage_index.end()) {
        return;
    }
    std::vector<Storage*> doomed;
    for (const auto& [begin, storage] : index_it->second.spans) {
        doomed.push_back(storage);
    }
    storage_index.erase(index_it);
    if (sparse_virtual_pools.erase(pool_serial) != 0) {
        mkw_sparse_users_dirty = true;
    }
    ++storage_index_epoch;
    for (auto it = storage_cache.begin(); it != storage_cache.end();) {
        if (it->first.pool_serial == pool_serial) {
            Storage& storage = *it->second;
            if (storage.image && !storage.sparse) {
                storage_cache_bytes -= (std::min)(storage_cache_bytes, storage.vram_bytes);
            }
            storage.cached = false;
            stats.storages_evicted.fetch_add(1, std::memory_order_relaxed);
            it = storage_cache.erase(it);
        } else {
            ++it;
        }
    }
    stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
}

bool Context::Impl::StorageSuperseded(const Storage& s) const {
    // Only what the GPU wrote and what can take part in an overlap transfer is judged.
    if (s.origin != StorageOrigin::Gpu || s.gpu_write_gen == 0 || !s.image || s.sparse ||
        s.samples != VK_SAMPLE_COUNT_1_BIT || s.layout.size == 0) {
        return false;
    }
    const auto index_it = storage_index.find(s.pool_serial);
    if (index_it == storage_index.end()) {
        return false;
    }
    const PoolIndex& index = index_it->second;
    const u64 begin = s.pool_offset;
    const u64 end = begin + s.layout.size;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    thread_local std::vector<std::pair<u64, u64>> covers;
    covers.clear();
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        const Storage& other = *it->second;
        const u64 other_end = it->first + other.layout.size;
        if (&other == &s || other_end <= begin || !other.image || other.sparse ||
            other.samples != VK_SAMPLE_COUNT_1_BIT || other.origin != StorageOrigin::Gpu ||
            other.gpu_write_gen <= s.gpu_write_gen) {
            continue;
        }
        // A pair whose contents no transfer could carry (ResolveOverlaps logged it) did not take
        // this storage's texels over.
        if (alias_logged.contains({reinterpret_cast<u64>(&s), reinterpret_cast<u64>(&other)})) {
            continue;
        }
        covers.emplace_back((std::max)(begin, it->first), (std::min)(end, other_end));
    }
    return RangesCover(covers, begin, end);
}

void Context::Impl::TrimStorageCache(bool at_present, u64 retire_seq) {
    // A pass walks the whole cache: it runs when the cache has grown since the last one or after
    // a few presents, not at every storage created.
    constexpr u64 PASS_INTERVAL_PRESENTS = 120;
    // Superseded-storage tests per pass (each scans the storages over the tested one's bytes).
    constexpr size_t MAX_SUPERSEDED_TESTS = 1024;
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    const bool over_budget = storage_cache_bytes > storage_cache_budget;
    if (trim_legacy) {
        if (at_present || !over_budget ||
            (storage_cache_bytes < trim_floor_bytes && presents < trim_next_present)) {
            return;
        }
    } else if (at_present) {
        // The periodic pass: textures died, storages aged, the device budget moved. Strict
        // device-local allocations over the budget (EDEN_NX2_TRY_VMA_DEVICE_STRICT) bring it
        // forward: the driver pages until the cache sheds what it can rebuild.
        const u64 strict_over_budget = spills.over_budget_count.load(std::memory_order_relaxed);
        if (presents < trim_next_present && strict_over_budget == trim_seen_over_budget) {
            return;
        }
        trim_seen_over_budget = strict_over_budget;
    } else if ((!over_budget && trim_device_excess == 0) ||
               (storage_cache_bytes < trim_floor_bytes && presents < trim_next_present)) {
        return;
    }
    // The device near its memory budget (85% soft threshold for MKW, 90% for other titles;
    // VK_EXT_memory_budget, the allocator's empty block space counts and the driver counts it too)
    // sheds rebuildable storages before spills begin.
    u64 device_budget = 0;
    trim_device_excess = 0;
    bool hard_pressure = false;
    if (!trim_legacy) {
        device_budget = device.GetDeviceMemoryBudget();
        if (device_budget != 0) {
            const u64 used = device.GetDeviceMemoryUsage() + device.GetDeviceMemorySlack();
            const u64 pressure_threshold = mario_kart_world
                                               ? device_budget - (device_budget * 15) / 100
                                               : device_budget - device_budget / 10;
            hard_pressure = mario_kart_world && used > device_budget;
            if (used > pressure_threshold) {
                trim_device_excess = used - pressure_threshold;
                if (!(!at_present && mario_kart_world && !over_budget && !hard_pressure)) {
                    stats.storage_trim_pressure_passes.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    if (!at_present && mario_kart_world && !over_budget && !hard_pressure) {
        return;
    }
    if (trim_device_excess != 0 && image_recycle_count != 0) {
        const u64 freed = image_recycle_bytes;
        ClearRecycledImages();
        trim_device_excess -= (std::min)(trim_device_excess, freed);
    }
    constexpr size_t MAX_SOFT_PRESSURE_OPS_PER_PASS = 48;
    constexpr u64 SOFT_PRESSURE_INTERVAL_PRESENTS = 4;
    const bool soft_pressure_capped =
        mario_kart_world && trim_device_excess != 0 && !hard_pressure;
    size_t pressure_ops = 0;
    const u64 budget_target = storage_cache_budget / 10 * 9;
    u64 target = over_budget ? budget_target : storage_cache_bytes;
    if (trim_device_excess != 0) {
        // The excess and a margin, so that the next pass does not find the device over at once.
        const u64 shed = trim_device_excess + device_budget / 32;
        target = (std::min)(target, storage_cache_bytes - (std::min)(storage_cache_bytes, shed));
    }
    const bool trim_for_budget = storage_cache_bytes > target;
    const bool trim_for_age = !trim_legacy && trim_age_presents != 0;
    const auto aged = [&](const Storage& storage) {
        return presents >= storage.last_used_present &&
               presents - storage.last_used_present >= trim_age_presents;
    };
    constexpr VkImageUsageFlags TARGET_USAGE =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    struct Candidate {
        u64 last_used;
        StorageKey key;
    };
    std::vector<Candidate> gpu_written;
    std::vector<Candidate> from_guest;
    for (const auto& [key, storage] : storage_cache) {
        // Only the cache holds it: no texture and no submission in flight uses it.
        if (storage.use_count() != 1 || !storage->image || storage->sparse) {
            continue;
        }
        const bool gpu = storage->origin == StorageOrigin::Gpu;
        if (!trim_for_budget &&
            (!trim_for_age || !aged(*storage) || (!gpu && (storage->usage & TARGET_USAGE) == 0))) {
            continue;
        }
        if (gpu) {
            gpu_written.push_back({storage->last_used_seq, key});
        } else {
            from_guest.push_back({storage->last_used_seq, key});
        }
    }
    const auto by_age = [](const Candidate& a, const Candidate& b) { return a.last_used < b.last_used; };
    enum class Reason { Budget, Superseded, Aged };
    const auto evict = [&](const StorageKey& key, Reason reason) {
        const auto it = storage_cache.find(key);
        if (it == storage_cache.end()) {
            return;
        }
        Storage& storage = *it->second;
        if (reason != Reason::Aged && storage_cache_bytes <= budget_target) {
            stats.storages_pressure_evicted.fetch_add(1, std::memory_order_relaxed);
        }
        storage_cache_bytes -= (std::min)(storage_cache_bytes, storage.vram_bytes);
        if (const auto index = storage_index.find(storage.pool_serial); index != storage_index.end()) {
            auto range = index->second.spans.equal_range(storage.pool_offset);
            for (auto s = range.first; s != range.second; ++s) {
                if (s->second == &storage) {
                    index->second.spans.erase(s);
                    break;
                }
            }
        }
        storage.cached = false;
        stats.storages_evicted.fetch_add(1, std::memory_order_relaxed);
        if (reason == Reason::Superseded) {
            stats.storages_superseded_evicted.fetch_add(1, std::memory_order_relaxed);
        } else if (reason == Reason::Aged) {
            stats.storages_aged_evicted.fetch_add(1, std::memory_order_relaxed);
        }
        if (trim_device_excess == 0 && !storage.sparse && storage.image &&
            it->second.use_count() == 1 &&
            storage.last_used_seq <= completed_seq.load(std::memory_order_acquire)) {
            RecycleImage(storage);
        }
        storage_cache.erase(it);
        ++pressure_ops;
    };
    size_t tests = 0;
    // Whether a later GPU write replaced every byte of a GPU-written storage (null: the pass is
    // out of tests, or nothing was written in its pool since it was found to be live).
    const auto superseded = [&](Storage& storage) -> std::optional<bool> {
        const auto index = storage_index.find(storage.pool_serial);
        const u64 gen = index != storage_index.end() ? index->second.write_gen : 0;
        if (storage.trim_checked_gen == gen || tests >= MAX_SUPERSEDED_TESTS) {
            return std::nullopt;
        }
        ++tests;
        if (StorageSuperseded(storage)) {
            return true;
        }
        storage.trim_checked_gen = gen;
        return false;
    };
    // Unless under hard device-memory pressure (used > device_budget), storages used in the last
    // few presents stay: transient/aliased targets a game re-uses every frame would otherwise be
    // evicted, recreated, and alias-transferred every frame.
    constexpr u64 PRESSURE_KEEP_PRESENTS = 4;
    const auto kept_recently_used = [&](const Storage& storage) {
        return !hard_pressure && presents >= storage.last_used_present &&
               presents - storage.last_used_present < PRESSURE_KEEP_PRESENTS;
    };
    // First the render targets the GPU wrote again since (the game re-uses the bytes for other
    // shapes; each shape keeps an image of its own): their texels live in the newer storages, so
    // they cost nothing but the image. Storages the GPU wrote that nothing replaced hold data no
    // guest bytes can rebuild: they are kept.
    std::sort(gpu_written.begin(), gpu_written.end(), by_age);
    for (const Candidate& candidate : gpu_written) {
        if (storage_cache_bytes <= target || tests >= MAX_SUPERSEDED_TESTS ||
            (soft_pressure_capped && pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS)) {
            break;
        }
        const auto it = storage_cache.find(candidate.key);
        if (it != storage_cache.end() && !kept_recently_used(*it->second) &&
            superseded(*it->second).value_or(false)) {
            evict(candidate.key, Reason::Superseded);
        }
    }
    // Then the storages that hold guest bytes (the next use uploads them again).
    std::sort(from_guest.begin(), from_guest.end(), by_age);
    for (const Candidate& candidate : from_guest) {
        if (storage_cache_bytes <= target ||
            (soft_pressure_capped && pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS)) {
            break;
        }
        const auto it = storage_cache.find(candidate.key);
        if (it != storage_cache.end() && !kept_recently_used(*it->second)) {
            evict(candidate.key, Reason::Budget);
        }
    }
    // If live CPU-sourced (guest memory) textures keep the cache over target (e.g., MKW keeping
    // thousands of Texture objects registered), demote the oldest idle images at present boundaries
    // while keeping the Storage entry intact so EnsureImage can rebuild them on future use.
    if (at_present && storage_cache_bytes > target && retire_seq != 0 &&
        !(soft_pressure_capped && pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS)) {
        const u64 done_seq = completed_seq.load(std::memory_order_acquire);
        constexpr u64 DEMOTE_KEEP_PRESENTS_PRESSURE = 60;
        constexpr u64 DEMOTE_KEEP_PRESENTS_BUDGET = 180;
        const u64 demote_keep =
            trim_device_excess != 0 ? DEMOTE_KEEP_PRESENTS_PRESSURE : DEMOTE_KEEP_PRESENTS_BUDGET;
        std::vector<Candidate> demotable;
        for (const auto& [key, storage] : storage_cache) {
            if (storage.use_count() <= 1 || !storage->image || storage->sparse || storage->scaled ||
                storage->origin != StorageOrigin::Cpu || storage->last_used_seq > done_seq) {
                continue;
            }
            if (presents >= storage->last_used_present &&
                presents - storage->last_used_present >= demote_keep) {
                demotable.push_back({storage->last_used_seq, key});
            }
        }
        std::sort(demotable.begin(), demotable.end(), by_age);
        for (const Candidate& candidate : demotable) {
            if (storage_cache_bytes <= target ||
                (soft_pressure_capped && pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS)) {
                break;
            }
            const auto it = storage_cache.find(candidate.key);
            if (it == storage_cache.end() || !it->second->image) {
                continue;
            }
            Storage& s = *it->second;
            auto old_image = std::make_shared<vk::Image>(std::move(s.image));
            auto old_views = std::make_shared<std::vector<vk::ImageView>>(
                draw_resources ? draw_resources->TakeStorageViews(s)
                               : std::vector<vk::ImageView>{});
            retired.push_back({retire_seq, [old_image, old_views]() mutable {
                                   old_views->clear();
                                   *old_image = vk::Image{};
                               }});
            s.image = vk::Image{};
            s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            s.contents_valid = false;
            s.origin = StorageOrigin::None;
            ++s.image_gen;
            if (trim_device_excess != 0 || storage_cache_bytes <= budget_target) {
                stats.storages_pressure_evicted.fetch_add(1, std::memory_order_relaxed);
            }
            storage_cache_bytes -= (std::min)(storage_cache_bytes, s.vram_bytes);
            stats.storages_evicted.fetch_add(1, std::memory_order_relaxed);
            ++pressure_ops;
        }
    }
    // Under the budget too: render targets nothing used for `trim_age_presents` that cost nothing
    // to rebuild. Ones that hold guest bytes (or nothing) upload them again; GPU-written ones go
    // only when superseded, never with texels no other storage holds.
    if (trim_for_age) {
        for (const Candidate& candidate : from_guest) {
            if (soft_pressure_capped && pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS) {
                break;
            }
            const auto it = storage_cache.find(candidate.key);
            if (it != storage_cache.end() && (it->second->usage & TARGET_USAGE) != 0 &&
                aged(*it->second)) {
                evict(candidate.key, Reason::Aged);
            }
        }
        for (const Candidate& candidate : gpu_written) {
            if (tests >= MAX_SUPERSEDED_TESTS ||
                (soft_pressure_capped && pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS)) {
                break;
            }
            const auto it = storage_cache.find(candidate.key);
            if (it != storage_cache.end() && aged(*it->second) &&
                superseded(*it->second).value_or(false)) {
                stats.storages_superseded_evicted.fetch_add(1, std::memory_order_relaxed);
                evict(candidate.key, Reason::Aged);
            }
        }
    }
    const bool soft_pressure_incomplete = soft_pressure_capped &&
                                          pressure_ops >= MAX_SOFT_PRESSURE_OPS_PER_PASS &&
                                          storage_cache_bytes > target;
    trim_floor_bytes = storage_cache_bytes + (std::max)(storage_cache_budget / 32, u64{1});
    trim_next_present =
        presents + (soft_pressure_incomplete ? SOFT_PRESSURE_INTERVAL_PRESENTS
                                             : PASS_INTERVAL_PRESENTS);
    stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
}

std::unordered_map<const Storage*, VC::TextureHolder> Context::Impl::StorageHolders() const {
    std::unordered_map<const Storage*, VC::TextureHolder> holders;
    const auto holder_of = [&](u64 serial) {
        const auto it = texture_holders.find(serial);
        return it != texture_holders.end() ? it->second : VC::TextureHolder::Game;
    };
    const auto note = [&](const Storage* storage, VC::TextureHolder holder) {
        const auto [it, inserted] = holders.try_emplace(storage, holder);
        if (!inserted && holder > it->second) {
            it->second = holder;
        }
    };
    for (const auto& [serial, texture] : textures) {
        if (texture && texture->storage) {
            note(texture->storage.get(), holder_of(serial));
        }
    }
    // A render-target alias goes with the texture it was made for.
    for (const auto& [key, texture] : target_aliases) {
        if (texture && texture->storage) {
            note(texture->storage.get(), holder_of(std::get<0>(key)));
        }
    }
    return holders;
}

void Context::Impl::NoteTextureHoldersNow(VC::TextureHolderReport& report) {
    texture_holders.clear();
    texture_holders.reserve(report.serials.size());
    for (const auto& [serial, holder] : report.serials) {
        texture_holders.emplace(serial, holder);
    }
    texture_holders_present = stats.presents.load(std::memory_order_relaxed);
    std::array<u64, VC::TEXTURE_HOLDER_KINDS> count{};
    std::array<u64, VC::TEXTURE_HOLDER_KINDS> bytes{};
    for (const auto& [storage, holder] : StorageHolders()) {
        if (!storage->image) {
            continue;
        }
        const size_t i = static_cast<size_t>(holder);
        ++count[i];
        bytes[i] += storage->vram_bytes;
        if (storage->scaled && storage->scaled->image) {
            bytes[i] += storage->scaled->vram_bytes;
        }
    }
    for (size_t i = 0; i < VC::TEXTURE_HOLDER_KINDS; ++i) {
        holder_storages[i].store(count[i], std::memory_order_relaxed);
        holder_storage_bytes[i].store(bytes[i], std::memory_order_relaxed);
    }
    holder_report_seen.store(true, std::memory_order_release);
}

void Context::Impl::LogStorageCensus() {
    std::unordered_set<const Storage*> seen;
    struct Kind {
        u64 count{};
        u64 bytes{};
    };
    Kind sparse, targets, depth, astc, other, gpu_written;
    std::array<Kind, ASTC_STANDIN_COUNT> astc_bc{}; // by AstcStandin
    u64 sparse_bound = 0;
    std::vector<const Storage*> largest;
    const auto add = [&](const Storage& s) {
        if (!s.image || !seen.insert(&s).second) {
            return;
        }
        Kind& kind = s.sparse ? sparse
                     : s.host.depth ? depth
                     : (s.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0 ? targets
                     : s.host.needs_transcode ? astc
                                              : other;
        ++kind.count;
        kind.bytes += s.vram_bytes;
        if (s.host.bc_standin) {
            Kind& bc = astc_bc[static_cast<size_t>(BcStandinOf(s.host))];
            ++bc.count;
            bc.bytes += s.vram_bytes;
        }
        if (s.sparse) {
            sparse_bound += s.sparse->BoundBytes();
        }
        if (s.origin == StorageOrigin::Gpu) {
            ++gpu_written.count;
            gpu_written.bytes += s.vram_bytes;
        }
        largest.push_back(&s);
    };
    for (const auto& [serial, texture] : textures) {
        if (texture && texture->storage) {
            add(*texture->storage);
        }
    }
    for (const auto& [key, storage] : storage_cache) {
        add(*storage);
    }
    const size_t top = (std::min)(largest.size(), size_t{8});
    std::partial_sort(largest.begin(), largest.begin() + top, largest.end(),
                      [](const Storage* a, const Storage* b) { return a->vram_bytes > b->vram_bytes; });
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    // The BC stand-ins by format ("as BC3 <MiB> (<n>)", one entry per format held), else what
    // new ones become.
    std::string as_bc;
    for (const AstcStandin bc : {AstcStandin::Bc1, AstcStandin::Bc3, AstcStandin::Bc4,
                                 AstcStandin::Bc5, AstcStandin::Bc7}) {
        const Kind& held = astc_bc[static_cast<size_t>(bc)];
        if (held.count != 0) {
            as_bc += fmt::format("{}as {} {} ({})", as_bc.empty() ? "" : ", ", AstcStandinName(bc),
                                 mib(held.bytes), held.count);
        }
    }
    if (as_bc.empty()) {
        as_bc = fmt::format("as {} 0 (0)", AstcStandinName(astc_standin));
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] storage census: {} images, estimated MiB (count): sparse {} ({}, bound {}), depth {} "
             "({}), colour targets {} ({}), compressed stand-ins {} ({}; {}), other {} "
             "({}); GPU-written {} ({}); cache {} MiB; sparse pages bound/reserved {}/{} MiB",
             seen.size(), mib(sparse.bytes), sparse.count, mib(sparse_bound), mib(depth.bytes),
             depth.count, mib(targets.bytes), targets.count, mib(astc.bytes), astc.count, as_bc,
             mib(other.bytes), other.count, mib(gpu_written.bytes), gpu_written.count,
             mib(storage_cache_bytes), sparse_pages ? mib(sparse_pages->UsedBytes()) : 0,
             sparse_pages ? mib(sparse_pages->ReservedBytes()) : 0);
    if (smart_bc.enabled) {
        const auto total = [](const std::atomic<u64>& value) {
            return value.load(std::memory_order_relaxed);
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] smart BC so far: {} scans (L {}, LA {}, RGB {}, RGBA {}, all zero {}), "
                 "{:.1f} MiB of ASTC in {:.1f} ms; {} re-classified to BC3, {} kept at BC3 by "
                 "copies",
                 total(smart_total.classes[0]) + total(smart_total.classes[1]) +
                     total(smart_total.classes[2]) + total(smart_total.classes[3]) +
                     total(smart_total.classes[4]),
                 total(smart_total.classes[0]), total(smart_total.classes[1]),
                 total(smart_total.classes[2]), total(smart_total.classes[3]),
                 total(smart_total.classes[4]),
                 static_cast<double>(total(smart_total.scan_bytes)) / 1048576.0,
                 static_cast<double>(total(smart_total.scan_ns)) / 1e6, total(smart_total.refits),
                 total(smart_total.pinned));
    }
    {
        // The GPU-written images only the cache holds (no texture, no submission in flight), and
        // how many of the first ones a later GPU write replaced (TrimStorageCache drops those
        // first when the cache is over budget).
        constexpr u64 TESTED_LIMIT = 1024;
        u64 idle = 0, idle_bytes = 0, tested = 0, superseded = 0, superseded_bytes = 0;
        for (const auto& [key, storage] : storage_cache) {
            if (storage.use_count() != 1 || storage->origin != StorageOrigin::Gpu ||
                !storage->image || storage->sparse) {
                continue;
            }
            ++idle;
            idle_bytes += storage->vram_bytes;
            if (tested < TESTED_LIMIT) {
                ++tested;
                if (StorageSuperseded(*storage)) {
                    ++superseded;
                    superseded_bytes += storage->vram_bytes;
                }
            }
        }
        size_t most_spans = 0;
        u64 widest = 0;
        for (const auto& [serial, index] : storage_index) {
            most_spans = (std::max)(most_spans, index.spans.size());
            widest = (std::max)(widest, index.max_size);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: idle GPU-written {} ({} MiB), of the first {} tested {} "
                 "superseded ({} MiB); budget {} MiB, evicted {} ({} superseded); most storages in "
                 "one pool {}, widest {} MiB",
                 idle, mib(idle_bytes), tested, superseded, mib(superseded_bytes),
                 mib(storage_cache_budget), stats.storages_evicted.load(std::memory_order_relaxed),
                 stats.storages_superseded_evicted.load(std::memory_order_relaxed), most_spans,
                 mib(widest));
    }
    {
        // Why the cache cannot reach its budget: who holds the cached images (only the cache
        // holds the ones a trim may drop), where their bytes came from, how long ago the render
        // targets were used, and the twins of the Resolution setting (not in the cache's count).
        std::unordered_set<const Storage*> live;
        for (const auto& [serial, texture] : textures) {
            if (texture && texture->storage) {
                live.insert(texture->storage.get());
            }
        }
        for (const auto& [key, texture] : target_aliases) {
            if (texture && texture->storage) {
                live.insert(texture->storage.get());
            }
        }
        Kind held_live, held_other, idle_gpu, idle_rebuild, twins;
        std::array<Kind, 3> origin{};
        // Presents since the last use: < 60, < 300, < 1800, older.
        constexpr std::array<u64, 3> AGE_LIMITS{60, 300, 1800};
        std::array<Kind, 4> target_age{};
        std::array<Kind, 4> live_target_age{};
        std::array<Kind, 4> idle_gpu_target_age{};
        // The render targets textures hold, by the strongest holder of their storage (the
        // recorder's last report: the game's texture, one waiting for its submissions, one only
        // recorded lists keep, one the game re-initialised without a finalize).
        const std::unordered_map<const Storage*, VC::TextureHolder> holders = StorageHolders();
        std::array<std::array<Kind, 4>, VC::TEXTURE_HOLDER_KINDS> holder_target_age{};
        const u64 presents = stats.presents.load(std::memory_order_relaxed);
        const auto age_bucket = [&](const Storage& s) {
            const u64 age = presents > s.last_used_present ? presents - s.last_used_present : 0;
            size_t bucket = 0;
            while (bucket < AGE_LIMITS.size() && age >= AGE_LIMITS[bucket]) {
                ++bucket;
            }
            return bucket;
        };
        const auto count = [](Kind& kind, const Storage& s) {
            ++kind.count;
            kind.bytes += s.vram_bytes;
        };
        for (const Storage* s : seen) {
            if (s->scaled && s->scaled->image) {
                count(twins, *s->scaled);
            }
        }
        for (const auto& [key, storage] : storage_cache) {
            const Storage& s = *storage;
            if (!s.image || s.sparse) {
                continue;
            }
            count(origin[static_cast<size_t>(s.origin)], s);
            const bool idle = storage.use_count() == 1;
            if (live.contains(&s)) {
                count(held_live, s);
            } else if (!idle) {
                count(held_other, s);
            } else if (s.origin == StorageOrigin::Gpu) {
                count(idle_gpu, s);
            } else {
                count(idle_rebuild, s);
            }
            if ((s.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) == 0) {
                continue;
            }
            const size_t bucket = age_bucket(s);
            count(target_age[bucket], s);
            if (live.contains(&s)) {
                count(live_target_age[bucket], s);
                const auto holder = holders.find(&s);
                count(holder_target_age[static_cast<size_t>(
                          holder != holders.end() ? holder->second : VC::TextureHolder::Game)]
                                       [bucket],
                      s);
            } else if (idle && s.origin == StorageOrigin::Gpu) {
                count(idle_gpu_target_age[bucket], s);
            }
        }
        const auto ages = [&](const std::array<Kind, 4>& kinds) {
            return fmt::format("<60 {} ({}), <300 {} ({}), <1800 {} ({}), older {} ({})",
                               mib(kinds[0].bytes), kinds[0].count, mib(kinds[1].bytes),
                               kinds[1].count, mib(kinds[2].bytes), kinds[2].count,
                               mib(kinds[3].bytes), kinds[3].count);
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: cache holders MiB (count): live textures {} ({}), other "
                 "holders {} ({}), idle GPU-written {} ({}), idle rebuildable {} ({}); origin "
                 "none/cpu/gpu {}/{}/{} MiB; scaled twins {} ({}); {} live textures",
                 mib(held_live.bytes), held_live.count, mib(held_other.bytes), held_other.count,
                 mib(idle_gpu.bytes), idle_gpu.count, mib(idle_rebuild.bytes), idle_rebuild.count,
                 mib(origin[0].bytes), mib(origin[1].bytes), mib(origin[2].bytes),
                 mib(twins.bytes), twins.count, textures.size());
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: render targets by presents since use MiB (count): {}; of "
                 "them held by live textures: {}; idle GPU-written: {}; trim {} age {} evicted "
                 "aged {} pressure {}, device over budget at the last pass {} MiB ({} passes)",
                 ages(target_age), ages(live_target_age), ages(idle_gpu_target_age),
                 trim_legacy ? "legacy" : "current", trim_age_presents,
                 stats.storages_aged_evicted.load(std::memory_order_relaxed),
                 stats.storages_pressure_evicted.load(std::memory_order_relaxed),
                 mib(trim_device_excess),
                 stats.storage_trim_pressure_passes.load(std::memory_order_relaxed));
        const auto by_holder = [&](VC::TextureHolder holder) {
            return ages(holder_target_age[static_cast<size_t>(holder)]);
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: render targets held by textures, by holder ({}), MiB "
                 "(count): game textures: {}; retiring: {}; guest lists: {}; synthetic lists: {}; "
                 "re-initialised: {}",
                 holder_report_seen.load(std::memory_order_relaxed)
                     ? fmt::format("recorder report of present {}", texture_holders_present)
                     : std::string{"no recorder report: all counted as game textures"},
                 by_holder(VC::TextureHolder::Game), by_holder(VC::TextureHolder::Retiring),
                 by_holder(VC::TextureHolder::GuestList),
                 by_holder(VC::TextureHolder::SyntheticList),
                 by_holder(VC::TextureHolder::Reinitialized));
    }
    for (size_t i = 0; i < top; ++i) {
        const Storage& s = *largest[i];
        LOG_INFO(Render_Vulkan, "[nvn] storage census: #{} {}x{}x{} lv {} fmt {:#x} flags {:#x} {} MiB{}{}{}{}",
                 i + 1, s.extent.width, s.extent.height, s.layers, s.levels, s.desc.format,
                 s.desc.flags, mib(s.vram_bytes), s.sparse ? " sparse" : "",
                 s.host.needs_transcode ? " stand-in" : "",
                 s.host.bc_standin ? fmt::format(" {}", AstcStandinName(BcStandinOf(s.host)))
                                   : std::string{},
                 s.origin == StorageOrigin::Gpu ? " gpu-written" : "");
    }
}

void Context::Impl::FlushTextureNow(CmdFlushTexture& command) {
    const auto finish = [&](bool ok) { command.done->set_value(ok); };
    const auto texture_it = textures.find(command.serial);
    if (texture_it == textures.end() || !texture_it->second->storage) {
        finish(false);
        return;
    }
    Storage& s = *texture_it->second->storage;
    // The game reads a texture the Resolution setting scaled: the guest-sized image first.
    if (s.origin == StorageOrigin::Gpu && Rescale::NativeIsStale(s.scale_state) &&
        !DownscaleNow(s)) {
        finish(false);
        return;
    }
    if (s.origin != StorageOrigin::Gpu || !s.image) {
        finish(true); // the guest bytes are what the texture holds
        return;
    }
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end() || (pool_it->second->flags & PoolFlag::CpuNoAccess) != 0 ||
        pool_it->second->IsVirtual()) {
        finish(false);
        return;
    }
    Pool& pool = *pool_it->second;
    const bool plain = !s.host.needs_transcode && !s.host.guest_layout_differs &&
                       s.samples == VK_SAMPLE_COUNT_1_BIT && !s.host.depth && !s.host.stencil;
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!plain || !block.bytes || s.current_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        static Common::BootCounter logged;
        if (logged.First()) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] ReadTexels of a GPU-written texture (format {:#x}): readback of "
                        "depth/stencil, multisampled or compressed storage is not supported",
                        s.desc.format);
        }
        finish(false);
        return;
    }
    // One staging buffer for every layer and level, copied out in one submission.
    const bool linear = IsLinearStorage(s.desc);
    struct Piece {
        u32 level;
        u32 layer;
        u32 cols, rows, slices;
        u32 width, height;
        u64 buffer_offset;
        u64 guest_offset;
        u32 tile_h, tile_d;
    };
    std::vector<Piece> pieces;
    u64 total = 0;
    const u32 layer_count = linear ? 1 : s.layers;
    const u32 level_count = linear ? 1 : s.levels;
    // Region offsets: a multiple of the texel block size too (12-byte formats).
    const u64 region_alignment = std::lcm(u64{64}, u64{block.bytes});
    for (u32 layer = 0; layer < layer_count; ++layer) {
        for (u32 level = 0; level < level_count; ++level) {
            Piece piece{};
            piece.level = level;
            piece.layer = layer;
            piece.width = Mip(s.extent.width, level);
            piece.height = Mip(s.extent.height, level);
            piece.slices = s.type == VK_IMAGE_TYPE_3D ? Mip(s.extent.depth, level) : 1u;
            piece.cols = (piece.width + block.width - 1) / block.width;
            piece.rows = (piece.height + block.height - 1) / block.height;
            piece.guest_offset = static_cast<u64>(layer) * s.layout.layer_size +
                                 (linear ? 0 : TextureLevelOffset(s.desc, level));
            TextureLevelTiles(s.desc, level, piece.tile_h, piece.tile_d);
            total = (total + region_alignment - 1) / region_alignment * region_alignment;
            piece.buffer_offset = total;
            total += static_cast<u64>(piece.cols) * piece.rows * piece.slices * block.bytes;
            pieces.push_back(piece);
        }
    }
    vk::Buffer download;
    try {
        download = CreateHostBuffer(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] ReadTexels readback buffer failed: {}", exception.what());
        finish(false);
        return;
    }
    const VkImageLayout previous_layout = s.current_layout;
    try {
        ImmediateSubmit([&](vk::CommandBuffer cmd) {
            const auto barrier = [&](VkImageLayout from, VkImageLayout to) {
                const VkImageMemoryBarrier image_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .oldLayout = from,
                    .newLayout = to,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *s.image,
                    .subresourceRange{
                        .aspectMask = s.aspect,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                };
                const VkMemoryBarrier memory{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, memory, {},
                                    image_barrier);
            };
            barrier(previous_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            std::vector<VkBufferImageCopy> regions;
            for (const Piece& piece : pieces) {
                regions.push_back({
                    .bufferOffset = piece.buffer_offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = piece.level,
                        .baseArrayLayer = piece.layer,
                        .layerCount = 1,
                    },
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {piece.width, piece.height, piece.slices},
                });
            }
            cmd.CopyImageToBuffer(*s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *download,
                                  regions);
            barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, previous_layout);
            const VkMemoryBarrier host{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                host, {}, {});
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] ReadTexels readback failed: {}", exception.what());
        finish(false);
        return;
    }
    // Swizzle into guest memory (the whole storage is read-modify-written: GOB padding stays). A
    // storage that leaves its pool (the driver does not check) only has its inside written.
    if (s.pool_offset >= pool.size) {
        finish(false);
        return;
    }
    const u64 storage_size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    std::vector<u8> scratch;
    u8* storage_bytes = pool.host ? pool.host + s.pool_offset : nullptr;
    if (!storage_bytes) {
        scratch.resize(static_cast<size_t>(storage_size));
        if (!memory || !memory->Read(pool.address + s.pool_offset, scratch.data(), scratch.size())) {
            finish(false);
            return;
        }
        storage_bytes = scratch.data();
    }
    download.Invalidate();
    const std::span<const u8> mapped = download.Mapped();
    for (const Piece& piece : pieces) {
        const u64 linear_size = static_cast<u64>(piece.cols) * piece.rows * piece.slices * block.bytes;
        if (piece.guest_offset >= storage_size || piece.buffer_offset + linear_size > mapped.size()) {
            continue;
        }
        const std::span<const u8> source{mapped.data() + piece.buffer_offset,
                                         static_cast<size_t>(linear_size)};
        if (linear) {
            const u64 pitch = s.layout.layer_size / (std::max)(piece.height, 1u);
            const u64 row_bytes = static_cast<u64>(piece.cols) * block.bytes;
            for (u32 row = 0; row < piece.rows; ++row) {
                const u64 destination = piece.guest_offset + row * pitch;
                if (destination + row_bytes > storage_size) {
                    break;
                }
                std::memcpy(storage_bytes + destination, source.data() + row * row_bytes,
                            static_cast<size_t>(row_bytes));
            }
        } else {
            const u64 needed = Tegra::Texture::CalculateSize(true, block.bytes, piece.cols,
                                                             piece.rows, piece.slices,
                                                             piece.tile_h, piece.tile_d);
            const u64 available = storage_size - piece.guest_offset;
            std::vector<u8> padded;
            u8* target = storage_bytes + piece.guest_offset;
            if (available < needed) {
                padded.assign(static_cast<size_t>(needed), 0);
                std::memcpy(padded.data(), target, static_cast<size_t>(available));
                target = padded.data();
            }
            Tegra::Texture::SwizzleTexture(std::span<u8>{target, static_cast<size_t>(needed)},
                                           source, block.bytes, piece.cols, piece.rows,
                                           piece.slices, piece.tile_h, piece.tile_d, 0, true);
            if (!padded.empty()) {
                std::memcpy(storage_bytes + piece.guest_offset, padded.data(),
                            static_cast<size_t>(available));
            }
        }
    }
    if (!scratch.empty() && memory &&
        !memory->Write(pool.address + s.pool_offset, scratch.data(), scratch.size())) {
        finish(false);
        return;
    }
    // An untracked write: a mirror pool hashes these pages again at their next use.
    if (!pool.mirror_pages.empty()) {
        const u64 end = s.pool_offset + storage_size;
        for (u64 page = s.pool_offset / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < end; ++page) {
            pool.mirror_pages[static_cast<size_t>(page)].trusted = false;
        }
    }
    // The guest bytes are the texture's contents now: from here on it is a CPU-sourced storage.
    s.origin = StorageOrigin::Cpu;
    s.linear_writeback_seq = 0;
    s.guest_hash = HashGuestBytes(storage_bytes, storage_size);
    s.sample_hash = SampleGuestBytes(storage_bytes, storage_size, mario_kart_world);
    s.cpu_trusted_gen = 0;
    s.page_hashes.clear();
    s.unchanged_checks = 0;
    s.next_check_tick = 0;
    s.next_full_tick = PollTick() + FULL_VERIFY_PRESENTS;
    stats.texels_flushed.fetch_add(1, std::memory_order_relaxed);
    finish(true);
}

} // namespace Vulkan::Nvn
