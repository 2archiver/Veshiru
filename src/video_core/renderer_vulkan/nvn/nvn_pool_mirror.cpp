// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_pool_mirror.h"

namespace Vulkan::Nvn {

MirrorPages::MirrorPages(u64 pool_size)
    : valid((pool_size + PAGE * 64 - 1) / (PAGE * 64), 0), size{pool_size},
      pages{(pool_size + PAGE - 1) / PAGE} {}

bool MirrorPages::PageSpan(u64 offset, u64 length, u64& first, u64& last) const {
    if (length == 0 || offset >= size) {
        return false;
    }
    first = offset / PAGE;
    last = ((std::min)(offset + length, size) - 1) / PAGE;
    return true;
}

bool MirrorPages::AllValid(u64 offset, u64 length) const {
    u64 first = 0;
    u64 last = 0;
    if (!PageSpan(offset, length, first, last)) {
        return false;
    }
    for (u64 page = first; page <= last;) {
        const u64 word = page / 64;
        const u64 bit = page % 64;
        const u64 count = (std::min)(64 - bit, last - page + 1);
        const u64 mask = (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
        if ((valid[word] & mask) != mask) {
            return false;
        }
        page += count;
    }
    return true;
}

void MirrorPages::Invalidate(u64 offset, u64 length) {
    u64 first = 0;
    u64 last = 0;
    if (!PageSpan(offset, length, first, last)) {
        return;
    }
    for (u64 page = first; page <= last;) {
        const u64 bit = page % 64;
        const u64 count = (std::min)(64 - bit, last - page + 1);
        const u64 mask = (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
        valid[page / 64] &= ~mask;
        page += count;
    }
}

void MirrorPages::InvalidateAll() {
    std::ranges::fill(valid, u64{0});
}

void MirrorPages::Set(u64 offset, u64 length) {
    u64 first = 0;
    u64 last = 0;
    if (!PageSpan(offset, length, first, last)) {
        return;
    }
    for (u64 page = first; page <= last;) {
        const u64 bit = page % 64;
        const u64 count = (std::min)(64 - bit, last - page + 1);
        const u64 mask = (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
        valid[page / 64] |= mask;
        page += count;
    }
}

void MirrorPages::TakeInvalidWithin(const MirrorPages& wanted,
                                    std::vector<std::pair<u64, u64>>& out) {
    const size_t words = (std::min)(valid.size(), wanted.valid.size());
    u64 run_begin = 0;
    bool in_run = false;
    const auto close = [&](u64 end_page) {
        out.emplace_back(run_begin * PAGE, (std::min)(end_page * PAGE, size));
        in_run = false;
    };
    for (size_t word = 0; word < words; ++word) {
        const u64 invalid = ~valid[word] & wanted.valid[word];
        if (invalid == 0 && !in_run) {
            continue;
        }
        valid[word] |= invalid;
        const u64 base = word * 64;
        if (invalid == ~u64{0} && in_run) {
            continue;
        }
        for (u64 bit = 0; bit < 64; ++bit) {
            const bool is_invalid = (invalid >> bit) & 1;
            if (is_invalid && !in_run) {
                run_begin = base + bit;
                in_run = true;
            } else if (!is_invalid && in_run) {
                close(base + bit);
            }
        }
    }
    if (in_run) {
        close(words * 64);
    }
}

void MirrorPages::TakeInvalid(u64 offset, u64 length, std::vector<std::pair<u64, u64>>& out) {
    u64 first = 0;
    u64 last = 0;
    if (!PageSpan(offset, length, first, last)) {
        return;
    }
    u64 run_begin = 0;
    bool in_run = false;
    const auto close = [&](u64 end_page) {
        out.emplace_back(run_begin * PAGE, (std::min)(end_page * PAGE, size));
        in_run = false;
    };
    for (u64 page = first; page <= last;) {
        const u64 word = page / 64;
        const u64 bit = page % 64;
        const u64 count = (std::min)(64 - bit, last - page + 1);
        const u64 mask = (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
        const u64 invalid = ~valid[word] & mask;
        if (invalid == mask) {
            // A whole stretch of invalid pages (the common first use).
            if (!in_run) {
                run_begin = page;
                in_run = true;
            }
        } else if (invalid == 0) {
            if (in_run) {
                close(page);
            }
        } else {
            for (u64 i = 0; i < count; ++i) {
                const bool is_invalid = (invalid >> (bit + i)) & 1;
                if (is_invalid && !in_run) {
                    run_begin = page + i;
                    in_run = true;
                } else if (!is_invalid && in_run) {
                    close(page + i);
                }
            }
        }
        valid[word] |= mask;
        page += count;
    }
    if (in_run) {
        close(last + 1);
    }
}

void Context::Impl::AttachStorageState(Pool& pool, PoolMirror& mirror) {
    mirror.storage = true;
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    const Debug::Config& debug = Debug::Get();
    if (debug.wb_monitor && !mirror.monitor) {
        mirror.monitor = std::make_shared<MirrorMonitor>();
        {
            std::scoped_lock lock{mirror_monitors_mutex};
            mirror.monitor->slot = static_cast<u32>(
                (std::min<size_t>)(mirror_monitors.size(), MIRROR_STAT_SLOTS - 1));
            mirror_monitors.push_back(mirror.monitor);
        }
        mirror.monitor->address = pool.address;
        mirror.monitor->size = pool.size;
    }
    if (debug.try_wb_lines != 0 && !mirror.line_guard && memory != nullptr &&
        memory->SupportsAccessTraps()) {
        mirror.line_guard = std::make_shared<MirrorLineGuard>(
            pool.size, line_guard_stats, replay_thread.get_id());
    }
    if (memory != nullptr && (mirror.monitor || mirror.line_guard)) {
        // Runs on the guest thread whose access hit an armed page, before the access lands.
        memory->SetAccessCallback(
            mirror.track_id, [monitor = mirror.monitor, guard = mirror.line_guard,
                              base = pool.address](u64 va, u64 size) {
                if (monitor) {
                    monitor->touch_events.fetch_add(1, std::memory_order_relaxed);
                    monitor->touched_pages.fetch_add(
                        (size + MirrorPages::PAGE - 1) / MirrorPages::PAGE,
                        std::memory_order_relaxed);
                }
                if (guard && va >= base) {
                    if (!guard->Touched(va - base, va - base + size)) {
                        LOG_WARNING(Render_Vulkan,
                                    "[nvn] pool {:#x}: a guest access waited for the write-back "
                                    "merge in flight and gave up; the pool merges exactly from now "
                                    "on",
                                    base);
                    }
                }
            });
    }
}

PoolMirror* Context::Impl::VramMirror(Pool& pool, bool create, bool storage) {
    // EDEN_NX2_TRY_VTXCOV_BIGPOOL: a read-only mirror of a pool over the TRY_STORAGE_VRAM limit
    // stays read-only, and the pool gets no storage mirror (the limit bounds what raw-global
    // dispatches copy in and snapshot); its storage ranges stay in guest memory, whose writes
    // (NotePoolWrite) make the read-only mirror's pages stale.
    const bool big_read_only = (vtxcov & VTXCOV_BIGPOOL) != 0 && try_storage_vram &&
                               pool.size > try_storage_vram_max_bytes;
    if (pool.vram_mirror) {
        if (storage && !pool.vram_mirror->storage && create && big_read_only) {
            return nullptr;
        }
        if (storage && !pool.vram_mirror->storage && create) {
            // A read-only mirror takes its first written storage range: from here on it is a
            // storage mirror (raw table, raw-writing snapshots, write-back merges). Its valid
            // pages stay valid: raw writes made them stale while it was read-only.
            AttachStorageState(pool, *pool.vram_mirror);
            pool_generation.fetch_add(1, std::memory_order_release);
            ++ro_alias_generation; // the mirror alias table's target changed
            stats.vram_mirrors_promoted.fetch_add(1, std::memory_order_relaxed);
        }
        return pool.vram_mirror.get();
    }
    if (!create || pool.vram_mirror_refused || pool_mirror_mode == 0 ||
        pool.backing != Pool::Backing::Imported || pool.size == 0 ||
        pool.address % MirrorPages::PAGE != 0) {
        return pool.vram_mirror.get();
    }
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    const Debug::Config& debug = Debug::Get();
    const auto refuse = [&](const char* why) -> PoolMirror* {
        pool.vram_mirror_refused = true;
        stats.vram_mirror_refused.fetch_add(1, std::memory_order_relaxed);
        LOG_DEBUG(Render_Vulkan, "[nvn] pool {:#x} ({:#x} bytes): no VRAM mirror: {}",
                  pool.address, pool.size, why);
        return nullptr;
    };
    if ((vtxcov & VTXCOV_BIGPOOL) != 0 && pool.size > try_storage_vram_max_bytes) {
        // Read-only mirrors of any size (the budget below still holds); a storage one is not
        // refused for good, so a later read-only use may still mirror the pool.
        if (storage && try_storage_vram) {
            return nullptr;
        }
    } else if (pool.size > try_storage_vram_max_bytes && (try_storage_vram || !storage)) {
        if (!storage && !try_storage_vram) {
            return nullptr; // only a read-only mirror is too big; a storage one is not limited
        }
        return refuse("TRY_STORAGE_VRAM pool-size limit");
    }
    // Read-only mirrors leave half of the budget to the storage ones (and are not refused for
    // good: a written storage range may still make the pool's mirror).
    if (!storage && vram_mirror_bytes + pool.size > debug.pool_mirror_max_bytes / 2) {
        return nullptr;
    }
    if (vram_mirror_bytes + pool.size > debug.pool_mirror_max_bytes) {
        if (!vram_mirror_budget_logged) {
            vram_mirror_budget_logged = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] pool mirrors: budget of {} MiB reached ({} MiB in {} pools); pool "
                        "{:#x} ({:#x} bytes) and later ones stay in guest memory "
                        "(EDEN_NX2_NVN_POOL_MIRROR_MAX_MIB)",
                        debug.pool_mirror_max_bytes >> 20, vram_mirror_bytes >> 20,
                        vram_mirrored.size(), pool.address, pool.size);
        }
        return refuse("budget");
    }
    const u64 track_id = memory != nullptr ? memory->Track(pool.address, pool.size) : 0;
    if (track_id == 0) {
        if (!vram_mirror_no_tracker_logged) {
            vram_mirror_no_tracker_logged = true;
            LOG_WARNING(Render_Vulkan, "[nvn] pool mirrors: guest memory cannot track CPU writes; "
                                       "pools stay in guest memory");
        }
        return refuse("no CPU write tracking");
    }
    auto mirror = std::make_unique<PoolMirror>();
    try {
        mirror->buffer = CreateHostBuffer(pool.size, pool_usage, MemoryUsage::DeviceLocal,
                                          AllocationKind::Mirror);
    } catch (const vk::Exception& exception) {
        memory->Untrack(track_id);
        LOG_WARNING(Render_Vulkan, "[nvn] pool {:#x} ({:#x} bytes): VRAM mirror allocation failed: {}",
                    pool.address, pool.size, exception.what());
        return refuse("allocation");
    }
    mirror->pages = MirrorPages{pool.size};
    mirror->track_id = track_id;
    mirror->write_epoch = memory->WriteEpoch(track_id);
    if (storage) {
        AttachStorageState(pool, *mirror);
    } else {
        mirror->storage = false;
    }
    pool.vram_mirror = std::move(mirror);
    vram_mirrored.push_back(&pool);
    vram_mirror_bytes += pool.size;
    if (storage) {
        pool_generation.fetch_add(1, std::memory_order_release); // the raw table gains the pool
    }
    stats.vram_mirrors.fetch_add(1, std::memory_order_relaxed);
    if (!storage) {
        stats.vram_mirrors_read_only.fetch_add(1, std::memory_order_relaxed);
    }
    ++ro_alias_generation; // the alias table gains this mirror
    LOG_INFO(Render_Vulkan, "[nvn] pool {:#x} ({:#x} bytes, flags {:#x}) has a {}VRAM mirror ({} MiB "
                            "in {} mirrors)",
             pool.address, pool.size, pool.flags, storage ? "" : "read-only ",
             vram_mirror_bytes >> 20, vram_mirrored.size());
    return pool.vram_mirror.get();
}

VtxCov Context::Impl::ReadOnlyMirrorMiss(const Pool& pool) const {
    // The rules of VramMirror(pool, true, false), in its order.
    if (!vtx_mirror) {
        return VtxCov::FewDraws;
    }
    if (pool.vram_mirror_refused) {
        return VtxCov::NoMirrorRefused;
    }
    if (pool.address % MirrorPages::PAGE != 0) {
        return VtxCov::NoMirrorAlign;
    }
    if (pool.size > try_storage_vram_max_bytes && (vtxcov & VTXCOV_BIGPOOL) == 0) {
        return VtxCov::NoMirrorSize;
    }
    if (vram_mirror_bytes + pool.size > Debug::Get().pool_mirror_max_bytes / 2) {
        return VtxCov::NoMirrorBudget;
    }
    return VtxCov::Other;
}

bool Context::Impl::MirrorCurrent(Pool& pool, u64 offset, u64 size) {
    PoolMirror* const mirror = pool.vram_mirror.get();
    if (mirror == nullptr || !mirror->pages.AllValid(offset, size)) {
        return false;
    }
    if (mirror->write_epoch &&
        mirror->write_epoch->load(std::memory_order_acquire) == mirror->epoch_clean) {
        return true; // no page of the pool was written since the last whole-pool collect
    }
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    return memory == nullptr || !memory->AnyWritten(mirror->track_id, pool.address + offset, size);
}

void Context::Impl::MirrorCollectCpuWrites(Pool& pool, PoolMirror& mirror, u64 offset, u64 size) {
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr) {
        return;
    }
    u64 epoch = 0;
    if (mirror.write_epoch) {
        epoch = mirror.write_epoch->load(std::memory_order_acquire);
        if (epoch == mirror.epoch_clean) {
            return; // nothing was written since a collect that left no page dirty
        }
        // Something was: collect the whole pool (a pool's dirty bitmap is small, and a page
        // invalidated early is only ever copied again when used), so the next check is free.
        offset = 0;
        size = pool.size;
    }
    // Pages the CPU wrote since they were copied in are copied again.
    thread_local std::vector<std::pair<u64, u64>> written;
    written.clear();
    memory->TakeWritten(mirror.track_id, pool.address + offset, size, written);
    for (const auto& [begin, end] : written) {
        const u64 first = (std::max)(begin, pool.address);
        if (end > first) {
            mirror.pages.Invalidate(first - pool.address, end - first);
            mirror.wanted_dirty = true;
        }
    }
    if (mirror.write_epoch) {
        mirror.epoch_clean = epoch; // read before the collect: a write in between shows as a change
    }
}

void Context::Impl::MirrorTakeStale(Pool& pool, u64 offset, u64 size,
                                    std::vector<std::pair<u64, u64>>& runs) {
    PoolMirror* const mirror = pool.vram_mirror.get();
    if (mirror == nullptr || size == 0 || offset >= pool.size) {
        return;
    }
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    MirrorCollectCpuWrites(pool, *mirror, offset, size);
    const size_t first_new = runs.size();
    mirror->pages.TakeInvalid(offset, size, runs);
    MirrorArmRuns(pool, *mirror, memory, first_new, runs);
}

void Context::Impl::MirrorMarkPrefetch(Pool& pool, u64 offset, u64 size) {
    PoolMirror* const mirror = pool.vram_mirror.get();
    if (mirror == nullptr) {
        return;
    }
    if (!mirror->wanted) {
        mirror->wanted = std::make_unique<MirrorPages>(pool.size);
    }
    if (!mirror->prefetch) {
        mirror->prefetch = true;
        vram_prefetch_mirrors.push_back({&pool, mirror, mirror->write_epoch.get()});
    }
    if (!mirror->wanted->AllValid(offset, size)) {
        mirror->wanted->Set(offset, size);
        mirror->wanted_dirty = true;
    }
}

void Context::Impl::MirrorTakeStaleWanted(Pool& pool, std::vector<std::pair<u64, u64>>& runs) {
    PoolMirror* const mirror = pool.vram_mirror.get();
    if (mirror == nullptr || !mirror->wanted) {
        return;
    }
    if (mirror->write_epoch &&
        mirror->write_epoch->load(std::memory_order_acquire) == mirror->epoch_clean &&
        !mirror->wanted_dirty) {
        return;
    }
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    MirrorCollectCpuWrites(pool, *mirror, 0, pool.size);
    const size_t first_new = runs.size();
    mirror->pages.TakeInvalidWithin(*mirror->wanted, runs);
    mirror->wanted_dirty = false;
    MirrorArmRuns(pool, *mirror, memory, first_new, runs);
}

void Context::Impl::MirrorArmRuns(Pool& pool, PoolMirror& mirror, VC::GuestMemory* memory,
                                  size_t first_new, const std::vector<std::pair<u64, u64>>& runs) {
    for (size_t i = first_new; i < runs.size(); ++i) {
        const auto [begin, end] = runs[i];
        if (memory != nullptr) {
            // Tracked from now on: a CPU write after this point is seen at the next use. The copy
            // runs on the GPU later and reads the guest bytes as they are then (at least as new).
            memory->Arm(mirror.track_id, pool.address + begin, end - begin);
        }
        stats.vram_copy_in_runs.fetch_add(1, std::memory_order_relaxed);
        if (mirror.monitor) {
            mirror.monitor->copy_in_bytes.fetch_add(end - begin, std::memory_order_relaxed);
            mirror.monitor->copy_in_runs.fetch_add(1, std::memory_order_relaxed);
        }
        stats.vram_copy_in_bytes.fetch_add(end - begin, std::memory_order_relaxed);
    }
}

void Context::Impl::MirrorInvalidate(Pool& pool, u64 offset, u64 size) {
    if (pool.vram_mirror) {
        pool.vram_mirror->pages.Invalidate(offset, (std::max<u64>)(size, 1));
        pool.vram_mirror->wanted_dirty = true;
    }
}

void Context::Impl::MirrorInvalidateReadOnly(bool written_through) {
    if (Debug::Get().diag_ro_raw_keep) [[unlikely]] {
        return;
    }
    bool any = false;
    bool through_read_only = false;
    for (Pool* const pool : vram_mirrored) {
        PoolMirror& mirror = *pool->vram_mirror;
        if (mirror.storage) {
            continue;
        }
        if (written_through && mirror.written_through) {
            through_read_only = true;
        } else {
            mirror.pages.InvalidateAll();
            mirror.wanted_dirty = true;
            any = true;
        }
    }
    if (through_read_only) {
        stats.vram_ro_written_through.fetch_add(1, std::memory_order_relaxed);
    }
    if (any) {
        stats.vram_invalidate_read_only.fetch_add(1, std::memory_order_relaxed);
    }
}

void Context::Impl::MirrorInvalidateAll() {
    if (vram_mirrored.empty()) {
        return;
    }
    for (Pool* const pool : vram_mirrored) {
        pool->vram_mirror->pages.InvalidateAll();
        pool->vram_mirror->wanted_dirty = true;
    }
    stats.vram_invalidate_all.fetch_add(1, std::memory_order_relaxed);
}

void Context::Impl::ReleaseVramMirror(Pool& pool) {
    if (!pool.vram_mirror) {
        return;
    }
    if (VC::GuestMemory* const memory = guest.load(std::memory_order_acquire)) {
        memory->Untrack(pool.vram_mirror->track_id);
    }
    std::erase(vram_mirrored, &pool);
    std::erase_if(vram_prefetch_mirrors,
                  [&](const PrefetchMirrorEntry& e) { return e.pool == &pool; });
    ++ro_alias_generation; // the alias table loses this mirror
    vram_mirror_bytes -= pool.size;
    // The buffer goes with the pool, after the last submission that used it completed.
}

u64 MirrorPages::ValidPages() const {
    u64 count = 0;
    for (const u64 word : valid) {
        count += static_cast<u64>(std::popcount(word));
    }
    return count;
}

} // namespace Vulkan::Nvn
