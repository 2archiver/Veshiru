// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "video_core/renderer_vulkan/nvn/nvn_line_guard.h"

namespace Vulkan::Nvn {

MirrorLineGuard::Token::Token(std::shared_ptr<MirrorLineGuard> guard_,
                              std::vector<std::pair<u64, u64>> pages_)
    : guard{std::move(guard_)}, pages{std::move(pages_)} {}

MirrorLineGuard::Token::~Token() {
    guard->Release(*this);
}

MirrorLineGuard::MirrorLineGuard(u64 pool_size, std::shared_ptr<LineGuardStats> stats_,
                                 std::thread::id replay_thread_,
                                 std::chrono::milliseconds wait_limit_)
    : pages{(pool_size + PAGE - 1) / PAGE}, stats{std::move(stats_)}, replay_thread{replay_thread_},
      wait_limit{wait_limit_}, hot((pages + 63) / 64, 0), armed((pages + 63) / 64, 0) {}

bool MirrorLineGuard::InFlight(u64 first, u64 last) const {
    return std::ranges::any_of(live, [&](const Token* token) {
        return std::ranges::any_of(token->pages, [&](const std::pair<u64, u64>& run) {
            return run.first <= last && first <= run.second;
        });
    });
}

void MirrorLineGuard::Release(const Token& token) {
    {
        std::scoped_lock lock{mutex};
        std::erase(live, &token);
    }
    idle.notify_all();
}

MirrorLineGuard::Recording MirrorLineGuard::Record(u64 begin, u64 end, bool lines) {
    Recording recording;
    if (end <= begin || pages == 0) {
        return recording;
    }
    end = (std::min)(end, pages * PAGE);
    if (begin >= end) {
        return recording;
    }
    std::scoped_lock lock{mutex};
    if (disarm_pending) {
        disarm_pending = false;
        recording.disarm_all = true;
    }
    const auto add_run = [&](u64 run_begin, u64 run_end, bool line_run) {
        if (!recording.runs.empty() && recording.runs.back().lines == line_run &&
            recording.runs.back().end == run_begin) {
            recording.runs.back().end = run_end;
        } else {
            recording.runs.push_back({run_begin, run_end, line_run});
        }
        (line_run ? stats->line_bytes : stats->exact_bytes)
            .fetch_add(run_end - run_begin, std::memory_order_relaxed);
    };
    if (!lines || exact_only) {
        add_run(begin, end, false);
        return recording;
    }
    const u64 first_page = begin / PAGE;
    const u64 last_page = (end - 1) / PAGE;
    std::vector<std::pair<u64, u64>> line_pages;
    const auto note_line_page = [&](u64 first, u64 last) {
        if (!line_pages.empty() && line_pages.back().second + 1 == first) {
            line_pages.back().second = last;
        } else {
            line_pages.emplace_back(first, last);
        }
    };
    const auto note_arm = [&](u64 first, u64 last) {
        if (!recording.arm.empty() && recording.arm.back().second + 1 == first) {
            recording.arm.back().second = last;
        } else {
            recording.arm.emplace_back(first, last);
        }
    };
    u64 armed_now = 0;
    for (u64 page = first_page; page <= last_page;) {
        // A whole word of untouched, armed pages is one step.
        if (page % 64 == 0 && last_page - page >= 63 && hot[page / 64] == 0 &&
            armed[page / 64] == ~u64{0}) {
            note_line_page(page, page + 63);
            add_run((std::max)(begin, page * PAGE), (std::min)(end, (page + 64) * PAGE), true);
            page += 64;
            continue;
        }
        const u64 run_begin = (std::max)(begin, page * PAGE);
        const u64 run_end = (std::min)(end, (page + 1) * PAGE);
        if (Bit(hot, page)) {
            add_run(run_begin, run_end, false);
        } else {
            if (!Bit(armed, page)) {
                SetBit(armed, page);
                note_arm(page, page);
                ++armed_now;
            }
            note_line_page(page, page);
            add_run(run_begin, run_end, true);
        }
        ++page;
    }
    stats->armed_pages.fetch_add(armed_now, std::memory_order_relaxed);
    if (!line_pages.empty()) {
        recording.token = std::make_shared<Token>(shared_from_this(), std::move(line_pages));
        live.push_back(recording.token.get());
    }
    return recording;
}

bool MirrorLineGuard::Touched(u64 begin, u64 end) {
    if (end <= begin || pages == 0 || begin >= pages * PAGE) {
        return true;
    }
    const u64 first = begin / PAGE;
    const u64 last = (std::min)((end - 1) / PAGE, pages - 1);
    std::unique_lock lock{mutex};
    for (u64 page = first; page <= last; ++page) {
        if (!Bit(hot, page)) {
            SetBit(hot, page);
            ClearBit(armed, page);
            ++hot_pages;
            stats->hot_pages.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!exact_only && hot_pages > MAX_HOT_PAGES) {
        exact_only = true;
        disarm_pending = true;
        stats->exact_pools.fetch_add(1, std::memory_order_relaxed);
    }
    if (std::this_thread::get_id() == replay_thread) {
        return true;
    }
    // Every merge recorded from here on sees these pages hot (same mutex); the ones recorded
    // before have to finish storing before the guest's access lands.
    const auto start = std::chrono::steady_clock::now();
    bool waited = false;
    const bool done = idle.wait_for(lock, wait_limit, [&] {
        if (InFlight(first, last)) {
            waited = true;
            return false;
        }
        return true;
    });
    if (waited) {
        stats->waits.fetch_add(1, std::memory_order_relaxed);
        stats->wait_us.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count()),
            std::memory_order_relaxed);
    }
    if (!done) {
        stats->timeouts.fetch_add(1, std::memory_order_relaxed);
        if (!exact_only) {
            exact_only = true;
            disarm_pending = true;
            stats->exact_pools.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return done;
}

bool MirrorLineGuard::ExactOnly() const {
    std::scoped_lock lock{mutex};
    return exact_only;
}

u32 MirrorLineGuard::HotPages() const {
    std::scoped_lock lock{mutex};
    return hot_pages;
}

bool MirrorLineGuard::HotPage(u64 page) const {
    std::scoped_lock lock{mutex};
    return page < pages && Bit(hot, page);
}

} // namespace Vulkan::Nvn
