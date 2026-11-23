// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>

#include "core/memory/page_tracker_state.h"

namespace Core::Memory {

namespace {

bool TestBit(const std::vector<u64>& bits, u64 index) {
    return ((bits[index >> 6] >> (index & 63)) & 1) != 0;
}

void SetBit(std::vector<u64>& bits, u64 index) {
    bits[index >> 6] |= u64{1} << (index & 63);
}

void ClearBit(std::vector<u64>& bits, u64 index) {
    bits[index >> 6] &= ~(u64{1} << (index & 63));
}

/// Bits [first, last) of word `word`, for a scan over [first, last) bit indices.
u64 WordMask(u64 word, u64 first, u64 last) {
    const u64 word_first = word << 6;
    const u64 lo = first > word_first ? first - word_first : 0;
    const u64 hi = (std::min)(last - word_first, u64{64});
    const u64 upper = hi == 64 ? ~u64{0} : (u64{1} << hi) - 1;
    return upper & ~((u64{1} << lo) - 1);
}

} // Anonymous namespace

PageTrackerState::~PageTrackerState() = default;

void PageTrackerState::RunEmitter::Add(u64 page, PageClaim page_claim) {
    if (end != begin && page == end && page_claim == claim) {
        ++end;
        return;
    }
    Flush();
    begin = page;
    end = page + 1;
    claim = page_claim;
}

void PageTrackerState::RunEmitter::AddRun(u64 run_begin, u64 run_end, PageClaim run_claim) {
    if (run_begin == run_end) {
        return;
    }
    if (end != begin && run_begin == end && run_claim == claim) {
        end = run_end;
        return;
    }
    Flush();
    begin = run_begin;
    end = run_end;
    claim = run_claim;
}

void PageTrackerState::RunEmitter::Flush() {
    if (end != begin) {
        sink.ApplyClaim(begin, end, claim);
    }
    begin = 0;
    end = 0;
}

template <typename Func>
void PageTrackerState::ForEachOwner(u64 page_begin, u64 page_end, Func&& func) const {
    // Owners are sorted by their first page; the longest range bounds how far back one can start
    // and still reach page_begin.
    auto it = std::upper_bound(owners.begin(), owners.end(), page_end - 1,
                               [](u64 page, const std::unique_ptr<Owner>& owner) {
                                   return page < owner->page_begin;
                               });
    while (it != owners.begin()) {
        --it;
        Owner& owner = **it;
        if (owner.page_begin + max_pages <= page_begin) {
            break;
        }
        if (owner.page_end > page_begin) {
            func(owner);
        }
    }
}

void PageTrackerState::TrackedSpans(u64 page_begin, u64 page_end) {
    spans.clear();
    ForEachOwner(page_begin, page_end, [&](const Owner& owner) {
        spans.emplace_back((std::max)(page_begin, owner.page_begin),
                           (std::min)(page_end, owner.page_end));
    });
    std::sort(spans.begin(), spans.end());
    size_t merged = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (merged != 0 && spans[i].first <= spans[merged - 1].second) {
            spans[merged - 1].second = (std::max)(spans[merged - 1].second, spans[i].second);
        } else {
            spans[merged++] = spans[i];
        }
    }
    spans.resize(merged);
}

PageTrackerState::Owner* PageTrackerState::Find(u64 id) {
    for (const auto& owner : owners) {
        if (owner->id == id) {
            return owner.get();
        }
    }
    return nullptr;
}

const PageTrackerState::Owner* PageTrackerState::Find(u64 id) const {
    return const_cast<PageTrackerState*>(this)->Find(id);
}

PageClaim PageTrackerState::ClaimOf(u64 page, bool& tracked) const {
    PageClaim claim{};
    tracked = false;
    ForEachOwner(page, page + 1, [&](const Owner& owner) {
        tracked = true;
        const u64 index = page - owner.page_begin;
        claim.device = claim.device || TestBit(owner.device, index);
        claim.write = claim.write || (TestBit(owner.armed, index) && !TestBit(owner.dirty, index));
        claim.access = claim.access || owner.access.Armed(page, page + 1);
    });
    return claim;
}

void PageTrackerState::RecomputeMaxPages() {
    max_pages = 0;
    for (const auto& owner : owners) {
        max_pages = (std::max)(max_pages, owner->page_end - owner->page_begin);
    }
}

u64 PageTrackerState::Register(u64 page_begin, u64 page_end, bool armed) {
    if (page_end <= page_begin) {
        return 0;
    }
    auto owner = std::make_unique<Owner>();
    owner->id = next_id++;
    owner->page_begin = page_begin;
    owner->page_end = page_end;
    const u64 pages = page_end - page_begin;
    const size_t words = static_cast<size_t>((pages + 63) / 64);
    owner->armed.assign(words, armed ? ~u64{0} : 0);
    owner->dirty.assign(words, 0);
    owner->device.assign(words, 0);
    // The device's state of each page: another owner's copy, or the page type, which only the
    // device memory manager sets on pages nobody tracks.
    for (u64 page = page_begin; page != page_end; ++page) {
        bool tracked;
        const PageClaim prior = ClaimOf(page, tracked);
        if (tracked ? prior.device : sink.DeviceCached(page)) {
            SetBit(owner->device, page - page_begin);
        }
    }
    const u64 id = owner->id;
    const auto position = std::upper_bound(owners.begin(), owners.end(), page_begin,
                                           [](u64 page, const std::unique_ptr<Owner>& other) {
                                               return page < other->page_begin;
                                           });
    owners.insert(position, std::move(owner));
    max_pages = (std::max)(max_pages, pages);
    owner_count.fetch_add(1, std::memory_order_release);
    if (armed) {
        RunEmitter emitter{sink};
        for (u64 page = page_begin; page != page_end; ++page) {
            bool tracked;
            emitter.Add(page, ClaimOf(page, tracked));
        }
    }
    return id;
}

void PageTrackerState::Unregister(u64 id) {
    const auto it = std::find_if(owners.begin(), owners.end(),
                                 [id](const std::unique_ptr<Owner>& owner) { return owner->id == id; });
    if (it == owners.end()) {
        return;
    }
    const std::unique_ptr<Owner> owner = std::move(*it);
    owners.erase(it);
    RecomputeMaxPages();
    access_pages.fetch_sub(owner->access.Size(), std::memory_order_release);
    owner_count.fetch_sub(1, std::memory_order_release);
    RunEmitter emitter{sink};
    for (u64 page = owner->page_begin; page != owner->page_end; ++page) {
        bool tracked;
        PageClaim claim = ClaimOf(page, tracked);
        if (!tracked) {
            claim.device = TestBit(owner->device, page - owner->page_begin);
        }
        emitter.Add(page, claim);
    }
}

void PageTrackerState::Arm(u64 id, u64 page_begin, u64 page_end) {
    Owner* const owner = Find(id);
    if (owner == nullptr) {
        return;
    }
    page_begin = (std::max)(page_begin, owner->page_begin);
    page_end = (std::min)(page_end, owner->page_end);
    RunEmitter emitter{sink};
    for (u64 page = page_begin; page < page_end; ++page) {
        const u64 index = page - owner->page_begin;
        const bool was_armed = TestBit(owner->armed, index);
        const bool was_dirty = TestBit(owner->dirty, index);
        if (was_armed && !was_dirty) {
            continue;
        }
        SetBit(owner->armed, index);
        if (was_dirty) {
            ClearBit(owner->dirty, index);
            --owner->dirty_count;
        }
        bool tracked;
        emitter.Add(page, ClaimOf(page, tracked));
    }
}

void PageTrackerState::CollectDirty(u64 id, u64 page_begin, u64 page_end,
                                    std::vector<std::pair<u64, u64>>& runs) {
    Owner* const owner = Find(id);
    if (owner == nullptr || owner->dirty_count == 0) {
        return;
    }
    page_begin = (std::max)(page_begin, owner->page_begin);
    page_end = (std::min)(page_end, owner->page_end);
    if (page_begin >= page_end) {
        return;
    }
    const size_t first_run = runs.size();
    const u64 first = page_begin - owner->page_begin;
    const u64 last = page_end - owner->page_begin;
    RunEmitter emitter{sink};
    for (u64 word = first >> 6; word <= (last - 1) >> 6; ++word) {
        u64 bits = owner->dirty[word] & WordMask(word, first, last);
        if (bits == 0) {
            continue;
        }
        owner->dirty[word] &= ~bits;
        while (bits != 0) {
            const u64 bit = static_cast<u64>(std::countr_zero(bits));
            bits &= bits - 1;
            const u64 page = owner->page_begin + (word << 6) + bit;
            --owner->dirty_count;
            if (runs.size() > first_run && runs.back().second == page) {
                ++runs.back().second;
            } else {
                runs.emplace_back(page, page + 1);
            }
            bool tracked;
            emitter.Add(page, ClaimOf(page, tracked));
        }
    }
}

bool PageTrackerState::AnyDirty(u64 id, u64 page_begin, u64 page_end) const {
    const Owner* const owner = Find(id);
    if (owner == nullptr || owner->dirty_count == 0) {
        return false;
    }
    page_begin = (std::max)(page_begin, owner->page_begin);
    page_end = (std::min)(page_end, owner->page_end);
    if (page_begin >= page_end) {
        return false;
    }
    const u64 first = page_begin - owner->page_begin;
    const u64 last = page_end - owner->page_begin;
    for (u64 word = first >> 6; word <= (last - 1) >> 6; ++word) {
        if ((owner->dirty[word] & WordMask(word, first, last)) != 0) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<const std::atomic<u64>> PageTrackerState::WriteEpoch(u64 id) const {
    const Owner* const owner = Find(id);
    return owner != nullptr ? owner->epoch : nullptr;
}

void PageTrackerState::SetAccessCallback(u64 id, AccessCallback callback) {
    Owner* const owner = Find(id);
    if (owner == nullptr) {
        return;
    }
    owner->callback =
        callback ? std::make_shared<const AccessCallback>(std::move(callback)) : nullptr;
}

void PageTrackerState::ArmAccess(u64 id, u64 page_begin, u64 page_end) {
    Owner* const owner = Find(id);
    if (owner == nullptr) {
        return;
    }
    page_begin = (std::max)(page_begin, owner->page_begin);
    page_end = (std::min)(page_end, owner->page_end);
    if (page_begin >= page_end) {
        return;
    }
    const size_t before = owner->access.Size();
    const u64 generation = owner->access.NewGeneration();
    RunEmitter emitter{sink};
    for (u64 page = page_begin; page != page_end; ++page) {
        owner->access.Arm(page, generation);
        bool tracked;
        emitter.Add(page, ClaimOf(page, tracked));
    }
    access_pages.fetch_add(owner->access.Size() - before, std::memory_order_release);
}

void PageTrackerState::DisarmAccess(u64 id, u64 page_begin, u64 page_end) {
    Owner* const owner = Find(id);
    if (owner == nullptr) {
        return;
    }
    std::vector<u64> pages;
    const size_t before = owner->access.Size();
    owner->access.Disarm(page_begin, page_end, ~u64{0}, [&](u64 page) { pages.push_back(page); });
    access_pages.fetch_sub(before - owner->access.Size(), std::memory_order_release);
    RunEmitter emitter{sink};
    for (const u64 page : pages) {
        bool tracked;
        emitter.Add(page, ClaimOf(page, tracked));
    }
}

bool PageTrackerState::CollectAccessHits(u64 page_begin, u64 page_end,
                                         std::vector<AccessHit>& hits) const {
    ForEachOwner(page_begin, page_end, [&](const Owner& owner) {
        if (owner.access.Empty()) {
            return;
        }
        const size_t first_hit = hits.size();
        const u64 generation = owner.access.Generation();
        owner.access.ForEach((std::max)(page_begin, owner.page_begin),
                             (std::min)(page_end, owner.page_end), [&](u64 page) {
                                 if (hits.size() > first_hit && hits.back().page_end == page) {
                                     ++hits.back().page_end;
                                     return;
                                 }
                                 hits.push_back(AccessHit{
                                     .callback = owner.callback,
                                     .owner = owner.id,
                                     .page_begin = page,
                                     .page_end = page + 1,
                                     .generation = generation,
                                 });
                             });
    });
    return NeedsDevicePath(page_begin, page_end);
}

void PageTrackerState::FinishAccess(std::span<const AccessHit> hits) {
    std::vector<u64> pages;
    for (const AccessHit& hit : hits) {
        Owner* const owner = Find(hit.owner);
        if (owner == nullptr) {
            continue;
        }
        const size_t before = owner->access.Size();
        owner->access.Disarm(hit.page_begin, hit.page_end, hit.generation,
                             [&](u64 page) { pages.push_back(page); });
        access_pages.fetch_sub(before - owner->access.Size(), std::memory_order_release);
    }
    RunEmitter emitter{sink};
    for (const u64 page : pages) {
        bool tracked;
        emitter.Add(page, ClaimOf(page, tracked));
    }
}

bool PageTrackerState::AccessArmed(u64 page) const {
    bool armed = false;
    ForEachOwner(page, page + 1, [&](const Owner& owner) {
        armed = armed || owner.access.Armed(page, page + 1);
    });
    return armed;
}

bool PageTrackerState::OnWrite(u64 page_begin, u64 page_end) {
    bool device_path = false;
    RunEmitter emitter{sink};
    for (u64 page = page_begin; page != page_end; ++page) {
        bool tracked = false;
        bool changed = false;
        ForEachOwner(page, page + 1, [&](Owner& owner) {
            tracked = true;
            const u64 index = page - owner.page_begin;
            if (TestBit(owner.armed, index) && !TestBit(owner.dirty, index)) {
                SetBit(owner.dirty, index);
                ++owner.dirty_count;
                owner.epoch->fetch_add(1, std::memory_order_release);
                changed = true;
            }
        });
        if (!tracked) {
            device_path = true;
            continue;
        }
        const PageClaim claim = ClaimOf(page, tracked);
        device_path = device_path || claim.device;
        if (changed) {
            emitter.Add(page, claim);
        }
    }
    return device_path;
}

bool PageTrackerState::NeedsDevicePath(u64 page_begin, u64 page_end) const {
    for (u64 page = page_begin; page != page_end; ++page) {
        bool tracked;
        const PageClaim claim = ClaimOf(page, tracked);
        if (!tracked || claim.device) {
            return true;
        }
    }
    return false;
}

void PageTrackerState::SetDeviceCached(u64 page_begin, u64 page_end, bool cached) {
    RunEmitter emitter{sink};
    // Large device ranges cross few tracked pages: untracked stretches go out as one run.
    TrackedSpans(page_begin, page_end);
    u64 cursor = page_begin;
    for (const auto& [span_begin, span_end] : spans) {
        emitter.AddRun(cursor, span_begin, PageClaim{.device = cached});
        for (u64 page = span_begin; page != span_end; ++page) {
            ForEachOwner(page, page + 1, [&](Owner& owner) {
                if (cached) {
                    SetBit(owner.device, page - owner.page_begin);
                } else {
                    ClearBit(owner.device, page - owner.page_begin);
                }
            });
            bool tracked;
            emitter.Add(page, ClaimOf(page, tracked));
        }
        cursor = span_end;
    }
    emitter.AddRun(cursor, page_end, PageClaim{.device = cached});
}

void PageTrackerState::OnRemap(u64 page_begin, u64 page_end, bool mapped) {
    RunEmitter emitter{sink};
    TrackedSpans(page_begin, page_end);
    for (const auto& [span_begin, span_end] : spans) {
        for (u64 page = span_begin; page != span_end; ++page) {
            ForEachOwner(page, page + 1, [&](Owner& owner) {
                const u64 index = page - owner.page_begin;
                ClearBit(owner.device, index);
                if (TestBit(owner.armed, index) && !TestBit(owner.dirty, index)) {
                    SetBit(owner.dirty, index);
                    ++owner.dirty_count;
                    owner.epoch->fetch_add(1, std::memory_order_release);
                }
            });
            if (mapped) {
                bool tracked;
                emitter.Add(page, ClaimOf(page, tracked));
            }
        }
    }
}

} // namespace Core::Memory
