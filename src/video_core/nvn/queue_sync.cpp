// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <utility>

#include "video_core/nvn/queue_sync.h"

namespace VideoCommon::Nvn {

bool EventModel::Satisfies(u32 value, u32 mode, u32 target) {
    switch (mode) {
    case EVENT_WAIT_EQUAL:
        return value == target;
    case EVENT_WAIT_GEQUAL_WRAP:
        return static_cast<s32>(value - target) >= 0;
    default:
        return false;
    }
}

EventModel::Event& EventModel::Find(u64 address) {
    const auto [it, inserted] = events.try_emplace(address);
    if (inserted) {
        it->second.value = read_guest ? read_guest(address).value_or(0) : 0;
    }
    return it->second;
}

void EventModel::Join(Event& event, SignalPoint point) {
    for (SignalPoint& existing : event.points) {
        if (existing.queue == point.queue) {
            existing.value = std::max(existing.value, point.value);
            return;
        }
    }
    event.points.push_back(point);
}

void EventModel::Signal(u64 address, u32 mode, u32 value, SignalPoint point) {
    Event& event = Find(address);
    if (mode == EVENT_SIGNAL_ADD) {
        event.value += value;
        Join(event, point);
    } else {
        event.value = value;
        event.points.clear();
        event.points.push_back(point);
    }
}

void EventModel::CpuChange(u64 address, EventCpuChange change, u32 value) {
    switch (change) {
    case EventCpuChange::Forget:
        events.erase(address);
        break;
    case EventCpuChange::Write: {
        // The word is replaced, so there is no need to read what the guest had there.
        Event& event = events[address];
        event.value = value;
        event.points.clear();
        break;
    }
    case EventCpuChange::Add:
        Find(address).value += value;
        break;
    }
}

EventModel::WaitCheck EventModel::Wait(u64 address, u32 mode, u32 target,
                                       std::vector<SignalPoint>& dependencies) {
    const Event& event = Find(address);
    WaitCheck check{.satisfied = Satisfies(event.value, mode, target), .current = event.value};
    if (check.satisfied) {
        dependencies.insert(dependencies.end(), event.points.begin(), event.points.end());
    }
    return check;
}

u32 EventModel::Value(u64 address) {
    return Find(address).value;
}

void StreamScheduler::Push(u32 queue, std::unique_ptr<StreamItem> item, Clock::time_point now) {
    streams[queue].push_back(Entry{std::move(item), now});
    ++pending;
}

void StreamScheduler::Pump(bool release_old, Clock::time_point now) {
    // The front of every queue is a candidate; they are tried in (Order(), queue) order without a
    // per-call container: each step picks the smallest key above the one just tried. The number of
    // queues is a handful, so the repeated scan is cheaper than sorting.
    struct Front {
        u64 order;
        u32 queue;
        std::deque<Entry>* entries;
    };
    const auto next_front = [this](const Front* after, Front& out) {
        bool found = false;
        for (auto& [queue, entries] : streams) {
            if (entries.empty()) {
                continue;
            }
            const u64 order = entries.front().item->Order();
            if (after && (order < after->order || (order == after->order && queue <= after->queue))) {
                continue;
            }
            if (!found || order < out.order || (order == out.order && queue < out.queue)) {
                out = Front{order, queue, &entries};
                found = true;
            }
        }
        return found;
    };

    // The next item of the queue starts its own timeout when it becomes the front: it has not
    // been blocked while the item before it was.
    const auto pop = [this, now](const Front& front) {
        front.entries->pop_front();
        --pending;
        if (!front.entries->empty()) {
            front.entries->front().since = now;
        }
    };

    for (;;) {
        bool progress = false;
        std::optional<Front> oldest_blocked; // lowest-order front past the timeout
        Front front{};
        Front last{};
        bool have_last = false;
        while (next_front(have_last ? &last : nullptr, front)) {
            last = front;
            have_last = true;
            Entry& entry = front.entries->front();
            const StreamItem::Step step = entry.item->Advance(false);
            if (step.done) {
                pop(front);
                progress = true;
                break;
            }
            if (step.progressed) {
                entry.since = now;
                progress = true;
                break;
            }
            if (release_old && !oldest_blocked && now - entry.since > timeout) {
                oldest_blocked = front;
            }
        }
        if (progress) {
            continue;
        }
        if (!oldest_blocked) {
            return;
        }
        // Nothing can run: pass the oldest wait that nothing satisfied in time. Everything that
        // could have run before it has run, so the release is the last resort.
        Entry& entry = oldest_blocked->entries->front();
        const StreamItem::Step step = entry.item->Advance(true);
        ++released;
        if (step.done) {
            pop(*oldest_blocked);
        } else {
            entry.since = now;
        }
    }
}

void StreamScheduler::Clear() {
    for (auto& stream : streams) {
        stream.second.clear();
    }
    pending = 0;
}

} // namespace VideoCommon::Nvn
