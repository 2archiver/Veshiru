// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/sink.h"

namespace VideoCommon::Nvn {

// Cross-queue event synchronisation of the replay (no Vulkan in here, so the rules are unit tested).
//
// NVN events are 32-bit words in GPU-visible memory. A command buffer signals one
// (nvnCommandBufferSignalEvent: WRITE stores a value, ADD adds one, after all earlier work of the
// queue) and waits on one (nvnCommandBufferWaitEvent: the queue stalls, in its command stream, until
// the word EQUALs the value or is GEQUAL_WRAP, i.e. `(int)(word - value) >= 0`). Queues are
// independent streams: a wait can name a signal another queue submits later, and the SDK does not
// order submissions of different queues. DK's two-queue frame is the example (every value a WRITE):
//
//     q1: Wait e0==0   Signal e4=1   Wait e0==2   Signal e4=3          (submitted first)
//     q0: Signal e0=0  Wait e4==1    Signal e0=2  Wait e4==3  reset every event
//
// The replay therefore does not run submissions whole and in submission order. A submission runs
// as a stream: up to the first wait the event model does not satisfy, then it is suspended and the
// other queues' streams run, until the signal arrives. Everything a stream recorded before it
// suspended is already on the GPU, so a wait that the model satisfies can name the signalling
// segment's timeline value (SignalPoint) and the host GPU orders the two. A wait that nothing can
// satisfy is released after a timeout and reported (never skipped silently).

/// NVNeventSignalMode / NVNeventWaitMode values as they appear in the NVN API (and the records).
inline constexpr u32 EVENT_SIGNAL_WRITE = 0;
inline constexpr u32 EVENT_SIGNAL_ADD = 1;
inline constexpr u32 EVENT_WAIT_EQUAL = 0;
inline constexpr u32 EVENT_WAIT_GEQUAL_WRAP = 1;

/// Where a GPU signal executes on the host: segment `value` of the timeline semaphore of `queue`
/// (the segment containing the signal, which completes at or after the signal).
struct SignalPoint {
    u32 queue = 0;
    u64 value = 0;
    friend bool operator==(const SignalPoint&, const SignalPoint&) = default;
};

/// The value of every event the replay has seen, as of the replay's position in the command
/// streams (not of the GPU's completion). Replay thread only.
class EventModel {
public:
    /// Reads an event's word from guest memory; nullopt when unreadable (taken as 0). Called for an
    /// event the model has not seen before (or forgot): the first use sees the value the guest left
    /// there, e.g. DK stores 0xFFFFFFFF before the first frame.
    using ReadGuestWord = std::function<std::optional<u32>(u64 address)>;

    explicit EventModel(ReadGuestWord read) : read_guest{std::move(read)} {}

    /// `value` satisfies a wait of `mode` for `target` (EQUAL: ==; GEQUAL_WRAP: (s32)(value - target) >= 0).
    /// An unknown mode never satisfies.
    [[nodiscard]] static bool Satisfies(u32 value, u32 mode, u32 target);

    /// A GPU signal at `point`. ADD adds to the word and joins the signals it is built from; any
    /// other mode is a WRITE (the word is replaced, and so are the signals it came from).
    void Signal(u64 address, u32 mode, u32 value, SignalPoint point);

    /// A CPU change (no GPU point: later waits on this value depend on nothing on the GPU, only on
    /// the signals that were added after it).
    void CpuChange(u64 address, EventCpuChange change, u32 value);

    struct WaitCheck {
        bool satisfied = false;
        u32 current = 0; ///< the word as the model has it (for the log)
    };
    /// Checks a wait. When it is satisfied the signals the word was built from (one point per
    /// queue, the latest) are appended to `dependencies`: the waiting stream waits for those.
    WaitCheck Wait(u64 address, u32 mode, u32 target, std::vector<SignalPoint>& dependencies);

    /// The word as the model has it (reads the guest for an unseen event).
    [[nodiscard]] u32 Value(u64 address);

    [[nodiscard]] size_t Size() const noexcept {
        return events.size();
    }

private:
    struct Event {
        u32 value = 0;
        std::vector<SignalPoint> points; ///< the GPU signals the word is built from, latest per queue
    };
    Event& Find(u64 address);
    static void Join(Event& event, SignalPoint point);

    ReadGuestWord read_guest;
    std::map<u64, Event> events;
};

/// One unit of work in a queue's stream: a submission replayed in pieces, or a present.
class StreamItem {
public:
    virtual ~StreamItem() = default;

    struct Step {
        bool done = false;       ///< finished: the scheduler drops it
        bool progressed = false; ///< did something this call (a piece ran) even if not done
    };
    /// Runs as far as it can. Returns when finished, or at the first thing that cannot be done yet
    /// (a wait nothing has satisfied, a sync nobody fenced): then `done` is false, and `progressed`
    /// tells whether anything ran before the block. Called again whenever something else made
    /// progress. `release` = pass the thing that blocks it, whatever it waits for (the scheduler's
    /// timeout expired): the item logs and counts it, then goes on.
    virtual Step Advance(bool release) = 0;
    /// Global submission order (seq): of two runnable streams the lower one runs first.
    [[nodiscard]] virtual u64 Order() const = 0;
};

/// The queues' streams. Per queue FIFO (later items never pass an earlier one of the same queue);
/// across queues the item with the lowest Order() that can make progress goes first, and a stream
/// that blocks lets the others run, until the next Push or Pump. Replay thread only.
class StreamScheduler {
public:
    using Clock = std::chrono::steady_clock;

    /// `timeout`: how long a front item may be blocked without progress before Pump(release=true)
    /// releases it.
    explicit StreamScheduler(Clock::duration timeout) : timeout{timeout} {}

    void Push(u32 queue, std::unique_ptr<StreamItem> item, Clock::time_point now);

    /// Runs everything that can run. With `release_old` a front item blocked for longer than the
    /// timeout is advanced with release = true (counted in Released()).
    void Pump(bool release_old, Clock::time_point now);

    [[nodiscard]] bool Idle() const noexcept {
        return pending == 0;
    }
    [[nodiscard]] size_t Pending() const noexcept {
        return pending;
    }
    /// Items released by the timeout since construction.
    [[nodiscard]] u64 Released() const noexcept {
        return released;
    }
    /// Drops every item without running it (shutdown).
    void Clear();

private:
    struct Entry {
        std::unique_ptr<StreamItem> item;
        Clock::time_point since; ///< last progress (or push) of this entry
    };

    Clock::duration timeout;
    std::map<u32, std::deque<Entry>> streams;
    size_t pending = 0;
    u64 released = 0;
};

} // namespace VideoCommon::Nvn
