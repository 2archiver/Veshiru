// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nvn/queue_sync.h"

using namespace VideoCommon::Nvn;

namespace {

constexpr u32 UNSET = 0xFFFFFFFF; // what DK leaves in its events before the first frame

constexpr u64 E0 = 0x1000;
constexpr u64 E4 = 0x1010;

using Points = std::vector<SignalPoint>;

Points Sorted(Points points) {
    std::ranges::sort(points, [](const SignalPoint& a, const SignalPoint& b) {
        return a.queue != b.queue ? a.queue < b.queue : a.value < b.value;
    });
    return points;
}

/// An EventModel over a fake guest memory that counts its reads.
struct ModelFixture {
    std::map<u64, u32> guest;
    int reads = 0;
    EventModel model{[this](u64 address) -> std::optional<u32> {
        ++reads;
        const auto it = guest.find(address);
        if (it == guest.end()) {
            return std::nullopt;
        }
        return it->second;
    }};
};

} // namespace

TEST_CASE("nvn queue sync: wait modes", "[nvn]") {
    SECTION("EQUAL compares for equality only") {
        REQUIRE(EventModel::Satisfies(5, EVENT_WAIT_EQUAL, 5));
        REQUIRE_FALSE(EventModel::Satisfies(6, EVENT_WAIT_EQUAL, 5));
        REQUIRE_FALSE(EventModel::Satisfies(4, EVENT_WAIT_EQUAL, 5));
        REQUIRE_FALSE(EventModel::Satisfies(UNSET, EVENT_WAIT_EQUAL, 0));
    }
    SECTION("GEQUAL_WRAP is a signed comparison of the difference") {
        REQUIRE(EventModel::Satisfies(5, EVENT_WAIT_GEQUAL_WRAP, 5));
        REQUIRE(EventModel::Satisfies(6, EVENT_WAIT_GEQUAL_WRAP, 5));
        REQUIRE_FALSE(EventModel::Satisfies(4, EVENT_WAIT_GEQUAL_WRAP, 5));
        // Across the 32-bit wrap: 0 is "after" 0xFFFFFFFF, and 1 after 0xFFFFFFFE.
        REQUIRE(EventModel::Satisfies(0, EVENT_WAIT_GEQUAL_WRAP, UNSET));
        REQUIRE_FALSE(EventModel::Satisfies(UNSET, EVENT_WAIT_GEQUAL_WRAP, 0));
        REQUIRE(EventModel::Satisfies(1, EVENT_WAIT_GEQUAL_WRAP, 0xFFFFFFFE));
        REQUIRE_FALSE(EventModel::Satisfies(0xFFFFFFFE, EVENT_WAIT_GEQUAL_WRAP, 1));
        // The window is half the value range.
        REQUIRE(EventModel::Satisfies(0x7FFFFFFF, EVENT_WAIT_GEQUAL_WRAP, 0));
        REQUIRE_FALSE(EventModel::Satisfies(0x80000000, EVENT_WAIT_GEQUAL_WRAP, 0));
    }
    SECTION("an unknown mode never satisfies") {
        REQUIRE_FALSE(EventModel::Satisfies(5, 2, 5));
        REQUIRE_FALSE(EventModel::Satisfies(5, 0xFFFFFFFF, 5));
    }
}

TEST_CASE("nvn queue sync: model signals", "[nvn]") {
    ModelFixture f;
    f.guest[E0] = 0;
    Points deps;

    SECTION("WRITE replaces the value and the signals it came from") {
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {0, 1});
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {1, 1});
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 7, {2, 4});
        REQUIRE(f.model.Value(E0) == 7);
        const auto check = f.model.Wait(E0, EVENT_WAIT_EQUAL, 7, deps);
        REQUIRE(check.satisfied);
        REQUIRE(check.current == 7);
        REQUIRE(deps == Points{{2, 4}});
    }
    SECTION("ADD adds to the word and joins the signals, one point per queue, the latest") {
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {0, 1});
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {1, 1});
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {0, 3});
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {0, 2}); // an older segment of queue 0
        REQUIRE(f.model.Value(E0) == 4);
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_GEQUAL_WRAP, 4, deps).satisfied);
        REQUIRE(Sorted(deps) == Points{{0, 3}, {1, 1}});
    }
    SECTION("ADD after WRITE keeps the written signal") {
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 5, {0, 1});
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {1, 1});
        REQUIRE(f.model.Value(E0) == 6);
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 6, deps).satisfied);
        REQUIRE(Sorted(deps) == Points{{0, 1}, {1, 1}});
    }
    SECTION("ADD wraps like the 32-bit word") {
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, UNSET, {0, 1});
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 2, {0, 2});
        REQUIRE(f.model.Value(E0) == 1);
    }
    SECTION("an unsatisfied wait names no dependencies and reports the word") {
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 3, {0, 1});
        const auto check = f.model.Wait(E0, EVENT_WAIT_EQUAL, 4, deps);
        REQUIRE_FALSE(check.satisfied);
        REQUIRE(check.current == 3);
        REQUIRE(deps.empty());
    }
    SECTION("satisfied waits append to what the caller already has") {
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 3, {0, 1});
        f.model.Signal(E4, EVENT_SIGNAL_WRITE, 1, {1, 5});
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 3, deps).satisfied);
        REQUIRE(f.model.Wait(E4, EVENT_WAIT_EQUAL, 1, deps).satisfied);
        REQUIRE(deps == Points{{0, 1}, {1, 5}});
    }
}

TEST_CASE("nvn queue sync: model CPU changes", "[nvn]") {
    ModelFixture f;
    f.guest[E0] = 0;
    Points deps;

    SECTION("a CPU write replaces the word and clears the signals") {
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 1, {0, 1});
        f.model.CpuChange(E0, EventCpuChange::Write, 9);
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 9, deps).satisfied);
        REQUIRE(deps.empty());
    }
    SECTION("a CPU write on an event the model has not seen") {
        f.model.CpuChange(E4, EventCpuChange::Write, 2);
        REQUIRE(f.model.Value(E4) == 2);
    }
    SECTION("a CPU add changes the word and keeps the signals") {
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 1, {0, 1});
        f.model.CpuChange(E0, EventCpuChange::Add, 1);
        REQUIRE(f.model.Value(E0) == 2);
        REQUIRE_FALSE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 1, deps).satisfied);
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 2, deps).satisfied);
        REQUIRE(deps == Points{{0, 1}});
    }
    SECTION("a signal after a CPU write depends only on the signals added since") {
        f.model.CpuChange(E0, EventCpuChange::Write, 4);
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 1, {1, 8});
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 5, deps).satisfied);
        REQUIRE(deps == Points{{1, 8}});
    }
}

TEST_CASE("nvn queue sync: model reads the guest word", "[nvn]") {
    ModelFixture f;
    Points deps;

    SECTION("first use reads it once; 0xFFFFFFFF does not satisfy EQUAL 0") {
        f.guest[E0] = UNSET;
        const auto check = f.model.Wait(E0, EVENT_WAIT_EQUAL, 0, deps);
        REQUIRE_FALSE(check.satisfied);
        REQUIRE(check.current == UNSET);
        REQUIRE(f.model.Value(E0) == UNSET);
        REQUIRE_FALSE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 0, deps).satisfied);
        REQUIRE(f.reads == 1);
        REQUIRE(f.model.Size() == 1);
    }
    SECTION("an ADD on an unseen event builds on the guest word") {
        f.guest[E0] = 10;
        f.model.Signal(E0, EVENT_SIGNAL_ADD, 5, {0, 1});
        REQUIRE(f.model.Value(E0) == 15);
        REQUIRE(f.reads == 1);
    }
    SECTION("an unreadable word is 0") {
        REQUIRE(f.model.Value(E0) == 0);
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 0, deps).satisfied);
        REQUIRE(f.reads == 1);
    }
    SECTION("Forget drops the value and the signals; the next use reads the guest again") {
        f.guest[E0] = 5;
        f.model.Signal(E0, EVENT_SIGNAL_WRITE, 6, {0, 1});
        REQUIRE(f.reads == 1);
        REQUIRE(f.model.Value(E0) == 6);

        f.guest[E0] = 8;
        f.model.CpuChange(E0, EventCpuChange::Forget, 0);
        REQUIRE(f.model.Size() == 0);
        REQUIRE(f.reads == 1); // forgetting reads nothing by itself
        REQUIRE(f.model.Wait(E0, EVENT_WAIT_EQUAL, 8, deps).satisfied);
        REQUIRE(deps.empty());
        REQUIRE(f.reads == 2);
        REQUIRE(f.model.Value(E0) == 8);
        REQUIRE(f.reads == 2);
    }
    SECTION("Forget of an unseen event is harmless") {
        f.model.CpuChange(E0, EventCpuChange::Forget, 0);
        REQUIRE(f.model.Size() == 0);
        REQUIRE(f.reads == 0);
    }
}

namespace {

/// The fake world of the scheduler tests: events, a log of the work that ran, and the waits.
struct World : ModelFixture {
    std::vector<int> log;                     ///< ids of the Work steps, in execution order
    std::map<int, Points> wait_dependencies; ///< wait label -> the points it depends on
    std::map<u32, u64> segments;              ///< per queue: the last signal point handed out
    int release_advances = 0;                 ///< calls of Advance(true)
    std::vector<int> released_waits;          ///< labels of the waits passed by a release
    int advances = 0;

    World() {
        for (const u64 event : {E0, E4}) {
            guest[event] = UNSET;
        }
    }
};

struct Op {
    enum class Kind { Work, Signal, Wait };
    Kind kind;
    int id;     ///< Work: marker; Wait: label
    u64 event;
    u32 mode;
    u32 value;
};

Op Work(int id) {
    return {Op::Kind::Work, id, 0, 0, 0};
}
Op Signal(u64 event, u32 mode, u32 value) {
    return {Op::Kind::Signal, 0, event, mode, value};
}
Op Wait(int label, u64 event, u32 mode, u32 value) {
    return {Op::Kind::Wait, label, event, mode, value};
}

/// A submission as a script: Work(id) logs, Signal goes through the model at a new segment of the
/// queue, Wait stops the stream until the model satisfies it (or the scheduler releases it).
class ScriptItem final : public StreamItem {
public:
    ScriptItem(World& world_, u32 queue_, u64 order_, std::vector<Op> ops_)
        : world{world_}, queue{queue_}, order{order_}, ops{std::move(ops_)} {}

    Step Advance(bool release) override {
        ++world.advances;
        if (release) {
            ++world.release_advances;
        }
        bool progressed = false;
        while (position < ops.size()) {
            const Op& op = ops[position];
            switch (op.kind) {
            case Op::Kind::Work:
                world.log.push_back(op.id);
                break;
            case Op::Kind::Signal:
                world.model.Signal(op.event, op.mode, op.value,
                                   SignalPoint{queue, ++world.segments[queue]});
                break;
            case Op::Kind::Wait: {
                Points deps;
                if (world.model.Wait(op.event, op.mode, op.value, deps).satisfied) {
                    world.wait_dependencies[op.id] = std::move(deps);
                } else if (release) {
                    world.released_waits.push_back(op.id);
                } else {
                    return {.done = false, .progressed = progressed};
                }
                break;
            }
            }
            ++position;
            progressed = true;
        }
        return {.done = true, .progressed = progressed};
    }
    u64 Order() const override {
        return order;
    }

private:
    World& world;
    u32 queue;
    u64 order;
    std::vector<Op> ops;
    size_t position = 0;
};

/// A present: one Advance, done at once.
class PresentItem final : public StreamItem {
public:
    PresentItem(World& world_, u64 order_, int id_) : world{world_}, order{order_}, id{id_} {}
    Step Advance(bool) override {
        world.log.push_back(id);
        return {.done = true, .progressed = true};
    }
    u64 Order() const override {
        return order;
    }

private:
    World& world;
    u64 order;
    int id;
};

using Clock = StreamScheduler::Clock;
constexpr auto TIMEOUT = std::chrono::seconds(5);
const Clock::time_point T0{};

std::unique_ptr<StreamItem> Script(World& world, u32 queue, u64 order, std::vector<Op> ops) {
    return std::make_unique<ScriptItem>(world, queue, order, std::move(ops));
}

// DK's two-queue frame (every value a WRITE), with a work marker between all operations.
std::vector<Op> FrameQ1() {
    return {Wait(10, E0, EVENT_WAIT_EQUAL, 0),
            Work(11),
            Signal(E4, EVENT_SIGNAL_WRITE, 1),
            Work(12),
            Wait(13, E0, EVENT_WAIT_EQUAL, 2),
            Work(14),
            Signal(E4, EVENT_SIGNAL_WRITE, 3),
            Work(15)};
}
std::vector<Op> FrameQ0() {
    return {Work(1),
            Signal(E0, EVENT_SIGNAL_WRITE, 0),
            Work(2),
            Wait(3, E4, EVENT_WAIT_EQUAL, 1),
            Work(4),
            Signal(E0, EVENT_SIGNAL_WRITE, 2),
            Work(5),
            Wait(6, E4, EVENT_WAIT_EQUAL, 3),
            Work(7),
            Signal(E0, EVENT_SIGNAL_WRITE, UNSET),
            Signal(E4, EVENT_SIGNAL_WRITE, UNSET),
            Work(8)};
}

void CheckPingPongFrame(World& world, int frame) {
    INFO("frame " << frame);
    // q0 runs to its first wait; q1 to its second (it signals e4=1 on the way); q0 to its
    // second (e0=2 on the way); q1 finishes (e4=3); q0 finishes.
    REQUIRE(world.log == std::vector<int>{1, 2, 11, 12, 4, 5, 14, 15, 7, 8});

    // Signal points: q0 makes 4 signals a frame (e0=0, e0=2, two resets), q1 makes 2.
    const u64 q0 = 4 * static_cast<u64>(frame);
    const u64 q1 = 2 * static_cast<u64>(frame);
    REQUIRE(world.wait_dependencies.at(10) == Points{{0, q0 + 1}}); // e0 == 0: q0's e0=0
    REQUIRE(world.wait_dependencies.at(3) == Points{{1, q1 + 1}});  // e4 == 1: q1's e4=1
    REQUIRE(world.wait_dependencies.at(13) == Points{{0, q0 + 2}}); // e0 == 2: q0's e0=2
    REQUIRE(world.wait_dependencies.at(6) == Points{{1, q1 + 2}});  // e4 == 3: q1's e4=3
}

} // namespace

TEST_CASE("nvn queue sync: DK ping-pong interleaves the two queues", "[nvn]") {
    for (const bool q1_first : {true, false}) {
        INFO("q1 pushed first: " << q1_first);
        World world;
        StreamScheduler scheduler{TIMEOUT};
        for (int frame = 0; frame < 3; ++frame) {
            world.log.clear();
            world.wait_dependencies.clear();
            // Submission order is the Order() values; push order differs from it in the second
            // variant.
            const u64 base = 10 * static_cast<u64>(frame);
            if (q1_first) {
                scheduler.Push(1, Script(world, 1, base + 1, FrameQ1()), T0);
                scheduler.Push(0, Script(world, 0, base + 2, FrameQ0()), T0);
            } else {
                scheduler.Push(0, Script(world, 0, base + 1, FrameQ0()), T0);
                scheduler.Push(1, Script(world, 1, base + 2, FrameQ1()), T0);
            }
            REQUIRE(scheduler.Pending() == 2);
            scheduler.Pump(true, T0);
            REQUIRE(scheduler.Idle());
            REQUIRE(scheduler.Released() == 0);
            CheckPingPongFrame(world, frame);
            // The resets left the events as the guest had them before the first frame.
            REQUIRE(world.model.Value(E0) == UNSET);
            REQUIRE(world.model.Value(E4) == UNSET);
        }
        REQUIRE(world.reads == 2); // each event was read from the guest once, in the first frame
    }
}

TEST_CASE("nvn queue sync: ping-pong order does not depend on how Pump is called", "[nvn]") {
    // Pushing the second queue after a first Pump (the first queue blocked) gives the same result.
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(1, Script(world, 1, 1, FrameQ1()), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log.empty());
    REQUIRE(scheduler.Pending() == 1);
    scheduler.Push(0, Script(world, 0, 2, FrameQ0()), T0);
    scheduler.Pump(true, T0);
    REQUIRE(scheduler.Idle());
    CheckPingPongFrame(world, 0);
}

TEST_CASE("nvn queue sync: submission order rules", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};

    SECTION("lowest Order first across queues, FIFO within a queue") {
        scheduler.Push(0, Script(world, 0, 2, {Work(20)}), T0);
        scheduler.Push(1, Script(world, 1, 3, {Work(30)}), T0);
        scheduler.Push(0, Script(world, 0, 5, {Work(50)}), T0);
        scheduler.Push(1, Script(world, 1, 4, {Work(40)}), T0);
        scheduler.Pump(false, T0);
        REQUIRE(world.log == std::vector<int>{20, 30, 40, 50});
        REQUIRE(scheduler.Idle());
    }
    SECTION("a later item never passes an earlier one of the same queue") {
        scheduler.Push(0, Script(world, 0, 9, {Work(90)}), T0);
        scheduler.Push(0, Script(world, 0, 1, {Work(10)}), T0);
        scheduler.Push(1, Script(world, 1, 5, {Work(50)}), T0);
        scheduler.Pump(false, T0);
        REQUIRE(world.log == std::vector<int>{50, 90, 10});
    }
    SECTION("equal Order goes to the lower queue index") {
        scheduler.Push(2, Script(world, 2, 7, {Work(2)}), T0);
        scheduler.Push(1, Script(world, 1, 7, {Work(1)}), T0);
        scheduler.Push(0, Script(world, 0, 7, {Work(0)}), T0);
        scheduler.Pump(false, T0);
        REQUIRE(world.log == std::vector<int>{0, 1, 2});
    }
    SECTION("an item that does part of its work and returns is called again") {
        // Work, a wait on an event a lower-order item of another queue signals: the first item
        // stops after its Work, the other queue runs, the first resumes.
        scheduler.Push(0, Script(world, 0, 1, {Work(1), Wait(1, E0, EVENT_WAIT_EQUAL, 1), Work(2)}), T0);
        scheduler.Push(1, Script(world, 1, 2, {Work(3), Signal(E0, EVENT_SIGNAL_WRITE, 1), Work(4)}), T0);
        scheduler.Pump(false, T0);
        REQUIRE(world.log == std::vector<int>{1, 3, 4, 2});
    }
}

TEST_CASE("nvn queue sync: a blocked stream blocks only its own queue", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0, Script(world, 0, 1, {Work(1), Wait(1, E0, EVENT_WAIT_EQUAL, 5), Work(2)}), T0);
    scheduler.Push(0, Script(world, 0, 2, {Work(3)}), T0);
    scheduler.Push(1, Script(world, 1, 3, {Work(4)}), T0);

    SECTION("nothing signals it: the other queue runs, the wait is released once after the timeout") {
        scheduler.Pump(true, T0);
        REQUIRE(world.log == std::vector<int>{1, 4});
        REQUIRE(scheduler.Pending() == 2);
        REQUIRE(scheduler.Released() == 0);

        // Not yet: the timeout is exceeded, not reached.
        scheduler.Pump(true, T0 + TIMEOUT);
        REQUIRE(scheduler.Released() == 0);
        REQUIRE(world.log == std::vector<int>{1, 4});

        // Without release_old nothing moves, at the same expired time that releases it below.
        scheduler.Pump(false, T0 + TIMEOUT + std::chrono::milliseconds(1));
        REQUIRE(scheduler.Released() == 0);
        REQUIRE(world.log == std::vector<int>{1, 4});

        scheduler.Pump(true, T0 + TIMEOUT + std::chrono::milliseconds(1));
        REQUIRE(scheduler.Released() == 1);
        REQUIRE(world.release_advances == 1);
        REQUIRE(world.released_waits == std::vector<int>{1});
        REQUIRE(world.log == std::vector<int>{1, 4, 2, 3});
        REQUIRE(scheduler.Idle());

        scheduler.Pump(true, T0 + TIMEOUT * 100);
        REQUIRE(scheduler.Released() == 1);
    }
}

TEST_CASE("nvn queue sync: progress restarts the timeout", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0,
                   Script(world, 0, 1,
                          {Wait(1, E0, EVENT_WAIT_EQUAL, 1), Work(1),
                           Wait(2, E4, EVENT_WAIT_EQUAL, 1), Work(2)}),
                   T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log.empty());

    // It runs up to its second wait at 4 s: that is progress, and the 5 s count starts there.
    world.model.Signal(E0, EVENT_SIGNAL_WRITE, 1, {1, 1});
    scheduler.Pump(true, T0 + std::chrono::seconds(4));
    REQUIRE(world.log == std::vector<int>{1});

    scheduler.Pump(true, T0 + std::chrono::seconds(8));
    REQUIRE(scheduler.Released() == 0);
    scheduler.Pump(true, T0 + std::chrono::seconds(9));
    REQUIRE(scheduler.Released() == 0);
    scheduler.Pump(true, T0 + std::chrono::seconds(9) + std::chrono::milliseconds(1));
    REQUIRE(scheduler.Released() == 1);
    REQUIRE(world.released_waits == std::vector<int>{2});
    REQUIRE(world.log == std::vector<int>{1, 2});
}

TEST_CASE("nvn queue sync: the next item of a queue is timed from when it becomes the front", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0, Script(world, 0, 1, {Wait(1, E0, EVENT_WAIT_EQUAL, 5), Work(1)}), T0);
    scheduler.Push(0, Script(world, 0, 2, {Wait(2, E0, EVENT_WAIT_EQUAL, 6), Work(2)}), T0);

    // The first is released at 6 s; the second only starts waiting then.
    scheduler.Pump(true, T0 + std::chrono::seconds(6));
    REQUIRE(scheduler.Released() == 1);
    REQUIRE(world.log == std::vector<int>{1});
    REQUIRE(scheduler.Pending() == 1);

    scheduler.Pump(true, T0 + std::chrono::seconds(11));
    REQUIRE(scheduler.Released() == 1);
    scheduler.Pump(true, T0 + std::chrono::seconds(11) + std::chrono::milliseconds(1));
    REQUIRE(scheduler.Released() == 2);
    REQUIRE(world.released_waits == std::vector<int>{1, 2});
    REQUIRE(world.log == std::vector<int>{1, 2});
    REQUIRE(scheduler.Idle());
}

TEST_CASE("nvn queue sync: a release is the last resort", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    const Clock::time_point late = T0 + TIMEOUT * 2;

    SECTION("a runnable stream that satisfies the old wait goes first") {
        scheduler.Push(0, Script(world, 0, 1, {Wait(1, E0, EVENT_WAIT_EQUAL, 1), Work(1)}), T0);
        scheduler.Push(1, Script(world, 1, 2, {Work(2), Signal(E0, EVENT_SIGNAL_WRITE, 1)}), T0);
        scheduler.Pump(true, late);
        REQUIRE(scheduler.Released() == 0);
        REQUIRE(world.log == std::vector<int>{2, 1});
        REQUIRE(scheduler.Idle());
    }
    SECTION("with every front blocked the lowest Order is released first, one at a time") {
        scheduler.Push(0, Script(world, 0, 1, {Wait(1, E0, EVENT_WAIT_EQUAL, 1), Work(1)}), T0);
        scheduler.Push(1, Script(world, 1, 2,
                                 {Wait(2, E4, EVENT_WAIT_EQUAL, 1), Work(2),
                                  Signal(E0, EVENT_SIGNAL_WRITE, 1)}),
                       T0);
        scheduler.Pump(true, late);
        REQUIRE(scheduler.Released() == 2);
        REQUIRE(world.released_waits == std::vector<int>{1, 2});
        REQUIRE(world.log == std::vector<int>{1, 2});
        REQUIRE(scheduler.Idle());
    }
    SECTION("a release that unblocks another stream is enough") {
        scheduler.Push(0, Script(world, 0, 1, {Wait(1, E4, EVENT_WAIT_EQUAL, 1), Work(1),
                                               Signal(E0, EVENT_SIGNAL_WRITE, 1)}),
                       T0);
        scheduler.Push(1, Script(world, 1, 2, {Wait(2, E0, EVENT_WAIT_EQUAL, 1), Work(2)}), T0);
        scheduler.Pump(true, late);
        REQUIRE(scheduler.Released() == 1);
        REQUIRE(world.released_waits == std::vector<int>{1});
        REQUIRE(world.log == std::vector<int>{1, 2});
        REQUIRE(scheduler.Idle());
    }
}

TEST_CASE("nvn queue sync: a signal pushed later unblocks the wait", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0, Script(world, 0, 1, {Work(1), Wait(1, E0, EVENT_WAIT_EQUAL, 1), Work(2)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1});

    scheduler.Push(1, Script(world, 1, 2, {Work(3), Signal(E0, EVENT_SIGNAL_WRITE, 1)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1, 3, 2});
    REQUIRE(scheduler.Idle());
    REQUIRE(scheduler.Released() == 0);
    REQUIRE(world.wait_dependencies.at(1) == Points{{1, 1}});
}

TEST_CASE("nvn queue sync: a CPU change unblocks an EQUAL wait", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0, Script(world, 0, 1, {Wait(1, E0, EVENT_WAIT_EQUAL, 7), Work(1)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log.empty());

    world.model.CpuChange(E0, EventCpuChange::Write, 7);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1});
    REQUIRE(world.wait_dependencies.at(1).empty());
    REQUIRE(scheduler.Idle());
    REQUIRE(scheduler.Released() == 0);
}

TEST_CASE("nvn queue sync: an ADD-signalled wait depends on every signalling queue", "[nvn]") {
    World world;
    world.guest[E0] = 0;
    StreamScheduler scheduler{TIMEOUT};
    // The waiter has the lowest order and blocks until both queues have signalled.
    scheduler.Push(2, Script(world, 2, 1, {Wait(1, E0, EVENT_WAIT_GEQUAL_WRAP, 2), Work(9)}), T0);
    scheduler.Push(0, Script(world, 0, 2, {Work(1), Signal(E0, EVENT_SIGNAL_ADD, 1)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1});
    REQUIRE(scheduler.Pending() == 1); // the waiter: one signal is not enough

    scheduler.Push(1, Script(world, 1, 3, {Work(2), Signal(E0, EVENT_SIGNAL_ADD, 1)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1, 2, 9});
    REQUIRE(Sorted(world.wait_dependencies.at(1)) == Points{{0, 1}, {1, 1}});
    REQUIRE(scheduler.Idle());
    REQUIRE(scheduler.Released() == 0);
}

TEST_CASE("nvn queue sync: a present waits behind a blocked submission of its queue", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0, Script(world, 0, 1, {Work(1), Wait(1, E0, EVENT_WAIT_EQUAL, 1), Work(2)}), T0);
    scheduler.Push(0, std::make_unique<PresentItem>(world, 2, 100), T0);
    scheduler.Push(1, Script(world, 1, 3, {Work(3)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1, 3}); // the present has not run
    REQUIRE(scheduler.Pending() == 2);

    scheduler.Push(1, Script(world, 1, 4, {Signal(E0, EVENT_SIGNAL_WRITE, 1)}), T0);
    scheduler.Pump(true, T0);
    REQUIRE(world.log == std::vector<int>{1, 3, 2, 100});
    REQUIRE(scheduler.Idle());
}

TEST_CASE("nvn queue sync: Clear drops every item without running it", "[nvn]") {
    World world;
    StreamScheduler scheduler{TIMEOUT};
    scheduler.Push(0, Script(world, 0, 1, {Work(1)}), T0);
    scheduler.Push(1, Script(world, 1, 2, {Work(2)}), T0);
    REQUIRE(scheduler.Pending() == 2);
    scheduler.Clear();
    REQUIRE(scheduler.Idle());
    scheduler.Pump(true, T0 + TIMEOUT * 2);
    REQUIRE(world.log.empty());
    REQUIRE(world.advances == 0);
    scheduler.Push(0, Script(world, 0, 3, {Work(3)}), T0);
    scheduler.Pump(false, T0);
    REQUIRE(world.log == std::vector<int>{3});
}
