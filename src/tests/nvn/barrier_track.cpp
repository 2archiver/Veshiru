// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Which buffer commands of the replay need a barrier between them (nvn_barrier_track.h,
// EDEN_NX2_TRY_BARRIER_MERGE): every read-after-write, write-after-read and write-after-write
// overlap keeps its barrier, disjoint ranges and read-read pairs do not. Pure logic, no device.

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_barrier_track.h"

namespace {

using Vulkan::Nvn::BarrierMergeMask;
using Vulkan::Nvn::BarrierTracker;
using Vulkan::Nvn::BufferAccess;

VkBuffer Buffer(std::uintptr_t value) {
    return reinterpret_cast<VkBuffer>(value);
}

BufferAccess Read(std::uintptr_t buffer, u64 begin, u64 end) {
    return {Buffer(buffer), begin, end, false};
}

BufferAccess Write(std::uintptr_t buffer, u64 begin, u64 end) {
    return {Buffer(buffer), begin, end, true};
}

bool Conflicts(const BarrierTracker& tracker, std::initializer_list<BufferAccess> accesses) {
    const std::vector<BufferAccess> list{accesses};
    return tracker.Conflicts(list);
}

void Note(BarrierTracker& tracker, std::initializer_list<BufferAccess> accesses) {
    const std::vector<BufferAccess> list{accesses};
    tracker.Note(list);
}

} // namespace

TEST_CASE("NVN barrier track: nothing noted means nothing conflicts", "[nvn][barrier]") {
    BarrierTracker tracker;
    CHECK(tracker.Empty());
    CHECK_FALSE(Conflicts(tracker, {Write(1, 0, 64)}));
    CHECK_FALSE(Conflicts(tracker, {Read(1, 0, 64), Write(2, 0, 64)}));
}

TEST_CASE("NVN barrier track: read after write, write after read and write after write overlap",
          "[nvn][barrier]") {
    // The three hazards: the barrier stays whenever one byte is shared and one side writes.
    struct Case {
        bool first_writes;
        bool second_writes;
        bool conflicts;
    };
    for (const Case test : {Case{true, false, true},   // RAW
                            Case{false, true, true},   // WAR
                            Case{true, true, true},    // WAW
                            Case{false, false, false}}) { // read-read
        INFO("first writes " << test.first_writes << " second writes " << test.second_writes);
        BarrierTracker tracker;
        Note(tracker, {BufferAccess{Buffer(1), 100, 200, test.first_writes}});
        CHECK_FALSE(tracker.Empty());
        CHECK(Conflicts(tracker, {BufferAccess{Buffer(1), 150, 160, test.second_writes}}) ==
              test.conflicts);
        // The first and last byte of the range are inside it.
        CHECK(Conflicts(tracker, {BufferAccess{Buffer(1), 199, 300, test.second_writes}}) ==
              test.conflicts);
        CHECK(Conflicts(tracker, {BufferAccess{Buffer(1), 0, 101, test.second_writes}}) ==
              test.conflicts);
        // Containing the whole noted range.
        CHECK(Conflicts(tracker, {BufferAccess{Buffer(1), 0, 1000, test.second_writes}}) ==
              test.conflicts);
    }
}

TEST_CASE("NVN barrier track: ranges that share no byte need no barrier", "[nvn][barrier]") {
    BarrierTracker tracker;
    Note(tracker, {Write(1, 100, 200)});
    // Half-open ranges: touching is not overlapping.
    CHECK_FALSE(Conflicts(tracker, {Write(1, 200, 300)}));
    CHECK_FALSE(Conflicts(tracker, {Write(1, 0, 100)}));
    CHECK_FALSE(Conflicts(tracker, {Read(1, 200, 201)}));
    // Another buffer is another memory.
    CHECK_FALSE(Conflicts(tracker, {Write(2, 100, 200)}));
    CHECK_FALSE(Conflicts(tracker, {Read(2, 0, 1000), Write(3, 0, 1000)}));
    // One conflicting access among disjoint ones keeps the barrier.
    CHECK(Conflicts(tracker, {Write(2, 0, 10), Read(1, 150, 151)}));
}

TEST_CASE("NVN barrier track: a barrier forgets everything before it", "[nvn][barrier]") {
    BarrierTracker tracker;
    Note(tracker, {Write(1, 0, 4096), Read(2, 0, 4096)});
    CHECK(Conflicts(tracker, {Read(1, 0, 4)}));
    CHECK(Conflicts(tracker, {Write(2, 0, 4)}));
    tracker.Clear();
    CHECK(tracker.Empty());
    CHECK_FALSE(Conflicts(tracker, {Read(1, 0, 4)}));
    CHECK_FALSE(Conflicts(tracker, {Write(2, 0, 4)}));
}

TEST_CASE("NVN barrier track: a write-back chain over disjoint entries keeps one barrier",
          "[nvn][barrier]") {
    // Three write-backs of disjoint pool ranges, as a conflict flush loop records them: each reads
    // its mirror and snapshot bytes and rewrites its pool range (every range widened by the merge's
    // line reach, 256 bytes). The first sits behind a barrier the caller emits (a game dispatch
    // wrote); the other two need none.
    constexpr std::uintptr_t MIRROR = 10;
    constexpr std::uintptr_t SNAPSHOT = 11;
    constexpr std::uintptr_t POOL = 12;
    constexpr u64 REACH = 256;
    const auto write_back = [&](u64 begin, u64 end, u64 snapshot_offset) {
        return std::array<BufferAccess, 3>{
            Read(MIRROR, begin > REACH ? begin - REACH : 0, end + REACH),
            Read(SNAPSHOT, snapshot_offset, snapshot_offset + (end - begin)),
            Write(POOL, begin > REACH ? begin - REACH : 0, end + REACH),
        };
    };
    BarrierTracker tracker;
    const auto first = write_back(0x1000, 0x2000, 0);
    const auto second = write_back(0x8000, 0x9000, 0x1000);
    const auto third = write_back(0x20000, 0x20400, 0x2000);
    CHECK_FALSE(tracker.Conflicts(first));
    tracker.Note(first);
    CHECK_FALSE(tracker.Conflicts(second));
    tracker.Note(second);
    CHECK_FALSE(tracker.Conflicts(third));
    tracker.Note(third);
    // Merging the same entry again (its bytes were rewritten by the first merge) conflicts.
    CHECK(tracker.Conflicts(first));
    // So does a copy that reads the pool bytes a merge wrote...
    CHECK(Conflicts(tracker, {Read(POOL, 0x8800, 0x8900)}));
    // ...and one that rewrites the mirror pages a merge reads.
    CHECK(Conflicts(tracker, {Write(MIRROR, 0x1000, 0x1010)}));
    // A copy-in of a page next to an entry, inside the merge's line reach, conflicts too.
    CHECK(Conflicts(tracker, {Write(MIRROR, 0x2000, 0x2100)}));
    CHECK_FALSE(Conflicts(tracker, {Write(MIRROR, 0x3000, 0x4000)}));
}

TEST_CASE("NVN barrier track: neighbouring entries within the line reach keep their barrier",
          "[nvn][barrier]") {
    // Line mode stores whole units (up to 32 words = 128 bytes) around a changed word: an entry
    // that ends inside the unit another entry starts in must be ordered after it.
    constexpr u64 REACH = 256;
    BarrierTracker tracker;
    Note(tracker, {Write(1, 0x1000 - REACH, 0x2000 + REACH)});
    CHECK(Conflicts(tracker, {Write(1, 0x2000 + 64 - REACH, 0x3000 + REACH)}));
    CHECK_FALSE(Conflicts(tracker, {Write(1, 0x2000 + 2 * REACH, 0x3000 + REACH)}));
}

TEST_CASE("NVN barrier track: copy-in then snapshot of the same range is a hazard",
          "[nvn][barrier]") {
    // ApplyMirrors: pages copied into the mirror, then the entry's snapshot reads them.
    constexpr std::uintptr_t POOL = 1;
    constexpr std::uintptr_t MIRROR = 2;
    constexpr std::uintptr_t SNAPSHOT = 3;
    BarrierTracker tracker;
    Note(tracker, {Read(POOL, 0x4000, 0x6000), Write(MIRROR, 0x4000, 0x6000)});
    CHECK(Conflicts(tracker, {Read(MIRROR, 0x4000, 0x5000), Write(SNAPSHOT, 0, 0x1000)}));
    // A snapshot of a range whose pages were already current needs none.
    CHECK_FALSE(Conflicts(tracker, {Read(MIRROR, 0x8000, 0x9000), Write(SNAPSHOT, 0x1000, 0x2000)}));
}

TEST_CASE("NVN barrier track: same-direction ranges of one buffer are joined", "[nvn][barrier]") {
    BarrierTracker tracker;
    Note(tracker, {Write(1, 0, 100), Write(1, 100, 200), Write(1, 50, 150), Read(1, 500, 600)});
    CHECK(tracker.Size() == 2);
    CHECK(Conflicts(tracker, {Read(1, 190, 191)}));
    CHECK_FALSE(Conflicts(tracker, {Read(1, 200, 500)}));
    // A read range does not join a write range of the same bytes.
    Note(tracker, {Read(1, 0, 10)});
    CHECK(tracker.Size() == 3);
}

TEST_CASE("NVN barrier track: too many ranges or a malformed access make everything conflict",
          "[nvn][barrier]") {
    {
        BarrierTracker tracker;
        for (u64 i = 0; i < BarrierTracker::MAX_RANGES; ++i) {
            Note(tracker, {Write(1, i * 1000, i * 1000 + 10)});
        }
        CHECK_FALSE(tracker.Saturated());
        CHECK_FALSE(Conflicts(tracker, {Write(1, 500, 510)}));
        Note(tracker, {Write(1, 900000, 900010)});
        CHECK(tracker.Saturated());
        CHECK_FALSE(tracker.Empty());
        CHECK(Conflicts(tracker, {Write(7, 0, 4)}));
        CHECK(Conflicts(tracker, {Read(7, 0, 4)}));
        tracker.Clear();
        CHECK(tracker.Empty());
        CHECK_FALSE(Conflicts(tracker, {Write(7, 0, 4)}));
    }
    {
        BarrierTracker tracker;
        // A command on a null buffer cannot be compared with anything.
        CHECK(Conflicts(tracker, {BufferAccess{VK_NULL_HANDLE, 0, 4, true}}));
        Note(tracker, {BufferAccess{VK_NULL_HANDLE, 0, 4, true}});
        CHECK(tracker.Saturated());
        CHECK(Conflicts(tracker, {Read(1, 0, 4)}));
    }
    {
        BarrierTracker tracker;
        // An empty range reads and writes nothing.
        Note(tracker, {Write(1, 8, 8)});
        CHECK(tracker.Empty());
        CHECK_FALSE(Conflicts(tracker, {Write(1, 0, 100)}));
        Note(tracker, {Write(1, 0, 100)});
        CHECK_FALSE(Conflicts(tracker, {Write(1, 8, 8)}));
        // An inverted range is a bug of the caller: kept on the safe side.
        CHECK(Conflicts(tracker, {Read(1, 20, 10)}));
    }
}

TEST_CASE("NVN barrier merge: the switch's values select the parts", "[nvn][barrier]") {
    using namespace Vulkan::Nvn;
    constexpr u32 ALL = BARRIER_MERGE_BUFFERS | BARRIER_MERGE_INLINE | BARRIER_MERGE_ADJACENT;
    CHECK(BarrierMergeMask(0) == 0);
    CHECK(BarrierMergeMask(1) == ALL);
    CHECK(BarrierMergeMask(2) == BARRIER_MERGE_BUFFERS);
    CHECK(BarrierMergeMask(3) == BARRIER_MERGE_INLINE);
    CHECK(BarrierMergeMask(4) == BARRIER_MERGE_ADJACENT);
    CHECK(BarrierMergeMask(7) == ALL);
}
