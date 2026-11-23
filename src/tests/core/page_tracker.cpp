// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_resource_limit.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "core/memory/page_tracker.h"
#include "core/memory/page_tracker_state.h"
#include "tests/core/system_test_context.h"
#include "video_core/renderer_vulkan/nvn/nvn_line_guard.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

using Core::Memory::PageClaim;
using Core::Memory::PageTrackerState;
using Runs = std::vector<std::pair<u64, u64>>;

/// Records the claim of every page the tracker applied; DeviceCached answers from a set.
class FakeSink final : public Core::Memory::PageClaimSink {
public:
    void ApplyClaim(u64 page_begin, u64 page_end, PageClaim claim) override {
        ++applies;
        for (u64 page = page_begin; page != page_end; ++page) {
            claims[page] = claim;
        }
    }
    bool DeviceCached(u64 page) const override {
        return device.contains(page);
    }
    bool Cached(u64 page) const {
        const auto it = claims.find(page);
        return it != claims.end() && it->second.Cached();
    }

    std::map<u64, PageClaim> claims;
    std::set<u64> device;
    size_t applies{};
};

Runs Collect(PageTrackerState& state, u64 id, u64 begin = 0, u64 end = ~u64{0} >> 12) {
    Runs runs;
    state.CollectDirty(id, begin, end, runs);
    return runs;
}

} // Anonymous namespace

TEST_CASE("PageTracker: armed pages trap the first write, then run free until collected",
          "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 id = state.Register(100, 108, true);
    REQUIRE(id != 0);
    for (u64 page = 100; page != 108; ++page) {
        CHECK(sink.claims[page] == PageClaim{.write = true});
    }
    CHECK(Collect(state, id).empty());
    CHECK_FALSE(state.AnyDirty(id, 100, 108));

    // The write lands: the page is dirty and released (writes no longer trap).
    CHECK_FALSE(state.OnWrite(103, 104));
    CHECK_FALSE(sink.Cached(103));
    CHECK(state.AnyDirty(id, 100, 108));
    CHECK(state.AnyDirty(id, 103, 104));
    CHECK_FALSE(state.AnyDirty(id, 104, 108));
    const size_t applies = sink.applies;
    CHECK_FALSE(state.OnWrite(103, 104));
    CHECK(sink.applies == applies); // already dirty: nothing to change

    state.OnWrite(104, 106);
    state.OnWrite(107, 108);
    CHECK(Collect(state, id) == Runs{{103, 106}, {107, 108}});
    for (u64 page = 100; page != 108; ++page) {
        CHECK(sink.Cached(page));
    }
    CHECK(Collect(state, id).empty());
}

TEST_CASE("PageTracker: a collect of a sub-range leaves the rest dirty", "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 id = state.Register(0, 200, true);
    state.OnWrite(10, 11);
    state.OnWrite(63, 66); // crosses a bitmap word
    state.OnWrite(150, 151);
    CHECK(Collect(state, id, 60, 140) == Runs{{63, 66}});
    CHECK(state.AnyDirty(id, 0, 60));
    CHECK(state.AnyDirty(id, 140, 200));
    CHECK_FALSE(state.AnyDirty(id, 60, 140));
    CHECK(Collect(state, id) == Runs{{10, 11}, {150, 151}});
}

TEST_CASE("PageTracker: the write epoch grows with each page that turns dirty, nothing else",
          "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 id = state.Register(0, 64, false);
    const std::shared_ptr<const std::atomic<u64>> epoch = state.WriteEpoch(id);
    REQUIRE(epoch != nullptr);
    CHECK(state.WriteEpoch(id + 100) == nullptr);
    const u64 start = epoch->load();

    // Unarmed pages do not count; arming does not either.
    state.OnWrite(0, 4);
    CHECK(epoch->load() == start);
    state.Arm(id, 0, 16);
    CHECK(epoch->load() == start);

    // A write to an armed page moves it; a second write to the same dirty page does not.
    state.OnWrite(3, 4);
    const u64 after_write = epoch->load();
    CHECK(after_write != start);
    state.OnWrite(3, 4);
    CHECK(epoch->load() == after_write);

    // A collect leaves it where it is: a user that read the epoch before collecting every page
    // knows that no page is dirty for as long as the epoch stays that value.
    const u64 before_collect = epoch->load();
    CHECK(Collect(state, id) == Runs{{3, 4}});
    CHECK(epoch->load() == before_collect);
    CHECK_FALSE(state.AnyDirty(id, 0, 64));

    // Every later dirty page changes it, also on another owner's epoch only for that owner.
    const u64 other = state.Register(32, 40, true);
    state.OnWrite(35, 36);
    CHECK(epoch->load() == before_collect);
    CHECK(state.WriteEpoch(other)->load() != 0);
    state.OnWrite(5, 6);
    CHECK(epoch->load() != before_collect);

    // A remap of an armed page makes it dirty: the epoch moves then too.
    Collect(state, id);
    const u64 settled = epoch->load();
    state.OnRemap(0, 4, true);
    CHECK(epoch->load() != settled);
    CHECK(state.AnyDirty(id, 0, 4));
}

TEST_CASE("PageTracker: unarmed pages neither trap nor count until armed",
          "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 id = state.Register(0, 16, false);
    CHECK(sink.claims.empty());
    CHECK_FALSE(state.OnWrite(2, 3));
    CHECK(Collect(state, id).empty());

    state.Arm(id, 4, 8);
    CHECK(sink.claims[4] == PageClaim{.write = true});
    CHECK_FALSE(sink.Cached(2));
    state.OnWrite(0, 16);
    CHECK(Collect(state, id) == Runs{{4, 8}});

    // Arm takes the contents of dirty pages too.
    state.OnWrite(5, 6);
    state.Arm(id, 5, 6);
    CHECK(Collect(state, id).empty());
    CHECK(sink.Cached(5));
}

TEST_CASE("PageTracker: the device memory manager's caching survives the tracker",
          "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    sink.device.insert(3);
    const u64 id = state.Register(0, 8, true);
    CHECK(sink.claims[3] == PageClaim{.device = true, .write = true});

    // A write to a page the device also caches still reaches the device path and stays cached.
    CHECK(state.OnWrite(3, 4));
    CHECK(sink.claims[3] == PageClaim{.device = true});
    CHECK_FALSE(state.OnWrite(4, 5));

    // The device releasing an armed page keeps it trapped for the tracker.
    state.SetDeviceCached(5, 6, true);
    state.SetDeviceCached(5, 6, false);
    CHECK(sink.claims[5] == PageClaim{.write = true});

    // Untracked pages follow the device as one run.
    const size_t applies = sink.applies;
    state.SetDeviceCached(100, 200, true);
    CHECK(sink.applies == applies + 1);
    CHECK(sink.claims[150] == PageClaim{.device = true});

    // A device range across tracked and untracked pages.
    state.SetDeviceCached(6, 12, true);
    CHECK(sink.claims[6] == PageClaim{.device = true, .write = true});
    CHECK(sink.claims[10] == PageClaim{.device = true});

    // Unregistering hands every page back to the device's state.
    state.Unregister(id);
    CHECK(state.Empty());
    CHECK(sink.claims[3] == PageClaim{.device = true});
    CHECK(sink.claims[6] == PageClaim{.device = true});
    CHECK_FALSE(sink.Cached(0));
    CHECK_FALSE(sink.Cached(4));
}

TEST_CASE("PageTracker: overlapping owners collect independently", "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 a = state.Register(0, 10, true);
    const u64 b = state.Register(5, 15, true);
    state.OnWrite(7, 8);
    CHECK_FALSE(sink.Cached(7));
    CHECK(Collect(state, a) == Runs{{7, 8}});
    CHECK(sink.Cached(7)); // a holds it again
    state.OnWrite(7, 8);   // b is already dirty there, a turns dirty
    CHECK(Collect(state, b) == Runs{{7, 8}});
    CHECK(Collect(state, a) == Runs{{7, 8}});
    state.Unregister(a);
    CHECK(sink.Cached(7)); // b still holds it
    CHECK_FALSE(sink.Cached(2));
    state.Unregister(b);
    CHECK_FALSE(sink.Cached(7));
}

TEST_CASE("PageTracker: access traps call back once and survive a re-arm during the callback",
          "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 id = state.Register(0, 8, false);
    std::vector<std::pair<u64, u64>> calls;
    state.SetAccessCallback(id, [&](u64 va, u64 size) { calls.emplace_back(va, size); });
    CHECK_FALSE(state.HasAccessTraps());
    state.ArmAccess(id, 2, 5);
    CHECK(state.HasAccessTraps());
    CHECK(sink.claims[2] == PageClaim{.access = true});
    CHECK(state.AccessArmed(4));
    CHECK_FALSE(state.AccessArmed(5));

    std::vector<PageTrackerState::AccessHit> hits;
    CHECK_FALSE(state.CollectAccessHits(3, 6, hits));
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].page_begin == 3);
    CHECK(hits[0].page_end == 5);
    REQUIRE(hits[0].callback);
    (*hits[0].callback)(hits[0].page_begin << 12, (hits[0].page_end - hits[0].page_begin) << 12);
    // New GPU work armed page 4 again while the callback ran: it stays armed.
    state.ArmAccess(id, 4, 5);
    state.FinishAccess(hits);
    CHECK(calls == std::vector<std::pair<u64, u64>>{{3 << 12, 2 << 12}});
    CHECK_FALSE(state.AccessArmed(3));
    CHECK_FALSE(sink.Cached(3));
    CHECK(state.AccessArmed(4));
    CHECK(state.AccessArmed(2));

    state.DisarmAccess(id, 0, 8);
    CHECK_FALSE(state.HasAccessTraps());
    CHECK_FALSE(sink.Cached(2));
    CHECK_FALSE(sink.Cached(4));

    // Unregister drops the traps without calling back.
    state.ArmAccess(id, 0, 8);
    state.Unregister(id);
    CHECK_FALSE(state.HasAccessTraps());
    CHECK(calls.size() == 1);
}

TEST_CASE("PageTracker: a guest remap makes tracked pages dirty", "[core][page_tracker]") {
    FakeSink sink;
    PageTrackerState state{sink};
    const u64 id = state.Register(0, 8, true);
    state.SetDeviceCached(1, 2, true);
    state.OnRemap(0, 4, false);
    state.OnRemap(0, 4, true);
    CHECK(Collect(state, id) == Runs{{0, 4}});
    CHECK(sink.claims[1] == PageClaim{.write = true}); // the device's mark went with the mapping
    CHECK(state.NeedsDevicePath(8, 9));
    CHECK_FALSE(state.NeedsDevicePath(0, 8));
}

namespace {

constexpr u64 RIG_BASE = 0x40000000;

/// A process whose page table maps `pages` host pages at RIG_BASE (no fastmem: the page types
/// route accesses exactly as under fastmem's slow path).
template <typename Body>
void WithProcessMemory(size_t pages, Body&& body, bool enable_tracker = true) {
    ScopedSystemTestContext environment;
    Core::System system;
    system.Initialize();
    auto& kernel = system.Kernel();
    Kernel::KThread bootstrap{kernel};
    REQUIRE(Kernel::KThread::InitializeDummyThread(system, &bootstrap, nullptr).IsSuccess());
    kernel.SetCurrentEmuThread(&bootstrap);
    kernel.Initialize();
    SCOPE_EXIT {
        system.CoreTiming().ClearPendingEvents();
        kernel.Shutdown();
        kernel.SetCurrentEmuThread(nullptr);
        kernel.UnregisterKernelObject(&bootstrap);
    };
    Kernel::KProcess process{kernel};
    Kernel::KAutoObject::Create(&process);
    SCOPE_EXIT {
        kernel.UnregisterKernelObject(&process);
    };
    auto& page_table = process.GetPageTable();
    auto& memory = process.GetMemory();
    REQUIRE(page_table
                .Initialize(Kernel::Svc::CreateProcessFlag::AddressSpace64Bit, false, false, false,
                            Kernel::KMemoryManager::Pool::Application, 0x8000000, 0x1000,
                            &kernel.GetSystemSystemResource(), kernel.GetSystemResourceLimit(),
                            memory, 0)
                .IsSuccess());
    SCOPE_EXIT {
        page_table.Finalize();
    };
    Kernel::Svc::CreateProcessParameter parameters{};
    parameters.flags = Kernel::Svc::CreateProcessFlag::AddressSpace64Bit;
    REQUIRE(
        process.Initialize(kernel, parameters, kernel.GetSystemResourceLimit(), false).IsSuccess());
    SCOPE_EXIT {
        process.GetResourceLimit()->Close(kernel);
    };
    memory.SetCurrentPageTable(process);
    if (enable_tracker) {
        memory.EnablePageTracker();
    }

    const size_t bytes = pages * Core::Memory::VESHIRU_PAGESIZE;
    std::unique_ptr<u8[]> storage(new u8[bytes + Core::Memory::VESHIRU_PAGESIZE]());
    u8* const host = reinterpret_cast<u8*>(
        (reinterpret_cast<uintptr_t>(storage.get()) + Core::Memory::VESHIRU_PAGEMASK) &
        ~uintptr_t{Core::Memory::VESHIRU_PAGEMASK});
    auto& table = page_table.GetImpl();
    for (size_t i = 0; i < pages; ++i) {
        table.entries.GetAndFault((RIG_BASE >> Core::Memory::VESHIRU_PAGEBITS) + i)
            .Store(false, Common::PageType::Memory, 1, reinterpret_cast<uintptr_t>(host) - RIG_BASE);
    }
    SCOPE_EXIT {
        for (size_t i = 0; i < pages; ++i) {
            table.entries.GetAndFault((RIG_BASE >> Core::Memory::VESHIRU_PAGEBITS) + i)
                .Store(false, Common::PageType::Unmapped, 1, 0);
        }
    };
    body(memory, table, host);
}

Common::PageType TypeOf(const Common::PageTable& table, u64 va) {
    return table.entries[va >> Core::Memory::VESHIRU_PAGEBITS].Type();
}

constexpr u64 PAGE = Core::Memory::VESHIRU_PAGESIZE;

} // Anonymous namespace

TEST_CASE("PageTracker: off unless enabled, and then memory behaves as without it",
          "[core][page_tracker]") {
    WithProcessMemory(
        2,
        [](Core::Memory::Memory& memory, Common::PageTable& table, u8* host) {
            auto& tracker = memory.PageTracker();
            CHECK(tracker.Register(RIG_BASE, 2 * PAGE) == 0);
            CHECK(TypeOf(table, RIG_BASE) == Common::PageType::Memory);
            memory.Write32(RIG_BASE, 7);
            REQUIRE(memory.WriteBlockUnsafe(RIG_BASE + PAGE, "abcd", 4));
            memory.NotifyHostWrite(RIG_BASE, 2 * PAGE);
            CHECK(host[0] == 7);
            CHECK(std::memcmp(host + PAGE, "abcd", 4) == 0);
            CHECK(TypeOf(table, RIG_BASE) == Common::PageType::Memory);
            CHECK(TypeOf(table, RIG_BASE + PAGE) == Common::PageType::Memory);
        },
        false);
}

TEST_CASE("PageTracker: guest writes through Memory are tracked, raw host writes are not",
          "[core][page_tracker]") {
    WithProcessMemory(8, [](Core::Memory::Memory& memory, Common::PageTable& table, u8* host) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, 8 * PAGE);
        REQUIRE(id != 0);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        CHECK(TypeOf(table, RIG_BASE) == Common::PageType::RasterizerCachedMemory);
        Runs runs;

        // Not counted: the HLE's own accesses (raw host pointer, the write-back of GPU results).
        std::memset(host + 2 * PAGE, 0x11, 16);
        REQUIRE(memory.WriteBlockUntracked(RIG_BASE + 3 * PAGE, "abcd", 4));
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs.empty());

        // Counted: scalar writes, blocks, zeroing, exclusive stores, data cache cleans.
        memory.Write32(RIG_BASE + PAGE + 12, 0xdeadbeef);
        CHECK(host[PAGE + 12] == 0xef);
        CHECK(TypeOf(table, RIG_BASE + PAGE) == Common::PageType::Memory);
        CHECK(TypeOf(table, RIG_BASE) == Common::PageType::RasterizerCachedMemory);
        REQUIRE(memory.WriteBlock(RIG_BASE + 4 * PAGE - 2, "wxyz", 4));
        REQUIRE(memory.ZeroBlock(RIG_BASE + 6 * PAGE, 8));
        CHECK(tracker.AnyDirty(id, RIG_BASE + PAGE, 1));
        CHECK_FALSE(tracker.AnyDirty(id, RIG_BASE + 2 * PAGE, PAGE));
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs == Runs{{RIG_BASE + PAGE, RIG_BASE + 2 * PAGE},
                           {RIG_BASE + 3 * PAGE, RIG_BASE + 5 * PAGE},
                           {RIG_BASE + 6 * PAGE, RIG_BASE + 7 * PAGE}});
        CHECK(TypeOf(table, RIG_BASE + PAGE) == Common::PageType::RasterizerCachedMemory);

        runs.clear();
        CHECK(memory.WriteExclusive32(RIG_BASE + 7 * PAGE, 5, 0));
        REQUIRE(memory.StoreDataCache(RIG_BASE + 5 * PAGE, 64).IsSuccess());
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs == Runs{{RIG_BASE + 5 * PAGE, RIG_BASE + 6 * PAGE},
                           {RIG_BASE + 7 * PAGE, RIG_BASE + 8 * PAGE}});

        // Reads do not count.
        runs.clear();
        CHECK(memory.Read32(RIG_BASE + PAGE + 12) == 0xdeadbeef);
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs.empty());
    });
}

TEST_CASE("PageTracker: a host writer's WriteBlockUnsafe and NotifyHostWrite are seen",
          "[core][page_tracker]") {
    // Nine pages, the tracker owns the first eight.
    WithProcessMemory(9, [](Core::Memory::Memory& memory, Common::PageTable& table, u8* host) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, 8 * PAGE);
        REQUIRE(id != 0);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        Runs runs;

        // A host service's block write skips the device caches but is a write of the guest's
        // memory: the page turns dirty and the bytes land.
        REQUIRE(memory.WriteBlockUnsafe(RIG_BASE + 2 * PAGE + 8, "abcd", 4));
        CHECK(std::memcmp(host + 2 * PAGE + 8, "abcd", 4) == 0);
        CHECK(TypeOf(table, RIG_BASE + 2 * PAGE) == Common::PageType::Memory);
        // One that straddles two pages dirties both.
        REQUIRE(memory.WriteBlockUnsafe(RIG_BASE + 5 * PAGE - 2, "wxyz", 4));
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs == Runs{{RIG_BASE + 2 * PAGE, RIG_BASE + 3 * PAGE},
                           {RIG_BASE + 4 * PAGE, RIG_BASE + 6 * PAGE}});

        // The HLE's write-back is not.
        runs.clear();
        REQUIRE(memory.WriteBlockUntracked(RIG_BASE + 3 * PAGE, "abcd", 4));
        CHECK(std::memcmp(host + 3 * PAGE, "abcd", 4) == 0);
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs.empty());

        // A raw-pointer writer (the kernel's IPC copies) says what it wrote; nothing is dirty
        // until it does.
        std::memset(host + 6 * PAGE + 100, 0x77, PAGE);
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs.empty());
        memory.NotifyHostWrite(RIG_BASE + 6 * PAGE + 100, PAGE);
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs == Runs{{RIG_BASE + 6 * PAGE, RIG_BASE + 8 * PAGE}});

        // An empty range and a range nobody tracks are no-ops.
        runs.clear();
        memory.NotifyHostWrite(RIG_BASE, 0);
        memory.NotifyHostWrite(RIG_BASE + 8 * PAGE, PAGE);
        CHECK(TypeOf(table, RIG_BASE + 8 * PAGE) == Common::PageType::Memory);
        tracker.CollectDirty(id, RIG_BASE, 8 * PAGE, runs);
        CHECK(runs.empty());
    });
}

TEST_CASE("PageTracker: a host service's write lands after the access trap's write-back, the HLE's "
          "own write-back does not call it",
          "[core][page_tracker]") {
    WithProcessMemory(2, [](Core::Memory::Memory& memory, Common::PageTable&, u8* host) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, 2 * PAGE, false);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        int calls = 0;
        // The callback stands in for the write-back: it writes the old contents over the range.
        tracker.SetAccessCallback(id, [&](u64, u64) {
            ++calls;
            std::memcpy(host, "wxyz", 4);
        });

        tracker.ArmAccessTrap(id, RIG_BASE, PAGE);
        REQUIRE(memory.WriteBlockUntracked(RIG_BASE, "abcd", 4));
        CHECK(calls == 0);
        CHECK(std::memcmp(host, "abcd", 4) == 0);
        REQUIRE(memory.WriteBlockUnsafe(RIG_BASE, "efgh", 4));
        CHECK(calls == 1);
        CHECK(std::memcmp(host, "efgh", 4) == 0);
    });
}

TEST_CASE("PageTracker: unregistering restores the plain page types", "[core][page_tracker]") {
    WithProcessMemory(4, [](Core::Memory::Memory& memory, Common::PageTable& table, u8* host) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE + 1, 2 * PAGE); // rounds out to 3 pages
        CHECK(TypeOf(table, RIG_BASE + 2 * PAGE) == Common::PageType::RasterizerCachedMemory);
        CHECK(TypeOf(table, RIG_BASE + 3 * PAGE) == Common::PageType::Memory);
        tracker.Unregister(id);
        for (u64 page = 0; page < 4; ++page) {
            CHECK(TypeOf(table, RIG_BASE + page * PAGE) == Common::PageType::Memory);
        }
        memory.Write32(RIG_BASE, 7);
        CHECK(host[0] == 7);
        CHECK(tracker.Register(0, 0) == 0);
    });
}

TEST_CASE("PageTracker: an access trap calls back before the guest's read or write lands",
          "[core][page_tracker]") {
    WithProcessMemory(4, [](Core::Memory::Memory& memory, Common::PageTable& table, u8* host) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, 4 * PAGE, false);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        std::vector<std::pair<u64, u64>> calls;
        // Stands in for the lazy write-back: it writes the GPU's result through the host pointer.
        tracker.SetAccessCallback(id, [&](u64 va, u64 size) {
            calls.emplace_back(va, size);
            const u32 gpu_value = 0x600d;
            std::memcpy(host + (va - RIG_BASE), &gpu_value, sizeof(gpu_value));
            std::memcpy(host + (va - RIG_BASE) + 4, &gpu_value, sizeof(gpu_value));
        });
        tracker.ArmAccessTrap(id, RIG_BASE, 2 * PAGE);
        CHECK(TypeOf(table, RIG_BASE) == Common::PageType::RasterizerCachedMemory);
        CHECK(TypeOf(table, RIG_BASE + 2 * PAGE) == Common::PageType::Memory);

        // A read sees the written-back value.
        CHECK(memory.Read32(RIG_BASE) == 0x600d);
        CHECK(calls == std::vector<std::pair<u64, u64>>{{RIG_BASE, PAGE}});
        CHECK(memory.Read32(RIG_BASE) == 0x600d); // disarmed: no second call
        CHECK(calls.size() == 1);
        CHECK(TypeOf(table, RIG_BASE) == Common::PageType::Memory);

        // A write: the write-back lands first, so the CPU's word wins and the GPU's other word
        // survives.
        memory.Write32(RIG_BASE + PAGE, 0xc0de);
        REQUIRE(calls.size() == 2);
        CHECK(calls[1] == std::pair<u64, u64>{RIG_BASE + PAGE, PAGE});
        CHECK(memory.Read32(RIG_BASE + PAGE) == 0xc0de);
        CHECK(memory.Read32(RIG_BASE + PAGE + 4) == 0x600d);

        // Block reads trigger it too.
        tracker.ArmAccessTrap(id, RIG_BASE + 3 * PAGE, PAGE);
        u32 value{};
        REQUIRE(memory.ReadBlock(RIG_BASE + 3 * PAGE + 4, &value, sizeof(value)));
        CHECK(value == 0x600d);
        CHECK(calls.size() == 3);
    });
}

TEST_CASE("PageTracker: write tracking and access traps on one page", "[core][page_tracker]") {
    WithProcessMemory(1, [](Core::Memory::Memory& memory, Common::PageTable& table, u8*) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, PAGE);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        int calls = 0;
        tracker.SetAccessCallback(id, [&](u64, u64) { ++calls; });
        tracker.ArmAccessTrap(id, RIG_BASE, PAGE);
        CHECK(memory.Read32(RIG_BASE) == 0);
        CHECK(calls == 1);
        // Write tracking stays after the access trap went.
        CHECK(TypeOf(table, RIG_BASE) == Common::PageType::RasterizerCachedMemory);
        memory.Write32(RIG_BASE, 1);
        CHECK(TypeOf(table, RIG_BASE) == Common::PageType::Memory);
        Runs runs;
        tracker.CollectDirty(id, RIG_BASE, PAGE, runs);
        CHECK(runs == Runs{{RIG_BASE, RIG_BASE + PAGE}});
    });
}

TEST_CASE("PageTracker: concurrent guest writers lose no write between collects",
          "[core][page_tracker]") {
    constexpr size_t pages = 64;
    constexpr size_t writers = 4;
    constexpr int rounds = 200;
    WithProcessMemory(pages, [&](Core::Memory::Memory& memory, Common::PageTable&, u8* host) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, pages * PAGE);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        // The mirror starts equal (Register's contract: the owner holds the contents).
        std::vector<u8> mirror(host, host + pages * PAGE);
        std::barrier sync{static_cast<std::ptrdiff_t>(writers + 1)};
        std::vector<std::jthread> threads;
        for (size_t w = 0; w < writers; ++w) {
            threads.emplace_back([&, w] {
                u32 state = static_cast<u32>(w * 7919 + 1);
                for (int round = 0; round < rounds; ++round) {
                    sync.arrive_and_wait();
                    // Writers collide on pages (and on the dirty bit) but own their words.
                    for (int i = 0; i < 32; ++i) {
                        state = state * 1664525u + 1013904223u;
                        const u64 page = (state >> 8) % pages;
                        const u64 word = (state >> 20) % (PAGE / 4 / writers) * writers + w;
                        memory.Write32(RIG_BASE + page * PAGE + word * 4,
                                       static_cast<u32>(round + 1));
                    }
                    sync.arrive_and_wait();
                }
            });
        }
        bool equal = true;
        Runs runs;
        for (int round = 0; round < rounds; ++round) {
            sync.arrive_and_wait();
            sync.arrive_and_wait();
            runs.clear();
            tracker.CollectDirty(id, RIG_BASE, pages * PAGE, runs);
            for (const auto& [begin, end] : runs) {
                std::memcpy(mirror.data() + (begin - RIG_BASE), host + (begin - RIG_BASE),
                            end - begin);
            }
            equal = equal && std::memcmp(mirror.data(), host, mirror.size()) == 0;
        }
        threads.clear();
        CHECK(equal);
    });
}

TEST_CASE("PageTracker: collects racing guest writers stay consistent", "[core][page_tracker]") {
    constexpr size_t pages = 32;
    WithProcessMemory(pages, [&](Core::Memory::Memory& memory, Common::PageTable& table, u8*) {
        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, pages * PAGE);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        std::atomic<bool> stop{false};
        std::vector<std::jthread> threads;
        for (u32 w = 0; w < 3; ++w) {
            threads.emplace_back([&, w] {
                u32 i = w;
                while (!stop.load(std::memory_order_relaxed)) {
                    memory.Write32(RIG_BASE + (i % pages) * PAGE + w * 4, i);
                    i += 3;
                }
            });
        }
        Runs runs;
        for (int i = 0; i < 2000; ++i) {
            runs.clear();
            tracker.CollectDirty(id, RIG_BASE, pages * PAGE, runs);
            CHECK(std::is_sorted(runs.begin(), runs.end()));
        }
        stop = true;
        threads.clear();
        // Quiet again: one collect catches up, then every page is armed and exact.
        runs.clear();
        tracker.CollectDirty(id, RIG_BASE, pages * PAGE, runs);
        for (u64 page = 0; page < pages; ++page) {
            CHECK(TypeOf(table, RIG_BASE + page * PAGE) == Common::PageType::RasterizerCachedMemory);
        }
        memory.Write32(RIG_BASE + 5 * PAGE, 1);
        runs.clear();
        tracker.CollectDirty(id, RIG_BASE, pages * PAGE, runs);
        CHECK(runs == Runs{{RIG_BASE + 5 * PAGE, RIG_BASE + 6 * PAGE}});
    });
}

TEST_CASE("PageTracker: cost of a guest write to a tracked page", "[.][bench][page_tracker]") {
    using Clock = std::chrono::steady_clock;
    constexpr size_t pages = 512;
    constexpr int rounds = 64;
    WithProcessMemory(pages + 1, [&](Core::Memory::Memory& memory, Common::PageTable&, u8*) {
        const auto ns_per = [](Clock::duration d, double n) {
            return std::chrono::duration<double, std::nano>(d).count() / n;
        };
        // Baseline: Memory::Write32 to a plain page.
        const u64 plain = RIG_BASE + pages * PAGE;
        auto start = Clock::now();
        for (int i = 0; i < rounds * static_cast<int>(pages); ++i) {
            memory.Write32(plain + (i & 1023) * 4, i);
        }
        const double plain_ns = ns_per(Clock::now() - start, rounds * pages);

        auto& tracker = memory.PageTracker();
        const u64 id = tracker.Register(RIG_BASE, pages * PAGE);
        SCOPE_EXIT {
            tracker.Unregister(id);
        };
        Clock::duration trapped{};
        Clock::duration free_writes{};
        Clock::duration collects{};
        Runs runs;
        for (int round = 0; round < rounds; ++round) {
            start = Clock::now();
            for (size_t page = 0; page < pages; ++page) {
                memory.Write32(RIG_BASE + page * PAGE, round); // first write: trapped
            }
            trapped += Clock::now() - start;
            start = Clock::now();
            for (size_t page = 0; page < pages; ++page) {
                memory.Write32(RIG_BASE + page * PAGE + 4, round); // dirty: free
            }
            free_writes += Clock::now() - start;
            start = Clock::now();
            runs.clear();
            tracker.CollectDirty(id, RIG_BASE, pages * PAGE, runs);
            collects += Clock::now() - start;
        }
        std::printf("page_tracker bench: plain Write32 %.1f ns; tracked clean page (trap path, "
                    "no host protect) %.1f ns; tracked dirty page %.1f ns; CollectDirty re-arm "
                    "%.1f ns/page (%zu pages in one run)\n",
                    plain_ns, ns_per(trapped, rounds * pages), ns_per(free_writes, rounds * pages),
                    ns_per(collects, rounds * pages), pages);
    });
#ifdef _WIN32
    // What fastmem adds on the host: the protection change of one page (both on release and on
    // re-arm), and the fault a fastmem store takes on a write-protected page.
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    constexpr size_t protect_pages = 256;
    u8* const region = static_cast<u8*>(VirtualAlloc(nullptr, protect_pages * PAGE,
                                                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    REQUIRE(region != nullptr);
    SCOPE_EXIT {
        VirtualFree(region, 0, MEM_RELEASE);
    };
    DWORD old{};
    auto start = Clock::now();
    for (int round = 0; round < 16; ++round) {
        for (size_t page = 0; page < protect_pages; ++page) {
            VirtualProtect(region + page * PAGE, PAGE, PAGE_READONLY, &old);
            VirtualProtect(region + page * PAGE, PAGE, PAGE_READWRITE, &old);
        }
    }
    const double protect_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
        (16.0 * protect_pages * 2);

    static u8* fault_region;
    fault_region = region;
    const auto handler = [](EXCEPTION_POINTERS* ex) -> LONG {
        if (ex->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        u8* const address = reinterpret_cast<u8*>(ex->ExceptionRecord->ExceptionInformation[1]);
        if (address < fault_region || address >= fault_region + protect_pages * PAGE) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        DWORD previous{};
        VirtualProtect(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) & ~(PAGE - 1)),
                       PAGE, PAGE_READWRITE, &previous);
        return EXCEPTION_CONTINUE_EXECUTION;
    };
    void* const veh = AddVectoredExceptionHandler(1, handler);
    REQUIRE(veh != nullptr);
    SCOPE_EXIT {
        RemoveVectoredExceptionHandler(veh);
    };
    Clock::duration faults{};
    for (int round = 0; round < 16; ++round) {
        VirtualProtect(region, protect_pages * PAGE, PAGE_READONLY, &old);
        start = Clock::now();
        for (size_t page = 0; page < protect_pages; ++page) {
            *reinterpret_cast<volatile u32*>(region + page * PAGE) = static_cast<u32>(round);
        }
        faults += Clock::now() - start;
    }
    const double fault_ns =
        std::chrono::duration<double, std::nano>(faults).count() / (16.0 * protect_pages);
    std::printf("page_tracker bench: VirtualProtect of one page %.1f ns; store fault on a "
                "write-protected page + VEH + unprotect %.1f ns\n",
                protect_ns, fault_ns);
#endif
    std::fflush(stdout);
}

namespace {

/// What the line-mode write-back merge does to one unit of 8 guest words (nvn_wb_merge.comp): it
/// reads the unit, then stores it whole with the GPU's words in it. The window between the two is
/// about a microsecond on the GPU; here it lasts until may_store says the guest's store is in it
/// (or 5 s pass), so a store that lands in it is lost for certain unless the guard holds the store
/// back.
struct SimulatedLineMerge {
    u32* unit;
    std::function<bool()> may_store;
    std::atomic<bool> read_done{false};
    std::jthread thread;

    void Start(std::shared_ptr<void> token) {
        thread = std::jthread{[this, token = std::move(token)]() mutable {
            u32 words[8];
            for (u32 i = 0; i < 8; ++i) {
                words[i] = std::atomic_ref<u32>(unit[i]).load();
            }
            read_done = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!may_store() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            words[0] = 0x600d; // the words the GPU changed
            words[1] = 0x600d;
            for (u32 i = 0; i < 8; ++i) {
                std::atomic_ref<u32>(unit[i]).store(words[i]);
            }
            token.reset(); // the GPU is done with the submission
        }};
    }
};

} // Anonymous namespace

TEST_CASE("PageTracker + NVN line guard: a CPU write during a line merge survives",
          "[core][page_tracker][nvn][line_guard]") {
    using Vulkan::Nvn::LineGuardStats;
    using Vulkan::Nvn::MirrorLineGuard;
    for (const bool guarded : {false, true}) {
        INFO("guarded " << guarded);
        WithProcessMemory(4, [&](Core::Memory::Memory& memory, Common::PageTable&, u8* host) {
            auto& tracker = memory.PageTracker();
            const u64 id = tracker.Register(RIG_BASE, 4 * PAGE, false);
            SCOPE_EXIT {
                tracker.Unregister(id);
            };
            const auto stats = std::make_shared<LineGuardStats>();
            // The test thread is the guest; the merge thread stands in for the GPU.
            const auto guard = std::make_shared<MirrorLineGuard>(
                4 * PAGE, stats, std::thread::id{}, std::chrono::milliseconds{5000});
            tracker.SetAccessCallback(id, [&](u64 va, u64 size) {
                guard->Touched(va - RIG_BASE, va - RIG_BASE + size);
            });
            auto plan = guard->Record(0, 4 * PAGE);
            REQUIRE(plan.token != nullptr);
            if (guarded) {
                for (const auto& [first, last] : plan.arm) {
                    tracker.ArmAccessTrap(id, RIG_BASE + first * PAGE, (last - first + 1) * PAGE);
                }
            }
            u32* const unit = reinterpret_cast<u32*>(host) + 8; // words 8..15 of page 0
            // Control: the merge stores after the guest's store completed. Guarded: after the
            // guest's store reached the guard (the page went hot) and waits for the merge.
            std::atomic<bool> guest_stored{false};
            SimulatedLineMerge merge{.unit = unit, .may_store = [&] {
                                         return guarded ? guard->HotPage(0)
                                                        : guest_stored.load();
                                     }};
            merge.Start(std::move(plan.token));
            while (!merge.read_done.load()) {
                std::this_thread::yield();
            }
            // The guest stores to an unchanged word of the unit while the merge sits between its
            // read and its store.
            memory.Write32(RIG_BASE + (8 + 4) * 4, 0xc0de);
            guest_stored = true;
            merge.thread.join();
            if (guarded) {
                CHECK(std::atomic_ref<u32>(unit[4]).load() == 0xc0de);
                CHECK(stats->waits.load() == 1);
                CHECK(guard->HotPage(0));
                // The page is merged exactly from now on and its trap is gone.
                const auto next = guard->Record(0, 4 * PAGE);
                CHECK(next.runs.front().lines == false);
                CHECK(next.runs.front().end == PAGE);
                CHECK(next.runs.back().lines == true);
            } else {
                // Control: without the trap the same schedule loses the store.
                CHECK(std::atomic_ref<u32>(unit[4]).load() == 0);
            }
            CHECK(std::atomic_ref<u32>(unit[0]).load() == 0x600d);
            CHECK(std::atomic_ref<u32>(unit[1]).load() == 0x600d);
        });
    }
}
