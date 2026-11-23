// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/guest_memory.h"

using namespace NvnHleTest;
using VideoCommon::Nvn::GuestTag;
using VideoCommon::Nvn::GUEST_TAG_MAGIC;
using VideoCommon::Nvn::ObjectKind;
using VideoCommon::Nvn::TextureDesc;

namespace {

constexpr u64 POOL = Rig::Obj(0);
constexpr u64 BUILDER = Rig::Obj(1);
constexpr u64 QUEUE = Rig::Obj(2);
constexpr u64 CMD = Rig::Obj(3);
constexpr u64 BUFFER = Rig::Obj(4);
constexpr u64 BUFFER_COPY = Rig::Obj(5);
constexpr u64 TEXTURE_POOL = Rig::Obj(6);
constexpr u64 SAMPLER_POOL = Rig::Obj(7);
constexpr u64 TEXTURE = Rig::Obj(8);
constexpr u64 SYNC = Rig::Obj(9);
constexpr u64 EVENT = Rig::Obj(10);
constexpr u64 PROGRAM = Rig::Obj(11);
constexpr u64 PROGRAM_2 = Rig::Obj(12);
constexpr u64 POOL_2 = Rig::Obj(13);
constexpr u64 SAMPLER = Rig::Obj(14);
constexpr u64 UNINITIALIZED = Rig::Obj(50);

constexpr u64 SCRATCH = Rig::SCRATCH;
constexpr u64 STORAGE = Rig::STORAGE;
constexpr u64 STORAGE_SIZE = Rig::STORAGE_SIZE;

/// Initializes a memory pool through the HLE bodies (builder at BUILDER).
bool InitPool(Rig& rig, u64 pool, u64 storage, u64 size, u32 flags) {
    rig.Call("nvnMemoryPoolBuilderSetDefaults", {Rig::I(BUILDER)});
    rig.Call("nvnMemoryPoolBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0xDE)});
    rig.Call("nvnMemoryPoolBuilderSetStorage", {Rig::I(BUILDER), Rig::I(storage), Rig::I(size)});
    rig.Call("nvnMemoryPoolBuilderSetFlags", {Rig::I(BUILDER), Rig::I(flags)});
    return rig.Call("nvnMemoryPoolInitialize", {Rig::I(pool), Rig::I(BUILDER)}) == 1;
}

/// Initializes a buffer through the HLE bodies (builder at BUILDER).
bool InitBuffer(Rig& rig, u64 buffer, u64 pool, u64 offset, u64 size) {
    rig.Call("nvnBufferBuilderSetDefaults", {Rig::I(BUILDER)});
    rig.Call("nvnBufferBuilderSetStorage", {Rig::I(BUILDER), Rig::I(pool), Rig::I(offset), Rig::I(size)});
    return rig.Call("nvnBufferInitialize", {Rig::I(buffer), Rig::I(BUILDER)}) == 1;
}

TextureDesc MakeTextureDesc() {
    TextureDesc desc{};
    desc.target = 1;
    desc.format = 0x25;
    desc.width = 64;
    desc.height = 64;
    desc.depth = 1;
    desc.levels = 1;
    return desc;
}

/// Submits an empty command list on QUEUE and returns the submission the sink received. Needs the
/// pool at POOL and the queue initialized.
const VideoCommon::Nvn::Submission& SubmitEmptyList(Rig& rig) {
    REQUIRE(rig.recorder.CommandBufferInitialize(CMD));
    rig.recorder.AddCommandMemory(CMD, POOL, 0x1000, 0x1000);
    rig.recorder.BeginRecording(CMD);
    const u64 handle = rig.recorder.EndRecording(CMD);
    rig.memory.Put<u64>(SCRATCH + 0x4000, handle);
    const size_t before = rig.sink.submissions.size();
    rig.Call("nvnQueueSubmitCommands", {Rig::I(QUEUE), Rig::I(1), Rig::I(SCRATCH + 0x4000)});
    REQUIRE(rig.sink.submissions.size() == before + 1);
    return rig.sink.submissions.back();
}

void InitQueueAndPool(Rig& rig) {
    REQUIRE(InitPool(rig, POOL, STORAGE, STORAGE_SIZE, 0));
    REQUIRE(rig.Call("nvnQueueInitialize", {Rig::I(QUEUE), Rig::I(BUILDER)}) == 1);
}

/// An NVN2 control section (version 0xc9) of `stage` with a code area of `code_size` bytes.
void PutControl(Rig& rig, u64 va, u16 stage, u32 code_size) {
    std::array<u8, 0x100> control{};
    const u32 magic = VideoCommon::Nvn::NVN2_CONTROL_MAGIC;
    const u32 version = VideoCommon::Nvn::NVN2_CONTROL_VERSION;
    std::memcpy(control.data(), &magic, 4);
    std::memcpy(control.data() + 4, &version, 4);
    std::memcpy(control.data() + 0x44, &code_size, 4);
    std::memcpy(control.data() + 0x56, &stage, 2);
    rig.memory.Put(va, control);
}

std::vector<u8> PutCode(Rig& rig, u64 va, u8 fill) {
    std::array<u8, 16> code{};
    code.fill(fill);
    rig.memory.Put(va, code);
    return std::vector<u8>(code.begin(), code.end());
}

} // namespace

// ---- Memory pools ----------------------------------------------------------------------------------------

TEST_CASE("nvn hle pool: builder in place layout (nxnative pool_builder_t)", "[nvn]") {
    Rig rig;
    std::array<u8, 0x40> garbage;
    garbage.fill(0xFF);
    rig.memory.Put(BUILDER, garbage);

    rig.Call("nvnMemoryPoolBuilderSetDefaults", {Rig::I(BUILDER)});
    // SDK: storage and size are reset, flags become CPU_NO_ACCESS | GPU_CACHED (0x21); the device
    // is left as it was; the rest of the builder is not ours.
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == ~0ull);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x10) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x18) == 0x21);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x20) == ~0ull);

    rig.Call("nvnMemoryPoolBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0x1122334455667788ull)});
    rig.Call("nvnMemoryPoolBuilderSetStorage", {Rig::I(BUILDER), Rig::I(STORAGE), Rig::I(0x40000)});
    rig.Call("nvnMemoryPoolBuilderSetFlags", {Rig::I(BUILDER), Rig::I(0x24)});
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == 0x1122334455667788ull); // dev
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == STORAGE);               // mem
    CHECK(rig.memory.Get<u64>(BUILDER + 0x10) == 0x40000);               // size
    CHECK(rig.memory.Get<u32>(BUILDER + 0x18) == 0x24);                  // flags
    CHECK(rig.memory.Get<u32>(BUILDER + 0x1C) == 0);

    CHECK(rig.Call("nvnMemoryPoolBuilderGetDevice", {Rig::I(BUILDER)}) == 0x1122334455667788ull);
    CHECK(rig.Call("nvnMemoryPoolBuilderGetMemory", {Rig::I(BUILDER)}) == STORAGE);
    CHECK(rig.Call("nvnMemoryPoolBuilderGetSize", {Rig::I(BUILDER)}) == 0x40000);
    CHECK(rig.Call("nvnMemoryPoolBuilderGetFlags", {Rig::I(BUILDER)}) == 0x24);
}

TEST_CASE("nvn hle pool: initialize reaches the sink, getters, map and finalize", "[nvn]") {
    Rig rig;
    REQUIRE(InitPool(rig, POOL, STORAGE, 0x40000, 0x24));
    REQUIRE(rig.sink.pools.size() == 1);
    CHECK(rig.sink.pools[0].guest_address == STORAGE);
    CHECK(rig.sink.pools[0].size == 0x40000);
    CHECK(rig.sink.pools[0].flags == 0x24);

    const GuestTag tag = rig.memory.Get<GuestTag>(POOL);
    CHECK(tag.magic == GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(ObjectKind::MemoryPool));
    CHECK(tag.serial == rig.sink.pools[0].serial);

    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(POOL)}) == 0x24);
    CHECK(rig.Call("nvnMemoryPoolGetSize", {Rig::I(POOL)}) == 0x40000);
    CHECK(rig.Call("nvnMemoryPoolGetBufferAddress", {Rig::I(POOL)}) == STORAGE);
    CHECK(rig.Call("nvnMemoryPoolMap", {Rig::I(POOL)}) == STORAGE);
    // A flush tells the replay which bytes the CPU wrote; an invalidate has nothing to say.
    rig.Call("nvnMemoryPoolFlushMappedRange", {Rig::I(POOL), Rig::I(0x100), Rig::I(0x40)});
    REQUIRE(rig.sink.cpu_writes.size() == 1);
    CHECK(rig.sink.cpu_writes[0] == std::make_pair(STORAGE + 0x100, u64{0x40}));
    rig.Call("nvnMemoryPoolInvalidateMappedRange", {Rig::I(POOL), Rig::I(0x100), Rig::I(0x40)});
    CHECK(rig.sink.cpu_writes.size() == 1);
    // A range that runs past the pool is clamped to it; one outside it is ignored.
    rig.Call("nvnMemoryPoolFlushMappedRange", {Rig::I(POOL), Rig::I(0x3FF00), Rig::I(0x1000)});
    REQUIRE(rig.sink.cpu_writes.size() == 2);
    CHECK(rig.sink.cpu_writes[1] == std::make_pair(STORAGE + 0x3FF00, u64{0x100}));
    rig.Call("nvnMemoryPoolFlushMappedRange", {Rig::I(POOL), Rig::I(0x40000), Rig::I(0x10)});
    CHECK(rig.sink.cpu_writes.size() == 2);
    rig.Call("nvnMemoryPoolSetDebugLabel", {Rig::I(POOL), Rig::I(SCRATCH)});

    // CPU_NO_ACCESS (0x1): Map answers NULL, the GPU address stays.
    REQUIRE(InitPool(rig, POOL_2, STORAGE + 0x80000, 0x10000, 0x1 | 0x10));
    CHECK(rig.Call("nvnMemoryPoolMap", {Rig::I(POOL_2)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolGetBufferAddress", {Rig::I(POOL_2)}) == STORAGE + 0x80000);
    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(POOL_2)}) == 0x11);

    // SDK: a non-virtual pool needs 4 KiB aligned storage and a size that is a non-zero multiple
    // of 4 KiB; a PHYSICAL pool a multiple of the 64 KiB pool page; a VIRTUAL pool no storage and
    // a non-zero multiple of the pool page. Nothing reaches the sink.
    const size_t pools = rig.sink.pools.size();
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), 0, 0x1000, 0));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), STORAGE + 0x800, 0x1000, 0));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), STORAGE, 0, 0));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), STORAGE, 0x1800, 0));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), STORAGE, 0x8000, 0x100));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), STORAGE, 0x10000, 0x200));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), 0, 0, 0x200));
    CHECK_FALSE(InitPool(rig, Rig::Obj(20), 0, 0x8000, 0x200));
    CHECK(rig.sink.pools.size() == pools);
    CHECK(rig.memory.Get<GuestTag>(Rig::Obj(20)).magic != GUEST_TAG_MAGIC);

    rig.Call("nvnMemoryPoolFinalize", {Rig::I(POOL)});
    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(POOL)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolGetSize", {Rig::I(POOL)}) == 0);
    CHECK(rig.memory.Get<GuestTag>(POOL).magic != GUEST_TAG_MAGIC);
    // The finalized pool's neighbour is untouched.
    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(POOL_2)}) == 0x11);
}

TEST_CASE("nvn hle pool: getters on a zeroed pool object answer 0 without a tag mismatch", "[nvn]") {
    // The SDK getters read the guest's pool object and never check it was initialised: an object
    // that is still zero (DK queries embedded pool objects before it initializes them) reads as an
    // empty pool. Only a tag that is neither valid nor zero is a mismatch.
    Rig rig;
    REQUIRE(InitPool(rig, POOL, STORAGE, STORAGE_SIZE, 0));
    const u64 before = rig.recorder.Stats().tag_mismatch;
    const u64 zeroed = Rig::Obj(21);
    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(zeroed)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolGetSize", {Rig::I(zeroed)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolMap", {Rig::I(zeroed)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolGetBufferAddress", {Rig::I(zeroed)}) == 0);
    CHECK(rig.recorder.Stats().tag_mismatch == before);

    GuestTag foreign{};
    foreign.magic = GUEST_TAG_MAGIC ^ 0x1u;
    rig.memory.Put(Rig::Obj(22), foreign);
    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(Rig::Obj(22))}) == 0);
    CHECK(rig.recorder.Stats().tag_mismatch == before + 1);
}

TEST_CASE("nvn hle pool: virtual pools register without storage and MapVirtual validates", "[nvn]") {
    Rig rig;
    REQUIRE(InitPool(rig, POOL, STORAGE, STORAGE_SIZE, 0));
    // A virtual pool (0x200) has no storage: the Recorder gives it a GPU address range of its own
    // (above every guest address) that nothing else uses, and the replay maps it with MapVirtual.
    constexpr u64 VIRTUAL_SIZE = 0x200000;
    REQUIRE(InitPool(rig, POOL_2, 0, VIRTUAL_SIZE, 0x200));
    REQUIRE(rig.sink.pools.size() == 2);
    CHECK(rig.sink.pools[1].guest_address >= 0x7D00'0000'0000'0000ULL);
    CHECK(rig.sink.pools[1].guest_address % 0x200000 == 0);
    CHECK(rig.Call("nvnMemoryPoolGetBufferAddress", {Rig::I(POOL_2)}) ==
          rig.sink.pools[1].guest_address);
    CHECK(rig.Call("nvnMemoryPoolMap", {Rig::I(POOL_2)}) == 0); // no CPU pointer
    CHECK(rig.sink.pools[1].size == VIRTUAL_SIZE);
    // The SDK stores a virtual pool as CPU_NO_ACCESS (and GPU_CACHED when no GPU mode is given).
    CHECK(rig.sink.pools[1].flags == 0x221);
    CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(POOL_2)}) == 0x221);
    CHECK(rig.Call("nvnMemoryPoolGetSize", {Rig::I(POOL_2)}) == VIRTUAL_SIZE);

    // NVNmappingRequest: {physical pool, physical offset, virtual offset, size, storage class}.
    const auto put_request = [&](u32 index, u64 pool, s64 physical, s64 virtual_offset, u64 size) {
        const u64 request = SCRATCH + 0x28ull * index;
        rig.memory.Put<u64>(request + 0x00, pool);
        rig.memory.Put<s64>(request + 0x08, physical);
        rig.memory.Put<s64>(request + 0x10, virtual_offset);
        rig.memory.Put<u64>(request + 0x18, size);
        rig.memory.Put<u64>(request + 0x20, 0);
    };
    put_request(0, POOL, 0x30000, 0, 0x10000);
    put_request(1, POOL, 0x20000, 0x10000, 0x10000);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(2), Rig::I(SCRATCH)}) == 1);
    REQUIRE(rig.sink.virtual_maps.size() == 1);
    CHECK(rig.sink.virtual_maps[0].first == rig.sink.pools[1].serial);
    REQUIRE(rig.sink.virtual_maps[0].second.size() == 2);
    CHECK(rig.sink.virtual_maps[0].second[1].virtual_offset == 0x10000);
    CHECK(rig.sink.virtual_maps[0].second[1].physical_offset == 0x20000);
    CHECK(rig.sink.virtual_maps[0].second[1].size == 0x10000);
    CHECK(rig.sink.virtual_maps[0].second[1].physical_pool_serial == rig.sink.pools[0].serial);

    put_request(0, 0, 0, 0x10000, 0x10000); // a null physical pool unmaps
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 1);
    REQUIRE(rig.sink.virtual_maps.size() == 2);
    CHECK(rig.sink.virtual_maps[1].second[0].physical_pool_serial == 0);

    put_request(0, UNINITIALIZED, 0, 0, 0x10000); // not a memory pool
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    put_request(0, POOL, 0, VIRTUAL_SIZE - 0x10000, 0x20000); // leaves the virtual pool
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    put_request(0, POOL, STORAGE_SIZE - 0x10000, 0, 0x20000); // leaves the physical pool
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    // A size near 2^64 must not wrap around the range checks.
    put_request(0, POOL, 0x10000, 0x20000, ~u64{0} - 0xFFFF);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    put_request(0, 0, 0, 0x20000, ~u64{0} - 0xFFFF); // unmap
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    // SDK: offsets and sizes are multiples of the 64 KiB pool page, or the request fails.
    put_request(0, POOL, 0x1000, 0, 0x10000);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    put_request(0, POOL, 0, 0x8000, 0x10000);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    put_request(0, 0, 0, 0, 0x1000);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(SCRATCH)}) == 0);
    CHECK(rig.sink.virtual_maps.size() == 2);
    // SDK: no requests (count <= 0) succeed on a virtual pool with nothing to do; a pool that is
    // not virtual (or not a pool) fails.
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(static_cast<u64>(-1)), Rig::I(SCRATCH)}) == 1);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(0), Rig::I(SCRATCH)}) == 1);
    CHECK(rig.sink.virtual_maps.size() == 2);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL), Rig::I(0), Rig::I(SCRATCH)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(UNINITIALIZED), Rig::I(0), Rig::I(SCRATCH)}) == 0);
    // SDK: requests go to the kernel in batches of 512; a failing batch leaves the earlier ones
    // mapped. 512 good unmaps, then a misaligned request in the second batch.
    for (u32 i = 0; i < 512; ++i) {
        put_request(i, 0, 0, 0x10000, 0x10000);
    }
    put_request(512, 0, 0, 0x1000, 0x10000);
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(513), Rig::I(SCRATCH)}) == 0);
    REQUIRE(rig.sink.virtual_maps.size() == 3);
    CHECK(rig.sink.virtual_maps[2].second.size() == 512);
    // A request array the guest cannot read fails without mapping anything.
    CHECK(rig.Call("nvnMemoryPoolMapVirtual", {Rig::I(POOL_2), Rig::I(1), Rig::I(0x10)}) == 0);
    CHECK(rig.sink.virtual_maps.size() == 3);
}

TEST_CASE("nvn hle pool: initialize stores the SDK's normalized flags", "[nvn]") {
    Rig rig;
    const auto flags_of = [&](u32 builder_flags) {
        const u64 pool = Rig::Obj(20);
        REQUIRE(InitPool(rig, pool, STORAGE, 0x10000, builder_flags));
        const u64 flags = rig.Call("nvnMemoryPoolGetFlags", {Rig::I(pool)});
        CHECK((rig.sink.pools.back().flags & 0x1fffff) == flags);
        rig.Call("nvnMemoryPoolFinalize", {Rig::I(pool)});
        return flags;
    };
    // The flags Donkey Kong Bananza passes are kept as they are.
    for (const u32 flags : {0x12u, 0x22u, 0x24u, 0x62u, 0x64u, 0x18cu}) {
        CHECK(flags_of(flags) == flags);
    }
    // No CPU or GPU mode: CPU_UNCACHED and GPU_CACHED.
    CHECK(flags_of(0) == 0x22);
    CHECK(flags_of(0x80) == 0xA2);
    // One CPU mode: UNCACHED over CACHED over NO_ACCESS; one GPU mode: CACHED over UNCACHED over
    // NO_ACCESS.
    CHECK(flags_of(0x1 | 0x2 | 0x4) == (0x2 | 0x20));
    CHECK(flags_of(0x1 | 0x4) == (0x4 | 0x20));
    CHECK(flags_of(0x8 | 0x10 | 0x20) == (0x2 | 0x20));
    CHECK(flags_of(0x8 | 0x10) == (0x2 | 0x10));
    CHECK(flags_of(0x1 | 0x8) == (0x1 | 0x8));
    // PHYSICAL has no GPU access.
    CHECK(flags_of(0x100 | 0x4 | 0x20) == (0x100 | 0x4 | 0x8));
    // GetFlags hides bits 21 and up (3.2.0 mask 0x1fffff); the bits from 6 up are kept as given.
    CHECK(flags_of(0x00400000 | 0x200000 | 0x1000 | 0x22) == (0x1000 | 0x22));
    // A device with flag bit 15 makes CPU-uncached pools CPU-cached.
    rig.runtime.device.flags = 0x8000;
    CHECK(flags_of(0x22) == 0x24);
    CHECK(flags_of(0) == 0x24);
    CHECK(flags_of(0x21) == 0x21);
    rig.runtime.device.flags = 0;
    // The 1.0.0-generation sdk (698 GetProcAddress names) masks with 0x3fffff: bit 21 shows.
    rig.runtime.device.sdk_proc_names = 698;
    {
        const u64 pool = Rig::Obj(20);
        REQUIRE(InitPool(rig, pool, STORAGE, 0x10000, 0x00400000 | 0x200000 | 0x22));
        CHECK(rig.Call("nvnMemoryPoolGetFlags", {Rig::I(pool)}) == (0x200000 | 0x22));
        rig.Call("nvnMemoryPoolFinalize", {Rig::I(pool)});
    }
    rig.runtime.device.sdk_proc_names = 726;
    CHECK(flags_of(0x200000 | 0x22) == 0x22);
    rig.runtime.device.sdk_proc_names = 0;

    // SDK: a PHYSICAL pool has no GPU address; its CPU pointer stays (CPU access given).
    REQUIRE(InitPool(rig, POOL, STORAGE, 0x20000, 0x100 | 0x4));
    CHECK(rig.Call("nvnMemoryPoolGetBufferAddress", {Rig::I(POOL)}) == 0);
    CHECK(rig.Call("nvnMemoryPoolMap", {Rig::I(POOL)}) == STORAGE);
    CHECK(rig.Call("nvnMemoryPoolGetSize", {Rig::I(POOL)}) == 0x20000);
}

// ---- Buffers ---------------------------------------------------------------------------------------------

TEST_CASE("nvn hle buffer: builder in place layout (nxnative buf_builder_t)", "[nvn]") {
    Rig rig;
    std::array<u8, 0x40> garbage;
    garbage.fill(0xFF);
    rig.memory.Put(BUILDER, garbage);
    rig.Call("nvnBufferBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0xDEull << 8)});
    rig.Call("nvnBufferBuilderSetStorage",
             {Rig::I(BUILDER), Rig::I(POOL), Rig::I(0x1000), Rig::I(0x200)});
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == 0xDE00); // dev
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == POOL);   // pool
    CHECK(rig.memory.Get<u64>(BUILDER + 0x10) == 0x1000); // off
    CHECK(rig.memory.Get<u64>(BUILDER + 0x18) == 0x200);  // size
    CHECK(rig.Call("nvnBufferBuilderGetDevice", {Rig::I(BUILDER)}) == 0xDE00);
    CHECK(rig.Call("nvnBufferBuilderGetMemoryPool", {Rig::I(BUILDER)}) == POOL);
    CHECK(rig.Call("nvnBufferBuilderGetMemoryOffset", {Rig::I(BUILDER)}) == 0x1000);
    CHECK(rig.Call("nvnBufferBuilderGetSize", {Rig::I(BUILDER)}) == 0x200);

    // SetDefaults keeps the device (nxnative) and zeroes the rest of the 0x20 bytes.
    rig.Call("nvnBufferBuilderSetDefaults", {Rig::I(BUILDER)});
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == 0xDE00);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x10) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x18) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x20) == ~0ull);
}

TEST_CASE("nvn hle buffer: initialize writes tag, offset and size; address follows the guest", "[nvn]") {
    Rig rig;
    REQUIRE(InitPool(rig, POOL, STORAGE, STORAGE_SIZE, 0));
    REQUIRE(InitBuffer(rig, BUFFER, POOL, 0x1000, 0x200));

    const GuestTag tag = rig.memory.Get<GuestTag>(BUFFER);
    CHECK(tag.magic == GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(ObjectKind::Buffer));
    CHECK(rig.memory.Get<u64>(BUFFER + 0x18) == 0x1000);
    CHECK(rig.memory.Get<u64>(BUFFER + 0x20) == 0x200);

    CHECK(rig.Call("nvnBufferGetAddress", {Rig::I(BUFFER)}) == STORAGE + 0x1000);
    CHECK(rig.Call("nvnBufferMap", {Rig::I(BUFFER)}) == STORAGE + 0x1000);
    CHECK(rig.Call("nvnBufferGetSize", {Rig::I(BUFFER)}) == 0x200);
    CHECK(rig.Call("nvnBufferGetMemoryOffset", {Rig::I(BUFFER)}) == 0x1000);
    CHECK(rig.Call("nvnBufferGetMemoryPool", {Rig::I(BUFFER)}) == POOL);
    CHECK(rig.Call("nvnBufferGetDebugID", {Rig::I(BUFFER)}) == tag.serial);
    rig.Call("nvnBufferFlushMappedRange", {Rig::I(BUFFER), Rig::I(0), Rig::I(0x10)});
    rig.Call("nvnBufferInvalidateMappedRange", {Rig::I(BUFFER), Rig::I(0), Rig::I(0x10)});
    rig.Call("nvnBufferSetDebugLabel", {Rig::I(BUFFER), Rig::I(SCRATCH)});

    // The game's ring allocators rewrite offset (+0x18) and size (+0x20): every query follows.
    rig.memory.Put<u64>(BUFFER + 0x18, 0x2000);
    rig.memory.Put<u64>(BUFFER + 0x20, 0x40);
    CHECK(rig.Call("nvnBufferGetAddress", {Rig::I(BUFFER)}) == STORAGE + 0x2000);
    CHECK(rig.Call("nvnBufferMap", {Rig::I(BUFFER)}) == STORAGE + 0x2000);
    CHECK(rig.Call("nvnBufferGetSize", {Rig::I(BUFFER)}) == 0x40);
    CHECK(rig.Call("nvnBufferGetMemoryOffset", {Rig::I(BUFFER)}) == 0x2000);

    // Games copy buffers by value: the tag travels, the copy reads its own offset.
    std::array<u8, 0x30> bytes{};
    REQUIRE(rig.memory.Read(BUFFER, bytes.data(), bytes.size()));
    rig.memory.Put(BUFFER_COPY, bytes);
    rig.memory.Put<u64>(BUFFER_COPY + 0x18, 0x3000);
    CHECK(rig.Call("nvnBufferGetAddress", {Rig::I(BUFFER_COPY)}) == STORAGE + 0x3000);
    CHECK(rig.Call("nvnBufferGetMemoryPool", {Rig::I(BUFFER_COPY)}) == POOL);

    rig.Call("nvnBufferFinalize", {Rig::I(BUFFER)});
    CHECK(rig.Call("nvnBufferGetAddress", {Rig::I(BUFFER)}) == 0);
}

TEST_CASE("nvn hle buffer: CPU-inaccessible pools map to NULL, bad pools fail", "[nvn]") {
    Rig rig;
    REQUIRE(InitPool(rig, POOL, STORAGE, STORAGE_SIZE, 0x1)); // CPU_NO_ACCESS
    REQUIRE(InitBuffer(rig, BUFFER, POOL, 0x100, 0x40));
    CHECK(rig.Call("nvnBufferMap", {Rig::I(BUFFER)}) == 0);
    CHECK(rig.Call("nvnBufferGetAddress", {Rig::I(BUFFER)}) == STORAGE + 0x100);

    // A pool that was never initialized (or finalized) cannot back a buffer.
    CHECK_FALSE(InitBuffer(rig, BUFFER_COPY, UNINITIALIZED, 0, 0x40));
    rig.Call("nvnMemoryPoolFinalize", {Rig::I(POOL)});
    CHECK_FALSE(InitBuffer(rig, BUFFER_COPY, POOL, 0, 0x40));
}

// ---- Descriptor pools ------------------------------------------------------------------------------------

TEST_CASE("nvn hle texture pool: initialize, getters and registrations reach the descriptor table",
          "[nvn]") {
    Rig rig;
    InitQueueAndPool(rig);
    REQUIRE(rig.recorder.TextureInitialize(TEXTURE, MakeTextureDesc(), POOL, 0));
    REQUIRE(rig.sink.textures.size() == 1);

    CHECK(rig.Call("nvnTexturePoolInitialize",
                   {Rig::I(TEXTURE_POOL), Rig::I(POOL), Rig::I(0x20000), Rig::I(100)}) == 1);
    CHECK(rig.Call("nvnTexturePoolGetMemoryPool", {Rig::I(TEXTURE_POOL)}) == POOL);
    CHECK(rig.Call("nvnTexturePoolGetMemoryOffset", {Rig::I(TEXTURE_POOL)}) == 0x20000);
    CHECK(rig.Call("nvnTexturePoolGetSize", {Rig::I(TEXTURE_POOL)}) == 100);
    rig.Call("nvnTexturePoolSetDebugLabel", {Rig::I(TEXTURE_POOL), Rig::I(SCRATCH)});
    // A pool without entries or on an unknown memory pool is refused.
    CHECK(rig.Call("nvnTexturePoolInitialize",
                   {Rig::I(Rig::Obj(21)), Rig::I(POOL), Rig::I(0), Rig::I(0)}) == 0);
    CHECK(rig.Call("nvnTexturePoolInitialize",
                   {Rig::I(Rig::Obj(21)), Rig::I(UNINITIALIZED), Rig::I(0), Rig::I(16)}) == 0);

    // id 5: the texture; id 6: a view (levels 1..2); id 7: the same view as a storage image.
    std::array<u8, 16> view{};
    view[0] = VideoCommon::Nvn::GUEST_VIEW_SET_LEVELS;
    view[2] = 1; // base_level
    view[3] = 2; // nlevels
    rig.memory.Put(SCRATCH + 0x100, view);
    rig.Call("nvnTexturePoolRegisterTexture",
             {Rig::I(TEXTURE_POOL), Rig::I(5), Rig::I(TEXTURE), Rig::I(0)});
    rig.Call("nvnTexturePoolRegisterTexture",
             {Rig::I(TEXTURE_POOL), Rig::I(6), Rig::I(TEXTURE), Rig::I(SCRATCH + 0x100)});
    rig.Call("nvnTexturePoolRegisterImage",
             {Rig::I(TEXTURE_POOL), Rig::I(7), Rig::I(TEXTURE), Rig::I(SCRATCH + 0x100)});
    REQUIRE(rig.sink.textures.size() == 3);
    CHECK(rig.sink.textures[1].view_of == rig.sink.textures[0].serial);
    CHECK(rig.sink.textures[1].view.set_levels == 1);
    CHECK(rig.sink.textures[1].view.base_level == 1);
    CHECK(rig.sink.textures[1].view.nlevels == 2);
    CHECK(rig.sink.textures[1].view.is_image == 0);
    CHECK(rig.sink.textures[2].view.is_image == 1);

    // An id outside the pool is counted, not registered.
    const u64 bad = rig.recorder.Stats().bad_argument;
    rig.Call("nvnTexturePoolRegisterTexture",
             {Rig::I(TEXTURE_POOL), Rig::I(100), Rig::I(TEXTURE), Rig::I(0)});
    CHECK(rig.recorder.Stats().bad_argument == bad + 1);

    const auto& submission = SubmitEmptyList(rig);
    REQUIRE(submission.texture_pool_snapshot.size() == 8);
    CHECK(submission.texture_pool_snapshot[5] == rig.sink.textures[0].serial);
    CHECK(submission.texture_pool_snapshot[6] == rig.sink.textures[1].serial);
    CHECK(submission.texture_pool_snapshot[7] == rig.sink.textures[2].serial);
    CHECK(submission.texture_pool_snapshot[4] == 0);

    rig.Call("nvnTexturePoolFinalize", {Rig::I(TEXTURE_POOL)});
    CHECK(rig.Call("nvnTexturePoolGetSize", {Rig::I(TEXTURE_POOL)}) == 0);
}

TEST_CASE("nvn hle sampler pool: registrations, sampler builder decoded in place", "[nvn]") {
    Rig rig;
    InitQueueAndPool(rig);
    CHECK(rig.Call("nvnSamplerPoolInitialize",
                   {Rig::I(SAMPLER_POOL), Rig::I(POOL), Rig::I(0x30000), Rig::I(16)}) == 1);
    CHECK(rig.Call("nvnSamplerPoolGetMemoryPool", {Rig::I(SAMPLER_POOL)}) == POOL);
    CHECK(rig.Call("nvnSamplerPoolGetMemoryOffset", {Rig::I(SAMPLER_POOL)}) == 0x30000);
    CHECK(rig.Call("nvnSamplerPoolGetSize", {Rig::I(SAMPLER_POOL)}) == 16);
    rig.Call("nvnSamplerPoolSetDebugLabel", {Rig::I(SAMPLER_POOL), Rig::I(SCRATCH)});
    // SDK: ids 0..2 hold reserved samplers (clamp to edge, LOD [0, 1000]); see the next test.
    REQUIRE(rig.sink.samplers.size() == 3);

    // NVNsamplerBuilder in place: {dev; samp_par {u8 minf, magf, ws, wt, wr, cmp_mode, cmp_func,
    // reduction; f32 lod_min, lod_max, lod_bias, aniso, border[4]}}.
    const u64 builder = Rig::Obj(30);
    rig.memory.Put<u64>(builder, 0xDE);
    const std::array<u8, 8> small{2, 1, 3, 4, 5, 1, 6, 2};
    rig.memory.Put(builder + 8, small);
    const std::array<float, 8> floats{-1.0f, 8.0f, 0.5f, 16.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    rig.memory.Put(builder + 16, floats);
    rig.Call("nvnSamplerPoolRegisterSamplerBuilder",
             {Rig::I(SAMPLER_POOL), Rig::I(3), Rig::I(builder)});
    REQUIRE(rig.sink.samplers.size() == 4);
    const VideoCommon::Nvn::SamplerDesc& desc = rig.sink.samplers[3];
    CHECK(desc.min_filter == 2);
    CHECK(desc.mag_filter == 1);
    CHECK(desc.wrap_s == 3);
    CHECK(desc.wrap_t == 4);
    CHECK(desc.wrap_r == 5);
    CHECK(desc.compare_mode == 1);
    CHECK(desc.compare_func == 6);
    CHECK(desc.reduction == 2);
    CHECK(desc.lod_min == -1.0f);
    CHECK(desc.lod_max == 8.0f);
    CHECK(desc.lod_bias == 0.5f);
    CHECK(desc.max_anisotropy == 16.0f);
    CHECK(desc.border[0] == 0.25f);
    CHECK(desc.border[1] == 0.5f);
    CHECK(desc.border[2] == 0.75f);
    CHECK(desc.border[3] == 1.0f);

    // The same parameters share one host sampler; a sampler object registers by its serial.
    rig.Call("nvnSamplerPoolRegisterSamplerBuilder",
             {Rig::I(SAMPLER_POOL), Rig::I(4), Rig::I(builder)});
    CHECK(rig.sink.samplers.size() == 4);
    VideoCommon::Nvn::SamplerDesc object_desc{};
    object_desc.min_filter = 7;
    REQUIRE(rig.recorder.SamplerInitialize(SAMPLER, object_desc));
    REQUIRE(rig.sink.samplers.size() == 5);
    rig.Call("nvnSamplerPoolRegisterSampler", {Rig::I(SAMPLER_POOL), Rig::I(2), Rig::I(SAMPLER)});
    // Out of range ids are ignored.
    rig.Call("nvnSamplerPoolRegisterSamplerBuilder",
             {Rig::I(SAMPLER_POOL), Rig::I(16), Rig::I(builder)});

    const auto& submission = SubmitEmptyList(rig);
    REQUIRE(submission.sampler_pool_snapshot.size() == 5);
    CHECK(submission.sampler_pool_snapshot[3] == rig.sink.samplers[3].serial);
    CHECK(submission.sampler_pool_snapshot[4] == rig.sink.samplers[3].serial);
    CHECK(submission.sampler_pool_snapshot[2] == rig.sink.samplers[4].serial);
    CHECK(submission.sampler_pool_snapshot[1] == rig.sink.samplers[1].serial);

    rig.Call("nvnSamplerPoolFinalize", {Rig::I(SAMPLER_POOL)});
    CHECK(rig.Call("nvnSamplerPoolGetSize", {Rig::I(SAMPLER_POOL)}) == 0);
}

TEST_CASE("nvn hle sampler pool: initialize registers the SDK's reserved samplers 0..2", "[nvn]") {
    Rig rig;
    InitQueueAndPool(rig);
    const u64 bad = rig.recorder.Stats().bad_argument;
    REQUIRE(rig.Call("nvnSamplerPoolInitialize",
                     {Rig::I(SAMPLER_POOL), Rig::I(POOL), Rig::I(0x30000), Rig::I(16)}) == 1);
    REQUIRE(rig.sink.samplers.size() == 3);
    for (u32 id = 0; id < 3; ++id) {
        const VideoCommon::Nvn::SamplerDesc& desc = rig.sink.samplers[id];
        const u8 filter = id == 2 ? 0 : 1; // NEAREST for id 2, LINEAR otherwise
        CHECK(desc.min_filter == filter);
        CHECK(desc.mag_filter == filter);
        CHECK(desc.wrap_s == 7); // CLAMP_TO_EDGE
        CHECK(desc.wrap_t == 7);
        CHECK(desc.wrap_r == 7);
        CHECK(desc.compare_mode == 0);
        CHECK(desc.compare_func == 2); // LESS
        CHECK(desc.reduction == 0);
        CHECK(desc.lod_min == 0.0f);
        CHECK(desc.lod_max == 1000.0f);
        CHECK(desc.lod_bias == 0.0f);
        CHECK(desc.max_anisotropy == (id == 0 ? 1.0f : 2.0f));
        for (const f32 border : desc.border) {
            CHECK(border == 0.0f);
        }
    }
    const auto& submission = SubmitEmptyList(rig);
    REQUIRE(submission.sampler_pool_snapshot.size() >= 3);
    for (u32 id = 0; id < 3; ++id) {
        CHECK(submission.sampler_pool_snapshot[id] == rig.sink.samplers[id].serial);
    }

    // A second pool shares the host samplers; a pool with fewer than three entries registers the
    // ids it has (the SDK writes all three descriptors, past the pool's end).
    REQUIRE(rig.Call("nvnSamplerPoolInitialize",
                     {Rig::I(Rig::Obj(31)), Rig::I(POOL), Rig::I(0x40000), Rig::I(2)}) == 1);
    CHECK(rig.sink.samplers.size() == 3);
    CHECK(rig.recorder.Stats().bad_argument == bad);
    // A pool the Recorder refuses registers nothing.
    CHECK(rig.Call("nvnSamplerPoolInitialize",
                   {Rig::I(Rig::Obj(32)), Rig::I(UNINITIALIZED), Rig::I(0), Rig::I(16)}) == 0);
}

// ---- Programs --------------------------------------------------------------------------------------------

TEST_CASE("nvn hle program: SetShaders reads NVNshaderData and takes stages from the control section",
          "[nvn]") {
    Rig rig;
    REQUIRE(rig.Call("nvnProgramInitialize", {Rig::I(PROGRAM), Rig::I(0xDE)}) == 1);
    CHECK(rig.memory.Get<GuestTag>(PROGRAM).kind == static_cast<u32>(ObjectKind::Program));

    const u64 vertex_control = SCRATCH;
    const u64 fragment_control = SCRATCH + 0x200;
    const u64 vertex_code = STORAGE + 0x8000;
    const u64 fragment_code = STORAGE + 0x9000;
    PutControl(rig, vertex_control, 0, 16);
    PutControl(rig, fragment_control, 1, 16);
    const std::vector<u8> vertex_bytes = PutCode(rig, vertex_code, 0xA1);
    const std::vector<u8> fragment_bytes = PutCode(rig, fragment_code, 0xB2);

    // NVNshaderData {data, control}, fragment listed first: the stage comes from the section.
    const u64 data = SCRATCH + 0x1000;
    rig.memory.Put<u64>(data + 0x00, fragment_code);
    rig.memory.Put<u64>(data + 0x08, fragment_control);
    rig.memory.Put<u64>(data + 0x10, vertex_code);
    rig.memory.Put<u64>(data + 0x18, vertex_control);
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(2), Rig::I(data)}) == 1);
    REQUIRE(rig.sink.programs.size() == 1);
    CHECK(rig.sink.programs[0].stage_mask == 0x3);
    CHECK(rig.sink.programs[0].code[0] == vertex_bytes);
    CHECK(rig.sink.programs[0].code[1] == fragment_bytes);
    CHECK(rig.sink.programs[0].control[0].size() == 0x100);

    // More than one entry per NVN stage (eight: 0..5, 6 task, 7 mesh) is refused and creates
    // nothing.
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(9), Rig::I(data)}) == 0);
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(static_cast<u64>(-1)), Rig::I(data)}) == 0);
    CHECK(rig.sink.programs.size() == 1);

    // Programs are immutable: setting shaders again creates a new host program.
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(1), Rig::I(data + 0x10)}) == 1);
    REQUIRE(rig.sink.programs.size() == 2);
    CHECK(rig.sink.programs[1].serial != rig.sink.programs[0].serial);
    CHECK(rig.sink.programs[1].stage_mask == 0x1);

    rig.Call("nvnProgramSetDebugLabel", {Rig::I(PROGRAM), Rig::I(SCRATCH)});
    rig.Call("nvnProgramFinalize", {Rig::I(PROGRAM)});
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(1), Rig::I(data)}) == 0);
}

// NVN2 task (stage 6) and mesh (stage 7) programs: the stage comes from control byte +0x56 as for
// the NVN1 stages (OoT's programs list task, mesh, fragment), not from the entry's position, and
// the whole control section (the mesh fields at +0x58..+0x63 included) reaches the sink.
TEST_CASE("nvn hle program: task and mesh stages come from the control section", "[nvn]") {
    Rig rig;
    REQUIRE(rig.Call("nvnProgramInitialize", {Rig::I(PROGRAM), Rig::I(0xDE)}) == 1);
    const u64 task_control = SCRATCH;
    const u64 mesh_control = SCRATCH + 0x200;
    const u64 fragment_control = SCRATCH + 0x400;
    PutControl(rig, task_control, 6, 16);
    PutControl(rig, mesh_control, 7, 16);
    PutControl(rig, fragment_control, 1, 16);
    // OoT's values (native log): task u16 +0x5a = 0xc80; mesh +0x58 = 0x20, +0x60 = 0x80, +0x62 = 0x40.
    rig.memory.Put<u16>(task_control + 0x5a, 0xc80);
    rig.memory.Put<u16>(mesh_control + 0x58, 0x20);
    rig.memory.Put<u16>(mesh_control + 0x60, 0x80);
    rig.memory.Put<u16>(mesh_control + 0x62, 0x40);
    const std::vector<u8> task_bytes = PutCode(rig, STORAGE + 0x8000, 0xA1);
    const std::vector<u8> mesh_bytes = PutCode(rig, STORAGE + 0x9000, 0xB2);
    const std::vector<u8> fragment_bytes = PutCode(rig, STORAGE + 0xA000, 0xC3);

    const u64 data = SCRATCH + 0x1000;
    rig.memory.Put<u64>(data + 0x00, STORAGE + 0x8000);
    rig.memory.Put<u64>(data + 0x08, task_control);
    rig.memory.Put<u64>(data + 0x10, STORAGE + 0x9000);
    rig.memory.Put<u64>(data + 0x18, mesh_control);
    rig.memory.Put<u64>(data + 0x20, STORAGE + 0xA000);
    rig.memory.Put<u64>(data + 0x28, fragment_control);
    const u64 fallback = rig.recorder.Stats().stage_fallback;
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(3), Rig::I(data)}) == 1);
    REQUIRE(rig.sink.programs.size() == 1);
    const auto& program = rig.sink.programs[0];
    CHECK(program.stage_mask == 0xC2);
    CHECK(program.code[6] == task_bytes);
    CHECK(program.code[7] == mesh_bytes);
    CHECK(program.code[1] == fragment_bytes);
    CHECK(program.code[0].empty());
    CHECK(program.code[2].empty());
    REQUIRE(program.control[6].size() == 0x100);
    REQUIRE(program.control[7].size() == 0x100);
    CHECK(program.control[6][0x56] == 6);
    CHECK(program.control[6][0x5a] == 0x80);
    CHECK(program.control[6][0x5b] == 0x0c);
    CHECK(program.control[7][0x56] == 7);
    CHECK(program.control[7][0x58] == 0x20);
    CHECK(program.control[7][0x60] == 0x80);
    CHECK(program.control[7][0x62] == 0x40);
    CHECK(rig.recorder.Stats().stage_fallback == fallback);

    // A stage byte past mesh is no NVN stage: the entry falls back to its position.
    PutControl(rig, task_control, 8, 16);
    CHECK(rig.Call("nvnProgramSetShaders", {Rig::I(PROGRAM), Rig::I(1), Rig::I(data)}) == 1);
    REQUIRE(rig.sink.programs.size() == 2);
    CHECK(rig.sink.programs[1].stage_mask == 0x1);
    CHECK(rig.recorder.Stats().stage_fallback == fallback + 1);
}

TEST_CASE("nvn hle program: SetShadersExt entries have a 0x18 stride", "[nvn]") {
    Rig rig;
    REQUIRE(rig.Call("nvnProgramInitialize", {Rig::I(PROGRAM), Rig::I(0xDE)}) == 1);
    const u64 vertex_control = SCRATCH;
    const u64 fragment_control = SCRATCH + 0x200;
    PutControl(rig, vertex_control, 0, 16);
    PutControl(rig, fragment_control, 1, 16);
    const std::vector<u8> vertex_bytes = PutCode(rig, STORAGE + 0x8000, 0xA1);
    const std::vector<u8> fragment_bytes = PutCode(rig, STORAGE + 0x9000, 0xB2);

    // {data, control, extension}: the extension is nonzero on DK and unused by the host.
    const u64 data = SCRATCH + 0x1000;
    rig.memory.Put<u64>(data + 0x00, STORAGE + 0x8000);
    rig.memory.Put<u64>(data + 0x08, vertex_control);
    rig.memory.Put<u64>(data + 0x10, 0x1234);
    rig.memory.Put<u64>(data + 0x18, STORAGE + 0x9000);
    rig.memory.Put<u64>(data + 0x20, fragment_control);
    rig.memory.Put<u64>(data + 0x28, 0);
    CHECK(rig.Call("nvnProgramSetShadersExt", {Rig::I(PROGRAM), Rig::I(2), Rig::I(data)}) == 1);
    REQUIRE(rig.sink.programs.size() == 1);
    CHECK(rig.sink.programs[0].stage_mask == 0x3);
    CHECK(rig.sink.programs[0].code[0] == vertex_bytes);
    CHECK(rig.sink.programs[0].code[1] == fragment_bytes);

    CHECK(rig.Call("nvnProgramSetShadersExt", {Rig::I(PROGRAM), Rig::I(9), Rig::I(data)}) == 0);
    CHECK(rig.Call("nvnProgramSetShadersExt", {Rig::I(PROGRAM), Rig::I(static_cast<u64>(-2)), Rig::I(data)}) == 0);
    // Entries that cannot be read fail without creating a program.
    CHECK(rig.Call("nvnProgramSetShadersExt",
                   {Rig::I(PROGRAM), Rig::I(1), Rig::I(Rig::BASE + Rig::MEMORY_SIZE - 8)}) == 0);
    CHECK(rig.sink.programs.size() == 1);
    CHECK(rig.Call("nvnProgramSetSampleShading", {Rig::I(PROGRAM), Rig::I(1)}) == 1);
}

TEST_CASE("nvn hle program: NVN2 builder in place and InitializeFromBuilder", "[nvn]") {
    Rig rig;
    std::array<u8, 0x40> garbage;
    garbage.fill(0xFF);
    rig.memory.Put(BUILDER, garbage);
    rig.Call("nvnProgramBuilderSetDefaults", {Rig::I(BUILDER)});
    // nxnative prog_builder_t {dev, sd, int n} = 0x18 bytes.
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == ~0ull); // the device survives (SDK sdk+0x5e47e0)
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x10) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x18) == ~0ull);

    PutControl(rig, SCRATCH, 1, 16);
    const std::vector<u8> fragment_bytes = PutCode(rig, STORAGE + 0x9000, 0xB2);
    const u64 data = SCRATCH + 0x1000;
    rig.memory.Put<u64>(data + 0x00, STORAGE + 0x9000);
    rig.memory.Put<u64>(data + 0x08, SCRATCH);

    rig.Call("nvnProgramBuilderSetDevice", {Rig::I(BUILDER), Rig::I(0xDE)});
    rig.Call("nvnProgramBuilderSetShaders", {Rig::I(BUILDER), Rig::I(1), Rig::I(data)});
    rig.Call("nvnProgramBuilderSetSampleShading", {Rig::I(BUILDER), Rig::I(2)});
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == 0xDE);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == data);
    CHECK(rig.memory.Get<s32>(BUILDER + 0x10) == 1);
    CHECK(rig.memory.Get<s32>(BUILDER + 0x14) == 2); // sample shading, in nxnative's padding
    CHECK(rig.Call("nvnProgramBuilderGetDevice", {Rig::I(BUILDER)}) == 0xDE);
    CHECK(rig.Call("nvnProgramBuilderGetShaders", {Rig::I(BUILDER)}) == data);
    CHECK(rig.Call("nvnProgramBuilderGetShaderCount", {Rig::I(BUILDER)}) == 1);
    CHECK(rig.Call("nvnProgramBuilderGetSampleShading", {Rig::I(BUILDER)}) == 2);

    CHECK(rig.Call("nvnProgramInitializeFromBuilder", {Rig::I(PROGRAM), Rig::I(BUILDER)}) == 1);
    REQUIRE(rig.sink.programs.size() == 1);
    CHECK(rig.sink.programs[0].stage_mask == 0x2);
    CHECK(rig.sink.programs[0].code[1] == fragment_bytes);
    CHECK(rig.memory.Get<GuestTag>(PROGRAM).kind == static_cast<u32>(ObjectKind::Program));

    // No shaders: an initialized, empty program.
    rig.Call("nvnProgramBuilderSetShaders", {Rig::I(BUILDER), Rig::I(0), Rig::I(0)});
    CHECK(rig.Call("nvnProgramInitializeFromBuilder", {Rig::I(PROGRAM_2), Rig::I(BUILDER)}) == 1);
    CHECK(rig.sink.programs.size() == 1);
    CHECK(rig.memory.Get<GuestTag>(PROGRAM_2).magic == GUEST_TAG_MAGIC);

    // A shader the game cannot supply fails the initialization.
    rig.Call("nvnProgramBuilderSetShaders", {Rig::I(BUILDER), Rig::I(9), Rig::I(data)});
    CHECK(rig.Call("nvnProgramInitializeFromBuilder", {Rig::I(PROGRAM_2), Rig::I(BUILDER)}) == 0);
}

// ---- Syncs -----------------------------------------------------------------------------------------------

namespace {
constexpr u64 SYNC_ALREADY_SIGNALED = 0;
constexpr u64 SYNC_CONDITION_SATISFIED = 1;
constexpr u64 SYNC_TIMEOUT_EXPIRED = 2;
constexpr u64 SYNC_FAILED = 3;
constexpr u64 INFINITE_TIMEOUT = ~0ull;

/// Fences SYNC on QUEUE and returns the sequence number of the fencing submission.
u64 FenceSync(Rig& rig) {
    const size_t before = rig.sink.submissions.size();
    rig.Call("nvnQueueFenceSync", {Rig::I(QUEUE), Rig::I(SYNC), Rig::I(0), Rig::I(0)});
    REQUIRE(rig.sink.submissions.size() == before + 1);
    return rig.sink.submissions.back().seq;
}
} // namespace

TEST_CASE("nvn hle sync: wait on unfenced, completed and uninitialized syncs", "[nvn]") {
    Rig rig;
    InitQueueAndPool(rig);
    REQUIRE(rig.Call("nvnSyncInitialize", {Rig::I(SYNC), Rig::I(0xDE)}) == 1);
    CHECK(rig.memory.Get<GuestTag>(SYNC).kind == static_cast<u32>(ObjectKind::Sync));
    rig.Call("nvnSyncSetDebugLabel", {Rig::I(SYNC), Rig::I(SCRATCH)});

    // SDK: a sync that was only initialized has nothing to wait for and FAILS, whatever the
    // timeout, without sleeping.
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_FAILED);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(INFINITE_TIMEOUT)}) == SYNC_FAILED);

    const u64 seq = FenceSync(rig);
    // A poll of a pending fence expires at once, and so does a timeout below one microsecond
    // (the SDK's deadline is in microseconds).
    const auto start = std::chrono::steady_clock::now();
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_TIMEOUT_EXPIRED);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(999)}) == SYNC_TIMEOUT_EXPIRED);
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{50});

    rig.sink.Complete(seq);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_ALREADY_SIGNALED);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(INFINITE_TIMEOUT)}) == SYNC_ALREADY_SIGNALED);

    // Not an initialized sync.
    CHECK(rig.Call("nvnSyncWait", {Rig::I(UNINITIALIZED), Rig::I(0)}) == SYNC_FAILED);
    rig.Call("nvnSyncFinalize", {Rig::I(SYNC)});
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_FAILED);
    // GL interop does not exist.
    CHECK(rig.Call("nvnSyncCreateGLSync", {Rig::I(SYNC)}) == 0);
}

TEST_CASE("nvn hle sync: the window's acquire sync is signalled without a fence", "[nvn]") {
    Rig rig;
    InitQueueAndPool(rig);
    const u64 window = Rig::Obj(40);
    for (u32 i = 0; i < 2; ++i) {
        const u64 texture = Rig::Obj(41 + i);
        REQUIRE(rig.recorder.TextureInitialize(texture, MakeTextureDesc(), POOL, 0x10000ull * i));
        rig.memory.Put<u64>(SCRATCH + 8 * i, texture);
    }
    rig.Call("nvnWindowBuilderSetDefaults", {Rig::I(BUILDER)});
    rig.Call("nvnWindowBuilderSetTextures", {Rig::I(BUILDER), Rig::I(2), Rig::I(SCRATCH)});
    REQUIRE(rig.Call("nvnWindowInitialize", {Rig::I(window), Rig::I(BUILDER)}) == 1);
    const u64 index = SCRATCH + 0x100;

    // SDK: the acquire copies the slot's sync (signalled, or the display's release fence) over
    // the game's sync object, initialized or not; the HLE's slot is free when the acquire returns.
    REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(window), Rig::I(SYNC), Rig::I(index)}) == 0);
    const GuestTag tag = rig.memory.Get<GuestTag>(SYNC);
    CHECK(tag.magic == GUEST_TAG_MAGIC);
    CHECK(tag.kind == static_cast<u32>(ObjectKind::Sync));
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_ALREADY_SIGNALED);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(INFINITE_TIMEOUT)}) == SYNC_ALREADY_SIGNALED);
    const auto state = rig.recorder.GetSyncState(SYNC);
    REQUIRE(state);
    CHECK_FALSE(state->fenced);
    CHECK(state->signaled);
    // A queue wait on it waits for nothing (no GPU-side wait is submitted).
    const size_t submissions = rig.sink.submissions.size();
    CHECK(rig.Call("nvnQueueWaitSync", {Rig::I(QUEUE), Rig::I(SYNC)}) == 1);
    CHECK(rig.sink.submissions.size() == submissions);

    // A later fence replaces the state.
    const u64 seq = FenceSync(rig);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_TIMEOUT_EXPIRED);
    rig.sink.Complete(seq);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_ALREADY_SIGNALED);

    // The next acquire signals the same sync object again (no new host object per frame).
    REQUIRE(rig.Call("nvnWindowAcquireTexture", {Rig::I(window), Rig::I(SYNC), Rig::I(index)}) == 0);
    CHECK(rig.memory.Get<GuestTag>(SYNC).serial == tag.serial);
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(0)}) == SYNC_ALREADY_SIGNALED);
}

TEST_CASE("nvn hle sync: a pending fence times out or is released by the completion", "[nvn]") {
    WaitProbe probe;
    // An infinite wait that is never released fails the test instead of hanging it.
    probe.infinite_limit = std::chrono::seconds{20};
    Rig rig;
    probe.Install(rig);
    InitQueueAndPool(rig);
    REQUIRE(rig.Call("nvnSyncInitialize", {Rig::I(SYNC), Rig::I(0xDE)}) == 1);
    const u64 seq = FenceSync(rig);

    // A finite timeout (in nanoseconds) expires.
    const auto start = std::chrono::steady_clock::now();
    CHECK(rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(30'000'000)}) == SYNC_TIMEOUT_EXPIRED);
    CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds{25});

    // A long timeout waits for the GPU; the completion releases it.
    const auto wait_in_thread = [&rig](u64 timeout) {
        auto result = std::make_shared<std::atomic<u64>>(~0ull);
        std::thread waiter{[&rig, result, timeout] {
            *result = rig.Call("nvnSyncWait", {Rig::I(SYNC), Rig::I(timeout)});
        }};
        return std::make_pair(std::move(waiter), result);
    };
    // The finite expiry above counted one wait.
    const u32 waits_before = probe.waiting.load();
    {
        auto [waiter, result] = wait_in_thread(60'000'000'000ull);
        CHECK(probe.AwaitWaiting(waits_before + 1));
        CHECK(result->load() == ~0ull);
        rig.sink.Complete(seq);
        waiter.join();
        CHECK(result->load() == SYNC_CONDITION_SATISFIED);
    }

    // An infinite wait (~0) on a fresh fence.
    const u64 second = FenceSync(rig);
    {
        auto [waiter, result] = wait_in_thread(INFINITE_TIMEOUT);
        CHECK(probe.AwaitWaiting(waits_before + 2));
        CHECK(probe.saw_infinite.load());
        CHECK(result->load() == ~0ull);
        rig.sink.Complete(second);
        waiter.join();
        CHECK(result->load() == SYNC_CONDITION_SATISFIED);
    }
}

// ---- Events ----------------------------------------------------------------------------------------------

TEST_CASE("nvn hle event: builder in place layout (nxnative ev_builder_t)", "[nvn]") {
    Rig rig;
    std::array<u8, 0x20> garbage;
    garbage.fill(0xFF);
    rig.memory.Put(BUILDER, garbage);
    rig.Call("nvnEventBuilderSetDefaults", {Rig::I(BUILDER)});
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x08) == 0);
    CHECK(rig.memory.Get<u64>(BUILDER + 0x10) == ~0ull);
    rig.Call("nvnEventBuilderSetStorage", {Rig::I(BUILDER), Rig::I(POOL), Rig::I(0x40)});
    CHECK(rig.memory.Get<u64>(BUILDER + 0x00) == POOL);
    CHECK(rig.memory.Get<s64>(BUILDER + 0x08) == 0x40);
    CHECK(rig.Call("nvnEventBuilderGetMemoryPool", {Rig::I(BUILDER)}) == POOL);
    CHECK(rig.Call("nvnEventBuilderGetMemoryOffset", {Rig::I(BUILDER)}) == 0x40);
}

TEST_CASE("nvn hle event: initialize, signal modes and value", "[nvn]") {
    Rig rig;
    REQUIRE(InitPool(rig, POOL, STORAGE, STORAGE_SIZE, 0));
    rig.memory.Put<u32>(STORAGE + 0x40, 3);
    rig.Call("nvnEventBuilderSetDefaults", {Rig::I(BUILDER)});
    rig.Call("nvnEventBuilderSetStorage", {Rig::I(BUILDER), Rig::I(POOL), Rig::I(0x40)});
    REQUIRE(rig.Call("nvnEventInitialize", {Rig::I(EVENT), Rig::I(BUILDER)}) == 1);
    CHECK(rig.memory.Get<GuestTag>(EVENT).kind == static_cast<u32>(ObjectKind::Event));
    CHECK(rig.Call("nvnEventGetMemoryPool", {Rig::I(EVENT)}) == POOL);
    CHECK(rig.Call("nvnEventGetMemoryOffset", {Rig::I(EVENT)}) == 0x40);

    // The value lives in the pool storage: the game's CPU reads and writes it directly.
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 3);
    // NVN_EVENT_SIGNAL_MODE_WRITE (0) stores, ADD (1) adds.
    rig.Call("nvnEventSignal", {Rig::I(EVENT), Rig::I(0), Rig::I(7)});
    CHECK(rig.memory.Get<u32>(STORAGE + 0x40) == 7);
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 7);
    rig.Call("nvnEventSignal", {Rig::I(EVENT), Rig::I(1), Rig::I(5)});
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 12);
    rig.Call("nvnEventSignal", {Rig::I(EVENT), Rig::I(1), Rig::I(0xFFFFFFFFull)});
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 11); // wraps modulo 2^32
    rig.Call("nvnEventSignal", {Rig::I(EVENT), Rig::I(0), Rig::I(0)});
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 0);
    rig.memory.Put<u32>(STORAGE + 0x40, 99);
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 99);

    // The sink hears about the new event and every CPU signal, with the raw values (the sum is
    // the guest word's business); the guest's own write to the word is invisible to it.
    constexpr u64 address = STORAGE + 0x40;
    using VideoCommon::Nvn::EventCpuChange;
    const auto& notes = rig.sink.event_cpu;
    REQUIRE(notes.size() == 5);
    const EventCpuChange expected_change[5] = {EventCpuChange::Forget, EventCpuChange::Write,
                                               EventCpuChange::Add, EventCpuChange::Add,
                                               EventCpuChange::Write};
    const u32 expected_value[5] = {0, 7, 5, 0xFFFFFFFF, 0};
    for (size_t i = 0; i < 5; ++i) {
        CHECK(notes[i].address == address);
        CHECK(notes[i].change == expected_change[i]);
        CHECK(notes[i].value == expected_value[i]);
    }

    rig.Call("nvnEventFinalize", {Rig::I(EVENT)});
    CHECK(rig.Call("nvnEventGetValue", {Rig::I(EVENT)}) == 0);
    CHECK(rig.Call("nvnEventGetMemoryPool", {Rig::I(EVENT)}) == 0);

    // A builder without storage cannot initialize an event.
    rig.Call("nvnEventBuilderSetDefaults", {Rig::I(BUILDER)});
    CHECK(rig.Call("nvnEventInitialize", {Rig::I(EVENT), Rig::I(BUILDER)}) == 0);
    CHECK(rig.sink.event_cpu.size() == 5);
}
