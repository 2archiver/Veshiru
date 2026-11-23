// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the NVN Vulkan replay (Vulkan::Nvn::Context). The ones that need a GPU create a
// headless device and SKIP when there is none.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/host_memory.h"
#include "common/literals.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/queue_sync.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/renderer_vulkan/present/pass_recorder.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_host_import.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

#ifdef _WIN32
#include <psapi.h> // after vulkan_wrapper.h, which includes windows.h
#endif

namespace {

namespace VC = VideoCommon::Nvn;
using namespace Common::Literals;
using namespace std::chrono_literals;

// --- guest memory -------------------------------------------------------------------------

/// A few aligned host regions standing in for guest memory.
class FakeMemory final : public VC::GuestMemory {
public:
    u8* Add(u64 va, size_t size, size_t alignment = 4096) {
        Region region;
        region.base = va;
        region.size = size;
        region.raw.resize(size + alignment);
        const auto raw = reinterpret_cast<uintptr_t>(region.raw.data());
        region.data = reinterpret_cast<u8*>((raw + alignment - 1) / alignment * alignment);
        std::memset(region.data, 0, size);
        regions.push_back(std::move(region));
        return regions.back().data;
    }

    bool Read(u64 va, void* dst, size_t size) override {
        const u8* source = Find(va, size);
        if (!source) {
            return false;
        }
        std::memcpy(dst, source, size);
        return true;
    }

    bool Write(u64 va, const void* src, size_t size) override {
        u8* target = Find(va, size);
        if (!target) {
            return false;
        }
        std::memcpy(target, src, size);
        return true;
    }

    u8* Pointer(u64 va) override {
        return Find(va, 1);
    }

private:
    struct Region {
        u64 base{};
        size_t size{};
        std::vector<u8> raw;
        u8* data{};
    };

    u8* Find(u64 va, size_t size) {
        for (Region& region : regions) {
            if (va >= region.base && va - region.base <= region.size &&
                size <= region.size - (va - region.base)) {
                return region.data + (va - region.base);
            }
        }
        return nullptr;
    }

    std::vector<Region> regions;
};

/// Guest memory backed by Common::HostMemory, the class that backs real guest memory.
class HostMemoryGuest final : public VC::GuestMemory {
public:
    explicit HostMemoryGuest(Common::HostMemory& memory_, u64 base_, size_t size_)
        : memory{memory_}, base{base_}, size{size_} {}

    bool Read(u64 va, void* dst, size_t count) override {
        const u8* source = Locate(va, count);
        if (!source) {
            return false;
        }
        std::memcpy(dst, source, count);
        return true;
    }

    bool Write(u64 va, const void* src, size_t count) override {
        u8* target = Locate(va, count);
        if (!target) {
            return false;
        }
        std::memcpy(target, src, count);
        return true;
    }

    u8* Pointer(u64 va) override {
        return Locate(va, 1);
    }

private:
    u8* Locate(u64 va, size_t count) {
        if (va < base || va - base > size || count > size - (va - base)) {
            return nullptr;
        }
        return memory.VirtualBasePointer() + va;
    }

    Common::HostMemory& memory;
    u64 base;
    size_t size;
};

// --- record encoding ------------------------------------------------------------------------

template <typename T>
void Push(std::vector<u64>& list, VC::Op op, const T& payload, u16 flags = 0) {
    constexpr size_t words = 1 + (sizeof(T) + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    VC::RecordHeader header{};
    header.op = static_cast<u16>(op);
    header.flags = flags;
    header.size = static_cast<u32>(words * 8);
    std::memcpy(&list[at], &header, sizeof(header));
    std::memcpy(&list[at + 1], &payload, sizeof(T));
}

void PushEnd(std::vector<u64>& list) {
    const size_t at = list.size();
    list.resize(at + 1, 0);
    VC::RecordHeader header{};
    header.op = static_cast<u16>(VC::Op::End);
    header.flags = 0;
    header.size = 8;
    std::memcpy(&list[at], &header, sizeof(header));
}

VC::Submission MakeSubmission(u64 seq, u32 queue, std::vector<u64> list,
                              std::vector<u64> retained = {}) {
    VC::Submission submission{};
    submission.seq = seq;
    submission.queue = queue;
    submission.lists.push_back(std::move(list));
    submission.retained_serials = std::move(retained);
    return submission;
}

// --- headless Vulkan ------------------------------------------------------------------------

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
    std::mutex submit_mutex;
};

/// nullptr when the machine has no usable Vulkan device.
std::unique_ptr<HeadlessVulkan> MakeHeadlessVulkan() {
    try {
        auto result = std::make_unique<HeadlessVulkan>();
        result->library = Vulkan::OpenLibrary(nullptr);
        if (!result->library) {
            return nullptr;
        }
        result->dld = std::make_unique<Vulkan::vk::InstanceDispatch>();
        result->instance = Vulkan::CreateInstance(*result->library, *result->dld,
                                                  VK_API_VERSION_1_1,
                                                  Core::Frontend::WindowSystemType::Headless, false);
        const std::vector<VkPhysicalDevice> devices = result->instance.EnumeratePhysicalDevices();
        if (devices.empty()) {
            return nullptr;
        }
        result->device = std::make_unique<Vulkan::Device>(
            *result->instance, Vulkan::vk::PhysicalDevice(devices[0], *result->dld),
            VK_NULL_HANDLE, *result->dld);
        return result;
    } catch (const Vulkan::vk::Exception&) {
        return nullptr;
    } catch (const std::exception&) {
        return nullptr;
    }
}

constexpr u64 POOL_CPU_NO_ACCESS = 1;
constexpr u64 POOL_CPU_CACHED = 4;

VC::TextureDesc MakeRgba8Texture(u64 serial, u64 pool_serial, u64 pool_offset, u32 width,
                                 u32 height) {
    VC::TextureDesc desc{};
    desc.target = 1; // 2D
    desc.format = 0x25; // RGBA8
    desc.flags = 0;
    desc.samples = 0;
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.levels = 1;
    desc.serial = serial;
    desc.pool_serial = pool_serial;
    desc.pool_offset = pool_offset;
    return desc;
}

struct CompletionLog {
    std::mutex mutex;
    std::vector<u64> seqs;
};

void Describe(const Vulkan::Nvn::ContextStats& stats, const char* label) {
    WARN("[nvn] " << label << ": pools imported=" << stats.pools_imported << " mirrored="
                  << stats.pools_mirrored << " device_local=" << stats.pools_device_local
                  << " import_failures=" << stats.host_import_failures);
}

/// Clears a 64x64 RGBA8 texture, copies it into a CPU-readable pool and checks every pixel.
void RunClearAndCopy(HeadlessVulkan& vulkan, bool allow_import, const char* label) {
    constexpr u64 TEXTURE_POOL = 1;
    constexpr u64 BUFFER_POOL = 2;
    constexpr u64 TEXTURE = 3;
    constexpr u64 TEXTURE_POOL_VA = 0x1000'0000;
    constexpr u64 BUFFER_POOL_VA = 0x2000'0000;
    constexpr u32 SIZE = 64;

    FakeMemory memory;
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    u8* readback = memory.Add(BUFFER_POOL_VA, 64_KiB);
    std::memset(readback, 0xCD, 64_KiB);

    // Declared before the Context: its workers stop before the callback state goes.
    CompletionLog log;
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetCompletionCallback([&log](u64 seq) {
        std::scoped_lock lock{log.mutex};
        log.seqs.push_back(seq);
    });
    context.SetGuestMemory(&memory);

    context.CreatePool({.serial = TEXTURE_POOL, .guest_address = TEXTURE_POOL_VA,
                        .size = 1_MiB, .flags = POOL_CPU_NO_ACCESS});
    context.CreatePool({.serial = BUFFER_POOL, .guest_address = BUFFER_POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreateTexture(MakeRgba8Texture(TEXTURE, TEXTURE_POOL, 0, SIZE, SIZE));

    std::vector<u64> list;
    VC::ClearTexturePayload clear{};
    clear.texture = {.serial = TEXTURE, .level = 0, .layer = 0};
    clear.region = {0, 0, 0, SIZE, SIZE, 1};
    clear.layer_count = 1;
    clear.is_integer = 0;
    const float color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    std::memcpy(clear.color, color, sizeof(color));
    Push(list, VC::Op::ClearTexture, clear, 0xF);
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = BUFFER_POOL_VA;
    copy.texture = {.serial = TEXTURE, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, SIZE, SIZE, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
    PushEnd(list);

    context.Submit(MakeSubmission(1, 0, std::move(list), {TEXTURE}));
    REQUIRE(context.Wait(1, 20s));

    // 0.25 / 0.5 / 0.75 of 255 are 63.75 / 127.5 / 191.25: the driver rounds the exact tie of the
    // green channel either way (the RTX 5090 stores 127), so allow one step on each channel.
    const auto close_to = [](u8 value, int expected) {
        return std::abs(int(value) - expected) <= 1;
    };
    size_t wrong = 0;
    for (u32 pixel = 0; pixel < SIZE * SIZE; ++pixel) {
        const u8* p = readback + pixel * 4;
        if (!close_to(p[0], 64) || !close_to(p[1], 128) || !close_to(p[2], 191) || p[3] != 255 ||
            std::memcmp(p, readback, 4) != 0) {
            ++wrong;
        }
    }
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    Describe(stats, label);
    REQUIRE(wrong == 0);
    WARN("[nvn] " << label << ": all " << SIZE * SIZE << " pixels are (" << int(readback[0]) << ", "
                   << int(readback[1]) << ", " << int(readback[2]) << ", " << int(readback[3])
                   << ") (expected (64, 128, 191, 255) +-1): pixel check passed");
    // The readback must be behind everything the completion callback reported.
    {
        std::scoped_lock lock{log.mutex};
        REQUIRE(log.seqs == std::vector<u64>{1});
    }
    REQUIRE(stats.textures_created == 1);
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    if (allow_import) {
        REQUIRE(stats.pools_imported + stats.pools_mirrored == 1);
    } else {
        REQUIRE(stats.pools_mirrored == 1);
        REQUIRE(stats.pools_imported == 0);
    }
    REQUIRE(stats.pools_device_local == 1);
}

/// Buffer to buffer copy on queue 0, then a second copy on queue 1 that waits for queue 0's fence.
/// With `consumer_first` queue 1's submission reaches the sink before the fence is submitted.
void RunCrossQueueCopy(HeadlessVulkan& vulkan, bool allow_import, bool consumer_first) {
    constexpr u64 POOL = 1;
    constexpr u64 POOL_VA = 0x3000'0000;
    constexpr u64 SYNC = 100;
    constexpr u32 BYTES = 1024;

    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    for (u32 i = 0; i < BYTES; ++i) {
        data[i] = static_cast<u8>(i * 7 + 3);
    }
    // Declared before the Context: its workers stop before the callback state goes.
    CompletionLog log;
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetCompletionCallback([&log](u64 seq) {
        std::scoped_lock lock{log.mutex};
        log.seqs.push_back(seq);
    });
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = POOL, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});

    std::vector<u64> producer;
    Push(producer, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = POOL_VA, .dst = POOL_VA + 4096, .size = BYTES});
    Push(producer, VC::Op::FenceSync, VC::FenceSyncPayload{.sync_serial = SYNC});
    PushEnd(producer);

    std::vector<u64> consumer;
    Push(consumer, VC::Op::WaitSync, VC::WaitSyncPayload{.sync_serial = SYNC});
    Push(consumer, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = POOL_VA + 4096, .dst = POOL_VA + 8192, .size = BYTES});
    PushEnd(consumer);

    if (consumer_first) {
        context.Submit(MakeSubmission(1, 1, std::move(consumer), {SYNC}));
        context.Submit(MakeSubmission(2, 0, std::move(producer), {SYNC}));
    } else {
        context.Submit(MakeSubmission(1, 0, std::move(producer), {SYNC}));
        context.Submit(MakeSubmission(2, 1, std::move(consumer), {SYNC}));
    }
    REQUIRE(context.Wait(2, 20s));

    REQUIRE(std::memcmp(data, data + 4096, BYTES) == 0);
    REQUIRE(std::memcmp(data, data + 8192, BYTES) == 0);
    {
        std::scoped_lock lock{log.mutex};
        REQUIRE(log.seqs == std::vector<u64>{1, 2});
    }
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.ops_failed == 0);
    if (consumer_first) {
        REQUIRE(stats.held_back_submissions == 1);
    } else {
        REQUIRE(stats.held_back_submissions == 0);
    }
}

/// Two submissions on one queue: the second reads what the first wrote.
void RunSameQueueOrdering(HeadlessVulkan& vulkan, bool allow_import) {
    constexpr u64 POOL_VA = 0x4000'0000;
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0x11, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    for (u32 round = 0; round < 8; ++round) {
        const u32 value = 0xA0A0A0A0u + round * 0x01010101u;
        std::vector<u64> fill;
        Push(fill, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = POOL_VA, .size = 4096, .value = value});
        PushEnd(fill);
        std::vector<u64> copy;
        Push(copy, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = POOL_VA, .dst = POOL_VA + 8192, .size = 4096});
        PushEnd(copy);
        context.Submit(MakeSubmission(round * 2 + 1, 0, std::move(fill)));
        context.Submit(MakeSubmission(round * 2 + 2, 0, std::move(copy)));
        REQUIRE(context.Wait(round * 2 + 2, 20s));
        for (u32 i = 0; i < 4096; ++i) {
            REQUIRE(data[8192 + i] == static_cast<u8>(value));
        }
    }
    REQUIRE(context.Stats().ops_failed == 0);
}

/// A texture in a CPU_NO_ACCESS pool still gets the guest bytes that were there.
void RunGpuOnlyPoolTexture(HeadlessVulkan& vulkan, bool allow_import) {
    constexpr u64 TEXTURE_POOL_VA = 0x5000'0000;
    constexpr u64 BUFFER_VA = 0x6000'0000;
    constexpr u32 SIZE = 64;
    const Vulkan::Nvn::TextureLayoutDesc layout_desc{
        .target = static_cast<s32>(Vulkan::Nvn::TexTarget::Dim2D), .format = 0x25,
        .width = SIZE, .height = SIZE, .depth = 1, .levels = 1};
    const Vulkan::Nvn::TextureLayout layout = Vulkan::Nvn::ComputeTextureLayout(layout_desc);

    std::vector<u8> linear(SIZE * SIZE * 4);
    for (size_t i = 0; i < linear.size(); ++i) {
        linear[i] = static_cast<u8>(i * 13 + (i >> 8));
    }
    FakeMemory memory;
    u8* texture_bytes = memory.Add(TEXTURE_POOL_VA, 64_KiB);
    u8* readback = memory.Add(BUFFER_VA, 64_KiB);
    Tegra::Texture::SwizzleTexture(std::span<u8>{texture_bytes, layout.size}, linear, 4, SIZE,
                                   SIZE, 1, layout.tile_h, layout.tile_d, 0, true);

    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = TEXTURE_POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_NO_ACCESS});
    context.CreatePool({.serial = 2, .guest_address = BUFFER_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreateTexture(MakeRgba8Texture(3, 1, 0, SIZE, SIZE));
    std::vector<u64> list;
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = BUFFER_VA;
    copy.texture = {.serial = 3, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, SIZE, SIZE, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list), {3}));
    REQUIRE(context.Wait(1, 20s));
    REQUIRE(std::memcmp(readback, linear.data(), linear.size()) == 0);
    REQUIRE(context.Stats().ops_failed == 0);
}

/// A submission waits for a fence nobody submits while other work keeps arriving: the timeout has
/// to fire on the arrivals, not only when the replay is idle.
void RunParkedTimeoutUnderLoad(HeadlessVulkan& vulkan) {
    constexpr u64 POOL_VA = 0x7000'0000;
    FakeMemory memory;
    memory.Add(POOL_VA, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.parked_timeout = 300ms});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    std::vector<u64> waiting;
    Push(waiting, VC::Op::WaitSync, VC::WaitSyncPayload{.sync_serial = 777});
    PushEnd(waiting);
    context.Submit(MakeSubmission(1, 1, std::move(waiting), {777}));
    u64 seq = 2;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < 1500ms) {
        std::vector<u64> work;
        Push(work, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = POOL_VA, .size = 256, .value = static_cast<u32>(seq)});
        PushEnd(work);
        context.Submit(MakeSubmission(seq++, 0, std::move(work)));
        std::this_thread::sleep_for(40ms);
    }
    // Every arrival restarts the idle wake-up, so only the check on arrival can have released it.
    REQUIRE(context.Wait(1, 100ms));
    REQUIRE(context.Wait(seq - 1, 20s));
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.parked_timeouts == 1);
    REQUIRE(stats.held_back_submissions == 1);
}

void PushSignalEvent(std::vector<u64>& list, u64 address, u32 mode, u32 value) {
    Push(list, VC::Op::SignalEvent,
         VC::SignalEventPayload{.event_serial = 0, .event_address = address, .mode = mode,
                                .value = value});
}

void PushWaitEvent(std::vector<u64>& list, u64 address, u32 mode, u32 value) {
    Push(list, VC::Op::WaitEvent,
         VC::WaitEventPayload{.event_address = address, .mode = mode, .value = value});
}

void PushFill(std::vector<u64>& list, u64 address, u32 value) {
    Push(list, VC::Op::ClearBuffer,
         VC::ClearBufferPayload{.address = address, .size = 256, .value = value});
}

void PushCopy(std::vector<u64>& list, u64 source, u64 destination) {
    Push(list, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = source, .dst = destination, .size = 256});
}

/// All 256 bytes at `data` are the little-endian words of `value`.
bool Filled(const u8* data, u32 value) {
    for (u32 i = 0; i < 256; ++i) {
        if (data[i] != static_cast<u8>(value >> (8 * (i % 4)))) {
            return false;
        }
    }
    return true;
}

/// DK's two-queue frame, three times, with a data dependency through pool memory at every stage
/// (each stage reads what the previous stage of the OTHER queue wrote, so a wrong order copies
/// stale bytes). Every event value is a WRITE; the events start as 0xFFFFFFFF in guest memory.
///     q1: Wait e0==0   copy A->B  Signal e4=1   Wait e0==2   copy C->D  Signal e4=3
///     q0: fill A  Signal e0=0   Wait e4==1  copy B->C  Signal e0=2   Wait e4==3  copy D->E
///         reset both events to 0xFFFFFFFF
/// With `q1_first` the queue-1 submission of each frame reaches the sink before queue 0's: its
/// waits cannot be satisfied when it runs and the replay has to suspend it.
void RunEventPingPong(HeadlessVulkan& vulkan, bool allow_import, bool q1_first) {
    constexpr u64 POOL_VA = 0x3100'0000;
    constexpr u64 EVENT_VA = 0x3200'0000;
    constexpr u64 E0 = EVENT_VA;
    constexpr u64 E4 = EVENT_VA + 16;
    constexpr u32 FRAMES = 3;
    constexpr u64 FRAME_STRIDE = 0x1000;
    constexpr u64 NONE = 0xFFFFFFFF;

    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    u8* events = memory.Add(EVENT_VA, 4_KiB);
    std::memset(events, 0xFF, 64);

    // Declared before the Context: its workers stop before the callback state goes.
    CompletionLog log;
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetCompletionCallback([&log](u64 seq) {
        std::scoped_lock lock{log.mutex};
        log.seqs.push_back(seq);
    });
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});

    for (u32 frame = 0; frame < FRAMES; ++frame) {
        const u64 base = POOL_VA + frame * FRAME_STRIDE;
        const u64 a = base;
        const u64 b = base + 0x100;
        const u64 c = base + 0x200;
        const u64 d = base + 0x300;
        const u64 e = base + 0x400;
        const u32 value = 0x11111111u * (frame + 1);

        std::vector<u64> q1;
        PushWaitEvent(q1, E0, VC::EVENT_WAIT_EQUAL, 0);
        PushCopy(q1, a, b);
        PushSignalEvent(q1, E4, VC::EVENT_SIGNAL_WRITE, 1);
        PushWaitEvent(q1, E0, VC::EVENT_WAIT_EQUAL, 2);
        PushCopy(q1, c, d);
        PushSignalEvent(q1, E4, VC::EVENT_SIGNAL_WRITE, 3);
        PushEnd(q1);

        std::vector<u64> q0;
        PushFill(q0, a, value);
        PushSignalEvent(q0, E0, VC::EVENT_SIGNAL_WRITE, 0);
        PushWaitEvent(q0, E4, VC::EVENT_WAIT_EQUAL, 1);
        PushCopy(q0, b, c);
        PushSignalEvent(q0, E0, VC::EVENT_SIGNAL_WRITE, 2);
        PushWaitEvent(q0, E4, VC::EVENT_WAIT_EQUAL, 3);
        PushCopy(q0, d, e);
        PushSignalEvent(q0, E0, VC::EVENT_SIGNAL_WRITE, static_cast<u32>(NONE));
        PushSignalEvent(q0, E4, VC::EVENT_SIGNAL_WRITE, static_cast<u32>(NONE));
        PushEnd(q0);

        const u64 first = frame * 2 + 1;
        if (q1_first) {
            context.Submit(MakeSubmission(first, 1, std::move(q1)));
            context.Submit(MakeSubmission(first + 1, 0, std::move(q0)));
        } else {
            context.Submit(MakeSubmission(first, 0, std::move(q0)));
            context.Submit(MakeSubmission(first + 1, 1, std::move(q1)));
        }
    }
    REQUIRE(context.Wait(FRAMES * 2, 20s));

    for (u32 frame = 0; frame < FRAMES; ++frame) {
        const u32 value = 0x11111111u * (frame + 1);
        const u8* base = data + frame * FRAME_STRIDE;
        for (u32 stage = 0; stage < 5; ++stage) { // A, B, C, D, E
            INFO("frame " << frame << " stage " << stage);
            REQUIRE(Filled(base + stage * 0x100, value));
        }
    }
    {
        std::scoped_lock lock{log.mutex};
        REQUIRE(log.seqs == std::vector<u64>{1, 2, 3, 4, 5, 6});
    }
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.waits_unsatisfied == 0);
    REQUIRE(stats.event_waits == FRAMES * 4);
    REQUIRE(stats.event_signals == FRAMES * 6);
    REQUIRE(stats.event_waits_blocked > 0);
}

/// A wait nobody signals: its stream is released by the timeout, counted and reported, while the
/// other queue's work (later in submission order) is not held up by it.
void RunEventNeverSignalled(HeadlessVulkan& vulkan) {
    constexpr u64 POOL_VA = 0x3300'0000;
    constexpr u64 EVENT_VA = 0x3400'0000;
    constexpr auto TIMEOUT = 1000ms;

    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    memory.Add(EVENT_VA, 4_KiB); // the word is 0 and nobody ever writes 5 there

    // Declared before the Context: its workers stop before the callback state goes.
    CompletionLog log;
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.parked_timeout = TIMEOUT});
    context.SetCompletionCallback([&log](u64 seq) {
        std::scoped_lock lock{log.mutex};
        log.seqs.push_back(seq);
    });
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});

    // Warm up: the pool is backed and the first submission's one-time work is done.
    std::vector<u64> warm;
    PushFill(warm, POOL_VA + 0x800, 0x01010101u);
    PushEnd(warm);
    context.Submit(MakeSubmission(1, 0, std::move(warm)));
    REQUIRE(context.Wait(1, 20s));

    std::vector<u64> stuck;
    PushWaitEvent(stuck, EVENT_VA, VC::EVENT_WAIT_EQUAL, 5);
    PushFill(stuck, POOL_VA, 0xAAAAAAAAu);
    PushEnd(stuck);
    std::vector<u64> other;
    PushFill(other, POOL_VA + 0x100, 0xBBBBBBBBu);
    PushEnd(other);
    const auto start = std::chrono::steady_clock::now();
    context.Submit(MakeSubmission(2, 1, std::move(stuck)));
    context.Submit(MakeSubmission(3, 0, std::move(other)));

    // Completion callbacks keep seq order, so the other queue's work shows in the pool memory
    // (imported pools: the GPU writes the guest bytes themselves) while seq 2 is still stuck.
    if (context.Stats().pools_imported == 1) {
        bool seen = false;
        while (std::chrono::steady_clock::now() - start < TIMEOUT / 2) {
            if (Filled(data + 0x100, 0xBBBBBBBBu)) {
                seen = true;
                break;
            }
            std::this_thread::sleep_for(5ms);
        }
        REQUIRE(seen);
        REQUIRE(!context.Wait(2, 0ms));
        REQUIRE(!Filled(data, 0xAAAAAAAAu));
    } else {
        WARN("[nvn] pool not imported: the other queue's immediate run is not observable");
    }
    REQUIRE(context.Wait(3, 20s));
    REQUIRE(std::chrono::steady_clock::now() - start >= TIMEOUT * 9 / 10);
    REQUIRE(Filled(data, 0xAAAAAAAAu));
    REQUIRE(Filled(data + 0x100, 0xBBBBBBBBu));
    {
        std::scoped_lock lock{log.mutex};
        REQUIRE(log.seqs == std::vector<u64>{1, 2, 3});
    }
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.waits_unsatisfied == 1);
    REQUIRE(stats.event_waits == 0);
    REQUIRE(stats.event_waits_blocked == 1);
    REQUIRE(stats.ops_failed == 0);
}

/// A CPU change of an event (Sink::NotifyEventCpu) releases the stream waiting for it: a WRITE for an
/// EQUAL wait, three ADDs for a GEQUAL_WRAP wait. No timeout is involved.
void RunEventCpuSignal(HeadlessVulkan& vulkan) {
    constexpr u64 POOL_VA = 0x3500'0000;
    constexpr u64 EVENT_VA = 0x3600'0000;
    constexpr u64 EQUAL_EVENT = EVENT_VA;
    constexpr u64 COUNT_EVENT = EVENT_VA + 16;

    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    memory.Add(EVENT_VA, 4_KiB);

    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});

    std::vector<u64> warm;
    PushFill(warm, POOL_VA + 0x800, 0x01010101u);
    PushEnd(warm);
    context.Submit(MakeSubmission(1, 0, std::move(warm)));
    REQUIRE(context.Wait(1, 20s));

    std::vector<u64> equal;
    PushWaitEvent(equal, EQUAL_EVENT, VC::EVENT_WAIT_EQUAL, 5);
    PushFill(equal, POOL_VA, 0xAAAAAAAAu);
    PushEnd(equal);
    context.Submit(MakeSubmission(2, 1, std::move(equal)));
    std::vector<u64> counted;
    PushWaitEvent(counted, COUNT_EVENT, VC::EVENT_WAIT_GEQUAL_WRAP, 3);
    PushFill(counted, POOL_VA + 0x100, 0xBBBBBBBBu);
    PushEnd(counted);
    context.Submit(MakeSubmission(3, 1, std::move(counted)));

    REQUIRE(!context.Wait(2, 200ms));
    context.NotifyEventCpu(EQUAL_EVENT, VC::EventCpuChange::Write, 4); // not what is waited for
    REQUIRE(!context.Wait(2, 100ms));
    context.NotifyEventCpu(EQUAL_EVENT, VC::EventCpuChange::Write, 5);
    REQUIRE(context.Wait(2, 20s));
    REQUIRE(Filled(data, 0xAAAAAAAAu));

    // Seq 3 is behind seq 2 on its queue and waits for a count of 3.
    REQUIRE(!context.Wait(3, 100ms));
    context.NotifyEventCpu(COUNT_EVENT, VC::EventCpuChange::Add, 1);
    context.NotifyEventCpu(COUNT_EVENT, VC::EventCpuChange::Add, 1);
    REQUIRE(!context.Wait(3, 100ms));
    context.NotifyEventCpu(COUNT_EVENT, VC::EventCpuChange::Add, 1);
    REQUIRE(context.Wait(3, 20s));
    REQUIRE(Filled(data + 0x100, 0xBBBBBBBBu));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.waits_unsatisfied == 0);
    REQUIRE(stats.event_waits == 2);
    REQUIRE(stats.event_waits_blocked == 2);
    REQUIRE(stats.ops_failed == 0);
}

} // Anonymous namespace

TEST_CASE("NVN Vulkan: texture layout of a 64x64 RGBA8 block-linear texture", "[nvn]") {
    using namespace Vulkan::Nvn;
    TextureLayoutDesc desc{};
    desc.target = static_cast<s32>(TexTarget::Dim2D);
    desc.format = 0x25;
    desc.width = 64;
    desc.height = 64;
    desc.depth = 1;
    desc.levels = 1;
    const TextureLayout layout = ComputeTextureLayout(desc);
    REQUIRE(layout.bytes_per_block == 4);
    REQUIRE(layout.size == 64 * 64 * 4);
    REQUIRE(layout.align == 512);
    REQUIRE(TextureLevelOffset(desc, 0) == 0);

    // A mip chain is laid out level after level, each a whole number of GOBs.
    desc.width = 256;
    desc.height = 256;
    desc.levels = 9;
    const TextureLayout chain = ComputeTextureLayout(desc);
    u64 previous = 0;
    for (u32 level = 1; level < 9; ++level) {
        const u64 offset = TextureLevelOffset(desc, level);
        REQUIRE(offset > previous);
        REQUIRE(offset % 512 == 0);
        previous = offset;
    }
    REQUIRE(chain.size > previous);
    REQUIRE(TextureLevelOffset(desc, 1) == 256 * 256 * 4);

    // Linear textures use their stride.
    desc = {};
    desc.target = static_cast<s32>(TexTarget::Dim2D);
    desc.format = 0x25;
    desc.flags = TexFlag::Linear;
    desc.width = 100;
    desc.height = 10;
    desc.depth = 1;
    desc.levels = 1;
    desc.stride = 512;
    REQUIRE(ComputeTextureLayout(desc).size == 5120);
}

TEST_CASE("NVN Vulkan: NVN formats map to pixel formats", "[nvn]") {
    using VideoCore::Surface::PixelFormat;
    using Vulkan::Nvn::NvnToPixelFormat;
    REQUIRE(NvnToPixelFormat(0x25) == PixelFormat::A8B8G8R8_UNORM);
    REQUIRE(NvnToPixelFormat(0x38) == PixelFormat::A8B8G8R8_SRGB);
    REQUIRE(NvnToPixelFormat(0x99) == PixelFormat::B8G8R8A8_UNORM);
    REQUIRE(NvnToPixelFormat(0x4D) == PixelFormat::BC7_UNORM);
    // Packed float / 10-bit formats (NVNformat RGB10A2 0x3D, RGB10A2UI 0x3E, R11G11B10F 0x3F,
    // RGB9E5F 0x40): R11G11B10F is VideoCore's B10G11R11_FLOAT, whose host format
    // VK_FORMAT_B10G11R11_UFLOAT_PACK32 keeps red in the low bits like the guest's texel.
    REQUIRE(NvnToPixelFormat(0x3D) == PixelFormat::A2B10G10R10_UNORM);
    REQUIRE(NvnToPixelFormat(0x3E) == PixelFormat::A2B10G10R10_UINT);
    REQUIRE(NvnToPixelFormat(0x3F) == PixelFormat::B10G11R11_FLOAT);
    REQUIRE(NvnToPixelFormat(0x40) == PixelFormat::E5B9G9R9_FLOAT);
    REQUIRE(NvnToPixelFormat(0x29) == PixelFormat::R16G16B16A16_FLOAT);
    REQUIRE(NvnToPixelFormat(0x79) == PixelFormat::ASTC_2D_4X4_UNORM);
    REQUIRE(NvnToPixelFormat(0x94) == PixelFormat::ASTC_2D_12X12_SRGB);
    REQUIRE(NvnToPixelFormat(0x34) == PixelFormat::D32_FLOAT);
    REQUIRE(!NvnToPixelFormat(0x19).has_value()); // RGB8: no host format
    REQUIRE(!NvnToPixelFormat(0).has_value());
    REQUIRE(!NvnToPixelFormat(0x1000).has_value());
}

#ifdef _WIN32
TEST_CASE("NVN Vulkan: MakeResident brings committed pages in and skips the others",
          "[nvn][import]") {
    constexpr size_t Page = 4096;
    // Reserved, committed and never touched, reserved, committed without access: reading the
    // first, third or fourth page would fault.
    u8* const memory =
        static_cast<u8*>(VirtualAlloc(nullptr, 4 * Page, MEM_RESERVE, PAGE_READWRITE));
    REQUIRE(memory != nullptr);
    REQUIRE(VirtualAlloc(memory + Page, Page, MEM_COMMIT, PAGE_READWRITE) != nullptr);
    REQUIRE(VirtualAlloc(memory + 3 * Page, Page, MEM_COMMIT, PAGE_NOACCESS) != nullptr);
    const auto resident = [](const void* address) {
        PSAPI_WORKING_SET_EX_INFORMATION info{};
        info.VirtualAddress = const_cast<void*>(address);
        REQUIRE(QueryWorkingSetEx(GetCurrentProcess(), &info, sizeof(info)));
        return info.VirtualAttributes.Valid != 0;
    };
    REQUIRE(!resident(memory + Page));

    Vulkan::Nvn::HostImporter::MakeResident(memory, 4 * Page);

    REQUIRE(resident(memory + Page));
    REQUIRE(memory[Page + 5] == 0);
    REQUIRE(VirtualFree(memory, 0, MEM_RELEASE));
}
#endif

TEST_CASE("NVN Vulkan: clear a texture and copy it to a buffer", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("pool import allowed") {
        RunClearAndCopy(*vulkan, true, "default pool path");
    }
    SECTION("device-local mirror") {
        RunClearAndCopy(*vulkan, false, "forced mirror pool path");
    }
}

TEST_CASE("NVN Vulkan: buffer copy, second queue waits for the first queue's fence", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("producer submitted first") {
        RunCrossQueueCopy(*vulkan, true, false);
    }
    SECTION("consumer submitted first, held back until the fence exists") {
        RunCrossQueueCopy(*vulkan, true, true);
    }
    SECTION("mirror pools") {
        RunCrossQueueCopy(*vulkan, false, true);
    }
}

TEST_CASE("NVN Vulkan: host pointer import probe on Common::HostMemory", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr size_t BACKING = 64_MiB;
    constexpr size_t VIRTUAL = size_t{1} << 34;
    constexpr size_t OFFSET = 0x1000'0000;
    constexpr size_t LENGTH = 2_MiB;
    Common::HostMemory memory(BACKING, VIRTUAL);
    REQUIRE(memory.BackingBasePointer() != nullptr);
    memory.Map(OFFSET, 0, LENGTH, Common::MemoryPermission::ReadWrite, false);
    u8* const pointer = memory.VirtualBasePointer() + OFFSET;
    std::memset(pointer, 0, LENGTH);

    const Vulkan::Device& device = *vulkan->device;
    VmaAllocator allocator = device.CreateThreadSafeAllocator();
    {
        Vulkan::Nvn::HostImporter importer(device, allocator);
        std::string result_text;
        VkResult result = VK_ERROR_EXTENSION_NOT_PRESENT;
        bool supported = false;
        std::string reason;
        if (importer.Available()) {
            Vulkan::Nvn::ImportedBuffer imported;
            result = importer.Import(pointer, LENGTH, Vulkan::Nvn::POOL_BUFFER_USAGE,
                                     imported, reason);
            supported = result == VK_SUCCESS;
        } else {
            reason = device.IsExtExternalMemoryHostSupported()
                         ? "vkGetMemoryHostPointerPropertiesEXT or alignment unavailable"
                         : "VK_EXT_external_memory_host not supported by the device";
        }
        if (supported) {
            WARN("[nvn] host import: supported (VK_SUCCESS), minImportedHostPointerAlignment = "
                 << importer.Alignment());
        } else {
            WARN("[nvn] host import: unsupported (" << static_cast<int>(result) << "): " << reason);
        }
    }
    vmaDestroyAllocator(allocator);

    // The same memory as a pool: whichever way the replay chose to back it, the GPU's write must
    // arrive in the guest bytes.
    HostMemoryGuest guest(memory, OFFSET, LENGTH);
    Vulkan::Nvn::Context context(device, vulkan->submit_mutex, nullptr);
    context.SetGuestMemory(&guest);
    context.CreatePool({.serial = 1, .guest_address = OFFSET, .size = LENGTH,
                        .flags = POOL_CPU_CACHED});
    std::vector<u64> list;
    Push(list, VC::Op::ClearBuffer,
         VC::ClearBufferPayload{.address = OFFSET + 4096, .size = 8192, .value = 0xA5A5A5A5});
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list)));
    REQUIRE(context.Wait(1, 20s));
    for (size_t i = 0; i < 8192; ++i) {
        REQUIRE(pointer[4096 + i] == 0xA5);
    }
    REQUIRE(pointer[4095] == 0);
    REQUIRE(pointer[4096 + 8192] == 0);
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    Describe(stats, "HostMemory-backed pool");
    REQUIRE(stats.pools_imported + stats.pools_mirrored == 1);
}

TEST_CASE("NVN Vulkan: pools in one guest region share one chunk import", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr size_t BACKING = 64_MiB;
    constexpr size_t VIRTUAL = size_t{1} << 34;
    // The region straddles the 256 MiB chunk grid line at 0x1000'0000.
    constexpr size_t OFFSET = 0x0FF0'0000;
    constexpr size_t LENGTH = 2_MiB;
    Common::HostMemory memory(BACKING, VIRTUAL);
    REQUIRE(memory.BackingBasePointer() != nullptr);
    memory.Map(OFFSET, 0, LENGTH, Common::MemoryPermission::ReadWrite, false);
    u8* const pointer = memory.VirtualBasePointer() + OFFSET;
    std::memset(pointer, 0, LENGTH);

    HostMemoryGuest guest(memory, OFFSET, LENGTH);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr);
    context.SetGuestMemory(&guest);
    constexpr u64 SMALL = 256_KiB;
    constexpr u64 SECOND = OFFSET + SMALL; // pools must not overlap
    constexpr u64 CROSSING = OFFSET + 0x8'0000; // 512 KiB..2 MiB: spans 0x1000'0000
    context.CreatePool({.serial = 1, .guest_address = OFFSET, .size = SMALL,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = SECOND, .size = SMALL,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 3, .guest_address = CROSSING, .size = LENGTH - 0x8'0000,
                        .flags = POOL_CPU_CACHED});
    std::vector<u64> list;
    for (const u64 address : {u64{OFFSET}, u64{SECOND}, u64{CROSSING} + 0xF'0000}) {
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = address, .size = 4096, .value = 0x5A5A5A5A});
    }
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list)));
    REQUIRE(context.Wait(1, 20s));
    for (const u64 address : {u64{OFFSET}, u64{SECOND}, u64{CROSSING} + 0xF'0000}) {
        for (size_t i = 0; i < 4096; ++i) {
            REQUIRE(pointer[address - OFFSET + i] == 0x5A);
        }
    }
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    Describe(stats, "chunked pools");
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.pools_imported + stats.pools_mirrored == 3);
    if (stats.pools_imported == 3) {
        // Three pools, one import: the chunk covers the whole mapped run, across the grid line.
        const std::string state = context.DescribeState();
        WARN("[nvn] " << state);
        REQUIRE(state.find("imports(n/ms/MB)=1/") != std::string::npos);
        REQUIRE(state.find("pools_in_chunks=3") != std::string::npos);
    }
}

TEST_CASE("NVN Vulkan: a buffer read from a GPU-only pool sees the guest bytes", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 SOURCE_VA = 0x5100'0000;
    constexpr u64 DESTINATION_VA = 0x5200'0000;
    FakeMemory memory;
    u8* source = memory.Add(SOURCE_VA, 64_KiB);
    u8* destination = memory.Add(DESTINATION_VA, 64_KiB);
    for (size_t i = 0; i < 8192; ++i) {
        source[i] = static_cast<u8>(i * 7 + 3);
    }
    std::memset(destination, 0xCD, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr);
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = SOURCE_VA, .size = 64_KiB,
                        .flags = POOL_CPU_NO_ACCESS});
    context.CreatePool({.serial = 2, .guest_address = DESTINATION_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    std::vector<u64> list;
    Push(list, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = SOURCE_VA, .dst = DESTINATION_VA, .size = 8192});
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list)));
    REQUIRE(context.Wait(1, 20s));
    REQUIRE(std::memcmp(destination, source, 8192) == 0);
    REQUIRE(destination[8192] == 0xCD);
    REQUIRE(context.Stats().ops_failed == 0);
}

namespace {

/// GPU-only pools whose guest storage is filled after the pools were created (and, with the
/// backing thread, after their buffers exist) are uploaded at their first use, not at creation.
void RunLateFilledPools(HeadlessVulkan& vulkan, bool background) {
    constexpr u64 SOURCE_BASE = 0x5300'0000;
    constexpr u64 DESTINATION_BASE = 0x5400'0000;
    constexpr u64 STRIDE = 0x10'0000;
    constexpr u64 POOLS = 12;
    constexpr u64 SIZE = 64_KiB;
    FakeMemory memory;
    std::vector<u8*> sources;
    std::vector<u8*> destinations;
    for (u64 i = 0; i < POOLS; ++i) {
        sources.push_back(memory.Add(SOURCE_BASE + i * STRIDE, SIZE));
        destinations.push_back(memory.Add(DESTINATION_BASE + i * STRIDE, SIZE));
        std::memset(destinations.back(), 0xCD, SIZE);
    }
    Vulkan::Nvn::ContextOptions options;
    options.background_backing = background;
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr, options);
    context.SetGuestMemory(&memory);
    for (u64 i = 0; i < POOLS; ++i) {
        context.CreatePool({.serial = 1 + i, .guest_address = SOURCE_BASE + i * STRIDE,
                            .size = SIZE, .flags = POOL_CPU_NO_ACCESS});
        context.CreatePool({.serial = 100 + i, .guest_address = DESTINATION_BASE + i * STRIDE,
                            .size = SIZE, .flags = POOL_CPU_CACHED});
    }
    // Let the backing thread create every buffer before the game fills the storage.
    std::this_thread::sleep_for(300ms);
    for (u64 i = 0; i < POOLS; ++i) {
        for (size_t j = 0; j < 4096; ++j) {
            sources[i][j] = static_cast<u8>(i * 31 + j * 7 + 1);
        }
    }
    std::vector<u64> list;
    for (u64 i = 0; i < POOLS; ++i) {
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = SOURCE_BASE + i * STRIDE,
                                           .dst = DESTINATION_BASE + i * STRIDE, .size = 4096});
    }
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list)));
    REQUIRE(context.Wait(1, 20s));
    for (u64 i = 0; i < POOLS; ++i) {
        REQUIRE(std::memcmp(destinations[i], sources[i], 4096) == 0);
        REQUIRE(destinations[i][4096] == 0xCD);
    }
    REQUIRE(context.Stats().ops_failed == 0);
    if (background) {
        // The pools were backed before the submission needed them.
        REQUIRE(context.DescribeState().find("backed_in_background=0 ") == std::string::npos);
    }
}

} // namespace

TEST_CASE("NVN Vulkan: pools backed in the background see bytes the game wrote later", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("backing thread") {
        RunLateFilledPools(*vulkan, true);
    }
    SECTION("backing on demand") {
        RunLateFilledPools(*vulkan, false);
    }
}

TEST_CASE("NVN Vulkan: submissions on one queue are ordered after each other", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("imported pool") {
        RunSameQueueOrdering(*vulkan, true);
    }
    SECTION("mirror pool") {
        RunSameQueueOrdering(*vulkan, false);
    }
}

TEST_CASE("NVN Vulkan: a texture in a GPU-only pool gets its guest contents", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("imported readback pool") {
        RunGpuOnlyPoolTexture(*vulkan, true);
    }
    SECTION("mirror readback pool") {
        RunGpuOnlyPoolTexture(*vulkan, false);
    }
}

TEST_CASE("NVN Vulkan: a fence nobody submits releases its waiter under load", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    RunParkedTimeoutUnderLoad(*vulkan);
}

TEST_CASE("NVN Vulkan: DK's two-queue event ping-pong runs in dependency order", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("queue 1 submitted first, imported pool") {
        RunEventPingPong(*vulkan, true, true);
    }
    SECTION("queue 0 submitted first, imported pool") {
        RunEventPingPong(*vulkan, true, false);
    }
    SECTION("queue 1 submitted first, mirror pool") {
        RunEventPingPong(*vulkan, false, true);
    }
    SECTION("queue 0 submitted first, mirror pool") {
        RunEventPingPong(*vulkan, false, false);
    }
}

TEST_CASE("NVN Vulkan: a wait nobody signals is released by the timeout, other queues run",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    RunEventNeverSignalled(*vulkan);
}

TEST_CASE("NVN Vulkan: a CPU event signal releases a waiting stream", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    RunEventCpuSignal(*vulkan);
}

TEST_CASE("NVN Vulkan: the GPU's unfinished writes to a range are known once the replay passed "
          "the submissions that make them",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 POOL_VA = 0x3700'0000;
    constexpr u64 EVENT_VA = 0x3800'0000;
    FakeMemory memory;
    memory.Add(POOL_VA, 64_KiB);
    memory.Add(EVENT_VA, 4_KiB);

    const auto poll_reached = [](Vulkan::Nvn::Context& context, u64 marker, u64 address, u64 size,
                                 u64 max_seq) {
        const auto begin = std::chrono::steady_clock::now();
        for (;;) {
            const VC::GpuWriteState state = context.PollGpuWrites(marker, address, size, max_seq);
            if (state.replay_reached) {
                return state;
            }
            REQUIRE(std::chrono::steady_clock::now() - begin < 20s);
            std::this_thread::sleep_for(1ms);
        }
    };

    SECTION("off: no marker") {
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
        CHECK(context.GpuWriteMarker() == 0);
    }
    SECTION("on: a write before an unsatisfied wait is known, one after it is not") {
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                     {.invalidate_wait_ms = 250});
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});
        // Seq 1 completes. Seq 2 writes +0, then waits for an event nobody signalled yet, then
        // would write +0x100: the flight stays unfinished while the first write is recorded.
        std::vector<u64> warm;
        PushFill(warm, POOL_VA + 0x800, 0x01010101u);
        PushEnd(warm);
        context.Submit(MakeSubmission(1, 0, std::move(warm)));
        REQUIRE(context.Wait(1, 20s));
        std::vector<u64> held;
        PushFill(held, POOL_VA, 0xAAAAAAAAu);
        PushWaitEvent(held, EVENT_VA, VC::EVENT_WAIT_EQUAL, 5);
        PushFill(held, POOL_VA + 0x100, 0xBBBBBBBBu);
        PushEnd(held);
        context.Submit(MakeSubmission(2, 1, std::move(held)));

        const u64 marker = context.GpuWriteMarker();
        REQUIRE(marker != 0);
        // The write of seq 2 before the wait is known; its write after the wait is not (not
        // recorded yet); seq 1's write is finished; a submission after `max_seq` is not asked for.
        VC::GpuWriteState state = poll_reached(context, marker, POOL_VA, 0x40, 2);
        CHECK(state.writer_seq == 2);
        CHECK(state.wait_limit_ms == 250);
        CHECK(context.PollGpuWrites(marker, POOL_VA, 0x40, 1).writer_seq == 0);
        CHECK(context.PollGpuWrites(marker, POOL_VA + 0x100, 0x40, 2).writer_seq == 0);
        CHECK(context.PollGpuWrites(marker, POOL_VA + 0x800, 0x40, 2).writer_seq == 0);
        CHECK(context.PollGpuWrites(marker, POOL_VA + 0x2000, 0x40, 2).writer_seq == 0);
        // A range that merely overlaps the write counts.
        CHECK(context.PollGpuWrites(marker, POOL_VA - 0x20, 0x40, 2).writer_seq == 2);
        CHECK_FALSE(context.Wait(2, 100ms));

        context.NotifyEventCpu(EVENT_VA, VC::EventCpuChange::Write, 5);
        REQUIRE(context.Wait(2, 20s));
        // Finished work is not waited for any more.
        CHECK(context.PollGpuWrites(marker, POOL_VA, 0x40, 2).writer_seq == 0);
        CHECK(context.PollGpuWrites(marker, POOL_VA + 0x100, 0x40, 2).writer_seq == 0);
        CHECK(context.Stats().ops_failed == 0);
    }
}

TEST_CASE("NVN Vulkan: a lost device completes the work without running it and tells the owner",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 POOL_VA = 0x7800'0000;
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0, 64_KiB);
    // The third submission reports VK_ERROR_DEVICE_LOST from its queue submit (the test hook); no
    // 15 s wait for Aftermath.
    // Declared before the Context: its workers stop before the callback state goes.
    CompletionLog log;
    std::mutex failure_mutex;
    std::vector<std::string> failures;
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.simulate_device_loss_from_seq = 3, .report_device_loss = false});
    context.SetCompletionCallback([&log](u64 seq) {
        std::scoped_lock lock{log.mutex};
        log.seqs.push_back(seq);
    });
    context.SetFailureCallback([&](std::string reason) {
        std::scoped_lock lock{failure_mutex};
        failures.push_back(std::move(reason));
    });
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});

    const auto fill = [&](u64 seq, u32 value) {
        std::vector<u64> list;
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = POOL_VA, .size = 256, .value = value});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, 0, std::move(list)));
    };
    fill(1, 0x11111111);
    fill(2, 0x22222222);
    REQUIRE(context.Wait(2, 20s));
    REQUIRE(context.Stats().device_lost == 0);
    REQUIRE(data[0] == 0x22);

    fill(3, 0x33333333); // the loss happens here
    fill(4, 0x44444444); // later work is completed without running
    fill(5, 0x55555555);
    // Nothing hangs: every submission completes, in order.
    REQUIRE(context.Wait(5, 20s));
    {
        std::scoped_lock lock{log.mutex};
        REQUIRE(log.seqs == std::vector<u64>{1, 2, 3, 4, 5});
    }
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    CHECK(stats.device_lost == 1);
    CHECK(stats.dropped_after_device_loss == 2); // seq 4 and 5 never ran
    CHECK(data[0] != 0x44);
    CHECK(data[0] != 0x55);
    {
        std::scoped_lock lock{failure_mutex};
        REQUIRE(failures.size() == 1); // told once, not per submission
        CHECK(failures[0].find("lost") != std::string::npos);
    }
    // A guest read-back of a texture cannot succeed and must not wait 10 s for the replay.
    const auto begin = std::chrono::steady_clock::now();
    CHECK_FALSE(context.FlushTextureToGuest(12345));
    CHECK(std::chrono::steady_clock::now() - begin < 2s);
}

TEST_CASE("NVN Vulkan: contexts created and destroyed back to back", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 POOL_VA = 0x7900'0000;
    // The second boot of a GUI session: a context dies right after its construction (its ASTC
    // decoder is still being built on a worker thread), then another one runs work.
    for (int boot = 0; boot < 3; ++boot) {
        FakeMemory memory;
        u8* data = memory.Add(POOL_VA, 64_KiB);
        std::memset(data, 0, 64_KiB);
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
        context.SetGuestMemory(&memory);
        if (boot == 0) {
            continue; // destroyed without a single command
        }
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});
        std::vector<u64> list;
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = POOL_VA, .size = 256, .value = 0x7F7F7F7Fu});
        PushEnd(list);
        context.Submit(MakeSubmission(1, 0, std::move(list)));
        REQUIRE(context.Wait(1, 20s));
        REQUIRE(data[0] == 0x7F);
        REQUIRE(context.Stats().ops_failed == 0);
        context.SetGuestMemory(nullptr);
    }
}

namespace {

void RunLinearTextureWriteback(HeadlessVulkan& vulkan, bool allow_import) {
    constexpr u64 TEX_POOL = 1;
    constexpr u64 READBACK_POOL = 2;
    constexpr u64 TEX_LINEAR_RT = 10;
    constexpr u64 TEX_BLOCK_SRC = 11;
    constexpr u64 TEX_LINEAR_COPY = 12;
    constexpr u64 TEX_POOL_VA = 0x7A00'0000;
    constexpr u64 READBACK_VA = 0x7B00'0000;
    constexpr u32 WIDTH = 5;
    constexpr u32 HEIGHT = 4;
    constexpr u32 STRIDE = 128;
    constexpr u32 ROW_BYTES = WIDTH * 8; // RGBA16F = 8 bytes/texel
    constexpr u64 RT_OFFSET = 256;
    constexpr u64 SRC_OFFSET = 4096;
    constexpr u64 COPY_OFFSET = 16384;

    FakeMemory memory;
    u8* tex_mem = memory.Add(TEX_POOL_VA, 64_KiB);
    u8* readback = memory.Add(READBACK_VA, 64_KiB);
    // Pre-fill the entire texture pool with fp16 -65504.0f (0xFBFF), the stale sentinel pattern
    // observed in MKW's recycled linear texture pools.
    for (size_t i = 0; i < 64_KiB; i += 2) {
        tex_mem[i] = 0xFF;
        tex_mem[i + 1] = 0xFB;
    }
    std::memset(readback, 0xCD, 64_KiB);

    // Declared before the Context: its workers stop before the callback state goes.
    CompletionLog log;
    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetCompletionCallback([&log](u64 seq) {
        std::scoped_lock lock{log.mutex};
        log.seqs.push_back(seq);
    });
    context.SetGuestMemory(&memory);

    context.CreatePool({.serial = TEX_POOL, .guest_address = TEX_POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = READBACK_POOL, .guest_address = READBACK_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});

    auto make_rgba16f = [](u64 serial, u64 offset, s32 flags, u32 stride) {
        VC::TextureDesc desc{};
        desc.target = 1;   // 2D
        desc.format = 0x29; // RGBA16F
        desc.flags = flags;
        desc.samples = 0;
        desc.width = WIDTH;
        desc.height = HEIGHT;
        desc.depth = 1;
        desc.levels = 1;
        desc.stride = stride;
        desc.serial = serial;
        desc.pool_serial = TEX_POOL;
        desc.pool_offset = offset;
        return desc;
    };

    context.CreateTexture(make_rgba16f(TEX_LINEAR_RT, RT_OFFSET,
                                       Vulkan::Nvn::TexFlag::LinearRenderTarget, STRIDE));
    context.CreateTexture(make_rgba16f(TEX_BLOCK_SRC, SRC_OFFSET, 0, 0));
    context.CreateTexture(
        make_rgba16f(TEX_LINEAR_COPY, COPY_OFFSET, Vulkan::Nvn::TexFlag::Linear, STRIDE));

    // Submission 1: GPU-clear the LinearRenderTarget texture and GPU-copy a block-linear RGBA16F
    // texture into the Linear texture.
    {
        std::vector<u64> list;
        VC::ClearTexturePayload clear_rt{};
        clear_rt.texture = {.serial = TEX_LINEAR_RT, .level = 0, .layer = 0};
        clear_rt.region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        clear_rt.layer_count = 1;
        clear_rt.is_integer = 0;
        const float rt_color[4] = {1.0f, -0.5f, 0.25f, 2.0f};
        std::memcpy(clear_rt.color, rt_color, sizeof(rt_color));
        Push(list, VC::Op::ClearTexture, clear_rt, 0xF);

        VC::ClearTexturePayload clear_src{};
        clear_src.texture = {.serial = TEX_BLOCK_SRC, .level = 0, .layer = 0};
        clear_src.region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        clear_src.layer_count = 1;
        clear_src.is_integer = 0;
        const float src_color[4] = {0.5f, -1.0f, 0.0f, 4.0f};
        std::memcpy(clear_src.color, src_color, sizeof(src_color));
        Push(list, VC::Op::ClearTexture, clear_src, 0xF);

        VC::CopyTextureToTexturePayload copy_lin{};
        copy_lin.src = {.serial = TEX_BLOCK_SRC, .level = 0, .layer = 0};
        copy_lin.src_region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        copy_lin.dst = {.serial = TEX_LINEAR_COPY, .level = 0, .layer = 0};
        copy_lin.dst_region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        Push(list, VC::Op::CopyTextureToTexture, copy_lin);
        PushEnd(list);

        context.Submit(MakeSubmission(1, 0, std::move(list),
                                      {TEX_LINEAR_RT, TEX_BLOCK_SRC, TEX_LINEAR_COPY}));
        REQUIRE(context.Wait(1, 20s));
    }

    {
        std::scoped_lock lock{log.mutex};
        REQUIRE(log.seqs == std::vector<u64>{1});
    }

    const auto check_linear_guest = [&](u64 offset, const std::array<u16, 4>& expected_texel) {
        const u8* base = tex_mem + offset;
        for (u32 y = 0; y < HEIGHT; ++y) {
            const u8* row = base + y * STRIDE;
            for (u32 x = 0; x < WIDTH; ++x) {
                std::array<u16, 4> actual{};
                std::memcpy(actual.data(), row + x * sizeof(actual), sizeof(actual));
                INFO("y=" << y << " x=" << x << " allow_import=" << allow_import);
                REQUIRE(actual == expected_texel);
            }
            for (u32 b = ROW_BYTES; b < STRIDE; ++b) {
                INFO("padding y=" << y << " byte=" << b << " allow_import=" << allow_import);
                REQUIRE(row[b] == ((b & 1) != 0 ? 0xFB : 0xFF));
            }
        }
    };

    // Exact IEEE-754 binary16 encodings for {1.0f, -0.5f, 0.25f, 2.0f} and {0.5f, -1.0f, 0.0f, 4.0f}.
    const std::array<u16, 4> expected_rt{0x3C00, 0xB800, 0x3400, 0x4000};
    const std::array<u16, 4> expected_copy{0x3800, 0xBC00, 0x0000, 0x4400};
    check_linear_guest(RT_OFFSET, expected_rt);
    check_linear_guest(COPY_OFFSET, expected_copy);

    // Submission 2: Without CPU writes, subsequent GPU reads must still see the GPU contents.
    {
        std::vector<u64> list;
        VC::CopyBufferTexturePayload rb{};
        rb.buffer = READBACK_VA;
        rb.texture = {.serial = TEX_LINEAR_RT, .level = 0, .layer = 0};
        rb.region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        Push(list, VC::Op::CopyTextureToBuffer, rb);
        PushEnd(list);
        context.Submit(MakeSubmission(2, 0, std::move(list), {TEX_LINEAR_RT}));
        REQUIRE(context.Wait(2, 20s));
    }
    for (u32 p = 0; p < WIDTH * HEIGHT; ++p) {
        std::array<u16, 4> actual{};
        std::memcpy(actual.data(), readback + p * sizeof(actual), sizeof(actual));
        REQUIRE(actual == expected_rt);
    }

    // Submission 3: Genuine CPU rewrites followed by NotifyCpuWrite must re-upload cleanly.
    const std::array<u16, 4> cpu_texel{0x3400, 0x3800, 0x3C00, 0x4000}; // {0.25f, 0.5f, 1.0f, 2.0f}
    for (u32 y = 0; y < HEIGHT; ++y) {
        u8* row = tex_mem + RT_OFFSET + y * STRIDE;
        for (u32 x = 0; x < WIDTH; ++x) {
            std::memcpy(row + x * sizeof(cpu_texel), cpu_texel.data(), sizeof(cpu_texel));
        }
    }
    context.NotifyCpuWrite(TEX_POOL_VA + RT_OFFSET, HEIGHT * STRIDE);
    {
        std::vector<u64> list;
        VC::CopyBufferTexturePayload rb{};
        rb.buffer = READBACK_VA + 4096;
        rb.texture = {.serial = TEX_LINEAR_RT, .level = 0, .layer = 0};
        rb.region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        Push(list, VC::Op::CopyTextureToBuffer, rb);
        PushEnd(list);
        context.Submit(MakeSubmission(3, 0, std::move(list), {TEX_LINEAR_RT}));
        REQUIRE(context.Wait(3, 20s));
    }
    for (u32 p = 0; p < WIDTH * HEIGHT; ++p) {
        std::array<u16, 4> actual{};
        std::memcpy(actual.data(), readback + 4096 + p * sizeof(actual), sizeof(actual));
        REQUIRE(actual == cpu_texel);
    }

    // Submission 4: Overlapping linear storages written A, B, A must flush in write order
    // so A's final value wins over B's earlier write.
    constexpr u64 TEX_LINEAR_A = 13;
    constexpr u64 TEX_LINEAR_B = 14;
    constexpr u64 OVERLAP_OFFSET = 32768;
    context.CreateTexture(make_rgba16f(TEX_LINEAR_A, OVERLAP_OFFSET,
                                       Vulkan::Nvn::TexFlag::LinearRenderTarget, STRIDE));
    context.CreateTexture(make_rgba16f(TEX_LINEAR_B, OVERLAP_OFFSET,
                                       Vulkan::Nvn::TexFlag::LinearRenderTarget, STRIDE));
    {
        std::vector<u64> list;
        auto push_clear = [&](u64 serial, float r, float g, float b, float a) {
            VC::ClearTexturePayload clear{};
            clear.texture = {.serial = serial, .level = 0, .layer = 0};
            clear.region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
            clear.layer_count = 1;
            clear.is_integer = 0;
            const float color[4] = {r, g, b, a};
            std::memcpy(clear.color, color, sizeof(color));
            Push(list, VC::Op::ClearTexture, clear, 0xF);
        };
        push_clear(TEX_LINEAR_A, 1.0f, -0.5f, 0.25f, 2.0f);
        push_clear(TEX_LINEAR_B, 0.5f, -1.0f, 0.0f, 4.0f);
        push_clear(TEX_LINEAR_A, 0.25f, 0.5f, 1.0f, 2.0f);
        PushEnd(list);
        context.Submit(MakeSubmission(4, 0, std::move(list), {TEX_LINEAR_A, TEX_LINEAR_B}));
        REQUIRE(context.Wait(4, 20s));
    }
    check_linear_guest(OVERLAP_OFFSET, cpu_texel);

    if (!allow_import) {
        // Submission 5: Padded mirror readback writes only the valid texel rows, preserving CPU
        // modifications written to row padding gaps between capture and completion.
        constexpr u64 EVENT_VA = READBACK_VA + 12288;
        class GapHookMemory final : public VC::GuestMemory {
        public:
            GapHookMemory(VC::GuestMemory& inner_, u8* gap_byte_)
                : inner{inner_}, gap_byte{gap_byte_} {}

            bool Read(u64 va, void* dst, size_t size) override {
                if (va == EVENT_VA && size == sizeof(u32) && !injected.exchange(true)) {
                    *gap_byte = 0x77;
                    const u32 signaled = 1;
                    std::memcpy(dst, &signaled, sizeof(signaled));
                    return true;
                }
                return inner.Read(va, dst, size);
            }
            bool Write(u64 va, const void* src, size_t size) override {
                return inner.Write(va, src, size);
            }
            u8* Pointer(u64 va) override {
                return inner.Pointer(va);
            }

            VC::GuestMemory& inner;
            u8* gap_byte;
            std::atomic<bool> injected{false};
        } gap_hook{memory, tex_mem + RT_OFFSET + ROW_BYTES + 4};

        context.SetGuestMemory(&gap_hook);
        std::vector<u64> list;
        VC::ClearTexturePayload clear{};
        clear.texture = {.serial = TEX_LINEAR_RT, .level = 0, .layer = 0};
        clear.region = {0, 0, 0, static_cast<s32>(WIDTH), static_cast<s32>(HEIGHT), 1};
        clear.layer_count = 1;
        clear.is_integer = 0;
        const float color[4] = {0.5f, -1.0f, 0.0f, 4.0f};
        std::memcpy(clear.color, color, sizeof(color));
        Push(list, VC::Op::ClearTexture, clear, 0xF);
        // Copy 4 bytes out of RT_OFFSET to force FlushLinearWritebacks (and mirror capture) before
        // WaitEvent reads EVENT_VA.
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = TEX_POOL_VA + RT_OFFSET,
                                            .dst = READBACK_VA + 8192,
                                            .size = 4});
        PushWaitEvent(list, EVENT_VA, VC::EVENT_WAIT_EQUAL, 1);
        PushEnd(list);
        context.Submit(MakeSubmission(5, 0, std::move(list), {TEX_LINEAR_RT}));
        const bool done = context.Wait(5, 20s);
        // Off the hook before it can go out of scope, also when the wait failed (the replay
        // reads the guest memory through the Context's current pointer).
        context.SetGuestMemory(&memory);
        REQUIRE(done);
        CHECK(gap_hook.injected.load());
        CHECK(tex_mem[RT_OFFSET + ROW_BYTES + 4] == 0x77);
        for (u32 y = 0; y < HEIGHT; ++y) {
            const u8* row = tex_mem + RT_OFFSET + y * STRIDE;
            for (u32 x = 0; x < WIDTH; ++x) {
                std::array<u16, 4> actual{};
                std::memcpy(actual.data(), row + x * sizeof(actual), sizeof(actual));
                REQUIRE(actual == expected_copy);
            }
        }
    }

    REQUIRE(context.Stats().ops_failed == 0);
    context.SetGuestMemory(nullptr);
}

} // namespace

TEST_CASE("NVN Vulkan: GPU writes to linear textures in CPU-readable pools write back to guest memory",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("imported pool") {
        RunLinearTextureWriteback(*vulkan, true);
    }
    SECTION("mirror pool") {
        RunLinearTextureWriteback(*vulkan, false);
    }
}
