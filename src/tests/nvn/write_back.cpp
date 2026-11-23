// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Write-back of GPU results held in device-local copies (nvn_write_back.h): the word-diff merge on
// the CPU and on the GPU (SKIP without a device), including CPU writes made after the snapshot that
// must survive.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_write_back.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace {

using Vulkan::Nvn::MergeChangedWords;
using Vulkan::Nvn::MergeRange;
using Vulkan::Nvn::WriteBackMerge;

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
};

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

/// 4-byte aligned storage, so offsets into it have the alignment phase the test chooses.
struct Bytes {
    explicit Bytes(size_t size) : words((size + 3) / 4, 0) {}
    u8* data() {
        return reinterpret_cast<u8*>(words.data());
    }
    std::vector<u32> words;
};

} // Anonymous namespace

TEST_CASE("NVN write-back merge: only changed words reach guest memory", "[nvn]") {
    Bytes guest(64);
    Bytes snapshot(64);
    Bytes result(64);
    for (u32 i = 0; i < 16; ++i) {
        snapshot.words[i] = 0x1000 + i;
    }
    guest.words = snapshot.words;
    result.words = snapshot.words;
    result.words[2] = 0xAAAA0002; // the GPU changed it
    result.words[9] = 0xAAAA0009;
    guest.words[5] = 0xC0DE0005;  // the CPU wrote it after the snapshot; the GPU did not touch it
    guest.words[9] = 0xC0DE0009;  // both wrote it: the GPU's value wins, as with a copy
    const u64 written = MergeChangedWords(std::span<u8>{guest.data(), 64},
                                          std::span<const u8>{snapshot.data(), 64},
                                          std::span<const u8>{result.data(), 64});
    CHECK(written == 8);
    CHECK(guest.words[2] == 0xAAAA0002);
    CHECK(guest.words[5] == 0xC0DE0005);
    CHECK(guest.words[9] == 0xAAAA0009);
    for (u32 i : {0u, 1u, 3u, 4u, 6u, 7u, 8u, 10u, 15u}) {
        CHECK(guest.words[i] == 0x1000 + i);
    }
}

TEST_CASE("NVN storage shadow: CPU write during a live shadow survives write-back", "[nvn]") {
    std::array<u32, 8> guest{0x1000, 0x1001, 0x1002, 0x1003, 0x1004, 0x1005, 0x1006, 0x1007};
    std::array<u32, 8> shadow = guest;
    shadow[1] = 0xAAAA0001; // GPU result outside the CPU-written word.
    shadow[6] = 0xAAAA0006;

    // NotifyCpuWriteNow refreshes the overlap in a live shadow before its whole-range write-back.
    guest[4] = 0xC0DE0004;
    shadow[4] = guest[4];
    guest = shadow;
    CHECK(guest[1] == 0xAAAA0001);
    CHECK(guest[4] == 0xC0DE0004);
    CHECK(guest[6] == 0xAAAA0006);

    // A mirror entry uses a word-diff merge: refresh both its result and snapshot so the CPU
    // update is not mistaken for a GPU change, while unrelated GPU-changed words still merge.
    Bytes mirror_guest(32);
    Bytes snapshot(32);
    Bytes result(32);
    for (u32 i = 0; i < 8; ++i) {
        snapshot.words[i] = 0x2000 + i;
    }
    mirror_guest.words = snapshot.words;
    result.words = snapshot.words;
    result.words[1] = 0xBBBB0001;
    result.words[6] = 0xBBBB0006;
    mirror_guest.words[4] = 0xD00D0004;
    result.words[4] = mirror_guest.words[4];
    snapshot.words[4] = mirror_guest.words[4];
    CHECK(MergeChangedWords(std::span<u8>{mirror_guest.data(), 32},
                            std::span<const u8>{snapshot.data(), 32},
                            std::span<const u8>{result.data(), 32}) == 8);
    CHECK(mirror_guest.words[1] == 0xBBBB0001);
    CHECK(mirror_guest.words[4] == 0xD00D0004);
    CHECK(mirror_guest.words[6] == 0xBBBB0006);
}

TEST_CASE("NVN write-back merge: unaligned ends go byte by byte, whole words stay whole", "[nvn]") {
    // The range is bytes [1, 11) of word-aligned storage: bytes 1-3 and 8-10 are the partial first
    // and last words, bytes 4-7 one whole word.
    Bytes guest(12);
    Bytes snapshot(12);
    Bytes result(12);
    for (u32 i = 0; i < 12; ++i) {
        snapshot.data()[i] = static_cast<u8>(0x10 + i);
    }
    std::memcpy(guest.data(), snapshot.data(), 12);
    std::memcpy(result.data(), snapshot.data(), 12);
    result.data()[0] = 0xF0;  // outside the range: never written
    result.data()[2] = 0xA2;  // GPU byte in the partial first word
    guest.data()[3] = 0xC3;   // CPU byte in the same word: survives (byte granular there)
    result.data()[4] = 0xA4;  // GPU byte in the whole word ...
    guest.data()[5] = 0xC5;   // ... and a CPU byte in it: the word is stored whole
    result.data()[10] = 0xAA; // GPU byte in the partial last word
    guest.data()[9] = 0xC9;   // CPU byte there: survives
    result.data()[11] = 0xFB; // outside the range
    const u64 written = MergeChangedWords(std::span<u8>{guest.data() + 1, 10},
                                          std::span<const u8>{snapshot.data() + 1, 10},
                                          std::span<const u8>{result.data() + 1, 10});
    CHECK(written == 1 + 4 + 1);
    const std::array<u8, 12> expected{0x10, 0x11, 0xA2, 0xC3, 0xA4, 0x15,
                                      0x16, 0x17, 0x18, 0xC9, 0xAA, 0x1B};
    for (u32 i = 0; i < 12; ++i) {
        INFO("byte " << i);
        CHECK(guest.data()[i] == expected[i]);
    }
}

TEST_CASE("NVN write-back merge: a CPU writing during the merge keeps its words", "[nvn]") {
    constexpr u32 WORDS = 1u << 16;
    Bytes guest(WORDS * 4);
    Bytes snapshot(WORDS * 4);
    Bytes result(WORDS * 4);
    for (u32 i = 0; i < WORDS; ++i) {
        snapshot.words[i] = i;
        // The GPU changed every even word; the odd ones belong to the CPU.
        result.words[i] = (i % 2 == 0) ? 0x80000000u | i : i;
    }
    guest.words = snapshot.words;
    std::atomic<bool> stop{false};
    std::atomic<u32> last{0};
    std::thread cpu{[&] {
        // A new particle record every iteration, into odd words the GPU does not touch.
        u32 value = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            for (u32 i = 1; i < WORDS; i += 2 * 97) {
                std::atomic_ref<u32>(guest.words[i]).store(value, std::memory_order_relaxed);
            }
            last.store(value, std::memory_order_release);
            ++value;
        }
    }};
    // Merge until the CPU has written its words many times over while merges ran.
    for (int round = 0; round < 50 || last.load(std::memory_order_acquire) < 1000; ++round) {
        MergeChangedWords(std::span<u8>{guest.data(), WORDS * 4},
                          std::span<const u8>{snapshot.data(), WORDS * 4},
                          std::span<const u8>{result.data(), WORDS * 4});
    }
    stop = true;
    cpu.join();
    const u32 final_value = last.load(std::memory_order_acquire);
    REQUIRE(final_value != 0);
    for (u32 i = 0; i < WORDS; ++i) {
        if (i % 2 == 0) {
            REQUIRE(guest.words[i] == (0x80000000u | i));
        } else if ((i - 1) % (2 * 97) == 0) {
            REQUIRE(guest.words[i] == final_value);
        } else {
            REQUIRE(guest.words[i] == i);
        }
    }
}

TEST_CASE("NVN write-back merge: the GPU merge writes what the CPU merge writes", "[nvn]") {
    using namespace Vulkan;
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Device& device = *vulkan->device;
    if (!WriteBackMerge::Supported(device)) {
        SKIP("the device has no push descriptors");
    }

    constexpr u64 BUFFER_SIZE = 64 * 1024;
    // Offsets equal modulo 4 but not aligned to anything else; the range ends in a partial word.
    constexpr u64 GUEST_OFF = 0x101;
    constexpr u64 RESULT_OFF = 0x2005;
    constexpr u64 SNAPSHOT_OFF = 0x109;
    constexpr u64 SIZE = 0x8002;

    std::mt19937 random(77);
    Bytes snapshot(BUFFER_SIZE);
    Bytes result(BUFFER_SIZE);
    Bytes guest(BUFFER_SIZE);
    for (u32& word : snapshot.words) {
        word = static_cast<u32>(random());
    }
    for (u32& word : guest.words) {
        word = static_cast<u32>(random());
    }
    for (u32& word : result.words) {
        word = static_cast<u32>(random());
    }
    // In the range: the guest starts as the snapshot, the result too, then each side changes its
    // own words (and some both).
    std::memcpy(guest.data() + GUEST_OFF, snapshot.data() + SNAPSHOT_OFF, SIZE);
    std::memcpy(result.data() + RESULT_OFF, snapshot.data() + SNAPSHOT_OFF, SIZE);
    std::vector<u64> gpu_words;
    std::vector<u64> cpu_words;
    for (u64 at = 4 - (GUEST_OFF & 3); at + 4 <= SIZE; at += 4) {
        const u32 pick = static_cast<u32>(random()) % 8;
        if (pick == 0 || pick == 1) {
            const u32 value = static_cast<u32>(random()) | 1;
            std::memcpy(result.data() + RESULT_OFF + at, &value, 4);
            gpu_words.push_back(at);
        }
        if (pick == 1 || pick == 2) {
            const u32 value = static_cast<u32>(random()) | 2;
            std::memcpy(guest.data() + GUEST_OFF + at, &value, 4);
            if (pick == 2) {
                cpu_words.push_back(at);
            }
        }
    }
    // The partial first word (3 bytes in the range): one GPU byte inside, one GPU byte outside
    // (another binding's), one CPU byte inside. The partial last word: one GPU byte inside, one
    // outside.
    result.data()[RESULT_OFF] ^= 0x5A;
    result.data()[RESULT_OFF - 1] ^= 0x5A;
    guest.data()[GUEST_OFF + 1] ^= 0x33;
    result.data()[RESULT_OFF + SIZE - 1] ^= 0x5A;
    result.data()[RESULT_OFF + SIZE] ^= 0x5A;

    // What the CPU merge makes of it.
    Bytes expected(BUFFER_SIZE);
    expected.words = guest.words;
    MergeChangedWords(std::span<u8>{expected.data() + GUEST_OFF, SIZE},
                      std::span<const u8>{snapshot.data() + SNAPSHOT_OFF, SIZE},
                      std::span<const u8>{result.data() + RESULT_OFF, SIZE});

    VmaAllocator vma = device.CreateThreadSafeAllocator();
    {
        MemoryAllocator allocator(device, vma);
        const auto make = [&](Bytes& bytes, MemoryUsage usage) {
            vk::Buffer buffer = allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = BUFFER_SIZE,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                usage);
            std::memcpy(buffer.Mapped().data(), bytes.data(), BUFFER_SIZE);
            buffer.Flush();
            return buffer;
        };
        vk::Buffer result_buffer = make(result, MemoryUsage::Upload);
        vk::Buffer snapshot_buffer = make(snapshot, MemoryUsage::Upload);
        vk::Buffer guest_buffer = make(guest, MemoryUsage::Download);

        WriteBackMerge merge;
        REQUIRE(merge.Init(device));
        REQUIRE(merge.Ready());
        // Diagnostic counters of EDEN_NX2_NVN_WB_MONITOR: slot 1 counts the main range's changes.
        vk::Buffer stats_buffer = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = 64,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Download);
        std::memset(stats_buffer.Mapped().data(), 0, 64);
        stats_buffer.Flush();
        merge.SetStats(*stats_buffer);

        // Every mode of the merge must leave the same guest bytes: the plain one and line mode with
        // units of 8 and 32 words (a unit with two or more changed words is stored whole, the
        // others' words as the guest holds them).
        // A range without `lines` (a page the guest touched) is merged exactly in every mode.
        struct Mode {
            u32 unit;
            bool range_lines;
        };
        for (const auto [unit, range_lines] : {Mode{0, false}, Mode{8, false}, Mode{8, true},
                                               Mode{32, false}, Mode{32, true}}) {
            INFO("unit words " << unit << " range lines " << range_lines);
            merge.SetLineMode(unit != 0, unit);
            std::memcpy(guest_buffer.Mapped().data(), guest.data(), BUFFER_SIZE);
            guest_buffer.Flush();
            std::memset(stats_buffer.Mapped().data(), 0, 64);
            stats_buffer.Flush();
            const vk::Device& logical = device.GetLogical();
            vk::CommandPool pool = logical.CreateCommandPool({
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = device.GetGraphicsFamily(),
            });
            vk::CommandBuffers buffers = pool.Allocate(1);
            const VkCommandBuffer raw = buffers[0];
            const vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
            cmd.Begin({
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr,
            });
            const std::array<MergeRange, 3> ranges{{
                {*result_buffer, RESULT_OFF, *snapshot_buffer, SNAPSHOT_OFF, *guest_buffer, GUEST_OFF,
                 SIZE, 1, range_lines},
                // Offsets of different phases cannot be merged word by word: refused, nothing written.
                {*result_buffer, RESULT_OFF + 1, *snapshot_buffer, SNAPSHOT_OFF, *guest_buffer, 0x9000,
                 0x100},
                {*result_buffer, 0, *snapshot_buffer, 0, *guest_buffer, 0xA000, 0},
            }};
            CHECK(merge.Record(cmd, ranges) == 1);
            const VkMemoryBarrier to_host{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                to_host, {}, {});
            cmd.End();
            vk::Fence fence = logical.CreateFence({
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
            });
            const VkSubmitInfo submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &raw,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr,
            };
            REQUIRE(device.GetGraphicsQueue().Submit(submit, *fence) == VK_SUCCESS);
            REQUIRE(fence.Wait() == VK_SUCCESS);
            guest_buffer.Invalidate();

            const u8* actual = guest_buffer.Mapped().data();
            size_t wrong = 0;
            for (u64 i = 0; i < BUFFER_SIZE; ++i) {
                if (actual[i] != expected.data()[i]) {
                    if (wrong < 8) {
                        WARN("byte " << i << ": gpu " << int(actual[i]) << " expected "
                                     << int(expected.data()[i]));
                    }
                    ++wrong;
                }
            }
            CHECK(wrong == 0);
            // And what that means: the GPU's words arrived, the CPU's words survived, the bytes
            // outside the range (and the refused range) were not touched.
            for (const u64 at : gpu_words) {
                CHECK(std::memcmp(actual + GUEST_OFF + at, result.data() + RESULT_OFF + at, 4) == 0);
            }
            for (const u64 at : cpu_words) {
                CHECK(std::memcmp(actual + GUEST_OFF + at, guest.data() + GUEST_OFF + at, 4) == 0);
            }
            CHECK(actual[GUEST_OFF - 1] == guest.data()[GUEST_OFF - 1]);
            CHECK(actual[GUEST_OFF] == result.data()[RESULT_OFF]);
            CHECK(actual[GUEST_OFF + 1] == guest.data()[GUEST_OFF + 1]);
            CHECK(actual[GUEST_OFF + SIZE - 1] == result.data()[RESULT_OFF + SIZE - 1]);
            CHECK(actual[GUEST_OFF + SIZE] == guest.data()[GUEST_OFF + SIZE]);
            CHECK(std::memcmp(actual + 0x9000, guest.data() + 0x9000, 0x100) == 0);
            // The counters: words the GPU changed (a partial first/last word counts when a byte
            // inside the range changed) and the 256-word groups (from the range's first word) that
            // hold one.
            u32 expected_words = 0;
            u32 expected_groups = 0;
            const u64 first_word = GUEST_OFF / 4;
            const u64 last_word = (GUEST_OFF + SIZE + 3) / 4;
            const u64 unit_words = unit == 0 ? 8 : unit; // the counters count units in every mode
            std::vector<bool> group_dirty((last_word - first_word + 2 * unit_words) / unit_words, false);
            for (u64 word = first_word; word < last_word; ++word) {
                bool changed = false;
                for (u64 byte = word * 4; byte < word * 4 + 4; ++byte) {
                    if (byte < GUEST_OFF || byte >= GUEST_OFF + SIZE) {
                        continue;
                    }
                    const u64 at = byte - GUEST_OFF;
                    changed |= snapshot.data()[SNAPSHOT_OFF + at] != result.data()[RESULT_OFF + at];
                }
                if (changed) {
                    ++expected_words;
                    group_dirty[word / unit_words - first_word / unit_words] = true;
                }
            }
            expected_groups = static_cast<u32>(std::ranges::count(group_dirty, true));
            stats_buffer.Invalidate();
            u32 counters[4];
            std::memcpy(counters, stats_buffer.Mapped().data(), sizeof(counters));
            CHECK(counters[0] == 0);
            CHECK(counters[1] == 0);
            CHECK(counters[2] == expected_words);
            CHECK(counters[3] == expected_groups);
        }
    }
    vmaDestroyAllocator(vma);
}

// Hidden benchmark ("tests.exe [.write-back-bench]"): what a merge of 16 MiB costs on the GPU for
// different change patterns, into host-cached memory (what an imported guest pool is) and into
// device-local memory. Tells whether the merge is bound by scanning the two inputs or by the stores
// that cross PCIe (VTX-HLE's OoT spends ~4 ms per frame in merges that change ~2.5 MiB).
TEST_CASE("NVN write-back merge: bench", "[.write-back-bench]") {
    using namespace Vulkan;
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Device& device = *vulkan->device;
    if (!WriteBackMerge::Supported(device)) {
        SKIP("the device has no push descriptors");
    }
    constexpr u64 SIZE = 16u << 20;
    constexpr u32 WORDS = static_cast<u32>(SIZE / 4);
    constexpr int REPEATS = 20;

    struct Pattern {
        const char* name;
        std::vector<u32> result;
        u32 changed_words;
    };
    std::mt19937 random(5);
    std::vector<u32> snapshot(WORDS);
    for (u32& word : snapshot) {
        word = static_cast<u32>(random());
    }
    std::vector<Pattern> patterns;
    const auto make = [&](const char* name, auto&& pick) {
        Pattern pattern{name, snapshot, 0};
        for (u32 i = 0; i < WORDS; ++i) {
            if (pick(i)) {
                pattern.result[i] ^= 0x9E3779B9u | 1u;
                ++pattern.changed_words;
            }
        }
        patterns.push_back(std::move(pattern));
    };
    make("none", [](u32) { return false; });
    make("sparse 5% words", [&](u32) { return random() % 100 < 5; });
    {
        // Measured on OoT: ~half of the 1 KiB groups with a change hold changed words in half of
        // their words. Here: 25% of the groups, half of their words at random.
        std::vector<bool> dirty(WORDS / 256);
        for (size_t i = 0; i < dirty.size(); ++i) {
            dirty[i] = random() % 4 == 0;
        }
        make("25% of 1 KiB groups, half of their words", [&](u32 i) {
            return dirty[i / 256] && (random() & 1) != 0;
        });
        make("25% of 1 KiB groups, all their words", [&](u32 i) { return dirty[i / 256]; });
    }
    make("runs of 16 words, 10%", [&](u32 i) { return (i / 16) % 10 == 3; });
    make("all words", [](u32) { return true; });

    VmaAllocator vma = device.CreateThreadSafeAllocator();
    {
        MemoryAllocator allocator(device, vma);
        const auto buffer_of = [&](VkBufferUsageFlags usage, MemoryUsage memory) {
            return allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = SIZE,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                memory);
        };
        const vk::Device& logical = device.GetLogical();
        vk::CommandPool pool = logical.CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = device.GetGraphicsFamily(),
        });
        const auto run = [&](auto&& record) {
            vk::CommandBuffers buffers = pool.Allocate(1);
            const VkCommandBuffer raw = buffers[0];
            const vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
            cmd.Begin({
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr,
            });
            record(cmd);
            cmd.End();
            vk::Fence fence = logical.CreateFence({
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
            });
            const VkSubmitInfo submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &raw,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr,
            };
            const auto begin = std::chrono::steady_clock::now();
            REQUIRE(device.GetGraphicsQueue().Submit(submit, *fence) == VK_SUCCESS);
            REQUIRE(fence.Wait() == VK_SUCCESS);
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                             begin)
                .count();
        };
        const auto upload = [&](const std::vector<u32>& words, vk::Buffer& destination) {
            vk::Buffer staging = buffer_of(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload);
            std::memcpy(staging.Mapped().data(), words.data(), SIZE);
            staging.Flush();
            run([&](vk::CommandBuffer cmd) {
                cmd.CopyBuffer(*staging, *destination, VkBufferCopy{0, 0, SIZE});
                // The fence only makes the copy available to the host: make it visible to the
                // merge's compute reads in the later submissions.
                const VkMemoryBarrier copied{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, copied);
            });
        };
        constexpr VkBufferUsageFlags STORAGE =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vk::Buffer snapshot_buffer = buffer_of(STORAGE, MemoryUsage::DeviceLocal);
        vk::Buffer result_buffer = buffer_of(STORAGE, MemoryUsage::DeviceLocal);
        upload(snapshot, snapshot_buffer);
        vk::Buffer host_guest = buffer_of(STORAGE, MemoryUsage::Download);
        vk::Buffer local_guest = buffer_of(STORAGE, MemoryUsage::DeviceLocal);

        WriteBackMerge merge;
        REQUIRE(merge.Init(device));
        std::printf("write-back merge bench, 16 MiB range, %d merges per measurement\n", REPEATS);
        for (const Pattern& pattern : patterns) {
            upload(pattern.result, result_buffer);
            for (const bool host : {true, false}) {
              for (const u32 unit : {0u, 8u, 32u}) {
                const bool lines = unit != 0;
                merge.SetLineMode(lines, unit);
                vk::Buffer& guest = host ? host_guest : local_guest;
                const std::array<MergeRange, 1> ranges{
                    {{*result_buffer, 0, *snapshot_buffer, 0, *guest, 0, SIZE, 0xFFFFFFFFu, true}}};
                double best = 1e9;
                for (int round = 0; round < 3; ++round) {
                    const double ms = run([&](vk::CommandBuffer cmd) {
                        for (int i = 0; i < REPEATS; ++i) {
                            merge.Record(cmd, ranges);
                            const VkMemoryBarrier barrier{
                                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                .pNext = nullptr,
                                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                                 VK_ACCESS_SHADER_WRITE_BIT,
                            };
                            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier,
                                                {}, {});
                        }
                    });
                    best = (std::min)(best, ms);
                }
                std::printf("  %-44s -> %-12s %-10s %7.3f ms per merge (%6.1f MiB changed)\n",
                            pattern.name, host ? "host memory" : "device-local",
                            lines ? "line mode" : "plain", best / REPEATS,
                            pattern.changed_words * 4.0 / (1 << 20));
              }
            }
        }
    }
    vmaDestroyAllocator(vma);
}
