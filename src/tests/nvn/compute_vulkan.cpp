// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// End-to-end compute: NVN2 (Ampere) compute programs dispatched through the replay on a headless
// Vulkan device (skipped without one). The programs are tiny, hand-assembled from encodings the
// DK compute shaders use (S2R, IMAD.MOV.U32 from the driver bank, IMAD, IADD3, STG.E, EXIT).

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/diag_request.h"
#include "common/game_presets.h"
#include "common/literals.h"
#include "common/settings.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/shader/nvn_program_map.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_bc_encode.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/renderer_vulkan/nvn/nvn_counters.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse_layout.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace {

namespace VC = VideoCommon::Nvn;
using namespace Common::Literals;
using namespace std::chrono_literals;

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
    /// Like the real guest memory: a range may cover several regions (host runs).
    bool Read(u64 va, void* dst, size_t size) override {
        auto* out = static_cast<u8*>(dst);
        while (size != 0) {
            const size_t chunk = Chunk(va, size);
            const u8* source = chunk ? Find(va, chunk) : nullptr;
            if (!source) {
                return false;
            }
            std::memcpy(out, source, chunk);
            out += chunk;
            va += chunk;
            size -= chunk;
        }
        return true;
    }
    bool Write(u64 va, const void* src, size_t size) override {
        const auto* in = static_cast<const u8*>(src);
        while (size != 0) {
            const size_t chunk = Chunk(va, size);
            u8* target = chunk ? Find(va, chunk) : nullptr;
            if (!target) {
                return false;
            }
            std::memcpy(target, in, chunk);
            in += chunk;
            va += chunk;
            size -= chunk;
        }
        return true;
    }
    u8* Pointer(u64 va) override {
        return Find(va, 1);
    }

    // Page tracking (GuestMemory::Track), as the core's page tracker: only armed pages count, and
    // only writes made through CpuWrite (the guest CPU), not through Write/Pointer (the HLE).
    u64 Track(u64 va, u64 size) override {
        if (!tracking) {
            return 0;
        }
        static_cast<void>(va);
        static_cast<void>(size);
        trackers.emplace_back();
        return trackers.size();
    }
    void Untrack(u64 id) override {
        trackers.at(id - 1) = {};
    }
    void Arm(u64 id, u64 va, u64 size) override {
        Tracker& t = trackers.at(id - 1);
        for (u64 page = va >> 12; page < (va + size + 0xFFF) >> 12; ++page) {
            t.armed.insert(page);
            t.dirty.erase(page);
        }
    }
    void TakeWritten(u64 id, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs) override {
        Tracker& t = trackers.at(id - 1);
        for (u64 page = va >> 12; page < (va + size + 0xFFF) >> 12; ++page) {
            if (t.dirty.erase(page) == 0) {
                continue;
            }
            if (!runs.empty() && runs.back().second == page << 12) {
                runs.back().second += 0x1000;
            } else {
                runs.emplace_back(page << 12, (page + 1) << 12);
            }
        }
    }
    std::shared_ptr<const std::atomic<u64>> WriteEpoch(u64 id) override {
        return trackers.at(id - 1).epoch;
    }
    bool AnyWritten(u64 id, u64 va, u64 size) override {
        const Tracker& t = trackers.at(id - 1);
        for (u64 page = va >> 12; page < (va + size + 0xFFF) >> 12; ++page) {
            if (t.dirty.contains(page)) {
                return true;
            }
        }
        return false;
    }
    /// A guest CPU store: the bytes change and armed pages become dirty.
    void CpuWrite(u64 va, const void* src, size_t size) {
        REQUIRE(Write(va, src, size));
        for (Tracker& t : trackers) {
            for (u64 page = va >> 12; page < (va + size + 0xFFF) >> 12; ++page) {
                if (t.armed.contains(page) && t.dirty.insert(page).second) {
                    t.epoch->fetch_add(1, std::memory_order_release);
                }
            }
        }
    }
    bool tracking{true};

private:
    struct Tracker {
        std::set<u64> armed;
        std::set<u64> dirty;
        std::shared_ptr<std::atomic<u64>> epoch{std::make_shared<std::atomic<u64>>(0)};
    };
    std::vector<Tracker> trackers;

    /// Bytes from `va` on that lie in one region, at most `size`; 0 when `va` is unmapped.
    size_t Chunk(u64 va, size_t size) const {
        for (const Region& region : regions) {
            if (va >= region.base && va - region.base < region.size) {
                return (std::min)(size, static_cast<size_t>(region.size - (va - region.base)));
            }
        }
        return 0;
    }
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

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
    std::mutex submit_mutex;
};

/// The device the NVN replay runs on in the emulator: created for a Switch 2 program, which makes
/// it enable VK_EXT_external_memory_host (pool imports; storage shadows and pool mirrors work on
/// imported pools only). The flag is put back afterwards.
std::unique_ptr<HeadlessVulkan> MakeHeadlessVulkan() {
    struct ProgramFlag {
        const bool previous = Settings::CurrentProgramIsSwitch2();
        ProgramFlag() {
            Settings::SetCurrentProgramIsSwitch2(true);
        }
        ~ProgramFlag() {
            Settings::SetCurrentProgramIsSwitch2(previous);
        }
    } const program_flag;
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
    } catch (const std::exception&) {
        return nullptr;
    }
}

// ---- record encoding ---------------------------------------------------------------------------

/// Header + `payload` + trailing `extra` bytes, padded to 8.
template <typename T>
void Push(std::vector<u64>& list, VC::Op op, const T& payload,
          std::span<const u8> extra = {}, u16 flags = 0) {
    const size_t bytes = sizeof(VC::RecordHeader) + sizeof(T) + extra.size();
    const size_t words = (bytes + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(op);
    header->flags = flags;
    header->size = static_cast<u32>(words * 8);
    auto* bytes_out = reinterpret_cast<u8*>(&list[at]) + sizeof(VC::RecordHeader);
    std::memcpy(bytes_out, &payload, sizeof(T));
    if (!extra.empty()) {
        std::memcpy(bytes_out + sizeof(T), extra.data(), extra.size());
    }
}

void PushEnd(std::vector<u64>& list) {
    list.push_back(0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list.back());
    header->op = static_cast<u16>(VC::Op::End);
    header->size = 8;
}

// ---- Ampere programs -----------------------------------------------------------------------------

struct Inst {
    u32 w[4];
};

constexpr u32 Sched = 0x000fe200;
constexpr Inst S2R(u32 rd, u32 special) {
    return {{0x00007919 | (rd << 16), 0, special << 8, 0x000e2200}};
}
/// IMAD.MOV.U32 rd, RZ, RZ, c[0][byte_offset]
constexpr Inst MovC0(u32 rd, u32 byte_offset) {
    return {{0xff007624 | (rd << 16), byte_offset << 6, 0x078e00ff, Sched}};
}
/// IMAD rd, ra, imm, RZ
constexpr Inst MulImm(u32 rd, u32 ra, u32 imm) {
    return {{0x00007824 | (rd << 16) | (ra << 24), imm, 0x078e02ff, Sched}};
}
/// IADD3 rd, ra, rb, RZ
constexpr Inst Add(u32 rd, u32 ra, u32 rb) {
    return {{0x00007210 | (rd << 16) | (ra << 24), rb, 0x07ffe0ff, Sched}};
}
/// IADD3 rd, ra, imm, RZ
constexpr Inst AddImm(u32 rd, u32 ra, u32 imm) {
    return {{0x00007810 | (rd << 16) | (ra << 24), imm, 0x07ffe0ff, Sched}};
}
/// STG.E.32 [ra.64], rs
constexpr Inst Stg(u32 ra, u32 rs) {
    return {{0x00007386 | (ra << 24), rs, 0x0010a900, Sched}};
}
constexpr Inst Exit{{0x0000794d, 0, 0x03800000, 0x000fea00}};

constexpr u32 SR_CTAID_X = 0x25;

void Put32(std::vector<u8>& bytes, size_t offset, u32 value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

struct Counts {
    u32 push_words{};
    u32 uniform{};
    u32 storage{};
    u32 textures{};
    u32 images{};
};

/// A compute program binary (control + code) with the driver bank layout of `counts`.
VC::ProgramDesc MakeProgram(u64 serial, std::span<const Inst> program, Counts counts) {
    const u32 inst_bytes = static_cast<u32>(program.size() * sizeof(Inst));
    const u32 c1_offset = (inst_bytes + 0xff) / 0x100 * 0x100;
    std::vector<u8> control(0x100, 0);
    Put32(control, 0, VC::ControlMagic);
    Put32(control, 4, VC::ControlVersion);
    Put32(control, 0x38, inst_bytes);
    Put32(control, 0x40, c1_offset);
    Put32(control, 0x44, c1_offset);
    control[0x4c] = 12;
    control[0x56] = 5; // compute
    Put32(control, 0x58, 1);
    Put32(control, 0x5c, 1);
    Put32(control, 0x60, 1);
    control[0xa9] = static_cast<u8>(counts.push_words);
    control[0xb1] = static_cast<u8>(counts.uniform);
    control[0xb2] = static_cast<u8>(counts.storage);
    control[0xb3] = static_cast<u8>(counts.textures);
    control[0xb6] = static_cast<u8>(counts.images);
    const u32 align = VC::ComputeEntryTableAlign;
    const u32 tables = (0x10 + counts.push_words * 4 + align - 1) / align * align;
    const u32 total = tables + (counts.uniform + counts.storage) * 0x10 +
                      (counts.textures + counts.images) * VC::HandleStride;
    control[0xba] = static_cast<u8>(total);
    control[0xbb] = static_cast<u8>(total >> 8);
    std::vector<u8> code(c1_offset, 0);
    std::memcpy(code.data(), program.data(), inst_bytes);
    VC::ProgramDesc desc{};
    desc.serial = serial;
    desc.control[5] = std::move(control);
    desc.code[5] = std::move(code);
    desc.stage_mask = 1u << 5;
    return desc;
}

constexpr u64 POOL_CPU_CACHED = 4;
constexpr u64 POOL_VA = 0x4000'0000;
constexpr u64 PROGRAM = 100;

/// Records BindProgram + BindStorageBuffers (slot 0) for the compute stage.
void BindCompute(std::vector<u64>& list, u64 program, u64 storage_address, u64 storage_size) {
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = program, .stages = 1u << 5});
    const VC::BufferRange range{storage_address, storage_size};
    Push(list, VC::Op::BindStorageBuffers,
         VC::BindBuffersPayload{.stage = 5, .first = 0, .count = 1},
         {reinterpret_cast<const u8*>(&range), sizeof(range)});
}

VC::Submission MakeSubmission(u64 seq, std::vector<u64> list, std::vector<u64> retained) {
    VC::Submission submission{};
    submission.seq = seq;
    submission.queue = 0;
    submission.lists.push_back(std::move(list));
    submission.retained_serials = std::move(retained);
    return submission;
}

/// Group `i` stores (i + 100) at word i of storage buffer 0:
///   R0 = CTAID.X; R2:R3 = c0[storage 0] + 4 * R0; STG [R2.64], R0 + 100
std::array<Inst, 8> FillProgram(u32 storage_entry_offset) {
    return {{S2R(0, SR_CTAID_X), MovC0(2, storage_entry_offset), MovC0(3, storage_entry_offset + 4),
             MulImm(5, 0, 4), Add(2, 2, 5), AddImm(6, 0, 100), Stg(2, 6), Exit}};
}

void RunFill(HeadlessVulkan& vulkan, bool allow_import, bool indirect) {
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    constexpr u64 STORAGE_VA = POOL_VA + 0x1000;
    constexpr u64 ARGS_VA = POOL_VA + 0x2000;
    constexpr u32 GROUPS = 6;
    const u32 args[3] = {GROUPS, 1, 1};
    std::memcpy(data + 0x2000, args, sizeof(args));

    Vulkan::Nvn::Context context(*vulkan.device, vulkan.submit_mutex, nullptr,
                                 {.allow_host_import = allow_import});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    // Uniform entry at 0x10, storage entry at 0x20.
    const auto program = FillProgram(0x20);
    context.CreateProgram(MakeProgram(PROGRAM, program, {.uniform = 1, .storage = 1}));

    std::vector<u64> list;
    BindCompute(list, PROGRAM, STORAGE_VA, 4096);
    if (indirect) {
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.indirect = ARGS_VA});
    } else {
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = GROUPS, .y = 1, .z = 1});
    }
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {PROGRAM, 1}));
    REQUIRE(context.Wait(1, 60s));

    u32 words[GROUPS + 2];
    std::memcpy(words, data + 0x1000, sizeof(words));
    for (u32 i = 0; i < GROUPS; ++i) {
        INFO("group " << i);
        REQUIRE(words[i] == 100 + i);
    }
    REQUIRE(words[GROUPS] == 0xCDCDCDCDu); // no group beyond the dispatch wrote anything
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    std::vector<VC::SinkStat> rows;
    context.CollectStats(rows);
    u64 dispatches = 0;
    u64 skipped = 0;
    for (const VC::SinkStat& row : rows) {
        if (row.reason == "dispatches") {
            dispatches = row.count;
        } else if (row.reason == "dispatches_skipped") {
            skipped = row.count;
        }
    }
    REQUIRE(dispatches == 1);
    REQUIRE(skipped == 0);
}

} // namespace

TEST_CASE("NVN compute: a dispatch runs its groups against a bound storage buffer",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("direct, host import") {
        RunFill(*vulkan, true, false);
    }
    SECTION("direct, mirror") {
        RunFill(*vulkan, false, false);
    }
}

TEST_CASE("NVN compute: an indirect dispatch reads its group counts from memory",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    SECTION("host import") {
        RunFill(*vulkan, true, true);
    }
    SECTION("mirror") {
        RunFill(*vulkan, false, true);
    }
}

TEST_CASE("NVN compute: push constants reach the driver bank", "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    // 4 push words at 0x10, one uniform entry at 0x20, one storage entry at 0x30. The program
    // stores push word 2 (c0+0x18) at the start of the storage buffer.
    const std::array<Inst, 6> program{{S2R(0, SR_CTAID_X), MovC0(2, 0x30), MovC0(3, 0x34),
                                       MovC0(6, 0x18), Stg(2, 6), Exit}};
    context.CreateProgram(MakeProgram(PROGRAM, program, {.push_words = 4, .uniform = 1, .storage = 1}));

    const u32 push[4] = {11, 22, 33, 44};
    std::vector<u64> list;
    BindCompute(list, PROGRAM, POOL_VA + 0x1000, 256);
    Push(list, VC::Op::PushConstants,
         VC::PushConstantsPayload{.compute = 1, .offset = 0, .size = sizeof(push)},
         {reinterpret_cast<const u8*>(push), sizeof(push)});
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {PROGRAM, 1}));
    REQUIRE(context.Wait(1, 60s));
    u32 value;
    std::memcpy(&value, data + 0x1000, sizeof(value));
    REQUIRE(value == 33);
    REQUIRE(context.Stats().ops_failed == 0);
}

TEST_CASE("NVN compute: the driver bank header holds its size and the group counts",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    // No push words, no uniform entry, one storage entry at 0x10: the bank is 0x20 bytes. The
    // program stores c0+0x0, +0x4, +0x8 and +0xc into consecutive words of the storage buffer.
    // DK's own dispatches show {bank size, group x, group y, group z} there (legacy cbuf 0).
    const std::array<Inst, 14> program{{MovC0(2, 0x10), MovC0(3, 0x14), MovC0(6, 0x0), Stg(2, 6),
                                        AddImm(2, 2, 4), MovC0(6, 0x4), Stg(2, 6),
                                        AddImm(2, 2, 4), MovC0(6, 0x8), Stg(2, 6),
                                        AddImm(2, 2, 4), MovC0(6, 0xc), Stg(2, 6), Exit}};
    context.CreateProgram(MakeProgram(PROGRAM, program, {.push_words = 0, .uniform = 0, .storage = 1}));

    std::vector<u64> list;
    BindCompute(list, PROGRAM, POOL_VA + 0x1000, 256);
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 3, .y = 2, .z = 5});
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {PROGRAM, 1}));
    REQUIRE(context.Wait(1, 60s));
    u32 header[4];
    std::memcpy(header, data + 0x1000, sizeof(header));
    CHECK(header[0] == 0x20);
    CHECK(header[1] == 3);
    CHECK(header[2] == 2);
    CHECK(header[3] == 5);
    REQUIRE(context.Stats().ops_failed == 0);
}

TEST_CASE("NVN compute: with asynchronous shaders a dispatch waits for its pipeline",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    struct RestoreSetting {
        bool previous = Settings::values.use_asynchronous_shaders.GetValue();
        ~RestoreSetting() {
            Settings::values.use_asynchronous_shaders.SetValue(previous);
        }
    } restore;
    Settings::values.use_asynchronous_shaders.SetValue(true);

    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    constexpr u32 GROUPS = 6;
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    const auto program = FillProgram(0x20);
    context.CreateProgram(MakeProgram(PROGRAM, program, {.uniform = 1, .storage = 1}));

    const auto dispatch = [&](u64 seq) {
        std::vector<u64> list;
        BindCompute(list, PROGRAM, POOL_VA + 0x1000, 4096);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = GROUPS, .y = 1, .z = 1});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list), {PROGRAM, 1}));
        REQUIRE(context.Wait(seq, 60s));
    };
    const auto row = [&](const char* name) {
        std::vector<VC::SinkStat> rows;
        context.CollectStats(rows);
        for (const VC::SinkStat& stat : rows) {
            if (stat.reason == name) {
                return stat.count;
            }
        }
        return u64{~0ull};
    };

    // The very first lookup compiles (the emulated path's compute pipelines wait for their build
    // too): the dispatch runs on its first use, nothing is skipped.
    dispatch(1);
    u32 words[GROUPS];
    std::memcpy(words, data + 0x1000, sizeof(words));
    for (u32 i = 0; i < GROUPS; ++i) {
        REQUIRE(words[i] == 100 + i);
    }
    REQUIRE(row("draw skipped awaiting async pipeline") == 0);
    REQUIRE(row("dispatches_skipped") == 0);
    REQUIRE(context.Stats().ops_failed == 0);
}

// ---- storage shadows (nvn_draw_shadow.cpp) -----------------------------------------------------
//
// A written storage range of an imported pool (up to 1 MiB) is bound through a device-local shadow;
// the guest's bytes must hold what the GPU wrote once the submission completed, whatever happened
// to the shadow in between.

namespace {

/// Group i stores (base + i) at word i of storage buffer 0.
std::array<Inst, 8> FillProgramBase(u32 storage_entry_offset, u32 base) {
    return {{S2R(0, SR_CTAID_X), MovC0(2, storage_entry_offset), MovC0(3, storage_entry_offset + 4),
             MulImm(5, 0, 4), Add(2, 2, 5), AddImm(6, 0, base), Stg(2, 6), Exit}};
}

constexpr u64 PROGRAM_2 = 101;
constexpr u64 PROGRAM_ARGS = 110;
constexpr u64 PROGRAM_COPY = 111;
constexpr u64 STORAGE_VA = POOL_VA + 0x1000;

/// LDG.E rd, [ra.64] (opcode 0x981; nvdisasm: "LDG.E R4[R2.64]" for rd 4, ra 2).
constexpr Inst LdgE(u32 rd, u32 ra) {
    return {{0x00007981 | (rd << 16) | (ra << 24), 0, 0x0c1e1900, 0x000ea800}};
}
constexpr u64 PROGRAM_RAW_READ = 113;
constexpr u64 PROGRAM_RAW_WRITE = 114;

/// Read a guest address carried in the push constants, then store the result through storage 0.
std::array<Inst, 7> RawReadProgram() {
    return {{MovC0(2, 0x10), MovC0(3, 0x14), LdgE(4, 2), MovC0(8, 0x20),
             MovC0(9, 0x24), Stg(8, 4), Exit}};
}

/// Store through both the raw guest pointer and storage descriptor (which must alias one mirror).
std::array<Inst, 8> RawWriteProgram() {
    return {{MovC0(2, 0x10), MovC0(3, 0x14), AddImm(6, 255, 0x1234), Stg(2, 6),
             MovC0(8, 0x20), MovC0(9, 0x24), Stg(8, 6), Exit}};
}

void PushRawAddress(std::vector<u64>& list, u64 address) {
    const VC::PushConstantsPayload payload{.compute = 1, .offset = 0, .size = 2 * sizeof(u32)};
    Push(list, VC::Op::PushConstants, payload,
         {reinterpret_cast<const u8*>(&address), sizeof(address)});
}

/// Group i stores word i of storage buffer 0 plus one at word i of storage buffer 1 (entries at
/// c0[0x20] and c0[0x30]).
std::array<Inst, 12> CopyPlusOneProgram() {
    return {{S2R(0, SR_CTAID_X), MovC0(2, 0x20), MovC0(3, 0x24), MulImm(5, 0, 4), Add(2, 2, 5),
             LdgE(4, 2), MovC0(8, 0x30), MovC0(9, 0x34), Add(8, 8, 5), AddImm(6, 4, 1), Stg(8, 6),
             Exit}};
}

/// Group i stores 100 + i at word i of storage buffer 0 and 200 + i at word i of storage buffer 1
/// (entries at c0[0x20] and c0[0x30]).
std::array<Inst, 13> FillTwoProgram() {
    return {{S2R(0, SR_CTAID_X), MovC0(2, 0x20), MovC0(3, 0x24), MulImm(5, 0, 4), Add(2, 2, 5),
             AddImm(6, 0, 100), Stg(2, 6), MovC0(8, 0x30), MovC0(9, 0x34), Add(8, 8, 5),
             AddImm(7, 0, 200), Stg(8, 7), Exit}};
}

constexpr u64 PROGRAM_TWO = 112;

/// BindProgram + BindStorageBuffers (slots 0 and 1) for the compute stage.
void BindCompute2(std::vector<u64>& list, u64 program, u64 source, u64 destination, u64 size) {
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = program, .stages = 1u << 5});
    const VC::BufferRange ranges[2] = {{source, size}, {destination, size}};
    Push(list, VC::Op::BindStorageBuffers,
         VC::BindBuffersPayload{.stage = 5, .first = 0, .count = 2},
         {reinterpret_cast<const u8*>(ranges), sizeof(ranges)});
}

u32 WordAt(const u8* data, u64 offset) {
    u32 value;
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
}

/// A context over one CPU-cached 64 KiB pool filled with 0xCD, two fill programs (base 100 and
/// 200), a fill program with base 1 (indirect group counts) and a copy program (word i of storage
/// 1 = word i of storage 0 + 1). `Imported()` is false only on a device without
/// VK_EXT_external_memory_host (or with imports switched off): the shadow and mirror paths are
/// not taken there, only the guest bytes are checked. On a device with it the pool must import,
/// so these tests never pass silently without taking the paths they are about. `pool_mirror`:
/// ContextOptions::pool_mirror; the second constructor takes the whole options.
struct ShadowRig {
    explicit ShadowRig(HeadlessVulkan& vulkan, u32 pool_mirror = 0)
        : ShadowRig(vulkan, Vulkan::Nvn::ContextOptions{.pool_mirror = pool_mirror}) {}
    ShadowRig(HeadlessVulkan& vulkan, const Vulkan::Nvn::ContextOptions& options)
        : context(*vulkan.device, vulkan.submit_mutex, nullptr, options),
          import_expected{vulkan.device->IsExtExternalMemoryHostSupported() &&
                          options.allow_host_import} {
        data = memory.Add(POOL_VA, 64_KiB);
        std::memset(data, 0xCD, 64_KiB);
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});
        context.CreateProgram(MakeProgram(PROGRAM, FillProgramBase(0x20, 100), {.uniform = 1, .storage = 1}));
        context.CreateProgram(MakeProgram(PROGRAM_2, FillProgramBase(0x20, 200), {.uniform = 1, .storage = 1}));
        context.CreateProgram(MakeProgram(PROGRAM_ARGS, FillProgramBase(0x20, 1), {.uniform = 1, .storage = 1}));
        context.CreateProgram(MakeProgram(PROGRAM_COPY, CopyPlusOneProgram(), {.uniform = 1, .storage = 2}));
        context.CreateProgram(MakeProgram(PROGRAM_TWO, FillTwoProgram(), {.uniform = 1, .storage = 2}));
        context.CreateProgram(MakeProgram(PROGRAM_RAW_READ, RawReadProgram(),
                                          {.push_words = 2, .storage = 1}));
        context.CreateProgram(MakeProgram(PROGRAM_RAW_WRITE, RawWriteProgram(),
                                          {.push_words = 2, .storage = 1}));
    }

    void Run(u64 seq, std::vector<u64> list) {
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list),
                                      {PROGRAM, PROGRAM_2, PROGRAM_ARGS, PROGRAM_COPY,
                                       PROGRAM_TWO, PROGRAM_RAW_READ, PROGRAM_RAW_WRITE, 1}));
        REQUIRE(context.Wait(seq, 60s));
    }
    /// After a Run: whether the pool was imported (required where the device can import it).
    [[nodiscard]] bool Imported() const {
        const Vulkan::Nvn::ContextStats stats = context.Stats();
        const bool imported = stats.pools_imported != 0;
        if (import_expected && !imported) {
            FAIL("the pool was not imported although the device has VK_EXT_external_memory_host "
                 "(mirrored " << stats.pools_mirrored << ", import failures "
                              << stats.host_import_failures << ")");
        }
        return imported;
    }

    FakeMemory memory;
    u8* data{};
    Vulkan::Nvn::Context context;
    const bool import_expected;
};

void Dispatch(std::vector<u64>& list, u64 program, u64 address, u64 size, u32 groups) {
    BindCompute(list, program, address, size);
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = groups, .y = 1, .z = 1});
}

} // namespace

TEST_CASE("NVN storage shadow: dispatches over one range share a shadow and write it back once",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan);
    std::vector<u64> list;
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
    Dispatch(list, PROGRAM_2, STORAGE_VA, 4096, 3);
    rig.Run(1, std::move(list));

    const u32 expected[] = {200, 201, 202, 103, 104, 105};
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == expected[i]);
    }
    // The range around the written words keeps the guest's bytes (the shadow started as a copy).
    REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
    REQUIRE(WordAt(rig.data, 0x1000 + 4092) == 0xCDCDCDCDu);
    REQUIRE(WordAt(rig.data, 0x0FFC) == 0xCDCDCDCDu);
    REQUIRE(WordAt(rig.data, 0x2000) == 0xCDCDCDCDu);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    if (rig.Imported()) {
        CHECK(stats.shadow_creates == 1);
        CHECK(stats.shadow_redirects == 2);
        CHECK(stats.shadow_flush_segment == 1);
        CHECK(stats.shadow_flush_conflict == 0);
        CHECK(stats.shadow_bytes_in == 4096);
        CHECK(stats.shadow_bytes_out == 4096);
    }
}

TEST_CASE("NVN storage shadow: a copy in the same submission reads what the dispatch wrote",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan);
    std::vector<u64> list;
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
    Push(list, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = STORAGE_VA, .dst = POOL_VA + 0x3000, .size = 32});
    rig.Run(1, std::move(list));

    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
        REQUIRE(WordAt(rig.data, 0x3000 + i * 4) == 100 + i);
    }
    REQUIRE(WordAt(rig.data, 0x3000 + 6 * 4) == 0xCDCDCDCDu);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    if (rig.Imported()) {
        CHECK(stats.shadow_creates == 1);
        CHECK(stats.shadow_flush_conflict == 1);
        CHECK(stats.shadow_flush_segment == 0);
    }
}

TEST_CASE("NVN storage shadow: a partly overlapping range is not served from the shadow",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan);
    std::vector<u64> list;
    // Words 0-5 of [0x1000, 0x2000), then words 4-6 of the same bytes through [0x1010, 0x2010).
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
    Dispatch(list, PROGRAM_2, STORAGE_VA + 16, 4096, 3);
    rig.Run(1, std::move(list));

    const u32 expected[] = {100, 101, 102, 103, 200, 201, 202};
    for (u32 i = 0; i < 7; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == expected[i]);
    }
    REQUIRE(WordAt(rig.data, 0x1000 + 7 * 4) == 0xCDCDCDCDu);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    if (rig.Imported()) {
        CHECK(stats.shadow_creates == 2);
        CHECK(stats.shadow_flush_conflict == 1);
        CHECK(stats.shadow_flush_segment == 1);
    }
}

TEST_CASE("NVN storage shadow: guest writes between submissions reach the next shadow",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan);
    std::vector<u64> first;
    Dispatch(first, PROGRAM, STORAGE_VA, 4096, 6);
    rig.Run(1, std::move(first));
    for (u32 i = 0; i < 6; ++i) {
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
    }
    // The game rewrites some of the bytes (and fills the rest of the range) after the fence.
    const u32 guest_word = 0x12345678;
    std::memcpy(rig.data + 0x1000 + 4 * 8, &guest_word, sizeof(guest_word));
    std::memcpy(rig.data + 0x1000 + 4 * 2, &guest_word, sizeof(guest_word));
    std::vector<u64> second;
    Dispatch(second, PROGRAM_2, STORAGE_VA, 4096, 2);
    rig.Run(2, std::move(second));

    REQUIRE(WordAt(rig.data, 0x1000) == 200);
    REQUIRE(WordAt(rig.data, 0x1004) == 201);
    REQUIRE(WordAt(rig.data, 0x1008) == guest_word); // not written by the second dispatch
    REQUIRE(WordAt(rig.data, 0x100c) == 103);
    REQUIRE(WordAt(rig.data, 0x1000 + 4 * 8) == guest_word);
    REQUIRE(rig.context.Stats().ops_failed == 0);
    if (rig.Imported()) {
        CHECK(rig.context.Stats().shadow_creates == 2);
        CHECK(rig.context.Stats().shadow_flush_segment == 2);
    }
}

namespace {

constexpr u64 TEXTURE_POOL_VA = 0x1000'0000;
constexpr u64 VERTEX_PROGRAM = 102;

/// A vertex program (NVN2 control + code behind the 0x50-byte graphics header).
VC::ProgramDesc MakeVertexProgram(u64 serial, std::span<const Inst> program) {
    const u32 header = VC::GraphicsHeaderBytes;
    const u32 inst_bytes = header + static_cast<u32>(program.size() * sizeof(Inst));
    const u32 c1_offset = (inst_bytes + 0xff) / 0x100 * 0x100;
    std::vector<u8> control(0x100, 0);
    Put32(control, 0, VC::ControlMagic);
    Put32(control, 4, VC::ControlVersion);
    Put32(control, 0x38, inst_bytes);
    Put32(control, 0x40, c1_offset);
    Put32(control, 0x44, c1_offset);
    control[0x4c] = 12;
    control[0x56] = 0; // vertex
    Put32(control, 0x58, 8);
    Put32(control, 0x5c, 4);
    Put32(control, 0x60, 2);
    std::vector<u8> code(c1_offset, 0);
    Put32(code, 0, 1u << 10); // SPH type: vertex
    std::memcpy(code.data() + header, program.data(), program.size_bytes());
    VC::ProgramDesc desc{};
    desc.serial = serial;
    desc.control[0] = std::move(control);
    desc.code[0] = std::move(code);
    desc.stage_mask = 1u << 0;
    return desc;
}

/// Every vertex stores `value` at word `word` of storage buffer 0 (the vertex driver bank holds the
/// storage entries at c0[0x110]).
std::array<Inst, 6> StoreProgram(u32 word, u32 value) {
    return {{MovC0(2, 0x110), MovC0(3, 0x114), AddImm(2, 2, word * 4), AddImm(6, 255, value),
             Stg(2, 6), Exit}};
}

} // namespace

TEST_CASE("NVN storage shadow: a render pass writing a storage range reaches the guest bytes",
          "[nvn][compute][mkw]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    const bool mkw = GENERATE(false, true);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.early_submit = false, .try_storage_vram = false});
    context.SetTitleId(mkw ? Settings::MarioKartWorldProgramId : Settings::OcarinaOfTimeProgramId);
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = 1 /* CPU_NO_ACCESS */});
    VC::TextureDesc target{};
    target.target = 1; // 2D
    target.format = 0x25;
    target.width = 16;
    target.height = 16;
    target.depth = 1;
    target.levels = 1;
    target.serial = 10;
    target.pool_serial = 2;
    target.pool_offset = 0;
    context.CreateTexture(target);
    // Two draws of one pass store through [0x1000, 0x2000) (one shadow, the second binding is
    // served from it); a third draw stores through [0x4000, 0x5000), which first appears in the
    // open pass and stays in host memory.
    const auto first = StoreProgram(0, 0x1337);
    const auto second = StoreProgram(1, 0x2468);
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, first));
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM + 1, second));

    std::vector<u64> list;
    const VC::RenderTargetRef color{.texture = {.serial = 10, .level = 0, .layer = 0}};
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
         {reinterpret_cast<const u8*>(&color), sizeof(color)});
    Push(list, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = 1});
    const auto draw = [&](u64 program, u64 address) {
        Push(list, VC::Op::SetDepthClamp, VC::EnablePayload{.enable = 0});
        Push(list, VC::Op::SetSampleMask, VC::U32Payload{.value = 0xFFFFFFFFu});
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = program, .stages = 1u << 0});
        const VC::BufferRange range{address, 4096};
        Push(list, VC::Op::BindStorageBuffers,
             VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&range), sizeof(range)});
        Push(list, VC::Op::Draw, VC::DrawPayload{.mode = 4, .first = 0, .count = 3, .instance_count = 1});
    };
    draw(VERTEX_PROGRAM, POOL_VA + 0x1000);
    draw(VERTEX_PROGRAM + 1, POOL_VA + 0x1000);
    draw(VERTEX_PROGRAM, POOL_VA + 0x4000);
    draw(VERTEX_PROGRAM, POOL_VA + 0x4000);
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {VERTEX_PROGRAM, VERTEX_PROGRAM + 1, 10, 1, 2}));
    REQUIRE(context.Wait(1, 60s));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    CHECK(WordAt(data, 0x1000) == 0x1337);
    CHECK(WordAt(data, 0x1004) == 0x2468);
    CHECK(WordAt(data, 0x1008) == 0xCDCDCDCDu);
    CHECK(WordAt(data, 0x4000) == 0x1337);
    CHECK(WordAt(data, 0x4004) == 0xCDCDCDCDu);
    CHECK(stats.driver_bank_uploads == (mkw ? 2u : 4u));
    CHECK(stats.pipeline_state_reuses == (mkw ? 1u : 0u));
    if (stats.pools_imported != 0) {
        CHECK(stats.shadow_creates == 1);
        CHECK(stats.shadow_redirects == 2);
        CHECK(stats.shadow_skipped_open_pass == 2);
        CHECK(stats.shadow_flush_segment == 1);
    }
}

TEST_CASE("Raw GPU addresses still resolve after a table rebuild", "[nvn][raw-table-rebuild]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    const bool mkw = GENERATE(false, true);
    ShadowRig rig(*vulkan);
    rig.context.SetTitleId(mkw ? Settings::MarioKartWorldProgramId : Settings::OcarinaOfTimeProgramId);
    constexpr u64 source_a = POOL_VA + 128_KiB;
    constexpr u64 source_b = POOL_VA + 256_KiB;
    constexpr u32 first = 0x12345678;
    constexpr u32 second = 0x87654321;
    std::memcpy(rig.memory.Add(source_a, 64_KiB), &first, sizeof(first));
    std::memcpy(rig.memory.Add(source_b, 64_KiB), &second, sizeof(second));
    const auto queue_read = [&](u64 seq, u64 source, u64 destination) {
        std::vector<u64> list;
        BindCompute(list, PROGRAM_RAW_READ, destination, 4096);
        PushRawAddress(list, source);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        PushEnd(list);
        rig.context.Submit(MakeSubmission(seq, std::move(list), {1, PROGRAM_RAW_READ, seq + 1}));
    };
    rig.context.CreatePool({.serial = 2, .guest_address = source_a, .size = 64_KiB,
                            .flags = 1 /* CPU_NO_ACCESS */});
    queue_read(1, source_a, POOL_VA + 4096);
    rig.context.CreatePool({.serial = 3, .guest_address = source_b, .size = 64_KiB,
                            .flags = 1 /* CPU_NO_ACCESS */});
    queue_read(2, source_b, POOL_VA + 8192);
    REQUIRE(rig.context.Wait(2, 60s));
    CHECK(WordAt(rig.data, 4096) == first);
    CHECK(WordAt(rig.data, 8192) == second);
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
    CHECK(stats.raw_global_rebuilds >= 2);
    CHECK(stats.raw_global_ranges_dropped == 0);
}

TEST_CASE("MKW first color attachment load preserves channels outside a masked clear",
          "[nvn][attachment-load]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool cpu_visible = GENERATE(false, true);
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    u8* texture = memory.Add(TEXTURE_POOL_VA, 1_MiB);
    std::memset(texture, 0x44, 1_MiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetTitleId(Settings::MarioKartWorldProgramId);
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = cpu_visible ? static_cast<u32>(POOL_CPU_CACHED)
                                             : 1u /* CPU_NO_ACCESS */});
    context.CreateTexture({.target = 1, .format = 0x25, .width = 16, .height = 16,
                           .depth = 1, .levels = 1, .serial = 10, .pool_serial = 2});
    std::vector<u64> list;
    const VC::RenderTargetRef color{.texture = {.serial = 10}};
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
         {reinterpret_cast<const u8*>(&color), sizeof(color)});
    Push(list, VC::Op::ClearColor,
         VC::ClearColorPayload{.index = 0, .mask = 1, .color = {std::bit_cast<u32>(1.0f)}});
    VC::CopyBufferTexturePayload readback{};
    readback.buffer = POOL_VA;
    readback.texture = {.serial = 10};
    readback.region = {0, 0, 0, 16, 16, 1};
    Push(list, VC::Op::CopyTextureToBuffer, readback);
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {10, 1, 2}));
    REQUIRE(context.Wait(1, 60s));
    CHECK(context.Stats().ops_failed == 0);
    CHECK(context.Stats().ops_unsupported == 0);
    const u32 expected = cpu_visible ? 0x444444FFu : 0x000000FFu;
    for (size_t i = 0; i < 16u * 16u; ++i) {
        REQUIRE(WordAt(data, i * sizeof(u32)) == expected);
    }
}

TEST_CASE("MKW graphics lookahead preserves live attachments across passes and lists",
          "[nvn][pipeline-lookahead]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.early_submit = false});
    context.SetTitleId(Settings::MarioKartWorldProgramId);
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = 1 /* CPU_NO_ACCESS */});
    VC::TextureDesc target{};
    target.target = 1;
    target.format = 0x25;
    target.width = 16;
    target.height = 16;
    target.depth = 1;
    target.levels = 1;
    target.serial = 10;
    target.pool_serial = 2;
    context.CreateTexture(target);
    target.format = 0x0B; // R32UI
    target.serial = 11;
    target.pool_offset = 64_KiB;
    context.CreateTexture(target);
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, StoreProgram(0, 0x1337)));

    const auto bind_target = [](std::vector<u64>& list, u64 serial) {
        const VC::RenderTargetRef color{.texture = {.serial = serial}};
        Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
             {reinterpret_cast<const u8*>(&color), sizeof(color)});
    };
    const auto draw = [](std::vector<u64>& list) {
        Push(list, VC::Op::Draw, VC::DrawPayload{.mode = 4, .count = 3, .instance_count = 1});
    };
    std::vector<u64> first;
    bind_target(first, 10);
    Push(first, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = 1});
    Push(first, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1});
    const VC::BufferRange range{POOL_VA + 0x1000, 4096};
    Push(first, VC::Op::BindStorageBuffers,
         VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
         {reinterpret_cast<const u8*>(&range), sizeof(range)});
    draw(first);
    PushEnd(first);
    auto submission = MakeSubmission(1, std::move(first), {VERTEX_PROGRAM, 10, 11, 1, 2});

    auto& next = submission.lists.emplace_back();
    bind_target(next, 11);
    Push(next, VC::Op::ClearColor,
         VC::ClearColorPayload{.index = 0, .mask = 0xF, .is_integer = 1, .color = {0x1234}});
    draw(next);
    bind_target(next, 10);
    draw(next);
    VC::CopyBufferTexturePayload readback{};
    readback.buffer = POOL_VA + 0x2000;
    readback.texture = {.serial = 11};
    readback.region = {0, 0, 0, 1, 1, 1};
    Push(next, VC::Op::CopyTextureToBuffer, readback);
    PushEnd(next);
    context.Submit(std::move(submission));
    REQUIRE(context.Wait(1, 60s));
    CHECK(context.Stats().ops_failed == 0);
    CHECK(context.Stats().ops_unsupported == 0);
    CHECK(WordAt(data, 0x1000) == 0x1337);
    CHECK(WordAt(data, 0x2000) == 0x1234);
}

TEST_CASE("NVN draw: vertex programs read the draw's base instance from c0[0x644]",
          "[nvn][compute]") {
    // OoT's text vertex program (f1f183d55f7989b9) indexes its glyph records with
    // c0[0x644] + instance id: two draws of one string share the records, the second starts at
    // its base instance. The SDK's instanced draw macros write the base instance there.
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    constexpr u64 INDICES = 0x8000;
    std::memset(data + INDICES, 0, 12); // three u32 indices 0
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = 1 /* CPU_NO_ACCESS */});
    VC::TextureDesc target{};
    target.target = 1; // 2D
    target.format = 0x25;
    target.width = 16;
    target.height = 16;
    target.depth = 1;
    target.levels = 1;
    target.serial = 10;
    target.pool_serial = 2;
    target.pool_offset = 0;
    context.CreateTexture(target);
    // Every vertex stores n = c0[0x644] + instance id at word n of storage buffer 0.
    //   MOV R6, c[0x0][0x644] (the text program's own word, register changed)
    //   ALD R7, a[0x2f8] (instance id)
    constexpr Inst MovBaseInstance{{0x00067a02, 0x00019100, 0x00000f00, Sched}};
    constexpr Inst AldInstanceId{{0xff077321, 0x0002f8ff, 0x00000000, 0x000e2800}};
    const std::array<Inst, 9> program{{MovC0(2, 0x110), MovC0(3, 0x114), MovBaseInstance,
                                       AldInstanceId, Add(7, 7, 6), MulImm(5, 7, 4),
                                       Add(2, 2, 5), Stg(2, 7), Exit}};
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, program));

    std::vector<u64> list;
    const VC::RenderTargetRef color{.texture = {.serial = 10, .level = 0, .layer = 0}};
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
         {reinterpret_cast<const u8*>(&color), sizeof(color)});
    Push(list, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = 1});
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1u << 0});
    const VC::BufferRange range{POOL_VA + 0x1000, 4096};
    Push(list, VC::Op::BindStorageBuffers,
         VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
         {reinterpret_cast<const u8*>(&range), sizeof(range)});
    Push(list, VC::Op::Draw, VC::DrawPayload{.mode = 4, .first = 0, .count = 3, .instance_count = 1});
    Push(list, VC::Op::Draw,
         VC::DrawPayload{.mode = 4, .first = 0, .count = 3, .instance_count = 2, .base_instance = 5});
    Push(list, VC::Op::DrawIndexed,
         VC::DrawIndexedPayload{.mode = 4, .index_type = 2, .count = 3, .base_vertex = 0,
                                .base_instance = 9, .instance_count = 2,
                                .index_address = POOL_VA + INDICES});
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {VERTEX_PROGRAM, 10, 1, 2}));
    REQUIRE(context.Wait(1, 60s));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    constexpr u32 UNTOUCHED = 0xCDCDCDCDu;
    const std::array<u32, 12> expected{0,         UNTOUCHED, UNTOUCHED, UNTOUCHED,
                                       UNTOUCHED, 5,         6,         UNTOUCHED,
                                       UNTOUCHED, 9,         10,        UNTOUCHED};
    for (u32 i = 0; i < expected.size(); ++i) {
        INFO("word " << i);
        CHECK(WordAt(data, 0x1000 + i * 4) == expected[i]);
    }
}

// ---- pool mirrors (nvn_pool_mirror.h, nvn_draw_shadow.cpp) ---------------------------------------
//
// With ContextOptions::pool_mirror the storage ranges of an imported pool are served from a VRAM
// copy of the pool that outlives command buffers: only pages that are stale in it are copied in
// (first use, guest CPU writes the page tracker saw, replay writes through the pool's buffer), and
// a written range goes back into the guest bytes word by word at the end of the command buffer.


TEST_CASE("NVN storage VRAM raw aliases preserve CPU and GPU visibility", "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }

    ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{});
    constexpr u64 SOURCE = POOL_VA + 0x1000;
    constexpr u64 DESTINATION = POOL_VA + 0x3000;
    constexpr u64 TARGET = POOL_VA + 0x4000;

    // Create the pool mirror before the CPU modifies a source page.
    std::vector<u64> seed;
    Dispatch(seed, PROGRAM, SOURCE, 4096, 1);
    rig.Run(1, std::move(seed));
    if (!rig.Imported()) {
        SKIP("raw-global storage mirroring requires host-imported guest pools");
    }
    // Default options enable storage VRAM on a device with the required raw-global features.
    REQUIRE(rig.context.Stats().vram_mirrors == 1);
    // The first raw read populates every pool page that a raw pointer could address.
    std::vector<u64> warm_read;
    BindCompute(warm_read, PROGRAM_RAW_READ, DESTINATION, 4096);
    PushRawAddress(warm_read, SOURCE);
    Push(warm_read, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
    rig.Run(2, std::move(warm_read));
    const u64 copy_in_before_cpu_write = rig.context.Stats().vram_copy_in_bytes;

    // Only the page the CPU changed should be copied in again.
    const u32 cpu_value = 0x12345678;
    rig.memory.CpuWrite(SOURCE, &cpu_value, sizeof(cpu_value));
    std::vector<u64> read;
    BindCompute(read, PROGRAM_RAW_READ, DESTINATION, 4096);
    PushRawAddress(read, SOURCE);
    Push(read, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
    rig.Run(3, std::move(read));
    REQUIRE(WordAt(rig.data, 0x3000) == cpu_value);
    CHECK(rig.context.Stats().vram_copy_in_bytes - copy_in_before_cpu_write == 4_KiB);

    // Raw and descriptor stores in one dispatch alias the same device-local pool copy.
    std::vector<u64> write;
    BindCompute(write, PROGRAM_RAW_WRITE, TARGET, 4096);
    PushRawAddress(write, TARGET);
    Push(write, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
    rig.Run(4, std::move(write));
    REQUIRE(WordAt(rig.data, 0x4000) == 0x1234);
    const u64 copy_in_after_write = rig.context.Stats().vram_copy_in_bytes;

    // A later raw read must consume the GPU-written mirror without importing stale guest bytes.
    std::vector<u64> reread;
    BindCompute(reread, PROGRAM_RAW_READ, DESTINATION, 4096);
    PushRawAddress(reread, TARGET);
    Push(reread, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
    rig.Run(5, std::move(reread));
    REQUIRE(WordAt(rig.data, 0x3000) == 0x1234);
    CHECK(rig.context.Stats().vram_copy_in_bytes == copy_in_after_write);
    REQUIRE(WordAt(rig.data, 0x4000) == 0x1234);

    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    REQUIRE(stats.raw_global_draws >= 2);
    if (rig.Imported()) {
        CHECK(stats.vram_copy_in_bytes >= 64_KiB);
        CHECK(stats.vram_write_backs_plain == 0);
    }
}

TEST_CASE("NVN pool mirror: raw-global draws keep the mirror entries live (TRY_RAW_KEEP_ENTRIES)",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    // Written for TRY_STORAGE_RAW_WRITE_THROUGH off (now on by default): pin it for this test.
    namespace D = Common::DiagRequest;
    struct SwitchGuard {
        bool initial = D::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH");
        SwitchGuard() {
            REQUIRE(D::SetTrySwitch("TRY_STORAGE_RAW_WRITE_THROUGH", false));
        }
        ~SwitchGuard() {
            static_cast<void>(D::SetTrySwitch("TRY_STORAGE_RAW_WRITE_THROUGH", initial));
        }
    } guard;
    constexpr u64 TARGET = POOL_VA + 0x4000;
    constexpr u64 RAW_A = POOL_VA + 0x8000;
    constexpr u64 RAW_B = POOL_VA + 0x9000;
    // The same list with the switch off (every raw draw ends the entries and the raw-writing
    // draws snapshot the pool again) and on: identical guest bytes, fewer entry round trips.
    // EDEN_NX2_TRY_RAW_KEEP_ENTRIES is on unless the environment says 0 (the unset case).
    const char* const env = std::getenv("EDEN_NX2_TRY_RAW_KEEP_ENTRIES");
    const bool default_on = env == nullptr || *env == '\0' || std::string_view{env} != "0";
    for (const std::optional<bool> option : {std::optional<bool>{false}, std::optional<bool>{true},
                                             std::optional<bool>{}}) {
        const bool keep = option.value_or(default_on);
        INFO("raw_keep_entries " << (option ? static_cast<int>(*option) : -1));
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{.pool_mirror = 1u,
                                                           .raw_keep_entries = option});
        std::vector<u64> list;
        Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
        BindCompute(list, PROGRAM_RAW_WRITE, TARGET, 4096);
        PushRawAddress(list, RAW_A);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        Dispatch(list, PROGRAM_2, STORAGE_VA, 4096, 3);
        BindCompute(list, PROGRAM_RAW_WRITE, TARGET + 0x100, 4096);
        PushRawAddress(list, RAW_B);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        rig.Run(1, std::move(list));
        if (!rig.Imported()) {
            SKIP("storage VRAM entries require host-imported guest pools");
        }
        const u32 expected[] = {200, 201, 202, 103, 104, 105};
        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == expected[i]);
        }
        REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
        REQUIRE(WordAt(rig.data, 0x4000) == 0x1234);
        REQUIRE(WordAt(rig.data, 0x4100) == 0x1234);
        REQUIRE(WordAt(rig.data, 0x8000) == 0x1234);
        REQUIRE(WordAt(rig.data, 0x9000) == 0x1234);
        REQUIRE(WordAt(rig.data, 0x8004) == 0xCDCDCDCDu);
        REQUIRE(WordAt(rig.data, 0x3000) == 0xCDCDCDCDu);
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.ops_failed == 0);
        REQUIRE(stats.ops_unsupported == 0);
        if (keep) {
            // One entry for the first dispatch, one whole-pool entry for both raw-writing draws.
            CHECK(stats.shadow_flush_raw == 0);
            CHECK(stats.shadow_creates == 2);
        } else {
            // The first raw draw ends the first entry, the second ends the whole-pool entry that
            // the first one made, and the second raw-writing draw snapshots the pool again.
            CHECK(stats.shadow_flush_raw == 2);
            CHECK(stats.shadow_creates == 3);
        }
    }
}

TEST_CASE("NVN storage mirror: raw writes reach guest memory without whole-pool snapshots",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    namespace D = Common::DiagRequest;
    const bool initial = D::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH");
    struct SwitchReset {
        bool initial;
        ~SwitchReset() {
            static_cast<void>(D::SetTrySwitch("TRY_STORAGE_RAW_WRITE_THROUGH", initial));
        }
    } reset{initial};
    for (const bool write_through : {false, true}) {
        REQUIRE(D::SetTrySwitch("TRY_STORAGE_RAW_WRITE_THROUGH", write_through));
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{.pool_mirror = 1u,
                                                          .raw_keep_entries = true});
        std::vector<u64> list;
        Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
        BindCompute(list, PROGRAM_RAW_WRITE, POOL_VA + 0x4000, 4096);
        PushRawAddress(list, POOL_VA + 0x8000);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        BindCompute2(list, PROGRAM_COPY, POOL_VA + 0x8000, POOL_VA + 0xA000, 4096);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        rig.Run(1, std::move(list));
        if (!rig.Imported()) {
            SKIP("storage VRAM write-through requires host-imported guest pools");
        }
        REQUIRE(WordAt(rig.data, 0x4000) == 0x1234);
        REQUIRE(WordAt(rig.data, 0x8000) == 0x1234);
        REQUIRE(WordAt(rig.data, 0xA000) == 0x1235);
        REQUIRE(WordAt(rig.data, 0xA004) == 0xCDCDCDCEu);
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.ops_failed == 0);
        REQUIRE(stats.ops_unsupported == 0);
        CHECK(stats.shadow_flush_raw == 0);
        if (write_through) {
            // The initial descriptor range and the two small descriptor writes are snapshotted;
            // raw writes do not add a full-pool snapshot.
            CHECK(stats.shadow_bytes_in == 3 * 4096);
        } else {
            // The raw write snapshots the whole imported pool for the segment merge.
            CHECK(stats.shadow_bytes_in == 4096 + 64_KiB);
        }
    }
}

TEST_CASE("NVN storage VRAM refuses pools above its configured size limit", "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    // vtxcov 0: TRY_VTXCOV_BIGPOOL (on by default) gives an oversized pool a read-only mirror.
    ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{
                                .try_storage_vram = true,
                                .try_storage_vram_max_bytes = 32_KiB,
                                .vtxcov = 0u});
    std::vector<u64> list;
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 1);
    rig.Run(1, std::move(list));
    if (!rig.Imported()) {
        SKIP("storage VRAM pool-size eligibility requires host-imported guest pools");
    }

    REQUIRE(WordAt(rig.data, 0x1000) == 100);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    REQUIRE(stats.vram_mirrors == 0);
    REQUIRE(stats.vram_mirror_refused == 1);
}

TEST_CASE("NVN storage VRAM: a written range of a refused pool goes through a copy shadow "
          "(TRY_REFUSED_SHADOW)",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    for (const bool refused_shadow : {false, true}) {
        INFO("refused_shadow " << refused_shadow);
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{
                                    .try_storage_vram = true,
                                    .try_storage_vram_max_bytes = 32_KiB,
                                    .vtxcov = 0u,
                                    .refused_shadow = refused_shadow});
        constexpr u64 ARGS = POOL_VA + 0x2000;
        constexpr u64 TARGET = POOL_VA + 0x3000;
        std::vector<u64> list;
        Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
        Dispatch(list, PROGRAM_2, STORAGE_VA, 4096, 3);
        // Groups 0-2 store 1, 2, 3: the indirect dispatch runs 1 x 2 x 3 groups, all with CTAID.X 0.
        Dispatch(list, PROGRAM_ARGS, ARGS, 4096, 3);
        BindCompute(list, PROGRAM_2, TARGET, 4096);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.indirect = ARGS});
        rig.Run(1, std::move(list));
        if (!rig.Imported()) {
            SKIP("storage VRAM pool-size eligibility requires host-imported guest pools");
        }

        const u32 expected[] = {200, 201, 202, 103, 104, 105};
        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == expected[i]);
        }
        REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
        REQUIRE(WordAt(rig.data, 0x0FFC) == 0xCDCDCDCDu);
        REQUIRE(WordAt(rig.data, 0x2000) == 1);
        REQUIRE(WordAt(rig.data, 0x2004) == 2);
        REQUIRE(WordAt(rig.data, 0x2008) == 3);
        REQUIRE(WordAt(rig.data, 0x3000) == 200);
        REQUIRE(WordAt(rig.data, 0x3004) == 0xCDCDCDCDu);
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.ops_failed == 0);
        REQUIRE(stats.ops_unsupported == 0);
        REQUIRE(stats.vram_mirrors == 0);
        REQUIRE(stats.vram_mirror_refused == 1);
        if (refused_shadow) {
            // The two dispatches share the storage shadow; the indirect arguments' shadow goes
            // back before the indirect dispatch reads them from the guest memory.
            CHECK(stats.shadow_creates == 3);
            CHECK(stats.shadow_skipped_no_room == 0);
            CHECK(stats.shadow_flush_conflict == 1);
            CHECK(stats.shadow_flush_segment == 2);
            CHECK(stats.shadow_bytes_out == 3 * 4096);
        } else {
            CHECK(stats.shadow_creates == 0);
            CHECK(stats.shadow_skipped_no_room >= 3);
        }
    }
}
TEST_CASE("NVN pool mirror: a written range stays in VRAM across submissions",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan, 1);
    std::vector<u64> first;
    Dispatch(first, PROGRAM, STORAGE_VA, 4096, 6);
    rig.Run(1, std::move(first));
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
    }
    REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
    REQUIRE(WordAt(rig.data, 0x0FFC) == 0xCDCDCDCDu);
    std::vector<u64> second;
    Dispatch(second, PROGRAM_2, STORAGE_VA, 4096, 3);
    rig.Run(2, std::move(second));

    const u32 expected[] = {200, 201, 202, 103, 104, 105};
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == expected[i]);
    }
    REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    if (rig.Imported()) {
        CHECK(stats.vram_mirrors == 1);
        CHECK(stats.vram_write_ranges == 2);
        // The page was copied in once: the second submission found it current.
        CHECK(stats.vram_copy_in_bytes == 4096);
        CHECK(stats.shadow_flush_segment == 2);
        CHECK(stats.vram_write_backs_plain == 0);
    }
}

TEST_CASE("NVN pool mirror: a page the guest CPU wrote is copied in again before a read",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan, 1);
    constexpr u64 SOURCE = POOL_VA + 0x1000;
    constexpr u64 TARGET = POOL_VA + 0x3000;
    std::vector<u64> first;
    Dispatch(first, PROGRAM, SOURCE, 4096, 6);
    rig.Run(1, std::move(first));
    // The game rewrites one word of the range after the fence (the page tracker sees it).
    const u32 guest_word = 0x12345678;
    rig.memory.CpuWrite(SOURCE + 2 * 4, &guest_word, sizeof(guest_word));
    std::vector<u64> second;
    BindCompute2(second, PROGRAM_COPY, SOURCE, TARGET, 4096);
    Push(second, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
    rig.Run(2, std::move(second));

    const u32 expected[] = {101, 102, guest_word + 1, 104, 105, 106};
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x3000 + i * 4) == expected[i]);
    }
    REQUIRE(WordAt(rig.data, 0x3000 + 6 * 4) == 0xCDCDCDCDu);
    REQUIRE(WordAt(rig.data, 0x1000 + 2 * 4) == guest_word);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    if (rig.Imported()) {
        // Source page (first use), source page again (CPU write), target page (first use).
        CHECK(stats.vram_copy_in_bytes == 3 * 4096);
        CHECK(stats.vram_bindings >= 3);
    }
}

TEST_CASE("NVN pool mirror: a guest flush only copies in the pages the guest wrote",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 SOURCE = POOL_VA + 0x1000;
    constexpr u64 TARGET = POOL_VA + 0x3000;
    enum class After { Nothing, GuestWrite, HostWrite };
    struct Case {
        std::optional<bool> flush_tracked; ///< unset: the switch's default decides
        After after;
    };
    // EDEN_NX2_TRY_FLUSH_TRACKED is on unless the environment says 0.
    const char* const env = std::getenv("EDEN_NX2_TRY_FLUSH_TRACKED");
    const bool default_on = env == nullptr || *env == '\0' || std::string_view{env} != "0";
    // `flush_tracked` off: a flush always makes the mirror copy the range in again.
    for (const Case test : {Case{false, After::Nothing}, Case{true, After::Nothing},
                            Case{true, After::GuestWrite}, Case{true, After::HostWrite},
                            Case{std::nullopt, After::Nothing},
                            Case{std::nullopt, After::GuestWrite}}) {
        const bool tracked = test.flush_tracked.value_or(default_on);
        INFO("flush_tracked " << (test.flush_tracked ? static_cast<int>(*test.flush_tracked) : -1)
                              << " after " << static_cast<int>(test.after));
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{.pool_mirror = 1u,
                                                           .flush_tracked = test.flush_tracked});
        std::vector<u64> first;
        Dispatch(first, PROGRAM, SOURCE, 4096, 6);
        rig.Run(1, std::move(first));
        const u32 word = 0x12345678;
        u32 expected[] = {101, 102, 103, 104, 105, 106};
        switch (test.after) {
        case After::Nothing:
            // The game flushes the range it did not write.
            rig.context.NotifyCpuWrite(SOURCE, 4096);
            break;
        case After::GuestWrite:
            rig.memory.CpuWrite(SOURCE + 2 * 4, &word, sizeof(word));
            rig.context.NotifyCpuWrite(SOURCE, 4096);
            expected[2] = word + 1;
            break;
        case After::HostWrite:
            // A host service's store through a host pointer: the tracker never sees it.
            REQUIRE(rig.memory.Write(SOURCE + 2 * 4, &word, sizeof(word)));
            rig.context.NotifyHostWrite(SOURCE + 2 * 4, 4);
            expected[2] = word + 1;
            break;
        }
        std::vector<u64> second;
        BindCompute2(second, PROGRAM_COPY, SOURCE, TARGET, 4096);
        Push(second, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        rig.Run(2, std::move(second));
        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x3000 + i * 4) == expected[i]);
        }
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.ops_failed == 0);
        if (rig.Imported()) {
            // Source page and target page on first use, the source page again when it was written
            // (or flushed without the flush filter).
            const bool copied_again = !tracked || test.after != After::Nothing;
            CHECK(stats.vram_copy_in_bytes == (copied_again ? 3u : 2u) * 4096);
            if (tracked && test.after == After::Nothing) {
                CHECK(stats.vram_flush_pages_kept == 1);
                CHECK(stats.vram_flush_pages_taken == 0);
            }
        }
    }
}

TEST_CASE("NVN pool mirror: a replay write through the pool's buffer is copied in again",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan, 1);
    constexpr u64 SOURCE = POOL_VA + 0x1000;
    constexpr u64 TARGET = POOL_VA + 0x3000;
    std::vector<u64> list;
    // The copy program reads the source through the mirror; then a buffer clear (through the
    // pool's own buffer) rewrites it; the second copy must read the cleared words.
    BindCompute2(list, PROGRAM_COPY, SOURCE, TARGET, 4096);
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 4, .y = 1, .z = 1});
    Push(list, VC::Op::ClearBuffer, VC::ClearBufferPayload{.address = SOURCE, .size = 16, .value = 7});
    BindCompute2(list, PROGRAM_COPY, SOURCE, TARGET + 0x100, 4096 - 0x100);
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 4, .y = 1, .z = 1});
    rig.Run(1, std::move(list));

    for (u32 i = 0; i < 4; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x3000 + i * 4) == 0xCDCDCDCEu);
        REQUIRE(WordAt(rig.data, 0x3100 + i * 4) == 8);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 7);
    }
    REQUIRE(rig.context.Stats().ops_failed == 0);
}

TEST_CASE("NVN pool mirror: indirect group counts a dispatch wrote are read from the mirror",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan, 1);
    constexpr u64 ARGS = POOL_VA + 0x2000;
    constexpr u64 TARGET = POOL_VA + 0x3000;
    std::vector<u64> list;
    // Groups 0-2 store 1, 2, 3: the indirect dispatch runs 1 x 2 x 3 groups, all with CTAID.X 0.
    Dispatch(list, PROGRAM_ARGS, ARGS, 4096, 3);
    BindCompute(list, PROGRAM_2, TARGET, 4096);
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.indirect = ARGS});
    rig.Run(1, std::move(list));

    REQUIRE(WordAt(rig.data, 0x2000) == 1);
    REQUIRE(WordAt(rig.data, 0x2004) == 2);
    REQUIRE(WordAt(rig.data, 0x2008) == 3);
    REQUIRE(WordAt(rig.data, 0x3000) == 200);
    REQUIRE(WordAt(rig.data, 0x3004) == 0xCDCDCDCDu);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    if (rig.Imported()) {
        // The arguments stayed in the entry (no write-back before the dispatch read them).
        CHECK(stats.shadow_flush_conflict == 0);
        CHECK(stats.vram_bindings >= 3);
    }
}

TEST_CASE("NVN pool mirror: a copy in the same submission reads what the dispatch wrote",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan, 1);
    std::vector<u64> list;
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
    Push(list, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = STORAGE_VA, .dst = POOL_VA + 0x3000, .size = 32});
    // A later dispatch over the same range still finds the mirror current.
    Dispatch(list, PROGRAM_2, STORAGE_VA, 4096, 2);
    rig.Run(1, std::move(list));

    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.data, 0x3000 + i * 4) == 100 + i);
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == (i < 2 ? 200 + i : 100 + i));
    }
    REQUIRE(WordAt(rig.data, 0x3000 + 6 * 4) == 0xCDCDCDCDu);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    if (rig.Imported()) {
        CHECK(stats.shadow_flush_conflict == 1);
        CHECK(stats.shadow_flush_segment == 1);
        // The copy wrote [0x3000, 0x3020) through the pool's buffer, not the mirror: no copy-in.
        CHECK(stats.vram_copy_in_bytes == 4096);
    }
}

TEST_CASE("NVN pool mirror: without CPU write tracking pools stay in guest memory",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    ShadowRig rig(*vulkan, 1);
    rig.memory.tracking = false;
    std::vector<u64> list;
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
    rig.Run(1, std::move(list));
    for (u32 i = 0; i < 6; ++i) {
        REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
    }
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    CHECK(stats.vram_mirrors == 0);
    if (rig.Imported()) {
        CHECK(stats.vram_mirror_refused == 1);
    }
}

// ---- an NVN image view in another format of the texel size -------------------------------------

TEST_CASE("NVN compute: an R32UI image view of an RGB9E5 texture takes the shader's stores",
          "[nvn][compute]") {
    // OoT's in-game compute passes bind an R32UI view of an RGB9E5 texture as an image and store the
    // shared-exponent words they pack themselves (SUST.P, one component). Vulkan stores no
    // E5B9G9R9, so the texture's image needs STORAGE usage for the view's format.
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    u8* texels = memory.Add(TEXTURE_POOL_VA, 1_MiB);
    constexpr u32 INITIAL_WORD = 0x80402010;
    const bool partial = GENERATE(false, true);
    if (partial) {
        std::memcpy(texels + 3 * sizeof(u32), &INITIAL_WORD, sizeof(INITIAL_WORD));
    }
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = partial ? static_cast<u32>(POOL_CPU_CACHED)
                                         : 1u /* CPU_NO_ACCESS */});
    constexpr s32 NVN_FORMAT_RGB9E5F = 0x40;
    constexpr u16 NVN_FORMAT_R32UI = 0x0B;
    constexpr u32 WIDTH = 4;
    VC::TextureDesc texture{};
    texture.target = 1; // 2D
    texture.format = NVN_FORMAT_RGB9E5F;
    texture.width = WIDTH;
    texture.height = 1;
    texture.depth = 1;
    texture.levels = 1;
    texture.serial = 10;
    texture.pool_serial = 2;
    texture.pool_offset = 0;
    context.CreateTexture(texture);
    VC::TextureDesc view = texture;
    view.serial = 11;
    view.view_of = 10;
    view.view.set_format = 1;
    view.view.format = NVN_FORMAT_R32UI;
    context.CreateTexture(view);

    // Group x stores 0x88000000 + x at texel (x, 0) of image slot 0 (handle word c0[0x10]):
    //   R0 = CTAID.X; R1 = 0; R2 = R0 + 0x88000000; SUST.P.2D.IGN [R0], R2, 0x0, 0x4
    constexpr Inst SustP2D{{0x00007b9b, 0x60000402, 0x00100f00, 0x000fe200}};
    const std::array<Inst, 5> program{
        {S2R(0, SR_CTAID_X), AddImm(1, 255, 0), AddImm(2, 0, 0x88000000), SustP2D, Exit}};
    context.CreateProgram(MakeProgram(PROGRAM, program, {.images = 1}));

    constexpr u32 IMAGE_ID = 5;
    std::vector<u64> list;
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = PROGRAM, .stages = 1u << 5});
    const u64 handle = IMAGE_ID;
    Push(list, VC::Op::BindImages, VC::BindHandlesPayload{.stage = 5, .first = 0, .count = 1},
         {reinterpret_cast<const u8*>(&handle), sizeof(handle)});
    const u32 written = partial ? WIDTH - 1 : WIDTH;
    Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = written, .y = 1, .z = 1});
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = POOL_VA + 0x1000;
    copy.texture = {.serial = 10, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, static_cast<s32>(WIDTH), 1, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
    PushEnd(list);
    VC::Submission submission = MakeSubmission(1, std::move(list), {PROGRAM, 10, 11, 1, 2});
    std::vector<u64> ids(IMAGE_ID + 1, 0);
    ids[IMAGE_ID] = 11;
    submission.texture_pool_snapshot =
        VC::IdSnapshot{std::make_shared<const std::vector<u64>>(std::move(ids))};
    context.Submit(std::move(submission));
    REQUIRE(context.Wait(1, 60s));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
    for (u32 x = 0; x < WIDTH; ++x) {
        INFO("texel " << x);
        CHECK(WordAt(data, 0x1000 + x * 4) == (x < written ? 0x88000000u + x : INITIAL_WORD));
    }
}

namespace {

// Sample the red channel at the centre of texel (group x, 0) in a four-wide texture.
std::array<Inst, 12> SampleTextureRedProgram() {
    const std::array<Inst, 12> reader{{
        S2R(0, SR_CTAID_X),
        {{0x00027245, 0x00000000, 0x00201000, 0x000fe400}},
        {{0x02047421, 0x3f000000, 0x00010000, Sched}},
        {{0x04047820, 0x3e800000, 0x00410000, Sched}},
        AddImm(3, 255, 0x3f000000),
        {{0x04047b60, 0x30000803, 0x049e01ff, 0x000f4800}},
        MovC0(6, 0x10),
        MovC0(7, 0x14),
        MulImm(5, 0, 4),
        Add(6, 6, 5),
        Stg(6, 4),
        Exit,
    }};
    return reader;
}

} // namespace

TEST_CASE("NVN compute: an image written as a storage image is sampled in GENERAL, ordered by the "
          "game's barrier",
          "[nvn][compute]") {
    // OoT's compute blur/mip chains: a dispatch stores a colour image (SUST.P), the game's barrier
    // (0x11, ORDER_PRIMITIVES | INVALIDATE_TEXTURE), the next dispatch samples it (TEX.LZ), and the
    // one after writes it again. The image stays in GENERAL (no layout transition per use), and
    // every sample sees the store the barrier ordered before it.
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = 1 /* CPU_NO_ACCESS */});
    constexpr s32 NVN_FORMAT_R32F = 0x0A;
    constexpr u32 WIDTH = 4;
    VC::TextureDesc texture{};
    texture.target = 1; // 2D
    texture.format = NVN_FORMAT_R32F;
    texture.width = WIDTH;
    texture.height = 1;
    texture.depth = 1;
    texture.levels = 1;
    texture.serial = 10;
    texture.pool_serial = 2;
    texture.pool_offset = 0;
    context.CreateTexture(texture);
    VC::SamplerDesc sampler{};
    sampler.min_filter = 0; // nearest
    sampler.mag_filter = 0;
    sampler.wrap_s = 7; // clamp-to-edge
    sampler.wrap_t = 7;
    sampler.wrap_r = 7;
    sampler.serial = 20;
    context.CreateSampler(sampler);

    // Writers: group x stores the float with bits `base + x` at texel (x, 0) of image slot 0
    // (handle c0[0x10]): R0 = CTAID.X; R1 = 0; R2 = R0 + base; SUST.P.2D.IGN [R0], R2, 0x0, 0x4.
    constexpr u32 LOW = 0x3e800000;  // 0.25f
    constexpr u32 HIGH = 0x3f400000; // 0.75f
    constexpr Inst SustP2D{{0x00007b9b, 0x60000402, 0x00100f00, 0x000fe200}};
    constexpr u64 WRITE_LOW = 301;
    constexpr u64 WRITE_HIGH = 302;
    constexpr u64 READ = 303;
    for (const auto& [serial, base] : {std::pair{WRITE_LOW, LOW}, std::pair{WRITE_HIGH, HIGH}}) {
        const std::array<Inst, 5> writer{
            {S2R(0, SR_CTAID_X), AddImm(1, 255, 0), AddImm(2, 0, base), SustP2D, Exit}};
        context.CreateProgram(MakeProgram(serial, writer, {.images = 1}));
    }
    // Reader: group x samples texture slot 0 (handle c0[0x20], after the storage entry at 0x10) at
    // the centre of texel (x, 0) and stores the red channel at word x of storage buffer 0:
    //   I2FP.F32.U32 R2, R0; FADD.FTZ R4, R2, 0.5; FMUL.FTZ R4, R4, 0.25; IADD3 R3, RZ, 0.5;
    //   TEX.SCR.LZ.NODEP RZ, R4, R4, R3, 0x0, 0x8, 2D, 0x1 (OoT's blur pass word);
    //   R6:R7 = c0[0x10] + 4 * R0; STG.E [R6], R4.
    context.CreateProgram(
        MakeProgram(READ, SampleTextureRedProgram(), {.storage = 1, .textures = 1}));

    constexpr u32 IMAGE_ID = 5;
    constexpr u32 TEXTURE_ID = 6;
    constexpr u32 SAMPLER_ID = 1;
    const auto write = [&](std::vector<u64>& list, u64 program) {
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = program, .stages = 1u << 5});
        const u64 handle = IMAGE_ID;
        Push(list, VC::Op::BindImages, VC::BindHandlesPayload{.stage = 5, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&handle), sizeof(handle)});
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = WIDTH, .y = 1, .z = 1});
    };
    const auto read = [&](std::vector<u64>& list, u64 output) {
        BindCompute(list, READ, output, WIDTH * 4);
        const u64 handle = TEXTURE_ID | u64{SAMPLER_ID} << 20;
        Push(list, VC::Op::BindTextures, VC::BindHandlesPayload{.stage = 5, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&handle), sizeof(handle)});
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = WIDTH, .y = 1, .z = 1});
    };
    const auto barrier = [](std::vector<u64>& list) {
        Push(list, VC::Op::Barrier, VC::BarrierPayload{.bits = 0x11});
    };
    const auto submit = [&](u64 seq, std::vector<u64> list) {
        PushEnd(list);
        VC::Submission submission =
            MakeSubmission(seq, std::move(list), {WRITE_LOW, WRITE_HIGH, READ, 10, 20, 1, 2});
        std::vector<u64> textures(TEXTURE_ID + 1, 0);
        textures[IMAGE_ID] = 10;
        textures[TEXTURE_ID] = 10;
        submission.texture_pool_snapshot =
            VC::IdSnapshot{std::make_shared<const std::vector<u64>>(std::move(textures))};
        submission.sampler_pool_snapshot = VC::IdSnapshot{
            std::make_shared<const std::vector<u64>>(std::vector<u64>{0, 20})};
        context.Submit(std::move(submission));
        REQUIRE(context.Wait(seq, 60s));
    };
    const auto row = [&](std::string_view reason) {
        std::vector<VC::SinkStat> rows;
        context.CollectStats(rows);
        for (const VC::SinkStat& stat : rows) {
            if (stat.reason == reason) {
                return stat.count;
            }
        }
        FAIL("no stat row " << reason);
        return u64{0};
    };
    constexpr std::string_view TRANSITIONS = "image layout transitions (one barrier entry each)";
    constexpr std::string_view IN_GENERAL = "sampled bindings left in GENERAL (no layout transition)";

    // The first store creates the image and puts it in GENERAL.
    std::vector<u64> first;
    write(first, WRITE_LOW);
    submit(1, std::move(first));
    const u64 transitions = row(TRANSITIONS);
    CHECK(transitions >= 1);

    // Sample, store again, sample: no layout transition; each sample sees the store before it.
    constexpr u64 OUT_LOW = POOL_VA + 0x1000;
    constexpr u64 OUT_HIGH = POOL_VA + 0x2000;
    std::vector<u64> chain;
    barrier(chain);
    read(chain, OUT_LOW);
    write(chain, WRITE_HIGH);
    barrier(chain);
    read(chain, OUT_HIGH);
    submit(2, std::move(chain));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
    CHECK(row("descriptors_unsupported") == 0);
    CHECK(row(TRANSITIONS) == transitions);
    CHECK(row(IN_GENERAL) == 2);
    for (u32 x = 0; x < WIDTH; ++x) {
        INFO("texel " << x);
        CHECK(WordAt(data, 0x1000 + x * 4) == LOW + x);
        CHECK(WordAt(data, 0x2000 + x * 4) == HIGH + x);
    }
}

TEST_CASE("NVN smart BC: the first sample uses the recompressed image view", "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan || vulkan->device->IsOptimalAstcSupported() ||
        !Vulkan::Nvn::BcGpuEncoder::Supported(*vulkan->device) ||
        !Vulkan::Nvn::BcGpuEncoder::SupportsBc45(*vulkan->device)) {
        SKIP("requires ASTC recompression with smart BC formats");
    }
    FakeMemory memory;
    u8* output = memory.Add(POOL_VA, 64_KiB);
    u8* astc = memory.Add(TEXTURE_POOL_VA, 1_MiB);
    // One void-extent ASTC LDR block, with 0.5 grey and opaque alpha.
    constexpr std::array<u8, 16> block{
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0xff, 0xff,
    };
    std::memcpy(astc, block.data(), block.size());
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool(
        {.serial = 1, .guest_address = POOL_VA, .size = 64_KiB, .flags = POOL_CPU_CACHED});
    context.CreatePool(
        {.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB, .flags = POOL_CPU_CACHED});
    VC::TextureDesc texture{};
    texture.target = 1;
    texture.format = 0x79; // RGBA_ASTC_4x4
    texture.width = 4;
    texture.height = 4;
    texture.depth = 1;
    texture.levels = 1;
    texture.serial = 10;
    texture.pool_serial = 2;
    context.CreateTexture(texture);
    VC::SamplerDesc sampler{};
    sampler.serial = 20;
    sampler.wrap_s = sampler.wrap_t = sampler.wrap_r = 7;
    context.CreateSampler(sampler);

    constexpr u64 SAMPLE_PROGRAM = 303;
    context.CreateProgram(
        MakeProgram(SAMPLE_PROGRAM, SampleTextureRedProgram(), {.storage = 1, .textures = 1}));
    const auto sample = [&](u64 seq, u64 texture_serial, f32 expected) {
        std::vector<u64> list;
        BindCompute(list, SAMPLE_PROGRAM, POOL_VA + seq * 0x1000, 16);
        const u64 handle = 1 | u64{1} << 20;
        Push(list, VC::Op::BindTextures, VC::BindHandlesPayload{.stage = 5, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&handle), sizeof(handle)});
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 4, .y = 1, .z = 1});
        PushEnd(list);
        VC::Submission submission =
            MakeSubmission(seq, std::move(list), {SAMPLE_PROGRAM, texture_serial, 20, 1, 2});
        submission.texture_pool_snapshot = VC::IdSnapshot{
            std::make_shared<const std::vector<u64>>(std::vector<u64>{0, texture_serial})};
        submission.sampler_pool_snapshot =
            VC::IdSnapshot{std::make_shared<const std::vector<u64>>(std::vector<u64>{0, 20})};
        context.Submit(std::move(submission));
        REQUIRE(context.Wait(seq, 60s));
        f32 observed{};
        std::memcpy(&observed, output + seq * 0x1000, sizeof(observed));
        CHECK(observed > expected - 0.03f);
        CHECK(observed < expected + 0.03f);
    };
    sample(1, 10, 0.5f); // BC3 key was planned before first use chose BC4.

    VC::TextureDesc srgb = texture;
    srgb.serial = 11;
    srgb.view_of = texture.serial;
    srgb.view.set_format = true;
    srgb.view.format = 0x87; // RGBA_ASTC_4x4_SRGB
    context.CreateTexture(srgb);
    sample(2, 11, 0.214f); // the sRGB alias needs an upload into BC1 before it can sample.
}

TEST_CASE("MKW BC3: sparse atlas sampling and streaming copies preserve residency",
          "[nvn][mkw][compute][mkw-vram]") {
    namespace Nvn = Vulkan::Nvn;
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (vulkan->device->IsOptimalAstcSupported() ||
        !Nvn::BcGpuEncoder::Supported(*vulkan->device)) {
        SKIP("requires desktop ASTC recompression");
    }
    struct Shape {
        u32 width;
        u32 layers;
        u32 levels;
    };
    const auto shape = GENERATE(Shape{8192, 68, 14}, Shape{16384, 1, 15}, Shape{8192, 1, 1});
    INFO("atlas " << shape.width << "x" << shape.width << "x" << shape.layers << ", levels "
                  << shape.levels);
    constexpr u64 VIRTUAL_VA = 0x7D00'0000'0000'0000ULL;
    constexpr u64 TILE = 64_KiB;
    FakeMemory memory;
    u8* output = memory.Add(POOL_VA, 64_KiB);
    u8* astc = memory.Add(TEXTURE_POOL_VA, 2_MiB);
    const auto fill = [&](u64 offset, u64 size, u16 red) {
        std::array<u8, 16> block{0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                 0,    0,    0,    0,    0,    0,    0xff, 0xff};
        std::memcpy(block.data() + 8, &red, sizeof(red));
        for (u64 i = 0; i < size; i += block.size()) {
            std::memcpy(astc + offset + i, block.data(), block.size());
        }
    };
    fill(0, 2_MiB, 0x8000);
    Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.SetTitleId(Settings::MarioKartWorldProgramId);
    context.CreatePool(
        {.serial = 1, .guest_address = POOL_VA, .size = 64_KiB, .flags = POOL_CPU_CACHED});
    context.CreatePool(
        {.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 2_MiB, .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 3, .guest_address = VIRTUAL_VA, .size = 2_GiB, .flags = 0x200});
    VC::TextureDesc atlas{};
    atlas.target = 4;
    atlas.format = 0x80; // ASTC 8x8
    atlas.flags = 0x20;
    atlas.width = atlas.height = shape.width;
    atlas.depth = shape.layers;
    atlas.levels = shape.levels;
    atlas.serial = 10;
    atlas.pool_serial = 3;
    context.CreateTexture(atlas);
    Nvn::Storage reference;
    reference.desc = {.target = 4,
                      .format = 0x80,
                      .flags = 0x20,
                      .width = shape.width,
                      .height = shape.width,
                      .depth = shape.layers,
                      .levels = shape.levels};
    reference.layout = Nvn::ComputeTextureLayout(reference.desc);
    reference.extent = {shape.width, shape.width, 1};
    reference.type = VK_IMAGE_TYPE_2D;
    reference.samples = VK_SAMPLE_COUNT_1_BIT;
    reference.levels = shape.levels;
    reference.layers = shape.layers;
    const auto layout = Nvn::BuildSparseLayout(reference);
    REQUIRE(layout);
    const u32 sample_x = shape.width / 8;
    const u32 sample_y = shape.width / 2;
    const auto& level = layout->level[0];
    const u32 id = (sample_y / layout->tile_height) * level.tiles_x + sample_x / layout->tile_width;
    const u64 tile_offset = layout->TileOffset(id);
    const u64 last_layer_tile = tile_offset + (shape.layers - 1) * layout->layer_size;
    context.MapVirtual(3, {{tile_offset, 2, 0, TILE}, {last_layer_tile, 2, 0, TILE}});
    const auto view = [&](u64 serial, u32 layer, u32 mip, bool srgb) {
        VC::TextureDesc desc = atlas;
        desc.serial = serial;
        desc.view_of = atlas.serial;
        desc.view.set_layers = true;
        desc.view.min_layer = static_cast<u16>(layer);
        desc.view.nlayers = 1;
        desc.view.set_target = true;
        desc.view.target = 1;
        desc.view.set_levels = true;
        desc.view.base_level = static_cast<u8>(mip);
        desc.view.nlevels = 1;
        desc.view.set_format = true;
        desc.view.format = srgb ? 0x8e : 0x80;
        context.CreateTexture(desc);
    };
    view(11, 0, 0, false);
    view(12, shape.layers - 1, 0, false);
    view(13, 0, 0, true);
    VC::SamplerDesc sampler{};
    sampler.serial = 20;
    sampler.wrap_s = sampler.wrap_t = sampler.wrap_r = 7;
    context.CreateSampler(sampler);
    constexpr u64 SAMPLE_PROGRAM = 304;
    context.CreateProgram(
        MakeProgram(SAMPLE_PROGRAM, SampleTextureRedProgram(), {.storage = 1, .textures = 1}));
    u64 seq = 0;
    const auto run = [&](std::vector<u64> list, u64 texture_serial) {
        BindCompute(list, SAMPLE_PROGRAM, POOL_VA, 16);
        const u64 handle = 1 | u64{1} << 20;
        Push(list, VC::Op::BindTextures, VC::BindHandlesPayload{.stage = 5, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&handle), sizeof(handle)});
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 4, .y = 1, .z = 1});
        PushEnd(list);
        auto submission = MakeSubmission(
            ++seq, std::move(list),
            {SAMPLE_PROGRAM, texture_serial, 10, 20, 1, 2, 3, 4, 30, 40, 41, 50, 51, 52, 53});
        submission.texture_pool_snapshot = VC::IdSnapshot{
            std::make_shared<const std::vector<u64>>(std::vector<u64>{0, texture_serial})};
        submission.sampler_pool_snapshot =
            VC::IdSnapshot{std::make_shared<const std::vector<u64>>(std::vector<u64>{0, 20})};
        context.Submit(std::move(submission));
        REQUIRE(context.Wait(seq, 60s));
        REQUIRE(context.Stats().ops_failed == 0);
        REQUIRE(context.Stats().ops_unsupported == 0);
        std::array<f32, 4> samples{};
        std::memcpy(samples.data(), output, sizeof(samples));
        return samples;
    };
    const auto check = [&](u64 texture, f32 expected) {
        const auto samples = run({}, texture);
        CHECK(samples[0] > expected - 0.035f);
        CHECK(samples[0] < expected + 0.035f);
        CHECK(samples[1] == 0.0f); // an unmapped tile still reads zero
    };
    // Tiny mip copies cannot use vkCmdCopyImage when their BC blocks do not align with
    // the destination. Preserve the surrounding texels and the raw UNORM/sRGB bits.
    fill(3 * TILE, TILE, 0xe000);
    VC::TextureDesc tiny{};
    tiny.target = 1;
    tiny.format = 0x8e;
    tiny.width = tiny.height = 2;
    tiny.depth = tiny.levels = 1;
    tiny.serial = 50;
    tiny.pool_serial = 2;
    tiny.pool_offset = 3 * TILE;
    context.CreateTexture(tiny);
    tiny.format = 0x80;
    tiny.width = tiny.height = 4;
    tiny.serial = 51;
    tiny.pool_offset = 4 * TILE;
    context.CreateTexture(tiny);
    VC::CopyTextureToTexturePayload partial{};
    partial.src = {.serial = 50};
    partial.dst = {.serial = 51};
    partial.src_region = {0, 0, 0, 2, 2, 1};
    partial.dst_region = {1, 1, 0, 2, 2, 1};
    std::vector<u64> partial_list;
    Push(partial_list, VC::Op::CopyTextureToTexture, partial);
    const auto partial_samples = run(std::move(partial_list), 51);
    for (u32 i = 0; i < 4; ++i) {
        const f32 expected = i == 1 || i == 2 ? 0.875f : 0.5f;
        // Re-encoding a mixed BC3 block can move its endpoints by several UNORM steps.
        CHECK(partial_samples[i] > expected - 0.06f);
        CHECK(partial_samples[i] < expected + 0.06f);
    }
    if (layout->HasTail()) {
        const u32 coarse_mip = static_cast<u32>(layout->level.size());
        const u32 coarse_size = (std::max)(shape.width >> coarse_mip, 1u);
        view(52, shape.layers - 1, coarse_mip, false);
        const auto empty_coarse = run({}, 52);
        for (f32 value : empty_coarse) {
            CHECK(value == 0.0f);
        }
        partial.dst = {.serial = 10, .level = coarse_mip, .layer = shape.layers - 1};
        partial.dst_region = {
            static_cast<s32>(coarse_size / 8), static_cast<s32>(coarse_size / 2), 0, 2, 2, 1};
        partial_list.clear();
        Push(partial_list, VC::Op::CopyTextureToTexture, partial);
        const auto coarse_samples = run(std::move(partial_list), 52);
        CHECK(coarse_samples[0] > 0.8f);
        CHECK(coarse_samples[1] == 0.0f);
        CHECK(coarse_samples[2] == 0.0f);
        CHECK(coarse_samples[3] == 0.0f);
        // Re-visiting an unmapped coarse mip must retain the new GPU texels.
        CHECK(run({}, 52) == coarse_samples);
        view(53, shape.layers - 1, shape.levels - 1, false);
        partial.dst.level = shape.levels - 1;
        partial.src_region = partial.dst_region = {0, 0, 0, 1, 1, 1};
        partial_list.clear();
        Push(partial_list, VC::Op::CopyTextureToTexture, partial);
        for (f32 value : run(std::move(partial_list), 53)) {
            CHECK(value > 0.8f);
        }
    }

    check(11, 0.5f);
    check(12, 0.5f);   // full array layer range, including the final layer
    check(13, 0.214f); // first sRGB alias sees the same BC bits
    const u64 tile_binds = shape.layers > 1 ? 4 : 2;
    if (layout->HasTail()) {
        CHECK(context.Stats().virtual_sparse_binds > tile_binds);
    } else {
        CHECK(context.Stats().virtual_sparse_binds == tile_binds);
    }
    fill(TILE, TILE, 0xe000);
    context.MapVirtual(3, {{tile_offset, 2, TILE, TILE}});
    check(11, 0.875f);
    check(12, shape.layers > 1 ? 0.5f : 0.875f);
    fill(TILE, TILE, 0x4000);
    context.NotifyCpuWrite(TEXTURE_POOL_VA + TILE, TILE);
    check(11, 0.25f);

    // Stream an ordinary UNORM ASTC texture into an sRGB sparse atlas without converting bits.
    VC::TextureDesc source{};
    source.target = 1;
    source.format = 0x80;
    source.width = layout->tile_width;
    source.height = layout->tile_height;
    source.depth = source.levels = 1;
    source.serial = 30;
    source.pool_serial = 2;
    source.pool_offset = 2 * TILE;
    fill(2 * TILE, TILE, 0xc000);
    context.CreateTexture(source);
    VC::CopyTextureToTexturePayload copy{};
    copy.src = {.serial = 30};
    copy.dst = {.serial = 13};
    copy.src_region = {
        0, 0, 0, static_cast<s32>(layout->tile_width), static_cast<s32>(layout->tile_height), 1};
    copy.dst_region = {static_cast<s32>((sample_x / layout->tile_width) * layout->tile_width),
                       static_cast<s32>((sample_y / layout->tile_height) * layout->tile_height),
                       0,
                       static_cast<s32>(layout->tile_width),
                       static_cast<s32>(layout->tile_height),
                       1};
    std::vector<u64> list;
    Push(list, VC::Op::CopyTextureToTexture, copy);
    const auto copied = run(std::move(list), 11);
    CHECK(copied[0] > 0.715f);
    CHECK(copied[0] < 0.785f);
    check(13, 0.523f);
    const auto uploads = context.Stats().uploads;
    context.MapVirtual(3, {{tile_offset, 2, TILE, TILE}}); // identical remap keeps GPU writes
    check(11, 0.75f);
    CHECK(context.Stats().uploads == uploads);
    context.MapVirtual(3, {{tile_offset, 0, 0, TILE}});
    check(11, 0.0f);
    check(12, shape.layers > 1 ? 0.5f : 0.0f);

    // Test distinct UNORM/sRGB storages too, rather than just format aliases of one image.
    context.CreatePool(
        {.serial = 4, .guest_address = VIRTUAL_VA + 4_GiB, .size = 2_GiB, .flags = 0x200});
    VC::TextureDesc srgb_atlas = atlas;
    srgb_atlas.serial = 40;
    srgb_atlas.pool_serial = 4;
    srgb_atlas.format = 0x8e;
    context.CreateTexture(srgb_atlas);
    context.MapVirtual(4, {{tile_offset, 2, 0, TILE}});
    VC::TextureDesc srgb_view = srgb_atlas;
    srgb_view.serial = 41;
    srgb_view.view_of = 40;
    srgb_view.view.set_target = true;
    srgb_view.view.target = 1;
    srgb_view.view.set_layers = true;
    srgb_view.view.nlayers = 1;
    context.CreateTexture(srgb_view);
    copy.dst.serial = 40;
    list.clear();
    Push(list, VC::Op::CopyTextureToTexture, copy);
    const auto srgb_copied = run(std::move(list), 41);
    CHECK(srgb_copied[0] > 0.488f);
    CHECK(srgb_copied[0] < 0.558f);
    // Reverse the copy back into the UNORM storage: no sRGB conversion in either direction.
    context.MapVirtual(3, {{tile_offset, 2, TILE, TILE}});
    copy.src = {.serial = 40};
    copy.src_region = copy.dst_region;
    copy.dst.serial = 10;
    list.clear();
    Push(list, VC::Op::CopyTextureToTexture, copy);
    const auto reverse_copied = run(std::move(list), 11);
    CHECK(reverse_copied[0] > 0.715f);
    CHECK(reverse_copied[0] < 0.785f);

    // Guest mip tails bind the host tail and any remaining host sparse levels separately.
    if (!layout->HasTail()) {
        return;
    }
    const u64 tail = layout->tail_offset;
    REQUIRE(layout->TailSize() <= 1_MiB);
    fill(1_MiB, layout->TailSize(), 0x8000);
    context.MapVirtual(3, {{tail, 2, 1_MiB, layout->TailSize()}});
    view(14, 0, shape.levels - 1, false);
    const auto tail_samples = run({}, 14);
    CHECK(tail_samples[0] > 0.465f);
    CHECK(tail_samples[0] < 0.535f);
}

// ---- SAMPLES_PASSED reports inside a render pass -----------------------------------------------

namespace {

constexpr u32 SAMPLES_PASSED = 1; // NVN_COUNTER_TYPE_SAMPLES_PASSED

u64 StatRow(const Vulkan::Nvn::Context& context, std::string_view reason) {
    std::vector<VC::SinkStat> rows;
    context.CollectStats(rows);
    for (const VC::SinkStat& row : rows) {
        if (row.reason == reason) {
            return row.count;
        }
    }
    return 0;
}

} // namespace

TEST_CASE("NVN occlusion: SAMPLES_PASSED reports and resets keep the render pass open",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool host_reset = vulkan->device->IsHostQueryResetSupported();
    for (const bool in_pass : {true, false}) {
        for (const bool import : {true, false}) {
            INFO("in-pass queries " << in_pass << ", host import " << import);
            FakeMemory memory;
            u8* data = memory.Add(POOL_VA, 64_KiB);
            std::memset(data, 0xCD, 64_KiB);
            memory.Add(TEXTURE_POOL_VA, 1_MiB);
            // No early-submitted pieces: they end the open pass after a recording time, which
            // depends on the machine's load (the pass counts below would change).
            Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                         {.allow_host_import = import,
                                          .in_pass_queries = in_pass,
                                          .early_submit = false});
            context.SetGuestMemory(&memory);
            context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                                .flags = POOL_CPU_CACHED});
            context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                                .flags = 1 /* CPU_NO_ACCESS */});
            VC::TextureDesc target{};
            target.target = 1; // 2D
            target.format = 0x25;
            target.width = 16;
            target.height = 16;
            target.depth = 1;
            target.levels = 1;
            target.serial = 10;
            target.pool_serial = 2;
            target.pool_offset = 0;
            context.CreateTexture(target);
            // Draws without fragments (rasterizer discard): they count 0 samples, but every one
            // is a counted draw of the open pass.
            const std::array<Inst, 1> nothing{{Exit}};
            const auto store = StoreProgram(0, 0x1337);
            context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, nothing));
            context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM + 1, store));

            constexpr u64 REPORTS = 0x2000;
            std::vector<u64> list;
            const VC::RenderTargetRef color{.texture = {.serial = 10, .level = 0, .layer = 0}};
            Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
                 {reinterpret_cast<const u8*>(&color), sizeof(color)});
            Push(list, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = 1});
            Push(list, VC::Op::BindProgram,
                 VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1u << 0});
            const auto draw = [&] {
                Push(list, VC::Op::Draw,
                     VC::DrawPayload{.mode = 4, .first = 0, .count = 3, .instance_count = 1});
            };
            const auto report = [&](u64 offset) {
                Push(list, VC::Op::ReportCounter,
                     VC::ReportCounterPayload{.address = POOL_VA + REPORTS + offset,
                                              .counter_type = SAMPLES_PASSED});
            };
            const auto reset = [&] {
                Push(list, VC::Op::ResetCounter,
                     VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
            };
            // OoT issues thousands of resets in a row outside any pass (they once overflowed the
            // in-pass batch and lost the device).
            for (u32 i = 0; i < 2500; ++i) {
                reset();
            }
            // More reports in one pass than a batch holds (a query and a report per draw): it is
            // written early, once.
            constexpr u32 MANY = Vulkan::Nvn::CounterBatch::MAX_EVENTS / 2 + 52;
            static_assert(REPORTS + 0x400 + MANY * 16 <= 64_KiB, "the reports fit in the pool");
            for (u32 i = 0; i < MANY; ++i) {
                draw();
                report(0x400 + i * 16);
            }
            draw();
            report(0);
            draw();
            report(0x100);
            reset();
            draw();
            report(0x200);
            // A draw storing into the counter of the last report: it comes after the report.
            Push(list, VC::Op::BindProgram,
                 VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM + 1, .stages = 1u << 0});
            const VC::BufferRange range{POOL_VA + REPORTS + 0x200, 256};
            Push(list, VC::Op::BindStorageBuffers,
                 VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
                 {reinterpret_cast<const u8*>(&range), sizeof(range)});
            draw();
            PushEnd(list);
            context.Submit(MakeSubmission(
                1, std::move(list), {VERTEX_PROGRAM, VERTEX_PROGRAM + 1, 10, 1, 2}));
            REQUIRE(context.Wait(1, 60s));

            const Vulkan::Nvn::ContextStats stats = context.Stats();
            REQUIRE(stats.ops_failed == 0);
            REQUIRE(stats.ops_unsupported == 0);
            CHECK(stats.counter_reports_samples == MANY + 3);
            CHECK(WordAt(data, REPORTS + 0x400) == 0);
            CHECK(WordAt(data, REPORTS + 0x400 + (MANY - 1) * 16 + 4) == 0);
            // Every report's 64-bit counter was written (0 samples) ...
            CHECK(WordAt(data, REPORTS) == 0);
            CHECK(WordAt(data, REPORTS + 4) == 0);
            CHECK(WordAt(data, REPORTS + 0x100) == 0);
            CHECK(WordAt(data, REPORTS + 0x104) == 0);
            CHECK(WordAt(data, REPORTS + 0x204) == 0);
            // ... and the store of the later draw is not overwritten by its report.
            CHECK(WordAt(data, REPORTS + 0x200) == 0x1337);
            CHECK(WordAt(data, REPORTS + 0x208) != 0xCDCDCDCDu); // the time field
            const u64 report_breaks = StatRow(context, "render_pass_break: ReportCounter");
            const u64 reset_breaks = StatRow(context, "render_pass_break: ResetCounter");
            const u64 read_breaks =
                StatRow(context, "render_pass_break: draw reads a counter report of its pass");
            if (in_pass && host_reset) {
                // One pass for the counted draws, written early once when its batch was full; the
                // store draw needs the last reports written first.
                CHECK(StatRow(context, "render_pass_begins") == 3);
                CHECK(report_breaks == 1);
                CHECK(reset_breaks == 0);
                CHECK(read_breaks == 1);
                // The full batch, the report that found it full (no pass open then: written right
                // away), the reports at the store draw, its query at the command buffer end.
                CHECK(stats.counter_occlusion_folds == 4);
            } else {
                CHECK(StatRow(context, "render_pass_begins") == MANY + 4);
                CHECK(report_breaks == MANY + 3);
                CHECK(reset_breaks == 0); // no pass is open at any reset
                CHECK(read_breaks == 0);
                CHECK(stats.counter_occlusion_folds == MANY + 4);
            }
        }
    }
}

namespace {

/// Every vertex copies word 0 of storage buffer 0 to word 0 of storage buffer 1 (LDG.E.32 /
/// STG.E.32 through the vertex driver bank's storage entries at c0[0x110] and c0[0x120]).
std::array<Inst, 7> CopyWordProgram() {
    constexpr Inst load{{0x00007381 | (6u << 16) | (2u << 24), 0, 0x001e8900, Sched}};
    return {{MovC0(2, 0x110), MovC0(3, 0x114), MovC0(4, 0x120), MovC0(5, 0x124), load, Stg(4, 6),
             Exit}};
}

} // namespace

TEST_CASE("NVN occlusion: a draw reads a report of its pass only after a game barrier ordered it",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsHostQueryResetSupported()) {
        SKIP("no host query reset: queries span passes and every report ends the pass");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.in_pass_queries = true, .early_submit = false});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = 1 /* CPU_NO_ACCESS */});
    VC::TextureDesc target{};
    target.target = 1; // 2D
    target.format = 0x25;
    target.width = 16;
    target.height = 16;
    target.depth = 1;
    target.levels = 1;
    target.serial = 10;
    target.pool_serial = 2;
    target.pool_offset = 0;
    context.CreateTexture(target);
    const std::array<Inst, 1> nothing{{Exit}};
    const auto copy = CopyWordProgram();
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, nothing));
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM + 1, copy));

    constexpr u64 ORDERED_REPORT = 0x200;
    constexpr u64 UNORDERED_REPORT = 0x300;
    // On another page than the reports, one page for both (a mirrored pool copies a page at its
    // first use, which ends the pass), apart (separate storage shadows, no alias).
    constexpr u64 ORDERED_COPY = 0x3000;
    constexpr u64 UNORDERED_COPY = 0x3800;
    std::vector<u64> list;
    const VC::RenderTargetRef color{.texture = {.serial = 10, .level = 0, .layer = 0}};
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
         {reinterpret_cast<const u8*>(&color), sizeof(color)});
    Push(list, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = 1});
    const auto draw = [&] {
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 4, .first = 0, .count = 3, .instance_count = 1});
    };
    // A counted draw, its report, then a draw copying the report's counter word elsewhere.
    const auto report_then_copy = [&](u64 report, u64 copy_to, bool barrier) {
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1u << 0});
        draw();
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = POOL_VA + report, .counter_type = SAMPLES_PASSED});
        if (barrier) {
            Push(list, VC::Op::Barrier, VC::BarrierPayload{.bits = 0x2}); // ORDER_FRAGMENTS
        }
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM + 1, .stages = 1u << 0});
        const std::array<VC::BufferRange, 2> ranges{
            {{POOL_VA + report, 16}, {POOL_VA + copy_to, 16}}};
        Push(list, VC::Op::BindStorageBuffers,
             VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 2},
             {reinterpret_cast<const u8*>(ranges.data()), sizeof(ranges)});
        draw();
    };
    // The barrier orders the report: the copy draw ends the pass and reads the written count.
    report_then_copy(ORDERED_REPORT, ORDERED_COPY, true);
    // No barrier in between: the read is not ordered after the report (as on the console), the
    // pass stays open and the draw reads the bytes from before the report.
    report_then_copy(UNORDERED_REPORT, UNORDERED_COPY, false);
    PushEnd(list);
    context.Submit(
        MakeSubmission(1, std::move(list), {VERTEX_PROGRAM, VERTEX_PROGRAM + 1, 10, 1, 2}));
    REQUIRE(context.Wait(1, 60s));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    // Which causes ended passes (shown with a failure).
    std::vector<VC::SinkStat> rows;
    context.CollectStats(rows);
    std::string causes;
    for (const VC::SinkStat& row : rows) {
        if (row.reason.starts_with("render_pass_break") || row.reason.starts_with("pools_")) {
            causes += std::string(row.reason) + " = " + std::to_string(row.count) + "; ";
        }
    }
    INFO(causes);
    CHECK(StatRow(context, "render_pass_break: draw reads a counter report of its pass") == 1);
    CHECK(StatRow(context, "render_pass_break: Barrier") == 0);
    CHECK(StatRow(context, "render_pass_begins") == 2);
    // Both reports were written (0 samples: rasterizer discard).
    CHECK(WordAt(data, ORDERED_REPORT) == 0);
    CHECK(WordAt(data, UNORDERED_REPORT) == 0);
    CHECK(WordAt(data, ORDERED_COPY) == 0);
    CHECK(WordAt(data, UNORDERED_COPY) == 0xCDCDCDCDu);
}

// NVN orders a dispatch after an earlier one's shader writes only behind nvnCommandBufferBarrier
// (the game separates dependent dispatches itself: OoT's compute queue has 51 barriers for 131
// dispatches). With ContextOptions::dispatch_order_by_barriers the replay leaves out the full
// barrier the game did not ask for; its own commands (copies, shadows) still order after the
// dispatches.
TEST_CASE("NVN compute: a dispatch orders after an earlier one only behind a game barrier",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 SOURCE = STORAGE_VA;               // pool + 0x1000: 100 + i
    constexpr u64 DESTINATION = STORAGE_VA + 0x4000; // pool + 0x5000: 101 + i
    constexpr u64 COPY_TO = POOL_VA + 0x9000;
    u64 barriers[2]{};
    for (const bool ordered_by_barriers : {false, true}) {
        INFO("dispatch ordering by barriers " << ordered_by_barriers);
        // No early pieces: a piece begins a command buffer, whose first op has a barrier anyway
        // (the pipeline compiles of the first dispatches take longer than the piece interval).
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{
                                   .early_submit = false,
                                   .dispatch_order_by_barriers = ordered_by_barriers,
                               });
        std::vector<u64> list;
        Dispatch(list, PROGRAM, SOURCE, 4096, 6);
        // ORDER_PRIMITIVES | INVALIDATE_SHADER: the copy dispatch reads what the fill wrote.
        Push(list, VC::Op::Barrier, VC::BarrierPayload{.bits = 0x21});
        BindCompute2(list, PROGRAM_COPY, SOURCE, DESTINATION, 4096);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        // No barrier: the same dispatch again writes the same values (a benign race on the
        // console as well) over the ranges the first one already uses.
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        // A copy after the dispatches with no barrier still reads what they wrote.
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = DESTINATION, .dst = COPY_TO, .size = 24});
        rig.Run(1, std::move(list));

        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
            REQUIRE(WordAt(rig.data, 0x5000 + i * 4) == 101 + i);
            REQUIRE(WordAt(rig.data, 0x9000 + i * 4) == 101 + i);
        }
        REQUIRE(WordAt(rig.data, 0x9000 + 6 * 4) == 0xCDCDCDCDu);
        REQUIRE(rig.context.Stats().ops_failed == 0);
        // Only the second copy dispatch followed a dispatch with nothing in between.
        CHECK(StatRow(rig.context, "dispatches after a dispatch with no barrier between (no game "
                                   "barrier, no replay command; EDEN_NX2_TRY_DISPATCH_ORDER)") == 1);
        barriers[ordered_by_barriers ? 1 : 0] =
            StatRow(rig.context, "memory barriers recorded (ALL_COMMANDS, between replayed ops)");
    }
    // The same commands but that one barrier.
    CHECK(barriers[0] == barriers[1] + 1);
}

// The shadows of one plan cover disjoint pool ranges in their own slots: their copies in (before
// the dispatch) and back (at the end of the command buffer) go behind one barrier per loop, so a
// dispatch writing two shadowed ranges records as many barriers as one writing a single range.
TEST_CASE("NVN storage shadow: the shadows of one dispatch are copied in and back behind one "
          "barrier each way",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 FIRST = STORAGE_VA;           // pool + 0x1000
    constexpr u64 SECOND = STORAGE_VA + 0x4000; // pool + 0x5000
    u64 barriers[2]{};
    bool imported = true;
    for (const bool two_ranges : {false, true}) {
        INFO("two written ranges " << two_ranges);
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{.early_submit = false});
        std::vector<u64> list;
        if (two_ranges) {
            BindCompute2(list, PROGRAM_TWO, FIRST, SECOND, 4096);
            Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        } else {
            Dispatch(list, PROGRAM, FIRST, 4096, 6);
        }
        rig.Run(1, std::move(list));

        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
            REQUIRE(WordAt(rig.data, 0x5000 + i * 4) == (two_ranges ? 200 + i : 0xCDCDCDCDu));
        }
        REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
        REQUIRE(WordAt(rig.data, 0x5000 + 6 * 4) == 0xCDCDCDCDu);
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.ops_failed == 0);
        imported = imported && rig.Imported();
        if (rig.Imported()) {
            CHECK(stats.shadow_creates == (two_ranges ? 2u : 1u));
            CHECK(stats.shadow_flush_segment == (two_ranges ? 2u : 1u));
        }
        barriers[two_ranges ? 1 : 0] =
            StatRow(rig.context, "memory barriers recorded (ALL_COMMANDS, between replayed ops)");
    }
    // Without imported pools there are no shadows: the pools' device-local mirrors upload each
    // range behind a barrier of its own (Recorder::UploadRange).
    if (imported) {
        CHECK(barriers[1] == barriers[0]);
    }
}

// Force maximum clocks: the renderer's TurboMode only keeps the GPU busy while it is told about
// queue submissions, and the replay submits to the queue itself.
TEST_CASE("NVN context: the on_queue_submit hook runs after the replay's queue submissions",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    std::atomic<int> submissions{0};
    ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{
                               .early_submit = false,
                               .on_queue_submit = [&submissions] { ++submissions; }});
    std::vector<u64> list;
    Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
    rig.Run(1, std::move(list));
    for (int wait = 0; wait < 500 && submissions.load() == 0; ++wait) {
        std::this_thread::sleep_for(10ms);
    }
    CHECK(submissions.load() >= 1);
    REQUIRE(rig.context.Stats().ops_failed == 0);
}

// A shadow that a dispatch's range partly overlaps goes back to the pool first; the new shadow over
// that range copies in the bytes the write-back wrote, so the write-back and the copy in are not
// one batch: a barrier stays between them (one more than when the second range is elsewhere).
TEST_CASE("NVN storage shadow: a shadow created after a conflict write-back copies in its bytes",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    u64 barriers[2]{};
    bool imported = true;
    for (const bool overlapping : {false, true}) {
        INFO("second range overlaps the first " << overlapping);
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{.early_submit = false});
        std::vector<u64> list;
        // Words 0-5 of [0x1000, 0x2000), then word 4 through [0x1010, 0x2010) (overlapping) or
        // word 0 of [0x5000, 0x6000).
        Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
        Dispatch(list, PROGRAM_2, overlapping ? STORAGE_VA + 16 : STORAGE_VA + 0x4000, 4096, 1);
        rig.Run(1, std::move(list));

        // Word 5 lies in the second shadow and its dispatch leaves it: it reaches the pool only
        // through that shadow's copy in (after the first one's write-back) and write-back.
        const u32 expected[] = {100, 101, 102, 103, overlapping ? 200u : 104u, 105};
        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == expected[i]);
        }
        REQUIRE(WordAt(rig.data, 0x1000 + 6 * 4) == 0xCDCDCDCDu);
        REQUIRE(WordAt(rig.data, 0x5000) == (overlapping ? 0xCDCDCDCDu : 200u));
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.ops_failed == 0);
        imported = imported && rig.Imported();
        if (rig.Imported()) {
            CHECK(stats.shadow_creates == 2);
            CHECK(stats.shadow_flush_conflict == (overlapping ? 1u : 0u));
            CHECK(stats.shadow_flush_segment == (overlapping ? 1u : 2u));
        }
        barriers[overlapping ? 1 : 0] =
            StatRow(rig.context, "memory barriers recorded (ALL_COMMANDS, between replayed ops)");
    }
    // Overlapping: the write-back's barrier and the copy in's. Elsewhere: the copy in's only (the
    // two write-backs at the end share one). Without imported pools there are no shadows.
    if (imported) {
        CHECK(barriers[1] == barriers[0] + 1);
    }
}

// EDEN_NX2_TRY_BARRIER_MERGE (nvn_barrier_track.h): the replay's own buffer commands skip the
// barrier in front of them when no byte they touch conflicts with the buffer commands since the
// last barrier. The tracker's rules are pinned in barrier_track.cpp; these run the recorder: every
// scenario must leave the pool bytes it leaves without the switch, and the scenarios with
// independent write-backs record fewer barriers.
namespace {

constexpr const char* BARRIER_MERGE_LEFT_OUT =
    "barrier merge: buffer commands whose barrier was left out (no overlap with the buffer commands "
    "since the last; EDEN_NX2_TRY_BARRIER_MERGE)";
constexpr const char* BARRIER_MERGE_HAZARD =
    "barrier merge: buffer commands that overlapped an earlier one (barrier recorded)";
constexpr const char* MEMORY_BARRIERS_ROW =
    "memory barriers recorded (ALL_COMMANDS, between replayed ops)";

struct MergeRun {
    std::vector<u8> pool;
    u64 barriers{};
    u64 left_out{};
    u64 hazards{};
    u64 flush_conflicts{};
    bool imported{};
};

/// One submission of `build` on a fresh rig with the given pool mirror mode and merge mode.
template <typename Build>
MergeRun RunWithMerge(HeadlessVulkan& vulkan, u32 pool_mirror, u32 merge, Build&& build) {
    ShadowRig rig(vulkan, Vulkan::Nvn::ContextOptions{.early_submit = false,
                                                      .pool_mirror = pool_mirror,
                                                      .barrier_merge = merge});
    std::vector<u64> list;
    build(rig, list);
    rig.Run(1, std::move(list));
    MergeRun out;
    out.pool.assign(rig.data, rig.data + 64_KiB);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    out.imported = rig.Imported();
    out.barriers = StatRow(rig.context, MEMORY_BARRIERS_ROW);
    out.left_out = StatRow(rig.context, BARRIER_MERGE_LEFT_OUT);
    out.hazards = StatRow(rig.context, BARRIER_MERGE_HAZARD);
    out.flush_conflicts = stats.shadow_flush_conflict;
    return out;
}

} // namespace

TEST_CASE("NVN barrier merge: write-backs of disjoint entries share one barrier", "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    // Two written ranges, then a dispatch whose range overlaps both partly: both entries go back
    // to the pool in one loop (two merges with a barrier between them without the switch).
    const auto scenario = [](ShadowRig&, std::vector<u64>& list) {
        Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);                // [0x1000, 0x2000): 100 + i
        Dispatch(list, PROGRAM_2, STORAGE_VA + 0x4000, 4096, 6);     // [0x5000, 0x6000): 200 + i
        Dispatch(list, PROGRAM, STORAGE_VA + 0x800, 0x4000, 6);      // [0x1800, 0x5800): 100 + i
    };
    for (const u32 pool_mirror : {0u, 1u}) {
        INFO("pool mirror mode " << pool_mirror);
        const MergeRun off = RunWithMerge(*vulkan, pool_mirror, 0, scenario);
        const MergeRun on = RunWithMerge(*vulkan, pool_mirror, 1, scenario);
        for (const MergeRun* run : {&off, &on}) {
            for (u32 i = 0; i < 6; ++i) {
                INFO("word " << i << (run == &on ? " (merge on)" : " (merge off)"));
                REQUIRE(WordAt(run->pool.data(), 0x1000 + i * 4) == 100 + i);
                REQUIRE(WordAt(run->pool.data(), 0x1800 + i * 4) == 100 + i);
                REQUIRE(WordAt(run->pool.data(), 0x5000 + i * 4) == 200 + i);
            }
            REQUIRE(WordAt(run->pool.data(), 0x1000 + 6 * 4) == 0xCDCDCDCDu);
            REQUIRE(WordAt(run->pool.data(), 0x1800 + 6 * 4) == 0xCDCDCDCDu);
            REQUIRE(WordAt(run->pool.data(), 0x5000 + 6 * 4) == 0xCDCDCDCDu);
        }
        CHECK(std::memcmp(off.pool.data(), on.pool.data(), off.pool.size()) == 0);
        CHECK(off.left_out == 0);
        if (!on.imported) {
            continue; // no shadows or mirrors without imported pools
        }
        CHECK(off.flush_conflicts == 2);
        CHECK(on.flush_conflicts == 2);
        if (pool_mirror == 1) {
            // The second merge needs no barrier: its mirror, snapshot and pool bytes are not the
            // first one's.
            CHECK(on.left_out >= 1);
            CHECK(on.barriers < off.barriers);
        } else {
            // Plain shadows already share one barrier per loop (ShadowCopyBatch).
            CHECK(on.barriers <= off.barriers);
        }
    }
}

TEST_CASE("NVN barrier merge: overlapping commands keep their barrier and every result is the same",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 SOURCE = POOL_VA + 0x1000;
    constexpr u64 TARGET = POOL_VA + 0x3000;
    struct Scenario {
        const char* name;
        std::function<void(ShadowRig&, std::vector<u64>&)> build;
    };
    const std::array<Scenario, 4> scenarios{{
        // A written range whose pages are copied in on first use: the snapshot reads what the
        // copy in wrote (read after write on the mirror).
        {"copy in, then snapshot of the same pages",
         [](ShadowRig&, std::vector<u64>& list) { Dispatch(list, PROGRAM, SOURCE, 4096, 6); }},
        // A range overlapping the first one by 16 bytes: its write-back, then the copy in and
        // snapshot of the second range over bytes the write-back wrote.
        {"overlapping ranges",
         [](ShadowRig&, std::vector<u64>& list) {
             Dispatch(list, PROGRAM, SOURCE, 4096, 6);
             Dispatch(list, PROGRAM_2, SOURCE + 16, 4096, 3);
         }},
        // The same range written, read by a copy dispatch, written again.
        {"write, copy, write again",
         [](ShadowRig&, std::vector<u64>& list) {
             Dispatch(list, PROGRAM, SOURCE, 4096, 6);
             BindCompute2(list, PROGRAM_COPY, SOURCE, TARGET, 4096);
             Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
             Dispatch(list, PROGRAM_2, SOURCE, 4096, 2);
         }},
        // A buffer clear through the pool's own buffer between two reads of the mirror.
        {"clear through the pool's buffer between dispatches",
         [](ShadowRig&, std::vector<u64>& list) {
             BindCompute2(list, PROGRAM_COPY, SOURCE, TARGET, 4096);
             Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 4, .y = 1, .z = 1});
             Push(list, VC::Op::ClearBuffer,
                  VC::ClearBufferPayload{.address = SOURCE, .size = 16, .value = 7});
             BindCompute2(list, PROGRAM_COPY, SOURCE, TARGET + 0x100, 4096 - 0x100);
             Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 4, .y = 1, .z = 1});
         }},
    }};
    for (const Scenario& scenario : scenarios) {
        for (const u32 pool_mirror : {0u, 1u}) {
            INFO(scenario.name << ", pool mirror mode " << pool_mirror);
            const MergeRun off = RunWithMerge(*vulkan, pool_mirror, 0, scenario.build);
            const MergeRun on = RunWithMerge(*vulkan, pool_mirror, 1, scenario.build);
            CHECK(std::memcmp(off.pool.data(), on.pool.data(), off.pool.size()) == 0);
            // Never more barriers than the commands of before: the switch only leaves some out.
            CHECK(on.barriers <= off.barriers);
            if (std::string_view{scenario.name}.starts_with("copy in") && on.imported &&
                pool_mirror == 1) {
                CHECK(on.hazards >= 1);
            }
        }
    }
}

TEST_CASE("NVN barrier merge: an inline update flush that copies nothing records no barrier",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 TARGET = POOL_VA + 0x3000;
    constexpr const char* NO_COPY_ROW =
        "barrier merge: inline update flushes that recorded no copy (no barrier)";
    const std::array<u8, 16> first{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0};
    const std::array<u8, 16> second{5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0};
    for (const bool resolved : {false, true}) {
        INFO("update target in a pool " << resolved);
        // A dispatch, then one flush of inline updates at the end of the command buffer: updates
        // of address 0 name no buffer (nothing to copy); two updates of one pool range copy the
        // second only (the first is rewritten whole by it).
        const auto scenario = [&](ShadowRig&, std::vector<u64>& list) {
            Dispatch(list, PROGRAM, STORAGE_VA, 4096, 6);
            const u64 address = resolved ? TARGET : 0;
            Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{address, 16},
                 first);
            Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{address, 16},
                 second);
        };
        u64 barriers[2]{};
        u64 no_copy[2]{};
        for (const u32 merge : {0u, 3u}) {
            INFO("merge mode " << merge);
            ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{.early_submit = false,
                                                               .pool_mirror = 0u,
                                                               .barrier_merge = merge});
            std::vector<u64> list;
            scenario(rig, list);
            rig.Run(1, std::move(list));
            REQUIRE(rig.context.Stats().ops_failed == 0);
            for (u32 i = 0; i < 6; ++i) {
                REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
            }
            if (resolved) {
                REQUIRE(WordAt(rig.data, 0x3000) == 5);
                REQUIRE(WordAt(rig.data, 0x3004) == 6);
                REQUIRE(WordAt(rig.data, 0x3008) == 7);
                REQUIRE(WordAt(rig.data, 0x300C) == 8);
            }
            barriers[merge != 0] = StatRow(rig.context, MEMORY_BARRIERS_ROW);
            no_copy[merge != 0] = StatRow(rig.context, NO_COPY_ROW);
        }
        CHECK(no_copy[0] == 0);
        if (resolved) {
            // A copy follows: the barrier in front of it stays.
            CHECK(no_copy[1] == 0);
            CHECK(barriers[1] <= barriers[0]);
        } else {
            CHECK(no_copy[1] == 1);
            CHECK(barriers[1] < barriers[0]);
        }
    }
}

// A game barrier that waits for nothing but pixel work (ORDER_FRAGMENTS | INVALIDATE_TEXTURE, OoT's
// most frequent one) leaves later dispatches unordered on the console (nvn_barrier.h). With
// dispatch ordering by barriers it records nothing; ORDER_INDIRECT_DATA still orders a dispatch
// that reads what the earlier one wrote.
TEST_CASE("NVN compute: a pixel-only game barrier does not order dispatches",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 SOURCE = STORAGE_VA;               // pool + 0x1000: 100 + i
    constexpr u64 DESTINATION = STORAGE_VA + 0x4000; // pool + 0x5000: 101 + i
    constexpr const char* WITHOUT_WAIT =
        "game barriers outside a pass without a wait for earlier work (no "
        "ORDER_PRIMITIVES/ORDER_INDIRECT_DATA; EDEN_NX2_TRY_DISPATCH_ORDER)";
    u64 barriers[2]{};
    for (const bool ordered_by_barriers : {false, true}) {
        INFO("dispatch ordering by barriers " << ordered_by_barriers);
        ShadowRig rig(*vulkan, Vulkan::Nvn::ContextOptions{
                                   .early_submit = false,
                                   .dispatch_order_by_barriers = ordered_by_barriers,
                               });
        std::vector<u64> list;
        Dispatch(list, PROGRAM, SOURCE, 4096, 6);
        // The same fill again behind a pixel-only barrier (a benign race on the console).
        Push(list, VC::Op::Barrier, VC::BarrierPayload{.bits = 0x12});
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        // ORDER_INDIRECT_DATA | INVALIDATE_SHADER: the copy dispatch reads what the fills wrote.
        Push(list, VC::Op::Barrier, VC::BarrierPayload{.bits = 0x28});
        BindCompute2(list, PROGRAM_COPY, SOURCE, DESTINATION, 4096);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 6, .y = 1, .z = 1});
        rig.Run(1, std::move(list));

        for (u32 i = 0; i < 6; ++i) {
            INFO("word " << i);
            REQUIRE(WordAt(rig.data, 0x1000 + i * 4) == 100 + i);
            REQUIRE(WordAt(rig.data, 0x5000 + i * 4) == 101 + i);
        }
        REQUIRE(rig.context.Stats().ops_failed == 0);
        // Off: every game barrier orders (the commands of before). On: the pixel-only one
        // records nothing and the second fill counts as unordered.
        CHECK(StatRow(rig.context, WITHOUT_WAIT) == (ordered_by_barriers ? 1u : 0u));
        CHECK(StatRow(rig.context, "dispatches after a dispatch with no barrier between (no game "
                                   "barrier, no replay command; EDEN_NX2_TRY_DISPATCH_ORDER)") ==
              (ordered_by_barriers ? 1u : 0u));
        barriers[ordered_by_barriers ? 1 : 0] =
            StatRow(rig.context, "memory barriers recorded (ALL_COMMANDS, between replayed ops)");
    }
    CHECK(barriers[0] == barriers[1] + 1);
}

TEST_CASE("NVN render conditions: predicates computed ahead keep the pass open and stay exact",
          "[nvn][compute]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsExtConditionalRendering()) {
        SKIP("VK_EXT_conditional_rendering is not available");
    }
    for (const bool import : {true, false}) {
        INFO("host import " << import);
        FakeMemory memory;
        u8* data = memory.Add(POOL_VA, 64_KiB);
        std::memset(data, 0xCD, 64_KiB);
        memory.Add(TEXTURE_POOL_VA, 1_MiB);
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                     {.allow_host_import = import, .early_submit = false});
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});
        context.CreatePool({.serial = 2, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                            .flags = 1 /* CPU_NO_ACCESS */});
        VC::TextureDesc target{};
        target.target = 1; // 2D
        target.format = 0x25;
        target.width = 16;
        target.height = 16;
        target.depth = 1;
        target.levels = 1;
        target.serial = 10;
        target.pool_serial = 2;
        target.pool_offset = 0;
        context.CreateTexture(target);
        const auto store = StoreProgram(0, 0x1337);
        context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, store));

        // Report pairs the CPU wrote (counter at +0 and at +16), as OoT's are long before the
        // conditions that read them.
        constexpr u64 PAIRS = 0x2000;
        constexpr u64 STORES = 0x4000;
        constexpr u32 IF_EQUAL = 0;
        constexpr u32 IF_NOT_EQUAL = 1;
        const auto pair = [&](u32 index, u64 a, u64 b) {
            std::memcpy(data + PAIRS + index * 0x20, &a, sizeof(a));
            std::memcpy(data + PAIRS + index * 0x20 + 16, &b, sizeof(b));
        };
        pair(0, 1, 0);      // not equal
        pair(1, 3, 3);      // equal
        pair(2, 5, 5);      // equal
        pair(3, 7, 9);      // not equal
        pair(4, 0, 0x1337); // not equal until a draw stores 0x1337 into the first counter

        std::vector<u64> list;
        const VC::RenderTargetRef color{.texture = {.serial = 10, .level = 0, .layer = 0}};
        Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
             {reinterpret_cast<const u8*>(&color), sizeof(color)});
        Push(list, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = 1});
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1u << 0});
        const auto draw = [&](u64 offset) {
            const VC::BufferRange range{POOL_VA + offset, 16};
            Push(list, VC::Op::BindStorageBuffers,
                 VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
                 {reinterpret_cast<const u8*>(&range), sizeof(range)});
            Push(list, VC::Op::Draw,
                 VC::DrawPayload{.mode = 4, .first = 0, .count = 3, .instance_count = 1});
        };
        const auto gated_draw = [&](u32 mode, u32 index, u64 offset) {
            Push(list, VC::Op::SetRenderCondition,
                 VC::SetRenderConditionPayload{
                     .kind = 0, .mode = mode, .address = POOL_VA + PAIRS + index * 0x20});
            draw(offset);
            Push(list, VC::Op::SetRenderCondition, VC::SetRenderConditionPayload{.kind = 1});
        };
        draw(STORES);                                 // opens the pass
        gated_draw(IF_NOT_EQUAL, 0, STORES + 0x100); // draws
        gated_draw(IF_NOT_EQUAL, 1, STORES + 0x200); // skipped
        gated_draw(IF_EQUAL, 2, STORES + 0x300);     // draws
        gated_draw(IF_EQUAL, 3, STORES + 0x400);     // skipped
        // A draw of the pass writes the first counter of pair 4 after its predicate was computed
        // ahead: the condition must see the store (computed again, at its own point).
        draw(PAIRS + 4 * 0x20);
        gated_draw(IF_EQUAL, 4, STORES + 0x500); // draws
        PushEnd(list);
        context.Submit(MakeSubmission(1, std::move(list), {VERTEX_PROGRAM, 10, 1, 2}));
        REQUIRE(context.Wait(1, 60s));

        const Vulkan::Nvn::ContextStats stats = context.Stats();
        REQUIRE(stats.ops_failed == 0);
        REQUIRE(stats.ops_unsupported == 0);
        CHECK(WordAt(data, STORES) == 0x1337);
        CHECK(WordAt(data, STORES + 0x100) == 0x1337);
        CHECK(WordAt(data, STORES + 0x200) == 0xCDCDCDCDu);
        CHECK(WordAt(data, STORES + 0x300) == 0x1337);
        CHECK(WordAt(data, STORES + 0x400) == 0xCDCDCDCDu);
        CHECK(WordAt(data, PAIRS + 4 * 0x20) == 0x1337);
        CHECK(WordAt(data, STORES + 0x500) == 0x1337);
        CHECK(stats.render_conditions_cpu == 0);
        const u64 condition_breaks = StatRow(context, "render_pass_break: render condition");
        if (stats.pools_imported != 0) {
            // One batch at the first condition computes all five predicates; the store drops the
            // last one, which ends the pass once more.
            CHECK(condition_breaks == 2);
            CHECK(stats.render_conditions_gpu == 6);
            CHECK(stats.render_conditions_prefetched == 3);
        } else {
            // Mirrored pools: every condition at its own point.
            CHECK(condition_breaks == 5);
            CHECK(stats.render_conditions_gpu == 5);
            CHECK(stats.render_conditions_prefetched == 0);
        }
    }
}

TEST_CASE("NVN draw: without render targets a draw rasterizes over NVN's no-target surface clip",
          "[nvn][compute]") {
    // OoT draws with no colour and no depth target bound (SetRenderTargets(0, NULL, ..., NULL)).
    // The SDK then programs a 32768 x 32768 surface clip: the draw rasterizes (its samples are
    // counted) wherever its viewport puts it, far beyond any earlier target's size.
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    // One point at NDC (1/64, 1/64): the centre of pixel (32, 32) of a 64 x 64 viewport.
    //   AST.128 a[0x70], R0 (position x, y, z, w)
    constexpr Inst AstPosition{{0xff007322, 0x00007000, 0x00000cff, 0x000fe200}};
    constexpr u32 ONE_64TH = 0x3c800000;
    constexpr u32 ONE = 0x3f800000;
    const std::array<Inst, 6> point{{AddImm(0, 255, ONE_64TH), AddImm(1, 255, ONE_64TH),
                                     AddImm(2, 255, 0), AddImm(3, 255, ONE), AstPosition, Exit}};
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, point));

    // The viewport ends 64 pixels short of the clip (or of the host's framebuffer limit).
    const u32 limit = (std::min)({32768u, vulkan->device->GetMaxFramebufferWidth(),
                                  vulkan->device->GetMaxFramebufferHeight()});
    const f32 corner = static_cast<f32>(limit - 128);
    constexpr u64 REPORTS = 0x2000;
    std::vector<u64> list;
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{0, 0}, {});
    const VC::ViewportRect viewport{corner, corner, 64.0f, 64.0f};
    Push(list, VC::Op::SetViewports, VC::SetViewportsPayload{.first = 0, .count = 1},
         {reinterpret_cast<const u8*>(&viewport), sizeof(viewport)});
    Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1u << 0});
    Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
    Push(list, VC::Op::Draw, VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
    Push(list, VC::Op::ReportCounter,
         VC::ReportCounterPayload{.address = POOL_VA + REPORTS, .counter_type = SAMPLES_PASSED});
    PushEnd(list);
    context.Submit(MakeSubmission(1, std::move(list), {VERTEX_PROGRAM, 1}));
    REQUIRE(context.Wait(1, 60s));

    const Vulkan::Nvn::ContextStats stats = context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
    CHECK(StatRow(context, "render_pass_begins") == 1);
    CHECK(WordAt(data, REPORTS) == 1);
    CHECK(WordAt(data, REPORTS + 4) == 0);
}

TEST_CASE("NVN draw: unchanged dynamic state survives a masked clear",
          "[nvn][compute][pass_state][shader-refresh]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    bool refresh = false;
    SECTION("masked clear in the same submission") {}
    SECTION("shader refresh between submissions") {
        refresh = true;
    }
    struct RestoreRefresh {
        bool previous = Settings::values.nvn_shader_refresh.GetValue();
        ~RestoreRefresh() {
            Settings::values.nvn_shader_refresh.SetValue(previous);
        }
    } restore;
    Settings::values.nvn_shader_refresh.SetValue(refresh);
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    memory.Add(TEXTURE_POOL_VA, 1_MiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.early_submit = false, .shader_refresh_interval = 0s});
    if (refresh) {
        context.SetTitleId(0x1234); // exercise OpenCache with refresh mode, without profile files
    }
    context.SetGuestMemory(&memory);
    context.CreatePool(
        {.serial = 1, .guest_address = POOL_VA, .size = 64_KiB, .flags = POOL_CPU_CACHED});
    context.CreatePool({.serial = 2,
                        .guest_address = TEXTURE_POOL_VA,
                        .size = 1_MiB,
                        .flags = 1 /* CPU_NO_ACCESS */});
    VC::TextureDesc target{};
    target.target = 1;
    target.format = 0x25;
    target.width = target.height = 64;
    target.depth = target.levels = 1;
    target.serial = 10;
    target.pool_serial = 2;
    context.CreateTexture(target);
    constexpr Inst position{{0xff007322, 0x00007000, 0x00000cff, 0x000fe200}};
    const std::array<Inst, 6> point{{AddImm(0, 255, 0x3c800000), AddImm(1, 255, 0x3c800000),
                                     AddImm(2, 255, 0), AddImm(3, 255, 0x3f800000), position,
                                     Exit}};
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, point));
    std::vector<u64> list;
    const VC::RenderTargetRef color{.texture = {.serial = 10}};
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
         {reinterpret_cast<const u8*>(&color), sizeof(color)});
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = VERTEX_PROGRAM, .stages = 1u << 0});
    Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
    u64 seq = 0;
    for (u32 draw = 0; draw < 2; ++draw) {
        if (draw != 0) {
            VC::ClearColorPayload clear{};
            clear.mask = 1;
            clear.color[0] = 1.0f;
            Push(list, VC::Op::ClearColor, clear);
        }
        const VC::ViewportRect viewport{0.0f, 0.0f, 64.0f, 64.0f};
        Push(list, VC::Op::SetViewports, VC::SetViewportsPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&viewport), sizeof(viewport)});
        Push(list, VC::Op::SetLineWidth, VC::F32Payload{.value = 1.0f});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = POOL_VA + draw * 16,
                                      .counter_type = SAMPLES_PASSED});
        if (refresh) {
            PushEnd(list);
            context.Submit(MakeSubmission(++seq, std::move(list), {VERTEX_PROGRAM, 10, 1, 2}));
            list.clear();
            REQUIRE(context.Wait(seq, 60s));
            CHECK(context.DescribeMemory().find("pipelines graphics 1") != std::string::npos);
            for (u32 i = 0; i < 2; ++i) {
                context.Present({.seq = ++seq, .queue = 0});
            }
            // A present's GPU completion may precede its cache maintenance on the replay thread.
            context.Submit(MakeSubmission(++seq, {}, {}));
            REQUIRE(context.Wait(seq, 60s));
        }
    }
    if (!refresh) {
        PushEnd(list);
        context.Submit(MakeSubmission(++seq, std::move(list), {VERTEX_PROGRAM, 10, 1, 2}));
        REQUIRE(context.Wait(seq, 60s));
    } else {
        CHECK(context.DescribeMemory().find("shader refreshes 3") != std::string::npos);
    }
    CHECK(context.Stats().ops_failed == 0);
    CHECK(context.Stats().ops_unsupported == 0);
    CHECK(WordAt(data, 0) == 1);
    CHECK(WordAt(data, 16) == 1);
}

TEST_CASE("Shader refresh retains queued compute pipelines until GPU completion",
          "[nvn][compute][shader-refresh]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    struct RestoreRefresh {
        bool previous = Settings::values.nvn_shader_refresh.GetValue();
        ~RestoreRefresh() {
            Settings::values.nvn_shader_refresh.SetValue(previous);
        }
    } restore;
    Settings::values.nvn_shader_refresh.SetValue(true);
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.early_submit = false, .shader_refresh_interval = 0s});
    context.SetGuestMemory(&memory);
    context.SetTitleId(0x1234);
    context.CreatePool(
        {.serial = 1, .guest_address = POOL_VA, .size = 64_KiB, .flags = POOL_CPU_CACHED});
    context.CreateProgram(
        MakeProgram(PROGRAM, FillProgramBase(0x20, 100), {.uniform = 1, .storage = 1}));
    constexpr u32 FLIGHTS = 8;
    constexpr u32 GROUP_COUNT = 16;
    u64 seq = 0;
    for (u32 flight = 0; flight < FLIGHTS; ++flight) {
        std::vector<u64> list;
        BindCompute(list, PROGRAM, POOL_VA + 0x1000 + flight * 0x100, GROUP_COUNT * sizeof(u32));
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = GROUP_COUNT, .y = 1, .z = 1});
        PushEnd(list);
        context.Submit(MakeSubmission(++seq, std::move(list), {PROGRAM, 1}));
        context.Present({.seq = ++seq, .queue = 0});
        context.Present({.seq = ++seq, .queue = 0});
    }
    context.Submit(MakeSubmission(++seq, {}, {}));
    REQUIRE(context.Wait(seq, 60s));
    CHECK(context.DescribeMemory().find("shader refreshes 15") != std::string::npos);
    for (u32 flight = 0; flight < FLIGHTS; ++flight) {
        for (u32 word = 0; word < GROUP_COUNT; ++word) {
            CHECK(WordAt(data, 0x1000 + flight * 0x100 + word * sizeof(u32)) == 100 + word);
        }
    }
    CHECK(context.Stats().ops_failed == 0);
    CHECK(context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN draw: with asynchronous shaders only the draws the emulated path may skip are "
          "skipped while their pipeline compiles",
          "[nvn][compute]") {
    // vk_pipeline_cache.cpp CanSkipDraw / BuiltPipeline: a depth-less draw of <= 6 elements (a
    // fullscreen texture-building pass such as OoT's sky_bake_inscatter slices) and a draw that
    // stores to memory wait for their pipeline; a depth-less draw of 7 elements is skipped.
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    struct RestoreSetting {
        bool previous = Settings::values.use_asynchronous_shaders.GetValue();
        ~RestoreSetting() {
            Settings::values.use_asynchronous_shaders.SetValue(previous);
        }
    } restore;
    Settings::values.use_asynchronous_shaders.SetValue(true);

    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    //   AST.128 a[0x70], R0 (position x, y, z, w)
    constexpr Inst AstPosition{{0xff007322, 0x00007000, 0x00000cff, 0x000fe200}};
    constexpr u32 ONE_64TH = 0x3c800000;
    constexpr u32 ONE = 0x3f800000;
    // Points at the centre of pixel (32, 32) of a 64 x 64 viewport; the second program differs
    // in its depth only, so it is a pipeline of its own.
    const auto point = [&](u32 z) {
        return std::array<Inst, 6>{{AddImm(0, 255, ONE_64TH), AddImm(1, 255, ONE_64TH),
                                    AddImm(2, 255, z), AddImm(3, 255, ONE), AstPosition, Exit}};
    };
    const auto small = point(0);
    const auto large = point(0x3e800000);
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM, small));
    context.CreateProgram(MakeVertexProgram(VERTEX_PROGRAM + 1, large));
    // Every vertex stores 0x1337 at word 0 of storage buffer 0; the header says so.
    const auto store = StoreProgram(0, 0x1337);
    VC::ProgramDesc storing = MakeVertexProgram(VERTEX_PROGRAM + 2, store);
    storing.code[0][2] |= 0x01; // SPH word 0 bit 16: DoesGlobalStore
    context.CreateProgram(storing);

    const auto row = [&](std::string_view name) {
        std::vector<VC::SinkStat> rows;
        context.CollectStats(rows);
        for (const VC::SinkStat& stat : rows) {
            if (stat.reason == name) {
                return stat.count;
            }
        }
        return u64{~0ull};
    };
    constexpr u64 REPORTS = 0x2000;
    constexpr u64 STORAGE = 0x4000;
    u64 seq = 0;
    /// One submission: `count` points of `program` (its samples reported at REPORTS + 16 * seq).
    const auto draw = [&](u64 program, u32 count, bool discard) {
        ++seq;
        std::vector<u64> list;
        Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{0, 0}, {});
        const VC::ViewportRect viewport{0.0f, 0.0f, 64.0f, 64.0f};
        Push(list, VC::Op::SetViewports, VC::SetViewportsPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&viewport), sizeof(viewport)});
        Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
        Push(list, VC::Op::SetRasterizerDiscard, VC::EnablePayload{.enable = discard ? 1u : 0u});
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = program, .stages = 1u << 0});
        const VC::BufferRange range{POOL_VA + STORAGE, 4096};
        Push(list, VC::Op::BindStorageBuffers,
             VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&range), sizeof(range)});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = count, .instance_count = 1});
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = POOL_VA + REPORTS + 16 * seq,
                                      .counter_type = SAMPLES_PASSED});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list), {program, 1}));
        REQUIRE(context.Wait(seq, 60s));
        return WordAt(data, REPORTS + 16 * seq);
    };

    // A 1-vertex depth-less draw waits for its pipeline: drawn on its first use.
    CHECK(draw(VERTEX_PROGRAM, 1, false) == 1);
    CHECK(row("draw skipped awaiting async pipeline") == 0);
    // A 7-vertex depth-less draw may pop in later: skipped on its first use.
    CHECK(draw(VERTEX_PROGRAM + 1, 7, false) == 0);
    CHECK(row("draw skipped awaiting async pipeline") == 1);
    // A draw that stores to memory waits whatever its size.
    draw(VERTEX_PROGRAM + 2, 7, true);
    CHECK(WordAt(data, STORAGE) == 0x1337);
    CHECK(row("draw skipped awaiting async pipeline") == 1);
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

// ---- mirror pools: dirty guest pages, several host runs ----------------------------------------
//
// A pool whose guest storage the host cannot import whole (not one contiguous mapping, or no
// import at all) gets a device-local mirror. Only the guest pages that changed since the mirror
// last saw them are copied in, and the bytes of a range spanning several host runs are right.

namespace {

/// LDG.E.32 rd, [ra.64] (bit 79, a cache/ordering hint, set as Eglantine's lowering requires)
constexpr Inst Ldg(u32 rd, u32 ra) {
    return {{0x00007381 | (rd << 16) | (ra << 24), 0, 0x001e8900, Sched}};
}

/// Group i adds 100 to word i of storage buffer 0.
std::array<Inst, 9> AddProgram() {
    return {{S2R(0, SR_CTAID_X), MovC0(2, 0x20), MovC0(3, 0x24), MulImm(5, 0, 4), Add(2, 2, 5),
             Ldg(6, 2), AddImm(6, 6, 100), Stg(2, 6), Exit}};
}

/// Group i only reads word i of storage buffer 0.
std::array<Inst, 7> ReadProgram() {
    return {{S2R(0, SR_CTAID_X), MovC0(2, 0x20), MovC0(3, 0x24), MulImm(5, 0, 4), Add(2, 2, 5),
             Ldg(6, 2), Exit}};
}

struct MirrorRig {
    MirrorRig(HeadlessVulkan& vulkan, std::span<const Inst> program)
        : context(*vulkan.device, vulkan.submit_mutex, nullptr, {.allow_host_import = false}) {
        context.SetGuestMemory(&memory);
        this->program = MakeProgram(PROGRAM, program, {.uniform = 1, .storage = 1});
    }
    void Start(u64 pool_size) {
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = pool_size,
                            .flags = POOL_CPU_CACHED});
        context.CreateProgram(program);
    }
    /// One submission: `groups` groups over the 4 KiB (or `size` bytes) at `address`.
    void Run(u64 address, u32 groups, u64 size = 4096) {
        std::vector<u64> list;
        BindCompute(list, PROGRAM, address, size);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = groups, .y = 1, .z = 1});
        PushEnd(list);
        ++seq;
        context.Submit(MakeSubmission(seq, std::move(list), {PROGRAM, 1}));
        REQUIRE(context.Wait(seq, 60s));
    }

    FakeMemory memory;
    Vulkan::Nvn::Context context;
    VC::ProgramDesc program;
    u64 seq{0};
};

} // namespace

TEST_CASE("NVN mirror pool: only the guest pages that changed are copied again",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    MirrorRig rig(*vulkan, ReadProgram());
    rig.memory.tracking = false; // no page tracker: every page is hashed at every use
    u8* data = rig.memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    rig.Start(64_KiB);

    // First use: the page is copied in.
    rig.Run(STORAGE_VA, 6);
    Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.pools_mirrored == 1);
    REQUIRE(stats.pools_mirrored_no_import == 1);
    REQUIRE(stats.pools_imported == 0);
    REQUIRE(stats.mirror_upload_bytes == 4096);
    REQUIRE(stats.mirror_upload_runs == 1);

    // Same bytes in the next submission: nothing is copied.
    rig.Run(STORAGE_VA, 6);
    stats = rig.context.Stats();
    REQUIRE(stats.mirror_upload_bytes == 4096);
    REQUIRE(stats.mirror_ranges_clean == 1);

    // The guest writes one word without telling anybody (CPU_UNCACHED semantics): that page, and
    // only that page, is copied again.
    data[0x1100] = 0x5A;
    rig.Run(STORAGE_VA, 6);
    stats = rig.context.Stats();
    REQUIRE(stats.mirror_upload_bytes == 8192);
    REQUIRE(stats.mirror_upload_runs == 2);

    // A range over two pages, one of them changed: one page is copied, the other is not.
    rig.Run(STORAGE_VA, 6, 8192); // page 2 is new: copied
    stats = rig.context.Stats();
    REQUIRE(stats.mirror_upload_bytes == 12288);
    data[0x1200] = 0x5B;
    rig.Run(STORAGE_VA, 6, 8192); // page 1 changed, page 2 not
    stats = rig.context.Stats();
    REQUIRE(stats.mirror_upload_bytes == 16384);
    REQUIRE(stats.mirror_upload_runs == 4);
    REQUIRE(stats.mirror_pages_trusted == 0);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("NVN mirror pool: pending GPU pages skip hashes without losing later CPU changes",
          "[nvn][compute][mirror][mirror-owned]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const u64 written_bytes = GENERATE(u64{16}, u64{4096});
    FakeMemory memory;
    memory.tracking = false;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xcd, 64_KiB);
    constexpr u64 OUTPUT_VA = POOL_VA + 1_MiB;
    u8* output = memory.Add(OUTPUT_VA, 4096);
    std::atomic<bool> release{false}, entered{false};
    std::atomic<u32> phase{};
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                 {.allow_host_import = false,
                                  .early_submit = false,
                                  .mirror_hash_tracked = false,
                                  .on_queue_submit = [&] {
                                      u32 expected = 3;
                                      if (!phase.compare_exchange_strong(expected, 4)) {
                                          expected = 5;
                                          phase.compare_exchange_strong(expected, 6);
                                      }
                                  }});
    struct ReleaseCompletion {
        std::atomic<bool>& release;
        ~ReleaseCompletion() {
            release.store(true);
        }
    } unblock{release};
    const auto await = [](const auto& ready) {
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!ready()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(1ms);
        }
        return true;
    };
    context.SetGuestMemory(&memory);
    context.CreatePool(
        {.serial = 1, .guest_address = POOL_VA, .size = 64_KiB, .flags = POOL_CPU_CACHED});
    context.CreatePool(
        {.serial = 2, .guest_address = OUTPUT_VA, .size = 4096, .flags = POOL_CPU_CACHED});
    context.CreateProgram(
        MakeProgram(PROGRAM, FillProgramBase(0x20, 100), {.uniform = 1, .storage = 1}));
    const auto copy = [&](u64 seq) {
        std::vector<u64> list;
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = STORAGE_VA, .dst = OUTPUT_VA, .size = 4096});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list), {1, 2}));
    };
    copy(1);
    REQUIRE(context.Wait(1, 20s));
    context.SetCompletionCallback([&](u64 seq) {
        if (seq == 2) {
            entered.store(true);
            while (!release.load()) {
                std::this_thread::sleep_for(1ms);
            }
        }
    });
    context.Submit(MakeSubmission(2, {}, {}));
    REQUIRE(await([&] { return entered.load(); }));
    std::vector<u64> writes;
    BindCompute(writes, PROGRAM, STORAGE_VA, written_bytes);
    Push(writes, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
    PushEnd(writes);
    phase.store(3);
    context.Submit(MakeSubmission(3, std::move(writes), {PROGRAM, 1}));
    REQUIRE(await([&] { return phase.load() == 4; }));
    const u64 skipped_before = context.Stats().mirror_pages_gpu_owned;
    const u32 pending_word = 0x12345678;
    memory.CpuWrite(STORAGE_VA + 2048, &pending_word, sizeof(pending_word));
    phase.store(5);
    copy(4);
    REQUIRE(await([&] { return phase.load() == 6; }));
    const u64 skipped = context.Stats().mirror_pages_gpu_owned - skipped_before;
    CHECK(skipped == (written_bytes == 4096 ? 1u : 0u));
    release.store(true);
    REQUIRE(context.Wait(4, 20s));
    CHECK(WordAt(output, 0) == 100);
    CHECK(WordAt(output, 2048) == (written_bytes == 4096 ? 0xcdcdcdcdu : pending_word));
    // After the GPU write-back ends, another CPU store must reach the mirror.
    const u32 cpu_word = 0x87654321;
    memory.CpuWrite(STORAGE_VA + 2048, &cpu_word, sizeof(cpu_word));
    copy(5);
    REQUIRE(context.Wait(5, 20s));
    CHECK(WordAt(output, 0) == 100);
    CHECK(WordAt(output, 2048) == cpu_word);
    CHECK(context.Stats().ops_failed == 0);
    CHECK(context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN mirror pool: with the page tracker only the pages written are hashed again",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    MirrorRig rig(*vulkan, ReadProgram());
    u8* data = rig.memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    rig.Start(64_KiB);

    // First use: hashed and copied in.
    rig.Run(STORAGE_VA, 6, 8192);
    Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.mirror_upload_bytes == 8192);
    REQUIRE(stats.mirror_pages_trusted == 0);
    u64 hashed = stats.mirror_pages_hashed;
    u64 trusted = stats.mirror_pages_trusted;
    // Hashes of the last run (the pass check and the upload may each hash a changed page).
    const auto next = [&] {
        stats = rig.context.Stats();
        const u64 delta = stats.mirror_pages_hashed - hashed;
        hashed = stats.mirror_pages_hashed;
        return delta;
    };

    // Nothing was written: no page is hashed, nothing is copied.
    rig.Run(STORAGE_VA, 6, 8192);
    REQUIRE(next() == 0);
    REQUIRE(stats.mirror_upload_bytes == 8192);
    REQUIRE(stats.mirror_pages_trusted >= trusted + 2);

    // A guest CPU store: that page alone is hashed and copied again, the other stays trusted.
    const u8 stored = 0x5A;
    rig.memory.CpuWrite(STORAGE_VA + 0x1100, &stored, 1);
    trusted = rig.context.Stats().mirror_pages_trusted;
    rig.Run(STORAGE_VA, 6, 8192);
    const u64 store_hashes = next();
    REQUIRE(store_hashes >= 1);
    REQUIRE(store_hashes <= 2);
    REQUIRE(stats.mirror_upload_bytes == 12288);
    REQUIRE(stats.mirror_pages_trusted > trusted);

    rig.Run(STORAGE_VA, 6, 8192);
    REQUIRE(next() == 0);

    // A host service's write through a host pointer, announced: that page is copied again.
    data[0x1200] = 0x5B;
    rig.context.NotifyHostWrite(STORAGE_VA + 0x200, 1);
    rig.Run(STORAGE_VA, 6, 8192);
    REQUIRE(next() >= 1);
    REQUIRE(stats.mirror_upload_bytes == 16384);

    // A guest flush of the whole pool without a store keeps the pages trusted.
    rig.context.NotifyCpuWrite(POOL_VA, 64_KiB);
    rig.Run(STORAGE_VA, 6, 8192);
    REQUIRE(next() == 0);
    REQUIRE(stats.mirror_upload_bytes == 16384);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("NVN mirror pool: a page the GPU wrote is hashed again after its write-back",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    MirrorRig rig(*vulkan, AddProgram());
    u8* data = rig.memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0, 64_KiB);
    rig.Start(64_KiB);

    rig.Run(STORAGE_VA, 6);
    REQUIRE(WordAt(data, 0x1000) == 100);
    // The write-back is not a tracked write, but the page is not trusted: the guest putting the
    // old word back (also untracked here) is still seen, and the GPU adds to it.
    const u32 zero = 0;
    std::memcpy(data + 0x1000, &zero, sizeof(zero));
    rig.Run(STORAGE_VA, 6);
    REQUIRE(WordAt(data, 0x1000) == 100);
    REQUIRE(WordAt(data, 0x1004) == 200);
    REQUIRE(rig.context.Stats().ops_failed == 0);
}

TEST_CASE("NVN mirror pool: a guest write between two uses reaches the GPU",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    MirrorRig rig(*vulkan, AddProgram());
    u8* data = rig.memory.Add(POOL_VA, 64_KiB);
    std::memset(data, 0xCD, 64_KiB);
    for (u32 i = 0; i < 6; ++i) {
        const u32 value = 1000 + i;
        std::memcpy(data + 0x1000 + i * 4, &value, sizeof(value));
    }
    rig.Start(64_KiB);

    rig.Run(STORAGE_VA, 6);
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(data, 0x1000 + i * 4) == 1100 + i);
    }

    // The guest replaces word 2 and the GPU adds 100 to what the guest holds now; the other words
    // carry on from what the GPU wrote before.
    const u32 replaced = 5000;
    std::memcpy(data + 0x1000 + 2 * 4, &replaced, sizeof(replaced));
    rig.Run(STORAGE_VA, 6);
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(data, 0x1000 + i * 4) == (i == 2 ? 5100u : 1200 + i));
    }
    REQUIRE(rig.context.Stats().ops_failed == 0);
}

TEST_CASE("NVN mirror pool: a pool over two host runs is mirrored and a range across them works",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    // Guest addresses are consecutive, the host storage is two separate allocations: no single
    // host pointer covers the pool.
    constexpr u64 RUN = 64_KiB;
    // The storage buffer starts 0x40 bytes (16 words) before the end of the first run; the
    // groups 0..15 write the first run, 16..31 the second.
    constexpr u64 START = RUN - 0x40;
    constexpr u32 GROUPS = 32;
    MirrorRig rig(*vulkan, AddProgram());
    u8* first = rig.memory.Add(POOL_VA, RUN);
    u8* second = rig.memory.Add(POOL_VA + RUN, RUN);
    const auto word_at = [&](u32 i) { return i < 16 ? first + START + i * 4 : second + (i - 16) * 4; };
    for (u32 i = 0; i < GROUPS; ++i) {
        const u32 value = 70 + i;
        std::memcpy(word_at(i), &value, sizeof(value));
    }
    rig.Start(2 * RUN);
    rig.Run(POOL_VA + START, GROUPS, 4096);

    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.pools_imported == 0);
    REQUIRE(stats.pools_mirrored == 1);
    REQUIRE(stats.pools_mirrored_discontiguous == 1);
    for (u32 i = 0; i < GROUPS; ++i) {
        INFO("word " << i);
        u32 value;
        std::memcpy(&value, word_at(i), sizeof(value));
        REQUIRE(value == 170 + i);
    }
    REQUIRE(stats.ops_failed == 0);
}

// ---- read-only pool mirrors: vertex ranges (EDEN_NX2_TRY_VTX_MIRROR) and uniform ranges ---------
//
// A draw's vertex and uniform ranges of an imported pool are served from a device-local mirror
// that copies in only the stale pages of the ranges used (nvn_pool_mirror.h, PoolMirror::storage).
// The programs place one point from a float they read, at the centre of pixel (32, 32) of a
// 64 x 64 viewport when it is 1/64 and off the screen when it is 5: SAMPLES_PASSED reports count
// the points that are drawn, so a stale mirror shows as a wrong count.

namespace {

constexpr u64 VERTEX_POOL_VA = 0x5000'0000;
constexpr u64 PROGRAM_ATTRIBUTE = 120; ///< x = attribute 0
constexpr u64 PROGRAM_UNIFORM = 121;   ///< x = c[3][0], the first uniform buffer's first word
constexpr u64 PROGRAM_RAW_STORE = 122; ///< stores 5.0f through the raw pointer of its push words
constexpr u64 PROGRAM_RAW_ATOMIC = 123; ///< adds `ATOMIC_ADD` to the word of its raw push pointer
constexpr u64 PROGRAM_STORAGE_RAW_ATOMIC = 124;
constexpr u64 PROGRAM_UNIFORM_XY = 125; ///< x = c[3][0x00], y = c[3][0x10]
/// 5.0f - 1/64: added to a vertex word that holds 1/64 it makes 5.0f (the point leaves the screen).
constexpr u32 ATOMIC_ADD = 0x04200000;
constexpr u32 F32_ONE_64TH = 0x3c800000;
constexpr u32 F32_ONE = 0x3f800000;
constexpr u32 F32_FIVE = 0x40a00000;
constexpr u32 NVN_FORMAT_R32F = 0x0a;

/// IMAD.MOV.U32 rd, RZ, RZ, c[bank][byte_offset]
constexpr Inst MovCb(u32 rd, u32 bank, u32 byte_offset) {
    return {{0xff007624 | (rd << 16), (bank << 22) | (byte_offset << 6), 0x078e00ff, Sched}};
}
/// ALD rd, a[address]
constexpr Inst AldAttribute(u32 rd, u32 address) {
    return {{0xff007321 | (rd << 16), (address << 8) | 0xff, 0x00000000, 0x000e2800}};
}
constexpr Inst AstPositionWord{{0xff007322, 0x00007000, 0x00000cff, 0x000fe200}};
/// ATOMG.E.ADD.STRONG.GPU PT, rd, [ra.64], rb (nvdisasm: "ATOMG.E.ADD.STRONG.GPU PT,R7[R2],R6")
constexpr Inst AtomgAdd(u32 rd, u32 ra, u32 rb) {
    return {{0x000073a8 | (rd << 16) | (ra << 24), rb, 0x001ee100, 0x001ea800}};
}

struct VertexMirrorRig {
    VertexMirrorRig(HeadlessVulkan& vulkan, bool vtx_mirror, bool ubo_mirror, u32 vtxcov = 0,
                    std::optional<u64> storage_vram_max_bytes = std::nullopt,
                    u32 vertex_pool_flags = static_cast<u32>(POOL_CPU_CACHED),
                    bool early_submit = true)
        : context(*vulkan.device, vulkan.submit_mutex, nullptr,
                  {.early_submit = early_submit,
                   .pool_mirror = 1,
                   .try_storage_vram_max_bytes = storage_vram_max_bytes,
                   .vtx_mirror = vtx_mirror,
                   .ubo_mirror = ubo_mirror,
                   .vtxcov = vtxcov,
                   .vtxcov_diag = true}) {
        report = memory.Add(POOL_VA, 64_KiB);
        std::memset(report, 0xCD, 64_KiB);
        vertex = memory.Add(VERTEX_POOL_VA, 64_KiB);
        std::memset(vertex, 0, 64_KiB);
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});
        context.CreatePool({.serial = 2, .guest_address = VERTEX_POOL_VA, .size = 64_KiB,
                            .flags = vertex_pool_flags});
        const std::array<Inst, 6> attribute{{AldAttribute(0, 0x80), AddImm(1, 255, 0),
                                              AddImm(2, 255, 0), AddImm(3, 255, F32_ONE),
                                              AstPositionWord, Exit}};
        const std::array<Inst, 6> uniform{{MovCb(0, 3, 0), AddImm(1, 255, 0), AddImm(2, 255, 0),
                                           AddImm(3, 255, F32_ONE), AstPositionWord, Exit}};
        const std::array<Inst, 6> uniform_xy{{MovCb(0, 3, 0x00), MovCb(1, 3, 0x10),
                                              AddImm(2, 255, 0), AddImm(3, 255, F32_ONE),
                                              AstPositionWord, Exit}};
        context.CreateProgram(MakeVertexProgram(PROGRAM_ATTRIBUTE, attribute));
        context.CreateProgram(MakeVertexProgram(PROGRAM_UNIFORM, uniform));
        context.CreateProgram(MakeVertexProgram(PROGRAM_UNIFORM_XY, uniform_xy));
        const std::array<Inst, 5> raw_store{{MovC0(2, 0x10), MovC0(3, 0x14),
                                             AddImm(6, 255, F32_FIVE), Stg(2, 6), Exit}};
        context.CreateProgram(MakeProgram(PROGRAM_RAW_STORE, raw_store,
                                          {.push_words = 2, .storage = 1}));
        const std::array<Inst, 5> raw_atomic{{MovC0(2, 0x10), MovC0(3, 0x14),
                                              AddImm(6, 255, ATOMIC_ADD), AtomgAdd(7, 2, 6), Exit}};
        context.CreateProgram(MakeProgram(PROGRAM_RAW_ATOMIC, raw_atomic,
                                          {.push_words = 2, .storage = 1}));
        const std::array<Inst, 8> raw_atomic_storage{
            {MovC0(2, 0x10), MovC0(3, 0x14), AddImm(6, 255, ATOMIC_ADD),
             AtomgAdd(7, 2, 6), MovC0(8, 0x20), MovC0(9, 0x24), Stg(8, 7), Exit}};
        context.CreateProgram(MakeProgram(PROGRAM_STORAGE_RAW_ATOMIC, raw_atomic_storage,
                                          {.push_words = 2, .storage = 1}));
    }

    void SetVertex(u32 index, u32 bits) {
        std::memcpy(vertex + index * 4, &bits, sizeof(bits));
    }
    /// A guest CPU store (the page tracker sees it).
    void CpuSetVertex(u64 offset, u32 bits) {
        memory.CpuWrite(VERTEX_POOL_VA + offset, &bits, sizeof(bits));
    }
    /// Binds the vertex pool's first page as stream 0 (R32F, stride 4).
    static void BindVertices(std::vector<u64>& list) {
        const u32 attribute = NVN_FORMAT_R32F;
        Push(list, VC::Op::BindVertexAttribs, VC::BindVertexAttribsPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&attribute), sizeof(attribute)});
        const VC::VertexStreamState stream{.stride = 4, .divisor = 0};
        Push(list, VC::Op::BindVertexStreams, VC::BindVertexStreamsPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&stream), sizeof(stream)});
        BindVertexBuffer(list, 0);
    }
    /// Binds the page at `offset` of the vertex pool as stream 0's buffer.
    static void BindVertexBuffer(std::vector<u64>& list, u64 offset) {
        const VC::BufferRange range{VERTEX_POOL_VA + offset, 4096};
        Push(list, VC::Op::BindVertexBuffers, VC::BindVertexBuffersPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&range), sizeof(range)});
    }
    /// Binds `size` bytes at `offset` of the vertex pool as uniform buffer 0.
    static void BindUniform(std::vector<u64>& list, u64 offset, u64 size) {
        const VC::BufferRange range{VERTEX_POOL_VA + offset, size};
        Push(list, VC::Op::BindUniformBuffers,
             VC::BindBuffersPayload{.stage = 0, .first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&range), sizeof(range)});
    }
    /// One submission: `draws` draws of `count` points of `program` after `before` (more
    /// records), and the SAMPLES_PASSED of all of them.
    u32 Draw(u64 program, u32 count, u32 draws, const std::function<void(std::vector<u64>&)>& binds,
             const std::function<void(std::vector<u64>&)>& before = {}) {
        ++seq;
        std::vector<u64> list;
        if (before) {
            before(list);
        }
        Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{0, 0}, {});
        const VC::ViewportRect viewport{0.0f, 0.0f, 64.0f, 64.0f};
        Push(list, VC::Op::SetViewports, VC::SetViewportsPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&viewport), sizeof(viewport)});
        Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = program, .stages = 1u << 0});
        binds(list);
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        for (u32 i = 0; i < draws; ++i) {
            Push(list, VC::Op::Draw,
                 VC::DrawPayload{.mode = 0, .first = 0, .count = count, .instance_count = 1});
        }
        const u64 report_offset = 0x2000 + 16 * seq;
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = POOL_VA + report_offset,
                                      .counter_type = SAMPLES_PASSED});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list),
                                      {program, PROGRAM_RAW_STORE, PROGRAM, 1, 2}));
        REQUIRE(context.Wait(seq, 60s));
        return WordAt(report, report_offset);
    }
    u32 SubmitAndCount(std::vector<u64>& list,
                       std::initializer_list<u64> keep = {PROGRAM_UNIFORM_XY, PROGRAM_ATTRIBUTE,
                                                          PROGRAM_UNIFORM, 1, 2}) {
        ++seq;
        const u64 report_offset = 0x2000 + 16 * seq;
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = POOL_VA + report_offset,
                                      .counter_type = SAMPLES_PASSED});
        PushEnd(list);
        std::vector<u64> ids(keep);
        context.Submit(MakeSubmission(seq, std::move(list), std::move(ids)));
        REQUIRE(context.Wait(seq, 60s));
        return WordAt(report, report_offset);
    }
    static void PushTargetSetup(std::vector<u64>& list, u64 program, u64 ubo_offset, u64 ubo_size) {
        Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{0, 0}, {});
        const VC::ViewportRect viewport{0.0f, 0.0f, 64.0f, 64.0f};
        Push(list, VC::Op::SetViewports, VC::SetViewportsPayload{.first = 0, .count = 1},
             {reinterpret_cast<const u8*>(&viewport), sizeof(viewport)});
        Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = program, .stages = 1u << 0});
        BindUniform(list, ubo_offset, ubo_size);
    }
    static void PushInlineU32(std::vector<u64>& list, u64 address, u32 value) {
        std::array<u8, sizeof(u32)> bytes{};
        std::memcpy(bytes.data(), &value, sizeof(value));
        Push(list, VC::Op::UpdateUniformBuffer,
             VC::UpdateUniformBufferPayload{address, sizeof(u32)}, bytes);
    }
    /// One compute dispatch that stores 5.0f through a raw pointer to `address`.
    void RawStore(u64 address) {
        ++seq;
        std::vector<u64> list;
        BindCompute(list, PROGRAM_RAW_STORE, POOL_VA + 0x3000, 4096);
        PushRawAddress(list, address);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list),
                                      {PROGRAM_RAW_STORE, 1, 2}));
        REQUIRE(context.Wait(seq, 60s));
    }
    /// One compute dispatch that adds ATOMIC_ADD to the word at `address` with a raw atomic.
    void RawAtomicAdd(u64 address) {
        ++seq;
        std::vector<u64> list;
        BindCompute(list, PROGRAM_RAW_ATOMIC, POOL_VA + 0x3000, 4096);
        PushRawAddress(list, address);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list), {PROGRAM_RAW_ATOMIC, 1, 2}));
        REQUIRE(context.Wait(seq, 60s));
    }
    /// One raw atomic also writes a descriptor range, promoting the raw target's pool to a mirror.
    void RawAtomicAddStorage(u64 address) {
        ++seq;
        std::vector<u64> list;
        BindCompute(list, PROGRAM_STORAGE_RAW_ATOMIC, VERTEX_POOL_VA + 0x3000, 4096);
        PushRawAddress(list, address);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        PushEnd(list);
        context.Submit(MakeSubmission(seq, std::move(list), {PROGRAM_STORAGE_RAW_ATOMIC, 1, 2}));
        REQUIRE(context.Wait(seq, 60s));
    }
    /// Draws (with `expected` samples) until a read-only mirror exists: a pool is backed by the
    /// first command that uses it and is mirrored by the first draw that finds it backed. False
    /// when none came (a device without host import or write tracking).
    bool WarmUp(u64 program, u32 count, u32 draws, u32 expected,
                const std::function<void(std::vector<u64>&)>& binds) {
        for (u32 attempt = 0; attempt < 4; ++attempt) {
            REQUIRE(Draw(program, count, draws, binds) == expected);
            if (context.Stats().vram_mirrors_read_only != 0) {
                return true;
            }
        }
        return false;
    }

    FakeMemory memory;
    u8* report{};
    u8* vertex{};
    Vulkan::Nvn::Context context;
    u64 seq{0};
};

} // namespace

TEST_CASE("NVN vertex mirror: a vertex range comes from a read-only mirror that sees guest writes",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    for (const bool mirror : {false, true}) {
        INFO("vtx_mirror " << mirror);
        VertexMirrorRig rig(*vulkan, mirror, false);
        for (u32 i = 0; i < 3; ++i) {
            rig.SetVertex(i, F32_ONE_64TH);
        }
        const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
        // Three points on screen.
        Vulkan::Nvn::ContextStats stats;
        if (mirror) {
            if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
                SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
            }
            stats = rig.context.Stats();
            CHECK(stats.vram_mirrors == 1);
            CHECK(stats.vram_mirrors_read_only == 1);
            // The bound range is one page: that page is copied in, not the pool.
            CHECK(stats.vram_copy_in_bytes == 4096);
        } else {
            REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 3);
            stats = rig.context.Stats();
            CHECK(stats.vram_mirrors == 0);
        }
        REQUIRE(stats.ops_failed == 0);
        REQUIRE(stats.ops_unsupported == 0);
        const u64 ranges = stats.vram_vertex_ranges;

        // The guest moves one point off the screen between two submissions.
        rig.CpuSetVertex(4, F32_FIVE);
        REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
        stats = rig.context.Stats();
        if (mirror) {
            CHECK(stats.vram_copy_in_bytes == 2 * 4096); // the written page, once more
            CHECK(stats.vram_vertex_ranges == ranges + 1);
        }
        // Nothing written: the mirror is current, no copy.
        REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
        stats = rig.context.Stats();
        if (mirror) {
            CHECK(stats.vram_copy_in_bytes == 2 * 4096);
            CHECK(stats.vram_vertex_ranges == ranges + 2);
        }
        // And it can come back.
        rig.CpuSetVertex(4, F32_ONE_64TH);
        REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 3);
        REQUIRE(rig.context.Stats().ops_failed == 0);
    }
}

TEST_CASE("NVN vertex mirror: a raw global store makes the read-only mirror stale",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    // Written for TRY_RO_MIRROR_EXACT off (now on by default): pin it for this test.
    namespace D = Common::DiagRequest;
    struct SwitchGuard {
        bool initial = D::TrySwitchEnabled("TRY_RO_MIRROR_EXACT");
        SwitchGuard() {
            REQUIRE(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", false));
        }
        ~SwitchGuard() {
            static_cast<void>(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", initial));
        }
    } guard;
    VertexMirrorRig rig(*vulkan, true, false);
    for (u32 i = 0; i < 3; ++i) {
        rig.SetVertex(i, F32_ONE_64TH);
    }
    const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
    if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }
    const u64 copied = rig.context.Stats().vram_copy_in_bytes;

    // A shader writes the second vertex through a raw pointer: the guest bytes change, the
    // page tracker sees nothing. The next draw must see it.
    rig.RawStore(VERTEX_POOL_VA + 4);
    REQUIRE(WordAt(rig.vertex, 4) == F32_FIVE);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    CHECK(stats.vram_invalidate_read_only >= 1);
    CHECK(stats.vram_copy_in_bytes == copied + 4096);
    CHECK(stats.ops_failed == 0);
}

TEST_CASE("NVN vertex mirror: a written-through raw store keeps the read-only mirror current "
          "(TRY_RO_MIRROR_EXACT)",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    namespace D = Common::DiagRequest;
    // The programs are translated with the switch (it is read at translation).
    struct SwitchGuard {
        bool initial = D::TrySwitchEnabled("TRY_RO_MIRROR_EXACT");
        SwitchGuard() {
            REQUIRE(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", true));
        }
        ~SwitchGuard() {
            static_cast<void>(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", initial));
        }
    } guard;
    VertexMirrorRig rig(*vulkan, true, false);
    for (u32 i = 0; i < 3; ++i) {
        rig.SetVertex(i, F32_ONE_64TH);
    }
    const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
    if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }
    const u64 copied = rig.context.Stats().vram_copy_in_bytes;

    // A shader writes the second vertex through a raw pointer: the guest bytes and the mirror's
    // page both take it, so the next draw sees it from the mirror without a copy.
    rig.RawStore(VERTEX_POOL_VA + 4);
    REQUIRE(WordAt(rig.vertex, 4) == F32_FIVE);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
    Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    CHECK(stats.vram_ro_written_through == 1);
    CHECK(stats.vram_raw_alias_rebuilds >= 1);
    CHECK(stats.vram_invalidate_read_only == 0);
    CHECK(stats.vram_copy_in_bytes == copied);
    CHECK(stats.vram_vertex_ranges >= 2);

    // A store outside every mirror (the report pool) changes nothing in it.
    rig.RawStore(POOL_VA + 0x100);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
    CHECK(rig.context.Stats().vram_copy_in_bytes == copied);

    // The guest CPU rewrites the third vertex (the page turns stale), then a raw store moves the
    // first one: the page is copied in again from the guest bytes, which hold both writes.
    rig.CpuSetVertex(8, F32_FIVE);
    rig.RawStore(VERTEX_POOL_VA + 0);
    REQUIRE(WordAt(rig.vertex, 0) == F32_FIVE);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 0);
    stats = rig.context.Stats();
    CHECK(stats.vram_copy_in_bytes == copied + 4096);
    CHECK(stats.vram_invalidate_read_only == 0);
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN vertex mirror: a raw atomic is written through to the read-only mirror too "
          "(TRY_RO_MIRROR_EXACT)",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    namespace D = Common::DiagRequest;
    // The programs are translated with the switch (it is read at translation).
    struct SwitchGuard {
        bool initial = D::TrySwitchEnabled("TRY_RO_MIRROR_EXACT");
        SwitchGuard() {
            REQUIRE(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", true));
        }
        ~SwitchGuard() {
            static_cast<void>(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", initial));
        }
    } guard;
    VertexMirrorRig rig(*vulkan, true, false);
    for (u32 i = 0; i < 3; ++i) {
        rig.SetVertex(i, F32_ONE_64TH);
    }
    const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
    if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }
    const u64 copied = rig.context.Stats().vram_copy_in_bytes;

    // The atomic moves the second vertex off the screen in the guest word; the mirror's word takes
    // the same addition, so the next draw sees it from the mirror without a copy.
    rig.RawAtomicAdd(VERTEX_POOL_VA + 4);
    REQUIRE(WordAt(rig.vertex, 4) == F32_FIVE);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    CHECK(stats.vram_ro_written_through == 1);
    CHECK(stats.vram_invalidate_read_only == 0);
    CHECK(stats.vram_copy_in_bytes == copied);
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN storage mirror: a raw atomic writes through to imported guest memory",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsDescriptorAliasingSupported()) {
        SKIP("raw-global memory is unsupported on this Vulkan device");
    }
    namespace D = Common::DiagRequest;
    const bool initial = D::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH");
    struct SwitchReset {
        bool initial;
        ~SwitchReset() {
            static_cast<void>(
                Common::DiagRequest::SetTrySwitch("TRY_STORAGE_RAW_WRITE_THROUGH", initial));
        }
    } reset{initial};
    REQUIRE(D::SetTrySwitch("TRY_STORAGE_RAW_WRITE_THROUGH", true));
    VertexMirrorRig rig(*vulkan, true, false);
    for (u32 i = 0; i < 3; ++i) {
        rig.SetVertex(i, F32_ONE_64TH);
    }
    const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
    if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }

    rig.RawAtomicAddStorage(VERTEX_POOL_VA + 4);
    REQUIRE(WordAt(rig.vertex, 4) == F32_FIVE);
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    CHECK(stats.shadow_bytes_in == 4096);
    CHECK(stats.vram_write_bytes == 4096);
    CHECK(stats.vram_raw_alias_rebuilds >= 1);
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
    CHECK(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 2);
}

TEST_CASE("NVN vertex mirror: a written storage range promotes the mirror and both writers keep "
          "their bytes",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    VertexMirrorRig rig(*vulkan, true, false);
    rig.context.CreateProgram(MakeProgram(PROGRAM, FillProgramBase(0x20, 100), {.uniform = 1, .storage = 1}));
    for (u32 i = 0; i < 3; ++i) {
        rig.SetVertex(i, F32_ONE_64TH);
    }
    const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
    if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }

    // The guest rewrites a vertex of page 0; the GPU fills words 0x200.. of the same page through
    // a storage binding (a written range: the mirror becomes a storage mirror) and then draws
    // the vertices in the same submission.
    rig.CpuSetVertex(4, F32_FIVE);
    const auto fill = [&](std::vector<u64>& list) {
        Dispatch(list, PROGRAM, VERTEX_POOL_VA + 0x800, 256, 6);
    };
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds, fill) == 2);
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.vertex, 0x800 + i * 4) == 100 + i); // the GPU's words reached the guest
    }
    REQUIRE(WordAt(rig.vertex, 4) == F32_FIVE);                 // the guest's word survived the merge
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    CHECK(stats.vram_mirrors_promoted == 1);
    CHECK(stats.ops_failed == 0);
    // Later draws read the page as the merge left it.
    rig.CpuSetVertex(4, F32_ONE_64TH);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 3);
    REQUIRE(WordAt(rig.vertex, 0x800 + 5 * 4) == 105);
}

// ---- vertex coverage (EDEN_NX2_NVN_VTXCOV_DIAG, EDEN_NX2_TRY_VTXCOV_*, nvn_vtxcov.h) ------------

namespace {

using Vulkan::Nvn::VtxCov;

u64 CovBinds(const Vulkan::Nvn::ContextStats& stats, VtxCov reason) {
    return stats.vtxcov_binds[static_cast<size_t>(reason)];
}

/// One submission of one pass: a draw of 3 points from page 2 of the vertex pool (bound before
/// the pass), the same SetRenderTargets again (the pass goes on, the staging scan stopped there)
/// and a draw of 3 points from page 0, bound only then. Returns the SAMPLES_PASSED of both.
u32 DrawInPass(VertexMirrorRig& rig) {
    ++rig.seq;
    std::vector<u64> list;
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{0, 0}, {});
    const VC::ViewportRect viewport{0.0f, 0.0f, 64.0f, 64.0f};
    Push(list, VC::Op::SetViewports, VC::SetViewportsPayload{.first = 0, .count = 1},
         {reinterpret_cast<const u8*>(&viewport), sizeof(viewport)});
    Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = PROGRAM_ATTRIBUTE, .stages = 1u << 0});
    VertexMirrorRig::BindVertices(list);
    VertexMirrorRig::BindVertexBuffer(list, 0x2000);
    Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
    Push(list, VC::Op::Draw, VC::DrawPayload{.mode = 0, .first = 0, .count = 3, .instance_count = 1});
    Push(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{0, 0}, {});
    VertexMirrorRig::BindVertexBuffer(list, 0);
    Push(list, VC::Op::Draw, VC::DrawPayload{.mode = 0, .first = 0, .count = 3, .instance_count = 1});
    const u64 report_offset = 0x2000 + 16 * rig.seq;
    Push(list, VC::Op::ReportCounter,
         VC::ReportCounterPayload{.address = POOL_VA + report_offset, .counter_type = SAMPLES_PASSED});
    PushEnd(list);
    rig.context.Submit(MakeSubmission(rig.seq, std::move(list), {PROGRAM_ATTRIBUTE, 1, 2}));
    REQUIRE(rig.context.Wait(rig.seq, 60s));
    return WordAt(rig.report, report_offset);
}

} // namespace

TEST_CASE("NVN vertex coverage: a stream bound inside an open pass is counted, and served from the "
          "mirror from the next pass on with TRY_VTXCOV_INPASS",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    {
        // No pool mirrors: page 2 is in a pass of too few draws to stage, page 0 was not scanned.
        VertexMirrorRig rig(*vulkan, false, false);
        for (u32 i = 0; i < 3; ++i) {
            rig.SetVertex(i, F32_ONE_64TH);
            rig.SetVertex(0x2000 / 4 + i, F32_FIVE);
        }
        REQUIRE(DrawInPass(rig) == 3); // the pool is backed by its first use
        REQUIRE(DrawInPass(rig) == 3);
        const Vulkan::Nvn::ContextStats before = rig.context.Stats();
        REQUIRE(DrawInPass(rig) == 3);
        const Vulkan::Nvn::ContextStats after = rig.context.Stats();
        const auto delta = [&](VtxCov reason) {
            return CovBinds(after, reason) - CovBinds(before, reason);
        };
        if (after.pools_imported == 0) {
            // No host import: the pool is device-local itself (Backing::Mirror).
            CHECK(delta(VtxCov::Device) == 2);
        } else {
            CHECK(delta(VtxCov::FewDraws) == 1);
            CHECK(delta(VtxCov::NotScannedTargets) == 1);
        }
        CHECK(delta(VtxCov::MirrorBind) == 0);
        CHECK(after.ops_failed == 0);
    }
    for (const bool inpass : {false, true}) {
        INFO("inpass " << inpass);
        VertexMirrorRig rig(*vulkan, true, false, inpass ? Vulkan::Nvn::VTXCOV_INPASS : 0u);
        for (u32 i = 0; i < 3; ++i) {
            rig.SetVertex(i, F32_ONE_64TH);               // page 0: on the screen
            rig.SetVertex(0x2000 / 4 + i, F32_FIVE);      // page 2: off the screen
        }
        bool mirrored = false;
        for (u32 attempt = 0; attempt < 4 && !mirrored; ++attempt) {
            REQUIRE(DrawInPass(rig) == 3);
            mirrored = rig.context.Stats().vram_mirrors_read_only != 0;
        }
        if (!mirrored) {
            SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
        }
        // One more pass: with the switch, page 0 joined the prefetch set at its bind.
        REQUIRE(DrawInPass(rig) == 3);
        Vulkan::Nvn::ContextStats before = rig.context.Stats();
        REQUIRE(DrawInPass(rig) == 3);
        Vulkan::Nvn::ContextStats after = rig.context.Stats();
        const auto delta = [&](VtxCov reason) {
            return CovBinds(after, reason) - CovBinds(before, reason);
        };
        CHECK(delta(VtxCov::Mirror) == 1); // page 2, through the pass's scan
        CHECK(delta(VtxCov::NotScannedTargets) == (inpass ? 0u : 1u));
        CHECK(delta(VtxCov::MirrorBind) == (inpass ? 1u : 0u));
        CHECK(after.ops_failed == 0);

        // The guest moves a point of page 0 off the screen: the prefetch at the next pass begin
        // copies the written page in, and the bind still finds the mirror current.
        rig.CpuSetVertex(4, F32_FIVE);
        before = rig.context.Stats();
        REQUIRE(DrawInPass(rig) == 2);
        after = rig.context.Stats();
        CHECK(delta(VtxCov::MirrorBind) == (inpass ? 1u : 0u));
        rig.CpuSetVertex(4, F32_ONE_64TH);
        REQUIRE(DrawInPass(rig) == 3);
        REQUIRE(rig.context.Stats().ops_failed == 0);
    }
}

TEST_CASE("NVN vertex coverage: TRY_VTXCOV_BIGPOOL mirrors a pool over the storage size limit "
          "read-only and never promotes it",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const auto binds = [&](std::vector<u64>& list) { VertexMirrorRig::BindVertices(list); };
    {
        // Without the switch the 64 KiB pool is over the (test) limit of 32 KiB: no mirror.
        VertexMirrorRig rig(*vulkan, true, false, 0, 32_KiB);
        for (u32 i = 0; i < 3; ++i) {
            rig.SetVertex(i, F32_ONE_64TH);
        }
        for (u32 i = 0; i < 4; ++i) {
            REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 3);
        }
        const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
        CHECK(stats.vram_mirrors_read_only == 0);
        CHECK(CovBinds(stats, VtxCov::Mirror) == 0);
    }
    VertexMirrorRig rig(*vulkan, true, false, Vulkan::Nvn::VTXCOV_BIGPOOL, 32_KiB);
    rig.context.CreateProgram(
        MakeProgram(PROGRAM, FillProgramBase(0x20, 100), {.uniform = 1, .storage = 1}));
    for (u32 i = 0; i < 3; ++i) {
        rig.SetVertex(i, F32_ONE_64TH);
    }
    if (!rig.WarmUp(PROGRAM_ATTRIBUTE, 3, 1, 3, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }
    CHECK(CovBinds(rig.context.Stats(), VtxCov::Mirror) >= 1);
    // A written storage range of the pool: with TRY_STORAGE_VRAM it stays in guest memory (the
    // mirror stays read-only) and makes the mirror's page stale; the next draw sees both writers.
    rig.CpuSetVertex(4, F32_FIVE);
    const auto fill = [&](std::vector<u64>& list) {
        Dispatch(list, PROGRAM, VERTEX_POOL_VA + 0x800, 256, 6);
    };
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds, fill) == 2);
    for (u32 i = 0; i < 6; ++i) {
        INFO("word " << i);
        REQUIRE(WordAt(rig.vertex, 0x800 + i * 4) == 100 + i);
    }
    const char* const storage_vram_env = std::getenv("EDEN_NX2_TRY_STORAGE_VRAM");
    const bool storage_vram = vulkan->device->IsBufferDeviceAddressSupported() &&
                              vulkan->device->IsShaderInt64Supported() &&
                              vulkan->device->IsDescriptorAliasingSupported() &&
                              (storage_vram_env == nullptr || std::string_view{storage_vram_env} != "0");
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    if (storage_vram) {
        CHECK(stats.vram_mirrors_promoted == 0);
        CHECK(stats.vram_mirrors == 1);
    }
    CHECK(stats.ops_failed == 0);
    rig.CpuSetVertex(4, F32_ONE_64TH);
    REQUIRE(rig.Draw(PROGRAM_ATTRIBUTE, 3, 1, binds) == 3);
    REQUIRE(rig.context.Stats().ops_failed == 0);
}

TEST_CASE("NVN uniform mirror: the first copy is the pages the bindings use, not the pool",
          "[nvn][compute][mirror]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    VertexMirrorRig rig(*vulkan, false, true);
    constexpr u64 UNIFORM = 0x5000; // page 5 of a 64 KiB pool
    const u32 on_screen = F32_ONE_64TH;
    std::memcpy(rig.vertex + UNIFORM, &on_screen, sizeof(on_screen));
    const auto binds = [&](std::vector<u64>& list) {
        VertexMirrorRig::BindUniform(list, UNIFORM, 256);
    };
    // Two draws of one pass share the binding: the first (before the pass) copies the page, the
    // second (inside it) is served because it is current.
    if (!rig.WarmUp(PROGRAM_UNIFORM, 1, 2, 2, binds)) {
        SKIP("pool mirrors are unavailable on this device (no host import / write tracking)");
    }
    const Vulkan::Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
    CHECK(stats.vram_mirrors == 1);
    CHECK(stats.vram_copy_in_bytes == 4096); // one page, not the 64 KiB pool

    // A page nobody binds changes: nothing is copied for it, at any pass begin.
    const u32 elsewhere = 0x12345678;
    rig.memory.CpuWrite(VERTEX_POOL_VA + 0x9000, &elsewhere, sizeof(elsewhere));
    REQUIRE(rig.Draw(PROGRAM_UNIFORM, 1, 2, binds) == 2);
    CHECK(rig.context.Stats().vram_copy_in_bytes == 4096);

    // The bound page changes: it is copied in again before the next pass reads it, and only it.
    rig.CpuSetVertex(UNIFORM, F32_FIVE);
    REQUIRE(rig.Draw(PROGRAM_UNIFORM, 1, 2, binds) == 0);
    CHECK(rig.context.Stats().vram_copy_in_bytes == 2 * 4096);
    rig.CpuSetVertex(UNIFORM, F32_ONE_64TH);
    REQUIRE(rig.Draw(PROGRAM_UNIFORM, 1, 2, binds) == 2);
    CHECK(rig.context.Stats().vram_copy_in_bytes == 3 * 4096);
    CHECK(rig.context.Stats().ops_failed == 0);
}

TEST_CASE("MKW split mirrors observe raw GPU stores before later descriptor reads",
          "[nvn][compute][mirror][mkw]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan || !vulkan->device->IsBufferDeviceAddressSupported() ||
        !vulkan->device->IsShaderInt64Supported() ||
        !vulkan->device->IsExtExternalMemoryHostSupported()) {
        SKIP("split raw imports unavailable");
    }
    const bool mkw = GENERATE(false, true);
    FakeMemory memory;
    constexpr u64 RUN = 64_KiB;
    u8* first = memory.Add(POOL_VA, RUN);
    memory.Add(POOL_VA + RUN, RUN);
    constexpr u64 OUTPUT = POOL_VA + 4 * RUN;
    u8* output = memory.Add(OUTPUT, RUN);
    Vulkan::Nvn::Context context(
        *vulkan->device, vulkan->submit_mutex, nullptr,
        {.early_submit = false, .try_storage_vram = false, .mirror_hash_tracked = true});
    context.SetGuestMemory(&memory);
    context.SetTitleId(mkw ? Settings::MarioKartWorldProgramId : Settings::OcarinaOfTimeProgramId);
    context.CreatePool(
        {.serial = 1, .guest_address = POOL_VA, .size = 2 * RUN, .flags = POOL_CPU_CACHED});
    context.CreatePool(
        {.serial = 2, .guest_address = OUTPUT, .size = RUN, .flags = POOL_CPU_CACHED});
    const std::array<Inst, 5> raw_store{MovC0(2, 0x10), MovC0(3, 0x14), AddImm(6, 255, 0x1234),
                                        Stg(2, 6), Exit};
    context.CreateProgram(MakeProgram(PROGRAM_RAW_WRITE, raw_store, {.push_words = 2}));
    u64 seq = 0;
    const auto run = [&](std::vector<u64> list) {
        PushEnd(list);
        context.Submit(MakeSubmission(++seq, std::move(list), {1, 2, PROGRAM_RAW_WRITE}));
        REQUIRE(context.Wait(seq, 20s));
    };
    const auto copy = [&](u64 source) {
        std::vector<u64> list;
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = source, .dst = OUTPUT, .size = 4});
        run(std::move(list));
    };
    copy(POOL_VA + 0x1000);
    if (context.Stats().pools_split_imported == 0) {
        SKIP("device could not import split runs");
    }
    REQUIRE(WordAt(output, 0) == 0);
    const u64 hashed = context.Stats().mirror_pages_hashed;
    copy(POOL_VA + 0x1000);
    REQUIRE(context.Stats().mirror_pages_hashed == hashed);
    for (const u64 offset : {0x1000u, 0x2000u, 0x1000u}) {
        std::vector<u64> list;
        Push(list, VC::Op::BindProgram,
             VC::BindProgramPayload{.program_serial = PROGRAM_RAW_WRITE, .stages = 1u << 5});
        PushRawAddress(list, POOL_VA + offset);
        Push(list, VC::Op::Dispatch, VC::DispatchPayload{.x = 1, .y = 1, .z = 1});
        run(std::move(list));
        REQUIRE(WordAt(first, offset) == 0x1234);
        copy(POOL_VA + offset);
        REQUIRE(WordAt(output, 0) == 0x1234);
        const u32 cpu = 0x4567;
        memory.CpuWrite(POOL_VA + offset, &cpu, sizeof(cpu));
        copy(POOL_VA + offset);
        REQUIRE(WordAt(output, 0) == cpu);
    }
    REQUIRE(context.Stats().ops_failed == 0);
    REQUIRE(context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN inline uniform: flushed inline update plus later partial update stays coherent",
          "[nvn][compute][inline]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 UBO_OFFSET = 0x5000;
    constexpr u64 UBO_VA = VERTEX_POOL_VA + UBO_OFFSET;
    const bool ubo_mirror = GENERATE(false, true);
    INFO("ubo_mirror " << ubo_mirror);
    VertexMirrorRig rig(*vulkan, false, ubo_mirror, 0, std::nullopt, POOL_CPU_CACHED, false);

    auto reset_ubo = [&]() {
        rig.CpuSetVertex(UBO_OFFSET + 0x00, F32_FIVE);
        rig.CpuSetVertex(UBO_OFFSET + 0x10, F32_FIVE);
    };

    // Case 1: outside an open pass — flush first inline update mid-submission, then update a
    // different offset of the same UBO before drawing.
    reset_ubo();
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x00, F32_ONE_64TH);
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = POOL_VA + 0x1000,
                                           .dst = POOL_VA + 0x1100,
                                           .size = 16});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }

    // Case 2: inside an open pass — first inline update was flushed before the pass, second
    // update arrives inside the pass; both apply without breaking the open pass.
    reset_ubo();
    {
        const auto break_count = [&]() {
            std::vector<VC::SinkStat> rows;
            rig.context.CollectStats(rows);
            const auto it = std::ranges::find(
                rows, std::string_view{"render_pass_break: inline uniform update flush"},
                &VC::SinkStat::reason);
            return it == rows.end() ? u64{0} : it->count;
        };
        const u64 inline_breaks_before = break_count();
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x00, F32_ONE_64TH);
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = POOL_VA + 0x1000,
                                           .dst = POOL_VA + 0x1100,
                                           .size = 16});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        // Opens the render pass; +0x10 is still F32_FIVE so this first draw emits 0 samples.
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        // Same open pass must see both +0x00 (flushed inline) and +0x10 (pending inline).
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
        CHECK(break_count() == inline_breaks_before);
    }

    // Case 3: non-inline GPU write (ClearBuffer at +0x00) + partial UpdateUniformBuffer(+0x10),
    // both outside a pass and inside an open pass.
    reset_ubo();
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 4, .value = F32_ONE_64TH});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }
    reset_ubo();
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 4, .value = F32_ONE_64TH});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }

    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN inline uniform: DeviceLocal UBO pool sees prior GPU write plus inline update",
          "[nvn][compute][inline]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 UBO_OFFSET = 0x5000;
    constexpr u64 UBO_VA = VERTEX_POOL_VA + UBO_OFFSET;
    VertexMirrorRig rig(*vulkan, false, false, 0, std::nullopt, 1 /* POOL_CPU_NO_ACCESS */, false);

    // Outside a pass
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 256, .value = F32_FIVE});
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 4, .value = F32_ONE_64TH});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }

    // Inside an open pass (must flush inline updates to DeviceLocal buffer)
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 256, .value = F32_FIVE});
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 4, .value = F32_ONE_64TH});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }

    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN inline uniform: early submit does not split pending updates outside an open pass",
          "[nvn][compute][inline]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 UBO_OFFSET = 0x5000;
    constexpr u64 UBO_VA = VERTEX_POOL_VA + UBO_OFFSET;
    VertexMirrorRig rig(*vulkan, false, false, 0, std::nullopt, POOL_CPU_CACHED, true);
    rig.CpuSetVertex(UBO_OFFSET + 0x00, F32_FIVE);
    rig.CpuSetVertex(UBO_OFFSET + 0x10, F32_FIVE);

    std::vector<u64> list;
    // Compile PROGRAM_ATTRIBUTE first so elapsed time passes early_first_us before the 32-op check,
    // then reset the occlusion counter to close the pass, queue one inline update, emit >32
    // harmless state ops (hitting MaybeSubmitEarly while !PassOpen() && HasInlineUpdates()), queue
    // the second inline update, and draw.
    VertexMirrorRig::PushTargetSetup(list, PROGRAM_ATTRIBUTE, UBO_OFFSET, 256);
    VertexMirrorRig::BindVertices(list);
    Push(list, VC::Op::Draw,
         VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
    Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
    Push(list, VC::Op::BindProgram,
         VC::BindProgramPayload{.program_serial = PROGRAM_UNIFORM_XY, .stages = 1u << 0});
    VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x00, F32_ONE_64TH);
    for (int i = 0; i < 40; ++i) {
        Push(list, VC::Op::SetPointSize, VC::F32Payload{.value = 1.0f});
    }
    VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
    Push(list, VC::Op::Draw,
         VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
    CHECK(rig.SubmitAndCount(list) == 1);
    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN inline uniform: misaligned DeviceLocal UBO preserves GPU writes plus inline update",
          "[nvn][compute][inline]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 UBO_OFFSET = 0x5040;
    constexpr u64 UBO_VA = VERTEX_POOL_VA + UBO_OFFSET;
    VertexMirrorRig rig(*vulkan, false, false, 0, std::nullopt, 1 /* POOL_CPU_NO_ACCESS */, false);

    // ClearBuffer writes F32_FIVE (off-screen) to +0x00 on the GPU while guest RAM stays 0.0f.
    // Losing GPU bytes to stale guest RAM would draw on-screen (1 sample) instead of clipping (0).
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 256, .value = F32_FIVE});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 0);
    }

    // Outside a pass: GPU ClearBuffer sets +0x00 to 1/64, inline update sets +0x10 to 1/64.
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 256, .value = F32_FIVE});
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 4, .value = F32_ONE_64TH});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }

    // Inside an open pass: pass ends, inline updates flush, and GPU copy preserves both.
    {
        std::vector<u64> list;
        VertexMirrorRig::PushTargetSetup(list, PROGRAM_UNIFORM_XY, UBO_OFFSET, 256);
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 256, .value = F32_FIVE});
        Push(list, VC::Op::ClearBuffer,
             VC::ClearBufferPayload{.address = UBO_VA + 0x00, .size = 4, .value = F32_ONE_64TH});
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = SAMPLES_PASSED});
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        VertexMirrorRig::PushInlineU32(list, UBO_VA + 0x10, F32_ONE_64TH);
        Push(list, VC::Op::Draw,
             VC::DrawPayload{.mode = 0, .first = 0, .count = 1, .instance_count = 1});
        CHECK(rig.SubmitAndCount(list) == 1);
    }

    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}
