// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the NVN Vulkan draw path that run without game shaders: render target passes, clears
// (scissored, channel-masked, depth, partial stencil) and inline uniform updates. GPU tests SKIP
// when the machine has no Vulkan device.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_viewport.h"
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

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
    std::mutex submit_mutex;
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
            *result->instance, Vulkan::vk::PhysicalDevice(devices[0], *result->dld), VK_NULL_HANDLE,
            *result->dld);
        return result;
    } catch (const Vulkan::vk::Exception&) {
        return nullptr;
    } catch (const std::exception&) {
        return nullptr;
    }
}

// --- record encoding -------------------------------------------------------------------------

/// A record: header, `prefix`, then `extra` bytes (padded to 8).
template <typename T>
void Push(std::vector<u64>& list, VC::Op op, const T& prefix, std::span<const u8> extra = {},
          u16 flags = 0) {
    const size_t bytes = sizeof(T) + extra.size();
    const size_t words = 1 + (bytes + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(op);
    header->flags = flags;
    header->size = static_cast<u32>(words * 8);
    auto* payload = reinterpret_cast<u8*>(&list[at + 1]);
    std::memcpy(payload, &prefix, sizeof(T));
    if (!extra.empty()) {
        std::memcpy(payload + sizeof(T), extra.data(), extra.size());
    }
}

void PushEnd(std::vector<u64>& list) {
    const size_t at = list.size();
    list.resize(at + 1, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(VC::Op::End);
    header->size = 8;
}

template <typename T>
std::span<const u8> Bytes(const T& value) {
    return {reinterpret_cast<const u8*>(&value), sizeof(T)};
}

VC::Submission MakeSubmission(u64 seq, std::vector<u64> list, std::vector<u64> retained) {
    VC::Submission submission{};
    submission.seq = seq;
    submission.queue = 0;
    submission.lists.push_back(std::move(list));
    submission.retained_serials = std::move(retained);
    return submission;
}

constexpr u64 POOL_CPU_NO_ACCESS = 1;
constexpr u64 POOL_CPU_CACHED = 4;
constexpr u64 TEXTURE_POOL_VA = 0x1000'0000;
constexpr u64 BUFFER_POOL_VA = 0x2000'0000;
constexpr u32 TEX_SIZE = 64;

VC::TextureDesc MakeTexture(u64 serial, s32 format, u64 pool_offset) {
    VC::TextureDesc desc{};
    desc.target = 1; // 2D
    desc.format = format;
    desc.width = TEX_SIZE;
    desc.height = TEX_SIZE;
    desc.depth = 1;
    desc.levels = 1;
    desc.serial = serial;
    desc.pool_serial = 1;
    desc.pool_offset = pool_offset;
    return desc;
}

VC::RenderTargetRef Target(u64 serial) {
    VC::RenderTargetRef ref{};
    ref.texture = {.serial = serial, .level = 0, .layer = 0};
    return ref;
}

void SetTargets(std::vector<u64>& list, std::vector<VC::RenderTargetRef> colors,
                std::optional<VC::RenderTargetRef> depth = std::nullopt) {
    std::vector<VC::RenderTargetRef> all = colors;
    if (depth) {
        all.push_back(*depth);
    }
    Push(list, VC::Op::SetRenderTargets,
         VC::SetRenderTargetsPayload{static_cast<u32>(colors.size()), depth ? 1u : 0u},
         {reinterpret_cast<const u8*>(all.data()), all.size() * sizeof(VC::RenderTargetRef)});
}

void ClearColor(std::vector<u64>& list, std::array<float, 4> color, u32 mask = 0xF) {
    VC::ClearColorPayload clear{};
    clear.index = 0;
    clear.mask = mask;
    std::memcpy(clear.color, color.data(), sizeof(clear.color));
    Push(list, VC::Op::ClearColor, clear);
}

void CopyToBuffer(std::vector<u64>& list, u64 texture, u64 buffer) {
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = buffer;
    copy.texture = {.serial = texture, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, TEX_SIZE, TEX_SIZE, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
}

/// A context with a 1 MiB GPU-only texture pool and a 1 MiB CPU readback pool.
struct Rig {
    explicit Rig(HeadlessVulkan& vulkan, const Vulkan::Nvn::ContextOptions& options = {})
        : context(*vulkan.device, vulkan.submit_mutex, nullptr, options) {
        texture_bytes = memory.Add(TEXTURE_POOL_VA, 1_MiB);
        readback = memory.Add(BUFFER_POOL_VA, 1_MiB);
        std::memset(readback, 0xCD, 1_MiB);
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = TEXTURE_POOL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_NO_ACCESS});
        context.CreatePool({.serial = 2, .guest_address = BUFFER_POOL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_CACHED});
    }

    void Run(std::vector<u64> list, std::vector<u64> retained) {
        PushEnd(list);
        context.Submit(MakeSubmission(++seq, std::move(list), std::move(retained)));
        REQUIRE(context.Wait(seq, 20s));
    }

    FakeMemory memory;
    Vulkan::Nvn::Context context;
    u8* texture_bytes{};
    u8* readback{};
    u64 seq{};
};

/// Returns the texel (x, y) of an RGBA8 readback at buffer offset `base`.
std::array<u8, 4> Texel(const u8* readback, u32 base, u32 x, u32 y) {
    const u8* p = readback + base + (y * TEX_SIZE + x) * 4;
    return {p[0], p[1], p[2], p[3]};
}

bool Near(u8 value, int expected) {
    return std::abs(int(value) - expected) <= 1;
}

} // namespace

TEST_CASE("NVN draw: a render target pass clears with a scissor", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(MakeTexture(10, 0x25, 0));
    std::vector<u64> list;
    SetTargets(list, {Target(10)});
    ClearColor(list, {0.2f, 0.4f, 0.6f, 0.8f});
    // Scissor (16, 8) 16x32: the second clear only touches it.
    Push(list, VC::Op::SetScissors, VC::SetScissorsPayload{0, 1},
         Bytes(VC::ScissorRect{16, 8, 16, 32}));
    ClearColor(list, {1.0f, 0.0f, 0.0f, 1.0f});
    CopyToBuffer(list, 10, BUFFER_POOL_VA);
    rig.Run(std::move(list), {10});

    const auto inside = Texel(rig.readback, 0, 20, 20);
    CHECK(Near(inside[0], 255));
    CHECK(Near(inside[1], 0));
    CHECK(Near(inside[2], 0));
    CHECK(Near(inside[3], 255));
    for (const auto& [x, y] : {std::pair{15u, 20u}, std::pair{32u, 20u}, std::pair{20u, 7u},
                               std::pair{20u, 40u}, std::pair{0u, 0u}, std::pair{63u, 63u}}) {
        const auto outside = Texel(rig.readback, 0, x, y);
        INFO("texel " << x << "," << y);
        CHECK(Near(outside[0], 51));
        CHECK(Near(outside[1], 102));
        CHECK(Near(outside[2], 153));
        CHECK(Near(outside[3], 204));
    }
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN draw: a color clear honours the channel write mask", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(MakeTexture(10, 0x25, 0));
    std::vector<u64> list;
    SetTargets(list, {Target(10)});
    ClearColor(list, {0.2f, 0.4f, 0.6f, 0.8f});
    ClearColor(list, {1.0f, 1.0f, 1.0f, 1.0f}, 0x9); // R and A only
    CopyToBuffer(list, 10, BUFFER_POOL_VA);
    rig.Run(std::move(list), {10});

    for (const auto& [x, y] : {std::pair{0u, 0u}, std::pair{31u, 17u}, std::pair{63u, 63u}}) {
        const auto texel = Texel(rig.readback, 0, x, y);
        INFO("texel " << x << "," << y);
        CHECK(Near(texel[0], 255));
        CHECK(Near(texel[1], 102));
        CHECK(Near(texel[2], 153));
        CHECK(Near(texel[3], 255));
    }
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN draw: switching render targets keeps both images' contents", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(MakeTexture(10, 0x25, 0));
    rig.context.CreateTexture(MakeTexture(11, 0x25, 64 * 1024));
    std::vector<u64> list;
    SetTargets(list, {Target(10)});
    ClearColor(list, {1.0f, 0.0f, 0.0f, 1.0f});
    SetTargets(list, {Target(11)});
    ClearColor(list, {0.0f, 0.0f, 1.0f, 1.0f});
    CopyToBuffer(list, 10, BUFFER_POOL_VA);
    CopyToBuffer(list, 11, BUFFER_POOL_VA + 64 * 1024);
    rig.Run(std::move(list), {10, 11});
    const auto a = Texel(rig.readback, 0, 5, 5);
    const auto b = Texel(rig.readback, 64 * 1024, 5, 5);
    CHECK((Near(a[0], 255) && Near(a[2], 0)));
    CHECK((Near(b[0], 0) && Near(b[2], 255)));
    CHECK(rig.context.Stats().ops_failed == 0);
}

// EDEN_NX2_TRY_BARRIER_MERGE: barrier requests with no command between them (the flush of inline
// updates, the transitions and the flush of a pass begin, the clear path's barrier before its own
// pass) record one barrier, and the inline updates that copy nothing record none. Passes, masked
// clears, inline updates and copies must leave the bytes they leave without the switch.
TEST_CASE("NVN draw: merged barriers leave the same pictures and buffers", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr const char* REQUESTS = "memory barriers recorded (ALL_COMMANDS, between replayed ops)";
    constexpr const char* FOLDED =
        "barrier merge: barrier requests folded into one still waiting for its next command (no "
        "command between them)";
    constexpr const char* EMITTED =
        "barrier merge: barrier commands recorded for waiting requests (each stands for one or "
        "more requests)";
    const std::array<u8, 16> update{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6};
    // Clear of the two 16 KiB texture readbacks at the start of the readback pool.
    constexpr u64 UPDATE_AT = BUFFER_POOL_VA + 192 * 1024;
    constexpr u64 COPY_TO = BUFFER_POOL_VA + 192 * 1024 + 4096;
    std::vector<u8> first_result;
    u64 off_requests = 0;
    // 0 off, 1 everything, 4 adjacent requests only.
    for (const u32 merge : {0u, 1u, 4u}) {
        INFO("merge mode " << merge);
        Rig rig{*vulkan, Vulkan::Nvn::ContextOptions{.barrier_merge = merge}};
        rig.context.CreateTexture(MakeTexture(10, 0x25, 0));
        rig.context.CreateTexture(MakeTexture(11, 0x25, 64 * 1024));
        std::vector<u64> list;
        SetTargets(list, {Target(10)});
        ClearColor(list, {1.0f, 0.0f, 0.0f, 1.0f});
        // A masked clear draws through a pass of its own (HostBarrierBeforePass).
        ClearColor(list, {0.0f, 1.0f, 0.0f, 1.0f}, 0x2);
        // Inline updates: one that copies, one at address 0 that names no buffer.
        Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{UPDATE_AT, 16},
             update);
        Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{0, 16}, update);
        SetTargets(list, {Target(11)});
        ClearColor(list, {0.0f, 0.0f, 1.0f, 1.0f});
        ClearColor(list, {1.0f, 0.0f, 1.0f, 1.0f}, 0x5);
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = UPDATE_AT, .dst = COPY_TO, .size = 16});
        CopyToBuffer(list, 10, BUFFER_POOL_VA);
        CopyToBuffer(list, 11, BUFFER_POOL_VA + 64 * 1024);
        rig.Run(std::move(list), {10, 11});
        REQUIRE(rig.context.Stats().ops_failed == 0);

        // The first target is red with the green channel the masked clear wrote; the second is blue
        // with its red channel written by the masked clear.
        const auto a = Texel(rig.readback, 0, 5, 5);
        const auto b = Texel(rig.readback, 64 * 1024, 5, 5);
        CHECK((Near(a[0], 255) && Near(a[1], 255) && Near(a[2], 0)));
        CHECK((Near(b[0], 255) && Near(b[1], 0) && Near(b[2], 255)));
        CHECK(std::memcmp(rig.readback + 192 * 1024 + 4096, update.data(), update.size()) == 0);

        const auto stat = [&](std::string_view reason) {
            std::vector<VC::SinkStat> rows;
            rig.context.CollectStats(rows);
            const auto it = std::ranges::find(rows, reason, &VC::SinkStat::reason);
            return it == rows.end() ? ~u64{0} : it->count;
        };
        const u64 requests = stat(REQUESTS);
        if (merge == 0) {
            first_result.assign(rig.readback, rig.readback + 1_MiB);
            off_requests = requests;
            CHECK(stat(FOLDED) == 0);
            CHECK(stat(EMITTED) == 0);
        } else {
            CHECK(std::memcmp(first_result.data(), rig.readback, first_result.size()) == 0);
            // Never more requests than before (the inline update that copied nothing asked for
            // none); every folded request shared the command of an earlier one.
            CHECK(requests <= off_requests);
            CHECK(stat(FOLDED) != ~u64{0});
            CHECK(stat(EMITTED) <= requests);
        }
    }
}

TEST_CASE("NVN draw: depth and partial stencil clears", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(MakeTexture(20, 0x34, 0));            // DEPTH32F
    rig.context.CreateTexture(MakeTexture(21, 0x31, 128 * 1024));   // STENCIL8
    std::vector<u64> list;
    SetTargets(list, {}, Target(20));
    Push(list, VC::Op::ClearDepthStencil, VC::ClearDepthStencilPayload{0.5f, 1, 0, 0});
    CopyToBuffer(list, 20, BUFFER_POOL_VA);
    SetTargets(list, {}, Target(21));
    Push(list, VC::Op::ClearDepthStencil, VC::ClearDepthStencilPayload{0.0f, 0, 0xAA, 0xFF});
    Push(list, VC::Op::ClearDepthStencil, VC::ClearDepthStencilPayload{0.0f, 0, 0x55, 0x0F});
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = BUFFER_POOL_VA + 64 * 1024;
    copy.texture = {.serial = 21, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, TEX_SIZE, TEX_SIZE, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
    rig.Run(std::move(list), {20, 21});

    float depth;
    std::memcpy(&depth, rig.readback + 100, sizeof(depth));
    CHECK(depth == 0.5f);
    const u8 stencil = rig.readback[64 * 1024 + 700];
    INFO("stencil " << int(stencil));
    CHECK(stencil == 0xA5);
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN draw: inline uniform updates reach memory before later copies", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    std::vector<u8> data(256);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<u8>(i * 5 + 1);
    }
    std::vector<u64> list;
    Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{BUFFER_POOL_VA + 4096, 256},
         data);
    Push(list, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = BUFFER_POOL_VA + 4096, .dst = BUFFER_POOL_VA + 8192,
                                       .size = 256});
    rig.Run(std::move(list), {});
    CHECK(std::memcmp(rig.readback + 8192, data.data(), data.size()) == 0);
    CHECK(rig.context.Stats().ops_failed == 0);
}

// Several updates of the same bytes in one flush: the guest bytes take the last one, and no two
// copies to the same bytes go without a barrier between them (sync validation WRITE_AFTER_WRITE).
TEST_CASE("NVN draw: inline uniform updates of the same bytes keep the last one", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    const auto filled = [](size_t size, u8 seed) {
        std::vector<u8> bytes(size);
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<u8>(i * 3 + seed);
        }
        return bytes;
    };
    const std::vector<u8> first = filled(64, 1);
    const std::vector<u8> second = filled(64, 100);
    const std::vector<u8> part = filled(16, 200);
    constexpr u64 AT = BUFFER_POOL_VA + 4096;
    std::vector<u64> list;
    // `first` is rewritten whole by `second`: no copy. `part` overlaps `second` partly: it is
    // copied behind a barrier.
    Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{AT, 64}, first);
    Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{AT, 64}, second);
    Push(list, VC::Op::UpdateUniformBuffer, VC::UpdateUniformBufferPayload{AT + 8, 16}, part);
    Push(list, VC::Op::CopyBufferToBuffer,
         VC::CopyBufferToBufferPayload{.src = AT, .dst = BUFFER_POOL_VA + 8192, .size = 64});
    rig.Run(std::move(list), {});

    std::vector<u8> expected = second;
    std::memcpy(expected.data() + 8, part.data(), part.size());
    CHECK(std::memcmp(rig.readback + 8192, expected.data(), expected.size()) == 0);
    CHECK(rig.context.Stats().ops_failed == 0);
    const auto stat = [&](std::string_view reason) {
        std::vector<VC::SinkStat> rows;
        rig.context.CollectStats(rows);
        const auto it = std::ranges::find(rows, reason, &VC::SinkStat::reason);
        return it == rows.end() ? ~u64{0} : it->count;
    };
    CHECK(stat("inline_uniform_updates: rewritten whole by a later one (no copy)") == 1);
    CHECK(stat("inline_uniform_updates: transfer barriers between overlapping copies") == 1);
}

namespace {

constexpr u32 SLICES = 4;

VC::TextureDesc Make3DTexture(u64 serial, u64 pool_offset) {
    VC::TextureDesc desc = MakeTexture(serial, 0x25, pool_offset);
    desc.target = 2; // 3D
    desc.depth = SLICES;
    return desc;
}

/// A render target of `count` slices of a 3D texture from `slice` (a view with set_layers: without
/// one the attachment is every slice from the base layer).
VC::RenderTargetRef SliceTarget(u64 serial, u32 slice, u32 count = 1) {
    VC::RenderTargetRef ref{};
    ref.texture = {.serial = serial, .level = 0, .layer = slice};
    ref.view.set_layers = 1;
    ref.view.min_layer = static_cast<u16>(slice);
    ref.view.nlayers = static_cast<u16>(count);
    return ref;
}

void CopySliceToBuffer(std::vector<u64>& list, u64 texture, u32 slice, u64 buffer) {
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = buffer;
    copy.texture = {.serial = texture, .level = 0, .layer = 0};
    copy.region = {0, 0, static_cast<s32>(slice), TEX_SIZE, TEX_SIZE, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
}

} // namespace

TEST_CASE("NVN draw: slices of a 3D texture are render targets", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(Make3DTexture(30, 0));
    std::vector<u64> list;
    SetTargets(list, {SliceTarget(30, 1)});
    ClearColor(list, {1.0f, 0.0f, 0.0f, 1.0f});
    SetTargets(list, {SliceTarget(30, 2)});
    ClearColor(list, {0.0f, 1.0f, 0.0f, 1.0f});
    // A view of a single slice (set_layers, count 1) is an ordinary 2D attachment of that slice.
    SetTargets(list, {SliceTarget(30, 3, 1)});
    ClearColor(list, {0.0f, 0.0f, 1.0f, 1.0f});
    for (u32 slice = 1; slice <= 3; ++slice) {
        CopySliceToBuffer(list, 30, slice, BUFFER_POOL_VA + slice * 64 * 1024);
    }
    rig.Run(std::move(list), {30});

    const auto s1 = Texel(rig.readback, 64 * 1024, 10, 10);
    const auto s2 = Texel(rig.readback, 2 * 64 * 1024, 10, 10);
    const auto s3 = Texel(rig.readback, 3 * 64 * 1024, 63, 63);
    CHECK((Near(s1[0], 255) && Near(s1[1], 0) && Near(s1[2], 0)));
    CHECK((Near(s2[0], 0) && Near(s2[1], 255) && Near(s2[2], 0)));
    CHECK((Near(s3[0], 0) && Near(s3[1], 0) && Near(s3[2], 255)));
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN draw: a slice range of a 3D texture clears every slice of the range", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(Make3DTexture(31, 0));
    std::vector<u64> list;
    SetTargets(list, {SliceTarget(31, 0, SLICES)});
    ClearColor(list, {0.0f, 0.0f, 0.0f, 1.0f});
    SetTargets(list, {SliceTarget(31, 1, 2)});
    ClearColor(list, {1.0f, 1.0f, 0.0f, 1.0f});
    for (u32 slice = 0; slice < SLICES; ++slice) {
        CopySliceToBuffer(list, 31, slice, BUFFER_POOL_VA + slice * 64 * 1024);
    }
    rig.Run(std::move(list), {31});
    for (u32 slice = 0; slice < SLICES; ++slice) {
        const auto texel = Texel(rig.readback, slice * 64 * 1024, 7, 7);
        INFO("slice " << slice);
        const bool in_range = slice == 1 || slice == 2;
        CHECK(Near(texel[0], in_range ? 255 : 0));
        CHECK(Near(texel[1], in_range ? 255 : 0));
        CHECK(Near(texel[2], 0));
    }
    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN draw: a masked clear on a 3D texture slice only touches that slice", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan};
    rig.context.CreateTexture(Make3DTexture(32, 0));
    std::vector<u64> list;
    SetTargets(list, {SliceTarget(32, 0, SLICES)});
    ClearColor(list, {0.2f, 0.4f, 0.6f, 0.8f});
    SetTargets(list, {SliceTarget(32, 2)});
    ClearColor(list, {1.0f, 1.0f, 1.0f, 1.0f}, 0x9); // R and A only, slice 2 only
    for (u32 slice = 0; slice < SLICES; ++slice) {
        CopySliceToBuffer(list, 32, slice, BUFFER_POOL_VA + slice * 64 * 1024);
    }
    rig.Run(std::move(list), {32});
    for (u32 slice = 0; slice < SLICES; ++slice) {
        const auto texel = Texel(rig.readback, slice * 64 * 1024, 30, 30);
        INFO("slice " << slice);
        CHECK(Near(texel[0], slice == 2 ? 255 : 51));
        CHECK(Near(texel[1], 102));
        CHECK(Near(texel[2], 153));
        CHECK(Near(texel[3], slice == 2 ? 255 : 204));
    }
    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}

TEST_CASE("NVN draw: viewport and winding rules of the window origin", "[nvn]") {
    using Vulkan::Nvn::ScissorTopRow;
    using Vulkan::Nvn::ToVulkanViewport;
    using Vulkan::Nvn::ViewportRect;
    using Vulkan::Nvn::VulkanFrontFaceCcw;

    // The full-surface viewport is negative-height for both origins (clip +y lands on row 0).
    const ViewportRect full{0.0f, 0.0f, 1440.0f, 810.0f};
    for (const bool lower_left : {false, true}) {
        const ViewportRect v = ToVulkanViewport(full, 810.0f, lower_left);
        CHECK(v.x == 0.0f);
        CHECK(v.y == 810.0f);
        CHECK(v.width == 1440.0f);
        CHECK(v.height == -810.0f);
    }

    // A sub-rectangle: UPPER_LEFT counts rows from the top, LOWER_LEFT from the bottom.
    const ViewportRect sub{10.0f, 100.0f, 200.0f, 300.0f};
    const ViewportRect upper = ToVulkanViewport(sub, 810.0f, false);
    CHECK(upper.y == 400.0f); // rows 100..400, clip +y on row 100
    CHECK(upper.height == -300.0f);
    const ViewportRect lower = ToVulkanViewport(sub, 810.0f, true);
    CHECK(lower.y == 710.0f); // rows 410..710 (810 - 100 from the bottom edge)
    CHECK(lower.height == -300.0f);

    // The experiment knob restores the positive-height viewport.
    const ViewportRect inverted = ToVulkanViewport(sub, 810.0f, false, true);
    CHECK(inverted.y == 100.0f);
    CHECK(inverted.height == 300.0f);

    CHECK(ScissorTopRow(100, 300, 810, false) == 100);
    CHECK(ScissorTopRow(100, 300, 810, true) == 410);
    CHECK(ScissorTopRow(0, 810, 810, true) == 0);

    // The NVN winding is the Vulkan one.
    CHECK(VulkanFrontFaceCcw(true));
    CHECK_FALSE(VulkanFrontFaceCcw(false));
    CHECK_FALSE(VulkanFrontFaceCcw(true, true));
}

TEST_CASE("NVN draw: block views of compressed textures alias one level in blocks", "[nvn]") {
    using namespace Vulkan::Nvn;
    constexpr s32 BC4 = 0x49;
    constexpr s32 BC7 = 0x4D;
    constexpr s32 RGBA8 = 0x25;
    constexpr s32 RGBA16_UINT = 0x2C;
    constexpr s32 RGBA32_UINT = 0x2F;
    CHECK(IsBlockView(BC4, RGBA16_UINT));
    CHECK(IsBlockView(BC7, RGBA32_UINT));
    CHECK_FALSE(IsBlockView(BC4, RGBA8));         // 4 bytes, not a BC4 block
    CHECK_FALSE(IsBlockView(RGBA8, RGBA16_UINT)); // not compressed
    CHECK_FALSE(IsBlockView(BC4, BC4));

    // OoT encodes 1040x1040-ish BC4 textures level by level (viewports 260, 130, 65, 32 ...).
    const TextureLayoutDesc base{.target = static_cast<s32>(TexTarget::Dim2DArray),
                                 .format = BC4,
                                 .width = 1040,
                                 .height = 1040,
                                 .depth = 3,
                                 .levels = 9};
    const TextureLayout base_layout = ComputeTextureLayout(base);
    for (u32 layer = 0; layer < 3; ++layer) {
        for (u32 level = 0; level < 9; ++level) {
            INFO("layer " << layer << " level " << level);
            const auto alias = BlockViewAliasOf(base, RGBA16_UINT, level, layer);
            REQUIRE(alias.has_value());
            const u32 texels = (std::max)(1040u >> level, 1u);
            CHECK(alias->desc.width == (texels + 3) / 4);
            CHECK(alias->desc.height == (texels + 3) / 4);
            CHECK(alias->desc.levels == 1);
            CHECK(alias->desc.format == RGBA16_UINT);
            CHECK(alias->offset == layer * base_layout.layer_size + TextureLevelOffset(base, level));
            // The same GOB columns and block height as the level: the same bytes per block.
            u32 base_h = 0, base_d = 0, alias_h = 0, alias_d = 0;
            TextureLevelTiles(base, level, base_h, base_d);
            TextureLevelTiles(alias->desc, 0, alias_h, alias_d);
            CHECK(alias_h == base_h);
            CHECK(TextureLevelGobs(alias->desc, 0) == TextureLevelGobs(base, level));
            const u64 next = level + 1 < 9 ? TextureLevelOffset(base, level + 1)
                                           : base_layout.layer_size;
            CHECK(ComputeTextureLayout(alias->desc).size <= next - TextureLevelOffset(base, level));
        }
    }
    CHECK_FALSE(BlockViewAliasOf(base, RGBA8, 0, 0).has_value());
    CHECK_FALSE(BlockViewAliasOf(base, RGBA16_UINT, 9, 0).has_value()); // no such level
    CHECK_FALSE(BlockViewAliasOf(base, RGBA16_UINT, 0, 3).has_value()); // no such layer

    // OoT's BC4 volumes (64x64x64, one level, encoded slice by slice) and a mip chain of one: a 3D
    // level's alias is 3D with all its slices (block-linear slices interleave in blocks of
    // 2^tile_d), laid out like the level.
    for (const u32 levels : {1u, 4u}) {
        const TextureLayoutDesc volume{.target = static_cast<s32>(TexTarget::Dim3D),
                                       .format = BC4,
                                       .width = 64,
                                       .height = 64,
                                       .depth = 64,
                                       .levels = levels};
        const TextureLayout volume_layout = ComputeTextureLayout(volume);
        REQUIRE(volume_layout.tile_d > 0); // slices interleave: no slice is a 2D image of its own
        for (u32 level = 0; level < levels; ++level) {
            INFO("levels " << levels << " level " << level);
            const auto alias = BlockViewAliasOf(volume, RGBA16_UINT, level, 0);
            REQUIRE(alias.has_value());
            CHECK(alias->desc.target == static_cast<s32>(TexTarget::Dim3D));
            CHECK(alias->desc.width == (64u >> level) / 4);
            CHECK(alias->desc.height == (64u >> level) / 4);
            CHECK(alias->desc.depth == 64u >> level);
            CHECK(alias->desc.levels == 1);
            CHECK(alias->offset == TextureLevelOffset(volume, level));
            u32 base_h = 0, base_d = 0, alias_h = 0, alias_d = 0;
            TextureLevelTiles(volume, level, base_h, base_d);
            TextureLevelTiles(alias->desc, 0, alias_h, alias_d);
            CHECK(alias_h == base_h);
            CHECK(alias_d == base_d);
            CHECK(TextureLevelGobs(alias->desc, 0) == TextureLevelGobs(volume, level));
            const u64 next = level + 1 < levels ? TextureLevelOffset(volume, level + 1)
                                                : volume_layout.layer_size;
            CHECK(ComputeTextureLayout(alias->desc).size == next - TextureLevelOffset(volume, level));
        }
        CHECK_FALSE(BlockViewAliasOf(volume, RGBA16_UINT, 0, 1).has_value()); // slices are layer 0
    }
    TextureLayoutDesc line = base;
    line.target = static_cast<s32>(TexTarget::Dim1D);
    CHECK_FALSE(BlockViewAliasOf(line, RGBA16_UINT, 0, 0).has_value());
}

namespace {

void ClearColorUint(std::vector<u64>& list, std::array<u32, 4> color) {
    VC::ClearColorPayload clear{};
    clear.index = 0;
    clear.mask = 0xF;
    clear.is_integer = 1;
    std::memcpy(clear.color, color.data(), sizeof(clear.color));
    Push(list, VC::Op::ClearColor, clear);
}

VC::RenderTargetRef BlockTarget(u64 serial, u32 level, s32 view_format) {
    VC::RenderTargetRef ref = Target(serial);
    ref.texture.level = level;
    ref.view.set_format = 1;
    ref.view.format = static_cast<u16>(view_format);
    return ref;
}

/// The four 16-bit words of BC4 block `index` of a tight block readback.
std::array<u16, 4> Block(const u8* readback, u64 base, u32 index) {
    std::array<u16, 4> words{};
    std::memcpy(words.data(), readback + base + u64{index} * 8, sizeof(words));
    return words;
}

} // namespace

TEST_CASE("NVN draw: rendering through a block view writes the compressed texture's blocks",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsOptimalBcnSupported()) {
        SKIP("the device has no native BC4 images to read the blocks of");
    }
    constexpr s32 BC4 = 0x49;
    constexpr s32 RGBA16_UINT = 0x2C;
    Rig rig{*vulkan};
    VC::TextureDesc bc4 = MakeTexture(20, BC4, 0);
    bc4.levels = 2; // 64x64 and 32x32 texels: 16x16 and 8x8 blocks
    rig.context.CreateTexture(bc4);

    constexpr std::array<u32, 4> A{0x1111, 0x2222, 0x3333, 0x4444};
    constexpr std::array<u32, 4> B{0xA1B2, 0xC3D4, 0xE5F6, 0x0718};
    constexpr std::array<u32, 4> C{0x0F0E, 0x0D0C, 0x0B0A, 0x0908};
    std::vector<u64> list;
    // Level 0 as 16x16 RGBA16_UINT texels (one per block): A, then B in blocks (4..7, 2..5).
    SetTargets(list, {BlockTarget(20, 0, RGBA16_UINT)});
    ClearColorUint(list, A);
    Push(list, VC::Op::SetScissors, VC::SetScissorsPayload{0, 1},
         Bytes(VC::ScissorRect{4, 2, 4, 4}));
    ClearColorUint(list, B);
    // Level 1 as 8x8 texels: C.
    Push(list, VC::Op::SetScissors, VC::SetScissorsPayload{0, 1},
         Bytes(VC::ScissorRect{0, 0, 8, 8}));
    SetTargets(list, {BlockTarget(20, 1, RGBA16_UINT)});
    ClearColorUint(list, C);
    // The BC4 texture itself, block rows tight: level 0 at 0, level 1 at 64 KiB.
    for (const u32 level : {0u, 1u}) {
        VC::CopyBufferTexturePayload copy{};
        copy.buffer = BUFFER_POOL_VA + level * 64 * 1024;
        copy.texture = {.serial = 20, .level = level, .layer = 0};
        const s32 size = static_cast<s32>(TEX_SIZE >> level);
        copy.region = {0, 0, 0, size, size, 1};
        Push(list, VC::Op::CopyTextureToBuffer, copy);
    }
    rig.Run(std::move(list), {20});

    const auto words = [](const std::array<u32, 4>& c) {
        return std::array<u16, 4>{static_cast<u16>(c[0]), static_cast<u16>(c[1]),
                                  static_cast<u16>(c[2]), static_cast<u16>(c[3])};
    };
    for (u32 y = 0; y < 16; ++y) {
        for (u32 x = 0; x < 16; ++x) {
            INFO("level 0 block " << x << "," << y);
            const bool inside = x >= 4 && x < 8 && y >= 2 && y < 6;
            CHECK(Block(rig.readback, 0, y * 16 + x) == words(inside ? B : A));
        }
    }
    for (u32 index = 0; index < 64; ++index) {
        INFO("level 1 block " << index);
        CHECK(Block(rig.readback, 64 * 1024, index) == words(C));
    }
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}

TEST_CASE("NVN draw: rendering through a block view of a 3D texture writes its slices' blocks",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsOptimalBcnSupported()) {
        SKIP("the device has no native BC4 images to read the blocks of");
    }
    constexpr s32 BC4 = 0x49;
    constexpr s32 RGBA16_UINT = 0x2C;
    constexpr u32 DEPTH = 64;
    Rig rig{*vulkan};
    // OoT's BC4 volume: 64x64x64 texels, 16x16 blocks per slice, encoded slice by slice.
    VC::TextureDesc bc4 = MakeTexture(21, BC4, 0);
    bc4.target = 2; // 3D
    bc4.depth = DEPTH;
    rig.context.CreateTexture(bc4);
    const auto block_target = [&](u32 slice, u32 count) {
        VC::RenderTargetRef ref = SliceTarget(21, slice, count);
        ref.view.set_format = 1;
        ref.view.format = static_cast<u16>(RGBA16_UINT);
        return ref;
    };

    constexpr std::array<u32, 4> Z{0x0101, 0x0202, 0x0303, 0x0404};
    constexpr std::array<u32, 4> A{0x1111, 0x2222, 0x3333, 0x4444};
    constexpr std::array<u32, 4> B{0xA1B2, 0xC3D4, 0xE5F6, 0x0718};
    constexpr std::array<u32, 4> C{0x0F0E, 0x0D0C, 0x0B0A, 0x0908};
    constexpr std::array<u32, 4> D{0x5A5A, 0x6B6B, 0x7C7C, 0x8D8D};
    constexpr std::array<u32, 4> E{0xFFFF, 0x0000, 0xFFFF, 0x0001};
    constexpr std::array<u32, 4> F{0x1234, 0x5678, 0x9ABC, 0xDEF0};
    std::vector<u64> list;
    // Every slice (a range target), then single slices as the game sets them, in both GOB-depth
    // blocks; slice 63 with a scissored rectangle; then a range of two slices.
    SetTargets(list, {block_target(0, DEPTH)});
    ClearColorUint(list, Z);
    SetTargets(list, {block_target(0, 1)});
    ClearColorUint(list, A);
    SetTargets(list, {block_target(1, 1)});
    ClearColorUint(list, B);
    SetTargets(list, {block_target(33, 1)});
    ClearColorUint(list, C);
    SetTargets(list, {block_target(63, 1)});
    ClearColorUint(list, D);
    Push(list, VC::Op::SetScissors, VC::SetScissorsPayload{0, 1},
         Bytes(VC::ScissorRect{4, 2, 4, 4}));
    ClearColorUint(list, E);
    Push(list, VC::Op::SetScissors, VC::SetScissorsPayload{0, 1},
         Bytes(VC::ScissorRect{0, 0, 16, 16}));
    SetTargets(list, {block_target(40, 2)});
    ClearColorUint(list, F);
    // The BC4 slices themselves, block rows tight, 4 KiB apart.
    constexpr std::array<u32, 8> SLICES_READ{0, 1, 2, 33, 40, 41, 42, 63};
    for (u32 i = 0; i < SLICES_READ.size(); ++i) {
        VC::CopyBufferTexturePayload copy{};
        copy.buffer = BUFFER_POOL_VA + i * 4096;
        copy.texture = {.serial = 21, .level = 0, .layer = 0};
        copy.region = {0, 0, static_cast<s32>(SLICES_READ[i]), TEX_SIZE, TEX_SIZE, 1};
        Push(list, VC::Op::CopyTextureToBuffer, copy);
    }
    rig.Run(std::move(list), {21});

    const auto words = [](const std::array<u32, 4>& c) {
        return std::array<u16, 4>{static_cast<u16>(c[0]), static_cast<u16>(c[1]),
                                  static_cast<u16>(c[2]), static_cast<u16>(c[3])};
    };
    const std::array<std::array<u32, 4>, 8> expected{A, B, Z, C, F, F, Z, D};
    for (u32 i = 0; i < SLICES_READ.size(); ++i) {
        for (u32 y = 0; y < 16; ++y) {
            for (u32 x = 0; x < 16; ++x) {
                INFO("slice " << SLICES_READ[i] << " block " << x << "," << y);
                const bool scissored = SLICES_READ[i] == 63 && x >= 4 && x < 8 && y >= 2 && y < 6;
                CHECK(Block(rig.readback, i * 4096, y * 16 + x) ==
                      words(scissored ? E : expected[i]));
            }
        }
    }
    const auto stats = rig.context.Stats();
    CHECK(stats.ops_failed == 0);
    CHECK(stats.ops_unsupported == 0);
}
