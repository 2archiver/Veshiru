// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the NVN replay's memory work: block-linear uploads, ASTC / BCn decoding, CPU write
// coherency, overlapping textures, virtual pools and the ReadTexels flush. The CPU-only ones run
// everywhere; the ones that need a GPU create a headless device and SKIP when there is none.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/cityhash.h"
#include "common/diag_image.h"
#include "common/game_presets.h"
#include "common/literals.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_pool_mirror.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace {

namespace VC = VideoCommon::Nvn;
namespace Nvn = Vulkan::Nvn;
using namespace Common::Literals;
using namespace std::chrono_literals;
using VideoCore::Surface::PixelFormat;

constexpr u32 NVN_FORMAT_RGBA8 = 0x25;
constexpr u32 NVN_FORMAT_R32UI = 0x0B;
constexpr u32 NVN_FORMAT_ASTC_4X4 = 0x79;
constexpr u32 NVN_FORMAT_BC1 = 0x42;
constexpr u32 NVN_FORMAT_D24S8 = 0x35;
constexpr u32 POOL_CPU_CACHED = 4;
constexpr u32 POOL_VIRTUAL = 0x200;
constexpr u32 POOL_CPU_NO_ACCESS = 1;

TEST_CASE("Mirror page ownership requires complete coverage of the actual page",
          "[nvn][mirror][mirror-owned]") {
    using Nvn::ClassifyMirrorWrites;
    using Nvn::MirrorWriteCoverage;
    const std::array<std::pair<u64, u64>, 2> full{{{0, 8192}, {16384, 32768}}};
    CHECK(ClassifyMirrorWrites(0, 4096, full) == MirrorWriteCoverage::Full);
    CHECK(ClassifyMirrorWrites(4096, 8192, full) == MirrorWriteCoverage::Full);
    CHECK(ClassifyMirrorWrites(8192, 12288, full) == MirrorWriteCoverage::None);
    CHECK(ClassifyMirrorWrites(4096, 12288, full) == MirrorWriteCoverage::Partial);
    CHECK(ClassifyMirrorWrites(32768, 36864, full) == MirrorWriteCoverage::None);
    CHECK(ClassifyMirrorWrites(8192, 8192, full) == MirrorWriteCoverage::None);
    CHECK(ClassifyMirrorWrites(16384, 16512, full) == MirrorWriteCoverage::Full);
    const std::array<std::pair<u64, u64>, 2> hole{{{4096, 5000}, {6000, 8192}}};
    CHECK(ClassifyMirrorWrites(4096, 8192, hole) == MirrorWriteCoverage::Partial);
    CHECK(ClassifyMirrorWrites(5000, 6000, hole) == MirrorWriteCoverage::None);
    CHECK(ClassifyMirrorWrites(4096, 8192, {}) == MirrorWriteCoverage::None);
}

// --- helpers ----------------------------------------------------------------------------------

/// Storage of a plain texture, the way CreateTextureNow builds it (without a VkImage).
Nvn::Storage MakeStorage(s32 format, u32 width, u32 height, u32 levels, PixelFormat pixel_format,
                         VkFormat vk_format, bool transcode = false) {
    Nvn::Storage s;
    s.desc = {.target = 1, .format = format, .flags = 0, .samples = 0, .width = width,
              .height = height, .depth = 1, .levels = levels, .stride = 0, .pkg_block_h = 0};
    s.layout = Nvn::ComputeTextureLayout(s.desc);
    s.host.format = vk_format;
    s.host.pixel_format = pixel_format;
    s.host.needs_transcode = transcode;
    s.type = VK_IMAGE_TYPE_2D;
    s.extent = {width, height, 1};
    s.levels = levels;
    s.layers = 1;
    return s;
}

/// Linear texels -> the storage's block-linear guest bytes (level 0 of a one-level texture).
std::vector<u8> SwizzleLevel0(const Nvn::Storage& s, std::span<const u8> linear, u32 bytes,
                              u32 cols, u32 rows) {
    u32 tile_h = 0;
    u32 tile_d = 0;
    Nvn::TextureLevelTiles(s.desc, 0, tile_h, tile_d);
    std::vector<u8> guest(static_cast<size_t>(s.layout.size), 0);
    Tegra::Texture::SwizzleTexture(guest, linear, bytes, cols, rows, 1, tile_h, tile_d, 0, true);
    return guest;
}

std::vector<u8> Gradient(u32 width, u32 height, u8 seed) {
    std::vector<u8> texels(static_cast<size_t>(width) * height * 4);
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            u8* p = texels.data() + (static_cast<size_t>(y) * width + x) * 4;
            p[0] = static_cast<u8>(x * 7 + seed);
            p[1] = static_cast<u8>(y * 5 + seed * 3);
            p[2] = static_cast<u8>(x ^ y);
            p[3] = static_cast<u8>(255 - seed);
        }
    }
    return texels;
}

/// An ASTC LDR void-extent block: one constant colour for the whole block (16-bit channels).
std::array<u8, 16> VoidBlock(u16 r, u16 g, u16 b, u16 a) {
    std::array<u8, 16> block{0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    const u16 channels[4] = {r, g, b, a};
    for (size_t i = 0; i < 4; ++i) {
        block[8 + i * 2] = static_cast<u8>(channels[i] & 0xFF);
        block[9 + i * 2] = static_cast<u8>(channels[i] >> 8);
    }
    return block;
}

bool Close(u8 value, int expected, int tolerance = 1) {
    return std::abs(int(value) - expected) <= tolerance;
}

// --- GPU rig ----------------------------------------------------------------------------------

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
        u8* const data = region.data;
        std::scoped_lock lock{regions_mutex};
        regions.push_back(std::move(region));
        return data;
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
        // A test may add a region while the replay reads the map.
        std::scoped_lock lock{regions_mutex};
        for (Region& region : regions) {
            if (va >= region.base && va - region.base <= region.size &&
                size <= region.size - (va - region.base)) {
                return region.data + (va - region.base);
            }
        }
        return nullptr;
    }

    // Regions never move: the vector holds them by value but `data` points into `raw`'s heap
    // block, which survives a vector reallocation (std::vector<u8> moves keep their buffer).
    std::vector<Region> regions;
    std::mutex regions_mutex;
};

template <typename T>
void Push(std::vector<u64>& list, VC::Op op, const T& payload, u16 flags = 0) {
    constexpr size_t words = 1 + (sizeof(T) + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(op);
    header->flags = flags;
    header->size = static_cast<u32>(words * 8);
    std::memcpy(&list[at + 1], &payload, sizeof(T));
}

void PushEnd(std::vector<u64>& list) {
    const size_t at = list.size();
    list.resize(at + 1, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(VC::Op::End);
    header->flags = 0;
    header->size = 8;
}

VC::Submission MakeSubmission(u64 seq, std::vector<u64> list) {
    VC::Submission submission{};
    submission.seq = seq;
    submission.queue = 0;
    submission.lists.push_back(std::move(list));
    return submission;
}

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
            *result->instance, Vulkan::vk::PhysicalDevice(devices[0], *result->dld),
            VK_NULL_HANDLE, *result->dld);
        return result;
    } catch (const std::exception&) {
        return nullptr;
    }
}

VC::TextureDesc MakeTexture(u64 serial, u64 pool_serial, u64 pool_offset, u32 format, u32 width,
                            u32 height) {
    VC::TextureDesc desc{};
    desc.target = 1;
    desc.format = static_cast<s32>(format);
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.levels = 1;
    desc.serial = serial;
    desc.pool_serial = pool_serial;
    desc.pool_offset = pool_offset;
    return desc;
}

VC::CopyBufferTexturePayload CopyToBuffer(u64 address, u64 texture, u32 width, u32 height) {
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = address;
    copy.texture = {.serial = texture, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, static_cast<s32>(width), static_cast<s32>(height), 1};
    return copy;
}

VC::ClearTexturePayload ClearColor(u64 texture, u32 width, u32 height, float r, float g, float b,
                                   float a) {
    VC::ClearTexturePayload clear{};
    clear.texture = {.serial = texture, .level = 0, .layer = 0};
    clear.region = {0, 0, 0, static_cast<s32>(width), static_cast<s32>(height), 1};
    clear.layer_count = 1;
    const float color[4] = {r, g, b, a};
    std::memcpy(clear.color, color, sizeof(color));
    return clear;
}

/// Submits one list and waits for it.
void Run(Nvn::Context& context, u64& seq, std::vector<u64> list) {
    PushEnd(list);
    ++seq;
    context.Submit(MakeSubmission(seq, std::move(list)));
    REQUIRE(context.Wait(seq, 30s));
}

} // Anonymous namespace

// ---------------------------------------------------------------------------------------------
// CPU only

TEST_CASE("MKW batched texture signatures keep every sampled line", "[nvn][mkw]") {
    const bool batch = GENERATE(false, true);
    const u64 size = GENERATE(0ULL, 1ULL, 4096ULL, 16384ULL, 16385ULL, 65537ULL, 1048579ULL);
    std::vector<u8> bytes(size + 64);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<u8>(i * 31 + (i >> 8));
    }
    const u64 signature = Nvn::SampleGuestBytes(bytes.data(), size, batch);
    u64 reads = 0;
    const u64 gathered = Nvn::SampleGuestBytesBy(
        size,
        [&](u64 offset, u64 count, u8* destination) {
            REQUIRE(offset + count <= size);
            ++reads;
            std::memcpy(destination, bytes.data() + offset, count);
        },
        batch);
    CHECK(gathered == signature);
    CHECK(reads == (size <= 16384 ? 1u : 256u));
    if (size <= 16384) {
        CHECK(signature == Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), size));
        if (size != 0) {
            bytes[size - 1] ^= 1;
            CHECK(Nvn::SampleGuestBytes(bytes.data(), size, batch) != signature);
        }
        return;
    }
    const u64 last_line = (size - 64) / 64;
    for (u64 line = 0; line < 256; ++line) {
        const u64 offset = last_line * line / 255 * 64;
        for (u64 edge : {0ULL, 63ULL}) {
            bytes[offset + edge] ^= 1;
            CHECK(Nvn::SampleGuestBytes(bytes.data(), size, batch) != signature);
            bytes[offset + edge] ^= 1;
        }
    }
}

TEST_CASE("nvn memory: a block-linear storage unswizzles in Switch 2 GOB order", "[nvn]") {
    Nvn::Storage s = MakeStorage(NVN_FORMAT_RGBA8, 48, 40, 1, PixelFormat::A8B8G8R8_UNORM,
                                 VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const std::vector<u8> texels = Gradient(48, 40, 3);
    const std::vector<u8> guest = SwizzleLevel0(s, texels, 4, 48, 40);

    const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, guest.size());
    REQUIRE(plan.unsupported == nullptr);
    REQUIRE(plan.pieces.size() == 1);
    REQUIRE(plan.staging_size == texels.size());
    std::vector<u8> staging(plan.staging_size);
    Nvn::FillStorageUpload(s, plan, guest.data(), guest.size(), staging.data());
    REQUIRE(staging == texels);

    // The same texels swizzled the pre-Switch 2 way land elsewhere: native_tiling is in use.
    std::vector<u8> old_order(guest.size(), 0);
    u32 tile_h = 0;
    u32 tile_d = 0;
    Nvn::TextureLevelTiles(s.desc, 0, tile_h, tile_d);
    Tegra::Texture::SwizzleTexture(old_order, texels, 4, 48, 40, 1, tile_h, tile_d, 0, false);
    REQUIRE(old_order != guest);
    std::vector<u8> wrong(plan.staging_size);
    Nvn::FillStorageUpload(s, plan, old_order.data(), old_order.size(), wrong.data());
    REQUIRE(wrong != texels);
}

TEST_CASE("nvn memory: ranges cover a span only without a gap", "[nvn]") {
    using Ranges = std::vector<std::pair<u64, u64>>;
    const auto covers = [](const Ranges& ranges, u64 begin, u64 end) {
        return Nvn::RangesCover(ranges, begin, end);
    };
    REQUIRE(covers({{0, 100}}, 0, 100));
    REQUIRE(covers({{0, 60}, {60, 100}}, 0, 100));
    REQUIRE(covers({{0, 70}, {50, 120}}, 0, 100));
    REQUIRE(covers({{0, 300}, {10, 20}}, 0, 100)); // a later range inside an earlier one
    REQUIRE(covers({{10, 20}}, 15, 15));           // nothing to cover
    REQUIRE_FALSE(covers({}, 0, 100));
    REQUIRE_FALSE(covers({{0, 99}}, 0, 100));
    REQUIRE_FALSE(covers({{1, 100}}, 0, 100));
    REQUIRE_FALSE(covers({{0, 40}, {41, 100}}, 0, 100)); // one byte missing between them
}

TEST_CASE("nvn memory: mip chains and layers get one piece each", "[nvn]") {
    Nvn::Storage s = MakeStorage(NVN_FORMAT_RGBA8, 64, 64, 4, PixelFormat::A8B8G8R8_UNORM,
                                 VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    s.layers = 1;
    const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, s.layout.size);
    REQUIRE(plan.unsupported == nullptr);
    REQUIRE(plan.pieces.size() == 4);
    u64 expected = 0;
    for (u32 level = 0; level < 4; ++level) {
        const Nvn::UploadPiece& piece = plan.pieces[level];
        REQUIRE(piece.level == level);
        REQUIRE(piece.width == (64u >> level));
        REQUIRE(piece.guest_offset == Nvn::TextureLevelOffset(s.desc, level));
        REQUIRE(piece.staging_offset % 64 == 0);
        REQUIRE(piece.staging_offset >= expected);
        expected = piece.staging_offset + piece.staging_size;
    }
    // A storage that stops after level 1 only uploads what it holds.
    const Nvn::UploadPlan partial =
        Nvn::PlanStorageUpload(s, Nvn::TextureLevelOffset(s.desc, 2));
    REQUIRE(partial.pieces.size() == 2);
}

TEST_CASE("nvn memory: ASTC blocks decode to RGBA8 on upload", "[nvn]") {
    // 8x8 texels, 4x4 blocks: 2x2 blocks of 16 bytes, each a constant colour.
    Nvn::Storage s = MakeStorage(NVN_FORMAT_ASTC_4X4, 8, 8, 1, PixelFormat::ASTC_2D_4X4_UNORM,
                                 VK_FORMAT_A8B8G8R8_UNORM_PACK32, true);
    REQUIRE(Nvn::FormatBlockInfo(NVN_FORMAT_ASTC_4X4).bytes == 16);
    const std::array<std::array<u8, 16>, 4> blocks{
        VoidBlock(0xFFFF, 0x8000, 0x0000, 0xFFFF), VoidBlock(0x0000, 0xFFFF, 0x0000, 0xFFFF),
        VoidBlock(0x0000, 0x0000, 0xFFFF, 0xFFFF), VoidBlock(0xFFFF, 0xFFFF, 0xFFFF, 0x8000),
    };
    std::vector<u8> linear;
    for (const auto& block : blocks) {
        linear.insert(linear.end(), block.begin(), block.end());
    }
    const std::vector<u8> guest = SwizzleLevel0(s, linear, 16, 2, 2);

    const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, guest.size());
    REQUIRE(plan.unsupported == nullptr);
    REQUIRE(plan.decode == Nvn::UploadPlan::Decode::Astc);
    REQUIRE(plan.staging_size == 8 * 8 * 4);
    std::vector<u8> staging(plan.staging_size);
    Nvn::UploadStats stats;
    Nvn::FillStorageUpload(s, plan, guest.data(), guest.size(), staging.data(), &stats);
    REQUIRE(stats.decode_bytes == plan.staging_size);

    const auto pixel = [&](u32 x, u32 y) { return staging.data() + (y * 8 + x) * 4; };
    const u8* p = pixel(1, 1); // block 0
    CHECK((Close(p[0], 255) && Close(p[1], 128) && Close(p[2], 0) && Close(p[3], 255)));
    p = pixel(5, 2); // block 1
    CHECK((Close(p[0], 0) && Close(p[1], 255) && Close(p[2], 0) && Close(p[3], 255)));
    p = pixel(2, 6); // block 2
    CHECK((Close(p[0], 0) && Close(p[1], 0) && Close(p[2], 255) && Close(p[3], 255)));
    p = pixel(7, 7); // block 3
    CHECK((Close(p[0], 255) && Close(p[1], 255) && Close(p[2], 255) && Close(p[3], 128)));
}

TEST_CASE("nvn memory: BC1 blocks decode when the device has no BCn", "[nvn]") {
    Nvn::Storage s = MakeStorage(NVN_FORMAT_BC1, 4, 4, 1, PixelFormat::BC1_RGBA_UNORM,
                                 VK_FORMAT_A8B8G8R8_UNORM_PACK32, true);
    // color0 = pure red (RGB565 0xF800), color1 = black, every index 0: red, opaque.
    const std::array<u8, 8> block{0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    const std::vector<u8> guest = SwizzleLevel0(s, block, 8, 1, 1);
    const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, guest.size());
    REQUIRE(plan.unsupported == nullptr);
    REQUIRE(plan.decode == Nvn::UploadPlan::Decode::Bcn);
    std::vector<u8> staging(plan.staging_size);
    Nvn::FillStorageUpload(s, plan, guest.data(), guest.size(), staging.data());
    REQUIRE(staging.size() == 4 * 4 * 4);
    for (u32 pixel = 0; pixel < 16; ++pixel) {
        const u8* p = staging.data() + pixel * 4;
        REQUIRE((Close(p[0], 255) && Close(p[1], 0) && Close(p[2], 0) && Close(p[3], 255)));
    }
}

TEST_CASE("nvn memory: depth/stencil texel layouts that differ are reported, not uploaded", "[nvn]") {
    Nvn::Storage s = MakeStorage(NVN_FORMAT_D24S8, 16, 16, 1, PixelFormat::D24_UNORM_S8_UINT,
                                 VK_FORMAT_D24_UNORM_S8_UINT);
    s.host.depth = true;
    s.host.stencil = true;
    s.host.guest_layout_differs = true;
    const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, s.layout.size);
    REQUIRE(plan.unsupported != nullptr);
    REQUIRE(plan.pieces.empty());
}

TEST_CASE("nvn memory: ASTC decode cost", "[nvn]") {
    // 1024x1024 ASTC 4x4 with random (mostly valid-looking) blocks, the shape of DK's big albedo
    // textures, decoded the way an upload does it. Informational: the time goes in the log.
    Nvn::Storage s = MakeStorage(NVN_FORMAT_ASTC_4X4, 1024, 1024, 1, PixelFormat::ASTC_2D_4X4_UNORM,
                                 VK_FORMAT_A8B8G8R8_UNORM_PACK32, true);
    std::mt19937 random{1234};
    std::vector<u8> guest(static_cast<size_t>(s.layout.size));
    for (u8& byte : guest) {
        byte = static_cast<u8>(random());
    }
    const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, guest.size());
    REQUIRE(plan.unsupported == nullptr);
    std::vector<u8> staging(plan.staging_size);
    Nvn::UploadStats stats;
    const auto begin = std::chrono::steady_clock::now();
    Nvn::FillStorageUpload(s, plan, guest.data(), guest.size(), staging.data(), &stats);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin);
    WARN("[nvn] ASTC 4x4 1024x1024 (one level, one piece): " << elapsed.count() << " ms for "
                                                              << (stats.decode_bytes >> 20)
                                                              << " MiB decoded");
    REQUIRE(stats.decode_bytes == 1024u * 1024u * 4u);
}

// ---------------------------------------------------------------------------------------------
// With a GPU

namespace {

struct Rig {
    static constexpr u64 TEXTURE_POOL = 1;
    static constexpr u64 BUFFER_POOL = 2;
    static constexpr u64 TEXTURE_POOL_VA = 0x1000'0000;
    static constexpr u64 BUFFER_POOL_VA = 0x2000'0000;

    explicit Rig(HeadlessVulkan& vulkan, bool allow_import,
                 u32 texture_pool_flags = POOL_CPU_CACHED,
                 std::optional<u64> storage_cache_budget = {},
                 std::optional<u64> trim_age_presents = {}, std::optional<bool> trim_legacy = {},
                 std::optional<bool> vma_device_strict = {},
                 std::unique_ptr<Nvn::PresentTarget> target = {},
                 std::optional<u64> sparse_device_budget = {})
        : context(*vulkan.device, vulkan.submit_mutex, std::move(target),
                  {.allow_host_import = allow_import,
                   .storage_cache_budget_bytes = storage_cache_budget,
                   .sparse_device_budget_bytes = sparse_device_budget,
                   .storage_trim_age_presents = trim_age_presents,
                   .storage_trim_legacy = trim_legacy,
                   .vma_device_strict = vma_device_strict}) {
        texture_bytes = memory.Add(TEXTURE_POOL_VA, 4_MiB);
        readback = memory.Add(BUFFER_POOL_VA, 1_MiB);
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = TEXTURE_POOL, .guest_address = TEXTURE_POOL_VA,
                            .size = 4_MiB, .flags = texture_pool_flags});
        context.CreatePool({.serial = BUFFER_POOL, .guest_address = BUFFER_POOL_VA,
                            .size = 1_MiB, .flags = POOL_CPU_CACHED});
    }

    /// Copies the whole texture into the readback pool and returns its linear bytes.
    std::vector<u8> Read(u64 texture, u32 width, u32 height) {
        std::memset(readback, 0xCD, width * height * 4);
        std::vector<u64> list;
        Push(list, VC::Op::CopyTextureToBuffer, CopyToBuffer(BUFFER_POOL_VA, texture, width, height));
        Run(context, seq, std::move(list));
        return std::vector<u8>(readback, readback + static_cast<size_t>(width) * height * 4);
    }

    /// Empty submissions after Destroy calls: the destroyed textures retire and the finished
    /// submissions let go of the storages they touched.
    void Settle() {
        for (int i = 0; i < 3; ++i) {
            Run(context, seq, {});
        }
    }

    /// Creates a texture over `offset` of a pool, clears it and destroys it again: the storage
    /// stays in the cache with nothing but the GPU's texels in it.
    void Paint(u64 serial, u64 offset, u32 width, u32 height, float r, float g, float b,
               u64 pool = TEXTURE_POOL) {
        context.CreateTexture(MakeTexture(serial, pool, offset, NVN_FORMAT_RGBA8, width, height));
        std::vector<u64> list;
        Push(list, VC::Op::ClearTexture, ClearColor(serial, width, height, r, g, b, 1.0f), 0xF);
        Run(context, seq, std::move(list));
        context.Destroy(serial);
        Settle();
    }

    /// Presents nothing `count` times (no present target: only the replay's present work runs).
    void PresentNothing(u32 count) {
        for (u32 i = 0; i < count; ++i) {
            context.Present({.seq = ++seq, .queue = 0, .texture_serial = 0, .window_slot = 0,
                             .present_interval = 1});
        }
        Run(context, seq, {});
    }

    FakeMemory memory;
    Nvn::Context context;
    u8* texture_bytes{};
    u8* readback{};
    u64 seq{0};
};

bool AllTexels(const std::vector<u8>& texels, u8 r, u8 g, u8 b, u8 a) {
    for (size_t i = 0; i + 3 < texels.size(); i += 4) {
        if (texels[i] != r || texels[i + 1] != g || texels[i + 2] != b || texels[i + 3] != a) {
            return false;
        }
    }
    return !texels.empty();
}

} // Anonymous namespace

TEST_CASE("nvn memory: CPU writes to a texture's pool reach the image", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool allow_import = GENERATE(true, false);
    constexpr u32 SIZE = 32;
    Rig rig{*vulkan, allow_import};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const auto put = [&](u8 seed) {
        const std::vector<u8> texels = Gradient(SIZE, SIZE, seed);
        const std::vector<u8> guest = SwizzleLevel0(reference, texels, 4, SIZE, SIZE);
        std::memcpy(rig.texture_bytes, guest.data(), guest.size());
        return texels;
    };

    // First use uploads what the game has written so far.
    std::vector<u8> expected = put(1);
    REQUIRE(rig.Read(10, SIZE, SIZE) == expected);

    // The game rewrites the pool and says so (FlushMappedRange): the next use sees it.
    expected = put(2);
    rig.context.NotifyCpuWrite(Rig::TEXTURE_POOL_VA, 4096);
    REQUIRE(rig.Read(10, SIZE, SIZE) == expected);

    // The game rewrites it without saying anything (a decoder writing a video frame): the
    // replay's own check finds it once its clock has moved on (here: 16 ms per idle tick).
    expected = put(3);
    std::this_thread::sleep_for(100ms);
    REQUIRE(rig.Read(10, SIZE, SIZE) == expected);

    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.uploads >= 3);
    REQUIRE(stats.cpu_changes >= 1);
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.ops_unsupported == 0);
    WARN("[nvn] coherency (" << (allow_import ? "imported" : "mirror") << "): uploads="
                             << stats.uploads << " cpu_checks=" << stats.cpu_checks
                             << " cpu_changes=" << stats.cpu_changes);
}

TEST_CASE("nvn memory: WriteTexels-style writes replace what the GPU wrote", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 16;
    Rig rig{*vulkan, true};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, SIZE, SIZE, 1.0f, 0.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));
    const std::vector<u8> red = rig.Read(10, SIZE, SIZE);
    REQUIRE(red[0] == 255);
    REQUIRE(red[1] == 0);

    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const std::vector<u8> texels = Gradient(SIZE, SIZE, 9);
    const std::vector<u8> guest = SwizzleLevel0(reference, texels, 4, SIZE, SIZE);
    // GPU contents stay while the guest bytes are unchanged, however often the pool is flushed.
    rig.context.NotifyCpuWrite(Rig::TEXTURE_POOL_VA, 4096);
    std::this_thread::sleep_for(100ms);
    REQUIRE(rig.Read(10, SIZE, SIZE) == red);
    // A CPU rewrite of a GPU-written texture (a decoder writing a plane the game cleared first)
    // is found by the replay's check and replaces the GPU contents.
    std::memcpy(rig.texture_bytes, guest.data(), guest.size());
    std::this_thread::sleep_for(100ms);
    REQUIRE(rig.Read(10, SIZE, SIZE) == texels);

    // A texel write to this very texture replaces GPU contents at once.
    std::vector<u64> blue;
    Push(blue, VC::Op::ClearTexture, ClearColor(10, SIZE, SIZE, 0.0f, 0.0f, 1.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(blue));
    REQUIRE(rig.Read(10, SIZE, SIZE)[2] == 255);
    const std::vector<u8> other = Gradient(SIZE, SIZE, 33);
    const std::vector<u8> other_guest = SwizzleLevel0(reference, other, 4, SIZE, SIZE);
    std::memcpy(rig.texture_bytes, other_guest.data(), other_guest.size());
    rig.context.NotifyTextureWritten(10);
    REQUIRE(rig.Read(10, SIZE, SIZE) == other);
}

TEST_CASE("nvn memory: a host write reaches a CPU_NO_ACCESS pool's buffer and textures", "[nvn]") {
    // nx2:vdecsvc writes decoded frames into guest memory the game made a GPU-only pool of; on
    // the console the GPU then reads the new bytes with no NVN call in between.
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 16;
    Rig rig{*vulkan, true, POOL_CPU_NO_ACCESS};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const auto put = [&](u8 seed) {
        const std::vector<u8> texels = Gradient(SIZE, SIZE, seed);
        const std::vector<u8> guest = SwizzleLevel0(reference, texels, 4, SIZE, SIZE);
        std::memcpy(rig.texture_bytes, guest.data(), guest.size());
        return std::make_pair(texels, guest.size());
    };

    // The texture: the GPU clears it, then the decoder writes a frame over it.
    std::vector<u64> clear;
    Push(clear, VC::Op::ClearTexture, ClearColor(10, SIZE, SIZE, 1.0f, 0.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(clear));
    REQUIRE(rig.Read(10, SIZE, SIZE)[0] == 255);
    const auto [frame, frame_bytes] = put(5);
    rig.context.NotifyHostWrite(Rig::TEXTURE_POOL_VA, frame_bytes);
    REQUIRE(rig.Read(10, SIZE, SIZE) == frame);

    // The pool's own buffer (a buffer copy source): uploaded once at the first use, then only a
    // host write brings new bytes.
    constexpr u64 OFFSET = 0x100000;
    constexpr u64 BYTES = 256;
    const auto copy_out = [&] {
        std::memset(rig.readback, 0xCD, BYTES);
        std::vector<u64> list;
        Push(list, VC::Op::CopyBufferToBuffer,
             VC::CopyBufferToBufferPayload{.src = Rig::TEXTURE_POOL_VA + OFFSET,
                                           .dst = Rig::BUFFER_POOL_VA, .size = BYTES});
        Run(rig.context, rig.seq, std::move(list));
        return std::vector<u8>(rig.readback, rig.readback + BYTES);
    };
    std::vector<u8> first(BYTES, 0x11);
    std::memcpy(rig.texture_bytes + OFFSET, first.data(), BYTES);
    REQUIRE(copy_out() == first);
    std::vector<u8> second(BYTES);
    for (u64 i = 0; i < BYTES; ++i) {
        second[i] = static_cast<u8>(i * 7 + 3);
    }
    std::memcpy(rig.texture_bytes + OFFSET, second.data(), BYTES);
    const u64 uploads_before = rig.context.Stats().host_write_uploads;
    rig.context.NotifyHostWrite(Rig::TEXTURE_POOL_VA + OFFSET, BYTES);
    REQUIRE(copy_out() == second);
    CHECK(rig.context.Stats().host_write_uploads == uploads_before + 1);
    CHECK(rig.context.Stats().ops_failed == 0);
}

TEST_CASE("nvn format: sRGB textures get storage views of their UNORM bits", "[nvn]") {
    // NVN images of sRGB textures read and write the bits with no sRGB conversion.
    CHECK(Nvn::SrgbStorageViewFormat(VK_FORMAT_A8B8G8R8_SRGB_PACK32) ==
          VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    CHECK(Nvn::SrgbStorageViewFormat(VK_FORMAT_R8G8B8A8_SRGB) == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(Nvn::SrgbStorageViewFormat(VK_FORMAT_B8G8R8A8_SRGB) == VK_FORMAT_B8G8R8A8_UNORM);
    CHECK(Nvn::SrgbStorageViewFormat(VK_FORMAT_A8B8G8R8_UNORM_PACK32) == VK_FORMAT_UNDEFINED);
    CHECK(Nvn::SrgbStorageViewFormat(VK_FORMAT_BC7_SRGB_BLOCK) == VK_FORMAT_UNDEFINED);
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr s32 NVN_FORMAT_RGBA8_SRGB = 0x38;
    const auto host = Nvn::ResolveHostFormat(*vulkan->device, NVN_FORMAT_RGBA8_SRGB);
    REQUIRE(host);
    CHECK(host->format == VK_FORMAT_A8B8G8R8_SRGB_PACK32);
    CHECK_FALSE(host->storage);
    CHECK(host->storage_view == VK_FORMAT_A8B8G8R8_UNORM_PACK32);
}

TEST_CASE("nvn format: unstorable colour formats get storage for views of their texel size",
          "[nvn]") {
    CHECK(Nvn::SameSizeUintFormat(1) == VK_FORMAT_R8_UINT);
    CHECK(Nvn::SameSizeUintFormat(2) == VK_FORMAT_R16_UINT);
    CHECK(Nvn::SameSizeUintFormat(4) == VK_FORMAT_R32_UINT);
    CHECK(Nvn::SameSizeUintFormat(8) == VK_FORMAT_R32G32_UINT);
    CHECK(Nvn::SameSizeUintFormat(16) == VK_FORMAT_R32G32B32A32_UINT);
    CHECK(Nvn::SameSizeUintFormat(12) == VK_FORMAT_UNDEFINED);
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr s32 NVN_FORMAT_RGB9E5F = 0x40;
    const auto rgb9e5 = Nvn::ResolveHostFormat(*vulkan->device, NVN_FORMAT_RGB9E5F);
    REQUIRE(rgb9e5);
    CHECK(rgb9e5->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
    CHECK_FALSE(rgb9e5->storage);
    CHECK(rgb9e5->storage_view == VK_FORMAT_UNDEFINED); // its own bits cannot be stored
    CHECK(rgb9e5->storage_alias);                        // an R32UI view of them can
    const auto r32ui = Nvn::ResolveHostFormat(*vulkan->device, NVN_FORMAT_R32UI);
    REQUIRE(r32ui);
    CHECK(r32ui->storage);
    CHECK_FALSE(r32ui->storage_alias);
    const auto depth = Nvn::ResolveHostFormat(*vulkan->device, NVN_FORMAT_D24S8);
    REQUIRE(depth);
    CHECK_FALSE(depth->storage_alias);
    const auto bc1 = Nvn::ResolveHostFormat(*vulkan->device, NVN_FORMAT_BC1);
    REQUIRE(bc1);
    CHECK_FALSE(bc1->storage_alias);
}

TEST_CASE("nvn memory: a partial texel write keeps the GPU's other texels", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 16;
    Rig rig{*vulkan, true};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, SIZE, SIZE, 1.0f, 0.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));
    // What WriteTexels does: flush the GPU's texels into the guest bytes, patch them there, tell
    // the replay.
    REQUIRE(rig.context.FlushTextureToGuest(10));
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    std::vector<u8> expected(static_cast<size_t>(SIZE) * SIZE * 4);
    for (size_t i = 0; i < expected.size(); i += 4) {
        expected[i] = 255;
        expected[i + 3] = 255;
    }
    // The flushed guest bytes hold the red texels.
    const std::vector<u8> flushed = SwizzleLevel0(reference, expected, 4, SIZE, SIZE);
    REQUIRE(std::memcmp(rig.texture_bytes, flushed.data(), flushed.size()) == 0);
    for (u32 y = 4; y < 8; ++y) {
        for (u32 x = 4; x < 8; ++x) {
            u8* p = expected.data() + (y * SIZE + x) * 4;
            p[0] = 0;
            p[1] = 255;
        }
    }
    const std::vector<u8> guest = SwizzleLevel0(reference, expected, 4, SIZE, SIZE);
    std::memcpy(rig.texture_bytes, guest.data(), guest.size());
    rig.context.NotifyTextureWritten(10);
    REQUIRE(rig.Read(10, SIZE, SIZE) == expected);
}

TEST_CASE("nvn memory: a re-created texture over rewritten bytes sees the new bytes", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 16;
    Rig rig{*vulkan, true};
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const auto put = [&](u8 seed) {
        const std::vector<u8> texels = Gradient(SIZE, SIZE, seed);
        const std::vector<u8> guest = SwizzleLevel0(reference, texels, 4, SIZE, SIZE);
        std::memcpy(rig.texture_bytes, guest.data(), guest.size());
        return texels;
    };
    // After a CPU upload: the bytes change while no texture exists, then a texture is created.
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    const std::vector<u8> first = put(1);
    REQUIRE(rig.Read(10, SIZE, SIZE) == first);
    std::vector<u8> expected = put(2);
    rig.context.Destroy(10);
    rig.context.CreateTexture(MakeTexture(11, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    REQUIRE(rig.Read(11, SIZE, SIZE) == expected);
    REQUIRE(rig.context.Stats().storages_reused == 1);

    // After a GPU write: packaged data copied by the driver is announced with NotifyTextureWritten.
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(11, SIZE, SIZE, 0.0f, 1.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));
    rig.context.Destroy(11);
    expected = put(3);
    rig.context.CreateTexture(MakeTexture(12, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    rig.context.NotifyTextureWritten(12);
    REQUIRE(rig.Read(12, SIZE, SIZE) == expected);
    REQUIRE(rig.context.Stats().storages_reused == 2);
}

TEST_CASE("nvn memory: textures over the same bytes reuse one image", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 64;
    Rig rig{*vulkan, true};
    std::vector<u64> list;
    // DK re-creates its transient render targets over the same bytes every frame.
    for (u64 frame = 0; frame < 20; ++frame) {
        rig.context.CreateTexture(
            MakeTexture(100 + frame, Rig::TEXTURE_POOL, 0x10000, NVN_FORMAT_RGBA8, SIZE, SIZE));
        std::vector<u64> frame_list;
        const float v = static_cast<float>(frame) / 32.0f;
        Push(frame_list, VC::Op::ClearTexture, ClearColor(100 + frame, SIZE, SIZE, v, 0, 0, 1), 0xF);
        Run(rig.context, rig.seq, std::move(frame_list));
        rig.context.Destroy(100 + frame);
    }
    // One more submission lets the destroys retire.
    std::vector<u64> empty;
    Run(rig.context, rig.seq, std::move(empty));
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.textures_created == 20);
    REQUIRE(stats.storages_created == 1);
    REQUIRE(stats.storages_reused == 19);
    REQUIRE(stats.alias_unsupported == 0);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("nvn memory: a render target the GPU wrote again is dropped over the budget and its "
          "bytes survive",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 WIDTH = 64;
    constexpr u32 SHORT = 128;
    constexpr u32 TALL = 136;
    REQUIRE(Nvn::ComputeTextureLayout({.target = 1, .format = NVN_FORMAT_RGBA8, .flags = 0,
                                       .samples = 0, .width = WIDTH, .height = SHORT, .depth = 1,
                                       .levels = 1, .stride = 0, .pkg_block_h = 0})
                .tile_h ==
            Nvn::ComputeTextureLayout({.target = 1, .format = NVN_FORMAT_RGBA8, .flags = 0,
                                       .samples = 0, .width = WIDTH, .height = TALL, .depth = 1,
                                       .levels = 1, .stride = 0, .pkg_block_h = 0})
                .tile_h);
    // Two shapes over the same bytes (32 KiB and 34 KiB of texels) do not fit 40 KiB.
    Rig rig{*vulkan, true, POOL_CPU_CACHED, 40_KiB};
    rig.Paint(10, 0, WIDTH, SHORT, 1.0f, 0.0f, 0.0f); // red, the shorter shape
    rig.Paint(11, 0, WIDTH, TALL, 0.0f, 1.0f, 0.0f);  // green over all of it, later
    REQUIRE(rig.context.Stats().storages_evicted == 0);
    // A storage elsewhere is created: the pass finds the red one replaced by the green one.
    rig.Paint(12, 1_MiB, WIDTH, 64, 0.0f, 0.0f, 1.0f);
    Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.storages_evicted == 1);
    REQUIRE(stats.storages_superseded_evicted == 1);
    const u64 created = stats.storages_created;
    const u64 cache_bytes = stats.storage_cache_bytes;
    // The shorter shape comes back: a new image that takes the bytes the green one holds.
    rig.context.CreateTexture(MakeTexture(13, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, WIDTH, SHORT));
    REQUIRE(AllTexels(rig.Read(13, WIDTH, SHORT), 0, 255, 0, 255));
    stats = rig.context.Stats();
    REQUIRE(stats.storages_created == created + 1);
    REQUIRE(stats.storage_cache_bytes > cache_bytes);
    REQUIRE(stats.alias_unsupported == 0);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("nvn memory: GPU-written storages nothing replaced stay over the budget", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 64;
    Rig rig{*vulkan, true, POOL_CPU_CACHED, 40_KiB};
    // Four 16 KiB render targets over different bytes: 64 KiB in all, over the 40 KiB budget.
    for (u64 i = 0; i < 4; ++i) {
        rig.Paint(20 + i, i * 256_KiB, SIZE, SIZE, i % 2 == 0 ? 0.0f : 1.0f, 0.0f, 1.0f);
    }
    Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.storages_created == 4);
    REQUIRE(stats.storages_evicted == 0);
    REQUIRE(stats.storage_cache_bytes == 4u * SIZE * SIZE * 4);
    // Their texels are still there when the guest makes a texture over them again.
    rig.context.CreateTexture(MakeTexture(30, Rig::TEXTURE_POOL, 256_KiB, NVN_FORMAT_RGBA8, SIZE, SIZE));
    REQUIRE(AllTexels(rig.Read(30, SIZE, SIZE), 255, 0, 255, 255));
    stats = rig.context.Stats();
    REQUIRE(stats.storages_created == 4);
    REQUIRE(stats.storages_reused == 1);
}

TEST_CASE("nvn memory: an idle render target of guest bytes ages out and uploads them again",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 32;
    const bool legacy = GENERATE(false, true);
    Rig rig{*vulkan, true, POOL_CPU_CACHED, {}, 1, legacy};
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const std::vector<u8> texels = Gradient(SIZE, SIZE, 7);
    const std::vector<u8> guest = SwizzleLevel0(reference, texels, 4, SIZE, SIZE);
    std::memcpy(rig.texture_bytes, guest.data(), guest.size());
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    REQUIRE(rig.Read(10, SIZE, SIZE) == texels);
    rig.context.Destroy(10);
    rig.Settle();
    REQUIRE(rig.context.Stats().storage_cache_bytes == SIZE * SIZE * 4);
    rig.PresentNothing(2);
    Nvn::ContextStats stats = rig.context.Stats();
    if (legacy) {
        // Under the budget the trim of before keeps it.
        REQUIRE(stats.storages_evicted == 0);
        REQUIRE(stats.storage_cache_bytes == SIZE * SIZE * 4);
        return;
    }
    REQUIRE(stats.storages_evicted == 1);
    REQUIRE(stats.storages_aged_evicted == 1);
    REQUIRE(stats.storage_cache_bytes == 0);
    // The texture over the same bytes gets a new image with the guest's texels.
    rig.context.CreateTexture(MakeTexture(11, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    REQUIRE(rig.Read(11, SIZE, SIZE) == texels);
    stats = rig.context.Stats();
    REQUIRE(stats.storages_created == 2);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("nvn memory: an idle GPU-written render target ages out only when it was replaced",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 WIDTH = 64;
    constexpr u32 SHORT = 128;
    constexpr u32 TALL = 136;
    Rig rig{*vulkan, true, POOL_CPU_CACHED, {}, 1};
    rig.Paint(10, 0, WIDTH, SHORT, 1.0f, 0.0f, 0.0f);    // red, the shorter shape
    rig.Paint(11, 0, WIDTH, TALL, 0.0f, 1.0f, 0.0f);     // green over all of it, later
    rig.Paint(12, 1_MiB, WIDTH, 64, 0.0f, 0.0f, 1.0f);   // blue, nothing over it
    REQUIRE(rig.context.Stats().storages_evicted == 0);
    rig.PresentNothing(2);
    Nvn::ContextStats stats = rig.context.Stats();
    // Far under the budget: the red one goes (its bytes are the green one's now), the green and
    // the blue ones hold texels nothing else does and stay.
    REQUIRE(stats.storages_evicted == 1);
    REQUIRE(stats.storages_aged_evicted == 1);
    REQUIRE(stats.storages_superseded_evicted == 1);
    REQUIRE(stats.storage_cache_bytes == (static_cast<u64>(TALL) + 64) * WIDTH * 4);
    rig.context.CreateTexture(MakeTexture(13, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, WIDTH, SHORT));
    REQUIRE(AllTexels(rig.Read(13, WIDTH, SHORT), 0, 255, 0, 255));
    rig.context.CreateTexture(MakeTexture(14, Rig::TEXTURE_POOL, 1_MiB, NVN_FORMAT_RGBA8, WIDTH, 64));
    REQUIRE(AllTexels(rig.Read(14, WIDTH, 64), 0, 0, 255, 255));
    stats = rig.context.Stats();
    REQUIRE(stats.alias_unsupported == 0);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("nvn memory: a storage a texture uses is not dropped even when a later write replaced it",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan, true, POOL_CPU_CACHED, 40_KiB};
    // The texture 10 stays alive (the guest keeps it) while a taller shape is painted over it.
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, 64, 128));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, 64, 128, 1.0f, 0.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));
    rig.Paint(11, 0, 64, 136, 0.0f, 1.0f, 0.0f);
    rig.Paint(12, 1_MiB, 64, 64, 0.0f, 0.0f, 1.0f);
    rig.Paint(13, 2_MiB, 64, 64, 0.0f, 0.0f, 1.0f);
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.storages_superseded_evicted == 0);
    REQUIRE(stats.storages_created == 4);
    // The storage the texture holds takes the texels the later write left in those bytes.
    REQUIRE(stats.storages_evicted == 0);
    REQUIRE(AllTexels(rig.Read(10, 64, 128), 0, 255, 0, 255));
}

TEST_CASE("nvn memory: render targets the game re-creates in ever other shapes stay bounded",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 WIDTH = 64;
    constexpr u32 FIRST = 128;
    constexpr u32 SHAPES = 24;
    constexpr u64 BUDGET = 64_KiB;
    // Every shape is 4 rows taller than the last one over the same first byte, the way a render
    // target that follows a changing resolution is allocated: each gets a storage of its own, and
    // each one covers the one before it.
    Rig rig{*vulkan, true, POOL_CPU_CACHED, BUDGET};
    u64 largest = 0;
    for (u32 i = 0; i < SHAPES; ++i) {
        const u32 height = FIRST + 4 * i;
        rig.Paint(100 + i, 0, WIDTH, height, static_cast<float>(i) / SHAPES, 0.5f, 0.5f);
        largest = static_cast<u64>(WIDTH) * height * 4;
    }
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.storages_created == SHAPES);
    // All but the last few were dropped as they were replaced (the cache of the untrimmed code
    // holds all 24, about 1.1 MB).
    REQUIRE(stats.storages_superseded_evicted + 4 >= SHAPES);
    REQUIRE(stats.storage_cache_bytes <= 3 * largest);
    REQUIRE(stats.ops_failed == 0);
}

TEST_CASE("nvn memory: a texture pool's storages go with the pool", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig{*vulkan, true};
    u64 evicted = 0;
    // Pools created and finalized over and over, the way the game's areas come and go: the cache
    // holds nothing of a pool that is gone.
    for (u64 cycle = 0; cycle < 4; ++cycle) {
        const u64 pool = 60 + cycle;
        const u64 address = 0x3000'0000 + cycle * 0x10'0000;
        rig.memory.Add(address, 1_MiB);
        rig.context.CreatePool({.serial = pool, .guest_address = address, .size = 1_MiB,
                                .flags = POOL_CPU_CACHED});
        for (u64 i = 0; i < 8; ++i) {
            rig.Paint(1000 + cycle * 8 + i, i * 64_KiB, 64, 64, 0.5f, 0.5f, 0.5f, pool);
        }
        REQUIRE(rig.context.Stats().storage_cache_bytes == 8u * 64 * 64 * 4);
        rig.context.Destroy(pool);
        rig.Settle();
        const Nvn::ContextStats stats = rig.context.Stats();
        REQUIRE(stats.storage_cache_bytes == 0);
        REQUIRE(stats.storages_evicted == evicted + 8);
        evicted = stats.storages_evicted;
    }
}

TEST_CASE("nvn memory: a texture reads what an overlapping texture of another size wrote", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool allow_import = GENERATE(true, false);
    // 64x136 and 64x128 RGBA8 over the same bytes: the same block height, so rows coincide.
    constexpr u32 WIDTH = 64;
    constexpr u32 TALL = 136;
    constexpr u32 SHORT = 128;
    const Nvn::TextureLayout tall = Nvn::ComputeTextureLayout(
        {.target = 1, .format = NVN_FORMAT_RGBA8, .flags = 0, .samples = 0, .width = WIDTH,
         .height = TALL, .depth = 1, .levels = 1, .stride = 0, .pkg_block_h = 0});
    const Nvn::TextureLayout shorter = Nvn::ComputeTextureLayout(
        {.target = 1, .format = NVN_FORMAT_RGBA8, .flags = 0, .samples = 0, .width = WIDTH,
         .height = SHORT, .depth = 1, .levels = 1, .stride = 0, .pkg_block_h = 0});
    REQUIRE(tall.tile_h == shorter.tile_h);

    Rig rig{*vulkan, allow_import};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, WIDTH, TALL));
    rig.context.CreateTexture(MakeTexture(11, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, WIDTH, SHORT));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, WIDTH, TALL, 0.0f, 1.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));
    const std::vector<u8> seen = rig.Read(11, WIDTH, SHORT);
    size_t wrong = 0;
    for (size_t pixel = 0; pixel < static_cast<size_t>(WIDTH) * SHORT; ++pixel) {
        if (seen[pixel * 4 + 0] != 0 || seen[pixel * 4 + 1] != 255 || seen[pixel * 4 + 2] != 0 ||
            seen[pixel * 4 + 3] != 255) {
            ++wrong;
        }
    }
    REQUIRE(wrong == 0);
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.alias_transfers == 1);
    REQUIRE(stats.alias_unsupported == 0);
    REQUIRE(stats.storages_created == 2);
}

TEST_CASE("nvn memory: overlapping textures of other layouts exchange contents through the pool's bytes",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool can_transfer = Nvn::AliasTransferGpu::Supported(*vulkan->device);
    Rig rig{*vulkan, true};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, 64, 64));
    rig.context.CreateTexture(MakeTexture(11, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, 32, 64));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, 64, 64, 1, 1, 1, 1), 0xF);
    Run(rig.context, rig.seq, std::move(list));
    const std::vector<u8> seen = rig.Read(11, 32, 64);
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.alias_unsupported == (can_transfer ? 0u : 1u));
    REQUIRE(stats.alias_byte_transfers == (can_transfer ? 1u : 0u));
    if (can_transfer) {
        // Every byte of the smaller texture lies in the white one's bytes, whatever the layouts.
        CHECK(std::ranges::all_of(seen, [](u8 byte) { return byte == 0xFF; }));
    }
    // Reading again is not another event for the same pair.
    rig.Read(11, 32, 64);
    REQUIRE(rig.context.Stats().alias_unsupported == (can_transfer ? 0u : 1u));
    REQUIRE(rig.context.Stats().alias_byte_transfers == (can_transfer ? 1u : 0u));
}

TEST_CASE("nvn memory: a texture reads another shape's texels at the byte offsets its own layout "
          "gives them",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!Nvn::AliasTransferGpu::Supported(*vulkan->device)) {
        SKIP("no alias transfer shader on this device");
    }
    struct Shape {
        u32 width;
        u32 height;
    };
    const bool allow_import = GENERATE(true, false);
    // The writer at the reader's start, and half way into a reader block row (the transfer cuts
    // the reader down to the block rows the shared bytes reach).
    const u32 writer_offset = GENERATE(0u, 0x8000u);
    const Shape shape = GENERATE(Shape{32, 64}, Shape{64, 96}, Shape{96, 64}, Shape{16, 128},
                                 Shape{128, 16}, Shape{20, 40}, Shape{64, 64}, Shape{64, 256},
                                 Shape{128, 256});
    constexpr u32 WIDTH = 64;
    constexpr u32 HEIGHT = 64;
    // The writer: a CPU gradient with a rectangle the GPU wrote (so its image is not the guest
    // bytes). The reader sees those bytes through its own shape.
    Rig rig{*vulkan, allow_import};
    Nvn::Storage writer = MakeStorage(NVN_FORMAT_RGBA8, WIDTH, HEIGHT, 1, PixelFormat::A8B8G8R8_UNORM,
                                      VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    Nvn::Storage reader = MakeStorage(NVN_FORMAT_RGBA8, shape.width, shape.height, 1,
                                      PixelFormat::A8B8G8R8_UNORM,
                                      VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    std::vector<u8> texels = Gradient(WIDTH, HEIGHT, 3);
    const std::vector<u8> gradient = SwizzleLevel0(writer, texels, 4, WIDTH, HEIGHT);
    std::memcpy(rig.texture_bytes + writer_offset, gradient.data(), gradient.size());
    rig.context.CreateTexture(
        MakeTexture(10, Rig::TEXTURE_POOL, writer_offset, NVN_FORMAT_RGBA8, WIDTH, HEIGHT));
    rig.context.CreateTexture(
        MakeTexture(11, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, shape.width, shape.height));
    VC::ClearTexturePayload clear = ClearColor(10, WIDTH, HEIGHT, 0.0f, 1.0f, 0.0f, 1.0f);
    clear.region = {8, 8, 0, 16, 16, 1};
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, clear, 0xF);
    Run(rig.context, rig.seq, std::move(list));
    for (u32 y = 8; y < 24; ++y) {
        for (u32 x = 8; x < 24; ++x) {
            u8* p = texels.data() + (static_cast<size_t>(y) * WIDTH + x) * 4;
            p[0] = 0;
            p[1] = 255;
            p[2] = 0;
            p[3] = 255;
        }
    }
    // The pool's bytes now: zeros, the writer's layout of its image at its offset.
    const std::vector<u8> written = SwizzleLevel0(writer, texels, 4, WIDTH, HEIGHT);
    u32 tile_h = 0;
    u32 tile_d = 0;
    Nvn::TextureLevelTiles(reader.desc, 0, tile_h, tile_d);
    const size_t needed = Tegra::Texture::CalculateSize(true, 4, shape.width, shape.height, 1,
                                                        tile_h, tile_d);
    std::vector<u8> memory((std::max)(writer_offset + written.size(), needed), 0);
    std::memcpy(memory.data() + writer_offset, written.data(), written.size());
    std::vector<u8> expected(static_cast<size_t>(shape.width) * shape.height * 4);
    Tegra::Texture::UnswizzleTexture(expected, memory, 4, shape.width, shape.height, 1, tile_h,
                                     tile_d, 0, true);
    const std::vector<u8> seen = rig.Read(11, shape.width, shape.height);
    size_t wrong = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        wrong += seen[i] != expected[i];
    }
    REQUIRE(wrong == 0);
    REQUIRE(rig.context.Stats().alias_unsupported == 0);
}

TEST_CASE("nvn memory: virtual pool ranges resolve through MapVirtual", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool allow_import = GENERATE(true, false);
    constexpr u64 PHYSICAL_POOL = 5;
    constexpr u64 VIRTUAL_POOL = 6;
    constexpr u64 PHYSICAL_VA = 0x3000'0000;
    constexpr u64 VIRTUAL_VA = 0x7D00'0000'0000'0000ULL;
    Rig rig{*vulkan, allow_import};
    u8* physical = rig.memory.Add(PHYSICAL_VA, 1_MiB);
    for (size_t i = 0; i < 1_MiB; ++i) {
        physical[i] = static_cast<u8>(i * 31 + (i >> 8));
    }
    rig.context.CreatePool({.serial = PHYSICAL_POOL, .guest_address = PHYSICAL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_CACHED});
    rig.context.CreatePool({.serial = VIRTUAL_POOL, .guest_address = VIRTUAL_VA, .size = 4_MiB,
                            .flags = POOL_VIRTUAL});
    // Virtual 0x10000.. -> physical 0x2000.., and virtual 0x14000.. -> physical 0x6000.. (adjacent
    // in both: one range for a buffer that spans them).
    rig.context.MapVirtual(VIRTUAL_POOL, {{0x10000, PHYSICAL_POOL, 0x2000, 0x4000},
                                           {0x14000, PHYSICAL_POOL, 0x6000, 0x4000}});
    std::vector<u64> list;
    VC::CopyBufferToBufferPayload copy{};
    copy.src = VIRTUAL_VA + 0x10000 + 0x3F00; // 0x200 bytes across the two mappings
    copy.dst = Rig::BUFFER_POOL_VA;
    copy.size = 0x200;
    Push(list, VC::Op::CopyBufferToBuffer, copy);
    Run(rig.context, rig.seq, std::move(list));
    REQUIRE(std::memcmp(rig.readback, physical + 0x2000 + 0x3F00, 0x200) == 0);
    REQUIRE(rig.context.Stats().ops_failed == 0);
    REQUIRE(rig.context.Stats().pools_unsupported == 0);

    // Remapping changes what the same virtual range reads.
    rig.context.MapVirtual(VIRTUAL_POOL, {{0x10000, PHYSICAL_POOL, 0x8000, 0x4000}});
    std::vector<u64> again;
    copy.src = VIRTUAL_VA + 0x10000 + 0x40;
    copy.size = 0x100;
    Push(again, VC::Op::CopyBufferToBuffer, copy);
    Run(rig.context, rig.seq, std::move(again));
    REQUIRE(std::memcmp(rig.readback, physical + 0x8000 + 0x40, 0x100) == 0);

    // An unmapped range cannot be read: the op is dropped and counted, nothing crashes.
    std::vector<u64> hole;
    copy.src = VIRTUAL_VA + 0x100000;
    Push(hole, VC::Op::CopyBufferToBuffer, copy);
    Run(rig.context, rig.seq, std::move(hole));
    REQUIRE(rig.context.Stats().virtual_unresolved == 1);
    REQUIRE(rig.context.Stats().ops_failed == 1);
}

TEST_CASE("nvn memory: a texture in a virtual pool reads the mapped bytes", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 PHYSICAL_POOL = 5;
    constexpr u64 VIRTUAL_POOL = 6;
    constexpr u64 PHYSICAL_VA = 0x3000'0000;
    constexpr u64 VIRTUAL_VA = 0x7D00'0000'0000'0000ULL;
    constexpr u32 SIZE = 32;
    Rig rig{*vulkan, true};
    u8* physical = rig.memory.Add(PHYSICAL_VA, 1_MiB);
    rig.context.CreatePool({.serial = PHYSICAL_POOL, .guest_address = PHYSICAL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_CACHED});
    rig.context.CreatePool({.serial = VIRTUAL_POOL, .guest_address = VIRTUAL_VA, .size = 4_MiB,
                            .flags = POOL_VIRTUAL});
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const std::vector<u8> texels = Gradient(SIZE, SIZE, 5);
    const std::vector<u8> guest = SwizzleLevel0(reference, texels, 4, SIZE, SIZE);
    REQUIRE(guest.size() <= 0x10000);
    std::memcpy(physical + 0x20000, guest.data(), guest.size());
    rig.context.MapVirtual(VIRTUAL_POOL, {{0x30000, PHYSICAL_POOL, 0x20000, 0x10000}});
    rig.context.CreateTexture(MakeTexture(10, VIRTUAL_POOL, 0x30000, NVN_FORMAT_RGBA8, SIZE, SIZE));
    REQUIRE(rig.Read(10, SIZE, SIZE) == texels);

    // The mapping moves to different physical bytes: the texture reads those.
    const std::vector<u8> other = Gradient(SIZE, SIZE, 77);
    const std::vector<u8> other_guest = SwizzleLevel0(reference, other, 4, SIZE, SIZE);
    std::memcpy(physical + 0x50000, other_guest.data(), other_guest.size());
    rig.context.MapVirtual(VIRTUAL_POOL, {{0x30000, PHYSICAL_POOL, 0x50000, 0x10000}});
    REQUIRE(rig.Read(10, SIZE, SIZE) == other);
    REQUIRE(rig.context.Stats().pools_unsupported == 0);
}

TEST_CASE("nvn memory: a large sparse texture is bound and uploaded tile by tile", "[nvn][mkw]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsSparseResidencyImageSupported(VK_IMAGE_TYPE_2D) ||
        !vulkan->device->CanUseUnbackedSparseImages()) {
        SKIP("no sparse residency images");
    }
    constexpr u64 PHYSICAL_POOL = 5;
    constexpr u64 VIRTUAL_POOL = 6;
    constexpr u64 PHYSICAL_VA = 0x3000'0000;
    constexpr u64 VIRTUAL_VA = 0x7D00'0000'0000'0000ULL;
    constexpr u32 NVN_FORMAT_R8 = 0x01;
    constexpr u32 WIDTH = 4096; // 16x8 tiles of 256x256 R8: 8 MiB
    constexpr u32 HEIGHT = 2048;
    constexpr u64 TILE = 0x10000;
    const bool mario_kart_world = GENERATE(false, true);
    Rig rig{*vulkan, true};
    rig.context.SetTitleId(mario_kart_world ? Settings::MarioKartWorldProgramId
                                            : Settings::OcarinaOfTimeProgramId);
    u8* physical = rig.memory.Add(PHYSICAL_VA, 1_MiB);
    rig.context.CreatePool({.serial = PHYSICAL_POOL, .guest_address = PHYSICAL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_CACHED});
    rig.context.CreatePool({.serial = VIRTUAL_POOL, .guest_address = VIRTUAL_VA, .size = 16_MiB,
                            .flags = POOL_VIRTUAL});
    VC::TextureDesc desc = MakeTexture(10, VIRTUAL_POOL, 0, NVN_FORMAT_R8, WIDTH, HEIGHT);
    desc.flags = 0x20; // sparse
    rig.context.CreateTexture(desc);

    // A tile's guest bytes: 256x256 texels swizzled as the one-tile block-linear picture.
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_R8, 256, 256, 1, PixelFormat::R8_UNORM,
                                         VK_FORMAT_R8_UNORM);
    const auto tile_texels = [](u8 seed) {
        std::vector<u8> texels(256 * 256);
        for (u32 i = 0; i < texels.size(); ++i) {
            texels[i] = static_cast<u8>(i * 31 + (i >> 8) * 7 + seed);
        }
        return texels;
    };
    const auto put_tile = [&](u64 physical_offset, const std::vector<u8>& texels) {
        std::vector<u8> guest(TILE, 0);
        Tegra::Texture::SwizzleTexture(guest, texels, 1, 256, 256, 1, 5, 0, 0, true);
        std::memcpy(physical + physical_offset, guest.data(), guest.size());
    };
    // The texels of tile (x, y) of the level as the GPU holds them.
    const auto read_tile = [&](u32 x, u32 y, u64 texture = 10) {
        std::memset(rig.readback, 0xCD, 256 * 256);
        VC::CopyBufferTexturePayload copy{};
        copy.buffer = Rig::BUFFER_POOL_VA;
        copy.texture = {.serial = texture, .level = 0, .layer = 0};
        copy.region = {static_cast<s32>(x * 256), static_cast<s32>(y * 256), 0, 256, 256, 1};
        std::vector<u64> list;
        Push(list, VC::Op::CopyTextureToBuffer, copy);
        Run(rig.context, rig.seq, std::move(list));
        return std::vector<u8>(rig.readback, rig.readback + 256 * 256);
    };

    const std::vector<u8> a = tile_texels(1);
    const std::vector<u8> b = tile_texels(2);
    const std::vector<u8> c = tile_texels(3);
    put_tile(0x10000, a);
    put_tile(0x20000, b);
    put_tile(0x30000, c);
    // Tile (0, 0) is tile 0 of the storage, tile (1, 1) is tile 17 (16 tiles per row).
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, PHYSICAL_POOL, 0x10000, TILE},
                                          {17 * TILE, PHYSICAL_POOL, 0x20000, TILE}});
    REQUIRE(read_tile(0, 0) == a);
    REQUIRE(read_tile(1, 1) == b);
    // A tile nobody mapped reads as zero.
    REQUIRE(read_tile(5, 0) == std::vector<u8>(256 * 256, 0));
    Nvn::ContextStats stats = rig.context.Stats();
    CHECK(stats.pools_unsupported == 0);
    CHECK(stats.virtual_sparse_binds >= 2);
    // Two tiles were uploaded, not the 8 MiB of the storage.
    CHECK(stats.uploads == 2);
    CHECK(stats.upload_bytes == 2 * TILE);

    // An unrelated buffer flush need not search the sparse atlas mappings on MKW.
    u64 checks = stats.sparse_write_pool_checks;
    rig.context.NotifyCpuWrite(Rig::BUFFER_POOL_VA, TILE);
    Run(rig.context, rig.seq, {});
    CHECK(rig.context.Stats().sparse_write_pool_checks == checks + (mario_kart_world ? 0 : 1));

    // Mapping the same bytes again changes nothing: nothing is uploaded.
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, PHYSICAL_POOL, 0x10000, TILE}});
    REQUIRE(read_tile(0, 0) == a);
    CHECK(rig.context.Stats().uploads == 2);

    // Mapping tile 0 to other bytes uploads that tile alone.
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, PHYSICAL_POOL, 0x30000, TILE}});
    REQUIRE(read_tile(0, 0) == c);
    REQUIRE(read_tile(1, 1) == b);
    CHECK(rig.context.Stats().uploads == 3);

    // The game rewrites the bytes behind a mapped tile and flushes the pool: that tile again.
    const std::vector<u8> d = tile_texels(4);
    put_tile(0x20000, d);
    checks = rig.context.Stats().sparse_write_pool_checks;
    rig.context.NotifyCpuWrite(PHYSICAL_VA + 0x20000, TILE);
    REQUIRE(read_tile(1, 1) == d);
    CHECK(rig.context.Stats().uploads == 4);
    CHECK(rig.context.Stats().sparse_write_pool_checks == checks + 1);

    // Unmapping a tile takes its memory back: it reads as zero.
    rig.context.MapVirtual(VIRTUAL_POOL, {{17 * TILE, 0, 0, TILE}});
    REQUIRE(read_tile(1, 1) == std::vector<u8>(256 * 256, 0));
    REQUIRE(read_tile(0, 0) == c);
    CHECK(rig.context.Stats().uploads == 4);

    // Another pool object maps the same physical bytes. MKW keeps the resident texels.
    constexpr u64 ALIAS_POOL = 7;
    rig.context.CreatePool({.serial = ALIAS_POOL,
                            .guest_address = PHYSICAL_VA + 0x30000,
                            .size = TILE,
                            .flags = POOL_CPU_CACHED});
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, ALIAS_POOL, 0, TILE}});
    REQUIRE(read_tile(0, 0) == c);
    CHECK(rig.context.Stats().uploads == (mario_kart_world ? 4u : 5u));

    // The cached physical users must follow a remap to another pool object.
    put_tile(0x30000, a);
    rig.context.NotifyHostWrite(PHYSICAL_VA + 0x30000, TILE);
    REQUIRE(read_tile(0, 0) == a);
    checks = rig.context.Stats().sparse_write_pool_checks;
    rig.context.NotifyCpuWrite(PHYSICAL_VA + 0x10000, TILE);
    Run(rig.context, rig.seq, {});
    CHECK(rig.context.Stats().sparse_write_pool_checks == checks + (mario_kart_world ? 0 : 1));

    // Add a sparse texture after its pool is mapped and the lookup is already populated.
    constexpr u64 SECOND_VIRTUAL_POOL = 8;
    rig.context.CreatePool({.serial = SECOND_VIRTUAL_POOL,
                            .guest_address = VIRTUAL_VA + 32_MiB,
                            .size = 16_MiB,
                            .flags = POOL_VIRTUAL});
    rig.context.MapVirtual(SECOND_VIRTUAL_POOL, {{0, ALIAS_POOL, 0, TILE}});
    rig.context.NotifyCpuWrite(PHYSICAL_VA + 0x30000, TILE);
    Run(rig.context, rig.seq, {});
    desc.serial = 11;
    desc.pool_serial = SECOND_VIRTUAL_POOL;
    rig.context.CreateTexture(desc);
    REQUIRE(read_tile(0, 0, 11) == a);

    put_tile(0x30000, b);
    checks = rig.context.Stats().sparse_write_pool_checks;
    rig.context.NotifyCpuWrite(PHYSICAL_VA + 0x30000, TILE);
    REQUIRE(read_tile(0, 0) == b);
    REQUIRE(read_tile(0, 0, 11) == b);
    CHECK(rig.context.Stats().sparse_write_pool_checks == checks + 2);

    // Removing a virtual pool drops it from the lookup before the next CPU notification.
    rig.context.Destroy(11);
    rig.context.Destroy(SECOND_VIRTUAL_POOL);
    put_tile(0x30000, c);
    checks = rig.context.Stats().sparse_write_pool_checks;
    rig.context.NotifyCpuWrite(PHYSICAL_VA + 0x30000, TILE);
    REQUIRE(read_tile(0, 0) == c);
    CHECK(rig.context.Stats().sparse_write_pool_checks == checks + 1);
}

TEST_CASE("MKW sparse row binds preserve distinct neighboring tiles", "[nvn][mkw][sparse]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan || !vulkan->device->IsSparseResidencyImageSupported(VK_IMAGE_TYPE_2D) ||
        !vulkan->device->CanUseUnbackedSparseImages()) {
        SKIP("no sparse residency images");
    }
    constexpr u64 PHYSICAL_POOL = 5;
    constexpr u64 VIRTUAL_POOL = 6;
    constexpr u64 PHYSICAL_VA = 0x3000'0000;
    constexpr u64 VIRTUAL_VA = 0x7D00'0000'0000'0000ULL;
    constexpr u32 SIZE = 256;
    constexpr u64 TILE = 64_KiB;
    Rig rig{*vulkan, true};
    rig.context.SetTitleId(Settings::MarioKartWorldProgramId);
    u8* physical = rig.memory.Add(PHYSICAL_VA, 4 * TILE);
    rig.context.CreatePool({.serial = PHYSICAL_POOL,
                            .guest_address = PHYSICAL_VA,
                            .size = 4 * TILE,
                            .flags = POOL_CPU_CACHED});
    rig.context.CreatePool({.serial = VIRTUAL_POOL,
                            .guest_address = VIRTUAL_VA,
                            .size = 64_MiB,
                            .flags = POOL_VIRTUAL});
    VC::TextureDesc desc = MakeTexture(10, VIRTUAL_POOL, 0, 0x01, 8192, 8192);
    desc.flags = 0x20;
    rig.context.CreateTexture(desc);
    std::array<std::vector<u8>, 4> texels;
    for (u32 tile = 0; tile < texels.size(); ++tile) {
        auto& pixels = texels[tile];
        pixels.resize(TILE);
        for (u32 i = 0; i < pixels.size(); ++i) {
            pixels[i] = static_cast<u8>(i * 31 + (i >> 8) * 7 + tile * 43);
        }
        std::vector<u8> guest(TILE);
        Tegra::Texture::SwizzleTexture(guest, pixels, 1, SIZE, SIZE, 1, 5, 0, 0, true);
        std::memcpy(physical + tile * TILE, guest.data(), TILE);
    }
    const auto read = [&](u32 tile) {
        VC::CopyBufferTexturePayload copy{};
        copy.buffer = Rig::BUFFER_POOL_VA;
        copy.texture = {.serial = 10};
        copy.region = {static_cast<s32>(tile * SIZE), 0, 0, SIZE, SIZE, 1};
        std::vector<u64> list;
        Push(list, VC::Op::CopyTextureToBuffer, copy);
        Run(rig.context, rig.seq, std::move(list));
        return std::vector<u8>(rig.readback, rig.readback + TILE);
    };
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, PHYSICAL_POOL, 0, 3 * TILE}});
    for (u32 tile = 0; tile < 3; ++tile) {
        CHECK(read(tile) == texels[tile]);
    }
    CHECK(rig.context.Stats().virtual_sparse_binds == 1);
    // A hole inside the compacted run must not unbind either neighbor.
    rig.context.MapVirtual(VIRTUAL_POOL, {{TILE, 0, 0, TILE}});
    CHECK(read(1) == std::vector<u8>(TILE, 0));
    CHECK(read(0) == texels[0]);
    CHECK(read(2) == texels[2]);
    rig.context.MapVirtual(VIRTUAL_POOL, {{TILE, PHYSICAL_POOL, 3 * TILE, TILE}});
    CHECK(read(1) == texels[3]);
    CHECK(read(2) == texels[2]);
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, 0, 0, 3 * TILE}});
    for (u32 tile = 0; tile < 3; ++tile) {
        CHECK(read(tile) == std::vector<u8>(TILE, 0));
    }
    // Refill after reclaiming whole blocks, including allocation slots returned to VMA.
    std::vector<VC::VirtualMapping> burst;
    constexpr u32 BURST_TILES = 384;
    for (u32 tile = 0; tile < BURST_TILES; ++tile) {
        burst.push_back({tile * TILE, PHYSICAL_POOL, (tile % 4) * TILE, TILE});
    }
    for (u32 refill = 0; refill < 2; ++refill) {
        rig.context.MapVirtual(VIRTUAL_POOL, std::vector<VC::VirtualMapping>{burst});
        CHECK(read(0) == texels[0]);
        CHECK(read(1) == texels[1]);
        CHECK(read(2) == texels[2]);
        rig.context.MapVirtual(VIRTUAL_POOL, {{0, 0, 0, BURST_TILES * TILE}});
        CHECK(read(0) == std::vector<u8>(TILE, 0));
    }
    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == 0);
}

TEST_CASE("MKW BC3: large atlas residency fits a constrained device heap",
          "[nvn][mkw][.vram-stress]") {
    auto vulkan = MakeHeadlessVulkan();
    REQUIRE(vulkan);
    REQUIRE_FALSE(vulkan->device->IsOptimalAstcSupported());
    const u64 budget = vulkan->device->GetDeviceMemoryBudget();
    REQUIRE(budget <= 12_GiB);
    INFO("device budget MiB " << (budget >> 20));
    Nvn::Context::Impl impl(*vulkan->device, vulkan->submit_mutex, nullptr,
                            {.background_backing = false, .vma_device_strict = true});
    impl.mario_kart_world = true;
    impl.astc_standin = Nvn::AstcStandin::Bc3;
    REQUIRE(impl.MkwBc3AtlasSupported());
    // Leave 4 GiB occupied by other device-local resources while binding the atlas.
    std::vector<Vulkan::vk::Buffer> pressure;
    for (u32 i = 0; i < 4; ++i) {
        pressure.push_back(impl.CreateHostBuffer(1_GiB, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 Vulkan::MemoryUsage::DeviceLocal,
                                                 Vulkan::AllocationKind::PoolBuffer));
        REQUIRE(pressure.back());
    }
    constexpr u64 VIRTUAL_BYTES = 2_GiB;
    impl.CreatePoolNow({.serial = 6,
                        .guest_address = 0x7D00'0000'0000'0000ULL,
                        .size = VIRTUAL_BYTES,
                        .flags = POOL_VIRTUAL});
    VC::TextureDesc desc = MakeTexture(10, 6, 0, 0x80, 8192, 8192);
    desc.target = 4;
    desc.depth = 68;
    desc.flags = 0x20;
    desc.levels = 14;
    auto texture = impl.BuildTexture(desc);
    REQUIRE(texture);
    auto& storage = *texture->storage;
    REQUIRE(storage.host.format == VK_FORMAT_BC3_UNORM_BLOCK);
    REQUIRE(storage.extent.width == 8192);
    REQUIRE(storage.layers == 68);
    REQUIRE(storage.sparse);
    REQUIRE(impl.EnsureImage(storage));
    REQUIRE(storage.sparse); // never silently materialize the full multi-gigabyte image
    auto& sparse = *storage.sparse;
    const auto& layout = sparse.Layout();
    Nvn::SparseBacking::Uploads uploads;
    REQUIRE(sparse.Prepare(impl, storage, uploads));
    const u64 tail_backing = sparse.BoundBytes();
    const u64 tail_binds = impl.stats.virtual_sparse_binds.load();
    CHECK(tail_backing > 0);
    CHECK(tail_backing < 32_MiB);
    CHECK(uploads.tiles.empty());
    CHECK(uploads.tails.empty()); // unmapped coarse levels must not overwrite GPU texels with zeros
    constexpr u32 RESIDENT_TILES = 11000;
    REQUIRE(layout.TileCount() > RESIDENT_TILES);
    for (u32 id = 0; id < RESIDENT_TILES; ++id) {
        const u64 offset = layout.TileOffset(id);
        sparse.Remap(offset, offset + Nvn::SPARSE_TILE_BYTES, [](u64, u64) { return true; });
    }
    REQUIRE(sparse.Prepare(impl, storage, uploads));
    CHECK(sparse.StarvedUnits() == 0);
    CHECK(uploads.tiles.size() == RESIDENT_TILES);
    // BC3 uses 2x2 host blocks per guest tile; each contiguous host row is one bind.
    CHECK(impl.stats.virtual_sparse_binds.load() - tail_binds == u64{RESIDENT_TILES} * 2);
    CHECK(sparse.BoundBytes() == tail_backing + u64{RESIDENT_TILES} * 256_KiB);
    CHECK(sparse.BoundBytes() < budget / 5 * 2);
    VmaTotalStatistics allocations{};
    vmaCalculateStatistics(impl.vma, &allocations);
    CHECK(allocations.total.statistics.allocationBytes >= 4_GiB + sparse.BoundBytes());
    CHECK(allocations.total.statistics.blockBytes <= budget);
    CHECK(impl.spills.bytes[static_cast<size_t>(Vulkan::AllocationKind::PoolBuffer)].load() == 0);
    // The same full guest tiles formerly needed 10.74 GiB of RGBA8, before other resources.
    CHECK(u64{RESIDENT_TILES} * 1_MiB > budget / 5 * 2);
    WARN("11000 mapped ASTC 8x8 tiles: BC3 "
         << (sparse.BoundBytes() >> 20)
         << " MiB; previous RGBA8 11000 MiB; other resources 4096 MiB; device budget "
         << (budget >> 20) << " MiB");
    sparse.Remap(0, storage.layout.size, [](u64, u64) { return false; });
    uploads = {};
    REQUIRE(sparse.Prepare(impl, storage, uploads));
    CHECK(sparse.BoundBytes() == tail_backing);
    CHECK(impl.sparse_pages->UsedBytes() == tail_backing);
    constexpr u64 POOL_BLOCK_BYTES =
        Nvn::SparsePagePool::PageBytes * Nvn::SparsePagePool::PagesPerBlock;
    const u64 reserved_limit =
        (tail_backing + POOL_BLOCK_BYTES - 1) / POOL_BLOCK_BYTES * POOL_BLOCK_BYTES +
        POOL_BLOCK_BYTES * Nvn::SparsePagePool::CachedBlockLimit;
    const u64 reserved_after_unmap = impl.sparse_pages->ReservedBytes();
    CHECK(reserved_after_unmap <= reserved_limit);
    CHECK(reserved_after_unmap < 32_MiB);
    // Refill after reclamation: page ids in active tail blocks must remain valid, and removed
    // allocation slots must be safe to reuse without growing the block table indefinitely.
    constexpr u32 REFILL_TILES = 128;
    for (u32 id = 0; id < REFILL_TILES; ++id) {
        const u64 offset = layout.TileOffset(id);
        sparse.Remap(offset, offset + Nvn::SPARSE_TILE_BYTES, [](u64, u64) { return true; });
    }
    uploads = {};
    REQUIRE(sparse.Prepare(impl, storage, uploads));
    CHECK(sparse.StarvedUnits() == 0);
    CHECK(sparse.BoundBytes() == tail_backing + REFILL_TILES * 256_KiB);
    sparse.Remap(0, storage.layout.size, [](u64, u64) { return false; });
    uploads = {};
    REQUIRE(sparse.Prepare(impl, storage, uploads));
    CHECK(impl.sparse_pages->ReservedBytes() <= reserved_limit);
    CHECK(sparse.BoundBytes() == tail_backing);
}

TEST_CASE("MKW BC6H lightmaps keep high layers and GPU-written unmapped coarse levels",
          "[nvn][mkw][sparse][mkw-lighting]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan || !vulkan->device->IsOptimalBcnSupported() ||
        !vulkan->device->IsSparseResidencyImageSupported(VK_IMAGE_TYPE_2D) ||
        !vulkan->device->CanUseUnbackedSparseImages()) {
        SKIP("requires native BC6H sparse residency");
    }
    constexpr u64 PHYSICAL_POOL = 5;
    constexpr u64 VIRTUAL_POOL = 6;
    constexpr u64 PHYSICAL_VA = 0x3000'0000;
    constexpr u64 VIRTUAL_VA = 0x7D00'0000'0000'0000ULL;
    constexpr u32 FORMAT = 0x50;
    Rig rig{*vulkan, true};
    rig.context.SetTitleId(Settings::MarioKartWorldProgramId);
    u8* physical = rig.memory.Add(PHYSICAL_VA, 2 * Nvn::SPARSE_TILE_BYTES);
    rig.context.CreatePool({.serial = PHYSICAL_POOL,
                            .guest_address = PHYSICAL_VA,
                            .size = 2 * Nvn::SPARSE_TILE_BYTES,
                            .flags = POOL_CPU_CACHED});
    rig.context.CreatePool({.serial = VIRTUAL_POOL,
                            .guest_address = VIRTUAL_VA,
                            .size = 8_GiB,
                            .flags = POOL_VIRTUAL});
    auto desc = MakeTexture(10, VIRTUAL_POOL, 0, FORMAT, 8192, 8192);
    desc.target = 4;
    desc.depth = 68;
    desc.levels = 12;
    desc.flags = 0x20;
    rig.context.CreateTexture(desc);
    auto reference =
        MakeStorage(FORMAT, 8192, 8192, 12, PixelFormat::BC6H_UFLOAT, VK_FORMAT_BC6H_UFLOAT_BLOCK);
    reference.desc.target = 4;
    reference.desc.depth = 68;
    reference.desc.flags = 0x20;
    reference.layers = 68;
    reference.layout = Nvn::ComputeTextureLayout(reference.desc);
    const auto layout = Nvn::BuildSparseLayout(reference);
    REQUIRE(layout);
    REQUIRE(layout->HasTail());
    REQUIRE(layout->tile_width == layout->tile_height);
    const u64 high_offset = layout->TileOffset(67 * layout->tiles_per_layer);
    REQUIRE(high_offset > 4_GiB);
    std::array<std::vector<u8>, 2> tiles;
    for (u32 tile = 0; tile < tiles.size(); ++tile) {
        auto& blocks = tiles[tile];
        blocks.resize(Nvn::SPARSE_TILE_BYTES);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = static_cast<u8>(i * 37 + (i >> 7) + tile * 61);
        }
        Nvn::Storage single = MakeStorage(FORMAT, layout->tile_width, layout->tile_height, 1,
                                          PixelFormat::BC6H_UFLOAT, VK_FORMAT_BC6H_UFLOAT_BLOCK);
        const auto guest =
            SwizzleLevel0(single, blocks, 16, layout->tile_width / 4, layout->tile_height / 4);
        REQUIRE(guest.size() == Nvn::SPARSE_TILE_BYTES);
        std::memcpy(physical + tile * Nvn::SPARSE_TILE_BYTES, guest.data(), guest.size());
    }
    rig.context.MapVirtual(VIRTUAL_POOL, {{0, PHYSICAL_POOL, 0, Nvn::SPARSE_TILE_BYTES},
                                          {high_offset, PHYSICAL_POOL, Nvn::SPARSE_TILE_BYTES,
                                           Nvn::SPARSE_TILE_BYTES}});
    const auto read = [&](u32 layer, u32 level, u32 size) {
        const size_t bytes = static_cast<size_t>((size + 3) / 4) * ((size + 3) / 4) * 16;
        std::memset(rig.readback, 0xCD, bytes);
        auto copy = CopyToBuffer(Rig::BUFFER_POOL_VA, 10, size, size);
        copy.texture.layer = layer;
        copy.texture.level = level;
        std::vector<u64> list;
        Push(list, VC::Op::CopyTextureToBuffer, copy);
        Run(rig.context, rig.seq, std::move(list));
        REQUIRE(rig.context.Stats().ops_failed == 0);
        REQUIRE(rig.context.Stats().ops_unsupported == 0);
        return std::vector<u8>(rig.readback, rig.readback + bytes);
    };
    CHECK(read(0, 0, layout->tile_width) == tiles[0]);
    CHECK(read(67, 0, layout->tile_width) == tiles[1]);

    // Real backing for unmapped coarse mips must start deterministic, including on a reused
    // sparse allocation. Later reads must retain the GPU copies rather than clearing again.
    for (const u32 mip : {static_cast<u32>(layout->level.size()), 11u}) {
        const u32 size = (std::max)(8192u >> mip, 1u);
        const size_t bytes = static_cast<size_t>((size + 3) / 4) * ((size + 3) / 4) * 16;
        CHECK(read(67, mip, size) == std::vector<u8>(bytes, 0));
    }

    // The game copies coarse atlas texels without ever mapping their guest tail bytes.
    // Exercise both the first coarse level and the final mip, in the last array layer.
    for (const u32 mip : {static_cast<u32>(layout->level.size()), 11u}) {
        const u32 size = (std::max)(8192u >> mip, 1u);
        const size_t bytes = static_cast<size_t>((size + 3) / 4) * ((size + 3) / 4) * 16;
        std::vector<u8> blocks(bytes);
        for (size_t i = 0; i < bytes; ++i) {
            blocks[i] = static_cast<u8>(i * 29 + mip * 17);
        }
        Nvn::Storage single = MakeStorage(FORMAT, size, size, 1, PixelFormat::BC6H_UFLOAT,
                                          VK_FORMAT_BC6H_UFLOAT_BLOCK);
        const auto guest = SwizzleLevel0(single, blocks, 16, (size + 3) / 4, (size + 3) / 4);
        const u64 source_offset = mip * 64_KiB;
        std::memcpy(rig.texture_bytes + source_offset, guest.data(), guest.size());
        rig.context.CreateTexture(
            MakeTexture(20 + mip, Rig::TEXTURE_POOL, source_offset, FORMAT, size, size));
        VC::CopyTextureToTexturePayload copy{};
        copy.src = {.serial = 20 + mip};
        copy.dst = {.serial = 10, .level = mip, .layer = 67};
        copy.src_region =
            copy.dst_region = {0, 0, 0, static_cast<s32>(size), static_cast<s32>(size), 1};
        std::vector<u64> list;
        Push(list, VC::Op::CopyTextureToTexture, copy);
        Run(rig.context, rig.seq, std::move(list));
        CHECK(read(67, mip, size) == blocks);
        // Revisit without a map: an upload of unmapped guest bytes must not erase it.
        CHECK(read(67, mip, size) == blocks);
    }
    rig.context.MapVirtual(VIRTUAL_POOL, {{high_offset, 0, 0, Nvn::SPARSE_TILE_BYTES}});
    CHECK(read(67, 0, layout->tile_width) == std::vector<u8>(Nvn::SPARSE_TILE_BYTES, 0));
    CHECK(read(0, 0, layout->tile_width) == tiles[0]);
}

TEST_CASE("nvn memory: FlushTextureToGuest writes a GPU-written texture into its storage",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool allow_import = GENERATE(true, false);
    constexpr u32 SIZE = 32;
    Rig rig{*vulkan, allow_import};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0x4000, NVN_FORMAT_RGBA8, SIZE, SIZE));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, SIZE, SIZE, 0.0f, 0.0f, 1.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));

    REQUIRE(rig.context.FlushTextureToGuest(10));
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_RGBA8, SIZE, SIZE, 1,
                                         PixelFormat::A8B8G8R8_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    std::vector<u8> blue(static_cast<size_t>(SIZE) * SIZE * 4);
    for (size_t i = 0; i < blue.size(); i += 4) {
        blue[i + 2] = 255;
        blue[i + 3] = 255;
    }
    const std::vector<u8> expected = SwizzleLevel0(reference, blue, 4, SIZE, SIZE);
    REQUIRE(std::memcmp(rig.texture_bytes + 0x4000, expected.data(), expected.size()) == 0);
    REQUIRE(rig.context.Stats().texels_flushed == 1);
    // The guest bytes are the texture now: it reads them back unchanged.
    REQUIRE(rig.Read(10, SIZE, SIZE) == blue);
    // A texture nobody wrote needs no flush.
    rig.context.CreateTexture(MakeTexture(11, Rig::TEXTURE_POOL, 0x20000, NVN_FORMAT_RGBA8, SIZE, SIZE));
    REQUIRE(rig.context.FlushTextureToGuest(11));
    REQUIRE_FALSE(rig.context.FlushTextureToGuest(999));
}

TEST_CASE("nvn memory: compressed textures upload and copy without unsupported ops", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u32 SIZE = 16;
    Rig rig{*vulkan, true};
    Nvn::Storage reference = MakeStorage(NVN_FORMAT_ASTC_4X4, SIZE, SIZE, 1,
                                         PixelFormat::ASTC_2D_4X4_UNORM,
                                         VK_FORMAT_A8B8G8R8_UNORM_PACK32, true);
    std::vector<u8> blocks;
    for (u32 i = 0; i < 16; ++i) {
        const auto block = VoidBlock(static_cast<u16>(i * 4096), 0x4000, 0xC000, 0xFFFF);
        blocks.insert(blocks.end(), block.begin(), block.end());
    }
    const std::vector<u8> guest = SwizzleLevel0(reference, blocks, 16, 4, 4);
    std::memcpy(rig.texture_bytes, guest.data(), guest.size());
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_ASTC_4X4, SIZE, SIZE));
    rig.context.CreateTexture(
        MakeTexture(11, Rig::TEXTURE_POOL, 0x40000, NVN_FORMAT_ASTC_4X4, SIZE, SIZE));
    std::vector<u64> list;
    VC::CopyTextureToTexturePayload copy{};
    copy.src = {.serial = 10, .level = 0, .layer = 0};
    copy.dst = {.serial = 11, .level = 0, .layer = 0};
    copy.src_region = {0, 0, 0, SIZE, SIZE, 1};
    copy.dst_region = {0, 0, 0, SIZE, SIZE, 1};
    Push(list, VC::Op::CopyTextureToTexture, copy);
    Run(rig.context, rig.seq, std::move(list));
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_unsupported == 0);
    REQUIRE(stats.ops_failed == 0);
    REQUIRE(stats.textures_transcode_unsupported == 0);
    if (vulkan->device->IsOptimalAstcSupported()) {
        REQUIRE(stats.decode_bytes == 0);
    } else {
        REQUIRE(stats.decode_bytes == SIZE * SIZE * 4);
        WARN("[nvn] ASTC decoded on upload: " << stats.decode_bytes << " bytes in "
                                              << stats.decode_ns / 1000 << " us");
    }
}

TEST_CASE("nvn memory: MKW copies decoded ASTC 8x8 texels between UNORM and sRGB", "[nvn][mkw]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (vulkan->device->IsOptimalAstcSupported()) {
        SKIP("this exercises the decoded ASTC path");
    }
    const bool mario_kart_world = GENERATE(false, true);
    constexpr u32 SIZE = 16;
    constexpr s32 ASTC_8X8 = 0x80;
    constexpr s32 ASTC_8X8_SRGB = 0x8e;
    // Match the host-format override installed by the frontend for VTX-HLE.
    Settings::SetAstcRecompressionOverride(Settings::AstcRecompression::Uncompressed);
    SCOPE_EXIT {
        Settings::SetAstcRecompressionOverride(std::nullopt);
    };
    Rig rig{*vulkan, true};
    rig.context.SetTitleId(mario_kart_world ? Settings::MarioKartWorldProgramId
                                            : Settings::OcarinaOfTimeProgramId);
    const Nvn::Storage reference =
        MakeStorage(ASTC_8X8, SIZE, SIZE, 1, PixelFormat::ASTC_2D_8X8_UNORM,
                    VK_FORMAT_A8B8G8R8_UNORM_PACK32, true);
    const auto block = VoidBlock(0x2000, 0x4000, 0xc000, 0xffff);
    std::array<u8, 64> blocks;
    for (size_t offset = 0; offset < blocks.size(); offset += block.size()) {
        std::ranges::copy(block, blocks.begin() + offset);
    }
    const auto bytes = SwizzleLevel0(reference, blocks, block.size(), 2, 2);
    std::memcpy(rig.texture_bytes, bytes.data(), bytes.size());
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, ASTC_8X8, SIZE, SIZE));
    rig.context.CreateTexture(
        MakeTexture(11, Rig::TEXTURE_POOL, 0x40000, ASTC_8X8_SRGB, SIZE, SIZE));
    std::vector<u64> list;
    Push(list, VC::Op::CopyTextureToTexture,
         VC::CopyTextureToTexturePayload{
             .src = {.serial = 10},
             .src_region = {0, 0, 0, SIZE, SIZE, 1},
             .dst = {.serial = 11},
             .dst_region = {0, 0, 0, SIZE, SIZE, 1},
         });
    Run(rig.context, rig.seq, std::move(list));
    CHECK(rig.context.Stats().ops_failed == 0);
    CHECK(rig.context.Stats().ops_unsupported == (mario_kart_world ? 0u : 1u));
}

TEST_CASE("nvn memory: spills count device-local requests placed in host memory types", "[nvn]") {
    Vulkan::AllocationSpills spills;
    using Vulkan::AllocationKind;
    // Device-local as asked, or host memory asked for: nothing spilled.
    CHECK_FALSE(Vulkan::CountSpill(spills, AllocationKind::PoolBuffer, true,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1_MiB));
    CHECK_FALSE(Vulkan::CountSpill(spills, AllocationKind::Staging, false,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1_MiB));
    // Device-local asked for, a host-visible system memory type taken.
    CHECK(Vulkan::CountSpill(spills, AllocationKind::StorageImage, true,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             3_MiB));
    CHECK(Vulkan::CountSpill(spills, AllocationKind::StorageImage, true, 0, 1_MiB));
    const auto index = [](AllocationKind kind) { return static_cast<size_t>(kind); };
    CHECK(spills.count[index(AllocationKind::StorageImage)].load() == 2);
    CHECK(spills.bytes[index(AllocationKind::StorageImage)].load() == 4_MiB);
    CHECK(spills.count[index(AllocationKind::PoolBuffer)].load() == 0);
    CHECK(spills.count[index(AllocationKind::Staging)].load() == 0);
    CHECK(std::string{Vulkan::AllocationKindName(AllocationKind::Shadow)} == "shadows");
}

TEST_CASE("nvn memory: the memory line splits storages by the recorder's holder report", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const bool strict = GENERATE(false, true);
    constexpr u32 SIZE = 32;
    Rig rig{*vulkan, true, POOL_CPU_CACHED, {}, {}, {}, strict};
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, NVN_FORMAT_RGBA8, SIZE, SIZE));
    rig.context.CreateTexture(
        MakeTexture(11, Rig::TEXTURE_POOL, 0x40000, NVN_FORMAT_RGBA8, SIZE, SIZE));
    std::vector<u64> list;
    Push(list, VC::Op::ClearTexture, ClearColor(10, SIZE, SIZE, 1.0f, 0.0f, 0.0f, 1.0f), 0xF);
    Push(list, VC::Op::ClearTexture, ClearColor(11, SIZE, SIZE, 0.0f, 1.0f, 0.0f, 1.0f), 0xF);
    Run(rig.context, rig.seq, std::move(list));

    VC::TextureHolderReport report;
    report.serials = {{11, VC::TextureHolder::Reinitialized}};
    rig.context.NoteTextureHolders(std::move(report));
    Run(rig.context, rig.seq, {}); // ordered after the report

    const std::string line = rig.context.DescribeMemory();
    INFO(line);
    CHECK(line.find("game 0 (1)") != std::string::npos);
    CHECK(line.find("re-initialised 0 (1)") != std::string::npos);
    CHECK(line.find("synthetic lists 0 (0)") != std::string::npos);
    CHECK(line.find(strict ? "strict on" : "strict off") != std::string::npos);
    CHECK(line.find("heaps budget/usage") != std::string::npos);
    CHECK(line.find("storage_images 0/0") != std::string::npos);
    const Nvn::ContextStats stats = rig.context.Stats();
    REQUIRE(stats.ops_failed == 0);
}

namespace {

class FilterReadbackTarget final : public Nvn::PresentTarget {
public:
    FilterReadbackTarget(HeadlessVulkan& vk, VkFormat format, u32 width = 79, u32 height = 57)
        : vulkan{vk}, format{format}, width{width}, height{height} {}

    void Attach(Vulkan::MemoryAllocator& allocator) override {
        using namespace Vulkan;
        const auto& device = *vulkan.device;
        frame.width = width;
        frame.height = height;
        frame.format = format;
        frame.image = allocator.CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {frame.width, frame.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        frame.image_view = CreateWrappedImageView(device, frame.image, format);
        pass = CreateWrappedRenderPass(device, format, VK_IMAGE_LAYOUT_UNDEFINED);
        frame.framebuffer =
            CreateWrappedFramebuffer(device, pass, frame.image_view, {frame.width, frame.height});
        frame.render_ready = device.GetLogical().CreateSemaphore({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        });
        download = allocator.CreateBuffer(
            {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = frame.width * frame.height * 4,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::Download);
    }

    Vulkan::Frame* AcquireFrame() override {
        return &frame;
    }

    void ReleaseFrame(Vulkan::Frame*) override {
        ++returned;
    }

    u32 ReturnedFrames() const {
        return returned;
    }

    void PresentFrame(Vulkan::Frame*) override {
        using namespace Vulkan;
        const auto& device = *vulkan.device;
        auto pool = device.GetLogical().CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = device.GetGraphicsFamily(),
        });
        auto buffers = pool.Allocate(1);
        const VkCommandBuffer raw = buffers[0];
        vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
        cmd.Begin({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT});
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            barrier, {}, {});
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {frame.width, frame.height, 1},
        };
        cmd.CopyImageToBuffer(*frame.image, VK_IMAGE_LAYOUT_GENERAL, *download, region);
        cmd.End();
        auto fence =
            device.GetLogical().CreateFence({.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO});
        const VkSemaphore ready = *frame.render_ready;
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &ready,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &raw,
        };
        {
            std::scoped_lock lock{vulkan.submit_mutex};
            vk::Check(device.GetGraphicsQueue().Submit(submit, *fence));
        }
        vk::Check(fence.Wait());
        download.Invalidate();
        const auto bytes = download.Mapped();
        result.set_value(
            std::vector<u8>(bytes.begin(), bytes.begin() + frame.width * frame.height * 4));
    }

    void Detach() override {
        download = {};
        frame = {};
        pass = {};
    }

    s32 FrameFormat() const override {
        return static_cast<s32>(format);
    }

    std::future<std::vector<u8>> Next() {
        result = {};
        return result.get_future();
    }

private:
    HeadlessVulkan& vulkan;
    VkFormat format;
    u32 width;
    u32 height;
    Vulkan::Frame frame{};
    Vulkan::vk::RenderPass pass;
    Vulkan::vk::Buffer download;
    std::promise<std::vector<u8>> result;
    u32 returned{};
};

} // namespace

TEST_CASE("NVN present keeps the previous picture until the new source has pixels",
          "[nvn][present-filter]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    auto target = std::make_unique<FilterReadbackTarget>(*vulkan, VK_FORMAT_R8G8B8A8_UNORM);
    auto* readback = target.get();
    Rig rig(*vulkan, false, POOL_CPU_CACHED, {}, {}, {}, {}, std::move(target));
    constexpr u64 LATE_POOL = 3;
    constexpr u64 LATE_VA = 0x3500'0000;
    constexpr u32 WIDTH = 32;
    constexpr u32 HEIGHT = 24;
    rig.context.CreatePool({.serial = LATE_POOL, .guest_address = LATE_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});
    rig.context.CreateTexture(MakeTexture(10, LATE_POOL, 0, NVN_FORMAT_RGBA8, WIDTH, HEIGHT));
    const auto present = [&] {
        rig.context.Present({.seq = ++rig.seq, .queue = 0, .texture_serial = 10,
                             .present_interval = 1});
    };
    auto missing = readback->Next();
    present();
    REQUIRE(rig.context.Wait(rig.seq, 20s));
    CHECK(readback->ReturnedFrames() == 1);
    CHECK(missing.wait_for(0ms) == std::future_status::timeout);

    const auto source = MakeStorage(NVN_FORMAT_RGBA8, WIDTH, HEIGHT, 1,
                                    PixelFormat::A8B8G8R8_UNORM, VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const auto rgb = Gradient(WIDTH, HEIGHT, 63);
    const auto bytes = SwizzleLevel0(source, rgb, 4, WIDTH, HEIGHT);
    std::memcpy(rig.memory.Add(LATE_VA, 64_KiB), bytes.data(), bytes.size());
    auto ready = readback->Next();
    present();
    REQUIRE(ready.wait_for(20s) == std::future_status::ready);
    const auto pixels = ready.get();
    CHECK_FALSE(std::ranges::all_of(pixels, [](u8 value) { return value == 0 || value == 255; }));
    CHECK(rig.context.Wait(rig.seq, 20s));
    CHECK(readback->ReturnedFrames() == 1);
}

TEST_CASE("NVN completed bulk uploads reuse staging without keeping the old bytes",
          "[nvn][staging-reuse]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Rig rig(*vulkan, false);
    constexpr u32 WIDTH = 32;
    constexpr u32 HEIGHT = 24;
    const auto source = MakeStorage(NVN_FORMAT_RGBA8, WIDTH, HEIGHT, 1,
                                    PixelFormat::A8B8G8R8_UNORM, VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    for (const u32 index : {0u, 1u}) {
        const u64 offset = index * 64_KiB;
        const auto expected = Gradient(WIDTH, HEIGHT, static_cast<u8>(63 + index));
        const auto bytes = SwizzleLevel0(source, expected, 4, WIDTH, HEIGHT);
        std::memcpy(rig.texture_bytes + offset, bytes.data(), bytes.size());
        rig.context.CreateTexture(
            MakeTexture(10 + index, Rig::TEXTURE_POOL, offset, NVN_FORMAT_RGBA8, WIDTH, HEIGHT));
        REQUIRE(rig.Read(10 + index, WIDTH, HEIGHT) == expected);
    }
    CHECK(rig.context.Stats().staging_chunks_reused == 1);
}

TEST_CASE("NVN window filters change actual presented pixels on every title and colour format",
          "[nvn][present-filter]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const auto format = GENERATE(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB);
    const auto title =
        GENERATE(Settings::MarioKartWorldProgramId, Settings::OcarinaOfTimeProgramId);
    const u32 source_format = GENERATE(0x25u, 0x38u);
    const auto saved_filter = Settings::values.scaling_filter.GetValue();
    SCOPE_EXIT {
        Settings::values.scaling_filter.SetValue(saved_filter);
    };
    auto target = std::make_unique<FilterReadbackTarget>(*vulkan, format);
    auto* readback = target.get();
    Rig rig(*vulkan, false, POOL_CPU_CACHED, {}, {}, {}, {}, std::move(target));
    rig.context.SetTitleId(title);
    constexpr u32 WIDTH = 32;
    constexpr u32 HEIGHT = 24;
    auto source = MakeStorage(NVN_FORMAT_RGBA8, WIDTH, HEIGHT, 1, PixelFormat::A8B8G8R8_UNORM,
                              VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    const auto bytes = SwizzleLevel0(source, Gradient(WIDTH, HEIGHT, 63), 4, WIDTH, HEIGHT);
    std::memcpy(rig.texture_bytes, bytes.data(), bytes.size());
    rig.context.CreateTexture(MakeTexture(10, Rig::TEXTURE_POOL, 0, source_format, WIDTH, HEIGHT));
    REQUIRE(rig.Read(10, WIDTH, HEIGHT).size() == WIDTH * HEIGHT * 4);
    const auto capture = [&](Settings::ScalingFilter filter) {
        Settings::values.scaling_filter.SetValue(filter);
        auto future = readback->Next();
        rig.context.Present({.seq = ++rig.seq,
                             .queue = 0,
                             .texture_serial = 10,
                             .window_slot = 0,
                             .present_interval = 1,
                             .crop = {.x = 2, .y = 1, .width = 28, .height = 20}});
        REQUIRE(future.wait_for(20s) == std::future_status::ready);
        auto result = future.get();
        REQUIRE(rig.context.Wait(rig.seq, 20s));
        REQUIRE(rig.context.Stats().ops_failed == 0);
        return result;
    };
    using Settings::ScalingFilter;
    const auto nearest = capture(ScalingFilter::NearestNeighbor);
    const auto bilinear = capture(ScalingFilter::Bilinear);
    REQUIRE(nearest != bilinear);
    for (const auto filter :
         {ScalingFilter::Bicubic, ScalingFilter::ZeroTangent, ScalingFilter::BSpline,
          ScalingFilter::Mitchell, ScalingFilter::Spline1, ScalingFilter::Gaussian,
          ScalingFilter::Lanczos, ScalingFilter::ScaleForce, ScalingFilter::Area,
          ScalingFilter::Mmpx, ScalingFilter::Fsr, ScalingFilter::Sgsr, ScalingFilter::SgsrEdge,
          ScalingFilter::Fsr3}) {
        INFO("filter " << static_cast<u32>(filter) << " format " << format << " title " << title);
        const auto filtered = capture(filter);
        CHECK(filtered != bilinear);
        CHECK(filtered.size() == bilinear.size());
        // Re-selecting a cached filter must keep producing the same pixels.
        CHECK(capture(filter) == filtered);
    }
    CHECK(capture(ScalingFilter::NearestNeighbor) == nearest);
    CHECK(capture(ScalingFilter::Bilinear) == bilinear);
}

TEST_CASE("MKW window filters process a saved gameplay image at display resolutions",
          "[nvn][present-filter][.mkw-filter-capture]") {
    // External local captures stay outside the source tree: two u32 dimensions, then RGBA8.
    const char* input = std::getenv("EDEN_NX2_TEST_FILTER_INPUT_RGBA");
    const char* output = std::getenv("EDEN_NX2_TEST_FILTER_OUTPUT");
    if (!input || !output) {
        SKIP("set EDEN_NX2_TEST_FILTER_INPUT_RGBA and EDEN_NX2_TEST_FILTER_OUTPUT");
    }
    std::ifstream in(input, std::ios::binary);
    REQUIRE(in.good());
    u32 width{}, height{};
    in.read(reinterpret_cast<char*>(&width), sizeof(width));
    in.read(reinterpret_cast<char*>(&height), sizeof(height));
    REQUIRE(width == 1280);
    REQUIRE(height == 720);
    std::vector<u8> pixels(static_cast<size_t>(width) * height * 4);
    in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    REQUIRE(in.good());
    auto vulkan = MakeHeadlessVulkan();
    REQUIRE(vulkan);
    const auto saved_filter = Settings::values.scaling_filter.GetValue();
    const auto saved_aa = Settings::values.anti_aliasing.GetValue();
    SCOPE_EXIT {
        Settings::values.scaling_filter.SetValue(saved_filter);
        Settings::values.anti_aliasing.SetValue(saved_aa);
    };
    // The user's FXAA selection must not suppress the separately selected window filter.
    Settings::values.anti_aliasing.SetValue(Settings::AntiAliasing::Fxaa);
    const u32 source_format = GENERATE(0x25u, 0x38u);
    const std::filesystem::path directory =
        std::filesystem::path(output) / (source_format == 0x25 ? "unorm" : "srgb");
    std::filesystem::create_directories(directory);
    std::string csv =
        "width,height,filter,changed_pixels,changed_over_1,pixels,mean_abs_rgb,max_rgb_delta\n";
    using Settings::ScalingFilter;
    const std::array filters{ScalingFilter::NearestNeighbor,
                             ScalingFilter::Bilinear,
                             ScalingFilter::Bicubic,
                             ScalingFilter::ZeroTangent,
                             ScalingFilter::BSpline,
                             ScalingFilter::Mitchell,
                             ScalingFilter::Spline1,
                             ScalingFilter::Gaussian,
                             ScalingFilter::Lanczos,
                             ScalingFilter::ScaleForce,
                             ScalingFilter::Area,
                             ScalingFilter::Mmpx,
                             ScalingFilter::Fsr,
                             ScalingFilter::Sgsr,
                             ScalingFilter::SgsrEdge,
                             ScalingFilter::Fsr3};
    struct Size {
        u32 width;
        u32 height;
    };
    for (const auto size : {Size{1280, 720}, Size{1920, 1080}, Size{2560, 1440}}) {
        auto target = std::make_unique<FilterReadbackTarget>(*vulkan, VK_FORMAT_B8G8R8A8_UNORM,
                                                             size.width, size.height);
        auto* readback = target.get();
        Rig rig(*vulkan, false, POOL_CPU_CACHED, {}, {}, {}, {}, std::move(target));
        rig.context.SetTitleId(Settings::MarioKartWorldProgramId);
        auto source = MakeStorage(NVN_FORMAT_RGBA8, width, height, 1, PixelFormat::A8B8G8R8_UNORM,
                                  VK_FORMAT_A8B8G8R8_UNORM_PACK32);
        const auto bytes = SwizzleLevel0(source, pixels, 4, width, height);
        REQUIRE(bytes.size() <= 4_MiB);
        std::memcpy(rig.texture_bytes, bytes.data(), bytes.size());
        // MKW reported source format 51 (packed RGBA UNORM), output 44 (BGRA UNORM).
        // The second variant also exercises the sRGB-source raw-bit presentation path.
        rig.context.CreateTexture(
            MakeTexture(10, Rig::TEXTURE_POOL, 0, source_format, width, height));
        REQUIRE(rig.Read(10, 1, 1).size() == 4); // upload the source before it becomes displayable
        const auto capture = [&](ScalingFilter filter) {
            Settings::values.scaling_filter.SetValue(filter);
            auto future = readback->Next();
            rig.context.Present({.seq = ++rig.seq,
                                 .queue = 0,
                                 .texture_serial = 10,
                                 .window_slot = 0,
                                 .present_interval = 1});
            REQUIRE(future.wait_for(30s) == std::future_status::ready);
            auto result = future.get();
            REQUIRE(rig.context.Wait(rig.seq, 30s));
            REQUIRE(rig.context.Stats().ops_failed == 0);
            for (size_t i = 0; i < result.size(); i += 4) {
                std::swap(result[i], result[i + 2]); // BGRA frame -> RGBA artifact
            }
            return result;
        };
        const auto bilinear = capture(ScalingFilter::Bilinear);
        for (const auto filter : filters) {
            INFO("output " << size.width << "x" << size.height << ", filter "
                           << static_cast<u32>(filter));
            const auto filtered = capture(filter);
            REQUIRE(filtered.size() == bilinear.size());
            CHECK(capture(filter) == filtered); // switching through the cached pass is stable
            u64 changed = 0, changed_over_1 = 0, sum = 0;
            u32 maximum = 0;
            std::array<s64, 3> drift{};
            for (size_t i = 0; i < filtered.size(); i += 4) {
                u32 pixel_max = 0;
                for (size_t c = 0; c < 3; ++c) {
                    const s32 delta = s32{filtered[i + c]} - s32{bilinear[i + c]};
                    const u32 magnitude = static_cast<u32>(std::abs(delta));
                    sum += magnitude;
                    drift[c] += delta;
                    pixel_max = (std::max)(pixel_max, magnitude);
                }
                changed += pixel_max > 0;
                changed_over_1 += pixel_max > 1;
                maximum = (std::max)(maximum, pixel_max);
            }
            const u64 count = u64{size.width} * size.height;
            const double mean = static_cast<double>(sum) / static_cast<double>(count * 3);
            csv += fmt::format("{},{},{},{},{},{},{:.6f},{}\n", size.width, size.height,
                               static_cast<u32>(filter), changed, changed_over_1, count, mean,
                               maximum);
            // A filter should not introduce a large colour/transfer-function shift.
            for (const auto channel : drift) {
                CHECK(std::abs(channel) < static_cast<s64>(count * 3));
            }
            if (filter == ScalingFilter::ScaleForce && size.width > width) {
                CHECK(changed_over_1 > count / 100); // visible-scale changes, not just rounding
                WARN("ScaleForce "
                     << size.width << "x" << size.height << ": "
                     << (100.0 * static_cast<double>(changed_over_1) / static_cast<double>(count))
                     << "% pixels differ by >1 from bilinear; mean RGB delta " << mean);
            }
            if (size.width == width &&
                (filter == ScalingFilter::NearestNeighbor ||
                 (filter == ScalingFilter::ScaleForce && source_format == 0x25))) {
                u32 identity_error = 0;
                for (size_t i = 0; i < pixels.size(); i += 4) {
                    for (size_t c = 0; c < 3; ++c) {
                        identity_error =
                            (std::max)(identity_error,
                                       static_cast<u32>(
                                           std::abs(s32{pixels[i + c]} - s32{filtered[i + c]})));
                    }
                }
                CHECK(identity_error <= 1); // orientation, channel order, crop and gamma
            }
            REQUIRE(Common::DiagRequest::WritePng(directory / fmt::format("{}x{}-filter-{}.png",
                                                                          size.width, size.height,
                                                                          static_cast<u32>(filter)),
                                                  size.width, size.height, filtered));
        }
    }
    REQUIRE(Common::DiagRequest::WriteTextFile(directory / "metrics.csv", csv));
}

TEST_CASE("MKW buffer lookup cache follows pool replacement and virtual remaps", "[nvn][mkw]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    auto impl =
        std::make_unique<Nvn::Context::Impl>(*vulkan->device, vulkan->submit_mutex, nullptr,
                                             Nvn::ContextOptions{.background_backing = false});
    impl->mario_kart_world = GENERATE(false, true);
    impl->CreatePoolNow({.serial = 1, .guest_address = 0x10000, .size = 0x10000});
    impl->CreatePoolNow({.serial = 2, .guest_address = 0x30000, .size = 0x10000});
    impl->CreatePoolNow(
        {.serial = 3, .guest_address = 0x50000, .size = 0x10000, .flags = POOL_VIRTUAL});
    const auto resolve = [&](u64 address, u64 size, u64 serial, u64 expected_offset) {
        Nvn::Pool* pool = nullptr;
        u64 offset = ~u64{0};
        REQUIRE(impl->ResolveBuffer(address, size, pool, offset));
        REQUIRE(pool->serial == serial);
        REQUIRE(offset == expected_offset);
    };
    resolve(0x10120, 64, 1, 0x120);
    resolve(0x10120, 64, 1, 0x120);
    CHECK(impl->mkw_resolve_hits == (impl->mario_kart_world ? 1u : 0u));
    impl->CreatePoolNow({.serial = 4, .guest_address = 0x10100, .size = 0x100});
    resolve(0x10120, 64, 4, 0x20);
    impl->DestroyNow(4);
    resolve(0x10120, 64, 1, 0x120);
    impl->MapVirtualNow(3, {{0, 1, 0x200, 0x1000}});
    resolve(0x50040, 64, 1, 0x240);
    resolve(0x50040, 64, 1, 0x240);
    impl->MapVirtualNow(3, {{0, 2, 0x400, 0x1000}});
    resolve(0x50040, 64, 2, 0x440);
    impl->MapVirtualNow(3, {{0, 0, 0, 0x1000}});
    Nvn::Pool* pool = nullptr;
    u64 offset = 0;
    REQUIRE_FALSE(impl->ResolveBuffer(0x50040, 64, pool, offset));
    REQUIRE_FALSE(impl->ResolveBuffer(0x3fff0, 32, pool, offset));
    resolve(0x3fff0, 16, 2, 0xfff0);
}

TEST_CASE("SparsePagePool tracks stale shared pages, IsShared, and full-budget key hits",
          "[nvn][sparse]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    Nvn::Context::Impl impl(*vulkan->device, vulkan->submit_mutex, nullptr,
                            {.background_backing = false});
    Nvn::SparsePagePool pool(*impl.allocator, Nvn::SparsePagePool::PageBytes);
    const Nvn::SparseAliasKey key_a{0x1000'0000ULL, 0xABCDULL};
    const Nvn::SparseAliasKey key_b{0x2000'0000ULL, 0xABCDULL};
    bool reused = false;
    const u32 page = pool.AcquireShared(key_a, ~0u, 1, reused);
    REQUIRE(page != Nvn::SparsePagePool::NoPage);
    CHECK_FALSE(reused);
    CHECK(pool.IsShared(page));
    CHECK_FALSE(pool.HasRoom());

    CHECK(pool.AcquireShared(key_a, ~0u, 2, reused, false) == page);
    CHECK(reused);
    pool.Free(page);

    CHECK(pool.AcquireShared(key_b, ~0u, 2, reused, false) == Nvn::SparsePagePool::NoPage);
    CHECK_FALSE(reused);

    pool.MarkSharedStale(page);
    CHECK(pool.AcquireShared(key_a, ~0u, 3, reused, false) == page);
    CHECK_FALSE(reused);
    pool.Free(page);

    pool.ClearSharedStale(page);
    CHECK(pool.AcquireShared(key_a, ~0u, 3, reused, false) == page);
    CHECK(reused);
    pool.Free(page);

    CHECK(pool.Allocate(~0u) == Nvn::SparsePagePool::NoPage);

    pool.Free(page);
    CHECK(pool.HasRoom());
    const u32 priv = pool.Allocate(~0u);
    REQUIRE(priv != Nvn::SparsePagePool::NoPage);
    CHECK_FALSE(pool.IsShared(priv));
    pool.Free(priv);
}
