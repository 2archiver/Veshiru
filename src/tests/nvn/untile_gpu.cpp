// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The GPU untile of the NVN replay (nvn_untile_gpu.h) against the CPU untile it replaces
// (FillStorageUpload, Tegra::Texture::UnswizzleTexture with the Switch 2 GOB order): the same
// random block-linear bytes must give the same linear bytes for 2D, 2D array, 3D, mip chains,
// compressed blocks and every texel size (1, 2, 3, 4, 6, 8, 12, 16 bytes). The GPU part needs a
// device (SKIP without).

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_untile_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace {

namespace Nvn = Vulkan::Nvn;
using Nvn::UntileGpu;

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
    } catch (const std::exception&) {
        return nullptr;
    }
}

/// One texture shape: NVN target and format, texel extent, levels.
struct Shape {
    const char* name;
    Nvn::TexTarget target;
    s32 format;
    u32 width;
    u32 height;
    u32 depth; ///< 3D depth or array layers
    u32 levels;
    /// Bytes cut off the end of the storage's guest bytes (the CPU pads the missing GOBs with
    /// zeros; the GPU input holds zeros there).
    u64 truncate{0};
};

/// The storage as CreateTextureNow shapes it (without a VkImage), no decode on upload.
Nvn::Storage MakeStorage(const Shape& shape) {
    Nvn::Storage s;
    s.desc = {.target = static_cast<s32>(shape.target),
              .format = shape.format,
              .flags = 0,
              .samples = 0,
              .width = shape.width,
              .height = shape.height,
              .depth = shape.depth,
              .levels = shape.levels,
              .stride = 0,
              .pkg_block_h = 0};
    s.layout = Nvn::ComputeTextureLayout(s.desc);
    s.host.needs_transcode = false;
    const bool is_3d = shape.target == Nvn::TexTarget::Dim3D;
    s.type = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    s.extent = {shape.width, shape.height, is_3d ? shape.depth : 1u};
    s.levels = shape.levels;
    s.layers = shape.target == Nvn::TexTarget::Dim2DArray ? (std::max)(s.layout.layers, 1u) : 1u;
    return s;
}

constexpr s32 FORMAT_R8 = 0x01;      // 1 byte
constexpr s32 FORMAT_RG8 = 0x05;     // 2 bytes
constexpr s32 FORMAT_RGB8 = 0x19;    // 3 bytes
constexpr s32 FORMAT_RGBA8 = 0x25;   // 4 bytes
constexpr s32 FORMAT_RGB16 = 0x1D;   // 6 bytes
constexpr s32 FORMAT_RGBA16 = 0x16;  // 8 bytes
constexpr s32 FORMAT_RGB32 = 0x22;   // 12 bytes
constexpr s32 FORMAT_RGBA32 = 0x2E;  // 16 bytes
constexpr s32 FORMAT_BC1 = 0x42;     // 8-byte 4x4 blocks
constexpr s32 FORMAT_BC3 = 0x44;     // 16-byte 4x4 blocks

using Nvn::TexTarget;

const std::vector<Shape>& Shapes() {
    static const std::vector<Shape> shapes{
        {"2D R8 odd width, 3 levels", TexTarget::Dim2D, FORMAT_R8, 77, 45, 1, 3},
        {"2D RG8, 2 levels", TexTarget::Dim2D, FORMAT_RG8, 130, 66, 1, 2},
        {"2D RGBA8 full mip chain", TexTarget::Dim2D, FORMAT_RGBA8, 256, 256, 1, 9},
        {"2D RGBA8 tall", TexTarget::Dim2D, FORMAT_RGBA8, 40, 600, 1, 1},
        {"2D array RGBA16, 5 layers, 3 levels", TexTarget::Dim2DArray, FORMAT_RGBA16, 100, 60, 5,
         3},
        {"3D RGBA8, 3 levels", TexTarget::Dim3D, FORMAT_RGBA8, 40, 24, 18, 3},
        {"3D R8 odd sides", TexTarget::Dim3D, FORMAT_R8, 33, 17, 9, 1},
        {"3D RGBA32", TexTarget::Dim3D, FORMAT_RGBA32, 16, 16, 8, 2},
        {"2D BC1, 4 levels", TexTarget::Dim2D, FORMAT_BC1, 200, 120, 1, 4},
        {"2D array BC3, 3 layers, 5 levels", TexTarget::Dim2DArray, FORMAT_BC3, 64, 64, 3, 5},
        {"2D RGB32 (8-byte elements)", TexTarget::Dim2D, FORMAT_RGB32, 30, 20, 1, 1},
        {"2D RGB32 (4-byte elements)", TexTarget::Dim2D, FORMAT_RGB32, 31, 7, 1, 1},
        {"2D RGB16 (2-byte elements)", TexTarget::Dim2D, FORMAT_RGB16, 21, 13, 1, 1},
        {"2D RGB8 (1-byte elements)", TexTarget::Dim2D, FORMAT_RGB8, 15, 15, 1, 1},
        {"2D RGBA8, storage short of its last GOBs", TexTarget::Dim2D, FORMAT_RGBA8, 96, 72, 1, 1,
         1000},
    };
    return shapes;
}

} // Anonymous namespace

TEST_CASE("NVN untile GPU: sizes and element choice follow the CPU untile", "[nvn]") {
    for (const Shape& shape : Shapes()) {
        INFO(shape.name);
        const Nvn::Storage s = MakeStorage(shape);
        const Nvn::UploadPlan plan = Nvn::PlanStorageUpload(s, s.layout.size);
        REQUIRE(plan.unsupported == nullptr);
        REQUIRE(!plan.linear);
        REQUIRE(plan.decode == Nvn::UploadPlan::Decode::None);
        const Nvn::BlockInfo block = Nvn::FormatBlockInfo(shape.format);
        for (const Nvn::UploadPiece& piece : plan.pieces) {
            const UntileGpu::Piece gpu{
                .columns = piece.cols,
                .rows = piece.rows,
                .slices = piece.slices,
                .bytes_per_block = block.bytes,
                .tile_height = piece.tile_h,
                .tile_depth = piece.tile_d,
            };
            REQUIRE(UntileGpu::Fits(gpu));
            CHECK(UntileGpu::InputSize(gpu) ==
                  Tegra::Texture::CalculateSize(true, block.bytes, piece.cols, piece.rows,
                                                piece.slices, piece.tile_h, piece.tile_d));
            CHECK(UntileGpu::OutputSize(gpu) == piece.staging_size);
            const UntileGpu::Params params = UntileGpu::MakeParams(gpu);
            const u32 row_bytes = piece.cols * block.bytes;
            CHECK(params.element_log2 ==
                  (std::min)(4u, static_cast<u32>(std::countr_zero(row_bytes))));
            CHECK((params.elements_x << params.element_log2) == row_bytes);
        }
    }
    // Too big for the shader's 32-bit byte offsets, or empty: the caller untiles on the CPU.
    const UntileGpu::Piece too_big{.columns = 65536, .rows = 65536, .slices = 1,
                                   .bytes_per_block = 16};
    CHECK(!UntileGpu::Fits(too_big));
    const UntileGpu::Piece empty{.columns = 0, .rows = 1, .slices = 1, .bytes_per_block = 4};
    CHECK(!UntileGpu::Fits(empty));
}

TEST_CASE("NVN untile GPU: the shader's linear bytes equal the CPU untile's", "[nvn]") {
    using namespace Vulkan;
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Device& device = *vulkan->device;
    if (!UntileGpu::Supported(device)) {
        SKIP("the device cannot run the GPU untile");
    }

    struct Case {
        const Shape* shape;
        u64 guest_size;
        Nvn::UploadPlan plan;
        std::vector<UntileGpu::Piece> pieces;
        std::vector<u8> expected;
        u64 input_base;
        u64 output_base;
        std::vector<u64> gpu_offsets; ///< per plan piece: where the shader writes it
    };
    // As the replay does (nvn_exec.cpp): each piece's output at an offset the shader can bind,
    // compared with the CPU untile's bytes at its own staging offset.
    const u64 alignment = (std::max)(std::max<u64>(device.GetStorageBufferAlignment(), 16), u64{64});
    const auto align = [alignment](u64 value) {
        return (value + alignment - 1) / alignment * alignment;
    };
    std::mt19937 random(4321);
    std::vector<Case> cases;
    std::vector<u8> input_bytes;
    u64 output_size = 0;
    for (const Shape& shape : Shapes()) {
        const Nvn::Storage s = MakeStorage(shape);
        Case c{.shape = &shape};
        REQUIRE(s.layout.size > shape.truncate);
        c.guest_size = s.layout.size - shape.truncate;
        c.plan = Nvn::PlanStorageUpload(s, c.guest_size);
        REQUIRE(c.plan.unsupported == nullptr);
        REQUIRE(!c.plan.pieces.empty());
        // Random tiled bytes; the input holds zeros past the guest bytes (whole GOBs of the last
        // piece), as the replay's staging copy does.
        std::vector<u8> guest(static_cast<size_t>(c.guest_size));
        for (u8& byte : guest) {
            byte = static_cast<u8>(random());
        }
        c.expected.resize(static_cast<size_t>(c.plan.staging_size));
        Nvn::FillStorageUpload(s, c.plan, guest.data(), guest.size(), c.expected.data());

        c.input_base = (input_bytes.size() + 4095) / 4096 * 4096;
        c.output_base = (output_size + 4095) / 4096 * 4096;
        const Nvn::BlockInfo block = Nvn::FormatBlockInfo(shape.format);
        u64 input_end = c.input_base + guest.size();
        u64 gpu_end = c.output_base;
        for (const Nvn::UploadPiece& piece : c.plan.pieces) {
            gpu_end = align(gpu_end);
            c.gpu_offsets.push_back(gpu_end);
            UntileGpu::Piece gpu{
                .input_offset = c.input_base + piece.guest_offset,
                .output_offset = gpu_end,
                .columns = piece.cols,
                .rows = piece.rows,
                .slices = piece.slices,
                .bytes_per_block = block.bytes,
                .tile_height = piece.tile_h,
                .tile_depth = piece.tile_d,
            };
            input_end = (std::max)(input_end, gpu.input_offset + UntileGpu::InputSize(gpu));
            gpu_end += piece.staging_size;
            c.pieces.push_back(gpu);
        }
        input_bytes.resize(static_cast<size_t>(input_end), 0);
        std::memcpy(input_bytes.data() + c.input_base, guest.data(), guest.size());
        // Whole 32-bit words: the shader's u32 view of the last piece's range.
        output_size = (gpu_end + 3) & ~u64{3};
        cases.push_back(std::move(c));
    }

    VmaAllocator vma = device.CreateThreadSafeAllocator();
    {
        MemoryAllocator allocator(device, vma);
        const UntileGpu untile(device);
        for (const Case& c : cases) {
            for (const UntileGpu::Piece& piece : c.pieces) {
                REQUIRE(piece.input_offset % untile.OffsetAlignment() == 0);
                REQUIRE(piece.output_offset % untile.OffsetAlignment() == 0);
            }
        }
        const auto make_buffer = [&](u64 size, MemoryUsage usage) {
            return allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = size,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                usage);
        };
        vk::Buffer input = make_buffer(input_bytes.size(), MemoryUsage::Upload);
        std::memcpy(input.Mapped().data(), input_bytes.data(), input_bytes.size());
        input.Flush();
        // A marker in every output byte: a byte the shader should write and did not shows up.
        vk::Buffer output = make_buffer(output_size, MemoryUsage::Download);
        std::memset(output.Mapped().data(), 0xA5, static_cast<size_t>(output_size));
        output.Flush();

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
        for (const Case& c : cases) {
            untile.Record(cmd, *input, *output, c.pieces);
        }
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
        output.Invalidate();

        const u8* const actual = output.Mapped().data();
        for (const Case& c : cases) {
            INFO(c.shape->name);
            size_t wrong = 0;
            for (size_t p = 0; p < c.plan.pieces.size(); ++p) {
                const Nvn::UploadPiece& piece = c.plan.pieces[p];
                const u8* got = actual + c.gpu_offsets[p];
                const u8* want = c.expected.data() + piece.staging_offset;
                for (u64 i = 0; i < piece.staging_size; ++i) {
                    if (got[i] != want[i]) {
                        if (wrong < 4) {
                            WARN(c.shape->name << ": level " << piece.level << " layer "
                                               << piece.layer << " byte " << i << ": gpu "
                                               << int(got[i]) << " cpu " << int(want[i]));
                        }
                        ++wrong;
                    }
                }
            }
            CHECK(wrong == 0);
        }
    }
    vmaDestroyAllocator(vma);
}
