// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The GPU ASTC decoder of the NVN replay (nvn_astc_gpu.h) against what the guest's bytes mean:
// block-linear (Switch 2 GOB order) void-extent ASTC blocks, each with its own colour, over several
// layers and levels of a texture whose sides are not block multiples. Needs a GPU (SKIP without).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_astc_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace {

using Vulkan::Nvn::AstcGpuDecoder;

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

/// A void-extent LDR block (spec 'Void-Extent Blocks') of one colour, 16 bits per channel.
std::array<u8, 16> VoidExtentBlock(const std::array<u16, 4>& rgba) {
    u64 low = 0x1FC;                  // block mode: void extent
    low |= u64{3} << 10;              // reserved bits are ones
    low |= u64{0x1FFF} << 12;         // no extent: every coordinate field is all ones
    low |= u64{0x1FFF} << 25;
    low |= u64{0x1FFF} << 38;
    low |= u64{0x1FFF} << 51;
    u64 high = 0;
    for (u32 i = 0; i < 4; ++i) {
        high |= u64{rgba[i]} << (16 * i);
    }
    std::array<u8, 16> block;
    std::memcpy(block.data(), &low, 8);
    std::memcpy(block.data() + 8, &high, 8);
    return block;
}

u8 ToUnorm8(u16 value) {
    return static_cast<u8>(std::lround(static_cast<double>(value) / 65535.0 * 255.0));
}

} // Anonymous namespace

TEST_CASE("NVN ASTC GPU: block-linear blocks decode into the right texels of every level and layer",
          "[nvn]") {
    using namespace Vulkan;
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Device& device = *vulkan->device;
    if (!AstcGpuDecoder::Supported(device)) {
        SKIP("the device cannot run the GPU ASTC decoder");
    }

    // ASTC 4x4 UNORM (NVN format 0x79), 2D array: 70x45 texels is 18x12 blocks, the next levels
    // are not block multiples either, and the layout picks a block height per level.
    constexpr s32 FORMAT = 0x79;
    constexpr u32 WIDTH = 70;
    constexpr u32 HEIGHT = 45;
    constexpr u32 LEVELS = 4;
    constexpr u32 LAYERS = 3;
    Nvn::TextureLayoutDesc desc{
        .target = static_cast<s32>(Nvn::TexTarget::Dim2DArray),
        .format = FORMAT,
        .flags = 0,
        .samples = 0,
        .width = WIDTH,
        .height = HEIGHT,
        .depth = LAYERS,
        .levels = LEVELS,
        .stride = 0,
        .pkg_block_h = 0,
    };
    const Nvn::TextureLayout layout = Nvn::ComputeTextureLayout(desc);
    const Nvn::BlockInfo block = Nvn::FormatBlockInfo(FORMAT);
    REQUIRE(block.bytes == 16);
    REQUIRE(layout.layers == LAYERS);

    // Guest bytes: every block of every level and layer gets its own random colour, swizzled the
    // way the game stores it.
    std::mt19937 random(1234);
    std::vector<u8> guest(static_cast<size_t>(layout.size) + 4096, 0);
    struct Expected {
        u32 level;
        u32 layer;
        u32 width;
        u32 height;
        std::vector<u8> texels; // RGBA8, tightly packed
    };
    std::vector<Expected> expected;
    std::vector<AstcGpuDecoder::Level> levels;
    for (u32 level = 0; level < LEVELS; ++level) {
        const u32 width = (std::max)(WIDTH >> level, 1u);
        const u32 height = (std::max)(HEIGHT >> level, 1u);
        const u32 columns = (width + block.width - 1) / block.width;
        const u32 rows = (height + block.height - 1) / block.height;
        u32 tile_h = 0;
        u32 tile_d = 0;
        Nvn::TextureLevelTiles(desc, level, tile_h, tile_d);
        for (u32 layer = 0; layer < LAYERS; ++layer) {
            std::vector<u8> linear(size_t{columns} * rows * 16);
            Expected texels{level, layer, width, height,
                            std::vector<u8>(size_t{width} * height * 4)};
            for (u32 row = 0; row < rows; ++row) {
                for (u32 column = 0; column < columns; ++column) {
                    const std::array<u16, 4> color{static_cast<u16>(random()),
                                                   static_cast<u16>(random()),
                                                   static_cast<u16>(random()),
                                                   static_cast<u16>(random())};
                    const auto encoded = VoidExtentBlock(color);
                    std::memcpy(&linear[(size_t{row} * columns + column) * 16], encoded.data(), 16);
                    for (u32 y = row * block.height; y < (std::min)((row + 1) * block.height, height);
                         ++y) {
                        for (u32 x = column * block.width;
                             x < (std::min)((column + 1) * block.width, width); ++x) {
                            u8* texel = &texels.texels[(size_t{y} * width + x) * 4];
                            for (u32 c = 0; c < 4; ++c) {
                                texel[c] = ToUnorm8(color[c]);
                            }
                        }
                    }
                }
            }
            const u64 offset = u64{layer} * layout.layer_size + Nvn::TextureLevelOffset(desc, level);
            Tegra::Texture::SwizzleTexture(
                std::span<u8>{guest.data() + offset, guest.size() - static_cast<size_t>(offset)},
                linear, 16, columns, rows, 1, tile_h, tile_d, 0, true);
            expected.push_back(std::move(texels));
        }
        levels.push_back({
            .level = level,
            .layers = LAYERS,
            .columns = columns,
            .rows = rows,
            .offset = Nvn::TextureLevelOffset(desc, level),
            .layer_stride = static_cast<u32>(layout.layer_size),
            .block_width = block.width,
            .block_height = block.height,
            .tile_height = tile_h,
            .tile_depth = tile_d,
        });
    }

    VmaAllocator vma = device.CreateThreadSafeAllocator();
    {
        MemoryAllocator allocator(device, vma);
        const AstcGpuDecoder decoder(device);
        for (const AstcGpuDecoder::Level& level : levels) {
            REQUIRE(level.offset % decoder.OffsetAlignment() == 0);
        }

        vk::Buffer input = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = guest.size(),
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Upload);
        std::memcpy(input.Mapped().data(), guest.data(), guest.size());
        input.Flush();

        // The image as the replay creates an sRGB ASTC stand-in: a format without storage support
        // of its own, written through UNORM views.
        vk::Image image = allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_A8B8G8R8_SRGB_PACK32,
            .extent = {WIDTH, HEIGHT, 1},
            .mipLevels = LEVELS,
            .arrayLayers = LAYERS,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });

        // One download buffer holds every (level, layer) region back to back.
        u64 download_size = 0;
        std::vector<VkBufferImageCopy> regions;
        for (const Expected& texels : expected) {
            regions.push_back({
                .bufferOffset = download_size,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, texels.level, texels.layer, 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {texels.width, texels.height, 1},
            });
            download_size += texels.texels.size();
        }
        vk::Buffer download = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = download_size,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Download);

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
        const VkImageMemoryBarrier to_general{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                 VK_REMAINING_ARRAY_LAYERS},
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, {}, {}, to_general);
        const std::vector<vk::ImageView> views =
            decoder.Record(cmd, *input, 0, guest.size(), *image, levels);
        REQUIRE(views.size() == levels.size());
        const VkMemoryBarrier to_transfer{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            to_transfer, {}, {});
        cmd.CopyImageToBuffer(*image, VK_IMAGE_LAYOUT_GENERAL, *download, regions);
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
        download.Invalidate();

        const u8* actual = download.Mapped().data();
        size_t wrong = 0;
        for (const Expected& texels : expected) {
            for (size_t i = 0; i < texels.texels.size(); ++i) {
                const int difference = std::abs(int(actual[i]) - int(texels.texels[i]));
                if (difference > 1) {
                    if (wrong < 8) {
                        WARN("level " << texels.level << " layer " << texels.layer << " texel "
                                      << i / 4 % texels.width << "," << i / 4 / texels.width
                                      << " channel " << i % 4 << ": gpu " << int(actual[i])
                                      << " expected " << int(texels.texels[i]));
                    }
                    ++wrong;
                }
            }
            actual += texels.texels.size();
        }
        REQUIRE(wrong == 0);
    }
    vmaDestroyAllocator(vma);
}
