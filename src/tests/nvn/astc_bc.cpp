// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// ASTC stand-ins held as BC1/BC3/BC7 (nvn_bc_encode.h, nvn_format.h ToBcStandin): the stand-in
// choice, the chunking and block rules, a GPU round trip of block-linear ASTC blocks through the
// replay's decoder, the BC scratch and the encoder into a BC image, and BC7 against BC3 on layered
// mip chains and alpha cut-outs, with the encode cost (SKIP without a GPU). The smart selection
// (nvn_format.h SmartBc): the ASTC channel classifier against the CPU decoder, the formats and
// view swizzles, BC4/BC5/BC1 against BC3 and the refined endpoints, and the emulated path's
// BC1/BC3 blocks against another build's encoder.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/scope_exit.h"
#include "video_core/renderer_vulkan/nvn/nvn_astc_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_bc_encode.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/texture_cache/decode_bc.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace {

using namespace Vulkan::Nvn;
using VideoCore::Surface::PixelFormat;

HostFormat AstcRgba8StandIn(PixelFormat pixel_format, bool srgb) {
    HostFormat host{};
    host.format = srgb ? VK_FORMAT_A8B8G8R8_SRGB_PACK32 : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    host.pixel_format = pixel_format;
    host.attachable = true;
    host.storage = !srgb;
    host.needs_transcode = true;
    return host;
}

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
    u64 low = 0x1FC;          // block mode: void extent
    low |= u64{3} << 10;      // reserved bits are ones
    low |= u64{0x1FFF} << 12; // no extent: every coordinate field is all ones
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

/// One level of an RGBA8 2D array, layer after layer.
struct Rgba8Level {
    u32 width{};
    u32 height{};
    u32 layers{};
    std::vector<u8> texels;

    u8* Texel(u32 x, u32 y, u32 layer) {
        return &texels[((size_t{layer} * height + y) * width + x) * 4];
    }
    const u8* Texel(u32 x, u32 y, u32 layer) const {
        return &texels[((size_t{layer} * height + y) * width + x) * 4];
    }
};

/// Level 0 from `texel`, then 2x2 box-filtered levels down to 1x1 (what a game's mips look like
/// after an ASTC decode, near enough).
template <typename Texel>
std::vector<Rgba8Level> MakeMipChain(u32 width, u32 height, u32 layers, Texel&& texel) {
    std::vector<Rgba8Level> levels;
    Rgba8Level base{width, height, layers, std::vector<u8>(size_t{width} * height * layers * 4)};
    for (u32 layer = 0; layer < layers; ++layer) {
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const std::array<u8, 4> value = texel(x, y, layer);
                std::memcpy(base.Texel(x, y, layer), value.data(), 4);
            }
        }
    }
    levels.push_back(std::move(base));
    while (levels.back().width > 1 || levels.back().height > 1) {
        const Rgba8Level& above = levels.back();
        Rgba8Level next{(std::max)(above.width / 2, 1u), (std::max)(above.height / 2, 1u), layers,
                        {}};
        next.texels.resize(size_t{next.width} * next.height * layers * 4);
        for (u32 layer = 0; layer < layers; ++layer) {
            for (u32 y = 0; y < next.height; ++y) {
                for (u32 x = 0; x < next.width; ++x) {
                    for (u32 c = 0; c < 4; ++c) {
                        u32 sum = 0;
                        for (u32 dy = 0; dy < 2; ++dy) {
                            for (u32 dx = 0; dx < 2; ++dx) {
                                const u32 sx = (std::min)(x * 2 + dx, above.width - 1);
                                const u32 sy = (std::min)(y * 2 + dy, above.height - 1);
                                sum += above.Texel(sx, sy, layer)[c];
                            }
                        }
                        next.Texel(x, y, layer)[c] = static_cast<u8>((sum + 2) / 4);
                    }
                }
            }
        }
        levels.push_back(std::move(next));
    }
    return levels;
}

struct GpuEncodeResult {
    std::vector<Rgba8Level> decoded; ///< per level, as a sampler reads the BC image
    double gpu_ms{};                 ///< staging copies, encodes and block copies, every level
    std::vector<u8> blocks;          ///< every level's blocks, when asked for (SmartEncode)
};

/// How EncodeOnGpu encodes beyond the stand-in kind (the smart selection's stand-ins).
struct SmartEncode {
    bool smart{};  ///< a smart stand-in: BC1 blocks never punch through (BcBlocksOf)
    bool refine{}; ///< refined BC1/BC3 colour endpoints
    std::span<const u32> spirv{}; ///< another build's bc_encoder.comp
    bool keep_blocks{};           ///< fill GpuEncodeResult::blocks
};

/// Runs `levels` through the recorder's BC upload sequence (Recorder::UploadBcFromStaging and
/// RecordBcUpload: RGBA8 copied into the scratch a chunk of layers at a time, encoded, the blocks
/// copied into the level of a BC image), reads the image back and decodes it on the CPU. BC4 and
/// BC5 decode as their stand-in views read them: (R, R, R, 1) and (R, R, R, G).
GpuEncodeResult EncodeOnGpu(const Vulkan::Device& device, std::span<const Rgba8Level> levels,
                            AstcStandin kind, u32 quality, bool decode = true,
                            const SmartEncode& smart = {}) {
    using namespace Vulkan;
    const bool bc7 = kind == AstcStandin::Bc7;
    const u32 block_bytes = kind == AstcStandin::Bc1 || kind == AstcStandin::Bc4 ? 8 : 16;
    const VkFormat format = bc7                        ? VK_FORMAT_BC7_UNORM_BLOCK
                            : kind == AstcStandin::Bc3 ? VK_FORMAT_BC3_UNORM_BLOCK
                            : kind == AstcStandin::Bc4 ? VK_FORMAT_BC4_UNORM_BLOCK
                            : kind == AstcStandin::Bc5 ? VK_FORMAT_BC5_UNORM_BLOCK
                                                       : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    const PixelFormat decode_format = bc7                        ? PixelFormat::BC7_UNORM
                                      : kind == AstcStandin::Bc3 ? PixelFormat::BC3_UNORM
                                      : kind == AstcStandin::Bc4 ? PixelFormat::BC4_UNORM
                                      : kind == AstcStandin::Bc5 ? PixelFormat::BC5_UNORM
                                                                 : PixelFormat::BC1_RGBA_UNORM;
    const u32 width = levels[0].width;
    const u32 height = levels[0].height;
    const u32 layers = levels[0].layers;
    const u32 level_count = static_cast<u32>(levels.size());
    GpuEncodeResult result;
    VmaAllocator vma = device.CreateThreadSafeAllocator();
    SCOPE_EXIT {
        vmaDestroyAllocator(vma);
    };
    {
        MemoryAllocator allocator(device, vma);
        const BcGpuEncoder encoder(device, bc7, smart.spirv);
        u32 scratch_layers = 1;
        for (const BcChunk& chunk : SplitBcLayers(0, 0, layers, width, height)) {
            scratch_layers = (std::max)(scratch_layers, chunk.layers);
        }
        const std::shared_ptr<BcScratch> scratch =
            MakeBcScratch(device, allocator, width, height, scratch_layers, bc7);

        u64 staging_size = 0;
        std::vector<u64> staging_offsets;
        for (const Rgba8Level& level : levels) {
            staging_offsets.push_back(staging_size);
            staging_size += level.texels.size();
        }
        vk::Buffer staging = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Upload);
        for (u32 level = 0; level < level_count; ++level) {
            std::memcpy(staging.Mapped().data() + staging_offsets[level],
                        levels[level].texels.data(), levels[level].texels.size());
        }
        staging.Flush();

        vk::Image image = allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = level_count,
            .arrayLayers = layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });

        u64 download_size = 0;
        std::vector<VkBufferImageCopy> regions;
        for (u32 level = 0; level < level_count; ++level) {
            regions.push_back({
                .bufferOffset = download_size,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, layers},
                .imageOffset = {0, 0, 0},
                .imageExtent = {levels[level].width, levels[level].height, 1},
            });
            download_size +=
                BcBlocksSize(levels[level].width, levels[level].height, layers, block_bytes);
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
        // Timing is only taken when the queue has timestamps; encoding and readback run anyway.
        const auto families = device.GetPhysical().GetQueueFamilyProperties();
        const u32 valid_bits = device.GetGraphicsFamily() < families.size()
                                   ? families[device.GetGraphicsFamily()].timestampValidBits
                                   : 0;
        const bool timing = valid_bits != 0;
        vk::QueryPool queries = logical.CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
            .pipelineStatistics = 0,
        });
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
        if (timing) {
            cmd.ResetQueryPool(*queries, 0, 2);
        }
        VkImageMemoryBarrier to_dst{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                 VK_REMAINING_ARRAY_LAYERS},
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            to_dst);
        if (timing) {
            cmd.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *queries, 0);
        }
        for (u32 level = 0; level < level_count; ++level) {
            const Rgba8Level& texels = levels[level];
            for (const BcChunk& chunk :
                 SplitBcLayers(level, 0, layers, texels.width, texels.height)) {
                const VkMemoryBarrier reuse{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                };
                const VkImageMemoryBarrier to_general{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *scratch->image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                         VK_REMAINING_ARRAY_LAYERS},
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, reuse, {}, to_general);
                const u64 layer_bytes = u64{texels.width} * texels.height * 4;
                const VkBufferImageCopy fill{
                    .bufferOffset = staging_offsets[level] + u64{chunk.first_layer} * layer_bytes,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, chunk.layers},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {texels.width, texels.height, 1},
                };
                cmd.CopyBufferToImage(*staging, *scratch->image, VK_IMAGE_LAYOUT_GENERAL, fill);
                const VkMemoryBarrier texels_written{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, texels_written);
                const u64 blocks_size =
                    BcBlocksSize(chunk.width, chunk.height, chunk.layers, block_bytes);
                if (bc7) {
                    encoder.RecordBc7(cmd, *scratch->view, chunk.width, chunk.height,
                                      chunk.layers, quality, *scratch->modes,
                                      BcBlocksSize(chunk.width, chunk.height, chunk.layers, 8),
                                      *scratch->blocks, blocks_size);
                } else {
                    encoder.Record(cmd, *scratch->view, chunk.width, chunk.height, chunk.layers,
                                   BcBlocksOf(kind, smart.smart), smart.refine, *scratch->blocks,
                                   blocks_size);
                }
                const VkMemoryBarrier blocks_written{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, blocks_written);
                const VkBufferImageCopy region{
                    .bufferOffset = 0,
                    .bufferRowLength = (chunk.width + 3) / 4 * 4,
                    .bufferImageHeight = (chunk.height + 3) / 4 * 4,
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, chunk.first_layer,
                                         chunk.layers},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {chunk.width, chunk.height, 1},
                };
                cmd.CopyBufferToImage(*scratch->blocks, *image,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
            }
        }
        if (timing) {
            cmd.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *queries, 1);
        }
        VkImageMemoryBarrier to_src = to_dst;
        to_src.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            to_src);
        cmd.CopyImageToBuffer(*image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *download, regions);
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
        if (timing) {
            std::array<u64, 2> ticks{};
            REQUIRE(logical.GetQueryResults(*queries, 0, 2, sizeof(ticks), ticks.data(),
                                            sizeof(u64),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
                    VK_SUCCESS);
            const u64 mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
            const double period = device.GetPhysical().GetProperties().limits.timestampPeriod;
            result.gpu_ms = static_cast<double>((ticks[1] - ticks[0]) & mask) * period / 1e6;
        }

        if (decode || smart.keep_blocks) {
            download.Invalidate();
        }
        if (smart.keep_blocks) {
            result.blocks.assign(download.Mapped().data(),
                                 download.Mapped().data() + download_size);
        }
        if (decode) {
            const u8* blocks = download.Mapped().data();
            const u32 decoded_bytes = VideoCommon::ConvertedBytesPerBlock(decode_format);
            for (u32 level = 0; level < level_count; ++level) {
                const Rgba8Level& texels = levels[level];
                Rgba8Level decoded{texels.width, texels.height, layers,
                                   std::vector<u8>(texels.texels.size())};
                const u64 layer_blocks = BcBlocksSize(texels.width, texels.height, 1, block_bytes);
                const size_t layer_pixels = size_t{texels.width} * texels.height;
                std::vector<u8> channels(layer_pixels * decoded_bytes);
                for (u32 layer = 0; layer < layers; ++layer) {
                    VideoCommon::BufferImageCopy copy{};
                    copy.image_subresource.num_layers = 1;
                    copy.image_extent = {texels.width, texels.height, 1};
                    copy.buffer_row_length = (texels.width + 3) / 4 * 4;
                    VideoCommon::DecompressBCn(
                        std::span<const u8>{blocks, static_cast<size_t>(layer_blocks)}, channels,
                        copy, decode_format);
                    u8* const out = decoded.texels.data() + layer * layer_pixels * 4;
                    for (size_t i = 0; i < layer_pixels; ++i) {
                        if (decoded_bytes == 4) {
                            std::memcpy(out + i * 4, channels.data() + i * 4, 4);
                            continue;
                        }
                        const u8 l = channels[i * decoded_bytes];
                        out[i * 4] = l;
                        out[i * 4 + 1] = l;
                        out[i * 4 + 2] = l;
                        out[i * 4 + 3] = decoded_bytes == 2 ? channels[i * 2 + 1] : u8{255};
                    }
                    blocks += layer_blocks;
                }
                result.decoded.push_back(std::move(decoded));
            }
        }
    }
    return result;
}

struct Psnr {
    double rgb{};
    double alpha{};
};

/// PSNR of one layer of `decoded` against `expected`, RGB together and alpha alone (99 dB when
/// exact).
Psnr LayerPsnr(const Rgba8Level& expected, const Rgba8Level& decoded, u32 layer) {
    double rgb = 0.0;
    double alpha = 0.0;
    for (u32 y = 0; y < expected.height; ++y) {
        for (u32 x = 0; x < expected.width; ++x) {
            const u8* a = expected.Texel(x, y, layer);
            const u8* b = decoded.Texel(x, y, layer);
            for (u32 c = 0; c < 4; ++c) {
                const double d = double(a[c]) - double(b[c]);
                (c < 3 ? rgb : alpha) += d * d;
            }
        }
    }
    const double texels = double(expected.width) * expected.height;
    const auto psnr = [](double mse) {
        return mse <= 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    };
    return {psnr(rgb / (texels * 3.0)), psnr(alpha / texels)};
}

/// Texels whose alpha a cut-out shader would see differently: 0 or 255 not kept exactly, or the
/// texel on the other side of an alpha test at 64, 128 or 192.
size_t AlphaTestChanges(const Rgba8Level& expected, const Rgba8Level& decoded) {
    size_t changes = 0;
    for (size_t i = 3; i < expected.texels.size(); i += 4) {
        const u32 want = expected.texels[i];
        const u32 got = decoded.texels[i];
        bool changed = (want == 0 || want == 255) && got != want;
        for (const u32 threshold : {64u, 128u, 192u}) {
            changed = changed || (want >= threshold) != (got >= threshold);
        }
        changes += changed ? 1 : 0;
    }
    return changes;
}

// Value noise: smooth bumps at `cell` texels, hashed per lattice point.
double Noise(u32 seed, double x, double y, double cell) {
    const auto hash = [seed](s64 ix, s64 iy) {
        u32 h = static_cast<u32>(ix * 374761393 + iy * 668265263) ^ (seed * 2246822519u);
        h = (h ^ (h >> 13)) * 1274126177u;
        return double((h ^ (h >> 16)) & 0xFFFF) / 65535.0;
    };
    const double fx = x / cell;
    const double fy = y / cell;
    const s64 ix = static_cast<s64>(std::floor(fx));
    const s64 iy = static_cast<s64>(std::floor(fy));
    const double tx = fx - double(ix);
    const double ty = fy - double(iy);
    const double sx = tx * tx * (3.0 - 2.0 * tx);
    const double sy = ty * ty * (3.0 - 2.0 * ty);
    const double top = hash(ix, iy) + (hash(ix + 1, iy) - hash(ix, iy)) * sx;
    const double bottom = hash(ix, iy + 1) + (hash(ix + 1, iy + 1) - hash(ix, iy + 1)) * sx;
    return top + (bottom - top) * sy;
}

u8 Clamp8(double value) {
    return static_cast<u8>(std::clamp(std::lround(value), 0l, 255l));
}

/// A terrain-like material layer: two colours blended by low-frequency noise (often opposite in
/// some channel, which a box diagonal can't follow), fine detail, a few hard edges, and an
/// alpha mask of soft blobs (the blend weights such arrays carry).
std::array<u8, 4> MaterialTexel(u32 x, u32 y, u32 layer) {
    const u32 seed = layer * 7919 + 13;
    const double fx = double(x);
    const double fy = double(y);
    const std::array<double, 3> a{40.0 + 30.0 * (layer % 3), 90.0 + 15.0 * (layer % 5), 30.0};
    const std::array<double, 3> b{170.0, 120.0 - 10.0 * (layer % 4), 60.0 + 25.0 * (layer % 7)};
    const double blend = Noise(seed, fx, fy, 24.0);
    const double detail = Noise(seed + 1, fx, fy, 3.0) - 0.5;
    const bool edge = Noise(seed + 2, fx, fy, 40.0) > 0.7;
    std::array<u8, 4> texel{};
    for (u32 c = 0; c < 3; ++c) {
        const double base = a[c] + (b[c] - a[c]) * blend + detail * 40.0;
        texel[c] = Clamp8(edge ? base * 0.45 : base);
    }
    texel[3] = Clamp8(Noise(seed + 3, fx, fy, 16.0) * 320.0 - 32.0);
    return texel;
}

/// Foliage cut-outs: opaque leaves (alpha 255) with holes (alpha 0) whose edges cut 4x4 blocks:
/// discs, checkers on 5, 6 and 10 texel cells (ASTC footprints, alpha jumping 0 -> 255 between
/// neighbours off the 4x4 grid) and a strip with a soft three-texel edge.
std::array<u8, 4> CutoutTexel(u32 x, u32 y, u32 layer) {
    const u32 seed = layer * 104729 + 7;
    std::array<u8, 4> texel{
        Clamp8(40.0 + 60.0 * Noise(seed, x, y, 5.0)),
        Clamp8(90.0 + 120.0 * Noise(seed + 1, x, y, 7.0)),
        Clamp8(20.0 + 50.0 * Noise(seed + 2, x, y, 4.0)),
        255,
    };
    const u32 band = y / 32;
    if (band % 4 == 0) {
        const u32 cell = band % 3 == 0 ? 5 : (band % 3 == 1 ? 6 : 10);
        texel[3] = ((x / cell + y / cell) & 1) != 0 ? 0 : 255;
    } else if (band % 4 == 1) {
        const double t = (double(x % 37) - 17.0) / 3.0;
        texel[3] = Clamp8(std::clamp(t, 0.0, 1.0) * 255.0);
    } else {
        const double cx = 11.0 + 23.0 * double((x + 5) / 23);
        const double cy = 13.0 + 19.0 * double((y + 3) / 19);
        const double r = 4.5 + 5.0 * Noise(seed + 3, cx, cy, 1.0);
        const double dx = double(x) - cx;
        const double dy = double(y) - cy;
        texel[3] = dx * dx + dy * dy < r * r ? 0 : 255;
    }
    return texel;
}

} // Anonymous namespace

TEST_CASE("NVN ASTC stand-in: the environment overrides the setting", "[nvn]") {
    using Settings::AstcRecompression;
    REQUIRE(ChooseAstcStandin(nullptr, AstcRecompression::Bc3) == AstcStandin::Bc3);
    REQUIRE(ChooseAstcStandin(nullptr, AstcRecompression::Bc1) == AstcStandin::Bc1);
    REQUIRE(ChooseAstcStandin(nullptr, AstcRecompression::Uncompressed) == AstcStandin::Rgba8);
    REQUIRE(ChooseAstcStandin("rgba8", AstcRecompression::Bc3) == AstcStandin::Rgba8);
    REQUIRE(ChooseAstcStandin("bc1", AstcRecompression::Uncompressed) == AstcStandin::Bc1);
    REQUIRE(ChooseAstcStandin("bc3", AstcRecompression::Bc1) == AstcStandin::Bc3);
    REQUIRE(ChooseAstcStandin("bc7", AstcRecompression::Bc3) == AstcStandin::Bc7);
    REQUIRE(ParseBc7Quality(nullptr) == BC7_QUALITY_DEFAULT);
    REQUIRE(ParseBc7Quality("0") == 0);
    REQUIRE(ParseBc7Quality("2") == 2);
    REQUIRE(ParseBc7Quality("3") == BC7_QUALITY_DEFAULT);
    REQUIRE(ParseBc7Quality("12") == BC7_QUALITY_DEFAULT);
    // Anything else is not set.
    REQUIRE(ChooseAstcStandin("", AstcRecompression::Bc1) == AstcStandin::Bc1);
    REQUIRE(ChooseAstcStandin("BC9", AstcRecompression::Uncompressed) == AstcStandin::Rgba8);
}

TEST_CASE("NVN ASTC stand-in: BC formats keep the sRGB-ness and drop render/storage use", "[nvn]") {
    const HostFormat unorm = AstcRgba8StandIn(PixelFormat::ASTC_2D_6X6_UNORM, false);
    const HostFormat srgb = AstcRgba8StandIn(PixelFormat::ASTC_2D_8X8_SRGB, true);

    const HostFormat bc3 = ToBcStandin(unorm, AstcStandin::Bc3);
    REQUIRE(bc3.format == VK_FORMAT_BC3_UNORM_BLOCK);
    REQUIRE(bc3.bc_standin);
    REQUIRE(bc3.needs_transcode);
    REQUIRE_FALSE(bc3.attachable);
    REQUIRE_FALSE(bc3.storage);
    REQUIRE(bc3.storage_view == VK_FORMAT_UNDEFINED);
    REQUIRE(BcStandinBlockBytes(bc3) == 16);

    REQUIRE(ToBcStandin(srgb, AstcStandin::Bc3).format == VK_FORMAT_BC3_SRGB_BLOCK);
    REQUIRE(ToBcStandin(srgb, AstcStandin::Bc1).format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
    const HostFormat bc1 = ToBcStandin(unorm, AstcStandin::Bc1);
    REQUIRE(bc1.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
    REQUIRE(BcStandinBlockBytes(bc1) == 8);
    const HostFormat bc7 = ToBcStandin(unorm, AstcStandin::Bc7);
    REQUIRE(bc7.format == VK_FORMAT_BC7_UNORM_BLOCK);
    REQUIRE(ToBcStandin(srgb, AstcStandin::Bc7).format == VK_FORMAT_BC7_SRGB_BLOCK);
    REQUIRE(BcStandinBlockBytes(bc7) == 16);
    REQUIRE(BcStandinOf(bc7) == AstcStandin::Bc7);
    REQUIRE(BcStandinOf(bc3) == AstcStandin::Bc3);
    REQUIRE(BcStandinOf(unorm) == AstcStandin::Rgba8);
    REQUIRE(ViewsBcStandin(VK_FORMAT_BC7_SRGB_BLOCK, VK_FORMAT_BC7_UNORM_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC7_UNORM_BLOCK));

    // Rgba8, non-ASTC and native formats are left alone.
    REQUIRE_FALSE(ToBcStandin(unorm, AstcStandin::Rgba8).bc_standin);
    REQUIRE(ToBcStandin(unorm, AstcStandin::Rgba8).format == VK_FORMAT_A8B8G8R8_UNORM_PACK32);
    HostFormat native = unorm;
    native.needs_transcode = false;
    REQUIRE_FALSE(ToBcStandin(native, AstcStandin::Bc3).bc_standin);
    HostFormat bcn = AstcRgba8StandIn(PixelFormat::BC7_UNORM, false);
    REQUIRE_FALSE(ToBcStandin(bcn, AstcStandin::Bc3).bc_standin);
    REQUIRE(BcStandinBlockBytes(unorm) == 0);

    // Views: the same BC format in either encoding, nothing else.
    REQUIRE(ViewsBcStandin(VK_FORMAT_BC3_SRGB_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK));
    REQUIRE(ViewsBcStandin(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_BC3_UNORM_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_BC3_UNORM_BLOCK));
}

TEST_CASE("NVN ASTC stand-in: BC copies follow the block edge rules", "[nvn]") {
    REQUIRE(BcRegionAligned(0, 0, 70, 45, 70, 45));   // a whole level of any size
    REQUIRE(BcRegionAligned(0, 0, 2, 1, 2, 1));       // a level smaller than a block
    REQUIRE(BcRegionAligned(64, 32, 6, 13, 70, 45));  // reaching the right and bottom edges
    REQUIRE(BcRegionAligned(8, 4, 16, 8, 70, 45));    // inside, whole blocks
    REQUIRE_FALSE(BcRegionAligned(5, 0, 10, 10, 70, 45)); // a 5x5 ASTC block offset
    REQUIRE_FALSE(BcRegionAligned(0, 0, 10, 8, 70, 45));  // ends inside a block
    REQUIRE_FALSE(BcRegionAligned(0, 0, 8, 6, 70, 45));
}

TEST_CASE("NVN ASTC stand-in: chunks and sizes", "[nvn]") {
    REQUIRE(BcBlocksSize(70, 45, 3, 16) == u64{18} * 12 * 3 * 16);
    REQUIRE(BcBlocksSize(1, 1, 1, 8) == 8);
    REQUIRE(BcBlocksSize(2, 3, 1, 16) == 16);

    // OoT's 1365x1365x90 arrays: 7.1 MiB of RGBA8 per layer, nine layers per 64 MiB chunk.
    const std::vector<BcChunk> chunks = SplitBcLayers(0, 0, 90, 1365, 1365);
    REQUIRE(chunks.size() == 10);
    u32 next = 0;
    for (const BcChunk& chunk : chunks) {
        REQUIRE(chunk.first_layer == next);
        REQUIRE(u64{chunk.width} * chunk.height * chunk.layers * 4 <= BC_SCRATCH_BUDGET);
        next += chunk.layers;
    }
    REQUIRE(next == 90);
    // A layer above the budget is one chunk of its own; offsets carry over.
    const std::vector<BcChunk> big = SplitBcLayers(2, 5, 2, 6144, 6144, 4, 8);
    REQUIRE(big.size() == 2);
    REQUIRE(big[1].first_layer == 6);
    REQUIRE(big[1].layers == 1);
    REQUIRE(big[1].level == 2);
    REQUIRE((big[1].x == 4 && big[1].y == 8));

    // What the census counts for such an array: 11 levels, every layer.
    u64 rgba8 = 0;
    u64 bc3 = 0;
    for (u32 level = 0; level < 11; ++level) {
        const u32 side = (std::max)(1365u >> level, 1u);
        rgba8 += u64{side} * side * 4;
        bc3 += BcBlocksSize(side, side, 1, 16);
    }
    REQUIRE(rgba8 * 90 > (u64{850} << 20));
    REQUIRE(bc3 * 90 < (u64{230} << 20));
}

TEST_CASE("NVN ASTC stand-in: block-linear ASTC decodes and re-encodes into a BC image", "[nvn]") {
    using namespace Vulkan;
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Device& device = *vulkan->device;
    if (!AstcGpuDecoder::Supported(device) || !BcGpuEncoder::Supported(device)) {
        SKIP("the device cannot run the GPU ASTC decoder and BC encoder");
    }

    // ASTC 4x4 UNORM (NVN format 0x79), 2D array of sides that are not block multiples. One colour
    // per ASTC block: every 4x4 BC block is one colour too, which BC holds within its 565/8-bit
    // quantization.
    constexpr s32 FORMAT = 0x79;
    constexpr u32 WIDTH = 70;
    constexpr u32 HEIGHT = 45;
    constexpr u32 LEVELS = 4;
    constexpr u32 LAYERS = 3;
    TextureLayoutDesc desc{
        .target = static_cast<s32>(TexTarget::Dim2DArray),
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
    const TextureLayout layout = ComputeTextureLayout(desc);
    const BlockInfo block = FormatBlockInfo(FORMAT);
    REQUIRE(block.bytes == 16);

    std::mt19937 random(4321);
    std::vector<u8> guest(static_cast<size_t>(layout.size) + 4096, 0);
    struct Expected {
        u32 level;
        u32 layer;
        u32 width;
        u32 height;
        std::vector<u8> texels; // RGBA8
    };
    std::vector<Expected> expected;
    std::vector<AstcGpuDecoder::Level> levels;
    for (u32 level = 0; level < LEVELS; ++level) {
        const u32 width = (std::max)(WIDTH >> level, 1u);
        const u32 height = (std::max)(HEIGHT >> level, 1u);
        const u32 columns = (width + 3) / 4;
        const u32 rows = (height + 3) / 4;
        u32 tile_h = 0;
        u32 tile_d = 0;
        TextureLevelTiles(desc, level, tile_h, tile_d);
        for (u32 layer = 0; layer < LAYERS; ++layer) {
            std::vector<u8> linear(size_t{columns} * rows * 16);
            Expected texels{level, layer, width, height,
                            std::vector<u8>(size_t{width} * height * 4)};
            for (u32 row = 0; row < rows; ++row) {
                for (u32 column = 0; column < columns; ++column) {
                    const std::array<u16, 4> color{
                        static_cast<u16>(random()), static_cast<u16>(random()),
                        static_cast<u16>(random()), static_cast<u16>(random())};
                    const auto encoded = VoidExtentBlock(color);
                    std::memcpy(&linear[(size_t{row} * columns + column) * 16], encoded.data(), 16);
                    for (u32 y = row * 4; y < (std::min)(row * 4 + 4, height); ++y) {
                        for (u32 x = column * 4; x < (std::min)(column * 4 + 4, width); ++x) {
                            for (u32 c = 0; c < 4; ++c) {
                                texels.texels[(size_t{y} * width + x) * 4 + c] = ToUnorm8(color[c]);
                            }
                        }
                    }
                }
            }
            const u64 offset = u64{layer} * layout.layer_size + TextureLevelOffset(desc, level);
            Tegra::Texture::SwizzleTexture(
                std::span<u8>{guest.data() + offset, guest.size() - static_cast<size_t>(offset)},
                linear, 16, columns, rows, 1, tile_h, tile_d, 0, true);
            expected.push_back(std::move(texels));
        }
        levels.push_back({
            .level = 0, // the scratch's level
            .layers = LAYERS,
            .columns = columns,
            .rows = rows,
            .offset = TextureLevelOffset(desc, level),
            .layer_stride = static_cast<u32>(layout.layer_size),
            .block_width = 4,
            .block_height = 4,
            .tile_height = tile_h,
            .tile_depth = tile_d,
        });
    }

    for (const bool bc3 : {true, false}) {
        INFO((bc3 ? "BC3" : "BC1"));
        const u32 block_bytes = bc3 ? 16 : 8;
        VmaAllocator vma = device.CreateThreadSafeAllocator();
        SCOPE_EXIT {
            vmaDestroyAllocator(vma);
        };
        {
            MemoryAllocator allocator(device, vma);
            const AstcGpuDecoder decoder(device);
            const BcGpuEncoder encoder(device);
            const std::shared_ptr<BcScratch> scratch =
                MakeBcScratch(device, allocator, WIDTH, HEIGHT, LAYERS);

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

            // The stand-in as the replay creates it (ToBcStandin): sampled and copied only.
            vk::Image image = allocator.CreateImage(VkImageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = bc3 ? VK_FORMAT_BC3_UNORM_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                .extent = {WIDTH, HEIGHT, 1},
                .mipLevels = LEVELS,
                .arrayLayers = LAYERS,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            });

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
                download_size += BcBlocksSize(texels.width, texels.height, 1, block_bytes);
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
            VkImageMemoryBarrier to_dst{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                     VK_REMAINING_ARRAY_LAYERS},
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                to_dst);
            std::vector<vk::ImageView> views;
            // The recorder's sequence (Recorder::RecordBcUpload), one level per scratch fill.
            for (u32 level = 0; level < LEVELS; ++level) {
                const u32 width = (std::max)(WIDTH >> level, 1u);
                const u32 height = (std::max)(HEIGHT >> level, 1u);
                const VkMemoryBarrier reuse{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                };
                const VkImageMemoryBarrier to_general{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *scratch->image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                         VK_REMAINING_ARRAY_LAYERS},
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, reuse, {}, to_general);
                for (vk::ImageView& view :
                     decoder.Record(cmd, *input, 0, guest.size(), *scratch->image,
                                    std::span{&levels[level], 1})) {
                    views.push_back(std::move(view));
                }
                const VkMemoryBarrier texels_written{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, texels_written);
                encoder.Record(cmd, *scratch->view, width, height, LAYERS,
                               bc3 ? BcBlocks::Bc3 : BcBlocks::Bc1, false, *scratch->blocks,
                               BcBlocksSize(width, height, LAYERS, block_bytes));
                const VkMemoryBarrier blocks_written{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, blocks_written);
                const VkBufferImageCopy region{
                    .bufferOffset = 0,
                    .bufferRowLength = (width + 3) / 4 * 4,
                    .bufferImageHeight = (height + 3) / 4 * 4,
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, LAYERS},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {width, height, 1},
                };
                cmd.CopyBufferToImage(*scratch->blocks, *image,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
            }
            VkImageMemoryBarrier to_src = to_dst;
            to_src.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                to_src);
            cmd.CopyImageToBuffer(*image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *download, regions);
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

            // The blocks as a sampler reads them, against the ASTC texels: 565 colour endpoints
            // (5 bits: within 5), BC3's 8-bit alpha (within 1), BC1's punch-through alpha.
            const u8* blocks = download.Mapped().data();
            size_t wrong = 0;
            for (const Expected& texels : expected) {
                const u64 size = BcBlocksSize(texels.width, texels.height, 1, block_bytes);
                std::vector<u8> decoded(texels.texels.size());
                VideoCommon::BufferImageCopy copy{};
                copy.image_subresource.num_layers = 1;
                copy.image_extent = {texels.width, texels.height, 1};
                copy.buffer_row_length = (texels.width + 3) / 4 * 4; // the decoder steps rows by it
                VideoCommon::DecompressBCn(std::span<const u8>{blocks, static_cast<size_t>(size)},
                                           decoded, copy,
                                           bc3 ? PixelFormat::BC3_UNORM
                                               : PixelFormat::BC1_RGBA_UNORM);
                for (size_t i = 0; i < decoded.size(); i += 4) {
                    std::array<int, 4> want{texels.texels[i], texels.texels[i + 1],
                                            texels.texels[i + 2], texels.texels[i + 3]};
                    if (!bc3) {
                        want = want[3] < 128 ? std::array<int, 4>{0, 0, 0, 0}
                                             : std::array<int, 4>{want[0], want[1], want[2], 255};
                    }
                    for (u32 c = 0; c < 4; ++c) {
                        const int difference = std::abs(int(decoded[i + c]) - want[c]);
                        if (difference > (c == 3 ? 1 : 5)) {
                            if (wrong < 8) {
                                WARN("level " << texels.level << " layer " << texels.layer
                                              << " texel " << i / 4 % texels.width << ","
                                              << i / 4 / texels.width << " channel " << c
                                              << ": bc " << int(decoded[i + c]) << " expected "
                                              << want[c]);
                            }
                            ++wrong;
                        }
                    }
                }
                blocks += size;
            }
            REQUIRE(wrong == 0);
        }
    }
}

TEST_CASE("NVN ASTC stand-in: BC7 keeps every layer of every level of an array", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::Supported(device) || !BcGpuEncoder::SupportsBc7(device)) {
        SKIP("the device cannot run the BC7 encoder");
    }
    // Odd sides: every level down to 1x1 ends in partial blocks; the eight layers are encoded
    // in one chunk per level, as the recorder does (BC_SCRATCH_BUDGET).
    REQUIRE(SplitBcLayers(0, 0, 8, 341, 341).size() == 1);
    for (const bool opaque : {true, false}) {
        const std::vector<Rgba8Level> levels =
            MakeMipChain(341, 341, 8, [opaque](u32 x, u32 y, u32 layer) {
                std::array<u8, 4> texel = MaterialTexel(x, y, layer);
                texel[3] = opaque ? 255 : texel[3];
                return texel;
            });
        REQUIRE(levels.size() == 9);
        const char* const content = opaque ? "opaque" : "blend alpha";

        const GpuEncodeResult bc3 = EncodeOnGpu(device, levels, AstcStandin::Bc3, 0);
        for (u32 quality = 0; quality <= BC7_QUALITY_MAX; ++quality) {
            const GpuEncodeResult bc7 = EncodeOnGpu(device, levels, AstcStandin::Bc7, quality);
            double worst_rgb = 99.0;
            double worst_alpha = 99.0;
            for (u32 level = 0; level < levels.size(); ++level) {
                std::string line =
                    fmt::format("{} BC7 q{} level {} ({}x{}) RGB/alpha dB, BC3 in ():",
                                content, quality, level, levels[level].width,
                                levels[level].height);
                for (u32 layer = 0; layer < levels[level].layers; ++layer) {
                    const Psnr ours = LayerPsnr(levels[level], bc7.decoded[level], layer);
                    const Psnr theirs = LayerPsnr(levels[level], bc3.decoded[level], layer);
                    line += fmt::format(" {:.1f}/{:.1f} ({:.1f}/{:.1f})", ours.rgb, ours.alpha,
                                        theirs.rgb, theirs.alpha);
                    worst_rgb = (std::min)(worst_rgb, ours.rgb);
                    worst_alpha = (std::min)(worst_alpha, ours.alpha);
                    INFO("level " << level << " layer " << layer);
                    // The same bounds for every layer at every level, and no worse than the BC3
                    // encode of the same layer: a layer after the first that breaks at a small
                    // level (dark grid lines between tiles) falls far below both.
                    CHECK(ours.rgb >= (opaque ? 33.0 : 27.0));
                    CHECK(ours.alpha >= 28.0);
                    CHECK(ours.rgb >= theirs.rgb - 2.0);
                }
                WARN(line);
            }
            WARN(fmt::format("{} BC7 q{}: worst layer/level RGB {:.2f} dB, alpha {:.2f} dB",
                             content, quality, worst_rgb, worst_alpha));
        }
    }
}

TEST_CASE("NVN ASTC stand-in: BC7 keeps alpha cut-outs", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::Supported(device) || !BcGpuEncoder::SupportsBc7(device)) {
        SKIP("the device cannot run the BC7 encoder");
    }
    const std::vector<Rgba8Level> levels = MakeMipChain(213, 157, 3, CutoutTexel);
    const GpuEncodeResult bc3 = EncodeOnGpu(device, levels, AstcStandin::Bc3, 0);
    for (u32 quality = 0; quality <= BC7_QUALITY_MAX; ++quality) {
        const GpuEncodeResult bc7 = EncodeOnGpu(device, levels, AstcStandin::Bc7, quality);
        for (u32 level = 0; level < levels.size(); ++level) {
            const size_t ours = AlphaTestChanges(levels[level], bc7.decoded[level]);
            const size_t theirs = AlphaTestChanges(levels[level], bc3.decoded[level]);
            Psnr ours_psnr{99.0, 99.0};
            Psnr theirs_psnr{99.0, 99.0};
            for (u32 layer = 0; layer < levels[level].layers; ++layer) {
                const Psnr a = LayerPsnr(levels[level], bc7.decoded[level], layer);
                const Psnr b = LayerPsnr(levels[level], bc3.decoded[level], layer);
                ours_psnr = {(std::min)(ours_psnr.rgb, a.rgb),
                             (std::min)(ours_psnr.alpha, a.alpha)};
                theirs_psnr = {(std::min)(theirs_psnr.rgb, b.rgb),
                               (std::min)(theirs_psnr.alpha, b.alpha)};
            }
            WARN(fmt::format("cut-outs q{} level {} ({}x{}): alpha-test changes BC7 {} BC3 {}; "
                             "worst layer RGB/alpha dB BC7 {:.1f}/{:.1f} BC3 {:.1f}/{:.1f}",
                             quality, level, levels[level].width, levels[level].height, ours,
                             theirs, ours_psnr.rgb, ours_psnr.alpha, theirs_psnr.rgb,
                             theirs_psnr.alpha));
            INFO("level " << level);
            CHECK(ours == 0);
        }
    }
}

TEST_CASE("NVN ASTC stand-in: BC7 encode cost of a 1365x1365 terrain array chunk", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::Supported(device) || !BcGpuEncoder::SupportsBc7(device)) {
        SKIP("the device cannot run the BC7 encoder");
    }
    // One recorder chunk of OoT's terrain arrays: nine 1365x1365 layers, eleven levels. The
    // material layers are harsher than a decoded ASTC 8x8 texture (fine noise in every block);
    // the smooth ones keep only their low-frequency blend, closer to what ASTC 8x8 leaves.
    constexpr u32 LAYERS = 9;
    for (const bool smooth : {false, true}) {
        const std::vector<Rgba8Level> levels =
            MakeMipChain(1365, 1365, LAYERS, [smooth](u32 x, u32 y, u32 layer) {
                if (!smooth) {
                    return MaterialTexel(x, y, layer);
                }
                const double blend = Noise(layer * 31 + 5, x, y, 48.0);
                const double grain = Noise(layer, x, y, 9.0);
                return std::array<u8, 4>{Clamp8(60.0 + 120.0 * blend),
                                         Clamp8(110.0 - 40.0 * blend + 20.0 * grain),
                                         Clamp8(40.0 + 50.0 * blend), 255};
            });
        REQUIRE(levels.size() == 11);
        const char* const content = smooth ? "smooth" : "material";
        const auto report = [&](const char* name, const GpuEncodeResult& result) {
            const double per_layer = result.gpu_ms / LAYERS;
            WARN(fmt::format("{} {}: {:.1f} ms for 1365x1365x{} with 11 levels ({:.2f} ms per "
                             "layer, ~{:.0f} ms for 90 layers)",
                             content, name, result.gpu_ms, LAYERS, per_layer, per_layer * 90.0));
        };
        // Warm up the pipelines and clocks, then measure.
        (void)EncodeOnGpu(device, levels, AstcStandin::Bc3, 0, false);
        report("BC3", EncodeOnGpu(device, levels, AstcStandin::Bc3, 0, false));
        for (u32 quality = 0; quality <= BC7_QUALITY_MAX; ++quality) {
            const GpuEncodeResult result =
                EncodeOnGpu(device, levels, AstcStandin::Bc7, quality, false);
            report(quality == 0 ? "BC7 q0" : (quality == 1 ? "BC7 q1" : "BC7 q2"), result);
        }
    }
}

namespace {

using Tegra::Texture::ASTC::Channels;

/// Channel bits of decoded texels (the Channels values): R, G and B unequal somewhere / alpha
/// below 255 somewhere.
constexpr u32 COLOUR = 2;
constexpr u32 ALPHA = 1;

/// Decodes 16-byte ASTC blocks of footprint `bw` x `bh` with the CPU decoder, in rows of up to 256
/// blocks; the texels of block i at i * bw * bh * 4.
std::vector<u8> DecodeAstcBlocks(std::span<const u8> blocks, u32 bw, u32 bh) {
    const u32 count = static_cast<u32>(blocks.size() / 16);
    const u32 columns = (std::min)(count, 256u);
    const u32 rows = (count + columns - 1) / columns;
    std::vector<u8> padded(size_t{columns} * rows * 16, 0);
    std::memcpy(padded.data(), blocks.data(), size_t{count} * 16);
    std::vector<u8> texels(size_t{columns} * bw * rows * bh * 4);
    Tegra::Texture::ASTC::Decompress(padded, columns * bw, rows * bh, 1, bw, bh, texels);
    std::vector<u8> out(size_t{count} * bw * bh * 4);
    for (u32 i = 0; i < count; ++i) {
        const u32 row = i / columns;
        const u32 column = i % columns;
        for (u32 y = 0; y < bh; ++y) {
            std::memcpy(&out[(size_t{i} * bw * bh + size_t{y} * bw) * 4],
                        &texels[((size_t{row} * bh + y) * columns * bw + size_t{column} * bw) * 4],
                        size_t{bw} * 4);
        }
    }
    return out;
}

u32 TexelChannels(std::span<const u8> texels) {
    u32 found = 0;
    for (size_t i = 0; i + 4 <= texels.size(); i += 4) {
        if (texels[i] != texels[i + 1] || texels[i + 1] != texels[i + 2]) {
            found |= COLOUR;
        }
        if (texels[i + 3] != 255) {
            found |= ALPHA;
        }
    }
    return found;
}

/// A pseudo-random block: anything (variant 0); one partition with a luminance mode (1); two to
/// four partitions sharing one luminance mode (2); two to four partitions with per-partition
/// modes (3); a void-extent block, grey or not, opaque or not (4).
std::array<u8, 16> RandomBlock(std::mt19937& random, u32 variant) {
    constexpr std::array<u32, 4> LUMINANCE{0, 1, 4, 5};
    std::array<u8, 16> block;
    for (u8& byte : block) {
        byte = static_cast<u8>(random());
    }
    u32 low;
    std::memcpy(&low, block.data(), sizeof(low));
    switch (variant % 5) {
    case 0:
        break;
    case 1:
        low = (low & ~(0x3u << 11) & ~(0xFu << 13)) | (LUMINANCE[random() % 4] << 13);
        break;
    case 2:
        low = (low & ~(0x3u << 11) & ~(0x3Fu << 23)) | ((1 + random() % 3) << 11) |
              (LUMINANCE[random() % 4] << 25);
        break;
    case 3:
        low = (low & ~(0x3u << 11)) | ((1 + random() % 3) << 11);
        if (((low >> 23) & 3) == 0) {
            low |= 1u << 23;
        }
        break;
    default: {
        const u16 l = static_cast<u16>(random());
        std::array<u16, 4> rgba{l, l, l, 0xFFFF};
        if (random() % 2 != 0) {
            rgba[random() % 3] = static_cast<u16>(random());
        }
        if (random() % 2 != 0) {
            rgba[3] = static_cast<u16>(random());
        }
        return VoidExtentBlock(rgba);
    }
    }
    std::memcpy(block.data(), &low, sizeof(low));
    return block;
}

/// A block of mode `mode`, one partition, colour endpoint mode `cem`, the rest random.
std::array<u8, 16> BlockWithMode(std::mt19937& random, u32 mode, u32 cem) {
    std::array<u8, 16> block;
    for (u8& byte : block) {
        byte = static_cast<u8>(random());
    }
    u32 low;
    std::memcpy(&low, block.data(), sizeof(low));
    low = (low & ~0x1FFFFu) | mode | (cem << 13);
    std::memcpy(block.data(), &low, sizeof(low));
    return block;
}

/// The first block mode that decodes (no error colour) with one weight plane at footprint
/// `bw` x `bh`, found with the CPU decoder alone: CEM 0 blocks of it decode to opaque grey, CEM 8
/// ones to colours.
u32 FindPlainMode(u32 bw, u32 bh) {
    std::mt19937 random(99);
    for (u32 mode = 0; mode < 2048; ++mode) {
        std::vector<u8> blocks;
        for (u32 i = 0; i < 8; ++i) {
            for (const u32 cem : {0u, 8u}) {
                const auto block = BlockWithMode(random, mode, cem);
                blocks.insert(blocks.end(), block.begin(), block.end());
            }
        }
        const std::vector<u8> texels = DecodeAstcBlocks(blocks, bw, bh);
        const size_t per_block = size_t{bw} * bh * 4;
        bool plain = true;
        for (u32 i = 0; i < 16 && plain; ++i) {
            const std::span<const u8> one{&texels[i * per_block], per_block};
            const u32 found = TexelChannels(one);
            plain = i % 2 == 0 ? found == 0 : found == COLOUR;
        }
        if (plain) {
            return mode;
        }
    }
    return 0;
}

std::vector<u8> Repeat(const std::array<u8, 16>& block, size_t count) {
    std::vector<u8> result(count * 16);
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(&result[i * 16], block.data(), 16);
    }
    return result;
}

Channels Classify(std::span<const u8> data, u32 bw = 8, u32 bh = 8, bool* empty = nullptr) {
    return Tegra::Texture::ASTC::ClassifyChannels(data, bw, bh, empty);
}

/// What a sampler returns for component `swizzle` of a texel `rgba`.
u8 Swizzled(const std::array<u8, 4>& rgba, VkComponentSwizzle swizzle, u32 component) {
    switch (swizzle) {
    case VK_COMPONENT_SWIZZLE_ZERO:
        return 0;
    case VK_COMPONENT_SWIZZLE_ONE:
        return 255;
    case VK_COMPONENT_SWIZZLE_R:
        return rgba[0];
    case VK_COMPONENT_SWIZZLE_G:
        return rgba[1];
    case VK_COMPONENT_SWIZZLE_B:
        return rgba[2];
    case VK_COMPONENT_SWIZZLE_A:
        return rgba[3];
    default:
        return rgba[component]; // IDENTITY
    }
}

/// Grey (L, L, L, 255) or grey with alpha (L, L, L, A) from a material layer.
std::array<u8, 4> GreyTexel(u32 x, u32 y, u32 layer, bool alpha) {
    const std::array<u8, 4> material = MaterialTexel(x, y, layer);
    const u8 l = static_cast<u8>((material[0] * 3 + material[1] * 6 + material[2] + 5) / 10);
    return {l, l, l, alpha ? material[3] : u8{255}};
}

/// The bc_encoder.comp SPIR-V of another build from `path`: its generated bc_encoder_comp_spv.h
/// (the 0x........ words) or the raw .spv.
std::vector<u32> LoadSpirv(const char* path) {
    std::ifstream file(path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::vector<u32> words;
    if (text.size() >= 4 && static_cast<u8>(text[0]) == 0x03 && static_cast<u8>(text[1]) == 0x02 &&
        static_cast<u8>(text[2]) == 0x23 && static_cast<u8>(text[3]) == 0x07) {
        words.resize(text.size() / 4);
        std::memcpy(words.data(), text.data(), words.size() * 4);
        return words;
    }
    for (size_t at = text.find("0x"); at != std::string::npos; at = text.find("0x", at + 2)) {
        if (at + 10 <= text.size()) {
            words.push_back(static_cast<u32>(std::stoul(text.substr(at + 2, 8), nullptr, 16)));
        }
    }
    return words;
}

} // Anonymous namespace

TEST_CASE("NVN smart BC: the classifier never claims fewer channels than the CPU decoder gives",
          "[nvn]") {
    // Every block alone, against what the decoder makes of it: luminance-only blocks must decode
    // grey, opaque ones opaque. Random blocks reach the error, dual-plane, multi-partition and
    // void-extent paths; most of the luminance ones come from variants 1 and 2.
    std::mt19937 random(2026);
    for (const auto& [bw, bh] : {std::pair{4u, 4u}, std::pair{6u, 6u}, std::pair{8u, 8u},
                                 std::pair{10u, 8u}, std::pair{12u, 12u}}) {
        constexpr u32 COUNT = 30000;
        std::vector<u8> blocks;
        blocks.reserve(COUNT * 16);
        for (u32 i = 0; i < COUNT; ++i) {
            const auto block = RandomBlock(random, i);
            blocks.insert(blocks.end(), block.begin(), block.end());
        }
        const std::vector<u8> texels = DecodeAstcBlocks(blocks, bw, bh);
        const size_t per_block = size_t{bw} * bh * 4;
        std::array<u32, 4> classes{};
        u32 wrong = 0;
        for (u32 i = 0; i < COUNT; ++i) {
            const Channels channels = Classify({&blocks[i * 16], 16}, bw, bh);
            const u32 decoded = TexelChannels({&texels[i * per_block], per_block});
            ++classes[static_cast<size_t>(channels)];
            if ((decoded & ~static_cast<u32>(channels)) != 0) {
                if (wrong++ < 8) {
                    WARN(fmt::format("{}x{} block {} variant {}: classified {}, decodes to {}", bw,
                                     bh, i, i % 5, static_cast<int>(channels), decoded));
                }
            }
        }
        WARN(fmt::format("{}x{}: {} random blocks: L {}, LA {}, RGB {}, RGBA {}", bw, bh, COUNT,
                         classes[0], classes[1], classes[2], classes[3]));
        INFO(bw << "x" << bh);
        CHECK(wrong == 0);
        // The fuzz reaches every class.
        for (const u32 count : classes) {
            CHECK(count >= 100);
        }
    }
}

TEST_CASE("NVN smart BC: classes of hand-built blocks, mixed textures, padding and the early exit",
          "[nvn]") {
    // Void-extent blocks of chosen colours.
    const auto grey = VoidExtentBlock({0x8000, 0x8000, 0x8000, 0xFFFF});
    const auto grey_alpha = VoidExtentBlock({0x8000, 0x8000, 0x8000, 0x4000});
    const auto colour = VoidExtentBlock({0x8000, 0x8001, 0x8000, 0xFFFF});
    const auto colour_alpha = VoidExtentBlock({0x1000, 0x8000, 0xF000, 0xFFFE});
    REQUIRE(Classify(grey) == Channels::Luminance);
    REQUIRE(Classify(grey_alpha) == Channels::LuminanceAlpha);
    REQUIRE(Classify(colour) == Channels::Rgb);
    REQUIRE(Classify(colour_alpha) == Channels::Rgba);
    // HDR void extent (mode bit 9) and reserved bits clear: both decoders fill the error colour.
    auto hdr = grey;
    hdr[1] |= 0x02;
    REQUIRE(Classify(hdr) == Channels::Rgba);
    auto reserved = grey;
    reserved[1] &= ~0x0C;
    REQUIRE(Classify(reserved) == Channels::Rgba);
    REQUIRE(Tegra::Texture::ASTC::IsOpaque(reserved)); // IsOpaque calls the error colour opaque

    // One-partition blocks of each endpoint mode, on a mode the decoder accepts.
    for (const auto& [bw, bh] : {std::pair{4u, 4u}, std::pair{8u, 8u}, std::pair{12u, 12u}}) {
        const u32 mode = FindPlainMode(bw, bh);
        INFO(bw << "x" << bh << " block mode " << mode);
        REQUIRE(mode != 0);
        std::mt19937 random(bw * 31 + bh);
        // CEM -> class: luminance 0 1, luminance+alpha 4 5, RGB 6 8 9, RGBA 10 12 13, HDR 2 3 7 11
        // 14 15 (counted as RGBA).
        constexpr std::array<Channels, 16> EXPECTED{
            Channels::Luminance, Channels::Luminance,      Channels::Rgba, Channels::Rgba,
            Channels::LuminanceAlpha, Channels::LuminanceAlpha, Channels::Rgb, Channels::Rgba,
            Channels::Rgb,       Channels::Rgb,            Channels::Rgba, Channels::Rgba,
            Channels::Rgba,      Channels::Rgba,           Channels::Rgba, Channels::Rgba,
        };
        for (u32 cem = 0; cem < 16; ++cem) {
            for (u32 i = 0; i < 16; ++i) {
                const auto block = BlockWithMode(random, mode, cem);
                const Channels channels = Classify(block, bw, bh);
                INFO("CEM " << cem);
                CHECK(channels == EXPECTED[cem]);
                const std::vector<u8> texels = DecodeAstcBlocks(block, bw, bh);
                CHECK((TexelChannels(texels) & ~static_cast<u32>(channels)) == 0);
            }
        }
        // A weight grid larger than the footprint decodes to the error colour: a 12x12 grid mode
        // is RGBA at 4x4.
        if (bw == 12) {
            bool grid_found = false;
            for (u32 m = 0; m < 2048 && !grid_found; ++m) {
                const auto block = BlockWithMode(random, m, 0);
                const std::vector<u8> big = DecodeAstcBlocks(block, 12, 12);
                const std::vector<u8> small = DecodeAstcBlocks(block, 4, 4);
                if (TexelChannels(big) == 0 && TexelChannels(small) == ALPHA &&
                    std::all_of(small.begin(), small.end(), [](u8 v) { return v == 0; })) {
                    grid_found = true;
                    CHECK(Classify(block, 12, 12) == Channels::Luminance);
                    CHECK(Classify(block, 4, 4) == Channels::Rgba);
                }
            }
            CHECK(grid_found);
        }
    }

    // Mixed textures: the union of the blocks' channels.
    const auto join = [](std::initializer_list<std::array<u8, 16>> blocks) {
        std::vector<u8> data;
        for (const auto& block : blocks) {
            data.insert(data.end(), block.begin(), block.end());
        }
        return data;
    };
    CHECK(Classify(join({grey, grey, grey_alpha})) == Channels::LuminanceAlpha);
    CHECK(Classify(join({grey, colour, grey})) == Channels::Rgb);
    CHECK(Classify(join({grey_alpha, colour})) == Channels::Rgba);
    CHECK(Classify(join({colour, grey_alpha, grey})) == Channels::Rgba);

    // All-zero blocks (block-linear padding) are skipped; other error blocks are RGBA.
    const std::array<u8, 16> zero{};
    bool empty = false;
    CHECK(Classify(join({zero, zero, zero}), 8, 8, &empty) == Channels::Luminance);
    CHECK(empty);
    CHECK(Classify(join({zero, grey, zero}), 8, 8, &empty) == Channels::Luminance);
    CHECK_FALSE(empty);
    CHECK(Classify(join({colour, zero}), 8, 8, &empty) == Channels::Rgb);
    CHECK_FALSE(empty);
    auto error = zero;
    error[9] = 1; // mode 0 (reserved) with data: not padding
    CHECK(Classify(join({grey, error}), 8, 8, &empty) == Channels::Rgba);

    // Cost: a whole scan of 64 MiB of grey or opaque colour blocks (what an L or BC1 texture
    // needs), and the early exit when the first block is RGBA.
    const size_t blocks = (size_t{64} << 20) / 16;
    const std::vector<u8> grey_texture = Repeat(grey, blocks);
    std::vector<u8> colour_texture = Repeat(colour, blocks);
    std::mt19937 random(5);
    const u32 mode = FindPlainMode(8, 8);
    for (size_t i = 0; i < blocks; i += 3) {
        // Endpoint blocks between the void-extent ones: the table and CEM path.
        const auto block = BlockWithMode(random, mode, i % 2 == 0 ? 8 : 0);
        std::memcpy(&colour_texture[i * 16], block.data(), 16);
    }
    std::vector<u8> early = grey_texture;
    std::memcpy(early.data() + 16, colour_alpha.data(), 16);
    const auto time = [](const std::vector<u8>& data, Channels& channels) {
        const auto begin = std::chrono::steady_clock::now();
        channels = Classify(data);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
            .count();
    };
    Channels channels{};
    (void)time(grey_texture, channels); // warm the table and the caches
    const double grey_ms = time(grey_texture, channels);
    CHECK(channels == Channels::Luminance);
    const double colour_ms = time(colour_texture, channels);
    CHECK(channels == Channels::Rgb);
    const double early_ms = time(early, channels);
    CHECK(channels == Channels::Rgba);
    const auto opaque_begin = std::chrono::steady_clock::now();
    const bool opaque = Tegra::Texture::ASTC::IsOpaque(colour_texture);
    const double opaque_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - opaque_begin)
                                 .count();
    CHECK(opaque);
    WARN(fmt::format("classifier: 64 MiB of ASTC: grey void extents {:.1f} ms ({:.3f} ms/MiB), "
                     "colour blocks {:.1f} ms ({:.3f} ms/MiB), IsOpaque on the same {:.1f} ms; "
                     "RGBA in block 2: {:.4f} ms",
                     grey_ms, grey_ms / 64.0, colour_ms, colour_ms / 64.0, opaque_ms, early_ms));
    CHECK(early_ms * 20.0 < grey_ms);
}

TEST_CASE("NVN smart BC: IsOpaque answers as before", "[nvn]") {
    // The classifier shares IsOpaque's mode table and endpoint-mode decoding: the emulated path's
    // BC1 choice must not move. Single blocks of every variant, then runs of them; the hash of
    // the answers is what IsOpaque of efe3468fc1 (before the sharing) gave on the same corpus,
    // compared answer by answer when the classifier was added.
    constexpr u64 BASE_HASH = 0x5ecccc9c77e1788full;
    constexpr u32 BASE_OPAQUE = 125266;
    std::mt19937 random(77);
    std::vector<u8> corpus;
    constexpr u32 COUNT = 200000;
    corpus.reserve(COUNT * 16);
    for (u32 i = 0; i < COUNT; ++i) {
        const auto block = RandomBlock(random, i);
        corpus.insert(corpus.end(), block.begin(), block.end());
    }
    u64 hash = 0xcbf29ce484222325ull;
    u32 opaque = 0;
    const auto note = [&](bool answer) {
        hash = (hash ^ (answer ? 1u : 0u)) * 0x100000001b3ull;
        opaque += answer ? 1 : 0;
    };
    for (u32 i = 0; i < COUNT; ++i) {
        note(Tegra::Texture::ASTC::IsOpaque(std::span<const u8>{&corpus[i * 16], 16}));
    }
    for (u32 run = 0; run < COUNT / 8; ++run) {
        note(Tegra::Texture::ASTC::IsOpaque(std::span<const u8>{&corpus[run * 8 * 16], 8 * 16}));
    }
    WARN(fmt::format("IsOpaque: {} of {} answers opaque, hash {:#018x}", opaque,
                     COUNT + COUNT / 8, hash));
    CHECK(opaque == BASE_OPAQUE);
    CHECK(hash == BASE_HASH);
}

TEST_CASE("NVN smart BC: switches, formats and views", "[nvn]") {
    // On by default with BC3 only (the nvn_smart_bc setting); the refined endpoints are opt-in
    // (BC_REFINE=1); `0` turns a part (or everything) off.
    const SmartBc on = ParseSmartBc(AstcStandin::Bc3, true, nullptr, nullptr, nullptr, nullptr);
    REQUIRE((on.enabled && on.bc1 && on.bc45 && !on.refine));
    const SmartBc setting_off =
        ParseSmartBc(AstcStandin::Bc3, false, nullptr, nullptr, nullptr, nullptr);
    REQUIRE_FALSE((setting_off.enabled || setting_off.bc1 || setting_off.bc45));
    for (const AstcStandin other : {AstcStandin::Rgba8, AstcStandin::Bc1, AstcStandin::Bc7}) {
        const SmartBc off = ParseSmartBc(other, true, nullptr, nullptr, nullptr, nullptr);
        REQUIRE_FALSE((off.enabled || off.bc1 || off.bc45 || off.refine));
    }
    // The env switch overrides the setting either way.
    REQUIRE_FALSE(ParseSmartBc(AstcStandin::Bc3, true, "0", nullptr, nullptr, nullptr).enabled);
    REQUIRE(ParseSmartBc(AstcStandin::Bc3, false, "1", nullptr, nullptr, nullptr).enabled);
    const SmartBc all_off = ParseSmartBc(AstcStandin::Bc3, true, "0", "1", "1", "1");
    REQUIRE_FALSE((all_off.enabled || all_off.bc1 || all_off.bc45 || all_off.refine));
    REQUIRE(SmartBcName(all_off) == "off");
    const SmartBc no_bc1 = ParseSmartBc(AstcStandin::Bc3, true, "1", "0", nullptr, nullptr);
    REQUIRE((no_bc1.enabled && !no_bc1.bc1 && no_bc1.bc45 && !no_bc1.refine));
    const SmartBc no_bc45 = ParseSmartBc(AstcStandin::Bc3, true, nullptr, nullptr, "0", nullptr);
    REQUIRE((no_bc45.bc1 && !no_bc45.bc45));
    const SmartBc refined = ParseSmartBc(AstcStandin::Bc3, true, nullptr, nullptr, nullptr, "1");
    REQUIRE((refined.bc1 && refined.bc45 && refined.refine));
    REQUIRE(SmartBcName(on) == "on (BC1, BC4/BC5)");
    REQUIRE(SmartBcName(refined) == "on (BC1, BC4/BC5, refined endpoints)");

    // Class -> stand-in, UNORM and sRGB, under each switch.
    using C = Channels;
    using S = AstcStandin;
    REQUIRE(SmartStandinFor(C::Luminance, false, on) == S::Bc4);
    REQUIRE(SmartStandinFor(C::LuminanceAlpha, false, on) == S::Bc5);
    REQUIRE(SmartStandinFor(C::Rgb, false, on) == S::Bc1);
    REQUIRE(SmartStandinFor(C::Rgba, false, on) == S::Bc3);
    REQUIRE(SmartStandinFor(C::Luminance, true, on) == S::Bc1);
    REQUIRE(SmartStandinFor(C::LuminanceAlpha, true, on) == S::Bc3);
    REQUIRE(SmartStandinFor(C::Rgb, true, on) == S::Bc1);
    REQUIRE(SmartStandinFor(C::Luminance, false, no_bc45) == S::Bc1);
    REQUIRE(SmartStandinFor(C::LuminanceAlpha, false, no_bc45) == S::Bc3);
    REQUIRE(SmartStandinFor(C::Rgb, false, no_bc1) == S::Bc3);
    REQUIRE(SmartStandinFor(C::Luminance, false, no_bc1) == S::Bc4);
    for (const C c : {C::Luminance, C::LuminanceAlpha, C::Rgb, C::Rgba}) {
        REQUIRE(SmartStandinFor(c, false, all_off) == S::Bc3);
    }
    // What each stand-in holds (a re-upload of another class recreates it).
    REQUIRE(StandinHolds(S::Bc4, C::Luminance));
    REQUIRE_FALSE(StandinHolds(S::Bc4, C::LuminanceAlpha));
    REQUIRE_FALSE(StandinHolds(S::Bc4, C::Rgb));
    REQUIRE(StandinHolds(S::Bc5, C::LuminanceAlpha));
    REQUIRE_FALSE(StandinHolds(S::Bc5, C::Rgb));
    REQUIRE(StandinHolds(S::Bc1, C::Rgb));
    REQUIRE(StandinHolds(S::Bc1, C::Luminance));
    REQUIRE_FALSE(StandinHolds(S::Bc1, C::LuminanceAlpha));
    REQUIRE_FALSE(StandinHolds(S::Bc1, C::Rgba));
    REQUIRE(StandinHolds(S::Bc3, C::Rgba));

    // Formats: BC4/BC5 UNORM only (sRGB falls back to BC1/BC3), 8 / 16 bytes per block.
    const HostFormat unorm = AstcRgba8StandIn(PixelFormat::ASTC_2D_8X8_UNORM, false);
    const HostFormat srgb = AstcRgba8StandIn(PixelFormat::ASTC_2D_8X8_SRGB, true);
    const HostFormat bc4 = ToBcStandin(unorm, S::Bc4);
    const HostFormat bc5 = ToBcStandin(unorm, S::Bc5);
    REQUIRE(bc4.format == VK_FORMAT_BC4_UNORM_BLOCK);
    REQUIRE(bc5.format == VK_FORMAT_BC5_UNORM_BLOCK);
    REQUIRE((bc4.bc_standin && !bc4.attachable && !bc4.storage));
    REQUIRE(BcStandinOf(bc4) == S::Bc4);
    REQUIRE(BcStandinOf(bc5) == S::Bc5);
    REQUIRE(BcStandinBlockBytes(bc4) == 8);
    REQUIRE(BcStandinBlockBytes(bc5) == 16);
    REQUIRE(ToBcStandin(srgb, S::Bc4).format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
    REQUIRE(ToBcStandin(srgb, S::Bc5).format == VK_FORMAT_BC3_SRGB_BLOCK);
    // A smart stand-in changes format from BC3 to its class's and back.
    REQUIRE(ToBcStandin(ToBcStandin(unorm, S::Bc3), S::Bc4).format == VK_FORMAT_BC4_UNORM_BLOCK);
    REQUIRE(ToBcStandin(bc4, S::Bc3).format == VK_FORMAT_BC3_UNORM_BLOCK);
    REQUIRE(std::string{AstcStandinName(S::Bc4)} == "BC4");
    REQUIRE(std::string{AstcStandinName(S::Bc5)} == "BC5");

    // Views: the storage's format, the texture's sRGB-ness; none in sRGB for BC4/BC5.
    REQUIRE(ViewsBcStandin(VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_UNORM_BLOCK));
    REQUIRE(ViewsBcStandin(VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK));
    REQUIRE_FALSE(ViewsBcStandin(VK_FORMAT_UNDEFINED, VK_FORMAT_BC4_UNORM_BLOCK));
    REQUIRE(StandinViewFormat(unorm, bc4) == VK_FORMAT_BC4_UNORM_BLOCK);
    REQUIRE(StandinViewFormat(unorm, bc5) == VK_FORMAT_BC5_UNORM_BLOCK);
    REQUIRE(StandinViewFormat(srgb, bc4) == VK_FORMAT_UNDEFINED);
    const HostFormat bc1 = ToBcStandin(unorm, S::Bc1);
    REQUIRE(StandinViewFormat(srgb, bc1) == VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
    REQUIRE(StandinViewFormat(unorm, ToBcStandin(srgb, S::Bc3)) == VK_FORMAT_BC3_UNORM_BLOCK);
    REQUIRE(StandinViewFormat(srgb, ToBcStandin(unorm, S::Bc7)) == VK_FORMAT_BC7_SRGB_BLOCK);

    // Encoder modes: a smart BC1 stand-in never punches through, the emulated path's BC1 may.
    REQUIRE(BcBlocksOf(S::Bc1, true) == BcBlocks::Bc1Opaque);
    REQUIRE(BcBlocksOf(S::Bc1, false) == BcBlocks::Bc1);
    REQUIRE(BcBlocksOf(S::Bc3, true) == BcBlocks::Bc3);
    REQUIRE(BcBlocksOf(S::Bc4, true) == BcBlocks::Bc4);
    REQUIRE(BcBlocksOf(S::Bc5, true) == BcBlocks::Bc5);
}

TEST_CASE("NVN smart BC: BC4/BC5 view swizzles read what the RGBA8 stand-in gives", "[nvn]") {
    // Proven on the mapping (no GPU sampling): for every guest swizzle of R, G, B, A, ZERO, ONE
    // (and IDENTITY), the BC4 texel (L, 0, 0, 1) and the BC5 texel (L, A, 0, 1) read through the
    // composed swizzle give what the RGBA8 stand-in's (L, L, L, 1) / (L, L, L, A) gives through
    // the guest's. BC1/BC3/BC7 keep the guest swizzle.
    constexpr std::array<VkComponentSwizzle, 7> CHOICES{
        VK_COMPONENT_SWIZZLE_R,    VK_COMPONENT_SWIZZLE_G,   VK_COMPONENT_SWIZZLE_B,
        VK_COMPONENT_SWIZZLE_A,    VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE,
        VK_COMPONENT_SWIZZLE_IDENTITY,
    };
    constexpr u8 L = 77;
    constexpr u8 A = 201;
    u32 checked = 0;
    u32 wrong = 0;
    for (u32 index = 0; index < 7 * 7 * 7 * 7; ++index) {
        const std::array<VkComponentSwizzle, 4> guest{CHOICES[index % 7], CHOICES[index / 7 % 7],
                                                      CHOICES[index / 49 % 7],
                                                      CHOICES[index / 343 % 7]};
        for (const AstcStandin kind : {AstcStandin::Bc4, AstcStandin::Bc5}) {
            const bool alpha = kind == AstcStandin::Bc5;
            const std::array<u8, 4> rgba8{L, L, L, alpha ? A : u8{255}};
            const std::array<u8, 4> bc{L, alpha ? A : u8{0}, 0, 255};
            const std::array<VkComponentSwizzle, 4> composed = ComposeStandinSwizzle(guest, kind);
            for (u32 c = 0; c < 4; ++c) {
                ++checked;
                if (Swizzled(rgba8, guest[c], c) != Swizzled(bc, composed[c], c)) {
                    ++wrong;
                }
                CHECK(composed[c] != VK_COMPONENT_SWIZZLE_IDENTITY);
            }
        }
        for (const AstcStandin kind : {AstcStandin::Bc1, AstcStandin::Bc3, AstcStandin::Bc7}) {
            CHECK(ComposeStandinSwizzle(guest, kind) == guest);
        }
    }
    WARN(fmt::format("view swizzles: {} components of {} guest swizzles checked", checked,
                     7 * 7 * 7 * 7));
    CHECK(wrong == 0);
}

TEST_CASE("NVN smart BC: BC4, BC5 and BC1 against BC3 on every layer of every level", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::SupportsBc45(device)) {
        SKIP("the device cannot run the BC encoder into BC4/BC5");
    }
    // 341x341x8 down to 1x1 (partial blocks at every level), one chunk per level as the recorder.
    const SmartEncode smart{.smart = true, .refine = true};
    const auto table = [](const char* name, const std::vector<Rgba8Level>& levels,
                          const GpuEncodeResult& ours, const GpuEncodeResult& theirs,
                          double tolerance, bool alpha_too) {
        double mean_gain = 0.0;
        u32 samples = 0;
        for (u32 level = 0; level < levels.size(); ++level) {
            std::string line = fmt::format("{} level {} ({}x{}) L{} dB, BC3 in ():", name, level,
                                           levels[level].width, levels[level].height,
                                           alpha_too ? "/A" : "");
            for (u32 layer = 0; layer < levels[level].layers; ++layer) {
                const Psnr a = LayerPsnr(levels[level], ours.decoded[level], layer);
                const Psnr b = LayerPsnr(levels[level], theirs.decoded[level], layer);
                line += alpha_too ? fmt::format(" {:.1f}/{:.1f} ({:.1f}/{:.1f})", a.rgb, a.alpha,
                                                b.rgb, b.alpha)
                                  : fmt::format(" {:.1f} ({:.1f})", a.rgb, b.rgb);
                INFO(name << " level " << level << " layer " << layer);
                CHECK(a.rgb >= b.rgb - tolerance);
                if (alpha_too) {
                    CHECK(a.alpha >= b.alpha - 0.01);
                }
                mean_gain += (std::min)(a.rgb, 60.0) - (std::min)(b.rgb, 60.0);
                ++samples;
            }
            WARN(line);
        }
        WARN(fmt::format("{}: mean gain over BC3 {:.2f} dB (capped at 60 dB)", name,
                         mean_gain / samples));
    };

    // Grey: BC4 (the luminance alone, 8-bit endpoints) against BC3 (565 colour block).
    const std::vector<Rgba8Level> grey =
        MakeMipChain(341, 341, 8, [](u32 x, u32 y, u32 l) { return GreyTexel(x, y, l, false); });
    const GpuEncodeResult grey_bc3 = EncodeOnGpu(device, grey, AstcStandin::Bc3, 0);
    const GpuEncodeResult grey_bc4 = EncodeOnGpu(device, grey, AstcStandin::Bc4, 0, true, smart);
    table("grey BC4", grey, grey_bc4, grey_bc3, 0.0, false);
    for (const Rgba8Level& level : grey_bc4.decoded) {
        for (size_t i = 3; i < level.texels.size(); i += 4) {
            REQUIRE(level.texels[i] == 255);
        }
    }

    // Grey with blend alpha: BC5 (L, A) against BC3; BC5's alpha block is BC3's.
    const std::vector<Rgba8Level> grey_alpha =
        MakeMipChain(341, 341, 8, [](u32 x, u32 y, u32 l) { return GreyTexel(x, y, l, true); });
    const GpuEncodeResult la_bc3 = EncodeOnGpu(device, grey_alpha, AstcStandin::Bc3, 0);
    const GpuEncodeResult la_bc5 =
        EncodeOnGpu(device, grey_alpha, AstcStandin::Bc5, 0, true, smart);
    table("grey+alpha BC5", grey_alpha, la_bc5, la_bc3, 0.0, true);

    // Opaque colour: smart BC1 (four-colour blocks, refined endpoints) against BC3 as today.
    const std::vector<Rgba8Level> opaque = MakeMipChain(341, 341, 8, [](u32 x, u32 y, u32 l) {
        std::array<u8, 4> texel = MaterialTexel(x, y, l);
        texel[3] = 255;
        return texel;
    });
    const GpuEncodeResult rgb_bc3 = EncodeOnGpu(device, opaque, AstcStandin::Bc3, 0);
    const GpuEncodeResult rgb_bc1 = EncodeOnGpu(device, opaque, AstcStandin::Bc1, 0, true, smart);
    table("opaque BC1", opaque, rgb_bc1, rgb_bc3, 0.25, false);
    size_t not_opaque = 0;
    for (const Rgba8Level& level : rgb_bc1.decoded) {
        for (size_t i = 3; i < level.texels.size(); i += 4) {
            not_opaque += level.texels[i] != 255 ? 1 : 0;
        }
    }
    CHECK(not_opaque == 0);
    // The grey texture as smart BC1 (BC4/BC5 switched off): as good as BC3 too.
    const GpuEncodeResult grey_bc1 = EncodeOnGpu(device, grey, AstcStandin::Bc1, 0, true, smart);
    table("grey BC1", grey, grey_bc1, grey_bc3, 0.25, false);
}

TEST_CASE("NVN smart BC: refined BC1/BC3 endpoints against the box fit", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::Supported(device)) {
        SKIP("the device cannot run the BC encoder");
    }
    const SmartEncode refined{.smart = true, .refine = true};
    for (const u32 content : {0u, 1u, 2u}) {
        const char* const name = content == 0 ? "opaque material"
                                 : content == 1 ? "blend material"
                                                : "smooth";
        const std::vector<Rgba8Level> levels =
            MakeMipChain(341, 341, 8, [content](u32 x, u32 y, u32 layer) {
                if (content == 2) {
                    const double blend = Noise(layer * 31 + 5, x, y, 48.0);
                    const double grain = Noise(layer, x, y, 9.0);
                    return std::array<u8, 4>{Clamp8(60.0 + 120.0 * blend),
                                             Clamp8(110.0 - 40.0 * blend + 20.0 * grain),
                                             Clamp8(40.0 + 50.0 * blend), 255};
                }
                std::array<u8, 4> texel = MaterialTexel(x, y, layer);
                texel[3] = content == 0 ? 255 : texel[3];
                return texel;
            });
        const GpuEncodeResult old_fit = EncodeOnGpu(device, levels, AstcStandin::Bc3, 0);
        const GpuEncodeResult new_fit =
            EncodeOnGpu(device, levels, AstcStandin::Bc3, 0, true, refined);
        double gain = 0.0;
        double worst = 99.0;
        u32 samples = 0;
        for (u32 level = 0; level < levels.size(); ++level) {
            std::string line = fmt::format("{} BC3 level {} ({}x{}) RGB dB refined (box):", name,
                                           level, levels[level].width, levels[level].height);
            for (u32 layer = 0; layer < levels[level].layers; ++layer) {
                const Psnr a = LayerPsnr(levels[level], new_fit.decoded[level], layer);
                const Psnr b = LayerPsnr(levels[level], old_fit.decoded[level], layer);
                line += fmt::format(" {:.2f} ({:.2f})", a.rgb, b.rgb);
                INFO(name << " level " << level << " layer " << layer);
                // The refined fit keeps the box fit of a block unless it measures less error with a
                // float palette; decoders round the 1/3 and 2/3 entries their own way, which can
                // cost a single-block level (2x2, 1x1) a fraction of a dB.
                CHECK(a.rgb >= b.rgb - 0.25);
                CHECK(a.alpha == b.alpha);
                gain += (std::min)(a.rgb, 60.0) - (std::min)(b.rgb, 60.0);
                worst = (std::min)(worst, a.rgb - b.rgb);
                ++samples;
            }
            WARN(line);
        }
        WARN(fmt::format("{}: refined endpoints {:+.2f} dB RGB on average over {} layer/levels "
                         "(least {:+.2f} dB)",
                         name, gain / samples, samples, worst));
    }
}

TEST_CASE("NVN smart BC: encode cost of a 1365x1365 terrain array chunk per format", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::SupportsBc45(device)) {
        SKIP("the device cannot run the BC encoder into BC4/BC5");
    }
    constexpr u32 LAYERS = 9;
    const std::vector<Rgba8Level> levels = MakeMipChain(1365, 1365, LAYERS, MaterialTexel);
    REQUIRE(levels.size() == 11);
    const SmartEncode smart{.smart = true, .refine = true};
    struct Run {
        const char* name;
        AstcStandin kind;
        bool smart;
    };
    constexpr std::array<Run, 6> RUNS{{
        {"BC3 (box fit, as today)", AstcStandin::Bc3, false},
        {"BC3 refined", AstcStandin::Bc3, true},
        {"BC1 refined", AstcStandin::Bc1, true},
        {"BC1 (box fit)", AstcStandin::Bc1, false},
        {"BC4", AstcStandin::Bc4, true},
        {"BC5", AstcStandin::Bc5, true},
    }};
    (void)EncodeOnGpu(device, levels, AstcStandin::Bc3, 0, false); // warm up
    std::array<double, RUNS.size()> per_layer{};
    for (size_t i = 0; i < RUNS.size(); ++i) {
        double best = 1e9;
        for (u32 repeat = 0; repeat < 3; ++repeat) {
            const GpuEncodeResult result = EncodeOnGpu(device, levels, RUNS[i].kind, 0, false,
                                                       RUNS[i].smart ? smart : SmartEncode{});
            best = (std::min)(best, result.gpu_ms / LAYERS);
        }
        per_layer[i] = best;
        WARN(fmt::format("{}: {:.3f} ms per 1365x1365 layer with 11 levels ({:.1f}x BC3; ~{:.0f} "
                         "ms for 90 layers)",
                         RUNS[i].name, best, best / per_layer[0], best * 90.0));
    }
    // The refined fit's budget: at most ~3x today's BC3 (with room for timing noise).
    CHECK(per_layer[1] <= per_layer[0] * 3.5);
}

TEST_CASE("NVN smart BC: the emulated path's BC1/BC3 blocks are those of another build", "[nvn]") {
    // EDEN_TEST_BC_ENCODER_BASE_SPV: another build's bc_encoder_comp_spv.h (or .spv). Without the
    // refine bit, modes 0 (BC1, punch-through) and 1 (BC3) must give the same bits.
    const char* const path = std::getenv("EDEN_TEST_BC_ENCODER_BASE_SPV");
    if (path == nullptr) {
        SKIP("EDEN_TEST_BC_ENCODER_BASE_SPV is not set");
    }
    const std::vector<u32> base = LoadSpirv(path);
    REQUIRE(base.size() > 5);
    REQUIRE(base[0] == 0x07230203);
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    if (!BcGpuEncoder::Supported(device)) {
        SKIP("the device cannot run the BC encoder");
    }
    const std::vector<std::vector<Rgba8Level>> inputs{
        MakeMipChain(341, 341, 8, MaterialTexel),
        MakeMipChain(213, 157, 3, CutoutTexel),
        MakeMipChain(130, 70, 2, [](u32 x, u32 y, u32 l) { return GreyTexel(x, y, l, true); }),
    };
    size_t compared = 0;
    for (size_t input = 0; input < inputs.size(); ++input) {
        for (const AstcStandin kind : {AstcStandin::Bc1, AstcStandin::Bc3}) {
            const GpuEncodeResult ours = EncodeOnGpu(device, inputs[input], kind, 0, false,
                                                     SmartEncode{.keep_blocks = true});
            const GpuEncodeResult theirs = EncodeOnGpu(
                device, inputs[input], kind, 0, false,
                SmartEncode{.spirv = base, .keep_blocks = true});
            INFO("input " << input << " " << AstcStandinName(kind));
            REQUIRE(ours.blocks.size() == theirs.blocks.size());
            size_t differing = 0;
            for (size_t i = 0; i < ours.blocks.size(); ++i) {
                differing += ours.blocks[i] != theirs.blocks[i] ? 1 : 0;
            }
            CHECK(differing == 0);
            compared += ours.blocks.size();
        }
    }
    WARN(fmt::format("emulated-path BC1/BC3: {} bytes of blocks compared with {}", compared, path));
}
