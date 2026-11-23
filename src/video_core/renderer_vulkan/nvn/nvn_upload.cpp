// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <deque>
#include <mutex>
#include <span>
#include <thread>

#include <fmt/format.h>

#include "common/frame_stats.h"
#include "common/cityhash.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/decode_bc.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"

namespace Vulkan::Nvn {
namespace {

using VideoCore::Surface::PixelFormat;

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

u32 CeilDiv(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

VkImageAspectFlags AspectOf(const HostFormat& host) {
    if (host.depth) {
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (host.stencil) {
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

/// A small persistent pool for ASTC / BCn decoding and big unswizzles. The replay thread takes part
/// in the work, so one decode never waits for a worker that is busy with another.
class DecodePool {
public:
    DecodePool() {
        const unsigned hardware = std::thread::hardware_concurrency();
        const unsigned count = std::clamp(hardware / 4, 2u, 6u);
        for (unsigned i = 0; i < count; ++i) {
            threads.emplace_back([this] {
                Common::SetCurrentThreadName("NvnDecode");
                Run();
            });
        }
    }

    ~DecodePool() {
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        condition.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    void Execute(size_t count, const std::function<void(size_t)>& fn) {
        if (count == 0) {
            return;
        }
        auto job = std::make_shared<Job>();
        job->fn = &fn;
        job->count = count;
        const size_t helpers = (std::min)(count - 1, threads.size());
        if (helpers > 0) {
            {
                std::scoped_lock lock{mutex};
                for (size_t i = 0; i < helpers; ++i) {
                    jobs.push_back(job);
                }
            }
            condition.notify_all();
        }
        Work(*job);
        // Every index was claimed; wait for the ones helpers are still working on.
        std::unique_lock lock{job->mutex};
        job->done_condition.wait(lock, [&] { return job->finished == job->count; });
    }

private:
    struct Job {
        const std::function<void(size_t)>* fn{};
        size_t count{};
        std::atomic<size_t> next{0};
        std::mutex mutex;
        std::condition_variable done_condition;
        size_t finished{0};
    };

    static void Work(Job& job) {
        for (;;) {
            const size_t index = job.next.fetch_add(1);
            if (index >= job.count) {
                return;
            }
            (*job.fn)(index);
            std::scoped_lock lock{job.mutex};
            if (++job.finished == job.count) {
                job.done_condition.notify_all();
            }
        }
    }

    void Run() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return stopping || !jobs.empty(); });
                if (stopping) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            Work(*job);
        }
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<Job>> jobs;
    std::vector<std::thread> threads;
    bool stopping{false};
};

DecodePool& GetDecodePool() {
    static DecodePool pool;
    return pool;
}

} // Anonymous namespace

void RunParallel(size_t count, const std::function<void(size_t)>& fn) {
    if (count <= 1) {
        for (size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    GetDecodePool().Execute(count, fn);
}

u64 HashGuestBytes(const u8* data, u64 size) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::CoherencyCheck};
    return Common::CityHash64(reinterpret_cast<const char*>(data), static_cast<size_t>(size));
}

u64 SampleGuestBytes(const u8* data, u64 size, bool batch_lines) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::CoherencyCheck};
    constexpr u64 LINE = 64;
    constexpr u64 LINES = 256;
    if (size <= LINE * LINES) {
        return HashGuestBytes(data, size);
    }
    // Evenly spaced 64-byte lines, the first and the last one included.
    const u64 last_line = (size - LINE) / LINE;
    if (batch_lines) {
        std::array<u8, LINE * LINES> samples;
        for (u64 i = 0; i < LINES; ++i) {
            const u64 offset = (last_line * i / (LINES - 1)) * LINE;
            std::memcpy(samples.data() + i * LINE, data + offset, LINE);
        }
        return Common::CityHash64(reinterpret_cast<const char*>(samples.data()), samples.size());
    }
    u64 hash = 0;
    for (u64 i = 0; i < LINES; ++i) {
        const u64 offset = (last_line * i / (LINES - 1)) * LINE;
        hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(data + offset), LINE, hash);
    }
    return hash;
}

u64 SampleGuestBytesBy(u64 size,
                       const std::function<void(u64 offset, u64 count, u8* destination)>& read,
                       bool batch_lines) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::CoherencyCheck};
    constexpr u64 LINE = 64;
    constexpr u64 LINES = 256;
    if (size <= LINE * LINES) {
        std::array<u8, LINE * LINES> all;
        read(0, size, all.data());
        return Common::CityHash64(reinterpret_cast<const char*>(all.data()),
                                  static_cast<size_t>(size));
    }
    // The same lines, in the same order, as SampleGuestBytes.
    const u64 last_line = (size - LINE) / LINE;
    if (batch_lines) {
        std::array<u8, LINE * LINES> samples;
        for (u64 i = 0; i < LINES; ++i) {
            const u64 offset = (last_line * i / (LINES - 1)) * LINE;
            read(offset, LINE, samples.data() + i * LINE);
        }
        return Common::CityHash64(reinterpret_cast<const char*>(samples.data()), samples.size());
    }
    u64 hash = 0;
    std::array<u8, LINE> line;
    for (u64 i = 0; i < LINES; ++i) {
        const u64 offset = (last_line * i / (LINES - 1)) * LINE;
        read(offset, LINE, line.data());
        hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(line.data()), LINE, hash);
    }
    return hash;
}

u32 DecodedTexelBytes(const HostFormat& host) {
    return VideoCore::Surface::IsPixelFormatBCn(host.pixel_format)
               ? VideoCommon::ConvertedBytesPerBlock(host.pixel_format)
               : 4u;
}

void DecodeBlocks(const HostFormat& host, const BlockInfo& block, std::span<const u8> blocks,
                  u32 width, u32 height, u32 slices, std::span<u8> out) {
    if (VideoCore::Surface::IsPixelFormatASTC(host.pixel_format)) {
        Tegra::Texture::ASTC::Decompress(blocks, width, height, slices, block.width, block.height,
                                         out);
        return;
    }
    VideoCommon::BufferImageCopy copy{};
    copy.image_subresource.num_layers = 1;
    copy.image_extent = {width, height, slices};
    VideoCommon::DecompressBCn(blocks, out, copy, host.pixel_format);
}

/// EDEN_NX2_NVN_DUMP_ASTC=<dir>: the base level of every decoded ASTC texture of at least 256
/// texels (TGA, at most 96 files), named by size, format and order, to look at what the decoder
/// produced.
static void DumpDecodedLevel(const Storage& s, const UploadPlan& plan, const UploadPiece& piece,
                             const u8* rgba) {
    static const char* const directory = std::getenv("EDEN_NX2_NVN_DUMP_ASTC");
    static std::atomic<u32> written{0};
    if (directory == nullptr || *directory == 0 || plan.decode != UploadPlan::Decode::Astc ||
        piece.level != 0 || piece.layer != 0 || piece.width < 256 || piece.slices != 1 ||
        written.load(std::memory_order_relaxed) >= 96) {
        return;
    }
    const u32 index = written.fetch_add(1, std::memory_order_relaxed);
    if (index >= 96) {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    const std::string path = fmt::format("{}/astc-{:03}-{}x{}-fmt{:x}.tga", directory, index,
                                         piece.width, piece.height, s.desc.format);
    if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
        const std::array<u8, 18> header{0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        static_cast<u8>(piece.width & 0xFF),
                                        static_cast<u8>(piece.width >> 8),
                                        static_cast<u8>(piece.height & 0xFF),
                                        static_cast<u8>(piece.height >> 8), 32, 0x28};
        std::fwrite(header.data(), 1, header.size(), file);
        std::vector<u8> bgra(static_cast<size_t>(piece.width) * piece.height * 4);
        for (size_t i = 0; i < bgra.size(); i += 4) {
            bgra[i] = rgba[i + 2];
            bgra[i + 1] = rgba[i + 1];
            bgra[i + 2] = rgba[i];
            bgra[i + 3] = rgba[i + 3];
        }
        std::fwrite(bgra.data(), 1, bgra.size(), file);
        std::fclose(file);
    }
}

UploadPlan PlanStorageUpload(const Storage& s, u64 guest_size) {
    UploadPlan plan;
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!block.bytes) {
        plan.unsupported = "the format has no guest layout";
        return plan;
    }
    if (s.host.guest_layout_differs) {
        plan.unsupported = "depth/stencil texel layout differs between guest and host";
        return plan;
    }
    plan.aspect = AspectOf(s.host);
    plan.linear = IsLinearStorage(s.desc);
    u32 converted_bytes = 0;
    if (s.host.needs_transcode) {
        if (VideoCore::Surface::IsPixelFormatASTC(s.host.pixel_format)) {
            plan.decode = UploadPlan::Decode::Astc;
            converted_bytes = 4;
        } else if (VideoCore::Surface::IsPixelFormatBCn(s.host.pixel_format)) {
            plan.decode = UploadPlan::Decode::Bcn;
            converted_bytes = VideoCommon::ConvertedBytesPerBlock(s.host.pixel_format);
        } else {
            plan.unsupported = "no decoder for the compressed format";
            return plan;
        }
    }
    const u32 layer_count = plan.linear ? 1 : s.layers;
    const u32 level_count = plan.linear ? 1 : s.levels;
    u64 total = 0;
    for (u32 layer = 0; layer < layer_count; ++layer) {
        for (u32 level = 0; level < level_count; ++level) {
            UploadPiece piece{};
            piece.level = level;
            piece.layer = layer;
            piece.width = Mip(s.extent.width, level);
            piece.height = Mip(s.extent.height, level);
            piece.depth = s.type == VK_IMAGE_TYPE_3D ? Mip(s.extent.depth, level) : 1u;
            piece.cols = CeilDiv(piece.width, block.width);
            piece.rows = CeilDiv(piece.height, block.height);
            piece.slices = piece.depth;
            piece.guest_offset = static_cast<u64>(layer) * s.layout.layer_size +
                                 (plan.linear ? 0 : TextureLevelOffset(s.desc, level));
            if (piece.guest_offset >= guest_size) {
                continue;
            }
            TextureLevelTiles(s.desc, level, piece.tile_h, piece.tile_d);
            switch (plan.decode) {
            case UploadPlan::Decode::None:
                piece.staging_size = static_cast<u64>(piece.cols) * block.bytes * piece.rows *
                                     piece.slices;
                break;
            case UploadPlan::Decode::Astc:
            case UploadPlan::Decode::Bcn:
                piece.staging_size = static_cast<u64>(piece.width) * piece.height * piece.slices *
                                     converted_bytes;
                break;
            }
            total = (total + 63) / 64 * 64;
            piece.staging_offset = total;
            total += piece.staging_size;
            plan.pieces.push_back(piece);
        }
    }
    plan.staging_size = total;
    return plan;
}

void FillStorageUpload(const Storage& s, const UploadPlan& plan, const u8* guest, u64 guest_size,
                       u8* staging, UploadStats* stats) {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    const auto begin = std::chrono::steady_clock::now();
    const auto fill = [&](size_t index) {
        const UploadPiece& piece = plan.pieces[index];
        u8* destination = staging + piece.staging_offset;
        const u64 block_bytes = static_cast<u64>(piece.cols) * block.bytes * piece.rows * piece.slices;
        std::vector<u8> blocks;
        u8* block_destination = destination;
        if (plan.decode != UploadPlan::Decode::None) {
            blocks.resize(static_cast<size_t>(block_bytes));
            block_destination = blocks.data();
        }
        if (plan.linear) {
            // Rows at the texture's pitch; the copy reads the pitch straight from the guest bytes.
            const u64 pitch = s.layout.layer_size / (std::max)(piece.height, 1u);
            std::memset(block_destination, 0, static_cast<size_t>(block_bytes));
            for (u32 row = 0; row < piece.rows; ++row) {
                const u64 source = piece.guest_offset + row * pitch;
                if (source >= guest_size) {
                    break;
                }
                const u64 row_bytes = static_cast<u64>(piece.cols) * block.bytes;
                std::memcpy(block_destination + row * row_bytes, guest + source,
                            static_cast<size_t>((std::min)(row_bytes, guest_size - source)));
            }
        }
        if (!plan.linear) {
            // The unswizzle reads whole GOBs: a storage that stops short of the last one is padded.
            const u64 needed = Tegra::Texture::CalculateSize(true, block.bytes, piece.cols,
                                                             piece.rows, piece.slices,
                                                             piece.tile_h, piece.tile_d);
            const u64 available = guest_size - piece.guest_offset;
            std::vector<u8> padded;
            std::span<const u8> input{guest + piece.guest_offset, static_cast<size_t>(available)};
            if (available < needed) {
                padded.assign(static_cast<size_t>(needed), 0);
                std::memcpy(padded.data(), input.data(), static_cast<size_t>(available));
                input = padded;
            }
            Tegra::Texture::UnswizzleTexture(
                std::span<u8>{block_destination, static_cast<size_t>(block_bytes)}, input,
                block.bytes, piece.cols, piece.rows, piece.slices, piece.tile_h, piece.tile_d, 0,
                true);
        }
        if (plan.decode != UploadPlan::Decode::None) {
            DecodeBlocks(s.host, block, blocks, piece.width, piece.height, piece.slices,
                         std::span<u8>{destination, static_cast<size_t>(piece.staging_size)});
            DumpDecodedLevel(s, plan, piece, destination);
        }
    };
    // Plain copies are memory-bound and small: only decodes and big unswizzles fan out.
    const bool heavy = plan.decode != UploadPlan::Decode::None || plan.staging_size >= (u64{1} << 20);
    if (heavy) {
        RunParallel(plan.pieces.size(), fill);
    } else {
        for (size_t i = 0; i < plan.pieces.size(); ++i) {
            fill(i);
        }
    }
    if (stats && plan.decode != UploadPlan::Decode::None) {
        stats->decode_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - begin)
                                                 .count());
        stats->decode_bytes += plan.staging_size;
    }
}

std::vector<VkBufferImageCopy> MakeUploadRegions(const Storage& s, const UploadPlan& plan,
                                                  u64 staging_base) {
    static_cast<void>(s);
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(plan.pieces.size());
    for (const UploadPiece& piece : plan.pieces) {
        regions.push_back({
            .bufferOffset = staging_base + piece.staging_offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = plan.aspect,
                .mipLevel = piece.level,
                .baseArrayLayer = piece.layer,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {piece.width, piece.height, piece.depth},
        });
    }
    return regions;
}

} // namespace Vulkan::Nvn
