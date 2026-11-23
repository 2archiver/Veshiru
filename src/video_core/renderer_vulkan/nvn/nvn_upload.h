// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Turns a storage's guest bytes into what vkCmdCopyBufferToImage takes: unswizzles block-linear
// levels (Switch 2 GOB order, `native_tiling`), copies linear rows, and decodes ASTC / BCn blocks
// to RGBA8 (or the BCn stand-in format) on devices that cannot hold them. Pure CPU code over a
// Storage's shape: the tests run it without a Vulkan device.

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"

namespace Vulkan::Nvn {

/// One (layer, level) of a storage.
struct UploadPiece {
    u64 guest_offset;   ///< byte offset of the piece in the storage's guest bytes
    u64 staging_offset; ///< where the piece goes in the staging area
    u64 staging_size;
    u32 level;
    u32 layer;
    u32 cols;   ///< blocks per row of the level
    u32 rows;   ///< block rows of the level
    u32 slices; ///< depth slices
    u32 width;  ///< texel extent of the level
    u32 height;
    u32 depth;
    u32 tile_h; ///< block height / depth of the level, log2 GOBs
    u32 tile_d;
};

struct UploadPlan {
    enum class Decode : u8 { None, Astc, Bcn };

    std::vector<UploadPiece> pieces;
    u64 staging_size{};
    bool linear{};
    Decode decode{Decode::None};
    VkImageAspectFlags aspect{VK_IMAGE_ASPECT_COLOR_BIT};
    /// Set when the guest bytes cannot be turned into this storage's image (the reason is for the
    /// log); `pieces` is empty then.
    const char* unsupported{nullptr};
};

struct UploadStats {
    u64 decode_ns{};
    u64 decode_bytes{};
};

/// Layout of the upload of every layer and level of `s` whose guest bytes lie inside the first
/// `guest_size` bytes of its storage.
[[nodiscard]] UploadPlan PlanStorageUpload(const Storage& s, u64 guest_size);

/// Writes the pieces of `plan` into `staging` (plan.staging_size bytes). Large decodes run on
/// worker threads; the call returns when everything is written.
void FillStorageUpload(const Storage& s, const UploadPlan& plan, const u8* guest, u64 guest_size,
                       u8* staging, UploadStats* stats = nullptr);

/// The copy regions of `plan`, the staging area starting at `staging_base` in its buffer.
[[nodiscard]] std::vector<VkBufferImageCopy> MakeUploadRegions(const Storage& s,
                                                                const UploadPlan& plan,
                                                                u64 staging_base);

/// Bytes per texel of the decoded image of a compressed stand-in format (ASTC: 4).
[[nodiscard]] u32 DecodedTexelBytes(const HostFormat& host);

/// Decodes tightly packed compressed blocks (`block` is the format's block shape) covering a
/// width x height x slices texel region into `out` (width * height * slices * DecodedTexelBytes).
void DecodeBlocks(const HostFormat& host, const BlockInfo& block, std::span<const u8> blocks,
                  u32 width, u32 height, u32 slices, std::span<u8> out);

/// Runs `fn(i)` for i in [0, count) on the caller and on worker threads. Returns when all are done.
void RunParallel(size_t count, const std::function<void(size_t)>& fn);

/// Content hash of guest bytes, for change detection.
[[nodiscard]] u64 HashGuestBytes(const u8* data, u64 size);

/// A signature of ~16 KiB of `data`, spread evenly over it (the whole of it when it is small).
/// Equal bytes give equal signatures; a signature that differs proves the bytes differ.
/// batch_lines hashes one gathered copy of the same sampled lines (a different signature).
[[nodiscard]] u64 SampleGuestBytes(const u8* data, u64 size, bool batch_lines = false);

/// SampleGuestBytes (the same value for the same bytes) of `size` bytes that are only reachable
/// through `read(offset, count, destination)`, which has to fill `count` bytes: it is asked for the
/// sampled lines only (the whole range when `size` is small), so a gathered (virtual pool) storage
/// is not copied for a signature.
[[nodiscard]] u64 SampleGuestBytesBy(
    u64 size, const std::function<void(u64 offset, u64 count, u8* destination)>& read,
    bool batch_lines = false);

} // namespace Vulkan::Nvn
