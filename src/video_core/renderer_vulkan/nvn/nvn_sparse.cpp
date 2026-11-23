// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <tuple>
#include <utility>

#include "common/logging.h"
#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"

namespace Vulkan::Nvn {
namespace {

/// Sparse storages smaller than this stay ordinary images (a remap re-uploads them whole, which is
/// cheap at this size).
constexpr u64 MIN_SPARSE_STORAGE_BYTES = u64{8} << 20;

/// Tiles one `NextVerifyTile` call looks at before giving up.
constexpr u32 VERIFY_SCAN_LIMIT = 4096;

u32 CeilDiv(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

} // Anonymous namespace

// Row batching follows mkw-recomp's 30af22de9062367d3831d3787d136a6f43ebe275
// (runtime/gfx/vk/tex/vk_sparse.c). Vulkan lays out the blocks of a bind x first.
void CompactSparseImageBinds(std::vector<VkSparseImageMemoryBind>& binds, u32 granule_x,
                             u32 granule_y) {
    if (binds.size() < 2 || granule_x == 0 || granule_y == 0) {
        return;
    }
    const auto position = [](const VkSparseImageMemoryBind& bind) {
        return std::tuple{bind.subresource.aspectMask,
                          bind.subresource.arrayLayer,
                          bind.subresource.mipLevel,
                          bind.offset.z,
                          bind.offset.y,
                          bind.offset.x};
    };
    std::ranges::sort(binds, {}, position);
    size_t count = 0;
    for (const VkSparseImageMemoryBind bind : binds) {
        if (count != 0) {
            auto& previous = binds[count - 1];
            const bool same_row =
                previous.subresource.aspectMask == bind.subresource.aspectMask &&
                previous.subresource.arrayLayer == bind.subresource.arrayLayer &&
                previous.subresource.mipLevel == bind.subresource.mipLevel &&
                previous.offset.y == bind.offset.y && previous.offset.z == bind.offset.z &&
                previous.extent.height == bind.extent.height &&
                previous.extent.height <= granule_y && previous.extent.depth == 1 &&
                bind.extent.depth == 1 && previous.extent.width % granule_x == 0 &&
                static_cast<s64>(previous.offset.x) + previous.extent.width == bind.offset.x;
            const bool same_memory =
                previous.memory == bind.memory && previous.flags == bind.flags &&
                (bind.memory == VK_NULL_HANDLE ||
                 previous.memoryOffset + (previous.extent.width / granule_x) * SPARSE_TILE_BYTES ==
                     bind.memoryOffset);
            if (same_row && same_memory) {
                previous.extent.width += bind.extent.width;
                continue;
            }
        }
        binds[count++] = bind;
    }
    binds.resize(count);
}

// ---- SparsePagePool ---------------------------------------------------------------------------

SparsePagePool::SparsePagePool(const MemoryAllocator& allocator, u64 budget_bytes)
    : allocator_{allocator}, budget_{budget_bytes} {}

SparsePagePool::~SparsePagePool() {
    for (const Block& block : blocks_) {
        allocator_.FreeSparseMemory(block.allocation);
    }
}

u32 SparsePagePool::Allocate(u32 memory_type_bits) {
    std::scoped_lock lock{mutex_};
    return AllocateLocked(memory_type_bits);
}

u32 SparsePagePool::AcquireShared(const SparseAliasKey& key, u32 memory_type_bits, u64 epoch,
                                  bool& reused, bool allow_alloc) {
    std::scoped_lock lock{mutex_};
    if (const auto found = by_key_.find(key); found != by_key_.end()) {
        const u32 page = found->second;
        ++refs_[page];
        ++shared_extra_refs_;
        reused = epoch_[page] < epoch && stale_[page] == 0;
        return page;
    }
    reused = false;
    if (!allow_alloc) {
        return NoPage;
    }
    const u32 page = AllocateLocked(memory_type_bits);
    if (page != NoPage) {
        by_key_.emplace(key, page);
        key_of_.emplace(page, key);
        epoch_[page] = epoch;
        stale_[page] = 0;
    }
    return page;
}

bool SparsePagePool::IsShared(u32 page) const {
    std::scoped_lock lock{mutex_};
    return key_of_.contains(page);
}

bool SparsePagePool::HoldsKey(u32 page, const SparseAliasKey& key) const {
    std::scoped_lock lock{mutex_};
    const auto found = key_of_.find(page);
    return found != key_of_.end() && found->second == key;
}

void SparsePagePool::MarkSharedStale(u32 page) {
    if (page == NoPage) {
        return;
    }
    std::scoped_lock lock{mutex_};
    if (page < stale_.size()) {
        stale_[page] = 1;
    }
}

void SparsePagePool::ClearSharedStale(u32 page) {
    if (page == NoPage) {
        return;
    }
    std::scoped_lock lock{mutex_};
    if (page < stale_.size()) {
        stale_[page] = 0;
    }
}

u32 SparsePagePool::AllocateLocked(u32 memory_type_bits) {
    if (used_ + PageBytes > budget_) {
        return NoPage;
    }
    // A block was allocated for one mask of allowed memory types and the allocator picked one of
    // them (it does not say which): only an image with the same mask is sure to accept it.
    const auto usable = [&](u32 page) {
        return blocks_[page / PagesPerBlock].type_bits == memory_type_bits;
    };
    for (size_t i = free_.size(); i-- > 0;) {
        if (usable(free_[i])) {
            const u32 page = free_[i];
            free_[i] = free_.back();
            free_.pop_back();
            if (blocks_[page / PagesPerBlock].used_pages++ == 0) {
                --empty_blocks_;
            }
            used_ += PageBytes;
            refs_[page] = 1;
            stale_[page] = 0;
            return page;
        }
    }
    const VkMemoryRequirements requirements{
        .size = PageBytes * PagesPerBlock,
        .alignment = PageBytes,
        .memoryTypeBits = memory_type_bits,
    };
    Block block;
    block.allocation =
        allocator_.AllocateSparseMemory(requirements, block.where.memory, block.where.offset);
    if (block.allocation == VK_NULL_HANDLE) {
        return NoPage;
    }
    block.type_bits = memory_type_bits;
    block.used_pages = 1;
    const auto vacant = std::ranges::find_if(
        blocks_, [](const Block& candidate) { return candidate.allocation == VK_NULL_HANDLE; });
    const u32 index = static_cast<u32>(vacant - blocks_.begin());
    const u32 first = index * PagesPerBlock;
    if (vacant == blocks_.end()) {
        blocks_.push_back(block);
    } else {
        *vacant = block;
    }
    for (u32 i = PagesPerBlock; i-- > 1;) {
        free_.push_back(first + i);
    }
    if (refs_.size() < blocks_.size() * PagesPerBlock) {
        refs_.resize(blocks_.size() * PagesPerBlock, 0);
        epoch_.resize(blocks_.size() * PagesPerBlock, 0);
        stale_.resize(blocks_.size() * PagesPerBlock, 0);
    }
    refs_[first] = 1;
    stale_[first] = 0;
    used_ += PageBytes;
    return first;
}

void SparsePagePool::Free(u32 page) {
    if (page == NoPage) {
        return;
    }
    std::scoped_lock lock{mutex_};
    if (refs_[page] > 1) {
        --refs_[page];
        --shared_extra_refs_;
        return;
    }
    refs_[page] = 0;
    stale_[page] = 0;
    if (const auto found = key_of_.find(page); found != key_of_.end()) {
        by_key_.erase(found->second);
        key_of_.erase(found);
    }
    free_.push_back(page);
    if (--blocks_[page / PagesPerBlock].used_pages == 0) {
        ++empty_blocks_;
    }
    used_ -= PageBytes;
    ++free_generation_;
}

void SparsePagePool::TrimUnusedBlocks() {
    std::scoped_lock lock{mutex_};
    if (empty_blocks_ <= CachedBlockLimit) {
        return;
    }
    u32 kept = 0;
    for (Block& block : blocks_) {
        if (block.allocation == VK_NULL_HANDLE || block.used_pages != 0) {
            continue;
        }
        if (kept++ < CachedBlockLimit) {
            continue;
        }
        allocator_.FreeSparseMemory(block.allocation);
        block = {};
        --empty_blocks_;
    }
    std::erase_if(free_, [&](u32 page) {
        return blocks_[page / PagesPerBlock].allocation == VK_NULL_HANDLE;
    });
    ++free_generation_; // a failed allocation may fit after releasing another format's blocks
}

SparsePagePool::Where SparsePagePool::Get(u32 page) const {
    std::scoped_lock lock{mutex_};
    const Block& block = blocks_[page / PagesPerBlock];
    return {block.where.memory, block.where.offset + PageBytes * (page % PagesPerBlock)};
}

u64 SparsePagePool::UsedBytes() const {
    std::scoped_lock lock{mutex_};
    return used_;
}

u64 SparsePagePool::ReservedBytes() const {
    std::scoped_lock lock{mutex_};
    return static_cast<u64>(std::ranges::count_if(
               blocks_, [](const Block& block) { return block.allocation != VK_NULL_HANDLE; })) *
           PageBytes * PagesPerBlock;
}

bool SparsePagePool::HasRoom() const {
    std::scoped_lock lock{mutex_};
    return used_ + PageBytes <= budget_;
}

// ---- SparseBacking ----------------------------------------------------------------------------

std::shared_ptr<SparseBacking> SparseBacking::Make(const Storage& s, const Device& device,
                                                   u64 pool_size, const char*& reason) {
    if (!device.IsSparseResidencyImageSupported(VK_IMAGE_TYPE_2D) ||
        !device.CanUseUnbackedSparseImages()) {
        reason = "the device has no sparse residency images";
        return nullptr;
    }
    if (!(s.desc.flags & TexFlag::Sparse)) {
        reason = "the texture is not sparse";
        return nullptr;
    }
    if (s.layout.size < MIN_SPARSE_STORAGE_BYTES) {
        reason = "smaller than 8 MiB";
        return nullptr;
    }
    if (s.pool_offset % SPARSE_TILE_BYTES != 0 || s.pool_offset > pool_size ||
        s.layout.size > pool_size - s.pool_offset) {
        reason = "not 64 KiB aligned or leaves its pool";
        return nullptr;
    }
    if (s.host.needs_transcode && !VideoCore::Surface::IsPixelFormatASTC(s.host.pixel_format) &&
        !VideoCore::Surface::IsPixelFormatBCn(s.host.pixel_format)) {
        reason = "the format has no decoder";
        return nullptr;
    }
    std::optional<SparseLayout> layout = BuildSparseLayout(s);
    if (!layout) {
        reason = "no tile layout (not 2D, multisampled, packaged, or too small for tiles)";
        return nullptr;
    }
    return FromLayout(std::move(*layout));
}

std::shared_ptr<SparseBacking> SparseBacking::FromLayout(SparseLayout layout) {
    std::shared_ptr<SparseBacking> backing{new SparseBacking};
    backing->layout_ = std::move(layout);
    backing->tile_flags_.assign(backing->layout_.TileCount(), 0);
    backing->tail_flags_.assign(backing->layout_.layers, 0);
    backing->tile_hash_.assign(backing->layout_.TileCount(), 0);
    return backing;
}

SparseBacking::~SparseBacking() {
    if (!pool_) {
        return;
    }
    for (const u32 page : tile_pages_) {
        pool_->Free(page);
    }
    for (const std::vector<u32>& pages : tail_pages_) {
        for (const u32 page : pages) {
            pool_->Free(page);
        }
    }
    pool_->TrimUnusedBlocks();
}

bool SparseBacking::Attach(Context::Impl& impl, const Storage& s, const char*& reason) {
    const auto requirements = impl.device.GetLogical().GetImageSparseMemoryRequirements(*s.image);
    const VkImageAspectFlags wanted_aspect =
        s.host.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    const VkSparseImageMemoryRequirements* color = nullptr;
    for (const VkSparseImageMemoryRequirements& entry : requirements) {
        if (entry.formatProperties.aspectMask == wanted_aspect) {
            color = &entry;
            break;
        }
    }
    if (!color) {
        reason = "the image reports no sparse requirements for its aspect";
        return false;
    }
    const VkSparseImageFormatProperties& props = color->formatProperties;
    if ((props.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0) {
        reason = "the mip tail is shared by the layers";
        return false;
    }
    if (props.imageGranularity.depth != 1 || props.imageGranularity.width == 0 ||
        props.imageGranularity.height == 0) {
        reason = "no 2D sparse block shape";
        return false;
    }
    const VkMemoryRequirements memory =
        impl.device.GetLogical().GetImageMemoryRequirements(*s.image);
    granule_x_ = props.imageGranularity.width;
    granule_y_ = props.imageGranularity.height;
    const u64 granule_bytes =
        s.host.bc_standin ? static_cast<u64>(CeilDiv(granule_x_, 4)) * CeilDiv(granule_y_, 4) *
                                BcStandinBlockBytes(s.host)
        : s.host.needs_transcode
            ? static_cast<u64>(granule_x_) * granule_y_ * DecodedTexelBytes(s.host)
            : static_cast<u64>(granule_x_) * granule_y_ * layout_.block.bytes /
                  (layout_.block.width * layout_.block.height);
    if (memory.alignment != SPARSE_TILE_BYTES || granule_bytes != SPARSE_TILE_BYTES) {
        reason = "the host sparse block is not 64 KiB";
        return false;
    }
    if (layout_.tile_width % granule_x_ != 0 || layout_.tile_height % granule_y_ != 0) {
        reason = "a guest tile is not a whole number of host sparse blocks";
        return false;
    }
    host_per_x_ = layout_.tile_width / granule_x_;
    host_per_y_ = layout_.tile_height / granule_y_;
    tail_first_lod_ = (std::min)(color->imageMipTailFirstLod, s.levels);
    if (tail_first_lod_ < layout_.level.size()) {
        reason = "the host mip tail starts inside the tiled levels";
        return false;
    }
    memory_type_bits_ = memory.memoryTypeBits;
    aspect_ = props.aspectMask;
    // Tiles of the same guest bytes decode to the same texels when everything that shapes the
    // decode and the host block layout matches: guest and host formats, the stand-in, the tile and
    // host block shape, and the image's flags, usage and memory types.
    share_ = impl.SparseShareEnabled();
    {
        u64 h = 0xCBF29CE484222325ULL;
        const auto mix = [&h](u64 value) {
            h ^= value + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        };
        mix(s.desc.format);
        mix(static_cast<u64>(s.host.format));
        mix(s.host.needs_transcode ? 1 : 0);
        mix(s.host.bc_standin ? 1 : 0);
        mix(s.create_flags);
        mix(s.usage);
        mix(layout_.tile_width);
        mix(layout_.tile_height);
        mix(granule_x_);
        mix(granule_y_);
        mix(memory_type_bits_);
        mix(aspect_);
        shape_ = h;
    }
    tail_size_ = 0;
    if (tail_first_lod_ < s.levels) {
        tail_size_ = color->imageMipTailSize;
        tail_resource_offset_ = color->imageMipTailOffset;
        tail_stride_ = color->imageMipTailStride;
        if (tail_size_ == 0 || tail_size_ % SPARSE_TILE_BYTES != 0 ||
            tail_resource_offset_ % SPARSE_TILE_BYTES != 0 ||
            (layout_.layers > 1 && tail_stride_ % SPARSE_TILE_BYTES != 0)) {
            reason = "the host mip tail is not made of 64 KiB pages";
            return false;
        }
    }
    if (!layout_.HasTail() && tail_first_lod_ != s.levels) {
        reason = "the host has a mip tail the guest layout does not";
        return false;
    }
    // Levels the guest keeps in its tail but the host binds in sparse blocks.
    tail_host_tiles_.clear();
    u32 slot = 0;
    for (u32 level = static_cast<u32>(layout_.level.size()); level < tail_first_lod_; ++level) {
        const u32 width = (std::max)(s.extent.width >> level, 1u);
        const u32 height = (std::max)(s.extent.height >> level, 1u);
        for (u32 y = 0; y < CeilDiv(height, granule_y_); ++y) {
            for (u32 x = 0; x < CeilDiv(width, granule_x_); ++x) {
                tail_host_tiles_.push_back({level, x * granule_x_, y * granule_y_,
                                            (std::min)(granule_x_, width - x * granule_x_),
                                            (std::min)(granule_y_, height - y * granule_y_),
                                            slot++});
            }
        }
    }
    tile_pages_.assign(static_cast<size_t>(layout_.TileCount()) * host_per_x_ * host_per_y_,
                       SparsePagePool::NoPage);
    tile_verified_pool_.assign(layout_.TileCount(), 0);
    tile_verified_gen_.assign(layout_.TileCount(), 0);
    tail_pages_.assign(layout_.layers, {});
    tail_plan_ = PlanSparseTail(s, layout_);
    if (tail_plan_.unsupported) {
        reason = tail_plan_.unsupported;
        return false;
    }
    if (!impl.sparse_pages) {
        const auto properties = impl.device.GetPhysical().GetMemoryProperties().memoryProperties;
        u64 local = 0;
        for (u32 i = 0; i < properties.memoryHeapCount; ++i) {
            const auto& heap = properties.memoryHeaps[i];
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                local += heap.size;
            }
        }
        // MKW also respects the usable budget and simulated heap limit. Keep physical memory
        // as the fallback on devices without VK_EXT_memory_budget.
        if (impl.mario_kart_world) {
            const u64 budget = impl.device.GetDeviceMemoryBudget();
            if (budget != 0) {
                local = (std::min)(local, budget);
            }
        }
        const u64 device_budget = impl.options.sparse_device_budget_bytes.value_or(
            (std::clamp)(local / 5 * 2, u64{1} << 30, u64{12} << 30));
        impl.sparse_pages = std::make_shared<SparsePagePool>(*impl.allocator, device_budget);
    }
    pool_ = impl.sparse_pages;
    // MKW's foliage/lightmap atlas copies write the coarse levels without MapVirtual for
    // their guest tail bytes. Keep host backing for these levels, as mkw-recomp does in
    // runtime/gfx/vk/tex/vk_sparse.c (tail_alloc), rather than discarding those GPU writes.
    persistent_tails_ = impl.mario_kart_world && layout_.HasTail();
    if (persistent_tails_) {
        for (u32 layer = 0; layer < layout_.layers; ++layer) {
            Queue(tail_flags_[layer], layer, true);
        }
    }
    return true;
}

void SparseBacking::Queue(u8& flags, u32 id, bool tail) {
    starved_only_ = false;
    if ((flags & Queued) == 0) {
        flags |= Queued;
        (tail ? pending_tails_ : pending_tiles_).push_back(id);
    }
}

void SparseBacking::SetMapped(u8& flags, u32 id, bool tail, bool mapped) {
    if (!tail && id < tile_verified_gen_.size()) {
        tile_verified_pool_[id] = 0;
        tile_verified_gen_[id] = 0;
    }
    const bool was = (flags & Mapped) != 0;
    if (mapped) {
        if (!was) {
            flags |= Mapped;
            mapped_tiles_ += tail ? 0 : 1;
        }
        flags |= Dirty;
        Queue(flags, id, tail);
    } else {
        if (was) {
            flags &= static_cast<u8>(~Mapped);
            mapped_tiles_ -= tail ? 0 : 1;
        }
        flags &= static_cast<u8>(~Dirty);
        if ((flags & Bound) != 0) {
            Queue(flags, id, tail);
        }
    }
}

void SparseBacking::SetTilePagesStale(u32 id, bool stale) {
    if (!share_ || !pool_) {
        return;
    }
    const u32 slots = host_per_x_ * host_per_y_;
    const u32* pages = &tile_pages_[static_cast<size_t>(id) * slots];
    for (u32 slot = 0; slot < slots; ++slot) {
        if (stale) {
            pool_->MarkSharedStale(pages[slot]);
        } else {
            pool_->ClearSharedStale(pages[slot]);
        }
    }
}

void SparseBacking::ContentChanged(u64 begin, u64 end) {
    layout_.ForEachUnit(
        begin, end,
        [&](u32 id) {
            if (id < tile_verified_gen_.size()) {
                tile_verified_pool_[id] = 0;
                tile_verified_gen_[id] = 0;
            }
            if ((tile_flags_[id] & Mapped) != 0) {
                tile_flags_[id] |= Dirty;
                if ((tile_flags_[id] & Bound) != 0) {
                    SetTilePagesStale(id, true);
                }
                Queue(tile_flags_[id], id, false);
            }
        },
        [&](u32 layer) {
            if ((tail_flags_[layer] & Mapped) != 0) {
                tail_flags_[layer] |= Dirty;
                Queue(tail_flags_[layer], layer, true);
            }
        });
}

void SparseBacking::MarkAllDirty() {
    ContentChanged(0, static_cast<u64>(layout_.layers) * layout_.layer_size);
}

void SparseBacking::MarkTileDirty(u32 id) {
    if (id < tile_verified_gen_.size()) {
        tile_verified_pool_[id] = 0;
        tile_verified_gen_[id] = 0;
    }
    if ((tile_flags_[id] & Mapped) != 0) {
        tile_flags_[id] |= Dirty;
        if ((tile_flags_[id] & Bound) != 0) {
            SetTilePagesStale(id, true);
        }
        Queue(tile_flags_[id], id, false);
    }
}

u32 SparseBacking::NextVerifyTile() {
    const u32 count = layout_.TileCount();
    for (u32 scanned = 0; scanned < (std::min)(count, VERIFY_SCAN_LIMIT); ++scanned) {
        const u32 id = verify_cursor_;
        verify_cursor_ = verify_cursor_ + 1 >= count ? 0 : verify_cursor_ + 1;
        constexpr u8 want = Mapped | Bound;
        if ((tile_flags_[id] & (want | Dirty | Queued)) == want) {
            return id;
        }
    }
    return ~0u;
}

void SparseBacking::HostTilesOfTile(u32 id, std::vector<HostTile>& out) const {
    out.clear();
    const SparseTileRef ref = layout_.Tile(id);
    const SparseLevel& level = layout_.level[ref.level];
    for (u32 ky = 0; ky < host_per_y_; ++ky) {
        for (u32 kx = 0; kx < host_per_x_; ++kx) {
            const u32 x = ref.x * layout_.tile_width + kx * granule_x_;
            const u32 y = ref.y * layout_.tile_height + ky * granule_y_;
            const u32 slot = ky * host_per_x_ + kx;
            if (x >= level.width || y >= level.height) {
                continue; // past the edge of the level: the host has no block there
            }
            out.push_back({ref.level, x, y, (std::min)(granule_x_, level.width - x),
                           (std::min)(granule_y_, level.height - y), slot});
        }
    }
}

bool SparseBacking::TileSource(Context::Impl& impl, const Storage& s, u32 id,
                               u64& source) const {
    const auto virtual_it = impl.pools.find(s.pool_serial);
    if (virtual_it == impl.pools.end()) {
        return false;
    }
    const auto& runs = virtual_it->second->virtual_runs;
    const u64 offset = s.pool_offset + layout_.TileOffset(id);
    auto run = runs.upper_bound(offset);
    if (run == runs.begin()) {
        return false;
    }
    --run;
    if (offset + SPARSE_TILE_BYTES > run->first + run->second.size) {
        return false; // not one run: no single source to share by
    }
    const auto physical_it = impl.pools.find(run->second.physical_serial);
    if (physical_it == impl.pools.end()) {
        return false;
    }
    source = physical_it->second->address + run->second.physical_offset + (offset - run->first);
    return true;
}

bool SparseBacking::TilePagesCurrent(Context::Impl& impl, const Storage& s, u32 id) {
    const u32 slots = host_per_x_ * host_per_y_;
    const u32* pages = &tile_pages_[static_cast<size_t>(id) * slots];
    u64 source = 0;
    const bool shareable = TileSource(impl, s, id, source);
    HostTilesOfTile(id, tile_scratch_);
    for (const HostTile& tile : tile_scratch_) {
        const u32 page = pages[tile.slot];
        if (page == SparsePagePool::NoPage) {
            return false;
        }
        // A private page stays right for an unshareable tile; a shared one only for its key.
        if (shareable) {
            if (!pool_->HoldsKey(page, KeyOf(source, tile.slot, tile.width, tile.height))) {
                return false;
            }
        } else if (pool_->IsShared(page)) {
            return false;
        }
    }
    return true;
}

bool SparseBacking::BindUnit(Context::Impl& impl, const Storage& s, bool tail, u32 id, bool bind,
                             std::vector<VkSparseImageMemoryBind>& image_binds,
                             std::vector<VkSparseMemoryBind>& opaque_binds, bool* reused,
                             bool allow_alloc) {
    const u32 layer = tail ? id : layout_.Tile(id).layer;
    std::span<const HostTile> tiles;
    u32 slots = 0;
    u32* pages = nullptr;
    std::vector<u32>* tail_vector = nullptr;
    u32 opaque_count = 0;
    if (tail) {
        tiles = tail_host_tiles_;
        opaque_count = static_cast<u32>(tail_size_ / SPARSE_TILE_BYTES);
        tail_vector = &tail_pages_[id];
        if (bind) {
            tail_vector->assign(tiles.size() + opaque_count, SparsePagePool::NoPage);
        }
        slots = static_cast<u32>(tail_vector->size());
        pages = tail_vector->data();
    } else {
        HostTilesOfTile(id, tile_scratch_);
        tiles = tile_scratch_;
        slots = host_per_x_ * host_per_y_;
        pages = &tile_pages_[static_cast<size_t>(id) * slots];
    }
    const auto release = [&](u32 index) {
        pool_->Free(pages[index]);
        pages[index] = SparsePagePool::NoPage;
    };
    if (bind) {
        // Tiles mapped to the same guest bytes share a page, as they share memory on the console.
        u64 source = 0;
        const bool shared = share_ && !tail && TileSource(impl, s, id, source);
        bool all_reused = shared;
        const u64 epoch = impl.sparse_prepare_epoch;
        // Memory for every host block first: a failure leaves nothing bound.
        const auto need = [&](u32 index, u32 width, u32 height) {
            if (pages[index] == SparsePagePool::NoPage) {
                if (shared) {
                    bool page_reused = false;
                    pages[index] = pool_->AcquireShared(KeyOf(source, index, width, height),
                                                        memory_type_bits_, epoch, page_reused,
                                                        allow_alloc);
                    all_reused = all_reused && page_reused;
                } else {
                    pages[index] = allow_alloc ? pool_->Allocate(memory_type_bits_)
                                               : SparsePagePool::NoPage;
                }
            } else {
                all_reused = false;
            }
            return pages[index] != SparsePagePool::NoPage;
        };
        if (reused) {
            *reused = false;
        }
        bool ok = true;
        for (const HostTile& tile : tiles) {
            ok = ok && need(tile.slot, tile.width, tile.height);
        }
        for (u32 i = 0; ok && i < opaque_count; ++i) {
            ok = need(static_cast<u32>(tiles.size()) + i, 0, 0);
        }
        if (!ok) {
            for (u32 i = 0; i < slots; ++i) {
                release(i);
            }
            return false;
        }
        if (reused) {
            *reused = all_reused;
        }
    }
    const auto memory_of = [&](u32 index) {
        return bind ? pool_->Get(pages[index]) : SparsePagePool::Where{};
    };
    for (const HostTile& tile : tiles) {
        const SparsePagePool::Where where = memory_of(tile.slot);
        image_binds.push_back({
            .subresource{.aspectMask = aspect_, .mipLevel = tile.level, .arrayLayer = layer},
            .offset{static_cast<s32>(tile.x), static_cast<s32>(tile.y), 0},
            .extent{tile.width, tile.height, 1},
            .memory = where.memory,
            .memoryOffset = where.offset,
            .flags = 0,
        });
    }
    for (u32 i = 0; i < opaque_count; ++i) {
        const SparsePagePool::Where where = memory_of(static_cast<u32>(tiles.size()) + i);
        opaque_binds.push_back({
            .resourceOffset = tail_resource_offset_ + layer * tail_stride_ + i * SPARSE_TILE_BYTES,
            .size = SPARSE_TILE_BYTES,
            .memory = where.memory,
            .memoryOffset = where.offset,
            .flags = 0,
        });
    }
    if (!bind) {
        deferred_free_.insert(deferred_free_.end(), pages, pages + slots);
        std::fill(pages, pages + slots, SparsePagePool::NoPage);
    }
    bound_pages_ += bind ? tiles.size() + opaque_count : 0;
    bound_pages_ -= bind ? 0 : (std::min<u64>)(bound_pages_, tiles.size() + opaque_count);
    return true;
}

bool SparseBacking::Prepare(Context::Impl& impl, const Storage& s, Uploads& uploads) {
    std::vector<VkSparseImageMemoryBind> image_binds;
    std::vector<VkSparseMemoryBind> opaque_binds;
    std::vector<std::pair<bool, u32>> bound_now;
    struct ReboundTile {
        u32 id;
        std::vector<u32> old_pages;
    };
    std::vector<ReboundTile> rebound_now;
    u64 binds = 0;
    u64 unbinds = 0;
    // The first unit that finds no page ends the allocating of this call; the generation is read
    // first, so a page freed meanwhile makes the units worth trying again.
    bool exhausted = false;
    const u64 generation = pool_->FreeGeneration();
    ++impl.sparse_prepare_epoch;
    const auto process = [&](bool tail, std::vector<u32>& pending, std::vector<u8>& flags_vector) {
        std::vector<u32> keep;
        for (const u32 id : pending) {
            u8& flags = flags_vector[id];
            const bool mapped = (flags & Mapped) != 0;
            const bool want = mapped || (tail && persistent_tails_);
            const bool have = (flags & Bound) != 0;
            bool reused = false;
            if (want && have && share_ && !tail && (flags & Dirty) != 0 &&
                !TilePagesCurrent(impl, s, id)) {
                // The guest mapped other bytes here: move the tile to the page of its new source
                // (or a private one). The new binds replace the old ones; the old pages go once
                // the binds ran. Out of memory: unbind it, it waits like a starved tile.
                const u32 slots = host_per_x_ * host_per_y_;
                u32* pages = &tile_pages_[static_cast<size_t>(id) * slots];
                std::vector<u32> old(pages, pages + slots);
                std::fill(pages, pages + slots, SparsePagePool::NoPage);
                const u64 bound_before = bound_pages_;
                if (BindUnit(impl, s, false, id, true, image_binds, opaque_binds, &reused,
                             !exhausted && pool_->HasRoom())) {
                    rebound_now.push_back({id, std::move(old)});
                    bound_pages_ = bound_before; // as many blocks bound as before
                    ++binds;
                } else {
                    std::copy(old.begin(), old.end(), pages);
                    BindUnit(impl, s, false, id, false, image_binds, opaque_binds);
                    flags &= static_cast<u8>(~Bound);
                    ++unbinds;
                    exhausted = true;
                    keep.push_back(id);
                    continue;
                }
            } else if (want && !have) {
                const bool allow_alloc = !exhausted && pool_->HasRoom();
                if ((!allow_alloc && (!share_ || tail)) ||
                    !BindUnit(impl, s, tail, id, true, image_binds, opaque_binds, &reused,
                              allow_alloc)) {
                    exhausted = true;
                    if (!std::exchange(starved_logged_, true)) {
                        LOG_WARNING(Render_Vulkan,
                                    "[nvn] sparse texture {}x{}x{}: out of device memory for tiles "
                                    "({} MiB of {} MiB in use); they read as zero until memory is "
                                    "free",
                                    s.extent.width, s.extent.height, layout_.layers,
                                    pool_->UsedBytes() >> 20, pool_->BudgetBytes() >> 20);
                    }
                    keep.push_back(id); // tried again at the next use
                    continue;
                }
                flags |= Bound;
                bound_now.emplace_back(tail, id);
                ++binds;
            } else if (!want && have) {
                BindUnit(impl, s, tail, id, false, image_binds, opaque_binds);
                flags &= static_cast<u8>(~Bound);
                ++unbinds;
            }
            flags &= static_cast<u8>(~Queued);
            if (reused) {
                // Its shared page already holds these bytes' texels (uploaded or GPU-written
                // earlier in the stream): uploading the guest bytes again could only undo a
                // GPU write the console would show through every alias.
                flags &= static_cast<u8>(~Dirty);
            }
            if (mapped && (flags & Dirty) != 0) {
                (tail ? uploads.tails : uploads.tiles).push_back(id);
            }
        }
        pending = std::move(keep);
    };
    if (persistent_tails_) {
        // Reserve the small coarse levels before spending the budget on the detailed tiles.
        process(true, pending_tails_, tail_flags_);
        process(false, pending_tiles_, tile_flags_);
    } else {
        process(false, pending_tiles_, tile_flags_);
        process(true, pending_tails_, tail_flags_);
    }
    starved_ = static_cast<u32>(pending_tiles_.size() + pending_tails_.size());
    starved_only_ = starved_ != 0;
    starved_generation_ = generation;
    if (!image_binds.empty() || !opaque_binds.empty()) {
        // Flush any queued replay draws before binding sparse memory so `mkw_last_submit_*`
        // covers every prior draw that may sample this atlas.
        impl.FlushSubmitBatch();
        if (impl.mario_kart_world) {
            CompactSparseImageBinds(image_binds, granule_x_, granule_y_);
        }
        const VkSparseImageMemoryBindInfo image_info{
            .image = *s.image,
            .bindCount = static_cast<u32>(image_binds.size()),
            .pBinds = image_binds.data(),
        };
        const VkSparseImageOpaqueMemoryBindInfo opaque_info{
            .image = *s.image,
            .bindCount = static_cast<u32>(opaque_binds.size()),
            .pBinds = opaque_binds.data(),
        };
        // On MKW, the GPU can still be reading the atlas while the guest streams new tiles.
        // Wait for earlier work before changing residency, then wait for this bind below.
        const bool wait_for_atlas_use =
            impl.mario_kart_world && impl.mkw_last_submit_semaphore != VK_NULL_HANDLE;
        const VkTimelineSemaphoreSubmitInfo timeline{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &impl.mkw_last_submit_value,
        };
        const VkBindSparseInfo info{
            .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
            .pNext = wait_for_atlas_use ? &timeline : nullptr,
            .waitSemaphoreCount = wait_for_atlas_use ? 1u : 0u,
            .pWaitSemaphores = wait_for_atlas_use ? &impl.mkw_last_submit_semaphore : nullptr,
            .bufferBindCount = 0,
            .pBufferBinds = nullptr,
            .imageOpaqueBindCount = opaque_binds.empty() ? 0U : 1U,
            .pImageOpaqueBinds = opaque_binds.empty() ? nullptr : &opaque_info,
            .imageBindCount = image_binds.empty() ? 0U : 1U,
            .pImageBinds = image_binds.empty() ? nullptr : &image_info,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        vk::Fence fence = impl.device.GetLogical().CreateFence(VkFenceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        });
        VkResult result;
        {
            std::scoped_lock lock{impl.submit_mutex};
            result = impl.queue.BindSparse(info, *fence);
        }
        if (result == VK_SUCCESS) {
            const auto begin = std::chrono::steady_clock::now();
            result = fence.Wait();
            const auto wait = std::chrono::steady_clock::now() - begin;
            if (impl.mario_kart_world && wait >= std::chrono::milliseconds{20}) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] sparse residency fence: {:.1f} ms, {} image binds, {} opaque binds",
                         std::chrono::duration<double, std::milli>(wait).count(), image_binds.size(),
                         opaque_binds.size());
            }
        }
        if (result != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "[nvn] sparse texture {}x{}x{}: binding {} blocks failed ({})",
                      s.extent.width, s.extent.height, layout_.layers,
                      image_binds.size() + opaque_binds.size(), static_cast<int>(result));
            // Nothing can be said about what is bound: give the new tiles' memory back, the
            // units are tried again at the next use.
            for (const auto& [tail, id] : bound_now) {
                u8& flags = tail ? tail_flags_[id] : tile_flags_[id];
                flags &= static_cast<u8>(~Bound);
                if ((flags & Mapped) != 0) {
                    flags |= Dirty;
                }
                Queue(flags, id, tail);
                std::vector<VkSparseImageMemoryBind> unused_image;
                std::vector<VkSparseMemoryBind> unused_opaque;
                BindUnit(impl, s, tail, id, false, unused_image, unused_opaque);
            }
            const u32 slots = host_per_x_ * host_per_y_;
            for (const ReboundTile& rebound : rebound_now) {
                u32* const pages = &tile_pages_[static_cast<size_t>(rebound.id) * slots];
                for (u32 i = 0; i < slots; ++i) {
                    pool_->Free(pages[i]);
                }
                std::copy(rebound.old_pages.begin(), rebound.old_pages.end(), pages);
                tile_flags_[rebound.id] |= Dirty;
                Queue(tile_flags_[rebound.id], rebound.id, false);
            }
            uploads.tiles.clear();
            uploads.tails.clear();
            for (const u32 page : deferred_free_) {
                pool_->Free(page);
            }
            deferred_free_.clear();
            return false;
        }
        impl.stats.virtual_sparse_binds.fetch_add(image_binds.size() + opaque_binds.size(),
                                                  std::memory_order_relaxed);
    }
    for (const ReboundTile& rebound : rebound_now) {
        for (const u32 page : rebound.old_pages) {
            pool_->Free(page);
        }
    }
    for (const u32 page : deferred_free_) {
        pool_->Free(page);
    }
    deferred_free_.clear();
    pool_->TrimUnusedBlocks();
    for (const auto& [tail, id] : bound_now) {
        if (tail && persistent_tails_ && (tail_flags_[id] & Mapped) == 0) {
            uploads.zero_tails.push_back(id);
        }
    }
    for (const u32 id : uploads.tiles) {
        tile_flags_[id] &= static_cast<u8>(~Dirty);
        SetTilePagesStale(id, false);
    }
    for (const u32 id : uploads.tails) {
        tail_flags_[id] &= static_cast<u8>(~Dirty);
    }
    return true;
}

} // namespace Vulkan::Nvn
