// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Sparse-residency backing of the NVN textures that live in virtual memory pools.
//
// A large sparse texture (a terrain virtual texture: 8192x8192 x 68 layers, gigabytes) holds only
// the 64 KiB tiles the game maps into its virtual pool; a tile nobody mapped reads as zero.
// Its VkImage is a sparse residency image created without memory; device memory is bound per
// 64 KiB host page where the pool's runs map a tile, and only mapped tiles are decoded and
// uploaded. A remap touches the tiles of its range, not the whole storage (nxnative: Metal
// placement sparse textures, docs/reference/nvn-objects.md "Large sparse textures as Metal
// placement sparse textures").
//
// Everything here runs on the replay thread, except the page pool, which the thread that drops
// the last reference of a storage also uses.

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse_layout.h"

namespace Vulkan::Nvn {

/// Combine adjacent blocks of a single host row without changing their physical mappings.
void CompactSparseImageBinds(std::vector<VkSparseImageMemoryBind>& binds, u32 granule_x,
                             u32 granule_y);

/// What a shared sparse page holds: the guest bytes at `source` (a guest address, 64 KiB aligned
/// for a tile), decoded for images of shape `shape` (formats, tile and host block shape, memory
/// types, host block slot). Tiles with equal keys hold equal texels, so they can share a page.
struct SparseAliasKey {
    u64 source{};
    u64 shape{};
    bool operator==(const SparseAliasKey&) const = default;
};
struct SparseAliasKeyHash {
    size_t operator()(const SparseAliasKey& key) const noexcept {
        return static_cast<size_t>(key.source * 0x9E3779B97F4A7C15ULL ^ key.shape);
    }
};

/// 64 KiB pages of device memory for sparse images, taken from the allocator in blocks.
class SparsePagePool {
public:
    static constexpr u32 NoPage = ~0u;
    static constexpr u64 PageBytes = SPARSE_TILE_BYTES;
    static constexpr u32 PagesPerBlock = 64;
    static constexpr u32 CachedBlockLimit = 4;

    SparsePagePool(const MemoryAllocator& allocator, u64 budget_bytes);
    ~SparsePagePool();
    SparsePagePool(const SparsePagePool&) = delete;
    SparsePagePool& operator=(const SparsePagePool&) = delete;

    /// A page usable by an image whose memory requirements allow `memory_type_bits`; NoPage when
    /// the budget or the device memory is used up.
    [[nodiscard]] u32 Allocate(u32 memory_type_bits);
    /// The page holding `key`'s texels, shared with every other tile of that key (one more
    /// reference). `reused`: the page existed before replay step `epoch`, so its texels were
    /// uploaded or written earlier in the command stream. NoPage when out of memory.
    [[nodiscard]] u32 AcquireShared(const SparseAliasKey& key, u32 memory_type_bits, u64 epoch,
                                    bool& reused, bool allow_alloc = true);
    /// Whether `page` is currently held in the shared key map (regardless of key).
    [[nodiscard]] bool IsShared(u32 page) const;
    /// Whether `page` is the shared page of `key`.
    [[nodiscard]] bool HoldsKey(u32 page, const SparseAliasKey& key) const;
    void MarkSharedStale(u32 page);
    void ClearSharedStale(u32 page);
    /// Drops one reference; the page is free when the last one goes.
    void Free(u32 page);
    /// Bytes saved by sharing: references beyond the first of each shared page.
    [[nodiscard]] u64 SharedSavedBytes() const {
        std::scoped_lock lock{mutex_};
        return shared_extra_refs_ * PageBytes;
    }
    /// Release wholly unused allocations after their pages are no longer in GPU use.
    void TrimUnusedBlocks();
    /// Counts the pages given back: a starved tile is worth another try only after it changed.
    [[nodiscard]] u64 FreeGeneration() const {
        std::scoped_lock lock{mutex_};
        return free_generation_;
    }
    struct Where {
        VkDeviceMemory memory{};
        VkDeviceSize offset{};
    };
    [[nodiscard]] Where Get(u32 page) const;
    /// Bytes in use (bound pages).
    [[nodiscard]] u64 UsedBytes() const;
    /// Allocation bytes, including cached free pages (before VMA's own block overhead).
    [[nodiscard]] u64 ReservedBytes() const;
    [[nodiscard]] u64 BudgetBytes() const noexcept {
        return budget_;
    }
    [[nodiscard]] bool HasRoom() const;

private:
    struct Block {
        VmaAllocation allocation{};
        Where where;
        u32 type_bits{}; ///< the memory types the block was allocated for
        u32 used_pages{};
    };

    u32 AllocateLocked(u32 memory_type_bits);

    const MemoryAllocator& allocator_;
    const u64 budget_;
    mutable std::mutex mutex_;
    std::vector<Block> blocks_;
    std::vector<u32> free_;
    std::vector<u32> refs_;  ///< per page: references (0 free)
    std::vector<u64> epoch_; ///< per shared page: the replay step that allocated it
    std::vector<u8> stale_;  ///< per shared page: owning tile has un-uploaded guest changes
    std::unordered_map<SparseAliasKey, u32, SparseAliasKeyHash> by_key_;
    std::unordered_map<u32, SparseAliasKey> key_of_;
    u64 shared_extra_refs_{0};
    u64 used_{0};
    u64 free_generation_{0};
    u32 empty_blocks_{0};
};

/// The tile state of one sparse storage.
class SparseBacking {
public:
    /// Tile flags.
    enum : u8 {
        Mapped = 1, ///< the pool maps the tile: it must hold the guest's bytes
        Dirty = 2,  ///< the bytes behind the tile changed since the last upload
        Bound = 4,  ///< device memory is bound for the tile's host pages
        Queued = 8, ///< on the pending list
    };

    /// The units a `Prepare` found ready to upload (mapped, bound, dirty).
    struct Uploads {
        std::vector<u32> tiles;
        std::vector<u32> tails;
        std::vector<u32> zero_tails; ///< newly resident, unmapped coarse levels
    };

    /// nullptr when `s` cannot be tiled (layout, offset not tile aligned, device without sparse
    /// residency images). Does not touch the image.
    [[nodiscard]] static std::shared_ptr<SparseBacking> Make(const Storage& s, const Device& device,
                                                             u64 pool_size, const char*& reason);
    /// The state of a storage with `layout` (no image, no memory): all tiles unmapped.
    [[nodiscard]] static std::shared_ptr<SparseBacking> FromLayout(SparseLayout layout);
    ~SparseBacking();

    /// Reads the sparse shape of `s.image` (created unbacked). False when it does not fit the
    /// tiles (the caller falls back to an ordinary image); `reason` says why.
    [[nodiscard]] bool Attach(Context::Impl& impl, const Storage& s, const char*& reason);

    [[nodiscard]] const SparseLayout& Layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] const UploadPlan& TailPlan() const noexcept {
        return tail_plan_;
    }

    /// The pool's mapping of [begin, end) of the storage changed. `mapped(offset, size)` says
    /// whether any run of the pool covers storage bytes [offset, offset + size).
    template <typename MappedFn>
    void Remap(u64 begin, u64 end, MappedFn&& mapped) {
        layout_.ForEachUnit(
            begin, end,
            [&](u32 id) {
                SetMapped(tile_flags_[id], id, false,
                          mapped(layout_.TileOffset(id), SPARSE_TILE_BYTES));
            },
            [&](u32 layer) {
                SetMapped(tail_flags_[layer], layer, true,
                          mapped(static_cast<u64>(layer) * layout_.layer_size + layout_.tail_offset,
                                 layout_.TailSize()));
            });
    }
    /// The bytes behind [begin, end) of the storage changed (the CPU wrote them): the mapped tiles
    /// in it are uploaded again.
    void ContentChanged(u64 begin, u64 end);
    /// Every mapped tile is uploaded again (the game wrote the texture itself).
    void MarkAllDirty();

    /// Units wait for a bind, an unbind or an upload. Starved units (no page was left) do not
    /// count until a page was freed.
    [[nodiscard]] bool HasWork() const {
        if (pending_tiles_.empty() && pending_tails_.empty()) {
            return false;
        }
        return !starved_only_ || !pool_ || pool_->FreeGeneration() != starved_generation_;
    }

    /// Binds memory for the pending mapped tiles, unbinds the pending unmapped ones, and lists
    /// what has to be uploaded. The binds are done when it returns (the queue has run them).
    /// The dirty marks of the listed units are cleared: the caller uploads them or drops them.
    [[nodiscard]] bool Prepare(Context::Impl& impl, const Storage& s, Uploads& uploads);

    void SetTileHash(u32 id, u64 hash) {
        tile_hash_[id] = hash;
    }
    [[nodiscard]] u32 TileVerifiedGen(u32 id) const {
        return id < tile_verified_gen_.size() ? tile_verified_gen_[id] : 0;
    }
    [[nodiscard]] u64 TileVerifiedPool(u32 id) const {
        return id < tile_verified_pool_.size() ? tile_verified_pool_[id] : 0;
    }
    void SetTileVerifiedGen(u32 id, u64 pool_serial, u32 gen) {
        if (id < tile_verified_gen_.size()) {
            tile_verified_pool_[id] = pool_serial;
            tile_verified_gen_[id] = gen;
        }
    }
    /// The next mapped, bound, clean tile to compare with its bytes (round robin), or ~0u.
    [[nodiscard]] u32 NextVerifyTile();
    [[nodiscard]] u64 TileHash(u32 id) const {
        return tile_hash_[id];
    }
    /// A verified tile whose bytes changed.
    void MarkTileDirty(u32 id);

    [[nodiscard]] u64 BoundBytes() const noexcept {
        return bound_pages_ * SparsePagePool::PageBytes;
    }
    [[nodiscard]] u32 MappedTiles() const noexcept {
        return mapped_tiles_;
    }
    [[nodiscard]] u32 StarvedUnits() const noexcept {
        return starved_;
    }

private:
    SparseBacking() = default;

    struct HostTile {
        u32 level;
        u32 x; ///< texels
        u32 y;
        u32 width; ///< extent, cut to the level
        u32 height;
        u32 slot; ///< index in the unit's page list
    };

    void SetMapped(u8& flags, u32 id, bool tail, bool mapped);
    void Queue(u8& flags, u32 id, bool tail);
    /// Host tiles of guest tile `id` / of the tail levels held in host tiles.
    void HostTilesOfTile(u32 id, std::vector<HostTile>& out) const;
    std::vector<u32>& TilePages(u32 id);
    bool BindUnit(Context::Impl& impl, const Storage& s, bool tail, u32 id, bool bind,
                  std::vector<VkSparseImageMemoryBind>& image_binds,
                  std::vector<VkSparseMemoryBind>& opaque_binds, bool* reused = nullptr,
                  bool allow_alloc = true);
    /// The guest address of tile `id`'s 64 KiB, when one physical run maps all of it.
    bool TileSource(Context::Impl& impl, const Storage& s, u32 id, u64& source) const;
    /// The alias key of host block `slot` of valid extent (`width`, `height`) whose bytes are at
    /// `source`.
    [[nodiscard]] SparseAliasKey KeyOf(u64 source, u32 slot, u32 width, u32 height) const {
        u64 key_shape = shape_ ^ (static_cast<u64>(slot + 1) * 0xC2B2AE3D27D4EB4FULL);
        key_shape ^= ((static_cast<u64>(width) << 32) | height) * 0x9E3779B97F4A7C15ULL;
        return {source, key_shape};
    }
    void SetTilePagesStale(u32 id, bool stale);
    /// Whether the bound pages of tile `id` still hold the bytes its guest mapping points at.
    bool TilePagesCurrent(Context::Impl& impl, const Storage& s, u32 id);

    SparseLayout layout_;
    UploadPlan tail_plan_;
    std::shared_ptr<SparsePagePool> pool_;
    // Image shape (Attach).
    u32 granule_x_{};
    u32 granule_y_{};
    u32 host_per_x_{}; ///< host tiles across a guest tile
    u32 host_per_y_{};
    u32 tail_first_lod_{};
    u32 memory_type_bits_{};
    VkDeviceSize tail_size_{};
    VkDeviceSize tail_resource_offset_{};
    VkDeviceSize tail_stride_{};
    VkImageAspectFlags aspect_{VK_IMAGE_ASPECT_COLOR_BIT};
    std::vector<HostTile> tail_host_tiles_;
    std::vector<HostTile> tile_scratch_; ///< reused by BindUnit on the replay thread
    // State.
    std::vector<u8> tile_flags_;
    std::vector<u8> tail_flags_;
    std::vector<u64> tile_hash_;
    std::vector<u64> tile_verified_pool_;
    std::vector<u32> tile_verified_gen_;
    std::vector<u32> tile_pages_;              ///< host_per_x_ * host_per_y_ per tile
    std::vector<std::vector<u32>> tail_pages_; ///< per layer: host tail tiles, then opaque pages
    std::vector<u32> pending_tiles_;
    std::vector<u32> pending_tails_;
    u64 bound_pages_{0};
    u32 mapped_tiles_{0};
    u32 starved_{0};
    u32 verify_cursor_{0};
    std::vector<u32> deferred_free_; ///< pages of unbound units, freed once the binds ran
    bool starved_logged_{false};
    bool share_{false}; ///< tiles share pages by guest source (the image is SPARSE_ALIASED)
    u64 shape_{0};      ///< SparseAliasKey::shape before the slot
    bool persistent_tails_{false}; ///< MKW writes coarse levels without mapping guest tail pages
    /// Everything pending found no page at generation `starved_generation_`.
    bool starved_only_{false};
    u64 starved_generation_{0};
};

} // namespace Vulkan::Nvn
