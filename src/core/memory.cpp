// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2015 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <set>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/atomic_ops.h"
#include "common/common_types.h"
#include "common/heap_tracker.h"
#include "common/logging.h"
#include "common/page_table.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#include "core/gpu_dirty_memory_manager.h"
#include "video_core/nx2_hazard.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/nx2_thread_stats.h"
#include "core/memory.h"
#include "core/memory/page_tracker.h"
#include "core/memory/page_tracker_state.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/host1x.h"
#include "video_core/nx2.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/page_filter.h"
#include "video_core/rasterizer_download_area.h"

namespace Core::Memory {

u32& UnmappedReadFlag() {
    thread_local u32 flag{};
    return flag;
}

static inline bool AddressSpaceContains(const Common::PageTable& table, const Common::ProcessAddress addr, const std::size_t size) {
    const Common::ProcessAddress max_addr = 1ULL << table.GetAddressSpaceBits();
    return addr + size >= addr && addr + size <= max_addr;
}

// Implementation class used to keep the specifics of the memory subsystem hidden
// from outside classes. This also allows modification to the internals of the memory
// subsystem without needing to rebuild all files that make use of the memory interface.
struct Memory::Impl final : PageClaimSink {
    explicit Impl(Core::System& system_) : system{system_} {}

    void SetCurrentPageTable(Kernel::KProcess& process) {
        current_page_table = &process.GetPageTable().GetImpl();

        if (process.IsApplication() && Settings::IsFastmemEnabled()) {
            current_page_table->fastmem_arena = system.DeviceMemory().buffer.VirtualBasePointer();
        } else {
            current_page_table->fastmem_arena = nullptr;
        }

#ifdef __ANDROID__
        heap_tracker.emplace(system.DeviceMemory().buffer);
        host_buffer = std::addressof(*heap_tracker);
#else
        host_buffer = std::addressof(system.DeviceMemory().buffer);
#endif
    }

    void MapMemoryRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                         Common::PhysicalAddress target, Common::MemoryPermission perms,
                         bool separate_heap) {
        ASSERT_MSG((size & VESHIRU_PAGEMASK) == 0, "non-page aligned size: {:016X}", size);
        ASSERT_MSG((base & VESHIRU_PAGEMASK) == 0, "non-page aligned base: {:016X}", GetInteger(base));
        ASSERT_MSG(target >= DramMemoryMap::Base, "Out of bounds target: {:016X}",
                   GetInteger(target));
        MapPages(page_table, base / VESHIRU_PAGESIZE, size / VESHIRU_PAGESIZE, target,
                 Common::PageType::Memory);

        if (current_page_table->fastmem_arena) {
            host_buffer->Map(GetInteger(base), GetInteger(target) - DramMemoryMap::Base, size, perms, separate_heap);
        }
        // Tracked contents changed with the mapping; their claims go back on the new pages, after
        // the host mapping that reset their protection. Only remaps of the tracked (current)
        // table move its claims.
        if (&page_table == current_page_table && !page_tracker_state.Empty()) [[unlikely]] {
            std::scoped_lock lk{read_trap_mutex};
            page_tracker_state.OnRemap(GetInteger(base) >> VESHIRU_PAGEBITS,
                                       (GetInteger(base) + size) >> VESHIRU_PAGEBITS, true);
        }
    }

    void UnmapRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                     bool separate_heap) {
        ASSERT_MSG((size & VESHIRU_PAGEMASK) == 0, "non-page aligned size: {:016X}", size);
        ASSERT_MSG((base & VESHIRU_PAGEMASK) == 0, "non-page aligned base: {:016X}", GetInteger(base));
        if (&page_table == current_page_table && !page_tracker_state.Empty()) [[unlikely]] {
            std::scoped_lock lk{read_trap_mutex};
            page_tracker_state.OnRemap(GetInteger(base) >> VESHIRU_PAGEBITS,
                                       (GetInteger(base) + size) >> VESHIRU_PAGEBITS, false);
        }
        MapPages(page_table, base / VESHIRU_PAGESIZE, size / VESHIRU_PAGESIZE, 0,
                 Common::PageType::Unmapped);

        if (current_page_table->fastmem_arena) {
            host_buffer->Unmap(GetInteger(base), size, separate_heap);
        }
    }

    void ProtectRegion(Common::PageTable& page_table, VAddr vaddr, u64 size,
                       Common::MemoryPermission perms) {
        ASSERT_MSG((size & VESHIRU_PAGEMASK) == 0, "non-page aligned size: {:016X}", size);
        ASSERT_MSG((vaddr & VESHIRU_PAGEMASK) == 0, "non-page aligned base: {:016X}", vaddr);

        if (!current_page_table->fastmem_arena) {
            return;
        }

        u64 protect_bytes = 0, protect_begin = 0;

        current_page_table->entries.CommitRegion(vaddr >> VESHIRU_PAGEBITS, (vaddr + size) >> VESHIRU_PAGEBITS);
        for (u64 addr = vaddr; addr < vaddr + size; addr += VESHIRU_PAGESIZE) {
            const Common::PageType page_type = current_page_table->entries.GetUnchecked(addr >> VESHIRU_PAGEBITS).Type();
            switch (page_type) {
            case Common::PageType::RasterizerCachedMemory:
                if (protect_bytes > 0) {
                    host_buffer->Protect(protect_begin, protect_bytes, perms);
                    protect_bytes = 0;
                }
                break;
            default:
                if (protect_bytes == 0)
                    protect_begin = addr;
                protect_bytes += VESHIRU_PAGESIZE;
            }
        }

        if (protect_bytes > 0) {
            host_buffer->Protect(protect_begin, protect_bytes, perms);
        }
    }

    [[nodiscard]] u8* GetPointerFromRasterizerCachedMemory(u64 vaddr) const {
        if (u64 paddr = current_page_table->entries[vaddr >> VESHIRU_PAGEBITS].Pointer(true); paddr)
            return reinterpret_cast<u8*>(paddr) + vaddr;
        return {};
    }

    [[nodiscard]] u8* GetPointerFromDebugMemory(u64 vaddr) const {
        if (u64 paddr = current_page_table->entries[vaddr >> VESHIRU_PAGEBITS].Pointer(true); paddr)
            return reinterpret_cast<u8*>(paddr) + vaddr;
        return {};
    }

    u8 Read8(const Common::ProcessAddress addr) {
        return Read<u8>(addr);
    }

    u16 Read16(const Common::ProcessAddress addr) {
        if ((addr & 1) == 0) {
            return Read<u16_le>(addr);
        } else {
            const u32 a{Read<u8>(addr)};
            const u32 b{Read<u8>(addr + sizeof(u8))};
            return static_cast<u16>((b << 8) | a);
        }
    }

    u32 Read32(const Common::ProcessAddress addr) {
        if ((addr & 3) == 0) {
            return Read<u32_le>(addr);
        } else {
            const u32 a{Read16(addr)};
            const u32 b{Read16(addr + sizeof(u16))};
            return (b << 16) | a;
        }
    }

    bool Read32Acquire(const Common::ProcessAddress addr, u32& value) {
        value = 0;
        if (!current_page_table || (addr & 3) != 0) {
            return false;
        }
        auto* pointer = GetPointerImpl(GetInteger(addr), [] {}, [&] {
            HandleRasterizerDownload(GetInteger(addr), sizeof(u32));
        });
        if (!pointer) {
            return false;
        }
        u32_le actual{};
        static_cast<void>(Common::AtomicCompareAndSwap(reinterpret_cast<u32_le*>(pointer),
                                                     u32_le{0}, u32_le{0}, actual));
        value = actual;
        return true;
    }

    u64 Read64(const Common::ProcessAddress addr) {
        if ((addr & 7) == 0) {
            return Read<u64_le>(addr);
        } else {
            const u32 a{Read32(addr)};
            const u32 b{Read32(addr + sizeof(u32))};
            return (static_cast<u64>(b) << 32) | a;
        }
    }

    void Write8(const Common::ProcessAddress addr, const u8 data) {
        Write<u8>(addr, data);
    }

    void Write16(const Common::ProcessAddress addr, const u16 data) {
        if ((addr & 1) == 0) {
            Write<u16_le>(addr, data);
        } else {
            Write<u8>(addr, static_cast<u8>(data));
            Write<u8>(addr + sizeof(u8), static_cast<u8>(data >> 8));
        }
    }

    void Write32(const Common::ProcessAddress addr, const u32 data) {
        if ((addr & 3) == 0) {
            Write<u32_le>(addr, data);
        } else {
            Write16(addr, static_cast<u16>(data));
            Write16(addr + sizeof(u16), static_cast<u16>(data >> 16));
        }
    }

    void Write64(const Common::ProcessAddress addr, const u64 data) {
        if ((addr & 7) == 0) {
            Write<u64_le>(addr, data);
        } else {
            Write32(addr, static_cast<u32>(data));
            Write32(addr + sizeof(u32), static_cast<u32>(data >> 32));
        }
    }

    bool WriteExclusive8(const Common::ProcessAddress addr, const u8 data, const u8 expected) {
        return WriteExclusive<u8>(addr, data, expected);
    }

    bool WriteExclusive16(const Common::ProcessAddress addr, const u16 data, const u16 expected) {
        return WriteExclusive<u16_le>(addr, data, expected);
    }

    bool WriteExclusive32(const Common::ProcessAddress addr, const u32 data, const u32 expected) {
        return WriteExclusive<u32_le>(addr, data, expected);
    }

    bool WriteExclusive64(const Common::ProcessAddress addr, const u64 data, const u64 expected) {
        return WriteExclusive<u64_le>(addr, data, expected);
    }

    std::string ReadCString(Common::ProcessAddress vaddr, std::size_t max_length) {
        std::string string;
        string.reserve(max_length);
        for (std::size_t i = 0; i < max_length; ++i) {
            const char c = Read<s8>(vaddr);
            if (c == '\0') {
                break;
            }
            string.push_back(c);
            ++vaddr;
        }
        string.shrink_to_fit();
        return string;
    }

    template<typename F1, typename F2, typename F3>
    inline bool WalkBlock(const Common::ProcessAddress addr, const std::size_t size, F1&& on_unmapped, F2&& on_memory, F3&& on_rasterizer) {
        std::size_t offset = 0;
        if (!AddressSpaceContains(*current_page_table, addr, size)) {
            on_unmapped(offset, size, addr);
            return false;
        }
        std::size_t remaining_size = size;
        std::size_t page_index = addr >> VESHIRU_PAGEBITS;
        std::size_t page_offset = addr & VESHIRU_PAGEMASK;
        bool user_accessible = true;

        current_page_table->entries.CommitRegion(page_index, page_index + (size >> VESHIRU_PAGEBITS) + 1);
        while (remaining_size != 0) {
            const std::size_t copy_amount = (std::min)(std::size_t(VESHIRU_PAGESIZE) - page_offset, remaining_size);
            const auto current_vaddr = u64((page_index << VESHIRU_PAGEBITS) + page_offset);
            const auto [pointer, type, _] = current_page_table->entries.GetUnchecked(page_index).PointerTypeBlock();
            switch (type) {
            case Common::PageType::Unmapped: {
                user_accessible = false;
                on_unmapped(offset, copy_amount, current_vaddr);
                break;
            }
            case Common::PageType::Memory: {
                u8* mem_ptr = reinterpret_cast<u8*>(pointer + page_offset + (page_index << VESHIRU_PAGEBITS));
                on_memory(offset, copy_amount, mem_ptr);
                break;
            }
            case Common::PageType::DebugMemory: {
                u8* const mem_ptr = GetPointerFromDebugMemory(current_vaddr);
                on_memory(offset, copy_amount, mem_ptr);
                break;
            }
            case Common::PageType::RasterizerCachedMemory: {
                u8* const host_ptr = GetPointerFromRasterizerCachedMemory(current_vaddr);
                on_rasterizer(current_vaddr, offset, copy_amount, host_ptr);
                break;
            }
            default:
                UNREACHABLE();
            }
            page_index++;
            page_offset = 0;
            offset += copy_amount;
            remaining_size -= copy_amount;
        }
        return user_accessible;
    }

    bool ReadBlockImpl(const Common::ProcessAddress addr, void* buffer, const std::size_t size, bool unsafe) {
        return WalkBlock(addr, size,
        [addr, size, &buffer](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(addr), size);
            std::memset(reinterpret_cast<u8*>(buffer) + offset, 0, copy_amount);
        },
        [&](const std::size_t offset, const std::size_t copy_amount, const u8* const ptr) {
            std::memcpy(reinterpret_cast<u8*>(buffer) + offset, ptr, copy_amount);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, const u8* const ptr) {
            if (!unsafe) {
                HandleRasterizerDownload(GetInteger(current_vaddr), copy_amount);
            }
            std::memcpy(reinterpret_cast<u8*>(buffer) + offset, ptr, copy_amount);
        });
    }

    [[nodiscard]] u8* GetSpan(const VAddr addr, const std::size_t size) const noexcept {
        if (!AddressSpaceContains(*current_page_table, addr, size) ||
            addr >= (u64{1} << current_page_table->GetAddressSpaceBits())) {
            return nullptr;
        }
        const u64 first_page = addr >> VESHIRU_PAGEBITS;
        const u64 last_page = (addr + (size != 0 ? size - 1 : 0)) >> VESHIRU_PAGEBITS;
        const auto offset = current_page_table->entries[first_page].Pointer(true);
        for (u64 page = first_page + 1; page <= last_page; ++page) {
            if (current_page_table->entries[page].Pointer(true) != offset) {
                return nullptr;
            }
        }
        return GetPointerSilent(addr);
    }

    /// Who writes: the guest or one of its services (Safe: device caches and the page tracker see
    /// it), a host writer that only skips the device caches (Unsafe: the page tracker still sees
    /// it, because the guest's memory changed) or the HLE's own result write-back (Untracked:
    /// the mirror that holds the bytes already has them).
    enum class WriteMode : u8 { Safe, Unsafe, Untracked };

    bool WriteBlockImpl(const Common::ProcessAddress addr, const void* buffer, const std::size_t size, WriteMode mode) {
        return WalkBlock(addr, size,
        [addr, size](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(addr), size);
        },
        [&](const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            std::memcpy(ptr, reinterpret_cast<u8 const*>(buffer) + offset, copy_amount);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            bool rasterizer_write = false;
            if (mode == WriteMode::Safe) {
                rasterizer_write = RasterizerPreWrite(GetInteger(current_vaddr), copy_amount);
            } else if (mode == WriteMode::Unsafe) {
                TrackerPreWrite(GetInteger(current_vaddr), copy_amount);
                // The caches aren't told, but constant buffer data equal to memory must be.
                VideoCommon::Nx2FastDraw::BeginMemoryChange();
                rasterizer_write = true;
            }
            std::memcpy(ptr, reinterpret_cast<u8 const*>(buffer) + offset, copy_amount);
            if (rasterizer_write) {
                EndRasterizerWrite();
            }
            if (mode != WriteMode::Untracked) {
                NotifyHostWrite(GetInteger(current_vaddr), copy_amount);
            }
        });
    }

    /// Memory changed through a raw pointer or a landed store: tracked pages turn dirty and lose
    /// their write trap. The device caches are not told.
    void NotifyHostWrite(u64 vaddr, size_t size) {
        if (size == 0 || page_tracker_state.Empty()) [[likely]] {
            return;
        }
        TrackerWrite(vaddr, size);
    }

    bool ZeroBlock(const Common::ProcessAddress addr, const std::size_t size) {
        return WalkBlock(addr, size,
        [addr, size](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(addr), size);
        },
        [=](const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            std::memset(ptr, 0, copy_amount);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            const bool rasterizer_write = RasterizerPreWrite(GetInteger(current_vaddr), copy_amount);
            std::memset(ptr, 0, copy_amount);
            if (rasterizer_write) {
                EndRasterizerWrite();
            }
            NotifyHostWrite(GetInteger(current_vaddr), copy_amount);
        });
    }

    bool CopyBlock(Common::ProcessAddress dest_addr, Common::ProcessAddress src_addr, const std::size_t size) {
        return WalkBlock(dest_addr, size,
        [&](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(src_addr), size);
            ZeroBlock(dest_addr + offset, copy_amount);
        },
        [&](const std::size_t offset, const std::size_t copy_amount, const u8* const ptr) {
            WriteBlockImpl(dest_addr + offset, ptr, copy_amount, WriteMode::Safe);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            HandleRasterizerDownload(GetInteger(current_vaddr), copy_amount);
            WriteBlockImpl(dest_addr + offset, ptr, copy_amount, WriteMode::Safe);
        });
    }

    template <typename Callback>
    Result PerformCacheOperation(Common::ProcessAddress dest_addr, std::size_t size, Callback&& cb) {
        bool had_unmapped = false;
        WalkBlock(dest_addr, size,
            [&](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
                LOG_ERROR(HW_Memory, "Unmapped @ {:#018X}", GetInteger(current_vaddr));
                had_unmapped = true;
            },
            [](const std::size_t offset, const std::size_t copy_amount, u8* const host_ptr) {},
            [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const host_ptr) { cb(current_vaddr, offset, copy_amount); }
        );
        return had_unmapped ? Kernel::ResultInvalidCurrentMemory : ResultSuccess;
    }

    Result InvalidateDataCache(Common::ProcessAddress dest_addr, std::size_t size) {
        auto on_rasterizer = [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t block_size) {
            // dc ivac: Invalidate to point of coherency
            // GPU flush -> CPU invalidate
            HandleRasterizerDownload(GetInteger(current_vaddr), block_size);
        };
        return PerformCacheOperation(dest_addr, size, on_rasterizer);
    }

    Result StoreDataCache(Common::ProcessAddress dest_addr, std::size_t size) {
        auto on_rasterizer = [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t block_size) {
            // dc cvac: Store to point of coherency
            // CPU flush -> GPU invalidate
            HandleCacheStore(GetInteger(current_vaddr), block_size);
        };
        return PerformCacheOperation(dest_addr, size, on_rasterizer);
    }

    Result FlushDataCache(Common::ProcessAddress dest_addr, std::size_t size) {
        auto on_rasterizer = [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t block_size) {
            // dc civac: Store to point of coherency, and invalidate from cache
            // CPU flush -> GPU invalidate
            HandleCacheStore(GetInteger(current_vaddr), block_size);
        };
        return PerformCacheOperation(dest_addr, size, on_rasterizer);
    }

    void MarkRegionDebug(u64 vaddr, u64 size, bool debug) {
        if (vaddr == 0 || !AddressSpaceContains(*current_page_table, vaddr, size)) {
            return;
        }

        if (current_page_table->fastmem_arena) {
            const auto perm{debug ? Common::MemoryPermission{}
                                  : Common::MemoryPermission::ReadWrite};
            host_buffer->Protect(vaddr, size, perm);
        }

        // Iterate over a contiguous CPU address space, marking/unmarking the region.
        // The region is at a granularity of CPU pages.

        const u64 num_pages = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) - (vaddr >> VESHIRU_PAGEBITS) + 1;

        current_page_table->entries.CommitRegion(vaddr >> VESHIRU_PAGEBITS, (vaddr >> VESHIRU_PAGEBITS) + num_pages);
        for (u64 i = 0; i < num_pages; ++i, vaddr += VESHIRU_PAGESIZE) {
            auto& entry = current_page_table->entries.GetUnchecked(vaddr >> VESHIRU_PAGEBITS);
            const auto [pointer, type, block] = entry.PointerTypeBlock(true);
            if (debug) {
                // Switch page type to debug if now debug
                switch (type) {
                case Common::PageType::Unmapped:
                    ASSERT(false && "Attempted to mark unmapped pages as debug");
                    break;
                case Common::PageType::RasterizerCachedMemory:
                case Common::PageType::DebugMemory:
                    // Page is already marked.
                    break;
                case Common::PageType::Memory:
                    entry.MarkDebug(pointer, block);
                    break;
                default:
                    UNREACHABLE();
                }
            } else {
                // Switch page type to non-debug if now non-debug
                switch (type) {
                case Common::PageType::Unmapped:
                    ASSERT(false && "Attempted to mark unmapped pages as non-debug");
                    break;
                case Common::PageType::RasterizerCachedMemory:
                case Common::PageType::Memory:
                    // Don't mess with already non-debug or rasterizer memory.
                    break;
                case Common::PageType::DebugMemory: {
                    entry.Store(false, Common::PageType::Memory, block, pointer);
                    break;
                }
                default:
                    UNREACHABLE();
                }
            }
        }
    }

    void RasterizerMarkRegionCached(u64 vaddr, u64 size, bool cached) {
        if (vaddr == 0 || !AddressSpaceContains(*current_page_table, vaddr, size)) {
            return;
        }
        if (!page_tracker_enabled.load(std::memory_order_relaxed)) [[likely]] {
            MarkRegionClaim(vaddr, size, PageClaim{.device = cached}, false);
            return;
        }
        // With the page tracker on, page types change under the trap lock, so the tracker never
        // acts on a page whose cached state changed after it looked at it.
        const u64 page_begin = vaddr >> VESHIRU_PAGEBITS;
        const u64 page_end = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1;
        std::scoped_lock lk{read_trap_mutex};
        if (page_tracker_state.Empty()) {
            MarkRegionClaim(vaddr, size, PageClaim{.device = cached}, true);
            return;
        }
        // Pages the tracker holds stay cached for it when the device releases them.
        page_tracker_state.SetDeviceCached(page_begin, page_end, cached);
    }

    /// Whether the device memory manager caches the page: only it sets the page type on pages
    /// the tracker does not cover. Called with read_trap_mutex held.
    bool DeviceCached(u64 page) const override {
        return page < current_page_table->entries.size() &&
               std::as_const(current_page_table->entries)[page].Type() ==
                   Common::PageType::RasterizerCachedMemory;
    }

    /// The page tracker's decision for [page_begin, page_end). Called with read_trap_mutex held.
    void ApplyClaim(u64 page_begin, u64 page_end, PageClaim claim) override {
        MarkRegionClaim(page_begin << VESHIRU_PAGEBITS, (page_end - page_begin) << VESHIRU_PAGEBITS,
                        claim, true);
    }

    /// Gives the pages of [vaddr, vaddr + size) the page type and host protection of `claim`:
    /// guest accesses to a cached page take the slow path (writes always, reads under reactive
    /// flushing or an access trap).
    void MarkRegionClaim(u64 vaddr, u64 size, PageClaim claim, bool trap_locked) {
        const bool cached = claim.Cached();
        if (current_page_table->fastmem_arena) {
            Common::MemoryPermission perm{};
            // EDEN_NX2_NO_REACTIVE=1 drops the read traps on Switch 2 programs. Opt-in: without
            // them a CPU read of GPU-written memory can see stale data.
            static const bool nx2_no_reactive = [] {
                const char* value = std::getenv("EDEN_NX2_NO_REACTIVE");
                return value != nullptr && value[0] == '1';
            }();
            const bool reactive_flushing =
                Settings::ReactiveFlushing() &&
                !(nx2_no_reactive && Settings::CurrentProgramIsSwitch2());
            if (!claim.access && !(claim.device && reactive_flushing)) {
                perm |= Common::MemoryPermission::Read;
            }
            if (!cached) {
                perm |= Common::MemoryPermission::Write;
            }
            // EDEN_NX2_PROTECT_STATS=1: how many protection changes the tracker makes, and what
            // they cost, per 5 s (diagnostic).
            static const bool stats_enabled = std::getenv("EDEN_NX2_PROTECT_STATS") != nullptr;
            const auto stats_start = stats_enabled ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
            host_buffer->Protect(vaddr, size, perm);
            if (stats_enabled) [[unlikely]] {
                static std::atomic<u64> calls[2]{}, pages[2]{}, nanos[2]{};
                static std::atomic<u64> window_start{0};
                const auto now = std::chrono::steady_clock::now();
                const u64 now_ns = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch())
                        .count());
                const int side = cached ? 1 : 0;
                calls[side].fetch_add(1, std::memory_order_relaxed);
                pages[side].fetch_add(size >> YUZU_PAGEBITS, std::memory_order_relaxed);
                nanos[side].fetch_add(
                    static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(now - stats_start)
                            .count()),
                    std::memory_order_relaxed);
                u64 started = window_start.load(std::memory_order_relaxed);
                if (started == 0) {
                    window_start.compare_exchange_strong(started, now_ns);
                } else if (now_ns - started > 5'000'000'000ULL &&
                           window_start.compare_exchange_strong(started, now_ns)) {
                    const double seconds = static_cast<double>(now_ns - started) / 1e9;
                    LOG_INFO(HW_Memory,
                             "nx2 protect: to read-only (tracked) {:.0f}/s ({:.0f} pages/s, {:.0f} "
                             "ms/s), to writable (untracked) {:.0f}/s ({:.0f} pages/s, {:.0f} ms/s)",
                             static_cast<double>(calls[1].exchange(0)) / seconds,
                             static_cast<double>(pages[1].exchange(0)) / seconds,
                             static_cast<double>(nanos[1].exchange(0)) / 1e6 / seconds,
                             static_cast<double>(calls[0].exchange(0)) / seconds,
                             static_cast<double>(pages[0].exchange(0)) / seconds,
                             static_cast<double>(nanos[0].exchange(0)) / 1e6 / seconds);
                }
            }
            if (cached && perm != Common::MemoryPermission{}) {
                ProtectArmedPages(vaddr, size, trap_locked);
            }
        }

        // Iterate over a contiguous CPU address space, which corresponds to the specified GPU
        // address space, marking the region as un/cached. The region is marked un/cached at a
        // granularity of CPU pages, hence why we iterate on a CPU page basis (note: GPU page size
        // is different). This assumes the specified GPU address region is contiguous as well.

        const u64 num_pages = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) - (vaddr >> VESHIRU_PAGEBITS) + 1;
        current_page_table->entries.CommitRegion(vaddr >> VESHIRU_PAGEBITS, (vaddr >> VESHIRU_PAGEBITS) + num_pages);
        for (u64 i = 0; i < num_pages; ++i, vaddr += VESHIRU_PAGESIZE) {
            auto& entry = current_page_table->entries.GetUnchecked(vaddr >> VESHIRU_PAGEBITS);
            const Common::PageType page_type = entry.Type();
            if (cached) {
                // Switch page type to cached if now cached
                switch (page_type) {
                case Common::PageType::Unmapped:
                    // It is not necessary for a process to have this region mapped into its address
                    // space, for example, a system module need not have a VRAM mapping.
                    break;
                case Common::PageType::DebugMemory:
                case Common::PageType::Memory:
                    entry.MarkRasterizerCached();
                    break;
                case Common::PageType::RasterizerCachedMemory:
                    // There can be more than one GPU region mapped per CPU region, so it's common
                    // that this area is already marked as cached.
                    break;
                default:
                    UNREACHABLE();
                }
            } else {
                // Switch page type to uncached if now uncached
                switch (page_type) {
                case Common::PageType::Unmapped: // NOLINT(bugprone-branch-clone)
                    // It is not necessary for a process to have this region mapped into its address
                    // space, for example, a system module need not have a VRAM mapping.
                    break;
                case Common::PageType::DebugMemory:
                case Common::PageType::Memory:
                    // There can be more than one GPU region mapped per CPU region, so it's common
                    // that this area is already unmarked as cached.
                    break;
                case Common::PageType::RasterizerCachedMemory: {
                    if (auto [ptr, _, block] = entry.PointerTypeBlock(true); ptr == 0) {
                        // It's possible that this function has been called while updating the
                        // pagetable after unmapping a VMA. In that case the underlying VMA will no
                        // longer exist, and we should just leave the pagetable entry blank.
                        entry.Store(false, Common::PageType::Unmapped, block, 0);
                    } else {
                        entry.Store(false, Common::PageType::Memory, block, ptr);
                    }
                    break;
                }
                default:
                    UNREACHABLE();
                }
            }
        }
    }

    /// Read traps replace reactive flushing's blanket read protection of every cached page: only
    /// pages the GPU writes are protected, and each one only until its first read
    /// (EDEN_NX2_READ_TRAPS=0 turns them off). Under NCE every fault is taken as a write, so they
    /// are not used there.
    bool ReadTrapsEnabled() const {
        static const bool disabled = [] {
            const char* value = std::getenv("EDEN_NX2_READ_TRAPS");
            return value != nullptr && value[0] == '0';
        }();
        return !disabled && current_page_table->fastmem_arena &&
               !Settings::ReactiveFlushing() &&
               Settings::values.cpu_backend.GetValue() != Settings::CpuBackend::Nce;
    }

    void RasterizerArmReadTraps(u64 vaddr, u64 size) {
        if (Settings::values.use_reactive_flushing.GetValue()) {
            // Clean cached read areas also need invalidation when reactive flushing is enabled.
            read_trap_generation.fetch_add(1, std::memory_order_release);
        }
        if (vaddr == 0 || size == 0 || !AddressSpaceContains(*current_page_table, vaddr, size) ||
            !ReadTrapsEnabled()) {
            return;
        }
        const u64 page_end = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1;
        std::scoped_lock lk{read_trap_mutex};
        u64 run_begin = 0;
        u64 run_pages = 0;
        const auto protect_run = [&] {
            if (run_pages > 0) {
                // Reads that hit a cached read area must not skip the flush of a new trap.
                read_trap_generation.fetch_add(1, std::memory_order_release);
                host_buffer->Protect(run_begin << VESHIRU_PAGEBITS, run_pages << VESHIRU_PAGEBITS,
                                     Common::MemoryPermission{});
                run_pages = 0;
            }
        };
        for (u64 page = vaddr >> VESHIRU_PAGEBITS; page != page_end; ++page) {
            const bool cached = std::as_const(current_page_table->entries)[page].Type() ==
                                Common::PageType::RasterizerCachedMemory;
            armed_read_trap_count.fetch_add(1, std::memory_order_relaxed);
            if (!cached || !armed_read_traps.insert(page).second) {
                armed_read_trap_count.fetch_sub(1, std::memory_order_relaxed);
                protect_run();
                continue;
            }
            if (run_pages == 0) {
                run_begin = page;
            }
            ++run_pages;
        }
        protect_run();
    }

    bool ReadTrapArmed(u64 vaddr, u64 size) {
        if (armed_read_trap_count.load(std::memory_order_relaxed) == 0) {
            return false;
        }
        std::scoped_lock lk{read_trap_mutex};
        const auto it = armed_read_traps.lower_bound(vaddr >> VESHIRU_PAGEBITS);
        return it != armed_read_traps.end() && *it <= ((vaddr + size - 1) >> VESHIRU_PAGEBITS);
    }

    /// Takes read permission away again from armed pages that were just cached again.
    void ProtectArmedPages(u64 vaddr, u64 size, bool trap_locked) {
        if (armed_read_trap_count.load(std::memory_order_relaxed) == 0) {
            return;
        }
        std::unique_lock lk{read_trap_mutex, std::defer_lock};
        if (!trap_locked) {
            lk.lock();
        }
        const u64 page_end = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1;
        for (auto it = armed_read_traps.lower_bound(vaddr >> VESHIRU_PAGEBITS);
             it != armed_read_traps.end() && *it < page_end; ++it) {
            host_buffer->Protect(*it << VESHIRU_PAGEBITS, VESHIRU_PAGESIZE, Common::MemoryPermission{});
        }
    }

    /// A slow-path read of a cached page: its trap, if any, has done its job.
    void DisarmReadTraps(u64 vaddr, size_t size) {
        if (armed_read_trap_count.load(std::memory_order_relaxed) == 0) {
            return;
        }
        std::scoped_lock lk{read_trap_mutex};
        const u64 page_end = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1;
        auto it = armed_read_traps.lower_bound(vaddr >> VESHIRU_PAGEBITS);
        while (it != armed_read_traps.end() && *it < page_end) {
            // Still cached (this read came through the rasterizer path): writes stay trapped, and
            // reads too while the page tracker waits for the next access.
            if (page_tracker_state.Empty() || !page_tracker_state.AccessArmed(*it)) {
                host_buffer->Protect(*it << VESHIRU_PAGEBITS, VESHIRU_PAGESIZE,
                                     Common::MemoryPermission::Read);
            }
            it = armed_read_traps.erase(it);
            armed_read_trap_count.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    /**
     * Maps a region of pages as a specific type.
     *
     * @param page_table The page table to use to perform the mapping.
     * @param base       The base address to begin mapping at.
     * @param size       The total size of the range in bytes.
     * @param target     The target address to begin mapping from.
     * @param type       The page type to map the memory as.
     */
    void MapPages(Common::PageTable& page_table, Common::ProcessAddress base_address, u64 size,
                  Common::PhysicalAddress target, Common::PageType type) {
        auto base = GetInteger(base_address);

        LOG_DEBUG(HW_Memory, "Mapping {:016X} onto {:016X}-{:016X}", GetInteger(target),
                  base * VESHIRU_PAGESIZE, (base + size) * VESHIRU_PAGESIZE);

        const auto end = base + size;
        ASSERT_MSG(end <= page_table.entries.size(), "out of range mapping at {:016X}", base + page_table.entries.size());

        if (!target) {
            ASSERT_MSG(type != Common::PageType::Memory,
                       "Mapping memory page without a pointer @ {:016x}", base * VESHIRU_PAGESIZE);

            last_mapped_table = nullptr;
            page_table.entries.ZeroRegion(base, end);
        } else {
            u64 mapping_id = block_count.load(std::memory_order_relaxed);
            constexpr u64 exhausted_count =
                u64{Common::PageTable::PageEntryData::INVALID_BLOCK_ID} + 1;
            while (mapping_id < exhausted_count &&
                   !block_count.compare_exchange_weak(mapping_id, mapping_id + 1,
                                                      std::memory_order_relaxed)) {
            }
            const u16 current_block =
                Common::PageTable::PageEntryData::BlockIdForMapping(mapping_id);

            page_table.entries.CommitRegion(base, end);
            while (base != end) {
                auto host_ptr = reinterpret_cast<u64>(system.DeviceMemory().GetPointer<u8>(target)) - (base << VESHIRU_PAGEBITS);;
                auto& entry = page_table.entries.GetUnchecked(base);

                entry.Store(false, type, current_block, host_ptr);
                ASSERT_MSG(page_table.entries[base].Pointer(),
                           "memory mapping base yield a nullptr within the table");

                base += 1;
                target += VESHIRU_PAGESIZE;
            }
        }
    }

    template<typename F, typename G>
    [[nodiscard]] u8* GetPointerImpl(u64 vaddr, F&& on_unmapped, G&& on_rasterizer) const {
        // AARCH64 masks the upper 16 bit of all memory accesses
        vaddr &= 0xffffffffffffULL;
        if (AddressSpaceContains(*current_page_table, vaddr, 1)) [[likely]] {
            // Avoid adding any extra logic to this fast-path block
            const auto raw = current_page_table->entries[vaddr >> VESHIRU_PAGEBITS].Raw();
            if (auto pointer = Common::PageTable::PageEntryData::ExtractPointer(raw); pointer) [[likely]] {
                return reinterpret_cast<u8*>(pointer + vaddr);
            } else {
                switch (static_cast<Common::PageType>(raw.type)) {
                case Common::PageType::Memory:
                    ASSERT_MSG(false, "Mapped memory page without a pointer @ {:#016x}", vaddr);
                    return nullptr;
                case Common::PageType::DebugMemory:
                    return GetPointerFromDebugMemory(vaddr);
                case Common::PageType::RasterizerCachedMemory: {
                    u8* const host_ptr{GetPointerFromRasterizerCachedMemory(vaddr)};
                    on_rasterizer();
                    return host_ptr;
                }
                case Common::PageType::Unmapped: [[unlikely]] {
                    on_unmapped();
                    return nullptr;
                }
                default:
                    UNREACHABLE();
                }
                return nullptr;
            }
        } else {
            on_unmapped();
            return nullptr;
        }
    }

    [[nodiscard]] u8* GetPointer(const Common::ProcessAddress vaddr) const {
        return GetPointerImpl(
            GetInteger(vaddr),
            [vaddr]() {
                LOG_ERROR(HW_Memory, "Unmapped GetPointer @ {:#016x}", GetInteger(vaddr));
            },
            []() {});
    }

    [[nodiscard]] u8* GetPointerSilent(const Common::ProcessAddress vaddr) const {
        return GetPointerImpl(
            GetInteger(vaddr), []() {}, []() {});
    }

    /// @brief Reads a particular data type out of memory at the given virtual address.
    /// @param vaddr The virtual address to read the data type from.
    /// @tparam T The data type to read out of memory.
    /// @returns The instance of T read from the specified virtual address.
    template <typename T>
    inline T Read(Common::ProcessAddress vaddr) noexcept requires(std::is_trivially_copyable_v<T>) {
        const u64 addr = GetInteger(vaddr);
        if (auto const ptr = GetPointerImpl(addr, [addr]() {
            // A guest spinning on a bad pointer reads unmapped memory tens of thousands of times
            // a second: keep the first ones and then a sample.
            static std::atomic<u64> unmapped_reads{};
            if (const u64 count = unmapped_reads.fetch_add(1); count < 1000 || count % 100000 == 0) {
                LOG_ERROR(HW_Memory, "Unmapped Read{} @ {:#016x} (#{})", sizeof(T) * 8, addr, count);
            }
            ++UnmappedReadFlag();
        }, [&]() {
            if (!ReadTrapsEnabled() || ReadTrapArmed(addr & 0xffffffffffffULL, sizeof(T))) {
                HandleRasterizerDownload(addr, sizeof(T));
            }
        }); ptr) [[likely]] {
            // It may be tempting to rewrite this particular section to use "reinterpret_cast";
            // afterall, it's trivially copyable so surely it can be copied ov- Alignment.
            // Remember, alignment. memcpy() will deal with all the alignment extremely fast.
            T result{};
            std::memcpy(&result, ptr, sizeof(T));
            return result;
        }
        return T{};
    }

    /// @brief Writes a particular data type to memory at the given virtual address.
    /// @param vaddr The virtual address to write the data type to.
    /// @tparam T The data type to write to memory.
    template <typename T>
    inline void Write(Common::ProcessAddress vaddr, const T data) noexcept requires(std::is_trivially_copyable_v<T>) {
        const u64 addr = GetInteger(vaddr);
        bool cached = false;
        bool rasterizer_write = false;
        if (auto const ptr = GetPointerImpl(addr, [addr, data]() {
            // As with unmapped reads, a guest spinning on a bad pointer writes unmapped memory
            // tens of thousands of times a second: keep the first ones and then a sample.
            // Ported from Lichii nx2 528dae4b (GPL-3.0-or-later).
            static std::atomic<u64> unmapped_writes{};
            if (const u64 count = unmapped_writes.fetch_add(1);
                count < 1000 || count % 100000 == 0) {
                LOG_ERROR(HW_Memory, "Unmapped Write{} @ {:#016x} = {:#016x} (#{})",
                          sizeof(T) * 8, addr, u64(data), count);
            }
        }, [&]() {
            cached = true;
            rasterizer_write = RasterizerPreWrite(addr, sizeof(T));
        }); ptr) [[likely]] {
            std::memcpy(ptr, &data, sizeof(T));
            if (cached) [[unlikely]] {
                NotifyHostWrite(addr, sizeof(T));
            }
        }
        if (rasterizer_write) [[unlikely]] {
            EndRasterizerWrite();
        }
    }

    template <typename T>
    bool WriteExclusive(Common::ProcessAddress vaddr, const T data, const T expected) {
        bool cached = false;
        bool rasterizer_write = false;
        u8* const ptr = GetPointerImpl(
            GetInteger(vaddr),
            [vaddr, data]() {
                static std::atomic<u64> unmapped_exclusive_writes{};
                if (const u64 count = unmapped_exclusive_writes.fetch_add(1);
                    count < 1000 || count % 100000 == 0) {
                    LOG_ERROR(HW_Memory, "Unmapped WriteExclusive{} @ {:#016x} = {:#016x} (#{})",
                              sizeof(T) * 8, GetInteger(vaddr), static_cast<u64>(data), count);
                }
            },
            [&]() {
                cached = true;
                rasterizer_write = RasterizerPreWrite(GetInteger(vaddr), sizeof(T));
            });
        SCOPE_EXIT {
            if (rasterizer_write) {
                EndRasterizerWrite();
            }
        };
        if (ptr) {
            const bool result = Common::AtomicCompareAndSwap(reinterpret_cast<T*>(ptr), data, expected);
            if (cached && result) [[unlikely]] {
                NotifyHostWrite(GetInteger(vaddr), sizeof(T));
            }
            return result;
        }
        return true;
    }

    bool WriteExclusive128(Common::ProcessAddress vaddr, const u128 data, const u128 expected) {
        bool cached = false;
        bool rasterizer_write = false;
        u8* const ptr = GetPointerImpl(
            GetInteger(vaddr),
            [vaddr, data]() {
                static std::atomic<u64> unmapped_exclusive_writes_128{};
                if (const u64 count = unmapped_exclusive_writes_128.fetch_add(1);
                    count < 1000 || count % 100000 == 0) {
                    LOG_ERROR(HW_Memory,
                              "Unmapped WriteExclusive128 @ {:#016x} = {:#016x}{:016X} (#{})",
                              GetInteger(vaddr), static_cast<u64>(data[1]),
                              static_cast<u64>(data[0]), count);
                }
            },
            [&]() {
                cached = true;
                rasterizer_write = RasterizerPreWrite(GetInteger(vaddr), sizeof(u128));
            });
        SCOPE_EXIT {
            if (rasterizer_write) {
                EndRasterizerWrite();
            }
        };
        if (ptr) {
            const bool result = Common::AtomicCompareAndSwap(reinterpret_cast<u64*>(ptr), data, expected);
            if (cached && result) [[unlikely]] {
                NotifyHostWrite(GetInteger(vaddr), sizeof(u128));
            }
            return result;
        }
        return true;
    }

    /// Runs the access-trap callbacks of tracked pages in [vaddr, vaddr + size) on this thread,
    /// before the access lands, without the trap lock. Returns whether the device caches still
    /// have to see the access (a page there is untracked or device cached).
    bool TrackerAccess(u64 vaddr, size_t size) {
        if (size == 0) {
            return true;
        }
        vaddr &= 0xffffffffffffULL;
        const u64 page_begin = vaddr >> VESHIRU_PAGEBITS;
        const u64 page_end = ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1;
        std::vector<PageTrackerState::AccessHit> hits;
        bool device_path;
        {
            std::scoped_lock lk{read_trap_mutex};
            device_path = page_tracker_state.CollectAccessHits(page_begin, page_end, hits);
        }
        if (hits.empty()) {
            return device_path;
        }
        for (const PageTrackerState::AccessHit& hit : hits) {
            if (hit.callback) {
                (*hit.callback)(hit.page_begin << VESHIRU_PAGEBITS,
                                (hit.page_end - hit.page_begin) << VESHIRU_PAGEBITS);
            }
        }
        std::scoped_lock lk{read_trap_mutex};
        page_tracker_state.FinishAccess(hits);
        // The callbacks ran without the lock and may have changed the tracking (untracked or
        // device-cached pages): decide on the state the access lands in.
        return page_tracker_state.NeedsDevicePath(page_begin, page_end);
    }

    /// A cached page is about to be written: access traps there call back first.
    void TrackerPreWrite(u64 vaddr, size_t size) {
        if (page_tracker_state.HasAccessTraps()) [[unlikely]] {
            TrackerAccess(vaddr, size);
        }
    }

    /// A guest write to cached pages: tracked pages turn dirty and lose their write trap. Returns
    /// whether the device caches still have to see the write.
    bool TrackerWrite(u64 vaddr, size_t size) {
        if (size == 0) {
            return true;
        }
        vaddr &= 0xffffffffffffULL;
        std::scoped_lock lk{read_trap_mutex};
        return page_tracker_state.OnWrite(vaddr >> VESHIRU_PAGEBITS,
                                          ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1);
    }

    /// Whether the device caches have to see an access to [vaddr, vaddr + size).
    bool TrackerDevicePath(u64 vaddr, size_t size) {
        if (page_tracker_state.Empty() || size == 0) {
            return true;
        }
        vaddr &= 0xffffffffffffULL;
        std::scoped_lock lk{read_trap_mutex};
        return page_tracker_state.NeedsDevicePath(vaddr >> VESHIRU_PAGEBITS,
                                                  ((vaddr + size - 1) >> VESHIRU_PAGEBITS) + 1);
    }

    /// A store to cached pages is about to land: access traps call back first, and the device
    /// caches see it unless only the page tracker caches those pages. The page tracker counts the
    /// store once it landed (NotifyHostWrite), so a collect in between cannot miss it.
    bool RasterizerPreWrite(VAddr v_address, size_t size) {
        if (!page_tracker_state.Empty()) [[unlikely]] {
            TrackerPreWrite(v_address, size);
            if (!TrackerDevicePath(v_address, size)) {
                return false;
            }
        }
        HandleRasterizerWrite(v_address, size);
        return true;
    }

    /// A data cache clean of cached pages: the page tracker and the device caches see a write.
    void HandleCacheStore(VAddr v_address, size_t size) {
        if (!page_tracker_state.Empty() && !TrackerWrite(v_address, size)) [[unlikely]] {
            return;
        }
        HandleRasterizerWrite(v_address, size);
        EndRasterizerWrite();
    }

    /// A rasterizer-cached page whose host pointer is outside the emulated physical memory has no
    /// device address to flush or invalidate (its lookup used to read far out of bounds).
    void ReportForeignPointer(VAddr v_address, const u8* p) const {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1, std::memory_order_relaxed) >= 16) {
            return;
        }
        const auto raw =
            current_page_table->entries[(v_address & 0xffffffffffffULL) >> VESHIRU_PAGEBITS].Raw();
        LOG_ERROR(HW_Memory,
                  "Rasterizer access at {:#x}: host pointer {} is outside the emulated physical "
                  "memory at {} (page type {}, block {})",
                  v_address, fmt::ptr(p),
                  fmt::ptr(system.DeviceMemory().buffer.BackingBasePointer()),
                  static_cast<u32>(raw.type), static_cast<u32>(raw.block));
    }

    void HandleRasterizerDownload(VAddr v_address, size_t size) {
        if (!page_tracker_state.Empty()) [[unlikely]] {
            // Access traps call back first. Pages only the tracker caches have nothing for the
            // device caches to flush.
            if (!TrackerAccess(v_address, size)) {
                return;
            }
        }
        if (!Kernel::Nx2ThreadStats::g_enabled) {
            DownloadFromRasterizer(v_address, size);
            return;
        }
        auto& timing = system.CoreTiming();
        const u64 start = timing.GetClockTicks();
        const u64 reads = VideoCore::AsyncReadback::t_reads;
        DownloadFromRasterizer(v_address, size);
        Kernel::Nx2ThreadStats::CountMemory(Kernel::Nx2ThreadStats::t_run,
                                            timing.GetClockTicks() - start,
                                            VideoCore::AsyncReadback::t_reads != reads);
    }

    void DownloadFromRasterizer(VAddr v_address, size_t size) {
        const auto* p = GetPointerImpl(
            v_address, []() {}, []() {});
        if (!gpu_device_memory) [[unlikely]] {
            gpu_device_memory = &system.Host1x().MemoryManager();
        }
        if (!gpu_device_memory->IsPhysicalPointer(p)) [[unlikely]] {
            ReportForeignPointer(v_address, p);
            return;
        }
        const size_t core = system.GetCurrentHostThreadID();
        if (core >= Core::Hardware::NUM_CPU_CORES) {
            // A service thread has no slot in the arrays below, and it does not use a slot of a
            // core. It keeps no area, so it asks on each read.
            if (static std::atomic_flag logged; !logged.test_and_set(std::memory_order_relaxed)) {
                LOG_INFO(HW_Memory, "Read of cached guest memory from a thread that is not a core");
            }
            Common::ScratchBuffer<u32> scratch;
            gpu_device_memory->ApplyOpOnPointer(p, scratch, [&](DAddr address) {
                DisarmReadTraps(v_address, size);
                (void)system.GPU().OnCPURead(address, size);
            });
            return;
        }
        auto& current_area = rasterizer_read_areas[core];
        if (const u64 generation = read_trap_generation.load(std::memory_order_acquire);
            read_area_generations[core] != generation) {
            current_area = {};
            read_area_generations[core] = generation;
        }
        gpu_device_memory->ApplyOpOnPointer(p, scratch_buffers[core], [&](DAddr address) {
            const DAddr end_address = address + size;
            if (current_area.start_address <= address && end_address <= current_area.end_address)
                [[likely]] {
                return;
            }
            DisarmReadTraps(v_address, size);
            current_area = system.GPU().OnCPURead(address, size);
        });
    }

    /// Callers store the bytes after this and call EndRasterizerWrite once they landed.
    void HandleRasterizerWrite(VAddr v_address, size_t size) {
        VideoCommon::Nx2FastDraw::BeginMemoryChange();
        if (!Kernel::Nx2ThreadStats::g_enabled) {
            WriteToRasterizer(v_address, size);
            return;
        }
        auto& timing = system.CoreTiming();
        const u64 start = timing.GetClockTicks();
        WriteToRasterizer(v_address, size);
        Kernel::Nx2ThreadStats::CountMemory(Kernel::Nx2ThreadStats::t_run,
                                            timing.GetClockTicks() - start, false);
    }

    void EndRasterizerWrite() {
        VideoCommon::Nx2FastDraw::EndMemoryChange();
    }

    void WriteToRasterizer(VAddr v_address, size_t size) {
        const auto* p = GetPointerImpl(
            v_address, []() {}, []() {});
        constexpr size_t sys_core = Core::Hardware::NUM_CPU_CORES - 1;
        const size_t core = (std::min)(system.GetCurrentHostThreadID(),
                                     sys_core); // any other calls threads go to syscore.
        if (!gpu_device_memory) {
            gpu_device_memory = &system.Host1x().MemoryManager();
        }
        if (!gpu_device_memory->IsPhysicalPointer(p)) {
            ReportForeignPointer(v_address, p);
            return;
        }
        // Guard on sys_core;
        if (core == sys_core) {
            sys_core_guard.lock();
        }
        SCOPE_EXIT {
            if (core == sys_core) {
                sys_core_guard.unlock();
            }
        };
        const bool nx2_ampere = Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
        gpu_device_memory->ApplyOpOnPointer(p, scratch_buffers[core], [&](DAddr address) {
            auto& current_area = rasterizer_write_areas[core];
            PAddr subaddress = address >> VESHIRU_PAGEBITS;
            if (current_area.last_address != subaddress) {
                if (address == 0 || size == 0) {
                    return;
                }
                // On Switch 2 (SPLIT=2), pages never written by the GPU need no synchronous
                // OnCPUWrite under BufferCache/TextureCache locks: collecting into
                // gpu_dirty_managers applies the invalidation at the next FlushCaching without
                // stalling guest CPU cores up to 5-6 ms behind VulkanWorker.
                const bool collect_unasked =
                    nx2_ampere && !Tegra::Nx2MayBeGpuWritten(address, size);
                const bool do_collection =
                    collect_unasked || system.GPU().OnCPUWrite(address, size);
                if (do_collection) {
                    current_area.last_address = subaddress;
                } else {
                    return;
                }
            }
            VideoCommon::Nx2Hazard::NoteGuestWrite(address, size);
            gpu_dirty_managers[core].Collect(address, size);
            // A download writing to guest memory here may have looked at the collected writes
            // before this one: its bytes would land over the ones stored after this returns.
            VideoCore::DownloadLanding::Await(address, size);
            VideoCommon::g_collected_write_pages.Mark(address, size);
        });
    }

    struct GPUDirtyState {
        PAddr last_address;
    };

    void InvalidateGPUMemory(u8* p, size_t size) {
        // NCE stores the bytes once the trap returns: flagged before they land (pages it
        // untracks are flagged once unwatched).
        VideoCommon::Nx2FastDraw::BeginMemoryChange();
        VideoCommon::Nx2FastDraw::EndMemoryChange();
        constexpr size_t sys_core = Core::Hardware::NUM_CPU_CORES - 1;
        const size_t core = (std::min)(system.GetCurrentHostThreadID(),
                                     sys_core); // any other calls threads go to syscore.
        if (!gpu_device_memory) [[unlikely]] {
            gpu_device_memory = &system.Host1x().MemoryManager();
        }
        // Guard on sys_core;
        if (core == sys_core) [[unlikely]] {
            sys_core_guard.lock();
        }
        SCOPE_EXIT {
            if (core == sys_core) [[unlikely]] {
                sys_core_guard.unlock();
            }
        };
        auto& gpu = system.GPU();
        gpu_device_memory->ApplyOpOnPointer(
            p, scratch_buffers[core], [&](DAddr address) { gpu.InvalidateRegion(address, size, true); });
    }

    Core::System& system;
    Tegra::MaxwellDeviceMemoryManager* gpu_device_memory{};
    Common::PageTable* current_page_table = nullptr;

    std::array<VideoCore::RasterizerDownloadArea, Core::Hardware::NUM_CPU_CORES> rasterizer_read_areas{};
    std::array<u64, Core::Hardware::NUM_CPU_CORES> read_area_generations{};
    std::atomic<u64> read_trap_generation{};
    std::set<u64> armed_read_traps;
    std::atomic<size_t> armed_read_trap_count{};
    /// Guards read traps, and page types and the page tracker while the tracker is enabled.
    std::mutex read_trap_mutex;
    std::atomic<bool> page_tracker_enabled{};
    PageTrackerState page_tracker_state{*this};
    Core::Memory::PageTracker page_tracker{*this};
    std::array<GPUDirtyState, Core::Hardware::NUM_CPU_CORES> rasterizer_write_areas{};
    std::array<Common::ScratchBuffer<u32>, Core::Hardware::NUM_CPU_CORES> scratch_buffers{};
    std::span<Core::GPUDirtyMemoryManager> gpu_dirty_managers;
    std::mutex sys_core_guard;
#ifdef __ANDROID__
    std::optional<Common::HeapTracker> heap_tracker;
    Common::HeapTracker* host_buffer{};
#else
    Common::HostMemory* host_buffer{};
#endif
    std::atomic<u64> block_count = 0;
    const Common::PageTable* last_mapped_table{};
    u64 last_mapped_end{};
    u64 last_mapped_host_ptr{};
    u16 last_mapped_block{Common::PageTable::PageEntryData::INVALID_BLOCK_ID};
};

Memory::Memory(Core::System& system_) : system{system_} {
    Reset();
}

Memory::~Memory() = default;

void Memory::Reset() {
    VideoCommon::g_collected_write_pages.Clear();
    VideoCommon::g_buffer_write_pages.Clear();
    impl = std::make_unique<Impl>(system);
}

void Memory::SetCurrentPageTable(Kernel::KProcess& process) {
    impl->SetCurrentPageTable(process);
}

void Memory::MapMemoryRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                             Common::PhysicalAddress target, Common::MemoryPermission perms,
                             bool separate_heap) {
    impl->MapMemoryRegion(page_table, base, size, target, perms, separate_heap);
}

void Memory::UnmapRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                         bool separate_heap) {
    impl->UnmapRegion(page_table, base, size, separate_heap);
}

void Memory::ProtectRegion(Common::PageTable& page_table, Common::ProcessAddress vaddr, u64 size,
                           Common::MemoryPermission perms) {
    impl->ProtectRegion(page_table, GetInteger(vaddr), size, perms);
}

bool Memory::IsValidVirtualAddress(const Common::ProcessAddress vaddr) const {
    const auto& page_table = *impl->current_page_table;
    const size_t page = vaddr >> VESHIRU_PAGEBITS;
    if (page >= page_table.entries.size()) {
        return false;
    }
    const auto [pointer, type, _] = page_table.entries[page].PointerTypeBlock();
    return pointer != 0 || type == Common::PageType::RasterizerCachedMemory ||
           type == Common::PageType::DebugMemory;
}

bool Memory::IsValidVirtualAddressRange(Common::ProcessAddress base, u64 size) const {
    Common::ProcessAddress end = base + size;
    if (end < base) {
        return false;
    }
    Common::ProcessAddress page = Common::AlignDown(GetInteger(base), VESHIRU_PAGESIZE);

    for (; page < end; page += VESHIRU_PAGESIZE) {
        if (!IsValidVirtualAddress(page)) {
            return false;
        }
    }

    return true;
}

u8* Memory::GetPointer(Common::ProcessAddress vaddr) {
    return impl->GetPointer(vaddr);
}

u8* Memory::GetPointerSilent(Common::ProcessAddress vaddr) {
    return impl->GetPointerSilent(vaddr);
}

const u8* Memory::GetPointer(Common::ProcessAddress vaddr) const {
    return impl->GetPointer(vaddr);
}

u8 Memory::Read8(const Common::ProcessAddress addr) {
    return impl->Read8(addr);
}

u16 Memory::Read16(const Common::ProcessAddress addr) {
    return impl->Read16(addr);
}

u32 Memory::Read32(const Common::ProcessAddress addr) {
    return impl->Read32(addr);
}

bool Memory::Read32Acquire(const Common::ProcessAddress addr, u32& value) {
    return impl->Read32Acquire(addr, value);
}

u64 Memory::Read64(const Common::ProcessAddress addr) {
    return impl->Read64(addr);
}

void Memory::Write8(Common::ProcessAddress addr, u8 data) {
    impl->Write8(addr, data);
}

void Memory::Write16(Common::ProcessAddress addr, u16 data) {
    impl->Write16(addr, data);
}

void Memory::Write32(Common::ProcessAddress addr, u32 data) {
    impl->Write32(addr, data);
}

void Memory::Write64(Common::ProcessAddress addr, u64 data) {
    impl->Write64(addr, data);
}

bool Memory::WriteExclusive8(Common::ProcessAddress addr, u8 data, u8 expected) {
    return impl->WriteExclusive8(addr, data, expected);
}

bool Memory::WriteExclusive16(Common::ProcessAddress addr, u16 data, u16 expected) {
    return impl->WriteExclusive16(addr, data, expected);
}

bool Memory::WriteExclusive32(Common::ProcessAddress addr, u32 data, u32 expected) {
    return impl->WriteExclusive32(addr, data, expected);
}

bool Memory::WriteExclusive64(Common::ProcessAddress addr, u64 data, u64 expected) {
    return impl->WriteExclusive64(addr, data, expected);
}

bool Memory::WriteExclusive128(Common::ProcessAddress addr, u128 data, u128 expected) {
    return impl->WriteExclusive128(addr, data, expected);
}

std::string Memory::ReadCString(Common::ProcessAddress vaddr, std::size_t max_length) {
    return impl->ReadCString(vaddr, max_length);
}

bool Memory::ReadBlock(const Common::ProcessAddress src_addr, void* dest_buffer,
                       const std::size_t size) {
    return impl->ReadBlockImpl(src_addr, dest_buffer, size, false);
}

bool Memory::ReadBlockUnsafe(const Common::ProcessAddress src_addr, void* dest_buffer,
                             const std::size_t size) {
    return impl->ReadBlockImpl(src_addr, dest_buffer, size, true);
}

const u8* Memory::GetSpan(const VAddr src_addr, const std::size_t size) const {
    return impl->GetSpan(src_addr, size);
}

u8* Memory::GetSpan(const VAddr src_addr, const std::size_t size) {
    return impl->GetSpan(src_addr, size);
}

bool Memory::WriteBlock(const Common::ProcessAddress dest_addr, const void* src_buffer,
                        const std::size_t size) {
    return impl->WriteBlockImpl(dest_addr, src_buffer, size, Impl::WriteMode::Safe);
}

bool Memory::WriteBlockUnsafe(const Common::ProcessAddress dest_addr, const void* src_buffer,
                              const std::size_t size) {
    return impl->WriteBlockImpl(dest_addr, src_buffer, size, Impl::WriteMode::Unsafe);
}

bool Memory::WriteBlockUntracked(const Common::ProcessAddress dest_addr, const void* src_buffer,
                                 const std::size_t size) {
    return impl->WriteBlockImpl(dest_addr, src_buffer, size, Impl::WriteMode::Untracked);
}

void Memory::NotifyHostWrite(Common::ProcessAddress dest_addr, std::size_t size) {
    impl->NotifyHostWrite(GetInteger(dest_addr), size);
}

bool Memory::CopyBlock(Common::ProcessAddress dest_addr, Common::ProcessAddress src_addr,
                       const std::size_t size) {
    return impl->CopyBlock(dest_addr, src_addr, size);
}

bool Memory::ZeroBlock(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->ZeroBlock(dest_addr, size);
}

void Memory::SetGPUDirtyManagers(std::span<Core::GPUDirtyMemoryManager> managers) {
    impl->gpu_dirty_managers = managers;
}

Result Memory::InvalidateDataCache(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->InvalidateDataCache(dest_addr, size);
}

Result Memory::StoreDataCache(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->StoreDataCache(dest_addr, size);
}

Result Memory::FlushDataCache(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->FlushDataCache(dest_addr, size);
}

void Memory::RasterizerMarkRegionCached(Common::ProcessAddress vaddr, u64 size, bool cached) {
    impl->RasterizerMarkRegionCached(GetInteger(vaddr), size, cached);
}

void Memory::RasterizerArmReadTraps(Common::ProcessAddress vaddr, u64 size) {
    impl->RasterizerArmReadTraps(GetInteger(vaddr), size);
}

void Memory::MarkRegionDebug(Common::ProcessAddress vaddr, u64 size, bool debug) {
    impl->MarkRegionDebug(GetInteger(vaddr), size, debug);
}

bool Memory::InvalidateNCE(Common::ProcessAddress vaddr, size_t size) {
    [[maybe_unused]] bool mapped = true;
    [[maybe_unused]] bool rasterizer = false;

    u8* const ptr = impl->GetPointerImpl(
        GetInteger(vaddr),
        [&] {
            LOG_ERROR(HW_Memory, "Unmapped InvalidateNCE for {} bytes @ {:#x}", size,
                      GetInteger(vaddr));
            mapped = false;
        },
        [&] { rasterizer = true; });
    if (rasterizer) {
        bool device_path = true;
        if (!impl->page_tracker_state.Empty()) [[unlikely]] {
            // The store runs again after this returns: access traps call back before it, and a
            // tracked page turns dirty and writable.
            impl->TrackerPreWrite(GetInteger(vaddr), size);
            device_path = impl->TrackerWrite(GetInteger(vaddr), size);
        }
        if (device_path) {
            impl->InvalidateGPUMemory(ptr, size);
        } else {
            // Even when only the page tracker watches these pages, bump the fast-draw change epoch.
            VideoCommon::Nx2FastDraw::BeginMemoryChange();
            VideoCommon::Nx2FastDraw::EndMemoryChange();
        }
    }

#ifdef __ANDROID__
    if (!rasterizer && mapped) {
        impl->host_buffer->DeferredMapSeparateHeap(GetInteger(vaddr));
    }
#endif

    return mapped && ptr != nullptr;
}

bool Memory::InvalidateSeparateHeap(void* fault_address) {
#ifdef __ANDROID__
    return impl->host_buffer->DeferredMapSeparateHeap(static_cast<u8*>(fault_address));
#else
    return false;
#endif
}

void Memory::EnablePageTracker() {
    impl->page_tracker_enabled.store(true, std::memory_order_relaxed);
}

PageTracker& Memory::PageTracker() {
    return impl->page_tracker;
}

u64 PageTracker::Register(u64 va, u64 size, bool armed) {
    va &= 0xffffffffffffULL;
    if (size == 0 || !impl.page_tracker_enabled.load(std::memory_order_relaxed) ||
        impl.current_page_table == nullptr ||
        !AddressSpaceContains(*impl.current_page_table, va, size)) {
        return 0;
    }
    std::scoped_lock lk{impl.read_trap_mutex};
    return impl.page_tracker_state.Register(va >> VESHIRU_PAGEBITS,
                                            ((va + size - 1) >> VESHIRU_PAGEBITS) + 1, armed);
}

void PageTracker::Unregister(u64 id) {
    std::scoped_lock lk{impl.read_trap_mutex};
    impl.page_tracker_state.Unregister(id);
}

void PageTracker::Arm(u64 id, u64 va, u64 size) {
    if (size == 0) {
        return;
    }
    va &= 0xffffffffffffULL;
    std::scoped_lock lk{impl.read_trap_mutex};
    impl.page_tracker_state.Arm(id, va >> VESHIRU_PAGEBITS, ((va + size - 1) >> VESHIRU_PAGEBITS) + 1);
}

void PageTracker::CollectDirty(u64 id, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs) {
    if (size == 0) {
        return;
    }
    va &= 0xffffffffffffULL;
    const size_t first = runs.size();
    {
        std::scoped_lock lk{impl.read_trap_mutex};
        impl.page_tracker_state.CollectDirty(id, va >> VESHIRU_PAGEBITS,
                                             ((va + size - 1) >> VESHIRU_PAGEBITS) + 1, runs);
    }
    for (size_t i = first; i < runs.size(); ++i) {
        runs[i].first <<= VESHIRU_PAGEBITS;
        runs[i].second <<= VESHIRU_PAGEBITS;
    }
}

bool PageTracker::AnyDirty(u64 id, u64 va, u64 size) {
    if (size == 0) {
        return false;
    }
    va &= 0xffffffffffffULL;
    std::scoped_lock lk{impl.read_trap_mutex};
    return impl.page_tracker_state.AnyDirty(id, va >> VESHIRU_PAGEBITS,
                                            ((va + size - 1) >> VESHIRU_PAGEBITS) + 1);
}

std::shared_ptr<const std::atomic<u64>> PageTracker::WriteEpoch(u64 id) {
    std::scoped_lock lk{impl.read_trap_mutex};
    return impl.page_tracker_state.WriteEpoch(id);
}

void PageTracker::SetAccessCallback(u64 id, AccessCallback callback) {
    std::scoped_lock lk{impl.read_trap_mutex};
    impl.page_tracker_state.SetAccessCallback(id, std::move(callback));
}

void PageTracker::ArmAccessTrap(u64 id, u64 va, u64 size) {
    if (size == 0) {
        return;
    }
    va &= 0xffffffffffffULL;
    std::scoped_lock lk{impl.read_trap_mutex};
    impl.page_tracker_state.ArmAccess(id, va >> VESHIRU_PAGEBITS,
                                      ((va + size - 1) >> VESHIRU_PAGEBITS) + 1);
}

void PageTracker::DisarmAccessTrap(u64 id, u64 va, u64 size) {
    if (size == 0) {
        return;
    }
    va &= 0xffffffffffffULL;
    std::scoped_lock lk{impl.read_trap_mutex};
    impl.page_tracker_state.DisarmAccess(id, va >> VESHIRU_PAGEBITS,
                                         ((va + size - 1) >> VESHIRU_PAGEBITS) + 1);
}

} // namespace Core::Memory
