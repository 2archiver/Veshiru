// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename T, typename Id = u32>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    void Invalidate() noexcept {
        // Cleared on the next read: the texture cache invalidates every channel's tables for
        // each image it deletes, dozens of times a frame while its collector runs.
        invalidated = true;
        ++changes;
    }

    std::span<Id> LiveBoundIds() noexcept {
        return std::span{bound_ids}.first(
            (std::min)(bound_ids.size(), size_t{current_limit} + 1));
    }

    template <class MemoryManager>
    [[nodiscard]] std::pair<T, bool> Read(const MemoryManager& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        const u64 page_offset = gpu_addr & Core::Memory::YUZU_PAGEMASK;
        std::pair<T, bool> result;
        // This runs for every texture and sampler of every draw. An entry within one page is
        // contiguous in host memory, so a single page lookup replaces the generic block read.
        const bool one_page =
            (gpu_addr & Core::Memory::VESHIRU_PAGEMASK) + sizeof(T) <= Core::Memory::VESHIRU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    template <class MemoryManager>
    [[nodiscard]] std::pair<const T&, bool> ReadReference(const MemoryManager& gpu_memory,
                                                          u32 index, bool graphics) noexcept {
        static_assert(sizeof(T) == sizeof(std::declval<T>().raw));
        if (!graphics) {
            const auto result = Read(gpu_memory, index);
            return {descriptors[index], result.second};
        }
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        const u64 page_offset = gpu_addr & Core::Memory::VESHIRU_PAGEMASK;
        const u8* source = nullptr;
        if (page_offset + sizeof(T) <= Core::Memory::VESHIRU_PAGESIZE) [[likely]] {
            if constexpr (requires { gpu_memory.GetID(); gpu_memory.HostPointerSerial(); }) {
                const GPUVAddr page_addr = gpu_addr - page_offset;
                auto& entry = page_cache[(page_addr >> Core::Memory::VESHIRU_PAGEBITS) &
                                         (PageCacheSize - 1)];
                const size_t memory_id = gpu_memory.GetID();
                const u64 host_serial = gpu_memory.HostPointerSerial();
                if (entry.page_addr != page_addr || entry.memory_id != memory_id ||
                    entry.host_serial != host_serial || entry.host_ptr == nullptr) {
                    entry = {page_addr, memory_id, host_serial, gpu_memory.GetPointer(page_addr)};
                }
                if (entry.host_ptr != nullptr) {
                    source = entry.host_ptr + page_offset;
                }
            } else {
                source = gpu_memory.GetPointer(gpu_addr);
            }
        }
        T fallback;
        if (source == nullptr) {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(fallback), sizeof(T));
            source = reinterpret_cast<const u8*>(std::addressof(fallback));
        }
        const u64 bit = u64{1} << (index % 64);
        u64& seen = read_descriptors[index / 64];
        static_assert(sizeof(T) == 4 * sizeof(u64));
        std::array<u64, 4> words;
        std::memcpy(&words[0], source, sizeof(u64));
        std::memcpy(&words[1], source + sizeof(u64), sizeof(u64));
        std::memcpy(&words[2], source + 2 * sizeof(u64), sizeof(u64));
        std::memcpy(&words[3], source + 3 * sizeof(u64), sizeof(u64));
        T& cached = descriptors[index];
        const bool changed = (seen & bit) == 0 || words[0] != cached.raw[0] ||
                             words[1] != cached.raw[1] || words[2] != cached.raw[2] ||
                             words[3] != cached.raw[3];
        if (changed) {
            cached.raw[0] = words[0];
            cached.raw[1] = words[1];
            cached.raw[2] = words[2];
            cached.raw[3] = words[3];
            seen |= bit;
        }
        return {cached, changed};
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        ++changes;
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        ClearReadDescriptors();
        invalidated = false;
        //
        descriptors.resize(num_descriptors);
        bound_ids.resize(num_descriptors, Id{0});
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    std::vector<Id> bound_ids;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    bool invalidated{}; ///< read_descriptors is to be cleared before the next read
    /// Bumped by every Invalidate and Refresh: the ids and descriptors read before are only good
    /// while it stays the same.
    u64 changes{};

private:
    static constexpr size_t PageCacheSize = 64;
    struct PageCacheEntry {
        GPUVAddr page_addr{~GPUVAddr{0}};
        size_t memory_id{};
        u64 host_serial{};
        const u8* host_ptr{};
    };
    std::array<PageCacheEntry, PageCacheSize> page_cache{};

    void ClearReadDescriptors() noexcept {
        page_cache = {};
        // Reserved capacity is not addressable. A later limit increase clears its new range.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        std::fill_n(read_descriptors.begin(), words, u64{0});
        const size_t ids = (std::min)(bound_ids.size(), size_t{current_limit} + 1);
        std::fill_n(bound_ids.begin(), ids, Id{0});
    }
};

} // namespace VideoCommon
