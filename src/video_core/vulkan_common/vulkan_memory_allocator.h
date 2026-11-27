// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/sparse_spill_report.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace Vulkan {

    class Device;

/// Hints and requirements for the backing memory type of a commit
    enum class MemoryUsage {
        DeviceLocal, ///< Requests device local host visible buffer, falling back to device local memory.
        Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
        Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
        Stream,      ///< Requests device local host visible buffer, falling back host memory.
    };

/// What an allocation is for, for the spill counters (MemoryAllocator::TrackSpills).
    enum class AllocationKind : u8 {
        Other,
        StorageImage,
        PoolBuffer,
        Shadow,
        Mirror,
        Arena,
        Staging,
        Count,
    };

    [[nodiscard]] const char* AllocationKindName(AllocationKind kind);

/// Allocations that asked for device-local memory (images, DeviceLocal and Stream buffers) and
/// were placed in a memory type without DEVICE_LOCAL: VMA moves on to the next memory type once
/// the device heap is at its budget, and the allocation stays in system memory for its lifetime.
/// Cumulative. Only allocators that opted in (TrackSpills) count.
    struct AllocationSpills {
        static constexpr size_t KINDS = static_cast<size_t>(AllocationKind::Count);
        std::array<std::atomic<u64>, KINDS> count{};
        std::array<std::atomic<u64>, KINDS> bytes{};
        /// Strict mode: device-local allocations made over the budget (the driver pages them) ...
        std::atomic<u64> over_budget_count{};
        std::atomic<u64> over_budget_bytes{};
        /// ... and the ones no device-local memory type took, placed in host memory.
        std::atomic<u64> host_fallback_count{};
        std::atomic<u64> host_fallback_bytes{};
    };

/// Counts `size` bytes of `kind` in `spills` when the allocation wanted device-local memory and
/// got a memory type of `flags` without DEVICE_LOCAL. Returns whether it counted.
    bool CountSpill(AllocationSpills& spills, AllocationKind kind, bool wanted_device_local,
                    VkMemoryPropertyFlags flags, u64 size);

    template<typename F>
    void ForEachDeviceLocalHostVisibleHeap(const Device &device, F &&f) {
        auto memory_props = device.GetPhysical().GetMemoryProperties().memoryProperties;
        for (size_t i = 0; i < memory_props.memoryTypeCount; i++) {
            auto &memory_type = memory_props.memoryTypes[i];
            if ((memory_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (memory_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                f(memory_type.heapIndex, memory_props.memoryHeaps[memory_type.heapIndex]);
            }
        }
    }

/// Ownership handle of a memory commitment (real VMA allocation).
    class MemoryCommit {
    public:
        MemoryCommit() noexcept = default;

        MemoryCommit(VmaAllocator allocator, VmaAllocation allocation,
                     const VmaAllocationInfo &info) noexcept;

        ~MemoryCommit();

        MemoryCommit(const MemoryCommit &) = delete;

        MemoryCommit &operator=(const MemoryCommit &) = delete;

        MemoryCommit(MemoryCommit &&) noexcept;

        MemoryCommit &operator=(MemoryCommit &&) noexcept;

        [[nodiscard]] std::span<u8> Map();

        [[nodiscard]] std::span<const u8> Map() const;

        void Unmap();

        explicit operator bool() const noexcept { return allocation != nullptr; }

        VkDeviceMemory Memory() const noexcept { return memory; }

        VkDeviceSize Offset() const noexcept { return offset; }

        VkDeviceSize Size() const noexcept { return size; }

        VmaAllocation Allocation() const noexcept { return allocation; }

    private:
        void Release();

        VmaAllocator allocator{};   ///< VMA allocator
        VmaAllocation allocation{};  ///< VMA allocation handle
        VkDeviceMemory memory{};      ///< Underlying VkDeviceMemory chosen by VMA
        VkDeviceSize offset{};      ///< Offset of this allocation inside VkDeviceMemory
        VkDeviceSize size{};        ///< Size of the allocation
        void *mapped_ptr{};  ///< Optional persistent mapped pointer
    };

/// Memory allocator container.
/// Allocates and releases memory allocations on demand.
    class MemoryAllocator {
    public:
        /**
         * Construct memory allocator
         *
         * @param device_  Device to allocate from
         *
         * @throw vk::Exception on failure
         */
        explicit MemoryAllocator(const Device &device_);

        /// Allocates from `allocator_` instead of the device's shared one (which is externally
        /// synchronized). Used by VTX-HLE's replay, which must not race the render thread.
        explicit MemoryAllocator(const Device &device_, VmaAllocator allocator_);

        ~MemoryAllocator();

        MemoryAllocator &operator=(const MemoryAllocator &) = delete;

        MemoryAllocator(const MemoryAllocator &) = delete;

        /// Counts this allocator's spills into `spills_` (null: none). `device_strict_`: images
        /// and DeviceLocal buffers require a DEVICE_LOCAL memory type, over the budget too (the
        /// driver pages), and take host memory only when no device-local type can hold them.
        /// Set before the allocator is used.
        void TrackSpills(AllocationSpills* spills_, bool device_strict_) noexcept {
            spills = spills_;
            device_strict = device_strict_;
        }

        /// `dedicated`: own device memory, returned to the driver on destruction (short-lived
        /// images would otherwise leave holes in the shared blocks).
        vk::Image CreateImage(const VkImageCreateInfo &ci, bool dedicated = false,
                              bool transient = false,
                              AllocationKind kind = AllocationKind::Other) const;

        vk::Image CreateImage(const VkImageCreateInfo &ci, bool dedicated,
                              AllocationKind kind, bool transient = false) const {
            return CreateImage(ci, dedicated, transient, kind);
        }

        /// CreateImage, but a null image instead of an exception when there is no memory for it.
        vk::Image TryCreateImage(const VkImageCreateInfo &ci, bool dedicated = false,
                                 bool transient = false,
                                 AllocationKind kind = AllocationKind::Other) const;

        /// A sparse resident image with nothing bound (reads return zero, writes are dropped), or
        /// a null image when the device can't make one.
        /// aliased: its sparse blocks may share memory (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT).
        vk::Image CreateUnbackedImage(const VkImageCreateInfo &ci, bool aliased = false) const;

        /// Render target and storage images in their own VMA pool; false when it doesn't apply
        /// or failed.
        bool CreateImageInSeparatePool(const VkImageCreateInfo& ci,
                                       const VmaAllocationCreateInfo& alloc_ci, VkImage& handle,
                                       VmaAllocation& allocation,
                                       VmaAllocationInfo& alloc_info) const;

        /// Device memory to bind to part of a sparse resource. Null when there is none left.
        /// Device-local buffers in their own VMA pool; false when it doesn't apply or failed.
        bool CreateBufferInSeparatePool(const VkBufferCreateInfo& ci, MemoryUsage usage,
                                        const VmaAllocationCreateInfo& alloc_ci, VkBuffer& handle,
                                        VmaAllocation& allocation,
                                        VmaAllocationInfo& alloc_info) const;

        VmaAllocation AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                           VkDeviceMemory &memory, VkDeviceSize &offset) const;

        /// Counts a sparse tile that device-local memory had no room for, and logs a warning
        /// with the counters at most once per SparseSpillReport::IntervalMs.
        void NoteSparseSpill(VkDeviceSize size, bool placed) const;

        void FreeSparseMemory(VmaAllocation allocation) const;

        /// Counts the images that could not be made at all (no memory left anywhere): the
        /// texture cache collects aggressively when it grows.
        u32 FailedImages() const noexcept {
            return failed_images.load(std::memory_order_relaxed);
        }

        vk::Buffer CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const;

        vk::Buffer CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                AllocationKind kind) const;

        /// CreateBuffer, but a null buffer instead of an exception when there is no memory for it.
        vk::Buffer TryCreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const;

        vk::Buffer CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                VkDeviceSize min_alignment) const;

        /**
         * Commits a memory with the specified requirements.
         *
         * @param requirements Requirements returned from a Vulkan call.
         * @param usage        Indicates how the memory will be used.
         *
         * @returns A memory commit.
         */
        MemoryCommit Commit(const VkMemoryRequirements &requirements, MemoryUsage usage);

        /// Commits memory required by the buffer and binds it (for buffers created outside VMA).
        MemoryCommit Commit(const vk::Buffer &buffer, MemoryUsage usage);

    private:
        static bool IsAutoUsage(VmaMemoryUsage u) noexcept {
            switch (u) {
                case VMA_MEMORY_USAGE_AUTO:
                case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
                case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
                    return true;
                default:
                    return false;
            }
        }

        /// CreateImage's fallbacks; the image is left null when the result is an error.
        /// may_fail: the caller handles VK_ERROR_OUT_OF_DEVICE_MEMORY (see UsesUnbackedFallback).
        VkResult AllocateImage(const VkImageCreateInfo &ci, bool dedicated, bool transient,
                               bool may_fail, AllocationKind kind, vk::Image &image) const;

        /// CreateBuffer's fallbacks; the buffer is left null when the result is an error.
        VkResult AllocateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                AllocationKind kind, vk::Buffer &buffer) const;

        const Device &device;                              ///< Device handle.
        VmaAllocator allocator;                           ///< VMA allocator.
        const VkPhysicalDeviceMemoryProperties properties; ///< Physical device memory properties.
        VkDeviceSize buffer_image_granularity;            ///< Adjacent buffer/image granularity
        u32 valid_memory_types{~0u};
        mutable std::atomic<u32> failed_images{0};
        AllocationSpills* spills{};
        bool device_strict{};
        /// Sparse tiles that device-local memory had no room for (AllocateSparseMemory).
        mutable std::mutex sparse_spill_mutex;
        mutable SparseSpillReport sparse_spill_report;
        mutable std::mutex buffer_type_mutex;
        /// Memory type of device-local buffers by usage and flags (CreateBufferInSeparatePool).
        mutable std::unordered_map<u64, u32> buffer_memory_types;
    };

    /// Logs VMA's blocks and allocations per memory type, and each heap's budget.
    void LogMemoryStatistics(const Device& device);

    /// One line each for the process's memory, what DXGI counts for the GPU (Windows), and VMA's
    /// heaps and memory types against what the driver reports (EDEN_NX2_MEM_TRACE).
    void LogMemoryTrace(const Device& device);

} // namespace Vulkan
