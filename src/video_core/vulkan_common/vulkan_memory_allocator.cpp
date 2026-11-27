// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <dxgi1_4.h>
#endif

#include <cstring>
#include <iterator>
#include <string>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include <ranges>
#include "video_core/vulkan_common/unbacked_fallback.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/unbacked_image_policy.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"

namespace Vulkan {
namespace {

// Helpers translating MemoryUsage to flags/usage

    [[maybe_unused]] VkMemoryPropertyFlags MemoryUsagePropertyFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            case MemoryUsage::Upload:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            case MemoryUsage::Download:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            case MemoryUsage::Stream:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    [[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
        if (usage == MemoryUsage::Download) {
            return VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                    : VkMemoryPropertyFlagBits{};
    }

    [[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::Upload:
            case MemoryUsage::Stream:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            case MemoryUsage::Download:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            case MemoryUsage::DeviceLocal:
                return {};
        }
        return {};
    }

    [[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
            case MemoryUsage::Stream:
                return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            case MemoryUsage::Upload:
            case MemoryUsage::Download:
                return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        }
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }

    /// Strict mode's over-budget allocations: the first ones and then every 256th are logged.
    [[nodiscard]] bool LogStrictFallback(std::atomic<u64>& counter) {
        const u64 n = counter.load(std::memory_order_relaxed);
        return n <= 32 || (n & 0xFF) == 0;
    }
} // namespace

const char* AllocationKindName(AllocationKind kind) {
    switch (kind) {
    case AllocationKind::Other:
        return "other";
    case AllocationKind::StorageImage:
        return "storage_images";
    case AllocationKind::PoolBuffer:
        return "pool_buffers";
    case AllocationKind::Shadow:
        return "shadows";
    case AllocationKind::Mirror:
        return "mirrors";
    case AllocationKind::Arena:
        return "arenas";
    case AllocationKind::Staging:
        return "staging";
    case AllocationKind::Count:
        break;
    }
    return "?";
}

bool CountSpill(AllocationSpills& spills, AllocationKind kind, bool wanted_device_local,
                VkMemoryPropertyFlags flags, u64 size) {
    const size_t index = static_cast<size_t>(kind);
    if (!wanted_device_local || (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 ||
        index >= AllocationSpills::KINDS) {
        return false;
    }
    spills.count[index].fetch_add(1, std::memory_order_relaxed);
    spills.bytes[index].fetch_add(size, std::memory_order_relaxed);
    return true;
}

//MemoryCommit is now VMA-backed
MemoryCommit::MemoryCommit(VmaAllocator alloc, VmaAllocation a,
                            const VmaAllocationInfo &info) noexcept
        : allocator{alloc}, allocation{a}, memory{info.deviceMemory},
            offset{info.offset}, size{info.size}, mapped_ptr{info.pMappedData} {
    // Log GPU memory allocation
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(memory),
            static_cast<u64>(size),
            0  // Memory property flags (not easily available from VMA)
        );
    }
}

MemoryCommit::~MemoryCommit() { Release(); }

MemoryCommit::MemoryCommit(MemoryCommit &&rhs) noexcept
        : allocator{std::exchange(rhs.allocator, nullptr)},
            allocation{std::exchange(rhs.allocation, nullptr)},
            memory{std::exchange(rhs.memory, VK_NULL_HANDLE)},
            offset{std::exchange(rhs.offset, 0)},
            size{std::exchange(rhs.size, 0)},
            mapped_ptr{std::exchange(rhs.mapped_ptr, nullptr)} {}

MemoryCommit &MemoryCommit::operator=(MemoryCommit &&rhs) noexcept {
    if (this != &rhs) {
        Release();
        allocator = std::exchange(rhs.allocator, nullptr);
        allocation = std::exchange(rhs.allocation, nullptr);
        memory = std::exchange(rhs.memory, VK_NULL_HANDLE);
        offset = std::exchange(rhs.offset, 0);
        size = std::exchange(rhs.size, 0);
        mapped_ptr = std::exchange(rhs.mapped_ptr, nullptr);
    }
    return *this;
}

std::span<u8> MemoryCommit::Map()
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        if (vmaMapMemory(allocator, allocation, &mapped_ptr) != VK_SUCCESS) return {};
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<u8>{static_cast<u8 *>(mapped_ptr), n};
}

std::span<const u8> MemoryCommit::Map() const
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        void *p = nullptr;
        if (vmaMapMemory(allocator, allocation, &p) != VK_SUCCESS) return {};
        const_cast<MemoryCommit *>(this)->mapped_ptr = p;
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<const u8>{static_cast<const u8 *>(mapped_ptr), n};
}

void MemoryCommit::Unmap()
{
    if (allocation && mapped_ptr) {
        vmaUnmapMemory(allocator, allocation);
        mapped_ptr = nullptr;
    }
}

void MemoryCommit::Release() {
    if (allocation && allocator) {
        // Log GPU memory deallocation
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue() &&
            memory != VK_NULL_HANDLE) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryDeallocation(
                reinterpret_cast<uintptr_t>(memory)
            );
        }

        if (mapped_ptr) {
            vmaUnmapMemory(allocator, allocation);
            mapped_ptr = nullptr;
        }
        vmaFreeMemory(allocator, allocation);
    }
    allocation = nullptr;
    allocator = nullptr;
    memory = VK_NULL_HANDLE;
    offset = 0;
    size = 0;
}

MemoryAllocator::MemoryAllocator(const Device &device_)
        : MemoryAllocator{device_, device_.GetAllocator()} {}

MemoryAllocator::MemoryAllocator(const Device &device_, VmaAllocator allocator_)
        : device{device_}, allocator{allocator_},
            properties{device_.GetPhysical().GetMemoryProperties().memoryProperties},
            buffer_image_granularity{
                    device_.GetPhysical().GetProperties().limits.bufferImageGranularity} {

    // Preserve the previous "RenderDoc small heap" trimming behavior that we had in original vma minus the heap bug
    if (device.HasDebuggingToolAttached())
    {
        using namespace Common::Literals;
        ForEachDeviceLocalHostVisibleHeap(device, [this](size_t heap_idx, VkMemoryHeap &heap) {
            if (heap.size <= 256_MiB) {
                for (u32 t = 0; t < properties.memoryTypeCount; ++t) {
                    if (properties.memoryTypes[t].heapIndex == heap_idx) {
                        valid_memory_types &= ~(1u << t);
                    }
                }
            }
        });
    }
}

MemoryAllocator::~MemoryAllocator() {
    // The warning is only logged when a spill happens, so the last interval's counters would
    // otherwise never be printed.
    SparseSpillReport::Counters total;
    {
        std::scoped_lock lock{sparse_spill_mutex};
        total = sparse_spill_report.Total();
    }
    if (total.tiles != 0 || total.failed != 0) {
        LOG_INFO(Render_Vulkan,
                 "Sparse texture tiles outside device-local memory this session: {} tiles "
                 "({:.1f} MiB), {} that no heap had room for",
                 total.tiles, static_cast<double>(total.bytes) / (1024.0 * 1024.0),
                 total.failed);
    }
}

vk::Image MemoryAllocator::CreateImage(const VkImageCreateInfo &ci, bool dedicated,
                                       bool transient, AllocationKind kind) const
{
    vk::Image image;
    vk::Check(AllocateImage(ci, dedicated, transient, false, kind, image));
    return image;
}

vk::Image MemoryAllocator::TryCreateImage(const VkImageCreateInfo &ci, bool dedicated,
                                          bool transient, AllocationKind kind) const {
    vk::Image image;
    const VkResult result = AllocateImage(ci, dedicated, transient, true, kind, image);
    if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
        vk::Check(result);
    }
    return image;
}

VkResult MemoryAllocator::AllocateImage(const VkImageCreateInfo &ci, bool dedicated,
                                        bool transient, bool may_fail, AllocationKind kind,
                                        vk::Image &image) const {
    static_cast<void>(transient);
    const VmaAllocationCreateInfo alloc_ci = {
            // Best fit: images freed by the thousand leave holes in the blocks, which new ones
            // should fill before VMA takes another block.
            .flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT |
                                                           VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT) |
                     (dedicated ? VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT}
                               : VmaAllocationCreateFlags{0}),
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            // Over budget, VMA would quietly place the image in host memory, where it stays and
            // the GPU samples it over PCIe. Fail instead so the pressure path below runs.
            .requiredFlags = device_strict
                                 ? VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}
                                 : (device.IsIntegrated()
                                        ? VkMemoryPropertyFlags{0}
                                        : VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}),
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 0.f,
    };

    VkImage handle{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    const u64 texels = u64{ci.extent.width} * ci.extent.height * ci.extent.depth * ci.arrayLayers;
    u64 bytes = 0;
    if (texels > MaxBackedImageTexels) {
        // Its memory size decides: an image the driver won't create is too large to back.
        const auto& dld = device.GetDispatchLoader();
        VkImage probe{};
        bytes = ~u64{0};
        if (dld.vkCreateImage(*device.GetLogical(), &ci, nullptr, &probe) == VK_SUCCESS) {
            bytes = device.GetLogical().GetImageMemoryRequirements(probe).size;
            dld.vkDestroyImage(*device.GetLogical(), probe, nullptr);
        }
    }
    const bool huge = IsTooLargeToBack(texels, bytes);
    // A small image is not evidence of VRAM pressure. VMA's budget is its own estimate of what
    // fits, and when it is momentarily spent a title asking for a 64x64, 7-level, format-37 shadow
    // map (21 KiB, in the supplied log) was refused and logged as "over the memory budget" even
    // though the card had room. Images this small fit whatever the estimate says, so they skip the
    // budget check and fail only when the allocation really fails.
    constexpr u64 BudgetExemptBytes = 4 * 1024 * 1024; // 4 MiB
    VmaAllocationCreateInfo alloc_ci_used = alloc_ci;
    // A bound on the image's size without creating it: texels times the largest texel.
    const u64 estimate_bytes = texels <= ~u64{0} / 16 ? texels * 16 : ~u64{0};
    if (!device_strict && estimate_bytes <= BudgetExemptBytes) {
        alloc_ci_used.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    }

    VkResult result =
        huge ? VK_ERROR_OUT_OF_DEVICE_MEMORY
        : !dedicated &&
                CreateImageInSeparatePool(ci, alloc_ci_used, handle, allocation, alloc_info)
            ? VK_SUCCESS
            : vmaCreateImage(allocator, &ci, &alloc_ci_used, &handle, &allocation, &alloc_info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge && device.HasPendingMemoryFrees()) {
        // Background vkFreeMemory calls may still hold VRAM the caches already evicted this frame;
        // drain them before declaring budget pressure or spilling an image into host memory.
        device.FlushPendingMemoryFrees();
        result = !dedicated && CreateImageInSeparatePool(ci, alloc_ci_used, handle, allocation,
                                                         alloc_info)
                     ? VK_SUCCESS
                     : vmaCreateImage(allocator, &ci, &alloc_ci_used, &handle, &allocation,
                                      &alloc_info);
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        if (device_strict) {
            VmaAllocationCreateInfo relaxed = alloc_ci;
            relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            if (result == VK_SUCCESS && spills != nullptr) {
                spills->over_budget_count.fetch_add(1, std::memory_order_relaxed);
                spills->over_budget_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
                if (LogStrictFallback(spills->over_budget_count)) {
                    LOG_WARNING(Render_Vulkan,
                                "[vma strict] {} image {}x{}x{} ({} MiB) placed in device-local "
                                "memory over the budget (#{})",
                                AllocationKindName(kind), ci.extent.width, ci.extent.height,
                                ci.extent.depth, alloc_info.size >> 20,
                                spills->over_budget_count.load(std::memory_order_relaxed));
                }
            }
            if (result == VK_ERROR_FEATURE_NOT_PRESENT) {
                // Strict mode's required device-local flag matched no memory type the image accepts:
                // take the host fallback below instead of failing.
                result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }
            if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
                VmaAllocationCreateInfo host = relaxed;
                host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                host.requiredFlags = 0;
                host.preferredFlags = 0;
                result = vmaCreateImage(allocator, &ci, &host, &handle, &allocation, &alloc_info);
                if (result == VK_SUCCESS && spills != nullptr) {
                    spills->host_fallback_count.fetch_add(1, std::memory_order_relaxed);
                    spills->host_fallback_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
                    LOG_WARNING(Render_Vulkan,
                                "[vma strict] {} image {}x{}x{} ({} MiB): no device-local memory "
                                "left, placed in host memory (#{})",
                                AllocationKindName(kind), ci.extent.width, ci.extent.height,
                                ci.extent.depth, alloc_info.size >> 20,
                                spills->host_fallback_count.load(std::memory_order_relaxed));
                }
            }
        } else {
            // Over the device memory budget (Switch 2 titles fill 8 GB cards): let the driver page
            // it out rather than failing the draw and aborting.
            //
            // This used to log the first 32 per process and then stop, and all three 2026-11-22
            // Ocarina logs stop at exactly 32: a session's real count was unknowable, and the same
            // five shapes filled every line. One line per distinct shape now (up to 64), plus the
            // running total every 256 allocations, so a capture can say how much went over and how
            // often.
            static std::mutex over_budget_mutex;
            static std::unordered_set<u64> over_budget_shapes;
            static u64 over_budget_total{};
            const u64 shape_key =
                (u64{ci.extent.width} & 0xFFFF) | ((u64{ci.extent.height} & 0xFFFF) << 16) |
                ((u64{ci.extent.depth} & 0x7) << 32) | ((u64{ci.mipLevels} & 0x1F) << 35) |
                ((u64{ci.arrayLayers} & 0xFFF) << 40) |
                ((u64{static_cast<u32>(ci.format)} & 0xFF) << 52);
            {
                std::scoped_lock lock{over_budget_mutex};
                ++over_budget_total;
                const bool first_of_shape =
                    over_budget_shapes.size() < 64 && over_budget_shapes.insert(shape_key).second;
                if (first_of_shape) {
                    LOG_WARNING(Render_Vulkan,
                                "Image {}x{}x{} ({} levels, {} layers, format {}) is over the "
                                "memory budget; allocating it anyway",
                                ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                                ci.arrayLayers, static_cast<u32>(ci.format));
                } else if (over_budget_total % 256 == 0) {
                    LOG_INFO(Render_Vulkan,
                             "{} images have gone over the memory budget so far, in {} shapes, and "
                             "all of them were allocated anyway",
                             over_budget_total, over_budget_shapes.size());
                }
            }
            // Short-lived dedicated scratch images (e.g. ASTC->BCn RGBA8 decode intermediates)
            // return their memory immediately after use; paging one of those out is not worth an
            // eviction cascade either.
            VmaAllocationCreateInfo relaxed = alloc_ci;
            relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && relaxed.requiredFlags != 0) {
                // VRAM is really full: host memory is slow, but a missing image crashes.
                relaxed.requiredFlags = 0;
                result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            }
        }
    } else if (result == VK_ERROR_FEATURE_NOT_PRESENT && device_strict && !huge) {
        VmaAllocationCreateInfo host = alloc_ci;
        host.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        host.requiredFlags = 0;
        host.preferredFlags = 0;
        result = vmaCreateImage(allocator, &ci, &host, &handle, &allocation, &alloc_info);
        if (result == VK_SUCCESS && spills != nullptr) {
            spills->host_fallback_count.fetch_add(1, std::memory_order_relaxed);
            spills->host_fallback_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan,
                        "[vma strict] {} image {}x{}x{} ({} MiB): no device-local memory left, "
                        "placed in host memory (#{})",
                        AllocationKindName(kind), ci.extent.width, ci.extent.height,
                        ci.extent.depth, alloc_info.size >> 20,
                        spills->host_fallback_count.load(std::memory_order_relaxed));
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && ci.imageType != VK_IMAGE_TYPE_1D &&
        UsesUnbackedFallback(texels, bytes, may_fail)) {
        // Too big to back at all, e.g. title 1's sparse virtual texture arrays (8192x8192,
        // 68 layers, ~23 GB): on the console only resident tiles have memory. Use an image with
        // nothing bound, so views and copies still work, instead of aborting. A smaller image
        // gets one only for a caller that would throw otherwise.
        if (vk::Image sparse = CreateUnbackedImage(ci)) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) does not fit; using an "
                        "unbacked sparse image",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
            image = std::move(sparse);
            return VK_SUCCESS;
        }
    }
    if (result != VK_SUCCESS) {
        failed_images.fetch_add(1, std::memory_order_relaxed);
        static u32 failed_logged{};
        if (failed_logged == 0) {
            // What ran out: the device's heaps or the host's commit. Before the error line,
            // which is the one that reaches the log file when the session ends here.
            LogMemoryTrace(device);
        }
        if (failed_logged++ < 32) {
            LOG_ERROR(Render_Vulkan, "Image {}x{}x{} ({} levels, {} layers, format {}) failed: {}",
                      ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                      ci.arrayLayers, static_cast<u32>(ci.format), static_cast<int>(result));
        }
        return result;
    }

    if (spills != nullptr) {
        VkMemoryPropertyFlags property_flags{};
        vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
        CountSpill(*spills, kind, true, property_flags, alloc_info.size);
    }

    // Log GPU memory allocation for images
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    image = vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                      device.GetDispatchLoader());
    return VK_SUCCESS;
}

vk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci, bool aliased) const {
    if (ci.imageType == VK_IMAGE_TYPE_1D) {
        return vk::Image{};
    }
    VkImageCreateInfo sparse_ci = ci;
    sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (aliased) {
        sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
    }
    VkImage sparse{};
    if (device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr,
                                                 &sparse) != VK_SUCCESS) {
        return vk::Image{};
    }
    return vk::Image(sparse, ci.usage, *device.GetLogical(), allocator, VK_NULL_HANDLE,
                     device.GetDispatchLoader());
}

VmaAllocation MemoryAllocator::AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                                   VkDeviceMemory &memory,
                                                   VkDeviceSize &offset) const {
    const VmaAllocationCreateInfo alloc_ci{
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .preferredFlags = 0,
        .memoryTypeBits = requirements.memoryTypeBits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    // In a pool of their own (see Device::SeparatePool); anything over half a block, or a pool
    // allocation that fails, takes the default blocks.
    u32 memory_type{};
    if (requirements.size <= Device::SeparatePoolBlockSize / 2 &&
        vmaFindMemoryTypeIndex(allocator, requirements.memoryTypeBits, &alloc_ci, &memory_type) ==
            VK_SUCCESS) {
        if (const VmaPool pool =
                device.GetSeparatePool(Device::SeparatePool::SparsePages, memory_type,
                                       allocator)) {
            VmaAllocationCreateInfo pool_ci = alloc_ci;
            pool_ci.pool = pool;
            if (vmaAllocateMemory(allocator, &requirements, &pool_ci, &allocation, &info) ==
                VK_SUCCESS) {
                memory = info.deviceMemory;
                offset = info.offset;
                return allocation;
            }
        }
    }
    VkResult result = vmaAllocateMemory(allocator, &requirements, &alloc_ci, &allocation, &info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        // MKW can exhaust the local heap mid-atlas on 4 GiB GPUs. Keep the tiles already
        // allocated in this batch and back the remaining ones on a compatible non-local heap.
        VmaAllocationCreateInfo spill = alloc_ci;
        spill.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        spill.requiredFlags = 0;
        constexpr VkMemoryPropertyFlags excluded = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT;
        for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
            if (properties.memoryTypes[i].propertyFlags & excluded) {
                spill.memoryTypeBits &= ~(1U << i);
            }
        }
        // VMA interprets a zero mask as unrestricted.
        if (spill.memoryTypeBits != 0) {
            result = vmaAllocateMemory(allocator, &requirements, &spill, &allocation, &info);
        }
        NoteSparseSpill(requirements.size, result == VK_SUCCESS);
    }
    if (result != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    memory = info.deviceMemory;
    offset = info.offset;
    return allocation;
}

void MemoryAllocator::NoteSparseSpill(VkDeviceSize size, bool placed) const {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::optional<SparseSpillReport::Report> report;
    {
        std::scoped_lock lock{sparse_spill_mutex};
        report = sparse_spill_report.Record(now_ms, size, placed);
    }
    if (!report) {
        return;
    }
    // Counters only. What a tile outside video memory costs in frame time has not been measured.
    constexpr double MiB = 1024.0 * 1024.0;
    LOG_WARNING(Render_Vulkan,
                "Video memory is full: sparse texture tiles are going to a non-device-local "
                "heap. Since the last report: {} tiles ({:.1f} MiB) placed there, {} that no "
                "heap had room for. This session: {} tiles ({:.1f} MiB), {} not placed. Reported "
                "at most every {} s.",
                report->interval.tiles, static_cast<double>(report->interval.bytes) / MiB,
                report->interval.failed, report->total.tiles,
                static_cast<double>(report->total.bytes) / MiB, report->total.failed,
                SparseSpillReport::IntervalMs / 1000);
}

void MemoryAllocator::FreeSparseMemory(VmaAllocation allocation) const {
    if (allocation != VK_NULL_HANDLE) {
        vmaFreeMemory(allocator, allocation);
    }
}

bool MemoryAllocator::CreateImageInSeparatePool(const VkImageCreateInfo& ci,
                                                const VmaAllocationCreateInfo& alloc_ci,
                                                VkImage& handle, VmaAllocation& allocation,
                                                VmaAllocationInfo& alloc_info) const {
    // Render targets and storage images up to half a block: in a pool of their own (see
    // Device::SeparatePool). Larger ones, and any that fails here, take the default path.
    constexpr VkImageUsageFlags target_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                               VK_IMAGE_USAGE_STORAGE_BIT;
    if ((ci.usage & target_usage) == 0) {
        return false;
    }
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    VkImage image{};
    if (dld.vkCreateImage(logical, &ci, nullptr, &image) != VK_SUCCESS) {
        return false;
    }
    const VkMemoryRequirements requirements = device.GetLogical().GetImageMemoryRequirements(image);
    u32 memory_type{};
    if (requirements.size <= Device::SeparatePoolBlockSize / 2) {
        if (vmaFindMemoryTypeIndex(allocator, requirements.memoryTypeBits, &alloc_ci,
                                   &memory_type) == VK_SUCCESS) {
            if (const VmaPool pool =
                    device.GetSeparatePool(Device::SeparatePool::RenderTargets, memory_type,
                                           allocator)) {
                VmaAllocationCreateInfo pool_ci = alloc_ci;
                pool_ci.pool = pool;
                if (vmaAllocateMemoryForImage(allocator, image, &pool_ci, &allocation,
                                              &alloc_info) == VK_SUCCESS) {
                    if (vmaBindImageMemory(allocator, allocation, image) == VK_SUCCESS) {
                        handle = image;
                        return true;
                    }
                    vmaFreeMemory(allocator, allocation);
                    allocation = VK_NULL_HANDLE;
                }
            }
        }
    } else if (vmaAllocateMemoryForImage(allocator, image, &alloc_ci, &allocation, &alloc_info) ==
               VK_SUCCESS) {
        // Reuse the VkImage already created above instead of destroying it and creating an
        // identical second VkImage in vmaCreateImage for every > 8 MiB target/storage image.
        if (vmaBindImageMemory(allocator, allocation, image) == VK_SUCCESS) {
            handle = image;
            return true;
        }
        vmaFreeMemory(allocator, allocation);
        allocation = VK_NULL_HANDLE;
    }
    dld.vkDestroyImage(logical, image, nullptr);
    return false;
}

bool MemoryAllocator::CreateBufferInSeparatePool(const VkBufferCreateInfo& ci, MemoryUsage usage,
                                                 const VmaAllocationCreateInfo& alloc_ci,
                                                 VkBuffer& handle, VmaAllocation& allocation,
                                                 VmaAllocationInfo& alloc_info) const {
    // The buffer cache's buffers, churned by its collector: in a pool of their own (see
    // Device::SeparatePool). Anything over half a block takes the default blocks (dedicated).
    if (usage != MemoryUsage::DeviceLocal || ci.size > Device::DeviceBufferBlockSize / 2) {
        return false;
    }
    u32 memory_type{};
    {
        std::scoped_lock lk{buffer_type_mutex};
        const auto key = (u64{ci.usage} << 32) | ci.flags;
        if (const auto it = buffer_memory_types.find(key); it != buffer_memory_types.end()) {
            memory_type = it->second;
        } else if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &ci, &alloc_ci, &memory_type) ==
                   VK_SUCCESS) {
            buffer_memory_types.emplace(key, memory_type);
        } else {
            return false;
        }
    }
    const VmaPool pool =
        device.GetSeparatePool(Device::SeparatePool::DeviceBuffers, memory_type, allocator);
    if (pool == VK_NULL_HANDLE) {
        return false;
    }
    VmaAllocationCreateInfo pool_ci = alloc_ci;
    pool_ci.pool = pool;
    pool_ci.memoryTypeBits = 0;
    return vmaCreateBuffer(allocator, &ci, &pool_ci, &handle, &allocation, &alloc_info) ==
           VK_SUCCESS;
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
    return CreateBuffer(ci, usage, AllocationKind::Other);
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         AllocationKind kind) const {
    vk::Buffer buffer;
    vk::Check(AllocateBuffer(ci, usage, kind, buffer));
    return buffer;
}

vk::Buffer MemoryAllocator::TryCreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
    vk::Buffer buffer;
    const VkResult result = AllocateBuffer(ci, usage, AllocationKind::Other, buffer);
    if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
        vk::Check(result);
    }
    return buffer;
}

VkResult MemoryAllocator::AllocateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         AllocationKind kind, vk::Buffer &buffer) const {
    // MESA will do memcpy() if not marked as host cached, so just force mark it for most buffers
    auto const anv_flags = (usage == MemoryUsage::Stream
        && device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
        ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    // Strict: a DeviceLocal buffer gets no silent fallback to a host memory type within the
    // budget (Stream buffers fall back to host memory by design).
    const bool strict = device_strict && usage == MemoryUsage::DeviceLocal;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = strict ? VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}
                                : VkMemoryPropertyFlags{0},
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = usage == MemoryUsage::Stream ? 0u : valid_memory_types,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    if (!CreateBufferInSeparatePool(ci, usage, alloc_ci, handle, allocation, alloc_info)) {
        VkResult result =
            vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        if (strict) {
            if (result != VK_SUCCESS) {
                VmaAllocationCreateInfo relaxed = alloc_ci;
                relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
                result = vmaCreateBuffer(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
                if (result == VK_SUCCESS && spills != nullptr) {
                    spills->over_budget_count.fetch_add(1, std::memory_order_relaxed);
                    spills->over_budget_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
                    if (LogStrictFallback(spills->over_budget_count)) {
                        LOG_WARNING(Render_Vulkan,
                                    "[vma strict] {} buffer ({} MiB) placed in device-local memory "
                                    "over the budget (#{})",
                                    AllocationKindName(kind), alloc_info.size >> 20,
                                    spills->over_budget_count.load(std::memory_order_relaxed));
                    }
                }
                if (result != VK_SUCCESS) {
                    VmaAllocationCreateInfo host = relaxed;
                    host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                    host.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    host.requiredFlags = 0;
                    result = vmaCreateBuffer(allocator, &ci, &host, &handle, &allocation, &alloc_info);
                    if (result == VK_SUCCESS && spills != nullptr) {
                        spills->host_fallback_count.fetch_add(1, std::memory_order_relaxed);
                        spills->host_fallback_bytes.fetch_add(alloc_info.size,
                                                              std::memory_order_relaxed);
                        LOG_WARNING(Render_Vulkan,
                                    "[vma strict] {} buffer ({} MiB): no device-local memory left, "
                                    "placed in host memory (#{})",
                                    AllocationKindName(kind), alloc_info.size >> 20,
                                    spills->host_fallback_count.load(std::memory_order_relaxed));
                    }
                }
            }
        } else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                   result == VK_ERROR_OUT_OF_HOST_MEMORY) {
            // Over the device memory budget, as images are in CreateImage: buffers have no
            // unbacked fallback, and the vk::Exception a failed staging or buffer cache
            // allocation throws from the GPU thread aborts the process. Phones share RAM with
            // the GPU, so the budget is often lower than the memory that is really free.
            static u32 logged{};
            if (logged++ < 32) {
                LOG_WARNING(Render_Vulkan,
                            "Buffer of {} bytes (usage {}) is over the memory budget; "
                            "allocating it anyway",
                            ci.size, static_cast<u32>(usage));
            }
            VmaAllocationCreateInfo relaxed = alloc_ci;
            relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateBuffer(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
                VmaAllocationCreateInfo host = relaxed;
                host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                host.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                host.requiredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                result = vmaCreateBuffer(allocator, &ci, &host, &handle, &allocation, &alloc_info);
                if (result == VK_SUCCESS && logged <= 32) {
                    LOG_WARNING(Render_Vulkan,
                                "Buffer of {} bytes (usage {}) is out of device memory; placed in "
                                "host memory",
                                ci.size, static_cast<u32>(usage));
                }
            }
        }
        if (result != VK_SUCCESS) {
            return result;
        }
    }
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
    if (spills != nullptr) {
        CountSpill(*spills, kind,
                   usage == MemoryUsage::DeviceLocal || usage == MemoryUsage::Stream,
                   property_flags, alloc_info.size);
    }

    // Log GPU memory allocation for buffers
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            property_flags
        );
    }

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    const std::span<u8> mapped_data = data ? std::span<u8>{data, ci.size} : std::span<u8>{};
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    buffer = vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data,
                        is_coherent, location, device.GetDispatchLoader());
    return VK_SUCCESS;
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         VkDeviceSize min_alignment) const {
    if (min_alignment <= 1) {
        return CreateBuffer(ci, usage);
    }
    VkMemoryPropertyFlags anv_flags = 0;
    if (usage == MemoryUsage::Stream &&
        device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
        anv_flags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    u32 memory_type_bits = valid_memory_types;
    if (usage == MemoryUsage::Stream) {
        memory_type_bits = 0u;
    }
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = memory_type_bits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    VkResult result = vmaCreateBufferWithAlignment(allocator, &ci, &alloc_ci, min_alignment,
                                                   &handle, &allocation, &alloc_info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
        VmaAllocationCreateInfo relaxed = alloc_ci;
        relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateBufferWithAlignment(allocator, &ci, &relaxed, min_alignment, &handle,
                                              &allocation, &alloc_info);
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            VmaAllocationCreateInfo host = relaxed;
            host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            host.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            host.requiredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            result = vmaCreateBufferWithAlignment(allocator, &ci, &host, min_alignment, &handle,
                                                  &allocation, &alloc_info);
        }
    }
    vk::Check(result);
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    std::span<u8> mapped_data{};
    if (data) {
        mapped_data = std::span<u8>{data, ci.size};
    }
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data, is_coherent,
                      location, device.GetDispatchLoader());
}

MemoryCommit MemoryAllocator::Commit(const VkMemoryRequirements &reqs, MemoryUsage usage)
{
    const auto vma_usage = MemoryUsageVma(usage);
    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
    ci.usage = vma_usage;
    ci.memoryTypeBits = reqs.memoryTypeBits & valid_memory_types;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);

    VmaAllocation a{};
    VmaAllocationInfo info{};

    VkResult res = vmaAllocateMemory(allocator, &reqs, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        // Relax 1: drop budget constraint
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemory(allocator, &reqs, &ci2, &a, &info);

        // Relax 2: if we preferred DEVICE_LOCAL, drop that preference
        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemory(allocator, &reqs, &ci3, &a, &info);
        }
    }

    vk::Check(res);
    return MemoryCommit(allocator, a, info);
}

MemoryCommit MemoryAllocator::Commit(const vk::Buffer &buffer, MemoryUsage usage) {
    // Allocate memory appropriate for this buffer automatically
    const auto vma_usage = MemoryUsageVma(usage);

    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
    ci.usage = vma_usage;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);
    ci.pool = VK_NULL_HANDLE;
    ci.pUserData = nullptr;
    ci.priority = 0.0f;

    const VkBuffer raw = *buffer;

    VmaAllocation a{};
    VmaAllocationInfo info{};

    // Let VMA infer memory requirements from the buffer
    VkResult res = vmaAllocateMemoryForBuffer(allocator, raw, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemoryForBuffer(allocator, raw, &ci2, &a, &info);

        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemoryForBuffer(allocator, raw, &ci3, &a, &info);
        }
    }

    vk::Check(res);
    vk::Check(vmaBindBufferMemory2(allocator, a, 0, raw, nullptr));
    return MemoryCommit(allocator, a, info);
}

#ifdef _WIN32
namespace {
/// The DXGI adapter of the Vulkan device (matched by LUID), or null. Looked up again when the
/// device changes (another GPU picked between sessions).
IDXGIAdapter3* FindDxgiAdapter(const Device& device) {
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    device.GetPhysical().GetProperties2(properties);
    if (!id.deviceLUIDValid) {
        return nullptr;
    }
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    static LUID cached_luid{};
    static IDXGIAdapter3* cached{};
    if (cached != nullptr && cached_luid.LowPart == luid.LowPart &&
        cached_luid.HighPart == luid.HighPart) {
        return cached;
    }
    if (cached != nullptr) {
        cached->Release();
        cached = nullptr;
    }
    cached_luid = luid;
    const HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    if (dxgi == nullptr) {
        return nullptr;
    }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create =
        reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    IDXGIFactory4* factory{};
    if (create == nullptr ||
        FAILED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) {
        return nullptr;
    }
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3),
                                          reinterpret_cast<void**>(&cached)))) {
        cached = nullptr;
    }
    factory->Release();
    return cached;
}
} // Anonymous namespace
#endif

void LogMemoryTrace(const Device& device) {
    const Common::ProcessMemoryUsage process = Common::GetProcessMemoryUsage();
    LOG_INFO(Render_Vulkan,
             "Memory trace: process commit {} MiB, working set {} MiB (private {} MiB), system "
             "commit {} of {} MiB",
             process.commit >> 20, process.working_set >> 20, process.private_working_set >> 20,
             process.system_commit >> 20, process.system_commit_limit >> 20);
#ifdef _WIN32
    // This process's use of the card's memory (local) and of system memory for the GPU
    // (non-local: host memory types, and what the OS moved out of video memory).
    IDXGIAdapter3* const adapter = FindDxgiAdapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    DXGI_QUERY_VIDEO_MEMORY_INFO non_local{};
    if (adapter != nullptr &&
        SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
        SUCCEEDED(
            adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local))) {
        LOG_INFO(Render_Vulkan,
                 "Memory trace: DXGI local {} of {} MiB, non-local {} of {} MiB",
                 local.CurrentUsage >> 20, local.Budget >> 20, non_local.CurrentUsage >> 20,
                 non_local.Budget >> 20);
    }
#endif
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    std::string heaps;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        const VmaBudget& budget = budgets[heap];
        fmt::format_to(std::back_inserter(heaps), "{}{} ({}): driver {} of {} MiB, VMA {}/{} MiB",
                       heap == 0 ? "" : "; ", heap,
                       (properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
                           ? "local"
                           : "host",
                       budget.usage >> 20, budget.budget >> 20,
                       budget.statistics.allocationBytes >> 20,
                       budget.statistics.blockBytes >> 20);
    }
    LOG_INFO(Render_Vulkan, "Memory trace: heaps {}", heaps);
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    std::string types;
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        const VmaStatistics& s = stats.memoryType[type].statistics;
        if (s.blockCount == 0) {
            continue;
        }
        fmt::format_to(std::back_inserter(types), "{}{} (heap {}, flags {:#x}): {} allocs {}/{} MiB",
                       types.empty() ? "" : "; ", type, properties->memoryTypes[type].heapIndex,
                       static_cast<u32>(properties->memoryTypes[type].propertyFlags),
                       s.allocationCount, s.allocationBytes >> 20, s.blockBytes >> 20);
    }
    LOG_INFO(Render_Vulkan, "Memory trace: types {}", types);
    // The separate pools, which the type totals above include.
    static constexpr std::array pool_names{"sparse pages", "device buffers", "render targets"};
    static_assert(pool_names.size() == static_cast<size_t>(Device::SeparatePool::Count));
    std::string pools;
    for (u32 kind = 0; kind < pool_names.size(); ++kind) {
        for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
            const VmaPool pool =
                device.GetSeparatePool(static_cast<Device::SeparatePool>(kind), type, allocator,
                                       false);
            if (pool == VK_NULL_HANDLE) {
                continue;
            }
            VmaDetailedStatistics d{};
            vmaCalculatePoolStatistics(allocator, pool, &d);
            fmt::format_to(std::back_inserter(pools),
                           "{}{} (type {}): {} allocs {}/{} MiB in {} blocks, {} free ranges "
                           "(largest {} MiB)",
                           pools.empty() ? "" : "; ", pool_names[kind], type,
                           d.statistics.allocationCount, d.statistics.allocationBytes >> 20,
                           d.statistics.blockBytes >> 20, d.statistics.blockCount,
                           d.unusedRangeCount, d.unusedRangeSizeMax >> 20);
        }
    }
    if (!pools.empty()) {
        LOG_INFO(Render_Vulkan, "Memory trace: pools {}", pools);
    }
}

void LogMemoryStatistics(const Device& device) {
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        const VmaStatistics& s = stats.memoryType[type].statistics;
        if (s.blockCount == 0) {
            continue;
        }
        LOG_DEBUG(Render_Vulkan, "VMA type {} (heap {}, flags {:#x}): {} blocks {} MiB, {} allocations {} MiB",
                 type, properties->memoryTypes[type].heapIndex,
                 static_cast<u32>(properties->memoryTypes[type].propertyFlags), s.blockCount,
                 s.blockBytes >> 20, s.allocationCount, s.allocationBytes >> 20);
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        LOG_INFO(Render_Vulkan, "VMA heap {}: usage {} MiB of budget {} MiB, VMA blocks {} MiB",
                 heap, budgets[heap].usage >> 20, budgets[heap].budget >> 20,
                 budgets[heap].statistics.blockBytes >> 20);
    }
}

} // namespace Vulkan
