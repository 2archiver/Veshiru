// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// What a memory pool buffer can be used for.
constexpr VkBufferUsageFlags POOL_BUFFER_USAGE =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

/// POOL_BUFFER_USAGE plus a device address (raw global memory, VK_KHR_buffer_device_address): used
/// when the device has the feature. Imports allocate their memory with DEVICE_ADDRESS then.
constexpr VkBufferUsageFlags POOL_BUFFER_USAGE_ADDRESSABLE =
    POOL_BUFFER_USAGE | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

/// A VkBuffer over imported host memory (VK_EXT_external_memory_host). The host range must stay
/// mapped for as long as the buffer exists.
struct ImportedBuffer {
    vk::DeviceMemory memory; // declared first: destroyed after the buffer
    vk::Buffer buffer;
    VkDeviceSize size{};
};

/// A large host range imported once as device memory; several pool buffers are bound into it at
/// their offsets (one import costs a few ms; see Common::HostMemory::BackingViewSize).
struct ImportedChunk {
    vk::DeviceMemory memory;
    u8* host{};
    VkDeviceSize size{};
    u32 memory_type_bits{}; ///< host-pointer types usable with pool buffers
    u32 type_index{};
    /// Guest (GPU) address of `host`: the chunk is [guest_base, guest_base + size) contiguous.
    u64 guest_base{};
    /// One buffer over the whole chunk with a device address, created the first time raw global
    /// memory needs the chunk (declared last: destroyed before the memory).
    vk::Buffer address_buffer;
    u64 device_address{};
};

/// VK_EXT_external_memory_host import. Loads vkGetMemoryHostPointerPropertiesEXT with
/// vkGetDeviceProcAddr and reads minImportedHostPointerAlignment itself.
class HostImporter {
public:
    /// `allocator` is stored in imported buffers so they are destroyed with the right owner; it is
    /// never used to allocate.
    HostImporter(const Device& device, VmaAllocator allocator);

    [[nodiscard]] bool Available() const noexcept {
        return get_properties != nullptr;
    }

    [[nodiscard]] u64 Alignment() const noexcept {
        return min_alignment;
    }

    /// True when `ptr`/`size` satisfy the import alignment.
    [[nodiscard]] bool IsAligned(const void* ptr, u64 size) const noexcept;

    /// Imports [ptr, ptr + size) as a buffer with `usage`. Returns VK_SUCCESS or the failing
    /// result; `error` is filled with a short reason on failure.
    VkResult Import(void* ptr, VkDeviceSize size, VkBufferUsageFlags usage, ImportedBuffer& out,
                    std::string& error) const;

    /// Gives every page of [ptr, ptr + size) its host page before an import. Pages nobody has
    /// touched yet have none: vkAllocateMemory would fault each one in (43 ms for 256 MiB) while
    /// it holds the driver lock every vkQueueSubmit of the device waits for. Reading one byte per
    /// page outside that lock does the same work (~50 ms per 256 MiB on the calling thread), and
    /// the import then only pins resident pages (~3 ms per 256 MiB). A read leaves the bytes as
    /// they are while the guest keeps writing them. On Windows only committed readable pages are
    /// read (the deferred-commit backing has reserved pages nobody was given). The imports call
    /// it themselves; calling it earlier, without locks held, moves the work off them.
    static void MakeResident(const u8* ptr, VkDeviceSize size);

    /// Where an import's time went: making the pages resident (MakeResident),
    /// vkGetMemoryHostPointerPropertiesEXT and vkAllocateMemory.
    struct ImportTiming {
        u64 resident_ns{};
        u64 props_ns{};
        u64 alloc_ns{};
    };

    /// Imports [ptr, ptr + size) as device memory for buffers with `usage` (chunk of many pools).
    /// `timing`, when given, receives where the time went.
    VkResult ImportChunk(u8* ptr, VkDeviceSize size, VkBufferUsageFlags usage, ImportedChunk& out,
                         std::string& error, ImportTiming* timing = nullptr) const;

    /// Creates a buffer of `size` bytes with `usage` over `chunk` at `offset` (a multiple of the
    /// buffer's alignment requirement).
    VkResult CreateBufferInChunk(const ImportedChunk& chunk, VkDeviceSize offset, VkDeviceSize size,
                                 VkBufferUsageFlags usage, vk::Buffer& out,
                                 std::string& error) const;

private:
    const Device& device;
    VmaAllocator allocator;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_properties{};
    u64 min_alignment{};
};

} // namespace Vulkan::Nvn
