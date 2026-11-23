// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <chrono>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/nvn/nvn_host_import.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vma.h"

namespace Vulkan::Nvn {

void HostImporter::MakeResident(const u8* ptr, VkDeviceSize size) {
    constexpr VkDeviceSize PageSize = 4096;
    const auto touch = [](const u8* begin, const u8* end) {
        for (const u8* page = begin; page < end; page += PageSize) {
            static_cast<void>(*reinterpret_cast<const volatile u8*>(page));
        }
    };
#ifdef _WIN32
    const u8* const end = ptr + size;
    for (const u8* cursor = ptr; cursor < end;) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(cursor, &region, sizeof(region)) == 0) {
            return;
        }
        const u8* const region_end =
            (std::min)(static_cast<const u8*>(region.BaseAddress) + region.RegionSize, end);
        constexpr DWORD Readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                                   PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                   PAGE_EXECUTE_WRITECOPY;
        if (region.State == MEM_COMMIT && (region.Protect & Readable) != 0 &&
            (region.Protect & PAGE_GUARD) == 0) {
            touch(cursor, region_end);
        }
        cursor = region_end;
    }
#else
    touch(ptr, ptr + size);
#endif
}

HostImporter::HostImporter(const Device& device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {
    if (!device.IsExtExternalMemoryHostSupported()) {
        return;
    }
    const auto& dld = device.GetDispatchLoader();
    get_properties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        dld.vkGetDeviceProcAddr(*device.GetLogical(), "vkGetMemoryHostPointerPropertiesEXT"));
    if (!get_properties) {
        return;
    }
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
        .pNext = nullptr,
        .minImportedHostPointerAlignment = 0,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &host_properties,
        .properties = {},
    };
    device.GetPhysical().GetProperties2(properties2);
    min_alignment = host_properties.minImportedHostPointerAlignment;
    if (min_alignment == 0 || !std::has_single_bit(min_alignment)) {
        get_properties = nullptr;
        min_alignment = 0;
    }
}

bool HostImporter::IsAligned(const void* ptr, u64 size) const noexcept {
    if (!Available() || size == 0) {
        return false;
    }
    return reinterpret_cast<uintptr_t>(ptr) % min_alignment == 0 && size % min_alignment == 0;
}

VkResult HostImporter::Import(void* ptr, VkDeviceSize size, VkBufferUsageFlags usage,
                              ImportedBuffer& out, std::string& error) const {
    if (!Available()) {
        error = "VK_EXT_external_memory_host is not available";
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    if (!IsAligned(ptr, size)) {
        error = fmt::format("pointer {:p} / size {:#x} not aligned to {:#x}", static_cast<const void*>(ptr), size,
                            min_alignment);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    MakeResident(static_cast<const u8*>(ptr), size);

    VkMemoryHostPointerPropertiesEXT pointer_properties{
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
        .pNext = nullptr,
        .memoryTypeBits = 0,
    };
    VkResult result = get_properties(
        logical, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &pointer_properties);
    if (result != VK_SUCCESS) {
        error = "vkGetMemoryHostPointerPropertiesEXT failed";
        return result;
    }

    const VkExternalMemoryBufferCreateInfo external_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_info,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer raw_buffer{};
    result = dld.vkCreateBuffer(logical, &buffer_ci, nullptr, &raw_buffer);
    if (result != VK_SUCCESS) {
        error = "vkCreateBuffer failed";
        return result;
    }
    // Wrapped right away so every early exit below destroys it. The allocator is only the owner
    // recorded for destruction: with a null allocation vmaDestroyBuffer just calls vkDestroyBuffer.
    vk::Buffer buffer{raw_buffer, logical, allocator, VK_NULL_HANDLE, {}, false, {}, dld};

    const VkMemoryRequirements requirements =
        device.GetLogical().GetBufferMemoryRequirements(raw_buffer);
    const auto memory_properties = device.GetPhysical().GetMemoryProperties().memoryProperties;
    const u32 candidates = requirements.memoryTypeBits & pointer_properties.memoryTypeBits;
    u32 type_index = ~0u;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if (!(candidates & (1u << i))) {
            continue;
        }
        const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            type_index = i;
            break;
        }
    }
    if (type_index == ~0u) {
        error = fmt::format("no host-coherent memory type (host pointer types {:#x}, buffer types "
                            "{:#x})",
                            pointer_properties.memoryTypeBits, requirements.memoryTypeBits);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (requirements.size > size) {
        error = fmt::format("buffer needs {:#x} bytes, only {:#x} imported", requirements.size,
                            size);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Buffers with SHADER_DEVICE_ADDRESS usage need their memory allocated with the flag.
    const VkMemoryAllocateFlagsInfo address_flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = nullptr,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        .deviceMask = 0,
    };
    const VkImportMemoryHostPointerInfoEXT import_info{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0 ? &address_flags : nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = ptr,
    };
    const VkMemoryAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = size,
        .memoryTypeIndex = type_index,
    };
    VkDeviceMemory raw_memory{};
    result = dld.vkAllocateMemory(logical, &allocate_info, nullptr, &raw_memory);
    if (result != VK_SUCCESS) {
        error = "vkAllocateMemory (import) failed";
        return result;
    }
    vk::DeviceMemory memory{raw_memory, logical, dld};
    result = dld.vkBindBufferMemory(logical, raw_buffer, raw_memory, 0);
    if (result != VK_SUCCESS) {
        error = "vkBindBufferMemory failed";
        return result;
    }
    out.buffer = std::move(buffer);
    out.memory = std::move(memory);
    out.size = size;
    return VK_SUCCESS;
}

VkResult HostImporter::ImportChunk(u8* ptr, VkDeviceSize size, VkBufferUsageFlags usage,
                                   ImportedChunk& out, std::string& error,
                                   ImportTiming* timing) const {
    using Clock = std::chrono::steady_clock;
    const auto ns_since = [](Clock::time_point begin) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
    };
    if (!Available()) {
        error = "VK_EXT_external_memory_host is not available";
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    if (!IsAligned(ptr, size)) {
        error = fmt::format("chunk {:p} / size {:#x} not aligned to {:#x}", static_cast<const void*>(ptr), size,
                            min_alignment);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    auto begin = Clock::now();
    MakeResident(ptr, size);
    if (timing) {
        timing->resident_ns = ns_since(begin);
    }
    VkMemoryHostPointerPropertiesEXT pointer_properties{
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
        .pNext = nullptr,
        .memoryTypeBits = 0,
    };
    begin = Clock::now();
    VkResult result = get_properties(
        logical, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &pointer_properties);
    if (timing) {
        timing->props_ns = ns_since(begin);
    }
    if (result != VK_SUCCESS) {
        error = "vkGetMemoryHostPointerPropertiesEXT failed";
        return result;
    }
    // Buffer memory requirements do not depend on the size for type bits: probe with a small one.
    const VkExternalMemoryBufferCreateInfo external_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    const VkBufferCreateInfo probe_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_info,
        .flags = 0,
        .size = min_alignment,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer raw_probe{};
    result = dld.vkCreateBuffer(logical, &probe_ci, nullptr, &raw_probe);
    if (result != VK_SUCCESS) {
        error = "vkCreateBuffer (probe) failed";
        return result;
    }
    vk::Buffer probe{raw_probe, logical, allocator, VK_NULL_HANDLE, {}, false, {}, dld};
    const VkMemoryRequirements requirements =
        device.GetLogical().GetBufferMemoryRequirements(raw_probe);
    const auto memory_properties = device.GetPhysical().GetMemoryProperties().memoryProperties;
    const u32 candidates = requirements.memoryTypeBits & pointer_properties.memoryTypeBits;
    u32 type_index = ~0u;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((candidates & (1u << i)) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            type_index = i;
            break;
        }
    }
    if (type_index == ~0u) {
        error = "no host-coherent memory type for the imported chunk";
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    // Buffers with SHADER_DEVICE_ADDRESS usage need their memory allocated with the flag.
    const VkMemoryAllocateFlagsInfo address_flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = nullptr,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        .deviceMask = 0,
    };
    const VkImportMemoryHostPointerInfoEXT import_info{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0 ? &address_flags : nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = ptr,
    };
    const VkMemoryAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = size,
        .memoryTypeIndex = type_index,
    };
    VkDeviceMemory raw_memory{};
    begin = Clock::now();
    result = dld.vkAllocateMemory(logical, &allocate_info, nullptr, &raw_memory);
    if (timing) {
        timing->alloc_ns = ns_since(begin);
    }
    if (result != VK_SUCCESS) {
        error = "vkAllocateMemory (chunk import) failed";
        return result;
    }
    out.memory = vk::DeviceMemory{raw_memory, logical, dld};
    out.host = ptr;
    out.size = size;
    out.memory_type_bits = candidates;
    out.type_index = type_index;
    return VK_SUCCESS;
}

VkResult HostImporter::CreateBufferInChunk(const ImportedChunk& chunk, VkDeviceSize offset,
                                           VkDeviceSize size, VkBufferUsageFlags usage,
                                           vk::Buffer& out, std::string& error) const {
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    const VkExternalMemoryBufferCreateInfo external_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_info,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer raw_buffer{};
    VkResult result = dld.vkCreateBuffer(logical, &buffer_ci, nullptr, &raw_buffer);
    if (result != VK_SUCCESS) {
        error = "vkCreateBuffer failed";
        return result;
    }
    vk::Buffer buffer{raw_buffer, logical, allocator, VK_NULL_HANDLE, {}, false, {}, dld};
    const VkMemoryRequirements requirements =
        device.GetLogical().GetBufferMemoryRequirements(raw_buffer);
    if (!(requirements.memoryTypeBits & (1u << chunk.type_index)) || offset >= chunk.size ||
        (std::max)(size, requirements.size) > chunk.size - offset ||
        (requirements.alignment != 0 && offset % requirements.alignment != 0)) {
        error = fmt::format("pool at chunk offset {:#x} (+{:#x}) does not fit the chunk's memory "
                            "type/alignment {:#x}",
                            offset, size, requirements.alignment);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    result = dld.vkBindBufferMemory(logical, raw_buffer, *chunk.memory, offset);
    if (result != VK_SUCCESS) {
        error = "vkBindBufferMemory (chunk) failed";
        return result;
    }
    out = std::move(buffer);
    return VK_SUCCESS;
}

} // namespace Vulkan::Nvn
