// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <utility>
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/settings_enums.h"

namespace Common {

enum class MemoryPermission : u32 {
    Read = 1 << 0,
    Write = 1 << 1,
    ReadWrite = Read | Write,
    Execute = 1 << 2,
};
DECLARE_ENUM_FLAG_OPERATORS(MemoryPermission)

/**
 * A low level linear memory buffer, which supports multiple mappings
 * Its purpose is to rebuild a given sparse memory layout, including mirrors.
 */
/// Whether HostMemory::HintCold pages memory out. Its passes stall the GPU threads for as long as
/// they run (Mario Kart World: 60-700 ms every 600 frames, the longer the session the longer).
/// The cold hint mode in effect: the cold_hint_mode setting (off by default), unless
/// EDEN_NX2_COLD_HINT (off, low, always) overrides it.
[[nodiscard]] Settings::ColdHintMode ColdHintModeInEffect();

/// Whether the system is short of memory: memory pressure stalls over 5% of the last 10 s (Linux
/// PSI, this process's cgroup or the whole system), or less than a tenth of RAM (at least 1 GiB)
/// available. Re-read at most once a second.
[[nodiscard]] bool MemoryIsLow();

enum class ColdHintAction { None, MarkedCold, PagedOut };

class HostMemory {
public:
    /// Windows maps the backing as views of this size placed back to back (one contiguous
    /// range). Anything that queries the host pages of a view (VirtualQuery,
    /// vkGetMemoryHostPointerPropertiesEXT and every VK_EXT_external_memory_host import on the
    /// NVIDIA driver) costs time proportional to the resident part of the whole view: with one
    /// 12 GiB view holding 6 GiB of guest data each such call took ~35 ms, even for 64 KiB; with
    /// 64 MiB views 0.2-0.5 ms (a 256 MiB import ~3 ms). Imports may span several views.
    static constexpr size_t BackingViewSize = size_t{64} << 20;

    explicit HostMemory(size_t backing_size_, size_t virtual_size_);
    ~HostMemory();

    /**
     * Copy constructors. They shall return a copy of the buffer without the mappings.
     * TODO: Implement them with COW if needed.
     */
    HostMemory(const HostMemory& other) = delete;
    HostMemory& operator=(const HostMemory& other) = delete;

    /**
     * Move constructors. They will move the buffer and the mappings to the new object.
     */
    HostMemory(HostMemory&& other) noexcept;
    HostMemory& operator=(HostMemory&& other) noexcept;

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms,
             bool separate_heap);

    void Unmap(size_t virtual_offset, size_t length, bool separate_heap);

    void Protect(size_t virtual_offset, size_t length, MemoryPermission perms);

    void EnableDirectMappedAddress();

    /// Clears guest memory being handed out. Memory still zero since it was created (or since its
    /// last full clear) is skipped: clearing it only made the host give every page real memory.
    void ClearBackingRegion(size_t physical_offset, size_t length, u32 fill_value);

    /// Memory written once and read again rarely: its pages leave the working set (backing and,
    /// when given, its view at virtual_offset), so the OS compresses or pages them out first when
    /// it needs RAM. The contents stay; using the pages brings them back.
    void HintCold(size_t physical_offset, size_t length,
                  std::optional<size_t> virtual_offset) const;
    /// What one TrimZeroPages pass looked at and achieved.
    struct ZeroTrimResult {
        /// Resident bytes of guest memory the pass looked at.
        size_t resident{};
        /// Bytes of zero pages that left the working set.
        size_t trimmed{};
        /// The pass stopped at `max_scan_bytes` with resident pages left over.
        bool truncated{};
    };

    /// Windows: resident pages of guest memory (backing and its views) that hold only zeros leave
    /// the working set, like HintCold. A pass over all of it takes seconds on a guest that holds
    /// gigabytes, so a caller that must not stall a running game passes `max_scan_bytes` and gets a
    /// bounded slice: pages already trimmed are no longer resident, so the next slice carries on
    /// past them without a cursor.
    ZeroTrimResult TrimZeroPages(size_t max_scan_bytes = 0) const;

    /// Bytes ClearBackingRegion has filled with zeros (memset) since start.
    static u64 ZeroFilledBytes() noexcept;

    /// Bytes Map, Unmap and Protect have covered since start.
    static u64 MappedBytes() noexcept;

    /// Pages the kernel hands out (cleared or not).
    void MarkUsed(size_t physical_offset, size_t length);

    /// Pages given back to the kernel: their contents are no longer known to be zero.
    void MarkDirty(size_t physical_offset, size_t length);

    /// A new kernel on this memory: everything a previous one used may hold data.
    void ForgetContents();

    [[nodiscard]] u8* BackingBasePointer() noexcept {
        return backing_base;
    }
    [[nodiscard]] const u8* BackingBasePointer() const noexcept {
        return backing_base;
    }

    [[nodiscard]] u8* VirtualBasePointer() noexcept {
        return virtual_base;
    }
    [[nodiscard]] const u8* VirtualBasePointer() const noexcept {
        return virtual_base;
    }

    [[nodiscard]] size_t VirtualSize() const noexcept {
        return virtual_size;
    }

    bool IsInVirtualRange(void* address) const noexcept {
        return address >= virtual_base && address < virtual_base + virtual_size;
    }

private:
    void Swap(HostMemory& other) noexcept;

    friend class HostMemoryTestAccess;
    struct AllocationTestControl {
        size_t backing_size;
        size_t virtual_size;
        bool fail_fastmem;
        bool fail_fallback;
        size_t fastmem_attempts{};
        size_t fallback_attempts{};
    };
    static thread_local AllocationTestControl* allocation_test_control;

    size_t backing_size{};
    size_t virtual_size{};

#if !(defined(__OPENORBIS__) || defined(__managarm__))
    // Low level handler for the platform dependent memory routines
    class Impl;
    std::unique_ptr<Impl> impl;
#endif
    u8* backing_base{};
    u8* virtual_base{};
    size_t virtual_base_offset{};

    static constexpr size_t GranuleShift = 16; ///< 64 KiB of backing per state byte
    static constexpr u8 GranuleUsed = 1 << 0;   ///< Handed out at least once
    static constexpr u8 GranuleDirty = 1 << 1;  ///< May hold non-zero data
    std::unique_ptr<std::atomic<u8>[]> granules;
    size_t num_granules{};
    // Windows requires it for kernels whom lack proper support for some functions!
    bool fallback_buffer{false};
};

} // namespace Common
