// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#ifdef _WIN32
#include <thread>
#endif

#include "common/host_memory.h"
#include "common/typed_address.h"

namespace Core {

/// Whether the zero-page trim asks for a pass now.
///
/// A pass reads every resident guest page under VAD locks and takes 1.0-2.1 s on a 32 GiB laptop
/// (Ocarina of Time logs, 2026-11-22), which the player sees as a freeze. The saving it makes is
/// optional, so a pass only runs while the guest is quiet (paused, loading, or in a menu) or when
/// host commit is actually under pressure; `since_last_pass` keeps a busy session from deferring
/// it forever.
///
/// @param mapping_active   Bulk mapping/unmapping/protection is running (loads, boots, spawns).
/// @param guest_presenting The guest completed at least one frame since the last probe.
/// @param since_last_pass  Time since the last pass that trimmed anything.
/// @param memory_pressure  The process working set is big enough that the saving matters.
/// @param escape           How long a presenting guest holds a pass back under pressure: long on a
///                         32 GiB host, whose commit has room (120 s); short on a 16 GiB one, where
///                         the trim is what keeps the commit bounded (20 s).
[[nodiscard]] constexpr bool ShouldTrimZeroPages(bool mapping_active, bool guest_presenting,
                                                 std::chrono::seconds since_last_pass,
                                                 bool memory_pressure,
                                                 std::chrono::seconds escape) noexcept {
    if (mapping_active) {
        return false;
    }
    if (!guest_presenting) {
        return true;
    }
    return memory_pressure && since_last_pass >= escape;
}

namespace DramMemoryMap {
enum : u64 {
    Base = 0x80000000ULL,
    KernelReserveBase = Base + 0x60000,
    SlabHeapBase = KernelReserveBase + 0x85000,
};
}; // namespace DramMemoryMap

class DeviceMemory {
public:
    explicit DeviceMemory();
    ~DeviceMemory();

    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory(const DeviceMemory&) = delete;

    template <typename T>
    Common::PhysicalAddress GetPhysicalAddr(const T* ptr) const {
        return GetPhysicalAddr(reinterpret_cast<uintptr_t>(ptr));
    }

    Common::PhysicalAddress GetPhysicalAddr(uintptr_t ptr) const {
        return (ptr - reinterpret_cast<uintptr_t>(buffer.BackingBasePointer())) +
               DramMemoryMap::Base;
    }

    template <typename T>
    PAddr GetRawPhysicalAddr(const T* ptr) const {
        return static_cast<PAddr>(reinterpret_cast<uintptr_t>(ptr) -
                                  reinterpret_cast<uintptr_t>(buffer.BackingBasePointer()));
    }

    template <typename T>
    T* GetPointer(Common::PhysicalAddress addr) {
        return reinterpret_cast<T*>(buffer.BackingBasePointer() +
                                    (GetInteger(addr) - DramMemoryMap::Base));
    }

    template <typename T>
    const T* GetPointer(Common::PhysicalAddress addr) const {
        return reinterpret_cast<T*>(buffer.BackingBasePointer() +
                                    (GetInteger(addr) - DramMemoryMap::Base));
    }

    template <typename T>
    T* GetPointerFromRaw(PAddr addr) {
        return reinterpret_cast<T*>(buffer.BackingBasePointer() + addr);
    }

    template <typename T>
    const T* GetPointerFromRaw(PAddr addr) const {
        return reinterpret_cast<T*>(buffer.BackingBasePointer() + addr);
    }

    Common::HostMemory buffer;

#ifdef _WIN32
private:
    void TrimZeroPages(std::stop_token stop);

    std::jthread zero_trim_thread;
#endif
};

} // namespace Core
