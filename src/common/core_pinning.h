// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Common {

/// One logical processor as the host reports it.
struct LogicalProcessor {
    /// What the host's affinity call takes (a CPU set id on Windows).
    u32 id;
    /// The physical core: equal for SMT siblings.
    u32 core;
    /// Higher is faster. Equal on every processor of a CPU that is not hybrid.
    u8 efficiency_class;
};

/// Physical cores of the fastest class needed before the GPU threads are pinned.
///
/// The front and back threads are the latency-critical pair: a stalled front delays every command
/// list, and a stalled back delays the Vulkan submission the frame waits on. Two dedicated cores
/// are enough to keep them out of each other's way, so a 4 P-core hybrid laptop (i5-12450H and
/// similar) is pinned too; before 3.2 it was skipped entirely and shared its P-cores with the
/// guest cores. The two guest cores need two more, hence the higher guest threshold.
constexpr std::size_t MinCoresForGpuThreadPinning = 4;

/// Physical cores of the fastest class needed before the guest cores are pinned to them as well.
constexpr std::size_t MinCoresForGuestCorePinning = 6;

/// Logical processor ids for each pinned role. All three are empty when the host has fewer than
/// MinCoresForGpuThreadPinning performance cores; guest is empty when it has fewer than
/// MinCoresForGuestCorePinning (the guest cores then run wherever the scheduler puts them, which
/// keeps them off the two cores the GPU threads own only if there are enough of the others).
struct GpuThreadPinning {
    std::size_t performance_cores{};
    std::vector<u32> front;
    std::vector<u32> back;
    std::vector<u32> guest;
};

/// The split GPU thread's front and back each get a whole physical performance core, so no SMT
/// sibling shares it, and the guest cores get the remaining performance cores. Core 0 takes most
/// interrupts, so the GPU threads get the next two.
[[nodiscard]] GpuThreadPinning PlanGpuThreadPinning(std::span<const LogicalProcessor> processors);

/// The CPUs of a Linux sysfs CPU list such as "0-3,8,10-11", in order. Empty for an empty or
/// malformed list.
[[nodiscard]] std::vector<u32> ParseCpuList(std::string_view list);

} // namespace Common
