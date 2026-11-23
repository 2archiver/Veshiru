// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Optional observer of the guest's NVN InvalidateMappedRange calls (the MCP server's watches
// sampled "at invalidate"). Unset, it costs one atomic load per call.

#include <atomic>

#include "common/common_types.h"

namespace Core::Trace {

/// Called on the guest thread inside nvnMemoryPool/BufferInvalidateMappedRange (VTX-HLE), after
/// any write-back wait and just before the call returns: the CPU is about to read
/// [address, address + size). A hook may still run briefly after it is cleared, so the function
/// it points to must stay valid.
using InvalidateHook = void (*)(u64 address, u64 size);
inline std::atomic<InvalidateHook> g_invalidate_hook{nullptr};

} // namespace Core::Trace
