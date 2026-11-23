// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// nv:: (the driver glue layer beneath NVN). nxnative replaces it wholesale; here the original guest
// code keeps running: nn::vi::Initialize (and the rest of the sdk) checks state that only the real
// nv::InitializeGraphics sets (a call-once flag at sdk+0xde66a0 with the pool arguments, read at
// sdk+0x3dfa40), and the run aborted with a nn::Result in nn::vi::Initialize when the call was
// answered by the host. The bodies here only observe the calls.
namespace Core::NvnHle::Hle {
namespace {

/// nv::InitializeGraphics(void* memory, size_t size): the arguments are logged (DK passes 0, 0).
void nvInitializeGraphics(Call& c) {
    const u64 memory = c.args.NextPointer();
    const u64 size = c.args.NextU64();
    c.runtime.nv_pool_size = size;
    c.runtime.nv_initialized = true;
    LOG_INFO(Core, "[nvn] nv::InitializeGraphics(memory={:#x}, size={:#x}): the sdk's own code runs",
             memory, size);
}

void nvFinalizeGraphics(Call& c) {
    c.runtime.nv_initialized = false;
    c.runtime.nv_pool_size = 0;
}

} // namespace

void RegisterNvFunctions(Registry& registry) {
    registry.ObserveNv<&nvInitializeGraphics>("_ZN2nv18InitializeGraphicsEPvm");
    registry.ObserveNv<&nvFinalizeGraphics>("_ZN2nv16FinalizeGraphicsEv");
}

} // namespace Core::NvnHle::Hle
