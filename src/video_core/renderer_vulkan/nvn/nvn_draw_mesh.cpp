// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Direct mesh-task draws of the NVN replay (nvnCommandBufferDrawMeshTasks, ...3D): task/mesh
// pipelines drawn with vkCmdDrawMeshTasksEXT (VK_EXT_mesh_shader). Everything a vertex draw
// does besides vertex input (pipeline, dynamic state, descriptors, render pass, render
// conditions) is the same `DrawContext::PrepareDraw`. The indirect entries are in
// nvn_draw_indirect.cpp.
//
// What the NVN2 draw macros do (MME code captured on the native path, Ocarina of Time): every
// mesh-task draw is a one-dimensional launch of `count` workgroups starting at `first`. A 3D
// draw writes its {x, y, z} into the driver bank (VC::DriverMeshDimsOffset) and launches x*y*z
// workgroups; the task program linearizes with those words. So the host launches (n, 1, 1)
// workgroups too and leaves the dims in the driver bank.

#include <algorithm>
#include <array>
#include <string>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/nvn/nvn_draw.h"

namespace Vulkan::Nvn {

void DrawContext::DrawMeshTasks(const VC::DrawMeshTasksPayload& p) {
    if (p.indirect != 0) {
        return DrawMeshTasksIndirect(p);
    }
    if (p.dims == 3) {
        // The macro writes the dims whatever the count (the words stay for later draws).
        const std::array<u32, 3> dims{p.x, p.y, p.z};
        if (dims != mesh_dims) {
            mesh_dims = dims;
            ++mesh_dims_gen;
        }
    }
    const u64 groups = u64{p.x} * p.y * p.z;
    if (groups == 0) {
        return; // no workgroups: the GPU launches nothing
    }
    if (p.first != 0) {
        // VK_EXT_mesh_shader numbers the task workgroups of a draw from 0; there is no first one.
        ++counters.draws;
        ++counters.draws_skipped;
        host->HostUnsupported("DrawMeshTasks",
                              "a first task workgroup other than 0 (VK_EXT_mesh_shader has none)");
        return;
    }
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    PreparedDraw prepared;
    HostIndex host_index;
    dbg_call = {static_cast<u32>(groups), 1, 0, false};
    if (impl.gpu_profiler) {
        prof_units = groups;
    }
    if (PrepareDraw(MESH_DRAW_MODE, 0, nullptr, prepared, host_index)) {
        // The workgroups the device can launch in one dimension: of the task stage, or of the
        // mesh stage when the pipeline has no task program.
        const VkPhysicalDeviceMeshShaderPropertiesEXT& limits = impl.device.MeshShaderProperties();
        const bool task = prepared.pipeline->stages[STAGE_TASK].present;
        const u32 limit = task ? (std::min)(limits.maxTaskWorkGroupCount[0],
                                            limits.maxTaskWorkGroupTotalCount)
                               : (std::min)(limits.maxMeshWorkGroupCount[0],
                                            limits.maxMeshWorkGroupTotalCount);
        if (groups > limit) {
            ++counters.draws_skipped;
            host->HostUnsupported("DrawMeshTasks",
                                  fmt::format("{} {} workgroups exceed the device's {}", groups,
                                              task ? "task" : "mesh", limit));
        } else {
            host->HostCmd().DrawMeshTasksEXT(static_cast<u32>(groups), 1, 1);
            impl.stats.mesh_task_draws.fetch_add(1, std::memory_order_relaxed);
        }
    }
    host->HostEndRenderCondition();
}

} // namespace Vulkan::Nvn
