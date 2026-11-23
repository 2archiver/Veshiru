// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include "common/settings.h"
#include "common/vtx_hle_settings.h"

namespace Settings::VtxHle {

namespace {

constexpr Reason Emulated = Reason::EmulatedGpuOnly;
constexpr Reason NotYet = Reason::NotYetSupported;

// A setting is listed when nothing reachable from VTX-HLE (core/hle/nvn, video_core/nvn,
// renderer_vulkan/nvn and the shared code they call: device creation, MaxwellToVK, present
// manager, swapchain, core memory) reads it. Settings that look emulated-only but have a reader on
// the VTX-HLE path stay out, and tests/common/vtx_hle_settings.cpp pins them:
//   async_presentation        PresentManager::use_present_thread; the replay presents through it
//   frame_pacing_mode         Settings::FrameRateCap in functions_window.cpp (nvnWindowPresent)
//   force_max_clock           the replay feeds TurboMode after each queue submission
//   use_video_framerate       PlanPresentPacing paces the present at 100% while NVDEC is active
//   use_disk_shader_cache     the NVN pipeline cache gets no directory when it is off
//   use_vulkan_driver_pipeline_cache  the NVN cache's .vkcache file is not used when it is off
//   use_asynchronous_shaders  NVN PipelineCacheOptions::async_compile
//   dyna_state                vulkan_device.cpp drops the EDS features; the fast-linked pipelines
//                             (nvn_pipeline_library) need EDS + EDS2
//   use_reactive_flushing     arm_dynarmic_64.cpp recompile_on_fastmem_read_fault, which the
//                             replay's read-armed pages reach
//   enable_buffer_history     nvnflinger buffer queue (a guest service)
//   vram_usage_mode           vulkan_device.cpp memory budget the replay's caches use
//   gpu_clock                 timestamp_mode.cpp (guest GPU clock divisor)
//   max_anisotropy            nvn_sampler_aniso.h: the replay's VkSamplers get the setting by the
//                             emulated sampler's rule (TSCEntry::MaxAnisotropy)
//   astc_recompression        the replay holds its ASTC stand-ins as BC1/BC3 by it (nvn_context.cpp
//                             astc_standin, nvn_format.h ToBcStandin)
//   nvn_smart_bc              the per-texture BC format choice of those stand-ins (nvn_context.cpp
//                             ParseSmartBc; EDEN_NX2_NVN_SMART_BC overrides it)
// (the first six rows are the HleSettingsPresent work, feat/hle-settings-present c1e152562a)
constexpr std::array<Entry, 20> UNAPPLIED{{
    // GPU command processor and its synchronisation (dma_pusher, fence_manager, maxwell_3d,
    // query_cache, rasterizer, gpu_thread).
    {"gpu_accuracy", Emulated},
    {"dma_accuracy", Emulated},
    {"gpu_fence_behavior", Emulated},
    {"sync_memory_operations", Emulated},      // dma_pusher.cpp only
    {"skip_cpu_inner_invalidation", Emulated}, // MemoryManager -> rasterizer InnerInvalidation
    // Emulated graphics pipelines (vk_pipeline_cache, vk_graphics_pipeline).
    {"enable_compute_pipelines", Emulated}, // vk_pipeline_cache.cpp HasBrokenCompute
    {"sample_shading_fraction", Emulated},  // vk_graphics_pipeline.cpp (the setting's key; its label is sample_shading)
    // vulkan_device.cpp keeps VK_EXT_vertex_input_dynamic_state whatever this says when VTX-HLE
    // runs (nvn_hle_keeps_vids).
    {"vertex_input_dynamic_state", Emulated},
    // Emulated texture cache (vk_texture_cache, texture_cache.h). The replay has its own ASTC
    // decode and untile (nvn_astc_gpu, nvn_untile_gpu, nvn_upload).
    {"barrier_feedback_loops", Emulated},
    {"fix_bloom_effects", Emulated},
    {"accelerate_astc", Emulated},
    {"gpu_unswizzle_enabled", Emulated},
    {"gpu_unswizzle_texture_size", Emulated},
    {"gpu_unswizzle_stream_size", Emulated},
    {"gpu_unswizzle_chunk_size", Emulated},
    // Eglantine's presentation passes (window filters past bilinear, anti-aliasing,
    // post-processing) and frame generation record into the emulated GPU's scheduler: the replay
    // presents with its plain blit (nvn_settings_notice.h).
    {"anti_aliasing", NotYet},
    {"post_shader_enabled", NotYet},
    {"post_shader_chain", NotYet},
    {"frame_gen", NotYet},
    {"frame_gen_type", NotYet},
}};

} // namespace

std::span<const Entry> Unapplied() {
    return UNAPPLIED;
}

const Entry* Find(std::string_view key) {
    const auto it = std::ranges::find(UNAPPLIED, key, &Entry::key);
    return it != UNAPPLIED.end() ? &*it : nullptr;
}

bool IsActive(const PathQuery& query) {
    if (query.game_running) {
        return query.running_hle;
    }
    return query.vulkan &&
           ChooseNvnHle(query.environment, query.selected, query.switch2_program).hle;
}

} // namespace Settings::VtxHle
