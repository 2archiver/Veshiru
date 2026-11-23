// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <set>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "common/vtx_hle_settings.h"

namespace {
using namespace Settings::VtxHle;
using Settings::GraphicsPath;

constexpr PathQuery Global(GraphicsPath selected) {
    return {.selected = selected};
}
} // namespace

TEST_CASE("VTX-HLE settings: every listed key is a real, unique setting", "[settings][vtxhle]") {
    std::set<std::string_view> seen;
    for (const Entry& entry : Unapplied()) {
        INFO(entry.key);
        // A typo would leave the setting enabled with nobody noticing.
        REQUIRE(Settings::values.linkage.by_key.contains(std::string{entry.key}));
        REQUIRE(seen.insert(entry.key).second);
        REQUIRE(Find(entry.key) == &entry);
    }
    REQUIRE(Find("no_such_setting") == nullptr);
}

TEST_CASE("VTX-HLE settings: the list holds the emulated GPU's own settings", "[settings][vtxhle]") {
    for (const std::string_view key :
         {"gpu_accuracy", "dma_accuracy", "gpu_fence_behavior", "sync_memory_operations",
          "enable_compute_pipelines", "barrier_feedback_loops", "fix_bloom_effects",
          "accelerate_astc", "gpu_unswizzle_enabled",
          "gpu_unswizzle_texture_size", "gpu_unswizzle_stream_size", "gpu_unswizzle_chunk_size",
          "sample_shading_fraction", "vertex_input_dynamic_state",
          "skip_cpu_inner_invalidation"}) {
        INFO(key);
        const Entry* const entry = Find(key);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->reason == Reason::EmulatedGpuOnly);
    }
}

TEST_CASE("VTX-HLE settings: settings with a reader on the HLE path are never listed",
          "[settings][vtxhle]") {
    // Each has a reader the VTX-HLE path reaches (see the comment in vtx_hle_settings.cpp), so
    // greying it would hide a setting that still does something.
    for (const std::string_view key : {"async_presentation",
                                       "frame_pacing_mode",
                                       "force_max_clock",
                                       "use_video_framerate",
                                       "use_disk_shader_cache",
                                       "use_vulkan_driver_pipeline_cache",
                                       "use_asynchronous_shaders",
                                       "dyna_state",
                                       "use_reactive_flushing",
                                       "enable_buffer_history",
                                       "vram_usage_mode",
                                       "use_docked_mode",
                                       "scaling_filter",
                                       "fsr_sharpening_slider",
                                       "use_vsync",
                                       "max_anisotropy",
                                       "astc_recompression",
                                       "resolution_setup",
                                       "graphics_path",
                                       "nvn_in_pass_queries",
                                       "nvn_pipeline_library",
                                       "nvn_sync_diet",
                                       "nvn_smart_bc",
                                       "nvn_shader_refresh"}) {
        INFO(key);
        REQUIRE(Find(key) == nullptr);
    }
}

TEST_CASE("VTX-HLE settings: presentation passes and frame generation are not applied yet",
          "[settings][vtxhle]") {
    // The replay presents with its plain blit in Eglantine (nvn_settings_notice.h).
    for (const std::string_view key : {"anti_aliasing", "post_shader_enabled", "post_shader_chain",
                                       "frame_gen", "frame_gen_type"}) {
        INFO(key);
        const Entry* const entry = Find(key);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->reason == Reason::NotYetSupported);
    }
}

TEST_CASE("VTX-HLE path: the global page follows the selected path", "[settings][vtxhle]") {
    REQUIRE(IsActive(Global(GraphicsPath::Hle)));
    REQUIRE_FALSE(IsActive(Global(GraphicsPath::Emulated)));
}

TEST_CASE("VTX-HLE path: the Vulkan renderer is required", "[settings][vtxhle]") {
    PathQuery query = Global(GraphicsPath::Hle);
    query.vulkan = false;
    REQUIRE_FALSE(IsActive(query));
}

TEST_CASE("VTX-HLE path: a Switch 1 program always uses the emulated GPU", "[settings][vtxhle]") {
    PathQuery query = Global(GraphicsPath::Hle);
    query.switch2_program = false;
    REQUIRE_FALSE(IsActive(query));
    query.switch2_program = true;
    REQUIRE(IsActive(query));
}

TEST_CASE("VTX-HLE path: EDEN_NX2_NVN_HLE overrides the selection like the game start does",
          "[settings][vtxhle]") {
    PathQuery query = Global(GraphicsPath::Emulated);
    query.environment = "1";
    REQUIRE(IsActive(query));
    query.environment = "";
    REQUIRE(IsActive(query)); // anything but "0" turns the HLE on
    query = Global(GraphicsPath::Hle);
    query.environment = "0";
    REQUIRE_FALSE(IsActive(query));
}

TEST_CASE("VTX-HLE path: a running game's actual path beats the selection", "[settings][vtxhle]") {
    // The dropdown only applies at the next start.
    PathQuery query = Global(GraphicsPath::Emulated);
    query.game_running = true;
    query.running_hle = true;
    REQUIRE(IsActive(query));

    query = Global(GraphicsPath::Hle);
    query.game_running = true;
    query.running_hle = false; // a Switch 1 game, or the device could not run the replay
    REQUIRE_FALSE(IsActive(query));

    query.vulkan = false; // the page's API choice does not move a running game either
    query.running_hle = true;
    REQUIRE(IsActive(query));
}
