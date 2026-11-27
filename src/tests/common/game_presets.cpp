// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <set>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#undef LoadString
#endif
#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"

TEST_CASE("Display filter defaults use ScaleForce and FXAA", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    CHECK(settings->scaling_filter.GetDefault() == Settings::ScalingFilter::ScaleForce);
    CHECK(settings->anti_aliasing.GetDefault() == Settings::AntiAliasing::Fxaa);
}

TEST_CASE("A game override keeps CPU inner invalidation on despite the skip hack",
          "[game_presets]") {
    auto& values = Settings::values;
    const bool skip = values.skip_cpu_inner_invalidation.GetValue();
    const bool keep = values.keep_cpu_inner_invalidation;
    values.skip_cpu_inner_invalidation.SetValue(true);
    values.keep_cpu_inner_invalidation = false;
    CHECK(Settings::SkipCpuInnerInvalidation());
    values.keep_cpu_inner_invalidation = true;
    CHECK_FALSE(Settings::SkipCpuInnerInvalidation());
    values.skip_cpu_inner_invalidation.SetValue(skip);
    values.keep_cpu_inner_invalidation = keep;
}

TEST_CASE("3D V-Cache thread placement and live RefreshThreadPolicies", "[common][thread_affinity]") {
    CHECK(Settings::values.prefer_3d_vcache.GetDefault());
#ifdef _WIN32
    const bool prev_setting = Settings::values.prefer_3d_vcache.GetValue();
    Settings::values.prefer_3d_vcache.SetValue(true);

    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    REQUIRE(GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask));

    const auto query_current_thread_mask = [process_mask]() -> DWORD_PTR {
        const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), process_mask);
        if (prev != 0) {
            SetThreadAffinityMask(GetCurrentThread(), prev);
        }
        return prev;
    };

    DWORD_PTR perf_mask = 0;
    DWORD_PTR eff_mask = 0;
    DWORD_PTR unrestricted_after_disable = 0;
    DWORD_PTR restored_after_enable = 0;

    std::thread worker([&] {
        Common::SetCurrentThreadToPerformanceCores();
        perf_mask = query_current_thread_mask();

        Common::SetCurrentThreadToEfficiencyCores();
        eff_mask = query_current_thread_mask();

        Settings::values.prefer_3d_vcache.SetValue(false);
        Common::RefreshThreadPolicies();
        unrestricted_after_disable = query_current_thread_mask();

        Settings::values.prefer_3d_vcache.SetValue(true);
        Common::RefreshThreadPolicies();
        restored_after_enable = query_current_thread_mask();
    });
    worker.join();

    Settings::values.prefer_3d_vcache.SetValue(prev_setting);
    Common::RefreshThreadPolicies();

    CHECK(perf_mask != 0);
    CHECK(eff_mask != 0);
    if (perf_mask != process_mask || eff_mask != process_mask) {
        CHECK((perf_mask & eff_mask) == 0);
        CHECK((perf_mask | eff_mask) == process_mask);
        CHECK(unrestricted_after_disable == process_mask);
        CHECK(restored_after_enable == eff_mask);
    }
#endif
}

TEST_CASE("Only selected recommendations are applied", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    const std::set<std::string> keys{"gpu_accuracy", "use_asynchronous_shaders"};
    REQUIRE(Settings::ApplyGamePreset(*settings, Settings::MarioKartWorldProgramId, true, &keys));
    CHECK(settings->gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK_FALSE(settings->gpu_accuracy.UsingGlobal());
    CHECK(settings->use_asynchronous_shaders.GetValue());
    CHECK(settings->renderer_backend.GetValue() == Settings::RendererBackend::Null);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->scaling_filter.GetValue() == Settings::ScalingFilter::NearestNeighbor);

    const std::set<std::string> none;
    CHECK_FALSE(Settings::ApplyGamePreset(*settings, Settings::MarioKartWorldProgramId, true, &none));
}

TEST_CASE("Game tweaks refer to valid per-game settings", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    for (const auto& [program_id, keys] : Settings::game_tweaks) {
        INFO(program_id);
        REQUIRE(Settings::FindGameTweaks(program_id) == &keys);
        for (const auto& key : keys) {
            INFO(key);
            const auto it = settings->linkage.by_key.find(key);
            REQUIRE(it != settings->linkage.by_key.end());
            CHECK(it->second->Switchable());
            CHECK(Settings::IsGameTweak(key));
        }
    }
    CHECK_FALSE(Settings::IsGameTweak("resolution_setup"));
    CHECK(Settings::FindGameTweaks(0) == nullptr);
}
