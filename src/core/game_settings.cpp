// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/game_settings.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_base.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace Core::GameSettings {

namespace {
// Set by LoadOverrides for the session; read from the emulation threads through
// GetClockTicks, so it is an atomic rather than a plain bool.
std::atomic_bool cpu_clock_held_at_normal{};
} // namespace

bool NeedsGuestClockPacing(std::uint64_t program_id) noexcept {
    switch (static_cast<TitleID>(program_id)) {
    case TitleID::DonkeyKongBananza:
    case TitleID::StarFoxSwitch2:
        return true;
    default:
        return false;
    }
}

bool CpuClockHeldAtNormal() noexcept {
    return cpu_clock_held_at_normal.load(std::memory_order_relaxed);
}

bool WantsStabilityProfile(std::uint64_t program_id) noexcept {
    return static_cast<TitleID>(program_id) == TitleID::OcarinaOfTime;
}

static GPUVendor GetGPU(const std::string& gpu_vendor_string) {
    struct Entry { const char* name; GPUVendor vendor; };
    static constexpr Entry GpuVendor[] = {
        // NVIDIA
        {"NVIDIA",   GPUVendor::Nvidia},
        {"Nouveau",  GPUVendor::Nvidia},
        {"NVK",      GPUVendor::Nvidia},
        {"Tegra",    GPUVendor::Nvidia},
        // AMD
        {"AMD",       GPUVendor::AMD},
        {"RadeonSI",  GPUVendor::AMD},
        {"RADV",      GPUVendor::AMD},
        {"AMDVLK",    GPUVendor::AMD},
        {"R600",      GPUVendor::AMD},
        // Intel
        {"Intel",     GPUVendor::Intel},
        {"ANV",       GPUVendor::Intel},
        {"i965",      GPUVendor::Intel},
        {"i915",      GPUVendor::Intel},
        {"OpenSWR",   GPUVendor::Intel},
        // Apple
        {"Apple",     GPUVendor::Apple},
        {"MoltenVK",  GPUVendor::Apple},
        // Qualcomm / Adreno
        {"Qualcomm",  GPUVendor::Qualcomm},
        {"Turnip",    GPUVendor::Qualcomm},
        // ARM / Mali
        {"Mali",      GPUVendor::ARM},
        {"PanVK",     GPUVendor::ARM},
        // Imagination / PowerVR
        {"PowerVR",   GPUVendor::Imagination},
        {"PVR",       GPUVendor::Imagination},
        // Microsoft / WARP / D3D12 GL
        {"D3D12",     GPUVendor::Microsoft},
        {"Microsoft", GPUVendor::Microsoft},
        {"WARP",      GPUVendor::Microsoft},
    };

    for (const auto& entry : GpuVendor) {
        if (gpu_vendor_string == entry.name) {
            return entry.vendor;
        }
    }

    // legacy (shouldn't be needed anymore, but just in case)
    std::string gpu = gpu_vendor_string;
    std::transform(gpu.begin(), gpu.end(), gpu.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    if (gpu.find("geforce") != std::string::npos) {
        return GPUVendor::Nvidia;
    }
    if (gpu.find("radeon") != std::string::npos || gpu.find("ati") != std::string::npos) {
        return GPUVendor::AMD;
    }

    return GPUVendor::Unknown;
}

static OS DetectOS() {
#if defined(_WIN32)
    return OS::Windows;
#elif defined(__FIREOS__)
    return OS::FireOS;
#elif defined(__ANDROID__)
    return OS::Android;
#elif defined(__OHOS__)
    return OS::HarmonyOS;
#elif defined(__HAIKU__)
    return OS::HaikuOS;
#elif defined(__DragonFly__)
    return OS::DragonFlyBSD;
#elif defined(__NetBSD__)
    return OS::NetBSD;
#elif defined(__OpenBSD__)
    return OS::OpenBSD;
#elif defined(_AIX)
    return OS::AIX;
#elif defined(__managarm__)
    return OS::Managarm;
#elif defined(__redox__)
    return OS::RedoxOS;
#elif defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    return OS::IOS;
#elif defined(__APPLE__)
    return OS::MacOS;
#elif defined(__FreeBSD__)
    return OS::FreeBSD;
#elif defined(__sun) && defined(__SVR4)
    return OS::Solaris;
#elif defined(__linux__)
    return OS::Linux;
#else
    return OS::Unknown;
#endif
}

EnvironmentInfo DetectEnvironment(const VideoCore::RendererBase& renderer) {
    EnvironmentInfo env{};
    env.os = DetectOS();
    env.vendor_string = renderer.GetDeviceVendor();
    env.vendor = GetGPU(env.vendor_string);
    return env;
}

void LoadOverrides(std::uint64_t program_id, const VideoCore::RendererBase& renderer) {
    const auto env = DetectEnvironment(renderer);

    Settings::values.force_reactive_flushing = false;
    Settings::values.keep_cpu_inner_invalidation = false;
    Settings::values.force_build_all_cached_shaders = false;

    // A guest-clock-paced title halves or worse its frame rate under an inherited Boost clock
    // (see NeedsGuestClockPacing). A clock the user set for this game is left alone.
    cpu_clock_held_at_normal.store(false, std::memory_order_relaxed);
    if (HoldsClockAtNormal(NeedsGuestClockPacing(program_id),
                           Settings::values.cpu_clock.UsingGlobal(),
                           Settings::values.cpu_clock.GetValue() == Settings::CpuClock::Boost)) {
        cpu_clock_held_at_normal.store(true, std::memory_order_relaxed);
        LOG_INFO(Core,
                 "Title {:016X} runs this session with CPU Clock at Normal: Boost paces its "
                 "guest clock ahead of wall time and lowers its frame rate",
                 program_id);
    }

    // The per-title stability profile (2.7). Ocarina of Time's logs on an i7-10750H + GTX 1650
    // 4 GB show the three things this address, all on the same first visit to a busy area:
    //   * 56 pipelines built on demand in one 10 s window, 5.8 s of draws held for them, worker
    //     builds of 0.2-2.1 s each, and single frames of 0.1-18.6 s;
    //   * compute dispatches whose pipeline is still building are skipped for the frame
    //     ("its dispatch waits" / SkipUnbuiltDispatch), which leaves a per-frame billboard where it
    //     was: the fairy's up-and-down flicker in the same windows;
    //   * 30-353 CPU reads of GPU-written memory per 10 s (5-7 ms average sync wait, 60-140 ms
    //     worst), so a CPU-side transform can be a frame stale.
    // Building the whole recorded cache at load removes the first two (nothing is left to build
    // or skip), and the flushing pair removes the third. Each switch is only taken where the
    // user's own configuration does not already provide it, and nothing is written to disk:
    // EDEN_NX2_OOT_PROFILE=0 turns the profile off for a session so it can be A/B'd on one machine.
    {
        static const bool profile_enabled = [] {
            const char* value = std::getenv("EDEN_NX2_OOT_PROFILE");
            return value == nullptr || value[0] != '0';
        }();
        const auto& build_all = Settings::values.build_all_cached_shaders;
        const auto profile = ResolveStabilityProfile(
            WantsStabilityProfile(program_id),
            build_all.UsingGlobal() && !build_all.GetValue(),
            !Settings::values.use_reactive_flushing.GetValue(),
            Settings::values.skip_cpu_inner_invalidation.GetValue(), profile_enabled);
        Settings::values.force_build_all_cached_shaders = profile.build_all_cached_shaders;
        Settings::values.force_reactive_flushing |= profile.reactive_flushing;
        Settings::values.keep_cpu_inner_invalidation |= profile.keep_inner_invalidation;
        if (profile.Any()) {
            std::string switches;
            const auto add = [&switches](bool on, const char* name) {
                if (!on) {
                    return;
                }
                if (!switches.empty()) {
                    switches += ", ";
                }
                switches += name;
            };
            add(profile.build_all_cached_shaders, "build_all_cached_shaders");
            add(profile.reactive_flushing, "reactive_flushing");
            add(profile.keep_inner_invalidation, "keep_cpu_inner_invalidation");
            LOG_INFO(Core,
                     "Title {:016X} runs this session with the stability profile: {} "
                     "(EDEN_NX2_OOT_PROFILE=0 disables it)",
                     program_id, switches);
        }
    }

    switch (static_cast<TitleID>(program_id)) {
        case TitleID::NinjaGaidenRagebound:
            Settings::values.use_squashed_iterated_blend = true;
            break;
        case TitleID::DuskbloodsNetworkTest:
            // The game reads values its GPU passes wrote (the tone pass's exposure among them)
            // back on the CPU. Without reactive flushing it reads stale ones and the whole picture
            // brightens and darkens every few frames.
            Settings::values.force_reactive_flushing = true;
            break;
        case TitleID::DonkeyKongBananza:
            // Gold pieces in the dirt read buffers that the GPU engines' inner writes update. With
            // "Skip CPU Inner Invalidation" those buffers keep stale data, the main pass writes NaN
            // there, and the bloom spreads it into large glows with black dots (#1418, #1739).
            Settings::values.keep_cpu_inner_invalidation = true;
            break;
        default:
            break;
    }

    LOG_INFO(Core, "Applied game settings for title ID {:016X} on OS {}, GPU vendor {} ({})",
             program_id,
             static_cast<int>(env.os),
             static_cast<int>(env.vendor),
             env.vendor_string);
}

} // namespace Core::GameSettings
