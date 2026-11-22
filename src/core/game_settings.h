// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace VideoCore { class RendererBase; }

namespace Core::GameSettings {

enum class OS {
    Windows,
    Linux,
    MacOS,
    IOS,
    Android,
    FireOS,
    HarmonyOS,
    FreeBSD,
    DragonFlyBSD,
    NetBSD,
    OpenBSD,
    HaikuOS,
    AIX,
    Managarm,
    RedoxOS,
    Solaris,
    Unknown,
};

enum class GPUVendor {
    Nvidia,
    AMD,
    Intel,
    Apple,
    Qualcomm,
    ARM,
    Imagination,
    Microsoft,
    Unknown,
};

enum class TitleID : std::uint64_t {
    NinjaGaidenRagebound = 0x0100781020710000ULL,
    DuskbloodsNetworkTest = 0x0400D1D02399A000ULL,
    DonkeyKongBananza = 0x040074A01BF12000ULL,
    StarFoxSwitch2 = 0x0400E88025B1A000ULL,
    OcarinaOfTime = 0x0400486024DEC000ULL,
};

struct EnvironmentInfo {
    OS os{OS::Unknown};
    GPUVendor vendor{GPUVendor::Unknown};
    std::string vendor_string; // raw string from driver
};

EnvironmentInfo DetectEnvironment(const VideoCore::RendererBase& renderer);

/// Titles that pace their own frames by the guest clock (CNTPCT) rather than by a host timer.
/// "CPU Clock: Boost" scales that guest clock 1.7x ahead of wall time (Core::Timing::
/// GetClockTicks), and their 60 Hz scenes then present about 39 times a second; Donkey Kong
/// Bananza also measures its own CPU frames over budget and falls back to swap interval 2 (30
/// FPS). NeedsGuestClockPacing names the titles, and LoadOverrides holds their clock at Normal
/// for the session when the setting is inherited from the global profile.
[[nodiscard]] bool NeedsGuestClockPacing(std::uint64_t program_id) noexcept;

/// Whether such a title's session holds the guest clock at Normal: it paces itself by the clock,
/// the clock is inherited from the global profile, and that clock is Boost. A clock set for the
/// game itself (Overclock included) is the user's choice and is left alone.
[[nodiscard]] constexpr bool HoldsClockAtNormal(bool paces_by_guest_clock, bool clock_inherited,
                                                bool clock_is_boost) noexcept {
    return paces_by_guest_clock && clock_inherited && clock_is_boost;
}

/// Whether the loaded title runs this session with the guest clock held at Normal. Read by
/// Core::Timing::GetClockTicks on the emulation threads; false until LoadOverrides ran.
[[nodiscard]] bool CpuClockHeldAtNormal() noexcept;

/// Titles whose first visit to a busy area still builds dozens of pipelines on demand. On a 4 GB
/// card that is one freeze per build (56 pipelines and 5.8 s of held draws in a single 10 s
/// window in Ocarina of Time's logs), the dispatches whose pipeline is still building are skipped
/// (a billboard drawn a frame late looks like the fairy flickering), and the title reads
/// GPU-written memory back on the CPU every frame. Ocarina of Time is the one measured.
[[nodiscard]] bool WantsStabilityProfile(std::uint64_t program_id) noexcept;

/// What the per-title stability profile turns on for a session. Each switch is taken only where
/// the user's own configuration does not already provide it, and none of them is written to disk.
struct StabilityProfile {
    /// Build every cached pipeline while the game loads, and keep it built.
    bool build_all_cached_shaders{};
    /// CPU reads of GPU-written memory see the GPU's data (no stale transforms).
    bool reactive_flushing{};
    /// Writes the GPU engines make inside a buffer invalidate the CPU's view of it.
    bool keep_inner_invalidation{};

    [[nodiscard]] bool Any() const noexcept {
        return build_all_cached_shaders || reactive_flushing || keep_inner_invalidation;
    }
};

/// Resolves the profile for a title. `enabled` is the session switch (EDEN_NX2_OOT_PROFILE), and
/// each `*_missing` says the user's configuration does not already do that part.
[[nodiscard]] constexpr StabilityProfile ResolveStabilityProfile(bool title_wants_profile,
                                                                 bool build_all_missing,
                                                                 bool flushing_missing,
                                                                 bool inner_invalidation_missing,
                                                                 bool enabled) noexcept {
    if (!title_wants_profile || !enabled) {
        return {};
    }
    return {
        .build_all_cached_shaders = build_all_missing,
        .reactive_flushing = flushing_missing,
        .keep_inner_invalidation = inner_invalidation_missing,
    };
}

void LoadOverrides(std::uint64_t program_id, const VideoCore::RendererBase& renderer);

} // namespace Core::GameSettings
