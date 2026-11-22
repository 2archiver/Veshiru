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

void LoadOverrides(std::uint64_t program_id, const VideoCore::RendererBase& renderer);

} // namespace Core::GameSettings
