// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "core/game_settings.h"

TEST_CASE("Game settings: guest-clock-paced titles are named", "[core][game_settings]") {
    // Donkey Kong Bananza and Star Fox (Switch 2) pace their frames by the guest clock.
    constexpr std::uint64_t bananza = 0x040074A01BF12000ULL;
    constexpr std::uint64_t star_fox = 0x0400E88025B1A000ULL;
    CHECK(Core::GameSettings::NeedsGuestClockPacing(bananza));
    CHECK(Core::GameSettings::NeedsGuestClockPacing(star_fox));
    // Mario Kart World, Ocarina of Time and the HOME Menu do not.
    CHECK_FALSE(Core::GameSettings::NeedsGuestClockPacing(0x0400C3F00006E000ULL));
    CHECK_FALSE(Core::GameSettings::NeedsGuestClockPacing(0x0400486024DEC000ULL));
    CHECK_FALSE(Core::GameSettings::NeedsGuestClockPacing(0x0100000000001000ULL));
    CHECK_FALSE(Core::GameSettings::NeedsGuestClockPacing(0));
}

TEST_CASE("Game settings: the clock is held at Normal only for an inherited Boost clock",
          "[core][game_settings]") {
    using Core::GameSettings::HoldsClockAtNormal;
    // Paced title, Boost inherited from the global profile: held at Normal.
    CHECK(HoldsClockAtNormal(true, true, true));
    // A clock chosen for the game itself, Boost or Overclock, is left alone.
    CHECK_FALSE(HoldsClockAtNormal(true, false, true));
    // Normal needs no hold, and other titles keep whatever clock they inherited.
    CHECK_FALSE(HoldsClockAtNormal(true, true, false));
    CHECK_FALSE(HoldsClockAtNormal(false, true, true));
    CHECK_FALSE(HoldsClockAtNormal(false, false, false));
}

TEST_CASE("Game settings: the stability profile names its titles", "[core][game_settings]") {
    using Core::GameSettings::WantsStabilityProfile;
    // Ocarina of Time: on-demand pipeline builds, skipped dispatches and per-frame CPU reads of
    // GPU-written memory (the fairy's flicker) on the 4 GB card.
    CHECK(WantsStabilityProfile(0x0400486024DEC000ULL));
    // Nothing else takes it, and neither does an empty title id.
    CHECK_FALSE(WantsStabilityProfile(0x0400C3F00006E000ULL));
    CHECK_FALSE(WantsStabilityProfile(0x040074A01BF12000ULL));
    CHECK_FALSE(WantsStabilityProfile(0x0100000000001000ULL));
    CHECK_FALSE(WantsStabilityProfile(0));
}

TEST_CASE("Game settings: the stability profile only fills in what the user left alone",
          "[core][game_settings]") {
    using Core::GameSettings::ResolveStabilityProfile;
    // Nothing missing: the profile is empty even for the title that wants it.
    const auto nothing_missing = ResolveStabilityProfile(true, false, false, false, true);
    CHECK_FALSE(nothing_missing.Any());
    CHECK_FALSE(nothing_missing.build_all_cached_shaders);
    CHECK_FALSE(nothing_missing.reactive_flushing);
    CHECK_FALSE(nothing_missing.keep_inner_invalidation);
    // All three missing, profile enabled: all three are turned on.
    const auto all = ResolveStabilityProfile(true, true, true, true, true);
    CHECK(all.Any());
    CHECK(all.build_all_cached_shaders);
    CHECK(all.reactive_flushing);
    CHECK(all.keep_inner_invalidation);
    // Each part is independent.
    const auto shaders_only = ResolveStabilityProfile(true, true, false, false, true);
    CHECK(shaders_only.build_all_cached_shaders);
    CHECK_FALSE(shaders_only.reactive_flushing);
    CHECK_FALSE(shaders_only.keep_inner_invalidation);
    const auto flushing_only = ResolveStabilityProfile(true, false, true, true, true);
    CHECK_FALSE(flushing_only.build_all_cached_shaders);
    CHECK(flushing_only.reactive_flushing);
    CHECK(flushing_only.keep_inner_invalidation);
    // EDEN_NX2_OOT_PROFILE=0 (enabled = false) disables the whole profile for the session.
    CHECK_FALSE(ResolveStabilityProfile(true, true, true, true, false).Any());
    // A title that does not want it never gets it.
    CHECK_FALSE(ResolveStabilityProfile(false, true, true, true, true).Any());
}
