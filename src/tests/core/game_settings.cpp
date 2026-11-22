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
