// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "input_common/helpers/mouse_mode.h"

using namespace InputCommon;
using Catch::Approx;

TEST_CASE("Joy-Con 2 mouse sensitivity is clamped to the supported range", "[input_common][mouse]") {
    CHECK(MouseMode::ClampSensitivity(100.0f) == Approx(1.0f));
    CHECK(MouseMode::ClampSensitivity(1.0f) == Approx(0.1f));
    CHECK(MouseMode::ClampSensitivity(1000.0f) == Approx(4.0f));
    CHECK(MouseMode::ClampSensitivity(250.0f) == Approx(2.5f));
}

TEST_CASE("Joy-Con 2 mouse scaling applies sensitivity and inversion", "[input_common][mouse]") {
    MouseMode::Config config{};
    config.sensitivity = 2.0f;
    CHECK((MouseMode::Scale(5, -3, config) == MouseMode::Delta{10.0f, -6.0f}));

    config.invert_y = true;
    CHECK((MouseMode::Scale(5, -3, config) == MouseMode::Delta{10.0f, 6.0f}));

    config.invert_x = true;
    config.invert_y = false;
    CHECK((MouseMode::Scale(5, -3, config) == MouseMode::Delta{-10.0f, -6.0f}));
}

TEST_CASE("Joy-Con 2 mouse deadzone drops jitter but keeps real motion", "[input_common][mouse]") {
    MouseMode::Config config{};
    config.deadzone_pixels = 2.0f;
    CHECK((MouseMode::Scale(1, -2, config) == MouseMode::Delta{0.0f, 0.0f}));
    CHECK((MouseMode::Scale(3, 0, config) == MouseMode::Delta{3.0f, 0.0f}));
}

TEST_CASE("Joy-Con 2 mouse accumulator clamps each report", "[input_common][mouse]") {
    MouseMode::Accumulator accumulator;
    accumulator.Add({1500.0f, -100.0f});
    accumulator.Add({1500.0f, 0.0f});
    CHECK((accumulator.Pending() == MouseMode::Delta{2047.0f, -100.0f}));
    CHECK((accumulator.Consume() == MouseMode::Delta{2047.0f, -100.0f}));
    CHECK((accumulator.Pending() == MouseMode::Delta{0.0f, 0.0f}));

    accumulator.Add({-10.0f, 20.0f});
    accumulator.Reset();
    CHECK((accumulator.Pending() == MouseMode::Delta{0.0f, 0.0f}));
}

TEST_CASE("Joy-Con 2 mouse stick fallback pushes and decays", "[input_common][mouse]") {
    MouseMode::StickProjection stick;
    CHECK_FALSE(stick.Active());
    CHECK((stick.Value() == MouseMode::Delta{0.0f, 0.0f}));

    stick.Push({50.0f, -50.0f});
    CHECK(stick.Active());
    // Positive host dx moves the stick right; positive host dy (down) moves it down.
    CHECK(stick.Value() == MouseMode::Delta{1.0f, 1.0f});

    stick.Decay(100.0f);
    CHECK(stick.Value().x < 1.0f);
    CHECK(stick.Value().x > 0.0f);

    // A full second of decay at 12/s leaves essentially nothing.
    stick.Decay(1000.0f);
    CHECK_FALSE(stick.Active());
    CHECK((stick.Value() == MouseMode::Delta{0.0f, 0.0f}));
}
