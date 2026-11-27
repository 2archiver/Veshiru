// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <ranges>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "input_common/helpers/wii_remote_protocol.h"

using namespace InputCommon;

TEST_CASE("Wii Remote core buttons decode from the button report", "[input_common][wii]") {
    // Up + Plus on the left byte, A + One on the right byte.
    const std::array<u8, 3> report{0x30, 0x18, 0x0A};
    const auto parsed = Wii::ParseInputReport(report);
    REQUIRE(parsed.has_value());
    CHECK(parsed->has_buttons);
    CHECK(parsed->buttons.up);
    CHECK(parsed->buttons.plus);
    CHECK(parsed->buttons.a);
    CHECK(parsed->buttons.one);
    CHECK_FALSE(parsed->buttons.left);
    CHECK_FALSE(parsed->buttons.b);
    CHECK_FALSE(parsed->buttons.home);
}

TEST_CASE("Wii Remote accelerometer low bits ride in the button bytes", "[input_common][wii]") {
    // 0x31: buttons lo = 0xC0, hi = 0x60, accel msb = 0x7F/0x80/0x81.
    const std::array<u8, 6> report{0x31, 0xC0, 0x60, 0x7F, 0x80, 0x81};
    const auto parsed = Wii::ParseInputReport(report);
    REQUIRE(parsed.has_value());
    CHECK(parsed->has_accel);
    CHECK(parsed->accel.x == 0x1FF);
    CHECK(parsed->accel.y == 0x202);
    CHECK(parsed->accel.z == 0x206);

    const auto g = Wii::AccelToG(parsed->accel);
    CHECK(g[0] < 0.0f);
    CHECK(g[1] > 0.0f);
    CHECK(g[2] > g[1]);
}

TEST_CASE("Wii Remote basic IR mode averages the visible dots", "[input_common][wii]") {
    // Two visible dots at (100, 200) and (132, 88); the second pair is fully invisible.
    std::array<u8, 10> ir{};
    ir[0] = 100 & 0xFF;
    ir[1] = 200 & 0xFF;
    ir[2] = 0; // All high bits of the first pair clear: both coordinates are below 256.
    ir[3] = 132;
    ir[4] = 88;
    // Second pair: 0x3FF in every coordinate means "no object".
    ir[5] = 0xFF;
    ir[6] = 0xFF;
    ir[7] = 0xFF;
    ir[8] = 0xFF;
    ir[9] = 0xFF;

    const auto frame = Wii::ParseBasicIr(ir);
    CHECK(frame.visible_dots == 2);
    REQUIRE(frame.pointer_valid);
    CHECK(frame.dots[0].x == 100);
    CHECK(frame.dots[0].y == 200);
    CHECK(frame.dots[2].visible == false);
    CHECK(frame.pointer_x == Catch::Approx(116.0f / 1023.0f));
    CHECK(frame.pointer_y == Catch::Approx(144.0f / 767.0f));
}

TEST_CASE("Wii Remote nunchuk decodes stick and active-low buttons", "[input_common][wii]") {
    // Stick at (255, 0) and both C and Z held: bits 0 and 1 of the last byte cleared.
    const std::array<u8, 6> ext{0xFF, 0x00, 0x80, 0x80, 0x80, 0x00};
    const auto nunchuk = Wii::ParseNunchuk(ext);
    CHECK(nunchuk.connected);
    CHECK(nunchuk.stick_x == Catch::Approx(1.0f));
    CHECK(nunchuk.stick_y == Catch::Approx(1.0f));
    CHECK(nunchuk.c);
    CHECK(nunchuk.z);

    const std::array<u8, 6> released{0x80, 0x80, 0x80, 0x80, 0x80, 0x03};
    const auto idle = Wii::ParseNunchuk(released);
    CHECK_FALSE(idle.c);
    CHECK_FALSE(idle.z);
    CHECK(idle.stick_x == Catch::Approx(0.0f));
}

TEST_CASE("Wii Remote only treats recognized Nunchuk extension bytes as Nunchuk input",
          "[input_common][wii]") {
    const std::array<u8, 6> nunchuk_id{0x00, 0x00, 0xA4, 0x20, 0x00, 0x00};
    const std::array<u8, 6> classic_id{0x00, 0x00, 0xA4, 0x20, 0x01, 0x01};
    const auto nunchuk = Wii::ParseExtensionId(nunchuk_id);
    const auto classic = Wii::ParseExtensionId(classic_id);
    Wii::InputReport report{};
    report.has_nunchuk = true;
    report.nunchuk.connected = true;

    CHECK(Wii::HasNunchukData(nunchuk, report));
    CHECK_FALSE(Wii::HasNunchukData(classic, report));

    report.nunchuk.connected = false;
    CHECK_FALSE(Wii::HasNunchukData(nunchuk, report));
    report.nunchuk.connected = true;
    report.has_nunchuk = false;
    CHECK_FALSE(Wii::HasNunchukData(nunchuk, report));
}

TEST_CASE("Wii Remote status reports battery, LEDs and extension presence", "[input_common][wii]") {
    const std::array<u8, 7> report{0x20, 0x00, 0x00, 0x1A, 0x00, 0x00, 0xC8};
    const auto parsed = Wii::ParseInputReport(report);
    REQUIRE(parsed.has_value());
    CHECK(parsed->has_status);
    CHECK(parsed->status.extension_connected);
    CHECK(parsed->status.ir_enabled);
    CHECK_FALSE(parsed->status.battery_low);
    CHECK(parsed->status.leds == 0x1);
    CHECK(parsed->status.battery_level == 0xC8);
}

TEST_CASE("Wii Remote extension ID read reply is recognised", "[input_common][wii]") {
    std::array<u8, 16> report{};
    report[0] = 0x21;
    report[3] = 0x50; // 6 bytes
    report[4] = 0x00;
    report[5] = 0xFA;
    const std::array<u8, 6> nunchuk_id{0x00, 0x00, 0xA4, 0x20, 0x00, 0x00};
    std::ranges::copy(nunchuk_id, report.begin() + 6);

    const auto extension = Wii::ParseExtensionIdResponse(report);
    REQUIRE(extension.has_value());
    CHECK(extension->nunchuk);
    CHECK(extension->valid);
    CHECK(extension->value == 0x0000A4200000ULL);
}

TEST_CASE("Wii Remote output reports carry the expected command bytes", "[input_common][wii]") {
    const auto leds = Wii::BuildLedReport(0x3, true);
    CHECK(leds[0] == 0x11);
    CHECK(leds[1] == 0x31);

    const auto reporting = Wii::BuildDataReportingReport(
        Wii::ReportId::ButtonsAccelIr10Extension6, true, false);
    CHECK(reporting[0] == 0x12);
    CHECK(reporting[1] == 0x04);
    CHECK(reporting[2] == 0x37);

    const auto read_id = Wii::BuildReadExtensionIdReport(false);
    CHECK(read_id[0] == 0x17);
    CHECK(read_id[2] == 0xA4);
    CHECK(read_id[4] == 0xFA);
    CHECK(read_id[6] == 0x06);

    const std::array<u8, 2> payload{0xB3, 0x04};
    const auto write = Wii::BuildWriteRegisterReport(0xB0001A, payload, false);
    CHECK(write[0] == 0x16);
    CHECK(write[2] == 0xB0);
    CHECK(write[3] == 0x00);
    CHECK(write[4] == 0x1A);
    CHECK(write[5] == 0x02);
    CHECK(write[6] == 0xB3);
    CHECK(write[7] == 0x04);
}
