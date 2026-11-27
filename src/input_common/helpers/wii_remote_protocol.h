// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <string_view>

#include "common/common_types.h"

/// Decoding and encoding of the Wii Remote's Bluetooth HID reports. This file has no platform or
/// SDL dependency so the report layouts can be unit tested without a physical remote.
/// Layout reference: https://wiibrew.org/wiki/Wiimote (data reporting, status, extension port).
namespace InputCommon::Wii {

/// Buttons exposed to the input system. Core buttons first, then Nunchuk, so indices stay stable
/// when an extension is hot-plugged.
enum class Button : u8 {
    A,
    B,
    One,
    Two,
    Plus,
    Minus,
    Home,
    Left,
    Right,
    Up,
    Down,
    NunchukC,
    NunchukZ,
    Count,
};

inline constexpr std::size_t ButtonCount = static_cast<std::size_t>(Button::Count);
[[nodiscard]] std::string_view GetButtonName(Button button) noexcept;

/// Axis indices published by the driver.
inline constexpr int NunchukStickX = 0;
inline constexpr int NunchukStickY = 1;
inline constexpr int IrPointerX = 2;
inline constexpr int IrPointerY = 3;
inline constexpr int AxisCount = 4;

struct CoreButtons {
    u16 raw{};
    bool a{};
    bool b{};
    bool one{};
    bool two{};
    bool plus{};
    bool minus{};
    bool home{};
    bool left{};
    bool right{};
    bool up{};
    bool down{};

    [[nodiscard]] bool Any() const noexcept;

    friend constexpr bool operator==(const CoreButtons&, const CoreButtons&) = default;
};

/// Raw 10-bit accelerometer sample. 0x200 is roughly zero g on every axis.
struct Accel {
    u16 x{};
    u16 y{};
    u16 z{};

    friend constexpr bool operator==(const Accel&, const Accel&) = default;
};

/// A single IR camera object. Empty slots report 0x3FF for both coordinates.
struct IrDot {
    bool visible{};
    u16 x{};
    u16 y{};
    u8 size{};

    friend constexpr bool operator==(const IrDot&, const IrDot&) = default;
};

struct IrFrame {
    std::array<IrDot, 4> dots{};
    u8 visible_dots{};
    /// Sensor Bar midpoint in normalised screen space ([0, 1], y down), when at least one dot was
    /// visible. Games that expect a pointer usually want this over the individual dots.
    float pointer_x{};
    float pointer_y{};
    bool pointer_valid{};
};

struct Nunchuk {
    bool connected{};
    bool c{};
    bool z{};
    /// Stick in [-1, 1], y positive up, like the emulated Switch sticks.
    float stick_x{};
    float stick_y{};
    Accel accel{};
};

struct Status {
    bool extension_connected{};
    bool battery_low{};
    bool speaker_enabled{};
    bool ir_enabled{};
    u8 leds{};
    u8 battery_level{};
};

/// Identity of the controller on the extension port.
struct ExtensionId {
    u64 value{};
    bool valid{};
    bool nunchuk{};
    bool classic_controller{};
    bool motion_plus{};
};

enum class ReportId : u8 {
    Status = 0x20,
    ReadData = 0x21,
    Acknowledge = 0x22,
    Buttons = 0x30,
    ButtonsAccel = 0x31,
    ButtonsExtension8 = 0x32,
    ButtonsAccelIr12 = 0x33,
    ButtonsExtension19 = 0x34,
    ButtonsAccelExtension16 = 0x35,
    ButtonsIr10Extension9 = 0x36,
    ButtonsAccelIr10Extension6 = 0x37,
    ExtensionOnly = 0x3D,
    InterleavedFirst = 0x3E,
    InterleavedSecond = 0x3F,
};

/// One decoded input report, as far as the emulator needs it. Fields not carried by the active
/// reporting mode keep their previous value in the driver.
struct InputReport {
    ReportId id{};
    CoreButtons buttons{};
    bool has_buttons{};

    Accel accel{};
    bool has_accel{};

    IrFrame ir{};
    bool has_ir{};

    Nunchuk nunchuk{};
    bool has_nunchuk{};

    Status status{};
    bool has_status{};

    ExtensionId extension{};
    bool has_extension{};
};

/// Extension bytes are only Nunchuk data when the remote has identified a Nunchuk; other
/// extensions also occupy six bytes in the same report slots.
[[nodiscard]] inline bool HasNunchukData(const ExtensionId& extension,
                                         const InputReport& report) noexcept {
    return extension.valid && extension.nunchuk && report.has_nunchuk && report.nunchuk.connected;
}

/// The largest payload the remote can send, including the report ID.
inline constexpr std::size_t MaxReportSize = 22;

[[nodiscard]] std::optional<InputReport> ParseInputReport(std::span<const u8> report);

[[nodiscard]] CoreButtons ParseCoreButtons(std::span<const u8> report);
[[nodiscard]] Accel ParseAccel(std::span<const u8> accel_bytes, u8 buttons_lo, u8 buttons_hi);
[[nodiscard]] IrFrame ParseBasicIr(std::span<const u8> ir_bytes);
[[nodiscard]] IrFrame ParseExtendedIr(std::span<const u8> ir_bytes);
[[nodiscard]] Nunchuk ParseNunchuk(std::span<const u8> extension_bytes);
[[nodiscard]] Status ParseStatus(std::span<const u8> report);
[[nodiscard]] ExtensionId ParseExtensionId(std::span<const u8> id_bytes);
/// Parses a 0x21 read response for the extension port ID (0xA400FA).
[[nodiscard]] std::optional<ExtensionId> ParseExtensionIdResponse(std::span<const u8> report);

// ---- Output reports ------------------------------------------------------------------------
// Every buffer starts with the report ID, as SDL_hid_write expects for a numbered report.

[[nodiscard]] std::array<u8, 2> BuildLedReport(u8 leds, bool rumble);
[[nodiscard]] std::array<u8, 2> BuildRumbleOnlyReport(bool rumble);
/// Mode is the input report ID to stream (0x31, 0x35, 0x37, ...).
[[nodiscard]] std::array<u8, 3> BuildDataReportingReport(ReportId mode, bool continuous,
                                                         bool rumble);
[[nodiscard]] std::array<u8, 2> BuildStatusRequest(bool rumble);
[[nodiscard]] std::array<u8, 7> BuildReadExtensionIdReport(bool rumble);
[[nodiscard]] std::array<u8, 2> BuildIrEnableReport(bool second_block, bool enable);
[[nodiscard]] std::array<u8, 22> BuildWriteRegisterReport(u32 address, std::span<const u8> data,
                                                          bool rumble);

/// The LED bit pattern for a 0-based player index (LED 1 is the player 1 LED).
[[nodiscard]] u8 LedPatternForPlayer(std::size_t player_index) noexcept;

/// Converts a raw accelerometer sample to g, assuming the nominal zero point and sensitivity of
/// the ADXL330 in a Wii Remote (0x200 at rest, about 170 counts per g).
[[nodiscard]] std::array<float, 3> AccelToG(const Accel& accel) noexcept;

/// True when the extension ID is one this driver understands and publishes.
[[nodiscard]] bool IsSupportedExtension(const ExtensionId& extension) noexcept;

} // namespace InputCommon::Wii
