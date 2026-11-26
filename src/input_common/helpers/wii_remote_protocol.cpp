// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "input_common/helpers/wii_remote_protocol.h"

#include <algorithm>
#include <cmath>

namespace InputCommon::Wii {

namespace {

constexpr u16 IrInvisible = 0x3FF;
constexpr u16 IrWidth = 1024;
constexpr u16 IrHeight = 768;

constexpr u64 ExtensionNunchuk = 0x0000A4200000ULL;
constexpr u64 ExtensionClassic = 0x0000A4200101ULL;
constexpr u64 ExtensionMotionPlus = 0x0000A4200405ULL;
constexpr u64 ExtensionMotionPlusAlt = 0x0000A4200505ULL;

constexpr float StickScale = 1.0f / 127.0f;
constexpr float AccelZero = 512.0f;
constexpr float AccelCountsPerG = 170.0f;

[[nodiscard]] bool Invisible(u16 x, u16 y) noexcept {
    return x == IrInvisible && y == IrInvisible;
}

/// Reads one extended-mode dot (3 bytes): X and Y are 10 bits, the low nibble of the third byte is
/// the blob size.
[[nodiscard]] IrDot ReadExtendedDot(std::span<const u8> bytes) {
    IrDot dot{};
    dot.x = static_cast<u16>(bytes[0] | ((bytes[2] & 0x30) << 4));
    dot.y = static_cast<u16>(bytes[1] | ((bytes[2] & 0xC0) << 2));
    dot.size = static_cast<u8>(bytes[2] & 0x0F);
    dot.visible = !Invisible(dot.x, dot.y);
    return dot;
}

/// Basic mode packs two dots into five bytes.
void ReadBasicPair(std::span<const u8> bytes, IrDot& first, IrDot& second) {
    first.x = static_cast<u16>(bytes[0] | (((bytes[2] >> 4) & 0x03) << 8));
    first.y = static_cast<u16>(bytes[1] | (((bytes[2] >> 6) & 0x03) << 8));
    second.x = static_cast<u16>(bytes[3] | ((bytes[2] & 0x03) << 8));
    second.y = static_cast<u16>(bytes[4] | (((bytes[2] >> 2) & 0x03) << 8));
    first.visible = !Invisible(first.x, first.y);
    second.visible = !Invisible(second.x, second.y);
}

void FinishIrFrame(IrFrame& frame) {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    u32 visible = 0;
    for (const IrDot& dot : frame.dots) {
        if (!dot.visible) {
            continue;
        }
        sum_x += static_cast<float>(dot.x);
        sum_y += static_cast<float>(dot.y);
        ++visible;
    }
    frame.visible_dots = static_cast<u8>(visible);
    if (visible == 0) {
        frame.pointer_valid = false;
        return;
    }
    const float x = sum_x / static_cast<float>(visible);
    const float y = sum_y / static_cast<float>(visible);
    frame.pointer_x = std::clamp(x / static_cast<float>(IrWidth - 1), 0.0f, 1.0f);
    frame.pointer_y = std::clamp(y / static_cast<float>(IrHeight - 1), 0.0f, 1.0f);
    frame.pointer_valid = true;
}

} // namespace

std::string_view GetButtonName(Button button) noexcept {
    switch (button) {
    case Button::A:
        return "A";
    case Button::B:
        return "B";
    case Button::One:
        return "1";
    case Button::Two:
        return "2";
    case Button::Plus:
        return "+";
    case Button::Minus:
        return "-";
    case Button::Home:
        return "Home";
    case Button::Left:
        return "Left";
    case Button::Right:
        return "Right";
    case Button::Up:
        return "Up";
    case Button::Down:
        return "Down";
    case Button::NunchukC:
        return "Nunchuk C";
    case Button::NunchukZ:
        return "Nunchuk Z";
    case Button::Count:
        break;
    }
    return "Unknown";
}

bool CoreButtons::Any() const noexcept {
    return a || b || one || two || plus || minus || home || left || right || up || down;
}

CoreButtons ParseCoreButtons(std::span<const u8> report) {
    CoreButtons buttons{};
    if (report.size() < 3) {
        return buttons;
    }
    const u8 lo = report[1];
    const u8 hi = report[2];
    buttons.raw = static_cast<u16>((static_cast<u16>(hi) << 8) | lo);
    buttons.left = (lo & 0x01) != 0;
    buttons.right = (lo & 0x02) != 0;
    buttons.down = (lo & 0x04) != 0;
    buttons.up = (lo & 0x08) != 0;
    buttons.plus = (lo & 0x10) != 0;
    buttons.two = (hi & 0x01) != 0;
    buttons.one = (hi & 0x02) != 0;
    buttons.b = (hi & 0x04) != 0;
    buttons.a = (hi & 0x08) != 0;
    buttons.minus = (hi & 0x10) != 0;
    buttons.home = (hi & 0x80) != 0;
    return buttons;
}

Accel ParseAccel(std::span<const u8> accel_bytes, u8 buttons_lo, u8 buttons_hi) {
    Accel accel{};
    if (accel_bytes.size() < 3) {
        return accel;
    }
    // The unused button bits carry the low bits of the accelerometer sample.
    accel.x = static_cast<u16>((static_cast<u16>(accel_bytes[0]) << 2) |
                               ((buttons_lo >> 6) & 0x03));
    accel.y = static_cast<u16>((static_cast<u16>(accel_bytes[1]) << 2) |
                               (((buttons_hi >> 5) & 0x01) << 1));
    accel.z = static_cast<u16>((static_cast<u16>(accel_bytes[2]) << 2) |
                               (((buttons_hi >> 6) & 0x01) << 1));
    return accel;
}

IrFrame ParseBasicIr(std::span<const u8> ir_bytes) {
    IrFrame frame{};
    if (ir_bytes.size() < 10) {
        return frame;
    }
    // Two objects per five bytes, four objects total.
    ReadBasicPair(ir_bytes.subspan(0, 5), frame.dots[0], frame.dots[1]);
    ReadBasicPair(ir_bytes.subspan(5, 5), frame.dots[2], frame.dots[3]);
    FinishIrFrame(frame);
    return frame;
}

IrFrame ParseExtendedIr(std::span<const u8> ir_bytes) {
    IrFrame frame{};
    if (ir_bytes.size() < 12) {
        return frame;
    }
    for (std::size_t i = 0; i < frame.dots.size(); ++i) {
        frame.dots[i] = ReadExtendedDot(ir_bytes.subspan(i * 3, 3));
    }
    FinishIrFrame(frame);
    return frame;
}

Nunchuk ParseNunchuk(std::span<const u8> extension_bytes) {
    Nunchuk nunchuk{};
    if (extension_bytes.size() < 6) {
        return nunchuk;
    }
    nunchuk.connected = true;
    nunchuk.stick_x =
        std::clamp((static_cast<float>(extension_bytes[0]) - 128.0f) * StickScale, -1.0f, 1.0f);
    nunchuk.stick_y =
        std::clamp((128.0f - static_cast<float>(extension_bytes[1])) * StickScale, -1.0f, 1.0f);
    nunchuk.accel.x = static_cast<u16>(extension_bytes[2]) << 2;
    nunchuk.accel.y = static_cast<u16>(extension_bytes[3]) << 2;
    nunchuk.accel.z = static_cast<u16>(
        (static_cast<u16>(extension_bytes[4]) << 2) | ((extension_bytes[5] >> 2) & 0x03));
    // C and Z are active low.
    nunchuk.z = (extension_bytes[5] & 0x01) == 0;
    nunchuk.c = (extension_bytes[5] & 0x02) == 0;
    return nunchuk;
}

Status ParseStatus(std::span<const u8> report) {
    Status status{};
    if (report.size() < 4) {
        return status;
    }
    const u8 flags = report[3];
    status.battery_low = (flags & 0x01) != 0;
    status.extension_connected = (flags & 0x02) != 0;
    status.speaker_enabled = (flags & 0x04) != 0;
    status.ir_enabled = (flags & 0x08) != 0;
    status.leds = static_cast<u8>(flags >> 4);
    if (report.size() >= 7) {
        status.battery_level = report[6];
    }
    return status;
}

ExtensionId ParseExtensionId(std::span<const u8> id_bytes) {
    ExtensionId id{};
    if (id_bytes.size() < 6) {
        return id;
    }
    id.valid = true;
    for (std::size_t i = 0; i < 6; ++i) {
        id.value = (id.value << 8) | id_bytes[i];
    }
    id.nunchuk = id.value == ExtensionNunchuk;
    id.classic_controller = id.value == ExtensionClassic;
    id.motion_plus =
        id.value == ExtensionMotionPlus || id.value == ExtensionMotionPlusAlt;
    return id;
}

std::optional<ExtensionId> ParseExtensionIdResponse(std::span<const u8> report) {
    if (report.size() < 12 || report[0] != static_cast<u8>(ReportId::ReadData)) {
        return std::nullopt;
    }
    const u16 offset = static_cast<u16>((static_cast<u16>(report[4]) << 8) | report[5]);
    if (offset != 0x00FA) {
        return std::nullopt;
    }
    const std::size_t size = static_cast<std::size_t>(report[3] >> 4) + 1;
    if (size < 6 || report.size() < 6 + size) {
        return std::nullopt;
    }
    return ParseExtensionId(report.subspan(6, 6));
}

std::optional<InputReport> ParseInputReport(std::span<const u8> report) {
    if (report.empty()) {
        return std::nullopt;
    }

    InputReport out{};
    out.id = static_cast<ReportId>(report[0]);

    switch (out.id) {
    case ReportId::Status:
        if (report.size() < 4) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.status = ParseStatus(report);
        out.has_status = true;
        return out;
    case ReportId::ReadData:
        if (const auto extension = ParseExtensionIdResponse(report); extension.has_value()) {
            out.extension = *extension;
            out.has_extension = true;
            return out;
        }
        return std::nullopt;
    case ReportId::Acknowledge:
        return std::nullopt;
    case ReportId::Buttons:
        if (report.size() < 3) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        return out;
    case ReportId::ButtonsAccel:
        if (report.size() < 6) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.accel = ParseAccel(report.subspan(3, 3), report[1], report[2]);
        out.has_accel = true;
        return out;
    case ReportId::ButtonsExtension8:
        if (report.size() < 11) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.nunchuk = ParseNunchuk(report.subspan(3, 6));
        out.has_nunchuk = out.nunchuk.connected;
        return out;
    case ReportId::ButtonsAccelIr12:
        if (report.size() < 18) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.accel = ParseAccel(report.subspan(3, 3), report[1], report[2]);
        out.has_accel = true;
        out.ir = ParseExtendedIr(report.subspan(6, 12));
        out.has_ir = true;
        return out;
    case ReportId::ButtonsExtension19:
        if (report.size() < 22) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.nunchuk = ParseNunchuk(report.subspan(3, 6));
        out.has_nunchuk = out.nunchuk.connected;
        return out;
    case ReportId::ButtonsAccelExtension16:
        if (report.size() < 22) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.accel = ParseAccel(report.subspan(3, 3), report[1], report[2]);
        out.has_accel = true;
        out.nunchuk = ParseNunchuk(report.subspan(6, 6));
        out.has_nunchuk = out.nunchuk.connected;
        return out;
    case ReportId::ButtonsIr10Extension9:
        if (report.size() < 22) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.ir = ParseBasicIr(report.subspan(3, 10));
        out.has_ir = true;
        out.nunchuk = ParseNunchuk(report.subspan(13, 6));
        out.has_nunchuk = out.nunchuk.connected;
        return out;
    case ReportId::ButtonsAccelIr10Extension6:
        if (report.size() < 22) {
            return std::nullopt;
        }
        out.buttons = ParseCoreButtons(report);
        out.has_buttons = true;
        out.accel = ParseAccel(report.subspan(3, 3), report[1], report[2]);
        out.has_accel = true;
        out.ir = ParseBasicIr(report.subspan(6, 10));
        out.has_ir = true;
        out.nunchuk = ParseNunchuk(report.subspan(16, 6));
        out.has_nunchuk = out.nunchuk.connected;
        return out;
    case ReportId::ExtensionOnly:
        if (report.size() < 7) {
            return std::nullopt;
        }
        out.nunchuk = ParseNunchuk(report.subspan(1, 6));
        out.has_nunchuk = out.nunchuk.connected;
        return out;
    case ReportId::InterleavedFirst:
    case ReportId::InterleavedSecond:
        // Pointer tracking does not need the interleaved mode; the driver never selects it.
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

std::array<u8, 2> BuildLedReport(u8 leds, bool rumble) {
    return {0x11, static_cast<u8>(((leds & 0x0F) << 4) | (rumble ? 0x01 : 0x00))};
}

std::array<u8, 2> BuildRumbleOnlyReport(bool rumble) {
    return {0x10, static_cast<u8>(rumble ? 0x01 : 0x00)};
}

std::array<u8, 3> BuildDataReportingReport(ReportId mode, bool continuous, bool rumble) {
    const u8 flags = static_cast<u8>((continuous ? 0x04 : 0x00) | (rumble ? 0x01 : 0x00));
    return {0x12, flags, static_cast<u8>(mode)};
}

std::array<u8, 2> BuildStatusRequest(bool rumble) {
    return {0x15, static_cast<u8>(rumble ? 0x01 : 0x00)};
}

std::array<u8, 7> BuildReadExtensionIdReport(bool rumble) {
    // Read register 0xA400FA (extension ID), 6 bytes.
    return {0x17, static_cast<u8>(0x04 | (rumble ? 0x01 : 0x00)), 0xA4, 0x00, 0xFA, 0x00, 0x06};
}

std::array<u8, 2> BuildIrEnableReport(bool second_block, bool enable) {
    return {static_cast<u8>(second_block ? 0x1A : 0x13),
            static_cast<u8>(enable ? 0x04 : 0x00)};
}

std::array<u8, 22> BuildWriteRegisterReport(u32 address, std::span<const u8> data, bool rumble) {
    std::array<u8, 22> report{};
    report[0] = 0x16;
    report[1] = static_cast<u8>(0x04 | (rumble ? 0x01 : 0x00));
    report[2] = static_cast<u8>(address >> 16);
    report[3] = static_cast<u8>(address >> 8);
    report[4] = static_cast<u8>(address);
    const std::size_t size = std::min<std::size_t>(data.size(), 16);
    report[5] = static_cast<u8>(size);
    std::copy_n(data.begin(), size, report.begin() + 6);
    return report;
}

u8 LedPatternForPlayer(std::size_t player_index) noexcept {
    switch (player_index) {
    case 1:
        return 0x2;
    case 2:
        return 0x4;
    case 3:
        return 0x8;
    case 0:
    default:
        return 0x1;
    }
}

std::array<float, 3> AccelToG(const Accel& accel) noexcept {
    return {
        (static_cast<float>(accel.x) - AccelZero) / AccelCountsPerG,
        (static_cast<float>(accel.y) - AccelZero) / AccelCountsPerG,
        (static_cast<float>(accel.z) - AccelZero) / AccelCountsPerG,
    };
}

bool IsSupportedExtension(const ExtensionId& extension) noexcept {
    return extension.valid && (extension.nunchuk || extension.classic_controller);
}

} // namespace InputCommon::Wii
