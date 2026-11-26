// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <SDL3/SDL_hidapi.h>

#include "common/common_types.h"
#include "common/param_package.h"
#include "common/uuid.h"
#include "input_common/helpers/wii_remote_protocol.h"
#include "input_common/input_engine.h"

namespace InputCommon {

/**
 * A Bluetooth Wii Remote (RVL-CNT-01 and RVL-CNT-01-TR), read through SDL's hidapi.
 *
 * Pairing is done by the operating system (Windows: Settings > Bluetooth > Add device, then press
 * 1+2 on the remote; Linux: `bluetoothctl`). Once paired, this driver enumerates the HID
 * interface, enables continuous reporting and reads buttons, the accelerometer, the IR pointer and
 * a Nunchuk on the extension port. It does not emulate a Wii Remote itself; a keyboard or mouse
 * can be mapped to the emulated controller as usual.
 */
class WiiRemote final : public InputEngine {
public:
    explicit WiiRemote(std::string input_engine_);
    ~WiiRemote() override;

    void Setup();
    void Reset();

    Common::Input::DriverResult SetLeds(
        const PadIdentifier& identifier, const Common::Input::LedStatus& led_status) override;
    Common::Input::DriverResult SetVibration(
        const PadIdentifier& identifier,
        const Common::Input::VibrationStatus& vibration) override;
    bool IsVibrationEnabled(const PadIdentifier& identifier) override;

    std::vector<Common::ParamPackage> GetInputDevices() const override;
    ButtonMapping GetButtonMappingForDevice(const Common::ParamPackage& params) override;
    AnalogMapping GetAnalogMappingForDevice(const Common::ParamPackage& params) override;
    MotionMapping GetMotionMappingForDevice(const Common::ParamPackage& params) override;
    Common::Input::ButtonNames GetUIName(const Common::ParamPackage& params) const override;

private:
    struct Remote;

    void ScanLoop(std::stop_token stop_token);
    void Scan();
    void RegisterController(Remote& remote);
    void ReleaseController(Remote& remote);
    void Publish(Remote& remote, const Wii::InputReport& report);
    void PublishExtensions(Remote& remote);
    void PublishAxis(Remote& remote, int axis, f32 value);
    void PublishBattery(Remote& remote, Common::Input::BatteryLevel level);

    Common::ParamPackage MakeButtonPackage(const Remote& remote, int button) const;
    Common::ParamPackage MakeAnalogPackage(const Remote& remote, int axis_x, int axis_y) const;

    mutable std::mutex remotes_mutex;
    std::vector<std::unique_ptr<Remote>> remotes;
    std::jthread scan_thread;
    bool hid_initialized{};
};

} // namespace InputCommon
