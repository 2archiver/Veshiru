// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "hid_core/hid_types.h"
#include "hid_core/resources/controller_base.h"

namespace Core::HID {
class HIDCore;
class EmulatedDevices;
} // namespace Core::HID

namespace Service::HID {
class Mouse final : public ControllerBase {
public:
    explicit Mouse(Core::HID::HIDCore& hid_core_);
    ~Mouse() override;

    // Called when the controller is initialized
    void OnInit() override;

    // When the controller is released
    void OnRelease() override;

    // When the controller is requesting an update for the shared memory
    void OnUpdate(const Core::Timing::CoreTiming& core_timing) override;

private:
    Core::HID::MouseState next_state{};
    Core::HID::AnalogStickState last_mouse_wheel_state{};
    /// Virtual cursor for Joy-Con 2 mouse mode. It only exists so x/y stay in the protocol's
    /// screen space; games read delta_x/delta_y for the sensor movement itself.
    f32 mouse_mode_x{};
    f32 mouse_mode_y{};
    bool mouse_mode_positioned{};
    Core::HID::EmulatedDevices* emulated_devices = nullptr;
};
} // namespace Service::HID
