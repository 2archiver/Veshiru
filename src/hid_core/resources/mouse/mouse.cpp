// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "core/core_timing.h"
#include "core/frontend/emu_window.h"
#include "hid_core/frontend/emulated_devices.h"
#include "hid_core/hid_core.h"
#include "hid_core/resources/applet_resource.h"
#include "hid_core/resources/mouse/mouse.h"
#include "hid_core/resources/shared_memory_format.h"

namespace Service::HID {

Mouse::Mouse(Core::HID::HIDCore& hid_core_) : ControllerBase{hid_core_} {
    emulated_devices = hid_core.GetEmulatedDevices();
}

Mouse::~Mouse() = default;

void Mouse::OnInit() {}
void Mouse::OnRelease() {}

void Mouse::OnUpdate(const Core::Timing::CoreTiming& core_timing) {
    std::scoped_lock shared_lock{*shared_mutex};
    const u64 aruid = applet_resource->GetActiveAruid();
    auto* data = applet_resource->GetAruidData(aruid);

    if (data == nullptr || !data->flag.is_assigned) {
        return;
    }

    MouseSharedMemoryFormat& shared_memory = data->shared_memory_format->mouse;

    if (!IsControllerActivated()) {
        shared_memory.mouse_lifo.buffer_count = 0;
        shared_memory.mouse_lifo.buffer_tail = 0;
        return;
    }

    next_state = {};

    const auto& last_entry = shared_memory.mouse_lifo.ReadCurrentEntry().state;
    next_state.sampling_number = last_entry.sampling_number + 1;

    if (Settings::values.joycon2_mouse_enabled.GetValue()) {
        // Joy-Con 2 mouse mode: relative host motion, carried as the report's delta_x/delta_y.
        // The absolute x/y stay inside the screen so games that draw a cursor keep it on screen.
        const auto& mouse_button_state = emulated_devices->GetMouseButtons();
        const auto& mouse_wheel_state = emulated_devices->GetMouseWheel();
        // Sensitivity and inversion were already applied when the render window captured the
        // motion, so only the report range clamp remains here.
        const auto delta = emulated_devices->ConsumeMouseDelta();
        constexpr f32 max_delta = 2047.0f;
        const f32 delta_x = std::clamp(delta.x, -max_delta, max_delta);
        const f32 delta_y = std::clamp(delta.y, -max_delta, max_delta);

        const f32 screen_width = static_cast<f32>(Layout::ScreenUndocked::Width);
        const f32 screen_height = static_cast<f32>(Layout::ScreenUndocked::Height);
        if (!mouse_mode_positioned) {
            mouse_mode_x = screen_width * 0.5f;
            mouse_mode_y = screen_height * 0.5f;
            mouse_mode_positioned = true;
        }
        mouse_mode_x = std::clamp(mouse_mode_x + delta_x, 0.0f, screen_width - 1.0f);
        mouse_mode_y = std::clamp(mouse_mode_y + delta_y, 0.0f, screen_height - 1.0f);

        next_state.attribute.is_connected.Assign(1);
        next_state.x = static_cast<s32>(mouse_mode_x);
        next_state.y = static_cast<s32>(mouse_mode_y);
        next_state.delta_x = static_cast<s32>(delta_x);
        next_state.delta_y = static_cast<s32>(delta_y);
        next_state.delta_wheel_x = mouse_wheel_state.x - last_mouse_wheel_state.x;
        next_state.delta_wheel_y = mouse_wheel_state.y - last_mouse_wheel_state.y;

        last_mouse_wheel_state = mouse_wheel_state;
        next_state.button = mouse_button_state;
    } else {
        mouse_mode_positioned = false;
        if (Settings::values.mouse_enabled) {
            const auto& mouse_button_state = emulated_devices->GetMouseButtons();
            const auto& mouse_position_state = emulated_devices->GetMousePosition();
            const auto& mouse_wheel_state = emulated_devices->GetMouseWheel();
            next_state.attribute.is_connected.Assign(1);
            next_state.x = static_cast<s32>(mouse_position_state.x * Layout::ScreenUndocked::Width);
            next_state.y = static_cast<s32>(mouse_position_state.y * Layout::ScreenUndocked::Height);
            next_state.delta_x = next_state.x - last_entry.x;
            next_state.delta_y = next_state.y - last_entry.y;
            next_state.delta_wheel_x = mouse_wheel_state.x - last_mouse_wheel_state.x;
            next_state.delta_wheel_y = mouse_wheel_state.y - last_mouse_wheel_state.y;

            last_mouse_wheel_state = mouse_wheel_state;
            next_state.button = mouse_button_state;
        }
    }

    shared_memory.mouse_lifo.WriteNextEntry(next_state);
}

} // namespace Service::HID
