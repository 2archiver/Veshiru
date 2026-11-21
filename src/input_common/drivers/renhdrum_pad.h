// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include <SDL3/SDL.h>

#include "common/input.h"

namespace InputCommon {

// Ported from the hdrumble-mvp branch of https://github.com/Octelly/Eglantine (samaBR85, hermes).

/// True when the build includes RenHDrum support (ENABLE_RENHDRUM).
bool IsRenHDrumBuilt();

/// True when renhdrum.dll (librenhdrum.so / .dylib) was found and loaded. Loads it on first call.
bool IsRenHDrumLoaded();

/// Kind of the controller RenHDrum connected most recently and still drives: 0 Switch
/// controller, 1 DualSense, 2 Xbox style, or -1 when none.
int GetRenHDrumConnectedKind();

/// For connected Xbox-style controllers: 1 when one exposes trigger motors, 0 when none does,
/// -1 when no Xbox-style controller is connected.
int GetRenHDrumXboxTriggers();

/// True when an official Xbox controller is connected but its trigger motors are out of reach:
/// on Windows SDL drives them only through raw input ("Enable XInput 8 player support").
bool GetRenHDrumXboxNeedsRawInput();

/**
 * Vibration for one SDL controller through RenHDrum (https://github.com/trzz1214/RenHDrum):
 * Switch Pro Controller / Joy-Con get HD Rumble packets, a wired DualSense gets audio haptics on
 * its voice coils, an Xbox-style pad gets a mix for its eccentric-mass motors. Settings come from
 * Settings::values and are re-read on every vibration and about ten times a second from Tick.
 */
class RenHDrumPad {
public:
    /// nullptr when RenHDrum is not loaded or the controller is not one it drives.
    static std::shared_ptr<RenHDrumPad> Create(SDL_Joystick* joystick, SDL_Gamepad* gamepad);

    ~RenHDrumPad();

    RenHDrumPad(const RenHDrumPad&) = delete;
    RenHDrumPad& operator=(const RenHDrumPad&) = delete;

    /// Releases the engine and the audio stream. Call before the SDL joystick is closed.
    void Shutdown();

    /// True while RenHDrum plays this controller's vibration.
    bool IsActive() const;

    /// True while this is a DualSense that RenHDrum's haptics are switched on for: the built-in
    /// DualSense haptics then leave the controller's audio device alone.
    bool OwnsDualSenseHaptics() const;

    /// HLE thread. True when RenHDrum took the vibration; false means use the standard rumble.
    bool Vibrate(std::int64_t now_us, bool is_right, const Common::Input::VibrationStatus& status);

    /// Input thread, about once per millisecond: due HD Rumble packets, Xbox motor refreshes and
    /// DualSense audio retries.
    void Tick(std::int64_t now_us);

    /// Emulation paused or stopped: silences the controller now, and keeps it silent until the
    /// guest sends the next vibration.
    void Halt(std::int64_t now_us);

private:
    struct Impl;
    explicit RenHDrumPad(std::unique_ptr<Impl> impl_);

    std::unique_ptr<Impl> impl;
};

} // namespace InputCommon
