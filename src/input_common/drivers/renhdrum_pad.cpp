// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ported from the hdrumble-mvp branch of https://github.com/Octelly/Eglantine (a540ba5):
// "input: HD Rumble and DualSense haptics through RenHDrum" by samaBR85 and the follow-up commits
// by hermes. RenHDrum itself is by Three-Dola (https://github.com/trzz1214/RenHDrum).
//
// Veshiru's changes: the Xbox output goes through helpers/xbox_hd_rumble (a mapping and a write
// scheduler that are unit tested), every Xbox write has a bounded duration and is refreshed,
// trigger motors are opt-in, the DualSense audio device is only opened while its setting is on,
// settings are picked up from Tick as well as from the next vibration, Halt() silences the pad
// when emulation pauses or stops, and refused rumble writes are logged rate-limited.

#include "input_common/drivers/renhdrum_pad.h"

#ifdef ENABLE_RENHDRUM

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <renhdrum.h>

#include "common/dynamic_library.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "input_common/helpers/xbox_hd_rumble.h"

#ifdef RENHDRUM_SDL_HDRUMBLE
// Added by .patch/sdl3-hdrumble/0002-hdrumble-idle-write.patch (SDL3 is linked statically). Each
// returns 1 when the mode is active, 0 when SDL refused it, -1 for a pad that is not a HIDAPI
// Switch pad.
extern "C" {
int SDLCALL SDL_HDRumbleSetNoQueue(SDL_Joystick* joystick, int enable, int gap_ms, int pause_ms);
int SDLCALL SDL_HDRumbleSetNoQueueBluetooth(SDL_Joystick* joystick, int enable, int gap_ms);
int SDLCALL SDL_HDRumbleSetBluetoothAfterInput(SDL_Joystick* joystick, int enable);
}
#endif

namespace InputCommon {

namespace {

// The RenHDrum C ABI, resolved from the shared library at runtime: nothing to link, and a build
// without the library just keeps the standard rumble.
struct RenHDrumApi {
    decltype(&renhdrum_engine_create) engine_create{};
    decltype(&renhdrum_engine_destroy) engine_destroy{};
    decltype(&renhdrum_engine_apply_settings) engine_apply_settings{};
    decltype(&renhdrum_engine_push_half_sample) engine_push_half_sample{};
    decltype(&renhdrum_engine_tick) engine_tick{};
    decltype(&renhdrum_settings_resolve) settings_resolve{};
    // Optional (RenHDrum 0.4.0+): the resolver that reads no IPC and no flag files. Preferred,
    // because this host has no Brontide-App config and the plain resolver would look for one.
    decltype(&renhdrum_settings_resolve_pure) settings_resolve_pure{};
    // Optional (RenHDrum 0.4.0+): the Switch 2 Mode 2 translation, chosen by the guest's Mode.
    decltype(&renhdrum_set_nx2_mode) set_nx2_mode{};
    decltype(&renhdrum_dualsense_dsp_create) dsp_create{};
    decltype(&renhdrum_dualsense_dsp_destroy) dsp_destroy{};
    decltype(&renhdrum_dualsense_dsp_set_half_target) dsp_set_half_target{};
    decltype(&renhdrum_dualsense_dsp_fill_f32) dsp_fill_f32{};
    // Optional (RenHDrum 0.2.1+): DualSense haptics over Bluetooth
    decltype(&renhdrum_dualsense_dsp_build_bt_report) dsp_build_bt_report{};
    // Optional (RenHDrum 0.2.0+): Xbox-style controllers
    decltype(&renhdrum_xbox_rumble_create) xbox_create{};
    decltype(&renhdrum_xbox_rumble_destroy) xbox_destroy{};
    decltype(&renhdrum_xbox_rumble_push_half_sample) xbox_push_half_sample{};
    decltype(&renhdrum_xbox_rumble_tick) xbox_tick{};
    decltype(&renhdrum_xbox_rumble_resend) xbox_resend{};
    bool HasXbox() const {
        return xbox_create && xbox_destroy && xbox_push_half_sample && xbox_tick && xbox_resend;
    }

    Common::DynamicLibrary library;
    bool loaded{};
};

const RenHDrumApi& Api() {
    static const RenHDrumApi api = [] {
        RenHDrumApi a;
#if defined(_WIN32)
        constexpr const char* name = "renhdrum.dll";
#elif defined(__APPLE__)
        constexpr const char* name = "librenhdrum.dylib";
#else
        constexpr const char* name = "librenhdrum.so";
#endif
        // Try, in order: the RENHDRUM_LIBRARY override, then the executable's directory. On
        // Windows nothing else: a bare name would let LoadLibrary pick a renhdrum.dll up from the
        // working directory or PATH, which is not a place Veshiru put it.
        std::vector<std::string> candidates;
        if (const char* override_path = std::getenv("RENHDRUM_LIBRARY");
            override_path != nullptr && *override_path != '\0') {
            candidates.emplace_back(override_path);
        }
        // SDL3 owns the string; it must not be freed.
        if (const char* base_path = SDL_GetBasePath()) {
            candidates.emplace_back(std::string{base_path} + name);
        }
#ifndef _WIN32
        // dlopen does not look in the executable's directory; the loader's search path is the
        // usual place for a packaged library there
        candidates.emplace_back(name);
#endif

        std::string opened;
        for (const auto& candidate : candidates) {
            a.library = Common::DynamicLibrary(candidate.c_str());
            if (a.library.IsOpen()) {
                opened = candidate;
                break;
            }
        }
        if (opened.empty()) {
            LOG_INFO(Input, "RenHDrum: {} not found, using the standard rumble", name);
            return a;
        }
        a.loaded = a.library.GetSymbol("renhdrum_engine_create", &a.engine_create) &&
                   a.library.GetSymbol("renhdrum_engine_destroy", &a.engine_destroy) &&
                   a.library.GetSymbol("renhdrum_engine_apply_settings", &a.engine_apply_settings) &&
                   a.library.GetSymbol("renhdrum_engine_push_half_sample",
                                       &a.engine_push_half_sample) &&
                   a.library.GetSymbol("renhdrum_engine_tick", &a.engine_tick) &&
                   a.library.GetSymbol("renhdrum_settings_resolve", &a.settings_resolve) &&
                   a.library.GetSymbol("renhdrum_dualsense_dsp_create", &a.dsp_create) &&
                   a.library.GetSymbol("renhdrum_dualsense_dsp_destroy", &a.dsp_destroy) &&
                   a.library.GetSymbol("renhdrum_dualsense_dsp_set_half_target",
                                       &a.dsp_set_half_target) &&
                   a.library.GetSymbol("renhdrum_dualsense_dsp_fill_f32", &a.dsp_fill_f32);
        if (a.loaded) {
            // Optional: a missing symbol leaves it null
            (void)a.library.GetSymbol("renhdrum_settings_resolve_pure", &a.settings_resolve_pure);
            (void)a.library.GetSymbol("renhdrum_set_nx2_mode", &a.set_nx2_mode);
            (void)a.library.GetSymbol("renhdrum_dualsense_dsp_build_bt_report",
                                      &a.dsp_build_bt_report);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_create", &a.xbox_create);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_destroy", &a.xbox_destroy);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_push_half_sample",
                                      &a.xbox_push_half_sample);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_tick", &a.xbox_tick);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_resend", &a.xbox_resend);
            LOG_INFO(Input, "RenHDrum: loaded {}", opened);
        } else {
            LOG_WARNING(Input, "RenHDrum: {} is missing functions, using the standard rumble",
                        opened);
        }
        return a;
    }();
    return api;
}

constexpr Uint16 NintendoVendorId = 0x057E;
constexpr Uint16 SonyVendorId = 0x054C;
constexpr Uint16 MicrosoftVendorId = 0x045E;
constexpr std::array<Uint8, 10> NeutralPacket{0x10, 0, 0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40};
constexpr int DualSenseAudioMaxAttempts = 10;
// SDL no-queue gap while the 5 ms envelope runs (Ryuhdr AmFmNoQueueGapMs)
[[maybe_unused]] constexpr int NoQueueEnvelopeGapMs = 10;

bool IsSwitchHdController(Uint16 vid, Uint16 pid) {
    return vid == NintendoVendorId && (pid == 0x2009 || pid == 0x2006 || pid == 0x2007);
}

/// Xbox One / Series and Xbox 360 style pads, Microsoft's or not (GameSir, 8BitDo, PowerA...).
/// RenHDrum's converter was tuned on an Xbox One S; others get the same four-motor mix, and
/// a pad whose path has no trigger motors (XInput) keeps only the grips.
bool IsXboxStyle(SDL_Joystick* joystick, SDL_Gamepad* gamepad) {
    if (gamepad) {
        const SDL_GamepadType type = SDL_GetGamepadType(gamepad);
        if (type == SDL_GAMEPAD_TYPE_XBOXONE || type == SDL_GAMEPAD_TYPE_XBOX360) {
            return true;
        }
    }
#ifdef _WIN32
    // With raw input on, SDL types a third-party XInput pad by its vendor ID and may not call it
    // an Xbox controller. Windows marks every XInput-capable HID device with "IG_" in its path.
    const char* path = SDL_GetJoystickPath(joystick);
    if (path && (std::strstr(path, "IG_") != nullptr || std::strstr(path, "ig_") != nullptr)) {
        return true;
    }
#else
    (void)joystick;
#endif
    return false;
}

/// Which input path SDL opened an Xbox-style pad through, for the log. SDL has no call that
/// names its driver; the joystick path does: the XInput driver reports "XInput#<slot>", the raw
/// input and HIDAPI drivers the HID device path.
const char* XboxBackendName(SDL_Joystick* joystick) {
    const char* path = SDL_GetJoystickPath(joystick);
    if (!path || path[0] == '\0') {
        return "an SDL driver that reports no device path";
    }
    if (std::strncmp(path, "XInput#", 7) == 0) {
        return "SDL XInput";
    }
    if (std::strstr(path, "HID#") != nullptr || std::strstr(path, "hid#") != nullptr) {
        return "SDL raw input / HIDAPI";
    }
    return "another SDL driver";
}

bool IsDualSense(Uint16 vid, Uint16 pid) {
    return vid == SonyVendorId && (pid == 0x0CE6 || pid == 0x0DF2);
}

struct UserSettings {
    bool hd_enabled{false};
    int usb_interval_ms{15};
    bool envelope_5ms{true};
    bool frequency_in_packet{true};
    bool crisp_notes{true};
    int bluetooth_level{3};
    bool bt_after_input{true};
    bool bt_fast{true};
    bool bt_crisp_notes{true};
    bool dualsense_enabled{false};
    int dualsense_volume{100};
    bool dualsense_crisp_notes{true};
    bool xbox_enabled{true};
    int xbox_strength{100};
    bool xbox_direct{false};
    int xbox_detail_mix{50};
    int xbox_hit_accents{50};
    int xbox_trigger_mix{20};
    bool xbox_triggers_enabled{false};
    int gain{100};
    int low_boost{0};
    bool operator==(const UserSettings&) const = default;
};

UserSettings ReadUserSettings() {
    const auto& s = Settings::values;
    return {
        .hd_enabled = s.hd_rumble_enabled.GetValue(),
        .usb_interval_ms = std::clamp(s.hd_rumble_usb_interval_ms.GetValue(), 5, 100),
        .envelope_5ms = s.hd_rumble_envelope_5ms.GetValue(),
        .frequency_in_packet = s.hd_rumble_frequency_in_packet.GetValue(),
        .crisp_notes = s.hd_rumble_crisp_notes.GetValue(),
        .bluetooth_level = std::clamp(s.hd_rumble_bluetooth_level.GetValue(), 1, 3),
        .bt_after_input = s.hd_rumble_bt_after_input.GetValue(),
        .bt_fast = s.hd_rumble_bt_fast.GetValue(),
        .bt_crisp_notes = s.hd_rumble_bt_crisp_notes.GetValue(),
        .dualsense_enabled = s.renhdrum_dualsense_enabled.GetValue(),
        .dualsense_volume = std::clamp(s.renhdrum_dualsense_volume.GetValue(), 0, 100),
        .dualsense_crisp_notes = s.renhdrum_dualsense_crisp_notes.GetValue(),
        .xbox_enabled = s.xbox_rumble_enabled.GetValue(),
        .xbox_strength = std::clamp(s.xbox_motor_strength.GetValue(), 0, 1000),
        .xbox_direct = s.xbox_rumble_mode.GetValue() == 1,
        .xbox_detail_mix = std::clamp(s.xbox_detail_mix.GetValue(), 0, 100),
        .xbox_hit_accents = std::clamp(s.xbox_hit_accents.GetValue(), 0, 100),
        .xbox_trigger_mix = std::clamp(s.xbox_trigger_mix.GetValue(), 0, 100),
        .xbox_triggers_enabled = s.xbox_trigger_rumble.GetValue(),
        .gain = std::clamp(s.hd_rumble_gain.GetValue(), 25, 1000),
        .low_boost = std::clamp(s.hd_rumble_low_boost.GetValue(), 0, 100),
    };
}

// Gain and weak-rumble lift, shared with the Xbox mapping (helpers/xbox_hd_rumble)
float ShapeAmplitude(float amplitude, const UserSettings& u) {
    return ShapeHdRumbleAmplitude(amplitude, static_cast<float>(u.gain) / 100.0f,
                                  static_cast<float>(u.low_boost) / 100.0f);
}

// How often Tick looks at the settings, so a change in the dialog reaches a pad that is not
// receiving vibrations (a held rumble is sent by the guest only once)
constexpr std::int64_t SettingsPollUs = 100'000;
// At most one warning per pad in this long for rumble calls the pad refuses
constexpr std::int64_t RumbleWarningIntervalUs = 10'000'000;
// A refused write is retried this many times in a row; after that only the refresh retries
constexpr int MaxRumbleRetries = 3;
// RenHDrum's own Xbox mix produces short trigger pulses; a lost stop must not leave them running
constexpr Uint32 MixTriggerDurationMs = 200;

// Controllers RenHDrum drives, per kind, for the settings dialog
std::array<std::atomic<int>, 3> connected_count{};

// Pads per kind (0 Switch, 1 DualSense, 2 Xbox). Callers only index a kind they know is valid.
constexpr std::size_t ConnectedCountIndex(int kind) {
    return static_cast<std::size_t>(kind);
}
std::atomic<int> last_connected_kind{-1};
// Connected Xbox-style controllers that expose trigger motors
std::atomic<int> xbox_trigger_pads{};
// Connected official (Microsoft) Xbox controllers whose trigger motors SDL does not expose
std::atomic<int> xbox_official_no_triggers{};

void SendNeutralPacket(SDL_Joystick* joystick) {
    SDL_SendJoystickEffect(joystick, NeutralPacket.data(), static_cast<int>(NeutralPacket.size()));
}

} // namespace

// Lock rule: `mutex` is never held while calling an SDL function that takes the SDL joystick lock
// (SDL_SendJoystickEffect, SDL_HDRumbleSet*, SDL_GetJoystickConnectionState, SDL_Rumble*). SDL
// runs the joystick-removed event watcher under that lock, and Shutdown() takes `mutex` from
// there. The joystick pointer is copied out and used after unlocking: SDL3 validates joystick
// handles, so a call on one closed meanwhile only fails.
//
// A DualSense on Bluetooth has no audio endpoint. A thread of its own sends the haptics as HID
// output report 0x32 on a second handle to the pad. That thread never takes `mutex`; it holds
// `bt_dsp_mutex` while it reads the DSP, and the DSP is replaced only under that lock too.
struct RenHDrumPad::Impl {
    bool is_bluetooth{};
    int kind{-1};
    // Rate limit for the refused-packet warning; only Tick writes it, but from two threads
    std::atomic<std::int64_t> next_effect_error_us{0};

    mutable std::mutex mutex;
    SDL_Joystick* joystick{};
    SDL_Gamepad* gamepad{};
    RenHdcEngine* engine{};
    RenHdcDualSenseDsp* dsp{};
    SDL_AudioStream* audio_stream{};
    bool audio_inited{};
    bool audio_opening{};
    int audio_attempts{};
    std::int64_t audio_next_us{};
    UserSettings user{};
    // Switch 2 Mode last handed to the engine; -1 forces the first update
    int nx2_mode{-1};

    // Xbox-style pad: RenHDrum's own four-motor mix (the Direct mode maps on the host instead)
    RenHdcXboxRumble* xbox{};
    bool xbox_has_rumble{};
    bool xbox_triggers{};
    bool official_no_triggers{};
    // The guest's latest sample for each half of the emulated controller (Direct mode)
    std::array<HdRumbleSide, 2> sides{};
    // Decides when the pad is written to, in both Xbox modes
    XboxRumbleScheduler scheduler;
    // The trigger motors may still be running from an earlier write
    bool triggers_running{};
    int rumble_failures{};
    std::int64_t next_rumble_warning_us{};
    std::int64_t next_settings_poll_us{};
    // Set when the emulator stops or pauses: nothing is played until the guest vibrates again,
    // so an engine's release tail cannot restart the motors of a stopped game
    bool halted{};

    /// The SDL calls one scheduler result turns into, made after the mutex is released.
    struct XboxWrite {
        XboxMotors motors{};
        bool triggers{};
        Uint32 trigger_duration_ms{};
    };

    /// HD rumble is playing this Xbox-style pad's vibration. Caller holds the mutex.
    bool XboxDriving() const {
        return xbox && xbox_has_rumble && user.hd_enabled && user.xbox_enabled;
    }

    /// The host-side mapping is in use instead of RenHDrum's mix.
    bool XboxDirect() const {
        return user.xbox_direct;
    }

    /// Trigger motors are driven only when the pad reports them and the user asked for them.
    bool XboxTriggersOn() const {
        return xbox_triggers && user.xbox_triggers_enabled;
    }

    XboxRumbleParams DirectParams() const {
        return {
            .gain = static_cast<float>(user.gain) / 100.0f,
            .low_boost = static_cast<float>(user.low_boost) / 100.0f,
            .strength = static_cast<float>(user.xbox_strength) / 100.0f,
            .detail_mix = static_cast<float>(user.xbox_detail_mix) / 100.0f,
            .trigger_mix = static_cast<float>(user.xbox_trigger_mix) / 100.0f,
        };
    }

    XboxRumbleScheduler::Config SchedulerConfig() const {
        XboxRumbleScheduler::Config config{};
        config.hit_accent = static_cast<float>(user.xbox_hit_accents) / 100.0f;
        return config;
    }

    /// RenHDrum's mix scaled by the motor strength. The mix tops out near 10 % of the range
    /// (tuned on the Xbox One S motors); weaker motors need more. Caller holds the mutex.
    XboxMotors ScaleMix(const std::array<std::uint16_t, 4>& motors) const {
        const auto strength = static_cast<std::uint32_t>(user.xbox_strength);
        const auto scaled = [strength](std::uint16_t motor) {
            const std::uint32_t value = static_cast<std::uint32_t>(motor) * strength / 100u;
            return static_cast<std::uint16_t>(std::min<std::uint32_t>(65535u, value));
        };
        const bool triggers = XboxTriggersOn();
        return {
            .left_motor = scaled(motors[0]),
            .right_motor = scaled(motors[1]),
            .left_trigger = triggers ? scaled(motors[2]) : std::uint16_t{0},
            .right_trigger = triggers ? scaled(motors[3]) : std::uint16_t{0},
        };
    }

    /// Caller holds the mutex.
    XboxWrite PlanXboxWrite(const XboxRumbleSend& send) {
        XboxWrite write{.motors = send.motors};
        if (!xbox_triggers) {
            return write;
        }
        const bool direct = XboxDirect();
        const bool has_values = send.motors.left_trigger != 0 || send.motors.right_trigger != 0;
        // Direct mode holds a level, so the triggers are refreshed along with the grips.
        // RenHDrum's mix plays short pulses, which a refresh must not repeat. A zero is only
        // written to triggers that may still be running.
        write.triggers = has_values ? (direct || !send.refresh) : triggers_running;
        write.trigger_duration_ms = direct ? XboxRumbleScheduler::DurationMs : MixTriggerDurationMs;
        if (write.triggers) {
            triggers_running = has_values;
        }
        return write;
    }

    /// Called without the mutex. Returns whether the pad took the grip motors' values.
    static bool WriteXbox(SDL_Joystick* js, const XboxWrite& write) {
        const bool ok = SDL_RumbleJoystick(js, write.motors.left_motor, write.motors.right_motor,
                                           XboxRumbleScheduler::DurationMs);
        if (write.triggers) {
            SDL_RumbleJoystickTriggers(js, write.motors.left_trigger, write.motors.right_trigger,
                                       write.trigger_duration_ms);
        }
        return ok;
    }

    /// Called without the mutex, right after WriteXbox. A refused write is retried a few times
    /// (some paths refuse rumble until the first button press); repeated refusals are logged at
    /// most once per RumbleWarningIntervalUs.
    void NoteXboxWrite(bool ok, std::int64_t now_us) {
        // SDL keeps the error per thread, so read it before anything else can replace it
        const std::string error = ok ? std::string{} : std::string{SDL_GetError()};
        int failures = 0;
        {
            std::scoped_lock lock{mutex};
            if (ok) {
                rumble_failures = 0;
                return;
            }
            ++rumble_failures;
            if (rumble_failures <= MaxRumbleRetries) {
                scheduler.MarkFailed();
                if (xbox && !XboxDirect() && Api().xbox_resend) {
                    Api().xbox_resend(xbox);
                }
                return;
            }
            if (now_us < next_rumble_warning_us) {
                return;
            }
            next_rumble_warning_us = now_us + RumbleWarningIntervalUs;
            failures = rumble_failures;
        }
        LOG_WARNING(Input, "HD rumble: the controller refused {} rumble writes in a row: {}",
                    failures, error);
    }

    std::mutex bt_dsp_mutex;
    SDL_hid_device* bt_hid{};
    std::thread bt_thread;
    std::atomic<bool> bt_running{};

    /// Engine settings from the user's choices (Ryuhdr 0.2.6's recommended mode by default).
    /// Caller holds the mutex.
    bool ResolveEngineSettings(RenHdcSettings& settings) const {
        RenHdcConfig config{};
        config.enable_hd_rumble = user.hd_enabled;
        config.hd_rumble_usb_interval_ms = user.usb_interval_ms;
        config.hd_rumble_bluetooth_level = user.bluetooth_level;
        config.hd_rumble_usb_auto_slow = true;
        config.hd_rumble_experimental_usb_no_queue = true;
        config.hd_rumble_experimental_envelope_5ms = user.envelope_5ms;
        config.hd_rumble_experimental_frequency_in_packet = user.frequency_in_packet;
        config.hd_rumble_experimental_frequency_both_bands = user.frequency_in_packet;
        config.hd_rumble_experimental_crisp_notes = user.crisp_notes;
        config.hd_rumble_experimental_bt_after_input = user.bt_after_input;
        config.hd_rumble_experimental_bt_fast = user.bt_fast;
        config.hd_rumble_experimental_bt_crisp_notes = user.bt_crisp_notes;
        const auto& api = Api();
        const auto resolve = api.settings_resolve_pure ? api.settings_resolve_pure
                                                       : api.settings_resolve;
        return resolve(&config, &settings) == RENHDRUM_OK;
    }

    /// Switches SDL's HD Rumble transport modes to match `settings`, then clears the modes SDL
    /// refused, so the engine never paces for writes SDL does not gate. Called without the mutex.
    void ApplyTransportModes(SDL_Joystick* js, RenHdcSettings& settings) const {
#ifdef RENHDRUM_SDL_HDRUMBLE
        SDL_HDRumbleSetNoQueue(js, 0, 0, 0);
        SDL_HDRumbleSetNoQueueBluetooth(js, 0, 0);
        SDL_HDRumbleSetBluetoothAfterInput(js, 0);
        if (is_bluetooth) {
            // B3 needs B1 on; SDL never runs both on one pad, B3 wins
            settings.bt_fast = settings.enabled && settings.bt_fast && settings.bt_after_input &&
                               SDL_HDRumbleSetNoQueueBluetooth(js, 1, NoQueueEnvelopeGapMs) == 1;
            settings.bt_after_input = settings.enabled && settings.bt_after_input &&
                                      !settings.bt_fast &&
                                      SDL_HDRumbleSetBluetoothAfterInput(js, 1) == 1;
            settings.usb_no_queue = false;
        } else {
            const int gap_ms =
                settings.envelope_5ms ? NoQueueEnvelopeGapMs : settings.usb_interval_ms;
            settings.usb_no_queue =
                settings.enabled && settings.usb_no_queue &&
                SDL_HDRumbleSetNoQueue(js, 1, gap_ms, settings.usb_pause_ms) == 1;
            settings.bt_after_input = false;
            settings.bt_fast = false;
        }
#else
        // Stock SDL queues every write; the engine paces by its clock.
        (void)js;
        settings.usb_no_queue = false;
        settings.bt_after_input = false;
        settings.bt_fast = false;
#endif
    }

    /// Resolves the settings, switches the SDL modes and hands the result to the engine.
    /// Called without the mutex.
    void ConfigureEngine() {
        RenHdcSettings settings{};
        SDL_Joystick* js = nullptr;
        {
            std::scoped_lock lock{mutex};
            if (!engine || !joystick || !ResolveEngineSettings(settings)) {
                return;
            }
            js = joystick;
        }
        ApplyTransportModes(js, settings);
        {
            std::scoped_lock lock{mutex};
            if (!engine) {
                return;
            }
            Api().engine_apply_settings(engine, &settings);
        }
        if (is_bluetooth) {
            LOG_INFO(Input, "RenHDrum HD Rumble Bluetooth: level {}, {}{}", settings.bluetooth_level,
                     settings.bt_fast          ? "5 ms envelope while SDL is idle"
                     : settings.bt_after_input ? "written right after an input report"
                                               : "clock based writes",
                     settings.enabled && settings.bt_crisp_notes ? ", crisp notes" : "");
        } else {
            const bool envelope = settings.usb_no_queue && settings.envelope_5ms;
            LOG_INFO(Input, "RenHDrum HD Rumble wired: {} ms, {}{}{}", settings.usb_interval_ms,
                     settings.usb_no_queue ? "no-queue writes" : "queued writes",
                     envelope ? ", 5 ms envelope" : "",
                     envelope && settings.crisp_notes ? ", crisp notes" : "");
        }
    }

    struct SettingsChange {
        bool reconfigure_engine{};
        bool stop_engine{};
        std::optional<XboxWrite> xbox_stop;
    };

    /// The DualSense haptics are wanted: both the master switch and their own setting are on.
    /// Caller holds the mutex.
    bool DualSenseWanted() const {
        return user.hd_enabled && user.dualsense_enabled;
    }

    /// Picks up changes made in the HD Rumble dialog. Caller holds the mutex; the returned SDL
    /// work is done after unlocking.
    SettingsChange RefreshUserSettings(std::int64_t now_us) {
        const UserSettings now = ReadUserSettings();
        if (now == user) {
            return {};
        }
        const UserSettings old = user;
        const bool xbox_was_driving = XboxDriving();
        const bool xbox_was_direct = XboxDirect();
        const bool dualsense_was_wanted = DualSenseWanted();
        user = now;
        std::optional<XboxWrite> xbox_stop;
        if (xbox_was_driving &&
            (!XboxDriving() || xbox_was_direct != XboxDirect() ||
             old.xbox_triggers_enabled != now.xbox_triggers_enabled)) {
            // HD rumble was switched off, or to the other mode: whatever plays next (the other
            // mode, the standard rumble, or nothing) starts from silent motors
            xbox_stop = PlanXboxWrite(StopXbox(now_us));
        }
        if (dsp && dualsense_was_wanted && !DualSenseWanted()) {
            // Give the audio device back, so the built-in DualSense haptics can open it
            CloseDualSenseOutput();
        }
        if (dsp && (old.dualsense_volume != now.dualsense_volume ||
                    old.dualsense_crisp_notes != now.dualsense_crisp_notes)) {
            // The audio callback runs with the stream locked, so it never sees a freed DSP. The
            // audio stream lock is not the joystick lock, and the callback never takes `mutex`.
            if (audio_stream) {
                SDL_LockAudioStream(audio_stream);
            }
            {
                std::scoped_lock bt_lock{bt_dsp_mutex};
                const auto& api = Api();
                api.dsp_destroy(dsp);
                dsp = api.dsp_create(now.dualsense_volume, now.dualsense_crisp_notes);
            }
            if (audio_stream) {
                SDL_UnlockAudioStream(audio_stream);
            }
        }
        return {
            .reconfigure_engine = engine != nullptr,
            .stop_engine = engine != nullptr && old.hd_enabled && !now.hd_enabled,
            .xbox_stop = xbox_stop,
        };
    }

    /// Forgets what the Xbox motors were playing and returns the zeros to write. Caller holds
    /// the mutex.
    XboxRumbleSend StopXbox(std::int64_t now_us) {
        sides = {};
        if (xbox) {
            // RenHDrum's mix restarts from silence too
            const auto& api = Api();
            api.xbox_push_half_sample(xbox, now_us, false, 0.0f, 160.0f, 0.0f, 320.0f);
            api.xbox_push_half_sample(xbox, now_us, true, 0.0f, 160.0f, 0.0f, 320.0f);
        }
        return scheduler.Stop(now_us);
    }

    /// Does the SDL work RefreshUserSettings asked for. Called without the mutex.
    void ApplySettingsChange(SDL_Joystick* js, const SettingsChange& change,
                             std::int64_t now_us) {
        if (change.xbox_stop) {
            NoteXboxWrite(WriteXbox(js, *change.xbox_stop), now_us);
        }
        if (change.stop_engine) {
            SendNeutralPacket(js);
        }
        if (change.reconfigure_engine) {
            ConfigureEngine();
        }
    }

    /// Stops the DualSense sender thread and closes its audio stream, leaving the DSP in place
    /// so the haptics can be opened again. Caller holds the mutex.
    void CloseDualSenseOutput() {
        // The sender never takes `mutex`, so joining here cannot deadlock
        bt_running.store(false);
        if (bt_thread.joinable()) {
            bt_thread.join();
        }
        if (bt_hid) {
            SDL_hid_close(bt_hid);
            bt_hid = nullptr;
        }
        if (audio_stream) {
            // Waits for a running audio callback; the callback never takes `mutex`
            SDL_DestroyAudioStream(audio_stream);
            audio_stream = nullptr;
        }
        if (audio_inited) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            audio_inited = false;
        }
        audio_attempts = 0;
        audio_next_us = 0;
    }

    /// The DualSense exposes its voice coils as a 4-channel USB audio device that Windows may
    /// list a moment after the HID device, hence the retries. Called without the mutex.
    void OpenDualSenseAudio() {
        SDL_Joystick* js = nullptr;
        SDL_Gamepad* gp = nullptr;
        {
            std::scoped_lock lock{mutex};
            if (!dsp || !joystick || audio_stream || audio_opening || !DualSenseWanted()) {
                return;
            }
            if (is_bluetooth) {
                // No audio endpoint on Bluetooth: one try at the HID haptics, no retries
                audio_attempts = DualSenseAudioMaxAttempts;
            } else {
                audio_opening = true;
                ++audio_attempts;
            }
            js = joystick;
            gp = gamepad;
        }
        if (is_bluetooth) {
            StartDualSenseBluetooth(js, gp);
            return;
        }
        // SDL counts subsystem references: every pad with a stream holds one, so one pad going
        // away does not stop another's audio.
        SDL_AudioStream* stream = nullptr;
        const bool inited = SDL_InitSubSystem(SDL_INIT_AUDIO);
        if (inited) {
            stream = OpenDualSenseAudioStream(audio_attempts == 1);
        } else {
            LOG_WARNING(Input, "RenHDrum: SDL audio init failed: {}", SDL_GetError());
        }
        bool opened = false;
        {
            std::scoped_lock lock{mutex};
            audio_opening = false;
            if (stream && joystick && dsp && !audio_stream && DualSenseWanted()) {
                audio_stream = stream;
                audio_inited = true;
                opened = true;
                // Start only once published: from here on, the DSP is replaced or freed only
                // under SDL_LockAudioStream or after SDL_DestroyAudioStream. The callback never
                // takes `mutex`.
                if (!SDL_ResumeAudioStreamDevice(stream)) {
                    // Not playing: unpublish so vibration falls back to standard rumble and
                    // the remaining retries can still open the device. The callback never ran.
                    LOG_WARNING(Input, "RenHDrum: DualSense audio did not start: {}",
                                SDL_GetError());
                    audio_stream = nullptr;
                    audio_inited = false;
                    opened = false;
                }
            }
        }
        if (!opened) {
            // No device yet, or shut down while opening
            if (stream) {
                SDL_DestroyAudioStream(stream);
            }
            if (inited) {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
        }
        if (opened) {
            // Stop SDL's HID rumble once so the pad plays the audio haptics
            if (gp) {
                SDL_RumbleGamepad(gp, 0, 0, 0);
            } else {
                SDL_RumbleJoystick(js, 0, 0, 0);
            }
        }
    }

    /// Opens a second HID handle on a Bluetooth DualSense and starts the report 0x32 sender.
    /// Called without the mutex. Without the 0.2.1 function or a HID path, the standard rumble
    /// stays.
    void StartDualSenseBluetooth(SDL_Joystick* js, SDL_Gamepad* gp) {
        if (!Api().dsp_build_bt_report) {
            LOG_INFO(Input, "RenHDrum: DualSense on Bluetooth needs RenHDrum 0.2.1 or newer, "
                            "using the standard rumble");
            return;
        }
        const char* path = SDL_GetJoystickPath(js);
        if (!path || !*path) {
            LOG_INFO(Input, "RenHDrum: DualSense on Bluetooth has no HID path, using the "
                            "standard rumble");
            return;
        }
        const std::string hid_path{path};
        SDL_hid_device* hid = SDL_hid_open_path(hid_path.c_str());
        if (!hid) {
            LOG_WARNING(Input,
                        "RenHDrum: could not open the DualSense HID path: {}, using the "
                        "standard rumble",
                        SDL_GetError());
            return;
        }
        {
            std::scoped_lock lock{mutex};
            if (!joystick || !dsp || bt_hid || !DualSenseWanted()) {
                SDL_hid_close(hid);
                return;
            }
            bt_hid = hid;
            bt_running.store(true);
            bt_thread = std::thread([this] { DualSenseBluetoothLoop(); });
        }
        // Stop SDL's HID rumble once so the pad plays the haptics
        if (gp) {
            SDL_RumbleGamepad(gp, 0, 0, 0);
        } else {
            SDL_RumbleJoystick(js, 0, 0, 0);
        }
        LOG_INFO(Input, "RenHDrum: DualSense haptics over Bluetooth (report 0x32)");
    }

    /// Runs on `bt_thread`: one report every RENHDRUM_DS5_BT_PERIOD_NS while vibrating and for
    /// 2 s after, then quiet on the radio. Stops after about 0.5 s of failed writes, which hands
    /// the vibration back to the standard rumble.
    void DualSenseBluetoothLoop() {
        Common::SetCurrentThreadName("RenHDrumDS5BT");
        constexpr Uint64 KeepSendingNs = 2'000'000'000ULL;
        constexpr Uint64 ResyncNs = 100'000'000ULL;
        constexpr int MaxFailsInRow = 50;
        const auto& api = Api();
        std::array<std::uint8_t, RENHDRUM_DS5_BT_REPORT_SIZE> report{};
        Uint64 next = SDL_GetTicksNS();
        Uint64 last_signal = 0;
        int fails_in_row = 0;
        while (bt_running.load()) {
            next += RENHDRUM_DS5_BT_PERIOD_NS;
            int built = -1;
            {
                std::scoped_lock lock{bt_dsp_mutex};
                if (dsp) {
                    built = api.dsp_build_bt_report(dsp, RENHDRUM_DS5_BT_BUFFER_LENGTH_DEFAULT,
                                                    report.data());
                }
            }
            const Uint64 now = SDL_GetTicksNS();
            if (built == 1) {
                last_signal = now;
            }
            if (built >= 0 && last_signal != 0 && now - last_signal < KeepSendingNs) {
                if (SDL_hid_write(bt_hid, report.data(), report.size()) < 0) {
                    ++fails_in_row;
                } else {
                    fails_in_row = 0;
                }
            }
            if (fails_in_row >= MaxFailsInRow) {
                LOG_WARNING(Input,
                            "RenHDrum: DualSense Bluetooth writes failing ({}), using the "
                            "standard rumble",
                            SDL_GetError());
                break;
            }
            if (next > now) {
                SDL_DelayPrecise(next - now);
            } else if (now - next > ResyncNs) {
                next = now; // stalled: resync instead of bursting
            }
        }
        bt_running.store(false);
    }

    /// Finds the DualSense audio device and opens a stream on it, paused until it is published
    /// in `audio_stream`, or nullptr. Called without the mutex, with the SDL audio subsystem
    /// initialized.
    SDL_AudioStream* OpenDualSenseAudioStream(bool verbose) {
        int count = 0;
        SDL_AudioDeviceID* devices = SDL_GetAudioPlaybackDevices(&count);
        SDL_AudioDeviceID matched = 0;
        for (int i = 0; devices && i < count; ++i) {
            const char* name = SDL_GetAudioDeviceName(devices[i]);
            SDL_AudioSpec spec{};
            const int channels =
                SDL_GetAudioDeviceFormat(devices[i], &spec, nullptr) ? spec.channels : 0;
            // Names differ per platform: WASAPI and PipeWire say "DualSense" or "Wireless
            // Controller", CoreAudio spells out the whole product name.
            const bool looks_like_dualsense =
                name && (std::strstr(name, "DualSense") != nullptr ||
                         std::strstr(name, "DualSense Edge") != nullptr ||
                         std::strstr(name, "Wireless Controller") != nullptr);
            if (!looks_like_dualsense) {
                continue;
            }
            if (channels >= 4) {
                matched = devices[i];
                break;
            }
            // The voice coils live on channels 3-4. A two-channel endpoint means the OS exposed
            // the pad as plain stereo (WASAPI shared mode does, and so does Bluetooth everywhere).
            LOG_INFO(Input,
                     "RenHDrum: \"{}\" exposes {} channel(s); haptics need 4, wired USB only",
                     name, channels);
        }
        if (matched == 0 && verbose) {
            // Once per attempt sequence, so a report from a platform we cannot test here says what
            // the audio server actually offered.
            LOG_INFO(Input, "RenHDrum: no 4-channel DualSense device yet, {} playback device(s):",
                     count);
            for (int i = 0; devices && i < count; ++i) {
                SDL_AudioSpec spec{};
                const int channels =
                    SDL_GetAudioDeviceFormat(devices[i], &spec, nullptr) ? spec.channels : 0;
                LOG_INFO(Input, "RenHDrum:   [{}] \"{}\" {} channel(s)", devices[i],
                         SDL_GetAudioDeviceName(devices[i]), channels);
            }
        }
        SDL_free(devices);
        if (matched == 0) {
            LOG_DEBUG(Input, "RenHDrum: no DualSense audio device yet");
            return nullptr;
        }

        const SDL_AudioSpec spec{
            .format = SDL_AUDIO_F32,
            .channels = 4,
            .freq = 48000,
        };
        // Four channels, because the DualSense voice coils listen on 3-4 and the pad exposes them
        // only through its raw UAC endpoint. Anything that swallows the stream between us and that
        // endpoint can flatten it to stereo and the coils fall silent: measured with a PipeWire
        // filter-chain sink (JamesDSP) as the default sink, a four-channel feed arrived as two
        // channels and the game audio took the pad's ports instead. Nothing on this side can see
        // or fix that -- it is a routing decision above us -- so the device name and the channel
        // counts below are logged to make it diagnosable in one run.
        // Give the stream a stable name so it is identifiable in an audio graph and can be pinned
        // by a server-side rule; every stream would otherwise carry the application's name.
        SDL_SetHint(SDL_HINT_AUDIO_DEVICE_STREAM_NAME, "RenHDrum haptics");
        SDL_AudioStream* stream = SDL_OpenAudioDeviceStream(matched, &spec, &AudioCallback, this);
        SDL_ResetHint(SDL_HINT_AUDIO_DEVICE_STREAM_NAME);
        if (!stream) {
            LOG_WARNING(Input, "RenHDrum: could not open the DualSense audio device: {}",
                        SDL_GetError());
            return nullptr;
        }
        // No channel-map call here: SDL 3.4.8's map only reorders or mutes channels (`{1,0}` swaps
        // stereo) and has no way to name speaker positions -- that enum is a later SDL. So what the
        // four channels are has to come from the device itself; this log line is how we tell
        // whether the server gave us a four-channel endpoint or a downmix.
        SDL_AudioSpec src_spec{};
        SDL_AudioSpec dst_spec{};
        if (SDL_GetAudioStreamFormat(stream, &src_spec, &dst_spec)) {
            LOG_INFO(Input,
                     "RenHDrum: DualSense haptics stream: feed {} channel(s) at {} Hz, device takes "
                     "{} at {} Hz (\"{}\")",
                     src_spec.channels, src_spec.freq, dst_spec.channels, dst_spec.freq,
                     SDL_GetAudioDeviceName(matched));
        }
        LOG_INFO(Input, "RenHDrum: DualSense haptics on \"{}\"", SDL_GetAudioDeviceName(matched));
        return stream;
    }

    static void SDLCALL AudioCallback(void* userdata, SDL_AudioStream* stream,
                                      int additional_amount, int /*total_amount*/) {
        auto* const self = static_cast<Impl*>(userdata);
        if (additional_amount <= 0 || !self || !self->dsp) {
            return;
        }
        constexpr int frame_bytes = 4 * static_cast<int>(sizeof(float));
        constexpr int chunk_frames = 256;
        std::array<float, chunk_frames * 4> chunk{};
        int frames = additional_amount / frame_bytes;
        while (frames > 0) {
            const int count = (std::min)(frames, chunk_frames);
            if (Api().dsp_fill_f32(self->dsp, chunk.data(), static_cast<size_t>(count)) !=
                RENHDRUM_OK) {
                chunk.fill(0.0f); // never send garbage to the actuators
            }
            SDL_PutAudioStreamData(stream, chunk.data(), count * frame_bytes);
            frames -= count;
        }
    }

    /// Frees the engine, the DSP and the audio stream. Caller holds the mutex. Returns whether
    /// an engine was running, so the caller sends the neutral packet after unlocking.
    bool Release() {
        const auto& api = Api();
        CloseDualSenseOutput();
        if (xbox) {
            api.xbox_destroy(xbox);
            xbox = nullptr;
        }
        const bool had_engine = engine != nullptr;
        if (engine) {
            api.engine_destroy(engine);
            engine = nullptr;
        }
        if (dsp) {
            api.dsp_destroy(dsp);
            dsp = nullptr;
        }
        return had_engine;
    }
};

bool IsRenHDrumBuilt() {
    return true;
}

bool IsRenHDrumLoaded() {
    return Api().loaded;
}

std::shared_ptr<RenHDrumPad> RenHDrumPad::Create(SDL_Joystick* joystick, SDL_Gamepad* gamepad) {
    if (!joystick || !IsRenHDrumLoaded()) {
        return nullptr;
    }
    const Uint16 vid = SDL_GetJoystickVendor(joystick);
    const Uint16 pid = SDL_GetJoystickProduct(joystick);
    const bool switch_pad = IsSwitchHdController(vid, pid);
    const auto& api_check = Api();
    const bool xbox_pad = !switch_pad && !IsDualSense(vid, pid) && api_check.HasXbox() &&
                          IsXboxStyle(joystick, gamepad);
    if (!switch_pad && !xbox_pad && !IsDualSense(vid, pid)) {
        LOG_INFO(Input, "RenHDrum: {:04X}:{:04X} \"{}\" is not a controller RenHDrum drives",
                 vid, pid, SDL_GetJoystickName(joystick) ? SDL_GetJoystickName(joystick) : "");
        return nullptr;
    }

    // Not shared yet: no other thread can reach `state` until it is returned.
    const auto& api = Api();
    auto state = std::make_unique<Impl>();
    state->joystick = joystick;
    state->gamepad = gamepad;
    state->is_bluetooth =
        SDL_GetJoystickConnectionState(joystick) == SDL_JOYSTICK_CONNECTION_WIRELESS;
    state->user = ReadUserSettings();
    if (xbox_pad) {
        // Unknown connection: the safer Bluetooth pacing (20 ms ticks)
        const bool wired =
            SDL_GetJoystickConnectionState(joystick) == SDL_JOYSTICK_CONNECTION_WIRED;
        state->xbox = api.xbox_create(!wired);
        if (!state->xbox) {
            return nullptr;
        }
        const SDL_PropertiesID properties = SDL_GetJoystickProperties(joystick);
        state->xbox_has_rumble =
            SDL_GetBooleanProperty(properties, SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, false);
        state->xbox_triggers = SDL_GetBooleanProperty(
            properties, SDL_PROP_JOYSTICK_CAP_TRIGGER_RUMBLE_BOOLEAN, false);
        const char* gamepad_name = gamepad ? SDL_GetGamepadName(gamepad) : nullptr;
        if (!gamepad_name) {
            gamepad_name = SDL_GetJoystickName(joystick);
        }
        // The one line that says what this pad can do: which Windows input path SDL opened it
        // through decides whether the trigger motors exist at all
        LOG_INFO(Input,
                 "HD rumble: Xbox-style pad {:04X}:{:04X} \"{}\" through {} (grip motors: {}, "
                 "trigger motors: {}, wired: {})",
                 vid, pid, gamepad_name ? gamepad_name : "", XboxBackendName(joystick),
                 state->xbox_has_rumble ? "yes" : "no",
                 state->xbox_triggers ? "yes" : "not exposed on this path", wired);
    } else if (switch_pad) {
        state->engine = api.engine_create(state->is_bluetooth, pid == 0x2009);
        if (!state->engine) {
            return nullptr;
        }
        LOG_INFO(Input, "RenHDrum: HD Rumble for {:04X}:{:04X} (Bluetooth: {})", vid, pid,
                 state->is_bluetooth);
        // Wired and wireless are the only answers SDL can give when it owns the pad through
        // hidapi. Anything else means the pad came in through evdev -- on Linux that is the
        // kernel's hid-nintendo driver -- and evdev answers every raw rumble packet with
        // SDL_Unsupported, so the engine's packets never leave the host.
        const SDL_JoystickConnectionState connection = SDL_GetJoystickConnectionState(joystick);
        if (connection != SDL_JOYSTICK_CONNECTION_WIRED &&
            connection != SDL_JOYSTICK_CONNECTION_WIRELESS) {
            LOG_WARNING(Input,
                        "RenHDrum: SDL reports neither wired nor wireless for this pad, so it "
                        "likely arrived through the kernel's driver and will refuse raw rumble "
                        "packets");
        }
        state->ConfigureEngine();
    } else {
        state->dsp = api.dsp_create(state->user.dualsense_volume, state->user.dualsense_crisp_notes);
        if (!state->dsp) {
            return nullptr;
        }
        LOG_INFO(Input, "RenHDrum: DualSense haptics for {:04X}:{:04X}", vid, pid);
        state->OpenDualSenseAudio();
    }
    state->kind = xbox_pad ? 2 : switch_pad ? 0 : 1;
    ++connected_count[ConnectedCountIndex(state->kind)];
    if (state->xbox_triggers) {
        ++xbox_trigger_pads;
    }
    state->official_no_triggers = xbox_pad && vid == MicrosoftVendorId && !state->xbox_triggers;
    if (state->official_no_triggers) {
        ++xbox_official_no_triggers;
    }
    last_connected_kind = state->kind;
    return std::shared_ptr<RenHDrumPad>(new RenHDrumPad(std::move(state)));
}

int GetRenHDrumConnectedKind() {
    const int last = last_connected_kind.load();
    if (last >= 0 && connected_count[ConnectedCountIndex(last)].load() > 0) {
        return last;
    }
    for (int kind = 0; kind < static_cast<int>(connected_count.size()); ++kind) {
        if (connected_count[ConnectedCountIndex(kind)].load() > 0) {
            return kind;
        }
    }
    return -1;
}

int GetRenHDrumXboxTriggers() {
    if (connected_count[2].load() == 0) {
        return -1;
    }
    return xbox_trigger_pads.load() > 0 ? 1 : 0;
}

bool GetRenHDrumXboxNeedsRawInput() {
#ifdef _WIN32
    return xbox_official_no_triggers.load() > 0;
#else
    return false;
#endif
}

RenHDrumPad::RenHDrumPad(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}

RenHDrumPad::~RenHDrumPad() {
    Shutdown();
}

void RenHDrumPad::Shutdown() {
    SDL_Joystick* joystick = nullptr;
    bool had_engine = false;
    bool had_xbox = false;
    bool triggers = false;
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return;
        }
        joystick = impl->joystick;
        // Only what HD rumble was playing gets a stop from here; with the feature off the
        // controller is closed exactly as it was before this code existed
        had_xbox = impl->XboxDriving();
        had_engine = impl->Release() && impl->user.hd_enabled;
        if (impl->kind >= 0) {
            --connected_count[ConnectedCountIndex(impl->kind)];
            if (impl->xbox_triggers) {
                --xbox_trigger_pads;
            }
            if (impl->official_no_triggers) {
                --xbox_official_no_triggers;
            }
            impl->kind = -1;
        }
        triggers = impl->xbox_triggers;
        impl->joystick = nullptr;
        impl->gamepad = nullptr;
    }
    if (had_engine) {
        SendNeutralPacket(joystick);
    }
    if (had_xbox) {
        // Unconditional: the pad is about to be closed, or is already gone, and nothing may be
        // left playing. Failures are expected for an unplugged pad and are not logged.
        SDL_RumbleJoystick(joystick, 0, 0, 0);
        if (triggers) {
            SDL_RumbleJoystickTriggers(joystick, 0, 0, 0);
        }
    }
}

bool RenHDrumPad::IsActive() const {
    std::scoped_lock lock{impl->mutex};
    if (!impl->joystick) {
        return false;
    }
    if (impl->xbox) {
        return impl->XboxDriving();
    }
    if (impl->engine) {
        return impl->user.hd_enabled;
    }
    return impl->dsp && impl->DualSenseWanted() &&
           (impl->audio_stream || impl->bt_running.load());
}

bool RenHDrumPad::OwnsDualSenseHaptics() const {
    std::scoped_lock lock{impl->mutex};
    return impl->joystick && impl->dsp && impl->DualSenseWanted();
}

bool RenHDrumPad::Vibrate(std::int64_t now_us, bool is_right,
                          const Common::Input::VibrationStatus& status) {
    Impl::SettingsChange change{};
    SDL_Joystick* joystick = nullptr;
    bool handled = false;
    bool engine_sample = false;
    std::optional<Impl::XboxWrite> xbox_write;
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return false;
        }
        joystick = impl->joystick;
        change = impl->RefreshUserSettings(now_us);
        impl->next_settings_poll_us = now_us + SettingsPollUs;
        impl->halted = false;
        const auto& u = impl->user;
        const auto& api = Api();
        // Guest Switch 2 Mode (commands 213 / 214): the engine picks its translation law from it.
        // Process-wide, so only push it when it changed.
        if (api.set_nx2_mode) {
            const int mode = Settings::values.nx2_active_mode.load(std::memory_order_relaxed);
            if (mode != impl->nx2_mode) {
                api.set_nx2_mode(mode);
                impl->nx2_mode = mode;
            }
        }
        if (impl->xbox) {
            if (!impl->XboxDriving()) {
                // Off, or a pad without rumble: the standard rumble plays this vibration
            } else if (impl->XboxDirect()) {
                // Both halves share one pad: each motor follows the stronger side
                impl->sides[is_right ? 1 : 0] = HdRumbleSide{
                    .low_amplitude = status.low_amplitude,
                    .low_frequency = status.low_frequency,
                    .high_amplitude = status.high_amplitude,
                    .high_frequency = status.high_frequency,
                };
                XboxMotors target =
                    MixXboxSides(impl->sides[0], impl->sides[1], impl->DirectParams());
                if (!impl->XboxTriggersOn()) {
                    target.left_trigger = 0;
                    target.right_trigger = 0;
                }
                if (const auto send =
                        impl->scheduler.Offer(now_us, target, impl->SchedulerConfig())) {
                    xbox_write = impl->PlanXboxWrite(*send);
                }
                handled = true;
            } else {
                api.xbox_push_half_sample(impl->xbox, now_us, is_right,
                                          ShapeAmplitude(status.low_amplitude, u),
                                          status.low_frequency,
                                          ShapeAmplitude(status.high_amplitude, u),
                                          status.high_frequency);
                handled = true;
                engine_sample = true;
            }
        } else if (impl->engine) {
            if (u.hd_enabled) {
                api.engine_push_half_sample(impl->engine, now_us, is_right,
                                            ShapeAmplitude(status.low_amplitude, u),
                                            status.low_frequency,
                                            ShapeAmplitude(status.high_amplitude, u),
                                            status.high_frequency);
                handled = true;
                engine_sample = true;
            }
        } else if (impl->dsp) {
            if (impl->DualSenseWanted()) {
                api.dsp_set_half_target(impl->dsp, is_right,
                                        ShapeAmplitude(status.low_amplitude, u),
                                        status.low_frequency,
                                        ShapeAmplitude(status.high_amplitude, u),
                                        status.high_frequency);
                handled = impl->audio_stream != nullptr || impl->bt_running.load();
            } else {
                // Haptics off: silence the voice coils, the standard rumble takes over
                api.dsp_set_half_target(impl->dsp, is_right, 0.0f, status.low_frequency, 0.0f,
                                        status.high_frequency);
            }
        }
    }
    impl->ApplySettingsChange(joystick, change, now_us);
    if (xbox_write) {
        impl->NoteXboxWrite(Impl::WriteXbox(joystick, *xbox_write), now_us);
    }
    if (engine_sample) {
        Tick(now_us);
    }
    return handled;
}

void RenHDrumPad::Tick(std::int64_t now_us) {
    std::array<Uint8, 10> packet{};
    Impl::SettingsChange change{};
    SDL_Joystick* joystick = nullptr;
    bool open_audio = false;
    bool send = false;
    std::optional<Impl::XboxWrite> xbox_write;
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return;
        }
        joystick = impl->joystick;
        if (now_us >= impl->next_settings_poll_us) {
            // A held vibration is sent by the guest once, so a setting switched off in the
            // dialog would otherwise only take effect at the next vibration
            impl->next_settings_poll_us = now_us + SettingsPollUs;
            change = impl->RefreshUserSettings(now_us);
        }
        if (impl->dsp && impl->DualSenseWanted() && !impl->audio_stream &&
            !impl->audio_opening && impl->audio_attempts < DualSenseAudioMaxAttempts &&
            now_us >= impl->audio_next_us) {
            impl->audio_next_us = now_us + 2'000'000;
            open_audio = true;
        }
        if (!impl->halted) {
            send = impl->engine && impl->user.hd_enabled &&
                   Api().engine_tick(impl->engine, now_us, packet.data()) == 1;
            if (impl->XboxDriving()) {
                const auto config = impl->SchedulerConfig();
                std::optional<XboxRumbleSend> xbox_send;
                if (!impl->XboxDirect()) {
                    std::array<std::uint16_t, 4> motors{};
                    if (Api().xbox_tick(impl->xbox, now_us, motors.data()) == 1) {
                        // RenHDrum's mix paces and shapes its own output: every value it hands
                        // out is written. The scheduler adds the refresh and the stop.
                        xbox_send = impl->scheduler.Force(now_us, impl->ScaleMix(motors));
                    }
                }
                if (!xbox_send) {
                    xbox_send = impl->scheduler.Tick(now_us, config);
                }
                if (xbox_send) {
                    xbox_write = impl->PlanXboxWrite(*xbox_send);
                }
            }
        }
    }
    impl->ApplySettingsChange(joystick, change, now_us);
    if (xbox_write) {
        impl->NoteXboxWrite(Impl::WriteXbox(joystick, *xbox_write), now_us);
    }
    if (open_audio) {
        impl->OpenDualSenseAudio();
    }
    if (send) {
        if (!SDL_SendJoystickEffect(joystick, packet.data(), static_cast<int>(packet.size())) &&
            now_us >= impl->next_effect_error_us.load(std::memory_order_relaxed)) {
            // SDL answers a raw packet with an error when the pad was not opened through its
            // HIDAPI driver. Silence here reads as "the engine does nothing", so say it once in
            // a while instead.
            impl->next_effect_error_us.store(now_us + RumbleWarningIntervalUs,
                                             std::memory_order_relaxed);
            LOG_WARNING(Input, "RenHDrum: the controller refused the rumble packet: {}",
                        SDL_GetError());
        }
    }
}

void RenHDrumPad::Halt(std::int64_t now_us) {
    SDL_Joystick* joystick = nullptr;
    bool neutral = false;
    std::optional<Impl::XboxWrite> xbox_write;
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return;
        }
        joystick = impl->joystick;
        impl->halted = true;
        const auto& api = Api();
        // Only what HD rumble itself is playing: with the feature off the standard rumble is
        // left exactly as it was before this code existed
        if (impl->xbox) {
            if (impl->XboxDriving()) {
                xbox_write = impl->PlanXboxWrite(impl->StopXbox(now_us));
                // Triggers get their zero even when nothing is known to be playing on them
                xbox_write->triggers = impl->xbox_triggers;
                impl->triggers_running = false;
            }
        } else if (impl->engine) {
            if (impl->user.hd_enabled) {
                api.engine_push_half_sample(impl->engine, now_us, false, 0.0f, 160.0f, 0.0f,
                                            320.0f);
                api.engine_push_half_sample(impl->engine, now_us, true, 0.0f, 160.0f, 0.0f,
                                            320.0f);
                neutral = true;
            }
        } else if (impl->dsp) {
            api.dsp_set_half_target(impl->dsp, false, 0.0f, 160.0f, 0.0f, 320.0f);
            api.dsp_set_half_target(impl->dsp, true, 0.0f, 160.0f, 0.0f, 320.0f);
        }
    }
    if (xbox_write) {
        impl->NoteXboxWrite(Impl::WriteXbox(joystick, *xbox_write), now_us);
    }
    if (neutral) {
        SendNeutralPacket(joystick);
    }
}

} // namespace InputCommon

#else // ENABLE_RENHDRUM

namespace InputCommon {

bool IsRenHDrumBuilt() {
    return false;
}

bool IsRenHDrumLoaded() {
    return false;
}

int GetRenHDrumConnectedKind() {
    return -1;
}

int GetRenHDrumXboxTriggers() {
    return -1;
}

bool GetRenHDrumXboxNeedsRawInput() {
    return false;
}

struct RenHDrumPad::Impl {};

std::shared_ptr<RenHDrumPad> RenHDrumPad::Create(SDL_Joystick*, SDL_Gamepad*) {
    return nullptr;
}

RenHDrumPad::RenHDrumPad(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}
RenHDrumPad::~RenHDrumPad() = default;
void RenHDrumPad::Shutdown() {}
bool RenHDrumPad::IsActive() const {
    return false;
}
bool RenHDrumPad::OwnsDualSenseHaptics() const {
    return false;
}
bool RenHDrumPad::Vibrate(std::int64_t, bool, const Common::Input::VibrationStatus&) {
    return false;
}
void RenHDrumPad::Tick(std::int64_t) {}
void RenHDrumPad::Halt(std::int64_t) {}

} // namespace InputCommon

#endif // ENABLE_RENHDRUM
