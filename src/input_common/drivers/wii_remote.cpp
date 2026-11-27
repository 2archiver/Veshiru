// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "input_common/drivers/wii_remote.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "common/logging.h"
#include "common/settings.h"

namespace InputCommon {

namespace {

constexpr u16 NintendoVendorId = 0x057E;
constexpr u16 WiiRemoteProductId = 0x0306;
constexpr u16 WiiRemotePlusProductId = 0x0330;

constexpr std::size_t MaxRemotes = 4;
constexpr int ReadTimeoutMs = 20;
constexpr auto StatusPollInterval = std::chrono::seconds{2};
constexpr auto PointerTimeout = std::chrono::milliseconds{250};

constexpr u32 IrCommandRegister = 0xB00030;
constexpr u32 IrSensitivityBlock1 = 0xB00000;
constexpr u32 IrSensitivityBlock2 = 0xB0001A;
constexpr u32 IrModeRegister = 0xB00033;

Common::UUID MakeGuid(std::string_view path) {
    // Stable per device path, like the other HID engines. FNV-1a twice with different seeds.
    std::array<u8, 16> bytes{};
    u64 first = 1469598103934665603ULL;
    u64 second = 1099511628211ULL;
    for (const char c : path) {
        const u8 byte = static_cast<u8>(c);
        first = (first ^ byte) * 1099511628211ULL;
        second = (second ^ byte) * 1469598103934665603ULL;
    }
    std::memcpy(bytes.data(), &first, sizeof(first));
    std::memcpy(bytes.data() + sizeof(first), &second, sizeof(second));
    if (bytes == std::array<u8, 16>{}) {
        bytes[0] = 0x57;
    }
    return Common::UUID{bytes};
}

Common::Input::BatteryLevel BatteryLevelFromStatus(const Wii::Status& status) {
    if (status.battery_low) {
        return Common::Input::BatteryLevel::Low;
    }
    if (status.battery_level >= 0xC0) {
        return Common::Input::BatteryLevel::Full;
    }
    if (status.battery_level >= 0x80) {
        return Common::Input::BatteryLevel::Medium;
    }
    if (status.battery_level >= 0x40) {
        return Common::Input::BatteryLevel::Low;
    }
    return Common::Input::BatteryLevel::Critical;
}

} // namespace

struct WiiRemote::Remote {
    Remote(WiiRemote& owner_, std::size_t port_) : owner{owner_}, port{port_} {}

    ~Remote() {
        Stop();
    }

    bool Open(SDL_hid_device_info* info) {
        path = info->path != nullptr ? info->path : "";
        guid = MakeGuid(path);
        identifier = PadIdentifier{
            .guid = guid,
            .port = port,
            .pad = 0,
        };

        handle = info->path != nullptr ? SDL_hid_open_path(info->path)
                                       : SDL_hid_open(info->vendor_id, info->product_id,
                                                      info->serial_number);
        if (handle == nullptr) {
            LOG_WARNING(Input, "Wii Remote at {} could not be opened", path);
            return false;
        }
        SDL_hid_set_nonblocking(handle, 0);
        owner.RegisterController(*this);
        LOG_INFO(Input, "Wii Remote {} connected on port {}", path, port + 1);
        return true;
    }

    void Send(std::span<const u8> report) {
        if (handle == nullptr) {
            return;
        }
        std::scoped_lock lock{write_mutex};
        if (SDL_hid_write(handle, report.data(), report.size()) < 0) {
            LOG_DEBUG(Input, "Wii Remote write failed on port {}", port + 1);
        }
    }

    void SendSlow(std::span<const u8> report) {
        Send(report);
        std::this_thread::sleep_for(std::chrono::milliseconds{25});
    }

    void ReadExtensionId() {
        Send(Wii::BuildReadExtensionIdReport(rumble));
    }

    void ApplyReportingMode() {
        const auto mode = Settings::values.wii_remote_ir_pointer.GetValue()
                              ? Wii::ReportId::ButtonsAccelIr10Extension6
                              : Wii::ReportId::ButtonsAccelExtension16;
        Send(Wii::BuildDataReportingReport(mode, true, rumble));
    }

    void EnableIr() {
        ir_enabled = true;
        SendSlow(Wii::BuildIrEnableReport(false, true));
        SendSlow(Wii::BuildIrEnableReport(true, true));

        const auto write = [this](u32 address, std::span<const u8> data) {
            SendSlow(Wii::BuildWriteRegisterReport(address, data, rumble));
        };
        const std::array<u8, 1> start{0x08};
        const std::array<u8, 9> sensitivity_block_1{
            0x02, 0x00, 0x00, 0x71, 0x01, 0x00, 0x96, 0x00, 0xB4,
        };
        const std::array<u8, 2> sensitivity_block_2{0xB3, 0x04};
        const std::array<u8, 1> basic_mode{0x01};

        write(IrCommandRegister, start);
        write(IrSensitivityBlock1, sensitivity_block_1);
        write(IrSensitivityBlock2, sensitivity_block_2);
        write(IrModeRegister, basic_mode);
        write(IrCommandRegister, start);
    }

    void Start() {
        SendSlow(Wii::BuildStatusRequest(false));
        ReadExtensionId();
        SendSlow(Wii::BuildLedReport(leds, false));
        if (Settings::values.wii_remote_ir_pointer.GetValue()) {
            EnableIr();
        }
        ApplyReportingMode();
        last_status = std::chrono::steady_clock::now();
        thread = std::jthread([this](std::stop_token token) { InputLoop(token); });
    }

    void Stop() {
        connected = false;
        if (thread.joinable()) {
            thread.request_stop();
            thread.join();
        }
        if (handle != nullptr) {
            SDL_hid_close(handle);
            handle = nullptr;
        }
    }

    void InputLoop(std::stop_token stop_token) {
        std::array<u8, 64> buffer{};
        int error_count = 0;
        while (!stop_token.stop_requested() && connected) {
            const int read = SDL_hid_read_timeout(handle, buffer.data(), buffer.size(),
                                                  ReadTimeoutMs);
            if (read < 0) {
                if (++error_count > 50) {
                    LOG_WARNING(Input, "Wii Remote on port {} stopped responding", port + 1);
                    connected = false;
                    break;
                }
            } else {
                error_count = 0;
                if (read > 0) {
                    HandleReport(std::span<const u8>{buffer.data(), static_cast<std::size_t>(read)});
                }
            }

            const auto now = std::chrono::steady_clock::now();
            if (now - last_status >= StatusPollInterval) {
                last_status = now;
                Send(Wii::BuildStatusRequest(rumble));
            }
            if (pointer_seen && now - last_pointer >= PointerTimeout) {
                pointer_seen = false;
                owner.PublishAxis(*this, Wii::IrPointerX, 0.0f);
                owner.PublishAxis(*this, Wii::IrPointerY, 0.0f);
            }
        }
    }

    void HandleReport(std::span<const u8> bytes) {
        const auto parsed = Wii::ParseInputReport(bytes);
        if (!parsed.has_value()) {
            return;
        }
        const Wii::InputReport& report = *parsed;

        if (report.has_extension) {
            const bool changed = !extension.valid || extension.value != report.extension.value;
            extension = report.extension;
            if (changed) {
                LOG_INFO(Input, "Wii Remote port {} extension {:012X}{}", port + 1,
                         static_cast<unsigned long long>(extension.value),
                         extension.nunchuk ? " (Nunchuk)" : "");
                owner.PublishExtensions(*this);
                // Connecting or removing an extension disables data reporting.
                ApplyReportingMode();
            }
        }

        if (report.has_status) {
            owner.PublishBattery(*this, BatteryLevelFromStatus(report.status));
            const bool extension_now = report.status.extension_connected;
            if (extension_now != extension_connected) {
                extension_connected = extension_now;
                if (!extension_now) {
                    extension = {};
                    owner.PublishExtensions(*this);
                } else {
                    ReadExtensionId();
                }
                ApplyReportingMode();
            }
        }

        owner.Publish(*this, report);
    }

    WiiRemote& owner;
    SDL_hid_device* handle{};
    std::string path;
    Common::UUID guid{};
    std::size_t port{};
    PadIdentifier identifier{};
    std::atomic<bool> connected{true};
    std::jthread thread;
    std::mutex write_mutex;

    bool rumble{};
    u8 leds{1};
    bool ir_enabled{};
    bool extension_connected{};
    Wii::ExtensionId extension{};
    std::array<bool, Wii::ButtonCount> button_state{};
    std::chrono::steady_clock::time_point last_status{};
    std::chrono::steady_clock::time_point last_pointer{};
    bool pointer_seen{};
    u64 last_motion_us{};
};

WiiRemote::WiiRemote(std::string input_engine_) : InputEngine(std::move(input_engine_)) {}

WiiRemote::~WiiRemote() {
    Reset();
}

void WiiRemote::Setup() {
    if (!Settings::values.enable_wii_remote_driver.GetValue()) {
        return;
    }
    if (SDL_hid_init() != 0) {
        LOG_ERROR(Input, "Wii Remote driver could not initialize hidapi");
        return;
    }
    hid_initialized = true;
    scan_thread = std::jthread([this](std::stop_token token) { ScanLoop(token); });
}

void WiiRemote::Reset() {
    if (scan_thread.joinable()) {
        scan_thread.request_stop();
        scan_thread.join();
    }
    {
        std::scoped_lock lock{remotes_mutex};
        for (const auto& remote : remotes) {
            remote->Stop();
        }
        remotes.clear();
    }
    if (hid_initialized) {
        SDL_hid_exit();
        hid_initialized = false;
    }
}

void WiiRemote::ScanLoop(std::stop_token stop_token) {
    while (!stop_token.stop_requested()) {
        Scan();
        for (int i = 0; i < 20 && !stop_token.stop_requested(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
    }
}

void WiiRemote::Scan() {
    std::vector<std::unique_ptr<Remote>> removed;
    {
        std::scoped_lock lock{remotes_mutex};
        auto it = remotes.begin();
        while (it != remotes.end()) {
            if (!(*it)->connected) {
                removed.push_back(std::move(*it));
                it = remotes.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const auto& remote : removed) {
        ReleaseController(*remote);
        remote->Stop();
    }

    SDL_hid_device_info* list = SDL_hid_enumerate(NintendoVendorId, 0);
    for (auto* info = list; info != nullptr; info = info->next) {
        if (info->product_id != WiiRemoteProductId && info->product_id != WiiRemotePlusProductId) {
            continue;
        }
        const std::string path = info->path != nullptr ? info->path : "";

        std::size_t port = MaxRemotes;
        {
            std::scoped_lock lock{remotes_mutex};
            if (remotes.size() >= MaxRemotes) {
                break;
            }
            const bool known = std::ranges::any_of(remotes, [&path](const auto& remote) {
                return remote->path == path;
            });
            if (known) {
                continue;
            }
            std::array<bool, MaxRemotes> used{};
            for (const auto& remote : remotes) {
                if (remote->port < used.size()) {
                    used[remote->port] = true;
                }
            }
            for (std::size_t i = 0; i < used.size(); ++i) {
                if (!used[i]) {
                    port = i;
                    break;
                }
            }
        }
        if (port >= MaxRemotes) {
            continue;
        }

        auto remote = std::make_unique<Remote>(*this, port);
        if (!remote->Open(info)) {
            continue;
        }
        remote->Start();
        std::scoped_lock lock{remotes_mutex};
        if (remotes.size() < MaxRemotes) {
            remotes.push_back(std::move(remote));
        }
    }
    SDL_hid_free_enumeration(list);
}

void WiiRemote::RegisterController(Remote& remote) {
    PreSetController(remote.identifier);
    for (std::size_t i = 0; i < Wii::ButtonCount; ++i) {
        PreSetButton(remote.identifier, static_cast<int>(i));
    }
    for (int axis = 0; axis < Wii::AxisCount; ++axis) {
        PreSetAxis(remote.identifier, axis);
    }
    PreSetMotion(remote.identifier, 0);
}

void WiiRemote::ReleaseController(Remote& remote) {
    for (std::size_t i = 0; i < Wii::ButtonCount; ++i) {
        remote.button_state[i] = false;
        SetButton(remote.identifier, static_cast<int>(i), false);
    }
    for (int axis = 0; axis < Wii::AxisCount; ++axis) {
        SetAxis(remote.identifier, axis, 0.0f);
    }
    SetMotion(remote.identifier, 0, BasicMotion{});
    remote.pointer_seen = false;
    remote.last_motion_us = 0;
    SetBattery(remote.identifier, Common::Input::BatteryLevel::None);
}

void WiiRemote::PublishAxis(Remote& remote, int axis, f32 value) {
    SetAxis(remote.identifier, axis, value);
}

void WiiRemote::PublishBattery(Remote& remote, Common::Input::BatteryLevel level) {
    SetBattery(remote.identifier, level);
}

void WiiRemote::PublishExtensions(Remote& remote) {
    const bool nunchuk = remote.extension.nunchuk;
    const auto set = [this, &remote](Wii::Button button, bool pressed) {
        const auto index = static_cast<std::size_t>(button);
        if (remote.button_state[index] == pressed) {
            return;
        }
        remote.button_state[index] = pressed;
        SetButton(remote.identifier, static_cast<int>(index), pressed);
    };
    set(Wii::Button::NunchukC, false);
    set(Wii::Button::NunchukZ, false);
    if (!nunchuk) {
        SetAxis(remote.identifier, Wii::NunchukStickX, 0.0f);
        SetAxis(remote.identifier, Wii::NunchukStickY, 0.0f);
    }
}

void WiiRemote::Publish(Remote& remote, const Wii::InputReport& report) {
    const auto set = [this, &remote](Wii::Button button, bool pressed) {
        const auto index = static_cast<std::size_t>(button);
        if (remote.button_state[index] == pressed) {
            return;
        }
        remote.button_state[index] = pressed;
        SetButton(remote.identifier, static_cast<int>(index), pressed);
    };

    if (report.has_buttons) {
        const auto& buttons = report.buttons;
        set(Wii::Button::A, buttons.a);
        set(Wii::Button::B, buttons.b);
        set(Wii::Button::One, buttons.one);
        set(Wii::Button::Two, buttons.two);
        set(Wii::Button::Plus, buttons.plus);
        set(Wii::Button::Minus, buttons.minus);
        set(Wii::Button::Home, buttons.home);
        set(Wii::Button::Left, buttons.left);
        set(Wii::Button::Right, buttons.right);
        set(Wii::Button::Up, buttons.up);
        set(Wii::Button::Down, buttons.down);
    }

    if (Wii::HasNunchukData(remote.extension, report)) {
        set(Wii::Button::NunchukC, report.nunchuk.c);
        set(Wii::Button::NunchukZ, report.nunchuk.z);
        SetAxis(remote.identifier, Wii::NunchukStickX, report.nunchuk.stick_x);
        SetAxis(remote.identifier, Wii::NunchukStickY, report.nunchuk.stick_y);
    } else if (!remote.extension.nunchuk) {
        // Keep extension bytes from another accessory (for example a Classic Controller) from
        // being interpreted as Nunchuk state. Reports without Nunchuk data while a Nunchuk is
        // connected intentionally retain the most recent stick and button state.
        set(Wii::Button::NunchukC, false);
        set(Wii::Button::NunchukZ, false);
        SetAxis(remote.identifier, Wii::NunchukStickX, 0.0f);
        SetAxis(remote.identifier, Wii::NunchukStickY, 0.0f);
    }

    if (report.has_accel) {
        const auto g = Wii::AccelToG(report.accel);
        const auto now = std::chrono::steady_clock::now();
        const u64 now_us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
        const u64 delta_us = remote.last_motion_us == 0 ? 0 : now_us - remote.last_motion_us;
        remote.last_motion_us = now_us;
        SetMotion(remote.identifier, 0,
                  BasicMotion{
                      .gyro_x = 0.0f,
                      .gyro_y = 0.0f,
                      .gyro_z = 0.0f,
                      .accel_x = g[0],
                      .accel_y = g[1],
                      .accel_z = g[2],
                      .delta_timestamp = delta_us,
                  });
    }

    if (report.has_ir) {
        if (report.ir.pointer_valid) {
            remote.pointer_seen = true;
            remote.last_pointer = std::chrono::steady_clock::now();
            SetAxis(remote.identifier, Wii::IrPointerX, report.ir.pointer_x * 2.0f - 1.0f);
            SetAxis(remote.identifier, Wii::IrPointerY, -(report.ir.pointer_y * 2.0f - 1.0f));
        } else if (remote.pointer_seen &&
                   std::chrono::steady_clock::now() - remote.last_pointer >= PointerTimeout) {
            remote.pointer_seen = false;
            SetAxis(remote.identifier, Wii::IrPointerX, 0.0f);
            SetAxis(remote.identifier, Wii::IrPointerY, 0.0f);
        }
    }
}

Common::Input::DriverResult WiiRemote::SetLeds(const PadIdentifier& identifier,
                                               const Common::Input::LedStatus& led_status) {
    std::scoped_lock lock{remotes_mutex};
    for (const auto& remote : remotes) {
        if (!(remote->identifier == identifier) || !remote->connected) {
            continue;
        }
        const u8 leds = static_cast<u8>((led_status.led_1 ? 0x1 : 0x0) |
                                        (led_status.led_2 ? 0x2 : 0x0) |
                                        (led_status.led_3 ? 0x4 : 0x0) |
                                        (led_status.led_4 ? 0x8 : 0x0));
        remote->leds = leds == 0 ? 0x1 : leds;
        remote->Send(Wii::BuildLedReport(remote->leds, remote->rumble));
        return Common::Input::DriverResult::Success;
    }
    return Common::Input::DriverResult::InvalidHandle;
}

Common::Input::DriverResult WiiRemote::SetVibration(
    const PadIdentifier& identifier, const Common::Input::VibrationStatus& vibration) {
    std::scoped_lock lock{remotes_mutex};
    for (const auto& remote : remotes) {
        if (!(remote->identifier == identifier) || !remote->connected) {
            continue;
        }
        const bool active = vibration.low_amplitude > 0.02f || vibration.high_amplitude > 0.02f;
        if (remote->rumble != active) {
            remote->rumble = active;
            remote->Send(Wii::BuildRumbleOnlyReport(active));
        }
        return Common::Input::DriverResult::Success;
    }
    return Common::Input::DriverResult::InvalidHandle;
}

bool WiiRemote::IsVibrationEnabled(const PadIdentifier& identifier) {
    std::scoped_lock lock{remotes_mutex};
    return std::ranges::any_of(remotes, [&identifier](const auto& remote) {
        return remote->identifier == identifier && remote->connected;
    });
}

std::vector<Common::ParamPackage> WiiRemote::GetInputDevices() const {
    std::scoped_lock lock{remotes_mutex};
    std::vector<Common::ParamPackage> devices;
    devices.reserve(remotes.size());
    for (const auto& remote : remotes) {
        if (!remote->connected) {
            continue;
        }
        Common::ParamPackage params;
        params.Set("engine", GetEngineName());
        params.Set("display", "Wii Remote " + std::to_string(remote->port + 1) +
                                  (remote->extension.nunchuk ? " (Nunchuk)" : ""));
        params.Set("guid", remote->guid.RawString());
        params.Set("port", static_cast<int>(remote->port));
        devices.emplace_back(std::move(params));
    }
    return devices;
}

Common::ParamPackage WiiRemote::MakeButtonPackage(const Remote& remote, int button) const {
    Common::ParamPackage params;
    params.Set("engine", GetEngineName());
    params.Set("guid", remote.guid.RawString());
    params.Set("port", static_cast<int>(remote.port));
    params.Set("button", button);
    return params;
}

Common::ParamPackage WiiRemote::MakeAnalogPackage(const Remote& remote, int axis_x,
                                                  int axis_y) const {
    Common::ParamPackage params;
    params.Set("engine", GetEngineName());
    params.Set("guid", remote.guid.RawString());
    params.Set("port", static_cast<int>(remote.port));
    params.Set("axis_x", axis_x);
    params.Set("axis_y", axis_y);
    params.Set("deadzone", 0.15f);
    params.Set("range", 1.0f);
    params.Set("threshold", 0.0f);
    params.Set("offset_x", 0.0f);
    params.Set("offset_y", 0.0f);
    params.Set("invert_x", "+");
    params.Set("invert_y", "+");
    return params;
}

ButtonMapping WiiRemote::GetButtonMappingForDevice(const Common::ParamPackage& params) {
    if (!params.Has("guid") || !params.Has("port")) {
        return {};
    }
    std::scoped_lock lock{remotes_mutex};
    const auto it = std::ranges::find_if(remotes, [&params](const auto& remote) {
        return remote->guid.RawString() == params.Get("guid", "") &&
               static_cast<int>(remote->port) == params.Get("port", -1);
    });
    if (it == remotes.end()) {
        return {};
    }
    const Remote& remote = **it;

    ButtonMapping mapping;
    const auto add = [&](Settings::NativeButton::Values native, Wii::Button button) {
        mapping.insert_or_assign(native,
                                 MakeButtonPackage(remote, static_cast<int>(button)));
    };
    add(Settings::NativeButton::A, Wii::Button::A);
    add(Settings::NativeButton::B, Wii::Button::B);
    add(Settings::NativeButton::X, Wii::Button::One);
    add(Settings::NativeButton::Y, Wii::Button::Two);
    add(Settings::NativeButton::Plus, Wii::Button::Plus);
    add(Settings::NativeButton::Minus, Wii::Button::Minus);
    add(Settings::NativeButton::Home, Wii::Button::Home);
    add(Settings::NativeButton::DLeft, Wii::Button::Left);
    add(Settings::NativeButton::DRight, Wii::Button::Right);
    add(Settings::NativeButton::DUp, Wii::Button::Up);
    add(Settings::NativeButton::DDown, Wii::Button::Down);
    add(Settings::NativeButton::ZL, Wii::Button::NunchukC);
    add(Settings::NativeButton::ZR, Wii::Button::NunchukZ);
    return mapping;
}

AnalogMapping WiiRemote::GetAnalogMappingForDevice(const Common::ParamPackage& params) {
    if (!params.Has("guid") || !params.Has("port")) {
        return {};
    }
    std::scoped_lock lock{remotes_mutex};
    const auto it = std::ranges::find_if(remotes, [&params](const auto& remote) {
        return remote->guid.RawString() == params.Get("guid", "") &&
               static_cast<int>(remote->port) == params.Get("port", -1);
    });
    if (it == remotes.end()) {
        return {};
    }
    const Remote& remote = **it;

    AnalogMapping mapping;
    mapping.insert_or_assign(Settings::NativeAnalog::LStick,
                             MakeAnalogPackage(remote, Wii::NunchukStickX, Wii::NunchukStickY));
    auto pointer = MakeAnalogPackage(remote, Wii::IrPointerX, Wii::IrPointerY);
    pointer.Set("deadzone", 0.0f);
    mapping.insert_or_assign(Settings::NativeAnalog::RStick, std::move(pointer));
    return mapping;
}

MotionMapping WiiRemote::GetMotionMappingForDevice(const Common::ParamPackage& params) {
    if (!params.Has("guid") || !params.Has("port")) {
        return {};
    }
    Common::ParamPackage motion;
    motion.Set("engine", GetEngineName());
    motion.Set("guid", params.Get("guid", ""));
    motion.Set("port", params.Get("port", 0));
    motion.Set("motion", 0);

    MotionMapping mapping;
    mapping.insert_or_assign(Settings::NativeMotion::MotionLeft, motion);
    mapping.insert_or_assign(Settings::NativeMotion::MotionRight, std::move(motion));
    return mapping;
}

Common::Input::ButtonNames WiiRemote::GetUIName(const Common::ParamPackage& params) const {
    if (params.Has("button")) {
        switch (static_cast<Wii::Button>(params.Get("button", 0))) {
        case Wii::Button::A:
            return Common::Input::ButtonNames::ButtonA;
        case Wii::Button::B:
            return Common::Input::ButtonNames::ButtonB;
        case Wii::Button::One:
            return Common::Input::ButtonNames::ButtonX;
        case Wii::Button::Two:
            return Common::Input::ButtonNames::ButtonY;
        case Wii::Button::Plus:
            return Common::Input::ButtonNames::ButtonPlus;
        case Wii::Button::Minus:
            return Common::Input::ButtonNames::ButtonMinus;
        case Wii::Button::Home:
            return Common::Input::ButtonNames::ButtonHome;
        case Wii::Button::Left:
            return Common::Input::ButtonNames::ButtonLeft;
        case Wii::Button::Right:
            return Common::Input::ButtonNames::ButtonRight;
        case Wii::Button::Up:
            return Common::Input::ButtonNames::ButtonUp;
        case Wii::Button::Down:
            return Common::Input::ButtonNames::ButtonDown;
        case Wii::Button::NunchukC:
            return Common::Input::ButtonNames::TriggerZL;
        case Wii::Button::NunchukZ:
            return Common::Input::ButtonNames::TriggerZR;
        case Wii::Button::Count:
            break;
        }
        return Common::Input::ButtonNames::Undefined;
    }
    if (params.Has("axis_x") || params.Has("axis")) {
        return Common::Input::ButtonNames::Value;
    }
    if (params.Has("motion")) {
        return Common::Input::ButtonNames::Engine;
    }
    return Common::Input::ButtonNames::Invalid;
}

} // namespace InputCommon
