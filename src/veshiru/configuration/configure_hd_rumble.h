// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ported from the hdrumble-mvp branch of https://github.com/Octelly/Eglantine (samaBR85, hermes).

#pragma once

#include <QDialog>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QListWidget;
class QSlider;
class QSpinBox;
class QStackedWidget;

/// Settings for HD rumble through RenHDrum: Xbox-style pads, Switch Pro Controller / Joy-Con and
/// the DualSense voice coil haptics. The SDL driver reads them on every vibration and about ten
/// times a second besides, so connected controllers pick up a change as soon as the dialog is
/// accepted, and switching it off silences a motor that is running.
class ConfigureHdRumble : public QDialog {
    Q_OBJECT

public:
    explicit ConfigureHdRumble(QWidget* parent);
    ~ConfigureHdRumble() override;

    void ApplyConfiguration();

private:
    QWidget* CreateSwitchPage();
    QWidget* CreateDualSensePage();
    QWidget* CreateXboxPage();
    void LoadConfiguration();
    void RestoreDefaults();
    void UpdateEnabledState();

    QCheckBox* hd_enabled{};
    QSlider* gain_slider{};
    QSpinBox* gain{};
    QSlider* boost_slider{};
    QSpinBox* boost{};
    QListWidget* page_list{};
    QStackedWidget* pages{};

    QGroupBox* wired_group{};
    QSpinBox* usb_interval{};
    QCheckBox* envelope_5ms{};
    QCheckBox* frequency_in_packet{};
    QCheckBox* crisp_notes{};

    QGroupBox* bt_group{};
    QComboBox* bt_level{};
    QCheckBox* bt_after_input{};
    QCheckBox* bt_fast{};
    QCheckBox* bt_crisp_notes{};

    QCheckBox* ds_enabled{};
    QSlider* ds_volume_slider{};
    QSpinBox* ds_volume{};
    QCheckBox* ds_crisp_notes{};

    QCheckBox* xbox_enabled{};
    QComboBox* xbox_mode{};
    QSlider* xbox_detail_slider{};
    QSpinBox* xbox_detail{};
    QSlider* xbox_accents_slider{};
    QSpinBox* xbox_accents{};
    QCheckBox* xbox_triggers_enabled{};
    QSlider* xbox_trigger_slider{};
    QSpinBox* xbox_trigger{};
    /// 1 when a connected Xbox-style pad exposes trigger motors, 0 when none does, -1 when no
    /// such pad is connected. Read once, when the dialog opens.
    int xbox_triggers_state{-1};
    QSlider* xbox_strength_slider{};
    QSpinBox* xbox_strength{};
};
