// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ported from the hdrumble-mvp branch of https://github.com/Octelly/Eglantine (samaBR85, hermes).
// Veshiru's changes: the master switch defaults to off, the trigger motors have their own switch,
// the motor strength is shown as the percentage that is stored, and the notes say what an
// eccentric-mass motor cannot play.

#include <algorithm>

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QVBoxLayout>

#include "common/settings.h"
#include "input_common/drivers/renhdrum_pad.h"
#include "veshiru/configuration/configure_hd_rumble.h"

namespace {

QLabel* MakeNote(const QString& text) {
    auto* label = new QLabel(text);
    label->setWordWrap(true);
    label->setStyleSheet(QStringLiteral("color: palette(mid);"));
    return label;
}

/// A slider and a spin box that follow each other, with a percent sign.
QHBoxLayout* MakePercentRow(QSlider*& slider, QSpinBox*& box, int minimum, int maximum, int step,
                            const QString& tip) {
    slider = new QSlider(Qt::Horizontal);
    slider->setRange(minimum, maximum);
    slider->setToolTip(tip);
    box = new QSpinBox;
    box->setRange(minimum, maximum);
    box->setSingleStep(step);
    box->setToolTip(tip);
    QObject::connect(slider, &QSlider::valueChanged, box, &QSpinBox::setValue);
    QObject::connect(box, qOverload<int>(&QSpinBox::valueChanged), slider, &QSlider::setValue);
    auto* row = new QHBoxLayout;
    row->addWidget(slider, 1);
    row->addWidget(box);
    row->addWidget(new QLabel(QStringLiteral("%")));
    return row;
}

} // namespace

ConfigureHdRumble::ConfigureHdRumble(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("HD Rumble"));
    setMinimumSize(660, 520);

    auto* root = new QVBoxLayout(this);

    // Header: master switch
    auto* header = new QHBoxLayout;
    auto* title = new QLabel(tr("<b>HD Rumble</b>"));
    hd_enabled = new QCheckBox(tr("Enabled"));
    hd_enabled->setToolTip(tr("Off (the default): every controller uses the standard rumble."));
    header->addWidget(title);
    header->addStretch();
    header->addWidget(hd_enabled);
    root->addLayout(header);
    if (!InputCommon::IsRenHDrumLoaded()) {
        auto* missing = new QLabel(
            tr("renhdrum.dll was not found next to the emulator, so the controllers use the "
               "standard rumble whatever is set here. It ships with Veshiru's release packages; "
               "it can also be taken from <a href=\"https://github.com/trzz1214/RenHDrum/"
               "releases\">the RenHDrum releases</a>. Restart the emulator after adding it."));
        missing->setWordWrap(true);
        missing->setOpenExternalLinks(true);
        root->addWidget(missing);
    }
    root->addWidget(MakeNote(
        tr("While HD Rumble plays a controller, the \"Enable Accurate Vibration\" filter in the "
           "vibration dialog is bypassed for it: this engine paces its own output, and the 15 ms "
           "filter would drop the short ticks Switch 2 games send.")));

    // Intensity: all controllers
    auto* gain_row = new QHBoxLayout;
    gain_row->addWidget(new QLabel(tr("Intensity")));
    gain_row->addLayout(
        MakePercentRow(gain_slider, gain, 25, 1000, 25,
                       tr("Multiplies the game's rumble before anything else. Some Switch 2 "
                          "games send very weak values.")),
        1);
    root->addLayout(gain_row);

    auto* boost_row = new QHBoxLayout;
    boost_row->addWidget(new QLabel(tr("Weak rumble boost")));
    boost_row->addLayout(
        MakePercentRow(boost_slider, boost, 0, 100, 5,
                       tr("Lifts weak vibrations more than strong ones. 0% keeps the game's "
                          "curve.")),
        1);
    root->addLayout(boost_row);

    // Body: controller list on the left, its settings on the right. The order of the pages is
    // the controller kind GetRenHDrumConnectedKind reports.
    auto* body = new QHBoxLayout;
    page_list = new QListWidget;
    page_list->addItem(tr("Switch Controller"));
    page_list->addItem(tr("DualSense"));
    page_list->addItem(tr("Xbox"));
    page_list->setFixedWidth(170);
    pages = new QStackedWidget;
    pages->addWidget(CreateSwitchPage());
    pages->addWidget(CreateDualSensePage());
    pages->addWidget(CreateXboxPage());
    connect(page_list, &QListWidget::currentRowChanged, pages, &QStackedWidget::setCurrentIndex);
    // Open on the page of the controller in use, marked as connected; the Xbox page otherwise
    constexpr int xbox_page = 2;
    const int connected = InputCommon::GetRenHDrumConnectedKind();
    if (connected >= 0 && connected < page_list->count()) {
        auto* item = page_list->item(connected);
        item->setText(tr("%1 (connected)").arg(item->text()));
    }
    page_list->setCurrentRow(connected >= 0 && connected < page_list->count() ? connected
                                                                              : xbox_page);
    body->addWidget(page_list);
    body->addWidget(pages, 1);
    root->addLayout(body, 1);

    root->addWidget(MakeNote(
        tr("Changes reach connected controllers within a moment of pressing OK. To feel them, "
           "open the Vibration settings and hold a button on the controller.")));

    auto* footer = new QHBoxLayout;
    auto* credit = new QLabel(tr("Powered by RenHDrum"));
    credit->setStyleSheet(QStringLiteral("color: palette(mid);"));
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
                                         QDialogButtonBox::RestoreDefaults);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked, this,
            &ConfigureHdRumble::RestoreDefaults);
    footer->addWidget(credit);
    footer->addStretch();
    footer->addWidget(buttons);
    root->addLayout(footer);

    connect(hd_enabled, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);
    connect(ds_enabled, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);
    connect(xbox_enabled, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);
    connect(xbox_triggers_enabled, &QCheckBox::toggled, this,
            &ConfigureHdRumble::UpdateEnabledState);
    connect(xbox_mode, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureHdRumble::UpdateEnabledState);
    connect(envelope_5ms, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);

    LoadConfiguration();
}

ConfigureHdRumble::~ConfigureHdRumble() = default;

QWidget* ConfigureHdRumble::CreateSwitchPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(MakeNote(tr("Pro Controller and Joy-Con: the game's HD Rumble sent with "
                                  "full detail instead of a plain motor rumble.")));
    // The built-in driver is a different path entirely: it talks to the controller over raw HID
    // and never goes through SDL, so this engine never sees those pads.
    layout->addWidget(MakeNote(tr("Veshiru's built-in Joy-Con/Pro Controller driver (Controls, "
                                  "Advanced) owns the controller instead of SDL and is not driven "
                                  "by this engine. The Pro Controller driver is off by default; "
                                  "the Joy-Con driver is on, so turning it off hands Joy-Cons to "
                                  "this path as well. The built-in driver is still what provides "
                                  "the IR camera, amiibo and the Ring-Con.")));

    wired_group = new QGroupBox(tr("Wired (USB)"));
    auto* wired = new QFormLayout(wired_group);
    usb_interval = new QSpinBox;
    usb_interval->setRange(5, 100);
    usb_interval->setSuffix(tr(" ms"));
    usb_interval->setToolTip(tr("Time between rumble packets. 15 ms is the console's rate."));
    envelope_5ms = new QCheckBox(tr("5 ms envelope (three samples per packet)"));
    frequency_in_packet = new QCheckBox(tr("Frequency changes inside a packet"));
    crisp_notes = new QCheckBox(tr("Crisp notes (stronger attack, softer sustain)"));
    wired->addRow(tr("Packet interval"), usb_interval);
    wired->addRow(envelope_5ms);
    wired->addRow(frequency_in_packet);
    wired->addRow(crisp_notes);
    layout->addWidget(wired_group);

    bt_group = new QGroupBox(tr("Bluetooth"));
    auto* bt = new QFormLayout(bt_group);
    bt_level = new QComboBox;
    bt_level->addItem(tr("1 - Stable (60 ms)"), 1);
    bt_level->addItem(tr("2 - Balanced (45 ms)"), 2);
    bt_level->addItem(tr("3 - Quality (35 ms)"), 3);
    bt_level->setToolTip(tr("Lower levels send less often, which leaves more of a busy "
                            "Bluetooth link to the input reports."));
    bt_after_input = new QCheckBox(tr("Send right after an input report"));
    bt_fast = new QCheckBox(tr("5 ms envelope while the link is idle"));
    bt_crisp_notes = new QCheckBox(tr("Crisp notes"));
    bt->addRow(tr("Level"), bt_level);
    bt->addRow(bt_after_input);
    bt->addRow(bt_fast);
    bt->addRow(bt_crisp_notes);
    layout->addWidget(bt_group);

    layout->addStretch();
    return page;
}

QWidget* ConfigureHdRumble::CreateDualSensePage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(MakeNote(tr("RenHDrum's rumble on the DualSense and DualSense Edge voice "
                                  "coils, wired or on Bluetooth.")));
    layout->addWidget(MakeNote(tr("Veshiru also has its own DualSense haptics (\"Enable "
                                  "DualSense HD Haptics\" in the Vibration settings, wired only), "
                                  "which stay as they are while this is off. While this is on, "
                                  "RenHDrum plays the controller instead.")));

    auto* group = new QGroupBox(tr("Haptics"));
    auto* form = new QFormLayout(group);
    ds_enabled = new QCheckBox(tr("Use RenHDrum for the haptic voice coils"));
    ds_enabled->setToolTip(tr("Off (the default): the DualSense is played as before."));
    auto* volume_row = MakePercentRow(ds_volume_slider, ds_volume, 0, 100, 1, QString{});
    ds_crisp_notes = new QCheckBox(tr("Crisp notes"));
    form->addRow(ds_enabled);
    form->addRow(tr("Volume"), volume_row);
    form->addRow(ds_crisp_notes);
    layout->addWidget(group);

    layout->addStretch();
    return page;
}

QWidget* ConfigureHdRumble::CreateXboxPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(MakeNote(tr("Xbox controllers and Xbox-style controllers from other "
                                  "brands (PowerA, 8BitDo, GameSir...).")));
    layout->addWidget(MakeNote(
        tr("These pads have two spinning weights, a heavy one in the left grip and a light one "
           "in the right. A spinning weight has a single control: how hard it is driven sets its "
           "speed and its force together, and it needs tens of milliseconds to spin up or down. "
           "So the pitch of an HD Rumble effect cannot be played and very short ticks are "
           "blurred. What is kept is how strong the low and the high part of the effect are, "
           "and when.")));

    auto* group = new QGroupBox(tr("Motors"));
    auto* form = new QFormLayout(group);
    xbox_enabled = new QCheckBox(tr("Use HD rumble on Xbox-style controllers"));
    xbox_enabled->setToolTip(tr("Off: Xbox-style controllers use the standard rumble."));
    xbox_mode = new QComboBox;
    xbox_mode->addItem(tr("Direct (low part on the heavy motor, high part on the light one)"),
                       1);
    xbox_mode->addItem(tr("RenHDrum mix (effects shaped inside the library)"), 0);
    xbox_mode->setToolTip(
        tr("Direct: mapped by Veshiru itself; the low part of the rumble drives the heavy motor "
           "and the high part the light one.\n"
           "RenHDrum mix: the library shapes the effects; tuned by its author on an official "
           "Xbox One S controller, and quiet unless the motor strength is raised."));
    auto* strength_row = MakePercentRow(
        xbox_strength_slider, xbox_strength, 0, 1000, 10,
        tr("How hard the motors are driven; 100% plays the mapping as it is. The RenHDrum mix "
           "tops out near a tenth of the motor range, so it usually wants several hundred "
           "percent."));
    form->addRow(xbox_enabled);
    form->addRow(tr("Mode"), xbox_mode);
    form->addRow(tr("Motor strength"), strength_row);
    form->addRow(tr("Detail on heavy motor"),
                 MakePercentRow(xbox_detail_slider, xbox_detail, 0, 100, 5,
                                tr("Adds part of the fine vibration to the heavy motor, where it "
                                   "is felt more. Direct mode.")));
    form->addRow(tr("Hit accents"),
                 MakePercentRow(xbox_accents_slider, xbox_accents, 0, 100, 5,
                                tr("A 25 ms burst when a vibration rises suddenly, so the motor "
                                   "spins up in time for a quick tap. 0% turns it off. Direct "
                                   "mode.")));

    // Whether a connected pad has trigger motors, known when the dialog opens
    xbox_triggers_state = InputCommon::GetRenHDrumXboxTriggers();
    xbox_triggers_enabled = new QCheckBox(tr("Also drive the trigger motors"));
    xbox_triggers_enabled->setToolTip(
        tr("Off by default. Only controllers with motors in the triggers have them, and only "
           "some Windows input paths reach them."));
    form->addRow(xbox_triggers_enabled);
    form->addRow(tr("Trigger share"),
                 MakePercentRow(xbox_trigger_slider, xbox_trigger, 0, 100, 5,
                                tr("Share of the fine vibration that is played on the triggers, "
                                   "the left side on LT and the right side on RT. Direct "
                                   "mode.")));
    QString trigger_state;
    if (xbox_triggers_state == 1) {
        trigger_state = tr("A connected controller reports trigger motors.");
    } else if (xbox_triggers_state == 0) {
        trigger_state = tr("The connected controller reports no trigger motors on this input "
                           "path, so only the grip motors are driven.");
    } else {
        trigger_state = tr("No Xbox-style controller is connected through SDL right now (or "
                           "renhdrum.dll is missing).");
    }
    form->addRow(MakeNote(trigger_state));
    layout->addWidget(group);

    layout->addWidget(MakeNote(
        tr("Windows reaches trigger motors only when SDL opens the controller through raw input "
           "(Controls > Advanced > \"Enable XInput 8 player support\", which disables the web "
           "applet). Through plain XInput, the default, a controller has its two grip motors "
           "only. The log names the path and the motors of each controller when it connects.")));
    layout->addStretch();
    return page;
}

void ConfigureHdRumble::LoadConfiguration() {
    const auto& s = Settings::values;
    hd_enabled->setChecked(s.hd_rumble_enabled.GetValue());
    gain->setValue(std::clamp(s.hd_rumble_gain.GetValue(), 25, 1000));
    boost->setValue(std::clamp(s.hd_rumble_low_boost.GetValue(), 0, 100));
    usb_interval->setValue(std::clamp(s.hd_rumble_usb_interval_ms.GetValue(), 5, 100));
    envelope_5ms->setChecked(s.hd_rumble_envelope_5ms.GetValue());
    frequency_in_packet->setChecked(s.hd_rumble_frequency_in_packet.GetValue());
    crisp_notes->setChecked(s.hd_rumble_crisp_notes.GetValue());
    bt_level->setCurrentIndex(std::clamp(s.hd_rumble_bluetooth_level.GetValue(), 1, 3) - 1);
    bt_after_input->setChecked(s.hd_rumble_bt_after_input.GetValue());
    bt_fast->setChecked(s.hd_rumble_bt_fast.GetValue());
    bt_crisp_notes->setChecked(s.hd_rumble_bt_crisp_notes.GetValue());
    ds_enabled->setChecked(s.renhdrum_dualsense_enabled.GetValue());
    ds_volume->setValue(std::clamp(s.renhdrum_dualsense_volume.GetValue(), 0, 100));
    ds_crisp_notes->setChecked(s.renhdrum_dualsense_crisp_notes.GetValue());
    xbox_enabled->setChecked(s.xbox_rumble_enabled.GetValue());
    xbox_strength->setValue(std::clamp(s.xbox_motor_strength.GetValue(), 0, 1000));
    xbox_mode->setCurrentIndex(
        (std::max)(0, xbox_mode->findData(s.xbox_rumble_mode.GetValue() == 0 ? 0 : 1)));
    xbox_detail->setValue(std::clamp(s.xbox_detail_mix.GetValue(), 0, 100));
    xbox_accents->setValue(std::clamp(s.xbox_hit_accents.GetValue(), 0, 100));
    xbox_triggers_enabled->setChecked(s.xbox_trigger_rumble.GetValue());
    xbox_trigger->setValue(std::clamp(s.xbox_trigger_mix.GetValue(), 0, 100));
    UpdateEnabledState();
}

void ConfigureHdRumble::RestoreDefaults() {
    // Everything but the master switch: restoring the tuning should not switch the feature the
    // user is tuning off
    const auto& s = Settings::values;
    gain->setValue(s.hd_rumble_gain.GetDefault());
    boost->setValue(s.hd_rumble_low_boost.GetDefault());
    usb_interval->setValue(s.hd_rumble_usb_interval_ms.GetDefault());
    envelope_5ms->setChecked(s.hd_rumble_envelope_5ms.GetDefault());
    frequency_in_packet->setChecked(s.hd_rumble_frequency_in_packet.GetDefault());
    crisp_notes->setChecked(s.hd_rumble_crisp_notes.GetDefault());
    bt_level->setCurrentIndex(std::clamp(s.hd_rumble_bluetooth_level.GetDefault(), 1, 3) - 1);
    bt_after_input->setChecked(s.hd_rumble_bt_after_input.GetDefault());
    bt_fast->setChecked(s.hd_rumble_bt_fast.GetDefault());
    bt_crisp_notes->setChecked(s.hd_rumble_bt_crisp_notes.GetDefault());
    ds_enabled->setChecked(s.renhdrum_dualsense_enabled.GetDefault());
    ds_volume->setValue(s.renhdrum_dualsense_volume.GetDefault());
    ds_crisp_notes->setChecked(s.renhdrum_dualsense_crisp_notes.GetDefault());
    xbox_enabled->setChecked(s.xbox_rumble_enabled.GetDefault());
    xbox_strength->setValue(s.xbox_motor_strength.GetDefault());
    xbox_mode->setCurrentIndex((std::max)(0, xbox_mode->findData(s.xbox_rumble_mode.GetDefault())));
    xbox_detail->setValue(s.xbox_detail_mix.GetDefault());
    xbox_accents->setValue(s.xbox_hit_accents.GetDefault());
    xbox_triggers_enabled->setChecked(s.xbox_trigger_rumble.GetDefault());
    xbox_trigger->setValue(s.xbox_trigger_mix.GetDefault());
    UpdateEnabledState();
}

void ConfigureHdRumble::UpdateEnabledState() {
    const bool on = hd_enabled->isChecked();
    pages->setEnabled(on);
    // The extra wired cards only run on top of the 5 ms envelope
    frequency_in_packet->setEnabled(envelope_5ms->isChecked());
    crisp_notes->setEnabled(envelope_5ms->isChecked());
    ds_volume_slider->setEnabled(ds_enabled->isChecked());
    ds_volume->setEnabled(ds_enabled->isChecked());
    ds_crisp_notes->setEnabled(ds_enabled->isChecked());
    const bool xbox_on = xbox_enabled->isChecked();
    xbox_strength_slider->setEnabled(xbox_on);
    xbox_strength->setEnabled(xbox_on);
    xbox_mode->setEnabled(xbox_on);
    xbox_triggers_enabled->setEnabled(xbox_on);
    const bool direct = xbox_on && xbox_mode->currentData().toInt() == 1;
    xbox_detail_slider->setEnabled(direct);
    xbox_detail->setEnabled(direct);
    xbox_accents_slider->setEnabled(direct);
    xbox_accents->setEnabled(direct);
    const bool trigger_share = direct && xbox_triggers_enabled->isChecked();
    xbox_trigger_slider->setEnabled(trigger_share);
    xbox_trigger->setEnabled(trigger_share);
}

void ConfigureHdRumble::ApplyConfiguration() {
    auto& s = Settings::values;
    s.hd_rumble_enabled.SetValue(hd_enabled->isChecked());
    s.hd_rumble_gain.SetValue(gain->value());
    s.hd_rumble_low_boost.SetValue(boost->value());
    s.hd_rumble_usb_interval_ms.SetValue(usb_interval->value());
    s.hd_rumble_envelope_5ms.SetValue(envelope_5ms->isChecked());
    s.hd_rumble_frequency_in_packet.SetValue(frequency_in_packet->isChecked());
    s.hd_rumble_crisp_notes.SetValue(crisp_notes->isChecked());
    s.hd_rumble_bluetooth_level.SetValue(bt_level->currentData().toInt());
    s.hd_rumble_bt_after_input.SetValue(bt_after_input->isChecked());
    s.hd_rumble_bt_fast.SetValue(bt_fast->isChecked());
    s.hd_rumble_bt_crisp_notes.SetValue(bt_crisp_notes->isChecked());
    // The master switch also turns RenHDrum's DualSense haptics off (the pad checks both)
    s.renhdrum_dualsense_enabled.SetValue(ds_enabled->isChecked());
    s.renhdrum_dualsense_volume.SetValue(ds_volume->value());
    s.renhdrum_dualsense_crisp_notes.SetValue(ds_crisp_notes->isChecked());
    s.xbox_rumble_enabled.SetValue(xbox_enabled->isChecked());
    s.xbox_motor_strength.SetValue(xbox_strength->value());
    s.xbox_rumble_mode.SetValue(xbox_mode->currentData().toInt());
    s.xbox_detail_mix.SetValue(xbox_detail->value());
    s.xbox_hit_accents.SetValue(xbox_accents->value());
    s.xbox_trigger_rumble.SetValue(xbox_triggers_enabled->isChecked());
    s.xbox_trigger_mix.SetValue(xbox_trigger->value());
}
