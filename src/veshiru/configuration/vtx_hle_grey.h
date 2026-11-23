// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>
#include <QObject>
#include <QString>

#include "common/settings_enums.h"
#include "common/vtx_hle_settings.h"

namespace ConfigurationShared {

class Widget;

/// One per settings dialog (the Builder owns it): greys out the widgets of the settings VTX-HLE
/// does not apply (common/vtx_hle_settings.h) while the dialog's games run VTX-HLE. Only the
/// widgets are touched, never a setting, so the saved values and what a custom configuration
/// writes stay as they are, and the Emulated GPU path sees every setting as before.
class VtxHleGrey final : public QObject {
    Q_OBJECT

public:
    /// `game_running`: a game is running (the Builder's runtime lock is off).
    explicit VtxHleGrey(bool game_running, QObject* parent);
    ~VtxHleGrey() override;

    /// Starts greying `widget` as `entry` says, with the state known now.
    void Track(Widget* widget, const Settings::VtxHle::Entry& entry);

    /// The Graphics path and API the dialog shows (their dropdowns' current choices, applied or
    /// not). Re-evaluates.
    void SetPathSelection(Settings::GraphicsPath path, bool vulkan);
    /// Whether the Properties dialog's game is a Switch 2 program. The global dialog has no game
    /// and keeps the default: true.
    void SetSwitch2Program(bool switch2_program);
    /// Whether the dialog's game is the running one: then its actual path decides, whatever the
    /// dropdown shows (the path only changes at the next game start). The global dialog is about
    /// the running game, if any; Properties of another game is not.
    void SetTargetIsRunningGame(bool is_running_game);

    /// Re-reads the running game's path (it settles a moment after the boot) and the environment.
    void Refresh();

    [[nodiscard]] bool Active() const {
        return active;
    }

    /// The tooltip text of a reason (translated).
    [[nodiscard]] static QString ReasonText(Settings::VtxHle::Reason reason);
    /// The short text appended to the setting's label (translated).
    [[nodiscard]] static QString LabelSuffix(Settings::VtxHle::Reason reason);

signals:
    void ActiveChanged(bool active);

private:
    struct Tracked;
    void Apply(Tracked& tracked) const;

    std::vector<Tracked> widgets;
    Settings::GraphicsPath selected;
    bool vulkan;
    bool switch2_program{true};
    bool target_is_running_game;
    bool active{};
};

} // namespace ConfigurationShared
