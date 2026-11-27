// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <set>
#include <string>
#include <utility>
#include <vector>
#include <QCoreApplication>
#include <QWidget>
#include "common/recommended_settings.h"

class QCheckBox;
class QGridLayout;

namespace Settings {
struct GamePreset;
}

/// Lists the recommended settings for one game beside their current values.
class RecommendedSettingsList : public QWidget {
    Q_OBJECT

public:
    explicit RecommendedSettingsList(const Settings::GamePreset& preset, QWidget* parent = nullptr);
    ~RecommendedSettingsList() override;

    void Refresh();
    std::set<std::string> SelectedKeys() const;
    bool HasChanges() const;
    void SetAllChecked(bool checked);

signals:
    void SelectionChanged();

private:
    const Settings::GamePreset& preset;
    QGridLayout* grid = nullptr;
    std::vector<std::pair<std::string, QCheckBox*>> rows;
};

/// The message boxes around the developer's recommended settings (common/recommended_settings.h).
class RecommendedSettingsDialog {
    Q_DECLARE_TR_FUNCTIONS(RecommendedSettingsDialog)

public:
    /// Asks whether to use the recommended settings. True when the user accepts.
    static bool Ask(QWidget* parent);

    /// Says what ApplyRecommendedSettings changed and where to undo it.
    static void ShowApplied(QWidget* parent,
                            const std::vector<Settings::RecommendedSettingChange>& changes);

    /// Says what RestorePreviousSettings changed back.
    static void ShowRestored(QWidget* parent,
                             const std::vector<Settings::RecommendedSettingChange>& changes);
};
