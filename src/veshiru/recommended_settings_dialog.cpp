// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <numeric>
#include <string>
#include <QCheckBox>
#include <QFrame>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QStringList>
#include "common/game_presets.h"
#include "common/settings.h"
#include "qt_common/config/shared_translation.h"
#include "veshiru/recommended_settings_dialog.h"

namespace {

struct RecommendedSettingText {
    QString label;
    QString current;
    QString recommended;
    bool already_set;
};

std::vector<RecommendedSettingText> DescribeRecommendedSettings(const Settings::GamePreset& preset) {
    const auto labels = ConfigurationShared::InitializeTranslations(nullptr);
    const auto enums = ConfigurationShared::ComboboxEnumeration(nullptr);
    enums->emplace(Settings::EnumMetadata<Settings::VSyncMode>::Index(),
                   ConfigurationShared::ComboboxTranslations{
                       {static_cast<u32>(Settings::VSyncMode::Immediate),
                        QCoreApplication::translate("RecommendedSettingsDialog", "Immediate")},
                       {static_cast<u32>(Settings::VSyncMode::Mailbox), QStringLiteral("Mailbox")},
                       {static_cast<u32>(Settings::VSyncMode::Fifo), QStringLiteral("FIFO")},
                       {static_cast<u32>(Settings::VSyncMode::FifoRelaxed),
                        QStringLiteral("FIFO Relaxed")}});

    std::vector<RecommendedSettingText> result;
    result.reserve(preset.settings.size());
    for (const auto& recommendation : preset.settings) {
        const auto* setting = Settings::values.linkage.by_key.at(recommendation.key);
        QString label = QString::fromStdString(recommendation.key);
        if (const auto it = labels->find(setting->Id());
            it != labels->end() && !it->second.first.isEmpty()) {
            label = it->second.first;
        }
        if (label.endsWith(QLatin1Char(':'))) {
            label.chop(1);
        }

        const auto format = [&](const std::string& encoded) {
            if (setting->IsEnum()) {
                if (const auto it = enums->find(setting->EnumIndex()); it != enums->end()) {
                    const auto enum_value = static_cast<u32>(std::stoul(encoded));
                    for (const auto& [id, text] : it->second) {
                        if (id == enum_value) {
                            return text;
                        }
                    }
                }
            } else if (setting->TypeId() == "bool") {
                return encoded == "true"
                           ? QCoreApplication::translate("RecommendedSettingsDialog", "On")
                           : QCoreApplication::translate("RecommendedSettingsDialog", "Off");
            }
            return QString::fromStdString(encoded);
        };

        const std::string current = setting->ToString();
        result.push_back({label, format(current), format(recommendation.value),
                          current == recommendation.value});
    }
    return result;
}

// One "name: previous value → new value" line per change, worded as the settings dialog words
// them.
QStringList Describe(QWidget* parent,
                     const std::vector<Settings::RecommendedSettingChange>& changes,
                     const QString& on, const QString& off) {
    const auto titles = ConfigurationShared::InitializeTranslations(parent);
    const auto enums = ConfigurationShared::ComboboxEnumeration(parent);

    const auto title_of = [&](const Settings::BasicSetting& setting) {
        const auto it = titles->find(setting.Id());
        return it == titles->end() ? QString{} : it->second.first;
    };
    const auto text_of = [&](const Settings::BasicSetting& setting, const std::string& value) {
        if (setting.TypeId() == "bool") {
            return value == "true" ? on : off;
        }
        if (const auto it = enums->find(setting.EnumIndex()); it != enums->end()) {
            for (const auto& [id, text] : it->second) {
                if (std::to_string(id) == value) {
                    return text;
                }
            }
        }
        QString text = QString::fromStdString(value);
        if (setting.Specialization() & Settings::Specialization::Percentage) {
            text.append(QLatin1Char{'%'});
        }
        return text;
    };

    QStringList lines;
    for (const auto& change : changes) {
        QString title = title_of(*change.setting);
        if (title.isEmpty()) {
            // A checkbox paired with a value is named by the value's setting.
            for (const auto& [key, other] : Settings::values.linkage.by_key) {
                if (other->PairedSetting() == change.setting) {
                    title = title_of(*other);
                }
            }
        }
        if (title.isEmpty()) {
            title = QString::fromStdString(change.setting->GetLabel());
        }
        if (title.endsWith(QLatin1Char{':'})) {
            title.chop(1);
        }
        if (!change.game.empty()) {
            title = RecommendedSettingsDialog::tr("%1 (game %2)")
                        .arg(title, QString::fromStdString(change.game));
        }
        lines.append(QStringLiteral("%1: %2 \u2192 %3")
                         .arg(title, text_of(*change.setting, change.previous),
                              text_of(*change.setting, change.current)));
    }
    return lines;
}

void ShowChanges(QWidget* parent, const QString& title, const QString& text,
                 const QStringList& lines, const QString& footer) {
    QMessageBox box(QMessageBox::Information, title, text, QMessageBox::Ok, parent);
    // A long list goes behind the details button so the box stays on the screen.
    if (lines.size() <= 12) {
        // Rich text, so that a line is not wrapped in the middle of a value.
        QString list;
        for (const QString& line : lines) {
            list += QStringLiteral("<nobr>%1</nobr><br>").arg(line.toHtmlEscaped());
        }
        box.setInformativeText(list + QStringLiteral("<br>") + footer.toHtmlEscaped());
    } else {
        box.setInformativeText(footer);
        box.setDetailedText(lines.join(QLatin1Char{'\n'}));
    }
    box.exec();
}

} // namespace

RecommendedSettingsList::RecommendedSettingsList(const Settings::GamePreset& preset_, QWidget* parent)
    : QWidget(parent), preset{preset_} {
    grid = new QGridLayout(this);
    grid->setContentsMargins(12, 8, 12, 8);
    grid->setHorizontalSpacing(16);
    grid->setVerticalSpacing(6);
    grid->setColumnStretch(0, 1);
    Refresh();
}

RecommendedSettingsList::~RecommendedSettingsList() = default;

void RecommendedSettingsList::Refresh() {
    while (QLayoutItem* item = grid->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    rows.clear();

    const auto add_header = [&](int column, const QString& text) {
        auto* header = new QLabel(text, this);
        QFont font = header->font();
        font.setBold(true);
        header->setFont(font);
        grid->addWidget(header, 0, column);
    };
    add_header(0, tr("Setting"));
    add_header(1, tr("Current"));
    add_header(3, tr("Recommended"));
    auto* rule = new QFrame(this);
    rule->setFrameShape(QFrame::HLine);
    rule->setFrameShadow(QFrame::Sunken);
    grid->addWidget(rule, 1, 0, 1, 4);

    const auto descriptions = DescribeRecommendedSettings(preset);
    std::vector<size_t> order(descriptions.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_partition(order.begin(), order.end(), [&](size_t i) {
        return !descriptions[i].already_set;
    });

    int row = 2;
    for (const size_t i : order) {
        const auto& text = descriptions[i];
        auto* check = new QCheckBox(text.label, this);
        auto* current = new QLabel(text.current, this);
        auto* arrow = new QLabel(QStringLiteral("\u2192"), this);
        auto* recommended = new QLabel(text.recommended, this);
        if (text.already_set) {
            check->setEnabled(false);
            check->setToolTip(tr("Already using the recommended value."));
            current->setEnabled(false);
            arrow->setEnabled(false);
            recommended->setText(tr("Already set"));
            recommended->setEnabled(false);
        } else {
            check->setChecked(true);
            QFont font = recommended->font();
            font.setBold(true);
            recommended->setFont(font);
            connect(check, &QCheckBox::toggled, this, [this, arrow, recommended](bool on) {
                arrow->setEnabled(on);
                recommended->setEnabled(on);
                emit SelectionChanged();
            });
        }
        grid->addWidget(check, row, 0);
        grid->addWidget(current, row, 1);
        grid->addWidget(arrow, row, 2);
        grid->addWidget(recommended, row, 3);
        rows.emplace_back(preset.settings[i].key, check);
        ++row;
    }
    emit SelectionChanged();
}

std::set<std::string> RecommendedSettingsList::SelectedKeys() const {
    std::set<std::string> keys;
    for (const auto& [key, check] : rows) {
        if (check->isEnabled() && check->isChecked()) {
            keys.insert(key);
        }
    }
    return keys;
}

bool RecommendedSettingsList::HasChanges() const {
    return std::any_of(rows.begin(), rows.end(), [](const auto& row) {
        return row.second->isEnabled();
    });
}

void RecommendedSettingsList::SetAllChecked(bool checked) {
    for (const auto& [key, check] : rows) {
        if (check->isEnabled()) {
            check->setChecked(checked);
        }
    }
}

bool RecommendedSettingsDialog::Ask(QWidget* parent) {
    QMessageBox box(parent);
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(tr("Recommended settings"));
    box.setText(tr("Do you want to use the recommended settings from the Veshiru developer?"));
    box.setInformativeText(
        tr("They improve performance for most games. Games with their own value for one of these "
           "settings in their per-game settings use the recommended value too; other per-game "
           "settings do not change. Veshiru keeps your own settings, global and per-game, so you "
           "can go back at any time with Help > Restore My Previous Settings."));
    auto* const use = box.addButton(tr("Use recommended settings"), QMessageBox::AcceptRole);
    box.addButton(tr("Not now"), QMessageBox::RejectRole);
    box.setDefaultButton(use);
    box.exec();
    return box.clickedButton() == use;
}

void RecommendedSettingsDialog::ShowApplied(
    QWidget* parent, const std::vector<Settings::RecommendedSettingChange>& changes) {
    if (changes.empty()) {
        QMessageBox::information(
            parent, tr("Recommended settings"),
            tr("Your settings already match the recommended settings. Nothing changed."));
        return;
    }
    ShowChanges(parent, tr("Recommended settings"),
                tr("Veshiru now uses the recommended settings, in every game."),
                Describe(parent, changes, tr("On"), tr("Off")),
                tr("To go back, use Help > Restore My Previous Settings."));
}

void RecommendedSettingsDialog::ShowRestored(
    QWidget* parent, const std::vector<Settings::RecommendedSettingChange>& changes) {
    if (changes.empty()) {
        QMessageBox::information(
            parent, tr("Previous settings"),
            tr("Your settings already match your previous settings. Nothing changed."));
        return;
    }
    ShowChanges(parent, tr("Previous settings"), tr("Veshiru uses your previous settings again."),
                Describe(parent, changes, tr("On"), tr("Off")),
                tr("To use the recommended settings again, use Help > Apply Recommended "
                   "Settings."));
}
