// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <vector>

#include <QMainWindow>

#include "common/logging.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QTimer;

/// Independent, non-modal live log window. It owns no emulation state: closing it only stops the
/// live-console capture backend, never the emulator. The backend is asynchronous and bounded.
class ConsoleWindow final : public QMainWindow {
    Q_OBJECT

public:
    static void Show();
    explicit ConsoleWindow(QWidget* parent = nullptr);
    ~ConsoleWindow() override;

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void Poll();
    void RebuildView();
    void Clear();
    void Copy() const;
    void Save();
    void ExportDiagnostics();
    void UpdateStatus();
    [[nodiscard]] bool PassesFilter(const Common::Log::ConsoleEntry& entry) const;
    void AppendEntries(const std::vector<Common::Log::ConsoleEntry>& entries);
    [[nodiscard]] QString FormatPlain(const Common::Log::ConsoleEntry& entry) const;

    QTimer* poll_timer{};
    QLineEdit* search{};
    QComboBox* level_combo{};
    QPlainTextEdit* view{};
    QCheckBox* auto_scroll{};
    QCheckBox* pause_display{};
    QLabel* status{};
    std::array<QCheckBox*, static_cast<std::size_t>(Common::Log::Subsystem::Count)> subsystem_checks{};
    u64 last_sequence{};
};
