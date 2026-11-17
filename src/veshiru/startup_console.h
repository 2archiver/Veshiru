// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include <QMainWindow>

#include "common/logging.h"

class QPlainTextEdit;
class QTimer;

/// Ryujinx-style startup console: a pure-black terminal window that opens automatically when
/// Veshiru launches. It shows a large custom Veshiru ASCII logo, a compact system summary, a small
/// version/build identifier and timestamped startup logs streamed from the Common::Log live-console
/// backend. INFO/SUCCESS lines are white, WARN lines yellow and ERROR lines red, on a traditional
/// monospaced terminal surface.
///
/// The console stays open for the whole session: closing it only hides it while log capture keeps
/// running, and it can be reopened through the View > Console menu or StartupConsole::Show().
class StartupConsole final : public QMainWindow {
    Q_OBJECT

public:
    /// Shows the existing console window (creates it on first use) and brings it to front.
    static void Show();

    explicit StartupConsole(QWidget* parent = nullptr);
    ~StartupConsole() override;

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void Poll();
    void RebuildView();
    void AppendEntries(const std::vector<Common::Log::ConsoleEntry>& entries);
    void PrintHeader();

    [[nodiscard]] QString FormatLine(const Common::Log::ConsoleEntry& entry) const;

    QTimer* poll_timer{};
    QPlainTextEdit* view{};
    QString header_html;
    u64 last_sequence{};
};
