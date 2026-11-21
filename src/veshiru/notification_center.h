// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QIcon>
#include <QObject>
#include <QString>

class QMenu;
class QSystemTrayIcon;
class QWidget;

namespace Veshiru {

/// The Veshiru application icon, loaded from the Qt resource compiled into the executable.
/// Never a relative file path, so launching from another working directory still finds it.
QIcon AppIcon();

/// The notification-area icon and the toasts it shows. Every message carries AppIcon(): the tray
/// icon and the per-message icon are the embedded Veshiru icon, never NoIcon or a null pixmap.
class NotificationCenter final : public QObject {
    Q_OBJECT

public:
    static NotificationCenter& Instance();

    /// Gives the tray menu a window to show/quits. Safe to call once, after the main window is
    /// constructed; notifications raised earlier still work (the tray is created lazily).
    void Initialize(QWidget* main_window);

    /// Shows one toast if the desktop has a notification area. A desktop without one, a full
    /// queue or a headless session simply does nothing.
    void Notify(const QString& title, const QString& message);

private:
    explicit NotificationCenter(QObject* parent = nullptr);

    void EnsureTray();
    void PopulateMenu();

    QSystemTrayIcon* tray{};
    QMenu* menu{};
    QWidget* window{};
};

} // namespace Veshiru
