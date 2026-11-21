// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "veshiru/notification_center.h"

#include <QAction>
#include <QApplication>
#include <QMenu>
#include <QSystemTrayIcon>
#include <QWidget>

namespace Veshiru {

QIcon AppIcon() {
    // The .ico is embedded by src/veshiru/veshiru.qrc. QIcon also picks the size Windows asks
    // for, so the tray, the taskbar and a toast all get the same artwork.
    return QIcon(QStringLiteral(":/img/veshiru.ico"));
}

NotificationCenter& NotificationCenter::Instance() {
    static NotificationCenter center;
    return center;
}

NotificationCenter::NotificationCenter(QObject* parent) : QObject{parent} {}

void NotificationCenter::Initialize(QWidget* main_window) {
    this->window = main_window;
    if (main_window != nullptr) {
        // The tray outlives the window; forget the menu before Qt deletes it.
        connect(main_window, &QObject::destroyed, this, [this] {
            this->window = nullptr;
            this->menu = nullptr;
            if (tray != nullptr) {
                tray->setContextMenu(nullptr);
            }
        });
    }
    EnsureTray();
    // A toast raised before the main window existed created a tray with no window actions.
    if (tray != nullptr && menu != nullptr && menu->actions().isEmpty()) {
        PopulateMenu();
    }
}

void NotificationCenter::EnsureTray() {
    if (tray != nullptr) {
        return;
    }
    if (QCoreApplication::instance() == nullptr || !QSystemTrayIcon::isSystemTrayAvailable()) {
        // Remote desktops and some shells have no notification area; nothing to create.
        return;
    }

    // Parent the tray to the application, not to this static object: it must be destroyed with
    // QApplication, before the process tears down Qt's platform plugin.
    tray = new QSystemTrayIcon(AppIcon(), qApp);
    tray->setToolTip(QStringLiteral("Veshiru"));

    menu = new QMenu(window);
    PopulateMenu();
    tray->setContextMenu(menu);
    tray->show();
}

void NotificationCenter::PopulateMenu() {
    if (menu == nullptr || window == nullptr) {
        return;
    }
    QAction* show_action = menu->addAction(QStringLiteral("Show Veshiru"));
    connect(show_action, &QAction::triggered, this, [this] {
        if (window == nullptr) {
            return;
        }
        window->show();
        window->setWindowState(window->windowState() & ~Qt::WindowMinimized);
        window->raise();
        window->activateWindow();
    });

    QAction* quit_action = menu->addAction(QStringLiteral("Quit"));
    // Closing the window runs the normal closeEvent path (including an emulation running
    // confirmation), unlike qApp->quit().
    connect(quit_action, &QAction::triggered, this, [this] {
        if (window != nullptr) {
            window->close();
        }
    });
}

void NotificationCenter::Notify(const QString& title, const QString& message) {
    EnsureTray();
    if (tray == nullptr || !tray->isVisible() || !QSystemTrayIcon::supportsMessages()) {
        return;
    }
    tray->showMessage(title, message, AppIcon(), 6000);
}

} // namespace Veshiru
