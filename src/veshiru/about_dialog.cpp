// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <QIcon>
#include <fmt/ranges.h>
#include "common/scm_rev.h"
#include "ui_aboutdialog.h"
#include "veshiru/about_dialog.h"

AboutDialog::AboutDialog(QWidget* parent)
    : QDialog(parent), ui{std::make_unique<Ui::AboutDialog>()} {
    static const std::string build_id = std::string{Common::g_build_id};
    static const std::string veshiru_build =
        fmt::format("{} | {} | {}", std::string{Common::g_build_name},
                    std::string{Common::g_build_version}, std::string{Common::g_compiler_id});

    const auto override_build =
        fmt::format(fmt::runtime(std::string(Common::g_title_bar_format_idle)), build_id);
    const auto veshiru_build_version = override_build.empty() ? veshiru_build : override_build;

    ui->setupUi(this);
    // Try and request the icon from Qt theme (Linux?)
    const QIcon veshiru_logo = QIcon::fromTheme(QStringLiteral("dev.veshiru.emulator"),
                                           QIcon(QStringLiteral(":/img/veshiru.ico")));
    if (!veshiru_logo.isNull()) {
        ui->labelLogo->setPixmap(veshiru_logo.pixmap(112));
    }
    ui->labelVeshiru->setPixmap(QIcon(QStringLiteral(":/img/veshiru_wordmark.png")).pixmap(113, 48));
    ui->labelVeshiru->setAccessibleName(QStringLiteral("Veshiru"));
    ui->verticalLayout_3->setContentsMargins(32, 28, 32, 20);
    ui->horizontalLayout->setSpacing(32);
    ui->verticalLayout->setSpacing(10);
    ui->labelBuildInfo->setText(
        ui->labelBuildInfo->text().arg(QString::fromStdString(veshiru_build_version),
                                       QString::fromUtf8(Common::g_build_date).left(10)));
}

AboutDialog::~AboutDialog() = default;
