// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "startup_console.h"

#include <string>
#include <thread>

#include <QCloseEvent>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QPlainTextEdit>
#include <QPointer>
#include <QScreen>
#include <QSysInfo>
#include <QStringList>
#include <QTextCursor>
#include <QTimer>

#include "common/cpu_features.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"
#include "qt_common/util/vk.h"

namespace {

QPointer<StartupConsole> instance;

// Large custom Veshiru ASCII logo shown at the top of the startup console.
constexpr std::string_view kLogo =
    " ███   ███    ███ █   █  ███  █████ █   █\n"
    "█   █ █      █    █   █   █   █   █ █   █\n"
    "█   █  ███    █    ███    █   █████ █   █\n"
    "█   █ █        █  █   █   █   █   █ █   █\n"
    " █ █   ███  ████  █   █  ███  █  ██  █ █ \n";

constexpr std::string_view kSeparator =
    "----------------------------------------"
    "----------------------------------------";

QString LevelColor(Common::Log::Level level) {
    using Common::Log::Level;
    // Traditional terminal palette: white for INFO/SUCCESS, yellow for WARN, red for ERROR.
    // Trace/Debug stay dim so ordinary output recedes behind the boot sequence.
    switch (level) {
    case Level::Trace:
        return QStringLiteral("#6a6a6a");
    case Level::Debug:
        return QStringLiteral("#9a9a9a");
    case Level::Info:
        return QStringLiteral("#ffffff");
    case Level::Warning:
        return QStringLiteral("#ffd34d");
    case Level::Error:
    case Level::Critical:
        return QStringLiteral("#ff5555");
    default:
        return QStringLiteral("#ffffff");
    }
}

QString FromView(std::string_view view) {
    return QString::fromUtf8(view.data(), static_cast<qsizetype>(view.size()));
}

} // namespace

void StartupConsole::Show() {
    if (!instance) {
        instance = new StartupConsole();
    }
    instance->show();
    instance->raise();
    instance->activateWindow();
}

StartupConsole::StartupConsole(QWidget* parent) : QMainWindow(parent) {
    setObjectName(QStringLiteral("VeshiruStartupConsole"));
    setWindowTitle(tr("Veshiru Console"));
    // Plain terminal surface: pure black, no borders, gradients or decorative styling.
    setStyleSheet(QStringLiteral(
        "QMainWindow { background-color: #000000; }\n"
        "QPlainTextEdit { background-color: #000000; color: #ffffff;"
        " border: none; padding: 8px; }\n"
        "QPlainTextEdit::selection { background-color: #333333; color: #ffffff; }"));

    resize(1000, 640);
    const auto screen = QGuiApplication::primaryScreen();
    if (screen != nullptr) {
        const auto geom = screen->availableGeometry();
        move(geom.left() + (geom.width() - width()) / 2, geom.top() + 40);
    }

    view = new QPlainTextEdit(this);
    view->setReadOnly(true);
    view->setLineWrapMode(QPlainTextEdit::NoWrap);
    view->setMaximumBlockCount(20000);
    auto console_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    console_font.setStyleHint(QFont::Monospace);
    console_font.setPointSize(11);
    view->setFont(console_font);
    setCentralWidget(view);

    PrintHeader();

    poll_timer = new QTimer(this);
    connect(poll_timer, &QTimer::timeout, this, &StartupConsole::Poll);
    poll_timer->start(100);

    RebuildView();
}

StartupConsole::~StartupConsole() = default;

void StartupConsole::closeEvent(QCloseEvent* event) {
    // Keep the console available for the whole session: hiding it leaves log capture running and
    // the window can be reopened from the menu or StartupConsole::Show().
    hide();
    event->accept();
}

void StartupConsole::PrintHeader() {
    // Logo + small version/build identifier + compact system summary.
    const QString version_line =
        QStringLiteral("Veshiru v%1 | build %2 | %3 | %4")
            .arg(QString::fromUtf8(Common::g_build_version),
                 QString::fromUtf8(Common::g_build_id),
                 QString::fromUtf8(Common::g_scm_rev),
                 QString::fromUtf8(Common::g_build_date));

    QStringList system_lines;
    const auto& mem = Common::GetMemInfo();
    const auto processor_count = std::thread::hardware_concurrency();
    std::string cpu_string;
    int cores = 0;
#ifdef ARCHITECTURE_x86_64
    cpu_string = Common::g_cpu_caps.cpu_string;
    if (Common::g_cpu_caps.avx || Common::g_cpu_caps.avx2 || Common::g_cpu_caps.avx512f) {
        cpu_string += " | AVX";
        if (Common::g_cpu_caps.avx512f) {
            cpu_string += "512";
        } else if (Common::g_cpu_caps.avx2) {
            cpu_string += '2';
        }
        if (Common::g_cpu_caps.fma) {
            cpu_string += " | FMA";
        }
    }
    if (const auto core_count = Common::GetProcessorCount(); core_count) {
        cores = *core_count;
    }
#else
    cpu_string = QSysInfo::currentCpuArchitecture().toStdString();
#endif
    system_lines << QStringLiteral("[SYS] CPU: %1").arg(QString::fromStdString(cpu_string));
    system_lines << QStringLiteral("[SYS] Cores: %1 | Threads: %2")
                        .arg(cores)
                        .arg(processor_count);
    system_lines << QStringLiteral("[SYS] OS: %1").arg(QSysInfo::prettyProductName());
    system_lines << QStringLiteral("[SYS] RAM: %1 GiB | Swap: %2 GiB")
                        .arg(mem.TotalPhysicalMemory / (1024.0 * 1024.0))
                        .arg(mem.TotalSwapMemory / (1024.0 * 1024.0));
    std::vector<VkDeviceInfo::Record> vk_records;
    VkDeviceInfo::PopulateRecords(vk_records, nullptr);
    for (std::size_t i = 0; i < vk_records.size(); ++i) {
        system_lines << QStringLiteral("[SYS] GPU [%1]: %2 (%3 MiB VRAM)")
                            .arg(i)
                            .arg(QString::fromStdString(vk_records[i].name))
                            .arg(vk_records[i].device_local_memory / (1024 * 1024));
    }
    if (vk_records.empty()) {
        system_lines << QStringLiteral("[SYS] GPU: no Vulkan device found");
    }

    QString html;
    html += QStringLiteral(
                "<span style=\"color:#ffffff; white-space:pre;\">%1</span>")
                .arg(QString::fromUtf8(kLogo.data(), static_cast<qsizetype>(kLogo.size()))
                         .toHtmlEscaped());
    html += QStringLiteral(
                "<span style=\"color:#d8d8d8; white-space:pre;\">%1</span>")
                .arg(version_line.toHtmlEscaped());
    html += QStringLiteral("<span style=\"color:#ffffff; white-space:pre;\">%1</span>")
                .arg(QString::fromUtf8(kSeparator.data(),
                                       static_cast<qsizetype>(kSeparator.size())));
    for (const auto& line : system_lines) {
        html += QStringLiteral(
                    "<span style=\"color:#ffffff; white-space:pre;\">%1</span>")
                    .arg(line.toHtmlEscaped());
    }
    html += QStringLiteral("<span style=\"color:#ffffff; white-space:pre;\">%1</span>")
                .arg(QString::fromUtf8(kSeparator.data(),
                                       static_cast<qsizetype>(kSeparator.size())));
    html += QStringLiteral(
        "<span style=\"color:#d8d8d8;\">Startup logs (INFO: white | WARN: yellow | "
        "ERROR: red)</span>");
    header_html = std::move(html);
}

void StartupConsole::Poll() {
    auto entries = Common::Log::ReadLiveConsole(last_sequence, 2000);
    if (!entries.empty()) {
        last_sequence = entries.back().sequence;
        AppendEntries(entries);
    }
}

void StartupConsole::RebuildView() {
    view->clear();
    view->appendHtml(header_html);
    auto entries = Common::Log::ReadLiveConsole(0, 20000);
    last_sequence = entries.empty() ? 0 : entries.back().sequence;
    AppendEntries(entries);
    view->moveCursor(QTextCursor::End);
}

void StartupConsole::AppendEntries(const std::vector<Common::Log::ConsoleEntry>& entries) {
    if (entries.empty()) {
        return;
    }
    view->setUpdatesEnabled(false);
    for (const auto& entry : entries) {
        view->appendHtml(QStringLiteral(
            "<span style=\"color:%1; white-space:pre;\">%2</span>")
            .arg(LevelColor(entry.log_level), FormatLine(entry).toHtmlEscaped()));
    }
    view->setUpdatesEnabled(true);
    view->moveCursor(QTextCursor::End);
}

QString StartupConsole::FormatLine(const Common::Log::ConsoleEntry& entry) const {
    const auto seconds = entry.timestamp_us / 1'000'000.0;
    return QStringLiteral("[%1s] [%2] %3 | %4")
        .arg(seconds, 0, 'f', 3)
        .arg(FromView(Common::Log::GetLogLevelName(entry.log_level)),
             FromView(Common::Log::GetLogClassName(entry.log_class)),
             QString::fromStdString(entry.message));
}
