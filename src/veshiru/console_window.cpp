// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "console_window.h"

#include <filesystem>
#include <string>

#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QScrollBar>
#include <QSysInfo>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>

#include <quazip.h>
#include <quazipfile.h>
#include <quazipnewinfo.h>

#include "common/cpu_features.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "qt_common/util/vk.h"

namespace {
QPointer<ConsoleWindow> instance;

QString LevelColor(Common::Log::Level level) {
    using Common::Log::Level;
    switch (level) {
    case Level::Trace:
        return QStringLiteral("#808080");
    case Level::Debug:
        return QStringLiteral("#00a0a0");
    case Level::Info:
        return QStringLiteral("#008000");
    case Level::Warning:
        return QStringLiteral("#b08000");
    case Level::Error:
        return QStringLiteral("#c00000");
    case Level::Critical:
        return QStringLiteral("#a000a0");
    default:
        return QStringLiteral("#000000");
    }
}

QString FromView(std::string_view view) {
    return QString::fromUtf8(view.data(), static_cast<qsizetype>(view.size()));
}

QString Timestamp(u64 timestamp_us) {
    return QStringLiteral("[%1.%2]")
        .arg(timestamp_us / 1'000'000)
        .arg(timestamp_us % 1'000'000, 6, 10, QLatin1Char('0'));
}
} // namespace

void ConsoleWindow::Show() {
    if (!instance) {
        instance = new ConsoleWindow();
    }
    instance->show();
    instance->raise();
    instance->activateWindow();
}

ConsoleWindow::ConsoleWindow(QWidget* parent) : QMainWindow(parent) {
    setObjectName(QStringLiteral("VeshiruConsole"));
    setWindowTitle(tr("Veshiru Console"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(1000, 620);

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    auto* controls = new QHBoxLayout;
    search = new QLineEdit(central);
    search->setPlaceholderText(tr("Search log text, subsystem or source…"));
    search->setClearButtonEnabled(true);
    controls->addWidget(search, 1);

    level_combo = new QComboBox(central);
    using Common::Log::Level;
    for (const auto level : {Level::Trace, Level::Debug, Level::Info, Level::Warning, Level::Error,
                             Level::Critical}) {
        level_combo->addItem(FromView(Common::Log::GetLogLevelName(level)),
                             static_cast<int>(level));
    }
    level_combo->setToolTip(tr("Minimum displayed severity. Capture verbosity is configured in "
                               "Emulation > Configure > Debug > Log Filter."));
    controls->addWidget(new QLabel(tr("Level:"), central));
    controls->addWidget(level_combo);

    pause_display = new QCheckBox(tr("Pause display"), central);
    pause_display->setToolTip(tr("Stop updating the view. Emulation and log capture continue."));
    controls->addWidget(pause_display);
    auto_scroll = new QCheckBox(tr("Auto-scroll"), central);
    auto_scroll->setChecked(true);
    controls->addWidget(auto_scroll);

    auto* clear_button = new QPushButton(tr("Clear"), central);
    auto* copy_button = new QPushButton(tr("Copy"), central);
    auto* save_button = new QPushButton(tr("Save .log…"), central);
    auto* diagnostics_button = new QPushButton(tr("Export diagnostics…"), central);
    controls->addWidget(clear_button);
    controls->addWidget(copy_button);
    controls->addWidget(save_button);
    controls->addWidget(diagnostics_button);
    layout->addLayout(controls);

    auto* subsystems = new QHBoxLayout;
    subsystems->addWidget(new QLabel(tr("Subsystems:"), central));
    for (std::size_t i = 0; i < subsystem_checks.size(); ++i) {
        const auto subsystem = static_cast<Common::Log::Subsystem>(i);
        auto* check = new QCheckBox(FromView(Common::Log::GetSubsystemName(subsystem)),
                                    central);
        check->setChecked(true);
        subsystem_checks[i] = check;
        subsystems->addWidget(check);
        connect(check, &QCheckBox::toggled, this, &ConsoleWindow::RebuildView);
    }
    subsystems->addStretch(1);
    layout->addLayout(subsystems);

    view = new QPlainTextEdit(central);
    view->setReadOnly(true);
    view->setLineWrapMode(QPlainTextEdit::NoWrap);
    view->setMaximumBlockCount(10000);
    view->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    view->setPlaceholderText(tr("Waiting for emulator log entries…"));
    layout->addWidget(view, 1);

    status = new QLabel(central);
    layout->addWidget(status);
    setCentralWidget(central);

    connect(search, &QLineEdit::textChanged, this, &ConsoleWindow::RebuildView);
    connect(level_combo, &QComboBox::currentIndexChanged, this, &ConsoleWindow::RebuildView);
    connect(clear_button, &QPushButton::clicked, this, &ConsoleWindow::Clear);
    connect(copy_button, &QPushButton::clicked, this, &ConsoleWindow::Copy);
    connect(save_button, &QPushButton::clicked, this, &ConsoleWindow::Save);
    connect(diagnostics_button, &QPushButton::clicked, this, &ConsoleWindow::ExportDiagnostics);

    poll_timer = new QTimer(this);
    connect(poll_timer, &QTimer::timeout, this, &ConsoleWindow::Poll);
    poll_timer->start(100);

    Common::Log::SetLiveConsoleCapacity(10000);
    Common::Log::SetLiveConsoleEnabled(true);
    RebuildView();
}

ConsoleWindow::~ConsoleWindow() = default;

void ConsoleWindow::closeEvent(QCloseEvent* event) {
    Common::Log::SetLiveConsoleEnabled(false);
    QMainWindow::closeEvent(event);
}

void ConsoleWindow::Poll() {
    if (pause_display->isChecked()) {
        UpdateStatus();
        return;
    }
    auto entries = Common::Log::ReadLiveConsole(last_sequence, 2000);
    if (!entries.empty()) {
        last_sequence = entries.back().sequence;
        AppendEntries(entries);
    }
    UpdateStatus();
}

void ConsoleWindow::RebuildView() {
    view->clear();
    auto entries = Common::Log::ReadLiveConsole(0, 10000);
    last_sequence = entries.empty() ? 0 : entries.back().sequence;
    AppendEntries(entries);
    UpdateStatus();
}

void ConsoleWindow::Clear() {
    Common::Log::ClearLiveConsole();
    view->clear();
    last_sequence = 0;
    UpdateStatus();
}

void ConsoleWindow::Copy() const {
    const auto cursor = view->textCursor();
    QGuiApplication::clipboard()->setText(cursor.hasSelection() ? cursor.selectedText()
                                                                : view->toPlainText());
}

void ConsoleWindow::Save() {
    const auto filename = QFileDialog::getSaveFileName(
        this, tr("Save Console Log"),
        QStringLiteral("veshiru-console.log"), tr("Log files (*.log);;Text files (*.txt)"));
    if (filename.isEmpty()) {
        return;
    }
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        QMessageBox::warning(this, tr("Save Console Log"),
                             tr("Could not write the selected file."));
        return;
    }
    QTextStream stream(&file);
    for (const auto& entry : Common::Log::ReadLiveConsole(0, 10000)) {
        if (PassesFilter(entry)) {
            stream << FormatPlain(entry) << '\n';
        }
    }
}

void ConsoleWindow::ExportDiagnostics() {
    const auto filename = QFileDialog::getSaveFileName(
        this, tr("Export Diagnostic Bundle"), QStringLiteral("veshiru-diagnostics.zip"),
        tr("Zip archives (*.zip)"));
    if (filename.isEmpty()) {
        return;
    }

    QuaZip zip(filename);
    if (!zip.open(QuaZip::mdCreate)) {
        QMessageBox::warning(this, tr("Export Diagnostics"), tr("Could not create the archive."));
        return;
    }

    const auto add_text = [&zip](const QString& name, const QByteArray& data) {
        QuaZipFile file(&zip);
        if (!file.open(QIODevice::WriteOnly, QuaZipNewInfo(name))) {
            return false;
        }
        const auto written = file.write(data);
        file.close();
        return written == data.size();
    };

    QString system;
    {
        QTextStream stream(&system);
        stream << "Veshiru diagnostic bundle\n";
        stream << "Generated: " << QDateTime::currentDateTimeUtc().toString(Qt::ISODate) << "\n\n";
        stream << "Build: " << QString::fromUtf8(Common::g_build_fullname) << "\n";
        stream << "Version: " << QString::fromUtf8(Common::g_build_version) << "\n";
        stream << "Commit: " << QString::fromUtf8(Common::g_scm_rev) << "\n";
        stream << "Build date: " << QString::fromUtf8(Common::g_build_date) << "\n";
        stream << "Compiler: " << QString::fromUtf8(Common::g_compiler_id) << "\n";
        stream << "OS: " << QSysInfo::prettyProductName() << " (" << QSysInfo::kernelVersion()
               << ", " << QSysInfo::currentCpuArchitecture() << ")\n";
#ifdef ARCHITECTURE_x86_64
        stream << "CPU: " << QString::fromUtf8(Common::g_cpu_caps.cpu_string) << "\n";
        stream << "CPU base/max MHz: " << Common::g_cpu_caps.base_frequency << " / "
               << Common::g_cpu_caps.max_frequency << "\n";
        stream << "AVX2: " << (Common::g_cpu_caps.avx2 ? "yes" : "no") << "\n";
#endif
        stream << "Logical processors: " << QThread::idealThreadCount() << "\n";
        const auto ram = Common::GetHostRam();
        stream << "RAM MiB: " << ram.bytes / (1024 * 1024) << "\n\n";
        stream << "Renderer backend: "
               << QString::fromStdString(std::string{Settings::CanonicalizeEnum(
                      Settings::values.renderer_backend.GetValue())})
               << "\n";
        stream << "Selected Vulkan device index: " << Settings::values.vulkan_device.GetValue()
               << "\n";
        stream << "GPU accuracy: "
               << QString::fromStdString(std::string{Settings::CanonicalizeEnum(
                      Settings::values.gpu_accuracy.GetValue())})
               << "\n";
        stream << "Texture cache memory limit MiB (0=Auto): "
               << Settings::values.texture_cache_memory_limit.GetValue() << "\n";
        stream << "Disk shader cache: " << Settings::values.use_disk_shader_cache.GetValue()
               << "\n";
        stream << "Vulkan driver pipeline cache: "
               << Settings::values.use_vulkan_driver_pipeline_cache.GetValue() << "\n";
        stream << "Asynchronous shaders: " << Settings::values.use_asynchronous_shaders.GetValue()
               << "\n";
        stream << "Log filter: " << QString::fromStdString(Settings::values.log_filter.GetValue())
               << "\n\n";

        std::vector<VkDeviceInfo::Record> records;
        VkDeviceInfo::PopulateRecords(records, nullptr);
        stream << "Vulkan devices:\n";
        for (const auto& record : records) {
            stream << "  - " << QString::fromStdString(record.name) << "\n";
            stream << "    Device-local memory MiB: " << record.device_local_memory / (1024 * 1024)
                   << "\n";
            stream << "    Broken compute: " << (record.has_broken_compute ? "yes" : "no") << "\n";
            stream << "    FSR4 capable: " << (record.fsr4_capable ? "yes" : "no") << "\n";
        }
    }

    QString console;
    {
        QTextStream stream(&console);
        for (const auto& entry : Common::Log::ReadLiveConsole(0, 10000)) {
            stream << FormatPlain(entry) << '\n';
        }
    }

    const auto log_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir);
    const auto add_sanitized_log = [&zip](const QString& name, const std::filesystem::path& path) {
        QFile input(QString::fromStdString(Common::FS::PathToUTF8String(path)));
        if (!input.open(QIODevice::ReadOnly)) {
            return true; // A rotated log may not exist yet.
        }
        QuaZipFile output(&zip);
        if (!output.open(QIODevice::WriteOnly, QuaZipNewInfo(name))) {
            return false;
        }
        while (!input.atEnd()) {
            const auto line = QString::fromUtf8(input.readLine());
            const auto sanitized =
                QString::fromStdString(Common::Log::SanitizeLogMessage(line.toStdString())).toUtf8();
            if (output.write(sanitized) != sanitized.size()) {
                output.close();
                return false;
            }
        }
        output.close();
        return true;
    };

    bool ok = true;
    ok &= add_text(QStringLiteral("README.txt"),
                   QByteArrayLiteral("This bundle contains sanitized emulator logs, build information "
                                     "and hardware/graphics details. It does not include game files, "
                                     "firmware, encryption keys, save data or personal documents. "
                                     "Review it before sharing.\n"));
    ok &= add_text(QStringLiteral("system.txt"), system.toUtf8());
    ok &= add_text(QStringLiteral("console.log"), console.toUtf8());
    ok &= add_sanitized_log(QStringLiteral("logs/") + QString::fromUtf8(LOG_FILE), log_dir / LOG_FILE);
    ok &= add_sanitized_log(QStringLiteral("logs/") + QString::fromUtf8(LOG_FILE) +
                                QStringLiteral(".old.txt"),
                            log_dir / (std::string{LOG_FILE} + ".old.txt"));
    ok &= add_sanitized_log(QStringLiteral("logs/eden_log_2.txt"), log_dir / "eden_log_2.txt");
    ok &= add_sanitized_log(QStringLiteral("logs/eden_log_2.txt.old.txt"),
                            log_dir / "eden_log_2.txt.old.txt");
    zip.close();

    if (!ok) {
        QMessageBox::warning(this, tr("Export Diagnostics"),
                             tr("The archive was created, but some files could not be written."));
        return;
    }
    QMessageBox::information(this, tr("Export Diagnostics"),
                             tr("Diagnostic bundle saved. No game files, saves, keys or credentials "
                                "are included."));
}

void ConsoleWindow::UpdateStatus() {
    const auto stats = Common::Log::GetLiveConsoleStats();
    status->setText(tr("Buffered: %1 | Dropped queue: %2 | Dropped buffer: %3 | Rate-limited: %4")
                        .arg(stats.buffered)
                        .arg(stats.dropped_queue)
                        .arg(stats.dropped_buffer)
                        .arg(stats.rate_limited));
}

bool ConsoleWindow::PassesFilter(const Common::Log::ConsoleEntry& entry) const {
    if (static_cast<int>(entry.log_level) < level_combo->currentData().toInt()) {
        return false;
    }
    if (!subsystem_checks[static_cast<std::size_t>(entry.subsystem)]->isChecked()) {
        return false;
    }
    const auto needle = search->text();
    return needle.isEmpty() || FormatPlain(entry).contains(needle, Qt::CaseInsensitive);
}

void ConsoleWindow::AppendEntries(const std::vector<Common::Log::ConsoleEntry>& entries) {
    if (entries.empty()) {
        return;
    }
    view->setUpdatesEnabled(false);
    for (const auto& entry : entries) {
        if (!PassesFilter(entry)) {
            continue;
        }
        view->appendHtml(QStringLiteral("<span style=\"color:%1; white-space:pre;\">%2</span>")
                             .arg(LevelColor(entry.log_level), FormatPlain(entry).toHtmlEscaped()));
    }
    view->setUpdatesEnabled(true);
    if (auto_scroll->isChecked()) {
        view->verticalScrollBar()->setValue(view->verticalScrollBar()->maximum());
    }
}

QString ConsoleWindow::FormatPlain(const Common::Log::ConsoleEntry& entry) const {
    return QStringLiteral("%1 %2 <%3> %4: %5 (%6)")
        .arg(Timestamp(entry.timestamp_us),
             FromView(Common::Log::GetSubsystemName(entry.subsystem)),
             FromView(Common::Log::GetLogLevelName(entry.log_level)),
             FromView(Common::Log::GetLogClassName(entry.log_class)),
             QString::fromStdString(entry.message), QString::fromStdString(entry.source));
}
