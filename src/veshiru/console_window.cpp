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
#include <QFont>
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
    // Severity colours on the Deep Current console surface. Trace/Debug stay muted so ordinary
    // output recedes, info is the primary text colour and warnings/errors keep their own hue
    // instead of the blue accent, so severity remains readable at a glance.
    switch (level) {
    case Level::Trace:
        return QStringLiteral("#55708c");
    case Level::Debug:
        return QStringLiteral("#849cb6");
    case Level::Info:
        return QStringLiteral("#eaf4ff");
    case Level::Warning:
        return QStringLiteral("#f2b84b");
    case Level::Error:
        return QStringLiteral("#ff647c");
    case Level::Critical:
        return QStringLiteral("#ff9ecb");
    default:
        return QStringLiteral("#eaf4ff");
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
    resize(1150, 700);

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
    save_button->setToolTip(tr("Save a sanitized .log file including build, hardware, Vulkan "
                               "device and session log diagnostics."));
    controls->addWidget(clear_button);
    controls->addWidget(copy_button);
    controls->addWidget(save_button);
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
    auto console_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    // Larger, high-contrast text on a near-black background, like a plain terminal console.
    console_font.setPointSize(12);
    console_font.setStyleHint(QFont::Monospace);
    view->setFont(console_font);
    view->setStyleSheet(QStringLiteral(
        "QPlainTextEdit { background-color: #050c16; color: #eaf4ff;"
        " border: 1px solid #1d3b58; border-radius: 8px; padding: 6px; }\n"
        "QPlainTextEdit:focus { border-color: #42d9f5; }\n"
        "QPlainTextEdit::selection { background-color: #1e5fa8; color: #ffffff; }"));
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
        this, tr("Save Diagnostic Log"),
        QStringLiteral("veshiru.log"), tr("Log files (*.log);;Text files (*.txt)"));
    if (filename.isEmpty()) {
        return;
    }
    Common::Log::Flush();

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
        QMessageBox::warning(this, tr("Save Diagnostic Log"),
                             tr("Could not write the selected file."));
        return;
    }
    QTextStream stream(&file);
    stream << "# Veshiru Sanitized Diagnostic Log\n";
    stream << "# Generated: " << QDateTime::currentDateTimeUtc().toString(Qt::ISODate) << "\n";
    stream << "# Build: " << QString::fromUtf8(Common::g_build_fullname).trimmed() << " ("
           << QString::fromUtf8(Common::g_build_version) << ", commit "
           << QString::fromUtf8(Common::g_scm_rev) << ")\n";
    stream << "# Built: " << QString::fromUtf8(Common::g_build_date) << " | Compiler: "
           << QString::fromUtf8(Common::g_compiler_id) << "\n";
    stream << "# OS: " << QSysInfo::prettyProductName() << " (" << QSysInfo::kernelVersion() << ", "
           << QSysInfo::currentCpuArchitecture() << ")\n";
#ifdef ARCHITECTURE_x86_64
    stream << "# CPU: " << QString::fromUtf8(Common::g_cpu_caps.cpu_string)
           << " | AVX2: " << (Common::g_cpu_caps.avx2 ? "yes" : "no") << "\n";
#endif
    const auto ram = Common::GetHostRam();
    stream << "# Logical processors: " << QThread::idealThreadCount()
           << " | RAM MiB: " << ram.bytes / (1024 * 1024) << "\n";
    stream << "# Renderer: "
           << QString::fromStdString(std::string{
                  Settings::CanonicalizeEnum(Settings::values.renderer_backend.GetValue())})
           << " | Vulkan device index: " << Settings::values.vulkan_device.GetValue()
           << " | GPU accuracy: "
           << QString::fromStdString(std::string{
                  Settings::CanonicalizeEnum(Settings::values.gpu_accuracy.GetValue())})
           << " | Texture cache limit MiB: "
           << Settings::values.texture_cache_memory_limit.GetValue() << "\n";
    stream << "# Disk shader cache: " << Settings::values.use_disk_shader_cache.GetValue()
           << " | Driver pipeline cache: "
           << Settings::values.use_vulkan_driver_pipeline_cache.GetValue()
           << " | Async shaders: " << Settings::values.use_asynchronous_shaders.GetValue()
           << " | Frame generation: "
           << QString::fromStdString(std::string{
                  Settings::CanonicalizeEnum(Settings::values.frame_gen_type.GetValue())})
           << "\n";
    std::vector<VkDeviceInfo::Record> records;
    VkDeviceInfo::PopulateRecords(records, nullptr);
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        stream << "# Vulkan GPU [" << i << "]: " << QString::fromStdString(record.name) << " ("
               << record.device_local_memory / (1024 * 1024) << " MiB VRAM)\n";
    }

    const bool filtered = HasActiveFilter();
    const auto active_log_path = Common::Log::GetActiveLogFilePath();
    bool wrote_full_session = false;
    if (!filtered && !active_log_path.empty()) {
        QFile input(QString::fromStdString(Common::FS::PathToUTF8String(active_log_path)));
        if (input.open(QIODevice::ReadOnly | QIODevice::Text)) {
            stream << "# Source: full sanitized session log\n\n";
            while (!input.atEnd()) {
                const auto line = QString::fromUtf8(input.readLine());
                stream << QString::fromStdString(
                    Common::Log::SanitizeLogMessage(line.toStdString()));
            }
            wrote_full_session = true;
        }
    }
    if (!wrote_full_session) {
        stream << (filtered ? "# Source: filtered console view\n\n"
                            : "# Source: live console buffer\n\n");
        for (const auto& entry : Common::Log::ReadLiveConsole(0, 10000)) {
            if (!filtered || PassesFilter(entry)) {
                stream << FormatPlain(entry) << '\n';
            }
        }
    }
}

void ConsoleWindow::UpdateStatus() {
    const auto stats = Common::Log::GetLiveConsoleStats();
    status->setText(tr("Buffered: %1 | Dropped queue: %2 | Dropped buffer: %3 | Rate-limited: %4")
                        .arg(stats.buffered)
                        .arg(stats.dropped_queue)
                        .arg(stats.dropped_buffer)
                        .arg(stats.rate_limited));
}

bool ConsoleWindow::HasActiveFilter() const {
    if (level_combo->currentData().toInt() > static_cast<int>(Common::Log::Level::Trace)) {
        return true;
    }
    if (!search->text().isEmpty()) {
        return true;
    }
    for (const auto* check : subsystem_checks) {
        if (check && !check->isChecked()) {
            return true;
        }
    }
    return false;
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
