// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/literals.h"

#include "common/memory_detect.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "frontend_common/data_manager.h"
#include "hid_core/hid_core.h"
#include "network/network.h"
#include "qt_common.h"

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/cpu_features.h"
#include "core/memory.h"

#include <QGuiApplication>
#include <QStringLiteral>
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/frontend/emu_window.h"
#include "qt_common/util/meta.h"

#include <QFile>

#include <QMessageBox>

#include <thread>
#include <JlCompress.h>
#include <QPainter>

#if !defined(WIN32) && !defined(__APPLE__)
#include <qpa/qplatformnativeinterface.h>
#elif defined(__APPLE__)
#include <objc/message.h>
#endif

#ifdef _WIN32
#include <QApplication>
#include <QSettings>
#include <windows.h>
#include "common/windows/timer_resolution.h"
#include "core/core_timing.h"
#endif

using namespace Common::Literals;

namespace QtCommon {

QWidget* rootObject = nullptr;

std::unique_ptr<Core::System> system = nullptr;
std::shared_ptr<FileSys::RealVfsFilesystem> vfs = nullptr;
std::unique_ptr<FileSys::ManualContentProvider> provider = nullptr;
std::unique_ptr<EmuThread> emu_thread = nullptr;

const QStringList supported_file_extensions = {QStringLiteral("nro"), QStringLiteral("nso"),
                                               QStringLiteral("nca"), QStringLiteral("xci"),
                                               QStringLiteral("xcz"), QStringLiteral("nsp"),
                                               QStringLiteral("nsz"), QStringLiteral("nup"),
                                               QStringLiteral("nx"),  QStringLiteral("kip")};

Core::Frontend::WindowSystemType GetWindowSystemType() {
    // Determine WSI type based on Qt platform.
    QString platform_name = QGuiApplication::platformName();
    if (platform_name == QStringLiteral("windows"))
        return Core::Frontend::WindowSystemType::Windows;
    else if (platform_name == QStringLiteral("xcb"))
        return Core::Frontend::WindowSystemType::X11;
    else if (platform_name == QStringLiteral("wayland"))
        return Core::Frontend::WindowSystemType::Wayland;
    else if (platform_name == QStringLiteral("wayland-egl"))
        return Core::Frontend::WindowSystemType::Wayland;
    else if (platform_name == QStringLiteral("cocoa"))
        return Core::Frontend::WindowSystemType::Cocoa;
    else if (platform_name == QStringLiteral("android"))
        return Core::Frontend::WindowSystemType::Android;
    else if (platform_name == QStringLiteral("haiku"))
        return Core::Frontend::WindowSystemType::Xcb;

    LOG_CRITICAL(Frontend, "Unknown Qt platform {}!", platform_name.toStdString());
    return Core::Frontend::WindowSystemType::Windows;
} // namespace Core::Frontend::WindowSystemType

Core::Frontend::EmuWindow::WindowSystemInfo GetWindowSystemInfo(QWindow* window) {
    Core::Frontend::EmuWindow::WindowSystemInfo wsi;
    wsi.type = GetWindowSystemType();

#if defined(WIN32)
    // Our Win32 Qt external doesn't have the private API.
    wsi.render_surface = reinterpret_cast<void*>(window->winId());
#elif defined(__APPLE__)
    id layer = reinterpret_cast<id (*)(id, SEL)>(objc_msgSend)(
        reinterpret_cast<id>(window->winId()), sel_registerName("layer"));

    // In Qt 6, the layer of the NSView might be a QContainerLayer.
    // MoltenVK needs a CAMetalLayer. We search for it in sublayers.
    Class metal_layer_class = objc_getClass("CAMetalLayer");
    id metal_layer = nullptr;

    if (layer) {
        if (reinterpret_cast<bool (*)(id, SEL, Class)>(objc_msgSend)(
                layer, sel_registerName("isKindOfClass:"), metal_layer_class)) {
            metal_layer = layer;
        } else {
            id sublayers = reinterpret_cast<id (*)(id, SEL)>(objc_msgSend)(
                layer, sel_registerName("sublayers"));
            if (sublayers) {
                unsigned long count = reinterpret_cast<unsigned long (*)(id, SEL)>(objc_msgSend)(
                    sublayers, sel_registerName("count"));
                for (unsigned long i = 0; i < count; ++i) {
                    id sublayer = reinterpret_cast<id (*)(id, SEL, unsigned long)>(objc_msgSend)(
                        sublayers, sel_registerName("objectAtIndex:"), i);
                    if (reinterpret_cast<bool (*)(id, SEL, Class)>(objc_msgSend)(
                            sublayer, sel_registerName("isKindOfClass:"), metal_layer_class)) {
                        metal_layer = sublayer;
                        break;
                    }
                }
            }
        }
    }
    wsi.render_surface = reinterpret_cast<void*>(metal_layer ? metal_layer : layer);
#else
    QPlatformNativeInterface* pni = QGuiApplication::platformNativeInterface();
    wsi.display_connection = pni->nativeResourceForWindow("display", window);
    if (wsi.type == Core::Frontend::WindowSystemType::Wayland)
        wsi.render_surface = window ? pni->nativeResourceForWindow("surface", window) : nullptr;
    else
        wsi.render_surface = window ? reinterpret_cast<void*>(window->winId()) : nullptr;
#endif
    wsi.render_surface_scale = window ? static_cast<float>(window->devicePixelRatio()) : 1.0f;

    return wsi;
}

const QString tr(const char* str) {
    return rootObject->tr(str);
}

const QString tr(const std::string& str) {
    return rootObject->tr(str.c_str());
}

static QString PrettyProductName() {
#ifdef _WIN32
    // After Windows 10 Version 2004, Microsoft decided to switch to a different notation: 20H2
    // With that notation change they changed the registry key used to denote the current version
    QSettings windows_registry(
        QStringLiteral("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion"),
        QSettings::NativeFormat);
    const QString release_id = windows_registry.value(QStringLiteral("ReleaseId")).toString();
    if (release_id == QStringLiteral("2009")) {
        const u32 current_build = windows_registry.value(QStringLiteral("CurrentBuild")).toUInt();
        const QString display_version =
            windows_registry.value(QStringLiteral("DisplayVersion")).toString();
        const u32 ubr = windows_registry.value(QStringLiteral("UBR")).toUInt();
        u32 version = 10;
        if (current_build >= 22000) {
            version = 11;
        }
        return QStringLiteral("Windows %1 Version %2 (Build %3.%4)")
            .arg(QString::number(version), display_version, QString::number(current_build),
                 QString::number(ubr));
    }
#endif
    return QSysInfo::prettyProductName();
}

static void RemoveCachedContents() {
    const auto cache_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir);
    const auto offline_fonts = cache_dir / "fonts";
    const auto offline_manual = cache_dir / "offline_web_applet_manual";
    const auto offline_legal_information = cache_dir / "offline_web_applet_legal_information";
    const auto offline_system_data = cache_dir / "offline_web_applet_system_data";

    Common::FS::RemoveDirRecursively(offline_fonts);
    Common::FS::RemoveDirRecursively(offline_manual);
    Common::FS::RemoveDirRecursively(offline_legal_information);
    Common::FS::RemoveDirRecursively(offline_system_data);
}

void LogStartupInfo() {
    LOG_INFO(Frontend, "Veshiru Version: {} | Commit: {} | Built: {} | {} | Qt {}",
             Common::g_build_version, Common::g_scm_rev, Common::g_build_date,
             Common::g_compiler_id, qVersion());
#ifdef ARCHITECTURE_x86_64
    const std::string_view cpu = Common::g_cpu_caps.cpu_string;
#else
    const auto cpu = QSysInfo::currentCpuArchitecture().toStdString();
#endif
    LOG_INFO(Frontend, "Host: {} | {} ({} threads) | {:.1f} GiB RAM",
             PrettyProductName().toStdString(), cpu, std::thread::hardware_concurrency(),
             Common::GetMemInfo().TotalPhysicalMemory / f64{1_GiB});
}

void Init(QWidget* root) {
    rootObject = root;

    system = std::make_unique<Core::System>();
    vfs = std::make_unique<FileSys::RealVfsFilesystem>();
    provider = std::make_unique<FileSys::ManualContentProvider>();

    Common::FS::CreateEdenPaths();
    system->Initialize();
    Network::Init();
    QtCommon::Meta::RegisterMetaTypes();

#ifdef _WIN32
    LOG_DEBUG(Frontend, "Host timer resolution: {:.4f} ms",
              std::chrono::duration_cast<std::chrono::duration<f64, std::milli>>(
                  Common::Windows::SetCurrentTimerResolutionToMaximum())
                  .count());
    QtCommon::system->CoreTiming().SetTimerResolutionNs(
        Common::Windows::GetCurrentTimerResolution());
#endif

    // Remove cached contents generated during the previous session
    RemoveCachedContents();
}

std::filesystem::path GetEdenCommand() {
    std::filesystem::path command;

    // TODO: flatpak?
    QString appimage = QString::fromLocal8Bit(getenv("APPIMAGE"));
    if (!appimage.isEmpty()) {
        command = std::filesystem::path{appimage.toStdString()};
    } else {
        const QStringList args = QGuiApplication::arguments();
        command = args[0].toStdString();
    }

    // If relative path, make it an absolute path
    if (command.c_str()[0] == '.') {
        command = Common::FS::GetCurrentDir() / command;
    }

    return command;
}

void SetupContentProviders() {
    system->SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
    system->RegisterContentProvider(FileSys::ContentProviderUnionSlot::FrontendManual,
                                    provider.get());
    system->GetFileSystemController().CreateFactories(*vfs);
}

void SetupHID() {
    system->HIDCore().ReloadInputDevices();
}

QString ReadableByteSize(qulonglong size) {
    return QString::fromStdString(FrontendCommon::DataManager::ReadableBytesSize(size));
}

QPixmap CreateCirclePixmapFromColor(const QColor& color) {
    QPixmap circle_pixmap(16, 16);
    circle_pixmap.fill(Qt::transparent);
    QPainter painter(&circle_pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(color);
    painter.setBrush(color);
    painter.drawEllipse({circle_pixmap.width() / 2.0, circle_pixmap.height() / 2.0}, 7.0, 7.0);
    return circle_pixmap;
}

} // namespace QtCommon
