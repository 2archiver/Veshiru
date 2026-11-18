// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <cstring>
#include <exception>
#include <utility>
#include <QApplication>
#include <QTimer>

#include "common/console_output.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "diagnostics_options.h"
#include "native_console.h"
#include "qt_common/qt_common.h"
#ifdef VESHIRU_CRASH_DUMPS
#include "breakpad.h"
#endif
#include "nx2_switches.h"
#include "startup_checks.h"

#if VESHIRU_ROOM
#include "dedicated_room/veshiru_room.h"
#endif
#ifdef __unix__
#include "qt_common/gui_settings.h"
#endif

#ifndef _WIN32
#include <sys/resource.h>
#endif

#if defined(__APPLE__)
#include <climits>
#include <cstdlib>
#include <cstring>
#endif

#include "main_window.h"

#ifdef _WIN32
#include <QScreen>
#include <shellapi.h>
#include <windows.h>
#include "common/windows/crash_handler.h"

static void OverrideWindowsFont() {
    // Qt5 chooses these fonts on Windows and they have fairly ugly alphanumeric/cyrillic characters
    // Asking to use "MS Shell Dlg 2" gives better other chars while leaving the Chinese Characters.
    const QString startup_font = QApplication::font().family();
    const QStringList ugly_fonts = {QStringLiteral("SimSun"), QStringLiteral("PMingLiU")};
    if (ugly_fonts.contains(startup_font)) {
        QApplication::setFont(QFont(QStringLiteral("MS Shell Dlg 2"), 9, QFont::Normal));
    }
}
#endif

static Qt::HighDpiScaleFactorRoundingPolicy GetHighDpiRoundingPolicy() {
#ifdef _WIN32
    // For Windows, we want to avoid scaling artifacts on fractional scaling ratios.
    // This is done by setting the optimal scaling policy for the primary screen.

    // Create a temporary QApplication.
    int temp_argc = 0;
    char** temp_argv = nullptr;
    QApplication temp{temp_argc, temp_argv};

    // Get the current screen geometry.
    const QScreen* primary_screen = QGuiApplication::primaryScreen();
    if (primary_screen == nullptr) {
        return Qt::HighDpiScaleFactorRoundingPolicy::PassThrough;
    }

    const QRect screen_rect = primary_screen->geometry();
    const qreal real_ratio = primary_screen->devicePixelRatio();
    const qreal real_width = std::trunc(screen_rect.width() * real_ratio);
    const qreal real_height = std::trunc(screen_rect.height() * real_ratio);

    // Recommended minimum width and height for proper window fit.
    // Any screen with a lower resolution than this will still have a scale of 1.
    constexpr qreal minimum_width = 1350.0;
    constexpr qreal minimum_height = 900.0;

    const qreal width_ratio = std::max(1.0, real_width / minimum_width);
    const qreal height_ratio = std::max(1.0, real_height / minimum_height);

    // Get the lower of the 2 ratios and truncate, this is the maximum integer scale.
    const qreal max_ratio = std::trunc(std::min(width_ratio, height_ratio));
    return max_ratio > real_ratio ? Qt::HighDpiScaleFactorRoundingPolicy::Round
                                  : Qt::HighDpiScaleFactorRoundingPolicy::Floor;
#else
    // Other OSes should be better than Windows at fractional scaling.
    return Qt::HighDpiScaleFactorRoundingPolicy::PassThrough;
#endif
}

static QList<QByteArray> NativeArguments(int argc, char* argv[]) {
#ifdef _WIN32
    int wide_argc{};
    if (auto* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc)) {
        SCOPE_EXIT {
            LocalFree(wide_argv);
        };
        QList<QByteArray> args;
        for (int i = 0; i < wide_argc; ++i) {
            args.push_back(QString::fromWCharArray(wide_argv[i]).toUtf8());
        }
        return args;
    }
#endif
    QList<QByteArray> args;
    for (int i = 0; i < argc; ++i) {
        args.push_back(QByteArray{argv[i]});
    }
    return args;
}

static std::vector<std::string_view> ArgumentViews(const QList<QByteArray>& args) {
    std::vector<std::string_view> views;
    views.reserve(static_cast<std::size_t>(args.size()));
    for (const auto& arg : args) {
        views.emplace_back(arg.constData(), static_cast<std::size_t>(arg.size()));
    }
    return views;
}

static QStringList FrontendArguments() {
    const QStringList args = QCoreApplication::arguments();
    QList<QByteArray> utf8;
    for (const auto& arg : args) {
        utf8.push_back(arg.toUtf8());
    }
    const auto options = Veshiru::ParseDiagnosticsOptions(ArgumentViews(utf8));
    QStringList remaining;
    for (const auto i : options.remaining) {
        remaining.push_back(args[i]);
    }
    return remaining;
}

static void WriteStartupOutput(std::string_view text, bool to_stderr = false) noexcept {
    try {
        Common::Log::StandardStreamSink stream;
        stream.Enqueue(std::string{text}, to_stderr, true);
    } catch (const std::exception&) {
        // Thread/allocation failure: do not block on a possibly unread inherited stream.
    }
}

static void QtLogHandler(QtMsgType type, const QMessageLogContext&, const QString& message) {
    switch (type) {
    case QtDebugMsg:
        LOG_DEBUG(Frontend, "Qt: {}", message.toStdString());
        break;
    case QtInfoMsg:
        LOG_INFO(Frontend, "Qt: {}", message.toStdString());
        break;
    case QtWarningMsg:
        LOG_WARNING(Frontend, "Qt: {}", message.toStdString());
        break;
    case QtCriticalMsg:
        LOG_ERROR(Frontend, "Qt: {}", message.toStdString());
        break;
    case QtFatalMsg:
        LOG_CRITICAL(Frontend, "Qt fatal error: {}", message.toStdString());
        Common::Log::Flush();
        Common::Log::FlushConsole();
        Veshiru::NativeConsole::ShowEarlyError(
            Common::Log::SanitizeLogMessage(message.toStdString()));
        break;
    }
}

static int RunFrontend(int argc, char* argv[]) {
#ifdef VESHIRU_CRASH_DUMPS
    Breakpad::InstallCrashHandler();
#endif

    // Init settings params
    QCoreApplication::setOrganizationName(QStringLiteral("eden"));
    QCoreApplication::setApplicationName(QStringLiteral("Veshiru"));

    // Increases the maximum open file limit.
    // TODO: This should be common to all frontends.
#ifdef _WIN32
    // MSVCRT limits this to 2048 for some inexplicable (and likely arcane) reason,
    // so we have to account for that as well.
#ifdef __MSVCRT__
    _setmaxstdio(2048);
#else
    _setmaxstdio(8192);
#endif // __MSVCRT__
#elif defined(__unix__) || defined(__APPLE__)
    // Raise the open file limit to at least 8192, or the hard limit.
    // Most sane systems should not hit the hard limit here.
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        rl.rlim_cur = std::min<rlim_t>(1048576, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }
#endif // _WIN32

#if defined(__APPLE__)
    // Convert the relative path to an absolute path before the chdir
    char resolved[PATH_MAX];
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "-input-profile") == 0) {
            ++i;
        } else if (argv[i][0] != '-' && argv[i][0] != '/' && realpath(argv[i], resolved)) {
            argv[i] = strdup(resolved);
        }
    }

    // If you start a bundle (binary) on OSX without the Terminal, the working directory is "/".
    // But since we require the working directory to be the executable path for the location of
    // the user folder in the Qt Frontend, we need to cd into that working directory
    const auto bin_path = Common::FS::GetBundleDirectory() / "..";
    chdir(Common::FS::PathToUTF8String(bin_path).c_str());
#endif

#ifdef __unix__
    // Set the DISPLAY variable in order to open web browsers
    // TODO (lat9nq): Find a better solution for AppImages to start external applications
    if (QString::fromLocal8Bit(qgetenv("DISPLAY")).isEmpty()) {
        qputenv("DISPLAY", ":0");
    }

    if (GraphicsBackend::GetForceX11() && qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
        qputenv("QT_QPA_PLATFORM", "xcb");

    // Fix the Wayland appId. This needs to match the name of the .desktop file without the .desktop
    // suffix.
    QGuiApplication::setDesktopFileName(QStringLiteral("dev.veshiru.emulator"));
#endif

    auto rounding_policy = GetHighDpiRoundingPolicy();
    QApplication::setHighDpiScaleFactorRoundingPolicy(rounding_policy);

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    // Disables the "?" button on all dialogs. Disabled by default on Qt6.
    QCoreApplication::setAttribute(Qt::AA_DisableWindowContextHelpButton);
#endif

    // Enables the core to make the qt created contexts current on std::threads
    QCoreApplication::setAttribute(Qt::AA_DontCheckOpenGLContextThreadAffinity);

#ifdef _WIN32
    // A native window (the render widget's, for the graphics surface) would otherwise make its
    // siblings native too: the loading screen and the game list, next to GRenderWindow. In
    // single-window borderless fullscreen those cover the main window, which Qt then no longer
    // repaints (see RenderWidget in bootmanager.cpp).
    QCoreApplication::setAttribute(Qt::AA_DontCreateNativeWidgetSiblings);
#endif

#ifdef _WIN32
    QApplication::setStyle(QStringLiteral("windowsvista"));
#endif

    QApplication app(argc, argv);

#ifdef _WIN32
    OverrideWindowsFont();
#endif

    // Workaround for QTBUG-85409, for Suzhou numerals the number 1 is actually \u3021
    // so we can see if we get \u3008 instead
    // TL;DR all other number formats are consecutive in unicode code points
    // This bug is fixed in Qt6, specifically 6.0.0-alpha1
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    const QLocale locale = QLocale::system();
    if (QStringLiteral("\u3008") == locale.toString(1)) {
        QLocale::setDefault(QLocale::system().name());
    }
#endif

    // Qt changes the locale and causes issues in float conversion using std::to_string() when
    // generating shaders
    setlocale(LC_ALL, "C");

    if (Veshiru::NativeConsole::ShutdownRequested()) {
        return EXIT_SUCCESS;
    }
    MainWindow main_window{FrontendArguments()};
    main_window.show();
    app.connect(&app, &QGuiApplication::applicationStateChanged, &main_window,
                &MainWindow::OnAppFocusStateChanged);
    QTimer::singleShot(0, &app, [] { LOG_INFO(Frontend, "Frontend ready"); });
#ifdef _WIN32
    QTimer console_control;
    QObject::connect(&console_control, &QTimer::timeout, &main_window, [&] {
        if (Veshiru::NativeConsole::ShutdownRequested()) {
            console_control.stop();
            LOG_INFO(Frontend, "Console interrupt; requesting shutdown");
            main_window.RequestConsoleShutdown();
        }
    });
    console_control.start(100);
#endif
    return app.exec();
}

int main(int argc, char* argv[]) try {
    if (argc == 2 && std::string_view{argv[1]} == VulkanProbeArgument) {
        return ProbeVulkan() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#if VESHIRU_ROOM
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--room") == 0) {
            LoadNx2Switches(argv[0]);
            LaunchRoom(argc, argv, true);
            return EXIT_SUCCESS;
        }
    }
#endif
    const auto args = NativeArguments(argc, argv);
    const auto options = Veshiru::ParseDiagnosticsOptions(ArgumentViews(args));
    const Veshiru::NativeConsole console{options.console};
    if (!options.error.empty()) {
        WriteStartupOutput(fmt::format("Veshiru: {} (see --help)\n", options.error), true);
        Veshiru::NativeConsole::ShowEarlyError(options.error);
        return EXIT_FAILURE;
    }

    SCOPE_EXIT {
        qInstallMessageHandler(nullptr);
        Common::Log::Shutdown();
    };
    try {
        // --help and --version do not load Qt's platform plugin or touch user data unless
        // a log was explicitly requested. Resolve the path before any platform-specific chdir.
        if ((!options.help && !options.version) || !options.log_path.empty()) {
            const auto path = options.log_path.empty()
                                  ? std::filesystem::path{}
                                  : std::filesystem::absolute(std::filesystem::path{
                                        Common::FS::ToU8String(options.log_path)});
            if (!Common::Log::Initialize(
                    {.console = options.console != Veshiru::ConsoleMode::Disable,
                     .asynchronous = true,
                     .fixed_level = options.level,
                     .extra_file = path})) {
                constexpr std::string_view error =
                    "Could not open --log-file; check the path and permissions";
                LOG_CRITICAL(Frontend, "{}", error);
                Common::Log::Flush();
                Common::Log::FlushConsole();
                Veshiru::NativeConsole::ShowEarlyError(error);
                return EXIT_FAILURE;
            }
            qInstallMessageHandler(QtLogHandler);
#ifdef _WIN32
            Common::Windows::InstallCrashHandler(
                Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir));
#endif
            QtCommon::LogStartupInfo();
        }

        if (options.help) {
            Common::Log::Flush();
            Common::Log::FlushConsole();
            WriteStartupOutput(
                "Usage: veshiru [options] [game file]\n"
                "  -g, --game PATH     Open a game.\n"
                "  -f, --fullscreen    Start the game in fullscreen mode.\n"
                "  -c, --config PATH   Load the specified configuration.\n"
                "  -u, --user PROFILE  Select a user profile.\n"
                "  --pack-nup DUMP     Pack a loose dump into a NUP.\n"
                "  --unpack-nup FILE   Extract a NUP.\n"
                "  --edit-nup FILE     Edit NUP metadata.\n"
                "  --console           Enable logging to the system console (default).\n"
                "  --no-console        Disable console logging and detach on Windows.\n"
                "  --log-level LEVEL   Trace, Debug, Info, Warning, Error or Critical.\n"
                "  --log-file PATH     Append a UTF-8 copy of the session log.\n"
                "  --help, --version   Print help or the build version and exit.\n");
            return EXIT_SUCCESS;
        }
        if (options.version) {
            Common::Log::Flush();
            Common::Log::FlushConsole();
            WriteStartupOutput(
                fmt::format("Veshiru {} ({})\n", Common::g_build_version, Common::g_scm_rev));
            return EXIT_SUCCESS;
        }

        LoadNx2Switches(argv[0]);
        return RunFrontend(argc, argv);
    } catch (const std::exception& exception) {
        const auto message = Common::Log::SanitizeLogMessage(exception.what());
        if (Common::Log::IsLevelEnabled(Common::Log::Class::Frontend,
                                        Common::Log::Level::Critical)) {
            LOG_CRITICAL(Frontend, "Veshiru startup stopped: {}", exception.what());
            Common::Log::Flush();
            Common::Log::FlushConsole();
        } else {
            WriteStartupOutput(fmt::format("Veshiru startup stopped: {}\n", message), true);
        }
        Veshiru::NativeConsole::ShowEarlyError(message);
        return EXIT_FAILURE;
    }
} catch (const std::exception&) {
    constexpr std::string_view error = "Could not initialize Veshiru startup diagnostics";
    WriteStartupOutput("Could not initialize Veshiru startup diagnostics.\n", true);
    Veshiru::NativeConsole::ShowEarlyError(error);
    return EXIT_FAILURE;
}
