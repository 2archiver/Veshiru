// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#endif

#include "common/console_output.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

#ifdef _WIN32
static int CheckConsoleColors() {
    using namespace Common::Log;
    const auto buffer = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                                   CONSOLE_TEXTMODE_BUFFER, nullptr);
    if (buffer == INVALID_HANDLE_VALUE) {
        return 5;
    }
    const auto out = GetStdHandle(STD_OUTPUT_HANDLE);
    const auto err = GetStdHandle(STD_ERROR_HANDLE);
    constexpr WORD original = BACKGROUND_BLUE | FOREGROUND_GREEN;
    constexpr std::array<WORD, 3> expected{
        BACKGROUND_BLUE | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
        BACKGROUND_BLUE | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY,
        BACKGROUND_BLUE | FOREGROUND_RED | FOREGROUND_INTENSITY,
    };
    std::array<WORD, 3> attributes{};
    DWORD read{};
    CONSOLE_SCREEN_BUFFER_INFO info{};
    bool ok = SetStdHandle(STD_OUTPUT_HANDLE, buffer) && SetStdHandle(STD_ERROR_HANDLE, buffer) &&
              SetConsoleCursorPosition(buffer, {0, 0}) && SetConsoleTextAttribute(buffer, original);
    if (ok) {
        StandardStreamSink stream;
        stream.Enqueue("I", false, false, ConsoleColor::Normal);
        stream.Enqueue("W", false, true, ConsoleColor::Warning);
        stream.Enqueue("E", true, true, ConsoleColor::Error);
        stream.Close();
        ok = ReadConsoleOutputAttribute(buffer, attributes.data(), 3, {0, 0}, &read) &&
             read == 3 && attributes == expected && GetConsoleScreenBufferInfo(buffer, &info) &&
             info.wAttributes == original;
    }
    SetStdHandle(STD_OUTPUT_HANDLE, out);
    SetStdHandle(STD_ERROR_HANDLE, err);
    CloseHandle(buffer);
    return ok ? 0 : 6;
}
#endif

int main(int argc, char* argv[]) {
    using namespace Common;
    namespace fs = std::filesystem;
    if (argc != 2) {
        return 1;
    }
    const std::string_view action{argv[1]};
#ifdef _WIN32
    if (action == "colors") {
        return CheckConsoleColors();
    }
#endif
    const auto directory = fs::current_path();
    FS::SetEdenPath(FS::EdenPath::LogDir, directory);
    const auto file = directory / "mirror.log";
    if (action == "fifo") {
        return !Log::Initialize({.asynchronous = true, .extra_file = directory / "log-fifo"}) ? 0
                                                                                              : 2;
    }
    if (!Log::Initialize({.console = true, .asynchronous = true, .extra_file = file})) {
        return 3;
    }
    LOG_INFO(Frontend, "Booting caf{}", "\xC3\xA9");
    LOG_ERROR(Frontend, "early error");
    if (action == "flood") {
        const std::string payload(1024, 'x');
        for (int i = 0; i < 24'000; ++i) {
            LOG_INFO(Frontend, "{} {}", i, payload);
        }
    }
    LOG_WARNING(Frontend, "warning persisted before shutdown");
    Log::Flush();
    std::ifstream in(file, std::ios::binary);
    const std::string flushed{std::istreambuf_iterator<char>{in}, {}};
    in.close();
    LOG_INFO(Frontend, "orderly shutdown");
    Log::Shutdown();
    return flushed.find("warning persisted before shutdown") != std::string::npos ? 0 : 4;
}
