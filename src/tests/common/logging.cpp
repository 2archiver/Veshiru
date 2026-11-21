// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <catch2/catch_test_macros.hpp>

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "veshiru/diagnostics_options.h"

using namespace Common::Log;

TEST_CASE("Console sanitisation removes personal paths and credential-like values", "[logging]") {
    const auto windows = SanitizeLogMessage(R"(Loading C:\Users\Alice\Documents\game.nsp)");
    CHECK(windows.find("Alice") == std::string::npos);
    CHECK(windows.find("<path>") != std::string::npos);

    const auto posix = SanitizeLogMessage("Failed to open /home/alice/private/save.bin");
    CHECK(posix.find("alice") == std::string::npos);
    CHECK(posix.find("<path>") != std::string::npos);

    const auto credential = SanitizeLogMessage("request token=abc123 password: hunter2");
    CHECK(credential.find("abc123") == std::string::npos);
    CHECK(credential.find("hunter2") == std::string::npos);
    CHECK(credential.find("<redacted>") != std::string::npos);

    const auto url = SanitizeLogMessage("fetch https://user:secret@example.invalid/path");
    CHECK(url.find("secret") == std::string::npos);
    CHECK(url.find("<redacted>@") != std::string::npos);

    const auto win_ext = SanitizeLogMessage(R"(Open \\?\C:\Users\Alice\Games\mkw.nup)");
    CHECK(win_ext.find("Alice") == std::string::npos);
    CHECK(win_ext.find("<path>") != std::string::npos);

    const auto auth = SanitizeLogMessage("Authorization: Bearer ghp_secret123456");
    CHECK(auth.find("ghp_secret123456") == std::string::npos);
    CHECK(auth.find("<redacted>") != std::string::npos);

    const auto truncated = SanitizeLogMessage("invalid response: api_key=\"two words");
    CHECK(truncated.find("two words") == std::string::npos);

    // Relative source locations and title IDs remain useful diagnostics.
    const auto network = SanitizeLogMessage("peer=192.168.1.42 mac=00:1a:2b:3c:4d:5e");
    CHECK(network.find("192.168.1.42") == std::string::npos);
    CHECK(network.find("00:1a:2b:3c:4d:5e") == std::string::npos);
    CHECK(network.find("<ip>") != std::string::npos);
    CHECK(network.find("<mac>") != std::string::npos);

    const auto relative = SanitizeLogMessage("core/loader/loader.cpp: 0100ABCD00000000");
    CHECK(relative.find("core/loader/loader.cpp") != std::string::npos);
    CHECK(relative.find("0100ABCD00000000") != std::string::npos);
}

TEST_CASE("Diagnostics options keep other launch arguments and accept UTF-8 paths", "[logging]") {
    using namespace std::literals;
    const std::array args{"veshiru"sv,    "--no-console"sv,
                          "--console"sv,  "--log-level=debug"sv,
                          "--log-file"sv, "logs/boot-\xC3\xA9.txt"sv,
                          "--game"sv,     "--console"sv,
                          "--"sv,         "--no-console"sv};
    const auto options = Veshiru::ParseDiagnosticsOptions(args);
    CHECK(options.error.empty());
    CHECK(options.console == Veshiru::ConsoleMode::Enable);
    CHECK(options.level == Level::Debug);
    CHECK(options.log_path == args[5]);
    CHECK(options.remaining == std::vector<int>{0, 6, 7, 8, 9});

    CHECK_FALSE(
        Veshiru::ParseDiagnosticsOptions(std::array{"veshiru"sv, "--log-file"sv}).error.empty());
    CHECK_FALSE(
        Veshiru::ParseDiagnosticsOptions(std::array{"veshiru"sv, "--log-level=W"sv}).error.empty());
    CHECK_FALSE(
        Veshiru::ParseDiagnosticsOptions(std::array{"veshiru"sv, "--console=yes"sv}).error.empty());
    const std::array flag_args{"veshiru"sv, "--filter"sv, "--no-console"sv};
    CHECK(Veshiru::ParseDiagnosticsOptions(flag_args).console == Veshiru::ConsoleMode::Disable);
    CHECK(Veshiru::ParseDiagnosticsOptions(std::array{"veshiru"sv, "--version"sv}).version);
}

TEST_CASE("Async logger drains both files and keeps the explicit level across config changes",
          "[logging]") {
    using namespace Common::FS;
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() /
                     ("veshiru-log-test-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(fs::create_directory(dir));
    const auto old_log_dir = GetEdenPath(EdenPath::LogDir);
    struct Cleanup {
        fs::path dir;
        fs::path old_log_dir;
        ~Cleanup() {
            Shutdown();
            SetEdenPath(EdenPath::LogDir, old_log_dir);
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
    } cleanup{dir, old_log_dir};
    SetEdenPath(EdenPath::LogDir, dir);
    const auto extra = dir / fs::path{u8"boot-\u00E9.log"};
    std::ofstream{extra, std::ios::binary} << "previous session\n";
    REQUIRE(Initialize({.asynchronous = true, .fixed_level = Level::Info, .extra_file = extra}));
    SetGlobalFilter(Filter{Level::Critical});
    LOG_DEBUG(Frontend, "debug record should not appear");
    LOG_INFO(Frontend, "UTF-8: caf{}", "\xC3\xA9");
    LOG_WARNING(Frontend, "private data: \"C:\\Users\\Person\\Private "
                          "Folder\\File.nsp\"\n\x1b[31msecret='two words'");
    LOG_INFO(Frontend, "invalid text: {}", "\xff ASCII survives");
    std::array<std::thread, 4> workers;
    for (std::size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([worker] {
            for (int i = 0; i < 64; ++i) {
                LOG_INFO(Frontend, "worker={} record={}", worker, i);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    Flush();
    LOG_ERROR(Frontend, "last queued record");
    Shutdown();
    const auto read = [](const fs::path& path) {
        std::ifstream file(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    };
    const auto file = read(dir / LOG_FILE);
    REQUIRE_FALSE(file.empty());
    CHECK(read(extra) == "previous session\n" + file);
    CHECK(file.find("caf\xC3\xA9") != std::string::npos);
    CHECK(file.find("\xEF\xBF\xBD ASCII survives") != std::string::npos);
    CHECK(file.find("Private Folder") == std::string::npos);
    CHECK(file.find("two words") == std::string::npos);
    CHECK(file.find('\x1b') == std::string::npos);
    CHECK(file.find("debug record should not appear") == std::string::npos);
    CHECK(file.find("<Error>") != std::string::npos);
    CHECK(file.find("last queued record") != std::string::npos);
    CHECK(std::ranges::count(file, '\n') == 260);
}
