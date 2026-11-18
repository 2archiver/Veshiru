// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

#include "common/fs/path_util.h"
#include "common/logging.h"

int main(int argc, char* argv[]) {
    using namespace Common;
    namespace fs = std::filesystem;
    if (argc != 2) {
        return 1;
    }
    const auto directory = fs::current_path();
    FS::SetEdenPath(FS::EdenPath::LogDir, directory);
    const auto file = directory / "mirror.log";
    const std::string_view action{argv[1]};
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
