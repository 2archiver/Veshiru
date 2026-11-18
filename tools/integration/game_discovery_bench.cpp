// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string_view>
#include <vector>

#include "qt_common/game_list/discovery.h"

int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    using namespace std::literals;
    if (argc != 3 || (argv[2] != "shallow"sv && argv[2] != "deep"sv)) {
        std::cerr << "Usage: game_discovery_bench FIXTURE_DIR shallow|deep\n";
        return 2;
    }
    const std::filesystem::path root{argv[1]};
    if (!std::filesystem::is_directory(root)) {
        return 2;
    }
    const auto supported = [](const auto& path) { return path.extension() == ".nsp"; };
    const auto scan = [&] {
        return GameListDiscovery::Discover(root, argv[2] == "deep"sv, supported,
                                           [] { return false; });
    };
    const auto warmup = scan();
    auto expected_files = warmup.files;
    auto expected_dirs = warmup.directories;
    std::ranges::sort(expected_files);
    std::ranges::sort(expected_dirs);
    if (warmup.errors || expected_files.empty()) {
        return 1;
    }
    std::vector<double> samples;
    for (int i = 0; i < 30; ++i) {
        const auto start = Clock::now();
        auto result = scan();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        std::ranges::sort(result.files);
        std::ranges::sort(result.directories);
        if (result.errors || result.files != expected_files ||
            result.directories != expected_dirs) {
            return 1;
        }
    }
    std::ranges::sort(samples);
    std::cout << "files=" << expected_files.size() << " watch_dirs=" << expected_dirs.size()
              << " median_ms=" << (samples[14] + samples[15]) / 2.0 << " p95_ms=" << samples[28]
              << '\n';
}
