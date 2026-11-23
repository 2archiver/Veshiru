// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <cstdint>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <catch2/catch_test_macros.hpp>

#include "tests/temp_path.h"

namespace Tests {

std::filesystem::path UniqueTempPath(std::string_view stem, std::string_view extension) {
    static std::atomic<std::uint64_t> sequence{0};
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    std::string name{stem};
    name += '-';
    name += std::to_string(pid);
    name += '-';
    name += std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    name += extension;
    return std::filesystem::temp_directory_path() / name;
}

TempDir::TempDir(std::string_view stem) : path{UniqueTempPath(stem)} {
    // A directory left behind by a crashed earlier process with the same pid is reused empty.
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
    REQUIRE(std::filesystem::create_directories(path));
}

TempDir::~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
}

} // namespace Tests
