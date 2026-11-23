// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <string_view>

namespace Tests {

/// A path in the system temp directory that no other process and no other call in this process
/// uses: "<stem>-<pid>-<sequence><extension>". Several tests.exe run at the same time on one
/// machine (and share %TEMP%), so a fixed temp name lets one run overwrite, delete or lock another's
/// file. Nothing is created at the path.
[[nodiscard]] std::filesystem::path UniqueTempPath(std::string_view stem,
                                                   std::string_view extension = {});

/// A fresh, empty directory at UniqueTempPath(stem), removed with its contents when this goes out
/// of scope.
class TempDir {
public:
    explicit TempDir(std::string_view stem);
    ~TempDir();

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& Path() const {
        return path;
    }

private:
    std::filesystem::path path;
};

} // namespace Tests
