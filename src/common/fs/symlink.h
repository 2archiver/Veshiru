// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <optional>
namespace Common::FS {

bool CreateSymlink(std::filesystem::path from, std::filesystem::path to);
bool IsSymlink(const std::filesystem::path &path);
std::optional<std::filesystem::path> ReadSymlinkTarget(const std::filesystem::path& path);

} // namespace Common::FS
