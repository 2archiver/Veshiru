// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "common/common_types.h"

namespace Common::DiagRequest {

/// Saves `rgba` (width * height * 4 bytes, first row at the top, tightly packed) as a PNG file.
/// Returns false (and logs) when the file cannot be written.
bool WritePng(const std::filesystem::path& path, u32 width, u32 height, std::span<const u8> rgba);

/// Writes `text` to `path`, replacing it. Returns false (and logs) on failure.
bool WriteTextFile(const std::filesystem::path& path, std::string_view text);

/// `text` as the body of a JSON string literal (quotes, backslashes and control characters escaped).
[[nodiscard]] std::string JsonEscape(std::string_view text);

} // namespace Common::DiagRequest
