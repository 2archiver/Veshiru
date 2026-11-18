// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Common::ArtworkCache {

/// Normalizes a Title ID to its base application Program ID (clears update/AOC low bits).
[[nodiscard]] constexpr u64 BaseProgramId(u64 title_id) noexcept {
    return title_id & ~0x1FFFULL;
}

/// Returns the canonical game title for a known Switch 1 or Switch 2 Program ID, or empty if unknown.
[[nodiscard]] std::string_view LookupTitleByProgramId(u64 title_id) noexcept;

/// Normalizes a game title into a lowercase ASCII hyphenated slug for fuzzy comparison.
[[nodiscard]] std::string NormalizeSlug(std::string_view title);

/// Strips dump noise (hex Title IDs, brackets, region/version/rev tags, underscores/dots) and
/// expands common abbreviations into a clean online search query.
[[nodiscard]] std::string CleanGameSearchQuery(std::string_view raw_name);

/// Resolves the highest-confidence canonical search title, preferring exact Title ID matches when
/// available and falling back to cleaned aliases/filenames.
[[nodiscard]] std::string ResolveCanonicalSearchTitle(std::string_view raw_name, u64 title_id);

/// Returns a trimmed subtitle after ':' when a title has a distinctive subtitle (>= 4 non-space
/// characters), or empty otherwise. Never returns the bare series prefix before ':'.
[[nodiscard]] std::string ExtractSubtitleQuery(std::string_view title);

/// Builds Libretro/No-Intro thumbnail base name variants (without region or .png suffix),
/// handling colon-to-dash conversion and trailing ", The" normalization.
[[nodiscard]] std::vector<std::string> BuildLibretroNameVariants(std::string_view title);

/// Scores an online search candidate title against a target query, penalizing demos, DLCs,
/// soundtracks, wrong editions, wrong series entries, and sequel number mismatches.
[[nodiscard]] int ScoreArtworkCandidate(std::string_view candidate, std::string_view target_query);

/// Validates a display/search title without accepting paths, markup or control characters.
[[nodiscard]] bool IsSafeArtworkTitle(std::string_view title) noexcept;

/// Checks the size, dimensions and chunk bounds of a cached PNG (not a full decode).
[[nodiscard]] bool IsValidImageBytes(std::span<const u8> bytes) noexcept;

/// Sidecar paths inside <CacheDir>/artwork/ (never modify game containers).
[[nodiscard]] std::filesystem::path ArtworkDirectory();
[[nodiscard]] std::filesystem::path ArtworkPngPath(u64 title_id);
[[nodiscard]] std::filesystem::path ArtworkMetaPath(u64 title_id);

/// Reads a validated cached HD icon from <CacheDir>/artwork/<BASE_TID>.png, or empty if absent.
[[nodiscard]] std::vector<u8> ReadCachedArtwork(u64 title_id);

/// Reads the resolved canonical title from the sidecar JSON (or exact Program ID table), if any.
[[nodiscard]] std::string ReadSidecarTitle(u64 title_id);

/// Replaces a validated PNG by rename. JSON metadata is stored separately and is best effort.
bool WriteCachedArtwork(u64 title_id, std::span<const u8> png_bytes,
                        std::string_view resolved_title, std::string_view source);

/// Removes a title's cached PNG and sidecar JSON.
void ClearCachedArtwork(u64 title_id);

} // namespace Common::ArtworkCache
