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

/// Scores an online search candidate title against a target query, penalizing demos, DLCs,
/// soundtracks, wrong editions, and sequel number mismatches.
[[nodiscard]] int ScoreArtworkCandidate(std::string_view candidate, std::string_view target_query);

/// Checks that raw image bytes carry a valid PNG (with sane IHDR dimensions) or JPEG header.
[[nodiscard]] bool IsValidImageBytes(std::span<const u8> bytes) noexcept;

/// Sidecar paths inside <CacheDir>/artwork/ (never modifies .nup or ROM containers on disk).
[[nodiscard]] std::filesystem::path ArtworkDirectory();
[[nodiscard]] std::filesystem::path ArtworkPngPath(u64 title_id);
[[nodiscard]] std::filesystem::path ArtworkMetaPath(u64 title_id);

/// Reads a validated cached HD icon from <CacheDir>/artwork/<BASE_TID>.png, or empty if absent.
[[nodiscard]] std::vector<u8> ReadCachedArtwork(u64 title_id);

/// Reads the resolved canonical title from the sidecar JSON (or exact Program ID table), if any.
[[nodiscard]] std::string ReadSidecarTitle(u64 title_id);

/// Atomically stores validated PNG artwork + JSON metadata in <CacheDir>/artwork/.
bool WriteCachedArtwork(u64 title_id, std::span<const u8> png_bytes,
                        std::string_view resolved_title, std::string_view source);

/// Records a failed lookup with a retry cooldown so offline/unknown titles do not spam requests.
void RecordArtworkFailure(u64 title_id, std::string_view reason, u64 cooldown_seconds = 3600);

/// Returns true when no valid cached icon exists and any previous failure cooldown has elapsed.
[[nodiscard]] bool ShouldAutoFetchArtwork(u64 title_id, u64 now_unix_seconds = 0);

} // namespace Common::ArtworkCache
