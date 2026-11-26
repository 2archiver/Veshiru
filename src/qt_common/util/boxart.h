// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common/common_types.h"

class QObject;

// Box art for games without an icon. This mirrors Satsuma's icon handling: a bundled icon
// (:/game_icons/<program ID>.png) wins over the game's own icon, then built-in box art
// (:/boxart/<program ID>.png), then a cover downloaded once from Nemyu's box art repository and
// Eden's, stored zstd-compressed in one pack file in the cache folder (cache/boxart.pack).
namespace QtCommon::Boxart {

/// The cover name the repositories use: the Switchbrew title table's (:/switch2_titles.json) cover
/// or name for a Switch 2 program, else made from `title`
/// ("Hyrule Warriors: Age of Imprisonment" -> "hyrule-warriors-age-of-imprisonment"). Empty when
/// none can be made.
[[nodiscard]] std::string GetSlug(u64 program_id, const std::string& title);

/// The game's cover image: the built-in one, else one downloaded before. Empty when there is none.
[[nodiscard]] std::vector<u8> FindCover(u64 program_id, const std::string& slug);

/// Downloads the cover in the background when it may exist (Switch 1 and Switch 2 games), at most
/// once a slug a session; listeners hear its slug once it is stored.
void Prefetch(u64 program_id, const std::string& slug);

/// `on_saved` runs on `context`'s thread with the slug of each cover stored; an empty function
/// removes the listener.
void OnSaved(QObject* context, std::function<void(const std::string& slug)> on_saved);

/// Stops the download thread (application exit).
void Shutdown();

} // namespace QtCommon::Boxart
