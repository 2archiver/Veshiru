// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

// The incremental game-list refresh: a session-scoped, in-memory cache of parsed rows.
//
// A library refresh re-parses every dump: the loader walks the container, the icon is read
// from the control data, the title and patch versions are resolved. Files that did not
// change since the last scan parse to the same row, so the worker caches the extracted row
// keyed by canonical path + file size + modification time. Unchanged files reuse the cached
// row; only new or modified files pay the full parse. Play time is looked up fresh when a
// cached row is rebuilt, because it changes while the file does not.
//
// What the cache deliberately does not freshness-bypass: patch versions come from the
// on-disk metadata cache (pv.txt) exactly as before, and the manual content provider is
// still refilled on every scan, so boot support never depends on the cache.
//
// Invalidation: the cache lives for the session and is cleared when decryption keys or
// firmware are installed and wherever the on-disk metadata cache is cleared, because a
// re-parsed file can then produce a different row. It is never written to disk.

/// Why a game-list refresh was started; reported in the refresh log line. Global like the
/// GameListWorker / GameListModel types it is passed between (the view class named GameList
/// would shadow a namespace-qualified spelling inside its own members).
enum class GameListRefreshReason {
    Startup,  ///< the initial scan after the window opens
    Settings, ///< a settings, tool or emulation-lifecycle change asked for a rescan
    Watcher,  ///< a filesystem watcher noticed the library itself changed
};

[[nodiscard]] constexpr std::string_view ToString(GameListRefreshReason reason) {
    switch (reason) {
    case GameListRefreshReason::Startup:
        return "startup";
    case GameListRefreshReason::Settings:
        return "settings";
    case GameListRefreshReason::Watcher:
        return "watcher";
    }
    return "unknown";
}

namespace GameList {
namespace ParseCache {

/// Everything needed to rebuild a game-list row without touching the file again. Plain
/// values only, so the cache stays Qt-free and trivially destructible.
struct EntryData {
    std::string path;             ///< physical name, already formatted for display
    std::string name;             ///< resolved title
    std::vector<u8> icon;         ///< icon bytes as ReadIcon produced them
    std::string file_type_string; ///< Loader::GetFileTypeString result
    u64 program_id{};
    u64 size{};                   ///< row size (0 for loose dumps, which show "Not scanned")
    std::string patch_versions;   ///< newline-joined patch names, as the UI expects
    std::string game_version;     ///< GameListVersion::Resolve result
    bool loose_dump{};            ///< an extracted NCA main, not a container
};

/// The session cache. Find and Store are called from the worker thread only; the game-list
/// model serializes scans (a new worker only starts after the previous one stopped), and
/// the map plus its entries are plain data. Kept without Qt so it can be unit-tested and
/// invalidated from non-Qt code.
class Cache {
public:
    /// Returns the cached row data when `path` was parsed before with the same size and
    /// modification time, and nullptr otherwise.
    [[nodiscard]] const EntryData* Find(const std::string& path, u64 size, s64 mtime_ns) {
        const auto it = entries.find(path);
        if (it == entries.end() || it->second.size != size || it->second.mtime_ns != mtime_ns) {
            return nullptr;
        }
        return &it->second.data;
    }

    /// Caches the parsed row data for `path` under the given fingerprint.
    void Store(EntryData data, u64 size, s64 mtime_ns) {
        Slot slot{size, mtime_ns, std::move(data)};
        entries.insert_or_assign(std::string{slot.data.path}, std::move(slot));
    }

    void Clear() {
        entries.clear();
    }

    [[nodiscard]] std::size_t Size() const {
        return entries.size();
    }

private:
    struct Slot {
        u64 size;
        s64 mtime_ns;
        EntryData data;
    };
    std::unordered_map<std::string, Slot> entries;
};

/// The session-wide cache instance. Destroyed at process exit.
[[nodiscard]] inline Cache& Global() {
    static Cache cache;
    return cache;
}

/// Clears the session cache. Called when decryption keys or firmware are installed and
/// wherever the on-disk metadata cache is cleared: the next scan must re-parse everything.
inline void Invalidate() {
    Global().Clear();
}

} // namespace ParseCache
} // namespace GameList
