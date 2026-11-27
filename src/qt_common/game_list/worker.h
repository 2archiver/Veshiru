// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <deque>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <QList>
#include <QObject>
#include <QRunnable>
#include <QString>

#include "common/thread.h"
#include "core/file_sys/registered_cache.h"
#include "frontend_common/play_time_manager.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/game_list/parse_cache.h"

namespace Core {
class System;
}

class GameListDir;
class GameListModel;
class QStandardItem;
enum class GameListItemType;

namespace FileSys {
class NCA;
class VfsFilesystem;
} // namespace FileSys

/**
 * Asynchronous worker object for populating the game list.
 * Communicates with other threads through Qt's signal/slot system.
 */
class GameListWorker : public QObject, public QRunnable {
    Q_OBJECT

public:
    explicit GameListWorker(std::shared_ptr<FileSys::VfsFilesystem> vfs_,
                            FileSys::ManualContentProvider* provider_,
                            QVector<UISettings::GameDir>& game_dirs_,
                            const PlayTime::PlayTimeManager& play_time_manager_,
                            Core::System& system_, GameListRefreshReason reason_);
    ~GameListWorker() override;

    /// Starts the processing of directory tree information.
    void run() override;

    /// Requests the worker to stop and blocks until processing has finished.
    void Stop();

    /// Builds the game-list row for entry data extracted earlier (possibly from the parse
    /// cache). `play_time` is looked up by the caller so a reused row stays current.
    static QList<QStandardItem*> BuildRow(const GameListParseCache::EntryData& data, u64 play_time);

public:
    /**
     * Synchronously processes any events queued by the worker.
     *
     * AddDirEntry is called on the model for every discovered directory.
     * AddEntry is called on the model for every discovered program.
     * DonePopulating is called on the model when processing completes.
     */
    void ProcessEvents(GameListModel* model);

signals:
    void DataAvailable();

private:
    template <typename F>
    void RecordEvent(F&& func);

private:
    void AddTitlesToGameList(GameListDir* parent_dir, GameListItemType parent_type);

    enum class ScanTarget {
        FillManualContentProvider,
        PopulateGameList,
    };

    void ScanFileSystem(ScanTarget target, const std::vector<std::filesystem::path>& files,
                        GameListDir* parent_dir,
                        FileSys::ManualContentProvider* target_provider = nullptr);

    /// Lists a row for every game the previous scan listed whose file is now gone, and forgets it.
    /// Without this a title that is moved or deleted simply stops appearing, which looks like the
    /// emulator losing the game: the player's own MKW .nup report.
    void ReportMissingGames(GameListDir* parent_dir);

    std::shared_ptr<FileSys::VfsFilesystem> vfs;
    FileSys::ManualContentProvider* provider;
    QVector<UISettings::GameDir>& game_dirs;
    const PlayTime::PlayTimeManager& play_time_manager;

    QStringList watch_list;

    /// The game files listed by this scan, as UTF-8 paths: compared with
    /// UISettings::values.listed_games (the previous scan) by ReportMissingGames.
    std::set<std::string> listed_this_scan;

    std::mutex lock;
    std::condition_variable cv;
    std::deque<std::function<void(GameListModel*)>> queued_events;
    std::atomic_bool stop_requested = false;
    Common::Event processing_completed;

    /// Refresh statistics for the summary log line: rows reused from the parse cache,
    /// rows parsed from scratch, and why this refresh ran.
    std::size_t reused_entries = 0;
    std::size_t parsed_entries = 0;

    Core::System& system;

    /// Why this refresh ran; declared after system to match the constructor's init list.
    GameListRefreshReason reason;
};
