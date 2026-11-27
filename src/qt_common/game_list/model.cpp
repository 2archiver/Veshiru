// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <vector>

#include <QDir>
#include <QIcon>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThreadPool>

#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/qt_common.h"
#include "qt_common/util/boxart.h"
#include "qt_common/util/game.h"

#include "qt_common/game_list/game_list_p.h"
#include "qt_common/game_list/model.h"
#include "qt_common/game_list/worker.h"

GameListModel::GameListModel(std::shared_ptr<FileSys::VfsFilesystem> vfs_,
                             FileSys::ManualContentProvider* provider_,
                             const PlayTime::PlayTimeManager& play_time_manager_,
                             Core::System& system_, QObject* parent)
    : QStandardItemModel{parent}, vfs{std::move(vfs_)}, provider{provider_},
      play_time_manager{play_time_manager_}, system{system_} {
    watcher = new QFileSystemWatcher(this);
    external_watcher = new QFileSystemWatcher(this);

    refresh_timer.setSingleShot(true);
    refresh_timer.setInterval(30);
    connect(&refresh_timer, &QTimer::timeout, this, [this] {
        // Both kinds of event can land inside one debounce window: run whichever were
        // requested, each exactly once, with a single repopulate at the end.
        const auto requested = pending_refreshes.Take();
        if (!requested.external && !requested.game_dir) {
            return;
        }
        if (requested.external) {
            LOG_INFO(Frontend, "External content directory changed. Clearing metadata cache.");
            QtCommon::Game::ResetMetadata(false);
        }
        if (requested.game_dir) {
            ResetExternalWatcher();
            LOG_INFO(Frontend, "Change detected in the games directory. Reloading game list.");
        }
        if (!UISettings::values.game_dirs.empty() && current_worker != nullptr) {
            Repopulate();
        }
    });
    connect(watcher, &QFileSystemWatcher::directoryChanged, this, [this] {
        pending_refreshes.OnGameDir();
        refresh_timer.start();
    });
    connect(external_watcher, &QFileSystemWatcher::directoryChanged, this, [this] {
        pending_refreshes.OnExternal();
        refresh_timer.start();
    });

    ResetExternalWatcher();

    insertColumns(0, COLUMN_COUNT);
    RetranslateUI();

    setSortRole(GameListItemPath::SortRole);

    QtCommon::Boxart::OnSaved(
        this, [this](const std::string& slug) { ShowCover(QString::fromStdString(slug)); });
}

GameListModel::~GameListModel() {
    QtCommon::Boxart::OnSaved(this, {});
    StopWorker();
}

void GameListModel::ShowCover(const QString& slug) {
    std::vector<u8> cover;
    // Games sit under their directory and in favorites, or at the top of the flat list.
    const auto update = [&](QStandardItem* game) {
        if (game == nullptr || game->data(GameListItemPath::CoverSlugRole).toString() != slug) {
            return;
        }
        if (cover.empty()) {
            const u64 program_id = game->data(GameListItemPath::ProgramIdRole).toULongLong();
            cover = QtCommon::Boxart::FindCover(program_id, slug.toStdString());
            if (cover.empty()) {
                return;
            }
        }
        const QByteArray bytes(reinterpret_cast<const char*>(cover.data()),
                               static_cast<qsizetype>(cover.size()));
        // Every game-list item derives GameListItem, and only GameListItemPath holds a cover slug
        // (checked above), so this cast is exact without RTTI, which the build disables.
        static_cast<GameListItem*>(game)->ApplyCover(bytes);
    };
    for (int i = 0; i < rowCount(); ++i) {
        auto* top = item(i);
        update(top);
        for (int j = 0; j < top->rowCount(); ++j) {
            update(top->child(j));
        }
    }
}

void GameListModel::SetScanEnabled(bool enabled) {
    scan_enabled = enabled;
    if (!enabled) {
        StopWorker();
    }
}

void GameListModel::PopulateAsync(QVector<UISettings::GameDir>& game_dirs,
                                  GameListRefreshReason reason) {
    if (!scan_enabled || system.IsPoweredOn()) {
        return;
    }
    emit PopulatingStarted();

    StopWorker();
    removeRows(0, rowCount());

    current_worker = std::make_unique<GameListWorker>(vfs, provider, game_dirs, play_time_manager,
                                                      system, reason);

    connect(current_worker.get(), &GameListWorker::DataAvailable, this, &GameListModel::WorkerEvent,
            Qt::QueuedConnection);

    QThreadPool::globalInstance()->start(current_worker.get());
}

void GameListModel::WorkerEvent() {
    if (current_worker) {
        current_worker->ProcessEvents(this);
    }
}

void GameListModel::AddDirEntry(GameListDir* entry_items) {
    if (m_flat) {
        delete entry_items;
        return;
    }
    invisibleRootItem()->appendRow(entry_items);
}

void GameListModel::AddEntry(const QList<QStandardItem*>& entry_items, GameListDir* parent) {
    if (m_flat) {
        invisibleRootItem()->appendRow(entry_items);
    } else {
        parent->appendRow(entry_items);
    }
}

void GameListModel::DonePopulating(const QStringList& watch_list) {
    emit ShowList(!IsEmpty());

    if (!m_flat) {
        invisibleRootItem()->appendRow(new GameListAddDir());
        invisibleRootItem()->insertRow(0, new GameListFavorites());

        for (const auto id : std::as_const(UISettings::values.favorited_ids)) {
            AddFavorite(id);
        }
    }

    emit PopulatingCompleted(watch_list);
}

bool GameListModel::IsEmpty() const {
    for (int i = 0; i < rowCount(); i++) {
        const QStandardItem* child = invisibleRootItem()->child(i);
        const auto type = static_cast<GameListItemType>(child->type());

        if (!child->hasChildren() &&
            (type == GameListItemType::SdmcDir || type == GameListItemType::UserNandDir ||
             type == GameListItemType::SysNandDir)) {
            invisibleRootItem()->removeRow(child->row());
            i--;
        }
    }

    return !invisibleRootItem()->hasChildren();
}

void GameListModel::ToggleFavorite(u64 program_id) {
    if (!UISettings::values.favorited_ids.contains(program_id)) {
        UISettings::values.favorited_ids.append(program_id);
        AddFavorite(program_id);
    } else {
        UISettings::values.favorited_ids.removeOne(program_id);
        RemoveFavorite(program_id);
    }
    emit SaveConfig();
}

void GameListModel::AddFavorite(u64 program_id) {
    auto* favorites_row = item(0);

    for (int i = 1; i < rowCount() - 1; i++) {
        const auto* folder = item(i);
        for (int j = 0; j < folder->rowCount(); j++) {
            if (folder->child(j)->data(GameListItemPath::ProgramIdRole).toULongLong() ==
                program_id) {
                QList<QStandardItem*> list;
                for (int k = 0; k < COLUMN_COUNT; k++) {
                    list.append(folder->child(j, k)->clone());
                }
                list[0]->setData(folder->child(j)->data(GameListItem::SortRole),
                                 GameListItem::SortRole);
                list[0]->setText(folder->child(j)->data(Qt::DisplayRole).toString());

                favorites_row->appendRow(list);
                return;
            }
        }
    }
}

void GameListModel::RemoveFavorite(u64 program_id) {
    auto* favorites_row = item(0);

    for (int i = 0; i < favorites_row->rowCount(); i++) {
        const auto* game = favorites_row->child(i);
        if (game->data(GameListItemPath::ProgramIdRole).toULongLong() == program_id) {
            favorites_row->removeRow(i);
            return;
        }
    }
}

void GameListModel::StopWorker() {
    if (!current_worker) {
        return;
    }

    // The worker may still be inside ProcessEvents further up the stack (a nested event loop
    // can trigger a repopulate), so stop it now but defer freeing it to the event loop.
    current_worker->disconnect();
    current_worker->Stop();
    current_worker.release()->deleteLater();
}

void GameListModel::Repopulate() {
    // CreateFactories replaces the content providers a running game reads from. Stopping
    // emulation repopulates the list anyway, so skip watcher-driven reloads until then.
    if (!scan_enabled || QtCommon::system->IsPoweredOn()) {
        LOG_INFO(Frontend, "Emulation is running, deferring game list reload.");
        return;
    }

    StopWorker();
    QtCommon::system->GetFileSystemController().CreateFactories(*QtCommon::vfs);
    PopulateAsync(UISettings::values.game_dirs, GameListRefreshReason::Watcher);
}

void GameListModel::RefreshGameDirectory() {
    ResetExternalWatcher();
    if (!UISettings::values.game_dirs.empty() && current_worker != nullptr) {
        LOG_INFO(Frontend, "Change detected in the games directory. Reloading game list.");
        Repopulate();
    }
}

void GameListModel::RefreshExternalContent() {
    if (!UISettings::values.game_dirs.empty() && current_worker != nullptr) {
        LOG_INFO(Frontend, "External content directory changed. Clearing metadata cache.");
        QtCommon::Game::ResetMetadata(false);
        Repopulate();
    }
}

void GameListModel::ResetExternalWatcher() {
    auto watch_dirs = external_watcher->directories();
    if (!watch_dirs.isEmpty()) {
        external_watcher->removePaths(watch_dirs);
    }

    for (const std::string& dir : Settings::values.external_content_dirs) {
        external_watcher->addPath(QString::fromStdString(dir));
    }
}

void GameListModel::RetranslateUI() {
    setHeaderData(COLUMN_NAME, Qt::Horizontal, tr("Name"));
    setHeaderData(COLUMN_COMPAT, Qt::Horizontal, tr("Compatibility"));
    setHeaderData(COLUMN_ADD_ONS, Qt::Horizontal, tr("Add-ons"));
    setHeaderData(COLUMN_FILE_TYPE, Qt::Horizontal, tr("File type"));
    setHeaderData(COLUMN_SIZE, Qt::Horizontal, tr("Size"));
    setHeaderData(COLUMN_PLAY_TIME, Qt::Horizontal, tr("Play time"));
}

QFileSystemWatcher* GameListModel::GetWatcher() const {
    return watcher;
}

void GameListModel::SetFlat(bool flat) {
    m_flat = flat;
}
