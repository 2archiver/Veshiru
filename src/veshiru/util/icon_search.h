// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <QByteArray>
#include <QFutureWatcher>
#include <QObject>
#include <QPointer>
#include <QString>

#include "common/common_types.h"

class QWidget;
class QProgressDialog;

enum class IconSearchStatus {
    Found,
    Choices,
    NotFound,
    Timeout,
    Failed,
    Cancelled,
    RateLimited,
    BadApiKey
};

struct IconSearchResult {
    IconSearchStatus status{IconSearchStatus::Failed};
    std::vector<QByteArray> choices;
};

// Explicit user-triggered SteamGridDB lookup, kept off the GUI and library-discovery paths.
class IconSearch : public QObject {
    Q_OBJECT
public:
    explicit IconSearch(QWidget* parent);
    ~IconSearch() override;

    static constexpr bool IsOnlineAvailable() {
#ifdef VESHIRU_EXPERIMENTAL_ONLINE_ARTWORK
        return true;
#else
        return false;
#endif
    }

    void Start(const QString& name, u64 title_id, bool write_cache);
    bool IsRunning() const;

signals:
    void RunningChanged(bool running);
    void IconSelected(const QByteArray& png);

private:
    void SetRunning(bool active);
    void BeginSearch(std::string query);
    void OnFinished();
    void ChooseIcon(std::vector<QByteArray> choices);
    void UseIcon(QByteArray png);

    QWidget* parent_widget;
    QFutureWatcher<IconSearchResult> watcher;
    std::shared_ptr<std::atomic_bool> cancel_flag{std::make_shared<std::atomic_bool>(false)};
    QPointer<QProgressDialog> progress;
    u64 cache_title_id{};
    std::string session_key;
    bool running{};
};
