// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <QByteArray>
#include <QFutureWatcher>
#include <QObject>
#include <QString>

#include "common/common_types.h"

class QWidget;

enum class IconSearchStatus {
    Found,
    Choices,
    NotFound,
    Timeout,
    NetworkError,
    Cancelled,
    NeedsApiKey,
    BadApiKey,
    InternalError,
};

struct IconSearchResult {
    IconSearchStatus status{IconSearchStatus::NotFound};
    std::vector<u8> icon_bytes;
    std::vector<std::vector<u8>> choices;
    std::string resolved_title;
    std::string source;
    std::string error;
};

// Looks up HD game artwork by exact Title ID first, then canonical game name across official CDN /
// switchiconshowdown.com / SteamGridDB (when an API key is configured). Interactive searches run
// under one 10-second deadline; silent background hydration uses FetchSilentAsync without modal UI
// or modifying any .nup / ROM container on disk.
class IconSearch : public QObject {
    Q_OBJECT
public:
    explicit IconSearch(QWidget* parent);
    ~IconSearch() override;

    void Start(const QString& name, u64 title_id = 0);
    bool IsRunning() const;

    /// Non-blocking silent background fetch for Game List missing-icon hydration. Writes validated
    /// HD PNG + JSON metadata into <CacheDir>/artwork/ and invokes `on_done` on `context`'s thread.
    static void FetchSilentAsync(
        u64 title_id, const QString& fallback_name, QObject* context,
        std::function<void(QByteArray png, QString resolved_title)> on_done);

signals:
    void RunningChanged(bool running);
    // A 256x256 or 512x512 HD PNG, and the canonical title when resolved (else empty).
    void IconSelected(const QByteArray& png, const QString& resolved_title);

private:
    void OnFinished();
    int ChooseIcon(const std::vector<std::vector<u8>>& choices);
    bool AskForApiKey(bool missing);

    QWidget* parent_widget;
    QFutureWatcher<IconSearchResult> watcher;
    std::shared_ptr<std::atomic_bool> cancel_flag{std::make_shared<std::atomic_bool>(false)};
    QString last_name;
    u64 last_title_id{};
};
