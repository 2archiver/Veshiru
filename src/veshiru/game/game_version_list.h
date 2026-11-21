// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QListView>

class GameListModel;
class QResizeEvent;

class GameVersionList final : public QListView {
    Q_OBJECT

public:
    explicit GameVersionList(QWidget* parent = nullptr);

    void SetModel(GameListModel* model);
    void ApplyFilter(const QString& text, GameListModel* model);
    void UpdateLayout();

protected:
    void resizeEvent(QResizeEvent* event) override;
};
