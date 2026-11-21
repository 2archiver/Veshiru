// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "veshiru/game/game_version_list.h"

#include <algorithm>

#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QStyledItemDelegate>

#include "qt_common/game_list/game_list_p.h"
#include "qt_common/game_list/model.h"
#include "veshiru/game/common.h"

namespace {
class VersionListDelegate final : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        if (index.data(GameListItem::TypeRole).value<GameListItemType>() !=
            GameListItemType::Game) {
            QStyledItemDelegate::paint(painter, option, index);
            return;
        }

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing);
        const QRect row = option.rect.adjusted(6, 3, -6, -3);
        const bool selected = option.state & QStyle::State_Selected;
        const bool hovered = option.state & QStyle::State_MouseOver;
        QColor background = option.palette.base().color();
        if (selected) {
            background = option.palette.highlight().color();
        } else if (hovered) {
            background = option.palette.alternateBase().color();
        }
        painter->setPen(Qt::NoPen);
        painter->setBrush(background);
        painter->drawRoundedRect(row, 9, 9);

        constexpr int icon_extent = 58;
        const QRect icon_rect(row.left() + 12, row.center().y() - icon_extent / 2,
                              icon_extent, icon_extent);
        const QPixmap icon = index.data(Qt::DecorationRole).value<QPixmap>();
        if (!icon.isNull()) {
            QPainterPath clip;
            clip.addRoundedRect(icon_rect, 7, 7);
            painter->save();
            painter->setClipPath(clip);
            painter->setRenderHint(QPainter::SmoothPixmapTransform);
            painter->drawPixmap(icon_rect, icon);
            painter->restore();
        }

        const int text_left = icon_rect.right() + 16;
        const int text_right = row.right() - 130;
        const int text_width = std::max(0, text_right - text_left);
        const QRect text_rect(text_left, row.top() + 10, text_width, row.height() - 20);
        const QString title = index.data(GameListItemPath::TitleRole).toString().trimmed();
        const QString fallback = index.data(Qt::DisplayRole).toString();
        const QString shown_title = title.isEmpty() ? fallback : title;

        QFont title_font = option.font;
        title_font.setWeight(QFont::DemiBold);
        title_font.setPointSizeF(title_font.pointSizeF() + 1.0);
        painter->setFont(title_font);
        painter->setPen(selected ? option.palette.highlightedText().color()
                                 : option.palette.text().color());
        painter->drawText(QRect(text_rect.left(), text_rect.top(), text_rect.width(), 25),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          painter->fontMetrics().elidedText(shown_title, Qt::ElideRight,
                                                            text_rect.width()));

        const QString type = index.data(GameListItemPath::FileTypeRole).toString();
        const QString play_time = index.siblingAtColumn(GameListModel::COLUMN_PLAY_TIME)
                                      .data(Qt::DisplayRole).toString();
        const QString subtitle = play_time.isEmpty() ? type : QStringLiteral("%1  ·  %2")
                                                               .arg(type, play_time);
        QFont subtitle_font = option.font;
        subtitle_font.setPointSizeF(std::max(8.0, subtitle_font.pointSizeF() - 1.0));
        painter->setFont(subtitle_font);
        painter->setPen(selected ? option.palette.highlightedText().color().lighter(150)
                                 : option.palette.mid().color());
        painter->drawText(QRect(text_rect.left(), text_rect.top() + 27, text_rect.width(), 19),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          painter->fontMetrics().elidedText(subtitle, Qt::ElideRight,
                                                            text_rect.width()));

        const QString version = index.data(GameListItemPath::VersionRole).toString();
        const QString version_label = QObject::tr("Version %1").arg(version);
        const QFontMetrics badge_metrics(option.font);
        const int badge_width = badge_metrics.horizontalAdvance(version_label) + 24;
        const QRect badge(row.right() - badge_width - 14, row.center().y() - 14,
                          badge_width, 28);
        QColor badge_color = selected ? option.palette.highlightedText().color()
                                      : option.palette.highlight().color();
        badge_color.setAlpha(selected ? 35 : 28);
        painter->setPen(Qt::NoPen);
        painter->setBrush(badge_color);
        painter->drawRoundedRect(badge, 14, 14);
        painter->setFont(option.font);
        painter->setPen(selected ? option.palette.highlightedText().color()
                                 : option.palette.highlight().color());
        painter->drawText(badge, Qt::AlignCenter, version_label);
        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem&, const QModelIndex&) const override {
        return {300, 88};
    }
};
} // namespace

GameVersionList::GameVersionList(QWidget* parent) : QListView{parent} {
    setItemDelegate(new VersionListDelegate(this));
    setViewMode(QListView::ListMode);
    setFlow(QListView::TopToBottom);
    setWrapping(false);
    setResizeMode(QListView::Adjust);
    setUniformItemSizes(true);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setEditTriggers(QAbstractItemView::NoEditTriggers);
    setContextMenuPolicy(Qt::CustomContextMenu);
    setSpacing(0);
    setStyleSheet(QStringLiteral("QListView{border:none;background:transparent;}"));
}

void GameVersionList::SetModel(GameListModel* model) {
    QListView::setModel(model);
    UpdateLayout();
}

void GameVersionList::ApplyFilter(const QString& text, GameListModel* model) {
    for (int row = 0; row < model->rowCount(); ++row) {
        const auto* item = model->item(row, 0);
        setRowHidden(row, !item || !Veshiru::FilterMatches(text, item));
    }
}

void GameVersionList::UpdateLayout() {
    const QSize size(std::max(300, viewport()->width()), 92);
    if (gridSize() != size) {
        setGridSize(size);
    }
}

void GameVersionList::resizeEvent(QResizeEvent* event) {
    QListView::resizeEvent(event);
    UpdateLayout();
}
