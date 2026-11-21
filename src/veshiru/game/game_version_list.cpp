// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "veshiru/game/game_version_list.h"

#include <algorithm>
#include <cmath>

#include <QColor>
#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QResizeEvent>
#include <QStringList>
#include <QStyle>
#include <QStyledItemDelegate>

#include "qt_common/game_list/game_list_p.h"
#include "qt_common/game_list/model.h"
#include "veshiru/game/common.h"

namespace {

// The 2.3 view painted its rows from QPalette::Base, which on the default Veshiru theme is the
// near-black #07111F app background: inside an already dark blue window that read as a black
// panel. This view takes every colour from the active palette instead and only derives shades
// from it, so it works on Deep Current and on the light/dark/custom themes without any raw
// background or text hex value.
constexpr int ROW_HEIGHT = 92;
constexpr int ROW_INSET_X = 8;
constexpr int ROW_INSET_Y = 4;
constexpr int ICON_SIZE = 60;
constexpr int ICON_LEFT = 14;
constexpr int ICON_GAP = 18;
constexpr int BADGE_HEIGHT = 26;
constexpr int BADGE_RIGHT = 16;
constexpr double AA_CONTRAST = 4.5;

double LinearChannel(int channel) {
    const double value = channel / 255.0;
    return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
}

double Luminance(const QColor& color) {
    return 0.2126 * LinearChannel(color.red()) + 0.7152 * LinearChannel(color.green()) +
           0.0722 * LinearChannel(color.blue());
}

double ContrastRatio(const QColor& first, const QColor& second) {
    const double a = Luminance(first);
    const double b = Luminance(second);
    return (std::max(a, b) + 0.05) / (std::min(a, b) + 0.05);
}

/// `foreground` over `background` with its alpha applied, returned opaque.
QColor Composite(const QColor& foreground, const QColor& background) {
    const double alpha = foreground.alphaF();
    return QColor::fromRgbF(foreground.redF() * alpha + background.redF() * (1.0 - alpha),
                            foreground.greenF() * alpha + background.greenF() * (1.0 - alpha),
                            foreground.blueF() * alpha + background.blueF() * (1.0 - alpha));
}

/// Text on a themed surface must stay readable in every theme, including a user theme whose
/// accent is close to its surface. Move the foreground towards the opposite end of the palette
/// until it passes, instead of hardcoding a colour.
QColor EnsureContrast(QColor foreground, const QColor& background, double minimum) {
    const bool lighter_than_background = Luminance(foreground) >= Luminance(background);
    for (int step = 0; step < 24 && ContrastRatio(foreground, background) < minimum; ++step) {
        foreground = lighter_than_background ? foreground.lighter(114) : foreground.darker(114);
    }
    return foreground;
}

/// A row surface that is visibly a shade of the theme's own background; never QPalette::Base,
/// which is the app window colour and produced the "black panel" look.
QColor SurfaceColor(const QPalette& palette) {
    const QColor window = palette.color(QPalette::Window);
    // Mid is the theme's own elevated shade: #132D46 on Deep Current, a visibly lighter blue
    // than the #07111F window rather than the near-black Base the old view used.
    QColor surface = palette.color(QPalette::Mid);
    if (ContrastRatio(surface, window) < 1.15) {
        surface = palette.color(QPalette::AlternateBase);
    }
    if (ContrastRatio(surface, window) < 1.15) {
        surface = palette.color(QPalette::Button);
    }
    if (ContrastRatio(surface, window) < 1.15) {
        surface = Luminance(window) < 0.5 ? window.lighter(126) : window.darker(126);
    }
    return surface;
}

QFont ScaledFont(const QFont& base, double delta_points, QFont::Weight weight = QFont::Normal) {
    QFont font = base;
    if (font.pointSizeF() > 0.0) {
        font.setPointSizeF(std::max(8.0, font.pointSizeF() + delta_points));
    } else {
        font.setPixelSize(std::max(10, font.pixelSize() + static_cast<int>(std::lround(
                                                          delta_points * 4.0 / 3.0))));
    }
    font.setWeight(weight);
    return font;
}

QString TitleIdText(const QModelIndex& index) {
    const u64 program_id = index.data(GameListItemPath::ProgramIdRole).toULongLong();
    if (program_id == 0) {
        return {};
    }
    return QStringLiteral("%1").arg(program_id, 16, 16, QLatin1Char{'0'});
}

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

        const QPalette& palette = option.palette;
        const QColor surface = SurfaceColor(palette);
        const QColor accent = palette.color(QPalette::Highlight);
        const bool selected = option.state & QStyle::State_Selected;
        const bool hovered = option.state & QStyle::State_MouseOver;
        const bool missing = index.data(GameListItemPath::MissingRole).toBool();

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing);
        painter->setRenderHint(QPainter::SmoothPixmapTransform);

        const QRect row = option.rect.adjusted(ROW_INSET_X, ROW_INSET_Y, -ROW_INSET_X,
                                               -ROW_INSET_Y);

        QColor background = surface;
        if (selected) {
            background = Composite(QColor(accent.red(), accent.green(), accent.blue(), 62),
                                   surface);
        } else if (hovered) {
            background = Composite(QColor(accent.red(), accent.green(), accent.blue(), 30),
                                   surface);
        }

        painter->setPen(Qt::NoPen);
        painter->setBrush(background);
        painter->drawRoundedRect(row, 10, 10);

        if (selected || hovered) {
            // A 3 px accent bar is cheaper than a border and reads at a glance.
            const QRect bar(row.left(), row.top() + 10, 3, row.height() - 20);
            painter->setBrush(accent);
            painter->drawRoundedRect(bar, 2, 2);
        }

        const QRect icon_rect(row.left() + ICON_LEFT, row.center().y() - ICON_SIZE / 2,
                              ICON_SIZE, ICON_SIZE);
        const QPixmap icon = index.data(Qt::DecorationRole).value<QPixmap>();
        if (!icon.isNull()) {
            QPainterPath clip;
            clip.addRoundedRect(icon_rect, 9, 9);
            painter->save();
            painter->setClipPath(clip);
            // A missing file still has its cached art; fading it makes the row read as unavailable
            // without needing an extra colour outside the palette.
            painter->setOpacity(missing ? 0.55 : 1.0);
            painter->drawPixmap(icon_rect, icon);
            painter->restore();
        }

        const int text_left = icon_rect.right() + ICON_GAP;
        const QString version = index.data(GameListItemPath::VersionRole).toString().trimmed();
        const QString file_type =
            index.siblingAtColumn(GameListModel::COLUMN_FILE_TYPE).data(Qt::DisplayRole).toString();
        const QString play_time =
            index.siblingAtColumn(GameListModel::COLUMN_PLAY_TIME).data(Qt::DisplayRole).toString();
        const QString size =
            index.siblingAtColumn(GameListModel::COLUMN_SIZE).data(Qt::DisplayRole).toString();
        const QString add_ons =
            index.siblingAtColumn(GameListModel::COLUMN_ADD_ONS).data(Qt::DisplayRole).toString();

        QFont badge_font = ScaledFont(option.font, -1.0, QFont::DemiBold);
        const QFontMetrics badge_metrics(badge_font);

        // Badge text: the version on a normal row, the worker's own localised file type on a
        // missing row. Missing rows keep the old path in their model tooltip.
        const QString badge_text =
            missing ? file_type : (version.isEmpty() ? QString{} : QStringLiteral("v%1").arg(version));
        int badge_width = 0;
        if (!badge_text.isEmpty()) {
            badge_width = badge_metrics.horizontalAdvance(badge_text) + 22;
        }

        const bool reserve_time = badge_width == 0 && !missing && !play_time.isEmpty();
        const int reserved_right = badge_width > 0 ? badge_width + 16
                                  : reserve_time ? 150
                                                 : 0;
        const int text_right = row.right() - BADGE_RIGHT - reserved_right;
        const int text_width = std::max(0, text_right - text_left);

        const QFont title_font = ScaledFont(option.font, 2.0, QFont::DemiBold);
        const QFont subtitle_font = ScaledFont(option.font, -1.0);
        const QFontMetrics title_metrics(title_font);
        const QFontMetrics subtitle_metrics(subtitle_font);

        const int title_height = title_metrics.height();
        const int subtitle_height = subtitle_metrics.height();
        const int block_height = title_height + 3 + subtitle_height;
        const int block_top = row.center().y() - block_height / 2;

        QColor title_color = EnsureContrast(palette.color(QPalette::Text), background,
                                            AA_CONTRAST);
        QColor secondary_color = Composite(QColor(palette.color(QPalette::Text).red(),
                                                  palette.color(QPalette::Text).green(),
                                                  palette.color(QPalette::Text).blue(), 190),
                                           background);
        secondary_color = EnsureContrast(secondary_color, background, AA_CONTRAST);

        const QString title = index.data(GameListItemPath::TitleRole).toString().trimmed();
        const QString fallback = index.data(Qt::DisplayRole).toString();
        const QString shown_title = title.isEmpty() ? fallback : title;

        painter->setFont(title_font);
        painter->setPen(title_color);
        painter->drawText(QRect(text_left, block_top, text_width, title_height),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          title_metrics.elidedText(shown_title, Qt::ElideRight, text_width));

        // Title ID, content type and file size are all secondary information on one line.
        QStringList details;
        if (!missing) {
            const QString title_id = TitleIdText(index);
            if (!title_id.isEmpty()) {
                details << title_id;
            }
            if (!file_type.isEmpty()) {
                details << file_type;
            }
            if (!size.isEmpty()) {
                details << size;
            }
        } else {
            // The worker already localised this in the add-ons column.
            details << (add_ons.isEmpty() ? QStringLiteral("File not found") : add_ons);
        }

        painter->setFont(subtitle_font);
        painter->setPen(secondary_color);
        painter->drawText(QRect(text_left, block_top + title_height + 3, text_width,
                                subtitle_height),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          subtitle_metrics.elidedText(details.join(QString::fromUtf8("  \xC2\xB7  ")),
                                                      Qt::ElideRight, text_width));

        if (badge_width > 0) {
            const bool show_play_time = !missing && !play_time.isEmpty();
            const int badge_y = show_play_time ? row.center().y() - BADGE_HEIGHT - 3
                                               : row.center().y() - BADGE_HEIGHT / 2;
            const QRect badge(row.right() - BADGE_RIGHT - badge_width, badge_y, badge_width,
                              BADGE_HEIGHT);

            QColor badge_fill = Composite(QColor(accent.red(), accent.green(), accent.blue(), 62),
                                          background);
            QColor badge_text_color = EnsureContrast(accent, badge_fill, AA_CONTRAST);
            painter->setPen(Qt::NoPen);
            painter->setBrush(badge_fill);
            painter->drawRoundedRect(badge, BADGE_HEIGHT / 2, BADGE_HEIGHT / 2);
            painter->setFont(badge_font);
            painter->setPen(badge_text_color);
            painter->drawText(badge, Qt::AlignCenter, badge_text);

            if (show_play_time) {
                const int time_width = std::max(badge.width(), 90);
                const QRect time_rect(row.right() - BADGE_RIGHT - time_width, badge.bottom() + 5,
                                      time_width, subtitle_height);
                painter->setFont(subtitle_font);
                painter->setPen(secondary_color);
                painter->drawText(time_rect, Qt::AlignRight | Qt::AlignVCenter,
                                  subtitle_metrics.elidedText(play_time, Qt::ElideRight,
                                                              time_rect.width()));
            }
        } else if (!missing && !play_time.isEmpty()) {
            const QRect time_rect(row.right() - BADGE_RIGHT - 150, row.center().y() - 10, 150, 20);
            painter->setFont(subtitle_font);
            painter->setPen(secondary_color);
            painter->drawText(time_rect, Qt::AlignRight | Qt::AlignVCenter,
                              subtitle_metrics.elidedText(play_time, Qt::ElideRight,
                                                          time_rect.width()));
        }

        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem&, const QModelIndex&) const override {
        return {300, ROW_HEIGHT};
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
    setMouseTracking(true);
    // Hover only needs a repaint of the row under the cursor; no animation is involved.
    viewport()->setAttribute(Qt::WA_Hover, true);
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
    const QSize size(std::max(300, viewport()->width()), ROW_HEIGHT);
    if (gridSize() != size) {
        setGridSize(size);
    }
}

void GameVersionList::resizeEvent(QResizeEvent* event) {
    QListView::resizeEvent(event);
    UpdateLayout();
}
