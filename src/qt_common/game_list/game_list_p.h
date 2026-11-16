// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2015 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include <QCoreApplication>
#include <QFileInfo>
#include <QGuiApplication>
#include <QImage>
#include <QObject>
#include <QPainter>
#include <QPainterPath>
#include <QRegularExpression>
#include <QStandardItem>
#include <QString>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/string_util.h"
#include "frontend_common/play_time_manager.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/qt_common.h"

enum class GameListItemType {
    Game = QStandardItem::UserType + 1,
    CustomDir = QStandardItem::UserType + 2,
    SdmcDir = QStandardItem::UserType + 3,
    UserNandDir = QStandardItem::UserType + 4,
    SysNandDir = QStandardItem::UserType + 5,
    AddDir = QStandardItem::UserType + 6,
    Favorites = QStandardItem::UserType + 7,
};

Q_DECLARE_METATYPE(GameListItemType);

/**
 * Gets the default icon (for games without valid title metadata)
 * @param size The desired width and height of the default icon.
 * @return QPixmap default icon
 */
static bool IsVeshiruTheme() {
    return UISettings::values.theme == "veshiru";
}

static qreal GameListIconDevicePixelRatio() {
    if (const auto* app = qGuiApp) {
        return (std::max)(qreal{1.0}, app->devicePixelRatio());
    }
    return 1.0;
}

static QPixmap GetDefaultIcon(u32 size) {
    const qreal dpr = GameListIconDevicePixelRatio();
    const int side = (std::max)(1, static_cast<int>(size * dpr + 0.5));
    if (size == 0 || !IsVeshiruTheme()) {
        QPixmap icon(side, side);
        icon.setDevicePixelRatio(dpr);
        icon.fill(Qt::transparent);
        return icon;
    }
    // Veshiru Blue draws a game without art as a crisp HiDPI tile with a controller glyph.
    QImage icon(side, side, QImage::Format_ARGB32_Premultiplied);
    icon.fill(Qt::transparent);
    QPainter painter(&icon);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(0x23, 0x24, 0x2a));
    painter.drawRoundedRect(QRectF(0, 0, side, side), side * 0.18, side * 0.18);
    const int glyph = side * 3 / 8;
    painter.drawImage(QRect((side - glyph) / 2, (side - glyph) / 2, glyph, glyph),
                      QImage(QStringLiteral(":/veshiru/icons/menu_gamepad@2x.png")));
    painter.end();
    QPixmap out = QPixmap::fromImage(icon);
    out.setDevicePixelRatio(dpr);
    return out;
}

static QPixmap RoundedIcon(const QPixmap& picture, qreal dpr = 1.0) {
    if (!IsVeshiruTheme() || picture.isNull()) {
        return picture;
    }
    QImage icon(picture.size(), QImage::Format_ARGB32_Premultiplied);
    icon.fill(Qt::transparent);
    QPainter painter(&icon);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    QPainterPath path;
    const qreal radius = picture.width() * 0.18;
    path.addRoundedRect(QRectF(icon.rect()), radius, radius);
    painter.setClipPath(path);
    QPixmap raw = picture;
    raw.setDevicePixelRatio(1.0);
    painter.drawPixmap(0, 0, raw);
    painter.end();
    QPixmap out = QPixmap::fromImage(icon);
    out.setDevicePixelRatio(dpr);
    return out;
}

static QPixmap FormatHiDpiGameIcon(const u8* data, std::size_t byte_len, u32 logical_size,
                                   bool* out_valid = nullptr) {
    const qreal dpr = GameListIconDevicePixelRatio();
    const int phys = (std::max)(1, static_cast<int>(logical_size * dpr + 0.5));
    QImage source;
    const bool loaded =
        data != nullptr && byte_len > 0 &&
        source.loadFromData(data, static_cast<int>(byte_len)) && !source.isNull();
    if (out_valid != nullptr) {
        *out_valid = loaded;
    }
    if (!loaded) {
        return GetDefaultIcon(logical_size);
    }
    QImage scaled =
        source.scaled(phys, phys, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    if (scaled.width() > phys || scaled.height() > phys) {
        scaled = scaled.copy((scaled.width() - phys) / 2, (scaled.height() - phys) / 2, phys, phys);
    }
    return RoundedIcon(QPixmap::fromImage(scaled), dpr);
}

static QPixmap ThemeIcon(const char* name) {
    if (IsVeshiruTheme()) {
        // Veshiru Blue keeps folder rows compact with the 16 px line icons from its set.
        const std::string_view icon{name};
        const char* veshiru_name = icon == "sd_card"    ? "sd"
                                  : icon == "chip"     ? "cpu"
                                  : icon == "list-add" ? "plus"
                                  : icon == "star"     ? "star"
                                                       : "folder";
        QPixmap pixmap(
            QStringLiteral(":/veshiru/icons/menu_%1@2x.png").arg(QLatin1String(veshiru_name)));
        pixmap.setDevicePixelRatio(2.0);
        return pixmap;
    }

    const int size = UISettings::values.folder_icon_size.GetValue();

    return QIcon::fromTheme(QLatin1String(name))
        .pixmap(size, size)
        .scaled(size, size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

class GameListItem : public QStandardItem {

public:
    // used to access type from item index
    static constexpr int TypeRole = Qt::UserRole + 1;
    static constexpr int SortRole = Qt::UserRole + 2;
    GameListItem() = default;
    explicit GameListItem(const QString& string) : QStandardItem(string) {
        setData(string, SortRole);
    }
};

/**
 * A specialization of GameListItem for path values.
 * This class ensures that for every full path value it holds, a correct string representation
 * of just the filename (with no extension) will be displayed to the user.
 * If this class receives valid title metadata, it will also display game icons and titles.
 */
class GameListItemPath : public GameListItem {
public:
    static constexpr int TitleRole = SortRole + 1;
    static constexpr int FullPathRole = SortRole + 2;
    static constexpr int ProgramIdRole = SortRole + 3;
    static constexpr int FileTypeRole = SortRole + 4;
    static constexpr int HasIconRole = SortRole + 5;

    GameListItemPath() = default;
    GameListItemPath(const QString& game_path, const std::vector<u8>& picture_data,
                     const QString& game_name, const QString& game_type, u64 program_id,
                     u64 play_time, const QString& patch_versions) {
        setData(type(), TypeRole);
        setData(game_path, FullPathRole);
        setData(game_name, TitleRole);
        setData(qulonglong(program_id), ProgramIdRole);
        setData(game_type, FileTypeRole);

        const auto readable_play_time =
            play_time > 0 ? QObject::tr("Play Time: %1")
                                .arg(QString::fromStdString(
                                    PlayTime::PlayTimeManager::GetReadablePlayTime(play_time)))
                          : QObject::tr("Never Played");

        const auto enabled_update = [patch_versions]() -> QString {
            const QStringList lines = patch_versions.split(QLatin1Char('\n'));
            const QRegularExpression regex{QStringLiteral(R"(^Update \(([0-9\.]+)\))")};
            for (const QString& line : std::as_const(lines)) {
                const auto match = regex.match(line);
                if (match.hasMatch() && match.hasCaptured(1))
                    return QObject::tr("Version: %1").arg(match.captured(1));
            }
            return QObject::tr("Version: 1.0.0");
        }();

        const auto tooltip = QStringLiteral("%1\n%2").arg(readable_play_time, enabled_update);

        setData(tooltip, Qt::ToolTipRole);

        const u32 size = UISettings::CurrentGameIconSize().GetValue();
        bool valid_icon = false;
        setData(FormatHiDpiGameIcon(picture_data.data(), picture_data.size(), size, &valid_icon),
                Qt::DecorationRole);
        setData(valid_icon, HasIconRole);
    }

    void ApplyArtwork(const QByteArray& png, const QString& resolved_title = {}) {
        const u32 size = UISettings::CurrentGameIconSize().GetValue();
        bool valid_icon = false;
        const QPixmap pixmap = FormatHiDpiGameIcon(reinterpret_cast<const u8*>(png.constData()),
                                                   static_cast<std::size_t>(png.size()), size,
                                                   &valid_icon);
        if (valid_icon) {
            setData(pixmap, Qt::DecorationRole);
            setData(true, HasIconRole);
        }
        if (!resolved_title.isEmpty()) {
            const QString current_title = data(TitleRole).toString();
            const QString base_filename = QFileInfo(data(FullPathRole).toString()).completeBaseName();
            if (current_title.isEmpty() || current_title == QStringLiteral(" ") ||
                current_title == base_filename) {
                setData(resolved_title, TitleRole);
            }
        }
    }

    int type() const override {
        return static_cast<int>(GameListItemType::Game);
    }

    QVariant data(int role) const override {
        if (role == Qt::DisplayRole || role == SortRole) {
            std::string filename;
            Common::SplitPath(data(FullPathRole).toString().toStdString(), nullptr, &filename,
                              nullptr);

            const std::array<QString, 4> row_data{{
                QString::fromStdString(filename),
                data(FileTypeRole).toString(),
                QString::fromStdString(fmt::format("{:016X}", data(ProgramIdRole).toULongLong())),
                data(TitleRole).toString(),
            }};

            const auto& row1 = row_data.at(UISettings::values.row_1_text_id.GetValue());
            // don't show row 2 on grid view
            switch (UISettings::values.game_list_mode.GetValue()) {

            case Settings::GameListMode::TreeView: {
                const int row2_id = UISettings::values.row_2_text_id.GetValue();

                if (role == SortRole) {
                    return row1.toLower();
                }

                // None
                if (row2_id == 4) {
                    return row1;
                }

                const auto& row2 = row_data.at(row2_id);

                if (row1 == row2) {
                    return row1;
                }

                return QStringLiteral("%1\n    %2").arg(row1, row2);
            }
            case Settings::GameListMode::GridView:
            case Settings::GameListMode::CarouselView:
                return row1;
            default:
                break;
            }
        }

        return GameListItem::data(role);
    }
};

class GameListItemCompat : public GameListItem {
    Q_DECLARE_TR_FUNCTIONS(GameListItemCompat)
public:
    static constexpr int CompatNumberRole = SortRole;
    GameListItemCompat() = default;
    explicit GameListItemCompat(const QString& compatibility) {
        setData(type(), TypeRole);

        struct CompatStatus {
            QString color;
            const char* text;
            const char* tooltip;
        };
        // clang-format off
        const auto ingame_status =
                       CompatStatus{QStringLiteral("#f2d624"), QT_TR_NOOP("Ingame"),     QT_TR_NOOP("Game starts, but crashes or major glitches prevent it from being completed.")};
        static const std::map<QString, CompatStatus> status_data = {
            {QStringLiteral("0"),  {QStringLiteral("#5c93ed"), QT_TR_NOOP("Perfect"),    QT_TR_NOOP("Game can be played without issues.")}},
            {QStringLiteral("1"),  {QStringLiteral("#47d35c"), QT_TR_NOOP("Playable"),   QT_TR_NOOP("Game functions with minor graphical or audio glitches and is playable from start to finish.")}},
            {QStringLiteral("2"),  ingame_status},
            {QStringLiteral("3"),  ingame_status}, // Fallback for the removed "Okay" category
            {QStringLiteral("4"),  {QStringLiteral("#FF0000"), QT_TR_NOOP("Intro/Menu"), QT_TR_NOOP("Game loads, but is unable to progress past the Start Screen.")}},
            {QStringLiteral("5"),  {QStringLiteral("#828282"), QT_TR_NOOP("Won't Boot"), QT_TR_NOOP("The game crashes when attempting to startup.")}},
            {QStringLiteral("99"), {QStringLiteral("#000000"), QT_TR_NOOP("Not Tested"), QT_TR_NOOP("The game has not yet been tested.")}},
        };
        // clang-format on

        auto iterator = status_data.find(compatibility);
        if (iterator == status_data.end()) {
            LOG_WARNING(Frontend, "Invalid compatibility number {}", compatibility.toStdString());
            return;
        }
        const CompatStatus& status = iterator->second;
        setData(compatibility, CompatNumberRole);
        setText(tr(status.text));
        setToolTip(tr(status.tooltip));
        setData(QtCommon::CreateCirclePixmapFromColor(status.color), Qt::DecorationRole);
    }

    int type() const override {
        return static_cast<int>(GameListItemType::Game);
    }

    bool operator<(const QStandardItem& other) const override {
        return data(CompatNumberRole).value<QString>() <
               other.data(CompatNumberRole).value<QString>();
    }
};

/**
 * A specialization of GameListItem for size values.
 * This class ensures that for every numerical size value it holds (in bytes), a correct
 * human-readable string representation will be displayed to the user.
 */
class GameListItemSize : public GameListItem {
public:
    static constexpr int SizeRole = SortRole;

    GameListItemSize() = default;
    explicit GameListItemSize(const qulonglong size_bytes) {
        setData(type(), TypeRole);
        setData(size_bytes, SizeRole);
    }

    void setData(const QVariant& value, int role) override {
        // By specializing setData for SizeRole, we can ensure that the numerical and string
        // representations of the data are always accurate and in the correct format.
        if (role == SizeRole) {
            qulonglong size_bytes = value.toULongLong();
            GameListItem::setData(QtCommon::ReadableByteSize(size_bytes), Qt::DisplayRole);
            GameListItem::setData(value, SizeRole);
        } else {
            GameListItem::setData(value, role);
        }
    }

    int type() const override {
        return static_cast<int>(GameListItemType::Game);
    }

    /**
     * This operator is, in practice, only used by the TreeView sorting systems.
     * Override it so that it will correctly sort by numerical value instead of by string
     * representation.
     */
    bool operator<(const QStandardItem& other) const override {
        return data(SizeRole).toULongLong() < other.data(SizeRole).toULongLong();
    }
};

/**
 * GameListItem for Play Time values.
 * This object stores the play time of a game in seconds, and its readable
 * representation in minutes/hours
 */
class GameListItemPlayTime : public GameListItem {
public:
    static constexpr int PlayTimeRole = SortRole;

    GameListItemPlayTime() = default;
    explicit GameListItemPlayTime(const qulonglong time_seconds) {
        setData(time_seconds, PlayTimeRole);
    }

    void setData(const QVariant& value, int role) override {
        qulonglong time_seconds = value.toULongLong();
        GameListItem::setData(
            QString::fromStdString(PlayTime::PlayTimeManager::GetReadablePlayTime(time_seconds)),
            Qt::DisplayRole);
        GameListItem::setData(value, PlayTimeRole);
    }

    bool operator<(const QStandardItem& other) const override {
        return data(PlayTimeRole).toULongLong() < other.data(PlayTimeRole).toULongLong();
    }
};

class GameListDir : public GameListItem {
public:
    static constexpr int GameDirRole = Qt::UserRole + 2;

    explicit GameListDir(UISettings::GameDir& directory,
                         GameListItemType dir_type_ = GameListItemType::CustomDir)
        : dir_type{dir_type_} {
        setData(type(), TypeRole);

        UISettings::GameDir* game_dir = &directory;
        setData(QVariant(UISettings::values.game_dirs.indexOf(directory)), GameDirRole);

        const char* icon_name = nullptr;

        switch (dir_type) {
        case GameListItemType::SdmcDir:
            icon_name = "sd_card";
            setData(QObject::tr("Installed SD Titles"), Qt::DisplayRole);
            break;
        case GameListItemType::UserNandDir:
            icon_name = "chip";
            setData(QObject::tr("Installed NAND Titles"), Qt::DisplayRole);
            break;
        case GameListItemType::SysNandDir:
            icon_name = "chip";
            setData(QObject::tr("System Titles"), Qt::DisplayRole);
            break;
        case GameListItemType::CustomDir: {
            const QString path = QString::fromStdString(game_dir->path);
            icon_name = QFileInfo::exists(path) ? "folder" : "bad_folder";
            setData(path, Qt::DisplayRole);
            break;
        }
        default:
            break;
        }

        if (icon_name != nullptr)
            setData(ThemeIcon(icon_name), Qt::DecorationRole);
    }

    int type() const override {
        return static_cast<int>(dir_type);
    }

    /**
     * Override to prevent automatic sorting between folders and the addDir button.
     */
    bool operator<(const QStandardItem& other) const override {
        return false;
    }

private:
    GameListItemType dir_type;
};

class GameListAddDir : public GameListItem {
public:
    explicit GameListAddDir() {
        setData(type(), TypeRole);

        setData(ThemeIcon("list-add"), Qt::DecorationRole);
        setData(QObject::tr("Add New Game Directory"), Qt::DisplayRole);
    }

    int type() const override {
        return static_cast<int>(GameListItemType::AddDir);
    }

    bool operator<(const QStandardItem& other) const override {
        return false;
    }
};

class GameListFavorites : public GameListItem {
public:
    explicit GameListFavorites() {
        setData(type(), TypeRole);

        setData(ThemeIcon("star"), Qt::DecorationRole);
        setData(QObject::tr("Favorites"), Qt::DisplayRole);
    }

    int type() const override {
        return static_cast<int>(GameListItemType::Favorites);
    }

    bool operator<(const QStandardItem& other) const override {
        return false;
    }
};
