// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QProgressDialog>
#include <QWidget>

#include "veshiru/util/icon_search.h"

#ifdef VESHIRU_EXPERIMENTAL_ONLINE_ARTWORK
#include <algorithm>
#include <utility>

#include <QBuffer>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QIcon>
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include "common/artwork_cache.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "veshiru/util/steamgriddb_provider.h"

namespace {

namespace Http = Veshiru::ArtworkHttp;
const auto search_gate = std::make_shared<Http::SearchGate>();

QByteArray DecodeIcon(std::string_view bytes) {
    if (!Common::ArtworkCache::IsValidImageBytes(
            {reinterpret_cast<const u8*>(bytes.data()), bytes.size()})) {
        return {};
    }
    QBuffer input;
    input.setData(QByteArray::fromRawData(bytes.data(), static_cast<qsizetype>(bytes.size())));
    input.open(QIODevice::ReadOnly);
    QImageReader reader(&input, QByteArrayLiteral("png"));
    reader.setAutoDetectImageFormat(false);
    const QSize size = reader.size();
    const auto image = reader.read();
    if (image.isNull() || image.size() != size) {
        return {};
    }
    const int square = std::min(size.width(), size.height());
    const int out_size = square >= 384 ? 512 : 256;
    // Re-encode just the pixels. No source comments, profile, or other chunks are carried over.
    QImage normalized(out_size, out_size, QImage::Format_RGBA8888);
    normalized.fill(Qt::transparent);
    QPainter painter(&normalized);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.drawImage(
        normalized.rect(), image,
        QRect{(size.width() - square) / 2, (size.height() - square) / 2, square, square});
    painter.end();
    QByteArray png;
    QBuffer output(&png);
    output.open(QIODevice::WriteOnly);
    return normalized.save(&output, "PNG") ? png : QByteArray{};
}

IconSearchStatus SearchStatus(Http::Status status) {
    switch (status) {
    case Http::Status::NotFound:
        return IconSearchStatus::NotFound;
    case Http::Status::Unauthorized:
        return IconSearchStatus::BadApiKey;
    case Http::Status::RateLimited:
        return IconSearchStatus::RateLimited;
    case Http::Status::Cancelled:
        return IconSearchStatus::Cancelled;
    case Http::Status::Timeout:
        return IconSearchStatus::Timeout;
    default:
        return IconSearchStatus::Failed;
    }
}

} // namespace
#endif

IconSearch::IconSearch(QWidget* parent) : QObject(parent), parent_widget{parent} {
#ifdef VESHIRU_EXPERIMENTAL_ONLINE_ARTWORK
    connect(&watcher, &QFutureWatcher<IconSearchResult>::finished, this, &IconSearch::OnFinished);
#endif
}

IconSearch::~IconSearch() {
    cancel_flag->store(true);
    delete progress.data();
}

void IconSearch::Start(const QString& name, u64 title_id, bool write_cache) {
#ifdef VESHIRU_EXPERIMENTAL_ONLINE_ARTWORK
    if (running) {
        return;
    }
    SetRunning(true);
    cache_title_id = write_cache ? title_id : 0;
    auto* dialog = new QDialog(parent_widget);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle(tr("Search SteamGridDB"));
    auto* layout = new QVBoxLayout(dialog);
    auto* help = new QLabel(
        tr("The title below and your SteamGridDB API key are sent to SteamGridDB. No game paths, "
           "keys for your games, or library list are sent. Use only artwork you have permission "
           "to use. See its <a href=\"https://www.steamgriddb.com/terms\">terms</a> and "
           "<a href=\"https://www.steamgriddb.com/privacy\">privacy policy</a>."),
        dialog);
    help->setTextFormat(Qt::RichText);
    help->setWordWrap(true);
    help->setOpenExternalLinks(true);
    layout->addWidget(help);
    auto* form = new QFormLayout;
    auto* title =
        new QLineEdit(QString::fromStdString(Common::ArtworkCache::ResolveCanonicalSearchTitle(
                          name.trimmed().toStdString(), title_id)),
                      dialog);
    title->setMaxLength(256);
    auto* key = new QLineEdit(QString::fromStdString(session_key), dialog);
    key->setEchoMode(QLineEdit::Password);
    key->setMaxLength(128);
    form->addRow(tr("Search title"), title);
    form->addRow(tr("API key (not saved)"), key);
    layout->addLayout(form);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, dialog);
    connect(buttons, &QDialogButtonBox::accepted, dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, dialog, &QDialog::reject);
    connect(dialog, &QDialog::rejected, this, [this] { SetRunning(false); });
    connect(dialog, &QDialog::accepted, this, [this, title, key] {
        auto query = title->text().trimmed().toStdString();
        if (!Common::ArtworkCache::IsSafeArtworkTitle(query)) {
            SetRunning(false);
            QMessageBox::warning(parent_widget, tr("Find Icon"),
                                 tr("Enter a title, not a path or URL."));
            return;
        }
        session_key = key->text().trimmed().toStdString();
        BeginSearch(std::move(query));
    });
    layout->addWidget(buttons);
    dialog->resize(480, dialog->sizeHint().height());
    dialog->open();
#else
    (void)name;
    (void)title_id;
    (void)write_cache;
#endif
}

bool IconSearch::IsRunning() const {
    return running;
}

#ifdef VESHIRU_EXPERIMENTAL_ONLINE_ARTWORK
void IconSearch::SetRunning(bool active) {
    running = active;
    emit RunningChanged(active);
}

void IconSearch::BeginSearch(std::string query) {
    if (!search_gate->TryBegin(Http::Clock::now())) {
        SetRunning(false);
        QMessageBox::information(parent_widget, tr("Find Icon"),
                                 tr("Another search is running or cooling down. Try again later."));
        return;
    }
    auto launch_guard = SCOPE_GUARD {
        search_gate->End(Http::Clock::now(), false);
    };
    cancel_flag->store(false);
    progress = new QProgressDialog(tr("Looking for artwork…"), tr("Cancel"), 0, 0, parent_widget);
    progress->setAutoReset(false);
    progress->setAutoClose(false);
    progress->setMinimumDuration(300);
    progress->setValue(0);
    connect(progress, &QProgressDialog::canceled, this, [this] { cancel_flag->store(true); });
    watcher.setFuture(QtConcurrent::run(
        [query = std::move(query), key = session_key, cancel = cancel_flag, gate = search_gate] {
            IconSearchResult result;
            bool rate_limited = false;
            SCOPE_EXIT {
                gate->End(Http::Clock::now(), rate_limited);
            };
            const auto response = Veshiru::SteamGridDB::FindIcons(query, key, *cancel);
            rate_limited = response.status == Http::Status::RateLimited;
            if (response.status != Http::Status::Ok) {
                result.status = SearchStatus(response.status);
                return result;
            }
            for (const auto& body : response.pngs) {
                if (*cancel) {
                    result.status = IconSearchStatus::Cancelled;
                    return result;
                }
                if (auto png = DecodeIcon(body); !png.isEmpty()) {
                    result.choices.push_back(std::move(png));
                }
            }
            result.status =
                result.choices.empty() ? IconSearchStatus::Failed : IconSearchStatus::Choices;
            return result;
        }));
    launch_guard.Cancel();
}

void IconSearch::ChooseIcon(std::vector<QByteArray> choices) {
    auto* dialog = new QDialog(parent_widget);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle(tr("Choose an Icon"));
    auto* layout = new QVBoxLayout(dialog);
    layout->addWidget(
        new QLabel(tr("Preview from SteamGridDB. Select an icon to use it."), dialog));
    auto* list = new QListWidget(dialog);
    list->setViewMode(QListView::IconMode);
    list->setIconSize(QSize(128, 128));
    list->setResizeMode(QListView::Adjust);
    list->setMovement(QListView::Static);
    list->setSpacing(8);
    for (std::size_t i = 0; i < choices.size(); ++i) {
        QPixmap map;
        if (map.loadFromData(choices[i], "PNG")) {
            auto* item = new QListWidgetItem(QIcon(map), QString::number(i + 1));
            item->setData(Qt::UserRole, static_cast<uint>(i));
            list->addItem(item);
        }
    }
    if (list->count() == 0) {
        dialog->deleteLater();
        SetRunning(false);
        return;
    }
    list->setCurrentRow(0);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, dialog);
    connect(buttons, &QDialogButtonBox::accepted, dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, dialog, &QDialog::reject);
    connect(list, &QListWidget::itemDoubleClicked, dialog, &QDialog::accept);
    connect(dialog, &QDialog::rejected, this, [this] { SetRunning(false); });
    connect(dialog, &QDialog::accepted, this, [this, list, choices = std::move(choices)] {
        if (const auto* selected = list->currentItem()) {
            UseIcon(choices[selected->data(Qt::UserRole).toUInt()]);
        } else {
            SetRunning(false);
        }
    });
    layout->addWidget(list);
    layout->addWidget(buttons);
    dialog->resize(520, 420);
    dialog->open();
}

void IconSearch::UseIcon(QByteArray png) {
    if (cache_title_id == 0) {
        emit IconSelected(png);
        SetRunning(false);
        return;
    }
    watcher.setFuture(QtConcurrent::run([id = cache_title_id, png = std::move(png)] {
        if (!Common::ArtworkCache::WriteCachedArtwork(id,
                                                      {reinterpret_cast<const u8*>(png.constData()),
                                                       static_cast<std::size_t>(png.size())},
                                                      {}, "steamgriddb")) {
            LOG_WARNING(Frontend,
                        "Selected artwork is available for this session but could not be cached");
        }
        return IconSearchResult{IconSearchStatus::Found, {png}};
    }));
}

void IconSearch::OnFinished() {
    delete progress.data();
    IconSearchResult result;
    try {
        result = watcher.result();
    } catch (...) {
        result.status = IconSearchStatus::Failed;
    }
    if (cancel_flag->load() || result.status == IconSearchStatus::Cancelled) {
        SetRunning(false);
        return;
    }
    if (result.status == IconSearchStatus::Choices) {
        ChooseIcon(std::move(result.choices));
        return;
    }
    if (result.status == IconSearchStatus::Found) {
        emit IconSelected(result.choices.front());
        SetRunning(false);
        return;
    }
    SetRunning(false);
    QString message;
    switch (result.status) {
    case IconSearchStatus::NotFound:
        message = tr("No matching artwork was found. Try a different title or use local artwork.");
        break;
    case IconSearchStatus::BadApiKey:
        session_key.clear();
        message = tr("Use your own SteamGridDB API key. Check that it is correct and try again.");
        break;
    case IconSearchStatus::RateLimited:
        message = tr("SteamGridDB limited requests. Wait at least one minute before trying again.");
        break;
    case IconSearchStatus::Timeout:
        message = tr("The artwork search timed out. Existing artwork has been kept.");
        break;
    default:
        message =
            tr("Could not get usable artwork from SteamGridDB. Existing artwork has been kept.");
        break;
    }
    QMessageBox::warning(parent_widget, tr("Find Icon"), message);
}
#endif
