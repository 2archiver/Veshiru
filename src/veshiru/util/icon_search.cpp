// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "veshiru/util/icon_search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/algorithm/string/replace.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <QBuffer>
#include <QByteArray>
#include <QDialog>
#include <QDialogButtonBox>
#include <QIcon>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMetaObject>
#include <QPixmap>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include "common/artwork_cache.h"
#include "common/httplib.h"
#include "common/logging.h"
#include "common/string_util.h"
#include "qt_common/config/uisettings.h"
#ifdef VESHIRU_BUNDLED_OPENSSL
#include <openssl/cert.h>
#endif

namespace {

constexpr int IconStandardSize = 256;
constexpr int IconHdSize = 512;
constexpr int MinCandidateScore = 300;
constexpr std::size_t MaxIconDownloads = 3;
constexpr std::size_t MaxSgdbChoices = 8;

std::vector<u8> EncodeSquareIcon(const QImage& source) {
    if (source.isNull()) {
        return {};
    }
    const int min_dim = std::min(source.width(), source.height());
    if (min_dim < 32) {
        return {};
    }
    // Preserve native 512x512 HD artwork when available; use 256x256 otherwise.
    const int target_size = min_dim >= 384 ? IconHdSize : IconStandardSize;
    QImage img = source.convertToFormat(QImage::Format_RGBA8888);
    if (img.width() != target_size || img.height() != target_size) {
        img = img.scaled(target_size, target_size, Qt::KeepAspectRatioByExpanding,
                         Qt::SmoothTransformation);
        if (img.width() > target_size || img.height() > target_size) {
            const int x = (img.width() - target_size) / 2;
            const int y = img.height() > img.width() ? (img.height() - target_size) / 3
                                                     : (img.height() - target_size) / 2;
            img = img.copy(x, y, target_size, target_size);
        }
    }
    QByteArray encoded;
    QBuffer buffer(&encoded);
    buffer.open(QIODevice::WriteOnly);
    img.save(&buffer, "PNG");
    return std::vector<u8>(encoded.begin(), encoded.end());
}

std::string UrlEncode(std::string_view value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size() * 3);
    for (const unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0x0F]);
        }
    }
    return out;
}

constexpr std::chrono::seconds SearchDeadline{10};
constexpr std::size_t MaxHttpBodySize = 6 * 1024 * 1024;
constexpr std::chrono::seconds TitleIdProbeDeadline{3};
constexpr std::chrono::seconds PrimarySourceDeadline{6};
constexpr std::string_view IconSiteHost = "https://switchiconshowdown.com";
constexpr std::string_view TitleIdApiHost = "https://api.nlib.cc";
constexpr std::string_view SgdbApiBase = "https://www.steamgriddb.com/api/v2";

bool IsAllowedIconHost(std::string_view host) {
    return host == "switchiconshowdown.com" || host == "api.nlib.cc" ||
           host == "img-eshop.cdn.nintendo.net" || host.ends_with(".nintendo.net") ||
           host == "steamgriddb.com" || host.ends_with(".steamgriddb.com");
}

enum class HttpStatus { Ok, NetworkError, Timeout, Cancelled, Unauthorized };

struct HttpResult {
    HttpStatus status{HttpStatus::NetworkError};
    std::string body;
    std::string detail;
};

// Every request shares one deadline and cancel flag, so a whole search can never run past it.
HttpResult HttpGet(const std::string& full_url, std::chrono::steady_clock::time_point deadline,
                   const std::atomic_bool& cancel, const std::string& bearer_token = {}) {
    using namespace std::chrono;
    HttpResult result;
    if (cancel) {
        result.status = HttpStatus::Cancelled;
        return result;
    }
    constexpr std::string_view scheme = "https://";
    if (!full_url.starts_with(scheme)) {
        result.detail = "unsupported URL";
        return result;
    }
    const auto path_start = full_url.find('/', scheme.size());
    const std::string origin =
        path_start == std::string::npos ? full_url : full_url.substr(0, path_start);
    const std::string path =
        path_start == std::string::npos ? std::string{"/"} : full_url.substr(path_start);
    if (!IsAllowedIconHost(std::string_view{origin}.substr(scheme.size()))) {
        result.detail = "unexpected host";
        return result;
    }
    const auto remaining = duration_cast<milliseconds>(deadline - steady_clock::now());
    if (remaining.count() <= 0) {
        result.status = HttpStatus::Timeout;
        return result;
    }
    try {
        httplib::Client client(origin);
        client.set_connection_timeout(remaining);
        client.set_read_timeout(remaining);
        client.set_write_timeout(remaining);
        client.set_max_timeout(remaining);
        client.set_follow_location(true);
#ifdef VESHIRU_BUNDLED_OPENSSL
        client.load_ca_cert_store(kCert, sizeof(kCert));
#endif
        httplib::Headers headers{{"User-Agent", "Veshiru"}};
        if (!bearer_token.empty()) {
            headers.emplace("Authorization", "Bearer " + bearer_token);
        }
        auto res = client.Get(path, headers);
        if (!res) {
            result.status = res.error() == httplib::Error::Canceled || steady_clock::now() >= deadline
                                ? HttpStatus::Timeout
                                : HttpStatus::NetworkError;
            result.detail = httplib::to_string(res.error());
            return result;
        }
        if (res->status != 200) {
            result.status = res->status == 401 || res->status == 403 ? HttpStatus::Unauthorized
                                                                      : HttpStatus::NetworkError;
            result.detail = fmt::format("HTTP {}", res->status);
            return result;
        }
        if (res->body.empty() || res->body.size() > MaxHttpBodySize) {
            result.detail = "unexpected response size";
            return result;
        }
        result.status = HttpStatus::Ok;
        result.body = std::move(res->body);
    } catch (const std::exception& e) {
        result.detail = e.what();
    } catch (...) {
        result.detail = "unknown error";
    }
    return result;
}

struct IconCandidate {
    int id{};
    std::string title;
};

std::string DecodeHtmlEntities(std::string text) {
    static constexpr std::pair<std::string_view, std::string_view> entities[]{
        {"&amp;", "&"},  {"&lt;", "<"},     {"&gt;", ">"},     {"&quot;", "\""},
        {"&#039;", "'"}, {"&#39;", "'"},    {"&apos;", "'"},   {"&eacute;", "\xC3\xA9"},
    };
    for (const auto& [from, to] : entities) {
        boost::replace_all(text, from, to);
    }
    return text;
}

// Rows look like: <a href='./detail.php?id=N'><img ...></a></td><td><center>Title</center>
std::vector<IconCandidate> ParseSearchResults(const std::string& html) {
    constexpr std::string_view id_marker = "detail.php?id=";
    constexpr std::string_view title_marker = "<td><center>";
    std::vector<IconCandidate> out;
    std::size_t pos = 0;
    while (out.size() < 16) {
        const auto id_pos = html.find(id_marker, pos);
        if (id_pos == std::string::npos) {
            break;
        }
        pos = id_pos + id_marker.size();
        const auto id_end = html.find_first_not_of("0123456789", pos);
        const auto digits = html.substr(pos, id_end == std::string::npos ? id_end : id_end - pos);
        if (digits.empty() || digits.size() > 9) {
            continue;
        }
        const auto title_pos = html.find(title_marker, pos);
        if (title_pos == std::string::npos) {
            break;
        }
        const auto title_begin = title_pos + title_marker.size();
        const auto title_end = html.find("</center>", title_begin);
        if (title_end == std::string::npos) {
            break;
        }
        auto title = DecodeHtmlEntities(html.substr(title_begin, title_end - title_begin));
        if (title.find('<') == std::string::npos && !title.empty()) {
            out.push_back({std::stoi(digits), std::move(title)});
        }
        pos = title_end;
    }
    return out;
}

std::string UrlEncodePathSegment(std::string value) {
    std::ranges::replace(value, '/', ' ');
    std::ranges::replace(value, '\\', ' ');
    auto encoded = UrlEncode(value);
    boost::replace_all(encoded, "+", "%20");
    return encoded;
}

// Icon image URLs from a SteamGridDB response, best size first.
std::vector<std::string> ParseSgdbIconUrls(const std::string& body) {
    std::vector<std::string> urls;
    const auto json = nlohmann::json::parse(body, nullptr, false);
    if (json.is_discarded() || !json.is_object() || !json.contains("data") ||
        !json["data"].is_array()) {
        return urls;
    }
    for (const auto& item : json["data"]) {
        if (urls.size() >= MaxSgdbChoices) {
            break;
        }
        if (item.is_object() && item.contains("url") && item["url"].is_string()) {
            urls.push_back(item["url"].get<std::string>());
        }
    }
    return urls;
}

int ScoreCandidateTitle(const std::string& candidate, const std::string& query) {
    return Common::ArtworkCache::ScoreArtworkCandidate(candidate, query);
}

IconSearchResult RunIconSearch(const std::string& raw_query, u64 tid, const std::string& api_key,
                               const std::atomic_bool& cancel, bool silent_first_choice = false) {
    using namespace std::chrono;
    IconSearchResult out;
    try {
        const auto start = steady_clock::now();
        const auto deadline = start + SearchDeadline;
        const u64 base_tid = Common::ArtworkCache::BaseProgramId(tid);
        std::string search_name =
            Common::ArtworkCache::ResolveCanonicalSearchTitle(raw_query, base_tid);
        if (!search_name.empty() && search_name != raw_query) {
            out.resolved_title = search_name;
        }

        const auto to_status = [](const HttpResult& res) {
            switch (res.status) {
            case HttpStatus::Timeout:
                return IconSearchStatus::Timeout;
            case HttpStatus::Cancelled:
                return IconSearchStatus::Cancelled;
            case HttpStatus::Unauthorized:
                return IconSearchStatus::BadApiKey;
            default:
                return IconSearchStatus::NetworkError;
            }
        };
        const auto decode_icon = [&out](const std::string& body) {
            QImage img;
            if (img.loadFromData(reinterpret_cast<const uchar*>(body.data()),
                                 static_cast<int>(body.size()))) {
                out.icon_bytes = EncodeSquareIcon(img);
            }
            return !out.icon_bytes.empty();
        };

        // 1. Exact Title ID lookup when the Program ID is known: resolves canonical title and
        //    official HD eShop icon when available, avoiding wrong editions completely.
        if (base_tid != 0 && Common::ArtworkCache::LookupTitleByProgramId(base_tid).empty()) {
            const auto tid_deadline = std::min(deadline, start + TitleIdProbeDeadline);
            const auto tid_res = HttpGet(
                fmt::format("{}/nx/{:016X}", TitleIdApiHost, base_tid), tid_deadline, cancel);
            if (tid_res.status == HttpStatus::Ok) {
                const auto meta = nlohmann::json::parse(tid_res.body, nullptr, false);
                if (!meta.is_discarded() && meta.is_object()) {
                    if (meta.contains("name") && meta["name"].is_string()) {
                        const auto exact_name = meta["name"].get<std::string>();
                        if (!exact_name.empty()) {
                            search_name = exact_name;
                            out.resolved_title = exact_name;
                        }
                    }
                    for (const char* icon_field : {"iconUrl", "icon_url", "icon"}) {
                        if (meta.contains(icon_field) && meta[icon_field].is_string()) {
                            const auto icon_url = meta[icon_field].get<std::string>();
                            if (icon_url.starts_with("https://")) {
                                const auto img_res = HttpGet(icon_url, tid_deadline, cancel);
                                if (img_res.status == HttpStatus::Ok && decode_icon(img_res.body)) {
                                    out.status = IconSearchStatus::Found;
                                    out.source = "nintendo_cdn";
                                    return out;
                                }
                            }
                        }
                    }
                }
            } else if (tid_res.status == HttpStatus::Cancelled) {
                out.status = IconSearchStatus::Cancelled;
                return out;
            }
        }

        if (search_name.empty()) {
            out.status = IconSearchStatus::NotFound;
            return out;
        }

        // 2. Primary HD icon source: switchiconshowdown.com (full-res PNG first, never stretching
        //    100x100 thumbnails unless no full-res asset exists).
        IconSearchStatus primary = IconSearchStatus::NotFound;
        std::string primary_error;
        const auto primary_deadline =
            std::min(deadline, steady_clock::now() + PrimarySourceDeadline);
        do {
            std::vector<std::string> queries{search_name};
            if (const auto colon = search_name.find(':');
                colon != std::string::npos && colon > 0) {
                queries.push_back(search_name.substr(0, colon));
            }
            std::vector<IconCandidate> candidates;
            for (const auto& query : queries) {
                const auto res =
                    HttpGet(std::string{IconSiteHost} + "/search.php?query=" + UrlEncode(query),
                            primary_deadline, cancel);
                if (res.status != HttpStatus::Ok) {
                    primary = to_status(res);
                    primary_error = res.detail;
                    break;
                }
                candidates = ParseSearchResults(res.body);
                if (!candidates.empty()) {
                    break;
                }
            }
            if (primary != IconSearchStatus::NotFound) {
                break;
            }

            // The site can list one game several times; keep the closest title first.
            std::vector<std::pair<int, IconCandidate>> scored;
            for (auto& cand : candidates) {
                const int score = ScoreCandidateTitle(cand.title, search_name);
                if (score >= MinCandidateScore) {
                    scored.emplace_back(score, std::move(cand));
                }
            }
            std::ranges::stable_sort(
                scored, [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t i = 0; i < scored.size() && i < MaxIconDownloads; ++i) {
                const int icon_id = scored[i].second.id;
                for (const std::string_view pattern :
                     {"/images/{}.png", "/images/256/{}.png", "/images/100/{}.png"}) {
                    const auto res = HttpGet(
                        std::string{IconSiteHost} + fmt::format(fmt::runtime(pattern), icon_id),
                        primary_deadline, cancel);
                    if (res.status == HttpStatus::Cancelled || res.status == HttpStatus::Timeout) {
                        primary = to_status(res);
                        primary_error = res.detail;
                        break;
                    }
                    if (res.status == HttpStatus::Ok && decode_icon(res.body)) {
                        if (out.resolved_title.empty()) {
                            out.resolved_title = scored[i].second.title;
                        }
                        out.status = IconSearchStatus::Found;
                        out.source = "switchiconshowdown";
                        return out;
                    }
                }
                if (primary == IconSearchStatus::Cancelled ||
                    primary == IconSearchStatus::Timeout) {
                    break;
                }
            }
        } while (false);

        if (primary == IconSearchStatus::Cancelled) {
            out.status = primary;
            return out;
        }

        // 3. Fallback HD source: SteamGridDB (scored by title relevance to avoid wrong editions).
        if (api_key.empty()) {
            out.status = primary == IconSearchStatus::NotFound ? IconSearchStatus::NeedsApiKey
                                                               : primary;
            out.error = primary_error;
            return out;
        }
        const auto sgdb_get = [&](const std::string& path) {
            return HttpGet(std::string{SgdbApiBase} + path, deadline, cancel, api_key);
        };
        const auto search = sgdb_get("/search/autocomplete/" + UrlEncodePathSegment(search_name));
        if (search.status != HttpStatus::Ok) {
            out.status = to_status(search);
            out.error = search.detail;
            return out;
        }
        const auto parsed = nlohmann::json::parse(search.body, nullptr, false);
        int best_game_id = 0;
        int best_score = MinCandidateScore - 1;
        std::string best_sgdb_title;
        if (!parsed.is_discarded() && parsed.is_object() && parsed.contains("data") &&
            parsed["data"].is_array()) {
            for (const auto& entry : parsed["data"]) {
                if (!entry.is_object() || !entry.contains("id") ||
                    !entry["id"].is_number_integer() || !entry.contains("name") ||
                    !entry["name"].is_string()) {
                    continue;
                }
                const auto cand_name = entry["name"].get<std::string>();
                const int score = ScoreCandidateTitle(cand_name, search_name);
                if (score > best_score) {
                    best_score = score;
                    best_game_id = entry["id"].get<int>();
                    best_sgdb_title = cand_name;
                }
            }
        }
        if (best_game_id <= 0) {
            out.status = IconSearchStatus::NotFound;
            return out;
        }
        const auto icons = sgdb_get(
            fmt::format("/icons/game/{}?dimensions=512,256&mimes=image/png", best_game_id));
        if (icons.status != HttpStatus::Ok) {
            out.status = to_status(icons);
            out.error = icons.detail;
            return out;
        }
        // Collect every icon that arrives in time; the user picks one.
        for (const auto& url : ParseSgdbIconUrls(icons.body)) {
            const auto img = HttpGet(url, deadline, cancel);
            if (img.status != HttpStatus::Ok) {
                if (img.status == HttpStatus::Cancelled || img.status == HttpStatus::Timeout) {
                    if (out.choices.empty()) {
                        out.status = to_status(img);
                        out.error = img.detail;
                        return out;
                    }
                    break;
                }
                continue;
            }
            if (decode_icon(img.body)) {
                if (silent_first_choice) {
                    if (out.resolved_title.empty()) {
                        out.resolved_title = best_sgdb_title;
                    }
                    out.status = IconSearchStatus::Found;
                    out.source = "steamgriddb";
                    return out;
                }
                out.choices.push_back(std::move(out.icon_bytes));
                out.icon_bytes.clear();
            }
        }
        if (!out.choices.empty()) {
            if (out.resolved_title.empty()) {
                out.resolved_title = best_sgdb_title;
            }
            out.status = IconSearchStatus::Choices;
            out.source = "steamgriddb";
            return out;
        }
        out.status = IconSearchStatus::NotFound;
    } catch (const std::exception& e) {
        out.status = IconSearchStatus::InternalError;
        out.error = e.what();
    } catch (...) {
        out.status = IconSearchStatus::InternalError;
        out.error = "unknown error";
    }
    return out;
}

} // namespace

IconSearch::IconSearch(QWidget* parent) : QObject(parent), parent_widget{parent} {
    connect(&watcher, &QFutureWatcher<IconSearchResult>::finished, this, &IconSearch::OnFinished);
}

IconSearch::~IconSearch() {
    // Never block the UI thread: the detached worker stops on its own once it sees the flag.
    cancel_flag->store(true);
}

void IconSearch::Start(const QString& name, u64 title_id) {
    if (watcher.isRunning()) {
        return;
    }
    const std::string query = name.trimmed().toStdString();
    if (query.empty() && title_id == 0) {
        QMessageBox::information(parent_widget, tr("Find Icon"),
                                 tr("Please enter a game name first so the icon can be found."));
        return;
    }
    last_name = name;
    last_title_id = title_id;
    cancel_flag->store(false);
    emit RunningChanged(true);
    // The worker captures no pointer to the dialog, so it may safely outlive it.
    watcher.setFuture(QtConcurrent::run(
        [query, title_id, api_key = UISettings::values.steamgriddb_api_key.GetValue(),
         cancel = cancel_flag] { return RunIconSearch(query, title_id, api_key, *cancel); }));
}

bool IconSearch::IsRunning() const {
    return watcher.isRunning();
}

void IconSearch::FetchSilentAsync(
    u64 title_id, const QString& fallback_name, QObject* context,
    std::function<void(QByteArray png, QString resolved_title)> on_done) {
    if (title_id == 0 || context == nullptr || !on_done) {
        return;
    }
    const std::string query = fallback_name.trimmed().toStdString();
    const std::string api_key = UISettings::values.steamgriddb_api_key.GetValue();
    (void)QtConcurrent::run([title_id, query, api_key, context = QPointer<QObject>(context),
                             cb = std::move(on_done)]() mutable {
        std::atomic_bool cancel{false};
        IconSearchResult res = RunIconSearch(query, title_id, api_key, cancel, true);
        if (res.status == IconSearchStatus::Found && !res.icon_bytes.empty()) {
            const std::string effective_title =
                !res.resolved_title.empty() ? res.resolved_title : query;
            (void)Common::ArtworkCache::WriteCachedArtwork(
                title_id, res.icon_bytes, effective_title,
                res.source.empty() ? "online" : res.source);
            if (context) {
                QByteArray bytes(reinterpret_cast<const char*>(res.icon_bytes.data()),
                                 static_cast<qsizetype>(res.icon_bytes.size()));
                QString title_q = QString::fromStdString(res.resolved_title);
                QMetaObject::invokeMethod(
                    context,
                    [cb = std::move(cb), bytes = std::move(bytes),
                     title_q = std::move(title_q)]() mutable {
                        cb(std::move(bytes), std::move(title_q));
                    },
                    Qt::QueuedConnection);
            }
        } else {
            Common::ArtworkCache::RecordArtworkFailure(
                title_id, res.error.empty() ? "not_found" : res.error);
        }
    });
}

int IconSearch::ChooseIcon(const std::vector<std::vector<u8>>& choices) {
    if (choices.size() == 1) {
        return 0;
    }
    QDialog dialog(parent_widget);
    dialog.setWindowTitle(tr("Choose an Icon"));
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(
        new QLabel(tr("SteamGridDB has several icons for this game. Pick one:"), &dialog));
    auto* list = new QListWidget(&dialog);
    list->setViewMode(QListView::IconMode);
    list->setIconSize(QSize(128, 128));
    list->setResizeMode(QListView::Adjust);
    list->setMovement(QListView::Static);
    list->setSpacing(8);
    list->setSelectionMode(QAbstractItemView::SingleSelection);
    for (std::size_t i = 0; i < choices.size(); ++i) {
        QPixmap map;
        if (map.loadFromData(choices[i].data(), static_cast<uint>(choices[i].size()))) {
            auto* item = new QListWidgetItem(QIcon(map), QString::number(i + 1));
            item->setData(Qt::UserRole, static_cast<int>(i));
            list->addItem(item);
        }
    }
    if (list->count() == 0) {
        return -1;
    }
    list->setCurrentRow(0);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(list, &QListWidget::itemDoubleClicked, &dialog, &QDialog::accept);
    layout->addWidget(list);
    layout->addWidget(buttons);
    dialog.resize(520, 420);
    if (dialog.exec() != QDialog::Accepted || list->currentItem() == nullptr) {
        return -1;
    }
    return list->currentItem()->data(Qt::UserRole).toInt();
}

bool IconSearch::AskForApiKey(bool missing) {
    QString text = missing ? tr("No matching icon was found on switchiconshowdown.com. A "
                                "SteamGridDB API key enables a fallback search.")
                           : tr("SteamGridDB rejected the saved API key.");
    if (UISettings::values.steamgriddb_api_key.GetValue().empty()) {
        text += QStringLiteral("<br><br>") +
                tr("Create a free key on the <a href=\"%1\">SteamGridDB API preferences</a> page.")
                    .arg(QStringLiteral("https://www.steamgriddb.com/profile/preferences/api"));
    }
    QDialog dialog(parent_widget);
    dialog.setWindowTitle(tr("SteamGridDB API Key"));
    auto* layout = new QVBoxLayout(&dialog);
    auto* label = new QLabel(text, &dialog);
    label->setTextFormat(Qt::RichText);
    label->setOpenExternalLinks(true);
    label->setWordWrap(true);
    auto* key_edit = new QLineEdit(
        QString::fromStdString(UISettings::values.steamgriddb_api_key.GetValue()), &dialog);
    key_edit->setEchoMode(QLineEdit::Password);
    key_edit->setPlaceholderText(tr("API key"));
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(label);
    layout->addWidget(key_edit);
    layout->addWidget(buttons);
    dialog.setMinimumWidth(420);
    if (dialog.exec() != QDialog::Accepted || key_edit->text().trimmed().isEmpty()) {
        return false;
    }
    UISettings::values.steamgriddb_api_key.SetValue(key_edit->text().trimmed().toStdString());
    return true;
}

void IconSearch::OnFinished() {
    emit RunningChanged(false);

    IconSearchResult result;
    try {
        result = watcher.result();
    } catch (const std::exception& e) {
        result.status = IconSearchStatus::InternalError;
        result.error = e.what();
    } catch (...) {
        result.status = IconSearchStatus::InternalError;
    }

    const auto select = [&](std::vector<u8> png) {
        if (last_title_id != 0 && !png.empty()) {
            const std::string effective_title = !result.resolved_title.empty()
                                                    ? result.resolved_title
                                                    : last_name.trimmed().toStdString();
            (void)Common::ArtworkCache::WriteCachedArtwork(
                last_title_id, png, effective_title,
                result.source.empty() ? "manual_search" : result.source);
        }
        emit IconSelected(QByteArray(reinterpret_cast<const char*>(png.data()),
                                     static_cast<qsizetype>(png.size())),
                          QString::fromStdString(result.resolved_title));
    };

    QString message;
    switch (result.status) {
    case IconSearchStatus::Found:
        select(std::move(result.icon_bytes));
        return;
    case IconSearchStatus::Choices: {
        const int index = ChooseIcon(result.choices);
        if (index >= 0) {
            select(std::move(result.choices[static_cast<std::size_t>(index)]));
        }
        return;
    }
    case IconSearchStatus::NotFound:
        message = tr("No matching icon was found on switchiconshowdown.com or SteamGridDB. Try a "
                     "different game name or choose an icon file.");
        break;
    case IconSearchStatus::NeedsApiKey:
    case IconSearchStatus::BadApiKey:
        if (AskForApiKey(result.status == IconSearchStatus::NeedsApiKey)) {
            Start(last_name, last_title_id);
        }
        return;
    case IconSearchStatus::Timeout:
        message = tr("Searching for an icon took longer than %1 seconds and was stopped. Check "
                     "your connection and try again.")
                      .arg(SearchDeadline.count());
        break;
    case IconSearchStatus::NetworkError:
        message = tr("Could not reach the icon sources (%1). Check your connection and try again.")
                      .arg(QString::fromStdString(result.error));
        break;
    case IconSearchStatus::Cancelled:
        return;
    case IconSearchStatus::InternalError:
        message = tr("Searching for an icon failed unexpectedly: %1")
                      .arg(QString::fromStdString(result.error));
        break;
    }
    LOG_WARNING(Frontend, "Icon search failed: {}", message.toStdString());
    QMessageBox::warning(parent_widget, tr("Find Icon"), message);
}
