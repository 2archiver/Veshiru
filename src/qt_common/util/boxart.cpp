// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stop_token>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common/httplib.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QObject>
#include <fmt/format.h>
#include <zstd.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread.h"
#include "qt_common/util/boxart.h"
#ifdef VESHIRU_BUNDLED_OPENSSL
#include <openssl/cert.h>
#endif

namespace QtCommon::Boxart {
namespace {

constexpr char NemyuBoxartUrl[] = "https://gitlab.com/nemyu-emulator/boxart/-/raw/main/img/";
// Eden's box art, which covers Switch 1 games.
constexpr char EdenBoxartUrl[] =
    "https://raw.githubusercontent.com/eden-emulator/boxart/refs/heads/master/img/";

constexpr std::size_t MaxQueued = 32;
constexpr std::size_t MaxCoverSize = 8 * 1024 * 1024;

bool IsValidSlug(std::string_view slug) {
    // The repositories also hold their logos and a placeholder, which are not box art.
    if (slug.empty() || slug == "placeholder" || slug == "nemyu" || slug.size() > 200) {
        return false;
    }
    return std::all_of(slug.begin(), slug.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '-' || c == '_';
    });
}

bool IsImage(std::string_view data) {
    constexpr std::string_view Png{"\x89PNG\r\n\x1a\n", 8};
    constexpr std::string_view Jpeg{"\xFF\xD8\xFF", 3};
    return data.starts_with(Png) || data.starts_with(Jpeg);
}

std::string MakeSlug(std::string_view title) {
    constexpr std::string_view EAcute = "\xC3\xA9";
    std::string slug;
    slug.reserve(title.size());
    for (std::size_t i = 0; i < title.size(); ++i) {
        char c = title[i];
        if (title.substr(i, EAcute.size()) == EAcute) {
            c = 'e';
            i += EAcute.size() - 1;
        } else if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        } else if (c == ' ') {
            c = '-';
        }
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            slug += c;
        } else if (c == '-' && !slug.empty() && slug.back() != '-') {
            slug += c;
        }
    }
    if (!slug.empty() && slug.back() == '-') {
        slug.pop_back();
    }
    return slug;
}

struct TitleEntry {
    std::string name;
    std::string cover;
};

/// The Switchbrew Switch 2 title table (:/switch2_titles.json), by program ID.
const std::unordered_map<u64, TitleEntry>& Titles() {
    static const auto titles = [] {
        std::unordered_map<u64, TitleEntry> map;
        QFile file(QStringLiteral(":/switch2_titles.json"));
        if (!file.open(QIODevice::ReadOnly)) {
            return map;
        }
        const QJsonObject json = QJsonDocument::fromJson(file.readAll()).object();
        for (auto it = json.begin(); it != json.end(); ++it) {
            bool ok = false;
            const u64 id = it.key().toULongLong(&ok, 16);
            if (!ok || it.key().size() != 16) {
                continue; // "_source" and other comments
            }
            TitleEntry entry;
            if (it.value().isString()) {
                entry.name = it.value().toString().toStdString();
            } else if (it.value().isObject()) {
                entry.name =
                    it.value().toObject().value(QStringLiteral("name")).toString().toStdString();
                entry.cover =
                    it.value().toObject().value(QStringLiteral("cover")).toString().toStdString();
            }
            map.emplace(id, std::move(entry));
        }
        return map;
    }();
    return titles;
}

// ---- The pack: every downloaded cover in one file, zstd-compressed --------------------------
// "SBXP", u32 version 1, then records: u16 slug length, the slug, u32 stored size, u32 size, the
// zstd frame. A later record of a slug replaces an earlier one; a cut-off last record is ignored
// and overwritten by the next append.

class CoverPack {
public:
    static CoverPack& Instance() {
        static CoverPack pack;
        return pack;
    }

    std::vector<u8> Read(const std::string& slug) {
        std::scoped_lock lock{mutex};
        Load();
        const auto it = index.find(slug);
        if (it == index.end()) {
            return {};
        }
        std::ifstream file(Path(), std::ios::binary);
        std::vector<u8> stored(it->second.stored);
        file.seekg(static_cast<std::streamoff>(it->second.offset));
        file.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(stored.size()));
        if (!file) {
            return {};
        }
        std::vector<u8> out(it->second.size);
        const std::size_t got =
            ZSTD_decompress(out.data(), out.size(), stored.data(), stored.size());
        if (ZSTD_isError(got) || got != out.size()) {
            return {};
        }
        return out;
    }

    bool Contains(const std::string& slug) {
        std::scoped_lock lock{mutex};
        Load();
        return index.contains(slug);
    }

    bool Add(const std::string& slug, std::string_view image) {
        std::vector<u8> stored(ZSTD_compressBound(image.size()));
        const std::size_t packed =
            ZSTD_compress(stored.data(), stored.size(), image.data(), image.size(), 9);
        if (ZSTD_isError(packed)) {
            return false;
        }
        stored.resize(packed);
        std::scoped_lock lock{mutex};
        Load();
        const auto path = Path();
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        std::fstream file;
        if (valid) {
            file.open(path, std::ios::binary | std::ios::in | std::ios::out);
        }
        if (!file.is_open()) {
            // No pack yet, or a file that is not one: a new pack.
            file.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
            file.write("SBXP\x01\0\0\0", 8);
            end = 8;
            index.clear();
            valid = true;
        }
        file.seekp(static_cast<std::streamoff>(end));
        const u16 slug_size = static_cast<u16>(slug.size());
        const u32 stored_size = static_cast<u32>(stored.size());
        const u32 size = static_cast<u32>(image.size());
        file.write(reinterpret_cast<const char*>(&slug_size), sizeof(slug_size));
        file.write(slug.data(), static_cast<std::streamsize>(slug.size()));
        file.write(reinterpret_cast<const char*>(&stored_size), sizeof(stored_size));
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        const u64 data_at = end + sizeof(slug_size) + slug.size() + 8;
        file.write(reinterpret_cast<const char*>(stored.data()),
                   static_cast<std::streamsize>(stored.size()));
        file.flush();
        if (!file) {
            return false;
        }
        index.insert_or_assign(slug, Entry{data_at, stored_size, size});
        end = data_at + stored_size;
        return true;
    }

private:
    struct Entry {
        u64 offset;
        u32 stored;
        u32 size;
    };

    static std::filesystem::path Path() {
        return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "boxart.pack";
    }

    void Load() {
        if (loaded) {
            return;
        }
        loaded = true;
        std::error_code ec;
        const u64 file_size = std::filesystem::file_size(Path(), ec);
        std::ifstream file(Path(), std::ios::binary);
        std::array<char, 8> head{};
        if (ec || !file.read(head.data(), head.size()) ||
            std::memcmp(head.data(), "SBXP\x01\0\0\0", 8) != 0) {
            return; // none yet, or another file: rewritten by the first Add
        }
        valid = true;
        end = 8;
        while (end < file_size) {
            u16 slug_size{};
            u32 stored{}, size{};
            std::string slug;
            if (!file.read(reinterpret_cast<char*>(&slug_size), sizeof(slug_size))) {
                break;
            }
            slug.resize(slug_size);
            if (!file.read(slug.data(), slug_size) ||
                !file.read(reinterpret_cast<char*>(&stored), sizeof(stored)) ||
                !file.read(reinterpret_cast<char*>(&size), sizeof(size)) || size > MaxCoverSize) {
                break;
            }
            const u64 data_at = end + sizeof(slug_size) + slug_size + 8;
            if (data_at + stored > file_size) {
                break; // cut off: the next Add writes over it
            }
            file.seekg(static_cast<std::streamoff>(data_at + stored));
            index.insert_or_assign(std::move(slug), Entry{data_at, stored, size});
            end = data_at + stored;
        }
    }

    std::mutex mutex;
    bool loaded = false;
    bool valid = false; ///< The file is a pack (else the first Add starts a new one)
    u64 end = 0;        ///< Where the next record goes (after the last whole one)
    std::unordered_map<std::string, Entry> index;
};

// ---- Downloads ------------------------------------------------------------------------------

// Fetches url into out. Returns the HTTP status, or 0 if the server could not be reached or did
// not send a PNG or JPEG image.
int Download(const std::string& url, std::string& out, std::stop_token stop_token) {
    const auto scheme_end = url.find("://");
    const auto path_start =
        scheme_end == std::string::npos ? scheme_end : url.find('/', scheme_end + 3);
    if (path_start == std::string::npos) {
        return 0;
    }
    try {
        httplib::Client client(url.substr(0, path_start));
        client.set_connection_timeout(3);
        client.set_read_timeout(5);
        client.set_follow_location(true);
        client.enable_server_certificate_verification(true);
        client.enable_server_hostname_verification(true);
#ifdef VESHIRU_BUNDLED_OPENSSL
        client.load_ca_cert_store(kCert, sizeof(kCert));
#endif
        const std::stop_callback cancel{stop_token, [&client] { client.stop(); }};
        int status = 0;
        const auto result = client.Get(
            url.substr(path_start),
            [&status](const httplib::Response& response) {
                status = response.status;
                if (status != 200) {
                    return true;
                }
                const std::string type = response.get_header_value("Content-Type");
                return (type.starts_with("image/png") || type.starts_with("image/jpeg") ||
                        type.starts_with("application/octet-stream") ||
                        type.starts_with("text/plain")) &&
                       response.get_header_value_u64("Content-Length") <= MaxCoverSize;
            },
            [&](const char* data, std::size_t length) {
                if (status != 200) {
                    return true;
                }
                if (stop_token.stop_requested() || out.size() + length > MaxCoverSize) {
                    return false;
                }
                out.append(data, length);
                return true;
            });
        if (!result || (result->status == 200 && !IsImage(out))) {
            return 0;
        }
        return result->status;
    } catch (const std::exception& e) {
        LOG_WARNING(Frontend, "Box art download from {} failed: {}", url, e.what());
        return 0;
    }
}

class Downloader {
public:
    static Downloader& Instance() {
        static Downloader instance;
        return instance;
    }

    void Push(std::string slug) {
        if (!IsValidSlug(slug) || CoverPack::Instance().Contains(slug)) {
            return;
        }
        {
            std::scoped_lock lock{mutex};
            // A full queue drops the request: the next game list refresh asks again.
            if (stopped || queue.size() >= MaxQueued || !seen.insert(slug).second) {
                return;
            }
            queue.push_back(std::move(slug));
            if (!worker.joinable()) {
                worker = std::jthread([this](std::stop_token stop_token) { Run(stop_token); });
            }
        }
        request_cv.notify_one();
    }

    void Stop() {
        std::jthread stopping;
        {
            std::scoped_lock lock{mutex};
            stopped = true;
            queue.clear();
            stopping = std::move(worker);
        }
        // Joined outside the lock, as the worker takes it to see the stop request.
    }

    void SetOnSaved(QObject* context, std::function<void(const std::string&)> callback) {
        std::scoped_lock lock{mutex};
        if (callback) {
            listeners.insert_or_assign(context, std::move(callback));
        } else {
            listeners.erase(context);
        }
    }

private:
    void Run(std::stop_token stop_token) {
        Common::SetCurrentThreadName("BoxartDownload");
        while (true) {
            std::string slug;
            {
                std::unique_lock lock{mutex};
                request_cv.wait(lock, stop_token, [this] { return !queue.empty(); });
                if (stop_token.stop_requested()) {
                    return;
                }
                slug = std::move(queue.front());
                queue.pop_front();
            }
            if (!Fetch(slug, stop_token)) {
                std::scoped_lock lock{mutex};
                seen.erase(slug); // may still exist: a later request tries again
            }
        }
    }

    bool Fetch(const std::string& slug, std::stop_token stop_token) {
        bool all_missing = true;
        for (const char* base : {NemyuBoxartUrl, EdenBoxartUrl}) {
            if (stop_token.stop_requested()) {
                return false;
            }
            const std::string url = fmt::format("{}{}.png", base, slug);
            std::string data;
            const int status = Download(url, data, stop_token);
            if (status == 200) {
                if (!CoverPack::Instance().Add(slug, data)) {
                    LOG_WARNING(Frontend, "Could not store the box art of {}", slug);
                    return false;
                }
                LOG_INFO(Frontend, "Saved box art from {}", url);
                Notify(slug);
                return true;
            }
            all_missing = all_missing && status == 404;
            LOG_INFO(Frontend, "No box art from {} (HTTP status {})", url, status);
        }
        return all_missing;
    }

    // Posted under the lock, so a listener that was just removed is never called.
    void Notify(const std::string& slug) {
        std::scoped_lock lock{mutex};
        for (const auto& [context, callback] : listeners) {
            QMetaObject::invokeMethod(
                context, [callback, slug] { callback(slug); }, Qt::QueuedConnection);
        }
    }

    std::mutex mutex;
    std::condition_variable_any request_cv;
    std::deque<std::string> queue;
    std::unordered_set<std::string> seen; ///< Queued, saved or missing everywhere this session
    bool stopped = false;
    std::unordered_map<QObject*, std::function<void(const std::string&)>> listeners;
    std::jthread worker;
};

} // Anonymous namespace

std::string GetSlug(u64 program_id, const std::string& title) {
    const auto& titles = Titles();
    if (const auto it = titles.find(program_id); it != titles.end()) {
        if (!it->second.cover.empty()) {
            return it->second.cover;
        }
        if (!it->second.name.empty()) {
            return MakeSlug(it->second.name);
        }
    }
    std::string slug = MakeSlug(title);
    return IsValidSlug(slug) ? slug : std::string{};
}

std::vector<u8> FindCover(u64 program_id, const std::string& slug) {
    QFile builtin(QStringLiteral(":/boxart/%1.png")
                      .arg(QString::fromStdString(fmt::format("{:016X}", program_id))));
    if (builtin.open(QIODevice::ReadOnly)) {
        const QByteArray bytes = builtin.readAll();
        return {bytes.begin(), bytes.end()};
    }
    if (!IsValidSlug(slug)) {
        return {};
    }
    return CoverPack::Instance().Read(slug);
}

void Prefetch(u64 program_id, const std::string& slug) {
    // Only games the repositories may have: Switch 1 and Switch 2 ones. Satsuma skips Switch 2
    // games missing from its title table; Veshiru ships a starter table, so it asks for the
    // title-derived slug too and lets a 404 settle it (once a session, as every slug is).
    const u64 platform = program_id >> 56;
    if (platform != 0x01 && platform != 0x04) {
        return;
    }
    Downloader::Instance().Push(slug);
}

void OnSaved(QObject* context, std::function<void(const std::string& slug)> on_saved) {
    Downloader::Instance().SetOnSaved(context, std::move(on_saved));
}

void Shutdown() {
    Downloader::Instance().Stop();
}

} // namespace QtCommon::Boxart
