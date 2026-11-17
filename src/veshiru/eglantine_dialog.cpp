// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "eglantine_dialog.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <QApplication>
#include <QCryptographicHash>
#include <QCursor>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/fs/symlink.h"
#include "common/logging.h"

using namespace Common::FS;

namespace fs = std::filesystem;

namespace EglantineLink {

namespace {
const fs::path kSaveSuffix = fs::path{"nand"} / "user" / "save" / "0000000000000000";
const fs::path kProfilesDatSuffix =
    fs::path{"nand"} / "system" / "save" / "8000000000000010" / "su" / "avators" / "profiles.dat";

QString PathToQString(const fs::path& path) {
    return QString::fromStdString(Common::FS::PathToUTF8String(path));
}

fs::path QStringToPath(const QString& text) {
#ifdef _WIN32
    return fs::path{text.toStdWString()};
#else
    return fs::path{text.toStdString()};
#endif
}

bool IsValidUserIdDirName(const std::string& name) {
    if (name.size() != 32) {
        return false;
    }
    bool any_nonzero = false;
    for (unsigned char ch : name) {
        if (!std::isxdigit(ch)) {
            return false;
        }
        if (ch != '0') {
            any_nonzero = true;
        }
    }
    return any_nonzero;
}

std::string NormalizeUpperHex(std::string s) {
    std::ranges::transform(s, s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return s;
}

struct EglantineProfileEntry {
    std::string user_id;
    std::string username;
    bool has_title_save{};
};

// Parses Eglantine's Account profiles.dat (0x10-byte header + 8 * 0xC8 UserRaw entries).
std::vector<EglantineProfileEntry> ReadEglantineProfilesDat(const fs::path& root) {
    constexpr std::size_t kHeaderBytes = 0x10;
    constexpr std::size_t kUserStride = 0xC8;
    constexpr std::size_t kMaxUsers = 8;
    constexpr std::size_t kUsernameOffset = 0x28; // uuid (0x10) + uuid2 (0x10) + timestamp (0x08)
    constexpr std::size_t kUsernameBytes = 0x20;

    std::vector<EglantineProfileEntry> entries;
    std::ifstream file(root / kProfilesDatSuffix, std::ios::binary);
    if (!file) {
        return entries;
    }
    std::array<u8, kHeaderBytes + kUserStride * kMaxUsers> buf{};
    file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (static_cast<std::size_t>(file.gcount()) < buf.size()) {
        return entries;
    }

    for (std::size_t slot = 0; slot < kMaxUsers; ++slot) {
        const u8* user = buf.data() + kHeaderBytes + slot * kUserStride;
        u64 lo = 0;
        u64 hi = 0;
        std::memcpy(&lo, user, sizeof(u64));
        std::memcpy(&hi, user + sizeof(u64), sizeof(u64));
        if (lo == 0 && hi == 0) {
            continue;
        }
        const std::string id = fmt::format("{:016X}{:016X}", hi, lo);
        const char* raw_name = reinterpret_cast<const char*>(user + kUsernameOffset);
        std::size_t name_len = 0;
        while (name_len < kUsernameBytes && raw_name[name_len] != '\0') {
            ++name_len;
        }
        entries.push_back(EglantineProfileEntry{
            .user_id = id,
            .username = std::string(raw_name, name_len),
        });
    }
    return entries;
}

bool LooksLikeEglantineRoot(const fs::path& root) {
    std::error_code ec;
    return fs::is_directory(root / kSaveSuffix, ec);
}
} // namespace

std::filesystem::path FindEglantineRoot() {
    // Eglantine build 17 kept its saves in the legacy "eden" data folder; later builds moved to
    // their own. Prefer the current layout, fall back to the lineage one.
    for (const EmuPath legacy : {EglantineDir, LegacyEdenDir}) {
        const fs::path root = GetLegacyPath(legacy);
        if (LooksLikeEglantineRoot(root)) {
            return root;
        }
    }
    return {};
}

std::filesystem::path ResolveEglantineRoot(QWidget* parent) {
    if (const fs::path found = FindEglantineRoot(); !found.empty()) {
        return found;
    }

    const QMessageBox::StandardButton answer = QMessageBox::question(
        parent, QObject::tr("Could not find Eglantine installation"),
        QObject::tr("Could not find an Eglantine installation.\n\nVeshiru looked for the "
                    "following folders:\n\t%1\n\t%2\n\nIf you use Eglantine in portable mode, "
                    "would you like to select its folder manually?")
            .arg(PathToQString(GetLegacyPath(EglantineDir)),
                 PathToQString(GetLegacyPath(LegacyEdenDir))),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    if (answer != QMessageBox::Yes) {
        QMessageBox::information(
            parent, QObject::tr("Save transfer cancelled"),
            QObject::tr("No Eglantine installation was found, so the save data cannot be "
                        "transferred right now."));
        return {};
    }

    const QString selected = QFileDialog::getExistingDirectory(
        parent, QObject::tr("Eglantine Portable Location"), QDir::homePath());
    if (selected.isEmpty()) {
        return {};
    }

    const fs::path picked = QStringToPath(selected);
    // Accept the picked folder itself, or its "eglantine"/"eden" children (the picked folder may
    // be the parent of a portable profile).
    for (const fs::path& candidate :
         {picked, picked / "eglantine", picked / "eden"}) {
        if (LooksLikeEglantineRoot(candidate)) {
            return candidate;
        }
    }

    QMessageBox::critical(
        parent, QObject::tr("Not a valid Eglantine folder"),
        QObject::tr("The folder you selected does not contain an Eglantine NAND layout.\n\n"
                    "Expected to find:\n\t%1")
            .arg(PathToQString(picked / kSaveSuffix)));
    return {};
}

std::string ResolveEglantineUserId(const std::filesystem::path& eglantine_root,
                                   const std::string& veshiru_user_id, u64 program_id,
                                   QWidget* parent) {
    const fs::path save_root = eglantine_root / kSaveSuffix;
    const std::string hex_program = fmt::format("{:016X}", program_id);
    std::error_code ec;

    // 1. Exact UUID match already exists in Eglantine (e.g. migrated profile).
    if (!veshiru_user_id.empty() && fs::is_directory(save_root / veshiru_user_id, ec)) {
        return veshiru_user_id;
    }

    // 2. Gather Eglantine profiles from profiles.dat and existing save user directories.
    std::vector<EglantineProfileEntry> candidates = ReadEglantineProfilesDat(eglantine_root);
    auto find_candidate = [&](const std::string& id) {
        const std::string upper = NormalizeUpperHex(id);
        return std::ranges::find_if(candidates, [&](const EglantineProfileEntry& e) {
            return NormalizeUpperHex(e.user_id) == upper;
        });
    };

    if (fs::is_directory(save_root, ec)) {
        for (const auto& entry : fs::directory_iterator(save_root, ec)) {
            if (!entry.is_directory(ec)) {
                continue;
            }
            const std::string dir_name = Common::FS::PathToUTF8String(entry.path().filename());
            if (!IsValidUserIdDirName(dir_name)) {
                continue;
            }
            if (auto it = find_candidate(dir_name); it != candidates.end()) {
                // Prefer the exact directory casing found on disk.
                it->user_id = dir_name;
            } else {
                candidates.push_back(EglantineProfileEntry{.user_id = dir_name});
            }
        }
    }

    // Mark which Eglantine profiles already have save data for this title.
    std::vector<std::size_t> with_save;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const fs::path title_dir = save_root / candidates[i].user_id / hex_program;
        if (fs::is_directory(title_dir, ec) || Common::FS::IsSymlink(title_dir)) {
            candidates[i].has_title_save = true;
            with_save.push_back(i);
        }
    }

    // 3. Unambiguous automatic selection: single profile owning this title's save, or single
    //    Eglantine user profile overall.
    if (with_save.size() == 1) {
        return candidates[with_save[0]].user_id;
    }
    if (candidates.size() == 1) {
        return candidates[0].user_id;
    }
    if (candidates.empty()) {
        return veshiru_user_id;
    }

    // 4. Multiple Eglantine profiles exist and none matches Veshiru's UUID directly: let the user
    //    choose which Eglantine profile to link.
    QStringList labels;
    labels.reserve(static_cast<int>(candidates.size()));
    int default_index = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        QString label = c.username.empty()
                            ? QString::fromStdString(c.user_id)
                            : QStringLiteral("%1 (%2)").arg(QString::fromUtf8(c.username),
                                                            QString::fromStdString(c.user_id));
        if (c.has_title_save) {
            label += QObject::tr(" [has save]");
            if (default_index == 0) {
                default_index = static_cast<int>(i);
            }
        }
        labels.push_back(label);
    }

    bool ok = false;
    const QString chosen = QInputDialog::getItem(
        parent, QObject::tr("Select Eglantine User Profile"),
        QObject::tr("Multiple Eglantine user profiles were found with different UUIDs from your "
                    "Veshiru profile.\nChoose which Eglantine profile to link for this title:"),
        labels, default_index, false, &ok);
    if (!ok) {
        return {};
    }
    const int idx = labels.indexOf(chosen);
    if (idx < 0 || static_cast<std::size_t>(idx) >= candidates.size()) {
        return {};
    }
    return candidates[static_cast<std::size_t>(idx)].user_id;
}

std::filesystem::path GetSaveDir(const std::filesystem::path& root, const std::string& user_id,
                                 u64 program_id) {
    return root / kSaveSuffix / user_id /
           fmt::format("{:016X}", program_id);
}

bool IsLink(const std::filesystem::path& path) {
    return Common::FS::IsSymlink(path);
}

} // namespace EglantineLink

namespace {

struct FileDigest {
    u64 size{};
    QByteArray sha256;
    bool operator==(const FileDigest&) const = default;
};

fs::path AppendSiblingSuffix(const fs::path& dir, const std::string& suffix) {
    fs::path result = dir.has_filename() ? dir : dir.parent_path();
    result += suffix;
    return result;
}

std::string PathToUtf8(const fs::path& path) {
    return Common::FS::PathToUTF8String(path);
}

QString PathToDisplayQString(const fs::path& path) {
    return QString::fromStdString(PathToUtf8(path));
}

QByteArray HashFileSha256(const fs::path& path) {
    QFile file(PathToDisplayQString(path));
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Could not open " + PathToUtf8(path) +
                                 " for checksum verification");
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&file)) {
        throw std::runtime_error("Could not read " + PathToUtf8(path) +
                                 " for checksum verification");
    }
    return hash.result();
}

// Relative path -> (file size, SHA-256 digest), for every regular file in the tree.
std::map<std::string, FileDigest> ListTree(const fs::path& root) {
    std::map<std::string, FileDigest> result;
    std::error_code ec;
    for (fs::recursive_directory_iterator it{root, ec}, end; !ec && it != end; it.increment(ec)) {
        const auto type = it->symlink_status().type();
        if (Common::FS::IsSymlink(it->path()) ||
            (type != fs::file_type::directory && type != fs::file_type::regular)) {
            throw std::runtime_error("Save data contains a link or special file: " +
                                     PathToUtf8(it->path()));
        }
        if (type != fs::file_type::regular) {
            continue;
        }
        const auto size = it->file_size(ec);
        if (ec) {
            throw std::runtime_error("Could not stat " + PathToUtf8(it->path()) + ": " +
                                     ec.message());
        }
        auto rel = fs::relative(it->path(), root, ec);
        if (ec) {
            throw std::runtime_error("Could not relativize " + PathToUtf8(it->path()));
        }
        const auto rel_u8 = rel.generic_u8string();
        std::string rel_key(reinterpret_cast<const char*>(rel_u8.data()), rel_u8.size());
        result.emplace(std::move(rel_key), FileDigest{size, HashFileSha256(it->path())});
    }
    if (ec) {
        throw std::runtime_error("Could not scan " + PathToUtf8(root) + ": " + ec.message());
    }
    return result;
}

QString Timestamp() {
    return QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
}

// Moves a directory aside to a timestamped backup sibling. Returns the backup path, or an
// empty path if the directory did not exist (nothing to back up).
fs::path BackupDirectory(const fs::path& dir) {
    if (Common::FS::IsSymlink(dir)) {
        // A junction is just a reparse point; remove it instead of backing it up.
        std::error_code ec;
        fs::remove(dir, ec);
        if (ec) {
            throw std::runtime_error("Could not remove the linked directory " + PathToUtf8(dir) +
                                     ": " + ec.message());
        }
        return {};
    }
    if (!fs::is_directory(dir)) {
        return {};
    }
    const fs::path backup = AppendSiblingSuffix(dir, ".veshiru-backup-" + Timestamp().toStdString());
    std::error_code ec;
    fs::rename(dir, backup, ec);
    if (ec) {
        throw std::runtime_error("Could not move the existing save directory aside to " +
                                 PathToUtf8(backup) + ": " + ec.message());
    }
    return backup;
}

// Copies src to dst with SHA-256 verification. dst must not exist yet. Throws std::runtime_error
// describing the failing step on error; the partially written dst is removed.
void VerifiedCopy(const fs::path& src, const fs::path& dst) {
    fs::create_directories(dst.parent_path());

    const auto before = ListTree(src);

    // Stage into a sibling temporary directory so dst is never half-written.
    const fs::path staging = AppendSiblingSuffix(dst, ".veilink-tmp-" + Timestamp().toStdString());
    std::error_code ec;
    fs::remove_all(staging, ec);

    try {
        fs::copy(src, staging, fs::copy_options::recursive);
        const auto after = ListTree(staging);
        if (before != after) {
            throw std::runtime_error(
                "Copied save data failed SHA-256 verification (file list, sizes, or contents "
                "differ from the source). The original data was not modified.");
        }
        fs::rename(staging, dst, ec);
        if (ec) {
            throw std::runtime_error("Could not move the verified save copy into place at " +
                                     PathToUtf8(dst) + ": " + ec.message() +
                                     ". A complete copy is staged at: " + PathToUtf8(staging));
        }
    } catch (...) {
        fs::remove_all(staging, ec);
        throw;
    }
}

} // namespace

namespace EglantineLink {

bool HandleAlreadyLinked(const std::filesystem::path& veshiru_dir,
                         const std::filesystem::path& eglantine_dir, QWidget* parent) {
    const bool veshiru_link = IsLink(veshiru_dir);
    const bool eglantine_link = IsLink(eglantine_dir);
    if (!veshiru_link && !eglantine_link) {
        return false;
    }

    const fs::path linked = veshiru_link ? veshiru_dir : eglantine_dir;
    const fs::path fallback_orig = veshiru_link ? eglantine_dir : veshiru_dir;
    const auto resolved_target = Common::FS::ReadSymlinkTarget(linked);
    const fs::path orig = resolved_target && !resolved_target->empty() ? *resolved_target : fallback_orig;

    const auto answer = QMessageBox::question(
        parent, QObject::tr("Already Linked"),
        QObject::tr("This title is already linked to:\n\t%1\n\nWould you like to unlink it and "
                    "keep independent verified copies?")
            .arg(PathToDisplayQString(orig)),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return true;
    }

    if (!fs::is_directory(orig)) {
        QMessageBox::critical(
            parent, QObject::tr("Failed to unlink"),
            QObject::tr("The linked save target directory does not exist:\n\t%1\n\nThe link at "
                        "%2 was kept untouched so you can inspect or repair the target path.")
                .arg(PathToDisplayQString(orig), PathToDisplayQString(linked)));
        return true;
    }

    // Stage and SHA-256-verify the copy BEFORE removing the link so a failed copy never leaves
    // the save path empty.
    const fs::path staged_unlink =
        AppendSiblingSuffix(linked, ".unlink-verified-" + Timestamp().toStdString());
    std::error_code cleanup_ec;
    fs::remove_all(staged_unlink, cleanup_ec);
    try {
        VerifiedCopy(orig, staged_unlink);
    } catch (const std::exception& e) {
        fs::remove_all(staged_unlink, cleanup_ec);
        QMessageBox::critical(
            parent, QObject::tr("Failed to copy save data"),
            QObject::tr("Could not stage a verified copy of the linked save data before "
                        "unlinking:\n\n%1\n\nThe existing link at %2 was left untouched.")
                .arg(QString::fromUtf8(e.what()), PathToDisplayQString(linked)));
        return true;
    }

    // NB: do NOT use remove_all on linked; on Windows remove_all deletes the junction's target
    // contents. fs::remove removes only the reparse point.
    std::error_code ec;
    fs::remove(linked, ec);
    if (ec) {
        fs::remove_all(staged_unlink, cleanup_ec);
        QMessageBox::critical(parent, QObject::tr("Failed to unlink"),
                              QObject::tr("Could not remove the link at:\n\t%1\n\n%2")
                                  .arg(PathToDisplayQString(linked),
                                       QString::fromStdString(ec.message())));
        return true;
    }

    fs::rename(staged_unlink, linked, ec);
    if (ec) {
        // Attempt to restore the link so Veshiru still points at the intact target.
        (void)Common::FS::CreateSymlink(orig, linked);
        QMessageBox::critical(
            parent, QObject::tr("Failed to finalize unlink"),
            QObject::tr("Could not move the verified copy into:\n\t%1\n\n%2\n\nA complete "
                        "verified copy remains at:\n\t%3")
                .arg(PathToDisplayQString(linked), QString::fromStdString(ec.message()),
                     PathToDisplayQString(staged_unlink)));
        return true;
    }

    QMessageBox::information(parent, QObject::tr("Unlink Successful"),
                             QObject::tr("Successfully unlinked save data. Verified independent "
                                         "copies have been kept intact."));
    return true;
}

} // namespace EglantineLink

namespace {

// Copies \p src_root to \p dst_root through VerifiedCopy after moving any existing dst_root
// aside to a backup. Returns the backup path (empty if there was nothing to back up).
fs::path ReplaceWithBackup(const fs::path& src_root, const fs::path& dst_root) {
    const fs::path backup = BackupDirectory(dst_root);
    try {
        VerifiedCopy(src_root, dst_root);
    } catch (...) {
        // Try to restore the backup so the destination side is not left missing.
        if (!backup.empty() && !fs::exists(dst_root)) {
            std::error_code ec;
            fs::rename(backup, dst_root, ec);
        }
        throw;
    }
    return backup;
}

} // namespace

void EglantineDialog::FromVeshiru() {
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QString backup_note;
    try {
        if (!fs::is_directory(m_veshiru)) {
            throw std::runtime_error(
                "No Veshiru save data was found for this game at:\n\t" + PathToUtf8(m_veshiru) +
                "\n\nLaunch the game in Veshiru (or use \"From Eglantine\") first.");
        }
        if (!fs::is_directory(m_veshiru.parent_path())) {
            fs::create_directories(m_veshiru.parent_path());
        }

        // 1. Copy the verified data into a sibling staging area of the Eglantine save, then
        //    replace the Eglantine save directory with it.
        const fs::path backup = ReplaceWithBackup(m_veshiru, m_eglantine);
        if (!backup.empty()) {
            backup_note = tr("The previous Eglantine save data was moved to:\n\t%1")
                              .arg(PathToDisplayQString(backup));
        }

        // 2. Move Veshiru's save aside and make it a junction to the Eglantine location, so
        //    both emulators reference the same files from now on.
        const fs::path veshiru_backup =
            AppendSiblingSuffix(m_veshiru, ".veshiru-backup-" + Timestamp().toStdString());
        std::error_code ec;
        fs::rename(m_veshiru, veshiru_backup, ec);
        if (ec) {
            throw std::runtime_error("Eglantine now holds a verified copy of the save data, but "
                                     "Veshiru's save folder could not be replaced: " +
                                     ec.message());
        }
        if (!Common::FS::CreateSymlink(m_eglantine, m_veshiru)) {
            std::error_code restore_ec;
            fs::rename(veshiru_backup, m_veshiru, restore_ec);
            if (restore_ec) {
                throw std::runtime_error(
                    "Eglantine now holds a verified copy of the save data, but the link at:\n\t" +
                    PathToUtf8(m_veshiru) +
                    "\ncould not be created, and Veshiru's save folder could not be moved back. "
                    "Your save data is at:\n\t" +
                    PathToUtf8(veshiru_backup));
            }
            throw std::runtime_error(
                "Eglantine now holds a verified copy of the save data, but the link at:\n\t" +
                PathToUtf8(m_veshiru) +
                "\ncould not be created. Veshiru's save folder was restored; both emulators "
                "now have independent, identical copies of the save.");
        }
        fs::remove_all(veshiru_backup, ec);
        if (ec) {
            LOG_WARNING(Common_Filesystem, "Could not delete the temporary backup {}",
                        PathToUtf8(veshiru_backup));
        }

        LOG_INFO(Common_Filesystem, "Save data for title transferred Veshiru -> Eglantine: {}",
                 PathToUtf8(m_veshiru));
    } catch (const std::exception& e) {
        LOG_ERROR(Common_Filesystem, "Save transfer Veshiru -> Eglantine failed: {}", e.what());
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, tr("Failed to link save data"),
                              tr("The save data could not be transferred.\n\n%1")
                                  .arg(QString::fromUtf8(e.what())));
        return;
    }
    QApplication::restoreOverrideCursor();

    QString message = tr("Save data has been linked.\nVeshiru and Eglantine now reference the "
                         "same save files.");
    if (!backup_note.isEmpty()) {
        message += QStringLiteral("\n\n") + backup_note;
    }
    QMessageBox::information(this, tr("Save data linked"), message);
    accept();
}

void EglantineDialog::FromEglantine() {
    QApplication::setOverrideCursor(Qt::WaitCursor);
    fs::path backup;
    try {
        if (!fs::is_directory(m_eglantine)) {
            throw std::runtime_error(
                "No Eglantine save data was found for this game at:\n\t" + PathToUtf8(m_eglantine) +
                "\n\nLaunch the game in Eglantine (or use \"From Veshiru\") first.");
        }

        // The Veshiru save directory becomes a junction to Eglantine's, so Eglantine's own
        // files are the single source of truth and nothing needs to be copied.
        if (!fs::is_directory(m_veshiru.parent_path())) {
            fs::create_directories(m_veshiru.parent_path());
        }
        backup = BackupDirectory(m_veshiru);

        if (!Common::FS::CreateSymlink(m_eglantine, m_veshiru)) {
            // Restore the previous Veshiru save directory.
            if (!backup.empty()) {
                std::error_code ec;
                fs::rename(backup, m_veshiru, ec);
            }
            throw std::runtime_error(
                "The link at:\n\t" + PathToUtf8(m_veshiru) +
                "\ncould not be created. Eglantine's save data was not modified, and Veshiru's "
                "save folder was restored.");
        }

        LOG_INFO(Common_Filesystem, "Save data for title transferred Eglantine -> Veshiru: {}",
                 PathToUtf8(m_eglantine));
    } catch (const std::exception& e) {
        LOG_ERROR(Common_Filesystem, "Save transfer Eglantine -> Veshiru failed: {}", e.what());
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, tr("Failed to link save data"),
                              tr("The save data could not be transferred.\n\n%1")
                                  .arg(QString::fromUtf8(e.what())));
        return;
    }
    QApplication::restoreOverrideCursor();

    QString message = tr("Save data has been linked.\nVeshiru and Eglantine now reference the "
                         "same save files.");
    if (!backup.empty()) {
        message += tr("\n\nThe previous Veshiru save data was moved to:\n\t%1")
                       .arg(PathToDisplayQString(backup));
    }
    QMessageBox::information(this, tr("Save data linked"), message);
    accept();
}

EglantineDialog::EglantineDialog(std::filesystem::path veshiru_dir,
                                 std::filesystem::path eglantine_dir, QWidget* parent)
    : QDialog(parent), m_veshiru(std::move(veshiru_dir)),
      m_eglantine(std::move(eglantine_dir)) {
    setObjectName(QStringLiteral("EglantineDialog"));
    setWindowTitle(tr("Eglantine Link"));
    setMinimumWidth(440);

    auto* layout = new QVBoxLayout(this);

    auto* label = new QLabel(
        tr("Linking save data to Eglantine lets both Veshiru and Eglantine reference the same "
           "save files for your games.\n\n"
           "By selecting \"From Veshiru\", previous save data stored in Eglantine will be moved "
           "to a timestamped backup, and vice versa for \"From Eglantine\".\n\n"
           "Close Veshiru and Eglantine before linking."),
        this);
    label->setWordWrap(true);
    layout->addWidget(label);

    auto* buttons = new QHBoxLayout;
    auto* from_veshiru = new QPushButton(tr("From Veshiru"), this);
    auto* from_eglantine = new QPushButton(tr("From Eglantine"), this);
    auto* cancel = new QPushButton(tr("Cancel"), this);
    buttons->addWidget(from_veshiru);
    buttons->addWidget(from_eglantine);
    buttons->addWidget(cancel);
    layout->addLayout(buttons);

    connect(from_veshiru, &QPushButton::clicked, this, &EglantineDialog::FromVeshiru);
    connect(from_eglantine, &QPushButton::clicked, this, &EglantineDialog::FromEglantine);
    connect(cancel, &QPushButton::clicked, this, &EglantineDialog::reject);
}

EglantineDialog::~EglantineDialog() = default;
