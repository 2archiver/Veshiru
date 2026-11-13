// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "eglantine_dialog.h"

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>

#include <QApplication>
#include <QCursor>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QHBoxLayout>
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
            .arg(QString::fromStdString(GetLegacyPath(EglantineDir).string()),
                 QString::fromStdString(GetLegacyPath(LegacyEdenDir).string())),
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

    const fs::path picked{selected.toStdString()};
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
            .arg(QString::fromStdString((picked / kSaveSuffix).string())));
    return {};
}

std::filesystem::path GetSaveDir(const std::filesystem::path& root, const std::string& user_id,
                                 u64 program_id) {
    return root / kSaveSuffix / user_id /
           fmt::format("{:016X}", program_id);
}

bool IsLink(const std::filesystem::path& path) {
    return Common::FS::IsSymlink(path);
}

bool HandleAlreadyLinked(const std::filesystem::path& veshiru_dir,
                         const std::filesystem::path& eglantine_dir, QWidget* parent) {
    const bool veshiru_link = IsLink(veshiru_dir);
    const bool eglantine_link = IsLink(eglantine_dir);
    if (!veshiru_link && !eglantine_link) {
        return false;
    }

    const auto answer = QMessageBox::question(
        parent, QObject::tr("Already Linked"),
        QObject::tr("This title is already linked to Eglantine. Would you like to unlink it?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return true;
    }

    const fs::path linked = veshiru_link ? veshiru_dir : eglantine_dir;
    const fs::path orig = veshiru_link ? eglantine_dir : veshiru_dir;

    // NB: do NOT use remove_all here; on Windows it would delete the link's *target* contents
    // instead of the link itself. fs::remove removes just the reparse point.
    try {
        fs::remove(linked);
    } catch (const std::exception& e) {
        QMessageBox::critical(parent, QObject::tr("Failed to unlink"),
                              QObject::tr("Could not remove the link at:\n\t%1\n\n%2")
                                  .arg(QString::fromStdString(linked.string()), e.what()));
        return true;
    }

    try {
        fs::copy(orig, linked, fs::copy_options::recursive);
    } catch (const std::exception& e) {
        // Nothing is lost: the original is still intact at its own path.
        QMessageBox::critical(parent, QObject::tr("Failed to copy save data"),
                              QObject::tr("The link was removed, but the save data could not be "
                                          "copied back to:\n\t%1\n\n%2\n\nThe original data is "
                                          "still intact at:\n\t%3")
                                  .arg(QString::fromStdString(linked.string()), e.what(),
                                       QString::fromStdString(orig.string())));
        return true;
    }

    QMessageBox::information(parent, QObject::tr("Unlink Successful"),
                             QObject::tr("Successfully unlinked Eglantine save data. Save data "
                                         "has been kept intact."));
    return true;
}

} // namespace EglantineLink

namespace {

// Relative path -> file size, for every regular file in the tree.
std::map<std::string, u64> ListTree(const fs::path& root) {
    std::map<std::string, u64> result;
    std::error_code ec;
    for (fs::recursive_directory_iterator it{root, ec}, end; !ec && it != end; it.increment(ec)) {
        const auto type = it->symlink_status().type();
        if (Common::FS::IsSymlink(it->path()) ||
            (type != fs::file_type::directory && type != fs::file_type::regular)) {
            throw std::runtime_error("Save data contains a link or special file: " +
                                     it->path().string());
        }
        if (type != fs::file_type::regular) {
            continue;
        }
        const auto size = it->file_size(ec);
        if (ec) {
            throw std::runtime_error("Could not stat " + it->path().string() + ": " +
                                     ec.message());
        }
        auto rel = fs::relative(it->path(), root, ec);
        if (ec) {
            throw std::runtime_error("Could not relativize " + it->path().string());
        }
        result.emplace(rel.generic_string(), size);
    }
    if (ec) {
        throw std::runtime_error("Could not scan " + root.string() + ": " + ec.message());
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
            throw std::runtime_error("Could not remove the linked directory " + dir.string() +
                                     ": " + ec.message());
        }
        return {};
    }
    if (!fs::is_directory(dir)) {
        return {};
    }
    const fs::path backup = dir.parent_path() /
                            (dir.filename().string() + ".veshiru-backup-" + Timestamp().toStdString());
    std::error_code ec;
    fs::rename(dir, backup, ec);
    if (ec) {
        throw std::runtime_error("Could not move the existing save directory aside to " +
                                 backup.string() + ": " + ec.message());
    }
    return backup;
}

// Copies src to dst with verification. dst must not exist yet. Throws std::runtime_error
// describing the failing step on error; the partially written dst is removed.
void VerifiedCopy(const fs::path& src, const fs::path& dst) {
    fs::create_directories(dst.parent_path());

    const auto before = ListTree(src);

    // Stage into a sibling temporary directory so dst is never half-written.
    const fs::path staging =
        dst.parent_path() / (dst.filename().string() + ".veilink-tmp-" + Timestamp().toStdString());
    std::error_code ec;
    fs::remove_all(staging, ec);

    try {
        fs::copy(src, staging, fs::copy_options::recursive);
        const auto after = ListTree(staging);
        if (before != after) {
            throw std::runtime_error("Copied save data failed verification (file list or sizes "
                                     "differ from the source). The original data was not "
                                     "modified.");
        }
        fs::rename(staging, dst, ec);
        if (ec) {
            throw std::runtime_error("Could not move the verified save copy into place at " +
                                     dst.string() + ": " + ec.message() +
                                     ". A complete copy is staged at: " + staging.string());
        }
    } catch (...) {
        fs::remove_all(staging, ec);
        throw;
    }
}

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
                "No Veshiru save data was found for this game at:\n\t" + m_veshiru.string() +
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
                              .arg(QString::fromStdString(backup.string()));
        }

        // 2. Move Veshiru's save aside and make it a junction to the Eglantine location, so
        //    both emulators reference the same files from now on.
        const fs::path veshiru_backup = m_veshiru.parent_path() /
                                        (m_veshiru.filename().string() + ".veshiru-backup-" +
                                         Timestamp().toStdString());
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
                    m_veshiru.string() +
                    "\ncould not be created, and Veshiru's save folder could not be moved back. "
                    "Your save data is at:\n\t" + veshiru_backup.string());
            }
            throw std::runtime_error(
                "Eglantine now holds a verified copy of the save data, but the link at:\n\t" +
                m_veshiru.string() +
                "\ncould not be created. Veshiru's save folder was restored; both emulators "
                "now have independent, identical copies of the save.");
        }
        fs::remove_all(veshiru_backup, ec);
        if (ec) {
            LOG_WARNING(Common_Filesystem, "Could not delete the temporary backup {}", veshiru_backup.string());
        }

        LOG_INFO(Common_Filesystem, "Save data for title transferred Veshiru -> Eglantine: {}",
                 m_veshiru.string());
    } catch (const std::exception& e) {
        LOG_ERROR(Common_Filesystem, "Save transfer Veshiru -> Eglantine failed: {}", e.what());
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, tr("Failed to link save data"),
                              tr("The save data could not be transferred.\n\n%1").arg(e.what()));
        return;
    }
    QApplication::restoreOverrideCursor();

    QString message = tr("Save data has been linked.\nVeshiru and Eglantine now reference the "
                         "same save files.");
    if (!backup_note.isEmpty()) {
        message += "\n\n" + backup_note;
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
                "No Eglantine save data was found for this game at:\n\t" + m_eglantine.string() +
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
                "The link at:\n\t" + m_veshiru.string() +
                "\ncould not be created. Eglantine's save data was not modified, and Veshiru's "
                "save folder was restored.");
        }

        LOG_INFO(Common_Filesystem, "Save data for title transferred Eglantine -> Veshiru: {}",
                 m_eglantine.string());
    } catch (const std::exception& e) {
        LOG_ERROR(Common_Filesystem, "Save transfer Eglantine -> Veshiru failed: {}", e.what());
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, tr("Failed to link save data"),
                              tr("The save data could not be transferred.\n\n%1").arg(e.what()));
        return;
    }
    QApplication::restoreOverrideCursor();

    QString message = tr("Save data has been linked.\nVeshiru and Eglantine now reference the "
                         "same save files.");
    if (!backup.empty()) {
        message += tr("\n\nThe previous Veshiru save data was moved to:\n\t%1")
                       .arg(QString::fromStdString(backup.string()));
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
