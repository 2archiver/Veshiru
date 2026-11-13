// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <string>

#include <QDialog>

#include "common/common_types.h"

// Save transfer between Veshiru and Eglantine.
//
// Both emulators store Switch saves in the Eden/Ryujinx-free NAND layout:
//
//     <profile>/nand/user/save/0000000000000000/<user_id>/<PROGRAM_ID>
//
// so no save parser is required - the transfer is a verified copy plus a junction on the
// Veshiru side pointing at the Eglantine save (same topology the old Ryujinx link used).
// Every step reports its own errors, the dialog stays open on failure and the replaced
// directory is moved aside to a timestamped *.veshiru-backup-* sibling instead of being
// deleted outright.
namespace EglantineLink {

/// Finds the Eglantine (or build 17 "eden" lineage) profile root that contains a
/// nand/user/save tree. Returns an empty path if there is none.
std::filesystem::path FindEglantineRoot();

/// Like FindEglantineRoot, but asks the user for a portable Eglantine folder if no
/// installation could be detected. Returns an empty path if none was found or selected.
std::filesystem::path ResolveEglantineRoot(QWidget* parent);

/// Expands a profile root to the save directory for a given user/title. The directory
/// itself may not exist yet (the title may never have been played there).
std::filesystem::path GetSaveDir(const std::filesystem::path& root, const std::string& user_id,
                                 u64 program_id);

/// True for symlinks and Windows junctions (reparse points).
bool IsLink(const std::filesystem::path& path);

/// Handles the "this title is already linked" case. Returns true if the title was already
/// linked and the user has been dealt with (unlinked or declined) and the caller should
/// stop. Returns false if nothing is linked yet and the caller may proceed with linking.
bool HandleAlreadyLinked(const std::filesystem::path& veshiru_dir,
                         const std::filesystem::path& eglantine_dir, QWidget* parent);

} // namespace EglantineLink

class EglantineDialog : public QDialog {
    Q_OBJECT

public:
    /// \p eglantine_root is the detected profile root (used only for messages); the
    /// directories are passed explicitly so the caller stays in charge of resolution.
    explicit EglantineDialog(std::filesystem::path veshiru_dir,
                             std::filesystem::path eglantine_dir, QWidget* parent = nullptr);
    ~EglantineDialog() override;

private slots:
    void FromVeshiru();
    void FromEglantine();

private:
    std::filesystem::path m_veshiru;
    std::filesystem::path m_eglantine;
};
