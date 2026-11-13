// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "symlink.h"

#include <boost/filesystem.hpp>
#include <cstddef>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <fmt/format.h>
#include <windows.h>
#include <winioctl.h>
#endif

#include "common/logging.h"

namespace fs = std::filesystem;

// The sole purpose of this file is to treat symlinks like symlinks on POSIX,
// or treat them as directory junctions on Windows.
// This is because, for some inexplicable reason, Microsoft has locked symbolic
// links behind a "security policy", whereas directory junctions--functionally identical
// for directories, by the way--are not. Why? I don't know.
//
// Junctions are created with a native FSCTL_SET_REPARSE_POINT call instead of shelling
// out to "mklink /J": the shell-out is unreliable (it flashes a console window, depends on
// the inherited console existing, and reports failure only through the exit code of cmd),
// which is why naive linking code silently fails inside a GUI application.

namespace Common::FS {

#ifdef _WIN32
namespace {

// Creates a directory junction at \p link pointing at \p target without touching the console.
bool CreateJunction(const fs::path& link, const fs::path& target) {
    if (!CreateDirectoryW(link.c_str(), nullptr)) {
        const auto error = GetLastError();
        LOG_ERROR(Common_Filesystem, "CreateDirectoryW({}) failed with error {}", link.string(),
                  error);
        return false;
    }

    // The substitute name is what the filesystem resolves internally; it must be of the
    // form \\?\<absolute path>. The print name is what is shown to users.
    std::wstring substitute = L"\\??\\" + target.wstring();
    const std::wstring print = target.wstring();

    const size_t substitute_bytes = substitute.size() * sizeof(wchar_t);
    const size_t print_bytes = print.size() * sizeof(wchar_t);
    // PathBuffer holds the substitute and print names, each followed by a null terminator.
    const size_t names_bytes = substitute_bytes + sizeof(wchar_t) + print_bytes + sizeof(wchar_t);
    // Tag-specific data: 4 USHORT header fields + PathBuffer content.
    const size_t reparse_data_bytes = 4 * sizeof(USHORT) + names_bytes;
    // Tag-agnostic header: ReparseTag + ReparseDataLength + Reserved.
    const size_t buffer_bytes = 2 * sizeof(USHORT) + sizeof(ULONG) + reparse_data_bytes;

    // REPARSE_DATA_BUFFER layout for IO_REPARSE_TAG_MOUNT_POINT.
    std::vector<char> buffer(buffer_bytes);
    auto* reparse = reinterpret_cast<REPARSE_DATA_BUFFER*>(buffer.data());
    reparse->ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
    reparse->ReparseDataLength = static_cast<USHORT>(reparse_data_bytes);
    reparse->Reserved = 0;
    reparse->MountPointReparseBuffer.SubstituteNameOffset = 0;
    reparse->MountPointReparseBuffer.SubstituteNameLength =
        static_cast<USHORT>(substitute_bytes);
    reparse->MountPointReparseBuffer.PrintNameOffset =
        static_cast<USHORT>(substitute_bytes + sizeof(wchar_t));
    reparse->MountPointReparseBuffer.PrintNameLength = static_cast<USHORT>(print_bytes);

    auto* path_buffer = reinterpret_cast<wchar_t*>(reparse->MountPointReparseBuffer.PathBuffer);
    std::memcpy(path_buffer, substitute.c_str(), substitute_bytes);
    path_buffer[substitute.size()] = L'\0';
    std::memcpy(path_buffer + substitute.size() + 1, print.c_str(), print_bytes);
    path_buffer[substitute.size() + 1 + print.size()] = L'\0';

    const HANDLE handle = CreateFileW(link.c_str(), GENERIC_WRITE,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING,
                                      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                                      nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const auto error = GetLastError();
        LOG_ERROR(Common_Filesystem, "CreateFileW({}) failed with error {}", link.string(),
                  error);
        RemoveDirectoryW(link.c_str());
        return false;
    }

    DWORD bytes_returned = 0;
    const BOOL ok = DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, reparse,
                                    static_cast<DWORD>(buffer_bytes), nullptr, 0,
                                    &bytes_returned, nullptr);
    const auto error = ok ? ERROR_SUCCESS : GetLastError();
    CloseHandle(handle);
    if (!ok) {
        LOG_ERROR(Common_Filesystem,
                  "FSCTL_SET_REPARSE_POINT on {} failed with error {}", link.string(), error);
        RemoveDirectoryW(link.c_str());
        return false;
    }
    return true;
}

} // namespace
#endif

bool CreateSymlink(fs::path from, fs::path to)
{
    from.make_preferred();
    to.make_preferred();

    std::error_code ec;
    fs::create_directory_symlink(from, to, ec);
#ifdef _WIN32
    if (ec) {
        // Directory junctions need neither elevation nor developer mode, unlike symlinks.
        const auto target = fs::absolute(from).lexically_normal();
        return CreateJunction(to, target);
    }
#endif

    return !ec;
}

bool IsSymlink(const fs::path &path)
{
#ifdef _WIN32
    // boost::filesystem::is_symlink does not detect directory junctions (mklink /J); check the
    // reparse point attribute directly, which covers both.
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#else
    return boost::filesystem::is_symlink(boost::filesystem::path{path});
#endif
}

} // namespace Common::FS
