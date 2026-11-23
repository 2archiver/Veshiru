// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include "common/assert.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "core/file_sys/errors.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_real.h"

// For FileTimeStampRaw
#include <sys/stat.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstdio> // _getmaxstdio
#include <windows.h>
#undef CreateFile
#undef CopyFile
#undef MoveFile
#undef DeleteFile
#undef CreateDirectory
#else
#include <sys/resource.h> // getrlimit
#endif

#ifdef _MSC_VER
#define stat _stat64
#endif

#ifdef __ANDROID__
#include "common/fs/fs_android.h"
#endif

namespace FileSys {

namespace FS = Common::FS;

#ifdef _WIN32
struct NativeReadHandle {
    HANDLE handle{INVALID_HANDLE_VALUE};

    explicit NativeReadHandle(HANDLE handle_) noexcept : handle{handle_} {}
    NativeReadHandle(const NativeReadHandle&) = delete;
    NativeReadHandle& operator=(const NativeReadHandle&) = delete;

    ~NativeReadHandle() {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
    }
};
#endif

namespace {

#ifdef _WIN32
// Kernel file handles on Windows are not subject to the C runtime's _getmaxstdio() cap, so a
// multi-thousand read-only cache keeps active RomFS archives and stream packs warm without
// touching CRT FILE* slots.
constexpr size_t MaxOpenReadHandles = 4096;

struct ThreadOverlappedEvent {
    HANDLE handle{CreateEventW(nullptr, TRUE, FALSE, nullptr)};

    ThreadOverlappedEvent() = default;
    ThreadOverlappedEvent(const ThreadOverlappedEvent&) = delete;
    ThreadOverlappedEvent& operator=(const ThreadOverlappedEvent&) = delete;

    ~ThreadOverlappedEvent() {
        if (handle != nullptr) {
            CloseHandle(handle);
        }
    }
};

std::shared_ptr<NativeReadHandle> OpenNativeReadHandle(const std::string& path) {
    const std::filesystem::path wide_path{FS::ToU8String(path)};
    const HANDLE handle =
        CreateFileW(wide_path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    if (GetFileType(handle) != FILE_TYPE_DISK) {
        CloseHandle(handle);
        return nullptr;
    }
    // Avoid acquiring the kernel FILE_OBJECT lock to signal the file handle on every synchronous
    // page-cache hit; async ERROR_IO_PENDING completions still signal OVERLAPPED::hEvent.
    SetFileCompletionNotificationModes(handle, FILE_SKIP_SET_EVENT_ON_HANDLE);
    return std::make_shared<NativeReadHandle>(handle);
}

std::size_t ReadNativeOverlapped(HANDLE handle, u8* data, std::size_t length, std::size_t offset) {
    if (length == 0) {
        return 0;
    }
    thread_local ThreadOverlappedEvent tls_event{};
    if (tls_event.handle == nullptr) {
        return 0;
    }

    constexpr std::size_t MaxChunkSize = std::size_t{1} << 30;
    std::size_t total_read = 0;
    u64 current_offset = offset;
    while (total_read < length) {
        const DWORD to_read = static_cast<DWORD>((std::min)(length - total_read, MaxChunkSize));
        OVERLAPPED overlapped{};
        overlapped.Offset = static_cast<DWORD>(current_offset & 0xFFFF'FFFFULL);
        overlapped.OffsetHigh = static_cast<DWORD>(current_offset >> 32);
        overlapped.hEvent = tls_event.handle;

        DWORD bytes_read = 0;
        if (!ReadFile(handle, data + total_read, to_read, &bytes_read, &overlapped)) {
            const DWORD err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                const BOOL ok = GetOverlappedResult(handle, &overlapped, &bytes_read, TRUE);
                const DWORD wait_err = ok ? ERROR_SUCCESS : GetLastError();
                ResetEvent(tls_event.handle);
                if (!ok) {
                    if (wait_err == ERROR_HANDLE_EOF) {
                        total_read += bytes_read;
                    }
                    break;
                }
            } else {
                if (err == ERROR_HANDLE_EOF) {
                    total_read += bytes_read;
                }
                break;
            }
        }

        total_read += bytes_read;
        if (bytes_read < to_read ||
            current_offset > (std::numeric_limits<u64>::max)() - bytes_read) {
            break;
        }
        current_offset += bytes_read;
    }
    return total_read;
}
#endif

// Files the cache keeps open. It has to stay below the process's open-file limit (the C runtime's
// on Windows, RLIMIT_NOFILE elsewhere) with room to spare: at the limit every other open fails
// (saves, shader caches, and the cache's own next file, which it only evicts for once it reaches
// its cap).
size_t MaxOpenFiles() {
#ifdef _WIN32
    static const size_t limit = static_cast<size_t>((std::max)(_getmaxstdio() - 512, 64));
#else
    static const size_t limit = [] {
        struct rlimit rl;
        if (getrlimit(RLIMIT_NOFILE, &rl) != 0 || rl.rlim_cur == RLIM_INFINITY) {
            return size_t{8192};
        }
        return static_cast<size_t>(std::max<rlim_t>(rl.rlim_cur, 576) - 512);
    }();
#endif
    return (std::min)(size_t{8192}, limit);
}

bool IsWithinRoot(std::string_view root, std::string_view full_path) {
    if (root.empty())
        return true;

    if (full_path.size() < root.size() || full_path.substr(0, root.size()) != root)
        return false;

    return full_path.size() == root.size() || full_path[root.size()] == '/' || full_path[root.size()] == '\\';
}

constexpr FS::FileAccessMode ModeFlagsToFileAccessMode(OpenMode mode) {
    switch (mode) {
    case OpenMode::Read:
        return FS::FileAccessMode::Read;
    case OpenMode::Write:
    case OpenMode::ReadWrite:
    case OpenMode::AllowAppend:
    case OpenMode::All:
        return FS::FileAccessMode::ReadWrite;
    default:
        return {};
    }
}

} // Anonymous namespace

RealVfsFilesystem::RealVfsFilesystem() : VfsFilesystem(nullptr) {}
RealVfsFilesystem::~RealVfsFilesystem() {
    in_dtor = true;
}

std::string RealVfsFilesystem::GetName() const {
    return "Real";
}

bool RealVfsFilesystem::IsReadable() const {
    return true;
}

bool RealVfsFilesystem::IsWritable() const {
    return true;
}

VfsEntryType RealVfsFilesystem::GetEntryType(std::string_view path_) const {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    if (!FS::Exists(path)) {
        return VfsEntryType::None;
    }
    if (FS::IsDir(path)) {
        return VfsEntryType::Directory;
    }

    return VfsEntryType::File;
}

VirtualFile RealVfsFilesystem::OpenFileFromEntry(std::string_view path_, std::optional<u64> size,
                                                 OpenMode perms) {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    std::scoped_lock lk{list_lock};
    return OpenFileLocked(path, size, perms);
}

VirtualFile RealVfsFilesystem::OpenFileLocked(const std::string& path, std::optional<u64> size,
                                              OpenMode perms) {
    if (auto it = cache.find(path); it != cache.end()) {
        if (auto file = it->second.file.lock(); file) {
            return file;
        }
        CloseReferenceLocked(*it->second.reference);
        DetachReferenceLocked(it);
    }

    if (!size && !FS::IsFile(path)) {
        return nullptr;
    }

    auto reference = std::make_unique<FileReference>();
    reference->path = path;
    reference->size = size;
    auto file = std::make_shared<RealVfsFile>(*this, std::move(reference), perms);
    InsertReferenceIntoListLocked(*file->reference);
    cache[path] = {file, file->reference.get()};

    return file;
}

VirtualFile RealVfsFilesystem::OpenFile(std::string_view path_, OpenMode perms) {
    return OpenFileFromEntry(path_, {}, perms);
}

VirtualFile RealVfsFilesystem::CreateFile(std::string_view path_, OpenMode perms) {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    std::scoped_lock lk{list_lock};
    const auto entry = cache.find(path);
#ifdef _WIN32
    if (entry != cache.end()) {
        CloseReferenceLocked(*entry->second.reference);
    }
#endif

    // Current usages of CreateFile expect to delete the contents of an existing file.
    if (FS::IsFile(path)) {
        FS::IOFile temp{path, FS::FileAccessMode::Write, FS::FileType::BinaryFile};

        if (!temp.IsOpen()) {
            return nullptr;
        }
    } else if (!FS::NewFile(path)) {
        return nullptr;
    }

    if (entry != cache.end()) {
        entry->second.reference->size.reset();
        DetachReferenceLocked(entry);
    }
    return OpenFileLocked(path, {}, perms);
}

VirtualFile RealVfsFilesystem::CopyFile(std::string_view old_path_, std::string_view new_path_) {
    // Unused
    return nullptr;
}

VirtualFile RealVfsFilesystem::MoveFile(std::string_view old_path_, std::string_view new_path_) {
    const auto old_path = FS::SanitizePath(old_path_, FS::DirectorySeparator::PlatformDefault);
    const auto new_path = FS::SanitizePath(new_path_, FS::DirectorySeparator::PlatformDefault);
    std::scoped_lock lk{list_lock};
    if (!RenameFileLocked(old_path, new_path)) {
        return nullptr;
    }
    return OpenFileLocked(new_path, {}, OpenMode::ReadWrite);
}

bool RealVfsFilesystem::RenameFileLocked(const std::string& old_path, const std::string& new_path) {
    if (old_path == new_path) {
        return false;
    }
    const auto source = cache.find(old_path);
#ifdef _WIN32
    if (source != cache.end()) {
        CloseReferenceLocked(*source->second.reference);
    }
#endif
    if (!FS::RenameFile(old_path, new_path)) {
        return false;
    }

    if (auto destination = cache.find(new_path); destination != cache.end()) {
        DetachReferenceLocked(destination);
    }
    if (source != cache.end()) {
#ifndef _WIN32
        CloseReferenceLocked(*source->second.reference);
#endif
        auto entry = cache.extract(source);
        entry.key() = new_path;
        entry.mapped().reference->path = new_path;
        cache.insert(std::move(entry));
    }
    return true;
}

bool RealVfsFilesystem::DeleteFile(std::string_view path_) {
    return DeleteFileWithResult(path_).IsSuccess();
}

Result RealVfsFilesystem::DeleteFileWithResult(std::string_view path_) {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    std::scoped_lock lk{list_lock};
    const auto entry = cache.find(path);
#ifdef _WIN32
    if (entry != cache.end()) {
        CloseReferenceLocked(*entry->second.reference);
    }
#endif
    std::error_code ec;
    if (!FS::RemoveFile(std::filesystem::path{FS::ToU8String(path)}, ec)) {
#ifdef _WIN32
        if (ec == std::error_code{ERROR_SHARING_VIOLATION, std::system_category()} ||
            ec == std::error_code{ERROR_LOCK_VIOLATION, std::system_category()}) {
            return ResultTargetLocked;
        }
#endif
        if (ec == std::errc::permission_denied || ec == std::errc::read_only_file_system) {
            return ResultPermissionDenied;
        }
        if (ec == std::errc::invalid_argument || ec == std::errc::is_a_directory) {
            return ResultInvalidPath;
        }
        return ResultUnknown;
    }
    if (entry != cache.end()) {
        DetachReferenceLocked(entry);
    }
    return ResultSuccess;
}

VirtualDir RealVfsFilesystem::OpenDirectory(std::string_view path_, OpenMode perms) {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    return std::make_shared<RealVfsDirectory>(*this, path, perms);
}

VirtualDir RealVfsFilesystem::CreateDirectory(std::string_view path_, OpenMode perms) {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    if (!FS::CreateDirs(path)) {
        return nullptr;
    }
    return std::make_shared<RealVfsDirectory>(*this, path, perms);
}

VirtualDir RealVfsFilesystem::CopyDirectory(std::string_view old_path_,
                                            std::string_view new_path_) {
    // Unused
    return nullptr;
}

VirtualDir RealVfsFilesystem::MoveDirectory(std::string_view old_path_,
                                            std::string_view new_path_) {
    const auto old_path = FS::SanitizePath(old_path_, FS::DirectorySeparator::PlatformDefault);
    const auto new_path = FS::SanitizePath(new_path_, FS::DirectorySeparator::PlatformDefault);
    {
        std::scoped_lock lk{list_lock};
        UncacheDirectoryLocked(old_path);
        UncacheDirectoryLocked(new_path);
    }

    if (!FS::RenameDir(old_path, new_path)) {
        return nullptr;
    }
    return OpenDirectory(new_path, OpenMode::ReadWrite);
}

bool RealVfsFilesystem::DeleteDirectory(std::string_view path_) {
    const auto path = FS::SanitizePath(path_, FS::DirectorySeparator::PlatformDefault);
    {
        std::scoped_lock lk{list_lock};
        UncacheDirectoryLocked(path);
    }
    return FS::RemoveDirRecursively(path);
}

void RealVfsFilesystem::RefreshReferenceLocked(FileReference& reference, OpenMode perms) {
    // Temporarily remove from list.
    this->RemoveReferenceFromListLocked(reference);

#ifdef _WIN32
    if (reference.read_handle) {
        reference.read_handle.reset();
        num_open_read_handles--;
    }
#endif

    // Restore file if needed.
    if (!reference.file && !reference.detached) {
        this->EvictSingleReferenceLocked();

        reference.file = FS::FileOpen(reference.path, ModeFlagsToFileAccessMode(perms),
                                      FS::FileType::BinaryFile);
        if (reference.file) {
            num_open_files++;
        }
    }

    // Reinsert into list.
    this->InsertReferenceIntoListLocked(reference);
}

#ifdef _WIN32
std::shared_ptr<NativeReadHandle> RealVfsFilesystem::AcquireReadHandle(FileReference& reference) {
    // Fast path: while below the 4,096-handle eviction threshold (or already MRU), concurrent
    // readers share `list_lock` without serializing or churning the intrusive LRU list on every
    // small archive/RomFS read.
    {
        std::shared_lock lk{list_lock};
        if (reference.detached) {
            return nullptr;
        }
        if (reference.read_handle &&
            (num_open_read_handles < MaxOpenReadHandles ||
             (!open_read_references.empty() && &open_read_references.front() == &reference))) {
            return reference.read_handle;
        }
    }

    std::scoped_lock lk{list_lock};
    if (reference.detached) {
        return nullptr;
    }
    if (reference.read_handle &&
        (num_open_read_handles < MaxOpenReadHandles ||
         (!open_read_references.empty() && &open_read_references.front() == &reference))) {
        return reference.read_handle;
    }

    this->RemoveReferenceFromListLocked(reference);

    if (reference.file) {
        reference.file.reset();
        num_open_files--;
    }

    if (!reference.read_handle) {
        this->EvictSingleReadReferenceLocked();

        reference.read_handle = OpenNativeReadHandle(reference.path);
        if (reference.read_handle) {
            num_open_read_handles++;
        }
    }

    this->InsertReferenceIntoListLocked(reference);

    return reference.read_handle;
}
#endif

void RealVfsFilesystem::DropReference(FileReference& reference) {
    if (in_dtor)
        return;

    std::scoped_lock lk{list_lock};

    // Remove from list.
    this->RemoveReferenceFromListLocked(reference);

    // Forget the path, unless it was opened again since this reference was detached.
    if (const auto it = cache.find(reference.path);
        it != cache.end() && it->second.reference == &reference) {
        cache.erase(it);
    }

    // Close the file.
    if (reference.file) {
        reference.file.reset();
        num_open_files--;
    }
#ifdef _WIN32
    if (reference.read_handle) {
        reference.read_handle.reset();
        num_open_read_handles--;
    }
#endif
}

void RealVfsFilesystem::DetachReferenceLocked(FileCache::iterator entry) {
    entry->second.reference->detached = true;
    cache.erase(entry);
}

void RealVfsFilesystem::EvictSingleReferenceLocked() {
    if (num_open_files < MaxOpenFiles() || open_references.empty()) {
        return;
    }

    // Get and remove from list.
    auto& reference = open_references.back();
    this->RemoveReferenceFromListLocked(reference);

    // Close the file.
    if (reference.file) {
        reference.file.reset();
        num_open_files--;
    }

    // Reinsert into closed list.
    this->InsertReferenceIntoListLocked(reference);
}

#ifdef _WIN32
void RealVfsFilesystem::EvictSingleReadReferenceLocked() {
    if (num_open_read_handles < MaxOpenReadHandles || open_read_references.empty()) {
        return;
    }

    auto& reference = open_read_references.back();
    this->RemoveReferenceFromListLocked(reference);

    if (reference.read_handle) {
        reference.read_handle.reset();
        num_open_read_handles--;
    }

    this->InsertReferenceIntoListLocked(reference);
}
#endif

void RealVfsFilesystem::CloseReferenceLocked(FileReference& reference) {
    if (!reference.file
#ifdef _WIN32
        && !reference.read_handle
#endif
    ) {
        return;
    }
    this->RemoveReferenceFromListLocked(reference);
    if (reference.file) {
        reference.file.reset();
        num_open_files--;
    }
#ifdef _WIN32
    if (reference.read_handle) {
        reference.read_handle.reset();
        num_open_read_handles--;
    }
#endif
    this->InsertReferenceIntoListLocked(reference);
}

void RealVfsFilesystem::UncacheDirectoryLocked(std::string_view dir_path) {
    for (auto it = cache.lower_bound(dir_path); it != cache.end();) {
        if (!it->first.starts_with(dir_path)) {
            break;
        }
        if (IsWithinRoot(dir_path, it->first)) {
            CloseReferenceLocked(*it->second.reference);
            it->second.reference->detached = true;
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

void RealVfsFilesystem::InsertReferenceIntoListLocked(FileReference& reference) {
    if (reference.file) {
        open_references.push_front(reference);
        return;
    }
#ifdef _WIN32
    if (reference.read_handle) {
        open_read_references.push_front(reference);
        return;
    }
#endif
    closed_references.push_front(reference);
}

void RealVfsFilesystem::RemoveReferenceFromListLocked(FileReference& reference) {
    if (reference.file) {
        open_references.erase(open_references.iterator_to(reference));
        return;
    }
#ifdef _WIN32
    if (reference.read_handle) {
        open_read_references.erase(open_read_references.iterator_to(reference));
        return;
    }
#endif
    closed_references.erase(closed_references.iterator_to(reference));
}

RealVfsFile::RealVfsFile(RealVfsFilesystem& base_, std::unique_ptr<FileReference> reference_,
                         OpenMode perms_)
    : base(base_), reference(std::move(reference_)), perms(perms_) {}

RealVfsFile::~RealVfsFile() {
    base.DropReference(*reference);
}

std::string RealVfsFile::GetName() const {
    std::shared_lock lk{base.list_lock};
    const auto& path = reference->path;
#ifdef __ANDROID__
    if (path[0] != '/') {
        return FS::Android::GetFilename(path);
    }
#endif
    // The last path component.
    const auto end = path.find_last_not_of("\\/");
    if (end == std::string::npos) {
        return "";
    }
    const auto separator = path.find_last_of("\\/", end);
    const auto begin = separator == std::string::npos ? 0 : separator + 1;
    return path.substr(begin, end + 1 - begin);
}

std::size_t RealVfsFile::GetSize() const {
    {
        std::shared_lock lk{base.list_lock};
        if (reference->size && !reference->detached) {
            return *reference->size;
        }
    }
#ifdef _WIN32
    if (perms == OpenMode::Read) {
        const auto handle = base.AcquireReadHandle(*reference);
        LARGE_INTEGER file_size{};
        if (!handle || !GetFileSizeEx(handle->handle, &file_size) || file_size.QuadPart < 0) {
            return 0;
        }
        return static_cast<std::size_t>(file_size.QuadPart);
    }
#endif
    std::scoped_lock lk{base.list_lock};
    base.RefreshReferenceLocked(*reference, perms);
    return reference->file ? reference->file->GetSize() : 0;
}

bool RealVfsFile::Resize(std::size_t new_size) {
    std::scoped_lock lk{base.list_lock};
    reference->size.reset();
    base.RefreshReferenceLocked(*reference, perms);
    return reference->file ? reference->file->SetSize(new_size) : false;
}

VirtualDir RealVfsFile::GetContainingDirectory() const {
    std::string parent;
    {
        std::shared_lock lk{base.list_lock};
        parent = FS::GetParentPath(reference->path);
    }
    return base.OpenDirectory(parent, perms);
}

bool RealVfsFile::IsWritable() const {
    return True(perms & OpenMode::Write);
}

bool RealVfsFile::IsReadable() const {
    return True(perms & OpenMode::Read);
}

std::size_t RealVfsFile::Read(u8* data, std::size_t length, std::size_t offset) const {
#ifdef _WIN32
    if (perms == OpenMode::Read) {
        if (length == 0) {
            return 0;
        }
        const auto handle = base.AcquireReadHandle(*reference);
        if (!handle) {
            return 0;
        }
        return ReadNativeOverlapped(handle->handle, data, length, offset);
    }
#endif
    std::scoped_lock lk{base.list_lock};
    base.RefreshReferenceLocked(*reference, perms);
    if (!reference->file || !reference->file->Seek(static_cast<s64>(offset))) {
        return 0;
    }
    return reference->file->ReadSpan(std::span{data, length});
}

std::size_t RealVfsFile::Write(const u8* data, std::size_t length, std::size_t offset) {
    std::scoped_lock lk{base.list_lock};
    reference->size.reset();
    base.RefreshReferenceLocked(*reference, perms);
    if (!reference->file || !reference->file->Seek(static_cast<s64>(offset))) {
        return 0;
    }
    return reference->file->WriteSpan(std::span{data, length});
}

bool RealVfsFile::Rename(std::string_view name) {
    std::scoped_lock lk{base.list_lock};
    if (reference->detached) {
        return false;
    }
    const auto& path = reference->path;
    const auto new_path = FS::SanitizePath(FS::GetParentPath(path) + '/' + std::string(name),
                                           FS::DirectorySeparator::PlatformDefault);
    return base.RenameFileLocked(path, new_path);
}

// TODO(DarkLordZach): MSVC would not let me combine the following two functions using 'if
// constexpr' because there is a compile error in the branch not used.

template <>
std::vector<VirtualFile> RealVfsDirectory::IterateEntries<RealVfsFile, VfsFile>() const {
    if (perms == OpenMode::AllowAppend) {
        return {};
    }

    std::vector<VirtualFile> out;

    const FS::DirEntryCallable callback = [this,
                                           &out](const std::filesystem::directory_entry& entry) {
        const auto full_path_string = FS::PathToUTF8String(entry.path());

        out.emplace_back(base.OpenFileFromEntry(full_path_string, entry.file_size(), perms));

        return true;
    };

    FS::IterateDirEntries(path, callback, FS::DirEntryFilter::File);

    return out;
}

template <>
std::vector<VirtualDir> RealVfsDirectory::IterateEntries<RealVfsDirectory, VfsDirectory>() const {
    if (perms == OpenMode::AllowAppend) {
        return {};
    }

    std::vector<VirtualDir> out;

    const FS::DirEntryCallable callback = [this,
                                           &out](const std::filesystem::directory_entry& entry) {
        const auto full_path_string = FS::PathToUTF8String(entry.path());

        out.emplace_back(base.OpenDirectory(full_path_string, perms));

        return true;
    };

    FS::IterateDirEntries(path, callback, FS::DirEntryFilter::Directory);

    return out;
}

RealVfsDirectory::RealVfsDirectory(RealVfsFilesystem& base_, const std::string& path_,
                                   OpenMode perms_)
    : base(base_), path(FS::RemoveTrailingSlash(path_)), parent_path(FS::GetParentPath(path)),
      path_components(FS::SplitPathComponentsCopy(path)), perms(perms_) {
    if (!FS::Exists(path) && True(perms & OpenMode::Write)) {
        void(FS::CreateDirs(path));
    }
}

RealVfsDirectory::~RealVfsDirectory() = default;

VirtualFile RealVfsDirectory::GetFileRelative(std::string_view relative_path) const {
    const auto full_path = FS::SanitizePath(path + '/' + std::string(relative_path));
    if (!FS::Exists(full_path) || FS::IsDir(full_path)
        || !IsWithinRoot(FS::SanitizePath(path), full_path)) {
        return nullptr;
    }
    return base.OpenFile(full_path, perms);
}

VirtualDir RealVfsDirectory::GetDirectoryRelative(std::string_view relative_path) const {
    const auto full_path = FS::SanitizePath(path + '/' + std::string(relative_path));
    if (!FS::Exists(full_path) || !FS::IsDir(full_path)
        || !IsWithinRoot(FS::SanitizePath(path), full_path)) {
        return nullptr;
    }
    return base.OpenDirectory(full_path, perms);
}

VirtualFile RealVfsDirectory::GetFile(std::string_view name) const {
    return GetFileRelative(name);
}

VirtualDir RealVfsDirectory::GetSubdirectory(std::string_view name) const {
    return GetDirectoryRelative(name);
}

VirtualFile RealVfsDirectory::CreateFileRelative(std::string_view relative_path) {
    const auto full_path = FS::SanitizePath(path + '/' + std::string(relative_path));
    if (!FS::CreateParentDirs(full_path) || !IsWithinRoot(FS::SanitizePath(path), full_path)) {
        return nullptr;
    }
    return base.CreateFile(full_path, perms);
}

VirtualDir RealVfsDirectory::CreateDirectoryRelative(std::string_view relative_path) {
    const auto full_path = FS::SanitizePath(path + '/' + std::string(relative_path));
    return base.CreateDirectory(full_path, perms);
}

bool RealVfsDirectory::DeleteSubdirectoryRecursive(std::string_view name) {
    const auto full_path = FS::SanitizePath(this->path + '/' + std::string(name));
    return base.DeleteDirectory(full_path);
}

std::vector<VirtualFile> RealVfsDirectory::GetFiles() const {
    return IterateEntries<RealVfsFile, VfsFile>();
}

FileTimeStampRaw RealVfsDirectory::GetFileTimeStamp(std::string_view path_) const {
    const auto full_path = FS::SanitizePath(path + '/' + std::string(path_));
    const auto fs_path = std::filesystem::path{FS::ToU8String(full_path)};

#ifdef _WIN32
    struct _stat64 file_status;
    const auto stat_result = _wstat64(fs_path.c_str(), &file_status);
#else
    struct stat file_status;
    const auto stat_result = stat(fs_path.c_str(), &file_status);
#endif

    if (stat_result != 0) {
        return {};
    }

    return {
        .created{static_cast<u64>(file_status.st_ctime)},
        .accessed{static_cast<u64>(file_status.st_atime)},
        .modified{static_cast<u64>(file_status.st_mtime)},
    };
}

std::vector<VirtualDir> RealVfsDirectory::GetSubdirectories() const {
    return IterateEntries<RealVfsDirectory, VfsDirectory>();
}

bool RealVfsDirectory::IsWritable() const {
    return True(perms & OpenMode::Write);
}

bool RealVfsDirectory::IsReadable() const {
    return True(perms & OpenMode::Read);
}

std::string RealVfsDirectory::GetName() const {
    return path_components.empty() ? "" : std::string(path_components.back());
}

VirtualDir RealVfsDirectory::GetParentDirectory() const {
    if (path_components.size() <= 1) {
        return nullptr;
    }

    return base.OpenDirectory(parent_path, perms);
}

VirtualDir RealVfsDirectory::CreateSubdirectory(std::string_view name) {
    const std::string subdir_path = (path + '/').append(name);
    return base.CreateDirectory(subdir_path, perms);
}

VirtualFile RealVfsDirectory::CreateFile(std::string_view name) {
    const std::string file_path = (path + '/').append(name);
    return base.CreateFile(file_path, perms);
}

bool RealVfsDirectory::DeleteSubdirectory(std::string_view name) {
    const std::string subdir_path = (path + '/').append(name);
    return FS::RemoveDir(subdir_path);
}

bool RealVfsDirectory::DeleteFile(std::string_view name) {
    return DeleteFileWithResult(name).IsSuccess();
}

Result RealVfsDirectory::DeleteFileWithResult(std::string_view name) {
    const std::string file_path = (path + '/').append(name);
    return base.DeleteFileWithResult(file_path);
}

bool RealVfsDirectory::Rename(std::string_view name) {
    const std::string new_name = (parent_path + '/').append(name);
    return base.MoveFile(path, new_name) != nullptr;
}

std::string RealVfsDirectory::GetFullPath() const {
    auto out = path;
    std::replace(out.begin(), out.end(), '\\', '/');
    return out;
}

std::map<std::string, VfsEntryType, std::less<>> RealVfsDirectory::GetEntries() const {
    if (perms == OpenMode::AllowAppend) {
        return {};
    }

    std::map<std::string, VfsEntryType, std::less<>> out;

    const FS::DirEntryCallable callback = [&out](const std::filesystem::directory_entry& entry) {
        const auto filename = FS::PathToUTF8String(entry.path().filename());
        out.insert_or_assign(filename,
                             entry.is_directory() ? VfsEntryType::Directory : VfsEntryType::File);
        return true;
    };

    FS::IterateDirEntries(path, callback);

    return out;
}

} // namespace FileSys
