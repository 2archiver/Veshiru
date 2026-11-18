// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/fs/path_util.h"
#include "core/file_sys/vfs/vfs_real.h"

namespace {

namespace fs = std::filesystem;

u8 PatternByte(std::size_t file_index, std::size_t offset) noexcept {
    const std::size_t mixed = (offset * 1315423911ULL) ^ (file_index * 2654435761ULL) ^ (offset >> 7);
    return static_cast<u8>(mixed & 0xFFU);
}

struct TempDir {
    fs::path root;

    explicit TempDir(const char* name)
        : root{fs::temp_directory_path() / name} {
        std::error_code ec;
        fs::remove_all(root, ec);
        fs::create_directories(root);
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    fs::path WritePatternFile(const std::string& relative, std::size_t file_index,
                              std::size_t size) const {
        const fs::path full = root / relative;
        fs::create_directories(full.parent_path());
        std::vector<u8> bytes(size);
        for (std::size_t i = 0; i < size; ++i) {
            bytes[i] = PatternByte(file_index, i);
        }
        std::ofstream out{full, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        out.close();
        return full;
    }
};

} // namespace

TEST_CASE("RealVfsFile serves concurrent multi-reader offsets and EOF boundaries accurately",
          "[core][vfs]") {
    TempDir temp{"veshiru-vfs-real-concurrent-test"};
    constexpr std::size_t SharedSize = 256 * 1024 + 37;
    constexpr std::size_t ExtraFiles = 24;
    constexpr std::size_t ExtraSize = 8192 + 19;

    const fs::path shared_path = temp.WritePatternFile("shared.bin", 0, SharedSize);
    std::vector<fs::path> extra_paths;
    extra_paths.reserve(ExtraFiles);
    for (std::size_t i = 0; i < ExtraFiles; ++i) {
        extra_paths.push_back(
            temp.WritePatternFile("pack_" + std::to_string(i) + ".bin", i + 1, ExtraSize));
    }

    FileSys::RealVfsFilesystem vfs;
    const auto shared =
        vfs.OpenFile(Common::FS::PathToUTF8String(shared_path), FileSys::OpenMode::Read);
    REQUIRE(shared != nullptr);
    REQUIRE(shared->GetSize() == SharedSize);

    std::vector<FileSys::VirtualFile> extras;
    extras.reserve(ExtraFiles);
    for (const auto& path : extra_paths) {
        auto file = vfs.OpenFile(Common::FS::PathToUTF8String(path), FileSys::OpenMode::Read);
        REQUIRE(file != nullptr);
        extras.push_back(std::move(file));
    }

    // Zero-length, tail-spanning, exact-EOF, and past-EOF reads.
    std::array<u8, 128> tail{};
    REQUIRE(shared->Read(tail.data(), 0, 17) == 0);
    REQUIRE(shared->Read(tail.data(), tail.size(), SharedSize) == 0);
    REQUIRE(shared->Read(tail.data(), tail.size(), SharedSize + 4096) == 0);
    REQUIRE(shared->Read(tail.data(), tail.size(), SharedSize - 29) == 29);
    for (std::size_t i = 0; i < 29; ++i) {
        REQUIRE(tail[i] == PatternByte(0, SharedSize - 29 + i));
    }

    // Eight concurrent threads reading overlapping and disjoint slices of both the shared file and
    // the file pool at once.
    constexpr std::size_t ThreadCount = 8;
    constexpr std::size_t Iterations = 160;
    std::barrier start_gate{static_cast<std::ptrdiff_t>(ThreadCount)};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    workers.reserve(ThreadCount);

    for (std::size_t t = 0; t < ThreadCount; ++t) {
        workers.emplace_back([&, t] {
            std::vector<u8> buffer(16384);
            start_gate.arrive_and_wait();
            for (std::size_t iter = 0; iter < Iterations; ++iter) {
                const std::size_t seed = t * 997 + iter * 131;
                const std::size_t offset = (seed * 521) % (SharedSize - 1);
                const std::size_t req_len = 1 + ((seed * 37) % buffer.size());
                const std::size_t expected = (std::min)(req_len, SharedSize - offset);
                const std::size_t got = shared->Read(buffer.data(), req_len, offset);
                if (got != expected) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                for (std::size_t i = 0; i < got; ++i) {
                    if (buffer[i] != PatternByte(0, offset + i)) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }

                const std::size_t file_idx = (t + iter) % ExtraFiles;
                const std::size_t small_off = (seed * 73) % ExtraSize;
                const std::size_t small_req = 1 + (seed % 512);
                const std::size_t small_exp = (std::min)(small_req, ExtraSize - small_off);
                const std::size_t small_got =
                    extras[file_idx]->Read(buffer.data(), small_req, small_off);
                if (small_got != small_exp) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                for (std::size_t i = 0; i < small_got; ++i) {
                    if (buffer[i] != PatternByte(file_idx + 1, small_off + i)) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    REQUIRE_FALSE(failed.load());
}

TEST_CASE("RealVfsFilesystem mutations cleanly release cached read handles", "[core][vfs]") {
    TempDir temp{"veshiru-vfs-real-mutation-test"};
    const std::string a_path =
        Common::FS::PathToUTF8String(temp.WritePatternFile("a.bin", 1, 1024));
    const std::string b_path =
        Common::FS::PathToUTF8String(temp.WritePatternFile("b.bin", 2, 1024));
    const std::string c_path =
        Common::FS::PathToUTF8String(temp.WritePatternFile("sub/c.bin", 3, 1024));
    const std::string moved_path = Common::FS::PathToUTF8String(temp.root / "b_moved.bin");
    const std::string sub_dir = Common::FS::PathToUTF8String(temp.root / "sub");

    FileSys::RealVfsFilesystem vfs;
    auto a = vfs.OpenFile(a_path, FileSys::OpenMode::Read);
    auto b = vfs.OpenFile(b_path, FileSys::OpenMode::Read);
    auto c = vfs.OpenFile(c_path, FileSys::OpenMode::Read);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);

    std::array<u8, 32> probe{};
    REQUIRE(a->Read(probe.data(), probe.size(), 0) == probe.size());
    REQUIRE(b->Read(probe.data(), probe.size(), 0) == probe.size());
    REQUIRE(c->Read(probe.data(), probe.size(), 0) == probe.size());

    // Delete, move, truncate-recreate, and recursive directory delete all work while VirtualFile
    // instances remain alive in the cache.
    REQUIRE(vfs.DeleteFile(a_path));
    REQUIRE_FALSE(fs::exists(temp.root / "a.bin"));

    auto moved = vfs.MoveFile(b_path, moved_path);
    REQUIRE(moved != nullptr);
    REQUIRE(moved->Read(probe.data(), probe.size(), 0) == probe.size());
    REQUIRE(probe[0] == PatternByte(2, 0));

    const std::array<u8, 4> replacement{0xde, 0xad, 0xbe, 0xef};
    auto recreated = vfs.CreateFile(moved_path, FileSys::OpenMode::ReadWrite);
    REQUIRE(recreated != nullptr);
    REQUIRE(recreated->Write(replacement.data(), replacement.size(), 0) == replacement.size());
    std::array<u8, 4> readback{};
    REQUIRE(recreated->Read(readback.data(), readback.size(), 0) == readback.size());
    REQUIRE(readback == replacement);

    REQUIRE(vfs.DeleteDirectory(sub_dir));
    REQUIRE_FALSE(fs::exists(temp.root / "sub"));
}
