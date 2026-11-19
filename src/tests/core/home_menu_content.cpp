// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "common/stb.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/decrypted_nca.h"
#include "core/file_sys/nup_writer.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/service/ns/home_menu_alias.h"
#include "core/loader/loader.h"

namespace {

namespace fs = std::filesystem;

struct TempDir {
    fs::path path;

    TempDir() {
        static std::atomic<u64> sequence{};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        do {
            path = fs::temp_directory_path() /
                   ("veshiru-home-menu-test-" + std::to_string(stamp) + "-" +
                    std::to_string(sequence.fetch_add(1)));
        } while (!fs::create_directory(path));
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void WriteBytes(const fs::path& file, std::string_view content) {
    fs::create_directories(file.parent_path());
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    REQUIRE(out.good());
}

std::string MinimalNpdm(u64 title_id) {
    std::string npdm(0x100, '\0');
    npdm.replace(0, 4, "META");
    const u32 aci_offset = 0x80;
    const u32 aci_size = 0x40;
    std::memcpy(npdm.data() + 0x70, &aci_offset, sizeof(aci_offset));
    std::memcpy(npdm.data() + 0x74, &aci_size, sizeof(aci_size));
    npdm.replace(0x80, 4, "ACI0");
    std::memcpy(npdm.data() + 0x90, &title_id, sizeof(title_id));
    return npdm;
}

void AppendVec(void* ctx, void* data, int size) {
    if (size <= 0) {
        return;
    }
    auto* out = static_cast<std::vector<u8>*>(ctx);
    const auto* bytes = static_cast<const u8*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

std::vector<u8> MakeSolidPng(int width, int height, u8 r, u8 g, u8 b) {
    std::vector<u8> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i + 0] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }
    std::vector<u8> png;
    REQUIRE(stbi_write_png_to_func(AppendVec, &png, width, height, 3, pixels.data(), width * 3) !=
            0);
    return png;
}

bool HasBaselineSof0Dimensions(std::span<const u8> jpeg, int expected_dim) {
    if (jpeg.size() < 4 || jpeg.size() > FileSys::HomeMenuMaxIconBytes || jpeg[0] != 0xFF ||
        jpeg[1] != 0xD8) {
        return false;
    }
    std::size_t pos = 2;
    while (pos + 3 < jpeg.size()) {
        if (jpeg[pos] != 0xFF) {
            return false;
        }
        while (pos < jpeg.size() && jpeg[pos] == 0xFF) {
            ++pos;
        }
        if (pos >= jpeg.size()) {
            return false;
        }
        const u8 marker = jpeg[pos++];
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue;
        }
        if (marker == 0xDA || marker == 0xD9 || pos + 2 > jpeg.size()) {
            return false;
        }
        const std::size_t len = (static_cast<std::size_t>(jpeg[pos]) << 8) | jpeg[pos + 1];
        if (len < 2 || pos + len > jpeg.size()) {
            return false;
        }
        if (marker == 0xC0 && len >= 8) {
            const int h = (static_cast<int>(jpeg[pos + 3]) << 8) | jpeg[pos + 4];
            const int w = (static_cast<int>(jpeg[pos + 5]) << 8) | jpeg[pos + 6];
            return w == expected_dim && h == expected_dim;
        }
        pos += len;
    }
    return false;
}

} // namespace

TEST_CASE("HOME Menu icon sanitizer converts PNG and empty icons to baseline JPEG", "[home-menu]") {
    constexpr u64 OotTitleId = 0x0400486024DEC000ULL;

    const auto fallback = FileSys::BuildHomeMenuIconJpeg(OotTitleId, {});
    CHECK(HasBaselineSof0Dimensions(fallback, FileSys::HomeMenuIconDimension));

    const auto png = MakeSolidPng(512, 512, 36, 148, 82);
    const auto converted = FileSys::BuildHomeMenuIconJpeg(OotTitleId, png);
    CHECK(HasBaselineSof0Dimensions(converted, FileSys::HomeMenuIconDimension));

    // Compliant baseline JPEG passes through without re-encoding.
    const auto passthrough = FileSys::BuildHomeMenuIconJpeg(OotTitleId, converted);
    CHECK(passthrough == converted);

    const auto grid =
        FileSys::BuildHomeMenuIconJpeg(OotTitleId, png, FileSys::HomeMenuGridIconDimension);
    CHECK(HasBaselineSof0Dimensions(grid, FileSys::HomeMenuGridIconDimension));
}

TEST_CASE("NUP Switch 1 and Switch 2 packages register playable HOME Menu entries", "[home-menu]") {
    constexpr u64 Sw1TitleId = 0x01007EF00011E000ULL;
    constexpr u64 Sw2TitleId = 0x0400486024DEC000ULL;

    TempDir tmp;
    const auto sw1_dump = tmp.path / "sw1";
    WriteBytes(sw1_dump / "base/exefs/main", "main");
    WriteBytes(sw1_dump / "base/exefs/main.npdm", MinimalNpdm(Sw1TitleId));
    WriteBytes(sw1_dump / "base/romfs/data.bin", "sw1");
    const auto sw1_nup = tmp.path / "sw1.nup";
    REQUIRE(FileSys::Nup::PackDump(sw1_dump, sw1_nup) == FileSys::Nup::PackResult::Success);

    const auto sw2_dump = tmp.path / "sw2";
    WriteBytes(sw2_dump / "base/exefs/main", "main");
    WriteBytes(sw2_dump / "base/exefs/main.npdm", MinimalNpdm(Sw2TitleId));
    WriteBytes(sw2_dump / "base/romfs/data.bin", "sw2");
    WriteBytes(sw2_dump / "dlc/0400486024DED001/0/romfs/dlc.bin", "dlc");
    const auto sw2_nup = tmp.path / "oot.nup";
    REQUIRE(FileSys::Nup::PackDump(sw2_dump, sw2_nup) == FileSys::Nup::PackResult::Success);

    const auto icon_png = MakeSolidPng(300, 300, 200, 160, 40);
    FileSys::Nup::MetadataEdit meta{};
    meta.title_name = "Ocarina of Time";
    meta.publisher = "Nintendo";
    meta.display_version = "1.2.0";
    meta.icon = icon_png;
    REQUIRE(FileSys::Nup::EditNupMetadata(sw2_nup, meta) == FileSys::Nup::EditResult::Success);

    FileSys::RealVfsFilesystem vfs;
    FileSys::ManualContentProvider provider;

    // only_content=true leaves application Program/Control unregistered.
    const auto sw2_file = vfs.OpenFile(sw2_nup.string(), FileSys::OpenMode::Read);
    REQUIRE(sw2_file != nullptr);
    REQUIRE(provider.AddEntriesFromContainer(sw2_file, true));
    CHECK_FALSE(provider.HasEntry(Sw2TitleId, FileSys::ContentRecordType::Program));
    CHECK_FALSE(provider.HasEntry(Sw2TitleId, FileSys::ContentRecordType::Control));
    CHECK(provider.HasEntry(0x0400486024DED001ULL, FileSys::ContentRecordType::Data));

    provider.ClearAllEntries();
    const auto sw1_file = vfs.OpenFile(sw1_nup.string(), FileSys::OpenMode::Read);
    REQUIRE(sw1_file != nullptr);
    REQUIRE(provider.AddEntriesFromContainer(sw1_file, false));
    REQUIRE(provider.AddEntriesFromContainer(sw2_file, false));

    for (const u64 tid : {Sw1TitleId, Sw2TitleId}) {
        CHECK(provider.HasEntry(tid, FileSys::ContentRecordType::Program));
        CHECK(provider.HasEntry(tid, FileSys::ContentRecordType::Control));
        // Non-NCA .nup Program files return nullptr cleanly from GetEntry instead of ErrorBadNCAHeader.
        CHECK(provider.GetEntry(tid, FileSys::ContentRecordType::Program) == nullptr);
        CHECK(provider.GetEntryRaw(tid, FileSys::ContentRecordType::Program) != nullptr);

        const auto info = provider.GetApplicationInfo(tid);
        REQUIRE(info.has_value());
        CHECK(!info->launch_path.empty());
        CHECK(info->occupied_size > 0);

        const auto control_nca = provider.GetEntry(tid, FileSys::ContentRecordType::Control);
        REQUIRE(control_nca != nullptr);
        REQUIRE(control_nca->GetStatus() == Loader::ResultStatus::Success);
        const auto romfs_dir = FileSys::ExtractRomFS(control_nca->GetRomFS());
        REQUIRE(romfs_dir != nullptr);

        const auto nacp_vf = romfs_dir->GetFile("control.nacp");
        REQUIRE(nacp_vf != nullptr);
        FileSys::NACP nacp{nacp_vf};
        for (const auto& entry : nacp.raw.language_entries.language_entries) {
            CHECK(entry.application_name[0] != '\0');
        }
        CHECK(static_cast<u32>(nacp.raw.supported_languages) != 0);

        const auto icon_vf = romfs_dir->GetFile("icon_AmericanEnglish.dat");
        REQUIRE(icon_vf != nullptr);
        const auto icon_bytes = icon_vf->ReadAllBytes();
        CHECK(HasBaselineSof0Dimensions(icon_bytes, FileSys::HomeMenuIconDimension));
    }

    // Verify Switch 2 title, publisher, display version, and deterministic HOME Menu alias.
    const auto sw2_ctrl = provider.GetEntry(Sw2TitleId, FileSys::ContentRecordType::Control);
    const auto sw2_nacp_vf = FileSys::ExtractRomFS(sw2_ctrl->GetRomFS())->GetFile("control.nacp");
    const FileSys::NACP sw2_nacp{sw2_nacp_vf};
    CHECK(sw2_nacp.raw.language_entries.language_entries[0].GetApplicationName() ==
          "Ocarina of Time");
    CHECK(sw2_nacp.raw.language_entries.language_entries[2].GetApplicationName() ==
          "Ocarina of Time");
    CHECK(sw2_nacp.raw.language_entries.language_entries[0].GetDeveloperName() == "Nintendo");
    CHECK(sw2_nacp.GetVersionString() == "1.2.0");

    CHECK(Service::NS::ToHomeMenuApplicationId(Sw1TitleId) == Sw1TitleId);
    const u64 alias_id = Service::NS::ToHomeMenuApplicationId(Sw2TitleId);
    CHECK(alias_id == 0x01F0486024DEC000ULL);
    CHECK(Service::NS::FromHomeMenuApplicationId(alias_id, provider) == Sw2TitleId);
    CHECK(Service::NS::FromHomeMenuApplicationId(Sw1TitleId, provider) == Sw1TitleId);

    // A genuine 0x01F0... entry takes priority over any Switch 2 alias.
    provider.AddEntry(FileSys::TitleType::Application, FileSys::ContentRecordType::Program,
                      alias_id, sw1_file);
    CHECK(Service::NS::FromHomeMenuApplicationId(alias_id, provider) == alias_id);
}
