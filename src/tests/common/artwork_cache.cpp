// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/artwork_cache.h"
#include "common/fs/path_util.h"

namespace {
std::vector<u8> MakeValidMinimalPngHeader(u32 width = 512, u32 height = 512) {
    std::vector<u8> bytes(72, 0);
    // PNG signature
    const std::array<u8, 8> sig{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::copy(sig.begin(), sig.end(), bytes.begin());
    // IHDR length (13) + "IHDR"
    bytes[11] = 13;
    bytes[12] = 'I';
    bytes[13] = 'H';
    bytes[14] = 'D';
    bytes[15] = 'R';
    bytes[16] = static_cast<u8>((width >> 24) & 0xFF);
    bytes[17] = static_cast<u8>((width >> 16) & 0xFF);
    bytes[18] = static_cast<u8>((width >> 8) & 0xFF);
    bytes[19] = static_cast<u8>(width & 0xFF);
    bytes[20] = static_cast<u8>((height >> 24) & 0xFF);
    bytes[21] = static_cast<u8>((height >> 16) & 0xFF);
    bytes[22] = static_cast<u8>((height >> 8) & 0xFF);
    bytes[23] = static_cast<u8>(height & 0xFF);
    return bytes;
}
} // namespace

TEST_CASE("Artwork lookup prioritizes exact Program ID and cleans dump filenames",
          "[artwork_cache]") {
    using namespace Common::ArtworkCache;

    // Exact Title ID overrides noisy dump filename or update bit suffix
    CHECK(ResolveCanonicalSearchTitle("0400C3F00006E000_USA_dump_v131072", 0x0400C3F00006E800ULL) ==
          "Mario Kart World");
    CHECK(ResolveCanonicalSearchTitle("", 0x0400486024DEC000ULL) ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(ResolveCanonicalSearchTitle("fast_fusion_repack", 0x040041E0212AE000ULL) ==
          "Fast Fusion");

    // Cleaned filename / alias queries when Program ID is unknown
    CHECK(CleanGameSearchQuery("Donkey_Kong_Bananza_[0400123456780000][v0](USA).nup") ==
          "Donkey Kong Bananza");
    CHECK(CleanGameSearchQuery("Donkey.Kong.Bananza.[0400123456780000](USA)(v1.1.0)") ==
          "Donkey Kong Bananza");
    CHECK(CleanGameSearchQuery("dkb") == "Donkey Kong Bananza");
    CHECK(CleanGameSearchQuery("welcome-tour") == "Nintendo Switch 2 Welcome Tour");
}

TEST_CASE("Artwork candidate scoring avoids wrong editions, demos, and sequel mismatches",
          "[artwork_cache]") {
    using Common::ArtworkCache::ScoreArtworkCandidate;

    const int mkw_exact = ScoreArtworkCandidate("Mario Kart World", "Mario Kart World");
    const int mk8_wrong = ScoreArtworkCandidate("Mario Kart 8 Deluxe", "Mario Kart World");
    const int mkw_ost = ScoreArtworkCandidate("Mario Kart World Soundtrack", "Mario Kart World");
    CHECK(mkw_exact > 1000);
    CHECK(mkw_exact > mk8_wrong + 700);
    CHECK(mk8_wrong < 300);
    CHECK(mkw_ost < 300);

    const int dkb_exact = ScoreArtworkCandidate("Donkey Kong Bananza", "Donkey Kong Bananza");
    const int dkb_demo = ScoreArtworkCandidate("Donkey Kong Bananza Demo", "Donkey Kong Bananza");
    CHECK(dkb_exact > dkb_demo + 500);
}

TEST_CASE("Sidecar artwork cache stores validated PNGs and enforces failure cooldown",
          "[artwork_cache]") {
    using namespace Common::ArtworkCache;
    constexpr u64 test_tid = 0x0400BEEF12340000ULL;
    std::error_code ec;
    std::filesystem::remove(ArtworkPngPath(test_tid), ec);
    std::filesystem::remove(ArtworkMetaPath(test_tid), ec);

    CHECK(ShouldAutoFetchArtwork(test_tid, 1000));

    // Reject truncated/non-image payloads
    const std::array<u8, 12> garbage{'n', 'o', 't', '-', 'a', '-', 'p', 'n', 'g', 0, 1, 2};
    CHECK_FALSE(IsValidImageBytes(garbage));
    CHECK_FALSE(WriteCachedArtwork(test_tid, garbage, "Test Game", "unit_test"));

    // Record failure cooldown and verify auto-fetch backs off until retry_after
    RecordArtworkFailure(test_tid, "HTTP 503", 600);
    CHECK_FALSE(ShouldAutoFetchArtwork(test_tid, 0));

    // Valid HD PNG succeeds and populates sidecar metadata
    const auto png = MakeValidMinimalPngHeader(512, 512);
    REQUIRE(IsValidImageBytes(png));
    REQUIRE(WriteCachedArtwork(test_tid, png, "Donkey Kong Bananza", "switchiconshowdown"));
    CHECK(ReadCachedArtwork(test_tid) == png);
    CHECK(ReadSidecarTitle(test_tid) == "Donkey Kong Bananza");
    CHECK_FALSE(ShouldAutoFetchArtwork(test_tid, 9999999999ULL));

    std::filesystem::remove(ArtworkPngPath(test_tid), ec);
    std::filesystem::remove(ArtworkMetaPath(test_tid), ec);
}
