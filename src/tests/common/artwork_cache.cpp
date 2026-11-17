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
    CHECK(ResolveCanonicalSearchTitle("dkb_dump", 0x040074A01BF12000ULL) ==
          "Donkey Kong Bananza");
    CHECK(ResolveCanonicalSearchTitle("", 0x0400D4201C6EA000ULL) == "Mario Tennis Fever");
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
    CHECK(CleanGameSearchQuery("oot") == "The Legend of Zelda: Ocarina of Time");
    CHECK(CleanGameSearchQuery("Ocarina_of_Time_[0400486024DEC000].nup") ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(CleanGameSearchQuery("zelda-oot") == "The Legend of Zelda: Ocarina of Time");
}

TEST_CASE("Artwork OOT refresh resolves dump filenames and platform-tagged titles",
          "[artwork_cache]") {
    using Common::ArtworkCache::CleanGameSearchQuery;
    using Common::ArtworkCache::ResolveCanonicalSearchTitle;

    // Regression (1.5): "The Legend of Zelda Ocarina of Time Switch 2" style dump filenames must
    // resolve to the canonical colon title so the Libretro/Wikipedia fallbacks can find the art.
    CHECK(CleanGameSearchQuery("The.Legend.of.Zelda.Ocarina.of.Time.Switch.2.nup") ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(CleanGameSearchQuery("The.Legend.of.Zelda.Ocarina.of.Time.Switch.nup") ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(CleanGameSearchQuery("The_Legend_of_Zelda_Ocarina_of_Time_[0400486024DEC000][v0](USA).nup") ==
          "The Legend of Zelda: Ocarina of Time");

    // Trademark symbols are stripped before matching.
    CHECK(CleanGameSearchQuery("The Legend of Zelda\u2122: Ocarina of Time") ==
          "The Legend of Zelda: Ocarina of Time");

    // A leading "Switch" that is a real part of a title (not a "Switch 2" platform tag) survives.
    CHECK(CleanGameSearchQuery("Switch Sports") == "Switch Sports");

    // Reconstruction also works when the Program ID is unknown and the colon was dropped.
    CHECK(ResolveCanonicalSearchTitle("The Legend of Zelda Ocarina of Time Switch 2", 0) ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(ResolveCanonicalSearchTitle("Hyrule Warriors Age of Imprisonment", 0) ==
          "Hyrule Warriors: Age of Imprisonment");

    // The bare franchise prefix must NOT be reconstructed into a specific sequel.
    CHECK(ResolveCanonicalSearchTitle("The Legend of Zelda", 0) == "The Legend of Zelda");
    CHECK(ResolveCanonicalSearchTitle("The Legend of Zelda Breath of the Wild", 0) ==
          "The Legend of Zelda: Breath of the Wild");
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

    // Regression: Ocarina of Time must never match bare "The Legend of Zelda" or unrelated Zelda
    // games (the v1.4.0 bug where franchise prefix matching scored +528 >= 300).
    const std::string oot = "The Legend of Zelda: Ocarina of Time";
    const int oot_exact = ScoreArtworkCandidate("The Legend of Zelda: Ocarina of Time", oot);
    const int oot_3d = ScoreArtworkCandidate("The Legend of Zelda: Ocarina of Time 3D", oot);
    const int oot_short = ScoreArtworkCandidate("Ocarina of Time", oot);
    const int oot_zelda_short = ScoreArtworkCandidate("Zelda: Ocarina of Time", oot);
    const int oot_ost = ScoreArtworkCandidate("Ocarina of Time Soundtrack", oot);
    const int zelda_bare = ScoreArtworkCandidate("The Legend of Zelda", oot);
    const int zelda_botw = ScoreArtworkCandidate("The Legend of Zelda: Breath of the Wild", oot);
    const int zelda_totk = ScoreArtworkCandidate("The Legend of Zelda: Tears of the Kingdom", oot);
    const int zelda_la = ScoreArtworkCandidate("The Legend of Zelda: Link's Awakening", oot);
    const int zelda_eow = ScoreArtworkCandidate("The Legend of Zelda: Echoes of Wisdom", oot);
    CHECK(oot_exact > 1000);
    CHECK(oot_3d >= 300);
    CHECK(oot_short >= 300);
    CHECK(oot_zelda_short >= 300);
    CHECK(oot_exact > oot_3d);
    CHECK(oot_exact > oot_zelda_short);
    CHECK(oot_ost < 300);
    CHECK(zelda_bare < 300);
    CHECK(zelda_botw < 300);
    CHECK(zelda_totk < 300);
    CHECK(zelda_la < 300);
    CHECK(zelda_eow < 300);

    // Subtitle guard also separates distinct entries in other colon-separated series
    const std::string aoi = "Hyrule Warriors: Age of Imprisonment";
    CHECK(ScoreArtworkCandidate("Hyrule Warriors: Age of Imprisonment", aoi) > 1000);
    CHECK(ScoreArtworkCandidate("Hyrule Warriors: Age of Calamity", aoi) < 300);
    CHECK(ScoreArtworkCandidate("Hyrule Warriors", aoi) < 300);
}

TEST_CASE("Artwork subtitle extraction and Libretro No-Intro name variants", "[artwork_cache]") {
    using namespace Common::ArtworkCache;

    CHECK(ExtractSubtitleQuery("The Legend of Zelda: Ocarina of Time") == "Ocarina of Time");
    CHECK(ExtractSubtitleQuery("Hyrule Warriors: Age of Imprisonment") == "Age of Imprisonment");
    CHECK(ExtractSubtitleQuery("Mario Kart World").empty());

    const auto oot_variants = BuildLibretroNameVariants("The Legend of Zelda: Ocarina of Time");
    REQUIRE(oot_variants.size() >= 2);
    CHECK(oot_variants[0] == "Legend of Zelda, The - Ocarina of Time");
    CHECK(oot_variants[1] == "The Legend of Zelda - Ocarina of Time");

    const auto mkw_variants = BuildLibretroNameVariants("Mario Kart World");
    REQUIRE(mkw_variants.size() == 1);
    CHECK(mkw_variants[0] == "Mario Kart World");
}

TEST_CASE("Sidecar artwork cache stores validated PNGs, clears cleanly, and enforces cooldown",
          "[artwork_cache]") {
    using namespace Common::ArtworkCache;
    constexpr u64 test_tid = 0x0400BEEF12340000ULL;
    ClearCachedArtwork(test_tid);

    CHECK(ShouldAutoFetchArtwork(test_tid, 1000));

    // Reject truncated/non-image payloads
    const std::array<u8, 12> garbage{'n', 'o', 't', '-', 'a', '-', 'p', 'n', 'g', 0, 1, 2};
    CHECK_FALSE(IsValidImageBytes(garbage));
    CHECK_FALSE(WriteCachedArtwork(test_tid, garbage, "Test Game", "unit_test"));

    // Record failure cooldown and verify auto-fetch backs off until retry_after
    RecordArtworkFailure(test_tid, "HTTP 503", 600);
    CHECK_FALSE(ShouldAutoFetchArtwork(test_tid, 0));

    // Clearing cached artwork removes failure cooldown immediately
    ClearCachedArtwork(test_tid);
    CHECK(ShouldAutoFetchArtwork(test_tid, 0));

    // Valid HD PNG succeeds and populates sidecar metadata
    const auto png = MakeValidMinimalPngHeader(512, 512);
    REQUIRE(IsValidImageBytes(png));
    REQUIRE(WriteCachedArtwork(test_tid, png, "Donkey Kong Bananza", "switchiconshowdown"));
    CHECK(ReadCachedArtwork(test_tid) == png);
    CHECK(ReadSidecarTitle(test_tid) == "Donkey Kong Bananza");
    CHECK_FALSE(ShouldAutoFetchArtwork(test_tid, 9999999999ULL));

    // Clearing cached artwork removes both the cached PNG and sidecar metadata
    ClearCachedArtwork(test_tid);
    CHECK(ReadCachedArtwork(test_tid).empty());
    CHECK(ReadSidecarTitle(test_tid).empty());
}
