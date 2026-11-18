// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/artwork_cache.h"
#include "common/fs/path_util.h"
#include "common/scope_exit.h"

namespace {
// A generated 32x32 black PNG, no external artwork.
constexpr std::array<u8, 75> test_png{
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x20, 0x08, 0x00, 0x00, 0x00, 0x00, 0x56,
    0x11, 0x25, 0x28, 0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60,
    0x18, 0x05, 0xa3, 0x60, 0x14, 0x8c, 0x02, 0xec, 0x00, 0x00, 0x04, 0x20, 0x00, 0x01, 0x86,
    0xb3, 0x3d, 0x44, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
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

TEST_CASE("Artwork queries strip UTF-8 trademark and registered signs", "[artwork_cache]") {
    using Common::ArtworkCache::CleanGameSearchQuery;
    using Common::ArtworkCache::NormalizeSlug;
    using Common::ArtworkCache::ResolveCanonicalSearchTitle;

    // The signs are removed as their UTF-8 byte sequences (U+2122 is E2 84 A2, U+00AE is C2 AE).
    // Comparing single bytes to a Unicode code point does not compile with clang-cl, so these
    // cases pin the behaviour as text a dump filename can actually contain.
    CHECK(CleanGameSearchQuery("The Legend of Zelda\u2122: Ocarina of Time") ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(CleanGameSearchQuery("Nintendo\u00AE Switch Sports") == "Nintendo Switch Sports");
    CHECK(CleanGameSearchQuery("Mario Kart World\u2122\u00AE") == "Mario Kart World");

    // Both signs in one title, once with a dump suffix and platform tag.
    CHECK(CleanGameSearchQuery("Kirby\u2122 Air Riders\u00AE [040077201BE50000](USA).nup") ==
          "Kirby Air Riders");
    // A sign in the middle of a word must not leave a gap or join the surrounding words.
    CHECK(CleanGameSearchQuery("Pok\u00E9mon\u2122 Pokopia") == "Pok\u00E9mon Pokopia");

    // A sign never changes resolution: the canonical Ocarina of Time title still comes out.
    CHECK(ResolveCanonicalSearchTitle("The Legend of Zelda\u2122 Ocarina of Time Switch 2", 0) ==
          "The Legend of Zelda: Ocarina of Time");
    CHECK(ResolveCanonicalSearchTitle("Zelda\u2122: Ocarina of Time", 0x0400486024DEC000ULL) ==
          "The Legend of Zelda: Ocarina of Time");

    // Ordinary ASCII titles are unchanged, so the sign handling cannot rewrite normal queries.
    CHECK(CleanGameSearchQuery("Switch Sports") == "Switch Sports");
    CHECK(CleanGameSearchQuery("Donkey Kong Bananza") == "Donkey Kong Bananza");
    CHECK(NormalizeSlug("The Legend of Zelda\u2122") == "the-legend-of-zelda");
    CHECK(NormalizeSlug("Mario Kart World") == "mario-kart-world");
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

TEST_CASE("Artwork sidecars replace atomically and reject corrupt or mis-keyed data",
          "[artwork_cache]") {
    using namespace Common::ArtworkCache;
    using namespace Common::FS;
    const auto original_cache = GetEdenPath(EdenPath::CacheDir);
    const auto parent =
        std::filesystem::temp_directory_path() /
        ("veshiru-artwork-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(std::filesystem::create_directory(parent));
    SetEdenPath(EdenPath::CacheDir, parent);
    SCOPE_EXIT {
        SetEdenPath(EdenPath::CacheDir, original_cache);
        std::error_code ec;
        std::filesystem::remove_all(parent, ec);
    };

    constexpr u64 title = 0x0400BEEF12340000ULL;
    CHECK_FALSE(IsSafeArtworkTitle("/home/test/game.nsp"));
    CHECK_FALSE(IsSafeArtworkTitle("<img src=x>"));
    CHECK_FALSE(IsValidImageBytes(std::span{test_png}.first(40)));
    auto huge = test_png;
    huge[17] = 1;
    CHECK_FALSE(IsValidImageBytes(huge));
    REQUIRE(IsValidImageBytes(test_png));
    REQUIRE(WriteCachedArtwork(title, test_png, "First local title", "unit_test"));
    CHECK(ReadCachedArtwork(title) == std::vector<u8>(test_png.begin(), test_png.end()));
    CHECK(ReadSidecarTitle(title) == "First local title");

    // A rejected replacement leaves old artwork and sidecar intact.
    CHECK_FALSE(WriteCachedArtwork(title, huge, "Wrong", "unit_test"));
    CHECK(ReadSidecarTitle(title) == "First local title");
    CHECK(WriteCachedArtwork(title, test_png, "New local title", "unit_test"));
    CHECK(ReadSidecarTitle(title) == "New local title");
    std::ofstream{ArtworkMetaPath(title)}
        << R"({"schema":1,"title_id":"0000000000000000","status":"ok","resolved_title":"Wrong title"})";
    CHECK(ReadSidecarTitle(title).empty());
    std::ofstream{ArtworkMetaPath(title)} << std::string(10'000, ' ');
    CHECK(ReadSidecarTitle(title).empty());

    // A full cache is read-only; do not discard old artwork or create temporary files.
    const auto fill = ArtworkDirectory() / "fill";
    std::ofstream{fill}.close();
    std::filesystem::resize_file(fill, 128 * 1024 * 1024);
    CHECK_FALSE(WriteCachedArtwork(title, test_png, "Over quota", "unit_test"));
    CHECK(ReadCachedArtwork(title) == std::vector<u8>(test_png.begin(), test_png.end()));
    ClearCachedArtwork(title);
    CHECK(ReadCachedArtwork(title).empty());
}
