// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/artwork_cache.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <set>
#include <unordered_map>

#include <boost/algorithm/string/replace.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/string_util.h"

namespace Common::ArtworkCache {

namespace {

u64 CurrentUnixSeconds() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count());
}

u32 ReadBigEndianU32(const u8* p) noexcept {
    return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) |
           (static_cast<u32>(p[2]) << 8) | static_cast<u32>(p[3]);
}

bool LooksLikeHexId(std::string_view token) noexcept {
    if (token.size() != 16) {
        return false;
    }
    return std::ranges::all_of(token, [](unsigned char c) { return std::isxdigit(c) != 0; });
}

std::set<std::string> SplitSlugWords(std::string_view slug) {
    std::set<std::string> words;
    std::size_t start = 0;
    while (start < slug.size()) {
        const auto dash = slug.find('-', start);
        const auto word = slug.substr(start, dash - start);
        if (!word.empty()) {
            words.emplace(word);
        }
        if (dash == std::string_view::npos) {
            break;
        }
        start = dash + 1;
    }
    return words;
}

bool IsNoiseParenthetical(std::string_view inner) {
    const std::string slug = NormalizeSlug(inner);
    if (slug.empty() || LooksLikeHexId(slug)) {
        return true;
    }
    if (slug[0] == 'v' && slug.size() > 1 && std::isdigit(static_cast<unsigned char>(slug[1]))) {
        return true;
    }
    static const std::set<std::string_view> noise_tags{
        "usa",       "eur",       "jpn",      "world",   "kor",     "cht",    "chs",
        "aus",       "en",        "ja",       "multi",   "rev-1",   "rev-2",  "rev1",
        "rev2",      "update",    "upd",      "base",    "dlc",     "nup",    "nx",
        "nsp",       "xci",       "nsz",      "dump",    "dumped",  "repack", "trimmed",
        "switch-2",  "switch2",   "nx2",      "eshop",
    };
    return noise_tags.contains(slug);
}

} // namespace

const std::unordered_map<u64, std::string_view>& ProgramIdCatalog() {
    static const std::unordered_map<u64, std::string_view> catalog{
        // Nintendo Switch 2 titles
        {0x0400C3F00006E000ULL, "Mario Kart World"},
        {0x040074A01BF12000ULL, "Donkey Kong Bananza"},
        {0x0400D4201C6EA000ULL, "Mario Tennis Fever"},
        {0x0400486024DEC000ULL, "The Legend of Zelda: Ocarina of Time"},
        {0x04003F40232DE000ULL, "Hyrule Warriors: Age of Imprisonment"},
        {0x040041E0212AE000ULL, "Fast Fusion"},
        {0x04007EF01F8AE000ULL, "Fire Emblem: Fortune's Weave"},
        {0x0400E88025B1A000ULL, "Star Fox"},
        {0x0400D1D02399A000ULL, "The Duskbloods"},
        // Nintendo Switch 1 / cross-gen titles
        {0x0100000000010000ULL, "Super Mario Odyssey"},
        {0x0100152000022000ULL, "Mario Kart 8 Deluxe"},
        {0x01007EF00011E000ULL, "The Legend of Zelda: Breath of the Wild"},
        {0x0100F2C0115B6000ULL, "The Legend of Zelda: Tears of the Kingdom"},
        {0x01006A800016E000ULL, "Super Smash Bros. Ultimate"},
        {0x01006F8002326000ULL, "Animal Crossing: New Horizons"},
        {0x010015100B514000ULL, "Super Mario Bros. Wonder"},
        {0x0100C2500FC20000ULL, "Splatoon 3"},
        {0x010074F013262000ULL, "Xenoblade Chronicles 3"},
        {0x0100E95004038000ULL, "Xenoblade Chronicles 2"},
        {0x0100A6301214E000ULL, "Fire Emblem Engage"},
        {0x010055D009F78000ULL, "Fire Emblem: Three Houses"},
        {0x0100B7D0022EE000ULL, "Luigi's Mansion 3"},
        {0x0100F43008C44000ULL, "Pikmin 4"},
        {0x010093801237C000ULL, "Metroid Dread"},
        {0x0100A930198C0000ULL, "Metroid Prime Remastered"},
        {0x01004D300C5AE000ULL, "Kirby and the Forgotten Land"},
        {0x0100A3D008C5C000ULL, "Pokemon Scarlet"},
        {0x01008F6008C5E000ULL, "Pokemon Violet"},
        {0x01001F5010DFA000ULL, "Pokemon Legends: Arceus"},
        {0x0100F3801B768000ULL, "Pokemon Legends: Z-A"},
        {0x010028600EBDA000ULL, "Super Mario 3D World + Bowser's Fury"},
        {0x01008CF01BAAC000ULL, "The Legend of Zelda: Echoes of Wisdom"},
        {0x01006BB00C6F0000ULL, "The Legend of Zelda: Link's Awakening"},
        {0x01002DA013484000ULL, "The Legend of Zelda: Skyward Sword HD"},
        {0x01003A401B292000ULL, "Super Mario Party Jamboree"},
        {0x010049900F546000ULL, "Super Mario RPG"},
        {0x010086C01A04C000ULL, "Paper Mario: The Thousand-Year Door"},
    };
    return catalog;
}

std::string_view LookupTitleByProgramId(u64 title_id) noexcept {
    const u64 base = BaseProgramId(title_id);
    if (base == 0) {
        return {};
    }
    const auto& catalog = ProgramIdCatalog();
    if (const auto it = catalog.find(base); it != catalog.end()) {
        return it->second;
    }
    return {};
}

namespace {

bool IsStopWord(std::string_view word) {
    return word == "the" || word == "of" || word == "and" || word == "a" || word == "for" ||
           word == "to" || word == "in";
}

// Rebuilds a canonical "Series: Subtitle" catalog title when the cleaned query carries the series
// prefix plus every significant subtitle word, but without the colon (e.g. dump filenames such as
// "The Legend of Zelda Ocarina of Time Switch 2" resolve to "The Legend of Zelda: Ocarina of
// Time"). Returns empty when no catalog entry matches.
std::string ReconstructCatalogTitle(std::string_view cleaned) {
    const std::string query_slug = NormalizeSlug(cleaned);
    if (query_slug.empty()) {
        return {};
    }
    const auto query_words = SplitSlugWords(query_slug);
    for (const auto& [title_id, title] : ProgramIdCatalog()) {
        const auto colon = title.find(':');
        if (colon == std::string_view::npos) {
            continue;
        }
        const std::string_view series = title.substr(0, colon);
        std::string_view subtitle = title.substr(colon + 1);
        while (!subtitle.empty() && std::isspace(static_cast<unsigned char>(subtitle.front()))) {
            subtitle.remove_prefix(1);
        }
        const std::string series_slug = NormalizeSlug(series);
        if (series_slug.empty() || query_slug == series_slug) {
            continue;
        }
        if (!query_slug.starts_with(series_slug + "-")) {
            continue;
        }
        const auto sub_words = SplitSlugWords(NormalizeSlug(subtitle));
        bool all_subtitle_words_present = true;
        for (const auto& word : sub_words) {
            if (IsStopWord(word)) {
                continue;
            }
            if (!query_words.contains(word)) {
                all_subtitle_words_present = false;
                break;
            }
        }
        if (all_subtitle_words_present) {
            return std::string{title};
        }
    }
    return {};
}

} // namespace

std::string NormalizeSlug(std::string_view raw_title) {
    std::string title = Common::ToLower(std::string{raw_title});
    boost::replace_all(title, "\xC3\xA9", "e"); // é
    boost::replace_all(title, "\xC3\xA8", "e"); // è
    boost::replace_all(title, "\xC3\xAA", "e"); // ê
    boost::replace_all(title, "\xC5\x8D", "o"); // ō
    boost::replace_all(title, "&", " and ");
    std::ranges::replace(title, ' ', '-');
    std::ranges::replace(title, '_', '-');
    std::ranges::replace(title, '.', '-');
    std::erase_if(title, [](unsigned char c) { return !std::isalnum(c) && c != '-'; });
    title.erase(title.begin(),
                std::find_if(title.begin(), title.end(), [](char ch) { return ch != '-'; }));
    title.erase(std::find_if(title.rbegin(), title.rend(), [](char ch) { return ch != '-'; }).base(),
                title.end());
    title.erase(std::unique(title.begin(), title.end(),
                            [](char a, char b) { return a == '-' && b == '-'; }),
                title.end());
    return title;
}

std::string CleanGameSearchQuery(std::string_view raw_name) {
    // Strip known container extensions if a full filename was passed.
    for (const std::string_view ext : {".nup", ".nsp", ".nsz", ".xci", ".xcz", ".nx", ".nro"}) {
        if (raw_name.size() > ext.size() &&
            Common::ToLower(std::string{raw_name.substr(raw_name.size() - ext.size())}) == ext) {
            raw_name.remove_suffix(ext.size());
            break;
        }
    }
    // Trademark (U+2122) and registered (U+00AE) signs are display noise for search (e.g.
    // "Zelda\u2122: Ocarina of Time" must still resolve to the canonical article/search title).
    // They are removed as their UTF-8 byte sequences here, because decomposing already-cleaned
    // UTF-8 into code points would shift every index the byte-oriented filename parsing below
    // relies on. The rest of this function and the output stay untouched.
    std::string owned_name{raw_name};
    for (const std::string_view symbol : {"\xE2\x84\xA2", "\xC2\xAE"}) {
        for (std::size_t at = owned_name.find(symbol); at != std::string::npos;
             at = owned_name.find(symbol, at)) {
            owned_name.erase(at, symbol.size());
        }
    }
    raw_name = owned_name;

    // Strip [...] tags completely and strip noisy (...) tags such as (USA), (v1.0), (Rev 1).
    std::string stripped;
    stripped.reserve(raw_name.size());
    for (std::size_t i = 0; i < raw_name.size();) {
        if (raw_name[i] == '[') {
            const auto end = raw_name.find(']', i + 1);
            if (end != std::string_view::npos) {
                stripped.push_back(' ');
                i = end + 1;
                continue;
            }
        } else if (raw_name[i] == '(') {
            const auto end = raw_name.find(')', i + 1);
            if (end != std::string_view::npos) {
                const auto inner = raw_name.substr(i + 1, end - i - 1);
                if (IsNoiseParenthetical(inner)) {
                    stripped.push_back(' ');
                    i = end + 1;
                    continue;
                }
            }
        }
        // Trademark/registered symbols were already removed above, as UTF-8 byte sequences.
        stripped.push_back(raw_name[i]);
        ++i;
    }

    // Replace underscores and filename dot separators (when no spaces exist or around words).
    std::ranges::replace(stripped, '_', ' ');
    if (stripped.find(' ') == std::string::npos) {
        std::ranges::replace(stripped, '.', ' ');
    }

    // Tokenize on whitespace and drop standalone 16-hex Title IDs or v123456 dump suffixes.
    std::string cleaned;
    std::size_t pos = 0;
    while (pos < stripped.size()) {
        while (pos < stripped.size() && std::isspace(static_cast<unsigned char>(stripped[pos]))) {
            ++pos;
        }
        if (pos >= stripped.size()) {
            break;
        }
        std::size_t end = pos;
        while (end < stripped.size() && !std::isspace(static_cast<unsigned char>(stripped[end]))) {
            ++end;
        }
        const std::string_view tok{stripped.data() + pos, end - pos};
        pos = end;
        if (LooksLikeHexId(tok)) {
            continue;
        }
        if ((tok[0] == 'v' || tok[0] == 'V') && tok.size() >= 2 &&
            std::ranges::all_of(tok.substr(1), [](unsigned char c) {
                return std::isdigit(c) != 0 || c == '.';
            })) {
            continue;
        }
        const std::string lower_tok = Common::ToLower(std::string{tok});
        if (lower_tok == "nup" || lower_tok == "dump" || lower_tok == "base" ||
            lower_tok == "usa" || lower_tok == "eur" || lower_tok == "jpn") {
            continue;
        }
        // Drop platform tags ("Switch 2", "Switch2", "NX2") that dump filenames append after the
        // real title. A bare "Switch" is only a platform tag when followed by a number or when it
        // is the last token, so titles such as "Switch Sports" keep their leading word.
        if (lower_tok == "switch2" || lower_tok == "switch-2" || lower_tok == "nx2") {
            continue;
        }
        if (lower_tok == "switch") {
            std::size_t next = pos;
            while (next < stripped.size() && std::isspace(static_cast<unsigned char>(stripped[next]))) {
                ++next;
            }
            std::size_t next_end = next;
            while (next_end < stripped.size() &&
                   !std::isspace(static_cast<unsigned char>(stripped[next_end]))) {
                ++next_end;
            }
            const bool trailing_number =
                next_end > next &&
                std::ranges::all_of(std::string_view{stripped.data() + next, next_end - next},
                                    [](unsigned char c) { return std::isdigit(c) != 0; });
            if (next_end == next || trailing_number) {
                pos = next_end;
                continue;
            }
        }
        if (!cleaned.empty()) {
            cleaned.push_back(' ');
        }
        cleaned.append(tok);
    }

    static const std::unordered_map<std::string_view, std::string_view> aliases{
        {"mkw", "Mario Kart World"},
        {"mariokartworld", "Mario Kart World"},
        {"mario-kart-world", "Mario Kart World"},
        {"dkb", "Donkey Kong Bananza"},
        {"bananza", "Donkey Kong Bananza"},
        {"donkeykongbananza", "Donkey Kong Bananza"},
        {"donkey-kong-bananza", "Donkey Kong Bananza"},
        {"oot", "The Legend of Zelda: Ocarina of Time"},
        {"ocarina-of-time", "The Legend of Zelda: Ocarina of Time"},
        {"ocarinaoftime", "The Legend of Zelda: Ocarina of Time"},
        {"the-legend-of-zelda-ocarina-of-time", "The Legend of Zelda: Ocarina of Time"},
        {"the-legend-of-zelda-ocarina-of-time-switch", "The Legend of Zelda: Ocarina of Time"},
        {"the-legend-of-zelda-ocarina-of-time-switch-2", "The Legend of Zelda: Ocarina of Time"},
        {"legend-of-zelda-ocarina-of-time", "The Legend of Zelda: Ocarina of Time"},
        {"zelda-ocarina-of-time", "The Legend of Zelda: Ocarina of Time"},
        {"welcometour", "Nintendo Switch 2 Welcome Tour"},
        {"nintendo-welcome-tour", "Nintendo Switch 2 Welcome Tour"},
        {"welcome-tour", "Nintendo Switch 2 Welcome Tour"},
        {"ns2wt", "Nintendo Switch 2 Welcome Tour"},
        {"switch-2-welcome-tour", "Nintendo Switch 2 Welcome Tour"},
        {"nintendo-switch-2-welcome-tour", "Nintendo Switch 2 Welcome Tour"},
        {"kar", "Kirby Air Riders"},
        {"kirbyairriders", "Kirby Air Riders"},
        {"mp4", "Metroid Prime 4: Beyond"},
        {"metroidprime4", "Metroid Prime 4: Beyond"},
        {"plza", "Pokemon Legends: Z-A"},
        {"pokemonlegendsza", "Pokemon Legends: Z-A"},
        {"fastfusion", "Fast Fusion"},
        {"hwaoi", "Hyrule Warriors: Age of Imprisonment"},
        {"mk8", "Mario Kart 8 Deluxe"},
        {"mk8d", "Mario Kart 8 Deluxe"},
        {"smo", "Super Mario Odyssey"},
        {"oot", "The Legend of Zelda: Ocarina of Time"},
        {"ocarina", "The Legend of Zelda: Ocarina of Time"},
        {"ocarina-of-time", "The Legend of Zelda: Ocarina of Time"},
        {"ocarinaoftime", "The Legend of Zelda: Ocarina of Time"},
        {"zelda-oot", "The Legend of Zelda: Ocarina of Time"},
        {"zeldaoot", "The Legend of Zelda: Ocarina of Time"},
        {"tloz-oot", "The Legend of Zelda: Ocarina of Time"},
        {"tlozoot", "The Legend of Zelda: Ocarina of Time"},
        {"botw", "The Legend of Zelda: Breath of the Wild"},
        {"totk", "The Legend of Zelda: Tears of the Kingdom"},
        {"ssbu", "Super Smash Bros. Ultimate"},
        {"acnh", "Animal Crossing: New Horizons"},
        {"smbw", "Super Mario Bros. Wonder"},
        {"splatoon3", "Splatoon 3"},
        {"xc2", "Xenoblade Chronicles 2"},
        {"xc3", "Xenoblade Chronicles 3"},
        {"lm3", "Luigi's Mansion 3"},
        {"pikmin4", "Pikmin 4"},
    };
    const std::string slug = NormalizeSlug(cleaned);
    if (const auto it = aliases.find(slug); it != aliases.end()) {
        return std::string{it->second};
    }
    return cleaned;
}

std::string ResolveCanonicalSearchTitle(std::string_view raw_name, u64 title_id) {
    // Exact Program ID match takes top priority so noisy dump filenames never pick the wrong game.
    if (const auto exact = LookupTitleByProgramId(title_id); !exact.empty()) {
        const std::string cleaned = CleanGameSearchQuery(raw_name);
        // Only keep raw_name when a caller explicitly typed a non-empty, non-hex query whose slug
        // already shares core words with the exact title or when title_id was not specified.
        if (cleaned.empty() || NormalizeSlug(cleaned) == NormalizeSlug(exact)) {
            return std::string{exact};
        }
        // Even if raw_name was a filename like "game_dump_final", exact Title ID is authoritative.
        return std::string{exact};
    }
    const std::string cleaned = CleanGameSearchQuery(raw_name);
    // Reconstruct the canonical "Series: Subtitle" title for dump filenames that dropped the colon
    // (e.g. "The Legend of Zelda Ocarina of Time Switch 2" -> "The Legend of Zelda: Ocarina of
    // Time") so the keyless Libretro/Wikipedia fallbacks can match the real boxart names.
    if (const auto reconstructed = ReconstructCatalogTitle(cleaned); !reconstructed.empty()) {
        return reconstructed;
    }
    return cleaned;
}

std::string ExtractSubtitleQuery(std::string_view title) {
    const auto colon = title.find(':');
    if (colon == std::string_view::npos || colon + 1 >= title.size()) {
        return {};
    }
    std::string_view sub = title.substr(colon + 1);
    while (!sub.empty() && std::isspace(static_cast<unsigned char>(sub.front()))) {
        sub.remove_prefix(1);
    }
    while (!sub.empty() && std::isspace(static_cast<unsigned char>(sub.back()))) {
        sub.remove_suffix(1);
    }
    std::size_t alnum_count = 0;
    for (const unsigned char c : sub) {
        if (std::isalnum(c)) {
            ++alnum_count;
        }
    }
    if (alnum_count < 4) {
        return {};
    }
    return std::string{sub};
}

std::vector<std::string> BuildLibretroNameVariants(std::string_view title) {
    const auto sanitize = [](std::string_view input) {
        std::string out;
        out.reserve(input.size());
        for (const char c : input) {
            switch (c) {
            case '&':
            case '*':
            case '/':
            case ':':
            case '`':
            case '<':
            case '>':
            case '?':
            case '\\':
            case '|':
            case '"':
                out.push_back('_');
                break;
            default:
                out.push_back(c);
                break;
            }
        }
        return out;
    };

    // Replace ": " or ":" with " - " (No-Intro / Libretro convention).
    std::string dash_form;
    dash_form.reserve(title.size() + 4);
    for (std::size_t i = 0; i < title.size(); ++i) {
        if (title[i] == ':') {
            while (!dash_form.empty() && dash_form.back() == ' ') {
                dash_form.pop_back();
            }
            dash_form += " - ";
            while (i + 1 < title.size() && title[i + 1] == ' ') {
                ++i;
            }
        } else {
            dash_form.push_back(title[i]);
        }
    }

    std::vector<std::string> variants;
    const auto add_unique = [&variants](std::string s) {
        if (!s.empty() && std::ranges::find(variants, s) == variants.end()) {
            variants.push_back(std::move(s));
        }
    };

    // Leading "The " moves to ", The" before " - " (e.g. "Legend of Zelda, The - Ocarina of Time").
    if (dash_form.size() > 4 && Common::ToLower(dash_form.substr(0, 4)) == "the ") {
        const std::string rest = dash_form.substr(4);
        if (const auto sep = rest.find(" - "); sep != std::string::npos) {
            add_unique(sanitize(rest.substr(0, sep) + ", The" + rest.substr(sep)));
        } else {
            add_unique(sanitize(rest + ", The"));
        }
    }

    add_unique(sanitize(dash_form));
    add_unique(sanitize(title));
    return variants;
}

int ScoreArtworkCandidate(std::string_view candidate, std::string_view target_query) {
    const std::string c_slug = NormalizeSlug(candidate);
    const std::string q_slug = NormalizeSlug(target_query);
    if (c_slug.empty() || q_slug.empty()) {
        return -1000;
    }
    if (c_slug == q_slug) {
        return 2000;
    }

    const auto q_words = SplitSlugWords(q_slug);
    const auto c_words = SplitSlugWords(c_slug);
    static const std::set<std::string_view> stop_words{"the", "of", "and", "a", "for", "to", "in"};
    static const std::set<std::string_view> benign_edition_words{
        "nintendo", "switch", "edition", "hd", "3d", "deluxe", "remastered", "dx",
    };

    int score = 0;
    if (c_slug == q_slug + "-nintendo-switch-2-edition" ||
        q_slug == c_slug + "-nintendo-switch-2-edition") {
        score += 750;
    } else if (c_slug.starts_with(q_slug + "-")) {
        // Candidate extends the full target query (e.g. "Ocarina of Time 3D").
        const auto suffix_words = SplitSlugWords(std::string_view{c_slug}.substr(q_slug.size() + 1));
        int non_edition_extra = 0;
        for (const auto& w : suffix_words) {
            if (!stop_words.contains(w) && !benign_edition_words.contains(w)) {
                ++non_edition_extra;
            }
        }
        if (non_edition_extra == 0) {
            score += 450 - static_cast<int>(c_slug.size() - q_slug.size()) * 4;
        } else {
            score += 200;
        }
    } else if (q_slug.starts_with(c_slug + "-")) {
        // Candidate is a shorter prefix of target query: ONLY reward if the omitted suffix is
        // harmless edition metadata, NEVER when it drops significant subtitle words (such as
        // matching bare "The Legend of Zelda" against "The Legend of Zelda: Ocarina of Time").
        const auto missing_suffix =
            SplitSlugWords(std::string_view{q_slug}.substr(c_slug.size() + 1));
        int dropped_significant = 0;
        for (const auto& w : missing_suffix) {
            if (!stop_words.contains(w) && !benign_edition_words.contains(w)) {
                ++dropped_significant;
            }
        }
        if (dropped_significant == 0) {
            score += 450 - static_cast<int>(q_slug.size() - c_slug.size()) * 4;
        }
    } else if (c_slug.find(q_slug) != std::string::npos) {
        score += 450;
    }

    int overlap = 0;
    int missing_significant = 0;
    for (const auto& w : q_words) {
        if (c_words.contains(w)) {
            overlap += stop_words.contains(w) ? 25 : 140;
        } else if (!stop_words.contains(w)) {
            ++missing_significant;
        }
    }

    // Penalize significant candidate words absent from target query (e.g. "Breath of the Wild"
    // or "Age of Calamity" when searching for "Ocarina of Time" or "Age of Imprisonment").
    int extra_significant = 0;
    for (const auto& w : c_words) {
        if (q_words.contains(w) || stop_words.contains(w)) {
            continue;
        }
        if (missing_significant == 0 && benign_edition_words.contains(w)) {
            continue;
        }
        ++extra_significant;
    }

    score += overlap - missing_significant * 240 - extra_significant * 220 -
             static_cast<int>(c_words.size()) * 12;

    // Subtitle guard: when target query specifies a subtitle after ':' (or " - "), require
    // meaningful subtitle word overlap so a franchise base title or a different sequel in the
    // same series cannot score high just from sharing the franchise prefix ("The Legend of Zelda").
    const std::string target_sub = ExtractSubtitleQuery(target_query);
    if (!target_sub.empty()) {
        const auto sub_words = SplitSlugWords(NormalizeSlug(target_sub));
        int sub_sig_total = 0;
        int sub_sig_matched = 0;
        for (const auto& w : sub_words) {
            if (stop_words.contains(w)) {
                continue;
            }
            ++sub_sig_total;
            if (c_words.contains(w)) {
                ++sub_sig_matched;
            }
        }
        if (sub_sig_total > 0 && sub_sig_matched == 0) {
            score -= 900;
        } else if (sub_sig_total > 1 && sub_sig_matched < sub_sig_total) {
            score -= (sub_sig_total - sub_sig_matched) * 180;
        } else if (sub_sig_total > 0 && sub_sig_matched == sub_sig_total &&
                   extra_significant == 0 && missing_significant > 0 &&
                   (sub_sig_total >= 2 || target_sub.size() >= 5)) {
            // Candidate cleanly matches the full distinctive subtitle with no conflicting words
            // (e.g. "Ocarina of Time" or "Zelda: Ocarina of Time" when searching for
            // "The Legend of Zelda: Ocarina of Time").
            score += 580;
        }
    }

    // Guard against wrong editions, demos, soundtracks, or DLC packs unless explicitly queried.
    static constexpr std::array<std::string_view, 10> bad_edition_tokens{
        "demo",       "trial",        "playtest", "soundtrack",   "ost",
        "artbook",    "dlc",          "pass",     "upgrade-pack", "amiibo",
    };
    for (const auto bad : bad_edition_tokens) {
        const std::string bad_str{bad};
        if (c_words.contains(bad_str) && !q_words.contains(bad_str)) {
            score -= 1000;
        }
    }
    const std::string lower_cand = Common::ToLower(std::string{candidate});
    if (lower_cand.find("(series)") != std::string::npos ||
        lower_cand.find("(character)") != std::string::npos) {
        score -= 650;
    }

    // Penalize mismatched sequel numerals (e.g., "Mario Kart 8 Deluxe" vs "Mario Kart World").
    static constexpr std::array<std::string_view, 12> sequel_tokens{
        "2", "3", "4", "5", "6", "7", "8", "9", "10", "ii", "iii", "iv",
    };
    for (const auto num : sequel_tokens) {
        const std::string num_str{num};
        const bool in_c = c_words.contains(num_str);
        const bool in_q = q_words.contains(num_str);
        if (in_c != in_q) {
            score -= 400;
        }
    }

    return score;
}

bool IsValidImageBytes(std::span<const u8> bytes) noexcept {
    if (bytes.size() < 64) {
        return false;
    }
    // PNG signature + IHDR chunk header check
    static constexpr std::array<u8, 8> png_sig{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (std::equal(png_sig.begin(), png_sig.end(), bytes.begin())) {
        if (bytes[12] != 'I' || bytes[13] != 'H' || bytes[14] != 'D' || bytes[15] != 'R') {
            return false;
        }
        const u32 width = ReadBigEndianU32(bytes.data() + 16);
        const u32 height = ReadBigEndianU32(bytes.data() + 20);
        return width >= 32 && height >= 32 && width <= 4096 && height <= 4096;
    }
    // JPEG SOI + marker check
    if (bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF) {
        return true;
    }
    return false;
}

std::filesystem::path ArtworkDirectory() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "artwork";
}

std::filesystem::path ArtworkPngPath(u64 title_id) {
    return ArtworkDirectory() / fmt::format("{:016X}.png", BaseProgramId(title_id));
}

std::filesystem::path ArtworkMetaPath(u64 title_id) {
    return ArtworkDirectory() / fmt::format("{:016X}.json", BaseProgramId(title_id));
}

std::vector<u8> ReadCachedArtwork(u64 title_id) {
    const u64 base = BaseProgramId(title_id);
    if (base == 0) {
        return {};
    }
    const auto path = ArtworkPngPath(base);
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size < 64 || size > 16 * 1024 * 1024) {
        return {};
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::vector<u8> data(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (in.gcount() != static_cast<std::streamsize>(data.size()) || !IsValidImageBytes(data)) {
        return {};
    }
    return data;
}

std::string ReadSidecarTitle(u64 title_id) {
    const u64 base = BaseProgramId(title_id);
    if (base == 0) {
        return {};
    }
    const auto meta_path = ArtworkMetaPath(base);
    std::error_code ec;
    if (std::filesystem::is_regular_file(meta_path, ec)) {
        std::ifstream in(meta_path, std::ios::binary);
        if (in) {
            const auto json = nlohmann::json::parse(in, nullptr, false);
            if (!json.is_discarded() && json.is_object() && json.contains("resolved_title") &&
                json["resolved_title"].is_string()) {
                const auto resolved = json["resolved_title"].get<std::string>();
                if (!resolved.empty()) {
                    return resolved;
                }
            }
        }
    }
    if (const auto exact = LookupTitleByProgramId(base); !exact.empty()) {
        return std::string{exact};
    }
    return {};
}

bool WriteCachedArtwork(u64 title_id, std::span<const u8> png_bytes,
                        std::string_view resolved_title, std::string_view source) {
    const u64 base = BaseProgramId(title_id);
    if (base == 0 || !IsValidImageBytes(png_bytes)) {
        return false;
    }
    std::error_code ec;
    const auto dir = ArtworkDirectory();
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARNING(Frontend, "Could not create artwork cache directory: {}", ec.message());
        return false;
    }

    const auto png_path = ArtworkPngPath(base);
    const auto tmp_png = dir / fmt::format("{:016X}.png.tmp", base);
    {
        std::ofstream out(tmp_png, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        out.write(reinterpret_cast<const char*>(png_bytes.data()),
                  static_cast<std::streamsize>(png_bytes.size()));
        out.flush();
        if (!out) {
            std::filesystem::remove(tmp_png, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp_png, png_path, ec);
    if (ec) {
        std::filesystem::remove(png_path, ec);
        ec.clear();
        std::filesystem::rename(tmp_png, png_path, ec);
        if (ec) {
            std::filesystem::remove(tmp_png, ec);
            return false;
        }
    }

    const nlohmann::json meta{
        {"title_id", fmt::format("{:016X}", base)},
        {"status", "ok"},
        {"resolved_title", resolved_title},
        {"source", source},
        {"fetched_at_unix", CurrentUnixSeconds()},
    };
    const auto meta_path = ArtworkMetaPath(base);
    const auto tmp_meta = dir / fmt::format("{:016X}.json.tmp", base);
    {
        std::ofstream out(tmp_meta, std::ios::binary | std::ios::trunc);
        if (out) {
            out << meta.dump(2) << '\n';
            out.flush();
        }
        const bool written = static_cast<bool>(out);
        out.close();
        if (written && out) {
            std::filesystem::rename(tmp_meta, meta_path, ec);
            if (ec) {
                std::filesystem::remove(meta_path, ec);
                ec.clear();
                std::filesystem::rename(tmp_meta, meta_path, ec);
            }
        }
    }
    return true;
}

void ClearCachedArtwork(u64 title_id) {
    const u64 base = BaseProgramId(title_id);
    if (base == 0) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(ArtworkPngPath(base), ec);
    ec.clear();
    std::filesystem::remove(ArtworkMetaPath(base), ec);
}

void RecordArtworkFailure(u64 title_id, std::string_view reason, u64 cooldown_seconds) {
    const u64 base = BaseProgramId(title_id);
    if (base == 0 || !ReadCachedArtwork(base).empty()) {
        return;
    }
    std::error_code ec;
    const auto dir = ArtworkDirectory();
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return;
    }
    const u64 now = CurrentUnixSeconds();
    const nlohmann::json meta{
        {"title_id", fmt::format("{:016X}", base)},
        {"status", "failed"},
        {"error", reason},
        {"failed_at_unix", now},
        {"retry_after_unix", now + cooldown_seconds},
    };
    const auto meta_path = ArtworkMetaPath(base);
    const auto tmp_meta = dir / fmt::format("{:016X}.json.tmp", base);
    std::ofstream out(tmp_meta, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    out << meta.dump(2) << '\n';
    out.flush();
    const bool written = static_cast<bool>(out);
    out.close();
    if (written && out) {
        std::filesystem::rename(tmp_meta, meta_path, ec);
        if (ec) {
            std::filesystem::remove(meta_path, ec);
            ec.clear();
            std::filesystem::rename(tmp_meta, meta_path, ec);
        }
    }
}

bool ShouldAutoFetchArtwork(u64 title_id, u64 now_unix_seconds) {
    const u64 base = BaseProgramId(title_id);
    if (base == 0) {
        return false;
    }
    if (!ReadCachedArtwork(base).empty()) {
        return false;
    }
    const auto meta_path = ArtworkMetaPath(base);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(meta_path, ec)) {
        return true;
    }
    std::ifstream in(meta_path, std::ios::binary);
    if (!in) {
        return true;
    }
    const auto json = nlohmann::json::parse(in, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        return true;
    }
    if (json.contains("retry_after_unix") && json["retry_after_unix"].is_number_unsigned()) {
        const u64 now = now_unix_seconds != 0 ? now_unix_seconds : CurrentUnixSeconds();
        return now >= json["retry_after_unix"].get<u64>();
    }
    return true;
}

} // namespace Common::ArtworkCache
