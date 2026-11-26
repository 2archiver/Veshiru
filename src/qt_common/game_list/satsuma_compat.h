// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace QtCommon::Game {

/// A status from Satsuma's compatibility list, best first.
struct SatsumaCompatStatus {
    int rank;               ///< Sort order, 0 = best
    std::string_view color; ///< Circle colour
    const char* text;       ///< Column text
    const char* tooltip;    ///< The status's meaning from the list
};

inline constexpr SatsumaCompatStatus CompatPlayable{
    0, "#47d35c", "Playable", "Can be played start to finish with minor issues"};
inline constexpr SatsumaCompatStatus CompatInGamePlus{
    1, "#a3d93c", "In Game +",
    "Plays well, but finishing it start to finish and steady performance are not assured"};
inline constexpr SatsumaCompatStatus CompatInGame{
    2, "#f2d624", "In Game", "Gets into gameplay, with issues that block or spoil play"};
inline constexpr SatsumaCompatStatus CompatMenus{3, "#e8553f", "Menus",
                                                 "Reaches menus or title screen only"};
inline constexpr SatsumaCompatStatus CompatBoots{4, "#828282", "Boots",
                                                 "Starts, but does not reach menus"};

/// The status Satsuma's compatibility list records for a program ID; nullopt for a game not on it.
inline std::optional<SatsumaCompatStatus> FindSatsumaCompat(u64 program_id) {
    struct Entry {
        u64 program_id;
        const SatsumaCompatStatus* status;
    };
    static constexpr std::array entries{
        Entry{0x04007EF01F8AE000, &CompatPlayable},   // Fire Emblem: Fortune's Weave
        Entry{0x0400380020056000, &CompatPlayable},   // Splatoon Raiders
        Entry{0x0400C3F00006E000, &CompatInGamePlus}, // Mario Kart World
        Entry{0x04008E001F098000, &CompatPlayable},   // Nintendo Switch Sports Resort
        Entry{0x0400245021D2C000, &CompatPlayable},   // Yoshi and the Mysterious Book
        Entry{0x040077201BE50000, &CompatPlayable},   // Kirby Air Riders
        Entry{0x0400486024DEC000, &CompatPlayable},   // Ocarina of Time
        Entry{0x010015100B514000, &CompatPlayable},   // Super Mario Wonder
    };
    for (const Entry& entry : entries) {
        if (entry.program_id == program_id) {
            return *entry.status;
        }
    }
    return std::nullopt;
}

} // namespace QtCommon::Game
