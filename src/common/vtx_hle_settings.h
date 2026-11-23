// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string_view>

#include "common/common_types.h"
#include "common/settings_enums.h"

// Which settings VTX-HLE (the NVN high-level graphics path) does not apply, and when a settings
// page has to treat the path as VTX-HLE. The Qt settings pages grey the listed settings out while
// the path is VTX-HLE; nothing here reads or writes a setting value.

namespace Settings::VtxHle {

enum class Reason : u8 {
    /// The setting tunes the emulated Maxwell GPU (its command processor, caches, pipelines or
    /// texture decoding); VTX-HLE runs none of that.
    EmulatedGpuOnly,
    /// VTX-HLE has no implementation of the setting yet.
    NotYetSupported,
};

struct Entry {
    std::string_view key; ///< the setting's config key (BasicSetting::GetLabel)
    Reason reason;
};

/// Every setting with no reader on the VTX-HLE path, in no particular order. A setting that
/// VTX-HLE gains support for is removed here, nowhere else.
[[nodiscard]] std::span<const Entry> Unapplied();

/// The entry of a setting's config key, nullptr when VTX-HLE applies the setting.
[[nodiscard]] const Entry* Find(std::string_view key);

/// What decides the path a settings page is about.
struct PathQuery {
    /// The page's game is running (the global page: any game; Properties: that very game).
    bool game_running{};
    /// The running game's actual path (Core::NvnHle::HleRunning); read only when `game_running`.
    bool running_hle{};
    /// The Graphics path the page shows: the saved choice, or the dropdown's current one.
    GraphicsPath selected{GraphicsPath::Emulated};
    /// The Vulkan API is selected: VTX-HLE is a Vulkan replay.
    bool vulkan{true};
    /// The page's game is a Switch 2 program. The global page has no game: it passes true.
    bool switch2_program{true};
    /// EDEN_NX2_NVN_HLE (nullptr when unset): it overrides the setting like it does at game start.
    const char* environment{};
};

/// True when the page's games run VTX-HLE: the running game's actual path while it runs, else the
/// path the game would start on (Settings::ChooseNvnHle) on the Vulkan renderer.
[[nodiscard]] bool IsActive(const PathQuery& query);

} // namespace Settings::VtxHle
