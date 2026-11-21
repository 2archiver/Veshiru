// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <utility>

namespace GameListRefresh {

/// The two kinds of refresh a debounced filesystem-watch window can accumulate. They are
/// tracked independently: a game-directory event and an external-content event arriving in the
/// same window must both run, not whichever set a shared flag last.
struct Coalescer {
    bool external_pending{};
    bool game_dir_pending{};

    void OnExternal() {
        external_pending = true;
    }

    void OnGameDir() {
        game_dir_pending = true;
    }

    struct Requested {
        bool external{};
        bool game_dir{};
    };

    /// Which refreshes were requested inside the window; clears the request.
    Requested Take() {
        const Requested requested{std::exchange(external_pending, false),
                                  std::exchange(game_dir_pending, false)};
        return requested;
    }

    bool Any() const {
        return external_pending || game_dir_pending;
    }
};

} // namespace GameListRefresh
