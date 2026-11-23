// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

#include "common/settings.h"

namespace Core::NvnHle {

/// Parsed NVN bridge switches. Kept as an explicit struct so tests can inject it.
struct Config {
    /// EDEN_NX2_NVN_CENSUS: count every NVN entry point the game obtains and calls.
    bool census{};
    /// Take over NVN: EDEN_NX2_NVN_HLE (1/0) when set, else the graphics_path setting for a
    /// Switch 2 program (Settings::ResolveNvnHle).
    bool hle{};
    /// EDEN_NX2_NVN_THREADS: the heartbeat also dumps every guest thread and the per-window NVN
    /// call counts (needs the per-slot call counters, which are otherwise only kept for census).
    /// It does not start the bridge by itself: it only produces output when census or the HLE
    /// path is on (AnyEnabled).
    bool thread_dump{};
    /// Output directory for the census / unimplemented / unsupported TSV files. Empty: the
    /// directory `OutputDirectory()` resolves.
    std::string output_dir;
    /// Seconds between periodic TSV rewrites.
    unsigned report_interval_seconds{10};

    /// Per-slot call counters are kept only when something reads them.
    [[nodiscard]] bool CountCalls() const {
        return census || thread_dump;
    }
    [[nodiscard]] bool AnyEnabled() const {
        return census || hle;
    }
};

/// A switch is on when it is set to anything other than "0".
[[nodiscard]] inline bool SwitchOn(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string_view{value} != "0";
}

/// Where the NVN reports go: `EDEN_NX2_NVN_OUTPUT_DIR` when set and non-empty, else `nvn` inside
/// the emulator's log directory. The single definition every NVN output site uses; it does not
/// create the directory (writers create it on first use).
[[nodiscard]] std::filesystem::path OutputDirectory();

[[nodiscard]] inline Config ConfigFromEnvironment() {
    Config config;
    config.census = SwitchOn("EDEN_NX2_NVN_CENSUS");
    config.hle = Settings::ResolveNvnHle().hle;
    config.thread_dump = SwitchOn("EDEN_NX2_NVN_THREADS");
    return config;
}

} // namespace Core::NvnHle
