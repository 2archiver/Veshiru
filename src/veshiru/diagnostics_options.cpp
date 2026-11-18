// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include "veshiru/diagnostics_options.h"

namespace Veshiru {
namespace {

bool TakesFollowingArgument(std::string_view option) {
    // Do not steal a logging option if it is actually a value for a frontend or Qt option.
    constexpr std::array valued = {"-c",
                                   "--config",
                                   "-d",
                                   "--debug",
                                   "-g",
                                   "--game",
                                   "-m",
                                   "--multiplayer",
                                   "-p",
                                   "-u",
                                   "--user",
                                   "-i",
                                   "--input-profile",
                                   "-N",
                                   "--pack-nup",
                                   "-U",
                                   "--unpack-nup",
                                   "-E",
                                   "--edit-nup",
                                   "--nup-name",
                                   "--nup-icon",
                                   "--nup-version",
                                   "--nup-publisher",
                                   "-platform",
                                   "-platformpluginpath",
                                   "-platformtheme",
                                   "-plugin",
                                   "-style",
                                   "-stylesheet",
                                   "-display",
                                   "-geometry"};
    return std::ranges::find(valued, option) != valued.end();
}

} // namespace

DiagnosticsOptions ParseDiagnosticsOptions(std::span<const std::string_view> args) {
    DiagnosticsOptions options;
    if (args.empty()) {
        return options;
    }
    options.remaining.push_back(0);
    const auto count = static_cast<int>(args.size());
    bool after_separator = false;
    for (int i = 1; i < count; ++i) {
        const auto arg = args[i];
        if (after_separator || arg == "--") {
            options.remaining.push_back(i);
            after_separator = true;
            continue;
        }
        const auto split = arg.find('=');
        const auto name = arg.substr(0, split);
        if (arg == "--console" || arg == "--no-console") {
            options.console = arg == "--console" ? ConsoleMode::Enable : ConsoleMode::Disable;
        } else if (name == "--console" || name == "--no-console") {
            options.error = "--console and --no-console do not take a value";
            return options;
        } else if (arg == "-h" || arg == "--help") {
            options.help = true;
        } else if (arg == "-v" || arg == "--version") {
            options.version = true;
        } else if (name == "--log-level" || name == "--log-file") {
            std::string_view value;
            if (split != arg.npos) {
                value = arg.substr(split + 1);
            } else if (i + 1 < count && !args[i + 1].starts_with('-')) {
                value = args[++i];
            }
            if (name == "--log-level") {
                options.level = Common::Log::ParseLevel(value);
                if (!options.level) {
                    options.error =
                        "--log-level requires Trace, Debug, Info, Warning, Error or Critical";
                    return options;
                }
            } else if (value.empty()) {
                options.error = "--log-file requires a writable file path";
                return options;
            } else {
                options.log_path = value;
            }
        } else {
            options.remaining.push_back(i);
            if (TakesFollowingArgument(arg) && i + 1 < count) {
                options.remaining.push_back(++i);
            }
        }
    }
    return options;
}

} // namespace Veshiru
