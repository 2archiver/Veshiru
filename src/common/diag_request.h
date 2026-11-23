// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

/// Runtime diagnostic requests for the Switch 2 paths (legacy GPU emulation and NVN HLE).
///
/// A tool (eden-mcp `request`, ab_runner `requests`, or a human) writes one command per line into
/// `<user dir>/nx2-request.txt`. `Poll` is called only from the NVN present path (the legacy GPU
/// present path does not poll); at most every 250 ms it
/// looks for that file, takes it (rename to `nx2-request.txt.taken`, then delete), logs
/// `[diag] request: <command>` for each line and runs the registered handler. Handlers run on the
/// polling (present) thread and must only arm a bounded one-shot capture or do a small dump; the
/// heavy work happens where the data is (draw path, completion thread).
///
/// All output of one request file goes to `<user dir>/diag/<YYYYMMDD-HHMMSS>[-n]/`, created when
/// the first command needs it and logged as `[diag] output folder: <path>`.
///
/// Commands: `textures [N]`, `frame [N]`, `threads`, `stats`, `nvn-pass <n>`, `nvn-draw <n>`
/// (see docs/NX2_DIAGNOSTICS.md). `EDEN_NX2_REQUESTS=0` turns polling off. Unset cost: one relaxed
/// atomic load and a compare per present.
namespace Common::DiagRequest {

struct Batch;

/// One command line of a consumed request file.
struct Request {
    std::string command;           ///< lower case
    std::vector<std::string> args; ///< whitespace separated words after the command
    Batch* batch{};

    /// The output folder of this request file (created on first use).
    [[nodiscard]] const std::filesystem::path& Directory() const;
    /// Numeric argument `index`, or `fallback` when missing / not a number.
    [[nodiscard]] u32 NumberArg(size_t index, u32 fallback) const;
};

using Handler = std::function<void(const Request&)>;

/// Registers (or replaces) the handler of `command` (lower case). Thread safe.
void Register(std::string command, Handler handler);

/// Registers `command` as a known-but-unavailable command with a reason, logged when requested.
/// A real `Register` replaces it.
void RegisterUnavailable(std::string command, std::string reason);

/// Whether a runtime-overridable EDEN_NX2_<name> TRY switch is enabled. Runtime override wins
/// over the process environment, which wins over a per-game recommendation when supplied; unset
/// switches without a recommendation retain their existing environment default.
[[nodiscard]] bool TrySwitchEnabled(std::string_view name);
[[nodiscard]] bool TrySwitchEnabled(std::string_view name, bool recommended_value);
[[nodiscard]] bool SetTrySwitch(std::string_view name, bool enabled);
/// Generation increments whenever a runtime TRY switch actually changes.
[[nodiscard]] u64 TrySwitchGeneration();
/// Whether a runtime override differs from the configured environment/default value.
[[nodiscard]] bool TrySwitchDiffersFromDefault(std::string_view name);

namespace Detail {
extern std::atomic<s64> g_next_poll_ns;
void PollSlow(s64 now_ns);
} // namespace Detail

/// Call from a cheap, frequently called point (the guest present). Thread safe.
inline void Poll(std::chrono::steady_clock::time_point now) {
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    if (now_ns < Detail::g_next_poll_ns.load(std::memory_order_relaxed)) {
        return;
    }
    Detail::PollSlow(now_ns);
}

/// A bounded one-shot capture budget shared between the request handler (arms it) and the path that
/// captures (claims one unit per draw / present until it is exhausted). `Active` is one relaxed
/// load, the only cost on the hot path while no request is pending.
class Budget {
public:
    void Arm(u32 count, std::filesystem::path directory);
    [[nodiscard]] bool Active() const noexcept {
        return remaining.load(std::memory_order_relaxed) != 0;
    }
    /// Consumes one unit. Returns false when exhausted; otherwise fills `directory` and `ordinal`
    /// (0-based count of units claimed since the last `Arm`).
    bool Claim(std::filesystem::path& directory, u32* ordinal = nullptr);
    /// Units left.
    [[nodiscard]] u32 Remaining() const noexcept {
        return remaining.load(std::memory_order_relaxed);
    }
    /// Increases on every `Arm`: lets a consumer reset per-request state (dedup sets, indexes).
    [[nodiscard]] u32 Epoch() const noexcept {
        return epoch.load(std::memory_order_relaxed);
    }
    /// The directory of the last `Arm`.
    [[nodiscard]] std::filesystem::path Directory() const;

private:
    std::atomic<u32> remaining{0};
    std::atomic<u32> epoch{0};
    u32 claimed{};
    mutable std::mutex mutex;
    std::filesystem::path directory;
};

/// `textures [N]`: textures sampled by the next N draws (legacy and NVN draw paths claim one per draw).
Budget& TextureBudget();
/// `frame [N]`: the next N presented frames are saved (legacy `CaptureFrame` and the NVN capture claim one per present).
Budget& FrameBudget();

/// Replaces the directory the request file is looked up in (tests). Empty restores `EdenDir`.
void SetUserDirectoryForTesting(std::filesystem::path directory);

} // namespace Common::DiagRequest
