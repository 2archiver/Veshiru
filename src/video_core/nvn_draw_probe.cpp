// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#include <fmt/format.h>

#include "common/fs/file.h"
#include "video_core/nvn_draw_probe.h"

namespace VideoCommon::NvnDrawProbe {

namespace Detail {
std::atomic<bool> g_active{false};
}

namespace {

using Key = std::tuple<std::array<u64, StageCount>, u32, bool, bool, bool>;

struct State {
    std::mutex mutex;
    // Census: `current` is filled by the replay, `last` is the one of the last present.
    u32 census_presents_left{};
    bool census_partial{}; ///< `current` started mid-frame: dropped at the next present
    std::map<Key, Row> current;
    u64 current_draws{};
    u64 current_skipped{};
    std::map<u64, DispatchRow> current_dispatch_rows;
    u64 current_dispatches{};
    u64 current_skipped_dispatches{};
    Census last;
    u64 presents{};
    // Skip set.
    std::vector<u64> skip;
    SkipMode skip_mode{SkipMode::Skip};
    // Dump.
    DumpState dump;
    std::string dump_directory;
    std::string dump_path; ///< fixed by the first claimed section
    u32 dump_claimed{};
    u64 dump_generation{};
};

State& S() {
    static State state;
    return state;
}

/// The armed dump has sections left to claim.
bool DumpWanted(const State& s) {
    return s.dump.armed && s.dump_claimed < s.dump.count && s.dump.error.empty();
}

void UpdateActiveLocked(State& s) {
    const bool active = s.census_presents_left != 0 || !s.skip.empty() ||
                        s.skip_mode == SkipMode::Only || DumpWanted(s);
    Detail::g_active.store(active, std::memory_order_relaxed);
}

std::optional<DumpTicket> ClaimDumpLocked(State& s, const char* kind) {
    if (s.dump_claimed == 0) {
        s.dump.kind = kind;
        s.dump_path = fmt::format("{}/nvn-{}-dump-{:016x}-p{}.txt", s.dump_directory, kind,
                                  s.dump.hash, s.presents);
    } else if (s.dump.kind != kind) {
        return std::nullopt;
    }
    DumpTicket ticket{s.dump_path, s.dump_claimed++, s.dump.count, s.dump_generation};
    UpdateActiveLocked(s);
    return ticket;
}

bool Matches(const std::vector<u64>& set, const DrawInfo& info) {
    for (const u64 hash : info.hashes) {
        if (hash != 0 && std::find(set.begin(), set.end(), hash) != set.end()) {
            return true;
        }
    }
    return false;
}

void ClearCurrentLocked(State& s) {
    s.current.clear();
    s.current_draws = 0;
    s.current_skipped = 0;
    s.current_dispatch_rows.clear();
    s.current_dispatches = 0;
    s.current_skipped_dispatches = 0;
}

} // namespace

bool OnDraw(const DrawInfo& info) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    bool skip = false;
    if (s.skip_mode == SkipMode::Only) {
        skip = !Matches(s.skip, info);
    } else if (!s.skip.empty()) {
        skip = Matches(s.skip, info);
    }
    if (s.census_presents_left != 0) {
        const Key key{info.hashes, info.mode, info.indexed, info.indirect, info.mesh};
        auto [it, inserted] = s.current.try_emplace(key);
        Row& row = it->second;
        if (inserted) {
            row.first = info;
        }
        ++row.draws;
        row.elements += info.count;
        row.skipped += skip ? 1 : 0;
        ++s.current_draws;
        s.current_skipped += skip ? 1 : 0;
    }
    return skip;
}

bool OnDispatch(const DispatchInfo& info) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    // Only mode selects draws; a dispatch is skipped in either mode when its hash is listed.
    const bool skip =
        info.hash != 0 && std::find(s.skip.begin(), s.skip.end(), info.hash) != s.skip.end();
    if (s.census_presents_left != 0) {
        DispatchRow& row = s.current_dispatch_rows[info.hash];
        row.hash = info.hash;
        ++row.count;
        if (info.indirect) {
            ++row.indirect;
        } else {
            row.groups += u64{info.groups[0]} * info.groups[1] * info.groups[2];
        }
        row.skipped += skip ? 1 : 0;
        ++s.current_dispatches;
        s.current_skipped_dispatches += skip ? 1 : 0;
    }
    return skip;
}

void OnPresent() {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    ++s.presents;
    if (s.census_presents_left == 0) {
        return;
    }
    if (s.census_partial) {
        s.census_partial = false;
        ClearCurrentLocked(s);
        return;
    }
    Census census;
    census.present = s.presents;
    census.draws = s.current_draws;
    census.skipped = s.current_skipped;
    census.rows.reserve(s.current.size());
    for (auto& [key, row] : s.current) {
        census.rows.push_back(std::move(row));
    }
    std::sort(census.rows.begin(), census.rows.end(),
              [](const Row& a, const Row& b) { return a.draws > b.draws; });
    census.dispatches = s.current_dispatches;
    census.skipped_dispatches = s.current_skipped_dispatches;
    census.dispatch_rows.reserve(s.current_dispatch_rows.size());
    for (const auto& [hash, row] : s.current_dispatch_rows) {
        census.dispatch_rows.push_back(row);
    }
    std::stable_sort(census.dispatch_rows.begin(), census.dispatch_rows.end(),
                     [](const DispatchRow& a, const DispatchRow& b) { return a.count > b.count; });
    s.last = std::move(census);
    ClearCurrentLocked(s);
    --s.census_presents_left;
    UpdateActiveLocked(s);
}

std::optional<DumpTicket> DumpRequested(const DrawInfo& info) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    if (!DumpWanted(s) ||
        std::find(info.hashes.begin(), info.hashes.end(), s.dump.hash) == info.hashes.end()) {
        return std::nullopt;
    }
    return ClaimDumpLocked(s, "draw");
}

std::optional<DumpTicket> DumpRequested(const DispatchInfo& info) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    if (!DumpWanted(s) || info.hash != s.dump.hash) {
        return std::nullopt;
    }
    return ClaimDumpLocked(s, "dispatch");
}

void WriteDump(const DumpTicket& ticket, std::string_view text) {
    State& s = S();
    // The file is written under the lock so that a reset or re-arm cannot race a stale ticket
    // into the new dump's file (dumps are rare, one-shot diagnostics).
    std::scoped_lock lock{s.mutex};
    if (ticket.generation != s.dump_generation) {
        return; // re-armed meanwhile
    }
    const size_t written =
        ticket.index == 0
            ? Common::FS::WriteStringToFile(ticket.path, Common::FS::FileType::BinaryFile, text)
            : Common::FS::AppendStringToFile(ticket.path, Common::FS::FileType::BinaryFile, text);
    if (written != text.size()) {
        s.dump.error = "cannot write " + ticket.path;
    } else {
        s.dump.path = ticket.path;
        ++s.dump.written;
    }
    s.dump.done = s.dump.written >= s.dump.count || !s.dump.error.empty();
    UpdateActiveLocked(s);
}

void AppendRows(std::string& text, std::span<const u8> bytes) {
    constexpr size_t Row = 16;
    size_t zero_run = 0;
    const auto flush_zeros = [&](size_t end) {
        if (zero_run == 1) {
            text += fmt::format("  {:06x}: zeros\n", end - Row);
        } else if (zero_run > 1) {
            text += fmt::format("  {:06x}..{:06x}: zeros ({} rows)\n", end - zero_run * Row,
                                end - 1, zero_run);
        }
        zero_run = 0;
    };
    for (size_t offset = 0; offset < bytes.size(); offset += Row) {
        const size_t length = (std::min)(Row, bytes.size() - offset);
        const auto row = bytes.subspan(offset, length);
        if (length == Row && std::all_of(row.begin(), row.end(), [](u8 b) { return b == 0; })) {
            ++zero_run;
            continue;
        }
        flush_zeros(offset);
        text += fmt::format("  {:06x}:", offset);
        for (size_t i = 0; i < Row; ++i) {
            text += i < length ? fmt::format(" {:02x}", row[i]) : std::string(3, ' ');
        }
        text += " | f32";
        for (size_t i = 0; i + 4 <= length; i += 4) {
            f32 value;
            std::memcpy(&value, row.data() + i, sizeof(value));
            text += fmt::format(" {:.7g}", value);
        }
        text += " | u32";
        for (size_t i = 0; i + 4 <= length; i += 4) {
            u32 value;
            std::memcpy(&value, row.data() + i, sizeof(value));
            text += fmt::format(" {}", value);
        }
        text += '\n';
    }
    flush_zeros(bytes.size() - bytes.size() % Row);
}

bool ArmCensus(u32 presents) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    const bool running = s.census_presents_left != 0;
    if (!running) {
        ClearCurrentLocked(s);
        s.census_partial = true;
    }
    s.census_presents_left = std::max(s.census_presents_left, presents);
    UpdateActiveLocked(s);
    return running;
}

Census LastCensus() {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    return s.last;
}

void Reset() {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    s.census_presents_left = 0;
    s.census_partial = false;
    ClearCurrentLocked(s);
    s.last = {};
    s.skip.clear();
    s.skip_mode = SkipMode::Skip;
    s.dump = {};
    s.dump_directory.clear();
    s.dump_path.clear();
    s.dump_claimed = 0;
    ++s.dump_generation;
    UpdateActiveLocked(s);
}

void SetSkip(std::vector<u64> hashes, SkipMode mode) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    s.skip = std::move(hashes);
    s.skip_mode = mode;
    UpdateActiveLocked(s);
}

void GetSkip(std::vector<u64>& hashes, SkipMode& mode) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    hashes = s.skip;
    mode = s.skip_mode;
}

void ArmDump(u64 hash, std::string directory, u32 count) {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    s.dump = DumpState{.armed = true, .hash = hash, .count = std::clamp(count, 1u, MaxDumpCount)};
    s.dump_directory = std::move(directory);
    s.dump_path.clear();
    s.dump_claimed = 0;
    ++s.dump_generation;
    UpdateActiveLocked(s);
}

DumpState GetDump() {
    State& s = S();
    std::scoped_lock lock{s.mutex};
    return s.dump;
}

} // namespace VideoCommon::NvnDrawProbe
