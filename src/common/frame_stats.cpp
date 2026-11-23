// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/frame_stats.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"

namespace Common::FrameStats {

namespace {

const char* Env() {
    const char* value = std::getenv("EDEN_NX2_FRAME_STATS");
    return (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) ? nullptr
                                                                                              : value;
}

constexpr size_t CauseCount = static_cast<size_t>(Cause::Count);
constexpr u32 WindowPresents = 600;

struct State {
    std::mutex mutex;
    bool started{};
    std::chrono::steady_clock::time_point first{};
    std::chrono::steady_clock::time_point last{};
    std::array<u64, CauseCount> last_stall{};
    u64 presents{};
    std::vector<u32> window; // dt in us
    std::array<u64, CauseCount> slow_ns{};       // cause time inside frames over 33 ms
    std::array<u32, CauseCount> slow_frames{};   // frames over 33 ms where the cause was >= 25% of dt
    std::array<u64, CauseCount> window_ns{};     // cause time over the whole window
    u64 lost_33_us{};                            // sum of (dt - 16.7 ms) over frames above 33 ms
    std::ofstream csv;
};

State& GetState() {
    static State state;
    return state;
}

u32 Percentile(const std::vector<u32>& sorted, double q) {
    if (sorted.empty()) {
        return 0;
    }
    const size_t index = std::min(sorted.size() - 1, static_cast<size_t>(q * (sorted.size() - 1) + 0.5));
    return sorted[index];
}

std::string Summarize(const State& state) {
    std::vector<u32> sorted = state.window;
    std::sort(sorted.begin(), sorted.end());
    u64 sum = 0;
    u32 over_17 = 0, over_33 = 0, over_50 = 0, over_100 = 0;
    for (const u32 dt : sorted) {
        sum += dt;
        over_17 += dt > 16'700;
        over_33 += dt > 33'400;
        over_50 += dt > 50'000;
        over_100 += dt > 100'000;
    }
    std::string causes;
    for (size_t i = 0; i < CauseCount; ++i) {
        if (state.slow_ns[i] >= 1'000'000) {
            causes += fmt::format(" {}={:.0f}ms/{}fr", CauseNames[i], state.slow_ns[i] / 1e6,
                                  state.slow_frames[i]);
        }
    }
    std::string totals;
    for (size_t i = 0; i < CauseCount; ++i) {
        if (state.window_ns[i] >= 1'000'000) {
            totals += fmt::format(" {}={:.0f}", CauseNames[i], state.window_ns[i] / 1e6);
        }
    }
    return fmt::format(
        "[framestats] presents={} window={}: mean {} us, p50 {}, p95 {}, p99 {}, max {} us; "
        ">16.7ms {}, >33ms {}, >50ms {}, >100ms {}; lost over 16.7ms in >33ms frames {:.0f} ms; "
        "cause ms (total in window):{}; cause ms inside >33ms frames (ms/frames):{}",
        state.presents, sorted.size(), sorted.empty() ? 0 : sum / sorted.size(),
        Percentile(sorted, 0.50), Percentile(sorted, 0.95), Percentile(sorted, 0.99),
        sorted.empty() ? 0 : sorted.back(), over_17, over_33, over_50, over_100,
        state.lost_33_us / 1000.0, totals, causes);
}

void FlushWindow(State& state) {
    LOG_INFO(Core, "{}", Summarize(state));
    state.window.clear();
    state.slow_ns.fill(0);
    state.slow_frames.fill(0);
    state.window_ns.fill(0);
    state.lost_33_us = 0;
    if (state.csv.is_open()) {
        state.csv.flush();
    }
}

} // namespace

namespace Detail {
std::atomic<u64> g_present_count{};
const bool g_enabled = Env() != nullptr;
std::array<std::atomic<u64>, CauseCount> g_stall_ns{};
} // namespace Detail

void OnPresent(std::chrono::steady_clock::time_point now) {
    Detail::g_present_count.fetch_add(1, std::memory_order_relaxed);
    if (!Detail::g_enabled) {
        return;
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    std::array<u64, CauseCount> stall{};
    for (size_t i = 0; i < CauseCount; ++i) {
        stall[i] = Detail::g_stall_ns[i].load(std::memory_order_relaxed);
    }
    ++state.presents;
    if (!state.started) {
        state.started = true;
        state.first = now;
        state.last = now;
        state.last_stall = stall;
        const char* env = Env();
        if (env != nullptr && std::strcmp(env, "1") != 0) {
            state.csv.open(env, std::ios::trunc);
            if (state.csv) {
                state.csv << "present,t_ms,dt_us";
                for (const char* name : CauseNames) {
                    state.csv << ',' << name;
                }
                state.csv << '\n';
            } else {
                LOG_ERROR(Core, "[framestats] cannot open {}", env);
            }
        }
        LOG_INFO(Core, "[framestats] enabled{}", state.csv.is_open() ? fmt::format(", csv {}", env) : "");
        return;
    }
    const u32 dt_us = static_cast<u32>(
        std::chrono::duration_cast<std::chrono::microseconds>(now - state.last).count());
    std::array<u64, CauseCount> delta{};
    for (size_t i = 0; i < CauseCount; ++i) {
        delta[i] = stall[i] - state.last_stall[i];
        state.window_ns[i] += delta[i];
    }
    state.last = now;
    state.last_stall = stall;
    state.window.push_back(dt_us);
    if (dt_us > 33'400) {
        state.lost_33_us += dt_us - 16'700;
        for (size_t i = 0; i < CauseCount; ++i) {
            state.slow_ns[i] += delta[i];
            state.slow_frames[i] += delta[i] * 4 >= static_cast<u64>(dt_us) * 1000;
        }
    }
    if (state.csv.is_open()) {
        state.csv << state.presents << ','
                  << std::chrono::duration_cast<std::chrono::milliseconds>(now - state.first).count()
                  << ',' << dt_us;
        for (size_t i = 0; i < CauseCount; ++i) {
            state.csv << ',' << delta[i] / 1000;
        }
        state.csv << '\n';
    }
    if (state.window.size() >= WindowPresents) {
        FlushWindow(state);
    }
}

u64 PresentCount() {
    return Detail::g_present_count.load(std::memory_order_relaxed);
}



void DumpNow(const std::filesystem::path& directory) {
    if (!Detail::g_enabled) {
        return;
    }
    State& state = GetState();
    std::string text;
    {
        std::scoped_lock lock{state.mutex};
        text = Summarize(state);
        text += fmt::format("\npresents since start: {}, since start {:.1f} s\n", state.presents,
                            std::chrono::duration<double>(state.last - state.first).count());
        text += "stall time since start (ms):";
        for (size_t i = 0; i < CauseCount; ++i) {
            const u64 ns = Detail::g_stall_ns[i].load(std::memory_order_relaxed);
            text += fmt::format(" {}={:.0f}", CauseNames[i], ns / 1e6);
        }
        text += "\ncurrent window frame times (us), histogram bucket upper bound: count\n";
        constexpr std::array<u32, 8> bounds{8'000, 12'000, 16'700, 20'000, 25'000, 33'400, 50'000, 100'000};
        std::array<u32, bounds.size() + 1> counts{};
        for (const u32 dt : state.window) {
            size_t bucket = 0;
            while (bucket < bounds.size() && dt > bounds[bucket]) {
                ++bucket;
            }
            ++counts[bucket];
        }
        for (size_t i = 0; i < counts.size(); ++i) {
            text += i < bounds.size() ? fmt::format("  <= {}: {}\n", bounds[i], counts[i])
                                      : fmt::format("  > {}: {}\n", bounds.back(), counts[i]);
        }
    }
    LOG_INFO(Core, "[diag] stats:\n{}", text);
    const std::filesystem::path path = directory / "frame-stats.txt";
    std::ofstream out{path, std::ios::trunc};
    out << text;
    out.close();
    if (!out) {
        LOG_ERROR(Core, "[diag] cannot write stats to {}", path.string());
        return;
    }
    LOG_INFO(Core, "[diag] stats written to {}", path.string());
}

} // namespace Common::FrameStats
