// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/nvn/timestamp_mode.h"

namespace VideoCommon::Nvn {
namespace {

std::mutex tick_source_mutex;
std::function<u64()> tick_source;
std::mutex guest_clock_mutex;
std::function<u64()> guest_clock;
std::atomic<int> mode_override{-1};

} // namespace

std::optional<TimestampMode> ParseTimestampMode(std::string_view text) {
    std::string lower;
    lower.reserve(text.size());
    for (const char c : text) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "legacy") {
        return TimestampMode::Legacy;
    }
    if (lower == "real") {
        return TimestampMode::Real;
    }
    return std::nullopt;
}

TimestampMode TimestampModeFromValue(const char* value, bool& valid) {
    valid = true;
    if (value == nullptr || *value == '\0') {
        return DefaultTimestampMode;
    }
    if (const auto mode = ParseTimestampMode(value)) {
        return *mode;
    }
    valid = false;
    return DefaultTimestampMode;
}

TimestampMode CurrentTimestampMode() {
    if (const int forced = mode_override.load(std::memory_order_relaxed); forced >= 0) {
        return static_cast<TimestampMode>(forced);
    }
    static const TimestampMode mode = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_TIMESTAMPS");
        bool valid = true;
        const TimestampMode result = TimestampModeFromValue(value, valid);
        if (!valid) {
            LOG_WARNING(Core, "[nvn] EDEN_NX2_NVN_TIMESTAMPS={} is not legacy or real; using real",
                        value);
        }
        LOG_INFO(Core, "[nvn] counter timestamps: {}",
                 result == TimestampMode::Legacy
                     ? "legacy (guest GPU clock / fast_gpu_time, as the GPU emulation path)"
                     : "real (GPU time of each report, on the guest clock)");
        return result;
    }();
    return mode;
}

void SetTimestampModeOverride(std::optional<TimestampMode> mode) {
    mode_override.store(mode ? static_cast<int>(*mode) : -1, std::memory_order_relaxed);
}

s64 GpuClockMap::Map(s64 gpu_ns, s64 now_ns) {
    const s64 candidate = now_ns - gpu_ns;
    if (policy == Policy::Global) {
        if (!started || candidate < offset) {
            started = true;
            offset = candidate;
        }
        return gpu_ns + offset;
    }
    if (!started) {
        started = true;
        bucket_start = now_ns;
        current_min = candidate;
    } else if (now_ns - bucket_start >= 2 * WindowNs) {
        // Nothing was read for two windows: both buckets are stale.
        has_previous = false;
        bucket_start = now_ns;
        current_min = candidate;
    } else if (now_ns - bucket_start >= WindowNs) {
        previous_min = current_min;
        has_previous = true;
        bucket_start = now_ns;
        current_min = candidate;
    } else {
        current_min = std::min(current_min, candidate);
    }
    offset = has_previous ? std::min(current_min, previous_min) : current_min;
    return gpu_ns + offset;
}

GpuClockMap::Policy GpuClockMap::PolicyFromEnvironment() {
    static const Policy policy = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_TS_OFFSET");
        return value != nullptr && std::string_view{value} == "global" ? Policy::Global
                                                                         : Policy::Window;
    }();
    return policy;
}

void SetGuestClockSource(std::function<u64()> source) {
    std::scoped_lock lock{guest_clock_mutex};
    guest_clock = std::move(source);
}

u64 GuestClockNanoseconds() {
    {
        std::scoped_lock lock{guest_clock_mutex};
        if (guest_clock) {
            return guest_clock();
        }
    }
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

u64 GuestNanosecondsToReportField(u64 guest_ns) {
    if (Settings::CurrentProgramIsSwitch2()) {
        return guest_ns;
    }
    return NanosecondsToGpuTicks(guest_ns) / CurrentGpuClockDivisor();
}

void SetLegacyTickSource(std::function<u64()> source) {
    std::scoped_lock lock{tick_source_mutex};
    tick_source = std::move(source);
}

u64 CurrentGpuClockDivisor() {
    return GpuClockDivisor(Settings::values.gpu_clock.GetValue(), Settings::CurrentProgramIsSwitch2());
}

u64 LegacyReportTicks() {
    {
        std::scoped_lock lock{tick_source_mutex};
        if (tick_source) {
            return tick_source();
        }
    }
    // No emulated system: the steady clock as 614.4 MHz ticks, divided like the guest GPU clock.
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return NanosecondsToGpuTicks(static_cast<u64>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(now).count())) /
           CurrentGpuClockDivisor();
}

u64 LegacyReportField() {
    const u64 ticks = LegacyReportTicks();
    return Settings::CurrentProgramIsSwitch2() ? GpuTicksToNanoseconds(ticks) : ticks;
}

bool TimestampTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_TSTRACE");
        return value != nullptr && std::string_view{value} != "0";
    }();
    return enabled;
}

void TraceSamplesReport(u64 address, u64 value) {
    // Value classes of the summary: 0, <64, <1024, <16384, ==16384 (a full 128x128 target), more.
    static constexpr std::array<const char*, 6> NAMES{"0", "1-63", "64-1023", "1024-16383",
                                                      "16384", ">16384"};
    static std::mutex mutex;
    static u64 count = 0;
    static u64 zero = 0;
    static std::array<u64, 6> classes{};
    static u64 window_reports = 0;
    static u64 window_sum = 0;
    static std::unordered_set<u64> window_addresses;
    static std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++count;
    zero += value == 0 ? 1 : 0;
    if (count <= 3000 || count % 64 == 0) {
        LOG_INFO(HW_GPU, "[nvn] tstrace samples address={:#x} value={} (zero so far {}, nonzero {})",
                 address, value, zero, count - zero);
    }
    const size_t klass = value == 0        ? 0
                         : value < 64      ? 1
                         : value < 1024    ? 2
                         : value < 16384   ? 3
                         : value == 16384  ? 4
                                           : 5;
    ++classes[klass];
    ++window_reports;
    window_sum += value;
    window_addresses.insert(address);
    const auto now = std::chrono::steady_clock::now();
    if (now - window_start >= std::chrono::seconds{10}) {
        std::string text;
        for (size_t i = 0; i < NAMES.size(); ++i) {
            text += fmt::format(" {}={}", NAMES[i], classes[i]);
        }
        LOG_INFO(HW_GPU, "[nvn] tstrace samples summary: reports={} addresses={} mean={}{}",
                 window_reports, window_addresses.size(), window_sum / window_reports, text);
        classes = {};
        window_reports = 0;
        window_sum = 0;
        window_addresses.clear();
        window_start = now;
    }
}

} // namespace VideoCommon::Nvn
