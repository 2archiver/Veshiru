// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/settings.h"
#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/timestamp_mode.h"

// The counter timestamp mode (EDEN_NX2_NVN_TIMESTAMPS): parsing, tick conversion, the legacy tick
// source and what nvnDeviceGetTimestampInNanoseconds / nvnDeviceGetCurrentTimestampInNanoseconds
// answer in each mode. The report writer is covered by the Vulkan test in indirect.cpp.
namespace {

using namespace VideoCommon::Nvn;
using NvnHleTest::Rig;

/// Puts the mode back after a test.
struct ModeGuard {
    ~ModeGuard() {
        SetTimestampModeOverride(std::nullopt);
        SetLegacyTickSource({});
        SetGuestClockSource({});
        Settings::SetCurrentProgramIsSwitch2(false);
    }
};

} // namespace

TEST_CASE("NVN timestamps: the mode names", "[nvn][timestamps]") {
    CHECK(ParseTimestampMode("legacy") == TimestampMode::Legacy);
    CHECK(ParseTimestampMode("real") == TimestampMode::Real);
    CHECK(ParseTimestampMode("REAL") == TimestampMode::Real);
    CHECK(ParseTimestampMode("Legacy") == TimestampMode::Legacy);
    CHECK_FALSE(ParseTimestampMode("").has_value());
    CHECK_FALSE(ParseTimestampMode("1").has_value());
    CHECK_FALSE(ParseTimestampMode("gpu").has_value());

    bool valid = false;
    // The default is the faithful one: real GPU time on the guest clock.
    static_assert(DefaultTimestampMode == TimestampMode::Real);
    CHECK(TimestampModeFromValue(nullptr, valid) == TimestampMode::Real);
    CHECK(valid);
    CHECK(TimestampModeFromValue("", valid) == TimestampMode::Real);
    CHECK(valid);
    CHECK(TimestampModeFromValue("legacy", valid) == TimestampMode::Legacy);
    CHECK(valid);
    // An unknown value selects the default and says so.
    CHECK(TimestampModeFromValue("fast", valid) == TimestampMode::Real);
    CHECK_FALSE(valid);
}

TEST_CASE("NVN timestamps: 614.4 MHz ticks convert to nanoseconds as 625/384", "[nvn][timestamps]") {
    static_assert(GpuTicksToNanoseconds(614'400'000) == 1'000'000'000);
    static_assert(GpuTicksToNanoseconds(384) == 625);
    static_assert(NanosecondsToGpuTicks(1'000'000'000) == 614'400'000);
    static_assert(NanosecondsToGpuTicks(625) == 384);
    CHECK(GpuTicksToNanoseconds(0) == 0);
    CHECK(GpuTicksToNanoseconds(1) == 1); // 625 / 384 = 1.63, truncated
    // Wide values (days of uptime) do not overflow: 2^62 ticks is below 2^64 ns.
    const u64 big = u64{1} << 62;
    CHECK(GpuTicksToNanoseconds(big) == big / 384 * 625 + big % 384 * 625 / 384);
    CHECK(GpuTicksToNanoseconds(big) > big);
    // Round trip within one tick.
    for (const u64 ns : {u64{1'000}, u64{123'456'789'012}, u64{86'400'000'000'000}}) {
        const u64 back = GpuTicksToNanoseconds(NanosecondsToGpuTicks(ns));
        CHECK(back <= ns);
        CHECK(ns - back <= 2);
    }
}

TEST_CASE("NVN timestamps: what the report field means", "[nvn][timestamps]") {
    // Switch 1 title: the field is GPU ticks, which the Tegra X1 library converts. With the
    // default `fast_gpu_time` (Boost, /256) one wall-clock second is 614.4 MHz / 256 = 2.4 M
    // ticks, which the library reads as 3.9 ms.
    const u64 one_second_boosted = 614'400'000 / 256;
    CHECK(ReportFieldToNanoseconds(false, one_second_boosted) == 3'906'250);
    // Switch 2 title: the field is nanoseconds, returned unchanged (chip multiplier and divisor 1).
    CHECK(ReportFieldToNanoseconds(true, 1'234'567'890) == 1'234'567'890);
}

TEST_CASE("NVN timestamps: the GPU emulation path writes a Switch 2 report on the guest clock",
          "[nvn][timestamps]") {
    // GPU::GetReportTimestamp, used by every report the GPU emulation path writes (query cache,
    // Maxwell3D long query, null/OpenGL rasterizers). The Switch 2 library returns the field
    // unchanged: the field must be the guest clock in nanoseconds (614.4 MHz ticks read as
    // nanoseconds put the end-of-frame report 0.39 x uptime in the past and every GPU time at
    // x0.61). A Switch 1 title keeps the divided ticks its library converts. Only the clock the
    // title uses is read.
    constexpr u64 GuestNs = 293'036'000'000'000;
    int guest_reads = 0;
    int tick_reads = 0;
    const auto guest = [&] {
        ++guest_reads;
        return GuestNs;
    };
    const auto ticks = [&] {
        ++tick_reads;
        return NanosecondsToGpuTicks(GuestNs) / 256;
    };
    const u64 switch2_field = EmulatedReportField(true, guest, ticks);
    CHECK(switch2_field == GuestNs);
    CHECK(ReportFieldToNanoseconds(true, switch2_field) == GuestNs);
    CHECK((guest_reads == 1 && tick_reads == 0));
    const u64 switch1_field = EmulatedReportField(false, guest, ticks);
    CHECK(switch1_field == NanosecondsToGpuTicks(GuestNs) / 256);
    CHECK((guest_reads == 1 && tick_reads == 1));
}

TEST_CASE("NVN timestamps: a real report field is the guest clock in the title's unit",
          "[nvn][timestamps]") {
    struct SettingsGuard {
        Settings::GpuClock clock = Settings::values.gpu_clock.GetValue();
        ~SettingsGuard() {
            Settings::values.gpu_clock.SetValue(clock);
        }
    } settings_guard;
    ModeGuard guard;
    std::atomic<u64> now{5'000'000'000'000};
    SetGuestClockSource([&] { return now.load(); });
    CHECK(GuestClockNanoseconds() == 5'000'000'000'000);
    // Switch 2: nanoseconds, whatever `fast_gpu_time` says.
    Settings::SetCurrentProgramIsSwitch2(true);
    Settings::values.gpu_clock.SetValue(Settings::GpuClock::Boost);
    CHECK(GuestNanosecondsToReportField(now) == 5'000'000'000'000);
    // Switch 1: 614.4 MHz ticks divided like the guest GPU clock; the library turns them back.
    Settings::SetCurrentProgramIsSwitch2(false);
    Settings::values.gpu_clock.SetValue(Settings::GpuClock::Normal);
    CHECK(GuestNanosecondsToReportField(1'000'000'000) == 614'400'000);
    Settings::values.gpu_clock.SetValue(Settings::GpuClock::Boost);
    CHECK(GuestNanosecondsToReportField(1'000'000'000) == 614'400'000 / 256);
    // Two reports a GPU time apart stay that far apart, as a game computing its GPU load
    // (DK's dynamic resolution) reads them.
    Settings::SetCurrentProgramIsSwitch2(true);
    const u64 first = GuestNanosecondsToReportField(now);
    const u64 second = GuestNanosecondsToReportField(now + 12'500'000);
    CHECK(ReportFieldToNanoseconds(true, second) - ReportFieldToNanoseconds(true, first) ==
          12'500'000);
}

TEST_CASE("NVN timestamps: a Switch 2 program's report and the current timestamp share an epoch",
          "[nvn][timestamps]") {
    // DK's frame-rate controller subtracts its frame-start tick from the report time, so a report
    // written now must read as now whatever `fast_gpu_time` is set to; a Switch 1 title keeps the
    // setting's divided clock.
    struct SettingsGuard {
        Settings::GpuClock clock = Settings::values.gpu_clock.GetValue();
        ~SettingsGuard() {
            Settings::values.gpu_clock.SetValue(clock);
            Settings::SetCurrentProgramIsSwitch2(false);
        }
    } settings_guard;
    ModeGuard guard;
    Rig rig;
    SetTimestampModeOverride(TimestampMode::Legacy);
    SetLegacyTickSource({}); // the rig has no emulated system: the steady clock stands in
    const u64 report = Rig::SCRATCH + 0x40;
    const auto report_vs_now_ms = [&] {
        rig.memory.Put<u64>(report, 0);
        rig.memory.Put<u64>(report + 8, LegacyReportField());
        const s64 reported = static_cast<s64>(
            rig.Call("nvnDeviceGetTimestampInNanoseconds", {Rig::I(Rig::Obj(1)), Rig::I(report)}));
        const s64 now = static_cast<s64>(
            rig.Call("nvnDeviceGetCurrentTimestampInNanoseconds", {Rig::I(Rig::Obj(1))}));
        return (now - reported) / 1'000'000;
    };
    for (const Settings::GpuClock clock :
         {Settings::GpuClock::Normal, Settings::GpuClock::Boost, Settings::GpuClock::Overclock}) {
        Settings::values.gpu_clock.SetValue(clock);
        Settings::SetCurrentProgramIsSwitch2(true);
        const s64 behind = report_vs_now_ms();
        CHECK(behind >= 0);
        CHECK(behind < 200);
        Settings::SetCurrentProgramIsSwitch2(false);
        if (clock != Settings::GpuClock::Normal) {
            CHECK(report_vs_now_ms() > 1000); // divided: far behind the clock
        }
    }
}

TEST_CASE("NVN timestamps: the legacy tick source", "[nvn][timestamps]") {
    ModeGuard guard;
    struct ProgramGuard {
        ~ProgramGuard() {
            Settings::SetCurrentProgramIsSwitch2(false);
        }
    } program_guard;
    std::atomic<u64> ticks{1000};
    SetLegacyTickSource([&] { return ticks.fetch_add(384); });
    CHECK(LegacyReportTicks() == 1000);
    CHECK(LegacyReportTicks() == 1384);
    // The report field: the ticks themselves for a Switch 1 title, as nanoseconds for Switch 2.
    Settings::SetCurrentProgramIsSwitch2(false);
    CHECK(LegacyReportField() == 1768);
    Settings::SetCurrentProgramIsSwitch2(true);
    CHECK(LegacyReportField() == GpuTicksToNanoseconds(2152));
    Settings::SetCurrentProgramIsSwitch2(false);
    SetLegacyTickSource({});
    // Without one the steady clock stands in, as 614.4 MHz / 256 ticks, increasing.
    const u64 first = LegacyReportTicks();
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    const u64 second = LegacyReportTicks();
    CHECK(second > first);
    // 20 ms is 48 ticks at 2.4 MHz.
    CHECK(second - first < 1'000'000);
}

TEST_CASE("NVN timestamps: nvnDeviceGetTimestampInNanoseconds follows the title's unit", "[nvn][timestamps]") {
    ModeGuard guard;
    Rig rig;
    const u64 report = Rig::SCRATCH + 0x40;
    rig.memory.Put<u64>(report, 0);       // counter
    rig.memory.Put<u64>(report + 8, 2'400'000); // timestamp field

    SetTimestampModeOverride(TimestampMode::Legacy);
    CHECK(rig.Call("nvnDeviceGetTimestampInNanoseconds", {Rig::I(Rig::Obj(1)), Rig::I(report)}) ==
          GpuTicksToNanoseconds(2'400'000));
    CHECK(GpuTicksToNanoseconds(2'400'000) == 3'906'250);
    // A Switch 2 title's library returns the field unchanged.
    Settings::SetCurrentProgramIsSwitch2(true);
    CHECK(rig.Call("nvnDeviceGetTimestampInNanoseconds", {Rig::I(Rig::Obj(1)), Rig::I(report)}) ==
          2'400'000);
    Settings::SetCurrentProgramIsSwitch2(false);

    // The unit follows the title, not the mode: a real-mode field of a Switch 2 title is nanoseconds.
    SetTimestampModeOverride(TimestampMode::Real);
    CHECK(rig.Call("nvnDeviceGetTimestampInNanoseconds", {Rig::I(Rig::Obj(1)), Rig::I(report)}) ==
          GpuTicksToNanoseconds(2'400'000));
    Settings::SetCurrentProgramIsSwitch2(true);
    CHECK(rig.Call("nvnDeviceGetTimestampInNanoseconds", {Rig::I(Rig::Obj(1)), Rig::I(report)}) ==
          2'400'000);
    Settings::SetCurrentProgramIsSwitch2(false);
}

TEST_CASE("NVN timestamps: nvnDeviceGetCurrentTimestampInNanoseconds is the guest clock", "[nvn][timestamps]") {
    ModeGuard guard;
    Rig rig;
    // An installed guest clock is the answer in both modes.
    SetGuestClockSource([] { return u64{777'000'000'000}; });
    for (const TimestampMode mode : {TimestampMode::Legacy, TimestampMode::Real}) {
        SetTimestampModeOverride(mode);
        CHECK(rig.Call("nvnDeviceGetCurrentTimestampInNanoseconds", {Rig::I(Rig::Obj(1))}) ==
              777'000'000'000);
    }
    SetGuestClockSource({});
    // Without one (the rig has no emulated system) both modes answer with the host steady clock.
    for (const TimestampMode mode : {TimestampMode::Legacy, TimestampMode::Real}) {
        SetTimestampModeOverride(mode);
        const u64 a = rig.Call("nvnDeviceGetCurrentTimestampInNanoseconds", {Rig::I(Rig::Obj(1))});
        const u64 b = rig.Call("nvnDeviceGetCurrentTimestampInNanoseconds", {Rig::I(Rig::Obj(1))});
        CHECK(a != 0);
        CHECK(b >= a);
    }
}

namespace {

/// A GPU clock `ppm` parts per million slower (negative: faster) than the guest clock, `delay_ns`
/// behind: reads a report written at guest time `t` and returns how far from `t` the mapped
/// report time lands (negative: earlier than the true time).
s64 MappedError(GpuClockMap& map, s64 t_ns, double ppm, s64 delay_ns) {
    const s64 gpu_ns = static_cast<s64>(static_cast<double>(t_ns) * (1.0 - ppm * 1e-6));
    return map.Map(gpu_ns, t_ns + delay_ns) - t_ns;
}

} // namespace

TEST_CASE("NVN timestamps: GpuClockMap follows a GPU clock that runs slow", "[nvn][timestamps]") {
    // 20 ppm is the drift measured against the host steady clock: 12 ms after ten minutes. The
    // title's frame-rate controller subtracts the frame start from the end-of-frame report, so a
    // report that lands that far behind the guest clock hides that much of the frame cost.
    GpuClockMap window{GpuClockMap::Policy::Window};
    GpuClockMap global{GpuClockMap::Policy::Global};
    constexpr s64 Delay = 300'000; // the report is read 0.3 ms after the GPU wrote it
    s64 window_error = 0;
    s64 global_error = 0;
    for (s64 t = 1'000'000'000; t <= 1'200'000'000'000; t += 16'000'000) {
        window_error = MappedError(window, t, 20.0, Delay);
        global_error = MappedError(global, t, 20.0, Delay);
    }
    // After 1200 s: the window mapping lands where the report was written plus the delay it cannot
    // see (0.3 ms), less the drift of one to two windows (at most 0.16 ms at 20 ppm); the all-time
    // minimum is 24 ms behind the true time.
    CHECK(window_error > 0);
    CHECK(window_error <= Delay);
    CHECK(global_error < -20'000'000);
}

TEST_CASE("NVN timestamps: GpuClockMap follows a GPU clock that runs fast", "[nvn][timestamps]") {
    GpuClockMap window{GpuClockMap::Policy::Window};
    GpuClockMap global{GpuClockMap::Policy::Global};
    s64 window_error = 0;
    s64 global_error = 0;
    for (s64 t = 1'000'000'000; t <= 600'000'000'000; t += 16'000'000) {
        window_error = MappedError(window, t, -20.0, 300'000);
        global_error = MappedError(global, t, -20.0, 300'000);
    }
    CHECK(window_error > -100'000);
    CHECK(window_error <= 300'000);
    CHECK(global_error > -100'000);
    CHECK(global_error <= 300'000);
}

TEST_CASE("NVN timestamps: GpuClockMap never maps a report into the future", "[nvn][timestamps]") {
    GpuClockMap map;
    s64 t = 5'000'000'000;
    for (int i = 0; i < 5000; ++i) {
        t += 16'000'000;
        // A delay between 0 and 40 ms, as a loaded completion thread produces.
        const s64 delay = static_cast<s64>((i * 7919) % 40) * 1'000'000;
        const s64 mapped = map.Map(t, t + delay);
        CHECK(mapped <= t + delay);
    }
}

TEST_CASE("NVN timestamps: GpuClockMap keeps the least delay of the window, not of one report",
          "[nvn][timestamps]") {
    GpuClockMap map;
    constexpr s64 Epoch = 10'000'000'000;
    // Many reports read 0.2 ms late, then a burst read 30 ms late: the burst maps to its real GPU
    // time (30 ms behind the read), not 30 ms later.
    s64 t = Epoch;
    for (int i = 0; i < 100; ++i, t += 16'000'000) {
        static_cast<void>(map.Map(t, t + 200'000));
    }
    const s64 mapped = map.Map(t, t + 30'000'000);
    CHECK(mapped - t == 200'000); // the offset is the 0.2 ms the earlier reports saw
    CHECK(t + 30'000'000 - mapped >= 29'000'000);
}

TEST_CASE("NVN timestamps: GpuClockMap forgets a window nothing was read in", "[nvn][timestamps]") {
    GpuClockMap map;
    static_cast<void>(map.Map(1'000'000'000, 1'000'200'000));
    // The GPU clock was restarted (device reset): 100 s later the offset is a different one.
    const s64 gpu = 500'000'000;
    const s64 now = 101'000'000'000;
    CHECK(map.Map(gpu, now + 200'000) == now + 200'000);
}

TEST_CASE("NVN timestamps: the mapping policy comes from EDEN_NX2_NVN_TS_OFFSET", "[nvn][timestamps]") {
    // Read once per process; the unset default is the sliding window, "global" the whole run.
    const char* const value = std::getenv("EDEN_NX2_NVN_TS_OFFSET");
    const auto expected = value != nullptr && std::string_view{value} == "global"
                              ? GpuClockMap::Policy::Global
                              : GpuClockMap::Policy::Window;
    CHECK(GpuClockMap::PolicyFromEnvironment() == expected);
}
