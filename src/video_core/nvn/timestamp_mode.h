// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <optional>
#include <string_view>

#include "common/common_types.h"
#include "common/settings_enums.h"

// What the game sees in the second field of an NVNcounterData ({ u64 counter; u64 timestamp; })
// and what nvnDeviceGetTimestampInNanoseconds / nvnDeviceGetCurrentTimestampInNanoseconds return.
//
// The field holds what the title's GPU writes: a Tegra X1 writes 614.4 MHz ticks (divided by the
// `fast_gpu_time` divisor, see GpuClockDivisor) and its NVN library converts them as
// ticks * 625 / 384 (nxnative nvn_timestamp.h); the Switch 2 GPU writes nanoseconds and its
// library returns the field unchanged (DK's sdk: chip GA10F, multiplier = divisor = 1).
//
//   EDEN_NX2_NVN_TIMESTAMPS=real (default)
//       What the hardware reports for the GPU work that was executed: the GPU time of a
//       vkCmdWriteTimestamp, mapped onto the guest clock (`GuestClockNanoseconds`, the clock
//       nvnDeviceGetCurrentTimestampInNanoseconds answers and the title's tick counter runs on).
//       The difference of two reports of one frame is the GPU time between them. DK's dynamic
//       resolution (agl::perf GPU-section meter: 1.0.0 sub_793c0, 3.2.0 0x79f0c, read by the DRS
//       controller 1.0.0 sub_72654, 3.2.0 0x87c8c) computes its GPU load from exactly such a pair
//       and lowers the render scale when it exceeds a 60 Hz frame; its frame-rate controller adds
//       (end-of-frame report - frame start) to its frame cost (`setGpuTimeStamp_`).
//   EDEN_NX2_NVN_TIMESTAMPS=legacy
//       The GPU emulation path's clock (A/B diagnosis): a report's timestamp is `GPU::GetTicks()`
//       when the report is written (wall time since boot, divided by the `fast_gpu_time`
//       divisor). Every report of a submission carries the same value, so a game that subtracts
//       two reports of one frame reads ~0 GPU time and never lowers its resolution.
//
// Counter values (the first field), SAMPLES_PASSED and render conditions do not depend on the mode.
namespace VideoCommon::Nvn {

enum class TimestampMode : u8 {
    Legacy,
    Real,
};

inline constexpr TimestampMode DefaultTimestampMode = TimestampMode::Real;

/// `legacy` / `real` (case-insensitive); anything else: nullopt.
[[nodiscard]] std::optional<TimestampMode> ParseTimestampMode(std::string_view text);

/// Mode named by an `EDEN_NX2_NVN_TIMESTAMPS` value (null or empty: DefaultTimestampMode). An
/// unknown value selects the default and sets `valid` to false.
[[nodiscard]] TimestampMode TimestampModeFromValue(const char* value, bool& valid);

/// The mode of this process: `EDEN_NX2_NVN_TIMESTAMPS`, read once (an unknown value is logged and
/// selects the default).
[[nodiscard]] TimestampMode CurrentTimestampMode();

/// Tests: forces the mode (nullopt: back to the environment's).
void SetTimestampModeOverride(std::optional<TimestampMode> mode);

/// NVN counter timestamps tick at 614.4 MHz: 384 ticks per 625 ns.
[[nodiscard]] constexpr u64 GpuTicksToNanoseconds(u64 ticks) {
    return ticks / 384 * 625 + ticks % 384 * 625 / 384;
}
[[nodiscard]] constexpr u64 NanosecondsToGpuTicks(u64 nanoseconds) {
    return nanoseconds / 625 * 384 + nanoseconds % 625 * 384 / 625;
}

/// The divisor of the guest GPU clock in a report timestamp (`fast_gpu_time`: Boost 256, Overclock
/// 512, Normal 1; the setting applies to Switch 1 titles only). A Switch 2 title compares the
/// report time with its own tick counter: DK's frame-rate controller folds the
/// end-of-frame report into its frame cost (`setGpuTimeStamp_`: cost = max(CPU ticks, report - frame start)). A divided clock puts the
/// report far behind the frame start, so the GPU never counts, DK keeps its 60 fps target on a
/// host that delivers fewer frames and its simulation (a fixed step per update) runs in slow
/// motion. The same holds for any divisor, so a Switch 2 title always reads the undivided clock,
/// whatever `fast_gpu_time` says (as on hardware, where reports share the guest clock's epoch).
[[nodiscard]] constexpr u64 GpuClockDivisor(Settings::GpuClock clock, bool switch2_title) {
    if (switch2_title) {
        return 1;
    }
    switch (clock) {
    case Settings::GpuClock::Boost:
        return 256;
    case Settings::GpuClock::Overclock:
        return 512;
    default:
        return 1;
    }
}

/// GpuClockDivisor of the current `fast_gpu_time` setting and program.
[[nodiscard]] u64 CurrentGpuClockDivisor();

/// What `nvnDeviceGetTimestampInNanoseconds` returns for a report whose timestamp field is `field`:
/// the Tegra X1 library converts ticks, the Switch 2 library returns the field.
[[nodiscard]] constexpr u64 ReportFieldToNanoseconds(bool switch2_title, u64 field) {
    return switch2_title ? field : GpuTicksToNanoseconds(field);
}

/// The timestamp field the GPU emulation path writes for a report now (`GPU::GetReportTimestamp`):
/// the guest clock in nanoseconds for a Switch 2 title, whose library returns the field unchanged
/// (OoT's and DK's sdk nvnDeviceGetTimestampInNanoseconds: field * mul / div, 1 / 1 on GA10F), so
/// a report reads on the clock of the title's tick counter; the divided 614.4 MHz ticks for a
/// Switch 1 title, whose library converts them. `guest_ns` and `divided_gpu_ticks` read the clocks;
/// only the one the title uses is read. The Switch 2 branch reads the guest clock directly, as
/// zortos293's 2940d9fab0 ("use native nanosecond timestamps for Ampere") does.
template <typename GuestNs, typename DividedTicks>
[[nodiscard]] u64 EmulatedReportField(bool switch2_title, GuestNs&& guest_ns,
                                      DividedTicks&& divided_gpu_ticks) {
    return switch2_title ? static_cast<u64>(guest_ns()) : static_cast<u64>(divided_gpu_ticks());
}

/// The timestamp field a title's GPU writes for the guest-clock time `guest_ns`: the nanoseconds
/// themselves for a Switch 2 title, divided 614.4 MHz ticks for a Switch 1 title.
[[nodiscard]] u64 GuestNanosecondsToReportField(u64 guest_ns);

/// Source of the guest clock in nanoseconds (the emulated system's `CoreTiming::GetGlobalTimeNs`);
/// installed by the bridge when the recorder is created, cleared when it goes away. Without one
/// (tools, tests) the steady clock stands in.
void SetGuestClockSource(std::function<u64()> source);

/// The guest clock: what nvnDeviceGetCurrentTimestampInNanoseconds returns and what a `real`
/// report is mapped onto.
[[nodiscard]] u64 GuestClockNanoseconds();

/// Source of the legacy report timestamp (`GPU::GetTicks()`); installed by the bridge when the
/// recorder is created. Without one (tools, tests) the steady clock stands in, scaled like the
/// guest GPU clock.
void SetLegacyTickSource(std::function<u64()> source);

/// The guest GPU clock in Legacy mode (614.4 MHz ticks, divided as GpuClockDivisor says).
[[nodiscard]] u64 LegacyReportTicks();

/// The timestamp field of a report written now in Legacy mode: LegacyReportTicks for a Switch 1
/// title, those ticks as nanoseconds for a Switch 2 title (see the top of the file).
[[nodiscard]] u64 LegacyReportField();

/// How report times on the GPU's clock are laid onto the guest clock (`real` mode).
///
/// A report is read some time after the GPU wrote it, so `guest_now - gpu_time` is the clock offset
/// plus that delay; the smallest value seen is the offset with the least delay. The GPU clock and
/// the guest clock are different oscillators and drift against each other (measured on an RTX 5090
/// against the host steady clock: 20 us per second, 12 ms after ten minutes). An all-time minimum
/// follows a GPU clock that runs fast (the minimum falls) but never one that runs slow: reports
/// then land further and further behind the guest clock, DK's frame-rate controller sees
/// (end-of-frame report - frame start) shrink by the drift (`setGpuTimeStamp_`) and, after a few
/// minutes, believes it fits 60 fps on a host that delivers 40. The minimum is therefore taken over
/// the last one to two windows only, so it follows the drift in both directions.
class GpuClockMap {
public:
    /// Length of one minimum bucket: the offset is the minimum of the current and the previous
    /// bucket (4 to 8 seconds, at most 0.2 ms of drift at 20 ppm).
    static constexpr s64 WindowNs = 4'000'000'000;

    enum class Policy : u8 {
        Window, ///< the minimum over the last one to two windows (default)
        Global, ///< the minimum over the whole run (the earlier behaviour, A/B only)
    };

    explicit GpuClockMap(Policy policy = Policy::Window) : policy{policy} {}

    /// `gpu_ns`: report time on the GPU clock; `now_ns`: guest clock when the report is read.
    /// Returns the report time on the guest clock; never later than `now_ns`.
    [[nodiscard]] s64 Map(s64 gpu_ns, s64 now_ns);

    /// The offset (guest minus GPU clock) the last Map used.
    [[nodiscard]] s64 Offset() const {
        return offset;
    }

    /// The mapping policy of this process: EDEN_NX2_NVN_TS_OFFSET=global selects Policy::Global.
    [[nodiscard]] static Policy PolicyFromEnvironment();

private:
    Policy policy;
    bool started{false};
    s64 bucket_start{};
    s64 current_min{};
    s64 previous_min{};
    bool has_previous{false};
    s64 offset{};
};

/// EDEN_NX2_NVN_TSTRACE=1: logs the counter reports the game issues, the timestamps it reads and
/// the render target sizes per present (diagnosis of dynamic resolution; costs nothing when unset).
[[nodiscard]] bool TimestampTraceEnabled();

/// EDEN_NX2_NVN_TSTRACE: the value the guest reads at a SAMPLES_PASSED report (both paths call it,
/// the HLE replay at completion and the GPU emulation path at the guest write). Logs the first 3000
/// reports and every 64th after them, and every 10 s a summary of the reports since the last one
/// (count, report addresses, values by size), comparable between the paths in a steady scene.
void TraceSamplesReport(u64 address, u64 value);

} // namespace VideoCommon::Nvn
