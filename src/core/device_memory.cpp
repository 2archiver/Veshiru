// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <new>

#include "common/logging.h"
#include "common/memory_detect.h"
#include "common/thread.h"
#endif

#include "core/core.h"
#include "core/device_memory.h"
#include "hle/kernel/board/nintendo/nx/k_system_control.h"

namespace Core {

#ifdef HAS_NCE
constexpr size_t VirtualReserveSize = 1ULL << 38;
#else
constexpr size_t VirtualReserveSize = 1ULL << 39;
#endif

DeviceMemory::DeviceMemory()
    : buffer{Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize(),
             VirtualReserveSize} {
#ifdef _WIN32
    zero_trim_thread = std::jthread{[this](std::stop_token stop) { TrimZeroPages(stop); }};
#endif
}

DeviceMemory::~DeviceMemory() = default;

#ifdef _WIN32
// Windows can't punch holes in the guest memory section, so pages holding only zeros (games clear
// their heaps while booting: Ocarina leaves ~7 GB) stay resident until they leave the working set.
// A pass comes every 5 s while they keep turning up, backing off to once every 5 minutes: after
// boot most passes trim 1-2 MiB for about a second of work. It waits out bulk mapping (loads,
// spawns), whose map calls stalled 10-14 ms while a pass ran, and (2.7) it waits out the guest
// presenting: a 1.7-2.1 s pass in the middle of gameplay is a freeze the player sees, and the
// saving is optional (ShouldTrimZeroPages).
//
// When host commit is tight enough to override the presenting rule (a 24 GiB laptop in the 3.2
// report trimmed 11 GiB in one 5.7 s pass and the game lost frames for seconds), the pass is cut
// into slices of GuestTrimSliceBytes: each one costs a fraction of a frame, and the pages it
// trimmed are no longer resident, so the next slice goes further without a cursor. The slices run
// back to back while the memory is tight and stop as soon as the guest pauses.
void DeviceMemory::TrimZeroPages(std::stop_token stop) {
    using namespace std::chrono_literals;
    Common::SetCurrentThreadName("ZeroTrim");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    const u64 host_ram_bytes = Common::GetMemInfo().TotalPhysicalMemory;
    // Hosts with >= 24 GiB physical RAM (such as 32 GiB laptops) do not suffer working-set
    // pressure during normal 4-6 GiB Switch 2 course loads, whereas scanning resident guest pages
    // takes ~1000 ms under VAD locks. Defer whenever guest mapping is active and only scan on idle
    // intervals once enough new zero-filled guest memory has accumulated.
    // 24 GiB hosts report 23.7 GiB, and they hold a 4-6 GiB guest as comfortably as a 32 GiB one:
    // the earlier 24 GiB cut-off treated them as tight and ran full passes during gameplay.
    const bool ample_host_ram = host_ram_bytes >= (u64{20} << 30);
    // What one pass may scan while the guest is presenting; a full pass when it is not.
    constexpr size_t GuestTrimSliceBytes = size_t{512} << 20;
    const size_t min_useful_trim = ample_host_ram ? (size_t{256} << 20) : (size_t{64} << 20);
    const u64 active_map_threshold = ample_host_ram ? (u64{8} << 20) : (u64{16} << 20);
    std::mutex mutex;
    std::condition_variable_any wake;
    std::chrono::seconds interval = ample_host_ram ? 30s : 5s;
    u64 last_zero_filled_bytes = 0;
    u32 consecutive_busy_deferrals = 0;
    u32 consecutive_presenting_deferrals = 0;
    // How long a game that keeps presenting holds a pass back once host commit is under pressure.
    const std::chrono::seconds trim_escape = ample_host_ram ? 120s : 20s;
    auto last_pass = std::chrono::steady_clock::now();
    // Time of the last bounded slice, which is not a pass for the escape interval above, and how
    // many of them ran since the last full pass.
    auto last_slice = std::chrono::steady_clock::time_point{};
    u32 slices_since_pass = 0;
    u64 last_seen_presents = Core::g_nx2_present_count.load(std::memory_order_relaxed);
    auto last_present_time = std::chrono::steady_clock::time_point{};
    while (true) {
        {
            std::unique_lock lock{mutex};
            wake.wait_for(lock, stop, interval, [] { return false; });
        }
        if (stop.stop_requested()) {
            return;
        }
        // Check whether bulk mapping/unmapping/protection (course loads, boots, spawns) is active.
        // Never force a 1-second zero-page scan in the middle of an active load on ample-RAM hosts.
        const u64 presents_before = Core::g_nx2_present_count.load(std::memory_order_relaxed);
        if (presents_before != last_seen_presents) {
            last_seen_presents = presents_before;
            last_present_time = std::chrono::steady_clock::now();
        }
        bool mapping_active = false;
        const int poll_checks = ample_host_ram ? 4 : 10;
        for (int wait = 0; wait < poll_checks; ++wait) {
            const u64 before = Common::HostMemory::MappedBytes();
            {
                std::unique_lock lock{mutex};
                wake.wait_for(lock, stop, 500ms, [] { return false; });
            }
            if (stop.stop_requested()) {
                return;
            }
            if (Common::HostMemory::MappedBytes() - before < active_map_threshold) {
                mapping_active = false;
                break;
            }
            mapping_active = true;
        }
        if (mapping_active && (ample_host_ram || consecutive_busy_deferrals < 12)) {
            ++consecutive_busy_deferrals;
            continue;
        }
        consecutive_busy_deferrals = 0;
        const auto now = std::chrono::steady_clock::now();
        const Common::ProcessMemoryUsage usage = Common::GetProcessMemoryUsage();
        const u64 presents_after = Core::g_nx2_present_count.load(std::memory_order_relaxed);
        if (presents_after != last_seen_presents) {
            last_seen_presents = presents_after;
            last_present_time = now;
        }
        const bool memory_pressure = !ample_host_ram || usage.working_set >= (host_ram_bytes / 3);
        const bool recently_presented =
            ample_host_ram && !memory_pressure &&
            last_present_time != std::chrono::steady_clock::time_point{} &&
            (now - last_present_time) < 15s;
        const bool guest_presenting =
            (presents_after != presents_before) || recently_presented;
        const auto since_pass =
            std::chrono::duration_cast<std::chrono::seconds>(now - last_pass);
        // A slice left pages behind: keep going at the slice cadence instead of waiting for the
        // escape interval again. The 2 s window covers the mapping poll between two slices.
        const bool slicing = last_slice != std::chrono::steady_clock::time_point{} &&
                             (now - last_slice) < 2s;
        if (!slicing && !ShouldTrimZeroPages(mapping_active, guest_presenting, since_pass,
                                             memory_pressure, trim_escape)) {
            // Silent after the first time: a session that plays for an hour would otherwise log a
            // deferral per interval. The first one explains where the earlier freezes went.
            if (consecutive_presenting_deferrals++ == 0) {
                LOG_INFO(HW_Memory,
                         "Deferring the all-zero guest page trim while the game runs (a pass takes "
                         "1-2 s); it runs when the game pauses, loads, or the host runs short");
            }
            // Keep the normal cadence: a pause should be noticed within one interval, not one
            // five-minute backoff.
            continue;
        }
        consecutive_presenting_deferrals = 0;
        if (ample_host_ram) {
            const u64 zero_now = Common::HostMemory::ZeroFilledBytes();
            if (usage.working_set < (host_ram_bytes / 3) &&
                zero_now - last_zero_filled_bytes < (u64{512} << 20)) {
                interval = (std::min)(interval * 2, std::chrono::seconds{300});
                continue;
            }
            last_zero_filled_bytes = zero_now;
        }
        const auto start = std::chrono::steady_clock::now();
        Common::HostMemory::ZeroTrimResult result;
        try {
            result = buffer.TrimZeroPages(guest_presenting ? GuestTrimSliceBytes : 0);
        } catch (const std::bad_alloc&) {
            // A pass needs 32 MiB and more of scratch. When Windows runs out of commit, the
            // exception leaving this thread ended the emulator (#1346); the pass is only a saving.
            interval = std::chrono::seconds{60};
            LOG_WARNING(HW_Memory,
                        "Skipped trimming all-zero guest pages: out of memory. Windows commit is "
                        "{} of {} MiB (this process {} MiB); the limit is RAM plus the page file",
                        usage.system_commit >> 20, usage.system_commit_limit >> 20,
                        usage.commit >> 20);
            continue;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        const auto now_pass = std::chrono::steady_clock::now();
        if (result.truncated && result.trimmed == 0) {
            // Slices that keep finding nothing are not progress: back off instead of spinning over
            // the same resident pages (VirtualUnlock does not always remove them at once).
            last_slice = {};
            interval = (std::min)(interval * 2, std::chrono::seconds{300});
            continue;
        }
        if (result.truncated) {
            // A slice: bounded work, and the next one carries on past the pages it trimmed. One
            // line for the first slice of a run and then every 16th, so a long trim stays legible.
            last_slice = now_pass;
            interval = ample_host_ram ? 1s : 250ms;
            if (slices_since_pass++ % 16 == 0) {
                LOG_INFO(HW_Memory,
                         "Trimmed {} MiB of all-zero guest pages out of {} MiB resident in {} ms (a "
                         "slice, {} this run; the guest kept running)",
                         result.trimmed >> 20, result.resident >> 20, ms, slices_since_pass);
            }
            continue;
        }
        last_slice = {};
        slices_since_pass = 0;
        if (result.trimmed < min_useful_trim) {
            interval = (std::min)(interval * 2, std::chrono::seconds{300});
            continue;
        }
        interval = ample_host_ram ? 30s : 5s;
        last_pass = now_pass;
        LOG_INFO(HW_Memory,
                 "Trimmed {} MiB of all-zero guest pages out of {} MiB resident in {} ms; the kernel "
                 "zero-filled {} MiB so far",
                 result.trimmed >> 20, result.resident >> 20, ms,
                 Common::HostMemory::ZeroFilledBytes() >> 20);
    }
}
#endif

} // namespace Core
