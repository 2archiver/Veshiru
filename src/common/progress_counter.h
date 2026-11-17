// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "common/common_types.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace Common {

/// A spin-wait hint for the current core.
inline void CpuRelax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

/// A monotonic completion position shared by producers of progress and any number of waiters:
/// ring positions consumed by a worker, fences signalled, ticks submitted.
class ProgressCounter {
public:
    enum class WaitResult {
        Reached,   ///< The value reached the target
        Cancelled, ///< Cancel() was called (or the abort predicate held) before it did
        TimedOut,  ///< The deadline passed before it did
    };

    static constexpr u64 NoTarget = ~u64{0};
    /// Default time a waiter spins before it blocks.
    static constexpr std::chrono::microseconds DefaultSpin{20};

    explicit ProgressCounter(u64 initial = 0) : value{initial} {}

    ProgressCounter(const ProgressCounter&) = delete;
    ProgressCounter& operator=(const ProgressCounter&) = delete;

    [[nodiscard]] u64 Value() const noexcept {
        return value.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool Reached(u64 target) const noexcept {
        return value.load(std::memory_order_acquire) >= target;
    }

    [[nodiscard]] bool IsCancelled() const noexcept {
        return cancelled.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 LowestTarget() const noexcept {
        return lowest_target.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool HasSleepers() const noexcept {
        return sleepers.load(std::memory_order_relaxed) != 0;
    }

    void Advance(u64 new_value) noexcept {
        value.store(new_value, std::memory_order_seq_cst);
        NotifyIfSleeping();
    }

    void Publish(u64 new_value) noexcept {
        value.store(new_value, std::memory_order_release);
    }

    void AdvanceAtLeast(u64 new_value) noexcept {
        u64 current = value.load(std::memory_order_relaxed);
        while (current < new_value &&
               !value.compare_exchange_weak(current, new_value, std::memory_order_seq_cst,
                                            std::memory_order_relaxed)) {
        }
        NotifyIfSleeping();
    }

    u64 Increment(u64 amount = 1) noexcept {
        const u64 result = value.fetch_add(amount, std::memory_order_seq_cst) + amount;
        NotifyIfSleeping();
        return result;
    }

    void Cancel() noexcept {
        cancelled.store(true, std::memory_order_seq_cst);
        std::scoped_lock lk{mutex};
        cv.notify_all();
    }

    void Reset(u64 new_value = 0) noexcept {
        value.store(new_value, std::memory_order_seq_cst);
        cancelled.store(false, std::memory_order_seq_cst);
        lowest_target.store(NoTarget, std::memory_order_relaxed);
    }

    WaitResult WaitFor(u64 target, std::chrono::nanoseconds spin = DefaultSpin) {
        return Wait(target, spin, std::chrono::steady_clock::time_point::max(), [] { return false; },
                    std::chrono::milliseconds{1});
    }

    WaitResult WaitUntil(u64 target, std::chrono::steady_clock::time_point deadline,
                         std::chrono::nanoseconds spin = DefaultSpin) {
        return Wait(target, spin, deadline, [] { return false; }, std::chrono::milliseconds{1});
    }

    template <typename Abort>
    WaitResult Wait(u64 target, std::chrono::nanoseconds spin,
                    std::chrono::steady_clock::time_point deadline, Abort&& abort,
                    std::chrono::nanoseconds poll) {
        if (Reached(target)) {
            return WaitResult::Reached;
        }
        waiters.fetch_add(1, std::memory_order_relaxed);
        RequestTarget(target);
        if (spin.count() > 0) {
            const auto spin_end = std::chrono::steady_clock::now() + spin;
            for (u32 i = 1;; ++i) {
                if (Reached(target)) {
                    return Finish(WaitResult::Reached);
                }
                if (IsCancelled()) {
                    return Finish(WaitResult::Cancelled);
                }
                CpuRelax();
                if ((i & 63) == 0) {
                    RequestTarget(target);
                    const auto now = std::chrono::steady_clock::now();
                    if (now >= spin_end || now >= deadline) {
                        break;
                    }
                    if (abort()) {
                        return Finish(Reached(target) ? WaitResult::Reached : WaitResult::Cancelled);
                    }
                }
            }
        }
        std::unique_lock lk{mutex};
        sleepers.fetch_add(1, std::memory_order_seq_cst);
        blocks.fetch_add(1, std::memory_order_relaxed);
        WaitResult result = WaitResult::Reached;
        for (;;) {
            if (value.load(std::memory_order_seq_cst) >= target) {
                result = WaitResult::Reached;
                break;
            }
            if (cancelled.load(std::memory_order_seq_cst)) {
                result = WaitResult::Cancelled;
                break;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                result = WaitResult::TimedOut;
                break;
            }
            lk.unlock();
            const bool aborted = abort();
            lk.lock();
            if (aborted) {
                result = value.load(std::memory_order_seq_cst) >= target ? WaitResult::Reached
                                                                          : WaitResult::Cancelled;
                break;
            }
            RequestTarget(target);
            const auto slice_end = (deadline - now) > poll ? now + poll : deadline;
            cv.wait_until(lk, slice_end, [&] {
                return value.load(std::memory_order_seq_cst) >= target ||
                       cancelled.load(std::memory_order_seq_cst);
            });
        }
        sleepers.fetch_sub(1, std::memory_order_relaxed);
        lk.unlock();
        return Finish(result);
    }

    [[nodiscard]] u64 BlockCount() const noexcept {
        return blocks.load(std::memory_order_relaxed);
    }

private:
    void NotifyIfSleeping() noexcept {
        if (sleepers.load(std::memory_order_seq_cst) != 0) {
            { std::scoped_lock lk{mutex}; }
            cv.notify_all();
        }
    }

    void RequestTarget(u64 target) noexcept {
        u64 current = lowest_target.load(std::memory_order_relaxed);
        while (target < current &&
               !lowest_target.compare_exchange_weak(current, target, std::memory_order_seq_cst,
                                                    std::memory_order_relaxed)) {
        }
    }

    WaitResult Finish(WaitResult result) noexcept {
        if (waiters.fetch_sub(1, std::memory_order_relaxed) == 1) {
            lowest_target.store(NoTarget, std::memory_order_relaxed);
        }
        return result;
    }

    alignas(64) std::atomic<u64> value;
    alignas(64) std::atomic<u32> sleepers{};
    std::atomic<u32> waiters{}; ///< Spinning or blocked
    std::atomic<bool> cancelled{};
    std::atomic<u64> lowest_target{NoTarget};
    std::atomic<u64> blocks{};
    std::mutex mutex;
    std::condition_variable cv;
};

} // namespace Common
