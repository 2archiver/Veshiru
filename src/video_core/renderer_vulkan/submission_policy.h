// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <charconv>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace Vulkan::SubmissionPolicy {

inline constexpr u32 DefaultNativeBudget = 128;
inline constexpr u32 WaitedSubmitDraws = 32; ///< Minimum recorded draws before an unblock submit
inline constexpr u32 IdleSubmitDraws = 64;   ///< Minimum recorded draws before an idle submit

enum class Action {
    None,
    Dispatch,
    Flush,
};

enum class IdleSubmitReason {
    None,
    FrontWaiting,
    GpuIdle,
};

constexpr IdleSubmitReason ChooseIdleSubmit(u32 draws, bool front_waiting,
                                            bool gpu_caught_up) noexcept {
    if (front_waiting && draws >= WaitedSubmitDraws) {
        return IdleSubmitReason::FrontWaiting;
    }
    if (gpu_caught_up && draws >= IdleSubmitDraws) {
        return IdleSubmitReason::GpuIdle;
    }
    return IdleSubmitReason::None;
}

inline std::optional<u32> ParseBudget(std::string_view value) {
    if (value.empty()) {
        return std::nullopt;
    }
    u32 budget{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), budget);
    if (error != std::errc{} || end != value.data() + value.size() || budget == 0) {
        return std::nullopt;
    }
    return budget;
}

constexpr Action Advance(u32& counter, bool native, bool android,
                         u32 native_budget = DefaultNativeBudget, u32 legacy_budget = 4096) {
    ++counter;
    if (native && !android) {
        if (counter >= native_budget) {
            counter = 0;
            return Action::Flush;
        }
        return counter % 32 == 0 ? Action::Dispatch : Action::None;
    }
    const u32 check_mask = android ? 3 : 31;
    if ((counter & check_mask) != check_mask) {
        return Action::None;
    }
    if (counter < (android ? 512U : legacy_budget)) {
        return Action::Dispatch;
    }
    counter = 0;
    return Action::Flush;
}

} // namespace Vulkan::SubmissionPolicy
