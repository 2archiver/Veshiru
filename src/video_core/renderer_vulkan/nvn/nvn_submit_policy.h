// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include "common/common_types.h"

namespace Vulkan::Nvn {

// Preserve CPU/GPU overlap without repeatedly flushing live storage shadows or short passes.
[[nodiscard]] constexpr bool MkwEarlySubmitDue(u32 pieces, u64 elapsed_us, u64 normal_us,
                                              u64 pass_us, bool pass_open, bool shadows) {
    if (pieces >= 2) {
        return false;
    }
    if (pass_open || shadows) {
        if (pass_us == 0) {
            return false;
        }
        normal_us = std::max(normal_us, pass_us);
    }
    return elapsed_us >= normal_us;
}

} // namespace Vulkan::Nvn
