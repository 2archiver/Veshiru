// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <vector>

#include "common/common_types.h"

namespace Vulkan {

/// Whether an acquire waits for the renderer scheduler's tick of the image (frame pacing and the
/// frame-rate cap). Frames the scheduler rendered do. Frames another queue user submitted (the NVN
/// replay) don't when the present timeline exists: the scheduler holds none of their work, so its
/// tick is always the next one and the wait would flush it and drain the whole queue, the frame
/// just submitted included, on every few presents. Those frames are paced by their own fences, the
/// game's own present pacing and the present timeline.
[[nodiscard]] constexpr bool WaitsForRendererTick(bool renderer_paced, bool has_present_timeline) {
    return renderer_paced || !has_present_timeline;
}

/// For each acquire semaphore of a swapchain, the present-timeline value signalled by the
/// submission that last waited on it. The semaphore may be used for another acquire only once
/// that submission ran (no wait may be pending on it); 0 means nothing is pending.
class PresentSemaphoreTicks {
public:
    /// New semaphores: none has a submission waiting on it.
    void Reset(std::size_t count) {
        ticks.assign(count, 0);
    }

    /// Timeline value to reach before acquiring with semaphore `slot` (0 = no wait).
    [[nodiscard]] u64 Pending(std::size_t slot) const {
        return ticks[slot];
    }

    /// Value the next submission waiting on an acquire semaphore signals.
    [[nodiscard]] u64 Next() const {
        return last + 1;
    }

    /// The submission waiting on semaphore `slot` was submitted signalling `Next()`.
    void Submitted(std::size_t slot) {
        ticks[slot] = ++last;
    }

private:
    std::vector<u64> ticks;
    u64 last{}; ///< Last value submitted; never reset, the timeline only counts up.
};

} // namespace Vulkan
