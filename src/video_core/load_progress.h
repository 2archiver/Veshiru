// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>
#include <utility>

#include "video_core/rasterizer_interface.h"

namespace VideoCore {

/// Where a disk cache that loads after the frontend's own load call reports its progress. The
/// rasterizer's LoadDiskResources drives the loading screen before the guest runs; the NVN
/// pipeline cache opens when the guest names its title, so its preload reports here instead, with
/// the same stages and counts (Prepare, Build value/total, Complete). The frontend sets the
/// callback for the session (it forwards to the loading screen); nothing is reported while none is
/// set. Thread safe: workers report, the frontend sets and clears.
class LoadProgressSink {
public:
    void Set(DiskResourceLoadCallback callback) {
        std::scoped_lock lock{mutex};
        target = std::move(callback);
    }

    void Clear() {
        std::scoped_lock lock{mutex};
        target = nullptr;
    }

    void Report(LoadCallbackStage stage, std::size_t value, std::size_t total) const {
        std::scoped_lock lock{mutex};
        if (target) {
            target(stage, value, total);
        }
    }

private:
    mutable std::mutex mutex;
    DiskResourceLoadCallback target;
};

} // namespace VideoCore
