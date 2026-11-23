// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/upscaler.h"

namespace VideoCore::Upscaler {

// Eglantine has no upscaler setting and no NVIDIA DLSS runtime: VTX-HLE runs the game's own
// network where the device can (cooperative matrices), FSR elsewhere.
Settings::Upscaler Effective() {
    return Settings::Upscaler::Auto;
}

void ResetSession() noexcept {}

} // namespace VideoCore::Upscaler
