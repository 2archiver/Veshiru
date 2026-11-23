// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/settings_enums.h"

namespace Vulkan::Nvn {

/// Whether the NVN HLE graphics path applies `type` when frame generation is on. In Eglantine no
/// type runs on it: the FSR3 and Lossless Scaling chains record into the emulated GPU's scheduler,
/// not into the replay's command buffer, and there is no DLSS frame generation.
[[nodiscard]] constexpr bool FrameGenTypeSupported(Settings::FrameGenType type) {
    static_cast<void>(type);
    return false;
}

/// Whether the presented frame image needs a storage view: DLAA writes it from a compute shader,
/// and the FSR3 and Lossless Scaling chains read it and write the generated frames through one.
[[nodiscard]] constexpr bool PresentFrameNeedsStorage(bool dlaa, bool frame_gen_enabled,
                                                      Settings::FrameGenType type) {
    return dlaa || (frame_gen_enabled && FrameGenTypeSupported(type));
}

} // namespace Vulkan::Nvn
