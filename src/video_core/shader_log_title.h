// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/settings.h"
#include "video_core/shader_log.h"

namespace VideoCommon::ShaderLog {

/// Tells the shader log which title it is logging. The log (`Start()`) can exist before the game
/// is known, so the first hooks that run with the log on call this until the core has set the
/// program id; after that it is two relaxed loads. Call it before the first Submit of a path, and
/// from a per-frame hook as a backstop.
inline void UpdateTitle() {
    if (!Enabled() || TitleKnown()) {
        return;
    }
    const u64 title_id = Settings::GetCurrentProgramID();
    if (title_id == 0) {
        return;
    }
    SetTitle(title_id, {});
}

} // namespace VideoCommon::ShaderLog
