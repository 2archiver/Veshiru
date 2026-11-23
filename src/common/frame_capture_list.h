// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>
#include <vector>

#include "common/common_types.h"

/// EDEN_NX2_FRAME_CAPTURE_AT=<present>[,<present>...]: EDEN_NX2_FRAME_CAPTURE also saves exactly
/// these presents (e.g. a recorded playtest's checkpoints). Given alone, without
/// EDEN_NX2_FRAME_CAPTURE_EVERY / _RANGE, it replaces the default every-120th schedule.
namespace Common::FrameCapture {

/// Present numbers of "a,b,c", sorted, without duplicates; items that aren't numbers are skipped.
[[nodiscard]] std::vector<u64> ParseList(std::string_view text);

/// True when EDEN_NX2_FRAME_CAPTURE_AT lists `present`.
[[nodiscard]] bool Listed(u64 present);

/// False when EDEN_NX2_FRAME_CAPTURE_AT is the only schedule given (the periodic one is off).
[[nodiscard]] bool Periodic();

} // namespace Common::FrameCapture
