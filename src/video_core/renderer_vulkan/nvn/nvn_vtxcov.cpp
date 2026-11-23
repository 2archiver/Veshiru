// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/nvn/nvn_vtxcov.h"

namespace Vulkan::Nvn {

std::string VtxCovSwitchNames(u32 switches) {
    static constexpr std::array<std::pair<u32, std::string_view>, 4> names{{
        {VTXCOV_INPASS, "inpass"},
        {VTXCOV_SHADER_WRITES, "shaderwrites"},
        {VTXCOV_BOUND, "bound"},
        {VTXCOV_BIGPOOL, "bigpool"},
    }};
    std::string text;
    for (const auto& [bit, name] : names) {
        if ((switches & bit) != 0) {
            text += text.empty() ? "" : ",";
            text += name;
        }
    }
    return text.empty() ? std::string{"none"} : text;
}

std::string FormatVtxCov(const VtxCovCounts& counts, u64 presents, u64 copy_in_bytes,
                         u64 staged_bytes, u32 switches) {
    const double per = presents != 0 ? 1.0 / static_cast<double>(presents) : 1.0;
    std::array<u64, 2> served{};
    std::array<u64, 2> missed{};
    std::array<u64, 2> served_bytes{};
    std::array<u64, 2> missed_bytes{};
    for (size_t kind = 0; kind < 2; ++kind) {
        for (size_t r = 0; r < VTXCOV_REASONS; ++r) {
            const bool ok = VtxCovServed(static_cast<VtxCov>(r));
            (ok ? served : missed)[kind] += counts.binds[kind][r];
            (ok ? served_bytes : missed_bytes)[kind] += counts.bytes[kind][r];
        }
    }
    const auto pair = [&](u64 vertex, u64 index) {
        return fmt::format("{:.1f}/{:.1f}", static_cast<double>(vertex) * per,
                           static_cast<double>(index) * per);
    };
    const auto kib = [&](u64 vertex, u64 index) {
        return fmt::format("{:.0f}/{:.0f}", static_cast<double>(vertex) * per / 1024.0,
                           static_cast<double>(index) * per / 1024.0);
    };
    const auto list = [&](bool want_served, bool bytes) {
        std::string text;
        for (size_t r = 0; r < VTXCOV_REASONS; ++r) {
            if (VtxCovServed(static_cast<VtxCov>(r)) != want_served ||
                counts.binds[0][r] + counts.binds[1][r] == 0) {
                continue;
            }
            text += text.empty() ? "" : ",";
            text += fmt::format("{}={}", VTXCOV_NAMES[r],
                                bytes ? kib(counts.bytes[0][r], counts.bytes[1][r])
                                      : pair(counts.binds[0][r], counts.binds[1][r]));
        }
        return text;
    };
    return fmt::format("presents={} per present, vertex/index: served={}{{{}}} not_served={}{{{}}} "
                       "bytes served_KiB={} not_served_KiB={}{{{}}} copy_in_KiB={:.0f} "
                       "stage_copy_KiB={:.0f} switches={}",
                       presents, pair(served[0], served[1]), list(true, false),
                       pair(missed[0], missed[1]), list(false, false),
                       kib(served_bytes[0], served_bytes[1]), kib(missed_bytes[0], missed_bytes[1]),
                       list(false, true), static_cast<double>(copy_in_bytes) * per / 1024.0,
                       static_cast<double>(staged_bytes) * per / 1024.0,
                       VtxCovSwitchNames(switches));
}

} // namespace Vulkan::Nvn
