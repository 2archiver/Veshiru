// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_backing_log.h"

namespace Vulkan::Nvn {

TEST_CASE("NVN backing log classifies overlays before pool backing", "[nvn][backing]") {
    CHECK(ClassifyBacking(true, true, true, true, true) == BackingClass::RawGlobalTable);
    CHECK(ClassifyBacking(false, true, true, true, true) == BackingClass::ShadowCopy);
    CHECK(ClassifyBacking(false, false, true, true, true) == BackingClass::PoolMirror);
}

TEST_CASE("NVN backing log classifies pool backing and unknown ranges", "[nvn][backing]") {
    CHECK(ClassifyBacking(false, false, false, true, false) == BackingClass::ImportedHostPool);
    CHECK(ClassifyBacking(false, false, false, false, true) == BackingClass::DeviceLocalPool);
    CHECK(ClassifyBacking(false, false, false, false, false) == BackingClass::Other);
    CHECK(BackingClassName(BackingClass::ImportedHostPool) == "imported host pool");
    CHECK(BackingClassName(BackingClass::RawGlobalTable) == "raw-global table");
}

} // namespace Vulkan::Nvn
