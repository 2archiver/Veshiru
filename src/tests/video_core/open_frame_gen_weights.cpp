// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-FileCopyrightText: Copyright (c) 2026 Prithivi Nair, Prajwal R, Madhav S, Rashmi N Ugarakhod (Open-Frame-gen)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/present/open_frame_gen_weights.h"

TEST_CASE("Open-Frame-Gen NanoModel folded weights unpack cleanly", "[video_core]") {
    namespace W = Vulkan::OpenFrameGenWeights;
    const auto weights = W::UnpackWeights();
    REQUIRE(weights.size() == W::kTotalFloatCount);
    REQUIRE(W::kHead2B + 3u == W::kTotalFloatCount);

    std::size_t non_zero = 0;
    for (const f32 v : weights) {
        REQUIRE(std::isfinite(v));
        REQUIRE(std::abs(v) < 64.0f);
        if (v != 0.0f) {
            ++non_zero;
        }
    }
    CHECK(non_zero > (W::kTotalFloatCount * 9) / 10);
}
