// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/memory_limits.h"

using namespace Common::Literals;
namespace TCM = VideoCommon::TextureCacheMemory;

TEST_CASE("Texture cache memory limit: Auto is three quarters of the device", "[video_core]") {
    CHECK(TCM::ResolveLimit(0, 16_GiB) == 12_GiB);
    CHECK(TCM::ResolveLimit(0, 8_GiB) == 6_GiB);
    CHECK(TCM::RecommendedLimit(0) == 0);
}

TEST_CASE("Texture cache memory limit: explicit values are clamped to the device", "[video_core]") {
    CHECK(TCM::ResolveLimit(4096, 16_GiB) == 4_GiB);
    CHECK(TCM::ResolveLimit(512, 16_GiB) == TCM::MinLimit);
    CHECK(TCM::ResolveLimit(65536, 16_GiB) == 16_GiB);
    // A device under the minimum is the ceiling.
    CHECK(TCM::ResolveLimit(512, 512_MiB) == 512_MiB);
}

TEST_CASE("Texture cache memory limit: unknown device memory", "[video_core]") {
    CHECK(TCM::ResolveLimit(0, 0) == ~u64{0});
    CHECK(TCM::ResolveLimit(3000, 0) == 3000_MiB);
}

TEST_CASE("Texture cache memory limit: warning below the recommendation", "[video_core]") {
    CHECK_FALSE(TCM::IsBelowRecommended(0, 8_GiB));
    CHECK(TCM::IsBelowRecommended(4096, 8_GiB));
    CHECK_FALSE(TCM::IsBelowRecommended(6144, 8_GiB));
    CHECK_FALSE(TCM::IsBelowRecommended(8192, 8_GiB));
    CHECK_FALSE(TCM::IsBelowRecommended(1024, 0));
}

TEST_CASE("VRAM budgeting tiers for 4 GB, 6 GB, and 8+ GB GPUs", "[video_core][vram_budget]") {
    // Dynamic driver budget refresh keeps tracking down to 1 GiB (not frozen at <= 2 GiB).
    CHECK(TCM::MinTrackedDeviceBudget == 1_GiB);

    // GTX 1650 4 GB (~3935 MiB device-local): tight VRAM tier kicks in.
    const auto gtx1650 = TCM::ResolveNativeBufferGcThresholds(3935_MiB);
    CHECK(gtx1650.tight_vram);
    CHECK(gtx1650.expected_bytes == 192_MiB);
    CHECK(gtx1650.critical_bytes == 384_MiB);
    CHECK(gtx1650.pressure_floor_bytes == 128_MiB);
    CHECK(TCM::ResolveTrimmedAstcArrayLimit(3935_MiB) == 128_MiB);

    // 6 GB GPU: mid tier.
    const auto gpu6g = TCM::ResolveNativeBufferGcThresholds(6_GiB);
    CHECK_FALSE(gpu6g.tight_vram);
    CHECK(gpu6g.expected_bytes == 320_MiB);
    CHECK(gpu6g.critical_bytes == 640_MiB);
    CHECK(TCM::ResolveTrimmedAstcArrayLimit(6_GiB) == 256_MiB);

    // 8+ GB GPU and unknown VRAM (0): preserve original 512 MiB / 1 GiB thresholds.
    for (const u64 vram : {8_GiB, 16_GiB, u64{0}}) {
        const auto high = TCM::ResolveNativeBufferGcThresholds(vram);
        CHECK_FALSE(high.tight_vram);
        CHECK(high.expected_bytes == 512_MiB);
        CHECK(high.critical_bytes == 1_GiB);
    }
    CHECK(TCM::ResolveTrimmedAstcArrayLimit(8_GiB) == 512_MiB);
    CHECK(TCM::ResolveTrimmedAstcArrayLimit(16_GiB) == 512_MiB);
}
