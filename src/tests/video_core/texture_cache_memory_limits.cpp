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

    // GTX 1650 4 GB (~3935 MiB device-local): keeps the 512 MiB collection start so ~300 MiB live
    // GPU-written working sets never thrash aggressive GC, allows clean-only shedding from 256 MiB
    // under device pressure, and caps aggressive eviction at 768 MiB.
    const auto gtx1650 = TCM::ResolveNativeBufferGcThresholds(3935_MiB);
    CHECK(gtx1650.tight_vram);
    CHECK(gtx1650.expected_bytes == 512_MiB);
    CHECK(gtx1650.critical_bytes == 768_MiB);
    CHECK(gtx1650.pressure_floor_bytes == 256_MiB);
    CHECK(TCM::ResolveTrimmedAstcArrayLimit(3935_MiB) == 128_MiB);

    // 6 GB GPU: keeps full 512 MiB / 1 GiB buffer GC thresholds while trimming ASTC arrays.
    const auto gpu6g = TCM::ResolveNativeBufferGcThresholds(6_GiB);
    CHECK_FALSE(gpu6g.tight_vram);
    CHECK(gpu6g.expected_bytes == 512_MiB);
    CHECK(gpu6g.critical_bytes == 1_GiB);
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

TEST_CASE("Integrated and unified GPU memory scales past 2048 MiB UMA carveouts and 4 GiB caps",
          "[video_core][vram_budget][unified_memory]") {
    // 16 GB Windows APU with 2048 MiB BIOS UMA carveout + 7 GiB shared system GPU segment
    // (OS sees 14 GiB): no longer locked to 2048 MiB; gets 6 GiB default and full 9 GiB pool max.
    CHECK(TCM::ResolveIntegratedGpuMemory(14_GiB, 2_GiB, 7_GiB, 9_GiB, false) == 6_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(14_GiB, 2_GiB, 7_GiB, 9_GiB, true) == 9_GiB);

    // 16 GB Windows APU with 512 MiB Auto UMA carveout + 7.75 GiB shared segment (OS sees 15.5 GiB).
    const u64 os_15_5g = 15_GiB + 512_MiB;
    const u64 shared_7_75g = 7_GiB + 768_MiB;
    CHECK(TCM::ResolveIntegratedGpuMemory(os_15_5g, 512_MiB, shared_7_75g, 0, false) == 6_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(os_15_5g, 512_MiB, shared_7_75g, 0, true) ==
          8_GiB + 256_MiB);

    // 24 GB handheld (2 GiB and 8 GiB UMA carveouts): scales to 9 GiB default and 13-15 GiB max.
    CHECK(TCM::ResolveIntegratedGpuMemory(22_GiB, 2_GiB, 11_GiB, 13_GiB, false) == 9_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(22_GiB, 2_GiB, 11_GiB, 13_GiB, true) == 13_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(16_GiB, 8_GiB, 8_GiB, 16_GiB, false) == 9_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(16_GiB, 8_GiB, 8_GiB, 16_GiB, true) == 15_GiB);

    // 32 GB Windows APU (2048 MiB UMA + 15 GiB shared) and 32 GB single-heap RADV unified iGPU:
    // scales far beyond the old 2/4 GiB cap while staying below 20 GiB so BC3 remains enabled.
    CHECK(TCM::ResolveIntegratedGpuMemory(30_GiB, 2_GiB, 15_GiB, 17_GiB, false) == 12_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(30_GiB, 2_GiB, 15_GiB, 17_GiB, true) == 17_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(30_GiB, 24_GiB, 0, 21_GiB, false) == 11_GiB + 256_MiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(30_GiB, 24_GiB, 0, 21_GiB, true) == 18_GiB + 768_MiB);

    // 64 GB and 128 GB unified RAM systems (including a dedicated 96 GiB BIOS UMA carveout).
    CHECK(TCM::ResolveIntegratedGpuMemory(62_GiB, 2_GiB, 31_GiB, 33_GiB, false) == 24_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(62_GiB, 2_GiB, 31_GiB, 33_GiB, true) == 33_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(64_GiB, 48_GiB, 0, 48_GiB, false) == 24_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(64_GiB, 48_GiB, 0, 48_GiB, true) == 40_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(32_GiB, 96_GiB, 16_GiB, 112_GiB, false) == 96_GiB);
    CHECK(TCM::ResolveIntegratedGpuMemory(32_GiB, 96_GiB, 16_GiB, 112_GiB, true) == 96_GiB);

    // Explicit user Texture Cache Memory Limit expands live iGPU budget up to max unified ceiling
    // without exceeding the live OS/driver budget.
    CHECK(TCM::ClampIntegratedGpuBudget(17_GiB, 12_GiB, 17_GiB, 0, 1_GiB) == 12_GiB);
    CHECK(TCM::ClampIntegratedGpuBudget(17_GiB, 12_GiB, 17_GiB, 14336, 1_GiB) == 15_GiB);
    CHECK(TCM::ClampIntegratedGpuBudget(13_GiB, 12_GiB, 17_GiB, 16384, 1_GiB) == 13_GiB);
}
