// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (memulator)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_gpu_writes.h"

// GpuWriteLog (nvn_gpu_writes.h): which unfinished submission writes which guest range, for
// EDEN_NX2_TRY_INVALIDATE_WRITEBACK.
using Vulkan::Nvn::GpuWriteLog;

TEST_CASE("GpuWriteLog: the newest unfinished writer of an overlapping range", "[nvn][gpu-writes]") {
    GpuWriteLog log;
    log.Note(0x1000, 0x1100, 5, 0);
    log.Note(0x2000, 0x2100, 7, 0);
    log.Note(0x1080, 0x1180, 9, 0);

    // Overlap by one byte counts, touching ranges do not.
    CHECK(log.Latest(0x10FF, 0x1101, ~u64{0}, 0) == 9);
    CHECK(log.Latest(0x1180, 0x1200, ~u64{0}, 0) == 0);
    CHECK(log.Latest(0x0FC0, 0x1000, ~u64{0}, 0) == 0);
    CHECK(log.Latest(0x0FC0, 0x1001, ~u64{0}, 0) == 5);
    // The newest one wins.
    CHECK(log.Latest(0x1000, 0x1200, ~u64{0}, 0) == 9);
    CHECK(log.Latest(0x1000, 0x2100, ~u64{0}, 0) == 9);
    CHECK(log.Latest(0x2000, 0x2001, ~u64{0}, 0) == 7);
    // Empty ranges and ranges nobody writes.
    CHECK(log.Latest(0x3000, 0x3100, ~u64{0}, 0) == 0);
    CHECK(log.Latest(0x1000, 0x1000, ~u64{0}, 0) == 0);
}

TEST_CASE("GpuWriteLog: completed submissions and later submissions are not waited for",
          "[nvn][gpu-writes]") {
    GpuWriteLog log;
    log.Note(0x1000, 0x1100, 5, 0);
    log.Note(0x1000, 0x1100, 9, 0);
    // `max_seq`: only the submissions made before the invalidate.
    CHECK(log.Latest(0x1000, 0x1100, 8, 0) == 5);
    CHECK(log.Latest(0x1000, 0x1100, 4, 0) == 0);
    CHECK(log.Latest(0x1000, 0x1100, 9, 0) == 9);
    // `completed_seq`: finished work is not waited for.
    CHECK(log.Latest(0x1000, 0x1100, 9, 5) == 9);
    CHECK(log.Latest(0x1000, 0x1100, 9, 9) == 0);
    CHECK(log.Latest(0x1000, 0x1100, 8, 5) == 0);
}

TEST_CASE("GpuWriteLog: writes of one submission merge, and the log stays bounded",
          "[nvn][gpu-writes]") {
    GpuWriteLog log;
    log.Note(0x1000, 0x1100, 5, 0);
    log.Note(0x1100, 0x1200, 5, 0); // adjacent: one entry
    log.Note(0x10F0, 0x1110, 5, 0); // overlapping: still one entry
    CHECK(log.Size() == 1);
    CHECK(log.Latest(0x11F0, 0x1200, ~u64{0}, 0) == 5);

    // Completed submissions are dropped once the log fills up.
    GpuWriteLog many;
    for (u64 i = 0; i < GpuWriteLog::PRUNE_AT - 1; ++i) {
        many.Note(i * 0x1000, i * 0x1000 + 0x10, i + 1, 0);
    }
    CHECK(many.Size() == GpuWriteLog::PRUNE_AT - 1);
    many.Note(0x900000, 0x900010, 1000, GpuWriteLog::PRUNE_AT);
    CHECK(many.Size() == 1);
    CHECK(many.Latest(0x900000, 0x900010, ~u64{0}, GpuWriteLog::PRUNE_AT) == 1000);

    // Unfinished writes beyond the bound collapse into one hull per submission: the wait for
    // any byte of the hull is only ever longer, never missed.
    GpuWriteLog flood;
    for (u64 i = 0; i < GpuWriteLog::PRUNE_AT * 3; ++i) {
        flood.Note(i * 0x1000, i * 0x1000 + 0x10, 7 + (i & 1), 0);
        CHECK(flood.Size() <= GpuWriteLog::PRUNE_AT);
    }
    CHECK(flood.Latest(0x1000, 0x1010, ~u64{0}, 0) == 7 + 1);
    CHECK(flood.Latest(0x0000, 0x0010, ~u64{0}, 0) == 7);
}
