// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <numeric>
#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/nx2_bytes_equal.h"
#include "video_core/page_filter.h"

using VideoCommon::PageFilter;

namespace {
constexpr u64 Page = 0x1000;
}

TEST_CASE("Page filter reports nothing before any mark", "[video_core][page_filter]") {
    PageFilter filter;
    CHECK_FALSE(filter.Any(0, Page));
    CHECK_FALSE(filter.Any(0x1234'5000, 64 * Page));
    CHECK_FALSE(filter.Any(0, 0));
}

TEST_CASE("Page filter covers every page a mark touches", "[video_core][page_filter]") {
    PageFilter filter;
    filter.Mark(3 * Page + 0x10, 2 * Page);
    CHECK_FALSE(filter.Any(2 * Page, Page));
    CHECK(filter.Any(3 * Page, 1));
    CHECK(filter.Any(4 * Page, Page));
    CHECK(filter.Any(5 * Page, 1));
    CHECK_FALSE(filter.Any(6 * Page, Page));
    CHECK(filter.Any(0, 16 * Page));
}

TEST_CASE("Page filter handles marks across 64-page words", "[video_core][page_filter]") {
    PageFilter filter;
    filter.Mark(60 * Page, 200 * Page);
    CHECK_FALSE(filter.Any(59 * Page, Page));
    CHECK(filter.Any(60 * Page, Page));
    CHECK(filter.Any(63 * Page, 2 * Page));
    CHECK(filter.Any(128 * Page, 64 * Page));
    CHECK(filter.Any(259 * Page, Page));
    CHECK_FALSE(filter.Any(260 * Page, Page));
}

TEST_CASE("Page filter ignores empty marks", "[video_core][page_filter]") {
    PageFilter filter;
    filter.Mark(8 * Page, 0);
    CHECK_FALSE(filter.Any(8 * Page, Page));
}

TEST_CASE("Page filter answers conservatively outside its range", "[video_core][page_filter]") {
    constexpr u64 Limit = u64{1} << 36;
    PageFilter filter;
    CHECK(filter.Any(Limit, Page));
    CHECK_FALSE(filter.Any(0, Page));

    filter.Mark(Limit, Page);
    CHECK(filter.Any(Limit, Page));
    CHECK_FALSE(filter.Any(0, Page));
}

TEST_CASE("Page filter Take consumes one page and Clear resets all marks",
          "[video_core][page_filter]") {
    constexpr u64 Limit = u64{1} << 36;
    PageFilter filter;
    filter.Mark(10 * Page + 0x40, 2 * Page);
    CHECK(filter.Take(10 * Page + 0xff));
    CHECK_FALSE(filter.Take(10 * Page));
    CHECK_FALSE(filter.Any(10 * Page, Page));
    CHECK(filter.Any(11 * Page, Page));

    filter.Clear();
    CHECK_FALSE(filter.Any(11 * Page, Page));
    CHECK(filter.Take(Limit));
}

TEST_CASE("Nx2BytesEqual matches across 128-byte, 16-byte, and unaligned tails",
          "[video_core][page_filter]") {
    std::array<u8, 4129> a{};
    std::iota(a.begin(), a.end(), u8{7});
    auto b = a;

    CHECK(VideoCommon::Nx2BytesEqual(a.data(), b.data(), 0));
    CHECK(VideoCommon::Nx2BytesEqual(a.data(), b.data(), Page));
    CHECK(VideoCommon::Nx2BytesEqual(a.data() + 3, b.data() + 3, 4099));

    for (const size_t pos : {size_t{0}, size_t{15}, size_t{127}, size_t{128}, size_t{4095},
                             size_t{4128}}) {
        b[pos] ^= 0x01;
        CHECK_FALSE(VideoCommon::Nx2BytesEqual(a.data(), b.data(), a.size()));
        b[pos] ^= 0x01;
    }
}
