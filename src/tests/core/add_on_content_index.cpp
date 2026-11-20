// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <span>

#include <catch2/catch_test_macros.hpp>

#include "core/file_sys/common_funcs.h"

namespace {

constexpr u64 Program = 0x040074A01BF12000;
constexpr std::array<u64, 3> AddOns{Program + 0x1001, Program + 0x1002, Program + 0x17FF};

} // namespace

TEST_CASE("Add-on index: the indices ListAddOnContent hands out are recognised",
          "[core][aoc]") {
    CHECK(FileSys::IsListedAddOnContentIndex(Program, 1, AddOns));
    CHECK(FileSys::IsListedAddOnContentIndex(Program, 2, AddOns));
    CHECK(FileSys::IsListedAddOnContentIndex(Program, 0x7FF, AddOns));
    // The caller may be a program index of the same application
    CHECK(FileSys::IsListedAddOnContentIndex(Program + 1, 1, AddOns));
}

TEST_CASE("Add-on index: an index that names nothing is not recognised", "[core][aoc]") {
    CHECK_FALSE(FileSys::IsListedAddOnContentIndex(Program, 0, AddOns));
    CHECK_FALSE(FileSys::IsListedAddOnContentIndex(Program, 3, AddOns));
    CHECK_FALSE(FileSys::IsListedAddOnContentIndex(Program, -1, AddOns));
    CHECK_FALSE(FileSys::IsListedAddOnContentIndex(Program, 1, std::span<const u64>{}));
}

TEST_CASE("Add-on index: another application's add-ons do not count", "[core][aoc]") {
    constexpr u64 other = 0x0400ABCD00022000;
    constexpr std::array<u64, 1> foreign{other + 0x1001};
    CHECK_FALSE(FileSys::IsListedAddOnContentIndex(Program, 1, foreign));
    CHECK(FileSys::IsListedAddOnContentIndex(other, 1, foreign));
}
