// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <charconv>
#include <cstdlib>

#include "common/frame_capture_list.h"
#include "common/logging.h"

namespace Common::FrameCapture {

std::vector<u64> ParseList(std::string_view text) {
    std::vector<u64> presents;
    while (!text.empty()) {
        const size_t comma = text.find(',');
        std::string_view item = text.substr(0, comma);
        text.remove_prefix(comma == std::string_view::npos ? text.size() : comma + 1);
        while (!item.empty() && item.front() == ' ') {
            item.remove_prefix(1);
        }
        while (!item.empty() && item.back() == ' ') {
            item.remove_suffix(1);
        }
        u64 present = 0;
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), present);
        if (error == std::errc{} && end == item.data() + item.size() && !item.empty()) {
            presents.push_back(present);
        }
    }
    std::ranges::sort(presents);
    presents.erase(std::unique(presents.begin(), presents.end()), presents.end());
    return presents;
}

namespace {

const std::vector<u64>& List() {
    static const std::vector<u64> list = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_CAPTURE_AT");
        return value != nullptr ? ParseList(value) : std::vector<u64>{};
    }();
    return list;
}

} // namespace

bool Listed(u64 present) {
    const std::vector<u64>& list = List();
    return !list.empty() && std::ranges::binary_search(list, present);
}

bool Periodic() {
    static const bool periodic = [] {
        const char* at = std::getenv("EDEN_NX2_FRAME_CAPTURE_AT");
        const bool at_supplied = at != nullptr && at[0] != '\0';
        if (at_supplied && List().empty()) {
            LOG_WARNING(Core, "EDEN_NX2_FRAME_CAPTURE_AT=\"{}\" has no valid present numbers", at);
        }
        return !at_supplied || std::getenv("EDEN_NX2_FRAME_CAPTURE_EVERY") != nullptr ||
               std::getenv("EDEN_NX2_FRAME_CAPTURE_RANGE") != nullptr;
    }();
    return periodic;
}

} // namespace Common::FrameCapture
