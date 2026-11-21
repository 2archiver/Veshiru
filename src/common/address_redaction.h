// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace Common::Log {

// A hand-written single-pass scanner for IPv4, IPv6 and MAC addresses in log lines: two
// std::regex passes on every line cost far more than the scan, and the regexes also hit driver
// and Vulkan version strings such as 1.3.274.1. Callers pre-check that the line contains at
// least one of '.', ':' or '-'.

namespace AddressRedaction {

constexpr bool IsDigit(char c) {
    return c >= '0' && c <= '9';
}

constexpr bool IsHexDigit(char c) {
    return IsDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

constexpr bool IsLetter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

/// A word such as "version" or a standalone "v" inside the `window` before an address: a
/// dotted quad next to one of these is a version, not an address.
inline bool VersionWordNearby(std::string_view window) {
    if (window.empty()) {
        return false;
    }
    constexpr std::string_view words[] = {"version", "driver", "vulkan", "api"};
    const auto word_at = [&](std::size_t at, std::string_view word) {
        if (at + word.size() > window.size()) {
            return false;
        }
        for (std::size_t i = 0; i < word.size(); ++i) {
            const char a = window[at + i], b = word[i];
            if (a != b && (a < 'A' || a > 'Z' || a - 'A' + 'a' != b)) {
                return false;
            }
        }
        const bool left = at == 0 || !(IsLetter(window[at - 1]) || IsDigit(window[at - 1]));
        const std::size_t end = at + word.size();
        const bool right = end == window.size() || !(IsLetter(window[end]) || IsDigit(window[end]));
        return left && right;
    };
    const auto contains_word = [&](char first, std::string_view word) {
        for (std::size_t at = window.find(first); at != std::string_view::npos;
             at = window.find(first, at + 1)) {
            if (word_at(at, word)) {
                return true;
            }
        }
        return false;
    };
    for (const auto word : words) {
        if (contains_word(word[0], word) ||
            (word[0] >= 'a' && word[0] <= 'z' && contains_word(word[0] - 'a' + 'A', word))) {
            return true;
        }
    }
    return contains_word('v', "v") || contains_word('V', "V");
}

/// Length of a redactable IPv4 at `at`, or 0: exactly four octets of 0-255 without leading
/// zeroes, delimited from digits, dots and letters on both sides, and not beside a fifth dotted
/// component or a version word ("driver 31.0.15.60" is a driver version).
inline std::size_t MatchIPv4(std::string_view text, std::size_t at) {
    if (at > 0) {
        const char before = text[at - 1];
        if (IsDigit(before) || before == '.' || IsLetter(before)) {
            return 0;
        }
    }
    std::size_t end = at;
    for (int octet = 0; octet < 4; ++octet) {
        if (octet != 0) {
            if (end >= text.size() || text[end] != '.') {
                return 0;
            }
            ++end;
        }
        std::size_t digits = 0;
        while (end < text.size() && digits < 3 && IsDigit(text[end])) {
            ++end;
            ++digits;
        }
        if (digits == 0) {
            return 0;
        }
        const std::size_t octet_begin = end - digits;
        if (digits > 1 && text[octet_begin] == '0') {
            return 0; // no leading zeroes beyond a lone 0
        }
        int value = 0;
        for (std::size_t d = 0; d < digits; ++d) {
            value = value * 10 + (text[octet_begin + d] - '0');
        }
        if (value > 255) {
            return 0;
        }
    }
    if (end < text.size()) {
        const char after = text[end];
        if (IsDigit(after) || after == '.' || IsLetter(after)) {
            return 0;
        }
    }
    if (VersionWordNearby(text.substr(at > 12 ? at - 12 : 0, at > 12 ? 12 : at))) {
        return 0;
    }
    return end - at;
}

/// Length of a redactable MAC at `at`, or 0: six pairs of two hex digits with one consistent
/// ':' or '-' separator, delimited from hex digits and the separator on both sides.
inline std::size_t MatchMAC(std::string_view text, std::size_t at, char separator) {
    if (separator != ':' && separator != '-') {
        return 0;
    }
    if (at > 0) {
        const char before = text[at - 1];
        if (IsHexDigit(before) || before == separator) {
            return 0;
        }
    }
    std::size_t end = at;
    for (int pair = 0; pair < 6; ++pair) {
        if (pair != 0) {
            if (end >= text.size() || text[end] != separator) {
                return 0;
            }
            ++end;
        }
        if (end + 2 > text.size() || !IsHexDigit(text[end]) || !IsHexDigit(text[end + 1])) {
            return 0;
        }
        end += 2;
    }
    if (end < text.size()) {
        const char after = text[end];
        if (IsHexDigit(after) || after == separator) {
            return 0;
        }
    }
    return end - at;
}

/// Length of a redactable IPv6 run at `at`, or 0: hex groups of 1-4 digits separated by ':',
/// either exactly eight groups, or at least three groups with one "::". The run is delimited
/// from hex digits, ':' and '.' on both sides.
inline std::size_t MatchIPv6(std::string_view text, std::size_t at) {
    if (at > 0) {
        const char before = text[at - 1];
        if (IsHexDigit(before) || before == ':' || before == '.') {
            return 0;
        }
    }
    std::size_t end = at;
    while (end < text.size() && (IsHexDigit(text[end]) || text[end] == ':')) {
        ++end;
    }
    if (end < text.size()) {
        const char after = text[end];
        if (IsHexDigit(after) || after == ':' || after == '.') {
            return 0;
        }
    }
    // Walk the groups of the run: 1-4 hex digits each, at most one "::".
    int groups = 0;
    bool double_colon = false;
    std::size_t i = at;
    bool group_started = false;
    while (i < end) {
        const char c = text[i];
        if (IsHexDigit(c)) {
            std::size_t digits = 0;
            while (i < end && IsHexDigit(text[i])) {
                ++i;
                ++digits;
            }
            if (digits > 4) {
                return 0;
            }
            ++groups;
            group_started = true;
        } else {
            std::size_t colons = 0;
            while (i < end && text[i] == ':') {
                ++i;
                ++colons;
            }
            if (colons > 2) {
                return 0;
            }
            if (colons == 2) {
                if (double_colon) {
                    return 0;
                }
                double_colon = true;
                group_started = false;
            } else if (i < end && !group_started) {
                return 0; // a single colon with no group before it, not part of "::"
            }
        }
    }
    if (groups == 0) {
        return 0;
    }
    if (double_colon ? groups < 3 : groups != 8) {
        return 0;
    }
    return end - at;
}

/// Replaces every redactable IPv4, IPv6 and MAC address in the text with "<ip>" or "<mac>".
inline std::string RedactAddresses(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (IsDigit(c)) {
            if (const std::size_t ipv4 = MatchIPv4(text, i); ipv4 != 0) {
                out += "<ip>";
                i += ipv4;
                continue;
            }
        }
        if (IsHexDigit(c) || c == ':') {
            if (const std::size_t ipv6 = MatchIPv6(text, i); ipv6 != 0) {
                out += "<ip>";
                i += ipv6;
                continue;
            }
        }
        if (IsHexDigit(c)) {
            if (const std::size_t mac = MatchMAC(text, i, ':'); mac != 0) {
                out += "<mac>";
                i += mac;
                continue;
            }
            if (const std::size_t mac = MatchMAC(text, i, '-'); mac != 0) {
                out += "<mac>";
                i += mac;
                continue;
            }
        }
        out += c;
        ++i;
    }
    return out;
}

} // namespace AddressRedaction

} // namespace Common::Log
