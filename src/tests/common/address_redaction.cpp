// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <regex>
#include <string>
#include <string_view>

#include "common/address_redaction.h"

namespace Redact = Common::Log::AddressRedaction;

TEST_CASE("The address scanner redacts IPv4, IPv6 and MAC addresses", "[logging]") {
    const auto redact = [](std::string_view line) {
        return Redact::RedactAddresses(line);
    };

    CHECK(redact("connected to 192.168.1.20 for sync") == "connected to <ip> for sync");
    CHECK(redact("peer 10.0.0.1:443 closed") == "peer <ip>:443 closed");
    CHECK(redact("listening on [fe80::1ff:fe23:4567:890a]:53") ==
          "listening on [<ip>]:53");
    CHECK(redact("adapter 2001:db8:85a3:0:0:8a2e:370:7334 up") == "adapter <ip> up");
    CHECK(redact("mac=AA:BB:CC:DD:EE:FF") == "mac=<mac>");
    CHECK(redact("mac 00-1A-2B-3C-4D-5E paired") == "mac <mac> paired");
}

TEST_CASE("The address scanner leaves versions and identifiers alone", "[logging]") {
    const auto redact = [](std::string_view line) {
        return Redact::RedactAddresses(line);
    };

    // Driver, API and Vulkan versions: dotted quads a version word precedes.
    CHECK(redact("Vulkan 1.3.280.0 driver") == "Vulkan 1.3.280.0 driver");
    CHECK(redact("driver 31.0.15.60 installed") == "driver 31.0.15.60 installed");
    CHECK(redact("API version 332.0.0.0") == "API version 332.0.0.0");
    // Invalid octets and five-component quads.
    CHECK(redact("1.3.274.1") == "1.3.274.1");
    CHECK(redact("999.1.1.1") == "999.1.1.1");
    CHECK(redact("1.2.3.4.5") == "1.2.3.4.5");
    CHECK(redact("v2.3.0") == "v2.3.0");
    CHECK(redact("v 1.2.3.4") == "v 1.2.3.4");
    // Timestamps and title IDs.
    CHECK(redact("12:34:56.789 frame") == "12:34:56.789 frame");
    CHECK(redact("12:34:56") == "12:34:56");
    CHECK(redact("0100ABCD00000000 loaded") == "0100ABCD00000000 loaded");
    // A two-group IPv6 with :: is too short to be sure of.
    CHECK(redact("fe80::1") == "fe80::1");
}

TEST_CASE("The address scanner stays quiet on text without address separators",
          "[logging]") {
    CHECK(Redact::RedactAddresses("no addresses here").find("<ip>") == std::string::npos);
    CHECK(Redact::RedactAddresses("Game 0100ABCD00000000 started").find("<") ==
          std::string::npos);
}

TEST_CASE("The address scanner is cheaper than the regexes it replaced", "[logging]") {
    // The implementation 2.2.0 shipped: two std::regex passes over every line.
    const auto old_redact = [](std::string_view line) {
        static const std::regex ipv4_address{R"(\b(?:\d{1,3}\.){3}\d{1,3}\b)"};
        static const std::regex mac_address{R"(\b(?:[[:xdigit:]]{2}[:-]){5}[[:xdigit:]]{2}\b)"};
        std::string text{line};
        text = std::regex_replace(text, ipv4_address, "<ip>");
        text = std::regex_replace(text, mac_address, "<mac>");
        return text;
    };

    const std::string typical[] = {
        "12:34:56.789 [   0.00] HW_GPU command buffer 0x12345678 submitted with 8192 bytes",
        "frontend: PatchManager initiated for id 100ABCD00000000 with update 1.8.0",
        "Vulkan 1.3.280 driver 31.0.15.60 on NVIDIA GeForce GTX 1650 (0000:01:00.0)",
        "core/hle/service/nvdrv/devices/nvhost_ctrl.cpp:42 syncpoint 0x1f waited 3347 us",
        "shader 0x0000ABCD1234EF01 stage Vertex translated in 47210 ns (6543 instructions)",
        "network interface eth0 mtu 1500 addr 192.168.1.20 mac 00:1A:2B:3C:4D:5E up",
        "audio_core: sysmix mix 7 rendered 1024 frames at 48000 Hz in 210 us",
        "no separator characters in this line at all",
    };

    constexpr int total_lines = 100000;
    std::string sink;

    const auto old_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < total_lines; ++i) {
        sink = old_redact(typical[i % std::size(typical)]);
    }
    const auto old_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - old_begin)
                            .count();

    const auto new_begin = std::chrono::steady_clock::now();
    int redacted_lines = 0;
    for (int i = 0; i < total_lines; ++i) {
        sink = Redact::RedactAddresses(typical[i % std::size(typical)]);
        redacted_lines += sink.find('<') != std::string::npos ? 1 : 0;
    }
    const auto new_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - new_begin)
                            .count();

    std::printf("[benchmark] address redaction, %d lines: regex %lld ms, scanner %lld ms\n",
                total_lines, static_cast<long long>(old_ms), static_cast<long long>(new_ms));
    // Reported, not asserted: CI machine timing noise must not fail a build. The scanner is
    // expected to be several times faster; the printf above carries the numbers.
    CHECK(redacted_lines > 0);
}
