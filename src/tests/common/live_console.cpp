// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <catch2/catch_test_macros.hpp>

#include "common/logging.h"

using namespace Common::Log;

TEST_CASE("Live console groups existing log classes into subsystems", "[live_console]") {
    CHECK(GetSubsystem(Class::Core_ARM) == Subsystem::CPU);
    CHECK(GetSubsystem(Class::Render_Vulkan) == Subsystem::GPU);
    CHECK(GetSubsystem(Class::Shader_SPIRV) == Subsystem::GPU);
    CHECK(GetSubsystem(Class::Service_AM) == Subsystem::HLE);
    CHECK(GetSubsystem(Class::Service_FS) == Subsystem::FS);
    CHECK(GetSubsystem(Class::Common_Filesystem) == Subsystem::FS);
    CHECK(GetSubsystem(Class::Audio_Sink) == Subsystem::Audio);
    CHECK(GetSubsystem(Class::Network) == Subsystem::Networking);
    CHECK(GetSubsystem(Class::Performance) == Subsystem::Performance);
    CHECK(GetSubsystem(Class::Compatibility) == Subsystem::Compatibility);
    CHECK(GetSubsystem(Class::Frontend) == Subsystem::Other);
    CHECK(GetSubsystemName(Subsystem::Networking) == "Networking");
    CHECK(GetLogClassName(Class::Render_Vulkan) == "Render.Vulkan");
    CHECK(GetLogLevelName(Level::Critical) == "Critical");
}

TEST_CASE("Console sanitisation removes personal paths and credential-like values",
          "[live_console]") {
    const auto windows = SanitizeLogMessage(R"(Loading C:\Users\Alice\Documents\game.nsp)");
    CHECK(windows.find("Alice") == std::string::npos);
    CHECK(windows.find("<path>") != std::string::npos);

    const auto posix = SanitizeLogMessage("Failed to open /home/alice/private/save.bin");
    CHECK(posix.find("alice") == std::string::npos);
    CHECK(posix.find("<path>") != std::string::npos);

    const auto credential = SanitizeLogMessage("request token=abc123 password: hunter2");
    CHECK(credential.find("abc123") == std::string::npos);
    CHECK(credential.find("hunter2") == std::string::npos);
    CHECK(credential.find("<redacted>") != std::string::npos);

    const auto url = SanitizeLogMessage("fetch https://user:secret@example.invalid/path");
    CHECK(url.find("secret") == std::string::npos);
    CHECK(url.find("<redacted>@") != std::string::npos);

    // Relative source locations and title IDs remain useful diagnostics.
    const auto relative = SanitizeLogMessage("core/loader/loader.cpp: 0100ABCD00000000");
    CHECK(relative.find("core/loader/loader.cpp") != std::string::npos);
    CHECK(relative.find("0100ABCD00000000") != std::string::npos);
}
