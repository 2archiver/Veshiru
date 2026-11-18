// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

inline constexpr char VulkanProbeArgument[] = "--internal-vulkan-probe";

/// Run in an isolated child, without touching config or session logs.
[[nodiscard]] bool ProbeVulkan();
/// Spawn and check the child. False means the frontend should offer a fallback renderer.
[[nodiscard]] bool CheckVulkanInChild(const char* executable);
