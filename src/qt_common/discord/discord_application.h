// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace DiscordRPC {

/// The Discord application ("client") ID Veshiru shakes hands with. Users never create or paste
/// one: presence works out of the box, and nothing private is sent (only the game title, its
/// version and an elapsed-time timestamp).
///
/// TODO(release): replace this with a Veshiru-owned application ID from
/// https://discord.com/developers/applications before a public release. This value is the
/// Eden-lineage community application ID the Windows uninstaller already cleans up
/// (`discord-1397286652128264252`); it keeps the feature working until a Veshiru-owned one
/// exists. This constant is the only place to change it.
inline constexpr char VESHIRU_DISCORD_APPLICATION_ID[] = "1397286652128264252";

} // namespace DiscordRPC
