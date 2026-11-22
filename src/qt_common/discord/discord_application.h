// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace DiscordRPC {

/// The Discord application ("client") ID Veshiru shakes hands with. Users never create or paste
/// one: presence works out of the box, and nothing private is sent (only the game title, its
/// version and an elapsed-time timestamp).
///
/// NOTE(2.5.0): presence ships with the Eden-lineage community application ID, so Discord labels
/// the activity with that application's registered name; the activity itself still reads
/// "Playing <game>", "Veshiru <version>" and shows the Veshiru icon. No client-side payload can
/// rename an application, so the only complete fix is a Veshiru-owned application: create one at
/// https://discord.com/developers/applications, copy its Application ID into this constant, and
/// update the cleanup key in dist/installer.nsi to match. This constant is the only place to
/// change it.
inline constexpr char VESHIRU_DISCORD_APPLICATION_ID[] = "1397286652128264252";

} // namespace DiscordRPC
