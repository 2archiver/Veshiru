// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2018 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "discord.h"

#include <cstdint>
#include <string>

namespace Core {
class System;
}

namespace DiscordRPC {

class DiscordImpl final : public DiscordInterface {
public:
    DiscordImpl(Core::System& system_, std::string application_id);
    ~DiscordImpl() override;

    void Pause() override;
    void Update() override;

private:
    void SetBrowsing();

    Core::System& system;
    std::string application_id;
    std::string current_game_key;
    std::int64_t game_started_at{};
};

} // namespace DiscordRPC
