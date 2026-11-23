// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Core::NvnHle {

/// The device-info constants (nvnDeviceGetInteger items) the game's own NVN2 SDK answers.
///
/// The SDK implements nvnDeviceGetInteger in two layers, both compiled to the same shape
/// (identical in Donkey Kong Bananza 1.0.0 and 3.2.0 and in Mario Kart World 1.8.0):
///   * an outer function `(item, device)` that bounds-checks `item` against the last NVN2 item,
///     calls the inner function with a stack slot (zeroed beforehand) and then jumps through a
///     byte table to one block per item. A block is a constant (`mov w0,#imm; b exit`), the
///     preloaded API major version (`b exit` with `mov w0,#imm` before the jump), a field of the
///     device object (not a constant), or the stack slot (the inner function's answer);
///   * an inner function `(item, &slot)` that is the device-independent table, with the same
///     jump-table shape (`mov wV,#imm; b exit`, preloaded default, `str wV,[x1]` at the exit).
/// Which constants a driver answers is a property of its generation (MKW's SDK 21.4 answers API
/// 231.39 and GPU code 201.20, DK 3.2.0 231.30 / 201.17, DK 1.0.0 231.16 / 201.11, and the games
/// check them), so the HLE takes them from the loaded game's SDK instead of one fixed table.
/// Constant case bodies are read from the machine code; nothing is guessed from the build id.
class SdkDeviceInfo {
public:
    /// Reads the switches out of a module image (module-relative addresses = offsets). nullopt
    /// when the image does not contain exactly one NVN2 DeviceGetInteger of the shape above.
    [[nodiscard]] static std::optional<SdkDeviceInfo> Read(std::span<const u8> image);

    /// The SDK's answer for `item`; nullopt when it is not a constant (read from the device
    /// object at run time). Items past what the SDK's two switches cover answer 0, as the SDK
    /// does.
    [[nodiscard]] std::optional<s32> Get(u32 item) const;

    /// Number of items answered with a constant (log/test).
    [[nodiscard]] u32 ConstantCount() const {
        return constants;
    }
    /// Module-relative address of the outer switch's indirect branch (log).
    [[nodiscard]] u64 SwitchAddress() const {
        return outer_switch;
    }

private:
    struct Entry {
        bool known{};
        s32 value{};
    };
    /// Items 0..entries.size()-1; every item at or past that answers 0.
    std::vector<Entry> entries;
    u32 constants{};
    u64 outer_switch{};
};

} // namespace Core::NvnHle
