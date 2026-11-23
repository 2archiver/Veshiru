// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "core/hle/nvn/sdk_impl_table.h"

namespace Core::NvnHle {
namespace {

constexpr u32 Ret = 0xd65f03c0;

/// The two words of a function at module offset `address` are `first; ret`.
bool IsAccessor(std::span<const u8> image, u64 address, u32 first) {
    return address % 4 == 0 && image.size() >= 8 && address <= image.size() - 8 &&
           SdkImageWord(image, address) == first && SdkImageWord(image, address + 4) == Ret;
}

std::optional<size_t> IndexOf(std::span<const std::string> names, std::string_view name) {
    const auto it = std::lower_bound(names.begin(), names.end(), name,
                                     [](const std::string& a, std::string_view b) { return a < b; });
    if (it == names.end() || *it != name) {
        return std::nullopt;
    }
    return static_cast<size_t>(it - names.begin());
}

} // namespace

u32 SdkImageWord(std::span<const u8> image, u64 offset) {
    u32 word{};
    std::memcpy(&word, image.data() + offset, sizeof(word));
    return word;
}

u64 SdkImageU64(std::span<const u8> image, u64 offset) {
    u64 value{};
    std::memcpy(&value, image.data() + offset, sizeof(value));
    return value;
}

std::vector<u64> FindSdkImplTables(std::span<const u8> image,
                                   std::span<const std::string> proc_names) {
    // The names are sorted (the SDK binary-searches them); the implementation table is indexed
    // the same way.
    if (!std::ranges::is_sorted(proc_names)) {
        return {};
    }
    const std::optional<size_t> get_device = IndexOf(proc_names, "nvnDlssNetworkBuilderGetDevice");
    const std::optional<size_t> set_device = IndexOf(proc_names, "nvnDlssNetworkBuilderSetDevice");
    const std::optional<size_t> get_flags = IndexOf(proc_names, "nvnDlssNetworkBuilderGetFlags");
    const std::optional<size_t> set_flags = IndexOf(proc_names, "nvnDlssNetworkBuilderSetFlags");
    if (!get_device || !set_device || !get_flags || !set_flags) {
        return {};
    }
    // ldr x0, [x0] / str x1, [x0] / ldr w0, [x0, #0x20] / str w1, [x0, #0x20], each then `ret`.
    constexpr u32 LdrX0 = 0xf9400000;
    constexpr u32 StrX1 = 0xf9000001;
    constexpr u32 LdrW0Flags = 0xb9402000;
    constexpr u32 StrW1Flags = 0xb9002001;

    const u64 table_bytes = u64{proc_names.size()} * 8;
    std::vector<u64> tables;
    for (u64 base = 0; base + table_bytes <= image.size(); base += 8) {
        const auto entry = [&](size_t i) { return SdkImageU64(image, base + 8 * i); };
        if (IsAccessor(image, entry(*get_device), LdrX0) &&
            IsAccessor(image, entry(*set_device), StrX1) &&
            IsAccessor(image, entry(*get_flags), LdrW0Flags) &&
            IsAccessor(image, entry(*set_flags), StrW1Flags)) {
            tables.push_back(base);
        }
    }
    return tables;
}

std::optional<u64> SdkImplOffset(std::span<const u8> image,
                                 std::span<const std::string> proc_names,
                                 std::span<const u64> tables, std::string_view name) {
    const std::optional<size_t> index = IndexOf(proc_names, name);
    if (!index || tables.empty()) {
        return std::nullopt;
    }
    std::optional<u64> found;
    for (const u64 table : tables) {
        const u64 slot = table + 8 * u64{*index};
        if (slot + 8 > image.size()) {
            return std::nullopt;
        }
        const u64 address = SdkImageU64(image, slot);
        if (address % 4 != 0 || image.size() < 4 || address > image.size() - 4 ||
            (found && *found != address)) {
            return std::nullopt;
        }
        found = address;
    }
    return found;
}

bool SdkTextureHandleNullForTextureZero(std::span<const u8> image, u64 address) {
    constexpr u32 MaxInstructions = 8;
    constexpr u32 CmpW1Zero = 0x7100003f; // subs wzr, w1, #0
    if (address % 4 != 0) {
        return false;
    }
    bool compared = false;
    bool selects_null = false;
    for (u32 i = 0; i < MaxInstructions && address + 4 * (i + 1) <= image.size(); ++i) {
        const u32 w = SdkImageWord(image, address + 4 * i);
        if (w == Ret) {
            return compared && selects_null;
        }
        if (w == CmpW1Zero) {
            compared = true;
        }
        // csel x0, xzr, xM, eq  /  csel x0, xN, xzr, ne, consuming the flags of that compare
        if (compared && ((w & 0xffe0ffff) == 0x9a8003e0 || (w & 0xfffffc1f) == 0x9a9f1000)) {
            selects_null = true;
        }
    }
    return false;
}

} // namespace Core::NvnHle
