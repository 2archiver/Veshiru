// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Core::NvnHle {

/// Bounds-checked location of the dynamic symbol table inside a module image.
struct DynsymLayout {
    u64 symtab{};
    u64 strtab{};
    u64 entry_size{};
};

/// Locates DT_SYMTAB/DT_STRTAB/DT_SYMENT of a module image (MOD0 at u32@4). Every read is checked
/// against the image size; returns false for anything malformed.
[[nodiscard]] bool ProbeDynsym(std::span<const u8> image, DynsymLayout& layout);

/// A defined symbol the bridge wants to redirect. `value` is the module-relative st_value.
struct DynsymHit {
    std::string name;
    u64 value{};
    u64 size{};
};

/// Finds the defined dynsym entries whose name satisfies `wanted`. Walks the table the bounded
/// probe above located, checking every symbol and name read against the image size.
[[nodiscard]] std::vector<DynsymHit> FindDynsymHits(std::span<const u8> image,
                                                    bool (*wanted)(std::string_view));

/// Rewrites st_value of every defined dynsym entry whose name is a key of `new_values`, in place.
/// Returns the number of entries rewritten.
size_t RewriteSymbolValues(std::span<u8> image,
                           const std::map<std::string, u64, std::less<>>& new_values);

} // namespace Core::NvnHle
