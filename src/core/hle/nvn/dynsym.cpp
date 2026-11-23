// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "common/elf.h"
#include "core/arm/symbols.h"
#include "core/hle/nvn/dynsym.h"

namespace Core::NvnHle {
namespace {

constexpr u32 ModMagic = 0x30444F4D; // "MOD0"
constexpr u64 SymValueOffset = offsetof(Common::ELF::Elf64_Sym, st_value);
constexpr u64 SymSize = sizeof(Common::ELF::Elf64_Sym);

template <typename T>
bool ReadAt(std::span<const u8> image, u64 offset, T& out) {
    if (offset > image.size() || image.size() - offset < sizeof(T)) {
        return false;
    }
    std::memcpy(&out, image.data() + offset, sizeof(T));
    return true;
}

} // namespace

bool ProbeDynsym(std::span<const u8> image, DynsymLayout& layout) {
    u32 mod_offset{};
    if (!ReadAt(image, 4, mod_offset)) {
        return false;
    }
    u32 magic{};
    u32 dynamic_offset{};
    if (!ReadAt(image, u64{mod_offset}, magic) || magic != ModMagic ||
        !ReadAt(image, u64{mod_offset} + 4, dynamic_offset)) {
        return false;
    }

    layout = {};
    layout.entry_size = SymSize;
    u64 cursor = u64{mod_offset} + dynamic_offset;
    // The dynamic section is terminated by DT_NULL; cap the walk so a corrupt image cannot spin.
    for (size_t i = 0; i < 4096; ++i, cursor += sizeof(Common::ELF::Elf64_Dyn)) {
        Common::ELF::Elf64_Dyn dyn{};
        if (!ReadAt(image, cursor, dyn)) {
            return false;
        }
        if (dyn.d_tag == Common::ELF::ElfDtNull) {
            break;
        }
        if (dyn.d_tag == Common::ELF::ElfDtStrtab) {
            layout.strtab = dyn.d_un.d_ptr;
        } else if (dyn.d_tag == Common::ELF::ElfDtSymtab) {
            layout.symtab = dyn.d_un.d_ptr;
        } else if (dyn.d_tag == Common::ELF::ElfDtSyment) {
            layout.entry_size = dyn.d_un.d_val;
        }
    }
    return layout.symtab != 0 && layout.strtab > layout.symtab &&
           layout.entry_size >= SymSize && layout.strtab <= image.size();
}

std::vector<DynsymHit> FindDynsymHits(std::span<const u8> image, bool (*wanted)(std::string_view)) {
    std::vector<DynsymHit> hits;
    DynsymLayout layout;
    if (!ProbeDynsym(image, layout)) {
        return hits;
    }
    // Same result as Core::Symbols::GetSymbols (name-sorted, last entry wins for a repeated name),
    // but every symbol and string read stays inside the image the probe accepted.
    std::map<std::string, std::pair<u64, u64>, std::less<>> symbols;
    for (u64 entry = layout.symtab; entry + SymSize <= layout.strtab; entry += layout.entry_size) {
        Common::ELF::Elf64_Sym symbol{};
        if (!ReadAt(image, entry, symbol)) {
            break;
        }
        const u64 name_offset = layout.strtab + symbol.st_name;
        if (name_offset >= image.size()) {
            continue;
        }
        const char* begin = reinterpret_cast<const char*>(image.data() + name_offset);
        const size_t max_length = image.size() - name_offset;
        const size_t length = strnlen(begin, max_length);
        if (length == max_length) {
            continue; // Unterminated name.
        }
        symbols[std::string{begin, length}] = {symbol.st_value, symbol.st_size};
    }
    for (const auto& [name, location] : symbols) {
        if (location.first != 0 && wanted(name)) {
            hits.push_back({name, location.first, location.second});
        }
    }
    return hits;
}

size_t RewriteSymbolValues(std::span<u8> image,
                           const std::map<std::string, u64, std::less<>>& new_values) {
    DynsymLayout layout;
    if (!ProbeDynsym(image, layout)) {
        return 0;
    }
    size_t rewritten = 0;
    for (u64 entry = layout.symtab; entry + SymSize <= layout.strtab; entry += layout.entry_size) {
        Common::ELF::Elf64_Sym symbol{};
        if (!ReadAt(image, entry, symbol) || symbol.st_value == 0) {
            continue;
        }
        const u64 name_offset = layout.strtab + symbol.st_name;
        if (name_offset >= image.size()) {
            continue;
        }
        const char* begin = reinterpret_cast<const char*>(image.data() + name_offset);
        const size_t max_length = image.size() - name_offset;
        const std::string_view name{begin, strnlen(begin, max_length)};
        const auto it = new_values.find(name);
        if (it == new_values.end()) {
            continue;
        }
        const u64 value = it->second;
        std::memcpy(image.data() + entry + SymValueOffset, &value, sizeof(value));
        ++rewritten;
    }
    return rewritten;
}

} // namespace Core::NvnHle
