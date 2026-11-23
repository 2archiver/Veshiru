// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <map>

#include "core/hle/nvn/sdk_device_info.h"
#include "video_core/nvn/nvn_types.h"

namespace Core::NvnHle {
namespace {

// ---- A64 decoders (only the forms the SDK's switches use) -------------------------------------

constexpr u32 RetWord = 0xD65F03C0;

constexpr bool IsBr(u32 w) {
    return (w & 0xFFFFFC1F) == 0xD61F0000;
}
constexpr u32 Rd(u32 w) {
    return w & 31;
}
constexpr u32 Rn(u32 w) {
    return (w >> 5) & 31;
}
constexpr u32 Rm(u32 w) {
    return (w >> 16) & 31;
}
constexpr s64 SignExtend(u64 value, u32 bits) {
    const u64 sign = u64{1} << (bits - 1);
    return static_cast<s64>((value ^ sign) - sign);
}
constexpr bool IsB(u32 w) {
    return (w & 0xFC000000) == 0x14000000;
}
constexpr bool IsBl(u32 w) {
    return (w & 0xFC000000) == 0x94000000;
}
constexpr u64 BranchTarget(u64 pc, u32 w) {
    return pc + static_cast<u64>(SignExtend(w & 0x3FFFFFF, 26) * 4);
}
/// b.hi
constexpr bool IsBhi(u32 w) {
    return (w & 0xFF00001F) == 0x54000008;
}
constexpr u64 ConditionalTarget(u64 pc, u32 w) {
    return pc + static_cast<u64>(SignExtend((w >> 5) & 0x7FFFF, 19) * 4);
}
/// add xd, xn, xm, lsl #2
constexpr bool IsAddLsl2(u32 w) {
    return (w & 0xFF200000) == 0x8B000000 && ((w >> 22) & 3) == 0 && ((w >> 10) & 63) == 2;
}
/// ldrb wt, [xn, xm] (no extend, no shift)
constexpr bool IsLdrbRegister(u32 w) {
    return (w & 0xFFE00C00) == 0x38600800 && ((w >> 13) & 7) == 3 && ((w >> 12) & 1) == 0;
}
constexpr bool IsAdr(u32 w) {
    return (w & 0x9F000000) == 0x10000000;
}
constexpr bool IsAdrp(u32 w) {
    return (w & 0x9F000000) == 0x90000000;
}
constexpr s64 AdrImmediate(u32 w) {
    return SignExtend((((w >> 5) & 0x7FFFF) << 2) | ((w >> 29) & 3), 21);
}
/// add xd, xn, #imm
constexpr bool IsAddImmediate64(u32 w) {
    return (w & 0xFF800000) == 0x91000000;
}
/// sub xd, xn, #imm
constexpr bool IsSubImmediate64(u32 w) {
    return (w & 0xFF800000) == 0xD1000000;
}
/// sub wd, wn, #imm
constexpr bool IsSubImmediate32(u32 w) {
    return (w & 0xFF800000) == 0x51000000;
}
constexpr u32 ArithmeticImmediate(u32 w) {
    const u32 imm = (w >> 10) & 0xFFF;
    return ((w >> 22) & 1) != 0 ? imm << 12 : imm;
}
/// cmp wn, #imm (subs wzr, wn, #imm)
constexpr bool IsCmpImmediate32(u32 w) {
    return (w & 0xFF80001F) == 0x7100001F;
}
/// mov wd, wm (orr wd, wzr, wm; wm = wzr gives `mov wd, wzr`)
constexpr bool IsMovRegister32(u32 w) {
    return (w & 0xFFE0FFE0) == 0x2A0003E0;
}
/// ldur wt, [xn, #simm9]
constexpr bool IsLdur32(u32 w) {
    return (w & 0xFFE00C00) == 0xB8400000;
}
/// str wt, [xn] (unsigned offset 0)
constexpr bool IsStr32ZeroOffset(u32 w) {
    return (w & 0xFFFFFC00) == 0xB9000000;
}

/// The value a 32-bit `mov wd, #imm` (movz, movn or the bitmask form of orr) writes to wd.
std::optional<std::pair<u32, u32>> DecodeMoveImmediate32(u32 w) {
    // The 32-bit forms only allow hw 0 or 1 (shift 0 or 16); hw 2 and 3 are reserved.
    if ((w & 0xFF800000) == 0x52800000) { // movz
        if (((w >> 21) & 3) > 1) {
            return std::nullopt;
        }
        return std::pair{Rd(w), ((w >> 5) & 0xFFFF) << (((w >> 21) & 1) * 16)};
    }
    if ((w & 0xFF800000) == 0x12800000) { // movn
        if (((w >> 21) & 3) > 1) {
            return std::nullopt;
        }
        return std::pair{Rd(w), ~(((w >> 5) & 0xFFFF) << (((w >> 21) & 1) * 16))};
    }
    if ((w & 0xFF800000) == 0x32000000 && Rn(w) == 31 && ((w >> 22) & 1) == 0) { // orr wd, wzr, #imm
        const u32 immr = (w >> 16) & 63;
        const u32 imms = (w >> 10) & 63;
        const u32 pattern = ~imms & 0x3F;
        if (pattern == 0) {
            return std::nullopt;
        }
        const u32 length = static_cast<u32>(std::bit_width(pattern)) - 1;
        const u32 size = 1u << length;
        if (size > 32) {
            return std::nullopt;
        }
        const u32 levels = size - 1;
        const u32 ones = imms & levels;
        const u32 rotate = immr & levels;
        if (ones == levels) {
            return std::nullopt;
        }
        const u64 element_mask = (u64{1} << size) - 1;
        const u64 element = (u64{1} << (ones + 1)) - 1;
        const u64 rotated = ((element >> rotate) | (element << (size - rotate))) & element_mask;
        u32 value = 0;
        for (u32 shift = 0; shift < 32; shift += size) {
            value |= static_cast<u32>(rotated << shift);
        }
        return std::pair{Rd(w), value};
    }
    return std::nullopt;
}

// ---- Image access -------------------------------------------------------------------------------

class Words {
public:
    explicit Words(std::span<const u8> image_) : image{image_} {}

    [[nodiscard]] std::optional<u32> At(u64 address) const {
        if ((address & 3) != 0 || image.size() < 4 || address > image.size() - 4) {
            return std::nullopt;
        }
        u32 word;
        std::memcpy(&word, image.data() + address, sizeof(word));
        return word;
    }
    [[nodiscard]] std::optional<u8> Byte(u64 address) const {
        if (address >= image.size()) {
            return std::nullopt;
        }
        return image[address];
    }
    [[nodiscard]] u64 Size() const {
        return image.size();
    }

private:
    std::span<const u8> image;
};

// ---- Jump-table switch --------------------------------------------------------------------------

/// Instructions inspected before an indirect branch to find the switch around it. The compiler
/// interleaves the table setup, the preloaded values and the bound check differently per SDK
/// build (the three SDKs checked differ in their order), so each part is looked up in the window
/// rather than at a fixed distance.
constexpr u64 WindowInstructions = 24;

struct Switch {
    u64 branch{};         ///< address of the `br`
    u64 table{};          ///< byte table: index -> (target - base) / 4
    u64 base{};           ///< address the table offsets are relative to
    u32 bound{};          ///< largest valid index
    u32 index_offset{};   ///< item = index + index_offset
    u64 default_target{}; ///< `b.hi` destination of the bound check
    u64 call_target{};    ///< `bl` before the bound check (0 when none)
    std::array<std::optional<u32>, 32> preloaded{}; ///< `mov wN, #imm` between check and branch
    u32 stack_slot_register_offset{};               ///< `sub x1, x29, #n` before the call (0 when none)
};

std::optional<Switch> ParseSwitch(const Words& words, u64 branch) {
    const auto branch_word = words.At(branch);
    if (!branch_word || !IsBr(*branch_word) || branch < WindowInstructions * 4) {
        return std::nullopt;
    }
    struct Op {
        u64 address;
        u32 word;
    };
    std::array<Op, WindowInstructions> ops;
    for (u64 i = 0; i < WindowInstructions; ++i) {
        const u64 address = branch - (WindowInstructions - i) * 4;
        ops[i] = {address, *words.At(address)};
    }
    // The nearest add xN, xB, xA, lsl #2 before the branch, then the ldrb that loads xA and the
    // adr that sets xB.
    const auto find_last = [&](auto&& match) -> const Op* {
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            if (match(*it)) {
                return &*it;
            }
        }
        return nullptr;
    };
    const u32 branch_register = Rn(*branch_word);
    const Op* add = find_last([&](const Op& op) { return IsAddLsl2(op.word) && Rd(op.word) == branch_register; });
    if (add == nullptr) {
        return std::nullopt;
    }
    const u32 base_register = Rn(add->word);
    const u32 offset_register = Rm(add->word);
    const Op* ldrb = find_last([&](const Op& op) { return IsLdrbRegister(op.word) && Rd(op.word) == offset_register; });
    const Op* adr = find_last([&](const Op& op) { return IsAdr(op.word) && Rd(op.word) == base_register; });
    if (ldrb == nullptr || adr == nullptr) {
        return std::nullopt;
    }
    const u32 table_register = Rn(ldrb->word);
    const u32 index_register = Rm(ldrb->word);

    Switch result;
    result.branch = branch;
    result.base = adr->address + static_cast<u64>(AdrImmediate(adr->word));

    // adrp xT, page ; add xT, xT, #lo
    bool have_table = false;
    for (size_t i = 0; i < ops.size() && !have_table; ++i) {
        if (!IsAdrp(ops[i].word) || Rd(ops[i].word) != table_register) {
            continue;
        }
        for (size_t j = i + 1; j < ops.size(); ++j) {
            const u32 w = ops[j].word;
            if (IsAddImmediate64(w) && Rd(w) == table_register && Rn(w) == table_register) {
                result.table = (ops[i].address & ~u64{0xFFF}) +
                               static_cast<u64>(AdrImmediate(ops[i].word) * 4096) +
                               ArithmeticImmediate(w);
                have_table = true;
                break;
            }
        }
    }
    if (!have_table) {
        return std::nullopt;
    }

    // cmp wX, #bound ; b.hi default
    size_t cmp_index = ops.size();
    u32 cmp_register = 0;
    for (size_t i = 0; i + 1 < ops.size(); ++i) {
        if (IsCmpImmediate32(ops[i].word) && IsBhi(ops[i + 1].word)) {
            cmp_index = i;
            cmp_register = Rn(ops[i].word);
            result.bound = ArithmeticImmediate(ops[i].word);
            result.default_target = ConditionalTarget(ops[i + 1].address, ops[i + 1].word);
        }
    }
    if (cmp_index == ops.size()) {
        return std::nullopt;
    }
    // The table is indexed by the checked register itself, a copy of it (mov w8, w20), and the
    // checked register is `item - k` (sub w9, w8, #k) or the item.
    if (index_register != cmp_register) {
        const bool copy = std::ranges::any_of(ops, [&](const Op& op) {
            return IsMovRegister32(op.word) && Rd(op.word) == index_register &&
                   Rm(op.word) == cmp_register;
        });
        if (!copy) {
            return std::nullopt;
        }
    }
    for (size_t i = 0; i < cmp_index; ++i) {
        if (IsSubImmediate32(ops[i].word) && Rd(ops[i].word) == cmp_register) {
            result.index_offset = ArithmeticImmediate(ops[i].word);
        }
    }
    for (size_t i = cmp_index + 1; i < ops.size(); ++i) {
        if (const auto move = DecodeMoveImmediate32(ops[i].word)) {
            result.preloaded[move->first] = move->second;
        }
    }
    for (size_t i = 0; i < cmp_index; ++i) {
        if (IsBl(ops[i].word)) {
            result.call_target = BranchTarget(ops[i].address, ops[i].word);
        }
        // sub x1, x29, #n: the out-pointer handed to the inner function
        if (IsSubImmediate64(ops[i].word) && Rd(ops[i].word) == 1 && Rn(ops[i].word) == 29) {
            result.stack_slot_register_offset = ArithmeticImmediate(ops[i].word);
        }
    }

    // The byte table and every target it names must be inside the image.
    for (u32 index = 0; index <= result.bound; ++index) {
        const auto entry = words.Byte(result.table + index);
        if (!entry || !words.At(result.base + u64{*entry} * 4)) {
            return std::nullopt;
        }
    }
    return result;
}

enum class Source : u8 {
    Constant,
    Device, ///< a field of the device object, not a constant
    Inner,  ///< the stack slot the inner function filled
};

struct CaseValue {
    Source source{Source::Device};
    s32 value{};
};

/// A case body: `exit` itself, or `[mov wR, #imm]` followed by `b exit` / the exit block.
CaseValue ClassifyBody(const Words& words, u64 target, u32 value_register,
                       const std::array<std::optional<u32>, 32>& preloaded, u64 exit) {
    const auto preset = [&]() -> CaseValue {
        if (const auto& value = preloaded[value_register]) {
            return {Source::Constant, static_cast<s32>(*value)};
        }
        return {Source::Device, 0};
    };
    if (target == exit) {
        return preset();
    }
    const auto first = words.At(target);
    if (!first) {
        return {};
    }
    if (IsB(*first)) {
        return BranchTarget(target, *first) == exit ? preset() : CaseValue{};
    }
    std::optional<u32> value;
    if (const auto move = DecodeMoveImmediate32(*first); move && move->first == value_register) {
        value = move->second;
    } else if (IsMovRegister32(*first) && Rd(*first) == value_register && Rm(*first) == 31) {
        value = 0;
    }
    if (!value) {
        return {};
    }
    const u64 next = target + 4;
    const auto second = words.At(next);
    if (next == exit || (second && IsB(*second) && BranchTarget(next, *second) == exit)) {
        return {Source::Constant, static_cast<s32>(*value)};
    }
    return {};
}

struct Inner {
    Switch sw;
    u32 value_register{};
    u64 exit{};
};

/// The inner function: its default (index out of range) is `ret` with the slot left at zero,
/// its exit block is `mov w0, #1; str wV, [x1]` and every case produces wV.
std::optional<Inner> ParseInner(const Words& words, u64 entry) {
    if (entry == 0) {
        return std::nullopt;
    }
    for (u64 address = entry; address < entry + WindowInstructions * 4; address += 4) {
        const auto word = words.At(address);
        if (!word || !IsBr(*word)) {
            continue;
        }
        auto sw = ParseSwitch(words, address);
        if (!sw || words.At(sw->default_target) != RetWord) {
            return std::nullopt;
        }
        // The exit block is where most constant cases jump to.
        std::map<u64, u32> exits;
        for (u32 index = 0; index <= sw->bound; ++index) {
            const u64 target = sw->base + u64{*words.Byte(sw->table + index)} * 4;
            const auto second = words.At(target + 4);
            if (second && IsB(*second)) {
                ++exits[BranchTarget(target + 4, *second)];
            }
        }
        if (exits.empty()) {
            return std::nullopt;
        }
        const u64 exit = std::ranges::max_element(exits, {}, &std::pair<const u64, u32>::second)->first;
        const auto mov_one = words.At(exit);
        const auto store = words.At(exit + 4);
        if (!mov_one || !store || !IsStr32ZeroOffset(*store) || Rn(*store) != 1) {
            return std::nullopt;
        }
        const auto one = DecodeMoveImmediate32(*mov_one);
        if (!one || one->first != 0 || one->second != 1) {
            return std::nullopt;
        }
        return Inner{*sw, Rd(*store), exit};
    }
    return std::nullopt;
}

struct Outer {
    Switch sw;
    u64 exit{};
};

/// The outer function: the default of its bound check is `ldur w0, [x29, #-n]` (the stack slot of
/// the inner call, whose address was formed with `sub x1, x29, #n`) then `b exit`.
std::optional<Outer> ParseOuter(const Words& words, u64 branch) {
    auto sw = ParseSwitch(words, branch);
    if (!sw || sw->call_target == 0 || sw->stack_slot_register_offset == 0 ||
        sw->bound < static_cast<u32>(VideoCommon::Nvn::kDeviceInfoLastNvn1) || sw->bound >= 0x200 ||
        !sw->preloaded[0]) {
        return std::nullopt;
    }
    const auto load = words.At(sw->default_target);
    const auto jump = words.At(sw->default_target + 4);
    if (!load || !jump || !IsLdur32(*load) || Rd(*load) != 0 || Rn(*load) != 29 || !IsB(*jump) ||
        SignExtend((*load >> 12) & 0x1FF, 9) != -static_cast<s64>(sw->stack_slot_register_offset)) {
        return std::nullopt;
    }
    return Outer{*sw, BranchTarget(sw->default_target + 4, *jump)};
}

} // namespace

std::optional<SdkDeviceInfo> SdkDeviceInfo::Read(std::span<const u8> image) {
    const Words words{image};
    std::optional<Outer> outer;
    std::optional<Inner> inner;
    for (u64 address = WindowInstructions * 4; address + 4 <= words.Size(); address += 4) {
        if (!IsBr(*words.At(address))) {
            continue;
        }
        auto candidate = ParseOuter(words, address);
        if (!candidate) {
            continue;
        }
        auto candidate_inner = ParseInner(words, candidate->sw.call_target);
        if (!candidate_inner) {
            continue;
        }
        if (outer) {
            return std::nullopt; // two SDK-shaped functions: do not guess
        }
        outer = candidate;
        inner = candidate_inner;
    }
    if (!outer || !inner) {
        return std::nullopt;
    }

    const u32 inner_first = inner->sw.index_offset;
    const u32 inner_last = inner->sw.index_offset + inner->sw.bound;
    const u32 item_count = std::max(outer->sw.bound, inner_last) + 1;
    SdkDeviceInfo info;
    info.outer_switch = outer->sw.branch;
    info.entries.resize(item_count);

    const auto inner_answer = [&](u32 item) -> Entry {
        if (item < inner_first || item > inner_last) {
            return {true, 0}; // the slot stays zero
        }
        const u32 index = item - inner_first;
        const u64 target = inner->sw.base + u64{*words.Byte(inner->sw.table + index)} * 4;
        const CaseValue body =
            ClassifyBody(words, target, inner->value_register, inner->sw.preloaded, inner->exit);
        return body.source == Source::Constant ? Entry{true, body.value} : Entry{};
    };
    for (u32 item = 0; item < item_count; ++item) {
        Entry entry;
        if (item > outer->sw.bound) {
            entry = inner_answer(item);
        } else {
            const u64 target = outer->sw.base + u64{*words.Byte(outer->sw.table + item)} * 4;
            if (target == outer->sw.default_target) {
                entry = inner_answer(item);
            } else {
                const CaseValue body = ClassifyBody(words, target, 0, outer->sw.preloaded, outer->exit);
                if (body.source == Source::Constant) {
                    entry = {true, body.value};
                }
            }
        }
        info.entries[item] = entry;
        info.constants += entry.known ? 1 : 0;
    }
    // Item 0 is the API major version: every NVN2 SDK answers it with a constant. Anything else
    // means the shape was misread.
    if (!info.entries[0].known) {
        return std::nullopt;
    }
    return info;
}

std::optional<s32> SdkDeviceInfo::Get(u32 item) const {
    if (item >= entries.size()) {
        return 0;
    }
    const Entry& entry = entries[item];
    return entry.known ? std::optional<s32>{entry.value} : std::nullopt;
}

} // namespace Core::NvnHle
