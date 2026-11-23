// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/nvn/sdk_device_info.h"
#include "tests/nvn/hle_rig.h"

// A synthetic SDK image with the two-layer shape of the NVN2 nvnDeviceGetInteger (see
// SdkDeviceInfo): an outer switch that falls back to the stack slot of an inner device-independent
// switch. It is assembled here from the instruction forms the SDK's compiler emits; the real
// images are not part of the repository.

using Core::NvnHle::SdkDeviceInfo;
using namespace NvnHleTest;

namespace {

constexpr u32 Nop = 0xD503201F;
constexpr u32 Ret = 0xD65F03C0;

constexpr u32 MovZ32(u32 rd, u32 imm16, u32 halfword = 0) {
    return 0x52800000 | (halfword << 21) | (imm16 << 5) | rd;
}
constexpr u32 MovN32(u32 rd, u32 imm16) {
    return 0x12800000 | (imm16 << 5) | rd;
}
constexpr u32 MovReg32(u32 rd, u32 rm) {
    return 0x2A0003E0 | (rm << 16) | rd;
}
/// orr wd, wzr, #bitmask
constexpr u32 OrrBitmask32(u32 rd, u32 immr, u32 imms) {
    return 0x32000000 | (immr << 16) | (imms << 10) | (31 << 5) | rd;
}
constexpr u32 SubImm32(u32 rd, u32 rn, u32 imm) {
    return 0x51000000 | (imm << 10) | (rn << 5) | rd;
}
constexpr u32 SubImm64(u32 rd, u32 rn, u32 imm) {
    return 0xD1000000 | (imm << 10) | (rn << 5) | rd;
}
constexpr u32 AddImm32(u32 rd, u32 rn, u32 imm) {
    return 0x11000000 | (imm << 10) | (rn << 5) | rd;
}
constexpr u32 AddImm64(u32 rd, u32 rn, u32 imm) {
    return 0x91000000 | (imm << 10) | (rn << 5) | rd;
}
constexpr u32 CmpImm32(u32 rn, u32 imm) {
    return 0x7100001F | (imm << 10) | (rn << 5);
}
constexpr u32 Bhi(u32 pc, u32 target) {
    return 0x54000008 | ((((target - pc) / 4) & 0x7FFFF) << 5);
}
constexpr u32 B(u32 pc, u32 target) {
    return 0x14000000 | (((target - pc) / 4) & 0x3FFFFFF);
}
constexpr u32 Bl(u32 pc, u32 target) {
    return 0x94000000 | (((target - pc) / 4) & 0x3FFFFFF);
}
constexpr u32 Adr(u32 rd, u32 pc, u32 target) {
    const u32 imm = (target - pc) & 0x1FFFFF;
    return 0x10000000 | ((imm & 3) << 29) | ((imm >> 2) << 5) | rd;
}
constexpr u32 Adrp(u32 rd, u32 pc, u32 target) {
    const u32 imm = ((target >> 12) - (pc >> 12)) & 0x1FFFFF;
    return 0x90000000 | ((imm & 3) << 29) | ((imm >> 2) << 5) | rd;
}
constexpr u32 Ldrb(u32 rt, u32 rn, u32 rm) {
    return 0x38606800 | (rm << 16) | (rn << 5) | rt;
}
constexpr u32 AddLsl2(u32 rd, u32 rn, u32 rm) {
    return 0x8B000800 | (rm << 16) | (rn << 5) | rd;
}
constexpr u32 Br(u32 rn) {
    return 0xD61F0000 | (rn << 5);
}
constexpr u32 Ldur32(u32 rt, u32 rn, s32 offset) {
    return 0xB8400000 | ((static_cast<u32>(offset) & 0x1FF) << 12) | (rn << 5) | rt;
}
/// stur wt, [xn, #offset]
constexpr u32 Stur32(u32 rt, u32 rn, s32 offset) {
    return 0xB8000000 | ((static_cast<u32>(offset) & 0x1FF) << 12) | (rn << 5) | rt;
}
/// ldr wt, [xn, #offset] (unsigned offset)
constexpr u32 Ldr32(u32 rt, u32 rn, u32 offset) {
    return 0xB9400000 | ((offset / 4) << 10) | (rn << 5) | rt;
}
constexpr u32 StrW8X1 = 0xB9000028;

// Layout (module-relative).
constexpr u32 InnerEntry = 0x200;
constexpr u32 InnerCases = 0x230;
constexpr u32 InnerExit = 0x264;
constexpr u32 InnerDefault = 0x26C;
constexpr u32 InnerTable = 0x300;
constexpr u32 OuterTable = 0x340;
constexpr u32 OuterEntry = 0x400;
constexpr u32 OuterCases = 0x438;
constexpr u32 OuterDefault = 0x450;
constexpr u32 OuterExit = 0x458;
constexpr u32 OuterBound = 0x70;
constexpr u32 ImageSize = 0x1000;

struct Image {
    std::vector<u8> bytes = std::vector<u8>(ImageSize);

    void Word(u32 address, u32 word) {
        std::memcpy(bytes.data() + address, &word, sizeof(word));
    }
    void Fill(u32 first, u32 last) {
        for (u32 address = first; address < last; address += 4) {
            Word(address, Nop);
        }
    }
};

Image MakeImage() {
    Image image;
    image.Fill(0, ImageSize);

    // Inner (item, &slot): items 2..10.
    image.Word(0x200, MovReg32(8, 0));
    image.Word(0x204, SubImm32(9, 8, 2));
    image.Word(0x208, MovReg32(0, 31));
    image.Word(0x20C, CmpImm32(9, 8));
    image.Word(0x210, Bhi(0x210, InnerDefault));
    image.Word(0x214, Adrp(10, 0x214, InnerTable));
    image.Word(0x218, AddImm64(10, 10, InnerTable & 0xFFF));
    image.Word(0x21C, MovZ32(8, 0xE));
    image.Word(0x220, Adr(11, 0x220, InnerCases));
    image.Word(0x224, Ldrb(12, 10, 9));
    image.Word(0x228, AddLsl2(11, 11, 12));
    image.Word(0x22C, Br(11));
    image.Word(0x230, MovZ32(8, 1, 1)); // item 3: 65536
    image.Word(0x234, B(0x234, InnerExit));
    image.Word(0x238, AddImm32(8, 8, 1)); // item 5: computed, not a constant
    image.Word(0x23C, B(0x23C, InnerExit));
    image.Word(0x240, OrrBitmask32(8, 0, 30)); // item 6: 0x7fffffff
    image.Word(0x244, B(0x244, InnerExit));
    image.Word(0x248, MovN32(8, 0)); // item 7: -1
    image.Word(0x24C, B(0x24C, InnerExit));
    image.Word(0x250, OrrBitmask32(8, 16, 15)); // item 8: 0xffff0000
    image.Word(0x254, B(0x254, InnerExit));
    image.Word(0x258, MovReg32(8, 31)); // item 9: 0
    image.Word(0x25C, B(0x25C, InnerExit));
    image.Word(0x260, MovZ32(8, 7)); // item 10: 7, falls into the exit
    image.Word(InnerExit, MovZ32(0, 1));
    image.Word(InnerExit + 4, StrW8X1);
    image.Word(InnerDefault, Ret);
    // item 2 and 4: the preloaded 14 (jump straight to the exit)
    const u8 inner_table[] = {13, 0, 13, 2, 4, 6, 8, 10, 12};
    std::memcpy(image.bytes.data() + InnerTable, inner_table, sizeof(inner_table));

    // Outer (item, device).
    image.Word(0x400, SubImm64(1, 29, 4));
    image.Word(0x404, MovReg32(20, 0));
    image.Word(0x408, Stur32(31, 29, -4)); // stur wzr, [x29, #-4]: the slot starts at zero
    image.Word(0x40C, Bl(0x40C, InnerEntry));
    image.Word(0x410, CmpImm32(20, OuterBound));
    image.Word(0x414, Bhi(0x414, OuterDefault));
    image.Word(0x418, MovReg32(8, 20));
    image.Word(0x41C, Adrp(9, 0x41C, OuterTable));
    image.Word(0x420, AddImm64(9, 9, OuterTable & 0xFFF));
    image.Word(0x424, MovZ32(0, 0xE7));
    image.Word(0x428, Adr(10, 0x428, OuterCases));
    image.Word(0x42C, Ldrb(11, 9, 8));
    image.Word(0x430, AddLsl2(10, 10, 11));
    image.Word(0x434, Br(10));
    image.Word(0x438, MovZ32(0, 39)); // item 1
    image.Word(0x43C, B(0x43C, OuterExit));
    image.Word(0x440, Ldr32(0, 19, 0x3C)); // item 2: a field of the device object
    image.Word(0x444, B(0x444, OuterExit));
    image.Word(0x448, MovZ32(0, 20)); // item 0x20
    image.Word(0x44C, B(0x44C, OuterExit));
    image.Word(OuterDefault, Ldur32(0, 29, -4));
    image.Word(OuterDefault + 4, B(OuterDefault + 4, OuterExit));
    image.Word(OuterExit, Ret);
    for (u32 item = 0; item <= OuterBound; ++item) {
        u32 target = OuterDefault;
        if (item == 0) {
            target = OuterExit; // the preloaded API major version
        } else if (item == 1) {
            target = 0x438;
        } else if (item == 2) {
            target = 0x440;
        } else if (item == 0x20) {
            target = 0x448;
        }
        image.bytes[OuterTable + item] = static_cast<u8>((target - OuterCases) / 4);
    }
    return image;
}

} // namespace

TEST_CASE("nvn sdk device info: reads the constants of the SDK's two switches", "[nvn][sdkinfo]") {
    const Image image = MakeImage();
    const auto info = SdkDeviceInfo::Read(image.bytes);
    REQUIRE(info.has_value());
    CHECK(info->SwitchAddress() == 0x434);

    CHECK(info->Get(0) == 231);  // the preloaded value, reached by a bare branch to the exit
    CHECK(info->Get(1) == 39);   // outer constant
    CHECK(!info->Get(2));        // a field of the device object: not a constant
    CHECK(info->Get(3) == 65536); // outer defers to the inner table: movz with a shift
    CHECK(info->Get(4) == 14);   // inner preloaded default
    CHECK(!info->Get(5));        // inner case that computes its value
    CHECK(info->Get(6) == 0x7FFFFFFF); // bitmask immediate
    CHECK(info->Get(7) == -1);   // movn
    CHECK(info->Get(8) == static_cast<s32>(0xFFFF0000u)); // rotated bitmask immediate
    CHECK(info->Get(9) == 0);    // mov wd, wzr
    CHECK(info->Get(10) == 7);   // falls into the exit block
    CHECK(info->Get(0x20) == 20); // outer constant between deferred items
    CHECK(info->Get(0x11) == 0); // past the inner table: the slot stays zero
    CHECK(info->Get(OuterBound + 1) == 0);
    CHECK(info->Get(0x1000) == 0);
    CHECK(info->ConstantCount() == OuterBound + 1 - 2); // every item but 2 and 5
}

TEST_CASE("nvn sdk device info: refuses what is not the NVN2 shape", "[nvn][sdkinfo]") {
    // No switch at all.
    CHECK(!SdkDeviceInfo::Read(std::vector<u8>(ImageSize)).has_value());
    CHECK(!SdkDeviceInfo::Read({}).has_value());

    // The outer default no longer reads the inner call's stack slot.
    Image image = MakeImage();
    image.Word(OuterDefault, Ldur32(0, 29, -8));
    CHECK(!SdkDeviceInfo::Read(image.bytes).has_value());

    // The inner function's exit does not store the value through its out pointer.
    image = MakeImage();
    image.Word(InnerExit + 4, Nop);
    CHECK(!SdkDeviceInfo::Read(image.bytes).has_value());

    // A bound below the last NVN1 item is not the NVN2 table.
    image = MakeImage();
    image.Word(0x410, CmpImm32(20, 0x20));
    CHECK(!SdkDeviceInfo::Read(image.bytes).has_value());
}

TEST_CASE("nvn sdk device info: nvnDeviceGetInteger answers the SDK first", "[nvn][sdkinfo]") {
    Rig rig;
    const u64 out = Rig::SCRATCH;
    constexpr u64 DEVICE = Rig::Obj(2);
    const auto get = [&](u32 item) {
        rig.memory.Put<s32>(out, 77);
        rig.Call("nvnDeviceGetInteger", {Rig::I(DEVICE), Rig::I(item), Rig::I(out)});
        return rig.memory.Get<s32>(out);
    };

    // Without an SDK reading the built-in table answers.
    CHECK(get(1) == 30);
    CHECK(get(0x20) == 2147483647);

    const Image image = MakeImage();
    auto info = SdkDeviceInfo::Read(image.bytes);
    REQUIRE(info.has_value());
    rig.runtime.device.SetSdkInfo(std::make_shared<const SdkDeviceInfo>(std::move(*info)));
    CHECK(get(0) == 231);
    CHECK(get(1) == 39);         // the SDK's minor version, not the table's 30
    CHECK(get(0x20) == 20);      // SDK constant over the table
    CHECK(get(2) == 14);         // not a constant in the SDK: the table
    CHECK(get(0x1000) == 0);     // the SDK answers 0 for items it has no case for
}

TEST_CASE("nvn sdk device info: the SDK generation follows its API minor, not its proc-name count",
          "[nvn][sdkinfo]") {
    Core::NvnHle::Hle::DeviceState device;
    device.sdk_proc_names = 698; // a 1.0.0-sized table
    CHECK(device.OlderSdk());    // nothing read: the table size decides

    const Image image = MakeImage(); // API minor 39
    auto info = SdkDeviceInfo::Read(image.bytes);
    REQUIRE(info.has_value());
    device.SetSdkInfo(std::make_shared<const SdkDeviceInfo>(std::move(*info)));
    CHECK(!device.OlderSdk()); // minor 39 is newer than the 3.2.0 generation whatever the table says

    device.SetSdkInfo(nullptr);
    CHECK(device.OlderSdk());
}
