// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// GPU execution tests of Ampere instruction forms: the instruction words are what ptxas (CUDA 13.3,
// -arch=sm_87) emits for the PTX instruction named in each test, the expected values are a CPU
// implementation of the PTX/SASS semantics. The words run through LowerToMaxwell, the Maxwell
// front end, the SPIR-V backend and a real Vulkan device (skipped without one); see ampere_exec.h.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "ampere_exec.h"

namespace {

using namespace AmpereExec;

// Operand-patched instruction templates ------------------------------------------------------------

constexpr Inst WithRegisters(Inst inst, u32 rd, u32 ra, u32 rb, u32 rc) {
    inst.w[0] = (inst.w[0] & ~0xff00ffffU) | 0 | (rd << 16) | (ra << 24) | (inst.w[0] & 0xffff);
    inst.w[1] = (inst.w[1] & ~0xffU) | rb;
    inst.w[2] = (inst.w[2] & ~0xffU) | rc;
    return inst;
}

/// IMAD.MOV.U32 rd, RZ, RZ, rc
constexpr Inst Mov(u32 rd, u32 rc) {
    return {{0xff007224U | (rd << 16), 0xff, 0x078e0000U | rc, Sched}};
}

float AsFloat(u32 bits) {
    return std::bit_cast<float>(bits);
}
u32 AsBits(float value) {
    return std::bit_cast<u32>(value);
}

std::vector<u32> FloatSamples() {
    std::vector<u32> bits;
    const auto add = [&](float value) {
        bits.push_back(AsBits(value));
        bits.push_back(AsBits(-value));
    };
    for (const float value :
         {0.0f, 1e-45f, 1e-40f, 1.1754942e-38f /* largest denormal */, 1.17549435e-38f /* smallest normal */,
          0.25f, 0.49999997f, 0.5f, 0.50000006f, 0.75f, 0.99999994f, 1.0f, 1.00000012f, 1.5f, 2.5f, 3.5f,
          2.4999998f, 2.5000002f, 100.5f, 8388607.5f, 8388608.0f, 16777216.0f, 33554432.0f,
          2147483520.0f, 2147483648.0f, 2147483904.0f, 4294967040.0f, 4294967296.0f, 1e10f, 3e38f,
          std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity()}) {
        add(value);
    }
    bits.push_back(0x7fc00000); // quiet NaN
    bits.push_back(0xffc00000);
    bits.push_back(0x7f800001); // signalling NaN
    bits.push_back(0x7fffffff);
    for (u32 bit = 0; bit < 23; ++bit) { // every denormal power of two and the runs below it
        for (const u32 magnitude : {1U << bit, (1U << (bit + 1)) - 1}) {
            bits.push_back(magnitude);
            bits.push_back(magnitude | 0x80000000U);
        }
    }
    std::mt19937 random(1234);
    for (int i = 0; i < 6000; ++i) {
        bits.push_back(random());
    }
    for (int i = 0; i < 2000; ++i) { // values in the integer range, with fractions
        const float fraction = static_cast<float>(random() & 0xffffff) / 16777216.0f;
        const int exponent = static_cast<int>(random() % 40);
        const float value = std::ldexp(fraction, exponent);
        const bool negative = (random() & 1) != 0;
        bits.push_back(AsBits(negative ? -value : value));
    }
    return bits;
}

std::vector<u32> IntegerSamples() {
    std::vector<u32> values{0, 1, 2, 3, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000,
                            0x7fffff, 0x800000, 0x800001, 0xffffff, 0x1000000, 0x1000001, 0x1000002,
                            0x1000003, 0x7ffffffe, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe,
                            0xffffffff, 0xfffffffd, 0xff000001, 0xfeffffff, 0xff7fffff, 0xdeadbeef};
    for (u32 bit = 1; bit < 32; ++bit) { // every power of two with its neighbours, both signs
        for (const s64 delta : {-1, 0, 1}) {
            values.push_back(static_cast<u32>((s64{1} << bit) + delta));
            values.push_back(static_cast<u32>(-((s64{1} << bit) + delta)));
        }
    }
    std::mt19937 random(99);
    for (int i = 0; i < 4000; ++i) {
        const u32 value = random();
        values.push_back(value >> (random() % 32));
        values.push_back(static_cast<u32>(-static_cast<s32>(value >> (random() % 32))));
    }
    return values;
}

/// 64-bit sources: the 32-bit edges widened, powers of two with neighbours around 2^24, 2^31, 2^32 and
/// 2^63, and values with a long run of ones below the 24-bit mantissa (the carry-out cases).
std::vector<u64> WideIntegerSamples() {
    std::vector<u64> values{0, 1, 0xffffffffULL, 0x100000000ULL, 0x7fffffffffffffffULL, 0x8000000000000000ULL,
                            0x8000000000000001ULL, 0xffffffffffffffffULL, 0xfffffffffffffffeULL,
                            0xffffff7fffffffffULL, 0xffffff8000000000ULL, 0x00ffffff7fffffffULL,
                            0x00ffffff80000000ULL, 0x00ffffff80000001ULL, 0xdeadbeefcafebabeULL};
    for (u32 bit = 1; bit < 64; ++bit) {
        for (const s64 delta : {-1, 0, 1}) {
            const u64 value = (u64{1} << bit) + static_cast<u64>(delta);
            values.push_back(value);
            values.push_back(u64{0} - value);
        }
    }
    u32 index = 0;
    for (const u32 low : IntegerSamples()) {
        values.push_back(low);
        values.push_back(static_cast<u64>(static_cast<s64>(static_cast<s32>(low))));
        values.push_back(u64{low} << 32);
        if (index++ % 4 == 0) {
            values.push_back((u64{low} << 20) | 0xfffffULL);
        }
    }
    return values;
}

// Reference semantics -------------------------------------------------------------------------------

enum class Round { Nearest = 0, Floor = 1, Ceil = 2, Trunc = 3 };

/// F2I: FTZ flushes a denormal input first; NaN gives 0; out-of-range values saturate.
u32 RefF2I(u32 bits, bool is_signed, Round round, bool ftz, u32 width_bits = 32) {
    float x = AsFloat(bits);
    if (std::isnan(x)) {
        return 0;
    }
    if (ftz && std::fpclassify(x) == FP_SUBNORMAL) {
        x = std::copysign(0.0f, x);
    }
    double r = x;
    switch (round) {
    case Round::Nearest: r = std::nearbyint(r); break; // default environment: ties to even
    case Round::Floor: r = std::floor(r); break;
    case Round::Ceil: r = std::ceil(r); break;
    case Round::Trunc: r = std::trunc(r); break;
    }
    if (is_signed) {
        const double high = std::ldexp(1.0, width_bits - 1) - 1;
        const double low = -std::ldexp(1.0, width_bits - 1);
        r = std::clamp(r, low, high);
        return static_cast<u32>(static_cast<s32>(r));
    }
    const double high = std::ldexp(1.0, width_bits) - 1;
    r = std::clamp(r, 0.0, high);
    return static_cast<u32>(static_cast<u64>(r));
}

/// Integer to float with an exact rounding mode (value as a signed or unsigned 64-bit integer).
u32 RefI2F64(u64 value, bool is_signed, Round round) {
    const bool negative = is_signed && static_cast<s64>(value) < 0;
    const u64 magnitude = negative ? u64{0} - value : value;
    if (magnitude == 0) {
        return 0;
    }
    const int msb = 63 - std::countl_zero(magnitude);
    int shift = 0;
    u64 mantissa = magnitude;
    if (msb >= 24) {
        shift = msb - 23;
        mantissa = magnitude >> shift;
        const u64 remainder = magnitude & ((u64{1} << shift) - 1);
        const u64 half = u64{1} << (shift - 1);
        bool up = false;
        switch (round) {
        case Round::Nearest: up = remainder > half || (remainder == half && (mantissa & 1)); break;
        case Round::Trunc: up = false; break;
        case Round::Floor: up = negative && remainder != 0; break;
        case Round::Ceil: up = !negative && remainder != 0; break;
        }
        mantissa += up;
        if (mantissa == (u64{1} << 24)) {
            mantissa >>= 1;
            ++shift;
        }
    }
    const float magnitude_float = std::ldexp(static_cast<float>(mantissa), shift);
    return AsBits(negative ? -magnitude_float : magnitude_float);
}

/// Integer to float from a 32-bit register.
u32 RefI2F(u32 value, bool is_signed, Round round) {
    return RefI2F64(is_signed ? static_cast<u64>(static_cast<s64>(static_cast<s32>(value))) : u64{value},
                    is_signed, round);
}

struct Mismatches {
    std::vector<std::string> lines;
    size_t total{};
    void Add(std::string line) {
        ++total;
        if (lines.size() < 12) {
            lines.push_back(std::move(line));
        }
    }
    std::string Describe() const {
        std::string text = fmt::format("{} mismatches", total);
        for (const auto& line : lines) {
            text += "\n  " + line;
        }
        return text;
    }
};

/// Runs `body` over `values` (input word 0) and compares output word 0 with `reference`.
template <typename Reference>
Mismatches Compare(Rig& rig, std::span<const Inst> body, const std::vector<u32>& values,
                   Reference reference, std::string& error) {
    std::vector<Words> inputs(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        inputs[i] = {values[i], 0, 0, 0, 0, 0, 0, 0};
    }
    Mismatches mismatches;
    const auto result = rig.Run(body, inputs);
    if (!result.error.empty()) {
        error = result.error;
        return mismatches;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        const u32 expected = reference(values[i]);
        const u32 actual = result.outputs[i][0];
        if (expected != actual) {
            mismatches.Add(fmt::format("input {:#010x}: expected {:#010x}, got {:#010x}", values[i],
                                       expected, actual));
        }
    }
    return mismatches;
}

} // namespace

namespace {
using namespace AmpereExec;

u32 PackOffsets(s32 x0, s32 y0, s32 x1, s32 y1) {
    return (static_cast<u32>(x0) & 0x3f) | ((static_cast<u32>(y0) & 0x3f) << 8) |
           ((static_cast<u32>(x1) & 0x3f) << 16) | ((static_cast<u32>(y1) & 0x3f) << 24);
}

constexpr u32 TextureMarker(u32 x, u32 y) {
    return 16 + x + 8 * y;
}

AmpereExec::Rig::SampledTexture MakePositionTexture() {
    constexpr u32 size = 8;
    AmpereExec::Rig::SampledTexture result{};
    result.texture.target = 1; // 2D
    result.texture.format = 0x25; // RGBA8
    result.texture.width = size;
    result.texture.height = size;
    result.texture.depth = 1;
    result.texture.levels = 1;
    result.texture.serial = 200;
    result.texture.pool_serial = 1;
    result.texture.pool_offset = 0x20000;
    result.sampler.min_filter = 0; // nearest
    result.sampler.mag_filter = 0;
    result.sampler.wrap_s = 7; // clamp-to-edge
    result.sampler.wrap_t = 7;
    result.sampler.wrap_r = 7;
    result.sampler.serial = 201;
    result.uniform_slot = 14 - 3; // Bound texture cbuffer 14 is uniform-buffer slot 11.
    result.uniform_offset = 0x4000;
    for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
            const float red = static_cast<float>(TextureMarker(x, y)) / 255.0f;
            VC::ClearTexturePayload clear{};
            clear.texture = {.serial = result.texture.serial, .level = 0, .layer = 0};
            clear.region = {static_cast<s32>(x), static_cast<s32>(y), 0, 1, 1, 1};
            clear.layer_count = 1;
            clear.color[0] = std::bit_cast<u32>(red);
            clear.color[3] = std::bit_cast<u32>(1.0f);
            result.clears.push_back(clear);
        }
    }
    return result;
}

} // namespace

TEST_CASE("Ampere GPU execution harness copies inputs to outputs", "[nvn][ampere_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    std::vector<Inst> body;
    for (u32 i = 0; i < 8; ++i) {
        body.push_back(Mov(FirstOutputRegister + i, FirstInputRegister + i));
    }
    std::vector<Words> inputs;
    for (u32 i = 0; i < 300; ++i) {
        inputs.push_back({i, i * 3, i * 7, ~i, 0x80000000u | i, 5, 6, 0xdeadbeef});
    }
    const auto result = rig->Run(body, inputs);
    REQUIRE(result.error == "");
    for (u32 i = 0; i < inputs.size(); ++i) {
        for (u32 j = 0; j < 8; ++j) {
            REQUIRE(result.outputs[i][j] == inputs[i][j]);
        }
    }
}

TEST_CASE("Ampere TLD4 per-tap offsets gather the four offset texels", "[nvn][ampere_exec][tld4]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    Inst gather{{0x040a7b63, 0x33800408, 0x041e2f08, Sched}};
    gather.w[0] = (gather.w[0] & 0x0000ffffU) | (FirstOutputRegister << 16) |
                  (FirstInputRegister << 24);
    gather.w[1] = (gather.w[1] & ~0xffU) | (FirstInputRegister + 2);
    gather.w[2] = (gather.w[2] & ~0xffU) | 0xffU;

    const std::array body{gather};
    const std::array<Words, 1> inputs{{{
        AsBits(3.5f / 8.0f), AsBits(3.5f / 8.0f),
        PackOffsets(-8, -8, -1, 1), PackOffsets(0, -1, 1, 0),
        0, 0, 0, 0,
    }}};
    const auto texture = MakePositionTexture();
    const auto result = rig->Run(body, inputs, &texture);
    REQUIRE(result.error.empty());

    // At (3.5, 3.5) texels, component .w is the lower-left tap; offsets clamp at the edge.
    constexpr std::array<u32, 4> expected_markers{
        TextureMarker(0, 0), TextureMarker(2, 4), TextureMarker(3, 2), TextureMarker(4, 3),
    };
    for (u32 tap = 0; tap < expected_markers.size(); ++tap) {
        const float expected = static_cast<float>(expected_markers[tap]) / 255.0f;
        const float actual = AsFloat(result.outputs[0][tap]);
        INFO("tap " << tap << " expected " << expected << " got " << actual);
        CHECK(std::abs(actual - expected) < 1e-6f);
    }
}

TEST_CASE("Ampere I2F from 32 bits rounds exactly in every mode", "[nvn][ampere_exec][i2f]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    struct Edge {
        u32 value;
        bool is_signed;
        std::array<u32, 4> expected; // RN, RM, RP, RZ
    };
    constexpr std::array edges{
        Edge{0x00000000, true, {0x00000000, 0x00000000, 0x00000000, 0x00000000}},
        Edge{0x01000000, true, {0x4b800000, 0x4b800000, 0x4b800000, 0x4b800000}},
        Edge{0x01000001, true, {0x4b800000, 0x4b800000, 0x4b800001, 0x4b800000}},
        Edge{0x01000002, true, {0x4b800001, 0x4b800001, 0x4b800001, 0x4b800001}},
        Edge{0x01000003, true, {0x4b800002, 0x4b800001, 0x4b800002, 0x4b800001}},
        Edge{0xfeffffff, true, {0xcb800000, 0xcb800001, 0xcb800000, 0xcb800000}},
        Edge{0xfefffffd, true, {0xcb800002, 0xcb800002, 0xcb800001, 0xcb800001}},
        Edge{0x7fffffff, true, {0x4f000000, 0x4effffff, 0x4f000000, 0x4effffff}},
        Edge{0x80000000, true, {0xcf000000, 0xcf000000, 0xcf000000, 0xcf000000}},
        Edge{0x80000001, true, {0xcf000000, 0xcf000000, 0xceffffff, 0xceffffff}},
        Edge{0xffffffff, false, {0x4f800000, 0x4f7fffff, 0x4f800000, 0x4f7fffff}},
        Edge{0xffffff80, false, {0x4f800000, 0x4f7fffff, 0x4f800000, 0x4f7fffff}},
        Edge{0xffffff7f, false, {0x4f7fffff, 0x4f7fffff, 0x4f800000, 0x4f7fffff}},
    };
    const auto samples = IntegerSamples();
    for (const bool is_signed : {true, false}) {
        for (const Round round : {Round::Nearest, Round::Floor, Round::Ceil, Round::Trunc}) {
            // Rounding in bits 78-79 (0 RN, 1 RM, 2 RP, 3 RZ). nvdisasm sm_86 (isa_oracle): I2FP.F32.{S32|U32}
            // [.RZ] R16, R8 (0x245) has only RN and RZ; the directed .RM/.RP are I2F[.U32].RM/.RP (0x306).
            const u32 field = round == Round::Nearest ? 0U : round == Round::Floor ? 1U
                              : round == Round::Ceil ? 2U : 3U;
            const u32 opcode = field == 1U || field == 2U ? 0x306U : 0x245U;
            const u32 mode = 0x00201000U | (is_signed ? 0x400U : 0U) | (field << 14);
            const std::array body{Inst{{0x00007000U | opcode | (16U << 16), 8, mode, Sched}}};
            std::string error;
            const auto mismatches = Compare(
                *rig, body, samples, [&](u32 value) { return RefI2F(value, is_signed, round); }, error);
            INFO("signed " << is_signed << " round " << static_cast<int>(round));
            INFO(error);
            REQUIRE(error.empty());
            INFO(mismatches.Describe());
            CHECK(mismatches.total == 0);
            std::vector<Words> inputs;
            std::vector<u32> expected;
            for (const auto& edge : edges) {
                if (edge.is_signed == is_signed) {
                    inputs.push_back({edge.value, 0, 0, 0, 0, 0, 0, 0});
                    expected.push_back(edge.expected[field]);
                }
            }
            const auto result = rig->Run(body, inputs);
            REQUIRE(result.error.empty());
            for (size_t i = 0; i < inputs.size(); ++i) {
                INFO("edge " << std::hex << inputs[i][0] << " expected " << expected[i] << " got "
                             << result.outputs[i][0]);
                CHECK(result.outputs[i][0] == expected[i]);
            }
        }
    }
}

TEST_CASE("Ampere F2I to 32 bits rounds, saturates and flushes as the hardware does",
          "[nvn][ampere_exec][f2i]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    struct Edge {
        u32 input;
        bool is_signed;
        Round round;
        bool ftz;
        u32 expected;
    };
    const std::array edges{
        Edge{0x00000001, true, Round::Floor, false, 0},
        Edge{0x00000001, false, Round::Floor, false, 0},
        Edge{0x00000001, true, Round::Ceil, false, 1},
        Edge{0x00000001, false, Round::Ceil, false, 1},
        Edge{0x00000001, true, Round::Floor, true, 0},
        Edge{0x00000001, false, Round::Floor, true, 0},
        Edge{0x00000001, true, Round::Ceil, true, 0},
        Edge{0x00000001, false, Round::Ceil, true, 0},
        Edge{0x80000001, true, Round::Floor, false, 0xffffffff},
        Edge{0x80000001, false, Round::Floor, false, 0},
        Edge{0x80000001, true, Round::Ceil, false, 0},
        Edge{0x80000001, false, Round::Ceil, false, 0},
        Edge{0x80000001, true, Round::Floor, true, 0},
        Edge{0x80000001, false, Round::Floor, true, 0},
        Edge{0x80000001, true, Round::Ceil, true, 0},
        Edge{0x80000001, false, Round::Ceil, true, 0},
        Edge{0x007fffff, true, Round::Ceil, false, 1},
        Edge{0x007fffff, true, Round::Floor, true, 0},
        Edge{0x807fffff, true, Round::Floor, false, 0xffffffff},
        Edge{0x807fffff, true, Round::Ceil, false, 0},
        Edge{0x00800000, true, Round::Floor, true, 0},
        Edge{0x00800000, true, Round::Ceil, true, 1},
        Edge{0x80800000, true, Round::Floor, true, 0xffffffff},
        Edge{0x80800000, true, Round::Ceil, true, 0},
        Edge{0x7fc00000, true, Round::Floor, false, 0},
        Edge{0x7fc00000, false, Round::Ceil, false, 0},
        Edge{0x7f800000, true, Round::Floor, false, 0x7fffffff},
        Edge{0xff800000, true, Round::Ceil, false, 0x80000000},
        Edge{0x7f800000, false, Round::Ceil, true, 0xffffffff},
        Edge{0xff800000, false, Round::Floor, false, 0},
        Edge{0x4f7fffff, false, Round::Trunc, false, 0xffffff00},
        Edge{0x4effffff, true, Round::Trunc, false, 0x7fffff80},
    };
    const auto samples = FloatSamples();
    for (const bool is_signed : {true, false}) {
        for (const Round round : {Round::Nearest, Round::Floor, Round::Ceil, Round::Trunc}) {
            for (const bool ftz : {false, true}) {
                // F2I.[FTZ.][U32.][FLOOR|CEIL|TRUNC.]NTZ R16, R8 as ptxas emits it:
                // bit 72 signed, bits 75-76 = 2 (32 bits), bit 77 NTZ, 78-79 rounding, bit 80 FTZ,
                // bits 84-86 = 2 (32-bit float source).
                const u32 mode = 0x00203000U | (is_signed ? 0x100U : 0U) |
                                 (static_cast<u32>(round) << 14) | (ftz ? 0x10000U : 0U);
                const std::array body{Inst{{0x00007305U | (16U << 16), 8, mode, Sched}}};
                std::string error;
                const auto mismatches = Compare(
                    *rig, body, samples,
                    [&](u32 bits) { return RefF2I(bits, is_signed, round, ftz); }, error);
                INFO("signed " << is_signed << " round " << static_cast<int>(round) << " ftz " << ftz);
                INFO(error);
                REQUIRE(error.empty());
                INFO(mismatches.Describe());
                CHECK(mismatches.total == 0);
                std::vector<Words> inputs;
                std::vector<u32> expected;
                for (const auto& edge : edges) {
                    if (edge.is_signed == is_signed && edge.round == round && edge.ftz == ftz) {
                        inputs.push_back({edge.input, 0, 0, 0, 0, 0, 0, 0});
                        expected.push_back(edge.expected);
                    }
                }
                if (inputs.empty()) continue;
                const auto result = rig->Run(body, inputs);
                REQUIRE(result.error.empty());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    INFO("edge " << std::hex << inputs[i][0] << " expected " << expected[i]
                                 << " got " << result.outputs[i][0]);
                    CHECK(result.outputs[i][0] == expected[i]);
                }
            }
        }
    }
}

TEST_CASE("Ampere F2I mixes FTZ and preserved denormals in one program",
          "[nvn][ampere_exec][f2i]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    const std::vector<Words> inputs{
        {0x00000001, 0, 0, 0, 0, 0, 0, 0}, {0x80000001, 0, 0, 0, 0, 0, 0, 0},
        {0x007fffff, 0, 0, 0, 0, 0, 0, 0}, {0x807fffff, 0, 0, 0, 0, 0, 0, 0},
        {0x00800000, 0, 0, 0, 0, 0, 0, 0}, {0x80800000, 0, 0, 0, 0, 0, 0, 0},
    };
    for (const bool is_signed : {true, false}) {
        for (const Round round : {Round::Floor, Round::Ceil}) {
            const u32 mode = 0x00203000U | (is_signed ? 0x100U : 0U) |
                             (static_cast<u32>(round) << 14);
            const std::array body{
                Inst{{0x00007305U | (16U << 16), 8, mode, Sched}},
                Inst{{0x00007305U | (17U << 16), 8, mode | 0x10000U, Sched}},
            };
            const auto result = rig->Run(body, inputs);
            INFO("signed " << is_signed << " round " << static_cast<int>(round));
            REQUIRE(result.error.empty());
            for (size_t i = 0; i < inputs.size(); ++i) {
                INFO("input " << std::hex << inputs[i][0]);
                CHECK(result.outputs[i][0] == RefF2I(inputs[i][0], is_signed, round, false));
                CHECK(result.outputs[i][1] == RefF2I(inputs[i][0], is_signed, round, true));
            }
        }
    }
}

TEST_CASE("Ampere I2F from 64 bits rounds exactly in every mode", "[nvn][ampere_exec][i2f]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    struct Edge {
        u64 value;
        bool is_signed;
        std::array<u32, 4> expected; // RN, RM, RP, RZ
    };
    constexpr std::array edges{
        Edge{0x8000000000000001ULL, true, {0xdf000000, 0xdf000000, 0xdeffffff, 0xdeffffff}},
        Edge{0x7fffffffffffffffULL, true, {0x5f000000, 0x5effffff, 0x5f000000, 0x5effffff}},
        Edge{0x8000000000000000ULL, true, {0xdf000000, 0xdf000000, 0xdf000000, 0xdf000000}},
        Edge{0xffffffffffffffffULL, true, {0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000}},
        Edge{0xffffffffffffffffULL, false, {0x5f800000, 0x5f7fffff, 0x5f800000, 0x5f7fffff}},
        Edge{0x0000000001000003ULL, true, {0x4b800002, 0x4b800001, 0x4b800002, 0x4b800001}},
        Edge{0x0000000100000001ULL, false, {0x4f800000, 0x4f800000, 0x4f800001, 0x4f800000}},
        Edge{0xfffffffeffffffffULL, true, {0xcf800000, 0xcf800001, 0xcf800000, 0xcf800000}},
    };
    const auto samples = WideIntegerSamples();
    std::vector<Words> inputs(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        inputs[i] = {static_cast<u32>(samples[i]), static_cast<u32>(samples[i] >> 32), 0, 0, 0, 0, 0, 0};
    }
    for (const bool is_signed : {true, false}) {
        for (const Round round : {Round::Nearest, Round::Floor, Round::Ceil, Round::Trunc}) {
            // I2F[.U64][.RM|.RP|.RZ] R16, R8 (0x312): bit 74 sign, bits 84-85 = 3 (64-bit source), rounding 78-79.
            const u32 field = static_cast<u32>(round);
            const u32 mode = 0x00301000U | (is_signed ? 0x400U : 0U) | (field << 14);
            const std::array body{Inst{{0x00007312U | (16U << 16), 8, mode, Sched}}};
            INFO("signed " << is_signed << " round " << static_cast<int>(round));
            const auto result = rig->Run(body, inputs);
            INFO(result.error);
            REQUIRE(result.error.empty());
            Mismatches mismatches;
            for (size_t i = 0; i < samples.size(); ++i) {
                const u32 expected = RefI2F64(samples[i], is_signed, round);
                if (result.outputs[i][0] != expected) {
                    mismatches.Add(fmt::format("input {:#018x}: expected {:#010x}, got {:#010x}", samples[i],
                                               expected, result.outputs[i][0]));
                }
            }
            INFO(mismatches.Describe());
            CHECK(mismatches.total == 0);
            std::vector<Words> edge_inputs;
            std::vector<u32> expected;
            for (const auto& edge : edges) {
                if (edge.is_signed == is_signed) {
                    edge_inputs.push_back({static_cast<u32>(edge.value), static_cast<u32>(edge.value >> 32), 0, 0, 0, 0, 0, 0});
                    expected.push_back(edge.expected[field]);
                }
            }
            const auto edge_result = rig->Run(body, edge_inputs);
            REQUIRE(edge_result.error.empty());
            for (size_t i = 0; i < edge_inputs.size(); ++i) {
                INFO("edge " << std::hex << edge_inputs[i][1] << ':' << edge_inputs[i][0] << " expected "
                             << expected[i] << " got " << edge_result.outputs[i][0]);
                CHECK(edge_result.outputs[i][0] == expected[i]);
            }
        }
    }
}

TEST_CASE("Ampere F2I applies |Rs| and -Rs before rounding and flushing", "[nvn][ampere_exec][f2i]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    const auto samples = FloatSamples();
    for (const bool is_signed : {true, false}) {
        for (const Round round : {Round::Nearest, Round::Floor, Round::Ceil, Round::Trunc}) {
            for (const bool ftz : {false, true}) {
                for (const u32 modifier : {1U << 30, 1U << 31, 3U << 30}) { // .ABS, .NEG, -|Rs|
                    const u32 mode = 0x00203000U | (is_signed ? 0x100U : 0U) |
                                     (static_cast<u32>(round) << 14) | (ftz ? 0x10000U : 0U);
                    const std::array body{Inst{{0x00007305U | (16U << 16), 8U | modifier, mode, Sched}}};
                    const auto modified = [&](u32 bits) {
                        const u32 absolute = (modifier & (1U << 30)) != 0 ? bits & 0x7fffffffU : bits;
                        return (modifier & (1U << 31)) != 0 ? absolute ^ 0x80000000U : absolute;
                    };
                    std::string error;
                    const auto mismatches = Compare(
                        *rig, body, samples,
                        [&](u32 bits) { return RefF2I(modified(bits), is_signed, round, ftz); }, error);
                    INFO("signed " << is_signed << " round " << static_cast<int>(round) << " ftz " << ftz
                                   << " modifier " << std::hex << modifier);
                    INFO(error);
                    REQUIRE(error.empty());
                    INFO(mismatches.Describe());
                    CHECK(mismatches.total == 0);
                }
            }
        }
    }
}

TEST_CASE("Ampere F2IP converts two floats to saturated bytes and packs them over Rc",
          "[nvn][ampere_exec][f2ip]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }
    // F2IP.{U8,S8}.F32[.TRUNC][.RELU].NTZ R16, R8, R9, R10 (OoT: 28297243 00000027 000004ff =
    // F2IP.U8.F32.NTZ R41,R40,R39,RZ, nvdisasm sm_86). Rd = Rc[15:0] << 16 | sat(Ra) << 8 | sat(Rb),
    // the layout PTX documents for cvt.pack.sat.u8 (I2IP); NaN converts to 0.
    const auto reference = [](u32 bits, bool is_signed, bool relu, bool trunc) -> u32 {
        const float value = std::bit_cast<float>(bits);
        if (std::isnan(value)) return 0;
        const double rounded = trunc ? std::trunc(double{value}) : std::nearbyint(double{value});
        const double low = is_signed && !relu ? -128.0 : 0.0, high = is_signed ? 127.0 : 255.0;
        return static_cast<u32>(static_cast<s32>(std::clamp(rounded, low, high))) & 0xffU;
    };
    const std::array<float, 22> values{
        0.0f, -0.0f, 0.25f, 0.5f, 1.5f, 2.5f, 127.4f, 127.5f, 128.0f, 254.5f, 255.0f, 255.5f, 256.0f,
        1e9f, -0.5f, -1.5f, -127.5f, -128.5f, -129.0f, -1e9f,
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    std::vector<Words> inputs;
    for (const float a : values) {
        for (const float b : values) {
            inputs.push_back({AsBits(a), AsBits(b), 0x1234abcd, 0, 0, 0, 0, 0});
        }
    }
    inputs.push_back({0x7fc00000, 0xffc00001, 0xffffffff, 0, 0, 0, 0, 0}); // NaNs, full Rc
    for (const bool is_signed : {false, true}) {
        for (const bool relu : {false, true}) {
            for (const bool trunc : {false, true}) {
                const u32 mode = (1U << 10) | (u32{relu} << 11) | (u32{is_signed} << 12) | (trunc ? 3U << 14 : 0U);
                // Ra/Rb/Rc set, then RZ for Ra and Rc (the form ptxas emits for cvt.*.sat.u8.f32).
                const std::array<Inst, 2> bodies{
                    Inst{{0x00007243U | (FirstOutputRegister << 16) | (FirstInputRegister << 24),
                          FirstInputRegister + 1, mode | (FirstInputRegister + 2), Sched}},
                    Inst{{0xff007243U | ((FirstOutputRegister + 1) << 16), FirstInputRegister + 1,
                          mode | 0xffU, Sched}},
                };
                const auto result = rig->Run(bodies, inputs);
                REQUIRE(result.error.empty());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const u32 a = reference(inputs[i][0], is_signed, relu, trunc);
                    const u32 b = reference(inputs[i][1], is_signed, relu, trunc);
                    INFO("signed " << is_signed << " relu " << relu << " trunc " << trunc << " a "
                                   << AsFloat(inputs[i][0]) << " b " << AsFloat(inputs[i][1]));
                    CHECK(result.outputs[i][0] == ((inputs[i][2] << 16) | (a << 8) | b));
                    CHECK(result.outputs[i][1] == b);
                }
            }
        }
    }
}
