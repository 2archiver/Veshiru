// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Synthetic, game-free tests of the NVN2 shader container extractor (nvn_shader_containers.h):
// SARC / BNSH / BAHS / NDSR / sead / NSO / zstd containers are built in memory and scanned.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/lz4_compression.h"
#include "nvn_shader_containers.h"
#include "tests/temp_path.h"

namespace VideoCommon::Nvn::Containers {
namespace {

using Blob = std::vector<u8>;

u32 Rd32(const Blob& b, size_t off) {
    u32 v;
    std::memcpy(&v, b.data() + off, sizeof(v));
    return v;
}

void Wr32(Blob& b, size_t off, u32 v) {
    std::memcpy(b.data() + off, &v, sizeof(v));
}

/// A shader with a valid control section; `seed` makes the code (and so the key) unique.
BahsEntry MakeShader(u32 stage, u32 seed) {
    const u32 code_size = 0x200 + 0x20 * (seed % 5);
    return {MakeControl(stage, code_size, 0x180), MakeCode(code_size, seed)};
}

BnshStage ToStage(u32 slot, const BahsEntry& shader) {
    return {slot, shader.control, shader.code};
}

ScanOutput Scan(const Blob& blob, const std::string& label = "test") {
    ScanOutput out;
    ScanBlob(blob, label, out);
    return out;
}

const ExtractedShader* FindByCode(const ScanOutput& out, const Blob& code) {
    for (const ExtractedShader& s : out.shaders) {
        if (s.code == code) {
            return &s;
        }
    }
    return nullptr;
}

bool HasFailure(const ScanOutput& out, const std::string& reason_prefix) {
    return std::any_of(out.failures.begin(), out.failures.end(), [&](const Failure& f) {
        return f.reason.rfind(reason_prefix, 0) == 0;
    });
}

/// Every extracted shader is internally consistent and every failure is a container failure.
void CheckInvariants(const ScanOutput& out) {
    for (const ExtractedShader& s : out.shaders) {
        REQUIRE(s.control.size() >= ControlKeyBytes);
        REQUIRE(Rd32(s.control, 0) == ControlMagic);
        REQUIRE(Rd32(s.control, 4) == ControlVersion);
        REQUIRE(Rd32(s.control, 0x44) == s.code.size());
        REQUIRE(s.stage_byte == (s.control.size() > 0x56 ? s.control[0x56] : 0u));
        REQUIRE(s.key == ShaderKey(s.control, s.code));
        REQUIRE(!s.source.empty());
        REQUIRE(!s.kind.empty());
    }
    for (const Failure& f : out.failures) {
        REQUIRE(f.reason.rfind("container:", 0) == 0);
    }
}

Blob Concat(const Blob& a, const Blob& b) {
    Blob r = a;
    r.insert(r.end(), b.begin(), b.end());
    return r;
}

using Members = std::vector<std::pair<std::string, Blob>>;

Blob Sarc(const Members& members, bool be = false) {
    return MakeSarc(members, be);
}

/// A minimal NSO: 0x100-byte header and three segments, each optionally LZ4 compressed.
Blob MakeNso(const std::array<Blob, 3>& segments, const std::array<bool, 3>& compress) {
    Blob nso(0x100);
    std::memcpy(nso.data(), "NSO0", 4);
    u32 flags = 0;
    u32 mem = 0;
    for (u32 i = 0; i < 3; ++i) {
        const Blob stored = compress[i] ? Common::Compression::CompressDataLZ4(
                                              segments[i].data(), segments[i].size())
                                        : segments[i];
        flags |= compress[i] ? (1u << i) : 0u;
        Wr32(nso, 0x10 + 0x10 * i, static_cast<u32>(nso.size()));
        Wr32(nso, 0x14 + 0x10 * i, mem);
        Wr32(nso, 0x18 + 0x10 * i, static_cast<u32>(segments[i].size()));
        Wr32(nso, 0x60 + 4 * i, static_cast<u32>(stored.size()));
        nso.insert(nso.end(), stored.begin(), stored.end());
        mem += static_cast<u32>((segments[i].size() + 0xfff) & ~size_t{0xfff});
    }
    Wr32(nso, 0xc, flags);
    return nso;
}

/// `size` junk bytes with `inner` embedded at `at`.
Blob Embed(size_t size, size_t at, const Blob& inner, u32 seed) {
    Blob data = MakeCode(std::max(size, at + inner.size() + 0x40), seed);
    // Keep the junk free of accidental magics.
    std::replace(data.begin(), data.end(), u8{'B'}, u8{'b'});
    std::copy(inner.begin(), inner.end(), data.begin() + static_cast<std::ptrdiff_t>(at));
    return data;
}

struct Corpus {
    std::string name;
    std::string label;
    Blob data;
};

/// Valid containers of every kind (and nested/compressed forms), used for the robustness sweeps.
std::vector<Corpus> BuildCorpus() {
    const BahsEntry vs = MakeShader(0, 101);
    const BahsEntry ps = MakeShader(1, 102);
    const BahsEntry cs = MakeShader(5, 103);
    const std::array<BnshVariation, 2> variations{
        BnshVariation{ToStage(0, vs), ToStage(4, ps)},
        BnshVariation{ToStage(5, cs)},
    };
    const Blob bnsh = MakeBnsh(variations);
    const std::array<BahsEntry, 3> entries{vs, ps, cs};
    const Blob bahs = MakeBahs(entries);
    const Blob ndsr = MakeNdsr(entries);
    const Blob sead = MakeSeadPrimitiveDrawer(vs, ps);
    const Blob sarc_le = Sarc({{"a.bnsh", bnsh}, {"b.sharcb", bahs}, {"c.pck", ndsr}});
    const Blob sarc_be = Sarc({{"a.bnsh", bnsh}, {"b.sharcb", bahs}}, true);
    const Blob nso = MakeNso({Blob(0x80, 0x90), Blob(0x100, 0x11), Embed(0x1800, 0x400, bnsh, 7)},
                             {true, true, true});
    const Blob nso_raw = MakeNso({Blob(0x80, 0x90), Embed(0x1400, 0x200, bnsh, 8), Blob(0x100, 3)},
                                 {false, false, false});
    return {
        {"bnsh", "bnsh", bnsh},
        {"sarc-le", "x.sarc", sarc_le},
        {"sarc-be", "x.sarc", sarc_be},
        {"bahs", "x.sharcb", bahs},
        {"ndsr", "x.pck", ndsr},
        {"sead", "romfs/Lib/sead/primitive_renderer/primitive_drawer_nvn2_shader.bin", sead},
        {"nso", "exefs/main", nso},
        {"nso-raw", "exefs/main", nso_raw},
        {"zstd-sarc", "x.sarc.zs", ZstdCompress(sarc_le)},
        {"zstd-sarc-unsized", "x.sarc.zs", ZstdCompress(sarc_le, false)},
    };
}

} // namespace

TEST_CASE("NVN containers: SHA-256 known answers", "[nvn]") {
    const auto hex = [](std::string_view text) {
        return Sha256Hex({reinterpret_cast<const u8*>(text.data()), text.size()});
    };
    REQUIRE(hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE(hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // 56 bytes: the padding spills into a second block.
    REQUIRE(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("NVN containers: control builder and shader key", "[nvn]") {
    const Blob control = MakeControl(5, 0x340, 0x2a0);
    REQUIRE(control.size() == ControlSize);
    REQUIRE(Rd32(control, 0) == 0x98761234);
    REQUIRE(Rd32(control, 4) == 0xc9);
    REQUIRE(Rd32(control, 0x30) == 0xd8);
    REQUIRE(Rd32(control, 0x38) == 0x2a0);
    REQUIRE(Rd32(control, 0x40) == 0x300);
    REQUIRE(Rd32(control, 0x44) == 0x340);
    REQUIRE(control[0x56] == 5);
    REQUIRE(Rd32(control, 0x58) == 32);

    const Blob code = MakeCode(0x340, 1);
    const std::string key = ShaderKey(control, code);
    REQUIRE(key.size() == 32);

    // The key is the SHA-256 prefix over control[0..0x48) with +0x30 zeroed, then the code.
    Blob expected_input(control.begin(), control.begin() + 0x48);
    Wr32(expected_input, 0x30, 0);
    expected_input.insert(expected_input.end(), code.begin(), code.end());
    REQUIRE(key == Sha256Hex(expected_input).substr(0, 32));

    // The own-length word and bytes after 0x48 do not take part in the key.
    Blob other = control;
    Wr32(other, 0x30, 0);
    other[0x50] ^= 0xff;
    REQUIRE(ShaderKey(other, code) == key);
    // Everything else in the first 0x48 bytes does.
    other = control;
    Wr32(other, 0x38, 0x2b0);
    REQUIRE(ShaderKey(other, code) != key);
    Blob other_code = code;
    other_code.back() ^= 1;
    REQUIRE(ShaderKey(control, other_code) != key);
    REQUIRE(ShaderKey(Blob(control.begin(), control.begin() + 0x40), code).empty());
}

TEST_CASE("NVN containers: BNSH with two variations", "[nvn]") {
    const BahsEntry vs = MakeShader(0, 1);
    const BahsEntry ps0 = MakeShader(1, 2);
    const BahsEntry ps1 = MakeShader(1, 3);
    const BahsEntry gs = MakeShader(2, 4);
    const std::array<BnshVariation, 3> variations{
        BnshVariation{ToStage(0, vs), ToStage(4, ps0)},
        BnshVariation{ToStage(0, vs), ToStage(3, gs), ToStage(4, ps1)},
        BnshVariation{}, // no program
    };
    const Blob bnsh = MakeBnsh(variations);
    REQUIRE(std::memcmp(bnsh.data(), "BNSH\0\0\0\0", 8) == 0);

    const ScanOutput out = Scan(bnsh, "romfs/Shader/X.bfsha");
    CheckInvariants(out);
    REQUIRE(out.failures.empty());
    REQUIRE(out.shaders.size() == 4);
    REQUIRE(out.duplicates == 1); // variation 1 shares the vertex shader

    const ExtractedShader* s = FindByCode(out, vs.code);
    REQUIRE(s != nullptr);
    CHECK(s->kind == "bnsh");
    CHECK(s->stage_byte == 0);
    CHECK(s->control == vs.control);
    CHECK(s->source.find("romfs/Shader/X.bfsha") == 0);
    CHECK(s->source.find("#var0.vs") != std::string::npos);

    s = FindByCode(out, ps1.code);
    REQUIRE(s != nullptr);
    CHECK(s->stage_byte == 1);
    CHECK(s->source.find("#var1.ps") != std::string::npos);

    s = FindByCode(out, gs.code);
    REQUIRE(s != nullptr);
    CHECK(s->stage_byte == 2);
    CHECK(s->source.find("#var1.gs") != std::string::npos);
}

TEST_CASE("NVN containers: BNSH is found anywhere in a file", "[nvn]") {
    const BahsEntry a = MakeShader(0, 11);
    const BahsEntry b = MakeShader(1, 12);
    const std::array<BnshVariation, 1> first{BnshVariation{ToStage(0, a)}};
    const std::array<BnshVariation, 1> second{BnshVariation{ToStage(4, b)}};

    Blob file = Embed(0x333, 0, {}, 5);
    const Blob one = MakeBnsh(first);
    file.insert(file.end(), one.begin(), one.end());
    const Blob pad = Embed(0x217, 0, {}, 6);
    file.insert(file.end(), pad.begin(), pad.end());
    // The second BNSH is not aligned and is followed by junk containing a bare 'BNSH' tag.
    const Blob two = MakeBnsh(second);
    file.insert(file.end(), two.begin(), two.end());
    const char tail[] = "BNSH\0\0\0\0 not a program";
    file.insert(file.end(), tail, tail + sizeof(tail));

    const ScanOutput out = Scan(file);
    CheckInvariants(out);
    REQUIRE(out.failures.empty());
    REQUIRE(out.shaders.size() == 2);
    REQUIRE(FindByCode(out, a.code) != nullptr);
    REQUIRE(FindByCode(out, b.code) != nullptr);
}

TEST_CASE("NVN containers: SARC members are expanded recursively", "[nvn]") {
    const BahsEntry a = MakeShader(0, 21);
    const BahsEntry b = MakeShader(1, 22);
    const BahsEntry c = MakeShader(1, 23);
    const std::array<BnshVariation, 1> va{BnshVariation{ToStage(0, a), ToStage(4, b)}};
    const std::array<BnshVariation, 1> vc{BnshVariation{ToStage(4, c)}};
    const Blob inner = Sarc({{"inner.bnsh", MakeBnsh(vc)}, {"inner.txt", MakeCode(40, 1)}});

    for (const bool be : {false, true}) {
        CAPTURE(be);
        const Blob sarc = Sarc({{"readme.txt", MakeCode(100, 9)},
                                {"dir/a.bnsh", MakeBnsh(va)},
                                {"nested.sarc", inner}},
                               be);
        const ScanOutput out = Scan(sarc, "pack.sarc");
        CheckInvariants(out);
        REQUIRE(out.failures.empty());
        REQUIRE(out.shaders.size() == 3);
        const ExtractedShader* s = FindByCode(out, a.code);
        REQUIRE(s != nullptr);
        CHECK(s->source.find("pack.sarc>dir/a.bnsh#BNSH@") == 0);
        s = FindByCode(out, c.code);
        REQUIRE(s != nullptr);
        CHECK(s->source.find("pack.sarc>nested.sarc>inner.bnsh#BNSH@") == 0);
    }
}

TEST_CASE("NVN containers: zstd frames", "[nvn]") {
    const BahsEntry a = MakeShader(0, 31);
    const BahsEntry b = MakeShader(1, 32);
    const std::array<BnshVariation, 1> va{BnshVariation{ToStage(0, a)}};
    const std::array<BnshVariation, 1> vb{BnshVariation{ToStage(4, b)}};
    const Blob bnsh_a = MakeBnsh(va);
    const Blob bnsh_b = MakeBnsh(vb);

    SECTION("single frame with and without a content size") {
        for (const bool sized : {true, false}) {
            CAPTURE(sized);
            const Blob zs = ZstdCompress(bnsh_a, sized);
            REQUIRE(zs.size() > 4);
            REQUIRE(Rd32(zs, 0) == 0xFD2FB528);
            const ScanOutput out = Scan(zs, "a.bfsha.zs");
            CheckInvariants(out);
            REQUIRE(out.failures.empty());
            REQUIRE(out.shaders.size() == 1);
            CHECK(out.shaders[0].code == a.code);
        }
    }

    SECTION("concatenated frames") {
        const Blob zs = Concat(ZstdCompress(bnsh_a, false), ZstdCompress(bnsh_b));
        const ScanOutput out = Scan(zs);
        CheckInvariants(out);
        REQUIRE(out.failures.empty());
        REQUIRE(out.shaders.size() == 2);
    }

    SECTION("compressed SARC and compressed members") {
        const Blob sarc = Sarc({{"a.bnsh.zs", ZstdCompress(bnsh_a)}, {"b.bnsh", bnsh_b}});
        const ScanOutput out = Scan(ZstdCompress(sarc, false), "p.sarc.zs");
        CheckInvariants(out);
        REQUIRE(out.failures.empty());
        REQUIRE(out.shaders.size() == 2);
        const ExtractedShader* s = FindByCode(out, a.code);
        REQUIRE(s != nullptr);
        CHECK(s->source.find("p.sarc.zs>a.bnsh.zs#BNSH@") == 0);
    }

    SECTION("larger than one streaming buffer") {
        // Several hundred KiB of padding around the BNSH: exercises growing the output buffer.
        Blob big = MakeCode(600 * 1024, 77);
        std::replace(big.begin(), big.end(), u8{'B'}, u8{'b'});
        big.insert(big.end(), bnsh_a.begin(), bnsh_a.end());
        for (const bool sized : {true, false}) {
            const ScanOutput out = Scan(ZstdCompress(big, sized));
            CheckInvariants(out);
            REQUIRE(out.failures.empty());
            REQUIRE(out.shaders.size() == 1);
        }
    }

    SECTION("a frame naming a dictionary") {
        // magic, frame header descriptor (1-byte dictionary id, no single-segment), window
        // descriptor, dictionary id 7, then an empty last raw block.
        const Blob frame{0x28, 0xb5, 0x2f, 0xfd, 0x01, 0x00, 0x07, 0x01, 0x00, 0x00};
        const ScanOutput out = Scan(frame, "dict.zs");
        REQUIRE(out.shaders.empty());
        REQUIRE(out.failures.size() == 1);
        CHECK(out.failures[0].source == "dict.zs");
        CHECK(out.failures[0].reason == "container:zstd-dictionary-required");
    }

    SECTION("truncated and corrupt frames fail with a reason") {
        const Blob zs = ZstdCompress(bnsh_a, false);
        for (const size_t keep : {size_t{5}, zs.size() / 2, zs.size() - 1}) {
            CAPTURE(keep);
            const ScanOutput out = Scan(Blob(zs.begin(), zs.begin() + static_cast<std::ptrdiff_t>(keep)));
            CheckInvariants(out);
            REQUIRE(out.shaders.empty());
            REQUIRE(out.failures.size() == 1);
            CHECK(out.failures[0].reason.rfind("container:zstd-", 0) == 0);
        }
        Blob garbage = MakeCode(300, 4);
        std::memcpy(garbage.data(), zs.data(), 4);
        const ScanOutput out = Scan(garbage);
        CheckInvariants(out);
        REQUIRE(out.shaders.empty());
        REQUIRE(out.failures.size() == 1);
    }
}

TEST_CASE("NVN containers: BAHS entries", "[nvn]") {
    const BahsEntry vs = MakeShader(0, 41);
    const BahsEntry ps = MakeShader(1, 42);
    const BahsEntry cs = MakeShader(5, 43);
    // BAHS controls store 0 at +0x30 (the BNSH ones store their own length).
    std::array<BahsEntry, 3> entries{vs, ps, cs};
    for (BahsEntry& e : entries) {
        Wr32(e.control, 0x30, 0);
    }
    // An entry that is not an NVN2 control section (version 1) is a stray, not a failure.
    BahsEntry legacy = MakeShader(0, 44);
    Wr32(legacy.control, 4, 1);
    const std::array<BahsEntry, 4> with_legacy{entries[0], legacy, entries[1], entries[2]};

    const Blob bahs = MakeBahs(with_legacy);
    const ScanOutput out = Scan(bahs, "agl.sharcb");
    CheckInvariants(out);
    REQUIRE(out.failures.empty());
    REQUIRE(out.shaders.size() == 3);
    for (const ExtractedShader& s : out.shaders) {
        CHECK(s.kind == "bahs");
        CHECK(s.source.find("agl.sharcb#BAHS@0x") == 0);
    }
    const ExtractedShader* s = FindByCode(out, cs.code);
    REQUIRE(s != nullptr);
    CHECK(s->stage_byte == 5);
    CHECK(s->control == entries[2].control);
    CHECK(s->control.size() == ControlSize);

    SECTION("a BNSH copy of the same shader has the same key") {
        const std::array<BnshVariation, 1> v{BnshVariation{ToStage(0, vs)}};
        ScanOutput both;
        ScanBlob(MakeBnsh(v), "a.bnsh", both);
        ScanBlob(bahs, "b.sharcb", both);
        CheckInvariants(both);
        REQUIRE(both.shaders.size() == 3);
        REQUIRE(both.duplicates == 1);
        const ExtractedShader* first = FindByCode(both, vs.code);
        REQUIRE(first != nullptr);
        CHECK(first->kind == "bnsh"); // first occurrence wins
        CHECK(Rd32(first->control, 0x30) == 0xd8);
    }

    SECTION("code region outside the file") {
        Blob bad = bahs;
        Wr32(bad, 0x18, 0x7fffffff);
        const ScanOutput broken = Scan(bad);
        REQUIRE(broken.shaders.empty());
        REQUIRE(HasFailure(broken, "container:bahs-code-region-out-of-bounds"));
    }
}

TEST_CASE("NVN containers: NDSR records", "[nvn]") {
    const BahsEntry vs = MakeShader(0, 51);
    const BahsEntry ps = MakeShader(1, 52);
    const BahsEntry cs = MakeShader(5, 53);
    // A record whose control is not an NVN2 control section holds constant data.
    const BahsEntry constants{MakeCode(0x40, 1), MakeCode(0x40, 2)};
    const std::array<BahsEntry, 4> records{vs, ps, constants, cs};
    const Blob ndsr = MakeNdsr(records);

    const ScanOutput out = Scan(ndsr, "agl_ndsr.pck");
    CheckInvariants(out);
    REQUIRE(out.failures.empty());
    REQUIRE(out.shaders.size() == 3);
    for (const ExtractedShader& s : out.shaders) {
        CHECK(s.kind == "ndsr");
        CHECK(s.source.find("agl_ndsr.pck#NDSR@0x") == 0);
    }
    const ExtractedShader* s = FindByCode(out, cs.code);
    REQUIRE(s != nullptr);
    CHECK(s->stage_byte == 5);
    CHECK(s->control == cs.control);

    SECTION("code offset outside the file") {
        Blob bad = ndsr;
        Wr32(bad, 0x10 + 8, 0x7ffffff0); // first record's code offset
        const ScanOutput broken = Scan(bad);
        CHECK(HasFailure(broken, "container:ndsr-code-out-of-bounds"));
        CHECK(broken.shaders.size() == 2);
        CheckInvariants(broken);
    }
}

TEST_CASE("NVN containers: sead primitive drawer", "[nvn]") {
    const BahsEntry vs = MakeShader(0, 61);
    const BahsEntry ps = MakeShader(1, 62);
    const Blob drawer = MakeSeadPrimitiveDrawer(vs, ps);
    const std::string name = "romfs/Lib/sead/primitive_renderer/primitive_drawer_nvn2_shader.bin";

    const ScanOutput out = Scan(drawer, name);
    CheckInvariants(out);
    REQUIRE(out.failures.empty());
    REQUIRE(out.shaders.size() == 2);
    CHECK(out.shaders[0].kind == "sead");
    CHECK(out.shaders[0].stage_byte == 0);
    CHECK(out.shaders[0].code == vs.code);
    CHECK(out.shaders[0].control == vs.control);
    CHECK(out.shaders[1].stage_byte == 1);
    CHECK(out.shaders[1].code == ps.code);

    SECTION("only by name hint") {
        const ScanOutput other = Scan(drawer, "romfs/Lib/sead/primitive_renderer/other.bin");
        CHECK(other.shaders.empty());
        CHECK(other.failures.empty());
    }

    SECTION("swapped stages and bad offsets are rejected") {
        const Blob swapped = MakeSeadPrimitiveDrawer(ps, vs);
        CHECK(HasFailure(Scan(swapped, name), "container:sead-stage-mismatch"));
        Blob bad = drawer;
        Wr32(bad, 4, 0x7fffff00);
        const ScanOutput broken = Scan(bad, name);
        CHECK(broken.shaders.empty());
        CHECK(HasFailure(broken, "container:sead-bad-control-offset"));
    }
}

TEST_CASE("NVN containers: NSO segments", "[nvn]") {
    const BahsEntry a = MakeShader(0, 71);
    const BahsEntry b = MakeShader(1, 72);
    const std::array<BnshVariation, 1> va{BnshVariation{ToStage(0, a)}};
    const std::array<BnshVariation, 1> vb{BnshVariation{ToStage(4, b)}};

    // The BNSH sits in the data segment (compressed) and another in the rodata segment (raw).
    const Blob nso = MakeNso({Blob(0x200, 0x90), Embed(0x1000, 0x300, MakeBnsh(vb), 3),
                              Embed(0x2000, 0x500, MakeBnsh(va), 4)},
                             {true, false, true});
    const ScanOutput out = Scan(nso, "exefs/main");
    CheckInvariants(out);
    REQUIRE(out.failures.empty());
    REQUIRE(out.shaders.size() == 2);
    for (const ExtractedShader& s : out.shaders) {
        CHECK(s.kind == "nso");
        CHECK(s.source.find("exefs/main>") == 0);
    }
    const ExtractedShader* s = FindByCode(out, a.code);
    REQUIRE(s != nullptr);
    CHECK(s->source.find("exefs/main>data#BNSH@0x500#var0.vs") == 0);
    s = FindByCode(out, b.code);
    REQUIRE(s != nullptr);
    CHECK(s->source.find("exefs/main>rodata#BNSH@0x300#var0.ps") == 0);

    SECTION("a BNSH in the text segment is not scanned") {
        const Blob text_only = MakeNso({Embed(0x1000, 0x300, MakeBnsh(va), 3), Blob(0x100, 1),
                                        Blob(0x100, 2)},
                                       {true, true, true});
        const ScanOutput none = Scan(text_only, "exefs/main");
        CHECK(none.shaders.empty());
        CHECK(none.failures.empty());
    }

    SECTION("damaged NSO files fail with a reason") {
        const ScanOutput truncated_header = Scan(Blob(nso.begin(), nso.begin() + 0x80));
        CHECK(HasFailure(truncated_header, "container:nso-truncated-header"));

        const ScanOutput cut = Scan(Blob(nso.begin(), nso.end() - 0x20));
        CheckInvariants(cut);
        CHECK(HasFailure(cut, "container:nso-segment-out-of-bounds"));

        Blob corrupt = nso;
        // Make the compressed data segment claim a larger decompressed size.
        Wr32(corrupt, 0x38, Rd32(corrupt, 0x38) + 0x100);
        const ScanOutput lz4 = Scan(corrupt);
        CheckInvariants(lz4);
        CHECK(HasFailure(lz4, "container:nso-lz4-corrupt"));
    }
}

TEST_CASE("NVN containers: invalid shaders and damaged tables are reported", "[nvn]") {
    const BahsEntry good = MakeShader(0, 81);

    SECTION("wrong control version") {
        BahsEntry bad = MakeShader(1, 82);
        Wr32(bad.control, 4, 1);
        const std::array<BnshVariation, 1> v{BnshVariation{ToStage(0, good), ToStage(4, bad)}};
        const ScanOutput out = Scan(MakeBnsh(v));
        CheckInvariants(out);
        REQUIRE(out.shaders.size() == 1);
        REQUIRE(out.failures.size() == 1);
        CHECK(out.failures[0].reason.rfind("container:bad-control", 0) == 0);
        CHECK(out.failures[0].source.find("#var0.ps") != std::string::npos);
    }

    SECTION("code area length disagrees with control +0x44") {
        BahsEntry bad = MakeShader(1, 83);
        bad.code.resize(bad.code.size() - 16);
        const std::array<BnshVariation, 1> v{BnshVariation{ToStage(4, bad)}};
        const ScanOutput out = Scan(MakeBnsh(v));
        CHECK(out.shaders.empty());
        CHECK(HasFailure(out, "container:bad-control"));
    }

    SECTION("control section shorter than 0x48") {
        BahsEntry bad = MakeShader(1, 84);
        bad.control.resize(0x20);
        const std::array<BnshVariation, 1> v{BnshVariation{ToStage(4, bad)}};
        const ScanOutput out = Scan(MakeBnsh(v));
        CHECK(out.shaders.empty());
        CHECK(HasFailure(out, "container:bad-control"));
    }

    SECTION("pointers outside the file") {
        const std::array<BnshVariation, 2> v{BnshVariation{ToStage(0, good)},
                                             BnshVariation{ToStage(4, MakeShader(1, 85))}};
        const Blob bnsh = MakeBnsh(v);

        Blob far_table = bnsh;
        std::memset(far_table.data() + 0x80, 0xff, 8);
        const ScanOutput table = Scan(far_table);
        CheckInvariants(table);
        CHECK(table.shaders.empty());
        CHECK(HasFailure(table, "container:bnsh-variation-table-out-of-bounds"));

        Blob far_program = bnsh;
        std::memset(far_program.data() + 0x100 + 0x10, 0xff, 8); // variation 0 program pointer
        const ScanOutput program = Scan(far_program);
        CheckInvariants(program);
        CHECK(program.shaders.size() == 1);
        CHECK(HasFailure(program, "container:bnsh-program-out-of-bounds"));

        Blob huge_count = bnsh;
        Wr32(huge_count, 0x7c, 0xffffffff);
        const ScanOutput many = Scan(huge_count);
        CheckInvariants(many);
        CHECK(many.shaders.size() >= 2);
        CHECK(HasFailure(many, "container:bnsh-"));
        CHECK(many.failures.size() <= 9);
    }

    SECTION("SARC with a member outside the data") {
        Blob sarc = Sarc({{"a.bnsh", MakeCode(64, 1)}});
        // The only node: end offset beyond the file.
        Wr32(sarc, 0x14 + 12 + 12, 0x7ffffff0);
        const ScanOutput out = Scan(sarc, "bad.sarc");
        CHECK(out.shaders.empty());
        CHECK(HasFailure(out, "container:sarc-member-out-of-bounds"));
    }

    SECTION("SARC with a bad byte order mark or layout") {
        Blob sarc = Sarc({{"a", MakeCode(64, 1)}});
        Blob bom = sarc;
        bom[6] = 0x12;
        CHECK(HasFailure(Scan(bom), "container:sarc-bad-bom"));
        Blob layout = sarc;
        Wr32(layout, 0xc, 0x20); // data offset inside the node table
        CHECK(HasFailure(Scan(layout), "container:sarc-bad-layout"));
        Blob sfat = sarc;
        sfat[0x14] = 'X';
        CHECK(HasFailure(Scan(sfat), "container:sarc-bad-sfat"));
    }

    SECTION("nesting deeper than the limit") {
        Blob nested = MakeCode(64, 1);
        for (int i = 0; i < 20; ++i) {
            nested = Sarc({{"n", nested}});
        }
        const ScanOutput out = Scan(nested);
        CHECK(out.shaders.empty());
        CHECK(HasFailure(out, "container:archive-too-deep"));
    }

    SECTION("unrelated data finds nothing and does not fail") {
        for (const size_t size : {size_t{0}, size_t{1}, size_t{3}, size_t{4}, size_t{7}, size_t{4096}}) {
            const ScanOutput out = Scan(MakeCode(size, 5), "junk.bin");
            CHECK(out.shaders.empty());
            CHECK(out.failures.empty());
        }
        const ScanOutput magic_only = Scan(Blob{'S', 'A', 'R', 'C'}, "junk.sarc");
        CHECK(magic_only.shaders.empty());
        CHECK(HasFailure(magic_only, "container:sarc-truncated-header"));
    }
}

TEST_CASE("NVN containers: truncated and corrupt inputs never crash", "[nvn]") {
    const std::vector<Corpus> corpus = BuildCorpus();

    // Every corpus entry is valid and yields shaders.
    for (const Corpus& c : corpus) {
        CAPTURE(c.name);
        const ScanOutput out = Scan(c.data, c.label);
        CheckInvariants(out);
        REQUIRE(out.failures.empty());
        REQUIRE(out.shaders.size() >= 2);
    }

    SECTION("every prefix") {
        for (const Corpus& c : corpus) {
            CAPTURE(c.name);
            // All short prefixes, then a stride that keeps the sweep bounded for larger files.
            const size_t stride = std::max<size_t>(1, c.data.size() / 700);
            for (size_t keep = 0; keep < c.data.size(); keep += keep < 0x180 ? 1 : stride) {
                const ScanOutput out = Scan(Blob(c.data.begin(), c.data.begin() + static_cast<std::ptrdiff_t>(keep)),
                                            c.label);
                CheckInvariants(out);
            }
        }
    }

    SECTION("random byte damage") {
        u32 rng = 0x1234abcd;
        const auto next = [&rng] {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            return rng;
        };
        for (const Corpus& c : corpus) {
            CAPTURE(c.name);
            for (int round = 0; round < 250; ++round) {
                Blob damaged = c.data;
                const u32 hits = 1 + next() % 4;
                for (u32 h = 0; h < hits; ++h) {
                    // Half the damage lands in the first 0x200 bytes (headers, tables).
                    const size_t limit = (next() & 1) ? std::min<size_t>(damaged.size(), 0x200)
                                                      : damaged.size();
                    damaged[next() % limit] = static_cast<u8>(next());
                }
                const ScanOutput out = Scan(damaged, c.label);
                CheckInvariants(out);
            }
        }
    }
}

TEST_CASE("NVN containers: ScanFile and ScanRoot", "[nvn]") {
    namespace fs = std::filesystem;
    const fs::path root = Tests::UniqueTempPath("nvn_containers");
    REQUIRE(fs::create_directories(root));
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{root};

    const auto write = [&](const std::string& rel, const Blob& data) {
        const fs::path path = root / rel;
        fs::create_directories(path.parent_path());
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        REQUIRE(f.good());
    };

    const BahsEntry bfsha_vs = MakeShader(0, 91);
    const BahsEntry agl_ps = MakeShader(1, 92);
    const BahsEntry agl_cs = MakeShader(5, 93);
    const BahsEntry sead_vs = MakeShader(0, 94);
    const BahsEntry sead_ps = MakeShader(1, 95);
    const BahsEntry main_vs = MakeShader(0, 96);
    const BahsEntry skipped = MakeShader(0, 97);
    const BahsEntry elsewhere = MakeShader(0, 98);
    const BahsEntry effect = MakeShader(0, 99);

    const std::array<BnshVariation, 1> v_bfsha{BnshVariation{ToStage(0, bfsha_vs)}};
    const std::array<BnshVariation, 1> v_main{BnshVariation{ToStage(0, main_vs)}};
    const std::array<BnshVariation, 1> v_skipped{BnshVariation{ToStage(0, skipped)}};
    const std::array<BnshVariation, 1> v_else{BnshVariation{ToStage(0, elsewhere)}};
    const std::array<BnshVariation, 1> v_effect{BnshVariation{ToStage(0, effect)}};
    const std::array<BahsEntry, 2> agl_entries{agl_ps, agl_cs};

    write("romfs/Shader/a.bfsha.zs", ZstdCompress(MakeBnsh(v_bfsha), false));
    write("romfs/Lib/agl/agl.sarc.zs",
          ZstdCompress(Sarc({{"agl_technique.sharcb", MakeBahs(agl_entries)},
                             {"agl_ndsr.pck", MakeNdsr(agl_entries)}})));
    write("romfs/Lib/sead/primitive_renderer/primitive_drawer_nvn2_shader.bin",
          MakeSeadPrimitiveDrawer(sead_vs, sead_ps));
    write("romfs/Effect/fx.esetb.byml.zs", ZstdCompress(MakeBnsh(v_effect)));
    write("exefs/main", MakeNso({Blob(0x100, 1), Blob(0x100, 2), Embed(0x1000, 0x80, MakeBnsh(v_main), 1)},
                                {true, true, true}));
    // Not scanned: texture payloads, plain text, a file called main outside exefs.
    write("romfs/Tex/t.bntx.zs", ZstdCompress(MakeBnsh(v_skipped)));
    write("romfs/readme.txt", MakeBnsh(v_skipped));
    write("romfs/other/main", MakeBnsh(v_else));
    // Scanned but broken: reported against its path.
    Blob broken = ZstdCompress(MakeBnsh(v_skipped));
    broken.resize(broken.size() / 2);
    write("romfs/bad.zs", broken);

    ScanOutput out;
    ScanRoot(root, out);
    CheckInvariants(out);

    // bfsha(1) + agl sharcb(2) + agl ndsr(2, duplicates of the BAHS ones) + sead(2) + effect(1)
    // + main(1) = 7 unique shaders
    CHECK(out.shaders.size() == 7);
    CHECK(out.duplicates == 2);
    for (const BahsEntry* s : {&bfsha_vs, &agl_ps, &agl_cs, &sead_vs, &sead_ps, &main_vs, &effect}) {
        CHECK(FindByCode(out, s->code) != nullptr);
    }
    CHECK(FindByCode(out, skipped.code) == nullptr);
    CHECK(FindByCode(out, elsewhere.code) == nullptr);

    const ExtractedShader* s = FindByCode(out, bfsha_vs.code);
    REQUIRE(s != nullptr);
    CHECK(s->source.find("romfs/Shader/a.bfsha.zs#BNSH@0x0#var0.vs") == 0);
    s = FindByCode(out, agl_ps.code);
    REQUIRE(s != nullptr);
    CHECK(s->source.find("romfs/Lib/agl/agl.sarc.zs>agl_") == 0);
    s = FindByCode(out, main_vs.code);
    REQUIRE(s != nullptr);
    CHECK(s->kind == "nso");
    CHECK(s->source.find("exefs/main>data#BNSH@0x80") == 0);

    REQUIRE(out.failures.size() == 1);
    CHECK(out.failures[0].source == "romfs/bad.zs");
    CHECK(out.failures[0].reason.rfind("container:zstd-", 0) == 0);

    SECTION("ScanFile labels and reads whole files") {
        ScanOutput one;
        REQUIRE(ScanFile(root / "romfs/Shader/a.bfsha.zs", "label.zs", one));
        REQUIRE(one.shaders.size() == 1);
        CHECK(one.shaders[0].source.find("label.zs#BNSH@") == 0);

        ScanOutput missing;
        CHECK_FALSE(ScanFile(root / "no/such/file.bin", "missing.bin", missing));
        REQUIRE(missing.failures.size() == 1);
        CHECK(missing.failures[0].source == "missing.bin");
        CHECK(missing.failures[0].reason.rfind("container:", 0) == 0);
    }

    SECTION("a missing root is reported") {
        ScanOutput none;
        ScanRoot(root / "nowhere", none);
        CHECK(none.shaders.empty());
        REQUIRE(none.failures.size() == 1);
        CHECK(none.failures[0].reason.rfind("container:directory-walk", 0) == 0);
    }
}

} // namespace VideoCommon::Nvn::Containers
