// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include <zstd.h>

#include "common/lz4_compression.h"
#include "nvn_shader_containers.h"

namespace VideoCommon::Nvn::Containers {

static_assert(std::endian::native == std::endian::little, "containers are read as little endian");

namespace {

using Bytes = std::span<const u8>;

constexpr size_t MaxDecompressed = size_t{1} << 30; // zstd / LZ4 output cap
constexpr int MaxDepth = 16;                         // nested archives (nxnative NEST)
constexpr u32 ZstdFrameMagic = 0xFD2FB528;
constexpr std::string_view SeadNameHint = "primitive_drawer_nvn2_shader";

// ---------------------------------------------------------------------------------------------
// Little helpers (every read is bounds-checked by the caller through Fits/Slice).
// ---------------------------------------------------------------------------------------------

u16 Rd16(Bytes d, size_t off) {
    u16 v;
    std::memcpy(&v, d.data() + off, sizeof(v));
    return v;
}

u32 Rd32(Bytes d, size_t off) {
    u32 v;
    std::memcpy(&v, d.data() + off, sizeof(v));
    return v;
}

u64 Rd64(Bytes d, size_t off) {
    u64 v;
    std::memcpy(&v, d.data() + off, sizeof(v));
    return v;
}

constexpr u16 Swap16(u16 v) {
    return static_cast<u16>((v >> 8) | (v << 8));
}

constexpr u32 Swap32(u32 v) {
    return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}

/// [off, off + n) lies inside a buffer of `size` bytes (no overflow for any u64 input).
constexpr bool Fits(size_t size, u64 off, u64 n) {
    return off <= size && n <= size - off;
}

bool Slice(Bytes d, u64 off, u64 n, Bytes& out) {
    if (!Fits(d.size(), off, n)) {
        return false;
    }
    out = d.subspan(static_cast<size_t>(off), static_cast<size_t>(n));
    return true;
}

bool HasMagic(Bytes d, size_t off, const char (&magic)[5]) {
    return Fits(d.size(), off, 4) && std::memcmp(d.data() + off, magic, 4) == 0;
}

std::string Hex(u64 v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
    return buf;
}

std::string Fmt2(const char* prefix, u64 a, u64 b) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s 0x%llx vs 0x%llx", prefix,
                  static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
    return buf;
}

bool IsZstd(Bytes d) {
    return d.size() >= 4 && Rd32(d, 0) == ZstdFrameMagic;
}

bool IsNvn2Control(Bytes c) {
    return c.size() >= ControlKeyBytes && Rd32(c, 0) == ControlMagic &&
           Rd32(c, 4) == ControlVersion;
}

// ---------------------------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------------------------

class Sha256 {
public:
    void Update(const u8* data, size_t size) {
        total += size;
        if (fill != 0) {
            const size_t take = std::min(size, block.size() - fill);
            std::memcpy(block.data() + fill, data, take);
            fill += take;
            data += take;
            size -= take;
            if (fill == block.size()) {
                Compress(block.data());
                fill = 0;
            }
        }
        while (size >= block.size()) {
            Compress(data);
            data += block.size();
            size -= block.size();
        }
        if (size != 0) {
            std::memcpy(block.data(), data, size);
            fill = size;
        }
    }

    std::array<u8, 32> Finish() {
        const u64 bits = total * 8;
        const u8 pad = 0x80;
        Update(&pad, 1);
        const u8 zero = 0;
        while (fill != 56) {
            Update(&zero, 1);
        }
        std::array<u8, 8> length;
        for (size_t i = 0; i < 8; ++i) {
            length[i] = static_cast<u8>(bits >> (56 - 8 * i));
        }
        Update(length.data(), length.size());
        std::array<u8, 32> digest;
        for (size_t i = 0; i < 8; ++i) {
            digest[i * 4 + 0] = static_cast<u8>(state[i] >> 24);
            digest[i * 4 + 1] = static_cast<u8>(state[i] >> 16);
            digest[i * 4 + 2] = static_cast<u8>(state[i] >> 8);
            digest[i * 4 + 3] = static_cast<u8>(state[i]);
        }
        return digest;
    }

private:
    static constexpr u32 Ror(u32 v, int n) {
        return (v >> n) | (v << (32 - n));
    }

    void Compress(const u8* p) {
        static constexpr std::array<u32, 64> k{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::array<u32, 64> w;
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (u32{p[i * 4]} << 24) | (u32{p[i * 4 + 1]} << 16) | (u32{p[i * 4 + 2]} << 8) |
                   u32{p[i * 4 + 3]};
        }
        for (size_t i = 16; i < 64; ++i) {
            const u32 s0 = Ror(w[i - 15], 7) ^ Ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const u32 s1 = Ror(w[i - 2], 17) ^ Ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        u32 a = state[0], b = state[1], c = state[2], d = state[3];
        u32 e = state[4], f = state[5], g = state[6], h = state[7];
        for (size_t i = 0; i < 64; ++i) {
            const u32 s1 = Ror(e, 6) ^ Ror(e, 11) ^ Ror(e, 25);
            const u32 ch = (e & f) ^ (~e & g);
            const u32 t1 = h + s1 + ch + k[i] + w[i];
            const u32 s0 = Ror(a, 2) ^ Ror(a, 13) ^ Ror(a, 22);
            const u32 maj = (a & b) ^ (a & c) ^ (b & c);
            const u32 t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::array<u32, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<u8, 64> block{};
    size_t fill = 0;
    u64 total = 0;
};

std::string ToHex(const u8* data, size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s(size * 2, '0');
    for (size_t i = 0; i < size; ++i) {
        s[i * 2] = digits[data[i] >> 4];
        s[i * 2 + 1] = digits[data[i] & 15];
    }
    return s;
}

// ---------------------------------------------------------------------------------------------
// zstd (streaming: real frames often carry no content size, and files may hold several frames)
// ---------------------------------------------------------------------------------------------

struct DCtxDeleter {
    void operator()(ZSTD_DCtx* ctx) const {
        ZSTD_freeDCtx(ctx);
    }
};

bool Unzstd(Bytes in, std::vector<u8>& out, std::string& why) {
    std::unique_ptr<ZSTD_DCtx, DCtxDeleter> dctx(ZSTD_createDCtx());
    if (!dctx) {
        why = "container:zstd-out-of-memory";
        return false;
    }
    out.clear();
    size_t used = 0;
    size_t pos = 0;
    bool any_frame = false;
    while (pos < in.size()) {
        const u8* frame = in.data() + pos;
        const size_t remaining = in.size() - pos;
        if (remaining < 4 || Rd32(in, pos) != ZstdFrameMagic) {
            break; // trailing bytes that are not a frame
        }
        if (ZSTD_getDictID_fromFrame(frame, remaining) != 0) {
            why = "container:zstd-dictionary-required";
            return false;
        }
        const unsigned long long content = ZSTD_getFrameContentSize(frame, remaining);
        if (content != ZSTD_CONTENTSIZE_UNKNOWN && content != ZSTD_CONTENTSIZE_ERROR) {
            if (content > MaxDecompressed || used + content > MaxDecompressed) {
                why = "container:zstd-too-large";
                return false;
            }
            // Pre-size from the header, but never trust it for more than 64 MiB up front: a
            // damaged size must not allocate a gigabyte before the data proves it.
            const size_t presize = static_cast<size_t>(std::min<unsigned long long>(content, 64ull << 20));
            if (out.size() < used + presize) {
                out.resize(used + presize);
            }
        }
        ZSTD_inBuffer input{frame, remaining, 0};
        for (;;) {
            if (out.size() == used) {
                // Double, but never past the cap plus one block of room: an output that fits is
                // not rejected for the size of the step towards it.
                const size_t cap = MaxDecompressed + ZSTD_DStreamOutSize();
                if (used >= cap) {
                    why = "container:zstd-too-large";
                    return false;
                }
                const size_t grow =
                    std::min(std::max<size_t>(ZSTD_DStreamOutSize(), out.size()), cap - used);
                out.resize(used + grow);
            }
            ZSTD_outBuffer output{out.data(), out.size(), used};
            const size_t ret = ZSTD_decompressStream(dctx.get(), &output, &input);
            if (ZSTD_isError(ret)) {
                why = std::string("container:zstd-corrupt: ") + ZSTD_getErrorName(ret);
                return false;
            }
            used = output.pos;
            if (used > MaxDecompressed) {
                why = "container:zstd-too-large";
                return false;
            }
            if (ret == 0) {
                break; // frame complete
            }
            if (output.pos == output.size) {
                continue; // needs more room
            }
            if (input.pos == input.size) {
                why = "container:zstd-truncated";
                return false;
            }
        }
        any_frame = true;
        pos += input.pos;
    }
    if (!any_frame) {
        why = "container:zstd-corrupt: no frame";
        return false;
    }
    out.resize(used);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Container scanner
// ---------------------------------------------------------------------------------------------

constexpr std::array<const char*, 6> NvnStageNames{"vs", "ps", "gs", "hs", "ds", "cs"};
constexpr std::array<const char*, 6> BnshSlotNames{"vs", "hs", "ds", "gs", "ps", "cs"};

class Scanner {
public:
    explicit Scanner(ScanOutput& out_) : out{out_} {}

    void Scan(Bytes d, const std::string& label, int depth) {
        if (depth > MaxDepth) {
            Fail(label, "container:archive-too-deep");
            return;
        }
        std::vector<u8> owned;
        if (IsZstd(d)) {
            std::string why;
            if (!Unzstd(d, owned, why)) {
                Fail(label, std::move(why));
                return;
            }
            if (depth > 0) {
                expand_left -= static_cast<s64>(owned.size());
                if (expand_left < 0) {
                    Fail(label, "container:archive-expansion-budget");
                    return;
                }
            }
            d = owned;
        }
        if (depth == 0) {
            budget = static_cast<s64>(MaxDepth) * static_cast<s64>(d.size());
            expand_left = budget;
        }
        if (HasMagic(d, 0, "SARC")) {
            Sarc(d, label, depth);
        } else if (HasMagic(d, 0, "BAHS")) {
            Bahs(d, label);
        } else if (HasMagic(d, 0, "NDSR")) {
            Ndsr(d, label);
        } else if (HasMagic(d, 0, "NSO0")) {
            Nso(d, label);
        } else if (label.find(SeadNameHint) != std::string::npos && Sead(d, label)) {
            return;
        } else {
            FindBnsh(d, label, "bnsh");
        }
    }

private:
    void Fail(const std::string& source, std::string reason) {
        out.failures.push_back({source, std::move(reason)});
    }

    static std::string Check(Bytes ctl, Bytes code) {
        if (ctl.size() < ControlKeyBytes) {
            return Fmt2("container:bad-control: control shorter than 0x48:", ctl.size(),
                        ControlKeyBytes);
        }
        if (Rd32(ctl, 0) != ControlMagic) {
            return Fmt2("container:bad-control: magic", Rd32(ctl, 0), ControlMagic);
        }
        if (Rd32(ctl, 4) != ControlVersion) {
            return Fmt2("container:bad-control: version", Rd32(ctl, 4), ControlVersion);
        }
        if (Rd32(ctl, 0x44) != code.size()) {
            return Fmt2("container:bad-control: control +0x44 does not match the code area:",
                        Rd32(ctl, 0x44), code.size());
        }
        return {};
    }

    /// Validates, dedupes and records one shader. Returns false when validation failed.
    bool Add(const std::string& source, const char* kind, Bytes ctl, Bytes code) {
        if (std::string why = Check(ctl, code); !why.empty()) {
            Fail(source, std::move(why));
            return false;
        }
        std::string key = ShaderKey(ctl, code);
        if (!out.seen_keys.insert(key).second) {
            ++out.duplicates;
            return true;
        }
        ExtractedShader shader;
        shader.source = source;
        shader.kind = kind;
        shader.stage_byte = ctl.size() > 0x56 ? ctl[0x56] : 0;
        shader.control.assign(ctl.begin(), ctl.end());
        shader.code.assign(code.begin(), code.end());
        shader.key = std::move(key);
        out.shaders.push_back(std::move(shader));
        return true;
    }

    // ---- SARC ---------------------------------------------------------------------------------

    void Sarc(Bytes d, const std::string& label, int depth) {
        if (d.size() < 0x14) {
            Fail(label, "container:sarc-truncated-header");
            return;
        }
        const u16 bom_le = Rd16(d, 6);
        bool be;
        if (bom_le == 0xFEFF) {
            be = false;
        } else if (bom_le == 0xFFFE) {
            be = true;
        } else {
            Fail(label, "container:sarc-bad-bom");
            return;
        }
        const auto u16_at = [&](size_t off) { return be ? Swap16(Rd16(d, off)) : Rd16(d, off); };
        const auto u32_at = [&](size_t off) { return be ? Swap32(Rd32(d, off)) : Rd32(d, off); };

        const size_t header_len = u16_at(4);
        const size_t data_off = u32_at(0xc);
        if (header_len < 0x14 || !Fits(d.size(), header_len, 12) ||
            !HasMagic(d, header_len, "SFAT")) {
            Fail(label, "container:sarc-bad-sfat");
            return;
        }
        const size_t count = u16_at(header_len + 6);
        const size_t nodes = header_len + 12;
        const size_t sfnt = nodes + 16 * count;
        const size_t names = sfnt + 8;
        if (!Fits(d.size(), nodes, 16 * count + 8) || !HasMagic(d, sfnt, "SFNT")) {
            Fail(label, "container:sarc-bad-sfnt");
            return;
        }
        if (!(names <= data_off && data_off <= d.size())) {
            Fail(label, Fmt2("container:sarc-bad-layout: data offset / names end", data_off, names));
            return;
        }
        const size_t data_size = d.size() - data_off;
        for (size_t i = 0; i < count; ++i) {
            const size_t node = nodes + 16 * i;
            const u32 attr = u32_at(node + 4);
            const u64 start = u32_at(node + 8);
            const u64 end = u32_at(node + 12);
            std::string name = "#" + std::to_string(i);
            if (attr >> 24) {
                // The name is NUL terminated inside the names block (which ends at the data).
                const u64 name_off = names + u64{attr & 0xffffff} * 4;
                if (name_off < data_off) {
                    const char* first = reinterpret_cast<const char*>(d.data() + name_off);
                    const size_t max_len = std::min<size_t>(data_off - name_off, 256);
                    const void* nul = std::memchr(first, 0, max_len);
                    name.assign(first,
                                nul ? static_cast<size_t>(static_cast<const char*>(nul) - first)
                                    : max_len);
                }
            }
            const std::string member_label = label + ">" + name;
            if (!(start <= end && end <= data_size)) {
                Fail(member_label, Fmt2("container:sarc-member-out-of-bounds: end vs data size",
                                        end, data_size));
                continue;
            }
            const Bytes member = d.subspan(data_off + static_cast<size_t>(start),
                                           static_cast<size_t>(end - start));
            budget -= static_cast<s64>(member.size());
            if (budget < 0) {
                Fail(member_label, "container:archive-budget-exceeded");
                return;
            }
            Scan(member, member_label, depth + 1);
        }
    }

    // ---- BNSH ---------------------------------------------------------------------------------

    void FindBnsh(Bytes d, const std::string& label, const char* kind) {
        static constexpr u8 needle[8] = {'B', 'N', 'S', 'H', 0, 0, 0, 0};
        const u8* const begin = d.data();
        const u8* const end = begin + d.size();
        const u8* p = begin;
        while (p != nullptr && p < end) {
            p = static_cast<const u8*>(std::memchr(p, 'B', static_cast<size_t>(end - p)));
            if (p == nullptr) {
                break;
            }
            if (static_cast<size_t>(end - p) < 8 || std::memcmp(p, needle, 8) != 0) {
                ++p;
                continue;
            }
            const size_t base = static_cast<size_t>(p - begin);
            p += 8;
            if (!Fits(d.size(), base, 0x20)) {
                continue; // too near the end for a BNSH header
            }
            const u32 size = Rd32(d, base + 0x1c);
            if (!Fits(d.size(), base, size) || !HasMagic(d, base + 0x60, "grsc")) {
                continue;
            }
            Bnsh(d.subspan(base), label + "#BNSH@" + Hex(base), kind);
        }
    }

    /// `b` starts at the BNSH header; every pointer is relative to it and bounded by the bytes that
    /// follow (like nxnative, not by the size the header claims).
    void Bnsh(Bytes b, const std::string& where, const char* kind) {
        if (!Fits(b.size(), 0x7c, 12)) {
            Fail(where, "container:bnsh-truncated-header");
            return;
        }
        const u32 count = Rd32(b, 0x7c);
        const u64 table = Rd64(b, 0x80);
        if (count != 0 && !Fits(b.size(), table, 0)) {
            Fail(where, Fmt2("container:bnsh-variation-table-out-of-bounds: table vs size", table,
                             b.size()));
            return;
        }
        // A damaged count or table must not flood the report: stop after a few failures.
        constexpr size_t MaxErrors = 8;
        size_t errors = 0;
        const auto fail = [&](const std::string& source, std::string reason) {
            ++errors;
            Fail(source, std::move(reason));
        };
        for (u32 v = 0; v < count; ++v) {
            if (errors >= MaxErrors) {
                Fail(where, "container:bnsh-too-many-errors");
                return;
            }
            const std::string var = where + "#var" + std::to_string(v);
            const u64 var_off = table + 0x40ull * v;
            Bytes var_bytes;
            if (!Slice(b, var_off, 0x40, var_bytes)) {
                fail(var, Fmt2("container:bnsh-variation-out-of-bounds: offset vs size", var_off,
                               b.size()));
                return; // later variations lie beyond this one
            }
            const u64 program = Rd64(var_bytes, 0x10);
            if (program == 0) {
                continue;
            }
            Bytes program_bytes;
            if (!Slice(b, program, 0x38, program_bytes)) {
                fail(var, Fmt2("container:bnsh-program-out-of-bounds: offset vs size", program,
                               b.size()));
                continue;
            }
            for (u32 slot = 0; slot < 6; ++slot) {
                const u64 stage = Rd64(program_bytes, 8 + 8 * slot);
                if (stage == 0) {
                    continue;
                }
                const std::string stage_label = var + "." + BnshSlotNames[slot];
                Bytes desc;
                if (!Slice(b, stage, 0x20, desc)) {
                    fail(stage_label,
                         Fmt2("container:bnsh-stage-out-of-bounds: offset vs size", stage, b.size()));
                    continue;
                }
                const u64 control_ptr = Rd64(desc, 8);
                const u64 code_ptr = Rd64(desc, 0x10);
                const u32 code_size = Rd32(desc, 0x18);
                const u32 control_size = Rd32(desc, 0x1c);
                Bytes control;
                Bytes code;
                if (!Slice(b, control_ptr, control_size, control) ||
                    !Slice(b, code_ptr, code_size, code)) {
                    fail(stage_label,
                         Fmt2("container:bnsh-blob-out-of-bounds: control/code end vs size",
                              std::max(control_ptr, code_ptr), b.size()));
                    continue;
                }
                std::string name = var + ".";
                name += control.size() > 0x56 && control[0x56] < NvnStageNames.size()
                            ? NvnStageNames[control[0x56]]
                            : BnshSlotNames[slot];
                if (!Add(name, kind, control, code)) {
                    ++errors;
                }
            }
        }
    }

    // ---- BAHS (agl sharcb) --------------------------------------------------------------------

    void Bahs(Bytes d, const std::string& label) {
        if (d.size() < 0x20) {
            Fail(label, "container:bahs-truncated-header");
            return;
        }
        const u32 region_size = Rd32(d, 0x18);
        const u32 region = Rd32(d, 0x1c);
        if (u64{region} + region_size > d.size()) {
            Fail(label, Fmt2("container:bahs-code-region-out-of-bounds: end vs size",
                             u64{region} + region_size, d.size()));
            return;
        }
        // Entries are found by scanning every aligned word of the table area (nxnative
        // bahs_shaders): {code offset, 0, control size, signed offset from the entry to the control}.
        // Words that merely look like an entry but do not describe a valid control section are
        // strays and are skipped silently.
        for (u64 a = 0; a + 16 < region; a += 4) {
            const u32 off = Rd32(d, a);
            const u32 zero = Rd32(d, a + 4);
            const u32 control_size = Rd32(d, a + 8);
            const s64 rel = static_cast<s32>(Rd32(d, a + 12));
            if (zero != 0 || off % 0x100 != 0 || control_size < ControlKeyBytes ||
                control_size >= 0x10000) {
                continue;
            }
            const s64 control_off = static_cast<s64>(a) + rel;
            if (control_off < 0 || static_cast<u64>(control_off) + control_size > region) {
                continue;
            }
            const Bytes control = d.subspan(static_cast<size_t>(control_off), control_size);
            if (!IsNvn2Control(control)) {
                continue;
            }
            const u32 code_size = Rd32(control, 0x44);
            if (u64{off} + code_size > region_size) {
                continue;
            }
            Add(label + "#BAHS@" + Hex(a), "bahs", control,
                d.subspan(static_cast<size_t>(region) + off, code_size));
        }
    }

    // ---- NDSR (agl_ndsr.pck) ------------------------------------------------------------------

    void Ndsr(Bytes d, const std::string& label) {
        if (d.size() < 0x20) {
            Fail(label, "container:ndsr-truncated-header");
            return;
        }
        for (size_t o = 0x10; o + 16 <= d.size(); o += 16) {
            const u32 control_off = Rd32(d, o);
            const u32 control_len = Rd32(d, o + 4);
            const u32 code_off = Rd32(d, o + 8);
            if (control_off == 0 && control_len == 0) {
                break;
            }
            if (!Fits(d.size(), control_off, ControlKeyBytes) ||
                !IsNvn2Control(d.subspan(control_off))) {
                continue; // constant data record
            }
            const std::string where = label + "#NDSR@" + Hex(control_off);
            const u64 len = std::max<u64>(control_len >> 4, ControlKeyBytes);
            Bytes control;
            if (!Slice(d, control_off, len, control)) {
                Fail(where, Fmt2("container:ndsr-control-out-of-bounds: end vs size",
                                 u64{control_off} + len, d.size()));
                continue;
            }
            Bytes code;
            if (!Slice(d, code_off, Rd32(control, 0x44), code)) {
                Fail(where, Fmt2("container:ndsr-code-out-of-bounds: end vs size",
                                 u64{code_off} + Rd32(control, 0x44), d.size()));
                continue;
            }
            Add(where, "ndsr", control, code);
        }
    }

    // ---- sead primitive drawer ----------------------------------------------------------------

    bool Sead(Bytes d, const std::string& label) {
        if (d.size() < 0x10) {
            Fail(label, "container:sead-truncated-header");
            return false;
        }
        struct Part {
            Bytes control;
            Bytes code;
            u32 control_off;
        };
        std::array<Part, 2> parts;
        for (size_t i = 0; i < 2; ++i) {
            const u32 control_off = Rd32(d, 4 * i);
            const u32 code_off = Rd32(d, 8 + 4 * i);
            if (control_off < 0x10 || !Fits(d.size(), control_off, ControlKeyBytes) ||
                !IsNvn2Control(d.subspan(control_off))) {
                Fail(label, "container:sead-bad-control-offset");
                return false;
            }
            const u64 own = Rd32(d, control_off + 0x30);
            Bytes control;
            if (!Slice(d, control_off, std::max<u64>(own, ControlKeyBytes), control)) {
                Fail(label, "container:sead-control-out-of-bounds");
                return false;
            }
            Bytes code;
            if (!Slice(d, code_off, Rd32(control, 0x44), code)) {
                Fail(label, "container:sead-code-out-of-bounds");
                return false;
            }
            if (control.size() <= 0x56 || control[0x56] != i) {
                Fail(label, "container:sead-stage-mismatch");
                return false;
            }
            parts[i] = {control, code, control_off};
        }
        for (const Part& part : parts) {
            Add(label + "#SEAD@" + Hex(part.control_off), "sead", part.control, part.code);
        }
        return true;
    }

    // ---- NSO (exefs/main) ---------------------------------------------------------------------

    void Nso(Bytes d, const std::string& label) {
        if (d.size() < 0x100) {
            Fail(label, "container:nso-truncated-header");
            return;
        }
        const u32 flags = Rd32(d, 0xc);
        if ((flags & 0x80) != 0) {
            Fail(label, "container:nso-zbic-unsupported");
            return;
        }
        static constexpr std::array<const char*, 3> names{"text", "rodata", "data"};
        // Shaders live in the read-only and data segments; the text segment is skipped.
        for (u32 i = 1; i < 3; ++i) {
            const std::string where = label + ">" + names[i];
            const u32 file_off = Rd32(d, 0x10 + 0x10 * i);
            const u32 size = Rd32(d, 0x18 + 0x10 * i);
            const u32 compressed_size = Rd32(d, 0x60 + 4 * i);
            const bool compressed = ((flags >> i) & 1) != 0;
            const u32 stored = compressed ? compressed_size : size;
            if (size > MaxDecompressed || stored > 0x7fffffff) {
                Fail(where, "container:nso-segment-too-large");
                continue;
            }
            Bytes raw;
            if (!Slice(d, file_off, stored, raw)) {
                Fail(where, Fmt2("container:nso-segment-out-of-bounds: end vs size",
                                 u64{file_off} + stored, d.size()));
                continue;
            }
            if (!compressed) {
                FindBnsh(raw, where, "nso");
                continue;
            }
            std::vector<u8> segment(size);
            const int decoded =
                Common::Compression::DecompressDataLZ4(segment.data(), segment.size(), raw.data(),
                                                       raw.size());
            if (decoded < 0 || static_cast<u32>(decoded) != size) {
                Fail(where, "container:nso-lz4-corrupt");
                continue;
            }
            FindBnsh(segment, where, "nso");
        }
    }

    ScanOutput& out;
    s64 budget = 0;      // bytes of archive members still allowed (NEST x the file)
    s64 expand_left = 0; // bytes nested zstd members may still expand to
};

bool ShouldScan(const std::filesystem::path& path) {
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (name == "main") {
        return path.parent_path().filename() == "exefs";
    }
    const auto ends_with = [&](std::string_view suffix) {
        return name.size() >= suffix.size() &&
               std::string_view{name}.substr(name.size() - suffix.size()) == suffix;
    };
    if (ends_with(".zs")) {
        // The compressed payload's own extension. Textures, models, audio, events, physics and the
        // generic .pack archives held no shader in the Donkey Kong Bananza dump (checked by
        // scanning them all); everything else (.blarc layouts, .esetb.byml effects, .sarc,
        // .bfsha, ...) is scanned. Note .byml.zs must stay: effects are .esetb.byml.zs.
        static constexpr std::array<std::string_view, 13> skipped{
            ".bntx.zs",  ".bwav.zs",  ".bars.zs",  ".alsarc.zs", ".bfres.zs",
            ".asb.zs",   ".baev.zs",  ".bfevfl.zs", ".bphsh.zs",  ".bphsc.zs",
            ".vmat.zs",  ".genvb.zs", ".pack.zs"};
        return std::none_of(skipped.begin(), skipped.end(),
                            [&](std::string_view s) { return ends_with(s); });
    }
    static constexpr std::array<std::string_view, 7> extensions{
        ".bfsha", ".bnsh", ".sarc", ".pck", ".bin", ".bahs", ".sharcb"};
    return std::any_of(extensions.begin(), extensions.end(),
                       [&](std::string_view s) { return ends_with(s); });
}

struct CCtxDeleter {
    void operator()(ZSTD_CCtx* ctx) const {
        ZSTD_freeCCtx(ctx);
    }
};

} // namespace

// -------------------------------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------------------------------

std::string Sha256Hex(std::span<const u8> data) {
    Sha256 sha;
    sha.Update(data.data(), data.size());
    const auto digest = sha.Finish();
    return ToHex(digest.data(), digest.size());
}

std::string ShaderKey(std::span<const u8> control, std::span<const u8> code) {
    if (control.size() < ControlKeyBytes) {
        return {};
    }
    std::array<u8, ControlKeyBytes> head;
    std::memcpy(head.data(), control.data(), head.size());
    std::memset(head.data() + 0x30, 0, 4);
    Sha256 sha;
    sha.Update(head.data(), head.size());
    sha.Update(code.data(), code.size());
    const auto digest = sha.Finish();
    return ToHex(digest.data(), 16);
}

void ScanBlob(std::span<const u8> blob, const std::string& label, ScanOutput& out) {
    try {
        Scanner scanner{out};
        scanner.Scan(blob, label, 0);
    } catch (const std::exception& e) {
        out.failures.push_back({label, std::string("container:exception: ") + e.what()});
    } catch (...) {
        out.failures.push_back({label, "container:exception"});
    }
}

bool ScanFile(const std::filesystem::path& file, const std::string& label, ScanOutput& out) {
    try {
        std::ifstream stream(file, std::ios::binary | std::ios::ate);
        if (!stream) {
            out.failures.push_back({label, "container:unreadable-file"});
            return false;
        }
        const std::streamoff size = stream.tellg();
        if (size < 0) {
            out.failures.push_back({label, "container:unreadable-file"});
            return false;
        }
        std::vector<u8> data(static_cast<size_t>(size));
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(data.data()), size);
        if (!stream && size != 0) {
            out.failures.push_back({label, "container:unreadable-file"});
            return false;
        }
        ScanBlob(data, label, out);
        return true;
    } catch (const std::exception& e) {
        out.failures.push_back({label, std::string("container:exception: ") + e.what()});
        return false;
    }
}

void ScanRoot(const std::filesystem::path& root, ScanOutput& out) {
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(
             root, std::filesystem::directory_options::skip_permission_denied, ec),
         end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code type_ec;
        if (it->is_regular_file(type_ec) && ShouldScan(it->path())) {
            files.push_back(it->path());
        }
    }
    if (ec) {
        out.failures.push_back({root.generic_string(), "container:directory-walk: " + ec.message()});
    }
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
        std::error_code rel_ec;
        const auto rel = std::filesystem::relative(file, root, rel_ec);
        ScanFile(file, (rel_ec ? file : rel).generic_string(), out);
    }
}

// -------------------------------------------------------------------------------------------------
// Builders
// -------------------------------------------------------------------------------------------------

namespace {

void Put16(std::vector<u8>& v, size_t off, u16 x, bool be = false) {
    if (v.size() < off + 2) {
        v.resize(off + 2);
    }
    if (be) {
        x = Swap16(x);
    }
    std::memcpy(v.data() + off, &x, 2);
}

void Put32(std::vector<u8>& v, size_t off, u32 x, bool be = false) {
    if (v.size() < off + 4) {
        v.resize(off + 4);
    }
    if (be) {
        x = Swap32(x);
    }
    std::memcpy(v.data() + off, &x, 4);
}

void Put64(std::vector<u8>& v, size_t off, u64 x) {
    if (v.size() < off + 8) {
        v.resize(off + 8);
    }
    std::memcpy(v.data() + off, &x, 8);
}

size_t AlignUp(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

/// Appends `data` at the next multiple of `align`, returns its offset.
size_t Append(std::vector<u8>& v, Bytes data, size_t align) {
    const size_t at = AlignUp(v.size(), align);
    v.resize(at + data.size());
    if (!data.empty()) {
        std::memcpy(v.data() + at, data.data(), data.size());
    }
    return at;
}

} // namespace

std::vector<u8> MakeControl(u32 stage_byte, u32 code_bytes_total, u32 inst_bytes) {
    std::vector<u8> c(ControlSize);
    Put32(c, 0, ControlMagic);
    Put32(c, 4, ControlVersion);
    Put32(c, 0x30, ControlSize);
    Put32(c, 0x38, inst_bytes);
    Put32(c, 0x40, static_cast<u32>(AlignUp(inst_bytes, 0x100)));
    Put32(c, 0x44, code_bytes_total);
    Put32(c, 0x4c, 16);
    c[0x56] = static_cast<u8>(stage_byte);
    if (stage_byte == 5) {
        Put32(c, 0x58, 32);
        Put32(c, 0x5c, 1);
        Put32(c, 0x60, 1);
    }
    return c;
}

std::vector<u8> MakeCode(size_t size, u32 seed) {
    std::vector<u8> code(size);
    u32 x = seed * 2654435761u + 0x9e3779b9u;
    for (u8& b : code) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        b = static_cast<u8>(x >> 11);
    }
    return code;
}

std::vector<u8> MakeBnsh(std::span<const BnshVariation> variations) {
    std::vector<u8> f(0x100);
    std::memcpy(f.data(), "BNSH", 4); // followed by four zero bytes
    std::memcpy(f.data() + 0x60, "grsc", 4);
    Put32(f, 0x7c, static_cast<u32>(variations.size()));
    Put64(f, 0x80, 0x100);

    const size_t table = f.size();
    f.resize(table + 0x40 * variations.size());
    for (size_t v = 0; v < variations.size(); ++v) {
        const BnshVariation& variation = variations[v];
        if (variation.empty()) {
            continue;
        }
        const size_t program = AlignUp(f.size(), 0x10);
        f.resize(program + 0x40);
        Put64(f, table + 0x40 * v + 0x10, program);
        for (const BnshStage& stage : variation) {
            const size_t desc = AlignUp(f.size(), 0x10);
            f.resize(desc + 0x20);
            Put64(f, program + 8 + 8 * stage.slot, desc);
            const size_t control = Append(f, stage.control, 0x10);
            const size_t code = Append(f, stage.code, 0x100);
            Put64(f, desc + 8, control);
            Put64(f, desc + 0x10, code);
            Put32(f, desc + 0x18, static_cast<u32>(stage.code.size()));
            Put32(f, desc + 0x1c, static_cast<u32>(stage.control.size()));
        }
    }
    Put32(f, 0x1c, static_cast<u32>(f.size()));
    return f;
}

std::vector<u8> MakeSarc(std::span<const std::pair<std::string, std::vector<u8>>> members,
                         bool big_endian) {
    const bool be = big_endian;
    struct Node {
        u32 hash;
        size_t index;
    };
    std::vector<Node> nodes;
    for (size_t i = 0; i < members.size(); ++i) {
        u32 h = 0;
        for (const char c : members[i].first) {
            h = h * 0x65 + static_cast<u32>(static_cast<signed char>(c));
        }
        nodes.push_back({h, i});
    }
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const Node& a, const Node& b) { return a.hash < b.hash; });

    std::vector<u8> f(0x14);
    std::memcpy(f.data(), "SARC", 4);
    Put16(f, 4, 0x14, be);
    Put16(f, 6, 0xFEFF, false); // BOM bytes are FF FE (LE) or FE FF (BE)
    if (be) {
        f[6] = 0xFE;
        f[7] = 0xFF;
    }
    Put32(f, 0x10, 0x100, be);

    f.resize(0x14 + 12 + 16 * nodes.size());
    std::memcpy(f.data() + 0x14, "SFAT", 4);
    Put16(f, 0x14 + 4, 0xc, be);
    Put16(f, 0x14 + 6, static_cast<u16>(nodes.size()), be);
    Put32(f, 0x14 + 8, 0x65, be);

    const size_t sfnt = f.size();
    f.resize(sfnt + 8);
    std::memcpy(f.data() + sfnt, "SFNT", 4);
    Put16(f, sfnt + 4, 8, be);

    std::vector<u32> name_off(members.size());
    for (size_t i = 0; i < members.size(); ++i) {
        const size_t at = AlignUp(f.size(), 4);
        f.resize(at + members[i].first.size() + 1);
        std::memcpy(f.data() + at, members[i].first.data(), members[i].first.size());
        name_off[i] = static_cast<u32>((at - (sfnt + 8)) / 4);
    }
    const size_t data_off = AlignUp(f.size(), 0x100);
    f.resize(data_off);
    Put32(f, 0xc, static_cast<u32>(data_off), be);

    std::vector<u32> start(members.size());
    std::vector<u32> end(members.size());
    for (size_t i = 0; i < members.size(); ++i) {
        const size_t at = AlignUp(f.size() - data_off, 0x10);
        const size_t abs = data_off + at;
        f.resize(abs + members[i].second.size());
        if (!members[i].second.empty()) {
            std::memcpy(f.data() + abs, members[i].second.data(), members[i].second.size());
        }
        start[i] = static_cast<u32>(at);
        end[i] = static_cast<u32>(at + members[i].second.size());
    }
    for (size_t n = 0; n < nodes.size(); ++n) {
        const size_t i = nodes[n].index;
        const size_t at = 0x14 + 12 + 16 * n;
        Put32(f, at, nodes[n].hash, be);
        Put32(f, at + 4, 0x01000000u | name_off[i], be);
        Put32(f, at + 8, start[i], be);
        Put32(f, at + 12, end[i], be);
    }
    Put32(f, 8, static_cast<u32>(f.size()), be);
    return f;
}

std::vector<u8> MakeBahs(std::span<const BahsEntry> entries) {
    std::vector<u8> f(0x20);
    std::memcpy(f.data(), "BAHS", 4);
    const size_t table = f.size();
    f.resize(table + 16 * entries.size());
    std::vector<size_t> control_at(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        control_at[i] = Append(f, entries[i].control, 0x10);
    }
    const size_t region = AlignUp(f.size(), 0x100);
    f.resize(region);
    std::vector<size_t> code_at(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        code_at[i] = Append(f, entries[i].code, 0x100) - region;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        const size_t a = table + 16 * i;
        Put32(f, a, static_cast<u32>(code_at[i]));
        Put32(f, a + 4, 0);
        Put32(f, a + 8, static_cast<u32>(entries[i].control.size()));
        Put32(f, a + 12, static_cast<u32>(static_cast<s64>(control_at[i]) - static_cast<s64>(a)));
    }
    // Keep the region non-empty so the header is well formed even without entries.
    f.resize(std::max(f.size(), region + 0x100));
    Put32(f, 0x18, static_cast<u32>(f.size() - region));
    Put32(f, 0x1c, static_cast<u32>(region));
    return f;
}

std::vector<u8> MakeNdsr(std::span<const BahsEntry> shaders) {
    std::vector<u8> f(0x10 + 16 * (shaders.size() + 1));
    std::memcpy(f.data(), "NDSR", 4);
    for (size_t i = 0; i < shaders.size(); ++i) {
        const size_t control = Append(f, shaders[i].control, 0x10);
        const size_t code = Append(f, shaders[i].code, 0x10);
        const size_t r = 0x10 + 16 * i;
        Put32(f, r, static_cast<u32>(control));
        Put32(f, r + 4, static_cast<u32>(shaders[i].control.size() << 4));
        Put32(f, r + 8, static_cast<u32>(code));
        Put32(f, r + 12, static_cast<u32>((shaders[i].code.size() / 16) << 8));
    }
    return f;
}

std::vector<u8> MakeSeadPrimitiveDrawer(const BahsEntry& vertex, const BahsEntry& pixel) {
    std::vector<u8> f(0x10);
    Put32(f, 0, static_cast<u32>(Append(f, vertex.control, 0x10)));
    Put32(f, 4, static_cast<u32>(Append(f, pixel.control, 0x10)));
    Put32(f, 8, static_cast<u32>(Append(f, vertex.code, 0x100)));
    Put32(f, 12, static_cast<u32>(Append(f, pixel.code, 0x100)));
    return f;
}

std::vector<u8> ZstdCompress(std::span<const u8> data, bool record_content_size) {
    std::unique_ptr<ZSTD_CCtx, CCtxDeleter> cctx(ZSTD_createCCtx());
    if (!cctx) {
        throw std::runtime_error("ZSTD_createCCtx failed");
    }
    // An error code is nonzero too: tell it from "more output to flush".
    const auto check = [](size_t ret) {
        if (ZSTD_isError(ret)) {
            throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(ret));
        }
        return ret;
    };
    check(ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_contentSizeFlag, record_content_size ? 1 : 0));
    check(ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_compressionLevel, 3));
    std::vector<u8> out(ZSTD_compressBound(data.size()));
    ZSTD_inBuffer input{data.data(), data.size(), 0};
    ZSTD_outBuffer output{out.data(), out.size(), 0};
    if (record_content_size) {
        check(ZSTD_CCtx_setPledgedSrcSize(cctx.get(), data.size()));
        if (check(ZSTD_compressStream2(cctx.get(), &output, &input, ZSTD_e_end)) != 0) {
            throw std::runtime_error("zstd frame incomplete");
        }
    } else {
        // Feed the data first and end the frame afterwards so the size is not known at the start.
        check(ZSTD_compressStream2(cctx.get(), &output, &input, ZSTD_e_continue));
        ZSTD_inBuffer none{nullptr, 0, 0};
        while (check(ZSTD_compressStream2(cctx.get(), &output, &none, ZSTD_e_end)) != 0) {
            if (output.pos == output.size) {
                throw std::runtime_error("zstd frame incomplete");
            }
        }
    }
    out.resize(output.pos);
    return out;
}

} // namespace VideoCommon::Nvn::Containers
