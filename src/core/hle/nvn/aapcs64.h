// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <functional>

#include "common/common_types.h"
#include "core/hle/kernel/svc_types.h"
#include "video_core/nvn/guest_memory.h"

// AAPCS64 argument and return-value marshaling for HLE'd guest functions. Header only, no JIT
// dependency: it works on a Kernel::Svc::ThreadContext and a memory callback for stack arguments.
//
// Rules implemented (Procedure Call Standard for the Arm 64-bit Architecture, section 6):
//  - Integers and pointers take x0-x7, then 8-byte stack slots. Sub-32-bit integers are masked to
//    their width, because Switch callers do not extend them (matters for NVNboolean).
//  - f32/f64 take the low part of v0-v7, then stack slots. Integer and floating-point register
//    classes are exhausted independently; the stack pointer slot cursor is shared.
//  - 16-byte integers take an even x pair (NGRN rounded up) or a 16-aligned stack slot.
//  - 128-bit vectors take a whole v register or a 16-aligned stack slot.
//  - Composites up to 16 bytes take ceil(size/8) consecutive x registers (or stack when they do
//    not fit entirely, which also exhausts the x registers). Larger composites are passed as a
//    pointer to a caller-made copy.
//  - Homogeneous float aggregates (1 to 4 members of f32 or f64) take consecutive v registers.
//  - Large struct returns go through the buffer in x8.
namespace Core::NvnHle::Aapcs64 {

using MemoryReader = std::function<bool(u64 address, void* destination, size_t size)>;
using MemoryWriter = std::function<bool(u64 address, const void* source, size_t size)>;

class ArgCursor {
public:
    /// Stack and by-reference arguments are read through `reader` (tests).
    ArgCursor(const Kernel::Svc::ThreadContext& context, MemoryReader reader)
        : ctx{context}, read{std::move(reader)} {}
    /// Production: reads go straight to the guest memory (no callable is built per call).
    ArgCursor(const Kernel::Svc::ThreadContext& context, VideoCommon::Nvn::GuestMemory& guest)
        : ctx{context}, memory{&guest} {}

    [[nodiscard]] u64 NextU64() {
        if (ngrn < 8) {
            return ctx.r[ngrn++];
        }
        return StackScalar<u64>(8);
    }
    [[nodiscard]] u64 NextPointer() {
        return NextU64();
    }
    [[nodiscard]] u32 NextU32() {
        return static_cast<u32>(NextU64());
    }
    [[nodiscard]] u16 NextU16() {
        return static_cast<u16>(NextU64());
    }
    [[nodiscard]] u8 NextU8() {
        return static_cast<u8>(NextU64());
    }
    [[nodiscard]] s32 NextS32() {
        return static_cast<s32>(NextU32());
    }
    [[nodiscard]] s16 NextS16() {
        return static_cast<s16>(NextU16());
    }
    [[nodiscard]] s8 NextS8() {
        return static_cast<s8>(NextU8());
    }
    [[nodiscard]] s64 NextS64() {
        return static_cast<s64>(NextU64());
    }
    /// NVNboolean is a one-byte integer the caller does not extend.
    [[nodiscard]] bool NextBool() {
        return NextU8() != 0;
    }

    [[nodiscard]] float NextF32() {
        u32 bits{};
        if (nsrn < 8) {
            bits = static_cast<u32>(ctx.v[nsrn++][0]);
        } else {
            bits = static_cast<u32>(StackScalar<u32>(4));
        }
        return std::bit_cast<float>(bits);
    }
    [[nodiscard]] double NextF64() {
        u64 bits{};
        if (nsrn < 8) {
            bits = ctx.v[nsrn++][0];
        } else {
            bits = StackScalar<u64>(8);
        }
        return std::bit_cast<double>(bits);
    }

    /// 16-byte integer (__int128): even register pair, or a 16-aligned stack slot.
    [[nodiscard]] u128 NextI128() {
        if (ngrn & 1) {
            ++ngrn;
        }
        if (ngrn + 2 <= 8) {
            const u128 value{ctx.r[ngrn], ctx.r[ngrn + 1]};
            ngrn += 2;
            return value;
        }
        ngrn = 8;
        return StackBytes128();
    }

    /// 128-bit vector (Q register) or 16-aligned stack slot.
    [[nodiscard]] u128 NextVector128() {
        if (nsrn < 8) {
            return ctx.v[nsrn++];
        }
        return StackBytes128();
    }

    /// Composite that is not an HFA. `size` bytes are copied to `destination`. Composites of up to
    /// 16 bytes come from registers/stack, larger ones through the pointer the caller passed.
    /// `alignment` is the type alignment (16 forces an even register pair). False if memory could
    /// not be read.
    [[nodiscard]] bool NextComposite(void* destination, size_t size, size_t alignment = 8) {
        if (size > 16) {
            const u64 pointer = NextU64();
            return Read(pointer, destination, size);
        }
        const size_t words = (size + 7) / 8;
        if (alignment >= 16 && (ngrn & 1)) {
            ++ngrn;
        }
        if (ngrn + words <= 8) {
            std::array<u64, 2> raw{};
            for (size_t i = 0; i < words; ++i) {
                raw[i] = ctx.r[ngrn + i];
            }
            ngrn += words;
            std::memcpy(destination, raw.data(), size);
            return true;
        }
        ngrn = 8;
        const u64 address = StackAddress(words * 8, alignment >= 16 ? 16 : 8);
        return Read(address, destination, size);
    }

    /// Homogeneous float aggregate of `count` (1..4) members of `member_size` (4 or 8) bytes.
    /// Members are written tightly packed to `destination`.
    [[nodiscard]] bool NextHfa(void* destination, size_t member_size, size_t count) {
        if (nsrn + count <= 8) {
            auto* out = static_cast<u8*>(destination);
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(out + i * member_size, &ctx.v[nsrn + i][0], member_size);
            }
            nsrn += count;
            return true;
        }
        nsrn = 8;
        const size_t total = member_size * count;
        const u64 address = StackAddress(total, 8);
        return Read(address, destination, total);
    }

    [[nodiscard]] size_t UsedIntegerRegisters() const {
        return ngrn;
    }
    [[nodiscard]] size_t UsedFloatRegisters() const {
        return nsrn;
    }
    /// Bytes of stack arguments consumed so far.
    [[nodiscard]] u64 StackBytesUsed() const {
        return nsaa;
    }

private:
    bool Read(u64 address, void* destination, size_t size) {
        return memory != nullptr ? memory->Read(address, destination, size)
                                 : read(address, destination, size);
    }
    u64 StackAddress(size_t size, size_t alignment) {
        nsaa = (nsaa + alignment - 1) & ~static_cast<u64>(alignment - 1);
        const u64 address = ctx.sp + nsaa;
        nsaa += (size + 7) & ~static_cast<size_t>(7);
        return address;
    }

    template <typename T>
    T StackScalar(size_t size) {
        T value{};
        // A failed read leaves zero; the caller cannot do anything more useful with a bad stack.
        (void)Read(StackAddress(size, 8), &value, sizeof(T));
        return value;
    }

    u128 StackBytes128() {
        u128 value{};
        (void)Read(StackAddress(16, 16), value.data(), sizeof(value));
        return value;
    }

    const Kernel::Svc::ThreadContext& ctx;
    VideoCommon::Nvn::GuestMemory* memory{};
    MemoryReader read;
    size_t ngrn{};
    size_t nsrn{};
    u64 nsaa{};
};

class ReturnWriter {
public:
    ReturnWriter(Kernel::Svc::ThreadContext& context, MemoryWriter writer)
        : ctx{context}, write{std::move(writer)} {}
    ReturnWriter(Kernel::Svc::ThreadContext& context, VideoCommon::Nvn::GuestMemory& guest)
        : ctx{context}, memory{&guest} {}

    void SetU64(u64 value) {
        ctx.r[0] = value;
    }
    /// Sub-64-bit integer results are zero-extended.
    void SetU32(u32 value) {
        ctx.r[0] = value;
    }
    void SetBool(bool value) {
        ctx.r[0] = value ? 1 : 0;
    }
    void SetI128(const u128& value) {
        ctx.r[0] = value[0];
        ctx.r[1] = value[1];
    }
    void SetF32(float value) {
        ctx.v[0] = {std::bit_cast<u32>(value), 0};
    }
    void SetF64(double value) {
        ctx.v[0] = {std::bit_cast<u64>(value), 0};
    }
    void SetVector128(const u128& value) {
        ctx.v[0] = value;
    }

    /// Composite that is not an HFA: up to 16 bytes in x0:x1, larger through the buffer in x8.
    /// False if the x8 buffer could not be written.
    [[nodiscard]] bool SetComposite(const void* source, size_t size) {
        if (size > 16) {
            const u64 buffer = ctx.r[8];
            if (!Write(buffer, source, size)) {
                return false;
            }
            ctx.r[0] = buffer;
            return true;
        }
        std::array<u64, 2> raw{};
        std::memcpy(raw.data(), source, size);
        ctx.r[0] = raw[0];
        if (size > 8) {
            ctx.r[1] = raw[1];
        }
        return true;
    }

    /// Homogeneous float aggregate returned in v0..v(count-1), each member in the low bits.
    void SetHfa(const void* source, size_t member_size, size_t count) {
        const auto* in = static_cast<const u8*>(source);
        for (size_t i = 0; i < count; ++i) {
            u64 bits{};
            std::memcpy(&bits, in + i * member_size, member_size);
            ctx.v[i] = {bits, 0};
        }
    }

private:
    bool Write(u64 address, const void* source, size_t size) {
        return memory != nullptr ? memory->Write(address, source, size)
                                 : write(address, source, size);
    }

    Kernel::Svc::ThreadContext& ctx;
    VideoCommon::Nvn::GuestMemory* memory{};
    MemoryWriter write;
};

} // namespace Core::NvnHle::Aapcs64
