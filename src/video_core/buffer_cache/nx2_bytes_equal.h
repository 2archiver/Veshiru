// SPDX-FileCopyrightText: Copyright 2026 Veshiru Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstring>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define VESHIRU_NX2_BYTES_EQUAL_SSE2 1
#else
#define VESHIRU_NX2_BYTES_EQUAL_SSE2 0
#endif

#include "common/common_types.h"

namespace VideoCommon {

/// Inline equality check tuned for 4 KiB uniform-bank page snapshots: steps 128 bytes at a time
/// in SSE2 registers without calling out to the C runtime's general-purpose memcmp.
[[nodiscard]] inline bool Nx2BytesEqual(const u8* a, const u8* b, std::size_t size) noexcept {
#if VESHIRU_NX2_BYTES_EQUAL_SSE2
    const __m128i zero = _mm_setzero_si128();
    std::size_t offset = 0;
    while (offset + 128 <= size) {
        const __m128i d0 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset)));
        const __m128i d1 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 16)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 16)));
        const __m128i d2 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 32)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 32)));
        const __m128i d3 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 48)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 48)));
        const __m128i d4 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 64)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 64)));
        const __m128i d5 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 80)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 80)));
        const __m128i d6 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 96)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 96)));
        const __m128i d7 = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset + 112)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset + 112)));
        const __m128i pair01 = _mm_or_si128(d0, d1);
        const __m128i pair23 = _mm_or_si128(d2, d3);
        const __m128i pair45 = _mm_or_si128(d4, d5);
        const __m128i pair67 = _mm_or_si128(d6, d7);
        const __m128i all =
            _mm_or_si128(_mm_or_si128(pair01, pair23), _mm_or_si128(pair45, pair67));
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(all, zero)) != 0xFFFF) {
            return false;
        }
        offset += 128;
    }
    while (offset + 16 <= size) {
        const __m128i diff = _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + offset)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + offset)));
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(diff, zero)) != 0xFFFF) {
            return false;
        }
        offset += 16;
    }
    return offset == size || std::memcmp(a + offset, b + offset, size - offset) == 0;
#else
    return size == 0 || std::memcmp(a, b, size) == 0;
#endif
}

} // namespace VideoCommon

#undef VESHIRU_NX2_BYTES_EQUAL_SSE2
