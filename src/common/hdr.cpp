// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <mutex>

#include "common/hdr.h"

namespace Common::Hdr {

namespace {

// SMPTE ST 2084 constants.
constexpr f32 PqM1 = 2610.0f / 16384.0f;
constexpr f32 PqM2 = 2523.0f / 4096.0f * 128.0f;
constexpr f32 PqC1 = 3424.0f / 4096.0f;
constexpr f32 PqC2 = 2413.0f / 4096.0f * 32.0f;
constexpr f32 PqC3 = 2392.0f / 4096.0f * 32.0f;

// Tone map knee, in light relative to paper white.
constexpr f32 ToneKnee = 0.7f;

struct State {
    std::mutex mutex;
    HostDisplay display{};
    GuestRequest request{};
    bool guest_told_hdr{false};
};

State& GetState() {
    static State state;
    return state;
}

template <typename F>
Rgb Map(Rgb value, F&& f) {
    return {f(value.r), f(value.g), f(value.b)};
}

Rgb Scale(Rgb value, f32 factor) {
    return {value.r * factor, value.g * factor, value.b * factor};
}

} // Anonymous namespace

f32 SrgbToLinear(f32 encoded) {
    const f32 magnitude = std::fabs(encoded);
    const f32 linear = magnitude <= 0.04045f ? magnitude / 12.92f
                                             : std::pow((magnitude + 0.055f) / 1.055f, 2.4f);
    return std::copysign(linear, encoded);
}

f32 LinearToSrgb(f32 linear) {
    const f32 magnitude = std::fabs(linear);
    const f32 encoded = magnitude <= 0.0031308f
                            ? magnitude * 12.92f
                            : 1.055f * std::pow(magnitude, 1.0f / 2.4f) - 0.055f;
    return std::copysign(encoded, linear);
}

f32 PqEncode(f32 nits) {
    const f32 y = std::clamp(nits / PqMaxNits, 0.0f, 1.0f);
    const f32 ym = std::pow(y, PqM1);
    return std::pow((PqC1 + PqC2 * ym) / (1.0f + PqC3 * ym), PqM2);
}

f32 PqDecode(f32 signal) {
    const f32 e = std::pow(std::clamp(signal, 0.0f, 1.0f), 1.0f / PqM2);
    const f32 numerator = std::max(e - PqC1, 0.0f);
    const f32 denominator = PqC2 - PqC3 * e;
    return PqMaxNits * std::pow(numerator / denominator, 1.0f / PqM1);
}

Rgb Bt709ToBt2020(Rgb c) {
    return {
        0.6274040f * c.r + 0.3292820f * c.g + 0.0433136f * c.b,
        0.0690970f * c.r + 0.9195400f * c.g + 0.0113612f * c.b,
        0.0163916f * c.r + 0.0880132f * c.g + 0.8955950f * c.b,
    };
}

Rgb Bt2020ToBt709(Rgb c) {
    return {
        1.6604910f * c.r - 0.5876411f * c.g - 0.0728499f * c.b,
        -0.1245505f * c.r + 1.1328999f * c.g - 0.0083494f * c.b,
        -0.0181508f * c.r - 0.1005789f * c.g + 1.1187297f * c.b,
    };
}

Rgb DecodeToNits(Rgb encoded, Encoding source, f32 paper_white_nits) {
    switch (source) {
    case Encoding::Pq2020:
        return Map(encoded, PqDecode);
    case Encoding::ScRgbLinear:
        return Scale(encoded, ScRgbWhiteNits);
    case Encoding::Srgb:
    default:
        return Scale(Map(encoded, SrgbToLinear), paper_white_nits);
    }
}

Rgb ToRelative709(Rgb encoded, Encoding source, f32 paper_white_nits) {
    const f32 paper_white = std::max(paper_white_nits, 1.0f);
    switch (source) {
    case Encoding::Pq2020:
        return Scale(Bt2020ToBt709(Map(encoded, PqDecode)), 1.0f / paper_white);
    case Encoding::ScRgbLinear:
        return Scale(encoded, ScRgbWhiteNits / paper_white);
    case Encoding::Srgb:
    default:
        return Map(encoded, SrgbToLinear);
    }
}

Rgb FromRelative709(Rgb relative, Encoding source, f32 paper_white_nits) {
    const f32 paper_white = std::max(paper_white_nits, 1.0f);
    switch (source) {
    case Encoding::Pq2020:
        return Map(Bt709ToBt2020(Scale(relative, paper_white)), PqEncode);
    case Encoding::ScRgbLinear:
        return Scale(relative, paper_white / ScRgbWhiteNits);
    case Encoding::Srgb:
    default:
        return Map(relative, LinearToSrgb);
    }
}

Rgb ToPostSpace(Rgb encoded, Encoding source, f32 paper_white_nits) {
    if (source == Encoding::Srgb) {
        return encoded;
    }
    return Map(ToRelative709(encoded, source, paper_white_nits), LinearToSrgb);
}

Rgb FromPostSpace(Rgb post, Encoding source, f32 paper_white_nits) {
    if (source == Encoding::Srgb) {
        return post;
    }
    return FromRelative709(Map(post, SrgbToLinear), source, paper_white_nits);
}

Rgb EnhanceSdr(Rgb linear_nits, f32 paper_white_nits, f32 peak_nits,
               f32 saturation_percent, f32 highlight_boost_percent) {
    if (saturation_percent == 100.0f && highlight_boost_percent == 0.0f) {
        return linear_nits;
    }

    constexpr f32 highlight_knee = 0.8f;
    const f32 paper_white = std::max(paper_white_nits, 1.0f);
    const f32 peak = std::max(peak_nits, 1.0f);
    Rgb result = linear_nits;

    // Skip neutral pixels to avoid introducing chroma through floating-point roundoff.
    if (saturation_percent != 100.0f && !(result.r == result.g && result.g == result.b)) {
        const f32 saturation = std::clamp(saturation_percent / 100.0f, 0.5f, 2.0f);
        const f32 luminance =
            0.2126f * result.r + 0.7152f * result.g + 0.0722f * result.b;
        result = {
            luminance + (result.r - luminance) * saturation,
            luminance + (result.g - luminance) * saturation,
            luminance + (result.b - luminance) * saturation,
        };
    }

    if (highlight_boost_percent != 0.0f && peak > paper_white) {
        const f32 boost = std::clamp(highlight_boost_percent / 100.0f, 0.0f, 1.0f);
        const f32 brightest = std::max({result.r, result.g, result.b});
        const f32 relative = brightest / paper_white;
        if (relative > highlight_knee) {
            const f32 t = std::clamp((relative - highlight_knee) / (1.0f - highlight_knee),
                                     0.0f, 1.0f);
            const f32 smooth = t * t * (3.0f - 2.0f * t);
            const f32 target_relative = 1.0f + boost * (peak / paper_white - 1.0f);
            const f32 expanded =
                std::min(relative + (target_relative - 1.0f) * smooth, peak / paper_white);
            result = Scale(result, expanded / relative);
        }
    }

    return Map(result, [peak](f32 value) { return std::clamp(value, 0.0f, peak); });
}

Rgb ConvertToOutput(Rgb encoded, Encoding source, OutputMode mode, f32 paper_white_nits,
                    f32 peak_nits, f32 saturation_percent, f32 highlight_boost_percent) {
    if (mode == OutputMode::Sdr) {
        return source == Encoding::Srgb ? encoded : ToneMapToSdr(encoded, source, paper_white_nits);
    }
    Rgb nits = DecodeToNits(encoded, source, paper_white_nits);
    if (source == Encoding::Srgb &&
        (saturation_percent != 100.0f || highlight_boost_percent != 0.0f)) {
        nits = EnhanceSdr(nits, paper_white_nits, peak_nits, saturation_percent,
                          highlight_boost_percent);
    }
    const bool source_is_2020 = source == Encoding::Pq2020;
    if (mode == OutputMode::Hdr10) {
        return Map(source_is_2020 ? nits : Bt709ToBt2020(nits), PqEncode);
    }
    const Rgb linear709 = source_is_2020 ? Bt2020ToBt709(nits) : nits;
    return Scale(linear709, 1.0f / ScRgbWhiteNits);
}

Rgb ToneMapToSdr(Rgb encoded, Encoding source, f32 paper_white_nits) {
    if (source == Encoding::Srgb) {
        return encoded;
    }
    const f32 paper_white = std::max(paper_white_nits, 1.0f);
    Rgb nits = DecodeToNits(encoded, source, paper_white);
    if (source == Encoding::Pq2020) {
        nits = Bt2020ToBt709(nits);
    }
    Rgb relative = Scale(nits, 1.0f / paper_white);
    relative = Map(relative, [](f32 v) { return std::max(v, 0.0f); });
    const f32 peak = std::max({relative.r, relative.g, relative.b});
    if (peak > ToneKnee) {
        const f32 range = 1.0f - ToneKnee;
        const f32 compressed = ToneKnee + range * (1.0f - std::exp(-(peak - ToneKnee) / range));
        relative = Scale(relative, compressed / peak);
    }
    return Map(relative, [](f32 v) {
        const f32 encoded = LinearToSrgb(std::clamp(v, 0.0f, 1.0f));
        // Keep the finite-precision knee monotonic at the top end. Values this close to the
        // display white point are indistinguishable after 8-bit capture quantisation.
        return encoded > 0.999999f ? 1.0f : encoded;
    });
}

NrInput PqToNrInput(Rgb pq, f32 paper_white_nits) {
    const Rgb relative = ToRelative709(pq, Encoding::Pq2020, paper_white_nits);
    const Rgb in_range = Map(relative, [](f32 v) { return std::clamp(v, 0.0f, 1.0f); });
    return NrInput{
        .base = Map(in_range, LinearToSrgb),
        .residual = {relative.r - in_range.r, relative.g - in_range.g, relative.b - in_range.b},
    };
}

Rgb NrOutputToPq(Rgb nr_output, Rgb residual, f32 paper_white_nits) {
    const Rgb in_range = Map(nr_output, SrgbToLinear);
    const Rgb relative{in_range.r + residual.r, in_range.g + residual.g, in_range.b + residual.b};
    return FromRelative709(relative, Encoding::Pq2020, paper_white_nits);
}

ContentMetadata DisplayMetadata(f32 peak_nits, f32 min_nits) {
    ContentMetadata metadata;
    metadata.max_luminance = peak_nits;
    metadata.min_luminance = min_nits;
    return metadata;
}

void PublishHostDisplay(const HostDisplay& display) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.display = display;
}

HostDisplay GetHostDisplay() {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    return state.display;
}

void PublishGuestRequest(GuestRequest request) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    request.generation = state.request.generation + 1;
    state.request = request;
}

GuestRequest GetGuestRequest() {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    return state.request;
}

void NoteGuestToldHdr(bool told) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.guest_told_hdr = told;
}

bool GuestToldHdr() {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    return state.guest_told_hdr;
}

void ResetState() {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.display = HostDisplay{};
    state.guest_told_hdr = false;
    const u64 generation = state.request.generation + 1;
    state.request = GuestRequest{};
    state.request.generation = generation;
}

HdrStatusReply MakeHdrStatusReply(const HostDisplay& display) {
    if (!display.Active()) {
        return HdrStatusReply{};
    }
    const u32 peak = display.peak_nits != 0 ? display.peak_nits : FallbackPeakNits;
    return HdrStatusReply{WireFormatHdr, static_cast<u16>(std::min<u32>(peak, 0xffff))};
}

u16 MakePaperWhiteReply(const HostDisplay& display) {
    if (!display.Active() || display.paper_white_nits == 0) {
        return static_cast<u16>(ReferencePaperWhiteNits);
    }
    return static_cast<u16>(std::min<u32>(display.paper_white_nits, 0xffff));
}

} // namespace Common::Hdr
