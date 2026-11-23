// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"

// HDR output support shared by the renderer (which owns the swapchain) and the guest-facing
// services (vi2 display queries, NVN window colour space / metadata). Pure maths first, then two
// small published states: what the host display is doing and what the guest asked for.
namespace Common::Hdr {

/// How the pixel values of a presented guest frame are encoded.
enum class Encoding : u32 {
    Srgb = 0,        ///< SDR: sRGB-encoded 0..1, BT.709 primaries (every SDR game)
    Pq2020 = 1,      ///< HDR10: SMPTE ST 2084 (PQ) encoded 0..1, BT.2020 primaries
    ScRgbLinear = 2, ///< scRGB: linear light, BT.709 primaries, 1.0 = 80 nits, may exceed 1 / be < 0
};

/// What the host swapchain carries.
enum class OutputMode : u32 {
    Sdr = 0,   ///< legacy path: 8-bit sRGB-encoded
    Hdr10 = 1, ///< A2B10G10R10_UNORM + HDR10_ST2084
    ScRgb = 2, ///< R16G16B16A16_SFLOAT + EXTENDED_SRGB_LINEAR
};

constexpr f32 ScRgbWhiteNits = 80.0f; ///< scRGB 1.0
constexpr f32 PqMaxNits = 10000.0f;
constexpr u32 ReferencePaperWhiteNits = 203; ///< BT.2408 reference white (used when nothing else is known)
constexpr u32 FallbackPeakNits = 1000;       ///< used when the display reports no peak
constexpr u32 SdrPeakNits = 400;             ///< what the SDR reply of vi2 104 carries

struct Rgb {
    f32 r{};
    f32 g{};
    f32 b{};
};

// ---- transfer functions ----
[[nodiscard]] f32 SrgbToLinear(f32 encoded);
[[nodiscard]] f32 LinearToSrgb(f32 linear);
/// ST 2084 inverse EOTF: absolute luminance in nits -> signal 0..1 (input clamped to 0..10000).
[[nodiscard]] f32 PqEncode(f32 nits);
/// ST 2084 EOTF: signal 0..1 (clamped) -> nits.
[[nodiscard]] f32 PqDecode(f32 signal);
/// scRGB: nits -> linear value (1.0 = 80 nits).
[[nodiscard]] constexpr f32 ScRgbEncode(f32 nits) {
    return nits / ScRgbWhiteNits;
}
[[nodiscard]] constexpr f32 ScRgbDecode(f32 value) {
    return value * ScRgbWhiteNits;
}

// ---- settings ----
/// Setting 0 = automatic: the display-reported value, else the fallback.
[[nodiscard]] constexpr u32 ResolvePeakNits(u32 setting, u32 reported) {
    return setting != 0 ? setting : (reported != 0 ? reported : FallbackPeakNits);
}
[[nodiscard]] constexpr u32 ResolvePaperWhiteNits(u32 setting, u32 reported) {
    return setting != 0 ? setting : (reported != 0 ? reported : ReferencePaperWhiteNits);
}

// ---- primaries ----
[[nodiscard]] Rgb Bt709ToBt2020(Rgb linear);
[[nodiscard]] Rgb Bt2020ToBt709(Rgb linear);

/// Absolute light of an encoded pixel, in nits, with the primaries of the encoding (BT.709 for
/// Srgb / ScRgbLinear, BT.2020 for Pq2020). `paper_white_nits` is where an Srgb 1.0 lands.
[[nodiscard]] Rgb DecodeToNits(Rgb encoded, Encoding source, f32 paper_white_nits);

/// Light of an encoded pixel as linear BT.709 relative to paper white (1.0 = paper white;
/// unbounded above, negative for colours outside BT.709). The one conversion every consumer that
/// needs "the SDR picture plus what exceeds it" is built on: post space (below) keeps the excess in
/// the values, DLSS NR (PqToNrInput) carries it in a residual because NR clamps to 0..1.
/// Srgb frames are the sRGB decode of the code values.
[[nodiscard]] Rgb ToRelative709(Rgb encoded, Encoding source, f32 paper_white_nits);
/// Inverse of ToRelative709 (PQ clamps negative light to 0).
[[nodiscard]] Rgb FromRelative709(Rgb relative, Encoding source, f32 paper_white_nits);

/// The space post-processing effects (ReShade FX) run in. They are written for an SDR picture:
/// sRGB-encoded, BT.709, 1.0 = white. A frame in an HDR encoding is moved into exactly that
/// representation, keeping light above paper white as values above 1 (and out-of-gamut light as
/// negative values, sign-symmetric transfer): paper white encodes to 1.0, so an effect sees the
/// same numbers it would see for the SDR picture. Srgb frames are returned unchanged.
[[nodiscard]] Rgb ToPostSpace(Rgb encoded, Encoding source, f32 paper_white_nits);
/// Inverse of ToPostSpace: the effects' result back in the frame's own encoding (PQ clamps
/// negative light to 0). Srgb frames are returned unchanged.
[[nodiscard]] Rgb FromPostSpace(Rgb post, Encoding source, f32 paper_white_nits);

/// SDR-to-HDR enhancement in linear BT.709 light. Saturation is adjusted around Rec.709
/// luminance and clamped to the displayable range. Highlight boost expands only the range above
/// 80% of paper white, smoothly approaching the selected peak without changing mid-tones.
/// Returns the input unchanged at 100% saturation and 0% highlight boost.
[[nodiscard]] Rgb EnhanceSdr(Rgb linear_nits, f32 paper_white_nits, f32 peak_nits,
                             f32 saturation_percent, f32 highlight_boost_percent);

/// The value to store in the swapchain image for an encoded pixel.
/// Hdr10: PQ of BT.2020 nits. ScRgb: BT.709 nits / 80. Sdr: sRGB of the pixel (Srgb sources pass
/// through untouched; HDR sources are tone-mapped, see ToneMapToSdr). Enhancements apply only to
/// SDR sources on an HDR swapchain.
[[nodiscard]] Rgb ConvertToOutput(Rgb encoded, Encoding source, OutputMode mode,
                                  f32 paper_white_nits, f32 peak_nits,
                                  f32 saturation_percent = 100.0f,
                                  f32 highlight_boost_percent = 0.0f);

/// HDR pixel -> sRGB-encoded 0..1 for 8-bit captures and screenshots. Paper white maps to ~0.95
/// (linear 0.89); light up to 0.7 of paper white is unchanged and the range above it is rolled
/// off smoothly towards 1.0. Srgb input is returned unchanged. Hue-preserving: the largest
/// channel is compressed and the others follow.
[[nodiscard]] Rgb ToneMapToSdr(Rgb encoded, Encoding source, f32 paper_white_nits);

/// An HDR10 pixel prepared for DLSS neural rendering, which is made for SDR frames and clamps its
/// output to 0..1. `base` is what an SDR frame of the same game holds: sRGB-encoded BT.709 light
/// relative to paper white (1.0 = paper white), within 0..1. `residual` is the linear relative
/// light outside that range (highlights above paper white, colours outside BT.709), carried
/// around NR and added back. See Vulkan::FinalNr.
struct NrInput {
    Rgb base;
    Rgb residual;
};
/// HDR10 pixel (PQ, BT.2020) -> NR input.
[[nodiscard]] NrInput PqToNrInput(Rgb pq, f32 paper_white_nits);
/// NR output (or an unchanged `base`) plus the residual -> HDR10 pixel. Exact inverse of
/// PqToNrInput when NR leaves the base alone (light below 0 nits is clamped by the PQ encode).
[[nodiscard]] Rgb NrOutputToPq(Rgb nr_output, Rgb residual, f32 paper_white_nits);

// ---- metadata ----
/// ST 2086 mastering display + CEA-861.3 content light levels (what VkHdrMetadataEXT carries).
struct ContentMetadata {
    std::array<f32, 2> display_primary_red{0.708f, 0.292f};   ///< CIE 1931 xy (BT.2020 default)
    std::array<f32, 2> display_primary_green{0.170f, 0.797f};
    std::array<f32, 2> display_primary_blue{0.131f, 0.046f};
    std::array<f32, 2> white_point{0.3127f, 0.3290f};
    f32 max_luminance{1000.0f}; ///< nits
    f32 min_luminance{0.001f};  ///< nits
    f32 max_content_light_level{0.0f};       ///< MaxCLL nits, 0 = unknown
    f32 max_frame_average_light_level{0.0f}; ///< MaxFALL nits, 0 = unknown

    friend bool operator==(const ContentMetadata&, const ContentMetadata&) = default;
};

/// Metadata the host sends when the game set none: the display's own values.
[[nodiscard]] ContentMetadata DisplayMetadata(f32 peak_nits, f32 min_nits);

// ---- published state ----
/// What the host display is doing. Written by the renderer when it (re)creates the swapchain,
/// read by the guest-facing services.
struct HostDisplay {
    OutputMode mode{OutputMode::Sdr};
    bool surface_offers_hdr{false}; ///< the surface lists an HDR10 or scRGB colour space
    u32 peak_nits{0};               ///< display peak (settings, else display-reported, else fallback)
    u32 paper_white_nits{ReferencePaperWhiteNits};
    f32 min_nits{0.001f};

    [[nodiscard]] bool Active() const {
        return mode != OutputMode::Sdr;
    }
    friend bool operator==(const HostDisplay&, const HostDisplay&) = default;
};

/// What the guest asked for on its window / layer. Written by the vi2 services and the NVN HLE,
/// read by the renderer at present.
struct GuestRequest {
    Encoding encoding{Encoding::Srgb};
    bool has_metadata{false};
    ContentMetadata metadata{};
    u64 generation{0}; ///< bumped by every Publish, lets the renderer notice changes cheaply

    [[nodiscard]] bool WantsHdr() const {
        return encoding != Encoding::Srgb;
    }
};

void PublishHostDisplay(const HostDisplay& display);
[[nodiscard]] HostDisplay GetHostDisplay();

/// Replaces the request (generation is assigned here).
void PublishGuestRequest(GuestRequest request);
[[nodiscard]] GuestRequest GetGuestRequest();

/// Whether the guest's last HDR status query (vi2 104) was answered "HDR". A title that was told
/// HDR and then presents a 10-bit or FP16 buffer without ever setting a window colour space
/// (Donkey Kong Bananza: it switches its display buffers to RGB10A2 and sets the layer's HDR
/// luminance metadata instead) means HDR by that format; see Vulkan::ResolveSourceEncoding.
void NoteGuestToldHdr(bool told);
[[nodiscard]] bool GuestToldHdr();

/// Back to SDR / no request (emulation stop, tests).
void ResetState();

// ---- what the guest is told ----
/// Reply of vi2 104. `format` is the WIRE value the SDK's vi2 client decodes (sdk+0x53fe28 in
/// Donkey Kong Bananza's build): 1 = SDR (nn::vi::HdrOutputFormat 0), 2 = HDR (HdrOutputFormat 1).
/// Any other value, including 0, makes the client fail the query with 0x1472.
inline constexpr u64 WireFormatSdr = 1;
inline constexpr u64 WireFormatHdr = 2;
struct HdrStatusReply {
    u64 format{WireFormatSdr};
    u16 peak_nits{static_cast<u16>(SdrPeakNits)};
};
[[nodiscard]] HdrStatusReply MakeHdrStatusReply(const HostDisplay& display);
/// Reply of vi2 105 in nits.
[[nodiscard]] u16 MakePaperWhiteReply(const HostDisplay& display);

} // namespace Common::Hdr
