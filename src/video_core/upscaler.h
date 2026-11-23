// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>

#include "common/settings_enums.h"

namespace VideoCore::Upscaler {

enum class Backend : std::uint8_t {
    Game,
    Fsr,
    Dlss,
    Dlaa,
    Bilinear,
};

enum class DlaaRequestStage : std::uint8_t {
    Dlaa,
    DlaaCrop,
    Dlss,
    Fsr,
};

/// Largest right/bottom excess, in pixels, an input may have over the output and still be taken
/// as the output's size (cropped, origin kept). The scaled twin of a game's render size is the
/// size times a whole factor (OoT's 854x480 at 3X is 2562x1440), so a small overhang is normal.
inline constexpr std::uint32_t DlaaCropSlack = 2;

/// An explicit DLAA request at the actual input/output sizes. Small right/bottom excess is
/// cropped without moving the input origin, keeping pixel centres aligned. A smaller input uses
/// DLSS at the caller's sizes with the existing ratio-based mode selection. A substantially
/// larger input retains the existing FSR1 path until a dedicated downsampled-input DLAA path is
/// designed.
[[nodiscard]] constexpr DlaaRequestStage SelectDlaaRequestStage(
    std::uint32_t input_width, std::uint32_t input_height, std::uint32_t output_width,
    std::uint32_t output_height) noexcept {
    if (input_width < output_width || input_height < output_height) {
        return DlaaRequestStage::Dlss;
    }
    if (input_width - output_width <= DlaaCropSlack &&
        input_height - output_height <= DlaaCropSlack) {
        return input_width == output_width && input_height == output_height
                   ? DlaaRequestStage::Dlaa
                   : DlaaRequestStage::DlaaCrop;
    }
    return DlaaRequestStage::Fsr;
}

/// Whether an upscale input of this size can be evaluated at the output's size: not larger than
/// the output on either axis (DLSS), or at least as large on both by at most `DlaaCropSlack`
/// (DLAA after the crop). A scaled twin that does not fit is replaced by the guest-sized image.
[[nodiscard]] constexpr bool UpscaleInputFitsOutput(std::uint32_t input_width,
                                                    std::uint32_t input_height,
                                                    std::uint32_t output_width,
                                                    std::uint32_t output_height) noexcept {
    if (input_width <= output_width && input_height <= output_height) {
        return true;
    }
    const DlaaRequestStage stage =
        SelectDlaaRequestStage(input_width, input_height, output_width, output_height);
    return stage == DlaaRequestStage::Dlaa || stage == DlaaRequestStage::DlaaCrop;
}

/// A DLSS request whose input is the output's size (or `DlaaCropSlack` larger) is DLAA: the
/// render ratio is 1, which NVIDIA's DLSS modes cannot express (Decide() already makes equal
/// extents DLAA). Every other backend and size is returned unchanged.
[[nodiscard]] constexpr Backend PromoteNearNativeToDlaa(Backend backend, std::uint32_t input_width,
                                                        std::uint32_t input_height,
                                                        std::uint32_t output_width,
                                                        std::uint32_t output_height) noexcept {
    if (backend != Backend::Dlss) {
        return backend;
    }
    const DlaaRequestStage stage =
        SelectDlaaRequestStage(input_width, input_height, output_width, output_height);
    return stage == DlaaRequestStage::Dlaa || stage == DlaaRequestStage::DlaaCrop
               ? Backend::Dlaa
               : backend;
}

struct Rect {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};

    [[nodiscard]] constexpr bool IsEmpty() const noexcept {
        return width == 0 || height == 0;
    }
};

/// Selects the earlier, game's-upscale-stage backend. `tensor_supported` must describe the
/// currently selected Vulkan device; it is deliberately supplied by the caller so this utility is
/// usable by both the legacy and NVN paths without making either renderer depend on the other.
[[nodiscard]] constexpr Backend SelectBackend(Settings::Upscaler setting, bool tensor_supported) noexcept {
    switch (setting) {
    case Settings::Upscaler::Bilinear:
        return Backend::Bilinear;
    case Settings::Upscaler::Fsr:
        return Backend::Fsr;
    case Settings::Upscaler::Dlss:
        return Backend::Dlss;
    case Settings::Upscaler::Dlaa:
        return Backend::Dlaa;
    case Settings::Upscaler::Game:
        return tensor_supported ? Backend::Game : Backend::Fsr;
    case Settings::Upscaler::Auto:
        return tensor_supported ? Backend::Game : Backend::Fsr;
    }
    return Backend::Fsr;
}

/// DLSS and DLAA cannot run without NVIDIA's runtime files, which the settings let an RTX user
/// pick before downloading them: such a session uses Auto, whatever else the setting says.
[[nodiscard]] constexpr Settings::Upscaler ResolveRequested(Settings::Upscaler setting,
                                                            bool dlss_runtime_ready) noexcept {
    if ((setting == Settings::Upscaler::Dlss || setting == Settings::Upscaler::Dlaa) &&
        !dlss_runtime_ready) {
        return Settings::Upscaler::Auto;
    }
    return setting;
}

/// The upscaler setting as the emulation uses it. Eglantine has no upscaler setting and no NVIDIA
/// DLSS runtime, so this always returns Auto.
[[nodiscard]] Settings::Upscaler Effective();

/// Called when the renderer is going away; currently does nothing (Effective keeps no state).
void ResetSession() noexcept;


[[nodiscard]] constexpr bool IsValidRect(Rect rect, std::uint32_t image_width,
                                         std::uint32_t image_height) noexcept {
    return rect.width != 0 && rect.height != 0 && rect.x <= image_width && rect.y <= image_height &&
           rect.width <= image_width - rect.x && rect.height <= image_height - rect.y;
}

/// A rectangle as a game passes it: signed origin and size, possibly outside its image.
struct GameRect {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t width{};
    std::int64_t height{};
};

/// The rectangles an upscale runs on after ClipUpscaleRects; `clipped` when either changed.
struct ClippedUpscaleRects {
    Rect input;
    Rect output;
    bool clipped{};
};

namespace Detail {

/// `numerator / denominator` rounded to nearest, halves away from zero; denominator > 0.
[[nodiscard]] constexpr std::int64_t RoundedDivide(std::int64_t numerator,
                                                   std::int64_t denominator) noexcept {
    return numerator >= 0 ? (numerator + denominator / 2) / denominator
                          : -((-numerator + denominator / 2) / denominator);
}

/// One axis of ClipUpscaleRects; false when either span ends up empty.
[[nodiscard]] constexpr bool ClipUpscaleAxis(std::int64_t in_pos, std::int64_t in_size,
                                             std::int64_t out_pos, std::int64_t out_size,
                                             std::int64_t in_limit, std::int64_t out_limit,
                                             std::uint32_t& in_begin, std::uint32_t& in_length,
                                             std::uint32_t& out_begin,
                                             std::uint32_t& out_length) noexcept {
    const std::int64_t out_end = out_pos + out_size;
    const std::int64_t clipped_out_begin = out_pos > 0 ? out_pos : 0;
    const std::int64_t clipped_out_end = out_end < out_limit ? out_end : out_limit;
    if (clipped_out_end <= clipped_out_begin) {
        return false;
    }
    // Both deltas are at most out_size in magnitude, so the products stay below 2^62.
    std::int64_t in_start =
        in_pos + RoundedDivide((clipped_out_begin - out_pos) * in_size, out_size);
    std::int64_t in_end =
        in_pos + in_size + RoundedDivide((clipped_out_end - out_end) * in_size, out_size);
    in_start = in_start > 0 ? in_start : 0;
    in_end = in_end < in_limit ? in_end : in_limit;
    if (in_end <= in_start) {
        return false;
    }
    in_begin = static_cast<std::uint32_t>(in_start);
    in_length = static_cast<std::uint32_t>(in_end - in_start);
    out_begin = static_cast<std::uint32_t>(clipped_out_begin);
    out_length = static_cast<std::uint32_t>(clipped_out_end - clipped_out_begin);
    return true;
}

} // namespace Detail

/// The part of a game's upscale that lies inside its images. A GPU drops writes outside an
/// image, so the output rectangle is clipped to its level; the input rectangle moves by the same
/// fraction of its size (rounded to the nearest texel) so the remaining output keeps its source,
/// and is then clipped to its own level. None when a size is not positive or either rectangle
/// ends up empty.
[[nodiscard]] constexpr std::optional<ClippedUpscaleRects> ClipUpscaleRects(
    GameRect input, GameRect output, std::uint32_t input_image_width,
    std::uint32_t input_image_height, std::uint32_t output_image_width,
    std::uint32_t output_image_height) noexcept {
    if (input.width <= 0 || input.height <= 0 || output.width <= 0 || output.height <= 0) {
        return std::nullopt;
    }
    // ClipUpscaleAxis's products stay in range for 32-bit origins and sizes (what the SDK
    // passes); anything wider is not a real rectangle.
    const auto fits_32 = [](const GameRect& rect) {
        constexpr std::int64_t min = INT32_MIN;
        constexpr std::int64_t max = INT32_MAX;
        return rect.x >= min && rect.x <= max && rect.y >= min && rect.y <= max &&
               rect.width <= max && rect.height <= max;
    };
    if (!fits_32(input) || !fits_32(output)) {
        return std::nullopt;
    }
    ClippedUpscaleRects result{};
    if (!Detail::ClipUpscaleAxis(input.x, input.width, output.x, output.width, input_image_width,
                                 output_image_width, result.input.x, result.input.width,
                                 result.output.x, result.output.width) ||
        !Detail::ClipUpscaleAxis(input.y, input.height, output.y, output.height,
                                 input_image_height, output_image_height, result.input.y,
                                 result.input.height, result.output.y, result.output.height)) {
        return std::nullopt;
    }
    const auto same = [](const Rect& rect, const GameRect& game) {
        return rect.x == game.x && rect.y == game.y && rect.width == game.width &&
               rect.height == game.height;
    };
    result.clipped = !same(result.input, input) || !same(result.output, output);
    return result;
}

/// NDSR's eight push words are inclusive bounds: x, y, src_w-1, src_h-1, out_x, out_y,
/// dst_w-1, dst_h-1. SASS clamps the relative coordinate before adding the origin. Rejecting
/// zero/overflow values here prevents a malformed guest dispatch from exceeding either image.
[[nodiscard]] constexpr bool DecodeNdsrRects(std::span<const std::uint32_t> words, Rect& input,
                                             Rect& output, std::uint32_t input_width,
                                             std::uint32_t input_height, std::uint32_t output_width,
                                             std::uint32_t output_height) noexcept {
    if (words.size() < 8 || words[2] == UINT32_MAX || words[3] == UINT32_MAX ||
        words[6] == UINT32_MAX || words[7] == UINT32_MAX) {
        return false;
    }
    input = {.x = words[0], .y = words[1], .width = words[2] + 1, .height = words[3] + 1};
    output = {.x = words[4], .y = words[5], .width = words[6] + 1, .height = words[7] + 1};
    return IsValidRect(input, input_width, input_height) &&
           IsValidRect(output, output_width, output_height);
}

/// DK's NDSR pack holds 24 compute programs, one per upscale-ratio/tile configuration, and which
/// one runs depends on the scene (newgame uses variants the town never does). They all share one
/// dispatch shape that no other compute program in the game has (checked against the 410-program
/// compute census): 128x1x1 threads, one sampled texture, two storage-image slots, and eight push
/// words holding the source/destination rectangles. Detect by shape, not by program hash.
[[nodiscard]] constexpr bool IsNdsrDispatchShape(std::array<std::uint32_t, 3> thread_group,
                                                 std::uint32_t texture_count,
                                                 std::uint32_t image_count,
                                                 std::size_t push_constant_bytes) noexcept {
    return thread_group[0] == 128 && thread_group[1] == 1 && thread_group[2] == 1 &&
           texture_count == 1 && image_count == 2 && push_constant_bytes >= sizeof(std::uint32_t) * 8;
}

/// SM86 HMMA (tensor-core matrix multiply) is opcode 0x23c in the low 12 bits of an instruction's
/// first word; instructions are 16 bytes.
inline constexpr std::uint32_t Sm86HmmaOpcode = 0x23c;

/// Whether a compute program's code holds a tensor-core matrix multiply: what separates a game's
/// neural upscaling network from the other compute programs of the same dispatch shape.
[[nodiscard]] inline bool CodeUsesTensorMath(std::span<const std::uint8_t> code) noexcept {
    for (std::size_t offset = 0; offset + 16 <= code.size(); offset += 16) {
        std::uint32_t word{};
        std::memcpy(&word, code.data() + offset, sizeof(word));
        if ((word & 0xfffu) == Sm86HmmaOpcode) {
            return true;
        }
    }
    return false;
}

/// A game's upscaling network dispatch, recognised by structure alone: the NDSR dispatch shape,
/// tensor-core math in the program, and rectangle words that describe a strictly larger output
/// (the input rectangle is smaller than the output one on both axes). DK's NDSR programs and OoT's
/// (same SDK family) both classify; no title, build or program hash is involved.
[[nodiscard]] constexpr bool IsUpscaleNetworkDispatch(std::array<std::uint32_t, 3> thread_group,
                                                      std::uint32_t texture_count,
                                                      std::uint32_t image_count,
                                                      std::size_t push_constant_bytes,
                                                      bool uses_tensor_math, const Rect& input,
                                                      const Rect& output) noexcept {
    return IsNdsrDispatchShape(thread_group, texture_count, image_count, push_constant_bytes) &&
           uses_tensor_math && input.width < output.width && input.height < output.height;
}

/// Whether the game's own upscaling network is replaced by the host's upscaler. `backend` is the
/// setting's selection (SelectBackend): everything but the game's own replaces it. The game's own
/// stays at the guest resolution (Auto, Game); with the Resolution setting above 1X, Auto
/// replaces it too (the network reads the scaled scene only at its native sample positions, so a
/// scaled scene would gain nothing), while an explicit Game keeps it.
[[nodiscard]] constexpr bool ReplacesGameNetwork(Backend backend, Settings::Upscaler requested,
                                                 bool resolution_scaled) noexcept {
    return backend != Backend::Game || (resolution_scaled && requested == Settings::Upscaler::Auto);
}

/// DLAA belongs to the Upscaler choice. It is a final-frame fallback only when no DLSS/DLAA
/// evaluation already produced this frame; FXAA/SMAA never participates in this decision.
[[nodiscard]] constexpr bool FinalDlaaAfterNetwork(Settings::Upscaler requested,
                                                   bool dlss_produced_frame) noexcept {
    return requested == Settings::Upscaler::Dlaa && !dlss_produced_frame;
}

struct HlePassPrecedence {
    bool replace_game_network{};
    bool final_dlaa{};
};

/// The two NVN HLE passes after runtime-file resolution of the upscaler setting.
[[nodiscard]] constexpr HlePassPrecedence SelectHlePassPrecedence(
    Backend backend, Settings::Upscaler requested, bool resolution_scaled,
    bool dlss_produced_frame) noexcept {
    return {.replace_game_network = ReplacesGameNetwork(backend, requested, resolution_scaled),
            .final_dlaa = FinalDlaaAfterNetwork(requested, dlss_produced_frame)};
}

/// Texture views share their base storage. The compute-chain provenance uses the base serial so
/// an R32 view written by OoT's packing dispatch matches the base texture the network samples.
[[nodiscard]] constexpr std::uint64_t NetworkTextureSerial(std::uint64_t serial,
                                                           std::uint64_t view_of) noexcept {
    return view_of != 0 ? view_of : serial;
}

/// One compute dispatch as the replay remembers it for `FindNetworkColour`: the texture serials
/// its storage images are bound to and its sampled textures (0 = none).
struct ComputeChainEntry {
    std::array<std::uint64_t, 4> written{};
    std::array<std::uint64_t, 8> sampled{};
};

/// The colour a game's upscaling network really consumes. Some games feed the network a packed
/// image that an earlier compute dispatch wrote from the scene (OoT: sqrt-encoded RGB9E5); the
/// scene is what the host upscaler needs. `chain` lists the dispatches before the network's, newest
/// first. When no dispatch wrote `network_input` it is the colour itself (DK) and is returned.
/// Otherwise the answer is the first sampled texture of the newest writer that `is_colour` accepts
/// and that no dispatch of the chain wrote (the other prepass outputs); 0 when there is none.
template <typename Accepts>
[[nodiscard]] std::uint64_t FindNetworkColour(std::span<const ComputeChainEntry> chain,
                                              std::uint64_t network_input, Accepts&& is_colour) {
    const auto written_by = [](const ComputeChainEntry& entry, std::uint64_t serial) {
        for (const std::uint64_t written : entry.written) {
            if (written == serial) {
                return true;
            }
        }
        return false;
    };
    if (network_input == 0) {
        return 0;
    }
    const ComputeChainEntry* producer = nullptr;
    for (const ComputeChainEntry& entry : chain) {
        if (written_by(entry, network_input)) {
            producer = &entry;
            break;
        }
    }
    if (producer == nullptr) {
        return network_input;
    }
    for (const std::uint64_t sampled : producer->sampled) {
        if (sampled == 0 || sampled == network_input || !is_colour(sampled)) {
            continue;
        }
        bool prepass_output = false;
        for (const ComputeChainEntry& entry : chain) {
            prepass_output |= written_by(entry, sampled);
        }
        if (!prepass_output) {
            return sampled;
        }
    }
    return 0;
}

} // namespace VideoCore::Upscaler
