// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string_view>

#include "common/common_types.h"
#include "video_core/ocarina_camera.h"

namespace VideoCommon {

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

/// The game's output passes the temporal upscaler can replace.
enum class OcarinaUpscalePass {
    None,
    /// The 3/4 upscale (1080p to 1440p docked, 810p to 1080p handheld).
    ThreeQuarter,
    /// The upscale to 2560x1440 from a 16:9 scene up to that size.
    Pass1440p,
    /// The same output pass at 1:1, 1920x1080 in handheld with the Max Resolution add-on.
    FullSize,
};

/// The compute pass whose depth texture is the scene depth the upscaler reprojects.
constexpr u64 OcarinaDepthReaderKey = 0x20ab4117f4db5bdbULL; // shader 1fc7a0423116281f

/// Passes are matched by native program key, which only the game's code decides: the shader hash
/// changes with every lowering change that touches the program.
constexpr OcarinaUpscalePass FindOcarinaUpscalePass(u64 native_key, u64 shader_hash) {
    using enum OcarinaUpscalePass;
    switch (native_key) {
    case 0x6f9367bd41dc0b58ULL: // shader e40f5efb5480e692
        return ThreeQuarter;
    case 0xda6df3e54c716df2ULL: // shader 6bd913e1f34e83eb
    case 0xe4cda162ff6009e2ULL: // shader 9302bc0d6d1d9bc1, a scaled scene in the field (#903)
        return Pass1440p;
    }
    // No capture of these programs has their native key yet. None of them ran in a v131072
    // session on RADV, handheld with and without Max Resolution, into the Deku Tree.
    switch (shader_hash) {
    case 0xef542df87cacfe9eULL:
        return ThreeQuarter;
    case 0x4c1ae2fe58f30a97ULL: // a scaled scene in menus (#903)
        return Pass1440p;
    case 0xc033c3ddc9f0897bULL:
        return FullSize;
    }
    return None;
}

/// render_scaled: a DLSS render-scale patch shrank the scene to a size FSR 3.1 takes, so FSR 3.1
/// reconstructs it when DLSS can't run.
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available, bool render_scaled,
                                                    std::string_view override) {
    if (override == "native") {
        return OcarinaUpscaleBackend::Native;
    }
    if (override == "fsr") {
        return OcarinaUpscaleBackend::Fsr;
    }
    if ((dlss_enabled || override == "dlss") && dlss_available) {
        return OcarinaUpscaleBackend::Dlss;
    }
    return fsr_enabled || render_scaled ? OcarinaUpscaleBackend::Fsr
                                        : OcarinaUpscaleBackend::Native;
}

/// Texture LOD bias added to Ocarina's mipmapped samplers while FSR upscales. AMD recommends
/// log2(render / output) - 1: -1.58 for docked 1440p -> 4K.
constexpr float OcarinaFsrLodBias(OcarinaUpscaleBackend backend) {
    return backend == OcarinaUpscaleBackend::Fsr ? -1.58f : 0.0f;
}

/// Whether a sampler with that bias keeps it for an image: only docked (handheld isn't measured),
/// not for images the GPU wrote (render targets would read a mip too sharp), and not once the
/// upscaler failed this session (the game's own pass runs again).
constexpr bool KeepOcarinaFsrLodBias(bool docked, bool upscaler_failed, bool gpu_written) {
    return docked && !upscaler_failed && !gpu_written;
}

namespace OcarinaUpscale {

/// Row-major matrices, multiplied by column vectors.
using Matrix = std::array<f32, 16>;

inline constexpr size_t CameraBuffer = OcarinaCameraBufferIndex;
inline constexpr size_t CameraBytes = OcarinaCameraBytes;
inline constexpr size_t ProjectionOffset = OcarinaCameraProjectionOffset;
inline constexpr size_t ViewProjectionOffset = OcarinaCameraViewProjectionOffset;

/// The captured projection has this exact perspective marker at elements six and seven.
[[nodiscard]] inline bool IsPerspective(const Matrix& projection) noexcept {
    return projection[6] == -1.0f && projection[7] == 0.0f;
}

/// Maps guest clip coordinates through a full-image Vulkan viewport to the clip coordinates
/// reconstructed from the depth image. The caller verifies that the viewport covers the image.
[[nodiscard]] inline Matrix ViewportMatrix(f32 x, f32 y, f32 width, f32 height,
                                           f32 y_sign, f32 depth_offset,
                                           f32 depth_scale) noexcept {
    Matrix result{};
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) ||
        !std::isfinite(height) || !std::isfinite(y_sign) || !std::isfinite(depth_offset) ||
        !std::isfinite(depth_scale) || width == 0.0f || height == 0.0f) {
        return result;
    }

    // A viewport's clip-to-pixel mapping is inverted back into full-image clip coordinates.
    // The depth image uses a top-left origin and [-1, 1] clip depth.
    const f32 width_sign = std::signbit(width) ? -1.0f : 1.0f;
    const f32 height_sign = std::signbit(height) ? -1.0f : 1.0f;
    const f32 absolute_width = std::abs(width);
    const f32 absolute_height = std::abs(height);
    result[0] = width_sign;
    result[3] = 2.0f * x / absolute_width + width_sign - 1.0f;
    result[5] = -height_sign * y_sign;
    result[7] = 1.0f - 2.0f * y / absolute_height - height_sign;
    result[10] = 2.0f * depth_scale;
    result[11] = 2.0f * depth_offset - 1.0f;
    result[15] = 1.0f;
    return result;
}

/// Combines the viewport mapping with the camera's view-projection matrix.
[[nodiscard]] inline Matrix ScreenMatrix(const Matrix& viewport,
                                         const Matrix& view_projection) noexcept {
    Matrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            for (size_t inner = 0; inner < 4; ++inner) {
                result[row * 4 + column] +=
                    viewport[row * 4 + inner] * view_projection[inner * 4 + column];
            }
        }
    }
    return result;
}

/// Maps this frame's clip coordinates to the previous frame's clip coordinates.
[[nodiscard]] inline std::optional<Matrix> Reprojection(const Matrix& previous,
                                                        const Matrix& current) {
    if (!std::all_of(previous.begin(), previous.end(),
                     [](f32 value) { return std::isfinite(value); }) ||
        !std::all_of(current.begin(), current.end(),
                     [](f32 value) { return std::isfinite(value); })) {
        return std::nullopt;
    }

    std::array<std::array<f64, 8>, 4> rows{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            rows[row][column] = current[row * 4 + column];
        }
        rows[row][row + 4] = 1.0;
    }
    for (size_t column = 0; column < 4; ++column) {
        size_t pivot = column;
        for (size_t row = column + 1; row < 4; ++row) {
            if (std::abs(rows[row][column]) > std::abs(rows[pivot][column])) {
                pivot = row;
            }
        }
        std::swap(rows[column], rows[pivot]);
        const f64 divisor = rows[column][column];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-10) {
            return std::nullopt;
        }
        for (f64& value : rows[column]) {
            value /= divisor;
        }
        for (size_t row = 0; row < 4; ++row) {
            if (row == column) {
                continue;
            }
            const f64 scale = rows[row][column];
            for (size_t index = 0; index < 8; ++index) {
                rows[row][index] -= scale * rows[column][index];
            }
        }
    }

    Matrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            f64 value{};
            for (size_t inner = 0; inner < 4; ++inner) {
                value += previous[row * 4 + inner] * rows[inner][column + 4];
            }
            result[row * 4 + column] = static_cast<f32>(value);
        }
    }
    if (!std::all_of(result.begin(), result.end(),
                     [](f32 value) { return std::isfinite(value); })) {
        return std::nullopt;
    }
    return result;
}

/// The centred Halton sample used by the temporal pass.
[[nodiscard]] inline std::array<f32, 2> Jitter(u32 phase) noexcept {
    return OcarinaJitter(phase, false);
}

} // namespace OcarinaUpscale

} // namespace VideoCommon
