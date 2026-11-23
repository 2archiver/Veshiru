// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstring>

#include "video_core/renderer_vulkan/nvn/nvn_screenshot.h"

namespace Vulkan::Nvn {
namespace {

/// IEEE binary16 at `bytes` (little endian) as a float.
f32 HalfToFloat(const u8* bytes) {
    u16 bits;
    std::memcpy(&bits, bytes, sizeof(bits));
    const u32 sign = static_cast<u32>(bits >> 15) << 31;
    const u32 exponent = (bits >> 10) & 0x1f;
    const u32 mantissa = bits & 0x3ff;
    u32 result;
    if (exponent == 0x1f) { // infinity / NaN
        result = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent != 0) {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    } else if (mantissa == 0) {
        result = sign;
    } else { // subnormal
        const f32 magnitude = std::ldexp(static_cast<f32>(mantissa), -24);
        return sign != 0 ? -magnitude : magnitude;
    }
    f32 value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

u8 QuantizeUnorm8(f32 value) {
    if (std::isnan(value)) {
        return 0;
    }
    return static_cast<u8>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

} // Anonymous namespace

PixelRect CropRectOf(s32 x, s32 y, s32 width, s32 height, u32 source_width, u32 source_height) {
    const s64 left = std::max<s64>(x, 0);
    const s64 top = std::max<s64>(y, 0);
    const s64 right = std::min<s64>(static_cast<s64>(x) + width, source_width);
    const s64 bottom = std::min<s64>(static_cast<s64>(y) + height, source_height);
    if (width <= 0 || height <= 0 || right <= left || bottom <= top) {
        return {0, 0, source_width, source_height};
    }
    return {static_cast<u32>(left), static_cast<u32>(top), static_cast<u32>(right - left),
            static_cast<u32>(bottom - top)};
}

PixelRect FitPresentRect(u32 source_width, u32 source_height, u32 frame_width, u32 frame_height) {
    if (source_width == 0 || source_height == 0 || frame_width == 0 || frame_height == 0) {
        return {0, 0, frame_width, frame_height};
    }
    const double scale = std::min(static_cast<double>(frame_width) / source_width,
                                  static_cast<double>(frame_height) / source_height);
    const s32 width = std::clamp(static_cast<s32>(std::lround(source_width * scale)), 1,
                                 static_cast<s32>(frame_width));
    const s32 height = std::clamp(static_cast<s32>(std::lround(source_height * scale)), 1,
                                  static_cast<s32>(frame_height));
    return {static_cast<u32>((static_cast<s32>(frame_width) - width) / 2),
            static_cast<u32>((static_cast<s32>(frame_height) - height) / 2),
            static_cast<u32>(width), static_cast<u32>(height)};
}

PixelRect PresentRectOf(const Common::Rectangle<u32>& screen, u32 layout_width, u32 layout_height,
                        u32 source_width, u32 source_height, u32 frame_width, u32 frame_height) {
    if (layout_width == frame_width && layout_height == frame_height && screen.left < frame_width &&
        screen.top < frame_height) {
        const u32 right = std::min(screen.right, frame_width);
        const u32 bottom = std::min(screen.bottom, frame_height);
        if (right > screen.left && bottom > screen.top) {
            return {screen.left, screen.top, right - screen.left, bottom - screen.top};
        }
    }
    return FitPresentRect(source_width, source_height, frame_width, frame_height);
}

void ConvertFramePixel(const u8* source, FramePixelFormat format, Common::Hdr::Encoding encoding,
                       f32 paper_white_nits, u8* target) {
    switch (format) {
    case FramePixelFormat::Rgba16F: {
        const Common::Hdr::Rgb sdr = Common::Hdr::ToneMapToSdr(
            {HalfToFloat(source), HalfToFloat(source + 2), HalfToFloat(source + 4)}, encoding,
            paper_white_nits);
        target[0] = QuantizeUnorm8(sdr.b);
        target[1] = QuantizeUnorm8(sdr.g);
        target[2] = QuantizeUnorm8(sdr.r);
        target[3] = 0xff;
        break;
    }
    case FramePixelFormat::Rgba8:
        target[0] = source[2];
        target[1] = source[1];
        target[2] = source[0];
        target[3] = source[3];
        break;
    case FramePixelFormat::Bgra8:
        std::memcpy(target, source, 4);
        break;
    }
}

PixelRect ScreenRect(const Common::Rectangle<u32>& screen, u32 width, u32 height) {
    const u32 left = (std::min)(screen.left, width);
    const u32 right = (std::min)(screen.right, width);
    const u32 top = (std::min)(screen.top, height);
    const u32 bottom = (std::min)(screen.bottom, height);
    if (right <= left || bottom <= top) {
        return {0, 0, width, height};
    }
    return {left, top, right - left, bottom - top};
}

bool ConvertPicture(std::span<const u8> pixels, const Picture& picture, bool opaque,
                    std::span<u8> destination) {
    const size_t pixel_count = static_cast<size_t>(picture.width) * picture.height;
    const size_t pixel_size = BytesPerPixel(picture.format);
    if (pixel_count == 0 || destination.size() < pixel_count * 4 ||
        pixels.size() < pixel_count * pixel_size) {
        const size_t blackened = (std::min)(destination.size(), pixel_count * 4) & ~size_t{3};
        for (size_t i = 0; i < blackened; i += 4) {
            destination[i] = 0;
            destination[i + 1] = 0;
            destination[i + 2] = 0;
            destination[i + 3] = 0xff;
        }
        return false;
    }
    for (size_t i = 0; i < pixel_count; ++i) {
        u8* const target = destination.data() + i * 4;
        ConvertFramePixel(pixels.data() + i * pixel_size, picture.format, picture.encoding,
                          picture.paper_white_nits, target);
        if (opaque) {
            target[3] = 0xff;
        }
    }
    return true;
}

} // namespace Vulkan::Nvn
