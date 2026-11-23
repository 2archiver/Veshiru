// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Pure pixel work of the NVN replay's captures and screenshots: where a present puts the guest's
// picture, where a readback puts it, and how a read-back picture becomes the BGRA image of a
// capture file or a RendererBase::RequestScreenshot. No Vulkan here: the replay blits the picture
// into an image of the requested size and reads it back, the helpers below convert.

#pragma once

#include <cstddef>
#include <span>

#include "common/common_types.h"
#include "common/hdr.h"
#include "common/math_util.h"

namespace Vulkan::Nvn {

/// A rectangle of pixels (top-left origin, rows down).
struct PixelRect {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};

    [[nodiscard]] constexpr bool operator==(const PixelRect&) const = default;
};

/// The part of a `source_width` x `source_height` texture that nvnWindowSetCrop shows: the crop
/// clipped to the texture, the whole texture when the crop is unset (zero or negative size) or
/// lies outside it.
[[nodiscard]] PixelRect CropRectOf(s32 x, s32 y, s32 width, s32 height, u32 source_width,
                                   u32 source_height);

/// Where a present draws a `source` sized picture in a `frame` sized image: scaled by the same
/// factor on both axes to fit, centred; the rest of the frame is black. A zero extent gives the
/// whole frame.
[[nodiscard]] PixelRect FitPresentRect(u32 source_width, u32 source_height, u32 frame_width,
                                       u32 frame_height);

/// Where a present draws the picture: the window layout's screen rectangle (the aspect ratio
/// setting, as the emulated GPU's presentation places it), when the layout describes a frame of
/// this size and the rectangle has area inside it; FitPresentRect otherwise (no window, or a frame
/// sized for an earlier window size).
[[nodiscard]] PixelRect PresentRectOf(const Common::Rectangle<u32>& screen, u32 layout_width,
                                      u32 layout_height, u32 source_width, u32 source_height,
                                      u32 frame_width, u32 frame_height);

/// Memory layout of a read-back frame pixel.
enum class FramePixelFormat : u8 {
    Bgra8,    ///< 4 bytes, B G R A (B8G8R8A8, also what an unknown frame format is treated as)
    Rgba8,    ///< 4 bytes, R G B A
    Rgba16F,  ///< 8 bytes, four IEEE binary16 R G B A (the HDR frame)
};

[[nodiscard]] constexpr size_t BytesPerPixel(FramePixelFormat format) {
    return format == FramePixelFormat::Rgba16F ? 8 : 4;
}

/// Writes the pixel at `source` as B G R A to `target[0..3]`. Half float pixels hold values in
/// `encoding`: they are tone mapped to sRGB (Common::Hdr::ToneMapToSdr, as the frame capture does)
/// and get alpha 0xff; the 8-bit formats keep their alpha.
void ConvertFramePixel(const u8* source, FramePixelFormat format, Common::Hdr::Encoding encoding,
                       f32 paper_white_nits, u8* target);

/// A picture read back from the GPU, rows tightly packed, top row first.
struct Picture {
    u32 width{};
    u32 height{};
    FramePixelFormat format{FramePixelFormat::Bgra8};
    /// What the half float values are (ignored for the 8-bit formats).
    Common::Hdr::Encoding encoding{Common::Hdr::Encoding::Srgb};
    f32 paper_white_nits{static_cast<f32>(Common::Hdr::ReferencePaperWhiteNits)};
};

/// Where a readback of a `width` x `height` image draws the guest's picture: `screen`
/// (Layout::FramebufferLayout::screen) clamped to the image, the whole image when it is empty.
/// The emulated path's screenshots stretch the guest layer into the same rectangle.
[[nodiscard]] PixelRect ScreenRect(const Common::Rectangle<u32>& screen, u32 width, u32 height);

/// Converts `picture` to `destination` (width * height * 4 bytes, B G R A, rows as read) with
/// ConvertFramePixel. `opaque` sets every alpha to 0xff (screenshots are RGB32), else the 8-bit
/// formats keep theirs. Returns false when a buffer is too small or an extent is zero;
/// `destination` is then black as far as it has room.
[[nodiscard]] bool ConvertPicture(std::span<const u8> pixels, const Picture& picture, bool opaque,
                                  std::span<u8> destination);

} // namespace Vulkan::Nvn
