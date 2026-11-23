// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Capture and screenshot helpers of the NVN replay: the rectangle a present draws the guest's
// picture in, the rectangle a readback draws it in, and the conversion of the read-back picture
// into the BGRA image of a capture file or a RendererBase::RequestScreenshot.

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/hdr.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/nvn/nvn_screenshot.h"

namespace {

using Vulkan::Nvn::ConvertFramePixel;
using Vulkan::Nvn::ConvertPicture;
using Vulkan::Nvn::FramePixelFormat;
using Vulkan::Nvn::FitPresentRect;
using Vulkan::Nvn::Picture;
using Vulkan::Nvn::PixelRect;
using Vulkan::Nvn::PresentRectOf;
using Vulkan::Nvn::ScreenRect;

/// B G R A bytes of every pixel of a 4-byte pixel image.
std::vector<u8> Bgra(std::initializer_list<std::array<u8, 4>> pixels) {
    std::vector<u8> bytes;
    for (const auto& pixel : pixels) {
        bytes.insert(bytes.end(), pixel.begin(), pixel.end());
    }
    return bytes;
}

Picture Pic(u32 width, u32 height, FramePixelFormat format = FramePixelFormat::Bgra8) {
    Picture picture;
    picture.width = width;
    picture.height = height;
    picture.format = format;
    return picture;
}

} // namespace

TEST_CASE("NVN screenshot: present rectangle keeps the aspect ratio and centres", "[nvn][screenshot]") {
    // Same aspect: the whole frame.
    CHECK(FitPresentRect(1280, 720, 1920, 1080) == PixelRect{0, 0, 1920, 1080});
    // Wider frame: bars left and right.
    CHECK(FitPresentRect(1280, 720, 2000, 720) == PixelRect{360, 0, 1280, 720});
    // Taller frame: bars above and below (562.5 rounds to 563, the same formula the blit uses).
    CHECK(FitPresentRect(1280, 720, 1000, 1000) == PixelRect{0, 218, 1000, 563});
    // Unknown extents: the whole frame.
    CHECK(FitPresentRect(0, 720, 640, 360) == PixelRect{0, 0, 640, 360});
    CHECK(FitPresentRect(1280, 720, 0, 0) == PixelRect{0, 0, 0, 0});
    // Never empty, never outside the frame.
    const PixelRect sliver = FitPresentRect(10000, 1, 100, 100);
    CHECK(sliver.width == 100);
    CHECK(sliver.height == 1);
    CHECK(sliver.y + sliver.height <= 100);
}

TEST_CASE("NVN present rectangle follows the window layout's screen", "[nvn][screenshot]") {
    // 4:3 aspect ratio setting in a 1920x1080 window: the layout's pillarboxed screen rectangle,
    // whatever the game's picture size (the emulated GPU's presentation stretches it into it).
    const Common::Rectangle<u32> four_three{240, 0, 1680, 1080};
    CHECK(PresentRectOf(four_three, 1920, 1080, 1280, 720, 1920, 1080) ==
          PixelRect{240, 0, 1440, 1080});
    // Stretch: the whole frame even for a 4:3 picture.
    const Common::Rectangle<u32> stretch{0, 0, 1920, 1080};
    CHECK(PresentRectOf(stretch, 1920, 1080, 640, 480, 1920, 1080) ==
          PixelRect{0, 0, 1920, 1080});
    // A screen rectangle reaching past the frame is cut at its edge.
    CHECK(PresentRectOf(Common::Rectangle<u32>{100, 50, 2500, 1200}, 1920, 1080, 1280, 720, 1920,
                        1080) == PixelRect{100, 50, 1820, 1030});
    // No window (zero layout), a layout of another size than the frame (the window was resized
    // after the frame was acquired) or an empty rectangle: fit the picture into the frame.
    CHECK(PresentRectOf({}, 0, 0, 1280, 720, 1000, 1000) == FitPresentRect(1280, 720, 1000, 1000));
    CHECK(PresentRectOf(four_three, 1280, 720, 1280, 720, 1920, 1080) ==
          FitPresentRect(1280, 720, 1920, 1080));
    CHECK(PresentRectOf(Common::Rectangle<u32>{500, 0, 500, 1080}, 1920, 1080, 1280, 720, 1920,
                        1080) == FitPresentRect(1280, 720, 1920, 1080));
    CHECK(PresentRectOf(Common::Rectangle<u32>{1920, 0, 2000, 1080}, 1920, 1080, 1280, 720, 1920,
                        1080) == FitPresentRect(1280, 720, 1920, 1080));
}

TEST_CASE("CropRectOf is the part of the texture nvnWindowSetCrop shows", "[nvn][screenshot]") {
    using Vulkan::Nvn::CropRectOf;
    // Dynamic resolution: a 2560x1440 render in a 3840x2160 texture.
    CHECK(CropRectOf(0, 0, 2560, 1440, 3840, 2160) == PixelRect{0, 0, 2560, 1440});
    CHECK(CropRectOf(64, 32, 100, 50, 3840, 2160) == PixelRect{64, 32, 100, 50});
    // Unset (zero size) or empty or outside: the whole texture.
    CHECK(CropRectOf(0, 0, 0, 0, 3840, 2160) == PixelRect{0, 0, 3840, 2160});
    CHECK(CropRectOf(0, 0, -5, 10, 3840, 2160) == PixelRect{0, 0, 3840, 2160});
    CHECK(CropRectOf(4000, 0, 100, 100, 3840, 2160) == PixelRect{0, 0, 3840, 2160});
    // Past the edge: cut there. Negative origin: cut at zero.
    CHECK(CropRectOf(3800, 2100, 200, 200, 3840, 2160) == PixelRect{3800, 2100, 40, 60});
    CHECK(CropRectOf(-10, -10, 100, 100, 3840, 2160) == PixelRect{0, 0, 90, 90});
}

TEST_CASE("NVN screenshot: pixel conversion", "[nvn][screenshot]") {
    u8 out[4];
    const u8 bgra[4] = {1, 2, 3, 4};
    ConvertFramePixel(bgra, FramePixelFormat::Bgra8, Common::Hdr::Encoding::Srgb, 203.0f, out);
    CHECK(std::vector<u8>(out, out + 4) == std::vector<u8>{1, 2, 3, 4});

    ConvertFramePixel(bgra, FramePixelFormat::Rgba8, Common::Hdr::Encoding::Srgb, 203.0f, out);
    CHECK(std::vector<u8>(out, out + 4) == std::vector<u8>{3, 2, 1, 4});

    // R16G16B16A16_SFLOAT, sRGB encoded values: 1.0, 0.5, -1.0 (R G B), alpha is dropped.
    const u8 half[8] = {0x00, 0x3c, 0x00, 0x38, 0x00, 0xbc, 0x00, 0x00};
    ConvertFramePixel(half, FramePixelFormat::Rgba16F, Common::Hdr::Encoding::Srgb, 203.0f, out);
    CHECK(out[0] == 0);   // B: negative clamps to 0
    CHECK(out[1] == 128); // G: 0.5
    CHECK(out[2] == 255); // R: 1.0
    CHECK(out[3] == 255);
}

TEST_CASE("NVN screenshot: HDR frames are tone mapped like the frame capture", "[nvn][screenshot]") {
    // scRGB, 1.0 = 80 nits, paper white 80 nits: R G B = 1.0 0.25 0.0
    const u8 half[8] = {0x00, 0x3c, 0x00, 0x34, 0x00, 0x00, 0x00, 0x3c};
    const f32 paper_white = 80.0f;
    u8 out[4];
    ConvertFramePixel(half, FramePixelFormat::Rgba16F, Common::Hdr::Encoding::ScRgbLinear,
                      paper_white, out);
    const Common::Hdr::Rgb expected = Common::Hdr::ToneMapToSdr(
        {1.0f, 0.25f, 0.0f}, Common::Hdr::Encoding::ScRgbLinear, paper_white);
    const auto quantize = [](f32 value) {
        return static_cast<u8>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    };
    CHECK(out[0] == quantize(expected.b));
    CHECK(out[1] == quantize(expected.g));
    CHECK(out[2] == quantize(expected.r));
    CHECK(out[3] == 255);
    // The HDR source is really converted, not read as sRGB: dim linear light is brighter in sRGB.
    CHECK(out[1] > 64);
}

TEST_CASE("NVN screenshot: the readback draws the picture into the layout's screen rectangle",
          "[nvn][screenshot]") {
    // The nx2:vi2 capture layout at 16:9: the whole 1280x720 image.
    CHECK(ScreenRect({0, 0, 1280, 720}, 1280, 720) == PixelRect{0, 0, 1280, 720});
    // A 4:3 screen in a wide request: the rectangle itself (the picture is stretched into it,
    // as the emulated path's screenshot does).
    CHECK(ScreenRect({160, 0, 1120, 720}, 1280, 720) == PixelRect{160, 0, 960, 720});
    // Clamped to the image.
    CHECK(ScreenRect({100, 50, 5000, 5000}, 640, 360) == PixelRect{100, 50, 540, 310});
    // Empty or outside: the whole image.
    CHECK(ScreenRect({}, 640, 360) == PixelRect{0, 0, 640, 360});
    CHECK(ScreenRect({700, 0, 800, 360}, 640, 360) == PixelRect{0, 0, 640, 360});
}

TEST_CASE("NVN screenshot: a picture is converted pixel by pixel", "[nvn][screenshot]") {
    const std::vector<u8> pixels = Bgra({{10, 20, 30, 0}, {40, 50, 60, 7}, {70, 80, 90, 0}, {1, 2, 3, 0}});
    // Captures keep the alpha of an 8-bit picture, rows as read.
    std::vector<u8> kept(16, 0x55);
    REQUIRE(ConvertPicture(pixels, Pic(2, 2), false, kept));
    CHECK(kept == pixels);
    // Screenshots are opaque.
    std::vector<u8> opaque(16, 0x55);
    REQUIRE(ConvertPicture(pixels, Pic(2, 2), true, opaque));
    CHECK(opaque == Bgra({{10, 20, 30, 255}, {40, 50, 60, 255}, {70, 80, 90, 255}, {1, 2, 3, 255}}));
    // RGBA pictures are swapped to BGRA.
    std::vector<u8> swapped(4);
    REQUIRE(ConvertPicture(Bgra({{1, 2, 3, 255}}), Pic(1, 1, FramePixelFormat::Rgba8), true, swapped));
    CHECK(swapped == Bgra({{3, 2, 1, 255}}));
}

TEST_CASE("NVN screenshot: HDR pictures are tone mapped with their encoding", "[nvn][screenshot]") {
    // Two scRGB pixels, 1.0 = 80 nits: (1.0, 0.25, 0.0) and (0, 0, 0).
    const std::vector<u8> half{0x00, 0x3c, 0x00, 0x34, 0x00, 0x00, 0x00, 0x3c,
                               0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c};
    Picture picture = Pic(2, 1, FramePixelFormat::Rgba16F);
    picture.encoding = Common::Hdr::Encoding::ScRgbLinear;
    picture.paper_white_nits = 80.0f;
    std::vector<u8> out(8);
    REQUIRE(ConvertPicture(half, picture, false, out));
    u8 first[4];
    ConvertFramePixel(half.data(), FramePixelFormat::Rgba16F, Common::Hdr::Encoding::ScRgbLinear,
                      80.0f, first);
    CHECK(std::vector<u8>(out.begin(), out.begin() + 4) == std::vector<u8>(first, first + 4));
    CHECK(out[1] > 64); // converted, not read as sRGB
    CHECK(std::vector<u8>(out.begin() + 4, out.end()) == std::vector<u8>{0, 0, 0, 255});
}

TEST_CASE("NVN screenshot: a picture that does not fit fails and leaves black", "[nvn][screenshot]") {
    const std::vector<u8> pixels(3 * 4, 0xff); // 3 pixels for a 2x2 picture
    std::vector<u8> out(2 * 2 * 4, 0x55);
    CHECK_FALSE(ConvertPicture(pixels, Pic(2, 2), true, out));
    CHECK(out == Bgra({{0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}}));

    std::vector<u8> small(4);
    const std::vector<u8> picture_pixels(2 * 2 * 4, 0x80);
    CHECK_FALSE(ConvertPicture(picture_pixels, Pic(2, 2), true, small));
    CHECK(small == Bgra({{0, 0, 0, 255}}));
    CHECK_FALSE(ConvertPicture(picture_pixels, Pic(0, 0), true, out));
}

TEST_CASE("NVN screenshot: a request is claimed by exactly one renderer", "[nvn][screenshot]") {
    VideoCore::RendererSettings settings;
    CHECK_FALSE(settings.ClaimScreenshot()); // nothing requested
    settings.screenshot_requested = true;
    CHECK(settings.ClaimScreenshot());
    CHECK_FALSE(settings.ClaimScreenshot()); // the Composite path and the replay never both serve it
    settings.ReleaseScreenshotClaim();       // the claimant could not render it: still pending
    CHECK(settings.screenshot_requested);
    CHECK(settings.ClaimScreenshot());
    settings.FinishScreenshot();
    CHECK_FALSE(settings.screenshot_requested);
    CHECK_FALSE(settings.ClaimScreenshot()); // answered: nothing to serve
    settings.screenshot_requested = true;    // a new request can be claimed again
    CHECK(settings.ClaimScreenshot());
}

TEST_CASE("Screenshot: an HDR10 frame of an SDR scene comes out as the SDR frame",
          "[nvn][screenshot][hdr]") {
    // What an HDR10 swapchain frame holds for an SDR picture at the user's paper white: each
    // sRGB code value as light relative to paper white, BT.709 -> BT.2020, PQ encoded. The
    // screenshot must give the SDR picture back (the same scene as an SDR frame), not a washed
    // out or darkened one.
    using Common::Hdr::Encoding;
    for (const f32 paper_white : {80.0f, 203.0f, 400.0f}) {
        for (const Common::Hdr::Rgb sdr : {Common::Hdr::Rgb{0.05f, 0.05f, 0.05f},
                                           Common::Hdr::Rgb{0.2f, 0.4f, 0.8f},
                                           Common::Hdr::Rgb{0.8f, 0.4f, 0.2f},
                                           Common::Hdr::Rgb{0.5f, 0.5f, 0.5f},
                                           Common::Hdr::Rgb{0.85f, 0.85f, 0.85f}}) {
            const Common::Hdr::Rgb nits{Common::Hdr::SrgbToLinear(sdr.r) * paper_white,
                                        Common::Hdr::SrgbToLinear(sdr.g) * paper_white,
                                        Common::Hdr::SrgbToLinear(sdr.b) * paper_white};
            const Common::Hdr::Rgb bt2020 = Common::Hdr::Bt709ToBt2020(nits);
            const Common::Hdr::Rgb pq{Common::Hdr::PqEncode(bt2020.r),
                                      Common::Hdr::PqEncode(bt2020.g),
                                      Common::Hdr::PqEncode(bt2020.b)};
            const Common::Hdr::Rgb out =
                Common::Hdr::ToneMapToSdr(pq, Encoding::Pq2020, paper_white);
            CHECK(std::abs(out.r - sdr.r) < 2.0f / 255.0f);
            CHECK(std::abs(out.g - sdr.g) < 2.0f / 255.0f);
            CHECK(std::abs(out.b - sdr.b) < 2.0f / 255.0f);
        }
        // SDR white: bright (not the washed out grey of reading PQ as sRGB) and not clipped.
        const f32 white = Common::Hdr::PqEncode(paper_white);
        const f32 mapped =
            Common::Hdr::ToneMapToSdr({white, white, white}, Encoding::Pq2020, paper_white).g;
        CHECK(mapped > 0.93f);
        CHECK(mapped < 1.0f);
    }
}
