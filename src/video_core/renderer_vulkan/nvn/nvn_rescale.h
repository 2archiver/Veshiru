// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The Resolution setting on the NVN replay: the rules that decide which images are drawn at a
// scaled size and how pixel coordinates follow them. Pure functions and values, no device (tests:
// src/tests/nvn/rescale.cpp). The recording side is in nvn_rescale.cpp and nvn_exec.cpp; the
// design is in DESIGN.md ("Resolution setting").
//
// Model. A guest storage keeps its guest-sized image (everything the game can observe: sizes,
// queries, copies to and from memory, readbacks, overlapping textures, uploads). An eligible one
// also gets a host-only twin, an image of the same format at the scaled size. `State` says which of
// the two holds the newest pixels. Render passes that draw into scaled targets, the sampled reads
// of such passes and of compute, and the present read the twin; everything else reads the guest-
// sized image, which is brought up to date first (a downscale blit) when the twin is newer.
// The rules mirror the emulated GPU's texture cache (image_info.cpp `rescaleable`,
// texture_cache.h `RescaleRenderTargets` / `ImageCanRescale`, and the 'blacklist' of storage images
// a shader writes), with one deliberate difference for depth (see `PassScaled`).

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <span>

#include "common/common_types.h"
#include "common/settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/renderer_vulkan/vk_viewport_scale.h"

namespace Vulkan::Nvn::Rescale {

/// `up` / 2^`shift`: the Resolution setting as Settings::ResolutionScalingInfo holds it.
struct Factor {
    u32 up{1};
    u32 shift{0};

    [[nodiscard]] constexpr bool Active() const noexcept {
        return up != 1 || shift != 0;
    }
    /// ResolutionScalingInfo::ScaleUp: what the emulated GPU makes of a guest size (never 0 for a
    /// non-zero size). The scaled image extents and the shader's `Scale` use this same arithmetic.
    [[nodiscard]] constexpr u32 Scale(u32 value) const noexcept {
        if (value == 0) {
            return 0;
        }
        return (std::max)(static_cast<u32>((static_cast<u64>(value) * up) >> shift), 1u);
    }
    /// The start of a range of pixels (rounds down) and the end of one (rounds up): a scaled range
    /// covers every pixel the guest range covers, and a non-empty range stays non-empty.
    [[nodiscard]] constexpr s32 ScaleStart(s32 value) const noexcept {
        return static_cast<s32>((static_cast<s64>(value) * up) >> shift);
    }
    [[nodiscard]] constexpr s32 ScaleEnd(s32 value) const noexcept {
        return static_cast<s32>(
            (static_cast<s64>(value) * up + ((s64{1} << shift) - 1)) >> shift);
    }
    [[nodiscard]] constexpr float UpFactor() const noexcept {
        return static_cast<float>(up) / static_cast<float>(1u << shift);
    }
    /// The value of the shader's down factor push constant (guest pixels per scaled pixel).
    [[nodiscard]] constexpr float DownFactor() const noexcept {
        return static_cast<float>(1u << shift) / static_cast<float>(up);
    }
    [[nodiscard]] constexpr bool operator==(const Factor&) const noexcept = default;
};

[[nodiscard]] inline Factor CurrentFactor() noexcept {
    const Settings::ResolutionScalingInfo& info = Settings::values.resolution_info;
    return {info.up_scale, info.down_shift};
}

/// Which of a guest storage's two images holds the newest pixels.
enum class State : u8 {
    Native, ///< the guest-sized image (the twin, if any, is stale or was never written)
    Scaled, ///< the twin: the guest-sized image is stale
    Both,   ///< both hold the same pixels (one was made from the other)
};

[[nodiscard]] constexpr bool NativeIsStale(State state) noexcept {
    return state == State::Scaled;
}
[[nodiscard]] constexpr bool TwinIsCurrent(State state) noexcept {
    return state != State::Native;
}
/// What a write does to the pair. A write to one image makes the other stale.
[[nodiscard]] constexpr State AfterNativeWrite() noexcept {
    return State::Native;
}
[[nodiscard]] constexpr State AfterScaledWrite() noexcept {
    return State::Scaled;
}
/// A blit made one image from the other: they are equal.
[[nodiscard]] constexpr State AfterBlit() noexcept {
    return State::Both;
}

// ---------------------------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------------------------

/// The emulated GPU's RescaleHeightThreshold (image_info.cpp): a colour image smaller than this is
/// a lookup table or an effect buffer, not a picture.
inline constexpr u32 HEIGHT_THRESHOLD = 288;

/// What the rules look at in a storage that is bound as a render target.
struct Shape {
    bool two_d{false};      ///< NVN target 2D or 2D array (not 1D/3D/cube/rectangle/buffer)
    bool linear{false};     ///< pitch-linear storage
    bool alias{false};      ///< a block-view alias texture (TargetAlias), not the game's texture
    bool sparse{false};
    bool transcoded{false}; ///< a decoded stand-in of a compressed format
    bool color{true};       ///< false: depth and/or stencil
    bool blit_ok{true};     ///< the device can blit the format (both directions)
    u32 levels{1};
    u32 samples{1};
    u32 width{0};
    u32 height{0};
    u32 max_dimension{16384}; ///< the device's image and framebuffer limit
};

enum class Reason : u8 {
    Eligible,
    Inactive,     ///< the Resolution setting is 1X
    NotTwoD,      ///< 1D, 3D, cube, rectangle or buffer
    Linear,
    Alias,
    Sparse,
    Transcoded,
    Levels,       ///< a mip chain
    Multisampled,
    NoBlit,       ///< the device cannot blit the format: the emulated GPU would need a helper pass
    TooSmall,     ///< a colour image of at most HEIGHT_THRESHOLD rows
    TooLarge,     ///< the scaled size exceeds the device's limits
};

[[nodiscard]] constexpr const char* ReasonName(Reason reason) noexcept {
    switch (reason) {
    case Reason::Eligible:
        return "eligible";
    case Reason::Inactive:
        return "resolution 1X";
    case Reason::NotTwoD:
        return "not a 2D / 2D array texture";
    case Reason::Linear:
        return "pitch-linear";
    case Reason::Alias:
        return "block-view alias";
    case Reason::Sparse:
        return "sparse";
    case Reason::Transcoded:
        return "decoded stand-in of a compressed format";
    case Reason::Levels:
        return "mip chain";
    case Reason::Multisampled:
        return "multisampled";
    case Reason::NoBlit:
        return "format cannot be blitted";
    case Reason::TooSmall:
        return "colour image not taller than 288 rows";
    case Reason::TooLarge:
        return "scaled size beyond the device limits";
    }
    return "?";
}

/// Whether a storage of this shape may be drawn at the scaled size.
[[nodiscard]] constexpr Reason Classify(const Factor& factor, const Shape& shape) noexcept {
    if (!factor.Active()) {
        return Reason::Inactive;
    }
    if (!shape.two_d) {
        return Reason::NotTwoD;
    }
    if (shape.linear) {
        return Reason::Linear;
    }
    if (shape.alias) {
        return Reason::Alias;
    }
    if (shape.sparse) {
        return Reason::Sparse;
    }
    if (shape.transcoded) {
        return Reason::Transcoded;
    }
    if (shape.levels != 1) {
        return Reason::Levels;
    }
    if (shape.samples != 1) {
        return Reason::Multisampled;
    }
    if (!shape.blit_ok) {
        return Reason::NoBlit;
    }
    if (shape.color && shape.height <= HEIGHT_THRESHOLD) {
        return Reason::TooSmall;
    }
    if (factor.Scale(shape.width) > shape.max_dimension ||
        factor.Scale(shape.height) > shape.max_dimension) {
        return Reason::TooLarge;
    }
    return Reason::Eligible;
}

/// The native sizes of the colour targets of scaled passes seen so far (the "scene" sizes).
class SceneSizes {
public:
    void Add(u32 width, u32 height) {
        sizes.insert(Key(width, height));
    }
    [[nodiscard]] bool Contains(u32 width, u32 height) const {
        return sizes.contains(Key(width, height));
    }
    [[nodiscard]] size_t Size() const noexcept {
        return sizes.size();
    }

private:
    static constexpr u64 Key(u32 width, u32 height) noexcept {
        return (u64{width} << 32) | height;
    }
    std::set<u64> sizes;
};

/// Whether a render pass draws at the scaled size. As `RescaleRenderTargets`: every attachment must
/// be eligible, else the whole pass is native (a scaled image another attachment keeps native is
/// scaled down for the pass). Depth differs from the emulated GPU, which scales every eligible
/// depth image (a 4096x4096 shadow map at 2X would be 8192x8192): a depth image is scaled when the
/// pass also draws a colour target, or when it is as large as a colour target of an earlier scaled
/// pass (the scene depth of a depth-only pre-pass). A pass with no attachment is native.
[[nodiscard]] inline bool PassScaled(const Factor& factor, std::span<const Shape> colors,
                                     const Shape* depth, const SceneSizes& scene) {
    if (!factor.Active() || (colors.empty() && depth == nullptr)) {
        return false;
    }
    for (const Shape& shape : colors) {
        if (Classify(factor, shape) != Reason::Eligible) {
            return false;
        }
    }
    if (depth != nullptr) {
        if (Classify(factor, *depth) != Reason::Eligible) {
            return false;
        }
        if (colors.empty() && !scene.Contains(depth->width, depth->height)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Pixel coordinates of a scaled pass
// ---------------------------------------------------------------------------------------------

struct Rect {
    s32 x{};
    s32 y{};
    u32 width{};
    u32 height{};
    [[nodiscard]] constexpr bool operator==(const Rect&) const noexcept = default;
};

/// A rectangle of guest pixels (a scissor, a clear rectangle, the area of a copy) in the scaled
/// image: every scaled pixel the guest rectangle covers, so adjacent rectangles still tile.
[[nodiscard]] constexpr Rect ScaleRect(const Factor& factor, const Rect& rect) noexcept {
    const s32 x0 = factor.ScaleStart(rect.x);
    const s32 y0 = factor.ScaleStart(rect.y);
    // An empty axis stays empty (rounding its end up would give it a pixel).
    const s32 x1 = rect.width == 0 ? x0 : factor.ScaleEnd(rect.x + static_cast<s32>(rect.width));
    const s32 y1 = rect.height == 0 ? y0 : factor.ScaleEnd(rect.y + static_cast<s32>(rect.height));
    return {x0, y0, static_cast<u32>((std::max)(x1 - x0, 0)),
            static_cast<u32>((std::max)(y1 - y0, 0))};
}

/// The viewport the guest set, in the scaled image (a downscale rounds to whole pixels).
[[nodiscard]] inline VkViewport ScaleViewport(const Factor& factor, VkViewport viewport) {
    const float scale = factor.UpFactor();
    viewport.x = ScaleViewportValue(viewport.x, scale);
    viewport.y = ScaleViewportValue(viewport.y, scale);
    if (viewport.width > 0.0f) {
        // Vulkan needs a positive width: a downscale must not round it to zero.
        viewport.width = (std::max)(ScaleViewportValue(viewport.width, scale), 1.0f);
    }
    viewport.height = ScaleViewportValue(viewport.height, scale);
    return viewport;
}

/// A region of a texture-to-texture copy done between two scaled images: offsets scale like
/// viewport origins, the extent like an image size, so a copy of a whole image covers the whole
/// scaled image. Clamped to what both scaled images hold.
struct CopyRegion {
    s32 src_x{};
    s32 src_y{};
    s32 dst_x{};
    s32 dst_y{};
    u32 width{};
    u32 height{};
    [[nodiscard]] constexpr bool operator==(const CopyRegion&) const noexcept = default;
};

[[nodiscard]] constexpr CopyRegion ScaleCopy(const Factor& factor, const CopyRegion& region,
                                             u32 src_scaled_width, u32 src_scaled_height,
                                             u32 dst_scaled_width,
                                             u32 dst_scaled_height) noexcept {
    CopyRegion out{
        .src_x = factor.ScaleStart(region.src_x),
        .src_y = factor.ScaleStart(region.src_y),
        .dst_x = factor.ScaleStart(region.dst_x),
        .dst_y = factor.ScaleStart(region.dst_y),
        .width = factor.Scale(region.width),
        .height = factor.Scale(region.height),
    };
    const auto room = [](u32 size, s32 at) {
        return at < 0 ? 0u : (size > static_cast<u32>(at) ? size - static_cast<u32>(at) : 0u);
    };
    out.width = (std::min)({out.width, room(src_scaled_width, out.src_x),
                            room(dst_scaled_width, out.dst_x)});
    out.height = (std::min)({out.height, room(src_scaled_height, out.src_y),
                             room(dst_scaled_height, out.dst_y)});
    return out;
}

// ---------------------------------------------------------------------------------------------
// The shader side
// ---------------------------------------------------------------------------------------------

inline constexpr u32 PUSH_WORDS = Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS;
inline constexpr u32 TEXTURE_BITS = Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS * 32;
inline constexpr u32 IMAGE_BITS = Shader::Backend::SPIRV::NUM_IMAGE_SCALING_WORDS * 32;

/// The push constant words that tell a shader which of its textures and storage images are the
/// scaled twins (bit k of the texture words = the k-th texture descriptor of the pipeline in stage
/// order, `Bindings::texture_scaling_index`; the same for images): the layout of the emulated
/// GPU's RescalingPushConstant.
class Words {
public:
    using Data = std::array<u32, PUSH_WORDS>;

    void SetTexture(u32 index) noexcept {
        if (index < TEXTURE_BITS) {
            words[index / 32] |= 1u << (index % 32);
        }
    }
    void SetImage(u32 index) noexcept {
        if (index < IMAGE_BITS) {
            words[Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS + index / 32] |=
                1u << (index % 32);
        }
    }
    [[nodiscard]] bool TextureIsSet(u32 index) const noexcept {
        return index < TEXTURE_BITS && ((words[index / 32] >> (index % 32)) & 1u) != 0;
    }
    [[nodiscard]] bool ImageIsSet(u32 index) const noexcept {
        return index < IMAGE_BITS &&
               ((words[Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS + index / 32] >>
                 (index % 32)) &
                1u) != 0;
    }
    [[nodiscard]] bool Any() const noexcept {
        for (const u32 word : words) {
            if (word != 0) {
                return true;
            }
        }
        return false;
    }
    [[nodiscard]] const Data& Get() const noexcept {
        return words;
    }
    void Clear() noexcept {
        words = {};
    }

private:
    Data words{};
};

} // namespace Vulkan::Nvn::Rescale
