// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string>

#include "common/common_types.h"
#include "video_core/nvn/sink.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

/// The `textures [N]` diagnostic request on the NVN path (see common/diag_request.h and
/// docs/NX2_DIAGNOSTICS.md). While `Common::DiagRequest::TextureBudget()` is active every draw the
/// replay prepares claims one unit and reports the textures its descriptors bind (`Note`); the
/// replay copies each not yet captured one to a host buffer when the draw's render pass has ended
/// (the Recorder's `HostDumpTexture`), the completion thread converts it to RGBA and writes
/// `tex-<NNN>-<address>.png` (`Complete`). `textures-nvn.json` (the index of every texture of every
/// claimed draw) is rewritten after each draw and each completed capture.
///
/// Mip 0 / layer 0 of the storage; textures wider or taller than 4096 are listed without a PNG; at
/// most 256 distinct textures (address + format + size) are captured per request.
namespace Vulkan::Nvn::DiagTextures {

/// One texture a draw binds.
struct Use {
    s32 stage{-1};         ///< NVN shader stage, -1 unknown
    s32 slot{-1};          ///< texture slot of the stage, -1 unknown
    s32 nvn_format{};      ///< NVNformat of the storage
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 levels{};
    u32 layers{};
    u64 address{};         ///< guest address of the texture storage (0 when its pool is unknown)
    const VideoCommon::Nvn::SamplerDesc* sampler{}; ///< null: no sampler (storage image)
    std::string note;
};

/// What the replay has to do for a noted use.
struct Job {
    u32 epoch{};
    u32 slot{};         ///< index of the capture slot (Complete / Skip)
    bool capture{false}; ///< read the texture back
};

/// A texture copy the completion thread converts (the replay filled the host buffer).
struct Readback {
    u32 epoch{};
    u32 slot{};
    VkFormat format{};
    VkImageAspectFlags aspect{};
    u32 width{};
    u32 height{};
    /// The host image holds BCn blocks (the device samples them natively): `pixel_format` decodes.
    bool bcn{false};
    VideoCore::Surface::PixelFormat pixel_format{};
    /// A BC4 / BC5 ASTC stand-in: its R / RG decode is (L) / (L, A), saved as (L, L, L, 1) /
    /// (L, L, L, A) as the game sees it.
    bool luminance{false};
};

/// Replay thread, at the start of a draw while the texture budget is active: claims one unit.
/// Returns false when the budget is exhausted.
bool BeginDraw();
/// Records a texture the current draw binds. The result says whether to read it back.
[[nodiscard]] Job Note(const Use& use);
/// All textures of the current draw are noted: rewrites the index.
void EndDraw();
/// The read back of a noted texture is not possible (`reason` goes to the index).
void Skip(const Job& job, const std::string& reason);
/// Completion thread: `pixels` is what the copy wrote (tightly packed rows of mip 0).
void Complete(const Readback& readback, std::span<const u8> pixels);

/// NVN format name (`RGBA8_SRGB`), or the number when it has none.
[[nodiscard]] std::string NvnFormatName(s32 format);

} // namespace Vulkan::Nvn::DiagTextures
