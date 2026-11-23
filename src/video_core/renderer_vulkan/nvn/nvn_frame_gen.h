// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "video_core/vulkan_common/vulkan.h"

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/renderer_vulkan/nvn/nvn_features.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
class MemoryAllocator;
class PassRecorder;
struct Frame;
} // namespace Vulkan

namespace Vulkan::Nvn {

/// What the replay's present does about the frame generation settings of the current frame.
enum class PresentFrameGenAction {
    Off,         ///< Frame generation is off.
    Dlss,        ///< DLSS frame generation: its inputs are recorded with the present.
    Interpolate, ///< FSR3 or Lossless Scaling: the chain is recorded with the present and the
                 ///< generated frames are presented before the real one.
    NotApplied,  ///< Frame generation is not part of this build (FrameGenTypeSupported).
};

[[nodiscard]] constexpr PresentFrameGenAction ClassifyPresentFrameGen(
    bool frame_gen_enabled, Settings::FrameGenType type) {
    if (!frame_gen_enabled) {
        return PresentFrameGenAction::Off;
    }
    if (!FrameGenTypeSupported(type)) {
        return PresentFrameGenAction::NotApplied;
    }
    return PresentFrameGenAction::Interpolate;
}

[[nodiscard]] constexpr const char* FrameGenTypeName(Settings::FrameGenType type) {
    switch (type) {
    case Settings::FrameGenType::Lsfg:
        return "Lsfg";
    case Settings::FrameGenType::Fsr3:
        return "Fsr3";
    case Settings::FrameGenType::NvidiaDlssG:
        return "NvidiaDlssG";
    case Settings::FrameGenType::OpenFrameGen:
        return "OpenFrameGen";
    }
    return "unknown";
}

/// Reports a type the HLE path does not apply once per type: the first present that sees it, and
/// again only after the user switched to another unsupported type.
class NotAppliedWarning {
public:
    [[nodiscard]] bool ShouldWarn(Settings::FrameGenType type) {
        if (warned == type) {
            return false;
        }
        warned = type;
        return true;
    }

private:
    std::optional<Settings::FrameGenType> warned;
};

/// The replay's lend of one present: where its work is recorded and how long it lives.
struct PresentRecording {
    vk::CommandBuffer cmdbuf;
    /// Sequence number of the submission carrying `cmdbuf`, and the non-blocking query of whether
    /// a submission has completed on the GPU.
    u64 seq{};
    std::function<bool(u64)> is_complete;
    /// Objects kept alive until the submission completes.
    std::vector<std::shared_ptr<void>>* keepalive{};
};

/// Frame generation of the frames the replay presents (the HLE counterpart of what
/// RendererVulkan::Composite does with FrameGen). Eglantine's FrameGen records into the emulated
/// GPU's scheduler, so no type is applied here (FrameGenTypeSupported): the replay presents its
/// frames as they are and warns once per type.
///
/// Replay thread only.
class PresentFrameGen {
public:
    PresentFrameGen();
    ~PresentFrameGen();

    void Attach(MemoryAllocator& replay_allocator, PassRecorder& recorder);

    /// Returns how many frames FSR3 / Lossless Scaling generate for this present (GenerateInto()
    /// fills them): always 0 here.
    size_t Process(const Device& device, Frame& frame, VkFormat swapchain_format,
                   VkExtent2D guest_extent, bool present_thread, const PresentRecording& recording,
                   size_t max_extra_frames);

    void GenerateInto(const Device& device, Frame& destination, size_t generation);

    /// Brackets the queue submission of the frame. No-ops.
    void RenderSubmitStart() {}
    void RenderSubmitEnd() {}

private:
    NotAppliedWarning not_applied_warning;
};

} // namespace Vulkan::Nvn
