// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/nvn/nvn_frame_gen.h"

namespace Vulkan::Nvn {

PresentFrameGen::PresentFrameGen() = default;

PresentFrameGen::~PresentFrameGen() = default;

void PresentFrameGen::Attach(MemoryAllocator& replay_allocator, PassRecorder& recorder) {
    static_cast<void>(replay_allocator);
    static_cast<void>(recorder);
}

size_t PresentFrameGen::Process(const Device& device, Frame& frame, VkFormat swapchain_format,
                                VkExtent2D guest_extent, bool present_thread,
                                const PresentRecording& recording, size_t max_extra_frames) {
    static_cast<void>(device);
    static_cast<void>(frame);
    static_cast<void>(swapchain_format);
    static_cast<void>(guest_extent);
    static_cast<void>(present_thread);
    static_cast<void>(recording);
    static_cast<void>(max_extra_frames);
    const Settings::FrameGenType type = Settings::values.frame_gen_type.GetValue();
    if (ClassifyPresentFrameGen(Settings::values.frame_gen.GetValue(), type) ==
            PresentFrameGenAction::NotApplied &&
        not_applied_warning.ShouldWarn(type)) {
        LOG_WARNING(Render_Vulkan,
                    "[nvn] frame generation ({}) is not available on VTX-HLE; the frames are "
                    "presented as they are",
                    FrameGenTypeName(type));
    }
    return 0;
}

void PresentFrameGen::GenerateInto(const Device& device, Frame& destination, size_t generation) {
    static_cast<void>(device);
    static_cast<void>(destination);
    static_cast<void>(generation);
}

} // namespace Vulkan::Nvn
