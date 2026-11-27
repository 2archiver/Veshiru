// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/renderer_vulkan/present/present_semaphore_ticks.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Vulkan {

class Device;
class Scheduler;

/// The HDR swapchain pair the surface lists: the preferred encoding, else the other one, else
/// nullopt (present SDR).
std::optional<VkSurfaceFormatKHR> ChooseHdrSurfaceFormat(
    std::span<const VkSurfaceFormatKHR> formats, Settings::HdrSwapchain preferred);

class Swapchain {
public:
    /// Creates the initial swapchain in the requested format (SDR unless HDR output is active).
    /// Throws on startup failure rather than constructing an unusable presenter.
    explicit Swapchain(
        VkSurfaceKHR_T* surface,
        const Device& device,
        Scheduler& scheduler,
        u32 width,
        u32 height,
        VkSurfaceFormatKHR requested = {VK_FORMAT_B8G8R8A8_UNORM,
                                        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
    ~Swapchain();

    /// Creates (or recreates) the swapchain; returns false when deferred for zero extent.
    bool Create(
        VkSurfaceKHR_T* surface,
        u32 width,
        u32 height,
        VkFormat requested_format = VK_FORMAT_B8G8R8A8_UNORM,
        VkColorSpaceKHR requested_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);

    /// Acquires the next image in the swapchain, waits as needed. Returns true when no image was
    /// acquired (out of date) and the swapchain has to be recreated first.
    /// `may_flush`: false on the async present thread (see Scheduler::Wait).
    /// `game_frame_presents`: total presents (generated + real) showing this game frame.
    /// `renderer_paced`: false for frames another submitter rendered (VTX-HLE's replay).
    bool AcquireNextImage(bool may_flush = true, u32 game_frame_presents = 1,
                          bool renderer_paced = true);

    /// Timeline semaphore the submission that waits on `CurrentPresentSemaphore` of a frame that
    /// is not renderer paced signals, with `NextPresentTimelineValue`; null unless VTX-HLE runs.
    /// Recorded by `MarkPresentTimelineSubmitted` once submitted.
    [[nodiscard]] VkSemaphore PresentTimeline() const {
        return *present_timeline;
    }

    [[nodiscard]] u64 NextPresentTimelineValue() const {
        return present_semaphore_ticks.Next();
    }

    /// The submission waiting on `CurrentPresentSemaphore` was submitted and signals
    /// `NextPresentTimelineValue`: the next acquire with this semaphore waits for it to have run.
    void MarkPresentTimelineSubmitted() {
        present_semaphore_ticks.Submitted(frame_index);
    }

    /// Presents the rendered image to the swapchain.
    /// True if Vulkan accepted this present (including suboptimal swapchains).
    [[nodiscard]] bool Present(VkSemaphore render_semaphore);

    /// Returns true when the swapchain needs to be recreated.
    bool NeedsRecreation() const {
        return !swapchain || IsSubOptimal() || NeedsPresentModeUpdate();
    }

    bool IsCreated() const {
        return static_cast<bool>(swapchain);
    }

    /// Returns true when the swapchain is outdated.
    bool IsOutDated() const {
        return is_outdated;
    }

    /// Returns true when the swapchain is suboptimal.
    bool IsSubOptimal() const {
        return is_suboptimal;
    }

    VkExtent2D GetSize() const {
        return extent;
    }

    std::size_t GetImageCount() const {
        return image_count;
    }

    std::size_t GetImageIndex() const {
        return image_index;
    }

    std::size_t GetFrameIndex() const {
        return frame_index;
    }

    VkImage GetImageIndex(std::size_t index) const {
        return images[index];
    }

    VkImage CurrentImage() const {
        return images[image_index];
    }

    VkFormat GetImageViewFormat() const {
        return image_view_format;
    }

    VkFormat GetImageFormat() const {
        return surface_format.format;
    }

    /// Hands the display the HDR10 mastering metadata (VK_EXT_hdr_metadata) of an HDR10 swapchain.
    void SetHdrMetadata(u16 peak_luminance, u16 reference_white) const;

    /// Returns the cached SDR surface format, including while an HDR swapchain is active.
    VkFormat GetSdrImageFormat() const {
        return sdr_image_format.load();
    }

    /// Returns the producer format requested by the latest creation attempt.
    VkFormat GetRequestedFormat() const {
        return requested_format;
    }

    VkColorSpaceKHR GetRequestedColorSpace() const {
        return requested_color_space;
    }

    VkSemaphore CurrentPresentSemaphore() const {
        return *present_semaphores[frame_index];
    }

    VkSemaphore CurrentRenderSemaphore() const {
        return *render_semaphores[image_index];
    }

    u32 GetWidth() const {
        return width;
    }

    u32 GetHeight() const {
        return height;
    }

    VkExtent2D GetExtent() const {
        return extent;
    }

private:
    void CreateSwapchain(const VkSurfaceCapabilitiesKHR& capabilities);
    void CreateSemaphores();
    void CreateImageViews();

    void Destroy();

    bool NeedsPresentModeUpdate() const;

    VkSurfaceKHR_T* surface;

    const Device& device;
    Scheduler& scheduler;

    vk::SwapchainKHR swapchain;

    std::size_t image_count{};
    std::vector<VkImage> images;
    std::vector<u64> resource_ticks;
    std::vector<vk::Semaphore> present_semaphores;
    std::vector<vk::Semaphore> render_semaphores;
    /// Present timeline (see PresentTimeline) and, per present semaphore, its value signalled
    /// by the submission that last waited on it.
    vk::Semaphore present_timeline;
    PresentSemaphoreTicks present_semaphore_ticks;

    u32 width;
    u32 height;

    u32 image_index{};
    u32 frame_index{};

    VkFormat image_view_format{};
    VkExtent2D extent{};
    VkPresentModeKHR present_mode{};
    VkSurfaceFormatKHR surface_format{};
    std::atomic<VkFormat> sdr_image_format{VK_FORMAT_B8G8R8A8_UNORM};
    VkFormat requested_format{VK_FORMAT_B8G8R8A8_UNORM};
    VkColorSpaceKHR requested_color_space{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    bool has_imm{false};
    bool has_mailbox{false};
    bool has_fifo_relaxed{false};

    bool is_outdated{};
    bool is_suboptimal{};
};

} // namespace Vulkan
