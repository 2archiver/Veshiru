// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/frame_gen_choice.h"
#include "video_core/renderer_vulkan/present/frame_gen.h"
#include "video_core/renderer_vulkan/present/open_frame_gen_policy.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vk_enum_string_helper.h"
#include "video_core/vulkan_common/vulkan_device.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_frame_gen.h"
#endif

namespace Vulkan {

namespace {

constexpr size_t COLOR_CHANNELS = 4;
constexpr u64 LSFG_REQUIRED_FRAMES = 2;
constexpr u32 LSFG_RECURRENCE_FRAMES = 2;
constexpr auto ACTIVITY_LOG_INTERVAL = std::chrono::seconds(10);

[[nodiscard]] std::string ModeText(Settings::FrameGenType active) {
    const char* type = active == Settings::FrameGenType::Fsr3           ? "FSR3"
                       : active == Settings::FrameGenType::OpenFrameGen ? "Open-Frame-Gen"
                                                                        : "LSFG";
    const u32 target_rate = Settings::values.frame_gen_target_rate.GetValue();
    if (target_rate != 0) {
        return fmt::format("{} up to {}x towards {} FPS", type, Settings::MAX_FRAME_GEN_MULTIPLIER,
                           target_rate);
    }
    return fmt::format("{} {}x", type, Settings::FrameGenMultiplier());
}

/// Why NVIDIA DLSS Frame Generation isn't this session's frame generator, if it isn't. Streamline
/// decided that when the game started.
[[nodiscard]] VideoCore::DlssgBlock DlssgBlocked() {
#ifdef EGLANTINE_STREAMLINE
    if (Streamline::FrameGenSession()) {
        return VideoCore::DlssgBlock::None;
    }
    return VideoCore::BlockedDlssg({
        .streamline = true,
        .dlss = Settings::values.nvidia_dlss.GetValue(),
        .program_id = Settings::GetCurrentProgramID(),
        .loaded = false,
    });
#else
    return VideoCore::DlssgBlock::NoStreamline;
#endif
}

[[nodiscard]] std::string DlssgBlockedText(VideoCore::DlssgBlock block) {
    switch (block) {
    case VideoCore::DlssgBlock::None:
        break;
    case VideoCore::DlssgBlock::NoStreamline:
        return "this build has no NVIDIA Streamline";
    case VideoCore::DlssgBlock::DlssOff:
        return "\"Use NVIDIA DLSS\" is off, and DLSS-G takes its inputs from DLSS";
    case VideoCore::DlssgBlock::NoGameInputs:
        return fmt::format("no game motion vectors for {:016X}", Settings::GetCurrentProgramID());
    case VideoCore::DlssgBlock::NotLoaded:
#ifdef EGLANTINE_STREAMLINE
        if (std::string reason = Streamline::FrameGenUnavailableReason(); !reason.empty()) {
            return reason;
        }
#endif
        return "NVIDIA Streamline did not start";
    }
    return {};
}

[[nodiscard]] f32 ManualFlowScale() {
    return static_cast<f32>(Settings::values.frame_gen_flow_scale.GetValue()) / 100.0f;
}

[[nodiscard]] f32 ConfiguredFlowScale(VkExtent2D guest_extent, VkExtent2D presented_extent) {
    if (!Settings::values.frame_gen_flow_scale_auto.GetValue()) {
        return ManualFlowScale();
    }
    if (guest_extent.width == 0 || presented_extent.width == 0) {
        return 1.0f;
    }

    const f32 rendered_width = static_cast<f32>(guest_extent.width) *
                               Settings::values.resolution_info.up_factor;
    const f32 ratio = rendered_width / static_cast<f32>(presented_extent.width);

    constexpr f32 FLOW_SCALE_STEPS = 20.0f;
    const f32 stepped = std::ceil(ratio * FLOW_SCALE_STEPS) / FLOW_SCALE_STEPS;
    return std::clamp(stepped, 0.25f, 1.0f);
}

bool IsBlueFirst(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
}

VkDeviceSize BytesPerTexel(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
        return 1;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    default:
        return COLOR_CHANNELS;
    }
}

void WritePortablePixmap(const std::filesystem::path& path, const std::string& magic,
                         VkExtent2D extent, std::span<const u8> pixels) {
    Common::FS::IOFile file{path, Common::FS::FileAccessMode::Write,
                            Common::FS::FileType::BinaryFile};
    if (!file.IsOpen()) {
        return;
    }

    const std::string header = magic + "\n" + std::to_string(extent.width) + " " +
                               std::to_string(extent.height) + "\n255\n";
    if (file.Write(header) != header.size()) {
        return;
    }

    void(file.Write(pixels));
    void(file.Flush());
}

void WriteGrayscalePgm(const std::filesystem::path& path, VkExtent2D extent,
                       std::span<const u8> pixels) {
    const size_t expected = static_cast<size_t>(extent.width) * extent.height;
    WritePortablePixmap(path, "P5", extent, pixels.subspan(0, (std::min)(expected, pixels.size())));
}

void WriteRaw(const std::filesystem::path& path, std::span<const u8> pixels) {
    Common::FS::IOFile file{path, Common::FS::FileAccessMode::Write,
                            Common::FS::FileType::BinaryFile};
    if (!file.IsOpen()) {
        return;
    }
    void(file.Write(pixels));
    void(file.Flush());
}

void WriteColorPpm(const std::filesystem::path& path, VkExtent2D extent,
                   std::span<const u8> pixels, bool blue_first) {
    const size_t pixel_count = static_cast<size_t>(extent.width) * extent.height;
    if (pixels.size() < pixel_count * COLOR_CHANNELS) {
        return;
    }

    std::vector<u8> rgb(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; ++i) {
        const u8 first = pixels[i * COLOR_CHANNELS];
        const u8 green = pixels[i * COLOR_CHANNELS + 1];
        const u8 third = pixels[i * COLOR_CHANNELS + 2];
        rgb[i * 3] = blue_first ? third : first;
        rgb[i * 3 + 1] = green;
        rgb[i * 3 + 2] = blue_first ? first : third;
    }

    WritePortablePixmap(path, "P6", extent, rgb);
}

VkImageMemoryBarrier MakeTransitionBarrier(VkImage image, VkAccessFlags src_access,
                                           VkAccessFlags dst_access, VkImageLayout old_layout,
                                           VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
}

VkImageCopy MakeCopyRegion(VkExtent2D extent) {
    return VkImageCopy{
        .srcSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .srcOffset = {},
        .dstSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .dstOffset = {},
        .extent = {.width = extent.width, .height = extent.height, .depth = 1},
    };
}

void CopyPresentedFrame(vk::CommandBuffer cmdbuf, VkImage source, LsfgImage& destination,
                        VkExtent2D extent) {
    const auto make_barrier = MakeTransitionBarrier;

    const std::array before{
        make_barrier(source, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        make_barrier(destination.Handle(), VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                     destination.Layout(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);

    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination.Handle(),
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, MakeCopyRegion(extent));

    const std::array after{
        make_barrier(source, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        make_barrier(destination.Handle(), VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, {}, {}, after);

    destination.SetLayout(VK_IMAGE_LAYOUT_GENERAL);
}

} // Anonymous namespace

FrameGen::FrameGen(MemoryAllocator& memory_allocator_, Scheduler& scheduler_)
    : memory_allocator{memory_allocator_}, scheduler{scheduler_} {}

FrameGen::~FrameGen() = default;

void FrameGen::Process(const Device& device, PresentManager& present, Frame* frame,
                       VkExtent2D guest_extent) {
    using VideoCore::FrameGenState;
    generated = false;
    const VkFormat format = frame->image_format;
    if (!Settings::values.frame_gen.GetValue()) {
        ResetHistory();
        blocked_config.reset();
        SetState(FrameGenState::Pending, format);
        return;
    }

    const auto chosen = Settings::values.frame_gen_type.GetValue();
    const auto type = chosen == Settings::FrameGenType::NvidiaDlssG
                          ? VideoCore::ActiveFrameGenType(chosen, DlssgBlocked())
                          : chosen;
    if (type == Settings::FrameGenType::NvidiaDlssG) {
        ResetHistory();
        SetState(FrameGenState::NvidiaFrameGen, format);
        return;
    }
    const bool use_open_fg = type == Settings::FrameGenType::OpenFrameGen;
    if (use_open_fg && frame->color_space != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        ResetHistory();
        SetState(FrameGenState::OpenFrameGenHdr, format);
        return;
    }
    if (!use_open_fg && !frame->storage_view) {
        ResetHistory();
        SetState(FrameGenState::UnsupportedSurface, format);
        return;
    }

    peak_guest_extent.width = (std::max)(peak_guest_extent.width, guest_extent.width);
    peak_guest_extent.height = (std::max)(peak_guest_extent.height, guest_extent.height);
    const VkExtent2D extent{.width = frame->width, .height = frame->height};
    const Config config{extent.width, extent.height, format,
                        ConfiguredFlowScale(peak_guest_extent, extent), type};
    if (blocked_config == config) {
        SetState(blocked_state, format);
        return;
    }
    if (current_config != config) {
        ResetHistory();
        void(present.TakeBlockedTime());
        blocked_config.reset();
        if (use_open_fg) {
            const auto physical = device.GetPhysical();
            if (!CanRunOpenFrameGen(physical.GetFormatProperties3(format).optimalTilingFeatures,
                                    physical.GetFormatProperties3(VK_FORMAT_R16G16B16A16_SFLOAT)
                                        .optimalTilingFeatures,
                                    device.IsFormatlessImageStoreSupported(),
                                    physical.GetProperties().limits, extent)) {
                blocked_config = config;
                blocked_state = FrameGenState::OpenFrameGenUnsupported;
                SetState(blocked_state, format);
                return;
            }
        }
        current_config = config;
        try {
            if (type == Settings::FrameGenType::Lsfg) {
                if (!shaders) {
                    shaders.emplace(device);
                }
                if (!shaders->IsValid()) {
                    blocked_config = config;
                    blocked_state = shaders->IsDeviceSupported() ? FrameGenState::MissingLosslessDll
                                                                 : FrameGenState::UnsupportedGpu;
                    ResetHistory();
                    SetState(blocked_state, format);
                    return;
                }
            }
            Rebuild(device, config);
        } catch (const vk::Exception& error) {
            if (error.GetResult() == VK_ERROR_DEVICE_LOST) {
                throw;
            }
            LOG_DEBUG(Render_Vulkan, "Could not build frame generation resources: {}",
                      error.what());
            ResourceFailed(format);
            return;
        }
    }

    SetState(FrameGenState::Running, format);
    PlanGenerations(present);
    const u64 count = frame_count++;
    last_count = count;
    last_generations = plan.generations;

    const bool warm = plan.warm && count + 1 >= LSFG_REQUIRED_FRAMES;
    warm_streak = warm ? warm_streak + 1 : 0;
    const bool want_generate = warm && warm_streak >= LSFG_RECURRENCE_FRAMES && plan.generations > 0;

    const bool use_fsr3 = type == Settings::FrameGenType::Fsr3;
    const bool capture_history = plan.warm;
    const bool compute_flow = want_generate;

    if (use_fsr3 && fsr3_chain) {
        if (capture_history || compute_flow) {
            scheduler.RequestOutsideRenderPassOperationContext();
        }
        fsr3_chain->Process(device, scheduler, *frame->image, *frame->storage_view, count,
                            capture_history, compute_flow);
        generated = want_generate && fsr3_chain->IsReady();
    } else if (use_open_fg && open_fg_chain) {
        if (capture_history || compute_flow) {
            scheduler.RequestOutsideRenderPassOperationContext();
        }
        try {
            open_fg_chain->Process(scheduler, *frame->image, count, capture_history, compute_flow);
            generated = want_generate && open_fg_chain->IsReady();
        } catch (const vk::Exception& error) {
            if (error.GetResult() == VK_ERROR_DEVICE_LOST) {
                throw;
            }
            LOG_DEBUG(Render_Vulkan, "Open-Frame-Gen could not prepare optical flow: {}",
                      error.what());
            ResourceFailed(format);
        }
    } else if (chain) {
        generated = want_generate;
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([this, source = *frame->image, extent, count,
                          dispatch = warm](vk::CommandBuffer cmdbuf) {
            CopyPresentedFrame(cmdbuf, source, chain->Input(count), extent);
            if (dispatch) {
                chain->DispatchShared(cmdbuf, count);
            }
        });

        const bool dump_requested = generated && Settings::values.frame_gen_dump_flow.GetValue();
        if (!dump_requested) {
            dumped = false;
        } else if (!dumped) {
            DumpDebugImages(count);
            dumped = true;
        }
    }
}

void FrameGen::CompleteFrame(size_t queued_generations, const PresentManager& present) {
    if (queued_generations != 0) {
        generated_frames.fetch_add(queued_generations, std::memory_order_relaxed);
    }
    if (state.load(std::memory_order_relaxed) == VideoCore::FrameGenState::Running) {
        LogActivity(queued_generations, present.PresentCounts());
    }
}

void FrameGen::ResourceFailed(VkFormat format) {
    blocked_config = current_config;
    blocked_state = VideoCore::FrameGenState::ResourceFailure;
    ResetHistory();
    SetState(blocked_state, format);
}

void FrameGen::SetState(VideoCore::FrameGenState new_state, VkFormat format) {
    if (state.exchange(new_state, std::memory_order_relaxed) == new_state) {
        return;
    }
    switch (new_state) {
    case VideoCore::FrameGenState::Pending:
        LOG_INFO(Render_Vulkan, "Frame generation off; no new generated frames");
        break;
    case VideoCore::FrameGenState::Running:
        if (Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::NvidiaDlssG) {
            LOG_INFO(Render_Vulkan,
                     "Frame generation: NVIDIA DLSS-G unavailable ({}), using {}, presenting {}",
                     DlssgBlockedText(DlssgBlocked()), ModeText(Settings::FrameGenType::Fsr3),
                     string_VkFormat(format));
            break;
        }
        LOG_INFO(Render_Vulkan, "Frame generation on: {}, presenting {}",
                 ModeText(Settings::values.frame_gen_type.GetValue()), string_VkFormat(format));
        break;
#ifdef EGLANTINE_STREAMLINE
    case VideoCore::FrameGenState::NvidiaFrameGen:
        LOG_INFO(Render_Vulkan,
                 "Frame generation on: NVIDIA DLSS-G {}x (numFramesToGenerateMax {}), presenting "
                 "{}",
                 Streamline::FrameGenGenerated() + 1, Streamline::FrameGenMaxGenerated(),
                 string_VkFormat(format));
        break;
#endif
    case VideoCore::FrameGenState::UnsupportedSurface:
        LOG_WARNING(Render_Vulkan,
                    "Frame generation is on but not running: the GPU can't write {} from a compute "
                    "shader, so generated frames can't be presented",
                    string_VkFormat(format));
        break;
    case VideoCore::FrameGenState::MissingLosslessDll:
        LOG_WARNING(Render_Vulkan,
                    "Frame generation is on but not running: LSFG needs Lossless.dll in the "
                    "lossless folder of the user directory. FSR3 needs nothing");
        break;
    case VideoCore::FrameGenState::OpenFrameGenHdr:
        LOG_INFO(Render_Vulkan,
                 "Open-Frame-Gen is paused during HDR output; presenting game frames");
        break;
    case VideoCore::FrameGenState::OpenFrameGenUnsupported:
        LOG_WARNING(Render_Vulkan,
                    "Open-Frame-Gen is unavailable: unsupported Vulkan compute or "
                    "storage image capabilities for {}; presenting game frames",
                    string_VkFormat(format));
        break;
    case VideoCore::FrameGenState::ResourceFailure:
        LOG_WARNING(Render_Vulkan, "Could not prepare frame generation resources; presenting game "
                                   "frames. Toggle frame generation off and on to retry");
        break;
    case VideoCore::FrameGenState::UnsupportedGpu:
        LOG_WARNING(Render_Vulkan,
                    "Frame generation is on but not running: this GPU lacks Vulkan features LSFG "
                    "needs. FSR3 may still work");
        break;
    default:
        break;
    }
    activity = {};
}

void FrameGen::LogActivity(size_t queued_generations, VideoCore::FramePresentCounts presents) {
    const auto now = std::chrono::steady_clock::now();
    if (activity.since == std::chrono::steady_clock::time_point{}) {
        activity.since = now;
        activity.presents = presents;
        return;
    }
    ++activity.frames;
    activity.generated += queued_generations;
    if (plan.hitch_skipped) {
        ++activity.hitch_skips;
    }
    if (now - activity.since < ACTIVITY_LOG_INTERVAL) {
        return;
    }
    const f32 per_frame =
        std::round(10.0f * static_cast<f32>(activity.generated) / static_cast<f32>(activity.frames)) /
        10.0f;
    if (per_frame != activity.logged_per_frame) {
        const f32 seconds = std::chrono::duration<f32>(now - activity.since).count();
        const f32 game_fps = static_cast<f32>(activity.frames) / seconds;
        const f32 gen_fps = static_cast<f32>(activity.generated) / seconds;
        const auto rates = VideoCore::PresentRatesBetween(activity.presents, presents, seconds);
        LOG_INFO(Render_Vulkan,
                 "Frame generation ({}) {}: {:.1f} game FPS, {:.1f} generated queued FPS; "
                 "Vulkan presents/s: {:.1f} original + {:.1f} generated ({:.1f} queued per game "
                 "frame, limit {}, {} hitch skips, {} display trims)",
                 ModeText(current_config->type), activity.generated ? "active" : "idle", game_fps,
                 gen_fps, rates.original, rates.generated, per_frame, pacer.CurrentLimit(),
                 activity.hitch_skips, activity.display_trims);
        activity.logged_per_frame = per_frame;
    }
    activity.since = now;
    activity.presents = presents;
    activity.frames = 0;
    activity.generated = 0;
    activity.hitch_skips = 0;
    activity.display_trims = 0;
}

void FrameGen::PlanGenerations(PresentManager& present) {
    plan = pacer.Plan(present.MaxExtraFrames(), present.TakeBlockedTime());
    const size_t without_waiting =
        GenerationsWithoutWaiting(plan.generations, present.FreeFrames());
    if (without_waiting < plan.generations) {
        ++activity.display_trims;
    }
    plan.generations = without_waiting;
}

size_t FrameGen::PlannedGenerations() const {
    return generated ? last_generations : 0;
}

void FrameGen::GenerateInto(const Device& device, Frame* destination, size_t generation) {
    const bool use_fsr3 = current_config->type == Settings::FrameGenType::Fsr3;
    const bool use_open_fg = current_config->type == Settings::FrameGenType::OpenFrameGen;

    if (use_fsr3 && fsr3_chain) {
        const f32 blend = static_cast<f32>(generation + 1) / static_cast<f32>(last_generations + 1);
        const f32 flow_scale = current_config->scale;

        scheduler.RequestOutsideRenderPassOperationContext();
        fsr3_chain->Generate(device, scheduler, *destination->image, *destination->storage_view,
                             blend, flow_scale);
    } else if (use_open_fg && open_fg_chain) {
        const f32 blend = static_cast<f32>(generation + 1) / static_cast<f32>(last_generations + 1);

        scheduler.RequestOutsideRenderPassOperationContext();
        open_fg_chain->Generate(device, scheduler, *destination->image, *destination->storage_view,
                                destination->allocation_id, blend);
    } else if (chain) {
        chain->SetTarget(device, last_generations, generation, destination->index,
                         *destination->storage_view);

        const VkExtent2D extent{.width = destination->width, .height = destination->height};

        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([this, count = last_count, generation_count = last_generations, generation,
                          target = destination->index, image = *destination->image,
                          extent](vk::CommandBuffer cmdbuf) {
            chain->DispatchGeneration(cmdbuf, count, generation_count, generation, target, image,
                                      extent);
        });
    }
}

void FrameGen::ResetHistory() {
    if (!current_config) {
        return;
    }
    if (chain || fsr3_chain || open_fg_chain) {
        scheduler.Finish();
        chain.reset();
        fsr3_chain.reset();
        open_fg_chain.reset();
    }
    current_config.reset();
    pacer.Reset();
    plan = {};
    frame_count = 0;
    warm_streak = 0;
    generated = false;
}

void FrameGen::Rebuild(const Device& device, const Config& config) {
    const VkExtent2D extent{config.width, config.height};
    if (config.type == Settings::FrameGenType::Fsr3) {
        fsr3_chain.emplace(device, memory_allocator, extent, config.format, config.scale);
    } else if (config.type == Settings::FrameGenType::OpenFrameGen) {
        open_fg_chain.emplace(device, memory_allocator, extent, config.format, config.scale);
    } else {
        chain.emplace(device, memory_allocator, *shaders, extent, config.format, config.scale);
    }
}

void FrameGen::DumpDebugImages(u64 count) {
    const std::filesystem::path directory =
        Common::FS::GetEdenPath(Common::FS::EdenPath::LosslessDir) / "debug";
    if (!Common::FS::CreateDirs(directory)) {
        return;
    }

    const auto dump = [&](const std::string& name, LsfgImage& image) {
        const VkExtent2D extent = image.Extent();
        const VkFormat format = image.Format();
        const VkDeviceSize texel_size = BytesPerTexel(format);
        const VkDeviceSize size =
            static_cast<VkDeviceSize>(extent.width) * extent.height * texel_size;

        vk::Buffer buffer = CreateWrappedBuffer(memory_allocator, size, MemoryUsage::Download);

        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record(
            [handle = image.Handle(), dst = *buffer, extent](vk::CommandBuffer cmdbuf) {
                DownloadColorImage(
                    cmdbuf, handle, dst,
                    VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1});
            });
        scheduler.Finish();

        buffer.Invalidate();
        const std::span<u8> mapped = buffer.Mapped();

        if (format == LSFG_FLOW_FORMAT) {
            WriteGrayscalePgm(directory / (name + ".pgm"), extent, mapped);
        } else if (texel_size == COLOR_CHANNELS) {
            WriteColorPpm(directory / (name + ".ppm"), extent, mapped, IsBlueFirst(format));
        } else {
            WriteRaw(directory / (name + "_" + std::to_string(extent.width) + "x" +
                                  std::to_string(extent.height) + ".f16"),
                     mapped.subspan(0, std::min<size_t>(size, mapped.size())));
        }
    };

    dump("in0", chain->Input(0));
    dump("in1", chain->Input(1));

    for (size_t level = 0; level < LSFG_MIP_LEVELS; ++level) {
        dump("flow_mip" + std::to_string(level), chain->FlowLevel(level));
    }
    for (size_t index = 0; index < 2; ++index) {
        dump("alpha0_" + std::to_string(index), chain->AlphaOutput(0, count, index));
        dump("alpha6_" + std::to_string(index), chain->AlphaOutput(LSFG_MIP_LEVELS - 1, count,
                                                                  index));
    }
    for (size_t level = 0; level < LSFG_BETA_OUTPUTS; ++level) {
        dump("beta_" + std::to_string(level), chain->BetaOutput(level));
    }

    dump("gamma0", chain->GammaOutput(0));
    dump("gamma6", chain->GammaOutput(LSFG_MIP_LEVELS - 1));
    dump("delta2_out1", chain->DeltaOutput1(LSFG_DELTA_INSTANCES - 1));
    dump("delta2_out2", chain->DeltaOutput2(LSFG_DELTA_INSTANCES - 1));
}

} // namespace Vulkan
