// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Upscaling of the NVN replay. Two callers write a rectangle of one texture from a rectangle of
// another, and both run through `RunUpscale`:
//   * a game's upscaling network, the NDSR compute dispatch (`UpscaleNetwork`: DK's, OoT's), which
//     the replay replaces; it binds only a colour image (OoT: a packed copy of the scene, so the
//     scene colour is traced back through the compute chain), so the host makes the motion
//     vectors and the exposure DLSS needs, and uses the scene's depth when the game's passes
//     show which one it is;
//   * the game's own nvnCommandBufferEvaluateDlss (Op::EvaluateDlss, `EvaluateDlssOp`), where the
//     game supplies depth, motion vectors, jitter, exposure and the history reset.
// The user's upscaler setting decides what writes the output: DLSS when it was asked for and can
// evaluate; FSR1 or bilinear otherwise, a failed DLSS evaluation included, so a texture the game
// expects to be written never stays unwritten.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/renderer_vulkan/nvn/nvn_game_dlss.h"
#endif
#include "video_core/renderer_vulkan/present/fsr.h"
#include "video_core/surface.h"
#include "video_core/upscaler.h"
#include "video_core/vulkan_common/cooperative_matrix.h"

namespace Vulkan::Nvn {

namespace {

/// One line the first time each stage replaces a game's own upscale (`what`).
void LogUpscaleStage(std::string_view what, const char* stage) {
    static Common::BootCounter boot;
    static std::set<std::pair<std::string, std::string>> logged;
    if (boot.First()) {
        logged.clear(); // the first line of a new boot
    }
    if (logged.emplace(what, stage).second) {
        LOG_INFO(Render_Vulkan, "[nvn] {} replaced by {} stage", what, stage);
    }
}

/// One line the first time a game's upscaling network is replaced by a stage at an input size.
void LogNetworkReplacement(const char* stage, const VideoCore::Upscaler::Rect& input,
                           const VideoCore::Upscaler::Rect& output, std::string_view basis) {
    static Common::BootCounter boot;
    static std::set<std::string> logged;
    if (boot.First()) {
        logged.clear();
    }
    std::string line = fmt::format("NDSR upscale replaced by {} (input {}x{} -> {}x{} {})", stage,
                                   input.width, input.height, output.width, output.height, basis);
    if (logged.insert(line).second) {
        LOG_INFO(Render_Vulkan, "[nvn] {}", line);
    }
}

[[nodiscard]] u32 LevelSize(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

void NoteUpscale(Context::Impl& impl, const auto& job, const char* stage) {
    using VideoCore::GameFrameSize::Stage;
    impl.ndsr_upscaled_since_present = true;
    const std::string_view name{stage};
    impl.dlss_upscaled_since_present = name.starts_with("DLSS") || name.starts_with("DLAA");
    impl.upscaled_stage = name.starts_with("DLSS") ? Stage::Dlss
                          : name.starts_with("DLAA") ? Stage::Dlaa
                          : name.starts_with("FSR1") ? Stage::Fsr1
                                                     : Stage::Bilinear;
    impl.upscaled_input_width = job.input_rect.width;
    impl.upscaled_input_height = job.input_rect.height;
    impl.upscaled_output_width = job.output_rect.width;
    impl.upscaled_output_height = job.output_rect.height;
}

/// The format an image is read or written as: the view's, when the game's view reinterprets it.
[[nodiscard]] VkFormat ViewFormat(const Texture& texture, VkFormat requested) {
    return requested != VK_FORMAT_UNDEFINED ? requested : texture.host.format;
}

} // namespace

bool DrawContext::EvaluateDlss(const UpscaleJob& job, bool dlaa,
                               const VideoCore::Upscaler::Rect* input_rect_override) {
#ifdef EGLANTINE_STREAMLINE
    // DLSS needs the game's depth and motion vectors (a network replacement has neither).
    if (impl.game_dlss_failed || job.host_inputs || job.depth.texture == nullptr ||
        job.motion_vectors.texture == nullptr || !GameDlss::Available()) {
        return false;
    }
    const VideoCore::Upscaler::Rect input_rect =
        input_rect_override != nullptr ? *input_rect_override : job.input_rect;
    const VkExtent2D render{input_rect.width, input_rect.height};
    const VkExtent2D output{job.output_rect.width, job.output_rect.height};
    const auto image_of = [&](const UpscaleImage& image, u32 width, u32 height) {
        GameDlss::Image out{};
        if (image.texture == nullptr) {
            return out;
        }
        const Storage& storage = *image.texture->storage;
        const VkImageAspectFlags aspect =
            (storage.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0
                ? VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT}
                : VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT};
        const ViewKey key{
            .format = ViewFormat(*image.texture, image.format),
            .type = VK_IMAGE_VIEW_TYPE_2D,
            .aspect = aspect,
            .base_level = image.level,
            .levels = 1,
            .base_layer = image.layer,
            .layers = 1,
        };
        out.view = resources.GetView(*image.texture, key);
        if (!out.view) {
            return out;
        }
        out.image = *storage.image;
        out.format = key.format;
        out.width = LevelSize(image.texture->desc.width, image.level);
        out.height = LevelSize(image.texture->desc.height, image.level);
        out.x = image.x;
        out.y = image.y;
        out.rect_width = width;
        out.rect_height = height;
        return out;
    };
    GameDlss::Request request{};
    request.color = image_of(job.input, render.width, render.height);
    request.color.x = input_rect.x;
    request.color.y = input_rect.y;
    request.depth = image_of(job.depth, render.width, render.height);
    request.motion_vectors =
        image_of(job.motion_vectors, job.motion_width != 0 ? job.motion_width : render.width,
                 job.motion_height != 0 ? job.motion_height : render.height);
    request.exposure = image_of(job.exposure, 1, 1);
    request.bias = image_of(job.bias, render.width, render.height);
    if (request.color.image == VK_NULL_HANDLE || request.depth.image == VK_NULL_HANDLE ||
        request.motion_vectors.image == VK_NULL_HANDLE) {
        return false;
    }

    // DLSS writes its own RGBA16F image (the game's output may be of any format and only a
    // rectangle of it is written); it is copied into the output rectangle below.
    constexpr VkFormat TargetFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr VkImageUsageFlags TargetUsage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    constexpr VkImageSubresourceRange TargetRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    Context::Impl::GameDlssViewport& state = impl.game_dlss[job.viewport];
    const u32 evaluation = impl.game_dlss_evaluations++;
    bool reset = job.reset;
    const bool new_target = !state.target || state.output.width != output.width ||
                            state.output.height != output.height;
    if (new_target || state.render.width != render.width || state.render.height != render.height) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] game DLSS (Streamline) network {}: {}x{} -> {}x{}{}{}{}, motion vectors "
                 "{}x{}",
                 job.viewport, render.width, render.height, output.width, output.height,
                 dlaa ? " DLAA" : "", job.hdr_color ? " HDR" : " LDR",
                 job.depth_inverted ? " inverted depth" : "", request.motion_vectors.rect_width,
                 request.motion_vectors.rect_height);
        reset = true;
    }
    if (new_target) {
        auto target = std::make_shared<Context::Impl::GameDlssTarget>();
        target->image = impl.allocator->CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = TargetFormat,
            .extent = {output.width, output.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = TargetUsage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        target->view = impl.device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *target->image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = TargetFormat,
            .components{},
            .subresourceRange = TargetRange,
        });
        state.target = std::move(target);
    }
    state.render = render;
    state.output = output;
    state.last_evaluation = evaluation;
    // Networks the game finalized long ago (each Initialize is a new viewport).
    if (impl.game_dlss.size() > 8) {
        for (auto it = impl.game_dlss.begin(); it != impl.game_dlss.end();) {
            if (evaluation - it->second.last_evaluation > 600) {
                GameDlss::FreeViewport(it->first);
                it = impl.game_dlss.erase(it);
            } else {
                ++it;
            }
        }
    }
    const std::shared_ptr<Context::Impl::GameDlssTarget> target = impl.game_dlss[job.viewport].target;
    host->HostFlight().keepalive.push_back(target);

    const auto target_barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access,
                                    VkImageLayout old_layout) {
        return VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = src_access,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *target->image,
            .subresourceRange = TargetRange,
        };
    };
    for (const UpscaleImage* image :
         {&job.input, &job.depth, &job.motion_vectors, &job.exposure, &job.bias}) {
        if (image->texture != nullptr) {
            host->HostTransition(*image->texture->storage, VK_IMAGE_LAYOUT_GENERAL);
        }
    }
    host->HostFlushBarrier();
    const vk::CommandBuffer cmd = host->HostCmd();
    // The target's previous contents are never read: DLSS writes all of it.
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        target_barrier(0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_UNDEFINED));

    request.output.image = *target->image;
    request.output.view = *target->view;
    request.output.format = TargetFormat;
    request.output.usage = TargetUsage;
    request.output.width = output.width;
    request.output.height = output.height;
    request.output.rect_width = output.width;
    request.output.rect_height = output.height;
    request.viewport = job.viewport;
    request.frame_index = evaluation;
    request.jitter[0] = job.jitter[0];
    request.jitter[1] = job.jitter[1];
    request.motion_scale[0] = job.motion_scale[0];
    request.motion_scale[1] = job.motion_scale[1];
    request.pre_exposure = job.pre_exposure;
    request.dlaa = dlaa;
    request.reset = reset;
    request.depth_inverted = job.depth_inverted;
    request.hdr_color = job.hdr_color;
    // HDR colour without an exposure texture: DLSS measures the exposure itself.
    request.auto_exposure =
        job.auto_exposure || (job.hdr_color && request.exposure.image == VK_NULL_HANDLE);
    request.motion_vectors_jittered = job.motion_vectors_jittered;
    const bool evaluated = GameDlss::Evaluate(*cmd, request);
    // DLSS bound its own pipelines and descriptors on the command buffer.
    bound_pipeline = VK_NULL_HANDLE;
    if (!evaluated) {
        // One failure is the device's or the runtime's: the fallback stage writes from now on.
        impl.game_dlss_failed = true;
        return false;
    }

    cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        target_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                       VK_IMAGE_LAYOUT_GENERAL));
    Texture& output_texture = *job.output.texture;
    host->HostTransition(*output_texture.storage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    host->HostFlushBarrier();
    const VkImageBlit region{
        .srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0},
                       {static_cast<s32>(output.width), static_cast<s32>(output.height), 1}},
        .dstSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, job.output.level,
                                                   job.output.layer, 1},
        .dstOffsets = {{static_cast<s32>(job.output_rect.x), static_cast<s32>(job.output_rect.y),
                        0},
                       {static_cast<s32>(job.output_rect.x + output.width),
                        static_cast<s32>(job.output_rect.y + output.height), 1}},
    };
    host->HostCmd().BlitImage(*target->image, VK_IMAGE_LAYOUT_GENERAL, *output_texture.storage->image,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region, VK_FILTER_NEAREST);
    host->HostTransition(*output_texture.storage, VK_IMAGE_LAYOUT_GENERAL);
    host->HostNoteTouched(output_texture.storage);
    return true;
#else
    // No NVIDIA DLSS runtime in this build: the caller's FSR1 / bilinear stage writes the output.
    static_cast<void>(job);
    static_cast<void>(dlaa);
    static_cast<void>(input_rect_override);
    return false;
#endif
}

bool DrawContext::UpscaleWithoutDlss(const UpscaleJob& job, bool use_fsr) {
    Texture& input = *job.input.texture;
    Texture& output = *job.output.texture;
    const VideoCore::Upscaler::Rect& input_rect = job.input_rect;
    const VideoCore::Upscaler::Rect& output_rect = job.output_rect;
    const VkImageSubresourceLayers input_layers{VK_IMAGE_ASPECT_COLOR_BIT, job.input.level,
                                                job.input.layer, 1};
    const VkImageSubresourceLayers output_layers{VK_IMAGE_ASPECT_COLOR_BIT, job.output.level,
                                                 job.output.layer, 1};
    if (use_fsr) {
        const VkExtent2D extent{output_rect.width, output_rect.height};
        if (!impl.upscaler_fsr || !impl.upscaler_fsr_extent ||
            impl.upscaler_fsr_extent->width != extent.width ||
            impl.upscaler_fsr_extent->height != extent.height) {
            impl.upscaler_fsr =
                std::make_shared<::Vulkan::FSR>(impl.device, *impl.allocator, 1, extent);
            impl.upscaler_fsr_extent = extent;
        }
        host->HostFlight().keepalive.push_back(impl.upscaler_fsr);
        ViewKey key{
            .format = ViewFormat(input, job.input.format),
            .type = VK_IMAGE_VIEW_TYPE_2D,
            .aspect = input.storage->aspect,
            .base_level = job.input.level,
            .levels = 1,
            .base_layer = job.input.layer,
            .layers = 1,
        };
        const VkImageView input_view = resources.GetView(input, key);
        if (!input_view) {
            return false;
        }
        host->HostTransition(*input.storage, VK_IMAGE_LAYOUT_GENERAL);
        host->HostFlushBarrier();
        const u32 input_width = LevelSize(input.desc.width, job.input.level);
        const u32 input_height = LevelSize(input.desc.height, job.input.level);
        const Common::Rectangle<f32> crop{
            static_cast<f32>(input_rect.x) / input_width,
            static_cast<f32>(input_rect.y) / input_height,
            static_cast<f32>(input_rect.x + input_rect.width) / input_width,
            static_cast<f32>(input_rect.y + input_rect.height) / input_height,
        };
        impl.upscaler_fsr->Record(impl.device, host->HostCmd(), 0, *input.storage->image,
                                  input_view, {input_width, input_height}, crop);
        host->HostFlight().keepalive.push_back(impl.upscaler_fsr->TakeRecordResources());
        host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        host->HostFlushBarrier();
        const VkImageBlit region{
            .srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0},
                           {static_cast<s32>(output_rect.width),
                            static_cast<s32>(output_rect.height), 1}},
            .dstSubresource = output_layers,
            .dstOffsets = {{static_cast<s32>(output_rect.x), static_cast<s32>(output_rect.y), 0},
                           {static_cast<s32>(output_rect.x + output_rect.width),
                            static_cast<s32>(output_rect.y + output_rect.height), 1}},
        };
        host->HostCmd().BlitImage(impl.upscaler_fsr->ResultImage(0), VK_IMAGE_LAYOUT_GENERAL,
                                  *output.storage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  region, VK_FILTER_LINEAR);
    } else {
        host->HostTransition(*input.storage, VK_IMAGE_LAYOUT_GENERAL);
        host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        host->HostFlushBarrier();
        const VkImageBlit region{
            .srcSubresource = input_layers,
            .srcOffsets = {{static_cast<s32>(input_rect.x), static_cast<s32>(input_rect.y), 0},
                           {static_cast<s32>(input_rect.x + input_rect.width),
                            static_cast<s32>(input_rect.y + input_rect.height), 1}},
            .dstSubresource = output_layers,
            .dstOffsets = {{static_cast<s32>(output_rect.x), static_cast<s32>(output_rect.y), 0},
                           {static_cast<s32>(output_rect.x + output_rect.width),
                            static_cast<s32>(output_rect.y + output_rect.height), 1}},
        };
        host->HostCmd().BlitImage(*input.storage->image, VK_IMAGE_LAYOUT_GENERAL,
                                  *output.storage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  region, VK_FILTER_LINEAR);
    }
    host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_GENERAL);
    host->HostNoteTouched(job.output.texture->storage);
    return true;
}

bool DrawContext::RunUpscale(const UpscaleJob& job, const char*& stage) {
    const bool same_extent = job.input_rect.width == job.output_rect.width &&
                             job.input_rect.height == job.output_rect.height;
    const Settings::Upscaler setting = VideoCore::Upscaler::Effective();
    auto backend = VideoCore::Upscaler::SelectBackend(setting, false);
    if ((job.game_dlss || job.prefer_dlss) &&
        (setting == Settings::Upscaler::Auto || setting == Settings::Upscaler::Game)) {
        // The game's own DLSS is the game's upscaler: the real DLSS (DLAA at 1:1) stands in for
        // it; DK's NDSR has no such counterpart (its network is not NGX's).
        backend = same_extent ? VideoCore::Upscaler::Backend::Dlaa
                              : VideoCore::Upscaler::Backend::Dlss;
    }
    if (job.host_inputs) {
        // A network replacement fed its scaled twin: at ratio 1 DLSS is DLAA (a two-pixel twin
        // overhang, OoT 854x480 at 3X, is cropped), whatever DLSS/Auto setting asked for it.
        backend = VideoCore::Upscaler::PromoteNearNativeToDlaa(
            backend, job.input_rect.width, job.input_rect.height, job.output_rect.width,
            job.output_rect.height);
    }
    const bool dlss_requested = backend == VideoCore::Upscaler::Backend::Dlss ||
                                backend == VideoCore::Upscaler::Backend::Dlaa;
    bool dlaa_input_too_large = false;
    // The stage that actually writes the output: DLSS when it can evaluate; otherwise FSR1 for
    // FSR and for a DLSS request that cannot run (unsupported device, missing runtime, a failed
    // evaluation); bilinear only when the user asked for it. The output is always written.
    if (dlss_requested) {
        const bool dlaa = backend == VideoCore::Upscaler::Backend::Dlaa;
        if (dlaa) {
            const auto choice = VideoCore::Upscaler::SelectDlaaRequestStage(
                job.input_rect.width, job.input_rect.height, job.output_rect.width,
                job.output_rect.height);
            using VideoCore::Upscaler::DlaaRequestStage;
            if (choice == DlaaRequestStage::Dlaa) {
                if (EvaluateDlss(job, true)) {
                    stage = "DLAA";
                    return true;
                }
            } else if (choice == DlaaRequestStage::DlaaCrop) {
                // Keep the top/left origin: cropping only the right/bottom excess preserves the
                // scene's pixel centres instead of shifting the image by half a texel.
                VideoCore::Upscaler::Rect cropped = job.input_rect;
                cropped.width = job.output_rect.width;
                cropped.height = job.output_rect.height;
                if (EvaluateDlss(job, true, &cropped)) {
                    stage = "DLAA (output-size crop)";
                    return true;
                }
            } else if (choice == DlaaRequestStage::Dlss) {
                if (EvaluateDlss(job, false)) {
                    stage = "DLSS (DLAA requested)";
                    return true;
                }
            } else {
                // Deliberately retain the old path for a substantially larger input. Downsampling
                // before DLAA needs its own history/motion-vector design; FSR1 is not hidden as
                // DLAA and the transition log names the actual sizes.
                dlaa_input_too_large = true;
            }
        } else if (EvaluateDlss(job, false)) {
            stage = "DLSS";
            return true;
        }
    }
    const bool use_fsr = backend == VideoCore::Upscaler::Backend::Fsr || dlss_requested;
    stage = dlaa_input_too_large ? "FSR1 (DLAA requested, input larger than output)"
            : use_fsr         ? (dlss_requested ? "FSR1 (DLSS unavailable)" : "FSR1")
                              : "bilinear";
    if (UpscaleWithoutDlss(job, use_fsr)) {
        return true;
    }
    if (!use_fsr) {
        return false;
    }
    // FSR records nothing when it fails (no view of the input): a plain bilinear blit still
    // writes the output.
    stage = "bilinear (FSR unavailable)";
    return UpscaleWithoutDlss(job, false);
}

bool DrawContext::UpscaleTwinOf(Texture& texture, u32 level, std::shared_ptr<Texture>& out,
                                const char** why) {
    const auto refuse = [&](const char* reason) {
        if (why != nullptr) {
            *why = reason;
        }
        return false;
    };
    Storage& storage = *texture.storage;
    if (!storage.scaled) {
        return refuse("the image was never drawn at the scaled size");
    }
    if (storage.scale_owner != nullptr) {
        return refuse("the image is itself a twin");
    }
    if (level != 0) {
        return refuse("mip level above 0");
    }
    if (!Rescale::TwinIsCurrent(storage.scale_state)) {
        return refuse("the twin is stale: the guest-sized image was written after it");
    }
    if (host->HostNeedsNativeUpdate(storage)) {
        return refuse("the twin lacks guest bytes written to the image");
    }
    out = impl.ScaledTextureOf(texture);
    if (out == nullptr || !impl.EnsureImage(*out->storage)) {
        return refuse("the twin image could not be created");
    }
    return true;
}

VideoCore::Upscaler::Backend DrawContext::SelectedUpscaleBackend(
    Settings::Upscaler requested) const {
    const bool setting_allows_tensor =
        requested == Settings::Upscaler::Auto || requested == Settings::Upscaler::Game;
    // The game's own network needs cooperative matrices, which Eglantine's lowering does not
    // use: the network is always replaced.
    static_cast<void>(setting_allows_tensor);
    constexpr bool tensor_supported = false;
    return VideoCore::Upscaler::SelectBackend(requested, tensor_supported);
}

bool DrawContext::NetworkReplacementWanted() const {
    const Settings::Upscaler requested = VideoCore::Upscaler::Effective();
    return VideoCore::Upscaler::ReplacesGameNetwork(SelectedUpscaleBackend(requested), requested,
                                                    impl.RescaleFactor().Active());
}

void DrawContext::NoteComputeChain() {
    const StageBindings& bound = stages[STAGE_COMPUTE];
    VideoCore::Upscaler::ComputeChainEntry entry;
    size_t count = 0;
    for (const u64 handle : bound.combined) {
        if (handle == 0) {
            continue;
        }
        if (const auto& texture = TextureOfId(VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle)))) {
            entry.sampled[count++] = VideoCore::Upscaler::NetworkTextureSerial(
                texture->serial, texture->desc.view_of);
            if (count == entry.sampled.size()) {
                break;
            }
        }
    }
    count = 0;
    for (const u64 handle : bound.image) {
        if (handle == 0) {
            continue;
        }
        if (const auto& texture = TextureOfId(VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle)))) {
            entry.written[count++] = VideoCore::Upscaler::NetworkTextureSerial(
                texture->serial, texture->desc.view_of);
            if (count == entry.written.size()) {
                break;
            }
        }
    }
    impl.compute_chain[impl.compute_chain_next] = entry;
    impl.compute_chain_next = (impl.compute_chain_next + 1) % impl.compute_chain.size();
    impl.compute_chain_count =
        (std::min)(impl.compute_chain_count + 1, static_cast<u32>(impl.compute_chain.size()));
}

bool DrawContext::ComputeProgramUsesTensorMath(u64 hash, const std::vector<u8>& code,
                                               u32 instruction_bytes) {
    const auto [it, inserted] = impl.program_tensor_math.try_emplace(hash, false);
    if (inserted) {
        const size_t bytes = (std::min<size_t>)(code.size(), instruction_bytes);
        it->second = VideoCore::Upscaler::CodeUsesTensorMath({code.data(), bytes});
    }
    return it->second;
}

bool DrawContext::UpscaleNetwork(const VideoCore::Upscaler::Rect& input_rect,
                                 const VideoCore::Upscaler::Rect& output_rect, u32 output_id) {
    const StageBindings& bound = stages[STAGE_COMPUTE];
    if (bound.combined.empty()) {
        return false;
    }
    const u32 input_id = VideoCommon::Nvn::TextureIdOf(static_cast<u32>(bound.combined[0]));
    const std::shared_ptr<Texture> network_input = TextureOfId(input_id);
    const std::shared_ptr<Texture> output = TextureOfId(output_id);
    if (!network_input || !output || !network_input->storage || !output->storage ||
        !network_input->storage->image || !output->storage->image) {
        static Common::BootCounter resource_logged;
        if (resource_logged.First()) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] NDSR replacement skipped (resources): input={} output={} "
                     "input storage={} output storage={} input image={} output image={}",
                     network_input != nullptr, output != nullptr,
                     network_input && network_input->storage, output && output->storage,
                     network_input && network_input->storage && network_input->storage->image,
                     output && output->storage && output->storage->image);
        }
        return false;
    }
    // The scene colour the network consumes: its input, or what an earlier compute dispatch
    // packed it from.
    const auto is_colour = [&](u64 serial) {
        const auto it = impl.textures.find(serial);
        if (it == impl.textures.end() || !it->second || !it->second->storage ||
            !it->second->storage->image) {
            return false;
        }
        const Texture& texture = *it->second;
        const Storage& storage = *texture.storage;
        return (storage.aspect & VK_IMAGE_ASPECT_COLOR_BIT) != 0 && !texture.host.depth &&
               !VideoCore::Surface::IsPixelFormatInteger(texture.host.pixel_format) &&
               storage.type == VK_IMAGE_TYPE_2D && storage.samples == VK_SAMPLE_COUNT_1_BIT &&
               texture.desc.width >= input_rect.x + input_rect.width &&
               texture.desc.height >= input_rect.y + input_rect.height;
    };
    std::array<VideoCore::Upscaler::ComputeChainEntry, 16> chain;
    const u32 chain_size = static_cast<u32>(impl.compute_chain.size());
    for (u32 i = 0; i < impl.compute_chain_count; ++i) {
        chain[i] = impl.compute_chain[(impl.compute_chain_next + chain_size - 1 - i) % chain_size];
    }
    const u64 colour_serial = VideoCore::Upscaler::FindNetworkColour(
        {chain.data(), impl.compute_chain_count}, network_input->serial, is_colour);
    const auto colour_it = colour_serial != 0 ? impl.textures.find(colour_serial) : impl.textures.end();
    std::shared_ptr<Texture> colour = colour_it != impl.textures.end() ? colour_it->second : nullptr;
    if (!colour || !is_colour(colour_serial)) {
        static Common::BootCounter no_colour_logged;
        if (no_colour_logged.First()) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] NDSR upscale not replaced: the network's input {}x{} (format {}) is "
                        "not a colour and no scene colour was found behind it; the game's own "
                        "network runs",
                        network_input->desc.width, network_input->desc.height,
                        static_cast<u32>(network_input->host.format));
        }
        return false;
    }
    // The scene's depth: the depth of the last pass that drew into that colour, at its size.
    std::shared_ptr<Texture> depth;
    if (const auto it = impl.pass_depth_of_color.find(colour->serial);
        it != impl.pass_depth_of_color.end()) {
        if (const auto texture = impl.textures.find(it->second);
            texture != impl.textures.end() && texture->second && texture->second->storage &&
            texture->second->storage->image && texture->second->host.depth &&
            (texture->second->storage->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
            texture->second->storage->type == VK_IMAGE_TYPE_2D &&
            texture->second->storage->samples == VK_SAMPLE_COUNT_1_BIT &&
            texture->second->desc.width == colour->desc.width &&
            texture->second->desc.height == colour->desc.height) {
            depth = texture->second;
        }
    }
    // As EvaluateDlss: the output's other texels survive unless the rectangle covers the level.
    const bool covers_output = output_rect.x == 0 && output_rect.y == 0 &&
                               output->storage->levels == 1 && output->storage->layers == 1 &&
                               output_rect.width == output->desc.width &&
                               output_rect.height == output->desc.height;
    if (!host->HostEnsureContents(*colour->storage, false) ||
        !host->HostEnsureContents(*output->storage, covers_output) ||
        (depth && !host->HostEnsureContents(*depth->storage, false))) {
        return false;
    }
    const u64 seq = host->HostFlight().seq;
    impl.NoteUse(*colour, seq);
    impl.NoteUse(*output, seq);
    if (depth) {
        impl.NoteUse(*depth, seq);
    }

    UpscaleJob job;
    job.input.texture = colour.get();
    job.output.texture = output.get();
    job.depth.texture = depth.get();
    job.input_rect = input_rect;
    job.output_rect = output_rect;
    job.host_inputs = true;
    // The Resolution setting: the scene was drawn at the scaled size, and the upscaler is fed the
    // twin (and the depth twin) with the rectangle in its pixels; anything the twins cannot serve
    // leaves the guest-sized images, and the log line says why. A twin the output's size or
    // smaller (DLSS), or larger by at most DlaaCropSlack (DLAA crops it: OoT's 854x480 at 3X is
    // 2562x1440 for a 2560x1440 output), is evaluated as it is.
    const Rescale::Factor factor = impl.RescaleFactor();
    std::array<std::shared_ptr<Texture>, 2> twins;
    std::string basis = "native"; // 1X: nothing else to say
    if (factor.Active()) {
        const Rescale::Rect scaled = Rescale::ScaleRect(
            factor, {static_cast<s32>(input_rect.x), static_cast<s32>(input_rect.y),
                     input_rect.width, input_rect.height});
        const char* twin_refusal = nullptr;
        if (scaled.width == 0 || scaled.height == 0) {
            basis = "native: empty scaled input rectangle";
        } else if (!VideoCore::Upscaler::UpscaleInputFitsOutput(
                       scaled.width, scaled.height, output_rect.width, output_rect.height)) {
            basis = fmt::format("native: the scaled input {}x{} does not fit the {}x{} output",
                                scaled.width, scaled.height, output_rect.width,
                                output_rect.height);
        } else if (!UpscaleTwinOf(*colour, 0, twins[0], &twin_refusal) || twins[0] == nullptr) {
            basis = fmt::format("native: no usable colour twin ({})",
                                twin_refusal != nullptr ? twin_refusal : "none");
        } else {
            job.input.texture = twins[0].get();
            job.input_rect = {static_cast<u32>(scaled.x), static_cast<u32>(scaled.y), scaled.width,
                              scaled.height};
            // Depth must match the colour's size: its twin, or none (the far plane).
            const char* depth_refusal = nullptr;
            if (depth && UpscaleTwinOf(*depth, 0, twins[1], &depth_refusal) && twins[1] != nullptr) {
                job.depth.texture = twins[1].get();
                basis = "twin";
            } else {
                job.depth.texture = nullptr;
                basis = fmt::format("twin, far-plane depth ({})",
                                    depth ? (depth_refusal != nullptr ? depth_refusal : "none")
                                          : "no scene depth found");
            }
        }
    }
    job.input.x = job.input_rect.x;
    job.input.y = job.input_rect.y;
    job.depth.x = job.input_rect.x;
    job.depth.y = job.input_rect.y;
    job.output.x = job.output_rect.x;
    job.output.y = job.output_rect.y;
    job.prefer_dlss = factor.Active() &&
                      VideoCore::Upscaler::Effective() == Settings::Upscaler::Auto;
    const char* stage = nullptr;
    if (!RunUpscale(job, stage)) {
        return false;
    }
    NoteUpscale(impl, job, stage);
    LogNetworkReplacement(stage, job.input_rect, job.output_rect, basis);
    return true;
}

void DrawContext::EvaluateDlssOp(const VC::EvaluateDlssPayload& p) {
    // The textures stay alive for the whole op, whatever the replay retires meanwhile.
    std::array<std::shared_ptr<Texture>, 6> held;
    std::array<std::shared_ptr<Texture>, 4> twins; ///< the scaled twins evaluated instead
    const auto resolve = [&](size_t slot, const VC::RenderTargetRef& ref, UpscaleImage& image) {
        image = {};
        if (ref.texture.serial == 0) {
            return false;
        }
        const auto it = impl.textures.find(ref.texture.serial);
        if (it == impl.textures.end() || !it->second || !it->second->storage) {
            return false;
        }
        impl.NoteUse(*it->second, host->HostFlight().seq);
        held[slot] = it->second;
        image.texture = it->second.get();
        image.level = ref.texture.level;
        image.layer = ref.texture.layer;
        if (ref.view.set_format) {
            if (const auto format = ResolveHostFormat(impl.device, ref.view.format)) {
                image.format = format->format;
            }
        }
        return true;
    };
    UpscaleJob job;
    if (!resolve(0, p.input, job.input) || !resolve(1, p.output, job.output)) {
        host->HostFailed("EvaluateDlss", "the input or the output texture is not a live texture");
        return;
    }
    resolve(2, p.motion_vectors, job.motion_vectors);
    resolve(3, p.depth, job.depth);
    resolve(4, p.bias, job.bias);
    resolve(5, p.exposure, job.exposure);

    // An image is read or written in one single-sample 2D level and layer of its texture.
    const auto usable = [](const UpscaleImage& image) {
        const Storage& storage = *image.texture->storage;
        return image.level < storage.levels && image.layer < storage.layers &&
               storage.type == VK_IMAGE_TYPE_2D && storage.samples == VK_SAMPLE_COUNT_1_BIT;
    };
    const auto level_width = [](const UpscaleImage& image) {
        return LevelSize(image.texture->desc.width, image.level);
    };
    const auto level_height = [](const UpscaleImage& image) {
        return LevelSize(image.texture->desc.height, image.level);
    };
    if (!usable(job.input) || !usable(job.output)) {
        host->HostFailed("EvaluateDlss",
                         fmt::format("the input (level {} layer {}) or the output (level {} layer "
                                     "{}) is not a single-sample 2D level of its texture",
                                     job.input.level, job.input.layer, job.output.level,
                                     job.output.layer));
        return;
    }
    // The SDK forwards the rectangles unchecked (OoT sdk+0x65a4e0) and a GPU drops writes outside
    // an image, so the part of the output inside its level is written, from the matching part of
    // the input.
    const VideoCore::Upscaler::GameRect game_input{p.input_base.x, p.input_base.y, p.input_width,
                                                   p.input_height};
    const VideoCore::Upscaler::GameRect game_output{p.output_base.x, p.output_base.y,
                                                    p.output_width, p.output_height};
    const auto rects = VideoCore::Upscaler::ClipUpscaleRects(
        game_input, game_output, level_width(job.input), level_height(job.input),
        level_width(job.output), level_height(job.output));
    if (!rects) {
        host->HostFailed(
            "EvaluateDlss",
            fmt::format("input rectangle {},{} {}x{} or output rectangle {},{} {}x{} has no part "
                        "inside its texture level",
                        p.input_base.x, p.input_base.y, p.input_width, p.input_height,
                        p.output_base.x, p.output_base.y, p.output_width, p.output_height));
        return;
    }
    if (rects->clipped) {
        static Common::BootCounter clip_logged;
        if (clip_logged.First()) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] EvaluateDlss: input rectangle {},{} {}x{} -> output rectangle "
                        "{},{} {}x{} lies partly outside its texture level; upscaling {},{} {}x{} "
                        "-> {},{} {}x{}",
                        p.input_base.x, p.input_base.y, p.input_width, p.input_height,
                        p.output_base.x, p.output_base.y, p.output_width, p.output_height,
                        rects->input.x, rects->input.y, rects->input.width, rects->input.height,
                        rects->output.x, rects->output.y, rects->output.width,
                        rects->output.height);
        }
    }
    job.input_rect = rects->input;
    job.output_rect = rects->output;
    job.input.x = job.input_rect.x;
    job.input.y = job.input_rect.y;
    job.output.x = job.output_rect.x;
    job.output.y = job.output_rect.y;
    // Resolution scaling: the game's colour, depth and (low-resolution) motion vectors were drawn
    // at the scaled size, and DLSS takes any input size up to the output's: it is evaluated on the
    // twins, with the rectangles, the jitter and the motion scale in their pixels, so the output
    // is rebuilt from more samples. Anything else (an input that is not a current twin, an input
    // larger than the output, a depth or motion vector image drawn at the guest size) leaves every
    // image at the guest size: EnsureContents makes the guest-sized images current.
    const bool motion_low_res = (p.flags & VC::DLSS_FLAG_MV_LOW_RES) != 0;
    const Rescale::Factor factor = impl.RescaleFactor();
    const VideoCore::Upscaler::Rect guest_input_rect = job.input_rect;
    bool dlss_scaled = false;
    if (factor.Active()) {
        const auto twin_of = [&](const UpscaleImage& image, std::shared_ptr<Texture>& out) {
            return image.texture == nullptr || UpscaleTwinOf(*image.texture, image.level, out);
        };
        const Rescale::Rect scaled_input = Rescale::ScaleRect(
            factor, {static_cast<s32>(guest_input_rect.x), static_cast<s32>(guest_input_rect.y),
                     guest_input_rect.width, guest_input_rect.height});
        dlss_scaled = scaled_input.width != 0 && scaled_input.height != 0 &&
                      scaled_input.width <= job.output_rect.width &&
                      scaled_input.height <= job.output_rect.height &&
                      job.input.texture != nullptr && twin_of(job.input, twins[0]) &&
                      twins[0] != nullptr && twin_of(job.depth, twins[1]) &&
                      (!motion_low_res || twin_of(job.motion_vectors, twins[2])) &&
                      twin_of(job.bias, twins[3]);
        if (dlss_scaled) {
            job.input.texture = twins[0].get();
            if (twins[1]) {
                job.depth.texture = twins[1].get();
            }
            if (twins[2]) {
                job.motion_vectors.texture = twins[2].get();
            }
            if (twins[3]) {
                job.bias.texture = twins[3].get();
            }
            job.input_rect = {static_cast<u32>(scaled_input.x), static_cast<u32>(scaled_input.y),
                              scaled_input.width, scaled_input.height};
            job.input.x = job.input_rect.x;
            job.input.y = job.input_rect.y;
        }
    }
    const auto scaled_at = [&](s64 value) -> s64 {
        return dlss_scaled ? factor.ScaleStart(static_cast<s32>(value)) : value;
    };

    if ((job.input.texture->storage->aspect & VK_IMAGE_ASPECT_COLOR_BIT) == 0 ||
        (job.output.texture->storage->aspect & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
        host->HostUnsupported("EvaluateDlss", "the input or the output texture is not a colour image");
        return;
    }
    if (job.input.texture->storage == job.output.texture->storage &&
        job.input.level == job.output.level && job.input.layer == job.output.layer) {
        host->HostFailed("EvaluateDlss", "the input and the output are the same image");
        return;
    }
    // NGX flags as the SDK forwards them (Initialize, OoT sdk+0x659e40).
    // The inputs that only help DLSS: one that cannot be used is left out, and the evaluation
    // then falls back to a plain upscale (DLSS needs the depth and the motion vectors). Their
    // rectangles follow the input rectangle (the motion vectors the output one when they are at
    // full resolution), moved as clipping moved it.
    const auto optional = [&](UpscaleImage& image, s64 x, s64 y, u32 width, u32 height,
                              const char* name) {
        if (image.texture == nullptr) {
            return;
        }
        const bool placed = usable(image) && x >= 0 && y >= 0 && x <= UINT32_MAX &&
                            y <= UINT32_MAX &&
                            VideoCore::Upscaler::IsValidRect(
                                {static_cast<u32>(x), static_cast<u32>(y), width, height},
                                level_width(image), level_height(image));
        if (!placed || !host->HostEnsureContents(*image.texture->storage, false)) {
            host->HostUnsupported("EvaluateDlss",
                                  fmt::format("the {} image cannot be used at {},{} {}x{}; DLSS "
                                              "is evaluated without it",
                                              name, x, y, width, height));
            image = {};
            return;
        }
        image.x = static_cast<u32>(x);
        image.y = static_cast<u32>(y);
    };
    const s64 input_dx = s64{job.input_rect.x} - p.input_base.x;
    const s64 input_dy = s64{job.input_rect.y} - p.input_base.y;
    const s64 output_dx = s64{job.output_rect.x} - p.output_base.x;
    const s64 output_dy = s64{job.output_rect.y} - p.output_base.y;
    // NGX's motion vectors are at the output resolution unless they are low-res ones.
    const VideoCore::Upscaler::Rect& motion_rect = motion_low_res ? job.input_rect : job.output_rect;
    job.motion_width = motion_rect.width;
    job.motion_height = motion_rect.height;
    // (A scaled evaluation reads the low-resolution images at their scaled positions.)
    const bool motion_scaled = dlss_scaled && motion_low_res;
    optional(job.motion_vectors,
             motion_scaled ? scaled_at(p.motion_vector_base.x + input_dx)
                           : p.motion_vector_base.x + (motion_low_res ? input_dx : output_dx),
             motion_scaled ? scaled_at(p.motion_vector_base.y + input_dy)
                           : p.motion_vector_base.y + (motion_low_res ? input_dy : output_dy),
             job.motion_width, job.motion_height, "motion vector");
    optional(job.depth, scaled_at(p.depth_base.x + input_dx), scaled_at(p.depth_base.y + input_dy),
             job.input_rect.width, job.input_rect.height, "depth");
    optional(job.bias, scaled_at(p.bias_base.x + input_dx), scaled_at(p.bias_base.y + input_dy),
             job.input_rect.width, job.input_rect.height, "bias mask");
    optional(job.exposure, 0, 0, 1, 1, "exposure");
    // The output's other texels survive unless the rectangle covers the whole level.
    const Storage& output_storage = *job.output.texture->storage;
    const bool covers_output =
        job.output_rect.x == 0 && job.output_rect.y == 0 && output_storage.levels == 1 &&
        output_storage.layers == 1 && job.output_rect.width == level_width(job.output) &&
        job.output_rect.height == level_height(job.output);
    if (!host->HostEnsureContents(*job.input.texture->storage, false) ||
        !host->HostEnsureContents(*job.output.texture->storage, covers_output)) {
        host->HostFailed("EvaluateDlss", "the input or the output image could not be created");
        return;
    }

    // The jitter is in pixels of the input and the game's vectors in pixels of its own render size:
    // in a scaled evaluation both are in scaled pixels.
    const f32 input_pixels = dlss_scaled ? factor.UpFactor() : 1.0f;
    const f32 motion_pixels = motion_scaled ? factor.UpFactor() : 1.0f;
    job.jitter[0] = p.jitter[0] * input_pixels;
    job.jitter[1] = p.jitter[1] * input_pixels;
    // The game's scale takes its vectors to pixels of the motion vector rectangle (NGX's MV
    // scale); Streamline's takes them to normalized units of that rectangle.
    job.motion_scale[0] = p.motion_scale[0] * motion_pixels / static_cast<f32>(job.motion_width);
    job.motion_scale[1] = p.motion_scale[1] * motion_pixels / static_cast<f32>(job.motion_height);
    job.pre_exposure = p.pre_exposure;
    job.reset = p.history_reset != 0;
    job.depth_inverted = (p.flags & VC::DLSS_FLAG_DEPTH_INVERTED) != 0;
    // NVN's bit 0 is NGX IsHDR inverted: clear means HDR colour.
    job.hdr_color = (p.flags & VC::DLSS_FLAG_LDR) == 0;
    job.auto_exposure = (p.flags & VC::DLSS_FLAG_AUTO_EXPOSURE) != 0;
    job.motion_vectors_jittered = (p.flags & VC::DLSS_FLAG_MV_JITTERED) != 0;
    job.game_dlss = true;
    job.viewport = p.network;
    job.game_method = p.method;
    job.game_preset = p.preset;

    const char* stage = nullptr;
    if (!RunUpscale(job, stage)) {
        host->HostFailed("EvaluateDlss", "no upscale could be recorded for the output");
        return;
    }
    host->HostMarkWritten();
    NoteUpscale(impl, job, stage);
    LogUpscaleStage("game DLSS evaluation", stage);
}

} // namespace Vulkan::Nvn
