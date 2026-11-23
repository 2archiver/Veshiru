// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Resolution scaling: the twin storages and the blits between the two images of a storage. The
// rules are in nvn_rescale.h; the recording around them (when a twin is made current, what a write
// does to the pair) is in nvn_exec.cpp (Recorder) and nvn_draw*.cpp (DrawContext).

#include <algorithm>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/surface.h"

namespace Vulkan::Nvn {

std::shared_ptr<Storage> MakeScaledTwin(Storage& native, const Rescale::Factor& factor) {
    auto twin = std::make_shared<Storage>();
    twin->host = native.host;
    // The guest description stays (formats, block info); the twin has no guest bytes of its own.
    twin->desc = native.desc;
    twin->layout = native.layout;
    twin->type = native.type;
    twin->extent = {factor.Scale(native.extent.width), factor.Scale(native.extent.height), 1};
    twin->samples = VK_SAMPLE_COUNT_1_BIT;
    twin->aspect = native.aspect;
    twin->levels = 1;
    twin->layers = native.layers;
    twin->create_flags = native.create_flags;
    twin->usage = native.usage;
    twin->vram_bytes = native.vram_bytes / (std::max)(native.extent.width, 1u) /
                       (std::max)(native.extent.height, 1u) * twin->extent.width *
                       twin->extent.height;
    // Nothing to upload, nothing to check against memory, never part of an overlap.
    twin->contents_valid = true;
    twin->origin = StorageOrigin::Gpu;
    twin->scale_owner = &native;
    return twin;
}

Rescale::Shape Context::Impl::RescaleShapeOf(const Storage& s, bool alias) const {
    Rescale::Shape shape;
    const auto target = static_cast<TexTarget>(s.desc.target);
    shape.two_d = target == TexTarget::Dim2D || target == TexTarget::Dim2DArray;
    shape.linear = IsLinearStorage(s.desc);
    shape.alias = alias;
    shape.sparse = s.sparse != nullptr;
    shape.transcoded = s.host.needs_transcode;
    shape.color = !(s.host.depth || s.host.stencil);
    shape.blit_ok = device.IsFormatSupported(
        s.host.format, VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT,
        FormatType::Optimal);
    shape.levels = s.levels;
    shape.samples = static_cast<u32>(s.samples);
    shape.width = s.extent.width;
    shape.height = s.extent.height;
    shape.max_dimension = (std::min)(device.GetMaxFramebufferWidth(), device.GetMaxFramebufferHeight());
    return shape;
}

Storage* Context::Impl::ScaledTwinOf(Storage& s) {
    if (s.scaled) {
        return s.scaled.get();
    }
    const Rescale::Factor factor = RescaleFactor();
    if (s.scale_owner != nullptr || !factor.Active()) {
        return nullptr;
    }
    s.scaled = MakeScaledTwin(s, factor);
    s.scale_state = Rescale::State::Native;
    return s.scaled.get();
}

std::shared_ptr<Texture> Context::Impl::ScaledTextureOf(Texture& texture) {
    if (texture.scaled) {
        return texture.scaled;
    }
    if (!texture.storage || ScaledTwinOf(*texture.storage) == nullptr) {
        return nullptr;
    }
    auto twin = std::make_shared<Texture>();
    twin->serial = next_alias_serial--;
    // The twin is a host-only object (the game never reaches it: its queries and copies name the
    // guest's texture), so its description is its own: the scaled size, what host code that reads a
    // texture's size (the upscale stages) needs.
    const Rescale::Factor factor = RescaleFactor();
    twin->desc = texture.desc;
    twin->desc.width = factor.Scale(texture.desc.width);
    twin->desc.height = factor.Scale(texture.desc.height);
    twin->host = texture.host;
    twin->storage = texture.storage->scaled;
    texture.scaled = twin;
    return twin;
}

void Context::Impl::RecordScaleBlit(vk::CommandBuffer cmd, Storage& native, bool to_scaled) const {
    Storage& twin = *native.scaled;
    Storage& source = to_scaled ? native : twin;
    Storage& destination = to_scaled ? twin : native;
    // Colour is filtered (as the emulated GPU's BlitScale); depth, stencil, integer formats and
    // formats without linear filtering may only be blitted nearest.
    const bool filtered = native.aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
                          !VideoCore::Surface::IsPixelFormatInteger(native.host.pixel_format) &&
                          device.IsFormatSupported(native.host.format,
                                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
                                                   FormatType::Optimal);
    const VkImageSubresourceLayers layers{
        .aspectMask = native.aspect,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = native.layers,
    };
    const VkImageBlit region{
        .srcSubresource = layers,
        .srcOffsets = {{0, 0, 0},
                       {static_cast<s32>(source.extent.width), static_cast<s32>(source.extent.height), 1}},
        .dstSubresource = layers,
        .dstOffsets = {{0, 0, 0},
                       {static_cast<s32>(destination.extent.width),
                        static_cast<s32>(destination.extent.height), 1}},
    };
    cmd.BlitImage(*source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *destination.image,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region,
                  filtered ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

bool Context::Impl::DownscaleNow(Storage& s) {
    if (!s.scaled || s.scale_state != Rescale::State::Scaled) {
        return true;
    }
    Storage& twin = *s.scaled;
    if (!EnsureImage(s) || !EnsureImage(twin)) {
        return false;
    }
    try {
        ImmediateSubmit([&](vk::CommandBuffer cmd) {
            const auto range = [](const Storage& storage) {
                return VkImageSubresourceRange{
                    .aspectMask = storage.aspect,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                };
            };
            const auto barrier = [&](Storage& storage, VkImageLayout from, VkImageLayout to) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .oldLayout = from,
                    .newLayout = to,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *storage.image,
                    .subresourceRange = range(storage),
                };
            };
            const std::array before{
                barrier(twin, twin.current_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                barrier(s, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, {}, {}, before);
            RecordScaleBlit(cmd, s, false);
            const std::array after{
                barrier(twin, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
                barrier(s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, {}, {}, after);
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] resolution scale: downscale for a readback failed: {}",
                  exception.what());
        return false;
    }
    twin.current_layout = VK_IMAGE_LAYOUT_GENERAL;
    s.current_layout = VK_IMAGE_LAYOUT_GENERAL;
    s.scale_state = Rescale::AfterBlit();
    return true;
}

} // namespace Vulkan::Nvn
