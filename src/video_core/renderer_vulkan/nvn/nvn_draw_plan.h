// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <vector>

#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"

namespace Vulkan::Nvn {

/// What a draw (or dispatch) needs done outside a render pass before it can record: images in the
/// right layout with their guest contents loaded, mirror pool ranges uploaded. Collected while the
/// descriptors are resolved (phase A: no image views yet; phase B, `FinishImages`, creates them once
/// the images exist and are in layout).
struct DrawPlan {
    struct ImageUse {
        std::shared_ptr<Storage> owner;
        VkImageLayout layout;
        /// Set by PrepareOutsidePass: loading the contents records commands.
        bool needs_contents{false};
        /// Set by PrepareOutsidePass: the image could not be created; the use is dropped.
        bool skip{false};
        /// Only sampled (no storage image or attachment use in the plan): an image already in the
        /// layout records nothing and is not noted as written by the GPU.
        bool sample_only{false};
        bool write{false};
    };
    /// A buffer range the draw (or dispatch) uses. Descriptor bindings carry the index of their
    /// VkDescriptorBufferInfo (`DrawContext::scratch_buffers`), which a storage shadow patches.
    struct RangeUse {
        static constexpr u32 NO_INFO = ~u32{0};
        Pool* pool;
        u64 offset;
        u64 size;
        bool write;
        /// A written storage range of an imported pool small enough to shadow (storage shadows).
        bool shadowable{false};
        u32 info_index{NO_INFO};
        /// Set by DrawContext::ApplyShadows: the binding is served from a live shadow.
        bool redirected{false};
        /// A uniform range that may create its pool's mirror (EDEN_NX2_TRY_UBO_MIRROR); with
        /// `shadowable` it is served from the pool's mirror like a read-only storage range.
        bool creates_mirror{false};
    };
    /// One image descriptor whose view is created in phase B; `index` is its VkDescriptorImageInfo
    /// in the draw context's scratch_images (a placeholder until then).
    struct ImageBinding {
        u32 index;
        Texture* texture; ///< the view is created from its storage (alive for the draw)
        ViewKey key;
        VkImageViewType dummy_type;
        bool is_image;
        bool scratch; ///< a feedback copy: not a texture object of the game
        /// Filled only while a `textures` diag request is capturing (DrawContext::diag_bindings):
        /// the texture the game bound (`texture` is its feedback copy when `scratch`), the
        /// shader stage and texture slot, and the sampler.
        std::shared_ptr<Texture> game_texture{};
        u32 stage{};
        u32 slot{};
        u64 sampler_serial{};
    };
    /// A render target of this draw that the draw also samples: copied into `scratch` before the
    /// pass (every level in [level, level + levels), every layer in [layer, layer + layers), a 3D
    /// image copies its whole depth).
    struct FeedbackCopy {
        std::shared_ptr<Storage> source;
        std::shared_ptr<Storage> scratch;
        std::shared_ptr<Texture> scratch_texture; ///< view source of the copy
        u32 level;
        u32 levels;
        u32 layer;
        u32 layers;
        /// Non-zero: the source is a 1x1x1 2D texture sampled as another type; its texel goes
        /// to a 1x1x1 3D scratch (1) or to every face of a cube scratch (6).
        u32 retype_layers{0};
        /// The source is a 2D texture sampled as 1D: its first row goes to a 1D scratch. Skipped
        /// like a feedback copy while the scratch still holds it (PruneFreshFeedback).
        bool retype_row{false};
    };
    /// GPU copies to arena uniform buffers for misaligned GPU-owned/DeviceLocal ranges.
    struct UniformCopy {
        VkBuffer src;
        u64 src_offset;
        VkBuffer dst;
        u64 dst_offset;
        u64 size;
    };
    std::vector<ImageUse> images;
    std::vector<RangeUse> ranges;
    std::vector<ImageBinding> bindings;
    std::vector<FeedbackCopy> feedback;
    std::vector<UniformCopy> uniform_copies;
    /// Allocated descriptor set (pipelines that cannot push), filled by PushPlan.
    VkDescriptorSet set{VK_NULL_HANDLE};
    /// The plan is for a graphics draw (false: a dispatch); only the fallback counters read it.
    bool graphics{true};
    /// The pipeline's shaders reach any pool through raw global memory: no storage range of the draw
    /// is shadowed and every live shadow is copied back before it.
    bool raw_global{false};
    /// A written storage range of the draw is a candidate for a shadow (some RangeUse::shadowable).
    bool shadow_candidates{false};
    /// The pipeline writes untracked global memory (any backed pool): the pass must end before an
    /// indirect draw reads arguments, as after a draw that writes a storage buffer.
    bool raw_global_writes{false};
    /// ... and every raw store also wrote each mirror alias table range the draw was given:
    /// read-only mirrors stay current and storage mirrors also reach the imported guest backing.
    /// (EDEN_NX2_TRY_RO_MIRROR_EXACT / EDEN_NX2_TRY_STORAGE_RAW_WRITE_THROUGH).
    bool raw_stores_written_through{false};
    /// The pipeline writes a bound storage image.
    bool image_writes{false};
    /// An overlapping inline uniform update must flush to the GPU pool before the draw.
    bool flush_inline{false};
    /// Resolution scaling: which texture and storage image descriptors (in the order of the
    /// pipeline's stages) the draw binds the scaled twin of (Rescale::Words).
    Rescale::Words rescale_words;

    /// Empties the plan, keeping its capacity (one plan serves every draw of a context).
    void Clear() {
        images.clear();
        ranges.clear();
        bindings.clear();
        feedback.clear();
        uniform_copies.clear();
        set = VK_NULL_HANDLE;
        raw_global_writes = false;
        raw_stores_written_through = false;
        raw_global = false;
        shadow_candidates = false;
        image_writes = false;
        flush_inline = false;
        rescale_words.Clear();
    }
    void UseImage(const std::shared_ptr<Storage>& storage, VkImageLayout layout,
                  bool sample_only = false, bool write = false) {
        image_writes |= write;
        for (ImageUse& use : images) {
            if (use.owner == storage) {
                if (use.layout != layout) {
                    use.layout = VK_IMAGE_LAYOUT_GENERAL;
                }
                use.sample_only = use.sample_only && sample_only;
                use.write |= write;
                return;
            }
        }
        images.push_back({storage, layout, false, false, sample_only, write});
    }
    /// The layout the plan puts `storage` in (`fallback` when the plan does not use it).
    [[nodiscard]] VkImageLayout LayoutOf(const Storage* storage, VkImageLayout fallback) const {
        for (const ImageUse& use : images) {
            if (use.owner.get() == storage) {
                return use.layout;
            }
        }
        return fallback;
    }
    void UseRange(Pool* pool, u64 offset, u64 size, bool write, u32 info_index = RangeUse::NO_INFO,
                  bool shadowable = false, bool creates_mirror = false) {
        if (pool != nullptr) {
            ranges.push_back({pool, offset, size, write, shadowable, info_index, false,
                              creates_mirror});
        }
    }
};

} // namespace Vulkan::Nvn
