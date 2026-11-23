// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// What the draw path has already recorded into the OPEN render pass of a command buffer, so a draw
// that would record the same thing again records nothing (tests: src/tests/nvn/pass_state.cpp).
//
// The state is valid between a pass begin and its end only: every other recorder (compute, copies,
// masked clears, present blits, the start of a command buffer) works outside a pass, so it can have
// disturbed anything and `Reset` runs at every pass begin, every pass end and every command buffer
// start. Inside the pass only draws record commands, so the Vulkan state is exactly what this
// class remembers. Vulkan rules relied on (spec "Pipeline Layout Compatibility"):
//  - Vertex and index buffer bindings and push descriptor contents survive a pipeline bind; a
//    different pipeline layout object disturbs the push descriptor set, so descriptors are only
//    skipped while the bound layout is the very same object (identity, not compatibility).
//  - Push constants survive a pipeline bind whose layout has identical push constant ranges; every
//    graphics layout of the replay that uses push constants has the same single range, one without
//    them disturbs the values.

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "video_core/vulkan_common/vulkan.h"

#include "common/common_types.h"

namespace Vulkan::Nvn {

/// Preserve exact floating-point bits when deciding whether dynamic state changed.
template <typename T>
[[nodiscard]] bool ReplaceDynamicState(T& current, const T& value, bool elide) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (elide && std::memcmp(&current, &value, sizeof(T)) == 0) {
        return false;
    }
    std::memcpy(&current, &value, sizeof(T));
    return true;
}

/// Levels and layers of an image (a feedback copy, or what a draw samples).
struct SubresourceSpan {
    u32 level{0};
    u32 levels{0};
    u32 layer{0};
    u32 layers{0};
    /// True when every level and layer of `other` is inside this span.
    [[nodiscard]] constexpr bool Covers(const SubresourceSpan& other) const noexcept {
        return other.level >= level && other.level + other.levels <= level + levels &&
               other.layer >= layer && other.layer + other.layers <= layer + layers;
    }
};

/// The epoch that decides whether the scratch copy of a sampled render target is current. The
/// scratch holds the source as of the epoch it was copied in; anything the guest can order against
/// the sampled texture starts a new epoch: a change of the bound render targets (a mip chain: pass
/// k renders level k, the next target change starts the epoch in which level k is sampled), a game
/// Barrier, and every pass end except the break of a feedback copy itself (it copies right away).
struct FeedbackEpoch {
    u64 value{1};
    void OnTargetChange() noexcept {
        ++value;
    }
    void OnBarrier() noexcept {
        ++value;
    }
    void OnPassEnd(bool feedback_break) noexcept {
        if (!feedback_break) {
            ++value;
        }
    }
};

/// What a feedback scratch holds: the source it was copied from, who copied it, in which epoch and
/// which levels/layers.
template <typename Source>
struct FeedbackSnapshot {
    std::weak_ptr<Source> from;
    const void* by{nullptr};
    u64 epoch{0};
    SubresourceSpan span;

    void Record(const std::shared_ptr<Source>& source, const void* context, u64 now,
                const SubresourceSpan& copied) {
        from = source;
        by = context;
        epoch = now;
        span = copied;
    }
    /// True when a draw of `context` in epoch `now` that samples `wanted` of `source` can use the
    /// scratch without copying again.
    [[nodiscard]] bool Serves(const std::shared_ptr<Source>& source, const void* context, u64 now,
                              const SubresourceSpan& wanted) const {
        return by == context && epoch == now && !from.owner_before(source) &&
               !source.owner_before(from) && span.Covers(wanted);
    }
};

/// The rescaling words and the down factor of a draw (Resolution scaling; all zero / 1.0 when the
/// pass is guest-sized): the part of the push constants that changes between draws. Declared
/// outside PassBoundState: a default argument and a member initializer of the class use it while
/// the class is incomplete, which a nested type with default member initializers does not allow.
struct PassPushWords {
    std::array<u32, 6> words{};
    float down_factor{1.0f};
    bool operator==(const PassPushWords&) const = default;
};

class PassBoundState {
public:
    static constexpr u32 MAX_STREAMS = 32;

    /// One write of a descriptor set, as the replay builds them: binding, type, array size.
    struct WriteSig {
        u32 binding;
        u32 type;
        u32 count;
        bool operator==(const WriteSig&) const = default;
    };
    struct ImageSig {
        VkSampler sampler;
        VkImageView view;
        VkImageLayout layout;
        bool operator==(const ImageSig&) const = default;
    };

    void Reset() noexcept {
        layout_ = VK_NULL_HANDLE;
        constants_ = false;
        area_ = false;
        descriptors_ = false;
        streams_valid_ = 0;
        index_valid_ = false;
    }

    /// A graphics pipeline with `layout` is bound (its first draw in the pass, or a change).
    /// `uses_push_constants`: the layout has the push constant range.
    void OnPipelineBind(VkPipelineLayout layout, bool uses_push_constants) noexcept {
        if (layout == layout_) {
            return;
        }
        layout_ = layout;
        descriptors_ = false;
        if (!uses_push_constants) {
            constants_ = false;
            area_ = false;
        }
    }

    using PushWords = PassPushWords;

    /// True when the push constants (rescaling words, down factor and, with `needs_area`, the
    /// render area) the draw would push are already in place; otherwise marks them pushed.
    [[nodiscard]] bool TakePushConstants(bool needs_area, const PushWords& pushed = {}) noexcept {
        if (constants_ && (!needs_area || area_) && pushed == words_) {
            return true;
        }
        constants_ = true;
        area_ = area_ || needs_area;
        words_ = pushed;
        return false;
    }

    /// True when the descriptor writes (`writes`, flat `buffers` and `images` of all of them) are
    /// exactly what is bound for `layout` already. Otherwise remembers them as bound.
    [[nodiscard]] bool TakeDescriptors(VkPipelineLayout layout, const std::vector<WriteSig>& writes,
                                       const std::vector<VkDescriptorBufferInfo>& buffers,
                                       const std::vector<ImageSig>& images) {
        if (descriptors_ && layout == layout_ && writes == writes_ && SameBuffers(buffers) &&
            images == images_) {
            return true;
        }
        writes_ = writes;
        buffers_ = buffers;
        images_ = images;
        descriptors_ = layout == layout_;
        return false;
    }
    /// The next descriptor push is not comparable (a write carries texel buffer views).
    void ForgetDescriptors() noexcept {
        descriptors_ = false;
    }

    /// True when vertex stream `stream` already has this buffer and offset bound; otherwise
    /// remembers it as bound.
    [[nodiscard]] bool TakeStream(u32 stream, VkBuffer buffer, VkDeviceSize offset) noexcept {
        if (stream >= MAX_STREAMS) {
            return false;
        }
        const u32 bit = 1u << stream;
        if ((streams_valid_ & bit) != 0 && stream_buffers_[stream] == buffer &&
            stream_offsets_[stream] == offset) {
            return true;
        }
        streams_valid_ |= bit;
        stream_buffers_[stream] = buffer;
        stream_offsets_[stream] = offset;
        return false;
    }

    [[nodiscard]] bool TakeIndexBuffer(VkBuffer buffer, VkDeviceSize offset,
                                       VkIndexType type) noexcept {
        if (index_valid_ && index_buffer_ == buffer && index_offset_ == offset &&
            index_type_ == type) {
            return true;
        }
        index_valid_ = true;
        index_buffer_ = buffer;
        index_offset_ = offset;
        index_type_ = type;
        return false;
    }
    /// An index buffer was bound outside TakeIndexBuffer (generated indices).
    void ForgetIndexBuffer() noexcept {
        index_valid_ = false;
    }

private:
    [[nodiscard]] bool SameBuffers(const std::vector<VkDescriptorBufferInfo>& other) const {
        return other.size() == buffers_.size() &&
               (other.empty() || std::memcmp(other.data(), buffers_.data(),
                                             other.size() * sizeof(VkDescriptorBufferInfo)) == 0);
    }

    VkPipelineLayout layout_{VK_NULL_HANDLE};
    bool constants_{false};
    bool area_{false};
    PushWords words_{};
    bool descriptors_{false};
    std::vector<WriteSig> writes_;
    std::vector<VkDescriptorBufferInfo> buffers_;
    std::vector<ImageSig> images_;
    u32 streams_valid_{0};
    std::array<VkBuffer, MAX_STREAMS> stream_buffers_{};
    std::array<VkDeviceSize, MAX_STREAMS> stream_offsets_{};
    bool index_valid_{false};
    VkBuffer index_buffer_{VK_NULL_HANDLE};
    VkDeviceSize index_offset_{0};
    VkIndexType index_type_{VK_INDEX_TYPE_UINT16};
};

/// Images that flip between attachment and sampled use, with the layout they are sampled in
/// (DrawContext::flip_targets, moved to it when a pass begins). Learned from layout breaks, at
/// most CAPACITY; a full list replaces the entry used longest ago, so it keeps learning the
/// targets of the scene in use (the images of earlier scenes stay alive in the storage cache).
template <typename T>
class FlipTargetList {
public:
    static constexpr size_t CAPACITY = 256;
    struct Entry {
        std::weak_ptr<T> target;
        VkImageLayout sampled_layout;
        u64 used; ///< clock at the last Note or Touch
    };

    /// A layout break of `target`, sampled in `layout`. True when a full list replaced an entry.
    bool Note(const std::shared_ptr<T>& target, VkImageLayout layout) {
        ++clock;
        for (Entry& entry : entries) {
            if (!entry.target.owner_before(target) && !target.owner_before(entry.target)) {
                entry.sampled_layout = layout;
                entry.used = clock;
                return false;
            }
        }
        if (entries.size() < CAPACITY) {
            entries.push_back({target, layout, clock});
            return false;
        }
        Entry* oldest = &entries.front();
        for (Entry& entry : entries) {
            oldest = entry.used < oldest->used ? &entry : oldest;
        }
        *oldest = {target, layout, clock};
        return true;
    }
    /// Entry `index` was found rendered to when a pass began: still in use.
    void Touch(size_t index) {
        entries[index].used = ++clock;
    }
    /// Forgets entry `index` (its image is gone); the last entry takes its place.
    void Erase(size_t index) {
        entries[index] = std::move(entries.back());
        entries.pop_back();
    }
    [[nodiscard]] size_t Size() const noexcept {
        return entries.size();
    }
    [[nodiscard]] Entry& operator[](size_t index) {
        return entries[index];
    }

private:
    std::vector<Entry> entries;
    u64 clock{0};
};

} // namespace Vulkan::Nvn
