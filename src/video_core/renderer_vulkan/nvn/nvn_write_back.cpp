// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/host_shaders/nvn_wb_merge_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_write_back.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

/// The push constant block of nvn_wb_merge.comp.
struct PushConstants {
    u32 result_word;   ///< first word of the part in the result binding
    u32 snapshot_word; ///< ... in the snapshot binding
    u32 guest_word;    ///< ... in the guest binding
    u32 words;         ///< words of the part (the first and the last may be partial)
    u32 head_mask;     ///< bytes of the first word inside the part
    u32 tail_mask;     ///< bytes of the last word inside the part
    u32 stat_slot;     ///< diagnostic counter slot, 0xFFFFFFFF for none
    u32 line_phase;    ///< (guest binding word offset + guest_word) % unit words: the first word's place in its unit
    u32 flags;         ///< FLAG_LINES, bits 8..11: log2 of the unit words
};
static_assert(sizeof(PushConstants) == 36);

constexpr u32 FLAG_LINES = 1;
constexpr u32 MAX_UNIT_WORDS = 32;

constexpr u32 GROUP_SIZE = 256;
constexpr u64 MAX_GROUPS_X = 65535;
/// Bytes one dispatch covers at most (one invocation per word).
constexpr u64 MAX_PART_BYTES = (MAX_GROUPS_X * GROUP_SIZE - MAX_UNIT_WORDS) * 4;

} // Anonymous namespace

u64 MergeChangedWords(std::span<u8> guest, std::span<const u8> snapshot,
                      std::span<const u8> result) {
    ASSERT(guest.size() == snapshot.size() && guest.size() == result.size());
    const size_t size = guest.size();
    u8* const out = guest.data();
    const u8* const before = snapshot.data();
    const u8* const after = result.data();
    u64 written = 0;
    const auto merge_byte = [&](size_t i) {
        if (before[i] != after[i]) {
            std::atomic_ref<u8>(out[i]).store(after[i], std::memory_order_relaxed);
            ++written;
        }
    };
    // Words follow the guest's own alignment: a 32-bit CPU store to one of them is never split.
    const size_t head =
        (std::min)(size, static_cast<size_t>((4 - (reinterpret_cast<uintptr_t>(out) & 3)) & 3));
    size_t i = 0;
    for (; i < head; ++i) {
        merge_byte(i);
    }
    const size_t words_end = head + (size - head) / 4 * 4;
    for (; i < words_end; i += 4) {
        u32 old_word;
        u32 new_word;
        std::memcpy(&old_word, before + i, sizeof(u32));
        std::memcpy(&new_word, after + i, sizeof(u32));
        if (old_word != new_word) {
            std::atomic_ref<u32>(*reinterpret_cast<u32*>(out + i))
                .store(new_word, std::memory_order_relaxed);
            written += sizeof(u32);
        }
    }
    for (; i < size; ++i) {
        merge_byte(i);
    }
    return written;
}

bool WriteBackMerge::Supported(const Device& device) {
    return device.IsKhrPushDescriptorSupported() && device.MaxPushDescriptors() >= 4;
}

WriteBackMerge::WriteBackMerge() = default;
WriteBackMerge::~WriteBackMerge() = default;

bool WriteBackMerge::Init(const Device& device) {
    if (pipeline) {
        return true;
    }
    if (!Supported(device)) {
        LOG_WARNING(Render_Vulkan, "[nvn] write-back merge unavailable: the device has no push "
                                   "descriptors for three storage buffers");
        return false;
    }
    alignment = (std::max)(device.GetStorageBufferAlignment(), VkDeviceSize{4});
    const u64 max_range = device.GetMaxStorageBufferRange();
    if (max_range <= alignment + 4) {
        LOG_WARNING(Render_Vulkan, "[nvn] write-back merge unavailable: storage buffer range {}",
                    max_range);
        return false;
    }
    // A part binds at most its own bytes plus the alignment below its first word in each buffer.
    part_bytes = (std::min)(MAX_PART_BYTES, (max_range - alignment) & ~u64{3});
    try {
        const vk::Device& logical = device.GetLogical();
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (u32 i = 0; i < bindings.size(); ++i) {
            bindings[i] = {
                .binding = i,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            };
        }
        set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };
        pipeline_layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = set_layout.address(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        });
        const std::span<const u32> code = NVN_WB_MERGE_COMP_SPV;
        module = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = static_cast<u32>(code.size_bytes()),
            .pCode = code.data(),
        });
        pipeline = logical.CreateComputePipeline(
            VkComputePipelineCreateInfo{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = *module,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
                },
                .layout = *pipeline_layout,
                .basePipelineHandle = {},
                .basePipelineIndex = 0,
            },
            // One small pipeline: the shared static pipeline cache is externally synchronized.
            VK_NULL_HANDLE);
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] write-back merge pipeline failed: {}", exception.what());
        pipeline = {};
        module = {};
        pipeline_layout = {};
        set_layout = {};
        return false;
    }
    return true;
}

u64 WriteBackMerge::Record(vk::CommandBuffer cmd, std::span<const MergeRange> ranges) const {
    ASSERT(pipeline);
    u64 recorded = 0;
    bool bound = false;
    for (const MergeRange& range : ranges) {
        if (range.size == 0) {
            continue;
        }
        if (((range.result_off ^ range.guest_off) & 3) != 0 ||
            ((range.snapshot_off ^ range.guest_off) & 3) != 0) {
            LOG_ERROR(Render_Vulkan,
                      "[nvn] write-back merge: offsets result {:#x} snapshot {:#x} guest {:#x} "
                      "differ modulo 4, {} bytes not merged",
                      range.result_off, range.snapshot_off, range.guest_off, range.size);
            continue;
        }
        if (!bound) {
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            bound = true;
        }
        // Parts end at guest offsets that are multiples of 4, so only the range's own first and
        // last words can be partial.
        u64 begin = 0;
        while (begin < range.size) {
            const u64 limit = ((range.guest_off + begin) & ~u64{3}) + part_bytes;
            const u64 end = (std::min)(range.size, limit - range.guest_off);
            RecordPart(cmd, range, begin, end);
            begin = end;
        }
        ++recorded;
    }
    return recorded;
}

void WriteBackMerge::RecordPart(vk::CommandBuffer cmd, const MergeRange& range, u64 begin,
                                u64 end) const {
    struct Binding {
        VkDescriptorBufferInfo info;
        u32 word;
    };
    const auto bind = [&](VkBuffer buffer, u64 offset) {
        const u64 first = (offset + begin) & ~u64{3};
        const u64 last = (offset + end + 3) & ~u64{3};
        const u64 base = first / alignment * alignment;
        return Binding{{.buffer = buffer, .offset = base, .range = last - base},
                       static_cast<u32>((first - base) / 4)};
    };
    const std::array<Binding, 4> bindings{
        bind(range.result, range.result_off),
        bind(range.snapshot, range.snapshot_off),
        bind(range.guest, range.guest_off),
        // The counters, or (unused: the shader leaves them alone) the guest buffer again.
        stats != VK_NULL_HANDLE
            ? Binding{{.buffer = stats, .offset = 0, .range = VK_WHOLE_SIZE}, 0}
            : bind(range.guest, range.guest_off),
    };
    std::array<VkWriteDescriptorSet, 4> writes{};
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = i,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bindings[i].info,
            .pTexelBufferView = nullptr,
        };
    }
    const u64 guest_begin = range.guest_off + begin;
    const u64 guest_end = range.guest_off + end;
    const u32 words = static_cast<u32>((((guest_end + 3) & ~u64{3}) - (guest_begin & ~u64{3})) / 4);
    const u32 head_shift = static_cast<u32>(guest_begin & 3) * 8;
    const u32 tail_bytes = static_cast<u32>(guest_end & 3);
    // The guest buffer is imported host memory (page aligned): its word n is at 4n bytes from a
    // unit boundary, so a unit of 8 or 32 words starts where (binding offset + word) % unit == 0.
    // The plain path has no units and needs no phase.
    const bool lines = line_mode && range.lines;
    const bool counted = stats != VK_NULL_HANDLE && range.stat_slot != 0xFFFFFFFFu;
    const u32 line_phase =
        counted || lines
            ? static_cast<u32>((bindings[2].info.offset / 4 + bindings[2].word) % unit_words)
            : 0;
    const PushConstants constants{
        .result_word = bindings[0].word,
        .snapshot_word = bindings[1].word,
        .guest_word = bindings[2].word,
        .words = words,
        .head_mask = ~u32{0} << head_shift,
        .tail_mask = tail_bytes == 0 ? ~u32{0} : (u32{1} << (tail_bytes * 8)) - 1,
        .stat_slot = stats != VK_NULL_HANDLE ? range.stat_slot : u32{0xFFFFFFFFu},
        .line_phase = line_phase,
        .flags = counted || lines ? (lines ? FLAG_LINES : 0) |
                                        (static_cast<u32>(std::countr_zero(unit_words)) << 8)
                                  : 0,
    };
    cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, writes);
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    cmd.Dispatch((words + line_phase + GROUP_SIZE - 1) / GROUP_SIZE, 1, 1);
}

} // namespace Vulkan::Nvn
