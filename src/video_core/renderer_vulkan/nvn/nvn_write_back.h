// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Write-back of GPU results held in device-local copies of guest memory (pool mirrors) to the
// guest's own bytes.
//
// A plain copy of the device-local range back over the guest range also writes every word the GPU
// left alone with the value the copy started from, so a CPU write made to the same range while the
// GPU ran is lost (TotK's particle emitters write new records into a buffer the simulation shaders
// update). The merge writes only the 4-byte words whose result differs from the snapshot the GPU
// work started from: words the GPU did not change keep whatever the CPU put there, a word both
// changed takes the GPU's value (as a copy would). The idea and the GPU form (one compute dispatch
// per range, changed words straight into the imported guest buffer) follow the TotK recompilation's
// write-back merge (vk_memory.c "write-back merge", MIT).

#include <span>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// Writes into `guest` the bytes of `result` that differ from `snapshot` (all three the same
/// length): every naturally aligned 4-byte word of `guest` (by its address) whose result differs
/// from its snapshot is stored whole, with one 32-bit store; the bytes before the first and after
/// the last whole word are compared and stored one by one. Words and bytes whose result equals the
/// snapshot are not written at all, so concurrent CPU writes to them survive. Returns the bytes
/// written.
u64 MergeChangedWords(std::span<u8> guest, std::span<const u8> snapshot,
                      std::span<const u8> result);

/// One range for WriteBackMerge: `size` bytes at `result_off` of `result` (the GPU's output, e.g.
/// the pool mirror), `snapshot_off` of `snapshot` (the same bytes before the GPU work) and
/// `guest_off` of `guest` (the imported pool buffer). The three offsets must be equal modulo 4
/// (the merge compares whole words of the guest's alignment), and each buffer must reach the end of
/// the range rounded up to 4 bytes.
struct MergeRange {
    VkBuffer result;
    u64 result_off;
    VkBuffer snapshot;
    u64 snapshot_off;
    VkBuffer guest;
    u64 guest_off;
    u64 size;
    /// Diagnostic counter slot of the stats buffer (WriteBackMerge::SetStats), NO_STAT for none.
    u32 stat_slot{0xFFFFFFFFu};
    /// Line mode may store whole units of this range (WriteBackMerge::SetLineMode must be on
    /// too). Only for ranges whose guest pages no CPU store can reach during the merge
    /// (MirrorLineGuard); false: the exact word merge.
    bool lines{false};
};

/// MergeChangedWords on the GPU (host_shaders/nvn_wb_merge.comp): one dispatch per range (per 64
/// MiB of it), each invocation one guest word. Whole words are stored when they changed; a partial
/// first or last word changes only its bytes inside the range that changed (atomicAnd/atomicOr,
/// the other bytes stay as they are). Only changed words cross PCIe. The caller orders the GPU
/// writes of the results before the merge and the merge before the host reads with barriers
/// (compute shader reads of the result and snapshot, shader writes of the guest buffer).
class WriteBackMerge {
public:
    /// Push descriptors for the four storage buffers.
    [[nodiscard]] static bool Supported(const Device& device);

    WriteBackMerge();
    ~WriteBackMerge();
    WriteBackMerge(const WriteBackMerge&) = delete;
    WriteBackMerge& operator=(const WriteBackMerge&) = delete;

    /// Creates the pipeline. False (and logged) when the device cannot run it.
    bool Init(const Device& device);
    /// Diagnostic counters (EDEN_NX2_NVN_WB_MONITOR): a storage buffer of 2 words per slot that
    /// merges of ranges with a `stat_slot` add their changed words and the dirty 32-word lines of the
    /// guest buffer to.
    void SetStats(VkBuffer buffer) noexcept {
        stats = buffer;
    }
    /// Line mode (EDEN_NX2_TRY_WB_LINES, nvn_wb_merge.comp): in a range with `lines` set, a unit of
    /// `unit_words` (8 or 32) words of the guest buffer with two or more changed words is stored
    /// whole (the other words as the guest holds them just before), isolated changed words one by
    /// one. Bytes the GPU changes are the same; a CPU store to an unchanged word of such a unit
    /// is lost when it lands within about a microsecond of that unit's store, so the caller only
    /// sets `lines` where no CPU store can come (MirrorLineGuard).
    void SetLineMode(bool enabled, u32 words_per_unit = 8) noexcept {
        line_mode = enabled;
        unit_words = words_per_unit >= 32 ? 32 : 8;
    }
    [[nodiscard]] u32 UnitWords() const noexcept {
        return unit_words;
    }
    [[nodiscard]] bool Ready() const noexcept {
        return static_cast<bool>(pipeline);
    }

    /// Records the merges of `ranges`. Returns how many ranges were recorded: a range whose offsets
    /// differ modulo 4 is not (logged), an empty one needs nothing.
    u64 Record(vk::CommandBuffer cmd, std::span<const MergeRange> ranges) const;

private:
    void RecordPart(vk::CommandBuffer cmd, const MergeRange& range, u64 begin, u64 end) const;

    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    u64 alignment{};
    u64 part_bytes{};
    VkBuffer stats{VK_NULL_HANDLE};
    bool line_mode{false};
    u32 unit_words{8};
};

} // namespace Vulkan::Nvn
