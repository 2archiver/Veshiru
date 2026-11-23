// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nvn/shader/nvn_compute_bank.h"

#include <algorithm>
#include <cstring>

namespace VideoCommon::Nvn {

namespace {

struct BufferEntry {
    u64 address;
    u32 size;
    u32 pad;
};
static_assert(sizeof(BufferEntry) == StorageBufferStride);

} // namespace

u32 ComputeBankSize(const ComputeDriverLayout& layout) noexcept {
    return (std::max)(layout.end_offset,
                      layout.push_constant_offset + layout.push_constant_words * 4);
}

void FillComputeDriverBank(const ComputeDriverLayout& layout, const ComputeBankInputs& inputs,
                           std::span<u8> bank) {
    std::memset(bank.data(), 0, bank.size());

    // Header: the bank size, then the group counts. DK's agl_common 16x16 shaders read c0+4.
    if (bank.size() >= 4 * sizeof(u32)) {
        const u32 header[4] = {layout.declared_total != 0 ? layout.declared_total : layout.end_offset,
                               inputs.group_counts[0], inputs.group_counts[1],
                               inputs.group_counts[2]};
        std::memcpy(bank.data(), header, sizeof(header));
    }

    const size_t push_bytes = (std::min<size_t>)(
        inputs.push_constants.size(), size_t{layout.push_constant_words} * sizeof(u32));
    if (push_bytes != 0 && layout.push_constant_offset + push_bytes <= bank.size()) {
        std::memcpy(bank.data() + layout.push_constant_offset, inputs.push_constants.data(),
                    push_bytes);
    }

    const auto fill_buffers = [&](u32 offset, u32 count, std::span<const BufferRange> ranges) {
        for (u32 slot = 0; slot < count && slot < ranges.size(); ++slot) {
            const BufferEntry entry{ranges[slot].address,
                                    static_cast<u32>((std::min<u64>)(ranges[slot].size, ~0U)), 0};
            const size_t position = offset + size_t{slot} * StorageBufferStride;
            if (position + sizeof(entry) <= bank.size()) {
                std::memcpy(bank.data() + position, &entry, sizeof(entry));
            }
        }
    };
    fill_buffers(layout.uniform_offset, layout.uniform_count, inputs.uniform_buffers);
    fill_buffers(layout.storage_offset, layout.storage_count, inputs.storage_buffers);

    const auto fill_handles = [&](u32 offset, u32 count, std::span<const u64> handles) {
        for (u32 slot = 0; slot < count && slot < handles.size(); ++slot) {
            const size_t position = offset + size_t{slot} * HandleStride;
            const u32 word = static_cast<u32>(handles[slot]);
            if (position + sizeof(u32) <= bank.size()) {
                std::memcpy(bank.data() + position, &word, sizeof(word));
            }
        }
    };
    fill_handles(layout.texture_offset, layout.texture_count, inputs.textures);
    fill_handles(layout.image_offset, layout.image_count, inputs.images);
}

std::vector<u8> BuildComputeDriverBank(const ComputeDriverLayout& layout,
                                       const ComputeBankInputs& inputs) {
    std::vector<u8> bank(ComputeBankSize(layout), 0);
    FillComputeDriverBank(layout, inputs, bank);
    return bank;
}

} // namespace VideoCommon::Nvn
