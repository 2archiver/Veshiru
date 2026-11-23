// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Descriptor resolution of the draw path: handle words -> textures/samplers, buffer bindings,
// the driver/handle constant banks, and pushing/allocating the descriptor set.

#include <algorithm>
#include <bit>
#include <cstring>
#include <unordered_map>

#include <fmt/format.h>

#include "common/logging.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/buffer_cache/raw_coverage.h"
#include "video_core/nvn/shader/nvn_handle_program.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_plan.h"
#include "video_core/renderer_vulkan/nvn/nvn_sampler_aniso.h"

namespace Vulkan::Nvn {
namespace {

constexpr u64 MAX_UNIFORM_RANGE = 65536;
constexpr u64 ZERO_SIZE_RANGE_MAX = u64{8} << 20; // longest range of a size-0 storage entry

enum class TargetKind { Dim1D, Dim2D, Dim3D, Dim1DArray, Dim2DArray, Cube, CubeArray, Buffer, Rect };

Shader::TextureType TypeOfTarget(s32 target) {
    switch (static_cast<TexTarget>(target)) {
    case TexTarget::Dim1D:
        return Shader::TextureType::Color1D;
    case TexTarget::Dim3D:
        return Shader::TextureType::Color3D;
    case TexTarget::Dim1DArray:
        return Shader::TextureType::ColorArray1D;
    case TexTarget::Dim2DArray:
    case TexTarget::Dim2DMultisampleArray:
        return Shader::TextureType::ColorArray2D;
    case TexTarget::Rectangle:
        return Shader::TextureType::Color2DRect;
    case TexTarget::Cubemap:
        return Shader::TextureType::ColorCube;
    case TexTarget::CubemapArray:
        return Shader::TextureType::ColorArrayCube;
    case TexTarget::Buffer:
        return Shader::TextureType::Buffer;
    default:
        return Shader::TextureType::Color2D;
    }
}

Shader::TexturePixelFormat ShaderFormatOf(VideoCore::Surface::PixelFormat format) {
    using PF = VideoCore::Surface::PixelFormat;
    using SF = Shader::TexturePixelFormat;
    switch (format) {
    case PF::A8B8G8R8_SNORM:
        return SF::A8B8G8R8_SNORM;
    case PF::A8B8G8R8_SINT:
        return SF::A8B8G8R8_SINT;
    case PF::A8B8G8R8_UINT:
        return SF::A8B8G8R8_UINT;
    case PF::R5G6B5_UNORM:
        return SF::R5G6B5_UNORM;
    case PF::B5G6R5_UNORM:
        return SF::B5G6R5_UNORM;
    case PF::A2B10G10R10_UNORM:
        return SF::A2B10G10R10_UNORM;
    case PF::A2B10G10R10_UINT:
        return SF::A2B10G10R10_UINT;
    case PF::R8_UNORM:
        return SF::R8_UNORM;
    case PF::R8_SNORM:
        return SF::R8_SNORM;
    case PF::R8_SINT:
        return SF::R8_SINT;
    case PF::R8_UINT:
        return SF::R8_UINT;
    case PF::R16G16B16A16_FLOAT:
        return SF::R16G16B16A16_FLOAT;
    case PF::R16G16B16A16_UNORM:
        return SF::R16G16B16A16_UNORM;
    case PF::R16G16B16A16_SNORM:
        return SF::R16G16B16A16_SNORM;
    case PF::R16G16B16A16_SINT:
        return SF::R16G16B16A16_SINT;
    case PF::R16G16B16A16_UINT:
        return SF::R16G16B16A16_UINT;
    case PF::B10G11R11_FLOAT:
        return SF::B10G11R11_FLOAT;
    case PF::R32G32B32A32_UINT:
        return SF::R32G32B32A32_UINT;
    case PF::R32G32B32A32_FLOAT:
        return SF::R32G32B32A32_FLOAT;
    case PF::R32G32B32A32_SINT:
        return SF::R32G32B32A32_SINT;
    case PF::R32G32_FLOAT:
        return SF::R32G32_FLOAT;
    case PF::R32G32_SINT:
        return SF::R32G32_SINT;
    case PF::B8G8R8A8_UNORM:
        return SF::B8G8R8A8_UNORM;
    default:
        return SF::A8B8G8R8_UNORM;
    }
}

u32 PackSwizzle(const std::array<VkComponentSwizzle, 4>& s) {
    return static_cast<u32>(s[0]) | static_cast<u32>(s[1]) << 4 | static_cast<u32>(s[2]) << 8 |
           static_cast<u32>(s[3]) << 12;
}

/// NVNtextureSwizzle (ZERO 0, ONE 1, R 2, G 3, B 4, A 5) of a 3-bit-packed word component.
VkComponentSwizzle FromNvn(u32 value) {
    switch (value) {
    case 0:
        return VK_COMPONENT_SWIZZLE_ZERO;
    case 1:
        return VK_COMPONENT_SWIZZLE_ONE;
    case 2:
        return VK_COMPONENT_SWIZZLE_R;
    case 3:
        return VK_COMPONENT_SWIZZLE_G;
    case 4:
        return VK_COMPONENT_SWIZZLE_B;
    default:
        return VK_COMPONENT_SWIZZLE_A;
    }
}

/// Component k of a swizzle result, as the source channel it reads (R..A) or ZERO/ONE.
VkComponentSwizzle Compose(VkComponentSwizzle outer, const std::array<VkComponentSwizzle, 4>& inner) {
    switch (outer) {
    case VK_COMPONENT_SWIZZLE_R:
        return inner[0];
    case VK_COMPONENT_SWIZZLE_G:
        return inner[1];
    case VK_COMPONENT_SWIZZLE_B:
        return inner[2];
    case VK_COMPONENT_SWIZZLE_A:
        return inner[3];
    default:
        return outer;
    }
}

std::array<VkComponentSwizzle, 4> TextureSwizzle(const Texture& texture) {
    std::array<VkComponentSwizzle, 4> result{VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                             VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const VC::TextureDesc& d = texture.desc;
    if (d.set_swizzle) {
        for (u32 c = 0; c < 4; ++c) {
            result[c] = FromNvn((d.swizzle >> (3 * c)) & 7);
        }
    }
    if (d.view.set_swizzle) {
        const auto inner = result;
        for (u32 c = 0; c < 4; ++c) {
            result[c] = Compose(FromNvn(d.view.swizzle[c]), inner);
        }
    }
    return result;
}

VkImageViewType DummyViewType(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::ColorArray1D:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::Color3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorCube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::ColorArrayCube:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case Shader::TextureType::ColorArray2D:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    default:
        return VK_IMAGE_VIEW_TYPE_2D;
    }
}

} // namespace

const std::shared_ptr<Texture>& DrawContext::TextureOfId(u32 id) const {
    static const std::shared_ptr<Texture> none;
    if (!submission || id == 0 || id >= submission->texture_pool_snapshot.size()) {
        return none;
    }
    const u64 serial = submission->texture_pool_snapshot[id];
    if (serial == 0) {
        return none;
    }
    TextureSlot& slot = texture_slots[id % texture_slots.size()];
    if (slot.id == id && slot.serial == serial && slot.texture && slot.texture->live) {
        return slot.texture;
    }
    const auto it = impl.textures.find(serial);
    if (it == impl.textures.end()) {
        return none;
    }
    slot = {id, serial, it->second};
    return slot.texture;
}

u64 DrawContext::SamplerSerialOfId(u32 id) const {
    if (!submission || id >= submission->sampler_pool_snapshot.size()) {
        return 0;
    }
    return submission->sampler_pool_snapshot[id];
}

VkSampler DrawContext::SamplerOf(u32 id, u64 serial, bool boosted) {
    SamplerSlot& slot = sampler_slots[id % sampler_slots.size()];
    if (slot.id != id || slot.serial != serial || slot.epoch != resources.samplers_epoch) {
        slot = {id, serial, resources.samplers_epoch, VK_NULL_HANDLE, VK_NULL_HANDLE};
    }
    VkSampler& cached = boosted ? slot.boosted : slot.plain;
    if (cached != VK_NULL_HANDLE) {
        return cached;
    }
    const VkSampler sampler = resources.GetSampler(serial, boosted);
    if (sampler != *resources.fallback_sampler) {
        cached = sampler;
    }
    return sampler;
}

std::optional<u32> DrawContext::ReadHandleWord(u32 stage, const VC::HandleRef& ref,
                                               VC::HandleWordKind kind) const {
    if (stage >= BIND_STAGES) {
        return std::nullopt;
    }
    const StageBindings& bindings = stages[stage];
    switch (ref.source) {
    case VC::HandleSource::Combined:
        if (ref.slot < bindings.combined.size()) {
            return static_cast<u32>(bindings.combined[ref.slot]);
        }
        return std::nullopt;
    case VC::HandleSource::SeparateTexture:
        if (ref.slot < bindings.separate_texture.size()) {
            return static_cast<u32>(bindings.separate_texture[ref.slot]);
        }
        return std::nullopt;
    case VC::HandleSource::SeparateSampler:
        if (ref.slot < bindings.separate_sampler.size()) {
            return static_cast<u32>(bindings.separate_sampler[ref.slot]);
        }
        return std::nullopt;
    case VC::HandleSource::Image:
        if (ref.slot < bindings.image.size()) {
            return static_cast<u32>(bindings.image[ref.slot]);
        }
        return std::nullopt;
    case VC::HandleSource::UniformBuffer: {
        // The game stored ids of the pools it bound (nvnDeviceGetTextureHandle); the hardware
        // indexes the bound pool with them, the replay's tables hold every pool.
        const std::optional<u32> word = ReadUniformWord(stage, ref.slot, ref.byte_offset);
        if (!word) {
            return std::nullopt;
        }
        return static_cast<u32>(
            VC::RebaseHandleWord(*word, kind, texture_pool_base, sampler_pool_base));
    }
    case VC::HandleSource::None:
        return 0u;
    case VC::HandleSource::ComputeTexture:
        if (ref.slot < stages[STAGE_COMPUTE].combined.size()) {
            return static_cast<u32>(stages[STAGE_COMPUTE].combined[ref.slot]);
        }
        return std::nullopt;
    case VC::HandleSource::ComputeImage:
        if (ref.slot < stages[STAGE_COMPUTE].image.size()) {
            return static_cast<u32>(stages[STAGE_COMPUTE].image[ref.slot]);
        }
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

BufferBinding DrawContext::ResolveStorageBinding(const VC::DescriptorBinding& binding,
                                                 u32 element) const {
    const u32 stage = static_cast<u32>(binding.stage);
    const u32 slot = binding.slot + element;
    switch (binding.storage_source) {
    case VC::StorageSource::DriverTable:
        if (stage < BIND_STAGES && slot < stages[stage].storage.size()) {
            return stages[stage].storage[slot];
        }
        break;
    case VC::StorageSource::ComputeStorageEntry:
        if (slot < stages[STAGE_COMPUTE].storage.size()) {
            return stages[STAGE_COMPUTE].storage[slot];
        }
        break;
    case VC::StorageSource::ComputeUniformEntry:
        if (slot < stages[STAGE_COMPUTE].uniform.size()) {
            return stages[STAGE_COMPUTE].uniform[slot];
        }
        break;
    case VC::StorageSource::UniformBufferWord: {
        const u32 offset = binding.storage_offset + element * VC::StorageBufferStride;
        const auto lo = ReadUniformWord(stage, binding.slot, offset);
        const auto hi = ReadUniformWord(stage, binding.slot, offset + 4);
        const auto size = ReadUniformWord(stage, binding.slot, offset + 8);
        if (lo && hi && size) {
            return {static_cast<u64>(*lo) | static_cast<u64>(*hi) << 32, *size};
        }
        break;
    }
    default:
        break;
    }
    return {};
}

std::optional<u32> DrawContext::ReadBindingHandleWord(const VC::DescriptorLayout& layout,
                                                      const VC::DescriptorBinding& binding,
                                                      const VC::HandleRef& ref,
                                                      VC::HandleWordKind kind) const {
    const u32 stage = static_cast<u32>(binding.stage);
    if (ref.source != VC::HandleSource::Program) {
        return ReadHandleWord(stage, ref, kind);
    }
    // The shader reads the handle from a table in memory (a storage buffer the game bound): the
    // hardware runs that load for each fragment, the replay evaluates it once for the draw from
    // the same words. A word the game stored holds ids of the pools it bound, like a word read
    // from a uniform buffer.
    VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire);
    const auto word = VC::EvaluateHandleProgram(
        binding.handle_program, impl.device.GetStorageBufferAlignment(),
        [&](u32 bank, u32 offset) -> u32 {
            if (bank >= VC::FirstUniformBank) {
                return ReadUniformWord(stage, bank - VC::FirstUniformBank, offset).value_or(0);
            }
            return 0;
        },
        [&](u32 index) -> VC::HandleProgramBuffer {
            for (const VC::DescriptorBinding& storage : layout.bindings) {
                if (storage.role == VC::DescriptorRole::StorageBuffer &&
                    storage.stage == binding.stage && storage.storage_index == index) {
                    const BufferBinding resolved = ResolveStorageBinding(storage, 0);
                    return {resolved.address, resolved.size};
                }
            }
            return {};
        },
        [&](u64 address) -> std::optional<u32> {
            u32 value = 0;
            if (memory == nullptr || !memory->Read(address, &value, sizeof(value))) {
                return std::nullopt;
            }
            return value;
        });
    if (!word) {
        return std::nullopt;
    }
    return static_cast<u32>(
        VC::RebaseHandleWord(*word, kind, texture_pool_base, sampler_pool_base));
}

namespace {

/// A texture buffer: a view of a pool range with no image (and so no storage).
bool IsTextureBuffer(const Texture& texture) {
    const s32 target = texture.desc.view.set_target ? texture.desc.view.target : texture.desc.target;
    return target == static_cast<s32>(TexTarget::Buffer) &&
           texture.host.format != VK_FORMAT_UNDEFINED;
}

/// What the shader translation needs to know about the texture behind a handle.
TextureInfo InfoOfTexture(const Texture& texture) {
    const s32 target = texture.desc.view.set_target ? texture.desc.view.target : texture.desc.target;
    TextureInfo info;
    info.type = TypeOfTarget(target);
    info.format = ShaderFormatOf(texture.host.pixel_format);
    info.is_integer = VideoCore::Surface::IsPixelFormatInteger(texture.host.pixel_format);
    return info;
}

/// Longest uniform buffer range a frozen oracle keeps (the NVN limit is 64 KiB).
constexpr u64 FROZEN_UNIFORM_BYTES = 65536;
constexpr u64 FROZEN_CHUNK = 4096;

/// A copy of the state QueryTexture reads (see DrawContext::FreezeOracle).
class FrozenOracle final : public TextureOracle {
public:
    struct Uniform {
        std::vector<u8> bytes;    ///< the bound range (inline updates applied)
        std::vector<u8> readable; ///< per FROZEN_CHUNK: guest memory could be read
        /// Byte spans written by inline updates, or read on their own in an unreadable chunk.
        std::vector<std::pair<u64, u64>> inline_written;

        [[nodiscard]] bool WordReadable(u64 offset) const noexcept {
            for (u64 byte = offset; byte < offset + sizeof(u32); ++byte) {
                if (readable[byte / FROZEN_CHUNK] != 0) {
                    continue;
                }
                bool covered = false;
                for (const auto& [lo, hi] : inline_written) {
                    if (lo <= byte && byte < hi) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    return false;
                }
            }
            return true;
        }
    };
    struct Stage {
        std::array<u32, 64> combined{};
        std::array<u32, 128> separate_texture{};
        std::array<u32, 32> image{};
        std::array<Uniform, 32> uniform;
    };

    [[nodiscard]] TextureInfo Query(u32 stage, const VC::NvnHandleRecord& record) const override {
        VC::HandleClassification classified = VC::ClassifyHandleWord(
            record.bank, record.offset, VC::HandleWant::Texture, compute ? &*compute : nullptr);
        if (classified.source == VC::HandleSource::Unsupported && compute) {
            classified = VC::ClassifyHandleWord(record.bank, record.offset, VC::HandleWant::Image,
                                                &*compute);
        }
        if (classified.source == VC::HandleSource::Unsupported) {
            return {};
        }
        const std::optional<u32> word = ReadWord(stage, classified.source, classified.slot,
                                                 record.offset);
        if (!word) {
            return {};
        }
        const auto it = textures.find(VC::TextureIdOf(*word));
        return it == textures.end() ? TextureInfo{} : it->second;
    }

    std::array<Stage, BIND_STAGES> stages;
    u32 texture_pool_base{};
    u32 sampler_pool_base{};
    std::optional<VC::ComputeHandleTables> compute;
    /// Texture id -> facts of the texture it names; ids without an entry have no texture.
    std::unordered_map<u32, TextureInfo> textures;

private:
    [[nodiscard]] std::optional<u32> ReadWord(u32 stage, VC::HandleSource source, u32 slot,
                                              u32 offset) const {
        if (stage >= BIND_STAGES) {
            return std::nullopt;
        }
        const Stage& bound = stages[stage];
        switch (source) {
        case VC::HandleSource::Combined:
            return slot < bound.combined.size() ? std::optional<u32>{bound.combined[slot]}
                                                : std::nullopt;
        case VC::HandleSource::SeparateTexture:
            return slot < bound.separate_texture.size()
                       ? std::optional<u32>{bound.separate_texture[slot]}
                       : std::nullopt;
        case VC::HandleSource::SeparateSampler:
            // A sampler word holds no texture id (sampler id << 20): no texture.
            return 0u;
        case VC::HandleSource::Image:
            return slot < bound.image.size() ? std::optional<u32>{bound.image[slot]}
                                             : std::nullopt;
        case VC::HandleSource::UniformBuffer: {
            if (slot >= bound.uniform.size()) {
                return std::nullopt;
            }
            const Uniform& uniform = bound.uniform[slot];
            if (u64{offset} + sizeof(u32) > uniform.bytes.size()) {
                return std::nullopt;
            }
            if (!uniform.WordReadable(offset)) {
                return std::nullopt;
            }
            u32 value;
            std::memcpy(&value, uniform.bytes.data() + offset, sizeof(value));
            return static_cast<u32>(VC::RebaseHandleWord(value, VC::HandleWordKind::Texture,
                                                         texture_pool_base, sampler_pool_base));
        }
        case VC::HandleSource::None:
            return 0u;
        case VC::HandleSource::ComputeTexture:
            return slot < stages[STAGE_COMPUTE].combined.size()
                       ? std::optional<u32>{stages[STAGE_COMPUTE].combined[slot]}
                       : std::nullopt;
        case VC::HandleSource::ComputeImage:
            return slot < stages[STAGE_COMPUTE].image.size()
                       ? std::optional<u32>{stages[STAGE_COMPUTE].image[slot]}
                       : std::nullopt;
        default:
            return std::nullopt;
        }
    }
};

} // namespace

std::unique_ptr<TextureOracle> DrawContext::FreezeOracle(
    u32 stage_mask, const VC::ComputeHandleTables* compute) const {
    auto frozen = std::make_unique<FrozenOracle>();
    frozen->texture_pool_base = texture_pool_base;
    frozen->sampler_pool_base = sampler_pool_base;
    if (compute != nullptr) {
        frozen->compute = *compute;
    }
    VC::GuestMemory* memory = impl.guest.load(std::memory_order_acquire);
    // Texture ids that can come out of the copied words, resolved to their facts now.
    const auto note_word = [&](u32 word) {
        const u32 id = VC::TextureIdOf(word);
        if (id == 0 || frozen->textures.contains(id)) {
            return;
        }
        const std::shared_ptr<Texture> texture = TextureOfId(id);
        if (texture && (texture->storage || IsTextureBuffer(*texture))) {
            frozen->textures.emplace(id, InfoOfTexture(*texture));
        }
    };
    for (u32 stage = 0; stage < BIND_STAGES; ++stage) {
        if ((stage_mask & (1u << stage)) == 0) {
            continue;
        }
        const StageBindings& bound = stages[stage];
        FrozenOracle::Stage& out = frozen->stages[stage];
        for (size_t i = 0; i < out.combined.size(); ++i) {
            out.combined[i] = static_cast<u32>(bound.combined[i]);
            note_word(out.combined[i]);
        }
        for (size_t i = 0; i < out.separate_texture.size(); ++i) {
            out.separate_texture[i] = static_cast<u32>(bound.separate_texture[i]);
            note_word(out.separate_texture[i]);
        }
        for (size_t i = 0; i < out.image.size(); ++i) {
            out.image[i] = static_cast<u32>(bound.image[i]);
            note_word(out.image[i]);
        }
        for (size_t slot = 0; slot < out.uniform.size(); ++slot) {
            const BufferBinding& binding = bound.uniform[slot];
            FrozenOracle::Uniform& uniform = out.uniform[slot];
            const u64 size = (std::min)(binding.size, FROZEN_UNIFORM_BYTES);
            if (binding.address == 0 || size < sizeof(u32) || memory == nullptr) {
                continue;
            }
            uniform.bytes.resize(size);
            uniform.readable.assign((size + FROZEN_CHUNK - 1) / FROZEN_CHUNK, 0);
            uniform.inline_written.clear();
            bool all_readable = false;
            if (ReadSnapshotBytes(binding.address, uniform.bytes.data(), size)) {
                std::fill(uniform.readable.begin(), uniform.readable.end(), u8{1});
                all_readable = true;
            } else {
                if (memory->Read(binding.address, uniform.bytes.data(), size)) {
                    std::fill(uniform.readable.begin(), uniform.readable.end(), u8{1});
                    all_readable = true;
                } else {
                    for (u64 chunk = 0; chunk < uniform.readable.size(); ++chunk) {
                        const u64 begin = chunk * FROZEN_CHUNK;
                        const u64 length = (std::min)(FROZEN_CHUNK, size - begin);
                        uniform.readable[chunk] =
                            memory->Read(binding.address + begin, uniform.bytes.data() + begin,
                                         length)
                                ? 1
                                : 0;
                        if (uniform.readable[chunk] != 0) {
                            continue;
                        }
                        // An unaligned chunk spans two guest pages: keep the half that reads
                        // (WordReadable counts the spans of inline_written as readable).
                        constexpr u64 GUEST_PAGE = 4096;
                        const u64 split =
                            (((binding.address + begin) | (GUEST_PAGE - 1)) + 1) - binding.address;
                        if (split >= begin + length) {
                            continue;
                        }
                        for (const auto& [lo, hi] :
                             {std::pair{begin, split}, std::pair{split, begin + length}}) {
                            if (memory->Read(binding.address + lo, uniform.bytes.data() + lo,
                                             hi - lo)) {
                                uniform.inline_written.emplace_back(lo, hi);
                            }
                        }
                    }
                }
                if (host != nullptr) {
                    impl.ApplyPendingInlineWrites(
                        binding.address, uniform.bytes.data(), size, host->HostFlight().seq,
                        all_readable ? nullptr : &uniform.inline_written);
                }
            }
            // Inline updates of this submission are newer than guest memory; the last one wins.
            for (const InlineUpdate& update : inline_updates) {
                const u64 lo = (std::max)(update.address, binding.address);
                const u64 hi = (std::min)(update.address + update.size, binding.address + size);
                if (lo < hi) {
                    std::memcpy(uniform.bytes.data() + (lo - binding.address),
                                update.data + (lo - update.address), hi - lo);
                    if (!all_readable) {
                        uniform.inline_written.emplace_back(lo - binding.address,
                                                            hi - binding.address);
                    }
                }
            }
            for (u64 at = 0; at + sizeof(u32) <= size; at += sizeof(u32)) {
                u32 word;
                std::memcpy(&word, uniform.bytes.data() + at, sizeof(word));
                note_word(static_cast<u32>(VC::RebaseHandleWord(
                    word, VC::HandleWordKind::Texture, texture_pool_base, sampler_pool_base)));
            }
        }
    }
    return frozen;
}

TextureInfo DrawContext::QueryTexture(u32 stage, const VC::NvnHandleRecord& record,
                                      const VC::ComputeHandleTables* compute) const {
    VC::HandleClassification classified =
        VC::ClassifyHandleWord(record.bank, record.offset, VC::HandleWant::Texture, compute);
    if (classified.source == VC::HandleSource::Unsupported && compute != nullptr) {
        // A compute shader's image handle words sit in the image table of its driver bank.
        classified = VC::ClassifyHandleWord(record.bank, record.offset, VC::HandleWant::Image,
                                            compute);
    }
    if (classified.source == VC::HandleSource::Unsupported) {
        return {};
    }
    const VC::HandleRef ref{classified.source, classified.slot, record.bank, record.offset};
    const std::optional<u32> word = ReadHandleWord(stage, ref);
    if (!word) {
        return {};
    }
    const std::shared_ptr<Texture>& texture = TextureOfId(VC::TextureIdOf(*word));
    if (!texture || (!texture->storage && !IsTextureBuffer(*texture))) {
        return {};
    }
    return InfoOfTexture(*texture);
}

// ---------------------------------------------------------------------------------------------

namespace {

/// Fills the c0 driver bank of a graphics stage. `mesh_dims`: the workgroup counts of the last
/// 3D mesh-task draw, which the draw macro leaves in the bank (VC::DriverMeshDimsOffset).
void FillDriverBank(u32 stage, const StageBindings& bindings, std::span<const u8> push_constants,
                    const std::array<u32, 3>& mesh_dims, u32 texture_pool_base,
                    u32 sampler_pool_base, u8* out) {
    std::memset(out, 0, DRIVER_BANK_SIZE);
    if (stage < VC::ShaderStageCount) {
        const u32 base = VC::StorageBufferBase(static_cast<VC::ShaderStage>(stage));
        for (u32 slot = 0; slot < VC::StorageBufferSlots; ++slot) {
            const BufferBinding& binding = bindings.storage[slot];
            u8* entry = out + base + slot * VC::StorageBufferStride;
            if (base + (slot + 1) * VC::StorageBufferStride > DRIVER_BANK_SIZE) {
                break;
            }
            std::memcpy(entry, &binding.address, sizeof(u64));
            const u32 size = static_cast<u32>((std::min<u64>)(binding.size, 0xFFFFFFFFull));
            std::memcpy(entry + 8, &size, sizeof(u32));
        }
    }
    // Push constants of the graphics block (offset 0x10 of the driver bank).
    std::memcpy(out + 0x10, push_constants.data(),
                (std::min<size_t>)(push_constants.size(), 0x100));
    // Largest point size the shader may write (nxnative: the host writes 2047).
    const float largest_point_size = 2047.0f;
    std::memcpy(out + 0x610, &largest_point_size, sizeof(largest_point_size));
    std::memcpy(out + VC::DriverMeshDimsOffset, mesh_dims.data(), sizeof(mesh_dims));
    // The bases a shader adds to the handles it reads from memory (bindless_table.h).
    std::memcpy(out + Shader::BINDLESS_DRIVER_TEXTURE_BASE, &texture_pool_base,
                sizeof(texture_pool_base));
    std::memcpy(out + Shader::BINDLESS_DRIVER_SAMPLER_BASE, &sampler_pool_base,
                sizeof(sampler_pool_base));
}

/// Fills the c2 handle tables.
void FillHandleBank(const StageBindings& bindings, u8* out) {
    std::memset(out, 0, VC::HandleBankEnd);
    const auto put = [&](u32 offset, u64 value) {
        const u32 word = static_cast<u32>(value);
        std::memcpy(out + offset, &word, sizeof(word));
    };
    for (u32 slot = 0; slot < VC::CombinedHandleSlots; ++slot) {
        put(VC::CombinedHandleBase + slot * VC::HandleStride, bindings.combined[slot]);
    }
    for (u32 slot = 0; slot < VC::ImageHandleSlots; ++slot) {
        if (bindings.image[slot] != 0) {
            put(VC::ImageHandleBase + slot * VC::HandleStride, bindings.image[slot]);
        }
    }
    for (u32 slot = 0; slot < VC::SeparateTextureSlots; ++slot) {
        put(VC::SeparateTextureBase + slot * VC::HandleStride, bindings.separate_texture[slot]);
        put(VC::SeparateSamplerBase + slot * VC::HandleStride, bindings.separate_sampler[slot]);
    }
}

} // namespace

void DrawContext::NoteFallback(bool graphics, DescReason reason, const std::string& detail) {
    if (Debug::Tracing()) {
        dbg_notes += fmt::format("{}[{}] ", static_cast<u32>(reason), detail);
    }
    const u32 bind = graphics ? 0 : 1;
    const u32 index = static_cast<u32>(reason);
    counters.fallbacks[bind][index].fetch_add(1, std::memory_order_relaxed);
    const u32 bit = 1u << index;
    if ((fallback_logged[bind] & bit) == 0) {
        fallback_logged[bind] |= bit;
        if (DescReasonIsFallback(reason)) {
            LOG_WARNING(Render_Vulkan, "[nvn] descriptor fallback ({}): {}; a dummy is bound",
                        graphics ? "graphics" : "compute", DescReasonName(reason));
        } else {
            // What the hardware itself does with an empty or unsized binding (see
            // DescReasonIsFallback): counted, not a fault.
            LOG_INFO(Render_Vulkan, "[nvn] descriptor ({}): {}", graphics ? "graphics" : "compute",
                     DescReasonName(reason));
        }
    }
}

void DrawContext::LogUnboundStorage(bool graphics, const PipelineBase& pipe,
                                    const VC::DescriptorBinding& binding, u32 slot,
                                    const BufferBinding& resolved) {
    const u32 stage = static_cast<u32>(binding.stage);
    u64 hash = 0;
    if (stage < VC::ShaderStageCount) {
        if (const auto it = resources.program_hashes.find(stages[stage].program_serial);
            it != resources.program_hashes.end()) {
            hash = it->second[stage];
        }
    }
    const u32 source = static_cast<u32>(binding.storage_source);
    if (!unbound_logged.emplace(hash, stage, source, slot).second) {
        return;
    }
    const StageBindings& bound = stages[stage < BIND_STAGES ? stage : 0];
    std::string storage_slots;
    std::string uniform_slots;
    for (u32 i = 0; i < bound.storage.size(); ++i) {
        if (bound.storage[i].address != 0) {
            storage_slots += fmt::format("{}:{:#x} ", i, bound.storage[i].size);
        }
    }
    for (u32 i = 0; i < bound.uniform.size(); ++i) {
        if (bound.uniform[i].address != 0) {
            uniform_slots += fmt::format("{}:{:#x} ", i, bound.uniform[i].size);
        }
    }
    const VC::ComputeDriverLayout& compute = pipe.compute_layout;
    LOG_INFO(Render_Vulkan,
             "[nvn] storage buffer without a range ({}): program {:016x} stage {} source {} slot {} "
             "(binding {}, {}, entry {:#x}/{:#x}); bound storage slots [{}] uniform slots [{}]; "
             "compute layout uniform {}@{:#x} storage {}@{:#x}",
             graphics ? "graphics" : "compute", hash, stage, source, slot, binding.binding,
             binding.is_written ? "written" : "read", resolved.address, resolved.size,
             storage_slots, uniform_slots, compute.uniform_count, compute.uniform_offset,
             compute.storage_count, compute.storage_offset);
}

bool DrawContext::ViewKeyOf(const Texture& texture, const Storage& s, Shader::TextureType type,
                            bool is_image, ViewKey& key) {
    // Everything below is a function of the texture object, its storage and the binding kind,
    // none of which changes while the texture lives: remember the answer.
    const u32 kind = static_cast<u32>(type) << 1 | (is_image ? 1u : 0u);
    const size_t hash = (reinterpret_cast<uintptr_t>(&texture) >> 4) * 31 + kind;
    ViewSlot& slot = view_slots[hash % view_slots.size()];
    // A smart BC stand-in's storage can change format (Recorder::RefitStandin): it is in the key.
    if (slot.texture == &texture && slot.serial == texture.serial && slot.storage == &s &&
        slot.kind == kind && slot.storage_format == s.host.format) {
        key = slot.key;
        return slot.compatible;
    }
    const u32 level_base = texture.desc.view.set_levels ? texture.desc.view.base_level : 0;
    key = ViewKey{};
    // A BC stand-in is viewed in its storage's BC format (each storage chooses its own), with the
    // texture object's sRGB-ness.
    key.format = s.host.bc_standin ? StandinViewFormat(texture.host, s.host) : texture.host.format;
    key.aspect = s.aspect;
    if (s.host.depth && s.host.stencil) {
        const bool want_stencil = texture.desc.stencil ||
                                  (texture.desc.view.set_dsmode && texture.desc.view.dsmode == 2);
        key.aspect = want_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    key.base_level = (std::min)(level_base, s.levels - 1);
    // A view cannot address zero levels or layers: a count of 0 (SetLevels / SetLayers with no
    // range) is the full range from the base on.
    key.levels = texture.desc.view.set_levels && texture.desc.view.nlevels != 0
                     ? (std::max<u32>)(1u, (std::min<u32>)(texture.desc.view.nlevels,
                                                           s.levels - key.base_level))
                     : s.levels - key.base_level;
    const u32 layer_base = texture.desc.view.set_layers ? texture.desc.view.min_layer : 0;
    // The layers a 2D / 2D_ARRAY view can address: the array layers, or the depth
    // slices of a 3D image (a 3D storage declared as 2D by the shader).
    const bool slices = s.type == VK_IMAGE_TYPE_3D && !is_image &&
                        (s.create_flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) != 0;
    const u32 total_layers =
        s.type == VK_IMAGE_TYPE_3D ? (std::max)(s.extent.depth >> key.base_level, 1u) : s.layers;
    const u32 layer_total = texture.desc.view.set_layers && texture.desc.view.nlayers != 0
                                ? texture.desc.view.nlayers
                                : total_layers;
    key.base_layer = (std::min)(layer_base, total_layers - 1);
    const u32 layer_count = (std::max)(1u, (std::min)(layer_total, total_layers - key.base_layer));
    // A storage image view must have the identity swizzle (0 = IDENTITY). A BC4/BC5 stand-in holds
    // the decoded (L, L, L, A) as R / RG: its swizzle maps that back under the guest's.
    key.swizzle = is_image ? 0
                           : PackSwizzle(ComposeStandinSwizzle(TextureSwizzle(texture),
                                                               BcStandinOf(s.host)));
    bool compatible = true;
    switch (type) {
    case Shader::TextureType::Color1D:
        key.type = VK_IMAGE_VIEW_TYPE_1D;
        key.layers = 1;
        compatible = s.type == VK_IMAGE_TYPE_1D;
        break;
    case Shader::TextureType::ColorArray1D:
        key.type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        key.layers = layer_count;
        compatible = s.type == VK_IMAGE_TYPE_1D;
        break;
    case Shader::TextureType::Color3D:
        key.type = VK_IMAGE_VIEW_TYPE_3D;
        key.layers = 1;
        key.base_layer = 0;
        compatible = s.type == VK_IMAGE_TYPE_3D;
        break;
    case Shader::TextureType::ColorCube:
        key.type = VK_IMAGE_VIEW_TYPE_CUBE;
        key.layers = 6;
        compatible = s.type == VK_IMAGE_TYPE_2D && layer_count >= 6 &&
                     (s.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0;
        break;
    case Shader::TextureType::ColorArrayCube:
        key.type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        key.layers = layer_count / 6 * 6;
        compatible = s.type == VK_IMAGE_TYPE_2D && layer_count >= 6 &&
                     (s.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0;
        break;
    case Shader::TextureType::ColorArray2D:
        key.type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        key.layers = layer_count;
        compatible = s.type == VK_IMAGE_TYPE_2D || slices;
        break;
    case Shader::TextureType::Buffer:
        compatible = false;
        break;
    default:
        key.type = VK_IMAGE_VIEW_TYPE_2D;
        key.layers = 1;
        compatible = s.type == VK_IMAGE_TYPE_2D || slices;
        break;
    }
    if (slices && key.type != VK_IMAGE_VIEW_TYPE_3D) {
        key.levels = 1; // a 2D view of a 3D image addresses one level
    }
    if (is_image) {
        // A storage view needs the image's storage usage and a view format that can be stored:
        // the texture's own, or the UNORM format of an sRGB texture's bits (an NVN image of an
        // sRGB texture reads and writes the bits without sRGB conversion).
        const bool own_format = texture.host.storage;
        if (!own_format && texture.host.storage_view != VK_FORMAT_UNDEFINED) {
            key.format = texture.host.storage_view;
        }
        if ((s.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0 || s.astc_gpu ||
            (!own_format && texture.host.storage_view == VK_FORMAT_UNDEFINED)) {
            compatible = false;
        }
    }
    if (s.host.bc_standin && !ViewsBcStandin(key.format, s.host.format)) {
        // An sRGB view of a BC4/BC5 stand-in requires an upload in an sRGB-capable format.
        // Keep it in the plan so the upload can run before FinishImages resolves the view.
        compatible = compatible && !is_image && key.format == VK_FORMAT_UNDEFINED &&
                     s.standin_srgb_view;
    }
    slot.texture = &texture;
    slot.serial = texture.serial;
    slot.storage = &s;
    slot.storage_format = s.host.format;
    slot.kind = kind;
    slot.compatible = compatible;
    slot.key = key;
    return compatible;
}

std::shared_ptr<Texture> DrawContext::ScaledTextureFor(Texture& texture, bool graphics,
                                                       bool is_image, bool written) {
    Storage& s = *texture.storage;
    // A shader that writes a storage image writes the guest-sized image: its invocations are not
    // multiplied (the emulated GPU's blacklist), so a written image is never a twin. A graphics
    // draw samples twins only in a pass drawn at the scaled size; compute dispatches always do.
    if (written || !s.scaled || s.scale_owner != nullptr || s.samples != VK_SAMPLE_COUNT_1_BIT ||
        (graphics && !targets_scaled)) {
        return nullptr;
    }
    Storage& twin_storage = *s.scaled;
    if (twin_storage.image_failed) {
        return nullptr;
    }
    bool use_twin = false;
    if (graphics && IsBoundTarget(&twin_storage)) {
        use_twin = true; // drawn by this very pass: the pass makes the twin current
    } else if (Rescale::TwinIsCurrent(s.scale_state) && !host->HostNeedsNativeUpdate(s)) {
        use_twin = impl.EnsureImage(twin_storage);
    }
    static_cast<void>(is_image);
    return use_twin ? impl.ScaledTextureOf(texture) : nullptr;
}

bool DrawContext::BuildDescriptors(PipelineBase& pipe, DrawPlan& plan, bool graphics) {
    plan.graphics = graphics;
    const VC::DescriptorLayout& layout = pipe.descriptors;
    scratch_writes.clear();
    scratch_buffers.clear();
    scratch_images.clear();
    scratch_views.clear();
    size_t total = 0;
    for (const VC::DescriptorBinding& binding : layout.bindings) {
        total += binding.count;
    }
    scratch_writes.reserve(layout.bindings.size());
    scratch_buffers.reserve(total);
    scratch_images.reserve(total);
    scratch_views.reserve(total);

    prof_zero_size = false;
    std::array<BufferRef, BIND_STAGES> driver_bank{};
    std::array<BufferRef, BIND_STAGES> handle_bank{};
    // Untracked global memory: every pool is backed and its device address is in the table the
    // driver bank points at. The submission retains the pools the table covers (once per
    // submission and table, not per draw) and the table buffer; when the draw writes it counts as
    // a writer of all of them.
    const Context::Impl::RawGlobalTable* raw_table = nullptr;
    // The raw coverage diagnostic's miss ring (0: off), next to the table in the driver bank.
    u64 coverage_ring = 0;
    // Raw-store aliases to the mirrors or imported storage backing.
    const Context::Impl::RawWriteThroughTable* alias_table = nullptr;
    RawAliasHeader alias_header{};
    u64 alias_generation = 0;
    // Raw-global pointers bypass storage descriptors. Under TRY_STORAGE_VRAM, mirror written
    // imported pools before constructing the raw table so both paths receive the same address.
    if (layout.uses_raw_global_memory && impl.try_storage_vram) {
        for (const VC::DescriptorBinding& binding : layout.bindings) {
            if (binding.kind != VC::DescriptorKind::StorageBuffer || !binding.is_written) {
                continue;
            }
            for (u32 i = 0; i < binding.count; ++i) {
                const BufferBinding resolved = ResolveStorageBinding(binding, i);
                Pool* pool = nullptr;
                u64 pool_offset = 0;
                if (resolved.address != 0 && resolved.size != 0 &&
                    impl.ResolveBuffer(resolved.address, resolved.size, pool, pool_offset) &&
                    pool->backing == Pool::Backing::Imported &&
                    pool->size <= impl.try_storage_vram_max_bytes) {
                    impl.VramMirror(*pool, true);
                }
            }
        }
    }
    if (layout.uses_raw_global_memory) {
        raw_table = &impl.RawGlobalMemoryTable();
        impl.stats.raw_global_draws.fetch_add(1, std::memory_order_relaxed);
        InFlight& raw_flight = host->HostFlight();
        if (raw_noted_seq != raw_flight.seq || raw_noted_generation != raw_table->generation) {
            raw_noted_seq = raw_flight.seq;
            raw_noted_generation = raw_table->generation;
            for (Pool* const pool : raw_table->pools) {
                impl.NoteUse(*pool, raw_flight.seq);
            }
        }
        if (raw_table->device && raw_flight.raw_table_kept != raw_table->device.get()) {
            raw_flight.raw_table_kept = raw_table->device.get();
            raw_flight.keepalive.push_back(raw_table->device);
        }
        plan.raw_global_writes = layout.stores_raw_global_memory;
        plan.raw_global = true;
        if (layout.raw_stores_written_through) {
            alias_table = &impl.RawStoreAliasTable();
            plan.raw_stores_written_through = true;
            alias_header = {alias_table->device.get(), alias_table->low, alias_table->high};
            alias_generation = alias_table->generation + 1;
            // The mirror buffers the stores reach live until the submission completes.
            if (alias_noted_seq != raw_flight.seq ||
                alias_noted_generation != alias_table->generation) {
                alias_noted_seq = raw_flight.seq;
                alias_noted_generation = alias_table->generation;
                for (Pool* const pool : alias_table->pools) {
                    impl.NoteUse(*pool, raw_flight.seq);
                }
                if (alias_table->device) {
                    raw_flight.keepalive.push_back(alias_table->device);
                }
            }
        }
        if (VideoCommon::RawCoverageEnabled()) {
            coverage_ring = impl.RawCoverageRingFor(raw_flight.seq);
            if (coverage_ring != 0) {
                host->HostNeedHostReadBarrier();
            }
        }
    }
    VkImageView dummy_2d = VK_NULL_HANDLE;
    const VkSampler fallback_sampler = *resources.fallback_sampler;

    const auto note = [&](DescReason reason) { NoteFallback(graphics, reason); };
    // A buffer binding: resolves the range, records the plan, falls back to the dummy buffer.
    // `sink`: the stage and table slot of a storage binding, so that a written binding with no
    // buffer behind it gets its own sink slice (resources.SinkSlice) instead of the dummy.
    struct SinkSite {
        u32 stage;
        u32 slot;
    };
    const auto buffer_info = [&](const BufferBinding& binding, bool write, bool uniform,
                                 const SinkSite* sink = nullptr,
                                 CachedBuffer* slot_cache = nullptr) {
        const auto fallback = [&]() {
            if (write && !uniform && sink != nullptr) {
                return VkDescriptorBufferInfo{
                    resources.SinkBuffer(),
                    resources.SinkSlice(ProfStageHash(sink->stage), sink->stage, sink->slot),
                    DrawResources::SINK_SLICE_SIZE};
            }
            return VkDescriptorBufferInfo{resources.DummyBuffer(), 0, 4096};
        };
        u64 bound_size = binding.size;
        bool unsized = false;
        if (binding.address == 0 || (bound_size == 0 && uniform)) {
            const bool zero_size = binding.address != 0;
            if (uniform && sink != nullptr) {
                // Which draw references an empty uniform slot (once per program and slot).
                const u64 hash = ProfStageHash(sink->stage);
                if (unbound_logged.emplace(hash, sink->stage, zero_size ? 101u : 100u, sink->slot)
                        .second) {
                    LOG_INFO(Render_Vulkan,
                             "[nvn] uniform buffer slot {} of program {:016x} stage {} is bound with "
                             "{}: it reads zero",
                             sink->slot, hash, sink->stage,
                             zero_size ? "an address and size 0" : "no address");
                }
            }
            // A constant buffer without an address or with size 0: every read is out of bounds
            // and returns zero.
            note(uniform ? (zero_size ? DescReason::UniformZeroSize : DescReason::UniformUnbound)
                         : DescReason::StorageUnbound);
            return fallback();
        }
        const u64 current_seq = host->HostFlight().seq;
        u64 size = 0;
        BufferRef ref;
        bool resolved = false;
        if (slot_cache != nullptr && bound_size != 0 && pass_open) {
            const u64 pool_gen = impl.pool_generation.load(std::memory_order_acquire);
            if (slot_cache->address == binding.address && slot_cache->binding_size == bound_size &&
                slot_cache->pool_gen == pool_gen &&
                slot_cache->mapping_gen == impl.mkw_mapping_generation &&
                slot_cache->alias_gen == impl.ro_alias_generation &&
                slot_cache->ref.pool != nullptr &&
                !slot_cache->ref.pool->pending.load(std::memory_order_acquire) &&
                !slot_cache->ref.pool->upload_pending &&
                (slot_cache->ref.pool->backing == Pool::Backing::Imported ||
                 slot_cache->ref.pool->backing == Pool::Backing::DeviceLocal)) {
                ref = slot_cache->ref;
                size = ref.range;
                resolved = true;
                impl.NoteUse(*ref.pool, current_seq);
            }
        }
        if (!resolved) {
            if (bound_size == 0) {
                if (Debug::Get().no_zero_size) {
                    // Experiment (EDEN_NX2_NVN_NO_ZERO_SIZE): the dummy buffer, as before these
                    // entries bound the rest of their pool.
                    note(DescReason::StorageZeroSize);
                    return fallback();
                }
                // A storage buffer bound with an address and size 0 is the rest of its pool: the
                // hardware loads and stores through the address and never looks at the size
                // (nxnative binds the address and counts the write as the rest of the buffer,
                // nvn_mtl.m sbw_len). Counted (StorageZeroSize) as a handled case.
                Pool* pool = nullptr;
                u64 pool_offset = 0;
                if (!host->HostResolvePoolRange(binding.address, 1, pool, pool_offset, "buffer")) {
                    note(DescReason::StorageUnresolved);
                    return fallback();
                }
                const u64 rest =
                    (std::min<u64>)(pool->size - pool_offset, ZERO_SIZE_RANGE_MAX) & ~u64{3};
                note(DescReason::StorageZeroSize);
                if (rest == 0) {
                    return fallback();
                }
                bound_size = rest;
                prof_zero_size = true;
                unsized = true;
            }
            size = uniform ? (std::min)(bound_size, MAX_UNIFORM_RANGE) : bound_size;
            if (uniform) {
                // The driver binds uniform buffers in whole 256-byte units
                // (nvnCommandBufferBindUniformBuffer rounds the size up), so the range may run up
                // to 255 bytes past the end of its pool (or mapped virtual range, both 4 KiB
                // granular): bind the part up to that boundary. Only a range that ends just past a
                // 4 KiB boundary pays the extra (silent, cached) lookups.
                const u64 end = binding.address + size;
                const u64 boundary = end & ~u64{0xFFF};
                Pool* pool = nullptr;
                u64 pool_offset = 0;
                if (boundary > binding.address && end != boundary && end - boundary < 0x100 &&
                    !impl.ResolveBuffer(binding.address, size, pool, pool_offset) &&
                    impl.ResolveBuffer(binding.address, boundary - binding.address, pool,
                                       pool_offset)) {
                    size = boundary - binding.address;
                }
            }
            resolved = ResolveBuffer(binding.address, size, write, ref, ShadowPolicy::Defer);
            if (resolved && slot_cache != nullptr && !unsized && ref.pool != nullptr &&
                !ref.pool->pending.load(std::memory_order_acquire) && !ref.pool->upload_pending &&
                (ref.pool->backing == Pool::Backing::Imported ||
                 ref.pool->backing == Pool::Backing::DeviceLocal)) {
                *slot_cache = {
                    .ref = ref,
                    .pool_gen = impl.pool_generation.load(std::memory_order_acquire),
                    .mapping_gen = impl.mkw_mapping_generation,
                    .alias_gen = impl.ro_alias_generation,
                    .address = binding.address,
                    .binding_size = bound_size,
                };
            }
        }
        // EDEN_NX2_NVN_UBO_SNAPSHOT: the bytes as of the submit call (nvn_uniform_snapshot.h).
        const size_t snapshot_range = uniform && !write && usnap.Active()
                                          ? usnap.Find(binding.address, size)
                                          : UniformSnapshotView::npos;
        // Inline updates that cover the range are applied on top of the memory.
        bool overlaps = false;
        if (!unsized && !inline_updates.empty()) {
            for (const InlineUpdate& update : inline_updates) {
                if (update.address < binding.address + size &&
                    binding.address < update.address + update.size) {
                    overlaps = true;
                    break;
                }
            }
        }
        const bool misaligned = resolved && uniform && (ref.offset % 256) != 0;
        const bool may_need_gpu_source =
            resolved && !write && snapshot_range == UniformSnapshotView::npos &&
            (misaligned || (overlaps && pass_open));
        const bool needs_gpu_source =
            may_need_gpu_source &&
            (ref.pool->backing == Pool::Backing::DeviceLocal ||
             HasOverlappingShadow(*ref.pool, ref.pool_offset, ref.pool_offset + size) ||
             impl.NeedsGpuFlushForUniform(binding.address, binding.address + size, current_seq));
        if (misaligned && needs_gpu_source) {
            const BufferRef copy = AllocArena(size, 256, nullptr);
            if (copy.buffer) {
                if (overlaps) {
                    plan.flush_inline = true;
                }
                plan.UseRange(ref.pool, ref.pool_offset, size, false, DrawPlan::RangeUse::NO_INFO,
                              false, false);
                plan.uniform_copies.push_back(
                    {ref.buffer, ref.offset, copy.buffer, copy.offset, size});
                if (cs_mode != CostSplit::Mode::Count) [[unlikely]] {
                    CostSplitNote(CostSplit::Data::Uniform, ref, size, true);
                }
                return VkDescriptorBufferInfo{copy.buffer, copy.offset, size};
            }
        }
        bool flush_inline_to_pool = false;
        if (overlaps && !misaligned && resolved && !write &&
            snapshot_range == UniformSnapshotView::npos) {
            if (!pass_open || needs_gpu_source) {
                flush_inline_to_pool = true;
                plan.flush_inline = true;
            }
        }
        const bool use_private_copy = (overlaps && !flush_inline_to_pool) || misaligned;
        if (cs_mode != CostSplit::Mode::Count) [[unlikely]] {
            CostSplitNote(uniform ? CostSplit::Data::Uniform : CostSplit::Data::Storage, ref, size,
                          use_private_copy);
        }
        if (use_private_copy) {
            if (write) {
                host->HostUnsupported("Draw", "storage buffer under an inline uniform update");
            }
            // A private copy: memory (when readable) with the updates applied in order.
            u8* pointer = nullptr;
            const BufferRef copy = AllocArena(size, 256, &pointer);
            if (copy.buffer) {
                if (snapshot_range != UniformSnapshotView::npos) {
                    std::memcpy(pointer, usnap.Bytes(snapshot_range, binding.address), size);
                } else {
                    if (const u8* source = impl.GuestPointer(binding.address, size)) {
                        std::memcpy(pointer, source, size);
                    } else if (VC::GuestMemory* memory = impl.guest.load(std::memory_order_acquire);
                               !memory || !memory->Read(binding.address, pointer, size)) {
                        std::memset(pointer, 0, size);
                    }
                    impl.ApplyPendingInlineWrites(binding.address, pointer, size, current_seq);
                }
                for (const InlineUpdate& update : inline_updates) {
                    const u64 begin = (std::max)(update.address, binding.address);
                    const u64 end = (std::min)(update.address + update.size, binding.address + size);
                    if (begin < end) {
                        std::memcpy(pointer + (begin - binding.address),
                                    update.data + (begin - update.address), end - begin);
                    }
                }
                return VkDescriptorBufferInfo{copy.buffer, copy.offset, size};
            }
        }
        if (snapshot_range != UniformSnapshotView::npos) {
            if (const auto info = SnapshotUniform(snapshot_range, binding.address, size)) {
                return *info;
            }
        }
        if (!resolved) {
            note(uniform ? DescReason::UniformUnresolved : DescReason::StorageUnresolved);
            return fallback();
        }
        // A written storage range of an imported pool lives in host memory: shader stores and
        // atomics would cross PCIe (15-19 ns per atomic measured). A small one is bound through a
        // device-local shadow for as long as nothing else uses the bytes (nvn_draw_shadow.cpp);
        // with pool mirrors every storage range of an imported pool is a candidate for the pool's
        // VRAM mirror (read-only ones too: their reads cross PCIe as well).
        const Debug::Config& debug = Debug::Get();
        const bool storage_candidate =
            !uniform && !unsized && (!layout.uses_raw_global_memory || impl.try_storage_vram) &&
            ref.pool->backing == Pool::Backing::Imported;
        // A read-only uniform range of an imported pool: served from the pool's mirror
        // (EDEN_NX2_TRY_UBO_MIRROR), which its first use creates.
        const bool ubo_candidate = uniform && !write && impl.ubo_mirror && ref.pool != nullptr &&
                                   ref.pool->backing == Pool::Backing::Imported;
        const bool shadowable =
            ubo_candidate ||
            (storage_candidate && (impl.pool_mirror_mode != 0
                                       ? true
                                       : write && debug.shadow_storage && size <= debug.shadow_max_bytes));
        plan.shadow_candidates |= shadowable;
        plan.UseRange(ref.pool, ref.pool_offset, size, write,
                      static_cast<u32>(scratch_buffers.size()), shadowable, ubo_candidate);
        if (cs_mode != CostSplit::Mode::Count && !write) [[unlikely]] {
            CostSplitToRing(uniform ? CostSplit::Data::Uniform : CostSplit::Data::Storage, ref,
                            binding.address, size);
        }
        return VkDescriptorBufferInfo{ref.buffer, ref.offset, size};
    };

    // Resolution scaling: a texture and an image descriptor each own a bit of the rescaling words
    // (the k-th one of the pipeline's stages in order, as the shader's `IsTextureScaled` counts).
    const bool rescaling = impl.RescaleFactor().Active();
    u32 texture_index = 0;
    u32 image_index = 0;
    for (const VC::DescriptorBinding& binding : layout.bindings) {
        const u32 stage = static_cast<u32>(binding.stage);
        const u32 count = binding.count;
        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.binding,
            .dstArrayElement = 0,
            .descriptorCount = count,
            .descriptorType = static_cast<VkDescriptorType>(binding.kind),
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        switch (binding.kind) {
        case VC::DescriptorKind::UniformBuffer: {
            const size_t first = scratch_buffers.size();
            for (u32 i = 0; i < count; ++i) {
                switch (binding.role) {
                case VC::DescriptorRole::DriverBank: {
                    BufferRef& ref = driver_bank[stage];
                    if (!ref.buffer) {
                        if (binding.stage == VC::ShaderStage::Compute) {
                            u8* pointer = nullptr;
                            ref = AllocArena(DRIVER_BANK_SIZE, 256, &pointer);
                            FillComputeBank(pipe.compute_layout, pointer);
                            if (raw_table != nullptr) {
                                WriteRawGlobalTable(raw_table->device.get(), coverage_ring,
                                                    std::span<u8>{pointer, DRIVER_BANK_SIZE},
                                                    alias_header);
                            }
                        } else {
                            // The arena copy of the last build serves every draw of the
                            // submission until what the bank is made of changes.
                            BankCache& cache = driver_banks[stage];
                            const u64 seq = host->HostFlight().seq;
                            const u64 raw_generation = raw_table ? raw_table->generation + 1 : 0;
                            const u64 pool_bases = u64{texture_pool_base} << 32 | sampler_pool_base;
                            const void* raw_device = raw_table ? raw_table->device.get() : nullptr;
                            if (cache.seq != seq || cache.a != storage_gen[stage] ||
                                cache.b != push_gen[0] || cache.c != raw_generation ||
                                cache.d != raw_device || cache.e != mesh_dims_gen ||
                                cache.f != coverage_ring || cache.g != alias_generation ||
                                cache.h != pool_bases) {
                                u8* pointer = nullptr;
                                cache.ref = AllocArena(DRIVER_BANK_SIZE, 256, &pointer);
                                ++counters.driver_bank_uploads;
                                FillDriverBank(stage, stages[stage], push_constants[0], mesh_dims,
                                               texture_pool_base, sampler_pool_base, pointer);
                                if (raw_table != nullptr) {
                                    WriteRawGlobalTable(raw_table->device.get(), coverage_ring,
                                                        std::span<u8>{pointer, DRIVER_BANK_SIZE},
                                                        alias_header);
                                }
                                cache.seq = seq;
                                cache.a = storage_gen[stage];
                                cache.b = push_gen[0];
                                cache.c = raw_generation;
                                cache.d = raw_device;
                                cache.e = mesh_dims_gen;
                                cache.f = coverage_ring;
                                cache.g = alias_generation;
                                cache.h = pool_bases;
                            }
                            ref = cache.ref;
                        }
                    }
                    scratch_buffers.push_back({ref.buffer, ref.offset, DRIVER_BANK_SIZE});
                    break;
                }
                case VC::DescriptorRole::HandleTable: {
                    BufferRef& ref = handle_bank[stage];
                    if (!ref.buffer) {
                        BankCache& cache = handle_banks[stage];
                        const u64 seq = host->HostFlight().seq;
                        if (cache.seq != seq || cache.a != handle_gen[stage]) {
                            u8* pointer = nullptr;
                            cache.ref = AllocArena(VC::HandleBankEnd, 256, &pointer);
                            ++counters.handle_bank_uploads;
                            FillHandleBank(stages[stage], pointer);
                            cache.seq = seq;
                            cache.a = handle_gen[stage];
                        }
                        ref = cache.ref;
                    }
                    scratch_buffers.push_back({ref.buffer, ref.offset, VC::HandleBankEnd});
                    break;
                }
                case VC::DescriptorRole::ProgramConstants:
                    scratch_buffers.push_back(
                        {resources.ProgramConstants(pipe, stage), 0, VK_WHOLE_SIZE});
                    break;
                default: {
                    const u32 slot = binding.slot + i;
                    const size_t block = binding.stage == VC::ShaderStage::Compute ? 1 : 0;
                    if (slot + VC::FirstUniformBank == VC::PushConstantBank &&
                        !push_constants[block].empty() &&
                        (slot >= stages[stage].uniform.size() ||
                         stages[stage].uniform[slot].size == 0)) {
                        // The push constants, where the shader reads them when the game bound
                        // no uniform buffer there (as the emulated path serves root table 0).
                        BankCache& cache = push_banks[block];
                        const u64 seq = host->HostFlight().seq;
                        if (cache.seq != seq || cache.a != push_gen[block]) {
                            u8* pointer = nullptr;
                            cache.ref = AllocArena(VC::PushConstantBankSize, 256, &pointer);
                            std::memset(pointer, 0, VC::PushConstantBankSize);
                            std::memcpy(pointer, push_constants[block].data(),
                                        (std::min<size_t>)(push_constants[block].size(),
                                                           VC::PushConstantBankSize));
                            cache.seq = seq;
                            cache.a = push_gen[block];
                        }
                        scratch_buffers.push_back(
                            {cache.ref.buffer, cache.ref.offset, VC::PushConstantBankSize});
                        break;
                    }
                    const SinkSite site{stage, slot};
                    CachedBuffer* const slot_cache =
                        slot < uniform_cache[stage].size() ? &uniform_cache[stage][slot] : nullptr;
                    scratch_buffers.push_back(buffer_info(
                        slot < stages[stage].uniform.size() ? stages[stage].uniform[slot]
                                                            : BufferBinding{},
                        false, true, &site, slot_cache));
                    break;
                }
                }
            }
            write.pBufferInfo = &scratch_buffers[first];
            break;
        }
        case VC::DescriptorKind::StorageBuffer: {
            const size_t first = scratch_buffers.size();
            for (u32 i = 0; i < count; ++i) {
                const u32 slot = binding.slot + i;
                const BufferBinding resolved = ResolveStorageBinding(binding, i);
                if (resolved.address == 0 || resolved.size == 0) {
                    LogUnboundStorage(graphics, pipe, binding, slot, resolved);
                }
                CachedBuffer* slot_cache = nullptr;
                if (binding.storage_source == VC::StorageSource::DriverTable &&
                    stage < BIND_STAGES && slot < storage_cache[stage].size()) {
                    slot_cache = &storage_cache[stage][slot];
                } else if (binding.storage_source == VC::StorageSource::ComputeStorageEntry &&
                           slot < storage_cache[STAGE_COMPUTE].size()) {
                    slot_cache = &storage_cache[STAGE_COMPUTE][slot];
                }
                const SinkSite site{stage, slot};
                scratch_buffers.push_back(
                    buffer_info(resolved, binding.is_written, false, &site, slot_cache));
            }
            write.pBufferInfo = &scratch_buffers[first];
            break;
        }
        case VC::DescriptorKind::CombinedImageSampler:
        case VC::DescriptorKind::StorageImage: {
            const bool is_image = binding.kind == VC::DescriptorKind::StorageImage;
            const size_t first = scratch_images.size();
            bool binding_scaled = false;
            for (u32 i = 0; i < count; ++i) {
                const VC::HandleRef texture_ref = VC::HandleElement(binding.texture, i);
                const VC::HandleRef sampler_ref = VC::HandleElement(binding.sampler, i);
                const u32 texture_word =
                    ReadBindingHandleWord(layout, binding, texture_ref,
                                          is_image ? VC::HandleWordKind::Image
                                                   : VC::HandleWordKind::Texture)
                        .value_or(0);
                const u32 sampler_word =
                    sampler_ref.source != VC::HandleSource::None
                        ? ReadHandleWord(stage, sampler_ref, VC::HandleWordKind::SeparateSampler)
                              .value_or(0)
                        : 0;
                const std::shared_ptr<Texture>& game_texture =
                    TextureOfId(VC::TextureIdOf(texture_word));
                // Resolution scaling: a texture the setting scaled is read at the scaled size when
                // this draw (or dispatch) can (ScaledTextureFor); `texture` is then its twin.
                std::shared_ptr<Texture> twin;
                if (rescaling && game_texture && game_texture->storage &&
                    !game_texture->storage->image_failed) {
                    twin = ScaledTextureFor(*game_texture, graphics, is_image,
                                            is_image && binding.is_written);
                }
                const std::shared_ptr<Texture>& texture = twin ? twin : game_texture;
                // The image view is made in phase B (FinishImages): the image of a storage does
                // not exist until its first use, so this only resolves what to bind.
                VkDescriptorImageInfo info{fallback_sampler, VK_NULL_HANDLE,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                bool bound = false;
                bool view_anisotropy = false;
                if (texture && texture->storage && !texture->storage->image_failed) {
                    Storage& s = *texture->storage;
                    ViewKey key{};
                    const bool compatible = ViewKeyOf(*texture, s, binding.type, is_image, key);
                    view_anisotropy =
                        compatible && ViewSupportsAnisotropy(key.format, key.type, key.levels);
                    if (compatible) {
                        const bool depth = s.host.depth || s.host.stencil;
                        // The scratch copy of a depth target has no attachment usage: it is
                        // sampled in SHADER_READ_ONLY (DEPTH_STENCIL_READ_ONLY needs the usage).
                        VkImageLayout wanted =
                            is_image ? VK_IMAGE_LAYOUT_GENERAL
                            : depth && (s.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0
                                ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        // A colour image in GENERAL (a shader wrote it as a storage image) is
                        // sampled where it is: GENERAL allows every device access (Vulkan spec,
                        // "Image Layouts"), so a write-then-sample chain (compute mip and blur
                        // passes) does not flip its layout, with a barrier and a pass break, at
                        // every use. NVN has no layouts; the game's barriers order the chain, and
                        // PrepareOutsidePass keeps the pending barrier for them.
                        if (!is_image && !depth && s.current_layout == VK_IMAGE_LAYOUT_GENERAL) {
                            wanted = VK_IMAGE_LAYOUT_GENERAL;
                            ++counters.sampled_in_general;
                        }
                        Texture* view_texture = texture.get();
                        bool scratch = false;
                        bound = true;
                        if (graphics && IsBoundTarget(&s)) {
                            // A render target of this very draw: the pass writes it, so sample a
                            // copy of it made before the pass (copy-on-read).
                            bound = false;
                            {
                                const bool real = TargetOverlapsView(s, key);
                                ++(real ? counters.feedback_overlapping : counters.feedback_disjoint);
                                if (Debug::Get().feedback_log && feedback_logged++ < 40) {
                                    LOG_INFO(Render_Vulkan,
                                             "[nvn] feedback: {}x{}x{} levels {} layers {} format {} samples {}; "
                                             "sampled level {}+{} layer {}+{} type {}; overlap {}",
                                             s.extent.width, s.extent.height, s.extent.depth, s.levels,
                                             s.layers, static_cast<int>(s.host.format),
                                             static_cast<int>(s.samples), key.base_level, key.levels,
                                             key.base_layer, key.layers, static_cast<int>(key.type), real);
                                    for (u32 t = 0; t < color_target_count; ++t) {
                                        if (color_targets[t].texture &&
                                            color_targets[t].texture->storage.get() == &s) {
                                            LOG_INFO(Render_Vulkan,
                                                     "[nvn]   target {}: level {} layer {} set_layers {} nlayers {}",
                                                     t, color_targets[t].ref.texture.level,
                                                     color_targets[t].ref.texture.layer,
                                                     color_targets[t].ref.view.set_layers,
                                                     color_targets[t].ref.view.nlayers);
                                        }
                                    }
                                }
                            }
                            if (is_image || s.samples != VK_SAMPLE_COUNT_1_BIT) {
                                ++counters.feedback_unsupported;
                                host->HostUnsupported(
                                    "Draw", fmt::format("render target {:#x} is also bound as a {} in "
                                                        "the same draw",
                                                        texture->serial,
                                                        is_image ? "storage image"
                                                                 : "multisampled texture"));
                            } else {
                                DrawPlan::FeedbackCopy* entry = nullptr;
                                for (DrawPlan::FeedbackCopy& existing : plan.feedback) {
                                    if (existing.source == texture->storage &&
                                        existing.retype_layers == 0 && !existing.retype_row) {
                                        entry = &existing;
                                        break;
                                    }
                                }
                                if (!entry) {
                                    DrawResources::Scratch* made = resources.GetScratch(
                                        s, static_cast<u32>(plan.feedback.size()));
                                    if (made) {
                                        plan.feedback.push_back({texture->storage, made->storage,
                                                                 made->texture, key.base_level,
                                                                 key.levels, key.base_layer,
                                                                 key.layers});
                                        entry = &plan.feedback.back();
                                    }
                                } else {
                                    const u32 level_end = (std::max)(entry->level + entry->levels,
                                                                     key.base_level + key.levels);
                                    const u32 layer_end = (std::max)(entry->layer + entry->layers,
                                                                     key.base_layer + key.layers);
                                    entry->level = (std::min)(entry->level, key.base_level);
                                    entry->layer = (std::min)(entry->layer, key.base_layer);
                                    entry->levels = level_end - entry->level;
                                    entry->layers = layer_end - entry->layer;
                                }
                                if (entry) {
                                    view_texture = entry->scratch_texture.get();
                                    scratch = true;
                                    bound = true;
                                    wanted = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                    plan.UseImage(entry->scratch, wanted);
                                } else {
                                    ++counters.feedback_unsupported;
                                    host->HostUnsupported(
                                        "Draw", "no scratch image for a render target that is also "
                                                "sampled");
                                }
                            }
                        } else {
                            plan.UseImage(texture->storage, wanted, !is_image,
                                          is_image && binding.is_written);
                        }
                        if (bound) {
                            if (twin) {
                                binding_scaled = true;
                                impl.NoteUse(*game_texture, host->HostFlight().seq);
                                ++impl.stats.rescale_twin_samples;
                            }
                            info.imageLayout = wanted;
                            plan.bindings.push_back({static_cast<u32>(scratch_images.size()),
                                                     view_texture, key,
                                                     DummyViewType(binding.type), is_image,
                                                     scratch});
                        }
                    } else {
                        // A 2D placeholder where the shader samples a 3D or cube texture: sample a
                        // one-texel image of the shader's type holding its texel (what the
                        // hardware returns for a single texel), refreshed by a copy per draw. One
                        // sampled as 1D (Hyrule Warriors' vertex shaders on its 8x8 default
                        // texture): a 1D image holding its first row.
                        const bool row = binding.type == Shader::TextureType::Color1D &&
                                         impl.device.IsKhrMaintenance5Supported();
                        const bool typed_target =
                            row || binding.type == Shader::TextureType::Color3D ||
                            binding.type == Shader::TextureType::ColorCube ||
                            binding.type == Shader::TextureType::ColorArrayCube;
                        if (typed_target && !is_image && s.type == VK_IMAGE_TYPE_2D &&
                            s.levels == 1 && s.layers == 1 &&
                            (row || binding.type == Shader::TextureType::Color3D ||
                             s.extent.width == s.extent.height) &&
                            s.layers == 1 && s.samples == VK_SAMPLE_COUNT_1_BIT &&
                            !s.host.depth && !s.host.stencil) {
                            const bool cube = !row && binding.type != Shader::TextureType::Color3D;
                            const auto target = row    ? DrawResources::RetypeTarget::Row
                                                : cube ? DrawResources::RetypeTarget::Cube
                                                       : DrawResources::RetypeTarget::Volume;
                            // Distinct sources of one draw get distinct scratch images: the slot
                            // counts the distinct copy sources before this one's first copy.
                            u32 prior_sources = 0;
                            for (size_t e = 0; e < plan.feedback.size(); ++e) {
                                const auto& entry_source = plan.feedback[e].source;
                                if (entry_source == texture->storage) {
                                    break;
                                }
                                bool seen = false;
                                for (size_t f = 0; f < e; ++f) {
                                    seen |= plan.feedback[f].source == entry_source;
                                }
                                prior_sources += seen ? 0u : 1u;
                            }
                            DrawResources::Scratch* made = resources.GetRetypedScratch(
                                s, 200u + static_cast<u32>(binding.type) + 64u * prior_sources,
                                target);
                            if (made) {
                                if (placeholder_logged
                                        .insert(texture->serial << 8 ^ static_cast<u64>(binding.type))
                                        .second) {
                                    LOG_INFO(Render_Vulkan,
                                             "[nvn] texture {:#x} ({}x{} 2D, format {:#x}) is "
                                             "sampled as {} by program {:016x} stage {}: bound as "
                                             "{}",
                                             texture->serial, s.extent.width, s.extent.height,
                                             s.desc.format,
                                             row ? "1D" : cube ? "a cube" : "a 3D texture",
                                             ProfStageHash(stage), stage,
                                             row    ? "a 1D texture holding its first row"
                                             : cube ? "a one-texel cube holding its texel"
                                                    : "a one-texel 3D texture holding its texel");
                                }
                                const u32 faces = cube ? 6u : 1u;
                                bool have_copy = false;
                                for (const DrawPlan::FeedbackCopy& existing : plan.feedback) {
                                    have_copy |= existing.source == texture->storage &&
                                                 existing.scratch == made->storage;
                                }
                                if (!have_copy) {
                                    plan.feedback.push_back({texture->storage, made->storage,
                                                             made->texture, 0, 1, 0, 1,
                                                             row ? 0u : faces, row});
                                }
                                key.type = row     ? VK_IMAGE_VIEW_TYPE_1D
                                           : !cube ? VK_IMAGE_VIEW_TYPE_3D
                                           : binding.type == Shader::TextureType::ColorCube
                                               ? VK_IMAGE_VIEW_TYPE_CUBE
                                               : VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
                                key.base_level = 0;
                                key.levels = 1;
                                key.base_layer = 0;
                                key.layers = faces;
                                const VkImageLayout typed_layout =
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                plan.UseImage(made->storage, typed_layout);
                                info.imageLayout = typed_layout;
                                plan.bindings.push_back({static_cast<u32>(scratch_images.size()),
                                                         made->texture.get(), key,
                                                         DummyViewType(binding.type), is_image,
                                                         true});
                                bound = true;
                            }
                        }
                        if (!bound &&
                            incompatible_logged.insert(texture->serial << 8 ^ static_cast<u64>(binding.type)).second) {
                            // Which ids of the pool name this texture (a placeholder is registered
                            // at many) and what it is.
                            const std::vector<u64>& table = submission->texture_pool_snapshot.Values();
                            size_t registered = 0;
                            u32 first_id = 0;
                            u32 last_id = 0;
                            for (u32 id = 0; id < table.size(); ++id) {
                                if (table[id] == texture->serial) {
                                    first_id = registered == 0 ? id : first_id;
                                    last_id = id;
                                    ++registered;
                                }
                            }
                            LOG_WARNING(Render_Vulkan,
                                        "[nvn] texture {:#x} is {}x{}x{} format {} levels {}; registered at "
                                        "{} ids of the pool ({}..{}); the shader samples it as type {}",
                                        texture->serial, s.extent.width, s.extent.height, s.extent.depth,
                                        static_cast<int>(s.host.format), s.levels, registered, first_id, last_id,
                                        static_cast<int>(binding.type));
                        }
                        if (!bound) {
                        host->HostUnsupported(
                            "Draw",
                            fmt::format("texture {:#x} (target {}{}, format {:#x}{}, image type {}) cannot be bound as "
                                        "{} of shader type {}: stage {} handle word {:#x} <- source {} slot {} bank {} "
                                        "offset {:#x}, texture id {} (pool base {})",
                                        texture->serial, texture->desc.target,
                                        texture->desc.view.set_target
                                            ? fmt::format(", view target {}", texture->desc.view.target)
                                            : std::string{},
                                        texture->desc.format,
                                        texture->desc.view.set_format
                                            ? fmt::format(", view format {:#x}", texture->desc.view.format)
                                            : std::string{},
                                        static_cast<int>(s.type), is_image ? "a storage image" : "a texture",
                                        static_cast<int>(binding.type), stage,
                                        texture_word, static_cast<int>(texture_ref.source),
                                        texture_ref.slot, texture_ref.bank, texture_ref.byte_offset,
                                        VC::TextureIdOf(texture_word), texture_pool_base));
                        }
                    }
                }
                if (!bound) {
                    ++counters.descriptors_unsupported;
                    const std::string detail =
                        Debug::Tracing()
                            ? fmt::format("stage{} word={:#x} tex={:x} target={} shader_type={}", stage,
                                          texture_word, texture ? texture->serial : 0,
                                          texture ? texture->desc.target : -1,
                                          static_cast<int>(binding.type))
                            : std::string{};
                    if (!texture) {
                        NoteFallback(graphics, DescReason::TextureNoHandle, detail);
                    } else if (!texture->storage || texture->storage->image_failed) {
                        NoteFallback(graphics, DescReason::TextureNoStorage, detail);
                    } else {
                        NoteFallback(graphics, DescReason::TextureIncompatible, detail);
                    }
                    info.imageView = resources.DummyView(DummyViewType(binding.type));
                    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    if (is_image) {
                        // A storage image needs GENERAL: no dummy can satisfy it; skip the write.
                        info.imageView = VK_NULL_HANDLE;
                    }
                }
                u64 noted_sampler = 0;
                if (!is_image) {
                    const u32 sampler_id = ((sampler_ref.source != VC::HandleSource::None
                                                 ? (texture_word | sampler_word)
                                                 : texture_word)) >> 20;
                    const u64 serial = SamplerSerialOfId(sampler_id);
                    noted_sampler = serial;
                    info.sampler = serial != 0 ? SamplerOf(sampler_id, serial, view_anisotropy)
                                               : fallback_sampler;
                } else {
                    info.sampler = VK_NULL_HANDLE;
                }
                if (diag_bindings && !plan.bindings.empty() &&
                    plan.bindings.back().index == scratch_images.size()) {
                    DrawPlan::ImageBinding& noted = plan.bindings.back();
                    noted.game_texture = game_texture;
                    noted.stage = stage;
                    noted.slot = texture_ref.slot;
                    noted.sampler_serial = noted_sampler;
                }
                scratch_images.push_back(info);
            }
            (void)dummy_2d;
            if (binding_scaled) {
                if (is_image) {
                    plan.rescale_words.SetImage(image_index);
                } else {
                    plan.rescale_words.SetTexture(texture_index);
                }
            }
            ++(is_image ? image_index : texture_index);
            write.pImageInfo = &scratch_images[first];
            break;
        }
        case VC::DescriptorKind::UniformTexelBuffer: {
            // A texture buffer the shader fetches texels from (Hyrule Warriors' light lists).
            const size_t first = scratch_views.size();
            bool usable = true;
            for (u32 i = 0; i < count; ++i) {
                VC::HandleRef texture_ref = binding.texture;
                texture_ref.slot += i;
                texture_ref.byte_offset += i * VC::HandleStride;
                const u32 texture_word =
                    ReadBindingHandleWord(layout, binding, texture_ref, VC::HandleWordKind::Texture)
                        .value_or(0);
                const std::shared_ptr<Texture>& texture =
                    TextureOfId(VC::TextureIdOf(texture_word));
                const bool is_buffer = texture && IsTextureBuffer(*texture);
                if (!is_buffer) {
                    note(DescReason::TexelBuffer);
                }
                const VkBufferView view = TexelBufferView(is_buffer ? texture.get() : nullptr);
                usable &= view != VK_NULL_HANDLE;
                scratch_views.push_back(view);
            }
            if (!usable) {
                ++counters.descriptors_unsupported;
                host->HostFailed("Draw", "no arena space for a texture buffer");
                continue;
            }
            write.pTexelBufferView = &scratch_views[first];
            break;
        }
        default:
            // Storage texel buffers: not supported yet.
            ++counters.descriptors_unsupported;
            note(DescReason::TexelBuffer);
            host->HostUnsupported("Draw", "storage texel buffer descriptors are not supported");
            continue;
        }
        scratch_writes.push_back(write);
    }
    return true;
}

VkBufferView DrawContext::TexelBufferView(const Texture* texture) {
    InFlight& flight = host->HostFlight();
    if (texel_views_seq != flight.seq) {
        texel_views.clear();
        texel_views_seq = flight.seq;
    }
    // The bytes as the CPU sees them now, like a uniform buffer's; no texture: 256 bytes of zeros.
    u64 address = 0;
    u64 size = 256;
    VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    const u8* source = nullptr;
    if (texture != nullptr) {
        const VC::TextureDesc& desc = texture->desc;
        const u64 bytes =
            u64{desc.width} * VideoCore::Surface::BytesPerBlock(texture->host.pixel_format);
        format = texture->host.format;
        if (const auto it = impl.pools.find(desc.pool_serial); it != impl.pools.end() && bytes != 0) {
            const u64 guest = it->second->address + desc.pool_offset;
            source = impl.GuestPointer(guest, bytes);
            if (source != nullptr) {
                address = guest;
                size = bytes;
            }
        }
        if (source == nullptr) {
            NoteFallback(true, DescReason::TexelBuffer);
        }
    }
    for (const TexelView& entry : texel_views) {
        if (entry.address == address && entry.size == size && entry.format == format) {
            return entry.view;
        }
    }
    u8* pointer = nullptr;
    const BufferRef copy = AllocArena(size, 256, &pointer);
    if (!copy.buffer) {
        return VK_NULL_HANDLE;
    }
    if (source != nullptr) {
        std::memcpy(pointer, source, size);
    } else {
        std::memset(pointer, 0, size);
    }
    auto view = std::make_shared<vk::BufferView>(impl.device.GetLogical().CreateBufferView({
        .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .buffer = copy.buffer,
        .format = format,
        .offset = copy.offset,
        .range = size,
    }));
    const VkBufferView handle = **view;
    flight.keepalive.push_back(std::move(view));
    texel_views.push_back({address, size, format, handle});
    return handle;
}

bool DrawContext::IsBoundTarget(const Storage* storage) const {
    for (u32 i = 0; i < color_target_count; ++i) {
        if (color_targets[i].texture && color_targets[i].texture->storage.get() == storage) {
            return true;
        }
    }
    return depth_target.texture && depth_target.texture->storage.get() == storage;
}

bool DrawContext::TargetOverlapsView(const Storage& storage, const ViewKey& key) const {
    const auto overlaps = [&](const TargetBinding& target) {
        if (!target.texture || target.texture->storage.get() != &storage) {
            return false;
        }
        const u32 level = target.ref.texture.level;
        const u32 layer = target.ref.texture.layer;
        const u32 total = storage.type == VK_IMAGE_TYPE_3D
                              ? (std::max)(storage.extent.depth >> level, 1u)
                              : storage.layers;
        u32 count = target.ref.view.set_layers ? target.ref.view.nlayers : total;
        count = (std::min)((std::max)(count, 1u), total > layer ? total - layer : 1u);
        const bool levels_overlap = level >= key.base_level && level < key.base_level + key.levels;
        const bool layers_overlap = layer < key.base_layer + key.layers && key.base_layer < layer + count;
        return levels_overlap && layers_overlap;
    };
    for (u32 i = 0; i < color_target_count; ++i) {
        if (overlaps(color_targets[i])) {
            return true;
        }
    }
    return overlaps(depth_target);
}

void DrawContext::RecordFeedbackCopies(DrawPlan& plan) {
    GpuProf::Flight* const prof = plan.feedback.empty() ? nullptr : ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                            .count = static_cast<u32>(plan.feedback.size()),
                            .hash_a = GpuProf::SUB_FEEDBACK,
                            .format = static_cast<s32>(plan.feedback.front().source->host.format),
                            .width = plan.feedback.front().source->extent.width,
                            .height = plan.feedback.front().source->extent.height}
             : GpuProf::Tag{}};
    for (const DrawPlan::FeedbackCopy& copy : plan.feedback) {
        Storage& source = *copy.source;
        Storage& scratch = *copy.scratch;
        host->HostNoteTouched(copy.source);
        host->HostNoteTouched(copy.scratch);
        if (host->HostNeedsCommands(source, false)) {
            host->HostEnsureContents(source, false);
        }
        if (!source.image || !scratch.image) {
            continue; // FinishImages binds a dummy for a view that cannot be made
        }
        {
            const std::array<std::pair<Storage*, VkImageLayout>, 2> layouts{{
                {&source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                {&scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
            }};
            host->HostTransitionMany(layouts);
        }
        host->HostFlushBarrier();
        if (copy.retype_layers != 0) {
            // A 1x1x1 2D placeholder sampled as 3D or cube: its single texel goes to the typed
            // scratch (a 2D layer copies to a 3D depth slice and to every cube face).
            for (u32 face = 0; face < copy.retype_layers; ++face) {
                const VkImageCopy region{
                    .srcSubresource = {.aspectMask = source.aspect, .mipLevel = 0,
                                       .baseArrayLayer = 0, .layerCount = 1},
                    .srcOffset = {0, 0, 0},
                    .dstSubresource = {.aspectMask = source.aspect, .mipLevel = 0,
                                       .baseArrayLayer = scratch.type == VK_IMAGE_TYPE_3D ? 0u : face,
                                       .layerCount = 1},
                    .dstOffset = {0, 0, 0},
                    .extent = {source.extent.width, source.extent.height, 1},
                };
                host->HostCmd().CopyImage(*source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                          *scratch.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          region);
            }
            host->HostMarkWritten();
            ++counters.placeholder_copies;
            continue;
        }
        if (copy.retype_row) {
            // A 2D texture sampled as 1D: its first row of level 0.
            const VkImageCopy region{
                .srcSubresource = {.aspectMask = source.aspect, .mipLevel = 0,
                                   .baseArrayLayer = 0, .layerCount = 1},
                .srcOffset = {0, 0, 0},
                .dstSubresource = {.aspectMask = source.aspect, .mipLevel = 0,
                                   .baseArrayLayer = 0, .layerCount = 1},
                .dstOffset = {0, 0, 0},
                .extent = {source.extent.width, 1, 1},
            };
            host->HostCmd().CopyImage(*source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      *scratch.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
            host->HostMarkWritten();
            if (DrawResources::Scratch* record = resources.FindScratch(scratch)) {
                record->copied.Record(copy.source, this, render_target_epoch.value, {0, 1, 0, 1});
            }
            ++counters.placeholder_copies;
            continue;
        }
        const bool is_3d = source.type == VK_IMAGE_TYPE_3D;
        for (u32 level = copy.level; level < copy.level + copy.levels; ++level) {
            const VkImageSubresourceLayers subresource{
                .aspectMask = source.aspect,
                .mipLevel = level,
                .baseArrayLayer = is_3d ? 0u : copy.layer,
                .layerCount = is_3d ? 1u : copy.layers,
            };
            const VkImageCopy region{
                .srcSubresource = subresource,
                .srcOffset = {0, 0, 0},
                .dstSubresource = subresource,
                .dstOffset = {0, 0, 0},
                .extent = {(std::max)(source.extent.width >> level, 1u),
                           (std::max)(source.extent.height >> level, 1u),
                           is_3d ? (std::max)(source.extent.depth >> level, 1u) : 1u},
            };
            host->HostCmd().CopyImage(*source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      *scratch.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
        }
        host->HostMarkWritten();
        if (DrawResources::Scratch* record = resources.FindScratch(scratch)) {
            record->copied.Record(copy.source, this, render_target_epoch.value,
                                  {copy.level, copy.levels, copy.layer, copy.layers});
        }
        ++counters.feedback_copies;
        ++((source.host.depth || source.host.stencil) ? counters.feedback_depth
                                                      : counters.feedback_color);
        if (source.feedback_epoch == render_target_epoch.value) {
            ++counters.feedback_repeat;
        }
        source.feedback_epoch = render_target_epoch.value;
    }
}

void DrawContext::PruneFreshFeedback(DrawPlan& plan) {
    std::erase_if(plan.feedback, [&](const DrawPlan::FeedbackCopy& copy) {
        if (copy.retype_layers != 0) {
            return false;
        }
        const DrawResources::Scratch* record = resources.FindScratch(*copy.scratch);
        if (!record ||
            !record->copied.Serves(copy.source, this, render_target_epoch.value,
                                   {copy.level, copy.levels, copy.layer, copy.layers})) {
            return false;
        }
        if (host->HostNeedsCommands(*copy.source, false)) {
            return false; // the guest changed the bytes: the scratch is stale
        }
        ++counters.feedback_skipped;
        return true;
    });
}

bool DrawContext::PrepareOutsidePass(DrawPlan& plan) {
    bool need = false;
    BreakCause cause = BreakCause::FeedbackCopy;
    // A counter report of the open pass is written when the pass ends. A draw that writes its
    // bytes (or any pool through raw global memory) must come after it; a draw that reads them
    // only once a game barrier ordered the report (pass_reports_ordered, see Op::Barrier): before
    // that the read is not ordered after the report on the console either.
    if (pass_open && host->HostReportsPending()) {
        bool uses_report = plan.raw_global_writes || (pass_reports_ordered && plan.raw_global);
        for (const DrawPlan::RangeUse& use : plan.ranges) {
            uses_report = uses_report ||
                          ((use.write || pass_reports_ordered) &&
                           host->HostReportPending(*use.pool, use.offset, use.size));
        }
        if (uses_report) {
            cause = BreakCause::ReportRead;
            need = true;
        }
    }
    // Every storage the draw uses gets its image (lazily created) and the question "would loading
    // its contents record commands" answered BEFORE the pass is begun or continued.
    for (DrawPlan::ImageUse& use : plan.images) {
        Storage& s = *use.owner;
        use.needs_contents = host->HostNeedsCommands(s, false);
        use.skip = !s.image;
        if (!use.skip && s.current_layout != use.layout &&
            (s.current_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
             s.current_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) &&
            (use.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
             use.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            NoteFlipTarget(use.owner, use.layout);
        }
        if (!use.skip && (use.needs_contents || s.current_layout != use.layout)) {
            if (!need) {
                cause = use.needs_contents ? BreakCause::ContentsUpload : BreakCause::LayoutChange;
                if (pass_open && Debug::Get().op_log &&
                    Debug::FrameSelected(impl.presents_recorded + 1)) {
                    LOG_INFO(Render_Vulkan,
                             "[nvn] pass break ({}): {}x{} format {} layout {} -> {} contents {} "
                             "origin {}",
                             BreakCauseName(cause), s.extent.width, s.extent.height,
                             static_cast<int>(s.host.format), static_cast<int>(s.current_layout),
                             static_cast<int>(use.layout), use.needs_contents,
                             static_cast<int>(s.origin));
                }
            }
            need = true;
        }
    }
    for (const DrawPlan::RangeUse& use : plan.ranges) {
        if (use.pool->backing == Pool::Backing::Mirror &&
            host->HostMirrorNeedsUpload(*use.pool, use.offset, use.size)) {
            if (!need) {
                cause = BreakCause::MirrorUpload;
            }
            need = true;
        }
    }
    if ((plan.flush_inline && !inline_updates.empty()) || !plan.uniform_copies.empty()) {
        if (!need) {
            cause = BreakCause::InlineUpdate;
        }
        need = true;
    }
    // Storage shadows: a use of the plan that the live shadows cannot serve ends the pass (the
    // shadow is copied back outside it).
    ShadowDecision shadow;
    if (plan.shadow_candidates || !shadows.empty()) {
        shadow = ClassifyShadows(plan);
        if (shadow.flush_needed || shadow.need_outside_pass) {
            if (!need) {
                cause = shadow.flush_needed ? BreakCause::ShadowAlias : BreakCause::MirrorUpload;
            }
            need = true;
        }
    }
    if (!plan.feedback.empty()) {
        if (!need && pass_open) {
            // Only the copies a still-current scratch cannot serve end the pass.
            PruneFreshFeedback(plan);
        }
        if (!plan.feedback.empty()) {
            if (!need) {
                cause = BreakCause::FeedbackCopy;
            }
            need = true;
        }
    }
    if (need && pass_open && Debug::Tracing()) {
        ++dbg.break_reasons[BreakCauseName(cause)];
    }
    if (need && pass_open) {
        EndPass(cause);
    }
    if (!pass_open) {
        if (plan.flush_inline && !inline_updates.empty()) {
            FlushInlineUpdates();
        }
        RecordFeedbackCopies(plan);
        transition_scratch.clear();
        for (DrawPlan::ImageUse& use : plan.images) {
            if (use.skip) {
                continue;
            }
            Storage& s = *use.owner;
            host->HostNoteTouched(use.owner);
            if (use.needs_contents) {
                host->HostEnsureContents(s, false);
            }
            if (use.sample_only) {
                // Sampled in GENERAL only where the image is (BuildDescriptors): when loading the
                // contents moved it out, it is sampled read-only.
                if (use.layout == VK_IMAGE_LAYOUT_GENERAL &&
                    s.current_layout != VK_IMAGE_LAYOUT_GENERAL) {
                    use.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
                if (s.current_layout == use.layout) {
                    continue;
                }
            }
            transition_scratch.emplace_back(&s, use.layout);
        }
        host->HostTransitionMany(transition_scratch, false);
        for (const DrawPlan::RangeUse& use : plan.ranges) {
            if (use.pool->backing == Pool::Backing::Mirror) {
                host->HostUploadRange(*use.pool, use.offset, use.size);
            }
        }
        if (shadow.any) {
            ApplyShadows(plan, true);
        }
        if (!plan.uniform_copies.empty()) {
            host->HostFlushBarrier();
            for (const DrawPlan::UniformCopy& copy : plan.uniform_copies) {
                const VkBufferCopy region{
                    .srcOffset = copy.src_offset,
                    .dstOffset = copy.dst_offset,
                    .size = copy.size,
                };
                host->HostCmd().CopyBuffer(copy.src, copy.dst, region);
            }
            host->HostMarkWritten();
        }
        // A dispatch orders after earlier dispatches only as NVN does (behind a game barrier);
        // the replay's own commands above flushed for themselves.
        if (plan.graphics) {
            host->HostFlushBarrier();
        } else {
            host->HostFlushBarrierBeforeDispatch();
        }
    } else {
        for (const DrawPlan::ImageUse& use : plan.images) {
            if (!use.skip) {
                host->HostNoteTouched(use.owner);
            }
        }
        if (shadow.any) {
            ApplyShadows(plan, false);
        }
    }
    for (const DrawPlan::RangeUse& use : plan.ranges) {
        if (use.write) {
            pass_buffer_writes = true;
            pass_shader_writes = true;
            if (use.redirected) {
                host->HostNoteShadowedWrite(*use.pool, use.offset, use.size);
            } else {
                host->HostNotePoolWrite(*use.pool, use.offset, use.size);
            }
        }
    }
    if (plan.image_writes) {
        pass_shader_writes = true;
        // GENERAL also permits reads. Track shader stores even when a pass stays open and
        // the image needed no transition, so aliases see the latest contents.
        for (const DrawPlan::ImageUse& use : plan.images) {
            if (use.write && !use.skip) {
                host->HostNoteImageWrite(*use.owner);
            }
        }
    }
    if (plan.raw_global_writes) {
        pass_buffer_writes = true;
        pass_shader_writes = true;
        host->HostNoteRawGlobalWrite(plan.raw_stores_written_through);
        // The raw table reaches the guest bytes of a read-only mirror's pool: the pages it holds
        // may be stale from this draw on, unless the stores also wrote the mirror (the read-only
        // mirror table the draw was given).
        impl.MirrorInvalidateReadOnly(plan.raw_stores_written_through);
    }
    return true;
}

void DrawContext::FinishImages(DrawPlan& plan) {
    const u64 seq = host->HostFlight().seq;
    for (const DrawPlan::ImageBinding& binding : plan.bindings) {
        VkDescriptorImageInfo& info = scratch_images[binding.index];
        const Storage& storage = *binding.texture->storage;
        ViewKey key = binding.key;
        if (!binding.is_image && storage.host.bc_standin &&
            !ViewsBcStandin(key.format, storage.host.format)) {
            // Preparing the upload can replace the backing after this descriptor was planned.
            key.format = StandinViewFormat(binding.texture->host, storage.host);
            key.swizzle = PackSwizzle(ComposeStandinSwizzle(TextureSwizzle(*binding.texture),
                                                          BcStandinOf(storage.host)));
        }
        const VkImageView view = storage.image && key.format != VK_FORMAT_UNDEFINED
                                     ? resources.GetView(*binding.texture, key)
                                     : VK_NULL_HANDLE;
        if (view) {
            info.imageView = view;
            // Two uses of one image in different layouts merge to GENERAL (DrawPlan::UseImage):
            // every descriptor of the image names the layout it is in.
            info.imageLayout = plan.LayoutOf(&storage, info.imageLayout);
            if (!binding.scratch && storage.scale_owner == nullptr) {
                impl.NoteUse(*binding.texture, seq);
            }
            continue;
        }
        ++counters.descriptors_unsupported;
        NoteFallback(plan.graphics, DescReason::TextureNoView);
        info.imageView = binding.is_image ? VK_NULL_HANDLE : resources.DummyView(binding.dummy_type);
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

bool DrawContext::PushPlan(PipelineBase& pipe, DrawPlan& plan, VkPipelineBindPoint bind_point) {
    FinishImages(plan);
    vk::CommandBuffer cmd = host->HostCmd();
    if (pipe.descriptors.uses_bindless_table) {
        BindlessBind(pipe, bind_point);
    }
    if (cs_mode == CostSplit::Mode::NoBind && bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
        pipe.layout) [[unlikely]] {
        // The pass's first class draw pushed for this layout: its descriptors and constants stay.
        if (cs_draw_index > 0 && *pipe.layout == cs_pushed_layout) {
            return true;
        }
        cs_pushed_layout = *pipe.layout;
    }
    // Inside an open pass a draw skips what the pass has recorded already (nvn_pass_state.h).
    const bool track = bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && pass_open &&
                       !Debug::Get().no_state_elide;
    // Resolution scaling: which of the draw's textures and images are twins (all zero, with a down
    // factor of 1, in a guest-sized pass: the zeros and 1.0 the replay always pushed before), and
    // the factor from a scaled pass's pixels to the game's (what gl_FragCoord is multiplied by).
    static_assert(Rescale::PUSH_WORDS == 6);
    PassBoundState::PushWords pushed;
    pushed.words = plan.rescale_words.Get();
    if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && targets_scaled) {
        pushed.down_factor = target_scale.DownFactor();
    }
    if (pipe.descriptors.UsesPushConstants() && pipe.layout &&
        track && bound_state.TakePushConstants(pipe.descriptors.uses_render_area, pushed)) {
        ++counters.elided_push_constants;
    } else if (pipe.descriptors.UsesPushConstants() && pipe.layout) {
        using namespace Shader::Backend::SPIRV;
        const VkShaderStageFlags stage_flags = pipe.push_constant_stages;
        cmd.PushConstants(*pipe.layout, stage_flags, RESCALING_LAYOUT_WORDS_OFFSET,
                          sizeof(pushed.words), pushed.words.data());
        cmd.PushConstants(*pipe.layout, stage_flags, RESCALING_LAYOUT_DOWN_FACTOR_OFFSET,
                          sizeof(pushed.down_factor), &pushed.down_factor);
        if (pipe.descriptors.uses_render_area) {
            const std::array<f32, 4> area{static_cast<f32>(framebuffer_extent.width),
                                          static_cast<f32>(framebuffer_extent.height), 0.0f, 0.0f};
            cmd.PushConstants(*pipe.layout, stage_flags, RENDERAREA_LAYOUT_OFFSET, sizeof(area),
                              area.data());
        }
    }
    if (scratch_writes.empty() || !pipe.set_layout) {
        return true;
    }
    const bool need_sigs = track || !pipe.push_descriptors;
    bool comparable = false;
    if (need_sigs) {
        comparable = true;
        scratch_write_sigs.clear();
        for (const VkWriteDescriptorSet& write : scratch_writes) {
            comparable &= write.pTexelBufferView == nullptr;
            scratch_write_sigs.push_back(
                {write.dstBinding, static_cast<u32>(write.descriptorType), write.descriptorCount});
        }
        scratch_image_sigs.clear();
        for (const VkDescriptorImageInfo& info : scratch_images) {
            scratch_image_sigs.push_back({info.sampler, info.imageView, info.imageLayout});
        }
        if (track) {
            if (!comparable) {
                bound_state.ForgetDescriptors();
            } else if (bound_state.TakeDescriptors(*pipe.layout, scratch_write_sigs,
                                                   scratch_buffers, scratch_image_sigs)) {
                ++counters.elided_descriptors;
                return true;
            }
        }
    }
    if (pipe.push_descriptors) {
        cmd.PushDescriptorSetKHR(bind_point, *pipe.layout, 0, scratch_writes);
        return true;
    }
    // An allocated set, kept alive by the submission. Reuse an identical set allocated earlier in
    // this submission when all descriptors are comparable (no texel buffer views).
    DescSetSlot* cache_slot = nullptr;
    u64 cache_hash = 0;
    const u64 flight_seq = host->HostFlight().seq;
    if (comparable) {
        if (desc_set_cache.empty()) {
            desc_set_cache.resize(512);
        }
        const auto mix = [](u64 h, u64 v) noexcept {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        };
        cache_hash = mix(0, std::bit_cast<u64>(*pipe.set_layout));
        for (const PassBoundState::WriteSig& w : scratch_write_sigs) {
            cache_hash = mix(cache_hash, (static_cast<u64>(w.binding) << 32) |
                                             (static_cast<u64>(w.type) << 16) | w.count);
        }
        for (const VkDescriptorBufferInfo& b : scratch_buffers) {
            cache_hash = mix(cache_hash, std::bit_cast<u64>(b.buffer));
            cache_hash = mix(cache_hash, b.offset);
            cache_hash = mix(cache_hash, b.range);
        }
        for (const PassBoundState::ImageSig& im : scratch_image_sigs) {
            cache_hash = mix(cache_hash, std::bit_cast<u64>(im.sampler));
            cache_hash = mix(cache_hash, std::bit_cast<u64>(im.view));
            cache_hash = mix(cache_hash, static_cast<u64>(im.layout));
        }
        cache_slot = &desc_set_cache[static_cast<size_t>(cache_hash) & (desc_set_cache.size() - 1)];
        if (cache_slot->seq == flight_seq && cache_slot->hash == cache_hash &&
            cache_slot->set_layout == *pipe.set_layout && cache_slot->set != VK_NULL_HANDLE &&
            cache_slot->writes == scratch_write_sigs &&
            cache_slot->buffers.size() == scratch_buffers.size() &&
            (scratch_buffers.empty() ||
             std::memcmp(cache_slot->buffers.data(), scratch_buffers.data(),
                         scratch_buffers.size() * sizeof(VkDescriptorBufferInfo)) == 0) &&
            cache_slot->images == scratch_image_sigs) {
            cmd.BindDescriptorSets(bind_point, *pipe.layout, 0, std::span{&cache_slot->set, 1}, {});
            return true;
        }
    }
    const VkDescriptorSet set = AllocateSet(*pipe.set_layout);
    if (!set) {
        bound_state.ForgetDescriptors();
        host->HostFailed("Draw", "descriptor set allocation failed");
        return false;
    }
    for (VkWriteDescriptorSet& write : scratch_writes) {
        write.dstSet = set;
    }
    impl.dld.vkUpdateDescriptorSets(impl.logical, static_cast<u32>(scratch_writes.size()),
                                    scratch_writes.data(), 0, nullptr);
    if (cache_slot != nullptr) {
        cache_slot->seq = flight_seq;
        cache_slot->hash = cache_hash;
        cache_slot->set_layout = *pipe.set_layout;
        cache_slot->set = set;
        cache_slot->writes = scratch_write_sigs;
        cache_slot->buffers = scratch_buffers;
        cache_slot->images = scratch_image_sigs;
    }
    cmd.BindDescriptorSets(bind_point, *pipe.layout, 0, std::span{&set, 1}, {});
    return true;
}

VkDescriptorSet DrawContext::AllocateSet(VkDescriptorSetLayout layout) {
    constexpr u32 SETS_PER_POOL = 256;
    // A pool can run out of one descriptor type before SETS_PER_POOL sets: start a new one and
    // retry once (the old pool stays alive in its flight's keepalive).
    for (u32 attempt = 0; attempt < 2; ++attempt) {
        const bool fresh = !set_pool || set_pool_used >= SETS_PER_POOL;
        if (fresh) {
            const std::array<VkDescriptorPoolSize, 6> sizes{{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SETS_PER_POOL * 24},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, SETS_PER_POOL * 16},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SETS_PER_POOL * 48},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, SETS_PER_POOL * 8},
                {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, SETS_PER_POOL * 4},
                {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, SETS_PER_POOL * 4},
            }};
            try {
                set_pool = std::make_shared<vk::DescriptorPool>(impl.device.GetLogical().CreateDescriptorPool({
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .maxSets = SETS_PER_POOL,
                    .poolSizeCount = static_cast<u32>(sizes.size()),
                    .pPoolSizes = sizes.data(),
                }));
            } catch (const vk::Exception& exception) {
                LOG_ERROR(Render_Vulkan, "[nvn] descriptor pool creation failed: {}", exception.what());
                set_pool.reset();
                return VK_NULL_HANDLE;
            }
            host->HostFlight().keepalive.push_back(set_pool);
            set_pool_used = 0;
        }
        const VkDescriptorSetAllocateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = **set_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult result = impl.dld.vkAllocateDescriptorSets(impl.logical, &info, &set);
        if (result == VK_SUCCESS) {
            ++set_pool_used;
            return set;
        }
        if (fresh || (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL)) {
            return VK_NULL_HANDLE;
        }
        set_pool_used = SETS_PER_POOL; // depleted: the next attempt starts a new pool
    }
    return VK_NULL_HANDLE;
}

bool DrawContext::BindDescriptors(PipelineBase& pipe, VkPipelineBindPoint bind_point) {
    // Dispatches run outside render passes.
    EndPass(BreakCause::Dispatch);
    DrawPlan& plan = draw_plan;
    plan.Clear();
    plan.ranges.insert(plan.ranges.end(), extra_uses.begin(), extra_uses.end());
    extra_uses.clear();
    if (!BuildDescriptors(pipe, plan, bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS)) {
        return false;
    }
    if (!PrepareOutsidePass(plan)) {
        return false;
    }
    return PushPlan(pipe, plan, bind_point);
}

} // namespace Vulkan::Nvn
