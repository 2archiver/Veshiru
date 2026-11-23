// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The bindless texture table of the replay (shader_recompiler/bindless_table.h). A shader that
// reads a texture handle at run time (a light or material record in a buffer the game fills)
// indexes arrays of every texture and sampler the game registered in its pools. The arrays are
// filled from those registrations, not guessed: entry `id` of an array is the image view of the
// texture registered at pool slot `id`, made with the view the game described (target, format,
// level and layer range), for the shader image type that target gives, and entry `id` of the
// sampler array is the sampler registered at that slot.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <span>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_bindless.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/surface.h"

namespace Vulkan::Nvn {
namespace {

constexpr u32 TEXTURE_BINDINGS = Shader::BINDLESS_BINDING_COUNT - 1;

[[nodiscard]] Shader::TextureType TypeOfBinding(u32 binding) {
    return static_cast<Shader::TextureType>((binding - 1) % Shader::BINDLESS_TYPE_COUNT);
}

[[nodiscard]] VkImageLayout ReadLayoutOf(const Storage& s) {
    const bool depth = s.host.depth || s.host.stencil;
    return depth && (s.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0
               ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
               : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

/// The shader image types a texture registered with `target` is sampled as (what a shader declares
/// for it): its own, and the one-layer array / array-less twin where the view allows it.
[[nodiscard]] std::array<std::optional<Shader::TextureType>, 2> TypesOfTarget(s32 target) {
    using T = Shader::TextureType;
    switch (static_cast<TexTarget>(target)) {
    case TexTarget::Dim1D:
        return {T::Color1D, std::nullopt};
    case TexTarget::Dim1DArray:
        return {T::ColorArray1D, std::nullopt};
    case TexTarget::Dim2D:
        return {T::Color2D, T::ColorArray2D};
    case TexTarget::Dim2DArray:
        return {T::ColorArray2D, T::Color2D};
    case TexTarget::Rectangle:
        return {T::Color2DRect, T::Color2D};
    case TexTarget::Dim3D:
        return {T::Color3D, std::nullopt};
    case TexTarget::Cubemap:
        return {T::ColorCube, std::nullopt};
    case TexTarget::CubemapArray:
        return {T::ColorArrayCube, std::nullopt};
    default: // multisampled and buffer textures have no table
        return {std::nullopt, std::nullopt};
    }
}

} // namespace

VkImageViewType BindlessViewType(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::ColorArray1D:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::ColorArray2D:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case Shader::TextureType::Color3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorCube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::ColorArrayCube:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default:
        return VK_IMAGE_VIEW_TYPE_2D;
    }
}

vk::DescriptorSetLayout MakeBindlessSetLayout(const Device& device) {
    constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS;
    std::array<VkDescriptorSetLayoutBinding, Shader::BINDLESS_BINDING_COUNT> bindings{};
    std::array<VkDescriptorBindingFlags, Shader::BINDLESS_BINDING_COUNT> flags{};
    constexpr VkDescriptorBindingFlags update_flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    bindings[0] = {
        .binding = Shader::BINDLESS_SAMPLER_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = Shader::BINDLESS_SAMPLER_CAPACITY,
        .stageFlags = stages,
        .pImmutableSamplers = nullptr,
    };
    flags[0] = update_flags;
    for (u32 binding = 1; binding < Shader::BINDLESS_BINDING_COUNT; ++binding) {
        bindings[binding] = {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = Shader::BINDLESS_TEXTURE_CAPACITY,
            .stageFlags = stages,
            .pImmutableSamplers = nullptr,
        };
        flags[binding] = update_flags;
    }
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = static_cast<u32>(flags.size()),
        .pBindingFlags = flags.data(),
    };
    return device.GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flags_ci,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
}

bool DrawContext::BindlessReady() {
    BindlessState& state = bindless;
    if (state.set != VK_NULL_HANDLE) {
        return true;
    }
    if (state.failed) {
        return false;
    }
    try {
        const Device& device = impl.device;
        state.layout = MakeBindlessSetLayout(device);
        const std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLER, Shader::BINDLESS_SAMPLER_CAPACITY},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
             Shader::BINDLESS_TEXTURE_CAPACITY * TEXTURE_BINDINGS},
        }};
        state.pool = device.GetLogical().CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = static_cast<u32>(sizes.size()),
            .pPoolSizes = sizes.data(),
        });
        const VkDescriptorSetLayout layout_handle = *state.layout;
        const VkDescriptorSetAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = *state.pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout_handle,
        };
        if (impl.dld.vkAllocateDescriptorSets(impl.logical, &allocate, &state.set) != VK_SUCCESS) {
            throw vk::Exception(VK_ERROR_OUT_OF_POOL_MEMORY);
        }
        // Every entry starts as the stand-in of its type, so a handle no registration names reads
        // black (partially bound arrays must not be read unwritten).
        std::vector<VkDescriptorImageInfo> infos(Shader::BINDLESS_TEXTURE_CAPACITY);
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(Shader::BINDLESS_BINDING_COUNT);
        for (u32 binding = 1; binding < Shader::BINDLESS_BINDING_COUNT; ++binding) {
            const VkImageView dummy =
                resources.DummyView(BindlessViewType(TypeOfBinding(binding)));
            state.dummy[binding] = dummy;
            state.written[binding].assign(
                Shader::BINDLESS_TEXTURE_CAPACITY,
                BindlessState::WrittenEntry{dummy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        }
        // One binding at a time: the infos are a copy of the binding's stand-in.
        for (u32 binding = 1; binding < Shader::BINDLESS_BINDING_COUNT; ++binding) {
            std::fill(infos.begin(), infos.end(),
                      VkDescriptorImageInfo{VK_NULL_HANDLE, state.dummy[binding],
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = state.set,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = Shader::BINDLESS_TEXTURE_CAPACITY,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .pImageInfo = infos.data(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            };
            device.GetLogical().UpdateDescriptorSets(std::span{&write, 1}, {});
        }
        state.written_samplers.assign(Shader::BINDLESS_SAMPLER_CAPACITY,
                                      *resources.fallback_sampler);
        std::vector<VkDescriptorImageInfo> sampler_infos(
            Shader::BINDLESS_SAMPLER_CAPACITY,
            VkDescriptorImageInfo{*resources.fallback_sampler, VK_NULL_HANDLE,
                                  VK_IMAGE_LAYOUT_UNDEFINED});
        const VkWriteDescriptorSet sampler_write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = state.set,
            .dstBinding = Shader::BINDLESS_SAMPLER_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = Shader::BINDLESS_SAMPLER_CAPACITY,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = sampler_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        device.GetLogical().UpdateDescriptorSets(std::span{&sampler_write, 1}, {});
        LOG_INFO(Render_Vulkan,
                 "[nvn] bindless table created: {} texture arrays of {} entries and {} samplers",
                 TEXTURE_BINDINGS, Shader::BINDLESS_TEXTURE_CAPACITY,
                 Shader::BINDLESS_SAMPLER_CAPACITY);
        return true;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] bindless table creation failed: {}", exception.what());
        state.failed = true;
        state.set = VK_NULL_HANDLE;
        return false;
    }
}

void DrawContext::BindlessSync(bool force) {
    BindlessState& state = bindless;
    const u64 seq = host->HostFlight().seq;
    const std::vector<u64>& texture_ids = submission->texture_pool_snapshot.Values();
    const std::vector<u64>& sampler_ids = submission->sampler_pool_snapshot.Values();
    // The registrations only change between submissions; what the table holds follows them (and
    // the images the replay created meanwhile) once per submission.
    if (!force && state.synced_seq == seq && state.synced_textures == texture_ids.data() &&
        state.synced_samplers == sampler_ids.data()) {
        return;
    }
    state.synced_seq = seq;
    state.synced_textures = texture_ids.data();
    state.synced_samplers = sampler_ids.data();

    const u32 count = static_cast<u32>(
        std::min<size_t>(std::max(texture_ids.size(), size_t{state.synced_count}),
                         Shader::BINDLESS_TEXTURE_CAPACITY));
    struct Run {
        u32 binding;
        u32 first_id;
        u32 first_info;
        u32 count;
    };
    std::vector<Run> runs;
    std::vector<VkDescriptorImageInfo> infos;
    std::vector<std::shared_ptr<Storage>> storages;
    storages.reserve(state.storages.size());
    std::array<VkImageView, TEXTURE_BINDINGS + 1> want{};
    std::array<VkImageLayout, TEXTURE_BINDINGS + 1> want_layout{};
    // The textures erased since the last sync: their node pointers are dead. With more erased
    // than the ring holds, every entry is looked up again.
    bool lookup_all = state.entries.empty();
    if (lookup_all) {
        state.entries.resize(Shader::BINDLESS_TEXTURE_CAPACITY);
    }
    thread_local std::vector<u64> erased;
    erased.clear();
    if (!lookup_all) {
        const u64 erases = impl.texture_epoch - state.entries_epoch;
        if (erases > impl.texture_erased.size()) {
            lookup_all = true;
        } else {
            for (u64 n = state.entries_epoch - 1; n != impl.texture_epoch - 1; ++n) {
                erased.push_back(impl.texture_erased[n % impl.texture_erased.size()]);
            }
            std::ranges::sort(erased);
        }
    }
    state.entries_epoch = impl.texture_epoch;
    const u64 erased_min = erased.empty() ? ~u64{0} : erased.front();
    const u64 erased_max = erased.empty() ? 0 : erased.back();
    const auto has_data = [](const Storage& s) {
        // Only what has an image holding the game's data: the table never creates one (a texture
        // the game registers and never uses would cost its memory).
        return s.image && !s.image_failed && s.contents_valid && s.samples == VK_SAMPLE_COUNT_1_BIT;
    };
    for (u32 id = 0; id < count; ++id) {
        // Pool slot 0 is "no texture" (the rebased handle keeps id 0 at 0).
        const u64 serial = id != 0 && id < texture_ids.size() ? texture_ids[id] : 0;
        BindlessState::Entry& entry = state.entries[id];
        if (!lookup_all && serial == entry.serial) {
            if (serial == 0) {
                continue; // (written as the stand-ins when the registration went)
            }
            const bool alive =
                entry.texture != nullptr &&
                (serial < erased_min || serial > erased_max ||
                 !std::ranges::binary_search(erased, serial));
            if (alive) {
                const Texture& texture = **entry.texture;
                const Storage* const s = texture.storage.get();
                if (s == entry.storage &&
                    (s == nullptr || (s->image_gen == entry.image_gen &&
                                      s->host.format == entry.format &&
                                      has_data(*s) == entry.ready &&
                                      (!entry.ready || ReadLayoutOf(*s) == entry.layout)))) {
                    if (entry.any) {
                        storages.push_back(texture.storage);
                    }
                    continue;
                }
            }
        }
        for (u32 binding = 1; binding <= TEXTURE_BINDINGS; ++binding) {
            want[binding] = state.dummy[binding];
            want_layout[binding] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        entry = {};
        entry.serial = serial;
        if (serial != 0) {
            const auto tex_it = impl.textures.find(serial);
            const std::shared_ptr<Texture>* const texture_ptr =
                tex_it != impl.textures.end() ? &tex_it->second : nullptr;
            const std::shared_ptr<Texture>& texture =
                texture_ptr != nullptr ? *texture_ptr : TextureOfId(0);
            if (texture_ptr != nullptr && texture) {
                // Stable pointer to the Impl::textures map node (never a replaceable cache slot).
                entry.texture = texture_ptr;
                entry.storage = texture->storage.get();
            }
            if (texture && texture->storage) {
                Storage& s = *texture->storage;
                entry.image_gen = s.image_gen;
                entry.format = s.host.format;
                entry.ready = has_data(s);
                if (entry.ready) {
                    entry.layout = ReadLayoutOf(s);
                    const s32 target = texture->desc.view.set_target ? texture->desc.view.target
                                                                     : texture->desc.target;
                    const bool is_integer =
                        VideoCore::Surface::IsPixelFormatInteger(texture->host.pixel_format);
                    bool any = false;
                    for (const std::optional<Shader::TextureType>& type : TypesOfTarget(target)) {
                        if (!type) {
                            continue;
                        }
                        ViewKey key{};
                        if (!ViewKeyOf(*texture, s, *type, false, key)) {
                            continue;
                        }
                        const VkImageView view = resources.GetView(*texture, key);
                        if (!view) {
                            // (not remembered: the next sync asks for the view again)
                            entry.texture = nullptr;
                            continue;
                        }
                        const VkImageLayout layout = entry.layout;
                        for (const u32 klass : {0u, 1u, 2u}) {
                            // Float and depth-compare samplers share the views; an integer
                            // sampler takes only an integer format.
                            if ((klass == 1) != is_integer) {
                                continue;
                            }
                            const u32 binding = Shader::BindlessTextureBinding(*type, klass);
                            want[binding] = view;
                            want_layout[binding] = layout;
                        }
                        any = true;
                    }
                    if (any) {
                        entry.any = true;
                        storages.push_back(texture->storage);
                    }
                }
            }
        }
        for (u32 binding = 1; binding <= TEXTURE_BINDINGS; ++binding) {
            const BindlessState::WrittenEntry desired{want[binding], want_layout[binding]};
            if (state.written[binding][id] == desired) {
                continue;
            }
            state.written[binding][id] = desired;
            infos.push_back({VK_NULL_HANDLE, want[binding], want_layout[binding]});
            runs.push_back({binding, id, static_cast<u32>(infos.size() - 1), 1});
        }
    }
    // Merge consecutive entries of one binding into one write each.
    std::stable_sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) {
        return a.binding != b.binding ? a.binding < b.binding : a.first_id < b.first_id;
    });
    std::vector<VkDescriptorImageInfo> sorted;
    sorted.reserve(infos.size());
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<Run> merged;
    for (const Run& run : runs) {
        if (!merged.empty() && merged.back().binding == run.binding &&
            merged.back().first_id + merged.back().count == run.first_id) {
            ++merged.back().count;
        } else {
            merged.push_back({run.binding, run.first_id, static_cast<u32>(sorted.size()), 1});
        }
        sorted.push_back(infos[run.first_info]);
    }
    for (const Run& run : merged) {
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = state.set,
            .dstBinding = run.binding,
            .dstArrayElement = run.first_id,
            .descriptorCount = run.count,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = sorted.data() + run.first_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        });
    }
    state.synced_count = static_cast<u32>(std::min<size_t>(texture_ids.size(),
                                                            Shader::BINDLESS_TEXTURE_CAPACITY));
    state.storages = std::move(storages);
    ++state.storages_gen;
    if (state.sync_logs++ % 600 == 0) {
        u32 registered = 0;
        for (u32 id = 1; id < texture_ids.size() && id < Shader::BINDLESS_TEXTURE_CAPACITY; ++id) {
            registered += texture_ids[id] != 0;
        }
        u32 cube = 0;
        u32 d2 = 0;
        for (u32 id = 0; id < state.synced_count; ++id) {
            const u32 cube_binding = Shader::BindlessTextureBinding(Shader::TextureType::ColorCube, 0);
            const u32 d2_binding = Shader::BindlessTextureBinding(Shader::TextureType::Color2D, 0);
            cube += state.written[cube_binding][id].view != state.dummy[cube_binding];
            d2 += state.written[d2_binding][id].view != state.dummy[d2_binding];
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] bindless table: {} textures registered, {} 2D and {} cube entries hold an "
                 "image ({} created for it so far), {} samplers registered, {} writes",
                 registered, d2, cube, state.primed_total,
                 static_cast<u32>(std::count_if(sampler_ids.begin(), sampler_ids.end(),
                                                [](u64 serial) { return serial != 0; })),
                 writes.size());
    }

    // Samplers: entry `id` is the sampler the game registered at that slot.
    const u32 sampler_count = static_cast<u32>(
        std::min<size_t>(std::max(sampler_ids.size(), size_t{state.synced_sampler_count}),
                         Shader::BINDLESS_SAMPLER_CAPACITY));
    std::vector<VkDescriptorImageInfo> sampler_infos;
    std::vector<std::pair<u32, u32>> sampler_runs; // first id, info index
    for (u32 id = 0; id < sampler_count; ++id) {
        VkSampler sampler = *resources.fallback_sampler;
        if (id != 0 && id < sampler_ids.size() && sampler_ids[id] != 0) {
            sampler = resources.GetSampler(sampler_ids[id], false);
        }
        if (state.written_samplers[id] == sampler) {
            continue;
        }
        state.written_samplers[id] = sampler;
        sampler_infos.push_back({sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
        sampler_runs.emplace_back(id, static_cast<u32>(sampler_infos.size() - 1));
    }
    std::vector<std::pair<u32, u32>> sampler_writes; // first id, count
    for (size_t i = 0; i < sampler_runs.size(); ++i) {
        if (!sampler_writes.empty() &&
            sampler_writes.back().first + sampler_writes.back().second == sampler_runs[i].first) {
            ++sampler_writes.back().second;
        } else {
            sampler_writes.emplace_back(sampler_runs[i].first, 1u);
        }
    }
    size_t info_index = 0;
    for (const auto& [first, run_count] : sampler_writes) {
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = state.set,
            .dstBinding = Shader::BINDLESS_SAMPLER_BINDING,
            .dstArrayElement = first,
            .descriptorCount = run_count,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = sampler_infos.data() + info_index,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        });
        info_index += run_count;
    }
    state.synced_sampler_count = static_cast<u32>(
        std::min<size_t>(sampler_ids.size(), Shader::BINDLESS_SAMPLER_CAPACITY));
    if (!writes.empty()) {
        impl.device.GetLogical().UpdateDescriptorSets(writes, {});
    }
}

void DrawContext::BindlessPrime(DrawPlan& plan) {
    BindlessState& state = bindless;
    // What a submission may create: the budget is memory and a count (a create is a pass break
    // and an upload), so a pool of thousands of textures fills over a few frames, not at once.
    static const u64 budget_bytes = [] {
        const char* text = std::getenv("EDEN_NX2_NVN_BINDLESS_PRIME_MB");
        return (text ? std::strtoull(text, nullptr, 10) : 64ull) << 20;
    }();
    constexpr u32 BUDGET_COUNT = 96;
    const u64 seq = host->HostFlight().seq;
    if (state.prime_seq != seq) {
        state.prime_seq = seq;
        state.prime_bytes = 0;
        state.prime_count = 0;
    }
    if (budget_bytes == 0 || state.prime_count >= BUDGET_COUNT || state.prime_bytes >= budget_bytes ||
        state.prime_exhausted_seq == seq) {
        return;
    }
    const std::vector<u64>& texture_ids = submission->texture_pool_snapshot.Values();
    const u32 count = static_cast<u32>(
        std::min<size_t>(texture_ids.size(), Shader::BINDLESS_TEXTURE_CAPACITY));
    if (count < 2) {
        return;
    }
    u32 id = state.prime_cursor < count ? state.prime_cursor : 1;
    for (u32 visited = 1; visited < count; ++visited, id = id + 1 < count ? id + 1 : 1) {
        if (texture_ids[id] == 0) {
            continue;
        }
        const std::shared_ptr<Texture>& texture = TextureOfId(id);
        if (!texture || !texture->storage) {
            continue;
        }
        Storage& s = *texture->storage;
        // Plain sampled textures only: what the game made as a render target gets its image and
        // contents from rendering, and one without guest data (image failed) is not retried.
        const VkImageUsageFlags attachment = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (s.image || s.image_failed || (s.usage & attachment) != 0 ||
            s.samples != VK_SAMPLE_COUNT_1_BIT || s.scale_owner != nullptr) {
            continue;
        }
        if (state.prime_bytes + s.vram_bytes > budget_bytes && state.prime_count != 0) {
            continue;
        }
        plan.UseImage(texture->storage, ReadLayoutOf(s), true);
        state.prime_bytes += s.vram_bytes;
        ++state.primed_total;
        state.dirty = true;
        if (++state.prime_count >= BUDGET_COUNT || state.prime_bytes >= budget_bytes) {
            id = id + 1 < count ? id + 1 : 1;
            state.prime_cursor = id;
            return;
        }
    }
    state.prime_cursor = id;
    // A whole lap found nothing more to create: the registrations change only between
    // submissions, so the rest of this one skips the walk. Every bindless draw walked the whole
    // pool once each texture had its image (Ocarina of Time: ~13% of the replay thread).
    state.prime_exhausted_seq = seq;
}

bool DrawContext::BindlessPrepare(DrawPlan& plan) {
    if (!BindlessReady()) {
        return false;
    }
    BindlessSync();
    BindlessPrime(plan);
    const u64 seq = host->HostFlight().seq;
    const u64 pass_breaks = counters.pass_breaks.load(std::memory_order_relaxed);
    const u64 pass_begins = counters.pass_begins.load(std::memory_order_relaxed);
    if (pass_open && !bindless.dirty && bindless.prepared_seq == seq &&
        bindless.prepared_write_clock == impl.write_clock &&
        bindless.prepared_pass_breaks == pass_breaks &&
        bindless.prepared_pass_begins == pass_begins &&
        bindless.prepared_storages_gen == bindless.storages_gen) {
        return true;
    }
    // A shader may read any entry: the images of the table must be in the layout the descriptors
    // name. Those a render pass or a copy left in another layout are moved like any sampled image
    // (a render target of this very draw stays: it cannot be both).
    bool any_transition = false;
    for (const std::shared_ptr<Storage>& storage : bindless.storages) {
        Storage& s = *storage;
        if (!s.image) {
            continue;
        }
        const VkImageLayout wanted = ReadLayoutOf(s);
        if (s.current_layout == wanted || IsBoundTarget(&s)) {
            continue;
        }
        plan.UseImage(storage, wanted, true);
        any_transition = true;
    }
    if (pass_open && !bindless.dirty && !any_transition) {
        bindless.prepared_seq = seq;
        bindless.prepared_write_clock = impl.write_clock;
        bindless.prepared_pass_breaks = pass_breaks;
        bindless.prepared_pass_begins = pass_begins;
        bindless.prepared_storages_gen = bindless.storages_gen;
    }
    return true;
}

void DrawContext::BindlessBind(PipelineBase& pipe, VkPipelineBindPoint bind_point) {
    if (!pipe.layout || bindless.set == VK_NULL_HANDLE) {
        return;
    }
    if (bindless.dirty) {
        // The plan created images before this draw recorded: their entries are written before
        // the set is bound (update after bind), so this very draw reads them.
        bindless.dirty = false;
        bindless.bound_cmd = VK_NULL_HANDLE;
        BindlessSync(true);
    }
    const vk::CommandBuffer cmd = host->HostCmd();
    const VkCommandBuffer cmd_handle = *cmd;
    const u64 seq = host->HostFlight().seq;
    const u64 pass_begins = counters.pass_begins.load(std::memory_order_relaxed);
    const u64 draw_seq = counters.draws.load(std::memory_order_relaxed);
    if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && pass_open &&
        !Debug::Get().no_state_elide && bindless.bound_cmd == cmd_handle &&
        bindless.bound_seq == seq && bindless.bound_pass_begins == pass_begins &&
        bindless.bound_layout == *pipe.layout && bindless.bound_point == bind_point &&
        bindless.bound_draw + 1 == draw_seq) {
        bindless.bound_draw = draw_seq;
        return;
    }
    const VkDescriptorSet set = bindless.set;
    cmd.BindDescriptorSets(bind_point, *pipe.layout, Shader::BINDLESS_SET,
                           std::span{&set, 1}, {});
    bindless.bound_cmd = cmd_handle;
    bindless.bound_layout = *pipe.layout;
    bindless.bound_point = bind_point;
    bindless.bound_seq = seq;
    bindless.bound_pass_begins = pass_begins;
    bindless.bound_draw = draw_seq;
}

} // namespace Vulkan::Nvn
