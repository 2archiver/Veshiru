// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// VK_EXT_graphics_pipeline_library path of the NVN pipeline cache (nvn_pipeline_library setting,
// PipelineCacheOptions::graphics_pipeline_library).
//
// A draw whose pipeline is not cached gets one built in these steps (on an async worker when
// asynchronous shaders are on and the draw may be skipped: it waits for it like a default-path
// draw; on the replay thread otherwise):
//   1. translated stages: reused from an earlier pipeline with the same programs and the same
//      TranslationState (what the translation reads of the state) whose texture answers still
//      hold (ShaderSet), else translated now (one translation per key at a time, TranslationHold);
//   2. the four library parts (GraphicsPipelineLibraryCache): vertex input and fragment output
//      parts per state, pre-rasterization and fragment shader parts per shader set (with the
//      DynamicFixedState states dynamic they do not depend on cull, depth or stencil state);
//      parts not compiled yet are compiled side by side on `part_workers`; they are the cost of a
//      "fast link" (a shader part is a full shader compile), the link itself is cheap;
//   3. a fast link of the parts (no link-time optimization);
//   4. the same pipeline compiled whole on the background `optimize_workers`, published in
//      GraphicsPipeline::optimized_handle: draws recorded from then on bind it.
// Programs are lowered when the game creates them (PrepareProgram), not at their first draw.
//
// The disk cache is unchanged: records hold the translated stages and the state, the boot
// preload compiles them whole, and the optimized compiles fill the VkPipelineCache it reads.
// After the preload, the library parts of the newest cached shader sets are compiled in the
// background too (QueueWarmShaderParts), so the first run-time pipeline of a known set links.

#include <chrono>
#include <string>
#include <unordered_set>

#include <boost/container/small_vector.hpp>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_impl.h"

namespace Vulkan::Nvn {

bool PipelineCache::Impl::ShaderSet::QueriesMatch(const TextureOracle& oracle) const {
    for (const StageArtifact& stage : translated.stages) {
        if (!stage.record) {
            continue;
        }
        for (const TextureQuery& query : stage.record->queries) {
            const TextureInfo now = oracle.Query(query.stage, query.record);
            if (now.type != query.info.type || now.format != query.info.format ||
                now.is_integer != query.info.is_integer) {
                return false;
            }
        }
    }
    return true;
}

std::shared_ptr<PipelineCache::Impl::ShaderSet> PipelineCache::Impl::FindShaderSet(
    const TranslationKey& key, const TextureOracle& oracle) {
    std::scoped_lock lock{translation_mutex};
    const auto it = translations.find(key);
    if (it == translations.end()) {
        return nullptr;
    }
    for (const std::shared_ptr<ShaderSet>& set : it->second) {
        if (set->QueriesMatch(oracle)) {
            return set;
        }
    }
    return nullptr;
}

std::shared_ptr<PipelineCache::Impl::ShaderSet> PipelineCache::Impl::AddShaderSet(
    const TranslationKey& key, TranslateOutput translated) {
    auto set = std::make_shared<ShaderSet>();
    set->translated = std::move(translated);
    std::scoped_lock lock{translation_mutex};
    translations[key].push_back(set);
    return set;
}

PipelineCache::Impl::TranslationHold::TranslationHold(Impl& impl_, const TranslationKey& key_)
    : impl{impl_}, key{key_} {
    {
        std::scoped_lock registry{impl.translation_mutex};
        auto& slot = impl.translation_flights[key];
        if (!slot) {
            slot = std::make_shared<TranslationFlight>();
        }
        flight = slot;
        ++flight->users;
    }
    lock = std::unique_lock{flight->mutex};
}

PipelineCache::Impl::TranslationHold::~TranslationHold() {
    lock.unlock();
    std::scoped_lock registry{impl.translation_mutex};
    if (--flight->users == 0) {
        impl.translation_flights.erase(key);
    }
}

void PipelineCache::Impl::CreateShaderSetObjects(ShaderSet& set,
                                                 const VC::DescriptorLayout& descriptors,
                                                 VkShaderStageFlags push_stages) {
    std::call_once(set.created, [&] {
        // A failure throws out of call_once and leaves `created` unset: the next pipeline of the
        // set tries again (the caller counts the failure).
        for (const StageArtifact& stage : set.translated.stages) {
            if (!stage.record) {
                continue;
            }
            const Disk::ShaderRecord& record = *stage.record;
            set.modules[record.stage] = device.GetLogical().CreateShaderModule({
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = record.spirv.size() * sizeof(u32),
                .pCode = record.spirv.data(),
            });
        }
        set.layout = CreateLayoutObjects(descriptors, false, push_stages);
    });
}

void PipelineCache::Impl::QueueOptimize(std::shared_ptr<GraphicsPipeline> pipeline,
                                        std::shared_ptr<ShaderSet> set,
                                        const GraphicsFixedState& state,
                                        const RenderPassKey& targets) {
    if (!optimize_workers) {
        return;
    }
    {
        std::scoped_lock lock{gpl_mutex};
        ++gpl_outstanding;
    }
    optimize_workers->QueueWork([this, pipeline = std::move(pipeline), set = std::move(set),
                                 state, targets] {
        if (!stopping.load()) {
            const auto start = std::chrono::steady_clock::now();
            try {
                boost::container::small_vector<VkPipelineShaderStageCreateInfo,
                                               GRAPHICS_STAGE_COUNT>
                    stage_infos;
                bool has_tessellation = false;
                for (const u32 stage : PIPELINE_ORDER) {
                    if (!set->translated.stages[stage].record) {
                        continue;
                    }
                    has_tessellation |= stage == STAGE_TESS_CONTROL || stage == STAGE_TESS_EVAL;
                    stage_infos.push_back({
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .stage = VkStageOf(stage),
                        .module = *set->modules[stage],
                        .pName = "main",
                        .pSpecializationInfo = nullptr,
                    });
                }
                // The state issues were noted when the fast-linked pipeline was described.
                std::vector<std::string> issues;
                GraphicsPipelineDescription description;
                DescribeGraphicsPipeline(
                    device,
                    GraphicsPipelineParams{
                        .state = &state,
                        .targets = &targets,
                        .render_pass = pipeline->render_pass,
                        .layout = *pipeline->layout,
                        .stages = stage_infos,
                        .viewport_count = pipeline->viewport_count,
                        .attributes_read = pipeline->attributes_read,
                        .color_outputs_written = pipeline->color_outputs_written,
                        .has_tessellation = has_tessellation,
                        .mesh = pipeline->mesh,
                        .vertex_input = vertex_input_mode,
                        .dynamic_fixed = pipeline->dynamic_fixed,
                    },
                    description, issues);
                pipeline->optimized =
                    device.GetLogical().CreateGraphicsPipeline(description.ci, *vk_cache);
                // Draws recorded from now on bind the whole compile (GraphicsPipeline::Handle).
                pipeline->optimized_handle.store(*pipeline->optimized,
                                                 std::memory_order_release);
                ++stat_gpl_optimized;
                ++new_since_vk_write;
            } catch (const std::exception& e) {
                // The fast-linked pipeline stays in use: it draws the same.
                ++stat_gpl_optimize_failed;
                NoteIssue(fmt::format("pipeline: optimized compile of a fast-linked pipeline "
                                      "failed (the fast-linked one stays): {}",
                                      FirstLine(e.what())));
            }
            const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now() - start)
                                                .count());
            stat_gpl_optimize_ns += ns;
            if (ns > 50'000'000) {
                ++stat_gpl_optimize_slow;
            }
        }
        {
            std::scoped_lock lock{gpl_mutex};
            --gpl_outstanding;
        }
        gpl_cv.notify_all();
    });
}

void PipelineCache::Impl::PrepareProgram(const std::array<StageProgram, STAGE_COUNT>& stages,
                                         std::shared_ptr<const void> owner) {
    if (!prepare_workers) {
        return;
    }
    {
        std::scoped_lock lock{gpl_mutex};
        ++gpl_outstanding;
    }
    prepare_workers->QueueWork([this, stages, owner = std::move(owner)] {
        try {
            for (u32 stage = 0; stage < STAGE_COUNT && !stopping.load(); ++stage) {
                const StageProgram& program = stages[stage];
                if (program.hash == 0 || program.control == nullptr || program.code == nullptr) {
                    continue;
                }
                // Cached by stage hash: the first draw of the program finds it (GetLowered).
                (void)GetLowered(program, stage);
                ++stat_prepared;
            }
        } catch (const std::exception& e) {
            // The first draw of the program lowers it again and reports.
            LOG_ERROR(Render_Vulkan, "[nvn] background shader preparation failed: {}", e.what());
        }
        {
            std::scoped_lock lock{gpl_mutex};
            --gpl_outstanding;
        }
        gpl_cv.notify_all();
    });
}

std::shared_ptr<PipelineCache::Impl::ShaderSet> PipelineCache::Impl::FindShaderSetOf(
    const TranslationKey& key, const std::array<StageArtifact, GRAPHICS_STAGE_COUNT>& stages) {
    std::scoped_lock lock{translation_mutex};
    const auto it = translations.find(key);
    if (it == translations.end()) {
        return nullptr;
    }
    for (const std::shared_ptr<ShaderSet>& set : it->second) {
        bool same = true;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT && same; ++stage) {
            const auto& have = set->translated.stages[stage].record;
            const auto& want = stages[stage].record;
            same = static_cast<bool>(have) == static_cast<bool>(want) &&
                   (!have || have->id == want->id);
        }
        if (same) {
            return set;
        }
    }
    return nullptr;
}

void PipelineCache::Impl::WarmShaderParts(
    const Disk::GraphicsPipelineRecord& record,
    const std::array<StageArtifact, GRAPHICS_STAGE_COUNT>& artifacts, const TranslationKey& key) {
    std::shared_ptr<ShaderSet> set;
    {
        // The set the first run-time pipeline of these programs finds (FindShaderSet): made from
        // the stages the cache file holds, as the preload's whole compiles are.
        const TranslationHold hold{*this, key};
        set = FindShaderSetOf(key, artifacts);
        if (!set) {
            TranslateOutput translated;
            translated.stages = artifacts;
            set = AddShaderSet(key, std::move(translated));
        }
    }
    std::array<const StageArtifact*, GRAPHICS_STAGE_COUNT> stages{};
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        if (set->translated.stages[stage].record) {
            stages[stage] = &set->translated.stages[stage];
        }
    }
    if (AssembleGraphics(record.state, record.targets, stages, set, true)) {
        ++stat_gpl_warmed;
    }
}

void PipelineCache::Impl::QueueWarmShaderParts(
    const std::vector<Disk::GraphicsPipelineRecord>& records,
    const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders) {
    // Compiling shader parts is the whole cost of a "fast link" (a shader part is a full shader
    // compile). The newest sets, one record each, in a few background jobs so the game keeps its
    // CPU: at most WARM_JOBS threads work on it at a time.
    constexpr size_t WARM_JOBS = 4;
    struct Item {
        Disk::GraphicsPipelineRecord record;
        std::array<StageArtifact, GRAPHICS_STAGE_COUNT> artifacts;
        TranslationKey key;
    };
    std::array<std::vector<Item>, WARM_JOBS> jobs;
    std::unordered_set<std::string> seen;
    size_t count = 0;
    for (auto it = records.rbegin(); it != records.rend() && count < MAX_WARMED_SHADER_SETS;
         ++it) {
        Item item;
        bool complete = true;
        std::string identity;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            if (it->shaders[stage].IsZero()) {
                continue;
            }
            const auto found = shaders.find(it->shaders[stage]);
            if (found == shaders.end() || found->second.record->stage != stage) {
                complete = false; // PreloadGraphics noted it
                break;
            }
            item.artifacts[stage] = found->second;
            item.key.hashes[stage] = found->second.record->program_hash;
            identity.append(reinterpret_cast<const char*>(&it->shaders[stage]),
                            sizeof(it->shaders[stage]));
        }
        if (!complete) {
            continue;
        }
        item.record = *it;
        item.key.state = MakeTranslationState(it->state, it->targets, vertex_input_mode,
                                              must_emulate_scaled_formats);
        item.key.try_switch_hash = TrySwitchesHash();
        item.key.Finalize();
        identity.append(reinterpret_cast<const char*>(&item.key.hash), sizeof(item.key.hash));
        if (!seen.insert(std::move(identity)).second) {
            continue; // another record of the same shader set: its shader parts are the same
        }
        jobs[count % WARM_JOBS].push_back(std::move(item));
        ++count;
    }
    for (std::vector<Item>& items : jobs) {
        if (items.empty()) {
            continue;
        }
        {
            std::scoped_lock lock{gpl_mutex};
            ++gpl_outstanding;
        }
        workers->QueueWork([this, items = std::move(items)] {
            for (const Item& item : items) {
                if (stopping.load()) {
                    break;
                }
                try {
                    WarmShaderParts(item.record, item.artifacts, item.key);
                } catch (const std::exception& e) {
                    NoteIssue(fmt::format("pipeline: warming shader parts failed: {}",
                                          FirstLine(e.what())));
                }
            }
            {
                std::scoped_lock lock{gpl_mutex};
                --gpl_outstanding;
            }
            gpl_cv.notify_all();
        });
    }
}

} // namespace Vulkan::Nvn
