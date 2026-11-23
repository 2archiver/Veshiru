// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Disk cache of the NVN pipeline cache: record posting, the driver VkPipelineCache blob and the
// background loader that rebuilds the cached pipelines.

#include <fstream>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/load_progress.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_impl.h"

namespace Vulkan::Nvn {

// ================================================================================================
// disk cache
// ================================================================================================

void PipelineCache::Impl::PostBytes(std::vector<u8> bytes) {
    writer.Post([this, bytes = std::move(bytes)] {
        if (appender.Append(bytes)) {
            ++stat_disk_written;
        } else {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache: appending to {} failed",
                      disk_path.string());
        }
    });
}

void PipelineCache::Impl::PostShader(const StageArtifact& artifact) {
    {
        std::scoped_lock lock{written_mutex};
        if (!written_shaders.insert(artifact.record->id).second) {
            return;
        }
    }
    writer.Post([this, record = artifact.record] {
        if (appender.Append(Disk::EncodeShader(*record))) {
            ++stat_disk_written;
        } else {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache: appending to {} failed",
                      disk_path.string());
        }
    });
}

namespace {
Disk::ContentId SignatureOf(const std::vector<u8>& encoded) {
    const u128 hash =
        Common::CityHash128(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    return {hash[0], hash[1]};
}
} // Anonymous namespace

void PipelineCache::Impl::PostGraphicsRecord(const Disk::GraphicsPipelineRecord& record) {
    std::vector<u8> bytes = Disk::EncodeGraphicsPipeline(record);
    {
        std::scoped_lock lock{written_mutex};
        if (!written_pipelines.insert(SignatureOf(bytes)).second) {
            return;
        }
    }
    PostBytes(std::move(bytes));
}

void PipelineCache::Impl::PostComputeRecord(const Disk::ComputePipelineRecord& record) {
    std::vector<u8> bytes = Disk::EncodeComputePipeline(record);
    {
        std::scoped_lock lock{written_mutex};
        if (!written_pipelines.insert(SignatureOf(bytes)).second) {
            return;
        }
    }
    PostBytes(std::move(bytes));
}

std::vector<u8> PipelineCache::Impl::ReadVkCacheBlob() const {
    std::ifstream file(vk_cache_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    const auto end = file.tellg();
    constexpr size_t HEADER = VK_CACHE_MAGIC.size() + sizeof(u32) + sizeof(u64) + sizeof(u32);
    if (end < 0 || static_cast<size_t>(end) < HEADER) {
        return {};
    }
    file.seekg(0);
    std::array<char, 8> magic{};
    u32 version{};
    u64 size{};
    u32 crc{};
    file.read(magic.data(), magic.size());
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    file.read(reinterpret_cast<char*>(&crc), sizeof(crc));
    if (!file || magic != VK_CACHE_MAGIC || version != VK_CACHE_VERSION ||
        size != static_cast<u64>(end) - HEADER) {
        LOG_WARNING(Render_Vulkan, "[nvn] driver pipeline cache {} discarded (header)",
                    vk_cache_path.string());
        return {};
    }
    std::vector<u8> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!file || Disk::Crc32(data) != crc) {
        LOG_WARNING(Render_Vulkan, "[nvn] driver pipeline cache {} discarded (checksum)",
                    vk_cache_path.string());
        return {};
    }
    return data;
}

void PipelineCache::Impl::WriteVkCache() {
    if (vk_cache_path.empty()) {
        return; // use_vulkan_driver_pipeline_cache is off: nothing is saved
    }
    // Pipelines created from here on may be missing from the blob: they stay counted.
    const u64 claimed = new_since_vk_write.load();
    size_t size = 0;
    if (vk_cache.Read(&size, nullptr) != VK_SUCCESS || size == 0) {
        return;
    }
    std::vector<u8> data(size);
    const VkResult result = vk_cache.Read(&size, data.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        return;
    }
    data.resize(size);
    std::vector<u8> file;
    file.reserve(VK_CACHE_MAGIC.size() + 16 + data.size());
    const auto put = [&file](const void* bytes, size_t count) {
        const auto* const first = static_cast<const u8*>(bytes);
        file.insert(file.end(), first, first + count);
    };
    const u32 version = VK_CACHE_VERSION;
    const u64 payload_size = data.size();
    const u32 crc = Disk::Crc32(data);
    put(VK_CACHE_MAGIC.data(), VK_CACHE_MAGIC.size());
    put(&version, sizeof(version));
    put(&payload_size, sizeof(payload_size));
    put(&crc, sizeof(crc));
    put(data.data(), data.size());
    if (Disk::WriteFileAtomic(vk_cache_path, file)) {
        new_since_vk_write.fetch_sub(claimed);
        LOG_INFO(Render_Vulkan, "[nvn] driver pipeline cache written: {} bytes", data.size());
    } else {
        LOG_ERROR(Render_Vulkan, "[nvn] driver pipeline cache {} could not be written",
                  vk_cache_path.string());
    }
}

void PipelineCache::Impl::MaybeWriteVkCache() {
    if (vk_cache_path.empty()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point last{
        std::chrono::steady_clock::duration{last_vk_write_ticks.load()}};
    if (now - last < VK_CACHE_WRITE_INTERVAL) {
        return;
    }
    last_vk_write_ticks = now.time_since_epoch().count();
    writer.Post([this] { WriteVkCache(); });
}

void PipelineCache::Impl::NotePreloadStep() {
    const size_t done = ++preload_done;
    if (!preload_reporting) {
        return;
    }
    const size_t total = preload_total.load();
    constexpr u64 MIN_INTERVAL_NS = 33'000'000; // the loading screen redraws at ~30 Hz at best
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    u64 last = preload_report_ticks.load();
    if (done < total) {
        if (now - last < MIN_INTERVAL_NS ||
            !preload_report_ticks.compare_exchange_strong(last, now)) {
            return;
        }
    }
    options.load_progress->Report(VideoCore::LoadCallbackStage::Build, done, total);
}

void PipelineCache::Impl::PreloadGraphics(
    const Disk::GraphicsPipelineRecord& record,
    const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders) {
    GraphicsKey key;
    std::array<StageArtifact, GRAPHICS_STAGE_COUNT> artifacts;
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        if (record.shaders[stage].IsZero()) {
            continue;
        }
        const auto it = shaders.find(record.shaders[stage]);
        if (it == shaders.end() || it->second.record->stage != stage) {
            NoteRow("disk cache: pipeline record refers to a missing shader record");
            NotePreloadStep();
            return;
        }
        artifacts[stage] = it->second;
        key.hashes[stage] = it->second.record->program_hash;
    }
    key.state = record.state;
    key.targets = record.targets;
    key.Finalize();

    std::shared_ptr<PendingKey> pk;
    {
        std::scoped_lock lock{pipeline_mutex};
        auto& slot = pending_graphics[key];
        if (!slot) {
            slot = std::make_shared<PendingKey>();
        }
        pk = slot;
        ++pk->outstanding;
    }
    workers->QueueWork([this, key, pk, artifacts = std::move(artifacts), record] {
        std::shared_ptr<GraphicsPipeline> pipeline;
        if (!stopping.load()) {
            std::array<const StageArtifact*, GRAPHICS_STAGE_COUNT> stages{};
            for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
                if (artifacts[stage].record) {
                    stages[stage] = &artifacts[stage];
                }
            }
            pipeline = AssembleGraphics(record.state, record.targets, stages);
        }
        {
            std::scoped_lock lock{pipeline_mutex};
            if (pipeline) {
                auto& variants = graphics[key];
                if (pipeline->log) {
                    pipeline->log->from_disk = true;
                    pipeline->log->variant = static_cast<u32>(variants.size());
                }
                variants.push_back(pipeline);
            }
            if (--pk->outstanding == 0) {
                pending_graphics.erase(key);
            }
        }
        {
            std::scoped_lock lock{pk->mutex};
        }
        pk->cv.notify_all();
        if (pipeline) {
            ++stat_loaded;
        }
        NotePreloadStep();
    });
}

void PipelineCache::Impl::PreloadCompute(
    const Disk::ComputePipelineRecord& record,
    const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders) {
    const auto it = shaders.find(record.shader);
    if (it == shaders.end() || it->second.record->stage != STAGE_COMPUTE) {
        NoteRow("disk cache: compute pipeline record refers to a missing shader record");
        NotePreloadStep();
        return;
    }
    ComputeKey key;
    key.hash = it->second.record->program_hash;
    key.environment_flags = record.environment_flags;
    key.local_memory_size = record.local_memory_size;
    key.shared_memory_size = record.shared_memory_size;
    std::shared_ptr<PendingKey> pk;
    {
        std::scoped_lock lock{pipeline_mutex};
        auto& slot = pending_compute[key];
        if (!slot) {
            slot = std::make_shared<PendingKey>();
        }
        pk = slot;
        ++pk->outstanding;
    }
    workers->QueueWork([this, key, pk, artifact = it->second, layout = record.compute_layout] {
        std::shared_ptr<ComputePipeline> pipeline;
        if (!stopping.load()) {
            pipeline = AssembleCompute(artifact, layout);
        }
        {
            std::scoped_lock lock{pipeline_mutex};
            if (pipeline) {
                auto& variants = compute[key];
                if (pipeline->log) {
                    pipeline->log->from_disk = true;
                    pipeline->log->variant = static_cast<u32>(variants.size());
                }
                variants.push_back(pipeline);
            }
            if (--pk->outstanding == 0) {
                pending_compute.erase(key);
            }
        }
        {
            std::scoped_lock lock{pk->mutex};
        }
        pk->cv.notify_all();
        if (pipeline) {
            ++stat_loaded;
        }
        NotePreloadStep();
    });
}

void PipelineCache::Impl::RunLoader(std::stop_token stop) {
    Common::SetCurrentThreadName("NvnPipelineLoad");
    const auto mark_queued = [this] {
        {
            std::scoped_lock lock{loader_mutex};
            loader_queued = true;
        }
        loader_cv.notify_all();
    };
    const auto finish = [this, &mark_queued] {
        mark_queued();
        {
            std::scoped_lock lock{loader_mutex};
            loader_done = true;
        }
        loader_cv.notify_all();
    };
    bool complete_reported = false;
    try {
        Disk::Contents contents;
        const Disk::LoadResult result = Disk::LoadFile(disk_path, disk_header, contents);
        switch (result.status) {
        case Disk::LoadStatus::Missing:
            break;
        case Disk::LoadStatus::Rejected:
            ++stat_disk_rejected;
            contents = {};
            LOG_WARNING(Render_Vulkan, "[nvn] pipeline cache {} rejected ({}); starting fresh",
                        disk_path.string(), result.reason);
            break;
        case Disk::LoadStatus::Loaded:
            if (result.HasBadTail()) {
                LOG_WARNING(Render_Vulkan,
                            "[nvn] pipeline cache {}: {} after {} valid records; the rest is "
                            "dropped",
                            disk_path.string(), result.reason, result.records);
            }
            break;
        }
        const bool prepared = Disk::PrepareFile(disk_path, disk_header, result) &&
                              appender.Open(disk_path);
        if (!prepared) {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache {} cannot be written; disk cache off",
                      disk_path.string());
            writer.SetState(DiskWriter::State::Disabled);
        } else {
            Disk::PruneCacheFiles(options.directory, disk_path, MAX_CACHE_FILES);
            writer.SetState(DiskWriter::State::Ready);
        }

        std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash> shaders;
        shaders.reserve(contents.shaders.size());
        for (Disk::ShaderRecord& record : contents.shaders) {
            StageArtifact artifact;
            if (record.has_program_constants) {
                artifact.constants =
                    std::make_shared<const std::vector<u32>>(std::move(record.program_constants));
                record.program_constants.clear();
            }
            const Disk::ContentId id = record.id;
            artifact.record = std::make_shared<const Disk::ShaderRecord>(std::move(record));
            shaders[id] = std::move(artifact);
        }
        {
            std::scoped_lock lock{written_mutex};
            for (const auto& [id, artifact] : shaders) {
                written_shaders.insert(id);
            }
        }
        // Records written before the state was canonical (or by another writer) differ only in
        // bits the pipeline ignores: fold them so they dedupe and match the keys of the draws.
        for (Disk::GraphicsPipelineRecord& record : contents.graphics) {
            Disk::CanonicalizeState(record.state, vertex_input_mode);
        }
        const auto preload_start = std::chrono::steady_clock::now();
        if (options.preload && workers && !stop.stop_requested()) {
            const size_t total = contents.graphics.size() + contents.compute.size();
            if (options.load_progress != nullptr && total != 0) {
                preload_total = total;
                preload_reporting = true;
                options.load_progress->Report(VideoCore::LoadCallbackStage::Prepare, 0, 0);
                options.load_progress->Report(VideoCore::LoadCallbackStage::Build, 0, total);
            }
            for (const Disk::GraphicsPipelineRecord& record : contents.graphics) {
                if (stop.stop_requested()) {
                    break;
                }
                {
                    std::scoped_lock lock{written_mutex};
                    if (!written_pipelines.insert(SignatureOf(Disk::EncodeGraphicsPipeline(record)))
                             .second) {
                        NotePreloadStep();
                        continue;
                    }
                }
                PreloadGraphics(record, shaders);
            }
            for (const Disk::ComputePipelineRecord& record : contents.compute) {
                if (stop.stop_requested()) {
                    break;
                }
                {
                    std::scoped_lock lock{written_mutex};
                    if (!written_pipelines.insert(SignatureOf(Disk::EncodeComputePipeline(record)))
                             .second) {
                        NotePreloadStep();
                        continue;
                    }
                }
                PreloadCompute(record, shaders);
            }
            // Every cached key is now pending: draws that missed so far may look again.
            mark_queued();
            workers->WaitForRequests(stop);
            if (GplEnabled() && !stop.stop_requested()) {
                // Background: the loading screen and the loader's completion do not wait for it.
                QueueWarmShaderParts(contents.graphics, shaders);
            }
        } else {
            {
                std::scoped_lock lock{written_mutex};
                for (const auto& record : contents.graphics) {
                    written_pipelines.insert(SignatureOf(Disk::EncodeGraphicsPipeline(record)));
                }
                for (const auto& record : contents.compute) {
                    written_pipelines.insert(SignatureOf(Disk::EncodeComputePipeline(record)));
                }
            }
            mark_queued();
        }
        if (preload_reporting) {
            // Whatever ended the preload (finished, stopped), the screen moves on.
            options.load_progress->Report(VideoCore::LoadCallbackStage::Complete, 0, 0);
            complete_reported = true;
        }
        if (!stop.stop_requested()) {
            if (result.status == Disk::LoadStatus::Loaded) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] pipeline cache preload: {} shader, {} graphics, {} compute "
                         "records read, {} ms",
                         contents.shaders.size(), contents.graphics.size(), contents.compute.size(),
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - preload_start)
                             .count());
            }
            LogCounts();
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache load failed: {}", e.what());
        writer.SetState(DiskWriter::State::Disabled);
        if (preload_reporting && !complete_reported) {
            options.load_progress->Report(VideoCore::LoadCallbackStage::Complete, 0, 0);
        }
    }
    finish();
}

} // namespace Vulkan::Nvn
