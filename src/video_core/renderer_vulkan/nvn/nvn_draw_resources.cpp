// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <fmt/format.h>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/game_presets.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_sampler_aniso.h"
#include "video_core/sampler_overrides.h"
#include "video_core/shader_cache_v2/cache.h"

VideoCommon::NvnCacheTransition VideoCommon::ClearShaderCachesForNvn(
    const std::filesystem::path& title_dir, bool mario_kart_world) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto title_status = fs::symlink_status(title_dir, ec);
    if (title_status.type() == fs::file_type::not_found) {
        return NvnCacheTransition::Unchanged;
    }
    if (ec || !fs::is_directory(title_status)) {
        return NvnCacheTransition::Failed;
    }
    std::vector<fs::path> legacy_files;
    const auto collect = [&](const fs::path& path) {
        if (fs::exists(path, ec)) {
            legacy_files.push_back(path);
        }
        return !ec;
    };
    const auto collect_dir = [&](const fs::path& directory) {
        if (!collect(directory / ShaderCacheV2::FileName) ||
            !collect(directory / ShaderCacheV2::DriverFileName)) {
            return false;
        }
        for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            const bool versioned_emulated =
                name.ends_with(".bin") && name.find(".fork-v") != std::string::npos &&
                (name.starts_with("vulkan") || name.starts_with("opengl"));
            if (versioned_emulated ||
                std::ranges::find(EMULATED_VULKAN_CACHE_FILES, name) !=
                    EMULATED_VULKAN_CACHE_FILES.end()) {
                legacy_files.push_back(it->path());
            }
        }
        return !ec;
    };
    if (!collect(title_dir / "opengl.bin")) {
        return NvnCacheTransition::Failed;
    }
    if (!collect_dir(title_dir)) {
        return NvnCacheTransition::Failed;
    }
    if (mario_kart_world) {
        const auto subdir = title_dir / MKW_EMULATED_CACHE_SUBDIR;
        if (fs::exists(subdir, ec)) {
            if (fs::is_symlink(subdir, ec) || ec || !collect_dir(subdir)) {
                return NvnCacheTransition::Failed;
            }
        } else if (ec) {
            return NvnCacheTransition::Failed;
        }
    }
    if (legacy_files.empty()) {
        return NvnCacheTransition::Unchanged;
    }
    // Keep the evidence of an emulated run until every NVN cache file has been removed.
    const auto remove_file = [&](const fs::path& path) {
        if (!fs::is_directory(path, ec) && !ec) {
            fs::remove(path, ec);
            return !ec;
        }
        return false;
    };
    std::vector<fs::path> nvn_files;
    for (fs::directory_iterator it(title_dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (IsNvnPipelineCacheFile(it->path().filename().string())) {
            nvn_files.push_back(it->path());
        }
    }
    if (ec || !std::ranges::all_of(nvn_files, remove_file) ||
        !std::ranges::all_of(legacy_files, remove_file)) {
        return NvnCacheTransition::Failed;
    }
    return NvnCacheTransition::Cleared;
}

namespace Vulkan::Nvn {
namespace {

constexpr u64 ARENA_CHUNK_SIZE = u64{4} << 20;
constexpr u64 DUMMY_BUFFER_SIZE = 64 * 1024;
constexpr u64 SHADOW_CHUNK_SIZE = u64{8} << 20;
/// Chunks kept for reuse: the vertex staging of a busy frame (nvn_vertex_stage.h, 20-40 MiB at
/// DK's city spot) fills a few chunks per submission, for every submission in flight, and a slow
/// GPU keeps more frames in flight. With 16 kept, a run that shared the GPU with another emulator
/// created and destroyed a chunk every third frame (F:/eg/results/vertex-stage2, L-new1-b).
constexpr u32 MAX_FREE_SHADOWS = 32;
constexpr u64 SHADOW_KEEP_MAX = u64{32} << 20;
constexpr u64 MAX_FREE_SHADOW_BYTES = u64{256} << 20;

/// What a shadow of a storage range is bound as, and copied from and to the pool buffer; staged
/// vertex and index ranges (nvn_vertex_stage.h) are carved from the same chunks.
constexpr VkBufferUsageFlags SHADOW_USAGE =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

constexpr VkBufferUsageFlags ARENA_USAGE =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;

VkSamplerAddressMode WrapMode(u32 nvn) {
    switch (nvn) {
    case 0: // CLAMP
    case 7: // CLAMP_TO_EDGE
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 1:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2: // MIRROR_CLAMP
    case 3: // MIRROR_CLAMP_TO_EDGE
        return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    case 4: // MIRROR_CLAMP_TO_BORDER
    case 5: // CLAMP_TO_BORDER
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 6:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

VkCompareOp CompareOp(u32 nvn) {
    // NVNcompareFunc NEVER=1 .. ALWAYS=8 -> VK_COMPARE_OP_NEVER=0 .. ALWAYS=7
    return static_cast<VkCompareOp>(std::clamp<u32>(nvn, 1, 8) - 1);
}

/// The sampler an NVN sampler descriptor asks for, before the diagnostic overrides, the reduction
/// mode and the border colour (GetSampler adds those). `steps`: the doublings the Anisotropic
/// Filtering setting adds (nvn_sampler_aniso.h); 0 is the guest's own sampler.
VkSamplerCreateInfo SamplerInfoOf(const VC::SamplerDesc& d, s32 steps, f32 max_limit) {
    const u32 min = d.min_filter;
    const bool mipmapped = min >= 2;
    const SamplerAnisotropyFields anisotropy = SamplerAnisotropyOf(d, steps, max_limit);
    return VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = d.mag_filter == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = (min & 1) != 0 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = min >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = WrapMode(d.wrap_s),
        .addressModeV = WrapMode(d.wrap_t),
        .addressModeW = WrapMode(d.wrap_r),
        .mipLodBias = d.lod_bias,
        .anisotropyEnable = anisotropy.enable,
        .maxAnisotropy = anisotropy.max_anisotropy,
        .compareEnable = static_cast<VkBool32>(d.compare_mode == 1),
        .compareOp = CompareOp(d.compare_func),
        .minLod = mipmapped ? d.lod_min : 0.0f,
        // A filter without mipmaps samples level 0 only.
        .maxLod = mipmapped ? d.lod_max : 0.25f,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
}

VkFormat DummyFormat() {
    return VK_FORMAT_R8G8B8A8_UNORM;
}

} // namespace

DrawResources::DrawResources(Context::Impl& impl_) : impl{impl_}, clears{impl_.device} {
    shader_refresh = Settings::values.nvn_shader_refresh.GetValue();
    dummy_buffer = impl.CreateHostBuffer(DUMMY_BUFFER_SIZE, ARENA_USAGE, MemoryUsage::Upload);
    std::memset(dummy_buffer.Mapped().data(), 0, DUMMY_BUFFER_SIZE);
    dummy_buffer.Flush();
    sink_buffer =
        impl.CreateHostBuffer(SINK_SLICE_SIZE * SINK_SLICES, ARENA_USAGE, MemoryUsage::Upload);
    std::memset(sink_buffer.Mapped().data(), 0, SINK_SLICE_SIZE * SINK_SLICES);
    sink_buffer.Flush();
    fallback_sampler = impl.device.GetLogical().CreateSampler({
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
    max_sampler_anisotropy =
        (std::max)(1.0f, impl.device.GetPhysical().GetProperties().limits.maxSamplerAnisotropy);
    anisotropy_steps = CurrentAnisotropyBoostSteps();
}

PipelineCache& DrawResources::Cache() {
    if (!cache_instance) {
        std::scoped_lock lock{cache_mutex};
        PipelineCacheOptions options;
        options.async_compile =
            !shader_refresh && Settings::values.use_asynchronous_shaders.GetValue();
        options.graphics_pipeline_library = Settings::values.nvn_pipeline_library.GetValue();
        options.sync_diet = Settings::values.nvn_sync_diet.GetValue();
        options.shader_notify = impl.options.shader_notify;
        options.load_progress = impl.options.load_progress;
        cache_instance = std::make_unique<PipelineCache>(impl.device, std::move(options));
    }
    return *cache_instance;
}

void DrawResources::CollectCacheRows(std::vector<PipelineCache::Row>& rows) {
    std::scoped_lock lock{cache_mutex};
    if (cache_instance) {
        cache_instance->CollectRows(rows);
    }
}

void DrawResources::OpenCache(u64 title_id) {
    if (cache_instance) {
        return;
    }
    PipelineCacheOptions options;
    options.title_id = title_id;
    options.revision = Common::g_scm_rev;
    options.prefetch_graphics = title_id == Settings::MarioKartWorldProgramId && !shader_refresh;
    options.async_compile = !shader_refresh && Settings::values.use_asynchronous_shaders.GetValue();
    options.graphics_pipeline_library = Settings::values.nvn_pipeline_library.GetValue();
    options.sync_diet = Settings::values.nvn_sync_diet.GetValue();
    options.shader_notify = impl.options.shader_notify;
    options.load_progress = impl.options.load_progress;
    if (options.async_compile) {
        // A dispatch never skips while its pipeline compiles, so a new compute program used to
        // be compiled on the replay thread at its first dispatch (seconds, every one of them in
        // one stall the first frame that meets them). Compile the dispatches a submission is
        // about to replay on the workers first (DrawContext::PrefetchUpcomingCompute).
        options.prefetch_compute = true;
        LOG_INFO(Render_Vulkan,
                 "[nvn] asynchronous shaders on: run-time pipeline compiles run on workers; the "
                 "draws that may be skipped are skipped until they finish, dispatches wait for "
                 "a compile the compute lookahead started earlier{}",
                 options.graphics_pipeline_library
                     ? " (nvn_pipeline_library: library parts are compiled and linked there too)"
                     : "");
    }
    LOG_INFO(Render_Vulkan, "[nvn] pass boundary syncs: {}",
             options.sync_diet ? "one barrier (nvn_sync_diet)"
                               : "barrier plus the render passes' two external dependencies");
    options.driver_cache_file = Settings::values.use_vulkan_driver_pipeline_cache.GetValue();
    const char* const own_dir = std::getenv("EDEN_NX2_NVN_CACHE_DIR");
    const bool isolated_cache = own_dir != nullptr && *own_dir != '\0';
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto title_dir = shader_dir / fmt::format("{:016x}", title_id);
    const auto transition =
        title_id != 0 && !isolated_cache && !shader_refresh &&
                Settings::values.use_disk_shader_cache.GetValue()
            ? VideoCommon::ClearShaderCachesForNvn(title_dir, impl.mario_kart_world)
            : VideoCommon::NvnCacheTransition::Unchanged;
    if (transition == VideoCommon::NvnCacheTransition::Cleared) {
        LOG_INFO(Render_Vulkan, "[nvn] cleared shader caches for {:016x} on the switch from "
                                "Emulated GPU to VTX-HLE", title_id);
    }
    // Isolate MKW's warm-start rendering regression from the opaque NVIDIA cache while keeping
    // the portable pipeline records available for preload. EDEN_NX2_NVN_MKW_DRIVER_CACHE=1 keeps
    // the driver cache for an A/B of that workaround (the setting is otherwise ignored for MKW).
    const char* const mkw_driver_cache = std::getenv("EDEN_NX2_NVN_MKW_DRIVER_CACHE");
    const bool keep_mkw_driver_cache =
        mkw_driver_cache != nullptr && *mkw_driver_cache != '\0' &&
        std::strcmp(mkw_driver_cache, "0") != 0;
    if (title_id == Settings::MarioKartWorldProgramId && impl.device.IsNvidia() &&
        options.driver_cache_file && !shader_refresh) {
        if (keep_mkw_driver_cache) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] Mario Kart World: keeping the NVIDIA driver pipeline cache "
                     "(EDEN_NX2_NVN_MKW_DRIVER_CACHE)");
        } else {
            options.driver_cache_file = false;
            LOG_INFO(Render_Vulkan,
                     "[nvn] Mario Kart World: start with an empty NVIDIA driver pipeline cache; "
                     "keep the NVN pipeline records");
        }
    }
    if (shader_refresh) {
        options.driver_cache_file = false;
        LOG_INFO(Render_Vulkan,
                 "[nvn] periodic shader refresh on: fresh synchronous compiles every 60 seconds; "
                 "disk caches are not read or written");
    } else if (!Settings::values.use_disk_shader_cache.GetValue()) {
        // As on the emulated GPU (no LoadDiskResources, no cache file): nothing is read from or
        // written to the profile's shader directory; every pipeline is translated and compiled
        // when a draw first needs it.
        LOG_INFO(Render_Vulkan,
                 "[nvn] disk shader cache off (use_disk_shader_cache): pipelines are not loaded "
                 "from or saved to disk");
    } else if (transition == VideoCommon::NvnCacheTransition::Failed) {
        LOG_ERROR(Render_Vulkan, "[nvn] could not clear {}; running with an in-memory cache",
                  title_dir.string());
    } else if (isolated_cache) {
        // A/B runs that must not share a profile's cache at all (the cache files of one
        // directory are pruned to a few). The cache creates the directory.
        options.directory = own_dir;
        LOG_INFO(Render_Vulkan, "[nvn] pipeline cache directory: {}", options.directory.string());
    } else {
        if (Common::FS::CreateDir(shader_dir) && Common::FS::CreateDir(title_dir)) {
            options.directory = title_dir;
        } else {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache: cannot create {}", title_dir.string());
        }
    }
    auto cache = std::make_unique<PipelineCache>(impl.device, std::move(options));
    std::scoped_lock lock{cache_mutex};
    cache_instance = std::move(cache);
}
DrawResources::~DrawResources() = default;

void DrawResources::MaybeRefreshShaders() {
    if (!shader_refresh || !cache_instance) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_shader_refresh == std::chrono::steady_clock::time_point{}) {
        last_shader_refresh = now;
    } else if (now - last_shader_refresh >= impl.options.shader_refresh_interval) {
        try {
            cache_instance->ClearShaders();
        } catch (const std::exception& e) {
            // A failed replacement allocation leaves the old cache usable. The present's GPU
            // work is already submitted: do not fail its flight because a refresh failed.
            LOG_WARNING(Render_Vulkan,
                        "[nvn] shader cache refresh failed; keeping current shaders: {}", e.what());
        }
        last_shader_refresh = std::chrono::steady_clock::now();
    }
}

void DrawResources::TracePass(u32 width, u32 height) {
    ++frame_trace.passes[{width, height}];
}

void DrawResources::TraceDispatch(bool tensor_like) {
    ++frame_trace.dispatches;
    frame_trace.dispatches_128 += tensor_like ? 1 : 0;
}

void DrawResources::TracePresent(u64 present) {
    FrameTrace& t = frame_trace;
    // The main scene target is the largest render pass extent that more than one pass uses; the
    // line is logged every 30th present and whenever that size changes.
    std::pair<u32, u32> main{};
    u64 main_area = 0;
    u32 total = 0;
    for (const auto& [size, count] : t.passes) {
        total += count;
        const u64 area = u64{size.first} * size.second;
        if (count >= 2 && area > main_area) {
            main = size;
            main_area = area;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = t.last_present == std::chrono::steady_clock::time_point{}
                          ? 0.0
                          : std::chrono::duration<double, std::milli>(now - t.last_present).count();
    t.last_present = now;
    if (present % 30 == 0 || main != t.last_main) {
        std::vector<std::pair<std::pair<u32, u32>, u32>> sizes(t.passes.begin(), t.passes.end());
        std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
            return u64{a.first.first} * a.first.second > u64{b.first.first} * b.first.second;
        });
        std::string list;
        for (size_t i = 0; i < sizes.size() && i < 6; ++i) {
            list += fmt::format(" {}x{}:{}", sizes[i].first.first, sizes[i].first.second,
                                sizes[i].second);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] tstrace present {}: main target {}x{}, {} passes, sizes (largest first){}, "
                 "{} dispatches ({} with 128 threads in x), {:.1f} ms since the previous present",
                 present, main.first, main.second, total, list, t.dispatches, t.dispatches_128, ms);
    }
    t.last_main = main;
    t.passes.clear();
    t.dispatches = 0;
    t.dispatches_128 = 0;
}

void DrawResources::NoteAsyncSkips(u64 present) {
    const u64 draws = counters.draws_skipped_async.load(std::memory_order_relaxed);
    if (draws == noted_draw_skips) {
        return;
    }
    counters.frames_with_async_skips.fetch_add(1, std::memory_order_relaxed);
    // One line per frame with skips, for the pop-in analysis; bounded so a pathological run
    // cannot flood the log.
    constexpr u64 MAX_LINES = 6000;
    if (async_skip_lines < MAX_LINES) {
        ++async_skip_lines;
        LOG_INFO(Render_Vulkan,
                 "[nvn] async pipelines: present {}: {} draws skipped awaiting a pipeline", present,
                 draws - noted_draw_skips);
    }
    noted_draw_skips = draws;
}

void DrawResources::OnProgramCreated(const std::shared_ptr<const VC::ProgramDesc>& program) {
    const VC::ProgramDesc& desc = *program;
    std::array<u64, STAGE_COUNT> hashes{};
    std::array<StageProgram, STAGE_COUNT> stages{};
    for (u32 stage = 0; stage < STAGE_COUNT; ++stage) {
        if (!desc.control[stage].empty() && !desc.code[stage].empty()) {
            hashes[stage] = HashProgramStage(desc.control[stage], desc.code[stage]);
            stages[stage] = {hashes[stage], &desc.control[stage], &desc.code[stage]};
        }
    }
    program_hashes[desc.serial] = hashes;
    PipelineCache* cache{};
    {
        std::scoped_lock lock{cache_mutex};
        cache = cache_instance.get();
    }
    if (cache != nullptr && cache->UsesGraphicsPipelineLibrary()) {
        // The stage pointers stay valid: the cache keeps `program` until it is done.
        cache->PrepareProgram(stages, program);
    }
}

void DrawResources::RefreshSamplerSetting() {
    const s32 steps = CurrentAnisotropyBoostSteps();
    if (steps != anisotropy_steps) {
        anisotropy_steps = steps;
        ++samplers_epoch;
    }
}

namespace {

/// The border colour of a sampler without custom border colours: the nearest built-in one.
VkBorderColor BuiltinBorderColor(const VC::SamplerDesc& d) {
    const bool opaque = d.border[3] > 0.5f;
    const bool white = d.border[0] > 0.5f;
    return !opaque ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
           : white ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                   : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
}

} // namespace

VkSampler DrawResources::GetSampler(u64 serial, bool boosted) {
    const auto desc_it = impl.samplers.find(serial);
    if (desc_it == impl.samplers.end()) {
        return *fallback_sampler;
    }
    const VC::SamplerDesc& d = desc_it->second;
    // The key: a sampler the setting leaves alone is the guest's own for every setting, one the
    // setting changes is built per setting value, so a changed setting never reuses a VkSampler.
    const s32 steps = SamplerStepsKey(d, boosted ? anisotropy_steps : 0, max_sampler_anisotropy);
    if (const auto it = samplers.find(serial); it != samplers.end()) {
        for (const SamplerVariant& variant : it->second) {
            if (variant.steps == steps) {
                return *variant.sampler;
            }
        }
    }
    VkSamplerCreateInfo info = SamplerInfoOf(d, steps, max_sampler_anisotropy);
    // EDEN_NX2_FORCE_FILTER / EDEN_NX2_FORCE_ANISO / EDEN_NX2_SAMPLER_DIAGNOSTICS
    // (video_core/sampler_overrides.h): the same diagnostics as the legacy Sampler.
    bool forced = VideoCore::SamplerOverrides::ApplyFilters(info.magFilter, info.minFilter,
                                                            info.mipmapMode, info.maxAnisotropy);
    forced |= VideoCore::SamplerOverrides::ApplyLod(info.minLod, info.maxLod);
    info.anisotropyEnable = static_cast<VkBool32>(info.maxAnisotropy > 1.0f);
    if (VideoCore::SamplerOverrides::ShouldLog()) {
        LOG_INFO(
            Render_Vulkan,
            "[nvn] sampler {:#x}: mag={} min={} mip={} lod=[{}, {}] bias={} aniso={} forced={}",
            serial, static_cast<u32>(info.magFilter), static_cast<u32>(info.minFilter),
            static_cast<u32>(info.mipmapMode), info.minLod, info.maxLod, info.mipLodBias,
            info.maxAnisotropy, forced);
    }
    VkSamplerReductionModeCreateInfo reduction{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
        .pNext = nullptr,
        .reductionMode = d.reduction == 1   ? VK_SAMPLER_REDUCTION_MODE_MIN
                         : d.reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX
                                            : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    };
    // A compare sampler must use WEIGHTED_AVERAGE (VUID-VkSamplerCreateInfo-compareEnable-01423).
    if (d.reduction != 0 && info.compareEnable == VK_FALSE &&
        impl.device.IsExtSamplerFilterMinmaxSupported()) {
        info.pNext = &reduction;
    }
    VkSamplerCustomBorderColorCreateInfoEXT custom_border{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
        .pNext = info.pNext,
        .customBorderColor = {.float32 = {d.border[0], d.border[1], d.border[2], d.border[3]}},
        .format = VK_FORMAT_UNDEFINED,
    };
    const bool border_used = info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    if (border_used) {
        if (impl.device.IsCustomBorderColorUsable()) {
            info.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            info.pNext = &custom_border;
        } else {
            info.borderColor = BuiltinBorderColor(d);
        }
    }
    try {
        auto& variants = samplers[serial];
        variants.push_back({steps, impl.device.GetLogical().CreateSampler(info)});
        return *variants.back().sampler;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] sampler {:#x} creation failed: {}", serial,
                  exception.what());
        return *fallback_sampler;
    }
}

std::optional<VideoCommon::PathDiff::Sampler> DrawResources::DescribeSampler(u64 serial) const {
    const auto desc_it = impl.samplers.find(serial);
    if (desc_it == impl.samplers.end()) {
        return std::nullopt;
    }
    const VC::SamplerDesc& d = desc_it->second;
    // What the emulated path reports too: the sampler with the setting applied, before its
    // per-view reduction.
    const VkSamplerCreateInfo info = SamplerInfoOf(
        d, SamplerStepsKey(d, anisotropy_steps, max_sampler_anisotropy), max_sampler_anisotropy);
    const bool border_used = info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    return VideoCommon::PathDiff::Sampler{
        .mag_filter = static_cast<u32>(info.magFilter),
        .min_filter = static_cast<u32>(info.minFilter),
        .mipmap_mode = static_cast<u32>(info.mipmapMode),
        .address = {static_cast<u32>(info.addressModeU), static_cast<u32>(info.addressModeV),
                    static_cast<u32>(info.addressModeW)},
        .lod_bias = info.mipLodBias,
        .min_lod = info.minLod,
        .max_lod = info.maxLod,
        .anisotropy = info.anisotropyEnable != VK_FALSE,
        .max_anisotropy = info.maxAnisotropy,
        .compare = info.compareEnable != VK_FALSE,
        .compare_op = static_cast<u32>(info.compareOp),
        // Only meaningful when an address mode clamps to the border (0 otherwise, both paths).
        .border_color = static_cast<u32>(!border_used ? VkBorderColor{}
                                         : impl.device.IsCustomBorderColorUsable()
                                             ? VK_BORDER_COLOR_FLOAT_CUSTOM_EXT
                                             : BuiltinBorderColor(d)),
        .border = {d.border[0], d.border[1], d.border[2], d.border[3]},
        .reduction =
            static_cast<u32>(d.reduction == 1   ? VK_SAMPLER_REDUCTION_MODE_MIN
                             : d.reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX
                                                : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE),
    };
}

void DrawResources::ReleaseSampler(u64 serial, u64 after_seq) {
    const auto it = samplers.find(serial);
    if (it == samplers.end()) {
        return;
    }
    auto variants = std::make_shared<std::vector<SamplerVariant>>(std::move(it->second));
    samplers.erase(it);
    ++samplers_epoch;
    impl.retired.push_back({after_seq, [variants]() mutable { variants->clear(); }});
}

VkImageView DrawResources::GetView(const Texture& texture, const ViewKey& key) {
    // A stand-in changes BC format when a later upload classifies it differently (smart BC,
    // RefitStandin recreates its image as BC1/BC4/BC5/BC3). A ViewKey made before that (a clear's,
    // an attachment's, one memoised by a caller) still names the old format, and a view of another
    // BC class than the image's is invalid usage: NVIDIA loses the device on it. Such a draw is
    // left without the texture for this binding (the next binding pass rebuilds the key from the
    // storage, ViewKeyOf) instead of creating the view.
    const ViewKey& key_valid = key;
    if (texture.storage && texture.storage->host.bc_standin &&
        !ViewsBcStandin(key.format, texture.storage->host.format)) {
        static std::atomic<u32> skipped{0};
        if (skipped.fetch_add(1) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] stand-in view {:#x} left out: texture {:#x}, storage format {:#x} "
                        "(a view key from before the stand-in's format changed)",
                        static_cast<int>(key.format), texture.serial,
                        static_cast<int>(texture.storage->host.format));
        }
        return VK_NULL_HANDLE;
    }
    // The texture remembers its texture_views node (stable until TakeViews erases it).
    if (texture.view_slot == nullptr || texture.view_slot_epoch != views_epoch) {
        auto& fresh = const_cast<Texture&>(texture);
        fresh.view_slot = &texture_views[texture.serial];
        fresh.view_slot_epoch = views_epoch;
    }
    TextureViews& views = *static_cast<TextureViews*>(texture.view_slot);
    if (!views.storage) {
        views.storage = texture.storage;
    }
    for (auto& [existing, view] : views.views) {
        if (existing == key_valid) {
            return *view;
        }
    }
    const auto swizzle = [&](u32 shift) {
        return static_cast<VkComponentSwizzle>((key.swizzle >> shift) & 0xF);
    };
    // A view inherits the image's usage; a format without storage support (sRGB) must not claim it.
    VkImageViewUsageCreateInfo usage_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .usage = texture.storage->usage,
    };
    if ((usage_info.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
        !impl.device.IsFormatSupported(key.format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                       FormatType::Optimal)) {
        usage_info.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
    }
    try {
        vk::ImageView view = impl.device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_info,
            .flags = 0,
            .image = *texture.storage->image,
            .viewType = key.type,
            .format = key.format,
            .components{
                .r = swizzle(0),
                .g = swizzle(4),
                .b = swizzle(8),
                .a = swizzle(12),
            },
            .subresourceRange{
                .aspectMask = key.aspect,
                .baseMipLevel = key.base_level,
                .levelCount = key.levels,
                .baseArrayLayer = key.base_layer,
                .layerCount = key.layers,
            },
        });
        views.views.emplace_back(key_valid, std::move(view));
        return *views.views.back().second;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan,
                  "[nvn] image view for texture {:#x} (format {}, type {}, levels {}+{}, layers "
                  "{}+{}) failed: {}",
                  texture.serial, static_cast<int>(key.format), static_cast<int>(key.type),
                  key.base_level, key.levels, key.base_layer, key.layers, exception.what());
        return VK_NULL_HANDLE;
    }
}

std::vector<vk::ImageView> DrawResources::TakeStorageViews(const Storage& storage) {
    std::vector<vk::ImageView> taken;
    for (auto& [serial, views] : texture_views) {
        if (views.storage.get() != &storage) {
            continue;
        }
        for (auto& [key, view] : views.views) {
            taken.push_back(std::move(view));
        }
        views.views.clear();
    }
    return taken;
}

std::shared_ptr<TextureViews> DrawResources::TakeViews(u64 serial) {
    const auto it = texture_views.find(serial);
    if (it == texture_views.end()) {
        return nullptr;
    }
    auto result = std::make_shared<TextureViews>(std::move(it->second));
    texture_views.erase(it);
    ++views_epoch;
    return result;
}

DrawResources::Scratch* DrawResources::FindScratch(const Storage& storage) {
    for (Scratch& scratch : scratch_images) {
        if (scratch.storage.get() == &storage) {
            return &scratch;
        }
    }
    return nullptr;
}

DrawResources::Scratch* DrawResources::GetScratch(const Storage& source, u32 slot) {
    const VkImageCreateFlags cube_flag = source.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    for (Scratch& scratch : scratch_images) {
        const Storage& s = *scratch.storage;
        if (scratch.slot == slot && s.host.format == source.host.format && s.type == source.type &&
            s.extent.width == source.extent.width && s.extent.height == source.extent.height &&
            s.extent.depth == source.extent.depth && s.levels == source.levels &&
            s.layers == source.layers && s.aspect == source.aspect &&
            (s.create_flags & cube_flag) == cube_flag) {
            return &scratch;
        }
    }
    auto storage = std::make_shared<Storage>();
    storage->host = source.host;
    storage->type = source.type;
    storage->extent = source.extent;
    storage->levels = source.levels;
    storage->layers = source.layers;
    storage->aspect = source.aspect;
    storage->samples = VK_SAMPLE_COUNT_1_BIT;
    storage->create_flags =
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
        (source.type == VK_IMAGE_TYPE_3D
             ? static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT)
             : 0u);
    storage->create_flags |= cube_flag;
    storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
    try {
        storage->image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = storage->create_flags,
            .imageType = storage->type,
            .format = storage->host.format,
            .extent = storage->extent,
            .mipLevels = storage->levels,
            .arrayLayers = storage->layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = storage->usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] feedback scratch image ({}x{}x{}, format {}) failed: {}",
                  source.extent.width, source.extent.height, source.extent.depth,
                  static_cast<int>(source.host.format), exception.what());
        return nullptr;
    }
    // Written by copies only: no guest bytes behind it, nothing to upload or check.
    storage->contents_valid = true;
    storage->origin = StorageOrigin::Gpu;
    auto texture = std::make_shared<Texture>();
    texture->serial = next_scratch_serial++;
    texture->storage = storage;
    texture->host = storage->host;
    scratch_images.push_back({std::move(storage), std::move(texture), slot});
    return &scratch_images.back();
}

DrawResources::Scratch* DrawResources::GetRetypedScratch(const Storage& source, u32 slot,
                                                         RetypeTarget target) {
    const bool cube = target == RetypeTarget::Cube;
    const VkImageType type = cube                          ? VK_IMAGE_TYPE_2D
                             : target == RetypeTarget::Row ? VK_IMAGE_TYPE_1D
                                                           : VK_IMAGE_TYPE_3D;
    const u32 layers = cube ? 6u : 1u;
    const VkExtent3D extent{source.extent.width,
                            target == RetypeTarget::Row ? 1u : source.extent.height, 1};
    for (Scratch& scratch : scratch_images) {
        const Storage& s = *scratch.storage;
        if (scratch.slot == slot && s.host.format == source.host.format && s.type == type &&
            s.extent.width == extent.width && s.extent.height == extent.height &&
            s.extent.depth == 1 &&
            s.layers == layers && s.aspect == source.aspect) {
            return &scratch;
        }
    }
    auto storage = std::make_shared<Storage>();
    storage->host = source.host;
    storage->type = type;
    storage->extent = extent;
    storage->levels = 1;
    storage->layers = layers;
    storage->aspect = source.aspect;
    storage->samples = VK_SAMPLE_COUNT_1_BIT;
    storage->create_flags =
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
        (cube ? static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) : 0u);
    storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
    try {
        storage->image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = storage->create_flags,
            .imageType = storage->type,
            .format = storage->host.format,
            .extent = storage->extent,
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = storage->usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] placeholder image (format {}, {}) failed: {}",
                  static_cast<int>(source.host.format),
                  cube ? "cube" : target == RetypeTarget::Row ? "1D" : "3D", exception.what());
        return nullptr;
    }
    storage->contents_valid = true;
    storage->origin = StorageOrigin::Gpu;
    auto texture = std::make_shared<Texture>();
    texture->serial = next_scratch_serial++;
    texture->storage = storage;
    texture->host = storage->host;
    scratch_images.push_back({std::move(storage), std::move(texture), slot});
    return &scratch_images.back();
}

std::shared_ptr<ArenaChunk> DrawResources::AcquireArena(u64 min_size) {
    {
        std::scoped_lock lock{arena_mutex};
        for (auto it = free_arenas.begin(); it != free_arenas.end(); ++it) {
            if ((*it)->size >= min_size) {
                std::shared_ptr<ArenaChunk> chunk = std::move(*it);
                free_arenas.erase(it);
                chunk->used = 0;
                return chunk;
            }
        }
    }
    auto chunk = std::make_shared<ArenaChunk>();
    chunk->size = (std::max)(min_size, ARENA_CHUNK_SIZE);
    chunk->buffer =
        impl.CreateHostBuffer(chunk->size, ARENA_USAGE, MemoryUsage::Upload, AllocationKind::Arena);
    chunk->pointer = chunk->buffer.Mapped().data();
    return chunk;
}

void DrawResources::ReleaseArenas(std::vector<std::shared_ptr<ArenaChunk>>& chunks) {
    std::scoped_lock lock{arena_mutex};
    for (auto& chunk : chunks) {
        // Keep the pool small: oversize one-offs are dropped.
        if (chunk->size == ARENA_CHUNK_SIZE && free_arenas.size() < 8) {
            free_arenas.push_back(std::move(chunk));
        }
    }
    chunks.clear();
}

std::shared_ptr<ShadowChunk> DrawResources::AcquireShadow(u64 min_size) {
    {
        std::scoped_lock lock{shadow_mutex};
        for (auto it = free_shadows.begin(); it != free_shadows.end(); ++it) {
            if ((*it)->size >= min_size) {
                std::shared_ptr<ShadowChunk> chunk = std::move(*it);
                free_shadows.erase(it);
                chunk->used = 0;
                return chunk;
            }
        }
    }
    auto chunk = std::make_shared<ShadowChunk>();
    // A multiple of 256: staged vertex copies by shader write whole 16-byte units up to the end of
    // their allocation (AppendVertexStageTasks).
    chunk->size = ((std::max)(min_size, SHADOW_CHUNK_SIZE) + 255) / 256 * 256;
    try {
        chunk->buffer = impl.CreateHostBuffer(chunk->size, SHADOW_USAGE, MemoryUsage::DeviceLocal,
                                              AllocationKind::Shadow);
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] storage shadow memory ({} bytes) failed: {}", chunk->size,
                  exception.what());
        return nullptr;
    }
    counters.shadow_chunks_created.fetch_add(1, std::memory_order_relaxed);
    return chunk;
}

void DrawResources::ReleaseShadows(std::vector<std::shared_ptr<ShadowChunk>>& chunks) {
    std::scoped_lock lock{shadow_mutex};
    u64 free_bytes = 0;
    for (const auto& existing : free_shadows) {
        free_bytes += existing->size;
    }
    for (auto& chunk : chunks) {
        // Chunks up to SHADOW_KEEP_MAX stay: the alias transfer's scratch buffer is a chunk sized
        // to the shared bytes of two textures, asked for many times a second.
        if (chunk->size <= SHADOW_KEEP_MAX && free_shadows.size() < MAX_FREE_SHADOWS &&
            free_bytes + chunk->size <= MAX_FREE_SHADOW_BYTES) {
            free_bytes += chunk->size;
            free_shadows.push_back(std::move(chunk));
        } else {
            counters.shadow_chunks_destroyed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    chunks.clear();
}

VkBuffer DrawResources::DummyBuffer() {
    return *dummy_buffer;
}

u64 DrawResources::SinkSlice(u64 program_hash, u32 stage, u32 slot) {
    const std::tuple<u64, u32, u32> key{program_hash, stage, slot};
    if (const auto it = sink_index.find(key); it != sink_index.end()) {
        return (u64{it->second} + 1) * SINK_SLICE_SIZE;
    }
    if (sink_keys.size() + 1 >= SINK_SLICES) {
        return 0; // the shared slice
    }
    const u32 index = static_cast<u32>(sink_keys.size());
    sink_keys.push_back({program_hash, stage, slot, false});
    sink_index.emplace(key, index);
    return (u64{index} + 1) * SINK_SLICE_SIZE;
}

void DrawResources::CheckSink(u64 present) {
    if (present % SINK_CHECK_PRESENTS != 0) {
        return;
    }
    sink_buffer.Invalidate();
    const u8* const base = sink_buffer.Mapped().data();
    const auto first_nonzero = [&](u64 slice) -> std::optional<std::pair<u64, u32>> {
        const u32* const words = reinterpret_cast<const u32*>(base + slice * SINK_SLICE_SIZE);
        for (u64 i = 0; i < SINK_SLICE_SIZE / sizeof(u32); ++i) {
            if (words[i] != 0) {
                return std::pair<u64, u32>{i * sizeof(u32), words[i]};
            }
        }
        return std::nullopt;
    };
    for (size_t i = 0; i < sink_keys.size(); ++i) {
        SinkKey& key = sink_keys[i];
        if (key.reported) {
            continue;
        }
        if (const auto found = first_nonzero(i + 1)) {
            key.reported = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] a shader wrote through a storage buffer binding without a buffer "
                        "(slot unbound, or its range is in no pool): "
                        "program {:016x} stage {} slot {} (word +{:#x} = {:#x}); the hardware "
                        "would fault there",
                        key.program_hash, key.stage, key.slot, found->first, found->second);
        }
    }
    if (!sink_overflow_reported) {
        if (const auto found = first_nonzero(0)) {
            sink_overflow_reported = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] a shader wrote through a storage buffer binding without a buffer "
                        "(more unbound slots than the sink has slices; word +{:#x} = {:#x})",
                        found->first, found->second);
        }
    }
}

VkImageView DrawResources::DummyView(VkImageViewType type) {
    size_t index = 0;
    VkImageType image_type = VK_IMAGE_TYPE_2D;
    u32 layers = 1;
    VkImageCreateFlags flags = 0;
    switch (type) {
    case VK_IMAGE_VIEW_TYPE_1D:
        index = 0;
        image_type = VK_IMAGE_TYPE_1D;
        break;
    case VK_IMAGE_VIEW_TYPE_2D:
        index = 1;
        break;
    case VK_IMAGE_VIEW_TYPE_3D:
        index = 2;
        image_type = VK_IMAGE_TYPE_3D;
        break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
        index = 3;
        layers = 6;
        flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
        index = 4;
        image_type = VK_IMAGE_TYPE_1D;
        layers = 2;
        break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
        index = 5;
        layers = 2;
        break;
    default: // CUBE_ARRAY
        index = 6;
        layers = 6;
        flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        break;
    }
    DummyImage& dummy = dummy_images[index];
    if (dummy.view) {
        return *dummy.view;
    }
    const bool is_array_view = type == VK_IMAGE_VIEW_TYPE_1D_ARRAY ||
                               type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ||
                               type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    try {
        dummy.image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .imageType = image_type,
            .format = DummyFormat(),
            .extent = {1, 1, 1},
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        dummy.view = impl.device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *dummy.image,
            .viewType = type,
            .format = DummyFormat(),
            .components{},
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_2D ||
                               type == VK_IMAGE_VIEW_TYPE_3D)
                                  ? 1u
                                  : layers,
            },
        });
        static_cast<void>(is_array_view);
        impl.ImmediateSubmit([&](vk::CommandBuffer cmd) {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = layers,
            };
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *dummy.image,
                .subresourceRange = range,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0, barrier);
            const VkClearColorValue black{.float32 = {0.0f, 0.0f, 0.0f, 0.0f}};
            cmd.ClearColorImage(*dummy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, black, range);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, barrier);
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] dummy image creation failed: {}", exception.what());
        dummy.view = vk::ImageView{};
        return VK_NULL_HANDLE;
    }
    return *dummy.view;
}

VkBuffer DrawResources::ProgramConstants(const PipelineBase& pipeline, u32 stage) {
    auto& buffers = constant_buffers[&pipeline];
    vk::Buffer& buffer = buffers[stage];
    if (buffer) {
        return *buffer;
    }
    const auto& words = pipeline.stages[stage].program_constants;
    const u64 size = (std::max<u64>)(words ? words->size() * sizeof(u32) : 0, 256);
    buffer = impl.CreateHostBuffer((size + 255) / 256 * 256, ARENA_USAGE, MemoryUsage::Upload,
                                   AllocationKind::Arena);
    std::span<u8> mapped = buffer.Mapped();
    std::memset(mapped.data(), 0, mapped.size());
    if (words && !words->empty()) {
        std::memcpy(mapped.data(), words->data(), words->size() * sizeof(u32));
    }
    buffer.Flush();
    return *buffer;
}

} // namespace Vulkan::Nvn
