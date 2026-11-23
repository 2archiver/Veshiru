// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Private definition of PipelineCache::Impl, shared by nvn_pipeline_cache.cpp (lookups,
// translation), nvn_pipeline_vk.cpp (render passes, layouts, VkPipeline assembly) and
// nvn_pipeline_cache_disk.cpp (disk cache). Not a public header.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/cityhash.h"
#include "common/diag_request.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "shader_recompiler/backend/bindings.h"
#include "video_core/native_tensor_programs.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_disk.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/shader_log.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn::Detail {
constexpr size_t MAX_REASON_CHARS = 100;
constexpr auto VK_CACHE_WRITE_INTERVAL = std::chrono::seconds(60);
constexpr u32 VK_CACHE_VERSION = VideoCommon::NVN_DRIVER_CACHE_VERSION;
constexpr auto VK_CACHE_MAGIC = VideoCommon::NVN_DRIVER_CACHE_MAGIC;
/// Cache files kept per directory, one per translation key (see Disk::FileNameForKey).
constexpr size_t MAX_CACHE_FILES = 6;
/// Shader sets of the disk cache whose library parts are compiled in the background after the
/// preload (GPL path): the newest ones, so a long-lived cache file cannot queue minutes of work.
constexpr size_t MAX_WARMED_SHADER_SETS = 1024;
constexpr const char* VK_CACHE_FILE_NAME = VideoCommon::NVN_DRIVER_CACHE_FILE_NAME.data();

/// Graphics stages in pipeline order as NVNshaderStage indices (task, mesh, VS, TCS, TES, GS,
/// FS): the order BuildDescriptorLayout and the shared Bindings are fed in.
constexpr std::array<u32, 7> PIPELINE_ORDER{6, 7, 0, 3, 4, 2, 1};
constexpr u32 STAGE_VERTEX = 0;
constexpr u32 STAGE_FRAGMENT = 1;
constexpr u32 STAGE_GEOMETRY = 2;
constexpr u32 STAGE_TESS_CONTROL = 3;
constexpr u32 STAGE_TESS_EVAL = 4;

constexpr u32 ENV_LOCAL_MEMORY = 1;
constexpr u32 ENV_SHARED_MEMORY = 2;

inline std::string FirstLine(std::string_view text, size_t max_chars = MAX_REASON_CHARS) {
    text = text.substr(0, text.find_first_of("\r\n"));
    if (text.size() > max_chars) {
        text = text.substr(0, max_chars);
    }
    return std::string{text};
}

inline VkShaderStageFlagBits VkStageOf(u32 nvn_stage) {
    switch (nvn_stage) {
    case 0:
        return VK_SHADER_STAGE_VERTEX_BIT;
    case 1:
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    case 2:
        return VK_SHADER_STAGE_GEOMETRY_BIT;
    case 3:
        return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case 4:
        return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case 6:
        return VK_SHADER_STAGE_TASK_BIT_EXT;
    case 7:
        return VK_SHADER_STAGE_MESH_BIT_EXT;
    default:
        return VK_SHADER_STAGE_COMPUTE_BIT;
    }
}

inline VkShaderStageFlags StageFlagsOf(const VC::DescriptorBinding& binding) {
    VkShaderStageFlags flags = 0;
    for (u32 stage = 0; stage < VC::ShaderStageCount; ++stage) {
        if ((binding.stage_mask & (1u << stage)) != 0) {
            flags |= VkStageOf(stage);
        }
    }
    return flags != 0 ? flags
                      : static_cast<VkShaderStageFlags>(VkStageOf(static_cast<u32>(binding.stage)));
}

inline std::string_view StageNameOf(u32 nvn_stage) {
    return VC::StageName(static_cast<VC::ShaderStage>(nvn_stage));
}

// --- keys -------------------------------------------------------------------------------------

static_assert(sizeof(RenderPassKey) == 10 * sizeof(u32), "RenderPassKey must have no padding");
static_assert(sizeof(GraphicsFixedState) ==
                  16 * sizeof(u32) + sizeof(u64) + sizeof(u64) * MAX_COLOR_TARGETS + 16 +
                      sizeof(u32) * (MAX_VERTEX_ATTRIBUTES + 2 * MAX_VERTEX_STREAMS) +
                      MAX_VERTEX_ATTRIBUTES,
              "GraphicsFixedState must have no padding: its bytes are hashed and compared");

struct RenderPassKeyHash {
    size_t operator()(const RenderPassKey& key) const noexcept {
        return static_cast<size_t>(
            Common::CityHash64(reinterpret_cast<const char*>(&key), sizeof(key)));
    }
};

struct GraphicsKey {
    std::array<u64, GRAPHICS_STAGE_COUNT> hashes{};
    GraphicsFixedState state{};
    RenderPassKey targets;
    u64 try_switch_hash{};
    u64 hash{};

    void Finalize() {
        u64 h = Common::CityHash64(reinterpret_cast<const char*>(hashes.data()), sizeof(hashes));
        h = Common::CityHash64WithSeed(reinterpret_cast<const char*>(&state), sizeof(state), h);
        h = Common::CityHash64WithSeed(reinterpret_cast<const char*>(&targets), sizeof(targets), h);
        if (try_switch_hash != 0) {
            h = Common::CityHash64WithSeed(reinterpret_cast<const char*>(&try_switch_hash),
                                           sizeof(try_switch_hash), h);
        }
        hash = h;
    }
};

struct GraphicsKeyHash {
    size_t operator()(const GraphicsKey& key) const noexcept {
        return static_cast<size_t>(key.hash);
    }
};

struct GraphicsKeyEqual {
    bool operator()(const GraphicsKey& a, const GraphicsKey& b) const noexcept {
        return a.hash == b.hash && a.hashes == b.hashes && a.targets == b.targets &&
               a.try_switch_hash == b.try_switch_hash &&
               std::memcmp(&a.state, &b.state, sizeof(GraphicsFixedState)) == 0;
    }
};

struct ComputeKey {
    u64 hash{};
    u64 try_switch_hash{};
    u32 environment_flags{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    [[nodiscard]] bool operator==(const ComputeKey&) const = default;
};

struct ComputeKeyHash {
    size_t operator()(const ComputeKey& key) const noexcept {
        return static_cast<size_t>(key.hash ^ key.try_switch_hash ^
                                   (u64{key.environment_flags} << 1) ^
                                   (u64{key.local_memory_size} * 0x9E3779B97F4A7C15ull) ^
                                   (u64{key.shared_memory_size} << 32));
    }
};

class DefaultOracle final : public TextureOracle {
public:
    TextureInfo Query(u32, const VC::NvnHandleRecord&) const override {
        return {};
    }
};
const DefaultOracle DEFAULT_ORACLE{};

struct LoweredEntry {
    std::shared_ptr<const VC::LoweredNvnProgram> program;
    std::shared_ptr<const std::vector<u32>> c1;
};

/// A translated stage, fresh from translation or loaded from disk.
struct StageArtifact {
    std::shared_ptr<const Disk::ShaderRecord> record;
    std::shared_ptr<const std::vector<u32>> constants;
};

struct PendingKey {
    // Set before publishing the key under pipeline_mutex.
    bool speculative{};
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<int> outstanding{0};
};

inline VC::NvnShaderPools& ThreadPools() {
    thread_local VC::NvnShaderPools pools;
    return pools;
}

/// Single background thread that owns the file appends. Jobs are held back until the loader
/// has validated and prepared the file.
class DiskWriter {
public:
    enum class State { Pending, Ready, Disabled };

    DiskWriter() : thread{[this] { Run(); }} {}
    ~DiskWriter() {
        Shutdown();
    }

    void Shutdown() {
        {
            std::scoped_lock lock{mutex};
            if (stop) {
                return;
            }
            stop = true;
            if (state == State::Pending) {
                state = State::Disabled;
            }
        }
        cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }

    void SetState(State new_state) {
        {
            std::scoped_lock lock{mutex};
            state = new_state;
        }
        cv.notify_all();
        idle.notify_all();
    }

    void Post(std::function<void()> job) {
        {
            std::scoped_lock lock{mutex};
            if (state == State::Disabled || stop) {
                return;
            }
            queue.push_back(std::move(job));
        }
        cv.notify_one();
    }

    void Flush() {
        std::unique_lock lock{mutex};
        idle.wait(lock,
                  [this] { return stop || (state != State::Pending && queue.empty() && !busy); });
    }

private:
    void Run() {
        Common::SetCurrentThreadName("NvnPipelineWriter");
        std::unique_lock lock{mutex};
        while (true) {
            cv.wait(lock, [this] { return stop || (state != State::Pending && !queue.empty()); });
            if (queue.empty()) {
                idle.notify_all();
                if (stop) {
                    return;
                }
                continue;
            }
            if (state == State::Disabled) {
                queue.clear();
                idle.notify_all();
                continue;
            }
            auto job = std::move(queue.front());
            queue.pop_front();
            busy = true;
            lock.unlock();
            job();
            lock.lock();
            busy = false;
            idle.notify_all();
        }
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable idle;
    std::deque<std::function<void()>> queue;
    State state{State::Pending};
    bool busy{};
    bool stop{};
    std::thread thread;
};

} // namespace Vulkan::Nvn::Detail

namespace Vulkan::Nvn {
namespace ShaderLog = VideoCommon::ShaderLog;
using namespace Detail;

struct PipelineCache::Impl {
    Impl(const Device& device_, PipelineCacheOptions options_);
    ~Impl();

    // --- API ----------------------------------------------------------------------------------
    /// `key_hash` receives the `GraphicsKey::hash` of the request (for the shader log hook).
    std::shared_ptr<GraphicsPipeline> GetGraphics(const GraphicsRequest& request, bool* not_ready,
                                                  u64& key_hash,
                                                  const std::function<void(u64)>& look_ahead = {},
                                                  std::optional<u64> prefetch_skip = {});
    std::shared_ptr<ComputePipeline> GetCompute(const ComputeRequest& request, bool* not_ready);
    VkRenderPass GetRenderPass(const RenderPassKey& key);
    void Flush();
    void ClearShaders();
    PipelineCacheStats Stats() const;
    void CollectRows(std::vector<Row>& out) const;

    // --- diagnostics --------------------------------------------------------------------------
    /// Counts `reason`; true the first time it is seen.
    bool NoteRow(const std::string& reason);
    /// Counts `reason` and logs it once.
    void NoteIssue(const std::string& reason);
    void Reject(u64 stage_hash, u32 nvn_stage, std::string_view phase, std::string_view message);
    bool AnyStageRejected(std::span<const u64> hashes);
    /// Counts as `LogCounts`, but only when a count changed since the last line.
    void LogCountsIfChanged() const;
    /// Counts a finished run-time compile (`start` to now, of which `translate` was spent in the
    /// shader translation and `create` in vkCreate*Pipelines) and logs one
    /// `[nvn] pipeline compile` line for it (a warning when it took more than 50 ms).
    void NoteCompileTime(std::string_view kind, u64 hash,
                         std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::duration translate,
                         std::chrono::steady_clock::duration create);
    void LogCounts() const;

    // --- translation --------------------------------------------------------------------------
    struct TranslateInput {
        bool speculative{};
        std::array<const StageProgram*, STAGE_COUNT> programs{};
        /// Graphics: everything the translation reads of the draw's state; null for compute.
        const TranslationState* translation{};
        const TextureOracle* oracle{};
        const VC::NvnEnvironmentOptions* environment{};
        u64 extra_key{};
    };
    struct TranslateOutput {
        std::array<StageArtifact, STAGE_COUNT> stages;
        VC::ComputeDriverLayout compute_layout;
        /// Shader log only (filled while `ShaderLog::Enabled()`): IR text and constant buffer
        /// use of the stages translated so far, also when a later stage failed.
        std::array<std::string, STAGE_COUNT> ir;
        std::array<std::array<u32, LOG_MAX_CBUFS>, STAGE_COUNT> cbuf_used{};
        u32 cbuf_known{};
    };
    LoweredEntry GetLowered(const StageProgram& program, u32 nvn_stage);
    bool Translate(const TranslateInput& in, TranslateOutput& out);
    Shader::RuntimeInfo MakeRuntimeInfo(u32 nvn_stage, const Shader::IR::Program& program,
                                        const Shader::IR::Program* previous,
                                        const TranslationState* translation, bool has_geometry);

    // --- Vulkan objects -----------------------------------------------------------------------
    struct LayoutObjects {
        vk::DescriptorSetLayout set_layout;
        /// Set BINDLESS_SET when a stage samples through the table (nvn_bindless.h).
        vk::DescriptorSetLayout bindless_layout;
        vk::PipelineLayout layout;
        bool push_descriptors{};
        u32 push_constant_size{};
    };
    bool CanUsePush(u32 descriptor_count) const;
    /// `push_stages`: the stages of the push-constant range (PipelineBase::push_constant_stages).
    LayoutObjects CreateLayoutObjects(const VC::DescriptorLayout& descriptors, bool is_compute,
                                      VkShaderStageFlags push_stages);
    VC::DescriptorLayout MergeLayouts(std::span<const StageArtifact* const> stages);
    struct ShaderSet;
    /// Without `gpl_set` the pipeline is compiled whole from modules made for the call. With it
    /// (GPL path) the set's modules are used and the pipeline is fast-linked from library parts
    /// (`fast_linked`), or compiled whole when the parts cannot express it. With `warm_parts` (GPL
    /// path, preload) only the parts are compiled and cached: the linked pipeline is returned for
    /// the caller to drop, nothing is counted, and a pipeline the parts cannot express gives null.
    std::shared_ptr<GraphicsPipeline> AssembleGraphics(
        const GraphicsFixedState& state, const RenderPassKey& targets,
        const std::array<const StageArtifact*, GRAPHICS_STAGE_COUNT>& stages,
        const std::shared_ptr<ShaderSet>& gpl_set = {}, bool warm_parts = false);
    std::shared_ptr<ComputePipeline> AssembleCompute(const StageArtifact& stage,
                                                     const VC::ComputeDriverLayout& layout);
    void FillStages(PipelineBase& pipeline, std::span<const StageArtifact* const> stages);

    // --- VK_EXT_graphics_pipeline_library path (nvn_pipeline_gpl.cpp) -------------------------
    /// The translated stages of one TranslationKey for one set of texture answers, and the
    /// Vulkan objects its shader library parts are compiled from (made by the first pipeline
    /// that links from it). Every pipeline built from these stages shares the set; its address
    /// stands for the pipeline layout in the library keys (GraphicsPipelineLibraryCache), so a
    /// set lives as long as the cache.
    struct ShaderSet {
        TranslateOutput translated;
        std::once_flag created; ///< `modules` and `layout` exist once it is set
        std::array<vk::ShaderModule, GRAPHICS_STAGE_COUNT> modules; ///< by NVNshaderStage
        LayoutObjects layout; ///< the layout the shader parts are compiled with
        /// True when `oracle` still answers every texture query of the stages as it did.
        [[nodiscard]] bool QueriesMatch(const TextureOracle& oracle) const;
    };
    /// What the translation of a graphics pipeline depends on, besides the texture answers.
    struct TranslationKey {
        std::array<u64, GRAPHICS_STAGE_COUNT> hashes{};
        TranslationState state{};
        u64 try_switch_hash{};
        u64 hash{};
        void Finalize() {
            u64 h =
                Common::CityHash64(reinterpret_cast<const char*>(hashes.data()), sizeof(hashes));
            h = Common::CityHash64WithSeed(reinterpret_cast<const char*>(&state), sizeof(state), h);
            if (try_switch_hash != 0) {
                h = Common::CityHash64WithSeed(reinterpret_cast<const char*>(&try_switch_hash),
                                               sizeof(try_switch_hash), h);
            }
            hash = h;
        }
    };
    struct TranslationKeyHash {
        size_t operator()(const TranslationKey& key) const noexcept {
            return static_cast<size_t>(key.hash);
        }
    };
    struct TranslationKeyEqual {
        bool operator()(const TranslationKey& a, const TranslationKey& b) const noexcept {
            return a.hash == b.hash && a.hashes == b.hashes &&
                   a.try_switch_hash == b.try_switch_hash &&
                   std::memcmp(&a.state, &b.state, sizeof(TranslationState)) == 0;
        }
    };
    [[nodiscard]] bool GplEnabled() const noexcept {
        return library_cache != nullptr;
    }
    /// The set of `key` whose texture answers `oracle` still gives, or null.
    std::shared_ptr<ShaderSet> FindShaderSet(const TranslationKey& key,
                                             const TextureOracle& oracle);
    std::shared_ptr<ShaderSet> AddShaderSet(const TranslationKey& key, TranslateOutput translated);
    /// The set of `key` made of exactly these stage artifacts (same shader record ids), or null.
    std::shared_ptr<ShaderSet> FindShaderSetOf(
        const TranslationKey& key, const std::array<StageArtifact, GRAPHICS_STAGE_COUNT>& stages);
    /// GPL preload: queues, on the preload workers (background), the compile of the library parts
    /// of the newest shader sets the loaded records stand for (MAX_WARMED_SHADER_SETS), so the
    /// first run-time pipeline of a known shader set links instead of compiling its shaders. The
    /// loading screen and `Flush` of the loader do not wait for it.
    void QueueWarmShaderParts(
        const std::vector<Disk::GraphicsPipelineRecord>& records,
        const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders);
    void WarmShaderParts(const Disk::GraphicsPipelineRecord& record,
                         const std::array<StageArtifact, GRAPHICS_STAGE_COUNT>& artifacts,
                         const TranslationKey& key);
    /// One translation of a TranslationKey in progress at a time: pipelines that share a key (their
    /// states differ only outside what the translation reads) are compiled on several workers at
    /// once, and the later ones wait for the first instead of translating the stages again and
    /// compiling the shader parts of a second set. Guarded by translation_mutex.
    struct TranslationFlight {
        std::mutex mutex;
        u32 users{};
    };
    /// Held while a thread translates the stages of `key`: a second thread asking for the same
    /// key blocks in the constructor until the first is done, then looks the set up again.
    class TranslationHold {
    public:
        TranslationHold(Impl& impl, const TranslationKey& key);
        ~TranslationHold();
        TranslationHold(const TranslationHold&) = delete;
        TranslationHold& operator=(const TranslationHold&) = delete;

    private:
        Impl& impl;
        TranslationKey key;
        std::shared_ptr<TranslationFlight> flight;
        std::unique_lock<std::mutex> lock;
    };
    /// Creates the set's shader modules and library layout once. Throws vk::Exception when the
    /// driver fails (the next pipeline of the set tries again).
    void CreateShaderSetObjects(ShaderSet& set, const VC::DescriptorLayout& descriptors,
                                VkShaderStageFlags push_stages);
    /// Queues the whole compile of a fast-linked pipeline on the optimize workers; the result is
    /// published in `GraphicsPipeline::optimized_handle`.
    void QueueOptimize(std::shared_ptr<GraphicsPipeline> pipeline, std::shared_ptr<ShaderSet> set,
                       const GraphicsFixedState& state, const RenderPassKey& targets);
    void PrepareProgram(const std::array<StageProgram, STAGE_COUNT>& stages,
                        std::shared_ptr<const void> owner);

    // --- compile paths ------------------------------------------------------------------------
    /// EDEN_NX2_NVN_CACHE_LOG: why a pipeline is compiled at run time.
    void LogRuntimeCompile(const GraphicsKey& key, bool had_variants);
    std::shared_ptr<GraphicsPipeline> CompileGraphics(const GraphicsRequest& request,
                                                      const GraphicsKey& key, bool had_variants,
                                                      bool speculative = false);
    std::shared_ptr<ComputePipeline> CompileCompute(const ComputeRequest& request,
                                                    const ComputeKey& key, bool had_variants);

    // --- asynchronous compiles (PipelineCacheOptions::async_compile) ---------------------------
    /// Deep copies of the program bytes of a request, owned by the job.
    struct OwnedProgram {
        std::vector<u8> control;
        std::vector<u8> code;
    };
    struct AsyncGraphicsJob {
        GraphicsRequest request; ///< points into `programs` and `oracle`
        std::array<OwnedProgram, GRAPHICS_STAGE_COUNT> programs;
        std::unique_ptr<TextureOracle> oracle;
        GraphicsKey key;
        bool had_variants{};
        bool speculative{};
        std::shared_ptr<PendingKey> pending;
    };
    struct AsyncComputeJob {
        ComputeRequest request;
        OwnedProgram program;
        std::unique_ptr<TextureOracle> oracle;
        ComputeKey key;
        bool had_variants{};
        std::shared_ptr<PendingKey> pending;
    };
    [[nodiscard]] bool AsyncEnabled(const bool* not_ready) const noexcept {
        return options.async_compile && not_ready != nullptr && async_workers != nullptr;
    }
    /// Counts a "not ready" answer and logs the first few with the program hashes.
    void NoteNotReady(std::string_view what, std::span<const u64> hashes);
    void RunAsyncGraphics(AsyncGraphicsJob& job);
    void RunAsyncCompute(AsyncComputeJob& job);
    /// Drops one claim on `pending` (and the map entry when it was the last), wakes waiters.
    template <typename Map, typename Key>
    void ReleasePending(Map& map, const Key& key, const std::shared_ptr<PendingKey>& pending);
    void AsyncQueued();
    void AsyncDone();

    // --- shader log (nvn_pipeline_log.cpp; every entry is a no-op unless ShaderLog::Enabled()) --
    /// What a failed compile of a key left for the log (the reason and the IR of the stages that
    /// were translated). Written by the compiling thread, read by the replay's first-use hook.
    struct CompileFailure {
        std::string reason;
        std::array<std::string, STAGE_COUNT> ir;
    };
    /// Called when CompileGraphics/CompileCompute produced no pipeline.
    void LogCompileFailed(u64 key_hash, std::array<std::string, STAGE_COUNT>&& ir);
    /// Called with the assembled pipeline before it is published (under `pipeline_mutex`):
    /// attaches what the log keeps. `variant` = number of pipelines the key already has.
    void LogCompileDone(PipelineBase& pipeline, TranslateOutput& translated, double compile_ms,
                        u32 variant);
    /// First-use hooks of the lookups (public wrappers); `result` null = no pipeline.
    void LogGraphicsUse(const GraphicsRequest& request, GraphicsPipeline* result, u64 key_hash,
                        bool not_ready);
    void LogComputeUse(const ComputeRequest& request, ComputePipeline* result, bool not_ready);
    void LogComputeFallback(const ComputeRequest& request, std::string_view reason);
    /// The reason a stage in `hashes` was rejected ("" when none was).
    [[nodiscard]] std::string RejectedReason(std::span<const u64> hashes);

    // --- disk ---------------------------------------------------------------------------------
    [[nodiscard]] bool DiskEnabled() const noexcept {
        return !options.directory.empty() &&
               try_generation_at_init == Common::DiagRequest::TrySwitchGeneration();
    }
    void PostShader(const StageArtifact& artifact);
    void PostGraphicsRecord(const Disk::GraphicsPipelineRecord& record);
    void PostComputeRecord(const Disk::ComputePipelineRecord& record);
    void PostBytes(std::vector<u8> bytes);
    void MaybeWriteVkCache();
    void WriteVkCache();
    std::vector<u8> ReadVkCacheBlob() const;
    void RunLoader(std::stop_token stop);
    /// Counts one handled cached record and reports Build done / total (rate limited, always on
    /// the last one).
    void NotePreloadStep();
    void PreloadGraphics(
        const Disk::GraphicsPipelineRecord& record,
        const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders);
    void PreloadCompute(
        const Disk::ComputePipelineRecord& record,
        const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders);
    void WaitPending(const std::shared_ptr<PendingKey>& pending);
    /// Blocks while the loader is still registering the cached pipelines as pending (false
    /// without waiting when it is done or there is no loader): a draw that misses before that
    /// would compile a pipeline the preload is about to deliver.
    bool WaitLoaderQueued();
    /// Adds a synchronous wait on the preload (loader queue or pending key) to the statistics.
    void NoteWait(std::chrono::steady_clock::time_point start);

    // --- state --------------------------------------------------------------------------------
    const Device& device;
    PipelineCacheOptions options;
    u64 try_generation_at_init{};
    Shader::Profile profile;
    /// Decided once from the device (and PipelineCacheOptions); part of every key form and of
    /// the disk header's profile hash.
    VertexInputMode vertex_input_mode{VertexInputMode::Static};
    bool must_emulate_scaled_formats{};
    /// How the title's compute programs lower their matrix multiplies (decided once, part of the
    /// translation key).
    VideoCommon::NativeTensorLowering tensor_lowering{VideoCommon::NativeTensorLowering::None};
    Shader::HostTranslateInfo host_info;
    Disk::Header disk_header;
    std::filesystem::path disk_path;
    std::filesystem::path vk_cache_path;
    vk::PipelineCache vk_cache;
    std::atomic<u64> shader_generation{};

    std::atomic<u64> stat_loaded{};
    mutable std::atomic<u64> logged_loaded{~u64{0}};
    mutable std::atomic<u64> logged_compiled{~u64{0}};
    std::atomic<u64> stat_compiled{};
    std::atomic<u64> stat_failures{};
    std::atomic<u64> stat_rejected_shaders{};
    std::atomic<u64> stat_disk_written{};
    std::atomic<u64> stat_disk_rejected{};
    std::atomic<u64> stat_retranslated{};
    std::atomic<u64> stat_async_queued{};
    std::atomic<u64> stat_async_failed{};
    std::atomic<u64> stat_async_not_ready{};
    std::atomic<u64> stat_wait_count{};
    std::atomic<u64> stat_wait_ns{};
    std::atomic<u64> stat_wait_max_ns{};
    std::atomic<u64> stat_compile_ns{};   ///< sum of the wall time of every run-time compile
    std::atomic<u64> stat_translate_ns{}; ///< ... of which in translation
    std::atomic<u64> stat_create_ns{};    ///< ... of which creating the Vulkan pipeline
    std::atomic<u64> stat_compile_max_ns{};
    std::atomic<u64> stat_async_peak{};        ///< most compile jobs queued or running at once
    std::atomic<u64> stat_async_first_ticks{}; ///< steady_clock ticks of the first queued job
    std::atomic<u64> stat_async_last_ticks{};  ///< ... and of the latest finished job
    size_t async_worker_count{};

    mutable std::mutex rows_mutex;
    std::map<std::string, u64> rows;
    std::set<std::string> logged;

    std::mutex reject_mutex;
    std::unordered_map<u64, std::string> rejected_stages; ///< stage hash -> row reason

    mutable std::mutex lowered_mutex;
    std::unordered_map<u64, LoweredEntry> lowered;

    std::mutex render_pass_mutex;
    std::unordered_map<RenderPassKey, vk::RenderPass, RenderPassKeyHash> render_passes;

    mutable std::mutex pipeline_mutex;
    std::unordered_map<GraphicsKey, std::vector<std::shared_ptr<GraphicsPipeline>>, GraphicsKeyHash,
                       GraphicsKeyEqual>
        graphics;
    std::unordered_map<ComputeKey, std::vector<std::shared_ptr<ComputePipeline>>, ComputeKeyHash>
        compute;
    std::unordered_map<GraphicsKey, std::shared_ptr<PendingKey>, GraphicsKeyHash, GraphicsKeyEqual>
        pending_graphics;
    std::unordered_map<ComputeKey, std::shared_ptr<PendingKey>, ComputeKeyHash> pending_compute;
    /// Keys whose asynchronous compile produced no pipeline (guarded by pipeline_mutex).
    std::unordered_set<GraphicsKey, GraphicsKeyHash, GraphicsKeyEqual> failed_graphics;
    std::unordered_set<ComputeKey, ComputeKeyHash> failed_compute;

    std::mutex written_mutex;
    std::unordered_set<Disk::ContentId, Disk::ContentIdHash> written_shaders;
    std::unordered_set<Disk::ContentId, Disk::ContentIdHash> written_pipelines;
    Disk::Appender appender; ///< touched by the writer thread only (and the loader before Ready)
    std::atomic<u64> new_since_vk_write{};
    std::atomic<s64> last_vk_write_ticks{};

    std::atomic<bool> stopping{};
    std::mutex loader_mutex;
    std::condition_variable loader_cv;
    bool loader_done{};
    std::atomic<bool>
        loader_queued{}; ///< every cached pipeline is registered in pending_* (or in the maps)
    /// Preload progress for the loading screen (options.load_progress): one step per cached
    /// pipeline record handled (rebuilt, skipped as a duplicate or as unusable).
    bool preload_reporting{};
    std::atomic<size_t> preload_total{};
    std::atomic<size_t> preload_done{};
    std::atomic<u64> preload_report_ticks{};
    DiskWriter writer;
    std::unique_ptr<Common::ThreadWorker> workers;
    std::mutex failure_mutex;
    std::unordered_map<u64, CompileFailure> failure_logs; ///< by key hash (shader log only)
    /// Shader log only (guarded by failure_mutex): rejected keys and fallbacks already logged
    /// (no pipeline object carries their `logged` flag).
    std::unordered_set<u64> logged_rejected_graphics; ///< by GraphicsKey::hash
    std::unordered_set<ComputeKey, ComputeKeyHash> logged_rejected_compute;
    std::vector<std::pair<u64, std::string>> logged_fallbacks; ///< program hash, reason
    /// Asynchronous run-time compiles (null unless options.async_compile).
    std::unique_ptr<Common::ThreadWorker> async_workers;
    std::mutex async_mutex;
    std::condition_variable async_cv;
    u64 async_outstanding{}; ///< jobs queued and not finished (guarded by async_mutex)

    // GPL path (null / empty unless options.graphics_pipeline_library and the device allows it).
    std::unique_ptr<GraphicsPipelineLibraryCache> library_cache;
    /// GPL pipelines declare the DynamicFixedState states dynamic (EDS and EDS2 present).
    bool gpl_dynamic_fixed{};
    std::mutex translation_mutex;
    std::unordered_map<TranslationKey, std::vector<std::shared_ptr<ShaderSet>>, TranslationKeyHash,
                       TranslationKeyEqual>
        translations;
    std::unordered_map<TranslationKey, std::shared_ptr<TranslationFlight>, TranslationKeyHash,
                       TranslationKeyEqual>
        translation_flights;
    std::atomic<u64> stat_prefetch_queued{};
    std::atomic<u64> stat_prefetch_used{};
    std::atomic<u64> stat_gpl_fast_links{};
    std::atomic<u64> stat_gpl_cached_links{};
    std::atomic<u64> stat_gpl_reused{};
    std::atomic<u64> stat_gpl_whole{};
    std::atomic<u64> stat_gpl_optimized{};
    std::atomic<u64> stat_gpl_warmed{}; ///< shader sets whose parts the preload compiled
    std::atomic<u64> stat_gpl_optimize_failed{};
    std::atomic<u64> stat_gpl_optimize_ns{};
    std::atomic<u64> stat_gpl_optimize_slow{}; ///< optimized compiles over 50 ms (background)
    std::atomic<u64> stat_prepared{};
    /// Optimize jobs and program preparations queued and not finished (guarded by gpl_mutex).
    std::mutex gpl_mutex;
    std::condition_variable gpl_cv;
    u64 gpl_outstanding{};
    /// Compiles the shader parts of one link side by side (the calling thread waits for them).
    std::unique_ptr<Common::ThreadWorker> part_workers;
    /// Whole compiles of fast-linked pipelines (background priority).
    std::unique_ptr<Common::ThreadWorker> optimize_workers;
    /// Lowering of the programs the game creates (background priority).
    std::unique_ptr<Common::ThreadWorker> prepare_workers;

    std::jthread loader;
};

/// Shader log: the reason of the last failure noted on this thread (`NoteRow`, only kept while
/// the log is on); the compile that failed reads it.
std::string& LastIssueSlot();
/// Shader log: the key a compute pipeline is known by before its texture-type variant is
/// known (graphics: `GraphicsKey::hash`).
u64 LogBaseKey(const ComputeKey& key);

/// Armed for the duration of a compile; if it ends without a pipeline, hands the reason (and the
/// IR of the stages translated) to the log. Fill `ir` before returning on a failed translation.
struct CompileFailureNote {
    CompileFailureNote(PipelineCache::Impl& impl_, u64 key_hash_)
        : impl{impl_}, key_hash{key_hash_}, armed{ShaderLog::Enabled()} {
        if (armed) {
            LastIssueSlot().clear();
        }
    }
    ~CompileFailureNote() {
        if (armed) {
            impl.LogCompileFailed(key_hash, std::move(ir));
        }
    }
    CompileFailureNote(const CompileFailureNote&) = delete;
    CompileFailureNote& operator=(const CompileFailureNote&) = delete;

    PipelineCache::Impl& impl;
    u64 key_hash;
    bool armed;
    std::array<std::string, STAGE_COUNT> ir;
};
} // namespace Vulkan::Nvn
