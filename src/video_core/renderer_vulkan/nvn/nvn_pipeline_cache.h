// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Shader translation, pipeline creation and the on-disk cache of the NVN HLE replay.
//
// The replay (nvn_exec.cpp / nvn_draw.cpp) tracks the NVN state of a draw, fills a
// GraphicsRequest and asks the cache for a pipeline. The cache owns everything between the guest
// shader bytes and a VkPipeline:
//   LowerNvnProgram -> NvnEnvironment + TranslateNvnStage -> ConvertLegacyToGeneric + EmitSPIRV
//   -> descriptor layout (BuildDescriptorLayout) -> VkShaderModule / VkPipeline.
// Compute pipelines go through the same module (GetCompute).
//
// Identity (nothing in a key is a serial: serials change between launches):
//   * a stage is identified by HashProgramStage(control, code) (content hash);
//   * a pipeline by (stage hashes, GraphicsFixedState bytes, RenderTargetFormats, texture type
//     queries). The type queries are the texture types the translated shaders asked about
//     (NvnEnvironment::HandleRecords + the oracle's answers): translation depends on them, so two
//     draws with the same state and programs but a 2D vs a 2D-array texture behind a handle are
//     different pipelines ("variants").
//   * the vertex layout (formats, offsets, streams, strides, divisors) is part of the key only in
//     VertexInputMode::Static; with VK_EXT_vertex_input_dynamic_state (VertexInputMode::Dynamic)
//     the key holds just the shader-visible type class per location and the replay sets the
//     layout per draw (see GraphicsFixedState).
//
// Threading: GetGraphics / GetCompute / GetRenderPass are called from the replay thread only. The
// disk-cache preload, the asynchronous compiles (`PipelineCacheOptions::async_compile`) and the
// cache writer use worker threads internally. A worker never sees replay state: an asynchronous
// request carries deep copies of the program bytes and a frozen oracle (TextureOracle::Freeze).

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/nvn/shader/nvn_descriptor_layout.h"
#include "video_core/nvn/shader/nvn_program_map.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
class LoadProgressSink;
}

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

namespace VC = VideoCommon::Nvn;

inline constexpr u32 MAX_COLOR_TARGETS = 8;
inline constexpr u32 MAX_VERTEX_ATTRIBUTES = 32;
inline constexpr u32 MAX_VERTEX_STREAMS = 16;
inline constexpr u32 MAX_VIEWPORTS = 16;
/// NVNshaderStage numbering: 0 vertex, 1 fragment, 2 geometry, 3 tess control, 4 tess
/// evaluation, 5 compute, 6 task, 7 mesh.
inline constexpr u32 STAGE_COMPUTE = 5;
inline constexpr u32 STAGE_TASK = 6;
inline constexpr u32 STAGE_MESH = 7;
/// Stage slots of a graphics request / key, indexed by NVNshaderStage. Slot 5 (compute) is never
/// set. A pipeline has a vertex program (vertex pipeline) or a mesh program (task/mesh pipeline,
/// task optional), never both.
inline constexpr u32 GRAPHICS_STAGE_COUNT = 8;
/// Every NVNshaderStage, compute included.
inline constexpr u32 STAGE_COUNT = 8;
/// Highest constant buffer bank + 1 a translated shader can use (Shader::Info::MAX_CBUFS).
inline constexpr u32 LOG_MAX_CBUFS = static_cast<u32>(Shader::Info::MAX_CBUFS);

/// Content hash of the code and control fields used by ParseControl, excluding the section's own
/// length and unused bytes. Includes compute descriptor counts and offsets. Never returns 0
/// (0 means "stage absent" in requests).
[[nodiscard]] u64 HashProgramStage(std::span<const u8> control, std::span<const u8> code);

/// How a pipeline receives the vertex layout. Chosen once per cache from the device
/// (VK_EXT_vertex_input_dynamic_state supported -> Dynamic) and part of the disk cache identity.
enum class VertexInputMode : u32 {
    /// Attribute words, strides and divisors are baked into every VkPipeline (and its key).
    Static = 0,
    /// The pipeline declares VK_DYNAMIC_STATE_VERTEX_INPUT_EXT and no bindings; the key keeps
    /// only `GraphicsFixedState::vertex_types`.
    Dynamic = 1,
};

/// Everything of the NVN fixed-function state a graphics pipeline is created from. Plain data, all
/// values raw NVN values (the packed state-object words of state_layouts.h, NVN enums), laid out
/// without padding so that the bytes are the key (hash with CityHash64, compare with memcmp).
/// Zero the whole struct before filling it. Fields only matter when the pipeline uses them; the
/// replay zeroes the ones that do not apply (e.g. `point_size_bits` unless the topology is
/// POINTS) so that pipelines that differ only in state their shaders ignore are shared.
///
/// Vertex layout: a draw request carries the raw words (`attributes`, `stream_stride`,
/// `stream_divisor`) and leaves `vertex_types` zero. The cache keys on the key form built by
/// `MakeKeyState` (nvn_pipeline_state.h): Static keeps the words and zeroes `vertex_types`;
/// Dynamic fills `vertex_types` from the words and zeroes the words, so draws that differ only in
/// formats of one type class, offsets, streams, strides or divisors share their pipeline.
struct GraphicsFixedState {
    u32 topology;           ///< NVNdrawPrimitive
    u32 patch_vertices;     ///< patches only (0 otherwise)
    u32 polygon;            ///< PolygonState word
    u32 color;              ///< ColorState word (blend enable mask, logic op, alpha test func)
    u32 channel_mask;       ///< ChannelMaskState word (4 bits per target)
    u32 primitive_restart;  ///< 0/1; `restart_index` only matters for VK when the index type is fixed by the draw
    u32 restart_index;
    u32 sample_mask;        ///< SetSampleMask (0xFFFFFFFF default)
    u32 depth_clamp;        ///< 0/1
    u32 rasterizer_discard; ///< 0/1
    u32 depth_mode;         ///< NVNdepthMode: 0 near = -w (GL), 1 near = 0
    u32 window_origin;      ///< NVNwindowOriginMode: 0 lower left, 1 upper left
    u32 point_size_bits;    ///< f32 bits, only for POINTS topology or when a geometry stage exists
    u32 alpha_ref_bits;     ///< f32 bits, only when the alpha test func is not ALWAYS
    /// SET_PROVOKING_VERTEX inverted, so that zero is NVN's default: 0 = the last vertex (the SDK's
    /// initial GPU state), 1 = the first (nvnCommandBufferSetProvokingVertexMode(.., 0)).
    u32 provoking_first;
    u32 pad1;
    u64 depth_stencil;      ///< DepthStencilState word
    std::array<u64, MAX_COLOR_TARGETS> blend; ///< BlendState word per target (defaults if never bound)
    std::array<u8, 16> multisample;           ///< MultisampleState bytes (first 10 used)
    /// va_st words (fmt:8 offset:19 stream:5); attribute k feeds shader location k. fmt 0 = unused.
    /// Draw request and Static key form only; all zero in the Dynamic key form.
    std::array<u32, MAX_VERTEX_ATTRIBUTES> attributes;
    std::array<u32, MAX_VERTEX_STREAMS> stream_stride;  ///< same rule as `attributes`
    std::array<u32, MAX_VERTEX_STREAMS> stream_divisor; ///< same rule as `attributes`
    /// Dynamic key form only (zero otherwise): `Shader::AttributeType` of location k as the
    /// vertex shader is translated for it (Float, SignedInt, UnsignedInt, SignedScaled,
    /// UnsignedScaled, Disabled).
    std::array<u8, MAX_VERTEX_ATTRIBUTES> vertex_types;
};
static_assert(sizeof(GraphicsFixedState) % 8 == 0);
static_assert(std::is_trivially_copyable_v<GraphicsFixedState>);

/// Vulkan description of the vertex layout of one draw (what `vkCmdSetVertexInputEXT` takes, and
/// what the Static pipeline is created from): the attributes of `attributes_read` that have a
/// vertex format, and the streams they use, with the instancing rate and divisor of each stream
/// (`BuildVertexInput`, nvn_pipeline_state.h). Fixed storage: no allocation.
struct VertexInput {
    u32 binding_count{};
    u32 attribute_count{};
    u32 used_streams{}; ///< bit s = stream s has a binding (a vertex buffer must be bound)
    std::array<VkVertexInputBindingDescription2EXT, MAX_VERTEX_STREAMS> bindings;
    std::array<VkVertexInputAttributeDescription2EXT, MAX_VERTEX_ATTRIBUTES> attributes;
};

/// Attachment formats and sample count of the render pass a draw runs in. VK_FORMAT_UNDEFINED =
/// no attachment in that slot.
struct RenderPassKey {
    std::array<s32, MAX_COLOR_TARGETS> color{}; ///< VkFormat
    s32 depth{};                                ///< VkFormat
    u32 samples{1};                             ///< VkSampleCountFlagBits (1, 2, 4, 8)
    [[nodiscard]] bool operator==(const RenderPassKey&) const = default;
};

/// The subpass dependencies of every render pass the replay creates: the two ALL_COMMANDS external
/// ones (before and after the pass) and the by-region self-dependency; with `sync_diet`
/// (PipelineCacheOptions::sync_diet) only the self-dependency, the explicit barrier between ops
/// does the rest.
[[nodiscard]] std::span<const VkSubpassDependency> RenderPassDependencies(bool sync_diet);

/// The rasterization and depth/stencil values of a pipeline that VK_EXT_extended_dynamic_state and
/// VK_EXT_extended_dynamic_state2 can set per draw (`MakeDynamicFixedState`,
/// nvn_pipeline_state.h). Pipelines of the default path have them static. Pipelines of the
/// VK_EXT_graphics_pipeline_library path (`PipelineCacheOptions::graphics_pipeline_library`)
/// declare them dynamic when the device has both extensions, so their shader libraries do not
/// depend on them, and the replay sets these values after binding such a pipeline.
struct DynamicFixedState {
    struct StencilFace {
        VkStencilOp fail{VK_STENCIL_OP_KEEP};
        VkStencilOp pass{VK_STENCIL_OP_KEEP};
        VkStencilOp depth_fail{VK_STENCIL_OP_KEEP};
        VkCompareOp compare{VK_COMPARE_OP_ALWAYS};
        [[nodiscard]] bool operator==(const StencilFace&) const = default;
    };
    VkCullModeFlags cull_mode{VK_CULL_MODE_NONE};
    VkFrontFace front_face{VK_FRONT_FACE_COUNTER_CLOCKWISE};
    bool depth_bias_enable{};
    bool depth_test_enable{};
    bool depth_write_enable{};
    VkCompareOp depth_compare_op{VK_COMPARE_OP_ALWAYS};
    bool stencil_test_enable{};
    StencilFace front;
    StencilFace back;
    [[nodiscard]] bool operator==(const DynamicFixedState&) const = default;
};
struct PipelineBase;
namespace Disk {
struct ShaderRecord;
} // namespace Disk

/// Shader log (video_core/shader_log.h): the replay's description of what a pipeline reads at
/// run time. Only called when the log is on, on the first use of a pipeline (or fallback).
class InputsProvider {
public:
    virtual ~InputsProvider() = default;
    /// A JSON object text `{"cbufs":[...],"textures":[...],...}` (the content of
    /// `p<key>.inputs.json`) of the draw/dispatch that uses `pipeline` right now. `pipeline` is
    /// null for a program that was replaced by a stand-in (no pipeline exists).
    [[nodiscard]] virtual std::string DescribeInputs(const PipelineBase* pipeline) const = 0;
};

/// The texture the replay will bind behind a handle word of a shader (what translation needs).
using TextureInfo = VC::NvnTextureInfo;

/// Answers "what is behind this handle word right now". Implemented by the replay from its
/// binding state; consulted while a stage is translated and, for cached pipelines, to check that
/// a cached variant is still valid for the draw.
class TextureOracle {
public:
    virtual ~TextureOracle() = default;
    /// `stage` is the NVNshaderStage the word was read in. Unbound or unknown: return the default
    /// (Color2D, A8B8G8R8_UNORM, not integer).
    [[nodiscard]] virtual TextureInfo Query(u32 stage, const VC::NvnHandleRecord& record) const = 0;
    /// An independent copy that answers every query of the stages in `stage_mask` (bit k = stage
    /// k) exactly as this oracle does now and stays valid after the replay state moved on; safe
    /// to use from any thread. nullptr when the oracle cannot be frozen: the cache then compiles
    /// synchronously.
    [[nodiscard]] virtual std::unique_ptr<TextureOracle> Freeze(u32 stage_mask) const {
        (void)stage_mask;
        return nullptr;
    }
};

/// One program stage as the replay holds it (borrowed for the duration of the call).
struct StageProgram {
    u64 hash{};                              ///< HashProgramStage; 0 = stage absent
    const std::vector<u8>* control{};
    const std::vector<u8>* code{};
};

struct GraphicsRequest {
    std::array<StageProgram, GRAPHICS_STAGE_COUNT> stages; ///< indexed by NVNshaderStage
    GraphicsFixedState state{};
    RenderPassKey targets;
    const TextureOracle* oracle{};
    const InputsProvider* inputs{}; ///< shader log only, see InputsProvider
};

/// Whether a stage of `request` stores to memory: the DoesGlobalStore bit (word 0 bit 16) of the
/// shader program header that opens every graphics stage's code. Known before translation, so a
/// draw can be judged while its pipeline is still compiling.
[[nodiscard]] bool StoresGlobalMemory(const GraphicsRequest& request) noexcept;

/// With asynchronous shaders, whether a draw whose pipeline is still compiling may be skipped
/// instead of waiting for the pipeline. The emulated path's rule (vk_pipeline_cache.cpp
/// BuiltPipeline and CanSkipDraw): a draw that stores to memory waits (skipping it loses state,
/// not a frame of geometry); else a draw with a depth target may be skipped (games do not render
/// one-time results with depth); else only a draw of more than 6 elements may be. A fullscreen
/// pass of 3 to 6 vertices, or an indirect or mesh draw (`element_count` 0), may build a texture
/// the game never draws again.
[[nodiscard]] constexpr bool MaySkipUnbuiltDraw(bool stores_memory, bool depth_target,
                                                u32 element_count) noexcept {
    if (stores_memory) {
        return false;
    }
    return depth_target || element_count > 6;
}

/// A texture-type fact a translation depended on (see "Identity" above).
struct TextureQuery {
    u32 stage{};
    VC::NvnHandleRecord record;
    TextureInfo info;
};

/// Per translated stage data the replay needs at draw time.
struct PipelineStage {
    bool present{};
    u32 stage{}; ///< NVNshaderStage
    /// Words of the program's constant segment (c1). Dynamic reads of c1 that survived folding
    /// are served from a buffer with this content; null when the stage has no ProgramConstants
    /// descriptor.
    std::shared_ptr<const std::vector<u32>> program_constants;
    /// Compute only: workgroup size from the control section.
    std::array<u32, 3> workgroup_size{};
    u32 shared_memory_size{};
};
/// What the shader log keeps of a pipeline (set at creation, only while the log is on).
struct PipelineLogInfo {
    /// The translated stages (by NVNshaderStage; null when absent), with their SPIR-V.
    std::array<std::shared_ptr<const Disk::ShaderRecord>, STAGE_COUNT> records;
    /// `Shader::Info::constant_buffer_used_sizes` of each stage as translated; valid for the
    /// stages in `cbuf_known` (bit k = NVNshaderStage k). Pipelines from the disk cache have none.
    std::array<std::array<u32, LOG_MAX_CBUFS>, STAGE_COUNT> cbuf_used{};
    u32 cbuf_known{};
    /// Text dump of the IR each stage was emitted from (translated this session only).
    std::array<std::string, STAGE_COUNT> ir;
    double compile_ms{}; ///< translate + create; 0 for a pipeline from the disk cache
    u32 variant{};       ///< 0 = first pipeline of its key, k = k-th texture-type variant
    bool from_disk{};
};


/// A created pipeline. Immutable after creation (but for a GPL pipeline's `optimized` handle,
/// see GraphicsPipeline); shared_ptr so in-flight submissions keep it alive.
struct PipelineBase {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::DescriptorSetLayout set_layout; ///< null when the pipeline has no descriptors
    /// Set BINDLESS_SET of the layout when a stage samples through the bindless table.
    vk::DescriptorSetLayout bindless_layout;
    /// The descriptor set is pushed with vkCmdPushDescriptorSetKHR; otherwise it is allocated.
    bool push_descriptors{};
    /// Descriptor bindings of every stage in pipeline order, numbered as in the SPIR-V.
    VC::DescriptorLayout descriptors;
    /// Push-constant range size in bytes (rescaling / render-area words), 0 if none.
    u32 push_constant_size{};
    /// Stages of the push-constant range (VK_SHADER_STAGE_ALL_GRAPHICS, task|mesh|fragment for a
    /// task/mesh pipeline, compute); what vkCmdPushConstants must name.
    VkShaderStageFlags push_constant_stages{};
    /// Stage 5 pipelines: the control-derived compute driver bank layout (default otherwise).
    VC::ComputeDriverLayout compute_layout;
    /// Texture types the shaders were translated for; revalidated with the oracle per draw.
    std::vector<TextureQuery> queries;
    /// True when every query reads only HandleBank (c2) of a stage < 8.
    bool queries_handle_bank_only{true};
    /// Bitmask of NVNshaderStage values present in `queries`.
    u32 query_stage_mask{0};
    /// Shader log: set at creation while the log is on, otherwise null. `logged` is claimed by
    /// the first lookup that returns the pipeline to a draw/dispatch after the log started.
    std::shared_ptr<PipelineLogInfo> log;
    std::atomic<bool> logged{false};
    std::array<PipelineStage, STAGE_COUNT> stages;
    /// Revalidation: true when `oracle` still answers every query as it did at translation.
    [[nodiscard]] bool QueriesMatch(const TextureOracle& oracle) const;
};

struct GraphicsPipeline : PipelineBase {
    VkRenderPass render_pass{VK_NULL_HANDLE}; ///< owned by the cache
    /// Number of viewports/scissors the pipeline declares (1, or MAX_VIEWPORTS when a stage
    /// writes the viewport index).
    u32 viewport_count{1};
    /// Vertex attribute locations the vertex stage reads (bit k = location k).
    u32 attributes_read{};
    /// The pipeline writes attachment k (bit k) - fragment outputs present.
    u32 color_outputs_written{};
    /// The pipeline declares VK_DYNAMIC_STATE_VERTEX_INPUT_EXT: the replay must
    /// vkCmdSetVertexInputEXT before drawing with it (`BuildVertexInput`).
    bool dynamic_vertex_input{};
    /// Task/mesh pipeline (VK_EXT_mesh_shader): no vertex input or input assembly; drawn with
    /// vkCmdDrawMeshTasks*EXT only.
    bool mesh{};
    /// The pipeline declares the states of DynamicFixedState dynamic (GPL path): the replay sets
    /// `fixed_values` after binding it.
    bool dynamic_fixed{};
    DynamicFixedState fixed_values;
    /// GPL path: `pipeline` was fast-linked from VK_EXT_graphics_pipeline_library parts.
    /// `optimized` is the same pipeline compiled whole in the background (same layout, render
    /// pass and dynamic states); the optimize worker writes it once, then publishes its handle in
    /// `optimized_handle`, and draws recorded from then on bind it. `pipeline` stays alive (draws
    /// in flight may use it) until the pipeline is destroyed.
    bool fast_linked{};
    std::atomic<bool> prefetched_unused{};
    vk::Pipeline optimized;
    std::atomic<VkPipeline> optimized_handle{VK_NULL_HANDLE};

    /// The handle draws bind: the optimized pipeline once it is published, else `pipeline`.
    [[nodiscard]] VkPipeline Handle() const noexcept {
        const VkPipeline best = optimized_handle.load(std::memory_order_acquire);
        return best != VK_NULL_HANDLE ? best : *pipeline;
    }
};

struct ComputeRequest {
    StageProgram program;
    const TextureOracle* oracle{};
    const InputsProvider* inputs{}; ///< shader log only, see InputsProvider
    /// Optional overrides of the environment (compute local/shared memory).
    VC::NvnEnvironmentOptions environment;
};

struct ComputePipeline : PipelineBase {};

struct PipelineCacheOptions {
    /// Directory of the per-title files (`nvn_pipelines.k<key>.bin`, `nvn_pipelines.vkcache`);
    /// empty disables the disk cache (the use_disk_shader_cache setting: the renderer gives no
    /// directory when it is off).
    std::filesystem::path directory;
    /// The use_vulkan_driver_pipeline_cache setting: the driver's pipeline cache is loaded from
    /// and saved to `nvn_pipelines.vkcache` in `directory`. Off: that file is never read,
    /// written or created (the driver's cache still serves this run in memory, and the
    /// translated shaders and pipeline records go to their own file as before).
    bool driver_cache_file{true};
    /// Build id of this binary (Common::g_scm_rev). Recorded in new cache files for diagnostics
    /// only: the cache is valid for the translation key, not for a revision.
    std::string revision;
    /// Identity of the translation sources (Disk::TranslationEpoch()); empty = this binary's.
    /// Tests set it to stand for a build whose translation changed.
    std::string translation_epoch;
    u64 title_id{};
    /// Translate and compile cached entries on worker threads at construction.
    bool preload{true};
    /// Use VK_EXT_vertex_input_dynamic_state when the device has it. False forces the static
    /// path (tests, diagnostics).
    bool allow_dynamic_vertex_input{true};
    /// A lookup that misses compiles on worker threads instead of the calling thread
    /// (Settings use_asynchronous_shaders): callers that pass `not_ready` to GetGraphics /
    /// GetCompute get "not ready" until the compile finished and skip the draw/dispatch, and a
    /// lookup that meets a preload still in flight does not wait for it either.
    bool async_compile{false};
    /// Compile upcoming graphics state on workers while replay builds the current pipeline.
    bool prefetch_graphics{false};
    /// The nvn_pipeline_library setting: graphics pipelines are built from
    /// VK_EXT_graphics_pipeline_library parts (vertex input, pre-rasterization shaders, fragment
    /// shader, fragment output) that are compiled once and fast-linked, then compiled whole in
    /// the background and swapped in (GraphicsPipeline::Handle). With `async_compile` the parts
    /// and the link of a draw that may be skipped are built on the async workers like any other
    /// miss; without it (or for a draw that must not be skipped) on the calling thread. Draws
    /// whose state differs only in what the shader translation does not read reuse the
    /// translated stages and the shader parts. Ignored when the device does not have the
    /// extension with fast linking (vulkan_device.cpp enables it only when a renderer asks for it).
    bool graphics_pipeline_library{false};
    /// Settings nvn_sync_diet: the render passes (GetRenderPass) carry no external dependencies
    /// (RenderPassDependencies); the replay's explicit barrier between a pass and the next op is
    /// the one sync at the boundary, and passes begun outside BeginPass flush it themselves
    /// (DrawHost::HostBarrierBeforePass).
    bool sync_diet{false};
    /// The status bar's "Building: N shader(s)" counter (GPU().ShaderNotify()): every compile
    /// handed to an async worker counts from the moment it is queued until it finished, failed
    /// or was dropped at shutdown. Null (tests, headless): nothing is counted. The disk preload
    /// is not counted (the loading screen shows it); an inline compile blocks its draw and is not
    /// a background build.
    VideoCore::ShaderNotify* shader_notify{};
    /// The loading screen's progress (GPU().LoadProgress()): the disk preload reports Prepare, then
    /// Build with done / total pipelines (the same stages and counts as the rasterizer's
    /// LoadDiskResources), then Complete; nothing when the cache file holds no pipelines. Null:
    /// nothing is reported.
    VideoCore::LoadProgressSink* load_progress{};
};

struct PipelineCacheStats {
    u64 loaded{};              ///< pipelines rebuilt from the disk cache
    u64 compiled_at_runtime{}; ///< pipelines translated/compiled because a draw needed them
    u64 translation_failures{};
    u64 rejected_shaders{};    ///< `Native ... shader rejected` equivalents (distinct stage hashes)
    u64 disk_entries_written{};
    u64 disk_rejected{};       ///< cache files discarded (format, translation key, corruption)
    u64 variants_retranslated{}; ///< same stages+state, different texture types
    u64 async_queued{};        ///< compiles handed to the workers
    u64 async_failed{};        ///< worker compiles that produced no pipeline (cached as rejected)
    u64 async_not_ready{};     ///< lookups answered "not ready" (a draw/dispatch was skipped)
    u64 preload_waits{};       ///< synchronous lookups that waited for the preload or a pending key
    u64 preload_wait_ns{};     ///< ... and the total time they waited
    u64 prefetch_queued{};
    u64 prefetch_used{};
    u64 gpl_fast_links{};      ///< GPL path: pipelines fast-linked from libraries
    u64 gpl_cached_links{};    ///< ... of which every library was compiled already
    u64 gpl_translations_reused{}; ///< GPL path: pipelines built from an earlier translation
    u64 gpl_whole{};           ///< GPL path: pipelines the libraries cannot express, compiled whole
    u64 gpl_optimized{};       ///< whole compiles published in place of a fast-linked pipeline
    u64 gpl_shader_sets_warmed{}; ///< GPL path: shader sets whose library parts the preload compiled
    u64 programs_prepared{};   ///< program stages lowered when the game created the program
    u64 graphics_pipelines{};  ///< pipelines held now
    u64 compute_pipelines{};
    u64 lowered_programs{}; ///< lowered stage programs held now
};

/// Thread-safety: see the file comment.
class PipelineCache {
public:
    PipelineCache(const Device& device, PipelineCacheOptions options);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    /// Returns the pipeline for the request, translating and compiling it (synchronously) when it
    /// is not cached. Null when a stage cannot be translated (counted, logged once per stage
    /// hash) or the pipeline cannot be created. The returned pipeline stays valid for as long as
    /// the shared_ptr lives.
    ///
    /// With `PipelineCacheOptions::async_compile` and a non-null `not_ready`, a miss is compiled
    /// on the workers instead: the call returns null and sets `*not_ready` (not a failure; ask
    /// again with a later draw). A key whose compile failed answers null without `*not_ready`
    /// (counted as rejected). Without `not_ready` the lookup is always synchronous, on the GPL
    /// path (`UsesGraphicsPipelineLibrary`) too: it translates, compiles the missing library
    /// parts and fast-links on the calling thread.
    [[nodiscard]] std::shared_ptr<GraphicsPipeline> GetGraphics(const GraphicsRequest& request,
                                                                bool* not_ready = nullptr,
                const std::function<void(u64)>& look_ahead = {});
    void PrefetchGraphics(const GraphicsRequest& request, u64 current_key);
    [[nodiscard]] std::shared_ptr<ComputePipeline> GetCompute(const ComputeRequest& request,
                                                              bool* not_ready = nullptr);
    /// Shader log only (call when `ShaderLog::Enabled()`): the compute program of `request` was
    /// not compiled because the replay substituted its own implementation (`reason`, e.g. the
    /// NDSR upscale stand-in). Logs the program (bytes, control) and a Fallback pipeline record
    /// once per program and reason.
    void LogComputeFallback(const ComputeRequest& request, std::string_view reason);

    /// GPL path only (a no-op otherwise): the game created a program. Its stages (by
    /// NVNshaderStage, pointing into `owner`, which the call keeps alive until done) are lowered
    /// on a background thread now, so the draw that first uses the program does not lower them.
    void PrepareProgram(const std::array<StageProgram, STAGE_COUNT>& stages,
                        std::shared_ptr<const void> owner);
    /// The cache fast-links graphics pipelines from VK_EXT_graphics_pipeline_library parts.
    [[nodiscard]] bool UsesGraphicsPipelineLibrary() const noexcept;
    /// The render passes have no external dependencies (PipelineCacheOptions::sync_diet).
    [[nodiscard]] bool SyncDiet() const noexcept;

    /// The render pass pipelines of this key are created for and that the replay begins with
    /// (attachments LOAD/STORE, color attachments in COLOR_ATTACHMENT_OPTIMAL, depth/stencil in
    /// DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the whole pass). Owned by the cache.
    [[nodiscard]] VkRenderPass GetRenderPass(const RenderPassKey& key);

    /// The Shader::Profile / HostTranslateInfo built from the device (as vk_pipeline_cache.cpp).
    [[nodiscard]] const Shader::Profile& Profile() const noexcept;
    /// Dynamic when the device supports VK_EXT_vertex_input_dynamic_state (and the options allow
    /// it), regardless of the legacy renderer's `vertex_input_dynamic_state` user setting.
    [[nodiscard]] VertexInputMode VertexInput() const noexcept;

    /// Counts `reason` in the unsupported rows and logs it once (the replay reports per-draw
    /// vertex layout problems of the Dynamic mode through this).
    void NoteIssue(const std::string& reason);

    /// Blocks until the preload and the asynchronous compiles finished and the queued disk writes
    /// are on disk (shutdown, tests).
    void Flush();

    /// Replay thread only, with no concurrent lookups: waits for cache workers, then drops all
    /// shader translations, pipelines, rejections, library parts and driver compilation data.
    /// Render passes and disk files stay valid. In-flight work must retain its pipeline
    /// shared_ptrs.
    void ClearShaders();
    /// Changes after ClearShaders; local pipeline memos must be invalidated before another draw.
    [[nodiscard]] u64 ShaderGeneration() const noexcept;

    [[nodiscard]] PipelineCacheStats Stats() const;

    /// One line per distinct reason a stage/pipeline could not be built, with counts
    /// (unsupported-<title>.tsv rows, `where` = shader).
    struct Row {
        std::string reason;
        u64 count;
    };
    void CollectRows(std::vector<Row>& out) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::Nvn
