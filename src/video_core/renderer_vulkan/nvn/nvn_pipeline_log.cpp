// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Shader logging mode (video_core/shader_log.h) of the NVN HLE pipeline cache.
//
// What lands in the log, per stage hash (once per session) and per pipeline (once per key):
//   * the program exactly as the game handed it over: the BNSH code area (graphics: it starts
//     with the 0x80-byte SPH, then the instructions, then the program constants) and the whole
//     control section, byte for byte, from the request that first used the stage;
//   * the pipeline: stage hashes, outcome (translated this session / rebuilt from the disk cache /
//     rejected with the cache's reason), translate + create milliseconds, the NVN state it was
//     translated and created with;
//   * our side of each stage next to it: the IR the SPIR-V was emitted from and the SPIR-V
//     (taken from the first pipeline that used the stage);
//   * the runtime inputs, described by the replay at the first use (InputsProvider).
//
// The hooks run on the replay thread when a lookup returns, once per pipeline object (an atomic
// flag in PipelineBase; a rejected key or a fallback in a per-cache set), so a repeated draw
// costs nothing; the compile threads only keep the artifacts (`PipelineLogInfo`, `failure_logs`)
// while the log is on.

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_impl.h"
#include "video_core/shader_log_title.h"

namespace Vulkan::Nvn {

namespace {

/// Salt that keeps the key of a rejected pipeline apart from a pipeline of the same base key that
/// is created later (a retry after a transient failure).
constexpr u64 REJECTED_SALT = 0xFA11ED5EADBEEF01ull;

std::string_view LogStageName(u32 nvn_stage) {
    switch (nvn_stage) {
    case STAGE_VERTEX:
        return "vs";
    case STAGE_FRAGMENT:
        return "fs";
    case STAGE_GEOMETRY:
        return "gs";
    case STAGE_TESS_CONTROL:
        return "tcs";
    case STAGE_TESS_EVAL:
        return "tes";
    default:
        return "cs";
    }
}

std::string_view TopologyName(u32 topology) {
    constexpr std::array<std::string_view, 15> NAMES{
        "points",         "lines",          "line_loop",
        "line_strip",     "triangles",      "triangle_strip",
        "triangle_fan",   "quads",          "quad_strip",
        "polygon",        "lines_adjacency", "line_strip_adjacency",
        "triangles_adjacency", "triangle_strip_adjacency", "patches"};
    return topology < NAMES.size() ? NAMES[topology] : "unknown";
}

std::string_view FaceName(u32 face) {
    constexpr std::array<std::string_view, 4> NAMES{"none", "front", "back", "front_and_back"};
    return face < NAMES.size() ? NAMES[face] : "unknown";
}

std::string Hex(std::span<const u8> bytes) {
    static constexpr char DIGITS[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const u8 byte : bytes) {
        out.push_back(DIGITS[byte >> 4]);
        out.push_back(DIGITS[byte & 0xF]);
    }
    return out;
}

std::vector<u8> BytesOf(std::string_view text) {
    return std::vector<u8>(text.begin(), text.end());
}

std::vector<u8> BytesOf(const std::vector<u32>& words) {
    std::vector<u8> bytes(words.size() * sizeof(u32));
    if (!words.empty()) {
        std::memcpy(bytes.data(), words.data(), bytes.size());
    }
    return bytes;
}

std::string QueriesJson(const PipelineBase& pipeline) {
    std::string out = "[";
    for (const TextureQuery& query : pipeline.queries) {
        if (out.size() > 1) {
            out += ',';
        }
        out += fmt::format(
            "{{\"stage\":{},\"bank\":{},\"offset\":{},\"type\":{},\"format\":{},\"is_integer\":{}}}",
            ShaderLog::Quote(LogStageName(query.stage)), query.record.bank, query.record.offset,
            static_cast<u32>(query.info.type), static_cast<u32>(query.info.format),
            query.info.is_integer);
    }
    return out + "]";
}

std::string GraphicsStateJson(const GraphicsFixedState& state, const RenderPassKey& targets,
                              VertexInputMode mode, const GraphicsPipeline* pipeline,
                              u64 base_key, u32 variant) {
    std::string out = fmt::format("{{\"kind\":\"graphics\",\"key_base\":\"{:016x}\",\"variant\":{}",
                                  base_key, variant);
    out += fmt::format(
        ",\"topology\":{{\"value\":{},\"name\":\"{}\"}},\"patch_vertices\":{}", state.topology,
        TopologyName(state.topology), state.patch_vertices);
    out += ",\"render_targets\":{\"color_vk_format\":[";
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        out += fmt::format("{}{}", slot != 0 ? "," : "", targets.color[slot]);
    }
    out += fmt::format("],\"depth_vk_format\":{},\"samples\":{}}}", targets.depth, targets.samples);
    out += fmt::format(
        ",\"polygon\":{{\"word\":\"{:#x}\",\"cull\":\"{}\",\"front_face\":{},\"mode\":{},"
        "\"offset_enables\":{}}}",
        state.polygon, FaceName(VC::PolygonState::CullFace(state.polygon)),
        VC::PolygonState::FrontFace(state.polygon), VC::PolygonState::PolygonMode(state.polygon),
        VC::PolygonState::OffsetEnables(state.polygon));
    out += fmt::format(
        ",\"color\":{{\"word\":\"{:#x}\",\"blend_enable_mask\":\"{:#x}\",\"logic_op\":{},"
        "\"alpha_func\":{}}}",
        state.color, VC::ColorState::BlendEnableMask(state.color),
        VC::ColorState::LogicOp(state.color), VC::ColorState::AlphaFunc(state.color));
    out += fmt::format(",\"channel_mask\":\"{:#x}\"", state.channel_mask);
    const u64 ds = state.depth_stencil;
    out += fmt::format(
        ",\"depth_stencil\":{{\"word\":\"{:#x}\",\"depth_test\":{},\"depth_write\":{},"
        "\"depth_func\":{},\"stencil_test\":{},\"front\":{{\"func\":{},\"fail\":{},\"depth_fail\":{},"
        "\"depth_pass\":{}}},\"back\":{{\"func\":{},\"fail\":{},\"depth_fail\":{},\"depth_pass\":{}}}}}",
        ds, VC::DepthStencilState::DepthTestEnable(ds), VC::DepthStencilState::DepthWriteEnable(ds),
        VC::DepthStencilState::DepthFunc(ds), VC::DepthStencilState::StencilTestEnable(ds),
        VC::DepthStencilState::FrontStencilFunc(ds), VC::DepthStencilState::FrontFailOp(ds),
        VC::DepthStencilState::FrontDepthFailOp(ds), VC::DepthStencilState::FrontDepthPassOp(ds),
        VC::DepthStencilState::BackStencilFunc(ds), VC::DepthStencilState::BackFailOp(ds),
        VC::DepthStencilState::BackDepthFailOp(ds), VC::DepthStencilState::BackDepthPassOp(ds));
    out += ",\"blend\":[";
    bool first_blend = true;
    for (u32 target = 0; target < MAX_COLOR_TARGETS; ++target) {
        const u64 word = state.blend[target];
        const bool enabled = VC::ColorState::BlendEnabled(state.color, static_cast<s32>(target));
        out += fmt::format(
            "{}{{\"target\":{},\"enabled\":{},\"word\":\"{:#x}\",\"src_rgb\":{},\"dst_rgb\":{},"
            "\"src_a\":{},\"dst_a\":{},\"eq_rgb\":{},\"eq_a\":{},\"advanced_mode\":{}}}",
            first_blend ? "" : ",", target, enabled, word, VC::BlendState::SrcRgbFunc(word),
            VC::BlendState::DstRgbFunc(word), VC::BlendState::SrcAlphaFunc(word),
            VC::BlendState::DstAlphaFunc(word), VC::BlendState::RgbEquation(word),
            VC::BlendState::AlphaEquation(word), VC::BlendState::AdvancedMode(word));
        first_blend = false;
    }
    out += "]";
    out += fmt::format(",\"multisample\":\"{}\"", Hex(state.multisample));
    out += fmt::format(
        ",\"primitive_restart\":{},\"restart_index\":{},\"sample_mask\":\"{:#x}\","
        "\"depth_clamp\":{},\"rasterizer_discard\":{},\"depth_mode\":{},\"window_origin\":{},"
        "\"point_size_bits\":\"{:#x}\",\"alpha_ref_bits\":\"{:#x}\",\"provoking_vertex\":\"{}\"",
        state.primitive_restart, state.restart_index, state.sample_mask, state.depth_clamp,
        state.rasterizer_discard, state.depth_mode, state.window_origin, state.point_size_bits,
        state.alpha_ref_bits, state.provoking_first != 0 ? "first" : "last");
    if (mode == VertexInputMode::Static) {
        out += ",\"vertex_input\":{\"mode\":\"static\",\"attributes\":[";
        bool first = true;
        for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
            const u32 word = state.attributes[location];
            if (VC::VertexAttribState::Format(word) == 0) {
                continue;
            }
            out += fmt::format(
                "{}{{\"location\":{},\"word\":\"{:#x}\",\"format\":{},\"offset\":{},\"stream\":{}}}",
                first ? "" : ",", location, word, VC::VertexAttribState::Format(word),
                VC::VertexAttribState::Offset(word), VC::VertexAttribState::StreamIndex(word));
            first = false;
        }
        out += "],\"streams\":[";
        first = true;
        for (u32 stream = 0; stream < MAX_VERTEX_STREAMS; ++stream) {
            if (state.stream_stride[stream] == 0 && state.stream_divisor[stream] == 0) {
                continue;
            }
            out += fmt::format("{}{{\"stream\":{},\"stride\":{},\"divisor\":{}}}", first ? "" : ",",
                               stream, state.stream_stride[stream], state.stream_divisor[stream]);
            first = false;
        }
        out += "]}";
    } else {
        out += ",\"vertex_input\":{\"mode\":\"dynamic\",\"types\":[";
        for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
            out += fmt::format("{}{}", location != 0 ? "," : "",
                               static_cast<u32>(state.vertex_types[location]));
        }
        out += "]}";
    }
    if (pipeline != nullptr) {
        out += fmt::format(
            ",\"viewport_count\":{},\"attributes_read\":\"{:#x}\",\"color_outputs_written\":\"{:#x}\","
            "\"texture_queries\":{}",
            pipeline->viewport_count, pipeline->attributes_read, pipeline->color_outputs_written,
            QueriesJson(*pipeline));
    }
    return out + "}";
}

std::string ComputeStateJson(const ComputeKey& key, const PipelineBase* pipeline, u64 base_key,
                             u32 variant, std::string_view fallback,
                             const VC::NvnControl* control) {
    std::string out = fmt::format("{{\"kind\":\"compute\",\"key_base\":\"{:016x}\",\"variant\":{}",
                                  base_key, variant);
    out += fmt::format(
        ",\"environment_flags\":{},\"local_memory_size\":{},\"shared_memory_size_override\":{}",
        key.environment_flags, key.local_memory_size, key.shared_memory_size);
    if (!fallback.empty()) {
        out += fmt::format(",\"fallback\":{}", ShaderLog::Quote(fallback));
    }
    VC::ComputeDriverLayout layout;
    if (pipeline != nullptr) {
        const PipelineStage& stage = pipeline->stages[STAGE_COMPUTE];
        out += fmt::format(",\"workgroup_size\":[{},{},{}],\"shared_memory_size\":{}",
                           stage.workgroup_size[0], stage.workgroup_size[1],
                           stage.workgroup_size[2], stage.shared_memory_size);
        layout = pipeline->compute_layout;
        out += fmt::format(",\"texture_queries\":{}", QueriesJson(*pipeline));
    } else if (control != nullptr) {
        out += fmt::format(",\"workgroup_size\":[{},{},{}],\"shared_memory_size\":{}",
                           control->thread_group[0], control->thread_group[1],
                           control->thread_group[2], control->shared_memory_bytes);
        layout = control->compute;
    }
    out += fmt::format(
        ",\"driver_bank\":{{\"push_constant_words\":{},\"uniform_count\":{},\"storage_count\":{},"
        "\"texture_count\":{},\"image_count\":{},\"declared_total\":{}}}",
        layout.push_constant_words, layout.uniform_count, layout.storage_count,
        layout.texture_count, layout.image_count, layout.declared_total);
    return out + "}";
}

/// First pipeline key of a texture-type variant: variant 0 is the base key itself (async skips
/// are counted on it), later variants mix in the hash of what the shaders asked about textures.
u64 VariantKey(u64 base, const PipelineBase& pipeline, u32 variant) {
    if (variant == 0) {
        return base;
    }
    const u64 queries = Disk::HashQueries(pipeline.queries);
    return Common::CityHash64WithSeed(reinterpret_cast<const char*>(&queries), sizeof(queries),
                                      base);
}

} // Anonymous namespace

std::string& LastIssueSlot() {
    thread_local std::string slot;
    return slot;
}

u64 LogBaseKey(const ComputeKey& key) {
    const std::array<u64, 3> words{key.hash,
                                   u64{key.environment_flags} | (u64{key.local_memory_size} << 32),
                                   u64{key.shared_memory_size}};
    return Common::CityHash64(reinterpret_cast<const char*>(words.data()), sizeof(words));
}

std::string PipelineCache::Impl::RejectedReason(std::span<const u64> hashes) {
    std::scoped_lock lock{reject_mutex};
    for (const u64 hash : hashes) {
        if (hash == 0) {
            continue;
        }
        if (const auto it = rejected_stages.find(hash); it != rejected_stages.end()) {
            return it->second;
        }
    }
    return {};
}

void PipelineCache::Impl::LogCompileFailed(u64 key_hash, std::array<std::string, STAGE_COUNT>&& ir) {
    CompileFailure failure;
    failure.reason = LastIssueSlot();
    failure.ir = std::move(ir);
    std::scoped_lock lock{failure_mutex};
    failure_logs[key_hash] = std::move(failure);
}

void PipelineCache::Impl::LogCompileDone(PipelineBase& pipeline, TranslateOutput& translated,
                                         double compile_ms, u32 variant) {
    if (!pipeline.log) {
        pipeline.log = std::make_shared<PipelineLogInfo>();
    }
    PipelineLogInfo& log = *pipeline.log;
    log.compile_ms = compile_ms;
    log.variant = variant;
    log.from_disk = false;
    log.ir = std::move(translated.ir);
    log.cbuf_used = translated.cbuf_used;
    log.cbuf_known = translated.cbuf_known;
}

namespace {

/// Submits the program record of `stage` when the hash is new to the session and appends its
/// `<hash>.spv` / `<hash>.ir.txt` to `files`. The bytes are the guest's, untouched.
void LogStageProgram(PipelineCache::Impl& impl, u32 stage, const StageProgram& program,
                     const PipelineLogInfo* log, const std::string& ir,
                     std::vector<ShaderLog::File>& files) {
    if (program.hash == 0 || !ShaderLog::ClaimProgram(program.hash)) {
        return;
    }
    ShaderLog::Program record;
    record.hash = program.hash;
    record.stage = LogStageName(stage);
    if (program.code != nullptr) {
        record.bytes = std::span<const u8>{program.code->data(), program.code->size()};
    }
    if (program.control != nullptr) {
        record.control = std::span<const u8>{program.control->data(), program.control->size()};
    }
    std::string reason = impl.RejectedReason(std::span<const u64>{&program.hash, 1});
    {
        std::scoped_lock lock{impl.lowered_mutex};
        if (const auto it = impl.lowered.find(program.hash); it != impl.lowered.end()) {
            const auto& lowered = it->second.program;
            if (lowered && !lowered->error.empty()) {
                record.failed_instruction = lowered->failed_instruction;
                if (reason.empty()) {
                    reason = lowered->error;
                }
            }
        }
    }
    record.rejected = !reason.empty();
    record.reason = reason;
    ShaderLog::SubmitProgram(record);

    if (!ir.empty()) {
        files.push_back({fmt::format("{:016x}.ir.txt", program.hash), BytesOf(ir)});
    }
    if (log != nullptr && log->records[stage]) {
        files.push_back({fmt::format("{:016x}.spv", program.hash),
                         BytesOf(log->records[stage]->spirv)});
    }
}

void LogInputsFile(const InputsProvider* provider, const PipelineBase* pipeline, u64 key,
                   std::vector<ShaderLog::File>& files) {
    if (provider == nullptr) {
        return;
    }
    std::string text = provider->DescribeInputs(pipeline);
    if (!text.empty()) {
        files.push_back({fmt::format("p{:016x}.inputs.json", key), BytesOf(text)});
    }
}

} // Anonymous namespace

void PipelineCache::Impl::LogGraphicsUse(const GraphicsRequest& request, GraphicsPipeline* result,
                                         u64 key_hash, bool not_ready) {
    if (result != nullptr &&
        (result->logged.load(std::memory_order_relaxed) ||
         result->logged.exchange(true, std::memory_order_relaxed))) {
        return;
    }
    if (result == nullptr && !not_ready) {
        // No pipeline object carries `logged`: a rejected key is remembered here instead.
        std::scoped_lock lock{failure_mutex};
        if (!logged_rejected_graphics.insert(key_hash).second) {
            return;
        }
    }
    ShaderLog::UpdateTitle();
    if (result == nullptr && not_ready) {
        ShaderLog::NoteAsyncSkip(key_hash);
        return;
    }
    GraphicsFixedState state = request.state;
    MakeKeyState(state, vertex_input_mode, must_emulate_scaled_formats);
    std::array<u64, GRAPHICS_STAGE_COUNT> hashes{};
    for (size_t i = 0; i < GRAPHICS_STAGE_COUNT; ++i) {
        hashes[i] = request.stages[i].hash;
    }

    const PipelineLogInfo* log = result != nullptr ? result->log.get() : nullptr;
    const u32 variant = log != nullptr ? log->variant : 0;
    const u64 pipeline_key =
        result != nullptr ? VariantKey(key_hash, *result, variant) : key_hash ^ REJECTED_SALT;
    if (!ShaderLog::ClaimPipeline(pipeline_key)) {
        return;
    }

    ShaderLog::Pipeline pipeline;
    pipeline.path = ShaderLog::Path::Hle;
    pipeline.key = pipeline_key;
    std::array<std::string, STAGE_COUNT> failed_ir;
    if (result != nullptr) {
        pipeline.outcome = log != nullptr && log->from_disk ? ShaderLog::Outcome::DiskCache
                                                            : ShaderLog::Outcome::Translated;
        pipeline.compile_ms = log != nullptr ? log->compile_ms : 0.0;
        if (log == nullptr) {
            pipeline.reason = "pipeline was created before the shader log started";
        }
    } else {
        pipeline.outcome = ShaderLog::Outcome::Rejected;
        {
            std::scoped_lock lock{failure_mutex};
            if (const auto it = failure_logs.find(key_hash); it != failure_logs.end()) {
                pipeline.reason = it->second.reason;
                failed_ir = it->second.ir;
            }
        }
        if (pipeline.reason.empty()) {
            pipeline.reason = RejectedReason(hashes);
        }
        if (pipeline.reason.empty()) {
            pipeline.reason = "pipeline could not be created (see the unsupported rows)";
        }
    }
    for (const u32 stage : PIPELINE_ORDER) {
        const StageProgram& program = request.stages[stage];
        if (program.hash == 0) {
            continue;
        }
        pipeline.stages.push_back({LogStageName(stage), program.hash});
        LogStageProgram(*this, stage, program, log, log != nullptr ? log->ir[stage] : failed_ir[stage],
                        pipeline.files);
    }
    pipeline.state_json = GraphicsStateJson(state, request.targets, vertex_input_mode, result,
                                            key_hash, variant);
    LogInputsFile(request.inputs, result, pipeline_key, pipeline.files);
    ShaderLog::SubmitPipeline(std::move(pipeline));
    if (result != nullptr && result->log) {
        // Logged: the IR text and the SPIR-V records are not needed again.
        result->log->ir = {};
        result->log->records = {};
    }
}

void PipelineCache::Impl::LogComputeUse(const ComputeRequest& request, ComputePipeline* result,
                                        bool not_ready) {
    if (result != nullptr &&
        (result->logged.load(std::memory_order_relaxed) ||
         result->logged.exchange(true, std::memory_order_relaxed))) {
        return;
    }
    ComputeKey key;
    key.hash = request.program.hash;
    if (request.environment.compute_local_memory_size) {
        key.environment_flags |= ENV_LOCAL_MEMORY;
        key.local_memory_size = *request.environment.compute_local_memory_size;
    }
    if (request.environment.compute_shared_memory_size) {
        key.environment_flags |= ENV_SHARED_MEMORY;
        key.shared_memory_size = *request.environment.compute_shared_memory_size;
    }
    if (result == nullptr && !not_ready) {
        // No pipeline object carries `logged`: a rejected key is remembered here instead.
        std::scoped_lock lock{failure_mutex};
        if (!logged_rejected_compute.insert(key).second) {
            return;
        }
    }
    ShaderLog::UpdateTitle();
    const u64 base = LogBaseKey(key);
    if (result == nullptr && not_ready) {
        ShaderLog::NoteAsyncSkip(base);
        return;
    }

    const PipelineLogInfo* log = result != nullptr ? result->log.get() : nullptr;
    const u32 variant = log != nullptr ? log->variant : 0;
    const u64 pipeline_key =
        result != nullptr ? VariantKey(base, *result, variant) : base ^ REJECTED_SALT;
    if (!ShaderLog::ClaimPipeline(pipeline_key)) {
        return;
    }

    ShaderLog::Pipeline pipeline;
    pipeline.path = ShaderLog::Path::Hle;
    pipeline.key = pipeline_key;
    std::array<std::string, STAGE_COUNT> failed_ir;
    if (result != nullptr) {
        pipeline.outcome = log != nullptr && log->from_disk ? ShaderLog::Outcome::DiskCache
                                                            : ShaderLog::Outcome::Translated;
        pipeline.compile_ms = log != nullptr ? log->compile_ms : 0.0;
        if (log == nullptr) {
            pipeline.reason = "pipeline was created before the shader log started";
        }
    } else {
        pipeline.outcome = ShaderLog::Outcome::Rejected;
        {
            std::scoped_lock lock{failure_mutex};
            if (const auto it = failure_logs.find(base); it != failure_logs.end()) {
                pipeline.reason = it->second.reason;
                failed_ir = it->second.ir;
            }
        }
        if (pipeline.reason.empty()) {
            pipeline.reason = RejectedReason(std::span<const u64>{&request.program.hash, 1});
        }
        if (pipeline.reason.empty()) {
            pipeline.reason = "pipeline could not be created (see the unsupported rows)";
        }
    }
    pipeline.stages.push_back({LogStageName(STAGE_COMPUTE), request.program.hash});
    LogStageProgram(*this, STAGE_COMPUTE, request.program, log,
                    log != nullptr ? log->ir[STAGE_COMPUTE] : failed_ir[STAGE_COMPUTE],
                    pipeline.files);
    std::optional<VC::NvnControl> control;
    if (result == nullptr && request.program.control != nullptr) {
        std::string error;
        control = VC::ParseControl(
            std::span<const u8>{request.program.control->data(), request.program.control->size()},
            error);
    }
    pipeline.state_json = ComputeStateJson(key, result, base, variant, {},
                                           control ? &*control : nullptr);
    LogInputsFile(request.inputs, result, pipeline_key, pipeline.files);
    ShaderLog::SubmitPipeline(std::move(pipeline));
    if (result != nullptr && result->log) {
        result->log->ir = {};
        result->log->records = {};
    }
}

void PipelineCache::Impl::LogComputeFallback(const ComputeRequest& request,
                                             std::string_view reason) {
    if (!ShaderLog::Enabled() || request.program.hash == 0) {
        return;
    }
    {
        // The replaced dispatch repeats every frame: once per program and reason per cache.
        std::scoped_lock lock{failure_mutex};
        if (std::ranges::any_of(logged_fallbacks, [&](const auto& entry) {
                return entry.first == request.program.hash && entry.second == reason;
            })) {
            return;
        }
        logged_fallbacks.emplace_back(request.program.hash, std::string{reason});
    }
    ShaderLog::UpdateTitle();
    ComputeKey key;
    key.hash = request.program.hash;
    const u64 reason_hash = Common::CityHash64(reason.data(), reason.size());
    const u64 pipeline_key =
        Common::CityHash64WithSeed(reinterpret_cast<const char*>(&reason_hash), sizeof(reason_hash),
                                   LogBaseKey(key));
    if (!ShaderLog::ClaimPipeline(pipeline_key)) {
        return;
    }
    ShaderLog::Pipeline pipeline;
    pipeline.path = ShaderLog::Path::Hle;
    pipeline.key = pipeline_key;
    pipeline.outcome = ShaderLog::Outcome::Fallback;
    pipeline.reason = std::string{reason};
    pipeline.stages.push_back({LogStageName(STAGE_COMPUTE), request.program.hash});
    LogStageProgram(*this, STAGE_COMPUTE, request.program, nullptr, std::string{}, pipeline.files);
    std::optional<VC::NvnControl> control;
    if (request.program.control != nullptr) {
        std::string error;
        control = VC::ParseControl(
            std::span<const u8>{request.program.control->data(), request.program.control->size()},
            error);
    }
    pipeline.state_json = ComputeStateJson(key, nullptr, LogBaseKey(key), 0, reason,
                                           control ? &*control : nullptr);
    LogInputsFile(request.inputs, nullptr, pipeline_key, pipeline.files);
    ShaderLog::SubmitPipeline(std::move(pipeline));
}

} // namespace Vulkan::Nvn
