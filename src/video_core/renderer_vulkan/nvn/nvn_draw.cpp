// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/diag_request.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "shader_recompiler/stage.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/nvn_draw_probe.h"
#include "video_core/shader_log.h"
#include "video_core/renderer_vulkan/nvn/nvn_barrier.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_plan.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_state.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_viewport.h"

namespace Vulkan::Nvn {
namespace {

constexpr u32 DIRTY_VIEWPORT = 1u << 0;
constexpr u32 DIRTY_SCISSOR = 1u << 1;
constexpr u32 DIRTY_BLEND = 1u << 2;
constexpr u32 DIRTY_STENCIL = 1u << 3;
constexpr u32 DIRTY_BIAS = 1u << 4;
constexpr u32 DIRTY_LINE = 1u << 5;

/// The surface clip (3D SURFACE_CLIP, method 0x0FF4) of a pass without render targets. The SDK's
/// nvnCommandBufferSetRenderTargets (OoT sdk+0x696c60) writes the clip as the smallest width and
/// height of the bound targets, starting from 0xFFFF and capped by a device limit of 32768 (the
/// device field +0x40, filled from the SDK's constant block next to the DeviceInfo items 0x2D,
/// 0x3E and 0x38): with no target bound, draws rasterize over 32768 x 32768 at (0, 0).
constexpr u32 NO_TARGET_SURFACE_CLIP = 32768;

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

template <typename T>
bool ReplaceState(T& current, const T& value, bool elide) {
    if (elide && current == value) {
        return false;
    }
    current = value;
    return true;
}

template <typename Prefix, typename Element>
std::span<const Element> Array(const VC::RecordHeader* header, const Prefix* prefix, u32 count) {
    const u32 available =
        (header->size - static_cast<u32>(sizeof(VC::RecordHeader) + sizeof(Prefix))) /
        static_cast<u32>(sizeof(Element));
    return {VC::TrailingArray<Element>(prefix), (std::min)(count, available)};
}

/// Format and aspect an attachment view is created with.
struct AttachmentFormat {
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageAspectFlags aspect{};
    bool attachable{};
    bool depth_or_stencil{};
};

} // namespace

bool DrawContext::OpKeepsPass(VC::Op op) noexcept {
    using VC::Op;
    switch (op) {
    case Op::End:
    case Op::CallList:
    case Op::FenceSync:
    case Op::SignalEvent:
        return true;             // record nothing on the command buffer
    case Op::SetRenderTargets:   // ends the pass itself, when it binds other targets
    case Op::Barrier:            // ends the pass itself, when the pass wrote through shaders
    case Op::SetRenderCondition: // the recorder ends the pass when a predicate must be computed
        return true;
    case Op::Dispatch:
    case Op::EvaluateDlss: // outside a pass: DLSS and the fallbacks record their own passes
    case Op::ReportValue:
    case Op::ReportCounter:
    case Op::ResetCounter:
        return false;
    default:
        break;
    }
    const u16 value = static_cast<u16>(op);
    return value >= static_cast<u16>(Op::SetRenderTargets) &&
           value <= static_cast<u16>(Op::SetProvokingVertex);
}

DrawContext::DrawContext(Context::Impl& impl_, u32 queue_)
    : impl{impl_}, resources{*impl_.draw_resources}, queue{queue_},
      costsplit{impl_.costsplit.get()} {
    pipeline_memo.resize(1024);
    std::memset(&fixed, 0, sizeof(fixed));
    fixed.blend.fill(VC::BlendState::Defaults());
    fixed.color = VC::ColorState::Defaults();
    fixed.channel_mask = VC::ChannelMaskState::Defaults();
    fixed.polygon = VC::PolygonState::Defaults();
    fixed.depth_stencil = VC::DepthStencilState::Defaults();
    fixed.sample_mask = 0xFFFFFFFFu;
    for (u32 i = 0; i < MAX_VIEWPORTS; ++i) {
        depth_ranges[i] = {0.0f, 1.0f};
    }
}

DrawContext::~DrawContext() {
    impl.uniform_snapshots.Deactivate(usnap);
}

void DrawContext::BeginSubmission(DrawHost& host_, const VC::Submission& submission_) {
    host = &host_;
    submission = &submission_;
    usnap.Reset(submission_.uniform_snapshot.get(), submission_.seq);
    impl.uniform_snapshots.Activate(usnap);
    pass_open = false;
    pipeline_dirty = true;
    vertex_dirty = true;
    dynamic_dirty = ~0u;
    bound_pipeline = VK_NULL_HANDLE;
    vertex_input_set = false;
    inline_updates.clear();
    shadows.clear();
    staged_vertex.clear();
    stage_covered_end = nullptr;
    extra_uses.clear();
    set_pool.reset();
    set_pool_used = 0;
    last_kept = nullptr;
    last_compute_kept = nullptr;
    last_framebuffer.handle = VK_NULL_HANDLE;
    resources.RefreshSamplerSetting();
}

void DrawContext::EndSubmission() {
    host = nullptr;
    submission = nullptr;
    impl.uniform_snapshots.Deactivate(usnap);
    usnap.Reset(nullptr);
    pass_open = false;
    inline_updates.clear();
    shadows.clear();
    staged_vertex.clear();
    stage_covered_end = nullptr;
    extra_uses.clear();
    set_pool.reset();
    if (texture_slots_cleaned_epoch != impl.texture_epoch) {
        for (TextureSlot& slot : texture_slots) {
            if (slot.texture && !slot.texture->live) {
                slot = {};
            }
        }
        texture_slots_cleaned_epoch = impl.texture_epoch;
    }
}

void DrawContext::OnSegmentBegin(bool keep_shadows) {
    pass_open = false;
    pass_shader_writes = false;
    pass_buffer_writes = false;
    pass_reports_ordered = false;
    pipeline_dirty = true;
    vertex_dirty = true;
    dynamic_dirty = ~0u;
    if (!keep_shadows) {
        shadows.clear();
    }
    staged_vertex.clear();
    stage_covered_end = nullptr;
    bound_pipeline = VK_NULL_HANDLE;
    bound_state.Reset();
    vertex_input_set = false;
}

void DrawContext::OnSegmentEnd() {
    EndPass(BreakCause::SegmentEnd);
}

// ---------------------------------------------------------------------------------------------
// Dispatch of draw-path ops
// ---------------------------------------------------------------------------------------------

bool DrawContext::Handle(const VC::RecordHeader* header) {
    using VC::Op;
    const auto op = static_cast<Op>(header->op);
    const size_t payload_size = header->size - sizeof(VC::RecordHeader);
    const auto fits = [&](size_t size) {
        if (payload_size < size) {
            host->HostFailed(fmt::format("op {}", header->op),
                             fmt::format("payload is {} bytes, expected {}", payload_size, size));
            return false;
        }
        return true;
    };
    switch (op) {
    case Op::SetRenderTargets:
        if (fits(sizeof(VC::SetRenderTargetsPayload))) {
            SetRenderTargets(header);
        }
        return true;
    case Op::ClearColor:
        if (fits(sizeof(VC::ClearColorPayload))) {
            ClearColor(*VC::RecordPayload<VC::ClearColorPayload>(header));
        }
        return true;
    case Op::ClearDepthStencil:
        if (fits(sizeof(VC::ClearDepthStencilPayload))) {
            ClearDepthStencil(*VC::RecordPayload<VC::ClearDepthStencilPayload>(header));
        }
        return true;
    case Op::SetViewports:
        if (fits(sizeof(VC::SetViewportsPayload))) {
            const auto* p = VC::RecordPayload<VC::SetViewportsPayload>(header);
            const auto values =
                Array<VC::SetViewportsPayload, VC::ViewportRect>(header, p, p->count);
            bool changed = false;
            for (u32 i = 0;
                 i < values.size() && p->first < MAX_VIEWPORTS && i < MAX_VIEWPORTS - p->first;
                 ++i) {
                const u32 slot = p->first + i;
                const u32 bit = 1u << slot;
                changed |= (viewports_set & bit) == 0;
                changed |=
                    ReplaceDynamicState(viewports[slot], values[i], !Debug::Get().no_state_elide);
                viewports_set |= bit;
            }
            if (changed) {
                dynamic_dirty |= DIRTY_VIEWPORT;
            }
        }
        return true;
    case Op::SetScissors:
        if (fits(sizeof(VC::SetScissorsPayload))) {
            const auto* p = VC::RecordPayload<VC::SetScissorsPayload>(header);
            const auto values = Array<VC::SetScissorsPayload, VC::ScissorRect>(header, p, p->count);
            bool changed = false;
            for (u32 i = 0;
                 i < values.size() && p->first < MAX_VIEWPORTS && i < MAX_VIEWPORTS - p->first;
                 ++i) {
                const u32 slot = p->first + i;
                const u32 bit = 1u << slot;
                changed |= (scissors_set & bit) == 0;
                changed |=
                    ReplaceDynamicState(scissors[slot], values[i], !Debug::Get().no_state_elide);
                scissors_set |= bit;
            }
            if (changed) {
                dynamic_dirty |= DIRTY_SCISSOR;
            }
        }
        return true;
    case Op::SetDepthRanges:
        if (fits(sizeof(VC::SetDepthRangesPayload))) {
            const auto* p = VC::RecordPayload<VC::SetDepthRangesPayload>(header);
            const auto values =
                Array<VC::SetDepthRangesPayload, VC::DepthRange>(header, p, p->count);
            bool changed = false;
            for (u32 i = 0;
                 i < values.size() && p->first < MAX_VIEWPORTS && i < MAX_VIEWPORTS - p->first;
                 ++i) {
                changed |= ReplaceDynamicState(depth_ranges[p->first + i], values[i],
                                               !Debug::Get().no_state_elide);
            }
            if (changed) {
                dynamic_dirty |= DIRTY_VIEWPORT;
            }
        }
        return true;
    case Op::BindProgram:
        if (fits(sizeof(VC::BindProgramPayload))) {
            SetProgram(*VC::RecordPayload<VC::BindProgramPayload>(header));
        }
        return true;
    case Op::BindState:
        if (fits(sizeof(VC::BindStatePayload))) {
            BindState(header);
        }
        return true;
    case Op::BindVertexAttribs:
        if (fits(sizeof(VC::BindVertexAttribsPayload))) {
            BindVertexAttribs(header);
        }
        return true;
    case Op::BindVertexStreams:
        if (fits(sizeof(VC::BindVertexStreamsPayload))) {
            BindVertexStreams(header);
        }
        return true;
    case Op::BindVertexBuffers:
        if (fits(sizeof(VC::BindVertexBuffersPayload))) {
            BindVertexBuffers(header);
        }
        return true;
    case Op::BindUniformBuffers:
    case Op::BindStorageBuffers:
        if (fits(sizeof(VC::BindBuffersPayload))) {
            BindBuffers(header, op == Op::BindUniformBuffers);
        }
        return true;
    case Op::BindTextures:
    case Op::BindSeparateTextures:
    case Op::BindSeparateSamplers:
    case Op::BindImages:
        if (fits(sizeof(VC::BindHandlesPayload))) {
            BindHandles(header, op);
        }
        return true;
    case Op::SetStencil:
        if (fits(sizeof(VC::SetStencilPayload))) {
            SetStencil(*VC::RecordPayload<VC::SetStencilPayload>(header));
        }
        return true;
    case Op::SetPolygonOffset:
        if (fits(sizeof(VC::SetPolygonOffsetPayload))) {
            const auto* p = VC::RecordPayload<VC::SetPolygonOffsetPayload>(header);
            const f32 values[]{p->factor, p->units, p->clamp};
            if (ReplaceDynamicState(polygon_offset, values, !Debug::Get().no_state_elide)) {
                dynamic_dirty |= DIRTY_BIAS;
            }
        }
        return true;
    case Op::Draw:
        if (fits(sizeof(VC::DrawPayload))) {
            Draw(*VC::RecordPayload<VC::DrawPayload>(header));
        }
        return true;
    case Op::DrawIndexed:
        if (fits(sizeof(VC::DrawIndexedPayload))) {
            DrawIndexed(*VC::RecordPayload<VC::DrawIndexedPayload>(header));
        }
        return true;
    case Op::DrawMeshTasks:
        if (fits(sizeof(VC::DrawMeshTasksPayload))) {
            DrawMeshTasks(*VC::RecordPayload<VC::DrawMeshTasksPayload>(header));
        }
        return true;
    case Op::Dispatch:
        if (fits(sizeof(VC::DispatchPayload))) {
            Dispatch(*VC::RecordPayload<VC::DispatchPayload>(header));
        }
        return true;
    case Op::EvaluateDlss:
        if (fits(sizeof(VC::EvaluateDlssPayload))) {
            EvaluateDlssOp(*VC::RecordPayload<VC::EvaluateDlssPayload>(header));
        }
        return true;
    case Op::Barrier:
        // A barrier is the game's texture/fragment ordering point: scratch copies of sampled
        // render targets made before it are not current after it.
        render_target_epoch.OnBarrier();
        // NVN orders later draws after the shader writes (storage buffers, storage images, raw
        // global memory) of earlier ones only when the game asks. Vulkan has no such order inside a
        // render pass (no self-dependency covers vertex-stage reads after fragment writes), so a
        // barrier that orders anything ends a pass that wrote through shaders. A pass without
        // such writes has nothing to order: attachment order is implicit, and texture/shader
        // invalidations are covered by the snapshots and the per-draw contents checks.
        // Counter reports are the same: the console writes a report as a semaphore release behind
        // the earlier draws, which later work does not wait for unless told to (Mesa NVK's
        // nvk_query_pool.c waits for idle after its report writes before anything may depend on
        // them; NVN's ordering point is the barrier). The reports of the pass are written when it
        // ends, so a barrier only marks them as ordered; a later draw that reads their bytes ends
        // the pass then (PrepareOutsidePass).
        if (fits(sizeof(VC::BarrierPayload))) {
            // What each bit waits for on the console: nvn_barrier.h.
            const u32 bits = VC::RecordPayload<VC::BarrierPayload>(header)->bits;
            if (pass_open) {
                if (pass_shader_writes && NvnBarrierOrders(bits)) {
                    EndPass(BreakCause::Barrier);
                    host->HostMarkWritten();
                } else {
                    if (NvnBarrierOrders(bits) && host->HostReportsPending()) {
                        pass_reports_ordered = true;
                    }
                    ++counters.barriers_kept;
                }
            } else if (NvnBarrierOrdersAllWork(bits) || !impl.dispatch_order_by_barriers) {
                host->HostMarkWritten();
            } else {
                // ORDER_FRAGMENTS / ORDER_FRAGMENTS_TILED / invalidations alone outside a pass:
                // the console waits for earlier pixel work only, which the last pass end put
                // behind the pending barrier already. Earlier dispatches stay unordered for later
                // dispatches, as on the console; every other consumer still flushes their writes
                // (HostFlushBarrierBeforeDispatch / FlushPending).
                ++counters.barriers_without_wait;
            }
        }
        return true;
    case Op::SetTexturePool:
    case Op::SetSamplerPool:
        // The submission's snapshots hold the registrations in the global id space; the bind
        // commands carry global ids already. What reads game ids from memory (a handle word of a
        // uniform buffer) needs the base of the pool the game addresses.
        if (fits(sizeof(VC::SetPoolPayload))) {
            const u32 base = VC::RecordPayload<VC::SetPoolPayload>(header)->base;
            const bool texture = op == Op::SetTexturePool;
            (texture ? texture_pool_base : sampler_pool_base) = base;
            std::vector<u32>& seen = pool_bases_logged[texture ? 0 : 1];
            if (std::find(seen.begin(), seen.end(), base) == seen.end() && seen.size() < 64) {
                seen.push_back(base);
                LOG_INFO(Render_Vulkan, "[nvn] {} pool bound at base {} of the global id space",
                         texture ? "texture" : "sampler", base);
            }
        }
        return true;
    case Op::SetBlendColor:
        if (fits(sizeof(VC::SetBlendColorPayload))) {
            if (ReplaceDynamicState(blend_color,
                                    VC::RecordPayload<VC::SetBlendColorPayload>(header)->color,
                                    !Debug::Get().no_state_elide)) {
                dynamic_dirty |= DIRTY_BLEND;
            }
        }
        return true;
    case Op::SetDepthClamp:
        if (fits(sizeof(VC::EnablePayload))) {
            const u32 value = VC::RecordPayload<VC::EnablePayload>(header)->enable ? 1u : 0u;
            pipeline_dirty |= ReplaceState(fixed.depth_clamp, value, impl.mario_kart_world);
        }
        return true;
    case Op::SetAlphaRef:
        if (fits(sizeof(VC::SetAlphaRefPayload))) {
            const u32 value =
                std::bit_cast<u32>(VC::RecordPayload<VC::SetAlphaRefPayload>(header)->reference);
            pipeline_dirty |= ReplaceState(fixed.alpha_ref_bits, value, impl.mario_kart_world);
        }
        return true;
    case Op::PushConstants:
        if (fits(sizeof(VC::PushConstantsPayload))) {
            PushConstantsOp(header);
        }
        return true;
    case Op::UpdateUniformBuffer:
        if (fits(sizeof(VC::UpdateUniformBufferPayload))) {
            UpdateUniformBuffer(header);
        }
        return true;
    case Op::SetPrimitiveRestart:
        if (fits(sizeof(VC::SetPrimitiveRestartPayload))) {
            const auto* p = VC::RecordPayload<VC::SetPrimitiveRestartPayload>(header);
            const u32 enabled = p->enable ? 1u : 0u;
            const u32 index = p->enable ? p->index : 0u;
            pipeline_dirty |= ReplaceState(fixed.primitive_restart, enabled, impl.mario_kart_world);
            pipeline_dirty |= ReplaceState(fixed.restart_index, index, impl.mario_kart_world);
        }
        return true;
    case Op::SetViewportSwizzles:
        host->HostUnsupported("SetViewportSwizzles", "viewport swizzles are not applied");
        return true;
    case Op::BindTransformFeedbackBuffer:
    case Op::TransformFeedbackState:
    case Op::DrawTransformFeedback:
        host->HostUnsupported("TransformFeedback", "transform feedback is not supported");
        return true;
    case Op::SetRasterizerDiscard:
        if (fits(sizeof(VC::EnablePayload))) {
            const u32 value = VC::RecordPayload<VC::EnablePayload>(header)->enable ? 1u : 0u;
            pipeline_dirty |= ReplaceState(fixed.rasterizer_discard, value, impl.mario_kart_world);
        }
        return true;
    case Op::SetSampleMask:
        if (fits(sizeof(VC::U32Payload))) {
            const u32 value = VC::RecordPayload<VC::U32Payload>(header)->value;
            pipeline_dirty |= ReplaceState(fixed.sample_mask, value, impl.mario_kart_world);
        }
        return true;
    case Op::SetProvokingVertex:
        if (fits(sizeof(VC::U32Payload))) {
            const u32 last = VC::RecordPayload<VC::U32Payload>(header)->value & 1;
            const u32 value = last != 0 ? 0u : 1u;
            pipeline_dirty |= ReplaceState(fixed.provoking_first, value, impl.mario_kart_world);
        }
        return true;
    case Op::SetPointSize:
        if (fits(sizeof(VC::F32Payload))) {
            const u32 value =
                std::bit_cast<u32>(VC::RecordPayload<VC::F32Payload>(header)->value);
            pipeline_dirty |= ReplaceState(fixed.point_size_bits, value, impl.mario_kart_world);
        }
        return true;
    case Op::SetLineWidth:
        if (fits(sizeof(VC::F32Payload))) {
            if (ReplaceDynamicState(line_width, VC::RecordPayload<VC::F32Payload>(header)->value,
                                    !Debug::Get().no_state_elide)) {
                dynamic_dirty |= DIRTY_LINE;
            }
        }
        return true;
    case Op::DrawTexture:
        host->HostUnsupported("DrawTexture", "DrawTexture is not implemented");
        return true;
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------------------------
// State ops
// ---------------------------------------------------------------------------------------------

void DrawContext::SetProgram(const VC::BindProgramPayload& p) {
    bool changed = false;
    for (u32 stage = 0; stage < BIND_STAGES; ++stage) {
        if ((p.stages >> stage) & 1) {
            changed |= stages[stage].program_serial != p.program_serial;
            stages[stage].program_serial = p.program_serial;
        }
    }
    pipeline_dirty |= !impl.mario_kart_world || changed;
}

void DrawContext::BindState(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::BindStatePayload>(header);
    const u8* bytes = reinterpret_cast<const u8*>(p + 1);
    const u32 available = header->size - static_cast<u32>(sizeof(VC::RecordHeader) + sizeof(*p));
    if (p->kind >= 6 || p->size != VC::STATE_OBJECT_SIZES[p->kind] || available < p->size) {
        host->HostFailed("BindState",
                         fmt::format("bad state object (kind {}, size {})", p->kind, p->size));
        return;
    }
    const auto assign = [&](auto& destination, const auto& value) {
        return ReplaceState(destination, value, impl.mario_kart_world);
    };
    bool changed = false;
    switch (static_cast<VC::StateKind>(p->kind)) {
    case VC::StateKind::Blend: {
        u64 word;
        std::memcpy(&word, bytes, sizeof(word));
        const u32 target = VC::BlendState::TargetIndex(word);
        changed = assign(fixed.blend[target], word);
        break;
    }
    case VC::StateKind::Color: {
        u32 word;
        std::memcpy(&word, bytes, sizeof(word));
        changed = assign(fixed.color, word);
        break;
    }
    case VC::StateKind::ChannelMask: {
        u32 word;
        std::memcpy(&word, bytes, sizeof(word));
        changed = assign(fixed.channel_mask, word);
        break;
    }
    case VC::StateKind::Multisample: {
        decltype(fixed.multisample) words{};
        std::memcpy(words.data(), bytes, VC::MultisampleState::SIZE);
        changed = assign(fixed.multisample, words);
        break;
    }
    case VC::StateKind::Polygon: {
        u32 word;
        std::memcpy(&word, bytes, sizeof(word));
        changed = assign(fixed.polygon, word);
        break;
    }
    case VC::StateKind::DepthStencil: {
        u64 word;
        std::memcpy(&word, bytes, sizeof(word));
        changed = assign(fixed.depth_stencil, word);
        break;
    }
    }
    pipeline_dirty |= changed;
}

void DrawContext::BindVertexAttribs(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::BindVertexAttribsPayload>(header);
    const auto words = Array<VC::BindVertexAttribsPayload, u32>(header, p, p->count);
    decltype(fixed.attributes) attributes{};
    for (u32 i = 0; i < words.size() && i < MAX_VERTEX_ATTRIBUTES; ++i) {
        attributes[i] = words[i];
    }
    if (impl.mario_kart_world && fixed.attributes == attributes) {
        return;
    }
    fixed.attributes = attributes;
    vertex_layout_dirty = true;
    pipeline_dirty = true;
}

void DrawContext::BindVertexStreams(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::BindVertexStreamsPayload>(header);
    const auto values =
        Array<VC::BindVertexStreamsPayload, VC::VertexStreamState>(header, p, p->count);
    decltype(fixed.stream_stride) strides{};
    decltype(fixed.stream_divisor) divisors{};
    for (u32 i = 0; i < values.size() && i < MAX_VERTEX_STREAMS; ++i) {
        strides[i] = values[i].stride;
        divisors[i] = values[i].divisor;
    }
    if (impl.mario_kart_world && fixed.stream_stride == strides &&
        fixed.stream_divisor == divisors) {
        return;
    }
    fixed.stream_stride = strides;
    fixed.stream_divisor = divisors;
    vertex_layout_dirty = true;
    pipeline_dirty = true;
}

void DrawContext::BindVertexBuffers(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::BindVertexBuffersPayload>(header);
    const auto ranges = Array<VC::BindVertexBuffersPayload, VC::BufferRange>(header, p, p->count);
    for (u32 i = 0; i < ranges.size() && p->first + i < MAX_VERTEX_STREAMS; ++i) {
        vertex_buffers[p->first + i] = ranges[i];
    }
    vertex_dirty = true;
}

void DrawContext::BindBuffers(const VC::RecordHeader* header, bool uniform) {
    const auto* p = VC::RecordPayload<VC::BindBuffersPayload>(header);
    if (p->stage >= BIND_STAGES) {
        host->HostFailed("BindBuffers", fmt::format("stage {}", p->stage));
        return;
    }
    const auto ranges = Array<VC::BindBuffersPayload, VC::BufferRange>(header, p, p->count);
    auto& table = uniform ? stages[p->stage].uniform : stages[p->stage].storage;
    bool changed = false;
    for (u32 i = 0; i < ranges.size() && p->first + i < table.size(); ++i) {
        changed |= table[p->first + i].address != ranges[i].address ||
                   table[p->first + i].size != ranges[i].size;
        table[p->first + i] = {ranges[i].address, ranges[i].size};
    }
    if (!uniform && (!impl.mario_kart_world || changed)) {
        storage_gen[p->stage] = ++bank_clock;
    }
}

void DrawContext::BindHandles(const VC::RecordHeader* header, VC::Op op) {
    const auto* p = VC::RecordPayload<VC::BindHandlesPayload>(header);
    if (p->stage >= BIND_STAGES) {
        host->HostFailed("BindHandles", fmt::format("stage {}", p->stage));
        return;
    }
    const auto handles = Array<VC::BindHandlesPayload, u64>(header, p, p->count);
    StageBindings& stage = stages[p->stage];
    bool changed = false;
    for (u32 i = 0; i < handles.size(); ++i) {
        const u32 slot = p->first + i;
        switch (op) {
        case VC::Op::BindTextures:
            if (slot < stage.combined.size()) {
                changed |= stage.combined[slot] != handles[i];
                stage.combined[slot] = handles[i];
            }
            break;
        case VC::Op::BindSeparateTextures:
            if (slot < stage.separate_texture.size()) {
                changed |= stage.separate_texture[slot] != handles[i];
                stage.separate_texture[slot] = handles[i];
            }
            break;
        case VC::Op::BindSeparateSamplers:
            if (slot < stage.separate_sampler.size()) {
                changed |= stage.separate_sampler[slot] != handles[i];
                stage.separate_sampler[slot] = handles[i];
            }
            break;
        default:
            if (slot < stage.image.size()) {
                changed |= stage.image[slot] != handles[i];
                stage.image[slot] = handles[i];
            }
            break;
        }
    }
    if (!impl.mario_kart_world || changed) {
        handle_gen[p->stage] = ++bank_clock;
    }
}

void DrawContext::SetStencil(const VC::SetStencilPayload& p) {
    bool changed = false;
    for (u32 face = 0; face < 2; ++face) {
        if (((p.face >> face) & 1) == 0) {
            continue;
        }
        StencilFace& state = stencil[face];
        const u32 value = static_cast<u32>(p.value) & 0xFF;
        switch (p.which) {
        case 0:
            changed |= ReplaceDynamicState(state.value_mask, value, !Debug::Get().no_state_elide);
            break;
        case 1:
            changed |= ReplaceDynamicState(state.write_mask, value, !Debug::Get().no_state_elide);
            break;
        default:
            changed |= ReplaceDynamicState(state.reference, value, !Debug::Get().no_state_elide);
            break;
        }
    }
    if (changed) {
        dynamic_dirty |= DIRTY_STENCIL;
    }
}

void DrawContext::PushConstantsOp(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::PushConstantsPayload>(header);
    const u32 available = header->size - static_cast<u32>(sizeof(VC::RecordHeader) + sizeof(*p));
    const u32 size = (std::min)(p->size, available);
    auto& block = push_constants[p->compute ? 1 : 0];
    if (p->offset >= block.size() || size > block.size() - p->offset) {
        host->HostUnsupported("PushConstants",
                              fmt::format("range {:#x}+{:#x} exceeds the {}-byte block", p->offset,
                                          size, PUSH_CONSTANT_BLOCK_SIZE));
        return;
    }
    if (impl.mario_kart_world &&
        std::memcmp(block.data() + p->offset, reinterpret_cast<const u8*>(p + 1), size) == 0) {
        return;
    }
    std::memcpy(block.data() + p->offset, reinterpret_cast<const u8*>(p + 1), size);
    push_gen[p->compute ? 1 : 0] = ++bank_clock;
}

void DrawContext::UpdateUniformBuffer(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::UpdateUniformBufferPayload>(header);
    const u32 available = header->size - static_cast<u32>(sizeof(VC::RecordHeader) + sizeof(*p));
    const u64 size = (std::min<u64>)(p->size, available);
    if (size == 0) {
        return;
    }
    u8* pointer = nullptr;
    const BufferRef slice = AllocArena(size, 16, &pointer);
    if (!slice.buffer) {
        host->HostFailed("UpdateUniformBuffer", "no arena space");
        return;
    }
    std::memcpy(pointer, reinterpret_cast<const u8*>(p + 1), size);
    inline_updates.push_back({p->address, size, pointer, slice.buffer, slice.offset});
    ++counters.inline_updates;
}

// ---------------------------------------------------------------------------------------------
// Render targets and passes
// ---------------------------------------------------------------------------------------------

void DrawContext::SetRenderTargets(const VC::RecordHeader* header) {
    const auto* p = VC::RecordPayload<VC::SetRenderTargetsPayload>(header);
    const u32 total = p->color_count + p->has_depth;
    const auto refs = Array<VC::SetRenderTargetsPayload, VC::RenderTargetRef>(header, p, total);
    if (p->color_count > MAX_COLOR_TARGETS || refs.size() < total) {
        host->HostFailed("SetRenderTargets", fmt::format("{} color targets", p->color_count));
        return;
    }
    ++counters.render_targets_ops;
    {
        // The game re-binds what is bound (it does so around state changes): keep the pass.
        bool same = p->color_count == color_target_count &&
                    (p->has_depth != 0) == (depth_target.ref.texture.serial != 0);
        for (u32 i = 0; same && i < total; ++i) {
            const TargetBinding& bound = i < p->color_count ? color_targets[i] : depth_target;
            same = std::memcmp(&bound.game_ref, &refs[i], sizeof(VC::RenderTargetRef)) == 0;
        }
        if (same) {
            ++counters.render_targets_same;
            return;
        }
    }
    render_target_epoch.OnTargetChange();
    EndPass(BreakCause::SetRenderTargets);
    const auto resolve = [&](const VC::RenderTargetRef& ref) {
        TargetBinding binding;
        binding.ref = ref;
        binding.game_ref = ref;
        if (ref.texture.serial != 0) {
            if (const auto it = impl.textures.find(ref.texture.serial); it != impl.textures.end()) {
                binding.texture = it->second;
                impl.NoteUse(*it->second, host->HostFlight().seq);
                // A block view of a compressed texture renders into its alias (whole blocks).
                if (std::shared_ptr<Texture> alias = impl.TargetAlias(*it->second, binding.ref)) {
                    binding.texture = std::move(alias);
                    binding.alias = true;
                }
            }
        }
        return binding;
    };
    color_target_count = p->color_count;
    for (u32 i = 0; i < MAX_COLOR_TARGETS; ++i) {
        color_targets[i] = i < p->color_count ? resolve(refs[i]) : TargetBinding{};
    }
    depth_target = p->has_depth ? resolve(refs[p->color_count]) : TargetBinding{};
    if (depth_target.texture && NetworkReplacementWanted()) {
        // The scene's depth for a later replacement of the game's upscaling network: the depth of
        // the last pass drawn into each colour texture (the guest's textures, before any twin).
        for (u32 i = 0; i < color_target_count; ++i) {
            if (color_targets[i].texture) {
                impl.pass_depth_of_color[color_targets[i].texture->serial] =
                    depth_target.texture->serial;
            }
        }
    }
    DecideTargetScale();
    pipeline_dirty = true;
    dynamic_dirty |= DIRTY_VIEWPORT | DIRTY_SCISSOR;
}

void DrawContext::DecideTargetScale() {
    targets_scaled = false;
    const Rescale::Factor factor = impl.RescaleFactor();
    if (!factor.Active()) {
        return;
    }
    // The rules (nvn_rescale.h) see the shapes of the bound storages; a pass is scaled only when
    // every attachment is eligible, so a draw never mixes scaled and guest-sized attachments.
    std::array<Rescale::Shape, MAX_COLOR_TARGETS> colors;
    std::array<TargetBinding*, MAX_COLOR_TARGETS> color_bindings{};
    size_t color_count = 0;
    for (u32 i = 0; i < color_target_count; ++i) {
        TargetBinding& target = color_targets[i];
        if (!target.texture) {
            continue;
        }
        if (!target.texture->storage) {
            return; // the draw is dropped anyway (TargetsUsable)
        }
        colors[color_count] = impl.RescaleShapeOf(*target.texture->storage, target.alias);
        color_bindings[color_count++] = &target;
    }
    Rescale::Shape depth;
    const bool has_depth = depth_target.texture != nullptr;
    if (has_depth) {
        if (!depth_target.texture->storage) {
            return;
        }
        depth = impl.RescaleShapeOf(*depth_target.texture->storage, depth_target.alias);
    }
    if (color_count == 0 && !has_depth) {
        return;
    }
    if (!Rescale::PassScaled(factor, std::span<const Rescale::Shape>{colors.data(), color_count},
                             has_depth ? &depth : nullptr, impl.rescale_scene_sizes)) {
        ++impl.stats.rescale_native_passes;
        return;
    }
    // Every attachment gets the texture object of its twin (a Texture over the twin storage); the
    // guest's texture stays in `native`. Nothing is recorded here.
    std::array<std::shared_ptr<Texture>, MAX_COLOR_TARGETS + 1> twins;
    for (size_t i = 0; i < color_count; ++i) {
        twins[i] = impl.ScaledTextureOf(*color_bindings[i]->texture);
        if (!twins[i]) {
            return;
        }
    }
    if (has_depth) {
        twins[MAX_COLOR_TARGETS] = impl.ScaledTextureOf(*depth_target.texture);
        if (!twins[MAX_COLOR_TARGETS]) {
            return;
        }
    }
    for (size_t i = 0; i < color_count; ++i) {
        color_bindings[i]->native = std::move(color_bindings[i]->texture);
        color_bindings[i]->texture = std::move(twins[i]);
        impl.rescale_scene_sizes.Add(colors[i].width, colors[i].height);
    }
    if (has_depth) {
        depth_target.native = std::move(depth_target.texture);
        depth_target.texture = std::move(twins[MAX_COLOR_TARGETS]);
    }
    targets_scaled = true;
    target_scale = factor;
}

namespace {

AttachmentFormat FormatOf(const Device& device, const Texture& texture,
                          const VC::RenderTargetRef& ref) {
    AttachmentFormat result;
    HostFormat host = texture.host;
    if (ref.view.set_format) {
        if (const auto resolved = ResolveHostFormat(device, ref.view.format)) {
            host = *resolved;
        }
    }
    result.format = host.format;
    result.attachable = host.attachable;
    result.depth_or_stencil = host.depth || host.stencil;
    result.aspect =
        (host.depth ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) : 0u) |
        (host.stencil ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_STENCIL_BIT) : 0u);
    if (result.aspect == 0) {
        result.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    return result;
}

} // namespace

bool DrawContext::NamesNoTarget() const {
    for (u32 i = 0; i < color_target_count; ++i) {
        if (color_targets[i].ref.texture.serial != 0) {
            return false;
        }
    }
    return depth_target.ref.texture.serial == 0;
}

bool DrawContext::TargetsUsable() const {
    // The images are created lazily (BeginPass makes them): a texture without a storage is the
    // only thing that cannot be attached. No target at all is a pass without attachments (see
    // NO_TARGET_SURFACE_CLIP); targets that are all unknown textures are not.
    bool any = false;
    for (u32 i = 0; i < color_target_count; ++i) {
        const TargetBinding& target = color_targets[i];
        if (!target.texture) {
            continue;
        }
        if (!target.texture->storage || target.texture->storage->image_failed) {
            return false;
        }
        any = true;
    }
    if (depth_target.texture) {
        if (!depth_target.texture->storage || depth_target.texture->storage->image_failed) {
            return false;
        }
        any = true;
    }
    return any || NamesNoTarget();
}

std::string DrawContext::DescribeUnusableTargets() const {
    // Why TargetsUsable() is false, slot by slot (only on that path: the draw is dropped anyway).
    const auto describe = [&](const TargetBinding& target, std::string_view slot) -> std::string {
        const u64 serial = target.ref.texture.serial;
        if (serial == 0) {
            return {};
        }
        if (!target.texture) {
            return fmt::format("{} texture {:#x} unknown; ", slot, serial);
        }
        const Texture& texture = *target.texture;
        const VC::TextureDesc& d = texture.desc;
        if (texture.storage && !texture.storage->image_failed) {
            return {};
        }
        std::string why;
        if (texture.storage) {
            why = "its image failed";
        } else if (d.view_of != 0) {
            why = fmt::format("view of {} texture {:#x}",
                              impl.textures.contains(d.view_of) ? "storage-less" : "unknown",
                              d.view_of);
        } else if (!ResolveHostFormat(impl.device, d.format)) {
            why = "format has no host counterpart";
        } else if (d.target == static_cast<s32>(TexTarget::Buffer)) {
            why = "texture buffer";
        } else {
            why = "not created";
        }
        return fmt::format("{} texture {:#x} (target {} format {:#x} {}x{}x{} levels {} samples {} "
                           "flags {:#x} view_of {:#x} view format {:#x}/{} layers {}+{}/{}) has no "
                           "storage: {}; ",
                           slot, serial, d.target, d.format, d.width, d.height, d.depth, d.levels,
                           d.samples, d.flags, d.view_of, d.view.format, d.view.set_format,
                           d.view.min_layer, d.view.nlayers, d.view.set_layers, why);
    };
    std::string out;
    for (u32 i = 0; i < color_target_count; ++i) {
        out += describe(color_targets[i], fmt::format("color {}", i));
    }
    out += describe(depth_target, "depth");
    if (out.empty()) {
        out = "no target resolved";
    }
    return out;
}

bool DrawContext::ShapeOf(const TargetBinding& target, bool is_depth, AttachmentShape& out) {
    const Texture& texture = *target.texture;
    const Storage& storage = *texture.storage;
    const AttachmentFormat format = FormatOf(impl.device, texture, target.ref);
    const bool is_3d = storage.type == VK_IMAGE_TYPE_3D;
    if (!format.attachable || is_depth != format.depth_or_stencil ||
        (is_3d && (storage.create_flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) == 0)) {
        host->HostUnsupported(
            "SetRenderTargets",
            fmt::format("texture {:#x} (format {} attachable={} type={} depth_or_stencil={}) "
                        "cannot be {} attachment",
                        texture.serial, static_cast<int>(format.format), format.attachable,
                        static_cast<int>(storage.type), format.depth_or_stencil,
                        is_depth ? "a depth" : "a color"));
        return false;
    }
    {
        // A view whose texels are not the storage's blocks (a block view TargetAlias could not
        // express) would make the GPU write the image at the wrong size: past its memory.
        const s32 view_format = RenderTargetFormat(texture, target.ref);
        const BlockInfo view_block = FormatBlockInfo(view_format);
        const BlockInfo storage_block = FormatBlockInfo(storage.desc.format);
        if (view_block.bytes != 0 && storage_block.bytes != 0 &&
            (view_block.bytes != storage_block.bytes || view_block.width != storage_block.width ||
             view_block.height != storage_block.height)) {
            // The storage's shape and the view's layers say why TargetAlias declined it.
            const TextureLayoutDesc& d = storage.desc;
            host->HostUnsupported(
                "SetRenderTargets",
                fmt::format("texture {:#x} (format {:#x}, {}x{} blocks of {} bytes) level {} "
                            "layer {} through a view of format {:#x} ({}x{} blocks of {} bytes); "
                            "storage target {} {}x{}x{} levels {} flags {:#x} pkg_block_h {:#x} "
                            "samples {}, view layers {}+{} set {}",
                            texture.serial, storage.desc.format, storage_block.width,
                            storage_block.height, storage_block.bytes, target.ref.texture.level,
                            target.ref.texture.layer, view_format, view_block.width,
                            view_block.height, view_block.bytes, d.target, d.width, d.height,
                            d.depth, d.levels, d.flags, d.pkg_block_h, d.samples,
                            target.ref.view.min_layer, target.ref.view.nlayers,
                            target.ref.view.set_layers));
            return false;
        }
    }
    const u32 level = target.ref.texture.level;
    const u32 layer = target.ref.texture.layer;
    // The layers of a 3D image are its depth slices (a 2D / 2D_ARRAY view of them).
    const u32 total = is_3d ? Mip(storage.extent.depth, level) : storage.layers;
    if (level >= storage.levels || layer >= total) {
        host->HostFailed("SetRenderTargets",
                         fmt::format("texture {:#x} level {} layer {} out of range", texture.serial,
                                     level, layer));
        return false;
    }
    u32 count = target.ref.view.set_layers ? target.ref.view.nlayers : total;
    count = std::clamp(count, 1u, total - layer);
    out.format = format.format;
    out.layer_count = count;
    out.extent = {Mip(storage.extent.width, level), Mip(storage.extent.height, level)};
    out.key = ViewKey{
        .format = format.format,
        .type = count == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .aspect = format.aspect,
        .base_level = level,
        .levels = 1,
        .base_layer = layer,
        .layers = count,
        .swizzle = 0,
    };
    return true;
}

void DrawContext::NoteFlipTarget(const std::shared_ptr<Storage>& storage,
                                 VkImageLayout sampled_layout) {
    if (flip_targets.Note(storage, sampled_layout)) {
        ++counters.flip_targets_replaced;
    }
}

void DrawContext::RevertTargetScale() {
    // A twin whose image cannot be created (out of memory): the pass draws at the guest size.
    for (u32 i = 0; i < color_target_count; ++i) {
        if (color_targets[i].native) {
            color_targets[i].texture = std::move(color_targets[i].native);
        }
    }
    if (depth_target.native) {
        depth_target.texture = std::move(depth_target.native);
    }
    targets_scaled = false;
}

bool DrawContext::BeginPass() {
    if (pass_open) {
        return true;
    }
    if (targets_scaled) {
        bool made = true;
        for (u32 i = 0; i < color_target_count; ++i) {
            made = made && (!color_targets[i].native ||
                            impl.EnsureImage(*color_targets[i].texture->storage));
        }
        made = made && (!depth_target.native || impl.EnsureImage(*depth_target.texture->storage));
        if (!made) {
            RevertTargetScale();
            LOG_WARNING(
                Render_Vulkan,
                "[nvn] resolution scale: a scaled render target image could not be created; "
                "this target set draws at the guest size");
        }
    }
    RenderPassKey key{};
    struct Attachment {
        Texture* texture;
        Storage* storage;
        AttachmentShape shape;
        VkImageView view;
    };
    std::array<Attachment, MAX_COLOR_TARGETS + 1> attachments{};
    u32 attachment_count = 0;
    VkExtent2D extent{~0u, ~0u};
    VkExtent2D native_extent{~0u, ~0u}; ///< the same in the game's pixels (differs when scaled)
    u32 layers = ~0u;
    u32 samples = 1;
    const Device& device = impl.device;

    const auto add = [&](TargetBinding& target, bool is_depth, u32 slot) {
        AttachmentShape shape;
        if (!ShapeOf(target, is_depth, shape)) {
            return false;
        }
        Texture& texture = *target.texture;
        Storage& storage = *texture.storage;
        extent.width = (std::min)(extent.width, shape.extent.width);
        extent.height = (std::min)(extent.height, shape.extent.height);
        if (target.native) {
            const Storage& guest = *target.native->storage;
            const u32 level = target.ref.texture.level;
            native_extent.width = (std::min)(native_extent.width, Mip(guest.extent.width, level));
            native_extent.height =
                (std::min)(native_extent.height, Mip(guest.extent.height, level));
        } else {
            native_extent.width = (std::min)(native_extent.width, shape.extent.width);
            native_extent.height = (std::min)(native_extent.height, shape.extent.height);
        }
        layers = (std::min)(layers, shape.layer_count);
        samples = static_cast<u32>(storage.samples);
        (is_depth ? key.depth : key.color[slot]) = static_cast<s32>(shape.format);
        attachments[attachment_count++] = {&texture, &storage, shape, VK_NULL_HANDLE};
        return true;
    };
    for (u32 slot = 0; slot < color_target_count; ++slot) {
        if (color_targets[slot].texture && !add(color_targets[slot], false, slot)) {
            return false;
        }
    }
    if (depth_target.texture && !add(depth_target, true, 0)) {
        return false;
    }
    if (attachment_count == 0) {
        if (!NamesNoTarget()) {
            return false; // the game's targets did not resolve
        }
        // No target: the pass has no attachments and spans NVN's surface clip for that case
        // (the emulated path sizes such a framebuffer by SURFACE_CLIP the same way).
        extent = {(std::min)(NO_TARGET_SURFACE_CLIP, device.GetMaxFramebufferWidth()),
                  (std::min)(NO_TARGET_SURFACE_CLIP, device.GetMaxFramebufferHeight())};
        native_extent = extent;
        layers = 1;
    }
    key.samples = samples;

    // Images, first-use contents and layout transitions happen outside the pass (none is open).
    transition_scratch.clear();
    for (u32 i = 0; i < attachment_count; ++i) {
        Attachment& attachment = attachments[i];
        const bool is_depth = attachment.storage->host.depth || attachment.storage->host.stencil;
        host->HostNoteTouched(attachment.texture->storage);
        if (Storage* const owner = attachment.storage->scale_owner) {
            // Drawing at the scaled size: the twin holds what the guest-sized image holds that it
            // does not (loaded by the pass).
            if (!host->HostPrepareScaledTarget(*owner)) {
                host->HostFailed("SetRenderTargets",
                                 fmt::format("no scaled image for render target {:#x}",
                                             attachment.texture->serial));
                return false;
            }
        } else if (attachment.storage->scaled) {
            // Drawing at the guest size into an image whose twin may be the newer one.
            host->HostMakeNativeCurrent(*attachment.storage);
        }
        if (!host->HostEnsureAttachmentContents(*attachment.storage)) {
            host->HostFailed("SetRenderTargets", fmt::format("no image for render target {:#x}",
                                                             attachment.texture->serial));
            return false;
        }
        transition_scratch.emplace_back(attachment.storage,
                                        is_depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    // Storages that flip between attachment and sampled use go to their sampled layout now, in
    // the same barrier: a later draw of this pass that samples one finds it in place.
    for (size_t i = 0; i < flip_targets.Size();) {
        const std::shared_ptr<Storage> flip = flip_targets[i].target.lock();
        if (!flip) {
            flip_targets.Erase(i);
            continue;
        }
        const size_t index = i++;
        const bool attached =
            flip->current_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
            flip->current_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        if (!flip->image || !attached) {
            continue;
        }
        // Rendered to since it was last sampled: still in use.
        flip_targets.Touch(index);
        bool is_target_of_pass = false;
        for (u32 a = 0; a < attachment_count; ++a) {
            is_target_of_pass |= attachments[a].storage == flip.get();
        }
        if (!is_target_of_pass) {
            host->HostNoteTouched(flip);
            transition_scratch.emplace_back(flip.get(), flip_targets[index].sampled_layout);
            ++counters.flip_transitions;
        }
    }
    // Pool mirrors made for uniform bindings: their stale pages are copied in before the pass (the
    // barrier flush below orders the copies before the draws).
    PrefetchMirrors();
    host->HostTransitionMany(transition_scratch);
    host->HostFlushBarrier();
    // Device-local copies of the vertex ranges this pass reads (nvn_vertex_stage.h), recorded after
    // the barrier that orders the work before the pass.
    StageVertexRanges();

    // The views need the images, which exist now.
    for (u32 i = 0; i < attachment_count; ++i) {
        Attachment& attachment = attachments[i];
        attachment.view = resources.GetView(*attachment.texture, attachment.shape.key);
        if (!attachment.view) {
            host->HostFailed("SetRenderTargets",
                             fmt::format("no image view for {:#x}", attachment.texture->serial));
            return false;
        }
        if (attachment.storage->scale_owner == nullptr) {
            // (A twin has no entry of its own: the game's texture was noted at SetRenderTargets.)
            impl.NoteUse(*attachment.texture, host->HostFlight().seq);
        }
    }

    const VkRenderPass render_pass = resources.Cache().GetRenderPass(key);
    if (!render_pass) {
        host->HostFailed("SetRenderTargets", "render pass creation failed");
        return false;
    }
    std::array<VkImageView, MAX_COLOR_TARGETS + 1> views{};
    for (u32 i = 0; i < attachment_count; ++i) {
        views[i] = attachments[i].view;
    }
    try {
        VkFramebuffer framebuffer = last_framebuffer.handle;
        if (!framebuffer || last_framebuffer.pass != render_pass ||
            last_framebuffer.extent.width != extent.width ||
            last_framebuffer.extent.height != extent.height || last_framebuffer.layers != layers ||
            last_framebuffer.views != views) {
            auto created = std::make_shared<vk::Framebuffer>(device.GetLogical().CreateFramebuffer({
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderPass = render_pass,
                .attachmentCount = attachment_count,
                .pAttachments = views.data(),
                .width = extent.width,
                .height = extent.height,
                .layers = layers,
            }));
            framebuffer = **created;
            host->HostFlight().keepalive.push_back(std::move(created));
            last_framebuffer = {render_pass, framebuffer, extent, layers, views};
        }
        const VkRenderPassBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = {{0, 0}, extent},
            .clearValueCount = 0,
            .pClearValues = nullptr,
        };
        if (impl.gpu_profiler) {
            ProfBeginPass(key, extent, layers, samples);
        }
        host->HostCmd().BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
    } catch (const vk::Exception& exception) {
        host->HostFailed("SetRenderTargets", fmt::format("framebuffer: {}", exception.what()));
        return false;
    }
    pass_open = true;
    ++counters.pass_begins;
    if (VideoCommon::Nvn::TimestampTraceEnabled()) {
        resources.TracePass(extent.width, extent.height);
    }
    framebuffer_extent = native_extent;
    host_extent = extent;
    framebuffer_layers = layers;
    if (targets_scaled) {
        ++impl.stats.rescale_passes;
    }
    if (Debug::Tracing()) {
        DebugOnPassBegin();
    }
    dynamic_dirty |= DIRTY_VIEWPORT | DIRTY_SCISSOR;
    bound_pipeline = VK_NULL_HANDLE;
    bound_state.Reset();
    if (costsplit) {
        CostSplitOnPass();
    }
    return true;
}

void DrawContext::EndPass(BreakCause cause) {
    if (!host) {
        return;
    }
    // Anything but the break of a feedback copy (which copies right away) may be followed by a
    // write to a sampled render target: copies made so far are not current any more.
    render_target_epoch.OnPassEnd(cause == BreakCause::FeedbackCopy);
    // Staged vertex copies serve later passes of the command buffer only across breaks that cannot
    // change pool memory (nvn_vertex_stage.h); ops recorded outside a pass end here too.
    if (!staged_vertex.empty() && !StagedVertexSurvives(cause, pass_open && pass_shader_writes,
                                                        pass_open && host->HostReportsPending(),
                                                        Debug::Get().try_stage_span)) {
        staged_vertex.clear();
        stage_covered_end = nullptr;
    }
    if (pass_open) {
        GpuProf::Flight* const prof = ProfFlight();
        if (prof) {
            ProfCloseDraw(*prof);
        }
        host->HostPassEnding();
        host->HostCmd().EndRenderPass();
        if (prof) {
            ProfAfterEndPass(*prof);
        }
        pass_open = false;
        pass_buffer_writes = false;
        pass_shader_writes = false;
        pass_reports_ordered = false;
        if (Debug::Tracing()) {
            DebugOnPassEnd();
        }
        if (!probe_images.empty()) [[unlikely]] {
            ProbeFlushImages();
        }
        ++counters.pass_breaks;
        ++counters.break_causes[static_cast<size_t>(cause)];
        host->HostMarkWritten();
        bound_pipeline = VK_NULL_HANDLE;
        bound_state.Reset();
        host->HostPassEnded();
    }
    if (!diag_textures.empty()) {
        DiagFlushTextures();
    }
    FlushInlineUpdates();
}

void DrawContext::FlushInlineUpdates() {
    if (inline_updates.empty()) {
        return;
    }
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                            .count = static_cast<u32>(inline_updates.size()),
                            .hash_a = GpuProf::SUB_INLINE}
             : GpuProf::Tag{}};
    // The pass's draws read their own update's arena slice; the guest bytes take the last update
    // of each byte. Vulkan does not order two copies to the same bytes without a barrier (sync
    // validation: WRITE_AFTER_WRITE, OoT updates one 16-byte range several times per pass), so an
    // update that a later one rewrites whole is left out, and one that overlaps a copy recorded
    // since the last barrier goes behind a transfer barrier.
    thread_local std::vector<BufferRef> targets;
    thread_local std::vector<BufferRef> copied;
    targets.assign(inline_updates.size(), BufferRef{});
    for (size_t i = 0; i < inline_updates.size(); ++i) {
        const InlineUpdate& update = inline_updates[i];
        if (!ResolveBuffer(update.address, update.size, true, targets[i])) {
            targets[i].buffer = VK_NULL_HANDLE;
        }
        targets[i].range = update.size;
    }
    const auto superseded_at = [&](size_t i) {
        const BufferRef& target = targets[i];
        return std::any_of(targets.begin() + static_cast<std::ptrdiff_t>(i) + 1, targets.end(),
                           [&](const BufferRef& later) {
                               return later.buffer == target.buffer &&
                                      later.offset <= target.offset &&
                                      target.offset + target.range <= later.offset + later.range;
                           });
    };
    if ((host->HostBarrierMerge() & BARRIER_MERGE_INLINE) != 0) {
        // EDEN_NX2_TRY_BARRIER_MERGE: nothing to copy (every update unresolved or rewritten whole
        // by a later one): no command follows, so neither the barrier in front of the copies nor
        // the mark after them orders anything. Whatever is pending stays pending for the next op.
        u64 resolved = 0;
        bool any_copy = false;
        for (size_t i = 0; i < targets.size(); ++i) {
            if (targets[i].buffer) {
                ++resolved;
                any_copy = any_copy || !superseded_at(i);
            }
        }
        if (!any_copy) {
            counters.inline_updates_superseded += resolved;
            ++counters.inline_flushes_without_copy;
            inline_updates.clear();
            return;
        }
    }
    // Earlier draws may still read the old contents, and the resolves may have written shadows
    // back over these bytes: order the copies after both.
    host->HostMarkWritten();
    host->HostFlushBarrier();
    const auto overlaps = [](const BufferRef& a, const BufferRef& b) {
        return a.buffer == b.buffer && a.offset < b.offset + b.range &&
               b.offset < a.offset + a.range;
    };
    copied.clear();
    for (size_t i = 0; i < inline_updates.size(); ++i) {
        const InlineUpdate& update = inline_updates[i];
        const BufferRef& target = targets[i];
        if (!target.buffer) {
            continue;
        }
        if (superseded_at(i)) {
            ++counters.inline_updates_superseded;
            continue;
        }
        if (std::ranges::any_of(copied, [&](const BufferRef& c) { return overlaps(c, target); })) {
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            };
            host->HostCmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, barrier);
            ++counters.inline_update_barriers;
            copied.clear();
        }
        const VkBufferCopy region{
            .srcOffset = update.buffer_offset,
            .dstOffset = target.offset,
            .size = update.size,
        };
        host->HostCmd().CopyBuffer(update.buffer, target.buffer, region);
        copied.push_back(target);
        if (target.pool) {
            // The uniform snapshots take the update's bytes (this submission's in command order;
            // nvn_uniform_snapshot.h): noted with them before the plain write.
            if (usnap.Active()) {
                ++counters.uniform_snapshot_layers;
            }
            const u64 seq = host->HostFlight().seq;
            impl.NoteFlushedInlinePoolWrite(update.address, update.data, update.size, seq);
            impl.uniform_snapshots.NoteInlineWrite(update.address, update.data, update.size, seq);
            impl.noting_inline_pool_write = true;
            host->HostNotePoolWrite(*target.pool, target.pool_offset, update.size);
            impl.noting_inline_pool_write = false;
            impl.uniform_snapshots.EndInlineWrite();
        }
    }
    inline_updates.clear();
    host->HostMarkWritten();
}

void DrawContext::OnPoolWrite(const Pool& pool, u64 offset, u64 size) {
    if (staged_vertex.empty()) {
        return;
    }
    if (pass_open && pass_shader_writes) {
        // The pass stopped using its copies at its first shader write and drops them at its end;
        // raw global writes name no range at all.
        staged_vertex.clear();
        stage_covered_end = nullptr;
        return;
    }
    const u64 end = offset + size;
    const size_t before = staged_vertex.size();
    std::erase_if(staged_vertex, [&](const auto& entry) {
        const StagedRange& staged = entry.second;
        return staged.pool == &pool && staged.pool_offset < end &&
               offset < staged.pool_offset + staged.size;
    });
    if (staged_vertex.size() != before) {
        stage_covered_end = nullptr; // the window's ranges are staged again by the next pass
    }
}

void DrawContext::StageVertexRanges() {
    const Debug::Config& debug = Debug::Get();
    if (!debug.vertex_stage || pass_shader_writes) {
        return;
    }
    // The scan stops at a render target change too: the ranges of a later pass are staged by that
    // pass (this pass ends there unless the targets stay the same). With try_stage_span it goes on
    // past target changes: the copies survive them (StagedVertexSurvives), so one gather serves the
    // small passes of the window, and the passes inside an already scanned window do not scan
    // again.
    const std::span<const u64> ahead = host->HostRecordsAhead();
    if (debug.try_stage_span && stage_covered_end != nullptr &&
        ahead.data() >= stage_covered_begin && ahead.data() < stage_covered_end) {
        return;
    }
    const IndexStageFilter indices{.enabled = debug.index_stage,
                                   .uint8_indices = impl.device.IsExtIndexTypeUint8Supported()};
    if (debug.try_stage_span) {
        ScanPassVertexRanges(ahead, std::span<const VC::BufferRange>{vertex_buffers},
                             VERTEX_STAGE_LIMITS, indices,
                             SpanWindow{[](VC::Op op) { return OpKeepsPass(op); }}, stage_scan);
        stage_covered_begin = ahead.data();
        stage_covered_end = stage_scan.end;
    } else {
        ScanPassVertexRanges(
            ahead, std::span<const VC::BufferRange>{vertex_buffers}, VERTEX_STAGE_LIMITS, indices,
            [](VC::Op op) { return op != VC::Op::SetRenderTargets && OpKeepsPass(op); },
            stage_scan);
    }
    const bool diag = impl.vtxcov_diag;
    if (diag) {
        VtxCovNoteScan(ahead);
    }
    const auto outcome = [&](const VC::BufferRange& range, VtxCov reason) {
        if (diag) {
            stage_outcomes.insert_or_assign(range.address, StageOutcome{reason, range.size});
        }
    };
    // Few draws do not pay for a copy of their own, but a range a pool mirror serves costs only
    // its stale pages (usually none): the pass is scanned for those whatever its size.
    const bool copy_worth = stage_scan.draws >= VERTEX_STAGE_LIMITS.min_draws;
    if (!copy_worth && !impl.vtx_mirror) {
        if (diag) {
            for (const VC::BufferRange& range : stage_scan.ranges) {
                outcome(range, VtxCov::FewDraws);
            }
        }
        return;
    }
    // EDEN_NX2_TRY_VTXCOV_BOUND raises the bound: the copies are exact at any size.
    const u64 mirror_copy_bound =
        (impl.vtxcov & VTXCOV_BOUND) != 0 ? VTXCOV_COPY_BYTES : VTX_MIRROR_COPY_BYTES;
    stage_sources.clear();
    u64 mirror_copy_bytes = 0;
    u64 reused = 0;
    u64 reused_bytes = 0;
    for (const VC::BufferRange& range : stage_scan.ranges) {
        // Copied for an earlier pass of the command buffer and still current (EndPass,
        // OnPoolWrite): the copy serves this pass too.
        if (const auto it = staged_vertex.find(range.address); it != staged_vertex.end()) {
            if (it->second.size >= range.size) {
                ++reused;
                reused_bytes += range.size;
                outcome(range, it->second.mirror ? VtxCov::Mirror : VtxCov::Staged);
                continue;
            }
            staged_vertex.erase(it);
        }
        Pool* pool = nullptr;
        u64 offset = 0;
        // Only host memory is worth a copy; ranges the draws cannot resolve stay theirs to report.
        if (!impl.ResolveBuffer(range.address, range.size, pool, offset)) {
            continue;
        }
        if (pool->pending.load(std::memory_order_acquire) ||
            pool->backing != Pool::Backing::Imported) {
            outcome(range,
                    pool->backing != Pool::Backing::Imported ? VtxCov::Device : VtxCov::Pending);
            continue;
        }
        // A live storage shadow holds newer bytes than the pool; a draw that reads them ends the
        // pass and copies the shadow back first (ClassifyShadows), so it is not staged here.
        const bool shadowed = std::ranges::any_of(shadows, [&](const Shadow& shadow) {
            return shadow.pool == pool && offset < shadow.end && shadow.begin < offset + range.size;
        });
        if (shadowed) {
            outcome(range, VtxCov::Shadowed);
            continue;
        }
        if (!pool->vram_mirror && impl.vtx_mirror) {
            impl.VramMirror(*pool, true, false); // read-only; null when refused (stays as before)
        }
        if (PoolMirror* const mirror = pool->vram_mirror.get()) {
            // The pool's VRAM mirror serves the range: only its stale pages are copied in, and
            // no more than the bound of them per pass (the later passes take the rest; the draws
            // of this one read the range from guest memory).
            if (impl.vtx_mirror && mirror_copy_bytes >= mirror_copy_bound) {
                impl.stats.vram_vertex_budget_skips.fetch_add(1, std::memory_order_relaxed);
                outcome(range, VtxCov::CopyBound);
                continue;
            }
            mirror_runs.clear();
            impl.MirrorTakeStale(*pool, offset, range.size, mirror_runs);
            for (const auto& [begin, end] : mirror_runs) {
                mirror_copies.push_back({pool, begin, end});
                mirror_copy_bytes += end - begin;
            }
            staged_vertex.insert_or_assign(range.address, StagedRange{.size = range.size,
                                                                      .buffer = *mirror->buffer,
                                                                      .offset = offset,
                                                                      .pool = pool,
                                                                      .pool_offset = offset,
                                                                      .mirror = true});
            impl.stats.vram_bindings.fetch_add(1, std::memory_order_relaxed);
            impl.stats.vram_vertex_ranges.fetch_add(1, std::memory_order_relaxed);
            outcome(range, VtxCov::Mirror);
            continue;
        }
        if (!copy_worth) {
            outcome(range, diag ? impl.ReadOnlyMirrorMiss(*pool) : VtxCov::Other);
            continue;
        }
        outcome(range, VtxCov::Staged);
        stage_sources.push_back({.pool = std::bit_cast<u64>(pool),
                                 .offset = offset,
                                 .size = range.size,
                                 .address = range.address});
    }
    counters.vertex_stage_reused_ranges += reused;
    counters.vertex_stage_reused_bytes += reused_bytes;
    // Copies into pool mirrors, ordered before the pass's vertex and index fetches.
    RecordMirrorCopies(true);
    if (stage_sources.empty()) {
        return;
    }
    // Ranges of a pool closer than a copy region costs share one region (nvn_vertex_stage.h).
    CoalesceVertexStageSources(stage_sources, VERTEX_STAGE_LIMITS, stage_spans);
    stage_copies.clear();
    stage_span_dst.clear();
    u64 bytes = 0;
    for (const VertexStageSpan& span : stage_spans) {
        VkBuffer buffer{};
        u64 staged_offset = 0;
        if (!AllocShadow(span.end - span.begin, span.begin, buffer, staged_offset)) {
            break;
        }
        stage_copies.push_back({std::bit_cast<const Pool*>(span.pool), buffer,
                                VkBufferCopy{.srcOffset = span.begin,
                                             .dstOffset = staged_offset,
                                             .size = span.end - span.begin}});
        stage_span_dst.push_back({buffer, staged_offset});
        bytes += span.end - span.begin;
    }
    if (diag) {
        for (const VertexStageSource& source : stage_sources) {
            if (source.span >= stage_span_dst.size()) {
                outcome({source.address, source.size}, VtxCov::NoDeviceMemory);
            }
        }
    }
    if (stage_copies.empty()) {
        return;
    }
    u64 ranges = 0;
    for (const VertexStageSource& source : stage_sources) {
        if (source.span >= stage_span_dst.size()) {
            continue; // its region found no device memory
        }
        const StagedSpan& span = stage_span_dst[source.span];
        staged_vertex.insert_or_assign(
            source.address,
            StagedRange{.size = source.size,
                        .buffer = span.buffer,
                        .offset = span.offset + (source.offset - stage_spans[source.span].begin),
                        .pool = std::bit_cast<const Pool*>(source.pool),
                        .pool_offset = source.offset});
        ++ranges;
    }
    GpuProf::Flight* const prof = ProfFlight();
    const GpuProf::Scope prof_scope{
        prof, prof ? host->HostCmd() : vk::CommandBuffer{},
        prof ? GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                            .count = static_cast<u32>((std::min<u64>)(bytes, ~u32{0})),
                            .hash_a = GpuProf::SUB_VERTEX}
             : GpuProf::Tag{}};
    vk::CommandBuffer cmd = host->HostCmd();
    // The regions are in pool order and their chunks fill in that order: one dispatch (or
    // transfer) per run of regions with the same source and destination. Shader loads keep the
    // copy fast while other processes share the GPU (nvn_vertex_stage_gpu.h); a source window the
    // shader cannot bind falls back to vkCmdCopyBuffer.
    VertexStageGpu* const gpu = impl.VertexStage();
    stage_runs.clear();
    stage_tasks.clear();
    for (size_t i = 0; i < stage_copies.size();) {
        const StageCopy& first = stage_copies[i];
        size_t end = i;
        u64 high = 0;
        for (; end < stage_copies.size() && stage_copies[end].source == first.source &&
               stage_copies[end].destination == first.destination;
             ++end) {
            const VkBufferCopy& region = stage_copies[end].region;
            high = (std::max)(high, region.srcOffset + region.size);
        }
        StageRun run{.begin = i,
                     .end = end,
                     .shader = false,
                     .window_base = 0,
                     .window_range = 0,
                     .first_task = 0,
                     .task_count = 0};
        if (gpu) {
            const u64 alignment = gpu->SourceAlignment();
            const u64 base = first.region.srcOffset / alignment * alignment;
            const u64 top = (high + 15) / 16 * 16;
            if (top <= first.source->size && top - base <= gpu->MaxSourceRange()) {
                run.shader = true;
                run.window_base = base;
                run.window_range = top - base;
                run.first_task = static_cast<u32>(stage_tasks.size());
                for (size_t k = i; k < end; ++k) {
                    const VkBufferCopy& region = stage_copies[k].region;
                    AppendVertexStageTasks(region.srcOffset - base, region.dstOffset, region.size,
                                           stage_tasks);
                }
                run.task_count = static_cast<u32>(stage_tasks.size()) - run.first_task;
            }
        }
        stage_runs.push_back(run);
        i = end;
    }
    BufferRef table{};
    if (!stage_tasks.empty()) {
        u8* pointer = nullptr;
        table = AllocArena(stage_tasks.size() * sizeof(VertexStageTask), 256, &pointer);
        std::memcpy(pointer, stage_tasks.data(), table.range);
    }
    VkPipelineStageFlags src_stages = 0;
    VkAccessFlags src_access = 0;
    for (const StageRun& run : stage_runs) {
        const StageCopy& first = stage_copies[run.begin];
        if (run.shader) {
            gpu->Copy(cmd, first.source->Handle(), run.window_base, run.window_range,
                      first.destination, table.buffer, table.offset, table.range, run.first_task,
                      run.task_count);
            src_stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            src_access |= VK_ACCESS_SHADER_WRITE_BIT;
            continue;
        }
        stage_regions.clear();
        for (size_t k = run.begin; k < run.end; ++k) {
            stage_regions.push_back(stage_copies[k].region);
        }
        cmd.CopyBuffer(first.source->Handle(), first.destination, stage_regions);
        src_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        src_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
    };
    cmd.PipelineBarrier(src_stages, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, barrier);
    ++counters.vertex_stage_passes;
    counters.vertex_stage_ranges += ranges;
    counters.vertex_stage_regions += stage_copies.size();
    counters.vertex_stage_bytes += bytes;
}

void DrawContext::VtxCovNoteScan(std::span<const u64> ahead) {
    stage_outcomes.clear();
    stage_unscanned = VtxCov::NotScanned;
    if (stage_scan.capped) {
        stage_unscanned = VtxCov::TotalCap;
        return;
    }
    const u64* const end = stage_scan.end;
    if (end == nullptr || end < ahead.data() || end > ahead.data() + ahead.size()) {
        return;
    }
    VC::RecordReader reader{end, static_cast<size_t>(ahead.data() + ahead.size() - end)};
    const VC::RecordHeader* const header = reader.Next();
    if (header == nullptr) {
        stage_unscanned = VtxCov::NotScannedEnd; // the list's End (or the end of its records)
        return;
    }
    switch (VC::RecordOp(header)) {
    case VC::Op::SetRenderTargets:
        stage_unscanned = VtxCov::NotScannedTargets;
        break;
    case VC::Op::CallList:
        stage_unscanned = VtxCov::NotScannedCall;
        break;
    default:
        break;
    }
}

VtxCov DrawContext::VtxCovServeOrMiss(BufferRef& ref, u64 address, u64 size, bool indirect_index,
                                      const DrawPlan& plan, VtxCov found) {
    Pool& pool = *ref.pool;
    // EDEN_NX2_TRY_VTXCOV_INPASS / _SHADERWRITES (DESIGN.md "Vertex and index coverage"): the
    // pool's mirror serves the range where it is current; its pages were copied in before this
    // pass began (copy-ins are recorded outside passes and ordered by the pass begin's barrier)
    // and nothing recorded since wrote them (NotePoolWrite and raw-writing draws invalidate at
    // record time, the page tracker sees the guest CPU). Else its pages join the pool's prefetch
    // set and PrefetchMirrors copies them in at the next pass begin.
    const u32 bind_switch = pass_shader_writes ? VTXCOV_SHADER_WRITES : VTXCOV_INPASS;
    if ((impl.vtxcov & bind_switch) != 0 && size <= VTXCOV_BIND_MAX_BYTES &&
        !pool.pending.load(std::memory_order_acquire)) {
        PoolMirror* const mirror = impl.VramMirror(pool, true, false);
        if (mirror != nullptr) {
            // A read-only mirror does not hold what this draw's raw stores write.
            if (!mirror->storage && plan.raw_global_writes) {
                return VtxCov::RawWrites;
            }
            if (impl.MirrorCurrent(pool, ref.pool_offset, size)) {
                ref.buffer = *mirror->buffer;
                ref.offset = ref.pool_offset;
                impl.stats.vram_bindings.fetch_add(1, std::memory_order_relaxed);
                return VtxCov::MirrorBind;
            }
            impl.MirrorMarkPrefetch(pool, ref.pool_offset, size);
            return VtxCov::MirrorStale;
        }
    }
    if (!impl.vtxcov_diag) {
        return VtxCov::Other;
    }
    if (!Debug::Get().vertex_stage) {
        return VtxCov::StageOff;
    }
    if (pass_shader_writes) {
        return VtxCov::ShaderWrites;
    }
    if (found == VtxCov::SizeShort) {
        return found;
    }
    if (indirect_index) {
        return VtxCov::IndexIndirect;
    }
    if (size > VERTEX_STAGE_LIMITS.max_range_bytes) {
        return VtxCov::TooLarge;
    }
    const auto it = stage_outcomes.find(address);
    if (it == stage_outcomes.end()) {
        return stage_unscanned;
    }
    if (it->second.size < size) {
        return VtxCov::SizeShort;
    }
    // Staged by the scan but not found at the bind: a pool write dropped the copy since.
    return VtxCovServed(it->second.reason) ? VtxCov::Dropped : it->second.reason;
}

VkRect2D DrawContext::ClearRect() const {
    // The game's rectangle in its own pixels, then in the pixels the pass draws at.
    VkRect2D rect{{0, 0}, framebuffer_extent};
    if (scissors_set & 1) {
        const VC::ScissorRect& s = scissors[0];
        const s32 x0 = std::clamp(s.x, 0, static_cast<s32>(framebuffer_extent.width));
        const s32 y0 = std::clamp(s.y, 0, static_cast<s32>(framebuffer_extent.height));
        const s32 x1 = std::clamp(s.x + s.width, x0, static_cast<s32>(framebuffer_extent.width));
        const s32 y1 = std::clamp(s.y + s.height, y0, static_cast<s32>(framebuffer_extent.height));
        rect = {{x0, y0}, {static_cast<u32>(x1 - x0), static_cast<u32>(y1 - y0)}};
    }
    return HostRect(rect);
}

VkRect2D DrawContext::HostRect(VkRect2D guest) const {
    const Rescale::Factor factor = PassFactor();
    if (!factor.Active()) {
        return guest;
    }
    const Rescale::Rect scaled = Rescale::ScaleRect(
        factor, {guest.offset.x, guest.offset.y, guest.extent.width, guest.extent.height});
    const s32 x0 = std::clamp(scaled.x, 0, static_cast<s32>(host_extent.width));
    const s32 y0 = std::clamp(scaled.y, 0, static_cast<s32>(host_extent.height));
    const s32 x1 = std::clamp(scaled.x + static_cast<s32>(scaled.width), x0,
                              static_cast<s32>(host_extent.width));
    const s32 y1 = std::clamp(scaled.y + static_cast<s32>(scaled.height), y0,
                              static_cast<s32>(host_extent.height));
    return {{x0, y0}, {static_cast<u32>(x1 - x0), static_cast<u32>(y1 - y0)}};
}

void DrawContext::ClearColor(const VC::ClearColorPayload& p) {
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    const auto finish = [&] { host->HostEndRenderCondition(); };
    if (p.index < 0 || static_cast<u32>(p.index) >= color_target_count ||
        !color_targets[p.index].texture) {
        host->HostFailed("ClearColor", fmt::format("no render target in slot {}", p.index));
        return finish();
    }
    {
        const bool suspended = host->HostSuspendCondition();
        const bool ready = TargetsUsable() && BeginPass();
        if (suspended) {
            host->HostResumeCondition();
        }
        if (!ready) {
            ++counters.draws_skipped;
            return finish();
        }
    }
    if ((p.mask & 0xF) == 0) {
        return finish();
    }
    // Attachment index inside the (compact) render pass.
    u32 attachment = 0;
    for (s32 i = 0; i < p.index; ++i) {
        attachment += color_targets[i].texture ? 1 : 0;
    }
    const VkRect2D rect = ClearRect();
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return finish();
    }
    if ((p.mask & 0xF) != 0xF) {
        const bool gated = host->HostSuspendCondition();
        if (!MaskedColorClear(p, rect, gated)) {
            host->HostUnsupported("ClearColor", "masked clear failed");
            if (gated) {
                host->HostResumeCondition();
            }
        }
        return finish();
    }
    VkClearAttachment clear{};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.colorAttachment = attachment;
    std::memcpy(clear.clearValue.color.uint32, p.color, sizeof(p.color));
    const VkClearRect clear_rect{
        .rect = rect,
        .baseArrayLayer = 0,
        .layerCount = framebuffer_layers,
    };
    host->HostCmd().ClearAttachments(std::span{&clear, 1}, std::span{&clear_rect, 1});
    finish();
}

void DrawContext::ClearDepthStencil(const VC::ClearDepthStencilPayload& p) {
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    const auto finish = [&] { host->HostEndRenderCondition(); };
    if (!depth_target.texture) {
        host->HostFailed("ClearDepthStencil", "no depth target");
        return finish();
    }
    {
        const bool suspended = host->HostSuspendCondition();
        const bool ready = TargetsUsable() && BeginPass();
        if (suspended) {
            host->HostResumeCondition();
        }
        if (!ready) {
            ++counters.draws_skipped;
            return finish();
        }
    }
    const Storage& storage = *depth_target.texture->storage;
    const u32 stencil_bits = static_cast<u32>(p.stencil_mask) & 0xFF;
    VkImageAspectFlags aspects = 0;
    if (p.depth_mask && storage.host.depth) {
        aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (stencil_bits == 0xFF && storage.host.stencil) {
        aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    const VkRect2D rect = ClearRect();
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return finish();
    }
    if (aspects != 0) {
        VkClearAttachment clear{};
        clear.aspectMask = aspects;
        clear.clearValue.depthStencil = {p.depth, static_cast<u32>(p.stencil) & 0xFF};
        const VkClearRect clear_rect{
            .rect = rect,
            .baseArrayLayer = 0,
            .layerCount = framebuffer_layers,
        };
        host->HostCmd().ClearAttachments(std::span{&clear, 1}, std::span{&clear_rect, 1});
    }
    if (stencil_bits != 0 && stencil_bits != 0xFF && storage.host.stencil) {
        const bool gated = host->HostSuspendCondition();
        if (!MaskedStencilClear(p, rect, gated)) {
            host->HostUnsupported("ClearDepthStencil", "partial stencil mask clear failed");
            if (gated) {
                host->HostResumeCondition();
            }
        }
    }
    finish();
}

// ---------------------------------------------------------------------------------------------
// Pipeline and dynamic state
// ---------------------------------------------------------------------------------------------

namespace {

class DrawOracle final : public TextureOracle, public InputsProvider {
public:
    explicit DrawOracle(const DrawContext& context_) : context{context_} {}
    [[nodiscard]] TextureInfo Query(u32 stage, const VC::NvnHandleRecord& record) const override {
        return context.QueryTexture(stage, record);
    }
    [[nodiscard]] std::string DescribeInputs(const PipelineBase* pipeline) const override {
        return context.DescribeInputs(pipeline, false);
    }
    [[nodiscard]] std::unique_ptr<TextureOracle> Freeze(u32 stage_mask) const override {
        return context.FreezeOracle(stage_mask);
    }

private:
    const DrawContext& context;
};

} // namespace

RenderPassKey DrawContext::MakePassKey() const {
    RenderPassKey key{};
    const Device& device = impl.device;
    for (u32 slot = 0; slot < color_target_count; ++slot) {
        if (color_targets[slot].texture) {
            const auto& target = color_targets[slot];
            key.color[slot] = static_cast<s32>(FormatOf(device, *target.texture, target.ref).format);
            key.samples = static_cast<u32>(color_targets[slot].texture->storage->samples);
        }
    }
    if (depth_target.texture) {
        key.depth = static_cast<s32>(FormatOf(device, *depth_target.texture, depth_target.ref).format);
        key.samples = static_cast<u32>(depth_target.texture->storage->samples);
    }
    return key;
}

std::optional<RenderPassKey> DrawContext::LookaheadPassKey(const VC::RecordHeader* header) const {
    if (header->size < sizeof(*header) + sizeof(VC::SetRenderTargetsPayload)) {
        return std::nullopt;
    }
    const auto* p = VC::RecordPayload<VC::SetRenderTargetsPayload>(header);
    if (p->color_count > MAX_COLOR_TARGETS || p->has_depth > 1) {
        return std::nullopt;
    }
    const u32 total = p->color_count + p->has_depth;
    const auto refs = Array<VC::SetRenderTargetsPayload, VC::RenderTargetRef>(header, p, total);
    if (refs.size() != total) {
        return std::nullopt;
    }
    RenderPassKey key{};
    for (u32 slot = 0; slot < total; ++slot) {
        const auto& ref = refs[slot];
        if (ref.texture.serial == 0) {
            continue;
        }
        const auto it = impl.textures.find(ref.texture.serial);
        if (it == impl.textures.end() || !it->second->storage) {
            return std::nullopt;
        }
        const Texture& texture = *it->second;
        const Storage& storage = *texture.storage;
        // Replay can replace this attachment with an alias, which may not exist yet.
        if (storage.image_failed ||
            IsBlockView(storage.desc.format, RenderTargetFormat(texture, ref))) {
            return std::nullopt;
        }
        const bool depth = slot == p->color_count;
        const auto format = FormatOf(impl.device, texture, ref);
        if (!format.attachable || format.depth_or_stencil != depth) {
            return std::nullopt;
        }
        (depth ? key.depth : key.color[slot]) = static_cast<s32>(format.format);
        key.samples = static_cast<u32>(storage.samples);
    }
    return key;
}

bool DrawContext::QueriesValidFor(const GraphicsPipeline& candidate,
                                  const std::array<u64, BIND_STAGES>& cached_gen,
                                  u64 cached_sub_seq, u64 cached_tex_epoch) const noexcept {
    if (candidate.queries.empty()) {
        return true;
    }
    if (!candidate.queries_handle_bank_only || submission == nullptr ||
        cached_sub_seq != submission->seq || cached_tex_epoch != impl.texture_change_epoch) {
        return false;
    }
    u32 mask = candidate.query_stage_mask;
    while (mask != 0) {
        const u32 stage = static_cast<u32>(std::countr_zero(mask));
        mask &= mask - 1;
        if (cached_gen[stage] != handle_gen[stage]) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<GraphicsPipeline> DrawContext::ResolvePipeline(u32 mode, u32 element_count,
    bool& awaiting_async, const PipelineLookahead* lookahead) {
    const u64 generation = resources.Cache().ShaderGeneration();
    if (!lookahead && shader_generation != generation) {
        pipeline.reset();
        for (PipelineMemo& memo : pipeline_memo) {
            memo.pipeline.reset();
        }
        query_cache = {};
        pipeline_dirty = true;
        bound_pipeline = VK_NULL_HANDLE;
        bound_state.Reset();
        last_kept = nullptr;
        shader_generation = generation;
    }
    const DrawOracle oracle{*this};
    const u64 sub_seq = submission != nullptr ? submission->seq : ~u64{0};
    const u64 tex_epoch = impl.texture_change_epoch;
    const auto queries_ok = [&](const GraphicsPipeline& candidate,
                                std::array<u64, BIND_STAGES>& cached_gen, u64& cached_sub_seq,
                                u64& cached_tex_epoch) {
        if (QueriesValidFor(candidate, cached_gen, cached_sub_seq, cached_tex_epoch)) {
            return true;
        }
        if (!candidate.QueriesMatch(oracle)) {
            return false;
        }
        if (!lookahead) {
            cached_gen = handle_gen;
            cached_sub_seq = sub_seq;
            cached_tex_epoch = tex_epoch;
        }
        return true;
    };
    if (!lookahead && !pipeline_dirty && pipeline && mode == last_mode &&
        queries_ok(*pipeline, query_cache.handle_gen, query_cache.sub_seq,
                   query_cache.tex_epoch)) {
        ++counters.pipeline_state_reuses;
        return pipeline;
    }
    // A mesh-task draw uses the task, mesh and fragment programs bound; a vertex draw the vertex
    // pipeline stages. Whatever else is bound (a vertex program left over from an earlier
    // draw, a mesh program bound next to a vertex one) is not part of the pipeline.
    const bool mesh = mode == MESH_DRAW_MODE;
    constexpr u32 VERTEX_STAGE_MASK = 0x1f; // vertex, fragment, geometry, tess control, tess eval
    constexpr u32 MESH_STAGE_MASK = 1u << STAGE_TASK | 1u << STAGE_MESH | 1u << 1;
    const u32 stage_mask = mesh ? MESH_STAGE_MASK : VERTEX_STAGE_MASK;
    // The state words are set again by nearly every draw: look for an earlier draw with the same
    // programs, state and pass formats before canonicalizing and hashing the key.
    GraphicsFixedState input = fixed;
    input.topology = mesh ? 0 : mode;
    if (mesh) {
        // No vertex input, input assembly or tessellation: the vertex layout, topology and
        // restart state of the last vertex draw must not split mesh pipelines.
        input.patch_vertices = 0;
        input.primitive_restart = 0;
        input.restart_index = 0;
        input.attributes.fill(0);
        input.stream_stride.fill(0);
        input.stream_divisor.fill(0);
        input.vertex_types.fill(0);
    }
    input.depth_mode = Debug::Get().depth_mode >= 0
                           ? static_cast<u32>(Debug::Get().depth_mode)
                           : impl.device_depth_mode.load(std::memory_order_relaxed);
    input.window_origin = impl.device_window_origin.load(std::memory_order_relaxed);
    const RenderPassKey targets = lookahead ? lookahead->targets : MakePassKey();
    std::array<u64, GRAPHICS_STAGE_COUNT> programs;
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        programs[stage] = ((stage_mask >> stage) & 1) != 0 ? stages[stage].program_serial : 0;
    }
    u64 memo_hash = programs[0] ^ (programs[1] * 0x9e3779b97f4a7c15ULL);
    memo_hash ^= programs[2] * 0xbf58476d1ce4e5b9ULL;
    memo_hash ^= programs[STAGE_TASK] * 0x517cc1b727220a95ULL;
    memo_hash ^= programs[7] * 0x94d049bb133111ebULL;
    memo_hash ^= (static_cast<u64>(static_cast<u32>(targets.color[0])) |
                  (static_cast<u64>(static_cast<u32>(targets.color[1])) << 32)) *
                 0xd6e8feb86659fd93ULL;
    memo_hash ^= (static_cast<u64>(static_cast<u32>(targets.depth)) |
                  (static_cast<u64>(targets.samples) << 32)) *
                 0xa0761d6478bd642fULL;
    memo_hash ^= (static_cast<u64>(input.topology) | (static_cast<u64>(input.polygon) << 32)) *
                 0xe7037ed1a0b428dbULL;
    memo_hash ^= (static_cast<u64>(input.color) | (static_cast<u64>(input.depth_stencil) << 32)) *
                 0x8ebc6af09c88c6e3ULL;
    memo_hash *= 0x9e3779b97f4a7c15ULL;
    const size_t memo_slot = static_cast<size_t>(memo_hash >> 54);
    if (!lookahead) {
        PipelineMemo& memo = pipeline_memo[memo_slot];
        if (memo.pipeline && memo.programs == programs && memo.targets == targets &&
            std::memcmp(&memo.state, &input, sizeof(input)) == 0) {
            if (queries_ok(*memo.pipeline, memo.query_handle_gen, memo.query_sub_seq,
                           memo.query_tex_epoch)) {
                pipeline = memo.pipeline;
                last_mode = mode;
                pipeline_dirty = false;
                query_cache = {memo.query_sub_seq, memo.query_tex_epoch, memo.query_handle_gen};
                return pipeline;
            }
        }
    }
    GraphicsRequest request;
    bool memoizable = true;
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        const u64 serial = programs[stage];
        if (serial == 0) {
            continue;
        }
        const auto hash_it = resources.program_hashes.find(serial);
        const auto program_it = impl.programs.find(serial);
        if (hash_it == resources.program_hashes.end() || program_it == impl.programs.end()) {
            memoizable = false;
            continue;
        }
        const u64 hash = hash_it->second[stage];
        if (hash == 0) {
            continue;
        }
        request.stages[stage] = {hash, &program_it->second->control[stage],
                                 &program_it->second->code[stage]};
    }
    if (mesh ? request.stages[STAGE_MESH].hash == 0 : request.stages[0].hash == 0) {
        if (!lookahead) {
            host->HostUnsupported(mesh ? "DrawMeshTasks" : "Draw",
                                  mesh ? "no mesh program bound" : "no vertex program bound");
        }
        return nullptr;
    }
    request.state = input;
    const bool has_geometry = request.stages[2].hash != 0;
    if (mesh || (mode != 0 && !has_geometry)) {
        // Mesh programs write their point size (the mesh output topology is the program's).
        request.state.point_size_bits = 0;
    }
    if (VC::ColorState::AlphaFunc(fixed.color) == VC::STATE_ALPHA_FUNC_ALWAYS) {
        request.state.alpha_ref_bits = 0;
    }
    if (request.stages[1].hash == 0) {
        request.state.alpha_ref_bits = 0;
    }
    request.targets = targets;
    request.oracle = &oracle;
    request.inputs = &oracle;
    if (lookahead) {
        resources.Cache().PrefetchGraphics(request, lookahead->skip_key);
        return nullptr;
    }
    // Only a draw that may be skipped asks without waiting for a compile on a worker.
    const bool may_skip = !impl.mario_kart_world &&
                          MaySkipUnbuiltDraw(StoresGlobalMemory(request),
                                             depth_target.texture != nullptr, element_count);
    std::shared_ptr<GraphicsPipeline> result =
        resources.Cache().GetGraphics(request, may_skip ? &awaiting_async : nullptr,
                                      [this, mode](u64 key) { PrefetchUpcomingGraphics(mode, key); });
    if (!result) {
        return nullptr;
    }
    pipeline = result;
    last_mode = mode;
    pipeline_dirty = false;
    query_cache = {sub_seq, tex_epoch, handle_gen};
    if (memoizable) {
        PipelineMemo& memo = pipeline_memo[memo_slot];
        memo.programs = programs;
        memo.state = input;
        memo.targets = targets;
        memo.pipeline = result;
        memo.query_handle_gen = handle_gen;
        memo.query_sub_seq = sub_seq;
        memo.query_tex_epoch = tex_epoch;
    }
    return result;
}

void DrawContext::PrefetchUpcomingGraphics(u32 current_mode, u64 current_key) {
    auto records = host->HostRecordsAhead();
    if (records.empty() || VideoCommon::ShaderLog::Enabled()) {
        return;
    }

    // Reuse the same state decoding as replay. Only the binding and fixed-state commands below
    // may run in the scan; every other command is a boundary.
    const auto saved_fixed = fixed;
    const auto saved_stages = stages;
    const auto saved_handles = handle_gen;
    const u64 saved_clock = bank_clock;
    const bool saved_pipeline_dirty = pipeline_dirty;
    const bool saved_layout_dirty = vertex_layout_dirty;
    SCOPE_EXIT {
        fixed = saved_fixed;
        stages = saved_stages;
        handle_gen = saved_handles;
        bank_clock = saved_clock;
        pipeline_dirty = saved_pipeline_dirty;
        vertex_layout_dirty = saved_layout_dirty;
    };

    constexpr size_t MAX_COMMANDS = 192;
    constexpr size_t MAX_DRAWS = 8;
    size_t draws = 0;
    u32 previous_mode = current_mode;
    bool changed = false;
    PipelineLookahead lookahead{current_key, MakePassKey()};
    auto following = host->HostFollowingLists();
    using VC::Op;
    for (size_t i = 0; i < MAX_COMMANDS && draws < MAX_DRAWS && !records.empty(); ++i) {
        const auto* header = reinterpret_cast<const VC::RecordHeader*>(records.data());
        const size_t words = header->size / sizeof(u64);
        if (header->size < sizeof(*header) || header->size % sizeof(u64) != 0 ||
            words > records.size()) {
            return;
        }
        records = records.subspan(words);
        const size_t bytes = header->size - sizeof(*header);
        u32 mode{};
        switch (static_cast<Op>(header->op)) {
        case Op::End:
            if (following.empty()) {
                return;
            }
            records = following.front();
            following = following.subspan(1);
            continue;
        case Op::SetRenderTargets: {
            const auto targets = LookaheadPassKey(header);
            if (!targets) {
                return;
            }
            changed |= lookahead.targets != *targets;
            lookahead.targets = *targets;
            continue;
        }
        case Op::BindProgram:
        case Op::BindState:
        case Op::BindVertexAttribs:
        case Op::BindVertexStreams:
        case Op::BindUniformBuffers:
        case Op::BindTextures:
        case Op::BindSeparateTextures:
        case Op::BindSeparateSamplers:
        case Op::BindImages:
        case Op::SetPrimitiveRestart:
        case Op::SetRasterizerDiscard:
        case Op::SetSampleMask:
        case Op::SetProvokingVertex:
        case Op::SetPointSize:
        case Op::SetDepthClamp:
        case Op::SetAlphaRef:
            Handle(header);
            changed = true;
            continue;
        case Op::SetViewports:
        case Op::SetScissors:
        case Op::SetDepthRanges:
        case Op::SetStencil:
        case Op::SetPolygonOffset:
        case Op::SetLineWidth:
        case Op::SetBlendColor:
        case Op::PushConstants:
        case Op::BindVertexBuffers:
        case Op::BindStorageBuffers:
        case Op::ClearColor:
        case Op::ClearDepthStencil:
        case Op::Barrier:
        case Op::SetRenderCondition:
            continue;
        case Op::Draw:
            if (bytes < sizeof(VC::DrawPayload)) {
                return;
            }
            mode = VC::RecordPayload<VC::DrawPayload>(header)->mode;
            break;
        case Op::DrawIndexed:
            if (bytes < sizeof(VC::DrawIndexedPayload)) {
                return;
            }
            mode = VC::RecordPayload<VC::DrawIndexedPayload>(header)->mode;
            break;
        case Op::DrawMeshTasks:
            if (bytes < sizeof(VC::DrawMeshTasksPayload)) {
                return;
            }
            mode = MESH_DRAW_MODE;
            break;
        default:
            return;
        }
        ++draws;
        if ((changed || mode != previous_mode) && mode != 14) {
            bool unused = false;
            (void)ResolvePipeline(mode, 0, unused, &lookahead);
            changed = false;
            previous_mode = mode;
        }
    }
}

void DrawContext::ApplyDynamicState(const GraphicsPipeline& active) {
    vk::CommandBuffer cmd = host->HostCmd();
    const u32 count = active.viewport_count;
    if (count != last_viewport_count) {
        dynamic_dirty |= DIRTY_VIEWPORT | DIRTY_SCISSOR;
        last_viewport_count = count;
    }
    // NVN clip space is OpenGL's (+y up) and the image's first row is its top whatever the window
    // origin, so a viewport is always negative-height in Vulkan's terms (clip +y lands on the
    // rectangle's first row). The origin only decides where the rectangle sits: rows count from the
    // top for UPPER_LEFT and from the bottom for LOWER_LEFT. Proven on DK (UPPER_LEFT): with a
    // positive height the G-buffer is stored upside down and render-to-texture text (the world
    // banner) is sampled flipped. EDEN_NX2_NVN_FLIP_VIEWPORT=1 inverts the rule (experiments).
    const bool lower_left = impl.device_window_origin.load(std::memory_order_relaxed) == 0;
    const float surface_height = static_cast<float>(framebuffer_extent.height);
    const bool invert_y = Debug::Get().flip_viewport;
    if (dynamic_dirty & DIRTY_VIEWPORT) {
        std::array<VkViewport, MAX_VIEWPORTS> vp{};
        for (u32 i = 0; i < count; ++i) {
            const VC::DepthRange& depth = depth_ranges[i];
            ViewportRect rect{0.0f, 0.0f, static_cast<float>(framebuffer_extent.width),
                              surface_height};
            if (viewports_set & (1u << i)) {
                const VC::ViewportRect& r = viewports[i];
                rect = {r.x, r.y, (std::max)(r.width, 1.0f), (std::max)(r.height, 1.0f)};
            }
            const ViewportRect placed =
                ToVulkanViewport(rect, surface_height, lower_left, invert_y);
            vp[i] = {
                .x = placed.x,
                .y = placed.y,
                .width = placed.width,
                .height = placed.height,
                .minDepth = depth.near_plane,
                .maxDepth = depth.far_plane,
            };
            if (const Rescale::Factor factor = PassFactor(); factor.Active()) {
                vp[i] = Rescale::ScaleViewport(factor, vp[i]); // the game's viewport, scaled
            }
        }
        cmd.SetViewport(0, std::span{vp.data(), count});
    }
    if (dynamic_dirty & DIRTY_SCISSOR) {
        std::array<VkRect2D, MAX_VIEWPORTS> rects{};
        for (u32 i = 0; i < count; ++i) {
            if (scissors_set & (1u << i)) {
                const VC::ScissorRect& s = scissors[i];
                const s32 top = ScissorTopRow(
                    s.y, s.height, static_cast<s32>(framebuffer_extent.height), lower_left);
                const s32 x0 = std::clamp(s.x, 0, static_cast<s32>(framebuffer_extent.width));
                const s32 y0 = std::clamp(top, 0, static_cast<s32>(framebuffer_extent.height));
                const s32 x1 = std::clamp(s.x + (std::max)(s.width, 0), x0,
                                          static_cast<s32>(framebuffer_extent.width));
                const s32 y1 = std::clamp(top + (std::max)(s.height, 0), y0,
                                          static_cast<s32>(framebuffer_extent.height));
                rects[i] =
                    HostRect({{x0, y0}, {static_cast<u32>(x1 - x0), static_cast<u32>(y1 - y0)}});
            } else {
                rects[i] = {{0, 0}, host_extent};
            }
        }
        cmd.SetScissor(0, std::span{rects.data(), count});
    }
    if (dynamic_dirty & DIRTY_BLEND) {
        cmd.SetBlendConstants(blend_color);
    }
    if (dynamic_dirty & DIRTY_STENCIL) {
        for (u32 face = 0; face < 2; ++face) {
            const VkStencilFaceFlags flag =
                face == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT;
            cmd.SetStencilCompareMask(flag, stencil[face].value_mask);
            cmd.SetStencilWriteMask(flag, stencil[face].write_mask);
            cmd.SetStencilReference(flag, stencil[face].reference);
        }
    }
    if (dynamic_dirty & DIRTY_BIAS) {
        cmd.SetDepthBias(polygon_offset[1], polygon_offset[2], polygon_offset[0]);
    }
    if (dynamic_dirty & DIRTY_LINE) {
        cmd.SetLineWidth(line_width);
    }
    dynamic_dirty = 0;
}

// ---------------------------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------------------------

BufferRef DrawContext::AllocArena(u64 size, u64 alignment, u8** pointer) {
    InFlight& flight = host->HostFlight();
    alignment = (std::max<u64>)(alignment, 16);
    if (flight.arena.empty() ||
        (flight.arena.back()->used + alignment - 1) / alignment * alignment + size >
            flight.arena.back()->size) {
        flight.arena.push_back(resources.AcquireArena(size + alignment));
    }
    ArenaChunk& chunk = *flight.arena.back();
    chunk.used = (chunk.used + alignment - 1) / alignment * alignment;
    BufferRef ref;
    ref.buffer = *chunk.buffer;
    ref.offset = chunk.used;
    ref.range = size;
    if (pointer) {
        *pointer = chunk.pointer + chunk.used;
    }
    chunk.used += size;
    return ref;
}

bool DrawContext::ResolveBuffer(u64 address, u64 size, bool write, BufferRef& out,
                                ShadowPolicy policy) {
    static_cast<void>(write);
    out = {};
    Pool* pool = nullptr;
    u64 offset = 0;
    if (address == 0 ||
        !host->HostResolvePoolRange(address, (std::max<u64>)(size, 1), pool, offset, "buffer")) {
        return false;
    }
    out.buffer = pool->Handle();
    out.offset = offset;
    out.range = size;
    out.pool = pool;
    out.pool_offset = offset;
    if (policy == ShadowPolicy::Flush && !shadows.empty()) {
        // The caller reads or writes these bytes through host memory right away.
        FlushShadowsOverlapping(*pool, offset, (std::max<u64>)(size, 1));
    }
    return out.buffer != VK_NULL_HANDLE;
}

std::optional<u32> DrawContext::ReadUniformWord(u32 stage, u32 slot, u32 byte_offset) const {
    if (stage >= BIND_STAGES || slot >= stages[stage].uniform.size()) {
        return std::nullopt;
    }
    const BufferBinding& binding = stages[stage].uniform[slot];
    if (binding.address == 0 || byte_offset + sizeof(u32) > binding.size) {
        return std::nullopt;
    }
    const u64 address = binding.address + byte_offset;
    if (!inline_updates.empty()) {
        for (auto it = inline_updates.rbegin(); it != inline_updates.rend(); ++it) {
            if (address >= it->address && address + sizeof(u32) <= it->address + it->size) {
                u32 value;
                std::memcpy(&value, it->data + (address - it->address), sizeof(value));
                return value;
            }
        }
    }
    u32 value = 0;
    if (usnap.Active() && ReadSnapshotBytes(address, &value, sizeof(value))) {
        return value;
    }
    bool read_ok = false;
    const CachedBuffer& cached = uniform_cache[stage][slot];
    if (cached.ref.pool != nullptr && cached.address == binding.address &&
        cached.binding_size == binding.size && byte_offset + sizeof(u32) <= cached.ref.range &&
        cached.pool_gen == impl.pool_generation.load(std::memory_order_acquire) &&
        cached.mapping_gen == impl.mkw_mapping_generation) {
        const Pool* const pool = cached.ref.pool;
        if (!pool->pending.load(std::memory_order_acquire) && pool->host != nullptr &&
            cached.ref.pool_offset + byte_offset + sizeof(u32) <= pool->size) {
            std::memcpy(&value, pool->host + cached.ref.pool_offset + byte_offset, sizeof(value));
            read_ok = true;
        }
    }
    if (!read_ok) {
        VC::GuestMemory* memory = impl.guest.load(std::memory_order_acquire);
        if (!memory || !memory->Read(address, &value, sizeof(value))) {
            return std::nullopt;
        }
    }
    if (host != nullptr) {
        impl.ApplyPendingInlineWrites(address, reinterpret_cast<u8*>(&value), sizeof(value),
                                      host->HostFlight().seq);
    }
    return value;
}

// ---------------------------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------------------------

namespace {

u32 IndexSize(u32 type) {
    return type == 0 ? 1 : type == 1 ? 2 : 4;
}

} // namespace

bool DrawContext::PrepareDraw(u32 mode, u32 element_count, const IndexRange* index,
                              PreparedDraw& out, HostIndex& host_index) {
    ++counters.draws;
    CostSplitWork cost_split{costsplit ? CostSplitBeginDraw() : nullptr, false};
    if (!TargetsUsable()) {
        ++counters.draws_skipped;
        host->HostUnsupported("Draw", "no usable render target");
        static Common::BootCounter logged;
        if (logged.Next() < 16) {
            LOG_WARNING(Render_Vulkan, "[nvn] draw without a usable render target: {}",
                        DescribeUnusableTargets());
        }
        return false;
    }
    if (mode == 14) {
        ++counters.draws_skipped;
        host->HostUnsupported("Draw", "tessellation patches are not supported");
        return false;
    }
    // The textures a `textures` diag request notes for this draw are not read back by a pass
    // break inside this call (the draw's own uploads and feedback copies come after the break).
    struct DiagHoldRelease {
        size_t& hold;
        ~DiagHoldRelease() {
            hold = ~size_t{0};
        }
    } diag_hold_release{diag_hold};
    bool awaiting_async = false;
    std::shared_ptr<GraphicsPipeline> active = ResolvePipeline(mode, element_count, awaiting_async);
    if (!active) {
        if (awaiting_async) {
            // The pipeline is compiling on a worker: skip the draw, ask again with the next one.
            ++counters.draws_skipped_async;
        } else {
            ++counters.draws_skipped;
        }
        return false;
    }
    out = {};
    out.pipeline = active.get();
    out.host_topology = mode;
    out.quad_list = mode == 7;
    out.quad_strip = mode == 8;
    out.line_loop = mode == 2;
    if (last_kept != active.get()) {
        host->HostFlight().keepalive.push_back(active);
        last_kept = active.get();
    }

    DrawPlan& plan = draw_plan;
    plan.Clear();
    plan.ranges.insert(plan.ranges.end(), extra_uses.begin(), extra_uses.end());
    extra_uses.clear();
    dbg_notes.clear();
    // A `textures` diag request: the descriptors record stage / slot / sampler, and the textures
    // of this draw are noted (one budget unit per draw).
    diag_bindings = Common::DiagRequest::TextureBudget().Active();
    const bool descriptors_built = BuildDescriptors(*active, plan, true);
    if (diag_bindings) {
        diag_bindings = false;
        if (descriptors_built) {
            DiagNoteTextures();
        }
    }
    if (!descriptors_built) {
        ++counters.draws_skipped;
        return false;
    }
    if (active->descriptors.uses_bindless_table && !BindlessPrepare(plan)) {
        ++counters.draws_skipped;
        host->HostUnsupported("Draw", "the bindless texture table cannot be created");
        return false;
    }
    // Vertex layout of this draw (rebuilt only when the words or the pipeline's attribute reads
    // changed) and the streams whose buffers it binds. The streams are exactly the bindings the
    // pipeline (static) or the dynamic vertex input (below) declares. Task/mesh pipelines have
    // no vertex input.
    if (!active->mesh && (vertex_layout_dirty || vertex_input_read != active->attributes_read)) {
        vertex_issues.clear();
        BuildVertexInput(impl.device, fixed, active->attributes_read, vertex_input, vertex_issues);
        vertex_input_read = active->attributes_read;
        vertex_layout_dirty = false;
        vertex_input_set = false;
        if (active->dynamic_vertex_input) {
            // Static pipelines report these when they are created.
            for (const std::string& issue : vertex_issues) {
                resources.Cache().NoteIssue(issue);
            }
        }
    }
    // Vertex buffers of the streams the attributes read.
    std::array<BufferRef, MAX_VERTEX_STREAMS> vertex{};
    const u32 used_streams = active->mesh ? 0u : vertex_input.used_streams;
    for (u32 s = 0; s < MAX_VERTEX_STREAMS; ++s) {
        if (!((used_streams >> s) & 1)) {
            continue;
        }
        const VC::BufferRange& range = vertex_buffers[s];
        if (!ResolveBuffer(range.address, range.size, false, vertex[s], ShadowPolicy::Defer)) {
            NoteFallback(true, DescReason::VertexUnresolved);
            vertex[s] = {resources.DummyBuffer(), 0, 0, nullptr, 0};
        } else {
            plan.UseRange(vertex[s].pool, vertex[s].pool_offset, range.size, false);
        }
        if (cs_mode != CostSplit::Mode::Count) [[unlikely]] {
            CostSplitNote(CostSplit::Data::Vertex, vertex[s], range.size);
            CostSplitToRing(CostSplit::Data::Vertex, vertex[s], range.address, range.size);
        }
    }
    BufferRef index_ref;
    if (index) {
        if (!ResolveBuffer(index->address, index->bytes, false, index_ref, ShadowPolicy::Defer)) {
            ++counters.draws_skipped;
            host->HostUnsupported(
                "Draw", fmt::format("index buffer {:#x} is not in a pool", index->address));
            return false;
        }
        if (!index->uploaded || index_ref.pool->backing == Pool::Backing::Imported) {
            plan.UseRange(index_ref.pool, index_ref.pool_offset, index->bytes, false);
        }
        if (cs_mode != CostSplit::Mode::Count) [[unlikely]] {
            CostSplitNote(CostSplit::Data::Index, index_ref, index->bytes);
            CostSplitToRing(CostSplit::Data::Index, index_ref, index->address, index->bytes);
        }
    }

    {
        // A conditional-rendering region cannot straddle a render pass begin or end.
        const bool suspended = host->HostSuspendCondition();
        const bool ready = PrepareOutsidePass(plan) && BeginPass();
        if (suspended) {
            host->HostResumeCondition();
        }
        if (!ready) {
            ++counters.draws_skipped;
            return false;
        }
    }
    host->HostBeforeDraw();
    // Streams (and the index buffer) of imported pools whose range the pass staged read the
    // device-local copy. Once a draw of the pass wrote through shaders, the pool may hold newer
    // bytes than the copy. EDEN_NX2_NVN_VTXCOV_DIAG counts every bind by where it is read from;
    // EDEN_NX2_TRY_VTXCOV_* serve more of them from pool mirrors (VtxCovServeOrMiss).
    const bool use_staged = !staged_vertex.empty() && !pass_shader_writes;
    const bool vtxcov = impl.vtxcov_diag || impl.vtxcov != 0;
    if (use_staged || vtxcov) {
        // True when the staged copy serves the range; `found` says why not otherwise.
        const auto take_staged = [&](BufferRef& ref, u64 address, u64 size, VtxCov& found) {
            found = VtxCov::Other;
            if (!use_staged) {
                return false;
            }
            const auto it = staged_vertex.find(address);
            if (it == staged_vertex.end()) {
                return false;
            }
            if (it->second.size < size) {
                found = VtxCov::SizeShort;
                return false;
            }
            ref.buffer = it->second.buffer;
            ref.offset = it->second.offset;
            found = it->second.mirror ? VtxCov::Mirror : VtxCov::Staged;
            return true;
        };
        for (u32 s = 0; s < MAX_VERTEX_STREAMS; ++s) {
            if (!((used_streams >> s) & 1) || vertex[s].pool == nullptr) {
                continue;
            }
            const u64 size = vertex_buffers[s].size;
            if (vertex[s].pool->backing != Pool::Backing::Imported) {
                if (impl.vtxcov_diag) {
                    impl.NoteVtxCov(false, VtxCov::Device, size);
                }
                continue;
            }
            VtxCov reason;
            if (take_staged(vertex[s], vertex_buffers[s].address, size, reason)) {
                ++counters.vertex_stage_binds;
            } else if (vtxcov) {
                reason = VtxCovServeOrMiss(vertex[s], vertex_buffers[s].address, size, false, plan,
                                           reason);
            }
            if (impl.vtxcov_diag) {
                impl.NoteVtxCov(false, reason, size);
            }
        }
        if (index && index_ref.pool != nullptr) {
            if (index_ref.pool->backing != Pool::Backing::Imported) {
                if (impl.vtxcov_diag) {
                    impl.NoteVtxCov(true, VtxCov::Device, index->bytes);
                }
            } else {
                VtxCov reason;
                if (take_staged(index_ref, index->address, index->bytes, reason)) {
                    ++counters.index_stage_binds;
                } else if (vtxcov) {
                    reason = VtxCovServeOrMiss(index_ref, index->address, index->bytes,
                                               element_count == 0, plan, reason);
                }
                if (impl.vtxcov_diag) {
                    impl.NoteVtxCov(true, reason, index->bytes);
                }
            }
        }
    }
    if (VideoCommon::NvnDrawProbe::Active()) [[unlikely]] {
        if (ProbeDraw(*active, mode, element_count, index, vertex, used_streams, active->mesh)) {
            ++counters.draws_skipped;
            return false;
        }
    }
    if (impl.gpu_profiler) {
        ProfOnDraw();
    }
    if (cs_mode == CostSplit::Mode::NoBind) [[unlikely]] {
        cs_draw_index = cs_pass_draws++;
    }
    vk::CommandBuffer cmd = host->HostCmd();
    // A GPL pipeline binds its whole compile once the background has published it.
    const VkPipeline handle = active->Handle();
    if (bound_pipeline != handle) {
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, handle);
        if (bound_pipeline == VK_NULL_HANDLE) {
            // A masked clear or another untracked graphics pipeline may have replaced dynamic
            // values with static state. Restore all of them before this draw.
            dynamic_dirty = ~0u;
        }
        if (bound_pipeline == VK_NULL_HANDLE || active->mesh) {
            // A pipeline bound by something else (the clear pipelines), a fresh pass/command
            // buffer, or a task/mesh pipeline (it has no vertex input state, so binding it may
            // leave the dynamic one undefined): the dynamic vertex input state is unknown.
            vertex_input_set = false;
        }
        if (bound_pipeline == VK_NULL_HANDLE || !active->dynamic_fixed) {
            // Unknown predecessor, or static cull/depth/stencil state replaced the dynamic one.
            fixed_state_set = false;
        }
        bound_pipeline = handle;
        vertex_dirty = true;
        bound_state.OnPipelineBind(*active->layout, active->descriptors.UsesPushConstants());
    }
    if (active->dynamic_fixed && (!fixed_state_set || fixed_state != active->fixed_values)) {
        const DynamicFixedState& v = active->fixed_values;
        cmd.SetCullModeEXT(v.cull_mode);
        cmd.SetFrontFaceEXT(v.front_face);
        cmd.SetDepthBiasEnableEXT(v.depth_bias_enable);
        cmd.SetDepthTestEnableEXT(v.depth_test_enable);
        cmd.SetDepthWriteEnableEXT(v.depth_write_enable);
        cmd.SetDepthCompareOpEXT(v.depth_compare_op);
        cmd.SetStencilTestEnableEXT(v.stencil_test_enable);
        cmd.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, v.front.fail, v.front.pass,
                            v.front.depth_fail, v.front.compare);
        cmd.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, v.back.fail, v.back.pass, v.back.depth_fail,
                            v.back.compare);
        fixed_state = v;
        fixed_state_set = true;
    }
    if (active->dynamic_vertex_input && !vertex_input_set) {
        // Strides and divisors are in the binding descriptions (no vkCmdBindVertexBuffers2).
        cmd.SetVertexInputEXT(
            vk::Span<VkVertexInputBindingDescription2EXT>{vertex_input.bindings.data(),
                                                          vertex_input.binding_count},
            vk::Span<VkVertexInputAttributeDescription2EXT>{vertex_input.attributes.data(),
                                                            vertex_input.attribute_count});
        vertex_input_set = true;
    }
    ApplyDynamicState(*active);
    if (cs_mode == CostSplit::Mode::NoRaster) [[unlikely]] {
        // Empty scissors: the vertex work and setup run, no fragment does.
        std::array<VkRect2D, MAX_VIEWPORTS> empty{};
        cmd.SetScissor(0, std::span{empty.data(), (std::max)(active->viewport_count, 1u)});
        dynamic_dirty |= DIRTY_SCISSOR;
    }
    if (!PushPlan(*active, plan, VK_PIPELINE_BIND_POINT_GRAPHICS)) {
        ++counters.draws_skipped;
        return false;
    }
    // Vertex buffers: contiguous runs of used streams the pass does not have bound already.
    const bool elide = !Debug::Get().no_state_elide;
    u32 stale_streams = 0;
    for (u32 s = 0; s < MAX_VERTEX_STREAMS; ++s) {
        if (!((used_streams >> s) & 1)) {
            continue;
        }
        if (cs_mode == CostSplit::Mode::NoBind && cs_draw_index > 0 &&
            ((cs_bound_streams >> s) & 1) != 0) [[unlikely]] {
            // The pass's first class draw bound this stream: kept.
        } else if (elide && bound_state.TakeStream(s, vertex[s].buffer, vertex[s].offset)) {
            ++counters.elided_vertex_binds;
        } else {
            stale_streams |= 1u << s;
        }
    }
    for (u32 s = 0; s < MAX_VERTEX_STREAMS;) {
        if (!((stale_streams >> s) & 1)) {
            ++s;
            continue;
        }
        u32 end = s;
        std::array<VkBuffer, MAX_VERTEX_STREAMS> buffers;
        std::array<VkDeviceSize, MAX_VERTEX_STREAMS> offsets;
        while (end < MAX_VERTEX_STREAMS && ((stale_streams >> end) & 1)) {
            buffers[end - s] = vertex[end].buffer;
            offsets[end - s] = vertex[end].offset;
            ++end;
        }
        cmd.BindVertexBuffers(s, end - s, buffers.data(), offsets.data());
        s = end;
    }
    vertex_dirty = false;
    if (cs_mode == CostSplit::Mode::NoBind) [[unlikely]] {
        cs_bound_streams |= stale_streams;
    }
    if (index) {
        host_index.buffer = index_ref.buffer;
        host_index.offset = index_ref.offset;
        host_index.type = index->type == 0   ? VK_INDEX_TYPE_UINT8_EXT
                          : index->type == 1 ? VK_INDEX_TYPE_UINT16
                                             : VK_INDEX_TYPE_UINT32;
        if (cs_mode == CostSplit::Mode::NoBind && cs_draw_index > 0 && cs_index_bound)
            [[unlikely]] {
            // The pass's first class draw bound an index buffer: kept.
        } else if (elide && bound_state.TakeIndexBuffer(host_index.buffer, host_index.offset,
                                                        host_index.type)) {
            ++counters.elided_index_binds;
        } else {
            cmd.BindIndexBuffer(host_index.buffer, host_index.offset, host_index.type);
        }
        if (cs_mode == CostSplit::Mode::NoBind) [[unlikely]] {
            cs_index_bound = true;
        }
    }
    if (Debug::Tracing()) {
        if (stages[2].program_serial != 0 && stages[2].storage[0].address != 0) {
            watch_hint = stages[2].storage[0].address;
        }
        DebugOnDraw(mode);
        if (!DebugDrawAllowed()) {
            return false;
        }
    }
    cost_split.recorded = true;
    return true;
}

bool DrawContext::GenerateIndices(const PreparedDraw& prepared, const IndexRange* source, u32 first,
                                  u32 count, s32 base_vertex, HostIndex& out, u32& out_count) {
    // Source vertex numbers: either first..first+count-1 or the game's indices.
    const u8* indices = nullptr;
    u32 index_type = 2;
    if (source) {
        indices = impl.GuestPointer(source->address, source->bytes);
        index_type = source->type;
        if (!indices) {
            host->HostUnsupported("Draw", "index conversion needs CPU-readable index memory");
            return false;
        }
    }
    const auto fetch = [&](u32 i) -> u32 {
        if (!indices) {
            return first + i;
        }
        switch (index_type) {
        case 0:
            return indices[i];
        case 1: {
            u16 value;
            std::memcpy(&value, indices + i * 2, 2);
            return value;
        }
        default: {
            u32 value;
            std::memcpy(&value, indices + i * 4, 4);
            return value;
        }
        }
    };
    static_cast<void>(base_vertex);
    u32 total = 0;
    if (prepared.quad_list) {
        total = count / 4 * 6;
    } else if (prepared.quad_strip) {
        total = count >= 4 ? (count / 2 - 1) * 6 : 0;
    } else if (prepared.line_loop) {
        total = count + 1;
    } else if (source) {
        total = count; // u8 widening
    }
    if (total == 0) {
        out_count = 0;
        return true;
    }
    u8* pointer = nullptr;
    const BufferRef ref = AllocArena(static_cast<u64>(total) * 4, 4, &pointer);
    if (!ref.buffer) {
        host->HostFailed("Draw", "no arena space for generated indices");
        return false;
    }
    u32* out_indices = reinterpret_cast<u32*>(pointer);
    u32 n = 0;
    if (prepared.quad_list) {
        for (u32 q = 0; q + 3 < count; q += 4) {
            const u32 a = fetch(q), b = fetch(q + 1), c = fetch(q + 2), d = fetch(q + 3);
            out_indices[n++] = a;
            out_indices[n++] = b;
            out_indices[n++] = c;
            out_indices[n++] = a;
            out_indices[n++] = c;
            out_indices[n++] = d;
        }
    } else if (prepared.quad_strip) {
        for (u32 q = 0; q + 3 < count; q += 2) {
            const u32 a = fetch(q), b = fetch(q + 1), c = fetch(q + 2), d = fetch(q + 3);
            out_indices[n++] = a;
            out_indices[n++] = b;
            out_indices[n++] = d;
            out_indices[n++] = a;
            out_indices[n++] = d;
            out_indices[n++] = c;
        }
    } else if (prepared.line_loop) {
        for (u32 i = 0; i < count; ++i) {
            out_indices[n++] = fetch(i);
        }
        out_indices[n++] = fetch(0);
    } else {
        // u8 widening keeps the restart marker the UINT8_EXT path would honour.
        const bool restart_u8 = indices && index_type == 0 && fixed.primitive_restart != 0;
        for (u32 i = 0; i < count; ++i) {
            const u32 value = fetch(i);
            out_indices[n++] = restart_u8 && value == 0xFFu ? 0xFFFFFFFFu : value;
        }
    }
    out.buffer = ref.buffer;
    out.offset = ref.offset;
    out.type = VK_INDEX_TYPE_UINT32;
    out_count = n;
    host->HostCmd().BindIndexBuffer(out.buffer, out.offset, out.type);
    bound_state.ForgetIndexBuffer();
    return true;
}

void DrawContext::Draw(const VC::DrawPayload& p) {
    if (p.indirect != 0 || p.count_address != 0) {
        // Indirect draws are the indirect family's (NvnIndirect) through PrepareDraw.
        return DrawIndirect(p);
    }
    if (p.count == 0) {
        return;
    }
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    PreparedDraw prepared;
    HostIndex host_index;
    dbg_call = {p.count, p.instance_count != 0 ? p.instance_count : 1, p.first, false};
    if (impl.gpu_profiler) {
        prof_units = u64{p.count} * (p.instance_count != 0 ? p.instance_count : 1);
    }
    if (PrepareDraw(p.mode, p.count, nullptr, prepared, host_index)) {
        const u32 instances = CostSplitInstances(p.instance_count != 0 ? p.instance_count : 1);
        vk::CommandBuffer cmd = host->HostCmd();
        if (prepared.quad_list || prepared.quad_strip || prepared.line_loop) {
            u32 count = 0;
            if (GenerateIndices(prepared, nullptr, static_cast<u32>(p.first), p.count, 0,
                                host_index, count) &&
                count != 0) {
                cmd.DrawIndexed(CostSplitCount(count), instances, 0, 0, p.base_instance);
            }
        } else {
            cmd.Draw(CostSplitCount(p.count), instances, static_cast<u32>(p.first),
                     p.base_instance);
        }
    }
    host->HostEndRenderCondition();
}

void DrawContext::DrawIndexed(const VC::DrawIndexedPayload& p) {
    if (p.indirect != 0 || p.count_address != 0) {
        return DrawIndexedIndirect(p);
    }
    if (p.count == 0) {
        return;
    }
    if (!host->HostBeginRenderCondition()) {
        return;
    }
    const u32 size = IndexSize(p.index_type);
    const IndexRange range{p.index_address, static_cast<u64>(p.count) * size, p.index_type};
    const bool convert = p.mode == 7 || p.mode == 8 || p.mode == 2 ||
                         (p.index_type == 0 && !impl.device.IsExtIndexTypeUint8Supported());
    PreparedDraw prepared;
    HostIndex host_index;
    dbg_call = {p.count, p.instance_count != 0 ? p.instance_count : 1, p.base_vertex, true,
                p.index_address};
    if (impl.gpu_profiler) {
        prof_units = u64{p.count} * (p.instance_count != 0 ? p.instance_count : 1);
    }
    if (PrepareDraw(p.mode, p.count, convert ? nullptr : &range, prepared, host_index)) {
        const u32 instances = CostSplitInstances(p.instance_count != 0 ? p.instance_count : 1);
        vk::CommandBuffer cmd = host->HostCmd();
        if (convert) {
            u32 count = 0;
            if (GenerateIndices(prepared, &range, 0, p.count, p.base_vertex, host_index, count) &&
                count != 0) {
                cmd.DrawIndexed(CostSplitCount(count), instances, 0, p.base_vertex,
                                p.base_instance);
            }
        } else {
            cmd.DrawIndexed(CostSplitCount(p.count), instances, 0, p.base_vertex, p.base_instance);
        }
    }
    host->HostEndRenderCondition();
}

u64 DrawContext::ProfStageHash(u32 stage) const {
    const auto it = resources.program_hashes.find(stages[stage].program_serial);
    return it == resources.program_hashes.end() ? 0 : it->second[stage];
}

void DrawContext::ProfBeginPass(const RenderPassKey& key, VkExtent2D extent, u32 layers,
                                u32 samples) {
    prof_pass = {};
    GpuProf::Flight* const prof = ProfFlight();
    if (!prof) {
        return;
    }
    u32 colors = 0;
    s32 format = 0;
    for (const s32 color : key.color) {
        if (color != 0) {
            format = colors == 0 ? color : format;
            ++colors;
        }
    }
    prof_pass.tag =
        GpuProf::Tag{.kind = GpuProf::Kind::Pass,
                     .format = format != 0 ? format : key.depth,
                     .width = extent.width,
                     .height = extent.height,
                     .shape = GpuProf::PackShape(colors, key.depth != 0, layers, samples)};
    prof_pass.begin = prof->Stamp(host->HostCmd());
    prof_pass.last = prof_pass.begin;
}

void DrawContext::ProfOnDraw() {
    GpuProf::Flight* const prof = ProfFlight();
    const u64 units = std::exchange(prof_units, 0);
    ProfPass& pass = prof_pass;
    if (!prof || pass.begin == GpuProf::NONE) {
        return;
    }
    const u64 vertex = ProfStageHash(0);
    const u64 fragment = ProfStageHash(1);
    if (pass.draws++ == 0) {
        pass.tag.hash_a = vertex;
        pass.tag.hash_b = fragment;
    }
    pass.tag.zero_size |= prof_zero_size;
    if (cs_mode != CostSplit::Mode::Count) {
        ++pass.split;
    }
    pass.units += units;
    if (!prof->PerDraw()) {
        return;
    }
    // Everything recorded since the last timestamp was the previous draw.
    ProfCloseDraw(*prof);
    pass.draw_open = true;
    pass.draw_units = units;
    pass.draw_tag = pass.tag;
    pass.draw_tag.kind = GpuProf::Kind::Draw;
    pass.draw_tag.count = 0;
    pass.draw_tag.zero_size = prof_zero_size;
    pass.draw_tag.hash_a = vertex;
    pass.draw_tag.hash_b = fragment;
}

void DrawContext::ProfCloseDraw(GpuProf::Flight& prof) {
    ProfPass& pass = prof_pass;
    if (!pass.draw_open) {
        return;
    }
    pass.draw_open = false;
    const u32 now = prof.Stamp(host->HostCmd(), true);
    prof.Add(pass.last, now, pass.draw_tag, pass.draw_units);
    if (now != GpuProf::NONE) {
        pass.last = now;
    }
}

void DrawContext::ProfAfterEndPass(GpuProf::Flight& prof) {
    ProfPass& pass = prof_pass;
    const u32 end = prof.Stamp(host->HostCmd());
    pass.tag.count = pass.draws;
    prof.Add(pass.begin, end, pass.tag, pass.units, pass.split);
    pass = {};
}

} // namespace Vulkan::Nvn
