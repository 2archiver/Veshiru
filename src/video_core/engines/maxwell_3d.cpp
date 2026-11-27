// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <fmt/ranges.h>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/macro_dma_forward.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_list_macro_trace.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

// EDEN_NX2_3D_METHODS: log the first write to each 3D register (finds methods Eden doesn't model).
static const bool g_trace_first_method_writes = std::getenv("EDEN_NX2_3D_METHODS") != nullptr;

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    reports_fast_draw_breaks = true;
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    fast_maxwell3d = !record_write_history &&
                     !trace_first_method_writes && !VideoCommon::Nx2Profile::Enabled();
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

bool Maxwell3D::IsFastDrawParameter(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.size):
    case MAXWELL3D_REG_INDEX(const_buffer.address_high):
    case MAXWELL3D_REG_INDEX(const_buffer.address_low):
    case MAXWELL3D_REG_INDEX(const_buffer.offset):
    case MAXWELL3D_REG_INDEX(global_base_vertex_index):
    case MAXWELL3D_REG_INDEX(global_base_instance_index):
    case MAXWELL3D_REG_INDEX(vertex_id_base):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.format):
        return true;
    default:
        return DrawManager::IsInstanceLoopMethod(method) || IsFastDrawDeltaMethod(method) ||
               IsFastDrawVertexBuffer(method) || IsFastDrawVertexFormat(method) ||
               IsFastDrawQueryMethod(method);
    }
}

bool Maxwell3D::IsFastDrawQueryMethod(u32 method) {
    // Neither is in what a delta draw keeps: the draw manager asks ShouldExecute for every draw,
    // every draw resumes the counters and host conditional rendering (NotifySegment,
    // CounterEnable), and a resolve that leaves the render pass ends the delta chain there.
    return method == MAXWELL3D_REG_INDEX(clear_report_value) ||
           method == MAXWELL3D_REG_INDEX(render_enable.mode);
}

bool Maxwell3D::IsFastDrawVertexFormat(u32 method) {
    // With dynamic vertex input only each attribute's type class is in the pipeline key, which
    // the reusable draw compares (TakeReusableDraw); the rest is recorded for every draw.
    constexpr u32 formats = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    return VideoCommon::Nx2FastDraw::g_dynamic_vertex_input.load(std::memory_order_relaxed) &&
           method >= formats && method < formats + Regs::NumVertexAttributes;
}

bool Maxwell3D::IsFastDrawVertexBuffer(u32 method) {
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 stream_words = sizeof(Regs::VertexStream) / sizeof(u32);
    if (method >= streams && method < streams + Regs::NumVertexArrays * stream_words) {
        // address_high and address_low; stride, enable and frequency where they are dynamic.
        const u32 word = (method - streams) % stream_words;
        return word == 1 || word == 2 ||
               VideoCommon::Nx2FastDraw::g_dynamic_vertex_streams.load(std::memory_order_relaxed);
    }
    constexpr u32 sizes = 0x600 / sizeof(u32); // SET_VERTEX_STREAM_SIZE_A/B (NativeVertexStreamSize)
    constexpr u32 limits = MAXWELL3D_REG_INDEX(vertex_stream_limits);
    constexpr u32 limit_words = sizeof(Regs::VertexStreamLimit) / sizeof(u32);
    return (method >= sizes && method < sizes + Regs::NumVertexArrays * 2) ||
           (method >= limits && method < limits + Regs::NumVertexArrays * limit_words);
}

bool Maxwell3D::IsFastDrawDeltaMethod(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + Regs::NumCBData) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (!DrawManager::IsInstanceLoopMethod(method) && !IsFastDrawDeltaMethod(method) &&
        !IsFastDrawQueryMethod(method)) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Method, method);
    }
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

void Maxwell3D::CallStateMethod(Core::System&, u32 method, u32 method_argument) {
    DEBUG_ASSERT(executing_macro == 0);
    DEBUG_ASSERT(method < MacroRegistersStart && !execution_mask[method]);
    RecordWrite(method, method_argument, 1, 1);
    if (g_trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, 1);
        if (regs.reg_array[method] == method_argument) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::SameValueWrites, 1);
        }
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
        ProcessDirtyRegisters(method, argument);
    }
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    // About half of the split back's 3D writes (~5 million a second in Ocarina of Time) are state
    // registers: CallMethodFast's state write, here without the calls to it. A register not in
    // execution_mask is not SET_SHADOW_RAM_CONTROL either (CallMethodSlow's). The verify switch
    // takes the longer path below; the profile, the write history and the first-write trace
    // leave fast_maxwell3d off.
    if (fast_maxwell3d && executing_macro == 0 && method < MacroRegistersStart &&
        !execution_mask[method] && !perf_verify) {
        return WriteStateRegister(method, method_argument);
    }
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (perf_verify) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    if (!execution_mask[method]) {
        return WriteStateRegister(method, method_argument);
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (perf_verify) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    if (!WriteBoundConstantBuffer(offset, values.data(), static_cast<u32>(bytes))) {
        if (VideoCommon::Nx2FastDraw::Enabled()) {
            NoteFastDrawConstantData(address + offset, address + offset + bytes);
        }
        memory_manager.WriteWordsCached(address + offset, values.data(), count);
    }
    regs.const_buffer.offset += static_cast<u32>(bytes);
}

void Maxwell3D::WriteStateRegisters(Core::System& system, std::span<const u64, 4> flags,
                                    std::span<const u64> writes, u32 last_break) {
    ConsumeSink(system);
    // The front sends only writes that change its copy of the register, which is the back's but
    // for registers the back changes itself: each normally changes the register here too, and is
    // then all CallMethod would do for it. Otherwise, and where CallMethod takes other paths,
    // the writes go one by one.
    bool changes = fast_maxwell3d && executing_macro == 0 && !perf_verify &&
                   SkipSameValueWrites() &&
                   shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay &&
                   !draw_manager.HasDeferredDraw();
    for (size_t i = 0; changes && i < writes.size(); ++i) {
        changes = regs.reg_array[static_cast<u32>(writes[i])] != static_cast<u32>(writes[i] >> 32);
    }
    if (reg_block_verify) {
        return VerifyStateRegisters(system, flags, writes, last_break, changes);
    }
    if (!changes) {
        for (const u64 write : writes) {
            CallMethod(system, static_cast<u32>(write), static_cast<u32>(write >> 32), true);
        }
        return;
    }
    StoreStateRegisters(flags, writes);
    if (last_break != NoBreak && VideoCommon::Nx2FastDraw::Enabled()) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Register, last_break);
    }
}

void Maxwell3D::StoreStateRegisters(std::span<const u64, 4> flags, std::span<const u64> writes) {
    const auto control = shadow_state.shadow_ram_control;
    const bool tracked =
        control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter;
    for (const u64 write : writes) {
        const u32 method = static_cast<u32>(write);
        const u32 value = static_cast<u32>(write >> 32);
        regs.reg_array[method] = value;
        if (tracked) {
            shadow_state.reg_array[method] = value;
        }
    }
    for (size_t word = 0; word < flags.size(); ++word) {
        for (u64 bits = flags[word]; bits != 0; bits &= bits - 1) {
            dirty.flags[word * 64 + static_cast<size_t>(std::countr_zero(bits))] = true;
        }
    }
}

void Maxwell3D::VerifyStateRegisters(Core::System& system, std::span<const u64, 4> flags,
                                     std::span<const u64> writes, u32 last_break, bool changes) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    static u64 blocks{};
    static u64 block_writes{};
    static u64 by_write{};
    static u64 differences{};
    static u64 total_differences{};
    static u64 hidden_breaks{};
    static auto last_report = std::chrono::steady_clock::now();
    const auto one_by_one = [&] {
        for (const u64 write : writes) {
            CallMethod(system, static_cast<u32>(write), static_cast<u32>(write >> 32), true);
        }
    };
    const auto method_at = [&writes](size_t i) { return static_cast<u32>(writes[i]); };
    if (!changes) {
        ++by_write;
        one_by_one();
    } else {
        // The block's stores and flags, then (from the same state) each write as CallMethod,
        // which stays.
        ++blocks;
        block_writes += writes.size();
        std::vector<u32> before(writes.size() * 2);
        for (size_t i = 0; i < writes.size(); ++i) {
            before[i * 2] = regs.reg_array[method_at(i)];
            before[i * 2 + 1] = shadow_state.reg_array[method_at(i)];
        }
        const DirtyState::Flags flags_before = dirty.flags;
        StoreStateRegisters(flags, writes);
        std::vector<u32> stored(writes.size() * 2);
        for (size_t i = 0; i < writes.size(); ++i) {
            stored[i * 2] = regs.reg_array[method_at(i)];
            stored[i * 2 + 1] = shadow_state.reg_array[method_at(i)];
        }
        const DirtyState::Flags stored_flags = dirty.flags;
        for (size_t i = writes.size(); i-- > 0;) {
            regs.reg_array[method_at(i)] = before[i * 2];
            shadow_state.reg_array[method_at(i)] = before[i * 2 + 1];
        }
        dirty.flags = flags_before;
        // Other threads break fast draws too (fences, maps), into the same serial and last
        // break: mark the last break, and judge only a register break this thread left there or
        // the mark left alone. Restored when untouched. The mark is a register break of no
        // register, so a draw that reads it meanwhile counts it in range.
        constexpr u32 unset = static_cast<u32>(FastDraw::Reason::Register) | 0xffffff00u;
        const u32 previous = FastDraw::g_last_break.exchange(unset, std::memory_order_relaxed);
        one_by_one();
        u32 last = unset;
        FastDraw::g_last_break.compare_exchange_strong(last, previous, std::memory_order_relaxed);
        const bool expected_break = last_break != NoBreak && FastDraw::Enabled();
        const u32 expected_last =
            static_cast<u32>(FastDraw::Reason::Register) | (expected_break ? last_break << 8 : 0);
        const bool own_break = (last & 0xff) == static_cast<u32>(FastDraw::Reason::Register);
        if (last != unset && !own_break) {
            ++hidden_breaks;
        }
        const bool break_differs = last == unset ? expected_break : own_break && last != expected_last;
        std::string register_differences;
        for (size_t i = 0; i < writes.size(); ++i) {
            const u32 method = method_at(i);
            if (stored[i * 2] != regs.reg_array[method] ||
                stored[i * 2 + 1] != shadow_state.reg_array[method]) {
                register_differences += fmt::format(" {:#x}={:#x}/{:#x} (one by one {:#x}/{:#x})",
                                                    method, stored[i * 2], stored[i * 2 + 1],
                                                    regs.reg_array[method],
                                                    shadow_state.reg_array[method]);
            }
        }
        const DirtyState::Flags flag_differences = stored_flags ^ dirty.flags;
        if (!register_differences.empty() || flag_differences.any() || break_differs) {
            ++differences;
            if (++total_differences <= 16) {
                std::string flag_list;
                for (size_t i = 0; i < flag_differences.size(); ++i) {
                    if (flag_differences[i]) {
                        flag_list += fmt::format(" {}={}", i, stored_flags[i] ? 1 : 0);
                    }
                }
                LOG_ERROR(HW_GPU,
                          "Register block verify: a block of {} writes differs from them one by one: "
                          "registers (block/shadow){}, block dirty flags{}, fast draw break {} "
                          "{:#x} against {} {:#x}",
                          writes.size(), register_differences, flag_list, expected_break ? 1 : 0,
                          expected_last, last != unset ? 1 : 0, last != unset ? last : 0);
            }
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    LOG_INFO(HW_GPU,
             "Register block verify (EDEN_NX2_REG_BLOCK_VERIFY) in {:.1f} s: {} blocks of {} writes "
             "compared with the writes one by one, {} differ ({} in all), {} breaks not judged as "
             "another thread's break came last; {} blocks written one by one",
             seconds, blocks, block_writes, differences, total_differences, hidden_breaks, by_write);
    blocks = 0;
    block_writes = 0;
    by_write = 0;
    differences = 0;
    hidden_breaks = 0;
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // Every word of a macro register run is a parameter of the macro it started, whichever macro
    // register it writes (as on the split front, FrontCallMethod). Only a write to an ordinary
    // register before the macro has all its parameters is an error.
    if (executing_macro != 0 && method < MacroRegistersStart) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method >= MacroRegistersStart);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, 1);
        const bool same = regs.reg_array[method] == method_argument;
        if (same) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::SameValueWrites, 1);
        }
        VideoCommon::Nx2Profile::CountMethod(method, same);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::InlineData);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    if (VideoCommon::Nx2FastDraw::Enabled() &&
        (buffer.enabled != (bind_data.valid.Value() != 0) ||
         buffer.address != regs.const_buffer.Address() || buffer.size != regs.const_buffer.size)) {
        VideoCommon::Nx2FastDraw::NoteBind(stage_index, bind_data.shader_slot);
    }
    buffer.enabled = bind_data.valid.Value() != 0;
    if (VideoCommon::ListMacroTrace::g_enabled && buffer.enabled) {
        VideoCommon::ListMacroTrace::NoteConstBufferBind(
            static_cast<u32>(stage_index), bind_data.shader_slot, regs.const_buffer.Address(),
            regs.const_buffer.size);
    }
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

namespace {
// Texture and sampler headers are read when image views are resolved, which delta draws skip.
constexpr u64 FastDrawHeaderSize = 0x20;

u64 FastDrawHeadersSize(const Maxwell3D::Regs& regs) {
    return (u64{regs.tex_header.limit} + 1) * FastDrawHeaderSize;
}

// As TextureCache::SynchronizeDescriptors sizes the sampler table.
u64 FastDrawSamplersSize(const Maxwell3D::Regs& regs) {
    return (u64{regs.sampler_binding == Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding
                    ? regs.tex_header.limit
                    : regs.tex_sampler.limit} +
            1) *
           FastDrawHeaderSize;
}
} // Anonymous namespace

bool Maxwell3D::WriteBoundConstantBuffer(u32 offset, const void* data, u32 bytes) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    if (!FastDraw::Enabled()) {
        return false;
    }
    const GPUVAddr address = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    const u64 map_serial = memory_manager.MapSerial();
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = FastDrawHeadersSize(regs);
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = FastDrawSamplersSize(regs);
    BoundConstantBuffer& bound = bound_constant_buffer;
    if (bound.address != address || bound.size != size || bound.map_serial != map_serial ||
        bound.tic != tic || bound.tic_size != tic_size || bound.tsc != tsc ||
        bound.tsc_size != tsc_size) {
        bound = {address, size, map_serial, tic, tic_size, tsc, tsc_size, 0};
        const GPUVAddr end = address + size;
        const std::optional<DAddr> device =
            address != 0 && size != 0 && memory_manager.IsContinuousRange(address, size)
                ? memory_manager.GpuToCpuAddress(address)
                : std::nullopt;
        // Data anywhere in a buffer over the headers, at its GPU addresses or through another
        // mapping of their device memory, takes the general path, which checks each write.
        if (device && !(address < tic + tic_size && tic < end) &&
            !(address < tsc + tsc_size && tsc < end)) {
            fast_draw_header_spans.Update(memory_manager, tic, tic_size);
            fast_draw_sampler_spans.Update(memory_manager, tsc, tsc_size);
            if (!fast_draw_header_spans.Intersects(*device, *device + size) &&
                !fast_draw_sampler_spans.Intersects(*device, *device + size)) {
                bound.device = *device;
            }
        }
    }
    const GPUVAddr begin = address + offset;
    // Words on Switch 2 sparse memory go one at a time, as Nx2BackSparseWrite saw them.
    if (bound.device == 0 ||
        (begin + bytes > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
         begin < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed))) {
        return false;
    }
    const DAddr device = bound.device + offset;
    // Data memory already holds changes neither the texture handles nor the uniforms a delta draw
    // keeps from the draw before (Donkey Kong Bananza writes most of its handles again unchanged).
    // GPU-written pages can be newer than memory, and memory can be newer than what the last full
    // resolve read (a guest CPU write, watch change or sync since, or a page CPU writes to don't
    // notify): writes there count.
    if (Tegra::Nx2MayBeGpuWritten(device, bytes) ||
        !memory_manager.ConstantBufferDataMatches(device, data, bytes) ||
        !FastDraw::HandleMemoryUnchanged() || !memory_manager.IsDeviceRangeCached(device, bytes)) {
        FastDraw::NoteConstantData(begin, begin + bytes);
        FastDraw::NoteDeviceData(device, device + bytes);
    }
    memory_manager.WriteConstantBufferData(device, data, bytes);
    return true;
}

void Maxwell3D::NoteFastDrawConstantData(GPUVAddr begin, GPUVAddr end) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = FastDrawHeadersSize(regs);
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = FastDrawSamplersSize(regs);
    if ((begin < tic + tic_size && tic < end) || (begin < tsc + tsc_size && tsc < end)) {
        FastDraw::Break(FastDraw::Reason::TextureHeaders);
        return;
    }
    FastDraw::NoteConstantData(begin, end);
    // The same device memory can be mapped at other GPU addresses: compare where it lands.
    fast_draw_header_spans.Update(memory_manager, tic, tic_size);
    fast_draw_sampler_spans.Update(memory_manager, tsc, tsc_size);
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    for (GPUVAddr page = begin; page < end;) {
        const GPUVAddr page_end = (std::min)(end, (page & ~(PageSize - 1)) + PageSize);
        if (const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page)) {
            const DAddr device_end = *device + (page_end - page);
            if (fast_draw_header_spans.Intersects(*device, device_end) ||
                fast_draw_sampler_spans.Intersects(*device, device_end)) {
                FastDraw::Break(FastDraw::Reason::TextureHeaders);
                return;
            }
            FastDraw::NoteDeviceData(*device, device_end);
        }
        page = page_end;
    }
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        if (!WriteBoundConstantBuffer(regs.const_buffer.offset, start_base,
                                      static_cast<u32>(write_size))) {
            if (VideoCommon::Nx2FastDraw::Enabled()) {
                NoteFastDrawConstantData(address, address + write_size);
            }
            memory_manager.WriteBlockCached(address, start_base, write_size);
        }
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself. Macros
// that read memory run on the back instead (FrontRunOnBack), and the registers they can write
// are the back's until the front writes them or reads them back (FrontRegister).
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    std::vector<u32> dma_words; ///< FrontDmaReadEarly's words
    /// Nx2Perf dma_forward: registers the back writes from DMA words the front never read
    /// (regs holds a stale value), and macros (by index) seen doing an MME DMA read.
    std::bitset<Regs::NUM_REGS> unknown_regs;
    u32 unknown_count{};
    std::bitset<256> dma_macros;
    void MarkUnknown(u32 method) {
        if (!unknown_regs[method]) {
            unknown_regs[method] = true;
            ++unknown_count;
        }
    }
    void ForgetUnknown(u32 method) {
        if (unknown_regs[method]) {
            unknown_regs[method] = false;
            --unknown_count;
        }
    }
    /// Takes a forwarded register's value (and shadow copy) from the back, after draining it.
    void Learn(Maxwell3D& engine, VideoCommon::GpuSplit& split, u32 method) {
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
        regs[method] = engine.regs.reg_array[method];
        shadow[method] = engine.shadow_state.reg_array[method];
        ForgetUnknown(method);
    }
    u64 calls{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    /// The back holds state writes in its method sink, which its next executable method consumes
    /// (issuing a deferred draw a changed one ends). Front() consumes it before copying the state.
    bool sink_pending{};
    u32 split_start{}; ///< GpuSplit::Starts() when this copy was made
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// The pending DrawIndexedIndirect call's record, drawn by the host GPU (FrontTakeIndirectRecord).
    GPUVAddr indirect_record{};
    /// Per macro: 0 unknown, 1 DrawIndexedIndirect, 2 something else (for indirect_generation).
    std::vector<u8> indirect_macros;
    u64 indirect_generation{~u64{0}};
    /// State writes go to the back in blocks (GpuSplit::StateWrite): skip_same, fast_maxwell3d
    /// and not EDEN_NX2_PERF_VERIFY, read once.
    bool state_blocks{};
    /// The back's draw manager is in an instance loop (DrawMode::Instance): draw.end does nothing.
    bool instance_loop{};
    bool same_first{};          ///< Nx2Perf FrontSameFirst, read once
    bool class_table{};         ///< Nx2Perf FrontClassTable, read once
    bool draw_begin_coalesce{}; ///< Nx2Perf DrawBeginCoalesce, read once
    bool verify{};              ///< Nx2Perf::Verify(), read once
    /// Cached draw-program proof key; invalidated by program/raster state changes.
    bool draw_key_valid{};
    bool draw_key_native{};
    bool draw_non_writer{};
    u64 draw_key_epoch{};
    u64 draw_key_version{};
    u64 draw_key_map{};
    /// A writer draw or epoch change makes later entries in this draw window writers too.
    bool draw_window_writer{};
    u64 draw_window_epoch{};

    /// Registers a macro the back ran may have written (FrontRunOnBack), and their shadow RAM
    /// entries: the ring position after the last such macro, 0 once the front knows the value.
    std::array<u64, Regs::NUM_REGS> back_regs{};
    std::array<u64, Regs::NUM_REGS> back_shadow{};
    std::bitset<Regs::NUM_REGS> back_owned; ///< back_regs or back_shadow set
    struct BackMacro {
        u32 start{};
        bool back{};                           ///< Runs on the back (else on the front)
        std::vector<u32> writes;               ///< The registers its code can write
        std::bitset<Regs::NUM_REGS> write_set; ///< The same, for the back's check
        std::atomic<bool> escaped{};           ///< The back saw it write another register
    };
    /// By start in the macro code of back_generation, once called.
    std::vector<std::unique_ptr<BackMacro>> back_macros;
    u64 back_generation{~u64{0}};
    std::vector<u32> back_demoted; ///< Starts of macros that wrote outside their write set
    /// The back's macro engine, its code a copy of the front's (when back_code_copied).
    std::unique_ptr<MacroEngine> back_engine;
    bool back_code_copied{};
    u64 back_last{};                 ///< The position after the last macro sent to the back
    std::atomic<u32> back_escapes{}; ///< Back macros that wrote outside their write set
    u32 back_escapes_seen{};
};

namespace {

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
bool FrontVerifyEnabled() {
    static const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    return verify;
}

struct FrontMacroSwitches {
    bool batch;
    bool borrow;
    bool keep_state;
    bool verify;
};

const FrontMacroSwitches& FrontMacroOptions() {
    using VideoCommon::Nx2Perf::On;
    using VideoCommon::Nx2Perf::Opt;
    static const FrontMacroSwitches switches{On(Opt::MacroBatch), On(Opt::MacroBorrow),
                                             On(Opt::MacroState), VideoCommon::Nx2Perf::Verify()};
    return switches;
}

/// A macro DMA read of `words` at `source`: whether the caches held them (`drained`), and else
/// whether it waited for queued writers and what it read before the drain (`undrained`).
void CheckDmaRead(u32 macro, GPUVAddr source, bool drained, bool waited,
                  std::span<const u32> undrained, std::span<const u32> words) {
    static u64 reads{};
    static u64 undrained_reads{};
    static u64 waits{};
    static u64 differences{};
    static u64 total_differences{};
    static std::chrono::steady_clock::time_point last_report{};
    ++reads;
    if (!drained) {
        ++undrained_reads;
        waits += waited ? 1 : 0;
        if (!std::ranges::equal(undrained, words)) {
            ++differences;
            if (++total_differences <= 16) {
                const size_t shown = std::min<size_t>(words.size(), 8);
                LOG_WARNING(HW_GPU,
                            "Macro DMA read check: macro {:#x} read {} words at {:#x} as {:08x} "
                            "without draining the split, as {:08x} after it",
                            macro, words.size(), source, fmt::join(undrained.first(shown), ","),
                            fmt::join(words.first(shown), ","));
            }
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_report == std::chrono::steady_clock::time_point{}) {
        last_report = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    if (seconds < 10.0) {
        return;
    }
    last_report = now;
    LOG_INFO(HW_GPU,
             "Macro DMA read checks (EDEN_NX2_DMA_READ_VERIFY) in {:.1f} s: {} reads, {} "
             "without draining the split ({} waited for queued writers), {} of those with other "
             "words than after a drain ({} in all)",
             seconds, reads, undrained_reads, waits, differences, total_differences);
    reads = 0;
    undrained_reads = 0;
    waits = 0;
    differences = 0;
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

// What an executable method may write on the back, besides a deferred draw it may issue
// (GpuSplit::Effect). Constant buffer data and reports write ranges the front notes.
VideoCommon::GpuSplit::Effect FrontMethodEffect(u32 method) {
    using Effect = VideoCommon::GpuSplit::Effect;
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return Effect::None;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return Effect::None;
    default:
        return FrontHandledOutsideDrawManager(method) ? Effect::Unknown : Effect::Gpu;
    }
}

[[nodiscard]] constexpr bool IsDrawMethod(u32 method) noexcept {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
        return true;
    default:
        return false;
    }
}

// Nx2Perf FrontClassTable: cache whether a method goes outside the draw manager, its memory
// effect, whether it submits a draw, and whether it uploads a constant-buffer word.
constexpr u8 ClassOutsideDrawManager = 1 << 0;
constexpr u32 ClassEffectShift = 1;
constexpr u8 ClassDraw = 1 << 3;
constexpr u8 ClassCbData = 1 << 4;

u8 ComputeFrontClass(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    return static_cast<u8>((FrontHandledOutsideDrawManager(method) ? ClassOutsideDrawManager : 0) |
                           (static_cast<u8>(FrontMethodEffect(method)) << ClassEffectShift) |
                           (IsDrawMethod(method) ? ClassDraw : 0) |
                           (method >= cb_data && method < cb_data + 16 ? ClassCbData : 0));
}

const std::array<u8, Maxwell3D::Regs::NUM_REGS> front_class_table = [] {
    std::array<u8, Maxwell3D::Regs::NUM_REGS> table{};
    for (u32 method = 0; method < table.size(); ++method) {
        table[method] = ComputeFrontClass(method);
    }
    return table;
}();

u8 FrontClass(u32 method, bool verify) {
    const u8 cached = front_class_table[method];
    if (verify) [[unlikely]] {
        const u8 direct = ComputeFrontClass(method);
        if (direct != cached) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf front: class table method {:#x} has {:#x}, functions {:#x}",
                          method, cached, direct);
            }
            return direct;
        }
    }
    return cached;
}

VideoCommon::GpuSplit::Effect ClassEffect(u8 cls) {
    return static_cast<VideoCommon::GpuSplit::Effect>((cls >> ClassEffectShift) & 3);
}

[[nodiscard]] constexpr bool IsDrawProgramRegister(u32 method) noexcept {
    constexpr u32 pipelines = MAXWELL3D_REG_INDEX(pipelines);
    constexpr u32 pipelines_end =
        pipelines + static_cast<u32>(sizeof(Maxwell3D::Regs::pipelines) / sizeof(u32));
    constexpr u32 program_region = MAXWELL3D_REG_INDEX(program_region);
    return (method >= pipelines && method < pipelines_end) || method == program_region ||
           method == program_region + 1 || method == MAXWELL3D_REG_INDEX(rasterize_enable) ||
           method == MAXWELL3D_REG_INDEX(transform_feedback_enabled) ||
           method == Maxwell3D::Regs::MeshControl;
}

[[nodiscard]] bool FrontDrawsOnly(u32 method) noexcept {
    return !FrontHandledOutsideDrawManager(method) &&
           method != MAXWELL3D_REG_INDEX(clear_surface) &&
           method != MAXWELL3D_REG_INDEX(draw_texture.src_y0);
}

template <typename DrawEffect>
VideoCommon::GpuSplit::Effect FrontExecEffect(VideoCommon::GpuSplit::Effect effect, u32 method,
                                              u8 cls, bool class_table, DrawEffect&& draw_effect) {
    using Effect = VideoCommon::GpuSplit::Effect;
    using VideoCommon::SplitWriteTracker;
    effect = SplitWriteTracker::Combine(effect,
                                        class_table ? ClassEffect(cls) : FrontMethodEffect(method));
    Effect method_effect = FrontMethodEffect(method);
    if (method_effect == Effect::Gpu && FrontDrawsOnly(method)) {
        method_effect = draw_effect();
    }
    return SplitWriteTracker::Combine(effect, method_effect);
}

// Where the next constant buffer data word goes (ProcessCBMultiData), from the front's registers.
GPUVAddr ConstBufferWriteAddress(std::span<const u32> regs) {
    return ((GPUVAddr{regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)]) +
           regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
}

// How many of the next constant buffer data words land before the bank's end (ProcessCBMultiData
// drops the rest).
u32 ConstBufferRoomWords(std::span<const u32> regs) {
    const u32 offset = regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
    const u32 size = regs[MAXWELL3D_REG_INDEX(const_buffer.size)];
    return offset < size ? (size - offset) / sizeof(u32) : 0;
}

// A report and what it may write: payload and timestamp.
GPUVAddr ReportAddress(std::span<const u32> regs) {
    return (GPUVAddr{regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
           regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
}
constexpr u64 ReportBytes = 16;

// Counters and timed releases are written when the host query completes, not when the back runs
// them: early MME DMA reads of their bytes wait until the back ran the report and the result
// landed (lichii #179, 166c21d96c). Short payload releases land when the back runs them.
void NoteLateReport(VideoCommon::GpuSplit& split, const MemoryManager& memory,
                    std::span<const u32> regs) {
    using Report = Maxwell3D::Regs::ReportSemaphore;
    decltype(Report::query) query{};
    query.raw = regs[MAXWELL3D_REG_INDEX(report_semaphore.query)];
    if (query.operation == Report::Operation::Release && query.short_query != 0 &&
        query.report == Report::Report::Payload) {
        return;
    }
    if (const auto device = memory.GpuToCpuAddress(ReportAddress(regs))) {
        split.NoteAsyncWrite(*device, ReportBytes);
    }
}

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

// Whether a macro the split's back runs may write `method` (FrontRunOnBack): the front then only
// loses track of the register's value. Not the methods with effects the front tracks (macro code
// and bindings, the shadow RAM mode, draws, constant buffer data, reports, uploads, syncpoints),
// nor the registers it reads to note their memory writes, nor registers the back changes itself,
// nor MME_DMA_READ, whose words land in the front's data RAM.
bool BackWritable(const Maxwell3D& maxwell3d, u32 method) {
    constexpr u32 const_buffer = MAXWELL3D_REG_INDEX(const_buffer);
    constexpr u32 report = MAXWELL3D_REG_INDEX(report_semaphore);
    constexpr u32 upload = MAXWELL3D_REG_INDEX(upload);
    constexpr u32 upload_end = MAXWELL3D_REG_INDEX(launch_dma);
    if (method - const_buffer < 4 || method - report < 4 || method - upload < upload_end - upload) {
        return false;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(load_mme.start_address_ptr): // Advanced by load_mme.start_address
    case MAXWELL3D_REG_INDEX(shadow_scratch):             // Set by falcon[4]
    case MAXWELL3D_REG_INDEX(logic_op.enable):            // Cleared by draws on AMD
    case 0x157:
        return false;
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true; // Executable, with effects on the back only
    default:
        return method < MacroRegistersStart && !maxwell3d.execution_mask[method];
    }
}

// split.WaitUntil for a macro the back runs in the front's place (FrontRunOnBack), counted for the
// async read-back line (EDEN_NX2_GPU_PROFILE).
bool WaitForBackMacros(VideoCommon::GpuSplit& split, u64 position, VideoCommon::DrainSite site) {
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return split.WaitUntil(position, site);
    }
    const u64 start = VideoCore::AsyncReadback::Now();
    const bool waited = split.WaitUntil(position, site);
    if (waited) {
        VideoCore::AsyncReadback::CountBackWait(VideoCore::AsyncReadback::Now() - start);
    }
    return waited;
}

} // Anonymous namespace

bool Maxwell3D::Nx2DrawProgramsOf(std::span<const u32> regs, bool native, u64 memory_id,
                                  u64 map_serial, Tegra::Nx2DrawPrograms& out) {
    if (regs.size() < Regs::NUM_REGS) {
        return false;
    }
    constexpr u32 rasterize = MAXWELL3D_REG_INDEX(rasterize_enable);
    constexpr u32 transform_feedback = MAXWELL3D_REG_INDEX(transform_feedback_enabled);
    constexpr u32 pipelines = MAXWELL3D_REG_INDEX(pipelines);
    constexpr u32 program_region = MAXWELL3D_REG_INDEX(program_region);
    constexpr u32 pipeline_words = sizeof(Regs::Pipeline) / sizeof(u32);
    // Without rasterization, with transform feedback, or with task and mesh programs, a draw
    // counts as a writer.
    if (regs[rasterize] == 0 || regs[transform_feedback] != 0 ||
        (native && (regs[Regs::MeshControl] & 1) != 0)) {
        return false;
    }
    out = {};
    out.memory_id = memory_id;
    out.map_serial = map_serial;
    out.native = native;
    const GPUVAddr base_addr = (GPUVAddr{regs[program_region]} << 32) | regs[program_region + 1];
    for (size_t index = 0; index < Regs::MaxShaderProgram; ++index) {
        const u32 pipeline = pipelines + static_cast<u32>(index) * pipeline_words;
        const bool enabled =
            index == static_cast<size_t>(Regs::ShaderType::VertexB) || (regs[pipeline] & 1) != 0;
        if (!enabled) {
            continue;
        }
        const GPUVAddr address =
            native ? (static_cast<u64>(regs[pipeline + 5] & 0xff) << 32) | regs[pipeline + 6]
                   : base_addr + regs[pipeline + 1];
        // Bit 63 tells an enabled program at address 0 from a disabled slot.
        out.addresses[index] = address | (u64{1} << 63);
    }
    return true;
}

VideoCommon::SplitWriteTracker::Effect Maxwell3D::FrontDrawEffect(VideoCommon::GpuSplit& split,
                                                                  FrontState& fs) {
    using Effect = VideoCommon::SplitWriteTracker::Effect;
    if (!VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet) ||
        fs.draw_window_writer) {
        return Effect::Gpu;
    }
    // The epoch before the proofs: a proof found below then holds for this epoch or is stale.
    const u64 epoch = Tegra::g_nx2_code_epoch.load(std::memory_order_acquire);
    const u64 version = Tegra::Nx2NonWriters().version.load(std::memory_order_acquire);
    const u64 map = memory_manager.MapSerial();
    const bool native = GetNativeMacroEngine().native_turing;
    if (!fs.draw_key_valid || fs.draw_key_epoch != epoch || fs.draw_key_version != version ||
        fs.draw_key_map != map || fs.draw_key_native != native) {
        fs.draw_key_valid = true;
        fs.draw_key_epoch = epoch;
        fs.draw_key_version = version;
        fs.draw_key_map = map;
        fs.draw_key_native = native;
        Tegra::Nx2DrawPrograms programs;
        fs.draw_non_writer =
            Nx2DrawProgramsOf(fs.regs, native, memory_manager.GetID(), map, programs) &&
            Tegra::Nx2IsNonWriterDraw(programs, epoch);
    }
    if (!fs.draw_non_writer || (fs.draw_window_epoch != 0 && fs.draw_window_epoch != epoch)) {
        fs.draw_window_writer = true;
        return Effect::Gpu;
    }
    fs.draw_window_epoch = epoch;
    split.DrawEpoch(epoch);
    return Effect::Draw;
}

bool Maxwell3D::FrontIsWriterDraw(std::span<const u32> regs) {
    if (regs.size() < Regs::NUM_REGS) {
        return true;
    }
    if (regs[MAXWELL3D_REG_INDEX(rasterize_enable)] == 0 ||
        regs[MAXWELL3D_REG_INDEX(transform_feedback_enabled)] != 0) {
        return true;
    }
    if (!Tegra::g_nx2_has_graphics_writer_shaders.load(std::memory_order_acquire)) {
        return false;
    }
    constexpr u32 pipelines = MAXWELL3D_REG_INDEX(pipelines);
    constexpr u32 pipeline_words = sizeof(Regs::Pipeline) / sizeof(u32);
    for (std::size_t stage = 0; stage < Regs::MaxShaderProgram; ++stage) {
        const u32 pipeline = pipelines + static_cast<u32>(stage) * pipeline_words;
        const bool enabled =
            stage == static_cast<size_t>(Regs::ShaderType::VertexB) || (regs[pipeline] & 1) != 0;
        if (!enabled) {
            continue;
        }
        const u32 addr_low = regs[pipeline + 6];
        if (Tegra::Nx2IsGraphicsWriterShader(addr_low)) {
            return true;
        }
    }
    return false;
}

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    // Copied during an earlier start: while the split was stopped, state writes went to the
    // method sink without reaching it (they hand the front back only once consumed).
    if (front && front->split_start != split.Starts()) [[unlikely]] {
        split.Drain(VideoCommon::DrainSite::FrontStart); // The engine's state is the back's.
        ReleaseFront();
    }
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        front->state_blocks = front->skip_same && fast_maxwell3d && !perf_verify;
        front->same_first = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FrontSameFirst);
        front->class_table = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FrontClassTable);
        front->draw_begin_coalesce =
            VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::DrawBeginCoalesce);
        front->verify = VideoCommon::Nx2Perf::Verify();
        front->split_start = split.Starts();
        front->executing_macro = executing_macro;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
    }
    return *front;
}

// With the split on, engine methods run on its back only: one here on another thread means the
// split stopped (accuracy settings changed mid-game). Hand the macros back: the front's copy
// goes stale from now on, and a restart copies the registers again (Front).
void Maxwell3D::ReleaseFront() {
    executing_macro = front->executing_macro;
    macro_params = std::move(front->parameters);
    // Entries macros the back ran may have written are the engine's already.
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (front->back_shadow[i] == 0) {
            shadow_state.reg_array[i] = front->shadow[i];
        }
    }
    shadow_state.shadow_ram_control = front->control;
    front.reset();
    front_macros = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: 3D macros back on the engine");
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    // As ProcessShadowRam does on the back. A register a macro the back ran may have written is
    // the front's again, and its write is sent even if the front's copies hold the value.
    const bool back_owned = fs.back_owned[method];
    const bool taken_over = back_owned && FrontTakeOver(split, fs, method);
    bool shadow_kept = !taken_over;
    const bool tracked = fs.control == Regs::ShadowRamControl::Track ||
                         fs.control == Regs::ShadowRamControl::TrackWithFilter;
    bool same_first = false;
    if (fs.same_first) {
        same_first = !execution_mask[method] && fs.skip_same &&
                     fs.control != Regs::ShadowRamControl::Replay && fs.regs[method] == argument &&
                     (!tracked || fs.shadow[method] == argument) && shadow_kept;
        if (same_first && !fs.verify) [[unlikely]] {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::MethodWrites, 1);
                Profile::Add(
                    from_macro ? Profile::Counter::MacroWrites : Profile::Counter::DirectWrites, 1);
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
    }
    u32 value = argument;
    if (tracked) {
        shadow_kept = shadow_kept && fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    if (fs.same_first && fs.verify) [[unlikely]] {
        const bool old_skip = !execution_mask[method] && fs.regs[method] == value &&
                              shadow_kept && fs.skip_same;
        if (old_skip != same_first) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf front: same-first method {:#x} argument {:#x} skip {}, "
                          "old {} (control {}, regs {:#x})", method, argument, same_first, old_skip,
                          static_cast<u32>(fs.control), fs.regs[method]);
            }
        }
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
        Profile::Add(from_macro ? Profile::Counter::MacroWrites : Profile::Counter::DirectWrites, 1);
    }
    using Effect = VideoCommon::GpuSplit::Effect;
    // What the entry may write on the back (FrontDmaRead): any method may issue a deferred draw
    // first.
    Effect effect = fs.draw_pending ? FrontDrawEffect(split, fs) : Effect::None;
    if (execution_mask[method] && FrontSkipsLoopWrite(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
        fs.regs[method] = value;
        if (IsDrawProgramRegister(method)) {
            fs.draw_key_valid = false;
        }
        // StateWrite applies these on the back as one state block, while preserving dirty flags
        // and ordering them with any pending draw or method sink.
        if (fs.state_blocks && !back_owned && fs.control != Regs::ShadowRamControl::Replay) {
            split.StateWrite(this, method, value, {dirty.tables[0][method], dirty.tables[1][method]},
                             !IsFastDrawParameter(method), fs.draw_pending);
            fs.sink_pending = false;
            return;
        }
        if (from_macro) {
            split.MacroWrite(this, method, argument, effect);
            fs.sink_pending = false;
        } else {
            split.Method(this, method, argument, is_last_call, nullptr, effect);
            fs.sink_pending = true;
        }
        return;
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    std::optional<GPUVAddr> cb_write;
    u32 cb_room{};
    fs.regs[method] = value;
    const u8 cls = fs.class_table ? FrontClass(method, fs.verify) : u8{0};
    if (IsDrawProgramRegister(method)) {
        fs.draw_key_valid = false;
    }
    effect = FrontExecEffect(effect, method, cls, fs.class_table,
                             [&] { return FrontDrawEffect(split, fs); });
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    switch (method) {
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
        // register, which a later Replay write reads back.
        fs.control = static_cast<Regs::ShadowRamControl>(argument);
        fs.shadow[method] = argument;
        break;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
        break;
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
        break;
    case MAXWELL3D_REG_INDEX(load_mme.start_address): {
        const auto positions = GetNativeMacroPositions();
        u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
        if (pointer < positions.size()) {
            positions[pointer] = value;
        }
        ++pointer;
        break;
    }
    case MAXWELL3D_REG_INDEX(falcon[4]):
        fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
        break;
    default:
        if (fs.class_table ? (cls & ClassCbData) != 0
                           : method >= cb_data && method < cb_data + 16) {
            cb_write = ConstBufferWriteAddress(fs.regs);
            cb_room = ConstBufferRoomWords(fs.regs);
            fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
        } else if (fs.class_table ? (cls & ClassOutsideDrawManager) == 0
                                  : !FrontHandledOutsideDrawManager(method)) {
            fs.draw_pending = true;
        }
        break;
    }
    if (fs.draw_begin_coalesce && method == MAXWELL3D_REG_INDEX(draw.begin) &&
        ((value >> 26) & 3) == 1) [[unlikely]] {
        // An instance loop's begin right after the same one only adds another instance on the
        // back. Verify mode sends the entry anyway and asks the back to check adjacency.
        if (split.RepeatableMethod(this, method, argument, is_last_call, from_macro, effect,
                                   fs.verify) &&
            Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::DrawBeginCoalesced, 1);
        }
        if (!from_macro && (fs.class_table ? (cls & ClassDraw) != 0 : IsDrawMethod(method)) &&
            FrontIsWriterDraw(fs.regs)) {
            split.NoteWriter();
        }
    } else if (from_macro) {
        split.MacroWrite(this, method, argument, effect);
    } else {
        split.Method(this, method, argument, is_last_call, nullptr, effect);
        if ((fs.class_table ? (cls & ClassDraw) != 0 : IsDrawMethod(method)) &&
            FrontIsWriterDraw(fs.regs)) {
            split.NoteWriter();
        }
    }
    fs.sink_pending = false;
    if (cb_write) {
        split.NoteConstBufferWrite(memory_manager, *cb_write, &value, 1, (std::min)(cb_room, 1U));
    } else if (from_macro || execution_mask[method]) {
        FrontNoteWrites(split, fs, method, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        FrontNoteReport(split, fs);
        NoteLateReport(split, memory_manager, fs.regs);
    }
}

void Maxwell3D::FrontNoteReport(VideoCommon::GpuSplit& split, const FrontState& fs) {
    // What ProcessQueryGet will write on the back: a payload report writes the payload, a counter
    // what the GPU counted.
    Regs::ReportSemaphore report;
    std::memcpy(&report, &fs.regs[MAXWELL3D_REG_INDEX(report_semaphore)], sizeof(report));
    const auto operation = report.query.operation.Value();
    if (operation != Regs::ReportSemaphore::Operation::Release &&
        operation != Regs::ReportSemaphore::Operation::ReportOnly) {
        return;
    }
    const auto device = memory_manager.GpuToCpuAddress(report.Address());
    if (!device) {
        return;
    }
    const bool payload = report.query.report == Regs::ReportSemaphore::Report::Payload;
    split.NoteSemaphoreWrite(*device, report.payload, payload);
    if (report.query.short_query == 0) {
        // 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, payload);
    }
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    if (front_run_record != nullptr) [[unlikely]] {
        front_run_record->push_back({method, argument, is_last_call ? 1U : 0U});
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMethodRun(Core::System& system, VideoCommon::GpuSplit& split,
                                   u32 method, const u32* words, u32 count, bool incrementing,
                                   u32 methods_pending) {
    // FrontCallMethod for each word, as DmaPusher::ProcessCommands calls it word by word.
    DEBUG_ASSERT(count <= methods_pending &&
                 (incrementing ? method + count : method + 1) <= MacroRegistersStart);
    auto& fs = Front(system, split);
    for (u32 i = 0; i < count; ++i) {
        FrontWrite(split, fs, incrementing ? method + i : method, words[i],
                   methods_pending - i <= 1, false);
    }
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        const auto& options = FrontMacroOptions();
        if (options.borrow && amount != 0 && amount == methods_pending &&
            fs.executing_macro == 0 && fs.parameters.empty()) {
            // The whole call in this run: the macro reads the words where they are.
            fs.executing_macro = method;
            if (!options.verify) [[likely]] {
                FrontRunMacro(system, split, fs, {base_start, amount});
                return;
            }
            static std::vector<u32> copy;
            copy.assign(base_start, base_start + amount);
            FrontRunMacro(system, split, fs, {base_start, amount});
            static u64 differences{};
            if (std::memcmp(copy.data(), base_start, amount * sizeof(u32)) != 0 &&
                ++differences <= 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf macro: macro {:#x}'s {} borrowed parameters changed during "
                          "the call ({} calls so far)",
                          method, amount, differences);
            }
            return;
        }
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words
    // (sent whatever the front's copies hold).
    if (fs.back_owned[method]) {
        FrontTakeOver(split, fs, method);
    }
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    using Effect = VideoCommon::GpuSplit::Effect;
    Effect effect = fs.draw_pending ? FrontDrawEffect(split, fs) : Effect::None;
    std::optional<GPUVAddr> cb_write;
    u32 cb_room{};
    const u8 cls = fs.class_table ? FrontClass(method, fs.verify) : u8{0};
    if (method >= cb_data && method < cb_data + 16) {
        cb_write = ConstBufferWriteAddress(fs.regs);
        cb_room = ConstBufferRoomWords(fs.regs);
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
        effect = Effect::Unknown;
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        FrontTrackInstanceLoop(fs, method, 0);
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
        effect = VideoCommon::SplitWriteTracker::Combine(effect, FrontDrawEffect(split, fs));
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            FrontTrackInstanceLoop(fs, method, value); // DrawBegin runs for every word
            if (IsDrawProgramRegister(method)) {
                fs.draw_key_valid = false;
            }
            if (!execution_mask[method]) {
                continue;
            }
            effect = FrontExecEffect(effect, method, cls, fs.class_table,
                                     [&] { return FrontDrawEffect(split, fs); });
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (fs.class_table ? (cls & ClassOutsideDrawManager) == 0
                                   : !FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr, effect);
    fs.sink_pending = false; // The back consumes the sink before a run.
    if (cb_write) {
        split.NoteConstBufferWrite(memory_manager, *cb_write, base_start, amount,
                                   (std::min)(cb_room, amount));
    } else {
        FrontNoteWrites(split, fs, method, amount);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount != 0) {
        FrontNoteReport(split, fs);
        NoteLateReport(split, memory_manager, fs.regs);
    }
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split, FrontState& fs,
                              std::span<const u32> borrowed) {
    namespace Profile = VideoCommon::Nx2Profile;
    namespace ListTrace = VideoCommon::ListMacroTrace;
    namespace DmaForward = VideoCommon::DmaForward;

    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    if (const GPUVAddr record = std::exchange(fs.indirect_record, GPUVAddr{});
        record != 0 && fs.parameters.size() == 6) {
        FrontRunIndirectRecord(system, split, fs, record);
        return;
    }

    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    const auto& options = FrontMacroOptions();
    const std::span<const u32> parameters =
        borrowed.empty() ? std::span<const u32>{fs.parameters} : borrowed;
    Profile::CountMethod(method, false);

    ListTrace::Call* const trace =
        ListTrace::g_enabled
            ? ListTrace::BeginCall(ListTrace::Engine::Maxwell3D,
                                   static_cast<EngineInterface*>(this), method, engine.native_code,
                                   engine.native_code_generation, start, parameters,
                                   native_data_ram, fs.regs)
            : nullptr;

    // EDEN_NX2_MACRO_TRACE=<macro index>: log the first 32 calls of one macro, its code once,
    // the parameters, and the method writes and DMA words in order.
    static const u32 trace_macro = [] {
        const char* value = std::getenv("EDEN_NX2_MACRO_TRACE");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : ~0U;
    }();
    static u32 traced_calls{};
    const u32 macro_index = (method - MacroRegistersStart) >> 1;
    const bool macro_trace = trace_macro != ~0U && macro_index == trace_macro && traced_calls < 32;
    std::string trace_line;
    if (macro_trace) [[unlikely]] {
        if (traced_calls == 0) {
            std::string code;
            for (u32 pc = start; pc < start + 96 && size_t{pc} * 3 + 2 < engine.native_code.size();
                 ++pc) {
                code += fmt::format(" {:08x}{:08x}{:08x}", engine.native_code[pc * 3 + 2],
                                    engine.native_code[pc * 3 + 1], engine.native_code[pc * 3]);
            }
            LOG_INFO(HW_GPU, "Nx2 macro trace {:#x}: start {:#x}, code (96-bit words):{}",
                     trace_macro, start, code);
        }
        trace_line = "params";
        for (size_t i = 0; i < parameters.size() && i < 24; ++i) {
            trace_line += fmt::format(" {:#x}", parameters[i]);
        }
        trace_line += " | writes";
    }

    // Nx2Perf dma_forward: a macro with a prior MME DMA read can run on the front without
    // waiting for its words when they only affect its last writes (currently render enable).
    std::optional<DmaForward::Plan> analyzed_plan;
    const DmaForward::Plan* dma_plan = nullptr;
    bool forwarded_call = false;
    bool verify_forward = false;
    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::DmaForward) && trace == nullptr &&
        !macro_trace && macro_index < fs.dma_macros.size() && fs.dma_macros[macro_index])
        [[unlikely]] {
        Profile::Scope profile_scope{Profile::Zone::Macro};
        static u64 forwarded_calls{};
        static u64 fallen_back{};
        static std::array<std::string_view, 8> reasons_logged{};
        const auto read_register = [&fs](u32 address, bool& unknown) -> u32 {
            if (address >= Regs::NUM_REGS) {
                unknown = false;
                return 0;
            }
            unknown = fs.unknown_count != 0 && fs.unknown_regs[address];
            return fs.regs[address];
        };
        const auto forwardable = [](u32 address) {
            return address == MAXWELL3D_REG_INDEX(render_enable.mode);
        };
        const auto analyze = [&](auto&& read) {
            return DmaForward::Analyze(std::span<const u32>{engine.native_code}, start, parameters,
                                       std::span<const u32>{native_data_ram}, MacroRegistersStart,
                                       std::forward<decltype(read)>(read), forwardable);
        };
        if (fs.control == Regs::ShadowRamControl::Replay) {
            // The back would write the shadow copy, not the words.
            static const DmaForward::Plan replay_plan = [] {
                DmaForward::Plan plan;
                plan.reason = "shadow RAM replay";
                return plan;
            }();
            dma_plan = &replay_plan;
        } else {
            dma_plan = &analyzed_plan.emplace(analyze(read_register));
        }

        if (!dma_plan->eligible) {
            ++fallen_back;
            const auto logged = std::ranges::find(reasons_logged, dma_plan->reason);
            const auto free = std::ranges::find(reasons_logged, std::string_view{});
            if (logged == reasons_logged.end() && free != reasons_logged.end()) {
                *free = dma_plan->reason;
                LOG_INFO(HW_GPU, "Nx2Perf dma_forward: macro {:#x} not forwarded: {}", macro_index,
                         dma_plan->reason);
            }
        } else if (options.verify) {
            verify_forward = true;
        } else {
            struct ForwardCall {
                Maxwell3D* engine;
                Core::System* system;
                MemoryManager* memory;
                DmaForward::Snapshot snapshot;

                void operator()() const {
                    std::array<u32, DmaForward::MaxDmaWords> words{};
                    const size_t bytes = size_t{snapshot.dma_words} * sizeof(u32);
                    if (memory->IsMemoryDirty(snapshot.dma_source, bytes,
                                              VideoCommon::CacheType::BufferCache)) {
                        memory->FlushRegion(snapshot.dma_source, bytes,
                                            VideoCommon::CacheType::BufferCache);
                    }
                    memory->FlushQueryResults(snapshot.dma_source, bytes);
                    memory->ReadWordsUnsafe(snapshot.dma_source, words.data(), snapshot.dma_words);
                    const bool ok = DmaForward::Recompute(
                        snapshot, std::span<const u32>{words.data(), snapshot.dma_words},
                        MacroRegistersStart, [this](u32 forwarded_method, u32 value) {
                            engine->ConsumeSink(*system);
                            engine->CallMethod(*system, forwarded_method, value, true);
                        });
                    static u32 failures{};
                    if (!ok && failures++ < 16) {
                        LOG_ERROR(HW_GPU,
                                  "Nx2Perf dma_forward: the back's run of a forwarded macro call "
                                  "failed or made other writes");
                    }
                }
            };

            for (size_t i = 0; i < dma_plan->num_writes; ++i) {
                const DmaForward::Write& write = dma_plan->writes[i];
                if (write.forwarded) {
                    if (i == dma_plan->snapshot.first_forwarded) {
                        split.Defer(
                            ForwardCall{this, &system, &memory_manager, dma_plan->snapshot});
                    }
                    fs.MarkUnknown(write.method);
                    continue;
                }
                // Queue the DMA read method itself, but let the forwarded entry read its words.
                FrontWrite(split, fs, write.method, write.value, true, true);
            }
            for (size_t i = 0; i < dma_plan->num_ram_writes; ++i) {
                native_data_ram[dma_plan->ram_writes[i].first] = dma_plan->ram_writes[i].second;
            }
            forwarded_call = true;
            if ((++forwarded_calls & 0xffff) == 0) {
                LOG_INFO(HW_GPU, "Nx2Perf dma_forward: {} macro calls forwarded, {} not",
                         forwarded_calls, fallen_back);
            }
        }
    }

    const bool on_back = !forwarded_call && !verify_forward && trace == nullptr && !macro_trace &&
                         FrontRunOnBack(system, split, fs, engine, start, parameters);
    if (on_back && macro_index < fs.dma_macros.size()) {
        fs.dma_macros[macro_index] = true;
    }
    std::array<std::array<u32, 2>, DmaForward::MaxWrites> verify_writes{};
    size_t verify_count{};
    std::array<u32, DmaForward::MaxDmaWords> verify_words{};
    u32 verify_word_count{};
    if (!on_back && !forwarded_call) {
        Profile::Scope profile_scope{Profile::Zone::Macro};
        engine.native_interpreter.active_dram = &native_data_ram;
        const size_t parameter_count = parameters.size();
        const auto read = [&](u32 address) {
            u32 value{};
            if (address < Regs::NUM_REGS) {
                if (fs.unknown_count != 0 && fs.unknown_regs[address]) [[unlikely]] {
                    fs.Learn(*this, split, address);
                }
                value =
                    fs.back_owned[address] ? FrontRegister(split, fs, address) : fs.regs[address];
            }
            ListTrace::NoteRead(trace, address, value);
            return value;
        };
        const auto same_value = [&fs, this](u32 address, u32 value) {
            if (address == 0x157 || address == 0x158 || execution_mask[address] || !fs.skip_same ||
                fs.back_owned[address]) {
                return false;
            }
            if (fs.control == Regs::ShadowRamControl::Track ||
                fs.control == Regs::ShadowRamControl::TrackWithFilter) {
                return fs.shadow[address] == value && fs.regs[address] == value;
            }
            const u32 stored =
                fs.control == Regs::ShadowRamControl::Replay ? fs.shadow[address] : value;
            return fs.regs[address] == stored;
        };
        const auto emit = [&](u32 address, u32 value) {
            ListTrace::NoteWrite(trace, address, value);
            if (macro_trace) [[unlikely]] {
                trace_line += fmt::format(" {:#x}={:#x}", address, value);
            }
            if (verify_forward && verify_count < verify_writes.size()) [[unlikely]] {
                verify_writes[verify_count++] = {address, value};
            }
            if (address >= MacroRegistersStart) {
                return false;
            }
            if (!options.batch || !same_value(address, value)) {
                FrontWrite(split, fs, address, value, true, true);
            } else if (!options.verify) [[likely]] {
                if (Profile::Enabled()) [[unlikely]] {
                    Profile::Add(Profile::Counter::MethodWrites, 1);
                    Profile::Add(Profile::Counter::MacroWrites, 1);
                    Profile::Add(Profile::Counter::SameValueWrites, 1);
                }
            } else {
                const u64 ring = split.WritePosition();
                const u32 reg = fs.regs[address];
                const u32 shadow = fs.shadow[address];
                const auto control = fs.control;
                FrontWrite(split, fs, address, value, true, true);
                static u64 differences{};
                if ((split.WritePosition() != ring || fs.regs[address] != reg ||
                     fs.shadow[address] != shadow || fs.control != control) &&
                    ++differences <= 64) {
                    LOG_ERROR(HW_GPU,
                              "Nx2Perf macro: macro {:#x} write {:#x} = {:#x} taken as same "
                              "value: register {:#x} -> {:#x}, shadow {:#x} -> {:#x}, ring "
                              "words {} ({} so far)",
                              macro_index, address, value, reg, fs.regs[address], shadow,
                              fs.shadow[address], split.WritePosition() - ring, differences);
                }
            }

            // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
            if (address == 0x157 || address == 0x158) {
                // A borrowed FIFO must be materialized if DMA words are appended to it.
                if (!borrowed.empty() && fs.parameters.empty()) {
                    fs.parameters.assign(borrowed.begin(), borrowed.end());
                }
                if (macro_index < fs.dma_macros.size()) {
                    fs.dma_macros[macro_index] = true;
                }
                const size_t parameters_before = fs.parameters.size();
                FrontDmaRead(split, fs, method, address, value);
                if (verify_forward && address == 0x158) [[unlikely]] {
                    for (size_t i = parameters_before;
                         i < fs.parameters.size() && verify_word_count < verify_words.size(); ++i) {
                        verify_words[verify_word_count++] = fs.parameters[i];
                    }
                }
                if (macro_trace) [[unlikely]] {
                    trace_line += " [dma";
                    if (address == 0x158) {
                        for (size_t i = parameters_before; i < fs.parameters.size(); ++i) {
                            trace_line += fmt::format(" {:#x}", fs.parameters[i]);
                        }
                    } else {
                        const u32 ram = fs.regs[0x156];
                        for (u32 i = 0; i < value && ram + i < native_data_ram.size(); ++i) {
                            trace_line +=
                                fmt::format(" ram[{:#x}]={:#x}", ram + i, native_data_ram[ram + i]);
                        }
                    }
                    trace_line += "]";
                }
            }
            return true;
        };

#ifdef ARCHITECTURE_x86_64
        if (!engine.native_jit) {
            engine.native_jit = std::make_shared<TuringMacroJit>();
        }
        engine.native_jit->Prepare(engine.native_code, engine.native_code_generation,
                                   options.batch);
        TuringMacroJit::Options jit_options{
            .batch_emits = options.batch,
            .keep_state = options.keep_state,
            .batch_excluded_method = MAXWELL3D_REG_INDEX(shadow_ram_control),
        };
        TuringMacroJit::Result result;
        if (trace != nullptr && trace->data_ram) {
            // The interpreter records data RAM accesses for ListMacroTrace; it needs owned FIFO
            // storage because it may append words while servicing MME_DMA_READ_FIFOED.
            if (!borrowed.empty() && fs.parameters.empty()) {
                fs.parameters.assign(borrowed.begin(), borrowed.end());
            }
            result = engine.native_interpreter.Execute(engine.native_code, start, fs.parameters,
                                                       read, emit);
        } else {
            result = engine.native_jit->Execute(start, fs.parameters, borrowed,
                                                std::span<u32>{native_data_ram}, read, emit,
                                                jit_options);
        }
#else
        if (!borrowed.empty() && fs.parameters.empty()) {
            fs.parameters.assign(borrowed.begin(), borrowed.end());
        }
        const auto result =
            engine.native_interpreter.Execute(engine.native_code, start, fs.parameters, read, emit);
#endif
        if (!result.success) {
            engine.native_execution_failed = true;
            LOG_ERROR(HW_GPU,
                      "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                      start, result.consumed, parameter_count, result.error);
        }

        if (verify_forward) [[unlikely]] {
            static u64 checked{};
            static u64 differed{};
            bool same = verify_count == dma_plan->num_writes;
            for (size_t i = 0; same && i < dma_plan->snapshot.first_forwarded; ++i) {
                same = verify_writes[i] ==
                       std::array<u32, 2>{dma_plan->writes[i].method, dma_plan->writes[i].value};
            }
            size_t index = dma_plan->snapshot.first_forwarded;
            const bool ran = DmaForward::Recompute(
                dma_plan->snapshot, std::span<const u32>{verify_words.data(), verify_word_count},
                MacroRegistersStart, [&](u32 forwarded_method, u32 value) {
                    same = same && index < verify_count &&
                           verify_writes[index] == std::array<u32, 2>{forwarded_method, value};
                    ++index;
                });
            same = same && ran && index == verify_count;
            ++checked;
            if (!same && differed++ < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf dma_forward: macro {:#x} call differs from the drained run "
                          "({} writes, plan {}; {} differences in {} calls)",
                          macro_index, verify_count, dma_plan->num_writes, differed, checked);
            }
            if ((checked & 0xffff) == 0) {
                LOG_INFO(HW_GPU, "Nx2Perf dma_forward: {} calls checked, {} differed", checked,
                         differed);
            }
        }
    }

    if (macro_trace) [[unlikely]] {
        LOG_INFO(HW_GPU, "Nx2 macro trace {:#x} call {}: {}", trace_macro, traced_calls,
                 trace_line);
        ++traced_calls;
    }
    if (trace != nullptr) {
        ListTrace::EndCall(*trace, on_back, engine.native_interpreter.data_ram_reads,
                           engine.native_interpreter.data_ram_writes);
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        const auto finish_effect = FrontDrawEffect(split, fs);
        fs.draw_pending = false;
        fs.draw_window_writer = false;
        fs.draw_window_epoch = 0;
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                    VideoCommon::GpuSplit::FastDrawNeutral, finish_effect);
        if (FrontIsWriterDraw(fs.regs)) {
            split.NoteWriter();
        }
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}
// A macro whose code reads memory into its FIFO (MME_DMA_READ_FIFOED) and writes only registers
// BackWritable allows runs on the back, at its point of the method stream: it reads the words as
// a drained front would, and the front goes on without waiting for the work queued before it.
// The registers it can write (and, in a tracked shadow RAM mode, their entries) are the back's
// until the front writes them (FrontTakeOver) or needs them (FrontRegister). Returns whether the
// call went to the back.
bool Maxwell3D::FrontRunOnBack(Core::System& system, VideoCommon::GpuSplit& split, FrontState& fs,
                               const MacroEngine& engine, u32 start,
                               std::span<const u32> parameters) {
    using DrainSite = VideoCommon::DrainSite;
    using BackMacro = FrontState::BackMacro;
    if (parameters.size() > 0x1000) {
        return false;
    }
    // A macro the back ran wrote another register: the front's copies may be stale anywhere. It
    // runs on the front from now on, and the front takes the back's registers (FrontVerify).
    const auto take_escapes = [&] {
        fs.back_escapes_seen = fs.back_escapes.load(std::memory_order_acquire);
        for (const auto& macro : fs.back_macros) {
            if (macro && macro->back && macro->escaped.load(std::memory_order_relaxed)) {
                macro->back = false;
                fs.back_demoted.push_back(macro->start);
                LOG_ERROR(HW_GPU,
                          "Split back macro at {:#x} wrote outside the registers its code "
                          "can write; it runs on the front from now on",
                          macro->start);
            }
        }
        FrontVerify(split, fs);
    };
    if (fs.back_generation != engine.native_code_generation) {
        // The macros sent before run the code they were sent with, and check their records.
        WaitForBackMacros(split, fs.back_last, DrainSite::BackCode);
        if (fs.back_escapes.load(std::memory_order_relaxed) != fs.back_escapes_seen) {
            take_escapes();
        }
        fs.back_macros.clear();
        fs.back_macros.resize(engine.native_code.size() / 3);
        fs.back_generation = engine.native_code_generation;
        fs.back_code_copied = false;
    } else if (fs.back_escapes.load(std::memory_order_relaxed) != fs.back_escapes_seen) {
        take_escapes();
    }
    if (start >= fs.back_macros.size()) {
        return false;
    }
    auto& known = fs.back_macros[start];
    if (!known) {
        known = std::make_unique<BackMacro>();
        known->start = start;
        const auto effects = TuringMacroInterpreter::AnalyzeEffects(engine.native_code, start);
        const auto& methods = effects.methods;
        const bool fifo_read = std::ranges::binary_search(methods, 0x158U);
        if (fifo_read || std::ranges::binary_search(methods, 0x157U)) {
            std::string reason;
            if (std::ranges::find(fs.back_demoted, start) != fs.back_demoted.end()) {
                reason = "it wrote another register on the back before";
            } else if (!effects.methods_known) {
                reason = "it computes the method of a write";
            } else if (effects.data_ram) {
                reason = "it uses the data RAM";
            } else if (const auto it = std::ranges::find_if(
                           methods, [this](u32 m) { return !BackWritable(*this, m); });
                       it != methods.end()) {
                reason = fmt::format("it writes {:#x}", *it);
            }
            known->back = reason.empty() && fifo_read;
            if (known->back) {
                known->writes = methods;
                for (const u32 written : methods) {
                    known->write_set.set(written);
                }
            }
            static u32 logged{};
            if (logged < 32) {
                ++logged;
                LOG_INFO(HW_GPU, "Split macro at {:#x} reads memory and runs on the {}: {}", start,
                         known->back ? "back" : "front",
                         known->back ? fmt::format("writes {:#x}", fmt::join(methods, ","))
                                     : reason);
            }
        }
    }
    if (!known->back) {
        return false;
    }
    if (!fs.back_engine) {
        fs.back_engine = std::make_unique<MacroEngine>(true);
        fs.back_engine->native_turing = true;
    }
    MacroEngine& back = *fs.back_engine;
    if (!fs.back_code_copied) {
        // Nothing sent with the old code is left to run (waited for above).
        back.native_code = engine.native_code;
        back.native_code_generation = engine.native_code_generation;
        fs.back_code_copied = true;
    }
    split.DeferWords(
        [this, system_ptr = &system, engine_ptr = &back, macro = known.get(),
         escapes = &fs.back_escapes](std::span<const u32> parameters) {
            // As ProcessMacro calls a macro, with the back's engine.
            namespace AsyncReadback = VideoCore::AsyncReadback;
            const bool profiled = VideoCommon::Nx2Profile::Enabled();
            const u64 start = profiled ? AsyncReadback::Now() : 0;
            ConsumeSink(*system_ptr);
            engine_ptr->split_writes = &macro->write_set;
            const u64 unexpected = engine_ptr->split_unexpected_writes;
            engine_ptr->Execute(*system_ptr, *this, macro->start, parameters);
            if (engine_ptr->split_unexpected_writes != unexpected) {
                macro->escaped.store(true, std::memory_order_relaxed);
                escapes->fetch_add(1, std::memory_order_release);
            }
            if (profiled) {
                AsyncReadback::CountBackMacro(AsyncReadback::Now() - start);
                AsyncReadback::MaybeReport();
            }
        },
        parameters);
    const u64 position = split.Position();
    fs.back_last = position;
    const bool tracked = fs.control == Regs::ShadowRamControl::Track ||
                         fs.control == Regs::ShadowRamControl::TrackWithFilter;
    for (const u32 written : known->writes) {
        fs.back_regs[written] = position;
        if (tracked) {
            fs.back_shadow[written] = position;
        }
        fs.back_owned[written] = true;
    }
    return true;
}

// A register as the back holds it once it ran what the front sent: the front's copy, or the
// back's once it ran the last macro there that can write it. No entry sent after that macro
// writes the register (a front write takes it over first, a later macro there would be the one
// waited for), so the back's value stays until the front writes the register again.
u32 Maxwell3D::FrontRegister(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    if (fs.back_regs[method] != 0) {
        WaitForBackMacros(split, fs.back_regs[method], VideoCommon::DrainSite::BackRegister);
        fs.regs[method] = regs.reg_array[method];
        fs.back_regs[method] = 0;
        fs.back_owned[method] = fs.back_shadow[method] != 0;
    }
    return fs.regs[method];
}

// Before the front writes `method`: the register, and its shadow RAM entry unless the mode
// leaves that alone, are the front's again; a replay takes the entry from the back first (as
// FrontRegister). Returns whether the back may hold another value or entry than the front's
// copies, so that the write must be sent even if it seems to change nothing.
bool Maxwell3D::FrontTakeOver(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    using Control = Regs::ShadowRamControl;
    const bool tracked = fs.control == Control::Track || fs.control == Control::TrackWithFilter;
    if (fs.back_shadow[method] != 0 && fs.control == Control::Replay) {
        WaitForBackMacros(split, fs.back_shadow[method], VideoCommon::DrainSite::BackRegister);
        fs.shadow[method] = shadow_state.reg_array[method];
        fs.back_shadow[method] = 0;
    }
    const bool differs = fs.back_regs[method] != 0 || (tracked && fs.back_shadow[method] != 0);
    fs.back_regs[method] = 0;
    if (tracked) {
        fs.back_shadow[method] = 0;
    }
    fs.back_owned[method] = fs.back_shadow[method] != 0;
    return differs;
}

void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}

bool Maxwell3D::FrontSkipsLoopWrite(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    FrontTrackInstanceLoop(fs, method, value);
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        break;
    default:
        return false;
    }
    // Sent while state writes wait in the back's sink: it consumes them at the next executable
    // method, and a changed one ends a deferred draw that must issue before the next engine's work
    // (a compute dispatch after a dropped vertex_buffer.count would otherwise run first).
    if (!fs.skip_same || fs.sink_pending || !shadow_kept || fs.regs[method] != value) {
        return false;
    }
    // Instance loop methods flush no deferred draw, and the draw manager ignores these: DrawEnd
    // draws nothing in an instance loop, and it only reads the vertex buffer first and count
    // (index_buffer writes also mark the index buffer dirty, so they are always sent).
    return method != MAXWELL3D_REG_INDEX(draw.end) || fs.instance_loop;
}

// What the back can write to guest memory running the entry just sent, `amount` words of `method`
// (GpuSplit::Writes::Noted), for the macro DMA reads after it (FrontDmaRead). The executable
// methods (IsMethodExecutable) not named here change registers and host state only: bindings,
// barriers, counter resets, the WFI's host query copies, shadow RAM and macro code.
void Maxwell3D::FrontNoteWrites(VideoCommon::GpuSplit& split, const FrontState& fs, u32 method,
                                u32 amount) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto address = [&fs](u32 high) {
        return (GPUVAddr{fs.regs[high]} << 32) | fs.regs[high + 1];
    };
    if (fs.draw_pending) {
        // A draw, or one an instance loop deferred to this entry (ProcessDirtyRegisters): where
        // its shaders store, and what the caches download for it, only the back knows.
        split.NoteAnyWrite();
        return;
    }
    if (method >= cb_data && method < cb_data + 16) {
        // FrontWrite and FrontCallMultiMethod advanced the offset past these words already.
        const u32 bytes = amount * static_cast<u32>(sizeof(u32));
        const u32 offset = fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] - bytes;
        split.NoteWrite(memory_manager,
                        address(MAXWELL3D_REG_INDEX(const_buffer.address_high)) + offset, bytes);
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        // CounterReport: the value and its timestamp, now or once the host result is in.
        split.NoteWrite(memory_manager, address(MAXWELL3D_REG_INDEX(report_semaphore.address_high)),
                        16);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        // Upload::State::ProcessData: lines of line_length_in bytes, pitch apart.
        const auto launch =
            std::bit_cast<Regs::LaunchDMA>(fs.regs[MAXWELL3D_REG_INDEX(launch_dma)]);
        const u32 lines = fs.regs[MAXWELL3D_REG_INDEX(upload.line_count)];
        const u32 length = fs.regs[MAXWELL3D_REG_INDEX(upload.line_length_in)];
        if (launch.memory_layout != Regs::LaunchDMA::Layout::Pitch) {
            split.NoteAnyWrite(); // Swizzled into the destination surface
        } else if (lines != 0 && length != 0) {
            split.NoteWrite(memory_manager, address(MAXWELL3D_REG_INDEX(upload.dest.address_high)),
                            u64{fs.regs[MAXWELL3D_REG_INDEX(upload.dest.pitch)]} * (lines - 1) +
                                length);
        }
        break;
    }
    case MAXWELL3D_REG_INDEX(render_enable.mode): {
        // A condition read from memory (ProcessQueryCondition) flushes every cache over it (also
        // under an override a macro the back ran may have set).
        constexpr u32 override_method = MAXWELL3D_REG_INDEX(render_enable_override);
        const auto mode =
            static_cast<Regs::RenderEnable::Mode>(fs.regs[MAXWELL3D_REG_INDEX(render_enable.mode)]);
        if ((fs.back_regs[override_method] != 0 ||
             static_cast<Regs::RenderEnable::Override>(fs.regs[override_method]) ==
                 Regs::RenderEnable::Override::UseRenderEnable) &&
            (mode == Regs::RenderEnable::Mode::Conditional ||
             mode == Regs::RenderEnable::Mode::IfEqual ||
             mode == Regs::RenderEnable::Mode::IfNotEqual)) {
            split.NoteAnyWrite();
        }
        break;
    }
    case MAXWELL3D_REG_INDEX(sync_info):
        // A syncpoint increment: macros read syncpoints through memory too (MemoryManager::Read).
        split.NoteAnyWrite();
        break;
    default:
        break;
    }
}

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    const GPUVAddr source = (GPUVAddr{FrontRegister(split, fs, 0x154) & 0x1ffffff} << 32) |
                            FrontRegister(split, fs, 0x155);
    const u32 ram = FrontRegister(split, fs, 0x156);
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadEarly) &&
        FrontDmaReadEarly(split, fs, macro, method, source, ram, words)) {
        return;
    }
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    // Drains the split, and has the back write out what its caches hold of the words: data the
    // GPU wrote (indirect draw arguments from a culling dispatch), reports the query cache holds
    // until their fence or host result (else the read returns stale occlusion counts).
    const auto write_out = [&] {
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
        if (VideoCommon::QueryDma::TraceEnabled()) {
            split.Run(
                [this, macro, method, source, read_bytes] {
                    if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer,
                                                                 source, read_bytes)) {
                        VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                     source, read_bytes);
                    }
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
        if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, read_bytes] {
                    memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
        split.Run(
            [this, source, read_bytes] {
                VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source,
                                                    read_bytes);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    };
    namespace AsyncReadback = VideoCore::AsyncReadback;
    const u64 wait_start = AsyncReadback::Now();
    // Only the work still queued that can write the words has to run first (FrontNoteWrites,
    // GpuSplit::WaitForWrites). Past it no entry can make them GPU-written or add reports there,
    // so the caches have nothing to write out unless they already hold it now.
    const bool waited = split.WaitForWrites(memory_manager, source, read_bytes);
    const bool drained =
        memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache) ||
        VideoCommon::QueryDma::HoldsReports(memory_manager, *rasterizer, source, read_bytes);
    if (drained) {
        write_out();
    }
    AsyncReadback::CountDmaRead(waited || drained, drained, AsyncReadback::Now() - wait_start);
    if (VideoCommon::Nx2Profile::Enabled()) {
        AsyncReadback::MaybeReport();
    }
    // EDEN_NX2_DMA_READ_VERIFY: a read made without draining the split drains it anyway and has
    // the caches write the words out, as a full drain would, and the macro gets those words.
    // CheckDmaRead counts the reads whose words differ from the ones read first.
    const bool verify = AsyncReadback::DmaReadVerify();
    std::vector<u32> undrained;
    std::vector<u32> checked;
    if (verify) {
        if (!drained) {
            for (u32 i = 0; i < words; ++i) {
                undrained.push_back(memory_manager.Read<u32>(source + u64{i} * sizeof(u32)));
            }
            write_out();
        }
        checked.reserve(words);
    }
    u32 changed{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
        if (verify) {
            checked.push_back(word);
        }
    }
    if (verify) {
        CheckDmaRead(macro, source, drained, waited, undrained, checked);
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

bool Maxwell3D::FrontDmaReadEarly(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                                  u32 method, GPUVAddr source, u32 ram, u32 words) {
    namespace Profile = VideoCommon::Nx2Profile;
    const u64 bytes = u64{words} * sizeof(u32);
    const bool sparse = source < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed) &&
                        Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) < source + bytes;
    if (words == 0 || sparse) {
        return false;
    }
    const u64 consumed = split.SettleWrites();
    // MacroReadWriterSet: wait only for the entries that can write memory, not for the draws
    // the back proved write none (checked again below, once the wait is over).
    const bool writer_set =
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet);
    std::array<u64, 5> pages;
    std::array<std::pair<u64, u64>, 5> read_ranges; // device bytes read in each page
    std::array<u64, 5> read_offsets;                 // their byte offsets in the read
    size_t page_count = 0;
    using WriteKind = VideoCommon::SplitWriteTracker::WriteKind;
    for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + bytes; page += 0x1000) {
        const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page);
        if (page_count == pages.size() || !device || Tegra::Nx2IsGpuWrittenRegion(*device) ||
            Tegra::Nx2IsGpuWritten(*device)) {
            return false;
        }
        const u64 device_page = *device & ~u64{0xfff};
        const GPUVAddr read_begin = (std::max)(page, source);
        const GPUVAddr read_end = (std::min)(page + 0x1000, source + bytes);
        read_ranges[page_count] = {device_page + (read_begin - page), read_end - read_begin};
        read_offsets[page_count] = read_begin - source;
        pages[page_count++] = device_page;
    }
    u64 target = 0;
    u64 write_target = 0;
    u64 other_write_target = 0; // noted writes other than constant buffer data
    for (size_t i = 0; i < page_count; ++i) {
        const u64 begin = pages[i];
        const u64 end = pages[i] + 0x1000;
        target = (std::max)(target, writer_set ? split.WriteTracker().WriterReadTarget(begin, end)
                                               : split.ReadTarget(begin, end));
        write_target = (std::max)(write_target, split.WriteTracker().LastWriteTo(begin, end));
        other_write_target =
            (std::max)(other_write_target,
                       split.WriteTracker().LastWriteTo(begin, end, WriteKind::Other));
    }
    // MacroReadCbOverlay: when the only queued writes noted to these pages are constant buffer
    // data, wait only for the unknown and GPU writers, then lay the queued words over what memory
    // holds. Invariant: after those, the entries still queued write the read bytes only through
    // noted writes (Effect::None), so no writer but the queued constant buffer words (kept in
    // order, by device address like the tracker) touches them before the back runs those words.
    // Async writes are checked below as before.
    const bool cb_overlay =
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadCbOverlay);
    const u64 unlocalized_target =
        (std::max)(split.WriteTracker().LastUnknown(), split.WriteTracker().LastGpu());
    const bool overlay = cb_overlay && write_target > unlocalized_target &&
                         other_write_target == 0 && split.ConstBufferWordsKept(consumed);
    if (overlay) {
        target = unlocalized_target;
    }
    const auto wait_reason = target == split.WriteTracker().LastUnknown()
                                 ? Profile::Counter::DmaReadsWaitUnknown
                             : target == write_target ? Profile::Counter::DmaReadsWaitWrite
                                                      : Profile::Counter::DmaReadsWaitGpu;
    split.WaitUntil(target, VideoCommon::DrainSite::MacroDmaWrites);
    const u64 consumed_now = split.ConsumedNow();
    // Draws still queued that the read did not wait for: their proofs must still hold (no
    // program dropped or deferred-written since they were classified), and no queued write
    // before them may reach proven code (the shader cache applies those at the next draw).
    const bool skipped_draws = writer_set && split.WriteTracker().LastGpu() > consumed_now;
    if (skipped_draws) {
        const u64 epoch = Tegra::g_nx2_code_epoch.load(std::memory_order_acquire);
        if (!split.WriteTracker().DrawsValid(consumed_now, epoch) ||
            split.WriteTracker().AnyPendingWrite(
                consumed_now, split.WriteTracker().LastGpu(),
                [](u64 begin, u64 end) { return Tegra::Nx2IsProvenCode(begin, end - begin); })) {
            if (Profile::Enabled()) {
                Profile::Add(overlay ? Profile::Counter::DmaReadsOverlayFallback
                                     : Profile::Counter::DmaReadsWriterSetFallback,
                             1);
            }
            return false;
        }
    }
    for (size_t i = 0; i < page_count; ++i) {
        if (Tegra::Nx2IsGpuWrittenRegion(pages[i]) || Tegra::Nx2IsGpuWritten(pages[i])) {
            return false;
        }
        // A counter report or delayed release to these bytes: read early only once the back ran
        // it and its result landed. The words in memory are then what a drained read sees.
        const auto [range_begin, range_size] = read_ranges[i];
        if (const u64 async_position = split.HasAsyncWrite(range_begin, range_size);
            async_position != 0 && (async_position > consumed_now ||
                                    !Tegra::Nx2AsyncReportsSettled(range_begin, range_size))) {
            return false;
        }
    }
    if (Profile::Enabled()) {
        Profile::Add(Profile::Counter::DmaReadBytes, bytes);
        Profile::Add(Profile::Counter::DmaReadsEarly, 1);
        if (target > consumed) {
            Profile::Add(wait_reason, 1);
            if (wait_reason == Profile::Counter::DmaReadsWaitWrite && cb_overlay) {
                Profile::Add(other_write_target != 0 ? Profile::Counter::DmaReadsWaitWriteOther
                             : !split.ConstBufferWordsKept(consumed)
                                 ? Profile::Counter::DmaReadsWaitWriteCbLost
                                 : Profile::Counter::DmaReadsWaitWriteCbOlder,
                             1);
            }
        }
        if (skipped_draws) {
            Profile::Add(Profile::Counter::DmaReadsWriterSet, 1);
        }
        if (overlay) {
            Profile::Add(target > consumed ? Profile::Counter::DmaReadsOverlayWaited
                                           : Profile::Counter::DmaReadsOverlayNoWait,
                         1);
        }
    }
    // Words the back ran up to consumed_now are in memory; the newer ones come from the overlay.
    const auto apply_overlay = [&](u32* out) {
        for (size_t i = 0; i < page_count; ++i) {
            split.OverlayConstBufferWords(consumed_now, read_ranges[i].first, read_ranges[i].second,
                                          reinterpret_cast<u8*>(out) + read_offsets[i]);
        }
    };
    if (!VideoCommon::Nx2Perf::Verify()) [[likely]] {
        if (overlay) {
            auto& words_read = fs.dma_words;
            words_read.resize(words);
            memory_manager.ReadWordsUnsafe(source, words_read.data(), words);
            apply_overlay(words_read.data());
            if (method == 0x158) {
                fs.parameters.insert(fs.parameters.end(), words_read.begin(), words_read.end());
            } else if (ram < native_data_ram.size()) {
                const size_t copy_words = (std::min<size_t>)(words, native_data_ram.size() - ram);
                std::memcpy(native_data_ram.data() + ram, words_read.data(),
                            copy_words * sizeof(u32));
            }
            return true;
        }
        if (method == 0x158) {
            const size_t old_size = fs.parameters.size();
            fs.parameters.resize(old_size + words);
            memory_manager.ReadWordsUnsafe(source, fs.parameters.data() + old_size, words);
        } else if (ram < native_data_ram.size()) {
            const size_t copy_words = (std::min<size_t>)(words, native_data_ram.size() - ram);
            memory_manager.ReadWordsUnsafe(source, native_data_ram.data() + ram, copy_words);
        }
        return true;
    }
    auto& read = fs.dma_words;
    read.resize(words);
    memory_manager.ReadWordsUnsafe(source, read.data(), words);
    if (overlay) {
        apply_overlay(read.data());
    }
    // The old path: drain, flush, read again. Its words are the ones used.
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    const bool verify_flush_buffers =
        memory_manager.IsMemoryDirty(source, bytes, VideoCommon::CacheType::BufferCache);
    split.Run(
        [memory = &memory_manager, source, bytes, verify_flush_buffers] {
            if (verify_flush_buffers) {
                memory->FlushRegion(source, bytes, VideoCommon::CacheType::BufferCache);
            }
            memory->FlushQueryResults(source, bytes);
        },
        VideoCommon::DrainSite::MacroDmaFlush);
    static u32 logged{};
    static u64 writer_set_reads{};
    static u64 writer_set_mismatches{};
    bool mismatch = false;
    for (u32 i = 0; i < words; ++i) {
        const u32 drained = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (drained != read[i]) {
            mismatch = true;
            if (logged < 64) {
                ++logged;
                LOG_ERROR(HW_GPU,
                          "Nx2Perf {}: macro {:#x} DMA read of {} words at {:#x}: word {} is "
                          "{:#x} without the drain, {:#x} after it",
                          skipped_draws ? "macro_read_writer_set" : "split", macro, words, source,
                          i, read[i], drained);
            }
        }
        read[i] = drained;
    }
    if (skipped_draws) {
        writer_set_mismatches += mismatch ? 1 : 0;
        if ((++writer_set_reads & 4095) == 0 || (mismatch && writer_set_mismatches <= 64)) {
            LOG_INFO(HW_GPU,
                     "Nx2Perf macro_read_writer_set: {} early reads past queued proven draws, {} "
                     "differed from the drained read",
                     writer_set_reads, writer_set_mismatches);
        }
    }
    for (u32 i = 0; i < words; ++i) {
        if (method == 0x158) {
            fs.parameters.push_back(read[i]);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = read[i];
        }
    }
    return true;
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    // What macros the back ran may have written, the front takes over (FrontRunOnBack).
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.back_regs[i] != 0) {
            fs.regs[i] = expected[i];
        }
        if (fs.back_shadow[i] != 0) {
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    fs.back_regs.fill(0);
    fs.back_shadow.fill(0);
    fs.back_owned.reset();
    static u64 control_differences{};
    if ((fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != shadow_state.reg_array[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.unknown_count != 0 && fs.unknown_regs[i]) {
            // Forwarded (dma_forward): the back's values are the right ones.
            fs.regs[i] = expected[i];
            fs.shadow[i] = shadow_state.reg_array[i];
            fs.ForgetUnknown(i);
            continue;
        }
        if (fs.regs[i] != expected[i]) {
            if (differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
            // The front's draw states were wrong: trust none of them until the window ends.
            fs.draw_key_valid = false;
            fs.draw_window_writer = true;
        }
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != shadow_state.reg_array[i]) {
            if (control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], shadow_state.reg_array[i]);
            }
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
}

void Maxwell3D::FrontRunIndirectRecord(Core::System& system, VideoCommon::GpuSplit& split,
                                       FrontState& fs, GPUVAddr record) {
    // DrawIndexedIndirect with a record GPU work may still write: the back draws it indirectly,
    // then the macro's own clean-up writes follow (zero bases, the driver constants at 0x640
    // cleared). The record's words in fs.parameters are not used.
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::IndirectRecords, 1);
    const u32 topology = fs.parameters[0];
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                    VideoCommon::GpuSplit::FastDrawNeutral);
    }
    split.Defer([this, system_ptr = &system, topology, record] {
        BackDrawIndexedIndirect(*system_ptr, topology, record);
    });
    split.NoteWriter();
    constexpr std::array<std::pair<u32, u32>, 7> cleanup{{
        {0x5f7, 0}, {0x446, 0}, {0x50d, 0}, {0x50e, 0}, {0x8e3, 0x640}, {0x8e4, 0}, {0x8e5, 0},
    }};
    for (const auto& [address, value] : cleanup) {
        FrontWrite(split, fs, address, value, true, true);
    }
}

bool Maxwell3D::IsIndirectDrawMacro(FrontState& fs, u32 method) {
#ifdef ARCHITECTURE_x86_64
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    if (positions.empty()) {
        return false;
    }
    const size_t index = ((method - MacroRegistersStart) >> 1) % positions.size();
    if (fs.indirect_generation != engine.native_code_generation ||
        fs.indirect_macros.size() != positions.size()) {
        fs.indirect_generation = engine.native_code_generation;
        fs.indirect_macros.assign(positions.size(), 0);
    }
    if (fs.indirect_macros[index] != 0) {
        return fs.indirect_macros[index] == 1;
    }
    // Dry run with recognizable parameters against a copy of the data RAM: NVN's
    // DrawIndexedIndirect writes the record's fields to these registers and the base vertex and
    // instance to the driver constants at 0x640 (as HLE_DrawIndexedIndirect's extended form),
    // draws once, and clears the bases again.
    constexpr u32 topology = 4, count = 0x111, instances = 1, first = 0x222, base_vertex = 0x333,
                  base_instance = 0x444;
    std::vector<u32> parameters{topology, count, instances, first, base_vertex, base_instance};
    constexpr std::array<std::pair<u32, u32>, 16> expected{{
        {0x5f7, first}, {0x446, base_vertex}, {0x50d, base_vertex}, {0x50e, base_instance},
        {0x8e3, 0x640}, {0x8e4, base_vertex}, {0x8e5, base_instance}, {0x586, topology},
        {0x5f8, count}, {0x585, 0}, {0x446, 0}, {0x50d, 0}, {0x50e, 0}, {0x8e3, 0x640},
        {0x8e4, 0}, {0x8e5, 0},
    }};
    std::vector<u32> scratch_ram(native_data_ram.begin(), native_data_ram.end());
    std::vector<std::pair<u32, u32>> writes;
    if (!engine.native_jit) {
        engine.native_jit = std::make_shared<TuringMacroJit>();
    }
    engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
    const auto result = engine.native_jit->Execute(
        positions[index], parameters, std::span<u32>{scratch_ram},
        [&fs](u32 address) { return address < Regs::NUM_REGS ? fs.regs[address] : 0U; },
        [&writes](u32 address, u32 value) {
            writes.emplace_back(address, value);
            return address < MacroRegistersStart && writes.size() < 64;
        });
    const bool match = result.success && writes.size() == expected.size() &&
                       std::equal(writes.begin(), writes.end(), expected.begin());
    fs.indirect_macros[index] = match ? 1 : 2;
    if (match) {
        LOG_INFO(HW_GPU, "Native macro {:#x} is DrawIndexedIndirect: GPU-written records are drawn "
                         "indirectly by the host GPU",
                 method);
    }
    return match;
#else
    (void)fs;
    (void)method;
    return false;
#endif
}

bool Maxwell3D::FrontTakeIndirectRecord(u32 parameter_method, GPUVAddr record) {
    // EDEN_NX2_INDIRECT_RECORDS=0: read the records on the CPU as before.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INDIRECT_RECORDS");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled || !front_macros || front == nullptr) {
        return false;
    }
    FrontState& fs = *front;
    if (fs.executing_macro == 0 || parameter_method != fs.executing_macro + 1 ||
        fs.parameters.size() != 1) {
        return false;
    }
    // Topologies the host draws indirectly as they are (no quad or polygon conversion).
    const u32 topology = fs.parameters[0];
    if (!(topology <= 6 || (topology >= 0xA && topology <= 0xD))) {
        return false;
    }
    if (!IsIndirectDrawMacro(fs, fs.executing_macro)) {
        return false;
    }
    fs.indirect_record = record;
    return true;
}

void Maxwell3D::BackDrawIndexedIndirect(Core::System& system, u32 topology, GPUVAddr record) {
    // State writes the front sent before this call wait in the method sink until an executable
    // method arrives (the macro's draw methods did that): apply them first, or the draw uses the
    // previous draw's index buffer, vertex buffers and shaders.
    ConsumeSink(system);
    const u32 estimate = static_cast<u32>(EstimateIndexBufferSize());
    regs.vertex_id_base = 0;
    regs.global_base_vertex_index = 0;
    regs.global_base_instance_index = 0;
    dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
    // The macro copies the record's base vertex and instance to the driver constants at 0x640 of
    // the selected constant buffer, which this draw skips: shaders read them from the draw
    // instead (the host GPU takes them from the record). NVN's draw macros keep them in c[0] in
    // Mario Kart World; the bank is looked up rather than assumed.
    // EDEN_NX2_INDIRECT_REPLACE=0: leave the constants at zero.
    static const bool replace_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INDIRECT_REPLACE");
        return value == nullptr || value[0] != '0';
    }();
    std::optional<u32> driver_bank;
    if (replace_enabled) {
        const GPUVAddr driver_cb = regs.const_buffer.Address();
        const auto& vertex_buffers = state.shader_stages[0].const_buffers;
        for (u32 bank = 0; bank < vertex_buffers.size(); ++bank) {
            if (vertex_buffers[bank].enabled && vertex_buffers[bank].address == driver_cb) {
                driver_bank = bank;
                break;
            }
        }
        static std::atomic<u32> logged_banks{};
        const u32 bit = driver_bank ? 1U << *driver_bank : 1U << 31;
        if ((logged_banks.fetch_or(bit, std::memory_order_relaxed) & bit) == 0) {
            if (driver_bank) {
                LOG_INFO(HW_GPU, "Indirect records: driver constants at {:#x} are vertex c[{}]",
                         driver_cb, *driver_bank);
            } else {
                LOG_WARNING(HW_GPU, "Indirect records: no vertex constant buffer binds the driver "
                                    "constants at {:#x}: base vertex and instance read zero",
                            driver_cb);
            }
        }
    }
    const bool replace = driver_bank.has_value();
    if (replace) {
        engine_state = EngineHint::OnHLEMacro;
        SetHLEReplacementAttributeType(*driver_bank, 0x640, HLEReplacementAttributeType::BaseVertex);
        SetHLEReplacementAttributeType(*driver_bank, 0x644,
                                       HLEReplacementAttributeType::BaseInstance);
    }
    auto& params = draw_manager.indirect_state;
    params.is_byte_count = false;
    params.is_indexed = true;
    params.include_count = false;
    params.count_start_address = 0;
    params.indirect_start_address = record;
    params.buffer_size = 5 * sizeof(u32);
    params.max_draw_counts = 1;
    params.stride = 0;
    params.from_record = true;
    draw_manager.DrawIndexedIndirect(*this, static_cast<Regs::PrimitiveTopology>(topology), 0,
                                     estimate);
    params.from_record = false;
    if (replace) {
        engine_state = EngineHint::None;
        replace_table.clear();
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

} // namespace Tegra::Engines
