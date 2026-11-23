// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/diag_request.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/buffer_cache/raw_coverage.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_disk.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_state.h"
#include "video_core/renderer_vulkan/nvn/nvn_viewport.h"
#include "video_core/vulkan_common/cooperative_matrix.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn {
namespace {

namespace VCN = VideoCommon::Nvn;

// NVNblendFunc values (nvn_enums.inc).
constexpr u32 BLEND_SRC1_COLOR = 0x10;
constexpr u32 BLEND_ONE_MINUS_SRC1_ALPHA = 0x13;

// NVNformat values the vertex tables use (nvn_enums.inc).
enum VertexNvnFormat : u32 {
    F_R8 = 0x01,
    F_R8SN,
    F_R8UI,
    F_R8I,
    F_R16F,
    F_R16,
    F_R16SN,
    F_R16UI,
    F_R16I,
    F_R32F,
    F_R32UI,
    F_R32I,
    F_RG8,
    F_RG8SN,
    F_RG8UI,
    F_RG8I,
    F_RG16F,
    F_RG16,
    F_RG16SN,
    F_RG16UI,
    F_RG16I,
    F_RG32F,
    F_RG32UI,
    F_RG32I,
    F_RGB8,
    F_RGB8SN,
    F_RGB8UI,
    F_RGB8I,
    F_RGB16F,
    F_RGB16,
    F_RGB16SN,
    F_RGB16UI,
    F_RGB16I,
    F_RGB32F,
    F_RGB32UI,
    F_RGB32I,
    F_RGBA8,
    F_RGBA8SN,
    F_RGBA8UI,
    F_RGBA8I,
    F_RGBA16F,
    F_RGBA16,
    F_RGBA16SN,
    F_RGBA16UI,
    F_RGBA16I,
    F_RGBA32F,
    F_RGBA32UI,
    F_RGBA32I,
    F_RGB10A2 = 0x3D,
    F_RGB10A2UI = 0x3E,
    F_R8_UI2F = 0x51,
    F_R8_I2F,
    F_R16_UI2F,
    F_R16_I2F,
    F_R32_UI2F,
    F_R32_I2F,
    F_RG8_UI2F,
    F_RG8_I2F,
    F_RG16_UI2F,
    F_RG16_I2F,
    F_RG32_UI2F,
    F_RG32_I2F,
    F_RGB8_UI2F,
    F_RGB8_I2F,
    F_RGB16_UI2F,
    F_RGB16_I2F,
    F_RGB32_UI2F,
    F_RGB32_I2F,
    F_RGBA8_UI2F,
    F_RGBA8_I2F,
    F_RGBA16_UI2F,
    F_RGBA16_I2F,
    F_RGBA32_UI2F,
    F_RGBA32_I2F,
    F_RGB10A2SN,
    F_RGB10A2I,
    F_RGB10A2_UI2F,
    F_RGB10A2_I2F,
    F_BGRA8 = 0x9A,
};

enum class Kind : u8 { None, Float, UInt, SInt, UScaled, SScaled };

struct VertexEntry {
    Kind kind{Kind::None};
    VkFormat format{
        VK_FORMAT_UNDEFINED}; ///< the format (scaled formats: the scaled one or UNDEFINED)
    VkFormat integer{VK_FORMAT_UNDEFINED}; ///< scaled formats: the raw integer equivalent
};

constexpr VertexEntry Plain(Kind kind, VkFormat format) {
    return {kind, format, VK_FORMAT_UNDEFINED};
}
constexpr VertexEntry Scaled(Kind kind, VkFormat scaled, VkFormat integer) {
    return {kind, scaled, integer};
}

constexpr VertexEntry LookupVertex(u32 f) {
    switch (f) {
    case F_R8:
        return Plain(Kind::Float, VK_FORMAT_R8_UNORM);
    case F_R8SN:
        return Plain(Kind::Float, VK_FORMAT_R8_SNORM);
    case F_R8UI:
        return Plain(Kind::UInt, VK_FORMAT_R8_UINT);
    case F_R8I:
        return Plain(Kind::SInt, VK_FORMAT_R8_SINT);
    case F_R16F:
        return Plain(Kind::Float, VK_FORMAT_R16_SFLOAT);
    case F_R16:
        return Plain(Kind::Float, VK_FORMAT_R16_UNORM);
    case F_R16SN:
        return Plain(Kind::Float, VK_FORMAT_R16_SNORM);
    case F_R16UI:
        return Plain(Kind::UInt, VK_FORMAT_R16_UINT);
    case F_R16I:
        return Plain(Kind::SInt, VK_FORMAT_R16_SINT);
    case F_R32F:
        return Plain(Kind::Float, VK_FORMAT_R32_SFLOAT);
    case F_R32UI:
        return Plain(Kind::UInt, VK_FORMAT_R32_UINT);
    case F_R32I:
        return Plain(Kind::SInt, VK_FORMAT_R32_SINT);
    case F_RG8:
        return Plain(Kind::Float, VK_FORMAT_R8G8_UNORM);
    case F_RG8SN:
        return Plain(Kind::Float, VK_FORMAT_R8G8_SNORM);
    case F_RG8UI:
        return Plain(Kind::UInt, VK_FORMAT_R8G8_UINT);
    case F_RG8I:
        return Plain(Kind::SInt, VK_FORMAT_R8G8_SINT);
    case F_RG16F:
        return Plain(Kind::Float, VK_FORMAT_R16G16_SFLOAT);
    case F_RG16:
        return Plain(Kind::Float, VK_FORMAT_R16G16_UNORM);
    case F_RG16SN:
        return Plain(Kind::Float, VK_FORMAT_R16G16_SNORM);
    case F_RG16UI:
        return Plain(Kind::UInt, VK_FORMAT_R16G16_UINT);
    case F_RG16I:
        return Plain(Kind::SInt, VK_FORMAT_R16G16_SINT);
    case F_RG32F:
        return Plain(Kind::Float, VK_FORMAT_R32G32_SFLOAT);
    case F_RG32UI:
        return Plain(Kind::UInt, VK_FORMAT_R32G32_UINT);
    case F_RG32I:
        return Plain(Kind::SInt, VK_FORMAT_R32G32_SINT);
    case F_RGB8:
        return Plain(Kind::Float, VK_FORMAT_R8G8B8_UNORM);
    case F_RGB8SN:
        return Plain(Kind::Float, VK_FORMAT_R8G8B8_SNORM);
    case F_RGB8UI:
        return Plain(Kind::UInt, VK_FORMAT_R8G8B8_UINT);
    case F_RGB8I:
        return Plain(Kind::SInt, VK_FORMAT_R8G8B8_SINT);
    case F_RGB16F:
        return Plain(Kind::Float, VK_FORMAT_R16G16B16_SFLOAT);
    case F_RGB16:
        return Plain(Kind::Float, VK_FORMAT_R16G16B16_UNORM);
    case F_RGB16SN:
        return Plain(Kind::Float, VK_FORMAT_R16G16B16_SNORM);
    case F_RGB16UI:
        return Plain(Kind::UInt, VK_FORMAT_R16G16B16_UINT);
    case F_RGB16I:
        return Plain(Kind::SInt, VK_FORMAT_R16G16B16_SINT);
    case F_RGB32F:
        return Plain(Kind::Float, VK_FORMAT_R32G32B32_SFLOAT);
    case F_RGB32UI:
        return Plain(Kind::UInt, VK_FORMAT_R32G32B32_UINT);
    case F_RGB32I:
        return Plain(Kind::SInt, VK_FORMAT_R32G32B32_SINT);
    case F_RGBA8:
        return Plain(Kind::Float, VK_FORMAT_R8G8B8A8_UNORM);
    case F_RGBA8SN:
        return Plain(Kind::Float, VK_FORMAT_R8G8B8A8_SNORM);
    case F_RGBA8UI:
        return Plain(Kind::UInt, VK_FORMAT_R8G8B8A8_UINT);
    case F_RGBA8I:
        return Plain(Kind::SInt, VK_FORMAT_R8G8B8A8_SINT);
    case F_RGBA16F:
        return Plain(Kind::Float, VK_FORMAT_R16G16B16A16_SFLOAT);
    case F_RGBA16:
        return Plain(Kind::Float, VK_FORMAT_R16G16B16A16_UNORM);
    case F_RGBA16SN:
        return Plain(Kind::Float, VK_FORMAT_R16G16B16A16_SNORM);
    case F_RGBA16UI:
        return Plain(Kind::UInt, VK_FORMAT_R16G16B16A16_UINT);
    case F_RGBA16I:
        return Plain(Kind::SInt, VK_FORMAT_R16G16B16A16_SINT);
    case F_RGBA32F:
        return Plain(Kind::Float, VK_FORMAT_R32G32B32A32_SFLOAT);
    case F_RGBA32UI:
        return Plain(Kind::UInt, VK_FORMAT_R32G32B32A32_UINT);
    case F_RGBA32I:
        return Plain(Kind::SInt, VK_FORMAT_R32G32B32A32_SINT);
    case F_RGB10A2:
        return Plain(Kind::Float, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    case F_RGB10A2UI:
        return Plain(Kind::UInt, VK_FORMAT_A2B10G10R10_UINT_PACK32);
    case F_RGB10A2SN:
        return Plain(Kind::Float, VK_FORMAT_A2B10G10R10_SNORM_PACK32);
    case F_RGB10A2I:
        return Plain(Kind::SInt, VK_FORMAT_A2B10G10R10_SINT_PACK32);
    case F_BGRA8:
        return Plain(Kind::Float, VK_FORMAT_B8G8R8A8_UNORM);
    case F_R8_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R8_USCALED, VK_FORMAT_R8_UINT);
    case F_R8_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R8_SSCALED, VK_FORMAT_R8_SINT);
    case F_R16_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R16_USCALED, VK_FORMAT_R16_UINT);
    case F_R16_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R16_SSCALED, VK_FORMAT_R16_SINT);
    case F_R32_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32_UINT);
    case F_R32_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32_SINT);
    case F_RG8_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_UINT);
    case F_RG8_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_R8G8_SINT);
    case F_RG16_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16_UINT);
    case F_RG16_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R16G16_SINT);
    case F_RG32_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32_UINT);
    case F_RG32_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32_SINT);
    case F_RGB8_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_UINT);
    case F_RGB8_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R8G8B8_SINT);
    case F_RGB16_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_UINT);
    case F_RGB16_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R16G16B16_SINT);
    case F_RGB32_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32B32_UINT);
    case F_RGB32_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32B32_SINT);
    case F_RGBA8_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_UINT);
    case F_RGBA8_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_R8G8B8A8_SINT);
    case F_RGBA16_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_R16G16B16A16_USCALED, VK_FORMAT_R16G16B16A16_UINT);
    case F_RGBA16_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_R16G16B16A16_SSCALED, VK_FORMAT_R16G16B16A16_SINT);
    case F_RGBA32_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32B32A32_UINT);
    case F_RGBA32_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_UNDEFINED, VK_FORMAT_R32G32B32A32_SINT);
    case F_RGB10A2_UI2F:
        return Scaled(Kind::UScaled, VK_FORMAT_A2B10G10R10_USCALED_PACK32,
                      VK_FORMAT_A2B10G10R10_UINT_PACK32);
    case F_RGB10A2_I2F:
        return Scaled(Kind::SScaled, VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
                      VK_FORMAT_A2B10G10R10_SINT_PACK32);
    default:
        return {};
    }
}

VkBlendFactor ToBlendFactor(u32 func, bool& known) {
    known = true;
    switch (func) {
    case 0x01:
        return VK_BLEND_FACTOR_ZERO;
    case 0x02:
        return VK_BLEND_FACTOR_ONE;
    case 0x03:
        return VK_BLEND_FACTOR_SRC_COLOR;
    case 0x04:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x05:
        return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x06:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x07:
        return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x08:
        return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x09:
        return VK_BLEND_FACTOR_DST_COLOR;
    case 0x0A:
        return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x0B:
        return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10:
        return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12:
        return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13:
        return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61:
        return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62:
        return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63:
        return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64:
        return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default:
        known = false;
        return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp ToBlendOp(u32 equation, bool& known) {
    known = true;
    switch (equation) {
    case 1:
        return VK_BLEND_OP_ADD;
    case 2:
        return VK_BLEND_OP_SUBTRACT;
    case 3:
        return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4:
        return VK_BLEND_OP_MIN;
    case 5:
        return VK_BLEND_OP_MAX;
    default:
        known = false;
        return VK_BLEND_OP_ADD;
    }
}

/// NVN depth/stencil/alpha funcs: NEVER=1 .. ALWAYS=8, in VkCompareOp order.
VkCompareOp ToCompareOp(u32 func) {
    if (func < 1 || func > 8) {
        return VK_COMPARE_OP_ALWAYS;
    }
    return static_cast<VkCompareOp>(func - 1);
}

/// NVNstencilOp KEEP=1 .. DECR_WRAP=8, in VkStencilOp order.
VkStencilOp ToStencilOp(u32 op) {
    if (op < 1 || op > 8) {
        return VK_STENCIL_OP_KEEP;
    }
    return static_cast<VkStencilOp>(op - 1);
}

VkPrimitiveTopology ToVkTopology(u32 topology) {
    switch (topology) {
    case 0:
        return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 1:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 2:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; // LINE_LOOP: the replay closes the loop
    case 3:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 4:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 5:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 6:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case 7:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // QUADS: converted by the replay
    case 8:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // QUAD_STRIP: converted by the replay
    case 9:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; // POLYGON
    case 10:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case 11:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case 12:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case 13:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case 14:
        return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    default:
        return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    }
}

bool IsStripOrFan(VkPrimitiveTopology topology) {
    return topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
           topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
           topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
           topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY ||
           topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
}

VkSampleCountFlagBits ToSampleCount(u32 samples) {
    switch (samples) {
    case 2:
        return VK_SAMPLE_COUNT_2_BIT;
    case 4:
        return VK_SAMPLE_COUNT_4_BIT;
    case 8:
        return VK_SAMPLE_COUNT_8_BIT;
    case 16:
        return VK_SAMPLE_COUNT_16_BIT;
    default:
        return VK_SAMPLE_COUNT_1_BIT;
    }
}

bool IsIntegerColorFormat(VkFormat format, bool& is_signed) {
    is_signed = false;
    switch (format) {
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_B8G8R8A8_UINT:
    case VK_FORMAT_A8B8G8R8_UINT_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT:
        return true;
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32:
    case VK_FORMAT_A2R10G10B10_SINT_PACK32:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT:
        is_signed = true;
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

bool FormatHasDepth(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

bool FormatHasStencil(VkFormat format) {
    switch (format) {
    case VK_FORMAT_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

Shader::AttributeType ColorOutputType(VkFormat format) {
    bool is_signed{};
    if (!IsIntegerColorFormat(format, is_signed)) {
        return Shader::AttributeType::Float;
    }
    return is_signed ? Shader::AttributeType::SignedInt : Shader::AttributeType::UnsignedInt;
}

Shader::InputTopology ToInputTopology(u32 nvn_topology) {
    switch (nvn_topology) {
    case 0:
        return Shader::InputTopology::Points;
    case 1:
    case 2:
    case 3:
        return Shader::InputTopology::Lines;
    case 10:
    case 11:
        return Shader::InputTopology::LinesAdjacency;
    case 12:
    case 13:
        return Shader::InputTopology::TrianglesAdjacency;
    default: // triangles, strips, fans, quads, polygon, patches
        return Shader::InputTopology::Triangles;
    }
}

std::optional<Shader::CompareFunction> ToAlphaTestFunction(u32 nvn_alpha_func) {
    switch (nvn_alpha_func) {
    case 1:
        return Shader::CompareFunction::Never;
    case 2:
        return Shader::CompareFunction::Less;
    case 3:
        return Shader::CompareFunction::Equal;
    case 4:
        return Shader::CompareFunction::LessThanEqual;
    case 5:
        return Shader::CompareFunction::Greater;
    case 6:
        return Shader::CompareFunction::NotEqual;
    case 7:
        return Shader::CompareFunction::GreaterThanEqual;
    default:
        return std::nullopt; // ALWAYS (8) and invalid values: no alpha test
    }
}

bool UsesDualSourceBlend(const GraphicsFixedState& state) {
    if (!VCN::ColorState::BlendEnabled(state.color, 0)) {
        return false;
    }
    const u64 blend = state.blend[0];
    const auto is_src1 = [](u32 func) {
        return func >= BLEND_SRC1_COLOR && func <= BLEND_ONE_MINUS_SRC1_ALPHA;
    };
    return is_src1(VCN::BlendState::SrcRgbFunc(blend)) ||
           is_src1(VCN::BlendState::DstRgbFunc(blend)) ||
           is_src1(VCN::BlendState::SrcAlphaFunc(blend)) ||
           is_src1(VCN::BlendState::DstAlphaFunc(blend));
}

TranslationState MakeTranslationState(const GraphicsFixedState& key_state,
                                      const RenderPassKey& targets, VertexInputMode mode,
                                      bool must_emulate_scaled_formats) {
    TranslationState out{};
    out.input_topology = static_cast<u32>(ToInputTopology(key_state.topology));
    out.point_size_bits = key_state.point_size_bits;
    const auto alpha = ToAlphaTestFunction(VCN::ColorState::AlphaFunc(key_state.color));
    out.alpha_func = alpha ? static_cast<u32>(*alpha) + 1 : 0;
    out.alpha_ref_bits = key_state.alpha_ref_bits;
    out.convert_depth_mode = key_state.depth_mode == 0 ? 1 : 0;
    out.dual_source_blend = UsesDualSourceBlend(key_state) ? 1 : 0;
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        out.color_output_types[slot] =
            static_cast<u8>(ColorOutputType(static_cast<VkFormat>(targets.color[slot])));
    }
    for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
        out.vertex_types[location] = static_cast<u8>(
            VertexAttributeType(key_state, mode, must_emulate_scaled_formats, location));
    }
    return out;
}

DynamicFixedState MakeDynamicFixedState(const GraphicsFixedState& state) {
    DynamicFixedState out;
    u32 offset_bit = 4; // NVNpolygonOffsetEnable::FILL
    switch (VCN::PolygonState::PolygonMode(state.polygon)) {
    case 0:
        offset_bit = 1; // POINT
        break;
    case 1:
        offset_bit = 2; // LINE
        break;
    default:
        break;
    }
    // The replay's viewports are negative-height whatever the window origin (nvn_viewport.h), so
    // the NVN winding is the Vulkan one.
    const bool nvn_ccw = VCN::PolygonState::FrontFace(state.polygon) != 0;
    out.front_face = VulkanFrontFaceCcw(nvn_ccw, Debug::Get().face == 1)
                         ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                         : VK_FRONT_FACE_CLOCKWISE;
    switch (VCN::PolygonState::CullFace(state.polygon)) {
    case 1:
        out.cull_mode = VK_CULL_MODE_FRONT_BIT;
        break;
    case 2:
        out.cull_mode = VK_CULL_MODE_BACK_BIT;
        break;
    case 3:
        out.cull_mode = VK_CULL_MODE_FRONT_AND_BACK;
        break;
    default:
        out.cull_mode = VK_CULL_MODE_NONE;
        break;
    }
    if (Debug::Get().face == 2) {
        out.cull_mode = VK_CULL_MODE_NONE;
    }
    out.depth_bias_enable = (VCN::PolygonState::OffsetEnables(state.polygon) & offset_bit) != 0;

    const u64 ds = state.depth_stencil;
    using DS = VCN::DepthStencilState;
    // EDEN_NX2_NVN_DEPTH_PROBE=1|2 (experiment): an EQUAL draw tests LESS_OR_EQUAL (1) or
    // GREATER_OR_EQUAL (2) and writes depth, so a dump of the depth target after the pass shows how
    // far the draw's depth is from what the pre-pass stored.
    const u32 probe = Debug::Get().depth_probe;
    const bool probing = probe != 0 && DS::DepthTestEnable(ds) != 0 && DS::DepthFunc(ds) == 3;
    out.depth_test_enable = DS::DepthTestEnable(ds) != 0 && !Debug::Get().no_depth;
    out.depth_write_enable = probing || DS::DepthWriteEnable(ds) != 0;
    out.depth_compare_op =
        probing ? (probe == 1   ? VK_COMPARE_OP_LESS_OR_EQUAL
                   : probe == 2 ? VK_COMPARE_OP_GREATER_OR_EQUAL
                                : VK_COMPARE_OP_ALWAYS)
        : DS::DepthTestEnable(ds) != 0
            ? ToCompareOp(Debug::Get().equal_lequal && DS::DepthFunc(ds) == 3 ? 4
                                                                              : DS::DepthFunc(ds))
            : VK_COMPARE_OP_ALWAYS;
    out.stencil_test_enable = DS::StencilTestEnable(ds) != 0;
    out.front = {
        .fail = ToStencilOp(DS::FrontFailOp(ds)),
        .pass = ToStencilOp(DS::FrontDepthPassOp(ds)),
        .depth_fail = ToStencilOp(DS::FrontDepthFailOp(ds)),
        .compare = ToCompareOp(DS::FrontStencilFunc(ds)),
    };
    out.back = {
        .fail = ToStencilOp(DS::BackFailOp(ds)),
        .pass = ToStencilOp(DS::BackDepthPassOp(ds)),
        .depth_fail = ToStencilOp(DS::BackDepthFailOp(ds)),
        .compare = ToCompareOp(DS::BackStencilFunc(ds)),
    };
    return out;
}

VertexFormatInfo ResolveVertexFormat(const Device& device, u32 nvn_format) {
    return ResolveVertexFormat(device.MustEmulateScaledFormats(), nvn_format);
}

VertexFormatInfo ResolveVertexFormat(bool must_emulate_scaled_formats, u32 nvn_format) {
    VertexFormatInfo info;
    if (nvn_format == 0) {
        return info;
    }
    const VertexEntry entry = LookupVertex(nvn_format);
    switch (entry.kind) {
    case Kind::None:
        return info;
    case Kind::Float:
        info.type = Shader::AttributeType::Float;
        info.format = entry.format;
        return info;
    case Kind::UInt:
        info.type = Shader::AttributeType::UnsignedInt;
        info.format = entry.format;
        return info;
    case Kind::SInt:
        info.type = Shader::AttributeType::SignedInt;
        info.format = entry.format;
        return info;
    case Kind::UScaled:
    case Kind::SScaled:
        break;
    }
    const bool unsigned_kind = entry.kind == Kind::UScaled;
    const bool native = entry.format != VK_FORMAT_UNDEFINED && !must_emulate_scaled_formats;
    if (native) {
        info.type = unsigned_kind ? Shader::AttributeType::UnsignedScaled
                                  : Shader::AttributeType::SignedScaled;
        info.format = entry.format;
        return info;
    }
    info.format = entry.integer;
    if (!must_emulate_scaled_formats) {
        // The shader expects a float input but the device cannot supply a scaled format.
        info.type =
            unsigned_kind ? Shader::AttributeType::UnsignedInt : Shader::AttributeType::SignedInt;
        info.scaled_without_support = true;
    } else {
        // The recompiler converts the raw integer itself.
        info.type = unsigned_kind ? Shader::AttributeType::UnsignedScaled
                                  : Shader::AttributeType::SignedScaled;
    }
    return info;
}

void MakeKeyState(GraphicsFixedState& state, VertexInputMode mode,
                  bool must_emulate_scaled_formats) {
    state.vertex_types.fill(0);
    if (mode == VertexInputMode::Dynamic) {
        for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
            const u32 format = VCN::VertexAttribState::Format(state.attributes[location]);
            state.vertex_types[location] =
                static_cast<u8>(ResolveVertexFormat(must_emulate_scaled_formats, format).type);
        }
    }
    Disk::CanonicalizeState(state, mode);
}

Shader::AttributeType VertexAttributeType(const GraphicsFixedState& key_state, VertexInputMode mode,
                                          bool must_emulate_scaled_formats, u32 location) {
    if (mode == VertexInputMode::Dynamic) {
        const u8 type = key_state.vertex_types[location];
        return type <= static_cast<u8>(Shader::AttributeType::Disabled)
                   ? static_cast<Shader::AttributeType>(type)
                   : Shader::AttributeType::Disabled;
    }
    const u32 format = VCN::VertexAttribState::Format(key_state.attributes[location]);
    return ResolveVertexFormat(must_emulate_scaled_formats, format).type;
}

void BuildVertexInput(const Device& device, const GraphicsFixedState& layout, u32 attributes_read,
                      VertexInput& out, std::vector<std::string>& issues) {
    out.binding_count = 0;
    out.attribute_count = 0;
    out.used_streams = 0;
    const u32 max_bindings = std::min<u32>(MAX_VERTEX_STREAMS, device.GetMaxVertexInputBindings());
    const u32 max_attributes =
        std::min<u32>(MAX_VERTEX_ATTRIBUTES, device.GetMaxVertexInputAttributes());
    for (u32 location = 0; location < MAX_VERTEX_ATTRIBUTES; ++location) {
        if ((attributes_read & (1u << location)) == 0) {
            continue;
        }
        const u32 word = layout.attributes[location];
        const u32 nvn_format = VCN::VertexAttribState::Format(word);
        if (nvn_format == 0) {
            continue; // unused: the shader reads an undefined value, as on hardware
        }
        const VertexFormatInfo format = ResolveVertexFormat(device, nvn_format);
        if (format.format == VK_FORMAT_UNDEFINED) {
            issues.push_back(fmt::format(
                "vertex attribute: NVNformat 0x{:x} is not a vertex format", nvn_format));
            continue;
        }
        if (format.scaled_without_support) {
            issues.push_back(fmt::format("vertex attribute: scaled NVNformat 0x{:x} has no scaled "
                                         "Vulkan format (raw integer)",
                                         nvn_format));
        }
        const u32 stream = VCN::VertexAttribState::StreamIndex(word);
        if (stream >= max_bindings || location >= max_attributes ||
            out.attribute_count >= max_attributes) {
            issues.push_back(fmt::format("vertex attribute: stream {}, location {} or attribute "
                                         "count beyond the device limit",
                                         stream, location));
            continue;
        }
        out.attributes[out.attribute_count++] = {
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = location,
            .binding = stream,
            .format = format.format,
            .offset = VCN::VertexAttribState::Offset(word),
        };
        out.used_streams |= 1u << stream;
    }
    for (u32 stream = 0; stream < MAX_VERTEX_STREAMS; ++stream) {
        if ((out.used_streams & (1u << stream)) == 0) {
            continue;
        }
        // NVN: divisor 0 = per vertex, otherwise per instance advancing every `divisor`
        // instances (the extension's binding divisor is 1 for vertex rate).
        const u32 divisor = layout.stream_divisor[stream];
        out.bindings[out.binding_count++] = {
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = stream,
            .stride = layout.stream_stride[stream],
            .inputRate = divisor != 0 ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = divisor != 0 ? divisor : 1,
        };
    }
}

/// Raw (untracked) global memory goes through buffer device addresses: the SPIR-V needs int64
/// and the 64-bit and 32-bit views of the driver bank (descriptor aliasing).
static bool RawGlobalMemorySupported(const Device& device) {
    static const bool disabled = std::getenv("EDEN_NX2_NVN_NO_RAW_GLOBAL") != nullptr;
    const bool supported = !disabled && device.IsBufferDeviceAddressSupported() &&
                           device.IsShaderInt64Supported() &&
                           device.IsDescriptorAliasingSupported();
    static const bool logged = [&] {
        LOG_INFO(Render_Vulkan,
                 "[nvn] raw global memory: {} (buffer device address {}, int64 {}, descriptor "
                 "aliasing {}{})",
                 supported ? "supported" : "unsupported", device.IsBufferDeviceAddressSupported(),
                 device.IsShaderInt64Supported(), device.IsDescriptorAliasingSupported(),
                 disabled ? ", disabled by EDEN_NX2_NVN_NO_RAW_GLOBAL" : "");
        return true;
    }();
    static_cast<void>(logged);
    return supported;
}

bool CooperativeMatrixLoweringEnabled(bool device_cooperative_matrix, const char* tensor_mode) {
    return device_cooperative_matrix && TensorModeAllowsCooperativeMatrix(tensor_mode);
}

Shader::Profile MakeProfile(const Device& device) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    return Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset =
            driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
            driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE || driver_id == VK_DRIVER_ID_MESA_RADV ||
            driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
            driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY || driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        // Fragment shuffles are expanded into broadcasts on devices that run subgroup operations in
        // the fragment stage and have the ballot feature but not the shuffle one.
        .has_broken_spirv_fragment_shuffle =
            (subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0 &&
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_BALLOT_BIT) &&
            !device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances(),
        .support_raw_global_memory = RawGlobalMemorySupported(device)};
}

Shader::HostTranslateInfo MakeHostInfo(const Device& device, const Shader::Profile& profile) {
    const VkDriverId driver_id{device.GetDriverID()};
    Shader::HostTranslateInfo host_info{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic =
            device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic =
            device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .nvn_bindless_table = device.IsBindlessTableSupported(),
        .nvn_raw_global_memory = profile.support_raw_global_memory,
        .raw_coverage = profile.support_raw_global_memory && VideoCommon::RawCoverageEnabled(),
        .raw_global_last_hit =
            profile.support_raw_global_memory && VideoCommon::RawGlobalFastEnabled(),
        .raw_global_store_alias =
            profile.support_raw_global_memory &&
            (Common::DiagRequest::TrySwitchEnabled("TRY_RO_MIRROR_EXACT") ||
             Common::DiagRequest::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH")),
        .compute_storage64 = std::getenv("EDEN_NX2_TRY_COMPUTE_STORAGE64") != nullptr &&
                             std::getenv("EDEN_NX2_TRY_COMPUTE_STORAGE64")[0] == '1',
        // The recompiler's rescaling pass runs only while the Resolution setting scales
        // (nvn_rescale.h): the replay pushes the words and the down factor it reads.
        .skip_rescaling_pass = !Settings::values.resolution_info.active,
    };
    host_info.ApplyDescriptorLimitPolicy();
    return host_info;
}

void DescribeGraphicsPipeline(const Device& device, const GraphicsPipelineParams& params,
                              GraphicsPipelineDescription& out, std::vector<std::string>& issues) {
    const GraphicsFixedState& state = *params.state;
    const RenderPassKey& targets = *params.targets;
    const u8* const ms = state.multisample.data();
    const DynamicFixedState fixed = MakeDynamicFixedState(state);

    // --- vertex input --------------------------------------------------------------------
    // Dynamic: no bindings or attributes here, the replay sets the layout per draw with
    // vkCmdSetVertexInputEXT (the key holds no layout). Static: the layout of the key. A task/mesh
    // pipeline has neither vertex input nor input assembly (VK_EXT_mesh_shader).
    const bool dynamic_vertex_input =
        !params.mesh && params.vertex_input == VertexInputMode::Dynamic;
    if (!dynamic_vertex_input && !params.mesh) {
        VertexInput layout;
        BuildVertexInput(device, state, params.attributes_read, layout, issues);
        for (u32 i = 0; i < layout.attribute_count; ++i) {
            const VkVertexInputAttributeDescription2EXT& attribute = layout.attributes[i];
            out.vertex_attributes.push_back({
                .location = attribute.location,
                .binding = attribute.binding,
                .format = attribute.format,
                .offset = attribute.offset,
            });
        }
        for (u32 i = 0; i < layout.binding_count; ++i) {
            const VkVertexInputBindingDescription2EXT& binding = layout.bindings[i];
            out.vertex_bindings.push_back({
                .binding = binding.binding,
                .stride = binding.stride,
                .inputRate = binding.inputRate,
            });
            if (binding.divisor > 1) {
                out.divisors.push_back({.binding = binding.binding, .divisor = binding.divisor});
            }
        }
    }
    out.vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(out.vertex_bindings.size()),
        .pVertexBindingDescriptions = out.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(out.vertex_attributes.size()),
        .pVertexAttributeDescriptions = out.vertex_attributes.data(),
    };
    // VK_EXT_vertex_attribute_divisor is one of the device's mandatory extensions.
    out.divisor = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .vertexBindingDivisorCount = static_cast<u32>(out.divisors.size()),
        .pVertexBindingDivisors = out.divisors.data(),
    };
    if (!out.divisors.empty()) {
        out.vertex_input.pNext = &out.divisor;
    }

    // --- input assembly / tessellation ---------------------------------------------------
    VkPrimitiveTopology topology = ToVkTopology(state.topology);
    if (topology == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM) {
        issues.push_back(fmt::format("topology: unknown NVNdrawPrimitive {}", state.topology));
        topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    if (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST && !params.has_tessellation) {
        issues.push_back("topology: PATCHES without tessellation stages (drawn as points)");
        topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    } else if (topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST && params.has_tessellation) {
        issues.push_back("topology: tessellation stages without PATCHES (drawn as patches)");
        topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    }
    const bool restart_supported =
        topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
            ? device.IsPatchListPrimitiveRestartSupported()
            : (IsStripOrFan(topology) || device.IsTopologyListPrimitiveRestartSupported());
    out.input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = topology,
        .primitiveRestartEnable =
            (state.primitive_restart != 0 && restart_supported) || device.IsMoltenVK() ? VK_TRUE
                                                                                       : VK_FALSE,
    };
    if (state.primitive_restart != 0 && !restart_supported) {
        issues.push_back("primitive restart: not supported for this topology on the device");
    }
    u32 patch_points = state.patch_vertices;
    if (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST && patch_points == 0) {
        issues.push_back("tessellation: patch size 0 (1 used)");
        patch_points = 1;
    }
    out.tessellation = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .patchControlPoints = patch_points,
    };

    // --- viewport -------------------------------------------------------------------------
    out.depth_clip_control = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = state.depth_mode == 0 ? VK_TRUE : VK_FALSE,
    };
    out.viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = params.viewport_count,
        .pViewports = nullptr,
        .scissorCount = params.viewport_count,
        .pScissors = nullptr,
    };
    if (device.IsExtDepthClipControlSupported()) {
        out.viewport.pNext = &out.depth_clip_control;
    }

    // --- rasterization --------------------------------------------------------------------
    VkPolygonMode vk_polygon_mode = VK_POLYGON_MODE_FILL;
    switch (VCN::PolygonState::PolygonMode(state.polygon)) {
    case 0:
        vk_polygon_mode = VK_POLYGON_MODE_POINT;
        break;
    case 1:
        vk_polygon_mode = VK_POLYGON_MODE_LINE;
        break;
    default:
        break;
    }
    // With dynamic fixed state the cull mode, front face and depth bias enable are set per draw
    // (the create-info values are ignored): fixed values keep the library keys equal.
    const bool dynamic_fixed = params.dynamic_fixed;
    out.rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = state.depth_clamp != 0 ? VK_TRUE : VK_FALSE,
        .rasterizerDiscardEnable = state.rasterizer_discard != 0 ? VK_TRUE : VK_FALSE,
        .polygonMode = vk_polygon_mode,
        .cullMode = dynamic_fixed ? VkCullModeFlags{VK_CULL_MODE_NONE} : fixed.cull_mode,
        .frontFace = dynamic_fixed ? VK_FRONT_FACE_COUNTER_CLOCKWISE : fixed.front_face,
        .depthBiasEnable = !dynamic_fixed && fixed.depth_bias_enable ? VK_TRUE : VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };
    // NVN draws flat-interpolated outputs from the last vertex unless the game asks for the first
    // (GraphicsFixedState::provoking_first); Vulkan's default is the first.
    const bool first_vertex = state.provoking_first != 0;
    out.provoking_vertex = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .provokingVertexMode = first_vertex ? VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
                                            : VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT,
    };
    if (!first_vertex && device.SupportsProvokingVertexLastMode()) {
        out.rasterization.pNext = &out.provoking_vertex;
    } else if (!first_vertex) {
        issues.push_back(
            "provoking vertex: the device has no last-vertex mode "
            "(VK_EXT_provoking_vertex provokingVertexLast); flat outputs come from the "
            "first vertex");
    }

    // --- multisample ----------------------------------------------------------------------
    const VkSampleCountFlagBits samples = ToSampleCount(targets.samples);
    // Dither only shapes the alpha-to-coverage mask (on by default in NVN; Vulkan leaves the mask
    // pattern to the implementation).
    if (VCN::MultisampleState::CoverageModulationMode(ms) != 0 ||
        VCN::MultisampleState::CoverageToColorEnable(ms) ||
        VCN::MultisampleState::SampleLocationsEnable(ms) ||
        (VCN::MultisampleState::AlphaToCoverage(ms) &&
         !VCN::MultisampleState::AlphaToCoverageDither(ms)) ||
        VCN::MultisampleState::CoverageTruncationEnable(ms)) {
        issues.push_back(
            "multisample: coverage modulation / coverage-to-color / sample locations / "
            "undithered alpha-to-coverage / coverage truncation ignored");
    }
    out.sample_mask = state.sample_mask;
    const bool writes_color0 = (params.color_outputs_written & 1u) != 0;
    out.multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = samples,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = samples <= VK_SAMPLE_COUNT_32_BIT ? &out.sample_mask : nullptr,
        .alphaToCoverageEnable =
            writes_color0 && VCN::MultisampleState::AlphaToCoverage(ms) ? VK_TRUE : VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    // --- depth / stencil ------------------------------------------------------------------
    // With dynamic fixed state the tests, writes, compare and stencil operations are set per
    // draw: the create info holds the defaults.
    const DynamicFixedState depth = dynamic_fixed ? DynamicFixedState{} : fixed;
    const auto stencil_face = [](const DynamicFixedState::StencilFace& face) {
        return VkStencilOpState{
            .failOp = face.fail,
            .passOp = face.pass,
            .depthFailOp = face.depth_fail,
            .compareOp = face.compare,
            .compareMask = 0,
            .writeMask = 0,
            .reference = 0,
        };
    };
    out.depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = depth.depth_test_enable ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = depth.depth_write_enable ? VK_TRUE : VK_FALSE,
        .depthCompareOp = depth.depth_compare_op,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = depth.stencil_test_enable ? VK_TRUE : VK_FALSE,
        .front = stencil_face(depth.front),
        .back = stencil_face(depth.back),
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    // --- color blend ----------------------------------------------------------------------
    u32 attachment_count = 0;
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        if (targets.color[slot] != VK_FORMAT_UNDEFINED) {
            attachment_count = slot + 1;
        }
    }
    const bool logic_op_enabled = VCN::ColorState::LogicOp(state.color) != VCN::STATE_LOGIC_OP_COPY;
    bool unknown_blend = false;
    bool advanced_blend = false;
    bool blend_on_integer = false;
    for (u32 slot = 0; slot < attachment_count; ++slot) {
        VkPipelineColorBlendAttachmentState attachment{};
        if (targets.color[slot] == VK_FORMAT_UNDEFINED) {
            out.blend_attachments.push_back(attachment); // unused slot: write mask 0
            continue;
        }
        attachment.colorWriteMask = VCN::ChannelMaskState::Channels(state.channel_mask, slot);
        // NVN's r/g/b/a nibble bits are VkColorComponentFlagBits' R/G/B/A bits.
        const u64 blend = state.blend[slot];
        bool is_signed{};
        const bool integer_target =
            IsIntegerColorFormat(static_cast<VkFormat>(targets.color[slot]), is_signed);
        const bool enabled = VCN::ColorState::BlendEnabled(state.color, static_cast<s32>(slot));
        if (enabled && integer_target) {
            blend_on_integer = true;
        }
        if (enabled && !integer_target) {
            bool known_all = true;
            bool known{};
            attachment.blendEnable = VK_TRUE;
            attachment.srcColorBlendFactor =
                ToBlendFactor(VCN::BlendState::SrcRgbFunc(blend), known);
            known_all &= known;
            attachment.dstColorBlendFactor =
                ToBlendFactor(VCN::BlendState::DstRgbFunc(blend), known);
            known_all &= known;
            attachment.srcAlphaBlendFactor =
                ToBlendFactor(VCN::BlendState::SrcAlphaFunc(blend), known);
            known_all &= known;
            attachment.dstAlphaBlendFactor =
                ToBlendFactor(VCN::BlendState::DstAlphaFunc(blend), known);
            known_all &= known;
            attachment.colorBlendOp = ToBlendOp(VCN::BlendState::RgbEquation(blend), known);
            known_all &= known;
            attachment.alphaBlendOp = ToBlendOp(VCN::BlendState::AlphaEquation(blend), known);
            known_all &= known;
            unknown_blend |= !known_all;
            advanced_blend |= VCN::BlendState::AdvancedMode(blend) != 0;
        }
        out.blend_attachments.push_back(attachment);
    }
    if (unknown_blend) {
        issues.push_back("blend: unknown NVN blend function or equation value (ONE/ADD used)");
    }
    if (advanced_blend) {
        issues.push_back("blend: advanced blend modes are not supported (basic funcs used)");
    }
    if (blend_on_integer) {
        issues.push_back("blend: enabled on an integer color target (blending disabled)");
    }
    out.color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = logic_op_enabled ? VK_TRUE : VK_FALSE,
        // NVNlogicOp and VkLogicOp share their numbering (CLEAR=0 ... SET=15).
        .logicOp = static_cast<VkLogicOp>(VCN::ColorState::LogicOp(state.color)),
        .attachmentCount = static_cast<u32>(out.blend_attachments.size()),
        .pAttachments = out.blend_attachments.data(),
        .blendConstants = {},
    };

    out.dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH,         VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    };
    // Strides and divisors come from the vertex input binding descriptions of
    // vkCmdSetVertexInputEXT; VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE is not used.
    if (dynamic_vertex_input) {
        out.dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    }
    if (dynamic_fixed) {
        for (const VkDynamicState dynamic : {
                 VK_DYNAMIC_STATE_CULL_MODE,
                 VK_DYNAMIC_STATE_FRONT_FACE,
                 VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                 VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                 VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                 VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                 VK_DYNAMIC_STATE_STENCIL_OP,
                 VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
             }) {
            out.dynamic_states.push_back(dynamic);
        }
    }
    out.dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(out.dynamic_states.size()),
        .pDynamicStates = out.dynamic_states.data(),
    };

    out.ci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<u32>(params.stages.size()),
        .pStages = params.stages.data(),
        .pVertexInputState = params.mesh ? nullptr : &out.vertex_input,
        .pInputAssemblyState = params.mesh ? nullptr : &out.input_assembly,
        .pTessellationState = params.has_tessellation ? &out.tessellation : nullptr,
        .pViewportState = &out.viewport,
        .pRasterizationState = &out.rasterization,
        .pMultisampleState = &out.multisample,
        .pDepthStencilState = &out.depth_stencil,
        .pColorBlendState = &out.color_blend,
        .pDynamicState = &out.dynamic_state,
        .layout = params.layout,
        .renderPass = params.render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    };
}

vk::Pipeline CreateGraphicsPipeline(const Device& device, VkPipelineCache cache,
                                    const GraphicsPipelineParams& params,
                                    std::vector<std::string>& issues) {
    GraphicsPipelineDescription description;
    DescribeGraphicsPipeline(device, params, description, issues);
    return device.GetLogical().CreateGraphicsPipeline(description.ci, cache);
}

TranslationSwitches CurrentTranslationSwitches() {
    // Each reader below is the code that consumes the switch while translating; keep the
    // interpretation identical (shader_recompiler/backend/spirv,
    // frontend/ampere/lower_to_maxwell.h). The settings are read live there, the EDEN_NX2_* ones
    // once per process: both are the process-wide value when a cache opens. Settings that only
    // change what is presented (HDR, upscaler, anti-aliasing, scaling filter, frame generation) or
    // how big the output is (resolution: the HLE never rescales) are not read by the translation
    // and are no input.
    TranslationSwitches switches;
    switches.loop_safety_checks_disabled =
        Settings::values.disable_shader_loop_safety_checks.GetValue();
    // Only while scaling: the rescaled SPIR-V bakes the factors in; at 1X the translation skips the
    // pass.
    const auto& scaling = Settings::values.resolution_info;
    switches.resolution_scaling_active = scaling.active;
    if (scaling.active) {
        switches.resolution_up_scale = scaling.up_scale;
        switches.resolution_down_shift = scaling.down_shift;
        switches.resolution_up_factor_bits = std::bit_cast<u32>(scaling.up_factor);
        switches.resolution_down_factor_bits = std::bit_cast<u32>(scaling.down_factor);
    }
    if (const char* value = std::getenv("EDEN_NX2_FP32_DENORM_PRESERVE")) {
        switches.fp32_denorm_preserve = value[0] == '1';
    }
    if (const char* value = std::getenv("EDEN_NAN_OUTPUTS")) {
        const char mode = value[0];
        switches.nan_outputs = (mode == 'm' || mode == 'z' || mode == 'b') ? mode : '\0';
    }
    if (const char* value = std::getenv("EDEN_NX2_UNDEF_ZERO")) {
        switches.undef_is_zero = value[0] != '0';
    }
    if (const char* value = std::getenv("EDEN_NX2_TENSOR")) {
        switches.tensor_mode_set = true;
        switches.tensor_mode = value;
    }
    if (const char* value = std::getenv("EDEN_NX2_LEAN_SSBO")) {
        switches.lean_global_memory = value[0] != '0';
    }
    if (const char* value = std::getenv("EDEN_NX2_STRONG_GLOBAL_FENCE")) {
        switches.strong_global_fence = value[0] == '1';
    }
    switches.raw_coverage = VideoCommon::RawCoverageEnabled();
    switches.raw_global_fast = VideoCommon::RawGlobalFastEnabled();
    switches.direct_unused_atomics =
        Common::DiagRequest::TrySwitchEnabled("TRY_DIRECT_UNUSED_ATOMICS");
    switches.native_f32_atomics = Common::DiagRequest::TrySwitchEnabled("TRY_NATIVE_F32_ATOMICS");
    switches.atomic_reserve = Common::DiagRequest::TrySwitchEnabled("TRY_ATOMIC_RESERVE");
    switches.ro_mirror_exact = Common::DiagRequest::TrySwitchEnabled("TRY_RO_MIRROR_EXACT");
    switches.storage_raw_write_through =
        Common::DiagRequest::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH");
    return switches;
}
u64 TrySwitchesHash() {
    // Only distinguish live overrides from the configured defaults. The default-on atomics
    // already participate in the profile hash; adding their default values here would invalidate
    // every existing pipeline-cache record before a runtime toggle is requested.
    return u64{Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_RAWGLOBAL_FAST")} |
           (u64{Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_DIRECT_UNUSED_ATOMICS")}
            << 1) |
           (u64{Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_NATIVE_F32_ATOMICS")} << 2) |
           (u64{Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_ATOMIC_RESERVE")} << 3) |
           (u64{Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_RO_MIRROR_EXACT")} << 4) |
           (u64{Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_STORAGE_RAW_WRITE_THROUGH")}
            << 5);
}

u64 HashTranslationSwitches(const TranslationSwitches& switches) {
    // The raw coverage, raw global fast and mirror write-through fields are only in the text when
    // on: the files without them stay valid. The resolution setting is in the text only while it
    // scales (the NVN translation skips the rescaling pass at 1X,
    // HostTranslateInfo::skip_rescaling_pass).
    const std::string text = fmt::format(
        "loop_off={} denorm_preserve={} nan={} undef_zero={} tensor={}/{}{}{}{}",
        switches.loop_safety_checks_disabled, switches.fp32_denorm_preserve,
        static_cast<int>(switches.nan_outputs), switches.undef_is_zero, switches.tensor_mode_set,
        switches.tensor_mode_set ? switches.tensor_mode : std::string{},
        switches.raw_coverage ? " raw_coverage=1" : "",
        switches.raw_global_fast ? " raw_global_fast=1" : "",
        switches.ro_mirror_exact ? " ro_mirror_exact=1" : "",
        switches.storage_raw_write_through ? " storage_raw_write_through=1" : "");
    std::string key = text;
    if (!switches.lean_global_memory) {
        key += " lean_global=0";
    }
    if (switches.strong_global_fence) {
        key += " strong_fence=1";
    }
    if (switches.resolution_scaling_active) {
        key += fmt::format(" scale={}/{}/{:08x}/{:08x}", switches.resolution_up_scale,
                           switches.resolution_down_shift, switches.resolution_up_factor_bits,
                           switches.resolution_down_factor_bits);
    }
    // Only in the text for the titles whose tensor programs are lowered: other keys stay valid.
    if (switches.tensor_lowering != 0) {
        key += fmt::format(" tensor_lowering={}", static_cast<int>(switches.tensor_lowering));
    }
    if (Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_DIRECT_UNUSED_ATOMICS")) {
        key += fmt::format(" direct_unused={}", switches.direct_unused_atomics);
    }
    if (Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_NATIVE_F32_ATOMICS")) {
        key += fmt::format(" native_f32={}", switches.native_f32_atomics);
    }
    if (Common::DiagRequest::TrySwitchDiffersFromDefault("TRY_ATOMIC_RESERVE")) {
        key += fmt::format(" atomic_reserve={}", switches.atomic_reserve);
    }
    return Common::CityHash64(key.data(), key.size());
}

} // namespace Vulkan::Nvn
