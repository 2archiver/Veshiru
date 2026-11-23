// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>

#include <vulkan/vulkan_core.h>

namespace Vulkan {

/// EDEN_NX2_TENSOR: unset and "fast" run the game's tensor programs on the cooperative-matrix
/// path; every other value keeps that path off ("stub" is a diagnostic that lowers them to a
/// passthrough, the rest rejects them).
constexpr bool TensorModeAllowsCooperativeMatrix(const char* mode) {
    return mode == nullptr || std::string_view{mode} == "fast";
}

constexpr bool IsKnownTensorMode(const char* mode) {
    return TensorModeAllowsCooperativeMatrix(mode) || std::string_view{mode} == "stub";
}

constexpr bool IsTensorMatrixShape(const VkCooperativeMatrixPropertiesNV& matrix) {
    return matrix.MSize == 16 && matrix.NSize == 8 && matrix.KSize == 16 &&
           matrix.AType == VK_COMPONENT_TYPE_FLOAT16_NV &&
           matrix.BType == VK_COMPONENT_TYPE_FLOAT16_NV &&
           matrix.CType == VK_COMPONENT_TYPE_FLOAT16_NV &&
           matrix.DType == VK_COMPONENT_TYPE_FLOAT16_NV && matrix.scope == VK_SCOPE_SUBGROUP_NV;
}

} // namespace Vulkan
