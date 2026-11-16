// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright (c) 2026 Prithivi Nair, Prajwal R, Madhav S, Rashmi N Ugarakhod (Open-Frame-gen)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include <cstddef>
#include <vector>

#include "common/common_types.h"

namespace Vulkan::OpenFrameGenWeights {

inline constexpr std::size_t kTotalFloatCount = 87995u;

inline constexpr u32 kE1DwW = 0u; // 54 floats
inline constexpr u32 kE1DwB = 54u; // 6 floats
inline constexpr u32 kE1PwW = 60u; // 288 floats
inline constexpr u32 kE1PwB = 348u; // 48 floats
inline constexpr u32 kE2DwW = 396u; // 432 floats
inline constexpr u32 kE2DwB = 828u; // 48 floats
inline constexpr u32 kE2PwW = 876u; // 4608 floats
inline constexpr u32 kE2PwB = 5484u; // 96 floats
inline constexpr u32 kE3DwW = 5580u; // 864 floats
inline constexpr u32 kE3DwB = 6444u; // 96 floats
inline constexpr u32 kE3PwW = 6540u; // 12288 floats
inline constexpr u32 kE3PwB = 18828u; // 128 floats
inline constexpr u32 kBtnDwW = 18956u; // 1152 floats
inline constexpr u32 kBtnDwB = 20108u; // 128 floats
inline constexpr u32 kBtnPwW = 20236u; // 12288 floats
inline constexpr u32 kBtnPwB = 32524u; // 96 floats
inline constexpr u32 kD1DwW = 32620u; // 2016 floats
inline constexpr u32 kD1DwB = 34636u; // 224 floats
inline constexpr u32 kD1PwW = 34860u; // 21504 floats
inline constexpr u32 kD1PwB = 56364u; // 96 floats
inline constexpr u32 kD2DwW = 56460u; // 1728 floats
inline constexpr u32 kD2DwB = 58188u; // 192 floats
inline constexpr u32 kD2PwW = 58380u; // 12288 floats
inline constexpr u32 kD2PwB = 70668u; // 64 floats
inline constexpr u32 kD3DwW = 70732u; // 1008 floats
inline constexpr u32 kD3DwB = 71740u; // 112 floats
inline constexpr u32 kD3PwW = 71852u; // 3584 floats
inline constexpr u32 kD3PwB = 75436u; // 32 floats
inline constexpr u32 kSeFc0W = 75468u; // 1152 floats
inline constexpr u32 kSeFc0B = 76620u; // 12 floats
inline constexpr u32 kSeFc2W = 76632u; // 1152 floats
inline constexpr u32 kSeFc2B = 77784u; // 96 floats
inline constexpr u32 kHead0W = 77880u; // 9216 floats
inline constexpr u32 kHead0B = 87096u; // 32 floats
inline constexpr u32 kHead2W = 87128u; // 864 floats
inline constexpr u32 kHead2B = 87992u; // 3 floats

/// Unpacks the BatchNorm-folded Open-Frame-gen NanoModel (v17) weights to IEEE-754 float32.
[[nodiscard]] std::vector<f32> UnpackWeights();

} // namespace Vulkan::OpenFrameGenWeights
