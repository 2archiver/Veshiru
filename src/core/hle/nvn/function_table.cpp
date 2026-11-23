// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "core/hle/nvn/function_table.h"

namespace Core::NvnHle {
namespace {

constexpr std::array function_names{
#define NVN_FUNCTION(id, name, has_signature) std::string_view{#name},
#include "video_core/nvn/nvn_functions.inc"
#undef NVN_FUNCTION
};

} // namespace

std::span<const std::string_view> FunctionNames() {
    return function_names;
}

} // namespace Core::NvnHle
