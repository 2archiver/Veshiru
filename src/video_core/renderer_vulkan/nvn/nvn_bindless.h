// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "shader_recompiler/bindless_table.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

/// The set layout of the bindless table (shader_recompiler/bindless_table.h): a sampler array at
/// binding 0 and a sampled image array per shader image type and class, each updatable after
/// binding and partially bound. Every graphics pipeline that samples through the table has a
/// layout holding it as set BINDLESS_SET; the replay's own descriptor set is made from the same
/// definition.
[[nodiscard]] vk::DescriptorSetLayout MakeBindlessSetLayout(const Device& device);

/// The view type a table binding's images are made as.
[[nodiscard]] VkImageViewType BindlessViewType(Shader::TextureType type);

} // namespace Vulkan::Nvn
