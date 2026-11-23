// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

/// The create info of the library part named by `library_info.flags` (one subset): `ci`, a
/// complete monolithic create info, reduced to the state of that subset, with `stages` as its
/// stages. Points into `ci`, `library_info` and `stages`. The vertex input interface part never
/// has null vertex input or input assembly state, even when `ci` has none (mesh pipelines).
/// Public for the tests; GraphicsPipelineLibraryCache::Link is the user.
[[nodiscard]] VkGraphicsPipelineCreateInfo MakeLibraryPartCreateInfo(
    const VkGraphicsPipelineCreateInfo& ci,
    const VkGraphicsPipelineLibraryCreateInfoEXT& library_info,
    std::span<const VkPipelineShaderStageCreateInfo> stages);

/// Builds graphics pipelines with VK_EXT_graphics_pipeline_library: from four parts compiled
/// separately ("libraries") instead of one monolithic compile.
///
/// - vertex input interface: vertex input and input assembly state;
/// - pre-rasterization shaders: vertex, tessellation and geometry stages, viewport,
///   rasterization and tessellation state;
/// - fragment shader: the fragment stage, depth/stencil and multisample state;
/// - fragment output interface: color blend and multisample state.
///
/// Each part is compiled once and cached under exactly what it was compiled with. A pipeline is
/// then "fast-linked" from its four parts (without VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT),
/// which costs a fraction of a compile when the shader parts are cached already: pipelines that
/// share shaders (ShaderReuseCache) and differ in blending, vertex formats or depth state only
/// compile their shaders once. The legacy renderer (vk_graphics_pipeline.cpp) and the NVN replay
/// (nvn_pipeline_gpl.cpp) build pipelines with it.
///
/// Thread safe. A thread that needs a part another thread is compiling waits for it.
class GraphicsPipelineLibraryCache {
public:
    explicit GraphicsPipelineLibraryCache(const Device& device);
    ~GraphicsPipelineLibraryCache();

    GraphicsPipelineLibraryCache(const GraphicsPipelineLibraryCache&) = delete;
    GraphicsPipelineLibraryCache& operator=(const GraphicsPipelineLibraryCache&) = delete;

    /// Builds the pipeline that `ci`, a complete monolithic create info, describes by
    /// fast-linking libraries, compiling the parts not cached yet (none with
    /// `cached_parts_only`). `shaders` owns the shader modules `ci` names and stands for its
    /// pipeline layout: its address is in the keys of the shader parts, and the cache keeps it
    /// alive with the parts compiled from it. With `part_workers`, parts that are not cached are
    /// compiled side by side on those threads and the calling thread (it returns once all are
    /// done); the workers must not be busy with other long jobs.
    ///
    /// Returns a null pipeline, before compiling anything, when `ci` holds something the parts
    /// cannot express (the caller then compiles the pipeline whole) or, with
    /// `cached_parts_only`, when a part is not compiled yet. Throws vk::Exception when the
    /// driver fails.
    [[nodiscard]] vk::Pipeline Link(const VkGraphicsPipelineCreateInfo& ci,
                                    VkPipelineCache pipeline_cache,
                                    const std::shared_ptr<const void>& shaders,
                                    bool cached_parts_only,
                                    Common::ThreadWorker* part_workers = nullptr);

private:
    /// Everything the driver reads from the create info of one part, as 32-bit words.
    struct Key {
        std::vector<u32> words;

        bool operator==(const Key& rhs) const noexcept {
            return words == rhs.words;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept;
    };
    struct Library;

    /// Returns the part `part_ci` creates, compiling it if it is not cached.
    VkPipeline GetLibrary(Key key, const VkGraphicsPipelineCreateInfo& part_ci,
                          VkPipelineCache pipeline_cache,
                          const std::shared_ptr<const void>& shaders);

    /// Returns the part if it is compiled already, else a null handle.
    VkPipeline FindLibrary(const Key& key);

    const Device& device;
    std::mutex mutex;
    std::unordered_map<Key, std::shared_ptr<Library>, KeyHash> libraries;
};

} // namespace Vulkan
