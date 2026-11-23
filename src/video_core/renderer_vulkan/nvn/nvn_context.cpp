// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/diag_request.h"
#include "common/game_presets.h"
#include "common/assert.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/game_frame_size.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/renderer_base.h"
#include "video_core/upscaler.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/present/fsr.h"
#include "video_core/renderer_vulkan/present/fsr3.h"
#include "video_core/renderer_vulkan/present/sgsr.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/renderer_vulkan/nvn/nvn_write_back.h"
#include "video_core/vulkan_common/vma.h"
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace Vulkan::Nvn {
namespace {

/// The render pass the frames' framebuffers are made for: the window adapting filter's, so the
/// presentation passes can draw into them (framebuffers are only used with compatible render
/// passes).
vk::RenderPass CreatePresentRenderPass(const Device& device, VkFormat format) {
    return CreateWrappedRenderPass(device, format, VK_IMAGE_LAYOUT_UNDEFINED);
}

/// Hands frames to the renderer's PresentManager and the window.
class PresentManagerTarget final : public PresentTarget {
public:
    PresentManagerTarget(const Device& device_, PresentManager& present_manager_,
                         Core::Frontend::EmuWindow& window_)
        : device{device_}, present_manager{present_manager_}, window{window_} {
        render_pass_format = present_manager.FrameFormat();
        render_pass = CreatePresentRenderPass(device, render_pass_format);
    }

    Frame* AcquireFrame() override {
        Frame* frame = present_manager.GetRenderFrame();
        const auto& layout = window.GetFramebufferLayout();
        const VkFormat format = present_manager.FrameFormat();
        const VkFormat image_format = present_manager.GetFrameImageFormat(format);
        if (format != render_pass_format) {
            render_pass = CreatePresentRenderPass(device, format);
            render_pass_format = format;
        }
        const bool storage = PresentFrameNeedsStorage(
            VideoCore::Upscaler::Effective() == Settings::Upscaler::Dlaa,
            Settings::values.frame_gen.GetValue(), Settings::values.frame_gen_type.GetValue());
        if (!frame->image || frame->format != format || frame->image_format != image_format ||
            frame->width != layout.width || frame->height != layout.height ||
            present_manager.NeedsStorage(frame, storage)) {
            // The frame's previous image is no longer in use: GetRenderFrame waited for its
            // presentation to finish. The image comes from the context's own thread-safe
            // allocator: the renderer's one is externally synchronized and other renderer threads
            // may still create images with it.
            try {
                present_manager.RecreateFrame(frame, layout.width, layout.height, image_format,
                                              format, *render_pass, storage, allocator);
            } catch (...) {
                // Give the frame back (the caller never sees it) and force a rebuild next time.
                frame->width = 0;
                present_manager.ReturnFrame(frame);
                throw;
            }
            if (allocator != nullptr &&
                std::ranges::find(owned_frames, frame) == owned_frames.end()) {
                owned_frames.push_back(frame);
            }
        }
        return frame;
    }

    void ReleaseFrame(Frame* frame) override {
        present_manager.ReturnFrame(frame);
    }

    [[nodiscard]] s32 FrameFormat() const override {
        return static_cast<s32>(present_manager.FrameFormat());
    }

    [[nodiscard]] s32 SwapchainImageFormat() const override {
        return static_cast<s32>(present_manager.SwapchainImageFormat());
    }

    [[nodiscard]] bool UsesPresentThread() const override {
        return present_manager.UsesPresentThread();
    }

    [[nodiscard]] Layout::FramebufferLayout ScreenLayout() const override {
        return window.GetFramebufferLayout();
    }

    [[nodiscard]] size_t MaxExtraFrames() const override {
        return present_manager.MaxExtraFrames();
    }

    void Attach(MemoryAllocator& allocator_) override {
        allocator = &allocator_;
    }

    void Drain() override {
        present_manager.WaitPresent();
    }

    void Detach() override {
        // The presentation worker has drained and the device is idle.
        for (Frame* frame : owned_frames) {
            frame->framebuffer = {};
            frame->storage_view = {};
            frame->image_view = {};
            frame->image = {};
            frame->width = 0;
            frame->height = 0;
        }
        owned_frames.clear();
        allocator = nullptr;
    }

    void PresentFrame(Frame* frame) override {
        present_manager.PresentSubmitted(frame);
        window.OnFrameDisplayed();
    }

private:
    const Device& device;
    PresentManager& present_manager;
    Core::Frontend::EmuWindow& window;
    vk::RenderPass render_pass;
    VkFormat render_pass_format{VK_FORMAT_UNDEFINED};
    MemoryAllocator* allocator{};
    std::vector<Frame*> owned_frames;
};

/// `nvn-pass <n>` / `nvn-draw <n>` of the diag request file: arm a one-shot dump of the next frame
/// (nvn_debug.h). Registered when a replay context exists (the NVN HLE is on).
void RegisterDiagRequests() {
    using Common::DiagRequest::Request;
    Common::DiagRequest::Register("nvn-pass", [](const Request& request) {
        const u32 pass = request.NumberArg(0, 0);
        if (pass == 0) {
            LOG_WARNING(Render_Vulkan, "[diag] nvn-pass: needs a pass number (1-based, as in the draw log)");
            return;
        }
        Debug::RequestDump(pass, 0, request.Directory());
        LOG_INFO(Render_Vulkan, "[diag] nvn-pass: dumping the targets of pass {} of the next frame into {}",
                 pass, request.Directory().string());
    });
    Common::DiagRequest::Register("nvn-draw", [](const Request& request) {
        const u32 draw = request.NumberArg(0, 0);
        if (draw == 0) {
            LOG_WARNING(Render_Vulkan, "[diag] nvn-draw: needs a draw number (1-based, as in the draw log)");
            return;
        }
        Debug::RequestDump(0, draw, request.Directory());
        LOG_INFO(Render_Vulkan, "[diag] nvn-draw: logging draw {} of the next frame (also to {}/nvn-draw-{}.txt)",
                 draw, request.Directory().string(), draw);
    });
}

} // Anonymous namespace

Context::Impl::Impl(const Device& device_, std::mutex& submit_mutex_,
                    std::unique_ptr<PresentTarget> present, ContextOptions options_,
                    std::unique_ptr<PresentFrameGen> present_frame_gen_)
    : device{device_}, dld{device_.GetDispatchLoader()}, logical{*device_.GetLogical()},
      submit_mutex{submit_mutex_}, present_target{std::move(present)}, options{options_},
      present_frame_gen{std::move(present_frame_gen_)} {
    if (const char* directory = std::getenv("EDEN_NX2_FRAME_CAPTURE"); directory != nullptr) {
        capture_directory = directory;
    }
    Debug::ResetRequests();
    RegisterDiagRequests();
    VideoCore::GameFrameSize::Reset();
    // Diagnostics: decide render conditions on the CPU and log the first ones (what the game
    // compares, and what it makes of it).
    if (const char* cpu = std::getenv("EDEN_NX2_NVN_CPU_CONDITIONS");
        cpu != nullptr && cpu[0] != '0') {
        options.allow_conditional_rendering = false;
    }
    // Experiment: CPU-accessible pools get device-local mirrors instead of imported host memory
    // (1-2 FPS in DK: every mirror range a draw binds is uploaded and ends the render pass).
    if (const char* no_import = std::getenv("EDEN_NX2_NVN_NO_HOST_IMPORT");
        no_import != nullptr && no_import[0] != '0') {
        options.allow_host_import = false;
    }
    vma = device.CreateThreadSafeAllocator();
    allocator = std::make_unique<MemoryAllocator>(device, vma);
    vma_device_strict = options.vma_device_strict.value_or(Debug::Get().try_vma_device_strict);
    allocator->TrackSpills(&spills, vma_device_strict);
    if (vma_device_strict) {
        LOG_INFO(Render_Vulkan, "[nvn] device-local allocations: strict (EDEN_NX2_TRY_VMA_DEVICE_STRICT), "
                                "over the budget they stay device-local, host memory last");
    }
    importer.emplace(device, vma);
    if (device.IsBufferDeviceAddressSupported() && device.IsShaderInt64Supported()) {
        pool_usage = POOL_BUFFER_USAGE_ADDRESSABLE;
    }
    const bool storage_vram_supported = device.IsBufferDeviceAddressSupported() &&
                                        device.IsShaderInt64Supported() &&
                                        device.IsDescriptorAliasingSupported();
    try_storage_vram = options.try_storage_vram.value_or(storage_vram_supported &&
                                                         Debug::Get().try_storage_vram);
    try_storage_vram_max_bytes =
        options.try_storage_vram_max_bytes.value_or(Debug::Get().try_storage_vram_max_bytes);
    pool_mirror_mode = (std::min)(
        options.pool_mirror.value_or(try_storage_vram ? 1u : Debug::Get().pool_mirror), 2u);
    // The uniform mirrors are pool mirrors: they need the mode's machinery (and the write tracker).
    ubo_mirror = options.ubo_mirror.value_or(Debug::Get().try_ubo_mirror) && pool_mirror_mode != 0;
    vtx_mirror = options.vtx_mirror.value_or(Debug::Get().try_vtx_mirror) && pool_mirror_mode != 0;
    vtxcov = vtx_mirror ? options.vtxcov.value_or(Debug::Get().try_vtxcov) & VTXCOV_ALL : 0u;
    vtxcov_diag = options.vtxcov_diag.value_or(Debug::Get().vtxcov_diag);
    if (vtxcov != 0 || vtxcov_diag) {
        LOG_INFO(Render_Vulkan, "[nvn] vtxcov: switches {} (EDEN_NX2_TRY_VTXCOV_*), diag {}{}",
                 VtxCovSwitchNames(vtxcov), vtxcov_diag ? "on" : "off",
                 options.vtxcov.value_or(Debug::Get().try_vtxcov) != 0 && !vtx_mirror
                     ? "; the switches need EDEN_NX2_TRY_VTX_MIRROR=1 and pool mirrors"
                     : "");
    }
    dispatch_order_by_barriers =
        options.dispatch_order_by_barriers.value_or(Debug::Get().try_dispatch_order);
    flush_tracked = options.flush_tracked.value_or(Debug::Get().try_flush_tracked);
    mirror_hash_tracked = options.mirror_hash_tracked.value_or(!Debug::Get().mirror_hash_always);
    raw_split_import.store(Debug::Get().raw_split_import, std::memory_order_relaxed);
    tex_hash_tracked = mirror_hash_tracked && !Debug::Get().tex_hash_always;
    tex_page_hash = tex_hash_tracked && Debug::Get().tex_page_hash;
    tex_flush_trusted = tex_hash_tracked && Debug::Get().tex_flush_trusted;
    invalidate_wait_ms = options.invalidate_wait_ms.value_or(Debug::Get().try_invalidate_writeback);
    raw_keep_entries = options.raw_keep_entries.value_or(Debug::Get().try_raw_keep_entries);
    refused_shadow = options.refused_shadow.value_or(Debug::Get().try_refused_shadow);
    storage_cache_budget = options.storage_cache_budget_bytes.value_or(
        Debug::Get().storage_budget_bytes != 0 ? Debug::Get().storage_budget_bytes
                                               : DEFAULT_STORAGE_CACHE_BUDGET);
    trim_age_presents = options.storage_trim_age_presents.value_or(Debug::Get().trim_age_presents);
    trim_legacy = options.storage_trim_legacy.value_or(Debug::Get().trim_legacy);
    barrier_merge =
        BarrierMergeMask(options.barrier_merge.value_or(Debug::Get().try_barrier_merge));
    astc_gpu_enabled = AstcGpuDecoder::Supported(device);
    if (astc_gpu_enabled) {
        astc_decoder_future = std::async(std::launch::async, [this] {
            Common::SetCurrentThreadName("NvnAstcInit");
            return std::make_unique<AstcGpuDecoder>(device);
        });
    }
    if (!device.IsOptimalAstcSupported()) {
        // Read past the override renderer_vulkan.cpp sets for MaxwellToVK (Uncompressed: the
        // replay's stand-ins start as RGBA8 and become BC per storage, ToBcStandin).
        const AstcStandin chosen = ChooseAstcStandin(
            std::getenv("EDEN_NX2_NVN_ASTC_STANDIN"),
            Settings::values.astc_recompression.GetValue());
        const bool encodable = BcGpuEncoder::Supported(device);
        astc_standin = !encodable ? AstcStandin::Rgba8
                       : chosen == AstcStandin::Bc7 && !BcGpuEncoder::SupportsBc7(device)
                           ? AstcStandin::Bc3
                           : chosen;
        bc7_quality = ParseBc7Quality(std::getenv("EDEN_NX2_NVN_BC7_QUALITY"));
        smart_bc = ParseSmartBc(astc_standin, Settings::values.nvn_smart_bc.GetValue(),
                                std::getenv("EDEN_NX2_NVN_SMART_BC"),
                                std::getenv("EDEN_NX2_NVN_SMART_BC1"),
                                std::getenv("EDEN_NX2_NVN_SMART_BC45"),
                                std::getenv("EDEN_NX2_NVN_BC_REFINE"));
        const bool bc45_unsupported = smart_bc.bc45 && !BcGpuEncoder::SupportsBc45(device);
        smart_bc.bc45 = smart_bc.bc45 && !bc45_unsupported;
        LOG_INFO(Render_Vulkan, "[nvn] ASTC stand-ins: {}{}{}{}", AstcStandinName(astc_standin),
                 astc_standin == AstcStandin::Bc7 ? fmt::format(" (quality {})", bc7_quality)
                                                  : std::string{},
                 chosen == astc_standin ? ""
                 : encodable            ? " (the device cannot hold BC7)"
                                        : " (the device cannot run the BC encoder)",
                 astc_standin == AstcStandin::Bc3
                     ? fmt::format(", smart BC {}{} (EDEN_NX2_NVN_SMART_BC, _SMART_BC1, "
                                   "_SMART_BC45, _BC_REFINE)",
                                   SmartBcName(smart_bc),
                                   bc45_unsupported ? "; the device cannot hold BC4/BC5" : "")
                     : std::string{});
        if (astc_standin != AstcStandin::Rgba8) {
            const bool bc7 = astc_standin == AstcStandin::Bc7;
            bc_encoder_future = std::async(std::launch::async, [this, bc7] {
                Common::SetCurrentThreadName("NvnBcInit");
                return std::make_unique<BcGpuEncoder>(device, bc7);
            });
            const auto families = device.GetPhysical().GetQueueFamilyProperties();
            const u32 valid_bits = device.GetGraphicsFamily() < families.size()
                                       ? families[device.GetGraphicsFamily()].timestampValidBits
                                       : 0;
            bc_timing = valid_bits != 0;
            bc_timestamp_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
            const float period = device.GetPhysical().GetProperties().limits.timestampPeriod;
            bc_timestamp_period = period > 0.0f ? period : 1.0;
        }
    }
    untile_gpu_enabled = UntileGpu::Supported(device);
    if (untile_gpu_enabled) {
        untile_gpu_future = std::async(std::launch::async, [this] {
            Common::SetCurrentThreadName("NvnUntileInit");
            return std::make_unique<UntileGpu>(device);
        });
    }
    if (AliasTransferGpu::Supported(device)) {
        alias_gpu_future = std::async(std::launch::async, [this] {
            Common::SetCurrentThreadName("NvnAliasInit");
            return std::make_unique<AliasTransferGpu>(device);
        });
    }
    if (present_target) {
        present_target->Attach(*allocator);
    }
    queue = device.GetGraphicsQueue();
    command_pool = device.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    // The performance HUD needs the replay's host GPU timestamps: mode 1, without the log reports
    // unless EDEN_NX2_NVN_GPU_PROFILE asked for them. The cost split (EDEN_NX2_NVN_COSTSPLIT)
    // reads its GPU time from the profiler: mode 1 at least.
    {
        if (const std::string& text = Debug::Get().costsplit; !text.empty()) {
            std::string error;
            if (std::optional<CostSplit::Config> config = CostSplit::Parse(text, error)) {
                std::string modes;
                for (const CostSplit::Mode mode : config->modes) {
                    modes += fmt::format("{}{}", modes.empty() ? "" : ",",
                                         CostSplit::ModeName(mode));
                }
                LOG_WARNING(Render_Vulkan,
                            "[nvn] costsplit: MEASUREMENT ONLY, the picture is wrong on purpose: "
                            "class {} from present {}, slots of {} presents (warm-up {}): base "
                            "between each of {}",
                            CostSplit::ClassLabel(*config), config->start, config->period,
                            (std::min)(CostSplit::WARMUP, config->period / 2), modes);
                costsplit = std::make_unique<CostSplit::Recorder>(std::move(*config));
            } else {
                LOG_ERROR(Render_Vulkan, "[nvn] costsplit: EDEN_NX2_NVN_COSTSPLIT='{}' ignored: {}",
                          text, error);
            }
        }
        const u32 requested = Debug::Get().gpu_profile;
        const u32 mode = requested != 0 ? requested : costsplit ? 1u : 0u;
        gpu_profiler = GpuProf::Profiler::Create(device, mode, requested == 0, costsplit.get());
        if (costsplit && !gpu_profiler) {
            LOG_ERROR(Render_Vulkan, "[nvn] costsplit: no GPU profiler on this device; the cost "
                                     "split measures nothing");
            costsplit.reset();
        }
    }
    draw_resources = std::make_unique<DrawResources>(*this);
    replay_thread = std::thread([this] {
        Common::SetCurrentThreadName("NvnReplay");
        ReplayMain();
    });
    if (options.background_backing) {
        backing_thread = std::thread([this] {
            Common::SetCurrentThreadName("NvnBacking");
                BackingMain();
        });
    }
    complete_thread = std::thread([this] {
        Common::SetCurrentThreadName("NvnComplete");
        CompleteMain();
    });
}

AstcGpuDecoder* Context::Impl::AstcDecoder() {
    if (!astc_decoder && astc_decoder_future.valid()) {
        if (astc_decoder_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return nullptr; // still compiling its pipeline
        }
        try {
            astc_decoder = astc_decoder_future.get();
        } catch (const std::exception& exception) {
            astc_decoder_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] GPU ASTC decoder unavailable, decoding on the CPU: {}",
                      exception.what());
        }
    }
    return astc_decoder.get();
}

BcGpuEncoder* Context::Impl::BcEncoder() {
    if (!bc_encoder && bc_encoder_future.valid()) {
        try {
            bc_encoder = bc_encoder_future.get();
        } catch (const std::exception& exception) {
            bc_encoder_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] BC encoder unavailable, BC stand-ins stay empty: {}",
                      exception.what());
        }
    }
    return bc_encoder.get();
}

std::shared_ptr<BcScratch> Context::Impl::AcquireBcScratch(u32 width, u32 height, u32 layers) {
    const bool bc7 = astc_standin == AstcStandin::Bc7;
    for (size_t i = 0; i < bc_scratches.size(); ++i) {
        if (bc_scratches[i] && bc_scratches[i]->Fits(width, height, layers, bc7)) {
            if (i != 0) {
                std::rotate(bc_scratches.begin(), bc_scratches.begin() + static_cast<std::ptrdiff_t>(i),
                            bc_scratches.begin() + static_cast<std::ptrdiff_t>(i + 1));
            }
            return bc_scratches.front();
        }
    }
    const auto texel_bytes = [](u32 w, u32 h, u32 l) { return u64{w} * h * l * 4; };
    u32 w = width;
    u32 h = height;
    u32 l = layers;
    bool replace_front = false;
    if (!bc_scratches.empty() && bc_scratches.front()) {
        // Grow to cover both shapes while that stays in the budget, else take the new shape.
        const u32 grown_w = (std::max)(bc_scratches.front()->width, width);
        const u32 grown_h = (std::max)(bc_scratches.front()->height, height);
        const u32 grown_l = (std::max)(bc_scratches.front()->layers, layers);
        if (texel_bytes(grown_w, grown_h, grown_l) <= BC_SCRATCH_BUDGET) {
            w = grown_w;
            h = grown_h;
            l = grown_l;
            replace_front = true;
        }
    }
    try {
        auto scratch = MakeBcScratch(device, *allocator, w, h, l, bc7);
        if (texel_bytes(w, h, l) <= BC_SCRATCH_BUDGET) {
            if (replace_front) {
                bc_scratches.front() = scratch;
            } else {
                bc_scratches.insert(bc_scratches.begin(), scratch);
                if (bc_scratches.size() > 4) {
                    bc_scratches.pop_back();
                }
            }
        }
        return scratch;
    } catch (const vk::Exception& exception) {
        if (!bc_scratch_failed_logged) {
            bc_scratch_failed_logged = true;
            LOG_ERROR(Render_Vulkan, "[nvn] BC stand-in scratch {}x{}x{} failed: {}", w, h, l,
                      exception.what());
        }
        return nullptr;
    }
}

void Context::Impl::CompleteBcTimes(const BcUploadTimes& times) {
    const u32 queries = static_cast<u32>(times.uploads.size() * 2);
    std::vector<u64> ticks(queries);
    if (queries == 0 ||
        device.GetLogical().GetQueryResults(*times.pool, 0, queries, ticks.size() * sizeof(u64),
                                            ticks.data(), sizeof(u64),
                                            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
        return;
    }
    for (size_t i = 0; i < times.uploads.size(); ++i) {
        const BcUploadTimes::Upload& upload = times.uploads[i];
        const u64 elapsed = (ticks[2 * i + 1] - ticks[2 * i]) & bc_timestamp_mask;
        const u64 ns = static_cast<u64>(static_cast<double>(elapsed) * bc_timestamp_period);
        bc_window.uploads.fetch_add(1, std::memory_order_relaxed);
        bc_window.bytes.fetch_add(upload.bytes, std::memory_order_relaxed);
        bc_window.gpu_ns.fetch_add(ns, std::memory_order_relaxed);
        u64 max = bc_window.max_ns.load(std::memory_order_relaxed);
        while (ns > max && !bc_window.max_ns.compare_exchange_weak(max, ns)) {
        }
        if (upload.bytes >= BC_TIMED_LOG_BYTES) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] {} stand-in upload {}x{}x{} lv {} fmt {:#x}: {} MiB, GPU {:.1f} ms "
                     "(ASTC decode or texel copy and encode)",
                     upload.kind, upload.width, upload.height, upload.layers, upload.levels,
                     upload.format, upload.bytes >> 20, static_cast<double>(ns) / 1e6);
        }
    }
}

void Context::Impl::LogBcTimeWindow(u64 present) {
    const u64 uploads = bc_window.uploads.exchange(0, std::memory_order_relaxed);
    const u64 untimed = bc_window.untimed.exchange(0, std::memory_order_relaxed);
    const u64 bytes = bc_window.bytes.exchange(0, std::memory_order_relaxed);
    const u64 gpu_ns = bc_window.gpu_ns.exchange(0, std::memory_order_relaxed);
    const u64 max_ns = bc_window.max_ns.exchange(0, std::memory_order_relaxed);
    if (uploads + untimed != 0) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] BC stand-in uploads completed in presents {}-{}: {} ({} MiB), GPU {:.1f} ms "
                 "(largest {:.1f} ms){}",
                 bc_window_first_present, present, uploads, bytes >> 20,
                 static_cast<double>(gpu_ns) / 1e6, static_cast<double>(max_ns) / 1e6,
                 untimed != 0 ? fmt::format(", {} more untimed", untimed) : std::string{});
    }
    std::array<u64, 5> classes{};
    u64 classified = 0;
    for (size_t i = 0; i < classes.size(); ++i) {
        classes[i] = smart_window.classes[i].exchange(0, std::memory_order_relaxed);
        classified += classes[i];
    }
    const u64 scan_bytes = smart_window.scan_bytes.exchange(0, std::memory_order_relaxed);
    const u64 scan_ns = smart_window.scan_ns.exchange(0, std::memory_order_relaxed);
    const u64 refits = smart_window.refits.exchange(0, std::memory_order_relaxed);
    const u64 pinned = smart_window.pinned.exchange(0, std::memory_order_relaxed);
    if (classified + refits + pinned != 0) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] smart BC scans in presents {}-{}: {} (L {}, LA {}, RGB {}, RGBA {}, all "
                 "zero {}), {:.1f} MiB of ASTC in {:.2f} ms; {} re-classified to BC3, {} kept at "
                 "BC3 by copies",
                 bc_window_first_present, present, classified, classes[0], classes[1], classes[2],
                 classes[3], classes[4], static_cast<double>(scan_bytes) / 1048576.0,
                 static_cast<double>(scan_ns) / 1e6, refits, pinned);
    }
    bc_window_first_present = present + 1;
}

UntileGpu* Context::Impl::Untile() {
    if (!untile_gpu && untile_gpu_future.valid()) {
        if (untile_gpu_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return nullptr; // still compiling its pipeline
        }
        try {
            untile_gpu = untile_gpu_future.get();
        } catch (const std::exception& exception) {
            LOG_ERROR(Render_Vulkan, "[nvn] GPU untile unavailable, untiling on the CPU: {}",
                      exception.what());
        }
    }
    return untile_gpu.get();
}

AliasTransferGpu* Context::Impl::AliasGpu() {
    if (!alias_gpu && alias_gpu_future.valid()) {
        // A texture read through another texture's bytes needs them now: wait for the pipeline.
        try {
            alias_gpu = alias_gpu_future.get();
        } catch (const std::exception& exception) {
            LOG_ERROR(Render_Vulkan,
                      "[nvn] alias transfer pipeline unavailable, overlapping textures of other "
                      "layouts cannot exchange contents: {}",
                      exception.what());
        }
    }
    return alias_gpu.get();
}

Context::Impl::~Impl() {
    {
        std::scoped_lock lock{state_mutex};
        stopping = true;
    }
    work_cv.notify_all();
    if (replay_thread.joinable()) {
        replay_thread.join();
    }
    // Suspended streams own recorders with open command buffers. Abort them now: the Vulkan
    // objects and the pieces they already submitted exist, and their flights are still unsubmitted
    // (the completion thread has not been told to drain them).
    try {
        streams.Clear();
    } catch (const std::exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] aborting suspended submissions failed: {}",
                  exception.what());
    }
    {
        std::scoped_lock lock{backing_queue_mutex};
        backing_stop = true;
        backing_queue.clear();
    }
    backing_cv.notify_all();
    if (backing_thread.joinable()) {
        backing_thread.join();
    }
    // Everything still queued never ran: complete it so the completion thread can drain.
    {
        std::scoped_lock lock{state_mutex};
        for (const auto& flight : inflight) {
            if (!flight->submitted) {
                flight->submitted = true;
            }
        }
        commands.clear();
    }
    // The ASTC decoder's and the untile pipelines are built on worker threads from this device: let
    // them finish (the results are dropped with the members) before the device is idled and their
    // objects freed.
    if (astc_decoder_future.valid()) {
        astc_decoder_future.wait();
    }
    if (untile_gpu_future.valid()) {
        untile_gpu_future.wait();
    }
    if (bc_encoder_future.valid()) {
        bc_encoder_future.wait();
    }
    if (present_target) {
        present_target->Drain();
    }
    {
        std::scoped_lock lock{submit_mutex};
        void(device.GetLogical().WaitIdle());
    }
    complete_cv.notify_all();
    if (complete_thread.joinable()) {
        complete_thread.join();
    }
    if (screenshot_thread.joinable()) {
        screenshot_thread.join();
    }
    if (loss_report_thread.joinable()) {
        loss_report_thread.join();
    }
    LOG_INFO(Render_Vulkan, "[nvn] early submits: {} extra command buffer submissions over {} submissions",
             stats.early_submits.load(), stats.submissions.load());
    // The raw coverage diagnostic's run summary; the GPU is idle, so the last, unfinished
    // window's misses can be counted. The rings go before the allocator they came from.
    if (raw_coverage) {
        if (raw_coverage->Valid()) {
            for (const std::string& line :
                 raw_coverage_report.Summary(raw_coverage->TailMisses(), true)) {
                LOG_INFO(HW_GPU, "{}", line);
            }
        }
        raw_coverage.reset();
    }
    // Release every Vulkan object before the allocator they came from.
    raw_table = {};
    raw_write_through_table = {};
    if (present_target) {
        present_target->Detach();
    }
    retired.clear();
    commands.clear();
    inflight.clear();
    draw_contexts.clear();
    draw_resources.reset();
    storage_index.clear();
    storage_cache.clear();
    ClearRecycledImages();
    textures.clear();
    target_aliases.clear();
    pools_by_address.clear();
    pools.clear();
    // Every storage that held a page is gone: the pool frees its blocks through the allocator,
    // which is destroyed below, so it has to go first (a live SparseBacking would keep it).
    if (sparse_pages) {
        ASSERT_MSG(sparse_pages.use_count() == 1, "a sparse backing outlives the context");
        sparse_pages.reset();
    }
    import_chunks.clear(); // after the pool buffers bound into them
    clear_render_passes.clear();
    counters_published.store(nullptr);
    counters_engine.reset();
    if (gpu_profiler) {
        gpu_profiler->ReportTotals();
        gpu_profiler.reset();
    }
    queues.clear();
    importer.reset();
    // The frame generation chains record through the presentation passes' recorder.
    present_frame_gen.reset();
    present_scratch = {};
    window_filter.reset();
    window_fsr.reset();
    window_fsr3.reset();
    window_sgsr.reset();

    final_fsr.reset();
    upscaler_fsr.reset();
    mirror_merge_stats = {}; // allocator-owned, like the pool buffers above
    bc_scratches.clear();
    staging_cache = {};
    quad_indirect.reset();
    allocator.reset();
    command_buffer_allocations.clear();
    command_pool.reset();
    if (vma) {
        vmaDestroyAllocator(vma);
    }
}

void Context::Impl::Enqueue(Command&& command, std::shared_ptr<InFlight> flight) {
    {
        std::scoped_lock lock{state_mutex};
        if (stopping) {
            return;
        }
        if (flight) {
            inflight.push_back(std::move(flight));
        }
        commands.push_back(std::move(command));
        commands_enqueued.fetch_add(1, std::memory_order_release);
    }
    work_cv.notify_one();
}

void Context::Impl::ReplayMain() {
    submit_batch.batching = true;
    SCOPE_EXIT {
        submit_batch.batching = false;
        try {
            FlushSubmitBatch();
        } catch (const std::exception&) {
        }
    };
    for (;;) {
        if (!submit_batch.entries.empty() || !submit_batch.marks.empty()) {
            bool idle_wait = false;
            {
                std::scoped_lock lock{state_mutex};
                idle_wait = !stopping && commands.empty();
            }
            if (idle_wait) {
                replay_stage.store("submit", std::memory_order_relaxed);
                try {
                    FlushSubmitBatch();
                } catch (const std::exception& exception) {
                    stats.ops_failed.fetch_add(1, std::memory_order_relaxed);
                    LOG_ERROR(Render_Vulkan, "[nvn] batched submit failed: {}", exception.what());
                    NoteIfDeviceLost("a batched submit");
                }
            }
        }
        Command command;
        const bool streams_waiting = !streams.Idle();
        {
            replay_stage.store("idle", std::memory_order_relaxed);
            std::unique_lock lock{state_mutex};
            if (streams_waiting) {
                // A wait nobody satisfies would hold its queue forever: look again.
                work_cv.wait_for(lock, std::chrono::milliseconds(250),
                                 [this] { return stopping || !commands.empty(); });
            } else {
                work_cv.wait(lock, [this] { return stopping || !commands.empty(); });
            }
            if (stopping) {
                break;
            }
            if (commands.empty()) {
                lock.unlock();
                PumpStreams(true);
                continue;
            }
            command = std::move(commands.front());
            commands.pop_front();
        }
        replay_stage.store("dispatch", std::memory_order_relaxed);
        try {
            Dispatch(command);
        } catch (const std::exception& exception) {
            stats.ops_failed.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render_Vulkan,
                      "[nvn] replay command failed: {} (command kind {}, stage {}, seq {})",
                      exception.what(), command.index(), replay_stage.load(), replay_seq.load());
            NoteIfDeviceLost("a replay command");
            // The flights were handed to the streams (the items fail their own flights).
            if (auto* submit = std::get_if<CmdSubmit>(&command); submit && submit->flight) {
                FailFlight(*submit->flight);
            } else if (auto* present = std::get_if<CmdPresent>(&command);
                       present && present->flight) {
                FailFlight(*present->flight);
            } else if (auto* flush = std::get_if<CmdFlushTexture>(&command)) {
                // The guest thread waits for this answer: a failed readback is an answer too.
                try {
                    flush->done->set_value(false);
                } catch (const std::future_error&) {
                    // already answered before the failure
                }
            }
        }
        commands_dispatched.fetch_add(1, std::memory_order_release);
        RetireCompleted();
    }
}

void Context::Impl::Dispatch(Command& command) {
    struct Visitor {
        Impl& impl;
        void operator()(CmdCreatePool& c) const {
            impl.CreatePoolNow(c.desc);
        }
        void operator()(CmdCreateTexture& c) const {
            impl.CreateTextureNow(c.desc);
        }
        void operator()(CmdCreateSampler& c) const {
            impl.samplers[c.desc.serial] = c.desc;
        }
        void operator()(CmdCreateProgram& c) const {
            const u64 serial = c.desc.serial;
            auto program = std::make_shared<VC::ProgramDesc>(std::move(c.desc));
            impl.draw_resources->OnProgramCreated(program);
            impl.programs[serial] = std::move(program);
        }
        void operator()(CmdDestroy& c) const {
            impl.DestroyNow(c.serial);
        }
        void operator()(CmdSubmit& c) const {
            impl.ProcessSubmit(std::move(c));
        }
        void operator()(CmdSetTitle& c) const {
            impl.mario_kart_world = c.title_id == Settings::MarioKartWorldProgramId;
            if (impl.mario_kart_world) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] Mario Kart World: sparse CPU writes use physical-pool lookup; "
                         "exact buffer lookup cache and lazy raw mirror invalidation active");
            }
            impl.mkw_coalesce_early_submit =
                impl.mario_kart_world && std::getenv("EDEN_NX2_NVN_EARLY_SUBMIT_US") == nullptr &&
                std::getenv("EDEN_NX2_NVN_EARLY_FIRST_US") == nullptr &&
                std::getenv("EDEN_NX2_NVN_EARLY_PASS_US") == nullptr;
            if (impl.mkw_coalesce_early_submit && impl.options.early_submit) {
                LOG_INFO(Render_Vulkan, "[nvn] Mario Kart World: early submit at most twice per "
                                        "submission; live passes and shadows use the pass budget");
            }
            if (!impl.options.barrier_merge &&
                std::getenv("EDEN_NX2_TRY_BARRIER_MERGE") == nullptr) {
                impl.barrier_merge = BarrierMergeMask(impl.mario_kart_world ? 1 : 0);
            }
            if (!impl.options.dispatch_order_by_barriers.has_value() &&
                std::getenv("EDEN_NX2_TRY_DISPATCH_ORDER") == nullptr) {
                impl.dispatch_order_by_barriers = impl.mario_kart_world;
            }
            if (impl.mario_kart_world && !impl.options.storage_trim_age_presents.has_value() &&
                std::getenv("EDEN_NX2_NVN_TRIM_AGE_PRESENTS") == nullptr &&
                impl.trim_age_presents == 0) {
                impl.trim_age_presents = 300;
                LOG_INFO(Render_Vulkan,
                         "[nvn] Mario Kart World: aging idle render targets after {} presents "
                         "(EDEN_NX2_NVN_TRIM_AGE_PRESENTS)",
                         impl.trim_age_presents);
            }
            if (impl.mario_kart_world && !impl.options.storage_cache_budget_bytes.has_value() &&
                Debug::Get().storage_budget_bytes == 0) {
                if (const u64 device_budget = impl.device.GetDeviceMemoryBudget();
                    device_budget != 0) {
                    const u64 scaled = device_budget / 10 * 6;
                    if (scaled > impl.storage_cache_budget) {
                        impl.storage_cache_budget = scaled;
                        LOG_INFO(Render_Vulkan,
                                 "[nvn] Mario Kart World: storage cache budget {} MiB (60% of {} "
                                 "MiB device budget; EDEN_NX2_NVN_STORAGE_BUDGET_MIB overrides)",
                                 impl.storage_cache_budget >> 20, device_budget >> 20);
                    }
                }
            }
            impl.draw_resources->OpenCache(c.title_id);
        }
        void operator()(CmdPresent& c) const {
            impl.ProcessPresent(std::move(c));
        }
        void operator()(CmdMapVirtual& c) const {
            impl.MapVirtualNow(c.pool_serial, c.mappings);
        }
        void operator()(CmdCpuWrite& c) const {
            if (c.host) {
                impl.NotifyHostWriteNow(c.address, c.size);
            } else {
                impl.NotifyCpuWriteNow(c.address, c.size);
            }
        }
        void operator()(CmdEventCpu& c) const {
            impl.NotifyEventCpuNow(c);
        }
        void operator()(CmdTextureWritten& c) const {
            impl.NotifyTextureWrittenNow(c.serial);
        }
        void operator()(CmdFlushTexture& c) const {
            impl.FlushSubmitBatch();
            impl.FlushTextureNow(c);
        }
        void operator()(CmdTextureHolders& c) const {
            impl.NoteTextureHoldersNow(c.report);
        }
    };
    const auto begin = std::chrono::steady_clock::now();
    std::visit(Visitor{*this}, command);
    const size_t kind = command.index();
    if (kind < command_count.size()) {
        command_count[kind].fetch_add(1, std::memory_order_relaxed);
        command_ns[kind].fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count()),
            std::memory_order_relaxed);
    }
}

void Context::Impl::RetireCompleted() {
    if (retired.empty()) {
        return;
    }
    const u64 done = completed_seq.load(std::memory_order_acquire);
    std::vector<Retired> release;
    size_t kept = 0;
    for (size_t i = 0; i < retired.size(); ++i) {
        if (retired[i].after_seq <= done) {
            release.push_back(std::move(retired[i]));
        } else {
            if (kept != i) {
                retired[kept] = std::move(retired[i]);
            }
            ++kept;
        }
    }
    retired.resize(kept);
    for (Retired& entry : release) {
        entry.release();
    }
}

void Context::Impl::MarkSubmitted(InFlight& flight, VkSemaphore semaphore, u64 value) {
    {
        std::scoped_lock lock{state_mutex};
        flight.semaphore = semaphore;
        flight.value = value;
        flight.submitted = true;
    }
    complete_cv.notify_all();
}

void Context::Impl::FailFlight(InFlight& flight) {
    MarkSubmitted(flight, VK_NULL_HANDLE, 0);
}

void Context::Impl::NoteDeviceLost(const char* where) {
    if (device_lost.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    LOG_CRITICAL(Render_Vulkan,
                 "[nvn] the Vulkan device was lost ({}): the replay completes every later "
                 "submission and present without running it, and the emulation is being stopped",
                 where);
    if (std::this_thread::get_id() == replay_thread.get_id()) {
        DescribeDeviceFault(); // the pools and the raw global table are replay-thread state
    }
    {
        // The owner stops the emulation; the guest threads are not left waiting for work that
        // will never run (the replay completes it as dropped).
        std::scoped_lock lock{failure_mutex};
        if (failure_callback) {
            failure_callback(fmt::format("Vulkan device lost ({})", where));
        }
    }
    if (options.report_device_loss) {
        loss_report_thread = std::thread([this] {
            Common::SetCurrentThreadName("NvnLossReport");
            device.ReportLoss();
        });
    }
}

void Context::Impl::DescribeDeviceFault() {
    // VK_EXT_device_fault is not used in Eglantine: Device::ReportLoss logs the loss.
}

void Context::Impl::NoteIfDeviceLost(const char* where) {
    // The project is built without RTTI (/GR-): a dynamic_cast of the exception faults ("no RTTI
    // data"). Catching the type matches through the exception's throw info instead.
    try {
        throw;
    } catch (const vk::Exception& vulkan) {
        if (vulkan.GetResult() == VK_ERROR_DEVICE_LOST) {
            NoteDeviceLost(where);
        }
    } catch (...) {
    }
}

CounterEngine* Context::Impl::Counters() {
    if (!counters_engine && !counters_failed) {
        try {
            counters_engine = std::make_unique<CounterEngine>(
                device, *allocator,
                [this](const std::function<void(vk::CommandBuffer)>& record) {
                    ImmediateSubmit(record);
                },
                [this](u64 seq) { return WaitCompleted(seq, std::chrono::seconds(2)); },
                options.allow_conditional_rendering, options.in_pass_queries);
            counters_published.store(counters_engine.get(), std::memory_order_release);
        } catch (const std::exception& exception) {
            counters_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] counters unavailable: {}", exception.what());
        }
    }
    return counters_engine.get();
}

MeshIndirectGpu* Context::Impl::MeshIndirect() {
    if (!mesh_indirect && !mesh_indirect_failed) {
        if (!MeshIndirectGpu::Supported(device)) {
            mesh_indirect_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] 1D indirect mesh-task draws need push descriptors "
                                     "(2 or more): they are skipped");
            return nullptr;
        }
        try {
            mesh_indirect = std::make_unique<MeshIndirectGpu>(device);
        } catch (const std::exception& exception) {
            mesh_indirect_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] mesh indirect argument conversion unavailable: {}",
                      exception.what());
        }
    }
    return mesh_indirect.get();
}

QuadIndirectGpu* Context::Impl::QuadIndirect() {
    if (!quad_indirect && !quad_indirect_failed) {
        if (!QuadIndirectGpu::Supported(device)) {
            quad_indirect_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] indirect draws of quads need push descriptors "
                                     "(2 or more): they are skipped");
            return nullptr;
        }
        try {
            quad_indirect = std::make_unique<QuadIndirectGpu>(
                device, CreateHostBuffer(u64{QUAD_INDIRECT_MAX_QUADS} * 6 * sizeof(u32),
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT, MemoryUsage::Upload));
        } catch (const std::exception& exception) {
            quad_indirect_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] quad indirect argument conversion unavailable: {}",
                      exception.what());
        }
    }
    return quad_indirect.get();
}

VertexStageGpu* Context::Impl::VertexStage() {
    if (!vertex_stage && !vertex_stage_failed) {
        if (!VertexStageGpu::Supported(device)) {
            vertex_stage_failed = true;
            LOG_WARNING(Render_Vulkan, "[nvn] vertex staging copies by shader need push "
                                       "descriptors (3 or more): staged with transfer commands");
            return nullptr;
        }
        try {
            vertex_stage = std::make_unique<VertexStageGpu>(device);
        } catch (const std::exception& exception) {
            vertex_stage_failed = true;
            LOG_ERROR(Render_Vulkan, "[nvn] vertex staging copies by shader unavailable: {}",
                      exception.what());
        }
    }
    return vertex_stage.get();
}

WriteBackMerge* Context::Impl::WriteBackMerger() {
    if (!write_back_merge && !write_back_merge_failed) {
        auto merge = std::make_unique<WriteBackMerge>();
        if (merge->Init(device)) {
            merge->SetLineMode(Debug::Get().try_wb_lines != 0, Debug::Get().try_wb_lines);
            if (Debug::Get().wb_monitor) {
                try {
                    mirror_merge_stats = CreateHostBuffer(MIRROR_STAT_SLOTS * 2 * sizeof(u32),
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                          MemoryUsage::Download);
                    std::ranges::fill(mirror_merge_stats.Mapped(), u8{0});
                    mirror_merge_stats.Flush();
                    merge->SetStats(*mirror_merge_stats);
                } catch (const vk::Exception& exception) {
                    LOG_WARNING(Render_Vulkan, "[nvn] write-back monitor: merge counters: {}",
                                exception.what());
                    mirror_merge_stats = {};
                }
            }
            write_back_merge = std::move(merge);
        } else {
            write_back_merge_failed = true;
            LOG_WARNING(Render_Vulkan, "[nvn] write-back word merge unavailable: pool mirror "
                                       "ranges are copied back whole");
        }
    }
    return write_back_merge.get();
}

bool Context::Impl::WaitCompleted(u64 seq, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        {
            std::unique_lock lock{wait_mutex};
            if (wait_cv.wait_for(lock, std::chrono::milliseconds(20), [this, seq] {
                    return completed_seq.load(std::memory_order_acquire) >= seq;
                })) {
                return true;
            }
        }
        {
            std::scoped_lock lock{state_mutex};
            if (stopping) {
                return false;
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
}

void Context::Impl::WaitFlight(u64 seq) {
    // The completion thread drops a flight from `inflight` once it has completed.
    VkSemaphore semaphore = VK_NULL_HANDLE;
    u64 value = 0;
    {
        std::scoped_lock lock{state_mutex};
        const auto it = std::ranges::find_if(
            inflight, [seq](const std::shared_ptr<InFlight>& flight) { return flight->seq == seq; });
        if (it == inflight.end()) {
            return; // retired
        }
        if (!(*it)->submitted) {
            // The passes only wait for flights recorded before the open one. A present flight is
            // submitted before the next present is recorded; a submission that is held back or
            // suspended at an event wait is not submitted yet, and waiting here would wait for the
            // replay thread itself.
            if (logged_messages.insert("present-pass-wait-unsubmitted").second) {
                LOG_ERROR(Render_Vulkan,
                          "[nvn] a presentation pass waited for flight {}, which was never "
                          "submitted; its resources are reused without the wait",
                          seq);
            }
            return;
        }
        semaphore = (*it)->semaphore;
        value = (*it)->value;
    }
    if (semaphore != VK_NULL_HANDLE) {
        WaitTimelines(std::span{&semaphore, 1}, std::span{&value, 1});
    }
}

void Context::Impl::WaitPreviousFlights(u64 seq) {
    std::vector<VkSemaphore> semaphores;
    std::vector<u64> values;
    {
        std::scoped_lock lock{state_mutex};
        for (const std::shared_ptr<InFlight>& flight : inflight) {
            // Held-back and suspended flights are not submitted: nothing of the passes is in them.
            if (flight->seq != seq && flight->submitted && flight->semaphore != VK_NULL_HANDLE) {
                semaphores.push_back(flight->semaphore);
                values.push_back(flight->value);
            }
        }
    }
    if (!semaphores.empty()) {
        WaitTimelines(semaphores, values);
    }
}

void Context::Impl::WaitTimelines(std::span<const VkSemaphore> semaphores,
                                  std::span<const u64> values) {
    const VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = static_cast<u32>(semaphores.size()),
        .pSemaphores = semaphores.data(),
        .pValues = values.data(),
    };
    const auto start = std::chrono::steady_clock::now();
    bool reported = false;
    for (;;) {
        const VkResult result = dld.vkWaitSemaphores(logical, &wait_info, 20'000'000);
        if (result != VK_TIMEOUT) {
            vk::Check(result);
            return;
        }
        {
            std::scoped_lock lock{state_mutex};
            if (stopping) {
                return;
            }
        }
        if (!reported && std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
            reported = true;
            LOG_ERROR(Render_Vulkan,
                      "[nvn] a presentation pass waited 5 s for {} submission(s) to complete",
                      semaphores.size());
        }
    }
}

DrawContext& Context::Impl::GetDrawContext(u32 queue_index) {
    std::unique_ptr<DrawContext>& slot = draw_contexts[queue_index];
    if (!slot) {
        slot = std::make_unique<DrawContext>(*this, queue_index);
    }
    return *slot;
}

QueueState& Context::Impl::GetQueue(u32 index) {
    auto it = queues.find(index);
    if (it == queues.end()) {
        const VkSemaphoreTypeCreateInfo type_ci{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0,
        };
        QueueState state;
        state.timeline = device.GetLogical().CreateSemaphore({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &type_ci,
            .flags = 0,
        });
        it = queues.emplace(index, std::move(state)).first;
    }
    return it->second;
}

VkCommandBuffer Context::Impl::AcquireCommandBuffer() {
    {
        std::scoped_lock lock{free_commands_mutex};
        if (!free_command_buffers.empty()) {
            const VkCommandBuffer buffer = free_command_buffers.back();
            free_command_buffers.pop_back();
            return buffer;
        }
    }
    command_buffer_allocations.push_back(command_pool.Allocate(1));
    return command_buffer_allocations.back()[0];
}

vk::Buffer Context::Impl::AcquireStaging(u64 size) {
    if (size == STAGING_CHUNK) {
        std::scoped_lock lock{staging_cache_mutex};
        for (auto& buffer : staging_cache) {
            if (buffer) {
                stats.staging_chunks_reused.fetch_add(1, std::memory_order_relaxed);
                return std::move(buffer);
            }
        }
    }
    // GPU untile and ASTC decode can read the staged GOBs directly.
    return CreateHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            MemoryUsage::Upload, AllocationKind::Staging);
}

void Context::Impl::CompleteMain() {
    for (;;) {
        std::shared_ptr<InFlight> flight;
        {
            std::unique_lock lock{state_mutex};
            complete_cv.wait(lock, [this] {
                return (!inflight.empty() && inflight.front()->submitted) ||
                       (stopping && inflight.empty());
            });
            if (inflight.empty()) {
                break;
            }
            flight = inflight.front();
        }
        bool gpu_complete = false;
        // After a device loss nothing is waited for: every semaphore reports the loss at once.
        if (flight->semaphore != VK_NULL_HANDLE && !device_lost.load(std::memory_order_acquire)) {
            const VkSemaphore semaphore = flight->semaphore;
            const u64 value = flight->value;
            const VkSemaphoreWaitInfo wait_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = 1,
                .pSemaphores = &semaphore,
                .pValues = &value,
            };
            for (;;) {
                const VkResult result = dld.vkWaitSemaphores(logical, &wait_info, 20'000'000);
                if (result == VK_TIMEOUT) {
                    std::scoped_lock lock{state_mutex};
                    if (stopping) {
                        break; // ~Impl waited for the device to go idle already
                    }
                    continue;
                }
                if (result == VK_ERROR_DEVICE_LOST) {
                    NoteDeviceLost("waiting for a submission to complete");
                } else if (result != VK_SUCCESS) {
                    LOG_ERROR(Render_Vulkan, "[nvn] waiting for seq {} failed: {}", flight->seq,
                              static_cast<int>(result));
                }
                gpu_complete = result == VK_SUCCESS;
                break;
            }
        }
        if (Debug::Timeline()) {
            Debug::TimelineEvent("gpu_done", flight->seq);
        }

        // Mirror write-back: GPU-written ranges go back to guest memory before anyone is told the
        // work is done.
        // (not after a device loss: the downloaded bytes are not the GPU's results)
        if (flight->download && !flight->readbacks.empty() &&
            !device_lost.load(std::memory_order_acquire)) {
            flight->download.Invalidate();
            const std::span<const u8> mapped = flight->download.Mapped();
            VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
            for (const InFlight::Readback& readback : flight->readbacks) {
                const u8* source = mapped.data() + readback.download_offset;
                if (readback.row_count > 0 && readback.row_bytes > 0) {
                    for (u32 row = 0; row < readback.row_count; ++row) {
                        const u64 rel = static_cast<u64>(row) * readback.row_stride;
                        const u64 len = (std::min)(readback.row_bytes, readback.size - rel);
                        if (readback.pool->host) {
                            std::memcpy(readback.pool->host + readback.pool_offset + rel,
                                        source + rel, static_cast<size_t>(len));
                        } else if (memory) {
                            memory->Write(readback.pool->address + readback.pool_offset + rel,
                                          source + rel, static_cast<size_t>(len));
                        }
                    }
                } else if (readback.pool->host) {
                    std::memcpy(readback.pool->host + readback.pool_offset, source, readback.size);
                } else if (memory) {
                    memory->Write(readback.pool->address + readback.pool_offset, source,
                                  readback.size);
                }
                if (mirror_hash_tracked) {
                    std::scoped_lock lock{readback.pool->dirty_mutex};
                    auto& landed = readback.pool->landed_writebacks;
                    if (landed.size() < 1024) {
                        landed.push_back(
                            {readback.pool_offset, readback.pool_offset + readback.size});
                    } else {
                        landed.assign(1, {0, readback.pool->size}); // not taken for long: all
                    }
                }
            }
        }
        if (flight->counters && counters_engine) {
            // Timestamp reports: the GPU ticks become the report's time (+8) before anyone is told
            // the work is done; the counter field was written by the GPU itself.
            counters_engine->Complete(*flight->counters, guest.load(std::memory_order_acquire));
        }
        if (flight->raw_writes) {
            const u64 next_epoch = split_raw_epoch.load(std::memory_order_relaxed) + 1;
            if (flight->raw_writes_unaliased) {
                // Publish unaliased epoch before split_raw_epoch (release) so observing the new
                // split_raw_epoch (acquire) guarantees observing its unaliased update.
                split_raw_epoch_unaliased.store(next_epoch, std::memory_order_release);
            }
            split_raw_epoch.store(next_epoch, std::memory_order_release);
        }
        if (flight->gpu_prof && gpu_profiler) {
            // Timestamps: read after the timeline value was reached, never on the replay thread.
            gpu_profiler->Complete(*flight->gpu_prof);
        }
        if (flight->bc_times) {
            CompleteBcTimes(*flight->bc_times);
        }
        if (flight->capture) {
            flight->capture->buffer.Invalidate();
            WriteCapture(*flight->capture);
            flight->capture.reset();
        }
        if (flight->screenshot) {
            DeliverScreenshot(*flight);
        }
        for (InFlight::Dump& dump : flight->dumps) {
            dump.buffer.Invalidate();
            const std::string summary = Debug::WriteTga(dump.info, dump.buffer.Mapped());
            LOG_INFO(Render_Vulkan, "[nvn] dump {} {}x{} fmt={} aspect={:#x}: {}", dump.info.path,
                     dump.info.width, dump.info.height, static_cast<int>(dump.info.format),
                     static_cast<u32>(dump.info.aspect), summary);
            if (!dump.info.report.empty()) {
                Debug::AppendLine(dump.info.report,
                                  fmt::format("dump {} {}x{} fmt={} aspect={:#x}: {}", dump.info.path,
                                              dump.info.width, dump.info.height,
                                              static_cast<int>(dump.info.format),
                                              static_cast<u32>(dump.info.aspect), summary));
            }
        }
        flight->dumps.clear();
        for (InFlight::TextureDump& dump : flight->texture_dumps) {
            dump.buffer.Invalidate();
            DiagTextures::Complete(dump.info, dump.buffer.Mapped());
        }
        flight->texture_dumps.clear();
        for (InFlight::Watch& watch : flight->watches) {
            watch.buffer.Invalidate();
            const std::span<u8> bytes = watch.buffer.Mapped();
            std::string words;
            for (u32 i = 0; i + 4 <= watch.size; i += 4) {
                u32 value;
                std::memcpy(&value, bytes.data() + i, 4);
                words += fmt::format(" {:x}", value);
            }
            LOG_INFO(Render_Vulkan, "[nvn] watch {}:{}", watch.tag, words);
        }
        flight->watches.clear();
        for (InFlight::PathDiff& pathdiff : flight->pathdiffs) {
            if (pathdiff.buffer) {
                pathdiff.buffer.Invalidate();
                const std::span<u8> bytes = pathdiff.buffer.Mapped();
                for (const InFlight::PathDiff::Copy& copy : pathdiff.copies) {
                    VideoCommon::PathDiff::Range& range = pathdiff.capture.ranges[copy.range];
                    std::vector<u8>& target = copy.after ? range.after : range.before;
                    target.assign(bytes.begin() + copy.offset,
                                  bytes.begin() + copy.offset + copy.size);
                }
            }
            VideoCommon::PathDiff::Write(pathdiff.capture);
        }
        flight->pathdiffs.clear();
        for (const std::shared_ptr<Pool>& pool : flight->dirty_pools) {
            std::scoped_lock lock{pool->dirty_mutex};
            std::erase_if(pool->gpu_dirty,
                          [&](const Pool::Dirty& dirty) { return dirty.seq == flight->seq; });
        }
        if (!flight->command_buffers.empty()) {
            std::scoped_lock lock{free_commands_mutex};
            free_command_buffers.insert(free_command_buffers.end(), flight->command_buffers.begin(),
                                        flight->command_buffers.end());
        }
        if (gpu_complete && !flight->staging.empty() &&
            !device_lost.load(std::memory_order_acquire)) {
            std::scoped_lock lock{staging_cache_mutex};
            auto free = staging_cache.begin();
            for (auto& buffer : flight->staging) {
                if (buffer.Mapped().size() != STAGING_CHUNK) {
                    continue;
                }
                free = std::find_if(free, staging_cache.end(), [](const auto& slot) { return !slot; });
                if (free == staging_cache.end()) {
                    break;
                }
                *free++ = std::move(buffer);
            }
        }
        flight->staging.clear();
        if (!flight->arena.empty()) {
            draw_resources->ReleaseArenas(flight->arena);
        }
        if (!flight->shadow.empty()) {
            draw_resources->ReleaseShadows(flight->shadow);
        }
        flight->download.reset();
        flight->keepalive.clear();
        {
            const u64 current = gpu_completed_seq.load(std::memory_order_relaxed);
            if (flight->seq > current) {
                gpu_completed_seq.store(flight->seq, std::memory_order_release);
            }
        }

        {
            // Held while the callback runs: SetCompletionCallback({}) therefore waits for a running
            // callback, and the owner can be destroyed right after it returns.
            std::scoped_lock callback_lock{callback_mutex};
            bool is_stopping;
            {
                std::scoped_lock lock{state_mutex};
                is_stopping = stopping;
            }
            if (completion_callback && !is_stopping) {
                completion_callback(flight->seq);
            }
        }
        if (Debug::Timeline()) {
            Debug::TimelineEvent("complete", flight->seq);
        }
        {
            std::scoped_lock lock{wait_mutex};
            u64 current = completed_seq.load(std::memory_order_relaxed);
            if (flight->seq > current) {
                completed_seq.store(flight->seq, std::memory_order_release);
            }
        }
        wait_cv.notify_all();
        {
            std::scoped_lock lock{state_mutex};
            inflight.pop_front();
        }
    }
}

Context::Context(const Device& device, Scheduler& scheduler, PresentManager& present_manager,
                 Core::Frontend::EmuWindow& render_window, VideoCore::ShaderNotify& shader_notify,
                 VideoCore::LoadProgressSink& load_progress,
                 std::function<void()> on_queue_submit)
    : impl{std::make_unique<Impl>(
          device, scheduler.submit_mutex,
          std::make_unique<PresentManagerTarget>(device, present_manager, render_window),
          ContextOptions{
              .in_pass_queries = Settings::values.nvn_in_pass_queries.GetValue(),
              .shader_notify = &shader_notify,
              .load_progress = &load_progress,
              .on_queue_submit = std::move(on_queue_submit)},
          std::make_unique<PresentFrameGen>())} {
}

Context::Context(const Device& device, std::mutex& submit_mutex,
                 std::unique_ptr<PresentTarget> present, ContextOptions options)
    : impl{std::make_unique<Impl>(device, submit_mutex, std::move(present), options)} {}

Context::~Context() = default;

void Context::CreatePool(const VC::PoolDesc& desc) {
    impl->Enqueue(CmdCreatePool{desc}, nullptr);
}

void Context::CreateTexture(const VC::TextureDesc& desc) {
    impl->Enqueue(CmdCreateTexture{desc}, nullptr);
}

void Context::CreateSampler(const VC::SamplerDesc& desc) {
    impl->Enqueue(CmdCreateSampler{desc}, nullptr);
}

void Context::CreateProgram(const VC::ProgramDesc& desc) {
    impl->Enqueue(CmdCreateProgram{desc}, nullptr);
}

void Context::Destroy(u64 serial) {
    impl->Enqueue(CmdDestroy{serial}, nullptr);
}

void Context::Submit(VC::Submission&& submission) {
    if (Debug::Timeline()) {
        size_t words = 0;
        for (const std::vector<u64>& list : submission.lists) {
            words += list.size();
        }
        Debug::TimelineEvent("enq_submit", submission.seq,
                             fmt::format("q={} words={}", submission.queue, words));
    }
    auto flight = std::make_shared<InFlight>();
    flight->seq = submission.seq;
    impl->Enqueue(CmdSubmit{std::move(submission), flight}, flight);
}

void Context::Present(const VC::PresentRequest& request) {
    if (Debug::Timeline()) {
        Debug::TimelineEvent("enq_present", request.seq);
    }
    auto flight = std::make_shared<InFlight>();
    flight->seq = request.seq;
    impl->Enqueue(CmdPresent{request, flight}, flight);
}

void Context::MapVirtual(u64 pool_serial, std::vector<VC::VirtualMapping>&& mappings) {
    impl->Enqueue(CmdMapVirtual{pool_serial, std::move(mappings)}, nullptr);
}

void Context::NotifyCpuWrite(u64 guest_address, u64 size) {
    impl->Enqueue(CmdCpuWrite{guest_address, size}, nullptr);
}

void Context::NotifyHostWrite(u64 guest_address, u64 size) {
    impl->Enqueue(CmdCpuWrite{guest_address, size, true}, nullptr);
}

u64 Context::GpuWriteMarker() {
    if (impl->invalidate_wait_ms == 0 || impl->device_lost.load(std::memory_order_acquire)) {
        return 0;
    }
    // Marker 0 means "off": the marker is the count of enqueued commands plus one.
    return impl->commands_enqueued.load(std::memory_order_acquire) + 1;
}

VC::GpuWriteState Context::PollGpuWrites(u64 marker, u64 guest_address, u64 size, u64 max_seq) {
    VC::GpuWriteState state;
    state.wait_limit_ms = impl->invalidate_wait_ms;
    state.replay_reached = impl->commands_dispatched.load(std::memory_order_acquire) + 1 >= marker;
    if (state.replay_reached) {
        state.writer_seq = impl->gpu_writes.Latest(
            guest_address, guest_address + size, max_seq,
            impl->completed_seq.load(std::memory_order_acquire));
    }
    return state;
}

void Context::NotifyTextureWritten(u64 texture_serial) {
    impl->Enqueue(CmdTextureWritten{texture_serial}, nullptr);
}

void Context::NotifyEventCpu(u64 address, VC::EventCpuChange change, u32 value) {
    if (Debug::Timeline()) {
        Debug::TimelineEvent("enq_eventcpu", 0,
                             fmt::format("addr={:#x} change={} value={}", address,
                                         static_cast<u32>(change), value));
    }
    impl->Enqueue(CmdEventCpu{address, change, value}, nullptr);
}

bool Context::FlushTextureToGuest(u64 texture_serial) {
    if (impl->device_lost.load(std::memory_order_acquire)) {
        return false; // nothing can be read back from a lost device
    }
    auto done = std::make_shared<std::promise<bool>>();
    std::future<bool> result = done->get_future();
    impl->Enqueue(CmdFlushTexture{texture_serial, done}, nullptr);
    // The readback waits for the GPU (a frame or two): never forever, and not for a replay that
    // is stopping or found the device lost (its command may have been dropped).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (result.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
        bool stopping;
        {
            std::scoped_lock lock{impl->state_mutex};
            stopping = impl->stopping;
        }
        if (stopping || impl->device_lost.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
    return result.get();
}

bool Context::Wait(u64 seq, std::chrono::nanoseconds timeout) {
    if (Debug::Timeline()) {
        Debug::TimelineEvent("wait_begin", seq);
    }
    std::unique_lock lock{impl->wait_mutex};
    const bool done = impl->wait_cv.wait_for(lock, timeout, [this, seq] {
        return impl->completed_seq.load(std::memory_order_acquire) >= seq;
    });
    if (Debug::Timeline()) {
        Debug::TimelineEvent(done ? "wait_end" : "wait_timeout", seq);
    }
    return done;
}

void Context::SetCompletionCallback(std::function<void(u64 seq)> callback) {
    std::scoped_lock lock{impl->callback_mutex};
    impl->completion_callback = std::move(callback);
}

void Context::SetFailureCallback(std::function<void(std::string reason)> callback) {
    std::scoped_lock lock{impl->failure_mutex};
    impl->failure_callback = std::move(callback);
}

void Context::SetDeviceModes(u32 window_origin, u32 depth_mode) {
    impl->device_window_origin.store(window_origin, std::memory_order_relaxed);
    impl->device_depth_mode.store(depth_mode, std::memory_order_relaxed);
}

void Context::SetTitleId(u64 title_id) {
    impl->Enqueue(CmdSetTitle{title_id}, nullptr);
}

void Context::SetGuestMemory(VC::GuestMemory* memory) {
    impl->guest.store(memory, std::memory_order_release);
}

void Context::SetPoolReconciler(VC::PoolReconciler* reconciler) {
    impl->reconciler.store(reconciler, std::memory_order_release);
}

ContextStats Context::Stats() const {
    const auto& s = impl->stats;
    constexpr auto relaxed = std::memory_order_relaxed;
    ContextStats stats{
        .pools_imported = s.pools_imported.load(relaxed),
        .pools_mirrored = s.pools_mirrored.load(relaxed),
        .pools_mirrored_discontiguous = s.pools_mirrored_discontiguous.load(relaxed),
        .pools_mirrored_no_import = s.pools_mirrored_no_import.load(relaxed),
        .pools_split_imported = s.pools_split_imported.load(relaxed),
        .split_runs = s.split_runs.load(relaxed),
        .mirror_uploads = s.mirror_uploads.load(relaxed),
        .mirror_upload_runs = s.mirror_upload_runs.load(relaxed),
        .mirror_upload_bytes = s.mirror_upload_bytes.load(relaxed),
        .mirror_ranges_clean = s.mirror_ranges_clean.load(relaxed),
        .mirror_pages_hashed = s.mirror_pages_hashed.load(relaxed),
        .mirror_pages_trusted = s.mirror_pages_trusted.load(relaxed),
        .staging_chunks_reused = s.staging_chunks_reused.load(relaxed),
        .mirror_pages_gpu_owned = s.mirror_pages_gpu_owned.load(relaxed),
        .vram_mirrors = s.vram_mirrors.load(relaxed),
        .vram_mirror_refused = s.vram_mirror_refused.load(relaxed),
        .vram_copy_in_runs = s.vram_copy_in_runs.load(relaxed),
        .vram_copy_in_bytes = s.vram_copy_in_bytes.load(relaxed),
        .vram_bindings = s.vram_bindings.load(relaxed),
        .vram_bindings_host = s.vram_bindings_host.load(relaxed),
        .vram_write_ranges = s.vram_write_ranges.load(relaxed),
        .vram_write_bytes = s.vram_write_bytes.load(relaxed),
        .vram_write_backs_plain = s.vram_write_backs_plain.load(relaxed),
        .vram_invalidate_all = s.vram_invalidate_all.load(relaxed),
        .vram_mirrors_read_only = s.vram_mirrors_read_only.load(relaxed),
        .vram_mirrors_promoted = s.vram_mirrors_promoted.load(relaxed),
        .vram_invalidate_read_only = s.vram_invalidate_read_only.load(relaxed),
        .vram_ro_written_through = s.vram_ro_written_through.load(relaxed),
        .vram_raw_alias_rebuilds = s.vram_raw_alias_rebuilds.load(relaxed),
        .vram_vertex_ranges = s.vram_vertex_ranges.load(relaxed),
        .vram_vertex_budget_skips = s.vram_vertex_budget_skips.load(relaxed),
        .vram_flush_pages_kept = s.vram_flush_pages_kept.load(relaxed),
        .vram_flush_pages_taken = s.vram_flush_pages_taken.load(relaxed),
        .pools_device_local = s.pools_device_local.load(relaxed),
        .pools_unsupported = s.pools_unsupported.load(relaxed),
        .host_import_failures = s.host_import_failures.load(relaxed),
        .textures_created = s.textures_created.load(relaxed),
        .textures_failed = s.textures_failed.load(relaxed),
        .textures_transcode_unsupported = s.textures_transcode_unsupported.load(relaxed),
        .alias_unsupported = s.alias_unsupported.load(relaxed),
        .alias_events = s.alias_events.load(relaxed),
        .storages_created = s.storages_created.load(relaxed),
        .storages_reused = s.storages_reused.load(relaxed),
        .storages_evicted = s.storages_evicted.load(relaxed),
        .storages_superseded_evicted = s.storages_superseded_evicted.load(relaxed),
        .storages_aged_evicted = s.storages_aged_evicted.load(relaxed),
        .storages_pressure_evicted = s.storages_pressure_evicted.load(relaxed),
        .storage_cache_bytes = s.storage_cache_bytes.load(relaxed),
        .alias_transfers = s.alias_transfers.load(relaxed),
        .alias_byte_transfers = s.alias_byte_transfers.load(relaxed),
        .alias_overlaps_seen = s.alias_overlaps_seen.load(relaxed),
        .uploads = s.uploads.load(relaxed),
        .upload_bytes = s.upload_bytes.load(relaxed),
        .cpu_checks = s.cpu_checks.load(relaxed),
        .cpu_changes = s.cpu_changes.load(relaxed),
        .host_write_uploads = s.host_write_uploads.load(relaxed),
        .decode_ns = s.decode_ns.load(relaxed),
        .decode_bytes = s.decode_bytes.load(relaxed),
        .hash_ns = s.hash_ns.load(relaxed),
        .hash_bytes = s.hash_bytes.load(relaxed),
        .depth_upload_unsupported = s.depth_upload_unsupported.load(relaxed),
        .virtual_maps = s.virtual_maps.load(relaxed),
        .virtual_unresolved = s.virtual_unresolved.load(relaxed),
        .virtual_sparse_binds = s.virtual_sparse_binds.load(relaxed),
        .sparse_write_pool_checks = s.sparse_write_pool_checks.load(relaxed),
        .texels_flushed = s.texels_flushed.load(relaxed),
        .ops_unsupported = s.ops_unsupported.load(relaxed),
        .ops_failed = s.ops_failed.load(relaxed),
        .submissions = s.submissions.load(relaxed),
        .presents = s.presents.load(relaxed),
        .held_back_submissions = s.held_back_submissions.load(relaxed),
        .parked_timeouts = s.parked_timeouts.load(relaxed),
        .event_signals = s.event_signals.load(relaxed),
        .event_waits = s.event_waits.load(relaxed),
        .event_waits_blocked = s.event_waits_blocked.load(relaxed),
        .waits_unsatisfied = s.waits_unsatisfied.load(relaxed),
    };
    stats.device_lost = impl->device_lost.load(relaxed) ? 1 : 0;
    stats.dropped_after_device_loss = s.dropped_after_device_loss.load(relaxed);
    stats.indirect_draws = s.indirect_draws.load(relaxed);
    stats.indirect_counted_draws = s.indirect_counted_draws.load(relaxed);
    stats.indirect_unsupported = s.indirect_unsupported.load(relaxed);
    stats.mesh_task_draws = s.mesh_task_draws.load(relaxed);
    stats.raw_global_draws = s.raw_global_draws.load(relaxed);
    stats.raw_global_pools_skipped = s.raw_global_pools_skipped.load(relaxed);
    stats.raw_global_ranges_dropped = s.raw_global_ranges_dropped.load(relaxed);
    stats.raw_global_rebuilds = s.raw_global_rebuilds.load(relaxed);
    if (impl->draw_resources) {
        const DrawCounters& draw = impl->draw_resources->counters;
        stats.driver_bank_uploads = draw.driver_bank_uploads.load(relaxed);
        stats.handle_bank_uploads = draw.handle_bank_uploads.load(relaxed);
        stats.pipeline_state_reuses = draw.pipeline_state_reuses.load(relaxed);
        stats.shadow_creates = draw.shadow_creates.load(relaxed);
        stats.shadow_redirects = draw.shadow_redirects.load(relaxed);
        stats.shadow_flush_segment = draw.shadow_flush_segment.load(relaxed);
        stats.shadow_flush_conflict = draw.shadow_flush_conflict.load(relaxed);
        stats.shadow_flush_raw = draw.shadow_flush_raw.load(relaxed);
        stats.shadow_bytes_in = draw.shadow_bytes_in.load(relaxed);
        stats.shadow_bytes_out = draw.shadow_bytes_out.load(relaxed);
        stats.shadow_skipped_alias = draw.shadow_skipped_alias.load(relaxed);
        stats.shadow_skipped_open_pass = draw.shadow_skipped_open_pass.load(relaxed);
        stats.shadow_skipped_no_room = draw.shadow_skipped_no_room.load(relaxed);
    }
    if (CounterEngine* engine = impl->counters_published.load(std::memory_order_acquire)) {
        const CounterEngineStats& c = engine->Stats();
        stats.render_conditions_gpu = c.conditions_gpu.load(relaxed);
        stats.render_conditions_prefetched = c.conditions_prefetched.load(relaxed);
        stats.render_conditions_cpu = c.conditions_cpu.load(relaxed);
        stats.render_conditions_false = c.conditions_false.load(relaxed);
        stats.render_conditions_unresolved = c.conditions_unresolved.load(relaxed);
        stats.render_ops_gated = c.ops_gated.load(relaxed);
        stats.render_ops_skipped = c.ops_skipped.load(relaxed);
        stats.counter_reports_timestamp = c.reports_timestamp.load(relaxed);
        stats.counter_reports_samples = c.reports_samples.load(relaxed);
        stats.counter_reports_value = c.reports_value.load(relaxed);
        stats.counter_reports_unsupported = c.reports_unsupported_type.load(relaxed);
        stats.counter_reports_samples_uncounted = c.reports_samples_uncounted.load(relaxed);
        stats.counter_occlusion_folds = c.occlusion_folds.load(relaxed);
    }
    for (size_t r = 0; r < VTXCOV_REASONS; ++r) {
        stats.vtxcov_binds[r] =
            impl->vtxcov_binds[0][r].load(relaxed) + impl->vtxcov_binds[1][r].load(relaxed);
    }
    return stats;
}

VtxCovCounts Context::Impl::VtxCovSnapshot() const {
    VtxCovCounts counts;
    for (size_t kind = 0; kind < 2; ++kind) {
        for (size_t r = 0; r < VTXCOV_REASONS; ++r) {
            counts.binds[kind][r] = vtxcov_binds[kind][r].load(std::memory_order_relaxed);
            counts.bytes[kind][r] = vtxcov_bytes[kind][r].load(std::memory_order_relaxed);
        }
    }
    return counts;
}

void Context::Impl::VtxCovOnPresent() {
    const auto now = std::chrono::steady_clock::now();
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    const u64 copy_in = stats.vram_copy_in_bytes.load(std::memory_order_relaxed);
    const u64 staged =
        draw_resources ? draw_resources->counters.vertex_stage_bytes.load(std::memory_order_relaxed)
                       : 0;
    if (vtxcov_logged_at == std::chrono::steady_clock::time_point{}) {
        vtxcov_logged_at = now;
        vtxcov_logged = VtxCovSnapshot();
        vtxcov_logged_copy_in = copy_in;
        vtxcov_logged_staged = staged;
        vtxcov_logged_presents = presents;
        return;
    }
    if (now - vtxcov_logged_at < std::chrono::seconds{10} || presents == vtxcov_logged_presents) {
        return;
    }
    const VtxCovCounts total = VtxCovSnapshot();
    VtxCovCounts delta;
    for (size_t kind = 0; kind < 2; ++kind) {
        for (size_t r = 0; r < VTXCOV_REASONS; ++r) {
            delta.binds[kind][r] = total.binds[kind][r] - vtxcov_logged.binds[kind][r];
            delta.bytes[kind][r] = total.bytes[kind][r] - vtxcov_logged.bytes[kind][r];
        }
    }
    LOG_INFO(Render_Vulkan, "[nvn] vtxcov: {}",
             FormatVtxCov(delta, presents - vtxcov_logged_presents, copy_in - vtxcov_logged_copy_in,
                          staged - vtxcov_logged_staged, vtxcov));
    vtxcov_logged_at = now;
    vtxcov_logged = total;
    vtxcov_logged_copy_in = copy_in;
    vtxcov_logged_staged = staged;
    vtxcov_logged_presents = presents;
}

void Context::Impl::WriteCapture(const InFlight::Capture& capture) const {
    // The nx2:vi2 path saves 1280x720 BGRA with a top-down 32-bit TGA header: the replay read the
    // picture back at that size (Recorder::RecordPictureReadback); only its encoding is converted.
    const Picture& picture = capture.picture;
    std::vector<u8> pixels(static_cast<size_t>(picture.width) * picture.height * 4);
    if (!ConvertPicture(capture.buffer.Mapped(), picture, false, pixels)) {
        LOG_ERROR(Render_Vulkan, "[nvn] capture: the read-back picture {}x{} does not fit",
                  picture.width, picture.height);
        return;
    }
    const bool writes_files = capture.to_environment_directory || !capture.request_directory.empty();
    if (!writes_files) {
        return;
    }
    std::array<u8, 18> header{};
    header[2] = 2; // uncompressed true colour
    header[12] = static_cast<u8>(picture.width);
    header[13] = static_cast<u8>(picture.width >> 8);
    header[14] = static_cast<u8>(picture.height);
    header[15] = static_cast<u8>(picture.height >> 8);
    header[16] = 32;
    header[17] = 0x28; // rows top to bottom
    const auto write = [&](const std::string& directory) {
        const std::string path = fmt::format("{}/nx2-frame-{:05}.tga", directory, capture.index);
        if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
            std::fwrite(header.data(), 1, header.size(), file);
            std::fwrite(pixels.data(), 1, pixels.size(), file);
            std::fclose(file);
            return true;
        }
        return false;
    };
    if (capture.to_environment_directory) {
        write(capture_directory);
    }
    if (!capture.request_directory.empty()) {
        std::error_code error;
        std::filesystem::create_directories(capture.request_directory, error);
        const std::string directory = capture.request_directory.string();
        const bool saved = write(directory);
        LOG_INFO(Render_Vulkan, "[diag] frame: nvn present {} {} in {}", capture.index,
                 saved ? "saved" : "could not be saved", directory);
    }
}

void Context::Impl::DeliverScreenshot(InFlight& flight) {
    InFlight::Screenshot shot = std::move(*flight.screenshot);
    flight.screenshot.reset();
    if (flight.semaphore == VK_NULL_HANDLE) {
        // The frame never reached the GPU (the present failed, or the context is shutting down):
        // the request stays pending for the next present.
        shot.settings->ReleaseScreenshotClaim();
        return;
    }
    shot.buffer.Invalidate();
    const std::span<const u8> mapped = shot.buffer.Mapped();
    const size_t picture_bytes = static_cast<size_t>(shot.picture.width) * shot.picture.height *
                                 BytesPerPixel(shot.picture.format);
    std::vector<u8> pixels(mapped.begin(),
                           mapped.begin() + static_cast<std::ptrdiff_t>(
                                                std::min(picture_bytes, mapped.size())));
    shot.buffer = {};
    // At most one screenshot is in flight (RendererSettings::ClaimScreenshot), so the previous
    // thread has answered its request by now.
    if (screenshot_thread.joinable()) {
        screenshot_thread.join();
    }
    screenshot_thread = std::thread([shot = std::move(shot), pixels = std::move(pixels)] {
        Common::SetCurrentThreadName("NvnScreenshot");
        const size_t output_bytes = static_cast<size_t>(shot.picture.width) * shot.picture.height * 4;
        if (!ConvertPicture(pixels, shot.picture, true, {shot.destination, output_bytes})) {
            LOG_ERROR(Render_Vulkan, "[nvn] screenshot: the read-back picture {}x{} does not fit",
                      shot.picture.width, shot.picture.height);
        }
        shot.complete(false); // rows top first
        shot.settings->FinishScreenshot();
    });
}

void Context::SetScreenshotRequests(VideoCore::RendererSettings& settings) {
    impl->screenshot_requests.store(&settings, std::memory_order_release);
}

void Context::NoteTextureHolders(VC::TextureHolderReport&& report) {
    impl->Enqueue(CmdTextureHolders{std::move(report)}, nullptr);
}

std::string Context::DescribeMemory() const {
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    std::string text;
    {
        // Private bytes: the commit charge of the process (host memory Vulkan allocations count,
        // device-local ones do not); the private working set: what of it is resident.
        u64 private_bytes = 0;
        u64 private_ws = 0;
        u64 shared_commit = 0;
#ifdef _WIN32
        // PROCESS_MEMORY_COUNTERS_EX2 (Windows 10 21H1+, newer than the SDK target the build
        // selects), declared here with its layout.
        struct CountersEx2 {
            DWORD cb;
            DWORD PageFaultCount;
            SIZE_T PeakWorkingSetSize;
            SIZE_T WorkingSetSize;
            SIZE_T QuotaPeakPagedPoolUsage;
            SIZE_T QuotaPagedPoolUsage;
            SIZE_T QuotaPeakNonPagedPoolUsage;
            SIZE_T QuotaNonPagedPoolUsage;
            SIZE_T PagefileUsage;
            SIZE_T PeakPagefileUsage;
            SIZE_T PrivateUsage;
            SIZE_T PrivateWorkingSetSize;
            ULONG64 SharedCommitUsage;
        };
        CountersEx2 counters{};
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                 sizeof(counters))) {
            private_bytes = counters.PrivateUsage;
            private_ws = counters.PrivateWorkingSetSize;
            shared_commit = counters.SharedCommitUsage;
        } else {
            PROCESS_MEMORY_COUNTERS_EX older{};
            if (GetProcessMemoryInfo(GetCurrentProcess(),
                                     reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&older),
                                     sizeof(older))) {
                private_bytes = older.PrivateUsage;
            }
        }
#endif
        text += fmt::format("process private {} MiB, private working set {} MiB, shared commit "
                            "{} MiB",
                            mib(private_bytes), mib(private_ws), mib(shared_commit));
    }
    {
        // Every heap: the driver's budget and the process's usage (VK_EXT_memory_budget), and
        // the blocks and allocations of the replay's own allocator in it.
        const VkPhysicalDeviceMemoryProperties* properties{};
        vmaGetMemoryProperties(impl->vma, &properties);
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(impl->vma, budgets.data());
        text += "; heaps budget/usage, replay blocks/allocations MiB:";
        for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
            const bool device_local =
                (properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            const VmaBudget& b = budgets[heap];
            text += fmt::format(" #{} {} {}/{}, {}/{};", heap, device_local ? "device" : "host",
                                mib(b.budget), mib(b.usage), mib(b.statistics.blockBytes),
                                mib(b.statistics.allocationBytes));
        }
    }
    {
        const AllocationSpills& s = impl->spills;
        text += " spilled to host types n/MiB:";
        for (size_t kind = 0; kind < AllocationSpills::KINDS; ++kind) {
            text += fmt::format(" {} {}/{}", AllocationKindName(static_cast<AllocationKind>(kind)),
                                s.count[kind].load(std::memory_order_relaxed),
                                mib(s.bytes[kind].load(std::memory_order_relaxed)));
        }
        text += fmt::format("; strict {}: over budget {}/{}, host fallbacks {}/{}",
                            impl->vma_device_strict ? "on" : "off",
                            s.over_budget_count.load(std::memory_order_relaxed),
                            mib(s.over_budget_bytes.load(std::memory_order_relaxed)),
                            s.host_fallback_count.load(std::memory_order_relaxed),
                            mib(s.host_fallback_bytes.load(std::memory_order_relaxed)));
    }
    if (impl->holder_report_seen.load(std::memory_order_acquire)) {
        // From the recorder's previous report (one status line old).
        const auto holder = [&](VC::TextureHolder kind) {
            const size_t i = static_cast<size_t>(kind);
            return fmt::format("{} ({})", mib(impl->holder_storage_bytes[i].load()),
                               impl->holder_storages[i].load());
        };
        text += fmt::format("; texture storages by strongest holder MiB (count): game {}, "
                            "retiring {}, guest lists {}, synthetic lists {}, re-initialised {}",
                            holder(VC::TextureHolder::Game), holder(VC::TextureHolder::Retiring),
                            holder(VC::TextureHolder::GuestList),
                            holder(VC::TextureHolder::SyntheticList),
                            holder(VC::TextureHolder::Reinitialized));
    }
    text += fmt::format("; storage cache {} MiB; import chunks {} ({} MiB)",
                        mib(impl->stats.storage_cache_bytes.load(std::memory_order_relaxed)),
                        impl->import_chunks_live.load(std::memory_order_relaxed),
                        mib(impl->import_chunks_live_bytes.load(std::memory_order_relaxed)));
    if (impl->draw_resources) {
        std::scoped_lock lock{impl->draw_resources->cache_mutex};
        if (const auto& cache = impl->draw_resources->cache_instance) {
            const PipelineCacheStats p = cache->Stats();
            text += fmt::format(
                "; pipelines graphics {} compute {}, lowered programs {}, shader refreshes {}",
                p.graphics_pipelines, p.compute_pipelines, p.lowered_programs,
                cache->ShaderGeneration());
        }
    }
    return text;
}

std::string Context::DescribeState() const {
    std::string text = fmt::format("replay stage={} seq={} completed={}",
                                   impl->replay_stage.load(), impl->replay_seq.load(),
                                   impl->completed_seq.load());
    std::scoped_lock lock{impl->state_mutex};
    text += fmt::format(" queued={} inflight={} streams={}", impl->commands.size(),
                        impl->inflight.size(), impl->streams_pending.load());
    text += fmt::format(" events(signals/waits/blocked/released)={}/{}/{}/{}",
                        impl->stats.event_signals.load(), impl->stats.event_waits.load(),
                        impl->stats.event_waits_blocked.load(),
                        impl->stats.waits_unsatisfied.load());
    if (!impl->inflight.empty()) {
        const InFlight& front = *impl->inflight.front();
        text += fmt::format(" front(seq={} submitted={} sem={} value={})", front.seq,
                            front.submitted, front.semaphore != VK_NULL_HANDLE, front.value);
    }
    static constexpr const char* names[] = {"pool",    "texture", "sampler",  "program",
                                            "destroy", "submit",  "present",  "title",
                                            "mapvirt", "cpuwrite", "texwrit", "flushtex",
                                            "eventcpu", "holders"};
    text += fmt::format(" pool_ms(pointer/import/buffer)={}/{}/{} pool_MB={}",
                        impl->pool_ns_pointer.load() / 1'000'000,
                        impl->pool_ns_import.load() / 1'000'000,
                        impl->pool_ns_buffer.load() / 1'000'000,
                        impl->pool_bytes.load() >> 20);
    text += fmt::format(" imports(n/ms/MB)={}/{}/{} pools_in_chunks={} import_fallbacks={} "
                        "backed_in_background={} raw_rebuilds={}",
                        impl->chunk_imports.load(), impl->chunk_import_ns.load() / 1'000'000,
                        impl->chunk_import_bytes.load() >> 20, impl->pools_in_chunks.load(),
                        impl->pool_import_fallbacks.load(), impl->pools_backed_background.load(),
                        impl->stats.raw_global_rebuilds.load());
    // What the process holds, as the commit charge sees it: host commit (private bytes), the
    // device-local heap usage the driver reports (a heap over its budget spills into host commit),
    // and the sparse texture pages.
    {
        u64 private_bytes = 0;
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX counters{};
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                 sizeof(counters))) {
            private_bytes = counters.PrivateUsage;
        }
#endif
        text += fmt::format(" commit_MB={} device(usage/budget/slack)_MB={}/{}/{} sparse_pages_MB={}",
                            private_bytes >> 20, impl->device.GetDeviceMemoryUsage() >> 20,
                            impl->device.GetDeviceMemoryBudget() >> 20,
                            impl->device.GetDeviceMemorySlack() >> 20,
                            impl->sparse_pages ? impl->sparse_pages->UsedBytes() >> 20 : 0);
    }
    text += " cmd(n/ms):";
    for (size_t i = 0; i < std::size(names); ++i) {
        text += fmt::format(" {}={}/{}", names[i], impl->command_count[i].load(),
                            impl->command_ns[i].load() / 1'000'000);
    }
    {
        const ContextStats c = Stats();
        text += fmt::format(" cond(gpu/cpu/false/gated/skipped/ahead)={}/{}/{}/{}/{}/{} "
                            "reports(ts/samples/value/unsup)={}/{}/{}/{} samples_uncounted={} "
                            "folds={} "
                            "indirect(draws/counted/unsup)={}/{}/{} mesh_task_draws={}",
                            c.render_conditions_gpu, c.render_conditions_cpu,
                            c.render_conditions_false, c.render_ops_gated, c.render_ops_skipped,
                            c.render_conditions_prefetched,
                            c.counter_reports_timestamp, c.counter_reports_samples,
                            c.counter_reports_value, c.counter_reports_unsupported,
                            c.counter_reports_samples_uncounted, c.counter_occlusion_folds,
                            c.indirect_draws, c.indirect_counted_draws, c.indirect_unsupported,
                            c.mesh_task_draws);
        text += fmt::format(" pools(imported/mirrored: runs,no_import)={}/{}: {},{}",
                            c.pools_imported, c.pools_mirrored, c.pools_mirrored_discontiguous,
                            c.pools_mirrored_no_import);
        text += fmt::format(
            " mirror(uploads/copies/MiB/clean/pages_hashed/pages_trusted)={}/{}/{}/{}/{}/{}",
            c.mirror_uploads, c.mirror_upload_runs, c.mirror_upload_bytes >> 20,
            c.mirror_ranges_clean, c.mirror_pages_hashed, c.mirror_pages_trusted);
        text += fmt::format(" mirror_gpu_owned_pages={} staging_chunks_reused={}",
                            c.mirror_pages_gpu_owned, c.staging_chunks_reused);
        if (impl->draw_resources) {
            const DrawCounters& draw = impl->draw_resources->counters;
            const auto breaks = [&](BreakCause cause) {
                return draw.break_causes[static_cast<u32>(cause)].load();
            };
            text += fmt::format(
                " pass_breaks(total/mirror/indirect_mirror)={}/{}/{} presents={}"
                " counter_breaks(report/reset/report_read/condition)={}/{}/{}/{}",
                draw.pass_breaks.load(), breaks(BreakCause::MirrorUpload),
                breaks(BreakCause::IndirectUpload), c.presents, breaks(BreakCause::ReportCounter),
                breaks(BreakCause::ResetCounter), breaks(BreakCause::ReportRead),
                breaks(BreakCause::RenderCondition));
        }
        text += fmt::format(" raw_global(draws/rebuilds/skipped/dropped)={}/{}/{}/{}",
                            c.raw_global_draws, c.raw_global_rebuilds, c.raw_global_pools_skipped,
                            c.raw_global_ranges_dropped);
        text += fmt::format(" shadows(created/redirected/flush seg,conflict,raw/skipped alias,"
                            "pass,room/MB in,out)={}/{}/{},{},{}/{},{},{}/{},{}",
                            c.shadow_creates, c.shadow_redirects, c.shadow_flush_segment,
                            c.shadow_flush_conflict, c.shadow_flush_raw, c.shadow_skipped_alias,
                            c.shadow_skipped_open_pass, c.shadow_skipped_no_room,
                            c.shadow_bytes_in >> 20, c.shadow_bytes_out >> 20);
        if (c.vram_mirrors != 0 || c.vram_mirror_refused != 0) {
            text += fmt::format(" pool_mirrors(n/refused/MiB/copy_in runs,MiB/served,host/"
                                "writes n,MiB,plain/invalidate_all/flushed pages kept,taken)="
                                "{}/{}/{}/{},{}/{},{}/{},{},{}/{}/{},{}",
                                c.vram_mirrors, c.vram_mirror_refused,
                                impl->vram_mirror_bytes >> 20, c.vram_copy_in_runs,
                                c.vram_copy_in_bytes >> 20, c.vram_bindings, c.vram_bindings_host,
                                c.vram_write_ranges, c.vram_write_bytes >> 20,
                                c.vram_write_backs_plain, c.vram_invalidate_all,
                                c.vram_flush_pages_kept, c.vram_flush_pages_taken);
            text += fmt::format(" ro_mirrors(n/promoted/raw invalidations/vertex+index ranges "
                                "served/copy-bound skips)={}/{}/{}/{}/{}",
                                c.vram_mirrors_read_only, c.vram_mirrors_promoted,
                                c.vram_invalidate_read_only, c.vram_vertex_ranges,
                                c.vram_vertex_budget_skips);
            if (c.vram_ro_written_through != 0 || c.vram_raw_alias_rebuilds != 0) {
                text += fmt::format(" ro_write_through(draws)/alias_tables={}/{}",
                                    c.vram_ro_written_through, c.vram_raw_alias_rebuilds);
            }
        }
    }
    const auto& s = impl->stats;
    text += fmt::format(
        " mem(storages new/reuse/evict={}/{}/{} recycled={} split_import(pools/runs)={}/{} uploads={} MB={} "
        "cpu check/trusted/pages/change={}/{}/{}/{} pages_rehashed={} "
        "hash_ms={} hash_MB={} (compressed {} dynamic {}) decode_ms={} decode_MB={} alias_xfer={} "
        "alias_seen={} virt={} astc gpu/fallback/bc={}/{}/{} untile gpu/direct/fallback={}/{}/{} "
        "untile_gpu_MB={} storage_cache(MB/superseded/aged/pressure_evicted)={}/{}/{}/{})",
        s.storages_created.load(), s.storages_reused.load(), s.storages_evicted.load(),
        s.storages_recycled.load(), s.pools_split_imported.load(), s.split_runs.load(),
        s.uploads.load(), s.upload_bytes.load() >> 20, s.cpu_checks.load(),
        s.cpu_checks_trusted.load(), s.cpu_checks_pages.load(), s.cpu_changes.load(),
        s.cpu_pages_rehashed.load(), s.hash_ns.load() / 1'000'000, s.hash_bytes.load() >> 20,
        s.hash_bytes_compressed.load() >> 20, s.hash_bytes_dynamic.load() >> 20,
        s.decode_ns.load() / 1'000'000, s.decode_bytes.load() >> 20, s.alias_transfers.load(),
        s.alias_overlaps_seen.load(), s.virtual_maps.load(), s.astc_gpu_uploads.load(),
        s.astc_cpu_fallbacks.load(), s.astc_bc_encodes.load(), s.untile_gpu_uploads.load(),
        s.untile_gpu_direct.load(),
        s.untile_cpu_fallbacks.load(), s.untile_gpu_bytes.load() >> 20,
        s.storage_cache_bytes.load() >> 20, s.storages_superseded_evicted.load(),
        s.storages_aged_evicted.load(), s.storages_pressure_evicted.load());
    if (const LineGuardStats& g = *impl->line_guard_stats;
        g.line_bytes.load() + g.exact_bytes.load() != 0) {
        // Line-mode write-back merges (EDEN_NX2_TRY_WB_LINES, nvn_line_guard.h), cumulative: MiB
        // merged in line mode / exactly, pages armed with an access trap, pages the guest touched
        // (exact from then on), touches that waited for a merge in flight (count/total ms), waits
        // that gave up, pools that went exact for good.
        text += fmt::format(" wbl[lineMiB={} exactMiB={} armed={} hot={} waits={}/{}ms timeouts={} "
                            "exactPools={}]",
                            g.line_bytes.load() >> 20, g.exact_bytes.load() >> 20,
                            g.armed_pages.load(), g.hot_pages.load(), g.waits.load(),
                            g.wait_us.load() / 1000, g.timeouts.load(), g.exact_pools.load());
    }
    if (Debug::Get().wb_monitor) {
        // Cumulative per mirrored pool, largest write-back first: pool address/size, entries
        // (whole-pool ones), entry MiB, pages armed after the merge (arm calls), pages the guest
        // touched while armed (callbacks), guest-written MiB copied in.
        std::vector<std::shared_ptr<MirrorMonitor>> monitors;
        {
            std::scoped_lock monitor_lock{impl->mirror_monitors_mutex};
            monitors = impl->mirror_monitors;
        }
        std::ranges::sort(monitors, [](const auto& a, const auto& b) {
            return a->entry_bytes.load() > b->entry_bytes.load();
        });
        if (monitors.size() > 10) {
            monitors.resize(10);
        }
        const u32* merge_stats = nullptr;
        if (impl->mirror_merge_stats) {
            impl->mirror_merge_stats.Invalidate();
            merge_stats = reinterpret_cast<const u32*>(impl->mirror_merge_stats.Mapped().data());
        }
        for (const auto& m : monitors) {
            // changedKiB: bytes of words the merges changed; dirtyKiB: the units of 8 or 32 words
            // of the guest buffer (32 or 128 bytes) holding one.
            const u64 changed_kib = merge_stats != nullptr && m->slot < Impl::MIRROR_STAT_SLOTS
                                        ? u64{merge_stats[2 * m->slot]} * 4 / 1024
                                        : 0;
            const u64 dirty_kib = merge_stats != nullptr && m->slot < Impl::MIRROR_STAT_SLOTS
                                      ? u64{merge_stats[2 * m->slot + 1]} *
                                            (impl->write_back_merge
                                                 ? impl->write_back_merge->UnitWords()
                                                 : 8) * 4 / 1024
                                      : 0;
            text += fmt::format(
                " wbmon[{:x}+{:x} entries={}({}) entryMiB={} changedKiB={} dirtyKiB={} "
                "armed={}({}) touched={}({}) copyinMiB={}({}) notify(calls/pages/tracker-dirty/"
                "already-invalid)={}/{}/{}/{}]",
                m->address, m->size, m->entries.load(), m->whole_entries.load(),
                m->entry_bytes.load() >> 20, changed_kib, dirty_kib, m->armed_pages.load(),
                m->armed_runs.load(), m->touched_pages.load(), m->touch_events.load(),
                m->copy_in_bytes.load() >> 20, m->copy_in_runs.load(), m->notify_calls.load(),
                m->notify_pages.load(), m->notify_dirty.load(), m->notify_invalid.load());
        }
    }
    return text;
}

void Context::CollectStats(std::vector<VideoCommon::Nvn::SinkStat>& out) const {
    const ContextStats s = Stats();
    out.push_back({"pools_imported", s.pools_imported});
    out.push_back({"pools_mirrored", s.pools_mirrored});
    out.push_back({"pools_mirrored: guest storage is several host runs",
                   s.pools_mirrored_discontiguous});
    out.push_back({"pools_mirrored: host import unavailable or failed", s.pools_mirrored_no_import});
    out.push_back({"pools_split_imported: several host runs, each imported (raw_split_import)",
                   s.pools_split_imported});
    out.push_back({"pools_split_imported: host runs imported", s.split_runs});
    out.push_back({"mirror uploads (range uses that copied pages)", s.mirror_uploads});
    out.push_back({"mirror uploads: copies (runs of pages)", s.mirror_upload_runs});
    out.push_back({"mirror uploads: KiB copied", s.mirror_upload_bytes >> 10});
    out.push_back({"mirror range uses that needed no copy", s.mirror_ranges_clean});
    out.push_back({"mirror guest pages compared", s.mirror_pages_hashed});
    out.push_back({"mirror guest pages trusted (no write tracked)", s.mirror_pages_trusted});
    out.push_back({"mirror guest pages not hashed (pending GPU writes)", s.mirror_pages_gpu_owned});
    out.push_back({"staging chunks reused", s.staging_chunks_reused});
    out.push_back({"pool mirrors (VRAM copies of imported pools)", s.vram_mirrors});
    out.push_back({"pool mirrors: refused (budget, allocation, no CPU write tracking)",
                   s.vram_mirror_refused});
    out.push_back({"pool mirrors: page runs copied in", s.vram_copy_in_runs});
    out.push_back({"pool mirrors: KiB copied in", s.vram_copy_in_bytes >> 10});
    out.push_back({"pool mirrors: storage bindings and indirect reads served", s.vram_bindings});
    out.push_back({"pool mirrors: bindings left in guest memory in an open pass",
                   s.vram_bindings_host});
    out.push_back({"pool mirrors: written ranges merged back", s.vram_write_ranges});
    out.push_back({"pool mirrors: KiB of written ranges", s.vram_write_bytes >> 10});
    out.push_back({"pool mirrors: written ranges copied back whole (no merge shader)",
                   s.vram_write_backs_plain});
    out.push_back({"pool mirrors: every page invalidated (raw global stores, aborts)",
                   s.vram_invalidate_all});
    out.push_back({"pool mirrors: read-only mirrors (vertex/index/uniform ranges)",
                   s.vram_mirrors_read_only});
    out.push_back({"pool mirrors: read-only mirrors promoted by a written storage range",
                   s.vram_mirrors_promoted});
    out.push_back({"pool mirrors: raw-writing draws that made read-only mirrors stale",
                   s.vram_invalidate_read_only});
    out.push_back({"pool mirrors: raw-writing draws whose stores also wrote the read-only mirrors",
                   s.vram_ro_written_through});
    out.push_back({"pool mirrors: raw write-through alias tables built",
                   s.vram_raw_alias_rebuilds});
    out.push_back({"pool mirrors: vertex and index ranges served from mirrors",
                   s.vram_vertex_ranges});
    out.push_back({"pool mirrors: vertex and index ranges left in guest memory (copy bound)",
                   s.vram_vertex_budget_skips});
    out.push_back({"pool mirrors: guest-flushed pages kept (clean in the page tracker)",
                   s.vram_flush_pages_kept});
    out.push_back({"pool mirrors: guest-flushed pages copied in again", s.vram_flush_pages_taken});
    out.push_back({"presents", s.presents});
    out.push_back({"pools_unsupported", s.pools_unsupported});
    out.push_back({"host_import_failures", s.host_import_failures});
    out.push_back({"textures_failed", s.textures_failed});
    out.push_back({"textures_transcode_unsupported", s.textures_transcode_unsupported});
    out.push_back({"alias_unsupported", s.alias_unsupported});
    out.push_back({"alias_unsupported: reads of such overlaps (events)", s.alias_events});
    out.push_back({"alias_unsupported: events, reader written by the GPU before (stale)",
                   impl->stats.alias_events_stale.load()});
    out.push_back({"alias_unsupported: events, reader holds the guest's initial bytes",
                   impl->stats.alias_events_initial.load()});
    out.push_back({"overlaps not copied over a guest upload (CPU rewrote the bytes after the GPU write)",
                   impl->stats.overlaps_skipped_cpu_rewrite.load()});
    out.push_back({"alias_transfers: through the pool's bytes (nvn_alias_transfer.comp)",
                   impl->stats.alias_byte_transfers.load()});
    out.push_back({"alias_transfers: MiB of window those moved",
                   impl->stats.alias_byte_bytes.load() >> 20});
    if (impl->RescaleFactor().Active()) {
        out.push_back({"resolution scale: passes drawn at the scaled size",
                       impl->stats.rescale_passes.load()});
        out.push_back({"resolution scale: passes that stayed guest size (an attachment is not "
                       "eligible)",
                       impl->stats.rescale_native_passes.load()});
        out.push_back({"resolution scale: downscale blits (guest-sized image made from its twin)",
                       impl->stats.rescale_downscales.load()});
        out.push_back({"resolution scale: upscale blits (twin made from the guest-sized image)",
                       impl->stats.rescale_upscales.load()});
        out.push_back({"resolution scale: copies done on twins",
                       impl->stats.rescale_copies.load()});
        out.push_back({"resolution scale: sampled bindings served by a twin",
                       impl->stats.rescale_twin_samples.load()});
    }
    out.push_back({"dispatches after a dispatch with no barrier between (no game barrier, no "
                   "replay command; EDEN_NX2_TRY_DISPATCH_ORDER)",
                   impl->stats.dispatch_barriers_left.load()});
    out.push_back({"memory barriers recorded (ALL_COMMANDS, between replayed ops)",
                   impl->stats.memory_barriers.load()});
    out.push_back({"barrier merge: buffer commands that needed no barrier (nothing wrote since the last)",
                   impl->stats.barrier_merge_ops_clean.load()});
    out.push_back({"barrier merge: buffer commands whose barrier was left out (no overlap with the "
                   "buffer commands since the last; EDEN_NX2_TRY_BARRIER_MERGE)",
                   impl->stats.barrier_merge_ops_left_out.load()});
    out.push_back({"barrier merge: buffer commands that overlapped an earlier one (barrier recorded)",
                   impl->stats.barrier_merge_ops_hazard.load()});
    out.push_back({"barrier merge: buffer commands after a draw/dispatch/image write (barrier recorded)",
                   impl->stats.barrier_merge_ops_unknown_writes.load()});
    out.push_back({"barrier merge: barrier requests folded into one still waiting for its next command "
                   "(no command between them)",
                   impl->stats.barrier_merge_folded.load()});
    out.push_back({"barrier merge: barrier commands recorded for waiting requests (each stands for "
                   "one or more requests)",
                   impl->stats.barrier_merge_emitted.load()});
    out.push_back({"depth_upload_unsupported", s.depth_upload_unsupported});
    out.push_back({"virtual_unresolved", s.virtual_unresolved});
    out.push_back({"sparse virtual pools checked for CPU writes", s.sparse_write_pool_checks});
    out.push_back({"graphics driver bank uploads", s.driver_bank_uploads});
    out.push_back({"handle bank uploads", s.handle_bank_uploads});
    out.push_back({"graphics pipeline state reuses", s.pipeline_state_reuses});
    out.push_back({"ops_unsupported", s.ops_unsupported});
    out.push_back({"ops_failed", s.ops_failed});
    out.push_back({"device_lost", s.device_lost});
    out.push_back({"dropped_after_device_loss", s.dropped_after_device_loss});
    out.push_back({"held_back_submissions", s.held_back_submissions});
    out.push_back({"parked_timeouts", s.parked_timeouts});
    out.push_back({"waits_unsatisfied", s.waits_unsatisfied});
    if (impl->draw_resources) {
        const DrawCounters& draw = impl->draw_resources->counters;
        out.push_back({"draws", draw.draws.load()});
        out.push_back({"draws_skipped", draw.draws_skipped.load()});
        out.push_back({"draw skipped awaiting async pipeline", draw.draws_skipped_async.load()});
        out.push_back({"frames with a draw skipped awaiting an async pipeline",
                       draw.frames_with_async_skips.load()});
        out.push_back({"descriptors_unsupported", draw.descriptors_unsupported.load()});
        out.push_back({"inline_uniform_updates", draw.inline_updates.load()});
        out.push_back({"inline_uniform_updates: rewritten whole by a later one (no copy)",
                       draw.inline_updates_superseded.load()});
        out.push_back({"inline_uniform_updates: transfer barriers between overlapping copies",
                       draw.inline_update_barriers.load()});
        out.push_back({"barrier merge: inline update flushes that recorded no copy (no barrier)",
                       draw.inline_flushes_without_copy.load()});
        out.push_back({"uniform snapshot: bindings served from the submit-time copy",
                       draw.uniform_snapshot_binds.load()});
        out.push_back({"uniform snapshot: bytes uploaded",
                       draw.uniform_snapshot_upload_bytes.load()});
        out.push_back({"uniform snapshot: inline updates layered into a submission's copy",
                       draw.uniform_snapshot_layers.load()});
        out.push_back({"render_pass_begins", draw.pass_begins.load()});
        out.push_back({"image layout transitions (one barrier entry each)",
                       impl->stats.image_transitions.load()});
        out.push_back({"sampled bindings left in GENERAL (no layout transition)",
                       draw.sampled_in_general.load()});
        out.push_back({"feedback_copy: sampled view overlaps an attached subresource",
                       draw.feedback_overlapping.load()});
        out.push_back({"feedback_copy: skipped (scratch current since the last target change)",
                       draw.feedback_skipped.load()});
        out.push_back({"feedback_copy: sampled view disjoint from the attachments",
                       draw.feedback_disjoint.load()});
        out.push_back({"draw state elided: push constants", draw.elided_push_constants.load()});
        out.push_back({"draw state elided: descriptor sets", draw.elided_descriptors.load()});
        out.push_back({"draw state elided: vertex buffer binds", draw.elided_vertex_binds.load()});
        out.push_back({"draw state elided: index buffer binds", draw.elided_index_binds.load()});
        out.push_back({"vertex staging: passes", draw.vertex_stage_passes.load()});
        out.push_back({"vertex staging: ranges copied", draw.vertex_stage_ranges.load()});
        out.push_back({"vertex staging: copy regions", draw.vertex_stage_regions.load()});
        out.push_back({"vertex staging: KiB copied", draw.vertex_stage_bytes.load() >> 10});
        out.push_back({"vertex staging: ranges reused from an earlier pass",
                       draw.vertex_stage_reused_ranges.load()});
        out.push_back({"vertex staging: KiB reused from an earlier pass",
                       draw.vertex_stage_reused_bytes.load() >> 10});
        out.push_back({"vertex staging: vertex buffer binds from a staged copy",
                       draw.vertex_stage_binds.load()});
        out.push_back({"vertex staging: index buffer binds from a staged copy",
                       draw.index_stage_binds.load()});
        out.push_back({"shadow chunks: created (device memory allocations)",
                       draw.shadow_chunks_created.load()});
        out.push_back({"shadow chunks: destroyed (free list full or too large)",
                       draw.shadow_chunks_destroyed.load()});
        out.push_back({"render_pass_breaks", draw.pass_breaks.load()});
        for (u32 cause = 0; cause < static_cast<u32>(BreakCause::Count); ++cause) {
            const u64 count = draw.break_causes[cause].load();
            if (count != 0) {
                out.push_back({fmt::format("render_pass_break: {}",
                                           BreakCauseName(static_cast<BreakCause>(cause))),
                               count});
            }
        }
        out.push_back({"feedback_copy", draw.feedback_copies.load()});
        out.push_back({"feedback_copy: depth/stencil target", draw.feedback_depth.load()});
        out.push_back({"feedback_copy: color target", draw.feedback_color.load()});
        out.push_back({"feedback_copy: target already copied since SetRenderTargets",
                       draw.feedback_repeat.load()});
        out.push_back({"set_render_targets", draw.render_targets_ops.load()});
        out.push_back({"set_render_targets: unchanged (pass kept)", draw.render_targets_same.load()});
        out.push_back({"barriers kept in a pass without shader writes", draw.barriers_kept.load()});
        out.push_back({"game barriers outside a pass without a wait for earlier work (no "
                       "ORDER_PRIMITIVES/ORDER_INDIRECT_DATA; EDEN_NX2_TRY_DISPATCH_ORDER)",
                       draw.barriers_without_wait.load()});
        out.push_back({"sampled targets moved to their layout at pass begin",
                       draw.flip_transitions.load()});
        out.push_back({"sampled targets: learned into a full list (oldest replaced)",
                       draw.flip_targets_replaced.load()});
        out.push_back({"feedback_unsupported", draw.feedback_unsupported.load()});
        out.push_back({"placeholder 1x1x1 2D sampled as 3D/cube (typed copy bound)",
                       draw.placeholder_copies.load()});
        for (u32 bind = 0; bind < 2; ++bind) {
            for (u32 reason = 0; reason < static_cast<u32>(DescReason::Count); ++reason) {
                const u64 count = draw.fallbacks[bind][reason].load();
                if (count != 0) {
                    out.push_back({fmt::format("descriptor fallback ({}): {}",
                                               bind == 0 ? "graphics" : "compute",
                                               DescReasonName(static_cast<DescReason>(reason))),
                                   count});
                }
            }
        }
        out.push_back({"dispatches", draw.dispatches.load()});
        out.push_back({"dispatches_indirect", draw.dispatches_indirect.load()});
        out.push_back({"dispatches_skipped", draw.dispatches_skipped.load()});
        std::vector<PipelineCache::Row> rows;
        impl->draw_resources->CollectCacheRows(rows);
        for (const auto& row : rows) {
            out.push_back({fmt::format("shader: {}", row.reason), row.count});
        }
    }
    out.push_back({"render_condition_unresolved", s.render_conditions_unresolved});
    out.push_back({"counter_type_unsupported", s.counter_reports_unsupported});
    out.push_back({"samples_report_after_uncounted_draws", s.counter_reports_samples_uncounted});
    out.push_back({"indirect_draw_unsupported", s.indirect_unsupported});
    out.push_back({"raw_global_pools_skipped", s.raw_global_pools_skipped});
    out.push_back({"raw_global_ranges_dropped", s.raw_global_ranges_dropped});
}

namespace {

constexpr u32 PENDING_BLOCK_SHIFT = 14;
constexpr u64 PENDING_MAX_BLOCKS = 64;
constexpr size_t PENDING_MAX_WRITES = 4096;

bool PendingLive(const Context::Impl::PendingPoolWrite& write) {
    return write.end > write.begin;
}

} // Anonymous namespace

template <typename Visitor>
bool Context::Impl::VisitPendingPoolWrites(u64 begin, u64 end, Visitor&& visit) const {
    if (end <= begin || pending_pool_writes.size() == pending_dead) {
        return false;
    }
    const u64 first = begin >> PENDING_BLOCK_SHIFT;
    const u64 last = (end - 1) >> PENDING_BLOCK_SHIFT;
    if (last - first >= PENDING_MAX_BLOCKS) {
        for (u32 index = 0; index < static_cast<u32>(pending_pool_writes.size()); ++index) {
            if (PendingLive(pending_pool_writes[index]) && visit(index)) {
                return true;
            }
        }
        return false;
    }
    for (const u32 index : pending_wide) {
        if (visit(index)) {
            return true;
        }
    }
    for (u64 block = first; block <= last; ++block) {
        const auto it = pending_blocks.find(block);
        if (it == pending_blocks.end()) {
            continue;
        }
        for (const u32 index : it->second) {
            if (visit(index)) {
                return true;
            }
        }
    }
    return false;
}

void Context::Impl::IndexPendingPoolWrite(u32 index, bool add) {
    const PendingPoolWrite& write = pending_pool_writes[index];
    const auto apply = [&](std::vector<u32>& list) {
        if (add) {
            list.push_back(index);
        } else if (const auto it = std::ranges::find(list, index); it != list.end()) {
            *it = list.back();
            list.pop_back();
        }
    };
    const u64 first = write.begin >> PENDING_BLOCK_SHIFT;
    const u64 last = (write.end - 1) >> PENDING_BLOCK_SHIFT;
    if (last - first >= PENDING_MAX_BLOCKS) {
        apply(pending_wide);
        return;
    }
    for (u64 block = first; block <= last; ++block) {
        if (add) {
            apply(pending_blocks[block]);
        } else if (const auto it = pending_blocks.find(block); it != pending_blocks.end()) {
            apply(it->second);
        }
    }
}

void Context::Impl::PushPendingPoolWrite(PendingPoolWrite&& write) {
    if (!write.inline_bytes.empty()) {
        ++pending_inline;
    }
    // (a write of a submission already passed is dropped by the next prune, as it always was)
    if (write.seq <= pending_pruned_done) {
        pending_prune_due = true;
    }
    pending_pool_writes.push_back(std::move(write));
    IndexPendingPoolWrite(static_cast<u32>(pending_pool_writes.size() - 1), true);
}

void Context::Impl::RemovePendingPoolWrite(u32 index) {
    PendingPoolWrite& write = pending_pool_writes[index];
    if (!PendingLive(write)) {
        return;
    }
    IndexPendingPoolWrite(index, false);
    if (!write.inline_bytes.empty()) {
        --pending_inline;
    }
    write = {};
    ++pending_dead;
}

void Context::Impl::RemovePendingPoolWritesWithin(u64 begin, u64 end) {
    thread_local std::vector<u32> contained;
    contained.clear();
    VisitPendingPoolWrites(begin, end, [&](u32 index) {
        const PendingPoolWrite& write = pending_pool_writes[index];
        if (write.begin >= begin && write.end <= end) {
            contained.push_back(index);
        }
        return false;
    });
    for (const u32 index : contained) {
        RemovePendingPoolWrite(index);
    }
    SettlePendingPoolWrites();
}

void Context::Impl::SettlePendingPoolWrites() {
    while (!pending_pool_writes.empty() && !PendingLive(pending_pool_writes.back())) {
        pending_pool_writes.pop_back();
        --pending_dead;
    }
    if (pending_dead > 64 && pending_dead * 2 > pending_pool_writes.size()) {
        std::erase_if(pending_pool_writes,
                      [](const PendingPoolWrite& write) { return !PendingLive(write); });
        pending_dead = 0;
        RebuildPendingPoolIndex();
    }
}

void Context::Impl::RebuildPendingPoolIndex() {
    if (pending_blocks.size() > 4 * pending_pool_writes.size() + 1024) {
        pending_blocks.clear();
    } else {
        for (auto& [block, list] : pending_blocks) {
            list.clear();
        }
    }
    pending_wide.clear();
    for (u32 index = 0; index < static_cast<u32>(pending_pool_writes.size()); ++index) {
        if (PendingLive(pending_pool_writes[index])) {
            IndexPendingPoolWrite(index, true);
        }
    }
}

void Context::Impl::CollapsePendingPoolWrites() {
    if (pending_pool_writes.size() - pending_dead < PENDING_MAX_WRITES) {
        return;
    }
    std::vector<PendingPoolWrite> collapsed;
    for (const PendingPoolWrite& write : pending_pool_writes) {
        if (!PendingLive(write)) {
            continue;
        }
        const auto it = std::ranges::find_if(
            collapsed, [&](const auto& hull) { return hull.seq == write.seq; });
        if (it == collapsed.end()) {
            collapsed.push_back({write.begin, write.end, write.seq, {}});
        } else {
            it->begin = (std::min)(it->begin, write.begin);
            it->end = (std::max)(it->end, write.end);
        }
    }
    pending_pool_writes = std::move(collapsed);
    pending_dead = 0;
    pending_inline = 0;
    pending_prune_due = true;
    RebuildPendingPoolIndex();
}

void Context::Impl::PrunePendingPoolWrites() {
    // Every write kept is of a submission that had not completed when the list was last pruned
    // (or raised pending_prune_due), so there is nothing to drop until gpu_completed_seq moves.
    const u64 done = gpu_completed_seq.load(std::memory_order_acquire);
    if (done == pending_pruned_done && !pending_prune_due) {
        return;
    }
    pending_pruned_done = done;
    pending_prune_due = false;
    for (u32 index = 0; index < static_cast<u32>(pending_pool_writes.size()); ++index) {
        const PendingPoolWrite& write = pending_pool_writes[index];
        if (PendingLive(write) && write.seq <= done) {
            RemovePendingPoolWrite(index);
        }
    }
    SettlePendingPoolWrites();
}

void Context::Impl::NoteFlushedInlinePoolWrite(u64 address, const u8* data, u64 size, u64 seq) {
    if (size == 0 || data == nullptr) {
        return;
    }
    PrunePendingPoolWrites();
    const u64 end = address + size;
    RemovePendingPoolWritesWithin(address, end);
    CollapsePendingPoolWrites();
    PushPendingPoolWrite(
        {address, end, seq, std::vector<u8>(data, data + static_cast<size_t>(size))});
}

void Context::Impl::NoteNonInlinePoolWrite(u64 address, u64 size, u64 seq) {
    if (size == 0 || noting_inline_pool_write) {
        return;
    }
    PrunePendingPoolWrites();
    const u64 end = address + size;
    RemovePendingPoolWritesWithin(address, end);
    if (pending_inline != 0) {
        VisitPendingPoolWrites(address, end, [&](u32 index) {
            PendingPoolWrite& write = pending_pool_writes[index];
            if (!write.inline_bytes.empty() && write.begin < end && address < write.end) {
                write.inline_bytes = {};
                --pending_inline;
            }
            return false;
        });
    }
    if (!pending_pool_writes.empty()) {
        // (settled above: the last entry is a live write)
        PendingPoolWrite& last = pending_pool_writes.back();
        if (last.seq == seq && last.inline_bytes.empty() && address <= last.end &&
            last.begin <= end) {
            const u32 index = static_cast<u32>(pending_pool_writes.size() - 1);
            IndexPendingPoolWrite(index, false);
            last.begin = (std::min)(last.begin, address);
            last.end = (std::max)(last.end, end);
            IndexPendingPoolWrite(index, true);
            return;
        }
    }
    CollapsePendingPoolWrites();
    PushPendingPoolWrite({address, end, seq, {}});
}

void Context::Impl::InvalidatePendingInlineBytes(u64 seq) {
    PrunePendingPoolWrites();
    if (pending_inline == 0) {
        return;
    }
    for (PendingPoolWrite& write : pending_pool_writes) {
        if (!write.inline_bytes.empty() && (seq == 0 || write.seq == seq)) {
            write.inline_bytes = {};
            --pending_inline;
        }
    }
}

void Context::Impl::InvalidatePendingInlineBytesOverlapping(u64 address, u64 size) {
    if (size == 0 || pending_pool_writes.size() == pending_dead) {
        return;
    }
    PrunePendingPoolWrites();
    if (pending_inline == 0) {
        return;
    }
    const u64 end = address + size;
    VisitPendingPoolWrites(address, end, [&](u32 index) {
        PendingPoolWrite& write = pending_pool_writes[index];
        if (!write.inline_bytes.empty() && write.begin < end && address < write.end) {
            write.inline_bytes = {};
            --pending_inline;
        }
        return false;
    });
}

void Context::Impl::ErasePendingPoolWritesForSeq(u64 seq) {
    for (u32 index = 0; index < static_cast<u32>(pending_pool_writes.size()); ++index) {
        const PendingPoolWrite& write = pending_pool_writes[index];
        if (PendingLive(write) && write.seq == seq) {
            RemovePendingPoolWrite(index);
        }
    }
    SettlePendingPoolWrites();
}

bool Context::Impl::NeedsGpuFlushForUniform(u64 begin, u64 end, u64 current_seq) {
    if (end <= begin || pending_pool_writes.size() == pending_dead) {
        return false;
    }
    PrunePendingPoolWrites();
    return VisitPendingPoolWrites(begin, end, [&](u32 index) {
        const PendingPoolWrite& write = pending_pool_writes[index];
        return write.begin < end && begin < write.end &&
               (write.inline_bytes.empty() || write.seq != current_seq);
    });
}

void Context::Impl::ApplyPendingInlineWrites(u64 base_address, u8* dst, u64 size,
                                             u64 current_seq,
                                             std::vector<std::pair<u64, u64>>* written) const {
    if (size == 0 || dst == nullptr || pending_inline == 0) {
        return;
    }
    const u64 end = base_address + size;
    // In the order noted: a later update of the same bytes wins.
    thread_local std::vector<u32> overlapping;
    overlapping.clear();
    VisitPendingPoolWrites(base_address, end, [&](u32 index) {
        const PendingPoolWrite& write = pending_pool_writes[index];
        if (write.seq == current_seq && !write.inline_bytes.empty() && write.begin < end &&
            base_address < write.end) {
            overlapping.push_back(index);
        }
        return false;
    });
    std::ranges::sort(overlapping);
    const auto repeated = std::ranges::unique(overlapping);
    overlapping.erase(repeated.begin(), repeated.end());
    for (const u32 index : overlapping) {
        const PendingPoolWrite& write = pending_pool_writes[index];
        const u64 lo = (std::max)(write.begin, base_address);
        const u64 hi = (std::min)(write.end, end);
        std::memcpy(dst + (lo - base_address), write.inline_bytes.data() + (lo - write.begin),
                    static_cast<size_t>(hi - lo));
        if (written != nullptr) {
            written->emplace_back(lo - base_address, hi - base_address);
        }
    }
}

} // namespace Vulkan::Nvn
