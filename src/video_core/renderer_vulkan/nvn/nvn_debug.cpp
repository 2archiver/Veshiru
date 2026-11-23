// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <mutex>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <system_error>

#include <limits>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_gpu_profile.h"
#include "video_core/renderer_vulkan/nvn/nvn_vtxcov.h"

namespace Vulkan::Nvn::Debug {
namespace {

u64 EnvNumber(const char* name, u64 fallback) {
    if (const char* value = std::getenv(name); value != nullptr && *value != '\0') {
        return std::strtoull(value, nullptr, 0);
    }
    return fallback;
}

bool EnvFlag(const char* name, bool fallback = false) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::strcmp(value, "0") != 0;
}

Config Read() {
    Config config;
    config.draw_log = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DRAWLOG", 0));
    if (const char* dir = std::getenv("EDEN_NX2_NVN_DUMP_TARGETS"); dir != nullptr && *dir != '\0') {
        config.dump_dir = dir;
    }
    if (const char* list = std::getenv("EDEN_NX2_NVN_DUMP_FRAME"); list != nullptr && *list != '\0') {
        for (const char* it = list; *it != '\0';) {
            char* end = nullptr;
            const u64 frame = std::strtoull(it, &end, 0);
            if (end == it) {
                break;
            }
            if (frame != 0) {
                config.dump_frames.push_back(frame);
            }
            it = *end == ',' ? end + 1 : end;
        }
    }
    config.dump_count = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_COUNT", 1));
    if (const char* dir = std::getenv("EDEN_NX2_NVN_DUMP_SPIRV"); dir != nullptr && *dir != '\0') {
        config.dump_spirv = dir;
    }
    config.dump_draw = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_DRAW", 0));
    config.dump_pass = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_PASS", 0));
    config.dump_max = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_MAX", 400));
    config.dump_width = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_WIDTH", 0));
    config.dump_thumb = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_THUMB", 0));
    config.dump_divisor = (std::max)(static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DUMP_DIVISOR", 1)), 1u);
    config.face = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_FACE", 0));
    config.equal_lequal = EnvFlag("EDEN_NX2_NVN_RELAX_DEPTH_EQUAL");
    config.cache_log = EnvFlag("EDEN_NX2_NVN_CACHE_LOG");
    config.async_workers = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_ASYNC_WORKERS", 0));
    config.upload_log = EnvFlag("EDEN_NX2_NVN_UPLOADLOG");
    config.storage_census = EnvFlag("EDEN_NX2_NVN_STORAGE_CENSUS");
    config.astc_verify = EnvFlag("EDEN_NX2_NVN_ASTC_VERIFY");
    config.no_depth = EnvFlag("EDEN_NX2_NVN_NO_DEPTH");
    config.depth_probe = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_DEPTH_PROBE", 0));
    if (const char* list = std::getenv("EDEN_NX2_NVN_ONLY_DRAWS"); list != nullptr && *list != '\0') {
        for (const char* p = list; *p != '\0';) {
            char* end = nullptr;
            const unsigned long value = std::strtoul(p, &end, 10);
            if (end == p) {
                break;
            }
            config.only_draws.push_back(static_cast<u32>(value));
            p = *end == ',' ? end + 1 : end;
        }
    }
    if (const char* list = std::getenv("EDEN_NX2_NVN_ONLY_VS"); list != nullptr && *list != '\0') {
        for (const char* p = list; *p != '\0';) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(p, &end, 16);
            if (end == p) {
                break;
            }
            config.only_vs.push_back(value);
            p = *end == ',' ? end + 1 : end;
        }
    }
    config.flip_viewport = EnvFlag("EDEN_NX2_NVN_FLIP_VIEWPORT");
    config.gpu_profile = GpuProf::ParseMode(std::getenv("EDEN_NX2_NVN_GPU_PROFILE"));
    config.no_zero_size = EnvFlag("EDEN_NX2_NVN_NO_ZERO_SIZE");
    config.no_state_elide = EnvFlag("EDEN_NX2_NVN_NO_STATE_ELIDE");
    if (const char* split = std::getenv("EDEN_NX2_NVN_COSTSPLIT");
        split != nullptr && *split != '\0') {
        config.costsplit = split;
    }
    if (const char* stage = std::getenv("EDEN_NX2_NVN_VERTEX_STAGE");
        stage != nullptr && std::strcmp(stage, "0") == 0) {
        config.vertex_stage = false;
    }
    if (const char* stage = std::getenv("EDEN_NX2_NVN_INDEX_STAGE");
        stage != nullptr && std::strcmp(stage, "0") == 0) {
        config.index_stage = false;
    }
    config.feedback_log = EnvFlag("EDEN_NX2_NVN_FEEDBACK_LOG");
    if (const char* shadow = std::getenv("EDEN_NX2_NVN_SHADOW_STORAGE");
        shadow != nullptr && std::strcmp(shadow, "0") == 0) {
        config.shadow_storage = false;
    }
    if (const char* prefetch = std::getenv("EDEN_NX2_NVN_CONDITION_PREFETCH");
        prefetch != nullptr && std::strcmp(prefetch, "0") == 0) {
        config.condition_prefetch = false;
    }
    config.shadow_max_bytes = EnvNumber("EDEN_NX2_NVN_SHADOW_MAX_KIB", 1024) << 10;
    config.shadow_log = EnvFlag("EDEN_NX2_NVN_SHADOW_LOG");
    config.pool_mirror = static_cast<u32>((std::min<u64>)(EnvNumber("EDEN_NX2_NVN_POOL_MIRROR", 0), 2));
    config.pool_mirror_max_bytes = EnvNumber("EDEN_NX2_NVN_POOL_MIRROR_MAX_MIB", 4096) << 20;
    config.storage_budget_bytes = EnvNumber("EDEN_NX2_NVN_STORAGE_BUDGET_MIB", 0) << 20;
    config.trim_age_presents = EnvNumber("EDEN_NX2_NVN_TRIM_AGE_PRESENTS", 0);
    config.trim_legacy = EnvFlag("EDEN_NX2_NVN_TRIM_LEGACY");
    config.try_vma_device_strict = EnvFlag("EDEN_NX2_TRY_VMA_DEVICE_STRICT");
    config.try_storage_vram = EnvFlag("EDEN_NX2_TRY_STORAGE_VRAM", true);
    config.try_storage_vram_max_bytes =
        EnvNumber("EDEN_NX2_TRY_STORAGE_VRAM_MAX_MIB", 64) << 20;
    config.wb_monitor = EnvFlag("EDEN_NX2_NVN_WB_MONITOR");
    config.try_wb_lines = static_cast<u32>(EnvNumber("EDEN_NX2_TRY_WB_LINES", 8));
    config.try_dispatch_order = EnvFlag("EDEN_NX2_TRY_DISPATCH_ORDER");
    config.try_flush_tracked = EnvFlag("EDEN_NX2_TRY_FLUSH_TRACKED", true);
    config.mirror_hash_always = EnvFlag("EDEN_NX2_NVN_MIRROR_HASH_ALWAYS");
    config.raw_split_import = EnvFlag("EDEN_NX2_NVN_RAW_SPLIT_IMPORT", true);
    config.tex_hash_always = EnvFlag("EDEN_NX2_NVN_TEX_HASH_ALWAYS");
    config.tex_page_hash = EnvFlag("EDEN_NX2_NVN_TEX_PAGE_HASH", true);
    config.tex_flush_trusted = EnvFlag("EDEN_NX2_NVN_TEX_FLUSH_TRUSTED", true);
    if (const u64 wait = EnvNumber("EDEN_NX2_TRY_INVALIDATE_WRITEBACK", 0); wait != 0) {
        config.try_invalidate_writeback =
            static_cast<u32>(wait == 1 ? 250 : (std::min<u64>)(wait, 5000));
    }
    config.try_barrier_merge = static_cast<u32>(EnvNumber("EDEN_NX2_TRY_BARRIER_MERGE", 0));
    config.try_stage_span = EnvFlag("EDEN_NX2_TRY_STAGE_SPAN");
    config.try_ubo_mirror = EnvFlag("EDEN_NX2_TRY_UBO_MIRROR");
    // Vertex/index ranges from VRAM: on by default (OoT Kokiri village 39-40 -> 36-37 ms GPU, every
    // range served). =0 turns a switch off; EDEN_NX2_TRY_VTXCOV=0 turns all four coverage switches off.
    config.try_vtx_mirror = EnvFlag("EDEN_NX2_TRY_VTX_MIRROR", true);
    config.try_vtxcov = EnvFlag("EDEN_NX2_TRY_VTXCOV", true) ? VTXCOV_ALL : 0u;
    const auto vtxcov_switch = [&](const char* name, u32 bit) {
        if (const char* value = std::getenv(name); value != nullptr && *value != '\0') {
            config.try_vtxcov = std::strcmp(value, "0") != 0 ? (config.try_vtxcov | bit)
                                                            : (config.try_vtxcov & ~bit);
        }
    };
    vtxcov_switch("EDEN_NX2_TRY_VTXCOV_INPASS", VTXCOV_INPASS);
    vtxcov_switch("EDEN_NX2_TRY_VTXCOV_SHADERWRITES", VTXCOV_SHADER_WRITES);
    vtxcov_switch("EDEN_NX2_TRY_VTXCOV_BOUND", VTXCOV_BOUND);
    vtxcov_switch("EDEN_NX2_TRY_VTXCOV_BIGPOOL", VTXCOV_BIGPOOL);
    // The per-bind coverage counters cost replay CPU, so they are opt-in now that the switches are on.
    config.vtxcov_diag = EnvFlag("EDEN_NX2_NVN_VTXCOV_DIAG");
    config.try_raw_keep_entries = EnvFlag("EDEN_NX2_TRY_RAW_KEEP_ENTRIES", true);
    // On by default: OoT's CMAA edge pass (3816717874cb1f93) keeps a 1.76 MiB counter + edge list in a
    // refused 305 MiB pool; in VRAM it drops from ~10 ms to ~1 ms (Kokiri village 36 -> 24 ms GPU).
    config.try_refused_shadow = EnvFlag("EDEN_NX2_TRY_REFUSED_SHADOW", true);
    config.refused_shadow_max_bytes = EnvNumber("EDEN_NX2_NVN_REFUSED_SHADOW_MAX_KIB", 4096) << 10;
    config.diag_ro_raw_keep = EnvFlag("EDEN_NX2_NVN_DIAG_RO_RAW_KEEP");
    if (const char* mode = std::getenv("EDEN_NX2_NVN_DEPTH_MODE"); mode != nullptr && *mode != '\0') {
        config.depth_mode = std::atoi(mode) != 0 ? 1 : 0;
    }
    config.early_interval_us = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_EARLY_SUBMIT_US", 1500));
    config.early_first_us = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_EARLY_FIRST_US", 300));
    config.early_pass_us = static_cast<u32>(EnvNumber("EDEN_NX2_NVN_EARLY_PASS_US", 4000));
    config.op_log = EnvFlag("EDEN_NX2_NVN_OPLOG");
    if (const char* watch = std::getenv("EDEN_NX2_NVN_WATCH"); watch != nullptr && *watch != '\0') {
        char* end = nullptr;
        // "gs": the first storage buffer of the geometry stage of the last draw that had one; "ind": the
        // arguments of the last indirect draw; "peek": the GPU view of the EDEN_NX2_NVN_PEEK range,
        // copied once right after each peeked dispatch.
        if (std::strncmp(watch, "peek", 4) == 0 || std::strncmp(watch, "disp", 4) == 0) {
            config.watch_address = watch[0] == 'p' ? 3 : 4;
            end = const_cast<char*>(watch + 4);
        } else if (std::strncmp(watch, "gs", 2) == 0 || std::strncmp(watch, "ind", 3) == 0) {
            config.watch_address = watch[0] == 'g' ? 1 : 2;
            end = const_cast<char*>(watch + (watch[0] == 'g' ? 2 : 3));
        } else {
            config.watch_address = std::strtoull(watch, &end, 16);
        }
        if (*end == ':') {
            config.watch_size = static_cast<u32>(std::strtoul(end + 1, nullptr, 16));
        }
    }
    // "<program hash>:<storage slot>:<hex offset>:<words>", hex hash and offset.
    if (const char* peek = std::getenv("EDEN_NX2_NVN_PEEK"); peek != nullptr && *peek != '\0') {
        char* end = nullptr;
        config.peek_program = std::strtoull(peek, &end, 16);
        if (*end == ':') {
            config.peek_slot = static_cast<u32>(std::strtoul(end + 1, &end, 10));
        }
        if (*end == ':') {
            config.peek_offset = std::strtoull(end + 1, &end, 16);
        }
        if (*end == ':') {
            config.peek_words = static_cast<u32>(std::strtoul(end + 1, &end, 10));
        }
    }
    if (const char* backing = std::getenv("EDEN_NX2_NVN_LOG_BACKING");
        backing != nullptr && *backing != '\0') {
        char* end = nullptr;
        config.log_backing_program = std::strtoull(backing, &end, 16);
        if (*end == ':') {
            const unsigned long max_dispatches = std::strtoul(end + 1, &end, 10);
            config.log_backing_max = static_cast<u32>(
                (std::min<unsigned long>)(max_dispatches, std::numeric_limits<u32>::max()));
        }
    }

    if (const char* poke = std::getenv("EDEN_NX2_NVN_POKE"); poke != nullptr && *poke != '\0') {
        char* end = nullptr;
        config.poke_index = static_cast<u32>(std::strtoul(poke, &end, 10));
        if (*end == ':') {
            config.poke_value = static_cast<u32>(std::strtoul(end + 1, nullptr, 16));
            config.poke_set = true;
        }
    }
    if (const char* chase = std::getenv("EDEN_NX2_NVN_PEEK_CHASE"); chase != nullptr && *chase != '\0') {
        char* end = nullptr;
        config.chase_word = static_cast<u32>(std::strtoul(chase, &end, 10));
        if (*end == ':') {
            config.chase_offset = std::strtoull(end + 1, &end, 16);
        }
        if (*end == ':') {
            config.chase_words = static_cast<u32>(std::strtoul(end + 1, &end, 10));
        }
    }
    config.tracing = config.draw_log != 0 || !config.dump_dir.empty() || !config.only_draws.empty() ||
                     !config.only_vs.empty() || config.op_log;
    return config;
}

float HalfToFloat(u16 h) {
    const u32 sign = (h >> 15) & 1;
    const u32 exponent = (h >> 10) & 0x1F;
    const u32 mantissa = h & 0x3FF;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0 ? INFINITY : NAN;
    } else {
        value = std::ldexp(static_cast<float>(mantissa | 0x400), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

/// Unsigned small float (5-bit exponent, `mantissa_bits` mantissa bits).
float SmallFloat(u32 bits, u32 mantissa_bits) {
    const u32 exponent = bits >> mantissa_bits;
    const u32 mantissa = bits & ((1u << mantissa_bits) - 1);
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissa_bits));
    }
    if (exponent == 31) {
        return mantissa == 0 ? INFINITY : NAN;
    }
    return std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissa_bits),
                      static_cast<int>(exponent) - 15);
}

template <typename T>
T Load(const u8* p) {
    T value;
    std::memcpy(&value, p, sizeof(T));
    return value;
}

struct Rgba {
    float r, g, b, a;
};

/// One pixel of `format` as float RGBA. `depth`: the value is a depth (replicated to rgb).
bool Decode(VkFormat format, VkImageAspectFlags aspect, const u8* p, Rgba& out) {
    constexpr float k8 = 1.0f / 255.0f;
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
        float depth = 0.0f;
        switch (format) {
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            depth = Load<float>(p);
            break;
        case VK_FORMAT_D16_UNORM:
            depth = static_cast<float>(Load<u16>(p)) / 65535.0f;
            break;
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
            depth = static_cast<float>(Load<u32>(p) & 0xFFFFFF) / 16777215.0f;
            break;
        default:
            return false;
        }
        out = {depth, depth, depth, 1.0f};
        return true;
    }
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        out = {p[0] * k8, p[1] * k8, p[2] * k8, p[3] * k8};
        return true;
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        out = {p[2] * k8, p[1] * k8, p[0] * k8, p[3] * k8};
        return true;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: {
        const u32 v = Load<u32>(p);
        out = {static_cast<float>(v & 0x3FF) / 1023.0f, static_cast<float>((v >> 10) & 0x3FF) / 1023.0f,
               static_cast<float>((v >> 20) & 0x3FF) / 1023.0f, static_cast<float>(v >> 30) / 3.0f};
        return true;
    }
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: {
        const u32 v = Load<u32>(p);
        out = {SmallFloat(v & 0x7FF, 6), SmallFloat((v >> 11) & 0x7FF, 6), SmallFloat(v >> 22, 5), 1.0f};
        return true;
    }
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        out = {HalfToFloat(Load<u16>(p)), HalfToFloat(Load<u16>(p + 2)), HalfToFloat(Load<u16>(p + 4)),
               HalfToFloat(Load<u16>(p + 6))};
        return true;
    case VK_FORMAT_R16G16B16A16_UNORM:
        out = {Load<u16>(p) / 65535.0f, Load<u16>(p + 2) / 65535.0f, Load<u16>(p + 4) / 65535.0f,
               Load<u16>(p + 6) / 65535.0f};
        return true;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        out = {Load<float>(p), Load<float>(p + 4), Load<float>(p + 8), Load<float>(p + 12)};
        return true;
    case VK_FORMAT_R8_UNORM:
        out = {p[0] * k8, p[0] * k8, p[0] * k8, 1.0f};
        return true;
    case VK_FORMAT_R8G8_UNORM:
        out = {p[0] * k8, p[1] * k8, 0.0f, 1.0f};
        return true;
    case VK_FORMAT_R16_SFLOAT: {
        const float v = HalfToFloat(Load<u16>(p));
        out = {v, v, v, 1.0f};
        return true;
    }
    case VK_FORMAT_R16_UNORM: {
        const float v = Load<u16>(p) / 65535.0f;
        out = {v, v, v, 1.0f};
        return true;
    }
    case VK_FORMAT_R16G16_SFLOAT:
        out = {HalfToFloat(Load<u16>(p)), HalfToFloat(Load<u16>(p + 2)), 0.0f, 1.0f};
        return true;
    case VK_FORMAT_R32G32_SFLOAT:
        out = {Load<float>(p), Load<float>(p + 4), 0.0f, 1.0f};
        return true;
    case VK_FORMAT_R32_SFLOAT: {
        const float v = Load<float>(p);
        out = {v, v, v, 1.0f};
        return true;
    }
    default:
        return false;
    }
}

u8 ToByte(float v) {
    if (!(v > 0.0f)) {
        return 0; // also NaN
    }
    return static_cast<u8>((std::min)(v, 1.0f) * 255.0f + 0.5f);
}

/// The 8-bit TGA holds 8-bit UNORM/sRGB colour exactly; depth and every wider or float format lose
/// values in it (clamped to [0, 1], quantized), so their unscaled dumps keep the raw texels too.
bool KeepsRaw(VkFormat format, VkImageAspectFlags aspect) {
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
        return true;
    }
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8G8_UNORM:
        return false;
    default:
        return true;
    }
}

} // namespace

Config g_config = Read();

bool g_timeline = EnvFlag("EDEN_NX2_NVN_TIMELINE");

void Reload() {
    g_config = Read();
    g_timeline = EnvFlag("EDEN_NX2_NVN_TIMELINE");
}

void TimelineEvent(const char* event, u64 seq, std::string_view detail) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    if (detail.empty()) {
        LOG_INFO(Render_Vulkan, "[nvn] tl {} {} {}", us, event, seq);
    } else {
        LOG_INFO(Render_Vulkan, "[nvn] tl {} {} {} {}", us, event, seq, detail);
    }
}

bool FrameSelected(u64 present_index) {
    const Config& config = Get();
    if (config.dump_frames.empty()) {
        return present_index % 120 == 1;
    }
    const u64 count = (std::max)(config.dump_count, 1u);
    return std::ranges::any_of(config.dump_frames, [&](u64 frame) {
        return present_index >= frame && present_index < frame + count;
    });
}

std::atomic<u32> g_request_phase{0};

namespace {

struct Requests {
    std::mutex mutex;
    RequestState pending;
    bool armed{false};
    RequestState active; ///< touched by the replay thread only
};

Requests& GetRequests() {
    static Requests requests;
    return requests;
}

} // namespace

void RequestDump(u32 pass, u32 draw, std::filesystem::path directory) {
    Requests& requests = GetRequests();
    std::scoped_lock lock{requests.mutex};
    if (pass != 0) {
        requests.pending.pass = pass;
    }
    if (draw != 0) {
        requests.pending.draw = draw;
    }
    requests.pending.directory = std::move(directory);
    requests.armed = true;
    u32 idle = 0;
    g_request_phase.compare_exchange_strong(idle, 1u, std::memory_order_relaxed);
}

bool OnPresent(u64 present_index) {
    if (g_request_phase.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    Requests& requests = GetRequests();
    std::scoped_lock lock{requests.mutex};
    if (requests.armed) {
        requests.active = std::move(requests.pending);
        requests.pending = {};
        requests.armed = false;
        requests.active.frame = present_index + 1;
        g_request_phase.store(2, std::memory_order_relaxed);
        return false;
    }
    if (g_request_phase.load(std::memory_order_relaxed) == 2 &&
        present_index >= requests.active.frame) {
        requests.active = {};
        g_request_phase.store(0, std::memory_order_relaxed);
        return true;
    }
    return false;
}

const RequestState* RequestFor(u64 frame) {
    if (g_request_phase.load(std::memory_order_relaxed) != 2) {
        return nullptr;
    }
    const Requests& requests = GetRequests();
    return requests.active.frame == frame ? &requests.active : nullptr;
}

void ResetRequests() {
    Requests& requests = GetRequests();
    std::scoped_lock lock{requests.mutex};
    requests.pending = {};
    requests.active = {};
    requests.armed = false;
    g_request_phase.store(0, std::memory_order_relaxed);
}

void AppendLine(const std::filesystem::path& path, std::string_view text) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream out{path, std::ios::app | std::ios::binary};
    if (out) {
        out << text << '\n';
    }
}

u32 PixelBytes(VkFormat format, VkImageAspectFlags aspect) {
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
        switch (format) {
        case VK_FORMAT_D16_UNORM:
            return 2;
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
            return 4;
        default:
            return 0;
        }
    }
    if (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) {
        return 0;
    }
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    case VK_FORMAT_R8_UNORM:
        return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
        return 2;
    default:
        return 0;
    }
}

std::string WriteTga(const PendingDump& dump, std::span<const u8> pixels) {
    const u32 bpp = PixelBytes(dump.format, dump.aspect);
    const size_t count = static_cast<size_t>(dump.width) * dump.height;
    if (bpp == 0 || dump.width == 0 || dump.height == 0 || pixels.size() < count * bpp) {
        return "unconvertible";
    }
    // Thumbnail: the smallest integer box factor that fits the width cap.
    const u32 factor = dump.thumb_width != 0 && dump.width > dump.thumb_width
                           ? (dump.width + dump.thumb_width - 1) / dump.thumb_width
                           : 1;
    const u32 out_width = (std::max)((dump.width + factor - 1) / factor, 1u);
    const u32 out_height = (std::max)((dump.height + factor - 1) / factor, 1u);
    std::vector<Rgba> box(static_cast<size_t>(out_width) * out_height);
    std::vector<u32> box_count(box.size());
    double sum[4]{};
    float lo = INFINITY;
    float hi = -INFINITY;
    size_t nonzero = 0;
    for (size_t i = 0; i < count; ++i) {
        Rgba c;
        if (!Decode(dump.format, dump.aspect, pixels.data() + i * bpp, c)) {
            return "unconvertible";
        }
        const u32 x = static_cast<u32>(i % dump.width) / factor;
        const u32 y = static_cast<u32>(i / dump.width) / factor;
        if (x < out_width && y < out_height) {
            Rgba& b = box[static_cast<size_t>(y) * out_width + x];
            b.r += std::isfinite(c.r) ? c.r : 0.0f;
            b.g += std::isfinite(c.g) ? c.g : 0.0f;
            b.b += std::isfinite(c.b) ? c.b : 0.0f;
            ++box_count[static_cast<size_t>(y) * out_width + x];
        }
        sum[0] += std::isfinite(c.r) ? c.r : 0.0;
        sum[1] += std::isfinite(c.g) ? c.g : 0.0;
        sum[2] += std::isfinite(c.b) ? c.b : 0.0;
        sum[3] += std::isfinite(c.a) ? c.a : 0.0;
        if (std::isfinite(c.r)) {
            lo = (std::min)(lo, c.r);
            hi = (std::max)(hi, c.r);
        }
        if (c.r != 0.0f || c.g != 0.0f || c.b != 0.0f) {
            ++nonzero;
        }
    }
    std::vector<u8> out(box.size() * 4);
    const bool is_float = dump.format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                          dump.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                          dump.format == VK_FORMAT_R32G32B32A32_SFLOAT;
    const float divisor = is_float ? static_cast<float>(Get().dump_divisor) : 1.0f;
    for (size_t i = 0; i < box.size(); ++i) {
        const float n = static_cast<float>((std::max)(box_count[i], 1u)) * divisor;
        u8* o = out.data() + i * 4;
        o[0] = ToByte(box[i].b / n);
        o[1] = ToByte(box[i].g / n);
        o[2] = ToByte(box[i].r / n);
        o[3] = 255;
    }
    bool written = false;
    if (std::FILE* file = std::fopen(dump.path.c_str(), "wb")) {
        std::array<u8, 18> header{};
        header[2] = 2;
        header[12] = static_cast<u8>(out_width & 0xFF);
        header[13] = static_cast<u8>(out_width >> 8);
        header[14] = static_cast<u8>(out_height & 0xFF);
        header[15] = static_cast<u8>(out_height >> 8);
        header[16] = 32;
        header[17] = 0x28; // 8 alpha bits, top-down
        written = std::fwrite(header.data(), 1, header.size(), file) == header.size();
        written = std::fwrite(out.data(), 1, out.size(), file) == out.size() && written;
        written = std::fclose(file) == 0 && written;
    }
    if (factor == 1 && KeepsRaw(dump.format, dump.aspect)) {
        // The 8-bit TGA hides small depth differences and clamps float colour: keep the raw
        // texels (the host format's bytes, tightly packed) next to it.
        if (std::FILE* file = std::fopen((dump.path + ".raw").c_str(), "wb")) {
            std::fwrite(pixels.data(), 1, count * bpp, file);
            std::fclose(file);
        }
    }
    const double n = static_cast<double>(count);
    return fmt::format("{}mean=({:.6g},{:.6g},{:.6g},{:.6g}) nonzero={:.2f}% r_range=[{:.4g},{:.4g}]",
                       written ? "" : "WRITE FAILED ", sum[0] / n, sum[1] / n, sum[2] / n,
                       sum[3] / n, 100.0 * static_cast<double>(nonzero) / n, lo, hi);
}

bool ToRgba8(VkFormat format, VkImageAspectFlags aspect, u32 width, u32 height,
             std::span<const u8> pixels, std::vector<u8>& rgba) {
    const u32 bpp = PixelBytes(format, aspect);
    const size_t count = static_cast<size_t>(width) * height;
    if (bpp == 0 || count == 0 || pixels.size() < count * bpp) {
        return false;
    }
    rgba.resize(count * 4);
    for (size_t i = 0; i < count; ++i) {
        Rgba c;
        if (!Decode(format, aspect, pixels.data() + i * bpp, c)) {
            return false;
        }
        u8* o = rgba.data() + i * 4;
        o[0] = ToByte(c.r);
        o[1] = ToByte(c.g);
        o[2] = ToByte(c.b);
        o[3] = ToByte(c.a);
    }
    return true;
}

} // namespace Vulkan::Nvn::Debug
