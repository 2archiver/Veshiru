// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <filesystem>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <fmt/format.h>

#include "common/diag_image.h"
#include "common/diag_request.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_diag_textures.h"
#include "video_core/texture_cache/decode_bc.h"

namespace Vulkan::Nvn::DiagTextures {
namespace {

constexpr u32 MaxDimension = 4096;
constexpr size_t MaxCaptured = 256;
constexpr u32 NoSlot = ~0u;

struct FormatName {
    s32 value;
    const char* name;
};

constexpr FormatName kFormatNames[] = {
    {0x0, "NONE"},
    {0x1, "R8"},
    {0x2, "R8SN"},
    {0x3, "R8UI"},
    {0x4, "R8I"},
    {0x5, "R16F"},
    {0x6, "R16"},
    {0x7, "R16SN"},
    {0x8, "R16UI"},
    {0x9, "R16I"},
    {0xA, "R32F"},
    {0xB, "R32UI"},
    {0xC, "R32I"},
    {0xD, "RG8"},
    {0xE, "RG8SN"},
    {0xF, "RG8UI"},
    {0x10, "RG8I"},
    {0x11, "RG16F"},
    {0x12, "RG16"},
    {0x13, "RG16SN"},
    {0x14, "RG16UI"},
    {0x15, "RG16I"},
    {0x16, "RG32F"},
    {0x17, "RG32UI"},
    {0x18, "RG32I"},
    {0x19, "RGB8"},
    {0x1A, "RGB8SN"},
    {0x1B, "RGB8UI"},
    {0x1C, "RGB8I"},
    {0x1D, "RGB16F"},
    {0x1E, "RGB16"},
    {0x1F, "RGB16SN"},
    {0x20, "RGB16UI"},
    {0x21, "RGB16I"},
    {0x22, "RGB32F"},
    {0x23, "RGB32UI"},
    {0x24, "RGB32I"},
    {0x25, "RGBA8"},
    {0x26, "RGBA8SN"},
    {0x27, "RGBA8UI"},
    {0x28, "RGBA8I"},
    {0x29, "RGBA16F"},
    {0x2A, "RGBA16"},
    {0x2B, "RGBA16SN"},
    {0x2C, "RGBA16UI"},
    {0x2D, "RGBA16I"},
    {0x2E, "RGBA32F"},
    {0x2F, "RGBA32UI"},
    {0x30, "RGBA32I"},
    {0x31, "STENCIL8"},
    {0x32, "DEPTH16"},
    {0x33, "DEPTH24"},
    {0x34, "DEPTH32F"},
    {0x35, "DEPTH24_STENCIL8"},
    {0x36, "DEPTH32F_STENCIL8"},
    {0x37, "RGBX8_SRGB"},
    {0x38, "RGBA8_SRGB"},
    {0x39, "RGBA4"},
    {0x3A, "RGB5"},
    {0x3B, "RGB5A1"},
    {0x3C, "RGB565"},
    {0x3D, "RGB10A2"},
    {0x3E, "RGB10A2UI"},
    {0x3F, "R11G11B10F"},
    {0x40, "RGB9E5F"},
    {0x41, "RGB_DXT1"},
    {0x42, "RGBA_DXT1"},
    {0x43, "RGBA_DXT3"},
    {0x44, "RGBA_DXT5"},
    {0x45, "RGB_DXT1_SRGB"},
    {0x46, "RGBA_DXT1_SRGB"},
    {0x47, "RGBA_DXT3_SRGB"},
    {0x48, "RGBA_DXT5_SRGB"},
    {0x49, "RGTC1_UNORM"},
    {0x4A, "RGTC1_SNORM"},
    {0x4B, "RGTC2_UNORM"},
    {0x4C, "RGTC2_SNORM"},
    {0x4D, "BPTC_UNORM"},
    {0x4E, "BPTC_UNORM_SRGB"},
    {0x4F, "BPTC_SFLOAT"},
    {0x50, "BPTC_UFLOAT"},
    {0x51, "R8_UI2F"},
    {0x52, "R8_I2F"},
    {0x53, "R16_UI2F"},
    {0x54, "R16_I2F"},
    {0x55, "R32_UI2F"},
    {0x56, "R32_I2F"},
    {0x57, "RG8_UI2F"},
    {0x58, "RG8_I2F"},
    {0x59, "RG16_UI2F"},
    {0x5A, "RG16_I2F"},
    {0x5B, "RG32_UI2F"},
    {0x5C, "RG32_I2F"},
    {0x5D, "RGB8_UI2F"},
    {0x5E, "RGB8_I2F"},
    {0x5F, "RGB16_UI2F"},
    {0x60, "RGB16_I2F"},
    {0x61, "RGB32_UI2F"},
    {0x62, "RGB32_I2F"},
    {0x63, "RGBA8_UI2F"},
    {0x64, "RGBA8_I2F"},
    {0x65, "RGBA16_UI2F"},
    {0x66, "RGBA16_I2F"},
    {0x67, "RGBA32_UI2F"},
    {0x68, "RGBA32_I2F"},
    {0x69, "RGB10A2SN"},
    {0x6A, "RGB10A2I"},
    {0x6B, "RGB10A2_UI2F"},
    {0x6C, "RGB10A2_I2F"},
    {0x6D, "RGBX8"},
    {0x6E, "RGBX8SN"},
    {0x6F, "RGBX8UI"},
    {0x70, "RGBX8I"},
    {0x71, "RGBX16F"},
    {0x72, "RGBX16"},
    {0x73, "RGBX16SN"},
    {0x74, "RGBX16UI"},
    {0x75, "RGBX16I"},
    {0x76, "RGBX32F"},
    {0x77, "RGBX32UI"},
    {0x78, "RGBX32I"},
    {0x79, "RGBA_ASTC_4x4"},
    {0x7A, "RGBA_ASTC_5x4"},
    {0x7B, "RGBA_ASTC_5x5"},
    {0x7C, "RGBA_ASTC_6x5"},
    {0x7D, "RGBA_ASTC_6x6"},
    {0x7E, "RGBA_ASTC_8x5"},
    {0x7F, "RGBA_ASTC_8x6"},
    {0x80, "RGBA_ASTC_8x8"},
    {0x81, "RGBA_ASTC_10x5"},
    {0x82, "RGBA_ASTC_10x6"},
    {0x83, "RGBA_ASTC_10x8"},
    {0x84, "RGBA_ASTC_10x10"},
    {0x85, "RGBA_ASTC_12x10"},
    {0x86, "RGBA_ASTC_12x12"},
    {0x87, "RGBA_ASTC_4x4_SRGB"},
    {0x88, "RGBA_ASTC_5x4_SRGB"},
    {0x89, "RGBA_ASTC_5x5_SRGB"},
    {0x8A, "RGBA_ASTC_6x5_SRGB"},
    {0x8B, "RGBA_ASTC_6x6_SRGB"},
    {0x8C, "RGBA_ASTC_8x5_SRGB"},
    {0x8D, "RGBA_ASTC_8x6_SRGB"},
    {0x8E, "RGBA_ASTC_8x8_SRGB"},
    {0x8F, "RGBA_ASTC_10x5_SRGB"},
    {0x90, "RGBA_ASTC_10x6_SRGB"},
    {0x91, "RGBA_ASTC_10x8_SRGB"},
    {0x92, "RGBA_ASTC_10x10_SRGB"},
    {0x93, "RGBA_ASTC_12x10_SRGB"},
    {0x94, "RGBA_ASTC_12x12_SRGB"},
    {0x95, "BGR565"},
    {0x96, "BGR5"},
    {0x97, "BGR5A1"},
    {0x98, "A1BGR5"},
    {0x99, "BGRX8"},
    {0x9A, "BGRA8"},
    {0x9B, "BGRX8_SRGB"},
    {0x9C, "BGRA8_SRGB"},
};

const char* WrapName(u32 wrap) {
    static constexpr const char* names[] = {
        "clamp",           "repeat",       "mirror_clamp",          "mirror_clamp_to_edge",
        "mirror_clamp_to_border", "clamp_to_border", "mirrored_repeat", "clamp_to_edge",
    };
    return wrap < std::size(names) ? names[wrap] : "unknown";
}

struct SamplerInfo {
    const char* mag{};
    const char* min{};
    const char* mip{};
    f32 lod_min{};
    f32 lod_max{};
    f32 bias{};
    f32 aniso{};
    const char* wrap[3]{};
    bool compare{};
};

struct Entry {
    u32 draw{};
    s32 stage{-1};
    s32 slot{-1};
    s32 nvn_format{};
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 levels{};
    u32 layers{};
    u64 address{};
    std::optional<SamplerInfo> sampler;
    std::string note;
    u32 capture_slot{NoSlot};
};

/// One PNG of the request.
struct Slot {
    std::string file;
    std::string note;
    bool failed{false};
};

struct State {
    std::mutex mutex;
    bool valid{false};
    u32 epoch{};
    std::filesystem::path directory;
    u32 draws{};
    u32 current_draw{};
    std::vector<Entry> entries;
    std::vector<Slot> slots;
    std::map<std::tuple<u64, s32, u32, u32>, u32> dedup;
};

State& GetState() {
    static State state;
    return state;
}

std::string Number(f32 value) {
    return std::isfinite(value) ? fmt::format("{}", value) : std::string{"null"};
}

std::string Quoted(std::string_view text) {
    return "\"" + Common::DiagRequest::JsonEscape(text) + "\"";
}

void AppendNote(std::string& note, std::string_view text) {
    if (!note.empty()) {
        note += "; ";
    }
    note += text;
}

/// Rewrites textures-nvn.json (state mutex held).
void WriteIndex(const State& state) {
    std::string json = fmt::format("{{\"path\": \"nvn\", \"draws\": {}, \"textures\": [", state.draws);
    bool first = true;
    for (const Entry& e : state.entries) {
        const Slot* slot = e.capture_slot < state.slots.size() ? &state.slots[e.capture_slot] : nullptr;
        std::string note = e.note;
        if (slot != nullptr && !slot->note.empty()) {
            AppendNote(note, slot->note);
        }
        json += first ? "\n" : ",\n";
        first = false;
        json += fmt::format("  {{\"file\": {}, \"draw\": {}, \"stage\": {}, \"slot\": {}, \"format\": {}, "
                            "\"width\": {}, \"height\": {}, \"depth\": {}, \"levels\": {}, \"layers\": {}, "
                            "\"address\": \"{:#x}\", \"sampler\": ",
                            slot != nullptr && !slot->failed ? Quoted(slot->file) : std::string{"null"},
                            e.draw, e.stage >= 0 ? std::to_string(e.stage) : std::string{"null"},
                            e.slot >= 0 ? std::to_string(e.slot) : std::string{"null"},
                            Quoted(NvnFormatName(e.nvn_format)), e.width, e.height, e.depth, e.levels,
                            e.layers, e.address);
        if (e.sampler) {
            const SamplerInfo& s = *e.sampler;
            json += fmt::format("{{\"mag\": \"{}\", \"min\": \"{}\", \"mip\": \"{}\", \"lod_min\": {}, "
                                "\"lod_max\": {}, \"bias\": {}, \"aniso\": {}, \"wrap\": [\"{}\", \"{}\", \"{}\"], "
                                "\"compare\": {}}}",
                                s.mag, s.min, s.mip, Number(s.lod_min), Number(s.lod_max), Number(s.bias),
                                Number(s.aniso), s.wrap[0], s.wrap[1], s.wrap[2], s.compare);
        } else {
            json += "null";
        }
        json += fmt::format(", \"note\": {}}}", note.empty() ? std::string{"null"} : Quoted(note));
    }
    json += first ? "]}\n" : "\n]}\n";
    Common::DiagRequest::WriteTextFile(state.directory / "textures-nvn.json", json);
}

} // namespace

std::string NvnFormatName(s32 format) {
    for (const FormatName& entry : kFormatNames) {
        if (entry.value == format) {
            return entry.name;
        }
    }
    return std::to_string(format);
}

bool BeginDraw() {
    Common::DiagRequest::Budget& budget = Common::DiagRequest::TextureBudget();
    std::filesystem::path directory;
    u32 ordinal = 0;
    if (!budget.Claim(directory, &ordinal)) {
        return false;
    }
    const u32 epoch = budget.Epoch();
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (!state.valid || state.epoch != epoch) {
        state.valid = true;
        state.epoch = epoch;
        state.draws = 0;
        state.entries.clear();
        state.slots.clear();
        state.dedup.clear();
    }
    state.directory = std::move(directory);
    state.current_draw = ordinal;
    state.draws = ordinal + 1;
    return true;
}

Job Note(const Use& use) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    Job job{state.epoch, NoSlot, false};
    Entry entry;
    entry.draw = state.current_draw;
    entry.stage = use.stage;
    entry.slot = use.slot;
    entry.nvn_format = use.nvn_format;
    entry.width = use.width;
    entry.height = use.height;
    entry.depth = use.depth;
    entry.levels = use.levels;
    entry.layers = use.layers;
    entry.address = use.address;
    entry.note = use.note;
    if (use.sampler != nullptr) {
        const VideoCommon::Nvn::SamplerDesc& d = *use.sampler;
        SamplerInfo s;
        s.mag = d.mag_filter == 0 ? "nearest" : "linear";
        s.min = (d.min_filter & 1) != 0 ? "linear" : "nearest";
        s.mip = d.min_filter < 2 ? "none" : d.min_filter >= 4 ? "linear" : "nearest";
        s.lod_min = d.lod_min;
        s.lod_max = d.lod_max;
        s.bias = d.lod_bias;
        s.aniso = d.max_anisotropy;
        s.wrap[0] = WrapName(d.wrap_s);
        s.wrap[1] = WrapName(d.wrap_t);
        s.wrap[2] = WrapName(d.wrap_r);
        s.compare = d.compare_mode == 1;
        entry.sampler = s;
    }
    const auto key = std::make_tuple(use.address, use.nvn_format, use.width, use.height);
    if (const auto it = state.dedup.find(key); it != state.dedup.end()) {
        job.slot = it->second;
    } else if (use.width > MaxDimension || use.height > MaxDimension) {
        AppendNote(entry.note, "larger than 4096: not captured");
    } else if (state.slots.size() >= MaxCaptured) {
        AppendNote(entry.note, "limit of 256 captured textures reached");
    } else {
        job.slot = static_cast<u32>(state.slots.size());
        job.capture = true;
        state.slots.push_back({fmt::format("tex-{:03}-{:016x}.png", job.slot, use.address), {}, false});
        state.dedup.emplace(key, job.slot);
    }
    entry.capture_slot = job.slot;
    state.entries.push_back(std::move(entry));
    return job;
}

void EndDraw() {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    WriteIndex(state);
    if (Common::DiagRequest::TextureBudget().Remaining() == 0) {
        LOG_INFO(Render_Vulkan, "[diag] textures: nvn: {} draws, {} texture uses, {} captured -> {}",
                 state.draws, state.entries.size(), state.slots.size(), state.directory.string());
    }
}

void Skip(const Job& job, const std::string& reason) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (!state.valid || state.epoch != job.epoch || job.slot >= state.slots.size()) {
        return;
    }
    state.slots[job.slot].failed = true;
    AppendNote(state.slots[job.slot].note, reason);
}

void Complete(const Readback& readback, std::span<const u8> pixels) {
    VkFormat format = readback.format;
    std::vector<u8> decoded;
    if (readback.bcn) {
        const u32 bytes = VideoCommon::ConvertedBytesPerBlock(readback.pixel_format);
        decoded.resize(static_cast<size_t>(readback.width) * readback.height * bytes);
        VideoCommon::BufferImageCopy copy{};
        copy.image_subresource.num_layers = 1;
        copy.image_extent = {readback.width, readback.height, 1};
        copy.buffer_row_length = (readback.width + 3) / 4 * 4; // DecompressBCn steps rows by it
        VideoCommon::DecompressBCn(pixels, decoded, copy, readback.pixel_format);
        pixels = decoded;
        format = bytes == 1   ? VK_FORMAT_R8_UNORM
                 : bytes == 2 ? VK_FORMAT_R8G8_UNORM
                 : bytes == 8 ? VK_FORMAT_R16G16B16A16_SFLOAT
                              : VK_FORMAT_R8G8B8A8_UNORM;
        if (readback.luminance && (bytes == 1 || bytes == 2)) {
            std::vector<u8> expanded(decoded.size() / bytes * 4);
            for (size_t i = 0; i < decoded.size() / bytes; ++i) {
                const u8 l = decoded[i * bytes];
                expanded[i * 4] = l;
                expanded[i * 4 + 1] = l;
                expanded[i * 4 + 2] = l;
                expanded[i * 4 + 3] = bytes == 2 ? decoded[i * bytes + 1] : u8{255};
            }
            decoded = std::move(expanded);
            pixels = decoded;
            format = VK_FORMAT_R8G8B8A8_UNORM;
        }
    }
    std::vector<u8> rgba;
    const bool converted =
        Debug::ToRgba8(format, readback.aspect, readback.width, readback.height, pixels, rgba);

    State& state = GetState();
    std::filesystem::path path;
    {
        std::scoped_lock lock{state.mutex};
        if (!state.valid || state.epoch != readback.epoch || readback.slot >= state.slots.size()) {
            return;
        }
        path = state.directory / state.slots[readback.slot].file;
    }
    std::string failure;
    if (!converted) {
        failure = fmt::format("host format {} cannot be converted to RGBA8: not captured",
                              static_cast<s32>(format));
    } else if (!Common::DiagRequest::WritePng(path, readback.width, readback.height, rgba)) {
        failure = "PNG could not be written";
    }
    std::scoped_lock lock{state.mutex};
    if (!state.valid || state.epoch != readback.epoch || readback.slot >= state.slots.size()) {
        return;
    }
    Slot& slot = state.slots[readback.slot];
    if (!failure.empty()) {
        slot.failed = true;
        AppendNote(slot.note, failure);
    } else if (readback.bcn) {
        AppendNote(slot.note, "block-compressed: saved as decoded RGBA");
    } else if (readback.aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
        AppendNote(slot.note, "depth: saved as grey");
    }
    WriteIndex(state);
}

} // namespace Vulkan::Nvn::DiagTextures
