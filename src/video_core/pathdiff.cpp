// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/pathdiff.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>

#include <fmt/format.h>

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/program_override.h"

namespace VideoCommon::PathDiff {

namespace detail {
std::atomic<bool> g_enabled{false};
} // namespace detail

namespace {

std::optional<u64> ParseHex(std::string_view text) {
    if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
    }
    if (text.empty() || text.size() > 16) {
        return std::nullopt;
    }
    u64 value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, 16);
    if (ec != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<u64> ParseDecimal(std::string_view text) {
    u64 value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (text.empty() || ec != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

struct State {
    std::mutex mutex;
    std::optional<Spec> spec;
    std::unordered_map<u64, u64> code_keys;
    std::array<std::filesystem::path, 2> directories;
    std::array<u64, 2> written{};
};

State& GetState() {
    static State state;
    return state;
}

std::optional<Spec> SpecFromEnvironment() {
    const char* text = std::getenv("EDEN_NX2_PATHDIFF");
    if (text == nullptr || *text == '\0') {
        return std::nullopt;
    }
    std::string error;
    std::optional<Spec> spec = ParseSpec(text, error);
    if (!spec) {
        // Runs before the log is set up: stderr.
        std::fprintf(stderr, "[pathdiff] EDEN_NX2_PATHDIFF=%s: %s; path diff off\n", text,
                     error.c_str());
        return std::nullopt;
    }
    if (const char* chase = std::getenv("EDEN_NX2_PATHDIFF_CHASE")) {
        if (const auto bytes = ParseDecimal(chase); bytes && *bytes <= (u64{1} << 24)) {
            spec->chase_bytes = static_cast<u32>(*bytes & ~u64{3});
        }
    }
    return spec;
}

/// Reads the environment once, before any path asks Enabled().
[[maybe_unused]] const bool g_initialized = [] {
    State& state = GetState();
    state.spec = SpecFromEnvironment();
    detail::g_enabled.store(state.spec.has_value(), std::memory_order_relaxed);
    return true;
}();

void AppendBlobRef(std::string& out, u64& offset, bool present, u64 size) {
    if (!present) {
        out += "null";
        return;
    }
    out += fmt::format("[{},{}]", offset, size);
    offset += size;
}

std::string Hex(u64 value) {
    return fmt::format("\"{:016x}\"", value);
}

std::string Float(float value) {
    if (value != value) {
        return "\"nan\"";
    }
    if (value == std::numeric_limits<float>::infinity()) {
        return "\"inf\"";
    }
    if (value == -std::numeric_limits<float>::infinity()) {
        return "\"-inf\"";
    }
    return fmt::format("{}", value);
}

std::string Quote(std::string_view text) {
    std::string out{"\""};
    for (const char c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                out += c;
            }
        }
    }
    out += '"';
    return out;
}

u32 NormalizedSwizzle(u32 value, u32 component) {
    // VK_COMPONENT_SWIZZLE_IDENTITY is the component itself (R = 3 ... A = 6).
    return value == 0 ? component + 3 : value;
}

} // Anonymous namespace

std::string_view PathName(Path path) noexcept {
    return path == Path::Emulated ? "emulated" : "hle";
}

std::optional<Spec> ParseSpec(std::string_view text, std::string& error) {
    Spec spec;
    const size_t first_colon = text.find(':');
    if (first_colon == std::string_view::npos) {
        error = "expected <hash>[,<hash>...]:<present>[:<count>]";
        return std::nullopt;
    }
    std::string_view hashes = text.substr(0, first_colon);
    std::string_view rest = text.substr(first_colon + 1);
    while (!hashes.empty()) {
        const size_t comma = hashes.find(',');
        const std::string_view item = hashes.substr(0, comma);
        const auto value = ParseHex(item);
        if (!value || *value == 0) {
            error = fmt::format("'{}' is not a program hash", item);
            return std::nullopt;
        }
        spec.hashes.push_back(*value);
        hashes = comma == std::string_view::npos ? std::string_view{} : hashes.substr(comma + 1);
    }
    if (spec.hashes.empty()) {
        error = "no program hash";
        return std::nullopt;
    }
    const size_t second_colon = rest.find(':');
    const auto present = ParseDecimal(rest.substr(0, second_colon));
    if (!present || *present == 0) {
        error = "the present must be a number from 1";
        return std::nullopt;
    }
    spec.present = *present;
    if (second_colon != std::string_view::npos) {
        const auto count = ParseDecimal(rest.substr(second_colon + 1));
        if (!count || *count == 0 || *count > 4096) {
            error = "the count must be 1..4096";
            return std::nullopt;
        }
        spec.count = static_cast<u32>(*count);
    }
    return spec;
}

const Spec& Config() {
    static const Spec none{};
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    return state.spec ? *state.spec : none;
}

void SetConfigForTest(std::optional<Spec> spec) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.spec = std::move(spec);
    state.directories = {};
    state.written = {};
    detail::g_enabled.store(state.spec.has_value(), std::memory_order_relaxed);
}

u64 CodeKey(std::span<const u8> code, u32 header_bytes) {
    using Shader::Ampere::Instruction;
    using Capture = Shader::Ampere::ProgramCaptureState;
    std::vector<Instruction> native;
    Capture capture;
    const auto is_mov_rz = [](const Instruction& instruction, u32 immediate) {
        return instruction.words[0] == 0x00ff7802U && instruction.words[1] == immediate;
    };
    const size_t limit = (std::min<size_t>)(code.size() / sizeof(Instruction),
                                            Capture::MaximumInstructions);
    native.reserve(limit);
    for (u32 index = 0; index < limit; ++index) {
        Instruction instruction;
        std::memcpy(&instruction, code.data() + size_t{index} * sizeof(Instruction),
                    sizeof(instruction));
        // The driver's size footer (MOV RZ, 0x12345678; MOV RZ, <inst_bytes>) ends the program.
        if (index != 0 && is_mov_rz(native.back(), 0x12345678U) &&
            is_mov_rz(instruction,
                      header_bytes + (index + 7) * static_cast<u32>(sizeof(Instruction)))) {
            native.pop_back();
            break;
        }
        native.push_back(instruction);
        const auto status = capture.Observe(instruction, index);
        if (status == Capture::Result::InvalidTarget || status == Capture::Result::Complete) {
            break;
        }
    }
    return Shader::Ampere::NativeProgramKey(native);
}

void NoteCodeKey(u64 stage_hash, u64 code_key) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.code_keys[stage_hash] = code_key;
}

u64 CodeKeyOf(u64 stage_hash) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const auto it = state.code_keys.find(stage_hash);
    return it != state.code_keys.end() ? it->second : 0;
}

std::optional<u32> Selector::Claim(u64 present, u64 stage_hash, u64 code_key) {
    const Spec& spec = Config();
    if (present < spec.present || Done()) {
        return std::nullopt;
    }
    const bool wanted = std::ranges::any_of(spec.hashes, [&](u64 hash) {
        return hash == stage_hash || (code_key != 0 && hash == code_key);
    });
    if (!wanted) {
        return std::nullopt;
    }
    if (present != current_present) {
        current_present = present;
        ordinals.clear();
    }
    const u32 ordinal = ordinals[code_key != 0 ? code_key : stage_hash]++;
    ++taken;
    return ordinal;
}

bool Selector::Done() const noexcept {
    return taken >= Config().count;
}

std::vector<Pointer> FindPointers(std::span<const u8> bytes,
                                  const std::function<bool(u64)>& resolves, u32 limit) {
    std::vector<Pointer> result;
    for (size_t offset = 0; offset + sizeof(u64) <= bytes.size() && result.size() < limit;
         offset += sizeof(u64)) {
        u64 value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        if (value != 0 && resolves(value)) {
            result.push_back({static_cast<u32>(offset), value});
        }
    }
    return result;
}

std::string Source(u32 cbuf) {
    return fmt::format("c{}", cbuf);
}

std::string Source(u32 cbuf, u32 offset) {
    return fmt::format("c{}+{:#x}", cbuf, offset);
}

std::vector<u8> Blob(const Capture& capture) {
    std::vector<u8> blob;
    for (const Range& range : capture.ranges) {
        blob.insert(blob.end(), range.before.begin(), range.before.end());
        if (range.has_after) {
            blob.insert(blob.end(), range.after.begin(), range.after.end());
        }
    }
    return blob;
}

std::string ToJson(const Capture& capture, std::string_view blob_name) {
    std::string out = fmt::format(
        "{{\"format\":\"fukko-pathdiff/1\",\"path\":\"{}\",\"present\":{},\"ordinal\":{},"
        "\"kind\":{},\"stage\":{},\"hash\":{},\"code_key\":{},\"blob\":{},",
        PathName(capture.path), capture.present, capture.ordinal, Quote(capture.kind),
        Quote(capture.stage), Hex(capture.hash), Hex(capture.code_key), Quote(blob_name));
    out += fmt::format("\"groups\":[{},{},{}],\"indirect\":{},\"block\":[{},{},{}],",
                       capture.groups[0], capture.groups[1], capture.groups[2], capture.indirect,
                       capture.block[0], capture.block[1], capture.block[2]);
    if (capture.draw) {
        const Draw& d = *capture.draw;
        out += fmt::format("\"draw\":{{\"indexed\":{},\"indirect\":{},\"count\":{},"
                           "\"instances\":{},\"first\":{},\"base_vertex\":{},"
                           "\"base_instance\":{},\"topology\":{}}},",
                           d.indexed, d.indirect, d.count, d.instances, d.first, d.base_vertex,
                           d.base_instance, d.topology);
    } else {
        out += "\"draw\":null,";
    }
    out += fmt::format("\"raw_global\":{},\"ranges\":[", capture.raw_global);
    u64 offset = 0;
    for (size_t i = 0; i < capture.ranges.size(); ++i) {
        const Range& r = capture.ranges[i];
        out += fmt::format("{}{{\"role\":{},\"index\":{},\"source\":{},\"address\":\"{:#x}\","
                           "\"bound_size\":{},\"size\":{},\"written\":{},\"view\":{},\"before\":",
                           i != 0 ? "," : "", Quote(r.role), r.index, Quote(r.source), r.address,
                           r.bound_size, r.before.size(), r.written, Quote(r.view));
        AppendBlobRef(out, offset, true, r.before.size());
        out += ",\"after\":";
        AppendBlobRef(out, offset, r.has_after, r.after.size());
        out += '}';
    }
    out += "],\"textures\":[";
    for (size_t i = 0; i < capture.textures.size(); ++i) {
        const Texture& t = capture.textures[i];
        out += fmt::format(
            "{}{{\"source\":{},\"index\":{},\"kind\":\"{}\",\"resolved\":{},\"handle\":\"{:#x}\","
            "\"width\":{},\"height\":{},\"depth\":{},\"levels\":{},\"layers\":{},\"format\":{},"
            "\"view_type\":{},\"base_level\":{},\"view_levels\":{},\"base_layer\":{},"
            "\"view_layers\":{},\"swizzle\":[{},{},{},{}],\"sampler\":",
            i != 0 ? "," : "", Quote(t.source), t.index, t.image ? "image" : "sampled", t.resolved,
            t.handle, t.width, t.height, t.depth, t.levels, t.layers, t.format, t.view_type,
            t.base_level, t.view_levels, t.base_layer, t.view_layers,
            NormalizedSwizzle(t.swizzle[0], 0), NormalizedSwizzle(t.swizzle[1], 1),
            NormalizedSwizzle(t.swizzle[2], 2), NormalizedSwizzle(t.swizzle[3], 3));
        if (t.sampler) {
            const Sampler& s = *t.sampler;
            out += fmt::format(
                "{{\"mag_filter\":{},\"min_filter\":{},\"mipmap_mode\":{},\"address\":[{},{},{}],"
                "\"lod_bias\":{},\"min_lod\":{},\"max_lod\":{},\"anisotropy\":{},"
                "\"max_anisotropy\":{},\"compare\":{},\"compare_op\":{},\"border_color\":{},"
                "\"border\":[{},{},{},{}],\"reduction\":{}}}",
                s.mag_filter, s.min_filter, s.mipmap_mode, s.address[0], s.address[1],
                s.address[2], Float(s.lod_bias), Float(s.min_lod), Float(s.max_lod), s.anisotropy,
                Float(s.max_anisotropy), s.compare, s.compare_op, s.border_color,
                Float(s.border[0]), Float(s.border[1]), Float(s.border[2]), Float(s.border[3]),
                s.reduction);
        } else {
            out += "null";
        }
        out += '}';
    }
    out += "],\"path_specific\":[";
    for (size_t i = 0; i < capture.path_specific.size(); ++i) {
        const PathSpecific& p = capture.path_specific[i];
        out += fmt::format("{}{{\"cbuf\":{},\"offset\":{},\"size\":{},\"why\":{}}}",
                           i != 0 ? "," : "", p.cbuf, p.offset, p.size, Quote(p.why));
    }
    out += "],\"notes\":[";
    for (size_t i = 0; i < capture.notes.size(); ++i) {
        out += fmt::format("{}{}", i != 0 ? "," : "", Quote(capture.notes[i]));
    }
    out += "]}\n";
    return out;
}

std::filesystem::path Directory(Path path) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    return state.directories[static_cast<size_t>(path)];
}

std::filesystem::path Write(const Capture& capture) {
    State& state = GetState();
    std::filesystem::path directory;
    u64 seq{};
    {
        std::scoped_lock lock{state.mutex};
        std::filesystem::path& dir = state.directories[static_cast<size_t>(capture.path)];
        if (dir.empty()) {
            std::filesystem::path root;
            if (const char* override_dir = std::getenv("EDEN_NX2_PATHDIFF_DIR");
                override_dir != nullptr && *override_dir != '\0') {
                root = override_dir;
            } else {
                root = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "pathdiff";
            }
            const std::time_t now = std::time(nullptr);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &now);
#else
            localtime_r(&now, &tm);
#endif
            char stamp[32]{};
            std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
            dir = root / fmt::format("{}-{}", PathName(capture.path), stamp);
            std::error_code error;
            std::filesystem::create_directories(dir, error);
            if (error) {
                LOG_ERROR(HW_GPU, "[pathdiff] cannot create {}: {}", dir.string(), error.message());
            } else {
                LOG_INFO(HW_GPU, "[pathdiff] {} captures go to {}", PathName(capture.path),
                         dir.string());
            }
        }
        directory = dir;
        seq = state.written[static_cast<size_t>(capture.path)]++;
    }
    const std::string base = fmt::format("{:04}-{:016x}-p{}-o{}", seq, capture.code_key,
                                         capture.present, capture.ordinal);
    const std::string blob_name = base + ".bin";
    const std::vector<u8> blob = Blob(capture);
    const std::string json = ToJson(capture, blob_name);
    const std::filesystem::path json_path = directory / (base + ".json");
    // The blob first and the JSON (which names it) only after it: a JSON file means a complete
    // capture.
    const auto write_file = [](const std::filesystem::path& path, const char* data, size_t size) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(data, static_cast<std::streamsize>(size));
        file.close();
        if (!file) {
            std::error_code error;
            std::filesystem::remove(path, error);
            LOG_ERROR(HW_GPU, "[pathdiff] cannot write {}", path.string());
            return false;
        }
        return true;
    };
    if (!write_file(directory / blob_name, reinterpret_cast<const char*>(blob.data()),
                    blob.size()) ||
        !write_file(json_path, json.data(), json.size())) {
        return {};
    }
    LOG_INFO(HW_GPU,
             "[pathdiff] {} capture {}: prog {:016x} (code {:016x}) {} present {} ordinal {}: {} "
             "ranges, {} textures, {} bytes -> {}",
             PathName(capture.path), seq, capture.hash, capture.code_key, capture.kind,
             capture.present, capture.ordinal, capture.ranges.size(), capture.textures.size(),
             blob.size(), json_path.string());
    return json_path;
}

} // namespace VideoCommon::PathDiff
