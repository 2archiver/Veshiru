// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <system_error>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/zstd_compression.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_disk.h"

namespace Vulkan::Nvn::Disk {
namespace {

static_assert(std::endian::native == std::endian::little, "the cache format is little-endian");
static_assert(std::is_trivially_copyable_v<GraphicsFixedState>);

constexpr u32 MAX_REVISION_BYTES = 256;
constexpr u32 MAX_KEY_BYTES = 512;
constexpr size_t KEY_TAG_DIGITS = 16;
constexpr std::string_view FILE_STEM = "nvn_pipelines";
constexpr u32 MAX_ISSUE_BYTES = 4096;
constexpr u32 MAX_BINDINGS = 1u << 16;
constexpr u32 MAX_QUERIES = 1u << 20;
constexpr u32 MAX_WORDS = 1u << 24;
constexpr u32 MAX_HANDLE_PROGRAM_WORDS = 1u << 16;
constexpr u32 SPIRV_MAGIC = 0x07230203;

class Writer {
public:
    std::vector<u8> data;

    void U8(u8 value) {
        data.push_back(value);
    }
    void U32(u32 value) {
        Raw(&value, sizeof(value));
    }
    void U64(u64 value) {
        Raw(&value, sizeof(value));
    }
    void Raw(const void* bytes, size_t size) {
        const auto* const first = static_cast<const u8*>(bytes);
        data.insert(data.end(), first, first + size);
    }
    void Str(const std::string& text) {
        U32(static_cast<u32>(text.size()));
        Raw(text.data(), text.size());
    }
    void Words(std::span<const u32> words) {
        U32(static_cast<u32>(words.size()));
        Raw(words.data(), words.size() * sizeof(u32));
    }
};

class Reader {
public:
    explicit Reader(std::span<const u8> bytes) : data{bytes} {}

    [[nodiscard]] bool Ok() const noexcept {
        return ok;
    }
    [[nodiscard]] size_t Remaining() const noexcept {
        return data.size() - pos;
    }
    [[nodiscard]] bool AtEnd() const noexcept {
        return pos == data.size();
    }
    u8 U8() {
        u8 value{};
        Raw(&value, sizeof(value));
        return value;
    }
    u32 U32() {
        u32 value{};
        Raw(&value, sizeof(value));
        return value;
    }
    u64 U64() {
        u64 value{};
        Raw(&value, sizeof(value));
        return value;
    }
    void Raw(void* out, size_t size) {
        if (!ok || size > Remaining()) {
            ok = false;
            std::memset(out, 0, size);
            return;
        }
        std::memcpy(out, data.data() + pos, size);
        pos += size;
    }
    std::span<const u8> View(size_t size) {
        if (!ok || size > Remaining()) {
            ok = false;
            return {};
        }
        const auto view = data.subspan(pos, size);
        pos += size;
        return view;
    }
    std::string Str(u32 max_size) {
        const u32 size = U32();
        if (size > max_size) {
            ok = false;
            return {};
        }
        const auto view = View(size);
        return ok ? std::string(reinterpret_cast<const char*>(view.data()), view.size())
                  : std::string{};
    }
    bool Words(std::vector<u32>& out, u32 max_count) {
        const u32 count = U32();
        if (!ok || count > max_count || size_t{count} * sizeof(u32) > Remaining()) {
            ok = false;
            return false;
        }
        out.resize(count);
        Raw(out.data(), size_t{count} * sizeof(u32));
        return ok;
    }
    /// An enum stored as u32 that must not exceed `last`.
    template <typename E>
    E Enum(u32 last) {
        const u32 value = U32();
        if (value > last) {
            ok = false;
            return E{};
        }
        return static_cast<E>(value);
    }

private:
    std::span<const u8> data;
    size_t pos{};
    bool ok{true};
};

constexpr std::array<u32, 256> MakeCrcTable() {
    std::array<u32, 256> table{};
    for (u32 i = 0; i < 256; ++i) {
        u32 c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        }
        table[i] = c;
    }
    return table;
}
constexpr std::array<u32, 256> CRC_TABLE = MakeCrcTable();

void WriteHandleRef(Writer& w, const VC::HandleRef& ref) {
    w.U32(static_cast<u32>(ref.source));
    w.U32(ref.slot);
    w.U32(ref.bank);
    w.U32(ref.byte_offset);
}

VC::HandleRef ReadHandleRef(Reader& r) {
    VC::HandleRef ref;
    ref.source = r.Enum<VC::HandleSource>(static_cast<u32>(VC::HandleSource::Unsupported));
    ref.slot = r.U32();
    ref.bank = r.U32();
    ref.byte_offset = r.U32();
    return ref;
}

void WriteLayout(Writer& w, const VC::DescriptorLayout& layout) {
    w.U32(layout.binding_count);
    w.U8(layout.uses_rescaling_uniform ? 1 : 0);
    w.U8(layout.uses_render_area ? 1 : 0);
    // 4: raw_stores_written_through (records from before it read 0: the stores are not).
    // 8: requires_guest_warp (records from before it read 0).
    w.U8((layout.uses_raw_global_memory ? 1 : 0) | (layout.stores_raw_global_memory ? 2 : 0) |
         (layout.raw_stores_written_through ? 4 : 0) | (layout.requires_guest_warp ? 8 : 0) |
         (layout.uses_bindless_table ? 16 : 0));
    w.U32(layout.unsupported_count);
    w.U32(static_cast<u32>(layout.bindings.size()));
    for (const VC::DescriptorBinding& b : layout.bindings) {
        w.U32(b.binding);
        w.U32(static_cast<u32>(b.kind));
        w.U32(static_cast<u32>(b.role));
        w.U32(b.count);
        w.U32(b.slot);
        w.U32(static_cast<u32>(b.storage_source));
        w.U32(b.storage_bank);
        w.U32(b.storage_offset);
        w.U32(b.storage_index);
        w.U32(static_cast<u32>(b.stage));
        w.U32(b.stage_mask);
        WriteHandleRef(w, b.texture);
        WriteHandleRef(w, b.sampler);
        w.Words(b.handle_program);
        w.U32(static_cast<u32>(b.type));
        w.U32(static_cast<u32>(b.image_format));
        const u8 flags = (b.is_depth ? 1 : 0) | (b.is_multisample ? 2 : 0) |
                         (b.is_integer ? 4 : 0) | (b.is_written ? 8 : 0) | (b.is_read ? 16 : 0) |
                         (b.supported ? 32 : 0) | (b.is_atomic_target ? 64 : 0) |
                         (b.is_raw_global_table ? 128 : 0);
        w.U8(flags);
        w.Str(b.issue);
    }
}

bool ReadLayout(Reader& r, VC::DescriptorLayout& out) {
    out = {};
    out.binding_count = r.U32();
    out.uses_rescaling_uniform = r.U8() != 0;
    out.uses_render_area = r.U8() != 0;
    const u8 raw_global = r.U8();
    out.uses_raw_global_memory = (raw_global & 1) != 0;
    out.stores_raw_global_memory = (raw_global & 2) != 0;
    out.raw_stores_written_through = (raw_global & 4) != 0;
    out.requires_guest_warp = (raw_global & 8) != 0;
    out.uses_bindless_table = (raw_global & 16) != 0;
    out.unsupported_count = r.U32();
    const u32 count = r.U32();
    // Every binding takes at least 78 bytes: a count the input cannot hold is garbage.
    if (!r.Ok() || count > MAX_BINDINGS || size_t{count} * 78 > r.Remaining()) {
        return false;
    }
    out.bindings.reserve(count);
    for (u32 i = 0; i < count && r.Ok(); ++i) {
        VC::DescriptorBinding b;
        b.binding = r.U32();
        // DescriptorKind values are VkDescriptorType numbers: 1, 3, 4, 5, 6, 7.
        const u32 kind = r.U32();
        if (kind != 1 && (kind < 3 || kind > 7)) {
            return false;
        }
        b.kind = static_cast<VC::DescriptorKind>(kind);
        b.role = r.Enum<VC::DescriptorRole>(static_cast<u32>(VC::DescriptorRole::TexelBuffer));
        b.count = r.U32();
        b.slot = r.U32();
        b.storage_source =
            r.Enum<VC::StorageSource>(static_cast<u32>(VC::StorageSource::UniformBufferWord));
        b.storage_bank = r.U32();
        b.storage_offset = r.U32();
        b.storage_index = r.U32();
        b.stage = r.Enum<VC::ShaderStage>(static_cast<u32>(VC::ShaderStage::Mesh));
        b.stage_mask = r.U32();
        b.texture = ReadHandleRef(r);
        b.sampler = ReadHandleRef(r);
        r.Words(b.handle_program, MAX_HANDLE_PROGRAM_WORDS);
        b.type = r.Enum<Shader::TextureType>(static_cast<u32>(Shader::TextureType::Color2DRect));
        b.image_format = r.Enum<Shader::ImageFormat>(
            static_cast<u32>(Shader::ImageFormat::R32G32B32A32_UINT));
        const u8 flags = r.U8();
        b.is_depth = (flags & 1) != 0;
        b.is_multisample = (flags & 2) != 0;
        b.is_integer = (flags & 4) != 0;
        b.is_written = (flags & 8) != 0;
        b.is_read = (flags & 16) != 0;
        b.supported = (flags & 32) != 0;
        b.is_atomic_target = (flags & 64) != 0;
        b.is_raw_global_table = (flags & 128) != 0;
        b.issue = r.Str(MAX_ISSUE_BYTES);
        out.bindings.push_back(std::move(b));
    }
    return r.Ok();
}

void WriteQueries(Writer& w, std::span<const TextureQuery> queries) {
    w.U32(static_cast<u32>(queries.size()));
    for (const TextureQuery& q : queries) {
        w.U32(q.stage);
        w.U32(q.record.bank);
        w.U32(q.record.offset);
        w.U32(static_cast<u32>(q.info.type));
        w.U32(static_cast<u32>(q.info.format));
        w.U8(q.info.is_integer ? 1 : 0);
    }
}

bool ReadQueries(Reader& r, std::vector<TextureQuery>& out) {
    const u32 count = r.U32();
    if (!r.Ok() || count > MAX_QUERIES || size_t{count} * 21 > r.Remaining()) {
        return false;
    }
    out.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        TextureQuery q;
        q.stage = r.U32();
        q.record.bank = r.U32();
        q.record.offset = r.U32();
        q.info.type = r.Enum<Shader::TextureType>(
            static_cast<u32>(Shader::TextureType::Color2DRect));
        // D32_FLOAT_S8_UINT is the last enumerator of TexturePixelFormat.
        q.info.format = r.Enum<Shader::TexturePixelFormat>(
            static_cast<u32>(Shader::TexturePixelFormat::D32_FLOAT_S8_UINT));
        q.info.is_integer = r.U8() != 0;
        out.push_back(q);
    }
    return r.Ok();
}

void WriteId(Writer& w, const ContentId& id) {
    w.U64(id.lo);
    w.U64(id.hi);
}

ContentId ReadId(Reader& r) {
    ContentId id;
    id.lo = r.U64();
    id.hi = r.U64();
    return id;
}

void WriteTargets(Writer& w, const RenderPassKey& key) {
    for (const s32 format : key.color) {
        w.U32(static_cast<u32>(format));
    }
    w.U32(static_cast<u32>(key.depth));
    w.U32(key.samples);
}

RenderPassKey ReadTargets(Reader& r) {
    RenderPassKey key;
    for (s32& format : key.color) {
        format = static_cast<s32>(r.U32());
    }
    key.depth = static_cast<s32>(r.U32());
    key.samples = r.U32();
    return key;
}

std::vector<u8> Frame(RecordType type, const std::vector<u8>& payload) {
    Writer w;
    w.data.reserve(FRAME_BYTES + payload.size());
    w.U32(RECORD_TAG);
    w.U32(static_cast<u32>(type));
    w.U32(static_cast<u32>(payload.size()));
    Writer crc_input;
    crc_input.U32(static_cast<u32>(type));
    crc_input.U32(static_cast<u32>(payload.size()));
    u32 crc = Crc32(crc_input.data);
    crc = Crc32(payload, crc);
    w.U32(crc);
    w.Raw(payload.data(), payload.size());
    return std::move(w.data);
}

bool DecodeShader(std::span<const u8> payload, ShaderRecord& out) {
    Reader r{payload};
    out = {};
    out.id = ReadId(r);
    out.stage = r.U32();
    out.program_hash = r.U64();
    out.flags = r.U32();
    out.attributes_read = r.U32();
    out.outputs_written = r.U32();
    for (u32& v : out.workgroup_size) {
        v = r.U32();
    }
    out.shared_memory_size = r.U32();
    out.has_program_constants = r.U8() != 0;
    if (!r.Words(out.program_constants, MAX_WORDS) || out.stage >= STAGE_COUNT) {
        return false;
    }
    const u32 spirv_words = r.U32();
    const u32 compressed_size = r.U32();
    if (!r.Ok() || spirv_words == 0 || spirv_words > MAX_WORDS) {
        return false;
    }
    const auto compressed = r.View(compressed_size);
    if (!r.Ok()) {
        return false;
    }
    out.spirv.resize(spirv_words);
    const std::span<u8> output{reinterpret_cast<u8*>(out.spirv.data()),
                               size_t{spirv_words} * sizeof(u32)};
    if (Common::Compression::DecompressDataZSTD(compressed, output) != output.size() ||
        out.spirv[0] != SPIRV_MAGIC) {
        return false;
    }
    const u32 layout_size = r.U32();
    const auto layout_bytes = r.View(layout_size);
    if (!r.Ok()) {
        return false;
    }
    if (!DeserializeDescriptorLayout(layout_bytes, out.layout)) {
        return false;
    }
    if (!ReadQueries(r, out.queries)) {
        return false;
    }
    return r.AtEnd() && !out.id.IsZero();
}

bool DecodeGraphics(std::span<const u8> payload, GraphicsPipelineRecord& out) {
    Reader r{payload};
    out = {};
    for (ContentId& id : out.shaders) {
        id = ReadId(r);
    }
    if (r.U32() != sizeof(GraphicsFixedState)) {
        return false;
    }
    r.Raw(&out.state, sizeof(out.state));
    out.targets = ReadTargets(r);
    out.push_constant_size = r.U32();
    out.viewport_count = r.U32();
    // A graphics pipeline has a vertex program or (task/mesh pipeline) a mesh program.
    return r.Ok() && r.AtEnd() && out.viewport_count >= 1 && out.viewport_count <= MAX_VIEWPORTS &&
           (!out.shaders[0].IsZero() || !out.shaders[STAGE_MESH].IsZero());
}

void WriteComputeLayout(Writer& w, const VC::ComputeDriverLayout& c) {
    for (const u32 v : {c.push_constant_offset, c.push_constant_words, c.uniform_offset,
                        c.uniform_count, c.storage_offset, c.storage_count, c.texture_offset,
                        c.texture_count, c.image_offset, c.image_count, c.end_offset,
                        c.declared_total}) {
        w.U32(v);
    }
    w.U8(c.total_matches ? 1 : 0);
}

VC::ComputeDriverLayout ReadComputeLayout(Reader& r) {
    VC::ComputeDriverLayout c;
    c.push_constant_offset = r.U32();
    c.push_constant_words = r.U32();
    c.uniform_offset = r.U32();
    c.uniform_count = r.U32();
    c.storage_offset = r.U32();
    c.storage_count = r.U32();
    c.texture_offset = r.U32();
    c.texture_count = r.U32();
    c.image_offset = r.U32();
    c.image_count = r.U32();
    c.end_offset = r.U32();
    c.declared_total = r.U32();
    c.total_matches = r.U8() != 0;
    return c;
}

bool DecodeCompute(std::span<const u8> payload, ComputePipelineRecord& out) {
    Reader r{payload};
    out = {};
    out.shader = ReadId(r);
    out.push_constant_size = r.U32();
    out.environment_flags = r.U32();
    out.local_memory_size = r.U32();
    out.shared_memory_size = r.U32();
    out.compute_layout = ReadComputeLayout(r);
    return r.Ok() && r.AtEnd() && !out.shader.IsZero();
}

struct ParsedHeader {
    Header header;
    u32 version{};
    size_t size{};
};

/// False when the bytes are not a header (magic, truncation, checksum).
bool ParseHeader(std::span<const u8> bytes, ParsedHeader& out, std::string& reason) {
    Reader r{bytes};
    std::array<char, 8> magic{};
    r.Raw(magic.data(), magic.size());
    if (!r.Ok() || magic != VideoCommon::NVN_PIPELINE_CACHE_MAGIC) {
        reason = "bad magic";
        return false;
    }
    out.version = r.U32();
    out.header.revision = r.Str(MAX_REVISION_BYTES);
    out.header.translation_key = r.Str(MAX_KEY_BYTES);
    out.header.vendor_id = r.U32();
    out.header.device_id = r.U32();
    out.header.driver_version = r.U32();
    out.header.cache_uuid_hash = r.U64();
    out.header.profile_hash = r.U64();
    const size_t crc_end = bytes.size() - r.Remaining();
    const u32 crc = r.U32();
    if (!r.Ok()) {
        reason = "truncated header";
        return false;
    }
    if (crc != Crc32(bytes.first(crc_end))) {
        reason = "header checksum mismatch";
        return false;
    }
    out.size = crc_end + sizeof(u32);
    return true;
}

} // Anonymous namespace

u32 Crc32(std::span<const u8> data, u32 crc) {
    crc = ~crc;
    for (const u8 byte : data) {
        crc = CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

ContentId MakeContentId(u32 stage, u64 program_hash, u64 runtime_hash, u64 queries_hash) {
    const std::array<u64, 4> words{stage, program_hash, runtime_hash, queries_hash};
    const u128 hash =
        Common::CityHash128(reinterpret_cast<const char*>(words.data()), sizeof(words));
    ContentId id{hash[0], hash[1]};
    if (id.IsZero()) {
        id.lo = 1;
    }
    return id;
}

u64 HashQueries(std::span<const TextureQuery> queries) {
    Writer w;
    w.data.reserve(queries.size() * 21);
    for (const TextureQuery& q : queries) {
        w.U32(q.stage);
        w.U32(q.record.bank);
        w.U32(q.record.offset);
        w.U32(static_cast<u32>(q.info.type));
        w.U32(static_cast<u32>(q.info.format));
        w.U8(q.info.is_integer ? 1 : 0);
    }
    return Common::CityHash64(reinterpret_cast<const char*>(w.data.data()), w.data.size());
}

void CanonicalizeState(GraphicsFixedState& s, VertexInputMode mode) {
    namespace N = VideoCommon::Nvn;
    using N::StateBits::Mask;
    s.provoking_first = s.provoking_first != 0 ? 1 : 0;
    s.pad1 = 0;
    s.primitive_restart = s.primitive_restart != 0 ? 1 : 0;
    s.restart_index = 0; // never read by pipeline creation (the draw supplies the index type)
    s.depth_clamp = s.depth_clamp != 0 ? 1 : 0;
    s.rasterizer_discard = s.rasterizer_discard != 0 ? 1 : 0;
    s.depth_mode = s.depth_mode != 0 ? 1 : 0;
    s.window_origin = s.window_origin == 1 ? 1 : 0;
    if (N::ColorState::AlphaFunc(s.color) == N::STATE_ALPHA_FUNC_ALWAYS) {
        s.alpha_ref_bits = 0;
    }

    // Words of state objects live in guest memory: bits no field covers are whatever the guest
    // left there. They differ between draws and launches, so they must not be part of the key.
    s.polygon &= static_cast<u32>(
        Mask(N::PolygonState::OffsetEnables_SHIFT + N::PolygonState::OffsetEnables_WIDTH));
    s.color &=
        static_cast<u32>(Mask(N::ColorState::AlphaFunc_SHIFT + N::ColorState::AlphaFunc_WIDTH));
    s.depth_stencil &= Mask(N::DepthStencilState::BackDepthPassOp_SHIFT +
                            N::DepthStencilState::BackDepthPassOp_WIDTH);
    if (N::DepthStencilState::DepthTestEnable(s.depth_stencil) == 0) {
        s.depth_stencil = N::DepthStencilState::WithDepthFunc(s.depth_stencil, 0);
    }
    if (N::DepthStencilState::StencilTestEnable(s.depth_stencil) == 0) {
        constexpr u32 FIRST = N::DepthStencilState::FrontStencilFunc_SHIFT;
        constexpr u32 END = N::DepthStencilState::BackDepthPassOp_SHIFT +
                            N::DepthStencilState::BackDepthPassOp_WIDTH;
        s.depth_stencil &= ~(Mask(END - FIRST) << FIRST);
    }
    // Blend words: the target index is the slot, only blending targets read the functions, and
    // the advanced-blend fields above the mode are not consulted.
    constexpr u32 BLEND_FIRST = N::BlendState::TargetIndex_WIDTH;
    constexpr u32 BLEND_END = N::BlendState::AdvancedMode_SHIFT + N::BlendState::AdvancedMode_WIDTH;
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        u64& blend = s.blend[slot];
        if (!N::ColorState::BlendEnabled(s.color, static_cast<s32>(slot))) {
            blend = 0;
        } else {
            blend &= Mask(BLEND_END - BLEND_FIRST) << BLEND_FIRST;
        }
    }
    // Multisample bytes: only the fields the pipeline reads, as 0/1; the rest is zero.
    using MS = N::MultisampleState;
    const std::array<u8, 16> source = s.multisample;
    s.multisample.fill(0);
    for (const u32 offset :
         {MS::OFFSET_ALPHA_TO_COVERAGE, MS::OFFSET_ALPHA_TO_COVERAGE_DITHER,
          MS::OFFSET_COVERAGE_MODULATION_MODE, MS::OFFSET_COVERAGE_TO_COLOR_ENABLE,
          MS::OFFSET_SAMPLE_LOCATIONS_ENABLE, MS::OFFSET_COVERAGE_TRUNCATION_ENABLE}) {
        s.multisample[offset] = source[offset] != 0 ? 1 : 0;
    }
    // Dither only shapes the alpha-to-coverage mask (NVN sets it by default).
    if (s.multisample[MS::OFFSET_ALPHA_TO_COVERAGE] == 0) {
        s.multisample[MS::OFFSET_ALPHA_TO_COVERAGE_DITHER] = 0;
    }
    if (mode == VertexInputMode::Dynamic) {
        // The layout is not part of the key: the shader-visible type class per location is all
        // that remains (values the decoder cannot produce are Disabled).
        s.attributes.fill(0);
        s.stream_stride.fill(0);
        s.stream_divisor.fill(0);
        constexpr u8 DISABLED = static_cast<u8>(Shader::AttributeType::Disabled);
        for (u8& type : s.vertex_types) {
            type = type <= DISABLED ? type : DISABLED;
        }
        return;
    }
    // Static: an attribute with format 0 is unused; a stream no used attribute points at is
    // unread; the type classes are derived from the words (not part of this form).
    s.vertex_types.fill(0);
    u32 streams_used = 0;
    for (u32& word : s.attributes) {
        if (N::VertexAttribState::Format(word) == 0) {
            word = 0;
        } else {
            streams_used |= 1u << N::VertexAttribState::StreamIndex(word);
        }
    }
    for (u32 stream = 0; stream < MAX_VERTEX_STREAMS; ++stream) {
        if ((streams_used & (1u << stream)) == 0) {
            s.stream_stride[stream] = 0;
            s.stream_divisor[stream] = 0;
        }
    }
}

std::vector<u8> EncodeHeader(const Header& header) {
    Writer w;
    w.Raw(VideoCommon::NVN_PIPELINE_CACHE_MAGIC.data(),
          VideoCommon::NVN_PIPELINE_CACHE_MAGIC.size());
    w.U32(FORMAT_VERSION);
    w.Str(header.revision.substr(0, MAX_REVISION_BYTES));
    w.Str(header.translation_key.substr(0, MAX_KEY_BYTES));
    w.U32(header.vendor_id);
    w.U32(header.device_id);
    w.U32(header.driver_version);
    w.U64(header.cache_uuid_hash);
    w.U64(header.profile_hash);
    w.U32(Crc32(w.data));
    return std::move(w.data);
}

std::vector<u8> EncodeShader(const ShaderRecord& record) {
    Writer w;
    WriteId(w, record.id);
    w.U32(record.stage);
    w.U64(record.program_hash);
    w.U32(record.flags);
    w.U32(record.attributes_read);
    w.U32(record.outputs_written);
    for (const u32 v : record.workgroup_size) {
        w.U32(v);
    }
    w.U32(record.shared_memory_size);
    w.U8(record.has_program_constants ? 1 : 0);
    w.Words(record.program_constants);
    const std::vector<u8> compressed = Common::Compression::CompressDataZSTDDefault(
        reinterpret_cast<const u8*>(record.spirv.data()), record.spirv.size() * sizeof(u32));
    w.U32(static_cast<u32>(record.spirv.size()));
    w.U32(static_cast<u32>(compressed.size()));
    w.Raw(compressed.data(), compressed.size());
    const std::vector<u8> layout = SerializeDescriptorLayout(record.layout);
    w.U32(static_cast<u32>(layout.size()));
    w.Raw(layout.data(), layout.size());
    WriteQueries(w, record.queries);
    return Frame(RecordType::Shader, w.data);
}

std::vector<u8> EncodeGraphicsPipeline(const GraphicsPipelineRecord& record) {
    Writer w;
    for (const ContentId& id : record.shaders) {
        WriteId(w, id);
    }
    w.U32(static_cast<u32>(sizeof(GraphicsFixedState)));
    w.Raw(&record.state, sizeof(record.state));
    WriteTargets(w, record.targets);
    w.U32(record.push_constant_size);
    w.U32(record.viewport_count);
    return Frame(RecordType::GraphicsPipeline, w.data);
}

std::vector<u8> EncodeComputePipeline(const ComputePipelineRecord& record) {
    Writer w;
    WriteId(w, record.shader);
    w.U32(record.push_constant_size);
    w.U32(record.environment_flags);
    w.U32(record.local_memory_size);
    w.U32(record.shared_memory_size);
    WriteComputeLayout(w, record.compute_layout);
    return Frame(RecordType::ComputePipeline, w.data);
}

std::vector<u8> SerializeDescriptorLayout(const VC::DescriptorLayout& layout) {
    Writer w;
    WriteLayout(w, layout);
    return std::move(w.data);
}

bool DeserializeDescriptorLayout(std::span<const u8> bytes, VC::DescriptorLayout& out) {
    Reader r{bytes};
    return ReadLayout(r, out) && r.AtEnd();
}

LoadResult Parse(std::span<const u8> bytes, const Header& expected, Contents& contents) {
    LoadResult result;
    result.input_bytes = bytes.size();
    if (bytes.empty()) {
        result.status = LoadStatus::Missing;
        return result;
    }
    ParsedHeader parsed;
    std::string reason;
    if (!ParseHeader(bytes, parsed, reason)) {
        result.status = LoadStatus::Rejected;
        result.reason = "damaged header: " + reason;
        return result;
    }
    if (parsed.version != FORMAT_VERSION) {
        result.status = LoadStatus::Rejected;
        result.reason = "format version " + std::to_string(parsed.version) + " (this build " +
                        std::to_string(FORMAT_VERSION) + ")";
        return result;
    }
    if (parsed.header.vendor_id != expected.vendor_id ||
        parsed.header.device_id != expected.device_id ||
        parsed.header.driver_version != expected.driver_version ||
        parsed.header.cache_uuid_hash != expected.cache_uuid_hash) {
        result.status = LoadStatus::Rejected;
        result.reason = "written for another device or driver";
        return result;
    }
    if (parsed.header.profile_hash != expected.profile_hash) {
        result.status = LoadStatus::Rejected;
        result.reason = "written for another shader profile";
        return result;
    }
    if (parsed.header.translation_key != expected.translation_key) {
        result.status = LoadStatus::Rejected;
        result.reason = "written for other translation inputs (file: " +
                        parsed.header.translation_key + ", build: " + expected.translation_key +
                        ", written by build " + parsed.header.revision + ")";
        return result;
    }

    result.status = LoadStatus::Loaded;
    size_t pos = parsed.size;
    result.valid_bytes = pos;
    while (pos < bytes.size()) {
        const size_t remaining = bytes.size() - pos;
        if (remaining < FRAME_BYTES) {
            result.reason = "truncated record frame";
            break;
        }
        Reader frame{bytes.subspan(pos, FRAME_BYTES)};
        const u32 tag = frame.U32();
        const u32 type = frame.U32();
        const u32 size = frame.U32();
        const u32 crc = frame.U32();
        if (tag != RECORD_TAG) {
            result.reason = "bad record tag";
            break;
        }
        if (size > MAX_RECORD_BYTES || size > remaining - FRAME_BYTES) {
            result.reason = "record larger than the rest of the file";
            break;
        }
        const auto payload = bytes.subspan(pos + FRAME_BYTES, size);
        Writer crc_input;
        crc_input.U32(type);
        crc_input.U32(size);
        if (Crc32(payload, Crc32(crc_input.data)) != crc) {
            result.reason = "record checksum mismatch";
            break;
        }
        bool decoded = true;
        try {
            switch (static_cast<RecordType>(type)) {
            case RecordType::Shader: {
                ShaderRecord record;
                decoded = DecodeShader(payload, record);
                if (decoded) {
                    contents.shaders.push_back(std::move(record));
                }
                break;
            }
            case RecordType::GraphicsPipeline: {
                GraphicsPipelineRecord record;
                decoded = DecodeGraphics(payload, record);
                if (decoded) {
                    contents.graphics.push_back(record);
                }
                break;
            }
            case RecordType::ComputePipeline: {
                ComputePipelineRecord record;
                decoded = DecodeCompute(payload, record);
                if (decoded) {
                    contents.compute.push_back(record);
                }
                break;
            }
            default:
                break; // a record type this build does not know: skipped, its frame was valid
            }
        } catch (const std::exception&) {
            decoded = false; // allocation failure on a garbage size: the record is malformed
        }
        if (!decoded) {
            result.reason = "malformed record payload (record " + std::to_string(result.records) +
                            ", type " + std::to_string(type) + ", " + std::to_string(size) +
                            " bytes at offset " + std::to_string(pos) + ")";
            break;
        }
        pos += FRAME_BYTES + size;
        result.valid_bytes = pos;
        ++result.records;
    }
    return result;
}

LoadResult LoadFile(const std::filesystem::path& path, const Header& expected,
                    Contents& contents) {
    LoadResult result;
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0) {
        return result; // Missing
    }
    if (size > (std::numeric_limits<u32>::max)() * size_t{4}) {
        result.status = LoadStatus::Rejected;
        result.input_bytes = static_cast<size_t>(size);
        result.reason = "implausible file size";
        return result;
    }
    std::vector<u8> bytes(static_cast<size_t>(size));
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open() ||
            !file.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()))) {
            result.status = LoadStatus::Rejected;
            result.input_bytes = bytes.size();
            result.reason = "unreadable";
            return result;
        }
    }
    return Parse(bytes, expected, contents);
}

bool WriteFileAtomic(const std::filesystem::path& path, std::span<const u8> data) {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return false;
        }
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size()));
        file.flush();
        if (!file.good()) {
            file.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

bool PrepareFile(const std::filesystem::path& path, const Header& header,
                 const LoadResult& load) {
    if (load.status != LoadStatus::Loaded) {
        return WriteFileAtomic(path, EncodeHeader(header));
    }
    if (load.HasBadTail()) {
        std::error_code error;
        std::filesystem::resize_file(path, load.valid_bytes, error);
        return !error;
    }
    return true;
}

bool Appender::Open(const std::filesystem::path& path) {
    file.open(path, std::ios::binary | std::ios::app);
    return file.is_open();
}

bool Appender::Append(std::span<const u8> record) {
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(record.data()),
               static_cast<std::streamsize>(record.size()));
    file.flush();
    return file.good();
}

std::string MakeTranslationKey(const TranslationInputs& inputs) {
    return fmt::format("nvn-pipeline-cache/{} epoch={} switches={:016x} vendor={:08x} "
                       "device={:08x} driver={:08x} uuid={:016x} profile={:016x}",
                       FORMAT_VERSION, inputs.epoch, inputs.switches_hash, inputs.vendor_id,
                       inputs.device_id, inputs.driver_version, inputs.cache_uuid_hash,
                       inputs.profile_hash);
}

std::string FileNameForKey(std::string_view key) {
    // The header holds the whole key and stays the authority; the file name only has to tell
    // keys apart. 64 bits of CityHash: a collision costs a rejected file, never wrong SPIR-V.
    return fmt::format("{}.k{:0{}x}.bin", FILE_STEM, Common::CityHash64(key.data(), key.size()),
                       KEY_TAG_DIGITS);
}

void PruneCacheFiles(const std::filesystem::path& directory,
                     const std::filesystem::path& keep_path, size_t keep) {
    namespace fs = std::filesystem;
    std::error_code error;
    const std::string stem{FILE_STEM};
    // Bump the file in use so a build that is still being run is not the oldest entry.
    fs::last_write_time(keep_path, fs::file_time_type::clock::now(), error);
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    for (fs::directory_iterator it{directory, error}, end; !error && it != end;
         it.increment(error)) {
        const std::string name = it->path().filename().string();
        // `<stem>.bin` (legacy) or `<stem>.k<key>.bin`; not `<stem>_backup.bin` and the like.
        const bool is_cache = name.size() >= stem.size() + 4 && name.starts_with(stem) &&
                              name[stem.size()] == '.' && name.ends_with(".bin") &&
                              it->path() != keep_path;
        if (!is_cache || !it->is_regular_file(error)) {
            continue;
        }
        const auto time = it->last_write_time(error);
        if (!error) {
            files.emplace_back(time, it->path());
        }
    }
    if (files.size() < keep) {
        return;
    }
    std::ranges::sort(files, std::greater<>{}, &decltype(files)::value_type::first);
    for (size_t i = keep - 1; i < files.size(); ++i) {
        std::error_code ignored;
        fs::remove(files[i].second, ignored);
    }
}

} // namespace Vulkan::Nvn::Disk
