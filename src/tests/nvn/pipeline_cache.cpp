// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the NVN pipeline cache module (renderer_vulkan/nvn/nvn_pipeline_cache.*): the program
// hash, the on-disk format (no GPU), the descriptor layout serializer, texture-query
// revalidation and, with a Vulkan device, render passes and the disk cache lifecycle. Translation
// of real guest shaders is exercised by the shader coverage tool and the game, not here.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/diag_request.h"
#include "common/dynamic_library.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "tests/temp_path.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/load_progress.h"
#include "video_core/native_tensor_programs.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/shader_notify.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_cache.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_disk.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace {

namespace VC = VideoCommon::Nvn;
namespace Disk = Vulkan::Nvn::Disk;
using Vulkan::Nvn::GraphicsFixedState;
using Vulkan::Nvn::HashProgramStage;
using Vulkan::Nvn::PipelineBase;
using Vulkan::Nvn::PipelineCache;
using Vulkan::Nvn::PipelineCacheOptions;
using Vulkan::Nvn::RenderPassKey;
using Vulkan::Nvn::TextureInfo;
using Vulkan::Nvn::TextureOracle;
using Vulkan::Nvn::TextureQuery;

// --- HashProgramStage ---------------------------------------------------------------------

std::vector<u8> PatternBytes(size_t size, u8 seed) {
    std::vector<u8> bytes(size);
    for (size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<u8>(seed + i * 7);
    }
    return bytes;
}

u64 HashOf(const std::vector<u8>& control, const std::vector<u8>& code) {
    return HashProgramStage(control, code);
}

// --- disk fixtures ------------------------------------------------------------------------

VC::DescriptorLayout MakeLayout() {
    VC::DescriptorLayout layout;
    layout.binding_count = 9;
    layout.uses_rescaling_uniform = true;
    layout.uses_render_area = true;
    layout.uses_raw_global_memory = true;
    layout.stores_raw_global_memory = true;
    layout.raw_stores_written_through = true;
    layout.requires_guest_warp = true;
    layout.unsupported_count = 1;

    VC::DescriptorBinding a;
    a.binding = 2;
    a.kind = VC::DescriptorKind::CombinedImageSampler;
    a.role = VC::DescriptorRole::Texture;
    a.count = 3;
    a.slot = 5;
    a.storage_source = VC::StorageSource::None;
    a.storage_bank = 0;
    a.storage_offset = 0;
    a.stage = VC::ShaderStage::Fragment;
    a.stage_mask = 1u << static_cast<u32>(VC::ShaderStage::Fragment);
    a.texture = {VC::HandleSource::Combined, 7, 2, 0x48};
    a.sampler = {VC::HandleSource::SeparateSampler, 3, 2, 0x568};
    a.type = Shader::TextureType::ColorArray2D;
    a.image_format = Shader::ImageFormat::Typeless;
    a.is_depth = true;
    a.is_multisample = false;
    a.is_integer = true;
    a.is_written = false;
    a.is_read = true;
    a.supported = true;

    VC::DescriptorBinding b;
    b.binding = 7;
    b.kind = VC::DescriptorKind::StorageBuffer;
    b.role = VC::DescriptorRole::StorageBuffer;
    b.count = 1;
    b.slot = 9;
    b.storage_source = VC::StorageSource::UniformBufferWord;
    b.storage_bank = 4;
    b.storage_offset = 0x30;
    b.stage = VC::ShaderStage::Vertex;
    b.stage_mask = 1;
    b.texture = {VC::HandleSource::UniformBuffer, 1, 3, 0x10};
    b.sampler = {};
    b.type = Shader::TextureType::Buffer;
    b.image_format = Shader::ImageFormat::R32G32B32A32_UINT;
    b.is_multisample = true;
    b.is_written = true;
    b.is_atomic_target = true;
    b.is_raw_global_table = true;
    b.supported = false;
    b.issue = "uniform-buffer word pair not supported";
    b.storage_index = 3;
    a.texture = {VC::HandleSource::Program, 0, 0, 0};
    a.handle_program = {5u, 4u, 0x88u, 1u};

    layout.bindings = {a, b};
    return layout;
}

void RequireSameHandle(const VC::HandleRef& x, const VC::HandleRef& y) {
    REQUIRE(x.source == y.source);
    REQUIRE(x.slot == y.slot);
    REQUIRE(x.bank == y.bank);
    REQUIRE(x.byte_offset == y.byte_offset);
}

void RequireSameLayout(const VC::DescriptorLayout& x, const VC::DescriptorLayout& y) {
    REQUIRE(x.binding_count == y.binding_count);
    REQUIRE(x.uses_rescaling_uniform == y.uses_rescaling_uniform);
    REQUIRE(x.uses_render_area == y.uses_render_area);
    REQUIRE(x.uses_raw_global_memory == y.uses_raw_global_memory);
    REQUIRE(x.stores_raw_global_memory == y.stores_raw_global_memory);
    REQUIRE(x.raw_stores_written_through == y.raw_stores_written_through);
    REQUIRE(x.requires_guest_warp == y.requires_guest_warp);
    REQUIRE(x.unsupported_count == y.unsupported_count);
    REQUIRE(x.bindings.size() == y.bindings.size());
    for (size_t i = 0; i < x.bindings.size(); ++i) {
        const VC::DescriptorBinding& p = x.bindings[i];
        const VC::DescriptorBinding& q = y.bindings[i];
        REQUIRE(p.binding == q.binding);
        REQUIRE(p.kind == q.kind);
        REQUIRE(p.role == q.role);
        REQUIRE(p.count == q.count);
        REQUIRE(p.slot == q.slot);
        REQUIRE(p.storage_source == q.storage_source);
        REQUIRE(p.storage_bank == q.storage_bank);
        REQUIRE(p.storage_offset == q.storage_offset);
        REQUIRE(p.storage_index == q.storage_index);
        REQUIRE(p.handle_program == q.handle_program);
        REQUIRE(p.stage == q.stage);
        REQUIRE(p.stage_mask == q.stage_mask);
        RequireSameHandle(p.texture, q.texture);
        RequireSameHandle(p.sampler, q.sampler);
        REQUIRE(p.type == q.type);
        REQUIRE(p.image_format == q.image_format);
        REQUIRE(p.is_depth == q.is_depth);
        REQUIRE(p.is_multisample == q.is_multisample);
        REQUIRE(p.is_integer == q.is_integer);
        REQUIRE(p.is_written == q.is_written);
        REQUIRE(p.is_read == q.is_read);
        REQUIRE(p.supported == q.supported);
        REQUIRE(p.is_atomic_target == q.is_atomic_target);
        REQUIRE(p.is_raw_global_table == q.is_raw_global_table);
        REQUIRE(p.issue == q.issue);
    }
}

std::vector<TextureQuery> MakeQueries() {
    TextureQuery a;
    a.stage = 1;
    a.record = {2, 0x168};
    a.info = {Shader::TextureType::ColorCube, Shader::TexturePixelFormat::R8_UINT, true};
    TextureQuery b;
    b.stage = 0;
    b.record = {3, 0x10};
    b.info = {Shader::TextureType::Color3D, Shader::TexturePixelFormat::A2B10G10R10_UNORM, false};
    return {a, b};
}

Disk::ShaderRecord MakeShader(u32 stage, u64 program_hash, size_t words) {
    Disk::ShaderRecord record;
    record.id = Disk::MakeContentId(stage, program_hash, 0x1234, 0x5678);
    record.stage = stage;
    record.program_hash = program_hash;
    record.flags = Disk::SHADER_STORES_VIEWPORT_INDEX;
    record.attributes_read = 0x0000'0F0Fu;
    record.outputs_written = 0b101;
    record.workgroup_size = {8, 4, 2};
    record.shared_memory_size = 1024;
    record.has_program_constants = true;
    record.program_constants = {1, 2, 3, 0xDEADBEEF};
    record.spirv.resize(words);
    for (size_t i = 0; i < words; ++i) {
        record.spirv[i] = static_cast<u32>(i * 2654435761u);
    }
    record.spirv[0] = 0x07230203; // SPIR-V magic: records without it are rejected
    record.layout = MakeLayout();
    record.queries = MakeQueries();
    return record;
}

void RequireSameShader(const Disk::ShaderRecord& x, const Disk::ShaderRecord& y) {
    REQUIRE(x.id == y.id);
    REQUIRE(x.stage == y.stage);
    REQUIRE(x.program_hash == y.program_hash);
    REQUIRE(x.flags == y.flags);
    REQUIRE(x.attributes_read == y.attributes_read);
    REQUIRE(x.outputs_written == y.outputs_written);
    REQUIRE(x.workgroup_size == y.workgroup_size);
    REQUIRE(x.shared_memory_size == y.shared_memory_size);
    REQUIRE(x.has_program_constants == y.has_program_constants);
    REQUIRE(x.program_constants == y.program_constants);
    REQUIRE(x.spirv == y.spirv);
    RequireSameLayout(x.layout, y.layout);
    REQUIRE(x.queries.size() == y.queries.size());
    for (size_t i = 0; i < x.queries.size(); ++i) {
        REQUIRE(x.queries[i].stage == y.queries[i].stage);
        REQUIRE(x.queries[i].record.bank == y.queries[i].record.bank);
        REQUIRE(x.queries[i].record.offset == y.queries[i].record.offset);
        REQUIRE(x.queries[i].info.type == y.queries[i].info.type);
        REQUIRE(x.queries[i].info.format == y.queries[i].info.format);
        REQUIRE(x.queries[i].info.is_integer == y.queries[i].info.is_integer);
    }
}

Disk::GraphicsPipelineRecord MakeGraphics(const Disk::ContentId& vertex,
                                          const Disk::ContentId& fragment) {
    Disk::GraphicsPipelineRecord record;
    std::memset(&record.state, 0, sizeof(record.state));
    record.state.topology = 4;
    record.state.polygon = 0x25;
    record.state.color = 0x8300;
    record.state.channel_mask = 0xFFFF'FFFFu;
    record.state.sample_mask = 0xFFFF'FFFFu;
    record.state.depth_stencil = 0x1234'5678'9ABCull;
    record.state.blend[3] = 0xFEDCBA987ull;
    record.state.attributes[0] = 0x25 | (16u << 8);
    record.state.stream_stride[0] = 32;
    record.state.stream_divisor[1] = 4;
    record.state.vertex_types[0] = 2; // raw bytes: the codec must keep the whole struct
    record.state.vertex_types[7] = 5;
    record.shaders[0] = vertex;
    record.shaders[1] = fragment;
    record.targets.color = {37, 0, 43, 0, 0, 0, 0, 0};
    record.targets.depth = 126;
    record.targets.samples = 4;
    record.push_constant_size = 44;
    record.viewport_count = 16;
    return record;
}

Disk::ComputePipelineRecord MakeCompute(const Disk::ContentId& shader) {
    Disk::ComputePipelineRecord record;
    record.shader = shader;
    record.push_constant_size = 44;
    record.environment_flags = 3;
    record.local_memory_size = 64;
    record.shared_memory_size = 2048;
    record.compute_layout.uniform_offset = 0x10;
    record.compute_layout.uniform_count = 2;
    record.compute_layout.storage_offset = 0x30;
    record.compute_layout.storage_count = 1;
    record.compute_layout.texture_offset = 0x40;
    record.compute_layout.texture_count = 3;
    record.compute_layout.image_offset = 0x58;
    record.compute_layout.image_count = 1;
    record.compute_layout.end_offset = 0x60;
    record.compute_layout.declared_total = 0x60;
    record.compute_layout.total_matches = true;
    return record;
}

Disk::Header MakeHeader() {
    Disk::Header header;
    header.revision = "test-revision-1";
    header.translation_key = "test-translation-key-1";
    header.vendor_id = 0x10DE;
    header.device_id = 0x2B85;
    header.driver_version = 0x1234;
    header.cache_uuid_hash = 0xABCDEF;
    header.profile_hash = 0x555;
    return header;
}

struct FileImage {
    std::vector<u8> bytes;
    std::vector<size_t> record_ends; ///< end offset of each record
    size_t header_size{};
    Disk::ShaderRecord vertex;
    Disk::ShaderRecord fragment;
    Disk::ShaderRecord compute_shader;
    Disk::GraphicsPipelineRecord graphics;
    Disk::ComputePipelineRecord compute;
};

FileImage MakeImage(const Disk::Header& header) {
    FileImage image;
    image.vertex = MakeShader(0, 0x1111, 40);
    image.fragment = MakeShader(1, 0x2222, 200);
    image.compute_shader = MakeShader(5, 0x3333, 64);
    image.graphics = MakeGraphics(image.vertex.id, image.fragment.id);
    image.compute = MakeCompute(image.compute_shader.id);
    image.bytes = Disk::EncodeHeader(header);
    image.header_size = image.bytes.size();
    const auto append = [&image](const std::vector<u8>& record) {
        image.bytes.insert(image.bytes.end(), record.begin(), record.end());
        image.record_ends.push_back(image.bytes.size());
    };
    append(Disk::EncodeShader(image.vertex));
    append(Disk::EncodeShader(image.fragment));
    append(Disk::EncodeGraphicsPipeline(image.graphics));
    append(Disk::EncodeShader(image.compute_shader));
    append(Disk::EncodeComputePipeline(image.compute));
    return image;
}

size_t CountContents(const Disk::Contents& contents) {
    return contents.shaders.size() + contents.graphics.size() + contents.compute.size();
}

/// Temp directory removed on scope exit.
struct TempDirectory {
    TempDirectory() {
        path = Tests::UniqueTempPath("nvn_pipeline_cache_test");
        std::filesystem::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};

void WriteBytes(const std::filesystem::path& path, std::span<const u8> bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

// --- oracle -------------------------------------------------------------------------------

class FakeOracle final : public TextureOracle {
public:
    TextureInfo Query(u32 stage, const VC::NvnHandleRecord& record) const override {
        ++queries;
        if (stage == last_stage && record.bank == last_bank && record.offset == last_offset) {
            return answer;
        }
        return {};
    }
    u32 last_stage{};
    u32 last_bank{};
    u32 last_offset{};
    TextureInfo answer{};
    mutable int queries{};
};

} // Anonymous namespace

TEST_CASE("HashProgramStage depends on code and the hashed control bytes only", "[nvn]") {
    const std::vector<u8> control = PatternBytes(0x100, 3);
    const std::vector<u8> code = PatternBytes(0x200, 9);
    const u64 base = HashOf(control, code);
    REQUIRE(base != 0);

    SECTION("unused bytes are ignored for graphics stages") {
        auto changed = control;
        for (const size_t offset : {size_t{0x48}, size_t{0x4F}, size_t{0x6C}, size_t{0xFF}}) {
            changed[offset] ^= 0xFF;
        }
        REQUIRE(HashOf(changed, code) == base);
        // The control section's length differs between containers: a longer section hashes the same
        // as long as the hashed bytes are equal.
        REQUIRE(HashOf(std::vector<u8>(control.begin(), control.begin() + 0x6C), code) == base);
    }
    SECTION("the register count matters") {
        auto changed = control;
        changed[0x4c] ^= 1;
        REQUIRE(HashOf(changed, code) != base);
    }
    SECTION("the compute fields at +0x50..+0x6b matter") {
        // Per-warp local memory (0x50), thread group (0x58..0x63) and shared memory (0x64) change
        // the translated program: two programs differing only there must not share a pipeline.
        for (const size_t offset : {size_t{0x50}, size_t{0x53}, size_t{0x58}, size_t{0x5F}, size_t{0x63},
                                    size_t{0x64}, size_t{0x67}, size_t{0x6B}}) {
            auto changed = control;
            changed[offset] ^= 0x01;
            REQUIRE(HashOf(changed, code) != base);
        }
    }
    SECTION("the u32 at +0x30 is ignored") {
        for (size_t offset = 0x30; offset < 0x34; ++offset) {
            auto changed = control;
            changed[offset] ^= 0xA5;
            REQUIRE(HashOf(changed, code) == base);
        }
    }
    SECTION("neighbouring control bytes matter") {
        for (const size_t offset : {size_t{0}, size_t{0x2F}, size_t{0x34}, size_t{0x47}}) {
            auto changed = control;
            changed[offset] ^= 0x01;
            REQUIRE(HashOf(changed, code) != base);
        }
    }
    SECTION("the code matters") {
        auto changed = code;
        changed.back() ^= 1;
        REQUIRE(HashOf(control, changed) != base);
        changed = code;
        changed.push_back(0);
        REQUIRE(HashOf(control, changed) != base);
    }
    SECTION("never zero, never crashes on short input") {
        REQUIRE(HashOf({}, {}) != 0);
        REQUIRE(HashOf(PatternBytes(0x31, 1), {}) != 0);
        REQUIRE(HashOf(PatternBytes(0x30, 1), code) != 0);
        REQUIRE(HashOf(PatternBytes(0x33, 1), code) != HashOf(PatternBytes(0x30, 1), code));
    }
}

TEST_CASE("Shader cache identity includes the compute driver bank layout",
          "[nvn][shader-refresh]") {
    std::vector<u8> control(VC::ComputeControlBytes, 0);
    control[0x56] = static_cast<u8>(VC::ShaderStage::Compute);
    const std::vector<u8> code = PatternBytes(0x100, 9);
    const u64 base = HashOf(control, code);
    for (const size_t offset : {0xa9, 0xb1, 0xb2, 0xb3, 0xb6, 0xba, 0xbb}) {
        INFO("control offset " << offset);
        auto changed = control;
        changed[offset] = 1;
        CHECK(HashOf(changed, code) != base);
    }
    // A missing counts table must not poison a valid program's lowered/pipeline cache entry.
    CHECK(HashOf(std::vector<u8>(control.begin(), control.begin() + 0x6c), code) != base);
}

TEST_CASE("Vulkan cache removal includes NVN caches and preserves other files",
          "[nvn][shader-refresh]") {
    for (const std::string_view name :
         {"vulkan.bin", "vulkan_pipelines.bin", "vulkan.mythos.bin", "vulkan_pipelines.mythos.bin",
          "nvn_pipelines.bin", "nvn_pipelines.vkcache", "nvn_pipelines.k0123456789abcdef.bin",
          "nvn_pipelines.2b7551f137ab.bin"}) {
        INFO(name);
        CHECK(VideoCommon::IsVulkanPipelineCacheFile(name));
    }
    for (const std::string_view name :
         {"opengl.bin", "nvn_pipelines.bin.backup", "nvn_pipelines_backup.bin", "unrelated.bin"}) {
        INFO(name);
        CHECK_FALSE(VideoCommon::IsVulkanPipelineCacheFile(name));
    }
}

TEST_CASE("MaySkipUnbuiltDraw follows the emulated path's async skip rule", "[nvn]") {
    // vk_pipeline_cache.cpp BuiltPipeline / CanSkipDraw: stores wait, depth may skip, else only
    // more than 6 elements may skip (indirect and mesh draws pass 0).
    using Vulkan::Nvn::MaySkipUnbuiltDraw;
    static_assert(!MaySkipUnbuiltDraw(true, true, 1000));
    static_assert(!MaySkipUnbuiltDraw(true, false, 1000));
    static_assert(MaySkipUnbuiltDraw(false, true, 0));
    static_assert(MaySkipUnbuiltDraw(false, true, 3));
    static_assert(!MaySkipUnbuiltDraw(false, false, 0));
    static_assert(!MaySkipUnbuiltDraw(false, false, 1));
    static_assert(!MaySkipUnbuiltDraw(false, false, 6));
    static_assert(MaySkipUnbuiltDraw(false, false, 7));
    SUCCEED();
}

TEST_CASE("StoresGlobalMemory reads the DoesGlobalStore bit of each present stage's header",
          "[nvn]") {
    const std::vector<u8> control(0x100, 0);
    std::vector<u8> plain(0x50, 0);
    plain[1] = 0x04; // SPH type vertex (bit 10), no store
    std::vector<u8> storing = plain;
    storing[2] = 0x01; // word 0 bit 16: DoesGlobalStore
    const std::vector<u8> short_code(3, 0xFF);

    Vulkan::Nvn::GraphicsRequest request;
    request.stages[0] = {1, &control, &plain};
    REQUIRE_FALSE(StoresGlobalMemory(request));
    request.stages[1] = {2, &control, &storing};
    REQUIRE(StoresGlobalMemory(request));
    // An absent stage (hash 0) does not count, whatever its bytes.
    request.stages[1].hash = 0;
    REQUIRE_FALSE(StoresGlobalMemory(request));
    // Other header bits (all of word 0 but bit 16) are not a store.
    std::vector<u8> others = plain;
    others[0] = 0xFF;
    others[1] = 0xFF;
    others[2] = 0xFE;
    others[3] = 0xFF;
    request.stages[1] = {2, &control, &others};
    REQUIRE_FALSE(StoresGlobalMemory(request));
    // Code too short for a header word is not read.
    request.stages[1] = {2, &control, &short_code};
    REQUIRE_FALSE(StoresGlobalMemory(request));
}

TEST_CASE("Descriptor layout serialization round trip", "[nvn]") {
    const VC::DescriptorLayout layout = MakeLayout();
    const std::vector<u8> bytes = Disk::SerializeDescriptorLayout(layout);
    REQUIRE(!bytes.empty());

    VC::DescriptorLayout decoded;
    REQUIRE(Disk::DeserializeDescriptorLayout(bytes, decoded));
    RequireSameLayout(layout, decoded);

    SECTION("an empty layout") {
        const VC::DescriptorLayout empty;
        VC::DescriptorLayout out = layout;
        REQUIRE(Disk::DeserializeDescriptorLayout(Disk::SerializeDescriptorLayout(empty), out));
        RequireSameLayout(empty, out);
    }
    SECTION("every strict prefix is rejected") {
        for (size_t size = 0; size < bytes.size(); ++size) {
            VC::DescriptorLayout out;
            REQUIRE_FALSE(Disk::DeserializeDescriptorLayout(
                std::span<const u8>{bytes.data(), size}, out));
        }
    }
    SECTION("trailing bytes are rejected") {
        auto longer = bytes;
        longer.push_back(0);
        VC::DescriptorLayout out;
        REQUIRE_FALSE(Disk::DeserializeDescriptorLayout(longer, out));
    }
    SECTION("a binding count the input cannot hold is rejected without allocating it") {
        auto garbage = bytes;
        // binding_count(4) rescaling(1) render_area(1) raw_global(1) unsupported(4) -> count at
        // offset 11
        const u32 huge = 0x7FFF'FFFFu;
        std::memcpy(garbage.data() + 11, &huge, sizeof(huge));
        VC::DescriptorLayout out;
        REQUIRE_FALSE(Disk::DeserializeDescriptorLayout(garbage, out));
    }
    SECTION("single-byte garbage never crashes") {
        for (size_t i = 0; i < bytes.size(); ++i) {
            auto garbage = bytes;
            garbage[i] = static_cast<u8>(garbage[i] ^ 0xFF);
            VC::DescriptorLayout out;
            (void)Disk::DeserializeDescriptorLayout(garbage, out);
        }
    }
}

TEST_CASE("Disk cache records round trip", "[nvn]") {
    const Disk::Header header = MakeHeader();
    const FileImage image = MakeImage(header);

    Disk::Contents contents;
    const Disk::LoadResult result = Disk::Parse(image.bytes, header, contents);
    REQUIRE(result.status == Disk::LoadStatus::Loaded);
    REQUIRE_FALSE(result.HasBadTail());
    REQUIRE(result.valid_bytes == image.bytes.size());
    REQUIRE(result.records == 5);
    REQUIRE(contents.shaders.size() == 3);
    REQUIRE(contents.graphics.size() == 1);
    REQUIRE(contents.compute.size() == 1);

    RequireSameShader(contents.shaders[0], image.vertex);
    RequireSameShader(contents.shaders[1], image.fragment);
    RequireSameShader(contents.shaders[2], image.compute_shader);

    const Disk::GraphicsPipelineRecord& graphics = contents.graphics[0];
    REQUIRE(graphics.shaders == image.graphics.shaders);
    REQUIRE(std::memcmp(&graphics.state, &image.graphics.state, sizeof(GraphicsFixedState)) == 0);
    REQUIRE(graphics.targets == image.graphics.targets);
    REQUIRE(graphics.push_constant_size == 44);
    REQUIRE(graphics.viewport_count == 16);

    const Disk::ComputePipelineRecord& compute = contents.compute[0];
    REQUIRE(compute.shader == image.compute.shader);
    REQUIRE(compute.environment_flags == 3);
    REQUIRE(compute.local_memory_size == 64);
    REQUIRE(compute.shared_memory_size == 2048);
    REQUIRE(compute.compute_layout.uniform_offset == 0x10);
    REQUIRE(compute.compute_layout.image_count == 1);
    REQUIRE(compute.compute_layout.total_matches);
    REQUIRE(compute.compute_layout.declared_total == 0x60);
}

TEST_CASE("Disk cache survives truncation at every byte offset", "[nvn]") {
    const Disk::Header header = MakeHeader();
    const FileImage image = MakeImage(header);
    for (size_t size = 0; size <= image.bytes.size(); ++size) {
        Disk::Contents contents;
        const Disk::LoadResult result =
            Disk::Parse(std::span<const u8>{image.bytes.data(), size}, header, contents);
        if (size == 0) {
            REQUIRE(result.status == Disk::LoadStatus::Missing);
            continue;
        }
        if (size < image.header_size) {
            REQUIRE(result.status == Disk::LoadStatus::Rejected);
            REQUIRE(CountContents(contents) == 0);
            continue;
        }
        REQUIRE(result.status == Disk::LoadStatus::Loaded);
        size_t complete = 0;
        size_t valid = image.header_size;
        for (const size_t end : image.record_ends) {
            if (end <= size) {
                ++complete;
                valid = end;
            }
        }
        REQUIRE(result.records == complete);
        REQUIRE(result.valid_bytes == valid);
        REQUIRE(result.HasBadTail() == (valid != size));
    }
}

TEST_CASE("Disk cache keeps the valid prefix when a record is damaged", "[nvn]") {
    const Disk::Header header = MakeHeader();
    const FileImage image = MakeImage(header);

    size_t start = image.header_size;
    for (size_t record = 0; record < image.record_ends.size(); ++record) {
        const size_t end = image.record_ends[record];
        for (size_t offset = start; offset < end; ++offset) {
            auto damaged = image.bytes;
            damaged[offset] ^= 0x40;
            Disk::Contents contents;
            const Disk::LoadResult result = Disk::Parse(damaged, header, contents);
            REQUIRE(result.status == Disk::LoadStatus::Loaded);
            REQUIRE(result.records == record);
            REQUIRE(result.valid_bytes == start);
            REQUIRE(result.HasBadTail());
            REQUIRE_FALSE(result.reason.empty());
        }
        start = end;
    }
}

TEST_CASE("Disk cache rejects a damaged or foreign header", "[nvn]") {
    const Disk::Header header = MakeHeader();
    const FileImage image = MakeImage(header);

    SECTION("a flipped byte anywhere in the header") {
        for (size_t offset = 0; offset < image.header_size; ++offset) {
            auto damaged = image.bytes;
            damaged[offset] ^= 0x01;
            Disk::Contents contents;
            const Disk::LoadResult result = Disk::Parse(damaged, header, contents);
            REQUIRE(result.status == Disk::LoadStatus::Rejected);
            REQUIRE(CountContents(contents) == 0);
        }
    }
    SECTION("garbage") {
        std::vector<u8> garbage(512);
        std::mt19937 random(7);
        for (u8& byte : garbage) {
            byte = static_cast<u8>(random());
        }
        Disk::Contents contents;
        REQUIRE(Disk::Parse(garbage, header, contents).status == Disk::LoadStatus::Rejected);
        REQUIRE(CountContents(contents) == 0);
    }
    SECTION("another build revision with the same translation key is accepted") {
        Disk::Header other = header;
        other.revision = "another-build";
        Disk::Contents expected;
        REQUIRE(Disk::Parse(image.bytes, header, expected).status == Disk::LoadStatus::Loaded);
        REQUIRE(CountContents(expected) > 0);
        Disk::Contents contents;
        const auto result = Disk::Parse(image.bytes, other, contents);
        REQUIRE(result.status == Disk::LoadStatus::Loaded);
        REQUIRE(CountContents(contents) == CountContents(expected));
    }
    SECTION("another translation key is rejected whatever the revision") {
        Disk::Header other = header;
        other.translation_key = "test-translation-key-2";
        Disk::Contents contents;
        const auto result = Disk::Parse(image.bytes, other, contents);
        REQUIRE(result.status == Disk::LoadStatus::Rejected);
        REQUIRE(result.reason.find("translation inputs") != std::string::npos);
        REQUIRE(CountContents(contents) == 0);
        other.revision = header.revision + "-other";
        REQUIRE(Disk::Parse(image.bytes, other, contents).status == Disk::LoadStatus::Rejected);
        REQUIRE(CountContents(contents) == 0);
    }
    SECTION("another device, driver, pipeline-cache uuid or profile") {
        const auto rejected = [&](const Disk::Header& other) {
            Disk::Contents contents;
            const auto result = Disk::Parse(image.bytes, other, contents);
            return result.status == Disk::LoadStatus::Rejected && CountContents(contents) == 0;
        };
        Disk::Header other = header;
        other.vendor_id ^= 1;
        REQUIRE(rejected(other));
        other = header;
        other.device_id ^= 1;
        REQUIRE(rejected(other));
        other = header;
        other.driver_version ^= 1;
        REQUIRE(rejected(other));
        other = header;
        other.cache_uuid_hash ^= 1;
        REQUIRE(rejected(other));
        other = header;
        other.profile_hash ^= 1;
        REQUIRE(rejected(other));
    }
    SECTION("another format version (valid checksum)") {
        auto other_version = image.bytes;
        const u32 version = Disk::FORMAT_VERSION + 1;
        std::memcpy(other_version.data() + 8, &version, sizeof(version));
        const u32 crc = Disk::Crc32(std::span<const u8>{other_version.data(), image.header_size - 4});
        std::memcpy(other_version.data() + image.header_size - 4, &crc, sizeof(crc));
        Disk::Contents contents;
        const auto result = Disk::Parse(other_version, header, contents);
        REQUIRE(result.status == Disk::LoadStatus::Rejected);
        REQUIRE(CountContents(contents) == 0);
    }
}

TEST_CASE("Disk cache file helpers append, cut a bad tail and replace rejected files", "[nvn]") {
    TempDirectory directory;
    const auto path = directory.path / "nvn_pipelines.bin";
    const Disk::Header header = MakeHeader();
    const FileImage image = MakeImage(header);

    SECTION("missing file") {
        Disk::Contents contents;
        const auto result = Disk::LoadFile(path, header, contents);
        REQUIRE(result.status == Disk::LoadStatus::Missing);
        REQUIRE(Disk::PrepareFile(path, header, result));
        Disk::Contents again;
        const auto reloaded = Disk::LoadFile(path, header, again);
        REQUIRE(reloaded.status == Disk::LoadStatus::Loaded);
        REQUIRE(reloaded.records == 0);
    }
    SECTION("append after a cut tail") {
        auto damaged = image.bytes;
        damaged.insert(damaged.end(), {1, 2, 3, 4, 5, 6, 7});
        WriteBytes(path, damaged);
        Disk::Contents contents;
        const auto result = Disk::LoadFile(path, header, contents);
        REQUIRE(result.status == Disk::LoadStatus::Loaded);
        REQUIRE(result.HasBadTail());
        REQUIRE(result.records == 5);
        REQUIRE(Disk::PrepareFile(path, header, result));
        {
            Disk::Appender appender;
            REQUIRE(appender.Open(path));
            REQUIRE(appender.Append(Disk::EncodeShader(image.vertex)));
        }
        Disk::Contents again;
        const auto reloaded = Disk::LoadFile(path, header, again);
        REQUIRE(reloaded.status == Disk::LoadStatus::Loaded);
        REQUIRE_FALSE(reloaded.HasBadTail());
        REQUIRE(reloaded.records == 6);
        REQUIRE(again.shaders.size() == 4);
    }
    SECTION("rejected file is replaced by a fresh one") {
        Disk::Header other = header;
        other.translation_key = "old-translation-key";
        const FileImage old_image = MakeImage(other);
        WriteBytes(path, old_image.bytes);
        Disk::Contents contents;
        const auto result = Disk::LoadFile(path, header, contents);
        REQUIRE(result.status == Disk::LoadStatus::Rejected);
        REQUIRE(Disk::PrepareFile(path, header, result));
        Disk::Contents again;
        const auto reloaded = Disk::LoadFile(path, header, again);
        REQUIRE(reloaded.status == Disk::LoadStatus::Loaded);
        REQUIRE(reloaded.records == 0);
    }
}

// --- regression: a DK-written file failed to load ("malformed record payload after 4 valid
// records"): the reader bounded the texture pixel format of a query at R16G16B16A16_SNORM, but the
// enum runs to D32_FLOAT_S8_UINT, and the fifth record of the file queried B10G11R11_FLOAT. ------

namespace {

constexpr u32 LAST_PIXEL_FORMAT = static_cast<u32>(Shader::TexturePixelFormat::D32_FLOAT_S8_UINT);

/// Every binding kind, role, storage source, stage, handle source, texture type and image format,
/// with and without issue strings (empty, short, long, non-ASCII).
VC::DescriptorLayout MakeFullLayout(u32 variant) {
    constexpr std::array kinds{VC::DescriptorKind::CombinedImageSampler,
                               VC::DescriptorKind::StorageImage,
                               VC::DescriptorKind::UniformTexelBuffer,
                               VC::DescriptorKind::StorageTexelBuffer,
                               VC::DescriptorKind::UniformBuffer,
                               VC::DescriptorKind::StorageBuffer};
    VC::DescriptorLayout layout;
    layout.binding_count = 40 + variant;
    layout.uses_rescaling_uniform = (variant & 1) != 0;
    layout.uses_render_area = (variant & 2) != 0;
    for (u32 i = 0; i < 72; ++i) {
        const u32 n = i + variant;
        VC::DescriptorBinding b;
        b.binding = i;
        b.kind = kinds[n % kinds.size()];
        b.role = static_cast<VC::DescriptorRole>(n % 8);
        b.count = 1 + n % 4;
        b.slot = n * 3;
        b.storage_source = static_cast<VC::StorageSource>(n % 5);
        b.storage_bank = n % 7;
        b.storage_offset = 0x10 * n;
        b.stage = static_cast<VC::ShaderStage>(n % 6);
        b.stage_mask = 1u << (n % 6);
        b.texture = {static_cast<VC::HandleSource>(n % 9), n, 2, 0x48 + 4 * n};
        b.sampler = {static_cast<VC::HandleSource>((n + 3) % 9), n + 1, 2, 0x568 + 4 * n};
        b.type = static_cast<Shader::TextureType>(n % 9);
        b.image_format = static_cast<Shader::ImageFormat>(n % 8);
        b.is_depth = (n & 1) != 0;
        b.is_multisample = (n & 2) != 0;
        b.is_integer = (n & 4) != 0;
        b.is_written = (n & 8) != 0;
        b.is_read = (n & 16) != 0;
        b.supported = n % 5 != 0;
        if (!b.supported) {
            ++layout.unsupported_count;
            b.issue = n % 11 == 0 ? std::string(300, 'x') + " \xC3\xA9"
                                  : "texel buffer descriptors are not supported (binding " +
                                        std::to_string(i) + ")";
        }
        layout.bindings.push_back(std::move(b));
    }
    return layout;
}

/// One query per TexturePixelFormat value, the texture types and stages cycling.
std::vector<TextureQuery> MakeFormatQueries(u32 variant) {
    std::vector<TextureQuery> queries;
    for (u32 format = 0; format <= LAST_PIXEL_FORMAT; ++format) {
        TextureQuery q;
        q.stage = (format + variant) % 5;
        q.record = {2 + format % 3, 0x20 + 8 * format};
        q.info = {static_cast<Shader::TextureType>((format + variant) % 9),
                  static_cast<Shader::TexturePixelFormat>(format), ((format + variant) & 1) != 0};
        queries.push_back(q);
    }
    return queries;
}

Disk::ShaderRecord MakeDkShader(u32 stage, u64 program_hash, u32 variant) {
    Disk::ShaderRecord record = MakeShader(stage, program_hash, 64 + variant * 37);
    record.has_program_constants = (variant % 3) == 0;
    record.program_constants.assign(record.has_program_constants ? 8 + variant : 0, 0xC1C1C1C1u);
    record.layout = MakeFullLayout(variant);
    record.queries = MakeFormatQueries(variant);
    record.id = Disk::MakeContentId(stage, program_hash, 0x77 + variant,
                                    Disk::HashQueries(record.queries));
    return record;
}

/// A framed record the way the writer frames them, around any payload (for damage tests).
std::vector<u8> FrameRecord(u32 type, std::span<const u8> payload) {
    std::vector<u8> out;
    const auto put = [&out](u32 value) {
        const auto* const first = reinterpret_cast<const u8*>(&value);
        out.insert(out.end(), first, first + sizeof(value));
    };
    put(Disk::RECORD_TAG);
    put(type);
    put(static_cast<u32>(payload.size()));
    std::vector<u8> crc_input;
    for (const u32 value : {type, static_cast<u32>(payload.size())}) {
        const auto* const first = reinterpret_cast<const u8*>(&value);
        crc_input.insert(crc_input.end(), first, first + sizeof(value));
    }
    put(Disk::Crc32(payload, Disk::Crc32(crc_input)));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void Append(std::vector<u8>& file, const std::vector<u8>& record) {
    file.insert(file.end(), record.begin(), record.end());
}

} // Anonymous namespace

TEST_CASE("Disk cache loads the file a DK run wrote: late pixel formats in texture queries",
          "[nvn]") {
    const Disk::Header header = MakeHeader();

    SECTION("the failing file: four plain shaders, then a query of B10G11R11_FLOAT") {
        std::vector<u8> file = Disk::EncodeHeader(header);
        std::vector<Disk::ShaderRecord> expected;
        for (u32 i = 0; i < 4; ++i) {
            Disk::ShaderRecord plain = MakeShader(i % 2, 0x1000 + i, 50 + i);
            plain.queries.clear();
            Append(file, Disk::EncodeShader(plain));
            expected.push_back(std::move(plain));
        }
        Disk::ShaderRecord fifth = MakeShader(1, 0x2000, 60);
        TextureQuery query;
        query.stage = 1;
        query.record = {2, 32};
        query.info = {Shader::TextureType::Color2D, Shader::TexturePixelFormat::B10G11R11_FLOAT,
                      false};
        fifth.queries = {query};
        Append(file, Disk::EncodeShader(fifth));
        expected.push_back(std::move(fifth));

        Disk::Contents contents;
        const Disk::LoadResult result = Disk::Parse(file, header, contents);
        REQUIRE(result.status == Disk::LoadStatus::Loaded);
        REQUIRE(result.reason.empty());
        REQUIRE_FALSE(result.HasBadTail());
        REQUIRE(result.records == 5);
        REQUIRE(contents.shaders.size() == 5);
        for (size_t i = 0; i < expected.size(); ++i) {
            RequireSameShader(contents.shaders[i], expected[i]);
        }
    }

    SECTION("every pixel format, texture type and image format alone in a query") {
        for (u32 format = 0; format <= LAST_PIXEL_FORMAT; ++format) {
            Disk::ShaderRecord shader = MakeShader(1, 0x5000 + format, 48);
            shader.queries = {MakeFormatQueries(0)[format]};
            std::vector<u8> file = Disk::EncodeHeader(header);
            Append(file, Disk::EncodeShader(shader));
            Disk::Contents contents;
            const Disk::LoadResult result = Disk::Parse(file, header, contents);
            INFO("pixel format " << format);
            REQUIRE(result.records == 1);
            REQUIRE_FALSE(result.HasBadTail());
            RequireSameShader(contents.shaders[0], shader);
        }
    }
}

TEST_CASE("Disk cache round trips a DK-like file written by several threads", "[nvn]") {
    const Disk::Header header = MakeHeader();

    // Shader records: every stage, c1 words or none, full descriptor layouts, a query per pixel
    // format. Encoded on four threads (the real writer encodes on the replay and loader threads).
    constexpr u32 SHADER_COUNT = 24;
    constexpr std::array<u32, 8> STAGES{0, 1, 1, 0, 2, 3, 4, 5};
    std::vector<Disk::ShaderRecord> shaders(SHADER_COUNT);
    std::vector<std::vector<u8>> shader_bytes(SHADER_COUNT);
    {
        std::vector<std::thread> threads;
        for (u32 t = 0; t < 4; ++t) {
            threads.emplace_back([&, t] {
                for (u32 i = t; i < SHADER_COUNT; i += 4) {
                    shaders[i] = MakeDkShader(STAGES[i % STAGES.size()], 0xA000 + i, i);
                    shader_bytes[i] = Disk::EncodeShader(shaders[i]);
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    // Pipeline records: vertex + fragment (+ optional geometry) of the shaders above.
    std::vector<Disk::GraphicsPipelineRecord> pipelines;
    std::vector<std::vector<u8>> pipeline_bytes;
    for (u32 i = 0; i < 160; ++i) {
        Disk::GraphicsPipelineRecord record =
            MakeGraphics(shaders[(i * 2) % SHADER_COUNT].id, shaders[(i * 3 + 1) % SHADER_COUNT].id);
        record.state.topology = i % 7;
        record.state.polygon = i * 0x9E3779B9u;
        record.state.sample_mask = ~i;
        record.state.stream_stride[i % 16] = 4 * i;
        record.targets.color[1] = static_cast<s32>(i);
        record.push_constant_size = 4 * (i % 12);
        record.viewport_count = 1 + i % 16;
        pipeline_bytes.push_back(Disk::EncodeGraphicsPipeline(record));
        pipelines.push_back(record);
    }
    std::vector<Disk::ComputePipelineRecord> computes;
    std::vector<std::vector<u8>> compute_bytes;
    for (u32 i = 0; i < 6; ++i) {
        Disk::ComputePipelineRecord record = MakeCompute(shaders[7].id);
        record.local_memory_size = 16 * i;
        computes.push_back(record);
        compute_bytes.push_back(Disk::EncodeComputePipeline(record));
    }

    // File: shaders first, then the pipelines interleaved with the compute records.
    std::vector<u8> file = Disk::EncodeHeader(header);
    for (const auto& bytes : shader_bytes) {
        Append(file, bytes);
    }
    for (size_t i = 0; i < pipeline_bytes.size(); ++i) {
        Append(file, pipeline_bytes[i]);
        if (i % 30 == 29) {
            Append(file, compute_bytes[i / 30]);
        }
    }

    Disk::Contents contents;
    const Disk::LoadResult result = Disk::Parse(file, header, contents);
    REQUIRE(result.status == Disk::LoadStatus::Loaded);
    REQUIRE(result.reason.empty());
    REQUIRE_FALSE(result.HasBadTail());
    REQUIRE(result.valid_bytes == file.size());
    REQUIRE(result.records == SHADER_COUNT + pipelines.size() + 5);
    REQUIRE(contents.shaders.size() == SHADER_COUNT);
    REQUIRE(contents.graphics.size() == pipelines.size());
    REQUIRE(contents.compute.size() == 5);
    for (size_t i = 0; i < shaders.size(); ++i) {
        RequireSameShader(contents.shaders[i], shaders[i]);
    }
    for (size_t i = 0; i < pipelines.size(); ++i) {
        REQUIRE(contents.graphics[i].shaders == pipelines[i].shaders);
        REQUIRE(std::memcmp(&contents.graphics[i].state, &pipelines[i].state,
                            sizeof(GraphicsFixedState)) == 0);
        REQUIRE(contents.graphics[i].targets == pipelines[i].targets);
        REQUIRE(contents.graphics[i].push_constant_size == pipelines[i].push_constant_size);
        REQUIRE(contents.graphics[i].viewport_count == pipelines[i].viewport_count);
    }
    for (size_t i = 0; i < contents.compute.size(); ++i) {
        REQUIRE(contents.compute[i].shader == computes[i].shader);
        REQUIRE(contents.compute[i].local_memory_size == computes[i].local_memory_size);
    }

    SECTION("appended through the Appender from several threads, record by record") {
        TempDirectory directory;
        const auto path = directory.path / "nvn_pipelines.bin";
        {
            Disk::Contents none;
            const auto load = Disk::LoadFile(path, header, none);
            REQUIRE(Disk::PrepareFile(path, header, load));
        }
        {
            // One owner appends (the cache's writer thread); producers hand it finished records.
            Disk::Appender appender;
            REQUIRE(appender.Open(path));
            std::mutex mutex;
            std::atomic<int> append_failures{0};
            std::vector<std::thread> threads;
            for (u32 t = 0; t < 4; ++t) {
                threads.emplace_back([&, t] {
                    for (u32 i = t; i < SHADER_COUNT; i += 4) {
                        const std::vector<u8> bytes = Disk::EncodeShader(shaders[i]);
                        std::scoped_lock lock{mutex};
                        if (!appender.Append(bytes)) {
                            ++append_failures;
                        }
                    }
                });
            }
            for (std::thread& thread : threads) {
                thread.join();
            }
            REQUIRE(append_failures.load() == 0);
            for (const auto& bytes : pipeline_bytes) {
                REQUIRE(appender.Append(bytes));
            }
        }
        Disk::Contents again;
        const auto loaded = Disk::LoadFile(path, header, again);
        REQUIRE(loaded.status == Disk::LoadStatus::Loaded);
        REQUIRE(loaded.reason.empty());
        REQUIRE_FALSE(loaded.HasBadTail());
        REQUIRE(again.shaders.size() == SHADER_COUNT);
        REQUIRE(again.graphics.size() == pipelines.size());
    }
}

TEST_CASE("Disk cache: a malformed record ends the parse, the tail is cut before the next append",
          "[nvn]") {
    TempDirectory directory;
    const auto path = directory.path / "nvn_pipelines.bin";
    const Disk::Header header = MakeHeader();

    std::vector<u8> file = Disk::EncodeHeader(header);
    std::vector<Disk::ShaderRecord> good;
    for (u32 i = 0; i < 4; ++i) {
        good.push_back(MakeDkShader(i % 2, 0xB000 + i, i));
        Append(file, Disk::EncodeShader(good.back()));
    }
    const size_t valid_end = file.size();

    // A record whose frame (tag, size, checksum) is intact and whose payload is not.
    std::vector<u8> garbage(333);
    std::mt19937 random(5);
    for (u8& byte : garbage) {
        byte = static_cast<u8>(random());
    }
    std::vector<u8> bad_record;
    SECTION("garbage payload") {
        bad_record = FrameRecord(static_cast<u32>(Disk::RecordType::Shader), garbage);
    }
    SECTION("a pixel format beyond the enum") {
        Disk::ShaderRecord shader = MakeDkShader(1, 0xB100, 9);
        std::vector<u8> encoded = Disk::EncodeShader(shader);
        // The queries are the tail of the payload: count(4) then 21-byte entries, the format is
        // the fifth u32 of an entry. Patch the last entry and re-frame.
        std::vector<u8> payload(encoded.begin() + Disk::FRAME_BYTES, encoded.end());
        const u32 beyond = LAST_PIXEL_FORMAT + 1;
        std::memcpy(payload.data() + payload.size() - 5, &beyond, sizeof(beyond));
        bad_record = FrameRecord(static_cast<u32>(Disk::RecordType::Shader), payload);
    }
    SECTION("a graphics record with a wrong state size") {
        std::vector<u8> encoded = Disk::EncodeGraphicsPipeline(MakeGraphics(good[0].id, good[1].id));
        std::vector<u8> payload(encoded.begin() + Disk::FRAME_BYTES, encoded.end());
        const u32 wrong = static_cast<u32>(sizeof(GraphicsFixedState)) - 8;
        // The state size follows one 16-byte content id per stage slot.
        std::memcpy(payload.data() + Vulkan::Nvn::GRAPHICS_STAGE_COUNT * 16, &wrong, sizeof(wrong));
        bad_record = FrameRecord(static_cast<u32>(Disk::RecordType::GraphicsPipeline), payload);
    }
    Append(file, bad_record);
    // Valid records after the damaged one are dropped with it: the file is only ever appended to,
    // so nothing after a malformed record can be trusted to belong to this layout.
    const Disk::ShaderRecord after = MakeDkShader(0, 0xB200, 3);
    Append(file, Disk::EncodeShader(after));

    Disk::Contents contents;
    const Disk::LoadResult result = Disk::Parse(file, header, contents);
    REQUIRE(result.status == Disk::LoadStatus::Loaded);
    REQUIRE(result.HasBadTail());
    REQUIRE(result.records == 4);
    REQUIRE(result.valid_bytes == valid_end);
    REQUIRE(result.reason.find("malformed record payload") != std::string::npos);
    REQUIRE(contents.shaders.size() == 4);
    REQUIRE(contents.graphics.empty());
    for (size_t i = 0; i < good.size(); ++i) {
        RequireSameShader(contents.shaders[i], good[i]);
    }

    WriteBytes(path, file);
    Disk::Contents loaded;
    const Disk::LoadResult load = Disk::LoadFile(path, header, loaded);
    REQUIRE(load.HasBadTail());
    REQUIRE(Disk::PrepareFile(path, header, load));
    REQUIRE(std::filesystem::file_size(path) == valid_end);
    {
        Disk::Appender appender;
        REQUIRE(appender.Open(path));
        REQUIRE(appender.Append(Disk::EncodeShader(after)));
    }
    Disk::Contents again;
    const Disk::LoadResult reloaded = Disk::LoadFile(path, header, again);
    REQUIRE(reloaded.status == Disk::LoadStatus::Loaded);
    REQUIRE_FALSE(reloaded.HasBadTail());
    REQUIRE(reloaded.records == 5);
    RequireSameShader(again.shaders[4], after);
}

// --- key stability ----------------------------------------------------------------------------

namespace {

/// A state a draw can really have: `i` picks one of a few dozen meaningful combinations.
GraphicsFixedState MakeMeaningfulState(u32 i) {
    GraphicsFixedState s{};
    s.topology = 1 + i % 7;
    s.polygon = VC::PolygonState::WithCullFace(VC::PolygonState::Defaults(), i % 3);
    s.color = VC::ColorState::Defaults();
    s.channel_mask = (i % 5) == 0 ? 0x7777'7777u : 0xFFFF'FFFFu;
    s.sample_mask = 0xFFFF'FFFFu;
    s.depth_clamp = (i / 7) % 2;
    s.provoking_first = (i / 11) % 2;
    s.depth_mode = 1;
    s.window_origin = 1;
    s.depth_stencil = VC::DepthStencilState::Defaults();
    if ((i % 4) < 2) {
        s.depth_stencil = VC::DepthStencilState::WithDepthTestEnable(s.depth_stencil, 1);
    }
    s.blend.fill(VC::BlendState::Defaults());
    if ((i % 3) != 0) {
        s.color = VC::ColorState::WithBlendEnabled(s.color, 0, true);
        s.blend[0] = VC::BlendState::WithDstRgbFunc(
            VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), 5 + i % 3), 6);
    }
    s.attributes[0] = 0x25u | (16u << 8);
    s.attributes[1] = 0x29u | (8u << 8) | (1u << 27);
    s.stream_stride[0] = 32;
    s.stream_stride[1] = 8 * (1 + i % 3);
    s.stream_divisor[1] = i % 2;
    s.multisample[2] = (i % 6) == 0 ? 1 : 0;
    return s;
}

/// What the guest leaves in the bits no field of the state objects covers, and in the fields the
/// pipeline does not read, differently on every call.
void AddGuestGarbage(GraphicsFixedState& s, std::mt19937_64& random) {
    constexpr u64 HIGH_DS = ~u64{0} << (VC::DepthStencilState::BackDepthPassOp_SHIFT +
                                        VC::DepthStencilState::BackDepthPassOp_WIDTH);
    if (s.provoking_first != 0) {
        s.provoking_first = 1 + static_cast<u32>(random() % 1000);
    }
    s.pad1 = static_cast<u32>(random());
    s.restart_index = static_cast<u32>(random());
    s.polygon |= static_cast<u32>(random()) & ~0xFFu;
    s.color |= static_cast<u32>(random()) & 0xFFFF'0000u;
    s.depth_stencil |= random() & HIGH_DS;
    for (u32 slot = 0; slot < s.blend.size(); ++slot) {
        if (VC::ColorState::BlendEnabled(s.color, static_cast<s32>(slot))) {
            s.blend[slot] = (s.blend[slot] & ~u64{7}) | (random() & 7) | (random() << 43);
        } else {
            s.blend[slot] = random();
        }
    }
    for (const u32 offset : {0u, 1u, 4u, 7u, 9u, 11u, 12u, 13u, 14u, 15u}) {
        s.multisample[offset] = static_cast<u8>(random());
    }
    if (s.multisample[2] != 0) {
        s.multisample[2] = static_cast<u8>(1 + random() % 255);
    }
    if (s.depth_clamp != 0) {
        s.depth_clamp = 1 + static_cast<u32>(random() % 1000);
    }
    for (u32& word : s.attributes) {
        if (VC::VertexAttribState::Format(word) == 0) {
            word = static_cast<u32>(random()) & ~0xFFu; // format 0: offset and stream are noise
        }
    }
    for (u32 stream = 2; stream < s.stream_stride.size(); ++stream) {
        s.stream_stride[stream] = static_cast<u32>(random()); // no attribute points at them
        s.stream_divisor[stream] = static_cast<u32>(random());
    }
    if (VC::ColorState::AlphaFunc(s.color) == VC::STATE_ALPHA_FUNC_ALWAYS) {
        s.alpha_ref_bits = static_cast<u32>(random());
    }
}

std::string Bytes(const GraphicsFixedState& s) {
    return std::string(reinterpret_cast<const char*>(&s), sizeof(s));
}

/// The Static key form of a draw's state (what the cache keys on when the device has no
/// VK_EXT_vertex_input_dynamic_state).
void StaticKey(GraphicsFixedState& s) {
    Vulkan::Nvn::MakeKeyState(s, Vulkan::Nvn::VertexInputMode::Static, false);
}

GraphicsFixedState KeyOf(GraphicsFixedState s, Vulkan::Nvn::VertexInputMode mode) {
    Vulkan::Nvn::MakeKeyState(s, mode, false);
    return s;
}

} // Anonymous namespace

TEST_CASE("CanonicalizeState: keys survive guest garbage and a relaunch", "[nvn]") {
    constexpr u32 MEANINGFUL = 42;

    SECTION("garbage folds away, idempotent") {
        std::mt19937_64 random(1);
        for (u32 i = 0; i < MEANINGFUL; ++i) {
            GraphicsFixedState clean = MakeMeaningfulState(i);
            GraphicsFixedState noisy = clean;
            AddGuestGarbage(noisy, random);
            REQUIRE(Bytes(noisy) != Bytes(clean));
            StaticKey(clean);
            StaticKey(noisy);
            REQUIRE(Bytes(noisy) == Bytes(clean));
            GraphicsFixedState twice = noisy;
            Disk::CanonicalizeState(twice, Vulkan::Nvn::VertexInputMode::Static);
            REQUIRE(Bytes(twice) == Bytes(noisy));
        }
    }

    SECTION("two launches with different garbage see the same set of keys") {
        // 3000 draws per launch cycling through 42 meaningful states, new garbage in every draw:
        // the DK file held 22656 pipelines for 533 real ones.
        const auto launch = [&](u64 seed) {
            std::mt19937_64 random(seed);
            std::set<std::string> keys;
            for (u32 draw = 0; draw < 3000; ++draw) {
                GraphicsFixedState state = MakeMeaningfulState(draw % MEANINGFUL);
                AddGuestGarbage(state, random);
                StaticKey(state);
                keys.insert(Bytes(state));
            }
            return keys;
        };
        const std::set<std::string> first = launch(11);
        const std::set<std::string> second = launch(987654321);
        std::set<std::string> distinct_meaning;
        for (u32 i = 0; i < MEANINGFUL; ++i) {
            GraphicsFixedState state = MakeMeaningfulState(i);
            StaticKey(state);
            distinct_meaning.insert(Bytes(state));
        }
        REQUIRE(first == distinct_meaning);
        REQUIRE(second == distinct_meaning);
    }

    SECTION("what the pipeline reads still tells states apart") {
        GraphicsFixedState base = MakeMeaningfulState(1); // blends target 0, depth test on
        StaticKey(base);
        const auto differs = [&base](auto&& change) {
            GraphicsFixedState other = MakeMeaningfulState(1);
            change(other);
            StaticKey(other);
            return Bytes(other) != Bytes(base);
        };
        REQUIRE(differs([](GraphicsFixedState& s) { s.topology = 4; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.polygon ^= 1; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.polygon ^= 1u << 7; })); // offset enables
        REQUIRE(differs([](GraphicsFixedState& s) { s.color ^= 1u << 8; }));   // logic op
        REQUIRE(differs([](GraphicsFixedState& s) { s.channel_mask ^= 0x10; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.sample_mask = 1; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.provoking_first = 1; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.depth_stencil ^= 1u << 1; })); // depth write
        REQUIRE(differs([](GraphicsFixedState& s) { s.depth_stencil ^= 1ull << 6; })); // stencil on
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.depth_stencil ^= 1ull << 38; })); // off
        REQUIRE(differs([](GraphicsFixedState& s) { s.blend[0] ^= 1ull << 10; }));  // dst rgb func
        REQUIRE(differs([](GraphicsFixedState& s) { s.blend[0] ^= 1ull << 37; }));  // adv. mode
        REQUIRE(differs([](GraphicsFixedState& s) { s.attributes[0] ^= 0x1; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.attributes[1] ^= 1u << 27; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.stream_stride[0] += 4; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.stream_divisor[1] ^= 1; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.multisample[2] = 1; }));      // a-to-c
        REQUIRE(differs([](GraphicsFixedState& s) { s.multisample[10] = 1; }));     // truncation
        REQUIRE(differs([](GraphicsFixedState& s) { s.depth_mode = 0; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.window_origin = 0; }));
        REQUIRE(differs([](GraphicsFixedState& s) { s.point_size_bits = 0x3F800000u; }));
        // ... and what it does not read does not.
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.blend[1] ^= 1ull << 10; })); // slot off
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.blend[0] ^= 7; }));          // index
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.restart_index = 0xFFFF; }));
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.polygon |= 0x100; }));
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.multisample[1] = 4; }));
        REQUIRE_FALSE(differs([](GraphicsFixedState& s) { s.alpha_ref_bits = 0x3F000000u; }));
    }
}

TEST_CASE("Vertex input mode: the dynamic key holds type classes, not the vertex layout", "[nvn]") {
    using Vulkan::Nvn::VertexInputMode;
    constexpr u32 F_RGBA8 = 0x25;
    constexpr u32 F_RGBA8UI = 0x27;
    constexpr u32 F_RGBA8I = 0x28;
    constexpr u32 F_RGBA16F = 0x29;
    constexpr u32 F_RGBA16 = 0x2A;
    constexpr u32 STREAM_SHIFT = 27;
    const auto with_format = [](u32 word, u32 format) { return (word & ~0xFFu) | format; };

    const GraphicsFixedState base = MakeMeaningfulState(1); // attributes 0 and 1: float RGBA
    const auto same_key = [&base](VertexInputMode mode, auto&& change) {
        GraphicsFixedState other = base;
        change(other);
        return Bytes(KeyOf(other, mode)) == Bytes(KeyOf(base, mode));
    };

    SECTION("states that differ only in the vertex layout share a key when dynamic") {
        const auto layout_only = [&](auto&& change) {
            REQUIRE(same_key(VertexInputMode::Dynamic, change));
            REQUIRE_FALSE(same_key(VertexInputMode::Static, change));
        };
        layout_only([](GraphicsFixedState& s) { s.attributes[0] += 4u << 8; }); // offset
        layout_only([](GraphicsFixedState& s) { s.attributes[1] += 8u << 8; });
        layout_only([&](GraphicsFixedState& s) {
            s.attributes[0] = with_format(s.attributes[0], F_RGBA16F); // another float format
        });
        layout_only([&](GraphicsFixedState& s) {
            s.attributes[1] = with_format(s.attributes[1], F_RGBA16);
        });
        layout_only([](GraphicsFixedState& s) { s.attributes[1] ^= 2u << STREAM_SHIFT; }); // stream
        layout_only([](GraphicsFixedState& s) { s.stream_stride[0] += 4; });
        layout_only([](GraphicsFixedState& s) { s.stream_stride[1] = 0; });
        layout_only([](GraphicsFixedState& s) { s.stream_divisor[1] ^= 1; });
        layout_only([](GraphicsFixedState& s) { s.stream_divisor[0] = 3; }); // instanced stream 0
    }

    SECTION("a type class change is a different key in both modes") {
        for (const VertexInputMode mode : {VertexInputMode::Static, VertexInputMode::Dynamic}) {
            REQUIRE_FALSE(same_key(mode, [&](GraphicsFixedState& s) { // float -> uint
                s.attributes[0] = with_format(s.attributes[0], F_RGBA8UI);
            }));
            REQUIRE_FALSE(same_key(mode, [&](GraphicsFixedState& s) { // float -> sint
                s.attributes[1] = with_format(s.attributes[1], F_RGBA8I);
            }));
            REQUIRE_FALSE(same_key(mode, [](GraphicsFixedState& s) { s.attributes[0] = 0; })); // unused
            REQUIRE_FALSE(same_key(mode, [](GraphicsFixedState& s) { // a new location
                s.attributes[5] = 0x25u | (4u << 27);
            }));

            GraphicsFixedState as_uint = base;
            as_uint.attributes[0] = with_format(as_uint.attributes[0], F_RGBA8UI);
            GraphicsFixedState as_sint = base;
            as_sint.attributes[0] = with_format(as_sint.attributes[0], F_RGBA8I);
            REQUIRE(Bytes(KeyOf(as_uint, mode)) != Bytes(KeyOf(as_sint, mode)));
        }
        // ... while formats of one class stay one key whatever else changed with them.
        REQUIRE(same_key(VertexInputMode::Dynamic, [&](GraphicsFixedState& s) {
            s.attributes[0] = with_format(s.attributes[0], F_RGBA8) + (64u << 8);
        }));
    }

    SECTION("the two key forms") {
        const GraphicsFixedState dynamic_key = KeyOf(base, VertexInputMode::Dynamic);
        const GraphicsFixedState static_key = KeyOf(base, VertexInputMode::Static);
        const auto zero = [](const auto& array) {
            return std::all_of(array.begin(), array.end(), [](auto v) { return v == 0; });
        };
        constexpr u8 TYPE_FLOAT = static_cast<u8>(Shader::AttributeType::Float);
        constexpr u8 TYPE_DISABLED = static_cast<u8>(Shader::AttributeType::Disabled);

        REQUIRE(zero(dynamic_key.attributes));
        REQUIRE(zero(dynamic_key.stream_stride));
        REQUIRE(zero(dynamic_key.stream_divisor));
        REQUIRE(dynamic_key.vertex_types[0] == TYPE_FLOAT);
        REQUIRE(dynamic_key.vertex_types[1] == TYPE_FLOAT);
        for (u32 location = 2; location < dynamic_key.vertex_types.size(); ++location) {
            REQUIRE(dynamic_key.vertex_types[location] == TYPE_DISABLED);
        }

        REQUIRE(zero(static_key.vertex_types));
        REQUIRE(static_key.attributes[0] == base.attributes[0]);
        REQUIRE(static_key.stream_stride[0] == base.stream_stride[0]);
        REQUIRE(Bytes(static_key) != Bytes(dynamic_key));

        // Idempotent: the disk loader applies the canonical form to every stored record.
        GraphicsFixedState again = dynamic_key;
        Disk::CanonicalizeState(again, VertexInputMode::Dynamic);
        REQUIRE(Bytes(again) == Bytes(dynamic_key));
        again = static_key;
        Disk::CanonicalizeState(again, VertexInputMode::Static);
        REQUIRE(Bytes(again) == Bytes(static_key));
    }

    SECTION("guest garbage does not reach the dynamic key") {
        std::mt19937_64 random(5);
        for (u32 i = 0; i < 42; ++i) {
            const GraphicsFixedState clean = MakeMeaningfulState(i);
            GraphicsFixedState noisy = clean;
            AddGuestGarbage(noisy, random);
            REQUIRE(Bytes(KeyOf(noisy, VertexInputMode::Dynamic)) ==
                    Bytes(KeyOf(clean, VertexInputMode::Dynamic)));
        }
    }

    SECTION("many layouts of one state: one dynamic key, many static keys") {
        std::set<std::string> dynamic_keys;
        std::set<std::string> static_keys;
        constexpr std::array FLOAT_FORMATS{F_RGBA8, F_RGBA16F, F_RGBA16};
        for (u32 i = 0; i < 64; ++i) {
            GraphicsFixedState s = base;
            s.attributes[0] = FLOAT_FORMATS[i % 3] | ((4 * i) << 8) | ((i % 4) << STREAM_SHIFT);
            s.attributes[1] = FLOAT_FORMATS[(i + 1) % 3] | ((8 * i) << 8) | (((i + 1) % 4) << STREAM_SHIFT);
            s.stream_stride[i % 4] = 4 * (1 + i % 5);
            s.stream_stride[(i + 1) % 4] = 8 * (1 + i % 3);
            s.stream_divisor[i % 4] = i % 3;
            dynamic_keys.insert(Bytes(KeyOf(s, VertexInputMode::Dynamic)));
            static_keys.insert(Bytes(KeyOf(s, VertexInputMode::Static)));
        }
        REQUIRE(dynamic_keys.size() == 1);
        REQUIRE(static_keys.size() > 16);
    }
}

TEST_CASE("ResolveVertexFormat gives the type class the shader is translated for", "[nvn]") {
    using Shader::AttributeType;
    using Vulkan::Nvn::ResolveVertexFormat;
    REQUIRE(ResolveVertexFormat(false, 0).type == AttributeType::Disabled);
    REQUIRE(ResolveVertexFormat(false, 0x25).type == AttributeType::Float);
    REQUIRE(ResolveVertexFormat(false, 0x27).type == AttributeType::UnsignedInt);
    REQUIRE(ResolveVertexFormat(false, 0x28).type == AttributeType::SignedInt);
    const auto invalid = ResolveVertexFormat(false, 0x3C);
    REQUIRE(invalid.type == AttributeType::Disabled);
    REQUIRE(invalid.format == VK_FORMAT_UNDEFINED);

    // R8_UI2F: a scaled attribute; native on a device that does not emulate scaled formats.
    const auto native = ResolveVertexFormat(false, 0x51);
    REQUIRE(native.type == AttributeType::UnsignedScaled);
    REQUIRE(native.format == VK_FORMAT_R8_USCALED);
    const auto emulated = ResolveVertexFormat(true, 0x51);
    REQUIRE(emulated.type == AttributeType::UnsignedScaled);
    REQUIRE(emulated.format == VK_FORMAT_R8_UINT);
    // R32_UI2F has no scaled Vulkan format: raw integer fed to an integer input.
    const auto raw = ResolveVertexFormat(false, 0x55);
    REQUIRE(raw.type == AttributeType::UnsignedInt);
    REQUIRE(raw.scaled_without_support);
}

TEST_CASE("Disk cache round trips graphics records of both vertex input modes", "[nvn]") {
    using Vulkan::Nvn::VertexInputMode;
    const Disk::Header header = MakeHeader();
    const Disk::ShaderRecord vertex = MakeShader(0, 0x1111, 40);
    const Disk::ShaderRecord fragment = MakeShader(1, 0x2222, 200);

    for (const VertexInputMode mode : {VertexInputMode::Static, VertexInputMode::Dynamic}) {
        Disk::GraphicsPipelineRecord record = MakeGraphics(vertex.id, fragment.id);
        record.state = KeyOf(MakeMeaningfulState(3), mode);
        std::vector<u8> file = Disk::EncodeHeader(header);
        for (const auto& bytes : {Disk::EncodeShader(vertex), Disk::EncodeShader(fragment),
                                  Disk::EncodeGraphicsPipeline(record)}) {
            file.insert(file.end(), bytes.begin(), bytes.end());
        }

        Disk::Contents contents;
        const Disk::LoadResult result = Disk::Parse(file, header, contents);
        REQUIRE(result.status == Disk::LoadStatus::Loaded);
        REQUIRE_FALSE(result.HasBadTail());
        REQUIRE(contents.graphics.size() == 1);
        GraphicsFixedState loaded = contents.graphics[0].state;
        REQUIRE(Bytes(loaded) == Bytes(record.state));
        // The loader folds every stored record again: a key written by this mode is unchanged.
        Disk::CanonicalizeState(loaded, mode);
        REQUIRE(Bytes(loaded) == Bytes(record.state));
    }

    SECTION("the format version covers the new state layout") {
        REQUIRE(Disk::FORMAT_VERSION >= 4);
        REQUIRE(sizeof(GraphicsFixedState) % 8 == 0);
    }
}

TEST_CASE("PipelineBase::QueriesMatch revalidates texture types", "[nvn]") {
    PipelineBase pipeline;
    TextureQuery query;
    query.stage = 1;
    query.record = {2, 0x168};
    query.info = {Shader::TextureType::ColorArray2D, Shader::TexturePixelFormat::A8B8G8R8_UNORM,
                  false};
    pipeline.queries = {query};

    FakeOracle oracle;
    oracle.last_stage = 1;
    oracle.last_bank = 2;
    oracle.last_offset = 0x168;
    oracle.answer = query.info;

    SECTION("same answers match") {
        REQUIRE(pipeline.QueriesMatch(oracle));
        REQUIRE(oracle.queries == 1);
    }
    SECTION("a changed type does not") {
        oracle.answer.type = Shader::TextureType::Color2D;
        REQUIRE_FALSE(pipeline.QueriesMatch(oracle));
    }
    SECTION("a changed format does not") {
        oracle.answer.format = Shader::TexturePixelFormat::R16G16B16A16_FLOAT;
        REQUIRE_FALSE(pipeline.QueriesMatch(oracle));
    }
    SECTION("a changed integer flag does not") {
        oracle.answer.is_integer = true;
        REQUIRE_FALSE(pipeline.QueriesMatch(oracle));
    }
    SECTION("a different handle record is answered with the default") {
        oracle.last_offset = 0x170;
        REQUIRE_FALSE(pipeline.QueriesMatch(oracle)); // default is Color2D, the query wants an array
    }
    SECTION("no queries always match") {
        PipelineBase empty;
        REQUIRE(empty.QueriesMatch(oracle));
        REQUIRE(oracle.queries == 0);
    }
}

// --- GPU ------------------------------------------------------------------------------------

namespace {

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
};

/// nullptr when the machine has no usable Vulkan device.
std::unique_ptr<HeadlessVulkan> MakeHeadlessVulkan() {
    try {
        auto result = std::make_unique<HeadlessVulkan>();
        result->library = Vulkan::OpenLibrary(nullptr);
        if (!result->library) {
            return nullptr;
        }
        result->dld = std::make_unique<Vulkan::vk::InstanceDispatch>();
        result->instance = Vulkan::CreateInstance(*result->library, *result->dld,
                                                  VK_API_VERSION_1_1,
                                                  Core::Frontend::WindowSystemType::Headless, false);
        const std::vector<VkPhysicalDevice> devices = result->instance.EnumeratePhysicalDevices();
        if (devices.empty()) {
            return nullptr;
        }
        result->device = std::make_unique<Vulkan::Device>(
            *result->instance, Vulkan::vk::PhysicalDevice(devices[0], *result->dld),
            VK_NULL_HANDLE, *result->dld);
        return result;
    } catch (const Vulkan::vk::Exception&) {
        return nullptr;
    } catch (const std::exception&) {
        return nullptr;
    }
}

} // Anonymous namespace

TEST_CASE("PipelineCache::GetRenderPass caches by key", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    PipelineCacheOptions options;
    options.revision = "test";
    PipelineCache cache{*vulkan->device, options};

    RenderPassKey a;
    a.color[0] = VK_FORMAT_R8G8B8A8_UNORM;
    a.depth = VK_FORMAT_D32_SFLOAT;
    RenderPassKey same = a;
    RenderPassKey other_format = a;
    other_format.color[0] = VK_FORMAT_B8G8R8A8_UNORM;
    RenderPassKey other_slot = a;
    other_slot.color[0] = VK_FORMAT_UNDEFINED;
    other_slot.color[2] = VK_FORMAT_R8G8B8A8_UNORM;
    RenderPassKey no_depth = a;
    no_depth.depth = VK_FORMAT_UNDEFINED;
    RenderPassKey samples = a;
    samples.samples = 4;

    const VkRenderPass pass = cache.GetRenderPass(a);
    REQUIRE(pass != VK_NULL_HANDLE);
    REQUIRE(cache.GetRenderPass(same) == pass);
    const std::array others{cache.GetRenderPass(other_format), cache.GetRenderPass(other_slot),
                            cache.GetRenderPass(no_depth), cache.GetRenderPass(samples)};
    for (size_t i = 0; i < others.size(); ++i) {
        REQUIRE(others[i] != VK_NULL_HANDLE);
        REQUIRE(others[i] != pass);
        for (size_t j = i + 1; j < others.size(); ++j) {
            REQUIRE(others[i] != others[j]);
        }
    }
    REQUIRE(cache.GetRenderPass(a) == pass);

    SECTION("a pass with no attachments is valid") {
        REQUIRE(cache.GetRenderPass(RenderPassKey{}) != VK_NULL_HANDLE);
    }
}

TEST_CASE("Render pass dependencies: the sync diet keeps only the self-dependency", "[nvn]") {
    const auto legacy = Vulkan::Nvn::RenderPassDependencies(false);
    const auto diet = Vulkan::Nvn::RenderPassDependencies(true);
    // Legacy: external -> 0, 0 -> external (both ALL_COMMANDS with every access), then the
    // by-region self-dependency that lets a pass hold a texture/attachment feedback barrier.
    REQUIRE(legacy.size() == 3);
    REQUIRE(legacy[0].srcSubpass == VK_SUBPASS_EXTERNAL);
    REQUIRE(legacy[0].dstSubpass == 0);
    REQUIRE(legacy[0].srcStageMask == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    REQUIRE(legacy[0].dstStageMask == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    REQUIRE(legacy[1].srcSubpass == 0);
    REQUIRE(legacy[1].dstSubpass == VK_SUBPASS_EXTERNAL);
    REQUIRE(legacy[1].srcStageMask == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    REQUIRE(legacy[1].dstStageMask == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    REQUIRE(legacy[2].srcSubpass == 0);
    REQUIRE(legacy[2].dstSubpass == 0);
    // The diet drops the two external ones and nothing else: no external dependency may remain
    // (an explicit one would bring the full sync back), and the self-dependency is unchanged.
    REQUIRE(diet.size() == 1);
    REQUIRE(diet[0].srcSubpass == 0);
    REQUIRE(diet[0].dstSubpass == 0);
    REQUIRE(diet[0].srcStageMask == legacy[2].srcStageMask);
    REQUIRE(diet[0].dstStageMask == legacy[2].dstStageMask);
    REQUIRE(diet[0].srcAccessMask == legacy[2].srcAccessMask);
    REQUIRE(diet[0].dstAccessMask == legacy[2].dstAccessMask);
    REQUIRE(diet[0].dependencyFlags == VK_DEPENDENCY_BY_REGION_BIT);
}

TEST_CASE("Translation epoch is a SHA-256 of the translation sources", "[nvn]") {
    const std::string_view epoch = Disk::TranslationEpoch();
    REQUIRE(epoch.size() == 64);
    REQUIRE(std::ranges::all_of(epoch, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
    REQUIRE(epoch != std::string(64, '0'));
}

TEST_CASE("Translation key changes if and only if an input of the cached output changes",
          "[nvn]") {
    const Disk::TranslationInputs base{
        .epoch = "epoch-a",
        .switches_hash = 0x1111,
        .vendor_id = 0x10DE,
        .device_id = 0x2B85,
        .driver_version = 0x1234,
        .cache_uuid_hash = 0xABCDEF,
        .profile_hash = 0x555,
    };
    const std::string key = Disk::MakeTranslationKey(base);
    REQUIRE(key == Disk::MakeTranslationKey(base));
    // The manual format bump is part of the key: a bump invalidates every file.
    REQUIRE(key.find("nvn-pipeline-cache/" + std::to_string(Disk::FORMAT_VERSION) + " ") == 0);

    const auto changes = [&](auto edit) {
        Disk::TranslationInputs other = base;
        edit(other);
        return Disk::MakeTranslationKey(other) != key;
    };
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.epoch = "epoch-b"; }));
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.switches_hash ^= 1; }));
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.vendor_id ^= 1; }));
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.device_id ^= 1; }));
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.driver_version ^= 1; }));
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.cache_uuid_hash ^= 1; }));
    REQUIRE(changes([](Disk::TranslationInputs& in) { in.profile_hash ^= 1; }));
    // The same inputs spelled again (another string with the same content) are the same key.
    const std::string epoch_copy = "epoch-a";
    Disk::TranslationInputs same = base;
    same.epoch = epoch_copy;
    REQUIRE(Disk::MakeTranslationKey(same) == key);
    // The key fits the header field and names the file.
    REQUIRE(key.size() < 256);
    REQUIRE(Disk::FileNameForKey(key) == Disk::FileNameForKey(Disk::MakeTranslationKey(same)));
    const std::string name = Disk::FileNameForKey(key);
    REQUIRE(name.starts_with("nvn_pipelines.k"));
    REQUIRE(name.ends_with(".bin"));
    REQUIRE(name.size() == std::string_view{"nvn_pipelines.k.bin"}.size() + 16);
    Disk::TranslationInputs other = base;
    other.epoch = "epoch-b";
    REQUIRE(Disk::FileNameForKey(Disk::MakeTranslationKey(other)) != name);
}

TEST_CASE("Translation switches: the hash covers what the recompiler reads and nothing else",
          "[nvn]") {
    using Vulkan::Nvn::HashTranslationSwitches;
    using Vulkan::Nvn::TranslationSwitches;
    const TranslationSwitches base;
    const u64 hash = HashTranslationSwitches(base);
    REQUIRE(hash == HashTranslationSwitches(TranslationSwitches{}));
    const auto changes = [&](auto edit) {
        TranslationSwitches other = base;
        edit(other);
        return HashTranslationSwitches(other) != hash;
    };
    REQUIRE(changes([](TranslationSwitches& s) { s.loop_safety_checks_disabled = true; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.fp32_denorm_preserve = true; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.nan_outputs = 'm'; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.undef_is_zero = false; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.tensor_mode_set = true; }));
    REQUIRE(changes([](TranslationSwitches& s) {
        s.tensor_mode_set = true;
        s.tensor_mode = "stub";
    }));
    REQUIRE(changes([](TranslationSwitches& s) { s.tensor_lowering = 1; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.tensor_lowering = 2; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.raw_coverage = true; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.raw_global_fast = true; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.lean_global_memory = !s.lean_global_memory; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.ro_mirror_exact = true; }));
    REQUIRE(changes([](TranslationSwitches& s) { s.strong_global_fence = true; }));
    REQUIRE_FALSE(changes([](TranslationSwitches&) {}));
}

TEST_CASE("Tensor programs are lowered for the titles the emulated GPU lowers them for",
          "[nvn]") {
    using VideoCommon::NativeTensorLowering;
    using VideoCommon::NativeTensorLoweringFor;
    // Ocarina of Time: compact where compute shaders can be 32 threads wide, else the expansion.
    CHECK(NativeTensorLoweringFor(VideoCommon::OcarinaOfTimeProgramId, true) ==
          NativeTensorLowering::Compact);
    CHECK(NativeTensorLoweringFor(VideoCommon::OcarinaOfTimeProgramId, false) ==
          NativeTensorLowering::Full);
    // Donkey Kong Bananza: the expansion.
    CHECK(NativeTensorLoweringFor(VideoCommon::DonkeyKongBananzaProgramId, true) ==
          NativeTensorLowering::Full);
    // Everyone else: rejected, as before.
    CHECK(NativeTensorLoweringFor(0x0100000000010000ULL, true) == NativeTensorLowering::None);
    CHECK(NativeTensorLoweringFor(0, false) == NativeTensorLowering::None);
}

TEST_CASE("Translation switches follow the settings the recompiler reads", "[nvn]") {
    using Vulkan::Nvn::CurrentTranslationSwitches;
    using Vulkan::Nvn::HashTranslationSwitches;
    const bool loop_safety = Settings::values.disable_shader_loop_safety_checks.GetValue();
    const u64 before = HashTranslationSwitches(CurrentTranslationSwitches());

    Settings::values.disable_shader_loop_safety_checks.SetValue(!loop_safety);
    const u64 loop_changed = HashTranslationSwitches(CurrentTranslationSwitches());
    Settings::values.disable_shader_loop_safety_checks.SetValue(loop_safety);
    const u64 after = HashTranslationSwitches(CurrentTranslationSwitches());

    REQUIRE(loop_changed != before);
    REQUIRE(after == before);
}

TEST_CASE("The translation key ignores presentation and resolution settings", "[nvn]") {
    using Vulkan::Nvn::CurrentTranslationSwitches;
    using Vulkan::Nvn::HashTranslationSwitches;
    const auto anti_aliasing = Settings::values.anti_aliasing.GetValue();
    const auto filter = Settings::values.scaling_filter.GetValue();
    const auto resolution = Settings::values.resolution_setup.GetValue();
    const auto scaling = Settings::values.resolution_info;
    const bool loop_safety = Settings::values.disable_shader_loop_safety_checks.GetValue();
    // Restored even when an assertion ends the test early.
    SCOPE_EXIT {
        Settings::values.anti_aliasing.SetValue(anti_aliasing);
        Settings::values.scaling_filter.SetValue(filter);
        Settings::values.resolution_setup.SetValue(resolution);
        Settings::values.resolution_info = scaling;
        Settings::values.disable_shader_loop_safety_checks.SetValue(loop_safety);
    };
    const auto key_now = [] {
        return Disk::MakeTranslationKey({
            .epoch = "e",
            .switches_hash = HashTranslationSwitches(CurrentTranslationSwitches()),
            .vendor_id = 1,
            .device_id = 2,
            .driver_version = 3,
            .cache_uuid_hash = 4,
            .profile_hash = 5,
        });
    };
    const std::string key = key_now();

    Settings::values.anti_aliasing.SetValue(Settings::AntiAliasing::Smaa);
    REQUIRE(key_now() == key);
    Settings::values.scaling_filter.SetValue(Settings::ScalingFilter::Fsr);
    REQUIRE(key_now() == key);
    // The resolution setting changes it only while it scales (the rescaling pass bakes the factors in).
    const std::string scaled_key = [&] {
        Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res2X,
                                          Settings::values.resolution_info);
        REQUIRE(Settings::values.resolution_info.active);
        return key_now();
    }();
    REQUIRE(scaled_key != key);
    Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res3X,
                                      Settings::values.resolution_info);
    REQUIRE(key_now() != scaled_key);
    Settings::values.resolution_info = scaling;
    REQUIRE(key_now() == key);

    Settings::values.anti_aliasing.SetValue(anti_aliasing);
    Settings::values.scaling_filter.SetValue(filter);
    Settings::values.resolution_setup.SetValue(resolution);
    Settings::values.resolution_info = scaling;
    REQUIRE(key_now() == key);

    // A switch the recompiler reads changes the file.
    Settings::values.disable_shader_loop_safety_checks.SetValue(!loop_safety);
    const std::string loop_key = key_now();
    Settings::values.disable_shader_loop_safety_checks.SetValue(loop_safety);
    REQUIRE(loop_key != key);
    REQUIRE(key_now() == key);
}

TEST_CASE("The NVN host info runs the rescaling pass only while the setting scales", "[nvn]") {
    // The replay pushes the rescaling words (DrawContext::PushPlan): at 1X the pass is skipped,
    // otherwise it runs. MakeHostInfo needs a device, the reference info does not.
    const auto saved = Settings::values.resolution_info;
    SCOPE_EXIT {
        Settings::values.resolution_info = saved;
    };
    Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res1X,
                                      Settings::values.resolution_info);
    REQUIRE(VideoCommon::Nvn::MakeReferenceHostInfo().skip_rescaling_pass);
    Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res2X,
                                      Settings::values.resolution_info);
    REQUIRE_FALSE(VideoCommon::Nvn::MakeReferenceHostInfo().skip_rescaling_pass);
    Settings::values.resolution_info = saved;
}

TEST_CASE("Runtime TRY switch changes the translation cache key", "[nvn]") {
    namespace D = Common::DiagRequest;
    using Vulkan::Nvn::CurrentTranslationSwitches;
    using Vulkan::Nvn::HashTranslationSwitches;
    const bool initial = D::TrySwitchEnabled("TRY_RAWGLOBAL_FAST");
    SCOPE_EXIT {
        (void)D::SetTrySwitch("TRY_RAWGLOBAL_FAST", initial);
    };
    const u64 base = HashTranslationSwitches(CurrentTranslationSwitches());
    REQUIRE(D::SetTrySwitch("TRY_RAWGLOBAL_FAST", !initial));
    const u64 changed = HashTranslationSwitches(CurrentTranslationSwitches());
    REQUIRE(changed != base);
    REQUIRE(D::SetTrySwitch("TRY_RAWGLOBAL_FAST", initial));
    REQUIRE(HashTranslationSwitches(CurrentTranslationSwitches()) == base);
}

TEST_CASE("Runtime TRY_RO_MIRROR_EXACT changes the translation cache key", "[nvn]") {
    namespace D = Common::DiagRequest;
    using Vulkan::Nvn::CurrentTranslationSwitches;
    using Vulkan::Nvn::HashTranslationSwitches;
    const bool initial = D::TrySwitchEnabled("TRY_RO_MIRROR_EXACT");
    const u64 base = HashTranslationSwitches(CurrentTranslationSwitches());
    const u64 base_try = Vulkan::Nvn::TrySwitchesHash();
    REQUIRE(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", !initial));
    CHECK(CurrentTranslationSwitches().ro_mirror_exact == !initial);
    CHECK(HashTranslationSwitches(CurrentTranslationSwitches()) != base);
    CHECK(Vulkan::Nvn::TrySwitchesHash() != base_try);
    REQUIRE(D::SetTrySwitch("TRY_RO_MIRROR_EXACT", initial));
    REQUIRE(HashTranslationSwitches(CurrentTranslationSwitches()) == base);
    REQUIRE(Vulkan::Nvn::TrySwitchesHash() == base_try);
}

TEST_CASE("Disk cache files are pruned oldest first, earlier naming schemes included", "[nvn]") {
    namespace fs = std::filesystem;
    TempDirectory directory;
    const auto now = fs::file_time_type::clock::now();
    std::vector<fs::path> files;
    for (int i = 0; i < 5; ++i) {
        files.push_back(directory.path / Disk::FileNameForKey("key" + std::to_string(i)));
        WriteBytes(files.back(), std::vector<u8>(8));
        fs::last_write_time(files.back(), now - std::chrono::hours(10 - i));
    }
    const auto vkcache = directory.path / "nvn_pipelines.vkcache";
    WriteBytes(vkcache, std::vector<u8>(8));
    // The files earlier builds wrote (one shared file, then one file per git revision) are
    // never read again (format 14 and older are rejected) and age out like any other.
    const auto legacy = directory.path / "nvn_pipelines.bin";
    WriteBytes(legacy, std::vector<u8>(8));
    fs::last_write_time(legacy, now - std::chrono::hours(100));
    const auto per_revision = directory.path / "nvn_pipelines.2b7551f137ab.bin";
    WriteBytes(per_revision, std::vector<u8>(8));
    fs::last_write_time(per_revision, now - std::chrono::hours(50));

    // Room for the file in use plus two others: the two newest others stay, the file in use
    // is the oldest and is kept (and refreshed) anyway.
    Disk::PruneCacheFiles(directory.path, files[0], 3);
    REQUIRE(fs::exists(files[0]));
    REQUIRE_FALSE(fs::exists(files[1]));
    REQUIRE_FALSE(fs::exists(files[2]));
    REQUIRE_FALSE(fs::exists(legacy));
    REQUIRE_FALSE(fs::exists(per_revision));
    REQUIRE(fs::exists(files[3]));
    REQUIRE(fs::exists(files[4]));
    REQUIRE(fs::exists(vkcache));
    REQUIRE(fs::last_write_time(files[0]) > now - std::chrono::hours(1));

    Disk::PruneCacheFiles(directory.path, files[0], 6);
    REQUIRE(fs::exists(files[3]));
    REQUIRE(fs::exists(files[4]));
}

namespace {

/// The cache files (`nvn_pipelines.*.bin`) in `directory`, sorted by name.
std::vector<std::filesystem::path> CacheFiles(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (name.starts_with("nvn_pipelines.") && name.ends_with(".bin")) {
            files.push_back(entry.path());
        }
    }
    std::ranges::sort(files);
    return files;
}

/// Opens and closes a cache with `options` and returns the cache file it uses (the options'
/// directory must hold no other cache file yet).
std::filesystem::path CreateCacheFile(const Vulkan::Device& device,
                                      const PipelineCacheOptions& options) {
    {
        PipelineCache cache{device, options};
        cache.Flush();
    }
    const auto files = CacheFiles(options.directory);
    REQUIRE(files.size() == 1);
    return files[0];
}

} // Anonymous namespace

TEST_CASE("PipelineCache disk cache lifecycle", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    TempDirectory directory;

    PipelineCacheOptions options;
    options.directory = directory.path;
    options.revision = "revision-a";
    options.translation_epoch = "epoch-a";

    SECTION("a fresh directory gets a header and nothing is rejected") {
        {
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
            REQUIRE(cache.Stats().disk_rejected == 0);
            REQUIRE(cache.Stats().loaded == 0);
        }
        const auto files = CacheFiles(directory.path);
        REQUIRE(files.size() == 1);
        REQUIRE(std::filesystem::file_size(files[0]) > 0);
        PipelineCache again{*vulkan->device, options};
        again.Flush();
        REQUIRE(again.Stats().disk_rejected == 0);
    }
    SECTION("a rebuild with the same translation inputs reuses the file") {
        const auto file = CreateCacheFile(*vulkan->device, options);
        const auto size = std::filesystem::file_size(file);
        options.revision = "revision-b";
        {
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
            REQUIRE(cache.Stats().disk_rejected == 0);
        }
        REQUIRE(CacheFiles(directory.path) == std::vector{file});
        REQUIRE(std::filesystem::file_size(file) == size);
    }
    SECTION("other translation inputs get their own file and leave this one intact") {
        const auto file = CreateCacheFile(*vulkan->device, options);
        const auto size_a = std::filesystem::file_size(file);
        options.translation_epoch = "epoch-b";
        {
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
            REQUIRE(cache.Stats().disk_rejected == 0);
        }
        REQUIRE(CacheFiles(directory.path).size() == 2);
        REQUIRE(std::filesystem::file_size(file) == size_a);
        options.translation_epoch = "epoch-a";
        PipelineCache again{*vulkan->device, options};
        again.Flush();
        REQUIRE(again.Stats().disk_rejected == 0);
        REQUIRE(CacheFiles(directory.path).size() == 2);
    }
    SECTION("a file whose header holds another key is discarded, never served") {
        // The file name only tells keys apart: put the file of epoch-a under the name epoch-b
        // looks for, and the header decides.
        const auto file_a = CreateCacheFile(*vulkan->device, options);
        PipelineCacheOptions other = options;
        other.translation_epoch = "epoch-b";
        TempDirectory scratch;
        other.directory = scratch.path;
        const auto name_b = CreateCacheFile(*vulkan->device, other).filename();
        std::filesystem::copy_file(file_a, scratch.path / name_b,
                                   std::filesystem::copy_options::overwrite_existing);
        {
            PipelineCache cache{*vulkan->device, other};
            cache.Flush();
            REQUIRE(cache.Stats().disk_rejected == 1);
            REQUIRE(cache.Stats().loaded == 0);
        }
        PipelineCache again{*vulkan->device, other};
        again.Flush();
        REQUIRE(again.Stats().disk_rejected == 0);
    }
    SECTION("a garbage file is rejected without a crash and replaced") {
        const auto file = CreateCacheFile(*vulkan->device, options);
        std::vector<u8> garbage(777);
        std::mt19937 random(11);
        for (u8& byte : garbage) {
            byte = static_cast<u8>(random());
        }
        WriteBytes(file, garbage);
        {
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
            REQUIRE(cache.Stats().disk_rejected == 1);
        }
        PipelineCache again{*vulkan->device, options};
        again.Flush();
        REQUIRE(again.Stats().disk_rejected == 0);
    }
    SECTION("a truncated file is never fatal") {
        const auto file = CreateCacheFile(*vulkan->device, options);
        const auto size = std::filesystem::file_size(file);
        for (const auto cut : {size_t{0}, size_t{1}, size_t{9}, static_cast<size_t>(size) - 1}) {
            std::filesystem::resize_file(file, std::min<size_t>(cut, size));
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
        }
        PipelineCache final_cache{*vulkan->device, options};
        final_cache.Flush();
        REQUIRE(final_cache.Stats().disk_rejected == 0);
    }
    SECTION("the vertex input mode follows the device and is part of the file identity") {
        using Vulkan::Nvn::VertexInputMode;
        options.allow_dynamic_vertex_input = false;
        {
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
            REQUIRE(cache.VertexInput() == VertexInputMode::Static);
            REQUIRE(cache.Stats().disk_rejected == 0);
        }
        options.allow_dynamic_vertex_input = true;
        const bool dynamic_device = vulkan->device->IsExtVertexInputDynamicStateSupported();
        {
            PipelineCache cache{*vulkan->device, options};
            cache.Flush();
            REQUIRE(cache.VertexInput() ==
                    (dynamic_device ? VertexInputMode::Dynamic : VertexInputMode::Static));
            // The mode is part of the key: the other mode has its own file, nothing is rejected.
            REQUIRE(cache.Stats().disk_rejected == 0);
        }
        REQUIRE(CacheFiles(directory.path).size() == (dynamic_device ? 2u : 1u));
        PipelineCache again{*vulkan->device, options};
        again.Flush();
        REQUIRE(again.Stats().disk_rejected == 0);
    }
}

// --- asynchronous compiles --------------------------------------------------------------------

namespace {

using Vulkan::Nvn::ComputeRequest;
using Vulkan::Nvn::GraphicsRequest;
using Vulkan::Nvn::StageProgram;

struct TestProgram {
    std::vector<u8> control;
    std::vector<u8> code;
    u64 hash{};

    [[nodiscard]] StageProgram View() const {
        return {hash, &control, &code};
    }
};

void PutWord(std::vector<u8>& bytes, size_t offset, u32 value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

/// An NVN2 program of `instructions` (4 words each) for NVNshaderStage `stage`.
TestProgram MakeTestProgram(u32 stage, std::span<const u32> instructions) {
    const bool compute = stage == 5;
    const u32 header = compute ? 0 : VC::GraphicsHeaderBytes;
    const u32 inst_bytes = header + static_cast<u32>(instructions.size_bytes());
    const u32 c1_offset = (inst_bytes + 0xff) / 0x100 * 0x100;
    TestProgram program;
    program.control.assign(0x100, 0);
    PutWord(program.control, 0, VC::ControlMagic);
    PutWord(program.control, 4, VC::ControlVersion);
    PutWord(program.control, 0x38, inst_bytes);
    PutWord(program.control, 0x40, c1_offset);
    PutWord(program.control, 0x44, c1_offset);
    program.control[0x4c] = 12;
    program.control[0x56] = static_cast<u8>(stage);
    PutWord(program.control, 0x58, compute ? 1 : 8);
    PutWord(program.control, 0x5c, compute ? 1 : 4);
    PutWord(program.control, 0x60, compute ? 1 : 2);
    if (compute) {
        program.control[0xba] = 0x10; // driver bank bytes, no push constants or buffers
    }
    program.code.assign(c1_offset, 0);
    if (!compute) {
        PutWord(program.code, 0, (stage == 0 ? 1u : 5u) << 10); // SPH type: vertex / fragment
    }
    std::memcpy(program.code.data() + header, instructions.data(), instructions.size_bytes());
    program.hash = HashProgramStage(program.control, program.code);
    return program;
}

constexpr std::array<u32, 4> EXIT_INSTRUCTION{0x0000794d, 0, 0x03800000, 0x000fea00};

/// IADD3 R6, R0, `immediate`, RZ ; EXIT. Different immediates are different programs.
TestProgram MakeAddProgram(u32 stage, u32 immediate) {
    const std::array<u32, 8> instructions{0x00007810 | (6u << 16), immediate, 0x07ffe0ff,
                                          0x000fe200,              EXIT_INSTRUCTION[0],
                                          EXIT_INSTRUCTION[1],     EXIT_INSTRUCTION[2],
                                          EXIT_INSTRUCTION[3]};
    return MakeTestProgram(stage, instructions);
}

/// The captured fragment program of the shader tests: two bindless TEX whose handle words are
/// c2[0x168] / c2[0x170] (the translation asks the oracle about both).
TestProgram MakeTexturedFragment() {
    static constexpr std::array<u32, 52> WORDS{
        0x00067ab9, 0x00815a00, 0x00000800, 0x000fe400, 0x00047ab9, 0x00805a00, 0x00000800,
        0x000fe400, 0x00077ab9, 0x00805c00, 0x00000800, 0x000fe400, 0x06047292, 0x00000004,
        0x0f8efc3f, 0x000fe400, 0x06067292, 0x00000007, 0x0f8efc3f, 0x000fc400, 0x3f057899,
        0x00000014, 0x08011604, 0x000fe400, 0x3f077899, 0x00000014, 0x08011606, 0x000fe200,
        0x00007308, 0x00000000, 0x00001000, 0x001e6400, 0x00047220, 0x00000001, 0x0041c000,
        0x042fe200, 0x00057220, 0x00000005, 0x0041c000, 0x004fca00, 0x04087f60, 0x30000605,
        0x081e03ff, 0x100f4200, 0x04067f60, 0x30000405, 0x081e01ff, 0x000f4200, 0x0000794d,
        0x00000000, 0x03800000, 0x00000000};
    return MakeTestProgram(1, WORDS);
}

ComputeRequest MakeComputeRequest(const TestProgram& program, const TextureOracle* oracle) {
    ComputeRequest request;
    request.program = program.View();
    request.oracle = oracle;
    return request;
}

GraphicsRequest MakeGraphicsRequest(const TestProgram& vertex, const TestProgram& fragment,
                                    const TextureOracle* oracle) {
    GraphicsRequest request;
    request.stages[0] = vertex.View();
    request.stages[1] = fragment.View();
    request.state.topology = 4; // triangles
    request.state.depth_mode = 1;
    request.state.window_origin = 1;
    request.state.sample_mask = 0xFFFFFFFFu;
    request.targets.color[0] = VK_FORMAT_R8G8B8A8_UNORM;
    request.oracle = oracle;
    return request;
}

/// A live oracle that can be frozen. The frozen copy answers what the live one answered at the
/// time and can be held up (`gate`) inside the worker, so a test can change the live answer while
/// a compile is running.
class FreezableOracle final : public TextureOracle {
public:
    struct Gate {
        std::mutex mutex;
        std::condition_variable cv;
        bool open{true};
        std::atomic<int> frozen_queries{};
    };

    [[nodiscard]] TextureInfo Query(u32, const VC::NvnHandleRecord&) const override {
        ++live_queries;
        return answer;
    }
    [[nodiscard]] std::unique_ptr<TextureOracle> Freeze(u32 stage_mask) const override {
        freeze_masks.push_back(stage_mask);
        return std::make_unique<Frozen>(answer, gate);
    }

    void SetGate(bool open) {
        {
            std::scoped_lock lock{gate->mutex};
            gate->open = open;
        }
        gate->cv.notify_all();
    }

    TextureInfo answer{};
    std::shared_ptr<Gate> gate = std::make_shared<Gate>();
    mutable std::vector<u32> freeze_masks;
    mutable int live_queries{};

private:
    class Frozen final : public TextureOracle {
    public:
        Frozen(TextureInfo answer_, std::shared_ptr<Gate> gate_)
            : answer{answer_}, gate{std::move(gate_)} {}
        [[nodiscard]] TextureInfo Query(u32, const VC::NvnHandleRecord&) const override {
            ++gate->frozen_queries;
            std::unique_lock lock{gate->mutex};
            gate->cv.wait_for(lock, std::chrono::seconds(30), [this] { return gate->open; });
            return answer;
        }

    private:
        TextureInfo answer;
        std::shared_ptr<Gate> gate;
    };
};

/// A live oracle that cannot be frozen (the default of TextureOracle).
class UnfreezableOracle final : public TextureOracle {
public:
    [[nodiscard]] TextureInfo Query(u32, const VC::NvnHandleRecord&) const override {
        return {};
    }
};

PipelineCacheOptions AsyncOptions(bool async) {
    PipelineCacheOptions options;
    options.revision = "async-test";
    options.async_compile = async;
    return options;
}

} // Anonymous namespace

TEST_CASE("Compute shader cache separates identical code with different bank layouts",
          "[nvn][shader-refresh]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    PipelineCache cache{*vulkan->device, {}};
    const TestProgram plain = MakeAddProgram(5, 42);
    TestProgram with_uniform = plain;
    with_uniform.control[0xb1] = 1;
    PutWord(with_uniform.control, 0xba, 0x20);
    with_uniform.hash = HashProgramStage(with_uniform.control, with_uniform.code);
    REQUIRE(plain.code == with_uniform.code);
    const auto first = cache.GetCompute(MakeComputeRequest(plain, nullptr));
    const auto second = cache.GetCompute(MakeComputeRequest(with_uniform, nullptr));
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(first != second);
    CHECK(cache.Stats().lowered_programs == 2);
    CHECK(cache.GetCompute(MakeComputeRequest(plain, nullptr)) == first);
}

TEST_CASE("Shader refresh drops pipelines while retaining existing owners and render passes",
          "[nvn][shader-refresh]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    bool async = false;
    SECTION("synchronous compiles") {}
    SECTION("queued compiles are drained before refresh") {
        async = true;
    }
    VideoCore::ShaderNotify notify;
    PipelineCacheOptions options = AsyncOptions(async);
    options.shader_notify = &notify;
    PipelineCache cache{*vulkan->device, options};
    const TestProgram vertex = MakeAddProgram(0, 23);
    const TestProgram fragment = MakeTexturedFragment();
    const TestProgram program = MakeAddProgram(5, 42);
    FreezableOracle oracle;
    const auto graphics_request = MakeGraphicsRequest(vertex, fragment, &oracle);
    const auto compute_request = MakeComputeRequest(program, &oracle);
    const auto graphics = cache.GetGraphics(graphics_request);
    const auto compute = cache.GetCompute(compute_request);
    REQUIRE(graphics != nullptr);
    REQUIRE(compute != nullptr);
    const VkPipeline graphics_handle = graphics->Handle();
    const VkPipeline compute_handle = *compute->pipeline;
    const VkRenderPass pass = cache.GetRenderPass(graphics_request.targets);
    for (u64 generation = 1; generation <= 3; ++generation) {
        if (async) {
            const auto extra = MakeAddProgram(5, 42 + static_cast<u32>(generation));
            bool not_ready = false;
            static_cast<void>(cache.GetCompute(MakeComputeRequest(extra, &oracle), &not_ready));
        }
        cache.ClearShaders();
        CHECK(cache.ShaderGeneration() == generation);
        CHECK(cache.Stats().graphics_pipelines == 0);
        CHECK(cache.Stats().compute_pipelines == 0);
        CHECK(cache.Stats().lowered_programs == 0);
        CHECK(notify.NumBuildingNoReport() == 0);
        CHECK(graphics->Handle() == graphics_handle);
        CHECK(*compute->pipeline == compute_handle);
        CHECK(cache.GetRenderPass(graphics_request.targets) == pass);
        const auto fresh_graphics = cache.GetGraphics(graphics_request);
        const auto fresh_compute = cache.GetCompute(compute_request);
        REQUIRE(fresh_graphics != nullptr);
        REQUIRE(fresh_compute != nullptr);
        CHECK(fresh_graphics != graphics);
        CHECK(fresh_compute != compute);
        CHECK(fresh_graphics->Handle() != graphics_handle);
        CHECK(*fresh_compute->pipeline != compute_handle);
    }
}

TEST_CASE("TextureOracle cannot be frozen unless it says so", "[nvn]") {
    const UnfreezableOracle oracle;
    REQUIRE(oracle.Freeze(0xFF) == nullptr);
    FreezableOracle freezable;
    REQUIRE(freezable.Freeze(0x3) != nullptr);
    REQUIRE(freezable.freeze_masks == std::vector<u32>{0x3});
}

TEST_CASE("Async compile: a miss is not ready, then a hit; the worker never sees the live oracle",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    PipelineCache cache{*vulkan->device, AsyncOptions(true)};
    const TestProgram program = MakeAddProgram(5, 42);
    FreezableOracle oracle;
    const ComputeRequest request = MakeComputeRequest(program, &oracle);

    bool not_ready = false;
    REQUIRE(cache.GetCompute(request, &not_ready) == nullptr);
    REQUIRE(not_ready);
    REQUIRE(oracle.freeze_masks == std::vector<u32>{1u << 5});
    // Asking again while it compiles is not ready as well and does not queue a second compile.
    not_ready = false;
    const auto while_compiling = cache.GetCompute(request, &not_ready);
    REQUIRE((while_compiling != nullptr) != not_ready);

    cache.Flush();
    not_ready = true;
    const auto pipeline = cache.GetCompute(request, &not_ready);
    REQUIRE(pipeline != nullptr);
    REQUIRE_FALSE(not_ready);
    REQUIRE(cache.GetCompute(request, &not_ready) == pipeline);
    // Without the out parameter the lookup is the synchronous API and finds the same pipeline.
    REQUIRE(cache.GetCompute(request) == pipeline);

    const auto stats = cache.Stats();
    REQUIRE(stats.async_queued == 1);
    REQUIRE(stats.compiled_at_runtime == 1);
    REQUIRE(stats.async_failed == 0);
    REQUIRE(stats.async_not_ready >= 1);
    REQUIRE(oracle.live_queries == 0); // only the frozen copy was asked
}

TEST_CASE("Async compile: many programs compile concurrently and each is delivered once",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    PipelineCache cache{*vulkan->device, AsyncOptions(true)};
    constexpr u32 COUNT = 16;
    std::vector<TestProgram> programs;
    for (u32 i = 0; i < COUNT; ++i) {
        programs.push_back(MakeAddProgram(5, 100 + i));
    }
    FreezableOracle oracle;
    for (int round = 0; round < 3; ++round) {
        for (const TestProgram& program : programs) {
            bool not_ready = false;
            [[maybe_unused]] const auto pipeline =
                cache.GetCompute(MakeComputeRequest(program, &oracle), &not_ready);
        }
    }
    cache.Flush();
    std::set<const void*> distinct;
    for (const TestProgram& program : programs) {
        bool not_ready = false;
        const auto pipeline = cache.GetCompute(MakeComputeRequest(program, &oracle), &not_ready);
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(not_ready);
        distinct.insert(pipeline.get());
    }
    REQUIRE(distinct.size() == COUNT);
    const auto stats = cache.Stats();
    REQUIRE(stats.async_queued == COUNT); // three rounds of requests, one compile per key
    REQUIRE(stats.compiled_at_runtime == COUNT);
    REQUIRE(stats.async_failed == 0);
}

TEST_CASE("Async compile: graphics answers at request time and keeps variants", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    PipelineCache cache{*vulkan->device, AsyncOptions(true)};
    const TestProgram vertex = MakeAddProgram(0, 7);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    oracle.answer.type = Shader::TextureType::Color2D;
    const GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);

    // The worker is held inside its first texture query while the live answer changes.
    oracle.SetGate(false);
    bool not_ready = false;
    REQUIRE(cache.GetGraphics(request, &not_ready) == nullptr);
    REQUIRE(not_ready);
    REQUIRE(oracle.freeze_masks == std::vector<u32>{0b00011});
    oracle.answer.type = Shader::TextureType::ColorArray2D;
    oracle.SetGate(true);
    cache.Flush();
    REQUIRE(oracle.gate->frozen_queries.load() > 0);
    REQUIRE(oracle.live_queries == 0);

    // The pipeline holds the answers of the request, not the later ones: it is not valid for the
    // draw that asks now, which compiles its own variant.
    not_ready = false;
    REQUIRE(cache.GetGraphics(request, &not_ready) == nullptr);
    REQUIRE(not_ready);
    cache.Flush();
    const auto array_variant = cache.GetGraphics(request, &not_ready);
    REQUIRE(array_variant != nullptr);
    REQUIRE_FALSE(not_ready);
    REQUIRE_FALSE(array_variant->queries.empty());
    for (const TextureQuery& query : array_variant->queries) {
        REQUIRE(query.info.type == Shader::TextureType::ColorArray2D);
    }
    oracle.answer.type = Shader::TextureType::Color2D;
    const auto plain_variant = cache.GetGraphics(request, &not_ready);
    REQUIRE(plain_variant != nullptr);
    REQUIRE_FALSE(not_ready);
    REQUIRE(plain_variant != array_variant);
    for (const TextureQuery& query : plain_variant->queries) {
        REQUIRE(query.info.type == Shader::TextureType::Color2D);
    }
    const auto stats = cache.Stats();
    REQUIRE(stats.async_queued == 2);
    REQUIRE(stats.variants_retranslated == 1);
    REQUIRE(stats.async_failed == 0);
}

TEST_CASE("Async compile: the shader notifier counts queued and running compiles", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    VideoCore::ShaderNotify notify;
    PipelineCacheOptions options = AsyncOptions(true);
    options.shader_notify = &notify;
    PipelineCache cache{*vulkan->device, std::move(options)};
    const TestProgram vertex = MakeAddProgram(0, 7);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    oracle.answer.type = Shader::TextureType::Color2D;
    const GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);
    REQUIRE(notify.NumBuildingNoReport() == 0);

    // The worker is held inside its texture query: the compile is running and counted.
    oracle.SetGate(false);
    bool not_ready = false;
    REQUIRE(cache.GetGraphics(request, &not_ready) == nullptr);
    REQUIRE(not_ready);
    REQUIRE(notify.NumBuildingNoReport() == 1);
    // Asking again while it compiles queues nothing more.
    not_ready = false;
    [[maybe_unused]] const auto again = cache.GetGraphics(request, &not_ready);
    REQUIRE(notify.NumBuildingNoReport() == 1);

    oracle.SetGate(true);
    cache.Flush();
    REQUIRE(notify.NumBuildingNoReport() == 0);
    REQUIRE(cache.Stats().async_queued == 1);
    // The status bar's reading: one build since the window opened.
    REQUIRE(notify.ShadersBuilding() == 1);

    // A hit queues nothing.
    not_ready = false;
    REQUIRE(cache.GetGraphics(request, &not_ready) != nullptr);
    REQUIRE(notify.NumBuildingNoReport() == 0);

    // Compute compiles are counted too.
    const TestProgram compute = MakeAddProgram(5, 42);
    FreezableOracle compute_oracle;
    not_ready = false;
    REQUIRE(cache.GetCompute(MakeComputeRequest(compute, &compute_oracle), &not_ready) == nullptr);
    cache.Flush();
    REQUIRE(notify.NumBuildingNoReport() == 0);
    REQUIRE(cache.Stats().async_queued == 2);
}

TEST_CASE("VTX-HLE clears the title shader caches once after emulated GPU use",
          "[nvn][vtxhle][shader-cache-transition]") {
    using namespace VideoCommon;
    TempDirectory directory;
    const auto title = directory.path / "0400c3f00006e000";
    const auto other = directory.path / "other-title";
    const auto legacy = title / MKW_EMULATED_CACHE_SUBDIR;
    std::filesystem::create_directories(legacy);
    std::filesystem::create_directories(other);
    const auto portable = title / "nvn_pipelines.k1234.bin";
    const auto driver = title / NVN_DRIVER_CACHE_FILE_NAME;
    const auto unrelated = title / "notes.txt";
    std::ofstream(portable) << "nvn";
    std::ofstream(driver) << "driver";
    std::ofstream(unrelated) << "notes";
    std::ofstream(other / "vulkan.bin") << "other";
    CHECK(ClearShaderCachesForNvn(title, true) == NvnCacheTransition::Unchanged);
    CHECK(std::filesystem::exists(portable));
    CHECK(std::filesystem::exists(driver));

    for (auto name : EMULATED_VULKAN_CACHE_FILES) {
        std::ofstream(legacy / name) << "legacy";
    }
    std::ofstream(title / "opengl.bin") << "gl";
    CHECK(ClearShaderCachesForNvn(title, true) == NvnCacheTransition::Cleared);
    CHECK_FALSE(std::filesystem::exists(portable));
    CHECK_FALSE(std::filesystem::exists(driver));
    CHECK_FALSE(std::filesystem::exists(title / "opengl.bin"));
    CHECK(std::filesystem::is_empty(legacy));
    CHECK(std::filesystem::exists(unrelated));
    CHECK(std::filesystem::exists(other / "vulkan.bin"));

    std::ofstream(portable) << "fresh NVN cache";
    CHECK(ClearShaderCachesForNvn(title, true) == NvnCacheTransition::Unchanged);
    CHECK(std::filesystem::exists(portable));
    CHECK(ClearShaderCachesForNvn(directory.path / "missing", false) ==
          NvnCacheTransition::Unchanged);
}

TEST_CASE("Graphics lookahead builds the next pipeline without skipping a draw",
          "[nvn][pipeline-lookahead]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    auto options = AsyncOptions(false);
    options.prefetch_graphics = true;
    PipelineCache cache{*vulkan->device, options};
    const auto vertex = MakeAddProgram(0, 12);
    const auto fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    const auto first = MakeGraphicsRequest(vertex, fragment, &oracle);
    auto next = first;
    next.targets.color[0] = VK_FORMAT_R32_UINT;
    REQUIRE(cache.GetGraphics(first, nullptr, [&](u64 current_key) {
        cache.PrefetchGraphics(first, current_key);
        cache.PrefetchGraphics(next, current_key);
        cache.PrefetchGraphics(next, current_key);
    }));
    bool waiting = true;
    REQUIRE(cache.GetGraphics(next, &waiting));
    CHECK_FALSE(waiting);
    cache.Flush();
    const auto stats = cache.Stats();
    CHECK(stats.compiled_at_runtime == 2);
    CHECK(stats.prefetch_queued == 1);
    CHECK(stats.prefetch_used == 1);
    CHECK(stats.async_not_ready == 0);
}

TEST_CASE("Async compile off: nothing is counted as building", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    VideoCore::ShaderNotify notify;
    PipelineCacheOptions options = AsyncOptions(false);
    options.shader_notify = &notify;
    PipelineCache cache{*vulkan->device, std::move(options)};
    const TestProgram program = MakeAddProgram(5, 77);
    FreezableOracle oracle;
    REQUIRE(cache.GetCompute(MakeComputeRequest(program, &oracle)) != nullptr);
    REQUIRE(notify.NumBuildingNoReport() == 0);
    REQUIRE(notify.ShadersBuilding() == 0);
}

TEST_CASE("Async compile: a rejected program is counted, cached as rejected and never fatal",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    PipelineCache cache{*vulkan->device, AsyncOptions(true)};
    // Valid control, code that no instruction decoder accepts.
    TestProgram garbage = MakeAddProgram(5, 1);
    std::mt19937 random(5);
    for (size_t i = 0; i < 32; ++i) {
        garbage.code[i] = static_cast<u8>(random());
    }
    garbage.hash = HashProgramStage(garbage.control, garbage.code);
    FreezableOracle oracle;
    const ComputeRequest request = MakeComputeRequest(garbage, &oracle);

    bool not_ready = false;
    REQUIRE(cache.GetCompute(request, &not_ready) == nullptr);
    cache.Flush();
    for (int i = 0; i < 3; ++i) {
        not_ready = true;
        REQUIRE(cache.GetCompute(request, &not_ready) == nullptr);
        REQUIRE_FALSE(not_ready); // failed, not "come back later"
    }
    const auto stats = cache.Stats();
    REQUIRE(stats.compiled_at_runtime == 0);
    REQUIRE(stats.translation_failures >= 1);
    REQUIRE(stats.rejected_shaders == 1);
    REQUIRE(stats.async_queued <= 1); // the rejection is remembered: no second compile

    // A graphics request with a rejected stage fails the same way.
    const TestProgram vertex = MakeAddProgram(0, 7);
    TestProgram bad_fragment = MakeTexturedFragment();
    PutWord(bad_fragment.control, 0, 0x12345678); // bad control magic
    bad_fragment.hash = HashProgramStage(bad_fragment.control, bad_fragment.code);
    const GraphicsRequest graphics = MakeGraphicsRequest(vertex, bad_fragment, &oracle);
    not_ready = false;
    REQUIRE(cache.GetGraphics(graphics, &not_ready) == nullptr);
    cache.Flush();
    not_ready = true;
    REQUIRE(cache.GetGraphics(graphics, &not_ready) == nullptr);
    REQUIRE_FALSE(not_ready);
}

TEST_CASE("Async compile off: every lookup is synchronous", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const TestProgram program = MakeAddProgram(5, 9);
    FreezableOracle oracle;

    SECTION("setting off, caller asks for not_ready") {
        PipelineCache cache{*vulkan->device, AsyncOptions(false)};
        bool not_ready = true;
        const auto pipeline = cache.GetCompute(MakeComputeRequest(program, &oracle), &not_ready);
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(not_ready);
        REQUIRE(oracle.freeze_masks.empty());
        REQUIRE(cache.Stats().async_queued == 0);
        REQUIRE(cache.Stats().compiled_at_runtime == 1);
    }
    SECTION("setting on, caller cannot skip") {
        PipelineCache cache{*vulkan->device, AsyncOptions(true)};
        REQUIRE(cache.GetCompute(MakeComputeRequest(program, &oracle)) != nullptr);
        REQUIRE(cache.Stats().async_queued == 0);
        REQUIRE(cache.Stats().compiled_at_runtime == 1);
    }
    SECTION("setting on, oracle cannot be frozen") {
        PipelineCache cache{*vulkan->device, AsyncOptions(true)};
        const UnfreezableOracle live;
        bool not_ready = true;
        const auto pipeline = cache.GetCompute(MakeComputeRequest(program, &live), &not_ready);
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(not_ready);
        REQUIRE(cache.Stats().async_queued == 0);
        // The claim of the key was released: the lookup finds the pipeline again.
        REQUIRE(cache.GetCompute(MakeComputeRequest(program, &live), &not_ready) == pipeline);
    }
    SECTION("graphics, setting off") {
        PipelineCache cache{*vulkan->device, AsyncOptions(false)};
        bool not_ready = true;
        const auto pipeline = cache.GetGraphics(
            MakeGraphicsRequest(MakeAddProgram(0, 3), MakeTexturedFragment(), &oracle),
            &not_ready);
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(not_ready);
        REQUIRE(cache.Stats().async_queued == 0);
    }
}

TEST_CASE("Async compile: compiles are written to the disk cache and a warm cache is not "
          "compiled again",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    TempDirectory directory;
    PipelineCacheOptions options = AsyncOptions(true);
    options.directory = directory.path;
    const TestProgram program = MakeAddProgram(5, 77);
    FreezableOracle oracle;
    {
        PipelineCache cold{*vulkan->device, options};
        cold.Flush(); // preload finished (nothing to load)
        bool not_ready = false;
        REQUIRE(cold.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) == nullptr);
        REQUIRE(not_ready);
        cold.Flush();
        REQUIRE(cold.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) != nullptr);
        REQUIRE(cold.Stats().compiled_at_runtime == 1);
        REQUIRE(cold.Stats().disk_entries_written >= 2); // the shader and the pipeline record
    }
    PipelineCache warm{*vulkan->device, options};
    // Asked at once: ready or not ready, never waited for and never compiled again.
    bool not_ready = false;
    [[maybe_unused]] const auto early =
        warm.GetCompute(MakeComputeRequest(program, &oracle), &not_ready);
    warm.Flush();
    not_ready = true;
    REQUIRE(warm.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) != nullptr);
    REQUIRE_FALSE(not_ready);
    const auto stats = warm.Stats();
    REQUIRE(stats.loaded == 1);
    REQUIRE(stats.compiled_at_runtime == 0);
    REQUIRE(stats.async_queued == 0);
    REQUIRE(stats.preload_waits == 0);
}

namespace {

struct ProgressEvent {
    VideoCore::LoadCallbackStage stage;
    size_t value;
    size_t total;
};

/// Records what the preload reports to the loading screen (workers report too).
struct ProgressRecorder {
    VideoCore::LoadProgressSink sink;
    std::mutex mutex;
    std::vector<ProgressEvent> events;

    ProgressRecorder() {
        sink.Set([this](VideoCore::LoadCallbackStage stage, size_t value, size_t total) {
            std::scoped_lock lock{mutex};
            events.push_back({stage, value, total});
        });
    }
};

} // Anonymous namespace

TEST_CASE("Preload reports Prepare, Build done/total and Complete to the loading screen",
          "[nvn]") {
    using VideoCore::LoadCallbackStage;
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    TempDirectory directory;
    PipelineCacheOptions options = AsyncOptions(true);
    options.directory = directory.path;
    constexpr u32 COUNT = 3;
    FreezableOracle oracle;
    ProgressRecorder cold_progress;
    {
        options.load_progress = &cold_progress.sink;
        PipelineCache cold{*vulkan->device, options};
        cold.Flush();
        for (u32 i = 0; i < COUNT; ++i) {
            bool not_ready = false;
            const TestProgram program = MakeAddProgram(5, 200 + i);
            REQUIRE(cold.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) == nullptr);
            cold.Flush();
        }
        REQUIRE(cold.Stats().compiled_at_runtime == COUNT);
    }
    // A cache file without pipelines has nothing to show.
    REQUIRE(cold_progress.events.empty());

    ProgressRecorder progress;
    options.load_progress = &progress.sink;
    PipelineCache warm{*vulkan->device, options};
    warm.Flush();
    REQUIRE(warm.Stats().loaded == COUNT);

    std::scoped_lock lock{progress.mutex};
    const std::vector<ProgressEvent>& events = progress.events;
    REQUIRE(events.size() >= 4);
    REQUIRE(events.front().stage == LoadCallbackStage::Prepare);
    REQUIRE(events[1].stage == LoadCallbackStage::Build);
    REQUIRE(events[1].value == 0);
    REQUIRE(events[1].total == COUNT);
    REQUIRE(events.back().stage == LoadCallbackStage::Complete);
    size_t previous = 0;
    for (size_t i = 1; i + 1 < events.size(); ++i) {
        REQUIRE(events[i].stage == LoadCallbackStage::Build);
        REQUIRE(events[i].total == COUNT);
        REQUIRE(events[i].value >= previous);
        REQUIRE(events[i].value <= COUNT);
        previous = events[i].value;
    }
    // The last Build is the full count, so the bar ends full before the screen moves on.
    REQUIRE(events[events.size() - 2].value == COUNT);
}

TEST_CASE("Preload reports nothing without a sink or with preload off", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    TempDirectory directory;
    PipelineCacheOptions options = AsyncOptions(true);
    options.directory = directory.path;
    FreezableOracle oracle;
    {
        PipelineCache cold{*vulkan->device, options};
        cold.Flush();
        bool not_ready = false;
        const TestProgram program = MakeAddProgram(5, 210);
        REQUIRE(cold.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) == nullptr);
        cold.Flush();
    }
    SECTION("no sink: loads as before") {
        PipelineCache warm{*vulkan->device, options};
        warm.Flush();
        REQUIRE(warm.Stats().loaded == 1);
    }
    SECTION("preload off: nothing is rebuilt, nothing reported") {
        ProgressRecorder progress;
        options.load_progress = &progress.sink;
        options.preload = false;
        PipelineCache warm{*vulkan->device, options};
        warm.Flush();
        REQUIRE(progress.events.empty());
    }
}

TEST_CASE("The load progress sink forwards to the callback set and ignores calls without one",
          "[nvn]") {
    using VideoCore::LoadCallbackStage;
    VideoCore::LoadProgressSink sink;
    sink.Report(LoadCallbackStage::Build, 1, 2); // no callback: ignored
    size_t calls = 0;
    size_t last_value = 0;
    sink.Set([&](LoadCallbackStage stage, size_t value, size_t total) {
        ++calls;
        last_value = value;
        REQUIRE(stage == LoadCallbackStage::Build);
        REQUIRE(total == 2);
    });
    sink.Report(LoadCallbackStage::Build, 2, 2);
    REQUIRE(calls == 1);
    REQUIRE(last_value == 2);
    sink.Clear();
    sink.Report(LoadCallbackStage::Build, 1, 2);
    REQUIRE(calls == 1);
}

// The use_disk_shader_cache and use_vulkan_driver_pipeline_cache settings (DrawResources::OpenCache
// gives the cache no directory when the first is off, `driver_cache_file = false` for the second).
TEST_CASE("Disk cache settings: off means no file is read, written or created", "[nvn]") {
    namespace fs = std::filesystem;
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    TempDirectory directory;
    const auto vkcache = directory.path / "nvn_pipelines.vkcache";
    const TestProgram program = MakeAddProgram(5, 91);
    FreezableOracle oracle;
    const auto compile_one = [&](const PipelineCacheOptions& options) {
        PipelineCache cache{*vulkan->device, options};
        cache.Flush();
        REQUIRE(cache.GetCompute(MakeComputeRequest(program, &oracle)) != nullptr);
        cache.Flush();
        return cache.Stats();
    };

    SECTION("the driver cache file on: it is created next to the pipeline file") {
        PipelineCacheOptions options = AsyncOptions(false);
        options.directory = directory.path;
        REQUIRE(compile_one(options).compiled_at_runtime == 1);
        REQUIRE(fs::exists(vkcache));
        REQUIRE(CacheFiles(directory.path).size() == 1);
    }
    SECTION("the driver cache file off: pipelines are cached on disk, the driver's blob is not") {
        PipelineCacheOptions options = AsyncOptions(false);
        options.directory = directory.path;
        options.driver_cache_file = false;
        REQUIRE(compile_one(options).compiled_at_runtime == 1);
        REQUIRE_FALSE(fs::exists(vkcache));
        REQUIRE(CacheFiles(directory.path).size() == 1);
        // A warm start still loads the pipeline record from its file.
        REQUIRE(compile_one(options).loaded == 1);
        REQUIRE_FALSE(fs::exists(vkcache));
    }
    SECTION("the driver cache file off leaves an existing file alone") {
        const std::vector<u8> old_blob(8, 0x5A);
        WriteBytes(vkcache, old_blob);
        PipelineCacheOptions options = AsyncOptions(false);
        options.directory = directory.path;
        options.driver_cache_file = false;
        REQUIRE(compile_one(options).compiled_at_runtime == 1);
        REQUIRE(fs::file_size(vkcache) == old_blob.size());
    }
    SECTION("no directory (disk shader cache off): nothing is loaded or written") {
        PipelineCacheOptions options = AsyncOptions(false);
        const auto cold = compile_one(options);
        REQUIRE(cold.compiled_at_runtime == 1);
        REQUIRE(cold.disk_entries_written == 0);
        const auto again = compile_one(options);
        REQUIRE(again.loaded == 0);
        REQUIRE(again.compiled_at_runtime == 1);
        REQUIRE(fs::is_empty(directory.path));
    }
}
TEST_CASE("Async compile: a new build with the same translation sources loads the pipelines of "
          "the one before; a changed translation compiles again",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    TempDirectory directory;
    PipelineCacheOptions options = AsyncOptions(true);
    options.directory = directory.path;
    options.translation_epoch = "translation-sources-1";
    options.revision = "build-1";
    const TestProgram program = MakeAddProgram(5, 78);
    FreezableOracle oracle;
    {
        PipelineCache first_build{*vulkan->device, options};
        first_build.Flush();
        bool not_ready = false;
        REQUIRE(first_build.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) ==
                nullptr);
        first_build.Flush();
        REQUIRE(first_build.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) !=
                nullptr);
        REQUIRE(first_build.Stats().compiled_at_runtime == 1);
    }
    SECTION("same translation sources, new revision: loaded, not compiled") {
        options.revision = "build-2";
        PipelineCache second_build{*vulkan->device, options};
        second_build.Flush();
        bool not_ready = true;
        REQUIRE(second_build.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) !=
                nullptr);
        REQUIRE_FALSE(not_ready);
        const auto stats = second_build.Stats();
        REQUIRE(stats.disk_rejected == 0);
        REQUIRE(stats.loaded == 1);
        REQUIRE(stats.compiled_at_runtime == 0);
    }
    SECTION("changed translation sources: the old file is not used") {
        options.revision = "build-2";
        options.translation_epoch = "translation-sources-2";
        PipelineCache second_build{*vulkan->device, options};
        second_build.Flush();
        const auto stats = second_build.Stats();
        REQUIRE(stats.disk_rejected == 0); // another file, not a rejected one
        REQUIRE(stats.loaded == 0);
        bool not_ready = false;
        REQUIRE(second_build.GetCompute(MakeComputeRequest(program, &oracle), &not_ready) ==
                nullptr);
        REQUIRE(not_ready);
        second_build.Flush();
        REQUIRE(second_build.Stats().compiled_at_runtime == 1);
    }
}

// --- VK_EXT_graphics_pipeline_library path (nvn_pipeline_library setting) ---------------------

namespace {

using Vulkan::Nvn::DynamicFixedState;
using Vulkan::Nvn::TranslationState;
using Vulkan::Nvn::VertexInputMode;

/// A headless device created for a Switch 2 program on `path` with the nvn_pipeline_library
/// setting on (what makes the device enable VK_EXT_graphics_pipeline_library for the NVN HLE);
/// the settings are put back afterwards. Null without Vulkan.
std::unique_ptr<HeadlessVulkan> MakeHeadlessVulkanWithGpl(
    Settings::GraphicsPath path = Settings::GraphicsPath::Hle) {
    const bool library = Settings::values.nvn_pipeline_library.GetValue();
    const Settings::GraphicsPath previous_path = Settings::values.graphics_path.GetValue();
    const bool switch2 = Settings::CurrentProgramIsSwitch2();
    Settings::values.nvn_pipeline_library.SetValue(true);
    Settings::values.graphics_path.SetValue(path);
    Settings::SetCurrentProgramIsSwitch2(true);
    auto vulkan = MakeHeadlessVulkan();
    Settings::values.nvn_pipeline_library.SetValue(library);
    Settings::values.graphics_path.SetValue(previous_path);
    Settings::SetCurrentProgramIsSwitch2(switch2);
    return vulkan;
}

PipelineCacheOptions GplOptions() {
    PipelineCacheOptions options;
    options.revision = "gpl-test";
    options.async_compile = true; // a GPL miss that may be skipped is built on the workers
    options.graphics_pipeline_library = true;
    return options;
}

/// The translation state of a draw state, as the cache keys it (Dynamic vertex input).
std::string TranslationBytes(GraphicsFixedState state, const RenderPassKey& targets) {
    Vulkan::Nvn::MakeKeyState(state, VertexInputMode::Dynamic, false);
    const TranslationState translation =
        Vulkan::Nvn::MakeTranslationState(state, targets, VertexInputMode::Dynamic, false);
    return std::string(reinterpret_cast<const char*>(&translation), sizeof(translation));
}

GraphicsFixedState TrianglesState() {
    GraphicsFixedState s{};
    s.topology = 4; // TRIANGLES
    s.polygon = VC::PolygonState::Defaults();
    s.color = VC::ColorState::Defaults();
    s.channel_mask = 0xFFFF'FFFFu;
    s.sample_mask = 0xFFFF'FFFFu;
    s.depth_mode = 1;
    s.window_origin = 1;
    s.depth_stencil = VC::DepthStencilState::Defaults();
    s.blend.fill(VC::BlendState::Defaults());
    s.attributes[0] = 0x25u;              // RGBA8 (unorm: a float input)
    s.attributes[1] = 0x0Au | (16u << 8); // R32F
    s.stream_stride[0] = 32;
    return s;
}

} // Anonymous namespace

TEST_CASE("Shader refresh rebuilds GPL shader parts and drains optimization workers",
          "[nvn][shader-refresh]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    PipelineCache cache{*vulkan->device, GplOptions()};
    const auto vertex = MakeAddProgram(0, 14);
    const auto fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    const auto request = MakeGraphicsRequest(vertex, fragment, &oracle);
    const auto first = cache.GetGraphics(request);
    REQUIRE(first != nullptr);
    REQUIRE(first->fast_linked);
    cache.ClearShaders();
    CHECK(cache.ShaderGeneration() == 1);
    const auto before = cache.Stats();
    CHECK(before.graphics_pipelines == 0);
    CHECK(before.lowered_programs == 0);
    const auto second = cache.GetGraphics(request);
    REQUIRE(second != nullptr);
    REQUIRE(second->fast_linked);
    CHECK(second != first);
    CHECK(first->Handle() != VK_NULL_HANDLE);
    CHECK(cache.Stats().gpl_translations_reused == before.gpl_translations_reused);
    CHECK(cache.Stats().gpl_cached_links == before.gpl_cached_links);
    cache.Flush();
}

TEST_CASE("TranslationState holds the state the translation reads and nothing else", "[nvn]") {
    RenderPassKey targets;
    targets.color[0] = VK_FORMAT_R8G8B8A8_UNORM;
    targets.depth = VK_FORMAT_D32_SFLOAT;
    const GraphicsFixedState base = TrianglesState();
    const std::string reference = TranslationBytes(base, targets);

    SECTION("pipeline state outside the shaders shares the translation") {
        std::vector<GraphicsFixedState> same;
        GraphicsFixedState s = base;
        s.polygon = VC::PolygonState::WithCullFace(base.polygon, VC::STATE_FACE_BACK);
        same.push_back(s);
        s = base;
        s.polygon = VC::PolygonState::WithFrontFace(base.polygon, 0);
        same.push_back(s);
        s = base;
        s.depth_stencil = VC::DepthStencilState::WithDepthTestEnable(base.depth_stencil, 1);
        s.depth_stencil = VC::DepthStencilState::WithDepthFunc(s.depth_stencil, 3);
        same.push_back(s);
        s = base;
        s.depth_stencil = VC::DepthStencilState::WithStencilTestEnable(base.depth_stencil, 1);
        same.push_back(s);
        s = base;
        s.color = VC::ColorState::WithBlendEnabled(base.color, 0, true);
        s.blend[0] = VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), 5);
        same.push_back(s);
        s = base;
        s.channel_mask = 0x7777'7777u;
        same.push_back(s);
        s = base;
        s.sample_mask = 0x0000'000Fu;
        s.depth_clamp = 1;
        same.push_back(s);
        s = base;
        s.attributes[0] = 0x26u | (4u << 8); // RGBA8 snorm at another offset: still float
        s.stream_stride[0] = 48;
        s.stream_divisor[0] = 2;
        same.push_back(s);
        s = base;
        s.topology = 5; // TRIANGLE_STRIP: the same input topology class
        same.push_back(s);
        for (const GraphicsFixedState& variant : same) {
            REQUIRE(TranslationBytes(variant, targets) == reference);
        }
        RenderPassKey other_float = targets;
        other_float.color[0] = VK_FORMAT_R16G16B16A16_SFLOAT;
        other_float.depth = VK_FORMAT_D24_UNORM_S8_UINT;
        other_float.samples = 4;
        REQUIRE(TranslationBytes(base, other_float) == reference);
    }

    SECTION("state the translation reads changes it") {
        std::vector<GraphicsFixedState> different;
        GraphicsFixedState s = base;
        s.topology = 0; // POINTS
        different.push_back(s);
        s = base;
        s.topology = 12; // TRIANGLES_ADJACENCY
        different.push_back(s);
        s = base;
        s.depth_mode = 0;
        different.push_back(s);
        s = base;
        s.color = VC::ColorState::WithAlphaFunc(base.color, 2); // LESS
        different.push_back(s);
        s = base;
        s.color = VC::ColorState::WithBlendEnabled(base.color, 0, true);
        s.blend[0] = VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), 0x10); // SRC1
        different.push_back(s);
        s = base;
        s.attributes[0] = 0x27u; // RGBA8 uint: an integer input
        different.push_back(s);
        s = base;
        s.topology = 0;
        s.point_size_bits = std::bit_cast<u32>(4.0f);
        different.push_back(s);
        std::set<std::string> seen{reference};
        for (const GraphicsFixedState& variant : different) {
            REQUIRE(seen.insert(TranslationBytes(variant, targets)).second);
        }
        GraphicsFixedState alpha = base;
        alpha.color = VC::ColorState::WithAlphaFunc(base.color, 2);
        GraphicsFixedState alpha_ref = alpha;
        alpha_ref.alpha_ref_bits = std::bit_cast<u32>(0.5f);
        REQUIRE(TranslationBytes(alpha, targets) != TranslationBytes(alpha_ref, targets));
        RenderPassKey integer = targets;
        integer.color[0] = VK_FORMAT_R32_UINT;
        REQUIRE(TranslationBytes(base, integer) != reference);
        RenderPassKey second_integer = targets;
        second_integer.color[3] = VK_FORMAT_R16G16_SINT;
        REQUIRE(TranslationBytes(base, second_integer) != reference);
    }
}

TEST_CASE("Dynamic fixed state: the static create info and the per-draw values agree", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    RenderPassKey targets;
    targets.color[0] = VK_FORMAT_R8G8B8A8_UNORM;
    targets.depth = VK_FORMAT_D24_UNORM_S8_UINT;
    std::vector<GraphicsFixedState> states;
    for (u32 i = 0; i < 12; ++i) {
        GraphicsFixedState s = TrianglesState();
        s.polygon = VC::PolygonState::WithCullFace(s.polygon, i % 4);
        s.polygon = VC::PolygonState::WithFrontFace(s.polygon, i % 2);
        s.polygon = VC::PolygonState::WithOffsetEnables(s.polygon, (i % 3) == 0 ? 4 : 0);
        s.depth_stencil = VC::DepthStencilState::WithDepthTestEnable(s.depth_stencil, i % 2);
        s.depth_stencil = VC::DepthStencilState::WithDepthWriteEnable(s.depth_stencil, (i / 2) % 2);
        s.depth_stencil = VC::DepthStencilState::WithDepthFunc(s.depth_stencil, 1 + i % 8);
        s.depth_stencil = VC::DepthStencilState::WithStencilTestEnable(s.depth_stencil, (i / 3) % 2);
        s.depth_stencil = VC::DepthStencilState::WithStencilOps(
            s.depth_stencil, VC::STATE_FACE_FRONT, 1 + i % 8, 1 + (i + 1) % 8, 1 + (i + 2) % 8);
        s.depth_stencil =
            VC::DepthStencilState::WithStencilFunc(s.depth_stencil, VC::STATE_FACE_BACK, 1 + i % 8);
        states.push_back(s);
    }
    const auto describe = [&](const GraphicsFixedState& state, bool dynamic_fixed,
                              Vulkan::Nvn::GraphicsPipelineDescription& out) {
        std::vector<std::string> issues;
        Vulkan::Nvn::DescribeGraphicsPipeline(*vulkan->device,
                                              Vulkan::Nvn::GraphicsPipelineParams{
                                                  .state = &state,
                                                  .targets = &targets,
                                                  .vertex_input = VertexInputMode::Dynamic,
                                                  .dynamic_fixed = dynamic_fixed,
                                              },
                                              out, issues);
    };
    constexpr std::array<VkDynamicState, 8> FIXED{
        VK_DYNAMIC_STATE_CULL_MODE,           VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,   VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,          VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    };
    const auto declares = [](const Vulkan::Nvn::GraphicsPipelineDescription& d, VkDynamicState s) {
        return std::ranges::find(d.dynamic_states, s) != d.dynamic_states.end();
    };
    Vulkan::Nvn::GraphicsPipelineDescription first_dynamic;
    describe(states[0], true, first_dynamic);
    for (const GraphicsFixedState& state : states) {
        const DynamicFixedState values = Vulkan::Nvn::MakeDynamicFixedState(state);
        // Static: the create info holds exactly the values the replay would set dynamically.
        Vulkan::Nvn::GraphicsPipelineDescription fixed;
        describe(state, false, fixed);
        REQUIRE(fixed.rasterization.cullMode == values.cull_mode);
        REQUIRE(fixed.rasterization.frontFace == values.front_face);
        REQUIRE((fixed.rasterization.depthBiasEnable == VK_TRUE) == values.depth_bias_enable);
        REQUIRE((fixed.depth_stencil.depthTestEnable == VK_TRUE) == values.depth_test_enable);
        REQUIRE((fixed.depth_stencil.depthWriteEnable == VK_TRUE) == values.depth_write_enable);
        REQUIRE(fixed.depth_stencil.depthCompareOp == values.depth_compare_op);
        REQUIRE((fixed.depth_stencil.stencilTestEnable == VK_TRUE) == values.stencil_test_enable);
        REQUIRE(fixed.depth_stencil.front.failOp == values.front.fail);
        REQUIRE(fixed.depth_stencil.front.passOp == values.front.pass);
        REQUIRE(fixed.depth_stencil.front.depthFailOp == values.front.depth_fail);
        REQUIRE(fixed.depth_stencil.front.compareOp == values.front.compare);
        REQUIRE(fixed.depth_stencil.back.compareOp == values.back.compare);
        for (const VkDynamicState dynamic : FIXED) {
            REQUIRE_FALSE(declares(fixed, dynamic));
        }
        // Dynamic: declared, and the create info no longer depends on them (equal library keys).
        Vulkan::Nvn::GraphicsPipelineDescription dynamic;
        describe(state, true, dynamic);
        for (const VkDynamicState s : FIXED) {
            REQUIRE(declares(dynamic, s));
        }
        const auto& raster = dynamic.rasterization;
        const auto& raster0 = first_dynamic.rasterization;
        REQUIRE(raster.cullMode == raster0.cullMode);
        REQUIRE(raster.frontFace == raster0.frontFace);
        REQUIRE(raster.depthBiasEnable == raster0.depthBiasEnable);
        const auto& ds = dynamic.depth_stencil;
        const auto& ds0 = first_dynamic.depth_stencil;
        REQUIRE(ds.depthTestEnable == ds0.depthTestEnable);
        REQUIRE(ds.depthWriteEnable == ds0.depthWriteEnable);
        REQUIRE(ds.depthCompareOp == ds0.depthCompareOp);
        REQUIRE(ds.stencilTestEnable == ds0.stencilTestEnable);
        using Faces = std::pair<VkStencilOpState, VkStencilOpState>;
        for (const auto& [face, face0] :
             std::array<Faces, 2>{Faces{ds.front, ds0.front}, Faces{ds.back, ds0.back}}) {
            REQUIRE(face.failOp == face0.failOp);
            REQUIRE(face.passOp == face0.passOp);
            REQUIRE(face.depthFailOp == face0.depthFailOp);
            REQUIRE(face.compareOp == face0.compareOp);
        }
        REQUIRE(dynamic.dynamic_states.size() == fixed.dynamic_states.size() + FIXED.size());
    }
}

TEST_CASE("Provoking vertex: NVN's last vertex by default, the first when the game asks", "[nvn]") {
    // The zero state is NVN's default (the SDK's initial GPU state selects the last vertex), so
    // the replay's initial state and every disk record written before the field existed keep it.
    GraphicsFixedState last = TrianglesState();
    REQUIRE(last.provoking_first == 0);
    GraphicsFixedState first = last;
    first.provoking_first = 7;
    Disk::CanonicalizeState(first, VertexInputMode::Dynamic);
    CHECK(first.provoking_first == 1);
    Disk::CanonicalizeState(last, VertexInputMode::Dynamic);
    CHECK(last.provoking_first == 0);

    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    RenderPassKey targets;
    targets.color[0] = VK_FORMAT_R8G8B8A8_UNORM;
    const auto describe = [&](const GraphicsFixedState& state,
                              Vulkan::Nvn::GraphicsPipelineDescription& out) {
        std::vector<std::string> issues;
        Vulkan::Nvn::DescribeGraphicsPipeline(*vulkan->device,
                                              Vulkan::Nvn::GraphicsPipelineParams{
                                                  .state = &state,
                                                  .targets = &targets,
                                                  .vertex_input = VertexInputMode::Dynamic,
                                              },
                                              out, issues);
        return std::ranges::count_if(issues, [](const std::string& issue) {
            return issue.starts_with("provoking vertex");
        });
    };
    const auto chained = [](const Vulkan::Nvn::GraphicsPipelineDescription& d)
        -> const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT* {
        for (auto* next = static_cast<const VkBaseInStructure*>(d.rasterization.pNext);
             next != nullptr; next = next->pNext) {
            if (next->sType ==
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT) {
                return reinterpret_cast<
                    const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT*>(next);
            }
        }
        return nullptr;
    };

    Vulkan::Nvn::GraphicsPipelineDescription last_ci;
    const auto last_issues = describe(last, last_ci);
    if (vulkan->device->SupportsProvokingVertexLastMode()) {
        const auto* mode = chained(last_ci);
        REQUIRE(mode != nullptr);
        CHECK(mode->provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
        CHECK(last_issues == 0);
    } else {
        // The fallback is visible: one pipeline issue, Vulkan's first vertex.
        CHECK(chained(last_ci) == nullptr);
        CHECK(last_issues == 1);
    }
    // The game's first vertex is Vulkan's default: nothing chained, nothing to report.
    Vulkan::Nvn::GraphicsPipelineDescription first_ci;
    CHECK(describe(first, first_ci) == 0);
    CHECK(chained(first_ci) == nullptr);
}

TEST_CASE("GPL: the vertex input part of a mesh pipeline has valid vertex input state", "[nvn]") {
    // A mesh pipeline has no vertex input and input assembly state (a mesh shader ignores both),
    // and its part list has no stage to say so: NVIDIA's driver 621.01 read
    // pInputAssemblyState of the vertex input part without a null check and crashed.
    VkGraphicsPipelineCreateInfo mesh_ci{};
    mesh_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    const auto library = [](VkGraphicsPipelineLibraryFlagsEXT flags) {
        return VkGraphicsPipelineLibraryCreateInfoEXT{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
            .pNext = nullptr,
            .flags = flags,
        };
    };
    const auto vertex_input_library =
        library(VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT);
    const VkGraphicsPipelineCreateInfo part =
        Vulkan::MakeLibraryPartCreateInfo(mesh_ci, vertex_input_library, {});
    REQUIRE(part.pVertexInputState != nullptr);
    REQUIRE(part.pInputAssemblyState != nullptr);
    CHECK(part.pVertexInputState->vertexBindingDescriptionCount == 0);
    CHECK(part.pVertexInputState->vertexAttributeDescriptionCount == 0);
    CHECK(part.pInputAssemblyState->primitiveRestartEnable == VK_FALSE);
    CHECK(part.stageCount == 0);
    CHECK(part.layout == VK_NULL_HANDLE);
    CHECK((part.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0);

    // The other parts do not take the two states, null or not.
    for (const VkGraphicsPipelineLibraryFlagsEXT flags :
         {VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
          VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
          VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT}) {
        const auto other_library = library(flags);
        const VkGraphicsPipelineCreateInfo other =
            Vulkan::MakeLibraryPartCreateInfo(mesh_ci, other_library, {});
        CHECK(other.pVertexInputState == nullptr);
        CHECK(other.pInputAssemblyState == nullptr);
    }

    // A pipeline that has the states keeps its own.
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST};
    VkGraphicsPipelineCreateInfo vertex_ci = mesh_ci;
    vertex_ci.pVertexInputState = &vertex_input;
    vertex_ci.pInputAssemblyState = &input_assembly;
    const VkGraphicsPipelineCreateInfo own =
        Vulkan::MakeLibraryPartCreateInfo(vertex_ci, vertex_input_library, {});
    CHECK(own.pVertexInputState == &vertex_input);
    CHECK(own.pInputAssemblyState == &input_assembly);

    // The driver takes the part (on 621.01 the null states crashed in here).
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    const Vulkan::vk::Pipeline created =
        vulkan->device->GetLogical().CreateGraphicsPipeline(part, VK_NULL_HANDLE);
    CHECK(*created != VK_NULL_HANDLE);
}

TEST_CASE("GPL: a miss is fast-linked on the calling thread and the whole compile is swapped in",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    if (!vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no VK_EXT_graphics_pipeline_library with fast linking");
    }
    // A lookup that may not skip the draw (no `not_ready`) builds the pipeline itself: the
    // asynchronous workers are for draws that can wait.
    PipelineCache cache{*vulkan->device, GplOptions()};
    REQUIRE(cache.UsesGraphicsPipelineLibrary());
    const TestProgram vertex = MakeAddProgram(0, 11);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    const GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);

    const auto pipeline = cache.GetGraphics(request);
    REQUIRE(pipeline != nullptr);
    REQUIRE(pipeline->fast_linked);
    REQUIRE(oracle.freeze_masks.empty()); // nothing went to the asynchronous workers

    cache.Flush(); // the optimize job finished
    const VkPipeline optimized = pipeline->optimized_handle.load();
    REQUIRE(optimized != VK_NULL_HANDLE);
    REQUIRE(optimized == *pipeline->optimized);
    REQUIRE(optimized != *pipeline->pipeline); // the fast-linked one stays alive beside it
    REQUIRE(pipeline->Handle() == optimized);
    // The same key finds the same pipeline object (its handle switched in place).
    REQUIRE(cache.GetGraphics(request) == pipeline);

    const auto stats = cache.Stats();
    REQUIRE(stats.compiled_at_runtime == 1);
    REQUIRE(stats.gpl_fast_links == 1);
    REQUIRE(stats.gpl_cached_links == 0);
    REQUIRE(stats.gpl_optimized == 1);
    REQUIRE(stats.gpl_whole == 0);
    REQUIRE(stats.async_queued == 0);
    REQUIRE(stats.async_not_ready == 0);
}

TEST_CASE("GPL with asynchronous shaders: a draw that may be skipped waits for the workers",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    PipelineCache cache{*vulkan->device, GplOptions()};
    REQUIRE(cache.UsesGraphicsPipelineLibrary());
    const TestProgram vertex = MakeAddProgram(0, 21);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    const GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);

    // The worker is held inside its first texture query (the translation, the first step of the
    // build): the lookup must answer at once, so neither the translation, the library parts nor
    // the link run on the calling thread.
    oracle.SetGate(false);
    bool not_ready = false;
    REQUIRE(cache.GetGraphics(request, &not_ready) == nullptr);
    REQUIRE(not_ready);
    REQUIRE(oracle.freeze_masks == std::vector<u32>{0b00011});
    REQUIRE(cache.Stats().async_queued == 1);
    REQUIRE(cache.Stats().compiled_at_runtime == 0);
    REQUIRE(cache.Stats().gpl_fast_links == 0);
    // Asked again while it builds: still not ready, nothing more queued.
    not_ready = false;
    REQUIRE(cache.GetGraphics(request, &not_ready) == nullptr);
    REQUIRE(not_ready);
    REQUIRE(cache.Stats().async_queued == 1);

    oracle.SetGate(true);
    cache.Flush(); // the build and the optimize job finished
    not_ready = true;
    const auto pipeline = cache.GetGraphics(request, &not_ready);
    REQUIRE(pipeline != nullptr);
    REQUIRE_FALSE(not_ready);
    REQUIRE(pipeline->fast_linked);
    REQUIRE(pipeline->optimized_handle.load() != VK_NULL_HANDLE);
    REQUIRE(pipeline->Handle() == *pipeline->optimized);

    const auto stats = cache.Stats();
    REQUIRE(stats.async_queued == 1);
    REQUIRE(stats.async_not_ready == 2);
    REQUIRE(stats.async_failed == 0);
    REQUIRE(stats.compiled_at_runtime == 1);
    REQUIRE(stats.gpl_fast_links == 1);
    REQUIRE(stats.gpl_optimized == 1);
}

TEST_CASE("GPL with asynchronous shaders: a draw that must not be skipped builds inline",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    FreezableOracle oracle;
    const TestProgram vertex = MakeAddProgram(0, 22);
    const TestProgram fragment = MakeTexturedFragment();
    const GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);
    SECTION("no not_ready: the default path's rule for draws that cannot be skipped") {
        PipelineCache cache{*vulkan->device, GplOptions()};
        const auto pipeline = cache.GetGraphics(request);
        REQUIRE(pipeline != nullptr);
        REQUIRE(pipeline->fast_linked);
        REQUIRE(oracle.freeze_masks.empty());
        cache.Flush();
        REQUIRE(cache.Stats().async_queued == 0);
        REQUIRE(cache.Stats().async_not_ready == 0);
    }
    SECTION("asynchronous shaders off: never not ready") {
        PipelineCacheOptions options = GplOptions();
        options.async_compile = false;
        PipelineCache cache{*vulkan->device, options};
        bool not_ready = true;
        const auto pipeline = cache.GetGraphics(request, &not_ready);
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(not_ready);
        REQUIRE(pipeline->fast_linked);
        REQUIRE(oracle.freeze_masks.empty());
        cache.Flush();
        REQUIRE(cache.Stats().async_queued == 0);
    }
}

TEST_CASE("GPL with asynchronous shaders: workers building one translation key translate it once",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    PipelineCache cache{*vulkan->device, GplOptions()};
    const TestProgram vertex = MakeAddProgram(0, 23);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);
    request.state.color = VC::ColorState::Defaults();
    request.state.blend.fill(VC::BlendState::Defaults());

    // Four pipelines that differ in the blend function only (the translation does not read it)
    // are queued while the first worker is held in the translation: the others must wait for
    // its set instead of translating and compiling shader parts of their own.
    constexpr u32 FIRST_FUNC = 3;
    constexpr u32 COUNT = 4;
    oracle.SetGate(false);
    for (u32 i = 0; i < COUNT; ++i) {
        GraphicsRequest blended = request;
        blended.state.color = VC::ColorState::WithBlendEnabled(request.state.color, 0, true);
        blended.state.blend[0] =
            VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), FIRST_FUNC + i);
        bool not_ready = false;
        REQUIRE(cache.GetGraphics(blended, &not_ready) == nullptr);
        REQUIRE(not_ready);
    }
    REQUIRE(cache.Stats().async_queued == COUNT);
    oracle.SetGate(true);
    cache.Flush();

    const auto stats = cache.Stats();
    REQUIRE(stats.async_failed == 0);
    REQUIRE(stats.compiled_at_runtime == COUNT);
    REQUIRE(stats.gpl_fast_links == COUNT);
    REQUIRE(stats.gpl_translations_reused == COUNT - 1);
    REQUIRE(stats.gpl_optimized == COUNT);
}

TEST_CASE("GPL: state variants reuse the translation and the shader parts", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    PipelineCache cache{*vulkan->device, GplOptions()};
    const TestProgram vertex = MakeAddProgram(0, 12);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);
    request.state.polygon = VC::PolygonState::Defaults();
    request.state.color = VC::ColorState::Defaults();
    request.state.channel_mask = 0xFFFF'FFFFu;
    request.state.depth_stencil = VC::DepthStencilState::Defaults();
    request.state.blend.fill(VC::BlendState::Defaults());
    request.targets.depth = VK_FORMAT_D32_SFLOAT;

    const auto first = cache.GetGraphics(request);
    REQUIRE(first != nullptr);
    REQUIRE(first->fast_linked);

    // Another blend state: a new fragment output part, everything else reused.
    GraphicsRequest blended = request;
    blended.state.color = VC::ColorState::WithBlendEnabled(request.state.color, 0, true);
    blended.state.blend[0] = VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), 5);
    const auto second = cache.GetGraphics(blended);
    REQUIRE(second != nullptr);
    REQUIRE(second != first);
    REQUIRE(second->fast_linked);
    REQUIRE(cache.Stats().gpl_translations_reused == 1);

    // Another cull and depth state: with dynamic cull/depth/stencil every part is cached.
    GraphicsRequest culled = request;
    culled.state.polygon = VC::PolygonState::WithCullFace(request.state.polygon,
                                                          VC::STATE_FACE_BACK);
    culled.state.depth_stencil =
        VC::DepthStencilState::WithDepthTestEnable(request.state.depth_stencil, 1);
    const auto third = cache.GetGraphics(culled);
    REQUIRE(third != nullptr);
    REQUIRE(third->fast_linked);
    REQUIRE(cache.Stats().gpl_translations_reused == 2);
    if (third->dynamic_fixed) {
        REQUIRE(cache.Stats().gpl_cached_links == 1);
        REQUIRE(third->fixed_values == Vulkan::Nvn::MakeDynamicFixedState(culled.state));
        REQUIRE(third->fixed_values.cull_mode == VK_CULL_MODE_BACK_BIT);
        REQUIRE(third->fixed_values.depth_test_enable);
        REQUIRE(first->fixed_values.cull_mode == VK_CULL_MODE_NONE);
    }

    // A texture answered with another type: the translation depends on it, so it is new.
    oracle.answer.type = Shader::TextureType::ColorArray2D;
    const auto array_variant = cache.GetGraphics(request);
    REQUIRE(array_variant != nullptr);
    REQUIRE(array_variant != first);
    REQUIRE(cache.Stats().gpl_translations_reused == 2);
    // Back to the first answer: the first pipeline again, nothing built.
    oracle.answer.type = Shader::TextureType::Color2D;
    REQUIRE(cache.GetGraphics(request) == first);

    // An integer color target changes the fragment translation.
    GraphicsRequest integer = request;
    integer.targets.color[0] = VK_FORMAT_R32_UINT;
    const auto integer_pipeline = cache.GetGraphics(integer);
    REQUIRE(integer_pipeline != nullptr);
    REQUIRE(cache.Stats().gpl_translations_reused == 2);

    cache.Flush();
    const auto stats = cache.Stats();
    REQUIRE(stats.compiled_at_runtime == 5);
    REQUIRE(stats.gpl_fast_links == 5);
    REQUIRE(stats.gpl_optimized == 5);
    for (const auto& pipeline : {first, second, third, array_variant, integer_pipeline}) {
        REQUIRE(pipeline->Handle() == *pipeline->optimized);
    }
}

TEST_CASE("GPL: the disk cache keeps working; a warm start loads whole pipelines", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    TempDirectory directory;
    PipelineCacheOptions options = GplOptions();
    options.directory = directory.path;
    options.translation_epoch = "gpl-epoch-1";
    const TestProgram vertex = MakeAddProgram(0, 13);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    const GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);
    {
        PipelineCache cold{*vulkan->device, options};
        cold.Flush();
        const auto pipeline = cold.GetGraphics(request);
        REQUIRE(pipeline != nullptr);
        REQUIRE(pipeline->fast_linked);
        cold.Flush();
        REQUIRE(cold.Stats().disk_entries_written >= 3); // two shaders and the pipeline
    }
    SECTION("same translation key: loaded, not compiled") {
        PipelineCache warm{*vulkan->device, options};
        warm.Flush();
        bool not_ready = true;
        const auto pipeline = warm.GetGraphics(request, &not_ready);
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(not_ready);
        REQUIRE_FALSE(pipeline->fast_linked); // the preload compiles whole
        const auto stats = warm.Stats();
        REQUIRE(stats.loaded == 1);
        REQUIRE(stats.compiled_at_runtime == 0);
        REQUIRE(stats.gpl_fast_links == 0);
    }
    SECTION("changed translation epoch: the old file is not used") {
        options.translation_epoch = "gpl-epoch-2";
        PipelineCache next{*vulkan->device, options};
        next.Flush();
        REQUIRE(next.Stats().loaded == 0);
        REQUIRE(next.GetGraphics(request) != nullptr);
        REQUIRE(next.Stats().gpl_fast_links == 1);
    }
}

TEST_CASE("GPL: the preload compiles the library parts of the cached shader sets", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    TempDirectory directory;
    PipelineCacheOptions options = GplOptions();
    options.directory = directory.path;
    options.translation_epoch = "gpl-warm-epoch";
    const TestProgram vertex = MakeAddProgram(0, 31);
    const TestProgram fragment = MakeTexturedFragment();
    FreezableOracle oracle;
    GraphicsRequest request = MakeGraphicsRequest(vertex, fragment, &oracle);
    request.state.polygon = VC::PolygonState::Defaults();
    request.state.color = VC::ColorState::Defaults();
    request.state.channel_mask = 0xFFFF'FFFFu;
    request.state.depth_stencil = VC::DepthStencilState::Defaults();
    request.state.blend.fill(VC::BlendState::Defaults());
    request.targets.depth = VK_FORMAT_D32_SFLOAT;
    // Three pipelines of one shader set (they differ in the blend function only) are cached.
    constexpr u32 FIRST_FUNC = 3;
    constexpr u32 RECORDS = 3;
    {
        PipelineCache cold{*vulkan->device, options};
        cold.Flush();
        for (u32 i = 0; i < RECORDS; ++i) {
            GraphicsRequest blended = request;
            blended.state.color = VC::ColorState::WithBlendEnabled(request.state.color, 0, true);
            blended.state.blend[0] =
                VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), FIRST_FUNC + i);
            REQUIRE(cold.GetGraphics(blended) != nullptr);
        }
        cold.Flush();
    }

    SECTION("one shader set is warmed; the first run-time pipeline of it links cached parts") {
        PipelineCache warm{*vulkan->device, options};
        warm.Flush(); // the preload and the part warm-up finished
        auto stats = warm.Stats();
        REQUIRE(stats.loaded == RECORDS);
        REQUIRE(stats.gpl_shader_sets_warmed == 1); // one record per set, not per pipeline
        // The warm-up is no run-time work: nothing counted as compiled or linked.
        REQUIRE(stats.compiled_at_runtime == 0);
        REQUIRE(stats.gpl_fast_links == 0);
        REQUIRE(stats.gpl_cached_links == 0);
        REQUIRE(stats.gpl_whole == 0);

        // A new key of the set (another cull and depth state: dynamic, so not in the parts):
        // the translation is the warmed set's and, when the state is dynamic, every part was
        // compiled by the warm-up.
        GraphicsRequest culled = request;
        culled.state.color = VC::ColorState::WithBlendEnabled(request.state.color, 0, true);
        // The warm-up takes the newest record of the set (the last one the cold run cached): the
        // blend function of the fragment output part is that record's.
        culled.state.blend[0] =
            VC::BlendState::WithSrcRgbFunc(VC::BlendState::Defaults(), FIRST_FUNC + RECORDS - 1);
        culled.state.polygon = VC::PolygonState::WithCullFace(request.state.polygon,
                                                              VC::STATE_FACE_BACK);
        culled.state.depth_stencil =
            VC::DepthStencilState::WithDepthTestEnable(request.state.depth_stencil, 1);
        const auto pipeline = warm.GetGraphics(culled);
        REQUIRE(pipeline != nullptr);
        REQUIRE(pipeline->fast_linked);
        stats = warm.Stats();
        REQUIRE(stats.gpl_translations_reused == 1);
        if (pipeline->dynamic_fixed) {
            REQUIRE(stats.gpl_cached_links == 1);
        }
    }
    SECTION("pipeline library off: the preload warms nothing") {
        options.graphics_pipeline_library = false;
        PipelineCache plain{*vulkan->device, options};
        plain.Flush();
        REQUIRE_FALSE(plain.UsesGraphicsPipelineLibrary());
        REQUIRE(plain.Stats().loaded == RECORDS);
        REQUIRE(plain.Stats().gpl_shader_sets_warmed == 0);
    }
    SECTION("without the preload nothing is warmed") {
        options.preload = false;
        PipelineCache lazy{*vulkan->device, options};
        lazy.Flush();
        REQUIRE(lazy.Stats().loaded == 0);
        REQUIRE(lazy.Stats().gpl_shader_sets_warmed == 0);
    }
}

TEST_CASE("GPL: programs are lowered when created; nothing happens with the switch off", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkanWithGpl();
    if (!vulkan || !vulkan->device->IsExtGraphicsPipelineLibrarySupported()) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    auto program = std::make_shared<std::array<TestProgram, 2>>(
        std::array<TestProgram, 2>{MakeAddProgram(0, 14), MakeTexturedFragment()});
    std::array<StageProgram, Vulkan::Nvn::STAGE_COUNT> stages{};
    stages[0] = (*program)[0].View();
    stages[1] = (*program)[1].View();
    {
        PipelineCache cache{*vulkan->device, GplOptions()};
        cache.PrepareProgram(stages, program);
        cache.Flush();
        REQUIRE(cache.Stats().programs_prepared == 2);
    }
    {
        PipelineCacheOptions options = GplOptions();
        options.graphics_pipeline_library = false;
        PipelineCache cache{*vulkan->device, options};
        REQUIRE_FALSE(cache.UsesGraphicsPipelineLibrary());
        cache.PrepareProgram(stages, program);
        cache.Flush();
        REQUIRE(cache.Stats().programs_prepared == 0);
        // The default path: compiled whole, the draw skipped until then (asynchronous setting).
        FreezableOracle oracle;
        bool not_ready = false;
        REQUIRE(cache.GetGraphics(MakeGraphicsRequest((*program)[0], (*program)[1], &oracle),
                                  &not_ready) == nullptr);
        REQUIRE(not_ready);
        cache.Flush();
        const auto pipeline =
            cache.GetGraphics(MakeGraphicsRequest((*program)[0], (*program)[1], &oracle));
        REQUIRE(pipeline != nullptr);
        REQUIRE_FALSE(pipeline->fast_linked);
        REQUIRE_FALSE(pipeline->dynamic_fixed);
        REQUIRE(pipeline->Handle() == *pipeline->pipeline);
        REQUIRE(cache.Stats().gpl_fast_links == 0);
    }
}

TEST_CASE("GPL: the emulated GPU path ignores nvn_pipeline_library", "[nvn]") {
    // The extension costs memory (VK_EXT_graphics_pipeline_library, see vulkan_device.cpp) and
    // only the NVN HLE uses it for this setting: on the emulated GPU the device leaves it off.
    bool hle_has_it = false;
    {
        const auto hle = MakeHeadlessVulkanWithGpl(Settings::GraphicsPath::Hle);
        hle_has_it = hle && hle->device->IsExtGraphicsPipelineLibrarySupported();
    }
    if (!hle_has_it) {
        SKIP("no Vulkan device with VK_EXT_graphics_pipeline_library");
    }
    const auto emulated = MakeHeadlessVulkanWithGpl(Settings::GraphicsPath::Emulated);
    REQUIRE(emulated);
    REQUIRE_FALSE(emulated->device->IsExtGraphicsPipelineLibrarySupported());
}
