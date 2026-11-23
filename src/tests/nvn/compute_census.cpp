// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Hidden census of the NVN2 compute shaders of a game, from the shader store the coverage
// supervisor writes (C:\eg\results\nvn\work\store.bin). Verifies the control-section fields the
// compute path relies on and lists, per shader, what translation produced:
//   tests "[compute-census]"     writes C:\eg\results\nvn\compute-census.tsv
// The reference profile has no cooperative-matrix support, so the tensor (HMMA) shaders are
// listed as rejected; EDEN_NX2_TENSOR=stub lowers them to a passthrough instead.
// EDEN_NVN_CENSUS_SKIP=<index,index,...> skips shaders that abort the process (rerun to resume).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "video_core/nvn/shader/nvn_descriptor_layout.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

namespace fs = std::filesystem;
using namespace VideoCommon::Nvn;

const fs::path ResultsRoot = "C:/eg/results/nvn";
const fs::path WorkDir = ResultsRoot / "work";
constexpr u32 StoreMagic = 0x3153564e; // "NVS1"

struct StoredShader {
    std::string key, source, kind;
    u32 stage_byte{};
    std::vector<u8> control, code;
};

bool ReadStored(std::ifstream& store, u64 offset, StoredShader& shader) {
    store.clear();
    store.seekg(static_cast<std::streamoff>(offset));
    u32 header[7];
    store.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!store || header[0] != StoreMagic) {
        return false;
    }
    shader.stage_byte = header[1];
    shader.source.resize(header[2]);
    shader.kind.resize(header[3]);
    shader.key.resize(header[4]);
    shader.control.resize(header[5]);
    shader.code.resize(header[6]);
    store.read(shader.source.data(), header[2]);
    store.read(shader.kind.data(), header[3]);
    store.read(shader.key.data(), header[4]);
    store.read(reinterpret_cast<char*>(shader.control.data()), header[5]);
    store.read(reinterpret_cast<char*>(shader.code.data()), header[6]);
    return static_cast<bool>(store);
}

std::string Clean(std::string text) {
    for (char& c : text) {
        if (c == '\t' || c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    return text.substr(0, 160);
}

} // namespace

TEST_CASE("NVN compute shader census", "[.compute-census]") {
    std::ifstream index_file(WorkDir / "store.idx", std::ios::binary | std::ios::ate);
    if (!index_file.is_open()) {
        SKIP("no shader store (run the coverage supervisor first)");
    }
    std::vector<u64> offsets(static_cast<size_t>(index_file.tellg()) / sizeof(u64));
    index_file.seekg(0);
    index_file.read(reinterpret_cast<char*>(offsets.data()),
                    static_cast<std::streamsize>(offsets.size() * sizeof(u64)));
    std::ifstream store(WorkDir / "store.bin", std::ios::binary);
    REQUIRE(store.is_open());

    std::set<u64> skip;
    if (const char* text = std::getenv("EDEN_NVN_CENSUS_SKIP")) {
        std::stringstream stream(text);
        std::string item;
        while (std::getline(stream, item, ',')) {
            skip.insert(std::strtoull(item.c_str(), nullptr, 10));
        }
    }
    const char* suffix = std::getenv("EDEN_NVN_CENSUS_SUFFIX");
    std::ofstream out(ResultsRoot / (std::string{"compute-census"} + (suffix ? suffix : "") + ".tsv"));
    REQUIRE(out.is_open());
    out << "index\tsource\tgroup\tpush_words\tuniform\tstorage\ttexture\timage\tshared\toutcome\t"
           "storage_desc(bank:offset:w)\tcbuf_desc(index:size)\ttex_desc\timg_desc\tuses_global\t"
           "nvn_used\tlayout_unsupported\tlayout_issues\tspv_words\n";
    out.flush();

    NvnShaderPools pools;
    const Shader::HostTranslateInfo host_info = MakeReferenceHostInfo();
    const Shader::Profile profile = MakeReferenceProfile();
    u64 compute_count = 0;
    for (u64 index = 0; index < offsets.size(); ++index) {
        StoredShader shader;
        if (!ReadStored(store, offsets[index], shader) ||
            shader.stage_byte != static_cast<u32>(ShaderStage::Compute) || skip.contains(index)) {
            continue;
        }
        ++compute_count;
        std::string error;
        const auto control = ParseControl(shader.control, error);
        if (!control) {
            out << fmt::format("{}\t{}\t\t\t\t\t\t\t\tcontrol: {}\n", index, Clean(shader.source),
                               Clean(error));
            continue;
        }
        const ComputeDriverLayout& layout = control->compute;
        std::string row = fmt::format(
            "{}\t{}\t{}x{}x{}\t{}\t{}\t{}\t{}\t{}\t{}\t", index, Clean(shader.source),
            control->thread_group[0], control->thread_group[1], control->thread_group[2],
            layout.push_constant_words, layout.uniform_count, layout.storage_count,
            layout.texture_count, layout.image_count, control->shared_memory_bytes);
        out << fmt::format("B\t{}\n", index);
        out.flush();
        const auto lowered =
            LowerNvnProgram(NvnProgramBinary{ShaderStage::Compute, shader.control, shader.code});
        if (!lowered->error.empty()) {
            out << row << "lower: " << Clean(lowered->error) << "\n";
            continue;
        }
        NvnEnvironment env{lowered, NvnEnvironmentOptions{}};
        Shader::IR::Program program;
        try {
            program = TranslateNvnStage(pools, env, host_info);
        } catch (const std::exception& e) {
            pools.ReleaseContents();
            out << row << "translate: " << Clean(e.what()) << "\n";
            continue;
        }
        const Shader::Info& info = program.info;
        std::string storage;
        for (const auto& desc : info.storage_buffers_descriptors) {
            storage += fmt::format("{}:{:x}:{}{},", desc.cbuf_index, desc.cbuf_offset,
                                   desc.is_written ? 'w' : 'r',
                                   desc.count > 1 ? fmt::format("x{}", desc.count) : "");
        }
        std::string cbufs;
        for (const auto& desc : info.constant_buffer_descriptors) {
            cbufs += fmt::format("{}:{:x},", desc.index, info.constant_buffer_used_sizes[desc.index]);
        }
        const StageInfo stage_info{ShaderStage::Compute, info, &layout};
        const auto descriptors = BuildDescriptorLayout(std::span<const StageInfo>{&stage_info, 1});
        std::set<std::string> issues;
        for (const auto& binding : descriptors.bindings) {
            if (!binding.supported) {
                issues.insert(Clean(binding.issue));
            }
        }
        std::string issue_text;
        for (const auto& issue : issues) {
            issue_text += issue + '|';
        }
        size_t spv_words = 0;
        std::string outcome = "ok";
        try {
            Shader::RuntimeInfo runtime_info;
            Shader::Backend::Bindings bindings;
            spv_words =
                Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings).size();
        } catch (const std::exception& e) {
            outcome = std::string{"emit: "} + Clean(e.what());
        }
        out << row << outcome << '\t' << storage << '\t' << cbufs << '\t'
            << info.texture_descriptors.size() << '\t' << info.image_descriptors.size() << '\t'
            << (info.uses_global_memory ? 1 : 0) << '\t' << info.nvn_buffer_used.count() << '\t'
            << descriptors.unsupported_count << '\t' << issue_text << '\t' << spv_words << '\n';
        out.flush();
        pools.ReleaseContents();
    }
    out << fmt::format("# compute shaders: {}\n", compute_count);
    out.flush();
    CHECK(out.good());
    CHECK(compute_count > 0);
}
