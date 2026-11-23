// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Offline coverage of the NVN2 shader mapping against a game's real shaders. Hidden tests:
//   tests "[coverage]"           the supervisor (needs EDEN_NVN_SHADER_DATA, Windows)
//   tests "[coverage-worker]"    one crash-isolated worker, started by the supervisor
//
// EDEN_NVN_SHADER_DATA is a comma list of game data roots (e.g. base and updates\458752). The
// supervisor extracts and deduplicates every shader once, writes them to a store under
// C:\eg\results\nvn\work (never into the repository), and runs workers that lower, translate,
// emit and validate each one. Translation can abort the process (ASSERT/UNREACHABLE), so a worker
// writes a begin marker before each shader; when a worker dies, the supervisor records the
// in-flight shader as `crash` and restarts the worker past it. A worker making no progress for
// 120 s is terminated and the shader recorded as `timeout`.
//
// Knobs: EDEN_NVN_COVERAGE_JOBS (workers, default 6), EDEN_NVN_COVERAGE_LIMIT (evenly spaced
// sample size), EDEN_NVN_SPIRV_VAL (spirv-val path, "none" disables), EDEN_NVN_KEEP_SPV,
// EDEN_NVN_RAW_GLOBAL=1 (translate and emit with the NVN raw global memory path),
// EDEN_NVN_COVERAGE_DEFAULT_HOST=1 (the host info of the default, non-HLE path for a Switch 2 title),
// EDEN_NVN_COVERAGE_COOP=1 (a host with cooperative matrix support: HMMA programs are lowered and emitted),
// EDEN_NVN_COVERAGE_WORKDIR / EDEN_NVN_COVERAGE_PREBUILT=1 (run on a store.bin/store.idx already in a
// work dir on another drive; EDEN_NVN_SHADER_DATA then only labels the report).

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <exception>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <optional>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dbghelp.h>
#include <crtdbg.h>
#include <stdlib.h>
#endif

#include "common/fs/path_util.h"
#include "nvn_shader_containers.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/nvn/shader/nvn_descriptor_layout.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

namespace fs = std::filesystem;
using namespace VideoCommon::Nvn;

const fs::path ResultsRoot = "C:/eg/results/nvn";
// EDEN_NVN_COVERAGE_WORKDIR moves the store, results and logs (default: results root/work), e.g. to a
// drive with room for a 1 GB store; EDEN_NVN_COVERAGE_PREBUILT=1 then uses the store.bin/store.idx
// already in it (the offline scan's stores of games the extractor cannot read, e.g. Kirby Air Riders)
// instead of extracting `EDEN_NVN_SHADER_DATA`.
const fs::path WorkDir = [] {
    const char* const dir = std::getenv("EDEN_NVN_COVERAGE_WORKDIR");
    return dir != nullptr && *dir != '\0' ? fs::path{dir} : ResultsRoot / "work";
}();
constexpr char DefaultSpirvVal[] = "C:/eg/VulkanSDK/1.4.341.1/Bin/spirv-val.exe";
constexpr int WatchdogSeconds = 120;

std::string Env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

/// EDEN_NVN_COVERAGE_DEFAULT_HOST=1: translate with the host info the default (non-HLE) path uses for
/// a Switch 2 title: raw global memory, unbiased tracking and the table in a storage buffer.
bool DefaultPathHostRequested() {
    const std::string value = Env("EDEN_NVN_COVERAGE_DEFAULT_HOST");
    return !value.empty() && value != "0";
}

/// EDEN_NVN_COVERAGE_COOP=1: lower and emit as a host with cooperative matrix support does (the
/// native HMMA.16816.F16 programs of the tensor upscalers), else they count as rejected.
bool CooperativeMatrixRequested() {
    const std::string value = Env("EDEN_NVN_COVERAGE_COOP");
    return !value.empty() && value != "0";
}

/// EDEN_NVN_RAW_GLOBAL=1 (or the default-path host info) translates with the raw global memory path.
bool RawGlobalRequested() {
    const std::string raw = Env("EDEN_NVN_RAW_GLOBAL");
    return (!raw.empty() && raw != "0") || DefaultPathHostRequested();
}

std::vector<std::string> Split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(separator, begin);
        parts.push_back(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return parts;
}

std::string Sanitize(std::string text, size_t limit = 300) {
    for (char& c : text) {
        if (c == '\t' || c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    if (text.size() > limit) {
        text.resize(limit);
    }
    return text;
}

/// Output file of the report; EDEN_NVN_REPORT_SUFFIX keeps experiment runs apart from the
/// default `shader-coverage.md`.
fs::path ReportPath(const std::string& stem, const std::string& extension) {
    return ResultsRoot / (stem + Env("EDEN_NVN_REPORT_SUFFIX") + extension);
}
/// Replaces hexadecimal and decimal runs by '#', so reasons differing only in numbers group.
std::string Normalize(const std::string& text) {
    std::string out;
    size_t i = 0;
    const auto is_hex = [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    };
    while (i < text.size() && out.size() < 160) {
        if (text[i] == '0' && i + 2 < text.size() && (text[i + 1] == 'x' || text[i + 1] == 'X') &&
            is_hex(text[i + 2])) {
            i += 2;
            while (i < text.size() && is_hex(text[i])) {
                ++i;
            }
            out += '#';
        } else if (text[i] >= '0' && text[i] <= '9') {
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                ++i;
            }
            out += '#';
        } else {
            out += text[i++];
        }
    }
    return out;
}

std::string Hex(std::span<const u8> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const u8 byte : bytes) {
        out += digits[byte >> 4];
        out += digits[byte & 15];
    }
    return out;
}

// -- shader store -------------------------------------------------------------------------------

constexpr u32 StoreMagic = 0x3153564e; // "NVS1"

struct StoredShader {
    std::string key;
    std::string source;
    std::string kind;
    u32 stage_byte{};
    std::vector<u8> control;
    std::vector<u8> code;
};

/// False when either file could not be written in full.
bool WriteStore(const std::vector<Containers::ExtractedShader>& shaders, const fs::path& store,
                const fs::path& index) {
    std::ofstream out(store, std::ios::binary);
    std::vector<u64> offsets;
    offsets.reserve(shaders.size());
    u64 position = 0;
    const auto put = [&](const void* data, size_t size) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        position += size;
    };
    for (const auto& shader : shaders) {
        offsets.push_back(position);
        const u32 header[7] = {StoreMagic,
                               shader.stage_byte,
                               static_cast<u32>(shader.source.size()),
                               static_cast<u32>(shader.kind.size()),
                               static_cast<u32>(shader.key.size()),
                               static_cast<u32>(shader.control.size()),
                               static_cast<u32>(shader.code.size())};
        put(header, sizeof(header));
        put(shader.source.data(), shader.source.size());
        put(shader.kind.data(), shader.kind.size());
        put(shader.key.data(), shader.key.size());
        put(shader.control.data(), shader.control.size());
        put(shader.code.data(), shader.code.size());
    }
    out.close();
    if (!out) {
        return false;
    }
    std::ofstream idx(index, std::ios::binary);
    idx.write(reinterpret_cast<const char*>(offsets.data()),
              static_cast<std::streamsize>(offsets.size() * sizeof(u64)));
    idx.close();
    return static_cast<bool>(idx);
}

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

// -- result rows --------------------------------------------------------------------------------

/// How far a shader got: the report's "translated / emitted / valid" funnel.
enum Reached : u32 { ReachedNothing = 0, ReachedLowered, ReachedTranslated, ReachedEmitted, ReachedValid };

struct Row {
    u64 index{};
    std::string key;
    std::string source;
    std::string kind;
    std::string stage;
    u32 reached{ReachedNothing};
    std::string klass;  // failure class ("" = ok): control, capture, lower, translate, emit, ...
    std::string reason; // normalized
    std::string detail; // raw message
    u32 error_offset{};
    std::string error_words;
    u32 instructions{};
    u32 spv_words{};
    u32 bindings{};
    u32 layout_unsupported{};
    std::string layout_issues;
    std::string flags;
    u32 millis{};

    [[nodiscard]] bool Failed() const {
        return !klass.empty();
    }
};

constexpr size_t RowFields = 19;

std::string Serialize(const Row& row) {
    return fmt::format("R\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                       row.index, row.key, Sanitize(row.source, 400), row.kind, row.stage,
                       row.reached, row.klass, Sanitize(row.reason), Sanitize(row.detail),
                       row.error_offset, row.error_words, row.instructions, row.spv_words,
                       row.bindings, row.layout_unsupported, Sanitize(row.layout_issues, 600),
                       row.flags, row.millis);
}

bool Deserialize(const std::string& line, Row& row) {
    const auto f = Split(line, '\t');
    if (f.size() != RowFields || f[0] != "R") {
        return false;
    }
    const auto number = [](const std::string& text) {
        return static_cast<u64>(std::strtoull(text.c_str(), nullptr, 10));
    };
    row.index = number(f[1]);
    row.key = f[2];
    row.source = f[3];
    row.kind = f[4];
    row.stage = f[5];
    row.reached = static_cast<u32>(number(f[6]));
    row.klass = f[7];
    row.reason = f[8];
    row.detail = f[9];
    row.error_offset = static_cast<u32>(number(f[10]));
    row.error_words = f[11];
    row.instructions = static_cast<u32>(number(f[12]));
    row.spv_words = static_cast<u32>(number(f[13]));
    row.bindings = static_cast<u32>(number(f[14]));
    row.layout_unsupported = static_cast<u32>(number(f[15]));
    row.layout_issues = f[16];
    row.flags = f[17];
    row.millis = static_cast<u32>(number(f[18]));
    return true;
}

// -- processes ----------------------------------------------------------------------------------

#ifdef _WIN32
std::wstring Widen(const std::string& text) {
    return std::wstring(text.begin(), text.end());
}

/// Runs a command line with stdout+stderr redirected to `output`. Returns the exit code, or -1
/// if the process could not be started or did not finish within `timeout_ms`.
int RunToFile(const std::string& command_line, const fs::path& output, DWORD timeout_ms) {
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    HANDLE file = CreateFileW(output.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ, &attributes,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return -1;
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = file;
    startup.hStdError = file;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    std::wstring mutable_line = Widen(command_line);
    const BOOL started = CreateProcessW(nullptr, mutable_line.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(file);
    if (!started) {
        return -1;
    }
    int result = -1;
    if (WaitForSingleObject(process.hProcess, timeout_ms) == WAIT_OBJECT_0) {
        DWORD code = 0;
        GetExitCodeProcess(process.hProcess, &code);
        result = static_cast<int>(code);
    } else {
        TerminateProcess(process.hProcess, 1); // our own child
        WaitForSingleObject(process.hProcess, 5000);
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return result;
}
#else
int RunToFile(const std::string& command_line, const fs::path& output, unsigned) {
    return std::system((command_line + " > \"" + output.string() + "\" 2>&1").c_str());
}
#endif

std::string FirstLine(const fs::path& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

// -- worker -------------------------------------------------------------------------------------

struct WorkerConfig {
    std::string spirv_val;
    bool keep_spv{};
};

// Where the worker is, for the message it leaves behind when the process aborts.
const char* g_phase = "idle";
u64 g_index = 0;

#ifdef _WIN32
// The top frames of the aborting thread, named from the PDB next to the executable (dbghelp is
// loaded at run time, nothing is linked): a fatal assert only logs through the asynchronous logger,
// which the abort has already stopped, so the stack is the only trace of where it fired.
void PrintBacktrace() {
    using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
    using SymInitializeFn = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
    using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
    const HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
    if (dbghelp == nullptr) {
        return;
    }
    const auto set_options = reinterpret_cast<SymSetOptionsFn>(GetProcAddress(dbghelp, "SymSetOptions"));
    const auto initialize = reinterpret_cast<SymInitializeFn>(GetProcAddress(dbghelp, "SymInitialize"));
    const auto from_address = reinterpret_cast<SymFromAddrFn>(GetProcAddress(dbghelp, "SymFromAddr"));
    if (set_options == nullptr || initialize == nullptr || from_address == nullptr) {
        return;
    }
    const HANDLE process = GetCurrentProcess();
    set_options(0x00000002 /*SYMOPT_UNDNAME*/ | 0x00000004 /*SYMOPT_DEFERRED_LOADS*/);
    if (!initialize(process, nullptr, TRUE)) {
        return;
    }
    void* frames[48];
    const USHORT count = CaptureStackBackTrace(0, 48, frames, nullptr);
    alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + 256];
    for (USHORT i = 0; i < count; ++i) {
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage);
        std::memset(symbol, 0, sizeof(storage));
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        DWORD64 displacement = 0;
        const DWORD64 address = reinterpret_cast<DWORD64>(frames[i]);
        if (from_address(process, address, &displacement, symbol)) {
            std::fprintf(stderr, "[worker]   #%u %s+%#llx\n", static_cast<unsigned>(i), symbol->Name,
                         static_cast<unsigned long long>(displacement));
        } else {
            std::fprintf(stderr, "[worker]   #%u %#llx\n", static_cast<unsigned>(i),
                         static_cast<unsigned long long>(address));
        }
    }
}
#endif

void OnAbortSignal(int) {
    std::fprintf(stderr, "[worker] abort() in phase '%s' (shader %llu)\n", g_phase,
                 static_cast<unsigned long long>(g_index));
#ifdef _WIN32
    PrintBacktrace();
#endif
}

void OnTerminate() {
    std::string what = "no active exception";
    if (const auto current = std::current_exception()) {
        try {
            std::rethrow_exception(current);
        } catch (const std::exception& e) {
            what = std::string{"uncaught exception: "} + e.what();
        } catch (...) {
            what = "uncaught non-standard exception";
        }
    }
    std::fprintf(stderr, "[worker] terminate: %s in phase '%s' (shader %llu)\n", what.c_str(),
                 g_phase, static_cast<unsigned long long>(g_index));
    std::abort();
}
void SetStage(Row& row, std::optional<ShaderStage> stage) {
    row.stage = stage ? std::string{StageName(*stage)} : "?";
}

Row ProcessShader(u64 index, const StoredShader& shader, const WorkerConfig& config,
                  NvnShaderPools& pools, std::ofstream& compute_out) {
    g_phase = "lower";
    const auto start = std::chrono::steady_clock::now();
    Row row;
    row.index = index;
    row.key = shader.key;
    row.source = shader.source;
    row.kind = shader.kind;
    const auto finish = [&] {
        row.millis = static_cast<u32>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count());
        return row;
    };
    const auto fail = [&](std::string klass, const std::string& message) {
        row.klass = std::move(klass);
        row.detail = message;
        row.reason = row.klass + ":" + Normalize(message);
        return finish();
    };

    std::string control_error;
    const auto control = ParseControl(shader.control, control_error);
    if (!control) {
        row.stage = "?";
        return fail("control", control_error);
    }
    std::optional<ShaderStage> stage;
    const bool is_compute = control->stage_byte == static_cast<u32>(ShaderStage::Compute);
    if (is_compute) {
        stage = ShaderStage::Compute;
    } else {
        stage = StageFromSph(shader.code);
        if (stage && static_cast<u32>(*stage) != control->stage_byte) {
            row.flags += "stage-mismatch ";
        }
    }
    SetStage(row, stage);
    if (!stage) {
        return fail("stage", "invalid SPH shader type");
    }

    const NvnProgramBinary binary{*stage, shader.control, shader.code};
    const auto lowered = LowerNvnProgram(binary, CooperativeMatrixRequested()
                                                     ? VideoCommon::NativeTensorLowering::Full
                                                     : VideoCommon::NativeTensorLowering::None);
    if (!lowered->capture_complete && lowered->error.empty()) {
        row.flags += "capture-incomplete ";
    }
    if (!lowered->error.empty()) {
        row.error_offset = lowered->error_offset;
        row.error_words = fmt::format("{:08x} {:08x} {:08x} {:08x}", lowered->error_words[0],
                                      lowered->error_words[1], lowered->error_words[2],
                                      lowered->error_words[3]);
        const size_t colon = lowered->error.find(':');
        return fail(lowered->error.substr(0, colon), lowered->error.substr(colon + 2));
    }
    row.reached = ReachedLowered;
    row.instructions = lowered->instruction_count;

    g_phase = "translate";
    NvnEnvironment env{lowered, NvnEnvironmentOptions{}};
    Shader::HostTranslateInfo host_info = MakeReferenceHostInfo();
    Shader::Profile profile = MakeReferenceProfile();
    if (CooperativeMatrixRequested()) {
        // A host with VK_NV_cooperative_matrix (the m16n8k16 half shape), 32-lane subgroups and the
        // shared memory of a desktop GPU, as vk_pipeline_cache.cpp / nvn_pipeline_state.cpp build it.
        profile.supported_spirv = 0x00010500;
        profile.supported_subgroup_stages = 1U << static_cast<u32>(Shader::Stage::Compute);
        profile.max_subgroup_size = 32;
    }
    if (RawGlobalRequested()) {
        // The NVN replay's raw global memory path (needs descriptor aliasing, as on the device).
        host_info.nvn_raw_global_memory = true;
        profile.support_raw_global_memory = true;
        profile.support_descriptor_aliasing = true;
    }
    if (DefaultPathHostRequested()) {
        // What the default path's PipelineCache::TranslateHostInfo sets for a Switch 2 title.
        host_info.nvn_raw_global_unbiased_tracking = true;
        host_info.nvn_raw_global_table_ssbo = true;
    }

    struct PoolGuard {
        NvnShaderPools& pools;
        ~PoolGuard() {
            pools.ReleaseContents();
        }
    } guard{pools};

    Shader::IR::Program program;
    try {
        program = TranslateNvnStage(pools, env, host_info);
    } catch (const Shader::NotImplementedException& e) {
        return fail("translate", std::string{"NotImplemented: "} + e.what());
    } catch (const Shader::LogicError& e) {
        return fail("translate", std::string{"Logic: "} + e.what());
    } catch (const Shader::RuntimeError& e) {
        return fail("translate", std::string{"Runtime: "} + e.what());
    } catch (const Shader::InvalidArgument& e) {
        return fail("translate", std::string{"InvalidArgument: "} + e.what());
    } catch (const std::exception& e) {
        return fail("translate", std::string{"Exception: "} + e.what());
    }
    row.reached = ReachedTranslated;
    if (env.HandleKeyOverflow()) {
        row.flags += "handle-key-overflow ";
    }

    // The descriptor layout is its own outcome: unsupported bindings do not fail the shader.
    const StageInfo stage_info{*stage, program.info,
                               is_compute ? &lowered->control.compute : nullptr};
    g_phase = "layout";
    const auto layout = BuildDescriptorLayout(std::span<const StageInfo>{&stage_info, 1});
    row.bindings = layout.binding_count;
    row.layout_unsupported = layout.unsupported_count;
    {
        std::set<std::string> issues;
        for (const auto& binding : layout.bindings) {
            if (!binding.supported) {
                // The unresolved handle word is part of the reason (c<bank>[offset]): the report
                // names the bank and offset a layout does not know.
                issues.insert(binding.issue);
            }
        }
        for (const auto& issue : issues) {
            row.layout_issues += issue + '|';
        }
    }

    if (is_compute && compute_out) {
        const auto& layout_info = lowered->control.compute;
        std::string storage;
        for (const auto& desc : program.info.storage_buffers_descriptors) {
            storage += fmt::format("{:x},", desc.cbuf_offset);
        }
        std::string cbufs;
        for (const auto& desc : program.info.constant_buffer_descriptors) {
            cbufs += fmt::format("{}:{:x},", desc.index,
                                 program.info.constant_buffer_used_sizes[desc.index]);
        }
        compute_out << fmt::format(
            "{}\t{}\t{}\t{}x{}x{}\t{}\t{}\t{}\t{:x}\t{:x}\t{:x}\t{:x}\t{:x}\t{:x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            index, shader.key, Sanitize(shader.source, 200), control->thread_group[0],
            control->thread_group[1], control->thread_group[2], control->register_count,
            control->inst_bytes, control->code_bytes, layout_info.uniform_offset,
            layout_info.storage_offset, layout_info.texture_offset, layout_info.image_offset,
            layout_info.end_offset, layout_info.declared_total, layout_info.total_matches ? 1 : 0,
            storage, cbufs, program.info.uses_local_memory ? 1 : 0,
            program.info.storage_buffers_descriptors.empty() ? 0 : 1,
            program.info.uses_workgroup_id ? 1 : 0,
            Hex(shader.control));
        compute_out.flush();
    }

    g_phase = "emit";
    Shader::RuntimeInfo runtime_info;
    runtime_info.previous_stage_stores.mask.set();
    if (*stage == ShaderStage::Geometry) {
        runtime_info.input_topology = Shader::InputTopology::Triangles;
    }
    std::vector<u32> spirv;
    try {
        Shader::Maxwell::ConvertLegacyToGeneric(program, runtime_info);
        Shader::Backend::Bindings bindings;
        spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
        if (bindings.unified != layout.binding_count) {
            row.flags += "binding-count-mismatch ";
        }
    } catch (const Shader::NotImplementedException& e) {
        return fail("emit", std::string{"NotImplemented: "} + e.what());
    } catch (const Shader::LogicError& e) {
        return fail("emit", std::string{"Logic: "} + e.what());
    } catch (const Shader::RuntimeError& e) {
        return fail("emit", std::string{"Runtime: "} + e.what());
    } catch (const std::exception& e) {
        return fail("emit", std::string{"Exception: "} + e.what());
    }
    row.reached = ReachedEmitted;
    row.spv_words = static_cast<u32>(spirv.size());

    if (config.spirv_val.empty()) {
        row.reached = ReachedValid; // validation unavailable: counted as emitted-and-unchecked
        row.flags += "unvalidated ";
        return finish();
    }
    const fs::path spv_path = WorkDir / fmt::format("s-{}.spv", row.index);
    const fs::path log_path = WorkDir / fmt::format("s-{}.log", row.index);
    {
        std::ofstream out(spv_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(spirv.data()),
                  static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
    }
    g_phase = "spirv-val";
    // The raw global memory profile reads the driver bank through 32-bit element arrays.
    const char* const layout_flag =
        RawGlobalRequested() ? "--uniform-buffer-standard-layout" : "";
    const int code = RunToFile(fmt::format("\"{}\" --target-env vulkan1.3 {} \"{}\"",
                                           config.spirv_val, layout_flag, spv_path.string()),
                               log_path, 60000);
    const std::string message = FirstLine(log_path);
    std::error_code ignored;
    if (!config.keep_spv) {
        fs::remove(spv_path, ignored);
    }
    fs::remove(log_path, ignored);
    if (code != 0) {
        return fail("spirv-val", message.empty() ? fmt::format("exit code {}", code) : message);
    }
    row.reached = ReachedValid;
    return finish();
}

std::set<u64> ReadHandledIndices(const fs::path& result_file) {
    std::set<u64> handled;
    std::ifstream in(result_file);
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() > 2 && (line[0] == 'R' || line[0] == 'C') && line[1] == '\t') {
            handled.insert(std::strtoull(line.c_str() + 2, nullptr, 10));
        }
    }
    return handled;
}

} // namespace

TEST_CASE("NVN shader coverage worker", "[.coverage-worker]") {
    const std::string worker_text = Env("EDEN_NVN_WORKER");
    const std::string workers_text = Env("EDEN_NVN_WORKERS");
    if (worker_text.empty() || workers_text.empty()) {
        SKIP("started by the coverage supervisor only");
    }
#ifdef _WIN32
    // A fatal assert aborts: keep it from opening dialogs, the supervisor reads the exit code.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    // The logger writes a file under the eden log directory: keep it out of the user's.
    Common::FS::SetEdenPath(Common::FS::EdenPath::LogDir, WorkDir / "eden-log");
    fs::create_directories(WorkDir / "eden-log");
    Common::Log::Initialize();
    Common::Log::SetGlobalFilter(Common::Log::Filter{Common::Log::Level::Error});
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();
    std::setvbuf(stdout, nullptr, _IONBF, 0); // abort() does not flush stdio: keep the message
    std::signal(SIGABRT, OnAbortSignal);
    std::set_terminate(OnTerminate);
#endif
    const u64 worker = std::strtoull(worker_text.c_str(), nullptr, 10);
    const u64 workers = std::strtoull(workers_text.c_str(), nullptr, 10);
    REQUIRE(workers > 0);
    WorkerConfig config;
    config.spirv_val = Env("EDEN_NVN_SPIRV_VAL_RESOLVED");
    config.keep_spv = !Env("EDEN_NVN_KEEP_SPV").empty();

    std::ifstream index_file(WorkDir / "store.idx", std::ios::binary | std::ios::ate);
    REQUIRE(index_file.is_open());
    std::vector<u64> offsets(static_cast<size_t>(index_file.tellg()) / sizeof(u64));
    index_file.seekg(0);
    index_file.read(reinterpret_cast<char*>(offsets.data()),
                    static_cast<std::streamsize>(offsets.size() * sizeof(u64)));
    std::ifstream store(WorkDir / "store.bin", std::ios::binary);
    REQUIRE(store.is_open());

    const fs::path result_path = WorkDir / fmt::format("result-{}.tsv", worker);
    const std::set<u64> handled = ReadHandledIndices(result_path);
    std::ofstream results(result_path, std::ios::app);
    std::ofstream compute_out(WorkDir / fmt::format("compute-{}.tsv", worker), std::ios::app);
    NvnShaderPools pools;
    for (u64 index = worker; index < offsets.size(); index += workers) {
        if (handled.contains(index)) {
            continue;
        }
        StoredShader shader;
        if (!ReadStored(store, offsets[index], shader)) {
            results << "C\t" << index << "\tcontainer\tunreadable store record\n";
            results.flush();
            continue;
        }
        g_index = index;
        g_phase = "start";
        results << "B\t" << index << '\n';
        results.flush();
        const Row row = ProcessShader(index, shader, config, pools, compute_out);
        results << Serialize(row) << '\n';
        results.flush();
    }
    results << "F\n";
    results.flush();
}

// -- supervisor ---------------------------------------------------------------------------------

namespace {

#ifdef _WIN32
struct WorkerProcess {
    u64 id{};
    HANDLE process{};
    HANDLE thread{};
    bool finished{};
    bool gave_up{};
    u32 restarts{};
    u32 fruitless_starts{};
    std::uintmax_t last_size{};
    std::chrono::steady_clock::time_point last_progress;
};

std::string ExecutablePath() {
    std::array<wchar_t, 4096> path{};
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    std::wstring wide(path.data(), length);
    return std::string(wide.begin(), wide.end());
}

bool Spawn(WorkerProcess& worker, u64 workers, const std::string& spirv_val) {
    SetEnvironmentVariableA("EDEN_NVN_WORKER", std::to_string(worker.id).c_str());
    SetEnvironmentVariableA("EDEN_NVN_WORKERS", std::to_string(workers).c_str());
    SetEnvironmentVariableA("EDEN_NVN_SPIRV_VAL_RESOLVED", spirv_val.c_str());
    // The worker's stdout/stderr (its log, including a fatal assert's message) go to a file.
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    const fs::path log_path = WorkDir / fmt::format("worker-{}.log", worker.id);
    HANDLE log = CreateFileW(log_path.wstring().c_str(), FILE_APPEND_DATA,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes, OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = log;
    startup.hStdError = log;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION info{};
    std::wstring line = Widen("\"" + ExecutablePath() + "\" \"NVN shader coverage worker\"");
    const BOOL started = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info);
    if (log != INVALID_HANDLE_VALUE) {
        CloseHandle(log);
    }
    if (!started) {
        return false;
    }
    worker.process = info.hProcess;
    worker.thread = info.hThread;
    worker.last_progress = std::chrono::steady_clock::now();
    return true;
}

/// Last begin marker without a result or a crash line, if any.
std::optional<u64> InFlight(const fs::path& result_file) {
    std::ifstream in(result_file);
    std::string line;
    std::optional<u64> in_flight;
    while (std::getline(in, line)) {
        if (line.size() > 2 && line[1] == '\t') {
            const u64 value = std::strtoull(line.c_str() + 2, nullptr, 10);
            if (line[0] == 'B') {
                in_flight = value;
            } else if ((line[0] == 'R' || line[0] == 'C') && in_flight == value) {
                in_flight.reset();
            }
        }
    }
    return in_flight;
}

/// The most telling recent line of a worker's log (a fatal assert if there is one).
std::string LastLogLine(const fs::path& log) {
    std::ifstream in(log, std::ios::binary | std::ios::ate);
    if (!in) {
        return {};
    }
    const auto size = static_cast<std::streamoff>(in.tellg());
    const std::streamoff window = std::min<std::streamoff>(size, 8192);
    std::string text(static_cast<size_t>(window), '\0');
    in.seekg(size - window);
    in.read(text.data(), window);
    std::string clean;
    for (size_t i = 0; i < text.size(); ++i) { // drop ANSI colour sequences
        if (text[i] == '\x1b') {
            while (i < text.size() && text[i] != 'm') {
                ++i;
            }
            continue;
        }
        clean += text[i];
    }
    std::string best;
    for (const auto& line : Split(clean, '\n')) {
        if (line.find_first_not_of(" \r") == std::string::npos) {
            continue;
        }
        if (best.empty() || line.find("[worker]") != std::string::npos ||
            (best.find("[worker]") == std::string::npos &&
             (line.find("ssert") != std::string::npos ||
              line.find("Critical") != std::string::npos))) {
            best = line;
        }
    }
    return Sanitize(best, 400);
}

bool HasFinished(const fs::path& result_file) {
    std::ifstream in(result_file);
    std::string line;
    bool finished = false;
    while (std::getline(in, line)) {
        finished = line == "F";
    }
    return finished;
}
#endif

struct Totals {
    u64 total{};
    u64 lowered{};
    u64 translated{};
    u64 emitted{};
    u64 valid{};
    u64 failed{};
    u64 layout_supported{};
};

void Tally(Totals& totals, const Row& row) {
    ++totals.total;
    totals.lowered += row.reached >= ReachedLowered;
    totals.translated += row.reached >= ReachedTranslated;
    totals.emitted += row.reached >= ReachedEmitted;
    totals.valid += row.reached >= ReachedValid && !row.Failed();
    totals.failed += row.Failed();
    totals.layout_supported += row.reached >= ReachedTranslated && row.layout_unsupported == 0;
}

std::string TopSource(const std::string& source) {
    return source.substr(0, source.find_first_of(">#"));
}

std::string Percent(u64 part, u64 whole) {
    return whole == 0 ? "n/a" : fmt::format("{:.1f}%", 100.0 * static_cast<double>(part) / static_cast<double>(whole));
}

/// Summary of the compute control sections of the translated compute shaders (their raw rows are
/// in shader-compute-controls.tsv). Fields: see the header of that file.
std::string ComputeFindings(const fs::path& compute_tsv) {
    std::ifstream in(compute_tsv);
    std::string line;
    std::getline(in, line); // header
    u64 shaders = 0;
    u64 matches = 0;
    u64 local = 0;
    u64 global = 0;
    u64 shared_nonzero = 0;
    u32 shared_max = 0;
    std::map<std::string, u64> groups;
    std::map<std::string, u64> layouts;
    while (std::getline(in, line)) {
        const auto f = Split(line, '\t');
        if (f.size() < 20 || f[19].size() < 0x68 * 2) {
            continue;
        }
        ++shaders;
        matches += f[13] == "1";
        local += f[16] == "1";
        global += f[17] == "1";
        ++groups[f[3]];
        ++layouts[fmt::format("uniform@{} storage@{} texture@{} image@{} end@{}", f[7], f[8],
                              f[9], f[10], f[11])];
        u32 shared = 0;
        for (size_t i = 0; i < 4; ++i) { // control +0x64, little endian
            shared |= static_cast<u32>(std::strtoul(f[19].substr((0x64 + i) * 2, 2).c_str(), nullptr, 16))
                      << (8 * i);
        }
        shared_nonzero += shared != 0;
        shared_max = std::max(shared_max, shared);
    }
    std::string text = "\n## Compute control-section findings (translated compute shaders)\n\n";
    text += fmt::format(
        "- Compute shaders that reached translation: {}. Driver bank layout derived from the "
        "counts (+0xa9 push words, +0xb1 uniform, +0xb2 storage, +0xb3 texture, +0xb6 image) ends "
        "exactly at the declared total (u16 +0xba) for {} of them.\n",
        shaders, matches);
    text += fmt::format(
        "- Shared memory: control +0x64 is non-zero for {} of {} (largest {} bytes). Translation "
        "uses it as the shared memory size.\n",
        shared_nonzero, shaders, shared_max);
    text += fmt::format(
        "- Local memory: {} compute shaders use local memory (the recompiler reports "
        "`uses_local_memory`); shaders with storage buffer descriptors: {}.\n",
        local, global);
    text += "- Thread groups: ";
    for (const auto& [group, count] : groups) {
        text += fmt::format("{} x{}; ", group, count);
    }
    text += "\n- Driver bank table layouts (distinct): " + std::to_string(layouts.size()) + "\n";
    return text;
}
void WriteReport(const std::vector<Row>& rows, const Containers::ScanOutput& scan,
                 const std::vector<std::string>& roots, const std::string& spirv_val,
                 size_t extracted, size_t sampled) {
    fs::create_directories(ResultsRoot);
    // Per-shader TSV.
    {
        std::ofstream tsv(ReportPath("shader-coverage", ".tsv"));
        tsv << "index\tkey\tsource\tkind\tstage\treached\tclass\treason\tdetail\terror_offset\t"
               "error_words\tinstructions\tspv_words\tbindings\tlayout_unsupported\tlayout_issues\t"
               "flags\tmillis\n";
        for (const Row& row : rows) {
            tsv << row.index << '\t' << row.key << '\t' << Sanitize(row.source, 400) << '\t'
                << row.kind << '\t' << row.stage << '\t' << row.reached << '\t' << row.klass
                << '\t' << Sanitize(row.reason) << '\t' << Sanitize(row.detail) << '\t'
                << row.error_offset << '\t' << row.error_words << '\t' << row.instructions << '\t'
                << row.spv_words << '\t' << row.bindings << '\t' << row.layout_unsupported << '\t'
                << Sanitize(row.layout_issues, 600) << '\t' << row.flags << '\t' << row.millis
                << '\n';
        }
    }

    Totals totals;
    std::map<std::string, Totals> by_stage;
    std::map<std::string, Totals> by_source;
    std::map<std::string, Totals> by_kind;
    struct Reason {
        u64 count{};
        std::vector<std::string> examples;
    };
    std::map<std::string, Reason> failures;
    std::map<std::string, Reason> layout_reasons;
    u64 stage_mismatch = 0;
    u64 incomplete_capture = 0;
    u64 binding_mismatch = 0;
    u64 unvalidated = 0;
    for (const Row& row : rows) {
        Tally(totals, row);
        Tally(by_stage[row.stage], row);
        Tally(by_source[TopSource(row.source)], row);
        Tally(by_kind[row.kind], row);
        stage_mismatch += row.flags.find("stage-mismatch") != std::string::npos;
        incomplete_capture += row.flags.find("capture-incomplete") != std::string::npos;
        binding_mismatch += row.flags.find("binding-count-mismatch") != std::string::npos;
        unvalidated += row.flags.find("unvalidated") != std::string::npos;
        const auto example = [&](const Row& r) {
            std::string text = fmt::format("{}#{}", Sanitize(r.source, 160), r.key);
            if (!r.error_words.empty() && r.error_words != "00000000 00000000 00000000 00000000") {
                text += fmt::format(" +{:#x} [{}]", r.error_offset, r.error_words);
            }
            return text;
        };
        if (row.Failed()) {
            auto& entry = failures[row.reason];
            ++entry.count;
            if (entry.examples.size() < 3) {
                entry.examples.push_back(example(row));
            }
        }
        if (row.reached >= ReachedTranslated && !row.layout_issues.empty()) {
            for (const auto& issue : Split(row.layout_issues, '|')) {
                if (issue.empty()) {
                    continue;
                }
                auto& entry = layout_reasons["layout:" + issue];
                ++entry.count;
                if (entry.examples.size() < 3) {
                    entry.examples.push_back(example(row));
                }
            }
        }
    }
    const auto ranked = [](const std::map<std::string, Reason>& map) {
        std::vector<std::pair<std::string, Reason>> items(map.begin(), map.end());
        std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            return a.second.count > b.second.count;
        });
        return items;
    };

    std::ofstream md(ReportPath("shader-coverage", ".md"));
    const std::time_t now = std::time(nullptr);
    char stamp[32]{};
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    md << "# NVN2 shader coverage\n\n";
    md << "Generated " << stamp << " by `tests [coverage]` (src/tests/nvn/shaders_coverage.cpp).\n\n";
    md << "- Roots: ";
    for (const auto& root : roots) {
        md << '`' << root << "` ";
    }
    md << "\n- Pipeline: `LowerNvnProgram` -> `NvnEnvironment` + `TranslateNvnStage` -> "
          "`ConvertLegacyToGeneric` + `EmitSPIRV` (reference profile, default RuntimeInfo; compute "
          "shared memory from control +0x64, local memory 0) -> "
       << (spirv_val.empty() ? "no spirv-val (unavailable)" : "`spirv-val --target-env vulkan1.3`")
       << "\n";
    md << "- Unique shaders extracted: " << extracted << (sampled != extracted ? fmt::format(" (sampled {})", sampled) : "")
       << ", duplicates dropped by key: " << scan.duplicates << ", container failures: "
       << scan.failures.size() << "\n\n";

    md << "## Totals\n\n| metric | count | of total |\n|---|---:|---:|\n";
    const auto line = [&](const char* name, u64 value) {
        md << "| " << name << " | " << value << " | " << Percent(value, totals.total) << " |\n";
    };
    line("shaders", totals.total);
    line("lowered", totals.lowered);
    line("translated to IR", totals.translated);
    line("emitted SPIR-V", totals.emitted);
    line(spirv_val.empty() ? "valid (unchecked)" : "valid (spirv-val)", totals.valid);
    line("failed", totals.failed);
    line("descriptor layout fully supported (of translated)", totals.layout_supported);
    md << "\nFlags: stage byte != SPH stage: " << stage_mismatch
       << "; capture did not complete within the instruction bytes (and lowering still passed): "
       << incomplete_capture << "; EmitSPIRV binding count != layout binding count: "
       << binding_mismatch << "; unvalidated: " << unvalidated << ".\n\n";

    const auto table = [&](const char* title, const std::map<std::string, Totals>& map,
                           size_t limit) {
        md << "## " << title << "\n\n| name | total | lowered | translated | emitted | valid | failed |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n";
        std::vector<std::pair<std::string, Totals>> items(map.begin(), map.end());
        std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            return a.second.total > b.second.total;
        });
        size_t shown = 0;
        for (const auto& [name, t] : items) {
            if (shown++ == limit) {
                md << "| ... " << (items.size() - limit) << " more | | | | | | |\n";
                break;
            }
            md << "| " << Sanitize(name, 120) << " | " << t.total << " | " << t.lowered << " | "
               << t.translated << " | " << t.emitted << " | " << t.valid << " | " << t.failed
               << " |\n";
        }
        md << '\n';
    };
    table("By stage", by_stage, 16);
    table("By container kind", by_kind, 16);
    table("By source file (top 40 by shader count)", by_source, 40);

    md << "## Failure reasons (ranked)\n\n| # | count | share of failures | reason | examples |\n"
          "|---:|---:|---:|---|---|\n";
    size_t rank = 0;
    for (const auto& [reason, entry] : ranked(failures)) {
        if (++rank > 80) {
            md << "| | | | ... " << (failures.size() - 80) << " more reasons in the TSV | |\n";
            break;
        }
        md << "| " << rank << " | " << entry.count << " | " << Percent(entry.count, totals.failed)
           << " | `" << Sanitize(reason, 200) << "` | ";
        for (size_t i = 0; i < entry.examples.size(); ++i) {
            md << (i ? "<br>" : "") << '`' << entry.examples[i] << '`';
        }
        md << " |\n";
    }
    md << "\n## Descriptor layout reasons (ranked; shaders still count as translated)\n\n"
          "| # | shaders | reason | examples |\n|---:|---:|---|---|\n";
    rank = 0;
    for (const auto& [reason, entry] : ranked(layout_reasons)) {
        if (++rank > 40) {
            break;
        }
        md << "| " << rank << " | " << entry.count << " | `" << Sanitize(reason, 200) << "` | ";
        for (size_t i = 0; i < entry.examples.size(); ++i) {
            md << (i ? "<br>" : "") << '`' << entry.examples[i] << '`';
        }
        md << " |\n";
    }
    md << ComputeFindings(ReportPath("shader-compute-controls", ".tsv"));
    md << "\n## Container failures (per container, not per shader)\n\n| count | reason | first source |\n"
          "|---:|---|---|\n";
    {
        std::map<std::string, std::pair<u64, std::string>> container;
        for (const auto& failure : scan.failures) {
            auto& entry = container[Normalize(failure.reason)];
            if (entry.first++ == 0) {
                entry.second = Sanitize(failure.source, 200);
            }
        }
        std::vector<std::pair<std::string, std::pair<u64, std::string>>> items(container.begin(),
                                                                               container.end());
        std::stable_sort(items.begin(), items.end(),
                         [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        size_t shown = 0;
        for (const auto& [reason, entry] : items) {
            if (shown++ == 40) {
                break;
            }
            md << "| " << entry.first << " | `" << Sanitize(reason, 200) << "` | `" << entry.second
               << "` |\n";
        }
    }
    md << "\nPer-shader rows: `" << ReportPath("shader-coverage", ".tsv").filename().string()
       << "`; compute control sections: `"
       << ReportPath("shader-compute-controls", ".tsv").filename().string() << "`.\n";
}

} // namespace

// Not tagged [nvn]: `tests "[nvn]"` must never run the coverage tool (or clear its work dir).
TEST_CASE("NVN shader coverage", "[.coverage]") {
#ifndef _WIN32
    SKIP("the coverage supervisor needs Windows process APIs");
#else
    const std::string data = Env("EDEN_NVN_SHADER_DATA");
    if (data.empty()) {
        SKIP("EDEN_NVN_SHADER_DATA is not set");
    }
    std::vector<std::string> roots;
    for (const auto& root : Split(data, ',')) {
        if (!root.empty()) {
            roots.push_back(root);
        }
    }
    const bool prebuilt = [] {
        const std::string value = Env("EDEN_NVN_COVERAGE_PREBUILT");
        return !value.empty() && value != "0";
    }();
    for (const auto& root : roots) {
        if (!prebuilt && !fs::exists(root)) {
            SKIP("shader data root missing: " + root);
        }
    }
    REQUIRE(!roots.empty());

    // Output lives under the results tree only (the default work dir, or an F: results dir).
    const std::string work_dir = fs::weakly_canonical(WorkDir).generic_string();
    // The root itself or a directory below it, not a sibling such as nvn-backup.
    REQUIRE((work_dir == "C:/eg/results/nvn" || work_dir.rfind("C:/eg/results/nvn/", 0) == 0 ||
             work_dir.rfind("F:/eg/results/", 0) == 0));
    fs::create_directories(WorkDir);
    for (const auto& entry : fs::directory_iterator(WorkDir)) {
        const auto name = entry.path().filename().string();
        if (name.rfind("result-", 0) == 0 || name.rfind("compute-", 0) == 0 ||
            (!prebuilt && name.rfind("store.", 0) == 0) || name.rfind("s-", 0) == 0 ||
            name.rfind("worker-", 0) == 0) {
            std::error_code ignored;
            fs::remove(entry.path(), ignored);
        }
    }

    // 1. Extract and deduplicate every shader once (or take the prebuilt store as it is).
    Containers::ScanOutput scan;
    size_t extracted = 0;
    size_t sampled = 0;
    if (prebuilt) {
        std::error_code ec;
        REQUIRE(fs::exists(WorkDir / "store.bin", ec));
        const auto index_bytes = fs::file_size(WorkDir / "store.idx", ec);
        REQUIRE(!ec);
        extracted = sampled = static_cast<size_t>(index_bytes / sizeof(u64));
        REQUIRE(extracted > 0);
        std::fprintf(stderr, "[coverage] prebuilt store: %zu shaders\n", extracted);
    } else {
        const auto scan_start = std::chrono::steady_clock::now();
        for (const auto& root : roots) {
            Containers::ScanRoot(root, scan);
        }
        extracted = scan.shaders.size();
        std::fprintf(stderr, "[coverage] extracted %zu unique shaders (%zu duplicates, %zu container failures) in %lld s\n",
                     extracted, scan.duplicates, scan.failures.size(),
                     static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(
                                                std::chrono::steady_clock::now() - scan_start)
                                                .count()));
        REQUIRE(extracted > 0);
        if (const std::string limit_text = Env("EDEN_NVN_COVERAGE_LIMIT"); !limit_text.empty()) {
            const size_t limit = static_cast<size_t>(std::strtoull(limit_text.c_str(), nullptr, 10));
            if (limit > 0 && limit < scan.shaders.size()) {
                std::vector<Containers::ExtractedShader> sample;
                sample.reserve(limit);
                for (size_t i = 0; i < limit; ++i) {
                    sample.push_back(std::move(scan.shaders[i * scan.shaders.size() / limit]));
                }
                scan.shaders = std::move(sample);
            }
        }
        sampled = scan.shaders.size();
        REQUIRE(WriteStore(scan.shaders, WorkDir / "store.bin", WorkDir / "store.idx"));
        {
            std::vector<Containers::ExtractedShader> release;
            scan.shaders.swap(release);
        }
    }

    // 2. Supervise the workers.
    u64 jobs = 6;
    if (const std::string text = Env("EDEN_NVN_COVERAGE_JOBS"); !text.empty()) {
        jobs = std::max<u64>(1, std::strtoull(text.c_str(), nullptr, 10));
    }
    jobs = std::min<u64>(jobs, sampled);
    std::string spirv_val = Env("EDEN_NVN_SPIRV_VAL");
    if (spirv_val.empty()) {
        spirv_val = DefaultSpirvVal;
    }
    if (spirv_val == "none" || !fs::exists(spirv_val)) {
        spirv_val.clear();
    }
    std::vector<WorkerProcess> workers(static_cast<size_t>(jobs));
    for (u64 i = 0; i < jobs; ++i) {
        workers[i].id = i;
        REQUIRE(Spawn(workers[i], jobs, spirv_val));
    }
    const auto result_path = [](u64 id) { return WorkDir / fmt::format("result-{}.tsv", id); };
    const auto append = [&](u64 id, u64 index, const char* klass, const std::string& text) {
        std::ofstream out(result_path(id), std::ios::app);
        out << "C\t" << index << '\t' << klass << '\t' << Sanitize(text) << '\n';
    };
    auto last_report = std::chrono::steady_clock::now();
    for (;;) {
        bool any_running = false;
        for (WorkerProcess& worker : workers) {
            if (worker.finished || worker.gave_up) {
                continue;
            }
            any_running = true;
            const auto path = result_path(worker.id);
            std::error_code ec;
            const auto size = fs::exists(path, ec) ? fs::file_size(path, ec) : 0;
            const auto now = std::chrono::steady_clock::now();
            if (size != worker.last_size) {
                worker.last_size = size;
                worker.last_progress = now;
                worker.fruitless_starts = 0;
            }
            DWORD wait = WaitForSingleObject(worker.process, 0);
            if (wait == WAIT_TIMEOUT &&
                now - worker.last_progress > std::chrono::seconds(WatchdogSeconds) &&
                InFlight(path).has_value()) {
                TerminateProcess(worker.process, 1); // our own worker
                if (WaitForSingleObject(worker.process, 10000) != WAIT_OBJECT_0) {
                    continue; // not gone yet: decide once it has exited
                }
                // The worker may have committed the shader before the termination took effect.
                if (const auto pending = InFlight(path)) {
                    append(worker.id, *pending, "timeout",
                           fmt::format("no progress for {} s", WatchdogSeconds));
                }
                wait = WAIT_OBJECT_0;
            } else if (wait == WAIT_OBJECT_0) {
                if (const auto in_flight = InFlight(path)) {
                    DWORD code = 0;
                    GetExitCodeProcess(worker.process, &code);
                    const std::string tail = LastLogLine(WorkDir / fmt::format("worker-{}.log", worker.id));
                    append(worker.id, *in_flight, "crash",
                           fmt::format("exit code {:#x}{}{}", code, tail.empty() ? "" : ": ", tail));
                }
            }
            if (wait != WAIT_OBJECT_0) {
                continue;
            }
            CloseHandle(worker.process);
            CloseHandle(worker.thread);
            if (HasFinished(path)) {
                worker.finished = true;
                continue;
            }
            ++worker.restarts;
            ++worker.fruitless_starts;
            if (worker.fruitless_starts > 3 || !Spawn(worker, jobs, spirv_val)) {
                worker.gave_up = true;
                std::fprintf(stderr, "[coverage] worker %llu gave up after %u restarts\n",
                             static_cast<unsigned long long>(worker.id), worker.restarts);
            }
        }
        if (!any_running) {
            break;
        }
        if (std::chrono::steady_clock::now() - last_report > std::chrono::seconds(60)) {
            last_report = std::chrono::steady_clock::now();
            u64 done = 0;
            for (const auto& worker : workers) {
                std::ifstream in(result_path(worker.id));
                std::string text;
                while (std::getline(in, text)) {
                    done += text.size() > 1 && (text[0] == 'R' || text[0] == 'C') && text[1] == '\t';
                }
            }
            std::fprintf(stderr, "[coverage] %llu / %zu shaders done\n",
                         static_cast<unsigned long long>(done), sampled);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // 3. Collect and report.
    std::vector<Row> rows;
    for (const auto& worker : workers) {
        std::ifstream in(result_path(worker.id));
        std::string text;
        while (std::getline(in, text)) {
            if (text.size() > 2 && text[0] == 'C' && text[1] == '\t') {
                const auto f = Split(text, '\t');
                if (f.size() >= 3) {
                    Row row;
                    row.index = std::strtoull(f[1].c_str(), nullptr, 10);
                    row.klass = f[2];
                    row.detail = f.size() > 3 ? f[3] : "";
                    row.reason = row.klass + ":" + Normalize(row.detail);
                    row.stage = "?";
                    row.key = fmt::format("index-{}", row.index);
                    rows.push_back(std::move(row));
                }
            } else if (Row row; Deserialize(text, row)) {
                rows.push_back(std::move(row));
            }
        }
    }
    // Crash/timeout rows know only an index: recover their identity from the store.
    {
        std::ifstream index_file(WorkDir / "store.idx", std::ios::binary | std::ios::ate);
        std::vector<u64> offsets(static_cast<size_t>(index_file.tellg()) / sizeof(u64));
        index_file.seekg(0);
        index_file.read(reinterpret_cast<char*>(offsets.data()),
                        static_cast<std::streamsize>(offsets.size() * sizeof(u64)));
        std::ifstream store(WorkDir / "store.bin", std::ios::binary);
        for (Row& row : rows) {
            if (row.stage == "?" && row.key.rfind("index-", 0) == 0 && row.index < offsets.size()) {
                StoredShader shader;
                if (ReadStored(store, offsets[row.index], shader)) {
                    row.key = shader.key;
                    row.source = shader.source;
                    row.kind = shader.kind;
                    std::string error;
                    if (const auto control = ParseControl(shader.control, error)) {
                        const auto stage = control->stage_byte == 5
                                               ? std::optional{ShaderStage::Compute}
                                               : StageFromSph(shader.code);
                        SetStage(row, stage);
                    }
                }
            }
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.index < b.index; });
    {
        std::ofstream compute(ReportPath("shader-compute-controls", ".tsv"));
        compute << "index\tkey\tsource\tthread_group\tregisters\tinst_bytes\tcode_bytes\t"
                   "uniform_off\tstorage_off\ttexture_off\timage_off\tend_off\tdeclared_total\t"
                   "total_matches\tstorage_desc_offsets\tcbuf_used\tuses_local\tuses_global\t"
                   "uses_workgroup_id\tcontrol_hex\n";
        for (const auto& worker : workers) {
            std::ifstream in(WorkDir / fmt::format("compute-{}.tsv", worker.id));
            // An empty or missing shard would leave `compute` failed for the rest.
            if (in && in.peek() != std::ifstream::traits_type::eof()) {
                compute << in.rdbuf();
            }
        }
    }
    WriteReport(rows, scan, roots, spirv_val, extracted, sampled);

    u64 failed = 0;
    for (const Row& row : rows) {
        failed += row.Failed();
    }
    std::fprintf(stderr, "[coverage] %zu shaders, %llu failed; report: %s\n", rows.size(),
                 static_cast<unsigned long long>(failed),
                 ReportPath("shader-coverage", ".md").string().c_str());
    CHECK(rows.size() == sampled);
    CHECK(failed < rows.size());
#endif
}
