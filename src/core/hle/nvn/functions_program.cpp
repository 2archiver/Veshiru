// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <atomic>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "core/hle/nvn/hle_call.h"
#include "core/hle/nvn/hle_runtime.h"

// Program family: NVNprogram and the NVN2 program builder. The Recorder copies the control sections
// and the code areas at SetShaders (nxnative keyed its replay by the stage's hash instead) and
// derives the stage of each entry from its control section.
//
// Functions of the program family without a body here (ray tracing: nvnProgramBuilder{Get,Set}
// MaxCallableRecursionDepth / MaxTraceRecursionDepth / ProgramRayTracingFlags / RayTracingModules,
// debug data hashes, subroutine linkage, nvnProgramGetDeviceComputeMemorySize / ScratchBytesPerWarp /
// ShaderGroupHandle / SubtileSize) have no prototype and no known semantics: they stay unregistered
// so the HLE counts them as unimplemented.
namespace Core::NvnHle::Hle {
namespace {

/// One entry per NVN stage at most (0..5 as on NVN1, 6 task and 7 mesh).
constexpr s32 MaxStages = static_cast<s32>(VideoCommon::Nvn::PROGRAM_STAGES);

// NVN2 NVNprogramBuilder in place (nxnative prog_builder_t {NVNdevice* dev; const NVNshaderData* sd;
// int n;}, 0x18 bytes with 4 bytes of padding after `n`). The padding holds the sample shading mode,
// which nxnative does not keep. [UNVERIFIED] the real NVN2 builder's size (nxnative uses 0x18 as
// well).
constexpr u64 PgbDevice = 0x00;
constexpr u64 PgbShaders = 0x08;
constexpr u64 PgbCount = 0x10;
constexpr u64 PgbSampleShading = 0x14;
constexpr u64 PgbBytes = 0x18;

// nvnProgramSetShadersExt entry: NVNshaderData{data, control} plus an 8-byte extension field
// (nxnative: filled from Super Mario Bros. Wonder's main 0x1074ff0, and by Donkey Kong Bananza).
constexpr u64 ExtEntryStride = 0x18;
constexpr u64 ExtData = 0x00;
constexpr u64 ExtControl = 0x08;
constexpr u64 ExtExtension = 0x10;

void nvnProgramInitialize(Call& c) {
    const u64 program = c.args.NextPointer();
    static_cast<void>(c.args.NextPointer()); // device
    c.ret.SetBool(c.rec.ProgramInitialize(program));
}

void nvnProgramFinalize(Call& c) {
    c.rec.ProgramFinalize(c.args.NextPointer());
}

void nvnProgramSetDebugLabel(Call& c) {
    static_cast<void>(c);
}

/// The entry checks of the SDK's SetShaders (3.2.0 sdk+0x5e38a0, 1.0.0 sdk+0x5bf8a0): every entry
/// needs code (`data` != 0) and an 8-byte aligned control section; the SDK has already emptied
/// the program when it fails (it clears the program before validating), so a refused set leaves
/// an initialized program without shaders. The control-section checks the SDK does on top
/// (GPU code major 201 and minor <= 17 (1.0.0: 11), one entry per stage, the section's arch
/// matching the device) need the SDK's control-section decoder; the Recorder reads the stage.
bool ValidShaderEntry(u64 data, u64 control) {
    return data != 0 && control != 0 && (control & 7) == 0;
}

/// Empties `program` as a refused SDK SetShaders does; returns false (the call's result).
bool RefuseShaders(Call& c, u64 program) {
    static_cast<void>(c.rec.ProgramSetShaders(program, std::span<const VideoCommon::Nvn::ShaderStageInput>{}));
    return false;
}

bool SetShaders(Call& c, u64 program, s32 count, u64 data) {
    if (count > 0 && count <= MaxStages) {
        std::array<u64, 2 * MaxStages> entries{};
        if (!c.mem.Read(data, entries.data(), sizeof(u64) * 2 * static_cast<size_t>(count))) {
            return false;
        }
        for (s32 i = 0; i < count; ++i) {
            if (!ValidShaderEntry(entries[2 * i], entries[2 * i + 1])) {
                return RefuseShaders(c, program);
            }
        }
    }
    return c.rec.ProgramSetShaders(program, count, data);
}

void nvnProgramSetShaders(Call& c) {
    const u64 program = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 data = c.args.NextPointer();
    c.ret.SetBool(SetShaders(c, program, count, data));
}

/// Each entry is 0x18 bytes: NVNshaderData plus an extension field the host does not use (nxnative
/// ignores it too; the stage comes from the control section). The SDK copies the pairs and the
/// extensions apart and runs SetShaders (3.2.0 sdk+0x5e42a0).
void nvnProgramSetShadersExt(Call& c) {
    const u64 program = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 data = c.args.NextPointer();
    if (count < 0 || count > MaxStages) {
        static Common::BootCounter logged;
        if (logged.First()) {
            LOG_WARNING(Core, "[nvn] ProgramSetShadersExt stage count {}", count);
        }
        c.ret.SetBool(false);
        return;
    }
    std::array<VideoCommon::Nvn::ShaderStageInput, MaxStages> stages{};
    for (s32 i = 0; i < count; ++i) {
        const u64 entry = data + static_cast<u64>(i) * ExtEntryStride;
        u64 fields[3]{};
        if (!c.mem.Read(entry, fields, sizeof(fields))) {
            c.ret.SetBool(false);
            return;
        }
        if (!ValidShaderEntry(fields[ExtData / 8], fields[ExtControl / 8])) {
            c.ret.SetBool(RefuseShaders(c, program));
            return;
        }
        stages[i] = VideoCommon::Nvn::ShaderStageInput{fields[ExtData / 8], fields[ExtControl / 8], -1};
        if (fields[ExtExtension / 8] != 0) {
            static Common::BootCounter logged;
            if (logged.First()) {
                LOG_INFO(Core,
                         "[nvn] ProgramSetShadersExt extension field nonzero ({:#x}): unused by the "
                         "host",
                         fields[ExtExtension / 8]);
            }
        }
    }
    c.ret.SetBool(c.rec.ProgramSetShaders(
        program, std::span<const VideoCommon::Nvn::ShaderStageInput>{stages.data(),
                                                                       static_cast<size_t>(count)}));
}

/// nvnProgramSetSampleShading (NVN 53+, no prototype in nvn.h): whether the fragment shader runs
/// once per sample of a multisampled target. Vulkan decides this from the shader (a read of the
/// sample id), as Metal does; the mode is accepted and not applied.
void nvnProgramSetSampleShading(Call& c) {
    static_cast<void>(c.args.NextPointer()); // program
    const s32 mode = c.args.NextS32();
    static Common::BootCounter logged;
    if (logged.First()) {
        LOG_INFO(Core, "[nvn] ProgramSetSampleShading mode {}: not applied (per-sample shading "
                       "follows the shader)",
                 mode);
    }
    c.ret.SetBool(true);
}

// ---- NVN2 program builder ----------------------------------------------------------------------------------

/// The SDK's SetDefaults (3.2.0 sdk+0x5e47e0, 1.0.0 sdk+0x5c0920) clears everything but the
/// device: keep it.
void nvnProgramBuilderSetDefaults(Call& c) {
    const u64 builder = c.args.NextPointer();
    const u64 device = c.Load<u64>(builder + PgbDevice);
    c.Zero(builder, PgbBytes);
    c.Store<u64>(builder + PgbDevice, device);
}

void nvnProgramBuilderSetDevice(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<u64>(builder + PgbDevice, c.args.NextPointer());
}

/// The builder only keeps the caller's array, which the game keeps until it initializes the
/// program (its stack frame), as other NVN builders keep their pointers.
void nvnProgramBuilderSetShaders(Call& c) {
    const u64 builder = c.args.NextPointer();
    const s32 count = c.args.NextS32();
    const u64 shaders = c.args.NextPointer();
    c.Store<s32>(builder + PgbCount, count);
    c.Store<u64>(builder + PgbShaders, shaders);
}

void nvnProgramBuilderSetSampleShading(Call& c) {
    const u64 builder = c.args.NextPointer();
    c.Store<s32>(builder + PgbSampleShading, c.args.NextS32());
}

void nvnProgramBuilderGetDevice(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + PgbDevice));
}

void nvnProgramBuilderGetShaderCount(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + PgbCount));
}

void nvnProgramBuilderGetShaders(Call& c) {
    c.ret.SetU64(c.Load<u64>(c.args.NextPointer() + PgbShaders));
}

void nvnProgramBuilderGetSampleShading(Call& c) {
    c.ret.SetU32(c.Load<u32>(c.args.NextPointer() + PgbSampleShading));
}

/// nvnProgramInitialize + SetShaders over the builder's array (3.2.0 sdk+0x5e56b0: the same object
/// setup, then the SetShaders path). A builder without shaders gives an empty program.
void nvnProgramInitializeFromBuilder(Call& c) {
    const u64 program = c.args.NextPointer();
    const u64 builder = c.args.NextPointer();
    const u64 shaders = c.Load<u64>(builder + PgbShaders);
    const s32 count = c.Load<s32>(builder + PgbCount);
    if (!c.rec.ProgramInitialize(program)) {
        c.ret.SetBool(false);
        return;
    }
    c.ret.SetBool(count == 0 || SetShaders(c, program, count, shaders));
}

} // namespace

void RegisterProgramFunctions(Registry& registry) {
    NVN_HLE(registry, nvnProgramInitialize);
    NVN_HLE(registry, nvnProgramFinalize);
    NVN_HLE(registry, nvnProgramSetDebugLabel);
    NVN_HLE(registry, nvnProgramSetShaders);
    NVN_HLE(registry, nvnProgramSetShadersExt);
    NVN_HLE(registry, nvnProgramSetSampleShading);
    NVN_HLE(registry, nvnProgramBuilderSetDefaults);
    NVN_HLE(registry, nvnProgramBuilderSetDevice);
    NVN_HLE(registry, nvnProgramBuilderSetShaders);
    NVN_HLE(registry, nvnProgramBuilderSetSampleShading);
    NVN_HLE(registry, nvnProgramBuilderGetDevice);
    NVN_HLE(registry, nvnProgramBuilderGetShaderCount);
    NVN_HLE(registry, nvnProgramBuilderGetShaders);
    NVN_HLE(registry, nvnProgramBuilderGetSampleShading);
    NVN_HLE(registry, nvnProgramInitializeFromBuilder);
}

} // namespace Core::NvnHle::Hle
