// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Executes Ampere (Switch 2) instruction sequences on a real GPU through the same pipeline the
// games use: LowerToMaxwell -> Maxwell translation -> SPIR-V -> Vulkan compute. Tests compare the
// outputs with a CPU reference of the instruction's hardware semantics.
//
// Every dispatched group is one case: it loads eight input words (R8..R15) from its slot of storage
// buffer 0, runs the test body, and stores R16..R23 back into the slot. Slots are 128 bytes: inputs
// at +0x00, outputs at +0x40. Bodies are Ampere instruction words (the encodings the DK shaders
// contain, verified with nvdisasm); registers below R8 and above R23 are the harness's or the
// body's scratch.
//
// Skipped (the Rig is null) without a Vulkan device.

#pragma once

#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "common/literals.h"
#include "common/settings.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/shader/nvn_program_map.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace AmpereExec {

namespace VC = VideoCommon::Nvn;

struct Inst {
    u32 w[4];
};

constexpr u32 Sched = 0x000fe200;
constexpr u32 SR_CTAID_X = 0x25;

constexpr Inst S2R(u32 rd, u32 special) {
    return {{0x00007919 | (rd << 16), 0, special << 8, 0x000e2200}};
}
/// IMAD.MOV.U32 rd, RZ, RZ, c[0][byte_offset]
constexpr Inst MovC0(u32 rd, u32 byte_offset) {
    return {{0xff007624 | (rd << 16), byte_offset << 6, 0x078e00ff, Sched}};
}
/// IMAD rd, ra, imm, RZ
constexpr Inst MulImm(u32 rd, u32 ra, u32 imm) {
    return {{0x00007824 | (rd << 16) | (ra << 24), imm, 0x078e02ff, Sched}};
}
/// IADD3 rd, ra, rb, RZ
constexpr Inst Add(u32 rd, u32 ra, u32 rb) {
    return {{0x00007210 | (rd << 16) | (ra << 24), rb, 0x07ffe0ff, Sched}};
}
/// LDG.E.32 rd, [ra.64 + offset]
constexpr Inst Ldg(u32 rd, u32 ra, u32 offset) {
    // Bit 79 (a cache/ordering hint) set as in the games' own loads: Eglantine's lowering only
    // accepts that form.
    return {{0x00007381 | (rd << 16) | (ra << 24), offset << 8, 0x001e8900, Sched}};
}
/// STG.E.32 [ra.64 + offset], rs
constexpr Inst Stg(u32 ra, u32 offset, u32 rs) {
    return {{0x00007386 | (ra << 24), rs | (offset << 8), 0x0010a900, Sched}};
}
constexpr Inst Exit{{0x0000794d, 0, 0x03800000, 0x000fea00}};

constexpr u32 InputWords = 8;
constexpr u32 OutputWords = 8;
constexpr u32 SlotBytes = 128;
constexpr u32 OutputOffset = 0x40;
constexpr u32 FirstInputRegister = 8;
constexpr u32 FirstOutputRegister = 16;
constexpr u32 MaxCases = 65535;

using Words = std::array<u32, InputWords>;

class FakeMemory final : public VC::GuestMemory {
public:
    u8* Add(u64 va, size_t size, size_t alignment = 4096) {
        Region region;
        region.base = va;
        region.size = size;
        region.raw.resize(size + alignment);
        const auto raw = reinterpret_cast<uintptr_t>(region.raw.data());
        region.data = reinterpret_cast<u8*>((raw + alignment - 1) / alignment * alignment);
        std::memset(region.data, 0, size);
        regions.push_back(std::move(region));
        return regions.back().data;
    }
    bool Read(u64 va, void* dst, size_t size) override {
        const u8* source = Find(va, size);
        if (source) {
            std::memcpy(dst, source, size);
        }
        return source != nullptr;
    }
    bool Write(u64 va, const void* src, size_t size) override {
        u8* target = Find(va, size);
        if (target) {
            std::memcpy(target, src, size);
        }
        return target != nullptr;
    }
    u8* Pointer(u64 va) override {
        return Find(va, 1);
    }

private:
    struct Region {
        u64 base{};
        size_t size{};
        std::vector<u8> raw;
        u8* data{};
    };
    u8* Find(u64 va, size_t size) {
        for (Region& region : regions) {
            if (va >= region.base && va - region.base <= region.size &&
                size <= region.size - (va - region.base)) {
                return region.data + (va - region.base);
            }
        }
        return nullptr;
    }
    std::vector<Region> regions;
};

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
    std::mutex submit_mutex;
};

inline std::unique_ptr<HeadlessVulkan> MakeHeadlessVulkan() {
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
    } catch (const std::exception&) {
        return nullptr;
    }
}

template <typename T>
void PushRecord(std::vector<u64>& list, VC::Op op, const T& payload,
                std::span<const u8> extra = {}, u16 flags = 0) {
    const size_t bytes = sizeof(VC::RecordHeader) + sizeof(T) + extra.size();
    const size_t words = (bytes + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(op);
    header->flags = flags;
    header->size = static_cast<u32>(words * 8);
    auto* out = reinterpret_cast<u8*>(&list[at]) + sizeof(VC::RecordHeader);
    std::memcpy(out, &payload, sizeof(T));
    if (!extra.empty()) {
        std::memcpy(out + sizeof(T), extra.data(), extra.size());
    }
}

/// A ready-to-run GPU. Programs run one at a time; `Run` is not thread safe.
class Rig {
public:
    static std::unique_ptr<Rig> Create() {
        auto vulkan = MakeHeadlessVulkan();
        if (!vulkan) {
            return nullptr;
        }
        return std::unique_ptr<Rig>(new Rig(std::move(vulkan)));
    }

    struct Result {
        std::string error; // empty when the program ran
        std::vector<std::array<u32, OutputWords>> outputs;
    };
    /// One sampled image, sampler, and uniform-buffer handle word for a texture execution test.
    struct SampledTexture {
        VC::TextureDesc texture;
        VC::SamplerDesc sampler;
        std::vector<VC::ClearTexturePayload> clears;
        u32 uniform_slot;
        u64 uniform_offset;
    };


    /// Runs `body` once per input. The body sees the inputs in R8..R15 and returns results in
    /// R16..R23. Every byte of the storage buffer starts as 0xCD.
    Result Run(std::span<const Inst> body, std::span<const Words> inputs,
               const SampledTexture* sampled_texture = nullptr) {
        using namespace std::chrono_literals;
        Result result;
        if (inputs.empty() || inputs.size() > MaxCases) {
            result.error = "case count out of range";
            return result;
        }
        if (sampled_texture) {
            // The handle word's uniform range must stay clear of the inputs and inside the pool.
            const u64 storage_end = kStorageOffset + inputs.size() * SlotBytes;
            const u64 uniform_begin = sampled_texture->uniform_offset;
            if (uniform_begin > kPoolSize - 0x40 ||
                (uniform_begin < storage_end && uniform_begin + 0x40 > kStorageOffset)) {
                result.error = "uniform range overlaps the inputs or leaves the pool";
                return result;
            }
        }
        std::vector<Inst> program;
        // R0 = group id; R2:R3 = storage base + R0 * SlotBytes; R8..R15 = inputs.
        program.push_back(S2R(0, SR_CTAID_X));
        program.push_back(MovC0(2, kStorageEntry));
        program.push_back(MovC0(3, kStorageEntry + 4));
        program.push_back(MulImm(5, 0, SlotBytes));
        program.push_back(Add(2, 2, 5));
        for (u32 i = 0; i < InputWords; ++i) {
            program.push_back(Ldg(FirstInputRegister + i, 2, i * 4));
        }
        program.insert(program.end(), body.begin(), body.end());
        for (u32 i = 0; i < OutputWords; ++i) {
            program.push_back(Stg(2, OutputOffset + i * 4, FirstOutputRegister + i));
        }
        program.push_back(Exit);

        const u64 serial = next_program++;
        std::memset(data + kStorageOffset, 0xCD, inputs.size() * SlotBytes);
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::memcpy(data + kStorageOffset + i * SlotBytes, inputs[i].data(), sizeof(Words));
        }
        if (sampled_texture) {
            const u32 handle = 1 | (1U << 20);
            std::memcpy(data + sampled_texture->uniform_offset + 0x10, &handle, sizeof(handle));
        }
        context->CreateProgram(MakeProgram(serial, program));
        if (sampled_texture) {
            context->CreateTexture(sampled_texture->texture);
            context->CreateSampler(sampled_texture->sampler);
        }

        std::vector<u64> list;
        if (sampled_texture) {
            for (const VC::ClearTexturePayload& clear : sampled_texture->clears) {
                PushRecord(list, VC::Op::ClearTexture, clear, {}, 0xF);
            }
        }
        PushRecord(list, VC::Op::BindProgram,
                   VC::BindProgramPayload{.program_serial = serial, .stages = 1u << 5});
        const VC::BufferRange range{kPoolVa + kStorageOffset, inputs.size() * SlotBytes};
        PushRecord(list, VC::Op::BindStorageBuffers,
                   VC::BindBuffersPayload{.stage = 5, .first = 0, .count = 1},
                   {reinterpret_cast<const u8*>(&range), sizeof(range)});
        if (sampled_texture) {
            const VC::BufferRange uniform_range{kPoolVa + sampled_texture->uniform_offset, 0x40};
            PushRecord(list, VC::Op::BindUniformBuffers,
                       VC::BindBuffersPayload{.stage = 5, .first = sampled_texture->uniform_slot,
                                              .count = 1},
                       {reinterpret_cast<const u8*>(&uniform_range), sizeof(uniform_range)});
        }
        PushRecord(list, VC::Op::Dispatch,
                   VC::DispatchPayload{.x = static_cast<u32>(inputs.size()), .y = 1, .z = 1});
        list.push_back(0);
        auto* end = reinterpret_cast<VC::RecordHeader*>(&list.back());
        end->op = static_cast<u16>(VC::Op::End);
        end->size = 8;

        VC::Submission submission{};
        submission.seq = ++next_seq;
        submission.queue = 0;
        submission.lists.push_back(std::move(list));
        submission.retained_serials = {serial, 1};
        if (sampled_texture) {
            submission.retained_serials.push_back(sampled_texture->texture.serial);
            submission.retained_serials.push_back(sampled_texture->sampler.serial);
            submission.texture_pool_snapshot =
                VC::IdSnapshot{std::make_shared<const std::vector<u64>>(
                    std::vector<u64>{0, sampled_texture->texture.serial})};
            submission.sampler_pool_snapshot =
                VC::IdSnapshot{std::make_shared<const std::vector<u64>>(
                    std::vector<u64>{0, sampled_texture->sampler.serial})};
        }
        const auto before = context->Stats();
        const auto stat_count = [&](const char* reason) {
            std::vector<VC::SinkStat> rows;
            context->CollectStats(rows);
            for (const VC::SinkStat& row : rows) {
                if (row.reason == reason) {
                    return row.count;
                }
            }
            return u64{0};
        };
        const u64 skipped_before = stat_count("dispatches_skipped");
        context->Submit(std::move(submission));
        if (!context->Wait(next_seq, 120s)) {
            result.error = "GPU wait timed out";
            return result;
        }
        const auto after = context->Stats();
        if (after.ops_failed != before.ops_failed ||
            after.ops_unsupported != before.ops_unsupported ||
            stat_count("dispatches_skipped") != skipped_before) {
            result.error = "the replay failed or skipped the dispatch (pipeline rejected)";
            return result;
        }
        result.outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::memcpy(result.outputs[i].data(),
                        data + kStorageOffset + i * SlotBytes + OutputOffset, OutputWords * sizeof(u32));
        }
        return result;
    }

private:
    static constexpr u64 kPoolVa = 0x4000'0000;
    static constexpr u64 kPoolSize = 16ull * 1024 * 1024;
    static constexpr u64 kStorageOffset = 0x1000;
    static constexpr u32 kStorageEntry = 0x20; // c0: header 0x10, one uniform entry, then storage 0
    static constexpr u64 kPoolCpuCached = 4;

    explicit Rig(std::unique_ptr<HeadlessVulkan> vulkan_) : vulkan(std::move(vulkan_)) {
        data = memory.Add(kPoolVa, kPoolSize);
        context = std::make_unique<Vulkan::Nvn::Context>(
            *vulkan->device, vulkan->submit_mutex, nullptr,
            Vulkan::Nvn::ContextOptions{.allow_host_import = true});
        context->SetGuestMemory(&memory);
        context->CreatePool({.serial = 1, .guest_address = kPoolVa, .size = kPoolSize,
                             .flags = kPoolCpuCached});
    }

    static void Put32(std::vector<u8>& bytes, size_t offset, u32 value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    VC::ProgramDesc MakeProgram(u64 serial, std::span<const Inst> program) {
        const u32 inst_bytes = static_cast<u32>(program.size() * sizeof(Inst));
        const u32 c1_offset = (inst_bytes + 0xff) / 0x100 * 0x100;
        std::vector<u8> control(0x100, 0);
        Put32(control, 0, VC::ControlMagic);
        Put32(control, 4, VC::ControlVersion);
        Put32(control, 0x38, inst_bytes);
        Put32(control, 0x40, c1_offset);
        Put32(control, 0x44, c1_offset);
        control[0x4c] = 12;
        control[0x56] = 5; // compute
        Put32(control, 0x58, 1);
        Put32(control, 0x5c, 1);
        Put32(control, 0x60, 1);
        control[0xb1] = 1; // one uniform entry
        control[0xb2] = 1; // one storage entry
        const u32 total = 0x10 + 2 * 0x10;
        control[0xba] = static_cast<u8>(total);
        control[0xbb] = static_cast<u8>(total >> 8);
        std::vector<u8> code(c1_offset, 0);
        std::memcpy(code.data(), program.data(), inst_bytes);
        VC::ProgramDesc desc{};
        desc.serial = serial;
        desc.control[5] = std::move(control);
        desc.code[5] = std::move(code);
        desc.stage_mask = 1u << 5;
        return desc;
    }

    std::unique_ptr<HeadlessVulkan> vulkan;
    FakeMemory memory;
    u8* data{};
    std::unique_ptr<Vulkan::Nvn::Context> context;
    u64 next_program = 100;
    u64 next_seq = 0;
};

} // namespace AmpereExec
