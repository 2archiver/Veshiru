// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <array>
#include <bit>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tests/nvn/hle_rig.h"
#include "video_core/nvn/guest_layouts.h"
#include "video_core/nvn/record.h"

// Command buffer state and bind bodies (functions_cmdbuf_bind.cpp): each test records through the
// registered NVN bodies, submits the list and inspects the records the sink receives.
namespace {

using namespace VideoCommon::Nvn;
using NvnHleTest::Arg;
using NvnHleTest::Rig;

Arg I(u64 value) {
    return Rig::I(value);
}
Arg F(float value) {
    return Rig::F(value);
}

constexpr u64 SCRATCH = Rig::SCRATCH;

// NVNshaderStage values.
constexpr u64 STAGE_VERTEX = 0;
constexpr u64 STAGE_FRAGMENT = 4;

struct BindRig {
    static constexpr u64 POOL = Rig::Obj(0);
    static constexpr u64 QUEUE = Rig::Obj(1);
    static constexpr u64 CMD = Rig::Obj(2);

    BindRig() {
        REQUIRE(rig.recorder.MemoryPoolInitialize(POOL, Rig::STORAGE, Rig::STORAGE_SIZE, 0));
        REQUIRE(rig.recorder.QueueInitialize(QUEUE));
        REQUIRE(rig.recorder.CommandBufferInitialize(CMD));
        rig.recorder.AddCommandMemory(CMD, POOL, 0, 0x10000);
        rig.recorder.BeginRecording(CMD);
    }

    void InitTexture(u64 va, u64 offset = 0) {
        TextureDesc desc{};
        desc.target = 1;
        desc.format = 0x25;
        desc.width = 64;
        desc.height = 64;
        desc.depth = 1;
        desc.levels = 1;
        REQUIRE(rig.recorder.TextureInitialize(va, desc, POOL, offset));
    }

    /// Ends the recording; returns the command handle for Submit.
    u64 End() {
        return rig.recorder.EndRecording(CMD);
    }

    /// Submits an ended recording; returns the submitted (expanded) list.
    const std::vector<u64>& Submit(u64 handle) {
        REQUIRE(rig.recorder.QueueSubmitCommands(QUEUE, std::array{handle}) != 0);
        return rig.sink.submissions.back().lists.at(0);
    }

    /// Ends the recording and submits it; returns the submitted (expanded) list.
    const std::vector<u64>& Submit() {
        return Submit(End());
    }

    Rig rig;
};

std::vector<const RecordHeader*> Find(const std::vector<u64>& list, Op op) {
    std::vector<const RecordHeader*> found;
    RecordReader reader{list.data(), list.size()};
    while (const RecordHeader* header = reader.Next()) {
        if (RecordOp(header) == op) {
            found.push_back(header);
        }
    }
    return found;
}

size_t RecordCount(const std::vector<u64>& list) {
    size_t count = 0;
    RecordReader reader{list.data(), list.size()};
    while (reader.Next() != nullptr) {
        ++count;
    }
    return count;
}

u32 Bits(float value) {
    return std::bit_cast<u32>(value);
}

} // namespace

TEST_CASE("nvn hle bind: state objects pass their kind and in-place bytes", "[nvn]") {
    BindRig f;
    struct Case {
        const char* name;
        StateKind kind;
    };
    const std::array<Case, 6> cases{{
        {"nvnCommandBufferBindBlendState", StateKind::Blend},
        {"nvnCommandBufferBindColorState", StateKind::Color},
        {"nvnCommandBufferBindChannelMaskState", StateKind::ChannelMask},
        {"nvnCommandBufferBindMultisampleState", StateKind::Multisample},
        {"nvnCommandBufferBindPolygonState", StateKind::Polygon},
        {"nvnCommandBufferBindDepthStencilState", StateKind::DepthStencil},
    }};
    std::array<std::array<u8, 16>, 6> contents{};
    for (u32 k = 0; k < 6; ++k) {
        // 16 bytes of per-kind data in the object slot; only the first STATE_OBJECT_SIZES[k] bytes
        // belong to the state, the rest must not be recorded.
        for (u32 i = 0; i < 16; ++i) {
            contents[k][i] = static_cast<u8>(0x10 * (k + 1) + i);
        }
        f.rig.memory.Put(Rig::Obj(10 + k), contents[k]);
        f.rig.Call(cases[k].name, {I(BindRig::CMD), I(Rig::Obj(10 + k))});
    }
    // The game reuses the state object after the call.
    for (u32 k = 0; k < 6; ++k) {
        f.rig.memory.Put(Rig::Obj(10 + k), std::array<u8, 16>{});
    }
    // An unreadable state pointer records nothing and does not throw.
    f.rig.Call("nvnCommandBufferBindBlendState", {I(BindRig::CMD), I(0)});

    const std::vector<u64>& list = f.Submit();
    const std::vector<const RecordHeader*> states = Find(list, Op::BindState);
    REQUIRE(states.size() == 6);
    REQUIRE(RecordCount(list) == 6);
    for (u32 k = 0; k < 6; ++k) {
        const auto* payload = RecordPayload<BindStatePayload>(states[k]);
        CHECK(payload->kind == static_cast<u32>(cases[k].kind));
        CHECK(payload->size == STATE_OBJECT_SIZES[k]);
        CHECK(std::memcmp(payload + 1, contents[k].data(), STATE_OBJECT_SIZES[k]) == 0);
    }
}

TEST_CASE("nvn hle bind: program, vertex state and vertex buffers", "[nvn]") {
    BindRig f;
    REQUIRE(f.rig.recorder.ProgramInitialize(Rig::Obj(40)));
    f.rig.Call("nvnCommandBufferBindProgram",
               {I(BindRig::CMD), I(Rig::Obj(40)), I(0xFFFF'FFFF'0000'001FULL)});
    // A program pointer without a tag records nothing.
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(Rig::Obj(41)), I(1)});

    f.rig.memory.Put(SCRATCH, std::array<u32, 2>{0xAAAA0001, 0xBBBB0002});
    f.rig.Call("nvnCommandBufferBindVertexAttribState", {I(BindRig::CMD), I(2), I(SCRATCH)});
    f.rig.memory.Put(SCRATCH + 0x40, std::array<u32, 4>{16, 0, 32, 1});
    f.rig.Call("nvnCommandBufferBindVertexStreamState", {I(BindRig::CMD), I(2), I(SCRATCH + 0x40)});
    // More states than NVN has attribute slots: dropped and counted.
    f.rig.Call("nvnCommandBufferBindVertexAttribState", {I(BindRig::CMD), I(17), I(SCRATCH)});

    f.rig.Call("nvnCommandBufferBindVertexBuffer",
               {I(BindRig::CMD), I(3), I(Rig::STORAGE + 0x100), I(0x40)});
    f.rig.memory.Put(SCRATCH + 0x80, std::array<u64, 4>{Rig::STORAGE + 0x200, 0x10,
                                                          Rig::STORAGE + 0x300, 0x20});
    f.rig.Call("nvnCommandBufferBindVertexBuffers",
               {I(BindRig::CMD), I(1), I(2), I(SCRATCH + 0x80)});

    const RecorderStats stats = f.rig.recorder.Stats();
    const std::vector<u64>& list = f.Submit();

    const auto programs = Find(list, Op::BindProgram);
    REQUIRE(programs.size() == 1);
    const auto* program = RecordPayload<BindProgramPayload>(programs[0]);
    CHECK(program->program_serial != 0);
    CHECK(program->stages == 0x1F); // the int argument, not its upper register half
    CHECK(stats.tag_mismatch >= 1);

    const auto attribs = Find(list, Op::BindVertexAttribs);
    REQUIRE(attribs.size() == 1);
    const auto* attrib = RecordPayload<BindVertexAttribsPayload>(attribs[0]);
    CHECK(attrib->first == 0);
    CHECK(attrib->count == 2);
    CHECK(TrailingArray<u32>(attrib)[0] == 0xAAAA0001);
    CHECK(TrailingArray<u32>(attrib)[1] == 0xBBBB0002);
    CHECK(stats.bad_argument >= 1);

    const auto streams = Find(list, Op::BindVertexStreams);
    REQUIRE(streams.size() == 1);
    const auto* stream = RecordPayload<BindVertexStreamsPayload>(streams[0]);
    CHECK(stream->count == 2);
    CHECK(TrailingArray<VertexStreamState>(stream)[0].stride == 16);
    CHECK(TrailingArray<VertexStreamState>(stream)[0].divisor == 0);
    CHECK(TrailingArray<VertexStreamState>(stream)[1].stride == 32);
    CHECK(TrailingArray<VertexStreamState>(stream)[1].divisor == 1);

    const auto buffers = Find(list, Op::BindVertexBuffers);
    REQUIRE(buffers.size() == 2);
    const auto* single = RecordPayload<BindVertexBuffersPayload>(buffers[0]);
    CHECK(single->first == 3);
    CHECK(single->count == 1);
    CHECK(TrailingArray<BufferRange>(single)[0].address == Rig::STORAGE + 0x100);
    CHECK(TrailingArray<BufferRange>(single)[0].size == 0x40);
    const auto* many = RecordPayload<BindVertexBuffersPayload>(buffers[1]);
    CHECK(many->first == 1);
    CHECK(many->count == 2);
    CHECK(TrailingArray<BufferRange>(many)[1].address == Rig::STORAGE + 0x300);
    CHECK(TrailingArray<BufferRange>(many)[1].size == 0x20);
}

TEST_CASE("nvn hle bind: uniform and storage buffers, immediate and deferred", "[nvn]") {
    BindRig f;
    // The stage is an int: the upper half of x1 is garbage the body must not read.
    f.rig.Call("nvnCommandBufferBindUniformBuffer",
               {I(BindRig::CMD), I(0xDEAD'BEEF'0000'0004ULL), I(2), I(Rig::STORAGE + 0x1000),
                I(0x100)});
    f.rig.Call("nvnCommandBufferBindStorageBuffer",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(7), I(Rig::STORAGE + 0x2000), I(0x200)});

    f.rig.memory.Put(SCRATCH, std::array<u64, 6>{Rig::STORAGE + 0x10, 1, Rig::STORAGE + 0x20, 2,
                                                  Rig::STORAGE + 0x30, 3});
    f.rig.Call("nvnCommandBufferBindUniformBuffers",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(1), I(3), I(SCRATCH)});
    f.rig.Call("nvnCommandBufferBindStorageBuffers",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(0), I(2), I(SCRATCH)});

    // Deferred: the ranges are read again at submit.
    f.rig.memory.Put(SCRATCH + 0x100, std::array<u64, 2>{Rig::STORAGE + 0x40, 4});
    f.rig.Call("nvnCommandBufferBindUniformBuffersDeferred",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(5), I(1), I(SCRATCH + 0x100)});
    f.rig.Call("nvnCommandBufferBindStorageBuffersDeferred",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(6), I(1), I(SCRATCH + 0x100)});
    const u64 handle = f.End();
    f.rig.memory.Put(SCRATCH + 0x100, std::array<u64, 2>{Rig::STORAGE + 0x50, 5});

    const std::vector<u64>& list = f.Submit(handle);
    const auto uniforms = Find(list, Op::BindUniformBuffers);
    REQUIRE(uniforms.size() == 3);
    const auto* ubo = RecordPayload<BindBuffersPayload>(uniforms[0]);
    CHECK(ubo->stage == 4);
    CHECK(ubo->first == 2);
    CHECK(ubo->count == 1);
    CHECK(TrailingArray<BufferRange>(ubo)[0].address == Rig::STORAGE + 0x1000);
    CHECK(TrailingArray<BufferRange>(ubo)[0].size == 0x100);
    const auto* ubos = RecordPayload<BindBuffersPayload>(uniforms[1]);
    CHECK(ubos->stage == STAGE_FRAGMENT);
    CHECK(ubos->first == 1);
    CHECK(ubos->count == 3);
    CHECK(TrailingArray<BufferRange>(ubos)[2].address == Rig::STORAGE + 0x30);
    CHECK(TrailingArray<BufferRange>(ubos)[2].size == 0x100); // 3 rounded up to 256 bytes
    const auto* ubo_deferred = RecordPayload<BindBuffersPayload>(uniforms[2]);
    CHECK(ubo_deferred->first == 5);
    CHECK(TrailingArray<BufferRange>(ubo_deferred)[0].address == Rig::STORAGE + 0x50);
    CHECK(TrailingArray<BufferRange>(ubo_deferred)[0].size == 0x100); // re-read 5, rounded

    const auto storages = Find(list, Op::BindStorageBuffers);
    REQUIRE(storages.size() == 3);
    const auto* ssbo = RecordPayload<BindBuffersPayload>(storages[0]);
    CHECK(ssbo->stage == STAGE_VERTEX);
    CHECK(ssbo->first == 7);
    CHECK(TrailingArray<BufferRange>(ssbo)[0].address == Rig::STORAGE + 0x2000);
    CHECK(TrailingArray<BufferRange>(ssbo)[0].size == 0x200);
    const auto* ssbos = RecordPayload<BindBuffersPayload>(storages[1]);
    CHECK(ssbos->count == 2);
    CHECK(TrailingArray<BufferRange>(ssbos)[1].address == Rig::STORAGE + 0x20);
    const auto* ssbo_deferred = RecordPayload<BindBuffersPayload>(storages[2]);
    CHECK(ssbo_deferred->first == 6);
    CHECK(TrailingArray<BufferRange>(ssbo_deferred)[0].size == 5); // storage sizes stay raw
}

TEST_CASE("nvn hle bind: texture handles go to the right op, slot and stage", "[nvn]") {
    BindRig f;
    const u64 combined = 0x00030002ULL | (u64{7} << 20);
    f.rig.Call("nvnCommandBufferBindTexture",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(5), I(combined)});
    // Bits above 32 never reach the GPU (EA FC binds separate textures with bit 32 set).
    f.rig.Call("nvnCommandBufferBindSeparateTexture",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(3), I(0x1'0000'01D1ULL)});
    f.rig.Call("nvnCommandBufferBindSeparateSampler",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(2), I(u64{9} << 20)});
    f.rig.Call("nvnCommandBufferBindImage", {I(BindRig::CMD), I(5), I(1), I(0x44)});

    const std::vector<u64>& list = f.Submit();
    REQUIRE(RecordCount(list) == 4);

    const auto textures = Find(list, Op::BindTextures);
    REQUIRE(textures.size() == 1);
    const auto* texture = RecordPayload<BindHandlesPayload>(textures[0]);
    CHECK(texture->stage == STAGE_FRAGMENT);
    CHECK(texture->first == 5);
    CHECK(texture->count == 1);
    CHECK(TrailingArray<u64>(texture)[0] == combined);

    const auto separate = Find(list, Op::BindSeparateTextures);
    REQUIRE(separate.size() == 1);
    const auto* separate_texture = RecordPayload<BindHandlesPayload>(separate[0]);
    CHECK(separate_texture->stage == STAGE_VERTEX);
    CHECK(separate_texture->first == 3);
    CHECK(TrailingArray<u64>(separate_texture)[0] == 0x1D1);

    const auto samplers = Find(list, Op::BindSeparateSamplers);
    REQUIRE(samplers.size() == 1);
    const auto* separate_sampler = RecordPayload<BindHandlesPayload>(samplers[0]);
    CHECK(separate_sampler->stage == STAGE_FRAGMENT);
    CHECK(separate_sampler->first == 2);
    CHECK(TrailingArray<u64>(separate_sampler)[0] == (u64{9} << 20));

    const auto images = Find(list, Op::BindImages);
    REQUIRE(images.size() == 1);
    const auto* image = RecordPayload<BindHandlesPayload>(images[0]);
    CHECK(image->stage == 5);
    CHECK(image->first == 1);
    CHECK(TrailingArray<u64>(image)[0] == 0x44);
}

TEST_CASE("nvn hle bind: texture handle arrays, counts and deferred reads", "[nvn]") {
    BindRig f;
    const std::array<u64, 3> handles{0x00010002, 0, 0x00050006ULL | (u64{3} << 20)};
    f.rig.memory.Put(SCRATCH, handles);
    f.rig.Call("nvnCommandBufferBindTextures",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(2), I(3), I(SCRATCH)});
    f.rig.Call("nvnCommandBufferBindImages", {I(BindRig::CMD), I(STAGE_VERTEX), I(0), I(2), I(SCRATCH)});
    f.rig.Call("nvnCommandBufferBindSeparateTextures",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(4), I(2), I(SCRATCH)});
    f.rig.Call("nvnCommandBufferBindSeparateSamplers",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(1), I(1), I(SCRATCH + 0x10)});
    // More handles than a command can carry: dropped and counted.
    f.rig.Call("nvnCommandBufferBindTextures",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(0), I(33), I(SCRATCH)});

    // Deferred variants: the handles are read at submit.
    f.rig.memory.Put(SCRATCH + 0x100, std::array<u64, 2>{0x11, 0x22});
    f.rig.Call("nvnCommandBufferBindTexturesDeferred",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(8), I(2), I(SCRATCH + 0x100)});
    f.rig.Call("nvnCommandBufferBindImagesDeferred",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(9), I(2), I(SCRATCH + 0x100)});
    f.rig.Call("nvnCommandBufferBindSeparateTexturesDeferred",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(10), I(1), I(SCRATCH + 0x100)});
    f.rig.Call("nvnCommandBufferBindSeparateSamplersDeferred",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(11), I(1), I(SCRATCH + 0x108)});
    const u64 handle = f.End();
    f.rig.memory.Put(SCRATCH + 0x100, std::array<u64, 2>{0x33, 0x44ULL << 20});

    const RecorderStats stats = f.rig.recorder.Stats();
    const std::vector<u64>& list = f.Submit(handle);
    CHECK(stats.bad_argument >= 1);

    const auto textures = Find(list, Op::BindTextures);
    REQUIRE(textures.size() == 2);
    const auto* immediate = RecordPayload<BindHandlesPayload>(textures[0]);
    CHECK(immediate->stage == STAGE_FRAGMENT);
    CHECK(immediate->first == 2);
    REQUIRE(immediate->count == 3);
    CHECK(TrailingArray<u64>(immediate)[0] == handles[0]);
    CHECK(TrailingArray<u64>(immediate)[1] == 0);
    CHECK(TrailingArray<u64>(immediate)[2] == handles[2]);
    const auto* deferred = RecordPayload<BindHandlesPayload>(textures[1]);
    CHECK(deferred->first == 8);
    REQUIRE(deferred->count == 2);
    CHECK(TrailingArray<u64>(deferred)[0] == 0x33);
    CHECK(TrailingArray<u64>(deferred)[1] == (u64{0x44} << 20)); // texture 0 stays 0, sampler 0x44

    const auto images = Find(list, Op::BindImages);
    REQUIRE(images.size() == 2);
    const auto* image = RecordPayload<BindHandlesPayload>(images[0]);
    CHECK(image->count == 2);
    CHECK(TrailingArray<u64>(image)[0] == handles[0]);
    const auto* image_deferred = RecordPayload<BindHandlesPayload>(images[1]);
    CHECK(image_deferred->first == 9);
    CHECK(TrailingArray<u64>(image_deferred)[0] == 0x33);

    const auto separate = Find(list, Op::BindSeparateTextures);
    REQUIRE(separate.size() == 2);
    CHECK(RecordPayload<BindHandlesPayload>(separate[0])->first == 4);
    CHECK(TrailingArray<u64>(RecordPayload<BindHandlesPayload>(separate[1]))[0] == 0x33);

    const auto samplers = Find(list, Op::BindSeparateSamplers);
    REQUIRE(samplers.size() == 2);
    CHECK(RecordPayload<BindHandlesPayload>(samplers[0])->first == 1);
    // The handle read from SCRATCH + 0x10 is handles[2]: sampler id 3.
    CHECK(TrailingArray<u64>(RecordPayload<BindHandlesPayload>(samplers[0]))[0] == (u64{3} << 20));
    CHECK(RecordPayload<BindHandlesPayload>(samplers[1])->first == 11);
    CHECK(TrailingArray<u64>(RecordPayload<BindHandlesPayload>(samplers[1]))[0] ==
          (u64{0x44} << 20));
}

TEST_CASE("nvn hle bind: texture and sampler pools rebase later handles", "[nvn]") {
    BindRig f;
    REQUIRE(f.rig.recorder.TexturePoolInitialize(Rig::Obj(20), BindRig::POOL, 0x20000, 64));
    REQUIRE(f.rig.recorder.TexturePoolInitialize(Rig::Obj(21), BindRig::POOL, 0x30000, 64));
    REQUIRE(f.rig.recorder.SamplerPoolInitialize(Rig::Obj(22), BindRig::POOL, 0x40000, 16));
    REQUIRE(f.rig.recorder.SamplerPoolInitialize(Rig::Obj(23), BindRig::POOL, 0x50000, 16));
    const DescriptorPoolInfo texture_pool = *f.rig.recorder.GetTexturePool(Rig::Obj(21));
    const DescriptorPoolInfo sampler_pool = *f.rig.recorder.GetSamplerPool(Rig::Obj(23));
    REQUIRE(texture_pool.base != 0);
    REQUIRE(sampler_pool.base != 0);

    f.rig.Call("nvnCommandBufferSetTexturePool", {I(BindRig::CMD), I(Rig::Obj(21))});
    f.rig.Call("nvnCommandBufferSetSamplerPool", {I(BindRig::CMD), I(Rig::Obj(23))});
    f.rig.Call("nvnCommandBufferBindTexture",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(0), I(4 | (u64{3} << 20))});
    // Not a pool: nothing recorded.
    f.rig.Call("nvnCommandBufferSetTexturePool", {I(BindRig::CMD), I(Rig::Obj(50))});

    const std::vector<u64>& list = f.Submit();
    const auto texture_pools = Find(list, Op::SetTexturePool);
    REQUIRE(texture_pools.size() == 1);
    const auto* texture_record = RecordPayload<SetPoolPayload>(texture_pools[0]);
    CHECK(texture_record->pool_serial == texture_pool.serial);
    CHECK(texture_record->base == texture_pool.base);
    const auto sampler_pools = Find(list, Op::SetSamplerPool);
    REQUIRE(sampler_pools.size() == 1);
    const auto* sampler_record = RecordPayload<SetPoolPayload>(sampler_pools[0]);
    CHECK(sampler_record->pool_serial == sampler_pool.serial);
    CHECK(sampler_record->base == sampler_pool.base);

    const auto binds = Find(list, Op::BindTextures);
    REQUIRE(binds.size() == 1);
    CHECK(TrailingArray<u64>(RecordPayload<BindHandlesPayload>(binds[0]))[0] ==
          ((4 + texture_pool.base) | (u64{3} + sampler_pool.base) << 20));
}

TEST_CASE("nvn hle bind: render targets", "[nvn]") {
    BindRig f;
    f.InitTexture(Rig::Obj(10), 0x0000);
    f.InitTexture(Rig::Obj(11), 0x4000);
    f.InitTexture(Rig::Obj(12), 0x8000);
    REQUIRE(f.rig.sink.textures.size() == 3);

    GuestTextureView view{};
    view.set_bits = GUEST_VIEW_SET_LEVELS | GUEST_VIEW_SET_LAYERS;
    view.base_level = 2;
    view.nlevels = 1;
    view.min_layer = 5;
    view.nlayers = 1;
    f.rig.memory.Put(SCRATCH + 0x100, view);
    f.rig.memory.Put(SCRATCH, std::array<u64, 2>{Rig::Obj(10), Rig::Obj(11)});
    f.rig.memory.Put(SCRATCH + 0x20, std::array<u64, 2>{SCRATCH + 0x100, 0});
    f.rig.Call("nvnCommandBufferSetRenderTargets",
               {I(BindRig::CMD), I(2), I(SCRATCH), I(SCRATCH + 0x20), I(Rig::Obj(12)), I(0)});
    // Depth only, no view arrays.
    f.rig.Call("nvnCommandBufferSetRenderTargets",
               {I(BindRig::CMD), I(0), I(0), I(0), I(Rig::Obj(12)), I(0)});
    // Colors only.
    f.rig.Call("nvnCommandBufferSetRenderTargets",
               {I(BindRig::CMD), I(1), I(SCRATCH), I(0), I(0), I(0)});
    // Counts outside 0..8 are clamped, as in nxnative.
    f.rig.Call("nvnCommandBufferSetRenderTargets",
               {I(BindRig::CMD), I(12), I(SCRATCH + 0x200), I(0), I(0), I(0)});
    f.rig.Call("nvnCommandBufferSetRenderTargets",
               {I(BindRig::CMD), I(0xFFFF'FFFFULL), I(SCRATCH), I(0), I(0), I(0)});

    const std::vector<u64>& list = f.Submit();
    const auto records = Find(list, Op::SetRenderTargets);
    REQUIRE(records.size() == 5);
    const u64 serial0 = f.rig.sink.textures[0].serial;
    const u64 serial1 = f.rig.sink.textures[1].serial;
    const u64 serial2 = f.rig.sink.textures[2].serial;

    const auto* full = RecordPayload<SetRenderTargetsPayload>(records[0]);
    CHECK(full->color_count == 2);
    CHECK(full->has_depth == 1);
    const RenderTargetRef* targets = TrailingArray<RenderTargetRef>(full);
    CHECK(targets[0].texture.serial == serial0);
    CHECK(targets[0].texture.level == 2);
    CHECK(targets[0].texture.layer == 5);
    CHECK(targets[0].view.set_levels == 1);
    CHECK(targets[0].view.base_level == 2);
    CHECK(targets[0].view.min_layer == 5);
    CHECK(targets[1].texture.serial == serial1);
    CHECK(targets[1].texture.level == 0);
    CHECK(targets[1].view.set_levels == 0);
    CHECK(targets[2].texture.serial == serial2);

    const auto* depth_only = RecordPayload<SetRenderTargetsPayload>(records[1]);
    CHECK(depth_only->color_count == 0);
    CHECK(depth_only->has_depth == 1);
    CHECK(TrailingArray<RenderTargetRef>(depth_only)[0].texture.serial == serial2);

    const auto* color_only = RecordPayload<SetRenderTargetsPayload>(records[2]);
    CHECK(color_only->color_count == 1);
    CHECK(color_only->has_depth == 0);
    CHECK(TrailingArray<RenderTargetRef>(color_only)[0].texture.serial == serial0);

    const auto* too_many = RecordPayload<SetRenderTargetsPayload>(records[3]);
    CHECK(too_many->color_count == 8);
    CHECK(too_many->has_depth == 0);
    for (u32 i = 0; i < 8; ++i) {
        CHECK(TrailingArray<RenderTargetRef>(too_many)[i].texture.serial == 0); // empty slots
    }
    const auto* negative = RecordPayload<SetRenderTargetsPayload>(records[4]);
    CHECK(negative->color_count == 0);
}

TEST_CASE("nvn hle bind: clear colors keep float, int and uint bit patterns", "[nvn]") {
    BindRig f;
    f.rig.memory.Put(SCRATCH, std::array<float, 4>{0.25f, 0.5f, -0.75f, 1.0f});
    f.rig.Call("nvnCommandBufferClearColor", {I(BindRig::CMD), I(3), I(SCRATCH), I(0xF)});
    f.rig.memory.Put(SCRATCH + 0x20, std::array<s32, 4>{-1, 2, -3, 0x7FFFFFFF});
    f.rig.Call("nvnCommandBufferClearColori", {I(BindRig::CMD), I(1), I(SCRATCH + 0x20), I(0x5)});
    f.rig.memory.Put(SCRATCH + 0x40, std::array<u32, 4>{0xFFFFFFFFu, 0x80000000u, 7, 0});
    f.rig.Call("nvnCommandBufferClearColorui", {I(BindRig::CMD), I(0), I(SCRATCH + 0x40), I(0x8)});
    // Unreadable values: nothing recorded.
    f.rig.Call("nvnCommandBufferClearColor", {I(BindRig::CMD), I(0), I(0), I(0xF)});

    const std::vector<u64>& list = f.Submit();
    const auto clears = Find(list, Op::ClearColor);
    REQUIRE(clears.size() == 3);

    const auto* floats = RecordPayload<ClearColorPayload>(clears[0]);
    CHECK(floats->index == 3);
    CHECK(floats->mask == 0xF);
    CHECK(floats->is_integer == 0);
    CHECK(floats->color[0] == Bits(0.25f));
    CHECK(floats->color[1] == Bits(0.5f));
    CHECK(floats->color[2] == Bits(-0.75f));
    CHECK(floats->color[3] == Bits(1.0f));

    const auto* ints = RecordPayload<ClearColorPayload>(clears[1]);
    CHECK(ints->index == 1);
    CHECK(ints->mask == 0x5);
    CHECK(ints->is_integer == 1);
    CHECK(static_cast<s32>(ints->color[0]) == -1);
    CHECK(static_cast<s32>(ints->color[1]) == 2);
    CHECK(static_cast<s32>(ints->color[2]) == -3);
    CHECK(static_cast<s32>(ints->color[3]) == 0x7FFFFFFF);

    const auto* uints = RecordPayload<ClearColorPayload>(clears[2]);
    CHECK(uints->index == 0);
    CHECK(uints->mask == 0x8);
    CHECK(uints->is_integer == 1);
    CHECK(uints->color[0] == 0xFFFFFFFFu);
    CHECK(uints->color[1] == 0x80000000u);
    CHECK(uints->color[2] == 7);
}

TEST_CASE("nvn hle bind: depth stencil clear decodes the boolean byte", "[nvn]") {
    BindRig f;
    f.rig.Call("nvnCommandBufferClearDepthStencil",
               {I(BindRig::CMD), F(0.5f), I(0xAB01), I(0x12), I(0xFF)});
    // NVNboolean is one byte: the garbage above bit 7 is not part of it.
    f.rig.Call("nvnCommandBufferClearDepthStencil",
               {I(BindRig::CMD), F(1.0f), I(0x100), I(0xFFFF'FFFFULL), I(0)});

    const std::vector<u64>& list = f.Submit();
    const auto clears = Find(list, Op::ClearDepthStencil);
    REQUIRE(clears.size() == 2);
    const auto* first = RecordPayload<ClearDepthStencilPayload>(clears[0]);
    CHECK(first->depth == 0.5f);
    CHECK(first->depth_mask == 1);
    CHECK(first->stencil == 0x12);
    CHECK(first->stencil_mask == 0xFF);
    const auto* second = RecordPayload<ClearDepthStencilPayload>(clears[1]);
    CHECK(second->depth == 1.0f);
    CHECK(second->depth_mask == 0);
    CHECK(second->stencil == -1);
    CHECK(second->stencil_mask == 0);
}

TEST_CASE("nvn hle bind: viewports, scissors and depth ranges", "[nvn]") {
    BindRig f;
    // The single forms take ints and record floats (viewport) or ints (scissor) at index 0.
    f.rig.Call("nvnCommandBufferSetViewport", {I(BindRig::CMD), I(10), I(0xFFFF'FFFFULL), I(1920), I(1080)});
    f.rig.Call("nvnCommandBufferSetScissor", {I(BindRig::CMD), I(0xFFFF'FFFBULL), I(6), I(7), I(8)});
    f.rig.Call("nvnCommandBufferSetDepthRange", {I(BindRig::CMD), F(0.25f), F(0.75f)});

    f.rig.memory.Put(SCRATCH, std::array<float, 8>{1.f, 2.f, 3.f, 4.f, 5.5f, 6.5f, 7.5f, 8.5f});
    f.rig.Call("nvnCommandBufferSetViewports", {I(BindRig::CMD), I(2), I(2), I(SCRATCH)});
    f.rig.memory.Put(SCRATCH + 0x40, std::array<s32, 8>{1, 2, 3, 4, -5, -6, 7, 8});
    f.rig.Call("nvnCommandBufferSetScissors", {I(BindRig::CMD), I(1), I(2), I(SCRATCH + 0x40)});
    f.rig.memory.Put(SCRATCH + 0x80, std::array<float, 4>{0.f, 0.5f, 0.125f, 1.f});
    f.rig.Call("nvnCommandBufferSetDepthRanges", {I(BindRig::CMD), I(3), I(2), I(SCRATCH + 0x80)});
    // Beyond the 16 viewports: dropped.
    f.rig.Call("nvnCommandBufferSetViewports", {I(BindRig::CMD), I(15), I(2), I(SCRATCH)});

    const RecorderStats stats = f.rig.recorder.Stats();
    const std::vector<u64>& list = f.Submit();
    CHECK(stats.bad_argument >= 1);

    const auto viewports = Find(list, Op::SetViewports);
    REQUIRE(viewports.size() == 2);
    const auto* single = RecordPayload<SetViewportsPayload>(viewports[0]);
    CHECK(single->first == 0);
    CHECK(single->count == 1);
    const ViewportRect& rect = TrailingArray<ViewportRect>(single)[0];
    CHECK(rect.x == 10.f);
    CHECK(rect.y == -1.f);
    CHECK(rect.width == 1920.f);
    CHECK(rect.height == 1080.f);
    const auto* multiple = RecordPayload<SetViewportsPayload>(viewports[1]);
    CHECK(multiple->first == 2);
    REQUIRE(multiple->count == 2);
    CHECK(TrailingArray<ViewportRect>(multiple)[1].x == 5.5f);
    CHECK(TrailingArray<ViewportRect>(multiple)[1].height == 8.5f);

    const auto scissors = Find(list, Op::SetScissors);
    REQUIRE(scissors.size() == 2);
    const auto* scissor = RecordPayload<SetScissorsPayload>(scissors[0]);
    CHECK(scissor->first == 0);
    CHECK(scissor->count == 1);
    CHECK(TrailingArray<ScissorRect>(scissor)[0].x == -5);
    CHECK(TrailingArray<ScissorRect>(scissor)[0].y == 6);
    CHECK(TrailingArray<ScissorRect>(scissor)[0].width == 7);
    CHECK(TrailingArray<ScissorRect>(scissor)[0].height == 8);
    const auto* scissor_array = RecordPayload<SetScissorsPayload>(scissors[1]);
    CHECK(scissor_array->first == 1);
    REQUIRE(scissor_array->count == 2);
    CHECK(TrailingArray<ScissorRect>(scissor_array)[1].x == -5);
    CHECK(TrailingArray<ScissorRect>(scissor_array)[1].y == -6);

    const auto ranges = Find(list, Op::SetDepthRanges);
    REQUIRE(ranges.size() == 2);
    const auto* range = RecordPayload<SetDepthRangesPayload>(ranges[0]);
    CHECK(range->first == 0);
    CHECK(range->count == 1);
    CHECK(TrailingArray<DepthRange>(range)[0].near_plane == 0.25f);
    CHECK(TrailingArray<DepthRange>(range)[0].far_plane == 0.75f);
    const auto* range_array = RecordPayload<SetDepthRangesPayload>(ranges[1]);
    CHECK(range_array->first == 3);
    REQUIRE(range_array->count == 2);
    CHECK(TrailingArray<DepthRange>(range_array)[1].near_plane == 0.125f);
    CHECK(TrailingArray<DepthRange>(range_array)[1].far_plane == 1.f);
}

TEST_CASE("nvn hle bind: viewport swizzles narrow the 32-bit enums to one byte", "[nvn]") {
    BindRig f;
    // Two viewports, x/y/z/w each: identity for the first, flipped y / negated z, w for the second.
    f.rig.memory.Put(SCRATCH, std::array<u32, 8>{0, 2, 4, 6, 1, 3, 5, 7});
    f.rig.Call("nvnCommandBufferSetViewportSwizzles", {I(BindRig::CMD), I(1), I(2), I(SCRATCH)});
    // Out of range: dropped.
    f.rig.Call("nvnCommandBufferSetViewportSwizzles", {I(BindRig::CMD), I(15), I(2), I(SCRATCH)});

    const std::vector<u64>& list = f.Submit();
    const auto swizzles = Find(list, Op::SetViewportSwizzles);
    REQUIRE(swizzles.size() == 1);
    CHECK(swizzles[0]->size == 8 + 8 + 8);
    const auto* payload = RecordPayload<SetViewportSwizzlesPayload>(swizzles[0]);
    CHECK(payload->first == 1);
    REQUIRE(payload->count == 2);
    const u8* bytes = TrailingArray<u8>(payload);
    const std::array<u8, 8> expected{0, 2, 4, 6, 1, 3, 5, 7};
    CHECK(std::memcmp(bytes, expected.data(), 8) == 0);
}

TEST_CASE("nvn hle bind: stencil face, which and value", "[nvn]") {
    BindRig f;
    f.rig.Call("nvnCommandBufferSetStencilValueMask", {I(BindRig::CMD), I(1), I(0x7F)});
    f.rig.Call("nvnCommandBufferSetStencilMask", {I(BindRig::CMD), I(2), I(0xF0)});
    f.rig.Call("nvnCommandBufferSetStencilRef", {I(BindRig::CMD), I(3), I(0xFFFF'FFFFULL)});

    const std::vector<u64>& list = f.Submit();
    const auto records = Find(list, Op::SetStencil);
    REQUIRE(records.size() == 3);
    const std::array<std::array<s32, 3>, 3> expected{{{1, 0, 0x7F}, {2, 1, 0xF0}, {3, 2, -1}}};
    for (size_t i = 0; i < 3; ++i) {
        const auto* payload = RecordPayload<SetStencilPayload>(records[i]);
        CHECK(payload->face == expected[i][0]);
        CHECK(payload->which == static_cast<u32>(expected[i][1]));
        CHECK(payload->value == expected[i][2]);
    }
}

TEST_CASE("nvn hle bind: polygon offset floats and blend color", "[nvn]") {
    BindRig f;
    // Three floats after the command buffer pointer: v0, v1, v2.
    f.rig.Call("nvnCommandBufferSetPolygonOffsetClamp",
               {I(BindRig::CMD), F(1.5f), F(-2.0f), F(0.125f)});
    f.rig.memory.Put(SCRATCH, std::array<float, 4>{0.1f, 0.2f, 0.3f, 0.4f});
    f.rig.Call("nvnCommandBufferSetBlendColor", {I(BindRig::CMD), I(SCRATCH)});
    f.rig.memory.Put(SCRATCH, std::array<float, 4>{});

    const std::vector<u64>& list = f.Submit();
    const auto offsets = Find(list, Op::SetPolygonOffset);
    REQUIRE(offsets.size() == 1);
    const auto* offset = RecordPayload<SetPolygonOffsetPayload>(offsets[0]);
    CHECK(offset->factor == 1.5f);
    CHECK(offset->units == -2.0f);
    CHECK(offset->clamp == 0.125f);

    const auto colors = Find(list, Op::SetBlendColor);
    REQUIRE(colors.size() == 1);
    const auto* color = RecordPayload<SetBlendColorPayload>(colors[0]);
    CHECK(color->color[0] == 0.1f);
    CHECK(color->color[1] == 0.2f);
    CHECK(color->color[2] == 0.3f);
    CHECK(color->color[3] == 0.4f);
}

TEST_CASE("nvn hle bind: rasterizer switches", "[nvn]") {
    BindRig f;
    // NVNboolean arguments: only the low byte counts.
    f.rig.Call("nvnCommandBufferSetDepthClamp", {I(BindRig::CMD), I(0x101)});
    f.rig.Call("nvnCommandBufferSetDepthClamp", {I(BindRig::CMD), I(0x100)});
    f.rig.Call("nvnCommandBufferSetAlphaRef", {I(BindRig::CMD), F(0.5f)});
    f.rig.Call("nvnCommandBufferSetPrimitiveRestart", {I(BindRig::CMD), I(1), I(0xFFFF)});
    f.rig.Call("nvnCommandBufferSetRasterizerDiscard", {I(BindRig::CMD), I(1)});
    f.rig.Call("nvnCommandBufferSetSampleMask", {I(BindRig::CMD), I(0xF0F0)});
    f.rig.Call("nvnCommandBufferSetPointSize", {I(BindRig::CMD), F(4.0f)});
    f.rig.Call("nvnCommandBufferSetLineWidth", {I(BindRig::CMD), F(2.5f)});
    f.rig.Call("nvnCommandBufferSetRenderEnable", {I(BindRig::CMD), I(0)});
    f.rig.Call("nvnCommandBufferSetRenderEnable", {I(BindRig::CMD), I(1)});
    f.rig.Call("nvnCommandBufferSetRenderEnableConditional",
               {I(BindRig::CMD), I(1), I(Rig::STORAGE + 0x80)});

    const std::vector<u64>& list = f.Submit();
    const auto clamps = Find(list, Op::SetDepthClamp);
    REQUIRE(clamps.size() == 2);
    CHECK(RecordPayload<EnablePayload>(clamps[0])->enable == 1);
    CHECK(RecordPayload<EnablePayload>(clamps[1])->enable == 0);

    CHECK(RecordPayload<SetAlphaRefPayload>(Find(list, Op::SetAlphaRef).at(0))->reference == 0.5f);

    const auto* restart =
        RecordPayload<SetPrimitiveRestartPayload>(Find(list, Op::SetPrimitiveRestart).at(0));
    CHECK(restart->enable == 1);
    CHECK(restart->index == 0xFFFF);

    CHECK(RecordPayload<EnablePayload>(Find(list, Op::SetRasterizerDiscard).at(0))->enable == 1);
    CHECK(RecordPayload<U32Payload>(Find(list, Op::SetSampleMask).at(0))->value == 0xF0F0);
    CHECK(RecordPayload<F32Payload>(Find(list, Op::SetPointSize).at(0))->value == 4.0f);
    CHECK(RecordPayload<F32Payload>(Find(list, Op::SetLineWidth).at(0))->value == 2.5f);

    const auto conditions = Find(list, Op::SetRenderCondition);
    REQUIRE(conditions.size() == 3);
    CHECK(RecordPayload<SetRenderConditionPayload>(conditions[0])->kind == 2); // render disabled
    CHECK(RecordPayload<SetRenderConditionPayload>(conditions[1])->kind == 1); // render enabled
    const auto* conditional = RecordPayload<SetRenderConditionPayload>(conditions[2]);
    CHECK(conditional->kind == 0);
    CHECK(conditional->mode == 1);
    CHECK(conditional->address == Rig::STORAGE + 0x80);
}

TEST_CASE("nvn hle bind: the provoking vertex mode is the SET_PROVOKING_VERTEX bit", "[nvn]") {
    BindRig f;
    // The SDK writes the mode into the method's data unmasked; the register is bit 0 (1 = last).
    f.rig.Call("nvnCommandBufferSetProvokingVertexMode", {I(BindRig::CMD), I(0)});
    f.rig.Call("nvnCommandBufferSetProvokingVertexMode", {I(BindRig::CMD), I(1)});
    f.rig.Call("nvnCommandBufferSetProvokingVertexMode", {I(BindRig::CMD), I(2)});

    const std::vector<u64>& list = f.Submit();
    const auto modes = Find(list, Op::SetProvokingVertex);
    REQUIRE(modes.size() == 3);
    CHECK(RecordPayload<U32Payload>(modes[0])->value == 0);
    CHECK(RecordPayload<U32Payload>(modes[1])->value == 1);
    CHECK(RecordPayload<U32Payload>(modes[2])->value == 0);
}

TEST_CASE("nvn hle bind: accepted-and-ignored switches record nothing", "[nvn]") {
    BindRig f;
    f.rig.Call("nvnCommandBufferSetDepthBounds", {I(BindRig::CMD), I(1), F(0.1f), F(0.9f)});
    f.rig.Call("nvnCommandBufferSetConservativeRasterEnable", {I(BindRig::CMD), I(1)});
    f.rig.Call("nvnCommandBufferSetGeometryProcessingMode", {I(BindRig::CMD), I(2)});
    f.rig.Call("nvnCommandBufferSetComputeOverlapMode", {I(BindRig::CMD), I(1)});
    f.rig.Call("nvnCommandBufferSetDepthClamp", {I(BindRig::CMD), I(1)});

    const std::vector<u64>& list = f.Submit();
    CHECK(RecordCount(list) == 1); // only the depth clamp
    const auto clamps = Find(list, Op::SetDepthClamp);
    REQUIRE(clamps.size() == 1);
    CHECK(RecordPayload<EnablePayload>(clamps[0])->enable == 1);
}

TEST_CASE("nvn hle bind: binds outside a recording are counted and dropped", "[nvn]") {
    BindRig f;
    f.rig.recorder.EndRecording(BindRig::CMD);
    const RecorderStats before = f.rig.recorder.Stats();
    f.rig.Call("nvnCommandBufferSetDepthClamp", {I(BindRig::CMD), I(1)});
    f.rig.Call("nvnCommandBufferBindVertexBuffer", {I(BindRig::CMD), I(0), I(Rig::STORAGE), I(16)});
    // Not a command buffer at all.
    f.rig.Call("nvnCommandBufferSetDepthClamp", {I(Rig::Obj(60)), I(1)});
    const RecorderStats after = f.rig.recorder.Stats();
    CHECK(after.records_dropped == before.records_dropped + 3);
    CHECK(after.records_recorded == before.records_recorded);
}

namespace {

/// Gives the program at `program_va` one 16-byte NVN2 code area per stage of `stage_mask`.
void SetProgramStages(BindRig& f, u64 program_va, u32 stage_mask) {
    std::vector<ShaderStageInput> inputs;
    for (u16 stage = 0; stage < 6; ++stage) {
        if (((stage_mask >> stage) & 1) == 0) {
            continue;
        }
        std::array<u8, 0x100> control{};
        const u32 magic = NVN2_CONTROL_MAGIC;
        const u32 version = NVN2_CONTROL_VERSION;
        const u32 code_size = 16;
        std::memcpy(control.data(), &magic, 4);
        std::memcpy(control.data() + 4, &version, 4);
        std::memcpy(control.data() + 0x44, &code_size, 4);
        std::memcpy(control.data() + 0x56, &stage, 2);
        const u64 control_va = SCRATCH + 0x400 + stage * 0x100;
        const u64 code_va = Rig::STORAGE + 0x9000 + stage * 0x100;
        f.rig.memory.Put(control_va, control);
        std::array<u8, 16> code{};
        code.fill(static_cast<u8>(0x10 + stage));
        f.rig.memory.Put(code_va, code);
        inputs.push_back(ShaderStageInput{code_va, control_va, -1});
    }
    REQUIRE(f.rig.recorder.ProgramSetShaders(program_va, inputs));
}

} // namespace

TEST_CASE("nvn hle bind: BindProgram follows the driver's stage rules", "[nvn]") {
    BindRig f;
    const u64 graphics = Rig::Obj(40);
    const u64 compute = Rig::Obj(41);
    REQUIRE(f.rig.recorder.ProgramInitialize(graphics));
    REQUIRE(f.rig.recorder.ProgramInitialize(compute));
    SetProgramStages(f, graphics, 0x3); // vertex + fragment
    SetProgramStages(f, compute, 0x20);

    // A mask narrower than the program: the driver copies the whole program, both stages bind.
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(graphics), I(0x1)});
    // Mask stages the program lacks are recorded with it (the replay finds no code: unbound).
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(graphics), I(0x1C)});
    // An empty mask does nothing, with or without a program.
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(graphics), I(0)});
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(0), I(0)});
    // NULL unbinds the mask's stages.
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(0), I(0x1F)});
    // Compute in the mask: compute path, no widening; the mask's graphics stages unbind.
    f.rig.Call("nvnCommandBufferBindProgram", {I(BindRig::CMD), I(compute), I(0x21)});

    const RecorderStats stats = f.rig.recorder.Stats();
    const std::vector<u64>& list = f.Submit();
    CHECK(stats.records_dropped == 0);
    const auto programs = Find(list, Op::BindProgram);
    REQUIRE(programs.size() == 4);
    const auto* narrow = RecordPayload<BindProgramPayload>(programs[0]);
    CHECK(narrow->program_serial != 0);
    CHECK(narrow->stages == 0x3);
    const auto* lacking = RecordPayload<BindProgramPayload>(programs[1]);
    CHECK(lacking->program_serial == narrow->program_serial);
    CHECK(lacking->stages == 0x1F);
    const auto* null_program = RecordPayload<BindProgramPayload>(programs[2]);
    CHECK(null_program->program_serial == 0);
    CHECK(null_program->stages == 0x1F);
    const auto* compute_bind = RecordPayload<BindProgramPayload>(programs[3]);
    CHECK(compute_bind->program_serial != 0);
    CHECK(compute_bind->program_serial != narrow->program_serial);
    CHECK(compute_bind->stages == 0x21);
}

TEST_CASE("nvn hle bind: conditional render modes other than 0 and 1 do nothing", "[nvn]") {
    BindRig f;
    f.rig.Call("nvnCommandBufferSetRenderEnableConditional",
               {I(BindRig::CMD), I(2), I(Rig::STORAGE + 0x80)});
    f.rig.Call("nvnCommandBufferSetRenderEnableConditional",
               {I(BindRig::CMD), I(0xFFFF'FFFFULL), I(Rig::STORAGE + 0x80)});
    f.rig.Call("nvnCommandBufferSetRenderEnableConditional",
               {I(BindRig::CMD), I(0), I(Rig::STORAGE + 0x80)});

    const std::vector<u64>& list = f.Submit();
    const auto conditions = Find(list, Op::SetRenderCondition);
    REQUIRE(conditions.size() == 1);
    const auto* conditional = RecordPayload<SetRenderConditionPayload>(conditions[0]);
    CHECK(conditional->kind == 0);
    CHECK(conditional->mode == 0);
}

TEST_CASE("nvn hle bind: vertex stream strides keep the driver's 12 bits", "[nvn]") {
    BindRig f;
    f.rig.memory.Put(SCRATCH, std::array<u32, 6>{0x1010, 0, 0xFFF, 2, 0xFFFF'F004u, 0});
    f.rig.Call("nvnCommandBufferBindVertexStreamState", {I(BindRig::CMD), I(3), I(SCRATCH)});
    // More streams than NVN has: dropped and counted.
    f.rig.Call("nvnCommandBufferBindVertexStreamState", {I(BindRig::CMD), I(17), I(SCRATCH)});

    const RecorderStats stats = f.rig.recorder.Stats();
    const std::vector<u64>& list = f.Submit();
    CHECK(stats.bad_argument >= 1);
    const auto streams = Find(list, Op::BindVertexStreams);
    REQUIRE(streams.size() == 1);
    const auto* stream = RecordPayload<BindVertexStreamsPayload>(streams[0]);
    CHECK(stream->first == 0);
    REQUIRE(stream->count == 3);
    const VertexStreamState* states = TrailingArray<VertexStreamState>(stream);
    CHECK(states[0].stride == 0x10);
    CHECK(states[0].divisor == 0);
    CHECK(states[1].stride == 0xFFF);
    CHECK(states[1].divisor == 2);
    CHECK(states[2].stride == 0x4);
}

TEST_CASE("nvn hle bind: uniform buffer sizes are the driver's 256-byte units", "[nvn]") {
    BindRig f;
    f.rig.Call("nvnCommandBufferBindUniformBuffer",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(0), I(Rig::STORAGE + 0x100), I(0x44)});
    f.rig.Call("nvnCommandBufferBindUniformBuffer",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(1), I(Rig::STORAGE + 0x200), I(0)});
    f.rig.Call("nvnCommandBufferBindUniformBuffer",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(2), I(Rig::STORAGE + 0x300), I(0x200)});
    f.rig.memory.Put(SCRATCH, std::array<u64, 6>{Rig::STORAGE + 0x400, 0x101, Rig::STORAGE + 0x500,
                                                  0, Rig::STORAGE + 0x600, 0xFF});
    f.rig.Call("nvnCommandBufferBindUniformBuffers",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(0), I(3), I(SCRATCH)});
    // Deferred: the sizes read at submit are rounded too.
    f.rig.memory.Put(SCRATCH + 0x100, std::array<u64, 2>{Rig::STORAGE + 0x700, 0x10});
    f.rig.Call("nvnCommandBufferBindUniformBuffersDeferred",
               {I(BindRig::CMD), I(STAGE_FRAGMENT), I(4), I(1), I(SCRATCH + 0x100)});
    f.rig.memory.Put(SCRATCH + 0x100, std::array<u64, 2>{Rig::STORAGE + 0x800, 0x1F01});
    // Storage buffers keep the raw size (the driver passes it through).
    f.rig.Call("nvnCommandBufferBindStorageBuffer",
               {I(BindRig::CMD), I(STAGE_VERTEX), I(0), I(Rig::STORAGE + 0x900), I(0x44)});

    const std::vector<u64>& list = f.Submit();
    const auto uniforms = Find(list, Op::BindUniformBuffers);
    REQUIRE(uniforms.size() == 5);
    const std::array<u64, 3> single{0x100, 0, 0x200};
    for (size_t i = 0; i < single.size(); ++i) {
        const auto* payload = RecordPayload<BindBuffersPayload>(uniforms[i]);
        CHECK(payload->first == i);
        CHECK(TrailingArray<BufferRange>(payload)[0].size == single[i]);
    }
    const auto* many = RecordPayload<BindBuffersPayload>(uniforms[3]);
    REQUIRE(many->count == 3);
    CHECK(TrailingArray<BufferRange>(many)[0].size == 0x200);
    CHECK(TrailingArray<BufferRange>(many)[1].size == 0);
    CHECK(TrailingArray<BufferRange>(many)[2].size == 0x100);
    const auto* deferred = RecordPayload<BindBuffersPayload>(uniforms[4]);
    REQUIRE(deferred->count == 1);
    CHECK(TrailingArray<BufferRange>(deferred)[0].address == Rig::STORAGE + 0x800);
    CHECK(TrailingArray<BufferRange>(deferred)[0].size == 0x2000);

    const auto storages = Find(list, Op::BindStorageBuffers);
    REQUIRE(storages.size() == 1);
    CHECK(TrailingArray<BufferRange>(RecordPayload<BindBuffersPayload>(storages[0]))[0].size ==
          0x44);
}
