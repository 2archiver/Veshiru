// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Rules of the recorded-state tracking of an open render pass (nvn_pass_state.h): what a draw may
// skip, and what forces it to record again; and which vertex ranges a pass stages
// (nvn_vertex_stage.h). Pure logic, no device.

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_barrier.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_pass_state.h"
#include "video_core/renderer_vulkan/nvn/nvn_vertex_stage.h"

namespace {

using Vulkan::Nvn::PassBoundState;

template <typename T>
T Handle(std::uintptr_t value) {
    return reinterpret_cast<T>(value);
}

struct Descriptors {
    std::vector<PassBoundState::WriteSig> writes{{0, 6, 1}, {1, 1, 2}};
    std::vector<VkDescriptorBufferInfo> buffers{{Handle<VkBuffer>(0x10), 0, 256}};
    std::vector<PassBoundState::ImageSig> images{
        {Handle<VkSampler>(0x20), Handle<VkImageView>(0x30),
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {Handle<VkSampler>(0x21), Handle<VkImageView>(0x31),
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
};

} // namespace

TEST_CASE("NVN dynamic values retain exact float bits and the force-record path",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::ReplaceDynamicState;
    float width = 0.0f;
    CHECK_FALSE(ReplaceDynamicState(width, 0.0f, true));
    CHECK(ReplaceDynamicState(width, -0.0f, true));
    CHECK(std::bit_cast<u32>(width) == 0x80000000u);
    CHECK_FALSE(ReplaceDynamicState(width, -0.0f, true));
    CHECK(ReplaceDynamicState(width, -0.0f, false));
    const float nan = std::bit_cast<float>(0x7fc00001u);
    CHECK(ReplaceDynamicState(width, nan, true));
    CHECK_FALSE(ReplaceDynamicState(width, nan, true));
    CHECK(ReplaceDynamicState(width, std::bit_cast<float>(0x7fc00002u), true));
    VideoCommon::Nvn::ViewportRect current{0.0f, 0.0f, 640.0f, 480.0f};
    auto next = current;
    CHECK_FALSE(ReplaceDynamicState(current, next, true));
    next.width = 1280.0f;
    CHECK(ReplaceDynamicState(current, next, true));
    CHECK(current.width == 1280.0f);
    CHECK_FALSE(ReplaceDynamicState(current, next, true));
    CHECK(ReplaceDynamicState(current, next, false));
}

TEST_CASE("NVN pass state: descriptors are skipped only when identical on the same layout",
          "[nvn][pass_state]") {
    PassBoundState state;
    const auto layout = Handle<VkPipelineLayout>(0x100);
    Descriptors d;
    state.OnPipelineBind(layout, true);
    REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images)); // first push
    REQUIRE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images));       // same again

    SECTION("a changed buffer offset records again") {
        d.buffers[0].offset = 256;
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images));
        REQUIRE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images));
    }
    SECTION("a changed buffer range, view, sampler or layout records again") {
        auto buffers = d.buffers;
        buffers[0].range = 512;
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, buffers, d.images));
        auto images = d.images;
        images[1].view = Handle<VkImageView>(0x99);
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, buffers, images));
        images[1].sampler = Handle<VkSampler>(0x98);
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, buffers, images));
        images[1].layout = VK_IMAGE_LAYOUT_GENERAL;
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, buffers, images));
    }
    SECTION("a different write shape records again") {
        auto writes = d.writes;
        writes[1].count = 3;
        REQUIRE_FALSE(state.TakeDescriptors(layout, writes, d.buffers, d.images));
    }
    SECTION("a different pipeline layout disturbs the push set") {
        const auto other = Handle<VkPipelineLayout>(0x200);
        state.OnPipelineBind(other, true);
        REQUIRE_FALSE(state.TakeDescriptors(other, d.writes, d.buffers, d.images));
        // Back to the first layout: its set was disturbed by the other layout's push.
        state.OnPipelineBind(layout, true);
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images));
    }
    SECTION("Reset (pass begin/end, command buffer start) forgets everything") {
        state.Reset();
        state.OnPipelineBind(layout, true);
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images));
    }
    SECTION("an uncomparable push forgets the descriptors") {
        state.ForgetDescriptors();
        REQUIRE_FALSE(state.TakeDescriptors(layout, d.writes, d.buffers, d.images));
    }
}

TEST_CASE("NVN pass state: a full flip target list replaces the entry used longest ago",
          "[nvn][pass_state]") {
    using List = Vulkan::Nvn::FlipTargetList<int>;
    List list;
    std::vector<std::shared_ptr<int>> images;
    for (size_t i = 0; i < List::CAPACITY; ++i) {
        images.push_back(std::make_shared<int>(static_cast<int>(i)));
        CHECK_FALSE(list.Note(images.back(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    }
    REQUIRE(list.Size() == List::CAPACITY);
    // Noting a known image again updates it in place.
    CHECK_FALSE(list.Note(images[0], VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL));
    CHECK(list[0].sampled_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    // Image 1 is noted longest ago unless a pass begin found it rendered to: then image 2 is.
    list.Touch(1);
    const auto fresh = std::make_shared<int>(-1);
    CHECK(list.Note(fresh, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    CHECK(list.Size() == List::CAPACITY);
    CHECK(list[2].target.lock() == fresh);
    CHECK(list[1].target.lock() == images[1]);
    CHECK(list[0].target.lock() == images[0]);
    // A full list learns every new image (the old code dropped it).
    const auto another = std::make_shared<int>(-2);
    CHECK(list.Note(another, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    CHECK(list[3].target.lock() == another);
    // Erase moves the last entry into the gap.
    list.Erase(0);
    CHECK(list.Size() == List::CAPACITY - 1);
    CHECK(list[0].target.lock() == images[List::CAPACITY - 1]);
}

TEST_CASE("NVN pass state: push constants survive layouts that keep the range",
          "[nvn][pass_state]") {
    PassBoundState state;
    const auto a = Handle<VkPipelineLayout>(0x100);
    const auto b = Handle<VkPipelineLayout>(0x200);
    state.OnPipelineBind(a, true);
    REQUIRE_FALSE(state.TakePushConstants(false));
    REQUIRE(state.TakePushConstants(false));
    // The render area was never pushed: a pipeline that reads it records it, once.
    REQUIRE_FALSE(state.TakePushConstants(true));
    REQUIRE(state.TakePushConstants(true));
    REQUIRE(state.TakePushConstants(false));
    // Another layout with the same push constant range keeps the values.
    state.OnPipelineBind(b, true);
    REQUIRE(state.TakePushConstants(true));
    // A layout without the range disturbs them.
    state.OnPipelineBind(a, false);
    state.OnPipelineBind(b, true);
    REQUIRE_FALSE(state.TakePushConstants(false));
    state.Reset();
    state.OnPipelineBind(a, true);
    REQUIRE_FALSE(state.TakePushConstants(false));
}

TEST_CASE("NVN pass state: a feedback scratch serves only what it copied", "[nvn][pass_state]") {
    using Vulkan::Nvn::SubresourceSpan;
    const SubresourceSpan copied{0, 5, 0, 1};
    REQUIRE(copied.Covers({0, 5, 0, 1}));
    REQUIRE(copied.Covers({1, 2, 0, 1}));
    REQUIRE_FALSE(copied.Covers({0, 6, 0, 1}));
    REQUIRE_FALSE(copied.Covers({4, 2, 0, 1}));
    REQUIRE_FALSE(copied.Covers({0, 5, 0, 2}));
    const SubresourceSpan cube{1, 3, 2, 6};
    REQUIRE(cube.Covers({1, 3, 2, 6}));
    REQUIRE_FALSE(cube.Covers({0, 3, 2, 6}));
    REQUIRE_FALSE(cube.Covers({1, 3, 1, 6}));
    REQUIRE_FALSE(cube.Covers({1, 3, 2, 7}));
}

TEST_CASE("NVN pass state: a mip chain copies the freshly rendered mip, repeat draws do not",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::FeedbackEpoch;
    using Vulkan::Nvn::FeedbackSnapshot;
    using Vulkan::Nvn::SubresourceSpan;
    const auto texture = std::make_shared<int>(0);
    const auto other_texture = std::make_shared<int>(1);
    const int context_a = 0;
    const int context_b = 0;
    const SubresourceSpan all_mips{0, 5, 0, 1}; // the view DK samples: every mip of the chain
    FeedbackEpoch epoch;
    FeedbackSnapshot<int> scratch;
    // Draws are asked in order; a copy is made (and recorded) when the scratch cannot serve.
    const auto draw = [&](const std::shared_ptr<int>& source, const void* context) {
        if (scratch.Serves(source, context, epoch.value, all_mips)) {
            return false;
        }
        scratch.Record(source, context, epoch.value, all_mips);
        return true;
    };

    // Pass for mip 0 (targets: level 0): the first draw copies, the next ones in the same
    // SetRenderTargets epoch, with no Barrier between, do not (each one used to).
    epoch.OnTargetChange();
    epoch.OnPassEnd(false);
    REQUIRE(draw(texture, &context_a));
    epoch.OnPassEnd(true); // the break of the feedback copy itself
    REQUIRE_FALSE(draw(texture, &context_a));
    REQUIRE_FALSE(draw(texture, &context_a));
    REQUIRE_FALSE(draw(texture, &context_a));

    // SetRenderTargets to mip 1: mip 0 was just rendered, the copy must include it.
    epoch.OnTargetChange();
    epoch.OnPassEnd(false);
    REQUIRE(draw(texture, &context_a));
    epoch.OnPassEnd(true);
    REQUIRE_FALSE(draw(texture, &context_a));

    // A Barrier orders the guest's writes against the texture: copy again.
    epoch.OnBarrier();
    REQUIRE(draw(texture, &context_a));
    REQUIRE_FALSE(draw(texture, &context_a));

    // Any other pass end (compute, copy, clear, upload) too.
    epoch.OnPassEnd(false);
    REQUIRE(draw(texture, &context_a));

    // Another source, another context, a wider range: the scratch does not serve them.
    REQUIRE(draw(other_texture, &context_a));
    REQUIRE(draw(texture, &context_a));
    REQUIRE(draw(texture, &context_b));
    REQUIRE_FALSE(scratch.Serves(texture, &context_b, epoch.value, {0, 6, 0, 1}));
    // A source that was destroyed and replaced does not match an expired snapshot.
    {
        const auto destroyed = std::make_shared<int>(2);
        scratch.Record(destroyed, &context_a, epoch.value, all_mips);
    }
    REQUIRE(scratch.from.expired());
    const auto replacement = std::make_shared<int>(0);
    REQUIRE_FALSE(scratch.Serves(replacement, &context_a, epoch.value, all_mips));
}

TEST_CASE("NVN pass state: vertex and index bindings", "[nvn][pass_state]") {
    PassBoundState state;
    const auto layout = Handle<VkPipelineLayout>(0x100);
    state.OnPipelineBind(layout, false);
    REQUIRE_FALSE(state.TakeStream(0, Handle<VkBuffer>(0x1), 0));
    REQUIRE_FALSE(state.TakeStream(2, Handle<VkBuffer>(0x2), 64));
    REQUIRE(state.TakeStream(0, Handle<VkBuffer>(0x1), 0));
    REQUIRE(state.TakeStream(2, Handle<VkBuffer>(0x2), 64));
    REQUIRE_FALSE(state.TakeStream(2, Handle<VkBuffer>(0x2), 128)); // offset differs
    REQUIRE_FALSE(state.TakeStream(0, Handle<VkBuffer>(0x3), 0));   // buffer differs
    // Vertex bindings survive a pipeline bind.
    state.OnPipelineBind(Handle<VkPipelineLayout>(0x300), true);
    REQUIRE(state.TakeStream(0, Handle<VkBuffer>(0x3), 0));

    REQUIRE_FALSE(state.TakeIndexBuffer(Handle<VkBuffer>(0x5), 16, VK_INDEX_TYPE_UINT16));
    REQUIRE(state.TakeIndexBuffer(Handle<VkBuffer>(0x5), 16, VK_INDEX_TYPE_UINT16));
    REQUIRE_FALSE(state.TakeIndexBuffer(Handle<VkBuffer>(0x5), 16, VK_INDEX_TYPE_UINT32));
    REQUIRE_FALSE(state.TakeIndexBuffer(Handle<VkBuffer>(0x5), 32, VK_INDEX_TYPE_UINT32));
    // Generated indices bound their own buffer behind the tracker's back.
    state.ForgetIndexBuffer();
    REQUIRE_FALSE(state.TakeIndexBuffer(Handle<VkBuffer>(0x5), 32, VK_INDEX_TYPE_UINT32));

    state.Reset();
    REQUIRE_FALSE(state.TakeStream(0, Handle<VkBuffer>(0x3), 0));
    REQUIRE_FALSE(state.TakeIndexBuffer(Handle<VkBuffer>(0x5), 32, VK_INDEX_TYPE_UINT32));
}

namespace {

namespace VC = VideoCommon::Nvn;

/// A command list in the recorder's format: header word, payload, trailing ranges, End.
struct Records {
    std::vector<u64> words;

    template <typename Payload>
    void Add(VC::Op op, const Payload& payload,
             std::initializer_list<VC::BufferRange> ranges = {}) {
        const size_t bytes =
            sizeof(VC::RecordHeader) + sizeof(Payload) + ranges.size() * sizeof(VC::BufferRange);
        const size_t size = (bytes + 7) / 8 * 8;
        const size_t at = words.size();
        words.resize(at + size / 8, 0);
        auto* base = reinterpret_cast<u8*>(&words[at]);
        const VC::RecordHeader header{static_cast<u16>(op), 0, static_cast<u32>(size)};
        std::memcpy(base, &header, sizeof(header));
        std::memcpy(base + sizeof(header), &payload, sizeof(Payload));
        size_t offset = sizeof(header) + sizeof(Payload);
        for (const VC::BufferRange& range : ranges) {
            std::memcpy(base + offset, &range, sizeof(range));
            offset += sizeof(range);
        }
    }
    void BindVertex(u32 first, std::initializer_list<VC::BufferRange> ranges) {
        Add(VC::Op::BindVertexBuffers,
            VC::BindVertexBuffersPayload{first, static_cast<u32>(ranges.size())}, ranges);
    }
    void Draws(u32 count) {
        for (u32 i = 0; i < count; ++i) {
            Add(VC::Op::DrawIndexed, VC::DrawIndexedPayload{});
        }
    }
    void DrawIndexed(u32 mode, u32 type, u32 count, u64 address, u64 indirect = 0) {
        VC::DrawIndexedPayload payload{};
        payload.mode = mode;
        payload.index_type = type;
        payload.count = count;
        payload.index_address = address;
        payload.indirect = indirect;
        Add(VC::Op::DrawIndexed, payload);
    }
    void End() {
        Add(VC::Op::End, u64{0});
    }
};

/// Draw-path ops stay in the pass, a dispatch does not (DrawContext::OpKeepsPass).
bool KeepsPass(VC::Op op) {
    return op != VC::Op::Dispatch;
}

constexpr Vulkan::Nvn::VertexStageLimits LIMITS{
    .min_draws = 2, .max_range_bytes = 0x1000, .max_total_bytes = 0x3000};

constexpr Vulkan::Nvn::IndexStageFilter NO_INDICES{};

} // namespace

TEST_CASE("NVN vertex staging: the pass's ranges up to the first op that ends it",
          "[nvn][pass_state]") {
    Records list;
    list.BindVertex(0, {{0x10000, 0x400}, {0x20000, 0x200}});
    list.Draws(2);
    list.BindVertex(1, {{0x30000, 0x100}});
    list.Draws(1);
    list.Add(VC::Op::Dispatch, VC::DispatchPayload{1, 1, 1});
    list.BindVertex(0, {{0x40000, 0x100}}); // after the dispatch: another pass
    list.Draws(5);
    list.End();

    const std::array<VC::BufferRange, 2> bound{{{0x50000, 0x80}, {0, 0}}};
    Vulkan::Nvn::VertexStageScan scan;
    Vulkan::Nvn::ScanPassVertexRanges(list.words, bound, LIMITS, NO_INDICES, KeepsPass, scan);
    REQUIRE(scan.draws == 3);
    REQUIRE(scan.ranges.size() == 4);
    CHECK(scan.ranges[0].address == 0x50000); // bound when the pass begins
    CHECK(scan.ranges[1].address == 0x10000);
    CHECK(scan.ranges[2].address == 0x20000);
    CHECK(scan.ranges[3].address == 0x30000);
    CHECK(scan.bytes == 0x80 + 0x400 + 0x200 + 0x100);
}

TEST_CASE("NVN vertex staging: one copy per address, the largest size, within the limits",
          "[nvn][pass_state]") {
    Records list;
    list.BindVertex(0, {{0x10000, 0x100}});
    list.Draws(1);
    list.BindVertex(0, {{0x10000, 0x300}}); // the same start, bound larger
    list.Draws(1);
    list.BindVertex(0, {{0x10000, 0x200}});  // smaller again: the copy already covers it
    list.BindVertex(1, {{0x20000, 0x2000}}); // above max_range_bytes: stays in host memory
    list.BindVertex(2, {{0x30000, 0}});      // size 0: nothing to read
    list.BindVertex(3, {{0x40000, 0x1000}, {0x50000, 0x1000}, {0x60000, 0x1000}});
    list.Draws(1);
    list.End();

    Vulkan::Nvn::VertexStageScan scan;
    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, NO_INDICES, KeepsPass, scan);
    REQUIRE(scan.draws == 3);
    REQUIRE(scan.ranges.size() == 3);
    CHECK(scan.ranges[0].address == 0x10000);
    CHECK(scan.ranges[0].size == 0x300);
    CHECK(scan.ranges[1].address == 0x40000);
    CHECK(scan.ranges[2].address == 0x50000);
    // 0x60000 would take the pass past max_total_bytes.
    CHECK(scan.bytes == 0x2300);
    CHECK(scan.capped); // EDEN_NX2_NVN_VTXCOV_DIAG: total_cap

    // A scan starts over: nothing of the previous pass is left.
    Records next;
    next.Draws(4);
    next.End();
    Vulkan::Nvn::ScanPassVertexRanges(next.words, {}, LIMITS, NO_INDICES, KeepsPass, scan);
    CHECK(scan.draws == 4);
    CHECK(scan.ranges.empty());
    CHECK(scan.bytes == 0);
    CHECK_FALSE(scan.capped);
}

TEST_CASE("NVN vertex staging: a malformed or missing list stops the scan", "[nvn][pass_state]") {
    Records list;
    list.BindVertex(0, {{0x10000, 0x100}});
    list.Draws(2);
    list.words.push_back(static_cast<u64>(VC::Op::Draw)); // a Draw record of size 0
    list.BindVertex(0, {{0x20000, 0x100}});
    list.Draws(2);

    Vulkan::Nvn::VertexStageScan scan;
    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, NO_INDICES, KeepsPass, scan);
    CHECK(scan.draws == 2);
    REQUIRE(scan.ranges.size() == 1);
    CHECK(scan.ranges[0].address == 0x10000);

    Vulkan::Nvn::ScanPassVertexRanges({}, {}, LIMITS, NO_INDICES, KeepsPass, scan);
    CHECK(scan.draws == 0);
    CHECK(scan.ranges.empty());
}

TEST_CASE("NVN vertex staging: index ranges of the draws that bind their guest indices",
          "[nvn][pass_state]") {
    Records list;
    list.BindVertex(0, {{0x10000, 0x400}});
    list.DrawIndexed(4, 1, 0x30, 0x80000);          // triangles, 16-bit: 0x60 bytes
    list.DrawIndexed(4, 2, 0x10, 0x80000);          // the same start, 32-bit: 0x40, kept 0x60
    list.DrawIndexed(4, 2, 0x40, 0x90000);          // 32-bit: 0x100 bytes
    list.DrawIndexed(4, 0, 0x20, 0xA0000);          // 8-bit: only where the device binds them
    list.DrawIndexed(7, 1, 0x40, 0xB0000);          // quads: converted, never bound
    list.DrawIndexed(2, 1, 0x40, 0xB1000);          // line loop: converted
    list.DrawIndexed(8, 1, 0x40, 0xB2000);          // quad strip: converted
    list.DrawIndexed(4, 1, 0x40, 0xC0000, 0xD0000); // indirect: the count is on the GPU
    list.DrawIndexed(4, 3, 0x40, 0xC1000);          // no such index type
    list.DrawIndexed(4, 1, 0x1000, 0xE0000);        // 0x2000 bytes: above max_range_bytes
    list.End();

    Vulkan::Nvn::VertexStageScan scan;
    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, NO_INDICES, KeepsPass, scan);
    CHECK(scan.draws == 10);
    REQUIRE(scan.ranges.size() == 1); // vertex ranges only

    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, {.enabled = true}, KeepsPass, scan);
    CHECK(scan.draws == 10);
    REQUIRE(scan.ranges.size() == 3);
    CHECK(scan.ranges[1].address == 0x80000);
    CHECK(scan.ranges[1].size == 0x60);
    CHECK(scan.ranges[2].address == 0x90000);
    CHECK(scan.ranges[2].size == 0x100);
    CHECK(scan.bytes == 0x400 + 0x60 + 0x100);

    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS,
                                      {.enabled = true, .uint8_indices = true}, KeepsPass, scan);
    REQUIRE(scan.ranges.size() == 4);
    CHECK(scan.ranges[3].address == 0xA0000);
    CHECK(scan.ranges[3].size == 0x20);
}

TEST_CASE("NVN vertex staging: neighbouring ranges of a pool share one copy region",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::VertexStageSource;
    constexpr Vulkan::Nvn::VertexStageLimits limits{.min_draws = 2,
                                                    .max_range_bytes = 0x1000,
                                                    .max_total_bytes = 0x100000,
                                                    .max_gap_bytes = 0x200,
                                                    .max_span_bytes = 0xD00};
    // In first-use order, two pools mixed.
    std::vector<VertexStageSource> sources{
        {.pool = 2, .offset = 0x1000, .size = 0x100, .address = 0xB1000},
        {.pool = 1, .offset = 0x1200, .size = 0x100, .address = 0xA1200}, // 0x100 after the next
        {.pool = 1, .offset = 0x1000, .size = 0x100, .address = 0xA1000},
        {.pool = 1, .offset = 0x1240, .size = 0x40, .address = 0xA1240},  // inside the previous
        {.pool = 1, .offset = 0x1600, .size = 0x100, .address = 0xA1600}, // 0x300 gap: too far
        {.pool = 1, .offset = 0x1700, .size = 0xC00, .address = 0xA1700}, // adjacent
        {.pool = 1, .offset = 0x2300, .size = 0x100, .address = 0xA2300}, // past max_span_bytes
    };
    std::vector<Vulkan::Nvn::VertexStageSpan> spans;
    const u64 covered = Vulkan::Nvn::CoalesceVertexStageSources(sources, limits, spans);
    REQUIRE(spans.size() == 4);
    CHECK((spans[0].pool == 1 && spans[0].begin == 0x1000 && spans[0].end == 0x1300));
    CHECK((spans[1].pool == 1 && spans[1].begin == 0x1600 && spans[1].end == 0x2300));
    CHECK((spans[2].pool == 1 && spans[2].begin == 0x2300 && spans[2].end == 0x2400));
    CHECK((spans[3].pool == 2 && spans[3].begin == 0x1000 && spans[3].end == 0x1100));
    CHECK(covered == 0x300 + 0xD00 + 0x100 + 0x100);
    // Sorted by pool and offset; every source lies inside its region.
    for (size_t i = 0; i < sources.size(); ++i) {
        const VertexStageSource& source = sources[i];
        const Vulkan::Nvn::VertexStageSpan& span = spans[source.span];
        CHECK(span.pool == source.pool);
        CHECK(source.offset >= span.begin);
        CHECK(source.offset + source.size <= span.end);
        if (i > 0) {
            CHECK(std::tie(sources[i - 1].pool, sources[i - 1].offset) <=
                  std::tie(source.pool, source.offset));
        }
    }
    CHECK(sources[0].address == 0xA1000);
    CHECK(sources[0].span == 0);
    CHECK(sources[2].address == 0xA1240);
    CHECK(sources[2].span == 0);
    CHECK(sources[6].address == 0xB1000);
    CHECK(sources[6].span == 3);
}

TEST_CASE("NVN vertex staging: merged gaps never exceed the bytes the draws read",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::VertexStageSource;
    constexpr Vulkan::Nvn::VertexStageLimits limits{.min_draws = 2,
                                                    .max_range_bytes = 0x1000,
                                                    .max_total_bytes = 0x100000,
                                                    .max_gap_bytes = 0x1000,
                                                    .max_span_bytes = 0x100000};
    // The four ranges read 0x400 bytes. The gaps before the second and third range (0x80 each)
    // stay within that; the 0x400 gap before the fourth would take the gaps to 0x500.
    std::vector<VertexStageSource> sources{
        {.pool = 7, .offset = 0x0000, .size = 0x100, .address = 0x10000},
        {.pool = 7, .offset = 0x0180, .size = 0x100, .address = 0x10180},
        {.pool = 7, .offset = 0x0300, .size = 0x100, .address = 0x10300},
        {.pool = 7, .offset = 0x0800, .size = 0x100, .address = 0x10800},
    };
    std::vector<Vulkan::Nvn::VertexStageSpan> spans;
    const u64 covered = Vulkan::Nvn::CoalesceVertexStageSources(sources, limits, spans);
    REQUIRE(spans.size() == 2);
    CHECK((spans[0].begin == 0x0000 && spans[0].end == 0x0400));
    CHECK((spans[1].begin == 0x0800 && spans[1].end == 0x0900));
    CHECK(covered == 0x500);

    // Nothing to merge: no regions.
    std::vector<VertexStageSource> none;
    CHECK(Vulkan::Nvn::CoalesceVertexStageSources(none, limits, spans) == 0);
    CHECK(spans.empty());
}

TEST_CASE("NVN vertex staging: copies outlive only pass ends that cannot write pool memory",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::BreakCause;
    using Vulkan::Nvn::StagedVertexSurvives;
    // Image or state changes only: later passes of the command buffer reuse the copies.
    for (const BreakCause cause :
         {BreakCause::SetRenderTargets, BreakCause::FeedbackCopy, BreakCause::LayoutChange,
          BreakCause::MaskedClear, BreakCause::ContentsUpload}) {
        CHECK(StagedVertexSurvives(cause, false, false));
        // Not after a pass that wrote through shaders or still has counter reports to write.
        CHECK_FALSE(StagedVertexSurvives(cause, true, false));
        CHECK_FALSE(StagedVertexSurvives(cause, false, true));
    }
    // Ops that run outside a pass, and breaks that write or upload pool memory.
    for (const BreakCause cause :
         {BreakCause::Barrier, BreakCause::Dispatch, BreakCause::ReportCounter,
          BreakCause::ReportValue, BreakCause::ResetCounter, BreakCause::RenderCondition,
          BreakCause::OtherOp, BreakCause::MirrorUpload, BreakCause::IndirectUpload,
          BreakCause::PassBufferWrites, BreakCause::ShadowAlias, BreakCause::ReportRead,
          BreakCause::MeshIndirectConvert, BreakCause::SegmentEnd, BreakCause::Other}) {
        CHECK_FALSE(StagedVertexSurvives(cause, false, false));
    }
}

TEST_CASE("NVN vertex staging: a mesh indirect conversion break keeps the copies only for the span "
          "window",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::BreakCause;
    using Vulkan::Nvn::StagedVertexSurvives;
    CHECK_FALSE(StagedVertexSurvives(BreakCause::MeshIndirectConvert, false, false));
    CHECK_FALSE(StagedVertexSurvives(BreakCause::MeshIndirectConvert, false, false, false));
    CHECK(StagedVertexSurvives(BreakCause::MeshIndirectConvert, false, false, true));
    // Not after shader writes or with reports pending, as every other cause.
    CHECK_FALSE(StagedVertexSurvives(BreakCause::MeshIndirectConvert, true, false, true));
    CHECK_FALSE(StagedVertexSurvives(BreakCause::MeshIndirectConvert, false, true, true));
    // The switch changes nothing else.
    CHECK_FALSE(StagedVertexSurvives(BreakCause::Dispatch, false, false, true));
    CHECK_FALSE(StagedVertexSurvives(BreakCause::OtherOp, false, false, true));
    CHECK(StagedVertexSurvives(BreakCause::SetRenderTargets, false, false, true));
}

TEST_CASE("NVN vertex staging span: the window goes on past render target changes",
          "[nvn][pass_state]") {
    Records list;
    list.BindVertex(0, {{0x10000, 0x100}});
    list.Draws(2);
    list.Add(VC::Op::SetRenderTargets, u64{0}); // pass 2: its own ranges
    list.BindVertex(0, {{0x20000, 0x100}});
    list.Draws(2);
    list.Add(VC::Op::SetRenderTargets, u64{0}); // pass 3
    list.BindVertex(0, {{0x30000, 0x100}});
    list.Draws(1);
    list.Add(VC::Op::Dispatch, VC::DispatchPayload{1, 1, 1}); // ends the window
    list.BindVertex(0, {{0x40000, 0x100}});
    list.Draws(9);
    list.End();
    const auto keeps = [](VC::Op op) { return op != VC::Op::Dispatch; };

    // Without the span: the window is the first pass.
    Vulkan::Nvn::VertexStageScan single;
    Vulkan::Nvn::ScanPassVertexRanges(
        list.words, {}, LIMITS, NO_INDICES,
        [&](VC::Op op) { return op != VC::Op::SetRenderTargets && keeps(op); }, single);
    CHECK(single.draws == 2);
    REQUIRE(single.ranges.size() == 1);
    CHECK(single.end > list.words.data());
    CHECK(single.end < list.words.data() + list.words.size());

    // With it: three passes, up to the dispatch.
    Vulkan::Nvn::VertexStageScan span;
    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, NO_INDICES,
                                      Vulkan::Nvn::SpanWindow{keeps}, span);
    CHECK(span.draws == 5);
    REQUIRE(span.ranges.size() == 3);
    CHECK(span.ranges[0].address == 0x10000);
    CHECK(span.ranges[1].address == 0x20000);
    CHECK(span.ranges[2].address == 0x30000);
    CHECK(span.end > single.end);
    // The window ends at the dispatch record itself (not consumed).
    const auto* stop = reinterpret_cast<const VC::RecordHeader*>(span.end);
    CHECK(VC::RecordOp(stop) == VC::Op::Dispatch);
}

TEST_CASE(
    "NVN vertex staging span: after a target change a barrier, condition or call ends the window",
    "[nvn][pass_state]") {
    const auto keeps = [](VC::Op) { return true; };
    for (const VC::Op stopper : {VC::Op::Barrier, VC::Op::SetRenderCondition, VC::Op::CallList}) {
        Records list;
        list.BindVertex(0, {{0x10000, 0x100}});
        list.Draws(1);
        list.Add(VC::Op::SetRenderTargets, u64{0});
        list.BindVertex(0, {{0x20000, 0x100}});
        list.Draws(1);
        list.Add(stopper, u64{0});
        list.BindVertex(0, {{0x30000, 0x100}});
        list.Draws(1);
        list.End();
        Vulkan::Nvn::VertexStageScan span;
        Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, NO_INDICES,
                                          Vulkan::Nvn::SpanWindow{keeps}, span);
        CHECK(span.draws == 2);
        CHECK(span.ranges.size() == 2);
        CHECK(VC::RecordOp(reinterpret_cast<const VC::RecordHeader*>(span.end)) == stopper);
    }
    // Inside the first pass (no target change yet) they stay in the window, as without the span.
    Records list;
    list.BindVertex(0, {{0x10000, 0x100}});
    list.Draws(1);
    list.Add(VC::Op::Barrier, u64{0});
    list.Draws(1);
    list.End();
    Vulkan::Nvn::VertexStageScan span;
    Vulkan::Nvn::ScanPassVertexRanges(list.words, {}, LIMITS, NO_INDICES,
                                      Vulkan::Nvn::SpanWindow{keeps}, span);
    CHECK(span.draws == 2);
    // The list's End is not consumed either: `end` points at it.
    CHECK(VC::RecordOp(reinterpret_cast<const VC::RecordHeader*>(span.end)) == VC::Op::End);
    CHECK(span.end == list.words.data() + list.words.size() - 2); // End = header + one word
}

TEST_CASE("NVN vertex staging: shader copy tasks cover whole 16-byte units, 64 KiB each",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::VertexStageTask;
    std::vector<VertexStageTask> tasks;
    // 0x20 bytes from 3 bytes into a unit: units 0x100..0x102 of the window, the destination
    // shifted alike (equal offsets modulo 16).
    Vulkan::Nvn::AppendVertexStageTasks(0x1003, 0x2003, 0x20, tasks);
    REQUIRE(tasks.size() == 1);
    CHECK(tasks[0].source == 0x100);
    CHECK(tasks[0].destination == 0x200);
    CHECK(tasks[0].count == 3);

    // Two full workgroups and a one-unit tail, appended after the first task.
    constexpr u64 units = 2 * Vulkan::Nvn::VERTEX_STAGE_TASK_UNITS + 1;
    Vulkan::Nvn::AppendVertexStageTasks(0x10, 0x4010, units * 16, tasks);
    REQUIRE(tasks.size() == 4);
    CHECK(tasks[1].source == 1);
    CHECK(tasks[1].destination == 0x401);
    CHECK(tasks[1].count == Vulkan::Nvn::VERTEX_STAGE_TASK_UNITS);
    CHECK(tasks[2].source == 1 + Vulkan::Nvn::VERTEX_STAGE_TASK_UNITS);
    CHECK(tasks[2].destination == 0x401 + Vulkan::Nvn::VERTEX_STAGE_TASK_UNITS);
    CHECK(tasks[3].source == 1 + 2 * Vulkan::Nvn::VERTEX_STAGE_TASK_UNITS);
    CHECK(tasks[3].count == 1);

    // An aligned range ending mid-unit takes the whole last unit.
    tasks.clear();
    Vulkan::Nvn::AppendVertexStageTasks(0x40, 0x140, 0x11, tasks);
    REQUIRE(tasks.size() == 1);
    CHECK(tasks[0].source == 4);
    CHECK(tasks[0].destination == 0x14);
    CHECK(tasks[0].count == 2);
}

TEST_CASE("NVN pass state: state ops keep the pass, the provoking vertex too",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::DrawContext;
    namespace VC = VideoCommon::Nvn;
    CHECK(DrawContext::OpKeepsPass(VC::Op::SetSampleMask));
    CHECK(DrawContext::OpKeepsPass(VC::Op::DrawMeshTasks));
    CHECK(DrawContext::OpKeepsPass(VC::Op::SetProvokingVertex));
    CHECK_FALSE(DrawContext::OpKeepsPass(VC::Op::Dispatch));
    CHECK_FALSE(DrawContext::OpKeepsPass(VC::Op::EvaluateDlss));
}

TEST_CASE("NVN draw plan: image uses merge their layouts, sample-only only when every use samples",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::DrawPlan;
    using Vulkan::Nvn::Storage;
    DrawPlan plan;
    const auto sampled = std::make_shared<Storage>();
    const auto written = std::make_shared<Storage>();
    const auto absent = std::make_shared<Storage>();

    // Two samples of one image in one layout: one use, nothing but sampling.
    plan.UseImage(sampled, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);
    plan.UseImage(sampled, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);
    // Sampled and bound as a storage image: GENERAL, and no longer sample-only.
    plan.UseImage(written, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true);
    plan.UseImage(written, VK_IMAGE_LAYOUT_GENERAL);
    REQUIRE(plan.images.size() == 2);
    CHECK(plan.images[0].owner == sampled);
    CHECK(plan.images[0].layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(plan.images[0].sample_only);
    CHECK(plan.images[1].owner == written);
    CHECK(plan.images[1].layout == VK_IMAGE_LAYOUT_GENERAL);
    CHECK_FALSE(plan.images[1].sample_only);

    // Every descriptor of an image names the merged layout; an image the plan does not use keeps
    // the descriptor's own.
    CHECK(plan.LayoutOf(written.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ==
          VK_IMAGE_LAYOUT_GENERAL);
    CHECK(plan.LayoutOf(sampled.get(), VK_IMAGE_LAYOUT_GENERAL) ==
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(plan.LayoutOf(absent.get(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) ==
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    // A storage use first, a sample later: still not sample-only.
    plan.Clear();
    plan.UseImage(written, VK_IMAGE_LAYOUT_GENERAL);
    plan.UseImage(written, VK_IMAGE_LAYOUT_GENERAL, true);
    REQUIRE(plan.images.size() == 1);
    CHECK(plan.images[0].layout == VK_IMAGE_LAYOUT_GENERAL);
    CHECK_FALSE(plan.images[0].sample_only);
}

// nvnCommandBufferBarrier: which bits make the console wait (nvn_barrier.h: the methods the NVN
// driver records per bit). OoT in Kokiri, one frame: 0x12 73 times, 0x11 28, 0x28 14 (of 134).
TEST_CASE("NVN barrier: only ORDER_PRIMITIVES and ORDER_INDIRECT_DATA wait for all earlier work",
          "[nvn][pass_state]") {
    using Vulkan::Nvn::NvnBarrierOrders;
    using Vulkan::Nvn::NvnBarrierOrdersAllWork;
    for (const u32 bits : {0x1u, 0x8u, 0x9u, 0x11u, 0x13u, 0x21u, 0x28u, 0x97u, 0x400u}) {
        INFO("bits " << bits);
        CHECK(NvnBarrierOrdersAllWork(bits));
        CHECK(NvnBarrierOrders(bits));
    }
    // Pixel work only: PIXEL_SHADER_BARRIER / the tiled cache barrier.
    for (const u32 bits : {0x2u, 0x4u, 0x12u, 0x92u}) {
        INFO("bits " << bits);
        CHECK_FALSE(NvnBarrierOrdersAllWork(bits));
        CHECK(NvnBarrierOrders(bits));
    }
    // Cache invalidations without a wait (the _NO_WFI methods), and nothing at all.
    for (const u32 bits : {0x0u, 0x10u, 0x20u, 0x40u, 0x80u, 0xF0u}) {
        INFO("bits " << bits);
        CHECK_FALSE(NvnBarrierOrdersAllWork(bits));
        CHECK_FALSE(NvnBarrierOrders(bits));
    }
}
