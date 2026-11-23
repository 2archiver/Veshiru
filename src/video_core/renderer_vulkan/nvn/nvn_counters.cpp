// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/host_shaders/nvn_counter_comp_spv.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/renderer_vulkan/nvn/nvn_counters.h"

namespace Vulkan::Nvn {
namespace {

constexpr VkPipelineStageFlags ALL_STAGES = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
constexpr VkAccessFlags ALL_ACCESS = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

/// Bytes of the scratch buffer: accumulators, the condition/fold slot ring, the resolve areas.
constexpr u32 RING_BASE = CounterEngine::MAX_QUEUES * CounterEngine::ACCUMULATOR_STRIDE;
constexpr u32 AREA_BASE = RING_BASE + CounterEngine::SLOT_COUNT * CounterEngine::SLOT_SIZE;
constexpr u32 SCRATCH_SIZE = AREA_BASE + CounterEngine::RESOLVE_RING_SIZE;

/// Byte offsets inside a 64-byte ring slot (the compute shader indexes the same layout in words).
constexpr u32 SLOT_COUNTER_A = 0;
constexpr u32 SLOT_COUNTER_B = 16;
constexpr u32 SLOT_PREDICATE = 32;
constexpr u32 SLOT_QUERY_RESULT = 40;

/// nvn_counter.comp modes.
constexpr u32 MODE_FOLD = 2;
constexpr u32 MODE_RESOLVE = 3;
constexpr u32 MODE_CONDITIONS = 4;
constexpr u32 PROGRAM_WORDS = CounterEngine::RESOLVE_CHUNK_EVENTS * 2 / 32;
static_assert(CounterEngine::MAX_CONDITION_BATCH <= PROGRAM_WORDS * 32,
              "one mode bit per batched condition");

struct PushConstants {
    u32 base;
    u32 mode;
    u32 accumulator;
};

/// MODE_RESOLVE: `base` and `out` are the word offsets of the chunk's first query result and
/// first report value, `program` holds 2 bits per event of the chunk (CounterBatch::Event).
struct ResolveConstants {
    u32 base;
    u32 mode;
    u32 accumulator;
    u32 count;
    u32 out;
    std::array<u32, PROGRAM_WORDS> program;
};
static_assert(sizeof(ResolveConstants) <= 128, "push constants beyond the guaranteed minimum");

} // Anonymous namespace

CounterEngine::CounterEngine(
    const Device& device_, MemoryAllocator& allocator,
    std::function<void(const std::function<void(vk::CommandBuffer)>&)> immediate_submit,
    std::function<bool(u64)> wait_completed_, bool allow_conditional_rendering,
    bool allow_in_pass_queries)
    : device{device_}, dld{device_.GetDispatchLoader()}, wait_completed{std::move(wait_completed_)},
      slot_seq(SLOT_COUNT, 0) {
    gpu_conditions = allow_conditional_rendering && device.IsExtConditionalRendering();
    const VkPhysicalDeviceProperties properties = device.GetPhysical().GetProperties();
    const VkPhysicalDeviceFeatures features = device.GetPhysical().GetFeatures();
    precise_occlusion = features.occlusionQueryPrecise != VK_FALSE;
    host_reset = device.IsHostQueryResetSupported();
    in_pass_queries = allow_in_pass_queries && host_reset;
    timestamp_period = properties.limits.timestampPeriod > 0.0f ? properties.limits.timestampPeriod
                                                                 : 1.0;
    const auto families = device.GetPhysical().GetQueueFamilyProperties();
    u32 valid_bits = 64;
    if (device.GetGraphicsFamily() < families.size()) {
        valid_bits = families[device.GetGraphicsFamily()].timestampValidBits;
    }
    timestamp_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (gpu_conditions) {
        usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    }
    scratch = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = SCRATCH_SIZE,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::DeviceLocal);

    const vk::Device& logical = device.GetLogical();
    const VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    set_layout = logical.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &binding,
    });
    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ResolveConstants),
    };
    pipeline_layout = logical.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    });
    const VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };
    descriptor_pool = logical.CreateDescriptorPool({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    });
    descriptor_sets = descriptor_pool.Allocate({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = set_layout.address(),
    });
    const VkDescriptorBufferInfo buffer_info{
        .buffer = *scratch,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = descriptor_sets[0],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &buffer_info,
        .pTexelBufferView = nullptr,
    };
    logical.UpdateDescriptorSets(write, {});

    const std::span<const u32> code = NVN_COUNTER_COMP_SPV;
    module = logical.CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = static_cast<u32>(code.size_bytes()),
        .pCode = code.data(),
    });
    pipeline = logical.CreateComputePipeline(
        VkComputePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = *module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            .layout = *pipeline_layout,
            .basePipelineHandle = {},
            .basePipelineIndex = 0,
        },
        device.StaticPipelineCache());

    // Accumulators and predicates start at zero.
    immediate_submit([this](vk::CommandBuffer cmd) {
        cmd.FillBuffer(*scratch, 0, SCRATCH_SIZE, 0);
    });
    LOG_INFO(Render_Vulkan,
             "[nvn] counters: render conditions on the {}, timestamp period {:.3f} ns, {} valid "
             "bits, precise occlusion {}, occlusion queries {}",
             gpu_conditions ? "GPU (VK_EXT_conditional_rendering)" : "CPU (flush and wait)",
             timestamp_period, valid_bits, precise_occlusion,
             in_pass_queries ? "inside render passes" : "across render passes");
}

CounterEngine::~CounterEngine() = default;

u32 CounterEngine::AllocateSlot(u64 seq) {
    return AllocateSlots(seq, 1);
}

u32 CounterEngine::AllocateSlots(u64 seq, u32 count) {
    ASSERT(count >= 1 && count <= MAX_CONDITION_BATCH);
    // A batch is consecutive: it starts over at slot 0 instead of wrapping.
    if (next_slot % SLOT_COUNT + count > SLOT_COUNT) {
        next_slot += SLOT_COUNT - next_slot % SLOT_COUNT;
    }
    const u32 first = static_cast<u32>(next_slot % SLOT_COUNT);
    next_slot += count;
    for (u32 index = first; index < first + count; ++index) {
        // The slot may still be read by an earlier submission (a long run of conditions in flight).
        if (slot_seq[index] != 0 && slot_seq[index] != seq && !wait_completed(slot_seq[index])) {
            LOG_WARNING(Render_Vulkan, "[nvn] counter scratch slot {} is still used by seq {} after "
                                       "2 s: reusing it",
                        index, slot_seq[index]);
        }
        slot_seq[index] = seq;
    }
    return RING_BASE + first * SLOT_SIZE;
}

u32 CounterEngine::AllocateArea(u64 seq, u32 bytes) {
    ASSERT(bytes <= RESOLVE_RING_SIZE);
    return AREA_BASE + area_ring.Allocate(seq, bytes, [this](u64 old_seq, u32 old_begin) {
               // As AllocateSlot: an earlier submission may still copy the report values out.
               if (!wait_completed(old_seq)) {
                   LOG_WARNING(Render_Vulkan,
                               "[nvn] counter resolve area at {:#x} is still used by seq {} "
                               "after 2 s: reusing it",
                               old_begin, old_seq);
               }
           });
}

void CounterEngine::Sync(vk::CommandBuffer cmd) const {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = ALL_ACCESS,
        .dstAccessMask = ALL_ACCESS,
    };
    cmd.PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, barrier);
}

void CounterEngine::Dispatch(vk::CommandBuffer cmd, u32 base_word, u32 mode,
                             u32 accumulator_word) const {
    const PushConstants constants{base_word, mode, accumulator_word};
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, descriptor_sets[0],
                           {});
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    cmd.Dispatch(1, 1, 1);
}

void CounterEngine::RecordCondition(vk::CommandBuffer cmd, VkBuffer a_buffer, u64 a_offset,
                                    VkBuffer b_buffer, u64 b_offset, u32 mode, u32 slot_offset) {
    const ConditionInput input{a_buffer, a_offset, b_buffer, b_offset, mode};
    RecordConditions(cmd, std::span{&input, 1}, slot_offset);
}

void CounterEngine::RecordConditions(vk::CommandBuffer cmd,
                                     std::span<const ConditionInput> conditions, u32 first_slot) {
    if (conditions.empty()) {
        return;
    }
    ASSERT(conditions.size() <= MAX_CONDITION_BATCH);
    Sync(cmd);
    // Both counters of every pair next to each other in its slot; one copy per source buffer.
    condition_copies.clear();
    for (size_t i = 0; i < conditions.size(); ++i) {
        const ConditionInput& c = conditions[i];
        const u64 slot = first_slot + i * SLOT_SIZE;
        condition_copies.emplace_back(c.a_buffer, VkBufferCopy{.srcOffset = c.a_offset,
                                                               .dstOffset = slot + SLOT_COUNTER_A,
                                                               .size = sizeof(u64)});
        condition_copies.emplace_back(c.b_buffer, VkBufferCopy{.srcOffset = c.b_offset,
                                                               .dstOffset = slot + SLOT_COUNTER_B,
                                                               .size = sizeof(u64)});
    }
    std::ranges::stable_sort(condition_copies, {}, [](const auto& copy) { return copy.first; });
    for (size_t i = 0; i < condition_copies.size();) {
        const VkBuffer source = condition_copies[i].first;
        condition_regions.clear();
        for (; i < condition_copies.size() && condition_copies[i].first == source; ++i) {
            condition_regions.push_back(condition_copies[i].second);
        }
        cmd.CopyBuffer(source, *scratch,
                       vk::Span<VkBufferCopy>(condition_regions.data(), condition_regions.size()));
    }
    Sync(cmd);
    ResolveConstants constants{};
    constants.base = first_slot / 4;
    constants.mode = MODE_CONDITIONS;
    constants.count = static_cast<u32>(conditions.size());
    for (size_t i = 0; i < conditions.size(); ++i) {
        if (conditions[i].mode == ConditionMode::RenderIfNotEqual) {
            constants.program[i / 32] |= 1u << (i % 32);
        }
    }
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, descriptor_sets[0],
                           {});
    cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
    cmd.Dispatch(1, 1, 1);
    Sync(cmd);
    stats.conditions_gpu.fetch_add(conditions.size(), std::memory_order_relaxed);
}

void CounterEngine::BeginConditional(vk::CommandBuffer cmd, u32 slot_offset) const {
    const VkConditionalRenderingBeginInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
        .pNext = nullptr,
        .buffer = *scratch,
        .offset = slot_offset + SLOT_PREDICATE,
        .flags = 0,
    };
    cmd.BeginConditionalRenderingEXT(info);
}

void CounterEngine::EndConditional(vk::CommandBuffer cmd) const {
    cmd.EndConditionalRenderingEXT();
}

QueryBlock CounterEngine::NewBlock() {
    {
        std::scoped_lock lock{block_mutex};
        if (!free_blocks.empty()) {
            QueryBlock block = std::move(free_blocks.back());
            free_blocks.pop_back();
            // The submission that used it completed: nothing reads or writes it any more.
            ResetBlock(block);
            return block;
        }
    }
    QueryBlock block;
    const vk::Device& logical = device.GetLogical();
    block.occlusion = logical.CreateQueryPool({
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_OCCLUSION,
        .queryCount = BLOCK_QUERIES,
        .pipelineStatistics = 0,
    });
    block.timestamp = logical.CreateQueryPool({
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = BLOCK_QUERIES,
        .pipelineStatistics = 0,
    });
    ResetBlock(block);
    return block;
}

void CounterEngine::ResetBlock(QueryBlock& block) {
    if (host_reset) {
        const vk::Device& logical = device.GetLogical();
        logical.ResetQueryPool(*block.occlusion, 0, BLOCK_QUERIES);
        logical.ResetQueryPool(*block.timestamp, 0, BLOCK_QUERIES);
    }
}

QueryBlock& CounterEngine::BlockWithRoom(CounterFlight& flight, bool occlusion, u32& index) {
    const auto full = [&](const QueryBlock& block) {
        return (occlusion ? block.occlusion_used : block.timestamp_used) >= BLOCK_QUERIES;
    };
    if (flight.blocks.empty() || full(flight.blocks.back())) {
        flight.blocks.push_back(NewBlock());
    }
    index = static_cast<u32>(flight.blocks.size() - 1);
    return flight.blocks.back();
}

QueryRef CounterEngine::BeginOcclusion(vk::CommandBuffer cmd, CounterFlight& flight) {
    u32 index = 0;
    QueryBlock& block = BlockWithRoom(flight, true, index);
    const u32 slot = block.occlusion_used++;
    if (!host_reset) {
        cmd.ResetQueryPool(*block.occlusion, slot, 1);
    }
    cmd.BeginQuery(*block.occlusion, slot, precise_occlusion ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
    return {index, slot};
}

void CounterEngine::EndOcclusion(vk::CommandBuffer cmd, CounterFlight& flight, QueryRef query) {
    cmd.EndQuery(*flight.blocks[query.block].occlusion, query.slot);
}

void CounterEngine::FoldOcclusion(vk::CommandBuffer cmd, CounterFlight& flight, QueryRef query,
                                  u32 queue, u64 seq) {
    const VkQueryPool pool = *flight.blocks[query.block].occlusion;
    cmd.EndQuery(pool, query.slot);
    const u32 slot_offset = AllocateSlot(seq);
    Sync(cmd);
    cmd.CopyQueryPoolResults(pool, query.slot, 1, *scratch, slot_offset + SLOT_QUERY_RESULT,
                             sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    Sync(cmd);
    Dispatch(cmd, slot_offset / 4, MODE_FOLD, (queue % MAX_QUEUES) * ACCUMULATOR_STRIDE / 4);
    Sync(cmd);
    stats.occlusion_folds.fetch_add(1, std::memory_order_relaxed);
}

void CounterEngine::ResetAccumulator(vk::CommandBuffer cmd, u32 queue) {
    Sync(cmd);
    cmd.FillBuffer(*scratch, (queue % MAX_QUEUES) * ACCUMULATOR_STRIDE, ACCUMULATOR_STRIDE, 0);
    Sync(cmd);
}

void CounterEngine::CopyAccumulator(vk::CommandBuffer cmd, u32 queue, VkBuffer dst,
                                    u64 dst_offset) {
    Sync(cmd);
    const VkBufferCopy region{.srcOffset = (queue % MAX_QUEUES) * ACCUMULATOR_STRIDE,
                              .dstOffset = dst_offset,
                              .size = sizeof(u64)};
    cmd.CopyBuffer(*scratch, dst, region);
    Sync(cmd);
}

void CounterEngine::ResolveBatch(vk::CommandBuffer cmd, CounterFlight& flight, CounterBatch& batch,
                                 u32 queue, u64 seq) {
    if (batch.Empty()) {
        return;
    }
    // The recorder resolves a batch before it holds more than MAX_EVENTS.
    ASSERT_MSG(batch.events.size() <= CounterBatch::MAX_EVENTS, "counter batch of {} events",
               batch.events.size());
    const u32 query_bytes = static_cast<u32>(batch.queries.size() * sizeof(u64));
    const u32 report_bytes = static_cast<u32>(batch.reports.size() * sizeof(u64));
    const u32 area = AllocateArea(seq, query_bytes + report_bytes);
    const u32 reports_area = area + query_bytes;
    Sync(cmd);
    // The query results in batch order; consecutive slots of one pool go in one copy.
    const size_t query_count = batch.queries.size();
    for (size_t i = 0; i < query_count;) {
        const QueryRef first = batch.queries[i];
        u32 run = 1;
        while (i + run < query_count && batch.queries[i + run].block == first.block &&
               batch.queries[i + run].slot == first.slot + run) {
            ++run;
        }
        cmd.CopyQueryPoolResults(*flight.blocks[first.block].occlusion, first.slot, run, *scratch,
                                 area + i * sizeof(u64), sizeof(u64),
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        i += run;
    }
    Sync(cmd);
    // One dispatch per RESOLVE_CHUNK_EVENTS events; each starts from the accumulator the one
    // before stored (the barrier between them orders that).
    ResolveConstants constants{};
    constants.mode = MODE_RESOLVE;
    constants.accumulator = (queue % MAX_QUEUES) * ACCUMULATOR_STRIDE / 4;
    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0, descriptor_sets[0],
                           {});
    u32 queries_before = 0;
    u32 reports_before = 0;
    for (size_t first = 0; first < batch.events.size(); first += RESOLVE_CHUNK_EVENTS) {
        const size_t count =
            (std::min)(batch.events.size() - first, static_cast<size_t>(RESOLVE_CHUNK_EVENTS));
        constants.base = (area + queries_before * static_cast<u32>(sizeof(u64))) / 4;
        constants.out = (reports_area + reports_before * static_cast<u32>(sizeof(u64))) / 4;
        constants.count = static_cast<u32>(count);
        constants.program.fill(0);
        for (size_t i = 0; i < count; ++i) {
            const CounterBatch::Event event = batch.events[first + i];
            constants.program[i / 16] |= static_cast<u32>(event) << ((i % 16) * 2);
            queries_before += event == CounterBatch::Event::Query ? 1 : 0;
            reports_before += event == CounterBatch::Event::Report ? 1 : 0;
        }
        if (first != 0) {
            Sync(cmd);
        }
        cmd.PushConstants(*pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmd.Dispatch(1, 1, 1);
    }
    Sync(cmd);
    // The report values, in guest order. Reports to one buffer share a copy until one overlaps an
    // earlier region written since the last barrier, of any buffer copied so far (the later
    // report must win: copied after a barrier).
    report_regions.clear();
    region_offsets.clear();
    VkBuffer region_buffer = VK_NULL_HANDLE;
    const auto flush = [&] {
        if (!report_regions.empty()) {
            cmd.CopyBuffer(*scratch, region_buffer,
                           vk::Span<VkBufferCopy>(report_regions.data(), report_regions.size()));
            report_regions.clear();
        }
    };
    // A collision of two buffers' keys only costs an extra barrier.
    const auto region_key = [](VkBuffer buffer, u64 offset) {
        return static_cast<u64>(std::hash<VkBuffer>{}(buffer)) * 0x9E3779B97F4A7C15ULL + offset;
    };
    for (size_t r = 0; r < batch.reports.size(); ++r) {
        const CounterBatch::Report& report = batch.reports[r];
        // Report offsets are 4-byte aligned (ReportCounter): two 8-byte values overlap when
        // their offsets are at most 4 apart.
        const u64 key = region_key(report.buffer, report.offset);
        const bool overlaps = region_offsets.contains(key) || region_offsets.contains(key + 4) ||
                              (report.offset >= 4 && region_offsets.contains(key - 4));
        if (report.buffer != region_buffer || overlaps) {
            flush();
            if (overlaps) {
                Sync(cmd);
                region_offsets.clear();
            }
            region_buffer = report.buffer;
        }
        report_regions.push_back(VkBufferCopy{.srcOffset = reports_area + r * sizeof(u64),
                                              .dstOffset = report.offset,
                                              .size = sizeof(u64)});
        region_offsets.insert(key);
    }
    flush();
    Sync(cmd);
    batch.Clear();
    stats.occlusion_folds.fetch_add(1, std::memory_order_relaxed);
}

void CounterEngine::WriteTimestamp(vk::CommandBuffer cmd, CounterFlight& flight,
                                   bool top_of_pipe, TimestampReport report) {
    u32 index = 0;
    QueryBlock& block = BlockWithRoom(flight, false, index);
    const u32 slot = block.timestamp_used++;
    if (!host_reset) {
        cmd.ResetQueryPool(*block.timestamp, slot, 1);
    }
    if (timestamp_mask != 0) {
        // A family without timestamp bits leaves the query unavailable: Complete stamps it with now.
        cmd.WriteTimestamp(top_of_pipe ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                       : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                           *block.timestamp, slot);
    }
    report.query = {index, slot};
    flight.timestamps.push_back(std::move(report));
}

u64 CounterEngine::TicksToNanoseconds(u64 ticks) {
    u64 raw = ticks & timestamp_mask;
    if (timestamp_mask != ~u64{0}) {
        // Counters narrower than 64 bits wrap; reports arrive roughly in time order.
        if (raw < last_ticks && last_ticks - raw > (timestamp_mask >> 1)) {
            ++tick_epoch;
        }
        last_ticks = raw;
        raw += tick_epoch * (timestamp_mask + 1);
    }
    const s64 gpu_ns = static_cast<s64>(static_cast<double>(raw) * timestamp_period);
    // A report is processed some time after the GPU wrote it, and the GPU clock drifts against the
    // guest clock (the epoch of the title's own tick counter and of
    // nvnDeviceGetCurrentTimestampInNanoseconds): GpuClockMap takes the offset with the least
    // delay over the last few seconds.
    return static_cast<u64>(
        clock_map.Map(gpu_ns, static_cast<s64>(VideoCommon::Nvn::GuestClockNanoseconds())));
}

void CounterEngine::Complete(CounterFlight& flight, VideoCommon::Nvn::GuestMemory* memory) {
    const bool legacy =
        VideoCommon::Nvn::CurrentTimestampMode() == VideoCommon::Nvn::TimestampMode::Legacy;
    for (TimestampReport& report : flight.timestamps) {
        u64 field = 0; // legacy: the guest GPU clock (LegacyReportField); real: GPU time on the guest clock
        if (legacy) {
            // The GPU emulation path's report time: the guest GPU clock when the report is
            // written (not GPU time). The query result is not needed.
            field = VideoCommon::Nvn::LegacyReportField();
        } else {
            u64 ticks = 0;
            const VkResult result = device.GetLogical().GetQueryResults(
                *flight.blocks[report.query.block].timestamp, report.query.slot, 1, sizeof(ticks),
                &ticks, sizeof(ticks), VK_QUERY_RESULT_64_BIT);
            // A submission that never ran (failed, aborted) has no ticks: stamp it with now.
            field = VideoCommon::Nvn::GuestNanosecondsToReportField(
                result == VK_SUCCESS ? TicksToNanoseconds(ticks)
                                     : VideoCommon::Nvn::GuestClockNanoseconds());
        }
        if (report.host) {
            std::memcpy(report.host + sizeof(u64), &field, sizeof(field));
        } else if (memory) {
            memory->Write(report.address + sizeof(u64), &field, sizeof(field));
        }
    }
    flight.timestamps.clear();
    for (TimestampReport& report : flight.traced_samples) {
        // EDEN_NX2_NVN_TSTRACE: the value the guest will read at this report (after the write-back).
        u64 count = 0;
        if (report.host) {
            std::memcpy(&count, report.host, sizeof(count));
        } else if (memory) {
            memory->Read(report.address, &count, sizeof(count));
        }
        VideoCommon::Nvn::TraceSamplesReport(report.address, count);
    }
    flight.traced_samples.clear();
    if (!flight.blocks.empty()) {
        std::scoped_lock lock{block_mutex};
        for (QueryBlock& block : flight.blocks) {
            block.occlusion_used = 0;
            block.timestamp_used = 0;
            free_blocks.push_back(std::move(block));
        }
    }
    flight.blocks.clear();
}

} // namespace Vulkan::Nvn
