// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the NVN GPU profiler (nvn_gpu_profile.*): tags and the timestamp round trip. The device
// test needs a GPU and SKIPs without one.

#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_gpu_profile.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace {

namespace GP = Vulkan::Nvn::GpuProf;

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
};

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
    } catch (const std::exception&) {
        return nullptr;
    }
}

} // namespace

TEST_CASE("nvn gpu profile: every kind has a distinct name", "[nvn]") {
    std::set<std::string> names;
    for (u32 i = 0; i < static_cast<u32>(GP::Kind::Count); ++i) {
        const std::string name = GP::KindName(static_cast<GP::Kind>(i));
        REQUIRE(name != "?");
        REQUIRE(names.insert(name).second);
    }
}

TEST_CASE("nvn gpu profile: tags order by every field", "[nvn]") {
    const GP::Tag base{.kind = GP::Kind::Pass, .count = 4, .hash_a = 1, .hash_b = 2, .format = 37,
                       .width = 1920, .height = 1080, .shape = GP::PackShape(1, true, 1, 1)};
    GP::Tag other = base;
    REQUIRE(base == other);
    other.zero_size = true;
    REQUIRE(base != other);
    other = base;
    other.count = 5;
    REQUIRE(base < other);
    other = base;
    other.shape = GP::PackShape(2, true, 1, 1);
    REQUIRE(base != other);
    REQUIRE(GP::PackShape(3, true, 6, 4) == (3u | 0x10u | (6u << 8) | (2u << 24)));
}

TEST_CASE("nvn gpu profile: timestamps of a submission are read back and aggregated", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    std::unique_ptr<GP::Profiler> profiler = GP::Profiler::Create(device, 1);
    if (!profiler) {
        SKIP("the device cannot do timestamps or host query reset");
    }
    Vulkan::vk::CommandPool pool = device.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });

    for (int round = 0; round < 2; ++round) {
        // The second round reuses the recycled query pool.
        std::unique_ptr<GP::Flight> flight = profiler->NewFlight(true);
        REQUIRE(flight);
        Vulkan::vk::CommandBuffers buffers = pool.Allocate(1);
        const VkCommandBuffer raw = buffers[0];
        const Vulkan::vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
        cmd.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        const u32 begin = flight->Stamp(cmd);
        {
            const GP::Scope outer{flight.get(), cmd, GP::Tag{.kind = GP::Kind::Copy, .count = 7}};
            // Nested scopes belong to the outer one.
            const GP::Scope inner{flight.get(), cmd, GP::Tag{.kind = GP::Kind::Clear}};
        }
        {
            const GP::Scope clear{flight.get(), cmd, GP::Tag{.kind = GP::Kind::Clear}};
        }
        const u32 end = flight->Stamp(cmd);
        flight->Add(begin, end, GP::Tag{.kind = GP::Kind::Segment});
        cmd.End();

        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &raw,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        Vulkan::vk::Check(device.GetGraphicsQueue().Submit(submit));
        void(device.GetLogical().WaitIdle());
        profiler->Complete(*flight);

        const u64 rounds = static_cast<u64>(round + 1);
        REQUIRE(profiler->KindTotal(GP::Kind::Segment).first == rounds);
        REQUIRE(profiler->KindTotal(GP::Kind::Copy).first == rounds);
        REQUIRE(profiler->KindTotal(GP::Kind::Clear).first == rounds); // the nested one is ignored
        REQUIRE(profiler->KindTotal(GP::Kind::Segment).second >=
                profiler->KindTotal(GP::Kind::Copy).second);
    }
}

TEST_CASE("nvn gpu profile: the mode text", "[nvn]") {
    REQUIRE(GP::ParseMode(nullptr) == 0);
    REQUIRE(GP::ParseMode("") == 0);
    REQUIRE(GP::ParseMode("0") == 0);
    REQUIRE(GP::ParseMode("1") == 1);
    REQUIRE(GP::ParseMode("2") == 2);
    REQUIRE(GP::ParseMode("frame") == GP::MODE_FRAME);
    // Mode 2 is the only per-draw one: the frame mode (3) must not turn the draw intervals on.
    REQUIRE(GP::MODE_FRAME > 2);
    REQUIRE(GP::ParseMode("ops") == GP::MODE_OPS);
    REQUIRE(GP::MODE_OPS != 2);
    REQUIRE(GP::MODE_OPS != GP::MODE_FRAME);
}

TEST_CASE("nvn gpu profile: ops mode measures each record up to the next one", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    Vulkan::vk::CommandPool pool = device.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    for (const u32 mode : {GP::MODE_OPS, 1u}) {
        INFO("mode " << mode);
        std::unique_ptr<GP::Profiler> profiler = GP::Profiler::Create(device, mode);
        if (!profiler) {
            SKIP("the device cannot do timestamps or host query reset");
        }
        REQUIRE(profiler->PerOp() == (mode == GP::MODE_OPS));
        std::unique_ptr<GP::Flight> flight = profiler->NewFlight(false);
        REQUIRE(flight);
        Vulkan::vk::CommandBuffers buffers = pool.Allocate(1);
        const VkCommandBuffer raw = buffers[0];
        const Vulkan::vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
        cmd.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        const u32 begin = flight->StampSegment(cmd);
        // Three records: the first holds a copy, the last ends with the command buffer.
        flight->MarkOp(cmd, 4);
        {
            const GP::Scope copy{flight.get(), cmd, GP::Tag{.kind = GP::Kind::Copy, .count = 7}};
        }
        flight->MarkOp(cmd, 9);
        flight->MarkOp(cmd, 9);
        flight->EndOps(cmd);
        // A second end is no interval.
        flight->EndOps(cmd);
        const u32 end = flight->StampSegment(cmd);
        flight->Add(begin, end, GP::Tag{.kind = GP::Kind::Segment});
        cmd.End();
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &raw,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        Vulkan::vk::Check(device.GetGraphicsQueue().Submit(submit));
        void(device.GetLogical().WaitIdle());
        profiler->Complete(*flight);
        REQUIRE(profiler->KindTotal(GP::Kind::Copy).first == 1);
        if (mode == GP::MODE_OPS) {
            REQUIRE(profiler->KindTotal(GP::Kind::Op).first == 3);
            // The first record contains the copy.
            REQUIRE(profiler->KindTotal(GP::Kind::Op).second >=
                    profiler->KindTotal(GP::Kind::Copy).second);
            REQUIRE(profiler->KindTotal(GP::Kind::Segment).second >=
                    profiler->KindTotal(GP::Kind::Op).second);
        } else {
            REQUIRE(profiler->KindTotal(GP::Kind::Op).first == 0);
        }
    }
}

TEST_CASE("nvn gpu profile: ops mode measures every record of a command buffer of 20k records",
          "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    Vulkan::vk::CommandPool pool = device.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    std::unique_ptr<GP::Profiler> profiler = GP::Profiler::Create(device, GP::MODE_OPS);
    if (!profiler) {
        SKIP("the device cannot do timestamps or host query reset");
    }
    std::unique_ptr<GP::Flight> flight = profiler->NewFlight(false);
    REQUIRE(flight);
    REQUIRE(flight->StampCapacity() == GP::Flight::OPS_CAPACITY);
    // An OoT gameplay command buffer: ~10k records, more than the other modes' capacity.
    constexpr u32 RECORDS = 20000;
    static_assert(RECORDS > GP::Flight::CAPACITY);
    Vulkan::vk::CommandBuffers buffers = pool.Allocate(1);
    const VkCommandBuffer raw = buffers[0];
    const Vulkan::vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
    cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    for (u32 i = 0; i < RECORDS; ++i) {
        flight->MarkOp(cmd, static_cast<u16>(i % 7));
    }
    flight->EndOps(cmd);
    cmd.End();
    const VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &raw,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    Vulkan::vk::Check(device.GetGraphicsQueue().Submit(submit));
    void(device.GetLogical().WaitIdle());
    profiler->Complete(*flight);
    REQUIRE(profiler->KindTotal(GP::Kind::Op).first == RECORDS);
}

TEST_CASE("nvn gpu profile: frame mode stamps only the command buffers", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    std::unique_ptr<GP::Profiler> profiler = GP::Profiler::Create(device, GP::MODE_FRAME);
    if (!profiler) {
        SKIP("the device cannot do timestamps or host query reset");
    }
    REQUIRE(profiler->FrameOnly());
    REQUIRE_FALSE(profiler->PerDraw());
    Vulkan::vk::CommandPool pool = device.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });

    // Two frames of two command buffers each (a submission, then the present).
    for (u64 frame = 1; frame <= 2; ++frame) {
        for (const bool present : {false, true}) {
            std::unique_ptr<GP::Flight> flight = profiler->NewFlight(present);
            REQUIRE(flight);
            Vulkan::vk::CommandBuffers buffers = pool.Allocate(1);
            const VkCommandBuffer raw = buffers[0];
            const Vulkan::vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
            cmd.Begin({
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = nullptr,
            });
            const u32 begin = flight->StampSegment(cmd);
            // Everything inside the command buffer is a no-op in this mode.
            REQUIRE(flight->Stamp(cmd) == GP::NONE);
            {
                const GP::Scope scope{flight.get(), cmd, GP::Tag{.kind = GP::Kind::Copy, .count = 7}};
            }
            const u32 end = flight->StampSegment(cmd);
            REQUIRE(begin != GP::NONE);
            REQUIRE(end == begin + 1); // no stamp in between
            flight->Add(begin, end, GP::Tag{.kind = GP::Kind::Segment});
            cmd.End();
            const VkSubmitInfo submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &raw,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr,
            };
            Vulkan::vk::Check(device.GetGraphicsQueue().Submit(submit));
            void(device.GetLogical().WaitIdle());
            profiler->Complete(*flight);
        }
        REQUIRE(profiler->KindTotal(GP::Kind::Segment).first == frame * 2);
        REQUIRE(profiler->KindTotal(GP::Kind::Copy).first == 0);
        const auto [frames, span_ms] = profiler->FrameSpanTotal();
        REQUIRE(frames == frame); // one sample per present, not per command buffer
        REQUIRE(span_ms >= 0.0);
    }
}

namespace {

/// The calibration workload: a compute shader whose cost is linear in a push constant.
///   #version 450
///   layout(local_size_x = 64) in;
///   layout(std430, binding = 0) buffer Out { uint data[]; };
///   layout(push_constant) uniform PC { uint iterations; } pc;
///   void main() {
///       uint x = gl_GlobalInvocationID.x;
///       for (uint i = 0u; i < pc.iterations; ++i) { x = x * 1664525u + 1013904223u; }
///       data[gl_GlobalInvocationID.x] = x;
///   }
/// glslc -O --target-env=vulkan1.1 -mfmt=num (a dependent chain: one group is latency bound, the
/// shape of OoT's 1x1x1 dispatches that the old stamps read as ~1 us).
constexpr u32 BUSY_COMP_SPV[] = {
    0x07230203, 0x00010300, 0x000d000b, 0x00000037, 0x00000000, 0x00020011,
    0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005,
    0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00040047, 0x0000000b,
    0x0000000b, 0x0000001c, 0x00030047, 0x00000017, 0x00000002, 0x00050048,
    0x00000017, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000029,
    0x00000006, 0x00000004, 0x00030047, 0x0000002a, 0x00000002, 0x00050048,
    0x0000002a, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000002c,
    0x00000021, 0x00000000, 0x00040047, 0x0000002c, 0x00000022, 0x00000000,
    0x00040047, 0x00000034, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020,
    0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b,
    0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020,
    0x0000000d, 0x00000001, 0x00000006, 0x0003001e, 0x00000017, 0x00000006,
    0x00040020, 0x00000018, 0x00000009, 0x00000017, 0x0004003b, 0x00000018,
    0x00000019, 0x00000009, 0x00040015, 0x0000001a, 0x00000020, 0x00000001,
    0x0004002b, 0x0000001a, 0x0000001b, 0x00000000, 0x00040020, 0x0000001c,
    0x00000009, 0x00000006, 0x00020014, 0x0000001f, 0x0004002b, 0x00000006,
    0x00000022, 0x0019660d, 0x0004002b, 0x00000006, 0x00000024, 0x3c6ef35f,
    0x0004002b, 0x0000001a, 0x00000027, 0x00000001, 0x0003001d, 0x00000029,
    0x00000006, 0x0003001e, 0x0000002a, 0x00000029, 0x00040020, 0x0000002b,
    0x0000000c, 0x0000002a, 0x0004003b, 0x0000002b, 0x0000002c, 0x0000000c,
    0x00040020, 0x00000030, 0x0000000c, 0x00000006, 0x0004002b, 0x00000006,
    0x00000032, 0x00000040, 0x0004002b, 0x00000006, 0x00000033, 0x00000001,
    0x0006002c, 0x00000009, 0x00000034, 0x00000032, 0x00000033, 0x00000033,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c,
    0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x000200f9, 0x00000011,
    0x000200f8, 0x00000011, 0x000700f5, 0x00000006, 0x00000036, 0x0000000f,
    0x00000005, 0x00000025, 0x00000012, 0x000700f5, 0x00000006, 0x00000035,
    0x0000000c, 0x00000005, 0x00000028, 0x00000012, 0x00050041, 0x0000001c,
    0x0000001d, 0x00000019, 0x0000001b, 0x0004003d, 0x00000006, 0x0000001e,
    0x0000001d, 0x000500b0, 0x0000001f, 0x00000020, 0x00000035, 0x0000001e,
    0x000400f6, 0x00000013, 0x00000012, 0x00000000, 0x000400fa, 0x00000020,
    0x00000012, 0x00000013, 0x000200f8, 0x00000012, 0x00050084, 0x00000006,
    0x00000023, 0x00000036, 0x00000022, 0x00050080, 0x00000006, 0x00000025,
    0x00000023, 0x00000024, 0x00050080, 0x00000006, 0x00000028, 0x00000035,
    0x00000027, 0x000200f9, 0x00000011, 0x000200f8, 0x00000013, 0x00060041,
    0x00000030, 0x00000031, 0x0000002c, 0x0000001b, 0x0000000f, 0x0003003e,
    0x00000031, 0x00000036, 0x000100fd, 0x00010038,
};

struct BusyVma {
    VmaAllocator vma{};
    ~BusyVma() {
        if (vma) {
            vmaDestroyAllocator(vma);
        }
    }
};

/// One-group dispatches of the busy shader, each in its own `Kind::Dispatch` scope, measured by
/// the profiler and by the host (submit to fence).
class BusyRig {
public:
    explicit BusyRig(const Vulkan::Device& device_) : device{device_} {
        holder.vma = device.CreateThreadSafeAllocator();
        allocator = std::make_unique<Vulkan::MemoryAllocator>(device, holder.vma);
        const Vulkan::vk::Device& logical = device.GetLogical();
        buffer = allocator->CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = 64 * sizeof(u32),
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            Vulkan::MemoryUsage::DeviceLocal);
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
        const VkPushConstantRange push{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(u32),
        };
        const VkDescriptorSetLayout set_layout_handle = *set_layout;
        layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout_handle,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push,
        });
        shader = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = sizeof(BUSY_COMP_SPV),
            .pCode = BUSY_COMP_SPV,
        });
        pipeline = logical.CreateComputePipeline({
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage =
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = *shader,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
                },
            .layout = *layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = 0,
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
        sets = descriptor_pool.Allocate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = *descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &set_layout_handle,
        });
        const VkDescriptorBufferInfo buffer_info{
            .buffer = *buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = sets[0],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_info,
            .pTexelBufferView = nullptr,
        };
        logical.UpdateDescriptorSets(write, {});
        command_pool = logical.CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.GetGraphicsFamily(),
        });
    }

    ~BusyRig() {
        void(device.GetLogical().WaitIdle());
    }

    /// One command buffer (a segment) of `dispatches` dispatches of one group, each ordered after
    /// the previous one by a compute-to-compute barrier (a dependent chain, as the replay records
    /// behind a game barrier). Returns the host time from submit to the fence, nanoseconds.
    double Run(GP::Profiler& profiler, u32 iterations, u32 dispatches) {
        std::unique_ptr<GP::Flight> flight = profiler.NewFlight(true);
        REQUIRE(flight);
        Vulkan::vk::CommandBuffers buffers = command_pool.Allocate(1);
        const VkCommandBuffer raw = buffers[0];
        const Vulkan::vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
        cmd.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        const u32 begin = flight->StampSegment(cmd);
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        const VkDescriptorSet set = sets[0];
        cmd.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
        cmd.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, iterations);
        for (u32 i = 0; i < dispatches; ++i) {
            if (i != 0) {
                const VkMemoryBarrier chain{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, chain);
            }
            const GP::Scope scope{flight.get(), cmd,
                                  GP::Tag{.kind = GP::Kind::Dispatch, .count = 1, .width = 1,
                                          .height = 1, .shape = 1}};
            cmd.Dispatch(1, 1, 1);
        }
        const u32 end = flight->StampSegment(cmd);
        flight->Add(begin, end, GP::Tag{.kind = GP::Kind::Segment});
        cmd.End();
        Vulkan::vk::Fence fence = device.GetLogical().CreateFence({
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        });
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &raw,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        const auto start = std::chrono::steady_clock::now();
        Vulkan::vk::Check(device.GetGraphicsQueue().Submit(submit, *fence));
        Vulkan::vk::Check(fence.Wait());
        const auto stop = std::chrono::steady_clock::now();
        profiler.Complete(*flight);
        return std::chrono::duration<double, std::nano>(stop - start).count();
    }

private:
    const Vulkan::Device& device;
    BusyVma holder;
    std::unique_ptr<Vulkan::MemoryAllocator> allocator;
    Vulkan::vk::Buffer buffer;
    Vulkan::vk::DescriptorSetLayout set_layout;
    Vulkan::vk::PipelineLayout layout;
    Vulkan::vk::ShaderModule shader;
    Vulkan::vk::Pipeline pipeline;
    Vulkan::vk::DescriptorPool descriptor_pool;
    Vulkan::vk::DescriptorSets sets;
    Vulkan::vk::CommandPool command_pool;
};

/// Iterations of the busy loop: ~4 ms for one group on an RTX 5090, far above the stamp and
/// submission overheads.
constexpr u32 BUSY_ITERATIONS = 1u << 22;

} // namespace

TEST_CASE("nvn gpu profile: a dispatch's stamped time scales with its work", "[nvn]") {
    // Calibration (GpuTruth, 2026-11-05): a bare BOTTOM_OF_PIPE timestamp does not wait for
    // compute on NVIDIA, so these dispatches used to read ~1 us however long they ran. The
    // stamped time must follow the work: 4x the iterations, >= 2x the time, and every interval
    // must hold a real share of the host's submit-to-fence time.
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    std::unique_ptr<GP::Profiler> profiler = GP::Profiler::Create(device, 1, true);
    if (!profiler) {
        SKIP("the device cannot do timestamps or host query reset");
    }
    BusyRig rig{device};
    rig.Run(*profiler, 1024, 1); // pipeline warm-up
    const auto dispatch_ns = [&] { return profiler->KindTotal(GP::Kind::Dispatch).second; };
    const auto segment_ns = [&] { return profiler->KindTotal(GP::Kind::Segment).second; };

    double before = dispatch_ns();
    const double wall_1x = rig.Run(*profiler, BUSY_ITERATIONS, 1);
    const double stamped_1x = dispatch_ns() - before;
    before = dispatch_ns();
    const double wall_4x = rig.Run(*profiler, BUSY_ITERATIONS * 4, 1);
    const double stamped_4x = dispatch_ns() - before;
    INFO("1x: stamped " << stamped_1x / 1e6 << " ms, host " << wall_1x / 1e6 << " ms; 4x: stamped "
                        << stamped_4x / 1e6 << " ms, host " << wall_4x / 1e6 << " ms");
    REQUIRE(stamped_1x > 0.0);
    REQUIRE(stamped_4x >= 2.0 * stamped_1x);
    REQUIRE(stamped_1x >= 0.25 * wall_1x);
    REQUIRE(stamped_4x >= 0.25 * wall_4x);
    REQUIRE(stamped_4x <= wall_4x);

    // A dependent chain of four: the intervals add up to the chain and sit inside the segment.
    before = dispatch_ns();
    const double segment_before = segment_ns();
    const double wall_chain = rig.Run(*profiler, BUSY_ITERATIONS, 4);
    const double stamped_chain = dispatch_ns() - before;
    const double stamped_segment = segment_ns() - segment_before;
    INFO("chain of 4: stamped " << stamped_chain / 1e6 << " ms, segment " << stamped_segment / 1e6
                                << " ms, host " << wall_chain / 1e6 << " ms");
    REQUIRE(stamped_chain >= 3.0 * stamped_1x);
    REQUIRE(stamped_chain >= 0.25 * wall_chain);
    REQUIRE(stamped_segment >= stamped_chain);
}

TEST_CASE("nvn gpu profile: frame mode's command buffer time covers its dispatches", "[nvn]") {
    const auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    const Vulkan::Device& device = *vulkan->device;
    std::unique_ptr<GP::Profiler> profiler = GP::Profiler::Create(device, GP::MODE_FRAME, true);
    if (!profiler) {
        SKIP("the device cannot do timestamps or host query reset");
    }
    BusyRig rig{device};
    rig.Run(*profiler, 1024, 1); // pipeline warm-up
    const double segment_before = profiler->KindTotal(GP::Kind::Segment).second;
    const double wall = rig.Run(*profiler, BUSY_ITERATIONS * 4, 1);
    const double segment = profiler->KindTotal(GP::Kind::Segment).second - segment_before;
    INFO("segment " << segment / 1e6 << " ms, host " << wall / 1e6 << " ms");
    REQUIRE(profiler->KindTotal(GP::Kind::Dispatch).first == 0);
    REQUIRE(segment >= 0.25 * wall);
    REQUIRE(segment <= wall);
}
