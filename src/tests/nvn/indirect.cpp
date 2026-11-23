// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the NVN counters, reports and render conditions (Vulkan::Nvn::CounterEngine and the
// replay ops built on it). They need a GPU: every test SKIPs without a Vulkan device.

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include <atomic>

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/renderer_vulkan/nvn/nvn_counters.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace {

namespace VC = VideoCommon::Nvn;
using namespace Common::Literals;
using namespace std::chrono_literals;
using Vulkan::Nvn::CounterEngine;
using Vulkan::Nvn::CounterFlight;
namespace CT = Vulkan::Nvn::CounterType;
namespace CM = Vulkan::Nvn::ConditionMode;

// --- guest memory, record encoding, headless Vulkan ------------------------------------------

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
        if (!source) {
            return false;
        }
        std::memcpy(dst, source, size);
        return true;
    }

    bool Write(u64 va, const void* src, size_t size) override {
        u8* target = Find(va, size);
        if (!target) {
            return false;
        }
        std::memcpy(target, src, size);
        return true;
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

template <typename T>
void Push(std::vector<u64>& list, VC::Op op, const T& payload, u16 flags = 0) {
    constexpr size_t words = 1 + (sizeof(T) + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(op);
    header->flags = flags;
    header->size = static_cast<u32>(words * 8);
    std::memcpy(&list[at + 1], &payload, sizeof(T));
}

void PushEnd(std::vector<u64>& list) {
    const size_t at = list.size();
    list.resize(at + 1, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(VC::Op::End);
    header->flags = 0;
    header->size = 8;
}

VC::Submission MakeSubmission(u64 seq, u32 queue, std::vector<u64> list) {
    VC::Submission submission{};
    submission.seq = seq;
    submission.queue = queue;
    submission.lists.push_back(std::move(list));
    return submission;
}

struct HeadlessVulkan {
    std::shared_ptr<Common::DynamicLibrary> library;
    std::unique_ptr<Vulkan::vk::InstanceDispatch> dld;
    Vulkan::vk::Instance instance;
    std::unique_ptr<Vulkan::Device> device;
    std::mutex submit_mutex;
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
    } catch (const Vulkan::vk::Exception&) {
        return nullptr;
    } catch (const std::exception&) {
        return nullptr;
    }
}

constexpr u32 POOL_CPU_CACHED = 4;

template <typename T>
T Get(const u8* base, u64 offset) {
    T value;
    std::memcpy(&value, base + offset, sizeof(T));
    return value;
}

template <typename T>
void Put(u8* base, u64 offset, T value) {
    std::memcpy(base + offset, &value, sizeof(T));
}

u64 SteadyNanoseconds() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// A guest clock that shares no epoch with the host steady clock.
constexpr u64 GUEST_EPOCH_SHIFT = 1'000'000'000'000;

// --- engine rig: a counter engine plus a tiny render target and pipeline ---------------------

#include "indirect_test_shaders.inc"

struct VmaHolder {
    VmaAllocator vma{};
    ~VmaHolder() {
        if (vma) {
            vmaDestroyAllocator(vma);
        }
    }
};

class EngineRig {
public:
    static constexpr u32 SIZE = 64;
    static constexpr u64 SAMPLES_PER_DRAW = u64{SIZE} * SIZE;

    EngineRig(HeadlessVulkan& vulkan_, bool conditional_rendering)
        : vulkan{vulkan_}, device{*vulkan_.device} {
        holder.vma = device.CreateThreadSafeAllocator();
        allocator = std::make_unique<Vulkan::MemoryAllocator>(device, holder.vma);
        command_pool = device.GetLogical().CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.GetGraphicsFamily(),
        });
        engine = std::make_unique<CounterEngine>(
            device, *allocator,
            [this](const std::function<void(Vulkan::vk::CommandBuffer)>& record) {
                Immediate(record);
            },
            [](u64) { return true; }, conditional_rendering, true);
        MakeTarget();
    }

    ~EngineRig() {
        void(device.GetLogical().WaitIdle());
    }

    void Immediate(const std::function<void(Vulkan::vk::CommandBuffer)>& record) {
        Vulkan::vk::CommandBuffers buffers = command_pool.Allocate(1);
        const VkCommandBuffer raw = buffers[0];
        const Vulkan::vk::CommandBuffer cmd{raw, device.GetDispatchLoader()};
        cmd.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        record(cmd);
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
        {
            std::scoped_lock lock{vulkan.submit_mutex};
            Vulkan::vk::Check(device.GetGraphicsQueue().Submit(submit, *fence));
        }
        Vulkan::vk::Check(fence.Wait());
    }

    Vulkan::vk::Buffer HostBuffer(u64 size, VkBufferUsageFlags usage, Vulkan::MemoryUsage hint) {
        return allocator->CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = size,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            hint);
    }

    /// One render pass with `draws` full-target draws. With `gate`, every draw sits in its own
    /// conditional rendering scope on the predicate of `gate_slot`.
    void DrawPass(Vulkan::vk::CommandBuffer cmd, u32 draws, bool gate = false, u32 gate_slot = 0) {
        BeginPass(cmd);
        for (u32 i = 0; i < draws; ++i) {
            if (gate) {
                engine->BeginConditional(cmd, gate_slot);
            }
            cmd.Draw(3, 1, 0, 0);
            if (gate) {
                engine->EndConditional(cmd);
            }
        }
        cmd.EndRenderPass();
    }

    /// A render pass with the triangle pipeline bound (draws: cmd.Draw(3, 1, 0, 0)).
    void BeginPass(Vulkan::vk::CommandBuffer cmd) {
        const VkRenderPassBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = *render_pass,
            .framebuffer = *framebuffer,
            .renderArea = {{0, 0}, {SIZE, SIZE}},
            .clearValueCount = 0,
            .pClearValues = nullptr,
        };
        cmd.BeginRenderPass(begin, VK_SUBPASS_CONTENTS_INLINE);
        cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
    }

    [[nodiscard]] bool PreciseOcclusion() const {
        return device.GetPhysical().GetFeatures().occlusionQueryPrecise != VK_FALSE;
    }

    HeadlessVulkan& vulkan;
    const Vulkan::Device& device;
    VmaHolder holder;
    std::unique_ptr<Vulkan::MemoryAllocator> allocator;
    Vulkan::vk::CommandPool command_pool;
    std::unique_ptr<CounterEngine> engine;

private:
    void MakeTarget() {
        const Vulkan::vk::Device& logical = device.GetLogical();
        image = allocator->CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {SIZE, SIZE, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        view = logical.CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .components{},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        });
        const VkAttachmentDescription attachment{
            .flags = 0,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_GENERAL};
        const VkSubpassDescription subpass{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &reference,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        render_pass = logical.CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        });
        const VkImageView attachment_view = *view;
        framebuffer = logical.CreateFramebuffer({
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = *render_pass,
            .attachmentCount = 1,
            .pAttachments = &attachment_view,
            .width = SIZE,
            .height = SIZE,
            .layers = 1,
        });
        vertex_module = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = sizeof(TRIANGLE_VERT_SPV),
            .pCode = TRIANGLE_VERT_SPV,
        });
        fragment_module = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = sizeof(TRIANGLE_FRAG_SPV),
            .pCode = TRIANGLE_FRAG_SPV,
        });
        pipeline_layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        });
        const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = *vertex_module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = *fragment_module,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
        }};
        const VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
        };
        const VkPipelineInputAssemblyStateCreateInfo input_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };
        const VkViewport viewport{0.0f, 0.0f, float(SIZE), float(SIZE), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, {SIZE, SIZE}};
        const VkPipelineViewportStateCreateInfo viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };
        const VkPipelineRasterizationStateCreateInfo rasterization{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,
        };
        const VkPipelineMultisampleStateCreateInfo multisample{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 0.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
        };
        const VkPipelineColorBlendAttachmentState blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = 0xF,
        };
        const VkPipelineColorBlendStateCreateInfo blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &blend_attachment,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
        };
        pipeline = logical.CreateGraphicsPipeline(
            VkGraphicsPipelineCreateInfo{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stageCount = static_cast<u32>(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pTessellationState = nullptr,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterization,
                .pMultisampleState = &multisample,
                .pDepthStencilState = nullptr,
                .pColorBlendState = &blend,
                .pDynamicState = nullptr,
                .layout = *pipeline_layout,
                .renderPass = *render_pass,
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0,
            },
            VK_NULL_HANDLE);
    }

    Vulkan::vk::Image image;
    Vulkan::vk::ImageView view;
    Vulkan::vk::RenderPass render_pass;
    Vulkan::vk::Framebuffer framebuffer;
    Vulkan::vk::ShaderModule vertex_module;
    Vulkan::vk::ShaderModule fragment_module;
    Vulkan::vk::PipelineLayout pipeline_layout;
    Vulkan::vk::Pipeline pipeline;
};

} // Anonymous namespace

TEST_CASE("NVN indirect: occlusion samples accumulate across queries and render passes",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    EngineRig rig{*vulkan, true};
    CounterEngine& engine = *rig.engine;
    constexpr u32 QUEUE = 3;
    Vulkan::vk::Buffer result = rig.HostBuffer(
        64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Vulkan::MemoryUsage::Download);

    CounterFlight flight;
    rig.Immediate([&](Vulkan::vk::CommandBuffer cmd) {
        // 1: one draw.
        auto query = engine.BeginOcclusion(cmd, flight);
        rig.DrawPass(cmd, 1);
        engine.FoldOcclusion(cmd, flight, query, QUEUE, 1);
        engine.CopyAccumulator(cmd, QUEUE, *result, 0);
        // 2: the query spans two whole render passes (three draws): the total is cumulative.
        query = engine.BeginOcclusion(cmd, flight);
        rig.DrawPass(cmd, 2);
        rig.DrawPass(cmd, 1);
        engine.FoldOcclusion(cmd, flight, query, QUEUE, 1);
        engine.CopyAccumulator(cmd, QUEUE, *result, 16);
        // 3: a reset starts from zero again; a query in flight when it happens is discarded.
        query = engine.BeginOcclusion(cmd, flight);
        rig.DrawPass(cmd, 1);
        engine.EndOcclusion(cmd, flight, query);
        engine.ResetAccumulator(cmd, QUEUE);
        query = engine.BeginOcclusion(cmd, flight);
        rig.DrawPass(cmd, 1);
        engine.FoldOcclusion(cmd, flight, query, QUEUE, 1);
        engine.CopyAccumulator(cmd, QUEUE, *result, 32);
        // 4: another queue's accumulator never saw a draw.
        engine.CopyAccumulator(cmd, QUEUE + 1, *result, 48);
    });
    result.Invalidate();
    const u8* data = result.Mapped().data();
    constexpr u64 DRAW = EngineRig::SAMPLES_PER_DRAW;
    if (rig.PreciseOcclusion()) {
        CHECK(Get<u64>(data, 0) == DRAW);
        CHECK(Get<u64>(data, 16) == 4 * DRAW);
        CHECK(Get<u64>(data, 32) == DRAW);
    } else {
        CHECK(Get<u64>(data, 0) > 0);
        CHECK(Get<u64>(data, 16) > Get<u64>(data, 0));
    }
    CHECK(Get<u64>(data, 48) == 0);
    CHECK(engine.Stats().occlusion_folds.load() == 3);
}

TEST_CASE("NVN indirect: a counter batch knows which report bytes are still to be written",
          "[nvn]") {
    using Vulkan::Nvn::CounterBatch;
    CounterBatch batch;
    const int pool_a = 0;
    const int pool_b = 0;
    CHECK(batch.Empty());
    CHECK(batch.Room(CounterBatch::MAX_EVENTS));
    CHECK_FALSE(batch.Room(CounterBatch::MAX_EVENTS + 1));
    batch.AddQuery({0, 0});
    batch.AddReset();
    batch.AddReset(); // a reset after a reset is no event
    CHECK(batch.events.size() == 2);
    CHECK_FALSE(batch.HasReports());
    CHECK_FALSE(batch.Writes(&pool_a, 0, 1_MiB));
    batch.AddReport(VK_NULL_HANDLE, 0x100, &pool_a);
    CHECK(batch.HasReports());
    CHECK(batch.Writes(&pool_a, 0x100, 8));
    CHECK(batch.Writes(&pool_a, 0x104, 4));   // the high word of the counter
    CHECK(batch.Writes(&pool_a, 0xF0, 0x20)); // a range around it
    CHECK_FALSE(batch.Writes(&pool_a, 0x108, 8)); // the time field (the completion thread's)
    CHECK_FALSE(batch.Writes(&pool_a, 0xF8, 8));
    CHECK_FALSE(batch.Writes(&pool_b, 0x100, 8));
    CHECK(batch.Room(CounterBatch::MAX_EVENTS - 3));
    CHECK_FALSE(batch.Room(CounterBatch::MAX_EVENTS - 2));
    batch.Clear();
    CHECK(batch.Empty());
    CHECK_FALSE(batch.HasReports());
    CHECK_FALSE(batch.Writes(&pool_a, 0x100, 8));
}

TEST_CASE("NVN indirect: in-pass occlusion queries resolve to cumulative reports", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    EngineRig rig{*vulkan, true};
    CounterEngine& engine = *rig.engine;
    if (!engine.InPassQueries()) {
        SKIP("no host query reset: occlusion queries span whole render passes");
    }
    constexpr u32 QUEUE = 5;
    constexpr u32 BIG = 200; // queries of one batch: more than a query block holds
    Vulkan::vk::Buffer result = rig.HostBuffer(
        128, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Vulkan::MemoryUsage::Download);
    CounterFlight flight;
    Vulkan::Nvn::CounterBatch batch;
    const auto counted = [&](Vulkan::vk::CommandBuffer cmd, u32 draws) {
        const auto query = engine.BeginOcclusion(cmd, flight);
        for (u32 i = 0; i < draws; ++i) {
            cmd.Draw(3, 1, 0, 0);
        }
        engine.EndOcclusion(cmd, flight, query);
        batch.AddQuery(query);
    };
    const auto report = [&](u64 offset) { batch.AddReport(*result, offset, nullptr); };

    rig.Immediate([&](Vulkan::vk::CommandBuffer cmd) {
        cmd.FillBuffer(*result, 0, 128, 0xCDCDCDCDu);
        // 1: one pass with three reports and a reset between them, written once the pass ended.
        rig.BeginPass(cmd);
        counted(cmd, 1);
        report(0);
        counted(cmd, 2);
        report(8);
        batch.AddReset();
        counted(cmd, 1);
        report(16);
        cmd.EndRenderPass();
        engine.ResolveBatch(cmd, flight, batch, QUEUE, 1);
        CHECK(batch.Empty());
        // 2: the queries of a pass without a report count into the next one (no reset between).
        rig.BeginPass(cmd);
        counted(cmd, 1);
        cmd.EndRenderPass();
        rig.BeginPass(cmd);
        counted(cmd, 1);
        report(24);
        cmd.EndRenderPass();
        engine.ResolveBatch(cmd, flight, batch, QUEUE, 1);
        // 3: the same report address twice in one batch: the later value stays.
        rig.BeginPass(cmd);
        report(32);
        counted(cmd, 1);
        report(32);
        cmd.EndRenderPass();
        engine.ResolveBatch(cmd, flight, batch, QUEUE, 1);
        // 4: the accumulator holds the batch total; another queue's never saw a draw.
        engine.CopyAccumulator(cmd, QUEUE, *result, 40);
        engine.CopyAccumulator(cmd, QUEUE + 1, *result, 48);
        // 5: a reset, then more queries than a query block holds, in one pass and one batch.
        rig.BeginPass(cmd);
        batch.AddReset();
        for (u32 i = 0; i < BIG; ++i) {
            counted(cmd, 1);
        }
        report(56);
        cmd.EndRenderPass();
        engine.ResolveBatch(cmd, flight, batch, QUEUE, 1);
        // 6: a batch ending in a reset leaves the accumulator at zero.
        batch.AddReset();
        engine.ResolveBatch(cmd, flight, batch, QUEUE, 1);
        engine.CopyAccumulator(cmd, QUEUE, *result, 64);
    });
    result.Invalidate();
    const u8* data = result.Mapped().data();
    constexpr u64 DRAW = EngineRig::SAMPLES_PER_DRAW;
    if (rig.PreciseOcclusion()) {
        CHECK(Get<u64>(data, 0) == DRAW);
        CHECK(Get<u64>(data, 8) == 3 * DRAW);
        CHECK(Get<u64>(data, 16) == DRAW);
        CHECK(Get<u64>(data, 24) == 3 * DRAW);
        CHECK(Get<u64>(data, 32) == 4 * DRAW);
        CHECK(Get<u64>(data, 40) == 4 * DRAW);
        CHECK(Get<u64>(data, 56) == BIG * DRAW);
    } else {
        CHECK(Get<u64>(data, 0) > 0);
        CHECK(Get<u64>(data, 8) > Get<u64>(data, 0));
        CHECK(Get<u64>(data, 56) > 0);
    }
    CHECK(Get<u64>(data, 48) == 0);
    CHECK(Get<u64>(data, 64) == 0);
    CHECK(engine.Stats().occlusion_folds.load() == 5);
}

TEST_CASE("NVN indirect: a counter batch longer than one resolve dispatch keeps the running total",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    EngineRig rig{*vulkan, true};
    CounterEngine& engine = *rig.engine;
    if (!engine.InPassQueries()) {
        SKIP("no host query reset: occlusion queries span whole render passes");
    }
    constexpr u32 QUEUE = 3;
    // A query and a report per draw, a reset in the middle: 2 * 300 + 1 events, three dispatches
    // of RESOLVE_CHUNK_EVENTS, a reset and reports on both sides of a chunk boundary.
    constexpr u32 DRAWS = 300;
    constexpr u32 RESET_AFTER = 150;
    static_assert(2 * DRAWS + 1 > 2 * CounterEngine::RESOLVE_CHUNK_EVENTS);
    static_assert(2 * DRAWS + 1 <= Vulkan::Nvn::CounterBatch::MAX_EVENTS);
    constexpr u64 ACCUMULATOR = DRAWS * 8;
    Vulkan::vk::Buffer result = rig.HostBuffer(ACCUMULATOR + 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               Vulkan::MemoryUsage::Download);
    CounterFlight flight;
    Vulkan::Nvn::CounterBatch batch;
    rig.Immediate([&](Vulkan::vk::CommandBuffer cmd) {
        // Zero: an untouched slot fails the `value > 0` check of non-precise devices.
        cmd.FillBuffer(*result, 0, ACCUMULATOR + 8, 0);
        engine.ResetAccumulator(cmd, QUEUE);
        rig.BeginPass(cmd);
        for (u32 i = 0; i < DRAWS; ++i) {
            if (i == RESET_AFTER) {
                batch.AddReset();
            }
            const auto query = engine.BeginOcclusion(cmd, flight);
            cmd.Draw(3, 1, 0, 0);
            engine.EndOcclusion(cmd, flight, query);
            batch.AddQuery(query);
            batch.AddReport(*result, u64{i} * 8, nullptr);
        }
        cmd.EndRenderPass();
        engine.ResolveBatch(cmd, flight, batch, QUEUE, 1);
        engine.CopyAccumulator(cmd, QUEUE, *result, ACCUMULATOR);
    });
    result.Invalidate();
    const u8* data = result.Mapped().data();
    constexpr u64 DRAW = EngineRig::SAMPLES_PER_DRAW;
    for (u32 i = 0; i < DRAWS; ++i) {
        INFO("report " << i);
        const u64 value = Get<u64>(data, u64{i} * 8);
        const u64 counted = i < RESET_AFTER ? i + 1 : i - RESET_AFTER + 1;
        if (rig.PreciseOcclusion()) {
            CHECK(value == counted * DRAW);
        } else {
            CHECK(value > 0);
        }
    }
    if (rig.PreciseOcclusion()) {
        CHECK(Get<u64>(data, ACCUMULATOR) == (DRAWS - RESET_AFTER) * DRAW);
    }
    CHECK(engine.Stats().occlusion_folds.load() == 1);
}

TEST_CASE("NVN counters: resolve areas come from a ring reused oldest first", "[nvn]") {
    Vulkan::Nvn::AreaRing ring{256};
    std::vector<std::pair<u64, u32>> waits;
    const auto wait = [&](u64 seq, u32 begin) { waits.emplace_back(seq, begin); };
    // Rounded up to 16 bytes, handed out in order.
    CHECK(ring.Allocate(1, 1, wait) == 0);
    CHECK(ring.Allocate(1, 100, wait) == 16);
    CHECK(ring.Allocate(2, 96, wait) == 128);
    CHECK(ring.Live() == 3);
    CHECK(waits.empty());
    // 48 bytes do not fit behind 224: the tail is skipped and the oldest areas at 0 and 16 are
    // reused, seq 1's after a wait for it.
    CHECK(ring.Allocate(3, 48, wait) == 0);
    REQUIRE(waits.size() == 2);
    CHECK((waits[0] == std::pair<u64, u32>{1, 0}));
    CHECK((waits[1] == std::pair<u64, u32>{1, 16}));
    CHECK(ring.Live() == 2);
    // The same submission's own areas are reused without a wait.
    waits.clear();
    CHECK(ring.Allocate(2, 160, wait) == 48);
    CHECK(waits.empty());
    CHECK(ring.Live() == 2);
    // Up to the end of the ring exactly, then the area at 0 (seq 3) is next.
    CHECK(ring.Allocate(2, 48, wait) == 208);
    CHECK(ring.Allocate(4, 16, wait) == 0);
    REQUIRE(waits.size() == 1);
    CHECK((waits[0] == std::pair<u64, u32>{3, 0}));
}

TEST_CASE("NVN indirect: a render condition compares the 64-bit counters of a report pair",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    EngineRig rig{*vulkan, true};
    CounterEngine& engine = *rig.engine;
    if (!engine.GpuConditions()) {
        SKIP("VK_EXT_conditional_rendering is not available");
    }
    constexpr u32 QUEUE = 1;
    struct Case {
        u64 a;
        u64 b;
        u32 mode;
        bool renders;
    };
    const Case cases[] = {
        {5, 5, CM::RenderIfEqual, true},
        {5, 5, CM::RenderIfNotEqual, false},
        {5, 7, CM::RenderIfEqual, false},
        {5, 7, CM::RenderIfNotEqual, true},
        // Same low word, different high word: the comparison is 64-bit.
        {0x1'0000'0005, 0x2'0000'0005, CM::RenderIfEqual, false},
        {0x1'0000'0005, 0x2'0000'0005, CM::RenderIfNotEqual, true},
        // The DK pattern: one counter set by the CPU, the other never written.
        {1, 0, CM::RenderIfNotEqual, true},
        {0, 0, CM::RenderIfNotEqual, false},
        {0, 0, CM::RenderIfEqual, true},
    };
    u64 seq = 1;
    for (const Case& c : cases) {
        INFO("a=" << c.a << " b=" << c.b << " mode=" << c.mode);
        Vulkan::vk::Buffer pair = rig.HostBuffer(
            32, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Vulkan::MemoryUsage::Upload);
        std::memset(pair.Mapped().data(), 0xEE, 32);
        Put<u64>(pair.Mapped().data(), 0, c.a);
        Put<u64>(pair.Mapped().data(), 16, c.b);
        pair.Flush();
        Vulkan::vk::Buffer result = rig.HostBuffer(
            16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Vulkan::MemoryUsage::Download);
        CounterFlight flight;
        rig.Immediate([&](Vulkan::vk::CommandBuffer cmd) {
            engine.ResetAccumulator(cmd, QUEUE);
            const u32 slot = engine.AllocateSlot(seq++);
            engine.RecordCondition(cmd, *pair, 0, *pair, 16, c.mode, slot);
            const auto query = engine.BeginOcclusion(cmd, flight);
            rig.DrawPass(cmd, 2, true, slot); // both draws are gated
            engine.FoldOcclusion(cmd, flight, query, QUEUE, seq);
            engine.CopyAccumulator(cmd, QUEUE, *result, 0);
        });
        result.Invalidate();
        const u64 samples = Get<u64>(result.Mapped().data(), 0);
        if (c.renders) {
            CHECK(samples > 0);
            if (rig.PreciseOcclusion()) {
                CHECK(samples == 2 * EngineRig::SAMPLES_PER_DRAW);
            }
        } else {
            CHECK(samples == 0);
        }
    }
}

TEST_CASE("NVN indirect: counter and value reports reach the guest, on both pool paths", "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    for (const bool import : {true, false}) {
        INFO("host import " << import);
        for (const bool real : {true, false}) {
        INFO("real timestamps " << real);
        VC::SetTimestampModeOverride(real ? VC::TimestampMode::Real : VC::TimestampMode::Legacy);
        // `real` runs as a Switch 2 title (the field is nanoseconds) on a guest clock that is
        // the steady clock plus a fixed epoch shift: reports must land on the guest clock.
        Settings::SetCurrentProgramIsSwitch2(real);
        VC::SetGuestClockSource([] { return SteadyNanoseconds() + GUEST_EPOCH_SHIFT; });
        constexpr u64 FAKE_TICKS = 0x1234'0000;
        std::atomic<u64> next_tick{FAKE_TICKS};
        VC::SetLegacyTickSource([&] { return next_tick.fetch_add(1); });
        // Restored even when an assertion aborts, after the Context below stopped.
        SCOPE_EXIT {
            VC::SetTimestampModeOverride(std::nullopt);
            VC::SetLegacyTickSource({});
            VC::SetGuestClockSource({});
            Settings::SetCurrentProgramIsSwitch2(false);
        };
        constexpr u64 POOL_VA = 0x1000'0000;
        FakeMemory memory;
        u8* data = memory.Add(POOL_VA, 64_KiB);
        std::memset(data, 0xAB, 64_KiB);
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                     {.allow_host_import = import});
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});

        std::vector<u64> list;
        const auto report = [&](u32 type, u64 offset) {
            Push(list, VC::Op::ReportCounter,
                 VC::ReportCounterPayload{.address = POOL_VA + offset, .counter_type = type});
        };
        report(CT::Timestamp, 0x100);
        report(CT::TimestampTop, 0x120);
        report(CT::SamplesPassed, 0x140); // no draws: zero samples
        report(0x7, 0x160);               // GEOMETRY_SHADER_INVOCATIONS: not counted
        Push(list, VC::Op::ResetCounter, VC::ResetCounterPayload{.counter_type = CT::SamplesPassed});
        report(CT::SamplesPassed, 0x180);
        Push(list, VC::Op::ReportValue, VC::ReportValuePayload{.address = POOL_VA + 0x1A0,
                                                                .value = 0xCAFE});
        PushEnd(list);
        const u64 before = SteadyNanoseconds() + GUEST_EPOCH_SHIFT;
        context.Submit(MakeSubmission(1, 0, std::move(list)));
        REQUIRE(context.Wait(1, 20s));
        const u64 after = SteadyNanoseconds() + GUEST_EPOCH_SHIFT;

        // Counter 0 in both modes. `real`: a time between the submit and now (the GPU timeline
        // mapped onto the guest clock). `legacy`: the guest GPU tick source's value (one per
        // report, in order), whatever the GPU did.
        for (const u64 offset : {u64{0x100}, u64{0x120}}) {
            CHECK(Get<u64>(data, offset) == 0);
            const u64 time = Get<u64>(data, offset + 8);
            if (real) {
                CHECK(time + 2'000'000'000 > before);
                CHECK(time < after + 2'000'000'000);
            } else {
                CHECK(time >= FAKE_TICKS);
                CHECK(time < FAKE_TICKS + 16);
            }
        }
        if (!real) {
            // Every report, SAMPLES_PASSED included, carries a tick (the type does not matter).
            for (const u64 offset : {u64{0x148}, u64{0x168}, u64{0x188}}) {
                CHECK(Get<u64>(data, offset) >= FAKE_TICKS);
                CHECK(Get<u64>(data, offset) < FAKE_TICKS + 16);
            }
        }
        CHECK(Get<u64>(data, 0x140) == 0);
        CHECK(Get<u64>(data, 0x148) != 0xABABABABABABABABull);
        CHECK(Get<u64>(data, 0x160) == 0);
        CHECK(Get<u64>(data, 0x180) == 0);
        CHECK(Get<u32>(data, 0x1A0) == 0xCAFE);
        CHECK(Get<u32>(data, 0x1A4) == 0xABABABAB); // ReportValue is 4 bytes
        // Report ordering: the second timestamp was taken after the first.
        CHECK(Get<u64>(data, 0x128) >= Get<u64>(data, 0x108));

        const Vulkan::Nvn::ContextStats stats = context.Stats();
        CHECK(stats.counter_reports_timestamp == 2);
        CHECK(stats.counter_reports_samples == 2);
        CHECK(stats.counter_reports_unsupported == 1);
        CHECK(stats.counter_reports_value == 1);
        CHECK(stats.ops_failed == 0);
        // The pool path that ran: the import is only covered when the device took it.
        if (!import) {
            CHECK(stats.pools_imported == 0);
        } else if (stats.pools_imported == 0) {
            WARN("host import unavailable: the reports ran on the mirror path only");
        }
        } // real
    }     // import
}

TEST_CASE("NVN indirect: render conditions evaluated on the CPU read the report pair",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    for (const bool import : {true, false}) {
        INFO("host import " << import);
        constexpr u64 POOL_VA = 0x2000'0000;
        FakeMemory memory;
        u8* data = memory.Add(POOL_VA, 64_KiB);
        std::memset(data, 0, 64_KiB);
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                     {.allow_host_import = import,
                                      .allow_conditional_rendering = false});
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                            .flags = POOL_CPU_CACHED});

        // Pair 0x100: the CPU wrote 1 and nothing else (the DK pattern). Pair 0x200: a timestamp
        // report on the GPU (counter 0) against a CPU-written 1.
        Put<u64>(data, 0x100, 1);
        Put<u64>(data, 0x110, 0);
        Put<u64>(data, 0x210, 1);
        const auto condition = [](u32 mode, u64 address) {
            return VC::SetRenderConditionPayload{.kind = 0, .mode = mode, .address = address};
        };
        std::vector<u64> list;
        Push(list, VC::Op::SetRenderCondition, condition(CM::RenderIfNotEqual, POOL_VA + 0x100)); // renders
        Push(list, VC::Op::SetRenderCondition, condition(CM::RenderIfEqual, POOL_VA + 0x100));    // off
        Push(list, VC::Op::SetRenderCondition, VC::SetRenderConditionPayload{.kind = 1});
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = POOL_VA + 0x200, .counter_type = CT::Timestamp});
        Push(list, VC::Op::SetRenderCondition, condition(CM::RenderIfEqual, POOL_VA + 0x200));    // off
        Push(list, VC::Op::SetRenderCondition, condition(CM::RenderIfNotEqual, POOL_VA + 0x200)); // renders
        // The submission goes on after every flush.
        Push(list, VC::Op::ReportValue, VC::ReportValuePayload{.address = POOL_VA + 0x300,
                                                                .value = 0xBEEF});
        // An address outside every pool: counted, rendering stays on.
        Push(list, VC::Op::SetRenderCondition, condition(CM::RenderIfEqual, 0x9000'0000));
        PushEnd(list);
        context.Submit(MakeSubmission(1, 0, std::move(list)));
        REQUIRE(context.Wait(1, 20s));

        const Vulkan::Nvn::ContextStats stats = context.Stats();
        CHECK(stats.render_conditions_cpu == 4);
        CHECK(stats.render_conditions_false == 2);
        CHECK(stats.render_conditions_gpu == 0);
        CHECK(stats.render_conditions_unresolved == 1);
        CHECK(Get<u32>(data, 0x300) == 0xBEEF);
        CHECK(Get<u64>(data, 0x200) == 0); // the GPU-written counter the CPU path read
    }
}

TEST_CASE("NVN indirect: render conditions are evaluated on the GPU when the device can",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    constexpr u64 POOL_VA = 0x3000'0000;
    FakeMemory memory;
    u8* data = memory.Add(POOL_VA, 64_KiB);
    Put<u64>(data, 0x100, 1);
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = POOL_VA, .size = 64_KiB,
                        .flags = POOL_CPU_CACHED});
    std::vector<u64> list;
    for (u32 i = 0; i < 3; ++i) {
        Push(list, VC::Op::SetRenderCondition,
             VC::SetRenderConditionPayload{.kind = 0, .mode = i & 1, .address = POOL_VA + 0x100});
        Push(list, VC::Op::SetRenderCondition, VC::SetRenderConditionPayload{.kind = 1});
    }
    Push(list, VC::Op::SetRenderCondition, VC::SetRenderConditionPayload{.kind = 2});
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list)));
    REQUIRE(context.Wait(1, 20s));
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    if (vulkan->device->IsExtConditionalRendering()) {
        CHECK(stats.render_conditions_cpu == 0);
        if (stats.pools_imported != 0) {
            // The first condition computes both modes of the pair in one batch; the second and
            // the third (mode 0 again) are served by those predicates.
            CHECK(stats.render_conditions_gpu == 2);
            CHECK(stats.render_conditions_prefetched == 2);
        } else {
            // Mirrored pools take CPU writes at the next upload: each condition at its own point.
            CHECK(stats.render_conditions_gpu == 3);
            CHECK(stats.render_conditions_prefetched == 0);
        }
    } else {
        CHECK(stats.render_conditions_gpu == 0);
        CHECK(stats.render_conditions_cpu == 3);
    }
    CHECK(stats.ops_failed == 0);
}

// --- render conditions gating the draw path's clears (the composite clear DK wraps) ----------

namespace {

constexpr u64 CLEAR_TEXTURE_POOL_VA = 0x1000'0000;
constexpr u64 CLEAR_BUFFER_POOL_VA = 0x2000'0000;
constexpr u32 CLEAR_SIZE = 64;
constexpr u64 SNAPSHOT_STRIDE = 0x4000;
constexpr u64 REPORT_OFFSET = 0x20000;

template <typename T>
void PushWith(std::vector<u64>& list, VC::Op op, const T& prefix, std::span<const u8> extra) {
    const size_t bytes = sizeof(T) + extra.size();
    const size_t words = 1 + (bytes + 7) / 8;
    const size_t at = list.size();
    list.resize(at + words, 0);
    auto* header = reinterpret_cast<VC::RecordHeader*>(&list[at]);
    header->op = static_cast<u16>(op);
    header->flags = 0;
    header->size = static_cast<u32>(words * 8);
    auto* payload = reinterpret_cast<u8*>(&list[at + 1]);
    std::memcpy(payload, &prefix, sizeof(T));
    std::memcpy(payload + sizeof(T), extra.data(), extra.size());
}

void PushClear(std::vector<u64>& list, float r, float g, float b) {
    VC::ClearColorPayload clear{};
    clear.index = 0;
    clear.mask = 0xF;
    const float color[4] = {r, g, b, 1.0f};
    std::memcpy(clear.color, color, sizeof(color));
    Push(list, VC::Op::ClearColor, clear);
}

void PushSnapshot(std::vector<u64>& list, u64 index) {
    VC::CopyBufferTexturePayload copy{};
    copy.buffer = CLEAR_BUFFER_POOL_VA + index * SNAPSHOT_STRIDE;
    copy.texture = {.serial = 10, .level = 0, .layer = 0};
    copy.region = {0, 0, 0, CLEAR_SIZE, CLEAR_SIZE, 1};
    Push(list, VC::Op::CopyTextureToBuffer, copy);
}

void PushCondition(std::vector<u64>& list, u32 kind, u32 mode = 0, u64 address = 0) {
    Push(list, VC::Op::SetRenderCondition,
         VC::SetRenderConditionPayload{.kind = kind, .mode = mode, .address = address});
}

bool Pixel(const u8* snapshot, int r, int g, int b) {
    const auto close_to = [](u8 value, int expected) { return std::abs(int(value) - expected) <= 1; };
    return close_to(snapshot[0], r) && close_to(snapshot[1], g) && close_to(snapshot[2], b) &&
           close_to(snapshot[3], 255);
}

} // Anonymous namespace

TEST_CASE("NVN indirect: ClearColor under SetRenderEnableConditional follows the report pair",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    for (const bool gpu_conditions : {true, false}) {
        INFO("conditions on the " << (gpu_conditions ? "GPU" : "CPU"));
        FakeMemory memory;
        memory.Add(CLEAR_TEXTURE_POOL_VA, 1_MiB);
        u8* buffers = memory.Add(CLEAR_BUFFER_POOL_VA, 1_MiB);
        std::memset(buffers, 0xCD, 1_MiB);
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                     {.allow_conditional_rendering = gpu_conditions});
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = CLEAR_TEXTURE_POOL_VA, .size = 1_MiB,
                            .flags = 1});
        context.CreatePool({.serial = 2, .guest_address = CLEAR_BUFFER_POOL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_CACHED});
        VC::TextureDesc texture{};
        texture.target = 1;
        texture.format = 0x25;
        texture.width = CLEAR_SIZE;
        texture.height = CLEAR_SIZE;
        texture.depth = 1;
        texture.levels = 1;
        texture.serial = 10;
        texture.pool_serial = 1;
        context.CreateTexture(texture);

        // The pattern DK uses (TOTK as well): the CPU sets the first counter of a report pair to 1
        // when its target needs a clear; the clear is followed by a TIMESTAMP_TOP report on that
        // address (counter 0) and the second counter is never written.
        const u64 report = CLEAR_BUFFER_POOL_VA + REPORT_OFFSET;
        Put<u64>(buffers, REPORT_OFFSET, 1);
        Put<u64>(buffers, REPORT_OFFSET + 16, 0);
        // A second pair that is all zero.
        Put<u64>(buffers, REPORT_OFFSET + 0x100, 0);
        Put<u64>(buffers, REPORT_OFFSET + 0x110, 0);

        VC::RenderTargetRef target{};
        target.texture = {.serial = 10, .level = 0, .layer = 0};
        std::vector<u64> list;
        PushWith(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
                 {reinterpret_cast<const u8*>(&target), sizeof(target)});
        PushClear(list, 0.0f, 0.0f, 1.0f); // blue, unconditional
        // 1 counters 1 / 0, render if not equal: the red clear happens ...
        PushCondition(list, 0, CM::RenderIfNotEqual, report);
        PushClear(list, 1.0f, 0.0f, 0.0f);
        Push(list, VC::Op::ReportCounter,
             VC::ReportCounterPayload{.address = report, .counter_type = CT::TimestampTop});
        PushCondition(list, 1);
        PushSnapshot(list, 0); // red
        // 2 ... the report made the counters 0 / 0: the same condition now skips the green clear.
        PushCondition(list, 0, CM::RenderIfNotEqual, report);
        PushClear(list, 0.0f, 1.0f, 0.0f);
        PushCondition(list, 1);
        // 3 SetRenderEnable(false) skips the yellow clear.
        PushCondition(list, 2);
        PushClear(list, 1.0f, 1.0f, 0.0f);
        PushCondition(list, 1);
        PushSnapshot(list, 1); // still red
        // 4 counters 0 / 0, render if equal: the magenta clear happens.
        PushCondition(list, 0, CM::RenderIfEqual, report + 0x100);
        PushClear(list, 1.0f, 0.0f, 1.0f);
        PushCondition(list, 1);
        PushSnapshot(list, 2); // magenta
        PushEnd(list);
        context.Submit(MakeSubmission(1, 0, std::move(list)));
        REQUIRE(context.Wait(1, 20s));

        const u8* first = buffers;
        for (u64 snapshot = 0; snapshot < 3; ++snapshot) {
            const u8* p = buffers + snapshot * SNAPSHOT_STRIDE;
            const u8* q = p + (CLEAR_SIZE * CLEAR_SIZE - 1) * 4;
            UNSCOPED_INFO("snapshot " << snapshot << " first " << int(p[0]) << "," << int(p[1]) << "," << int(p[2]) << "," << int(p[3]) << " last " << int(q[0]) << "," << int(q[1]) << "," << int(q[2]) << "," << int(q[3]));
        }
        CHECK(Pixel(first, 255, 0, 0));
        CHECK(Pixel(first + (CLEAR_SIZE * CLEAR_SIZE - 1) * 4, 255, 0, 0));
        CHECK(Pixel(buffers + SNAPSHOT_STRIDE, 255, 0, 0));
        CHECK(Pixel(buffers + 2 * SNAPSHOT_STRIDE, 255, 0, 255));
        CHECK(Get<u64>(buffers, REPORT_OFFSET) == 0); // the GPU wrote the counter

        const Vulkan::Nvn::ContextStats stats = context.Stats();
        CHECK(stats.ops_failed == 0);
        CHECK(stats.ops_unsupported == 0);
        if (gpu_conditions && vulkan->device->IsExtConditionalRendering()) {
            CHECK(stats.render_conditions_gpu == 3);
            CHECK(stats.render_ops_gated == 3);   // red, green, magenta
            CHECK(stats.render_ops_skipped == 1); // yellow
        } else {
            CHECK(stats.render_conditions_cpu == 3);
            CHECK(stats.render_conditions_false == 1);
            CHECK(stats.render_ops_skipped == 2); // green, yellow
        }
    }
}

TEST_CASE("NVN indirect: indirect draws that cannot run are counted, not dropped silently",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    FakeMemory memory;
    memory.Add(CLEAR_TEXTURE_POOL_VA, 1_MiB);
    u8* buffers = memory.Add(CLEAR_BUFFER_POOL_VA, 1_MiB);
    Put<u32>(buffers, 0, 3); // vertex count of an indirect command
    Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr, {});
    context.SetGuestMemory(&memory);
    context.CreatePool({.serial = 1, .guest_address = CLEAR_TEXTURE_POOL_VA, .size = 1_MiB,
                        .flags = 1});
    context.CreatePool({.serial = 2, .guest_address = CLEAR_BUFFER_POOL_VA, .size = 1_MiB,
                        .flags = POOL_CPU_CACHED});
    VC::TextureDesc texture{};
    texture.target = 1;
    texture.format = 0x25;
    texture.width = CLEAR_SIZE;
    texture.height = CLEAR_SIZE;
    texture.depth = 1;
    texture.levels = 1;
    texture.serial = 10;
    texture.pool_serial = 1;
    context.CreateTexture(texture);

    VC::RenderTargetRef target{};
    target.texture = {.serial = 10, .level = 0, .layer = 0};
    std::vector<u64> list;
    PushWith(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
             {reinterpret_cast<const u8*>(&target), sizeof(target)});
    VC::DrawPayload draw{};
    draw.instance_count = 1;
    draw.max_draws = 1;
    draw.indirect = CLEAR_BUFFER_POOL_VA;
    draw.mode = 2; // LINE_LOOP: needs indices built from the GPU-written count
    Push(list, VC::Op::Draw, draw);
    draw.mode = 4;
    draw.indirect = 0x9000'0000; // not in any pool
    Push(list, VC::Op::Draw, draw);
    draw.indirect = CLEAR_BUFFER_POOL_VA;
    draw.count_address = CLEAR_BUFFER_POOL_VA + 64;
    draw.max_draws = 4;
    draw.stride = 6; // not a multiple of 4
    Push(list, VC::Op::Draw, draw);
    PushEnd(list);
    context.Submit(MakeSubmission(1, 0, std::move(list)));
    REQUIRE(context.Wait(1, 20s));
    const Vulkan::Nvn::ContextStats stats = context.Stats();
    CHECK(stats.indirect_unsupported == 2); // LINE_LOOP and the odd stride
    CHECK(stats.indirect_draws == 0);
    CHECK(stats.ops_failed >= 1); // the address outside every pool
}

TEST_CASE("NVN indirect: a timestamp report inside an open render pass is written in full",
          "[nvn]") {
    auto vulkan = MakeHeadlessVulkan();
    if (!vulkan) {
        SKIP("no Vulkan device");
    }
    for (const bool import : {true, false}) {
        INFO("host import " << import);
        FakeMemory memory;
        memory.Add(CLEAR_TEXTURE_POOL_VA, 1_MiB);
        u8* buffers = memory.Add(CLEAR_BUFFER_POOL_VA, 1_MiB);
        std::memset(buffers, 0xAB, 1_MiB);
        // The overrides set below are restored even when an assertion aborts, after the Context
        // stopped.
        SCOPE_EXIT {
            VC::SetTimestampModeOverride(std::nullopt);
            VC::SetGuestClockSource({});
            Settings::SetCurrentProgramIsSwitch2(false);
        };
        // No early submission: a piece would end the pass the reports must land in.
        Vulkan::Nvn::Context context(*vulkan->device, vulkan->submit_mutex, nullptr,
                                     {.allow_host_import = import, .early_submit = false});
        context.SetGuestMemory(&memory);
        context.CreatePool({.serial = 1, .guest_address = CLEAR_TEXTURE_POOL_VA, .size = 1_MiB,
                            .flags = 1});
        context.CreatePool({.serial = 2, .guest_address = CLEAR_BUFFER_POOL_VA, .size = 1_MiB,
                            .flags = POOL_CPU_CACHED});
        VC::TextureDesc texture{};
        texture.target = 1;
        texture.format = 0x25;
        texture.width = CLEAR_SIZE;
        texture.height = CLEAR_SIZE;
        texture.depth = 1;
        texture.levels = 1;
        texture.serial = 10;
        texture.pool_serial = 1;
        context.CreateTexture(texture);

        VC::RenderTargetRef target{};
        target.texture = {.serial = 10, .level = 0, .layer = 0};
        std::vector<u64> list;
        PushWith(list, VC::Op::SetRenderTargets, VC::SetRenderTargetsPayload{1, 0},
                 {reinterpret_cast<const u8*>(&target), sizeof(target)});
        PushClear(list, 1.0f, 0.0f, 0.0f);
        // The pass of the clear is still open here.
        const u64 before = SteadyNanoseconds() + GUEST_EPOCH_SHIFT;
        for (u64 i = 0; i < 3; ++i) {
            Push(list, VC::Op::ReportCounter,
                 VC::ReportCounterPayload{.address = CLEAR_BUFFER_POOL_VA + REPORT_OFFSET + i * 16,
                                          .counter_type = i == 1 ? CT::TimestampTop
                                                                 : CT::Timestamp});
            PushClear(list, 0.0f, 1.0f, 0.0f);
        }
        PushSnapshot(list, 0);
        PushEnd(list);
        // GPU time on the guest clock, in a Switch 2 title's unit (nanoseconds).
        VC::SetTimestampModeOverride(VC::TimestampMode::Real);
        Settings::SetCurrentProgramIsSwitch2(true);
        VC::SetGuestClockSource([] { return SteadyNanoseconds() + GUEST_EPOCH_SHIFT; });
        context.Submit(MakeSubmission(1, 0, std::move(list)));
        REQUIRE(context.Wait(1, 20s));
        VC::SetTimestampModeOverride(std::nullopt);
        VC::SetGuestClockSource({});
        Settings::SetCurrentProgramIsSwitch2(false);
        const u64 after = SteadyNanoseconds() + GUEST_EPOCH_SHIFT;
        u64 previous = 0;
        for (u64 i = 0; i < 3; ++i) {
            const u64 counter = Get<u64>(buffers, REPORT_OFFSET + i * 16);
            const u64 time = Get<u64>(buffers, REPORT_OFFSET + i * 16 + 8);
            CHECK(counter == 0);
            CHECK(time + 2'000'000'000 > before);
            CHECK(time < after + 2'000'000'000);
            CHECK(time >= previous);
            previous = time;
        }
        CHECK(Pixel(buffers, 0, 255, 0));
        const Vulkan::Nvn::ContextStats stats = context.Stats();
        CHECK(stats.ops_failed == 0);
        // The pool path that ran: the import is only covered when the device took it.
        if (!import) {
            CHECK(stats.pools_imported == 0);
        } else if (stats.pools_imported == 0) {
            WARN("host import unavailable: the reports ran on the mirror path only");
        }
    }
}
