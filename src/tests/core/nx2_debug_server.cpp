// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <limits>
#include <thread>

#include <boost/asio.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common/fs/file.h"
#include "common/fs/fs_util.h"
#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/debugger/nx2_debug_server.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "tests/temp_path.h"
#include "video_core/nvn_draw_probe.h"
#include "video_core/nx2_debug_control.h"

namespace {

using namespace std::chrono_literals;
using Core::Nx2Debug::ClientGone;
using Core::Nx2Debug::Handlers;
using nlohmann::json;

Handlers EchoHandlers() {
    return {
        {"echo", [](const json& params, const ClientGone&) { return params; }},
        {"fail", [](const json&, const ClientGone&) -> json { throw std::runtime_error{"broken"}; }},
    };
}

void MapPage(Common::PageTable& table, u64 page, u8* host, Common::PageType type) {
    auto& entry = table.entries.GetAndFault(page);
    entry.Store(false, Common::PageType::Memory, 1, reinterpret_cast<uintptr_t>(host) - (page << 12));
    if (type == Common::PageType::RasterizerCachedMemory) {
        entry.MarkRasterizerCached();
    }
}

/// Guest pages 4, 5, 7 (rasterizer cached) and 8 over host pages 1, 0, 2 and 3 of a zeroed
/// backing; page 6 and everything else are unmapped.
struct GuestPages {
    GuestPages() {
        system.Initialize();
        auto& table = process.GetPageTable().GetImpl();
        table.Resize(20, 12);
        backing = system.DeviceMemory().buffer.BackingBasePointer() + 0x10000;
        std::memset(backing, 0, 0x4000);
        MapPage(table, 4, backing + 0x1000, Common::PageType::Memory);
        MapPage(table, 5, backing, Common::PageType::Memory);
        MapPage(table, 7, backing + 0x2000, Common::PageType::RasterizerCachedMemory);
        MapPage(table, 8, backing + 0x3000, Common::PageType::Memory);
        process.GetMemory().SetCurrentPageTable(process);
    }
    ~GuestPages() {
        system.Kernel().UnregisterKernelObject(&process);
    }

    ScopedSystemTestContext context;
    Core::System system;
    Kernel::KProcess process{system.Kernel()};
    u8* backing{};
};

} // Anonymous namespace

TEST_CASE("Debug server endpoints are loopback only", "[core][nx2_debug_server]") {
    std::string error;
    const auto port_only = Core::Nx2Debug::ParseEndpoint("7777", error);
    REQUIRE(port_only);
    REQUIRE(port_only->host == "127.0.0.1");
    REQUIRE(port_only->port == 7777);
    const auto localhost = Core::Nx2Debug::ParseEndpoint("localhost:1234", error);
    REQUIRE(localhost);
    REQUIRE(localhost->host == "127.0.0.1");
    const auto ipv6 = Core::Nx2Debug::ParseEndpoint("[::1]:80", error);
    REQUIRE(ipv6);
    REQUIRE(ipv6->host == "::1");
    REQUIRE_FALSE(Core::Nx2Debug::ParseEndpoint("0.0.0.0:7777", error));
    REQUIRE(error.find("loopback") != std::string::npos);
    REQUIRE_FALSE(Core::Nx2Debug::ParseEndpoint("192.168.1.2:7777", error));
    REQUIRE_FALSE(Core::Nx2Debug::ParseEndpoint("127.0.0.1:", error));
    REQUIRE_FALSE(Core::Nx2Debug::ParseEndpoint("127.0.0.1:70000", error));
    REQUIRE_FALSE(Core::Nx2Debug::ParseEndpoint("12ab", error));
}

TEST_CASE("Debug frame times measure intervals and retain the ring tail", "[core][nx2_debug_server]") {
    auto& ring = Core::g_nx2_present_times_ns;
    SCOPE_EXIT {
        for (auto& timestamp : ring) {
            timestamp.store(0);
        }
    };
    const auto read = [](u64 after, u64 end_ns) {
        return Core::Nx2Debug::ReadFrameTimes(0, after, 1, end_ns, 60.0);
    };
    REQUIRE(read(0, 1)["frames"] == 0);
    ring[1] = 1;
    REQUIRE(read(1, 1)["mean_ms"] == 0.0);
    ring[2] = 16'666'668;
    ring[3] = 50'000'001;
    const auto mixed = read(3, 50'000'001);
    REQUIRE(mixed["frames"] == 2);
    REQUIRE(mixed["mean_ms"] == 25.0);
    REQUIRE(mixed["p50_ms"] == 25.0);
    REQUIRE(mixed["fps"] == 40.0);
    REQUIRE(mixed["vsync_buckets"]["1"] == 0.5);
    REQUIRE(mixed["vsync_buckets"]["2"] == 0.5);
    REQUIRE(mixed["intervals_ms"] == json::array({16.7, 33.3}));
    REQUIRE(mixed["stutters"] == 0);
    REQUIRE(mixed["truncated"] == false);
    ring[4] = 150'000'001;
    REQUIRE(read(4, 150'000'001)["stutters"] == 1);
    // An overwritten slot must not make an interval across the missing sample.
    ring[2] = 200'000'001;
    REQUIRE(read(3, 50'000'001)["frames"] == 0);
    REQUIRE(read(3, 50'000'001)["truncated"] == true);
    for (u64 frame = 1; frame <= 5000; ++frame) {
        ring[frame % ring.size()] = frame * 1'000'000;
    }
    const auto overflow = read(5000, 5'000'000'000);
    REQUIRE(overflow["frames"] == ring.size() - 1);
    REQUIRE(overflow["mean_ms"] == 1.0);
    REQUIRE(overflow["truncated"] == true);
    REQUIRE(overflow["intervals_ms"].size() == 120);
    // A counted present whose timestamp store is pending still holds the previous lap's value.
    ring[5000 % ring.size()] = (5000 - ring.size()) * 1'000'000;
    const auto pending = read(5000, 5'000'000'000);
    REQUIRE(pending["frames"] == ring.size() - 2);
    REQUIRE(pending["max_ms"] == 1.0);
}

TEST_CASE("Debug server requests map to handlers", "[core][nx2_debug_server]") {
    const Handlers handlers = EchoHandlers();
    REQUIRE(json::parse(Core::Nx2Debug::HandleRequest(
                handlers, R"({"id":7,"method":"echo","params":{"a":1}})")) ==
            json{{"id", 7}, {"result", {{"a", 1}}}});
    REQUIRE(json::parse(Core::Nx2Debug::HandleRequest(handlers, R"({"id":"x","method":"echo"})")) ==
            json{{"id", "x"}, {"result", json::object()}});
    REQUIRE(json::parse(Core::Nx2Debug::HandleRequest(handlers, R"({"id":1,"method":"fail"})")) ==
            json{{"id", 1}, {"error", "broken"}});
    const json unknown = json::parse(Core::Nx2Debug::HandleRequest(handlers, R"({"id":2,"method":"nope"})"));
    REQUIRE(unknown["id"] == 2);
    REQUIRE(unknown["error"].get<std::string>().find("unknown method") != std::string::npos);
    const json garbage = json::parse(Core::Nx2Debug::HandleRequest(handlers, "{not json"));
    REQUIRE(garbage["id"].is_null());
    REQUIRE(garbage.contains("error"));
    const json bad_params =
        json::parse(Core::Nx2Debug::HandleRequest(handlers, R"({"id":3,"method":"echo","params":[1]})"));
    REQUIRE(bad_params.contains("error"));
}

TEST_CASE("Debug server shader hashes need all 16 hex digits", "[core][nx2_debug_server]") {
    REQUIRE(Core::Nx2Debug::ParseShaderHash("00000000621c4755") == 0x621c4755);
    REQUIRE(Core::Nx2Debug::ParseShaderHash("3DDFB1D7621C4755") == 0x3ddfb1d7621c4755);
    REQUIRE(Core::Nx2Debug::ParseShaderHash("0x3ddfb1d7621c4755") == 0x3ddfb1d7621c4755);
    for (const char* bad : {"621c4755", "3ddfb1d7", "0x621c4755", "", "0x", "3ddfb1d7621c47551",
                            "3ddfb1d7621c475g", "-ddfb1d7621c4755", " 3ddfb1d7621c475"}) {
        try {
            (void)Core::Nx2Debug::ParseShaderHash(bad);
            FAIL("accepted " << bad);
        } catch (const std::runtime_error& error) {
            REQUIRE(std::string_view{error.what()}.find("16 hex digits") != std::string_view::npos);
        }
    }
}

TEST_CASE("Debug server base64 pads like RFC 4648", "[core][nx2_debug_server]") {
    const auto encode = [](std::string_view text) {
        return Core::Nx2Debug::Base64(
            {reinterpret_cast<const u8*>(text.data()), text.size()});
    };
    REQUIRE(encode("") == "");
    REQUIRE(encode("f") == "Zg==");
    REQUIRE(encode("fo") == "Zm8=");
    REQUIRE(encode("foo") == "Zm9v");
    REQUIRE(encode("foobar") == "Zm9vYmFy");
}

TEST_CASE("Debug server answers over loopback TCP", "[core][nx2_debug_server]") {
    Core::Nx2Debug::Server server{{"127.0.0.1", 0}, EchoHandlers()};
    REQUIRE(server.Port() != 0);
    boost::asio::io_context io;
    boost::asio::ip::tcp::socket socket{io};
    socket.connect({boost::asio::ip::make_address("127.0.0.1"), server.Port()});
    const std::string requests =
        "{\"id\":1,\"method\":\"echo\",\"params\":{\"n\":1}}\n\r\n{\"id\":2,\"method\":\"fail\"}\r\n";
    boost::asio::write(socket, boost::asio::buffer(requests));
    std::string input;
    for (int line = 0; line < 2; ++line) {
        const size_t length = boost::asio::read_until(socket, boost::asio::dynamic_buffer(input), '\n');
        const json response = json::parse(input.substr(0, length - 1));
        input.erase(0, length);
        if (line == 0) {
            REQUIRE(response == json{{"id", 1}, {"result", {{"n", 1}}}});
        } else {
            REQUIRE(response == json{{"id", 2}, {"error", "broken"}});
        }
    }
}

TEST_CASE("Debug control hands captures to the rasterizer once", "[core][nx2_debug_server]") {
    VideoCore::Nx2Debug::BeginSession();
    REQUIRE_FALSE(VideoCore::Nx2Debug::HasRequests());
    VideoCore::Nx2Debug::RequestCapture({.id = 5, .directory = "dir", .sampled = true, .buffers = true});
    REQUIRE(VideoCore::Nx2Debug::HasRequests());
    const auto request = VideoCore::Nx2Debug::TakeCapture();
    REQUIRE(request);
    REQUIRE(request->id == 5);
    REQUIRE(request->sampled);
    REQUIRE(request->buffers);
    REQUIRE_FALSE(VideoCore::Nx2Debug::HasRequests());
    REQUIRE_FALSE(VideoCore::Nx2Debug::TakeCapture());
    REQUIRE_FALSE(VideoCore::Nx2Debug::WaitCapture(5, 1ms));
    std::thread writer{[] { VideoCore::Nx2Debug::FinishCapture(5, {.error = "disk full"}); }};
    const auto result = VideoCore::Nx2Debug::WaitCapture(5, 10s);
    writer.join();
    REQUIRE(result);
    REQUIRE(result->error == "disk full");

    VideoCore::Nx2Debug::RequestCapture({.id = 6, .directory = "dir", .sampled = false, .buffers = false});
    VideoCore::Nx2Debug::CancelCapture(6);
    REQUIRE_FALSE(VideoCore::Nx2Debug::HasRequests());

    // Ending the session wakes a waiter instead of leaving it to its timeout.
    std::thread stopper{[] {
        std::this_thread::sleep_for(20ms);
        VideoCore::Nx2Debug::EndSession();
    }};
    const auto begin = std::chrono::steady_clock::now();
    REQUIRE_FALSE(VideoCore::Nx2Debug::WaitCapture(7, 30s));
    stopper.join();
    REQUIRE(std::chrono::steady_clock::now() - begin < 10s);
}

TEST_CASE("Debug control counts reject generations", "[core][nx2_debug_server]") {
    VideoCore::Nx2Debug::BeginSession();
    REQUIRE_FALSE(VideoCore::Nx2Debug::AnyRejected());
    REQUIRE_FALSE(VideoCore::Nx2Debug::IsRejected(0x1234));
    const std::array<u64, 2> hashes{0x1234, 0x5678};
    const u64 generation = VideoCore::Nx2Debug::SetRejected(hashes, true);
    REQUIRE(VideoCore::Nx2Debug::IsRejected(0x1234));
    REQUIRE(VideoCore::Nx2Debug::SetRejected(hashes, true) == generation);
    REQUIRE_FALSE(VideoCore::Nx2Debug::WaitRejectApplied(generation, 1ms));
    const auto change = VideoCore::Nx2Debug::TakeRejectChange();
    REQUIRE(change);
    REQUIRE_FALSE(VideoCore::Nx2Debug::TakeRejectChange());
    VideoCore::Nx2Debug::NoteRejectApplied(change->generation);
    REQUIRE(VideoCore::Nx2Debug::WaitRejectApplied(generation, 1ms));
    const std::array<u64, 2> both{0x1234, 0x5678};
    VideoCore::Nx2Debug::SetRejected(both, false);
    REQUIRE_FALSE(VideoCore::Nx2Debug::AnyRejected());
    REQUIRE(VideoCore::Nx2Debug::TakeRejectChange()->unrejected);
    VideoCore::Nx2Debug::EndSession();
}

TEST_CASE("Debug memory reads guest pages and names the first unmapped byte", "[core][nx2_debug_server]") {
    GuestPages guest;
    for (u32 i = 0; i < 0x4000; ++i) {
        guest.backing[i] = static_cast<u8>(i * 7 + (i >> 12));
    }
    auto& memory = guest.process.GetMemory();
    std::array<u8, 0x20> out{};
    REQUIRE_FALSE(Core::Nx2Debug::ReadGuestMemory(memory, 0x4ff0, out));
    REQUIRE(std::memcmp(out.data(), guest.backing + 0x1ff0, 0x10) == 0);
    REQUIRE(std::memcmp(out.data() + 0x10, guest.backing, 0x10) == 0);
    REQUIRE_FALSE(Core::Nx2Debug::ReadGuestMemory(memory, 0x7000, out));
    REQUIRE(std::memcmp(out.data(), guest.backing + 0x2000, out.size()) == 0);
    REQUIRE(Core::Nx2Debug::ReadGuestMemory(memory, 0x5ff0, out) == u64{0x6000});
    REQUIRE(Core::Nx2Debug::ReadGuestMemory(memory, 0x6800, out) == u64{0x6800});
    REQUIRE(Core::Nx2Debug::ReadGuestMemory(memory, 0x100000, out) == u64{0x100000});
    // The top 16 bits would alias page 4 through GetPointerSilent's mask.
    REQUIRE(Core::Nx2Debug::ReadGuestMemory(memory, (1ULL << 48) | 0x4000, out) ==
            ((1ULL << 48) | 0x4000));
}

TEST_CASE("Debug memory search spans mapped pages, stops at limits and resumes", "[core][nx2_debug_server]") {
    GuestPages guest;
    constexpr std::array<u8, 5> pattern{0xde, 0xad, 0xbe, 0xef, 0x11};
    const auto plant = [&](u8* host, size_t first, size_t count) {
        std::memcpy(host, pattern.data() + first, count);
    };
    plant(guest.backing + 0x1ffe, 0, 2); // 0x4ffe, continued on page 5
    plant(guest.backing, 2, 3);
    plant(guest.backing + 0x0ffe, 0, 2); // 0x5ffe, cut off by unmapped page 6
    plant(guest.backing + 0x2010, 0, 5); // 0x7010
    plant(guest.backing + 0x3000, 0, 5); // 0x8000
    plant(guest.backing + 0x3100, 0, 5); // 0x8100
    auto& memory = guest.process.GetMemory();
    const auto later = std::chrono::steady_clock::now() + 60s;

    const auto all = Core::Nx2Debug::SearchGuestMemory(memory, pattern, 0x1000, 0x20000, 256, later);
    REQUIRE(all.hits == std::vector<u64>{0x4ffe, 0x7010, 0x8000, 0x8100});
    REQUIRE(all.complete);
    REQUIRE(all.scanned == 0x4000);

    const auto first = Core::Nx2Debug::SearchGuestMemory(memory, pattern, 0x1000, 0x20000, 2, later);
    REQUIRE(first.hits == std::vector<u64>{0x4ffe, 0x7010});
    REQUIRE_FALSE(first.complete);
    REQUIRE(first.next == 0x7011);
    const auto rest = Core::Nx2Debug::SearchGuestMemory(memory, pattern, first.next, 0x20000, 2, later);
    REQUIRE(rest.hits == std::vector<u64>{0x8000, 0x8100});

    const auto inside = Core::Nx2Debug::SearchGuestMemory(memory, pattern, 0x4fff, 0x8004, 256, later);
    REQUIRE(inside.hits == std::vector<u64>{0x7010});

    const auto expired = Core::Nx2Debug::SearchGuestMemory(memory, pattern, 0x4000, 0x9000, 256,
                                                           std::chrono::steady_clock::now());
    REQUIRE(expired.hits.empty());
    REQUIRE_FALSE(expired.complete);
    REQUIRE(expired.next == 0x4000);
}

TEST_CASE("Debug memory patterns are hex byte pairs", "[core][nx2_debug_server]") {
    const std::vector<u8> one{0x00, 0x00, 0x80, 0x3f};
    REQUIRE(Core::Nx2Debug::ParseHexBytes("0000803f") == one);
    REQUIRE(Core::Nx2Debug::ParseHexBytes("0x00 00 80 3F") == one);
    for (const char* bad : {"", "0x", "abc", "zz", "0x-1"}) {
        CAPTURE(bad);
        REQUIRE_THROWS_AS(Core::Nx2Debug::ParseHexBytes(bad), std::runtime_error);
    }
}

TEST_CASE("Debug memory formats as hex, words and floats", "[core][nx2_debug_server]") {
    std::array<u8, 16> bytes{};
    const std::array<float, 4> floats{1.0f, std::numeric_limits<float>::quiet_NaN(),
                                      -std::numeric_limits<float>::infinity(), -2.5f};
    std::memcpy(bytes.data(), floats.data(), bytes.size());
    REQUIRE(Core::Nx2Debug::FormatMemory(std::span{bytes}.first(4), "hex") ==
            json{{"hex", "0000803f"}});
    REQUIRE(Core::Nx2Debug::FormatMemory(bytes, "f32") ==
            json{{"f32", {1.0, "nan", "-inf", -2.5}}});
    REQUIRE(Core::Nx2Debug::FormatMemory(std::span{bytes}.first(4), "u32") ==
            json{{"u32", {0x3f800000u}}});
    REQUIRE(Core::Nx2Debug::FormatMemory(std::span{bytes}.first(8), "u64") ==
            json{{"u64", {"0x7fc000003f800000"}}});
    REQUIRE_THROWS_AS(Core::Nx2Debug::FormatMemory(std::span{bytes}.first(12), "u64"),
                      std::runtime_error);
    REQUIRE_THROWS_AS(Core::Nx2Debug::FormatMemory(bytes, "f16"), std::runtime_error);
}

TEST_CASE("Debug control hands GPU reads to the rasterizer once", "[core][nx2_debug_server]") {
    VideoCore::Nx2Debug::BeginSession();
    VideoCore::Nx2Debug::RequestGpuRead({.id = 3, .gpu_address = 0x1000, .size = 8});
    REQUIRE(VideoCore::Nx2Debug::HasRequests());
    const auto request = VideoCore::Nx2Debug::TakeGpuRead();
    REQUIRE(request);
    REQUIRE(request->gpu_address == 0x1000);
    REQUIRE(request->size == 8);
    REQUIRE_FALSE(VideoCore::Nx2Debug::HasRequests());
    REQUIRE_FALSE(VideoCore::Nx2Debug::TakeGpuRead());
    std::thread rasterizer{[] {
        VideoCore::Nx2Debug::GpuReadResult result{};
        result.bytes = {1, 2, 3};
        result.address_space = 9;
        VideoCore::Nx2Debug::FinishGpuRead(3, std::move(result));
    }};
    const auto result = VideoCore::Nx2Debug::WaitGpuRead(3, 10s);
    rasterizer.join();
    REQUIRE(result);
    REQUIRE(result->bytes == std::vector<u8>{1, 2, 3});
    REQUIRE(result->address_space == 9);

    VideoCore::Nx2Debug::RequestGpuRead({.id = 4, .gpu_address = 0, .size = 4});
    VideoCore::Nx2Debug::CancelGpuRead(4);
    REQUIRE_FALSE(VideoCore::Nx2Debug::HasRequests());
    REQUIRE_FALSE(VideoCore::Nx2Debug::WaitGpuRead(4, 1ms));
    VideoCore::Nx2Debug::EndSession();
}

TEST_CASE("Debug server tells a waiting handler that its client left", "[core][nx2_debug_server]") {
    std::atomic<bool> noticed{};
    Handlers handlers{{"wait", [&noticed](const json&, const ClientGone& client_gone) -> json {
                           const auto deadline = std::chrono::steady_clock::now() + 10s;
                           while (!client_gone()) {
                               if (std::chrono::steady_clock::now() >= deadline) {
                                   return {};
                               }
                               std::this_thread::sleep_for(2ms);
                           }
                           noticed = true;
                           return {};
                       }}};
    Core::Nx2Debug::Server server{{"127.0.0.1", 0}, std::move(handlers)};
    {
        boost::asio::io_context io;
        boost::asio::ip::tcp::socket socket{io};
        socket.connect({boost::asio::ip::make_address("127.0.0.1"), server.Port()});
        boost::asio::write(socket, boost::asio::buffer(std::string{"{\"id\":1,\"method\":\"wait\"}\n"}));
        std::this_thread::sleep_for(50ms);
    }
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!noticed && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
    REQUIRE(noticed);
}

TEST_CASE("Debug input steps parse buttons, sticks and frames", "[core][nx2_debug_server]") {
    const auto step = Core::Nx2Debug::ParseInputStep(
        json{{"buttons", {"A", "ZL"}}, {"left_stick", {1.0, -0.5}}, {"frames", 3}});
    REQUIRE(step.frames == 3);
    REQUIRE(step.input.buttons == (Core::HID::NpadButton::A | Core::HID::NpadButton::ZL));
    REQUIRE(step.input.left->x == Core::HID::HID_JOYSTICK_MAX);
    REQUIRE(step.input.left->y == -16384);
    REQUIRE_FALSE(step.input.right);
    const auto idle = Core::Nx2Debug::ParseInputStep(json{{"frames", 1}});
    REQUIRE(idle.input.buttons == Core::HID::NpadButton::None);
    for (const json& bad : {json{{"buttons", {"A"}}}, json{{"frames", 0}}, json{{"frames", 36001}},
                            json{{"frames", 1}, {"buttons", {"Home"}}},
                            json{{"frames", 1}, {"stick_l", {0, 0}}},
                            json{{"frames", 1}, {"left_stick", {1.5, 0}}},
                            json{{"frames", 1}, {"right_stick", {0}}}, json::array()}) {
        CAPTURE(bad.dump());
        REQUIRE_THROWS_AS(Core::Nx2Debug::ParseInputStep(bad), std::exception);
    }
}

TEST_CASE("Debug input steps follow the present count exactly", "[core][nx2_debug_server]") {
    std::vector<Core::Nx2Debug::InputStep> steps(3);
    steps[0] = {.input = {.buttons = Core::HID::NpadButton::A, .left = {}, .right = {}}, .frames = 2};
    steps[1] = {.input = {}, .frames = 1};
    steps[2] = {.input = {.buttons = Core::HID::NpadButton::B, .left = {}, .right = {}}, .frames = 1};
    REQUIRE_FALSE(Core::Nx2Debug::InputAt(steps, 10, 9));
    REQUIRE(Core::Nx2Debug::InputAt(steps, 10, 10)->buttons == Core::HID::NpadButton::A);
    REQUIRE(Core::Nx2Debug::InputAt(steps, 10, 11)->buttons == Core::HID::NpadButton::A);
    REQUIRE(Core::Nx2Debug::InputAt(steps, 10, 12)->buttons == Core::HID::NpadButton::None);
    REQUIRE(Core::Nx2Debug::InputAt(steps, 10, 13)->buttons == Core::HID::NpadButton::B);
    REQUIRE_FALSE(Core::Nx2Debug::InputAt(steps, 10, 14));
}

TEST_CASE("Injected input merges buttons and replaces sticks while installed", "[core][nx2_debug_server]") {
    Core::HID::EmulatedController controller{Core::HID::NpadIdType::Player1};
    REQUIRE(controller.GetNpadButtons().raw == Core::HID::NpadButton::None);
    Core::HID::InjectedInput input{.buttons = Core::HID::NpadButton::A | Core::HID::NpadButton::ZR,
                                   .left = Core::HID::AnalogStickState{.x = -30000, .y = 20000},
                                   .right = {}};
    bool active = true;
    controller.SetInputInjector([&]() -> std::optional<Core::HID::InjectedInput> {
        return active ? std::optional{input} : std::nullopt;
    });
    using enum Core::HID::NpadButton;
    REQUIRE(controller.GetNpadButtons().raw == (A | ZR | StickLLeft | StickLUp));
    REQUIRE(controller.GetSticks().left.x == -30000);
    REQUIRE(controller.GetSticks().left.y == 20000);
    REQUIRE(controller.GetSticks().right.x == 0);
    active = false;
    REQUIRE(controller.GetNpadButtons().raw == None);
    active = true;
    controller.SetInputInjector({});
    REQUIRE(controller.GetNpadButtons().raw == None);
    REQUIRE(controller.GetSticks().left.x == 0);
}

TEST_CASE("Debug server draw hashes take 1 to 16 hex digits", "[core][nx2_debug_server]") {
    REQUIRE(Core::Nx2Debug::ParseDrawHash("621c4755") == 0x621c4755);
    REQUIRE(Core::Nx2Debug::ParseDrawHash("0x3DDFB1D7621C4755") == 0x3ddfb1d7621c4755);
    for (const char* bad : {"", "0x", "3ddfb1d7621c47551", "3ddfb1d7621c475g", " 621c4755"}) {
        REQUIRE_THROWS_AS(Core::Nx2Debug::ParseDrawHash(bad), std::runtime_error);
    }
}

TEST_CASE("Debug server nvn_stats keeps the latest heartbeat lines", "[core][nx2_debug_server]") {
    const std::vector<std::string> lines{
        "[1] Core <Info> [nvn] status: presents=10",
        "[1] Core <Info> [nvn] memory: synthetic lists 1",
        "[1] Render.Vulkan <Info> [nvn] gpuprof 10 s window (10.0 s): presents=300",
        "[1] Render.Vulkan <Info> [nvn] gpuprof 10 s window: gpu ms per frame old",
        "[1] Render.Vulkan <Info> [nvn] storage census: 5 images, estimated MiB (count): old",
        "[2] Core <Info> [nvn] status: presents=20",
        "[2] Core <Info> [nvn] replay stage=3 seq=9 completed=8",
        "[2] Render.Vulkan <Info> [nvn] gpuprof 10 s window (10.0 s): presents=600",
        "[2] Render.Vulkan <Info> [nvn] gpuprof 10 s window: gpu ms per frame new",
        "[2] Render.Vulkan <Info> [nvn] gpuprof tags top  1: other",
        "[2] Render.Vulkan <Info> [nvn] storage census: 7 images, estimated MiB (count): new",
        "[2] Render.Vulkan <Info> [nvn] storage census: cache holders MiB (count): a",
        "[2] Render.Vulkan <Info> [nvn] storage census: #1 64x64x1",
    };
    const json stats = Core::Nx2Debug::NvnStatsFromLog(lines, 2);
    REQUIRE(stats["status"] == lines[5]);
    REQUIRE(stats["memory"] == lines[1]);
    REQUIRE(stats["replay_stage"] == lines[6]);
    REQUIRE(stats["gpuprof_window"] == json{lines[7], lines[8]});
    REQUIRE(stats["storage_census"] == json{lines[10], lines[11]});
    REQUIRE(stats["storage_census_total"] == 3);
    REQUIRE_FALSE(stats.contains("note"));
    const json empty = Core::Nx2Debug::NvnStatsFromLog({}, 2);
    REQUIRE(empty["memory"].is_null());
    REQUIRE(empty["gpuprof_window"].empty());
    REQUIRE(empty.contains("note"));
}

TEST_CASE("NVN draw probe skips by stage hash and takes a census per full frame",
          "[core][nx2_debug_server]") {
    namespace Probe = VideoCommon::NvnDrawProbe;
    Probe::Reset();
    SCOPE_EXIT {
        Probe::Reset();
    };
    REQUIRE_FALSE(Probe::Active());
    Probe::DrawInfo a{};
    a.hashes[0] = 0xa0;
    a.hashes[1] = 0xa1;
    a.count = 3;
    Probe::DrawInfo b{};
    b.hashes[0] = 0xb0;
    b.count = 6;

    Probe::SetSkip({0xa1}, Probe::SkipMode::Skip);
    REQUIRE(Probe::Active());
    REQUIRE(Probe::OnDraw(a));
    REQUIRE_FALSE(Probe::OnDraw(b));
    Probe::SetSkip({0xa1}, Probe::SkipMode::Only);
    REQUIRE_FALSE(Probe::OnDraw(a));
    REQUIRE(Probe::OnDraw(b));
    Probe::SetSkip({}, Probe::SkipMode::Skip);
    REQUIRE_FALSE(Probe::Active());

    REQUIRE_FALSE(Probe::ArmCensus(1));
    REQUIRE(Probe::Active());
    (void)Probe::OnDraw(b); // the partial frame before the first present is dropped
    Probe::OnPresent();
    REQUIRE(Probe::LastCensus().present == 0);
    (void)Probe::OnDraw(a);
    (void)Probe::OnDraw(b);
    (void)Probe::OnDraw(b);
    Probe::OnPresent();
    const Probe::Census census = Probe::LastCensus();
    REQUIRE(census.present != 0);
    REQUIRE(census.draws == 3);
    REQUIRE(census.rows.size() == 2);
    REQUIRE(census.rows[0].first.hashes[0] == 0xb0);
    REQUIRE(census.rows[0].draws == 2);
    REQUIRE(census.rows[0].elements == 12);
    REQUIRE_FALSE(Probe::Active());

    const Tests::TempDir temp_dir{"eden-nvn-draw-probe-test"};
    const std::filesystem::path& directory = temp_dir.Path();
    const std::string dir = Common::FS::ToUTF8String(directory.generic_u8string());
    Probe::ArmDump(0xb0, dir);
    REQUIRE_FALSE(Probe::DumpRequested(a).has_value());
    const auto ticket = Probe::DumpRequested(b);
    REQUIRE(ticket.has_value());
    REQUIRE(ticket->path.starts_with(dir + "/nvn-draw-dump-00000000000000b0-p"));
    REQUIRE(ticket->count == 1);
    REQUIRE_FALSE(Probe::DumpRequested(b).has_value());
    REQUIRE_FALSE(Probe::Active());
    REQUIRE_FALSE(Probe::GetDump().done);
    Probe::WriteDump(*ticket, "draw\n");
    REQUIRE(Probe::GetDump().done);
    REQUIRE(Probe::GetDump().path == ticket->path);
}

TEST_CASE("NVN draw probe dumps the next dispatches of a compute program into one file",
          "[core][nx2_debug_server]") {
    namespace Probe = VideoCommon::NvnDrawProbe;
    Probe::Reset();
    const Tests::TempDir temp_dir{"eden-nvn-dispatch-probe-test"};
    const std::filesystem::path& directory = temp_dir.Path();
    SCOPE_EXIT {
        Probe::Reset();
    };
    const std::string dir = Common::FS::ToUTF8String(directory.generic_u8string());
    const Probe::DispatchInfo sky{.hash = 0xfd1176a6c7fe62ce, .groups = {1, 1, 1}};
    const Probe::DispatchInfo other{.hash = 0xc1, .groups = {8, 1, 1}};
    Probe::DrawInfo draw{};
    draw.hashes[0] = 0xfd1176a6c7fe62ce; // a draw never matches a dispatch dump that started

    Probe::ArmDump(sky.hash, dir, 2);
    REQUIRE(Probe::Active());
    REQUIRE(Probe::GetDump().count == 2);
    REQUIRE_FALSE(Probe::DumpRequested(other).has_value());
    const auto first = Probe::DumpRequested(sky);
    REQUIRE(first.has_value());
    REQUIRE(first->path.starts_with(dir + "/nvn-dispatch-dump-fd1176a6c7fe62ce-p"));
    REQUIRE((first->index == 0 && first->count == 2));
    REQUIRE(Probe::GetDump().kind == "dispatch");
    REQUIRE_FALSE(Probe::DumpRequested(draw).has_value());
    REQUIRE(Probe::Active()); // one section left

    // A constant buffer row as the replay formats it: 8.0 at 0x10 of c3, zero rows collapsed.
    std::array<u8, 0x60> c3{};
    const std::array<f32, 4> exposure{8.0f, 1.0f, 0.5f, -2.0f};
    std::memcpy(c3.data() + 0x10, exposure.data(), sizeof(exposure));
    const u32 marker = 0xdeadbeef;
    std::memcpy(c3.data() + 0x50, &marker, sizeof(marker));
    std::string section = "NVN dispatch dump 1 of 2\nbinding 2 compute c3 uniform slot 0:\n";
    Probe::AppendRows(section, c3);
    Probe::WriteDump(*first, section);
    REQUIRE(Probe::GetDump().written == 1);
    REQUIRE_FALSE(Probe::GetDump().done);

    const auto second = Probe::DumpRequested(sky);
    REQUIRE(second.has_value());
    REQUIRE((second->index == 1 && second->path == first->path));
    REQUIRE_FALSE(Probe::Active());
    REQUIRE_FALSE(Probe::DumpRequested(sky).has_value());
    Probe::WriteDump(*second, "NVN dispatch dump 2 of 2\n");
    const Probe::DumpState state = Probe::GetDump();
    REQUIRE((state.done && state.written == 2 && state.error.empty()));
    REQUIRE(state.path == first->path);

    const std::string text =
        Common::FS::ReadStringFromFile(first->path, Common::FS::FileType::BinaryFile);
    INFO(text);
    REQUIRE(text == section + "NVN dispatch dump 2 of 2\n");
    REQUIRE(text.find("  000000: zeros\n") != std::string::npos);
    REQUIRE(text.find("  000010: 00 00 00 41 00 00 80 3f 00 00 00 3f 00 00 00 c0 | f32 8 1 0.5 -2 "
                      "| u32 1090519040 1065353216 1056964608 3221225472\n") != std::string::npos);
    REQUIRE(text.find("  000020..00004f: zeros (3 rows)\n") != std::string::npos);
    REQUIRE(text.find("  000050: ef be ad de") != std::string::npos);

    // Re-arming forgets the old dump; a section claimed before it is not counted.
    Probe::ArmDump(sky.hash, dir, 99);
    REQUIRE(Probe::GetDump().count == Probe::MaxDumpCount);
    Probe::WriteDump(*second, "late\n");
    REQUIRE(Probe::GetDump().written == 0);
}

TEST_CASE("NVN draw probe skips dispatches by compute hash and takes their census",
          "[core][nx2_debug_server]") {
    namespace Probe = VideoCommon::NvnDrawProbe;
    Probe::Reset();
    SCOPE_EXIT {
        Probe::Reset();
    };
    const Probe::DispatchInfo direct{.hash = 0xc0, .groups = {4, 2, 1}, .indirect = false};
    const Probe::DispatchInfo indirect{.hash = 0xc0, .groups = {}, .indirect = true};
    const Probe::DispatchInfo other{.hash = 0xc1, .groups = {8, 1, 1}, .indirect = false};
    Probe::DrawInfo draw{};
    draw.hashes[0] = 0xa0;

    Probe::SetSkip({0xc0}, Probe::SkipMode::Skip);
    REQUIRE(Probe::Active());
    REQUIRE(Probe::OnDispatch(direct));
    REQUIRE(Probe::OnDispatch(indirect));
    REQUIRE_FALSE(Probe::OnDispatch(other));
    REQUIRE_FALSE(Probe::OnDraw(draw));
    // Only mode selects draws; listed compute hashes still skip their dispatches.
    Probe::SetSkip({0xa0, 0xc0}, Probe::SkipMode::Only);
    REQUIRE_FALSE(Probe::OnDraw(draw));
    REQUIRE(Probe::OnDispatch(direct));
    REQUIRE_FALSE(Probe::OnDispatch(other));
    Probe::SetSkip({}, Probe::SkipMode::Only);
    REQUIRE(Probe::OnDraw(draw));
    REQUIRE_FALSE(Probe::OnDispatch(direct));

    Probe::SetSkip({0xc0}, Probe::SkipMode::Skip);
    REQUIRE_FALSE(Probe::ArmCensus(1));
    (void)Probe::OnDispatch(other); // the partial frame before the first present is dropped
    Probe::OnPresent();
    REQUIRE(Probe::LastCensus().dispatch_rows.empty());
    (void)Probe::OnDispatch(direct);
    (void)Probe::OnDispatch(direct);
    (void)Probe::OnDispatch(indirect);
    (void)Probe::OnDispatch(other);
    (void)Probe::OnDraw(draw);
    Probe::OnPresent();
    const Probe::Census census = Probe::LastCensus();
    REQUIRE(census.present != 0);
    REQUIRE(census.draws == 1);
    REQUIRE(census.dispatches == 4);
    REQUIRE(census.skipped_dispatches == 3);
    REQUIRE(census.dispatch_rows.size() == 2);
    const Probe::DispatchRow& first = census.dispatch_rows[0];
    REQUIRE(first.hash == 0xc0);
    REQUIRE(first.count == 3);
    REQUIRE(first.groups == 16);
    REQUIRE(first.indirect == 1);
    REQUIRE(first.skipped == 3);
    const Probe::DispatchRow& second = census.dispatch_rows[1];
    REQUIRE(second.hash == 0xc1);
    REQUIRE(second.count == 1);
    REQUIRE(second.groups == 8);
    REQUIRE(second.indirect == 0);
    REQUIRE(second.skipped == 0);

    // The census is over; the skip set alone keeps the probe active.
    REQUIRE(Probe::Active());
    Probe::SetSkip({}, Probe::SkipMode::Skip);
    REQUIRE_FALSE(Probe::Active());
}
