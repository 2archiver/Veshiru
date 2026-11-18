// SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "common/scope_exit.h"
#include "veshiru/util/artwork_http.h"
#include "veshiru/util/steamgriddb_provider.h"

using namespace Veshiru::ArtworkHttp;
using namespace std::chrono_literals;

TEST_CASE("Artwork response schemas and asset hosts are constrained", "[artwork_http]") {
    CHECK(ParseData(R"({"success":true,"data":[{"id":1,"name":"Test title"}]})").size() == 1);
    CHECK(ParseData("{").is_discarded());
    CHECK(ParseData(R"({"success":true,"data":{"id":1}})").is_discarded());
    CHECK(ParseData(R"({"success":false,"data":[]})").is_discarded());
    CHECK(ParseData(std::string(50, '[') + std::string(50, ']')).is_discarded());
    CHECK(AssetPath("https://cdn2.steamgriddb.com/icon/image.png") == "/icon/image.png");
    for (const auto* url : {"https://www.steamgriddb.com/image.png",
                            "https://cdn2.steamgriddb.com@other.invalid/image.png",
                            "https://cdn2.steamgriddb.com/icon/../image.png",
                            "https://cdn2.steamgriddb.com/icon/%2e%2e/image.png",
                            "http://cdn2.steamgriddb.com/image.png",
                            "https://cdn2.steamgriddb.com/image.png?token=x"}) {
        CHECK_FALSE(AssetPath(url));
    }
}

TEST_CASE("Artwork requests are single-flight and obey cooldown", "[artwork_http]") {
    SearchGate gate;
    const auto started = Clock::time_point{};
    CHECK(gate.TryBegin(started));
    CHECK_FALSE(gate.TryBegin(started));
    gate.End(started, false);
    CHECK_FALSE(gate.TryBegin(started + 4s));
    CHECK(gate.TryBegin(started + 5s));
    gate.End(started + 5s, true);
    CHECK_FALSE(gate.TryBegin(started + 64s));
    CHECK(gate.TryBegin(started + 65s));
}

TEST_CASE("Artwork GET limits streams and does not follow redirects", "[artwork_http]") {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic_int redirect_hits{};
    std::atomic_int serving_slow{};
    std::atomic_bool leave_slow{};
    server.Get("/json", [&](const httplib::Request&, httplib::Response& response) {
        ++redirect_hits;
        response.set_content(R"({"success":true,"data":[]})", "application/json");
    });
    server.Get("/redirect", [](const httplib::Request&, httplib::Response& response) {
        response.set_redirect("/json");
    });
    server.Get("/rate-limit",
               [](const httplib::Request&, httplib::Response& response) { response.status = 429; });
    server.Get("/wrong-type", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("Not an image", "text/html");
    });
    server.Get("/too-large", [](const httplib::Request&, httplib::Response& response) {
        response.set_chunked_content_provider("application/json",
                                              [](size_t offset, httplib::DataSink& sink) {
                                                  const std::string chunk(4096, 'x');
                                                  if (offset >= 256 * 1024) {
                                                      sink.done();
                                                      return true;
                                                  }
                                                  return sink.write(chunk.data(), chunk.size());
                                              });
    });
    server.Get("/slow", [&](const httplib::Request&, httplib::Response& response) {
        ++serving_slow;
        const auto deadline = Clock::now() + 3s;
        while (!leave_slow && Clock::now() < deadline) {
            std::this_thread::sleep_for(5ms);
        }
        response.set_content(R"({"success":true,"data":[]})", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::jthread listener{[&] { server.listen_after_bind(); }};
    server.wait_until_ready();
    SCOPE_EXIT {
        leave_slow = true;
        server.stop();
    };
    std::atomic_bool cancelled{};
    const auto request = [&](const std::string& path, Kind kind, Clock::duration timeout = 2s) {
        httplib::Client client{"127.0.0.1", port};
        return Get(client, path, {}, kind, Clock::now() + timeout, cancelled);
    };
    CHECK(request("/redirect", Kind::Json).status == Status::Failed);
    CHECK(redirect_hits == 0);
    CHECK(request("/json", Kind::Json).status == Status::Ok);
    CHECK(request("/rate-limit", Kind::Json).status == Status::RateLimited);
    CHECK(request("/too-large", Kind::Json).status == Status::Invalid);
    CHECK(request("/wrong-type", Kind::Icon).status == Status::Invalid);
    CHECK(request("/slow", Kind::Json, 120ms).status == Status::Timeout);
    std::jthread cancel_thread{[&](std::stop_token stop) {
        const auto deadline = Clock::now() + 1s;
        while (serving_slow < 2 && !stop.stop_requested() && Clock::now() < deadline) {
            std::this_thread::sleep_for(5ms);
        }
        cancelled = true;
    }};
    CHECK(request("/slow", Kind::Json).status == Status::Cancelled);
}

TEST_CASE("SteamGridDB lookup matches the requested edition and keeps credentials off the CDN",
          "[artwork_http]") {
    httplib::Server api;
    httplib::Server cdn;
    std::atomic_bool received_query{};
    std::atomic_bool requested_match{};
    std::atomic_int cdn_requests{};
    std::atomic_bool cdn_has_credentials{};
    api.Get("/api/v2/search/autocomplete/.*", [&](const auto& request, auto& response) {
        received_query =
            request.path == "/api/v2/search/autocomplete/The Legend of Zelda: Ocarina of Time" &&
            request.get_header_value("Authorization") == "Bearer mock-key";
        response.set_content(R"({"success":true,"data":[
            {"id":7,"name":"The Legend of Zelda: Ocarina of Time Demo"},
            {"id":8,"name":"The Legend of Zelda: Ocarina of Time\u2122"},
            {"id":9,"name":"The Legend of Zelda: Majora's Mask"},
            {"id":"10","name":"The Legend of Zelda: Ocarina of Time"}
        ]})",
                             "application/json");
    });
    api.Get("/api/v2/icons/game/8", [&](const auto& request, auto& response) {
        requested_match = request.get_header_value("Authorization") == "Bearer mock-key" &&
                          request.get_param_value("mimes") == "image/png" &&
                          request.get_param_value("nsfw") == "false";
        response.set_content(R"({"success":true,"data":[
            {"url":"https://example.invalid/icon/rejected.png"},
            {"url":"https://cdn2.steamgriddb.com/icon/ok.png"},
            {"url":"https://cdn2.steamgriddb.com/icon/ok.png"}
        ]})",
                             "application/json");
    });
    cdn.Get("/icon/ok.png", [&](const auto& request, auto& response) {
        ++cdn_requests;
        cdn_has_credentials = request.has_header("Authorization") || request.has_header("Cookie");
        // It's the decoder's job to check the bytes, not the HTTP layer's.
        response.set_content("image bytes", "image/png");
    });
    const int api_port = api.bind_to_any_port("127.0.0.1");
    const int cdn_port = cdn.bind_to_any_port("127.0.0.1");
    REQUIRE(api_port > 0);
    REQUIRE(cdn_port > 0);
    api.new_task_queue = [] { return new httplib::ThreadPool(1); };
    cdn.new_task_queue = [] { return new httplib::ThreadPool(1); };
    std::jthread api_listener{[&] { api.listen_after_bind(); }};
    std::jthread cdn_listener{[&] { cdn.listen_after_bind(); }};
    api.wait_until_ready();
    cdn.wait_until_ready();
    SCOPE_EXIT {
        api.stop();
        cdn.stop();
    };
    httplib::Client api_client{"127.0.0.1", api_port};
    httplib::Client cdn_client{"127.0.0.1", cdn_port};
    const std::atomic_bool cancelled{};
    const auto icons = Veshiru::SteamGridDB::FindIcons(api_client, cdn_client, {},
                                                       "The Legend of Zelda: Ocarina of Time",
                                                       "mock-key", Clock::now() + 2s, cancelled);
    CHECK(received_query);
    CHECK(requested_match);
    CHECK(cdn_requests == 1);
    CHECK_FALSE(cdn_has_credentials);
    CHECK(icons.status == Status::Ok);
    CHECK(icons.pngs == std::vector<std::string>{"image bytes"});
}
