// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The cost-split switch (nvn_costsplit.h): parsing, the slot schedule, what each mode does to a
// draw or dispatch, the placement tally and the log lines. Pure logic, no device.

#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_costsplit.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"

namespace {

using namespace Vulkan::Nvn::CostSplit;

Config MustParse(std::string_view text) {
    std::string error;
    const std::optional<Config> config = Parse(text, error);
    INFO(text << ": " << error);
    REQUIRE(config.has_value());
    return *config;
}

std::string ParseError(std::string_view text) {
    std::string error;
    const std::optional<Config> config = Parse(text, error);
    REQUIRE_FALSE(config.has_value());
    return error;
}

struct Lines {
    std::vector<std::string> lines;
    static void Sink(void* user, const std::string& line) {
        static_cast<Lines*>(user)->lines.push_back(line);
    }
    [[nodiscard]] size_t Count(std::string_view needle) const {
        size_t n = 0;
        for (const std::string& line : lines) {
            n += line.find(needle) != std::string::npos ? 1 : 0;
        }
        return n;
    }
};

} // namespace

TEST_CASE("NVN costsplit: switch parsing", "[nvn][costsplit]") {
    SECTION("class and start; default modes and period") {
        const Config depth = MustParse("depth:1800");
        CHECK(depth.cls == Class::Depth);
        CHECK(depth.start == 1800);
        CHECK(depth.period == DEFAULT_PERIOD);
        CHECK(depth.modes == std::vector<Mode>{Mode::NoInst, Mode::NoRaster, Mode::Vram, Mode::Tri3,
                                               Mode::NoBind});
        CHECK(MustParse("compute:0").modes ==
              std::vector<Mode>{Mode::NoGroups, Mode::OneGroup, Mode::Vram});
        CHECK(MustParse("all:5").modes.size() == 7);
        CHECK(MustParse("color:5").cls == Class::Color);
    }
    SECTION("modes and period; base is implicit and duplicates collapse") {
        const Config config = MustParse("depth:100:vram,base,tri3,vram:900");
        CHECK(config.modes == std::vector<Mode>{Mode::Vram, Mode::Tri3});
        CHECK(config.period == 900);
        CHECK(MustParse("depth:100::300").modes.size() == 5); // empty mode list: defaults
        CHECK(MustParse("depth:100::300").period == 300);
    }
    SECTION("program class takes a hex hash") {
        const Config config = MustParse("prog:0x00ab12cd34ef5678:10:noinst,nogroups");
        CHECK(config.cls == Class::Program);
        CHECK(config.program == 0x00ab12cd34ef5678ULL);
        CHECK(config.start == 10);
        CHECK(config.modes == std::vector<Mode>{Mode::NoInst, Mode::NoGroups});
        CHECK(ClassLabel(config) == "prog:00ab12cd34ef5678");
        CHECK(MustParse("prog:ff:0").program == 0xff);
    }
    SECTION("malformed input is rejected with a reason") {
        CHECK(ParseError("").find("unknown class") != std::string::npos);
        CHECK(ParseError("shadow:100").find("unknown class") != std::string::npos);
        CHECK(ParseError("depth").find("start") != std::string::npos);
        CHECK(ParseError("depth:abc").find("start") != std::string::npos);
        CHECK(ParseError("depth:100:warp").find("unknown mode") != std::string::npos);
        CHECK(ParseError("depth:100:nogroups").find("does not apply") != std::string::npos);
        CHECK(ParseError("compute:100:noraster").find("does not apply") != std::string::npos);
        CHECK(ParseError("depth:100:vram:60").find("period") != std::string::npos);
        CHECK(ParseError("depth:100:vram:600:1").find("too many") != std::string::npos);
        CHECK(ParseError("prog:zz:100").find("hash") != std::string::npos);
        CHECK(ParseError("prog:0:100").find("hash") != std::string::npos);
    }
}

TEST_CASE("NVN costsplit: schedule interleaves base slots and skips the warm-up",
          "[nvn][costsplit]") {
    const Config config = MustParse("depth:1000:vram,tri3:600");
    CHECK_FALSE(SlotAt(config, 999).has_value());
    const auto slot = [&](u64 presents) {
        const std::optional<Slot> s = SlotAt(config, presents);
        REQUIRE(s.has_value());
        return *s;
    };
    // base, vram, base, tri3, base, vram, ...
    const std::vector<Mode> expected{Mode::Base, Mode::Vram, Mode::Base, Mode::Tri3,
                                     Mode::Base, Mode::Vram};
    for (u64 i = 0; i < expected.size(); ++i) {
        const Slot first = slot(1000 + i * 600);
        CHECK(first.index == i);
        CHECK(first.mode == expected[i]);
        CHECK(first.first_present == 1000 + i * 600);
        CHECK_FALSE(first.measured);
        CHECK_FALSE(slot(1000 + i * 600 + WARMUP - 1).measured);
        CHECK(slot(1000 + i * 600 + WARMUP).measured);
        CHECK(slot(1000 + i * 600 + 599).index == i);
    }
    // Short periods keep at least half of each slot.
    const Config short_period = MustParse("depth:0:vram:120");
    CHECK_FALSE(SlotAt(short_period, 59).value().measured);
    CHECK(SlotAt(short_period, 60).value().measured);
}

TEST_CASE("NVN costsplit: class predicates", "[nvn][costsplit]") {
    const std::vector<u64> hashes{0x11, 0x22, 0, 0, 0, 0, 0};
    const Config depth = MustParse("depth:0");
    CHECK(DrawInClass(depth, 0, true, hashes));
    CHECK_FALSE(DrawInClass(depth, 1, true, hashes));
    CHECK_FALSE(DrawInClass(depth, 0, false, hashes));
    CHECK_FALSE(DispatchInClass(depth, 0x11));
    const Config color = MustParse("color:0");
    CHECK(DrawInClass(color, 2, true, hashes));
    CHECK_FALSE(DrawInClass(color, 0, true, hashes));
    const Config compute = MustParse("compute:0");
    CHECK_FALSE(DrawInClass(compute, 1, true, hashes));
    CHECK(DispatchInClass(compute, 0x33));
    const Config prog = MustParse("prog:22:0");
    CHECK(NeedsPrograms(prog));
    CHECK(DrawInClass(prog, 1, false, hashes));
    CHECK_FALSE(DrawInClass(prog, 1, false, std::vector<u64>{0x11, 0x23}));
    CHECK(DispatchInClass(prog, 0x22));
    CHECK_FALSE(DispatchInClass(prog, 0x11));
    const Config all = MustParse("all:0");
    CHECK(DrawInClass(all, 0, false, hashes));
    CHECK(DispatchInClass(all, 0));
    // Modes of the other kind leave the work as in base.
    CHECK(AppliesToDraws(Mode::Vram));
    CHECK(AppliesToDispatches(Mode::Vram));
    CHECK_FALSE(AppliesToDraws(Mode::NoGroups));
    CHECK_FALSE(AppliesToDispatches(Mode::NoInst));
}

TEST_CASE("NVN costsplit: count modes clamp draws and dispatches", "[nvn][costsplit]") {
    CHECK(Instances(Mode::NoInst, 7) == 0);
    CHECK(Instances(Mode::Base, 7) == 7);
    CHECK(Instances(Mode::Tri3, 7) == 7);
    CHECK(VertexCount(Mode::Tri3, 3000) == 3);
    CHECK(VertexCount(Mode::Tri3, 2) == 2);
    CHECK(VertexCount(Mode::NoInst, 3000) == 3000);
    CHECK(Groups(Mode::NoGroups, {8, 4, 2}) == std::array<u32, 3>{0, 0, 0});
    CHECK(Groups(Mode::OneGroup, {8, 4, 2}) == std::array<u32, 3>{1, 1, 1});
    CHECK(Groups(Mode::OneGroup, {0, 0, 0}) == std::array<u32, 3>{1, 1, 1}); // indirect
    CHECK(Groups(Mode::Vram, {8, 4, 2}) == std::array<u32, 3>{8, 4, 2});
}

TEST_CASE("NVN costsplit: the per-kind vram modes copy one kind of data", "[nvn][costsplit]") {
    using Vulkan::Nvn::CostSplit::CopiesToRing;
    using Vulkan::Nvn::CostSplit::Data;
    for (const Data data : {Data::Uniform, Data::Vertex, Data::Index, Data::Storage}) {
        CHECK(CopiesToRing(Mode::Vram, data));
        CHECK_FALSE(CopiesToRing(Mode::Base, data));
        CHECK_FALSE(CopiesToRing(Mode::NoInst, data));
    }
    CHECK(CopiesToRing(Mode::VramUbo, Data::Uniform));
    CHECK_FALSE(CopiesToRing(Mode::VramUbo, Data::Vertex));
    CHECK_FALSE(CopiesToRing(Mode::VramUbo, Data::Storage));
    CHECK(CopiesToRing(Mode::VramVtx, Data::Vertex));
    CHECK(CopiesToRing(Mode::VramVtx, Data::Index));
    CHECK_FALSE(CopiesToRing(Mode::VramVtx, Data::Uniform));
    CHECK(CopiesToRing(Mode::VramSsbo, Data::Storage));
    CHECK_FALSE(CopiesToRing(Mode::VramSsbo, Data::Index));
    // Draw modes by name only: not in the default lists, not on dispatches.
    CHECK(AppliesToDraws(Mode::VramUbo));
    CHECK(AppliesToDraws(Mode::VramVtx));
    CHECK(AppliesToDraws(Mode::VramSsbo));
    CHECK_FALSE(AppliesToDispatches(Mode::VramUbo));
    const Config split = MustParse("color:100:vramu,vramv,vrams");
    CHECK(split.modes == std::vector<Mode>{Mode::VramUbo, Mode::VramVtx, Mode::VramSsbo});
    CHECK(std::string{ModeName(Mode::VramUbo)} == "vramu");
    CHECK(std::string{ModeName(Mode::VramVtx)} == "vramv");
    CHECK(std::string{ModeName(Mode::VramSsbo)} == "vrams");
    const Config defaults = MustParse("color:100");
    CHECK(std::find(defaults.modes.begin(), defaults.modes.end(), Mode::VramUbo) ==
          defaults.modes.end());
}

TEST_CASE("NVN costsplit: tally buckets per backing", "[nvn][costsplit]") {
    using Vulkan::Nvn::Pool;
    CHECK(PlacementOfBacking(static_cast<u32>(Pool::Backing::None)) == Placement::None);
    CHECK(PlacementOfBacking(static_cast<u32>(Pool::Backing::Imported)) == Placement::Imported);
    CHECK(PlacementOfBacking(static_cast<u32>(Pool::Backing::Mirror)) == Placement::Mirror);
    CHECK(PlacementOfBacking(static_cast<u32>(Pool::Backing::DeviceLocal)) == Placement::Device);
    CHECK(PlacementOfBacking(static_cast<u32>(Pool::Backing::Virtual)) == Placement::Virtual);
    CHECK(PlacementOfBacking(99) == Placement::None);

    Stats stats;
    stats.frames = 2;
    stats.draws = 4;
    stats.tally.Note(Data::Uniform, Placement::Imported, 256);
    stats.tally.Note(Data::Uniform, Placement::Arena, 768);
    stats.tally.Note(Data::Vertex, Placement::Imported, 4096);
    stats.tally.Note(Data::Vertex, Placement::Imported, 2048);
    stats.tally.Note(Data::Index, Placement::Mirror, 600);
    stats.tally.Note(Data::Storage, Placement::Device, 64);
    stats.tally.Note(Data::Storage, Placement::Virtual, 64);
    stats.tally.Note(Data::Storage, Placement::None, 64);
    stats.tally.clamped = 3;
    CHECK(stats.tally.Bindings(Data::Vertex) == 2);
    const Config config = MustParse("depth:0:noinst");
    const std::string line = FormatTally(config, Mode::NoInst, stats, std::nullopt, "device");
    INFO(line);
    CHECK(line.starts_with("[nvn] costsplit tally class=depth mode=noinst scope=run frames=2 "));
    CHECK(line.find(" ubo=0/1/0/0/0/1 ") != std::string::npos);
    CHECK(line.find(" vtx=0/2/0/0/0/0 ") != std::string::npos);
    CHECK(line.find(" idx=0/0/1/0/0/0 ") != std::string::npos);
    CHECK(line.find(" ssbo=1/0/0/1/1/0 ") != std::string::npos);
    CHECK(line.find(" bytes=512/3072/600/64 ") != std::string::npos);
    CHECK(line.find(" ring=device ") != std::string::npos);
    CHECK(line.ends_with(" unclamped=1")); // 4 draws, 3 changed

    Tally merged;
    merged.Merge(stats.tally);
    merged.Merge(stats.tally);
    const auto at = [](Data data, Placement placement) {
        return std::pair{static_cast<size_t>(data), static_cast<size_t>(placement)};
    };
    const auto [vertex, imported] = at(Data::Vertex, Placement::Imported);
    CHECK(merged.count[vertex][imported] == 4);
    CHECK(merged.bytes[static_cast<size_t>(Data::Vertex)] == 12288);
    CHECK(merged.clamped == 6);
}

TEST_CASE("NVN costsplit: recorder sums slots and logs windows and run totals",
          "[nvn][costsplit]") {
    Recorder recorder{MustParse("depth:10:vram:120")};
    Lines lines;
    recorder.SetSink(&Lines::Sink, &lines);
    CHECK_FALSE(recorder.Current().has_value());

    // Presents 0..9: before the start. Then two slots (base, vram) and a third (base) started.
    for (u64 presents = 1; presents <= 10 + 2 * 120 + 70; ++presents) {
        const std::optional<Slot> slot = recorder.Current();
        if (slot && slot->measured) {
            // Each measured frame: 100 class draws, 2 ms of replay time, one imported vertex range.
            recorder.Pending().draws += 100;
            recorder.Pending().cpu_ns += 2e6;
            recorder.Pending().tally.Note(Data::Vertex, Placement::Imported, 4096);
            // The GPU side of the frame (completion thread, in order), base 8 ms, vram 2 ms.
            const double class_ns = slot->mode == Mode::Vram ? 2e6 : 8e6;
            recorder.AddGpu(slot->index, true, 20e6, class_ns, 3);
            recorder.ReportFinished(slot->index);
        }
        recorder.OnPresent(presents);
    }
    CHECK(lines.Count("[nvn] costsplit schedule class=depth slot=") == 3);
    // Slots 0 and 1 are finished: one line and one tally each, each followed by the totals.
    REQUIRE(lines.Count("scope=window") == 4);
    CHECK(lines.Count("[nvn] costsplit class=depth mode=base scope=window slot=0 start=10 "
                      "frames=60 busy_ms=20.000 class_ms=8.000 passes=3.0 draws=100.0 "
                      "dispatches=0.0 us_per_draw=80.000 cpu_ms=2.000") == 1);
    CHECK(lines.Count("[nvn] costsplit class=depth mode=vram scope=window slot=1 start=130 "
                      "frames=60 busy_ms=20.000 class_ms=2.000") == 1);
    CHECK(lines.Count("[nvn] costsplit tally class=depth mode=vram scope=window slot=1") == 1);
    CHECK(lines.Count("scope=window slot=1 start=130 frames=60 draws=6000 dispatches=0 "
                      "placement=none/imported/mirror/device/virtual/arena ubo=0/0/0/0/0/0 "
                      "vtx=0/60/0/0/0/0 ") == 1);
    // Totals after slot 0 (base) and after slot 1 (base, vram).
    CHECK(lines.Count("[nvn] costsplit class=depth mode=base scope=run frames=60 ") == 2);
    CHECK(lines.Count("[nvn] costsplit class=depth mode=vram scope=run frames=60 ") == 1);

    recorder.ReportRun();
    // The open slot 2 (base, 10 measured frames), then base and vram over the run.
    CHECK(lines.Count("[nvn] costsplit class=depth mode=base scope=window slot=2 start=250 "
                      "frames=10 ") == 1);
    CHECK(lines.Count("[nvn] costsplit tally class=depth mode=base scope=window slot=2 start=250 "
                      "frames=10 ") == 1);
    // The last totals: base with slot 2, vram unchanged.
    CHECK(lines.Count("[nvn] costsplit class=depth mode=base scope=run frames=70 ") == 1);
    CHECK(lines.Count("[nvn] costsplit class=depth mode=vram scope=run frames=60 ") == 2);
    CHECK(lines.Count("[nvn] costsplit tally class=depth mode=base scope=run frames=70 ") == 1);
    CHECK(lines.lines.back().starts_with("[nvn] costsplit tally class=depth mode=vram scope=run"));
}
