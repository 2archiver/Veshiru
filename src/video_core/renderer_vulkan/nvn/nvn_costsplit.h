// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Cost split of the NVN replay, `EDEN_NX2_NVN_COSTSPLIT=<class>:<start>[:<modes>][:<period>]`
// (MEASUREMENT ONLY: the picture is wrong on purpose while it runs). See DESIGN.md "Cost split".
//
// From present <start> on, the draws (or dispatches) of one class run in a schedule of slots of
// <period> presents: base, mode 1, base, mode 2, ... Each mode removes one part of the work
// (instances, rasterization, host-memory placement, vertices, rebinding, workgroups), so the GPU
// time of the class in a slot, set against the base slots around it, splits the class's cost into
// its causes. The first WARMUP presents of every slot are not summed. Numbers come from the GPU
// profiler (forced on) and the replay thread, and are logged as `[nvn] costsplit ...` lines.
//
// Off (the default) `Context::Impl::costsplit` is null and the draw path tests one cached pointer.

#pragma once

#include <array>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace Vulkan::Nvn::CostSplit {

enum class Class : u8 {
    Depth,   ///< draws into a depth target without colour targets (shadow maps, depth pre-pass)
    Color,   ///< draws into at least one colour target
    Compute, ///< dispatches
    Program, ///< draws and dispatches whose programs include `Config::program`
    All,     ///< every draw and dispatch
};

enum class Mode : u8 {
    Base,     ///< unchanged
    NoInst,   ///< draws: zero instances (what is left is recording + driver overhead)
    NoRaster, ///< draws: empty scissor (vertex work and setup, no fragments)
    Vram,     ///< uniform / vertex / index / read-only storage ranges in host memory are copied
              ///< into a device-local ring (memory placement)
    Tri3,     ///< draws: at most three vertices / indices (the fixed cost per draw)
    NoBind,   ///< draws: the pass's first descriptors and buffers are reused (binding cost)
    NoGroups, ///< dispatches: zero workgroups
    OneGroup, ///< dispatches: one workgroup
    VramUbo,  ///< like Vram, uniform buffer ranges only
    VramVtx,  ///< like Vram, vertex and index buffer ranges only
    VramSsbo, ///< like Vram, read-only storage buffer ranges only
    Count,    ///< no mode: the work is outside the class or the schedule has not started
};
inline constexpr size_t MODE_COUNT = static_cast<size_t>(Mode::Count);

inline constexpr u64 DEFAULT_PERIOD = 600;
inline constexpr u64 MIN_PERIOD = 120;
inline constexpr u64 WARMUP = 60;

[[nodiscard]] const char* ClassName(Class cls) noexcept;
[[nodiscard]] const char* ModeName(Mode mode) noexcept;
/// Whether `mode` changes draws / dispatches (Base and Vram change both kinds).
[[nodiscard]] bool AppliesToDraws(Mode mode) noexcept;
[[nodiscard]] bool AppliesToDispatches(Mode mode) noexcept;

struct Config {
    Class cls{Class::Depth};
    u64 program{};     ///< Class::Program: the program hash (as the GPU profiler prints it)
    u64 start{};       ///< presents recorded before the schedule starts
    u64 period{DEFAULT_PERIOD};
    std::vector<Mode> modes; ///< measured modes in order; base slots go between them
};

/// Parses the switch: `<class>:<start>[:<modes>][:<period>]`, class `depth`, `color`, `compute`,
/// `prog:<hex hash>` or `all`, modes a comma list (default: every mode of the class). nullopt
/// with `error` set when malformed.
[[nodiscard]] std::optional<Config> Parse(std::string_view text, std::string& error);

/// The class, without the start/period, as the log lines name it (`prog:<hash>` for programs).
[[nodiscard]] std::string ClassLabel(const Config& config);

/// One slot of the schedule.
struct Slot {
    u64 index{};         ///< 0 from the start present on
    Mode mode{Mode::Base};
    bool measured{};     ///< past the slot's warm-up
    u64 first_present{}; ///< presents recorded when the slot began
};

/// The slot for `presents` presents recorded so far; nullopt before the start.
[[nodiscard]] std::optional<Slot> SlotAt(const Config& config, u64 presents) noexcept;

/// Class predicates. `stage_hashes`: the program hash of each bound stage (0 for none); only read
/// for Class::Program.
[[nodiscard]] bool DrawInClass(const Config& config, u32 color_targets, bool depth_target,
                               std::span<const u64> stage_hashes) noexcept;
[[nodiscard]] bool DispatchInClass(const Config& config, u64 program) noexcept;
/// Whether the class predicate of a draw reads the program hashes.
[[nodiscard]] inline bool NeedsPrograms(const Config& config) noexcept {
    return config.cls == Class::Program;
}

/// What a mode does to a direct draw / dispatch.
[[nodiscard]] constexpr u32 Instances(Mode mode, u32 instances) noexcept {
    return mode == Mode::NoInst ? 0u : instances;
}
[[nodiscard]] constexpr u32 VertexCount(Mode mode, u32 count) noexcept {
    return mode == Mode::Tri3 && count > 3 ? 3u : count;
}
[[nodiscard]] constexpr std::array<u32, 3> Groups(Mode mode, std::array<u32, 3> groups) noexcept {
    return mode == Mode::NoGroups   ? std::array<u32, 3>{0, 0, 0}
           : mode == Mode::OneGroup ? std::array<u32, 3>{1, 1, 1}
                                    : groups;
}

/// Where the bytes of a bound range live.
enum class Placement : u8 {
    None,     ///< not resolved
    Imported, ///< guest memory imported as a host buffer (read over PCIe)
    Mirror,   ///< device-local mirror
    Device,   ///< device-local pool
    Virtual,  ///< virtual pool
    Arena,    ///< private copy (inline uniform updates, misaligned uniform)
    Count,
};
inline constexpr size_t PLACEMENT_COUNT = static_cast<size_t>(Placement::Count);
/// `Pool::Backing` (None, Imported, Mirror, DeviceLocal, Virtual: nvn_impl.h) as a placement.
[[nodiscard]] Placement PlacementOfBacking(u32 backing) noexcept;

enum class Data : u8 { Uniform, Vertex, Index, Storage, Count };
inline constexpr size_t DATA_COUNT = static_cast<size_t>(Data::Count);

/// Whether `mode` copies the ranges of `data` into the ring (Vram: all kinds; VramUbo / VramVtx /
/// VramSsbo: one kind, vertex with index).
[[nodiscard]] constexpr bool CopiesToRing(Mode mode, Data data) noexcept {
    switch (mode) {
    case Mode::Vram:
        return true;
    case Mode::VramUbo:
        return data == Data::Uniform;
    case Mode::VramVtx:
        return data == Data::Vertex || data == Data::Index;
    case Mode::VramSsbo:
        return data == Data::Storage;
    default:
        return false;
    }
}

/// Where the class's bound ranges live, counted per binding.
struct Tally {
    std::array<std::array<u64, PLACEMENT_COUNT>, DATA_COUNT> count{};
    std::array<u64, DATA_COUNT> bytes{};
    u64 copied{};     ///< ranges copied into the ring (Mode::Vram)
    u64 ring_bytes{}; ///< ... their bytes
    u64 clamped{};    ///< direct draws / dispatches a count mode changed

    void Note(Data data, Placement placement, u64 size) noexcept {
        ++count[static_cast<size_t>(data)][static_cast<size_t>(placement)];
        bytes[static_cast<size_t>(data)] += size;
    }
    void Merge(const Tally& other) noexcept;
    [[nodiscard]] u64 Bindings(Data data) const noexcept;
};

/// The numbers of one scope: a slot past its warm-up, or a mode over the whole run.
struct Stats {
    u64 frames{};      ///< presents completed (GPU)
    double busy_ns{};  ///< command buffer time (GPU)
    double class_ns{}; ///< passes / dispatches holding class work (GPU)
    u64 passes{};      ///< ... their number
    u64 draws{};       ///< class draws recorded (replay thread)
    u64 dispatches{};  ///< class dispatches recorded
    double cpu_ns{};   ///< replay thread time preparing them
    Tally tally;

    void Merge(const Stats& other) noexcept;
};

/// `[nvn] costsplit class=.. mode=.. scope=.. ...` and `[nvn] costsplit tally ...`.
/// `slot`: the slot (scope=window) or nullopt (scope=run).
[[nodiscard]] std::string FormatLine(const Config& config, Mode mode, const Stats& stats,
                                     const std::optional<Slot>& slot);
[[nodiscard]] std::string FormatTally(const Config& config, Mode mode, const Stats& stats,
                                      const std::optional<Slot>& slot, std::string_view ring);

/// The schedule and the sums. The replay thread owns the current slot and the pending numbers
/// of the frame being recorded; the GPU profiler's completion thread adds the GPU side of each
/// completed submission. The slot table between them is locked.
class Recorder {
public:
    explicit Recorder(Config config);

    [[nodiscard]] const Config& GetConfig() const noexcept {
        return config;
    }

    // ---- replay thread ----
    /// The slot work is recorded in now (nullopt before the start).
    [[nodiscard]] const std::optional<Slot>& Current() const noexcept {
        return current;
    }
    /// Numbers of the frame being recorded (counted only in a measured slot).
    [[nodiscard]] Stats& Pending() noexcept {
        return pending;
    }
    /// A present was recorded (`presents` recorded so far): the frame's numbers join its slot;
    /// a new slot is logged when one begins.
    void OnPresent(u64 presents);
    /// How the ring of Mode::Vram copies is backed ("device", "host"; "none" until made).
    void SetRing(std::string_view kind);

    // ---- completion thread ----
    /// The GPU side of one completed submission recorded in `slot` (Slot::index).
    void AddGpu(u64 slot, bool present, double busy_ns, double class_ns, u64 passes);
    /// Logs the slots the GPU has finished: every slot before `slot` (a submission past the
    /// warm-up of `slot` completed; the slots before have completed by then), then the totals
    /// per mode so far (scope=run; the last ones count, a killed run keeps them).
    void ReportFinished(u64 slot);
    /// End of the run (replay and completion threads stopped): the open slots and the totals.
    void ReportRun();

    /// Log lines go here (tests); default LOG_INFO.
    void SetSink(void (*sink)(void* user, const std::string& line), void* user) {
        sink_fn = sink;
        sink_user = user;
    }

private:
    struct SlotStats {
        Slot slot;
        Stats stats;
    };
    void Emit(const std::string& line);
    void ReportSlot(const SlotStats& entry);
    void ReportTotals();

    const Config config;
    // Replay thread.
    std::optional<Slot> current;
    Stats pending;
    // Shared.
    std::mutex mutex;
    std::map<u64, SlotStats> slots; ///< not yet logged, by Slot::index
    std::array<Stats, MODE_COUNT> run{};
    std::string ring{"none"};
    void (*sink_fn)(void*, const std::string&){};
    void* sink_user{};
};

} // namespace Vulkan::Nvn::CostSplit
