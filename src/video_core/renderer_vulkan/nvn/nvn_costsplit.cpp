// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <charconv>
#include <cstring>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_costsplit.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn::CostSplit {

static_assert(static_cast<u32>(Pool::Backing::None) == 0 &&
                  static_cast<u32>(Pool::Backing::Imported) == 1 &&
                  static_cast<u32>(Pool::Backing::Mirror) == 2 &&
                  static_cast<u32>(Pool::Backing::DeviceLocal) == 3 &&
                  static_cast<u32>(Pool::Backing::Virtual) == 4,
              "PlacementOfBacking maps Pool::Backing by value");

namespace {

constexpr std::array<Mode, 5> DRAW_MODES{Mode::NoInst, Mode::NoRaster, Mode::Vram, Mode::Tri3,
                                         Mode::NoBind};
constexpr std::array<Mode, 3> COMPUTE_MODES{Mode::NoGroups, Mode::OneGroup, Mode::Vram};
/// Only by name (never in the default lists): the placement split per kind of data.
constexpr std::array<Mode, 3> KIND_MODES{Mode::VramUbo, Mode::VramVtx, Mode::VramSsbo};
constexpr std::array<Mode, 7> EVERY_MODE{Mode::NoInst, Mode::NoRaster, Mode::Vram, Mode::Tri3,
                                         Mode::NoBind, Mode::NoGroups, Mode::OneGroup};

std::vector<std::string_view> Split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    size_t begin = 0;
    while (true) {
        const size_t end = text.find(separator, begin);
        parts.push_back(text.substr(begin, end == std::string_view::npos ? end : end - begin));
        if (end == std::string_view::npos) {
            return parts;
        }
        begin = end + 1;
    }
}

bool ParseNumber(std::string_view text, int base, u64& out) {
    if (text.empty()) {
        return false;
    }
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out, base);
    return ec == std::errc{} && end == text.data() + text.size();
}

std::optional<Mode> ModeOfName(std::string_view name) {
    for (size_t m = 0; m < MODE_COUNT; ++m) {
        if (name == ModeName(static_cast<Mode>(m))) {
            return static_cast<Mode>(m);
        }
    }
    return std::nullopt;
}

double PerFrame(double value, u64 frames) {
    return frames != 0 ? value / static_cast<double>(frames) : 0.0;
}

std::string Buckets(const Tally& tally, Data data) {
    const auto& c = tally.count[static_cast<size_t>(data)];
    return fmt::format("{}/{}/{}/{}/{}/{}", c[0], c[1], c[2], c[3], c[4], c[5]);
}

u64 AverageBytes(const Tally& tally, Data data) {
    const u64 n = tally.Bindings(data);
    return n != 0 ? tally.bytes[static_cast<size_t>(data)] / n : 0;
}

std::string ScopeText(const std::optional<Slot>& slot) {
    return slot ? fmt::format("scope=window slot={} start={}", slot->index, slot->first_present)
                : std::string{"scope=run"};
}

} // namespace

const char* ClassName(Class cls) noexcept {
    switch (cls) {
    case Class::Depth:
        return "depth";
    case Class::Color:
        return "color";
    case Class::Compute:
        return "compute";
    case Class::Program:
        return "prog";
    case Class::All:
        return "all";
    }
    return "?";
}

const char* ModeName(Mode mode) noexcept {
    switch (mode) {
    case Mode::Base:
        return "base";
    case Mode::NoInst:
        return "noinst";
    case Mode::NoRaster:
        return "noraster";
    case Mode::Vram:
        return "vram";
    case Mode::Tri3:
        return "tri3";
    case Mode::NoBind:
        return "nobind";
    case Mode::NoGroups:
        return "nogroups";
    case Mode::OneGroup:
        return "onegroup";
    case Mode::VramUbo:
        return "vramu";
    case Mode::VramVtx:
        return "vramv";
    case Mode::VramSsbo:
        return "vrams";
    case Mode::Count:
        break;
    }
    return "?";
}

bool AppliesToDraws(Mode mode) noexcept {
    return mode == Mode::Base ||
           std::find(DRAW_MODES.begin(), DRAW_MODES.end(), mode) != DRAW_MODES.end() ||
           std::find(KIND_MODES.begin(), KIND_MODES.end(), mode) != KIND_MODES.end();
}

bool AppliesToDispatches(Mode mode) noexcept {
    return mode == Mode::Base ||
           std::find(COMPUTE_MODES.begin(), COMPUTE_MODES.end(), mode) != COMPUTE_MODES.end();
}

std::optional<Config> Parse(std::string_view text, std::string& error) {
    std::vector<std::string_view> parts = Split(text, ':');
    Config config;
    size_t next = 1;
    const std::string_view cls = parts[0];
    if (cls == "depth") {
        config.cls = Class::Depth;
    } else if (cls == "color") {
        config.cls = Class::Color;
    } else if (cls == "compute") {
        config.cls = Class::Compute;
    } else if (cls == "all") {
        config.cls = Class::All;
    } else if (cls == "prog") {
        config.cls = Class::Program;
        std::string_view hash = parts.size() > 1 ? parts[1] : std::string_view{};
        if (hash.starts_with("0x") || hash.starts_with("0X")) {
            hash.remove_prefix(2);
        }
        if (!ParseNumber(hash, 16, config.program) || config.program == 0) {
            error = "prog needs a non-zero hex program hash (prog:<hash>)";
            return std::nullopt;
        }
        next = 2;
    } else {
        error = fmt::format("unknown class '{}' (depth, color, compute, prog:<hash>, all)", cls);
        return std::nullopt;
    }
    if (parts.size() <= next || !ParseNumber(parts[next], 10, config.start)) {
        error = "missing or bad start present (<class>:<start>[:<modes>][:<period>])";
        return std::nullopt;
    }
    ++next;
    const auto allowed = [&](Mode mode) {
        switch (config.cls) {
        case Class::Depth:
        case Class::Color:
            return AppliesToDraws(mode);
        case Class::Compute:
            return AppliesToDispatches(mode);
        case Class::Program:
        case Class::All:
            return true;
        }
        return false;
    };
    if (parts.size() > next && !parts[next].empty()) {
        for (const std::string_view name : Split(parts[next], ',')) {
            const std::optional<Mode> mode = ModeOfName(name);
            if (!mode) {
                error = fmt::format("unknown mode '{}'", name);
                return std::nullopt;
            }
            if (!allowed(*mode)) {
                error = fmt::format("mode '{}' does not apply to class '{}'", name, cls);
                return std::nullopt;
            }
            // Base slots are always there, between the measured modes.
            if (*mode != Mode::Base &&
                std::find(config.modes.begin(), config.modes.end(), *mode) == config.modes.end()) {
                config.modes.push_back(*mode);
            }
        }
    }
    if (parts.size() > next) {
        ++next;
    }
    if (config.modes.empty()) {
        const bool draws = config.cls == Class::Depth || config.cls == Class::Color;
        const std::span<const Mode> defaults = draws ? std::span<const Mode>{DRAW_MODES}
                                               : config.cls == Class::Compute
                                                   ? std::span<const Mode>{COMPUTE_MODES}
                                                   : std::span<const Mode>{EVERY_MODE};
        config.modes.assign(defaults.begin(), defaults.end());
    }
    if (parts.size() > next) {
        if (!ParseNumber(parts[next], 10, config.period) || config.period < MIN_PERIOD) {
            error = fmt::format("period must be a number of presents >= {}", MIN_PERIOD);
            return std::nullopt;
        }
        ++next;
    }
    if (parts.size() > next) {
        error = "too many fields (<class>:<start>[:<modes>][:<period>])";
        return std::nullopt;
    }
    return config;
}

std::string ClassLabel(const Config& config) {
    return config.cls == Class::Program ? fmt::format("prog:{:016x}", config.program)
                                        : std::string{ClassName(config.cls)};
}

std::optional<Slot> SlotAt(const Config& config, u64 presents) noexcept {
    if (presents < config.start || config.period == 0 || config.modes.empty()) {
        return std::nullopt;
    }
    const u64 since = presents - config.start;
    Slot slot;
    slot.index = since / config.period;
    // Even slots are base, odd slots the measured modes in turn.
    slot.mode = slot.index % 2 == 0 ? Mode::Base
                                    : config.modes[(slot.index / 2) % config.modes.size()];
    slot.measured = since % config.period >= (std::min)(WARMUP, config.period / 2);
    slot.first_present = config.start + slot.index * config.period;
    return slot;
}

bool DrawInClass(const Config& config, u32 color_targets, bool depth_target,
                 std::span<const u64> stage_hashes) noexcept {
    switch (config.cls) {
    case Class::Depth:
        return depth_target && color_targets == 0;
    case Class::Color:
        return color_targets != 0;
    case Class::Compute:
        return false;
    case Class::Program:
        return std::find(stage_hashes.begin(), stage_hashes.end(), config.program) !=
               stage_hashes.end();
    case Class::All:
        return true;
    }
    return false;
}

bool DispatchInClass(const Config& config, u64 program) noexcept {
    switch (config.cls) {
    case Class::Compute:
    case Class::All:
        return true;
    case Class::Program:
        return program == config.program;
    case Class::Depth:
    case Class::Color:
        return false;
    }
    return false;
}

Placement PlacementOfBacking(u32 backing) noexcept {
    return backing < static_cast<u32>(Placement::Arena) ? static_cast<Placement>(backing)
                                                        : Placement::None;
}

void Tally::Merge(const Tally& other) noexcept {
    for (size_t d = 0; d < DATA_COUNT; ++d) {
        for (size_t p = 0; p < PLACEMENT_COUNT; ++p) {
            count[d][p] += other.count[d][p];
        }
        bytes[d] += other.bytes[d];
    }
    copied += other.copied;
    ring_bytes += other.ring_bytes;
    clamped += other.clamped;
}

u64 Tally::Bindings(Data data) const noexcept {
    u64 n = 0;
    for (const u64 c : count[static_cast<size_t>(data)]) {
        n += c;
    }
    return n;
}

void Stats::Merge(const Stats& other) noexcept {
    frames += other.frames;
    busy_ns += other.busy_ns;
    class_ns += other.class_ns;
    passes += other.passes;
    draws += other.draws;
    dispatches += other.dispatches;
    cpu_ns += other.cpu_ns;
    tally.Merge(other.tally);
}

std::string FormatLine(const Config& config, Mode mode, const Stats& s,
                       const std::optional<Slot>& slot) {
    const u64 f = s.frames;
    const u64 work = s.draws + s.dispatches;
    const double us_per = work != 0 ? s.class_ns / static_cast<double>(work) / 1e3 : 0.0;
    // Compute names its unit dispatches; classes that hold draws name it per draw (prog and all:
    // per draw or dispatch).
    const char* per_key = config.cls == Class::Compute ? "us_per_dispatch" : "us_per_draw";
    return fmt::format("[nvn] costsplit class={} mode={} {} frames={} busy_ms={:.3f} "
                       "class_ms={:.3f} passes={:.1f} draws={:.1f} dispatches={:.1f} {}={:.3f} "
                       "cpu_ms={:.3f}",
                       ClassLabel(config), ModeName(mode), ScopeText(slot), f,
                       PerFrame(s.busy_ns, f) / 1e6, PerFrame(s.class_ns, f) / 1e6,
                       PerFrame(static_cast<double>(s.passes), f),
                       PerFrame(static_cast<double>(s.draws), f),
                       PerFrame(static_cast<double>(s.dispatches), f), per_key, us_per,
                       PerFrame(s.cpu_ns, f) / 1e6);
}

std::string FormatTally(const Config& config, Mode mode, const Stats& s,
                        const std::optional<Slot>& slot, std::string_view ring) {
    const Tally& t = s.tally;
    const bool count_mode = mode == Mode::NoInst || mode == Mode::Tri3 ||
                            mode == Mode::NoGroups || mode == Mode::OneGroup;
    const u64 work = mode == Mode::NoGroups || mode == Mode::OneGroup ? s.dispatches : s.draws;
    return fmt::format(
        "[nvn] costsplit tally class={} mode={} {} frames={} draws={} dispatches={} "
        "placement=none/imported/mirror/device/virtual/arena ubo={} vtx={} idx={} ssbo={} "
        "bytes={}/{}/{}/{} copied={} ring_mib={:.1f} ring={} unclamped={}",
        ClassLabel(config), ModeName(mode), ScopeText(slot), s.frames, s.draws, s.dispatches,
        Buckets(t, Data::Uniform), Buckets(t, Data::Vertex), Buckets(t, Data::Index),
        Buckets(t, Data::Storage), AverageBytes(t, Data::Uniform), AverageBytes(t, Data::Vertex),
        AverageBytes(t, Data::Index), AverageBytes(t, Data::Storage), t.copied,
        static_cast<double>(t.ring_bytes) / (1024.0 * 1024.0), ring,
        count_mode && work > t.clamped ? work - t.clamped : 0);
}

Recorder::Recorder(Config config_) : config{std::move(config_)} {
    current = SlotAt(config, 0);
}

void Recorder::OnPresent(u64 presents) {
    if (current && current->measured) {
        std::scoped_lock lock{mutex};
        SlotStats& entry = slots[current->index];
        entry.slot = *current;
        entry.stats.draws += pending.draws;
        entry.stats.dispatches += pending.dispatches;
        entry.stats.cpu_ns += pending.cpu_ns;
        entry.stats.tally.Merge(pending.tally);
    }
    pending = {};
    const std::optional<Slot> next = SlotAt(config, presents);
    if (next && (!current || current->index != next->index)) {
        Emit(fmt::format("[nvn] costsplit schedule class={} slot={} mode={} present={} "
                         "warmup={} period={}",
                         ClassLabel(config), next->index, ModeName(next->mode), presents,
                         (std::min)(WARMUP, config.period / 2), config.period));
    }
    current = next;
}

void Recorder::SetRing(std::string_view kind) {
    std::scoped_lock lock{mutex};
    ring = kind;
}

void Recorder::AddGpu(u64 slot, bool present, double busy_ns, double class_ns, u64 passes) {
    std::scoped_lock lock{mutex};
    SlotStats& entry = slots[slot];
    if (const std::optional<Slot> s = SlotAt(config, config.start + slot * config.period)) {
        entry.slot.index = s->index;
        entry.slot.mode = s->mode;
        entry.slot.first_present = s->first_present;
        entry.slot.measured = true;
    }
    entry.stats.frames += present ? 1 : 0;
    entry.stats.busy_ns += busy_ns;
    entry.stats.class_ns += class_ns;
    entry.stats.passes += passes;
}

void Recorder::ReportFinished(u64 slot) {
    std::vector<SlotStats> done;
    {
        std::scoped_lock lock{mutex};
        while (!slots.empty() && slots.begin()->first < slot) {
            done.push_back(std::move(slots.begin()->second));
            slots.erase(slots.begin());
        }
    }
    for (const SlotStats& entry : done) {
        ReportSlot(entry);
    }
    if (!done.empty()) {
        // The totals so far: a run that is killed (no clean exit) still has them.
        ReportTotals();
    }
}

void Recorder::ReportSlot(const SlotStats& entry) {
    const size_t mode = static_cast<size_t>(entry.slot.mode);
    std::string ring_kind;
    {
        std::scoped_lock lock{mutex};
        run[mode].Merge(entry.stats);
        ring_kind = ring;
    }
    if (entry.stats.frames == 0) {
        return;
    }
    Emit(FormatLine(config, entry.slot.mode, entry.stats, entry.slot));
    Emit(FormatTally(config, entry.slot.mode, entry.stats, entry.slot, ring_kind));
}

void Recorder::ReportRun() {
    // The frame being recorded when the replay stopped never reached a present: dropped. The
    // slots still open are logged with the totals after them.
    ReportFinished(~u64{0});
}

void Recorder::ReportTotals() {
    std::array<Stats, MODE_COUNT> totals;
    std::string ring_kind;
    {
        std::scoped_lock lock{mutex};
        totals = run;
        ring_kind = ring;
    }
    // Base first, then the measured modes in schedule order.
    std::vector<Mode> order{Mode::Base};
    order.insert(order.end(), config.modes.begin(), config.modes.end());
    for (const Mode mode : order) {
        const Stats& stats = totals[static_cast<size_t>(mode)];
        if (stats.frames == 0) {
            continue;
        }
        Emit(FormatLine(config, mode, stats, std::nullopt));
        Emit(FormatTally(config, mode, stats, std::nullopt, ring_kind));
    }
}

void Recorder::Emit(const std::string& line) {
    if (sink_fn != nullptr) {
        sink_fn(sink_user, line);
        return;
    }
    LOG_INFO(Render_Vulkan, "{}", line);
}

} // namespace Vulkan::Nvn::CostSplit

// ---------------------------------------------------------------------------------------------
// The draw path's hooks (DrawContext; replay thread)
// ---------------------------------------------------------------------------------------------

namespace Vulkan::Nvn {

namespace {

/// Vram: the ring and the largest range copied into it. Ranges are not fenced: the ring is big
/// enough for several frames of a class's ranges, older copies are overwritten after it wraps.
constexpr u64 RING_SIZE = u64{1} << 30;
constexpr u64 RING_COPY_MAX = u64{4} << 20;
constexpr u64 RING_ALIGNMENT = 256;

} // namespace

DrawContext* DrawContext::CostSplitBeginDraw() {
    cs_mode = CostSplit::Mode::Count;
    cs_draw_mode = CostSplit::Mode::Base;
    const std::optional<CostSplit::Slot>& slot = costsplit->Current();
    if (!slot) {
        return nullptr;
    }
    const CostSplit::Config& config = costsplit->GetConfig();
    u32 colors = 0;
    for (u32 i = 0; i < color_target_count; ++i) {
        colors += color_targets[i].texture ? 1 : 0;
    }
    std::array<u64, 7> hashes{};
    if (CostSplit::NeedsPrograms(config)) {
        constexpr std::array<u32, 7> DRAW_STAGES{0, 1, 2, 3, 4, STAGE_TASK, STAGE_MESH};
        for (size_t i = 0; i < DRAW_STAGES.size(); ++i) {
            hashes[i] = ProfStageHash(DRAW_STAGES[i]);
        }
    }
    if (!CostSplit::DrawInClass(config, colors, depth_target.texture != nullptr, hashes)) {
        return nullptr;
    }
    cs_mode = CostSplit::AppliesToDraws(slot->mode) ? slot->mode : CostSplit::Mode::Base;
    cs_measured = slot->measured;
    if (cs_measured) {
        cs_start = std::chrono::steady_clock::now();
    }
    return this;
}

DrawContext* DrawContext::CostSplitBeginDispatch(u64 program) {
    cs_mode = CostSplit::Mode::Count;
    const std::optional<CostSplit::Slot>& slot = costsplit->Current();
    if (!slot || !CostSplit::DispatchInClass(costsplit->GetConfig(), program)) {
        return nullptr;
    }
    cs_mode = CostSplit::AppliesToDispatches(slot->mode) ? slot->mode : CostSplit::Mode::Base;
    cs_measured = slot->measured;
    if (cs_measured) {
        cs_start = std::chrono::steady_clock::now();
    }
    return this;
}

void DrawContext::CostSplitEnd(bool dispatch, bool recorded) {
    if (cs_measured) {
        CostSplit::Stats& pending = costsplit->Pending();
        pending.cpu_ns +=
            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - cs_start)
                .count();
        if (recorded) {
            ++(dispatch ? pending.dispatches : pending.draws);
        }
    }
    if (!dispatch) {
        cs_draw_mode = cs_mode;
    }
    cs_mode = CostSplit::Mode::Count;
}

void DrawContext::CostSplitOnPass() {
    cs_pass_draws = 0;
    cs_draw_index = 0;
    cs_pushed_layout = VK_NULL_HANDLE;
    cs_bound_streams = 0;
    cs_index_bound = false;
}

void DrawContext::CostSplitNote(CostSplit::Data data, const BufferRef& ref, u64 size, bool arena) {
    if (!cs_measured) {
        return;
    }
    const CostSplit::Placement placement =
        arena         ? CostSplit::Placement::Arena
        : ref.pool    ? CostSplit::PlacementOfBacking(static_cast<u32>(ref.pool->backing))
                      : CostSplit::Placement::None;
    costsplit->Pending().tally.Note(data, placement, size);
}

bool DrawContext::CostSplitToRing(CostSplit::Data data, BufferRef& ref, u64 address, u64 size) {
    if (!CostSplit::CopiesToRing(cs_mode, data) || ref.pool == nullptr ||
        ref.pool->backing != Pool::Backing::Imported || size == 0 || size > RING_COPY_MAX ||
        cs_ring_failed) {
        return false;
    }
    if (!cs_ring) {
        try {
            cs_ring = impl.CreateHostBuffer(RING_SIZE,
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                            MemoryUsage::Stream);
        } catch (const vk::Exception& exception) {
            LOG_ERROR(Render_Vulkan, "[nvn] costsplit: vram ring: {}", exception.what());
        }
        if (!cs_ring || !cs_ring.IsHostVisible() || !cs_ring.IsHostCoherent()) {
            LOG_ERROR(Render_Vulkan, "[nvn] costsplit: no host-visible coherent ring; vram copies "
                                     "nothing");
            cs_ring = {};
            cs_ring_failed = true;
            costsplit->SetRing("failed");
            return false;
        }
        const VkPhysicalDeviceMemoryProperties props =
            impl.device.GetPhysical().GetMemoryProperties().memoryProperties;
        const u32 type = cs_ring.Location().memory_type;
        const VkMemoryPropertyFlags flags =
            type < props.memoryTypeCount ? props.memoryTypes[type].propertyFlags : 0;
        const bool device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        costsplit->SetRing(device_local ? "device" : "host");
        LOG_INFO(Render_Vulkan,
                 "[nvn] costsplit: vram ring {} MiB, memory type {} flags {:#x} ({})",
                 RING_SIZE >> 20, type, flags, device_local ? "device-local" : "host memory");
    }
    const u8* source = impl.GuestPointer(address, size);
    if (source == nullptr) {
        return false;
    }
    u64 offset = (cs_ring_used + RING_ALIGNMENT - 1) & ~(RING_ALIGNMENT - 1);
    if (offset + size > RING_SIZE) {
        offset = 0;
    }
    std::memcpy(cs_ring.Mapped().data() + offset, source, size);
    cs_ring_used = offset + size;
    ref = BufferRef{*cs_ring, offset, size, nullptr, 0};
    if (cs_measured) {
        CostSplit::Tally& tally = costsplit->Pending().tally;
        ++tally.copied;
        tally.ring_bytes += size;
    }
    return true;
}

void DrawContext::CostSplitClamped() {
    if (cs_measured) {
        ++costsplit->Pending().tally.clamped;
    }
}

} // namespace Vulkan::Nvn
