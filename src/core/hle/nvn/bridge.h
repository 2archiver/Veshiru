// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "core/hle/kernel/svc_types.h"
#include "core/hle/nvn/config.h"
#include "core/hle/nvn/thunk_encoding.h"

namespace Core {
class ArmInterface;
class System;
} // namespace Core

namespace Kernel {
class KThread;
}

namespace Core::NvnHle {

/// HLE function body. Reads its arguments from / writes its results to the guest register context.
/// The bridge sets pc = x30 after the body returns, so the thunk's ldr/br tail never runs.
using HleFn = void (*)(Core::System&, Kernel::Svc::ThreadContext&);

/// Guest register and memory access for the bridge. Real implementation wraps the ArmInterface of
/// the halted core; tests supply a fake.
class GuestPort {
public:
    virtual ~GuestPort() = default;
    virtual void GetContext(Kernel::Svc::ThreadContext& ctx) const = 0;
    virtual void SetContext(const Kernel::Svc::ThreadContext& ctx) = 0;
    virtual bool ReadBlock(u64 address, void* destination, size_t size) = 0;
    virtual bool WriteBlock(u64 address, const void* source, size_t size) = 0;
    virtual std::string ReadCString(u64 address, size_t max_length) = 0;
};

enum class SlotKind : u8 {
    Free,
    /// Pass-through counting thunk. May carry an HLE body (nv:: hooks) when HLE is active.
    Count,
    /// Count + log x0/x1 (nv::InitializeGraphics).
    CountLog,
    /// Pass-through that swaps lr for the return trampoline and replaces x0 on return.
    WrapProc,
    /// HLE-implemented NVN function (registered body, or the unimplemented stub).
    Hle,
    /// HLE nvnBootstrapLoader(name)
    HleBootstrap,
    /// HLE nvnDeviceGetProcAddress(device, name)
    HleProcAddress,
    /// Shared return trampoline (slot 2047).
    Return,
};

/// Everything LoadModule needs to prepare the sdk image (load pass).
struct SdkImage {
    std::span<u8> image;       ///< codeset.memory, already extended by Thunk::RegionBytes
    u32 region_offset{};       ///< offset of the thunk region in `image` (= module-relative)
    u64 load_base{};           ///< guest VA the module is loaded at
    std::string_view build_id; ///< hex build id for the log
};

/// The names the game sdk's own nvnDeviceGetProcAddress / nvnBootstrapLoader resolve (3.2.0
/// sdk+0x5da700: a binary search over a sorted table of names without the "nvn" prefix, then a
/// strcmp chain of extension names). Read from the module image itself.
struct SdkProcNames {
    std::vector<std::string> table;  ///< the binary-searched table, sorted, with the "nvn" prefix
    std::vector<std::string> extras; ///< the strcmp-chain names (NVX extensions), sorted
};

/// Finds the table in `image` (empty when the image has none). The table: an array of u32 string
/// offsets that starts at offset 0 and lists ascending identifiers, located through the name
/// "DeviceGetProcAddress" every NVN SDK has. The extras: the other NUL-delimited "nvn<Upper>..."
/// identifiers of the image outside the dynamic string table.
[[nodiscard]] SdkProcNames FindSdkProcNames(std::span<const u8> image);

class Bridge {
public:
    Bridge(Config config, std::span<const std::string_view> function_names);
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    /// Process-wide bridge configured from the environment and the real NVN function table.
    static Bridge& Instance();

    /// PhysicalCore hook: handles `svc` when it is a thunk SVC of the registered region. Returns
    /// false when the call is not ours, in which case the caller falls through to Svc::Call.
    static bool Handle(Core::System& system, Kernel::KThread* thread, Core::ArmInterface& interface,
                       u32 svc);

    /// System shutdown hook: stops the reporter and writes the final reports. Idempotent.
    static void Shutdown();

    /// Returns true when the renderer can host NVN HLE (RendererBase::NvnSink() != nullptr). It is
    /// evaluated at the first HLE entry, because the renderer does not exist yet at module load.
    using BackendProbe = bool (*)(Core::System*);
    void SetBackendProbe(BackendProbe probe);

    [[nodiscard]] const Config& GetConfig() const {
        return config;
    }

    /// Whether the next PrepareSdk takes NVN over. The process-wide bridge outlives a game, and
    /// the graphics path is chosen per game, so the loader sets it before every PrepareSdk.
    void SetHle(bool on) {
        config.hle = on;
    }

    /// Replaces the NVN function name table (id = index). Names must outlive the bridge.
    void SetFunctionNames(std::span<const std::string_view> names);

    /// Registers the HLE body of NVN function `function_id` (index in nvn_functions.inc).
    void RegisterHle(u32 function_id, HleFn fn);

    /// Registers an HLE body for a hooked nv:: symbol (exact mangled name). Used when HLE is
    /// active; without a body the original guest function runs (pass-through).
    void RegisterNvHook(std::string_view mangled_name, HleFn fn);

    /// Like RegisterNvHook, but the body only observes the call: the original guest function runs
    /// afterwards (the body must not change the registers). For nv:: functions whose real work
    /// the rest of the sdk depends on (nv::InitializeGraphics sets the state nn::vi::Initialize
    /// checks).
    void RegisterNvObserver(std::string_view mangled_name, HleFn fn);

    /// Load pass: writes the slot code, resets all state, allocates hook slots for the sdk dynsym
    /// entries we intercept and rewrites their st_value in `sdk.image`. HLE is requested when
    /// the config says so; it is confirmed or dropped at the first guest call.
    /// Returns the number of hooks installed.
    size_t PrepareSdk(const SdkImage& sdk);

    /// After process.LoadModule + reprotect: marks the region live for `owner`.
    void Activate(const void* owner, u64 region_base, u64 title_id);

    /// Dispatch of one thunk SVC. See Handle.
    bool Dispatch(Core::System* system, GuestPort& port, const void* owner, u32 svc);

    /// Writes census-<title>.tsv and (HLE) unimplemented-<title>.tsv using `frames` for
    /// calls_per_frame. Returns false if a file could not be written.
    bool WriteReports(u64 frames, bool final_report = false);

    /// Stops the reporter thread and writes the final reports.
    void Stop();

    /// One row of unsupported-<title>.tsv.
    struct UnsupportedRow {
        std::string where; ///< recorder, replay or shader
        std::string reason;
        u64 count{};
    };
    using UnsupportedProvider = std::function<std::vector<UnsupportedRow>()>;
    /// The HLE runtime supplies every once-logged unsupported reason; written with the other TSVs.
    void SetUnsupportedProvider(UnsupportedProvider provider);

    // ---- Introspection (reports, tests) ----
    struct Row {
        std::string name;
        u64 calls{};
    };
    /// Every obtained function aggregated by name (0-call rows included), sorted by calls desc.
    [[nodiscard]] std::vector<Row> CensusRows() const;
    [[nodiscard]] std::vector<Row> UnimplementedRows() const;
    [[nodiscard]] static std::string FormatTsv(const std::vector<Row>& rows, u64 frames);
    [[nodiscard]] bool IsActive() const {
        return active.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsHleActive() const {
        return hle_active.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 SlotCalls(u32 slot) const;
    [[nodiscard]] SlotKind GetSlotKind(u32 slot) const;
    [[nodiscard]] std::string_view SlotName(u32 slot) const;
    [[nodiscard]] u32 AllocatedSlots() const {
        return next_slot.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 RegionBase() const {
        return region_base;
    }
    /// Names in the game sdk's GetProcAddress table (0 = not found: proc lookups fall back to the
    /// HLE function table).
    [[nodiscard]] u32 SdkProcTableSize() const {
        return sdk_table_size;
    }

private:
    struct Slot {
        std::atomic<u64> calls{0};
        std::atomic<u8> kind{static_cast<u8>(SlotKind::Free)};
        u32 function_id{~0u};
        u8 name_arg{};
        HleFn hle{};
        bool observer{};
        std::string name;
    };

    struct Frame {
        u64 tpidr;
        u64 sp;
        u64 lr;
        u32 slot;
        std::string name;
    };

    /// Per-slot call counters feed the census and the thread dump only; without either, a call
    /// does no counting.
    void CountCall(Slot& slot) const {
        if (count_calls) {
            slot.calls.fetch_add(1, std::memory_order_relaxed);
        }
    }
    /// `config.output_dir`, or `OutputDirectory()` when the config leaves it empty.
    [[nodiscard]] std::filesystem::path ReportDirectory() const;
    s32 AllocateSlot();
    void Publish(u32 slot, SlotKind kind, std::string name, u32 function_id, u8 name_arg,
                 HleFn hle, bool observer = false);
    /// Slot (thunk) address for a proc the game obtained. 0 when no slot is left.
    u64 CountThunkFor(GuestPort& port, const std::string& name, u64 target);
    u64 HleThunkFor(u32 function_id);
    /// nvnDeviceGetProcAddress / nvnBootstrapLoader under HLE: NULL for a name the game's sdk does
    /// not resolve, the HLE thunk otherwise. Logs each distinct name once with its answer.
    u64 ResolveHleProc(const std::string& name);
    [[nodiscard]] u64 SlotAddress(u32 slot) const {
        return region_base + Thunk::SlotOffset(slot);
    }

    [[nodiscard]] std::optional<u32> FunctionId(std::string_view name) const;
    [[nodiscard]] std::string_view FunctionName(u32 function_id) const;
    bool ResolveHle(Core::System* system);
    void CallHle(Core::System* system, GuestPort& port, Kernel::Svc::ThreadContext& ctx, HleFn fn);
    void CallUnimplemented(GuestPort& port, Kernel::Svc::ThreadContext& ctx, u32 function_id);
    bool BeginWrappedProc(GuestPort& port, Kernel::Svc::ThreadContext& ctx, Slot& slot,
                          u32 slot_index);
    bool EndWrappedProc(GuestPort& port, Kernel::Svc::ThreadContext& ctx);
    void ReporterLoop(std::stop_token token);
    void LogReport(const std::vector<Row>& rows, u64 frames, bool everything) const;

    static constexpr size_t MaxFunctions = 4096;

    Config config;
    const bool count_calls;
    std::span<const std::string_view> names;
    std::map<std::string, u32, std::less<>> name_ids;
    std::array<HleFn, MaxFunctions> hle_bodies{};
    std::map<std::string, HleFn, std::less<>> nv_hooks;
    std::map<std::string, HleFn, std::less<>> nv_observers;
    std::array<std::atomic<u64>, MaxFunctions> unimplemented_calls{};
    std::array<std::atomic<bool>, MaxFunctions> unimplemented_logged{};

    std::unique_ptr<Slot[]> slots;
    std::atomic<u32> next_slot{0};
    std::mutex alloc_mutex;
    std::mutex thunk_mutex;
    std::map<std::pair<u64, std::string>, u32> count_thunks;
    std::vector<s32> hle_slots;
    /// FindSdkProcNames of the prepared sdk: table and extras, with the "nvn" prefix.
    std::set<std::string, std::less<>> sdk_procs;
    u32 sdk_table_size{};
    std::mutex proc_log_mutex;
    std::set<std::string, std::less<>> proc_logged;

    std::mutex frame_mutex;
    std::vector<Frame> shadow_frames;

    std::atomic<bool> active{false};
    std::atomic<bool> shut_down{false};
    const void* owner{};
    u64 region_base{};
    u64 title_id{};
    std::string build_id;
    std::atomic<bool> hle_active{false};
    std::atomic<u8> hle_state{0}; // 0 undecided, 1 active, 2 backend unavailable
    BackendProbe backend_probe;

    std::mutex report_mutex;
    std::mutex provider_mutex;
    UnsupportedProvider unsupported_provider;
    std::mutex stop_mutex;
    std::condition_variable_any stop_cv;
    std::jthread reporter;
    bool stopped{true};
    bool census_written{};
    std::atomic<bool> missing_frame_logged{false};
};

} // namespace Core::NvnHle
