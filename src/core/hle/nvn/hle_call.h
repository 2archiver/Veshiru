// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

#include "common/common_types.h"
#include "core/hle/kernel/svc_types.h"
#include "core/hle/nvn/aapcs64.h"
#include "core/hle/nvn/bridge.h"
#include "core/hle/nvn/sdk_device_info.h"
#include "core/hle/nvn/sdk_dlss_layout.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/recorder.h"

namespace Core {
class System;
}

/// NVN function bodies. One `functions_<family>.cpp` per family registers its handlers with the
/// bridge; a handler receives a `Call`, reads its arguments with `args` in prototype order (the
/// AAPCS64 rules of aapcs64.h), acts on the guest objects and the `Recorder`, and writes its result
/// with `ret`. The bridge then returns to the caller (pc = lr).
namespace Core::NvnHle::Hle {

class Runtime;

/// State of the NVN device the game created. NVN has one device per process; the HLE keeps what
/// the game sets on it here (it is not stored in the guest object, which only carries the tag).
struct DeviceState {
    std::atomic<s32> window_origin_mode{0}; ///< NVNwindowOriginMode: 0 lower left (default), 1 upper left
    std::atomic<s32> depth_mode{0};         ///< NVNdepthMode: 0 near is -w (default), 1 near is 0
    std::atomic<u32> flags{0};
    /// Names in the game sdk's own GetProcAddress table (Bridge::PrepareSdk; 0 = not found). Tells
    /// the SDK generation apart where the SDKs answer differently.
    std::atomic<u32> sdk_proc_names{0};
    /// The sdk's own nvnDeviceGetInteger(API_MINOR), -1 when its switch was not read.
    std::atomic<s32> sdk_api_minor{-1};
    /// API minor of the 3.2.0-generation SDK: the first one that is not "older".
    static constexpr s32 NewestKnownOlderApiMinor = 30;
    /// The sdk's nvnDeviceGetTextureHandle answers the null handle (0) for texture id 0, whatever
    /// the sampler id (SdkTextureHandleNullForTextureZero; set by Bridge::PrepareSdk). Ocarina of
    /// Time's SDK does, Donkey Kong Bananza's (1.0.0 and 3.2.0) does not.
    std::atomic<bool> texture_zero_null_handle{false};

    /// The game's sdk is older than the 3.2.0-generation SDK (API minor 30; 1.0.0 has 16). The
    /// generation is the sdk's own API minor version when its device-info switch was read (a newer
    /// SDK such as Mario Kart World's, minor 39, is newer whatever its proc-name table holds);
    /// otherwise the size of its GetProcAddress table (726 names; 1.0.0 has 698). Unknown counts as
    /// the newest.
    [[nodiscard]] bool OlderSdk() const {
        const s32 minor = sdk_api_minor.load(std::memory_order_relaxed);
        if (minor >= 0) {
            return minor < NewestKnownOlderApiMinor;
        }
        const u32 names = sdk_proc_names.load(std::memory_order_relaxed);
        return names != 0 && names < 726;
    }

    /// What the loaded game's own SDK answers to nvnDeviceGetInteger (set when the bridge hooks
    /// the SDK, empty when its switches were not recognised).
    void SetSdkInfo(std::shared_ptr<const SdkDeviceInfo> info) {
        const std::optional<s32> minor = info ? info->Get(1) : std::nullopt;
        sdk_api_minor.store(minor.value_or(-1), std::memory_order_relaxed);
        std::scoped_lock lock{sdk_info_mutex};
        sdk_info = std::move(info);
    }
    [[nodiscard]] std::shared_ptr<const SdkDeviceInfo> SdkInfo() const {
        std::scoped_lock lock{sdk_info_mutex};
        return sdk_info;
    }

    /// The DLSS object layout of the loaded game's SDK (set when the bridge hooks the SDK; the
    /// newest known SDK's until then, and when the SDK's DLSS functions were not recognised).
    void SetDlssLayout(const DlssLayout& layout) {
        std::scoped_lock lock{sdk_info_mutex};
        dlss_layout = layout;
    }
    [[nodiscard]] DlssLayout GetDlssLayout() const {
        std::scoped_lock lock{sdk_info_mutex};
        return dlss_layout;
    }

private:
    mutable std::mutex sdk_info_mutex;
    std::shared_ptr<const SdkDeviceInfo> sdk_info;
    DlssLayout dlss_layout{NewestDlssLayout()};
};

/// One in-flight NVN function call.
class Call {
public:
    Call(Runtime& runtime, Core::System* system, Kernel::Svc::ThreadContext& ctx,
         VideoCommon::Nvn::GuestMemory& memory, VideoCommon::Nvn::Recorder& recorder);

    Runtime& runtime;
    Core::System* const system; ///< null in unit tests
    Kernel::Svc::ThreadContext& ctx;
    VideoCommon::Nvn::GuestMemory& mem;
    VideoCommon::Nvn::Recorder& rec;
    /// Arguments in prototype order: `args.NextPointer()`, `args.NextS32()`, `args.NextBool()`,
    /// `args.NextF32()`, ... (NVNboolean is one byte the caller does not extend: always NextBool).
    Aapcs64::ArgCursor args;
    /// Result: `ret.SetU64`, `SetU32`, `SetBool`, `SetF32`, ...
    Aapcs64::ReturnWriter ret;

    /// Reads a trivially copyable value from guest memory (zero when unmapped).
    template <typename T>
    [[nodiscard]] T Load(u64 va) {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        mem.Read(va, &value, sizeof(T));
        return value;
    }
    template <typename T>
    bool Store(u64 va, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        return mem.Write(va, &value, sizeof(T));
    }
    bool Zero(u64 va, size_t size);
    [[nodiscard]] std::string ReadString(u64 va, size_t max_length);

    /// Runs the registered HLE body of another NVN function (or hooked symbol) `name` as if the
    /// guest had called it with the integer arguments `int_args` (x0, x1, ...), on a copy of this
    /// call's registers; returns its x0 (0 when nothing is registered under `name`). The way an
    /// HLE body that stands for SDK code makes the NVN calls that code would make.
    u64 CallNvn(std::string_view name, std::initializer_list<u64> int_args);

    /// Blocks the calling guest thread until `ready()` returns true or `timeout` expired; negative
    /// timeout = infinite. The guest thread sleeps in the kernel (the physical core runs other
    /// guest threads meanwhile) and is woken by `Runtime::WakeWaiters` (done on every completed
    /// GPU sequence number and every CPU event signal), so `ready` must only look at atomics or at
    /// recorder state. Returns whether `ready()` became true.
    bool WaitUntil(const std::function<bool()>& ready, std::chrono::nanoseconds timeout,
                   std::source_location where = std::source_location::current());
};

using Handler = void (*)(Call&);

/// The CPU wrote [address, address + size) of a pool or buffer (nvnMemoryPoolFlushMappedRange,
/// nvnBufferFlushMappedRange): tells the replay. functions_pool.cpp.
void NoteMappedRangeWritten(Call& c, u64 address, u64 size);
/// The CPU is about to read GPU results from the range (nvnMemoryPoolInvalidateMappedRange,
/// nvnBufferInvalidateMappedRange). functions_pool.cpp.
void InvalidateMappedRange(Call& c, u64 address, u64 size);

/// Collects the handlers of every family. With a bridge, each handler is installed as the HLE body
/// of its function; without one (unit tests) they are only collected and can be invoked by name.
class Registry {
public:
    explicit Registry(Bridge* bridge) : bridge{bridge} {}

    /// Registers `H` as the body of the NVN function `name` (must be in nvn_functions.inc).
    template <Handler H>
    void Add(std::string_view name) {
        AddImpl(name, H, &Invoke<H>);
    }
    /// Registers `H` as the body of the hooked nv:: symbol `mangled_name`.
    template <Handler H>
    void AddNv(std::string_view mangled_name) {
        AddNvImpl(mangled_name, H, &Invoke<H>, false);
    }
    /// Registers `H` as an observer of the hooked nv:: symbol: it sees the call, then the original
    /// guest function runs (the body must not change the registers).
    template <Handler H>
    void ObserveNv(std::string_view mangled_name) {
        AddNvImpl(mangled_name, H, &InvokeObserver<H>, true);
    }

    /// The handler registered under `name` by any Registry of this process (Call::CallNvn).
    [[nodiscard]] static Handler Find(std::string_view name);

    [[nodiscard]] const std::map<std::string, Handler, std::less<>>& Handlers() const {
        return handlers;
    }
    [[nodiscard]] size_t Count() const {
        return handlers.size();
    }
    /// Names that were not in the function table (a typo), or registered twice.
    [[nodiscard]] size_t Rejected() const {
        return rejected;
    }

private:
    template <Handler H>
    static void Invoke(Core::System& system, Kernel::Svc::ThreadContext& ctx) {
        Dispatch(system, ctx, H, false);
    }
    template <Handler H>
    static void InvokeObserver(Core::System& system, Kernel::Svc::ThreadContext& ctx) {
        Dispatch(system, ctx, H, true);
    }
    static void Dispatch(Core::System& system, Kernel::Svc::ThreadContext& ctx, Handler handler,
                         bool observer);
    void AddImpl(std::string_view name, Handler handler, HleFn bridge_fn);
    void AddNvImpl(std::string_view name, Handler handler, HleFn bridge_fn, bool observer);

    Bridge* bridge;
    std::map<std::string, Handler, std::less<>> handlers;
    size_t rejected{};
};

/// Family registration functions (one per `functions_<family>.cpp`).
void RegisterDeviceFunctions(Registry&);
void RegisterQueueFunctions(Registry&);
void RegisterWindowFunctions(Registry&);
void RegisterNvFunctions(Registry&);
void RegisterNativeImageFunctions(Registry&);
void RegisterPoolFunctions(Registry&);
void RegisterBufferFunctions(Registry&);
void RegisterTextureFunctions(Registry&);
void RegisterSamplerFunctions(Registry&);
void RegisterProgramFunctions(Registry&);
void RegisterSyncFunctions(Registry&);
void RegisterEventFunctions(Registry&);
void RegisterDlssFunctions(Registry&);
void RegisterStateFunctions(Registry&);
void RegisterCommandBufferFunctions(Registry&);
void RegisterCommandBufferBindFunctions(Registry&);

/// Registers every family. Called before the sdk module is prepared.
void RegisterAll(Registry& registry);

/// Registers every family on `bridge` (production entry point).
void RegisterAll(Bridge& bridge);

} // namespace Core::NvnHle::Hle

/// Registers the handler `fn` (a function `void fn(Call&)` named exactly like the NVN function).
#define NVN_HLE(registry, fn) (registry).Add<&fn>(#fn)
