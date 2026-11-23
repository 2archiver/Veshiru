// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>

#include <fmt/format.h>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/nvn/bridge.h"
#include "core/hle/nvn/hle_running.h"
#include "core/hle/nvn/dynsym.h"
#include "core/hle/nvn/function_table.h"
#include "core/hle/nvn/hle_runtime.h"
#include "core/hle/nvn/sdk_device_info.h"
#include "core/hle/nvn/sdk_dlss_layout.h"
#include "core/hle/nvn/sdk_impl_table.h"
#include "core/memory.h"
#include "video_core/nvn/recorder.h"
#include "video_core/renderer_base.h"

namespace Core::NvnHle {
namespace {

constexpr size_t MaxProcNameLength = 256;
constexpr std::string_view BootstrapName = "nvnBootstrapLoader";
constexpr std::string_view ProcAddressName = "nvnDeviceGetProcAddress";

std::atomic<Bridge*>& InstancePointer() {
    static std::atomic<Bridge*> pointer{nullptr};
    return pointer;
}

/// The renderer exists by the time the guest runs; only the Vulkan one provides a sink.
bool RendererHostsNvn(Core::System* system) {
    return system != nullptr && system->Renderer().NvnSink() != nullptr;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

/// The sdk dynsym entries the bridge redirects to thunk slots.
bool IsHookedSymbol(std::string_view name) {
    static constexpr std::array<std::string_view, 8> exact{
        "nvnBootstrapLoader",          "nvnDeviceGetProcAddress",
        "_ZN2nv9InitCheckEv",          "_ZN2nv16FinalizeGraphicsEv",
        "_ZN2nv22DumpGraphicsMemoryInfoEv",
        // NVN2 native images: SDK code over driver extensions the HLE does not have, so the
        // HLE answers them (functions_native_image.cpp).
        "_Z24nvnNativeImageInitializeP14NVNnativeImagePK21NVNnativeImageBuilder",
        "_Z22nvnNativeImageFinalizeP14NVNnativeImage",
        "_Z35nvnTextureInitializeFromNativeImageP10NVNtextureP14NVNnativeImage",
    };
    static constexpr std::array<std::string_view, 4> prefixes{
        "_ZN2nv18InitializeGraphics",
        "_ZN2nv20SetGraphicsAllocator",
        "_ZN2nv28SetGraphicsDevtoolsAllocator",
        "_ZN2nv21GetGraphicsMemoryInfo",
    };
    return std::ranges::find(exact, name) != exact.end() ||
           std::ranges::any_of(prefixes, [&](std::string_view p) { return StartsWith(name, p); });
}

/// Real port: the halted core's ArmInterface and the process memory.
class CorePort final : public GuestPort {
public:
    CorePort(Core::System& system_, Core::ArmInterface& interface_, Core::Memory::Memory& memory_,
             Kernel::KThread* thread_)
        : system{system_}, interface{interface_}, memory{memory_}, thread{thread_} {}

    void GetContext(Kernel::Svc::ThreadContext& ctx) const override {
        interface.GetContext(ctx);
    }
    void SetContext(const Kernel::Svc::ThreadContext& ctx) override {
        // A handler that slept in the kernel can resume on another physical core: the thread's
        // registers are then loaded into that core's interface (as Svc::Call does for its results).
        Core::ArmInterface* target = interface_of_current_core();
        (target != nullptr ? *target : interface).SetContext(ctx);
        // Keep the kernel's copy in sync, as the breakpoint path in PhysicalCore does.
        thread->GetContext() = ctx;
    }
    bool ReadBlock(u64 address, void* destination, size_t size) override {
        return memory.ReadBlock(address, destination, size);
    }
    bool WriteBlock(u64 address, const void* source, size_t size) override {
        return memory.WriteBlock(address, source, size);
    }
    std::string ReadCString(u64 address, size_t max_length) override {
        return memory.ReadCString(address, max_length);
    }

private:
    Core::ArmInterface* interface_of_current_core() const {
        return thread->GetOwnerProcess()->GetArmInterface(system.Kernel().CurrentPhysicalCoreIndex());
    }

    Core::System& system;
    Core::ArmInterface& interface;
    Core::Memory::Memory& memory;
    Kernel::KThread* thread;
};

bool WriteFileAtomically(const std::filesystem::path& path, const std::string& text) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out{temporary, std::ios::binary | std::ios::trunc};
        if (!out) {
            return false;
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.close();
        if (!out) {
            std::filesystem::remove(temporary, ec);
            return false;
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        // Some filesystems refuse to replace a file that a reader holds open: write in place.
        std::filesystem::remove(temporary, ec);
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.close();
        return static_cast<bool>(out);
    }
    return true;
}

bool IsIdentifierChar(u8 c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

bool IsIdentifier(std::string_view text) {
    return !text.empty() &&
           std::ranges::all_of(text, [](char c) { return IsIdentifierChar(static_cast<u8>(c)); });
}

/// The NUL-terminated string at `offset`; empty when it runs off the image.
std::string_view CStringAt(std::span<const u8> image, size_t offset) {
    if (offset >= image.size()) {
        return {};
    }
    const auto* begin = reinterpret_cast<const char*>(image.data()) + offset;
    const size_t limit = image.size() - offset;
    const size_t length = std::string_view{begin, limit}.find('\0');
    return length == std::string_view::npos ? std::string_view{} : std::string_view{begin, length};
}

u32 U32At(std::span<const u8> image, size_t offset) {
    u32 value{};
    std::memcpy(&value, image.data() + offset, sizeof(value));
    return value;
}

/// [start, end) of the module's dynamic string table, or {0, 0}. The end is the end of the longest
/// symbol name (DynsymLayout has no DT_STRSZ); the symbols are the entries before DT_STRTAB.
std::pair<size_t, size_t> DynstrRange(std::span<const u8> image) {
    DynsymLayout layout;
    if (!ProbeDynsym(image, layout) || layout.entry_size < sizeof(u32) ||
        layout.strtab <= layout.symtab || layout.strtab >= image.size()) {
        return {0, 0};
    }
    size_t end = layout.strtab;
    for (u64 entry = layout.symtab; entry + layout.entry_size <= layout.strtab;
         entry += layout.entry_size) {
        const size_t name = layout.strtab + U32At(image, entry);
        const std::string_view text = CStringAt(image, name);
        end = std::max(end, name + text.size() + 1);
    }
    return {layout.strtab, std::min(end, image.size())};
}

} // namespace

SdkProcNames FindSdkProcNames(std::span<const u8> image) {
    // 1. The table's string blob: NUL-separated names without the "nvn" prefix in ascending
    // order. Of the occurrences of one name every NVN SDK has, take the longest sorted run.
    constexpr std::string_view Anchor = "DeviceGetProcAddress";
    const std::string_view text{reinterpret_cast<const char*>(image.data()), image.size()};
    std::string needle{'\0'};
    needle.append(Anchor).push_back('\0');
    size_t run_first = 0;
    size_t run_count = 0;
    for (size_t pos = text.find(needle); pos != std::string_view::npos;
         pos = text.find(needle, pos + 1)) {
        const size_t start = pos + 1;
        size_t first = start;
        std::string_view lowest = Anchor;
        size_t count = 1;
        while (first >= 2 && image[first - 1] == 0) {
            size_t previous = first - 1;
            while (previous > 0 && image[previous - 1] != 0) {
                --previous;
            }
            const std::string_view name = text.substr(previous, first - 1 - previous);
            if (!IsIdentifier(name) || !(name < lowest)) {
                break;
            }
            lowest = name;
            first = previous;
            ++count;
        }
        std::string_view highest = Anchor;
        for (size_t next = start + Anchor.size() + 1;;) {
            const std::string_view name = CStringAt(image, next);
            if (!IsIdentifier(name) || !(highest < name)) {
                break;
            }
            highest = name;
            ++count;
            next += name.size() + 1;
        }
        if (count > run_count) {
            run_first = first;
            run_count = count;
        }
    }
    SdkProcNames result;
    if (run_count < 2) {
        return result;
    }
    // 2. The u32 offset table the SDK binary-searches (entry 0 = offset 0 = the blob's start). The
    // run may include unrelated identifiers before the blob's real first name (anything that sorts
    // lower and follows a NUL), so try the run's first few names as the table's first name: find
    // the entry pair that holds that name and the next one (offset 0 for the blob's first name),
    // and walk back to the zero entry.
    constexpr size_t FirstNameAttempts = 4;
    const size_t words = image.size() / sizeof(u32);
    size_t first_name = run_first;
    for (size_t attempt = 0; attempt < FirstNameAttempts && result.table.empty(); ++attempt) {
        const size_t delta = CStringAt(image, first_name).size() + 1;
        for (size_t i = 0; i + 1 < words; ++i) {
            const u32 offset = U32At(image, i * 4);
            if (offset > first_name || U32At(image, (i + 1) * 4) != offset + delta) {
                continue;
            }
            const size_t base = first_name - offset;
            size_t entry = i;
            while (entry > 0 && U32At(image, entry * 4) != 0 &&
                   base + U32At(image, (entry - 1) * 4) < image.size()) {
                --entry;
            }
            if (U32At(image, entry * 4) != 0) {
                continue;
            }
            std::vector<std::string> names;
            std::string_view previous;
            for (size_t k = entry; k < words; ++k) {
                const std::string_view name = CStringAt(image, base + U32At(image, k * 4));
                if (!IsIdentifier(name) || (!names.empty() && !(previous < name))) {
                    break;
                }
                names.push_back(fmt::format("nvn{}", name));
                previous = name;
            }
            if (names.size() + attempt >= run_count) {
                result.table = std::move(names);
                break;
            }
        }
        first_name += delta;
    }
    if (result.table.empty()) {
        return result;
    }
    // 3. The names the SDK compares one by one after the table (the NVX extensions): the other
    // whole "nvn<Upper>..." strings of the image, outside the dynamic string table.
    const auto [dynstr_begin, dynstr_end] = DynstrRange(image);
    std::set<std::string, std::less<>> extras;
    for (size_t pos = text.find("nvn"); pos != std::string_view::npos; pos = text.find("nvn", pos + 1)) {
        if ((pos != 0 && image[pos - 1] != 0) || (pos >= dynstr_begin && pos < dynstr_end) ||
            pos + 3 >= image.size() || image[pos + 3] < 'A' || image[pos + 3] > 'Z') {
            continue;
        }
        const std::string_view name = CStringAt(image, pos);
        if (IsIdentifier(name) &&
            !std::binary_search(result.table.begin(), result.table.end(), name, std::less<>{})) {
            extras.emplace(name);
        }
    }
    result.extras.assign(extras.begin(), extras.end());
    return result;
}

std::filesystem::path OutputDirectory() {
    if (const char* directory = std::getenv("EDEN_NX2_NVN_OUTPUT_DIR");
        directory != nullptr && *directory != '\0') {
        return std::filesystem::path{directory};
    }
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "nvn";
}

std::filesystem::path Bridge::ReportDirectory() const {
    return config.output_dir.empty() ? OutputDirectory() : std::filesystem::path{config.output_dir};
}

Bridge::Bridge(Config config_, std::span<const std::string_view> function_names)
    : config{std::move(config_)}, count_calls{config.CountCalls()},
      slots{std::make_unique<Slot[]>(Thunk::SlotCount)}, backend_probe{&RendererHostsNvn} {
    SetFunctionNames(function_names);
}

Bridge::~Bridge() {
    Stop();
}

Bridge& Bridge::Instance() {
    static Bridge instance{ConfigFromEnvironment(), FunctionNames()};
    InstancePointer().store(&instance, std::memory_order_release);
    return instance;
}

bool HleRunning() {
    const Bridge* const bridge = InstancePointer().load(std::memory_order_acquire);
    return bridge != nullptr && bridge->IsActive() && bridge->IsHleActive();
}

void Bridge::SetBackendProbe(BackendProbe probe) {
    backend_probe = probe;
}

bool Bridge::ResolveHle(Core::System* system) {
    u8 state = hle_state.load(std::memory_order_acquire);
    if (state == 0) {
        // The probe is cheap and idempotent, so racing first calls may both run it.
        const bool available = backend_probe != nullptr && backend_probe(system);
        if (available) {
            state = 1;
            LOG_INFO(Core, "[nvn] HLE active");
        } else {
            state = 2;
            hle_active.store(false, std::memory_order_release);
            LOG_WARNING(Core, "[nvn] graphics path: emulated GPU (the renderer has no NVN replay: "
                              "it is not the Vulkan one, or its context was not created, see "
                              "above)");
        }
        hle_state.store(state, std::memory_order_release);
    }
    return state == 1;
}

void Bridge::SetFunctionNames(std::span<const std::string_view> function_names) {
    names = function_names.first(std::min(function_names.size(), MaxFunctions));
    name_ids.clear();
    for (u32 id = 0; id < names.size(); ++id) {
        name_ids.emplace(std::string{names[id]}, id);
    }
}

void Bridge::RegisterHle(u32 function_id, HleFn fn) {
    if (function_id >= MaxFunctions) {
        LOG_ERROR(Core, "[nvn] RegisterHle: function id {} out of range", function_id);
        return;
    }
    hle_bodies[function_id] = fn;
}

void Bridge::RegisterNvHook(std::string_view mangled_name, HleFn fn) {
    nv_hooks[std::string{mangled_name}] = fn;
}

void Bridge::RegisterNvObserver(std::string_view mangled_name, HleFn fn) {
    nv_observers[std::string{mangled_name}] = fn;
}

std::optional<u32> Bridge::FunctionId(std::string_view name) const {
    const auto it = name_ids.find(name);
    if (it == name_ids.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string_view Bridge::FunctionName(u32 function_id) const {
    return function_id < names.size() ? names[function_id] : std::string_view{};
}

s32 Bridge::AllocateSlot() {
    std::scoped_lock lock{alloc_mutex};
    const u32 index = next_slot.load(std::memory_order_relaxed);
    if (index >= Thunk::ReturnSlot) {
        return -1;
    }
    next_slot.store(index + 1, std::memory_order_release);
    return static_cast<s32>(index);
}

void Bridge::Publish(u32 slot, SlotKind kind, std::string name, u32 function_id, u8 name_arg,
                     HleFn hle, bool observer) {
    Slot& entry = slots[slot];
    entry.observer = observer;
    entry.name = std::move(name);
    entry.function_id = function_id;
    entry.name_arg = name_arg;
    entry.hle = hle;
    entry.kind.store(static_cast<u8>(kind), std::memory_order_release);
}

size_t Bridge::PrepareSdk(const SdkImage& sdk) {
    const bool hle = config.hle;
    Stop();
    active.store(false, std::memory_order_release);
    shut_down.store(false, std::memory_order_release);
    for (u32 i = 0; i < Thunk::SlotCount; ++i) {
        slots[i].calls.store(0, std::memory_order_relaxed);
        slots[i].kind.store(static_cast<u8>(SlotKind::Free), std::memory_order_relaxed);
        slots[i].name.clear();
        slots[i].function_id = ~0u;
        slots[i].hle = nullptr;
    }
    next_slot.store(0, std::memory_order_relaxed);
    count_thunks.clear();
    hle_slots.assign(names.size(), -1);
    shadow_frames.clear();
    for (auto& counter : unimplemented_calls) {
        counter.store(0, std::memory_order_relaxed);
    }
    for (auto& logged : unimplemented_logged) {
        logged.store(false, std::memory_order_relaxed);
    }
    census_written = false;
    missing_frame_logged.store(false, std::memory_order_relaxed);
    hle_active.store(hle, std::memory_order_relaxed);
    hle_state.store(0, std::memory_order_relaxed);
    build_id = std::string{sdk.build_id};

    // The capabilities the game checks (API version, GPU code, ...) are the ones its own SDK's
    // nvnDeviceGetInteger answers, so they are read from it rather than taken from one fixed table.
    std::shared_ptr<const SdkDeviceInfo> device_info;
    if (hle) {
        if (auto info = SdkDeviceInfo::Read(sdk.image)) {
            LOG_INFO(Core, "[nvn] sdk device info: API {}.{}, GPU code {}.{}, {} constant items from "
                           "the switch at sdk+{:#x}",
                     info->Get(0).value_or(-1), info->Get(1).value_or(-1), info->Get(0x2E).value_or(-1),
                     info->Get(0x30).value_or(-1), info->ConstantCount(), info->SwitchAddress());
            device_info = std::make_shared<const SdkDeviceInfo>(std::move(*info));
        } else {
            LOG_WARNING(Core, "[nvn] the sdk's nvnDeviceGetInteger switch was not recognised: device "
                              "info answers from the built-in table");
        }
    }
    Hle::Runtime::Instance().device.SetSdkInfo(std::move(device_info));

    const auto region = sdk.image.subspan(sdk.region_offset, Thunk::RegionBytes);
    std::ranges::fill(region, u8{0});
    Thunk::WriteSlotCode(region);
    Publish(Thunk::ReturnSlot, SlotKind::Return, "<return>", ~0u, 0, nullptr);

    LOG_INFO(Core, "[nvn] sdk build id {} load base {:#x} thunk region sdk+{:#x}", build_id,
             sdk.load_base, sdk.region_offset);

    // The names the game's own sdk resolves through GetProcAddress, and its size as the SDK
    // generation the HLE bodies answer for (nvnDeviceGetInteger).
    {
        SdkProcNames procs = FindSdkProcNames(sdk.image.first(sdk.region_offset));
        sdk_table_size = static_cast<u32>(procs.table.size());
        sdk_procs.clear();
        sdk_procs.insert(procs.table.begin(), procs.table.end());
        sdk_procs.insert(procs.extras.begin(), procs.extras.end());
        std::scoped_lock lock{proc_log_mutex};
        proc_logged.clear();
        if (sdk_table_size == 0) {
            LOG_WARNING(Core, "[nvn] sdk GetProcAddress table not found: proc lookups answer from "
                              "the HLE function table");
        } else {
            LOG_INFO(Core, "[nvn] sdk GetProcAddress table: {} names, {} extension names",
                     sdk_table_size, procs.extras.size());
        }
        // The SDK's own implementations (its GetProcAddress implementation table) answer what
        // differs between SDK generations: the DLSS object sizes, the texture handle rule.
        const std::span<const u8> code = sdk.image.first(sdk.region_offset);
        const std::vector<u64> impl_tables =
            hle ? FindSdkImplTables(code, procs.table) : std::vector<u64>{};
        if (hle) {
            LOG_INFO(Core, "[nvn] sdk implementation table: {}",
                     impl_tables.size() == 1 ? fmt::format("sdk+{:#x}", impl_tables[0])
                                             : fmt::format("{} candidates", impl_tables.size()));
        }
        // The DLSS objects (builder, network, parameters) are plain data the SDK lays out
        // differently per generation: take their sizes from its own code.
        DlssLayout dlss = NewestDlssLayout();
        if (hle) {
            if (const std::optional<SdkDlssSizes> sizes =
                    ReadSdkDlssSizes(code, procs.table, impl_tables)) {
                dlss = DlssLayoutFor(*sizes);
                LOG_INFO(Core,
                         "[nvn] sdk DLSS objects: builder {:#x}, network {:#x} ({}), parameters "
                         "{:#x} bytes (SetDefaults sdk+{:#x}, Initialize sdk+{:#x}, parameters "
                         "SetDefaults sdk+{:#x})",
                         sizes->builder, sizes->network,
                         dlss.network != nullptr ? dlss.network->sdk : "no known layout",
                         sizes->parameters, sizes->builder_set_defaults,
                         sizes->network_initialize, sizes->parameters_set_defaults);
                if (dlss.network == nullptr) {
                    LOG_WARNING(Core, "[nvn] the sdk's DLSS network ({:#x} bytes) is of no known "
                                      "layout: nvnDlssNetworkInitialize refuses every network",
                                sizes->network);
                }
            } else if (sdk_procs.contains("nvnDlssNetworkInitialize")) {
                LOG_WARNING(Core, "[nvn] the sdk's DLSS functions were not recognised: the DLSS "
                                  "objects use the {} layout",
                            dlss.network->sdk);
            }
        }
        Hle::Runtime::Instance().device.SetDlssLayout(dlss);
        bool texture_zero_null = false;
        if (const std::optional<u64> texture_handle =
                SdkImplOffset(code, procs.table, impl_tables, "nvnDeviceGetTextureHandle")) {
            texture_zero_null = SdkTextureHandleNullForTextureZero(code, *texture_handle);
            LOG_INFO(Core, "[nvn] sdk nvnDeviceGetTextureHandle sdk+{:#x}: texture id 0 gives {}",
                     *texture_handle,
                     texture_zero_null ? "the null handle" : "id | sampler << 20 like any id");
        }
        Hle::Runtime::Instance().device.texture_zero_null_handle.store(texture_zero_null,
                                                                       std::memory_order_relaxed);
    }
    Hle::Runtime::Instance().device.sdk_proc_names.store(sdk_table_size, std::memory_order_relaxed);

    std::map<std::string, u64, std::less<>> rewrites;
    for (const DynsymHit& hit : FindDynsymHits(sdk.image, IsHookedSymbol)) {
        const s32 slot = AllocateSlot();
        if (slot < 0) {
            LOG_ERROR(Core, "[nvn] out of thunk slots while hooking {}", hit.name);
            break;
        }
        const u32 index = static_cast<u32>(slot);
        // The slot's pass-through target is the original function.
        const u64 original = sdk.load_base + hit.value;
        std::memcpy(region.data() + Thunk::TableEntryOffset(index), &original, sizeof(original));

        SlotKind kind = SlotKind::Count;
        u8 name_arg = 0;
        HleFn hle_body = nullptr;
        bool observer = false;
        if (hit.name == BootstrapName) {
            kind = hle ? SlotKind::HleBootstrap : SlotKind::WrapProc;
            name_arg = 0;
        } else if (hit.name == ProcAddressName) {
            kind = hle ? SlotKind::HleProcAddress : SlotKind::WrapProc;
            name_arg = 1;
        } else if (StartsWith(hit.name, "_ZN2nv18InitializeGraphics")) {
            kind = SlotKind::CountLog;
        }
        if (hle) {
            if (const auto it = nv_hooks.find(hit.name); it != nv_hooks.end()) {
                hle_body = it->second;
            } else if (const auto observed = nv_observers.find(hit.name);
                       observed != nv_observers.end()) {
                hle_body = observed->second;
                observer = true;
            }
        }
        Publish(index, kind, hit.name, ~0u, name_arg, hle_body, observer);
        rewrites[hit.name] = u64{sdk.region_offset} + Thunk::SlotOffset(index);
        LOG_INFO(Core, "[nvn] hook {} sdk+{:#x} -> slot {}", hit.name, hit.value, index);
    }
    const size_t rewritten = RewriteSymbolValues(sdk.image, rewrites);
    if (rewritten == 0) {
        LOG_WARNING(Core, "[nvn] no sdk dynsym entries were hooked (build id {})", build_id);
    }
    return rewritten;
}

void Bridge::Activate(const void* owner_, u64 base, u64 title) {
    owner = owner_;
    region_base = base;
    title_id = title;
    {
        std::scoped_lock lock{stop_mutex};
        stopped = false;
    }
    active.store(true, std::memory_order_release);
    if (config.census || config.hle) {
        const std::filesystem::path directory = ReportDirectory();
        if (!Common::FS::CreateDirs(directory)) {
            LOG_ERROR(Core, "[nvn] could not create the output directory {}", directory.string());
        }
        LOG_INFO(Core, "[nvn] output directory {}", directory.string());
        reporter = std::jthread{[this](std::stop_token token) { ReporterLoop(token); }};
    }
}

bool Bridge::Handle(Core::System& system, Kernel::KThread* thread, Core::ArmInterface& interface,
                    u32 svc) {
    if (svc < Thunk::SvcBase || svc >= Thunk::SvcBase + Thunk::SlotCount) {
        return false;
    }
    Bridge* bridge = InstancePointer().load(std::memory_order_acquire);
    if (bridge == nullptr) {
        return false;
    }
    if (!bridge->IsActive()) {
        if (!bridge->shut_down.load(std::memory_order_acquire)) {
            return false;
        }
        // Threads still running while the process shuts down: every thunk returns 0 to its caller.
        Kernel::Svc::ThreadContext ctx;
        interface.GetContext(ctx);
        ctx.r[0] = 0;
        ctx.v[0] = {0, 0};
        ctx.pc = ctx.lr;
        interface.SetContext(ctx);
        thread->GetContext() = ctx;
        return true;
    }
    auto* process = thread->GetOwnerProcess();
    CorePort port{system, interface, process->GetMemory(), thread};
    return bridge->Dispatch(&system, port, process, svc);
}

void Bridge::Shutdown() {
    if (Bridge* bridge = InstancePointer().load(std::memory_order_acquire)) {
        bridge->Stop();
        // After the final reports: nothing may dispatch into a stopped bridge.
        bridge->shut_down.store(true, std::memory_order_release);
        bridge->active.store(false, std::memory_order_release);
        bridge->owner = nullptr;
    }
}

void Bridge::SetUnsupportedProvider(UnsupportedProvider provider) {
    std::scoped_lock lock{provider_mutex};
    unsupported_provider = std::move(provider);
}

void Bridge::CallHle(Core::System* system, GuestPort& port, Kernel::Svc::ThreadContext& ctx,
                     HleFn fn) {
    fn(*system, ctx);
    // HLE functions return to their caller directly; the thunk's ldr/br tail never runs.
    ctx.pc = ctx.lr;
    port.SetContext(ctx);
}

void Bridge::CallUnimplemented(GuestPort& port, Kernel::Svc::ThreadContext& ctx, u32 function_id) {
    unimplemented_calls[function_id].fetch_add(1, std::memory_order_relaxed);
    if (!unimplemented_logged[function_id].exchange(true, std::memory_order_relaxed)) {
        LOG_WARNING(Core, "[nvn] unimplemented {}", FunctionName(function_id));
    }
    // Zero what a caller may read as the result: x0 (integers, pointers, small structs), x1 (a
    // 16-byte struct) and v0 (floats). No NVN function in the signature table returns a composite
    // through the x8 buffer, so there is no buffer to clear.
    ctx.r[0] = 0;
    ctx.r[1] = 0;
    ctx.v[0] = {0, 0};
    ctx.pc = ctx.lr;
    port.SetContext(ctx);
}

bool Bridge::BeginWrappedProc(GuestPort& port, Kernel::Svc::ThreadContext& ctx, Slot& slot,
                              u32 slot_index) {
    // The name argument does not survive the call, so read it now.
    std::string name = port.ReadCString(ctx.r[slot.name_arg], MaxProcNameLength);
    {
        std::scoped_lock lock{frame_mutex};
        shadow_frames.push_back({ctx.tpidr, ctx.sp, ctx.lr, slot_index, std::move(name)});
    }
    // The original function returns into the return trampoline instead of the game.
    ctx.lr = SlotAddress(Thunk::ReturnSlot);
    port.SetContext(ctx);
    return true;
}

bool Bridge::EndWrappedProc(GuestPort& port, Kernel::Svc::ThreadContext& ctx) {
    Frame frame;
    {
        std::scoped_lock lock{frame_mutex};
        const auto it = std::find_if(shadow_frames.rbegin(), shadow_frames.rend(), [&](const Frame& f) {
            return f.sp == ctx.sp && f.tpidr == ctx.tpidr;
        });
        if (it == shadow_frames.rend()) {
            // The caller is unknown, so there is nothing to return to. Leave pc on the slot's
            // ldr: slot 2047's table entry is 0, so the guest faults at address 0 (a normal guest
            // fault) instead of the host hitting the unhandled-SVC abort.
            if (!missing_frame_logged.exchange(true, std::memory_order_relaxed)) {
                LOG_CRITICAL(Core, "[nvn] return trampoline hit without a matching shadow frame");
            }
            return true;
        }
        frame = std::move(*it);
        shadow_frames.erase(std::next(it).base());
    }
    if (ctx.r[0] != 0) {
        if (const u64 thunk = CountThunkFor(port, frame.name, ctx.r[0]); thunk != 0) {
            ctx.r[0] = thunk;
        }
    }
    ctx.lr = frame.lr;
    ctx.pc = frame.lr;
    port.SetContext(ctx);
    return true;
}

u64 Bridge::CountThunkFor(GuestPort& port, const std::string& name, u64 target) {
    std::scoped_lock lock{thunk_mutex};
    const auto key = std::make_pair(target, name);
    if (const auto it = count_thunks.find(key); it != count_thunks.end()) {
        return SlotAddress(it->second);
    }
    const s32 slot = AllocateSlot();
    if (slot < 0) {
        LOG_ERROR(Core, "[nvn] out of thunk slots; {} stays uncounted", name);
        return 0;
    }
    const u32 index = static_cast<u32>(slot);
    if (!port.WriteBlock(region_base + Thunk::TableEntryOffset(index), &target, sizeof(target))) {
        LOG_ERROR(Core, "[nvn] could not write thunk table entry for {}", name);
        return 0;
    }
    SlotKind kind = SlotKind::Count;
    u8 name_arg = 0;
    if (name == ProcAddressName) {
        kind = SlotKind::WrapProc;
        name_arg = 1;
    } else if (name == BootstrapName) {
        kind = SlotKind::WrapProc;
    }
    Publish(index, kind, name, ~0u, name_arg, nullptr);
    count_thunks.emplace(key, index);
    return SlotAddress(index);
}

u64 Bridge::ResolveHleProc(const std::string& name) {
    u64 address = 0;
    std::string_view answer;
    if (sdk_table_size != 0 && !sdk_procs.contains(name)) {
        answer = "NULL (the sdk does not resolve it)";
    } else if (const auto id = FunctionId(name); !id) {
        answer = sdk_table_size != 0 ? "NULL (the sdk resolves it; the HLE has no function id for it)"
                                     : "NULL (no HLE function id)";
    } else {
        address = HleThunkFor(*id);
        answer = address == 0                ? "NULL (out of thunk slots)"
                 : hle_bodies[*id] != nullptr ? "HLE body"
                                              : "unimplemented stub (no HLE body)";
    }
    // Bounded: a guest probing many distinct names must not grow the cache without limit.
    constexpr size_t MaxProcLogged = 8192;
    std::scoped_lock lock{proc_log_mutex};
    if (proc_logged.size() < MaxProcLogged && proc_logged.insert(name).second) {
        LOG_INFO(Core, "[nvn] GetProcAddress {} -> {}", name, answer);
    }
    return address;
}

u64 Bridge::HleThunkFor(u32 function_id) {
    std::scoped_lock lock{thunk_mutex};
    if (function_id >= hle_slots.size()) {
        return 0;
    }
    if (hle_slots[function_id] < 0) {
        const s32 slot = AllocateSlot();
        if (slot < 0) {
            LOG_ERROR(Core, "[nvn] out of thunk slots for {}", FunctionName(function_id));
            return 0;
        }
        const std::string_view name = FunctionName(function_id);
        SlotKind kind = SlotKind::Hle;
        u8 name_arg = 0;
        if (name == ProcAddressName) {
            kind = SlotKind::HleProcAddress;
            name_arg = 1;
        } else if (name == BootstrapName) {
            kind = SlotKind::HleBootstrap;
        }
        Publish(static_cast<u32>(slot), kind, std::string{name}, function_id, name_arg, nullptr);
        hle_slots[function_id] = slot;
    }
    return SlotAddress(static_cast<u32>(hle_slots[function_id]));
}

bool Bridge::Dispatch(Core::System* system, GuestPort& port, const void* owner_, u32 svc) {
    if (!active.load(std::memory_order_acquire) || owner_ != owner) {
        return false;
    }
    if (svc < Thunk::SvcBase || svc >= Thunk::SvcBase + Thunk::SlotCount) {
        return false;
    }
    const u32 index = svc - Thunk::SvcBase;
    Kernel::Svc::ThreadContext ctx;
    port.GetContext(ctx);
    // The JIT advances pc past the svc before halting: it must sit right after our slot's svc.
    if (ctx.pc - 4 != region_base + Thunk::SlotOffset(index)) {
        return false;
    }

    Slot& slot = slots[index];
    switch (static_cast<SlotKind>(slot.kind.load(std::memory_order_acquire))) {
    case SlotKind::Free:
        return false;
    case SlotKind::CountLog:
        LOG_INFO(Core, "[nvn] {}(x0={:#x}, x1={:#x})", slot.name, ctx.r[0], ctx.r[1]);
        [[fallthrough]];
    case SlotKind::Count:
        CountCall(slot);
        if (slot.hle != nullptr && hle_active.load(std::memory_order_acquire) && ResolveHle(system)) {
            if (slot.observer) {
                // Observed only: the thunk's ldr/br tail runs the original function.
                slot.hle(*system, ctx);
            } else {
                CallHle(system, port, ctx, slot.hle);
            }
        }
        return true;
    case SlotKind::WrapProc:
        CountCall(slot);
        return BeginWrappedProc(port, ctx, slot, index);
    case SlotKind::Return:
        return EndWrappedProc(port, ctx);
    case SlotKind::Hle: {
        CountCall(slot);
        const HleFn fn = hle_bodies[slot.function_id];
        if (fn != nullptr) {
            // Names the call in the Recorder's rejected-argument logs (see g_current_nvn_call).
            const char* const outer_call = VideoCommon::Nvn::g_current_nvn_call;
            VideoCommon::Nvn::g_current_nvn_call = slot.name.c_str();
            CallHle(system, port, ctx, fn);
            VideoCommon::Nvn::g_current_nvn_call = outer_call;
        } else {
            CallUnimplemented(port, ctx, slot.function_id);
        }
        return true;
    }
    case SlotKind::HleBootstrap:
    case SlotKind::HleProcAddress: {
        CountCall(slot);
        if (!ResolveHle(system)) {
            // GPU path: behave as the census wrapper (or a plain pass-through without census).
            return config.census ? BeginWrappedProc(port, ctx, slot, index) : true;
        }
        ctx.r[0] = ResolveHleProc(port.ReadCString(ctx.r[slot.name_arg], MaxProcNameLength));
        ctx.pc = ctx.lr;
        port.SetContext(ctx);
        return true;
    }
    }
    return false;
}

u64 Bridge::SlotCalls(u32 slot) const {
    return slot < Thunk::SlotCount ? slots[slot].calls.load(std::memory_order_relaxed) : 0;
}

SlotKind Bridge::GetSlotKind(u32 slot) const {
    return slot < Thunk::SlotCount
               ? static_cast<SlotKind>(slots[slot].kind.load(std::memory_order_acquire))
               : SlotKind::Free;
}

std::string_view Bridge::SlotName(u32 slot) const {
    return slot < Thunk::SlotCount && GetSlotKind(slot) != SlotKind::Free
               ? std::string_view{slots[slot].name}
               : std::string_view{};
}

std::vector<Bridge::Row> Bridge::CensusRows() const {
    std::map<std::string, u64> by_name;
    const u32 count = next_slot.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        const SlotKind kind = GetSlotKind(i);
        if (kind == SlotKind::Free || kind == SlotKind::Return) {
            continue;
        }
        by_name[slots[i].name] += slots[i].calls.load(std::memory_order_relaxed);
    }
    std::vector<Row> rows;
    rows.reserve(by_name.size());
    for (auto& [name, calls] : by_name) {
        rows.push_back({name, calls});
    }
    std::ranges::stable_sort(rows, [](const Row& a, const Row& b) { return a.calls > b.calls; });
    return rows;
}

std::vector<Bridge::Row> Bridge::UnimplementedRows() const {
    std::vector<Row> rows;
    for (u32 id = 0; id < names.size(); ++id) {
        const u64 calls = unimplemented_calls[id].load(std::memory_order_relaxed);
        if (calls != 0) {
            rows.push_back({std::string{names[id]}, calls});
        }
    }
    std::ranges::stable_sort(rows, [](const Row& a, const Row& b) { return a.calls > b.calls; });
    return rows;
}

std::string Bridge::FormatTsv(const std::vector<Row>& rows, u64 frames) {
    std::string text = "name\tcalls\tcalls_per_frame\n";
    for (const Row& row : rows) {
        const double per_frame =
            frames != 0 ? static_cast<double>(row.calls) / static_cast<double>(frames) : 0.0;
        text += fmt::format("{}\t{}\t{:.4f}\n", row.name, row.calls, per_frame);
    }
    return text;
}

void Bridge::LogReport(const std::vector<Row>& rows, u64 frames, bool everything) const {
    u64 total = 0;
    size_t called = 0;
    for (const Row& row : rows) {
        total += row.calls;
        called += row.calls != 0;
    }
    LOG_INFO(Core,
             "[nvn] census: obtained={} called={} total_calls={} frames={} build_id={}",
             rows.size(), called, total, frames, build_id);
    size_t shown = 0;
    for (const Row& row : rows) {
        if (row.calls == 0 || (!everything && shown >= 40)) {
            break;
        }
        ++shown;
        const double per_frame =
            frames != 0 ? static_cast<double>(row.calls) / static_cast<double>(frames) : 0.0;
        LOG_INFO(Core, "[nvn] census {} calls={} per_frame={:.3f}", row.name, row.calls,
                 per_frame);
    }
}

bool Bridge::WriteReports(u64 frames, bool final_report) {
    std::scoped_lock lock{report_mutex};
    const std::filesystem::path directory = ReportDirectory();
    bool ok = true;
    if (config.census) {
        const auto rows = CensusRows();
        LogReport(rows, frames, final_report);
        const auto path = directory / fmt::format("census-{:016X}.tsv", title_id);
        if (WriteFileAtomically(path, FormatTsv(rows, frames))) {
            if (!census_written) {
                census_written = true;
                LOG_INFO(Core, "[nvn] census written to {}", path.string());
            }
        } else {
            LOG_ERROR(Core, "[nvn] could not write {}", path.string());
            ok = false;
        }
    }
    if (hle_active.load(std::memory_order_acquire)) {
        const auto rows = UnimplementedRows();
        const auto path = directory / fmt::format("unimplemented-{:016X}.tsv", title_id);
        if (!WriteFileAtomically(path, FormatTsv(rows, frames))) {
            LOG_ERROR(Core, "[nvn] could not write {}", path.string());
            ok = false;
        }
        std::vector<UnsupportedRow> unsupported;
        {
            std::scoped_lock provider_lock{provider_mutex};
            if (unsupported_provider) {
                unsupported = unsupported_provider();
            }
        }
        if (!unsupported.empty()) {
            std::ranges::stable_sort(unsupported, [](const UnsupportedRow& a, const UnsupportedRow& b) {
                return a.count > b.count;
            });
            std::string text = "where\treason\tcount\tper_frame\n";
            for (const UnsupportedRow& row : unsupported) {
                const double per_frame =
                    frames != 0 ? static_cast<double>(row.count) / static_cast<double>(frames) : 0.0;
                text += fmt::format("{}\t{}\t{}\t{:.4f}\n", row.where, row.reason, row.count, per_frame);
            }
            const auto unsupported_path = directory / fmt::format("unsupported-{:016X}.tsv", title_id);
            if (!WriteFileAtomically(unsupported_path, text)) {
                LOG_ERROR(Core, "[nvn] could not write {}", unsupported_path.string());
                ok = false;
            }
        }
    }
    return ok;
}

void Bridge::ReporterLoop(std::stop_token token) {
    const auto interval = std::chrono::seconds{std::max(1u, config.report_interval_seconds)};
    while (!token.stop_requested()) {
        {
            std::unique_lock lock{stop_mutex};
            stop_cv.wait_for(lock, token, interval, [] { return false; });
        }
        if (token.stop_requested()) {
            break;
        }
        WriteReports(Core::g_nx2_present_count.load(std::memory_order_relaxed));
    }
}

void Bridge::Stop() {
    {
        std::scoped_lock lock{stop_mutex};
        if (stopped) {
            return;
        }
        stopped = true;
    }
    if (reporter.joinable()) {
        reporter.request_stop();
        reporter.join();
    }
    if (active.load(std::memory_order_acquire)) {
        WriteReports(Core::g_nx2_present_count.load(std::memory_order_relaxed), true);
    }
}

} // namespace Core::NvnHle
