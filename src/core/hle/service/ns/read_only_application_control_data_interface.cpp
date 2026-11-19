// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <vector>
#include <optional>
#include <string>

#include "common/artwork_cache.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/decrypted_nca.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_transfer_memory.h"
#include "core/hle/result.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/cmif_types.h"
#include "core/hle/service/hle_ipc.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/ns/home_menu_alias.h"
#include "core/hle/service/ns/language.h"
#include "core/hle/service/ns/ns_results.h"
#include "core/hle/service/ns/ns_types.h"
#include "core/hle/service/ns/read_only_application_control_data_interface.h"
#include "core/hle/service/set/settings_server.h"
#include "core/memory.h"

namespace Service::NS {

namespace {

FileSys::PatchManager::Metadata LoadControlMetadata(Core::System& system, u64 real_app_id) {
    return FileSys::PatchManager::GetMetadataFromBaseOrUpdate(system, real_app_id);
}

FileSys::RawNACP BuildPreparedNacp(const FileSys::NACP* control_nacp, u64 requested_app_id,
                                   u64 real_app_id, bool system_menu_caller) {
    FileSys::RawNACP raw{};
    if (control_nacp != nullptr) {
        const auto bytes = control_nacp->GetRawBytes();
        const size_t copy_len = (std::min)(bytes.size(), sizeof(FileSys::RawNACP));
        std::memcpy(&raw, bytes.data(), copy_len);
    }
    if (system_menu_caller) {
        const std::string fallback_title = Common::ArtworkCache::ReadSidecarTitle(real_app_id);
        FileSys::NormalizeNacpForHomeMenu(raw, requested_app_id, fallback_title);
        if (requested_app_id != real_app_id) {
            raw.save_data_owner_id = requested_app_id;
            raw.presence_group_id = requested_app_id;
            // Clear Ounce-only platform marker so Switch 1 qlaunch treats the tile as launchable.
            raw.apparent_platform = FileSys::ApparentPlatform::NX;
        }
    }
    return raw;
}

std::vector<u8> BuildPreparedIcon(const FileSys::VirtualFile& icon_file, u64 real_app_id,
                                  bool system_menu_caller, bool grid_thumbnail) {
    std::vector<u8> raw_icon;
    if (icon_file != nullptr) {
        const size_t full_size = icon_file->GetSize();
        if (full_size > 0) {
            raw_icon.resize(full_size);
            icon_file->Read(raw_icon.data(), full_size);
        }
    }
    if (!system_menu_caller && !grid_thumbnail) {
        return raw_icon;
    }
    if (raw_icon.empty() && system_menu_caller) {
        raw_icon = Common::ArtworkCache::ReadCachedArtwork(real_app_id);
    }
    const int target_dim =
        grid_thumbnail ? FileSys::HomeMenuGridIconDimension : FileSys::HomeMenuIconDimension;
    return FileSys::BuildHomeMenuIconJpeg(real_app_id, raw_icon, target_dim);
}

} // namespace


// IAsyncValue implementation for ListApplicationTitle
// https://switchbrew.org/wiki/NS_services#ListApplicationTitle
class IAsyncValue final : public ServiceFramework<IAsyncValue> {
public:
    explicit IAsyncValue(Core::System& system_, s32 offset, s32 size)
        : ServiceFramework{system_, "IAsyncValue"}
        , service_context{system_, "IAsyncValue"}
        , data_offset{offset}
        , data_size{size}
    {
        static const FunctionInfo functions[] = {
            {0, D<&IAsyncValue::GetSize>, "GetSize"},
            {1, D<&IAsyncValue::Get>, "Get"},
            {2, D<&IAsyncValue::Cancel>, "Cancel"},
            {3, D<&IAsyncValue::GetErrorContext>, "GetErrorContext"},
        };
        RegisterHandlers(functions);
        completion_event = service_context.CreateEvent("IAsyncValue:Completion");
        completion_event->GetReadableEvent().Signal(system.Kernel());
    }

    ~IAsyncValue() override {
        service_context.CloseEvent(completion_event);
    }

    Kernel::KReadableEvent& ReadableEvent() const {
        return completion_event->GetReadableEvent();
    }

private:
    Result GetSize(Out<s64> out_data_size) {
        LOG_DEBUG(Service_NS, "called");
        *out_data_size = data_size;
        R_SUCCEED();
    }
    Result Get(OutBuffer<BufferAttr_HipcMapAlias> out_data_offset) {
        LOG_DEBUG(Service_NS, "called");
        std::memcpy(out_data_offset.data(), &data_offset, sizeof(s32));
        R_SUCCEED();
    }
    Result Cancel() {
        LOG_DEBUG(Service_NS, "called");
        R_SUCCEED();
    }
    Result GetErrorContext() {
        LOG_DEBUG(Service_NS, "called");
        R_SUCCEED();
    }
    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* completion_event{};
    s32 data_offset;
    s32 data_size;
};

IReadOnlyApplicationControlDataInterface::IReadOnlyApplicationControlDataInterface(
    Core::System& system_)
    : ServiceFramework{system_, "IReadOnlyApplicationControlDataInterface"} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IReadOnlyApplicationControlDataInterface::GetApplicationControlData>, "GetApplicationControlData"},
        {1, D<&IReadOnlyApplicationControlDataInterface::GetApplicationDesiredLanguage>, "GetApplicationDesiredLanguage"},
        {2, D<&IReadOnlyApplicationControlDataInterface::ConvertApplicationLanguageToLanguageCode>, "ConvertApplicationLanguageToLanguageCode"},
        {3, D<&IReadOnlyApplicationControlDataInterface::ConvertLanguageCodeToApplicationLanguage>, "ConvertLanguageCodeToApplicationLanguage"},
        {4, D<&IReadOnlyApplicationControlDataInterface::SelectApplicationDesiredLanguage>, "SelectApplicationDesiredLanguage"},
        {5, D<&IReadOnlyApplicationControlDataInterface::GetApplicationControlData2>, "GetApplicationControlData"},
        {10, &IReadOnlyApplicationControlDataInterface::ListApplicationIcon, "ListApplicationIcon"},
        {13, &IReadOnlyApplicationControlDataInterface::ListApplicationTitle, "ListApplicationTitle"},
        {19, D<&IReadOnlyApplicationControlDataInterface::GetApplicationControlData3>, "GetApplicationControlData"},
        {23, D<&IReadOnlyApplicationControlDataInterface::GetApplicationControlData3>, "GetApplicationControlData"}, //23.0.0+
    };
    // clang-format on

    RegisterHandlers(functions);
}

IReadOnlyApplicationControlDataInterface::~IReadOnlyApplicationControlDataInterface() = default;

Result IReadOnlyApplicationControlDataInterface::GetApplicationControlData(
    OutBuffer<BufferAttr_HipcMapAlias> out_buffer, Out<u32> out_actual_size,
    ApplicationControlSource application_control_source, u64 application_id) {
    const bool menu_caller = IsSystemAppletProcess(system);
    const u64 real_app_id = ResolveCallerApplicationId(system, application_id);
    LOG_INFO(Service_NS,
             "called with control_source={}, application_id={:016X} (resolved={:016X})",
             application_control_source, application_id, real_app_id);

    const auto control = LoadControlMetadata(system, real_app_id);
    const auto size = out_buffer.size();
    constexpr size_t nacp_size = sizeof(FileSys::RawNACP);

    if (size < nacp_size) {
        LOG_ERROR(Service_NS, "output buffer is too small! (actual={:016X}, expected_min=0x4000)",
                  size);
        R_THROW(ResultUnknown);
    }

    const FileSys::RawNACP raw_nacp =
        BuildPreparedNacp(control.first.get(), application_id, real_app_id, menu_caller);
    std::memcpy(out_buffer.data(), &raw_nacp, nacp_size);
    if (control.first == nullptr && !menu_caller) {
        LOG_WARNING(Service_NS, "missing NACP data for application_id={:016X}, defaulting to zero",
                    real_app_id);
    }

    const auto icon_data = BuildPreparedIcon(control.second, real_app_id, menu_caller, false);
    const size_t icon_area = size - nacp_size;
    if (!menu_caller && size < nacp_size + icon_data.size()) {
        LOG_ERROR(Service_NS, "output buffer is too small! (actual={:016X}, needed={:016X})", size,
                  nacp_size + icon_data.size());
        R_THROW(ResultUnknown);
    }
    const size_t icon_copy = (std::min)(icon_data.size(), icon_area);
    if (icon_copy > 0) {
        std::memcpy(out_buffer.data() + nacp_size, icon_data.data(), icon_copy);
    }
    if (icon_copy < icon_area && menu_caller) {
        std::memset(out_buffer.data() + nacp_size + icon_copy, 0, icon_area - icon_copy);
    }

    *out_actual_size = static_cast<u32>(nacp_size + icon_copy);
    R_SUCCEED();
}

Result IReadOnlyApplicationControlDataInterface::GetApplicationDesiredLanguage(
    Out<ApplicationLanguage> out_desired_language, u32 supported_languages) {
    LOG_INFO(Service_NS, "called with supported_languages={:08X}", supported_languages);

    // Get language code from settings
    const auto language_code =
        Set::GetLanguageCodeFromIndex(static_cast<s32>(Settings::values.language_index.GetValue()));

    // Convert to application language, get priority list
    const auto application_language = ConvertToApplicationLanguage(language_code);
    if (application_language == std::nullopt) {
        LOG_ERROR(Service_NS, "Could not convert application language! language_code={}", language_code);
        R_THROW(Service::NS::ResultApplicationLanguageNotFound);
    }
    const auto priority_list = GetApplicationLanguagePriorityList(*application_language);
    if (!priority_list) {
        LOG_ERROR(Service_NS, "Could not find application language priorities! application_language={}", *application_language);
        R_THROW(Service::NS::ResultApplicationLanguageNotFound);
    }

    // Try to find a valid language.
    for (const auto lang : *priority_list) {
        const auto supported_flag = GetSupportedLanguageFlag(lang);
        if (supported_languages == 0 || (supported_languages & supported_flag) == supported_flag) {
            *out_desired_language = lang;
            R_SUCCEED();
        }
    }

    LOG_ERROR(Service_NS, "Could not find a valid language! supported_languages={:08X}",
              supported_languages);
    R_THROW(Service::NS::ResultApplicationLanguageNotFound);
}

Result IReadOnlyApplicationControlDataInterface::ConvertApplicationLanguageToLanguageCode(
    Out<u64> out_language_code, ApplicationLanguage application_language) {
    const auto language_code = ConvertToLanguageCode(application_language);
    if (language_code == std::nullopt) {
        LOG_ERROR(Service_NS, "Language not found! application_language={}", application_language);
        R_THROW(Service::NS::ResultApplicationLanguageNotFound);
    }

    *out_language_code = static_cast<u64>(*language_code);
    R_SUCCEED();
}

Result IReadOnlyApplicationControlDataInterface::ConvertLanguageCodeToApplicationLanguage(
    Out<ApplicationLanguage> out_application_language, u64 language_code) {
    const auto lang = ConvertToApplicationLanguage(static_cast<Set::LanguageCode>(language_code));
    if (!lang.has_value()) {
        LOG_ERROR(Service_NS, "Application language not found! language_code={:016X}",
                  language_code);
        R_THROW(Service::NS::ResultApplicationLanguageNotFound);
    }
    *out_application_language = *lang;
    R_SUCCEED();
}

Result IReadOnlyApplicationControlDataInterface::SelectApplicationDesiredLanguage(
    Out<ApplicationLanguage> out_desired_language, u32 supported_languages) {
    R_RETURN(GetApplicationDesiredLanguage(out_desired_language, supported_languages));
}

Result IReadOnlyApplicationControlDataInterface::GetApplicationControlData2(
    OutBuffer<BufferAttr_HipcMapAlias> out_buffer, Out<u64> out_total_size,
    ApplicationControlSource application_control_source, u8 flag1, u8 flag2, u64 application_id) {
    const bool menu_caller = IsSystemAppletProcess(system);
    const u64 real_app_id = ResolveCallerApplicationId(system, application_id);
    LOG_INFO(Service_NS,
             "called with control_source={}, flags=({:02X},{:02X}), application_id={:016X} "
             "(resolved={:016X})",
             application_control_source, flag1, flag2, application_id, real_app_id);

    const auto control = LoadControlMetadata(system, real_app_id);
    const auto size = out_buffer.size();
    constexpr size_t nacp_size = sizeof(FileSys::RawNACP);

    if (size < nacp_size) {
        LOG_ERROR(Service_NS, "output buffer is too small! (actual={:016X}, expected_min={:08X})",
                  size, nacp_size);
        R_THROW(ResultUnknown);
    }

    const FileSys::RawNACP raw_nacp =
        BuildPreparedNacp(control.first.get(), application_id, real_app_id, menu_caller);
    std::memcpy(out_buffer.data(), &raw_nacp, nacp_size);

    const auto icon_area_size = size - nacp_size;
    std::vector<u8> final_icon_data;
    if (icon_area_size > 0 || control.second != nullptr || menu_caller) {
        final_icon_data =
            BuildPreparedIcon(control.second, real_app_id, menu_caller, flag1 == 1);
    }

    size_t copied_icon = 0;
    if (icon_area_size > 0) {
        copied_icon = (std::min)(final_icon_data.size(), icon_area_size);
        if (copied_icon > 0) {
            std::memcpy(out_buffer.data() + nacp_size, final_icon_data.data(), copied_icon);
        }
        if (copied_icon < icon_area_size) {
            std::memset(out_buffer.data() + nacp_size + copied_icon, 0,
                        icon_area_size - copied_icon);
        }
    }

    const u32 total_available =
        static_cast<u32>(nacp_size + (icon_area_size > 0 ? copied_icon : final_icon_data.size()));
    *out_total_size = (static_cast<u64>(total_available) << 32) | static_cast<u64>(flag1);
    R_SUCCEED();
}

void IReadOnlyApplicationControlDataInterface::ListApplicationIcon(HLERequestContext& ctx) {
    const auto app_ids_buffer = ctx.ReadBuffer();
    const u64 app_count = app_ids_buffer.size() / sizeof(u64);
    std::vector<u64> application_ids(static_cast<size_t>(app_count));
    if (app_count > 0) {
        std::memcpy(application_ids.data(), app_ids_buffer.data(),
                    application_ids.size() * sizeof(u64));
    }

    const bool menu_caller = IsSystemAppletProcess(system);
    auto t_mem_obj = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(0));
    auto* t_mem = t_mem_obj.GetPointerUnsafe();
    size_t out_length = 0;
    if (t_mem != nullptr && t_mem->GetOwner() != nullptr && app_count > 0) {
        auto& memory = t_mem->GetOwner()->GetMemory();
        const auto t_mem_address = t_mem->GetSourceAddress();
        const size_t t_mem_size = t_mem->GetSize();
        const size_t header_bytes = static_cast<size_t>((1 + app_count) * sizeof(u64));

        if (header_bytes <= t_mem_size) {
            const size_t remaining_budget = t_mem_size - header_bytes;
            const bool compact_icons =
                remaining_budget / static_cast<size_t>(app_count) < 0x18000U;

            std::vector<std::vector<u8>> icons(application_ids.size());
            size_t payload_used = 0;
            for (size_t i = 0; i < application_ids.size(); ++i) {
                const u64 real_id = ResolveCallerApplicationId(system, application_ids[i]);
                const auto control = LoadControlMetadata(system, real_id);
                auto icon =
                    BuildPreparedIcon(control.second, real_id, menu_caller, compact_icons);
                if (!icon.empty() && payload_used + icon.size() > remaining_budget &&
                    !compact_icons) {
                    icon = BuildPreparedIcon(control.second, real_id, menu_caller, true);
                }
                if (payload_used + icon.size() <= remaining_budget) {
                    payload_used += icon.size();
                    icons[i] = std::move(icon);
                }
            }

            memory.WriteBlock(t_mem_address + out_length, &app_count, sizeof(u64));
            out_length += sizeof(u64);
            for (const auto& icon : icons) {
                const u64 icon_size = icon.size();
                memory.WriteBlock(t_mem_address + out_length, &icon_size, sizeof(u64));
                out_length += sizeof(u64);
            }
            for (const auto& icon : icons) {
                if (!icon.empty()) {
                    memory.WriteBlock(t_mem_address + out_length, icon.data(), icon.size());
                    out_length += icon.size();
                }
            }
        } else {
            LOG_WARNING(Service_NS,
                        "ListApplicationIcon transfer memory too small (size={}, count={})",
                        t_mem_size, app_count);
        }
    }
    auto async_value = std::make_shared<IAsyncValue>(system, 0, static_cast<s32>(out_length));
    IPC::ResponseBuilder rb{ctx, 2, 1, 1};
    rb.Push(ResultSuccess);
    rb.PushCopyObjects(ctx, async_value->ReadableEvent());
    rb.PushIpcInterface(ctx, std::move(async_value));
}

void IReadOnlyApplicationControlDataInterface::ListApplicationTitle(HLERequestContext& ctx) {
    const auto app_ids_buffer = ctx.ReadBuffer();
    const size_t app_count = app_ids_buffer.size() / sizeof(u64);

    std::vector<u64> application_ids(app_count);
    if (app_count > 0) {
        std::memcpy(application_ids.data(), app_ids_buffer.data(), app_count * sizeof(u64));
    }

    const bool menu_caller = IsSystemAppletProcess(system);
    auto t_mem_obj = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(0));
    auto* t_mem = t_mem_obj.GetPointerUnsafe();
    constexpr size_t title_entry_size = sizeof(FileSys::LanguageEntry);
    constexpr s32 data_offset = 0;
    size_t written_bytes = 0;
    if (t_mem != nullptr && app_count > 0) {
        auto& memory = t_mem->GetOwner() != nullptr ? t_mem->GetOwner()->GetMemory()
                                                    : system.ApplicationMemory();
        const auto t_mem_address = t_mem->GetSourceAddress();
        const size_t max_entries = (std::min)(app_count, t_mem->GetSize() / title_entry_size);
        const auto language_code = Set::GetLanguageCodeFromIndex(
            static_cast<s32>(Settings::values.language_index.GetValue()));
        const auto app_lang = ConvertToApplicationLanguage(language_code)
                                  .value_or(ApplicationLanguage::AmericanEnglish);
        const size_t preferred_slot =
            (std::min)(static_cast<size_t>(app_lang), size_t{15});

        for (size_t i = 0; i < max_entries; ++i) {
            const u64 requested_id = application_ids[i];
            const u64 real_id = ResolveCallerApplicationId(system, requested_id);
            const auto control = LoadControlMetadata(system, real_id);
            FileSys::LanguageEntry entry{};
            if (control.first != nullptr &&
                control.first->raw.titles_data_format == FileSys::TitleDataFormat::Compressed) {
                entry = control.first->GetLanguageEntry();
            } else if (menu_caller) {
                const FileSys::RawNACP prepared =
                    BuildPreparedNacp(control.first.get(), requested_id, real_id, true);
                entry = prepared.language_entries.language_entries[preferred_slot];
            } else if (control.first != nullptr) {
                entry = control.first->GetLanguageEntry();
            }
            const size_t offset = i * title_entry_size;
            memory.WriteBlock(t_mem_address + offset, &entry, title_entry_size);
            written_bytes += title_entry_size;
        }
    }
    auto async_value =
        std::make_shared<IAsyncValue>(system, data_offset, static_cast<s32>(written_bytes));
    IPC::ResponseBuilder rb{ctx, 2, 1, 1};
    rb.Push(ResultSuccess);
    rb.PushCopyObjects(ctx, async_value->ReadableEvent());
    rb.PushIpcInterface(ctx, std::move(async_value));
}

Result IReadOnlyApplicationControlDataInterface::GetApplicationControlData3(
    OutBuffer<BufferAttr_HipcMapAlias> out_buffer, Out<u32> out_flags_a, Out<u32> out_flags_b,
    Out<u32> out_actual_size, ApplicationControlSource application_control_source, u8 flag1,
    u8 flag2, u64 application_id) {
    const bool menu_caller = IsSystemAppletProcess(system);
    const u64 real_app_id = ResolveCallerApplicationId(system, application_id);
    LOG_INFO(Service_NS,
             "called with control_source={}, flags=({:02X},{:02X}), application_id={:016X} "
             "(resolved={:016X})",
             application_control_source, flag1, flag2, application_id, real_app_id);

    const auto control = LoadControlMetadata(system, real_app_id);
    const auto size = out_buffer.size();
    constexpr size_t nacp_size = sizeof(FileSys::RawNACP);

    if (size < nacp_size) {
        LOG_ERROR(Service_NS, "output buffer is too small! (actual={:016X}, expected_min={:08X})",
                  size, nacp_size);
        R_THROW(ResultUnknown);
    }

    const FileSys::RawNACP raw_nacp =
        BuildPreparedNacp(control.first.get(), application_id, real_app_id, menu_caller);
    std::memcpy(out_buffer.data(), &raw_nacp, nacp_size);

    const auto icon_area_size = size - nacp_size;
    std::vector<u8> final_icon_data;
    if (icon_area_size > 0 || control.second != nullptr || menu_caller) {
        final_icon_data =
            BuildPreparedIcon(control.second, real_app_id, menu_caller, flag1 == 1);
    }

    size_t copied_icon = 0;
    if (icon_area_size > 0) {
        copied_icon = (std::min)(final_icon_data.size(), icon_area_size);
        if (copied_icon > 0) {
            std::memcpy(out_buffer.data() + nacp_size, final_icon_data.data(), copied_icon);
        }
        if (copied_icon < icon_area_size) {
            std::memset(out_buffer.data() + nacp_size + copied_icon, 0,
                        icon_area_size - copied_icon);
        }
    }

    const u32 actual_total_size =
        static_cast<u32>(nacp_size + (icon_area_size > 0 ? copied_icon : final_icon_data.size()));
    *out_flags_a = 0x10001;
    *out_flags_b = (flag1 == 1) ? 0x10001 : 0;
    *out_actual_size = actual_total_size;

    R_SUCCEED();
}

} // namespace Service::NS
