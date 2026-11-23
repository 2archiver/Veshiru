// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nvn/nvn_types.h"

namespace VN = VideoCommon::Nvn;

namespace {

// Lines of the upstream names.txt, or empty when it is not available. names.txt is never committed.
std::vector<std::string> LoadNamesFile() {
    const char* env = std::getenv("EDEN_NVN_NAMES");
    const std::string path = env != nullptr ? env : "C:\\eg\\results\\nvn\\src\\names.txt";
    std::ifstream file(path);
    std::vector<std::string> lines;
    std::string line;
    while (file && std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

} // namespace

TEST_CASE("NVN opaque object sizes", "[nvn]") {
    CHECK(sizeof(VN::NVNdevice) == 0x3000);
    CHECK(sizeof(VN::NVNdeviceBuilder) == 0x40);
    CHECK(sizeof(VN::NVNqueue) == 0x2000);
    CHECK(sizeof(VN::NVNqueueBuilder) == 0x40);
    CHECK(sizeof(VN::NVNwindow) == 384);
    CHECK(sizeof(VN::NVNwindowBuilder) == 0x40);
    CHECK(sizeof(VN::NVNtexture) == 0xC0);
    CHECK(sizeof(VN::NVNtextureView) == 0x28);
    CHECK(sizeof(VN::NVNtextureBuilder) == 0x80);
    CHECK(sizeof(VN::NVNprogram) == 0xC0);
    CHECK(sizeof(VN::NVNcommandBuffer) == 0xA0);
    CHECK(sizeof(VN::NVNbuffer) == 0x30);
    CHECK(sizeof(VN::NVNbufferBuilder) == 0x40);
    CHECK(sizeof(VN::NVNsync) == 0x40);
    CHECK(sizeof(VN::NVNmemoryPool) == 0x100);
    CHECK(sizeof(VN::NVNmemoryPoolBuilder) == 0x40);
    CHECK(sizeof(VN::NVNtexturePool) == 0x20);
    CHECK(sizeof(VN::NVNsamplerPool) == 0x20);
    CHECK(sizeof(VN::NVNsampler) == 0x30);
    CHECK(sizeof(VN::NVNsamplerBuilder) == 0x60);
    CHECK(sizeof(VN::NVNevent) == 0x40);
    CHECK(sizeof(VN::NVNeventBuilder) == 0x20);
    CHECK(sizeof(VN::NVNblendState) == 8);
    CHECK(sizeof(VN::NVNcolorState) == 4);
    CHECK(sizeof(VN::NVNchannelMaskState) == 4);
    CHECK(sizeof(VN::NVNpolygonState) == 4);
    CHECK(sizeof(VN::NVNdepthStencilState) == 8);
    CHECK(sizeof(VN::NVNvertexAttribState) == 4);
    CHECK(sizeof(VN::NVNvertexStreamState) == 8);
    CHECK(sizeof(VN::NVNmultisampleState) == 0x18);
}

TEST_CASE("NVN plain struct sizes and guest-written offsets", "[nvn]") {
    CHECK(sizeof(VN::NVNrectangle) == 16);
    CHECK(sizeof(VN::NVNbufferRange) == 16);
    CHECK(sizeof(VN::NVNcopyRegion) == 24);
    CHECK(sizeof(VN::NVNshaderData) == 16);
    CHECK(sizeof(VN::NVNcounterData) == 16);
    CHECK(sizeof(VN::NVNtextureSparseTileLayout) == 32);
    CHECK(sizeof(VN::NVNmappingRequest) == 0x28);

    CHECK(offsetof(VN::NVNbufferGuestFields, offset) == 0x18);
    CHECK(offsetof(VN::NVNbufferGuestFields, size) == 0x20);
    CHECK(VN::kBufferOffsetFieldOffset == 0x18);
    CHECK(VN::kBufferSizeFieldOffset == 0x20);
    CHECK(offsetof(VN::NVNcommandBufferGuestFields, guest_byte) == 0x10);
    CHECK(VN::kCommandBufferGuestByteOffset == 0x10);
}

TEST_CASE("NVN layout status table", "[nvn]") {
    using VN::LayoutStatus;
    // 30 opaque objects + 11 plain structs + NVNqueueErrorInfo.
    REQUIRE(VN::LayoutTable.size() == 42);
    CHECK(VN::CountLayoutStatus(LayoutStatus::Verified) == 11);
    CHECK(VN::CountLayoutStatus(LayoutStatus::Nvn1CarryOver) == 27);
    CHECK(VN::CountLayoutStatus(LayoutStatus::ContradictedByNewerSdk) == 3);
    CHECK(VN::CountLayoutStatus(LayoutStatus::Unknown) == 1);

    std::set<std::string_view> names;
    for (const VN::LayoutEntry& entry : VN::LayoutTable) {
        CHECK(names.insert(entry.name).second);
        CHECK(entry.min_known_size <= entry.size);
        if (entry.status == LayoutStatus::ContradictedByNewerSdk) {
            CHECK(entry.min_known_size < entry.size);
        }
    }

    const VN::LayoutEntry* texture = VN::FindLayout("NVNtexture");
    REQUIRE(texture != nullptr);
    CHECK(texture->size == 0xC0);
    CHECK(texture->min_known_size == 0x80);
    CHECK(texture->status == LayoutStatus::ContradictedByNewerSdk);
    const VN::LayoutEntry* view = VN::FindLayout("NVNtextureView");
    REQUIRE(view != nullptr);
    CHECK(view->min_known_size == 0x10);
    const VN::LayoutEntry* program = VN::FindLayout("NVNprogram");
    REQUIRE(program != nullptr);
    CHECK(program->min_known_size == 0x70);
    CHECK(VN::FindLayout("NoSuchObject") == nullptr);
}

TEST_CASE("NVN enum values", "[nvn]") {
    static_assert(sizeof(VN::NVNformat) == 4);
    CHECK(static_cast<s32>(VN::NVNformat::NONE) == 0);
    CHECK(static_cast<s32>(VN::NVNformat::RGBA8) == 0x25);
    CHECK(static_cast<s32>(VN::NVNformat::DEPTH24_STENCIL8) == 0x35);
    CHECK(static_cast<s32>(VN::NVNdeviceInfo::API_MAJOR_VERSION) == 0);
    CHECK(static_cast<s32>(VN::NVNdeviceInfo::EVENTS_SUPPORT_REDUCTION_OPERATIONS) == VN::kDeviceInfoLastNvn1);
    CHECK(static_cast<s32>(VN::NVNmemoryPoolFlags::VIRTUAL) == 0x200);
    CHECK(static_cast<s32>(VN::NVNdrawPrimitive::TRIANGLES) == 4);
    CHECK(static_cast<s32>(VN::NVNtextureFlags::LINEAR) == 0x10);
    CHECK(static_cast<s32>(VN::NVNtextureTarget::TARGET_2D) == 1);
    CHECK(static_cast<s32>(VN::NVNshaderStageBits::ALL_GRAPHICS_BITS) == 31);
}

TEST_CASE("NVN function table", "[nvn]") {
    REQUIRE(VN::NumFunctions == 734);

    std::set<std::string_view> names;
    size_t with_signature = 0;
    for (size_t i = 0; i < VN::FunctionTable.size(); ++i) {
        const VN::FunctionInfo& info = VN::FunctionTable[i];
        CHECK(info.id == i);
        CHECK(names.insert(info.name).second);
        with_signature += info.has_signature ? 1 : 0;
        CHECK((VN::FindSignature(info.id) != nullptr) == info.has_signature);
    }
    CHECK(with_signature == VN::SignatureTable.size());
    CHECK(with_signature == 477);

    // The bootstrap loader is the game's import, not a table entry; GetProcAddress is the first thing it
    // hands out after the device builder calls.
    CHECK_FALSE(VN::FindFunctionId("nvnBootstrapLoader").has_value());
    CHECK(VN::FindFunctionId("nvnDeviceBuilderSetDefaults") == 0u);
    CHECK(VN::FindFunctionId("nvnDeviceInitialize") == 3u);
    CHECK(VN::FindFunctionId("nvnDeviceGetProcAddress") == 6u);
    CHECK(VN::FindFunctionId("nvnDeviceGetInteger") == 7u);
    CHECK(VN::FindFunctionId("nvnWindowSetHdrDisplayMetadata") == 733u);
    CHECK(VN::FunctionName(6) == "nvnDeviceGetProcAddress");
    CHECK(VN::FunctionName(734).empty());
    CHECK(static_cast<u32>(VN::FunctionId::nvnDeviceGetProcAddress) == 6);
    CHECK(static_cast<u32>(VN::FunctionId::nvnWindowSetHdrDisplayMetadata) == 733);
}

TEST_CASE("NVN function table signatures", "[nvn]") {
    // Ids ascend, so FindSignature can bisect.
    for (size_t i = 1; i < VN::SignatureTable.size(); ++i) {
        REQUIRE(VN::SignatureTable[i - 1].id < VN::SignatureTable[i].id);
    }
    for (const VN::SignatureInfo& sig : VN::SignatureTable) {
        CHECK(VN::FunctionName(sig.id) == sig.name);
    }

    const auto signature = [](std::string_view name) {
        const std::optional<u32> id = VN::FindFunctionId(name);
        REQUIRE(id.has_value());
        const VN::SignatureInfo* sig = VN::FindSignature(*id);
        REQUIRE(sig != nullptr);
        return sig;
    };

    // Corrected nxnative header bugs.
    CHECK(signature("nvnMemoryPoolBuilderGetMemory")->return_type == "void*");
    CHECK(signature("nvnBufferBuilderGetMemoryPool")->return_type == "NVNmemoryPool*");
    CHECK(signature("nvnTextureBuilderGetMemoryPool")->return_type == "NVNmemoryPool*");
    CHECK(signature("nvnMultisampleStateGetRasterSamples")->parameters == "const NVNmultisampleState*");
    CHECK(signature("nvnDeviceInstallDebugCallback")->parameters ==
          "NVNdevice*, const PFNNVNDEBUGCALLBACKPROC, void*, NVNboolean");

    // A prototype that was fine upstream.
    CHECK(signature("nvnDeviceInitialize")->return_type == "NVNboolean");
    CHECK(signature("nvnDeviceInitialize")->parameters == "NVNdevice*, const NVNdeviceBuilder*");

    // Newer interfaces (ray tracing, ProgramBuilder, ...) are known by name only.
    const std::optional<u32> primary = VN::FindFunctionId("nvnQueueBuilderSetPrimaryQueue");
    REQUIRE(primary.has_value());
    CHECK(VN::FindSignature(*primary) == nullptr);
    CHECK_FALSE(VN::FunctionTable[*primary].has_signature);
}

TEST_CASE("NVN function table matches names.txt", "[nvn]") {
    // Only the implicit fixture may be missing: a file named by EDEN_NVN_NAMES must be read.
    const char* const env = std::getenv("EDEN_NVN_NAMES");
    const bool explicit_file = env != nullptr;
    if (explicit_file) {
        INFO("EDEN_NVN_NAMES=" << env);
        REQUIRE(std::ifstream(env).is_open());
    }
    const std::vector<std::string> lines = LoadNamesFile();
    if (lines.empty() && !explicit_file) {
        SKIP("names.txt not found (set EDEN_NVN_NAMES or place it in C:\\eg\\results\\nvn\\src)");
    }
    REQUIRE(lines.size() == VN::NumFunctions);
    for (size_t i = 0; i < lines.size(); ++i) {
        INFO("line " << i);
        REQUIRE(VN::FunctionTable[i].name == lines[i]);
    }
}
