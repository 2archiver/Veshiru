// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/diag_image.h"

#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/stb.h"

namespace Common::DiagRequest {

namespace {
void AppendBytes(void* context, void* data, int size) {
    auto* out = static_cast<std::vector<u8>*>(context);
    const u8* bytes = static_cast<const u8*>(data);
    out->insert(out->end(), bytes, bytes + size);
}
} // namespace

bool WritePng(const std::filesystem::path& path, u32 width, u32 height, std::span<const u8> rgba) {
    if (width == 0 || height == 0 || rgba.size() < static_cast<size_t>(width) * height * 4) {
        LOG_ERROR(Core, "[diag] cannot write {}: bad image {}x{} ({} bytes)", path.string(), width,
                  height, rgba.size());
        return false;
    }
    std::vector<u8> png;
    if (!stbi_write_png_to_func(AppendBytes, &png, static_cast<int>(width), static_cast<int>(height),
                                STBI_rgb_alpha, rgba.data(), static_cast<int>(width * 4))) {
        LOG_ERROR(Core, "[diag] PNG encode of {} failed", path.string());
        return false;
    }
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    out.close();
    if (!out) {
        LOG_ERROR(Core, "[diag] cannot write {}", path.string());
        return false;
    }
    return true;
}

bool WriteTextFile(const std::filesystem::path& path, std::string_view text) {
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) {
        LOG_ERROR(Core, "[diag] cannot write {}", path.string());
        return false;
    }
    return true;
}

std::string JsonEscape(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (const char c : text) {
        switch (c) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                result += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                result += c;
            }
        }
    }
    return result;
}

} // namespace Common::DiagRequest
