// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Service {

/// What nx2:gsv knows about one native image (a handle from CreateNativeImage, gsv 429) that the
/// HLE NVN needs to make a pool and a texture out of it, as nn::gpu::sdk::QueryNativeImage
/// (gsv 428) describes it to the guest.
struct Nx2NativeImageInfo {
    u64 address{};                  ///< guest address of the image's first byte
    u64 size{};                     ///< bytes of its layout
    std::array<u32, 32> descriptor{}; ///< the 0x80-byte GpuOunceUserImageDescriptor
    u32 flags{};                    ///< GpuOunceUserImageFlags
};

/// Process-wide lookup of native images, filled by the nx2:gsv sessions. The HLE NVN has no
/// service session of its own, so it asks here the same question the SDK's driver would put to
/// gsv 428.
class Nx2NativeImageDirectory {
public:
    using Resolver = std::function<std::optional<Nx2NativeImageInfo>(u32 handle)>;

    static Nx2NativeImageDirectory& Get() {
        static Nx2NativeImageDirectory directory;
        return directory;
    }

    /// `owner` identifies the session so that it can take its resolver back on destruction.
    void Add(const void* owner, Resolver resolver) {
        std::scoped_lock lock{mutex};
        resolvers.emplace_back(owner, std::move(resolver));
    }
    void Remove(const void* owner) {
        std::scoped_lock lock{mutex};
        std::erase_if(resolvers, [owner](const auto& entry) { return entry.first == owner; });
    }

    /// Sessions that can be asked (log).
    [[nodiscard]] size_t SessionCount() const {
        std::scoped_lock lock{mutex};
        return resolvers.size();
    }

    [[nodiscard]] std::optional<Nx2NativeImageInfo> Resolve(u32 handle) const {
        std::scoped_lock lock{mutex};
        for (const auto& [owner, resolver] : resolvers) {
            if (auto info = resolver(handle)) {
                return info;
            }
        }
        return std::nullopt;
    }

private:
    mutable std::mutex mutex;
    std::vector<std::pair<const void*, Resolver>> resolvers;
};

} // namespace Service
