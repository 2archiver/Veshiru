// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>

namespace VideoCore::GameFrameSize {

enum class Stage : std::uint32_t {
    Game,
    Dlss,
    Dlaa,
    Fsr1,
    Bilinear,
};

/// The sizes the NVN HLE path sees a game render and present at, for the frontend's status bar.
/// The game creates its render targets at its own sizes; the Resolution setting scales the ones it
/// draws into on top of that (host-side: the game never sees it), so a scale factor alone says
/// nothing about the picture: the sizes are the game's, and `scale_percent` is the setting.
struct Sizes {
    /// The input of the game's own upscale stage (its render resolution, in the pixels it was
    /// evaluated at: scaled when the Resolution setting scaled it) in the last frame, or 0x0 when
    /// the game did not upscale that frame.
    std::uint32_t render_width{};
    std::uint32_t render_height{};
    /// The texture the game presented last (its own size); 0x0 before the first present.
    std::uint32_t frame_width{};
    std::uint32_t frame_height{};
    /// The Resolution setting in percent (100 = 1X; 200 = 2X; 50 = 0.5X).
    std::uint32_t scale_percent{100};
    /// The host stage that actually wrote the upscaled frame, or Game when the game's own stage
    /// ran (or no upscale ran).
    Stage stage{Stage::Game};
};

/// The last successful Streamline evaluation on either GPU path. Mode and preset retain the
/// sl::DLSSMode / sl::DLSSPreset numeric values so video_core does not expose Streamline headers
/// to the frontend.
struct DlssInfo {
    std::uint32_t input_width{};
    std::uint32_t input_height{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
    std::uint32_t mode{};
    std::uint32_t preset{};
};

namespace Detail {
// NVN textures are at most 16384 texels wide and high: 16 bits per field.
inline std::atomic<std::uint64_t> packed{};
inline std::atomic<std::uint32_t> scale_percent{100};
inline std::atomic<std::uint32_t> stage{};
inline std::atomic<std::uint64_t> dlss_packed{};
inline std::atomic<std::uint64_t> dlss_options{};

[[nodiscard]] constexpr std::uint64_t Field(std::uint32_t value) noexcept {
    return value > 0xFFFF ? 0xFFFF : value;
}
} // namespace Detail

/// Replay thread, at every present.
inline void Publish(const Sizes& sizes) noexcept {
    using Detail::Field;
    Detail::packed.store(Field(sizes.render_width) | Field(sizes.render_height) << 16 |
                             Field(sizes.frame_width) << 32 | Field(sizes.frame_height) << 48,
                         std::memory_order_relaxed);
    Detail::scale_percent.store(sizes.scale_percent, std::memory_order_relaxed);
    Detail::stage.store(static_cast<std::uint32_t>(sizes.stage), std::memory_order_relaxed);
    if (sizes.stage != Stage::Dlss && sizes.stage != Stage::Dlaa) {
        Detail::dlss_packed.store(0, std::memory_order_relaxed);
        Detail::dlss_options.store(0, std::memory_order_relaxed);
    }
}

/// Called only after Streamline accepts the evaluation, on either GPU path.
inline void PublishDlss(const DlssInfo& info) noexcept {
    using Detail::Field;
    Detail::dlss_options.store(static_cast<std::uint64_t>(info.mode) |
                                   static_cast<std::uint64_t>(info.preset) << 32,
                               std::memory_order_relaxed);
    Detail::dlss_packed.store(Field(info.input_width) | Field(info.input_height) << 16 |
                                  Field(info.output_width) << 32 | Field(info.output_height) << 48,
                              std::memory_order_relaxed);
}

/// A failed evaluation falls back to another stage; do not leave the previous DLSS run visible.
inline void ClearDlss() noexcept {
    Detail::dlss_packed.store(0, std::memory_order_relaxed);
    Detail::dlss_options.store(0, std::memory_order_relaxed);
}

/// Forgets the previous session's sizes (the replay context starts).
inline void Reset() noexcept {
    Detail::packed.store(0, std::memory_order_relaxed);
    Detail::scale_percent.store(100, std::memory_order_relaxed);
    Detail::stage.store(0, std::memory_order_relaxed);
    ClearDlss();
}

[[nodiscard]] inline DlssInfo ReadDlss() noexcept {
    const std::uint64_t value = Detail::dlss_packed.load(std::memory_order_relaxed);
    const std::uint64_t options = Detail::dlss_options.load(std::memory_order_relaxed);
    const auto field = [value](int shift) {
        return static_cast<std::uint32_t>((value >> shift) & 0xFFFF);
    };
    return {field(0), field(16), field(32), field(48), static_cast<std::uint32_t>(options),
            static_cast<std::uint32_t>(options >> 32)};
}

[[nodiscard]] inline Sizes Read() noexcept {
    const std::uint64_t value = Detail::packed.load(std::memory_order_relaxed);
    const auto field = [value](int shift) { return static_cast<std::uint32_t>((value >> shift) & 0xFFFF); };
    return {field(0), field(16), field(32), field(48),
            Detail::scale_percent.load(std::memory_order_relaxed),
            static_cast<Stage>(Detail::stage.load(std::memory_order_relaxed))};
}

} // namespace VideoCore::GameFrameSize
