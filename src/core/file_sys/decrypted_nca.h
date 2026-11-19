// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

struct RawNACP;

/// Maximum JPEG icon byte size accepted by Switch HOME Menu (qlaunch).
inline constexpr std::size_t HomeMenuMaxIconBytes = 0x20000;

/// Standard Switch HOME Menu square JPEG icon dimension in pixels.
inline constexpr int HomeMenuIconDimension = 256;

/// Compact grid thumbnail JPEG icon dimension used by GetApplicationControlData flag1=1.
inline constexpr int HomeMenuGridIconDimension = 174;

/// A decrypted view of an NCA, read on demand with the console keys: plaintext header with the
/// magic DNCA, every FS encryption type set to None and the FS header hashes recomputed, and
/// plaintext sections. The view has the size of the NCA. An NCA that is already decrypted is
/// returned as is. On failure, returns nullptr and sets `error`.
VirtualFile OpenDecryptedNca(VirtualFile nca, std::string& error);

/// A decrypted NSP (DNSP) view of an NSP or XCI: a PFS0 of the package's files, or of the XCI
/// secure partition, with every NCA replaced by its decrypted view. Nothing is written to disk.
VirtualFile OpenDecryptedPackage(VirtualFile package, std::string& error);

/// A decrypted gamecard image (DXCI) view of an XCI: the same bytes, with the magic DXCI and
/// every NCA outside the update partition replaced by its decrypted view. The update partition
/// (the console firmware) is kept as it is. Nothing is written to disk.
VirtualFile OpenDecryptedGamecard(VirtualFile image, std::string& error);

/// A read-only PFS0 (NSP layout) holding `files` in order, without copying their bytes.
VirtualFile BuildPartitionFile(std::string name, const std::vector<VirtualFile>& files);

/// A decrypted add-on content NCA for `title_id` whose one section is `romfs`, without a hash
/// layer, for add-ons kept as an extracted tree. Nothing is copied.
VirtualFile BuildAddOnContentNca(u64 title_id, VirtualFile romfs);

/// A decrypted control NCA for `title_id` whose one section is `romfs` (holding control.nacp and
/// icon_*.dat), without a hash layer, for loose/packed titles (.nup, .nx, .nro) shown in qlaunch.
VirtualFile BuildControlNca(u64 title_id, VirtualFile romfs);

/// Normalizes a RawNACP copy for HOME Menu display: ensures every language slot has a non-empty
/// title/publisher, populates version_string/supported_languages, and sets owner/presence IDs.
void NormalizeNacpForHomeMenu(RawNACP& nacp, u64 display_title_id,
                              std::string_view fallback_title = {},
                              std::string_view fallback_publisher = {},
                              std::string_view fallback_version = {});

/// Returns a valid baseline (SOF0) square JPEG of `dimension x dimension` pixels capped at
/// HomeMenuMaxIconBytes. Reuses an already-compliant baseline JPEG directly, transcodes/resizes
/// PNG or non-conforming JPEG icons, and synthesizes a clean fallback tile when `raw_icon` is
/// empty or undecodable.
std::vector<u8> BuildHomeMenuIconJpeg(u64 title_id, std::span<const u8> raw_icon,
                                      int dimension = HomeMenuIconDimension);

/// Builds a complete decrypted Control NCA from a RawNACP and optional raw JPEG/PNG icon bytes.
VirtualFile BuildControlNcaFromMetadata(u64 title_id, const RawNACP& nacp,
                                        std::span<const u8> raw_icon,
                                        std::string_view fallback_title = {},
                                        std::string_view fallback_publisher = {},
                                        std::string_view fallback_version = {});

} // namespace FileSys
