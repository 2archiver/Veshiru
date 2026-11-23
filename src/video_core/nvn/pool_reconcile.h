// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCommon::Nvn {

/// Resolves texture descriptor pool ids nothing was registered at.
///
/// The SDK's own code writes pool descriptors on the guest CPU: bulk table writes such as
/// `nvnTexturePoolWriteDescriptorTableData` carry no driver function id (the bootstrap loader
/// returns NULL and the SDK resolves the call itself), so the HLE never sees the registrations and
/// the recorder's descriptor table stays empty at those ids. Draws that sample them bind the dummy
/// (transparent) texture: whole UI layers vanish. The Recorder implements this interface over the
/// pool's guest bytes (a pool entry is a Tegra texture header naming the storage by address); the
/// replay asks it when a draw's handle names no texture.
class PoolReconciler {
public:
    virtual ~PoolReconciler() = default;

    /// The serial of the texture a guest-written descriptor at the global pool `id` names, 0 when
    /// the entry holds nothing the replay knows. A hit registers the id in the descriptor table,
    /// so later submissions resolve it from their snapshot without asking again.
    [[nodiscard]] virtual u64 ResolveTextureId(u32 id) = 0;
};

} // namespace VideoCommon::Nvn
