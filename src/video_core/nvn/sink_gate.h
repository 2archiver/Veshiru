// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <shared_mutex>

#include "video_core/nvn/sink.h"

namespace VideoCommon::Nvn {

/// A Sink that forwards to another sink until it is detached, and ignores every call afterwards.
///
/// The Recorder lives in the HLE runtime, the sink in the renderer; they do not die together. At
/// shutdown the renderer is destroyed while guest threads can still be inside an NVN call, and the
/// Recorder (whose destructor calls the sink) is destroyed later, once those threads are gone. The
/// Recorder is built over a gate: Detach() is called while the renderer is alive and, from then
/// on, nothing reaches the sink any more, whoever still holds the Recorder.
///
/// Calls take a shared lock, Detach an exclusive one: Detach returns after every call that was
/// inside the sink has returned (a blocking Wait / FlushTextureToGuest included, which are
/// bounded), and the sink may be destroyed right after it.
class SinkGate final : public Sink {
public:
    explicit SinkGate(Sink& target_) : target{&target_} {}
    ~SinkGate() override = default;

    /// Disconnects from the sink: clears the completion/failure callbacks (waiting for one that is
    /// running) and the guest memory pointer the Recorder gave it. Idempotent.
    void Detach();
    /// Disconnects without touching the sink, which may already be gone.
    void Abandon();
    [[nodiscard]] bool Attached() const;

    void CreatePool(const PoolDesc& desc) override;
    void CreateTexture(const TextureDesc& desc) override;
    void CreateSampler(const SamplerDesc& desc) override;
    void CreateProgram(const ProgramDesc& desc) override;
    void Destroy(u64 serial) override;
    void Submit(Submission&& submission) override;
    void Present(const PresentRequest& request) override;
    bool Wait(u64 seq, std::chrono::nanoseconds timeout) override;
    void SetCompletionCallback(std::function<void(u64 seq)> callback) override;
    void SetFailureCallback(std::function<void(std::string reason)> callback) override;
    void SetGuestMemory(GuestMemory* memory) override;
    void SetDeviceModes(u32 window_origin, u32 depth_mode) override;
    void SetTitleId(u64 title_id) override;
    void CollectStats(std::vector<SinkStat>& out) const override;
    void MapVirtual(u64 pool_serial, std::vector<VirtualMapping>&& mappings) override;
    void NotifyCpuWrite(u64 guest_address, u64 size) override;
    void NotifyHostWrite(u64 guest_address, u64 size) override;
    [[nodiscard]] u64 GpuWriteMarker() override;
    [[nodiscard]] GpuWriteState PollGpuWrites(u64 marker, u64 guest_address, u64 size,
                                              u64 max_seq) override;
    void NotifyEventCpu(u64 address, EventCpuChange change, u32 value) override;
    void NotifyTextureWritten(u64 texture_serial) override;
    bool FlushTextureToGuest(u64 texture_serial) override;
    [[nodiscard]] std::string DescribeState() const override;
    [[nodiscard]] std::string DescribeMemory() const override;
    void NoteTextureHolders(TextureHolderReport&& report) override;

private:
    mutable std::shared_mutex mutex;
    Sink* target;
};

} // namespace VideoCommon::Nvn
