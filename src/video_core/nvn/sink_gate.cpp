// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <mutex>

#include "video_core/nvn/sink_gate.h"

namespace VideoCommon::Nvn {

void SinkGate::Detach() {
    std::unique_lock lock{mutex};
    if (target == nullptr) {
        return;
    }
    // Both calls wait for a callback that is running on a sink thread: none of them runs after
    // this, and the callbacks (which point into the Recorder) can be destroyed.
    target->SetCompletionCallback({});
    target->SetFailureCallback({});
    target->SetGuestMemory(nullptr);
    target = nullptr;
}

void SinkGate::Abandon() {
    std::unique_lock lock{mutex};
    target = nullptr;
}

bool SinkGate::Attached() const {
    std::shared_lock lock{mutex};
    return target != nullptr;
}

void SinkGate::CreatePool(const PoolDesc& desc) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->CreatePool(desc);
    }
}

void SinkGate::CreateTexture(const TextureDesc& desc) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->CreateTexture(desc);
    }
}

void SinkGate::CreateSampler(const SamplerDesc& desc) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->CreateSampler(desc);
    }
}

void SinkGate::CreateProgram(const ProgramDesc& desc) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->CreateProgram(desc);
    }
}

void SinkGate::Destroy(u64 serial) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->Destroy(serial);
    }
}

void SinkGate::Submit(Submission&& submission) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->Submit(std::move(submission));
    }
}

void SinkGate::Present(const PresentRequest& request) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->Present(request);
    }
}

bool SinkGate::Wait(u64 seq, std::chrono::nanoseconds timeout) {
    std::shared_lock lock{mutex};
    return target != nullptr && target->Wait(seq, timeout);
}

void SinkGate::SetCompletionCallback(std::function<void(u64 seq)> callback) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->SetCompletionCallback(std::move(callback));
    }
}

void SinkGate::SetFailureCallback(std::function<void(std::string reason)> callback) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->SetFailureCallback(std::move(callback));
    }
}

void SinkGate::SetGuestMemory(GuestMemory* memory) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->SetGuestMemory(memory);
    }
}

void SinkGate::SetDeviceModes(u32 window_origin, u32 depth_mode) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->SetDeviceModes(window_origin, depth_mode);
    }
}

void SinkGate::SetTitleId(u64 title_id) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->SetTitleId(title_id);
    }
}

void SinkGate::CollectStats(std::vector<SinkStat>& out) const {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->CollectStats(out);
    }
}

void SinkGate::MapVirtual(u64 pool_serial, std::vector<VirtualMapping>&& mappings) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->MapVirtual(pool_serial, std::move(mappings));
    }
}

void SinkGate::NotifyCpuWrite(u64 guest_address, u64 size) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->NotifyCpuWrite(guest_address, size);
    }
}

void SinkGate::NotifyHostWrite(u64 guest_address, u64 size) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->NotifyHostWrite(guest_address, size);
    }
}

u64 SinkGate::GpuWriteMarker() {
    std::shared_lock lock{mutex};
    return target != nullptr ? target->GpuWriteMarker() : 0;
}

GpuWriteState SinkGate::PollGpuWrites(u64 marker, u64 guest_address, u64 size, u64 max_seq) {
    std::shared_lock lock{mutex};
    return target != nullptr ? target->PollGpuWrites(marker, guest_address, size, max_seq)
                             : GpuWriteState{};
}

void SinkGate::NotifyEventCpu(u64 address, EventCpuChange change, u32 value) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->NotifyEventCpu(address, change, value);
    }
}

void SinkGate::NotifyTextureWritten(u64 texture_serial) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->NotifyTextureWritten(texture_serial);
    }
}

bool SinkGate::FlushTextureToGuest(u64 texture_serial) {
    std::shared_lock lock{mutex};
    return target != nullptr && target->FlushTextureToGuest(texture_serial);
}

std::string SinkGate::DescribeState() const {
    std::shared_lock lock{mutex};
    return target != nullptr ? target->DescribeState() : std::string{};
}

std::string SinkGate::DescribeMemory() const {
    std::shared_lock lock{mutex};
    return target != nullptr ? target->DescribeMemory() : std::string{};
}

void SinkGate::NoteTextureHolders(TextureHolderReport&& report) {
    std::shared_lock lock{mutex};
    if (target != nullptr) {
        target->NoteTextureHolders(std::move(report));
    }
}

} // namespace VideoCommon::Nvn
