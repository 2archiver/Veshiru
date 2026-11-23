// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <QPointer>

#include "common/settings.h"
#include "core/hle/nvn/hle_running.h"
#include "veshiru/configuration/shared_widget.h"
#include "veshiru/configuration/vtx_hle_grey.h"

namespace ConfigurationShared {

struct VtxHleGrey::Tracked {
    QPointer<Widget> widget;
    Settings::VtxHle::Reason reason;
};

VtxHleGrey::VtxHleGrey(bool game_running, QObject* parent)
    : QObject{parent}, selected{Settings::values.graphics_path.GetValue()},
      vulkan{Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan},
      target_is_running_game{game_running} {}

VtxHleGrey::~VtxHleGrey() = default;

void VtxHleGrey::Track(Widget* widget, const Settings::VtxHle::Entry& entry) {
    widgets.push_back({QPointer<Widget>{widget}, entry.reason});
    Apply(widgets.back());
}

void VtxHleGrey::SetPathSelection(Settings::GraphicsPath path, bool is_vulkan) {
    selected = path;
    vulkan = is_vulkan;
    Refresh();
}

void VtxHleGrey::SetSwitch2Program(bool is_switch2_program) {
    switch2_program = is_switch2_program;
    Refresh();
}

void VtxHleGrey::SetTargetIsRunningGame(bool is_running_game) {
    target_is_running_game = is_running_game;
    Refresh();
}

void VtxHleGrey::Refresh() {
    const Settings::VtxHle::PathQuery query{
        .game_running = target_is_running_game,
        .running_hle = target_is_running_game && Core::NvnHle::HleRunning(),
        .selected = selected,
        .vulkan = vulkan,
        .switch2_program = switch2_program,
        .environment = std::getenv("EDEN_NX2_NVN_HLE"),
    };
    const bool now = Settings::VtxHle::IsActive(query);
    const bool changed = now != active;
    active = now;
    std::erase_if(widgets, [](const Tracked& tracked) { return tracked.widget.isNull(); });
    for (Tracked& tracked : widgets) {
        Apply(tracked);
    }
    if (changed) {
        emit ActiveChanged(active);
    }
}

void VtxHleGrey::Apply(Tracked& tracked) const {
    if (tracked.widget.isNull()) {
        return;
    }
    if (active) {
        tracked.widget->SetUnavailable(ReasonText(tracked.reason), LabelSuffix(tracked.reason));
    } else {
        tracked.widget->SetUnavailable(QString(), QString());
    }
}

QString VtxHleGrey::ReasonText(Settings::VtxHle::Reason reason) {
    switch (reason) {
    case Settings::VtxHle::Reason::EmulatedGpuOnly:
        return tr("Emulated GPU only (not used by VTX-HLE).\n"
                  "Switch 1 games and the Emulated GPU path still use it.");
    case Settings::VtxHle::Reason::NotYetSupported:
        return tr("Not yet supported on VTX-HLE.\n"
                  "The Emulated GPU path applies it.");
    }
    return {};
}

QString VtxHleGrey::LabelSuffix(Settings::VtxHle::Reason reason) {
    switch (reason) {
    case Settings::VtxHle::Reason::EmulatedGpuOnly:
        return tr(" (Emulated GPU only)");
    case Settings::VtxHle::Reason::NotYetSupported:
        return tr(" (not yet supported on VTX-HLE)");
    }
    return {};
}

} // namespace ConfigurationShared
