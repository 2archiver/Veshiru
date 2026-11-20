// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <map>
#include <ranges>
#include <vector>
#include <QCheckBox>
#include <QLabel>
#include <QSpinBox>
#include <qnamespace.h>
#include "common/settings.h"
#include "core/core.h"
#include "qt_common/config/shared_translation.h"
#include "ui_configure_graphics_advanced.h"
#include "video_core/texture_cache/memory_limits.h"
#include "veshiru/configuration/configuration_shared.h"
#include "veshiru/configuration/configure_graphics_advanced.h"
#include "veshiru/configuration/shared_widget.h"

namespace {

/// Settings Veshiru's automatic behaviour covers. They stay supported (config file, per-game
/// overrides) but leave the normal interface: memory sizing follows the GPU's driver budget on its
/// own, and the reserve/limit pair only existed because 1.x stacked margins that fought each other.
/// Showing them invited users to cap the cache at 2 GiB and stutter.
bool IsExpertOnly(const Settings::BasicSetting& setting) {
    return setting.Id() == Settings::values.texture_cache_memory_limit.Id() ||
           setting.Id() == Settings::values.vram_reserve_mib.Id() ||
           setting.Id() == Settings::values.vram_usage_mode.Id();
}

} // namespace

ConfigureGraphicsAdvanced::ConfigureGraphicsAdvanced(
    const Core::System& system_, std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
    const ConfigurationShared::Builder& builder, QWidget* parent)
    : Tab(group_, parent), ui{std::make_unique<Ui::ConfigureGraphicsAdvanced>()}, system{system_} {

    ui->setupUi(this);

    Setup(builder);

    SetConfiguration();

    checkbox_enable_compute_pipelines->setVisible(false);
}

ConfigureGraphicsAdvanced::~ConfigureGraphicsAdvanced() = default;

void ConfigureGraphicsAdvanced::SetConfiguration() {}

void ConfigureGraphicsAdvanced::Setup(const ConfigurationShared::Builder& builder) {
    auto& normal_layout = *ui->normal_target->layout();

    // The memory controls go behind "Show expert options" instead of into the normal list.
    show_expert = new QCheckBox(tr("Show memory expert options"), this);
    show_expert->setToolTip(
        tr("Veshiru sizes video memory automatically for this GPU. Change these only when testing "
           "memory behaviour."));

    // A map will sort the data for us
    std::map<u32, QWidget*> normal_hold{};

    for (auto setting :
         Settings::values.linkage.by_category[Settings::Category::RendererAdvanced]) {
        const bool is_expert = IsExpertOnly(*setting);
        const bool is_memory_limit =
            setting->Id() == Settings::values.texture_cache_memory_limit.Id();
        const bool is_vram_reserve = setting->Id() == Settings::values.vram_reserve_mib.Id();
        ConfigurationShared::Widget* widget =
            (is_memory_limit || is_vram_reserve)
                ? builder.BuildWidget(setting, apply_funcs, ConfigurationShared::RequestType::Default,
                                      true, 1.0f, nullptr, tr(" MiB"))
                : builder.BuildWidget(setting, apply_funcs);

        if (widget == nullptr) {
            continue;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            continue;
        }

        const auto id = setting->Id();

        if (is_vram_reserve && widget->spinbox != nullptr) {
            widget->spinbox->setSpecialValueText(tr("Auto"));
            widget->spinbox->setSingleStep(64);
        }

        if (is_memory_limit && widget->spinbox != nullptr) {
            memory_limit_widget = widget;
            memory_limit_widget->spinbox->setSpecialValueText(tr("Auto"));
            memory_limit_widget->spinbox->setSingleStep(256);
            memory_limit_note = new QLabel(this);
            memory_limit_note->setWordWrap(true);
            memory_limit_note->setTextFormat(Qt::RichText);
            connect(memory_limit_widget->spinbox, qOverload<int>(&QSpinBox::valueChanged), this,
                    [this] { UpdateMemoryLimit(); });
        }

        // Keep track of enable_compute_pipelines so we can display it when needed
        if (setting->Id() == Settings::values.enable_compute_pipelines.Id()) {
            checkbox_enable_compute_pipelines = widget;
        }

        if (is_expert) {
            expert_widgets.push_back(widget);
        } else {
            normal_hold.emplace(id, widget);
        }
    }

    for (const auto& [id, widget] : normal_hold) {
        normal_layout.addWidget(widget);
    }
    normal_layout.addWidget(show_expert);

    expert_note = new QLabel(this);
    expert_note->setWordWrap(true);
    expert_note->setTextFormat(Qt::RichText);
    expert_note->setText(
        tr("<span style=\"color:#8b949e\">Auto follows this GPU's driver budget and keeps about "
           "387 MiB free on a 4 GB card, or 960 MiB on 8 GB. Lowering either value makes Veshiru "
           "reload textures and stutter.</span>"));
    normal_layout.addWidget(expert_note);

    for (QWidget* const widget : expert_widgets) {
        normal_layout.addWidget(widget);
        if (widget == memory_limit_widget) {
            normal_layout.addWidget(memory_limit_note);
        }
    }

    // A per-game or global override of an expert setting is someone's deliberate choice: show the
    // controls rather than hide the override.
    const bool any_expert_overridden = std::ranges::any_of(
        Settings::values.linkage.by_category[Settings::Category::RendererAdvanced],
        [](const Settings::BasicSetting* setting) {
            return IsExpertOnly(*setting) && !setting->UsingGlobal();
        });
    show_expert->setChecked(any_expert_overridden);
    connect(show_expert, &QCheckBox::toggled, this, [this] { UpdateExpertVisibility(); });

    UpdateExpertVisibility();
    UpdateMemoryLimit();
}

void ConfigureGraphicsAdvanced::UpdateExpertVisibility() {
    const bool show = show_expert != nullptr && show_expert->isChecked();
    for (QWidget* const widget : expert_widgets) {
        widget->setVisible(show);
    }
    if (expert_note != nullptr) {
        expert_note->setVisible(show);
    }
    if (memory_limit_note != nullptr) {
        memory_limit_note->setVisible(show && !memory_limit_note->text().isEmpty());
    }
}

void ConfigureGraphicsAdvanced::SetDeviceMemory(u64 bytes, u64 max_bytes) {
    device_memory = bytes;
    max_device_memory = max_bytes != 0 ? (std::max)(bytes, max_bytes) : bytes;
    UpdateMemoryLimit();
}

void ConfigureGraphicsAdvanced::UpdateMemoryLimit() {
    if (memory_limit_widget == nullptr) {
        return;
    }
    QSpinBox* const spinbox = memory_limit_widget->spinbox;
    if (device_memory == 0) {
        spinbox->setMaximum(1048576);
        memory_limit_note->hide();
        return;
    }
    const u64 ceiling = (std::max)(device_memory, max_device_memory);
    const int device_mib = static_cast<int>((std::min<u64>)(device_memory >> 20, 1048576));
    const int max_mib = static_cast<int>((std::min<u64>)(ceiling >> 20, 1048576));
    const int recommended_mib =
        static_cast<int>(VideoCommon::TextureCacheMemory::RecommendedLimit(device_memory) >> 20);
    // Keep any higher saved value intact, and let integrated/unified GPUs select up to their full
    // unified memory ceiling rather than locking the spinbox to a small UMA carveout.
    spinbox->setMaximum((std::max)(max_mib, spinbox->value()));

    const int value = spinbox->value();
    const QString below =
        tr("<span style=\"color:#d29922\">&#9888;</span> Below the recommended %1 MiB for this "
           "GPU: textures may be removed and loaded again, which can cause stutter.")
            .arg(recommended_mib);
    const QString auto_note =
        max_mib > device_mib
            ? tr("Auto uses %1 MiB (three quarters of this GPU's %2 MiB budget; up to %3 MiB "
                 "unified memory selectable).")
                  .arg(recommended_mib)
                  .arg(device_mib)
                  .arg(max_mib)
            : tr("Auto uses %1 MiB (three quarters of this GPU's %2 MiB).")
                  .arg(recommended_mib)
                  .arg(device_mib);
    const QString low_note = tr("Values under 1024 MiB are raised to 1024 MiB.");
    if (value == 0) {
        memory_limit_note->setText(auto_note);
    } else if (value < recommended_mib) {
        memory_limit_note->setText(value < 1024 ? below + QStringLiteral(" ") + low_note : below);
    } else {
        memory_limit_note->setText(QString{});
    }
    memory_limit_note->setVisible(show_expert != nullptr && show_expert->isChecked() &&
                                 !memory_limit_note->text().isEmpty());
}

void ConfigureGraphicsAdvanced::ApplyConfiguration() {
    const bool is_powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(is_powered_on);
    }
}

void ConfigureGraphicsAdvanced::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureGraphicsAdvanced::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureGraphicsAdvanced::ExposeComputeOption() {
    checkbox_enable_compute_pipelines->setVisible(true);
}
