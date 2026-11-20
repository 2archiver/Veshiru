// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <vector>
#include <QWidget>
#include "common/common_types.h"
#include "veshiru/configuration/configuration_shared.h"

class QCheckBox;
class QLabel;

namespace Core {
class System;
}

namespace Ui {
class ConfigureGraphicsAdvanced;
}

namespace ConfigurationShared {
class Builder;
class Widget;
}

class ConfigureGraphicsAdvanced : public ConfigurationShared::Tab {
    Q_OBJECT

public:
    explicit ConfigureGraphicsAdvanced(
        const Core::System& system_, std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group,
        const ConfigurationShared::Builder& builder, QWidget* parent = nullptr);
    ~ConfigureGraphicsAdvanced() override;

    void ApplyConfiguration() override;
    void SetConfiguration() override;

    void ExposeComputeOption();

    /// Sets the default and maximum selectable memory of the selected GPU for the texture cache limit.
    void SetDeviceMemory(u64 bytes, u64 max_bytes = 0);

private:
    void Setup(const ConfigurationShared::Builder& builder);
    void changeEvent(QEvent* event) override;
    void RetranslateUI();

    std::unique_ptr<Ui::ConfigureGraphicsAdvanced> ui;

    const Core::System& system;

    std::vector<std::function<void(bool)>> apply_funcs;

    void UpdateMemoryLimit();
    void UpdateExpertVisibility();

    QWidget* checkbox_enable_compute_pipelines{};
    ConfigurationShared::Widget* memory_limit_widget{};
    QLabel* memory_limit_note{};
    QCheckBox* show_expert{};
    QLabel* expert_note{};
    std::vector<QWidget*> expert_widgets;
    u64 device_memory{};
    u64 max_device_memory{};
};
