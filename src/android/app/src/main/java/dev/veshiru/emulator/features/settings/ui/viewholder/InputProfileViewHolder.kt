// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.ui.viewholder

import android.view.View
import dev.veshiru.emulator.databinding.ListItemSettingBinding
import dev.veshiru.emulator.features.settings.model.view.InputProfileSetting
import dev.veshiru.emulator.features.settings.model.view.SettingsItem
import dev.veshiru.emulator.features.settings.ui.SettingsAdapter
import dev.veshiru.emulator.R
import dev.veshiru.emulator.utils.ViewUtils.setVisible

class InputProfileViewHolder(val binding: ListItemSettingBinding, adapter: SettingsAdapter) :
    SettingViewHolder(binding.root, adapter) {
    private lateinit var setting: InputProfileSetting

    override fun bind(item: SettingsItem) {
        setting = item as InputProfileSetting
        binding.textSettingName.text = setting.title
        binding.textSettingValue.text =
            setting.getCurrentProfile().ifEmpty { binding.root.context.getString(R.string.not_set) }

        binding.textSettingDescription.setVisible(false)
        binding.buttonClear.setVisible(false)
        binding.icon.setVisible(false)
        binding.buttonClear.setVisible(false)
    }

    override fun onClick(clicked: View) =
        adapter.onInputProfileClick(setting, bindingAdapterPosition)

    override fun onLongClick(clicked: View): Boolean = false
}
