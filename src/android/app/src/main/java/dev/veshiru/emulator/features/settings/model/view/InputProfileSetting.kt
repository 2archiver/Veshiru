// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.model.view

import dev.veshiru.emulator.R
import dev.veshiru.emulator.features.input.NativeInput
import dev.veshiru.emulator.utils.NativeConfig

class InputProfileSetting(private val playerIndex: Int) :
    SettingsItem(emptySetting, R.string.profile, "", 0, "") {
    override val type = TYPE_INPUT_PROFILE

    fun getCurrentProfile(): String =
        NativeConfig.getInputSettings(true)[playerIndex].profileName

    fun getProfileNames(): Array<String> = NativeInput.getInputProfileNames()

    fun isProfileNameValid(name: String): Boolean = NativeInput.isProfileNameValid(name)

    fun createProfile(name: String): Boolean = NativeInput.createProfile(name, playerIndex)

    fun deleteProfile(name: String): Boolean = NativeInput.deleteProfile(name, playerIndex)

    fun loadProfile(name: String): Boolean {
        val result = NativeInput.loadProfile(name, playerIndex)
        NativeInput.reloadInputDevices()
        return result
    }

    fun saveProfile(name: String): Boolean = NativeInput.saveProfile(name, playerIndex)
}
