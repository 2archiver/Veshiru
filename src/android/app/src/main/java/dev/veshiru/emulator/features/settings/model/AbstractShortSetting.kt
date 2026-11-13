// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.model

interface AbstractShortSetting : AbstractSetting {
    fun getShort(needsGlobal: Boolean = false): Short
    fun setShort(value: Short)
}
