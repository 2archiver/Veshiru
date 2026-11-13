// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.model

interface AbstractIntSetting : AbstractSetting {
    fun getInt(needsGlobal: Boolean = false): Int
    fun setInt(value: Int)
}
