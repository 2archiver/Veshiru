// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.model

interface AbstractLongSetting : AbstractSetting {
    fun getLong(needsGlobal: Boolean = false): Long
    fun setLong(value: Long)
}
