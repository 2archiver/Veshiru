// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.model

interface AbstractByteSetting : AbstractSetting {
    fun getByte(needsGlobal: Boolean = false): Byte
    fun setByte(value: Byte)
}
