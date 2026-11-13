// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.model

interface SelectableItem {
    var selected: Boolean
    fun onSelectionStateChanged(selected: Boolean)
}
