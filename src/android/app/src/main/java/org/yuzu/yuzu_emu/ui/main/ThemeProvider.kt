// SPDX-License-Identifier: GPL-2.0-or-later

package org.yuzu.yuzu_emu.ui.main

interface ThemeProvider {
    /**
     * Provides theme ID by overriding an activity's 'setTheme' method and returning that result
     */
    var themeId: Int
}
