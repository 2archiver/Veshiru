// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.settings.utils

import android.net.Uri
import dev.veshiru.emulator.model.Game
import java.io.*
import dev.veshiru.emulator.utils.DirectoryInitialization
import dev.veshiru.emulator.utils.FileUtil
import dev.veshiru.emulator.utils.NativeConfig

/**
 * Contains static methods for interacting with .ini files in which settings are stored.
 */
object SettingsFile {
    const val FILE_NAME_CONFIG = "config.ini"

    fun getSettingsFile(fileName: String): File =
        File(DirectoryInitialization.userDirectory + "/config/" + fileName)

    fun getCustomSettingsFile(game: Game): File =
        File(DirectoryInitialization.userDirectory + "/config/custom/" + game.settingsName + ".ini")

    fun loadCustomConfig(game: Game) {
        val fileName = FileUtil.getFilename(Uri.parse(game.path))
        NativeConfig.initializePerGameConfig(game.programId, fileName)
    }
}
