// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.fragments

import android.app.Dialog
import android.os.Bundle
import androidx.fragment.app.DialogFragment
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import dev.veshiru.emulator.R
import dev.veshiru.emulator.features.settings.ui.SettingsActivity

class ResetSettingsDialogFragment : DialogFragment() {
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val settingsActivity = requireActivity() as SettingsActivity

        return MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.reset_all_settings)
            .setMessage(R.string.reset_all_settings_description)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                settingsActivity.onSettingsReset()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    companion object {
        const val TAG = "ResetSettingsDialogFragment"
    }
}
