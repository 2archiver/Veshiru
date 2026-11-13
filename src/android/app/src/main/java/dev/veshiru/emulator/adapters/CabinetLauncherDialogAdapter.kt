// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.res.ResourcesCompat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import dev.veshiru.emulator.HomeNavigationDirections
import dev.veshiru.emulator.NativeLibrary
import dev.veshiru.emulator.R
import dev.veshiru.emulator.VeshiruApplication
import dev.veshiru.emulator.databinding.DialogListItemBinding
import dev.veshiru.emulator.model.CabinetMode
import dev.veshiru.emulator.adapters.CabinetLauncherDialogAdapter.CabinetModeViewHolder
import dev.veshiru.emulator.model.AppletInfo
import dev.veshiru.emulator.model.Game
import dev.veshiru.emulator.viewholder.AbstractViewHolder

class CabinetLauncherDialogAdapter(val fragment: Fragment) :
    AbstractListAdapter<CabinetMode, CabinetModeViewHolder>(
        CabinetMode.values().copyOfRange(1, CabinetMode.entries.size).toList()
    ) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CabinetModeViewHolder {
        DialogListItemBinding.inflate(LayoutInflater.from(parent.context), parent, false)
            .also { return CabinetModeViewHolder(it) }
    }

    inner class CabinetModeViewHolder(val binding: DialogListItemBinding) :
        AbstractViewHolder<CabinetMode>(binding) {
        override fun bind(model: CabinetMode) {
            binding.icon.setImageDrawable(
                ResourcesCompat.getDrawable(
                    binding.icon.context.resources,
                    model.iconId,
                    binding.icon.context.theme
                )
            )
            binding.title.setText(model.titleId)

            binding.root.setOnClickListener { onClick(model) }
        }

        private fun onClick(mode: CabinetMode) {
            val appletPath = NativeLibrary.getAppletLaunchPath(AppletInfo.Cabinet.entryId)
            NativeLibrary.setCurrentAppletId(AppletInfo.Cabinet.appletId)
            NativeLibrary.setCabinetMode(mode.id)
            val appletGame = Game(
                title = VeshiruApplication.appContext.getString(R.string.cabinet_applet),
                path = appletPath
            )
            val action = HomeNavigationDirections.actionGlobalEmulationActivity(appletGame)
            fragment.findNavController().navigate(action)
        }
    }
}
