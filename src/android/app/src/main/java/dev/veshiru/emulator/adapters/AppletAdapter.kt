// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.os.bundleOf
import androidx.core.content.res.ResourcesCompat
import androidx.fragment.app.FragmentActivity
import androidx.navigation.findNavController
import dev.veshiru.emulator.NativeLibrary
import dev.veshiru.emulator.R
import dev.veshiru.emulator.VeshiruApplication
import dev.veshiru.emulator.databinding.CardSimpleOutlinedBinding
import dev.veshiru.emulator.model.Applet
import dev.veshiru.emulator.model.AppletInfo
import dev.veshiru.emulator.model.Game
import dev.veshiru.emulator.viewholder.AbstractViewHolder

class AppletAdapter(val activity: FragmentActivity, applets: List<Applet>) :
    AbstractListAdapter<Applet, AppletAdapter.AppletViewHolder>(applets) {
    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): AppletAdapter.AppletViewHolder {
        CardSimpleOutlinedBinding.inflate(LayoutInflater.from(parent.context), parent, false)
            .also { return AppletViewHolder(it) }
    }

    inner class AppletViewHolder(val binding: CardSimpleOutlinedBinding) :
        AbstractViewHolder<Applet>(binding) {
        override fun bind(model: Applet) {
            binding.title.setText(model.titleId)
            binding.description.setText(model.descriptionId)
            binding.icon.setImageDrawable(
                ResourcesCompat.getDrawable(
                    binding.icon.context.resources,
                    model.iconId,
                    binding.icon.context.theme
                )
            )

            binding.root.setOnClickListener { onClick(model) }
        }

        fun onClick(applet: Applet) {
            val appletPath = NativeLibrary.getAppletLaunchPath(applet.appletInfo.entryId)
            if (appletPath.isEmpty()) {
                Toast.makeText(
                    binding.root.context,
                    R.string.applets_error_applet,
                    Toast.LENGTH_SHORT
                ).show()
                return
            }

            if (applet.appletInfo == AppletInfo.Cabinet) {
                binding.root.findNavController()
                    .navigate(R.id.action_appletLauncherFragment_to_cabinetLauncherDialogFragment)
                return
            }

            NativeLibrary.setCurrentAppletId(applet.appletInfo.appletId)
            val appletGame = Game(
                title = VeshiruApplication.appContext.getString(applet.titleId),
                path = appletPath
            )
            binding.root.findNavController().navigate(
                R.id.action_global_emulationActivity,
                bundleOf(
                    "game" to appletGame,
                    "custom" to false
                )
            )
        }
    }
}
