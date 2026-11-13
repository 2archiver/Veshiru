// SPDX-License-Identifier: GPL-2.0-or-later

package dev.veshiru.emulator.features.input

import android.content.Context
import android.os.Build
import android.os.CombinedVibration
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.InputDevice
import androidx.annotation.Keep
import androidx.annotation.RequiresApi
import dev.veshiru.emulator.VeshiruApplication

@Keep
@Suppress("DEPRECATION")
interface VeshiruVibrator {
    fun supportsVibration(): Boolean

    fun vibrate(intensity: Float)

    companion object {
        fun getControllerVibrator(device: InputDevice): VeshiruVibrator =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                VeshiruVibratorManager(device.vibratorManager)
            } else {
                VeshiruVibratorManagerCompat(device.vibrator)
            }

        fun getSystemVibrator(): VeshiruVibrator =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vibratorManager = VeshiruApplication.appContext
                    .getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                VeshiruVibratorManager(vibratorManager)
            } else {
                val vibrator = VeshiruApplication.appContext
                    .getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                VeshiruVibratorManagerCompat(vibrator)
            }

        fun getVibrationEffect(intensity: Float): VibrationEffect? {
            if (intensity > 0f) {
                return VibrationEffect.createOneShot(
                    50,
                    (255.0 * intensity).toInt().coerceIn(1, 255)
                )
            }
            return null
        }
    }
}

@RequiresApi(Build.VERSION_CODES.S)
class VeshiruVibratorManager(private val vibratorManager: VibratorManager) : VeshiruVibrator {
    override fun supportsVibration(): Boolean {
        return vibratorManager.vibratorIds.isNotEmpty()
    }

    override fun vibrate(intensity: Float) {
        val vibration = VeshiruVibrator.getVibrationEffect(intensity) ?: return
        vibratorManager.vibrate(CombinedVibration.createParallel(vibration))
    }
}

class VeshiruVibratorManagerCompat(private val vibrator: Vibrator) : VeshiruVibrator {
    override fun supportsVibration(): Boolean {
        return vibrator.hasVibrator()
    }

    override fun vibrate(intensity: Float) {
        val vibration = VeshiruVibrator.getVibrationEffect(intensity) ?: return
        vibrator.vibrate(vibration)
    }
}
