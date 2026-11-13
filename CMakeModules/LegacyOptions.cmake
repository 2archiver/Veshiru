# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# The build options were renamed from YUZU_<name> to VESHIRU_<name> (docs/nx2/rename.md).
# A configure that still passes or caches the old name carries its value to the new one,
# so existing build directories and scripts keep their settings. A configure cannot tell an
# explicit -DVESHIRU_<name> from its cache entry, so the old name keeps deciding while
# VESHIRU_<name> still holds the value last carried from it, and warns when that replaces a
# different value. A VESHIRU_<name> set to anything else wins and the old name is ignored.
set(_veshiru_renamed_options
    BUILD_PRESET CMD CRASH_DUMPS DISABLE_LLVM DOWNLOAD_ANDROID_VVL DOWNLOAD_TIME_ZONE_DATA
    INSTALL_UDEV_RULES LEGACY QT_LANGUAGES QT_MIRROR ROOM ROOM_STANDALONE STATIC_BUILD STATIC_ROOM
    TESTS TZDB_PATH USE_BUNDLED_FFMPEG USE_BUNDLED_MOLTENVK USE_BUNDLED_OPENSSL USE_BUNDLED_QT
    USE_BUNDLED_SDL3 USE_BUNDLED_SIRIT USE_EXTERNAL_FFMPEG USE_QT_MULTIMEDIA USE_QT_WEB_ENGINE)

foreach (_name IN LISTS _veshiru_renamed_options)
    if (NOT DEFINED CACHE{YUZU_${_name}})
        continue()
    endif()
    set(_old "${YUZU_${_name}}")
    # The new name was carried from the old one before (or never set), so the old name still decides.
    if (NOT DEFINED VESHIRU_${_name} OR "${VESHIRU_${_name}}" STREQUAL "${_VESHIRU_CARRIED_${_name}}")
        if (DEFINED VESHIRU_${_name} AND NOT "${VESHIRU_${_name}}" STREQUAL "${_old}")
            # A configure cannot tell a -DVESHIRU_<name> repeating the carried value from the cached carry.
            message(WARNING "YUZU_${_name}=${_old} replaces VESHIRU_${_name}=${VESHIRU_${_name}}; pass only VESHIRU_${_name}")
        endif()
        message(DEPRECATION "YUZU_${_name} is now VESHIRU_${_name}; using YUZU_${_name}=${_old}")
        set(VESHIRU_${_name} "${_old}" CACHE STRING "" FORCE)
        set(_VESHIRU_CARRIED_${_name} "${_old}" CACHE INTERNAL "")
    else()
        message(WARNING "YUZU_${_name}=${_old} is ignored: VESHIRU_${_name}=${VESHIRU_${_name}} is set")
    endif()
    unset(YUZU_${_name} CACHE)
endforeach()
