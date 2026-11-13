# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-License-Identifier: GPL-3.0-or-later

add_library(veshiru-android SHARED
    emu_window/emu_window.cpp
    emu_window/emu_window.h
    native.cpp
    native.h
    native_config.cpp
    native_freedreno.cpp
    android_settings.cpp
    game_metadata.cpp
    native_log.cpp
    android_config.cpp
    android_config.h
    native_input.cpp
    native_post_processing.cpp
)

set_property(TARGET veshiru-android PROPERTY IMPORTED_LOCATION ${FFmpeg_LIBRARY_DIR})

target_link_libraries(veshiru-android PRIVATE audio_core common core input_common frontend_common video_core)
target_link_libraries(veshiru-android PRIVATE android camera2ndk EGL glad jnigraphics log GLESv2)
if (ARCHITECTURE_arm64)
    target_link_libraries(veshiru-android PRIVATE adrenotools)
endif()

target_link_libraries(veshiru-android PRIVATE ${FFmpeg_LIBRARIES} OpenSSL::SSL cpp-jwt::cpp-jwt)
if (ENABLE_UPDATE_CHECKER)
    target_compile_definitions(veshiru-android PUBLIC ENABLE_UPDATE_CHECKER)
endif()

set(CPACK_PACKAGE_EXECUTABLES ${CPACK_PACKAGE_EXECUTABLES} veshiru-android)

target_link_options(veshiru-android PRIVATE "-Wl,-Bsymbolic")
