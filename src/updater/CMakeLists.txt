# SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(veshiru-updater main.cpp ../frontend_common/update_installer.cpp)
target_include_directories(veshiru-updater PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(veshiru-updater PRIVATE cxx_std_20)
target_link_libraries(veshiru-updater PRIVATE user32 shell32)
set_target_properties(veshiru-updater PROPERTIES WIN32_EXECUTABLE ON)

# The helper runs from a temporary folder, away from the DLLs next to veshiru.exe.
if (MSVC)
    set_property(TARGET veshiru-updater PROPERTY MSVC_RUNTIME_LIBRARY
                 "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_link_options(veshiru-updater PRIVATE -static)
endif()
