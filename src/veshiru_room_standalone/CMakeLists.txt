# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(veshiru_room_standalone veshiru_room_standalone.cpp)

set_target_properties(veshiru_room_standalone PROPERTIES OUTPUT_NAME "eden-room")

target_link_libraries(veshiru_room_standalone PRIVATE veshiru-room)

if(UNIX AND NOT APPLE)
    install(TARGETS veshiru_room_standalone RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

if (VESHIRU_STATIC_ROOM)
    target_link_options(veshiru_room_standalone PRIVATE "-static")
endif()

create_target_directory_groups(veshiru_room_standalone)
