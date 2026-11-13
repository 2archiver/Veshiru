# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# Build system, CI, packaging and the source directory names.

NOT_ANDROID = ["src/android/*"]

# Every tracked path outside Android: src/veshiru -> src/veshiru, yuzu_cmd -> veshiru_cmd, dist/veshiru.manifest ...
PATHS = [
    (r"yuzu", "veshiru", {"exclude": NOT_ANDROID}),
]

# CMake options and the compile definitions they produce. The old -DYUZU_* spellings
# keep working through CMakeModules/LegacyOptions.cmake.
OPTIONS = [
    "BOOST_v1", "BUILD_PRESET", "BUNDLED_OPENSSL", "CMD", "CMD_ENTRY", "CRASH_DUMPS",
    "DISABLE_LLVM", "DOWNLOAD_ANDROID_VVL", "DOWNLOAD_TIME_ZONE_DATA", "EXE_DIR", "ICON",
    "INSTALL_UDEV_RULES", "LEGACY", "MSVC_DIAGNOSTICS", "MSVC_DIAGNOSTIC_CODES",
    "MSVC_DIAGNOSTIC_LEVEL", "QT_COMPONENTS", "QT_LANGUAGES", "QT_MIRROR", "QT_OPTIONAL",
    "ROOM", "ROOM_STANDALONE", "STATIC_BUILD", "STATIC_ROOM", "TESTS", "TZDB_PATH",
    "USE_BUNDLED_FFMPEG", "USE_BUNDLED_MOLTENVK", "USE_BUNDLED_OPENSSL", "USE_BUNDLED_QT",
    "USE_BUNDLED_SDL2", "USE_BUNDLED_SDL3", "USE_BUNDLED_SIRIT", "USE_CPM",
    "USE_EXTERNAL_FFMPEG", "USE_EXTERNAL_SDL", "USE_EXTERNAL_SDL3", "USE_FP16",
    "USE_QT_MULTIMEDIA", "USE_QT_WEB_ENGINE", "ANDROID_ARGS",
]

BUILD_FILES = [
    "CMakeLists.txt", "*/CMakeLists.txt", "*.cmake", ".github/*", ".ci/*", "hooks/*",
    "*.rc", "*.qrc", "*.manifest", "*.desktop", "*.metainfo.xml", "*/PKGBUILD", "*.nsi",
    "tools/*.sh", "tools/translations/*", ".reuse/dep5", "Doxyfile",
]

TEXT = [
    # Paths to the renamed directories and files, in any file.
    (r"\bsrc/yuzu(_cmd|_room_standalone)?\b", r"src/veshiru\1"),
    (r"\btests/yuzu_cmd\b", "tests/veshiru_cmd"),
    (r"(?<=[\"<])yuzu(_cmd)?/", r"veshiru\1/"),
    (r"(?<=<header>)yuzu/", "veshiru/"),
    (r"(#include\s*[\"<][^\">]*)", lambda m: m.group(1).replace("yuzu", "veshiru")),
    (r"\byuzu(_room_standalone|_room|_icon)?\.(rc|qrc|cpp|h|manifest|bmp)\b", r"veshiru\1.\2"),
    # CMake option and definition names, wherever they appear (C++ #ifdef, docs, CI).
    (r"YUZU_(" + "|".join(OPTIONS) + r")(?![A-Za-z0-9_])", r"VESHIRU_\1",
     {"files": ["*", "externals/CMakeLists.txt", "externals/*/CMakeLists.txt", "externals/*/*.cmake"]}),
    # Targets.
    (r"\byuzu-cmd\b", "veshiru-cmd"),
    (r"\byuzu-room\b", "veshiru-room"),
    (r"\byuzu_cmd\b", "veshiru_cmd"),
    (r"\byuzu-android\b", "veshiru-android"),
    (r"\byuzu_room_standalone\b", "veshiru_room_standalone"),
    # Everything else in build files and in the small frontends this area moves.
    (r"\byuzu\b", "veshiru", {"files": BUILD_FILES}),
    (r"\bYuzu\b", "Veshiru", {"files": BUILD_FILES}),
    (r"yuzu", "veshiru", {"files": ["src/veshiru_cmd/*", "src/veshiru_room_standalone/*", "src/dedicated_room/*",
                                   "src/tests/veshiru_cmd/*"]}),
    (r"Yuzu", "Veshiru", {"files": ["src/veshiru_cmd/*", "src/veshiru_room_standalone/*", "src/dedicated_room/*",
                                   "src/tests/veshiru_cmd/*"]}),
]

KEEP = [
    (r"YUZU_\$\{_?\w+\}|\"YUZU_\"", "compat", "old -DYUZU_* CMake options read by LegacyOptions.cmake",
     "CMakeModules/LegacyOptions.cmake"),
    (r"YuzuRoom-BanList-1", "compat", "ban list file magic of existing room ban lists"),
    (r"derived from Yuzu and Sudachi|inherited from yuzu and Sudachi", "attribution", "project lineage"),
]
