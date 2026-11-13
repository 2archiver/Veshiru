# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# The Android frontend, its JNI glue and the Android names other files refer to.
#
# The Kotlin package org.yuzu.yuzu_emu becomes dev.veshiru.emulator, the id of the Linux
# desktop entry. The applicationId (dev.eden.eden_emulator) is not a yuzu name and stays, so
# the default SharedPreferences file, the provider authorities and the app's storage are the
# same as before.

ANDROID = ["src/android/*", "src/common/android/*"]

PATHS = [
    (r"/java/org/yuzu/yuzu_emu/", "/java/dev/veshiru/emulator/", {"files": "src/android/*"}),
    (r"Yuzu", "Veshiru", {"files": "src/android/*"}),
    (r"yuzu", "veshiru", {"files": "src/android/*"}),
]

TEXT = [
    # The package in Kotlin, the manifest, layouts, navigation graphs and docs, as a JNI class
    # name or signature, and in JNI export names ("_" in a package segment mangles to "_1").
    (r"org\.yuzu\.yuzu_emu(?![A-Za-z0-9_])", "dev.veshiru.emulator"),
    (r"org/yuzu/yuzu_emu(?![A-Za-z0-9_])", "dev/veshiru/emulator"),
    (r"\bJava_org_yuzu_yuzu_1emu_", "Java_dev_veshiru_emulator_"),
    # Android names used outside the Android tree: the launcher icons update-icons.sh writes,
    # the logcat tag in common/logging.cpp, and the input device accessors and Kotlin
    # interface the input driver uses.
    (r"\bic_yuzu", "ic_veshiru"),
    (r"\bYuzuNative\b", "VeshiruNative"),
    (r"Yuzu", "Veshiru", {"files": ["src/input_common/drivers/android.*"]}),
    # Everything else in the Android tree: classes (YuzuApplication, YuzuInputDevice, ...),
    # themes and styles, colour and string resource names, and the JNI id cache.
    (r"Yuzu", "Veshiru", {"files": ANDROID}),
    (r"yuzu", "veshiru", {"files": ANDROID}),
    (r"YUZU", "VESHIRU", {"files": ANDROID}),
]

KEEP = [
    (r"https://yuzu-(mirror\.github\.io|emu\.org)/help/quickstart[^\s\"<\\]*", "removed",
     "link to the yuzu quickstart guide: delete the string and its helpLinkId use (docs/nx2/rename.md)",
     "src/android/app/src/main/res/values*/strings.xml"),
]
