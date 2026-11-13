# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
# The Qt frontend (src/veshiru), the shared Qt code (src/qt_common) and the translations.

# Build files in these directories belong to the build area, Markdown to the docs area.
QT = {
    "files": ["src/veshiru/*", "src/qt_common/*", "dist/languages/*", "dist/english_plurals/*",
              "dist/qt_themes/*", "dist/icons/*"],
    "exclude": ["*/CMakeLists.txt", "*.cmake", "*.qrc", "*.rc", "*.manifest", "*.md"],
}

# A translation is looked up by its exact source text, so every rule rewrites .ui, .cpp and
# .ts the same way. The boundaries are ASCII so that "yuzu" followed by a CJK or Hangul
# particle in a translation still matches.
TEXT = [
    (r"com\.yuzu-emu\.yuzu", "dev.veshiru.emulator", {"files": "src/veshiru/Info.plist"}),
    # Environment variables the frontend sets for its own startup-check child process.
    (r"\bYUZU_(IS_CHILD|DO_STARTUP_CHECKS)\b", r"VESHIRU_\1"),
    (r"\bYUZU_(?=[A-Z])", "VESHIRU_", QT),
    # Include paths in .ui <header> elements, then identifiers.
    (r"(?<![A-Za-z0-9_])yuzu(?=/)", "veshiru", QT),
    (r"(?<![A-Za-z0-9])yuzu_", "veshiru_", QT),
    # Namespace, camel-case object names and prose.
    (r"Yuzu", "Veshiru", QT),
    (r"(?<![A-Za-z0-9_])yuzu(?![A-Za-z0-9_])", "Veshiru", QT),
]

KEEP = [
    (r"QT_TR_NOOP\(\"Yuzu\"\), Yuzu\)", "compat",
     "interop: yuzu is one of the other emulators whose data the user can import (Common::FS::YuzuDir)",
     "src/veshiru/migration_worker.h"),
    (r"<(source|translation)>Yuzu</(source|translation)>", "compat",
     "interop: translation of the importable emulator name kept in migration_worker.h",
     "dist/languages/*.ts"),
    (r"https://github\.com/eden-emulator/yuzu-mod-archive", "compat",
     "external URL: Eden's archive of mods made for yuzu, opened by the mods page action"),
    (r"Copyright \d{4}(-\d{4})? yuzu Emulator Project", "attribution", "copyright notice"),
    (r".*GPL ?v3.*", "attribution", "Eden about text in old translations: based on the yuzu emulator",
     "dist/languages/*.ts"),
]
