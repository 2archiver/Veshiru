# Changelog

All notable changes to Veshiru are documented here. Each release's notes are taken from its
`## [X.Y.Z]` section (release candidates use the section of the version they are a candidate for).
New entries go under `## [Unreleased]`; `tools/ci/bump-version.sh` turns them into a version section.

## [Unreleased]

### Added

- **Save transfer to Eglantine**: right-click a game and choose **Link to Eglantine** to share
  save data between Veshiru and Eglantine (both emulators now reference the same save files).
  The transfer copies and verifies the save data before linking, keeps a timestamped
  `*.veshiru-backup-*` copy of the replaced saves instead of deleting them, and every failure
  is shown to the user with the exact step and error instead of failing silently. Portable
  Eglantine installations are supported via folder selection; the legacy `eden` data folder
  used by Eglantine build 17 is detected automatically.

### Changed

- **Console appearance**: the in-app console now uses a dark background with larger, bright
  high-contrast text (similar to Ryubing's console), making log output much easier to read.
- "Link to Ryujinx" has been removed and replaced by "Link to Eglantine".

### Fixed

- Save transfer no longer fails silently: filesystem errors, missing saves and link failures
  are now reported with a visible message and written to the log.
- Windows junction creation no longer shells out to `cmd /c mklink /J`; it uses a native
  `FSCTL_SET_REPARSE_POINT` call, which is reliable inside a GUI application.
- Windows junction detection now uses the reparse point attribute, so existing linked saves
  are recognised correctly (boost reports junctions as ordinary directories).
- The console no longer shows a Critical "assert" (and no soft assert fires) the first time a
  Switch 2 title calls an unimplemented `nx2` service command that Veshiru auto-stubs; it is
  logged as a warning with the usual report file instead.

## [1.1.0] - 2026-11-13

### Performance

- General FPS performance refinements.
- Improved frame pacing and gameplay consistency.
- Ongoing CPU and GPU overhead optimisations.
- Smoother performance in demanding gameplay scenarios.
- General responsiveness improvements.

### Stability

- General stability refinements.
- Improved reliability during extended gameplay sessions.
- Minor fixes and internal optimisations.

### Compatibility

- Continued refinement of game compatibility.
- General improvements to rendering and emulation behaviour.

### Notes

- Actual FPS improvements vary depending on hardware, graphics drivers, emulator settings and
  individual games. Performance gains are not guaranteed across all titles.
- Veshiru is now a Windows-exclusive emulator: Linux and macOS packages are no longer built or
  published, and the documentation and CI reflect Windows-only support.
- New wave-style application icon.

## [0.1.0]

- Initial release candidate
