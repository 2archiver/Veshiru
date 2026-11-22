<!--
# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
-->

# Release notes

One file per stable version, named after the version: `2.7.0.md` for the `v2.7.0` tag.

The release workflow publishes `docs/release-notes/<version>.md` as the description at the top of
the release page when it exists (`tools/ci/release-description.sh` picks it, and the CHANGELOG
section when there is no such file); the download tables and install instructions are appended after
it.

Write for the person who is deciding whether to download, not for the person who wrote the change:

- **Short.** Three to five headlines, two sentences each at most. Nothing about files, functions or
  log identifiers unless the reader has to type them.
- **What they will notice**, in their words: "stutters less", "the fairy stops flickering", not
  "the boot plan builds the spare pipelines".
- **What to do about it**, when there is something: a setting, an environment variable, a title that
  needs nothing.
- **Known issues**, honestly. A release that costs something (a slower first load) says so.

`CHANGELOG.md` stays the engineering record: what changed, why, and what has been verified. It is
long on purpose. The release page shows these notes instead.
