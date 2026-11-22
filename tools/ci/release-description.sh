#!/usr/bin/env bash
# usage: release-description.sh <version>
# Prints what a release page says about the release itself, before the download tables: the short,
# user-facing notes in docs/release-notes/<version>.md when that file exists, and the matching
# CHANGELOG section otherwise (a version whose notes have not been written yet still publishes, in
# engineering voice). An rc version (<version>-rcN) uses the notes of the release it is a candidate
# for.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
version="${1:?usage: release-description.sh <version>}"
core="${version%%-rc*}"

notes="docs/release-notes/${core}.md"
if [[ -f "$notes" ]]; then
  # The file's leading SPDX comment block is for the source tree, not for the release page.
  if [[ "$(head -n 1 "$notes")" == "<!--" ]]; then
    sed '1,/^-->/d' "$notes"
  else
    cat "$notes"
  fi
else
  bash tools/ci/changelog-section.sh "$core"
fi
