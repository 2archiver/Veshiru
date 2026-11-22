#!/usr/bin/env bash
# usage: release-notes.sh <output.md>
# Env: KIND TAG SHA REPO BRANCH PREV_TAG MACOS_RUNNER ASSET_DIR (default release-assets).
# The download tables include only assets that exist, so one-platform manual builds are supported.
set -euo pipefail

out="${1:?usage: release-notes.sh <output.md>}"
dir="${ASSET_DIR:-release-assets}"
repo="${REPO:?REPO is required}"
tag="${TAG:?TAG is required}"
sha="${SHA:?SHA is required}"
kind="${KIND:?KIND is required}"
base="https://github.com/${repo}/releases/download/${tag}"
short="${sha:0:7}"
macos_ver="${MACOS_RUNNER:-macos-15}"
macos_ver="${macos_ver#macos-}"

link() { if [[ -e "$dir/$2" ]]; then printf '[%s](%s/%s)' "$1" "$base" "$2"; fi; return 0; }
join() { if [[ -n "$1" && -n "$2" ]]; then printf '%s · %s' "$1" "$2"; else printf '%s%s' "$1" "$2"; fi; }
row() { if [[ -n "$2$3" ]]; then printf '| %s | %s | %s | %s |\n' "$1" "$2" "$3" "${4:-}"; fi; }
commits() { git log --no-merges --pretty='- %s (`%h`)' "$@" 2>/dev/null | head -n 100 || true; }

{
  case "$kind" in
    stable|rc)
      core="${tag#v}"
      core="${core%%-rc*}"
      if [[ "$kind" == rc ]]; then
        printf '> **Release candidate of %s.** Please test it and report problems before the final release.\n\n' "$core"
      fi
      # The short user-facing notes when they exist, the CHANGELOG section otherwise.
      section=$(bash tools/ci/release-description.sh "$core")
      [[ -n "$(tr -d '[:space:]' <<< "$section")" ]] || {
        echo "No release notes and no non-empty CHANGELOG [$core] section" >&2
        exit 1
      }
      printf '%s\n\n' "$section"
      ;;
    daily)
      printf 'Automatic daily build of `main` at `%s`.\n\n## Changes since the previous daily build\n\n' "$short"
      if [[ -n "${PREV_TAG:-}" ]]; then commits "$PREV_TAG..$sha"; else commits -n 30 "$sha"; fi
      printf '\n'
      ;;
    branch)
      printf 'Test build of branch `%s` at `%s`.\n\n**This is not a release.** It is for testing, and the next push to the branch replaces it.\n\n## Commits on top of main\n\n' "${BRANCH:-?}" "$short"
      commits "origin/nx2..$sha"
      printf '\n'
      ;;
    *)
      echo "Unknown kind: $kind" >&2
      exit 1
      ;;
  esac

  w_std=$(link zip veshiru-windows-x86_64.zip)
  w_pgo=$(link 'PGO zip' veshiru-windows-x86_64-pgo.zip)
  if [[ -n "$w_std$w_pgo" ]]; then
    cat <<'EOF2'
## Windows

Veshiru is a Windows-exclusive emulator: it is built and tested for 64-bit Windows 10 and 11 only.
The packages are in-place zip files: extract the whole archive and run `veshiru.exe`, or
`veshiru-cmd.exe` from a terminal.

| Build Type | Standard | PGO (Recommended) | Notes |
| --- | --- | --- | --- |
EOF2
    row 'amd64 / x86_64' "$w_std" "$w_pgo" 'Both built with clang-cl and ThinLTO; the PGO zip adds a profile trained in Veshiru on Ocarina of Time'
    printf '\n'
  fi

  if [[ -e "$dir/SHA256SUMS.txt" ]]; then
    printf 'SHA-256 checksums for the archives are in `SHA256SUMS.txt`.\n\n'
  fi

  if [[ -n "$w_pgo" ]]; then
    echo "The PGO build is the same program compiled with profile-guided optimization using a profile trained in Veshiru on Ocarina of Time, so other games gain less. Prefer it when your CPU supports it."
  fi
} > "$out"
