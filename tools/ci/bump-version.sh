#!/usr/bin/env bash
# usage: bump-version.sh major|minor|patch
# Prepare a new stable version and move [Unreleased] entries into its dated section.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bump="${1:-}"
cur=$(tr -d '[:space:]' < VERSION)
re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-rc([1-9][0-9]*))?$'
[[ "$cur" =~ $re ]] || {
  echo "VERSION must be X.Y.Z or X.Y.Z-rcN (without leading zeroes): $cur" >&2
  exit 1
}
maj=${BASH_REMATCH[1]}
min=${BASH_REMATCH[2]}
pat=${BASH_REMATCH[3]}
rc=${BASH_REMATCH[5]:-}
core="$maj.$min.$pat"
today=$(date -u +%F)

section_exists() {
  awk -v name="$1" '
    $0 == "## [" name "]" || index($0, "## [" name "] - ") == 1 { found = 1 }
    END { exit !found }
  ' CHANGELOG.md
}

section_has_body() {
  local body
  body=$(bash tools/ci/changelog-section.sh "$1")
  [[ -n "$(tr -d '[:space:]' <<< "$body")" ]]
}

case "$bump" in
  major|minor|patch)
    [[ -z "$rc" ]] || {
      echo "$cur is a release candidate; set VERSION to a stable version first" >&2
      exit 1
    }
    case "$bump" in
      major) core="$((maj + 1)).0.0" ;;
      minor) core="$maj.$((min + 1)).0" ;;
      patch) core="$maj.$min.$((pat + 1))" ;;
    esac
    new="$core"
    section_exists Unreleased || {
      echo "CHANGELOG.md is missing the exact ## [Unreleased] heading" >&2
      exit 1
    }
    section_has_body Unreleased || {
      echo "CHANGELOG.md has no entries under [Unreleased]" >&2
      exit 1
    }
    ! section_exists "$core" || {
      echo "CHANGELOG.md already contains a [$core] section" >&2
      exit 1
    }
    # Move the Unreleased body below the new version heading and leave a fresh empty Unreleased.
    awk -v ver="$core" -v date="$today" '
      $0 == "## [Unreleased]" && !done {
        print "## [Unreleased]"
        print ""
        print "## [" ver "] - " date
        done = 1
        next
      }
      { print }
      END { if (!done) exit 1 }
    ' CHANGELOG.md > CHANGELOG.md.tmp
    mv CHANGELOG.md.tmp CHANGELOG.md
    ;;
  *)
    echo "usage: bump-version.sh major|minor|patch" >&2
    exit 1
    ;;
esac

printf '%s\n' "$new" > VERSION
printf '%s\n' "$new"
