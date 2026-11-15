#!/usr/bin/env bash
# Summarise a failed ctest run.
#
# The Windows runner's raw log is not always reachable from outside (job log download and
# artifact download can both fail), but check-run annotations are. So publish the useful
# part of the failure here: one ::error:: per failing test plus the assertion text, and the
# same content in the job step summary.
#
# Usage: tools/ci/ctest-annotate.sh <build-dir> <output-file>
set -u -o pipefail

BUILD_DIR=${1:-build}
OUT_FILE=${2:-test-output.txt}

if [ ! -s "$OUT_FILE" ]; then
  echo "::warning::no test output was captured at $OUT_FILE"
  exit 0
fi

# ctest prints "The following tests FAILED:" then lines like "  3 - filename (Failed)".
mapfile -t FAILED < <(
  sed -n '/The following tests FAILED:/,$p' "$OUT_FILE" |
    grep -oE '^[[:space:]]*[0-9]+ - .*' |
    sed -E 's/^[[:space:]]*[0-9]+ - //; s/[[:space:]]+\((Failed|Timeout|SEGFAULT|Subprocess aborted|Exception)\)$//'
)

emit() { printf '%s\n' "$1" >>"$GITHUB_STEP_SUMMARY"; }

{
  echo "### Test failures"
  if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo "**${#FAILED[@]} failing test(s):**"
    echo ""
    for t in "${FAILED[@]}"; do echo "- \`$t\`"; done
    echo ""
  else
    echo ""
    echo "ctest did not report a failing-test list; see output below."
  fi
} >>"$GITHUB_STEP_SUMMARY"

if [ "${#FAILED[@]}" -gt 0 ]; then
  for t in "${FAILED[@]}"; do
    echo "::error title=Test failed::${t}"
  done
fi

# Include the assertion / exception lines. These are what a human needs first.
DETAIL=$(grep -E 'REQUIRE|CHECK|error:|Assertion|signal|exception|terminate called|what\(\)' \
  "$OUT_FILE" | head -60 || true)
if [ -n "$DETAIL" ]; then
  {
    echo "#### Assertion and error lines"
    echo ""
    echo '```'
    printf '%s\n' "$DETAIL"
    echo '```'
  } >>"$GITHUB_STEP_SUMMARY"
  # Annotations are capped (10 per step), so only the first few go on the check run.
  printf '%s\n' "$DETAIL" |
    head -8 |
    while IFS= read -r line; do
      echo "::error title=Test detail::$(printf '%s' "$line" | cut -c1-400)"
    done
fi

# Then a bounded tail of the full output for context.
{
  echo ""
  echo "#### Captured output (tail)"
  echo ""
  echo '```'
  tail -100 "$OUT_FILE"
  echo '```'
} >>"$GITHUB_STEP_SUMMARY"

{
  echo ""
  echo "#### Test inventory"
  echo ""
  echo '```'
  ctest --test-dir "$BUILD_DIR" -N 2>&1 | head -60 || true
  echo '```'
} >>"$GITHUB_STEP_SUMMARY"

echo "Summarised ${#FAILED[@]} failing test(s) into the job summary."
