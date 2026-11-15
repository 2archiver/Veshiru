#!/usr/bin/env bash
# Summarise a failed ctest run.
#
# The Windows runner's raw log cannot be downloaded from outside, and neither can its
# artifacts, but check-run annotations are readable. So publish the useful part of the
# failure here: one ::error:: per failing test plus the assertion text, and the whole
# picture in the job step summary.
#
# Usage: tools/ci/ctest-annotate.sh <build-dir> <output-file>
set -u -o pipefail

BUILD_DIR=${1:-build}
OUT_FILE=${2:-test-output.txt}

if [ ! -s "$OUT_FILE" ]; then
  echo "::warning::no test output was captured at $OUT_FILE"
  exit 0
fi

# ctest prints "The following tests FAILED:" then "  3 - name (Failed)".
mapfile -t FAILED < <(
  sed -n '/The following tests FAILED:/,$p' "$OUT_FILE" |
    grep -oE '^[[:space:]]*[0-9]+ - .*' |
    sed -E 's/^[[:space:]]*[0-9]+ - //; s/[[:space:]]+\((Failed|Timeout|SEGFAULT|Subprocess aborted|Exception|Not Run)\)$//'
)

# Assertion / exception lines. Catch2 and QTest both use these shapes.
DETAIL_RE=': FAILED:|with expansion|FAIL!|Compared values|^[[:space:]]+(Actual|Expected)|REQUIRE\(|CHECK\(|CHECK_FALSE\(|INFO\(|ERROR:|Assertion|Loc:|signal [0-9]|terminate called|uncaught exception|no tests were found|Test #[0-9]+:.*\*\*\*'

# Per-test detail lines, keyed by the "Start <n>: <name>" markers ctest emits.
parse_detail() {
  awk -v re="$DETAIL_RE" '
    /^[[:space:]]*[0-9]+\/[0-9]+ Test[[:space:]]+#[0-9]+:/ { next }
    /^[[:space:]]*[0-9]+:[[:space:]]/ { next }
    /^[[:space:]]*Start[[:space:]]+[0-9]+: / {
      name = $0
      sub(/^[[:space:]]*Start[[:space:]]+[0-9]+:[[:space:]]*/, "", name)
      current = name
      next
    }
    # "with expansion:" is only useful together with the line that follows it.
    carry == 1 { print (current == "" ? "unknown" : current) "\t" $0; carry = 0; next }
    /with expansion:/ { carry = 1; print (current == "" ? "unknown" : current) "\t" $0; next }
    $0 ~ re { print (current == "" ? "unknown" : current) "\t" $0 }
  ' "$OUT_FILE"
}

DETAIL=$(parse_detail | head -200 || true)

if [ -z "$DETAIL" ]; then
  # No recognisable assertion line (QTest, a crash, a timeout...). Fall back to the raw lines
  # following each "Start <n>: <name>" marker so the failure is still identifiable.
  DETAIL=$(awk '
    /^[[:space:]]*Start[[:space:]]+[0-9]+: / {
      name = $0
      sub(/^[[:space:]]*Start[[:space:]]+[0-9]+:[[:space:]]*/, "", name)
      current = name; shown = 0; next
    }
    current != "" && shown < 25 && $0 !~ /^[[:space:]]*[0-9]+\/[0-9]+ Test/ { print current "\t" $0; shown++ }
    /^[[:space:]]*[0-9]+\/[0-9]+ Test[[:space:]]+#/ { current = "" }
  ' "$OUT_FILE" | head -60 || true)
fi

emit() { printf '%s\n' "$1" >>"$GITHUB_STEP_SUMMARY"; }

{
  echo "### Test failures"
  echo ""
  if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "**${#FAILED[@]} failing test(s):**"
    echo ""
    for t in "${FAILED[@]}"; do echo "- \`$t\`"; done
  else
    echo "ctest did not report a failing-test list; see the output below."
  fi
  echo ""

  if [ -n "$DETAIL" ]; then
    echo "#### Assertion and error lines"
    echo ""
    echo '```'
    printf '%s\n' "$DETAIL" | sed 's/\t/ -- /' 
    echo '```'
    echo ""
  fi

  echo "#### Captured output (tail)"
  echo ""
  echo '```'
  tail -150 "$OUT_FILE"
  echo '```'
  echo ""
  echo "#### Test inventory"
  echo ""
  echo '```'
  ctest --test-dir "$BUILD_DIR" -N 2>&1 | head -60 || true
  echo '```'
} >>"$GITHUB_STEP_SUMMARY"

# Annotations are capped at 10 per step, so spend them: names first, then details
# round-robin across the failing tests rather than all from the first one.
BUDGET=10
for t in "${FAILED[@]}"; do
  [ "$BUDGET" -le 0 ] && break
  echo "::error title=Failing test::${t}"
  BUDGET=$((BUDGET - 1))
done

if [ -n "$DETAIL" ] && [ "$BUDGET" -gt 0 ]; then
  # Round-robin over the tests named in DETAIL so a second failing test is not starved.
  printf '%s\n' "$DETAIL" |
    awk -F'\t' -v budget="$BUDGET" '
      { key = $1; n = ++count[key]; line[key, n] = $2; if (n > max) max = n; order[key] = 1 }
      END {
        emitted = 0
        for (round = 1; round <= max && emitted < budget; round++)
          for (k in order) {
            if (round in line_seen) {}
            if ((k, round) in line && emitted < budget) {
              printf "::error title=%s::%s\n", k, substr(line[k, round], 1, 380)
              emitted++
            }
          }
      }'
fi

echo "Summarised ${#FAILED[@]} failing test(s); detail lines: $(printf '%s\n' "$DETAIL" | grep -c . || true)"
