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

# Annotations are capped at 10 per step, so spend them: names first, then the digest, then a raw
# tail of the output. The raw tail is unconditional on purpose: it is the only channel that keeps
# working when the log and artifacts cannot be downloaded.
BUDGET=10
for t in "${FAILED[@]}"; do
  [ "$BUDGET" -le 0 ] && break
  echo "::error title=Failing test::${t}"
  BUDGET=$((BUDGET - 1))
done

annotate_block() {
  # $1 title, $2 body. Multi-line annotation messages are allowed; keep the body bounded.
  local title=$1 body=$2
  body=$(printf '%s\n' "$body" | cut -c1-3000)
  printf '::error title=%s::%s\n' "$title" "${body//$'\n'/%0A}"
}

if [ -n "$DETAIL" ] && [ "$BUDGET" -gt 1 ]; then
  annotate_block "Test details" "$(printf '%s\n' "$DETAIL" | sed 's/\t/ -- /' | head -15)"
  BUDGET=$((BUDGET - 1))
fi

if [ "$BUDGET" -gt 0 ]; then
  annotate_block "Test output (tail)" "$(tail -25 "$OUT_FILE")"
fi

echo "Summarised ${#FAILED[@]} failing test(s); detail lines: $(printf '%s' "$DETAIL" | grep -c . || true)"
