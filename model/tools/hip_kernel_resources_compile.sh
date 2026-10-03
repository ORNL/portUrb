#!/usr/bin/env bash

set -euo pipefail

output=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "-o" ]]; then
    output="$argument"
    break
  fi
  previous="$argument"
done

if [[ -z "$output" ]]; then
  exec "$@"
fi

remarks_file="${output}.hip-resource-remarks"
rm -f "$remarks_file"
diagnostics="$(mktemp "${output}.hip-resource-diagnostics.XXXXXX")"

set +e
"$@" 2> "$diagnostics"
status=$?
set -e

awk '
  function flush_pending( i) {
    for (i=1; i <= pending_count; i++) print pending[i]
    pending_count=0
  }
  /^In file included from / || (pending_count && /^[[:space:]]+from /) || (pending_count && $0 == "") {
    pending[++pending_count]=$0
    next
  }
  /remark:.*Rpass-analysis=kernel-resource-usage/ {
    pending_count=0
    skip_context=1
    next
  }
  skip_context && /^[[:space:]]*[0-9]+[[:space:]]+[|]/ { next }
  skip_context && /^[[:space:]]+[|]/ { next }
  { flush_pending(); skip_context=0; print }
  END { flush_pending() }
' "$diagnostics" >&2
if grep -q 'Rpass-analysis=kernel-resource-usage' "$diagnostics"; then
  mv "$diagnostics" "$remarks_file"
else
  rm -f "$diagnostics"
fi

exit "$status"