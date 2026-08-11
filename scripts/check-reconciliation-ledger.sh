#!/bin/sh
set -eu

ledger="docs/reconciliation/a74db-to-01e48.md"

test -f "$ledger"

expected_count=3
actual_count=$(grep -Ec '^\| `[0-9a-f]{40}` \| (already-present|ported|intentionally-superseded) \|' "$ledger")
if [ "$actual_count" -ne "$expected_count" ]; then
    echo "reconciliation ledger has $actual_count classified commits, expected $expected_count" >&2
    exit 1
fi

for commit in \
    fa5d5f164ab6abef521ee7f1280273b60535dc6f \
    0adab1a3bb98d875cc15a7ffd0aee59043712b21 \
    a74dbb79f96e0ebad8b0737ee1d3c9c1deb185af
do
    count=$(grep -Ec "^\\| \`$commit\` \\| (already-present|ported|intentionally-superseded) \\|" "$ledger")
    if [ "$count" -ne 1 ]; then
        echo "reconciliation ledger must classify $commit exactly once" >&2
        exit 1
    fi
done

echo "reconciliation ledger ok: all three stale commits have one disposition"
