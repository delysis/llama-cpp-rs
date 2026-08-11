#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
checker="$repo_root/scripts/check-workflow-policy.sh"
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT
mkdir -p "$fixture/.github/workflows"

write_workflow() {
  printf '%s\n' "$1" > "$fixture/.github/workflows/check.yml"
}

write_workflow 'steps:
  - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # reviewed
  - uses: ./local-action
  - uses: docker://alpine:3.22'
"$checker" "$fixture"

write_workflow 'steps:
  - uses: actions/checkout@v4'
if "$checker" "$fixture" >/dev/null 2>&1; then
  echo "mutable action tag unexpectedly passed" >&2
  exit 1
fi

write_workflow 'steps:
  - run: ssh-keyscan github.com'
if "$checker" "$fixture" >/dev/null 2>&1; then
  echo "ssh-keyscan unexpectedly passed" >&2
  exit 1
fi

echo "workflow policy fixtures passed"
