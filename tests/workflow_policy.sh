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

write_workflow 'steps:
  - run: gh auth login --with-token'
if "$checker" "$fixture" >/dev/null 2>&1; then
  echo "interactive gh authentication unexpectedly passed" >&2
  exit 1
fi

rm "$fixture/.github/workflows/check.yml"
cat > "$fixture/.github/workflows/update-llama-cpp.yml" <<'YAML'
permissions:
  actions: write
  pull-requests: write
  contents: write
steps:
  - env:
      GH_TOKEN: ${{ github.token }}
    run: gh pr create --fill
  - env:
      GH_TOKEN: ${{ github.token }}
    run: gh workflow run llama-cpp-rs-check.yml --ref update-llama-cpp-${{ env.DATE }}
YAML
"$checker" "$fixture"

cat >> "$fixture/.github/workflows/update-llama-cpp.yml" <<'YAML'
  - env:
      GH_TOKEN: ${{ secrets.CUSTOM_PAT }}
    run: gh pr edit 1
YAML
if "$checker" "$fixture" >/dev/null 2>&1; then
  echo "custom nightly updater secret unexpectedly passed" >&2
  exit 1
fi

cat > "$fixture/.github/workflows/update-llama-cpp.yml" <<'YAML'
permissions:
  actions: write
  pull-requests: write
  contents: write
  issues: write
steps:
  - env:
      GH_TOKEN: ${{ github.token }}
    run: gh pr create --fill
  - env:
      GH_TOKEN: ${{ github.token }}
    run: gh workflow run llama-cpp-rs-check.yml --ref update-llama-cpp-${{ env.DATE }}
YAML
if "$checker" "$fixture" >/dev/null 2>&1; then
  echo "excess nightly updater permission unexpectedly passed" >&2
  exit 1
fi

echo "workflow policy fixtures passed"
