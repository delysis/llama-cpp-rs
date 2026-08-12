#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
failed=0

while IFS= read -r -d '' file; do
  while IFS= read -r line; do
    use="${line#*uses: }"
    use="${use%%[[:space:]#]*}"

    case "$use" in
      ./*|docker://*) continue ;;
    esac

    if [[ ! "$use" =~ ^[^/@]+/[^/@]+@[0-9a-f]{40}$ ]]; then
      printf '%s\n' "mutable or malformed action reference: $file: $use" >&2
      failed=1
    fi
  done < <(grep -E '^[[:space:]]*-[[:space:]]+uses:[[:space:]]+' "$file" || true)

  if grep -nE '\bssh-keyscan\b' "$file"; then
    printf '%s\n' "live ssh-keyscan is forbidden in $file" >&2
    failed=1
  fi

  if grep -nE '\bgh[[:space:]]+auth[[:space:]]+login\b' "$file"; then
    printf '%s\n' "interactive GitHub CLI authentication is forbidden in $file" >&2
    failed=1
  fi
done < <(find "$root/.github/workflows" -type f \( -name '*.yml' -o -name '*.yaml' \) -print0)

nightly="$root/.github/workflows/update-llama-cpp.yml"
if [[ -f "$nightly" ]]; then
  permission_entries=0
  actions_write=0
  contents_write=0
  pull_requests_write=0
  while IFS= read -r permission; do
    permission_entries=$((permission_entries + 1))
    case "$permission" in
      actions:write) actions_write=1 ;;
      contents:write) contents_write=1 ;;
      pull-requests:write) pull_requests_write=1 ;;
      *)
        printf '%s\n' "unexpected nightly updater permission: $permission" >&2
        failed=1
        ;;
    esac
  done < <(
    awk '
      /^permissions:[[:space:]]*$/ { in_permissions = 1; next }
      in_permissions && /^[^[:space:]]/ { exit }
      in_permissions && /^[[:space:]]+[A-Za-z-]+:[[:space:]]+[A-Za-z-]+[[:space:]]*$/ {
        gsub(/[[:space:]]/, "")
        print
      }
    ' "$nightly"
  )

  if [[ "$permission_entries" -ne 3 || "$actions_write" -ne 1 || "$contents_write" -ne 1 || "$pull_requests_write" -ne 1 ]]; then
    printf '%s\n' "nightly updater requires only actions, contents, and pull-request write access" >&2
    failed=1
  fi

  if grep -nE '\$\{\{[[:space:]]*secrets\.[A-Za-z0-9_]+' "$nightly"; then
    printf '%s\n' "nightly updater must not depend on repository secrets" >&2
    failed=1
  fi

  if ! grep -Fq 'GH_TOKEN: ${{ github.token }}' "$nightly"; then
    printf '%s\n' "nightly updater must authenticate gh with the workflow token" >&2
    failed=1
  fi

  if ! grep -Fq 'gh workflow run llama-cpp-rs-check.yml --ref update-llama-cpp-${{ env.DATE }}' "$nightly"; then
    printf '%s\n' "nightly updater must explicitly dispatch checks for its generated branch" >&2
    failed=1
  fi
fi

exit "$failed"
