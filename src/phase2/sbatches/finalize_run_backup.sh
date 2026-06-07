#!/bin/bash
set -euo pipefail

# Finalize one array-run folder and back it up to git.
# Usage:
#   bash src/phase2/sbatches/finalize_run_backup.sh <array_job_id> <scenario> <architecture>

if (( $# < 3 )); then
  echo "Usage: $0 <array_job_id> <scenario> <architecture>"
  exit 2
fi

ARRAY_JOB_ID="$1"
SCENARIO="$2"
ARCHITECTURE="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_META_DIR="outputs/.runmeta/${ARRAY_JOB_ID}"
RUN_NAME_FILE="${RUN_META_DIR}/run_name.txt"

if [[ ! -f "$RUN_NAME_FILE" ]]; then
  echo "[finalize] No run_name.txt for array ${ARRAY_JOB_ID}. Nothing to do."
  exit 0
fi

RUN_NAME="$(<"$RUN_NAME_FILE")"
RUN_ROOT="outputs/${RUN_NAME}"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "[finalize] Run root missing: ${RUN_ROOT}. Nothing to do."
  exit 0
fi

DONE_MARKER="${RUN_ROOT}/.done"
STATUS_JSON="${RUN_ROOT}/run_finalize_status.json"

COMPLETED_AT="$(date -Iseconds)"

cat >"$STATUS_JSON" <<EOF
{
  "array_job_id": "${ARRAY_JOB_ID}",
  "scenario": "${SCENARIO}",
  "architecture": "${ARCHITECTURE}",
  "run_root": "${RUN_ROOT}",
  "completed_at": "${COMPLETED_AT}"
}
EOF

touch "$DONE_MARKER"

echo "[finalize] Marked run as done: ${RUN_ROOT}"

# Safety: do not auto-commit if user already has staged work.
if [[ -n "$(git diff --cached --name-only)" ]]; then
  echo "[finalize] Staged changes detected. Skipping auto-commit to avoid mixing commits."
  exit 0
fi

git add "$RUN_ROOT"

if git diff --cached --quiet; then
  echo "[finalize] No new git changes for ${RUN_ROOT}."
  exit 0
fi

COMMIT_MSG="auto-backup outputs: ${RUN_NAME} (${SCENARIO}/${ARCHITECTURE}, array ${ARRAY_JOB_ID})"
git commit -m "$COMMIT_MSG"

if git push origin HEAD; then
  echo "[finalize] Backup pushed for ${RUN_NAME}."
else
  echo "[finalize] Commit created locally, but push failed. Please push manually later."
fi
