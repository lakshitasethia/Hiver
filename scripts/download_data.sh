#!/usr/bin/env bash
# Download the Kaggle "Customer Support on Twitter" dataset and thread it into
# data/conversations.jsonl (the format load_conversations / the pipeline expect).
#
# Prereqs: a Kaggle account + API credentials, any one of:
#   - KAGGLE_API_TOKEN=KGAT_...            (newer access-token style), or
#   - ~/.kaggle/access_token  containing that KGAT_... token (chmod 600), or
#   - ~/.kaggle/kaggle.json   {"username":..., "key":...}   (classic), or
#   - KAGGLE_USERNAME / KAGGLE_KEY env vars
# Or skip auth entirely: download twcs.csv by hand from the dataset page and drop
# it at data/twcs.csv, then just run this script to do the threading step.
set -euo pipefail

# The kaggle CLI reads KAGGLE_API_TOKEN from the environment; also honour the
# ~/.kaggle/access_token file if that is where the token was saved.
if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "${HOME}/.kaggle/access_token" ]]; then
  KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "${HOME}/.kaggle/access_token")"
  export KAGGLE_API_TOKEN
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
DATA_DIR="${ROOT}/data"
RAW="${DATA_DIR}/twcs.csv"
OUT="${DATA_DIR}/conversations.jsonl"
MAX_THREADS="${MAX_THREADS:-40000}"

mkdir -p "${DATA_DIR}"

if [[ ! -f "${RAW}" ]]; then
  if ! "${PY}" -c "import kaggle" 2>/dev/null; then
    echo ">> installing kaggle client into the venv"
    "${ROOT}/.venv/bin/pip" install -q kaggle
  fi
  echo ">> downloading thoughtvector/customer-support-on-twitter (~250 MB zip)"
  "${ROOT}/.venv/bin/kaggle" datasets download -d thoughtvector/customer-support-on-twitter \
    -p "${DATA_DIR}" --unzip
  # the archive contains twcs/twcs.csv
  if [[ -f "${DATA_DIR}/twcs/twcs.csv" ]]; then
    mv "${DATA_DIR}/twcs/twcs.csv" "${RAW}"
    rmdir "${DATA_DIR}/twcs" 2>/dev/null || true
  fi
fi

echo ">> threading conversations (max ${MAX_THREADS}) -> ${OUT}"
"${PY}" - "$RAW" "$OUT" "$MAX_THREADS" <<'PYEOF'
import sys
from support_agent.data.load import load_conversations, dump_jsonl

raw, out, max_threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
convs = load_conversations(raw, max_threads=max_threads)
dump_jsonl(convs, out)
resolved = sum(c.resolved for c in convs)
print(f"wrote {len(convs)} conversations ({resolved} resolved) to {out}")
PYEOF

echo ">> next: make train && make labelset  (then hand-label per docs/labeling-guide.md) && make eval"
