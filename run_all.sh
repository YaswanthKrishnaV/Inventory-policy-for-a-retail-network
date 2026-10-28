#!/usr/bin/env bash
# Reproduce every number in memo_A.pdf and memo_B.pdf: both pipelines, tests, then the memo PDFs.
#   ./run_all.sh                                    # cutoff d_1885, hold-out = final 13 weeks
#   CUTOFF=1857 HOLDOUT_START=2023-07-03 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
CUTOFF="${CUTOFF:-1885}"
HOLDOUT_START="${HOLDOUT_START:-}"

python -m project_a.run_project_a --cutoff "$CUTOFF"
python -m project_b.run_project_b ${HOLDOUT_START:+--holdout-start "$HOLDOUT_START"}
python -m pytest -q tests

if command -v pandoc >/dev/null && command -v wkhtmltopdf >/dev/null; then
  python memo/build_memo.py
else
  echo "pandoc / wkhtmltopdf not found: skipped rebuilding memo PDFs (memo_A.pdf, memo_B.pdf are committed)"
fi
