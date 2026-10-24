#!/usr/bin/env sh
# Full local cycle on the public data, as the organizers run it: adapt on train, eval on test, score.
# Usage: scripts/run_public.sh [run_name]   (set PYTHON to the interpreter with judge's requirements)
set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
RUN="out/${1:-public}"
"$PYTHON" src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir "$RUN/input"
"$PYTHON" -m judge adapt --train "$RUN/input/train.jsonl" --artifacts "$RUN/artifacts"
"$PYTHON" -m judge eval --test "$RUN/input/test.jsonl" --artifacts "$RUN/artifacts" --output "$RUN/output"
"$PYTHON" src/evaluate.py --test data/public_test.csv --predictions "$RUN/output/predictions.csv" --out "$RUN/score.json"
