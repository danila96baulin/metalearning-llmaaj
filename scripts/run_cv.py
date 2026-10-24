"""Dev tool: k-fold cross-validation of the full adapt -> eval cycle on the labeled public data.

Usage: python scripts/run_cv.py [run_name] [--pool all|train] [--folds-file data/public_folds.csv] [--no-archive]
Each fold: adapt on the other folds, eval on this one, exactly as the organizers run the two stages.
Folds come from data/public_folds.csv, stratified by topic and label; the file is created once if missing and
must stay fixed, so that runs are compared on the same splits. --pool all uses train and test (230 items),
--pool train only the 150 training items and keeps public_test untouched.
Writes out/<run>/fold_<k>/... and out/<run>/cv_predictions.csv, then archives the run with archive_run.py --cv.
Set INPUT_LANGUAGE (original, es, en) to choose the translation.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS = 5
csv.field_size_limit(sys.maxsize)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, header)
        writer.writeheader()
        writer.writerows(rows)


def make_folds(rows, k=FOLDS):
    """Stratified by topic and label: within each stratum, items sorted by a hash of their id are dealt round robin."""
    folds = {}
    strata = sorted({(r["topic"], r["label"]) for r in rows})
    offset = 0
    for stratum in strata:
        items = sorted((r for r in rows if (r["topic"], r["label"]) == stratum),
                       key=lambda r: hashlib.sha256(r["item_id"].encode()).hexdigest())
        for i, r in enumerate(items):
            folds[r["item_id"]] = (offset + i) % k
        offset += len(items)  # continue dealing where the last stratum stopped, so fold sizes stay even
    return folds


def run(*args, env=None):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="cv")
    parser.add_argument("--pool", choices=["all", "train"], default="all")
    parser.add_argument("--folds-file", default="data/public_folds.csv")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    header, rows = read_csv(ROOT / "data" / "public_train.csv")
    if args.pool == "all":
        rows += read_csv(ROOT / "data" / "public_test.csv")[1]
    folds_file = ROOT / args.folds_file
    if not folds_file.exists():
        _, every = read_csv(ROOT / "data" / "public_train.csv")
        every += read_csv(ROOT / "data" / "public_test.csv")[1]
        folds = make_folds(every)
        write_csv(folds_file, ["item_id", "fold"], [{"item_id": i, "fold": f} for i, f in sorted(folds.items())])
        print(f"created {folds_file}: commit it, so the whole team uses the same folds")
    fold_of = {r["item_id"]: int(r["fold"]) for r in read_csv(folds_file)[1]}
    out = ROOT / "out" / args.name
    language = os.getenv("INPUT_LANGUAGE", "original")
    results = []
    for k in sorted({fold_of[r["item_id"]] for r in rows}):
        fold = out / f"fold_{k}"
        train = [r for r in rows if fold_of[r["item_id"]] != k]
        test = [r for r in rows if fold_of[r["item_id"]] == k]
        print(f"fold {k}: adapt on {len(train)}, eval on {len(test)}", flush=True)
        write_csv(fold / "train.csv", header, train)
        write_csv(fold / "test.csv", header, test)
        run("src/prepare_inputs.py", "--train", fold / "train.csv", "--test", fold / "test.csv",
            "--out-dir", fold / "input", "--language", language)
        run("-m", "judge", "adapt", "--train", fold / "input" / "train.jsonl", "--artifacts", fold / "artifacts")
        env = {**os.environ, "JUDGE_TRACE_FILE": str(fold / "output" / "trace.jsonl")}
        run("-m", "judge", "eval", "--test", fold / "input" / "test.jsonl", "--artifacts", fold / "artifacts",
            "--output", fold / "output", env=env)
        _, pred = read_csv(fold / "output" / "predictions.csv")
        trace_path = fold / "output" / "trace.jsonl"  # missing if eval kept fallback labels only
        trace = {t["item_id"]: t for t in map(json.loads, trace_path.open(encoding="utf-8"))} if trace_path.exists() else {}
        expert = {r["item_id"]: r for r in test}
        results += [{"item_id": p["item_id"], "topic": expert[p["item_id"]]["topic"], "expert": expert[p["item_id"]]["label"],
                     "judge": p["label"], "status": trace.get(p["item_id"], {}).get("status", ""),
                     "votes_bad": trace.get(p["item_id"], {}).get("votes_bad", ""),
                     "votes_total": trace.get(p["item_id"], {}).get("votes_total", ""),
                     "fired": " ".join(trace.get(p["item_id"], {}).get("fired", [])), "fold": k} for p in pred]
    write_csv(out / "cv_predictions.csv",
              ["item_id", "topic", "expert", "judge", "status", "votes_bad", "votes_total", "fired", "fold"], results)
    topics = sorted({r["topic"] for r in results})
    accs = {t: sum(r["expert"] == r["judge"] for r in results if r["topic"] == t) / sum(r["topic"] == t for r in results)
            for t in topics}
    print(f"cv macro {sum(accs.values()) / len(accs):.4f}: " + ", ".join(f"{t} {a:.3f}" for t, a in accs.items()))
    if not args.no_archive:
        run("scripts/archive_run.py", out, "--cv")


if __name__ == "__main__":
    main()
