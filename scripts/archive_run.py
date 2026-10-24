"""Dev tool: archive a finished run so the team can track how the judge improves over time.

Usage: python scripts/archive_run.py out/<run> [--baseline]
Writes to $RUNS_ARCHIVE (default ../plata-expert-judge/runs):
  <run_id>/test.csv, <run_id>/train.csv  item_id, topic, expert label, judge label
  <run_id>/meta.json                     time, git commit, configuration, scores
  history.csv                            one row per run and conversation, appended
  summary.csv                            one row per run, appended
"""
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import TOPICS, load_dataset  # noqa: E402

CONFIG_VARS = ["LLM_PROVIDER", "INDUCER_MODEL", "CHECKER_MODEL", "INDUCER_THINKING_LEVEL", "CHECKER_THINKING_LEVEL",
               "VOTES", "REFINE_ROUNDS", "ONLY_TOPICS"]


def append(path, header, rows):
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(header)
        writer.writerows(rows)


def main():
    run = Path(sys.argv[1])
    kind = "baseline" if "--baseline" in sys.argv else "judge"
    archive = Path(os.getenv("RUNS_ARCHIVE", ROOT.parent / "plata-expert-judge" / "runs"))
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_id = f"{stamp}_{run.name}"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    expert = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_test.csv")}
    with (run / "output" / "predictions.csv").open(encoding="utf-8") as stream:
        test = [(r["item_id"], expert[r["item_id"]]["topic"], expert[r["item_id"]]["label"], r["label"]) for r in csv.DictReader(stream)]
    train = []
    audit = run / "artifacts" / "train_audit.jsonl"
    if audit.exists():
        train = [(a["item_id"], a["topic"], a["label"], a["prediction"]) for a in map(json.loads, audit.open(encoding="utf-8"))]

    def macro(rows):
        accs = [sum(e == j for _, t, e, j in rows if t == topic) / max(1, sum(t == topic for _, t, *_ in rows)) for topic in TOPICS]
        return round(sum(accs) / len(accs), 4), [round(a, 4) for a in accs]

    out = archive / run_id
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test", test), ("train", train)):
        if rows:
            with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows([("item_id", "topic", "expert", "judge"), *rows])
    test_macro, test_topics = macro(test)
    train_macro = macro(train)[0] if train else ""
    config = {k: os.getenv(k, "") for k in CONFIG_VARS}
    manifest = json.loads((run / "artifacts" / "manifest.json").read_text()) if (run / "artifacts" / "manifest.json").exists() else {}
    meta = {"run_id": run_id, "kind": kind, "commit": commit, "config": config, "manifest": manifest,
            "test_macro_accuracy": test_macro, "test_by_topic": dict(zip(TOPICS, test_topics)), "train_macro_accuracy": train_macro}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    append(archive / "history.csv", ["run_id", "kind", "split", "item_id", "topic", "expert", "judge"],
           [(run_id, kind, split, *row) for split, rows in (("test", test), ("train", train)) for row in rows])
    append(archive / "summary.csv",
           ["run_id", "kind", "commit", "checker_model", "inducer_model", "votes", "only_topics", "test_macro",
            *[f"test_{t}" for t in TOPICS], "train_macro"],
           [(run_id, kind, commit, manifest.get("checker_model") or config["CHECKER_MODEL"], manifest.get("inducer_model", ""),
             manifest.get("votes", ""), config["ONLY_TOPICS"], test_macro, *test_topics, train_macro)])
    print(f"archived {run_id}: test macro {test_macro}, train macro {train_macro or 'n/a'} -> {out}")


if __name__ == "__main__":
    main()
