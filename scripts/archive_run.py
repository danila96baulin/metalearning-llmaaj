"""Dev tool: archive a finished run so the team can track how the judge improves over time.

Usage: python scripts/archive_run.py out/<run> [--baseline | --cv]
Writes to $RUNS_ARCHIVE (default ../plata-expert-judge/runs):
  <run_id>/test.csv, train.csv or cv.csv  per item: expert and judge label, status, vote split, fired rules, fold
  <run_id>/meta.json                      time, git commit, effective configuration, model call counters, scores
  <run_id>/artifacts/                     rulebook, prompts, EXPLANATION.md, train audit, eval trace (cv: per fold)
  history.csv                             one row per run and conversation, appended
  summary.csv                             one row per run, appended
Older history.csv and summary.csv files are migrated to new columns in place; missing values stay empty.
Set INPUT_LANGUAGE (original, es, en) and REPEAT (repeat number of the same configuration) when running.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from common import TOPICS, load_dataset  # noqa: E402
from judge import config as judge_config  # noqa: E402
from stats import bootstrap_macro_ci, macro, unstable_share  # noqa: E402

ITEM_FIELDS = ["item_id", "topic", "expert", "judge", "status", "votes_bad", "votes_total", "fired", "fold"]
HISTORY = ["run_id", "kind", "split", *ITEM_FIELDS]
SUMMARY = ["run_id", "kind", "commit", "dirty", "language", "repeat", "checker_model", "inducer_model", "votes",
           "only_topics", "eval_split", "test_macro", "test_ci_low", "test_ci_high", *[f"test_{t}" for t in TOPICS],
           "train_macro", "n_fallback", "unstable_share", "cache_hit_rate"]
COPIED = ["rulebook.json", "manifest.json", "EXPLANATION.md", "boundaries.json", "train_audit.jsonl", "fallback.json", "prompts"]


def append(path, header, rows):
    """Append dict rows; if the file has other columns, rewrite it under the new header first."""
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            old = list(reader)
        if reader.fieldnames != header:
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(old)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, header, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def item(row, expert, trace=None, fold=""):
    trace = trace or {}
    return {"item_id": row["item_id"], "topic": expert["topic"], "expert": expert["label"], "judge": row["label"],
            "status": trace.get("status", ""), "votes_bad": trace.get("votes_bad", ""),
            "votes_total": trace.get("votes_total", ""), "fired": " ".join(trace.get("fired", [])), "fold": fold}


def scores(rows):
    accs = []
    for topic in TOPICS:
        t = [r for r in rows if r["topic"] == topic]
        accs.append(round(sum(r["expert"] == r["judge"] for r in t) / len(t), 4) if t else "")
    return round(macro((r["topic"], r["expert"] == r["judge"]) for r in rows), 4), accs


def copy_artifacts(src, dst):
    for name in COPIED:
        path = src / name
        if path.is_dir():
            shutil.copytree(path, dst / name, dirs_exist_ok=True)
        elif path.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst / name)


def collect(run, kind, expert):
    """(split, item rows) pairs and, per part of the run (one per fold for cv), its artifacts and eval output dirs."""
    if kind == "cv":
        with (run / "cv_predictions.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        folds = sorted(p for p in run.glob("fold_*") if p.is_dir())
        return [("cv", rows)], {f.name: (f / "artifacts", f / "output") for f in folds}
    output = run / "output" if (run / "output" / "predictions.csv").exists() else run  # the baseline writes into the run folder
    trace = {t["item_id"]: t for t in read_jsonl(output / "trace.jsonl")}
    with (output / "predictions.csv").open(encoding="utf-8") as stream:
        test = [item(r, expert[r["item_id"]], trace.get(r["item_id"])) for r in csv.DictReader(stream)]
    train = [item({"item_id": a["item_id"], "label": a["prediction"]}, {"topic": a["topic"], "label": a["label"]}, a)
             for a in read_jsonl(run / "artifacts" / "train_audit.jsonl")]
    return [("test", test)] + ([("train", train)] if train else []), {"": (run / "artifacts", output)}


def main():
    run = Path(sys.argv[1])
    kind = "cv" if "--cv" in sys.argv else "baseline" if "--baseline" in sys.argv else "judge"
    archive = Path(os.getenv("RUNS_ARCHIVE", ROOT.parent / "plata-expert-judge" / "runs"))
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{run.name}"
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    expert = {r["item_id"]: r for name in ("public_train", "public_test") for r in load_dataset(ROOT / "data" / f"{name}.csv")}
    splits, parts = collect(run, kind, expert)

    out = archive / run_id
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits:
        with (out / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, ITEM_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    for name, (artifacts, output) in parts.items():
        copy_artifacts(artifacts, out / "artifacts" / name)
        for extra in ("trace.jsonl", "trace_stats.json"):
            if (output / extra).exists():
                (out / "eval" / name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / extra, out / "eval" / name / extra)
    if (run / "score.json").exists():
        shutil.copy2(run / "score.json", out / "score.json")

    held_split, held = splits[0]
    test_macro, test_topics = scores(held)
    ci_low, ci_high = bootstrap_macro_ci(held)
    train = dict(splits).get("train")
    manifests = [json.loads(p.read_text()) for a, _ in parts.values() if (p := a / "manifest.json").exists()]
    eval_stats = [json.loads(p.read_text()) for _, o in parts.values() if (p := o / "trace_stats.json").exists()]
    calls = [m["llm"] for m in manifests if m.get("llm")] + [s["llm"] for s in eval_stats if s.get("llm")]
    total_calls = sum(c["calls"] + c["cache_hits"] for c in calls)
    # models as the run actually used them: adapt's manifest leaves them empty when it had no model access
    first = manifests[0] if manifests else {}
    if kind == "baseline":
        models = {"provider": judge_config.PROVIDER, "inducer_model": None, "checker_model": judge_config.CHECKER_MODEL,
                  "inducer_thinking_level": None, "checker_thinking_level": judge_config.CHECKER_THINKING_LEVEL or "default",
                  "votes": None}
    else:
        models = {k: first.get(k) for k in ("provider", "inducer_model", "checker_model", "inducer_thinking_level",
                                             "checker_thinking_level", "votes")}
    effective = {
        **models, "refine_rounds": judge_config.REFINE_ROUNDS if models["inducer_model"] else None,
        "only_topics": judge_config.ONLY_TOPICS,
        "language": os.getenv("INPUT_LANGUAGE", "original"), "repeat": int(os.getenv("REPEAT", "1")),
        "llm_cache": bool(judge_config.CACHE_DIR)}
    meta = {"run_id": run_id, "kind": kind, "commit": commit, "dirty": dirty, "config": effective,
            "manifest": manifests[0] if len(manifests) == 1 else manifests, "eval_stats": eval_stats,
            "eval_split": held_split, "test_macro_accuracy": test_macro, "test_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "test_by_topic": dict(zip(TOPICS, test_topics)), "train_macro_accuracy": scores(train)[0] if train else None,
            "unstable_share": unstable_share(held),
            "cache_hit_rate": round(sum(c["cache_hits"] for c in calls) / total_calls, 4) if total_calls else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    append(archive / "history.csv", HISTORY,
           [{"run_id": run_id, "kind": kind, "split": split, **r} for split, rows in splits for r in rows])
    share = meta["unstable_share"]
    append(archive / "summary.csv", SUMMARY, [{
        "run_id": run_id, "kind": kind, "commit": commit[:7], "dirty": dirty, "language": effective["language"],
        "repeat": effective["repeat"], "checker_model": effective["checker_model"] or "",
        "inducer_model": effective["inducer_model"] or "", "votes": effective["votes"] or "",
        "only_topics": ",".join(effective["only_topics"]), "eval_split": held_split, "test_macro": test_macro,
        "test_ci_low": meta["test_ci95"][0], "test_ci_high": meta["test_ci95"][1],
        **{f"test_{t}": a for t, a in zip(TOPICS, test_topics)}, "train_macro": meta["train_macro_accuracy"] or "",
        "n_fallback": sum(r["status"].startswith("fallback") for r in held),
        "unstable_share": round(share, 4) if share is not None else "", "cache_hit_rate": meta["cache_hit_rate"] or ""}])
    print(f"archived {run_id}: {held_split} macro {test_macro} [{ci_low:.3f}, {ci_high:.3f}], "
          f"train macro {meta['train_macro_accuracy'] or 'n/a'}{', DIRTY working tree' if dirty else ''} -> {out}")


if __name__ == "__main__":
    main()
