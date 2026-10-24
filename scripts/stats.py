"""Dev tool: confidence intervals for one archived run and paired tests between two runs.

Usage:
  python scripts/stats.py RUN [--split test|train|cv]          per-topic Wilson CI, bootstrap CI for macro, confusion
  python scripts/stats.py RUN_A RUN_B [--split test|train|cv]  exact McNemar and paired bootstrap of the macro difference
RUN may be a unique prefix or suffix of a run_id. Reads history.csv from $RUNS_ARCHIVE (default ../plata-expert-judge/runs).
Bootstrap resamples within each topic, because every topic has equal weight in macro_topic_accuracy.
"""
import argparse
import csv
import math
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ["credit", "financial_position", "savings", "spending"]
RESAMPLES = 10000


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n; stays inside [0, 1] on small samples."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def macro(items):
    """items: (topic, correct) pairs; mean of per-topic accuracies over the topics present."""
    by_topic = {}
    for topic, correct in items:
        by_topic.setdefault(topic, []).append(correct)
    return sum(sum(v) / len(v) for v in by_topic.values()) / len(by_topic) if by_topic else 0.0


def _strata(rows):
    by_topic = {}
    for i, r in enumerate(rows):
        by_topic.setdefault(r["topic"], []).append(i)
    return list(by_topic.values())


def bootstrap_macro_ci(rows, resamples=RESAMPLES, seed=0):
    """95% percentile interval for macro accuracy; rows need topic, expert, judge."""
    rng = random.Random(seed)
    strata = _strata(rows)
    correct = [r["expert"] == r["judge"] for r in rows]
    scores = sorted(sum(sum(correct[rng.choice(s)] for _ in s) / len(s) for s in strata) / len(strata)
                    for _ in range(resamples))
    return scores[int(0.025 * resamples)], scores[int(0.975 * resamples) - 1]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: b items only A got right, c items only B got right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(rows_a, rows_b, resamples=RESAMPLES, seed=0):
    """Macro difference B - A on the same items: 95% interval and the share of resamples where B is better."""
    rng = random.Random(seed)
    b_by_id = {r["item_id"]: r for r in rows_b}
    rows = [r for r in rows_a if r["item_id"] in b_by_id]
    strata = _strata(rows)
    delta = [(b_by_id[r["item_id"]]["judge"] == r["expert"]) - (r["judge"] == r["expert"]) for r in rows]
    diffs = sorted(sum(sum(delta[rng.choice(s)] for _ in s) / len(s) for s in strata) / len(strata)
                   for _ in range(resamples))
    return diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1], sum(d > 0 for d in diffs) / resamples


def unstable_share(rows):
    """Share of items whose checker votes disagreed, among items with more than one vote."""
    multi = [r for r in rows if str(r.get("votes_total", "")).isdigit() and int(r["votes_total"]) > 1]
    split = [r for r in multi if 0 < int(r["votes_bad"]) < int(r["votes_total"])]
    return len(split) / len(multi) if multi else None


def load(archive, run, split):
    with (archive / "history.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == split]
    ids = sorted({r["run_id"] for r in rows})
    matches = [i for i in ids if i == run] or [i for i in ids if i.startswith(run) or i.endswith(run)]
    if len(matches) != 1:
        raise SystemExit(f"run '{run}' matches {matches or 'nothing'} in split '{split}'; known runs: {', '.join(ids)}")
    return matches[0], [r for r in rows if r["run_id"] == matches[0]]


def describe(run_id, rows):
    print(f"{run_id}: {len(rows)} items")
    for topic in TOPICS:
        t = [r for r in rows if r["topic"] == topic]
        if not t:
            continue
        k = sum(r["expert"] == r["judge"] for r in t)
        lo, hi = wilson(k, len(t))
        cm = {(e, j): sum(r["expert"] == e and r["judge"] == j for r in t) for e in ("good", "bad") for j in ("good", "bad")}
        print(f"  {topic:<19} {k:>3}/{len(t):<3} acc {k / len(t):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"expert good->judge bad {cm['good', 'bad']}, expert bad->judge good {cm['bad', 'good']}")
    lo, hi = bootstrap_macro_ci(rows)
    print(f"  macro {macro((r['topic'], r['expert'] == r['judge']) for r in rows):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    statuses = {}
    for r in rows:
        statuses[r.get("status") or "unknown"] = statuses.get(r.get("status") or "unknown", 0) + 1
    share = unstable_share(rows)
    print(f"  status {statuses}" + (f", unstable votes {share:.1%}" if share is not None else ""))


def compare(a, rows_a, b, rows_b):
    b_by_id = {r["item_id"]: r for r in rows_b}
    common = [r for r in rows_a if r["item_id"] in b_by_id]
    only_a = sum(r["judge"] == r["expert"] != b_by_id[r["item_id"]]["judge"] for r in common)
    only_b = sum(b_by_id[r["item_id"]]["judge"] == r["expert"] != r["judge"] for r in common)
    flips = sum(r["judge"] != b_by_id[r["item_id"]]["judge"] for r in common)
    macro_a = macro((r["topic"], r["judge"] == r["expert"]) for r in common)
    macro_b = macro((r["topic"], b_by_id[r["item_id"]]["judge"] == r["expert"]) for r in common)
    lo, hi, p_better = paired_bootstrap(common, rows_b)
    print(f"A = {a}\nB = {b}\n{len(common)} common items, labels differ on {flips}")
    print(f"  macro A {macro_a:.4f}, B {macro_b:.4f}, B - A {macro_b - macro_a:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
          f"B better in {p_better:.1%} of resamples")
    print(f"  only A right {only_a}, only B right {only_b}, exact McNemar p = {mcnemar_exact(only_a, only_b):.3f}")
    for topic in TOPICS:
        t = [r for r in common if r["topic"] == topic]
        if t:
            ra = sum(r["judge"] == r["expert"] for r in t)
            rb = sum(b_by_id[r["item_id"]]["judge"] == r["expert"] for r in t)
            print(f"  {topic:<19} A {ra}/{len(t)}, B {rb}/{len(t)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="test", choices=["test", "train", "cv"])
    args = parser.parse_args()
    archive = Path(os.getenv("RUNS_ARCHIVE", ROOT.parent / "plata-expert-judge" / "runs"))
    loaded = [load(archive, run, args.split) for run in args.runs[:2]]
    if len(loaded) == 1:
        describe(*loaded[0])
    else:
        compare(*loaded[0], *loaded[1])


if __name__ == "__main__":
    main()
