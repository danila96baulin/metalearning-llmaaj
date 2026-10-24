"""Dev tool: estimate the jury score I (0-5) of a run's EXPLANATION.md and the task's Final = 10 * acc + 0.4 * I.

Usage: python scripts/score_explanation.py out/<run> [out/<run> ...]
Each jury criterion is checked mechanically; this is a lower-bound sanity check, not the jury's opinion.
"""
import json
import re
import sys
from pathlib import Path


def criteria(run):
    art = run / "artifacts"
    text = (art / "EXPLANATION.md").read_text(encoding="utf-8")
    rulebook = json.loads((art / "rulebook.json").read_text(encoding="utf-8"))
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    active = [r for r in rulebook["global"]["rules"] + [r for t in rulebook["topics"].values() for r in t["rules"]] if r["active"]]
    examples = re.search(r"## 3\..*?(?=\n## 4\.)", text, re.S)
    examples = examples.group(0) if examples else ""
    with_quote = [r for r in active if re.search(rf"\*\*{re.escape(r['id'])}\*\*[^\n]*\n(?:- [^\n]*\n)*?- BAD [^\n]*✓", examples)]
    boundaries = re.search(r"## 4\..*?(?=\n## 5\.)", text, re.S)
    boundaries = boundaries.group(0) if boundaries else ""
    hashes_ok = manifest["rulebook_sha256"] in text and all(h in text for h in manifest["prompt_sha256"].values())
    return {
        "clear criteria: active rules with applies-when and violation-if": bool(active) and "| Rule | Applies when | Violation if" in text,
        "clear priorities: decision procedure section": "## 1. Decision procedure" in text and "strict majority" in text,
        "links to training examples: every active rule has a verified quote": bool(active) and len(with_quote) == len(active),
        "clear boundaries: concrete items with conversation ids": len(re.findall(r"`[0-9a-f]{8,}", boundaries)) >= 3,
        "consistency: hashes, decide() source and replay": hashes_ok and "def decide(" in text and "Replay:" in text,
    }, len(with_quote), len(active)


def main():
    print("| Run | acc | I (estimate) | Final = 10·acc + 0.4·I | rules with a verified quote |")
    print("|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        run = Path(arg)
        acc = json.loads((run / "score.json").read_text())["macro_topic_accuracy"]
        checks, quoted, active = criteria(run)
        i = sum(checks.values())
        print(f"| {run.name} | {acc:.3f} | {i}/5 | {10 * acc + 0.4 * i:.2f} | {quoted}/{active} |")
        for name, ok in checks.items():
            if not ok:
                print(f"|  | ✗ {name} | | | |")


if __name__ == "__main__":
    main()
