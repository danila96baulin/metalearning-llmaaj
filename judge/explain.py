"""EXPLANATION.md for the jury, rendered only from the artifacts that eval uses."""
import inspect
import json
import re
from collections import defaultdict

from . import check
from .data import TOPICS

MAX_EXAMPLES = 3


def _norm(text):
    return re.sub(r"\s+", " ", text).strip().strip("\"'«»“”").lower()


def verified(quote, row):
    """True if the quote is a verbatim (whitespace-normalized) part of an assistant message."""
    q = _norm(quote)
    text = _norm(" ".join(m["content"] for m in row["conversation"] if m["role"] == "assistant"))
    return len(q) >= 8 and q in text


def _cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def _short(text, limit=220):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def render(rows, rulebook, fallback, audit, manifest):
    by_id = {r["item_id"]: r for r in rows}
    audit_by_id = {a["item_id"]: a for a in audit}
    votes_n = rulebook.get("votes", 1)
    out = ["# How this judge labels conversations", "",
           f"Generated automatically by `adapt` from {len(rows)} labeled training conversations. "
           f"Inducer model: `{manifest['inducer_model']}`. Checker model: `{manifest['checker_model']}`, "
           f"{votes_n} independent runs per conversation.", ""]

    out += ["## 1. Decision procedure and priorities", "",
            "1. Each conversation is checked against the global stop rules and the stop rules of its topic, listed in section 2. "
            "For every rule the checker first decides whether the rule applies, then whether an assistant message violates it.",
            f"2. A rule fires when it applies and is violated in a strict majority of the {votes_n} checker runs.",
            "3. The conversation is BAD if at least one active rule of its topic fires; otherwise it is GOOD. "
            "Strengths of a reply never offset a violation, and a rule's exceptions override its violation test.",
            "4. Rules that flagged at least as many GOOD as BAD training conversations are disabled: "
            "the checker still answers them, but they never decide the label.",
            "5. If a topic has no rules or the model API fails, the topic's majority training label is used: "
            + ", ".join(f"{t} = {fallback[t]}" for t in TOPICS) + ".", ""]

    def table(rules):
        lines = ["| Rule | Applies when | Violation if | Exceptions | BAD caught | GOOD flagged | Status |",
                 "|---|---|---|---|---|---|---|"]
        for r in rules:
            s = r.get("stats", {"tp": "n/a", "fp": "n/a"})
            lines.append(f"| **{r['id']}** {_cell(r['title'])} | {_cell(r['applies_when'])} | {_cell(r['violation_if'])} | "
                         f"{_cell(r['exceptions'] or 'none')} | {s['tp']} | {s['fp']} | {'active' if r['active'] else 'disabled'} |")
        return lines + [""]

    out += ["## 2. Rules", ""]
    if rulebook["global"]["rules"]:
        out += ["### Global rules: checked in every topic", ""] + table(rulebook["global"]["rules"])
    for topic in TOPICS:
        entry = rulebook["topics"][topic]
        topic_rows = [r for r in rows if r["topic"] == topic]
        correct = sum(audit_by_id[r["item_id"]]["prediction"] == r["label"] for r in topic_rows)
        out += [f"### {topic}: {correct}/{len(topic_rows)} training conversations reproduced", ""]
        if entry["status"] != "llm":
            out += [f"No rules: every conversation gets the fallback label `{fallback[topic]}`.", ""]
            continue
        out += table(entry["rules"]) if entry["rules"] else ["Only the global rules apply.", ""]

    out += ["## 3. Training examples for each rule", "",
            "Quotes are checked against the conversation text; ✓ marks a verbatim match. Labels are the expert's.", ""]
    scopes = [("all", rulebook["global"]["rules"])] + [(t, rulebook["topics"][t]["rules"]) for t in TOPICS]
    for scope, scope_rules in scopes:
        for r in scope_rules:
            out.append(f"**{r['id']}** {r['title']}")
            hits, clean = [], []
            for a in audit:
                if scope != "all" and a["topic"] != scope:
                    continue
                _, fired = check.decide(a["votes"], [r["id"]])
                checks = [v[r["id"]] for v in a["votes"] if r["id"] in v]
                if fired and a["label"] == "bad":
                    evidence = next((c["evidence"] for c in checks if c["violated"] and c["evidence"]), "")
                    hits.append((a["item_id"], evidence))
                elif a["label"] == "good" and checks and all(c["applies"] and not c["violated"] for c in checks):
                    clean.append(a["item_id"])
            for item_id, evidence in hits[:MAX_EXAMPLES]:
                mark = "✓" if verified(evidence, by_id[item_id]) else "description"
                out.append(f"- BAD `{item_id}`: \"{_short(evidence)}\" ({mark})")
            if not hits:
                for e in r["examples"][:MAX_EXAMPLES]:
                    out.append(f"- `{e['item_id']}` (expert: {by_id[e['item_id']]['label']}), cited at induction: \"{_short(e['quote'])}\"")
            if clean:
                out.append(f"- GOOD `{clean[0]}`: the rule applies and is respected.")
            out.append("")

    out += ["## 4. Boundaries", ""]
    errors = [a for a in audit if a["prediction"] != a["label"]]
    out += [f"Training conversations the rules still misjudge: {len(errors)} of {len(audit)}.", ""]
    if errors:
        out += ["| Conversation | Topic | Expert | Judge | Fired rules |", "|---|---|---|---|---|"]
        out += [f"| `{a['item_id']}` | {a['topic']} | {a['label']} | {a['prediction']} | {', '.join(a['fired']) or 'none'} |" for a in errors]
        out.append("")
    split = [a["item_id"] for a in audit if len(a["votes"]) > 1 and
             len({tuple(sorted(rid for rid, c in v.items() if c["applies"] and c["violated"])) for v in a["votes"]}) > 1]
    out += [f"Conversations where checker runs disagreed: {len(split)}"
            + (f", for example {', '.join(f'`{i}`' for i in split[:8])}." if split else "."), ""]
    disabled = [(s, r) for s, scope_rules in scopes for r in scope_rules if not r["active"]]
    if disabled:
        out += ["Disabled rules, because they flagged as many GOOD conversations as BAD:", ""]
        out += [f"- {r['id']} {r['title']}: caught {r['stats']['tp']} BAD, flagged {r['stats']['fp']} GOOD" for _, r in disabled]
        out.append("")
    groups = defaultdict(list)
    for r in rows:
        groups[json.dumps(r["context"], sort_keys=True)].append(r)
    mixed = [g for g in groups.values() if len({r["label"] for r in g}) > 1]
    if mixed:
        out += ["Same context, different expert labels: the decisive difference is in the assistant's replies.", ""]
        out += ["- " + ", ".join(f"`{r['item_id']}` ({r['label']})" for r in g) for g in mixed[:8]]
        out.append("")

    out += ["## 5. Consistency with the implementation", "",
            f"- `rulebook.json` sha256: `{manifest['rulebook_sha256']}`.",
            *[f"- `prompts/checker_{t}.txt` sha256: `{h}`" for t, h in manifest["prompt_sha256"].items()],
            "- `eval` loads exactly these prompt files, checks their hashes, and labels with the `decide()` function below.",
            f"- Replay: the same `decide()` applied to the stored training votes in `train_audit.jsonl` reproduces "
            f"{sum(a['prediction'] == a['label'] for a in audit)}/{len(audit)} expert labels; per topic, see section 2.", "",
            "```python", inspect.getsource(check.decide).rstrip(), "```", ""]
    return "\n".join(out)
