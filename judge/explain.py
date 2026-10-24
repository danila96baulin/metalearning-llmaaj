"""EXPLANATION.md for the jury, rendered only from the artifacts that eval uses."""
import inspect
import json
import re
from collections import defaultdict

from . import check
from .data import TOPICS
from .rules import guidance_text

MAX_EXAMPLES = 3
MAX_ERRORS = 15


def _norm(text):
    return re.sub(r"\s+", " ", text).strip().strip("\"'«»“”").lower()


def verified(evidence, row):
    """True if the evidence, or a quoted span inside it, is a verbatim (whitespace-normalized) part of an assistant message."""
    text = _norm(" ".join(m["content"] for m in row["conversation"] if m["role"] == "assistant"))
    spans = [evidence] + re.findall(r"[«“\"]([^»”\"]{8,})[»”\"]", evidence)
    return any(len(_norm(s)) >= 8 and _norm(s).rstrip(".…") in text for s in spans)


def _cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def _short(text, limit=220):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _reply(row, limit=160):
    """The last assistant message, shortened: what the expert judged."""
    replies = [m["content"] for m in row["conversation"] if m["role"] == "assistant"]
    return _short(replies[-1], limit) if replies else ""


def _why(a, row):
    """Evidence of the fired rules for a false BAD, the assistant's reply for a missed BAD."""
    if a.get("status", "ok") != "ok":
        return "fallback label: the model did not judge this conversation"
    if not a["fired"]:
        return f"no rule fired; reply: \"{_cell(_reply(row))}\""
    parts = []
    for rid in a["fired"]:
        evidence = next((v[rid]["evidence"] for v in a["votes"] if v.get(rid, {}).get("violated") and v[rid]["evidence"]), "")
        parts.append(f"{rid}: \"{_cell(_short(evidence, 120))}\"" if evidence else rid)
    return "; ".join(parts)


def render(rows, rulebook, fallback, audit, manifest, boundaries=None):
    by_id = {r["item_id"]: r for r in rows}
    audit_by_id = {a["item_id"]: a for a in audit}
    votes_n = rulebook.get("votes", 1)
    out = ["# How this judge labels conversations", "",
           f"Generated automatically by `adapt` from {len(rows)} labeled training conversations. "
           f"Inducer model: `{manifest['inducer_model']}`. Checker model: `{manifest['checker_model']}`, "
           f"{votes_n} independent runs per conversation.", ""]

    out += ["## 1. Decision procedure and priorities", "",
            "1. Each conversation is checked against the global stop rules and the stop rules of its topic, listed in section 2. "
            "The checker also receives arithmetic computed in code from the context (sums, averages, interest, balances), "
            "with every number in the assistant's replies matched to the closest computed value. "
            "For every rule the checker first decides whether the rule applies, then whether an assistant message violates it.",
            f"2. A rule fires when it applies and is violated in a strict majority of the {votes_n} checker runs.",
            "3. The conversation is BAD if at least one active rule of its topic fires; otherwise it is GOOD. "
            "Strengths of a reply never offset a violation, and a rule's exceptions override its violation test.",
            "4. Safety net OTHER: the checker also reports a clear error that no rule covers (data contradicting the context, "
            "an unanswered question, an invented fact). It fires only if all runs agree, and it is used only if it improved "
            f"training accuracy: {'active' if rulebook['other']['active'] else 'disabled'}"
            + (f", it caught {rulebook['other']['new_bad']} more BAD and {rulebook['other']['new_good']} more GOOD "
               "training conversations than the rules alone." if 'new_bad' in rulebook['other'] else "."),
            "5. A rule is disabled if it flagged at least as many GOOD as BAD training conversations, or if it is a topic rule "
            f"that caught fewer than {rulebook.get('min_support', 1)} BAD training conversations: a rule built on a single "
            "example is a hypothesis. A global rule that flagged no GOOD conversation stays active as common sense even "
            "without a caught BAD. The checker still answers disabled rules, but they never decide the label.",
            "6. If a topic has no rules or the model API fails, the topic's majority training label is used: "
            + ", ".join(f"{t} = {fallback[t]}" for t in TOPICS) + ".", ""]

    def table(rules):
        lines = ["| Rule | Applies when | Violation if | Exceptions | BAD caught | GOOD flagged | Status |",
                 "|---|---|---|---|---|---|---|"]
        for r in rules:
            s = r.get("stats", {"tp": "n/a", "fp": "n/a"})
            lines.append(f"| **{r['id']}** {_cell(r['title'])} | {_cell(r['applies_when'])} | {_cell(r['violation_if'])} | "
                         f"{_cell(r['exceptions'] or 'none')} | {s['tp']} | {s['fp']} | {'active' if r['active'] else 'disabled'} |")
        return lines + [""]

    out += ["## 2. Rules", "",
            "Training accuracy below is in-sample: the rules were induced and refined on these same conversations, "
            "so it overestimates accuracy on new conversations.", ""]
    budget = rulebook.get("rule_budget")
    if budget:
        out += [f"Rule budget given to the inducer, growing with the square root of the training data: at most "
                f"{budget['global']} global rules, and per topic " + ", ".join(f"{t} {n}" for t, n in budget["topics"].items()) + ".", ""]
    if rulebook["global"]["rules"]:
        out += ["### Global rules: checked in every topic", ""] + table(rulebook["global"]["rules"])
    for topic in TOPICS:
        entry = rulebook["topics"][topic]
        topic_rows = [r for r in rows if r["topic"] == topic]
        correct = sum(audit_by_id[r["item_id"]]["prediction"] == r["label"] for r in topic_rows)
        out += [f"### {topic}: {correct}/{len(topic_rows)} training conversations reproduced (in-sample)", ""]
        if entry["status"] != "llm":
            out += [f"No rules: every conversation gets the fallback label `{fallback[topic]}`.", ""]
            continue
        out += table(entry["rules"]) if entry["rules"] else ["Only the global rules apply.", ""]
        if entry.get("global_exceptions"):
            out += [f"Exceptions to global rules that hold in {topic} only, added by refinement:", ""]
            out += [f"- **{rid}**: {_cell(text)}" for rid, text in entry["global_exceptions"].items()] + [""]

    out += ["## 3. Training examples for each rule", "",
            "Quotes are checked against the conversation text; ✓ marks a verbatim match. Labels are the expert's.", ""]
    other = rulebook["other"]
    other_rule = {"id": other["id"], "title": other["title"], "active": other["active"],
                  "stats": {"tp": other.get("tp", "n/a"), "fp": other.get("fp", "n/a")}, "examples": []}
    scopes = ([("all", rulebook["global"]["rules"])] + [(t, rulebook["topics"][t]["rules"]) for t in TOPICS]
              + [("all", [other_rule])])
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
    for group in boundaries or []:
        ids = [i for i in group["item_ids"] if i in by_id]
        out.append(f"- **{group['kind'].replace('_', ' ')}**: {group['pattern']} "
                   + ", ".join(f"`{i}` (expert: {by_id[i]['label']})" for i in ids))
    if boundaries:
        out.append("")
    if errors:
        out += ["| Conversation | Topic | Expert | Judge | Why the judge disagrees |", "|---|---|---|---|---|"]
        out += [f"| `{a['item_id']}` | {a['topic']} | {a['label']} | {a['prediction']} | {_why(a, by_id[a['item_id']])} |"
                for a in errors[:MAX_ERRORS]]
        if len(errors) > MAX_ERRORS:
            out.append(f"\n{len(errors) - MAX_ERRORS} more are listed in `train_audit.jsonl`.")
        out.append("")
    split = [a["item_id"] for a in audit if len(a["votes"]) > 1 and
             len({tuple(sorted(rid for rid, c in v.items() if c["applies"] and c["violated"])) for v in a["votes"]}) > 1]
    out += [f"Conversations where checker runs disagreed: {len(split)}"
            + (f", for example {', '.join(f'`{i}`' for i in split[:8])}." if split else "."), ""]
    disabled = [(s, r) for s, scope_rules in scopes for r in scope_rules if not r["active"]]
    if disabled:
        out += ["Disabled rules (section 1, item 5):", ""]
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
            "```python", inspect.getsource(check.decide).rstrip(), "```", "",
            "Hand-written guidance: written by the team, not induced from the data. It is added to the induce, refine "
            "and checker prompts, so it is part of the prompts hashed above.", ""]
    empty = []
    for scope in ["all", *TOPICS]:
        name = "all topics" if scope == "all" else scope
        text = guidance_text(scope)
        if text:
            out += [f"Guidance for {name}, sha256 `{manifest.get('guidance_sha256', {}).get(scope, 'n/a')}`:",
                    "", "```text", text, "```", ""]
        else:
            empty.append(name)
    if empty:
        out += [f"No guidance for: {', '.join(empty)}.", ""]
    return "\n".join(out)
