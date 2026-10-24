"""Rule induction, consolidation into global and topic blocks, and refinement with the strong model."""
import json
import re
from pathlib import Path

from . import config
from .data import TOPICS, card

PROMPTS = Path(__file__).parent / "prompts"
PREFIX = {"all": "GL", "credit": "CR", "financial_position": "FP", "savings": "SV", "spending": "SP"}
RULE_FIELDS = ("title", "applies_when", "violation_if", "how_to_check", "exceptions")
MAX_RULES = 12
MAX_ERRORS_SHOWN = 15

RULE_PROPERTIES = {
    "id": {"type": "string"},
    "title": {"type": "string"},
    "applies_when": {"type": "string"},
    "violation_if": {"type": "string"},
    "how_to_check": {"type": "string"},
    "exceptions": {"type": "string"},
    "examples": {"type": "array", "items": {
        "type": "object",
        "properties": {"item_id": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["item_id", "quote"]}},
}
RULE_REQUIRED = ["id", *RULE_FIELDS, "examples"]
RULES_SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {"type": "object", "properties": RULE_PROPERTIES,
                                                         "required": RULE_REQUIRED}},
                   "notes": {"type": "string"}},
    "required": ["rules", "notes"],
}
BOUNDARIES_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["expert_inconsistent", "rule_gap", "rule_too_broad",
                                                            "checker_unstable"]},
                       "pattern": {"type": "string"},
                       "item_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["kind", "pattern", "item_ids"]}}},
    "required": ["groups"],
}
SCOPED_SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {
        "type": "object",
        "properties": {**RULE_PROPERTIES, "scope": {"type": "string", "enum": ["all", *TOPICS]}},
        "required": [*RULE_REQUIRED, "scope"]}},
        "notes": {"type": "string"}},
    "required": ["rules", "notes"],
}


def guidance_text(scope):
    """Optional hand-written guidance for a topic, or for all topics when scope is 'all'; empty if none."""
    name = "global" if scope == "all" else scope
    return re.sub(r"<!--.*?-->", "", (PROMPTS / "topics" / f"{name}.md").read_text(encoding="utf-8"), flags=re.S).strip()


def guidance(scope):
    text = guidance_text(scope)
    return f"<guidance>\n{text}\n</guidance>" if text else ""


def prompt(name, topic="all"):
    """Prompt template with the topic and the guidance filled in."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    topic_part = guidance(topic) if topic != "all" else ""
    return (text.replace("{topic}", topic).replace("{global_guidance}", guidance("all"))
            .replace("{topic_guidance}", topic_part).strip())


def normalize(scope, raw_rules, ids, keep_ids=(), limit=MAX_RULES):
    """Unique rule ids per scope, at most `limit` rules, examples limited to known training items."""
    rules, used = [], set()
    for raw in raw_rules[:limit]:
        rule = {k: str(raw.get(k, "")).strip() for k in RULE_FIELDS}
        rid = str(raw.get("id", "")).strip()
        if rid not in keep_ids or rid in used:
            n = 1
            while f"{PREFIX[scope]}-{n}" in used or f"{PREFIX[scope]}-{n}" in keep_ids:
                n += 1
            rid = f"{PREFIX[scope]}-{n}"
        used.add(rid)
        rule.update(id=rid, scope=scope, active=True,
                    examples=[e for e in raw.get("examples", []) if e.get("item_id") in ids][:5])
        rules.append(rule)
    return rules


def public(rule):
    """The rule fields a model may see."""
    return {k: rule[k] for k in ("id", *RULE_FIELDS)}


async def induce_topic(llm, topic, rows, twins):
    """Induce the topic's rule set from all its labeled conversations, each BAD next to its nearest GOOD."""
    topic_rows = [r for r in rows if r["topic"] == topic]
    parts, shown = [], set()
    for row in (r for r in topic_rows if r["label"] == "bad"):
        parts.append(f"<bad item_id=\"{row['item_id']}\">\n{card(row)}\n</bad>")
        twin = twins.get(row["item_id"])
        if twin:
            parts.append(f"<similar_good item_id=\"{twin['item_id']}\">\n{card(twin)}\n</similar_good>")
            shown.add(twin["item_id"])
    parts += [f"<good item_id=\"{r['item_id']}\">\n{card(r)}\n</good>"
              for r in topic_rows if r["label"] == "good" and r["item_id"] not in shown]
    parts.append(f"Based on the labeled conversations above, write the expert's rule set for the topic \"{topic}\".")
    reply = await llm.json(config.INDUCER_MODEL, prompt("induce", topic), "\n\n".join(parts), RULES_SCHEMA, 32768,
                           thinking_level=config.INDUCER_THINKING_LEVEL)
    if not reply or not reply.get("rules"):
        return None
    return normalize(topic, reply["rules"], {r["item_id"] for r in topic_rows})


async def consolidate(llm, topic_rules, ids):
    """Merge requirements repeated across topics into global rules; return (global_rules, {topic: rules}) or None."""
    shown = {t: [{**public(r), "examples": r["examples"]} for r in rules] for t, rules in topic_rules.items()}
    user = (f"<rule_sets>\n{json.dumps(shown, ensure_ascii=False, indent=1)}\n</rule_sets>\n\n"
            "Based on the rule sets above, return one consolidated list where every rule has scope \"all\" or its topic.")
    reply = await llm.json(config.INDUCER_MODEL, prompt("consolidate"), user, SCOPED_SCHEMA, 32768,
                           thinking_level=config.INDUCER_THINKING_LEVEL)
    if not reply or not reply.get("rules"):
        return None
    scoped = {s: [r for r in reply["rules"] if r.get("scope") == s] for s in ["all", *topic_rules]}
    by_topic = {t: normalize(t, scoped[t], ids) for t in topic_rules}
    if not any(by_topic.values()) and not scoped["all"]:
        return None
    return normalize("all", scoped["all"], ids, limit=10), by_topic


async def refine_topic(llm, topic, global_rules, rules, stats, missed, flagged, votes):
    """Ask the strong model for a revised topic block given the checker's errors on part of the training data."""
    shown = [{**public(r), "caught_bad": stats[r["id"]]["tp"], "flagged_good": stats[r["id"]]["fp"]} for r in rules]
    parts = [f"<global_rules read_only=\"true\">\n{json.dumps([public(r) for r in global_rules], ensure_ascii=False, indent=1)}\n</global_rules>",
             f"<topic_rules>\n{json.dumps(shown, ensure_ascii=False, indent=1)}\n</topic_rules>"]
    parts += [f"<missed expert=\"bad\" checker=\"good\" item_id=\"{r['item_id']}\">\n{card(r)}\n</missed>"
              for r in missed[:MAX_ERRORS_SHOWN]]
    for r in flagged[:MAX_ERRORS_SHOWN]:
        fired = {rid: c.get("evidence", "") for v in votes.get(r["item_id"], []) for rid, c in v.items()
                 if c.get("applies") and c.get("violated")}
        parts.append(f"<wrongly_flagged expert=\"good\" checker=\"bad\" item_id=\"{r['item_id']}\">\n"
                     f"checker flags: {json.dumps(fired, ensure_ascii=False)}\n{card(r)}\n</wrongly_flagged>")
    parts.append(f"Based on the errors above, return the complete revised topic rules for \"{topic}\".")
    reply = await llm.json(config.INDUCER_MODEL, prompt("refine", topic), "\n\n".join(parts), RULES_SCHEMA, 32768,
                           thinking_level=config.INDUCER_THINKING_LEVEL)
    if not reply or not reply.get("rules"):
        return None
    ids = {e["item_id"] for r in rules for e in r["examples"]} | {r["item_id"] for r in missed + flagged}
    return normalize(topic, reply["rules"], ids, keep_ids={r["id"] for r in rules})


async def describe_boundaries(llm, rules, errors, votes):
    """Group the training conversations the final rules misjudge by why they are hard, for the jury."""
    errors = errors[:2 * MAX_ERRORS_SHOWN]
    parts = [f"<rules>\n{json.dumps([public(r) for r in rules], ensure_ascii=False, indent=1)}\n</rules>"]
    for row, prediction in errors:
        fired = {rid: c.get("evidence", "") for v in votes.get(row["item_id"], []) for rid, c in v.items()
                 if c.get("applies") and c.get("violated")}
        parts.append(f"<misjudged item_id=\"{row['item_id']}\" expert=\"{row['label']}\" checker=\"{prediction}\">\n"
                     f"checker flags: {json.dumps(fired, ensure_ascii=False)}\n{card(row)}\n</misjudged>")
    parts.append("Based on the misjudged conversations above, group them by why they are hard.")
    reply = await llm.json(config.INDUCER_MODEL, prompt("boundaries"), "\n\n".join(parts), BOUNDARIES_SCHEMA, 16384,
                           thinking_level=config.INDUCER_THINKING_LEVEL)
    ids = {row["item_id"] for row, _ in errors}
    groups = [{"kind": g.get("kind", "rule_gap"), "pattern": str(g.get("pattern", "")).strip(),
               "item_ids": [i for i in g.get("item_ids", []) if i in ids]} for g in (reply or {}).get("groups", [])[:6]]
    return [g for g in groups if g["item_ids"] and g["pattern"]]
