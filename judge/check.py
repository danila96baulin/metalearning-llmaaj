"""The checker: one prompt per topic, structured per-rule answers, and the decision rule."""
import asyncio
import sys

from . import config
from .data import card
from .llm import LLMUnavailable
from .rules import prompt


def render_rule(rule):
    text = (f"[{rule['id']}] {rule['title']}\n"
            f"  applies when: {rule['applies_when']}\n"
            f"  violation if: {rule['violation_if']}\n"
            f"  how to check: {rule['how_to_check']}\n"
            f"  exceptions: {rule['exceptions'] or 'none'}")
    quote = next((e["quote"] for e in rule.get("examples", []) if e.get("quote")), "")
    return text + (f"\n  example of a violation: \"{quote}\"" if quote else "")


def checker_prompt(topic, global_rules, topic_rules):
    """One template with a global block and a topic block; adapt saves it and eval reuses the file byte for byte."""
    return (prompt("checker", topic)
            .replace("{global_rules}", "\n\n".join(render_rule(r) for r in global_rules) or "none")
            .replace("{topic_rules}", "\n\n".join(render_rule(r) for r in topic_rules) or "none"))


def checker_schema(rule_ids):
    return {
        "type": "object",
        "properties": {"checks": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "enum": rule_ids},
                "applies": {"type": "boolean"},
                "violated": {"type": "boolean"},
                "evidence": {"type": "string"},
            },
            "required": ["rule_id", "applies", "violated", "evidence"]}}},
        "required": ["checks"],
    }


async def run_checks(llm, rows, prompts, rules_by_topic, votes):
    """item_id -> list of valid votes; a vote maps rule_id -> {applies, violated, evidence}."""
    async def one(row, sample):
        rules = rules_by_topic.get(row["topic"])
        if not rules:
            return row["item_id"], None
        ids = [r["id"] for r in rules]
        user = f"{card(row)}\n\nCheck the conversation above against every rule."
        reply = await llm.json(config.CHECKER_MODEL, prompts[row["topic"]], user, checker_schema(ids), 8192, sample,
                               thinking_level=config.CHECKER_THINKING_LEVEL)
        if not reply:
            return row["item_id"], None
        return row["item_id"], {c["rule_id"]: {"applies": bool(c.get("applies")), "violated": bool(c.get("violated")),
                                               "evidence": str(c.get("evidence", ""))}
                                for c in reply.get("checks", []) if c.get("rule_id") in ids}

    results = await asyncio.gather(*(one(r, k) for r in rows for k in range(votes)), return_exceptions=True)
    votes_by_item = {r["item_id"]: [] for r in rows}
    unavailable = [r for r in results if isinstance(r, LLMUnavailable)]
    if unavailable:
        print(f"[check] model API unavailable, affected items keep fallback labels: {unavailable[0]}", file=sys.stderr)
    for result in results:
        if isinstance(result, tuple) and result[1] is not None:
            votes_by_item[result[0]].append(result[1])
    return votes_by_item


def decide(votes, active_rule_ids):
    """BAD iff an active rule applies and is violated in a strict majority of valid votes; None if no votes."""
    if not votes:
        return None, []
    fired = [rid for rid in active_rule_ids
             if 2 * sum(1 for v in votes if v.get(rid, {}).get("applies") and v.get(rid, {}).get("violated")) > len(votes)]
    return ("bad" if fired else "good"), fired
