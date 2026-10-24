"""The adapt and eval stages, plus the no-rules baseline used as the 'before' in the demo."""
import asyncio
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

from . import config
from .check import OTHER_ID, checker_prompt, decide, run_checks
from .data import TOPICS, card, load_dataset, majority_labels, nearest_good, write_atomic
from .explain import render
from .llm import LLM, LLMUnavailable
from .rules import PROMPTS, consolidate, induce_topic, refine_topic


def log(message):
    print(f"[judge] {message}", file=sys.stderr, flush=True)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def accuracy(rows, pred):
    return sum(pred.get(r["item_id"]) == r["label"] for r in rows) / len(rows) if rows else 0.0


def macro_accuracy(rows, pred):
    scores = [accuracy([r for r in rows if r["topic"] == t], pred) for t in TOPICS if any(r["topic"] == t for r in rows)]
    return sum(scores) / len(scores)


def live_topics(rulebook):
    return [t for t in TOPICS if rulebook["topics"][t]["status"] == "llm"]


def topic_rules(rulebook, topic):
    """Global block followed by the topic block: everything the checker sees for this topic."""
    return rulebook["global"]["rules"] + rulebook["topics"][topic]["rules"]


def prompts_for(rulebook):
    return {t: checker_prompt(t, rulebook["global"]["rules"], rulebook["topics"][t]["rules"]) for t in live_topics(rulebook)}


def rule_stats(rows, votes, rules):
    """Per rule: how many BAD (tp) and GOOD (fp) training conversations it fires on."""
    stats = {}
    for rule in rules:
        tp = fp = 0
        for row in rows:
            if decide(votes.get(row["item_id"]), [rule["id"]])[1]:
                tp += row["label"] == "bad"
                fp += row["label"] == "good"
        stats[rule["id"]] = {"tp": tp, "fp": fp}
    return stats


def predictions(rows, votes, rulebook, fallback):
    pred, fired = {}, {}
    live = live_topics(rulebook)
    for row in rows:
        label, fired[row["item_id"]] = None, []
        if row["topic"] in live:
            active = [r["id"] for r in topic_rules(rulebook, row["topic"]) if r["active"]]
            if rulebook.get("other", {}).get("active"):
                active.append(OTHER_ID)
            label, fired[row["item_id"]] = decide(votes.get(row["item_id"]), active)
        pred[row["item_id"]] = label or fallback[row["topic"]]
    return pred, fired


def write_predictions(path, pred):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["item_id", "label"])
    writer.writerows(pred.items())
    write_atomic(path, buffer.getvalue())


def save_artifacts(art, rows, rulebook, fallback, votes, started):
    prompts = prompts_for(rulebook)
    for topic, text in prompts.items():
        write_atomic(art / "prompts" / f"checker_{topic}.txt", text)
    rulebook_text = json.dumps(rulebook, ensure_ascii=False, indent=1)
    write_atomic(art / "rulebook.json", rulebook_text)
    pred, fired = predictions(rows, votes, rulebook, fallback)
    audit = [{"item_id": r["item_id"], "topic": r["topic"], "label": r["label"], "prediction": pred[r["item_id"]],
              "fired": fired[r["item_id"]], "votes": votes.get(r["item_id"], [])} for r in rows]
    write_atomic(art / "train_audit.jsonl", "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
    manifest = {"provider": config.PROVIDER, "inducer_model": config.INDUCER_MODEL, "checker_model": config.CHECKER_MODEL,
                "inducer_thinking_level": config.INDUCER_THINKING_LEVEL or "default",
                "checker_thinking_level": config.CHECKER_THINKING_LEVEL or "default",
                "votes": rulebook["votes"], "rulebook_sha256": sha(rulebook_text),
                "prompt_sha256": {t: sha(p) for t, p in prompts.items()}, "train_items": len(rows),
                "train_macro_accuracy": round(macro_accuracy(rows, pred), 4), "seconds": round(time.monotonic() - started)}
    write_atomic(art / "manifest.json", json.dumps(manifest, indent=1))
    write_atomic(art / "EXPLANATION.md", render(rows, rulebook, fallback, audit, manifest))
    return pred


async def adapt(train_path, art_dir):
    started = time.monotonic()
    art = Path(art_dir)
    rows = load_dataset(train_path, require_labels=True)
    fallback = majority_labels(rows)
    write_atomic(art / "fallback.json", json.dumps(fallback))
    rulebook = {"votes": config.VOTES, "global": {"rules": []},
                "other": {"id": OTHER_ID, "title": "Clear error no rule covers: data contradicting the context, "
                          "an unanswered question or an invented fact", "active": False},
                "topics": {t: {"status": "fallback", "rules": []} for t in TOPICS}}
    save_artifacts(art, rows, rulebook, fallback, {}, started)
    try:
        llm = LLM(deadline=started + config.ADAPT_TIME_BUDGET_S * 0.9)
    except LLMUnavailable as e:
        log(f"no model access, artifacts use fallback labels only: {e}")
        return

    topics = [t for t in TOPICS if not config.ONLY_TOPICS or t in config.ONLY_TOPICS]
    log(f"inducing rules for {', '.join(topics)} with {config.INDUCER_MODEL}")
    twins = nearest_good(rows)
    induced = await asyncio.gather(*(induce_topic(llm, t, rows, twins) for t in topics), return_exceptions=True)
    for topic, rules in zip(topics, induced):
        if isinstance(rules, list) and rules:
            rulebook["topics"][topic] = {"status": "llm", "rules": rules}
            log(f"{topic}: {len(rules)} rules induced")
        else:
            log(f"{topic}: induction failed ({rules}), fallback label '{fallback[topic]}'")
    live = live_topics(rulebook)
    if len(live) >= 2:
        try:
            merged = await consolidate(llm, {t: rulebook["topics"][t]["rules"] for t in live}, {r["item_id"] for r in rows})
        except LLMUnavailable as e:
            merged = None
            log(f"consolidation skipped: {e}")
        if merged:
            rulebook["global"]["rules"], by_topic = merged
            for topic in live:
                rulebook["topics"][topic]["rules"] = by_topic[topic]
            log(f"consolidated: {len(merged[0])} global rules, "
                + ", ".join(f"{t} {len(by_topic[t])}" for t in live))
    live_rows = [r for r in rows if r["topic"] in live]

    def current():
        return prompts_for(rulebook), {t: topic_rules(rulebook, t) for t in live}

    votes = await run_checks(llm, live_rows, *current(), 1)
    save_artifacts(art, rows, rulebook, fallback, votes, started)
    half = {r["item_id"]: int(sha(r["item_id"]), 16) % 2 for r in rows}
    global_rules = rulebook["global"]["rules"]
    global_ids = [r["id"] for r in global_rules]

    async def refine_one(topic, shown, round_no):
        rows_t = [r for r in rows if r["topic"] == topic]
        rules = rulebook["topics"][topic]["rules"]
        ids = global_ids + [x["id"] for x in rules]
        pred = {r["item_id"]: decide(votes[r["item_id"]], ids)[0] or fallback[topic] for r in rows_t}
        errors = [r for r in rows_t if half[r["item_id"]] == shown and pred[r["item_id"]] != r["label"]]
        if not errors:
            return
        candidate = await refine_topic(llm, topic, global_rules, rules, rule_stats(rows_t, votes, rules),
                                       [r for r in errors if r["label"] == "bad"],
                                       [r for r in errors if r["label"] == "good"], votes)
        if not candidate:
            return
        cand_votes = await run_checks(llm, rows_t, {topic: checker_prompt(topic, global_rules, candidate)},
                                      {topic: global_rules + candidate}, 1)
        cand_ids = global_ids + [x["id"] for x in candidate]
        cand_pred = {r["item_id"]: decide(cand_votes[r["item_id"]], cand_ids)[0] or fallback[topic] for r in rows_t}
        unseen = [r for r in rows_t if half[r["item_id"]] != shown]
        before, after = accuracy(rows_t, pred), accuracy(rows_t, cand_pred)
        if accuracy(unseen, cand_pred) >= accuracy(unseen, pred) and after > before:
            rulebook["topics"][topic]["rules"] = candidate
            votes.update(cand_votes)
            log(f"{topic}: refinement round {round_no} accepted, train accuracy {before:.3f} -> {after:.3f}")
        else:
            log(f"{topic}: refinement round {round_no} rejected, train accuracy {before:.3f} -> {after:.3f}")

    for round_no in range(config.REFINE_ROUNDS):
        await asyncio.gather(*(refine_one(t, round_no % 2, round_no + 1) for t in live), return_exceptions=True)

    log(f"final audit with {config.VOTES} votes per conversation")
    votes = await run_checks(llm, live_rows, *current(), config.VOTES)
    scopes = [(global_rules, live_rows)] + [(rulebook["topics"][t]["rules"], [r for r in rows if r["topic"] == t]) for t in live]
    for rules, scope_rows in scopes:
        stats = rule_stats(scope_rows, votes, rules)
        for rule in rules:
            rule["stats"] = stats[rule["id"]]
            rule["active"] = not (stats[rule["id"]]["fp"] >= stats[rule["id"]]["tp"] and stats[rule["id"]]["fp"] > 0)
    # OTHER stays on only if, among conversations the active rules pass as good, it catches more BAD than GOOD
    rules_pred, _ = predictions(rows, votes, rulebook, fallback)
    other = rulebook["other"]
    other.update(rule_stats(live_rows, votes, [other])[OTHER_ID])
    new_hits = [r for r in live_rows if rules_pred[r["item_id"]] == "good" and decide(votes[r["item_id"]], [OTHER_ID])[1]]
    other["new_bad"] = sum(r["label"] == "bad" for r in new_hits)
    other["new_good"] = sum(r["label"] == "good" for r in new_hits)
    other["active"] = other["new_bad"] > other["new_good"]
    log(f"OTHER safety net: catches {other['new_bad']} more BAD and {other['new_good']} more GOOD -> "
        f"{'active' if other['active'] else 'disabled'}")
    pred = save_artifacts(art, rows, rulebook, fallback, votes, started)
    manifest_path = art / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(adapt_calls=llm.calls, adapt_spent_usd=round(llm.spent, 3))
    write_atomic(manifest_path, json.dumps(manifest, indent=1))
    log(f"adapt done: train macro accuracy {macro_accuracy(rows, pred):.3f}, {llm.calls} calls, "
        f"{llm.failed_calls} failed, ${llm.spent:.2f} estimated, {round(time.monotonic() - started)} s")


async def evaluate(test_path, art_dir, out_dir):
    started = time.monotonic()
    art, out = Path(art_dir), Path(out_dir) / "predictions.csv"
    rows = load_dataset(test_path)
    try:
        stored = json.loads((art / "fallback.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    fallback = {t: stored.get(t, "good") for t in TOPICS}
    pred = {r["item_id"]: fallback[r["topic"]] for r in rows}
    write_predictions(out, pred)
    try:
        rulebook = json.loads((art / "rulebook.json").read_text(encoding="utf-8"))
        manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
        live = live_topics(rulebook)
        prompts = {t: (art / "prompts" / f"checker_{t}.txt").read_text(encoding="utf-8") for t in live}
    except (OSError, ValueError, KeyError) as e:
        log(f"artifacts unreadable, fallback labels kept: {e}")
        return
    for topic in live:
        if sha(prompts[topic]) != manifest["prompt_sha256"].get(topic):
            log(f"WARNING: prompt for {topic} differs from the hash recorded by adapt")
    try:
        llm = LLM(deadline=started + config.EVAL_TIME_BUDGET_S * 0.9)
    except LLMUnavailable as e:
        log(f"no model access, fallback labels kept: {e}")
        return
    live_rows = [r for r in rows if r["topic"] in live]
    votes = await run_checks(llm, live_rows, prompts, {t: topic_rules(rulebook, t) for t in live}, rulebook["votes"])
    pred, fired = predictions(rows, votes, rulebook, fallback)
    write_predictions(out, pred)
    if config.TRACE_FILE:  # dev only: votes and evidence per conversation, for error analysis
        write_atomic(config.TRACE_FILE, "".join(json.dumps({"item_id": r["item_id"], "topic": r["topic"], "prediction": pred[r["item_id"]],
                                                            "fired": fired[r["item_id"]], "votes": votes.get(r["item_id"], [])},
                                                           ensure_ascii=False) + "\n" for r in rows))
    judged = sum(1 for r in live_rows if votes[r["item_id"]])
    log(f"eval done: {judged}/{len(rows)} conversations judged by the model, {llm.failed_calls} failed calls, "
        f"${llm.spent:.2f} estimated, {round(time.monotonic() - started)} s")


BASELINE_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["good", "bad"]},
                                                    "reason": {"type": "string"}}, "required": ["label", "reason"]}


async def baseline(test_path, out_dir):
    """Same checker model, one plain request per conversation, no learned rules."""
    rows = load_dataset(test_path)
    llm = LLM()
    system = (PROMPTS / "baseline.md").read_text(encoding="utf-8")

    async def one(row):
        reply = await llm.json(config.CHECKER_MODEL, system, card(row), BASELINE_SCHEMA, 8192,
                               thinking_level=config.CHECKER_THINKING_LEVEL)
        return row["item_id"], (reply or {}).get("label", "good")

    write_predictions(Path(out_dir) / "predictions.csv", dict(await asyncio.gather(*(one(r) for r in rows))))
