"""Dev tool: list conversations where the judge and the expert disagree, one Markdown file per person.

Usage: python scripts/disagreements.py out/<run> <output_dir> name=topic,topic [name=topic,topic ...]
Needs the run's artifacts/train_audit.jsonl and output/trace.jsonl (eval with JUDGE_TRACE_FILE set).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def short(text, limit=600):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def evidence(votes, rule_id):
    return next((v[rule_id]["evidence"] for v in votes if rule_id in v and v[rule_id]["violated"] and v[rule_id]["evidence"]), "")


def section(row, split, item, rules):
    fired = item["fired"]
    lines = [f"### `{row['item_id']}` · {split} · эксперт **{row['label']}**, судья **{item['prediction']}**", ""]
    if fired:
        lines += [f"- **{rid}** {rules.get(rid, '')}: «{short(evidence(item['votes'], rid), 300)}»" for rid in fired]
    else:
        near = sorted({rid for v in item["votes"] for rid, c in v.items() if c["violated"]})
        lines.append("- Ни одно правило не сработало большинством голосов"
                     + (f"; отдельные голоса за: {', '.join(near)}" if near else "") + ".")
    lines += ["", "<details><summary>Диалог и контекст</summary>", ""]
    lines += [f"- **{m['role']}**: {short(m['content'])}" for m in row["conversation"]]
    lines += ["", f"`context`: {short(json.dumps(row['context'], ensure_ascii=False), 1500)}", "", "</details>", ""]
    return lines


def main():
    run, out_dir, owners = Path(sys.argv[1]), Path(sys.argv[2]), [a.split("=") for a in sys.argv[3:]]
    rulebook = json.loads((run / "artifacts" / "rulebook.json").read_text(encoding="utf-8"))
    rules = {r["id"]: r["title"] for r in rulebook["global"]["rules"]}
    rules.update({r["id"]: r["title"] for t in rulebook["topics"].values() for r in t["rules"]})
    rules["OTHER"] = rulebook.get("other", {}).get("title", "")
    train = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_train.csv")}
    test = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_test.csv")}
    audit = [json.loads(line) for line in (run / "artifacts" / "train_audit.jsonl").open(encoding="utf-8")]
    trace = [json.loads(line) for line in (run / "output" / "trace.jsonl").open(encoding="utf-8")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, topics in owners:
        topics = topics.split(",")
        lines = [f"# Расхождения судьи и эксперта: {name}", "",
                 f"Прогон `{run.name}`, топики: {', '.join(topics)}. Сначала тест: это честная оценка. Потом train.", "",
                 "| Топик | Test: верно | Test: пропущено bad | Test: ложно bad | Train: ошибок |", "|---|---|---|---|---|"]
        blocks = []
        for topic in topics:
            t_items = [i for i in trace if i["topic"] == topic]
            t_err = [i for i in t_items if i["prediction"] != test[i["item_id"]]["label"]]
            a_err = [a for a in audit if a["topic"] == topic and a["prediction"] != a["label"]]
            lines.append(f"| {topic} | {len(t_items) - len(t_err)}/{len(t_items)} | "
                         f"{sum(test[i['item_id']]['label'] == 'bad' for i in t_err)} | "
                         f"{sum(test[i['item_id']]['label'] == 'good' for i in t_err)} | {len(a_err)}/{sum(a['topic'] == topic for a in audit)} |")
            blocks += [f"## {topic}", ""]
            blocks += [line for i in t_err for line in section(test[i["item_id"]], "test", i, rules)]
            blocks += [line for a in a_err for line in section(train[a["item_id"]], "train", a, rules)]
        (out_dir / f"{name}.md").write_text("\n".join(lines + [""] + blocks), encoding="utf-8")
        print(f"{out_dir / f'{name}.md'}")


if __name__ == "__main__":
    main()
