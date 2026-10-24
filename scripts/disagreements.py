"""Dev tool: readable case files per person: errors, overfitting checks and random correct cases.

Usage: python scripts/disagreements.py out/<run> <output_dir> name=topic,topic [name=topic,topic ...]
Needs the run's artifacts/ and output/trace.jsonl (eval with JUDGE_TRACE_FILE set).
For every case: expert label, judge label, why the judge decided so (rules, votes, evidence), the dialogue and the context.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from common import load_dataset  # noqa: E402
from judge.check import decide  # noqa: E402
from judge.explain import verified  # noqa: E402

RANDOM_PER_TOPIC = 4
ROLE = {"user": "👤 Клиент", "assistant": "🤖 Ассистент", "system": "⚙️ System", "tool": "🔧 Tool"}


def why(item, titles):
    """The judge's reasoning, exactly as the checker returned it."""
    votes = item["votes"]
    n = len(votes)
    if not n:
        return ["- Модель не ответила: метка взята из fallback (мажоритарная по топику)."]
    lines = []
    for rid in item["fired"]:
        hits = [v[rid] for v in votes if rid in v and v[rid]["applies"] and v[rid]["violated"]]
        lines.append(f"- ❗ **{rid}** {titles.get(rid, '')} — нарушено в {len(hits)} из {n} прогонов")
        lines += [f"  > {' '.join(h['evidence'].split())}" for h in hits[:1] if h["evidence"]]
    if not item["fired"]:
        lines.append(f"- Ни одно правило не нарушено большинством из {n} прогонов.")
    ids = sorted({rid for v in votes for rid in v})
    applied = [rid for rid in ids if rid not in item["fired"] and rid != "OTHER"
               and 2 * sum(1 for v in votes if v.get(rid, {}).get("applies")) > n]
    if applied:
        lines.append("- Применялись и соблюдены: " + ", ".join(f"**{r}** {titles.get(r, '')}" for r in applied))
    for rid in ids:
        minority = [v[rid] for v in votes if v.get(rid, {}).get("applies") and v[rid].get("violated")]
        if rid not in item["fired"] and minority:
            kind = ("⚪ Отключено, метку не решает" if 2 * len(minority) > n or rid == "OTHER" and len(minority) == n
                    else "⚠️ Меньшинство")
            lines.append(f"- {kind}: **{rid}** {titles.get(rid, '')} — нарушение увидели {len(minority)} из {n} прогонов:")
            lines.append(f"  > {' '.join(minority[0]['evidence'].split())}")
    return lines


def case(row, split, item, titles):
    ok = "✅ совпали" if row["label"] == item["prediction"] else "❌ разошлись"
    lines = [f"### `{row['item_id'][:8]}` · {split} · {row['topic']}", "",
             f"**Эксперт:** `{row['label']}` · **Наш ИИ:** `{item['prediction']}` · {ok}", "",
             "**Почему ИИ так решил** (ответ чекера как есть):", ""] + why(item, titles) + ["", "**Диалог:**", ""]
    for m in row["conversation"]:
        lines += [f"{ROLE.get(m['role'], m['role'])}:", "", "> " + m["content"].replace("\n", "\n> "), ""]
    lines += ["<details><summary>Контекст (данные, на которые должен опираться ассистент)</summary>", "", "```json",
              json.dumps(row["context"], ensure_ascii=False, indent=1), "```", "", "</details>", "",
              "- [ ] эксперт прав  - [ ] ИИ прав  - [ ] спорно", "- **Комментарий:** ", "", "---", ""]
    return lines


def main():
    run, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    owners = [a.split("=") for a in sys.argv[3:]]
    rulebook = json.loads((run / "artifacts" / "rulebook.json").read_text(encoding="utf-8"))
    rules = rulebook["global"]["rules"] + [r for t in rulebook["topics"].values() for r in t["rules"]]
    titles = {r["id"]: r["title"] for r in rules}
    titles["OTHER"] = "Явная ошибка вне правил"
    train = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_train.csv")}
    test = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_test.csv")}
    audit = [json.loads(line) for line in (run / "artifacts" / "train_audit.jsonl").open(encoding="utf-8")]
    trace = [json.loads(line) for line in (run / "output" / "trace.jsonl").open(encoding="utf-8")]
    rng = random.Random(42)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, topics in owners:
        topics = topics.split(",")
        summary, errors, overfit, sample = [], [], [], []
        for topic in topics:
            te = [i for i in trace if i["topic"] == topic]
            tr = [a for a in audit if a["topic"] == topic]
            te_err = [i for i in te if i["prediction"] != test[i["item_id"]]["label"]]
            tr_err = [a for a in tr if a["prediction"] != a["label"]]
            summary.append(f"| {topic} | {len(te) - len(te_err)}/{len(te)} | {len(tr) - len(tr_err)}/{len(tr)} |")
            errors += [("test", test[i["item_id"]], i) for i in te_err] + [("train", train[a["item_id"]], a) for a in tr_err]
            # Overfitting checks: rules that never fired on test or flagged a GOOD test case, with their support on train
            topic_rules = rulebook["topics"][topic]["rules"] + rulebook["global"]["rules"]
            for rule in topic_rules:
                te_fire = [i for i in te if decide(i["votes"], [rule["id"]])[1]]
                false_te = [i for i in te_fire if test[i["item_id"]]["label"] == "good"]
                if rule["active"] and (not te_fire or false_te):
                    tr_fire = [a for a in tr if decide(a["votes"], [rule["id"]])[1]]
                    reason = "ни разу не сработало на test" if not te_fire else "сработало на GOOD в test"
                    overfit.append(f"#### {rule['id']} {rule['title']}: {reason}")
                    overfit += [("test", test[i["item_id"]], i) for i in te_fire]
                    overfit += [("train", train[a["item_id"]], a) for a in tr_fire[:3]]
            shaky = [i for i in te if i["prediction"] == test[i["item_id"]]["label"] == "bad"
                     and not any(verified(next((v[r]["evidence"] for v in i["votes"] if r in v and v[r]["violated"]), ""),
                                          test[i["item_id"]]) for r in i["fired"])]
            if shaky:
                overfit.append("#### Верно `bad`, но доказательство не дословная цитата")
                overfit += [("test", test[i["item_id"]], i) for i in shaky]
            correct = [i for i in te if i["prediction"] == test[i["item_id"]]["label"] and i not in shaky]
            sample += [("test", test[i["item_id"]], i) for i in rng.sample(correct, min(RANDOM_PER_TOPIC, len(correct)))]

        def render(entries):
            lines = []
            for e in entries:
                lines += [e, ""] if isinstance(e, str) else case(e[1], e[0], e[2], titles)
            return lines or ["Нет.", ""]

        lines = [f"# Кейсы на проверку: {name}", "",
                 f"Прогон `{run.name}`: правила от `gemini-4-argon`, чекер `gpt-6.1-sol`, 3 прогона на диалог. Топики: {', '.join(topics)}.", "",
                 "| Топик | Test верно | Train верно |", "|---|---|---|", *summary, "",
                 "В каждом кейсе отметьте, кто прав, и оставьте комментарий. «Почему ИИ так решил» — дословный ответ чекера.", "",
                 "## 1. Ошибки", ""] + render(errors) + \
                ["## 2. Проверка на переобучение", "",
                 "Правила, которые на test не сработали ни разу или сработали на GOOD, и их опорные примеры на train. "
                 "Вопрос: это настоящее правило эксперта или подгонка под train?", ""] + render(overfit) + \
                [f"## 3. Случайные верные кейсы, по {RANDOM_PER_TOPIC} на топик", ""] + render(sample)
        (out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"{out_dir / f'{name}.md'}: errors {len(errors)}, overfit cases {sum(not isinstance(e, str) for e in overfit)}, random {len(sample)}")


if __name__ == "__main__":
    main()
