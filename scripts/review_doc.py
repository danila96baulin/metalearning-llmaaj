"""Dev tool: one Markdown file for the team to validate rules and prompts, with overfitting signals.

Usage: python scripts/review_doc.py out/<run> <output.md> name=topic,topic [name=topic,topic ...]
Needs artifacts/ and output/trace.jsonl of the run (eval with JUDGE_TRACE_FILE set).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from common import TOPICS, load_dataset  # noqa: E402
from judge.check import decide  # noqa: E402
from judge.explain import verified  # noqa: E402

CHECKBOXES = "- [ ] ✅ согласен  - [ ] ✏️ поправить  - [ ] ❌ убрать\n- **Комментарий:** "


def fires(items, rule_id):
    return [i for i in items if decide(i["votes"], [rule_id])[1]]


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    owners = {t: name for name, topics in (a.split("=") for a in sys.argv[3:]) for t in topics.split(",")}
    art = run / "artifacts"
    rulebook = json.loads((art / "rulebook.json").read_text(encoding="utf-8"))
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    train = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_train.csv")}
    test = {r["item_id"]: r for r in load_dataset(ROOT / "data" / "public_test.csv")}
    audit = [json.loads(line) for line in (art / "train_audit.jsonl").open(encoding="utf-8")]
    trace = [json.loads(line) for line in (run / "output" / "trace.jsonl").open(encoding="utf-8")]

    lines = [f"# Валидация правил и промптов: прогон `{run.name}`", "",
             f"Индуктор `{manifest['inducer_model']}`, чекер `{manifest['checker_model']}`, {manifest['votes']} голоса. "
             "Правила выведены автоматически из `public_train`. Отметьте в каждой карточке решение и оставьте комментарий.", "",
             "## 1. Разрыв train → test по топикам", "",
             "| Топик | Владелец | Train | Test | Разрыв |", "|---|---|---|---|---|"]
    for t in TOPICS:
        tr = [a for a in audit if a["topic"] == t]
        te = [i for i in trace if i["topic"] == t]
        acc_tr = sum(a["prediction"] == a["label"] for a in tr) / len(tr)
        acc_te = sum(i["prediction"] == test[i["item_id"]]["label"] for i in te) / len(te)
        lines.append(f"| {t} | {owners.get(t, '—')} | {acc_tr:.3f} | {acc_te:.3f} | {acc_tr - acc_te:+.3f} |")
    lines += ["", "Разрыв больше 0.05 по топику — сигнал переобучения: правила подогнаны под train.", "",
              "## 2. Правила", "",
              "Сигналы переобучения в карточке: ⚠️ **одна опора** — правило поймало на train не больше одного BAD; "
              "⚠️ **не обобщилось** — ни разу не сработало на test; ⚠️ **ложно на test** — сработало на GOOD в test.", ""]

    def card(rule, scope):
        tr_items = audit if scope == "all" else [a for a in audit if a["topic"] == scope]
        te_items = trace if scope == "all" else [i for i in trace if i["topic"] == scope]
        tr_fire, te_fire = fires(tr_items, rule["id"]), fires(te_items, rule["id"])
        tp = sum(a["label"] == "bad" for a in tr_fire)
        fp = sum(a["label"] == "good" for a in tr_fire)
        te_bad = sum(test[i["item_id"]]["label"] == "bad" for i in te_fire)
        te_good = len(te_fire) - te_bad
        flags = [f for f, on in (("⚠️ одна опора", tp <= 1), ("⚠️ не обобщилось", rule["active"] and not te_fire),
                                 ("⚠️ ложно на test", te_good > 0)) if on]
        example = next((e for e in rule.get("examples", []) if e.get("quote")), None)
        out_lines = [f"### {rule['id']} · {rule['title']}", "",
                     f"- **Применимо, когда:** {rule['applies_when']}",
                     f"- **Нарушение, если:** {rule['violation_if']}",
                     f"- **Как проверить:** {rule['how_to_check']}",
                     f"- **Исключения:** {rule['exceptions'] or 'нет'}"]
        if example:
            out_lines.append(f"- **Пример нарушения:** «{' '.join(example['quote'].split())[:300]}» (`{example['item_id'][:8]}`)")
        out_lines += [f"- **Train:** поймало BAD {tp}, ложно GOOD {fp} · **Test:** сработало {len(te_fire)} раз: BAD {te_bad}, GOOD {te_good}"
                      f" · **Статус:** {'active' if rule['active'] else 'disabled'}"
                      + (f" · {' · '.join(flags)}" if flags else ""), CHECKBOXES, ""]
        return out_lines

    lines += ["### Общий блок: все топики, валидируют оба", ""]
    for rule in rulebook["global"]["rules"]:
        lines += card(rule, "all")
    for t in TOPICS:
        lines += [f"### Топик {t}: владелец {owners.get(t, '—')}", ""]
        for rule in rulebook["topics"][t]["rules"]:
            lines += card(rule, t)

    lines += ["## 3. Верно, но с сомнительным обоснованием", "",
              "Примеры test, где судья совпал с экспертом на `bad`, но цитата-доказательство не найдена в ответах ассистента. "
              "Возможно, правило сработало не по делу.", ""]
    shaky = []
    for i in trace:
        row = test[i["item_id"]]
        if i["prediction"] == row["label"] == "bad":
            quotes = [next((v[r]["evidence"] for v in i["votes"] if r in v and v[r]["violated"]), "") for r in i["fired"]]
            if not any(verified(q, row) for q in quotes):
                shaky.append((row, i["fired"], quotes))
    if shaky:
        lines += ["| Пример | Топик | Сработало | Доказательство судьи |", "|---|---|---|---|"]
        lines += [f"| `{row['item_id'][:8]}` | {row['topic']} | {', '.join(f)} | {' / '.join(' '.join(q.split())[:160] for q in qs)} |"
                  for row, f, qs in shaky]
    else:
        lines.append("Таких нет: у всех верных `bad` цитата найдена в тексте.")
    lines += ["", "## 4. Промпты", "",
              "Файлы в форке `judge/prompts/`. Общие промпты меняем вместе; `topics/<topic>.md` — подсказки владельца топика.", ""]
    for name in ["induce", "consolidate", "refine", "checker", "other_violation"]:
        lines += [f"### `{name}.md`", "", "```text", (ROOT / "judge" / "prompts" / f"{name}.md").read_text(encoding="utf-8").strip(),
                  "```", "", CHECKBOXES, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out, f"| shaky correct: {len(shaky)}")


if __name__ == "__main__":
    main()
