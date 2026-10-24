"""Loading examples, rendering them as compact case cards, finding contrast pairs."""
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import LABELS, TOPICS, load_dataset  # noqa: E402

__all__ = ["LABELS", "TOPICS", "load_dataset", "load_rows", "card", "nearest_good", "write_atomic", "majority_labels"]


def load_rows(path, require_labels=False):
    """The organizers' strict loader when the file passes it; otherwise row by row, so that one malformed row never
    stops a run. Every row with an item_id is kept for eval, which labels what it cannot judge with the fallback;
    adapt learns only from rows with a known topic, a label and at least one message."""
    try:
        return load_dataset(path, require_labels=require_labels)
    except Exception as e:  # any validation error: fall back to reading row by row
        print(f"[judge] input fails the strict check ({e}); reading it row by row", file=sys.stderr)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if str(path).endswith(".csv"):
        csv.field_size_limit(sys.maxsize)
        records = list(csv.DictReader(io.StringIO(text)))
    else:
        records = []
        for line in text.splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    rows, seen = [], set()
    for raw in records:
        item_id = str(raw.get("item_id") or "").strip() if isinstance(raw, dict) else ""
        if not item_id or item_id in seen:
            continue
        context, conversation = _parsed(raw.get("context")), _parsed(raw.get("conversation"))
        messages = [{"role": str(m.get("role") or ""), "content": str(m.get("content") or "")}
                    for m in (conversation if isinstance(conversation, list) else []) if isinstance(m, dict)]
        row = {"item_id": item_id, "topic": raw.get("topic") if raw.get("topic") in TOPICS else None,
               "context": context if isinstance(context, dict) else {},
               "conversation": [m for m in messages if m["content"].strip()]}
        if raw.get("label") in LABELS:
            row["label"] = raw["label"]
        if require_labels and not (row["topic"] and row.get("label") and row["conversation"]):
            continue
        seen.add(item_id)
        rows.append(row)
    return rows


def _parsed(value):
    """A JSON field that may arrive as a string; None if it does not parse."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def card(row, calculator=True):
    """Topic, compact context JSON, numbered turns in the original language, and the calculator block."""
    from .calc import calculator_block
    turns = "\n".join(f"[{i}] {m['role']}: {m['content']}" for i, m in enumerate(row["conversation"], 1))
    context = json.dumps(row["context"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = f"TOPIC: {row['topic']}\nCONTEXT: {context}\nCONVERSATION:\n{turns}"
    if not calculator:
        return text
    try:
        return f"{text}\n{calculator_block(row)}"
    except Exception as e:  # unexpected data must never break a check: fall back to the card without arithmetic
        print(f"[calc] skipped for {row['item_id']}: {type(e).__name__}: {e}", file=sys.stderr)
        return text


def nearest_good(rows):
    """Map each bad item_id to the most similar good example of the same topic (char TF-IDF)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    result = {}
    for topic in TOPICS:
        bad = [r for r in rows if r["topic"] == topic and r["label"] == "bad"]
        good = [r for r in rows if r["topic"] == topic and r["label"] == "good"]
        if not bad or not good:
            continue
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([card(r) for r in bad + good])
        similarity = cosine_similarity(vectorizer.transform([card(r) for r in bad]),
                                       vectorizer.transform([card(r) for r in good]))
        for i, row in enumerate(bad):
            result[row["item_id"]] = good[int(similarity[i].argmax())]
    return result


def majority_labels(rows):
    """Majority label per topic; ties go to good."""
    result = {}
    for topic in TOPICS:
        labels = [r["label"] for r in rows if r["topic"] == topic]
        result[topic] = "bad" if labels.count("bad") > labels.count("good") else "good"
    return result


def write_atomic(path, text):
    """Write via a temp file and rename, so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(tmp, path)
