"""Loading examples, rendering them as compact case cards, finding contrast pairs."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import LABELS, TOPICS, load_dataset  # noqa: E402

__all__ = ["LABELS", "TOPICS", "load_dataset", "card", "nearest_good", "write_atomic", "majority_labels"]


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
