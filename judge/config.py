"""Runtime settings, all read from environment variables."""
import os
from pathlib import Path


def load_dotenv(path=".env"):
    """Minimal .env loader for local runs; real environment variables win."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))


load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini: GEMINI_API_KEY; vertex: gcloud ADC
INDUCER_MODEL = os.getenv("INDUCER_MODEL", "gemini-4-argon")
CHECKER_MODEL = os.getenv("CHECKER_MODEL", "gemini-3.8-flash")
# Gemini 3.x/4 thinking depth: empty = model default; e.g. low, medium, high (minimal only on Flash-Lite)
INDUCER_THINKING_LEVEL = os.getenv("INDUCER_THINKING_LEVEL", "")
CHECKER_THINKING_LEVEL = os.getenv("CHECKER_THINKING_LEVEL", "")
VOTES = int(os.getenv("VOTES", "3"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
REFINE_ROUNDS = int(os.getenv("REFINE_ROUNDS", "2"))
ADAPT_TIME_BUDGET_S = int(os.getenv("ADAPT_TIME_BUDGET_S", "3000"))
EVAL_TIME_BUDGET_S = int(os.getenv("EVAL_TIME_BUDGET_S", "1500"))
CALL_TIMEOUT_S = int(os.getenv("LLM_CALL_TIMEOUT_S", "300"))
# Hard cap on estimated model spend per adapt or eval run; past it, the remaining items keep fallback labels
SPEND_LIMIT_USD = float(os.getenv("SPEND_LIMIT_USD", "15"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
CACHE_DIR = os.getenv("LLM_CACHE_DIR", "")  # dev only: reuse identical calls across runs
# Dev only: eval also writes per-conversation votes and evidence to this file
TRACE_FILE = os.getenv("JUDGE_TRACE_FILE", "")
# Dev only: run the LLM pipeline for these topics, the rest get fallback labels
ONLY_TOPICS = [t for t in os.getenv("ONLY_TOPICS", "").split(",") if t]
