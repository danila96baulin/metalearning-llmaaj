"""Async JSON calls to Gemini (API key or Vertex ADC) and OpenAI, with retries, a concurrency cap,
a spend limit and an optional disk cache. The provider is chosen per call from the model name."""
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from . import config


class LLMUnavailable(RuntimeError):
    """The model API cannot be used; callers keep their fallback labels."""


# $ per 1M tokens (input, output incl. thinking). List prices, not promos, so the limit errs on the safe side.
PRICES = {
    "gemini-4-argon": (4.0, 20.0), "gemini-3.1-pro-preview": (2.0, 12.0), "gemini-3.8-flash": (1.5, 7.5),
    "gemini-3.5-flash": (1.5, 9.0), "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemma-4-26b-a4b-it": (0.0, 0.0), "gemma-4-31b-it": (0.0, 0.0),
    "gpt-6-astra": (10.0, 50.0), "gpt-6.1-sol": (2.0, 10.0), "gpt-6-sol": (2.0, 10.0), "gpt-6-luna": (0.10, 0.50),
    "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.0, 8.0),
}
UNKNOWN_PRICE = (10.0, 50.0)


def is_openai(model):
    return model.startswith(("gpt-", "o1", "o3", "o4"))


class LLM:
    def __init__(self, deadline=None):
        self.gemini = self.openai = None
        if config.PROVIDER == "vertex" or os.getenv("GEMINI_API_KEY"):
            from google import genai
            self.gemini = (genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
                           if config.PROVIDER == "vertex" else genai.Client(vertexai=False, api_key=os.environ["GEMINI_API_KEY"]))
        if os.getenv("OPENAI_API_KEY"):
            from openai import AsyncOpenAI
            self.openai = AsyncOpenAI(max_retries=0)
        needed = {config.INDUCER_MODEL, config.CHECKER_MODEL}
        if any(is_openai(m) for m in needed) and not self.openai:
            raise LLMUnavailable("OPENAI_API_KEY is not set but an OpenAI model is configured")
        if any(not is_openai(m) for m in needed) and not self.gemini:
            raise LLMUnavailable("GEMINI_API_KEY is not set: set it, or use LLM_PROVIDER=vertex with gcloud ADC")
        self.sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
        self.deadline = deadline
        self.failures_in_a_row = 0
        self.calls = 0
        self.failed_calls = 0
        self.spent = 0.0
        self.cache = Path(config.CACHE_DIR) if config.CACHE_DIR else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _charge(self, model, tokens_in, tokens_out):
        price_in, price_out = PRICES.get(model, UNKNOWN_PRICE)
        self.spent += (tokens_in * price_in + tokens_out * price_out) / 1e6

    async def _gemini(self, model, system, user, schema, max_output_tokens, thinking_level):
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", response_json_schema=schema,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None)
        response = await self.gemini.aio.models.generate_content(model=model, contents=user, config=cfg)
        usage = response.usage_metadata
        if usage:
            self._charge(model, usage.prompt_token_count or 0,
                         (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
        return json.loads(response.text)

    async def _openai(self, model, system, user, schema, max_output_tokens, thinking_level):
        extra = {"reasoning_effort": thinking_level} if thinking_level else {}
        response = await self.openai.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": False}},
            max_completion_tokens=max_output_tokens, **extra)
        if response.usage:
            self._charge(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return json.loads(response.choices[0].message.content)

    async def json(self, model, system, user, schema, max_output_tokens=8192, sample=0, thinking_level=""):
        """Return the parsed JSON reply, or None if this request keeps failing."""
        key = hashlib.sha256(json.dumps([model, system, user, schema, sample, thinking_level],
                                        ensure_ascii=False).encode()).hexdigest()
        cached = self.cache / f"{key}.json" if self.cache else None
        if cached and cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        call = self._openai if is_openai(model) else self._gemini
        error = None
        for attempt in range(config.MAX_RETRIES):
            if self.deadline and time.monotonic() > self.deadline:
                raise LLMUnavailable("time budget exhausted")
            if self.spent >= config.SPEND_LIMIT_USD:
                raise LLMUnavailable(f"spend limit reached: ${self.spent:.2f} of ${config.SPEND_LIMIT_USD:.2f}")
            if self.failures_in_a_row >= 25:
                raise LLMUnavailable(f"model API keeps failing, last error: {error}")
            async with self.sem:
                self.calls += 1
                try:
                    result = await asyncio.wait_for(call(model, system, user, schema, max_output_tokens, thinking_level),
                                                    config.CALL_TIMEOUT_S)
                    self.failures_in_a_row = 0
                    if cached:
                        cached.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                    return result
                except Exception as e:  # classify below; never crash the pipeline on one call
                    error = f"{type(e).__name__}: {str(e)[:300]}"
                    self.failed_calls += 1
                    self.failures_in_a_row += 1
                    code = str(getattr(e, "code", "") or getattr(e, "status_code", ""))
                    if code in ("401", "403", "404"):
                        raise LLMUnavailable(f"model API rejected the request ({error}); check the key, provider and model name") from e
                    if code == "400":
                        print(f"[llm] {model}: request rejected, not retried: {error}", file=sys.stderr)
                        return None
            await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        print(f"[llm] {model}: giving up after {config.MAX_RETRIES} attempts: {error}", file=sys.stderr)
        return None
