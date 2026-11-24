import json
import math
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"


def load_credentials() -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    pid = os.environ["MOSS_PROJECT_ID"].strip()
    pkey = os.environ["MOSS_PROJECT_KEY"].strip()
    if not pid or not pkey:
        raise RuntimeError("MOSS_PROJECT_ID and MOSS_PROJECT_KEY must be set in .env")
    return pid, pkey


def scrub_secrets(obj: Any, secrets: list[str]) -> Any:
    if isinstance(obj, str):
        out = obj
        for s in secrets:
            if s:
                out = out.replace(s, "[REDACTED]")
        return out
    if isinstance(obj, list):
        return [scrub_secrets(x, secrets) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_secrets(v, secrets) for k, v in obj.items()}
    return obj


def save_log(filename: str, data: Any, secrets: list[str] | None = None) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / filename
    cleaned = scrub_secrets(data, secrets or [])
    path.write_text(json.dumps(cleaned, indent=2, default=str, ensure_ascii=False))
    return path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
    }


# Published Moss Inference pricing (USD per 1M tokens), docs.moss.dev/docs/inference/models (fetched 2026-11-25)
MODEL_PRICES_PER_1M = {
    "moss/fastlane-1": {"input": 0.25, "cached": 0.03, "output": 1.75},
    "openai/gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50},
    "openai/gpt-6.1-terra": {"input": 1.00, "cached": 0.05, "output": 5.00},
    "anthropic/claude-haiku-5.5": {"input": 0.10, "cached": 0.01, "output": 0.50},
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "cached": 0.10, "output": 10.00},
    "google/gemini-3.8-flash": {"input": 0.75, "cached": 0.075, "output": 3.75},
    "google/gemini-3.5-flash-lite": {"input": 0.30, "cached": 0.03, "output": 2.50},
    "z-ai/glm-5.3-flash": {"input": 0.18, "cached": 0.036, "output": 0.60},
    "moonshotai/kimi-k3": {"input": 0.29, "cached": 0.25, "output": 12.96},
    "deepseek/deepseek-v4.1-flash": {"input": 0.054, "cached": 0.0072, "output": 0.60},
    "minimax/minimax-m3.1-flash": {"input": 0.09, "cached": 0.009, "output": 0.36},
    "qwen/qwen3.8-flash": {"input": 0.18, "cached": 0.019, "output": 0.56},
}
RETRIEVAL_PRICE_USD = 0.0005  # $0.50 / 1k retrievals


def estimate_inference_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    retrievals: int = 0,
) -> float:
    prices = MODEL_PRICES_PER_1M.get(model, {"input": 0.25, "cached": 0.03, "output": 1.75})
    uncached = max(0, input_tokens - cached_tokens)
    cost = (
        (uncached / 1_000_000) * prices["input"]
        + (cached_tokens / 1_000_000) * prices["cached"]
        + (output_tokens / 1_000_000) * prices["output"]
        + retrievals * RETRIEVAL_PRICE_USD
    )
    return round(cost, 6)
