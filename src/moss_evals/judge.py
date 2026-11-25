"""Faithfulness / correctness judge, via Moss Inference (`/v1/responses`).

The judge only sees the passages, the question, the answer, and an optional reference.
It returns a json_schema verdict. Run `moss-evals calibrate` before you trust it.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Sequence

import httpx

BASE_URL = "https://inference.moss.dev/v1"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"}, "supported": {"type": "boolean"}}, "required": ["claim", "supported"], "additionalProperties": False}},
        "abstained": {"type": "boolean"},
        "correct": {"type": ["boolean", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["claims", "abstained", "correct", "rationale"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = (
    "You grade a retrieval-augmented answer. Split the ANSWER into atomic factual claims. A claim is supported only if the "
    "PASSAGES state it (paraphrase is fine; outside knowledge is not). Set abstained=true if the answer declines because the "
    "information is missing. If a REFERENCE is given, set correct=true when the answer conveys the reference facts without "
    "contradicting them, else false; if no REFERENCE, set correct=null. Keep rationale under 30 words."
)


def _passages_block(passages: Sequence[str]) -> str:
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "(none)"


class MossLLM:
    def __init__(self, project_id: str, project_key: str, timeout: float = 90.0):
        self._auth = f"Bearer {project_id}:{project_key}"
        self._client = httpx.AsyncClient(timeout=timeout)
        self.usage: list[dict[str, Any]] = []

    async def close(self) -> None:
        await self._client.aclose()

    async def respond(self, body: dict[str, Any], headers: Optional[dict[str, str]] = None, retries: int = 4) -> dict[str, Any]:
        hdrs = {"Authorization": self._auth, "Content-Type": "application/json", **(headers or {})}
        for attempt in range(retries + 1):
            r = await self._client.post(f"{BASE_URL}/responses", headers=hdrs, json=body)
            if r.status_code in (429, 502, 503) and attempt < retries:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
                await asyncio.sleep(min(wait, 20.0))
                continue
            r.raise_for_status()
            data = r.json()
            self.usage.append({"model": body.get("model"), **(data.get("usage") or {}),
                               "retrieve_ms": r.headers.get("x-moss-retrieve-ms")})
            return data
        raise RuntimeError("unreachable")

    @staticmethod
    def text_of(resp: dict[str, Any]) -> str:
        parts = []
        for item in resp.get("output") or []:
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        parts.append(c.get("text") or "")
        return "".join(parts)

    async def generate(self, question: str, passages: Sequence[str], model: str = "moss/fastlane-1", max_output_tokens: int = 160) -> str:
        resp = await self.respond({
            "model": model,
            "instructions": "Answer in 1-2 sentences using ONLY the passages. If they do not contain the answer, say you do not have that information.\n\nPASSAGES:\n" + _passages_block(passages),
            "input": question,
            "max_output_tokens": max_output_tokens,
        }, headers={"X-Moss-Cache": "off"})
        return self.text_of(resp)

    async def judge(self, question: str, passages: Sequence[str], answer: str, reference: Optional[str] = None,
                    model: str = "openai/gpt-6-luna") -> dict[str, Any]:
        content = f"QUESTION: {question}\n\nPASSAGES:\n{_passages_block(passages)}\n\nANSWER: {answer}\n\nREFERENCE: {reference or '(none)'}"
        resp = await self.respond({
            "model": model,
            "instructions": JUDGE_INSTRUCTIONS,
            "input": content,
            "max_output_tokens": 900,
            "text": {"format": {"type": "json_schema", "name": "verdict", "strict": True, "schema": JUDGE_SCHEMA}},
        })
        raw = self.text_of(resp)
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            return {"parse_error": True, "raw": raw[:300]}
        claims = v.get("claims") or []
        v["faithfulness"] = (sum(1 for c in claims if c.get("supported")) / len(claims)) if claims else (1.0 if v.get("abstained") else 0.0)
        return v
