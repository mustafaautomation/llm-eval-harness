"""Optional LLM-as-judge, plus Cohen's kappa to check a judge against human labels.

A judge is only useful if it agrees with people. ``cohen_kappa`` measures that agreement on a
labelled sample before the judge's verdicts are trusted in a gate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

RUBRIC = (
    "You grade a support answer. Reply with JSON only: "
    '{"verdict": "pass" | "fail", "reason": "<one sentence>"}. '
    "Fail if the answer states anything not supported by the context, misses the main point "
    "of the question, or answers when it should have said it does not know."
)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def parse_verdict(raw: str) -> Verdict:
    """Strict parse: anything other than the expected JSON shape is an error, not a pass."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    verdict = data.get("verdict")
    if verdict not in ("pass", "fail"):
        raise ValueError(f"judge returned an invalid verdict: {verdict!r}")
    return Verdict(passed=verdict == "pass", reason=str(data.get("reason", "")))


class ChatJudge:
    def __init__(self, client: httpx.Client, model: str) -> None:
        self._client = client
        self._model = model

    def grade(self, question: str, context: str, answer: str) -> Verdict:
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": RUBRIC},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}",
                    },
                ],
            },
        )
        response.raise_for_status()
        return parse_verdict(response.json()["choices"][0]["message"]["content"])


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two raters beyond chance. 1.0 is perfect, 0.0 is chance level."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and the same length")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)
