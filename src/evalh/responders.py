"""Where answers come from: recorded runs (offline, used in CI) or a live chat endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import httpx

SYSTEM_PROMPT = (
    "You are the support assistant for Inkwell Notes. Answer only from the context. "
    "Cite the documents you used as [doc-id]. If the context does not contain the answer, "
    "say: I don't know, that is not covered by the documentation."
)


class Responder(Protocol):
    def answer(self, case_id: str, question: str, context: str) -> str: ...


class RecordedResponder:
    """Replays answers captured earlier (one JSON object per line: {"id", "answer"})."""

    def __init__(self, path: Path) -> None:
        self._answers: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self._answers[row["id"]] = row["answer"]

    def answer(self, case_id: str, question: str, context: str) -> str:
        if case_id not in self._answers:
            raise KeyError(f"no recorded answer for case {case_id!r}")
        return self._answers[case_id]


class ChatCompletionsResponder:
    """Calls any OpenAI-compatible /chat/completions endpoint (for example a local Ollama)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._model = model

    def answer(self, case_id: str, question: str, context: str) -> str:
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("chat completion returned no text content")
        return content.strip()
