import json
from pathlib import Path

import httpx
import pytest

from evalh.judge import ChatJudge, cohen_kappa, parse_verdict
from evalh.responders import ChatCompletionsResponder, RecordedResponder


def _chat_reply(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_recorded_responder_missing_case(root: Path) -> None:
    responder = RecordedResponder(root / "runs" / "candidate.jsonl")
    with pytest.raises(KeyError, match="no recorded answer"):
        responder.answer("does-not-exist", "q", "ctx")


def test_chat_responder_sends_context_and_zero_temperature() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_reply("  It costs 6 USD per month [billing-plans].  "))

    client = httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))
    responder = ChatCompletionsResponder("http://llm.test/v1", "test-model", client=client)

    answer = responder.answer("plus-price", "How much is Plus?", "[billing-plans]\nPlus costs 6 USD")

    assert answer == "It costs 6 USD per month [billing-plans]."
    body = seen["body"]
    assert isinstance(body, dict)
    assert seen["path"] == "/v1/chat/completions"
    assert body["temperature"] == 0
    assert body["model"] == "test-model"
    assert "Plus costs 6 USD" in body["messages"][1]["content"]


def test_chat_responder_raises_on_http_error() -> None:
    client = httpx.Client(
        base_url="http://llm.test/v1", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(httpx.HTTPStatusError):
        ChatCompletionsResponder("http://llm.test/v1", "m", client=client).answer("id", "q", "c")


class TestVerdictParsing:
    def test_pass(self) -> None:
        v = parse_verdict('{"verdict": "pass", "reason": "grounded"}')
        assert v.passed and v.reason == "grounded"

    def test_code_fenced_json(self) -> None:
        assert not parse_verdict('```json\n{"verdict": "fail", "reason": "invented price"}\n```').passed

    @pytest.mark.parametrize("raw", ['{"verdict": "maybe"}', '{"reason": "no verdict"}'])
    def test_invalid_verdict_is_an_error_not_a_pass(self, raw: str) -> None:
        with pytest.raises(ValueError):
            parse_verdict(raw)

    def test_non_json_is_an_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_verdict("Looks good to me!")


def test_chat_judge_uses_rubric() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "JSON only" in body["messages"][0]["content"]
        return httpx.Response(200, json=_chat_reply('{"verdict": "fail", "reason": "unsupported claim"}'))

    client = httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))
    verdict = ChatJudge(client, "judge-model").grade("q", "ctx", "answer")
    assert not verdict.passed
    assert verdict.reason == "unsupported claim"


class TestCohenKappa:
    def test_perfect_agreement(self) -> None:
        assert cohen_kappa(["pass", "fail", "pass"], ["pass", "fail", "pass"]) == 1.0

    def test_textbook_value(self) -> None:
        # 50 items: 20 both yes, 15 both no, 5 yes/no, 10 no/yes -> kappa 0.4
        a = ["y"] * 20 + ["n"] * 15 + ["y"] * 5 + ["n"] * 10
        b = ["y"] * 20 + ["n"] * 15 + ["n"] * 5 + ["y"] * 10
        assert cohen_kappa(a, b) == pytest.approx(0.4)

    def test_systematic_disagreement_is_negative(self) -> None:
        assert cohen_kappa(["pass", "fail"] * 5, ["fail", "pass"] * 5) < 0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            cohen_kappa(["pass"], [])
