# llm-eval-harness

[![CI](https://github.com/mustafaautomation/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafaautomation/llm-eval-harness/actions/workflows/ci.yml)

A small, dataset-driven evaluation harness for a retrieval-augmented (RAG) support assistant, with a
regression gate that can fail a CI build. The goal is to treat LLM answer quality the way a QA team
treats any other behaviour: written test cases, measurable checks, thresholds, a baseline to compare
against, and a pipeline that goes red when quality drops.

This is a personal portfolio project. The knowledge base describes "Inkwell Notes", a made-up
note-taking product, and is not based on any employer's or client's system or data.

## What it does

1. **Retrieval:** a BM25 retriever (pure Python, no external service) picks the top `k` documents from
   `data/kb/` for each question.
2. **Answer:** a *responder* produces the answer, either
   - `RecordedResponder`: replays answers from a JSONL file (offline and deterministic, used in CI), or
   - `ChatCompletionsResponder`: calls any OpenAI-compatible `/chat/completions` endpoint, such as a
     local [Ollama](https://ollama.com) server, with the retrieved context in the prompt.
3. **Score:** each answer is checked against its test case in `data/cases.jsonl`.
4. **Gate:** aggregate scores are compared with `thresholds.toml` and, optionally, a baseline run.
   Any breach gives exit code 1, a Markdown report on stdout, and a GitHub Actions job summary.

## Metrics

All per-answer metrics are rule-based, so they are cheap and repeatable. They catch regressions; they
do not understand meaning.

| Metric | What it checks |
|---|---|
| `retrieval_recall_at_k`, `retrieval_mrr` | Did the retriever return the documents the case marks as relevant? |
| `fact_coverage` | Share of required facts present; each fact can list several accepted phrasings. |
| `groundedness` | Share of answer sentences whose content words mostly appear in the retrieved context. A heuristic for invented claims, not proof of faithfulness. |
| `citation_validity` | Every `[doc-id]` the answer cites must be one of the retrieved documents. |
| `refusal_accuracy` | Out-of-scope questions should be declined; in-scope ones should not. |
| `forbidden_content_cases` | Answers containing a phrase the case forbids (for example a wrong refund promise). |
| `pii_leak_cases` | Answers containing emails, phone-like numbers or Luhn-valid card numbers. |
| `pass_rate` | Share of cases that pass every check above. |

`evalh.judge` adds an optional LLM-as-judge (`ChatJudge`) with a strict verdict parser, where malformed
judge output is an error and never a pass. Before trusting a judge, `evalh calibrate` measures its
agreement with human labels using Cohen's kappa. The sample in `data/judge-calibration/` (20 labelled
answers) gives kappa 0.762 against a minimum of 0.6. The judge is not part of the default gate.

## The recorded runs

`runs/` holds three sets of hand-written answers for the 18 cases (3 of them out of scope), so the gate
can be shown both passing and failing without calling a model:

| File | Purpose | Gate result |
|---|---|---|
| `runs/baseline.jsonl` | An older, slightly weaker version, used as the comparison point | n/a |
| `runs/candidate.jsonl` | The "current" version | passes against the baseline |
| `runs/regressed.jsonl` | Deliberately broken: missing facts, a forbidden refund promise, a citation to a document that was not retrieved, and a phone number leaked in a wrong answer to an out-of-scope question | fails |

These are fixtures written to exercise the checks. They are not output from a real model, and the
scores they produce say nothing about any real model's quality.

## Usage

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# offline, as CI runs it
uv run evalh run --responses runs/candidate.jsonl --baseline runs/baseline.jsonl
uv run evalh run --responses runs/regressed.jsonl --baseline runs/candidate.jsonl   # exits 1

# write the full JSON report as well
uv run evalh run --responses runs/candidate.jsonl --json reports/candidate.json

# judge calibration
uv run evalh calibrate --human data/judge-calibration/human.jsonl \
                       --judge data/judge-calibration/judge.jsonl

# live, against a local model (no key needed for Ollama; EVALH_API_KEY is sent as a bearer token if set)
ollama pull llama3.2
uv run evalh run --base-url http://localhost:11434/v1 --model llama3.2
```

A live run uses the same thresholds. A small local model will probably not pass them, which is the
point: the gate reports which cases fail and why.

## Test cases

Each line of `data/cases.jsonl` is one case:

```json
{"id": "plus-price", "question": "How much does the Plus plan cost?",
 "relevant_docs": ["billing-plans"],
 "must_include": [["6 USD", "$6"], ["per month", "a month", "monthly"]],
 "must_not_include": [], "should_refuse": false, "category": "billing"}
```

Adding a case means adding a line (and an answer to each recorded run you want to keep using).

## Development

```bash
uv run pytest            # unit tests for metrics, retriever, responders, judge, gate and CLI
uv run ruff check . && uv run ruff format --check .
uv run mypy              # strict mode
```

The live responder and judge are tested with `httpx.MockTransport`, so the test suite makes no network
calls.

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

- **checks** (Python 3.13 and 3.14): ruff lint and format, strict mypy, and pytest with coverage.
- **eval-gate**: judge calibration, the candidate run (must pass), and the regressed run, which must
  be rejected with exit code 1. If the gate ever stops catching the regression, the build fails.

## Limits

- The metrics are lexical heuristics. They can miss a wrong answer that reuses the right words, and
  they can flag a correct paraphrase. The calibrated judge is the intended way to cover that gap.
- The dataset is small (18 cases) and synthetic. It shows the method; it is not a benchmark.
- BM25 is a deliberate choice to keep retrieval deterministic and dependency-free. Swapping in an
  embedding retriever would only touch `evalh.retriever`.

## License

MIT
