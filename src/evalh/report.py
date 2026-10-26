"""JSON and Markdown reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from evalh.evaluate import RunSummary


def to_json(summary: RunSummary, failures: list[str]) -> str:
    return json.dumps(
        {
            "passed": not failures,
            "failures": failures,
            "scores": summary.scores,
            "cases": [asdict(r) | {"passed": r.passed} for r in summary.results],
        },
        indent=2,
    )


def to_markdown(summary: RunSummary, failures: list[str], baseline: dict[str, float] | None = None) -> str:
    lines = ["## LLM evaluation", "", f"**Gate: {'PASS' if not failures else 'FAIL'}**", ""]
    lines += ["| Metric | Score | Baseline |", "|---|---:|---:|"]
    for name, value in summary.scores.items():
        base = f"{baseline[name]:.3f}" if baseline and name in baseline else "-"
        lines.append(f"| {name} | {value:.3f} | {base} |")
    if failures:
        lines += ["", "### Gate failures", *[f"- {f}" for f in failures]]
    failed = [r for r in summary.results if not r.passed]
    if failed:
        lines += ["", "### Failing cases", "| Case | Why |", "|---|---|"]
        for r in failed:
            why = []
            if not r.refusal_correct:
                why.append("refusal wrong")
            if r.fact_coverage < 1.0:
                why.append(f"facts {r.fact_coverage:.2f}")
            if r.forbidden:
                why.append("forbidden: " + ", ".join(r.forbidden))
            if r.pii:
                why.append("PII: " + ", ".join(r.pii))
            if r.citation_validity < 1.0:
                why.append("cites a document that was not retrieved")
            lines.append(f"| {r.case_id} | {'; '.join(why)} |")
    return "\n".join(lines) + "\n"
