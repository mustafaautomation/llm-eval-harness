"""Deterministic per-answer metrics.

These are rule-based checks, not semantic judgements. They are cheap, repeatable and good at
catching regressions; ``evalh.judge`` covers the cases that need a model to judge.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from evalh.text import content_tokens, normalise, sentences

REFUSAL_PATTERNS = (
    r"\bi (?:don't|do not|can't|cannot) (?:know|find|help|answer)\b",
    r"\b(?:not|isn't|is not) (?:covered|in) (?:by )?(?:the|our) (?:documentation|docs|help centre|help center)\b",
    r"\bi can only help with\b",
    r"\boutside (?:of )?what i can help with\b",
)
_REFUSAL = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){9,14}\d(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def fact_coverage(answer: str, facts: Sequence[Sequence[str]]) -> float:
    """Share of required facts present; a fact is present if any of its phrasings appears."""
    if not facts:
        return 1.0
    text = f" {normalise(answer)} "
    covered = sum(1 for alts in facts if any(f" {normalise(a)} " in text for a in alts))
    return covered / len(facts)


def forbidden_hits(answer: str, forbidden: Iterable[str]) -> list[str]:
    text = f" {normalise(answer)} "
    return [f for f in forbidden if f" {normalise(f)} " in text]


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


def citations(answer: str) -> list[str]:
    return _CITATION.findall(answer)


def citation_validity(answer: str, retrieved_ids: Iterable[str]) -> float:
    """Share of cited document ids that were actually in the retrieved context (1.0 if none cited)."""
    cited = citations(answer)
    if not cited:
        return 1.0
    allowed = set(retrieved_ids)
    return sum(1 for c in cited if c in allowed) / len(cited)


def groundedness(answer: str, context: str, min_support: float = 0.6) -> float:
    """Share of answer sentences whose content words mostly appear in the retrieved context.

    A heuristic for unsupported claims: a sentence whose words are largely absent from the
    context is likely invented. Citation markers are ignored.
    """
    ctx = set(content_tokens(context))
    claims = [s for s in sentences(_CITATION.sub("", answer)) if content_tokens(s)]
    if not claims:
        return 1.0
    supported = 0
    for sentence in claims:
        words = content_tokens(sentence)
        if sum(1 for w in words if w in ctx) / len(words) >= min_support:
            supported += 1
    return supported / len(claims)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def pii_findings(answer: str) -> list[str]:
    """Emails, phone-like numbers and Luhn-valid card numbers that appear in an answer."""
    found = [f"email:{m}" for m in _EMAIL.findall(answer)]
    for m in _CARD.findall(answer):
        digits = re.sub(r"\D", "", m)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append(f"card:{digits[-4:]}")
    if not any(f.startswith("card:") for f in found):
        found += [f"phone:{re.sub(r'\D', '', m)[-4:]}" for m in _PHONE.findall(answer)]
    return found


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    if not rel:
        return 1.0
    return len(rel.intersection(retrieved)) / len(rel)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    if not rel:
        return 1.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0
