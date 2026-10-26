"""Small text helpers shared by the retriever and the metrics."""

from __future__ import annotations

import re

_STOPWORD_TEXT = """
a an and are as at be but by can do does for from has have how i if in into is it its my no not of on
or our so than that the their them then there these they this to up was we were what when where which
who why will with you your yes also each any every only
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
