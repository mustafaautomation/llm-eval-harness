"""Loading the golden dataset and the knowledge base."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Case:
    """One golden test case.

    ``must_include`` is a list of facts; each fact is a list of acceptable phrasings, and the
    answer covers the fact if it contains any of them.
    """

    id: str
    question: str
    relevant_docs: tuple[str, ...]
    must_include: tuple[tuple[str, ...], ...] = ()
    must_not_include: tuple[str, ...] = ()
    should_refuse: bool = False
    category: str = "general"


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass
class Dataset:
    cases: list[Case]
    documents: dict[str, Document] = field(default_factory=dict)


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = raw["id"]
        if case_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate case id {case_id!r}")
        seen.add(case_id)
        cases.append(
            Case(
                id=case_id,
                question=raw["question"],
                relevant_docs=tuple(raw.get("relevant_docs", [])),
                must_include=tuple(tuple(alts) for alts in raw.get("must_include", [])),
                must_not_include=tuple(raw.get("must_not_include", [])),
                should_refuse=bool(raw.get("should_refuse", False)),
                category=raw.get("category", "general"),
            )
        )
    return cases


def load_documents(kb_dir: Path) -> dict[str, Document]:
    docs = {p.stem: Document(p.stem, p.read_text(encoding="utf-8")) for p in sorted(kb_dir.glob("*.md"))}
    if not docs:
        raise ValueError(f"no .md documents found in {kb_dir}")
    return docs


def load_dataset(cases_path: Path, kb_dir: Path) -> Dataset:
    documents = load_documents(kb_dir)
    cases = load_cases(cases_path)
    for case in cases:
        unknown = [d for d in case.relevant_docs if d not in documents]
        if unknown:
            raise ValueError(f"case {case.id!r} references unknown documents {unknown}")
    return Dataset(cases=cases, documents=documents)
