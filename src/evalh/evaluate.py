"""Runs every case through retrieval + a responder and scores the answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from evalh import metrics
from evalh.dataset import Case, Dataset
from evalh.responders import Responder
from evalh.retriever import Bm25Retriever


@dataclass
class CaseResult:
    case_id: str
    category: str
    retrieved: list[str]
    answer: str
    recall_at_k: float
    reciprocal_rank: float
    fact_coverage: float
    groundedness: float
    citation_validity: float
    refusal_correct: bool
    forbidden: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.refusal_correct
            and not self.forbidden
            and not self.pii
            and self.fact_coverage == 1.0
            and self.citation_validity == 1.0
        )


@dataclass
class RunSummary:
    results: list[CaseResult]
    scores: dict[str, float]


def evaluate_case(case: Case, retriever: Bm25Retriever, responder: Responder, k: int) -> CaseResult:
    hits = retriever.search(case.question, k=k)
    retrieved = [h.doc_id for h in hits]
    context = retriever.context(hits)
    answer = responder.answer(case.id, case.question, context)
    refused = metrics.is_refusal(answer)
    return CaseResult(
        case_id=case.id,
        category=case.category,
        retrieved=retrieved,
        answer=answer,
        recall_at_k=metrics.recall_at_k(retrieved, case.relevant_docs),
        reciprocal_rank=metrics.reciprocal_rank(retrieved, case.relevant_docs),
        fact_coverage=1.0 if case.should_refuse else metrics.fact_coverage(answer, case.must_include),
        groundedness=1.0 if refused else metrics.groundedness(answer, context),
        citation_validity=metrics.citation_validity(answer, retrieved),
        refusal_correct=refused == case.should_refuse,
        forbidden=metrics.forbidden_hits(answer, case.must_not_include),
        pii=metrics.pii_findings(answer),
    )


def run(dataset: Dataset, responder: Responder, k: int = 3) -> RunSummary:
    retriever = Bm25Retriever(dataset.documents)
    results = [evaluate_case(c, retriever, responder, k) for c in dataset.cases]
    answerable = [r for r, c in zip(results, dataset.cases, strict=True) if not c.should_refuse]
    scores = {
        "pass_rate": mean(1.0 if r.passed else 0.0 for r in results),
        "retrieval_recall_at_k": mean(r.recall_at_k for r in answerable) if answerable else 1.0,
        "retrieval_mrr": mean(r.reciprocal_rank for r in answerable) if answerable else 1.0,
        "fact_coverage": mean(r.fact_coverage for r in answerable) if answerable else 1.0,
        "groundedness": mean(r.groundedness for r in results),
        "citation_validity": mean(r.citation_validity for r in results),
        "refusal_accuracy": mean(1.0 if r.refusal_correct else 0.0 for r in results),
        "forbidden_content_cases": float(sum(1 for r in results if r.forbidden)),
        "pii_leak_cases": float(sum(1 for r in results if r.pii)),
    }
    return RunSummary(results=results, scores=scores)
