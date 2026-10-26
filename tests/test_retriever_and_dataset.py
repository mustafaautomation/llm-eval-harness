import json
from pathlib import Path

import pytest

from evalh.dataset import Dataset, load_cases, load_dataset
from evalh.retriever import Bm25Retriever


def test_every_answerable_case_retrieves_its_document_first(dataset: Dataset) -> None:
    retriever = Bm25Retriever(dataset.documents)
    for case in dataset.cases:
        if case.should_refuse:
            continue
        top = [h.doc_id for h in retriever.search(case.question, k=3)]
        assert top and top[0] in case.relevant_docs, f"{case.id}: got {top}"


def test_results_are_ranked_and_limited(dataset: Dataset) -> None:
    hits = Bm25Retriever(dataset.documents).search("refund payment method business days", k=2)
    assert len(hits) <= 2
    assert hits[0].doc_id == "refunds"
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_unrelated_query_returns_nothing(dataset: Dataset) -> None:
    assert Bm25Retriever(dataset.documents).search("zebra quantum volcano") == []


def test_context_labels_each_document(dataset: Dataset) -> None:
    retriever = Bm25Retriever(dataset.documents)
    context = retriever.context(retriever.search("export notebook pdf", k=1))
    assert context.startswith("[export-data]\n")


def test_dataset_has_answerable_and_out_of_scope_cases(dataset: Dataset) -> None:
    assert sum(c.should_refuse for c in dataset.cases) >= 3
    assert sum(not c.should_refuse for c in dataset.cases) >= 10
    assert all(c.must_include for c in dataset.cases if not c.should_refuse)


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "a", "question": "q"}, {"id": "a", "question": "q2"}])
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_unknown_document_reference_is_rejected(tmp_path: Path, root: Path) -> None:
    path = _write(tmp_path, [{"id": "a", "question": "q", "relevant_docs": ["nope"]}])
    with pytest.raises(ValueError, match="unknown documents"):
        load_dataset(path, root / "data" / "kb")


def test_empty_knowledge_base_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \.md documents"):
        load_dataset(_write(tmp_path, []), tmp_path)
