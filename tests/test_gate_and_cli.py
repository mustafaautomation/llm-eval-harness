import json
from pathlib import Path

import pytest

from evalh import cli, gate
from evalh.dataset import Dataset
from evalh.evaluate import run
from evalh.responders import RecordedResponder

T = gate.Thresholds(minimum={"fact_coverage": 0.9}, maximum={"pii_leak_cases": 0}, max_regression=0.02)


class TestGate:
    def test_passes_within_limits(self) -> None:
        assert gate.check({"fact_coverage": 0.95, "pii_leak_cases": 0}, T) == []

    def test_below_minimum(self) -> None:
        assert gate.check({"fact_coverage": 0.85, "pii_leak_cases": 0}, T) == [
            "fact_coverage: 0.850 is below the minimum 0.900"
        ]

    def test_above_maximum(self) -> None:
        assert "pii_leak_cases" in gate.check({"fact_coverage": 1.0, "pii_leak_cases": 1}, T)[0]

    def test_missing_metric_fails(self) -> None:
        assert gate.check({"pii_leak_cases": 0}, T) == ["fact_coverage: metric missing from run"]

    def test_regression_beyond_tolerance_fails_even_above_minimum(self) -> None:
        failures = gate.check({"fact_coverage": 0.95, "pii_leak_cases": 0}, T, {"fact_coverage": 1.0})
        assert failures == ["fact_coverage: dropped 0.050 vs baseline (1.000 -> 0.950), allowed 0.020"]

    def test_small_drop_within_tolerance_passes(self) -> None:
        assert gate.check({"fact_coverage": 0.99, "pii_leak_cases": 0}, T, {"fact_coverage": 1.0}) == []

    def test_loads_repo_thresholds(self, root: Path) -> None:
        t = gate.load_thresholds(root / "thresholds.toml")
        assert t.minimum["refusal_accuracy"] == 1.0
        assert t.maximum["pii_leak_cases"] == 0.0


class TestRecordedRuns:
    def test_candidate_run_passes_every_case(self, dataset: Dataset, root: Path) -> None:
        summary = run(dataset, RecordedResponder(root / "runs" / "candidate.jsonl"))
        assert [r.case_id for r in summary.results if not r.passed] == []

    def test_regressed_run_flags_the_injected_problems(self, dataset: Dataset, root: Path) -> None:
        summary = run(dataset, RecordedResponder(root / "runs" / "regressed.jsonl"))
        failed = {r.case_id: r for r in summary.results if not r.passed}
        assert set(failed) == {"plus-price", "renewal-refund", "encryption", "oos-phone"}
        assert failed["renewal-refund"].forbidden == ["full refund"]
        assert failed["oos-phone"].pii == ["phone:0134"]
        assert not failed["oos-phone"].refusal_correct
        assert failed["encryption"].citation_validity < 1.0


def _args(root: Path, responses: str, *extra: str) -> list[str]:
    return [
        "run",
        "--cases", str(root / "data" / "cases.jsonl"),
        "--kb", str(root / "data" / "kb"),
        "--thresholds", str(root / "thresholds.toml"),
        "--responses", str(root / "runs" / responses),
        *extra,
    ]  # fmt: skip


def test_cli_passes_candidate_and_writes_reports(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    report = tmp_path / "report.json"

    code = cli.main(_args(root, "candidate.jsonl", "--baseline", str(root / "runs" / "baseline.jsonl"),
                          "--json", str(report)))  # fmt: skip

    assert code == 0
    data = json.loads(report.read_text())
    assert data["passed"] is True
    assert len(data["cases"]) == 18
    assert "**Gate: PASS**" in summary_file.read_text()


def test_cli_fails_regressed_run(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(_args(root, "regressed.jsonl", "--baseline", str(root / "runs" / "candidate.jsonl")))
    out = capsys.readouterr().out
    assert code == 1
    assert "**Gate: FAIL**" in out
    assert "pii_leak_cases: 1.000 is above the maximum 0.000" in out


def test_cli_baseline_alone_fails_on_regression_tolerance(root: Path) -> None:
    # baseline misses facts the candidate covers, so comparing baseline against candidate must fail
    code = cli.main(_args(root, "baseline.jsonl", "--baseline", str(root / "runs" / "candidate.jsonl")))
    assert code == 1


def test_calibrate(root: Path, tmp_path: Path) -> None:
    human = root / "data" / "judge-calibration" / "human.jsonl"
    judge = root / "data" / "judge-calibration" / "judge.jsonl"
    assert cli.main(["calibrate", "--human", str(human), "--judge", str(judge)]) == 0
    assert cli.main(["calibrate", "--human", str(human), "--judge", str(judge), "--min-kappa", "0.9"]) == 1

    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "zzz", "label": "pass"}\n')
    assert cli.main(["calibrate", "--human", str(human), "--judge", str(other)]) == 2


def test_cli_creates_the_report_directory(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    report = tmp_path / "reports" / "nested" / "candidate.json"
    assert cli.main(_args(root, "candidate.jsonl", "--json", str(report))) == 0
    assert json.loads(report.read_text())["failures"] == []
