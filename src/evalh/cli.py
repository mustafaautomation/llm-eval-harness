"""Command line: ``evalh run`` and ``evalh calibrate``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evalh import gate, report
from evalh.dataset import load_dataset
from evalh.evaluate import run
from evalh.judge import cohen_kappa
from evalh.responders import ChatCompletionsResponder, RecordedResponder, Responder


def cmd_run(args: argparse.Namespace) -> int:
    responder: Responder
    if args.responses:
        responder = RecordedResponder(Path(args.responses))
    else:
        responder = ChatCompletionsResponder(
            args.base_url, args.model, api_key=os.environ.get("EVALH_API_KEY")
        )
    dataset = load_dataset(Path(args.cases), Path(args.kb))
    summary = run(dataset, responder, k=args.k)

    baseline = None
    if args.baseline:
        baseline = run(dataset, RecordedResponder(Path(args.baseline)), k=args.k).scores

    failures = gate.check(summary.scores, gate.load_thresholds(Path(args.thresholds)), baseline)
    markdown = report.to_markdown(summary, failures, baseline)
    print(markdown)
    if args.json:
        Path(args.json).write_text(report.to_json(summary, failures), encoding="utf-8")
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(markdown)
    return 1 if failures else 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    def labels(path: str) -> dict[str, str]:
        rows = [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return {r["id"]: r["label"] for r in rows}

    human, judge = labels(args.human), labels(args.judge)
    ids = sorted(human.keys() & judge.keys())
    if not ids:
        print("no overlapping ids between the two label files", file=sys.stderr)
        return 2
    kappa = cohen_kappa([human[i] for i in ids], [judge[i] for i in ids])
    print(f"cases: {len(ids)}  cohen_kappa: {kappa:.3f}  minimum: {args.min_kappa:.3f}")
    return 0 if kappa >= args.min_kappa else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="evaluate answers and apply the gate")
    r.add_argument("--cases", default="data/cases.jsonl")
    r.add_argument("--kb", default="data/kb")
    r.add_argument("--thresholds", default="thresholds.toml")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--responses", help="recorded answers (JSONL)")
    src.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1")
    r.add_argument("--model", default="llama3.2")
    r.add_argument("--baseline", help="recorded baseline answers to compare against")
    r.add_argument("--json", help="write the full JSON report here")
    r.add_argument("-k", type=int, default=3, help="documents retrieved per question")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="agreement between judge labels and human labels")
    c.add_argument("--human", required=True)
    c.add_argument("--judge", required=True)
    c.add_argument("--min-kappa", type=float, default=0.6)
    c.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
