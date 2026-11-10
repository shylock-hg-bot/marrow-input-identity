#!/usr/bin/env python3
"""One command from a fresh copy: generate fake data, train, test, report.

    python3 run_demo.py

No network, capture device, credential or paid dependency is used.  The output
is deterministic for the default seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from marrow.evaluate import evaluate_sessions, write_report
from marrow.events import load_sessions
from marrow.synth import write_demo_data


def main() -> int:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "synthetic.jsonl"
    report_json = root / "reports" / "demo_report.json"
    report_md = root / "reports" / "demo_report.md"

    count = write_demo_data(str(data_path), seed=1883, subjects_per_style=5)
    print(f"generated {count} synthetic sessions -> {data_path}")
    result = evaluate_sessions(load_sessions(data_path))
    write_report(result, str(report_json), str(report_md))
    metrics = result["results"]
    print(
        "held-out results: "
        f"owner false-reject={metrics['false_reject_rate']:.1%}, "
        f"stranger false-accept={metrics['stranger_false_accept_rate']:.1%}, "
        f"mimic false-accept={metrics['mimic_false_accept_rate']:.1%}, "
        f"swap-test accept={metrics['swap_accept_rate']:.1%}"
    )
    print(f"reports -> {report_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
