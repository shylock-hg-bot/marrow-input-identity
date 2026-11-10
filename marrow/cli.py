"""Command-line entry point for training, scoring and the reproducible demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_sessions, write_report
from .events import load_sessions, write_sessions
from .features import FEATURE_NAMES, extract_features
from .model import (
    build_model,
    fit_reference,
    load_model,
    predict_identification,
    save_model,
)
from .synth import generate_dataset


def cmd_generate(args: argparse.Namespace) -> int:
    sessions = generate_dataset(seed=args.seed, subjects_per_style=args.subjects)
    write_sessions(args.out, sessions)
    print(f"wrote {len(sessions)} synthetic sessions to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    sessions = load_sessions(args.recordings)
    result = evaluate_sessions(sessions)
    write_report(result, args.json_out, args.markdown_out)
    print(json.dumps(result["results"], indent=2, sort_keys=True))
    print(f"wrote {args.json_out} and {args.markdown_out}")
    return 0


def _select_enrolment(sessions, subject_id):
    matching = [session for session in sessions if session["subject_id"] == subject_id]
    if not matching:
        subjects = sorted({session["subject_id"] for session in sessions})
        raise SystemExit(
            f"subject {subject_id!r} not found; available: {', '.join(subjects)}"
        )
    labelled = [session for session in matching if "split" in session]
    if labelled:
        matching = [
            session
            for session in matching
            if session.get("split", "train") == "train"
            and session.get("role", "owner") == "owner"
        ]
    return matching


def cmd_train(args: argparse.Namespace) -> int:
    sessions = load_sessions(args.enrol)
    if args.background:
        background = load_sessions(args.background)
        # Keep only background subjects, and avoid duplicate session ids.
        seen = {session["session_id"] for session in sessions}
        sessions = sessions + [
            session for session in background if session["session_id"] not in seen
        ]
    enrolment = _select_enrolment(sessions, args.subject_id)
    if len(enrolment) < 2:
        raise SystemExit("need at least two enrolment recordings")
    model = build_model(sessions, args.subject_id)
    model["training_recordings"] = [s["session_id"] for s in enrolment]
    save_model(args.out, model)
    print(
        f"trained {args.subject_id} from {len(enrolment)} recordings; "
        f"threshold={model['profiles'][args.subject_id]['threshold']:.2f}; "
        f"wrote {args.out}"
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    sessions = load_sessions(args.recordings)
    output = []
    for session in sessions:
        if args.subject_id and session["subject_id"] != args.subject_id:
            continue
        prediction = predict_identification(
            model, session["subject_id"], extract_features(session)
        )
        output.append(
            {
                "session_id": session["session_id"],
                "subject_id": session["subject_id"],
                "input_style": session.get("input_style"),
                **prediction,
            }
        )
    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in output) + "\n"
        )
        print(f"scored {len(output)} recordings; wrote {args.out}")
    else:
        for row in output:
            print(json.dumps(row, sort_keys=True))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    sessions = load_sessions(args.recordings)
    for session in sessions:
        print(
            json.dumps(
                {
                    "session_id": session["session_id"],
                    "feature_names": FEATURE_NAMES,
                    "features": extract_features(session),
                },
                sort_keys=True,
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m marrow.cli",
        description="Offline behaviour-based identity prototype for the Marrow listing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write the deterministic fake dataset")
    generate.add_argument("--out", default="data/synthetic.jsonl")
    generate.add_argument("--seed", type=int, default=1883)
    generate.add_argument("--subjects", type=int, default=5)
    generate.set_defaults(func=cmd_generate)

    train = subparsers.add_parser(
        "train", help="train one profile from enrolment recordings"
    )
    train.add_argument("--subject-id", required=True)
    train.add_argument("--enrol", required=True, help="JSONL recordings")
    train.add_argument(
        "--background",
        help="optional JSONL recordings from other people, used as negatives",
    )
    train.add_argument("--out", required=True, help="model JSON path")
    train.set_defaults(func=cmd_train)

    score = subparsers.add_parser(
        "score", help="score recordings against saved profile(s)"
    )
    score.add_argument("--model", required=True)
    score.add_argument("--recordings", required=True)
    score.add_argument("--subject-id")
    score.add_argument("--out")
    score.set_defaults(func=cmd_score)

    features = subparsers.add_parser(
        "features", help="print the feature vector for each recording"
    )
    features.add_argument("--recordings", required=True)
    features.set_defaults(func=cmd_features)

    evaluate = subparsers.add_parser(
        "evaluate", help="run held-out metrics and the swap test on labelled data"
    )
    evaluate.add_argument("--recordings", required=True)
    evaluate.add_argument("--json-out", default="reports/demo_report.json")
    evaluate.add_argument("--markdown-out", default="reports/demo_report.md")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
