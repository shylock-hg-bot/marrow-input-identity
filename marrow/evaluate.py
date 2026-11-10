"""Held-out evaluation and a swap test for the fake-data demo."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .features import FEATURE_NAMES, extract_features
from .model import MARGIN, THRESHOLD, build_model, predict_identification


def _is_enrolment(session: dict[str, Any]) -> bool:
    return (
        session.get("role", "owner") == "owner"
        and session.get("split", "train") == "train"
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_sessions(sessions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    enrolment = [session for session in sessions if _is_enrolment(session)]
    test = [session for session in sessions if session.get("split") == "test"]
    if not enrolment:
        raise ValueError("no enrolment recordings found")
    if not test:
        raise ValueError("no held-out recordings found")

    model = build_model(enrolment)

    decisions: list[dict[str, Any]] = []
    for session in test:
        subject_id = session["subject_id"]
        if subject_id not in model["profiles"]:
            continue
        prediction = predict_identification(
            model, subject_id, extract_features(session)
        )
        decisions.append(
            {
                "session_id": session["session_id"],
                "subject_id": subject_id,
                "input_style": session.get("input_style"),
                "role": session.get("role"),
                "tired": bool(session.get("tired")),
                **prediction,
            }
        )

    owner = [row for row in decisions if row["role"] == "owner"]
    stranger = [row for row in decisions if row["role"] == "stranger"]
    mimic = [row for row in decisions if row["role"] == "mimic"]
    owner_rejected = sum(not row["accepted"] for row in owner)
    stranger_accepted = sum(row["accepted"] for row in stranger)
    mimic_accepted = sum(row["accepted"] for row in mimic)

    # Swap test: for every profile B, try the genuine owner test recordings of
    # every *other* subject A in the same style.  B must reject A's recording.
    swap_attempts: list[dict[str, Any]] = []
    test_by_id = {session["session_id"]: session for session in test}
    for owner_session in owner:
        actual_subject = owner_session["subject_id"]
        style = owner_session.get("input_style")
        source_session = test_by_id[owner_session["session_id"]]
        vector = extract_features(source_session)
        for profile_subject in sorted(model["profiles"]):
            profile = model["profiles"][profile_subject]
            if profile_subject == actual_subject:
                continue
            if style is not None and profile.get("input_style") != style:
                continue
            prediction = predict_identification(model, profile_subject, vector)
            swap_attempts.append(
                {
                    "session_id": owner_session["session_id"],
                    "actual_subject_id": actual_subject,
                    "profile_subject_id": profile_subject,
                    **prediction,
                }
            )
    swap_accepted = sum(row["accepted"] for row in swap_attempts)

    by_style: dict[str, dict[str, Any]] = {}
    for style in sorted({row["input_style"] for row in decisions if row["input_style"]}):
        rows = [row for row in decisions if row["input_style"] == style]
        style_owner = [row for row in rows if row["role"] == "owner"]
        style_stranger = [row for row in rows if row["role"] == "stranger"]
        style_mimic = [row for row in rows if row["role"] == "mimic"]
        style_swap = [
            row for row in swap_attempts if row["session_id"].startswith(style + "_")
        ]
        by_style[style] = {
            "owner_recordings": len(style_owner),
            "owner_rejected": sum(not row["accepted"] for row in style_owner),
            "false_reject_rate": _rate(
                sum(not row["accepted"] for row in style_owner), len(style_owner)
            ),
            "stranger_recordings": len(style_stranger),
            "stranger_accepted": sum(row["accepted"] for row in style_stranger),
            "stranger_false_accept_rate": _rate(
                sum(row["accepted"] for row in style_stranger), len(style_stranger)
            ),
            "mimic_recordings": len(style_mimic),
            "mimic_accepted": sum(row["accepted"] for row in style_mimic),
            "mimic_false_accept_rate": _rate(
                sum(row["accepted"] for row in style_mimic), len(style_mimic)
            ),
            "swap_attempts": len(style_swap),
            "swap_accepted": sum(row["accepted"] for row in style_swap),
            "swap_accept_rate": _rate(
                sum(row["accepted"] for row in style_swap), len(style_swap)
            ),
        }

    return {
        "method": (
            "per-user logistic verifier over OS-neutral input features, "
            "with an open-set identification margin"
        ),
        "feature_count": len(FEATURE_NAMES),
        "threshold_policy": {
            "rule": (
                f"profile score >= {THRESHOLD:.2f} AND profile score >= "
                f"best other same-style profile + {MARGIN:.2f}"
            ),
            "estimated_before_test": True,
            "test_recordings_used": False,
        },
        "dataset": {
            "sessions": len(sessions),
            "enrolment": len(enrolment),
            "held_out": len(test),
            "styles": sorted({session.get("input_style") for session in sessions}),
            "roles": dict(Counter(session.get("role") for session in sessions)),
            "tired_owner_test_recordings": sum(
                1
                for session in test
                if session.get("role") == "owner" and session.get("tired")
            ),
        },
        "profiles": {
            subject_id: {
                "enrolment_recordings": len(
                    [
                        session
                        for session in enrolment
                        if session["subject_id"] == subject_id
                    ]
                ),
                "background_recordings": profile["background_recordings"],
                "threshold": profile["threshold"],
                "margin": profile["margin"],
            }
            for subject_id, profile in sorted(model["profiles"].items())
        },
        "results": {
            "owner_recordings": len(owner),
            "owner_rejected": owner_rejected,
            "false_reject_rate": _rate(owner_rejected, len(owner)),
            "stranger_recordings": len(stranger),
            "stranger_accepted": stranger_accepted,
            "stranger_false_accept_rate": _rate(stranger_accepted, len(stranger)),
            "mimic_recordings": len(mimic),
            "mimic_accepted": mimic_accepted,
            "mimic_false_accept_rate": _rate(mimic_accepted, len(mimic)),
            "impostor_recordings": len(stranger) + len(mimic),
            "impostor_accepted": stranger_accepted + mimic_accepted,
            "impostor_false_accept_rate": _rate(
                stranger_accepted + mimic_accepted, len(stranger) + len(mimic)
            ),
            "swap_attempts": len(swap_attempts),
            "swap_accepted": swap_accepted,
            "swap_accept_rate": _rate(swap_accepted, len(swap_attempts)),
        },
        "by_style": by_style,
        "decisions": decisions,
        "swap_test": swap_attempts,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def results_to_markdown(result: dict[str, Any]) -> str:
    r = result["results"]
    lines = [
        "# Marrow prototype — held-out report",
        "",
        f"Method: **{result['method']}**  ",
        f"Features: **{result['feature_count']}** scale-free behavioural features  ",
        f"Threshold policy: **{result['threshold_policy']['rule']}**, "
        "fixed before any held-out recording was scored.",
        "",
        "## Dataset",
        "",
        f"- Sessions: {result['dataset']['sessions']} "
        f"({result['dataset']['enrolment']} enrolment, "
        f"{result['dataset']['held_out']} held out)",
        f"- Input styles: {', '.join(result['dataset']['styles'])}",
        f"- Roles: {json.dumps(result['dataset']['roles'], sort_keys=True)}",
        f"- Tired-owner held-out recordings: "
        f"{result['dataset']['tired_owner_test_recordings']}",
        "",
        "## Results",
        "",
        "| Measure | Count | Rate |",
        "|---|---:|---:|",
        f"| Real owner doubted (false reject) | {r['owner_rejected']} / {r['owner_recordings']} | {_pct(r['false_reject_rate'])} |",
        f"| Stranger let in (false accept) | {r['stranger_accepted']} / {r['stranger_recordings']} | {_pct(r['stranger_false_accept_rate'])} |",
        f"| Mimic let in (false accept) | {r['mimic_accepted']} / {r['mimic_recordings']} | {_pct(r['mimic_false_accept_rate'])} |",
        f"| Any impostor let in | {r['impostor_accepted']} / {r['impostor_recordings']} | {_pct(r['impostor_false_accept_rate'])} |",
        f"| Swap test: profile trained on someone else accepts the owner | {r['swap_accepted']} / {r['swap_attempts']} | {_pct(r['swap_accept_rate'])} |",
        "",
        "## By input style",
        "",
        "| Style | Owner FRR | Stranger FAR | Mimic FAR | Swap accept |",
        "|---|---:|---:|---:|---:|",
    ]
    for style, row in result["by_style"].items():
        lines.append(
            f"| `{style}` | {_pct(row['false_reject_rate'])} "
            f"({row['owner_rejected']}/{row['owner_recordings']}) | "
            f"{_pct(row['stranger_false_accept_rate'])} "
            f"({row['stranger_accepted']}/{row['stranger_recordings']}) | "
            f"{_pct(row['mimic_false_accept_rate'])} "
            f"({row['mimic_accepted']}/{row['mimic_recordings']}) | "
            f"{_pct(row['swap_accept_rate'])} "
            f"({row['swap_accepted']}/{row['swap_attempts']}) |"
        )
    lines += [
        "",
        "## Profiles",
        "",
        "| Profile | Enrolment recordings | Background negatives | Threshold | Margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for subject, row in result["profiles"].items():
        lines.append(
            f"| `{subject}` | {row['enrolment_recordings']} | "
            f"{row['background_recordings']} | {row['threshold']:.2f} | "
            f"{row['margin']:.2f} |"
        )
    lines += [
        "",
        "A stranger/mimic accepted here is a **false accept**. An owner rejected is a "
        "**false reject**. The threshold and margin were not changed after these "
        "numbers were computed.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict[str, Any], json_path: str, markdown_path: str) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path(markdown_path).write_text(results_to_markdown(result) + "\n")
