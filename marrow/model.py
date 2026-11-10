"""Per-user verification profiles with a threshold fixed before testing.

The model is intentionally small enough to audit and uses only the Python
standard library:

1. A robust reference centre/scale is estimated from *enrolment* recordings.
2. Each user's profile is a logistic verifier: positives are that user's
   enrolment recordings, and negatives are the enrolment recordings of the
   other people in the background set.  Each feature is normalised by the
   reference centre/scale first.
3. A per-profile score is the logistic probability that a recording belongs to
   the enrolled user.
4. The acceptance rule is fixed in code before held-out recordings are scored:

       profile score >= 0.90
       AND profile score >= best other same-style profile + 0.20

   The relative term is the open-set identification margin: when several
   people are enrolled, the claimed profile must be the best match, not merely
   over a global probability line.  With a single enrolled profile the
   relative term is skipped.

No test recording contributes to the reference, weights or threshold.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from .events import validate_session
from .features import FEATURE_NAMES, extract_features

SCHEMA_VERSION = 2
THRESHOLD = 0.90
MARGIN = 0.20
L2 = 0.50
LEARNING_RATE = 0.25
TRAINING_STEPS = 1500
SCALE_FLOOR = 0.03
SCALE_REFERENCE_FRACTION = 0.75


def _median(values: Sequence[float], default: float = 0.0) -> float:
    return float(statistics.median(values)) if values else default


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    centre = _median(values)
    return _median([abs(value - centre) for value in values], 0.0)


def fit_reference(feature_rows: Iterable[Sequence[float]]) -> dict[str, list[float]]:
    rows = [list(map(float, row)) for row in feature_rows]
    if not rows:
        raise ValueError("at least one enrolment recording is required")
    width = len(FEATURE_NAMES)
    if any(len(row) != width for row in rows):
        raise ValueError("feature vector width does not match the schema")
    medians = [_median([row[i] for row in rows]) for i in range(width)]
    mads = [_mad([row[i] for row in rows]) for i in range(width)]
    return {"median": medians, "mad": mads}


def normalise_vector(
    vector: Sequence[float], reference: dict[str, Sequence[float]]
) -> list[float]:
    medians = reference["median"]
    scales = [
        max(float(reference["mad"][i]) * SCALE_REFERENCE_FRACTION, SCALE_FLOOR)
        for i in range(len(FEATURE_NAMES))
    ]
    return [
        (float(value) - float(medians[i])) / scales[i]
        for i, value in enumerate(vector)
    ]


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _fit_logistic(
    positives: Sequence[Sequence[float]],
    negatives: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if not positives:
        raise ValueError("a profile needs enrolment positives")
    width = len(positives[0])
    weights = [0.0] * width
    bias = 0.0
    positive_weight = len(negatives) / max(1, len(positives))
    rows = [(row, 1.0, positive_weight) for row in positives]
    rows += [(row, 0.0, 1.0) for row in negatives]

    for _ in range(TRAINING_STEPS):
        gradient = [0.0] * width
        gradient_bias = 0.0
        for row, label, class_weight in rows:
            value = bias + sum(weights[i] * row[i] for i in range(width))
            error = (_sigmoid(value) - label) * class_weight
            for i in range(width):
                gradient[i] += error * row[i]
            gradient_bias += error
        for i in range(width):
            gradient[i] += L2 * weights[i]
        denominator = float(len(rows))
        for i in range(width):
            weights[i] -= LEARNING_RATE * gradient[i] / denominator
        bias -= LEARNING_RATE * gradient_bias / denominator

    return {"weights": weights, "bias": bias}


def _profile_score(profile: dict[str, Any], normalised: Sequence[float]) -> float:
    value = float(profile["bias"]) + sum(
        profile["weights"][i] * normalised[i] for i in range(len(normalised))
    )
    return _sigmoid(value)


def _enrolment_by_subject(
    sessions: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        if session.get("role", "owner") != "owner":
            continue
        if session.get("split", "train") != "train":
            continue
        grouped.setdefault(session["subject_id"], []).append(session)
    return grouped


def build_model(
    sessions: Sequence[dict[str, Any]],
    subject_id: str | None = None,
    reference: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Train profiles from labelled enrolment sessions.

    When ``subject_id`` is omitted, every subject with at least two enrolment
    recordings gets a profile and the others provide background negatives.
    """
    grouped = _enrolment_by_subject(sessions)
    if not grouped:
        raise ValueError("no enrolment recordings found")
    if reference is None:
        all_rows = [
            extract_features(session)
            for rows in grouped.values()
            for session in rows
        ]
        reference = fit_reference(all_rows)

    selected = sorted(grouped) if subject_id is None else [subject_id]
    profiles: dict[str, Any] = {}
    for target in selected:
        positives_raw = [extract_features(s) for s in grouped.get(target, [])]
        if len(positives_raw) < 2:
            raise ValueError(f"not enough enrolment recordings for {target}")
        negatives_raw = [
            extract_features(session)
            for other, rows in grouped.items()
            if other != target
            for session in rows
        ]
        if not negatives_raw:
            raise ValueError(
                f"no background negatives available for {target}; "
                "add at least one other subject's enrolment recordings"
            )
        positives = [normalise_vector(row, reference) for row in positives_raw]
        negatives = [normalise_vector(row, reference) for row in negatives_raw]
        logistic = _fit_logistic(positives, negatives)
        style = grouped[target][0].get("input_style")
        profiles[target] = {
            **logistic,
            "subject_id": target,
            "input_style": style,
            "threshold": THRESHOLD,
            "margin": MARGIN,
            "enrolment_session_ids": [
                session["session_id"] for session in grouped[target]
            ],
            "background_recordings": len(negatives_raw),
        }

    return {
        "reference": reference,
        "profiles": profiles,
        "threshold_policy": {
            "threshold": THRESHOLD,
            "margin": MARGIN,
            "fixed_before_test": True,
            "test_recordings_used": False,
        },
        "training": {
            "l2": L2,
            "learning_rate": LEARNING_RATE,
            "steps": TRAINING_STEPS,
            "scale_floor": SCALE_FLOOR,
            "background": "all other subjects' enrolment recordings",
        },
    }


def score_profile(
    model: dict[str, Any], profile_subject_id: str, vector: Sequence[float]
) -> dict[str, Any]:
    try:
        profile = model["profiles"][profile_subject_id]
    except KeyError as error:
        raise ValueError(f"unknown profile {profile_subject_id!r}") from error
    normalised = normalise_vector(vector, model["reference"])
    return {
        "score": _profile_score(profile, normalised),
        "threshold": float(profile["threshold"]),
    }


def predict_identification(
    model: dict[str, Any],
    claimed_subject_id: str,
    vector: Sequence[float],
) -> dict[str, Any]:
    claimed = model["profiles"].get(claimed_subject_id)
    if claimed is None:
        raise ValueError(f"unknown profile {claimed_subject_id!r}")

    normalised = normalise_vector(vector, model["reference"])
    own_score = _profile_score(claimed, normalised)
    style = claimed.get("input_style")
    competitors: list[tuple[float, str]] = []
    for subject_id, profile in model["profiles"].items():
        if subject_id == claimed_subject_id:
            continue
        if style is not None and profile.get("input_style") != style:
            continue
        competitors.append((_profile_score(profile, normalised), subject_id))
    competitors.sort(reverse=True)
    best_other_score, best_other_id = competitors[0] if competitors else (0.0, None)
    margin_ok = not competitors or own_score >= best_other_score + float(
        claimed["margin"]
    )
    accepted = own_score >= float(claimed["threshold"]) and margin_ok
    return {
        "score": own_score,
        "threshold": float(claimed["threshold"]),
        "best_other_subject_id": best_other_id,
        "best_other_score": best_other_score,
        "relative_margin": own_score - best_other_score if competitors else None,
        "accepted": bool(accepted),
    }


# Backwards-compatible short names used by the CLI and tests.
def predict_profile(
    model: dict[str, Any], profile_subject_id: str, vector: Sequence[float]
) -> dict[str, Any]:
    return predict_identification(model, profile_subject_id, vector)


def features_for_sessions(sessions: Iterable[dict[str, Any]]) -> list[list[float]]:
    return [extract_features(validate_session(session)) for session in sessions]


def save_model(path: str | Path, model: dict[str, Any]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        **model,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_model(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text())
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported model schema")
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("model feature schema does not match this code")
    return model
