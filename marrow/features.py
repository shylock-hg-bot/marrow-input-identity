"""Scale-free feature extraction from the OS-neutral event stream.

The features are deliberately behavioural rather than device-specific: pointer
positions are already fractions of the screen, and every timing feature is a
duration or a rate.  Nothing here reads a capture application, OS, window size,
or pixel coordinate.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

# Keep this list stable: model files use it as the feature schema.
FEATURE_NAMES = [
    "pointer_speed_median_log",
    "pointer_speed_p90_log",
    "pointer_accel_median_log",
    "pointer_pause_ratio",
    "pointer_gap_median_log",
    "pointer_direction_change_rate",
    "pointer_path_efficiency",
    "pointer_curvature_mean",
    "click_rate",
    "click_interval_cv",
    "click_interval_median_log",
    "click_position_spread",
    "text_chars_log",
    "text_rate_log",
    "text_interval_median_log",
    "text_interval_cv",
    "text_burst_size_median_log",
    "backspace_ratio",
    "dictation_char_ratio",
    "typed_char_ratio",
    "onscreen_char_ratio",
]


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, value))


def _median(values: list[float], default: float = 0.0) -> float:
    return float(statistics.median(values)) if values else default


def _cv(values: list[float], default: float = 0.0) -> float:
    """Coefficient of variation, stable for tiny samples."""
    if len(values) < 2:
        return default
    mean = statistics.fmean(values)
    if abs(mean) < 1e-12:
        return default
    return float(statistics.pstdev(values) / abs(mean))


def _quantile(values: list[float], fraction: float, default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _angle_change(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = a[0] * b[0] + a[1] * b[1]
    norm = math.hypot(*a) * math.hypot(*b)
    if norm <= 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / norm))
    return math.acos(cosine)


def extract_features(session: dict[str, Any]) -> list[float]:
    """Return one finite feature vector for a validated session."""
    events = session["events"]
    moves = [e for e in events if e["type"] == "move"]
    clicks = [e for e in events if e["type"] == "click"]
    texts = [e for e in events if e["type"] == "text"]

    # ---- pointer dynamics -------------------------------------------------
    speeds: list[float] = []
    accelerations: list[float] = []
    gaps: list[float] = []
    direction_changes = 0
    total_distance = 0.0
    curvature: list[float] = []
    previous_direction: tuple[float, float] | None = None
    for previous, current in zip(moves, moves[1:]):
        dt_ms = current["t"] - previous["t"]
        if dt_ms <= 0:
            continue
        dt_s = dt_ms / 1000.0
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        distance = math.hypot(dx, dy)
        speed = distance / dt_s
        speeds.append(speed)
        gaps.append(dt_s)
        total_distance += distance
        direction = (dx, dy) if distance > 1e-12 else None
        if direction is not None and previous_direction is not None:
            change = _angle_change(previous_direction, direction)
            direction_changes += int(change > math.pi / 3.0)
            if distance > 1e-9:
                curvature.append(change / distance)
        if direction is not None:
            previous_direction = direction

    for left, right in zip(speeds, speeds[1:]):
        if right >= 0:
            accelerations.append(abs(right - left))

    if moves:
        first, last = moves[0], moves[-1]
        net_displacement = math.hypot(last["x"] - first["x"], last["y"] - first["y"])
        elapsed = max(1e-9, (moves[-1]["t"] - moves[0]["t"]) / 1000.0)
    else:
        net_displacement = 0.0
        elapsed = 1.0
    path_efficiency = net_displacement / total_distance if total_distance > 1e-9 else 0.0
    pause_ratio = (
        sum(1 for gap in gaps if gap >= 0.30) / len(gaps) if gaps else 0.0
    )
    pointer_duration = max(1e-9, elapsed)
    direction_change_rate = direction_changes / pointer_duration

    # ---- clicks -----------------------------------------------------------
    click_times = [e["t"] / 1000.0 for e in clicks]
    click_intervals = [
        right - left for left, right in zip(click_times, click_times[1:]) if right > left
    ]
    click_rate = len(clicks) / pointer_duration
    click_spread = 0.0
    if len(clicks) >= 2:
        xs = [e["x"] for e in clicks]
        ys = [e["y"] for e in clicks]
        click_spread = math.sqrt(statistics.pvariance(xs) + statistics.pvariance(ys))

    # ---- text arrival -----------------------------------------------------
    inserted = 0
    deleted = 0
    source_chars = {"typed": 0, "onscreen": 0, "dictation": 0}
    text_times: list[float] = []
    correction_runs: list[int] = []
    current_run = 0
    for event in texts:
        text = event["text"]
        source = event["source"]
        text_times.append(event["t"] / 1000.0)
        if text == "\b":
            deleted += 1
            current_run += 1
            continue
        if current_run:
            correction_runs.append(current_run)
            current_run = 0
        count = len(text)
        inserted += count
        source_chars[source] += count
    if current_run:
        correction_runs.append(current_run)

    text_intervals = [
        right - left for left, right in zip(text_times, text_times[1:]) if right > left
    ]
    positive_interval = [value for value in text_intervals if value > 0]
    text_span = (
        max(1e-9, text_times[-1] - text_times[0]) if len(text_times) >= 2 else 1.0
    )
    text_rate = inserted / text_span
    bursts: list[int] = []
    burst = 0
    for event in texts:
        if event["text"] == "\b":
            if burst:
                bursts.append(burst)
            burst = 0
        else:
            burst += len(event["text"])
    if burst:
        bursts.append(burst)

    total_source_chars = sum(source_chars.values())
    denominator = max(1, inserted)
    vector = [
        _log1p(_median(speeds)),
        _log1p(_quantile(speeds, 0.90)),
        _log1p(_median(accelerations)),
        pause_ratio,
        _log1p(_median(gaps)),
        direction_change_rate,
        path_efficiency,
        _median(curvature),
        click_rate,
        _cv(click_intervals),
        _log1p(_median(click_intervals)),
        click_spread,
        _log1p(inserted),
        _log1p(text_rate),
        _log1p(_median(positive_interval)),
        _cv(positive_interval),
        _log1p(_median(bursts)),
        deleted / denominator,
        source_chars["dictation"] / max(1, total_source_chars),
        source_chars["typed"] / max(1, total_source_chars),
        source_chars["onscreen"] / max(1, total_source_chars),
    ]

    # The model must never see NaN/inf, even on a malformed but accepted stream.
    clean: list[float] = []
    for value in vector:
        if not math.isfinite(value):
            value = 0.0
        clean.append(float(value))
    return clean
