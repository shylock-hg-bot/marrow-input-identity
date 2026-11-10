"""Deterministic fake-data generator.

All recordings in the demo are synthetic.  No human input is recorded, uploaded
or required.  The generator creates two input styles, each with enrolment
sessions and held-out sessions for a tired owner, a stranger and a mimic who
copies the owner's coarse habits.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

from .events import write_sessions

STYLES = ("mouse_keyboard", "slow_pointer_dictation")
SOURCES = ("typed", "onscreen", "dictation")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _profile(rng: random.Random, style: str, index: int) -> dict[str, Any]:
    if style == "mouse_keyboard":
        speed = rng.lognormvariate(math.log(0.38), 0.34)
        move_interval = rng.lognormvariate(math.log(0.036), 0.30)
        pause_prob = rng.uniform(0.02, 0.24)
        turn_rate = rng.uniform(0.4, 2.4)
        source_weights = (0.78, 0.15, 0.07)
        text_rate = rng.lognormvariate(math.log(5.2), 0.30)
        chunk_size = 1.4
        correction_rate = rng.uniform(0.015, 0.14)
    else:
        speed = rng.lognormvariate(math.log(0.16), 0.34)
        move_interval = rng.lognormvariate(math.log(0.085), 0.30)
        pause_prob = rng.uniform(0.10, 0.45)
        turn_rate = rng.uniform(0.30, 1.7)
        source_weights = (0.10, 0.05, 0.85)
        text_rate = rng.lognormvariate(math.log(3.7), 0.30)
        chunk_size = rng.uniform(3.5, 13.0)
        correction_rate = rng.uniform(0.02, 0.18)

    return {
        "profile_id": f"{style}_{index}",
        "style": style,
        "pointer_speed": _clamp(speed, 0.06, 1.2),
        "move_interval": _clamp(move_interval, 0.012, 0.20),
        "path_jitter": rng.lognormvariate(math.log(0.08), 0.45),
        "pause_prob": pause_prob,
        "pause_scale": rng.lognormvariate(math.log(0.55), 0.40),
        "turn_rate": turn_rate,
        "click_interval": rng.lognormvariate(math.log(2.6), 0.42),
        "click_spread": rng.uniform(0.06, 0.34),
        "text_rate": _clamp(text_rate, 1.3, 10.5),
        "text_interval_cv": rng.uniform(0.18, 0.80),
        "micro_jitter": rng.uniform(0.05, 0.16),
        "source_weights": source_weights,
        "chunk_size": chunk_size,
        "correction_rate": correction_rate,
    }


def _copy(profile: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = dict(profile)
    result.update(overrides)
    return result


def _mimic_profile(
    rng: random.Random, owner: dict[str, Any], other: dict[str, Any]
) -> dict[str, Any]:
    """Copy the owner's visible pointer/click habits, not their private signature.

    A mimic can watch the pointer speed, click cadence and general style.  The
    text-production rhythm, path wander, pauses and correction behaviour here
    come from a different person (``other``), because those are harder to copy
    from a distance.
    """
    return _copy(
        owner,
        profile_id=owner["profile_id"] + "_mimic",
        pointer_speed=owner["pointer_speed"] * rng.uniform(0.94, 1.06),
        move_interval=owner["move_interval"] * rng.uniform(0.94, 1.06),
        click_interval=owner["click_interval"] * rng.uniform(0.92, 1.08),
        click_spread=owner["click_spread"] * rng.uniform(0.90, 1.10),
        path_jitter=other["path_jitter"] * rng.uniform(0.85, 1.25),
        pause_prob=other["pause_prob"],
        pause_scale=other["pause_scale"],
        turn_rate=other["turn_rate"],
        text_rate=other["text_rate"],
        text_interval_cv=other["text_interval_cv"],
        chunk_size=other["chunk_size"],
        correction_rate=other["correction_rate"],
        micro_jitter=_clamp(owner["micro_jitter"] * rng.uniform(1.5, 2.5), 0.15, 1.1),
    )

def _tired_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return _copy(
        profile,
        profile_id=profile["profile_id"] + "_tired",
        pointer_speed=profile["pointer_speed"] * 0.68,
        move_interval=profile["move_interval"] * 1.35,
        pause_prob=_clamp(profile["pause_prob"] + 0.12, 0.0, 0.8),
        pause_scale=profile["pause_scale"] * 1.35,
        click_interval=profile["click_interval"] * 1.25,
        text_rate=profile["text_rate"] * 0.78,
        correction_rate=_clamp(profile["correction_rate"] * 1.35, 0.0, 1.0),
    )


def _pick_source(rng: random.Random, weights: Iterable[float]) -> str:
    values = list(weights)
    roll = rng.random()
    cumulative = 0.0
    for source, weight in zip(SOURCES, values):
        cumulative += weight
        if roll <= cumulative:
            return source
    return SOURCES[-1]


def _text_event(
    rng: random.Random,
    profile: dict[str, Any],
    t: float,
    source: str,
    text: str,
) -> dict[str, Any]:
    return {
        "t": int(round(t * 1000)),
        "type": "text",
        "text": text,
        "source": source,
    }


def _generate_text_events(
    rng: random.Random,
    profile: dict[str, Any],
    duration_s: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    t = rng.uniform(0.25, 0.8)
    alphabet = "etaoinshrdlucmfwypvbgkjqxz"
    while t < duration_s - 0.5:
        source = _pick_source(rng, profile["source_weights"])
        if source == "dictation":
            chunk = max(1, int(round(rng.lognormvariate(
                math.log(max(1.0, profile["chunk_size"])), 0.35
            ))))
            delay = chunk / max(0.4, profile["text_rate"])
            delay *= rng.uniform(0.75, 1.35)
        elif source == "onscreen":
            chunk = rng.choice([1, 1, 2, 3])
            delay = chunk / max(0.4, profile["text_rate"])
            delay *= rng.uniform(0.9, 1.6)
        else:
            chunk = 1
            delay = 1.0 / max(0.4, profile["text_rate"])
            delay *= math.exp(rng.gauss(0.0, profile["micro_jitter"] * 0.35))

        # Corrections are emitted just before the replacement text.
        correction_probability = profile["correction_rate"]
        if rng.random() < correction_probability:
            corrections = 1 + int(rng.random() < 0.35)
            for _ in range(corrections):
                t += max(0.04, delay * rng.uniform(0.2, 0.5))
                events.append(_text_event(rng, profile, t, source, "\b"))
            t += max(0.05, delay * rng.uniform(0.2, 0.5))

        text = "".join(rng.choice(alphabet) for _ in range(chunk))
        events.append(_text_event(rng, profile, t, source, text))

        jitter = math.exp(rng.gauss(0.0, profile["micro_jitter"] * 0.55))
        t += max(0.03, delay * jitter)

    return events


def _generate_pointer_events(
    rng: random.Random,
    profile: dict[str, Any],
    duration_s: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    x = rng.random() * 0.8 + 0.1
    y = rng.random() * 0.8 + 0.1
    target_x = rng.random() * 0.8 + 0.1
    target_y = rng.random() * 0.8 + 0.1
    t = 0.0
    while t < duration_s:
        base_dt = profile["move_interval"] * math.exp(
            rng.gauss(0.0, profile["micro_jitter"] * 0.55)
        )
        base_dt = max(0.008, base_dt)
        if rng.random() < profile["pause_prob"]:
            base_dt += profile["pause_scale"] * rng.uniform(0.5, 1.8)
        t += base_dt
        if t >= duration_s:
            break

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        step = profile["pointer_speed"] * base_dt * rng.uniform(0.65, 1.35)
        if distance <= step or distance < 1e-6:
            x, y = target_x, target_y
            # Turning toward a new target is where personal path style shows.
            if rng.random() < 0.55:
                target_x = rng.random() * 0.9 + 0.05
                target_y = rng.random() * 0.9 + 0.05
            else:
                target_x = _clamp(x + rng.uniform(-0.35, 0.35), 0.02, 0.98)
                target_y = _clamp(y + rng.uniform(-0.35, 0.35), 0.02, 0.98)
        else:
            ux, uy = dx / distance, dy / distance
            x += ux * step
            y += uy * step
            # Small perpendicular wander, scaled by the user's path jitter.
            wobble = rng.gauss(0.0, profile["path_jitter"] * step)
            x += -uy * wobble
            y += ux * wobble
            if rng.random() < 0.08 * profile["turn_rate"]:
                target_x = _clamp(x + rng.uniform(-0.25, 0.25), 0.02, 0.98)
                target_y = _clamp(y + rng.uniform(-0.25, 0.25), 0.02, 0.98)

        x = _clamp(x, 0.0, 1.0)
        y = _clamp(y, 0.0, 1.0)
        events.append(
            {
                "t": int(round(t * 1000)),
                "type": "move",
                "x": round(x, 6),
                "y": round(y, 6),
            }
        )

    # Clicks are placed at the pointer position at the nearest preceding move.
    click_time = rng.uniform(0.8, 2.0)
    while click_time < duration_s - 0.2 and events:
        nearest = min(events, key=lambda event: abs(event["t"] / 1000.0 - click_time))
        events.append(
            {
                "t": int(round(click_time * 1000)),
                "type": "click",
                "x": nearest["x"],
                "y": nearest["y"],
                "button": "left",
            }
        )
        click_time += profile["click_interval"] * math.exp(
            rng.gauss(0.0, profile["micro_jitter"] * 0.35)
        )
    return events


def generate_session(
    rng: random.Random,
    profile: dict[str, Any],
    *,
    session_id: str,
    subject_id: str,
    role: str,
    tired: bool,
    split: str,
) -> dict[str, Any]:
    effective = _tired_profile(profile) if tired else profile
    duration_s = rng.uniform(35.0, 55.0)
    events = _generate_pointer_events(rng, effective, duration_s)
    events += _generate_text_events(rng, effective, duration_s)
    events.sort(key=lambda event: (event["t"], {"move": 0, "click": 1, "text": 2}[event["type"]]))
    return {
        "session_id": session_id,
        "subject_id": subject_id,
        "input_style": profile["style"],
        "role": role,
        "tired": tired,
        "split": split,
        "capture": {"kind": "synthetic", "os": None, "app": None},
        "events": events,
    }


def generate_dataset(
    *, seed: int = 1883, subjects_per_style: int = 5
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    profiles: dict[str, list[dict[str, Any]]] = {}
    for style in STYLES:
        profiles[style] = [
            _profile(rng, style, index + 1) for index in range(subjects_per_style)
        ]

    sessions: list[dict[str, Any]] = []
    for style in STYLES:
        group = profiles[style]
        for index, owner in enumerate(group):
            subject_id = f"{style}_subject_{index + 1:02d}"
            # Six enrolment recordings.  Two are tired so the profile learns a
            # small amount of within-person variation instead of one mood only.
            for slot in range(6):
                sessions.append(
                    generate_session(
                        rng,
                        owner,
                        session_id=f"{subject_id}_enrol_{slot + 1:02d}",
                        subject_id=subject_id,
                        role="owner",
                        tired=slot in (4, 5),
                        split="train",
                    )
                )
            # Held-out owner recordings: two tired, one alert.
            for slot in range(3):
                sessions.append(
                    generate_session(
                        rng,
                        owner,
                        session_id=f"{subject_id}_owner_test_{slot + 1:02d}",
                        subject_id=subject_id,
                        role="owner",
                        tired=slot < 2,
                        split="test",
                    )
                )
            # Two strangers, using another person's profile from the same style.
            others = [candidate for candidate in group if candidate is not owner]
            for slot in range(2):
                stranger = others[(index + slot) % len(others)]
                sessions.append(
                    generate_session(
                        rng,
                        stranger,
                        session_id=f"{subject_id}_stranger_{slot + 1:02d}",
                        subject_id=subject_id,
                        role="stranger",
                        tired=False,
                        split="test",
                    )
                )
            # Two mimics, copying the target's coarse habits but not micro-motor
            # timing/correction behaviour.
            mimic_other = others[(index + 1) % len(others)]
            mimic = _mimic_profile(rng, owner, mimic_other)
            for slot in range(2):
                sessions.append(
                    generate_session(
                        rng,
                        mimic,
                        session_id=f"{subject_id}_mimic_{slot + 1:02d}",
                        subject_id=subject_id,
                        role="mimic",
                        tired=False,
                        split="test",
                    )
                )
    return sessions


def write_demo_data(path: str, *, seed: int = 1883, subjects_per_style: int = 5) -> int:
    sessions = generate_dataset(seed=seed, subjects_per_style=subjects_per_style)
    write_sessions(path, sessions)
    return len(sessions)
