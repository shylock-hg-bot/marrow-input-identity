"""Plain, operating-system-neutral recording format.

A recording file is UTF-8 JSON Lines.  Each non-empty line is one session:

    {
      "session_id": "unique-string",
      "subject_id": "enrolled-user-id",
      "input_style": "mouse_keyboard" | "slow_pointer_dictation" | ...,
      "role": "owner" | "stranger" | "mimic",       # evaluation label
      "tired": false,
      "events": [
        {"t": 0,   "type": "move",  "x": 0.12, "y": 0.44},
        {"t": 19,  "type": "click", "x": 0.12, "y": 0.44, "button": "left"},
        {"t": 50,  "type": "text",  "text": "h", "source": "typed"},
        {"t": 73,  "type": "text",  "text": "\b", "source": "typed"},
        {"t": 91,  "type": "text",  "text": "hello", "source": "dictation"}
      ]
    }

Requirements represented by the format:

* ``t`` is a millisecond timestamp.  It may be relative to any origin; only
  differences are used.
* ``x`` and ``y`` are fractions of the screen in [0, 1], not pixels.
* Clicks are explicit click events.  A click may carry ``button`` and
  ``duration_ms``; neither is required.
* Text events carry the arriving text in ``text`` and the arrival path in
  ``source``.  ``source`` must be one of ``typed``, ``onscreen``, or
  ``dictation``.  A deletion/correction is represented by the text ``"\\b"``
  (one or more consecutive backspace events).
* Extra metadata (capture application, OS, display name, ...) is allowed and
  ignored by feature extraction.  The method never infers identity from where
  or how a recording was captured.

The loader validates shape and ranges, but deliberately leaves session labels
and roles optional so a real deployment can score unlabelled recordings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

VALID_SOURCES = {"typed", "onscreen", "dictation"}
VALID_EVENT_TYPES = {"move", "click", "text"}


class RecordingError(ValueError):
    """Raised for a malformed recording."""


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordingError(f"{field} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise RecordingError(f"{field} must be finite")
    return number


def validate_session(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RecordingError("session must be a JSON object")
    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RecordingError("session_id must be a non-empty string")
    subject_id = raw.get("subject_id")
    if not isinstance(subject_id, str) or not subject_id:
        raise RecordingError("subject_id must be a non-empty string")
    events = raw.get("events")
    if not isinstance(events, list):
        raise RecordingError("events must be a list")

    clean_events: list[dict[str, Any]] = []
    last_t = -1.0
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise RecordingError(f"event {index} must be an object")
        kind = event.get("type")
        if kind not in VALID_EVENT_TYPES:
            raise RecordingError(f"event {index} has unsupported type {kind!r}")
        t = _number(event.get("t"), f"event {index}.t")
        if t < 0:
            raise RecordingError(f"event {index}.t must be non-negative")
        if t < last_t:
            raise RecordingError(f"event {index}.t goes backwards")
        last_t = t

        item: dict[str, Any] = {"t": t, "type": kind}
        if kind in {"move", "click"}:
            for axis in ("x", "y"):
                coordinate = _number(event.get(axis), f"event {index}.{axis}")
                if not 0.0 <= coordinate <= 1.0:
                    raise RecordingError(
                        f"event {index}.{axis} must be a fraction in [0, 1]"
                    )
                item[axis] = coordinate
        if kind == "click":
            if event.get("button") is not None:
                if not isinstance(event["button"], str):
                    raise RecordingError(f"event {index}.button must be a string")
                item["button"] = event["button"]
            if event.get("duration_ms") is not None:
                item["duration_ms"] = _number(
                    event["duration_ms"], f"event {index}.duration_ms"
                )
        if kind == "text":
            text = event.get("text")
            if not isinstance(text, str) or not text:
                raise RecordingError(f"event {index}.text must be a non-empty string")
            source = event.get("source")
            if source not in VALID_SOURCES:
                raise RecordingError(
                    f"event {index}.source must be one of {sorted(VALID_SOURCES)}"
                )
            item["text"] = text
            item["source"] = source
        clean_events.append(item)

    clean = dict(raw)
    clean["events"] = clean_events
    if "role" in raw and raw["role"] is not None and not isinstance(raw["role"], str):
        raise RecordingError("role must be a string when present")
    if "tired" in raw and not isinstance(raw["tired"], bool):
        raise RecordingError("tired must be a boolean when present")
    return clean


def load_sessions(path: str | Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecordingError(f"{path}:{line_number}: {error}") from error
            try:
                sessions.append(validate_session(raw))
            except RecordingError as error:
                raise RecordingError(f"{path}:{line_number}: {error}") from error
    return sessions


def write_sessions(path: str | Path, sessions: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def iter_sessions(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from load_sessions(path)
