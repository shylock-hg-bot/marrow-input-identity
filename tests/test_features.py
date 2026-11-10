import math
import unittest

from marrow.events import RecordingError, validate_session
from marrow.features import FEATURE_NAMES, extract_features


def session(events):
    return validate_session(
        {
            "session_id": "test-session",
            "subject_id": "tester",
            "input_style": "mouse_keyboard",
            "role": "owner",
            "tired": False,
            "events": events,
        }
    )


class FeatureTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_schema_sized(self):
        events = [
            {"t": 0, "type": "move", "x": 0.1, "y": 0.2},
            {"t": 40, "type": "move", "x": 0.2, "y": 0.25},
            {"t": 90, "type": "click", "x": 0.2, "y": 0.25, "button": "left"},
            {"t": 200, "type": "text", "text": "h", "source": "typed"},
            {"t": 250, "type": "text", "text": "\b", "source": "typed"},
            {"t": 300, "type": "text", "text": "hi", "source": "dictation"},
        ]
        vector = extract_features(session(events))
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in vector))

    def test_source_mix_and_correction_features(self):
        events = []
        for index in range(20):
            events.append(
                {"t": index * 10, "type": "move", "x": 0.1 + index / 100, "y": 0.5}
            )
        for index in range(10):
            events.append(
                {"t": 300 + index * 100, "type": "text", "text": "x", "source": "typed"}
            )
        events.append({"t": 1500, "type": "text", "text": "\b", "source": "typed"})
        vector = extract_features(session(events))
        named = dict(zip(FEATURE_NAMES, vector))
        self.assertGreater(named["backspace_ratio"], 0.0)
        self.assertAlmostEqual(
            named["typed_char_ratio"]
            + named["onscreen_char_ratio"]
            + named["dictation_char_ratio"],
            1.0,
            places=6,
        )

    def test_rejects_out_of_range_coordinates(self):
        with self.assertRaises(RecordingError):
            session([{"t": 0, "type": "move", "x": 1.2, "y": 0.5}])

    def test_rejects_unknown_text_source(self):
        with self.assertRaises(RecordingError):
            session([{"t": 0, "type": "text", "text": "x", "source": "paste"}])

    def test_rejects_time_going_backwards(self):
        with self.assertRaises(RecordingError):
            session(
                [
                    {"t": 10, "type": "move", "x": 0.1, "y": 0.2},
                    {"t": 5, "type": "move", "x": 0.2, "y": 0.2},
                ]
            )


if __name__ == "__main__":
    unittest.main()
