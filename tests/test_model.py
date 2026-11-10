import unittest

from marrow.evaluate import evaluate_sessions
from marrow.model import build_model
from marrow.synth import generate_dataset


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = generate_dataset(seed=1883, subjects_per_style=5)

    def test_demo_has_two_styles_and_all_roles(self):
        styles = {session["input_style"] for session in self.sessions}
        roles = {session["role"] for session in self.sessions}
        self.assertEqual(styles, {"mouse_keyboard", "slow_pointer_dictation"})
        self.assertEqual(roles, {"owner", "stranger", "mimic"})
        self.assertTrue(any(session.get("tired") for session in self.sessions))

    def test_swap_test_passes_and_rates_are_reported(self):
        result = evaluate_sessions(self.sessions)
        metrics = result["results"]
        self.assertEqual(metrics["swap_accepted"], 0)
        self.assertEqual(metrics["swap_accept_rate"], 0.0)
        self.assertGreater(metrics["owner_recordings"], 0)
        self.assertGreater(metrics["impostor_recordings"], 0)
        for key in (
            "false_reject_rate",
            "stranger_false_accept_rate",
            "mimic_false_accept_rate",
            "swap_accept_rate",
        ):
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_profile_needs_background_negatives(self):
        one_subject = [
            session
            for session in self.sessions
            if session["subject_id"] == "mouse_keyboard_subject_01"
            and session.get("split") == "train"
        ]
        with self.assertRaises(ValueError):
            build_model(one_subject, "mouse_keyboard_subject_01")


if __name__ == "__main__":
    unittest.main()
