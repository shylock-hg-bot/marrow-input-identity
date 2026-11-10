# Pre-registered decision rule

This file records the decision rule **before** the held-out synthetic test split
is scored.

- Demo seed: `1883`
- Enrolment recordings per subject: `6` (including two tired)
- Held-out recordings per subject: `3` owner (two tired), `2` stranger, `2` mimic
- Input styles: `mouse_keyboard`, `slow_pointer_dictation`
- Subjects per style: `5`
- Profile score: logistic probability from a per-user verifier
- Background negatives: the other subjects' enrolment recordings only
- Accept when:
  1. the claimed profile score is `>= 0.90`, and
  2. the claimed profile score is `>= best other same-style profile score + 0.20`
     (when another same-style profile exists)
- Training settings: L2 `0.50`, learning rate `0.25`, `1500` full-batch steps,
  robust feature scale floor `0.03`

No held-out recording is used to select the threshold, the margin, the features,
the training settings or the reference scale. The metrics are written by
`python3 run_demo.py` after this file is fixed.

If the threshold is changed later, that is a new deployment policy and requires
a new held-out evaluation; it must not be presented as this run's result.
