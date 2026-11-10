# Marrow prototype — held-out report

Method: **per-user logistic verifier over OS-neutral input features, with an open-set identification margin**  
Features: **21** scale-free behavioural features  
Threshold policy: **profile score >= 0.90 AND profile score >= best other same-style profile + 0.20**, fixed before any held-out recording was scored.

## Dataset

- Sessions: 130 (60 enrolment, 70 held out)
- Input styles: mouse_keyboard, slow_pointer_dictation
- Roles: {"mimic": 20, "owner": 90, "stranger": 20}
- Tired-owner held-out recordings: 20

## Results

| Measure | Count | Rate |
|---|---:|---:|
| Real owner doubted (false reject) | 3 / 30 | 10.0% |
| Stranger let in (false accept) | 0 / 20 | 0.0% |
| Mimic let in (false accept) | 2 / 20 | 10.0% |
| Any impostor let in | 2 / 40 | 5.0% |
| Swap test: profile trained on someone else accepts the owner | 0 / 120 | 0.0% |

## By input style

| Style | Owner FRR | Stranger FAR | Mimic FAR | Swap accept |
|---|---:|---:|---:|---:|
| `mouse_keyboard` | 20.0% (3/15) | 0.0% (0/10) | 20.0% (2/10) | 0.0% (0/60) |
| `slow_pointer_dictation` | 0.0% (0/15) | 0.0% (0/10) | 0.0% (0/10) | 0.0% (0/60) |

## Profiles

| Profile | Enrolment recordings | Background negatives | Threshold | Margin |
|---|---:|---:|---:|---:|
| `mouse_keyboard_subject_01` | 6 | 54 | 0.90 | 0.20 |
| `mouse_keyboard_subject_02` | 6 | 54 | 0.90 | 0.20 |
| `mouse_keyboard_subject_03` | 6 | 54 | 0.90 | 0.20 |
| `mouse_keyboard_subject_04` | 6 | 54 | 0.90 | 0.20 |
| `mouse_keyboard_subject_05` | 6 | 54 | 0.90 | 0.20 |
| `slow_pointer_dictation_subject_01` | 6 | 54 | 0.90 | 0.20 |
| `slow_pointer_dictation_subject_02` | 6 | 54 | 0.90 | 0.20 |
| `slow_pointer_dictation_subject_03` | 6 | 54 | 0.90 | 0.20 |
| `slow_pointer_dictation_subject_04` | 6 | 54 | 0.90 | 0.20 |
| `slow_pointer_dictation_subject_05` | 6 | 54 | 0.90 | 0.20 |

A stranger/mimic accepted here is a **false accept**. An owner rejected is a **false reject**. The threshold and margin were not changed after these numbers were computed.

