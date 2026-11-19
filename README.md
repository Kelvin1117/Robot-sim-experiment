# Robot simulation experiments

Latest published experiment: **2026-11-17 — LIBERO Goal, 591/600 (98.50%)**.
Selected checkpoint: **episode 600**, validation **299/300 (99.67%)**.

## LIBERO Goal per-task test results (2026-11-17)

Episode-600 checkpoint; seeds 0/1/2 × initial states 0–19, 60 scored episodes per task. These are test results, separate from the 299/300 validation score.

| Task | Task name | Successes | Success rate |
|---:|---|---:|---:|
| 0 | open_the_middle_drawer_of_the_cabinet | 60/60 | 100.00% |
| 1 | put_the_bowl_on_the_stove | 60/60 | 100.00% |
| 2 | put_the_wine_bottle_on_top_of_the_cabinet | 59/60 | 98.33% |
| 3 | open_the_top_drawer_and_put_the_bowl_inside | 58/60 | 96.67% |
| 4 | put_the_bowl_on_top_of_the_cabinet | 60/60 | 100.00% |
| 5 | push_the_plate_to_the_front_of_the_stove | 60/60 | 100.00% |
| 6 | put_the_cream_cheese_in_the_bowl | 60/60 | 100.00% |
| 7 | turn_on_the_stove | 60/60 | 100.00% |
| 8 | put_the_bowl_on_the_plate | 59/60 | 98.33% |
| 9 | put_the_wine_bottle_on_the_rack | 55/60 | 91.67% |
| **Total** | **10 tasks** | **591/600** | **98.50%** |

[Download per-task CSV](experiments/dygro-20261117-985/results.csv) · [Full evaluation JSON](experiments/dygro-20261117-985/recovered_summary.json)

- [Download best.pt](https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/dygro-20261117-985/best.pt)
- [Download sanitized code + configs + results ZIP](https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/dygro-20261117-985/dygro-20261117-985-code.zip)
- [Download SHA256SUMS](https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/dygro-20261117-985/SHA256SUMS)
- [Experiment details and dependencies](experiments/dygro-20261117-985/README.md)
- [Release page](https://github.com/Kelvin1117/Robot-sim-experiment/releases/tag/dygro-20261117-985)

The checkpoint contains the outer correction component, not a standalone pi0.5 model. The published code/configs are sanitized derivatives; see the dependency notes before evaluation.

## Previous experiment

Previously published evaluation: **job 1289270 — LIBERO Goal, 583/600 (97.17%)**.

- [Download best.pt](https://raw.githubusercontent.com/Kelvin1117/Robot-sim-experiment/main/experiments/job-1289270/best.pt)
- [Download weights + code + results bundle](https://raw.githubusercontent.com/Kelvin1117/Robot-sim-experiment/main/downloads/job-1289270.tar.gz)
- [Instructions, per-task results and dependencies](experiments/job-1289270/README.md)
- [Bundle SHA-256](downloads/SHA256SUMS)

The checkpoint is a correction component and requires external model/data assets; see the linked instructions. Existing `experiment/`, `parent_source/`, `environment/`, `reproduction/` and `reproduce.py` retain the earlier experiment and reproduction tooling.

## Security update

The current code/weights download uses sanitized metadata and safe-by-default checkpoint loaders. Historical source hashes remain provenance only; see [SECURITY.md](SECURITY.md). The legacy 37 GB full-environment release is temporarily unavailable pending complete inspection. External model dependencies are not included in the current small download.
