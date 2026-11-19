# DyGRO-VLA: 2026-11-17 LIBERO Goal experiment

- Test: 591/600 (98.50%); tasks 0–9, 60 episodes each, seeds 0/1/2 and initial states 0–19.
- Selected checkpoint: current-run episode 600; validation 299/300 (99.67%). Total run: 1800 training episodes.
- Architecture: frozen original 8-expert top-2 MoE + frozen local 4-expert top-2 correction (tasks 0/6/9 only at inference) + trainable all-task 4-expert top-2 correction. Independent routers.
- Initialization of the outer correction: 2026-11-16 no-MSAA run episode 1200, NOT the 2026-11-15 97.17% checkpoint. This run uses A100 EGL and 4x MSAA.

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

[Per-task CSV](results.csv) · [Full evaluation JSON](recovered_summary.json)

## Downloads

- [Download best.pt](https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/dygro-20261117-985/best.pt) — selected episode 600 outer correction weights.
- [Download code + configs + results ZIP](https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/dygro-20261117-985/dygro-20261117-985-code.zip).
- [Download SHA256SUMS](https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/dygro-20261117-985/SHA256SUMS).

`sha256sum -c SHA256SUMS` verifies both downloads when placed in the same directory.

## Checkpoint scope and dependencies

`best.pt` contains the OUTERMOST correction state_dict and checkpoint metadata. It is not the complete pi0.5 model. Evaluation also needs the original pi0.5 assets, Stage I validated weights, original Stage II best actor, and the 2026-11-10 retention best correction. Historical `config.json` records their paths and SHA256 hashes. The legacy full-environment release is temporarily unpublished pending a complete security audit; external parent/base assets are not included in this small download. Published paths use a generic researcher alias; do not expect it to run on a new machine without setting up those dependencies and adapting paths. `source_manifest.json` retains historical hashes for provenance, not verification of the sanitized source; use PUBLIC_SHA256SUMS for the public files. The historical inventory also references initial_best.pt, which is not included in the code-only archive.

The original evaluation entry point is `code/recovery_run.py --stage evaluate --run RUN_DIR`, using `environment.sh` and `job.sbatch`; reconstruct the complete historical run dependencies before invoking it. No training or evaluation was started while preparing this publication.

## Interpretation

The test states were inspected in earlier iterations: this is an iterative result, not blind test performance. Exact repeatability probes did not pass. See recovered_summary.json and evaluation.md.

## Selected best.pt SHA256

`f81d8623d2173bfe0d37a02e94c47557b58c8b10b15bf8b32eb20bd44b266782`

## Security update

This is a sanitized public derivative, not the untouched historical source. Weights are unchanged. Use PUBLIC_SHA256SUMS for public-file integrity; historical manifests record original-run provenance. The legacy environment download is suspended; see ../../SECURITY.md.
