# Job 1289270: LIBERO Goal evaluation

Evaluated `best.pt` directly with correction enabled on compute-node (A100). Job completed with exit code 0 in 01:26:14. Overall success: **583/600 (97.17%)**. Protocol: 10 tasks × 3 seeds (0, 1, 2) × 20 initial states (0–19), max 400 steps. Repeatability check was skipped.

## Download

- [best.pt](https://raw.githubusercontent.com/Kelvin1117/Robot-sim-experiment/main/experiments/job-1289270/best.pt) (3.5 MB)
- [Complete bundle](https://raw.githubusercontent.com/Kelvin1117/Robot-sim-experiment/main/downloads/job-1289270.tar.gz): weights, sanitized source snapshot, settings, results and checksums
- [Per-task CSV](results.csv)
- [Full evaluation JSON](best_direct_summary.json)

```bash
curl -fL https://raw.githubusercontent.com/Kelvin1117/Robot-sim-experiment/main/downloads/job-1289270.tar.gz -o job-1289270.tar.gz
curl -fL https://raw.githubusercontent.com/Kelvin1117/Robot-sim-experiment/main/downloads/SHA256SUMS -o SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf job-1289270.tar.gz
cd job-1289270
sha256sum -c SHA256SUMS
```

## Results

| ID | Task | Success | Rate |
|---|---|---:|---:|
| 0 | open_the_middle_drawer_of_the_cabinet | 58/60 | 96.67% |
| 1 | put_the_bowl_on_the_stove | 60/60 | 100.00% |
| 2 | put_the_wine_bottle_on_top_of_the_cabinet | 60/60 | 100.00% |
| 3 | open_the_top_drawer_and_put_the_bowl_inside | 55/60 | 91.67% |
| 4 | put_the_bowl_on_top_of_the_cabinet | 59/60 | 98.33% |
| 5 | push_the_plate_to_the_front_of_the_stove | 60/60 | 100.00% |
| 6 | put_the_cream_cheese_in_the_bowl | 60/60 | 100.00% |
| 7 | turn_on_the_stove | 60/60 | 100.00% |
| 8 | put_the_bowl_on_the_plate | 58/60 | 96.67% |
| 9 | put_the_wine_bottle_on_the_rack | 53/60 | 88.33% |

## Source and dependencies

`code/` is the sanitized source snapshot used by job 1289270, including `evaluate_best_direct.py`, `recovery_run.py`, simulator bridge and the `dygro` package. `best_direct.sbatch`, `environment.sh` and `config.json` preserve the original cluster configuration and absolute paths for provenance; they must be adapted for another machine. Invoke the evaluator with the configured Python environment and the run directory as its argument.

**This checkpoint contains correction weights, not a standalone pi0.5 model.** Evaluation also needs the pi0.5 assets, tokenizer, normalization, Stage I adapter, Stage II parent checkpoint, frozen recovery checkpoint and demonstration data referenced by `config.json` and the parent configuration. The existing repository provides parent sources, environment lockfiles and asset-download tooling; this bundle does not include all external model/data assets. Do not expect the original Slurm script to run unchanged outside the original cluster.

The source model trained on states 25–34; expanded validation is held out only for the new correction. Test states 0–19 were inspected in prior iterations, so this is an iterative evaluation, not a blind benchmark or an exact reproduction of the original DyGRO paper.

## SHA-256

Checksums are published in `SHA256SUMS` and `downloads/SHA256SUMS` for corruption detection and artifact identification. They do not independently establish publisher authenticity. The checkpoint SHA-256 is:

`98f5c2993bafbd0c502b3d48d63b4ca525964b810d3f0d6353a0f9626b809069`

PyTorch checkpoints should only be loaded from trusted sources.

## Security update

This is a sanitized public derivative, not the untouched historical source. Weights are unchanged. Use PUBLIC_SHA256SUMS for public-file integrity; historical manifests record original-run provenance. The legacy environment download is suspended; see ../../SECURITY.md.
