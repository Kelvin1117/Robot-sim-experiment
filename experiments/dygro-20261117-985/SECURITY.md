# Security and publication policy

## Current public artifacts

The 2026-11-17 best.pt is the original episode-600 correction checkpoint. Its tensor weights and file SHA256 are unchanged. Public source, configs and logs are sanitized derivatives of the historical run: account identifiers, compute-node names and GPU UUIDs are removed or replaced with generic aliases. They are not byte-identical historical source snapshots.

`PUBLIC_SHA256SUMS` verifies the sanitized public files. Historical source manifests, config hashes inside checkpoints, and training completion reports refer to the original private run, not the sanitized source. Do not bypass a hash check to make modified code appear to be the original experiment. The archived launchers are research provenance and require adaptation of paths and provenance checks before reuse; they are not a verified portable reproduction.

## Checkpoint loading

Published Python loaders default to `torch.load(..., weights_only=True)`. This restricts pickle loading; it is not a complete sandbox or a guarantee against all malformed files. Use an updated PyTorch installation and verify artifact hashes. Never load unknown checkpoints on a machine containing credentials.

Some historical optimizer/RNG checkpoints need unrestricted pickle. The only fallback is an explicit operator opt-in, `DYGRO_TRUST_CHECKPOINT_PICKLE=1`; this can execute code and must only be used for independently authenticated, reviewed legacy checkpoints inside an isolated environment. It is never set by the published launchers. The selected best.pt does not need this opt-in.

## Legacy environment archive

The old 2026-11-10 full environment release (approximately 37 GB compressed) is retained as a private draft while its contents await full inspection. Public automatic downloading is suspended. Existing downloaded copies and external caches cannot be recalled. The current code/weights release remains downloadable; full external model/environment dependencies are not bundled in it.

## Scope and residual exposure

The public main branch and both release tags were rebuilt from the sanitized snapshot with owner authorization. Their reachable history no longer includes the original commits. Existing clones must be replaced or carefully reset to avoid reintroducing old history. Forks, downloaded files, GitHub caches and objects addressable by old commit IDs may persist; this rewrite cannot guarantee their erasure. Contact GitHub Support if server-side cached sensitive material must be removed.

Common token/private-key patterns are scanned before publishing. SHA256 checks identify bytes, not whether code or data is safe. Avoid uploading .env files, credential stores, SSH keys, raw home directories, full environments or unreviewed logs. Report a suspected credential privately to the repository owner; do not post credential values in an issue.
