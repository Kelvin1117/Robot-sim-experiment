"""Read-only artifact audit; does not construct a policy or infer training defaults."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors header')
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= min(path.stat().st_size - 8, 32 * 1024 * 1024):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(length))
    widths = {'F32': 4, 'BF16': 2, 'F16': 2, 'F64': 8, 'I64': 8,
              'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    intervals = []
    counts = {}
    for name, tensor in tensors.items():
        shape = tensor['shape']
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'Invalid shape: {name}')
        count = math.prod(shape)
        start, end = tensor['data_offsets']
        if start < 0 or end < start or end - start != count * widths[tensor['dtype']]:
            raise ValueError(f'Invalid offsets: {name}')
        intervals.append((start, end))
        counts[tensor['dtype']] = counts.get(tensor['dtype'], 0) + count
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Overlapping tensors or gaps in safetensors payload')
        cursor = end
    if cursor + 8 + length != path.stat().st_size:
        raise ValueError('Truncated or trailing safetensors payload')
    return tensors, counts


def check_stats(path):
    stats = json.loads(Path(path).read_text())['norm_stats']
    for key, dim in [('state', 8), ('actions', 7)]:
        for field in ['mean', 'std', 'q01', 'q99']:
            values = stats[key][field]
            if len(values) != dim or not all(math.isfinite(v) for v in values):
                raise ValueError(f'Invalid normalization: {key}.{field}')
        if any(v <= 0 for v in stats[key]['std']):
            raise ValueError('Nonpositive normalization std')
        if any(a >= b for a, b in zip(stats[key]['q01'], stats[key]['q99'])):
            raise ValueError('Invalid quantile range')
    return {'state_dim': 8, 'environment_action_dim': 7,
            'sha256': sha256(path), 'normalization_mode_from_stats_alone': 'UNDETERMINED'}


def audit(source, converted, remote_manifest, *, hash_weights=False):
    source, converted = Path(source), Path(converted)
    manifest = json.loads(Path(remote_manifest).read_text())
    missing = []
    mismatched = []
    unverified_sizes = []
    for entry in manifest['siblings']:
        relative = entry['rfilename']
        if not relative.startswith('5000/'):
            continue
        path = source / relative
        if not path.is_file():
            missing.append(relative)
        elif 'size' not in entry:
            unverified_sizes.append(relative)
        elif path.stat().st_size != entry['size']:
            mismatched.append(relative)
    weights = converted / 'model.safetensors'
    tensors, counts = read_header(weights)
    config = json.loads((converted / 'config.json').read_text())
    conversion = json.loads((converted / 'conversion-audit.json').read_text())
    projections = {k: v['shape'] for k, v in tensors.items()
                   if k.startswith(('action_in_proj.', 'action_out_proj.', 'time_mlp_'))}
    if projections.get('action_in_proj.weight', [0, 0])[-1] != 32:
        raise ValueError('Expected OpenPI padded action width 32')
    if projections.get('action_out_proj.weight', [0])[0] != 32:
        raise ValueError('Expected flow output width 32')
    source_stats = source / '5000/assets/libero_one_shot/norm_stats.json'
    converted_stats = converted / 'assets/libero_one_shot/norm_stats.json'
    normalization = check_stats(converted_stats)
    normalization['matches_remaining_source_asset'] = sha256(source_stats) == sha256(converted_stats)
    return {
        'model_id': manifest['id'], 'historical_revision': manifest['sha'],
        'source': str(source), 'converted': str(converted),
        'source_missing_files': missing, 'source_size_mismatches': mismatched,
        'source_sizes_unverified': unverified_sizes,
        'source_orbax_complete': not missing and not mismatched and not unverified_sizes,
        'converted_file_bytes': weights.stat().st_size,
        'converted_sha256': sha256(weights) if hash_weights else 'NOT_COMPUTED',
        'stored_tensor_count': len(tensors), 'stored_scalar_count': sum(counts.values()),
        'stored_scalars_by_dtype': counts, 'projection_shapes': projections,
        'converted_config': config, 'historical_conversion_audit': conversion,
        'normalization': normalization,
        'trained_horizon_provenance': 'UNVERIFIED: conversion config is not original training config',
        'jax_torch_numerical_equivalence': 'NOT_TESTED',
        'strict_load_this_audit': 'NOT_TESTED',
        'finite_tensor_values_this_audit': 'NOT_TESTED: header inspection only',
        'formal_training_ready': False,
        'blockers': ['OpenPI IB adapter and complete training/evaluation launchers not implemented',
                     'Verified primitive-rate Goal demonstration manifest not supplied',
                     'Paper does not specify complete training hyperparameters or BC/filter rules',
                     'One-Shot Stage I/II GPU training smoke not passed',
                     'latest/best validation, scheduler and recovery integration incomplete'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--converted', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash-weights', action='store_true')
    args = parser.parse_args()
    report = audit(args.source, args.converted, args.manifest, hash_weights=args.hash_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
