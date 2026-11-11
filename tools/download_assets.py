#!/usr/bin/env python3
"""Download and verify the complete experiment backup from its public release.

Usage: python tools/download_assets.py --directory /path/with/enough/space
Restore: cat experiment.tar.zst.part-* | zstd -dc | tar -xf - -C /restore/root
Then:    zstd -dc supplemental-files.tar.zst | tar -xf - -C /restore/root
The archive uses original filesystem paths without their leading slash.
Restored Python environments require compatible Linux and NVIDIA drivers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MANIFEST_URL = 'https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/retention-20261110/asset_manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--directory', type=Path, default=Path('experiment-assets'))
    parser.add_argument('--manifest', type=Path, help='Use a pinned local manifest instead of fetching the release manifest')
    parser.add_argument('--check-only', action='store_true', help='Verify downloaded parts without downloading')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.directory / 'asset_manifest.json'
    if args.manifest:
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_path.write_bytes(manifest_bytes)
    elif not args.check_only:
        with urllib.request.urlopen(MANIFEST_URL, timeout=120) as response:
            manifest_bytes = response.read()
        manifest = json.loads(manifest_bytes)
        manifest_path.write_bytes(manifest_bytes)
    else:
        manifest = json.loads(manifest_path.read_bytes())
    if not manifest.get('uploaded') or not manifest.get('zstd_integrity_verified'):
        raise SystemExit('The release manifest does not describe a complete verified archive.')
    for part in manifest['parts'] + manifest.get('supplemental_archives', []) + manifest.get('metadata_assets', []):
        name = part['file']
        if Path(name).name != name:
            raise SystemExit('Unexpected archive part name')
        path = args.directory / name
        if path.exists() and path.stat().st_size == part['size'] and sha256(path) == part['sha256']:
            print('Verified existing', name, flush=True)
            continue
        if args.check_only:
            raise SystemExit('Missing or corrupt part: ' + name)
        url = part['url']
        if not url.startswith('https://github.com/Kelvin1117/Robot-sim-experiment/releases/download/retention-20261110/'):
            raise SystemExit('Unexpected download origin')
        partial = path.with_name(path.name + '.partial')
        digest = hashlib.sha256()
        size = 0
        print('Downloading', name, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
            while block := response.read(8 * 1024 * 1024):
                digest.update(block)
                output.write(block)
                size += len(block)
        if size != part['size'] or digest.hexdigest() != part['sha256']:
            raise SystemExit('Checksum mismatch: ' + name)
        partial.replace(path)
        print('Verified', name, flush=True)
    print('All archive parts verified. Concatenate numbered parts in order before zstd/tar extraction.')


if __name__ == '__main__':
    main()
